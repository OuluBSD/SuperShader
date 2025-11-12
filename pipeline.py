#!/usr/bin/env python3
"""
Data Processing Pipeline for SuperShader Project

This script creates a comprehensive pipeline for processing shaders in batches,
organized by tags and optimized for large-scale analysis. It builds upon the
tag analysis and feature cataloging systems to process shaders efficiently.
"""

import os
import json
import glob
import pickle
import hashlib
from pathlib import Path
from collections import defaultdict
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import Dict, List, Set, Tuple
import sqlite3


class ShaderProcessingPipeline:
    def __init__(self, json_dir='json', output_dir='processed', db_path='shaders.db'):
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        self.db_path = Path(db_path)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize database schema (but connections will be created per thread)
        self._init_database_schema()
        
        # Cache for processed shader info
        self.processed_cache_file = self.output_dir / 'processed_cache.pkl'
        self.processed_cache = self._load_cache()

    def _init_database_schema(self):
        """Initialize SQLite database schema."""
        # Create a temporary connection just to initialize the schema
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for shader metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shaders (
                id TEXT PRIMARY KEY,
                name TEXT,
                username TEXT,
                description TEXT,
                tags TEXT,
                file_path TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create table for shader features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shader_features (
                shader_id TEXT,
                feature_type TEXT,
                feature_name TEXT,
                count INTEGER DEFAULT 1,
                FOREIGN KEY (shader_id) REFERENCES shaders(id)
            )
        ''')
        
        # Create table for tags
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                name TEXT PRIMARY KEY,
                usage_count INTEGER DEFAULT 1
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_shader_features_type ON shader_features(feature_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_shader_features_name ON shader_features(feature_name)')
        
        conn.commit()
        conn.close()
    
    def _create_thread_db_connection(self):
        """Create a new database connection for the current thread."""
        return sqlite3.connect(self.db_path)

    def _load_cache(self):
        """Load processed cache if it exists."""
        if self.processed_cache_file.exists():
            try:
                with open(self.processed_cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Save processed cache."""
        with open(self.processed_cache_file, 'wb') as f:
            pickle.dump(self.processed_cache, f)

    def _extract_shader_info(self, json_file_path):
        """Extract basic information from a shader JSON file."""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or 'info' not in data:
            return None
            
        info = data['info']
        shader_id = info.get('id', json_file_path.stem)
        
        # Extract GLSL code for further analysis
        glsl_code = ""
        if 'renderpass' in data:
            for pass_data in data['renderpass']:
                if 'code' in pass_data:
                    glsl_code += pass_data['code'] + "\n"
        
        return {
            'id': shader_id,
            'name': info.get('name', ''),
            'username': info.get('username', ''),
            'description': info.get('description', ''),
            'tags': info.get('tags', []),
            'file_path': str(json_file_path),
            'glsl_code': glsl_code
        }

    def _extract_features(self, shader_info):
        """Extract features from shader info (similar to catalog_features.py)."""
        import re
        
        features = []
        glsl_code = shader_info['glsl_code']
        
        # Extract functions
        function_pattern = r'\b(\w+)\s+(\w+)\s*\([^)]*\)\s*\{'
        functions = re.findall(function_pattern, glsl_code)
        
        builtin_functions = {
            'main', 'mainImage', 'mainFragment', 'mainVertex', 
            'sin', 'cos', 'tan', 'pow', 'sqrt', 'abs', 'sign', 
            'floor', 'ceil', 'fract', 'mod', 'min', 'max', 'clamp', 
            'mix', 'step', 'smoothstep', 'length', 'distance', 'dot', 
            'cross', 'normalize', 'faceforward', 'reflect', 'refract', 
            'texture', 'texture2D', 'textureCube', 'texelFetch', 
            'dFdx', 'dFdy', 'fwidth', 'noise', 'cellnoise', 'snoise'
        }
        
        for return_type, func_name in functions:
            if func_name.lower() not in builtin_functions:
                features.append(('function', func_name, 1))
        
        # Extract types
        for glsl_type in ['vec2', 'vec3', 'vec4', 'mat2', 'mat3', 'mat4', 'sampler2D', 'samplerCube']:
            if glsl_type in glsl_code:
                features.append(('type', glsl_type, 1))
        
        # Extract renderpass types
        # This would be detected from the actual JSON, not GLSL code
        # For now, we'll add a placeholder method
        
        return features

    def _store_in_db(self, shader_info, features):
        """Store shader info and features in the database."""
        conn = self._create_thread_db_connection()
        cursor = conn.cursor()
        
        # Insert shader metadata
        cursor.execute('''
            INSERT OR REPLACE INTO shaders 
            (id, name, username, description, tags, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            shader_info['id'],
            shader_info['name'],
            shader_info['username'],
            shader_info['description'],
            ','.join(shader_info['tags']),
            shader_info['file_path']
        ))
        
        # Insert features
        for feature_type, feature_name, count in features:
            cursor.execute('''
                INSERT OR REPLACE INTO shader_features 
                (shader_id, feature_type, feature_name, count)
                VALUES (?, ?, ?, ?)
            ''', (shader_info['id'], feature_type, feature_name, count))
        
        # Update tag usage counts
        for tag in shader_info['tags']:
            cursor.execute('''
                INSERT INTO tags (name, usage_count) 
                VALUES (?, 1)
                ON CONFLICT(name) DO UPDATE SET usage_count = usage_count + 1
            ''', (tag,))
        
        conn.commit()
        conn.close()

    def process_shader_batch(self, file_paths, batch_size=100):
        """Process a batch of shader files."""
        results = []
        
        for i, file_path in enumerate(file_paths):
            try:
                shader_info = self._extract_shader_info(Path(file_path))
                if shader_info:
                    features = self._extract_features(shader_info)
                    self._store_in_db(shader_info, features)
                    results.append(shader_info['id'])
                    
                    # Update cache
                    self.processed_cache[shader_info['id']] = {
                        'file_path': file_path,
                        'processed_at': time.time()
                    }
                    
                    # Save cache periodically
                    if i % batch_size == 0:
                        self._save_cache()
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
                
        return results

    def process_by_tag(self, tag, max_shaders=None):
        """Process shaders with a specific tag."""
        # First, get all shader IDs with this tag from the database
        conn = self._create_thread_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, file_path FROM shaders 
            WHERE tags LIKE ?
        ''', (f'%{tag}%',))
        
        matching_shaders = cursor.fetchall()
        conn.close()
        
        if max_shaders:
            matching_shaders = matching_shaders[:max_shaders]
        
        print(f"Processing {len(matching_shaders)} shaders with tag '{tag}'")
        
        results = []
        for shader_id, file_path in matching_shaders:
            try:
                shader_info = self._extract_shader_info(Path(file_path))
                if shader_info:
                    features = self._extract_features(shader_info)
                    self._store_in_db(shader_info, features)
                    results.append(shader_info['id'])
                    
                    # Update cache
                    self.processed_cache[shader_id] = {
                        'file_path': file_path,
                        'processed_at': time.time()
                    }
            except Exception as e:
                print(f"Error processing {file_path} for tag '{tag}': {e}")
                continue
        
        self._save_cache()
        return results

    def process_all(self, max_workers=4, batch_size=100):
        """Process all JSON files in the directory."""
        json_files = list(self.json_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files to process")
        
        # Filter out already processed files
        unprocessed_files = [
            f for f in json_files 
            if f.stem not in self.processed_cache or 
               self._needs_reprocessing(f, self.processed_cache[f.stem].get('file_path'))
        ]
        
        print(f"{len(unprocessed_files)} files need processing")
        
        # Process in batches using multiple threads
        total_processed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Break into batches
            batches = [unprocessed_files[i:i + batch_size] 
                      for i in range(0, len(unprocessed_files), batch_size)]
            
            futures = {executor.submit(self.process_shader_batch, [str(f) for f in batch]): batch 
                      for batch in batches}
            
            for future in as_completed(futures):
                batch_result = future.result()
                total_processed += len(batch_result)
                print(f"Completed batch: {len(batch_result)} shaders processed")
        
        print(f"Total processed: {total_processed} shaders")
        return total_processed

    def _needs_reprocessing(self, file_path, cached_path):
        """Check if a file needs reprocessing."""
        if cached_path and os.path.exists(cached_path):
            # Compare file modification times
            return os.path.getmtime(file_path) > os.path.getmtime(cached_path)
        return True

    def get_shader_by_feature(self, feature_type, feature_name):
        """Get shaders that contain a specific feature."""
        conn = self._create_thread_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.* FROM shaders s
            JOIN shader_features sf ON s.id = sf.shader_id
            WHERE sf.feature_type = ? AND sf.feature_name = ?
        ''', (feature_type, feature_name))
        
        results = cursor.fetchall()
        conn.close()
        return results

    def get_top_features(self, feature_type=None, limit=20):
        """Get top features of a certain type."""
        conn = self._create_thread_db_connection()
        cursor = conn.cursor()
        
        if feature_type:
            cursor.execute('''
                SELECT feature_name, SUM(count) as total_count
                FROM shader_features
                WHERE feature_type = ?
                GROUP BY feature_name
                ORDER BY total_count DESC
                LIMIT ?
            ''', (feature_type, limit))
        else:
            cursor.execute('''
                SELECT feature_type, feature_name, SUM(count) as total_count
                FROM shader_features
                GROUP BY feature_type, feature_name
                ORDER BY total_count DESC
                LIMIT ?
            ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        return results

    def get_tag_stats(self):
        """Get statistics about tags."""
        conn = self._create_thread_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM tags ORDER BY usage_count DESC')
        results = cursor.fetchall()
        conn.close()
        return results

    def generate_report(self):
        """Generate a comprehensive processing report."""
        report_path = self.output_dir / 'processing_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Shader Processing Pipeline Report\n")
            f.write("=" * 50 + "\n")
            
            conn = self._create_thread_db_connection()
            cursor = conn.cursor()
            
            # Shader count
            cursor.execute('SELECT COUNT(*) FROM shaders')
            shader_count = cursor.fetchone()[0]
            f.write(f"Total Shaders Processed: {shader_count}\n")
            
            # Tag stats
            tag_stats = self.get_tag_stats()
            f.write(f"Total Unique Tags: {len(tag_stats)}\n\n")
            
            f.write("Top 20 Tags:\n")
            f.write("-" * 30 + "\n")
            for tag, count in tag_stats[:20]:
                f.write(f"{tag}: {count}\n")
            
            # Feature stats
            f.write(f"\nTop 20 Shader Features:\n")
            f.write("-" * 30 + "\n")
            top_features = self.get_top_features(limit=20)
            for row in top_features:
                if len(row) == 2:  # feature_name, total_count
                    f.write(f"{row[0]}: {row[1]}\n")
                else:  # feature_type, feature_name, total_count
                    f.write(f"{row[0]}:{row[1]}: {row[2]}\n")
            
            conn.close()
        
        print(f"Processing report saved to {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description="Shader Processing Pipeline")
    parser.add_argument("--json-dir", type=str, default="json", 
                       help="Directory containing JSON shader files")
    parser.add_argument("--output-dir", type=str, default="processed", 
                       help="Directory for output files")
    parser.add_argument("--db-path", type=str, default="shaders.db", 
                       help="Path for the SQLite database")
    parser.add_argument("--process-all", action="store_true", 
                       help="Process all shaders in the directory")
    parser.add_argument("--process-by-tag", type=str, 
                       help="Process shaders by a specific tag")
    parser.add_argument("--max-shaders", type=int, 
                       help="Maximum number of shaders to process (for tag processing)")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of worker threads")
    parser.add_argument("--batch-size", type=int, default=100, 
                       help="Batch size for processing")
    parser.add_argument("--generate-report", action="store_true", 
                       help="Generate a processing report")
    parser.add_argument("--top-features", type=int, default=20, 
                       help="Number of top features to show")
    
    args = parser.parse_args()
    
    pipeline = ShaderProcessingPipeline(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        db_path=args.db_path
    )
    
    if args.process_all:
        print("Starting batch processing of all shaders...")
        start_time = time.time()
        processed_count = pipeline.process_all(
            max_workers=args.workers,
            batch_size=args.batch_size
        )
        elapsed = time.time() - start_time
        print(f"Processed {processed_count} shaders in {elapsed:.2f} seconds")
    
    if args.process_by_tag:
        print(f"Processing shaders with tag '{args.process_by_tag}'...")
        start_time = time.time()
        processed_count = pipeline.process_by_tag(
            args.process_by_tag,
            max_shaders=args.max_shaders
        )
        elapsed = time.time() - start_time
        print(f"Processed {len(processed_count)} shaders with tag '{args.process_by_tag}' in {elapsed:.2f} seconds")
    
    if args.generate_report:
        report_path = pipeline.generate_report()
        print(f"Report generated: {report_path}")
    
    if args.top_features:
        print(f"\nTop {args.top_features} features:")
        top_features = pipeline.get_top_features(limit=args.top_features)
        for i, row in enumerate(top_features, 1):
            if len(row) == 2:
                print(f"{i:2d}. {row[0]}: {row[1]}")
            else:
                print(f"{i:2d}. {row[0]}:{row[1]}: {row[2]}")


if __name__ == "__main__":
    main()