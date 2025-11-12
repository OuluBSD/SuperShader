#!/usr/bin/env python3
"""
Shader Feature Cataloging System for SuperShader Project

This script creates and manages a catalog of shader features extracted from JSON files.
It identifies and catalogs various shader components for systematic analysis and module creation.
"""

import json
import os
import glob
from collections import defaultdict, Counter
from pathlib import Path
import pickle
import hashlib
import argparse


class ShaderFeatureCataloger:
    def __init__(self, json_dir='json', catalog_dir='catalogs'):
        self.json_dir = json_dir
        self.catalog_dir = catalog_dir
        os.makedirs(catalog_dir, exist_ok=True)
        
        # Define cache file for storing feature catalog
        self.cache_file = os.path.join(catalog_dir, 'feature_catalog_cache.dat')
        
        # Initialize data structures
        self.features = defaultdict(Counter)  # feature_type -> {feature_name: count}
        self.shader_features = {}  # shader_id -> list of features
        self.feature_shader_map = defaultdict(set)  # feature -> set of shader_ids
        self.total_shaders = 0

    def _extract_uniforms(self, shader_json):
        """Extract uniform variables from shader JSON."""
        uniforms = set()
        
        # Check for uniforms in renderpass
        if 'renderpass' in shader_json:
            for pass_data in shader_json['renderpass']:
                if 'inputs' in pass_data:
                    for inp in pass_data['inputs']:
                        # Add input types as features
                        inp_type = inp.get('type', 'unknown')
                        uniforms.add(f"input_type:{inp_type}")
                        
                        # Add sampler features
                        sampler = inp.get('sampler', {})
                        for key, value in sampler.items():
                            uniforms.add(f"sampler_{key}:{str(value).lower()}")
        
        return list(uniforms)

    def _extract_functions(self, glsl_code):
        """Extract function names from GLSL code."""
        import re
        # Pattern to match function declarations in GLSL
        function_pattern = r'\b(\w+)\s+(\w+)\s*\([^)]*\)\s*\{'
        functions = re.findall(function_pattern, glsl_code)
        
        # Extract and return function names (excluding common GLSL built-ins)
        builtin_functions = {
            'main', 'mainImage', 'mainFragment', 'mainVertex', 
            'sin', 'cos', 'tan', 'pow', 'sqrt', 'abs', 'sign', 
            'floor', 'ceil', 'fract', 'mod', 'min', 'max', 'clamp', 
            'mix', 'step', 'smoothstep', 'length', 'distance', 'dot', 
            'cross', 'normalize', 'faceforward', 'reflect', 'refract', 
            'texture', 'texture2D', 'textureCube', 'texelFetch', 
            'dFdx', 'dFdy', 'fwidth', 'noise', 'cellnoise', 'snoise'
        }
        
        extracted = []
        for return_type, func_name in functions:
            if func_name.lower() not in builtin_functions:
                extracted.append(f"function:{func_name}")
        
        return extracted

    def _extract_language_features(self, glsl_code):
        """Extract GLSL-specific language features."""
        import re
        
        features = set()
        
        # Check for specific GLSL constructs
        if '#version' in glsl_code:
            version_match = re.search(r'#version\s+(\d+)', glsl_code)
            if version_match:
                features.add(f"glsl_version:{version_match.group(1)}")
        
        # Check for precision qualifiers
        if 'precision' in glsl_code:
            precision_matches = re.findall(r'precision\s+(lowp|mediump|highp)', glsl_code)
            for prec in precision_matches:
                features.add(f"precision:{prec}")
        
        # Check for specific GLSL types
        glsl_types = ['vec2', 'vec3', 'vec4', 'mat2', 'mat3', 'mat4', 'sampler2D', 'samplerCube']
        for glsl_type in glsl_types:
            if glsl_type in glsl_code:
                features.add(f"type:{glsl_type}")
        
        # Check for shader-specific constructs
        if 'gl_Position' in glsl_code:
            features.add("vertex_shader_feature:gl_Position")
        if 'gl_FragColor' in glsl_code or 'fragColor' in glsl_code:
            features.add("fragment_shader_feature:gl_FragColor")
        if 'gl_FragDepth' in glsl_code:
            features.add("fragment_shader_feature:gl_FragDepth")
        
        return list(features)

    def _extract_from_json(self, shader_json, shader_id):
        """Extract features from a shader JSON object."""
        all_features = set()
        
        # Extract uniforms
        uniforms = self._extract_uniforms(shader_json)
        all_features.update(uniforms)
        
        # Extract from renderpass code
        if 'renderpass' in shader_json:
            for i, pass_data in enumerate(shader_json['renderpass']):
                if 'code' in pass_data:
                    code = pass_data['code']
                    
                    # Extract functions
                    functions = self._extract_functions(code)
                    all_features.update(functions)
                    
                    # Extract language features
                    lang_features = self._extract_language_features(code)
                    all_features.update(lang_features)
        
        # Extract from info section
        info = shader_json.get('info', {})
        if 'tags' in info:
            for tag in info['tags']:
                all_features.add(f"tag:{tag.lower()}")
        
        # Extract renderpass types
        if 'renderpass' in shader_json:
            for i, pass_info in enumerate(shader_json['renderpass']):
                pass_type = pass_info.get('type', 'fragment')
                all_features.add(f"renderpass_type:{pass_type}")
                
                # Check for multiple render passes
                if i > 0:
                    all_features.add("feature:multi_pass_shader")
        
        return list(all_features)

    def catalog_features(self, force_rebuild=False):
        """Catalog features from all JSON shaders in the directory."""
        # Check if cache exists and use it unless force_rebuild is True
        if not force_rebuild and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.features = cached_data.get('features', defaultdict(Counter))
                    self.shader_features = cached_data.get('shader_features', {})
                    self.feature_shader_map = cached_data.get('feature_shader_map', defaultdict(set))
                    self.total_shaders = cached_data.get('total_shaders', 0)
                    return self.features, self.shader_features, self.feature_shader_map
            except Exception as e:
                print(f"Cache loading failed: {e}, rebuilding...")
        
        print("Building feature catalog...")
        
        # Get all JSON files
        json_pattern = os.path.join(self.json_dir, "*.json")
        json_files = glob.glob(json_pattern)
        
        print(f"Processing {len(json_files)} JSON files...")
        
        for i, filepath in enumerate(json_files):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(json_files)} files...")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    shader_json = json.load(f)
                
                shader_id = os.path.basename(filepath).replace('.json', '')
                
                # Extract features from this shader
                shader_features = self._extract_from_json(shader_json, shader_id)
                
                # Update global feature counts
                for feature in shader_features:
                    feature_type, feature_name = feature.split(':', 1) if ':' in feature else ('unknown', feature)
                    self.features[feature_type][feature_name] += 1
                
                # Map shader to its features
                self.shader_features[shader_id] = shader_features
                
                # Update feature to shader mapping
                for feature in shader_features:
                    self.feature_shader_map[feature].add(shader_id)
                
                self.total_shaders += 1
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
        
        # Save to cache
        cache_data = {
            'features': self.features,
            'shader_features': self.shader_features,
            'feature_shader_map': self.feature_shader_map,
            'total_shaders': self.total_shaders
        }
        
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")
        
        return self.features, self.shader_features, self.feature_shader_map

    def get_top_features(self, feature_type=None, limit=20):
        """Get top features of a specific type."""
        if feature_type:
            if feature_type in self.features:
                return self.features[feature_type].most_common(limit)
            else:
                return []
        else:
            # Return top features across all types
            all_features = []
            for ft, counter in self.features.items():
                for feature, count in counter.most_common(limit):
                    all_features.append((f"{ft}:{feature}", count))
            all_features.sort(key=lambda x: x[1], reverse=True)
            return all_features[:limit]

    def get_shaders_with_feature(self, feature):
        """Get all shaders that have a specific feature."""
        return list(self.feature_shader_map.get(feature, set()))

    def save_catalog_report(self):
        """Save a comprehensive report of the feature catalog."""
        report_path = os.path.join(self.catalog_dir, 'feature_catalog_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Shader Feature Catalog Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Shaders Processed: {self.total_shaders}\n")
            f.write(f"Total Feature Types: {len(self.features)}\n")
            f.write(f"Total Unique Features: {sum(len(counter) for counter in self.features.values())}\n\n")
            
            for feature_type, counter in self.features.items():
                f.write(f"\n{feature_type.upper()} FEATURES:\n")
                f.write("-" * 30 + "\n")
                for feature, count in counter.most_common(20):  # Top 20 of each type
                    f.write(f"{feature}: {count}\n")
        
        print(f"Catalog report saved to {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description="Catalog shader features from JSON files")
    parser.add_argument("--json-dir", type=str, default="json", help="Directory containing JSON shader files")
    parser.add_argument("--catalog-dir", type=str, default="catalogs", help="Directory to store catalog files")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of feature catalog")
    parser.add_argument("--feature-type", type=str, help="Specific feature type to report on")
    parser.add_argument("--limit", type=int, default=20, help="Limit number of features to show")
    
    args = parser.parse_args()
    
    cataloger = ShaderFeatureCataloger(json_dir=args.json_dir, catalog_dir=args.catalog_dir)
    
    print("Starting shader feature cataloging...")
    features, shader_features, feature_shader_map = cataloger.catalog_features(force_rebuild=args.force_rebuild)
    
    print(f"\nCataloging complete!")
    print(f"Total shaders processed: {cataloger.total_shaders}")
    print(f"Feature types found: {len(features)}")
    
    # Show top features
    top_features = cataloger.get_top_features(feature_type=args.feature_type, limit=args.limit)
    print(f"\nTop {args.limit} features{' of type ' + args.feature_type if args.feature_type else ''}:")
    for feature, count in top_features:
        print(f"  {feature}: {count}")
    
    # Save comprehensive report
    report_path = cataloger.save_catalog_report()
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()