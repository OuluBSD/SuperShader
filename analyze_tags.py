#!/usr/bin/env python3
"""
Script to analyze JSON shader files and extract all available tags.
Creates a comprehensive list of tags to organize shaders by genre,
groups shaders by tags, and documents the tag distribution.
"""

import os
import json
import glob
from collections import Counter, defaultdict
from pathlib import Path


def analyze_shader_tags(json_dir='json'):
    """
    Analyze JSON files to extract all available tags.
    
    Args:
        json_dir (str): Directory containing JSON shader files
    
    Returns:
        tuple: (tag_count, tag_to_shaders, tag_distribution)
            - tag_count: Counter of how many times each tag appears
            - tag_to_shaders: dict mapping tags to lists of shader IDs
            - tag_distribution: dict with overall tag statistics
    """
    print("Analyzing shader tags...")
    
    # Initialize data structures
    tag_count = Counter()
    tag_to_shaders = defaultdict(list)
    total_shaders = 0
    
    # Get all JSON files
    json_pattern = os.path.join(json_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    print(f"Found {len(json_files)} JSON files to analyze")
    
    # Process each JSON file
    for i, filepath in enumerate(json_files):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(json_files)} files...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if data is a dict (object) and has 'info' key
            if isinstance(data, dict) and 'info' in data:
                info = data.get('info', {})
                shader_id = info.get('id', os.path.basename(filepath).replace('.json', ''))
                
                # Extract tags from the info section
                tags = info.get('tags', [])
                
                # If there are no tags, try to extract from filename or other sources
                if not tags:
                    # Extract from filename if possible
                    filename = os.path.basename(filepath)
                    name = info.get('name', filename)
                    # You could implement more sophisticated tag extraction logic here
                    # For now, we'll just note that this shader has no tags
                    pass
                
                # Update counters and mappings
                for tag in tags:
                    # Normalize the tag (lowercase, remove special characters if needed)
                    normalized_tag = tag.lower().strip()
                    if normalized_tag:  # Only add non-empty tags
                        tag_count[normalized_tag] += 1
                        tag_to_shaders[normalized_tag].append({
                            'id': shader_id,
                            'name': info.get('name', ''),
                            'username': info.get('username', ''),
                            'description': info.get('description', '')
                        })
                
                total_shaders += 1
            else:
                # Skip if data is not in expected format
                continue
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filepath}: {e}")
            continue
    
    print(f"Analysis complete. Processed {total_shaders} valid shader files.")
    
    # Calculate tag distribution
    tag_distribution = {
        'total_shaders': total_shaders,
        'unique_tags': len(tag_count),
        'tags_with_counts': dict(tag_count.most_common()),
        'tag_to_shader_count': {tag: len(shaders) for tag, shaders in tag_to_shaders.items()}
    }
    
    return tag_count, tag_to_shaders, tag_distribution


def categorize_shaders_by_genre(tag_to_shaders):
    """
    Categorize shaders by genre based on common tags.
    
    Args:
        tag_to_shaders: dict mapping tags to lists of shader IDs
    
    Returns:
        dict: Mapping of genre categories to shader lists
    """
    genre_mapping = {
        'geometry': ['geometry', 'polygon', 'triangle', 'mesh', 'model', 'shape', 'primitive', '3d', '2d'],
        'lighting': ['lighting', 'light', 'shadow', 'specular', 'diffuse', 'ambient', 'phong', 'pbr', 'illumination'],
        'effects': ['effect', 'effects', 'post', 'filter', 'blur', 'glow', 'fx', 'distortion', 'vignette', 'glitch'],
        'animation': ['animation', 'animate', 'moving', 'motion', 'time', 'dynamic', 'sequence'],
        'procedural': ['procedural', 'noise', 'fractal', 'generative', 'algorithmic', 'pattern', 'algorithm'],
        'raymarching': ['raymarching', 'ray', 'raymarch', 'sdf', 'distance', 'field', 'raytracing'],
        'particles': ['particle', 'particles', 'physics', 'simulation', 'fluid', 'dynamics'],
        'texturing': ['texture', 'texturing', 'uv', 'mapping', 'sampling', 'sampling'],
        'audio': ['audio', 'sound', 'music', 'frequency', 'spectrum', 'visualization', 'visualizer'],
        'ui': ['ui', 'interface', '2d', 'gui', 'graphics', 'hud', 'overlay'],
        'experimental': ['experimental', 'test', 'testing', 'debug', 'prototype', 'concept']
    }
    
    # Map shaders to genres
    genre_to_shaders = defaultdict(list)
    unassigned_shaders = []
    
    # Keep track of which shaders have been assigned to avoid duplication
    assigned_shader_ids = set()
    
    for genre, genre_tags in genre_mapping.items():
        for tag, shaders in tag_to_shaders.items():
            # Check if this tag belongs to this genre
            if any(genre_tag in tag for genre_tag in genre_tags):
                for shader in shaders:
                    if shader['id'] not in assigned_shader_ids:
                        genre_to_shaders[genre].append(shader)
                        assigned_shader_ids.add(shader['id'])
    
    # Find shaders that weren't assigned to any genre
    for tag, shaders in tag_to_shaders.items():
        for shader in shaders:
            if shader['id'] not in assigned_shader_ids:
                # Check if it's already in unassigned list to avoid duplicates
                if not any(s['id'] == shader['id'] for s in unassigned_shaders):
                    unassigned_shaders.append(shader)
                assigned_shader_ids.add(shader['id'])
    
    return dict(genre_to_shaders), unassigned_shaders


def save_tag_analysis(tag_distribution, tag_to_shaders, genre_to_shaders, unassigned_shaders):
    """
    Save the tag analysis to files.
    """
    # Create analysis directory
    os.makedirs('analysis', exist_ok=True)
    
    # Save tag frequencies
    with open('analysis/tag_frequencies.txt', 'w', encoding='utf-8') as f:
        f.write("Tag Frequency Analysis\n")
        f.write("=" * 50 + "\n")
        for tag, count in tag_distribution['tags_with_counts'].items():
            f.write(f"{tag}: {count}\n")
    
    # Save genre categorization
    with open('analysis/genre_categorization.txt', 'w', encoding='utf-8') as f:
        f.write("Genre Categorization\n")
        f.write("=" * 50 + "\n")
        for genre, shaders in genre_to_shaders.items():
            f.write(f"\n{genre.upper()} ({len(shaders)} shaders):\n")
            f.write("-" * 30 + "\n")
            for shader in shaders[:10]:  # Limit to first 10 for readability
                f.write(f"  - {shader['id']}: {shader['name']} by {shader['username']}\n")
            if len(shaders) > 10:
                f.write(f"  ... and {len(shaders) - 10} more\n")
        
        f.write(f"\nUNASSIGNED ({len(unassigned_shaders)} shaders):\n")
        f.write("-" * 30 + "\n")
        for shader in unassigned_shaders[:10]:  # Limit to first 10 for readability
            f.write(f"  - {shader['id']}: {shader['name']} by {shader['username']}\n")
        if len(unassigned_shaders) > 10:
            f.write(f"  ... and {len(unassigned_shaders) - 10} more\n")
    
    # Save detailed tag to shaders mapping
    with open('analysis/tag_to_shaders.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed Tag to Shaders Mapping\n")
        f.write("=" * 50 + "\n")
        for tag, shaders in list(tag_to_shaders.items())[:20]:  # Limit for file size
            f.write(f"\nTAG: {tag} ({len(shaders)} shaders)\n")
            f.write("-" * 30 + "\n")
            for shader in shaders[:5]:  # Limit to first 5 for readability
                f.write(f"  ID: {shader['id']}\n")
                f.write(f"    Name: {shader['name']}\n")
                f.write(f"    Author: {shader['username']}\n")
                f.write(f"    Description: {shader['description'][:100] + '...' if len(shader['description']) > 100 else shader['description']}\n")
    
    print("Analysis results saved to analysis/ directory")


def print_summary(tag_distribution, genre_to_shaders, unassigned_shaders):
    """
    Print a summary of the analysis.
    """
    print("\n" + "="*60)
    print("TAG ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total shaders analyzed: {tag_distribution['total_shaders']}")
    print(f"Unique tags found: {tag_distribution['unique_tags']}")
    print(f"Most common tags:")
    for tag, count in list(tag_distribution['tags_with_counts'].items())[:10]:
        print(f"  {tag}: {count}")
    
    print(f"\nGenre distribution:")
    for genre, shaders in genre_to_shaders.items():
        print(f"  {genre}: {len(shaders)} shaders")
    
    print(f"\nUnassigned shaders: {len(unassigned_shaders)}")
    print("="*60)


def main():
    # Analyze tags in JSON files
    tag_count, tag_to_shaders, tag_distribution = analyze_shader_tags()
    
    # Categorize shaders by genre
    genre_to_shaders, unassigned_shaders = categorize_shaders_by_genre(tag_to_shaders)
    
    # Print summary
    print_summary(tag_distribution, genre_to_shaders, unassigned_shaders)
    
    # Save results to files
    save_tag_analysis(tag_distribution, tag_to_shaders, genre_to_shaders, unassigned_shaders)
    
    print("\nTag extraction and categorization completed!")


if __name__ == "__main__":
    main()