#!/usr/bin/env python3
"""
Process particle/physics simulation shaders from JSON files to identify common patterns
and extract reusable modules.
"""

import json
import os
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path


def find_particle_physics_shaders(json_dir='json'):
    """
    Find all JSON files that contain particle/physics simulation related tags.

    Args:
        json_dir (str): Directory containing JSON shader files

    Returns:
        list: List of tuples (filepath, shader_info) for particle/physics shaders
    """
    print("Finding particle/physics simulation related shaders...")
    
    keywords = [
        'particle', 'particles', 'physics', 'simulation', 'fluid', 'dynamics',
        'velocity', 'acceleration', 'force', 'mass', 'gravity', 'collision',
        'verlet', 'euler', 'integrat', 'motion', 'rigid', 'body', 'spring',
        'cloth', 'hair', 'rope', 'pendulum', 'fluid', 'wave', 'wave equation',
        'navier', 'stokes', 'buoyancy', 'drag', 'turbulence', 'smoke', 'fire',
        'constraints', 'verlet', 'integration', 'bounce', 'friction'
    ]
    
    physics_shaders = []
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"Scanning {len(json_files)} JSON files for particle/physics tags...")
    
    for i, filepath in enumerate(json_files):
        if i % 1000 == 0:
            print(f"Scanned {i}/{len(json_files)} files...")
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict) and 'info' in data:
                info = data['info']
                tags = [tag.lower() for tag in info.get('tags', [])]
                name = info.get('name', '').lower()
                description = info.get('description', '').lower()
                
                # Check if this shader is particle/physics simulation related 
                is_physics_related = False
                
                # Check tags
                for tag in tags:
                    if any(keyword in tag for keyword in keywords):
                        is_physics_related = True
                        break
                
                # Check name
                if not is_physics_related:
                    for keyword in keywords:
                        if keyword in name:
                            is_physics_related = True
                            break
                
                # Check description
                if not is_physics_related:
                    for keyword in keywords:
                        if keyword in description:
                            is_physics_related = True
                            break
                
                if is_physics_related:
                    shader_info = {
                        'id': info.get('id', os.path.basename(filepath).replace('.json', '')),
                        'name': info.get('name', ''),
                        'tags': tags,
                        'username': info.get('username', ''),
                        'description': info.get('description', ''),
                        'filepath': filepath
                    }
                    physics_shaders.append((filepath, shader_info))
                    
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not process {filepath}: {e}")
            continue

    print(f"Found {len(physics_shaders)} particle/physics simulation related shaders")
    return physics_shaders


def extract_shader_code(filepath):
    """
    Extract GLSL code from a JSON shader file.

    Args:
        filepath (str): Path to the JSON shader file

    Returns:
        str: GLSL code extracted from the file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        shader_data = json.load(f)

    glsl_code = []

    # Extract GLSL code based on render passes
    if 'renderpass' in shader_data:
        for i, pass_data in enumerate(shader_data['renderpass']):
            if 'code' in pass_data:
                code = pass_data['code']
                pass_type = pass_data.get('type', 'fragment')
                name = pass_data.get('name', f'Pass {i}')
                
                glsl_code.append(f"// {name} ({pass_type})")
                glsl_code.append(code)
                glsl_code.append("")  # Empty line separator
    else:
        # Try to find shader code in other possible fields
        possible_fields = ['fragment_shader', 'vertex_shader', 'shader', 'code', 'main']
        for field in possible_fields:
            if field in shader_data and isinstance(shader_data[field], str):
                code = shader_data[field]
                glsl_code.append(f"// From field: {field}")
                glsl_code.append(code)
                glsl_code.append("")
    
    return "\n".join(glsl_code)


def identify_physics_patterns(shader_code):
    """
    Identify common physics simulation patterns in shader code.

    Args:
        shader_code (str): GLSL code to analyze

    Returns:
        dict: Dictionary of identified physics patterns
    """
    patterns = {
        # Particle systems
        'particle_system': 'particle' in shader_code.lower(),
        'particle_update': 'update' in shader_code.lower() and 'particle' in shader_code.lower(),
        'particle_render': 'render' in shader_code.lower() and 'particle' in shader_code.lower(),
        'particle_attributes': 'position' in shader_code.lower() or 'velocity' in shader_code.lower() or 'acceleration' in shader_code.lower(),
        
        # Physics integration methods
        'euler_integration': 'euler' in shader_code.lower() and 'integrate' in shader_code.lower(),
        'verlet_integration': 'verlet' in shader_code.lower(),
        'rk4_integration': 'rk4' in shader_code.lower() or ('runge' in shader_code.lower() and 'kutta' in shader_code.lower()),
        
        # Forces and physics concepts
        'gravity_force': 'gravity' in shader_code.lower() or ('force' in shader_code.lower() and 'grav' in shader_code.lower()),
        'spring_force': 'spring' in shader_code.lower() or ('hook' in shader_code.lower() and 'force' in shader_code.lower()),
        'collision_detection': 'collision' in shader_code.lower() and 'detect' in shader_code.lower(),
        'collision_response': 'collision' in shader_code.lower() and 'response' in shader_code.lower(),
        'constraint_solver': 'constraint' in shader_code.lower() and ('solve' in shader_code.lower() or 'solver' in shader_code.lower()),
        
        # Fluid simulation
        'fluid_simulation': 'fluid' in shader_code.lower() and 'simulate' in shader_code.lower(),
        'navier_stokes': 'navier' in shader_code.lower() and 'stokes' in shader_code.lower(),
        'smoke_simulation': 'smoke' in shader_code.lower() and 'simulate' in shader_code.lower(),
        
        # Physics properties
        'mass_property': 'mass' in shader_code.lower(),
        'velocity_property': 'velocity' in shader_code.lower(),
        'acceleration_property': 'acceleration' in shader_code.lower(),
        
        # Physics equations
        'f_ma': 'f' in shader_code.lower() and 'ma' in shader_code.lower() and '=' in shader_code,  # F=ma
        'momentum': 'momentum' in shader_code.lower(),
        'energy_conservation': 'energy' in shader_code.lower() and 'conserv' in shader_code.lower(),
        
        # Damping and friction
        'damping': 'damp' in shader_code.lower(),
        'friction': 'friction' in shader_code.lower(),
        'drag': 'drag' in shader_code.lower(),
        
        # Specialized physics
        'rigid_body': 'rigid' in shader_code.lower() and 'body' in shader_code.lower(),
        'cloth_simulation': 'cloth' in shader_code.lower() and 'simulate' in shader_code.lower(),
        'wave_equation': 'wave' in shader_code.lower() and 'equation' in shader_code.lower(),
    }
    
    # Filter only the patterns that were found
    active_patterns = {k: v for k, v in patterns.items() if v}
    return active_patterns


def find_matching_brace(code, start_pos):
    """
    Find the matching closing brace for an opening brace at start_pos.
    
    Args:
        code (str): The code string
        start_pos (int): Position of the opening brace
    
    Returns:
        int: Position of the matching closing brace, or -1 if not found
    """
    brace_count = 1
    pos = start_pos + 1
    
    while pos < len(code) and brace_count > 0:
        if code[pos] == '{':
            brace_count += 1
        elif code[pos] == '}':
            brace_count -= 1
        pos += 1
    
    return pos - 1 if brace_count == 0 else -1


def extract_complete_functions(code, pattern):
    """
    Extract complete functions that match a given pattern.
    
    Args:
        code (str): GLSL code to search in
        pattern (str): Base pattern to match (like 'integrate', 'particle', etc.)
    
    Returns:
        list: List of complete function definitions
    """
    functions = []
    
    # Updated regex to find function declarations with name containing the pattern
    func_patterns = [
        rf'(\w+)\s+[\w\d_]*{pattern}[\w\d_]*\s*\([^)]*\)\s*\{{',
        rf'(\w+)\s+[\w\d_]*{pattern}[\w\d_]*\s*\([^)]*\)\s+[\w\d_]*\s*\{{',  # With qualifier like 'const'
        rf'(\w+)\s+[\w\d_]*{pattern}[\w\d_]*\s*\([^)]*\)\s*\n\s*\{{',  # With newline
    ]
    
    for func_pattern in func_patterns:
        matches = re.finditer(func_pattern, code, re.IGNORECASE)
        
        for match in matches:
            # Find the opening brace position
            open_brace_pos = match.end() - 1  # Position of the opening brace
            
            # Find the matching closing brace
            close_brace_pos = find_matching_brace(code, open_brace_pos)
            
            if close_brace_pos != -1:
                # Extract the complete function
                func_start = match.start()
                func_end = close_brace_pos + 1
                function_code = code[func_start:func_end]
                
                # Clean up and add to results if it's not already there
                function_code = function_code.strip()
                if function_code and function_code not in functions:
                    functions.append(function_code)
    
    return functions


def analyze_physics_shaders():
    """
    Main function to analyze particle/physics simulation shaders.
    """
    print("Analyzing particle/physics simulation shaders...")
    
    physics_shaders = find_particle_physics_shaders()
    
    # Store shader codes and identified patterns
    shader_codes = []
    all_patterns = []
    pattern_counts = Counter()
    
    print("\nAnalyzing physics patterns in shaders...")
    for i, (filepath, shader_info) in enumerate(physics_shaders):
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(physics_shaders)} physics shaders...")
        
        shader_code = extract_shader_code(filepath)
        patterns = identify_physics_patterns(shader_code)
        
        shader_codes.append({
            'info': shader_info,
            'code': shader_code,
            'patterns': patterns
        })
        
        all_patterns.append(patterns)
        
        # Update pattern counts
        for pattern in patterns:
            pattern_counts[pattern] += 1
    
    print(f"\nAnalysis complete! Found {len(physics_shaders)} particle/physics simulation shaders.")
    
    # Print pattern distribution
    print(f"\nCommon physics patterns found:")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern.replace('_', ' ').title()}: {count} shaders")
    
    # Save analysis results
    save_physics_analysis(shader_codes, pattern_counts)
    
    return shader_codes, pattern_counts


def save_physics_analysis(shader_codes, pattern_counts):
    """
    Save the physics analysis results to files.
    """
    os.makedirs('analysis/physics', exist_ok=True)
    
    # Save pattern statistics
    with open('analysis/physics/pattern_stats.txt', 'w', encoding='utf-8') as f:
        f.write("Physics Pattern Statistics\n")
        f.write("=" * 50 + "\n")
        for pattern, count in pattern_counts.most_common():
            f.write(f"{pattern.replace('_', ' ').title()}: {count}\n")
    
    # Save detailed shader analysis
    with open('analysis/physics/shader_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Detailed Particle/Physics Shader Analysis\n")
        f.write("=" * 50 + "\n")
        for shader_data in shader_codes[:50]:  # Only first 50 for file size
            info = shader_data['info']
            patterns = shader_data['patterns']
            
            f.write(f"\nShader ID: {info['id']}\n")
            f.write(f"Name: {info['name']}\n")
            f.write(f"Author: {info['username']}\n")
            f.write(f"Tags: {', '.join(info['tags'])}\n")
            f.write(f"Patterns: {', '.join([p.replace('_', ' ').title() for p in patterns])}\n")
            f.write("-" * 30 + "\n")
    
    print("Particle/physics shader analysis saved to analysis/physics/ directory")


def extract_physics_modules(shader_codes):
    """
    Extract reusable physics modules from analyzed shaders.

    Args:
        shader_codes (list): List of shader data from analysis

    Returns:
        dict: Dictionary of extracted physics modules
    """
    print("\nExtracting reusable physics modules...")
    
    modules = {
        'particle_systems': set(),
        'integration_methods': set(),
        'forces': set(),
        'collision_detection': set(),
        'constraints': set(),
        'fluid_simulation': set()
    }
    
    total_processed = 0
    
    for shader_data in shader_codes:
        code = shader_data['code']
        
        # Extract particle system functions
        particle_funcs = extract_complete_functions(code, 'particle')
        particle_funcs += extract_complete_functions(code, 'update')
        modules['particle_systems'].update(particle_funcs)
        
        # Extract integration methods
        integration_funcs = extract_complete_functions(code, 'euler')
        integration_funcs += extract_complete_functions(code, 'verlet')
        integration_funcs += extract_complete_functions(code, 'integrate')
        modules['integration_methods'].update(integration_funcs)
        
        # Extract force functions
        force_funcs = extract_complete_functions(code, 'force')
        force_funcs += extract_complete_functions(code, 'gravity')
        force_funcs += extract_complete_functions(code, 'spring')
        modules['forces'].update(force_funcs)
        
        # Extract collision functions
        collision_funcs = extract_complete_functions(code, 'collision')
        collision_funcs += extract_complete_functions(code, 'collide')
        modules['collision_detection'].update(collision_funcs)
        
        # Extract constraint functions
        constraint_funcs = extract_complete_functions(code, 'constraint')
        constraint_funcs += extract_complete_functions(code, 'solve')
        modules['constraints'].update(constraint_funcs)
        
        # Extract fluid simulation functions
        fluid_funcs = extract_complete_functions(code, 'fluid')
        fluid_funcs += extract_complete_functions(code, 'smoke')
        modules['fluid_simulation'].update(fluid_funcs)
        
        total_processed += 1
        if total_processed % 100 == 0:
            print(f"Processed {total_processed}/{len(shader_codes)} shaders...")
    
    print(f"Extraction complete! Found:")
    for module_type, funcs in modules.items():
        print(f"  {module_type}: {len(funcs)} functions")
    
    # Save modules
    save_physics_modules(modules)
    
    return modules


def save_physics_modules(modules):
    """
    Save extracted physics modules to files.
    """
    os.makedirs('modules/physics', exist_ok=True)
    
    for module_type, func_list in modules.items():
        if func_list:  # Only save if there are modules of this type
            with open(f'modules/physics/{module_type}_functions.glsl', 'w', encoding='utf-8') as f:
                f.write(f"// Reusable {module_type.replace('_', ' ').title()} Physics Functions\n")
                f.write("// Automatically extracted from particle/physics simulation-related shaders\n\n")
                
                for i, func in enumerate(func_list, 1):
                    f.write(f"// Function {i}\n")
                    f.write(func)
                    f.write("\n\n")
    
    print("Physics modules saved to modules/physics/ directory")


def create_standardized_physics_modules():
    """
    Create standardized physics modules based on patterns found.
    """
    print("Creating standardized physics modules...")
    
    # Define standardized module templates with actual GLSL implementations
    standardized_modules = {
        'particle_system.glsl': generate_particle_system_glsl(),
        'integration_methods.glsl': generate_integration_methods_glsl(),
        'forces.glsl': generate_forces_glsl(),
        'collision_detection.glsl': generate_collision_detection_glsl(),
        'constraints.glsl': generate_constraints_glsl(),
        'fluid_simulation.glsl': generate_fluid_simulation_glsl()
    }
    
    os.makedirs('modules/physics/standardized', exist_ok=True)
    
    # Create standardized modules
    for filename, code in standardized_modules.items():
        filepath = f'modules/physics/standardized/{filename}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
    
    print(f"Created {len(standardized_modules)} standardized physics modules")


def generate_particle_system_glsl():
    """Generate GLSL implementation for particle systems."""
    return """// Particle system module
// Standardized particle system implementations

// Particle structure
struct Particle {
    vec3 position;
    vec3 velocity;
    vec3 acceleration;
    vec3 color;
    float size;
    float mass;
    float lifetime;
    float age;
};

// Initialize a particle
Particle initParticle(vec3 pos, vec3 vel, vec3 col, float sz, float m) {
    Particle p;
    p.position = pos;
    p.velocity = vel;
    p.acceleration = vec3(0.0);
    p.color = col;
    p.size = sz;
    p.mass = m;
    p.lifetime = 5.0; // Default lifetime
    p.age = 0.0;
    return p;
}

// Update particle position with velocity
Particle updateParticle(Particle p, float deltaTime) {
    p.velocity += p.acceleration * deltaTime;
    p.position += p.velocity * deltaTime;
    p.age += deltaTime;
    p.acceleration = vec3(0.0); // Reset acceleration
    return p;
}

// Apply force to particle (F = ma => a = F/m)
Particle applyForce(Particle p, vec3 force) {
    p.acceleration += force / p.mass;
    return p;
}

// Check if particle is still alive
bool isAlive(Particle p) {
    return p.age < p.lifetime;
}

// Create a particle emitter
vec3 emitParticle(vec2 uv, float time, vec3 emitterPos) {
    // Use time and UV to create unique particle properties
    float angle = time * 2.0 + uv.x * 10.0;
    float speed = 1.0 + sin(time + uv.y * 5.0);
    vec3 direction = vec3(cos(angle), sin(angle), 0.0);
    return emitterPos + direction * speed * 0.1;
}
"""


def generate_integration_methods_glsl():
    """Generate GLSL implementation for integration methods."""
    return """// Integration methods module
// Standardized physics integration implementations

// Euler integration (simple but less stable)
vec3 eulerIntegration(vec3 position, vec3 velocity, vec3 acceleration, float deltaTime) {
    velocity += acceleration * deltaTime;
    position += velocity * deltaTime;
    return position;
}

// Velocity-based Euler integration
vec3 velocityEulerIntegration(vec3 position, vec3 velocity, vec3 acceleration, float deltaTime) {
    vec3 newVelocity = velocity + acceleration * deltaTime;
    vec3 newPosition = position + newVelocity * deltaTime;
    return newPosition;
}

// Verlet integration (more stable for physics simulation)
vec3 verletIntegration(vec3 currentPosition, vec3 previousPosition, vec3 acceleration, float deltaTime) {
    vec3 temp = currentPosition;
    vec3 newPosition = 2.0 * currentPosition - previousPosition + acceleration * deltaTime * deltaTime;
    // Store current position for next frame as previous position
    previousPosition = temp;
    return newPosition;
}

// Verlet integration with damping
vec3 verletIntegrationDamped(vec3 currentPosition, vec3 previousPosition, vec3 acceleration, float deltaTime, float damping) {
    vec3 temp = currentPosition;
    vec3 newPosition = 2.0 * currentPosition - previousPosition + acceleration * deltaTime * deltaTime;
    newPosition = mix(newPosition, currentPosition, damping); // Apply damping
    previousPosition = temp;
    return newPosition;
}

// Calculate velocity from positions (for Verlet integration)
vec3 calculateVelocity(vec3 currentPosition, vec3 previousPosition, float deltaTime) {
    return (currentPosition - previousPosition) / deltaTime;
}
"""


def generate_forces_glsl():
    """Generate GLSL implementation for force calculations."""
    return """// Forces module
// Standardized force calculation implementations

// Apply gravity force
vec3 applyGravity(vec3 position, vec3 gravityVector, float mass) {
    return gravityVector * mass; // F = mg
}

// Apply spring force (Hooke's law: F = -kx)
vec3 applySpringForce(vec3 particlePos, vec3 anchorPos, float stiffness, float restLength) {
    vec3 displacement = particlePos - anchorPos;
    float currentLength = length(displacement);
    vec3 normalizedDisp = normalize(displacement);
    
    // F = -k * (currentLength - restLength)
    float forceMagnitude = -stiffness * (currentLength - restLength);
    return forceMagnitude * normalizedDisp;
}

// Apply damping force (proportional to velocity)
vec3 applyDamping(vec3 velocity, float dampingCoefficient) {
    return -velocity * dampingCoefficient;
}

// Apply drag force (proportional to velocity squared)
vec3 applyDrag(vec3 velocity, float dragCoefficient) {
    float speed = length(velocity);
    float dragMagnitude = dragCoefficient * speed * speed;
    return -normalize(velocity) * dragMagnitude;
}

// Apply repulsion force between particles
vec3 applyRepulsion(vec3 pos1, vec3 pos2, float strength) {
    vec3 diff = pos1 - pos2;
    float distance = max(length(diff), 0.1); // Avoid division by zero
    return normalize(diff) * strength / (distance * distance);
}

// Apply attraction force between particles
vec3 applyAttraction(vec3 pos1, vec3 pos2, float strength) {
    vec3 diff = pos2 - pos1;
    float distance = max(length(diff), 0.1); // Avoid division by zero
    return normalize(diff) * strength / (distance * distance);
}

// Apply wind force
vec3 applyWindForce(vec3 velocity, vec3 windDirection, float windStrength) {
    vec3 windEffect = windDirection * windStrength;
    return windEffect;
}
"""


def generate_collision_detection_glsl():
    """Generate GLSL implementation for collision detection."""
    return """// Collision detection module
// Standardized collision detection implementations

// Check collision between sphere and point
bool spherePointCollision(vec3 spherePos, float sphereRadius, vec3 point) {
    float distance = length(point - spherePos);
    return distance < sphereRadius;
}

// Check collision between two spheres
bool sphereSphereCollision(vec3 pos1, float radius1, vec3 pos2, float radius2) {
    float minDistance = radius1 + radius2;
    float actualDistance = length(pos2 - pos1);
    return actualDistance < minDistance;
}

// Check collision between sphere and plane
bool spherePlaneCollision(vec3 spherePos, float sphereRadius, vec3 planeNormal, float planeDistance) {
    float distanceToPlane = dot(spherePos, planeNormal) - planeDistance;
    return abs(distanceToPlane) < sphereRadius;
}

// Get collision response vector for sphere-plane collision
vec3 spherePlaneCollisionResponse(vec3 spherePos, vec3 sphereVel, vec3 planeNormal, float sphereRadius, float planeDistance) {
    float distance = dot(spherePos, planeNormal) - planeDistance;
    
    if (distance < sphereRadius) {
        // Calculate reflection vector
        vec3 reflection = reflect(sphereVel, planeNormal);
        // Add some damping to the reflection
        return reflection * 0.8;
    }
    
    return sphereVel; // No collision, return original velocity
}

// Check collision with rectangular bounds
bool boundsCollision(vec3 position, vec3 minBounds, vec3 maxBounds) {
    return position.x < minBounds.x || position.x > maxBounds.x ||
           position.y < minBounds.y || position.y > maxBounds.y ||
           position.z < minBounds.z || position.z > maxBounds.z;
}

// Constrain position to rectangular bounds with bounce
vec3 bounceBounds(vec3 position, vec3 velocity, vec3 minBounds, vec3 maxBounds) {
    vec3 newPos = position;
    vec3 newVel = velocity;
    
    if (position.x <= minBounds.x || position.x >= maxBounds.x) {
        newVel.x *= -0.8; // Reverse and dampen velocity
        newPos.x = position.x <= minBounds.x ? minBounds.x : maxBounds.x;
    }
    
    if (position.y <= minBounds.y || position.y >= maxBounds.y) {
        newVel.y *= -0.8;
        newPos.y = position.y <= minBounds.y ? minBounds.y : maxBounds.y;
    }
    
    if (position.z <= minBounds.z || position.z >= maxBounds.z) {
        newVel.z *= -0.8;
        newPos.z = position.z <= minBounds.z ? minBounds.z : maxBounds.z;
    }
    
    return newVel;
}
"""


def generate_constraints_glsl():
    """Generate GLSL implementation for constraints."""
    return """// Constraints module
// Standardized constraint solving implementations

// Distance constraint between two points
vec3 applyDistanceConstraint(vec3 posA, vec3 posB, float restDistance) {
    vec3 delta = posB - posA;
    float currentDistance = length(delta);
    
    if (currentDistance > 0.0) {
        float diff = (currentDistance - restDistance) / currentDistance;
        vec3 offset = delta * diff * 0.5;
        return offset;
    }
    
    return vec3(0.0);
}

// Solve distance constraint for two particles
void solveDistanceConstraint(inout vec3 posA, inout vec3 posB, float restDistance) {
    vec3 delta = posB - posA;
    float currentDistance = length(delta);
    float diff = (currentDistance - restDistance) / currentDistance;
    
    // Move both particles proportionally
    vec3 offset = delta * diff * 0.5;
    posA += offset;
    posB -= offset;
}

// Angle constraint between three points (A-B-C where B is the vertex)
void solveAngleConstraint(inout vec3 posA, inout vec3 posB, inout vec3 posC, float targetAngle) {
    vec3 ab = normalize(posA - posB);
    vec3 cb = normalize(posC - posB);
    
    float currentAngle = acos(dot(ab, cb));
    float angleDiff = targetAngle - currentAngle;
    
    // Rotate points to achieve target angle
    // This is a simplified implementation
    if (abs(angleDiff) > 0.01) {
        // Calculate rotation axis
        vec3 axis = normalize(cross(ab, cb));
        
        // Apply small rotation toward target angle
        float rotation = angleDiff * 0.1;
        mat3 rotationMatrix = rotationMatrix(axis, rotation);
        
        vec3 newA = posB + rotationMatrix * (posA - posB);
        vec3 newC = posB + rotationMatrix * (posC - posB);
        
        posA = newA;
        posC = newC;
    }
}

// Create rotation matrix around axis by angle
mat3 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat3(
        oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
        oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
        oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c
    );
}

// Position constraint (keep point within bounds)
vec3 applyPositionConstraint(vec3 position, vec3 minBounds, vec3 maxBounds) {
    return clamp(position, minBounds, maxBounds);
}

// Keep particles within a spherical volume
vec3 applySphericalConstraint(vec3 position, vec3 center, float radius) {
    vec3 offset = position - center;
    float distance = length(offset);
    
    if (distance > radius) {
        return center + normalize(offset) * radius;
    }
    
    return position;
}
"""


def generate_fluid_simulation_glsl():
    """Generate GLSL implementation for fluid simulation."""
    return """// Fluid simulation module
// Standardized fluid simulation implementations

// Simple grid-based fluid simulation - density advection
vec4 advectDensity(vec2 uv, vec2 velocity, float deltaTime, sampler2D density) {
    vec2 prevPos = uv - velocity * deltaTime;
    return texture(density, prevPos);
}

// Velocity advection step
vec2 advectVelocity(vec2 uv, vec2 velocity, float deltaTime, sampler2D velocityField) {
    vec2 prevPos = uv - velocity * deltaTime;
    return texture(velocityField, prevPos).xy;
}

// Apply simple buoyancy force
vec2 applyBuoyancy(vec3 velocity, float density, float temperature, vec3 ambientTemp, vec3 gravity) {
    // Simple buoyancy model: warmer and less dense fluid rises
    vec3 buoyancy = (temperature - ambientTemp) * vec3(0.0, 1.0, 0.0) * 0.1;
    return velocity + buoyancy + gravity;
}

// Apply vorticity confinement to preserve small-scale details
vec3 vorticityConfinement(vec2 uv, sampler2D velocityField, float gridScale) {
    // Calculate vorticity (curl of velocity field)
    vec2 dx = vec2(gridScale, 0.0);
    vec2 dy = vec2(0.0, gridScale);
    
    float vorticity = (texture(velocityField, uv + dx).y - texture(velocityField, uv - dx).y) / (2.0 * gridScale) 
                    - (texture(velocityField, uv + dy).x - texture(velocityField, uv - dy).x) / (2.0 * gridScale);
    
    // Calculate vorticity force
    vec2 force = vec2(-dFdy(vorticity), dFdx(vorticity));
    float curl = length(vec2(dFdx(vorticity), dFdy(vorticity)));
    force *= 0.01 / max(curl, 0.01); // Normalize and scale
    
    return vec3(force, 0.0);
}

// Simple particle-based fluid simulation using SPH (Smoothed Particle Hydrodynamics)
float sphKernel(vec3 r, float h) {
    float r2 = dot(r, r);
    float h2 = h * h;
    
    if (r2 > h2) return 0.0;
    
    float x = 1.0 - r2 / h2;
    return 8.0 / (3.14159 * h2) * x * x * x;
}

// SPH gradient for pressure calculation
vec3 sphPressureGradient(vec3 r, float h) {
    float r_len = length(r);
    float h2 = h * h;
    
    if (r_len > h || r_len < 0.0001) return vec3(0.0);
    
    float x = 1.0 - r_len / h;
    vec3 grad = -45.0 / (3.14159 * h2 * h) * x * x * normalize(r);
    return grad;
}
"""


def main():
    # Find particle/physics shaders
    physics_shaders = find_particle_physics_shaders()
    
    # Extract shader codes for a subset (first 500) for efficiency
    shader_codes = []
    for filepath, shader_info in physics_shaders[:500]:  # Limit to first 500 for efficiency
        shader_code = extract_shader_code(filepath)
        shader_codes.append({
            'info': shader_info,
            'code': shader_code
        })
    
    # Analyze shader patterns
    analyzed_shaders, pattern_counts = analyze_physics_shaders()
    
    # Extract specific physics functions
    modules = extract_physics_modules(analyzed_shaders)
    
    # Create standardized modules
    create_standardized_physics_modules()
    
    print("Particle/physics simulation shader analysis and module extraction completed!")


if __name__ == "__main__":
    main()