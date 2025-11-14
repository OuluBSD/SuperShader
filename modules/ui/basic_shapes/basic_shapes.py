#!/usr/bin/env python3
'''
Basic Shapes Module with Interface Definition
Extracted from common UI/2D graphics patterns in shader analysis
Pattern frequency: 2584 occurrences
'''

# Interface definition
INTERFACE = {
    'inputs': [
        {'name': 'fragCoord', 'type': 'vec2', 'direction': 'in', 'semantic': 'fragment_coordinates'},
        {'name': 'shapePosition', 'type': 'vec2', 'direction': 'uniform', 'semantic': 'shape_position'},
        {'name': 'shapeSize', 'type': 'vec2', 'direction': 'uniform', 'semantic': 'shape_dimensions'},
        {'name': 'time', 'type': 'float', 'direction': 'uniform', 'semantic': 'time_parameter'},
        {'name': 'antialiasing', 'type': 'float', 'direction': 'uniform', 'semantic': 'antialiasing_factor'}
    ],
    'outputs': [
        {'name': 'shapeMask', 'type': 'float', 'direction': 'out', 'semantic': 'shape_coverage'},
        {'name': 'distance', 'type': 'float', 'direction': 'out', 'semantic': 'distance_to_edge'},
        {'name': 'color', 'type': 'vec3', 'direction': 'out', 'semantic': 'shape_color'}
    ],
    'uniforms': [
        {'name': 'shapePosition', 'type': 'vec2', 'semantic': 'shape_position'},
        {'name': 'shapeSize', 'type': 'vec2', 'semantic': 'shape_dimensions'},
        {'name': 'time', 'type': 'float', 'semantic': 'time_parameter'},
        {'name': 'antialiasing', 'type': 'float', 'semantic': 'antialiasing_factor'}
    ]
}

# Pseudocode for basic shapes
pseudocode = '''
// Basic Shapes Implementation

// Rectangle drawing with antialiasing
float drawRectangle(vec2 fragCoord, vec2 position, vec2 size, float antialiasing) {
    vec2 halfSize = size * 0.5;
    vec2 coord = fragCoord - position - halfSize;
    vec2 edge = abs(coord) - halfSize;
    float dist = length(max(edge, 0.0)) + min(max(edge.x, edge.y), 0.0);
    
    // Apply antialiasing
    float alpha = 1.0 - smoothstep(-antialiasing, antialiasing, dist);
    
    return alpha;
}

// Circle drawing with antialiasing
float drawCircle(vec2 fragCoord, vec2 position, float radius, float antialiasing) {
    float dist = distance(fragCoord, position);
    float alpha = 1.0 - smoothstep(radius - antialiasing, radius + antialiasing, dist);
    
    return alpha;
}

// Ellipse drawing
float drawEllipse(vec2 fragCoord, vec2 position, vec2 radius, float antialiasing) {
    vec2 coord = fragCoord - position;
    float dist = length(coord / radius);
    float alpha = 1.0 - smoothstep(1.0 - antialiasing, 1.0 + antialiasing, dist);
    
    return alpha;
}

// Line drawing
float drawLine(vec2 fragCoord, vec2 start, vec2 end, float thickness, float antialiasing) {
    vec2 line = end - start;
    float length = length(line);
    vec2 dir = line / length;
    
    vec2 coord = fragCoord - start;
    float proj = dot(coord, dir);
    
    // Clamp projection to line segment
    proj = clamp(proj, 0.0, length);
    
    vec2 closestPoint = start + dir * proj;
    float dist = distance(fragCoord, closestPoint);
    
    float alpha = 1.0 - smoothstep(thickness * 0.5 - antialiasing, thickness * 0.5 + antialiasing, dist);
    
    return alpha;
}

// Triangle drawing
float drawTriangle(vec2 fragCoord, vec2 p0, vec2 p1, vec2 p2, float antialiasing) {
    // Calculate barycentric coordinates
    vec2 e0 = p1 - p0;
    vec2 e1 = p2 - p1;
    vec2 e2 = p0 - p2;
    
    vec2 v0 = fragCoord - p0;
    vec2 v1 = fragCoord - p1;
    vec2 v2 = fragCoord - p2;
    
    float area = cross(e0, e2);
    float s = cross(v0, e2) / area;
    float t = cross(e0, v0) / area;
    
    float w0 = s;
    float w1 = t;
    float w2 = 1.0 - s - t;
    
    float inside = step(0.0, w0) * step(0.0, w1) * step(0.0, w2);
    
    // Calculate distance to edges for antialiasing
    float d0 = sign(cross(e0, v0)) * length(cross(e0, v0)) / length(e0);
    float d1 = sign(cross(e1, v1)) * length(cross(e1, v1)) / length(e1);
    float d2 = sign(cross(e2, v2)) * length(cross(e2, v2)) / length(e2);
    
    float dist = min(min(d0, d1), d2);
    
    // Apply antialiasing
    float alpha = inside * (1.0 - smoothstep(0.0, antialiasing, -dist));
    
    return alpha;
}

// Rounded rectangle
float drawRoundedRect(vec2 fragCoord, vec2 position, vec2 size, float radius, float antialiasing) {
    vec2 halfSize = size * 0.5;
    vec2 coord = fragCoord - position - halfSize;
    vec2 corner = abs(coord) - halfSize + radius;
    
    float dist = min(max(corner.x, corner.y), 0.0) + length(max(corner, 0.0)) - radius;
    
    float alpha = 1.0 - smoothstep(-antialiasing, antialiasing, dist);
    
    return alpha;
}

// Gradient fill rectangle
vec3 gradientFill(vec2 fragCoord, vec2 position, vec2 size, vec3 color1, vec3 color2, float angle) {
    vec2 rectCoord = (fragCoord - position) / size;
    
    // Rotate coordinate system for angle
    float cosA = cos(angle);
    float sinA = sin(angle);
    mat2 rot = mat2(cosA, -sinA, sinA, cosA);
    vec2 rotatedCoord = rot * rectCoord;
    
    // Create gradient
    float t = rotatedCoord.x;
    t = clamp(t, 0.0, 1.0);  // Ensure value is in range [0, 1]
    
    return mix(color1, color2, t);
}

// Checkered pattern
float checkeredPattern(vec2 fragCoord, float scale) {
    vec2 c = floor(fragCoord * scale);
    return mod(c.x + c.y, 2.0);
}

// Stroke for shapes
float shapeStroke(float shapeMask, float thickness, float antialiasing) {
    float distance = shapeMask;
    float outer = 1.0 - smoothstep(-antialiasing, antialiasing, distance);
    float inner = 1.0 - smoothstep(-antialiasing, antialiasing, distance - thickness);
    
    return outer - inner;
}

// Shadow effect
float dropShadow(vec2 fragCoord, vec2 position, vec2 size, vec2 offset, float radius, float antialiasing) {
    vec2 shadowPos = position + offset;
    vec2 halfSize = size * 0.5;
    vec2 coord = fragCoord - shadowPos - halfSize;
    vec2 edge = abs(coord) - halfSize - radius * 0.5;
    float dist = length(max(edge, 0.0)) + min(max(edge.x, edge.y), 0.0);
    
    float alpha = 1.0 - smoothstep(-antialiasing, antialiasing, dist - radius);
    
    // Fade with distance
    alpha *= 1.0 - distance(fragCoord, shadowPos) / 100.0;  // Example fade
    
    return alpha * 0.5;  // Shadow intensity
}
'''

def get_interface():
    '''Return the interface definition for this module'''
    return INTERFACE

def get_pseudocode():
    '''Return the pseudocode for this UI/2D graphics module'''
    return pseudocode

def get_metadata():
    '''Return metadata about this module'''
    return {
        'name': 'basic_shapes',
        'type': 'ui',
        'patterns': ['Basic Shapes', 'UI Rendering', '2D Graphics'],
        'frequency': 2584,
        'dependencies': [],
        'conflicts': [],
        'description': 'Basic 2D shapes drawing functions for UI elements',
        'interface': INTERFACE
    }