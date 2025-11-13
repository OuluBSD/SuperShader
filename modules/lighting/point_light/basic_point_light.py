
#!/usr/bin/env python3
'''
Basic Point Light Module with Interface Definition
Extracted from common lighting patterns in shader analysis
Pattern frequency: 405 occurrences
'''

# Interface definition
INTERFACE = {
    'inputs': [
        {'name': 'FragPos', 'type': 'vec3', 'direction': 'in', 'semantic': 'position'},
        {'name': 'Normal', 'type': 'vec3', 'direction': 'in', 'semantic': 'normal'}, 
        {'name': 'lightPos', 'type': 'vec3', 'direction': 'uniform', 'semantic': 'light_position'},
        {'name': 'lightColor', 'type': 'vec3', 'direction': 'uniform', 'semantic': 'light_color'}
    ],
    'outputs': [
        {'name': 'lightColorOut', 'type': 'vec3', 'direction': 'out', 'semantic': 'light_contribution'}
    ],
    'uniforms': [
        {'name': 'lightPos', 'type': 'vec3', 'semantic': 'light_position'},
        {'name': 'lightColor', 'type': 'vec3', 'semantic': 'light_color'}
    ]
}

# Pseudocode for basic point light calculation
pseudocode = '''
// Basic Point Light Implementation
vec3 calculatePointLight(vec3 position, vec3 normal, vec3 lightPos, vec3 lightColor) {
    // Calculate light direction
    vec3 lightDir = normalize(lightPos - position);
    
    // Calculate distance and attenuation
    float distance = length(lightPos - position);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    
    // Diffuse lighting
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Apply attenuation
    diffuse *= attenuation;
    
    return diffuse;
}
'''

def get_interface():
    '''Return the interface definition for this module'''
    return INTERFACE

def get_pseudocode():
    '''Return the pseudocode for this lighting module'''
    return pseudocode

def get_metadata():
    '''Return metadata about this module'''
    return {
        'name': 'basic_point_light',
        'type': 'lighting',
        'patterns': ['Point Light', 'Light Attenuation'],
        'frequency': 405,
        'dependencies': [],
        'conflicts': [],
        'description': 'Basic point light calculation with attenuation',
        'interface': INTERFACE
    }
