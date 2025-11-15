#!/usr/bin/env python3
"""
Advanced Game Module with Branching for Conflicting Features
This module demonstrates different game-related approaches with branching for conflicting features
"""

# Interface definition with branching options
INTERFACE = {
    'inputs': [
        {'name': 'fragCoord', 'type': 'vec2', 'direction': 'in', 'semantic': 'fragment_coordinates'},
        {'name': 'mousePos', 'type': 'vec2', 'direction': 'uniform', 'semantic': 'mouse_position'},
        {'name': 'clickState', 'type': 'float', 'direction': 'uniform', 'semantic': 'click_status'},
        {'name': 'gameTime', 'type': 'float', 'direction': 'uniform', 'semantic': 'game_time'},
        {'name': 'playerPos', 'type': 'vec2', 'direction': 'uniform', 'semantic': 'player_position'},
        {'name': 'enemyPos', 'type': 'vec2', 'direction': 'uniform', 'semantic': 'enemy_position'},
        {'name': 'health', 'type': 'float', 'direction': 'uniform', 'semantic': 'player_health'},
        {'name': 'score', 'type': 'int', 'direction': 'uniform', 'semantic': 'player_score'}
    ],
    'outputs': [
        {'name': 'visualOutput', 'type': 'vec4', 'direction': 'out', 'semantic': 'game_visualization'},
        {'name': 'interactionResult', 'type': 'float', 'direction': 'out', 'semantic': 'interaction_status'},
        {'name': 'hudElements', 'type': 'vec3', 'direction': 'out', 'semantic': 'hud_visualization'}
    ],
    'uniforms': [
        {'name': 'mousePos', 'type': 'vec2', 'semantic': 'mouse_position'},
        {'name': 'clickState', 'type': 'float', 'semantic': 'click_status'},
        {'name': 'gameTime', 'type': 'float', 'semantic': 'game_time'},
        {'name': 'playerPos', 'type': 'vec2', 'semantic': 'player_position'},
        {'name': 'enemyPos', 'type': 'vec2', 'semantic': 'enemy_position'},
        {'name': 'health', 'type': 'float', 'semantic': 'player_health'},
        {'name': 'score', 'type': 'int', 'semantic': 'player_score'}
    ],
    'branches': {
        'hud_style': {
            'minimalist': {
                'name': 'Minimalist HUD',
                'description': 'Clean, minimal heads-up display',
                'requires': [],
                'conflicts': ['detailed', 'themed']
            },
            'detailed': {
                'name': 'Detailed HUD',
                'description': 'Information-rich HUD with many elements',
                'requires': [],
                'conflicts': ['minimalist', 'themed']
            },
            'themed': {
                'name': 'Themed HUD',
                'description': 'Stylistically consistent with game theme',
                'requires': ['theme_assets'],
                'conflicts': ['minimalist', 'detailed']
            }
        },
        'interaction_mode': {
            'click': {
                'name': 'Click-based Interaction',
                'description': 'Interaction through mouse clicks',
                'requires': [],
                'conflicts': ['hover', 'touch']
            },
            'hover': {
                'name': 'Hover-based Interaction',
                'description': 'Interaction through mouse hover',
                'requires': [],
                'conflicts': ['click', 'touch']
            },
            'touch': {
                'name': 'Touch-based Interaction',
                'description': 'Interaction through touch input',
                'requires': ['touch_support'],
                'conflicts': ['click', 'hover']
            }
        },
        'visual_feedback': {
            'subtle': {
                'name': 'Subtle Feedback',
                'description': 'Minimal visual feedback for interactions',
                'requires': [],
                'conflicts': ['flashy', 'animated']
            },
            'flashy': {
                'name': 'Flashy Feedback',
                'description': 'Highly visible visual feedback',
                'requires': [],
                'conflicts': ['subtle', 'animated']
            },
            'animated': {
                'name': 'Animated Feedback',
                'description': 'Animated feedback effects',
                'requires': ['animation_support'],
                'conflicts': ['subtle', 'flashy']
            }
        }
    }
}

# Pseudocode for different game algorithms
pseudocode = {
    'minimalist_hud': '''
// Minimalist HUD implementation
vec3 renderMinimalistHUD(float health, int score, float time) {
    vec3 hudColor = vec3(0.0);
    
    // Simple health indicator (left side)
    float healthWidth = health * 0.2;  // Max 20% of screen width
    if (gl_FragCoord.x < healthWidth && gl_FragCoord.y < 0.02) {
        hudColor = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), health);
    }
    
    // Simple score display (right side)
    float scoreX = 0.95;  // Right aligned
    if (gl_FragCoord.x > scoreX && gl_FragCoord.y < 0.03) {
        // In a real implementation, this would render digits with SDFs
        hudColor = vec3(1.0);  // White text
    }
    
    return hudColor;
}
    ''',
    
    'detailed_hud': '''
// Detailed HUD implementation
vec3 renderDetailedHUD(float health, int score, vec2 playerPos, vec2 enemyPos, float time) {
    vec3 hudColor = vec3(0.0);
    
    // Health bar with gradient
    float healthBarX = 0.1;
    float healthBarY = 0.05;
    float healthBarWidth = 0.2;
    float healthBarHeight = 0.02;
    
    if (gl_FragCoord.x > healthBarX && gl_FragCoord.x < healthBarX + healthBarWidth * health &&
        gl_FragCoord.y > healthBarY && gl_FragCoord.y < healthBarY + healthBarHeight) {
        // Color gradient from red to green
        hudColor = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), health);
    }
    
    // Mini-map
    float mapSize = 0.15;
    float mapX = 0.8;
    float mapY = 0.05;
    
    if (gl_FragCoord.x > mapX && gl_FragCoord.x < mapX + mapSize &&
        gl_FragCoord.y > mapY && gl_FragCoord.y < mapY + mapSize) {
        
        // Scale positions to map coordinates
        vec2 normPlayerPos = playerPos * 0.5 + 0.5;  // Normalize to 0-1
        vec2 normEnemyPos = enemyPos * 0.5 + 0.5;
        vec2 mapPos = (gl_FragCoord.xy - vec2(mapX, mapY)) / mapSize;
        
        // Draw player as blue dot
        float playerDist = distance(mapPos, normPlayerPos);
        if (playerDist < 0.02) {
            hudColor = vec3(0.0, 0.5, 1.0);
        }
        
        // Draw enemy as red dot
        float enemyDist = distance(mapPos, normEnemyPos);
        if (enemyDist < 0.02) {
            hudColor = vec3(1.0, 0.0, 0.0);
        }
    }
    
    // Score and additional stats
    if (gl_FragCoord.y < 0.03) {
        hudColor = vec3(0.8, 0.8, 0.8);  // Light gray for text
    }
    
    return hudColor;
}
    ''',
    
    'themed_hud': '''
// Themed HUD implementation
vec3 renderThemedHUD(float health, int score, float time) {
    vec3 hudColor = vec3(0.0);
    
    // Themed health bar with decorative elements
    float healthBarX = 0.05;
    float healthBarY = 0.05;
    float healthWidth = health * 0.3;
    
    // Outer frame
    if ((abs(gl_FragCoord.x - healthBarX) < 0.005 || abs(gl_FragCoord.x - (healthBarX + 0.3)) < 0.005) ||
        (abs(gl_FragCoord.y - healthBarY) < 0.005 || abs(gl_FragCoord.y - (healthBarY + 0.03)) < 0.005)) {
        hudColor = vec3(0.8, 0.6, 0.2);  // Gold frame
    }
    
    // Health fill
    if (gl_FragCoord.x > healthBarX && gl_FragCoord.x < healthBarX + healthWidth &&
        gl_FragCoord.y > healthBarY && gl_FragCoord.y < healthBarY + 0.03) {
        // Themed color based on game style
        vec3 healthColor = mix(vec3(0.8, 0.1, 0.1), vec3(0.2, 0.8, 0.2), health);
        
        // Add pulsing effect
        float pulse = 0.1 * sin(time * 8.0) * health;
        hudColor = healthColor + vec3(pulse);
    }
    
    // Themed score display
    float scoreX = 0.7;
    if (gl_FragCoord.x > scoreX && gl_FragCoord.y < 0.05) {
        // Themed text appearance
        hudColor = vec3(0.9, 0.85, 0.6);  // Amber color
    }
    
    return hudColor;
}
    ''',
    
    'click_interaction': '''
// Click-based interaction handling
float handleClickInteraction(vec2 fragCoord, vec2 mousePos, vec4 interactionArea, float clickState) {
    // Check if within interaction area
    bool inX = fragCoord.x > interactionArea.x && fragCoord.x < interactionArea.x + interactionArea.z;
    bool inY = fragCoord.y > interactionArea.y && fragCoord.y < interactionArea.y + interactionArea.w;
    
    // Check if clicked while over the area
    float over = inX && inY ? 1.0 : 0.0;
    return over * clickState;
}
    ''',
    
    'hover_interaction': '''
// Hover-based interaction handling
float handleHoverInteraction(vec2 fragCoord, vec2 mousePos, vec4 interactionArea) {
    // Check if within interaction area
    bool inX = fragCoord.x > interactionArea.x && fragCoord.x < interactionArea.x + interactionArea.z;
    bool inY = fragCoord.y > interactionArea.y && fragCoord.y < interactionArea.y + interactionArea.w;
    
    return inX && inY ? 1.0 : 0.0;
}
    ''',
    
    'touch_interaction': '''
// Touch-based interaction handling
float handleTouchInteraction(vec2 fragCoord, vec2 touchPos, float touchRadius) {
    float dist = distance(fragCoord, touchPos);
    return 1.0 - smoothstep(0.0, touchRadius, dist);
}
    ''',
    
    'subtle_feedback': '''
// Subtle visual feedback
vec3 subtleFeedback(vec2 fragCoord, float isActive, float time) {
    // Minimal effect - slight color variation
    vec3 feedback = vec3(0.0);
    if (isActive > 0.5) {
        feedback = vec3(0.1, 0.1, 0.1);  // Slight brightening
    }
    return feedback;
}
    ''',
    
    'flashy_feedback': '''
// Flashy visual feedback
vec3 flashyFeedback(vec2 fragCoord, float isActive, float time) {
    // Bright, attention-grabbing effect
    vec3 feedback = vec3(0.0);
    if (isActive > 0.5) {
        // Bright pulsing effect
        float pulse = 0.5 + 0.5 * sin(time * 20.0);
        feedback = vec3(1.0, 0.8, 0.0) * pulse;  // Bright yellow
    }
    return feedback;
}
    ''',
    
    'animated_feedback': '''
// Animated visual feedback
vec3 animatedFeedback(vec2 fragCoord, float isActive, float time) {
    // Animated effect with movement
    vec3 feedback = vec3(0.0);
    if (isActive > 0.5) {
        // Animated expanding circle
        float distFromCenter = distance(fragCoord, vec2(0.5, 0.5));
        float anim = sin((distFromCenter - time * 5.0) * 5.0) * 0.5 + 0.5;
        anim *= 1.0 - smoothstep(0.0, 0.5, distFromCenter);
        
        feedback = vec3(0.0, 0.8, 1.0) * anim;  // Cyan animated effect
    }
    return feedback;
}
    '''
}

def get_interface():
    """Return the interface definition for this module"""
    return INTERFACE

def get_pseudocode(branch_name=None):
    """Return the pseudocode for this game module or specific branch"""
    if branch_name and branch_name in pseudocode:
        return pseudocode[branch_name]
    else:
        # Return all pseudocodes
        return pseudocode

def get_metadata():
    """Return metadata about this module"""
    return {
        'name': 'game_advanced_branching',
        'type': 'game',
        'patterns': ['Minimalist HUD', 'Detailed HUD', 'Themed HUD', 'Click-based Interaction',
                     'Hover-based Interaction', 'Touch-based Interaction', 'Subtle Feedback',
                     'Flashy Feedback', 'Animated Feedback'],
        'frequency': 200,
        'dependencies': [],
        'conflicts': [],
        'description': 'Advanced game-related algorithms with branching for different approaches',
        'interface': INTERFACE,
        'branches': INTERFACE['branches']
    }

def validate_branches(selected_branches):
    """Validate that the selected branches don't have conflicts"""
    # Check for conflicts between different branch categories
    return True