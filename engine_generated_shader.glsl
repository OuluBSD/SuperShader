#version 330 core

// Uniforms
uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform sampler2D normalMap;
uniform sampler2D shadowMap;

// Input variables
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

// Output
out vec4 FragColor;


void main() {
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    // Initialize color
    vec3 result = vec3(0.0);

    // Final color
    FragColor = vec4(result, 1.0);
}
