# blinn_phong_functions

**Category:** lighting
**Type:** extracted

## Dependencies
normal_mapping, lighting

## Tags
lighting

## Code
```glsl
// Reusable Blinn Phong Lighting Functions
// Automatically extracted from lighting-related shaders

// Function 1
float Blinn(in vec3 hn, in vec3 rd, in vec3 lv, in float roughness) {
    vec3 H = normalize(rd + lv);
    float dotNH = clamp(dot(hn, H), 0., 1.);
    return (roughness + 2.) / (8. * pi) * pow(dotNH, roughness);
}

// Function 2
float BlinnPhongRef(float shininess, vec3 n, vec3 vd, vec3 ld){
    vec3 h  = normalize(-vd+ld);
    return 1.-pow(max(0., dot(h, n)), shininess);
}

// Function 3
float normalizedBlinnPhong(float shininess, vec3 n, vec3 vd, vec3 ld){
    float norm_factor = (shininess+1.) / (2.*PI);
    vec3 h  = normalize(-vd+ld);
    return pow(max(0., dot(h, n)), shininess) * norm_factor;
}


```