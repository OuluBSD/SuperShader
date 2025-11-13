# phong_functions

**Category:** lighting
**Type:** extracted

## Dependencies
normal_mapping, lighting

## Tags
lighting, color

## Code
```glsl
// Reusable Phong Lighting Functions
// Automatically extracted from lighting-related shaders

// Function 1
float normalizedPhongRef(float shininess, vec3 n, vec3 vd, vec3 ld){
    return 1.-normalizedPhong(shininess, n, vd, ld);
}

// Function 2
float phong(vec3 l, vec3 e, vec3 n, float power) {
    float nrm = (power + 8.0) / (PI * 8.0);
    return pow(max(dot(l,reflect(e,n)),0.0), power) * nrm;
}

// Function 3
float BlinnPhongRef(float shininess, vec3 n, vec3 vd, vec3 ld){
    vec3 h  = normalize(-vd+ld);
    return 1.-pow(max(0., dot(h, n)), shininess);
}

// Function 4
vec3 phong( vec3 v, vec3 n, vec3 eye ) {
	// ...add lights here...
	
	float shininess = 16.0;
	
	vec3 final = vec3( 0.0 );
	
	vec3 ev = normalize( v - eye );
	vec3 ref_ev = reflect( ev, n );
	
	// light 0
	{
		vec3 light_pos   = vec3( 20.0, 20.0, 20.0 );
		vec3 light_color = vec3( 1.0, 0.7, 0.7 );
	
		vec3 vl = normalize( light_pos - v );
	
		float diffuse  = max( 0.0, dot( vl, n ) );
		float specular = max( 0.0, dot( vl, ref_ev ) );
		specular = pow( specular, shininess );
		
		final += light_color * ( diffuse + specular ); 
	}
	
	// light 1
	{
		vec3 light_pos   = vec3( -20.0, -20.0, -20.0 );
		vec3 light_color = vec3( 0.3, 0.7, 1.0 );
	
		vec3 vl = normalize( light_pos - v );
		float diffuse  = max( 0.0, dot( vl, n ) );
		float specular = max( 0.0, dot( vl, ref_ev ) );
		specular = pow( specular, shininess );
		
		final += light_color * ( diffuse + specular ); 
	}

	return final;
}

// Function 5
vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light1Pos = vec3(4.0,
                          2.0,
                          4.0);
    vec3 light1Intensity = vec3(0.8);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light1Pos,
                                  light1Intensity);   
    return color;
}

// Function 6
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    lightPos = eye;
    vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    float dotLN = dot(L, N);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

// Function 7
vec3 phong(vec3 normal, vec3 light, vec3 view)
{
    const vec3 ambientColor = vec3(1., 1., 1.);
    const vec3 diffuseColor = vec3(1., 1., 1.);
    const vec3 specularColor = vec3(1., 1., 1.);
    const float shininess = 16.;
    const float ambientStrength = .25;
    
    vec3 diffuse = max(dot(normal, light), 0.) * diffuseColor;
    // light is negated because the first argument to reflect is the incident vector
    vec3 specular = pow(max(dot(reflect(-light, normal), view), 0.), shininess) * specularColor;
    vec3 ambient = ambientStrength * ambientColor;
    
    return diffuse + specular + ambient;
}

// Function 8
float normalizedPhong(float shininess, vec3 n, vec3 vd, vec3 ld){
    float norm_factor = (shininess+1.) / (2.*PI);
    vec3 reflect_light = normalize(reflect(ld, n));
    return pow(max(dot(-vd, reflect_light), 0.), shininess) * norm_factor;
}

// Function 9
float normalizedBlinnPhong(float shininess, vec3 n, vec3 vd, vec3 ld){
    float norm_factor = (shininess+1.) / (2.*PI);
    vec3 h  = normalize(-vd+ld);
    return pow(max(0., dot(h, n)), shininess) * norm_factor;
}


```