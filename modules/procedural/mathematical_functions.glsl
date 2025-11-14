// Reusable Mathematical Procedural Functions
// Automatically extracted from procedural-related shaders

// Function 1
vec3 hsv2rgb_trigonometric( in vec3 c )
{
    vec3 rgb = 0.5 + 0.5*cos((c.x*6.0+vec3(0.0,4.0,2.0))*3.14159/3.0);

	return c.z * mix( vec3(1.0), rgb, c.y);
}

// Function 2
vec2 trig( float a ) {
    //return trug2( a );
    return vec2( cos( a  * TAU ), sin( a * TAU ) );
}

// Function 3
float trigNoise3D(in vec3 p){

    
    float res = 0., sum = 0.;
    
    // IQ's cheap, texture-lookup noise function. Very efficient, but still 
    // a little too processor intensive for multiple layer usage in a largish 
    // "for loop" setup. Therefore, just one layer is being used here.
    float n = n3D(p*8. + iTime*2.);


    // Two sinusoidal layers. I'm pretty sure you could get rid of one of 
    // the swizzles (I have a feeling the GPU doesn't like them as much), 
    // which I'll try to do later.
    
    vec3 t = sin(p.yzx*3.14159265 + cos(p.zxy*3.14159265+1.57/2.))*0.5 + 0.5;
    p = p*1.5 + (t - 1.5); //  + iTime*0.1
    res += (dot(t, vec3(0.333)));

    t = sin(p.yzx*3.14159265 + cos(p.zxy*3.14159265+1.57/2.))*0.5 + 0.5;
    res += (dot(t, vec3(0.333)))*0.7071;    
	 
	return ((res/1.7071))*0.85 + n*0.15;
}

// Function 4
float trigger(float a, float b, float t) {
	return step(a, t) - step(b, t);
}

// Function 5
float TrigNoise(vec3 x)
{
    return TrigNoise(x, 2.0, 1.0);
}

// Function 6
float trig3(in vec3 p){
    p = cos(p*2. + (cos(p.yzx) + 1.)*1.57);// + iTime*1.
    return dot(p, vec3(0.1666)) + 0.5;
}

// Function 7
float distriGGX (in vec3 N, in vec3 H, in float roughness) {
    float a2     = roughness * roughness;
    float NdotH  = max (dot (N, H), .0);
    float NdotH2 = NdotH * NdotH;

    float nom    = a2;
    float denom  = (NdotH2 * (a2 - 1.) + 1.);
    denom        = PI * denom * denom;

    return nom / denom;
}

// Function 8
float trigNoise3D(in vec3 p){

    // 3D transformation matrix.
    const mat3 m3RotTheta = mat3(0.25, -0.866, 0.433, 0.9665, 0.25, -0.2455127, -0.058, 0.433, 0.899519 )*1.5;
  
	float res = 0.;

    float t = trig3(p*PI);
	p += (t - iTime*0.25);
    p = m3RotTheta*p;
    //p = (p+0.7071)*1.5;
    res += t;
    
    t = trig3(p*PI); 
	p += (t - iTime*0.25)*0.7071;
    p = m3RotTheta*p;
     //p = (p+0.7071)*1.5;
    res += t*0.7071;

    t = trig3(p*PI);
	res += t*0.5;
	 
	return res/2.2071;
}

// Function 9
float bracketRight(vec2 uv){
    uv.x-=size.x*1.5;
    float p = 1.3;
    uv.y = abs(uv.y);
    float a = atan(uv.x, uv.y);
    float x = abs(length(uv)-size.x*2.);
    uv.y = -uv.y;
    x = mix(x, length(uv+vec2(cos(p), sin(p))*size.x*2.), step(-.3, a));
    return x;
}

// Function 10
float trigNoise3D(in vec3 p){

    p /= 2.;
    float res = 0., sum = 0.;
    
    // IQ's cheap, texture-lookup noise function. Very efficient, but still 
    // a little too processor intensive for multiple layer usage in a largish 
    // "for loop" setup. Therefore, just one layer is being used here.
    float n = pn(p*8. + iTime*2.);


    // Two sinusoidal layers. I'm pretty sure you could get rid of one of 
    // the swizzles (I have a feeling the GPU doesn't like them as much), 
    // which I'll try to do later.
    
    vec3 t = sin(p.yzx*3.14159265 + cos(p.zxy*3.14159265+1.57/2.))*0.5 + 0.5;
    p = p*1.5 + (t - 1.5); //  + iTime*0.1
    res += (dot(t, vec3(0.333)));

    t = sin(p.yzx*3.14159265 + cos(p.zxy*3.14159265+1.57/2.))*0.5 + 0.5;
    res += (dot(t, vec3(0.333)))*0.7071;    
	 
	return ((res/1.7071))*0.85 + n*0.15;
}

// Function 11
float TrigNoise(vec3 x, float a, float b)
{
    vec4 u = vec4(dot(x, vec3( 1.0, 1.0, 1.0)), 
                  dot(x, vec3( 1.0,-1.0,-1.0)), 
                  dot(x, vec3(-1.0, 1.0,-1.0)),
                  dot(x, vec3(-1.0,-1.0, 1.0))) * a;

    return dot(sin(x     + cos(u.xyz) * b), 
               cos(x.zxy + sin(u.zwx) * b));
}

// Function 12
vec3 getTriGrad(vec3 p, float eps) 
{ 
    vec2 d=vec2(eps,0); 
    float d0=triDist(p);
    return vec3(triDist(p+d.xyy)-d0,triDist(p+d.yxy)-d0,triDist(p+d.yyx)-d0)/eps; 
}

// Function 13
float trig3(in vec3 p){
    p = cos(p*2. + (cos(p.yzx) + 1. + iTime*4.)*1.57);
    return dot(p, vec3(0.1666)) + 0.5;
}

// Function 14
float trigNoise3D(in vec3 p){

    // 3D transformation matrix.
    const mat3 m3RotTheta = mat3(0.25, -0.866, 0.433, 0.9665, 0.25, -0.2455127, -0.058, 0.433, 0.899519 )*1.5;
  
	float res = 0.;

    float t = trig3(p*3.14159265);
	p += (t);
    p = m3RotTheta*p;
    //p = (p+0.7071)*1.5;
    res += t;
    
    t = trig3(p*3.14159265); 
	p += (t)*0.7071;
    p = m3RotTheta*p;
     //p = (p+0.7071)*1.5;
    res += t*0.7071;

    t = trig3(p*3.14159265);
	res += t*0.5;
	 
	return res/2.2071;
}

