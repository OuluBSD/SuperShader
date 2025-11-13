# ambient_functions

**Category:** lighting
**Type:** extracted

## Dependencies
normal_mapping, lighting, raymarching

## Tags
lighting

## Code
```glsl
// Reusable Ambient Lighting Functions
// Automatically extracted from lighting-related shaders

// Function 1
float AmbientOcclusion( in vec3 pos, in vec3 nor )
{
	float totao = 0.0;
    float sca = 1.0;
    for( int aoi=0; aoi<8; aoi++ )
    {
        float hr = 0.01 + 1.2*pow(float(aoi)/8.0,1.5);
        vec3 aopos =  nor * hr + pos;
        float dd = Map( aopos, 1.0 ).x;
        totao += -(dd-hr)*sca;
        sca *= 0.85;
    }
    return clamp( 1.0 - 0.6*totao, 0.0, 1.0 );
}

// Function 2
float AmbientOcclusion (vec3 p,vec3 n,float d,float s){float r=1.;
  for(int i=0;i<5;++i){if(--s<0.)break;r-=(s*d-(df(p+n*s*d)))/pow(2.,s);}return r;}

// Function 3
vec4 ambient(light _l)
{
	return .05 * _l.c;	
}

// Function 4
float AmbientOcclusion(in vec3 pos, in vec3 n)
{
    float ao = 0.0;
    float amp = 0.5;
    
    const float step_d = 0.02;
    float distance = step_d;
    
    for (int i = 0; i < 10; i++)
    {
        pos = pos + distance * n;
        ao += amp * clamp(distFunc(pos) / distance, 0.0, 1.0);
        amp *= 0.5;
        distance += step_d;
    }
    
    return ao;
}

// Function 5
float ambientOcclusion(vec3 p, vec3 n)
{
	float stepSize = 0.002f;
	float t = stepSize;
	float oc = 0.0f;
	for(int i = 0; i < 10; ++i)
	{
		vec2 obj = map(p + n * t);
		oc += t - obj.x;
		t += pow(float(i), 2.2) * stepSize;
	}

	return 1.0 - clamp(oc * 0.2, 0.0, 1.0);
}

// Function 6
float ambientOcclusion( in vec3 p, in vec3 n, float maxDist, float falloff )
{
	const int nbIte = 32;
    const float nbIteInv = 1./float(nbIte);
    const float rad = 1.-1.*nbIteInv; //Hemispherical factor (self occlusion correction)
    
	float ao = 0.0;
    
    for( int i=0; i<nbIte; i++ )
    {
        float l = hash(float(i))*maxDist;
        vec3 rd = normalize(n+randomHemisphereDir(n, l )*rad)*l; // mix direction with the normal
        													    // for self occlusion problems!
        
        ao += (l - distf( p + rd )) / pow(1.+l, falloff);
    }
	
    return clamp( 1.-ao*nbIteInv, 0., 1.);
}

// Function 7
float AmbientOcclusion (vec3 p,vec3 n,float d,float s){float r=1.;int t;
  for(int i=0;i<5;++i){if(--s<0.)break;r-=(s*d-(df(p+n*s*d,t)))/pow(2.,s);}return r;}

// Function 8
float ambient_occlusion(vec3 p, vec3 n, int reflection) {
	const int steps = 5;
	float sample_distance = 0.7;
	float occlusion = 0.0;
	for (int i = 1; i <= steps; i++) {
		float k = float(i) / float(steps);
		k *= k;
		float current_radius = sample_distance * k;
		float distance_in_radius = total_distance(p + current_radius * n);
		occlusion += pow(0.5, k * float(steps)) * (current_radius - distance_in_radius);
	}
	float scale = 0.4;
	return 1.0 - clamp(scale * occlusion, 0.0, 1.0 );
}

// Function 9
float ambientOcclusion(vec3 p, vec3 n, float t)
{
    const int steps = 3;
    const float delta = 0.5;

    float a = 0.0;
    float weight = 1.0;
    float m;
    for(int i=1; i<=steps; i++) {
        float d = (float(i) / float(steps)) * delta; 
        a += weight*(d - scene(p + n*d, m, t));
        weight *= 0.5;
    }
    return clamp(1.0 - a, 0.0, 1.0);
}

// Function 10
float ambientOcclusion(vec3 p, vec3 n){
    const int steps = 3;
    const float delta = 0.5;

    float a = 0.0;
    float weight = 0.75;
    float m;
    for(int i=1; i<=steps; i++) {
        float d = (float(i) / float(steps)) * delta; 
        a += weight*(d - scene(p + n*d));
        weight *= 0.5;
    }
    return clamp(1.0 - a, 0.0, 1.0);
}

// Function 11
float evaluateAmbient(vec3 pos, vec3 normal)
{
    vec3 viewDir = normalize(cameraPos - pos);
    float nvDot = dot(normal, viewDir);
    float fresnel = evaluateSchlickFresnel(nvDot);
    float refl = mix(0.03, 1., fresnel);
    return 1. - refl;
}

// Function 12
float WriteAmbientString(in vec2 textCursor, in vec2 uv, in vec2 fragCoord, in float scale, in float ambient)
{START_TEXT AMBIENT_STRING bV+=WriteFloat(tP, uv, 1.1, ambient, true); END_TEXT}

// Function 13
float ambientOcclusion(vec3 p, vec3 n) {
    float step = 8.;
    float ao = 0.;
    float dist;
    for (int i = 1; i <= 3; i++) {
        dist = step * float(i);
		ao += max(0., (dist - map(p + n * dist).y) / dist);  
    }
    return 1. - ao * 0.1;
}

// Function 14
float ambientOcculusion(vec3 pos, vec3 nor) {
	float occ = 0.;
    float sca = 1.;
    for(int i = 0; i < 5; i++) {
    	float h = .01 + .11 * float(i) / 4.;
        vec3 opos = pos + h * nor;
        float d = scene(opos).x;
        occ += (h - d) * sca;
        sca *= .95;
    }
    return clamp(1. - 2. * occ, 0., 1.);
}


```