// Reusable Ambient Lighting Modules

// Module 1
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

// Module 2
float ambientOcclusion( vec3 p, vec3 n, float maxDist, float falloff )
{
	const int nbIte = 8;
    const float nbIteInv = 1./float(nbIte);
    const float rad = 1.-1.*nbIteInv; //Hemispherical factor (self occlusion correction)
    
	float ao = 0.0;
    
    for( int i=0; i<nbIte; i++ )
    {
        float l = hash(float(i))*maxDist;
        vec3 rd = normalize(n+randomHemisphereDir(n, l )*rad)*l; // mix direction with the normal
        													    // for self occlusion problems!
        
        ao += (l - max(world( p + rd ),0.)) / maxDist * falloff;
    }

// Module 3
float phong_ambient() {
    return 1.0;
}

// Module 4
float ambientOcclusion( in vec3 p, in vec3 n, in float maxDist, in float falloff ){
	const int nbIte = 32;
    const float nbIteInv = 1./float(nbIte);
    const float rad = 1.-1.*nbIteInv; //Hemispherical factor (self occlusion correction)
    
	float ao = 0.0;
    
    for( int i=0; i<nbIte; i++ )
    {
        float l = hash(float(i))*maxDist;
        vec3 rd = normalize(n+randomHemisphereDir(n, l )*rad)*l; // mix direction with the normal
        													    // for self occlusion problems!
        
        ao += (l - max(dist( p + rd ),0.)) / maxDist * falloff;
    }

// Module 5
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
        
        ao += (l - map( p + rd )) / pow(1.+l, falloff);
    }

// Module 6
vec4 ambient(light _l)
{
	return .05 * _l.c;	
}

// Module 7
float calcAmbientOcclusion(VoxelHit hit)
{
    float ambientOcc = 0.0;
    
    // for each of the 28 voxels surrounding the hit voxel
    for (int i=-1; i<=1; i++) for (int j=-1; j<=1; j++) for (int k=-1; k<=1; k++) {
        if (i == 0 && j == 0 && k == 0) continue; // skip the hit voxel
        ivec3 offset = ivec3(i, j, k);
        // TODO: find some way to skip these voxels
        // if (dot(hit.hitRel, vec3(offset)) < 0.0) continue; 
        
        int terrainType; vec2 occlusions;
        getVoxelAndOcclusionsAt(hit.mapPos + offset, terrainType, occlusions);
        if (terrainType != VOXEL_NONE && terrainType != VOXEL_WATER) {
            
            // use the distance from just above the intersection to estimate occlusion
            float dist = dfVoxel(hit.hitRel + hit.hitNormal*0.5 - vec3(offset), terrainType);
            ambientOcc += smoothstep(1.0, 0.0, dist);
        }

// Module 8
float AmbientOcclusion(in vec3 ro, in vec3 n, in vec3 bmin, in vec3 bmax) 
{   
    const float nf = 0.707;
    const vec3 v0 = (vec3(1.0, 1.0, 0.0) * nf) + EPS;
    const vec3 v1 = (vec3(-1.0, 1.0, 0.0) * nf) + EPS;
    const vec3 v2 = (vec3(0.0, 1.0, 1.0) * nf) + EPS;
    const vec3 v3 = (vec3(0.0, 1.0, -1.0) * nf) + EPS;
    
    const vec3 v4 = -v0;
    const vec3 v5 = -v1;
    const vec3 v6 = -v2;
    const vec3 v7 = -v3;
    
    const vec3 invv0 = 1.0/v0;
    const vec3 invv1 = 1.0/v1;
    const vec3 invv2 = 1.0/v2;
    const vec3 invv3 = 1.0/v3;
    const vec3 invv4 = 1.0/v4;
    const vec3 invv5 = 1.0/v5; 
    const vec3 invv6 = 1.0/v6;
    const vec3 invv7 = 1.0/v7;
    vec3 invn = 1.0/(n);
    
    float r = 0.0;
    r += AOFactor(ro, n, n, invn, bmin, bmax);
	r += AOFactor(ro, n, v0, invv0, bmin, bmax);
    r += AOFactor(ro, n, v1, invv1, bmin, bmax);
    r += AOFactor(ro, n, v2, invv2, bmin, bmax);
    r += AOFactor(ro, n, v3, invv3, bmin, bmax);
    r += AOFactor(ro, n, v4, invv4, bmin, bmax);
    r += AOFactor(ro, n, v5, invv5, bmin, bmax);
	r += AOFactor(ro, n, v6, invv6, bmin, bmax);
    r += AOFactor(ro, n, v7, invv7, bmin, bmax);
    
    return clamp(r * 0.2, 0.0, 1.0);
}

// Module 9
float GetAmbientOcclusion(const in C_HitInfo intersection, const in C_Surface surface)
{
    #ifdef ENABLE_AMBIENT_OCCLUSION    
		vec3 vPos = intersection.vPos;
		vec3 vNormal = surface.vNormal;
	
		float fAmbientOcclusion = 1.0;
	
		float fDist = 0.0;
		for(int i=0; i<=5; i++)
		{
			fDist += 0.1;
	
			vec4 vSceneDist = GetDistanceScene(vPos + vNormal * fDist, kNoTransparency);
	
			fAmbientOcclusion *= 1.0 - max(0.0, (fDist - vSceneDist.x) * 0.2 / fDist );                                  
		}

// Module 10
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

// Module 11
float ambientOcclusion(vec3 position, vec3 normal){

	float ao = 0.0;
    //step size
    float del = 0.08;
    float weight = 0.1;
    
    //Travel out from point with fixed step size and accumulate proximity to other surfaces
    //iq slides include `1.0/pow(2.0, i)` factor to reduce the effect of farther objects
    //but Peer Play uses just `1.0/dist`
    int id;
    for(int i = ZERO; i < 5; i++){
        float dist = float(i+1) * del;
    	//Ignore measurements from inside objects
    	ao += max(0.0, (dist - getSDF(position + normal * dist, id))/dist);
    }

// Module 12
float WriteAmbientString(in vec2 textCursor, in vec2 uv, in vec2 fragCoord, in float scale, in float ambient)
{START_TEXT AMBIENT_STRING bV+=WriteFloat(tP, uv, 1.1, ambient, true); END_TEXT}

// Module 13
vec3 ambientLightingBRDF(Material param, BRDFDesc desc)
{
    // Fresnel with Roughness
    float dotNV = dot(desc.normal, desc.viewDir);
    vec3 f0 = 0.04 * (1.0 - param.metallic) + param.diffuseColor.rgb * param.metallic;
    vec3 fTerm = fresnelRoughness(f0, dotNV, param.roughness);

    vec3 kS = fTerm;
    vec3 kD = (1.0 - kS) * (1.0 - param.metallic);
    
    // IBL
    vec3 diffuseIrradiance = textureLod(iChannel2, desc.normal, 5.0).rgb * param.diffuseColor * 0.1 * kD;
    vec3 diffuseIBL = diffuseIrradiance * param.diffuseColor * (1.0 - param.metallic) * desc.aoAttenuation;
    
    float soAttenuation = saturate(pow(dotNV + desc.aoAttenuation, exp2(-16.0 * param.roughness - 1.0)) - 1.0 + desc.aoAttenuation);
    vec3 specularIrradiance = param.reflectionColor;
    vec3 specularIBL = specularIrradiance * fTerm * soAttenuation;
    
    return diffuseIBL + specularIBL;
}

// Module 14
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

// Module 15
float evaluateAmbient(vec3 pos, vec3 normal)
{
    vec3 viewDir = normalize(cameraPos - pos);
    float nvDot = dot(normal, viewDir);
    float fresnel = evaluateSchlickFresnel(nvDot);
    float refl = mix(0.03, 1., fresnel);
    return 1. - refl;
}

// Module 16
float ambientOcclusion(vec3 p, vec3 nor, float k) 
{
    float sum = 0.0;
    for (float i = 0.0; i < 5.0; i++) 
    {
        sum += 1.0 / pow(2.0, i) * (i * 0.15 - sceneMap3D(p + nor * i * 0.15, LIGHT_POS));
    }

// Module 17
float doAmbientOcclusion(in vec2 tcoord,in vec2 uv, in vec3 p, in vec3 cnorm)
{
    vec3 diff = getPosition(tcoord + uv) - p;
    float l = length(diff);
    vec3 v = diff/l;
    float d = l*SCALE;
    float ao = max(0.0,dot(cnorm,v)-BIAS)*(1.0/(1.0+d));
    ao *= smoothstep(MAX_DISTANCE,MAX_DISTANCE * 0.5, l);
    return ao;

}

// Module 18
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

// Module 19
vec3 ambientLight(vec3 pos){
    vec3 oldPos = pos;
    float expectedDist = distanceEstimation(pos) * pow(2.0, float(AoSteps));
    vec3 n, ambientColor = vec3(0.0), gi = vec3(0.0);
    for(int i = 0; i < AoSteps; i++){
        n = normalEstimation(pos);
        pos += distanceEstimation(pos) * n;
        ambientColor += background(n);
        if(i != 0 && fract(float(i)/float(GiSkipSteps)) == 0.0) gi += directLight(pos, n);
    }

// Module 20
float AmbientOcclusion(vec3 p,vec3 n,float d,float s){float r=1.;int t;
  for(int i=0;i<5;++i){if(--s<0.)break;r-=(s*d-(df(p+n*s*d,t)))/pow(2.,s);}

// Module 21
vec3 GetAmbientSkyColor()
{
    return SKY_AMBIENT_MULTIPLIER * GetBaseSkyColor(vec3(0, 1, 0));
}

// Module 22
float ambientOcclusion( in vec3 ro, in vec3 rd )
{
    const int maxSteps = 7;
    const float stepSize = 0.05;
    
    float t = 0.0;
    float res = 0.0;
    
    // starting d
    float d0 = map(ro).x;
    
    for(int i = 0; i < maxSteps; ++i) {
        
        float d = map(ro + rd*t).x;
		float diff = max(d-d0, 0.0);
        
        res += diff;
        
        t += stepSize;
    }

// Module 23
float ambientVisibility(vec3 P, vec3 n) {
    const int   steps    = 3;
    float a = 0.0;
    float weight = 3.0;

    for (int i = 1; i <= steps; ++i) {
        float d = 0.25 * square((float(i) + 0.5) / (0.5 + float(steps)));
        float r = distanceEstimate(lego, P + n * d);

        a += weight * max(d - r, 0.0);
        weight *= 0.5;
    }

// Module 24
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

// Module 25
float ambientOcclusion(vec3 p, vec3 n)
{
	float stepSize = 0.0016f;
	float t = stepSize;
	float oc = 0.0f;
	for(int i = 0; i < 10; ++i)
	{
		vec2 obj = map(p + n * t, true);
		oc += t - obj.x;
		t += pow(float(i), 2.2) * stepSize;
	}

// Module 26
vec3 ambientLight()
{
    return (sampleEnvironment(vec3( 1, 0, 0)) +
			sampleEnvironment(vec3(-1, 0, 0)) +
			sampleEnvironment(vec3( 0, 1, 0)) +
			sampleEnvironment(vec3( 0,-1, 0)) +
			sampleEnvironment(vec3( 0, 0, 1)) +
			sampleEnvironment(vec3( 0, 0,-1))) / 6.0;
}

// Module 27
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

// Module 28
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

// Module 29
vec3 RgbAmbient()
{
	return RgbSky() / g_gPi;
}

// Module 30
float ambientOcclusion(vec3 p, vec3 n) {
    float step = 8.;
    float ao = 0.;
    float dist;
    for (int i = 1; i <= 3; i++) {
        dist = step * float(i);
		ao += max(0., (dist - map(p + n * dist).y) / dist);  
    }

// Module 31
float AmbientOcclusion (vec3 p,vec3 n,float d,float s){float r=1.;
  for(int i=0;i<5;++i){if(--s<0.)break;r-=(s*d-(df(p+n*s*d)))/pow(2.,s);}

// Module 32
float ambientOcclusion(in vec3 rp, in vec3 norm) {
    float sum = 0., s = 0.5;
    vec3 lastp;
    
    for (int i = 0; i < 32; i++) {
        vec3 p = rp+randomHemiRay(norm,lastp,.4)*s;
        sum += max(0., (s-df(p))/(s*s));//randomHemiRay(norm,rp,.5)*s);
        lastp = p;
        s += .2;
    }

// Module 33
float AmbientOcclusion (vec3 p,vec3 n,float d,float s){float r=1.;int t;
  for(int i=0;i<5;++i){if(--s<0.)break;r-=(s*d-(df(p+n*s*d,t)))/pow(2.,s);}

// Module 34
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

// Module 35
vec3 ambientColor(vec2 p) {
    float d = scene(p);
    if(d <= MINDISTANCE) return vec3(0.3, 0.4, 0.5); // objects
    return vec3(0.1); // floor
}

// Module 36
vec3 GetAmbientShadowColor()
{
    return vec3(0, 0, 0.2);
}

// Module 37
float ambientOclussion( in vec3 p, in vec3 n) {
        float step = 0.1;
        float res = 1.0;
        vec4 vstep = vec4( n*step, step );
        vec4 np = vec4(p,0.0) + vstep;
        for (int i=0; i < 5; i++) {
            res -= (np.w - map( np.xyz ).x) * 0.25;
            np += vstep;
        }

// Module 38
float ambientOcclusion(vec3 point, float delta, int samples)
{
	vec3 normal = getNormal(point);
	float occ = 0.;
	for(float i = 1.; i < float(samples); ++i)
	{
		occ += (2.0/i) * (i * delta - distFunc(point + i * delta * normal));
	}

// Module 39
vec3 GetAmbientLight(const in vec3 vNormal)
{
    return GetSkyGradient(vNormal);
}

// Module 40
float AmbientOcclusion(vec3 p,vec3 n,float d,float s){float r=1.;int t;
  for(int i=0;i<5;++i){if(--s<0.)break;r-=(s*d-(gd1(p+n*s*d,t)))/pow(2.,s);}

