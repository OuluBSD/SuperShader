# diffuse

**Category:** lighting
**Type:** extracted

## Dependencies
texture_sampling, normal_mapping, lighting, raymarching

## Tags
lighting, texturing, color

## Code
```glsl
// Reusable Diffuse Lighting Modules

// Module 1
float lightPointDiffuseSoftShadow(vec3 pos, vec3 lightPos, vec3 normal) {
	vec3 lightDir = normalize(lightPos - pos);
	float lightDist = length(lightPos - pos);
	float color = max(dot(normal, lightDir), 0.0) / (lightDist * lightDist);
	if (color > 0.00) color *= castSoftShadowRay(pos, lightPos);
	return max(0.0, color);
}

// Module 2
vec3 diffuse(in sampler2D rng, in sampler2D mask, in sampler2D image, in vec2 uv, in vec2 scale)
{
    const float percent_pixels = 5.0;
    const float threshold_rescale = 100.0/percent_pixels;
    
    /* increase speed by increasing tau (keep it under 0.25 for stability) */
 	const float tau = 0.01;
    vec3 val = texture(mask, uv).xyz;
    vec3 c = texture(image, uv).xyz;
    vec3 w = texture(image, uv + vec2(-1,0) * scale).xyz;
    vec3 e = texture(image, uv + vec2(1,0) * scale).xyz;
    vec3 s = texture(image, uv + vec2(0,-1) * scale).xyz;
    vec3 n = texture(image, uv + vec2(0,+1) * scale).xyz;
    return threshold_rescale*texelFetch(rng, ivec2(uv/scale), 0).x < 0.5 ? val : c + tau*(e+w+n+s - 4.0 * c);   
}

// Module 3
float pdfDiffuse(in vec3 L_local) {
    return INV_PI * L_local.z;
}

// Module 4
vec3 diffuseLightning(vec3 n, vec3 lightDir, vec3 lightColor) {
    float diffuse = dot(n, lightDir);        
    return lightColor * max(0.0, diffuse);
}

// Module 5
float orenNayarDiffuse(
  vec3 ld,
  vec3 vd,
  vec3 sn,
  float r,
  float a) {
  
  float LdotV = dot(ld, vd);
  float NdotL = dot(ld, sn);
  float NdotV = dot(sn, vd);

  float s = LdotV - NdotL * NdotV;
  float t = mix(1., max(NdotL, NdotV), step(.0, s));

  float sigma2 = r * r;
  float A = 1. - .5 * (sigma2/((sigma2 + .33) + .000001));
  float B = .45 * sigma2 / (sigma2 + .09) + .00001;
    
  float ga = dot(vd-sn*NdotV,sn-sn*NdotL);

  return max(0., NdotL) * (A + B * max(0., ga) * sqrt((1.0-NdotV*NdotV)*(1.0-NdotL*NdotL)) / max(NdotL, NdotV));
}

// Module 6
float lightPointDiffuseSpecularShadow(vec3 pos, vec3 lightPos, vec3 cameraPos, vec3 normal) {
	vec3 lightDir = normalize(lightPos - pos);
	float lightDist = length(lightPos - pos);
	float color = dot(normal, lightDir) / square(lightDist);
	if (color > 0.01) {
		vec3 cameraDir = normalize(cameraPos - pos);
		color += dot(cameraDir, lightDir);
		color *= castShadowRay(pos, lightPos, 0.001);
	}

// Module 7
vec4 diffuse(ray _r, light _l)
{		
	// diffuse
	vec3 ld = normalize(_l.p - _r.hp);
	float LdotN = max(dot(ld,_r.n),0.);
	
	vec4 diff = LdotN * _l.di * _l.c;
	
	return vec4(diff);
}

// Module 8
float LTC_EvaluateDiffuse(vec3 P, vec3 N, Object light)
{
	if (IsQuad(light)) {
		vec3 V = light.pos - P;
        vec3 bx = GetQuadBasisX(light);
		vec3 by = GetQuadBasisY(light);
		vec3 corners[4] = vec3[4](
			light.pos - bx + by,
			light.pos + bx + by,
			light.pos + bx - by,
			light.pos - bx - by
		);
		const bool clipToHorizon = true;
		const bool twoSided = false;
		return LTC_Evaluate(P, N, V, mat3(1), corners, clipToHorizon, twoSided);
	}

// Module 9
float diffuse(vec3 P, vec3 N, vec3 lightPos, float lightRad)
{	// based on Seb Lagarde's Siggraph 2014 stuff - https://seblagarde.wordpress.com/
	vec3 vec = lightPos - P;
	float dst = sqrt(dot(vec, vec));
	vec3 dir = vec / dst;
	
	float cosA = dot(N, dir);
	float sinB = lightRad / dst;
	
	if (abs(cosA / sinB) > 1.0) return cosA;
	
	float sinA = length(cross(N, dir));
	float cotA = cosA / sinA;
	
	float cosB = sqrt(1.0 - sinB * sinB);
	float cotB = cosB / sinB;
	
	float x = sqrt(1.0 - cotA * cotA * cotB * cotB) * sinA;
	
	return (acos(-cotA * cotB) * cosA - x * cotB + atan(x / cotB) / (sinB * sinB)) / PI;
}

// Module 10
float Diffuse_Burley( float linearRoughness, float NoV, float NoL, float VoH, float LoH)
{
	float FD90 = 0.5 + 2.0 * VoH * VoH * linearRoughness;
	float FdV = 1.0 + (FD90 - 1.0) * exp2( (-5.55473 * NoV - 6.98316) * NoV );
	float FdL = 1.0 + (FD90 - 1.0) * exp2( (-5.55473 * NoL - 6.98316) * NoL );
	float epicNormalization = 1.0 - linearRoughness * 0.333;
	return FdV * FdL * epicNormalization; // 1/PI * NoL must still be applied
}

// Module 11
vec3 disneyDiffuse(const in float NdotL, const in float NdotV, const in float LdotH, const in MaterialInfo material) {
    
    float FL = schlickWeight(NdotL), FV = schlickWeight(NdotV);
    
    float Fd90 = 0.5 + 2. * LdotH*LdotH * material.roughness;
    float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);
    
    return (1./PI) * Fd * material.baseColor;
}

// Module 12
float GetUniformDiffusePDF(){
    return 0.5 * M_1_PI_F;
}

// Module 13
vec3 CalculateDiffuse(
    in vec3 albedo)
{                              
    return (albedo * ONE_OVER_PI);
}

// Module 14
vec3 DiffuseIrradiance(const vec3 n) 
{
    return max(
          vec3( 0.754554516862612,  0.748542953903366,  0.790921515418539)
        + vec3(-0.083856548007422,  0.092533500963210,  0.322764661032516) * (n.y)
        + vec3( 0.308152705331738,  0.366796330467391,  0.466698181299906) * (n.z)
        + vec3(-0.188884931542396, -0.277402551592231, -0.377844212327557) * (n.x)
        , 0.0);
}

// Module 15
float diffuse( const in vec3 n, const in vec3 l) { 
    return clamp(dot(n,l),0.,1.);
}

// Module 16
float diffuseLight (vec3 p, vec3 n, vec3 r){
    return dot(n, -r);
}

// Module 17
float atm_delta_eddington_Fminus_diffuse( float g, float tau50 )
{
    // Simplified Eddington downwelling flux component
    // for diffuse light input at the top interface,
    // assuming conservative scattering (omega0 = 1),
    // and no bottom reflection.
    float f = g * g;
    g = g / ( g + 1. );
    tau50 = ( 1. - f ) * tau50;
    return 4. / ( 4. + 3. * LN2 * tau50 * ( 1. - g ) );
}

// Module 18
vec3 GetDiffuseColor(float l) {

    bvec4 dist = lessThan(
        vec4(l), 
        vec4(
            GradientColorStep1.a,
            GradientColorStep2.a,
            GradientColorStep3.a,
            GradientColorStep4.a
        ));

    if(dist.x) return GradientColorStep1.xyz;
    else if(dist.y) return mixGrad(GradientColorStep1, GradientColorStep2, l);
    else if (dist.z) return mixGrad(GradientColorStep2, GradientColorStep3, l);
    else if(dist.a) return mixGrad(GradientColorStep3, GradientColorStep4, l);
    else return GradientColorStep4.xyz;
}

// Module 19
vec3 LambertianDiffuse(vec3 albedo)
{
    return albedo * OneOverPi;
}

// Module 20
float GetDiffuse(in int type, in float NdotV, in float NdotL, in float LdotH, in float linearRoughness)
{
    if(type == DISNEY_DIFFUSE)
    {
        return DisneyDiffuse(NdotV, NdotL, LdotH, linearRoughness) / PI;
    }

// Module 21
float wrappedDiffuse(vec3 N, vec3 L, float w, float n) {
	return pow(saturate((dot(N, L)+ w)/ (1.0+ w)), n)* (1.0+ n)/ (2.0* (1.0+ w));
}

// Module 22
float lambertDiffuse(in LightVariables lv)
{
    return lv.LdotN / PI;
}

// Module 23
vec3 Diffuse(in vec3 normal, in vec3 lightVec, in vec3 diffuse)
{
    float nDotL = dot(normal, lightVec);
    return clamp(nDotL * diffuse, 0.0, 1.0);
}

// Module 24
float lightPointDiffuseShadow(vec3 pos, vec3 lightPos, vec3 normal) {
	vec3 lightDir = normalize(lightPos - pos);
	float lightDist = length(lightPos - pos);
	float color = square(dot(normal, lightDir)) / square(lightDist);
	if (color > 0.00) color *= castShadowRay(pos, lightPos, 0.05);
	return max(0.0, color);
}

// Module 25
float WriteDiffuseString(in vec2 textCursor, in vec2 uv, in vec2 fragCoord, in float scale, in int diffuseType)
{
    START_TEXT  
    DIFFUSE_STRING       
    if(diffuseType == LAMBERT_DIFFUSE){LAMBERT_STRING}

// Module 26
float DiffuseBurley(float linearRoughness, float NdotV, float NdotL, float LdotH)
{
    float f90 = 0.5 + 2.0 * linearRoughness * LdotH * LdotH;
    float lightScatter = FresnelSchlick(NdotL, 1.0, f90);
    float viewScatter  = FresnelSchlick(NdotV, 1.0, f90);
    return lightScatter * viewScatter * (1.0 / PI);
}

// Module 27
vec3 
neighbors_diffuse( vec3 hp, vec3 n, vec3 cell_coord )
{

    vec3 rs = sign(n);
    vec3 bounce = vec3(0.);
    bounce += neighbor_light(hp, n, cell_coord + vec3(0.,   0., rs.z));
    bounce += neighbor_light(hp, n, cell_coord + vec3(rs.x, 0., rs.z));
    bounce += neighbor_light(hp, n, cell_coord + vec3(rs.x, 0., 0.));

    // TODO: debranch
    if (abs(n.z) > abs(n.x)) {
        bounce += neighbor_light(hp, n, cell_coord + vec3(-rs.x, 0., rs.z));
    }

// Module 28
vec3 GetDiffuse(RayInfo r)
{
    RayInfo tmpRay = r;
    vec3 surfPos = r.pos;
    
    
    vec3 diffuseCol = vec3(0);
    //accretion disk
    float accretion = blackHoleAccretionDisk;
    
    int mat = rayMat;
    vec3 objVel = objects[rayObj].vel;
    
    vec3 lHalo = vec3(0);
    
    float[numLights] halos;
    
    for (int L=0; L<numLights; L++)
    {
        halos[L] = o_lights[L].haloResult;
    }

// Module 29
vec3 GetDiffuse(Ray r
){vec3 cDiff=vec3(0)
 ;Ray tmpRay = r
 ;float vma2=vma;//accretion disk before recalculation

 ;float mat=float(rayMat)//trippy bug caused by copying/moving this linefurther down.
 ;vec3 objVel = objVel[rayObj]//es100 error , no array of class allowed
 ;vec3 lHalo = vec3(0)
 ;float[numLights] halos//es100 error , no 1st class array
 ;for(int L=0;L<numLights;L++)halos[L]=oliHal[L]
 ;for (int L=0;L<numLights;L++
 ){float lightLate
  ;for (int i=0; i<10; i++
  ){lightLate=cLag*length(oliPos[L]
      -r.b)/cSpe
   ;ProcessLightValue(r.time-lightLate);}

// Module 30
vec3 sampleDiffuseEnvironment(in vec3 Wi)
{
#if (USE_IBL)
    float samples = 0.0;
    
    vec3 u = perpendicularVector(Wi);
	vec3 v = cross(u, Wi);
    vec3 result = vec3(0.0);
    vec3 Wo = Wi;
    for (int i = 0; i < MAX_CUBEMAP_SAMPLES; ++i)
    {
        Wo = randomDirection(Wi, u, v, 1.0, Wo + result);
        float weight = dot(Wi, Wo);
        result += weight * texture(iChannel0, Wo).xyz;
        samples += weight;
	}

// Module 31
vec3 getDiffuseLightColor( vec3 N ) {
    // This is not correct. You need to do a look up in a correctly pre-computed HDR environment map.
    return .25 +pow(textureLod(iChannel0, N, DIFFUSE_LOD).rgb, vec3(3.)) * 1.;
}

// Module 32
float diffuseSphere(vec2 p,vec2 c, float r,vec3 l)
{
    float px = p.x - c.x;
    float py = p.y - c.y;
    float sq = r*r - px*px - py*py;
    if(sq<0.)
    {
    	return 0.;
        //return smoothstep(-.1,0.,sq);
    }

// Module 33
vec3 indirectDiffuseCast(in Ray ray, inout int seed)
{
    vec3 pos;
    vec3 norm;
    Material mat;
    
    //return vec3(nextFloat(seed));
    
    vec3 color = vec3(1.0);
    vec3 emitted = vec3(0);
    Ray traceRay = ray;
    for (int i = 0; i < depth; i++)
    {
        bool intersected = raytraceScene(traceRay, pos, norm, mat);
    
    	if (!intersected) return vec3(0);
    	if (mat.emissive) return emitted;// + mat.albedo * dot(norm, traceRay.direction);
    
    	mat3 coordSys = createCoordinateSystem(norm);
        int s = seed;
    	vec3 smpl = randomSample(coordSys, seed);
    	color *= mat.albedo ;
        
        vec3 pointInSource = vec3(nextFloat(seed), 0.0, nextFloat(seed)) * vec3(2.0*WIDTH,0,2.0*WIDTH) + vec3(-WIDTH,HEIGHT,-WIDTH);
        vec3 L = pointInSource - pos;
        float rr = dot(L, L);
        L = normalize(L);

        Ray shadowRay = Ray(pos + L * BIAS, L);
        if (L.y > BIAS && dot(norm, L) > 0. && !shadowHit(shadowRay)) {
	        const float area = (WIDTH*WIDTH*4.0);
            float weight = area * L.y * dot(norm, L) / (3.14 * rr);
            emitted += color * materials[0].albedo * weight;
        }

// Module 34
float diffuse(vec3 n,vec3 l) { 
    return clamp(dot(n,l),0.,1.);
}

// Module 35
vec3 ray_march_diffuse(vec3 rayOrigin, vec3 rayDir, vec3 diffuse) {
        float distanceTravelled = 0.0;
        const int NUMBER_OF_STEPS = 64;
        const float MINIMUM_HIT_DISTANCE = 0.001;
        const float MAXIMUM_TRACE_DISTANCE = 1000.0;
        const int BOUNCE_AMOUNTS = 3;
        vec3 finalDiffuse = diffuse;
        bool done = false;
        for(int r = 0; r < BOUNCE_AMOUNTS; r++) {
            if (done) {
                break;
            }

// Module 36
float diffuse(vec3 n, vec3 l) {
  return max(dot(n, l), 0.);
}

// Module 37
vec3 Diffuse(vec3 l, vec3 n, int normalized, vec3 color) {
	return (normalized == 1 ? dot(l,n)*0.5+0.5 : max(dot(l,n), 0.0)) * color;
}

// Module 38
float phong_diffuse(vec3 position, vec3 normal, SpotLight light) {
    vec3 dir = normalize(light.position - position);
    return dot(dir, normal);
}

// Module 39
float diffuse(in vec3 nor){
  float diff;
  
  diff = max(0.0, dot(nor, -lig));
  return diff;
}

// Module 40
vec3 getDiffuse(vec3 rayPosition)
{
    vec3 normal = getNormal(rayPosition);
    vec3 dir = normalize(light0 - rayPosition);
    float diffuse = max( dot(dir, normal), 0.0);
    return light0_color * diffuse;
}

// Module 41
vec3 calc_diffuse_term(float dot_nl, float dot_nv, float dot_lh, vec3 base_color, float rough_s)
{
    float fd_90_minus_1 = 2.0 * dot_lh * dot_lh * rough_s - 0.5;
    
    return base_color * c_1_over_pi 
        * (1.0 + fd_90_minus_1 * pow(1.0 - dot_nl, 5.0))
        * (1.0 + fd_90_minus_1 * pow(1.0 - dot_nv, 5.0));
}

// Module 42
vec3 GetDiffuse(Ray r
){Ray tmpRay = r
 ;vec3 surfPos = r.b
 ;float accretion = vma
 ;vec3 diffuseCol = vec3(0)
 ;//accretion disk
 ;int mat = rayMat
 ;vec3 objVel = objVel[rayObj]//es100 error , no array of class allowed
 ;vec3 lHalo = vec3(0)
 ;float[numLights] halos//es100 error , no 1st class array
 ;for(int L=0;L<numLights;L++)halos[L]=oliHal[L]
 ;for (int L=0;L<numLights;L++
 ){float lightLate
  ;for (int i=0; i<10; i++
  ){lightLate=cLag*length(oliPos[L]
      -surfPos)/cSpe
   ;ProcessLightValue(r.time-lightLate);}

// Module 43
vec3 calcDiffuseLight(Object o, Light l, vec3 pos)
{
    vec3 dir = normalize(l.pos - pos);
    return (o.color) * l.intensity * l.color * clamp(dot(o.normal, dir), 0.0, 1.0) * o.difVal;   
}

// Module 44
float DisneyDiffuse_BRDF(float NoV, float NoL, float HoL, float linearRoughness)
{
	float energyBias   = mix(0.0,     0.5 , linearRoughness);
	float energyFactor = mix(1.0, 1.0/1.51, linearRoughness);
    
	float fd90 = energyBias + 2.0 * (HoL*HoL) * linearRoughness;
    
	const float f0 = 1.0;
    
	float lightScatter = FresnelSchlick(NoL, f0, fd90);
	float viewScatter  = FresnelSchlick(NoV, f0, fd90);
	
	return lightScatter * viewScatter * energyFactor * rpi;
}

// Module 45
float diffuse(vec3 normal, vec3 lightVector)
{
    return max(dot(normal, lightVector), 0.0);
}

// Module 46
float orenNayarDiffuse(
  vec3 lightDirection,
  vec3 viewDirection,
  vec3 surfaceNormal,
  float roughness,
  float albedo) {
  
  float LdotV = dot(lightDirection, viewDirection);
  float NdotL = dot(lightDirection, surfaceNormal);
  float NdotV = dot(surfaceNormal, viewDirection);

  float s = LdotV - NdotL * NdotV;
  float t = mix(1.0, max(NdotL, NdotV), step(0.0, s));

  float sigma2 = roughness * roughness;
  float A = 1.0 + sigma2 * (albedo / (sigma2 + 0.13) + 0.5 / (sigma2 + 0.33));
  float B = 0.45 * sigma2 / (sigma2 + 0.09);

  return albedo * max(0.0, NdotL) * (A + B * s / t) / PI;
}

// Module 47
float brdfEvalBrdfDiffuse( in vec3 N, in vec3 L ){
    return clamp( dot( N, L ), 0.0, 1.0 )*INV_PI;
}

// Module 48
vec3 SGDiffuseInnerProduct(in SG lightingLobe, in vec3 normal, in vec3 albedo)
{
    vec3 brdf = albedo / 3.141592;
    return SGIrradianceFitted(lightingLobe, normal) * brdf;
}

// Module 49
vec3 diffuseTerm(Material mat, float lightIntensity) {
    return (mat.albedo / PI * (1.0 - mat.metalness)) * mat.NdotL * lightIntensity;
}

// Module 50
float DisneyDiffuse(in float NdotV, in float NdotL, in float LdotH, in float linearRoughness)
{
    float energyBias = mix(0.0, 0.5, linearRoughness);
    float energyFactor = mix(1.0, 1.0 / 1.51,  linearRoughness);
    float fd90 = energyBias + 2.0 * LdotH * LdotH * linearRoughness;
    
    vec3 f0 = vec3(1.0);
    float lightScatter = SchlickFresnel(NdotL,f0, fd90).x;
    float viewScatter = SchlickFresnel(NdotV, f0, fd90).x;
    return lightScatter * viewScatter * energyFactor;
}

// Module 51
vec3 diffuse(vec3 p, vec2 n)
{	
	vec3 col = vec3(0.0);
	for (float i = 0.0; i < 4.0; i++)
		for (float j = 0.0; j < 8.0; j++)		
		{
			vec2 s = vec2(i, j)+n;
			float u = (rand(p.xy+s)+i)*0.25;
			float v = (rand(p.yz+s)+j)*0.125;
			
			vec3 ns = sampleHemisphere(u*0.5, v, p);
			col += pow(texture(iChannel1, ns).rgb, vec3(2.2));
		}

// Module 52
vec3 f_diffuse(vec3 wo, vec3 wi)
{
    if (wo.z <= 0.)
        return vec3(0., 0., 0.);
    if (wi.z <= 0.)
        return vec3(0., 0., 0.);

    return vec3(0.8, 0., 0.) * IPI * wi.z;
}

// Module 53
float diffuse_white(in ray r_in, in hit_rec rec, out ray r_out, inout seed_t seed) {
    generic_diffuse(r_in, rec, r_out, seed);

    int wl_idx = wavelength_to_idx(r_out.wavelength);

    return 1.0;
}


```