# diffuse_functions

**Category:** lighting
**Type:** extracted

## Dependencies
texture_sampling, normal_mapping, lighting

## Tags
lighting, texturing, color

## Code
```glsl
// Reusable Diffuse Lighting Functions
// Automatically extracted from lighting-related shaders

// Function 1
mat3 shDiffuseConvolutionPI(mat3 sh) {
	return PI * shDiffuseConvolution(sh);
}

// Function 2
float diffuse_white(in ray r_in, in hit_rec rec, out ray r_out, inout seed_t seed) {
    generic_diffuse(r_in, rec, r_out, seed);

    int wl_idx = wavelength_to_idx(r_out.wavelength);

    return 1.0;
}

// Function 3
vec3 disneyDiffuse(const in float NdotL, const in float NdotV, const in float LdotH, const in MaterialInfo material) {
    
    float FL = schlickWeight(NdotL), FV = schlickWeight(NdotV);
    
    float Fd90 = 0.5 + 2. * LdotH*LdotH * material.roughness;
    float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);
    
    return (1./PI) * Fd * material.baseColor;
}

// Function 4
vec3 f_diffuse(vec3 wo, vec3 wi)
{
    if (wo.z <= 0.)
        return vec3(0., 0., 0.);
    if (wi.z <= 0.)
        return vec3(0., 0., 0.);

    return vec3(0.8, 0., 0.) * IPI * wi.z;
}

// Function 5
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
	} else
		return 0.0; // not implemented for sphere
}

// Function 6
vec3 SGDiffuseInnerProduct(in SG lightingLobe, in vec3 normal, in vec3 albedo)
{
    vec3 brdf = albedo / 3.141592;
    return SGIrradianceFitted(lightingLobe, normal) * brdf;
}

// Function 7
mat3 shDiffuseConvolution(mat3 sh) {
	mat3 r = sh;
    
    r[0][0] *= convCoeff.x;

    r[0][1] *= convCoeff.y;
    r[0][2] *= convCoeff.y;
    r[1][0] *= convCoeff.y;
    
    r[1][1] *= convCoeff.z;
    r[1][2] *= convCoeff.z;
    r[2][0] *= convCoeff.z;
    r[2][1] *= convCoeff.z;
    r[2][2] *= convCoeff.z;    
    
	return r;
}

// Function 8
vec4 diffuse(ray _r, light _l)
{		
	// diffuse
	vec3 ld = normalize(_l.p - _r.hp);
	float LdotN = max(dot(ld,_r.n),0.);
	
	vec4 diff = LdotN * _l.di * _l.c;
	
	return vec4(diff);
}

// Function 9
float diffuseLight (vec3 p, vec3 n, vec3 r){
    return dot(n, -r);
}

// Function 10
float diffuse(in vec3 nor){
  float diff;
  
  diff = max(0.0, dot(nor, -lig));
  return diff;
}

// Function 11
vec3 DiffuseIrradiance(const vec3 n) 
{
    return max(
          vec3( 0.754554516862612,  0.748542953903366,  0.790921515418539)
        + vec3(-0.083856548007422,  0.092533500963210,  0.322764661032516) * (n.y)
        + vec3( 0.308152705331738,  0.366796330467391,  0.466698181299906) * (n.z)
        + vec3(-0.188884931542396, -0.277402551592231, -0.377844212327557) * (n.x)
        , 0.0);
}

// Function 12
float lightPointDiffuseSoftShadow(vec3 pos, vec3 lightPos, vec3 normal) {
	vec3 lightDir = normalize(lightPos - pos);
	float lightDist = length(lightPos - pos);
	float color = max(dot(normal, lightDir), 0.0) / (lightDist * lightDist);
	if (color > 0.00) color *= castSoftShadowRay(pos, lightPos);
	return max(0.0, color);
}

// Function 13
vec3 Diffuse(vec3 l, vec3 n, int normalized, vec3 color) {
	return (normalized == 1 ? dot(l,n)*0.5+0.5 : max(dot(l,n), 0.0)) * color;
}

// Function 14
float WriteDiffuseString(in vec2 textCursor, in vec2 uv, in vec2 fragCoord, in float scale, in int diffuseType)
{
    START_TEXT  
    DIFFUSE_STRING       
    if(diffuseType == LAMBERT_DIFFUSE){LAMBERT_STRING}
    else if(diffuseType == DISNEY_DIFFUSE){DISNEY_STRING}   
    END_TEXT
}

// Function 15
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

// Function 16
void disneyDiffuseSample(out vec3 wi, const in vec3 wo, out float pdf, const in vec2 u, const in vec3 normal, const in MaterialInfo material) {
    cosineSample
}

// Function 17
float diffuse(vec3 normal, vec3 lightVector)
{
    return max(dot(normal, lightVector), 0.0);
}

// Function 18
float GetDiffuse(in int type, in float NdotV, in float NdotL, in float LdotH, in float linearRoughness)
{
    if(type == DISNEY_DIFFUSE)
    {
        return DisneyDiffuse(NdotV, NdotL, LdotH, linearRoughness) / PI;
    }
    return 1.0 / PI; // otherwise we assume lambert
}

// Function 19
vec3 diffuseTerm(Material mat, float lightIntensity) {
    return (mat.albedo / PI * (1.0 - mat.metalness)) * mat.NdotL * lightIntensity;
}

// Function 20
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
            for(int i = 0; i < NUMBER_OF_STEPS; i++) {
                vec3 currPos = rayOrigin + distanceTravelled * rayDir;
                vec4 sceneData = scene_dist(currPos);
                float sceneDist = sceneData.x;
                vec3 sceneDiffuse = sceneData.yzw;
                if (sceneDist < MINIMUM_HIT_DISTANCE) {
                    float addWeight = pow(0.5, float(r + 1));
                    finalDiffuse = (1.0 - addWeight) * finalDiffuse + addWeight * sceneDiffuse;
                    vec3 normal = calculate_normal(currPos);
                    rayOrigin = currPos;
                    rayDir = reflect(rayDir, normal);
                    break;
                }
                if (sceneDist > MAXIMUM_TRACE_DISTANCE) {
                    done = true;
                    break;
                }
                distanceTravelled += sceneDist;
            }
        }
        finalDiffuse = 0.6 * Sky(rayOrigin, rayDir) + 0.4 * finalDiffuse;
        return finalDiffuse;
    }

// Function 21
int NextDiffuseType(in int currentDiffuseType)
{
    return (currentDiffuseType >= DISNEY_DIFFUSE) ? 0 : (currentDiffuseType + 1);
}

// Function 22
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

// Function 23
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
    else
    {
        bounce += neighbor_light(hp, n, cell_coord + vec3(rs.x, 0., -rs.z));
    }
    
    return bounce;
}

// Function 24
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

// Function 25
bool IsDiffuse(Object obj)
{
#if DIFFUSE_ONLY
    return true; // always!
#else
    return obj.materialType == MATERIAL_TYPE_DIFFUSE;
#endif
}

// Function 26
float DiffuseBurley(float linearRoughness, float NdotV, float NdotL, float LdotH)
{
    float f90 = 0.5 + 2.0 * linearRoughness * LdotH * LdotH;
    float lightScatter = FresnelSchlick(NdotL, 1.0, f90);
    float viewScatter  = FresnelSchlick(NdotV, 1.0, f90);
    return lightScatter * viewScatter * (1.0 / PI);
}

// Function 27
float diffuse(vec3 n,vec3 l) { 
    return clamp(dot(n,l),0.,1.);
}

// Function 28
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
        
        traceRay = Ray(pos + smpl * BIAS, smpl);
    }
    
    return emitted;
}

// Function 29
float GetUniformDiffusePDF(){
    return 0.5 * M_1_PI_F;
}

// Function 30
vec3 LambertianDiffuse(vec3 albedo)
{
    return albedo * OneOverPi;
}

// Function 31
void generic_diffuse(in ray r_in, in hit_rec rec, out ray r_out, inout seed_t seed) {
    r_out.direction = random_cos_weighted_hemisphere_direction(rec.normal, seed);
    r_out.origin = rec.position + T_MIN * r_out.direction;
    r_out.wavelength = r_in.wavelength;
    r_out.ior = r_in.ior;
}


```