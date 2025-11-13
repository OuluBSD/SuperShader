# specular_functions

**Category:** lighting
**Type:** extracted

## Dependencies
texture_sampling, normal_mapping, lighting, raymarching

## Tags
lighting, texturing, color

## Code
```glsl
// Reusable Specular Lighting Functions
// Automatically extracted from lighting-related shaders

// Function 1
vec3 get_specular(vec3 p, vec3 n_normal_smooth, vec3 n_direction)
{
    vec3 n_reflection_vector = reflect(n_direction, n_normal_smooth);
    vec2 reflection_uv = get_reflection_uv(p, n_reflection_vector);
    
    // Approximate the reflection:
    vec3 reflection = to_linear_space(texture(iChannel0, reflection_uv * albedo_texture_scale).rgb);

    return(reflection);
}

// Function 2
float specularLight (vec3 p, vec3 n, vec3 r){
    vec3 nr = reflect(r, n);
    return pow(dot(nr, -r), 2.);
}

// Function 3
vec3 specular(const Ray ray, const RaycastHit hit, float shininess) {
    
 	vec3 ret = vec3(0.,0.,0.);
    vec3 l1 = light1 - hit.point;
         
    vec3 ref = reflect(ray.direction, hit.normal);
   	ret += pow(max(dot(normalize(ref),normalize(l1)),0.),shininess);
    
    return ret;
    
}

// Function 4
float Specular(vec3 n,vec3 h,float l,float p){
 return pow(max(.0,dot(n,h))*l,p)*p/32.;}

// Function 5
vec3 specularTerm(Material mat) {
    vec3 irradiatedColor = vec3(.3);
    float specularRoughness = mat.roughness * (1.0 - mat.metalness) + mat.metalness;
    float D = TrowbridgeReitzNDF(mat.NdotH, specularRoughness);
    
    float Cspec0 = 0.02;
    vec3 F = vec3(mix(Cspec0, 1.0, SchlickFresnel(mat.HdotL)));
    float alphaG = pow(specularRoughness * 0.5 + 0.5, 2.0);
    float G = Geometry(mat.NdotL, alphaG) * Geometry(mat.NdotV, alphaG);
    
    return (D * G * F * irradiatedColor) * (1.0 + mat.metalness * mat.albedo) +
                                           irradiatedColor * mat.metalness * mat.albedo;
}

// Function 6
float evaluateGGXSpecularDistribution(float nhDot, highp float roughness)
{
    // Walter et al. 2007, "Microfacet models for refraction through rough surfaces"
    // http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    highp float a = roughness * roughness;
    highp float aa = a * a;
    highp float t = nhDot * nhDot * (aa - 1.) + 1.;
    return aa /
        (t * t + 1.e-20);
}

// Function 7
vec3 SpecularTermSGWarp(in SG light, in vec3 normal, in float roughness, in vec3 view, in vec3 specAlbedo)
{
    SG ndf = DistributionTermSG(normal, roughness);
    SG warpedNDF = WarpDistributionSG(ndf, view);
 
    vec3 result = SGInnerProduct(warpedNDF, light);
 
    vec3 warpDir = warpedNDF.Axis;
    float m2 = roughness * roughness;
    
    // I'm still not sure I understand this, it's obscuring a lot of the fresnel contribution :/
    float nDotL = saturate(dot(normal, warpDir));
    result *= nDotL;
    
    float nDotV = saturate(dot(normal, view));
    vec3 h = normalize(warpedNDF.Axis + view);
 
    result *= GGX_V1(m2, nDotL) * GGX_V1(m2, nDotV);
    
    float powTerm = pow((1.0f - saturate(dot(warpDir, h))), 5.0);
    result *= specAlbedo + (1.0f - specAlbedo) * powTerm;
    
    return max(result, 0.0f);
}

// Function 8
float specular(vec3 n,vec3 l,vec3 e,float s) {    
    float nrm = (s + 8.0) / (3.1415 * 8.0);
    return pow(max(dot(reflect(e,n),l),0.0),s) * nrm;
}

// Function 9
float specular(in vec3 rd, in vec3 norm, in vec3 lightDir, float roughness, float fresnel) {

    float NdotL = dot(norm, lightDir);
    float NdotV = dot(norm, -rd);

    float spe = 0.0;
    if (NdotL > 0.0 && NdotV > 0.0) {

        vec3 h = normalize(-rd + lightDir);

        float NdotH = max(dot(norm, h), 0.0);
        float VdotH = max(dot(-rd, h), 0.000001);
        float LdotH = max(dot(lightDir, h), 0.000001);

        // Beckmann distrib
        float cos2a = NdotH * NdotH;
        float tan2a = (cos2a - 1.0) / cos2a;
        float r = max(roughness, 0.01);
        float r2 = r * r;
        float D = exp(tan2a / r2) / (r2 * cos2a * cos2a);

        // Fresnel term - Schlick approximation
        float F = fresnel + (1.0 - fresnel) * pow(1.0 - VdotH, 5.0);

        // Geometric attenuation term
        float G = min(1.0, (2.0 * NdotH / VdotH) * min(NdotV, NdotL));

        // Cook Torrance
        spe = D * F * G / (4.0 * NdotV * NdotL);
    }

    return spe;
}

// Function 10
vec3 Specular(vec3 l, vec3 n, vec3 eyeVec, vec3 color, float s) {
	return pow(max(dot(normalize(reflect(-eyeVec, n)), l), 0.0), s)*color;
}

// Function 11
bool matIsSpecular( const in float mat ) {
    return mat > 4.5;
}

// Function 12
float Specular_CookTorrance(vec3 N, vec3 V, vec3 L, float Roughness, float AlphaPrim, float F0) {
    // Disney remapping of alpha
    float Alpha = Roughness*Roughness;
	vec3 H = normalize (V + L);

	float NDotL = clamp (dot (N, L), 0.0, 1.0);
	float NDotV = clamp (dot (N, V), 0.0, 1.0);
	float NDotH = clamp (dot (N, H), 0.0, 1.0);
	float LDotH = clamp (dot (L, H), 0.0, 1.0);
    float VDotH = clamp (dot (V, H), 0.0, 1.0);
    
    // NDF : GGX/Trowbridge-Reitz
    #if USE_AREA_LIGHT
    float D = D_GGX_AreaLight(NDotH, Alpha, AlphaPrim);
    #else
    float D = D_GGX(NDotH, Alpha);
    #endif
    
    // Visibility term (G) : Smith with Schlick's approximation
    float G = G_Schlick( NDotL, NDotV, Alpha );
       
    // Fresnel (Schlick)
    float F = F_Schlick(VDotH, F0);


	return (D * F * G);
}

// Function 13
vec4 specular(ray _r, light _l)
{		
	// specular
	vec3 ld = normalize(_l.p - _r.hp);
	vec3 rf = reflect(ld,_r.n);
	float RdotE = max(dot(normalize(rf),normalize(_r.d)),0.);
	
	vec4 spec = RdotE * _l.si * _l.c;
	
	spec = pow(spec,vec4(12.)) * 4.;
	
	return vec4(spec);
}

// Function 14
vec3 f_specular(vec3 wo, vec3 wi)
{
    if(wo.z <= 0.) return vec3(0.,0.,0.);
    if(wi.z <= 0.) return vec3(0.,0.,0.);
    vec3 wh = normalize(wo+wi);
    if(wh.z <= 0.) return vec3(0.,0.,0.);
    // Local masking shadowing
    if (dot(wo, wh) <= 0. || dot(wi, wh) <= 0.) return vec3(0.);
    float wi_dot_wh = clamp(dot(wi,wh),0.,1.);

    float D = ndf_beckmann_anisotropic(wh,0.1, 0.1);
    // V-cavity masking shadowing
    float G1wowh = min(1., 2. * wh.z * wo.z / dot(wo, wh));
    float G1wiwh = min(1., 2. * wh.z * wi.z / dot(wi, wh));
    float G = G1wowh * G1wiwh;
    
	vec3 F  = fresnel_schlick(wi_dot_wh,vec3(1., 1., 1.));
        
    return (D * F * G) / ( 4. * wo.z );
}

// Function 15
float cookTorranceSpecular(
  vec3 lightDirection,
  vec3 viewDirection,
  vec3 surfaceNormal,
  float roughness,
  float fresnel) {

  float VdotN = max(dot(viewDirection, surfaceNormal), 0.01);
  float LdotN = max(dot(lightDirection, surfaceNormal), 0.01);

  //Half angle vector
  vec3 H = normalize(lightDirection + viewDirection);

  //Geometric term
  float NdotH = max(dot(surfaceNormal, H), 0.01);
  float VdotH = max(dot(viewDirection, H), 0.0001);
  float LdotH = max(dot(lightDirection, H), 0.0001);
  float G1 = (2.0 * NdotH * VdotN) / VdotH;
  float G2 = (2.0 * NdotH * LdotN) / LdotH;
  float G = max(0.00001, min(1.0, min(G1, G2)));
  
  //Distribution term
  float D = GGX(roughness, VdotN, LdotN);
  //float D = beckmannDistribution(roughness, NdotH);
  //Fresnel term
  float F = pow(1.0 - VdotN, fresnel);
	//return D;
  //Multiply terms and done
  return  G *F * D;
}

// Function 16
float getSpecular(Surface surface, vec3 lightDir, float diffuse, vec3 cameraPos) {
  	//vec3 lightDir = light.position - surface.position;
  	vec3 ref = reflect(-normalize(lightDir), surface.normal);
  	float specular = 0.;
  	if(diffuse > 0.) {
    	specular = max(0., dot(ref, normalize(cameraPos - surface.normal)));
    	float specularPower = surface.specularPower;
    	specular = pow(specular, specularPower);
  	}
  	return specular;
}


```