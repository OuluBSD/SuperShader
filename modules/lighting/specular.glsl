// Reusable Specular Lighting Modules

// Module 1
float specular(vec3 normal, vec3 light, vec3 viewdir, float s)
{
	float nrm = (s + 8.0) / (3.1415 * 8.0);
	float k = max(0.0, dot(viewdir, reflect(light, normal)));
    return  pow(k, s);
}

// Module 2
float lightPointDiffuseSpecularShadow(vec3 pos, vec3 lightPos, vec3 cameraPos, vec3 normal) {
	vec3 lightDir = normalize(lightPos - pos);
	float lightDist = length(lightPos - pos);
	float color = dot(normal, lightDir) / square(lightDist);
	if (color > 0.01) {
		vec3 cameraDir = normalize(cameraPos - pos);
		color += dot(cameraDir, lightDir);
		color *= castShadowRay(pos, lightPos, 0.001);
	}

// Module 3
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

// Module 4
float pdfSpecular(in float alphau, in float alphav, in vec3 E_local, in vec3 L_local) {
    vec3 wh = normalize(E_local + L_local);
    return ggx_pdf(E_local, wh, alphau, alphav) / (4.0 * dot(E_local, wh));
}

// Module 5
float specular(vec3 normal, vec3 light, vec3 viewdir, float s)
{
	float nrm = (s + 8.0) / (3.1415 * 8.0);
	float k = max(0.0, dot(viewdir, reflect(light, normal)));
    return pow(k, s);
}

// Module 6
float gaussianSpecular_529295689(
  vec3 lightDirection,
  vec3 viewDirection,
  vec3 surfaceNormal,
  float shininess) {
  vec3 H = normalize(lightDirection + viewDirection);
  float theta = acos(dot(H, surfaceNormal));
  float w = theta / shininess;
  return exp(-w*w);
}

// Module 7
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

// Module 8
float Specular(vec3 n,vec3 h,float l,float p){
 return pow(max(.0,dot(n,h))*l,p)*p/32.;}

// Module 9
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

// Module 10
float RSpecularLight(vec3 normalRay, vec3 normal, float gGgxAlpha, out float o_rDiffuse)
{
	float gGgxAlphaSqr = GSqr(gGgxAlpha);

	vec3 normalHalf = normalize(g_normalLight - normalRay);
	float gDotHalf = saturate(dot(normalHalf, normal));

	float uFresnel = UFresnel(gDotHalf);

	float rSpecular = mix(g_rSpecular, 1.0, uFresnel);

	float gNdf = gGgxAlphaSqr / GSqr(GSqr(gDotHalf) * (gGgxAlphaSqr - 1.0) + 1.0);
	float gVis = 1.0 / (GGgxVisRcp(gGgxAlphaSqr, dot(-normalRay, normal)) *
						GGgxVisRcp(gGgxAlphaSqr, dot(g_normalLight, normal)));

	o_rDiffuse = 1.0 - rSpecular;

#if ENABLE_IS
	return 0.0;
#else
	return gNdf * gVis * rSpecular * g_rSunSpecScale;
#endif
}

// Module 11
float specularLight (vec3 p, vec3 n, vec3 r){
    vec3 nr = reflect(r, n);
    return pow(dot(nr, -r), 2.);
}

// Module 12
float phong_specular(vec3 position, vec3 normal, SpotLight light, Camera camera, float alpha) {
    vec3 light_dir = normalize(light.position - position);
    if (dot(light_dir, normal) < 0.0) return 0.0;
    vec3 camera_dir = normalize(camera.position - position);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float cos_alpha = clamp(dot(camera_dir, reflect_dir), 0.0, 1.0);
    return pow(cos_alpha, alpha);
}

// Module 13
float gaussianSpecular(
  vec3 lightDirection,
  vec3 viewDirection,
  vec3 surfaceNormal,
  float shininess) {
  vec3 H = normalize(lightDirection + viewDirection);
  float theta = acos(dot(H, surfaceNormal));
  float w = theta / shininess;
  return exp(-w*w);
}

// Module 14
float blinnPhongSpecular(
  vec3 lightDirection,
  vec3 viewDirection,
  vec3 surfaceNormal,
  float shininess) {

  //Calculate Blinn-Phong power
  vec3 H = normalize(viewDirection + lightDirection);
  return pow(max(0.0, dot(surfaceNormal, H)), shininess);
}

// Module 15
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

// Module 16
float microfacetSpecular(in LightVariables lv)
{
#if (USE_ANALYTICAL)
    float r2 = lv.roughness * lv.roughness;
    
    float r4 = r2 * r2;
    
    float D = D_GGX(r4, lv.HdotN);
    
    float G = HEIGHT_CORRELATED_SMITH ? 
        G_SmithJointApprox(lv.VdotN, lv.LdotN, r2) : 
    	G_Smith(lv.VdotN, lv.LdotN, r4);
        
    return D * G * lv.LdotN / PI;
#else
    return 0.0;
#endif
}

// Module 17
float getSpecularity(sampler2D tex, vec2 coords)
{
	return texture(tex, coords).b*.5;
}

// Module 18
float specular(vec3 n,vec3 l,vec3 e,float s) {    
    float nrm = (s + 8.0) / (3.1415 * 8.0);
    return pow(max(dot(reflect(e,n),l),0.0),s) * nrm;
}

// Module 19
float Specular(in vec3 reflection, in vec3 lightDirection, float shininess)
{
    return 0.05 * pow(max(0.0, dot(reflection, lightDirection)), shininess);
}

// Module 20
float pdfSpecular(in vec3 E_local, in vec3 L_local, in float alpha) {
    vec3 wh = normalize(E_local + L_local);
    return ggx_pdf(E_local, wh, alpha, alpha) / (4.0 * dot(E_local, wh));
}

// Module 21
float cookTorranceSpecular_1460171947(
  vec3 lightDirection,
  vec3 viewDirection,
  vec3 surfaceNormal,
  float roughness,
  float fresnel) {

  float VdotN = max(dot(viewDirection, surfaceNormal), 0.0);
  float LdotN = max(dot(lightDirection, surfaceNormal), 0.0);

  //Half angle vector
  vec3 H = normalize(lightDirection + viewDirection);

  //Geometric term
  float NdotH = max(dot(surfaceNormal, H), 0.0);
  float VdotH = max(dot(viewDirection, H), 0.000001);
  float LdotH = max(dot(lightDirection, H), 0.000001);
  float G1 = (2.0 * NdotH * VdotN) / VdotH;
  float G2 = (2.0 * NdotH * LdotN) / LdotH;
  float G = min(1.0, min(G1, G2));

  //Distribution term
  float D = beckmannDistribution_2315452051(NdotH, roughness);

  //Fresnel term
  float F = pow(1.0 - VdotN, fresnel);

  //Multiply terms and done
  return  G * F * D / max(3.14159265 * VdotN, 0.000001);
}

// Module 22
float getSpecular(Surface surface, vec3 lightDir, float diffuse, vec3 cameraPos) {
  	//vec3 lightDir = light.position - surface.position;
  	vec3 ref = reflect(-normalize(lightDir), surface.normal);
  	float specular = 0.;
  	if(diffuse > 0.) {
    	specular = max(0., dot(ref, normalize(cameraPos - surface.normal)));
    	float specularPower = surface.specularPower;
    	specular = pow(specular, specularPower);
  	}

// Module 23
float cookTorranceSpecular(
  vec3 lightDirection,
  vec3 viewDirection,
  vec3 surfaceNormal,
  float roughness,
  float fresnel)
{
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
    float G = clamp(min(G1, G2), 0.001, 1.);
  
    //Distribution term
    //float D = GGX(roughness, VdotN, LdotN);
    float D = beckmannDistribution(roughness, NdotH);
    //Fresnel term
    float F = pow(1.0 - VdotN, fresnel);

    //Multiply terms and done
    return  G * F * D / max(3.14159265 * VdotN * LdotN, 0.01);
}

// Module 24
float specular( const in vec3 n, const in vec3 l, const in vec3 e, const in float s) {    
    float nrm = (s + 8.0) / (3.1415 * 8.0);
    return pow(max(dot(reflect(e,n),l),0.0),s) * nrm;
}

// Module 25
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
  float G = min(1.0, min(G1, G2));
  
  //Distribution term
  float D = beckmannDistribution(roughness, NdotH);

  //Fresnel term
  float F = pow(1.0 - VdotN, fresnel);

  //Multiply terms and done
  return  G * F * D / max(3.14159265 * VdotN * LdotN, 0.000001);
}

