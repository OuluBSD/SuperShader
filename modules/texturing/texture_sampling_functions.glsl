// Reusable Texture Sampling Texturing Functions
// Automatically extracted from texturing/mapping-related shaders

// Function 1
vec3 GetTextureOffset(vec2 coords, vec2 textureSize, vec2 texelOffset)
{
    vec2 texelSize = 1.0 / textureSize;
    vec2 offsetCoords = coords + texelSize * texelOffset;
    
    vec2 halfTexelSize = texelSize / 2.0;
    vec2 clampedOffsetCoords = clamp(offsetCoords, halfTexelSize, 1.0 - halfTexelSize);
    
    return texture(iChannel0, clampedOffsetCoords).rgb;
}

// Function 2
void findTextureTargets(out vec2 pA, out vec2 pB)
{	
	vec3 Ctot = vec3(0.);
	pA = vec2(0.); float Atot=0.;
	pB = vec2(0.); float Btot=0.;
	for (int j=0; j< SAMPLE; j++)
	  for (int i=0; i< SAMPLE; i++)
	  {
		  vec2 pos = (.5+vec2(i,j))/float(SAMPLE);
		  vec3 c = texture(iChannel0,pos,LEVEL).rgb;
		  Ctot += c;
		  float v;
		  
		  v = match(c,targetA);
		  pA   += pos*v;
		  Atot += v;
		  
		  v = match(c,targetB);
		  pB   += pos*v;
		  Btot += v;	  
	  }
	pA /= Atot;
	pB /= Btot;
	_ambientI = lum(Ctot)/float(SAMPLE*SAMPLE);
	return;		 
}

// Function 3
float textureSpiral(vec2 uv) {
	float angle = ATAN(uv.y, uv.x),
	shear = length(uv),
	blur = 0.5;
	return smoothstep(-blur, blur, cos_(8.0 * angle + 200.0 * time - 12.0 * shear));
}

// Function 4
float3 Sample_Emitter ( int I, float3 O, inout float seed ) {
  float3 lorig = normalize((Sample_Uniform3(seed)-0.5)*2.0);
  lorig *= float3(0.01, lights[I].radius);
  lorig = inverse(Look_At(lights[I].N))*lorig;
  lorig += lights[I].ori;
  return normalize(lorig - O);
}

// Function 5
float Sample( vec2 pos )
{
	return EvalTestSignal( pos );
	//return texture( iChannel0, pos*0.5 + 0.5 ).x;
}

// Function 6
vec3 sampleImage(vec2 coord){
   return pow3(texture(iChannel0,viewport(coord)).rgb,GAMMA);
}

// Function 7
vec4 SampleTextureCatmullRom(sampler2D sceneTexture, vec2 uv, vec2 texSize, float mipLevel, int getPacked)
{
    vec4 result = vec4(0.0);
    if(getPacked == unpackedNone)
    {
        // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
        // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
        // location [1, 1] in the grid, where [0, 0] is the top left corner.
        vec2 samplePos = uv * texSize;
        vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

        // Compute the fractional offset from our starting texel to our original sample location, which we'll
        // feed into the Catmull-Rom spline function to get our filter weights.
        vec2 f = samplePos - texPos1;

        // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
        // These equations are pre-expanded based on our knowledge of where the texels will be located,
        // which lets us avoid having to evaluate a piece-wise function.
        vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
        vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
        vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
        vec2 w3 = f * f * (-0.5 + 0.5 * f);

        // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
        // simultaneously evaluate the middle 2 samples from the 4x4 grid.
        vec2 w12 = w1 + w2;
        vec2 offset12 = w2 / w12;

        // Compute the final UV coordinates we'll use for sampling the texture
        vec2 texPos0 = texPos1 - vec2(1.0);
        vec2 texPos3 = texPos1 + vec2(2.0);
        vec2 texPos12 = texPos1 + offset12;

        texPos0 /= texSize;
        texPos3 /= texSize;
        texPos12 /= texSize;
        
        result += sampleLevel0(sceneTexture, vec2(texPos0.x,  texPos0.y), mipLevel) * w0.x * w0.y;
        result += sampleLevel0(sceneTexture, vec2(texPos12.x, texPos0.y), mipLevel) * w12.x * w0.y;
        result += sampleLevel0(sceneTexture, vec2(texPos3.x,  texPos0.y), mipLevel) * w3.x * w0.y;

        result += sampleLevel0(sceneTexture, vec2(texPos0.x,  texPos12.y), mipLevel) * w0.x * w12.y;
        result += sampleLevel0(sceneTexture, vec2(texPos12.x, texPos12.y), mipLevel) * w12.x * w12.y;
        result += sampleLevel0(sceneTexture, vec2(texPos3.x,  texPos12.y), mipLevel) * w3.x * w12.y;

        result += sampleLevel0(sceneTexture, vec2(texPos0.x,  texPos3.y), mipLevel) * w0.x * w3.y;
        result += sampleLevel0(sceneTexture, vec2(texPos12.x, texPos3.y), mipLevel) * w12.x * w3.y;
        result += sampleLevel0(sceneTexture, vec2(texPos3.x,  texPos3.y), mipLevel) * w3.x * w3.y;
    }
    
    return result;
}

// Function 8
void sampleCubemap(vec2 p, out float size, out bool inShape){
    vec4 t = T(p,0);
    inShape = t.x>0.5;
    size = t.g;
}

// Function 9
vec3 rotateSample(vec3 sampleDir, float range_01, float circular_01, out float range_angle)
{
    const float PI = 3.14159;    
    float theta = 2.0*PI*circular_01;
    
	vec3 notColinear = (abs(sampleDir.y)<0.8)?vec3(0,1,0):vec3(1,0,0);
	vec3 othogonalAxis = normalize(cross(notColinear,sampleDir));
    
    range_angle = atan( sqrt(range_01)/sqrt(1.0-range_01) );
    float cost = sqrt(1.0-range_01);//=cos(range_angle);
    float sint = sqrt(range_01);//=sin(range_angle);
	mat3 m1 = UTIL_axisRotationMatrix(othogonalAxis, cost, sint);
	mat3 m2 = UTIL_axisRotationMatrix(sampleDir, cos(theta), sin(theta));
    return sampleDir*m1*m2;
}

// Function 10
float sample_ao(vec3 vp, vec3 p, vec3 n)
{
    const float s = 0.5;
    const float i = 1.0 - s;
    vec3 b = vp + n;
    vec3 e0 = n.zxy;
    vec3 e1 = n.yzx;
    float a = 1.0;
    if (voxel(b + e0))
        a *= i + s * sqi(fract(dot(-e0, p)));
    if (voxel(b - e0))
        a *= i + s * sqi(fract(dot(e0, p)));
    if (voxel(b + e1))
        a *= i + s * sqi(fract(dot(-e1, p)));
    if (voxel(b - e1))
        a *= i + s * sqi(fract(dot(e1, p)));
    if (voxel(b + e0 + e1))
        a = min(a, i + s * sqi(min(1.0, length(fract((-e0 - e1) * p)))));
    if (voxel(b + e0 - e1))
        a = min(a, i + s * sqi(min(1.0, length(fract((-e0 + e1) * p)))));
    if (voxel(b - e0 + e1))
        a = min(a, i + s * sqi(min(1.0, length(fract((e0 - e1) * p)))));
    if (voxel(b - e0 - e1))
        a = min(a, i + s * sqi(min(1.0, length(fract((e0 + e1) * p)))));
    return a;
}

// Function 11
vec4 SampleAA(sampler2D sampler, in vec2 uv)
{
    vec4 source = texture(sampler, uv);
    
    // IDs are the integer part of the .w (depth is stored in fractional)
    float sourceID   = floor(source.w);     
    float sourceDiff = 0.0;
    
    vec3 color = vec3(0.0);
    vec2 s = vec2(1.0 / iResolution.x, 1.0 / iResolution.y) * 2.25;
    
    float t = iTime + 0.1;
    
    for (int i = 0; i < SampleSteps; i++)
    {
        for (int j = 0; j < SampleSteps; j++) 
        {
            vec2 q = t * vec2(float(i), float(j));
            vec2 n = noise2(uv , q);
            vec2 offset = vec2(n.x, n.y) - vec2(0.5, 0.5);
            
            vec4 tx = texture(sampler, uv + offset * s);
            color += tx.rgb;
            
            sourceDiff += abs(sourceID - tx.w);
        }
    }
    
    color      /= float(SampleSteps * SampleSteps); 
    sourceDiff /= float(SampleSteps * SampleSteps);
    sourceDiff  = pow(sourceDiff, 8.0);
    
    return vec4(mix(source.rgb, color, clamp(sourceDiff, 0.2, 1.0)), source.w);
}

// Function 12
vec2 sampleFunction (vec2 z, vec2 zMouse) {
  return cmul(cdiv(z - vec2(1, 0), z + vec2(1, 0)), z - zMouse);
}

// Function 13
vec3 Sample_SphLight_MIS2(vec2 s0, vec2 s1, float s2, vec3 V, vec3 p, vec3 N, vec3 albedo, float roughness, vec3 F0)
{
    float alpha = GGXAlphaFromRoughness(roughness);
    
    float ct; vec3 Lc, L0; float sang;
    Sample_SolidAngle(s0, p, LightPos, R2, /*out*/ ct, /*out*/ Lc, /*out*/ L0, /*out*/ sang);
    float pdf00 = 1.0/sang;

    vec3 L1; vec3 f1; float pdf11;
    Sample_GGX_R(s1, V, N, alpha, F0, /*out*/ L1, /*out*/ f1, /*out*/ pdf11);

    bool couldL1HitLight = dot(L1, Lc) > ct;
    
    vec3 f0 = Frostbite_R(V, N, L0, albedo, roughness, F0);
         f1 = Frostbite_R(V, N, L1, albedo, roughness, F0);

    float pdf01 = couldL1HitLight ? pdf00 : 0.0;
    float pdf10 = EvalPDF_GGX_R(V, N, L0, alpha);

    float w0, w1;
    #if 1
    w0 = Pow2(pdf00) / (Pow2(pdf00) + Pow2(pdf10));
    w1 = Pow2(pdf11) / (Pow2(pdf11) + Pow2(pdf01));        
    #elif 1
    w0 = (pdf00) / ((pdf00) + (pdf10));
    w1 = (pdf11) / ((pdf11) + (pdf01)); 
    #else
    w0 = 0.5; 
    w1 = 1.0 - w1;
    #endif

    float wn = couldL1HitLight == false ? 1.0 : w0 / (w0 + w1);

    bool doUseSmpl0 = s2 <= wn;

    float denom = doUseSmpl0 ? pdf00 * wn : pdf11 * (1.0 - wn);

    vec3 L = doUseSmpl0 ? L0 : L1;

    if(dot(N, L) <= 0.0 || denom == 0.0) return vec3(0.0);
    
    float t2; vec3 n2; vec3 a2; bool isLight2 = true;
    bool hit2 = Intersect_Scene(p, L, false, /*out*/ t2, n2, a2, isLight2);

    if(hit2 && isLight2)
    {
        if(doUseSmpl0)
            return f0 / denom * w0 * Radiance;
        else
            return f1 / denom * w1 * Radiance;
    }
}

// Function 14
vec4 edgeSample(vec2 coord)
{
    float t = iTime*0.002;
    return vec4(sample1(coord + vec2(0., t)),  // left
                sample2(coord + vec2(t, 1.)),  // top
                sample1(coord + vec2(1., t)),  // right
                sample2(coord + vec2(t, 0.))); // bottom
}

// Function 15
vec3 textureGround(vec2 uv) {
    const vec2 RES = vec2(16.0, 9.0);    
    float n = hash(floor((uv * RES)));
    n = n * 0.2 + 0.5;
    // return vec3(n*0.4,n*0.2,n*0.8);
    return vec3(n*0.4,n*0.2,n*n); // make blue quadratic
}

// Function 16
vec4 sampleWithNearestNeighbor(sampler2D sampler, vec2 samplerSize, vec2 uv)
{
        vec2 nearestTexelPos = round(samplerSize * uv - vec2(0.5)) + vec2(0.5);
        return texture(sampler, nearestTexelPos / samplerSize);
}

// Function 17
vec3 colorsampler(vec3 src, vec3 col)
{
    vec3 delta = src - col;
    if(dot(delta,delta)<=0.1)
    {
        return vec3(0.0,1.0,0.0);
    }
    else
    {
        return src;
    }
}

// Function 18
vec3 SampleScreen( vec3 vUVW )
{   
    vec3 vAmbientEmissive = vec3(0.1);
    vec3 vBlackEmissive = vec3(0.02);
    float fBrightness = 1.75;
    vec2 vResolution = vec2(480.0f, 576.0f);
    vec2 vPixelCoord = vUVW.xy * vResolution;
    
    vec3 vPixelMatrix = GetPixelMatrix( vPixelCoord );
    float fScanline = GetScanline( vPixelCoord );
      
    vec2 vTextureUV = vUVW.xy;
    //vec2 vTextureUV = vPixelCoord;
    vTextureUV = floor(vTextureUV * vResolution * 2.0) / (vResolution * 2.0f);
    
    Interference interference = GetInterference( vTextureUV );

    float noiseIntensity = 0.1;
    
    //vTextureUV.x += (interference.scanLineRandom * 2.0f - 1.0f) * 0.025f * noiseIntensity;
    
    
    vec3 vPixelEmissive = textureLod( iChannel0, vTextureUV.xy, 0.0 ).rgb;
        
    vPixelEmissive = clamp( vPixelEmissive + (interference.noise - 0.5) * 2.0 * noiseIntensity, 0.0, 1.0 );
    
	vec3 vResult = (vPixelEmissive * vPixelEmissive * fBrightness + vBlackEmissive) * vPixelMatrix * fScanline + vAmbientEmissive;
    
    // TODO: feather edge?
    if( any( greaterThanEqual( vUVW.xy, vec2(1.0) ) ) || any ( lessThan( vUVW.xy, vec2(0.0) ) ) || ( vUVW.z > 0.0 ) )
    {
        return vec3(0.0);
    }
    
    return vResult;
    
}

// Function 19
vec3 sampleReflectionMap(vec3 sp, float lodBias){
    #ifdef LOD_BIAS
    	lodBias = LOD_BIAS;
    #endif
    vec3 color = SRGBtoLINEAR(textureLod(iChannel0, sp, lodBias).rgb);
    #if defined (HDR_FOR_POORS)
    	//color *= 1.0 + 2.0*smoothstep(hdrThreshold, 1.0, dot(LUMA, color)); //HDR for poors
    	color = InvTM(color, hdrThreshold);
   	#endif
    return color;
}

// Function 20
float textureAtlas(vec2 uv, int hitid)
{
    return alphatex(uv, ivec2(hitid, hitid >> 4) & 15);
    // TODO various symmetry modes to extend the available shapes
    // simple extrusions, lathes on various axes, vary orientation, etc.
    // kind of limited in 2D though.
}

// Function 21
void Sample_GGX_R(vec2 s, vec3 V, vec3 N, float alpha, vec3 F0, out vec3 L, out vec3 f, out float pdf)
{
    vec3 H;
    {
    	vec3 ox, oz;
		OrthonormalBasisRH(N, /*out*/ ox, oz);
    	
    	vec3 Vp = vec3(dot(V, ox), dot(V, oz), dot(V, N));
    	
        vec3 Hp = Sample_GGX_VNDF(Vp, alpha, alpha, s.x, s.y);
    	
        H = ox*Hp.x + N*Hp.z + oz*Hp.y;
    }
    
    vec3 F = FresnelSchlick(dot(H, V), F0);

    L = 2.0 * dot(H, V) * H - V;
    
    float NoV = clamp01(dot(N, V));
    float NoL = clamp01(dot(N, L));
    float HoV = clamp01(dot(H, V));
    float NoH = clamp01(dot(N, H));
    
    float G1 = GGX_G(NoV, alpha);
    float G2 = GGX_G(NoV, NoL, alpha);
    float D  = GGX_D(NoH, alpha);
    
    f   = NoV == 0.0 ? vec3(0.0) : (F * G2 * D) * 0.25 / NoV;
    pdf = NoV == 0.0 ?      0.0  : (    G1 * D) * 0.25 / NoV;
}

// Function 22
vec4 textureNearest (sampler2D smp, int smpi, vec2 uv)
{
	vec2 size = textureSizef(smpi);
	return textureLinear(smp, smpi, floor(uv * size) / size);
}

// Function 23
vec4 texture3d (sampler2D t, vec3 p, vec3 n, float scale) {
	return 
		texture(t, p.yz * scale) * abs (n.x) +
		texture(t, p.xz * scale) * abs (n.y) +
		texture(t, p.xy * scale) * abs (n.z);
}

// Function 24
float getSampleDim2(int sampleIndex,vec2 fragCoord)
{
	return fract(getDimensionHash(2,fragCoord) + radicalInverse(sampleIndex, 5));
}

// Function 25
vec4 sampleXYTall(ivec2 XYTall){
    ivec3 XYFace = ivec3(XYTall.x, XYTall.y%1024, XYTall.y/1024);
    return texture(iChannel0, XYFaceToRayDir(XYFace));
}

// Function 26
vec4 texture_Bicubic( sampler2D tex, vec2 t )
{
    vec2 res = iChannelResolution[1].xy;
    vec2 p = res*t - 0.5;
    vec2 f = fract(p);
    vec2 i = floor(p);

    return spline( f.y, spline( f.x, SAM(-1,-1), SAM( 0,-1), SAM( 1,-1), SAM( 2,-1)),
                        spline( f.x, SAM(-1, 0), SAM( 0, 0), SAM( 1, 0), SAM( 2, 0)),
                        spline( f.x, SAM(-1, 1), SAM( 0, 1), SAM( 1, 1), SAM( 2, 1)),
                        spline( f.x, SAM(-1, 2), SAM( 0, 2), SAM( 1, 2), SAM( 2, 2)));
}

// Function 27
void VXAAUpsampleT4x( out vec4 vtex[4], vec4 current, vec4 history, vec4 currN[4], vec4 histN[4] )
{
    vec4 n1[4], n2[4];
    
    n1[VXAA_W] = currN[VXAA_W];
    n1[VXAA_E] = current;
    n1[VXAA_N] = history;
    n1[VXAA_S] = histN[VXAA_S];
    
    n2[VXAA_W] = history;
    n2[VXAA_E] = histN[VXAA_E];
    n2[VXAA_N] = currN[VXAA_N];
    n2[VXAA_S] = current;
    
    
    vec4 weights = vec4( VXAADifferentialBlendWeight( n1 ), VXAADifferentialBlendWeight( n2 ) );
    vtex[VXAA_NW] = history;
    vtex[VXAA_NE] = VXAADifferentialBlend( n2, weights.zw );
    vtex[VXAA_SW] = VXAADifferentialBlend( n1, weights.xy );
    vtex[VXAA_SE] = current;
}

// Function 28
float texture_lum(sampler2D tex, vec2 uv) {
	vec3 rgb = texture(tex, uv).rgb;
    return 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
}

// Function 29
vec3 sampleBSDF(	in vec3 x,
                  	in vec3 ng,
                  	in vec3 ns,
                	in vec3 wi,
                	in float time,
                  	in Material mtl,
                	out vec3 wo,
                	out float brdfPdfW,
                	out vec3 fr,
                	out bool hitRes,
                	out SurfaceHitInfo hit) {
    vec3 Lo = vec3(0.0);
    float Xi1 = rnd();
    float Xi2 = rnd();
    fr = mtlSample(mtl, ng, ns, wi, Xi1, Xi2, wo, brdfPdfW);

    //fr = eval(mtl, ng, ns, wi, wo);

    float dotNWo = dot(wo, ns);
    //Continue if sampled direction is under surface
    if ((dot(fr,fr)>0.0) && (brdfPdfW > EPSILON)) {
        Ray shadowRay = Ray(x + ng*EPSILON, wo, time);

        //abstractLight* pLight = 0;
        float cosAtLight = 1.0;
        float distanceToLight = -1.0;
        vec3 Li = vec3(0.0);

        float distToHit;

        if(raySceneIntersection( shadowRay, EPSILON, false, hit, distToHit )) {
            if(hit.mtl_id_>=LIGHT_ID_BASE) {
                distanceToLight = distToHit;
                cosAtLight = dot(hit.normal_, -wo);
                if(cosAtLight > 0.0) {
                    Li = getRadiance(hit.uv_);
                    //Li = lights[0].color_*lights[0].intensity_;
                }
            } else {
                hitRes = true;
            }
        } else {
            hitRes = false;
            //TODO check for infinite lights
        }

        if (distanceToLight>0.0) {
            if (cosAtLight > 0.0) {
                Lo += ((Li * fr * dotNWo) / brdfPdfW) * misWeight(brdfPdfW, sampleLightSourcePdf(x, ns, wi, distanceToLight, cosAtLight));
            }
        }
    }

    return Lo;
}

// Function 30
vec4 DrawtexturedUVQuad(vec2 a, vec2 b, vec2 c, vec2 d,vec2 uva, vec2 uvb, vec2 uvc, vec2 uvd, float t, vec2 co, sampler2D s){
    float i = DrawQuad(a,b,c,d,t,co);
    if (i<=0.) return vec4(0.);
    vec3 baria = toBari(a,b,c,co);
    vec3 barib = toBari(a,d,c,co);
    vec3 baric = toBari(b,c,d,co);
    vec3 barid = toBari(b,a,d,co);
    vec2 coord = vec2(0.);
    coord+= toCartesian(uvb,uvc,uvd,baric);
    coord+= toCartesian(uvb,uva,uvd,barid);
    coord+= toCartesian(uva,uvb,uvc,baria);
    coord+= toCartesian(uva,uvd,uvc,barib);
    
    return texture(s,coord/4.)*i;
}

// Function 31
vec3 sampleQuad(vec2 uv, float W00, float W01, float W10, float W11) {
    float uu = fu(uv.x, W00, W01, W10, W11);
    float vv = v(uv.y, uu,  W00, W01, W10, W11);
    float u = uu;//uv.x;
    float v = vv;//uv.y;
    float pdf = 4. * (
        (1.-u)*(1.-v)*W00
        + u*(1.-v)*W10
        + u*v*W11
        + (1.-u)*v*W01) / (W00 + W01 + W10 + W11);
    return vec3(uu, vv, pdf);
}

// Function 32
vec3 BSDF_Oren_Nayar_Sample(OrenNayarBsdf bsdf,vec3 Ng,vec3 vDir,float x1, float x2,out vec3 eval,out vec3 wi,out float pdf){
	//pre values
    BSDF_Oren_Nayar_Setup(bsdf);
    wi = sample_uniform_hemisphere(bsdf.nDir, x1, x2, pdf);
	if(dot(Ng, wi) > 0.){
		eval = BSDF_Oren_Nayar_GetIntensity(bsdf, bsdf.nDir, vDir, wi);
	}
	else{
		pdf = 0.;
		eval = vec3(0.);
	}
	return eval;
}

// Function 33
float pomSample( in sampler2D t, in vec2 uv )
{
    float r = texture(t, uv*POMSCALE).r;
    return r*r*VOLHEIGHT;
}

// Function 34
void sampleUniform(
	float u,
	float maxDistance,
	out float dist,
	out float pdf)
{
	dist = u*maxDistance;
	pdf = 1.0/maxDistance;
}

// Function 35
vec3 RandSample(vec2 v) {
    float theta=sqrt(v.x);
    float phi=2.*3.14159*v.y;
    float x=theta*cos(phi);
    float z=theta*sin(phi);
    return vec3(x,z,sqrt(max(0.,1.-v.x)));
}

// Function 36
vec4 SampleCharacterTex( sampler2D sFontSampler, uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vUV = (vec2(iChPos) + vCharUV) / 16.0f;
    return textureLod( sFontSampler, vUV, 0.0 );
}

// Function 37
GBuffer sampleGBuffer(sampler2D tex, ivec2 uv) {
  return unpackGBuffer(texelFetch(tex, uv, 0));
}

// Function 38
vec4 SampleTextureCatmullRom( vec2 uv, vec2 texSize )
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 samplePos = uv * texSize;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
    vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    
    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / w12;

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 texPos0 = texPos1 - vec2(1.0);
    vec2 texPos3 = texPos1 + vec2(2.0);
    vec2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    vec4 result = vec4(0.0);
    result += sampleLevel0( vec2(texPos0.x,  texPos0.y)) * w0.x * w0.y;
    result += sampleLevel0( vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos0.y)) * w3.x * w0.y;

    result += sampleLevel0( vec2(texPos0.x,  texPos12.y)) * w0.x * w12.y;
    result += sampleLevel0( vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos12.y)) * w3.x * w12.y;

    result += sampleLevel0( vec2(texPos0.x,  texPos3.y)) * w0.x * w3.y;
    result += sampleLevel0( vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos3.y)) * w3.x * w3.y;

    return result;
}

// Function 39
mat3 sample_map(vec2 x, float w) {
    float q00 = map(x + vec2(-w,-w));
    float q10 = map(x + vec2( 0,-w));
    float q20 = map(x + vec2( w,-w));
    float q01 = map(x + vec2(-w, 0));
    float q11 = map(x + vec2( 0, 0));
    float q21 = map(x + vec2( w, 0));
    float q02 = map(x + vec2(-w, w));
    float q12 = map(x + vec2( 0, w));
    float q22 = map(x + vec2( w, w));
    return mat3(
		(q00 + q10 + q01 + q11) / 4.0, // 00
        (q11 + q10) / 2.0,  // 10
        (q10 + q20 + q11 + q21) / 4.0, // 20
        (q11 + q01) / 2.0, // 01
        q11, // 11
        (q11 + q21) / 2.0, // 21
        (q01 + q11 + q02 + q12) / 4.0, // 02
        (q11 + q12) / 2.0, // 12
        (q11 + q21 + q12 + q22) / 4.0 // 22
    );// * w;
}

// Function 40
vec4 sample_blured(vec2 uv, float radius, float gamma)
{
    vec4 pix = vec4(0.);
    float norm = 0.001;
    //weighted integration over mipmap levels
    for(float i = 0.; i < 10.; i += 0.5)
    {
        float k = weight(i, log2(1. + radius), gamma);
        pix += k*texture(iChannel0, uv, i); 
        norm += k;
    }
    //nomalize
    return pix/norm;
}

// Function 41
vec3 sampleIndirectLight(vec3 pos,vec3 normal
){vec3 dir
 ;vec3 abso=vec3(1.),light=vec3(0.),dc,ec
 ;for(int i=0;i<Bounces;i++
 ){dir=getCosineWeightedSample(normal)
  ;if(!trace(pos,dir,normal))return light+abso*background(dir)
  ;sdf(pos,dc,ec)
  ;light+=abso*(ec+dc*directLight(pos,normal))
  ;abso*=dc;}
 ;return light;}

// Function 42
vec4 textureBlocky(in sampler2D tex, in vec2 uv, in vec2 res) {
    uv *= res; // enter texel coordinate space.
    
    
    vec2 seam = floor(uv+.5); // find the nearest seam between texels.
    
    // here's where the magic happens. scale up the distance to the seam so that all
    // interpolation happens in a one-pixel-wide space.
    uv = (uv-seam)/v2len(dFdx(uv),dFdy(uv))+seam;
    
    uv = clamp(uv, seam-.5, seam+.5); // clamp to the center of a texel.
    
    
    return texture(tex, uv/res, -1000.).xxxx; // convert back to 0..1 coordinate space.
}

// Function 43
vec3 Sample_PointLight(vec3 V, vec3 p, vec3 N, vec3 albedo, float roughness, vec3 F0)
{
    float alpha = GGXAlphaFromRoughness(roughness);
    
    vec3 pl = LightPos;
    vec3 vecl = pl - p;
    vec3 L = normalize(vecl);
    float d2 = dot(vecl, vecl);

    float t2; vec3 n2; vec3 a2; bool hitLight2 = false;
    bool hit = Intersect_Scene(p, L, false, /*out*/ t2, n2, a2, hitLight2);

    if(hit && t2*t2 < d2) return vec3(0.0);
        
    float att = 1.0 / d2;

    return Frostbite_R(V, N, L, albedo, roughness, F0) * att * Intensity;
}

// Function 44
float getLavaStoneTexture(vec2 uv, float f) {
    return getFlowSpots(uv, f);
}

// Function 45
vec3 importanceSampleGGX(vec2 randomHemisphere, vec3 N, float roughness){
    float a = roughness*roughness;
	
    float phi = 2.0 * PI * randomHemisphere.x;
    float cosTheta = sqrt((1.0 - randomHemisphere.y) / (1.0 + (a*a - 1.0) * randomHemisphere.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    //From spherical coordinates to cartesian coordinates
    vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
	
    //From tangent-space vector to world-space sample vector
    vec3 tangent;
    vec3 bitangent;
    
    pixarONB(N, tangent, bitangent);
    
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);
	
    vec3 sampleDir = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleDir);
}

// Function 46
float shadow_sample (vec3 org, vec3 dir) {
    float res = 1.0;
    float t = epsilon * 200.0;
    for (int i =0; i < 100; ++i){
        float h = get_distance (org + dir*t).x;
		if (h <= epsilon) {
            return 0.0;
		}
        res = min (res, 32.0 * h / t);
        t += h;
		if (t >= max_distance) {
      		return res;
		}
		
    }
    return res;
}

// Function 47
void sampleSphericalLight( in vec3 x, in Sphere sphere, float Xi1, float Xi2, out LightSamplingRecord sampleRec ) {
#ifdef SAMPLE_LIGHT_AREA
    vec3 n = randomDirection( Xi1, Xi2 );
    vec3 p = sphere.pos + n*sphere.radius;
    float pdfA = 1.0/sphere.area;
    
    vec3 Wi = p - x;
    
    float d2 = dot(Wi,Wi);
    sampleRec.d = sqrt(d2);
    sampleRec.w = Wi/sampleRec.d; 
    float cosTheta = max( 0.0, dot(n, -sampleRec.w) );
    sampleRec.pdf = PdfAtoW( pdfA, d2, cosTheta );
#else
    vec3 w = sphere.pos - x;	//direction to light center
	float dc_2 = dot(w, w);		//squared distance to light center
    float dc = sqrt(dc_2);		//distance to light center
    
    if( dc_2 > sphere.radiusSq ) {
    	float sin_theta_max_2 = sphere.radiusSq / dc_2;
		float cos_theta_max = sqrt( 1.0 - clamp( sin_theta_max_2, 0.0, 1.0 ) );
    	float cos_theta = mix( cos_theta_max, 1.0, Xi1 );
        float sin_theta_2 = 1.0 - cos_theta*cos_theta;
    	float sin_theta = sqrt(sin_theta_2);
        sampleRec.w = uniformDirectionWithinCone( w, TWO_PI*Xi2, sin_theta, cos_theta );
    	sampleRec.pdf = 1.0/( TWO_PI * (1.0 - cos_theta_max) );
        
        //Calculate intersection distance
		//http://ompf2.com/viewtopic.php?f=3&t=1914
        sampleRec.d = dc*cos_theta - sqrt(sphere.radiusSq - dc_2*sin_theta_2);
    } else {
        sampleRec.w = randomDirection( Xi1, Xi2 );
        sampleRec.pdf = 1.0/FOUR_PI;
    	raySphereIntersection( Ray(x,sampleRec.w), sphere, sampleRec.d );
    }
#endif
}

// Function 48
vec4 drawTextureValues(in vec2 uv, in float mode, in int menu, in vec2 tvals)
{
    vec4 tcol;
    float lsize = 0.015;
    vec2 start = vec2(0.016, 0.18);
    
    int mval = getIValue(tvals.x, TEXTURES_NUM);
    int sval = getIValue(tvals.y, TEXTURES_NUM);
    
    if (in_zone(mode, APPL_UI) && menu == MENU_OPT_TEXTURE)
    {
        tcol += drawTextHorizontal(uv, start, lsize, vec2[10](_M, _a, _i, _n, _COLON, _X, _X, _X, _X, _X), 5);
        tcol += drawTextHorizontal(uv, start - vec2(0., 0.03), lsize, _SECONDARY, 10);
        
        vec2 mstart = start + 6.*vec2(lsize, 0.);
        vec2 ststart = start + 6.*vec2(lsize, 0.) - vec2(0., 0.03);
        
        if (mval == DIFFUSE_MAP)
        {
            tcol += drawTextHorizontal(uv, mstart, lsize, _DIFFUSEMAP, 9);
        }
        else if (mval == HEIGHT_MAP)
        {
            tcol += drawTextHorizontal(uv, mstart, lsize, _HEIGHTMAP, 10);
        }
        else if (mval == SPECULAR_MAP)
        {
            tcol += drawTextHorizontal(uv, mstart, lsize, _SPECULAR, 8);
        }
        
        if (sval == DIFFUSE_MAP)
        {
            tcol += drawTextHorizontal(uv, ststart, lsize, _DIFFUSEMAP, 9);
        }
        else if (sval == HEIGHT_MAP)
        {
            tcol += drawTextHorizontal(uv, ststart, lsize, _HEIGHTMAP, 10);
        }
        else if (sval == SPECULAR_MAP)
        {
            tcol += drawTextHorizontal(uv, ststart, lsize, _SPECULAR, 8);
        }
    }
    
    return tcol;
}

// Function 49
vec3 SampleEnvironment(in vec3 reflVec)
{
    reflVec = normalize(reflVec);
    return texture(iChannel3, reflVec).rgb;
}

// Function 50
vec3 sampleTexture( in vec3 uvw, in vec3 nor, in float mid )
{
    return mytexture( uvw, nor, mid );
}

// Function 51
vec4 texture3( sampler3D sam, vec3 uv )
{
#ifndef SHOW_DERIVATIVES    
    return texture(sam,uv);
#else    
    float res = float(textureSize(sam,0).x);
    uv = uv*res - 0.5;
    vec3 iuv = floor(uv);
    vec3 f = fract(uv);
	vec4 rg1 = textureLod( sam, (iuv+ vec3(0.5,0.5,0.5))/res, 0.0 );
	vec4 rg2 = textureLod( sam, (iuv+ vec3(1.5,0.5,0.5))/res, 0.0 );
	vec4 rg3 = textureLod( sam, (iuv+ vec3(0.5,1.5,0.5))/res, 0.0 );
	vec4 rg4 = textureLod( sam, (iuv+ vec3(1.5,1.5,0.5))/res, 0.0 );
	vec4 rg5 = textureLod( sam, (iuv+ vec3(0.5,0.5,1.5))/res, 0.0 );
	vec4 rg6 = textureLod( sam, (iuv+ vec3(1.5,0.5,1.5))/res, 0.0 );
	vec4 rg7 = textureLod( sam, (iuv+ vec3(0.5,1.5,1.5))/res, 0.0 );
	vec4 rg8 = textureLod( sam, (iuv+ vec3(1.5,1.5,1.5))/res, 0.0 );
	return mix(mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y ),
               mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y ), f.z );
#endif    
}

// Function 52
vec4 previousSample(vec4 hit//reprojection
){vec2 prevUv=pos2uv(getCam(iTime-iTimeDelta,iMouse,iR),hit.xyz)
 ;vec2 prevu=prevUv*iR.y+iR.xy/2.
 ;vec2 pfc,pfcf
 ;float dist,f=MaxDist//dist finaldist
 ;for(int x=-1;x<=1;x++
 ){for(int y=-1;y<=1;y++
  ){pfc=prevu+PixelCheckDistance*vec2(x,y)
   ;dist=distancePixel(pfc,hit)
   ;if(dist<f
   ){pfcf=pfc;f=dist;}}}
 ;Camera cam=getCam(iTime,iMouse,iR)
 ;if(f<PixelAcceptance*length(hit.xyz-cam.pos)/cam.focalLength/iR.y
 ){return texture(iChannel0,pfcf/iR.xy);}
 ;return vec4(0.);}

// Function 53
float sample_dist_local_bilateral(vec2 uv, float font_size) {
    
    const int nstep = 4;  
    const float spos  = 0.95;
    const float sdist = 5e-3;
    const float k_ctr = 0.25;
    
    float bump = float((nstep + 1) % 2)*0.5;
    const int ngrid = nstep*nstep;
        
    ivec2 st = ivec2(floor(uv*TEX_RES + bump));
    vec2 uv0 = (vec2(st) + 0.5)/TEX_RES;
    vec2 duv0 = uv - uv0;
    
    float dists[ngrid];
    float wpos[ngrid];
    
    float dctr = 0.0;
    
    for (int i=0; i<nstep; ++i) {
        int di = i - nstep/2;
        for (int j=0; j<nstep; ++j) {            
            int dj = j - nstep/2;
            
            vec3 grad_dist = fetch_grad_dist(st + ivec2(di, dj));
            
            vec2 uvdelta = duv0 - vec2(di, dj) / TEX_RES;                        
            
            vec2 tdelta = uvdelta * TEX_RES;           
            
            vec2 pdelta = uvdelta * GLYPHS_PER_UV;            
            
            float dline = grad_dist.z + dot(grad_dist.xy, pdelta);
            
            vec2 w = max(vec2(0.0), 1.0 - abs(tdelta));

            dctr += w.x*w.y*mix(grad_dist.z, dline, k_ctr);
                        
            int idx = nstep*i + j;
            dists[idx] = dline;
            wpos[idx] = dot(tdelta, tdelta);
            
        }
    }                
    
    float dsum = 0.0;
    float wsum = 0.0;
    
    for (int i=0; i<nstep; ++i) {
        for (int j=0; j<nstep; ++j) {
            int idx = nstep*i + j;
            float ddist = dists[idx] - dctr;
            float wij = exp(-wpos[idx]/(2.0*spos*spos) + 
                            -ddist*ddist/(2.0*sdist*sdist));
            dsum += wij * dists[idx];
            wsum += wij;
        }
    }
        
    return font_size*dsum/wsum;
    
}

// Function 54
float perlinTexture(in vec2 pos)
{
    const float N = 5.0;
    
    vec2 pos2 = pos;
    
    float ind = 0.0;
    float ampl = 1.0;
    
    for(float n = 0.0; n < N; n ++)
    {
    	ind += texture(iChannel0, pos / iChannelResolution[0].x).x * ampl;
    
        ampl *= 0.5;
        pos *= 2.0;
    }
    
    return ind / (1.0 - pow(0.5, N+1.0)) * 0.5;
}

// Function 55
vec3 sample_grad_dist(vec2 uv, float font_size) {
    
    vec3 grad_dist = (textureLod(iChannel0, uv, 0.).yzw - TEX_BIAS) * font_size;

    grad_dist.y = -grad_dist.y;
    grad_dist.xy = normalize(grad_dist.xy + 1e-5);
    
    return grad_dist;
    
}

// Function 56
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness){
    float a = roughness*roughness;
	
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    // from spherical coordinates to cartesian coordinates
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
	
    // from tangent-space vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
	
    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}

// Function 57
vec3 texturef( in vec2 p )
{
    return texture( iChannel0, p ).xyz;
}

// Function 58
void textureWall(in vec3 block, inout ray ray, inout rayMarchHit hit, inout vec3 colour, bool isReflection, in float time) {
    vec2 uv;
    vec3 absNormal = abs(hit.surfaceNormal);
    float scale = 2.0;

    if(absNormal.y-0.8 > absNormal.x && absNormal.y-0.8 > absNormal.z) {
        uv.xy=fract(hit.origin.xz*scale)-0.5;
    } else {
        if(absNormal.x > absNormal.z) {
            if(hit.surfaceNormal.x>0.0) {
                uv.x=1.0-fract((hit.origin.z)*scale);
            } else {
                uv.x=fract((hit.origin.z)*scale);
            }
        } else {
            if(hit.surfaceNormal.z>0.0) {
                uv.x=fract((hit.origin.x)*scale);
            } else {
                uv.x=1.0-fract((hit.origin.x)*scale);
            }
        }
        uv.y=fract(hit.origin.y*scale);
		uv-=0.5;
        //vec2 windowSize=vec2(1.0);
        vec2 windowSize=vec2(hash21(block.xy*39.195),hash21(block.xy*26.389))*0.7+0.2;

        if (windowSize.x > 0.8){
            windowSize.x=1.0;
        }

        if (windowSize.y > 0.8){
            windowSize.y=1.0;
        }

        float round=0.0;

        if (windowSize.x < 1.0 && windowSize.y < 1.0) {
            round = min(windowSize.x,windowSize.y) * hash21(block.xy*87.981);
        }

       if ( abs(uv.x*2.0) < windowSize.x+PARALLAX_WINDOW_SURROUND_THICKNESS && abs(uv.y*2.0) < windowSize.y+PARALLAX_WINDOW_SURROUND_THICKNESS) {
            float distance = sdBox(uv*2.0,windowSize-round)-round;
            if(distance < 0.0) {
                vec3 cell = floor(hit.origin*scale);
                bool on = (hash31(cell) + sin(time*0.5)*0.05) > 0.5;
                hit.materialId = MAT_WINDOW;
                if(on) {
                    float brightness = clamp(hash31(cell),0.1,1.0);
                    vec3 lightColour = clamp(hash33(cell)+0.5,0.0,1.0);
                    
                    #if defined(DEBUG_PARALLAX)
                        if(isReflection) {
                            colour=lightColour*3.0*brightness;
                        } else {
                            float distanceRatio=hit.distance/RAY_MAX_DISTANCE;
                            if(distanceRatio<0.5) {
                                textureParallaxWindow(block, ray, hit, uv, cell, lightColour, brightness, colour, time);
                                colour*=3.0;
                                if(distanceRatio>0.25) {
                                    colour=mix(colour,lightColour*3.0*brightness,(distanceRatio-0.25)*4.0);
                                }
                                //shade the edge of the glass a bit.
                                colour = mix(PARALLAX_WINDOW_SURROUND_COLOUR,colour,clamp(abs(distance*20.0),0.0,1.0));
                            } else {
                                colour=lightColour*3.0*brightness;
                            }
                        }
                    #else
                    	colour=lightColour*3.0*brightness;
                    #endif
                } else {
                    colour=vec3(0.0);
                }
            } else if(distance < PARALLAX_WINDOW_SURROUND_THICKNESS) {
                hit.materialId = MAT_WINDOW;
                colour=PARALLAX_WINDOW_SURROUND_COLOUR;
            } 
        }
    }

    if (hit.materialId != MAT_WINDOW){
        float concrete = getConcrete(hit.origin, hit.surfaceNormal, false);
        colour = hash33(block.xyx) * vec3(0.25,0.1,0.2) + 0.5;
        colour = clamp(colour,vec3(0.0),vec3(1.0));
        colour *= concrete;
    }
}

// Function 59
vec3 mytexture( vec3 p, vec3 n, float matid )
{
	p += 0.1;
	vec3 ip  = floor(p/20.0);
	vec3 fp  = fract(0.5+p/20.0);

	float id = fract(sin(dot(ip,vec3(127.1,311.7, 74.7)))*58.5453123);
	id = mix( id, 0.3, matid );
	
	float f = mod( ip.x + mod(ip.y + mod(ip.z, 2.0), 2.0), 2.0 );
	
	float g = 0.5 + 1.0*noise( p * mix( vec3(0.2+0.8*f,1.0,1.0-0.8*f), vec3(1.0), matid) );
	
	g *= mix( smoothstep( 0.03, 0.04, abs(fp.x-0.5)/0.5 )*
	          smoothstep( 0.03, 0.04, abs(fp.z-0.5)/0.5 ),
			  1.0,
			  matid );
	
	vec3 col = 0.5 + 0.5*sin( 1.0 + 2.0*id + vec3(0.0,1.0,2.0) );
	
	return col * g;
}

// Function 60
vec4 AntiAliasPointSampleTexture_None(vec2 uv, vec2 texsize) {	
	return texture(iChannel0, (floor(uv+0.5)+0.5) / texsize, -99999.0);
}

// Function 61
void sampleSphereUniform(vec3 viewer, in Sphere sphere, inout SurfaceLightSample sls){
    vec2 ksi = rand2();
    float polar = acos(1.0f - 2.0f * ksi.x);
    float azimuth = ksi.y * 2.0f * PI;
    sls.normal = vec3(sin(polar) *cos(azimuth), sin(polar) * sin(azimuth), cos(polar));
    sls.point = sphere.position + (sphere.radius ) * sls.normal;
    sls.pdf = 1.0f/ (FOUR_PI * sphere.radius2);
}

// Function 62
vec4 barkTexture(vec3 p, vec3 nor)
{
    vec2 r = floor(p.xz / 5.0) * 0.02;
    
    float br = texture(iChannel1, r).x;
	vec3 mat = texCube(iChannel3, p, nor) * vec3(.35, .25, .25);
    mat += texCube(iChannel3, p*1.73, nor)*smoothstep(0.0,.2, mat.y)*br * vec3(1,.9,.8);
    //mat*=mat*2.5;
   	return vec4(mat, .1);
}

// Function 63
vec3 BicubicLagrangeTextureSample (vec2 P)
{
    vec2 pixel = P * c_textureSize + 0.5;
    
    vec2 frac = fract(pixel);
    pixel = floor(pixel) / c_textureSize - vec2(c_onePixel/2.0);
    
    vec3 C00 = texture(iChannel0, pixel + vec2(-c_onePixel ,-c_onePixel)).rgb;
    vec3 C10 = texture(iChannel0, pixel + vec2( 0.0        ,-c_onePixel)).rgb;
    vec3 C20 = texture(iChannel0, pixel + vec2( c_onePixel ,-c_onePixel)).rgb;
    vec3 C30 = texture(iChannel0, pixel + vec2( c_twoPixels,-c_onePixel)).rgb;
    
    vec3 C01 = texture(iChannel0, pixel + vec2(-c_onePixel , 0.0)).rgb;
    vec3 C11 = texture(iChannel0, pixel + vec2( 0.0        , 0.0)).rgb;
    vec3 C21 = texture(iChannel0, pixel + vec2( c_onePixel , 0.0)).rgb;
    vec3 C31 = texture(iChannel0, pixel + vec2( c_twoPixels, 0.0)).rgb;    
    
    vec3 C02 = texture(iChannel0, pixel + vec2(-c_onePixel , c_onePixel)).rgb;
    vec3 C12 = texture(iChannel0, pixel + vec2( 0.0        , c_onePixel)).rgb;
    vec3 C22 = texture(iChannel0, pixel + vec2( c_onePixel , c_onePixel)).rgb;
    vec3 C32 = texture(iChannel0, pixel + vec2( c_twoPixels, c_onePixel)).rgb;    
    
    vec3 C03 = texture(iChannel0, pixel + vec2(-c_onePixel , c_twoPixels)).rgb;
    vec3 C13 = texture(iChannel0, pixel + vec2( 0.0        , c_twoPixels)).rgb;
    vec3 C23 = texture(iChannel0, pixel + vec2( c_onePixel , c_twoPixels)).rgb;
    vec3 C33 = texture(iChannel0, pixel + vec2( c_twoPixels, c_twoPixels)).rgb;    
    
    vec3 CP0X = CubicLagrange(C00, C10, C20, C30, frac.x);
    vec3 CP1X = CubicLagrange(C01, C11, C21, C31, frac.x);
    vec3 CP2X = CubicLagrange(C02, C12, C22, C32, frac.x);
    vec3 CP3X = CubicLagrange(C03, C13, C23, C33, frac.x);
    
    return CubicLagrange(CP0X, CP1X, CP2X, CP3X, frac.y);
}

// Function 64
vec3 sample_light(vec3 x, vec3 n, vec3 rd)
{
    vec3 Lo = vec3(0.);
    
    for(int i = 0; i < LIGHT_SAMPLES; ++i)
    {
        vec3 Li, wi;
        light_pick(n, Li, wi);
        
        float cos_theta = max(0., dot(n, wi));
        
        if(cos_theta > 0.00001)
        {
        	float sha = shadow(x, wi);
        	Lo += Li * brdf(wi, -rd, n, x) * cos_theta * sha;
        }
    }
    
    Lo /= float(LIGHT_SAMPLES);
    
    return Lo;
}

// Function 65
float Sample(sampler2D channel, vec2 uv, vec2 texelCount)
{
    vec2 uv0 = uv;
    
    uv -= vec2(0.5);
    
    vec2 uvi = floor(uv);
    vec2 uvf = uv - uvi;

    vec2 mo = uvf - uvf*uvf;
    
   #if 0
    mo = (mo * -0.5 + 1.0) * mo;// use this if it improves quality
   #endif
    
    uvf = (uvf - mo) / (1.0 - 2.0 * mo);// map modulator to s-curve

    uv = uvi + uvf + vec2(0.5);

    vec4 v = textureLod(channel, uv / texelCount, 0.0);
    
    mo *= fract(uvi * 0.5) * 4.0 - 1.0;// flip modulator bump on every 2nd interval
    
    return dot(v, vec4(mo.xy, mo.x*mo.y, 1.0));
}

// Function 66
vec3 sampleColor(vec2 uv)
{
    //float mouseU = 2. * PI * 2. * (iMouse.x / iResolution.x - 0.5);
    //float mouseV = PI * (iMouse.y / iResolution.y);
    float mouseU = 2. * PI * 2. * (texture(iChannel1, vec2(0.75)).x - 0.5);
    float mouseV = PI/2. * texture(iChannel1, vec2(0.75)).y;
    
    float texU = 2. * (texture(iChannel1, vec2(0.25)).x - 0.5);
    float texV = 2. * (texture(iChannel1, vec2(0.25)).y - 0.5);
    vec3 trash;
    
    vec3 cam = vec3(0,0,4);
    vec3 screenPos = vec3(uv, -0.5);
    
    pR(cam.yz, mouseV);
    pR(screenPos.yz, mouseV);
    
    pR(cam.xz, mouseU);
    pR(screenPos.xz, mouseU);
    
    vec3 ray = normalize(screenPos);
    
    vec3 norm;
    float d = trace(cam, ray, norm);
    vec3 pt = cam + ray * d;
    
    if (d > inf - 1.)
    {
        return vec3(0);;
    }
    
    vec3 ray2 = normalize(pt - vec3(0,2,0));
    d = trace(pt, ray2, trash);
    vec3 q = pt + ray2 * d;
    if (d > inf - 1.)
        q = pt;
    
    vec3 albedo = vec3(1);
    
    // colored pattern
    if (xor(mod(q.x, 2.0) < 1.0, mod(q.z, 2.0) < 1.0))
        albedo = color(0.4);
    else
        albedo = color(0.3);
    if (texture(iChannel0, vec2(-q.x, q.z) / 16.).x > 0.5)
        albedo = color(0.2);
    
    // minor axes
    albedo = mix(color(0.1), albedo, pow(smoothstep(0., 0.03, absCircular(q.x)), 3.) );
    albedo = mix(color(0.1), albedo, pow(smoothstep(0., 0.03, absCircular(q.z)), 3.) );
    
    // main axes
    albedo = mix(color(0.0), albedo, pow(smoothstep(0.0, 0.08, abs(q.x)), 3.0));
    albedo = mix(color(0.0), albedo, pow(smoothstep(0.0, 0.08, abs(q.z)), 3.0));
    
    // the line!
    {
        float a = texU;
        float b = -1.;
        float d = 4. * texV;
        
        vec3 e1 = vec3(-d/a, 0, d/b);
        vec3 e2 = vec3(0, 2, d/b);
        vec3 n = normalize(cross(e1, e2));
        
        d = dot(n, pt) - n.y * 2.;
        albedo = mix(color(0.99), albedo, pow(smoothstep(0.0, 0.03, abs(d)), 10.0));
        
    }
    
    //--------------------------------------------------
    // Lighting Time
    //--------------------------------------------------
    float ambient = 0.1;
    float ao = smoothstep(0., 1.4, length(q)); // cheapest AO in history
    
    // Lighting
    vec3 light = 2. * vec3(0., 1., 1.);
    vec3 lightDir = light - pt;
    float lightIntensity = 5.0;
    
    // soft shadows
    float shadow = 1.;
    if (pt.y < 0.1)
    {
        vec3 nlightDir = normalize(lightDir);
        float totalL = dot(nlightDir, vec3(0,1,0) - pt);
    
        vec3 closestToCenter = nlightDir * totalL;
        vec3 dO = (vec3(0,1,0) - pt) - closestToCenter;
        float O = length(dO);
        shadow = smoothstep(1., 1.1, O);
    }
    
    
    float illum = shadow * lightIntensity * max(0., dot(norm, normalize(lightDir) )) / length(lightDir);
    
    // bad lighting. don't do this at home, kids!
    illum = illum / (illum + 1.);
    vec3 final = illum * albedo + ambient * ao * albedo;
    
    return final;
}

// Function 67
float sampleNormalDistribution(float U, float mu, float sigma)
{
    float x = sigma * 1.414213f * erfinv(2.0f * U - 1.0f) + mu;
    return x;
}

// Function 68
float sampleBunny(float3 uvs)
{
    float3 voxelUvs = max(float3(0.0),min(uvs*float3(BUNNY_VOLUME_SIZE), float3(BUNNY_VOLUME_SIZE)-1.0));
    uint3 intCoord = uint3(voxelUvs);
    uint arrayCoord = intCoord.x + intCoord.z*uint(BUNNY_VOLUME_SIZE);
	
    // Very simple clamp to edge. It would be better to do it for each texture sample
    // before the filtering but that would be more expenssive...
    // Also adding small offset to catch cube intersection floating point error
    if(uvs.x<-0.001 || uvs.y<-0.001 || uvs.z<-0.001 ||
      uvs.x>1.001 || uvs.y>1.001 || uvs.z>1.001)
    	return 0.0;
   
    // 1 to use nearest instead
#if VOLUME_FILTERING_NEAREST
    // sample the uint representing a packed volume data of 32 voxel (1 or 0)
    uint bunnyDepthData = packedBunny[arrayCoord];
    float voxel = (bunnyDepthData & (1u<<intCoord.y)) > 0u ? 1.0 : 0.0;
#else
    uint3 intCoord2 = min(intCoord+uint3(1), uint3(BUNNY_VOLUME_SIZE-1));
    
    uint arrayCoord00 = intCoord.x  + intCoord.z *uint(BUNNY_VOLUME_SIZE);
    uint arrayCoord01 = intCoord.x  + intCoord2.z*uint(BUNNY_VOLUME_SIZE);
    uint arrayCoord10 = intCoord2.x + intCoord.z *uint(BUNNY_VOLUME_SIZE);
    uint arrayCoord11 = intCoord2.x + intCoord2.z*uint(BUNNY_VOLUME_SIZE);
    
    uint bunnyDepthData00 = packedBunny[arrayCoord00];
    uint bunnyDepthData01 = packedBunny[arrayCoord01];
    uint bunnyDepthData10 = packedBunny[arrayCoord10];
    uint bunnyDepthData11 = packedBunny[arrayCoord11];
        
    float voxel000 = (bunnyDepthData00 & (1u<<intCoord.y)) > 0u ? 1.0 : 0.0;
    float voxel001 = (bunnyDepthData01 & (1u<<intCoord.y)) > 0u ? 1.0 : 0.0;
    float voxel010 = (bunnyDepthData10 & (1u<<intCoord.y)) > 0u ? 1.0 : 0.0;
    float voxel011 = (bunnyDepthData11 & (1u<<intCoord.y)) > 0u ? 1.0 : 0.0;
    float voxel100 = (bunnyDepthData00 & (1u<<intCoord2.y)) > 0u ? 1.0 : 0.0;
    float voxel101 = (bunnyDepthData01 & (1u<<intCoord2.y)) > 0u ? 1.0 : 0.0;
    float voxel110 = (bunnyDepthData10 & (1u<<intCoord2.y)) > 0u ? 1.0 : 0.0;
    float voxel111 = (bunnyDepthData11 & (1u<<intCoord2.y)) > 0u ? 1.0 : 0.0;
    
    float3 d = voxelUvs - float3(intCoord);
    
    voxel000 = mix(voxel000,voxel100, d.y);
    voxel001 = mix(voxel001,voxel101, d.y);
    voxel010 = mix(voxel010,voxel110, d.y);
    voxel011 = mix(voxel011,voxel111, d.y);
    
    voxel000 = mix(voxel000,voxel010, d.x);
    voxel001 = mix(voxel001,voxel011, d.x);
    
    float voxel = mix(voxel000,voxel001, d.z);
#endif
    
    return voxel;
}

// Function 69
vec3 cosineSampleHemisphere(const in vec2 u) {
    vec2 d = concentricSampleDisk(u);
    float z = sqrt(max(EPSILON, 1. - d.x * d.x - d.y * d.y));
    return vec3(d.x, d.y, z);
}

// Function 70
float sample_dist_gaussian(vec2 uv) {

    float dsum = 0.;
    float wsum = 0.;
    
    const int nstep = 3;
    
    const float w[3] = float[3](1., 2., 1.);
    
    for (int i=0; i<nstep; ++i) {
        for (int j=0; j<nstep; ++j) {
            
            vec2 delta = vec2(float(i-1), float(j-1))/TEX_RES;
            
            float dist = textureLod(iChannel0, uv-delta, 0.).w - TEX_BIAS;
            float wij = w[i]*w[j];
            
            dsum += wij * dist;
            wsum += wij;

        }
    }
    
    return dsum / wsum;
}

// Function 71
vec3 sampleHemisphereCosWeighted( in float Xi1, in float Xi2 ) {
    float theta = acos(clamp(sqrt(1.0-Xi1),-1.0, 1.0));
    float phi = TWO_PI * Xi2;
    return sph2cart( 1.0, phi, theta );
}

// Function 72
vec3 DirectLightSample(in Intersection intersecNow,out vec3 wi,out float pdf){
	vec3 Li = vec3(0.);
    float x1 = GetRandom(),x2 = GetRandom();
    float dist = INFINITY;
    vec3 AssumeLi = LightSample(intersecNow.surface,x1,x2,wi,dist,pdf);
    Ray shadowRay = Ray(intersecNow.surface,wi);
    Intersection intersecNext;
    SceneIntersect(shadowRay, intersecNext);
    if(intersecNext.type == LIGHT){
    	Li = AssumeLi;
    }
    return Li;
}

// Function 73
vec3 cosine_weighted_hemi_sample(inout float seed)
{
    vec2 p = uniform_disk_sample(seed);
    return normalize(vec3(sin(p.x) * p.y, cos(p.x) * p.y, sqrt(1. - p.y * p.y)));
}

// Function 74
vec3 BicubicHermiteTextureSample (vec2 P)
{
    vec2 pixel = P * c_textureSize + 0.5;
    
    vec2 frac = fract(pixel);
    pixel = floor(pixel) / c_textureSize - vec2(c_onePixel/2.0);
    
    vec3 C00 = texture(iChannel0, pixel + vec2(-c_onePixel ,-c_onePixel)).rgb;
    vec3 C10 = texture(iChannel0, pixel + vec2( 0.0        ,-c_onePixel)).rgb;
    vec3 C20 = texture(iChannel0, pixel + vec2( c_onePixel ,-c_onePixel)).rgb;
    vec3 C30 = texture(iChannel0, pixel + vec2( c_twoPixels,-c_onePixel)).rgb;
    
    vec3 C01 = texture(iChannel0, pixel + vec2(-c_onePixel , 0.0)).rgb;
    vec3 C11 = texture(iChannel0, pixel + vec2( 0.0        , 0.0)).rgb;
    vec3 C21 = texture(iChannel0, pixel + vec2( c_onePixel , 0.0)).rgb;
    vec3 C31 = texture(iChannel0, pixel + vec2( c_twoPixels, 0.0)).rgb;    
    
    vec3 C02 = texture(iChannel0, pixel + vec2(-c_onePixel , c_onePixel)).rgb;
    vec3 C12 = texture(iChannel0, pixel + vec2( 0.0        , c_onePixel)).rgb;
    vec3 C22 = texture(iChannel0, pixel + vec2( c_onePixel , c_onePixel)).rgb;
    vec3 C32 = texture(iChannel0, pixel + vec2( c_twoPixels, c_onePixel)).rgb;    
    
    vec3 C03 = texture(iChannel0, pixel + vec2(-c_onePixel , c_twoPixels)).rgb;
    vec3 C13 = texture(iChannel0, pixel + vec2( 0.0        , c_twoPixels)).rgb;
    vec3 C23 = texture(iChannel0, pixel + vec2( c_onePixel , c_twoPixels)).rgb;
    vec3 C33 = texture(iChannel0, pixel + vec2( c_twoPixels, c_twoPixels)).rgb;    
    
    vec3 CP0X = CubicHermite(C00, C10, C20, C30, frac.x);
    vec3 CP1X = CubicHermite(C01, C11, C21, C31, frac.x);
    vec3 CP2X = CubicHermite(C02, C12, C22, C32, frac.x);
    vec3 CP3X = CubicHermite(C03, C13, C23, C33, frac.x);
    
    return CubicHermite(CP0X, CP1X, CP2X, CP3X, frac.y);
}

// Function 75
vec3 sampleIndirectLight(vec3 pos, vec3 normal){
    vec3 dir = getCosineWeightedSample(normal);
    vec3 light = vec3(0.);
    for(int i = 0; i < Bounces; i++){
        if(!trace(pos, dir, normal)) return light+background(dir);
        else light += directLight(pos, normal);
    }
    return light;
}

// Function 76
float sample_world(vec3 world_position) {

    float body = 999.0;
    vec3 co = world_position;
    //co.z *= 0.5;
    
    float scale = 0.2;
    mat4 m = mat4(
		vec4(0.6373087, -0.0796581,  0.7664804, 0.0),
  		vec4(0.2670984,  0.9558195, -0.1227499, 0.0),
  		vec4(-0.7228389,  0.2829553,  0.6304286, 0.0),
        vec4(0.1, 0.6, 0.2, 0.0)
    );
    //mat4 m = mat4(1.0);
    
    for (int i=0; i<3; i++) {
        co = (m * vec4(co, float(i))).xyz;
        scale *= (3.0);
        
        float field = distance_field(co * scale) / scale;
     	body = smin(body, field, 0.05);
    }
    
    return -body;
}

// Function 77
void TextureEnvBlured(in vec3 N, in vec3 Rv, out vec3 iblDiffuse, out vec3 iblSpecular) {
    iblDiffuse = vec3(0.0);
    iblSpecular = vec3(0.0);

    vec2 sum = vec2(0.0);

    vec2 ts = vec2(textureSize(reflectTex, 0));
    float maxMipMap = log2(max(ts.x, ts.y));

    vec2 lodBias = vec2(maxMipMap - 7.0, 4.0);

    for (int i=0; i < ENV_SMPL_NUM; ++i) {
        vec3 sp = SpherePoints_GoldenAngle(float(i), float(ENV_SMPL_NUM));

        vec2 w = vec2(
            dot(sp, N ) * 0.5 + 0.5,
            dot(sp, Rv) * 0.5 + 0.5);


        w = pow(w, vec2(4.0, 32.0));

        vec3 iblD = sampleReflectionMap(sp, lodBias.x);
        vec3 iblS = sampleReflectionMap(sp, lodBias.y);

        iblDiffuse  += iblD * w.x;
        iblSpecular += iblS * w.y;

        sum += w;
    }

    iblDiffuse  /= sum.x;
    iblSpecular /= sum.y;
}

// Function 78
vec4 textureNoTile_4weights( sampler2D samp, in vec2 uv )
{
    // compute per-tile integral and fractional uvs.
    // flip uvs for 'odd' tiles to make sure tex samples are coherent
    vec2 fuv = mod( uv, 2. ), iuv = uv - fuv;
    vec3 BL_one = vec3(0.,0.,1.); // xy = bot left coords, z = 1
    if( fuv.x >= 1. ) fuv.x = 2.-fuv.x, BL_one.x = 2.;
    if( fuv.y >= 1. ) fuv.y = 2.-fuv.y, BL_one.y = 2.;
    
    // smoothstep for fun and to limit blend overlap
    vec2 b = smoothstep(0.25,0.75,fuv);
    
    // fetch and blend
    vec4 res = mix(
        		mix( texture( samp, transformUVs( iuv + BL_one.xy, uv ) ), 
                     texture( samp, transformUVs( iuv + BL_one.zy, uv ) ), b.x ), 
                mix( texture( samp, transformUVs( iuv + BL_one.xz, uv ) ),
                     texture( samp, transformUVs( iuv + BL_one.zz, uv ) ), b.x),
        		b.y );

    // debug vis: colour based on num samples taken for vis purposes - always takes 4 samples!
    if( iMouse.z > 0. ) res.gb *= .25;
    
    return res;
}

// Function 79
vec3 PBR_nudgeSample(vec3 sampleDir, float roughness, float e1, float e2, out float range)
{
    const float PI = 3.14159;
    //Importance sampling :
    //Source : http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
    //The higher the roughness, the broader the range.
    //In any case, wide angles are less probable than narrow angles.
    range = atan( roughness*sqrt(e1)/sqrt(1.0-e1) );
    //Circular angle has an even distribution (could be improved?).
	float phi = 2.0*PI*e2;
    
	vec3 up = vec3(0,1,0); //arbitrary
	vec3 tAxis = cross(up,sampleDir);
	mat3 m1 = UTIL_axisRotationMatrix(normalize(tAxis),range);
	mat3 m2 = UTIL_axisRotationMatrix(normalize(sampleDir), phi);
        
	return sampleDir*m1*m2;
}

// Function 80
vec3 Sample_SphLight_ClmpCos(vec3 V, vec3 p, vec3 N, inout uint h, vec3 albedo, float roughness, vec3 F0)
{
    float alpha = GGXAlphaFromRoughness(roughness);
    
    vec3 L;
    {
        float h0 = Hash11(h);
        float h1 = Hash01(h);

        L = Sample_ClampedCosineLobe(h0, h1, N);
    }

    float t2; vec3 n2; vec3 a2; bool isLight2 = true;
    bool hit = Intersect_Scene(p, L, false, /*out*/ t2, n2, a2, isLight2);

    if(!isLight2) return vec3(0.0);
    
    return Frostbite_R(V, N, L, albedo, roughness, F0) * Radiance * pi;
}

// Function 81
vec3 Sample_Hemisphere(float s0, float s1, vec3 normal)
{
    vec3 smpl = Sample_Sphere(s0, s1);

    if(dot(smpl, normal) < 0.0)
        return -smpl;
    else
        return smpl;
}

// Function 82
float SampleDigit(const in float fDigit, const in vec2 vUV)
{
	const float x0 = 0.0 / FONT_RATIO.x;
	const float x1 = 1.0 / FONT_RATIO.x;
	const float x2 = 2.0 / FONT_RATIO.x;
	const float x3 = 3.0 / FONT_RATIO.x;
	const float x4 = 4.0 / FONT_RATIO.x;
	
	const float y0 = 0.0 / FONT_RATIO.y;
	const float y1 = 1.0 / FONT_RATIO.y;
	const float y2 = 2.0 / FONT_RATIO.y;
	const float y3 = 3.0 / FONT_RATIO.y;
	const float y4 = 4.0 / FONT_RATIO.y;
	const float y5 = 5.0 / FONT_RATIO.y;

	// In this version each digit is made of up to 3 rectangles which we XOR together to get the result
	
	vec4 vRect0 = vec4(0.0);
	vec4 vRect1 = vec4(0.0);
	vec4 vRect2 = vec4(0.0);
		
	if(fDigit < 0.5) // 0
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x1, y1, x2, y4);
	}
	else if(fDigit < 1.5) // 1
	{
		vRect0 = vec4(x1, y0, x2, y5); vRect1 = vec4(x0, y0, x0, y0);
	}
	else if(fDigit < 2.5) // 2
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x0, y3, x2, y4); vRect2 = vec4(x1, y1, x3, y2);
	}
	else if(fDigit < 3.5) // 3
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x0, y3, x2, y4); vRect2 = vec4(x0, y1, x2, y2);
	}
	else if(fDigit < 4.5) // 4
	{
		vRect0 = vec4(x0, y1, x2, y5); vRect1 = vec4(x1, y2, x2, y5); vRect2 = vec4(x2, y0, x3, y3);
	}
	else if(fDigit < 5.5) // 5
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x1, y3, x3, y4); vRect2 = vec4(x0, y1, x2, y2);
	}
	else if(fDigit < 6.5) // 6
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x1, y3, x3, y4); vRect2 = vec4(x1, y1, x2, y2);
	}
	else if(fDigit < 7.5) // 7
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x0, y0, x2, y4);
	}
	else if(fDigit < 8.5) // 8
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x1, y1, x2, y2); vRect2 = vec4(x1, y3, x2, y4);
	}
	else if(fDigit < 9.5) // 9
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x1, y3, x2, y4); vRect2 = vec4(x0, y1, x2, y2);
	}
	else if(fDigit < 10.5) // '.'
	{
		vRect0 = vec4(x1, y0, x2, y1);
	}
	else if(fDigit < 11.5) // '-'
	{
		vRect0 = vec4(x0, y2, x3, y3);
	}	
	
	float fResult = InRect(vUV, vRect0) + InRect(vUV, vRect1) + InRect(vUV, vRect2);
	
	return mod(fResult, 2.0);
}

// Function 83
vec4 textureQuadratic( in sampler2D sam, in vec2 p )
{
    float texSize = float(textureSize(sam,0).x); 
    
#if 1
    //Roger/iq style
	p = p*texSize;
	vec2 i = floor(p);
	vec2 f = fract(p);
	p = i + f*0.5;
	p = p/texSize;
    //f = f*f*(3.0-2.0*f); // optional for extra sweet
	float w = 0.5/texSize;
	return mix(mix(texture2(sam,p+vec2(0,0)),
                   texture2(sam,p+vec2(w,0)),f.x),
               mix(texture2(sam,p+vec2(0,w)),
                   texture2(sam,p+vec2(w,w)),f.x), f.y);
    
#else
    // paniq style (https://www.shadertoy.com/view/wtXXDl)
    vec2 f = fract(p*texSize);
    vec2 c = (f*(f-1.0)+0.5) / texSize;
    vec2 w0 = p - c;
    vec2 w1 = p + c;
    return (texture(sam, vec2(w0.x, w0.y))+
    	    texture(sam, vec2(w0.x, w1.y))+
    	    texture(sam, vec2(w1.x, w1.y))+
    	    texture(sam, vec2(w1.x, w0.y)))/4.0;
#endif    

    
}

// Function 84
vec3 GetSampleColor(vec2 uv
){Ray r
 ;r.dir = vec3(0,0,1)
 ;if (fishEye
 ){vec3 crossv=cross(r.dir,vec3(uv,0))
  ;r.dir=qr(aa2q(length(uv)*FOV,normalize(crossv)),r.dir)
  ;}else r.dir = vec3(uv.xy*FOV,1.)
 ;//apply look dir
 ;r.b = objPos[oCam]//es100 error , no array of class allowed
 ;r.dir = qr(objRot[oCam],r.dir)//es100 error , no array of class allowed
 ;MarchPOV(r,playerTime)
 ;return GetDiffuse(r);}

// Function 85
float sample2(vec2 coord)
{
    float value = texture(iChannel0, coord.xy/iChannelResolution[0].xy).x;
    return fract(value * 789.1798684);
}

// Function 86
vec3 CreateTexture(in RayHit hit)
{
    // Create toned down diffuse textures to account for later gamma correction
    vec3 grass = GrassColor - (texture(iChannel0, hit.surfPos.xz * 0.1).r * 0.3);
    vec3 cliff = CliffColor - (abs(sin(texture(iChannel1, hit.surfPos.xy * 0.01).r)) * 0.5);
    
    vec3 color = mix(cliff, grass, SteepnessRatio(hit.steepness)) * 0.2;
    
    return TimeLerp(vec3(0.2), color, TIME_SteepnessB, TIME_Texture);
}

// Function 87
vec3 SampleDiffuse( const in vec3 vDir )
{
	vec3 vSample = textureLod(iChannel1, vDir, 0.0).rgb;
	vSample = vSample * vSample;
	
	// hack bright spots out of blurred environment
	float fMag = length(vSample);	
	vSample = mix(vSample, vec3(0.15, 0.06, 0.03), smoothstep(0.1, 0.25, fMag));
	
	return vSample * fSceneIntensity;
}

// Function 88
int SampleMaterial(vec3 p)
{
    // We only have one material
    return MATERIAL_CRYSTAL;
}

// Function 89
float SampleBackbuffer( vec2 vCoord )
{
    if ( any( greaterThanEqual( vCoord, vFlameResolution ) ) )
    {
        return 0.0;
    }

    if ( vCoord.x < 0.0 )
    {
        return 0.0;
    }
    
	return clamp( texture(iChannel0, vCoord / iResolution.xy).r, 0.0, 1.0 );
}

// Function 90
void textureSolid(in vec3 block, inout ray ray, inout rayMarchHit hit, inout vec3 colour, bool isReflection, in float time) {
    float concrete = getConcrete(hit.origin, hit.surfaceNormal, true);
    colour = hash33(block.xyx) * vec3(0.25,0.1,0.2) + 0.5;
    colour = clamp(colour,vec3(0.0),vec3(1.0));
    colour *= concrete;
}

// Function 91
vec3 texture_wood2(vec3 p)
{
    p /= 2.;
    
    vec3 p0 = p;
    
    
    // Old trick to mix things up. I use it too much. 
    p = sin(p*4.3 + cos(p.yzx*6.7));

    
    float n = dot(p + sin(p*13.)*.03, vec3(3.));
    
    float grain = 1.-abs(dot(sin(p0*120.5 + n*6.283 + sin(p0.zxy*121.3)), vec3(.333)));

    
    // Smooth fract. Old trick. Like fract, but smoother.
    n = fract(n + fract(n*4.)*.1);
    n = min(n, n*(1.-n)*6.); // The final term controls the smoothing.
    
    float w = min(n*.85 + grain*.2, 1.);
    
    // Quick coloring. Needs work. 
    return mix(vec3(.5, .15, .025), vec3(.75, .3, .1)*2., w*.75 + .25)*(w*.6 + .4);
    
    //return vec3(w);
}

// Function 92
vec3 sampleBlackBorders(vec2 uv)
{
    if (length(clamp(uv, vec2(0.0, 0.0), vec2(1.0, 1.0)) - uv) == 0.0)
    {
        return texture(iChannel0, uv).xyz;
    }
    else
    {
        return vec3(0, 0, 0);
    }
}

// Function 93
float mat_sample(mat3 m, vec2 p) {
    vec3 s = coeffs(p.x);
	vec3 t = coeffs(p.y);    
	return dot(vec3(dot(m[0], s), dot(m[1], s), dot(m[2], s)), t);
}

// Function 94
float linesTextureGradBox( in float p, in float ddx, in float ddy, int id )
{
    float N = float( 2 + 7*((id>>1)&3) );

    float w = max(abs(ddx), abs(ddy)) + 0.01;
    float a = p + 0.5*w;                        
    float b = p - 0.5*w;           
    return 1.0 - (floor(a)+min(fract(a)*N,1.0)-
                  floor(b)-min(fract(b)*N,1.0))/(N*w);
}

// Function 95
vec3 mtlSample(Material mtl, in vec3 Ng, in vec3 Ns, in vec3 E, in vec2 xi, out vec3 L, out float pdf, out float spec) {
    float alpha = mtl.specular_roughness_;
    mat3 trans = mat3FromNormal(Ns);
    mat3 inv_trans = mat3Inverse( trans );
    
    //convert directions to local space
    vec3 E_local = inv_trans * E;
    vec3 L_local;
    
    if (E_local.z == 0.0) return vec3(0.);
    
    //Sample specular or diffuse lobe based on fresnel
    if(rnd() < mtl.specular_weight_) {
    	vec3 wh = ggx_sample(E_local, alpha, alpha, xi);
        L_local = reflect(-E_local, wh);
        pdf = pdfSpecular(alpha, alpha, E_local, L_local);
    } else {
        L_local = sampleHemisphereCosWeighted( xi );
        pdf = pdfDiffuse(L_local);
    }
    
    //convert directions to global space
    L = trans*L_local;
    
    if(!sameHemisphere(Ns, E, L) || !sameHemisphere(Ng, E, L)) {
        pdf = 0.0;
    }
    
    return mtlEval(mtl, Ng, Ns, E, L);
}

// Function 96
vec2 Sample_Gauss2D(float u, float v)
{
    float l = sqrt(-2.0 * log(u));
    
    return vec2(cos(v * Pi), sin(v * Pi)) * l;
}

// Function 97
vec4 sample_tile(sampler2D buffer, vec2 resolution, vec2 uv, float tile_id_f) {
    float tile_height = 1.0 / pow(2.0, tile_id_f);
    
    uv.x += 1.0;
    vec2 area_uv = uv * tile_height;
    
    // Compensate for GL.LINEAR sampling - we need to sample the middle of the pixel
    vec2 tile_resolution = resolution * tile_height;
    vec2 inv_resolution = 1.0 / resolution.xy;
    area_uv -= mod(area_uv, inv_resolution) - inv_resolution * 0.5;
    
    return texture(buffer, area_uv);
}

// Function 98
vec3 texCubeSampleWeights(float i) {
	return vec3((1.0 - i) * (1.0 - i), 2.0 * i * (1.0 - i), i * i);
}

// Function 99
vec2 concentricSampleDisk(const in vec2 u) {
    vec2 uOffset = 2. * u - vec2(1., 1.);

    if (uOffset.x == 0. && uOffset.y == 0.) return vec2(0., 0.);

    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = PI/4. * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PI/2. - PI/4. * (uOffset.x / uOffset.y);
    }
    return r * vec2(cos(theta), sin(theta));
}

// Function 100
float sample1(vec2 coord)
{
    float value = texture(iChannel0, coord.xy/iChannelResolution[0].xy).x;
    return fract(value * 373.7681691);
}

// Function 101
vec3 textureNoTile( sampler2D samp, in vec2 uv, float v )
{
    vec2 p = floor( uv );
    vec2 f = fract( uv );
	
    // derivatives (for correct mipmapping)
    vec2 ddx = dFdx( uv );
    vec2 ddy = dFdy( uv );
    
	vec3 va = vec3(0.0);
	float w1 = 0.0;
    float w2 = 0.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2( float(i),float(j) );
		vec4 o = hash4( p + g );
		vec2 r = g - f + o.xy;
		float d = dot(r,r);
        float w = exp(-5.0*d );
        vec3 c = textureGrad( samp, uv + v*o.zw, ddx, ddy ).xyz;
		va += w*c;
		w1 += w;
        w2 += w*w;
    }
    
    // normal averaging --> lowers contrasts
    //return va/w1;

    // contrast preserving average
    float mean = 0.3;// textureGrad( samp, uv, ddx*16.0, ddy*16.0 ).x;
    vec3 res = mean + (va-w1*mean)/sqrt(w2);
    return mix( va/w1, res, v );
}

// Function 102
vec3 sample_brdf( SurfaceInfo surface,
                 MaterialInfo material,
                out float pdf)
{
           
    vec2 u12 = hash21(material.seed);
    
    float cosTheta = pow(max(0., u12.x), 1./(material.specExponent+1.));
    float sinTheta = sqrt(max(0., 1. - cosTheta * cosTheta));
    float phi = u12.y * TWO_PI;
    
    vec3 whLocal = polar_to_cartesian(sinTheta, cosTheta, sin(phi), cos(phi));

    vec3 tangent = vec3(0.), binormal = vec3(0.);
    calc_binormals(surface.normal, tangent, binormal);
    
    vec3 wh = whLocal.x * tangent + whLocal.y * binormal + whLocal.z * surface.normal;
    
    vec3 wo = -surface.incomingRayDir;    
    if (dot(wo, wh) < 0.)
    {
       wh *= -1.;
    }
            
    vec3 wi = reflect(surface.incomingRayDir, wh);
    
    pdf = ((material.specExponent + 1.) * pow(clamp(abs(dot(wh, surface.normal)),0.,1.), material.specExponent))/(TWO_PI * 4. * dot(wo, wh));
    return wi;
}

// Function 103
vec4 SampleFontTex(vec2 uv)
{
    // Do some tricks with the UVs to spell out "TexFont" in the middle.
    vec2 fl = floor(uv + 0.5);
    if (fl.y == 0.0) {
        if (fl.x == -3.0) fl = vec2(4.0, 10.0);
    	else if (fl.x == -2.0) fl = vec2(5.0, 9.0);
    	else if (fl.x == -1.0) fl = vec2(8.0, 8.0);
    	else if (fl.x == 0.0) fl = vec2(6.0, 11.0);
    	else if (fl.x == 1.0) fl = vec2(15.0, 9.0);
    	else if (fl.x == 2.0) fl = vec2(14.0, 9.0);
    	else if (fl.x == 3.0) fl = vec2(4.0, 8.0);
    }
    uv = fl + fract(uv+0.5)-0.5;

    // Sample the font texture. Make sure to not use mipmaps.
    // Add a small amount to the distance field to prevent a strange bug on some gpus. Slightly mysterious. :(
    return texture(iChannel0, (uv+0.5)*(1.0/16.0), -100.0) + vec4(0.0, 0.0, 0.0, 0.000000001);
}

// Function 104
vec4 grassTexture(vec3 pos, vec3 nor)
{
    
    float g = texture(iChannel1, pos.xz*.5).x;
    float s = texture(iChannel1, pos.xz*.015).x*.2;
    
    
    vec3 flower = texture(iChannel2, pos.xz*.15).xyz;
    float rand = texture(iChannel1, pos.xz*.003).x;
    rand *= rand*rand;
    
    flower =pow(flower,vec3(8, 15, 5)) *10. * rand;
    vec4 mat = vec4(g*.05+s, g*.65, 0, g*.1);
    mat.xyz += flower;
    
	return min(mat, 1.0);
}

// Function 105
vec4 textureUgly(in sampler2D tex, in vec2 uv, in vec2 res) {
    return textureLod(tex, (floor(uv*res)+.5)/res, 0.0).xxxx;
}

// Function 106
float SampleNoise(vec3 p)
{
    float h = noise(p) * 1.1 - 0.1;
    h *= noise(p / 2.);
    return h;
}

// Function 107
vec4 SampleCubic(vec2 mPos)
{    
	mPos -= 0.5f;

	vec2 fuvw = fract(mPos);
	mPos -= fuvw;

	vec4 cubicX = cubic2(fuvw.x);
	vec4 cubicY = cubic2(fuvw.y);

	vec2 cX = mPos.xx + vec2(-0.5f, 1.5f);
	vec2 cY = mPos.yy + vec2(-0.5f, 1.5f);

	vec2 sX = cubicX.xz + cubicX.yw;
	vec2 sY = cubicY.xz + cubicY.yw;

	vec2 offsetX = cX + cubicX.yw / sX;
	vec2 offsetY = cY + cubicY.yw / sY;

	vec4 value0;
	vec4 value1;
	vec4 value2;
	vec4 value3;

	value0 = textureLod(iChannel0, vec2(offsetX.x, offsetY.x) / iResolution.xy, 0.0);
	value1 = textureLod(iChannel0, vec2(offsetX.y, offsetY.x) / iResolution.xy, 0.0);
	value2 = textureLod(iChannel0, vec2(offsetX.x, offsetY.y) / iResolution.xy, 0.0);
	value3 = textureLod(iChannel0, vec2(offsetX.y, offsetY.y) / iResolution.xy, 0.0);

	float lX = sX.x / (sX.x + sX.y);
	float lY = sY.x / (sY.x + sY.y);

	return mix(mix(value3, value2, lX), mix(value1, value0, lX), lY);
}

// Function 108
vec3 sampleBSDF(	in vec3 x,
                  	in vec3 ng,
                  	in vec3 ns,
                	in vec3 wi,
                  	in Material mtl,
                  	in bool useMIS,
                  	in int strataCount,
                  	in int strataIndex,
                	out vec3 wo,
                	out float brdfPdfW,
                	out vec3 fr,
                	out bool hitRes,
                	out SurfaceHitInfo hit,
               		out float spec) {
    float strataSize = 1.0 / float(strataCount);
    vec3 Lo = vec3(0.0);
    for(int i=0; i<DL_SAMPLES; i++){
        vec2 xi = vec2(rnd(), strataSize * (float(strataIndex) + rnd()));
        fr = mtlSample(mtl, ng, ns, wi, xi, wo, brdfPdfW, spec);
        
        //fr = eval(mtl, ng, ns, wi, wo);

        float dotNWo = dot(wo, ns);
        //Continue if sampled direction is under surface
        if ((dot(fr,fr)>0.0) && (brdfPdfW > EPSILON)) {
            Ray shadowRay = Ray(x, wo);

            //abstractLight* pLight = 0;
            float cosAtLight = 1.0;
            float distanceToLight = -1.0;
            vec3 Li = vec3(0.0);

            {
                float distToHit;

                if(raySceneIntersection( shadowRay, EPSILON, false, hit, distToHit )) {
                    if(hit.mtl_id_>=LIGHT_ID_BASE) {
                        distanceToLight = distToHit;
                        cosAtLight = dot(hit.normal_, -wo);
                        if(cosAtLight > 0.0) {
                            Li = getRadiance(hit.uv_);
                            //Li = lights[0].color_*lights[0].intensity_;
                        }
                    } else {
                        hitRes = true;
                    }
                } else {
                    hitRes = false;
                    //TODO check for infinite lights
                }
            }

            if (distanceToLight>0.0) {
                if (cosAtLight > 0.0) {
                    vec3 contribution = (Li * fr * dotNWo) / brdfPdfW;

                    if (useMIS/* && !(mtl->isSingular())*/) {
                        float lightPickPdf = 1.0;//lightPickingPdf(x, n);
                        float lightPdfW = sampleLightSourcePdf( x, wi, distanceToLight, cosAtLight );
                        //float lightPdfW = sphericalLightSamplingPdf( x, wi );//pLight->pdfIlluminate(x, wo, distanceToLight, cosAtLight) * lightPickPdf;

                        contribution *= misWeight(brdfPdfW, lightPdfW);
                    }

                    Lo += contribution;
                }
            }
        }
    }

    return Lo*(1.0/float(DL_SAMPLES));
}

// Function 109
float filterFlowToTexture(float f, vec2 uv) {
    f += smoothstep(.3, .5, f) * .8 * (1. - f);
    f = smoothstep(0., 1., f);
    f = filterEdgesOfLava(f, uv);
    return f;
}

// Function 110
float get_secondary_texture_index(in sampler2D s)
{
    return texelFetch(s, CTRL_TEXTURE, 0).z;
}

// Function 111
vec3 cosineSampleHemisphere(float u1, float u2)
{
    const float r = Sqrt(u1);
    const float theta = 2 * kPi * u2;
 
    const float x = r * Cos(theta);
    const float y = r * Sin(theta);
 
    return Vector3(x, y, Sqrt(Max(0.0f, 1 - u1)));
}

// Function 112
vec3 URandSample(vec2 v) {
    float z=v.x*2.-1.;
    float rxy=sqrt(1.-z*z);
    float phi=v.y*2.*PI;
    return vec3(rxy*cos(phi), rxy*sin(phi), z);
}

// Function 113
float gridTexture( in vec2 p )
{
	// filter kernel
    vec2 w = fwidth(p) + 0.01;

	// analytic (box) filtering
    vec2 a = p + 0.5*w;                        
    vec2 b = p - 0.5*w;           
    vec2 i = (floor(a)+min(fract(a)*N,1.0)-
              floor(b)-min(fract(b)*N,1.0))/(N*w);
    //pattern
    return (1.0-i.x)*(1.0-i.y);
}

// Function 114
vec3 sampleLight( const in vec3 ro, inout float seed ) {
    vec3 n = randomSphereDirection( seed ) * lightSphere.w;
    return lightSphere.xyz + n;
}

// Function 115
vec4 textureN( in sampler2D tex ,in vec2 pg)
{
    vec2 p=pg*(iResolution.xy)-0.5;
    vec2 i = floor( p );
    vec2 f = fract( p );
	

    return (hashg( tex,i + vec2(1.0,0.0))*f.x+hashg( tex,i + vec2(0.0,0.0))*(1.0-f.x))*(1.0-f.y)+f.y*(hashg( tex,i + vec2(1.0,1.0))*f.x+hashg( tex,i + vec2(0.0,1.0))*(1.0-f.x));
}

// Function 116
vec3 SampleEnvironmentWithRoughness(vec3 samplingVec, float roughness) {
    float maxLodLevel = log2(float(textureSize(iChannel0, 0).x));

    // makes roughness of reflection scale perceptually much more linear
    // Assumes "CubeTexSizeReflection" = 1024
    maxLodLevel -= 4.0;

    float lodBias = maxLodLevel * roughness;

    return sampleReflectionMap(samplingVec, lodBias);
}

// Function 117
float checker_texture(vec2 texcoord) {
    vec2 repeated = mod(texcoord, vec2(1.0));
    vec2 chequer = smoothstep(0.4, 0.5, repeated) * smoothstep(1.0, 0.95, repeated);
    return abs(1.0-chequer.x-chequer.y);
}

// Function 118
vec4 hexsample(vec2 p, sampler2D tex)
{
    float an = acos(-1.) / 3.;
    float hr = sin(an) * 2.;

    p.x -= min(fract(p.y / 2.) * 2., 2. - fract(p.y / 2.) * 2.) * .5;

    vec2 c = floor(p);
    vec2 cf = p - c;

    vec4 va = texelFetch(tex, ivec2(c + vec2(0, 0)), 0);
    vec4 vb = texelFetch(tex, ivec2(c + vec2(1, 0)), 0);
    vec4 vc = texelFetch(tex, ivec2(c + vec2(0, 1)), 0);
    vec4 vd = texelFetch(tex, ivec2(c + vec2(1, 1)), 0);

    vec4 r = mix(mix(va, vb, cf.x), mix(vc, vd, cf.x), cf.y);

    return r;
}

// Function 119
float sampleLightSourcePdf(in vec3 x, vec3 ns, in vec3 wi, float d, float cosAtLight) {
    return PdfAtoW(1.0 / (light.size.x*light.size.y), d*d, cosAtLight);
}

// Function 120
vec4 textureAspect(sampler2D tex, vec3 channelResolution, vec3 iResolution, vec2 fragCoord)
{
    vec2 U = fragCoord;
    vec2 margin = vec2(0),
         Sres = iResolution.xy -2.*margin,
         Tres = channelResolution.xy,
         ratio = Sres/Tres;
    
    U -= margin;
    
    // centering the blank part in case of rectangle fit
    U -= .5*Tres*max(vec2(ratio.x-ratio.y,ratio.y-ratio.x),0.);
    
  //U /= Tres*ratio.y;               // fit height, keep ratio
  //U /= Tres*ratio.x;               // fit width, keep ratio
    U /= Tres*min(ratio.x,ratio.y);  // fit rectangle,  keep ratio
    U *= 1.;                         // zoom out factor 
    
	vec4 result = fract(U)==U 
        ? texture(tex, U)
        : vec4(0.0);
        
        return result;
}

// Function 121
float Sample(sampler2D channel, vec2 uv, vec2 texelCount)
{
    vec2 uv0 = uv;
    
    uv -= vec2(0.5);
    
    vec2 uvi = floor(uv);
    vec2 uvf = uv - uvi;

    vec2 mo = uvf - uvf*uvf;
    
   #if 0
    mo = (mo * -0.5 + 1.0) * mo;// use this if it improves quality
   #endif
    
    //uvf = (uvf - mo) / (1.0 - 2.0 * mo);// map modulator to s-curve

    uvf.y = cubic(uvf.y);
    
    uv = uvi + uvf + vec2(0.5);

    vec4 v = textureLod(channel, uv / texelCount, 0.0);

    if(false)
    v.x = mix(mix(texelFetch(channel, ivec2(uvi)+ivec2(0,0), 0).x, texelFetch(channel, ivec2(uvi)+ivec2(1,0), 0).x, uvf.x),
              mix(texelFetch(channel, ivec2(uvi)+ivec2(0,1), 0).x, texelFetch(channel, ivec2(uvi)+ivec2(1,1), 0).x, uvf.x), uvf.y);
    
    mo *= fract(uvi * 0.5) * 4.0 - 1.0;// flip modulator bump on every 2nd interval
    
    return v.x * mo.x;//exact 
    return dot(v, vec4(mo.xy, mo.x*mo.y, 1.0));
}

// Function 122
vec4 fruitTexture(vec3 p, vec3 nor, float i)
{
    
    
    float rand = texCube(iChannel2, p*.1 ,nor).x;
    float t = dot(nor, normalize(vec3(.8, .1, .1)));
	vec3 mat = vec3(1.,abs(t)*rand,0);
    mat = mix(vec3(0,1,0), mat, i/10.);

   	return vec4(mat, .5);
}

// Function 123
vec3 bsdfSample(out vec3 wi, const in vec3 wo, const in vec3 X, const in vec3 Y,  out float pdf, const in SurfaceInteraction interaction, const in MaterialInfo material) {
    
    vec3 f = vec3(0.);
    pdf = 0.0;
	wi = vec3(0.);
    
    vec2 u = vec2(random(), random());
    float rnd = random();
	if( rnd <= 0.3333 ) {
       disneyDiffuseSample(wi, wo, pdf, u, interaction.normal, material);
    }
    else if( rnd >= 0.3333 && rnd < 0.6666 ) {
       disneyMicrofacetAnisoSample(wi, wo, X, Y, u, interaction, material);
    }
    else {
       disneyClearCoatSample(wi, wo, u, interaction, material);
    }
    f = bsdfEvaluate(wi, wo, X, Y, interaction, material);
    pdf = bsdfPdf(wi, wo, X, Y, interaction, material);
    if( pdf < EPSILON )
        return vec3(0.);
	return f;
}

// Function 124
vec3 ggx_sample(vec3 wi, float alphax, float alphay, vec2 xi) {
    //stretch view
    vec3 v = normalize(vec3(wi.x * alphax, wi.y * alphay, wi.z));

    //orthonormal basis
    vec3 t1 = (v.z < 0.9999) ? normalize(cross(v, vec3(0.0, 0.0, 1.0))) : vec3(1.0, 0.0, 0.0);
    vec3 t2 = cross(t1, v);

    //sample point with polar coordinates
    float a = 1.0 / (1.0 + v.z);
    float r = sqrt(xi.x);
    float phi = (xi.y < a) ? xi.y / a*PI : PI + (xi.y - a) / (1.0 - a) * PI;
    float p1 = r*cos(phi);
    float p2 = r*sin(phi)*((xi.y < a) ? 1.0 : v.z);

    //compute normal
    vec3 n = p1*t1 + p2*t2 + v*sqrt(1.0 - p1*p1 - p2*p2);

    //unstretch
    return normalize(vec3(n.x * alphax, n.y * alphay, n.z));
}

// Function 125
vec4 SampleCubic(vec2 uv)
{
    uv += 0.5;
    vec2 uv0 = floor(uv);
    vec2 fuv = fract(uv);
    
    vec4 col = vec4(0.0);
    for(float y = 0.0; y < 4.0; ++y)
    for(float x = 0.0; x < 4.0; ++x)
    {
        vec2 o = vec2(x, y);
        vec2 w = 1.0 - abs(fuv - o);
        w = BSpline(fuv - o+1.0);
        
    	col += texelFetch(iChannel0, ivec2(uv0+o)-2, 0) * (w.x*w.y);
    }
    
    return col;
}

// Function 126
vec3 sampleEnvironment(vec3 dir)
{
    return pow(texture(iChannel0, dir).rgb, GAMMA);
}

// Function 127
float getSampleAt(vec2 uv, mat4 viewMatrix, mat4 inverseViewMatrix)
{
	//This is a vector from the camera to the near plane
	vec3 cameraToNear = vec3(0, 0, (1.0 / tan(verticalFov)));
	
	//Direction of line from camera to near plane in eye coordinates, this is the "ray"
	vec3 lineDirection = vec3(uv.x , uv.y, 0) - cameraToNear;
	
	//Plane point in eye coordinates
	vec3 transformedCenterPointOnPlane = vec3(viewMatrix * vec4(centerPointOnPlane, 1.0));
	
	//Plane normal in eye coordinates
	vec3 transformedNormalToPlane = vec3(viewMatrix * vec4(normalToPlane, 0.0));
	
	//Distance to line/plane intersection 
	float distanceAlongLine = dot(transformedCenterPointOnPlane, transformedNormalToPlane) / (dot(lineDirection, transformedNormalToPlane));
	
	//Convert point on plane in eye coordinates to object coordinates
	vec4 pointInBasis = inverseViewMatrix * vec4(distanceAlongLine * lineDirection, 1.0);

	float value = 0.0;
	//If the point is inside the plane boundaries
	if(abs(pointInBasis.x) <= (planeWidth / 2.0) && abs(pointInBasis.y) <= (planeHeight / 2.0))
	//if(length(pointInBasis.xy) <= (planeWidth / 2.0)) // circle
	{
		value = 1.0;	
	}
	
	return value;
}

// Function 128
bool get_load_texture(in sampler2D s)
{
    return texelFetch(s, CTRL_LOAD_TEXTURE, 0).w > 0.05;
}

// Function 129
vec3 texsample(const int x, const int y, in vec2 fragCoord)
{
	vec2 uv = fragCoord.xy / iResolution.xy * iChannelResolution[0].xy;
	uv = (uv + vec2(x, y)) / iChannelResolution[0].xy;
	return texture(iChannel0, uv).xyz;
}

// Function 130
vec3 texture_cube(vec3 usphp, float scale)
{
    // cheap way of turning a 2D texture into a cube
    // map lookup by ray casting sphere intersection direction 
    // to the appropriate plane of the surrounding cube and then
    // using the uv of that plane intersection as a 2d vector.
    
    vec3 p = usphp;
#if PROJECTION_TYPE == SLIDING_PROJECTION
    p.y -= sin(.1 * iTime);
#endif
    
    float ml = max(abs(p.x), max(abs(p.y), abs(p.z)));
    vec3 ansphp = abs(p/ml); 
    
    // select the plane offset of a unit cube
    vec3 upo = sign(p) * step(vec3(1. - SMALL_FLOAT), ansphp);
    
    // scale the plane we are intersecting by the offset
    vec2 pr = intersect_plane(vec3(0.), p, -upo, scale * upo);
    vec3 pp = pr.y * p;

    // get the uv lookup of the plane intersection.
    vec2 uv = step(1. - SMALL_FLOAT, ansphp.x) * pp.yz;
    uv += step(1. - SMALL_FLOAT, ansphp.y) * pp.xz;
    uv += step(1. - SMALL_FLOAT, ansphp.z) * pp.xy;

#if PROJECTION_TYPE == STATIC_TEXTURE | PROJECTION_TYPE == SLIDING_PROJECTION
    
    // filter texture lookup more when unit cube is closer to
    // unit sphere - cheap hack to compensate for the fact
    // that the texture lookup is more stretched near the
    // corners.
    float f = 1.2 * (1. - length(pp/scale - p));
    
    return texture(iChannel0, .5 * uv + .5, f).rgb;
#elif PROJECTION_TYPE == ANIMATED_NOISE
    
    return hash32(.5 * uv + .5);
    
#endif

}

// Function 131
vec4 leavesTexture(vec3 p, vec3 nor)
{
    
    vec3 rand = texCube(iChannel2, p*.15,nor);
	vec3 mat = vec3(0.4,1.2,0) *rand;
    if (nor.y < 0.0) mat += vec3(1., 0.5,.5);
    
   	return vec4(mat, .0);
}

// Function 132
vec2 sampleSnowflakes(vec2 p, float lod)
{
    return textureLod(iChannel0, vec3(1., -p.yx), min(lod, 7.)).rg;
}

// Function 133
void sampleWeightsPolar(vec2 sample_point, vec2 points[POINT_COUNT], out float weights[POINT_COUNT] )
{
    
    const float kDirScale   = 2.0;
    
    float   total_weight    = 0.0;
    
    float   sample_mag      = length( sample_point );
    
    for( int i = 0; i < POINT_COUNT; ++i )
    {      
        vec2    point_i     = points[i];
        float   point_mag_i = length(point_i);
        
        float   weight      = 1.0;
        
        for( int j = 0; j < POINT_COUNT; ++j )
        {
            if( j == i ) 
                continue;
            
            vec2    point_j         = points[j];
            float   point_mag_j     = length( point_j );
            
            float   ij_avg_mag      = (point_mag_j + point_mag_i) * 0.5;
            
            // Calc angle and mag for i -> sample
            float   mag_is          = (sample_mag - point_mag_i) / ij_avg_mag;
            float   angle_is		= signedAngle(point_i, sample_point);
            
            // Calc angle and mag for i -> j
            float   mag_ij          = (point_mag_j - point_mag_i) / ij_avg_mag;
            float   angle_ij		= signedAngle(point_i, point_j);
            
            // Calc vec for i -> sample
            vec2    vec_is;
            vec_is.x                = mag_is;
            vec_is.y                = angle_is * kDirScale;
            
            // Calc vec for i -> j
            vec2    vec_ij;
            vec_ij.x                = mag_ij;
            vec_ij.y                = angle_ij * kDirScale;
            
            // Calc weight
         	float lensq_ij      = dot( vec_ij, vec_ij );
            float new_weight    = dot( vec_is, vec_ij ) / lensq_ij;
            new_weight          = 1.0-new_weight;
            new_weight          = clamp( new_weight, 0.0, 1.0 );
            
            weight              = min( new_weight, weight );
        }
        
        weights[i]          = weight;
        
        total_weight        += weight;
    }
    
    for( int i = 0; i < POINT_COUNT; ++i )
    {
		weights[i] = weights[i] / total_weight;
    }
}

// Function 134
vec3 betterTextureSample256(sampler2D tex, vec2 uv) {	
	float textureResolution = 256.0;
	uv = uv*textureResolution + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*(3.0-2.0*fuv); // fuv*fuv*fuv*(fuv*(fuv*6.0-15.0)+10.0);;
	uv = (uv - 0.5)/textureResolution;
	return textureLod(tex, uv, 0.0).rgb;
}

// Function 135
float2 Normal_Sampler ( in sampler2D s, in float2 uv ) {
  float2 eps = float2(0.003, 0.0);
  return float2(length(texture(s, uv+eps.xy)) - length(texture(s, uv-eps.xy)),
                length(texture(s, uv+eps.yx)) - length(texture(s, uv-eps.yx)));
}

// Function 136
vec4 textureNoTile_3weights( sampler2D samp, in vec2 uv )
{
    vec4 res = vec4(0.);
    int sampleCnt = 0; // debug vis
    
    // compute per-tile integral and fractional uvs.
    // flip uvs for 'odd' tiles to make sure tex samples are coherent
    vec2 fuv = mod( uv, 2. ), iuv = uv - fuv;
    vec3 BL_one = vec3(0.,0.,1.); // xy = bot left coords, z = 1
    if( fuv.x >= 1. ) fuv.x = 2.-fuv.x, BL_one.x = 2.;
    if( fuv.y >= 1. ) fuv.y = 2.-fuv.y, BL_one.y = 2.;
    
    
    // weight orthogonal to diagonal edge = 3rd texture sample
    vec2 iuv3;
    float w3 = (fuv.x+fuv.y) - 1.;
    if( w3 < 0. ) iuv3 = iuv + BL_one.xy, w3 = -w3; // bottom left corner, offset negative, weight needs to be negated
    else iuv3 = iuv + BL_one.zz; // use transform from top right corner
    w3 = smoothstep(BLEND_WIDTH, 1.-BLEND_WIDTH, w3);
    
    // if third sample doesnt dominate, take first two
    if( w3 < 0.999 )
    {
        // use weight along long diagonal edge
        float w12 = dot(fuv,vec2(.5,-.5)) + .5;
        w12 = smoothstep(1.125*BLEND_WIDTH, 1.-1.125*BLEND_WIDTH, w12);

        // take samples from texture for each side of diagonal edge
        if( w12 > 0.001 ) res +=     w12  * texture( samp, transformUVs( iuv + BL_one.zy, uv ) ), sampleCnt++;
        if( w12 < 0.999 ) res += (1.-w12) * texture( samp, transformUVs( iuv + BL_one.xz, uv ) ), sampleCnt++;
    }
    
	// first two samples aren't dominating, take third
    if( w3 > 0.001 ) res = mix( res, texture( samp, transformUVs( iuv3, uv ) ), w3 ), sampleCnt++;

    
    // debug vis: colour based on num samples taken for vis purposes
    if( iMouse.z > 0. )
    {
        if( sampleCnt == 1 ) res.rb *= .25;
        if( sampleCnt == 2 ) res.b *= .25;
        if( sampleCnt == 3 ) res.gb *= .25;
    }
    
    return res;
}

// Function 137
vec4 get_texture_params(in sampler2D s)
{
    return texelFetch(s, CTRL_TEXTURE, 0);
}

// Function 138
float samplef(in vec2 uv) {
	ivec2 p = ivec2(0.0);
	p.x = int( uv.x*16.);
	p.y = int( uv.y*16.);
    p.x = (p.x>7) ? 15-p.x : p.x;

    float rr=0.0;

    if(p.y== 0) rr=MKNum( _i,_i,_i,_i,_i,WW,WW,WW);
    if(p.y== 1) rr=MKNum( _i,_i,_i,WW,WW,WW,OO,cc);
    if(p.y== 2) rr=MKNum( _i,_i,WW,WW,OO,OO,OO,cc);
    if(p.y== 3) rr=MKNum( _i,WW,WW,cc,OO,OO,cc,cc);
    if(p.y== 4) rr=MKNum( _i,WW,OO,cc,cc,cc,cc,cc);
    if(p.y== 5) rr=MKNum( WW,WW,OO,OO,cc,cc,OO,OO);
    if(p.y== 6) rr=MKNum( WW,OO,OO,OO,cc,OO,OO,OO);
    if(p.y== 7) rr=MKNum( WW,OO,OO,OO,cc,OO,OO,OO);
    if(p.y== 8) rr=MKNum( WW,OO,OO,cc,cc,OO,OO,OO);
    if(p.y== 9) rr=MKNum( WW,cc,cc,cc,cc,cc,OO,OO);
    if(p.y==10) rr=MKNum( WW,cc,cc,WW,WW,WW,WW,WW);
    if(p.y==11) rr=MKNum( WW,WW,WW,WW,_i,_i,WW,_i);
    if(p.y==12) rr=MKNum( _i,WW,WW,_i,_i,_i,WW,_i);
    if(p.y==13) rr=MKNum( _i,_i,WW,_i,_i,_i,_i,_i);
    if(p.y==14) rr=MKNum( _i,_i,WW,WW,_i,_i,_i,_i);
    if(p.y==15) rr=MKNum( _i,_i,_i,WW,WW,WW,WW,WW);

    return mod( floor(rr / pow(4.0,float(p.x))), 4.0 )/3.0;
}

// Function 139
float textureHologram(vec2 t, vec3 e) {
    float r = length(t);
    t.x += e.x * 0.2;
    
    float l3 = smoothstep(0.5,0.52,r);
    float l0 = smoothstep(0.98,0.97,r) * l3;
    float l1 = saturate(sin(t.y*40.0)*8.0) * saturate(sin((t.y-t.x)*10.0)*8.0+6.0);
    float l2 = saturate(sin(t.y*160.0)*8.0) * saturate(sin((t.y+t.x)*40.0)*8.0+6.0);
    float l4 = smoothstep(0.42,0.4,r) * smoothstep(0.39,0.399,r);
    float l5 = smoothstep(1.0,0.99,r) * smoothstep(0.97,0.98,r);
    
    float sum = 0.0;
    sum += (1.0-l3) * 0.5;
    sum += l1 * l0;
    sum += l2 * l0 * (1.0 - l1) * 0.2;
    sum += l4 * 0.5;
    sum += l5;
    return sum;
}

// Function 140
vec3 TriplanarTextureMapping(const vec3 p, const vec3 n, const int texID)
{
    mat3 samples;
    
    switch(texID)
    {
        // iChannel0 is for the SkyBox
        case 1:
        	samples = mat3 (texture(iChannel1, p.yz).rgb,
                         	texture(iChannel1, p.xz).rgb,
                         	texture(iChannel1, p.xy).rgb );
        	break;
        case 2:
        	samples = mat3 (texture(iChannel2, p.yz).rgb,
                         	texture(iChannel2, p.xz).rgb,
                         	texture(iChannel2, p.xy).rgb );
        	break;
        case 3:
        	samples = mat3 (texture(iChannel3, p.yz).rgb,
                         	texture(iChannel3, p.xz).rgb,
                         	texture(iChannel3, p.xy).rgb );
        	break;
        default:
        	samples = mat3(0);
        	break;
    }
    
    // Weight the samples with the normal to get the one more aligned
    return samples * abs(n);
}

// Function 141
vec3 samplef(in vec2 in_uv) {
    vec2 suv = (in_uv + iTime / uv_scale);
    vec2 n = floor(suv);
    vec2 f = fract(suv)*2.0-1.0;
    
    vec3 total = vec3(0.0);
    float w = 0.0;
    
    vec2 uv;
    for (uv.y = -1.0f; uv.y <= 1.0f; ++uv.y) {
        for (uv.x = -1.0f; uv.x <= 1.0f; ++uv.x) {
            float a;    
            a = compute_area(f-uv*2.0);
            total += fetch_nn(n + uv) * a;
            w += a;
        }
    }
    
    return ((in_uv.x+in_uv.y-m*uv_scale) < 0.5)?fetch_iq(suv):(total/w);
    
}

// Function 142
vec3 cosineSampleHemisphere() {
    vec2 u = vec2(rnd(), rnd());
    float r = sqrt(u.x);
    float theta = 2.0 * PI * u.y;
    return vec3(r * cos(theta), r * sin(theta), sqrt(saturate(1.0 - u.x)));
}

// Function 143
void sampleEquiAngular(
	Ray ray,
	float maxDistance,
	float Xi,
	vec3 lightPos,
	out float dist,
	out float pdf)
{
	// get coord of closest point to light along (infinite) ray
	float delta = dot(lightPos - ray.origin, ray.dir);
	
	// get distance this point is from light
	float D = length(ray.origin + delta*ray.dir - lightPos);

	// get angle of endpoints
	float thetaA = atan(0.0 - delta, D);
	float thetaB = atan(maxDistance - delta, D);
	
	// take sample
	float t = D*tan(mix(thetaA, thetaB, Xi));
	dist = delta + t;
	pdf = D/((thetaB - thetaA)*(D*D + t*t));
}

// Function 144
vec2 sampleLight(vec3 rp, vec3 n)
{
    vec2 energy = vec2(0);
    vec3 lightdir = normalize(normalize(vec3(.2, 4., 1.)) +
                              (vec3(rand(), rand(), rand()) * 2. - 1.) * .015);
    vec3 n2, uvw;
    float t = traceScene(rp, lightdir, n2, uvw).x;

    vec3 lrp = rp + lightdir * t;

    // Directional 'sky' lighting.
    if((lrp.y > .999 && abs(lrp.x- -.3) < .6 && abs(lrp.z - .1) < .8))
        energy += vec2(1.5, .8).yx * max(0., dot(n, lightdir)) * 2.;

    vec3 lo = vec3(.7, .8, .1), ls = vec3(1, 0, 0) * .2, lt = vec3(0, 0, 1) * .2;
    vec3 ln = normalize(cross(ls, lt));
    
    int light_sample_count = 2;
    
    // Parallelogram local lightsource.
    for(int j = 0; j < light_sample_count; ++j)
    {
        float lu = rand() * 2. - 1., lv = rand() * 2. - 1.;
        vec3 lp = lo + ls * lu + lt * lv, n2;
        float ld = dot(normalize(lp - rp), n), ld2 = dot(normalize(rp - lp), ln);
        if(ld > 0. && ld2 > 0. && traceSceneShadow(rp + n * 1e-4, lp - rp))
            energy += vec2(1.5, .5) *
            	(1. / dot(rp - lp, rp - lp) * ld * ld2) / float(light_sample_count);
    }

    return energy;
}

// Function 145
float sampleConcrete(vec3 position) {
    float concrete = valueNoise3du(position * 4.0);
    concrete += valueNoise3du(position * 8.0) * 0.5;
    concrete += valueNoise3du(position * 16.0) * 0.25;
    concrete += valueNoise3du(position * 32.0) * 0.125;
    concrete /= 1.875;
    concrete = (abs(concrete*2.0-1.0));
    return concrete;
}

// Function 146
vec2 sample_biquadratic_gradient(sampler2D channel, vec2 res, vec2 uv) {
    vec2 q = fract(uv * res);
    vec2 c = (q*(q - 1.0) + 0.5) / res;
    vec2 w0 = uv - c;
    vec2 w1 = uv + c;
    vec2 cc = 0.5 / res;
    vec2 ww0 = uv - cc;
    vec2 ww1 = uv + cc;
    float nx0 = texture(channel, vec2(ww1.x, w0.y)).r - texture(channel, vec2(ww0.x, w0.y)).r;
    float nx1 = texture(channel, vec2(ww1.x, w1.y)).r - texture(channel, vec2(ww0.x, w1.y)).r;
    
    float ny0 = texture(channel, vec2(w0.x, ww1.y)).r - texture(channel, vec2(w0.x, ww0.y)).r;
    float ny1 = texture(channel, vec2(w1.x, ww1.y)).r - texture(channel, vec2(w1.x, ww0.y)).r;
    
	return vec2(nx0 + nx1, ny0 + ny1) / 2.0;
}

// Function 147
float gridTexture( in vec2 p )
{
    // coordinates
    vec2 i = step( fract(p), vec2(1.0/N) );
    //pattern
    return (1.0-i.x)*(1.0-i.y);   // grid (N=10)
    
    // other possible patterns are these
    //return 1.0-i.x*i.y;           // squares (N=4)
    //return 1.0-i.x-i.y+2.0*i.x*i.y; // checker (N=2)
}

// Function 148
vec2 radialSampleDisk(in vec2 xi) {
    float r = sqrt(1.0 - xi.x);
    float theta = xi.y*TWO_PI;
	return vec2(cos(theta), sin(theta))*r;
}

// Function 149
vec3 sampleSky(vec3 dir){
    return pow(texture(iChannel0, dir).xyz, vec3(2.2));
}

// Function 150
vec3 BilinearTextureSample (vec2 P)
{
    vec2 pixel = P * c_textureSize + 0.5;
    
    vec2 frac = fract(pixel);
    pixel = (floor(pixel) / c_textureSize) - vec2(c_onePixel/2.0);

    vec3 C11 = texture(iChannel0, pixel + vec2( 0.0        , 0.0)).rgb;
    vec3 C21 = texture(iChannel0, pixel + vec2( c_onePixel , 0.0)).rgb;
    vec3 C12 = texture(iChannel0, pixel + vec2( 0.0        , c_onePixel)).rgb;
    vec3 C22 = texture(iChannel0, pixel + vec2( c_onePixel , c_onePixel)).rgb;

    vec3 x1 = mix(C11, C21, frac.x);
    vec3 x2 = mix(C12, C22, frac.x);
    return mix(x1, x2, frac.y);
}

// Function 151
vec3 sampleBSDF(float objId,vec3 p,vec3 n)
{
	vec3 dir=uniformHemisphere(rnd(),rnd());
	return l2w(dir,n);
}

// Function 152
bool texture_store(in sampler2D s)
{
    return texelFetch(s, CTRL_TEXTURE, 0).y < 0.5;
}

// Function 153
vec4 getSampleAt(vec2 uv, mat4 viewMatrix, mat4 inverseViewMatrix)
{
			
	//This is a vector from the camera to the near plane
	vec3 cameraToNear = vec3(0, 0, (1.0 / tan(verticalFov)));
	
	//Direction of line from camera to near plane in eye coordinates, this is the "ray"
	vec3 lineDirection = vec3(uv.x, uv.y, 0) - cameraToNear;
	
	//Plane point in eye coordinates
	vec3 transformedCenterPointOnPlane = vec3(viewMatrix * vec4(centerPointOnPlane, 1.0));
	
	//Plane normal in eye coordinates
	vec3 transformedNormalToPlane = vec3(viewMatrix * vec4(normalToPlane, 0.0));
	
	//Distance to line/plane intersection 
	float distanceAlongLine = dot(transformedCenterPointOnPlane, transformedNormalToPlane) / (dot(lineDirection, transformedNormalToPlane));
	
	//Convert point on plane in eye coordinates to object coordinates
	vec4 pointInBasis = inverseViewMatrix * vec4(distanceAlongLine * lineDirection, 1.0);

	vec4 color = vec4(0,0,0,0);
	//If the point is inside the plane boundaries
	if(abs(pointInBasis.x) <= (planeWidth / 2.0) && abs(pointInBasis.y) <= (planeHeight / 2.0))
	{
		float value = 1.0 / 64.0;
		color = vec4(value, value, value, 0);	
	}
	
	return color;
}

// Function 154
void sampleLine(vec3 pos,
                vec3 normal,
                vec3 line_p0,
                vec3 line_dir,
                float line_len,
                float Xi,
               	out float t,
                out float tPdfL ) {
    
    
	float t_fix = 0.0;
    
#ifdef LIGHT_CLIPPING
    // clipping line segment
    float line_len_max;
    if(intersectPlane(normal, pos, line_p0, line_dir, line_len_max)) {
        
        //line_len = 0.0;//line_len_max;
        if(line_len > line_len_max) {
        
            if(dot(line_dir, normal) > 0.0) {
                line_p0 += line_dir*line_len_max;
                line_len -= line_len_max;
              	t_fix = line_len_max;
            } else {
                line_len = line_len_max;
            }
        }
    }
#endif
    
    // project shading point on line and calculate length
    float delta = dot(pos - line_p0, line_dir);

    // distance from pos to line
    float D = length(line_p0 + delta*line_dir - pos);

    // get angle of endpoints
    float thetaA = atan(0.0 - delta, D);
    float thetaB = atan(line_len - delta, D);
    float theta = mix(thetaA, thetaB, Xi);

    // take sample
    float delta1 = D*tan(theta);
    float cosTheta = cos(theta);
    float dist = D/cosTheta;
    t = (delta1 + delta);
    t += t_fix;
    float tPdfW = 1.0 / (thetaB - thetaA);
    tPdfL = PdfWtoA(tPdfW, dist, cosTheta);
}

// Function 155
vec4 SampleMip2(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*16.,sp.z+32.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(16.,0.))*IRES),fract(sp.y));
}

// Function 156
void sampleLine2(	vec3 p,				//shading point position
                	vec3 n,				//normal at shadint point
                    vec3 line_p0,		//Endpoint of the light segment
                    vec3 line_dir,		//direction of light segment
                    float line_len,		//length of the light segment
                    float Xi,			//uniform random number
                    out float t,		//sampled t parameter on the light segment
                    out float tPdfL ) {	//corresponding pdf
    vec3 p0p = line_p0 - p;
    float ap = length(p0p);
	p0p = normalize(line_p0 - p);
    line_dir = normalize(line_dir);	//ensure that line dir is normalized
    vec3 line_p1 = line_p0 + line_dir * line_len;
    vec3 p1p = normalize(line_p1 - p);
    
    float angle_kap = acos(dot(line_dir, -p0p));
    
    vec3 plane_z = normalize(cross(p0p, p1p));
    //vec3 plane_y = normalize(cross(plane_z,cross(normal, plane_z)));
    vec3 plane_y = normalize(n - plane_z * dot(n, plane_z));
    vec3 plane_x = cross(plane_y, plane_z);
    
    mat3 trans = mat3(plane_x, plane_y, plane_z);
    mat3 trans_inv = mat3Inverse( trans );
    
    vec3 p0p_local = trans_inv * p0p;
    vec3 p1p_local = trans_inv * p1p;
    float thetaA = cartesian_to_angular(p0p_local.xy);
    float thetaB = cartesian_to_angular(p1p_local.xy);
    float theta = mix(thetaA, thetaB, Xi);
    
    float angle_apk = abs(thetaA - theta);
    float angle_akp = PI - angle_apk - angle_kap;
    
    //sin theorem
    float ak = abs((ap * sin(angle_apk)) / sin(angle_akp));
    t = ak;
    float kp = abs((ap * sin(angle_kap)) / sin(angle_akp));
    float dist = abs(kp);
    float cosTheta = cos(0.5*PI - angle_akp);
    float tPdfW = 1.0 / (abs(thetaB - thetaA));
    tPdfL = PdfWtoA(tPdfW, dist, cosTheta);
}

// Function 157
float sampleHeight( vec2 coord, vec2 c, vec4 t ) {
    return 0.046 * dot(
        getImage( coord, c, t ).xyz,
        vec3( 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 )
    );
}

// Function 158
vec2 sampleSnowflakes(vec2 p)
{
    vec4 s = hexsample(p * 256., iChannel0);
    return s.rb * s.ga;
}

// Function 159
vec3 GetBoatTexture(RayHit marchResult)
{
  vec3 checkPos = TranslateBoat(marchResult.hitPos); 
  vec3 bCol= vec3(62, 52, 47)*1.3/255.;
  float frontDist = max(0., (0.25*(0.16*pow(length(checkPos.z-0.), 2.))));
  float n = 1.+(0.2*noise(vec3(checkPos.zx*0.01, checkPos.x)*34.));
  n *= 0.9+(0.1*noise2D(checkPos.xy*26.));  
  bCol = mix(vec3(0.6), bCol*n, step(-0.625, checkPos.y-frontDist));
  bCol = mix(vec3(0.05), bCol, step(0.08, length(-.7-(checkPos.y-frontDist))));
  bCol = mix(bCol*0.8, bCol*1.2, smoothstep(0., 0.18, length(-0.23-(checkPos.y-frontDist))));   
  bCol = mix(bCol, bCol*0.47, smoothstep(0.0, 0.32, length(0.-mod(checkPos.y-frontDist, 0.3)))); 
  return mix(bCol, bCol*0.8, smoothstep(-.1, 0.8, noise2D(checkPos.xz*3.7)));  
}

// Function 160
vec3 sampleReflectedEnvironment(in vec3 Wi, in float r, in float f0)
{
#if (USE_IBL)    
    float r2 = r * r;
    float samples = 0.0;
    vec3 u = perpendicularVector(Wi);
	vec3 v = cross(u, Wi);
    vec3 result = vec3(0.0);
    vec3 Wo = Wi;
    for (int i = 0; i < MAX_CUBEMAP_SAMPLES; ++i)
    {
        Wo = randomDirection(Wi, u, v, r2, Wo + result);
        vec3 H = normalize(Wi + Wo);
        float weight = microfacetWeight(r2, f0, dot(H, Wi), 1.0, dot(Wo, Wi));
        result += weight * texture(iChannel0, Wo).xyz;
        samples += weight;
	}    
    return result / samples;
	// */
#else
    return vec3(0.0);
#endif
}

// Function 161
float sampleCubeMap(float i, vec3 rd) {
	vec3 col = textureLod(iChannel0, rd * vec3(1.0,-1.0,1.0), 0.0).xyz; 
    return dot(texCubeSampleWeights(i), col);
}

// Function 162
void sampleWeightsCartesian(vec2 sample_point, vec2 points[POINT_COUNT], out float weights[POINT_COUNT] )
{
    float total_weight = 0.0;
    
    for( int i = 0; i < POINT_COUNT; ++i )
    {
        // Calc vec i -> sample
        vec2    point_i = points[i];
        vec2    vec_is  = sample_point - point_i;
        
        float   weight  = 1.0;
        
        for( int j = 0; j < POINT_COUNT; ++j )
        {
            if( j == i ) 
                continue;
            
            // Calc vec i -> j
            vec2    point_j     = points[j];            
            vec2    vec_ij      = point_j - point_i;      
            
            // Calc Weight
            float lensq_ij      = dot( vec_ij, vec_ij );
            float new_weight    = dot( vec_is, vec_ij ) / lensq_ij;
            new_weight          = 1.0 - new_weight;
            new_weight          = clamp(new_weight, 0.0, 1.0 );
            
            weight              = min(weight, new_weight);
        }
       
        weights[i]          = weight;
        total_weight        += weight;
    }
    
    for( int i = 0; i < POINT_COUNT; ++i )
    {
        weights[i] = weights[i] / total_weight;
    }
}

// Function 163
vec2 sample_biquadratic_gradient_approx(sampler2D channel, vec2 res, vec2 uv) {
    vec2 q = fract(uv * res);
    vec2 cc = 0.5 / res;
    vec2 ww0 = uv - cc;
    vec2 ww1 = uv + cc;
    float nx = texture(channel, vec2(ww1.x, uv.y)).r - texture(channel, vec2(ww0.x, uv.y)).r;
    float ny = texture(channel, vec2(uv.x, ww1.y)).r - texture(channel, vec2(uv.x, ww0.y)).r;
	return vec2(nx, ny);
}

// Function 164
float getSampleDim4(int sampleIndex,vec2 fragCoord)
{
	return fract(getDimensionHash(4,fragCoord) + radicalInverse(sampleIndex, 11));
}

// Function 165
vec4 sample3D(sampler2D tex, vec3 uvw, vec3 vres)
{
    uvw = mod(floor(uvw * iVResolution), iVResolution);
    float idx = (uvw.z * (iVResolution.x*iVResolution.y)) + (uvw.y * iVResolution.x) + uvw.x;
    vec2 uv = vec2(mod(idx, iResolution.x), floor(idx / iResolution.x));
    
    return texture(tex, (uv + 0.5) / iResolution.xy);
}

// Function 166
vec4 sample_biquadratic_exact(sampler2D channel, vec2 res, vec2 uv) {
    vec2 q = fract(uv * res);
    ivec2 t = ivec2(uv * res);
    ivec3 e = ivec3(-1, 0, 1);
    vec4 s00 = texelFetch(channel, t + e.xx, 0);
    vec4 s01 = texelFetch(channel, t + e.xy, 0);
    vec4 s02 = texelFetch(channel, t + e.xz, 0);
    vec4 s12 = texelFetch(channel, t + e.yz, 0);
    vec4 s11 = texelFetch(channel, t + e.yy, 0);
    vec4 s10 = texelFetch(channel, t + e.yx, 0);
    vec4 s20 = texelFetch(channel, t + e.zx, 0);
    vec4 s21 = texelFetch(channel, t + e.zy, 0);
    vec4 s22 = texelFetch(channel, t + e.zz, 0);    
    vec2 q0 = (q+1.0)/2.0;
    vec2 q1 = q/2.0;	
    vec4 x0 = mix(mix(s00, s01, q0.y), mix(s01, s02, q1.y), q.y);
    vec4 x1 = mix(mix(s10, s11, q0.y), mix(s11, s12, q1.y), q.y);
    vec4 x2 = mix(mix(s20, s21, q0.y), mix(s21, s22, q1.y), q.y);    
	return mix(mix(x0, x1, q0.x), mix(x1, x2, q1.x), q.x);
}

// Function 167
void sampleStorage(vec2 p, out float lod, out bool inShape)
{
    vec4 t = T(p, 0);
    inShape = t.x > .5;
    lod = t.y;
}

// Function 168
float sampleLightSourcePdf(in vec3 x, vec3 ns, in vec3 wi, float d, float cosAtLight) {
    vec3 s = light.pos - vec3(1., 0., 0.) * light.size.x * 0.5 -
        				 vec3(0., 0., 1.) * light.size.y * 0.5;
    vec3 ex = vec3(light.size.x, 0., 0.);
    vec3 ey = vec3(0., 0., light.size.y);
    
    SphQuad squad;
    SphQuadInit(s, ex, ey, x, squad);
    return 1. / squad.S;
}

// Function 169
void Sample_GGX_R(vec2 s, vec3 V, vec3 N, float alpha, vec3 F0, out vec3 L, out vec3 w)
{
    float l = rsqrt((alpha*alpha)/s.y + 1.0 - (alpha*alpha));
    
    vec3 H = Sample_Sphere(s.x * 2.0 - 1.0, l, N);

    L = 2.0 * dot(V, H) * H - V;
    
    float HoV = clamp01(dot(H, V));
    float NoV = clamp01(dot(N, V));
    float NoL = clamp01(dot(N, L));
    float NoH = clamp01(dot(N, H));

    vec3  F = FresnelSchlick(HoV, F0);  
    float G = GGX_G(NoV, NoL, alpha);
    
    float denom = NoV * NoH;
    
    w = denom == 0.0 ? vec3(0.0) : F * G * HoV / denom;
}

// Function 170
float sampleLinear(float u, float a, float b) {
    return (sqrt(mix(a*a, b*b, u)) - a) / (b - a);
}

// Function 171
vec3 sampleReflectionMap(vec3 p, float b) {
    vec3 col = textureLod(reflectTex, p, b).rgb;
    
    // fake HDR
    //col *= 1.0 + 1.0 * smoothstep(0.5, 1.0, dot(LUMA, col));
    
    return col;
}

// Function 172
float sampleWt( float wt, bool even )
{
    return even ? (2.-wt) : wt;
}

// Function 173
float sampleHeightfield(vec2 p)
{
    float h = 	textureLod(iChannel0, p / 40. + iTime / 400., 2.).b *
    			textureLod(iChannel1, p / 8., 2.).r * 1.6;
    
    return clamp(h, 0., 1. - 1e-4) * maxHeight;
}

// Function 174
float sampleLightSourcePdf( in vec3 x,
                            in vec3 wi,
                           	in float d,
                            in float cosAtLight ) {
    float sph_r2 = objects[0].params_[1];
    vec3 sph_p = toVec3( objects[0].transform_*vec4(vec3(0.0,0.0,0.0), 1.0) );
    float solidangle;
    vec3 w = sph_p - x;			//direction to light center
	float dc_2 = dot(w, w);		//squared distance to light center
    float dc = sqrt(dc_2);		//distance to light center
    
    if( dc_2 > sph_r2 ) {
    	float sin_theta_max_2 = clamp( sph_r2 / dc_2, 0.0, 1.0);
		float cos_theta_max = sqrt( 1.0 - sin_theta_max_2 );
    	solidangle = TWO_PI * (1.0 - cos_theta_max);
    } else { 
    	solidangle = FOUR_PI;
    }
    
    return 1.0/solidangle;
}

// Function 175
float SampleNoiseFractal(vec3 p)
{
    float h = noisedFractal(p).x;
    h += noisedFractal(p*2. + 100.).x * 0.5;
    h += noisedFractal(p*4. - 100.).x * 0.25;
    h += noisedFractal(p*8. + 1000.).x * 0.125;
    return h / (1.865);
}

// Function 176
vec4 createTexture( in vec2 p )
{
    vec2 cc = vec2( -0.1, 0.68 );

	vec4 dmin = vec4(1000.0);
    float w = 0.0;
    vec2 z = 1.1*(-1.0 + 2.0*p)*vec2(iChannelResolution[0].x/iChannelResolution[0].y,1.0);
    for( int i=0; i<80; i++ )
    {
        z = cc + vec2( z.x*z.x - z.y*z.y, 2.0*z.x*z.y );

		dmin=min(dmin, vec4(length( z-0.5), 
							abs(-0.5+z.x + 0.2*sin(5.0*z.y)), 
							dot(z,z),
						    length( fract(z/8.0)-0.5) ) );
        if( dot(z,z)>4.0 ) w=1.0;
    }
   
    vec3 col = vec3(0.6,0.6,0.6);
    col * 0.4+0.6*w;
    col *= mix( vec3(1.0,0.45,0.1), vec3(1.0), w );
    col *= 0.65 + dmin.w;
    col = mix( col, 1.5*vec3(0.7,0.7,0.7),1.0-clamp(dmin.y*15.0,0.0,1.0) );
    col = mix( col, vec3(1.1,1.1,1.0),1.0-clamp(dmin.x*2.0,0.0,1.0) );
	col *= 0.5 + 0.5*clamp(dmin.z*50.0,0.0,1.0);


    return vec4( col, 1.0 );
}

// Function 177
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
    return result / samples;
#else
    return vec3(1.0);
#endif
}

// Function 178
float checkerTexture(in vec2 pos)
{
    float ind = 1.0;
    
    if(mod(pos.x, 1.0) > 0.5)
        ind *= -1.0;
    
    if(mod(pos.y, 1.0) > 0.5)
        ind *= -1.0;
    
    ind = max(ind, 0.0);
    
    return ind;
}

// Function 179
vec3 sampleHemisphere(float u1, float u2, vec3 normal)
{
	vec3 u = normal;
	vec3 v = abs(u.y) < abs(u.z) ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
	vec3 w = normalize(cross(u, v));
	v = cross(w, u);

	float r = sqrt(u1);
	float theta = 2.0 * 3.1415926535 * u2;
	float x = r * cos(theta);
	float y = r * sin(theta);
	return normalize(u * sqrt(1.0 - u1) + v * x + w * y);
}

// Function 180
vec3 sampleBlinn( in vec3 N, in vec3 E, in float roughness, in float r1, in float r2, out float pdf ) {
    float cosTheta = pow( r1, 1.0/( roughness ) );
    float phi = r2*TWO_PI;
    float theta = acos( cosTheta );
    vec3 H = localToWorld( sphericalToCartesian( 1.0, phi, theta ), N );
    float dotNH = dot(H,N);
    vec3 L = reflect( E*(-1.0), H );
    
    pdf = pdfBlinn(N, E, L, roughness );
    
    return L;
}

// Function 181
void disneyDiffuseSample(out vec3 wi, const in vec3 wo, out float pdf, const in vec2 u, const in vec3 normal, const in MaterialInfo material) {
    cosineSample
}

// Function 182
vec4 sampleLevel0( vec2 uv )
{
    return texture( iChannel0, uv, -10.0 );
}

// Function 183
void sample_BSDF(in Intersection inter, inout DirectionSample ds){
    float n = inter.material.shininess;
    bool is_lambert = n==0.0f;
    vec3 r = float(is_lambert) * inter.normal + 
        	 float(!is_lambert) * reflection(-inter.ray.direction, inter.normal);
    
    ds.direction = randomDirectionHemisphere(r, max(n,1.0f));
    float cosr = dot(r, ds.direction);
    float correct_r = float(cosr > 0.0f) - float(cosr <= 0.0f);
    
    cosr *= correct_r;
    ds.direction *= correct_r;
    
    float coswo = dot(inter.normal, ds.direction);
    bool ok = coswo > 0.0f;
    float cosr_pow_n = pow( abs(cosr),n);
	
	float w = (n + 1.0f + float(is_lambert))/ TWO_PI;
    	
    ds.pdf = w * (float(is_lambert) * coswo + 
                  float(!is_lambert) * cosr_pow_n); 
    ds.bsdf = float(ok) * w * inter.material.albedo * cosr_pow_n ;
}

// Function 184
void sampleSphereSA(vec3 viewer, in Sphere sphere, inout SurfaceLightSample sls){
    // get costheta and phi
    vec3 main_direction = (viewer - sphere.position);
    float d = length(main_direction);
    main_direction /= d;
    float d2 = d*d;
    float sinthetamax = sphere.radius /d;
    
    //float thetamax = asin(sinthetamax);
    float costhetamax = sqrt(1.0f - sinthetamax * sinthetamax);//cos(thetamax);
    
    float costheta = 1.0f - rand1()  * (1.0f - costhetamax);
    
    float sintheta = sqrt(1.0 - costheta * costheta);//sin(acos(costheta))
    float phi = rand1() * TWO_PI;
    
    // D = 1 - d² sin² θ / r²
    float sintheta2 =  sintheta * sintheta;
    float D = 1.0 - d2 * sintheta2 / sphere.radius2;
    bool D_positive = D > 0.0f;
    
    float cosalpha = float(D_positive) * (sintheta2 / sinthetamax +  costheta * sqrt(abs(D)))
        			+float(!D_positive) * sinthetamax;
    
    float sinalpha = sin(acos(cosalpha));//sqrt(1.0 - cosalpha * cosalpha);

    vec3 direction = vec3(sinalpha * cos(phi), sinalpha * sin(phi), cosalpha);
    if(abs(main_direction.z) > 0.99999f){
        sls.normal = direction * sign(main_direction.z);
    }
    else{
        vec3 axis = normalize(cross(UP, main_direction));
        float angle = acos(main_direction.z);

        sls.normal = rotate(axis, angle, direction);
    }
    sls.point = sphere.position + sphere.radius * sls.normal;
    float solid_angle = TWO_PI * (1.0 - costhetamax);
    sls.pdf = 1.0f / solid_angle;
}

// Function 185
bool sampleDistComp(vec2 uv, float radius_)
{
    float threshold = 0.005;
    float radius = 0.1 + radius_;
    
    return sqrt(dot(uv, uv)) < radius*(1.0+threshold) && sqrt(dot(uv, uv)) > radius * (1.0-threshold);
}

// Function 186
float gridTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )
{
	// filter kernel
    vec2 w = max(abs(ddx), abs(ddy)) + 0.01;

	// analytic (box) filtering
    vec2 a = p + 0.5*w;                        
    vec2 b = p - 0.5*w;           
    vec2 i = (floor(a)+min(fract(a)*N,1.0)-
              floor(b)-min(fract(b)*N,1.0))/(N*w);
    //pattern
    return (1.0-i.x)*(1.0-i.y);
}

// Function 187
vec3 sampleHemisphereCosWeighted(in vec2 xi) {
#ifdef CONCENTRIC_DISK
    vec2 xy = concentricSampleDisk(xi);
    float r2 = xy.x*xy.x + xy.y*xy.y;
    return vec3(xy, sqrt(max(0.0, 1.0 - r2)));
#else
    float theta = acos(sqrt(1.0-xi.x));
    float phi = TWO_PI * xi.y;
    return sphericalToCartesian( 1.0, phi, theta );
#endif
}

// Function 188
vec3 sampleTextureWithFilter( in vec3 uvw, in vec3 ddx_uvw, in vec3 ddy_uvw, in vec3 nor, in float mid )
{
    int sx = 1 + int( clamp( 4.0*length(ddx_uvw-uvw), 0.0, float(MaxSamples-1) ) );
    int sy = 1 + int( clamp( 4.0*length(ddy_uvw-uvw), 0.0, float(MaxSamples-1) ) );

	vec3 no = vec3(0.0);

	#if 1
    for( int j=0; j<MaxSamples; j++ )
    for( int i=0; i<MaxSamples; i++ )
    {
        if( j<sy && i<sx )
        {
            vec2 st = vec2( float(i), float(j) ) / vec2( float(sx),float(sy) );
            no += mytexture( uvw + st.x*(ddx_uvw-uvw) + st.y*(ddy_uvw-uvw), nor, mid );
        }
    }
    #else
    for( int j=0; j<sy; j++ )
    for( int i=0; i<sx; i++ )
    {
        vec2 st = vec2( float(i), float(j) )/vec2(float(sx),float(sy));
        no += mytexture( uvw + st.x * (ddx_uvw-uvw) + st.y*(ddy_uvw-uvw), nor, mid );
    }
    #endif		

	return no / float(sx*sy);
}

// Function 189
vec4 textureSky(vec2 uv) {
    const vec2 RES = vec2(8.0, 32.0);    
    float n = noise1(uv * RES);
    n = n * 0.05 + 0.8;
    return vec4(0.5,n*1.0,n*1.1,1.0);
}

// Function 190
void Sample_ScatteredDir(vec2 s0, vec2 s1, float s2, inout vec3 rd, inout vec3 W, vec3 N, vec3 albedo, float roughness, vec3 F0)
{
    float alpha = GGXAlphaFromRoughness(roughness);
    
    vec3 V = -rd;

    vec3 L0, w0;
    {
        vec3 L = Sample_ClampedCosineLobe(s0.x * 2.0 - 1.0, s0.y, N);
        
        vec3 H = normalize(V + L);
        
        float HoV = clamp01(dot(H, V));
    	float NoV = clamp01(dot(N, V));
    	float NoL = clamp01(dot(N, L));
        
    	w0 = albedo * DisneyDiffuse_BRDF(NoV, NoL, HoV, roughness) * pi;
        L0 = L;
    }
    
    vec3 L1, w1;
    {
        vec3 L, w;
    	Sample_GGX_R(s1, V, N, alpha, F0, /*out*/ L1, /*out*/ w1);
    }

    float w0s = dot(w0, vec3(0.2, 0.7, 0.1));
    float w1s = dot(w1, vec3(0.2, 0.7, 0.1));
    
    if(w0s == 0.0 && w1s == 0.0)
    {
        W = vec3(0.0);
        rd = L0;
        
        return;
    }
    
    #if 0
    w0s = 0.5;
    w1s = 1.0 - w0s;
    #elif 0
    float wn = (w0s*w0s) / ((w0s*w0s) + (w1s*w1s));
    #else
    float wn = w0s / (w0s + w1s);
	#endif
    
    bool doUseSmpl0 = s2 <= wn;

    float denom = doUseSmpl0 ? wn : (1.0 - wn);

    rd = doUseSmpl0 ? L0 : L1;
    W *= doUseSmpl0 ? w0 : w1;

    W /= denom == 0.0 ? 1.0 : denom;
}

// Function 191
float SampleDigit(const in float n, const in vec2 vUV)
{		
	if(vUV.x  < 0.0) return 0.0;
	if(vUV.y  < 0.0) return 0.0;
	if(vUV.x >= 1.0) return 0.0;
	if(vUV.y >= 1.0) return 0.0;
	
	float data = 0.0;
	
	     if(n < 0.5) data = 7.0 + 5.0*16.0 + 5.0*256.0 + 5.0*4096.0 + 7.0*65536.0;
	else if(n < 1.5) data = 2.0 + 2.0*16.0 + 2.0*256.0 + 2.0*4096.0 + 2.0*65536.0;
	else if(n < 2.5) data = 7.0 + 1.0*16.0 + 7.0*256.0 + 4.0*4096.0 + 7.0*65536.0;
	else if(n < 3.5) data = 7.0 + 4.0*16.0 + 7.0*256.0 + 4.0*4096.0 + 7.0*65536.0;
	else if(n < 4.5) data = 4.0 + 7.0*16.0 + 5.0*256.0 + 1.0*4096.0 + 1.0*65536.0;
	else if(n < 5.5) data = 7.0 + 4.0*16.0 + 7.0*256.0 + 1.0*4096.0 + 7.0*65536.0;
	else if(n < 6.5) data = 7.0 + 5.0*16.0 + 7.0*256.0 + 1.0*4096.0 + 7.0*65536.0;
	else if(n < 7.5) data = 4.0 + 4.0*16.0 + 4.0*256.0 + 4.0*4096.0 + 7.0*65536.0;
	else if(n < 8.5) data = 7.0 + 5.0*16.0 + 7.0*256.0 + 5.0*4096.0 + 7.0*65536.0;
	else if(n < 9.5) data = 7.0 + 4.0*16.0 + 7.0*256.0 + 5.0*4096.0 + 7.0*65536.0;
	
	vec2 vPixel = floor(vUV * vec2(4.0, 5.0));
	float fIndex = vPixel.x + (vPixel.y * 4.0);
	
	return mod(floor(data / exp2(fIndex)), 2.0);
}

// Function 192
vec4 SampleCharacterTex( uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vUV = (vec2(iChPos) + vCharUV) / 16.0f;
    return textureLod( iChannelFont, vUV, 0.0 );
}

// Function 193
vec4 SampleMip0(vec3 sp) {
    sp.y=sp.y-0.5; float fy=floor(sp.y);
    vec2 cuv1=vec2(sp.x+floor(fy*0.2)*64.,sp.z+mod(fy,5.)*64.);
    vec2 cuv2=vec2(sp.x+floor((fy+1.)*0.2)*64.,sp.z+mod(fy+1.,5.)*64.);
    return mix(texture(iChannel1,cuv1*IRES),
               texture(iChannel1,cuv2*IRES),fract(sp.y));
}

// Function 194
vec4 SampleMip5(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*4.,sp.z+60.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(2.,0.))*IRES),fract(sp.y));
}

// Function 195
vec4 SampleMip1(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*32.,sp.z);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(32.,0.))*IRES),fract(sp.y));
}

// Function 196
vec3 LinearSample0(vec2 uv, out float coeff, out vec3 Moments, vec3 CN) {
    vec4 Attr=texture(iChannel0,(uv+vec2(0.,HRES.y))*IRES);
    if (Attr.w>9999. || Box2(uv,HRES)>0.) {coeff=0.001; Moments=vec3(0.); return vec3(0.); }
    coeff=max(0.001,dot(CN,(Read3(Attr.y)*2.-1.)*I09));
    vec4 Light=texture(iChannel1,(uv+vec2(0.,HRES.y))*IRES);
    Moments=Read3(Light.w)*coeff;
    return Light.xyz*coeff;
}

// Function 197
float3 BSDF_Sample ( float3 N, float3 wi, float3 P, Material mat, out float pdf,
                     inout float seed) {
  if ( mat.transmittive > 0.0 ) {
    pdf = 1.0f;
    return refract(wi, N, mat.transmittive);
  }
    if (mat.diffuse == 0.0) { pdf = 1.0f; return reflect(wi, N); }
  float diff_chance = Sample_Uniform(seed);
  if ( diff_chance < mat.diffuse ) {
    return Sample_Cos_Hemisphere(wi, N, pdf, seed);
  }
  float2 xi = Sample_Uniform2(seed);
  float k = mat.alpha*mat.alpha;
  float phi   = TAU * xi.x,
        theta = asin( sqrt( ( k*log(1.0-xi.y) )/( k*log(1.0-xi.y)-1.0 )));
  float3 wo = Reorient_Hemisphere(normalize(To_Cartesian(theta, phi)), N); 
  pdf = PDF_Cosine_Hemisphere(wi, N);
  return wo;
}

// Function 198
vec3 SamplePixel(vec2 scPix, out vec2 scrP, float time)
{
    scrP = 2.0*scPix/iResolution.xy - vec2(1.0);
	scrP.x *= iResolution.x/iResolution.y;
	
	vec3 camP = getCameraPos(time);
	
	// Look-at
	vec3 trgP      = vec3( 0, 0, 0);
	vec3 upV       = vec3( 0.0, 1.0, 0.0 );
	vec3 camV      = normalize( trgP - camP );
	vec3 camRightV = cross( upV, camV );
	vec3 camUpV    = cross( camV, camRightV );
	
	vec3 rayV = normalize( camV + camRightV*scrP.x + camUpV*scrP.y );
	
	vec3 normal;
	vec3 pos = castScene( camP, rayV, normal, time );
	
	return shadeSphere( normal, pos, rayV, time ); 
}

// Function 199
vec4 textureHermite(sampler2D tex, vec2 uv, vec2 res)
{
	uv = uv*res + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*(3.0-2.0*fuv);
	uv = (uv - 0.5)/res;
	return texture( tex, uv );
}

// Function 200
vec4 texture0( in vec2 x )
{
    //return texture( iChannel0, x );
    vec2 res = iChannelResolution[0].xy;
    vec2 u = x*res - 0.5;
    vec2 p = floor(u);
    vec2 f = fract(u);
    f = f*f*(3.0-2.0*f);    
    vec4 a = texture( iChannel0, (p+vec2(0.5,0.5))/res, -64.0 );
	vec4 b = texture( iChannel0, (p+vec2(1.5,0.5))/res, -64.0 );
	vec4 c = texture( iChannel0, (p+vec2(0.5,1.5))/res, -64.0 );
	vec4 d = texture( iChannel0, (p+vec2(1.5,1.5))/res, -64.0 );
    return mix(mix(a,b,f.x), mix(c,d,f.x),f.y);
}

// Function 201
vec3 Sample_Uniform3(inout float seed) {
    return fract(sin(vec3(seed+=0.1,seed+=0.1,seed+=0.1))*
                 vec3(43758.5453123,22578.1459123,842582.632592));
}

// Function 202
vec4 sample2D(sampler2D sampler,vec2 resolution, vec2 uv)
{
    return texture(sampler, uv / resolution);
}

// Function 203
vec2 sampleRay(vec3 ro, vec3 rd)
{
    vec2 energy = vec2(0);
    vec2 spectrum = vec2(1.);
    vec2 mats = vec2(.9, .5) * .99;
    vec3 lo = vec3(.7, .8, .1), ls = vec3(1, 0, 0) * .2, lt = vec3(0, 0, 1) * .2;
    vec3 ln = normalize(cross(ls, lt));
    float lightArea = length(ls) * length(lt);
    float lightRadiance = 50.0;

    for(int i = 0; i < 3; ++i)
    {
        vec3 n, p0, p1, uvw;
        vec2 res = traceScene(ro, rd, n, uvw);
        vec3 rp = ro + rd * res.x;
        if(res.x < 0. || res.x > 1e3)
            break;

        float t = dot(lo - ro, ln) / dot(rd, ln);
        if(t > 0. && t < res.x && dot(rd, ln) < 0.)
        {
            vec3 rp = ro + rd * t;
            vec2 uv = vec2(dot(rp - lo, ls) / dot(ls, ls), dot(rp - lo, lt) / dot(lt, lt));
            if(abs(uv.x) < 1. && abs(uv.y) < 1.)
            {
                energy += spectrum * lightRadiance;
            }
        }

        float fr = mix(0.001, 1.0, pow(1. - clamp(dot(-rd, n), 0., 1.), 3.));

        vec3 absuvw = abs(uvw);
        vec2 uv = absuvw.x > absuvw.y ? (absuvw.x > absuvw.z ? uvw.yz : uvw.xy) : (absuvw.y > absuvw.z ? uvw.xz : uvw.xy);

            if(res.y < .5)
            {
                // No intersection.
                break;
            }
        else if(res.y < 1.5)
        {
            // Diffuse box 1.
            if(rp.y > .99)
                spectrum *= .5;
            else
                spectrum *= mats;
            vec2 dc = texture(iChannel0,uvw).rg/float(iFrame)*12.;
            dc *= mix(.3, 1., textureLod(iChannel2, uv / 2., 1.).r);
            if(rp.y > .999 && abs(rp.x - -.3) < .6 && abs(rp.z - .1) < .8)
            {
                energy += spectrum * vec2(1.3, 1.).yx * mix(.7, 1., textureLod(iChannel1, rd, 2.).r);
                fr = 1.;
            }
            else
            {
                energy += spectrum * dc * (1. - fr);
            }
        }
        else if(res.y < 2.5)
        {
            // Diffuse box 2.
            spectrum *= mats.yx;
            vec2 dc = texture(iChannel0,uvw).ba / float(iFrame) * 12.;
            dc *= mix(.3, 1., textureLod(iChannel3, uv / 2., 1.).b);
            energy += spectrum * dc * (1. - fr);
        }
        else if(res.y < 3.5)
        {
            // Mirror box.
            fr = 1.;
        	fr *= mix(1., .25, pow(textureLod(iChannel2, rp.zx, 1.).r, 3.));
        }
        spectrum *= .9 * fr;
        ro = rp + n * 5e-3;
        rd = reflect(rd, n);
        
        if(max(spectrum.x, spectrum.y) < 1e-4)
            break;
    }
    return energy;
}

// Function 204
vec4 galaxyTexture(vec2 uv){
    
    // Cut if original uv is not in [0,1]
    bool out_of_tex = abs(uv.x-.5)>.5 || abs(uv.y-.5)>.5;
    
    // Spiral mesh
    vec4 mesh = texture(bufAChannel,uv);
    
    // Galaxy photo distortion to get a nice, not distorted spiral
    vec2 screen_ratio = vec2((R.x/R.y)/(640./360.),1);
    vec2 img_compr = GALAXY_DISTORTION*screen_ratio;
    uv = (uv-.5)*img_compr + GALAXY_CENTER;
    
    vec3 color = mix(texture(galaxyTexChannel,uv).rgb,
                     mesh.rgb,
                     DRAW_MESH ? mesh.a : 0.);
    
    out_of_tex = out_of_tex
          || abs(uv.x-.5)>.5 || abs(uv.y-.5)>.5 // Cut if new uv is not in [0,1]
          || (GALAXY==M51 && uv.x>.700); // Cut the satellite galaxy on M51
    
    return vec4(color, !out_of_tex);
}

// Function 205
float SampleDigit(const in float n, const in vec2 vUV)
{		
	if(vUV.x  < 0.0) return 0.0;
	if(vUV.y  < 0.0) return 0.0;
	if(vUV.x >= 1.0) return 0.0;
	if(vUV.y >= 1.0) return 0.0;
	
	float data = 0.0;
	
	     if(n < 0.5) data = 7.0 + 5.0*16.0 + 5.0*256.0 + 5.0*4096.0 + 7.0*65536.0;
	else if(n < 1.5) data = 2.0 + 2.0*16.0 + 2.0*256.0 + 2.0*4096.0 + 2.0*65536.0;
	else if(n < 2.5) data = 7.0 + 1.0*16.0 + 7.0*256.0 + 4.0*4096.0 + 7.0*65536.0;
	else if(n < 3.5) data = 7.0 + 4.0*16.0 + 7.0*256.0 + 4.0*4096.0 + 7.0*65536.0;
	else if(n < 4.5) data = 4.0 + 7.0*16.0 + 5.0*256.0 + 1.0*4096.0 + 1.0*65536.0;
	else if(n < 5.5) data = 7.0 + 4.0*16.0 + 7.0*256.0 + 1.0*4096.0 + 7.0*65536.0;
	else if(n < 6.5) data = 7.0 + 5.0*16.0 + 7.0*256.0 + 1.0*4096.0 + 7.0*65536.0;
	else if(n < 7.5) data = 4.0 + 4.0*16.0 + 4.0*256.0 + 4.0*4096.0 + 7.0*65536.0;
	else if(n < 8.5) data = 7.0 + 5.0*16.0 + 7.0*256.0 + 5.0*4096.0 + 7.0*65536.0;
	else if(n < 9.5) data = 7.0 + 4.0*16.0 + 7.0*256.0 + 5.0*4096.0 + 7.0*65536.0;
	
	vec2 vPixel = floor(vUV * vec2(4.0, 5.0));
	float fIndex = vPixel.x + (vPixel.y * 4.0);
	
	return mod(floor(data / pow(2.0, fIndex)), 2.0);
}

// Function 206
vec3 Sample(float idx)
{
    idx = floor(idx);
    vec2 uv = IdxtoUV(idx, float(IMGRES));
    vec3 c = textureLod(iChannel0, uv / float(IMGRES),0.0).rgb;
    return floor(c * (MAX_RGB-1.0));
}

// Function 207
vec3 sampletex( vec2 uv )
{
    #ifdef SRGBLIN
    	return srgb2lin( texture( iChannel0, uv, -10.0 ).rgb );
    #else
    	return  texture( iChannel0, uv, -10.0 ).rgb ;
    #endif
}

// Function 208
bool sampleDist(vec2 uv)
{
    float radius = 0.1;
    return sqrt(dot(uv, uv)) < radius;
}

// Function 209
vec3 sampleDisp(vec2 uv, vec2 dispNorm, float disp) {
    vec3 col = vec3(0);
    const float SD = 1.0 / float(SAMPLES);
    float wl = 0.0;
    vec3 denom = vec3(0);
    for(int i = 0; i < SAMPLES; i++) {
        vec3 sw = sampleWeights(wl);
        denom += sw;
        col += sw * texture(iChannel1, uv + dispNorm * disp * wl).xyz;
        wl  += SD;
    }
    
    // For a large enough number of samples,
    // the return below is equivalent to 3.0 * col * SD;
    return col / denom;
}

// Function 210
vec4 textureNearestAA (sampler2D smp, int smpi, vec2 uv)
{
	vec2 span = fwidth(uv);
	vec2 hspan = span / 2.0;
	vec4 uva = vec4(uv - hspan, uv + hspan);

	vec2 ss = sign(span);
	vec2 fmul = ss / (1.0 - ss + span);  // 1.0/span or 0.0
	vec2 size = textureSizef(smpi);
	vec2 f = min((uva.zw - floor(uva.zw * size)/size) * fmul, 1.0);

	return mix(
		mix(textureNearest(smp, smpi, uva.xy), textureNearest(smp, smpi, uva.zy), f.x),
		mix(textureNearest(smp, smpi, uva.xw), textureNearest(smp, smpi, uva.zw), f.x), f.y
	);
}

// Function 211
vec3 sampleHemisphereCosWeighted( in vec3 n, in float Xi1, in float Xi2 ) {
    float theta = acos(sqrt(1.0-Xi1));
    float phi = TWO_PI * Xi2;

    return localToWorld( sphericalToCartesian( 1.0, phi, theta ), n );
}

// Function 212
vec4 blurTexture(sampler2D sampler, vec2 uv, float minMipmap, float maxMipmap)
{
 	vec4 sumCol = vec4(0.0);
    
    //adding different mipmaps
    for (float i = minMipmap; i < maxMipmap; i++)
    {
     	sumCol += texture(sampler, uv, i);   
    }
    
    //average
    return sumCol / (maxMipmap - minMipmap);
}

// Function 213
vec3 sampleLightType( const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, out float visibility, float seed, const in MaterialInfo material) {
    if( !light.enabled )
        return vec3(0.);
    
    if( light.type == LIGHT_TYPE_SPHERE ) {
        vec3 L = lightSample(light, interaction, wi, lightPdf, seed, material);
        vec3 shadowRayDir =normalize(light.position - interaction.point);
        visibility = visibilityTest(interaction.point + shadowRayDir * .01, shadowRayDir);
        return L;
    }
    else if( light.type == LIGHT_TYPE_SUN ) {
        vec3 L = sampleSun(light, interaction, wi, lightPdf, seed);
        visibility = visibilityTestSun(interaction.point + wi * .01, wi);
        return L;
    }
    else {
        return vec3(0.);
    }
}

// Function 214
vec2 sampleRay(vec3 ro, vec3 rd)
{
    vec2 energy = vec2(0);
    vec2 spectrum = vec2(1.);

    vec3 ln = normalize(cross(ls, lt));
    float lightArea = length(ls) * length(lt);

    for(int i = 0; i < 3; ++i)
    {
        vec3 n, p0, p1, uvw;
        vec2 res = traceScene(ro, rd, n, uvw);
        vec3 rp = ro + rd * res.x;
        
        if(res.x < 0. || res.x > 1e3)
            break;
        
        vec3 lrd = lambertNoTangent(n, vec2(rand(), rand()));
        
        if(res.y < .5)
        {
            // No intersection.
            break;
        }
        else if(res.y < 1.5)
        {
            // Diffuse box 1.
            if(rp.y > .999)
                spectrum *= .5;
            else
                spectrum *= mats;
            ro = rp + n * 1e-4;
            rd = lrd;
        }
        else if(res.y < 2.5)
        {
            // Diffuse box 2.
            spectrum *= mats.yx;
            ro = rp + n * 1e-4;
            rd = lrd;
        }
        else if(res.y < 3.5)
        {
            // Mirror box.
            spectrum *= .9;
            ro = rp + n * 1e-4;
            rd = reflect(rd, n);
        }
        
        
        if(res.y < 2.5)
        {
            // For diffuse materials, sample lights directly.
            energy += spectrum * sampleLight(rp, n) * lightRadiance * lightArea;
        }
        else
        {
            // Test for intersection with the parallelogram lightsource.
            float t = dot(lo - ro, ln) / dot(rd, ln);
            if(t > 0.)
            {
                vec3 rp = ro + rd * t;
                vec2 uv = vec2(dot(rp - lo, ls) / dot(ls, ls), dot(rp - lo, lt) / dot(lt, lt));
                if(abs(uv.x) < 1. && abs(uv.y) < 1.)
                {
                    energy += spectrum * lightRadiance;
                }
            }
        }
    }
    return energy;
}

// Function 215
float get_texture_index(in sampler2D s)
{
    return texelFetch(s, CTRL_TEXTURE, 0).w;
}

// Function 216
vec3 samplef2(vec2 position) {
	float d = sample_biquadratic(iChannel0, iChannelResolution[0].xy, position).r;
    vec2 n = sample_biquadratic_gradient_approx(iChannel0, iChannelResolution[0].xy, position);
    return vec3(n, d);
}

// Function 217
float3 Sample_Uniform_Cone ( float lobe, out float pdf, inout float seed ) {
  float2 u = Sample_Uniform2(seed);
  float phi = TAU*u.x,
        cos_theta = 1.0 - u.y*(1.0 - cos(lobe));
  pdf = PDF_Cone(lobe);
  return To_Cartesian(cos_theta, phi);
}

// Function 218
vec4 texture2( sampler2D sam, vec2 uv )
{
#ifndef SHOW_DERIVATIVES    
    return texture(sam,uv);
#else    
    float res = float(textureSize(sam,0).x);
    uv = uv*res - 0.5;
    vec2 iuv = floor(uv);
    vec2 f = fract(uv);
	vec4 rg1 = textureLod( sam, (iuv+ vec2(0.5,0.5))/res, 0.0 );
	vec4 rg2 = textureLod( sam, (iuv+ vec2(1.5,0.5))/res, 0.0 );
	vec4 rg3 = textureLod( sam, (iuv+ vec2(0.5,1.5))/res, 0.0 );
	vec4 rg4 = textureLod( sam, (iuv+ vec2(1.5,1.5))/res, 0.0 );
	return mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
#endif    
}

// Function 219
float noiseTexture(vec2 p, float t)
{
    p = .5*p + t;
    float val = tex(p);
    //return val;
    
    if keypress(32)
    {
        val *= .5;
        val += tex(p*2.) *.25;
        val += tex(p*4.) *.125;
        val += tex(p*8.) *.0625;
    }
    
    return val;
}

// Function 220
vec3 getSample(vec3 ro, vec3 rd, float t, float densityIntegral, float density, bool type){
    
    vec3 p = ro+rd*t;
    vec3 rdL, color;
    vec3 light = vec3(0);
    float maxLt, lightDensityIntegral;
    if (type){
        rdL = normalize(light-p);
        maxLt = distance(light,p);
        lightDensityIntegral = transmittanceBetween(p, light);
        color = vec3(0,1,1)*10.;
    }else{
        rdL = normalize(vec3(-1));
        maxLt = 1.;
        lightDensityIntegral = 0.5+transmittanceBetween(p, p-rdL*10.);
        color = vec3(1,1,1)*130.;
    }
    
    //lightDensityIntegral = 1.;
    
    float schlickSquare = 1. - schlickK*dot(rd,rdL);
	float schlickMie = (1. - schlickK*schlickK)/(4.*PI*schlickSquare*schlickSquare);
    
    return schlickMie * scattering * density * exp(-1.* densityIntegral *(absorbtion+scattering))
            *exp(-1.* lightDensityIntegral *(absorbtion+scattering))/maxLt/maxLt*color;
}

// Function 221
vec4 textureBorderClamp(sampler2D sampler, vec2 uv, vec2 size, vec4 borderColor, bvec2 wrap)
{
#if 0
    // UNTESTED: to support mipmapping, we should be able to do this:
    float lod = textureQueryLod(sampler, uv).y; // OpImageQueryLod
    vec2 size = textureSize(sampler, lod); // OpImageQuerySizeLod
    vec4 ret = textureLod(sampler, uv, lod); // OpImageSampleExplicitLod
#else
    vec4 ret = texture(sampler, uv); // OpImageSampleImplicitLod
#endif
    vec2 limit = vec2(1.0, 0.5); // 0.5 for GL_CLAMP, 1.0 for GL_CLAMP_TO_BORDER
	vec2 factors = clamp(0.5 + (abs(uv - 0.5) - 0.5) * size, vec2(0.0), limit);
    if (wrap.x)
    	ret = mix(ret, borderColor, factors.x);

    if (wrap.y)
        ret = mix(ret, borderColor, factors.y);
	return ret;
}

// Function 222
vec4 myTexture(vec2 uv) {
    
    vec2 res = iChannelResolution[0].xy;
    uv = uv*res + 0.5;
    
    // tweak fractionnal value of the texture coordinate
    vec2 fl = floor(uv);
    vec2 fr = fract(uv);
    vec2 aa = fwidth(uv)*0.75;
    fr = smoothstep( vec2(0.5)-aa, vec2(0.5)+aa, fr);
    
    uv = (fl+fr-0.5) / res;
    return texture(iChannel0, uv);
    
}

// Function 223
vec4 leavesTexture(vec3 p, vec3 nor)
{
    
    vec3 rand = texCube(iChannel2, p*.15,nor);
	vec3 mat = vec3(0.4,1.2,0) *rand;
   	return vec4(mat, .0);
}

// Function 224
vec3 sample_light( SurfaceInfo surface,
                   MaterialInfo material,
                   vec4 light,
                 out float pdf )
{
    vec2 u12 = hash21(material.seed);
    
    vec3 tangent = vec3(0.), binormal = vec3(0.);
    vec3 ldir = normalize(light.xyz - surface.point);
    calc_binormals(ldir, tangent, binormal);
    
    float sinThetaMax2 = light.w * light.w / dist_squared(light.xyz, surface.point);
    float cosThetaMax = sqrt(max(0., 1. - sinThetaMax2));
    vec3 light_sample = uniform_sample_cone(u12, cosThetaMax, tangent, binormal, ldir);
    
    pdf = -1.;
    if (dot(light_sample, surface.normal) > 0.)
    {
        pdf = 1. / (TWO_PI * (1. - cosThetaMax));
    }
    
    return light_sample;
    
}

// Function 225
vec3 uniformSampleCone(vec2 u12, float cosThetaMax, vec3 xbasis, vec3 ybasis, vec3 zbasis) {
    float cosTheta = (1. - u12.x) + u12.x * cosThetaMax;
    float sinTheta = sqrt(1. - cosTheta * cosTheta);
    float phi = u12.y * TWO_PI;
    vec3 samplev = sphericalDirection(sinTheta, cosTheta, sin(phi), cos(phi));
    return samplev.x * xbasis + samplev.y * ybasis + samplev.z * zbasis;
}

// Function 226
vec4 SampleCharacter( uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vClampedCharUV = clamp(vCharUV, vec2(0.01), vec2(0.99));
    vec2 vUV = (vec2(iChPos) + vClampedCharUV) / 16.0f;

    vec4 vSample;
    
    float l = length( (vClampedCharUV - vCharUV) );

#if 0
    // Simple but not efficient - samples texture for each character
    // Extends distance field beyond character boundary
    vSample = textureLod( FONT_SAMPLER, vUV, 0.0 );
    vSample.gb = vSample.gb * 2.0f - 1.0f;
    vSample.a -= 0.5f+1.0/256.0;    
    vSample.w += l * 0.75;
#else    
    // Skip texture sample when not in character boundary
    // Ok unless we have big shadows / outline / font weight
    if ( l > 0.01f )
    {
        vSample.rgb = vec3(0);
		vSample.w = 2000000.0; 
    }
    else
    {
		vSample = textureLod( FONT_SAMPLER, vUV, 0.0 );    
        vSample.gb = vSample.gb * 2.0f - 1.0f;
        vSample.a -= 0.5f + 1.0/256.0;    
    }
#endif    
        
    return vSample;
}

// Function 227
vec3 sampleCube(vec3 v)
{
	vec4 reflcol = texture (iChannel0, v);
	
	vec3 col = reflcol.rgb * vec3(reflcol.a);
	
	float Y = dot(vec3(0.30, 0.59, 0.11), col);
	float YD = exposure * (exposure/brightMax + 1.0) / (exposure + 1.0); 
	col *= YD;

	return col;
}

// Function 228
float escherTextureY(vec2 p)
{
    vec2 pp = vec2(mod(p.x, 2.0), mod(p.y-.5, 1.0)+0.5);
    
    float d = 1000.0;
    for(int i=19; i<vert.length()-1; ++i)
    	if(abs(verticalDistance(pp, vert[i], vert[i+1])) < abs(d))
            d=verticalDistance(pp, vert[i], vert[i+1]);
    

    pp = vec2(mod(p.x-1.0, 2.0), mod(p.y-.5, 1.0)+0.5);
    for(int i=19; i<vert.length()-1; ++i)
    	if(abs(verticalDistance(pp, vert[i], vert[i+1])) < abs(d)) 
            d=verticalDistance(pp, vert[i], vert[i+1]);
    
    float val = smoothstep(0.0, 1.0, d/0.05);
    val = d;
    
    if(mod(p.y-0.5, 2.0)-1.0 > 0.)
        val = -val;
    
    return val;
}

// Function 229
C_Sample SampleMaterial(const in vec2 vUV, sampler2D sampler,  const in vec2 vTextureSize, const in float fNormalScale)
{
	C_Sample result;
	
	vec2 vInvTextureSize = vec2(1.0) / vTextureSize;
	
	vec3 cSampleNegXNegY = texture(sampler, vUV + (vec2(-1.0, -1.0)) * vInvTextureSize.xy).rgb;
	vec3 cSampleZerXNegY = texture(sampler, vUV + (vec2( 0.0, -1.0)) * vInvTextureSize.xy).rgb;
	vec3 cSamplePosXNegY = texture(sampler, vUV + (vec2( 1.0, -1.0)) * vInvTextureSize.xy).rgb;
	
	vec3 cSampleNegXZerY = texture(sampler, vUV + (vec2(-1.0, 0.0)) * vInvTextureSize.xy).rgb;
	vec3 cSampleZerXZerY = texture(sampler, vUV + (vec2( 0.0, 0.0)) * vInvTextureSize.xy).rgb;
	vec3 cSamplePosXZerY = texture(sampler, vUV + (vec2( 1.0, 0.0)) * vInvTextureSize.xy).rgb;
	
	vec3 cSampleNegXPosY = texture(sampler, vUV + (vec2(-1.0,  1.0)) * vInvTextureSize.xy).rgb;
	vec3 cSampleZerXPosY = texture(sampler, vUV + (vec2( 0.0,  1.0)) * vInvTextureSize.xy).rgb;
	vec3 cSamplePosXPosY = texture(sampler, vUV + (vec2( 1.0,  1.0)) * vInvTextureSize.xy).rgb;

	// convert to linear	
	vec3 cLSampleNegXNegY = cSampleNegXNegY * cSampleNegXNegY;
	vec3 cLSampleZerXNegY = cSampleZerXNegY * cSampleZerXNegY;
	vec3 cLSamplePosXNegY = cSamplePosXNegY * cSamplePosXNegY;

	vec3 cLSampleNegXZerY = cSampleNegXZerY * cSampleNegXZerY;
	vec3 cLSampleZerXZerY = cSampleZerXZerY * cSampleZerXZerY;
	vec3 cLSamplePosXZerY = cSamplePosXZerY * cSamplePosXZerY;

	vec3 cLSampleNegXPosY = cSampleNegXPosY * cSampleNegXPosY;
	vec3 cLSampleZerXPosY = cSampleZerXPosY * cSampleZerXPosY;
	vec3 cLSamplePosXPosY = cSamplePosXPosY * cSamplePosXPosY;

	// Average samples to get albdeo colour
	result.vAlbedo = ( cLSampleNegXNegY + cLSampleZerXNegY + cLSamplePosXNegY 
		    	     + cLSampleNegXZerY + cLSampleZerXZerY + cLSamplePosXZerY
		    	     + cLSampleNegXPosY + cLSampleZerXPosY + cLSamplePosXPosY ) / 9.0;	
	
	vec3 vScale = vec3(0.3333);
	
	#ifdef USE_LINEAR_FOR_BUMPMAP
		
		float fSampleNegXNegY = dot(cLSampleNegXNegY, vScale);
		float fSampleZerXNegY = dot(cLSampleZerXNegY, vScale);
		float fSamplePosXNegY = dot(cLSamplePosXNegY, vScale);
		
		float fSampleNegXZerY = dot(cLSampleNegXZerY, vScale);
		float fSampleZerXZerY = dot(cLSampleZerXZerY, vScale);
		float fSamplePosXZerY = dot(cLSamplePosXZerY, vScale);
		
		float fSampleNegXPosY = dot(cLSampleNegXPosY, vScale);
		float fSampleZerXPosY = dot(cLSampleZerXPosY, vScale);
		float fSamplePosXPosY = dot(cLSamplePosXPosY, vScale);
	
	#else
	
		float fSampleNegXNegY = dot(cSampleNegXNegY, vScale);
		float fSampleZerXNegY = dot(cSampleZerXNegY, vScale);
		float fSamplePosXNegY = dot(cSamplePosXNegY, vScale);
		
		float fSampleNegXZerY = dot(cSampleNegXZerY, vScale);
		float fSampleZerXZerY = dot(cSampleZerXZerY, vScale);
		float fSamplePosXZerY = dot(cSamplePosXZerY, vScale);
		
		float fSampleNegXPosY = dot(cSampleNegXPosY, vScale);
		float fSampleZerXPosY = dot(cSampleZerXPosY, vScale);
		float fSamplePosXPosY = dot(cSamplePosXPosY, vScale);	
	
	#endif
	
	// Sobel operator - http://en.wikipedia.org/wiki/Sobel_operator
	
	vec2 vEdge;
	vEdge.x = (fSampleNegXNegY - fSamplePosXNegY) * 0.25 
			+ (fSampleNegXZerY - fSamplePosXZerY) * 0.5
			+ (fSampleNegXPosY - fSamplePosXPosY) * 0.25;

	vEdge.y = (fSampleNegXNegY - fSampleNegXPosY) * 0.25 
			+ (fSampleZerXNegY - fSampleZerXPosY) * 0.5
			+ (fSamplePosXNegY - fSamplePosXPosY) * 0.25;

	result.vNormal = normalize(vec3(vEdge * fNormalScale, 1.0));	
	
	return result;
}

// Function 230
float pencil_texture (vec2 u) {
    vec2 o = u;
    float col = watercolor(o*ro(+1.35)*10.);
    col *= pencil_hatching(u+vec2(2.15, 0.), vec3(90.,9., 8.));
    col *= pencil_hatching(u+vec2(3.15, 0.), vec3(80.,7., 8.));
    col += pencil_hatching(u+vec2(3.15, 0.), vec3(15.,7., 4.));
    col *= watercolor(o*ro(-2.95)*1.)*.7;
    col = mix(col, 1., 1.-texture(iChannel0, u*10.).x*2.);
    return col;
}

// Function 231
vec3 sample_bump(in vec2 uv, in vec3 L) {
    
    vec3 color = texture(iChannel0, uv).xyz;
    
    const vec2 eps = vec2(1.0/1024.0, 0.0);
    
    vec3 N = normalize(vec3(texture(iChannel0, uv - eps.xy).x - 
                       texture(iChannel0, uv + eps.xy).x,
                       texture(iChannel0, uv - eps.yx).x - 
                       texture(iChannel0, uv + eps.yx).x, 
                       0.1));
    
    color *= 0.3 + 0.7 * clamp(dot(N, L), 0.0, 1.0);    
    
    return color;
       
}

// Function 232
vec4 GetTextureSample(vec2 pos, float freq, float seed)
{
    vec3 hash = hash33(vec3(seed, 0.0, 0.0));
    float ang = hash.x * 2.0 * pi;
    mat2 rotation = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
    
    vec2 uv = rotation * pos * freq + hash.yz;
    return texture(iChannel0, uv);
}

// Function 233
vec4 sample_blured(vec2 uv, float radius, float gamma)
{
    vec4 pix = vec4(0.);
    float norm = 0.001;
    //weighted integration over mipmap levels
    for(float i = 0.; i < 5.; i += 0.5)
    {
        float k = weight(i, log2(1. + radius), gamma);
        pix += k*texture(iChannel0, uv, i); 
        norm += k;
    }
    //nomalize
    return pix/norm;
}

// Function 234
vec3 texture_terrain(vec3 p, vec3 nrm, bool is_edge)
{
    vec3 col = vec3(0.);
    vec3 stone = vec3(.6);
    vec3 grass = vec3(.4, .6, .4);
    vec3 snow = vec3(1.1, 1.1, 1.2)*4.;
    float stone_mask = min(max((dot(nrm, vec3(0,0,1))-.87), 0.)*8., 1.);
    float snow_mask = min(max((p.z-.14), 0.)*64., 1.);
    
    if(is_edge)//draw edge
    {
        float h = get_terrain(p.xy).w-p.z;
        stone_mask += 1.-min(max(h-.01, 0.)*64., 1.);
        
    }
    
    vec3 ground = mix(grass, snow, snow_mask);
    
    col = mix(stone, ground, stone_mask);

    
    return col;
}

// Function 235
vec4 GetScrollingTextureSample(vec2 pos, float freq, vec2 nodePoint, vec2 velocity)
{
    vec3 hash = hash33(vec3(nodePoint.xy, 0));
    float ang = hash.x * 2.0 * pi;
    mat2 rotation = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
    
    vec2 dir = normalize(velocity);
    mat2 flowMatrix = mat2(dir.x, dir.y, -dir.y, dir.x);
    mat2 flowStretch = mat2(2.0, 0.0, 0.0, 1.0);
    vec2 flowPos = flowStretch * (inverse(flowMatrix) * pos * freq + vec2(iTime, 0.0));
    vec2 uv = rotation * flowMatrix * flowPos + hash.yz;
    return texture(iChannel0, uv);
}

// Function 236
vec3 samplef(vec2 tc)
{
	return texture(iChannel0, tc).xyz;
}

// Function 237
vec3 getHemisphereCosineSample(vec3 n, out float weight) {
    float cosTheta2 = getRandom();
    float cosTheta = sqrt(cosTheta2);
    float sinTheta = sqrt(1. - cosTheta2);
    
    float phi = 2. * M_PI * getRandom();
    
    // Spherical to cartesian
    vec3 t = normalize(cross(n.yzx, n));
    vec3 b = cross(n, t);
    
	vec3 l = (t * cos(phi) + b * sin(phi)) * sinTheta + n * cosTheta;
    
    // Sample weight
    float pdf = (1. / M_PI) * cosTheta;
    weight = (.5 / M_PI) / (pdf + 1e-6);
    
    return l;
}

// Function 238
vec3 sampleSphereUniform(vec2 uv)
{
	float cosTheta = 2.0*uv.x - 1.0;
	float phi = 2.0*PI*uv.y;
	return unitVecFromPhiCosTheta(phi, cosTheta);
}

// Function 239
void sampleScattering(
	float u,
	float maxDistance,
	out float dist,
	out float pdf)
{
	// remap u to account for finite max distance
	float minU = exp(-SIGMA*maxDistance);
	float a = u*(1.0 - minU) + minU;

	// sample with pdf proportional to exp(-sig*d)
	dist = -log(a)/SIGMA;
	pdf = SIGMA*a/(1.0 - minU);
}

// Function 240
vec3 sampleTextureWithFilter( in vec3 uvw, in vec3 ddx_uvw, in vec3 ddy_uvw, in float detail)
{
    int sx = 1 + int( clamp( detail*length(ddx_uvw-uvw), 0.0, float(MaxSamples-1) ) );
    int sy = 1 + int( clamp( detail*length(ddy_uvw-uvw), 0.0, float(MaxSamples-1) ) );

	vec3 no = vec3(0.0);

    for( int j=0; j<sy; j++ )
    for( int i=0; i<sx; i++ )
    {
        vec2 st = vec2( float(i), float(j) )/vec2(float(sx),float(sy));
        //filtering something using a step() function is a real problem. To be addressed later
        no += getWaterAlbedo( uvw + st.x * (ddx_uvw-uvw) + st.y*(ddy_uvw-uvw)).xyz;
        no += lavaFloorAlbedo( uvw + st.x * (ddx_uvw-uvw) + st.y*(ddy_uvw-uvw)).xyz;
    }

	return no / pow(float(sx*sy),2.);
}

// Function 241
vec4 sampleWithMitchellNetravali(sampler2D sampler, vec2 samplerSize,
                                 vec2 stepxy, vec2 uv)
{
        vec2 texel = 1.0 / samplerSize;

        vec2 texelPos = samplerSize * uv;
        vec2 bottomLeftTexelPos = floor(texelPos - vec2(0.5)) + vec2(0.5);

        vec4 xpos = vec4(
                            (bottomLeftTexelPos.x - 1.0) * texel.x,
                            (bottomLeftTexelPos.x + 0.0) * texel.x,
                            (bottomLeftTexelPos.x + 1.0) * texel.x,
                            (bottomLeftTexelPos.x + 2.0) * texel.x
                    );

        vec4 ypos = vec4(
                            (bottomLeftTexelPos.y - 1.0) * texel.y,
                            (bottomLeftTexelPos.y + 0.0) * texel.y,
                            (bottomLeftTexelPos.y + 1.0) * texel.y,
                            (bottomLeftTexelPos.y + 2.0) * texel.y
                    );

        vec2 f = texelPos - bottomLeftTexelPos;
        if (f.x >= 1.0 || f.y >= 1.0 || f.x < 0.0 || f.y < 0.0) {
                return vec4(1.0, 0.0, 0.0, 0.0);
        }

        vec2 speed = min(vec2(1.0), texel / stepxy);
        vec4 linetaps = vec4(mitchellNetravali(speed.x*(-1.0 - f.x)),
                             mitchellNetravali(speed.x*(0.0-f.x)),
                             mitchellNetravali(speed.x*(1.0-f.x)),
                             mitchellNetravali(speed.x*(2.0-f.x))
                            );
        linetaps /= dot(linetaps, vec4(1.0));
        vec4 columntaps = vec4(mitchellNetravali(speed.y*(-1.0 - f.y)),
                               mitchellNetravali(speed.y*(0.0-f.y)),
                               mitchellNetravali(speed.y*(1.0-f.y)),
                               mitchellNetravali(speed.y*(2.0-f.y))
                              );
        columntaps /= dot(columntaps, vec4(1.0));

        return kernel4(sampler, xpos, linetaps, ypos, columntaps);
}

// Function 242
void SphQuadSample(in vec3 x, SphQuad squad, float u, float v, out LightSamplingRecord sampleRec) {
    // 1. compute ’cu’
    float au = u * squad.S + squad.k;
    float fu = (cos(au) * squad.b0 - squad.b1) / sin(au);
    float cu = 1./sqrt(fu*fu + squad.b0sq) * (fu>0. ? +1. : -1.);
    cu = clamp(cu, -1., 1.); // avoid NaNs
    // 2. compute ’xu’
    float xu = -(cu * squad.z0) / sqrt(1. - cu*cu);
    xu = clamp(xu, squad.x0, squad.x1); // avoid Infs
    // 3. compute ’yv’
    float d = sqrt(xu*xu + squad.z0sq);
    float h0 = squad.y0 / sqrt(d*d + squad.y0sq);
    float h1 = squad.y1 / sqrt(d*d + squad.y1sq);
    float hv = h0 + v * (h1-h0), hv2 = hv*hv;
    float yv = (hv2 < 1.-EPSILON) ? (hv*d)/sqrt(1.-hv2) : squad.y1;
    // 4. transform (xu,yv,z0) to world coords
    
    vec3 p = (squad.o + xu*squad.x + yv*squad.y + squad.z0*squad.z);
    sampleRec.w = p - x;
    sampleRec.d = length(sampleRec.w);
    sampleRec.w = normalize(sampleRec.w);
    sampleRec.pdf = 1. / squad.S;
}

// Function 243
vec4 sampleWithLanczos3Interpolation(sampler2D sampler, vec2 samplerSize,
                                     vec2 stepxy, vec2 uv)
{
        vec2 texel = 1.0 / samplerSize;
        vec2 texelPos = uv / texel;
        vec2 bottomLeftTexelPos = floor(texelPos - vec2(0.5)) + vec2(0.5);

        vec3 x0_2 = vec3(
                            (bottomLeftTexelPos.x - 2.0) * texel.x,
                            (bottomLeftTexelPos.x - 1.0) * texel.x,
                            (bottomLeftTexelPos.x + 0.0) * texel.x
                    );
        vec3 x3_5 = vec3(
                            (bottomLeftTexelPos.x + 1.0) * texel.x,
                            (bottomLeftTexelPos.x + 2.0) * texel.x,
                            (bottomLeftTexelPos.x + 3.0) * texel.x
                    );

        vec3 y0_2 = vec3(
                            (bottomLeftTexelPos.y - 2.0) * texel.y,
                            (bottomLeftTexelPos.y - 1.0) * texel.y,
                            (bottomLeftTexelPos.y + 0.0) * texel.y
                    );
        vec3 y3_5 = vec3(
                            (bottomLeftTexelPos.y + 1.0) * texel.y,
                            (bottomLeftTexelPos.y + 2.0) * texel.y,
                            (bottomLeftTexelPos.y + 3.0) * texel.y
                    );

        vec2 f = texelPos - bottomLeftTexelPos;
        vec2 speed = min(vec2(1.0), texel / stepxy);
        vec3 ltaps0_2 = vec3(
                                lanczos3(speed.x*(-2.0 - f.x)),
                                lanczos3(speed.x*(-1.0 - f.x)),
                                lanczos3(speed.x*(0.0 - f.x))
                        );
        vec3 ltaps3_5 = vec3(
                                lanczos3(speed.x*(1.0 - f.x)),
                                lanczos3(speed.x*(2.0 - f.x)),
                                lanczos3(speed.x*(3.0 - f.x))
                        );
        float lsum = dot(ltaps0_2, vec3(1)) + dot(ltaps3_5, vec3(1));

        ltaps0_2 /= lsum;
        ltaps3_5 /= lsum;

        vec3 coltaps0_2 = vec3(
                                  lanczos3(speed.y*(-2.0 - f.y)),
                                  lanczos3(speed.y*(-1.0 - f.y)),
                                  lanczos3(speed.y*( 0.0 - f.y))
                          );
        vec3 coltaps3_5 = vec3(
                                  lanczos3(speed.y*(1.0 - f.y)),
                                  lanczos3(speed.y*(2.0 - f.y)),
                                  lanczos3(speed.y*(3.0 - f.y))
                          );
        float csum = dot(coltaps0_2, vec3(1.0)) + dot(coltaps3_5, vec3(1.0));

        coltaps0_2 /= csum;
        coltaps3_5 /= csum;

        return kernel3(sampler, x0_2, ltaps0_2, y0_2, coltaps0_2) +
               kernel3(sampler, x3_5, ltaps3_5, y0_2, coltaps0_2) +
               kernel3(sampler, x0_2, ltaps0_2, y3_5, coltaps3_5) +
               kernel3(sampler, x3_5, ltaps3_5, y3_5, coltaps3_5);
}

// Function 244
vec3 Sample_Sphere(float s0, float s1, vec3 normal)
{	 
    vec3 sph = Sample_Sphere(s0, s1);

    vec3 ox, oz;
    OrthonormalBasisRH(normal, ox, oz);

    return (ox * sph.x) + (normal * sph.y) + (oz * sph.z);
}

// Function 245
vec3 getCueTexture(vec2 uv) {
    vec3 wood = texture( iChannel1, uv.yx ).xyz;
    
    if(uv.y > 1.0) {
        return wood;
    } else {
    	float k = fract(uv.x / 0.2);
    	float h = 0.3;
    	float a = mix(-1.0, 1.0, float(k < h));
    	return (a*(h-k)*0.3 < uv.y-h)?wood:vec3(0.01);
    }
}

// Function 246
void sampleDirectLight( vec3 pos,
                       	vec3 normal,
                        float Xi1,
                        float Xi2, 
                       	out vec3 dir,
                       	out float pdf ) {
    float height = objects[0].params_[2] - objects[0].params_[1];
    float r = objects[0].params_[0];
    float pdfA;
    float d2;
    float aCosThere;
    float theta;
    float thetaPdf;
    float h;
    float hPdf;
    
    //convert position to object space
    pos = toVec3( objects[0].transform_inv_*vec4(pos, 1.0) );
    normal = toVec3( objects[0].transform_inv_*vec4(normal, 0.0) );
    
    vec3 v1 = vec3(objects[0].params_[0], objects[0].params_[1], 0.0);
    vec3 v2 = vec3(objects[0].params_[2], objects[0].params_[3], 0.0);
    vec3 v3 = vec3(objects[0].params_[4], objects[0].params_[5], 0.0);
    vec3 n = vec3(0.0, 0.0, 1.0);
    
    if(samplingTechnique == SAMPLE_TOTAL_AREA){
        vec3 p = uniformPointWitinTriangle( v1, v2, v3, Xi1, Xi2 );
        float triangleArea = length(cross(v1-v2,v3-v2)) * 0.5;
        pdfA = 1.0/triangleArea;
        
        dir = p - pos;
        d2 = dot(dir,dir);
        dir /= sqrt(d2);
        aCosThere = max(0.0,dot(-dir,n));
        pdf = PdfAtoW( pdfA, d2, aCosThere );
    } else {
        vec3 A = normalize(v1 - pos);
        vec3 B = normalize(v2 - pos);
        vec3 C = normalize(v3 - pos);
        sampleSphericalTriangle(A, B, C, Xi1, Xi2, dir, pdf);
        if(dot(-dir,n) < 0.0){
            pdf = 0.0;
        }
    }
    
    //convert dir to world space
    dir = toVec3( objects[0].transform_*vec4(dir,0.0) );
}

// Function 247
vec4 texture_Bilinear( sampler2D tex, vec2 t )
{
    vec2 res = iChannelResolution[0].xy;
    vec2 p = res*t - 0.5;
    vec2 f = fract(p);
    vec2 i = floor(p);

    return lerp( f.y, lerp( f.x, SAM(0,0), SAM(1,0)),
                      lerp( f.x, SAM(0,1), SAM(1,1)) );
}

// Function 248
vec3 sample_map(float x, float w) {
    float q0 = map(x - w);
    float q1 = map(x);
    float q2 = map(x + w);
    return vec3((q0+q1)*0.5,q1,(q1+q2)*0.5);
}

// Function 249
vec3 Sample_SphLight_MIS(vec2 s0, vec2 s1, vec3 V, vec3 p, vec3 N, vec3 albedo, float roughness, vec3 F0)
{
    float alpha = GGXAlphaFromRoughness(roughness);
    
    float ct; vec3 Lc, L0; float sang;
    Sample_SolidAngle(s0, p, LightPos, R2, /*out*/ ct, /*out*/ Lc, /*out*/ L0, /*out*/ sang);
    float pdf00 = 1.0/sang;

    vec3 L1; vec3 f1; float pdf11;
    Sample_GGX_R(s1, V, N, alpha, F0, /*out*/ L1, /*out*/ f1, /*out*/ pdf11);

    bool couldL1HitLight = dot(L1, Lc) > ct;
    
    vec3 f0 = Frostbite_R(V, N, L0, albedo, roughness, F0);
         f1 = Frostbite_R(V, N, L1, albedo, roughness, F0);

    float pdf01 = couldL1HitLight ? pdf00 : 0.0;
    float pdf10 = EvalPDF_GGX_R(V, N, L0, alpha);

    float w0, w1;
    #if 1
    w0 = (pdf00) / (Pow2(pdf00) + Pow2(pdf10));
    w1 = (pdf11) / (Pow2(pdf11) + Pow2(pdf01));        
    #else
    w0 = 1.0 / (pdf00 + pdf23);
    w1 = 1.0 / (pdf11 + pdf32);
    #endif

    float t2; vec3 n2; vec3 a2; bool isLight2 = true;
    bool hit2 = Intersect_Scene(p, L0, false, /*out*/ t2, n2, a2, isLight2);

    float t3; vec3 n3; vec3 a3; bool isLight3 = true;
    bool hit3 = Intersect_Scene(p, L1, false, /*out*/ t3, n3, a3, isLight3);

    if((isLight2 == false && t2 < dot(LightPos-p, Lc)) || dot(N, L0) <= 0.0) f0 = vec3(0.0);
    if(couldL1HitLight == false || isLight3 == false) f1 = vec3(0.0);

    vec3 res  = pdf00 == 0.0 ? vec3(0.0) : f0 * w0;
         res += pdf11 == 0.0 ? vec3(0.0) : f1 * w1;

    return res * Radiance;       
}

// Function 250
vec3 sampleLambertian( in vec3 N, in float r1, in float r2, out float pdf ){
    vec3 L = sampleHemisphereCosWeighted( N, r1, r2 );
    pdf = pdfLambertian(N, L);
    return L;
}

// Function 251
vec3 ImportanceSample(
    in vec2  xi,
    in float roughness,
    in vec3  surfNorm)
{
	float a = (roughness * roughness);
    
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - (cosTheta * cosTheta));
    
    vec3 H = vec3((sinTheta * cos(phi)), (sinTheta * sin(phi)), cosTheta);
    
    vec3 upVector = (0.999 > surfNorm.z) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 tangentX = normalize(cross(upVector, surfNorm));
    vec3 tangentY = cross(surfNorm, tangentX);
    
    return ((tangentX * H.x) + (tangentY * H.y) + (surfNorm * H.z));
}

// Function 252
float xorTexture( in vec2 pos )
{
    float xor = 0.0;
    for( int i=0; i<8; i++ )
    {
        xor += mod( floor(pos.x)+floor(pos.y), 2.0 );

        pos *= 0.5;
        xor *= 0.5;
    }
    return xor;
}

// Function 253
vec3 ImportanceSampleGGX(vec3 d,vec3 V,float roughness,out vec3 H)
{
    roughness =max(0.04,roughness);
    vec2 rand = hash2();
    float phi = 6.28318530718*rand.x;
    float xiy = rand.y;
	float a = roughness * roughness;
	float CosTheta = sqrt((1.0 - xiy) / (1.0 + (a*a - 1.0) * xiy));
	float SinTheta = sqrt(1.0 - CosTheta * CosTheta);
	H = vec3(SinTheta * cos(phi),SinTheta * sin(phi), CosTheta);
	vec3 w = (d);
	vec3 u = (cross(w.yzx, w));
	vec3 v = cross(w, u);
	H = v * H.x + u * H.y + w * H.z;
	vec3 R = H*2.0 * dot(V,H)  - V;
    return normalize(R);
}

// Function 254
vec3 textureAVG(samplerCube tex, vec3 tc) {
    const float diff0 = 0.35;
    const float diff1 = 0.12;
 	vec3 s0 = texture(tex,tc).xyz;
    vec3 s1 = texture(tex,tc+vec3(diff0)).xyz;
    vec3 s2 = texture(tex,tc+vec3(-diff0)).xyz;
    vec3 s3 = texture(tex,tc+vec3(-diff0,diff0,-diff0)).xyz;
    vec3 s4 = texture(tex,tc+vec3(diff0,-diff0,diff0)).xyz;
    
    vec3 s5 = texture(tex,tc+vec3(diff1)).xyz;
    vec3 s6 = texture(tex,tc+vec3(-diff1)).xyz;
    vec3 s7 = texture(tex,tc+vec3(-diff1,diff1,-diff1)).xyz;
    vec3 s8 = texture(tex,tc+vec3(diff1,-diff1,diff1)).xyz;
    
    return (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) * 0.111111111;
}

// Function 255
vec3 disneyMicrofacetSample(out vec3 wi, const in vec3 wo, out float pdf, const in vec2 u, const in SurfaceInteraction interaction, const in MaterialInfo material) {
    float cosTheta = 0., phi = (2. * PI) * u[1];
    float alpha = material.roughness * material.roughness;
    float tanTheta2 = alpha * alpha * u[0] / (1.0 - u[0]);
    cosTheta = 1. / sqrt(1. + tanTheta2);
    
    float sinTheta = sqrt(max(EPSILON, 1. - cosTheta * cosTheta));
    vec3 whLocal = sphericalDirection(sinTheta, cosTheta, sin(phi), cos(phi));
     
    vec3 tangent = vec3(0.), binormal = vec3(0.);
    createBasis(interaction.normal, tangent, binormal);
    
    vec3 wh = whLocal.x * tangent + whLocal.y * binormal + whLocal.z * interaction.normal;
    
    if(!sameHemiSphere(wo, wh, interaction.normal)) {
       wh *= -1.;
    }
            
    wi = reflect(-wo, wh);
    
    float NdotL = dot(interaction.normal, wo);
    float NdotV = dot(interaction.normal, wi);

    if (NdotL < 0. || NdotV < 0.) {
        pdf = 0.; // If not set to 0 here, create's artifacts. WHY EVEN IF SET OUTSIDE??
        return vec3(0.);
    }
    
    vec3 H = normalize(wo+wi);
    float NdotH = dot(interaction.normal,H);
    float LdotH = dot(wo,H);
    
    pdf = pdfMicrofacet(wi, wo, interaction, material);
    return disneyMicrofacetIsotropic(NdotL, NdotV, NdotH, LdotH, material);
}

// Function 256
vec4 texture_bicubic(sampler2D tex, vec2 uv, vec4 texelSize)
{
	uv = uv*texelSize.zw + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );

    float g0x = g0(fuv.x);
    float g1x = g1(fuv.x);
    float h0x = h0(fuv.x);
    float h1x = h1(fuv.x);
    float h0y = h0(fuv.y);
    float h1y = h1(fuv.y);

	vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - 0.5) * texelSize.xy;
	vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - 0.5) * texelSize.xy;
	vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - 0.5) * texelSize.xy;
	vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - 0.5) * texelSize.xy;
	
    return g0(fuv.y) * (g0x * texture(tex, p0)  +
                        g1x * texture(tex, p1)) +
           g1(fuv.y) * (g0x * texture(tex, p2)  +
                        g1x * texture(tex, p3));
}

// Function 257
vec2 sampleScene(vec3 p, vec3 n)
{
    vec2 energy = vec2(0);

    vec3 ln = normalize(cross(ls, lt));
    float lightArea = length(ls) * length(lt);
    int count = 1;
    for(int i = 0; i < count; ++i)
    {
        vec3 ro = p + n * 1e-4;
        vec3 rd = lambertNoTangent(n, vec2(rand(), rand()));

        energy += sampleRay(ro, rd);
        energy += sampleLight(ro, n) * lightRadiance * lightArea;
    }

    energy /= float(count);
    return energy;
}

// Function 258
float texture(vec2 uv )
{
	float t = voronoi( uv * 8.0 + vec2(iTime) );
    t *= 1.0-length(uv * 2.0);
	
	return t;
}

// Function 259
vec4 textureQuadratic( in sampler3D sam, in vec3 p )
{
    float texSize = float(textureSize(sam,0).x); 

	p = p*texSize;
	vec3 i = floor(p);
	vec3 f = fract(p);
	p = i + f*0.5;
	p = p/texSize;
    
	float w = 0.5/texSize;

	return mix(mix(mix(texture3(sam,p+vec3(0,0,0)),
                       texture3(sam,p+vec3(w,0,0)),f.x),
                   mix(texture3(sam,p+vec3(0,w,0)),
                       texture3(sam,p+vec3(w,w,0)),f.x), f.y),
               mix(mix(texture3(sam,p+vec3(0,0,w)),
                       texture3(sam,p+vec3(w,0,w)),f.x),
                   mix(texture3(sam,p+vec3(0,w,w)),
                       texture3(sam,p+vec3(w,w,w)),f.x), f.y), f.z);
}

// Function 260
vec3 texture_wood(vec3 pos) {
    pos = quat_mul(quat(vec3(1,0,0),-0.0), pos);
    //pos.z -= 1.0;
    vec2 core = vec2(cos(pos.z), sin(pos.z))*0.1;
    pos.xy -= core;

    float r = length(pos.xy);
    float a = (TAU/2.0 + atan(pos.x,pos.y)) / TAU;

    float r_noise = noise(vec2(cos(a*TAU*2.0), sin(a*TAU*2.0)));
    r_noise += noise(vec2(10.0) + vec2(cos(a*TAU*4.0), sin(a*TAU*4.0))) * 0.5; // squigglyness
    r_noise += noise(vec2(100.0) + vec2(cos(a*TAU*8.0), sin(a*TAU*8.0))) * 0.4; // squigglyness
    r_noise += noise(vec2(1000.0) + vec2(cos(a*TAU*16.0), sin(a*TAU*16.0))) * 0.2; // squigglyness

    r_noise += noise(pos.z*0.5)*3.0; // knottyness

    r_noise *= noise(r*3.0)*5.0; // whorlyness
    r += r_noise*0.05*clamp(r,0.0,1.0); // scale and reduce at center

    vec3 col = vec3(1.0,0.65,0.35);
    //float c = 0.5 + 0.5*sin(r*100.0); // 100 rings per meter ~ 1cm rings
    float c = fract(r*5.0);
    //c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, (c-0.15)/0.85);
    c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, sqrt(clamp((c-0.15)/0.85,0.0,1.0)));
    //c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, pow(clamp((c-0.15)/0.85,0.0,1.0), 0.25));
    col = mix(col, vec3(0.4,0.1,0.0), c); // ring gradient
    col = mix(col, col*vec3(0.8, 0.5, 0.5), noise(r*20.0)); // ring-to-ring brightness

    return col;
}

// Function 261
void Sample_SolidAngle(vec2 s, vec3 p, vec3 lp, float lr2, 
                       out float ct, out vec3 Lc, out vec3 L, out float sang)
{
    vec3 lvec = lp - p;
    
   	float len2 = dot(lvec, lvec);
    
    if(len2 == 0.0)
    {
        ct = 0.0;
        Lc = vec3(0.0, 1.0, 0.0);
        L  = vec3(0.0, 1.0, 0.0);
        sang = pi2;
        
        return;
    }
    
	float rlen = rsqrt(len2);

    Lc = lvec * rlen;
    
    ct = sqrt(clamp01(1.0 - lr2 * (rlen * rlen)));
    
    L = Sample_Sphere(s.x * 2.0 - 1.0, mix(ct, 1.0, s.y), Lc);

    sang = ct * -pi2 + pi2;
}

// Function 262
vec4 SampleCharacter( sampler2D sFontSampler, uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vClampedCharUV = clamp(vCharUV, vec2(0.01), vec2(0.99));
    vec2 vUV = (vec2(iChPos) + vClampedCharUV) / 16.0f;

    vec4 vSample;
    
    float l = length( (vClampedCharUV - vCharUV) );

    // Skip texture sample when not in character boundary
    // Ok unless we have big font weight
    if ( l > 0.01f )
    {
        vSample.rgb = vec3(0);
		vSample.w = 2000000.0; 
    }
    else
    {
		vSample = textureLod( sFontSampler, vUV, 0.0 );    
        vSample.gb = vSample.gb * 2.0f - 1.0f;
        vSample.a -= 0.5f + 1.0/256.0;    
    }
        
    return vSample;
}

// Function 263
vec3 sampleBlur(sampler2D tex, vec2 uv, vec2 k)
{
    vec2 s = 1.0 / vec2(textureSize(tex, 0));
    vec3 avg = vec3(0.0);
    vec2 hk = k * 0.5;
    for(float x = 0.0 - hk.x; x < hk.x; x += 1.0)
    {
        for(float y = 0.0 - hk.y; y < hk.y; y += 1.0)
            avg += texture(tex, uv + s * vec2(x,y)).rgb;
    }
    return avg / (k.x * k.y);
}

// Function 264
vec4 sample_biquadratic(sampler2D channel, vec2 res, vec2 uv) {
    vec2 q = fract(uv * res);
    vec2 c = (q*(q - 1.0) + 0.5) / res;
    vec2 w0 = uv - c;
    vec2 w1 = uv + c;
    vec4 s = texture(channel, vec2(w0.x, w0.y))
    	   + texture(channel, vec2(w0.x, w1.y))
    	   + texture(channel, vec2(w1.x, w0.y))
    	   + texture(channel, vec2(w1.x, w1.y));
	return s / 4.0;
}

// Function 265
void sampleSphereHemisphereUniform(vec3 viewer, in Sphere sphere, inout SurfaceLightSample sls){
    
    vec3 main_direction = normalize(viewer - sphere.position);
    sls.normal = randomDirectionHemisphere(main_direction, 0.0f);
   	if(dot(sls.normal, main_direction) < 0.0f) sls.normal *= -1.0f;
    
    sls.point = sphere.position + (sphere.radius ) * sls.normal;
    
    
    sls.pdf = 1.0f / (TWO_PI * sphere.radius2);
}

// Function 266
float MAT_scratchTexture(vec2 p)
{
    const float squareWidth = 0.10*2.0;
    const float moveAmp   = squareWidth*0.75;
    const float lineWidth = 0.0005;
    float repeatInterval = squareWidth+moveAmp;
    repeatInfo rInfo = UTIL_repeat(p,repeatInterval);
    float margin = repeatInterval-squareWidth;
    
    vec2 a = moveAmp*noise(rInfo.anchor);
    vec2 b = -moveAmp*noise(rInfo.anchor+10.0);
    float dseg = 1000.0*UTIL_distanceToLineSeg(rInfo.pRepeated, a, b)/squareWidth;
    return saturate(10.0/dseg-0.5)*0.25;
}

// Function 267
vec2 sample_dist_smart(vec2 uv, float font_size) {
        
#ifdef HIGH_QUALITY
    const int nstep = 4;
    const float w[4] = float[4](1., 2., 2., 1.);
#else
    const int nstep = 3;
    const float w[3] = float[3](1., 2., 1.);
#endif
    
    vec2  dsum = vec2(0.);
    float wsum = 0.;
    
    for (int i=0; i<nstep; ++i) {
        
        float ui = float(i)/float(nstep-1);
                
        for (int j=0; j<nstep; ++j) {
            
            float uj = float(j)/float(nstep-1);
            
            vec2 delta = (-1.  + 2.*vec2(ui,uj))/TEX_RES;
            
            vec3 grad_dist = sample_grad_dist(uv-delta, font_size);
            vec2 pdelta = delta * GLYPHS_PER_UV * font_size;
            
            float dline = grad_dist.z + dot(grad_dist.xy, pdelta);
               
            float wij = w[i]*w[j];
            
            dsum += wij * vec2(dline, grad_dist.z);
            wsum += wij;

        }
    }
    
    return dsum / wsum;
    
}

// Function 268
void bumpNormalTexture ( inout Ray ray, in float size, in sampler2D tex, in vec2 texture_size) {
	vec3 uaxis = normalize(cross(normalize(abs(ray.direction)), ray.normal));
	vec3 vaxis = normalize(cross(uaxis, ray.normal));
	mat3 space = mat3( uaxis, vaxis, ray.normal );

	float A = texture2D(tex, ray.object.uv - vec2(0,0) / texture_size).r;
	float B = texture2D(tex, ray.object.uv - vec2(1,0) / texture_size).r;
    float C = texture2D(tex, ray.object.uv - vec2(0,1) / texture_size).r;
	ray.normal = normalize(vec3(B - A, C - A, clamp(1.- size, 0.,1.)));
	ray.normal = normalize(space * ray.normal);
}

// Function 269
vec3 samplef(vec2 uv) {
    vec2 suv = uv / 4.0;
    vec2 n = floor(suv);
    vec2 f = fract(suv);
    
    ivec2 iuv = ivec2(n + 0.5);
    mat3 p;
    for (int i = 0; i <= 2; ++i) {
        for (int j = 0; j <= 2; ++j) {
            float col = fetch(iuv + ivec2(i-1,j-1));
            p[j][i] = col;
        }
    }
    
    return interpolate2d_grad(p, f);
}

// Function 270
vec3 GetBridgeTexture(RayHit marchResult)
{
  vec3 checkPos = TranslateBridge(marchResult.hitPos); 
  vec3 woodTexture = vec3(BoxMap(iChannel1, vec3(checkPos.z*0.01, checkPos.yx*0.31), (marchResult.normal), 0.5).r);
  vec3 bridgeColor =  woodTexture*(0.6+(0.4*noise(checkPos.zx*17.)));
  float n = noise2D(checkPos.xz*1.3);
  return mix(bridgeColor*MOSSCOLOR2, bridgeColor, smoothstep(-.64-(2.*n), 2.269-(2.*n), marchResult.hitPos.y));
}

// Function 271
vec4 previousSample(vec4 hit){
    vec2 prevUv = pos2uv(getCam(iTime-iTimeDelta), hit.xyz);
    vec2 prevFragCoord = prevUv * iResolution.y + iResolution.xy/2.0;
    
    vec2 pfc, finalpfc;
    float dist, finaldist = MaxDist;
    for(int x = -1; x <= 1; x++){
        for(int y = -1; y <= 1; y++){
            pfc = prevFragCoord + PixelCheckDistance*vec2(x, y);
            dist = distancePixel(pfc, hit);
            if(dist < finaldist){
                finalpfc = pfc;
                finaldist = dist;
            }
    	}
    }
    
    Camera cam = getCam(iTime);
    if(finaldist < PixelAcceptance*length(hit.xyz-cam.pos)/cam.focalLength/iResolution.y)
        return texture(iChannel0, finalpfc/iResolution.xy);
    return vec4(0.);
}

// Function 272
vec4 fetchTexture (vec2 uvCoord, vec2 textureRes)
{
    vec2 fetchCoord = uvCoord * textureRes;
    vec2 fetchFract = fract (fetchCoord);
    vec4 fetch00 = texelFetch (iChannel0, ivec2 (fetchCoord.xy), 0);
    vec4 fetch10 = texelFetch (iChannel0, ivec2 (fetchCoord.xy) + ivec2 (1, 0), 0);
    vec4 fetch01 = texelFetch (iChannel0, ivec2 (fetchCoord.xy) + ivec2 (0, 1), 0);
    vec4 fetch11 = texelFetch (iChannel0, ivec2 (fetchCoord.xy) + ivec2 (1, 1), 0);
    
    return mix (mix (fetch00, fetch10, fetchFract.x), mix (fetch01, fetch11, fetchFract.x), fetchFract.y);
}

// Function 273
float escherTextureContour(vec2 p, float linewidth, float pixel_size)
{
    vec2 pp = mod(p,1.0);
    
    float d = 10000000.0;
    for(int i=0; i<vert.length(); ++i)
    {       
        for(int j=0; j<textureTiles.length(); ++j)
        {
            d = min(d, PointSegDistance2(pp+textureTiles[j], vert[i], vert[i+1%vert.length()]));
        }
    }
    
    d = smoothstep(0.0, 1.0, (sqrt(d)-linewidth)/pixel_size);
    
    return d;
}

// Function 274
void Sample_ScatteredDirMIS(vec2 s0, vec2 s1, float s2, inout vec3 rd, inout vec3 W, vec3 N, vec3 albedo, float roughness, vec3 F0)
{
    float alpha = GGXAlphaFromRoughness(roughness);
    
    vec3 V = -rd;

    vec3 L0; float pdf00;
    {
        L0 = Sample_ClampedCosineLobe(s0.x * 2.0 - 1.0, s0.y, N);
		pdf00 = dot(N, L0) * rpi;        
    }

    vec3 L1; vec3 f1; float pdf11;
    Sample_GGX_R(s1, V, N, alpha, F0, /*out*/ L1, /*out*/ f1, /*out*/ pdf11);

    vec3 f0 = Frostbite_R(V, N, L0, albedo, roughness, F0);
         f1 = Frostbite_R(V, N, L1, albedo, roughness, F0);

    float pdf01 = dot(N, L1) * rpi;
    float pdf10 = EvalPDF_GGX_R(V, N, L0, alpha);

    float w0, w1;
    #if 0
    w0 = 0.5; 
    w1 = 1.0 - w1;
    #elif 1
    w0 = Pow2(pdf00) / (Pow2(pdf00) + Pow2(pdf10));
    w1 = Pow2(pdf11) / (Pow2(pdf11) + Pow2(pdf01));        
    #else
    w0 = (pdf00) / ((pdf00) + (pdf10));
    w1 = (pdf11) / ((pdf11) + (pdf01));  
    #endif

    #if 0
    if(albedo.r == 0.0 && albedo.g == 0.0 && albedo.b == 0.0)
    {
    	w0 = 0.0;
    	w1 = 1.0;
    }
    #endif
    
    float wn = w0 / (w0 + w1);

    bool doUseSmpl0 = s2 <= wn;

    float denom = doUseSmpl0 ? pdf00 *        wn : 
                               pdf11 * (1.0 - wn);

    rd = doUseSmpl0 ? L0 : L1;

    if(denom == 0.0)
    {
        W = vec3(0.0);
        
        return;
    }
    
    if(doUseSmpl0)
        W *= f0 * w0;
    else
        W *= f1 * w1;

    W /= denom;
}

// Function 275
vec3 sampleLightSource(		in vec3 x,
                          	float Xi1, float Xi2,
                       out LightSamplingRecord sampleRec) {
    float min_x = objects[0].params_[0];			//min x
    float min_y = objects[0].params_[1];			//min y
    float max_x = objects[0].params_[2];			//max x
    float max_y = objects[0].params_[3];			//max y
    float dim_x = max_x - min_x;
    float dim_y = max_y - min_y;
    vec3 p_local = vec3(min_x + dim_x*Xi1, min_y + dim_y*Xi2, 0.0);
    vec3 n_local = vec3(0.0, 0.0, 1.0);
    vec3 p_global = toVec3( objects[0].transform_*vec4(p_local, 1.0) );
    vec3 n_global = toVec3( objects[0].transform_*vec4(n_local, 0.0) );
    
    float pdfA = 1.0 / (dim_x*dim_y);
    sampleRec.w = p_global - x;
    sampleRec.d = length(sampleRec.w);
    sampleRec.w = normalize(sampleRec.w);
    float cosAtLight = dot(n_global, -sampleRec.w);
    vec3 L = cosAtLight>0.0?getRadiance(vec2(Xi1,Xi2)):vec3(0.0);
    sampleRec.pdf = PdfAtoW(pdfA, sampleRec.d*sampleRec.d, cosAtLight);
    
	return L;
}

// Function 276
vec3 textureGamma(sampler2D channel, vec2 uv)
{
    vec3 tex = texture(channel, uv).xyz;
    return tex * tex;
}

// Function 277
float sampleSong(float x) {
    return texture(iChannel1,vec2(x,0.05)).x*0.2;
}

// Function 278
vec4 sample3DLinear(sampler2D tex, vec3 uvw, vec3 vres)
{
    vec3 blend = fract(uvw*vres);
    vec4 off = vec4(1.0/vres, 0.0);
    
    //2x2x2 sample blending
    vec4 b000 = sample3D(tex, uvw + off.www, vres);
    vec4 b100 = sample3D(tex, uvw + off.xww, vres);
    
    vec4 b010 = sample3D(tex, uvw + off.wyw, vres);
    vec4 b110 = sample3D(tex, uvw + off.xyw, vres);
    
    vec4 b001 = sample3D(tex, uvw + off.wwz, vres);
    vec4 b101 = sample3D(tex, uvw + off.xwz, vres);
    
    vec4 b011 = sample3D(tex, uvw + off.wyz, vres);
    vec4 b111 = sample3D(tex, uvw + off.xyz, vres);
    
    return mix(mix(mix(b000,b100,blend.x), mix(b010,b110,blend.x), blend.y), 
               mix(mix(b001,b101,blend.x), mix(b011,b111,blend.x), blend.y),
               blend.z);
}

// Function 279
vec3 NearestTextureSample (vec2 P)
{
    vec2 pixel = P * c_textureSize;
    
    vec2 frac = fract(pixel);
    pixel = (floor(pixel) / c_textureSize);
    return texture(iChannel0, pixel + vec2(c_onePixel/2.0)).rgb;
}

// Function 280
vec3 sampleWeights(float i) {
	return vec3(i * i, 46.6666*pow((1.0-i)*i,3.0), (1.0 - i) * (1.0 - i));
}

// Function 281
vec4 barkTexture(vec3 p, vec3 nor)
{
    vec2 r = floor(p.xz / 5.0) * 0.02;
    float br = texture(iChannel1, r).x;
	vec3 mat = texCube(iChannel3, p*.4, nor) * vec3(.4, .3, .1*br) *br;
    mat += texCube(iChannel3, p*.53, nor)*smoothstep(0.0,.3, mat.x)*br;
   	return vec4(mat, .1);
}

// Function 282
vec3 Sample_DirLight(vec3 V, vec3 p, vec3 N, vec3 L, vec3 albedo, float roughness, vec3 F0)
{
    float alpha = GGXAlphaFromRoughness(roughness);
    
    float t2; vec3 n2; vec3 a2; bool hitLight2 = false;
    bool hit = Intersect_Scene(p, L, false, /*out*/ t2, n2, a2, hitLight2);

    if(hit) return vec3(0.0);

    return Frostbite_R(V, N, L, albedo, roughness, F0) * (Intensity * Pow2(0.125));// just set brightness heuristically here based on point light intensity
}

// Function 283
vec4 sampleWithBilinearInterpolation(sampler2D sampler, vec2 samplerSize,
                                     vec2 uv)
{
        vec2 texel = 1.0 / samplerSize;
        vec2 texelPos = samplerSize * uv;

        // we get the position of the texel. Watch out that
        // texels start at the center of a position (hence the 0.5)
        vec2 bottomLeftTexelPos = floor(texelPos - vec2(0.5)) + vec2(0.5);

        vec4 bl = texture(sampler, (bottomLeftTexelPos + vec2(0.0, 0.0)) * texel);
        vec4 br = texture(sampler, (bottomLeftTexelPos + vec2(1.0, 0.0)) * texel);
        vec4 tl = texture(sampler, (bottomLeftTexelPos + vec2(0.0, 1.0)) * texel);
        vec4 tr = texture(sampler, (bottomLeftTexelPos + vec2(1.0, 1.0)) * texel);

        vec2 fractFromBottomLeftTexelPos = texelPos - bottomLeftTexelPos;
        if (fractFromBottomLeftTexelPos.x > 1.0) {
                return vec4(1.0, 0.0, 0.0, 0.0);
        }
        if (fractFromBottomLeftTexelPos.y > 1.0) {
                return vec4(1.0, 0.0, 0.0, 0.0);
        }

        vec4 tA = mix(bl, br, fractFromBottomLeftTexelPos.x);
        vec4 tB = mix(tl, tr, fractFromBottomLeftTexelPos.x);
        return mix(tA, tB, fractFromBottomLeftTexelPos.y);
}

// Function 284
vec4 SampleAA(in vec2 uv)
{
    vec2 s = vec2(1.0 / iResolution.x, 1.0 / iResolution.y);
    vec2 o = vec2(0.11218413712, 0.33528304367) * s;
    
    return (texture(iChannel1, uv + vec2(-o.x,  o.y)) +
            texture(iChannel1, uv + vec2( o.y,  o.x)) +
            texture(iChannel1, uv + vec2( o.x, -o.y)) +
            texture(iChannel1, uv + vec2(-o.y, -o.x))) * 0.25;
}

// Function 285
float escherTexture(vec2 p, float pixel_size)
{
    float x = escherTextureX(p);
    float y = escherTextureY(p);
    
    x = smoothstep(-1.0, 1.0, x/pixel_size);
    y = smoothstep(-1.0, 1.0, y/pixel_size);
    
    float d = x+y - 2.0 * x*y;
    
    return d;
}

// Function 286
vec4 textureNoTile( sampler2D samp, in vec2 uv )
{
    vec2 iuv = floor( uv );
    vec2 fuv = fract( uv );

#ifdef USEHASH    
    // generate per-tile transform (needs GL_NEAREST_MIPMAP_LINEARto work right)
    vec4 ofa = texture( iChannel1, (iuv + vec2(0.5,0.5))/256.0 );
    vec4 ofb = texture( iChannel1, (iuv + vec2(1.5,0.5))/256.0 );
    vec4 ofc = texture( iChannel1, (iuv + vec2(0.5,1.5))/256.0 );
    vec4 ofd = texture( iChannel1, (iuv + vec2(1.5,1.5))/256.0 );
#else
    // generate per-tile transform
    vec4 ofa = hash4( iuv + vec2(0.0,0.0) );
    vec4 ofb = hash4( iuv + vec2(1.0,0.0) );
    vec4 ofc = hash4( iuv + vec2(0.0,1.0) );
    vec4 ofd = hash4( iuv + vec2(1.0,1.0) );
#endif
    
    vec2 ddx = dFdx( uv );
    vec2 ddy = dFdy( uv );

    // transform per-tile uvs
    ofa.zw = sign(ofa.zw-0.5);
    ofb.zw = sign(ofb.zw-0.5);
    ofc.zw = sign(ofc.zw-0.5);
    ofd.zw = sign(ofd.zw-0.5);
    
    // uv's, and derivarives (for correct mipmapping)
    vec2 uva = uv*ofa.zw + ofa.xy; vec2 ddxa = ddx*ofa.zw; vec2 ddya = ddy*ofa.zw;
    vec2 uvb = uv*ofb.zw + ofb.xy; vec2 ddxb = ddx*ofb.zw; vec2 ddyb = ddy*ofb.zw;
    vec2 uvc = uv*ofc.zw + ofc.xy; vec2 ddxc = ddx*ofc.zw; vec2 ddyc = ddy*ofc.zw;
    vec2 uvd = uv*ofd.zw + ofd.xy; vec2 ddxd = ddx*ofd.zw; vec2 ddyd = ddy*ofd.zw;
        
    // fetch and blend
    vec2 b = smoothstep(0.25,0.75,fuv);
    
    return mix( mix( textureGrad( samp, uva, ddxa, ddya ), 
                     textureGrad( samp, uvb, ddxb, ddyb ), b.x ), 
                mix( textureGrad( samp, uvc, ddxc, ddyc ),
                     textureGrad( samp, uvd, ddxd, ddyd ), b.x), b.y );
}

// Function 287
float Sample_Triangle(float s) 
{ 
    float v = 1.0 - sqrt(1.0 - abs(s));
    
    return s < 0.0 ? -v : v; 
}

// Function 288
vec3 importanceSampleGGX(vec2 xi, float a, vec3 n, float mnl)
{
	float phi = 6.2831853*xi.x;
	float cosTh = sqrt((1.0 - xi.y)/(1.0 + (a*a - 1.0)*xi.y));		
	float sinTh = sqrt(1.0 - cosTh*cosTh);
    vec3 v = vec3(sinTh * cos(phi), sinTh * sin(phi), cosTh);
    vec3 tx, ty;
    basis(n, ty, tx);
	return (tx*v.x + ty*v.y + n*v.z);
}

// Function 289
float hairtexture(vec2 uv, float scale, float angle){
    vec2 offsets[9] = vec2[9](vec2(0.), vec2(.5), vec2(-.5),
                              vec2(.5,0.), vec2(-.5,0.),
                              vec2(0.,.5), vec2(0.,-.5),
                              vec2(.5,-.5), vec2(-.5,.5));

    float f = 0.0;

    for(int i = 0; i < 9; i++){
        f = max(f, hair(uv, offsets[i], scale, angle));
    } 
    
    return smoothstep(0.0, 1.0, f);
}

// Function 290
vec4 texture(sampler2D sampler, vec2 uv)
{
    return texture(sampler, uv);
}

// Function 291
vec3 ggx_sample( vec3 N, float alpha, float Xi1, float Xi2 ) {
    vec3 Z = N;
    vec3 X = sampleHemisphere( N, Xi1, Xi2 );
    vec3 Y = cross( X, Z );
    X = cross( Z, Y );
    
    float alpha2 = alpha * alpha;
    float tanThetaM2 = alpha2 * Xi1 / (1.0 - Xi1);
    float cosThetaM  = 1.0 / sqrt(1.0 + tanThetaM2);
    float sinThetaM  = cosThetaM * sqrt(tanThetaM2);
    float phiM = TWO_PI * Xi2;
    
    return X*( cos(phiM) * sinThetaM ) + Y*( sin(phiM) * sinThetaM ) + Z*cosThetaM;
}

// Function 292
vec4 textureWall(vec2 uv) {
    const vec2 RES = vec2(32.0, 16.0);
    vec2 iuv = floor(uv * RES);    
    float n = noise1(uv * RES);
    n = n * 0.5 + 0.25;
    float nc = n * (smoothstep(1.0,0.4, iuv.x / RES.x) * 0.5 + 0.5);    
    return vec4(nc * 0.4, nc * 1.0, nc * 0.5, n + uv.x - abs(uv.y-0.5) );
}

// Function 293
vec2 sample_lpv_trilin(vec3 p) {
    p = p * size;
    float inr = inrange(p, vec3(0.0), size);
    vec3 pc = clamp(p, vec3(0.0), size);
    float cubedist = distance(p, pc);
    vec2 e = vec2(0.0,1.0);
    vec4 p000 = fetch_lpv(pc + e.xxx);
    vec4 p001 = fetch_lpv(pc + e.xxy);
    vec4 p010 = fetch_lpv(pc + e.xyx);
    vec4 p011 = fetch_lpv(pc + e.xyy);
    vec4 p100 = fetch_lpv(pc + e.yxx);
    vec4 p101 = fetch_lpv(pc + e.yxy);
    vec4 p110 = fetch_lpv(pc + e.yyx);
    vec4 p111 = fetch_lpv(pc + e.yyy);

    vec3 w = fract(pc);

    vec3 q = 1.0 - w;

    vec2 h = vec2(q.x,w.x);
    vec4 k = vec4(h*q.y, h*w.y);
    vec4 s = k * q.z;
    vec4 t = k * w.z;
        
    vec4 tril = 
          p000*s.x + p100*s.y + p010*s.z + p110*s.w
        + p001*t.x + p101*t.y + p011*t.z + p111*t.w;
    
    //return vec2(inr * tril.x, (1.0 - inr) * cubedist);
    return vec2(tril.x, (1.0 - inr) * cubedist);

}

// Function 294
vec3 Sample_GGX_VNDF(vec3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
	// Section 3.2: transforming the view direction to the hemisphere configuration
	vec3 Vh = normalize(Ve * vec3(alpha_x, alpha_y, 1.0));
	
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
	float lensq = (Vh.x*Vh.x) + (Vh.y*Vh.y);
	vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
	vec3 T2 = cross(Vh, T1);
	
    // Section 4.2: parameterization of the projected area
	float r = sqrt(U1);
	float phi = 2.0 * pi * U2;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5 * (1.0 + Vh.z);
	t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
    
	// Section 4.3: reprojection onto hemisphere
	vec3 Nh = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2))*Vh;
    
	// Section 3.4: transforming the normal back to the ellipsoid configuration
	vec3 Ne = normalize(vec3(Nh.xy, max(0.0, Nh.z)) * vec3(alpha_x, alpha_y, 1.0));
    
	return Ne;
}

// Function 295
vec3 resampleColor(Bounce[WAVELENGTHS] bounces) {
    vec3 col = vec3(0.0);
    
    for (int i = 0; i < WAVELENGTHS; i++) {        
        float reflectance = bounces[i].reflectance;
        float index = float(i) / float(WAVELENGTHS - 1);
        float texCubeIntensity = filmic_gamma_inverse(
            clamp(bounces[i].attenuation * sampleCubeMap(index, bounces[i].ray_direction), 0.0, 0.99)
        );
    	float intensity = texCubeIntensity + reflectance;
        col += sampleWeights(index) * intensity;
    }

    return 1.4 * filmic_gamma(3.0 * col / float(WAVELENGTHS));
}

// Function 296
vec3 sampleLightSource( 	in vec3 x,
                          	float Xi1, float Xi2,
                          	out LightSamplingRecord sampleRec ) {
    float sph_r2 = objects[0].params_[1];
    vec3 sph_p = toVec3( objects[0].transform_*vec4(vec3(0.0,0.0,0.0), 1.0) );
    
    vec3 w = sph_p - x;			//direction to light center
	float dc_2 = dot(w, w);		//squared distance to light center
    float dc = sqrt(dc_2);		//distance to light center
    
    
    float sin_theta_max_2 = sph_r2 / dc_2;
	float cos_theta_max = sqrt( 1.0 - clamp( sin_theta_max_2, 0.0, 1.0 ) );
    float cos_theta = mix( cos_theta_max, 1.0, Xi1 );
    float sin_theta_2 = 1.0 - cos_theta*cos_theta;
    float sin_theta = sqrt(sin_theta_2);
    sampleRec.w = uniformDirectionWithinCone( w, TWO_PI*Xi2, sin_theta, cos_theta );
    sampleRec.pdf = 1.0/( TWO_PI * (1.0 - cos_theta_max) );
        
    //Calculate intersection distance
	//http://ompf2.com/viewtopic.php?f=3&t=1914
    sampleRec.d = dc*cos_theta - sqrt(sph_r2 - dc_2*sin_theta_2);
    
    return lights[0].color_*lights[0].intensity_;
}

// Function 297
float sample_tex(vec3 coord) {
    // Pretend our texture contains a SDF in each axis.
    // Munge some of the sample coordinates to reduce periodicity
    float t1 = textureLod(WORLD_TEX, coord.xy, 0.0).r;
    float t2 = textureLod(WORLD_TEX, coord.yz * 0.76, 0.0).g;
    float t3 = textureLod(WORLD_TEX, coord.xz * 1.23, 0.0).b;
    
    return (t1 + t2 + t3) / 3.0;
}

// Function 298
vec3 Sample_Sphere(float s0, float s1)
{
    float ang = Pi * s0;
    float s1p = sqrt(1.0 - s1*s1);
    
    return vec3(cos(ang) * s1p, 
                           s1 , 
                sin(ang) * s1p);
}

// Function 299
void sampleDirectLight( vec3 pos,
                       	vec3 normal,
                        float Xi1,
                        float Xi2, 
                       	out vec3 dir,
                       	out float pdf ) {
    float height = objects[0].params_[2] - objects[0].params_[1];
    float r = objects[0].params_[0];
    float pdfA;
    float d2;
    float aCosThere;
    float theta;
    float thetaPdf;
    float h;
    float hPdf;
    
    //convert position to object space
    pos = toVec3( objects[0].transform_inv_*vec4(pos, 1.0) );
    normal = toVec3( objects[0].transform_inv_*vec4(normal, 0.0) );
    
    if(samplingTechnique == SAMPLE_TOTAL_AREA){
        theta = Xi1*TWO_PI;
        thetaPdf = 1.0/TWO_PI;
        h = objects[0].params_[1] + Xi2*height;
        hPdf = 1.0/height;
        
        vec3 n = vec3(cos(theta), sin(theta), 0.0);
        vec3 p = n*r;
        p.z = h;
        dir = p - pos;
        d2 = dot(dir,dir);
        dir /= sqrt(d2);
        aCosThere = max(0.0,dot(-dir,n));

        pdfA = thetaPdf*hPdf*(1.0/r);
        pdf = PdfAtoW( pdfA, d2, aCosThere );
    } else {
        vec3 cylinderPos = vec3(0.0, 0.0, objects[0].params_[1]);
        vec3 cylinderVec = vec3(0.0, 0.0, 1.0);
        
        float dc_2 = dot(pos.xy,pos.xy);
        float dc = sqrt(dc_2);
        vec2 dirToPos = (-pos.xy)/dc;
        float alphaMax = acos(r/dc);
        float thetaMax = 0.5*PI - alphaMax;
        theta = mix(0.0,thetaMax, Xi1);
        float sinTheta = sin(theta);
        float sin2Theta = sinTheta*sinTheta;
        float cosTheta = sqrt(1.0-sin2Theta);
        
    	float ds = dc * cosTheta - sqrt(max(0.0, r*r - dc_2 * sin2Theta));
    	float cosAlpha = (dc * dc + r*r - ds*ds) / (2.0 * dc * r);
        
        float alpha = acos(cosAlpha);
        
        //convert alpha to global angle
        float baseAngle = atan(dirToPos.y,dirToPos.x) + PI;
        float sign;
        if(Xi2<0.5) {
            sign =  1.0;
            //normalize Xi
            Xi2 /= 0.5;
        } else {
            sign = -1.0;
            //normalize Xi
            Xi2 = (Xi2 - 0.5) / 0.5;
        }
        
        float alphaGlobal = baseAngle + alpha*sign;
       
        thetaPdf = 1.0/(2.0*thetaMax);// In angle measure
        thetaPdf = PdfWtoA(thetaPdf, ds, cos(alpha+theta));
        
        vec3 n = vec3(cos(alphaGlobal), sin(alphaGlobal), 0.0);
    	vec3 p = n*r;
        
        //Sampling h
        //We do equiangular sampling from shading point
        {
      		sampleLine( pos, normalize(normal), p, normalize(cylinderVec), height, Xi2, h, hPdf );
        }
        
        p.z = cylinderPos.z + h;
        dir = p - pos;
        d2 = dot(dir,dir);
        dir /= sqrt(d2);
        aCosThere = max(0.0,dot(-dir,n));

        pdfA = thetaPdf*hPdf;
        pdf = PdfAtoW( pdfA, d2, aCosThere );
    }
    
    //convert dir to world space
    dir = toVec3( objects[0].transform_*vec4(dir,0.0) );
}

// Function 300
vec2 ImportanceSampleGGXTransform( const vec2 uniformSamplePos, const in float alpha2 )
{
	// [Karis2013]  Real Shading in Unreal Engine 4
	// http://blog.tobias-franke.eu/2014/03/30/notes_on_importance_sampling.html

	float theta = acos( sqrt( (1.0f - uniformSamplePos.y) /
							( (alpha2 - 1.0f) * uniformSamplePos.y + 1.0f )
							) );

	float phi = 2.0f * PI * uniformSamplePos.x;

	return vec2( theta, phi );
}

// Function 301
vec2 textureSizef (int i)
{
	vec2 size = i == 0 ? iChannelResolution[0].xy : iChannelResolution[1].xy;
#ifdef NYAN
	if (i == 1)
		size.x = 42.0;
#endif
	return size;
}

// Function 302
vec2 TransformFromCanvasTextureToFramedTexture(
	vec2 canvasTextureCoord,
	vec2 canvasTextureSize,
	vec2 framedTextureSize)
{	
	vec2 result = (canvasTextureCoord / canvasTextureSize);

	float canvasAspectRatio = (canvasTextureSize.x / canvasTextureSize.y);
	float framedAspectRatio = (framedTextureSize.x / framedTextureSize.y);

	if (framedAspectRatio < canvasAspectRatio)
	{
		float relativeAspectRatio = (canvasAspectRatio / framedAspectRatio);

		result.x *= relativeAspectRatio;
		result.x -= (0.5 * (relativeAspectRatio - 1.0));
	}
	else
	{
		float relativeAspectRatio = (framedAspectRatio / canvasAspectRatio);

		result.y *= relativeAspectRatio;
		result.y -= (0.5 * (relativeAspectRatio - 1.0));
	}

	return result;
}

// Function 303
vec3 SampleImage( vec2 vUV, int image )
{
    if ( image >= 0 )
    {
        vUV.x *= 0.5;
    }
    
    if (image > 0 )
    {
        vUV.x += 0.5;
    }
    
	vec4 vImageSample = textureLod( iChannel0, vUV, 0.0 ).rgba;

    return vImageSample.rgb;
}

// Function 304
vec3 Sample_SphLight_SolidAngle(vec2 s, vec3 V, vec3 p, vec3 N, vec3 albedo, float roughness, vec3 F0)
{
    float alpha = GGXAlphaFromRoughness(roughness);
    
    float ct; vec3 Lc, L; float sang;
    Sample_SolidAngle(s, p, LightPos, R2, /*out*/ ct, /*out*/ Lc, /*out*/ L, /*out*/ sang);

    float NoL = dot(N, L);

    if(NoL <= 0.0) return vec3(0.0);
    
    float t2; vec3 n2; vec3 a2; bool isLight2 = true;
    bool hit = Intersect_Scene(p, L, false, /*out: */ t2, n2, a2, isLight2);

    if(!isLight2 && t2 < dot(LightPos-p, Lc)) return vec3(0.0);
    
    vec3 f = Frostbite_R(V, N, L, albedo, roughness, F0);
    float rpdf = sang;

    return f * rpdf * Radiance;
}

// Function 305
void sampleCamera(vec2 fragCoord, vec2 u, out vec3 rayOrigin, out vec3 rayDir)
{
	vec2 filmUv = (fragCoord.xy + u)/iResolution.xy;

	float tx = (2.0*filmUv.x - 1.0)*(iResolution.x/iResolution.y);
	float ty = (1.0 - 2.0*filmUv.y);
	float tz = 0.0;

	rayOrigin = vec3(0.0, 0.0, 5.0);
	rayDir = normalize(vec3(tx, ty, tz) - rayOrigin);
}

// Function 306
vec3 circleSampleColor(vec2 dist1, vec2 center1, vec2 dist2, vec2 center2)
{
    vec2 curRadius1 = vec2(0.0);
    vec2 curRadius2 = vec2(0.0);
	vec3 returnValue = vec3(0.0);
    
    for (int c = 0; c < CIRCLE_NUMBER; ++c)
    {
    	float normalizedAngle = 0.0;
        curRadius1 += dist1;
        curRadius2 += dist2;
        for (int s = 0; s < SAMPLE_PER_CIRCLE; ++s)
        {
            float angle = normalizedAngle * 3.1415 * 2.0;
            vec2 uvToSample1 = center1 + vec2(cos(angle), sin(angle)) * curRadius1;
            vec2 uvToSample2 = center2 + vec2(cos(angle), sin(angle)) * curRadius2;
            vec3 sampledColor1 = texture(iChannel0, uvToSample1).rgb;
            vec3 sampledColor2 = texture(iChannel1, uvToSample2).rgb;
            if (passTest(sampledColor1))
				returnValue += sampledColor2 / float(CIRCLE_NUMBER * SAMPLE_PER_CIRCLE);
            else
                returnValue += sampledColor1 / float(CIRCLE_NUMBER * SAMPLE_PER_CIRCLE);
            normalizedAngle += 1.0 / float(SAMPLE_PER_CIRCLE);
        }
    }
    return (returnValue);
}

// Function 307
float getTextureIndex(float _v)
{
    return floor(_v * TEXTURES_NUM);
}

// Function 308
vec3 sampleLight(
	vec2 uv,
	vec3 planeNormal,
	out vec3 lightPos,
	out float areaPdf)
{
	vec3 planeTangent = normalize(cross(planeNormal, vec3(0.0, 1.0, 0.0)));
	vec3 planeBitangent = normalize(cross(planeNormal, planeTangent));
	float x = 0.5 - uv.x;
	float y = uv.y - 0.5;
	lightPos = x*planeTangent + y*planeBitangent;
	return evaluateLight(uv, areaPdf);
}

// Function 309
vec2 concentricSampleDisk(in vec2 xi) {
    // Map uniform random numbers to $[-1,1]^2$
    vec2 uOffset = 2. * xi - vec2(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0.0 && uOffset.y == 0.0) return vec2(.0);

    // Apply concentric mapping to point
    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = (PI/4.0) * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = (PI/2.0) - (PI/4.0) * (uOffset.x / uOffset.y);
    }
    return r * vec2(cos(theta), sin(theta));
}

// Function 310
vec4 fontTextureLookup(vec2 xy)
{
    /* low quality font lookup */
    /*return texture(fontChannel,xy);*/
    
    /* high quality font lookup*/
	float dxy = 1024.*1.5;
	vec2 dx = vec2(1.,0.)/dxy;
	vec2 dy = vec2(0.,1.)/dxy;

    return 
        (    texture(fontChannel,xy + dx + dy)
        +    texture(fontChannel,xy + dx - dy)
        +    texture(fontChannel,xy - dx - dy)
        +    texture(fontChannel,xy - dx + dy)
        + 2.*texture(fontChannel,xy)
        )/6.
    ;
}

// Function 311
float sample_dist_gaussian(vec2 uv, float font_size) {

    float dsum = 0.;
    float wsum = 0.;
    
    const int nstep = 3;
    
    const float w[3] = float[3](1., 2., 1.);
    
    for (int i=0; i<nstep; ++i) {
        for (int j=0; j<nstep; ++j) {
            
            vec2 delta = vec2(float(i-1), float(j-1))/TEX_RES;
            
            float dist = sample_grad_dist(uv-delta, font_size).z;
            float wij = w[i]*w[j];
            
            dsum += wij * dist;
            wsum += wij;

        }
    }
    
    return dsum / wsum;
}

// Function 312
vec3 sample_triquadratic_gradient_approx(sampler3D channel, vec3 res, vec3 uv) {
    vec3 q = fract(uv * res);
    vec3 cc = 0.5 / res;
    vec3 ww0 = uv - cc;
    vec3 ww1 = uv + cc;
    float nx = texture(channel, vec3(ww1.x, uv.y, uv.z)).r - texture(channel, vec3(ww0.x, uv.y, uv.z)).r;
    float ny = texture(channel, vec3(uv.x, ww1.y, uv.z)).r - texture(channel, vec3(uv.x, ww0.y, uv.z)).r;
    float nz = texture(channel, vec3(uv.x, uv.y, ww1.z)).r - texture(channel, vec3(uv.x, uv.y, ww0.z)).r;
	return vec3(nx, ny, nz);
}

// Function 313
C_Sample SampleMaterial(const in vec2 vUV, sampler2D sampler, const in vec3 vLight, const in vec2 vTextureSize, const in float fNormalScale)
{
	C_Sample result;
	
	vec2 vInvTextureSize = vec2(1.0) / vTextureSize;

	vec3 offset_dir = vec3( vLight.x, vLight.y, .05 ) * 2.5;

	vec3 cBaseSample = texture(sampler, vUV ).rgb;	
	
	float bump;
	
	
	if (vUV.x >= 0.5 )
	{
		vec3 cOffsetSampleX = texture(sampler, vUV + vec2( offset_dir.x, 0.0 ) * vInvTextureSize.xy).rgb;
		vec3 cOffsetSampleY = texture(sampler, vUV + vec2( 0.0, offset_dir.y ) * vInvTextureSize.xy).rgb;
		
		vec3 normal = vec3( cOffsetSampleX.g -cBaseSample.g , cOffsetSampleY.g -cBaseSample.g , 0.3 );
	
		normal = normalize( normal );
	
		bump = 1.2 * dot( normal, vLight );
	}
	else
	{
		vec3 cOffsetSample = texture(sampler, vUV +  vec2( offset_dir.x, offset_dir.y ) * vInvTextureSize.xy).rgb;
	
	    bump = cBaseSample.g - cOffsetSample.g;
        
        bump = smoothstep(0.0,1.0,bump);
	
		bump += 0.5 * vLight.z + 0.25;
		
		bump *= 1.2;
	}


#ifdef USE_DIFFUSE
	result.diffuse = cBaseSample * bump;
#else
	result.diffuse = vec3( 1., 1., 1. );
#endif
	result.bump = bump;
	if ( abs( vUV.x - 0.5 ) < 0.003 )
	{
		result.diffuse = vec3( 0.0, 0.0, 0.0 );
		result.bump = 1.0;
	}

	return result;
}

// Function 314
vec3 Sample_ClampedCosineLobe(float s0, float s1)
{	 
    vec2 d  = Sample_Disk(s0, s1);
    float y = sqrt(clamp01(1.0 - s1));
    
    return vec3(d.x, y, d.y);
}

// Function 315
vec4 sampleCell(sampler2D tex, vec2 index) {
    return texture(tex, index / GRID_SIZE + GRID_HALF);
}

// Function 316
vec3 TrilinearSamplerIBL(vec3 dir,float roughness){
    vec3 axis = getAxis(dir);
   	vec3 uvw = GetUVW(axis,dir);
    float roughness_ID = roughness * ID_Range.x;
    float pre_ID  = floor(30.-roughness_ID);
    float next_ID = pre_ID - 1.;
    vec3 pre_dir  = GetReadMipMapUVW_Dir2(uvw,axis,pre_ID);
	vec3 next_dir = GetReadMipMapUVW_Dir2(uvw,axis,next_ID);
    vec3 preCol   = IBL(pre_dir);
    vec3 nextCol  = IBL(next_dir);
    return mix(preCol,nextCol,fract(roughness * ID_Range.x));
}

// Function 317
vec3 hsample(vec2 tc)
{
	return samplef(tc);
}

// Function 318
float SampleRef(sampler2D channel, vec2 uv)
{
    uv -= vec2(0.5);
    
    vec2 uvi = floor(uv);
    vec2 uvf = uv - uvi;

    ivec2 uv0 = ivec2(uvi);
    
    vec2 sn = vec2((uv0.x & 1) == 0 ? -1.0 : 1.0,
                   (uv0.y & 1) == 0 ? -1.0 : 1.0);
    
    float r = 0.0;
    for(int j = 0; j < 2; ++j)
    for(int i = 0; i < 2; ++i)
    {
        vec4 c = texelFetch(channel, uv0 + ivec2(i, j), 0);
        
        vec2 l = uvf;
        
        vec2 sn0 = sn;
        
        if(i != 0) {l.x -= 1.0; sn0.x *= -1.0;}
        if(j != 0) {l.y -= 1.0; sn0.y *= -1.0;}
        
        c.xyz *= vec3(sn0, sn0.x*sn0.y);// un-flip derivative sample signs; we usually don't need this for the ground truth reconstruction
        
        r += dot(c, kern(l));
    }
    
	return r;
}

// Function 319
vec3 uniform_sample_cone(vec2 u12, 
                         float cosThetaMax, 
                         vec3 xbasis, vec3 ybasis, vec3 zbasis)
{
    float cosTheta = (1. - u12.x) + u12.x * cosThetaMax;
    float sinTheta = sqrt(1. - cosTheta * cosTheta);
    float phi = u12.y * TWO_PI;
    vec3 samplev = polar_to_cartesian(sinTheta, cosTheta, sin(phi), cos(phi));
    return samplev.x * xbasis + samplev.y * ybasis + samplev.z * zbasis;
}

// Function 320
float getSampleDim1(int sampleIndex,vec2 fragCoord)
{
	return fract(getDimensionHash(1,fragCoord) + radicalInverse(sampleIndex, 3));
}

// Function 321
vec3 SampleNormalMap(in vec2 uv, in float height)
{
    const float strength = 40.0;    
    float d0 = SampleTexture(uv.xy);
    float dX = SampleTexture(uv.xy - vec2(EPSILON, 0.0));
    float dY = SampleTexture(uv.xy - vec2(0.0, EPSILON));
    return normalize(vec3((dX - d0) * strength, (dY - d0) * strength, 1.0));
}

// Function 322
vec4 GetSampleColor(ivec2 currentCoord, ivec2 samplePosition, float sampleResolution)
{
    ivec2 sampleOffset = currentCoord - samplePosition;
    ivec2 sampleCoord = ivec2(floor(vec2(sampleOffset) / sampleResolution));
    vec4 sampleColor = texture(iChannel0, vec2(samplePosition + sampleCoord) / iResolution.xy);
    return sampleColor;
}

// Function 323
vec4 sampleIDStage(int ID, int stage){
    ivec2 XYTall = ivec2(ID/16, ID%16+stage*16);
    return sampleXYTall(XYTall);
}

// Function 324
vec4 SampleMip3(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*8.,sp.z+48.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(8.,0.))*IRES),fract(sp.y));
}

// Function 325
vec4 SampleTextureBilinearlyAndUnpack(sampler2D tex, vec2 uv)
{
    vec4 sample_color = texture(tex, uv, 0.0);
#ifdef PACK_SIGNED_TO_UNSIGNED
    sample_color = 2.0 * sample_color - 1.0;
#endif // PACK_SIGNED_TO_UNSIGNED
    return sample_color;
}

// Function 326
vec4 SampleCharacterTex( uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vUV = (vec2(iChPos) + vCharUV) / 16.0f;
    return textureLod( FONT_SAMPLER, vUV, 0.0 );
}

// Function 327
float textureInvader(vec2 uv) {
	float y = 7.-floor((uv.y)*16.+4.);
	if(y < 0. || y > 7.) return 0.;
	float x = floor((abs(uv.x))*16.);
//	if(x < 0. || x > 14.) return 0.;
	float v=(y>6.5)? 6.:(y>5.5)? 40.:(y>4.5)? 47.:(y>3.5)? 63.:
			(y>2.5)? 27.:(y>1.5)? 15.:(y>0.5)? 4.: 8.;
	return floor(mod(v/pow(2.,x), 2.0)) == 0. ? 0.: 1.;
}

// Function 328
vec3 mtlSample(Material mtl, in vec3 Ng, in vec3 Ns, in vec3 E, in float Xi1, in float Xi2, out vec3 L, out float pdf, out float spec) {
    mat3 trans = mat3FromNormal(Ns);
    mat3 inv_trans = mat3Inverse( trans );
    
    //convert directions to local space
    vec3 E_local = inv_trans * E;
    vec3 L_local;
    
    float alpha = mtl.specular_roughness_;
    // Sample microfacet orientation $\wh$ and reflected direction $\wi$
    if (E_local.z == 0.0) return vec3(0.);
    vec3 wh = ggx_sample(E_local, alpha, alpha, Xi1, Xi2);
    if (!sameHemisphere(vec3(0.0, 0.0, 1.0), E_local, wh)) {
        wh = -wh;
    }
    
    float F = length(fresnelConductor(dot(L_local, wh), vec3(1.5/1.0), vec3(1.0)));
    //Sample specular or diffuse lobe based on fresnel
    if(rnd() < F) {
        L_local = reflect(E_local, wh);
    
        if(!sameHemisphere(E_local, L_local)){
           L_local = -L_local;
        }

        if (!sameHemisphere(E_local, L_local)) {
            pdf = 0.0;
        } else {
            // Compute PDF of _wi_ for microfacet reflection
            pdf = ggx_pdf(E_local, wh, alpha, alpha) / (4.0 * dot(E_local, wh));
        }
        //pdf *= F;
    } else {
        L = sampleHemisphereCosWeighted( Xi1, Xi2 );
    	pdf = INV_PI;
        //pdf *= 1.0 - F;
    }
    
    //convert directions to global space
    L = trans*L_local;
    
    if(!sameHemisphere(Ns, E, L) || !sameHemisphere(Ng, E, L)) {
        pdf = 0.0;
    }
    
    return mtlEval(mtl, Ng, Ns, E, L);
}

// Function 329
vec4 textureGround(vec2 uv) {
    const vec2 RES = vec2(8.0, 8.0);    
    float n = noise1(uv * RES);
    n = n * 0.2 + 0.5;
    return vec4(n*0.9,n*0.6,n*0.4,1.0);
}

// Function 330
vec4 takeSample(in vec2 position, float pixelSize) {
  const float fov = pi / 2.0;
  Ray ray = createRayPerspective(iResolution.xy, position, fov);
  return trace(ray);
}

// Function 331
vec3 sampleOnATriangle(float r1, float r2, vec3 corner1, vec3 corner2, vec3 corner3 ){
  return (1. - sqrt(r1))*corner1 + (sqrt(r1)*(1. - r2))*corner2
				+ (r2*sqrt(r1)) * corner3;   
}

// Function 332
vec3 sampleLight(vec3 v)
{
    return texture (iChannel1, v).rgb;
}

// Function 333
vec4 fontTextureLookup(vec2 xy){	float dxy = 1024.*1.5; 	vec2 dx = vec2(1.,0.)/dxy; 	vec2 dy = vec2(0.,1.)/dxy; return (texture(fontChannel,xy + dx + dy)+texture(fontChannel,xy + dx - dy)+texture(fontChannel,xy - dx - dy)+texture(fontChannel,xy - dx + dy)+ 2.*texture(fontChannel,xy))/6.;}

// Function 334
vec3 TextureEnvBlured(in vec3 N) {
    vec3 iblDiffuse = vec3(0.0);

    float sum = 0.0;

    vec2 ts = vec2(textureSize(iChannel0, 0));
    float maxMipMap = log2(max(ts.x, ts.y));

    float lodBias = maxMipMap - 6.0;

    for (int i=0; i < ENV_SMPL_NUM; ++i) {
        vec3 sp = SpherePoints_GoldenAngle(float(i), float(ENV_SMPL_NUM));

        float w = dot(sp, N ) * 0.5 + 0.5;

        // 4 is somehow a magic number that makes results of
        // weighted sampling and spherical harmonics convolutions match almost perfectly (~2% difference)
        w = pow(w, 4.0); 

        vec3 iblD = sampleReflectionMap(sp, lodBias);

        iblDiffuse  += iblD * w;

        sum += w;
    }

    iblDiffuse  /= sum;
    return iblDiffuse;
}

// Function 335
float Sample_Triangle(float s) 
{ 
    float v = 1.0 - sqrt(abs(s));
    
    return s < 0.0 ? -v : v; 
}

// Function 336
vec4 earthTexture(vec2 p)
{
    uint v = 0u;
	v = p.y == 99. ? 2576980377u : v;
	v = p.y == 98. ? 2576980377u : v;
	v = p.y == 97. ? 2576980377u : v;
	v = p.y == 96. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2576980377u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2005506457u : (p.x < 96. ? 2574743415u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2004457881u : (p.x < 136. ? 2543417207u : 2576980377u))))))))))))))))) : v;
	v = p.y == 95. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2576980377u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2040109465u : (p.x < 96. ? 2004318071u : (p.x < 104. ? 2004318071u : (p.x < 112. ? 2004318071u : (p.x < 120. ? 2004318071u : (p.x < 128. ? 2004318071u : (p.x < 136. ? 2004318020u : (p.x < 144. ? 2004318071u : (p.x < 152. ? 2004318071u : (p.x < 160. ? 2574743415u : 2576980377u)))))))))))))))))))) : v;
	v = p.y == 94. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2575866265u : (p.x < 64. ? 2576910745u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 288638841u : (p.x < 104. ? 858992913u : (p.x < 112. ? 2004317235u : (p.x < 120. ? 322123127u : (p.x < 128. ? 1144197393u : (p.x < 136. ? 1146582323u : (p.x < 144. ? 858993459u : (p.x < 152. ? 1999844147u : (p.x < 160. ? 2004318617u : (p.x < 168. ? 2575857527u : 2576980377u))))))))))))))))))))) : v;
	v = p.y == 93. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2541189529u : (p.x < 64. ? 2576840569u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 286331207u : (p.x < 104. ? 286331153u : (p.x < 112. ? 2005365777u : (p.x < 120. ? 286331767u : (p.x < 128. ? 286331153u : (p.x < 136. ? 823202065u : (p.x < 144. ? 286331187u : (p.x < 152. ? 1125191953u : (p.x < 160. ? 1199011700u : (p.x < 168. ? 2004317251u : (p.x < 176. ? 2541188983u : 2576980377u)))))))))))))))))))))) : v;
	v = p.y == 92. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2004457369u : (p.x < 64. ? 2004103989u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 286331255u : (p.x < 104. ? 286331153u : (p.x < 112. ? 323236113u : (p.x < 120. ? 286331155u : (p.x < 128. ? 286331153u : (p.x < 136. ? 1950626065u : (p.x < 144. ? 286331767u : (p.x < 152. ? 1145245969u : (p.x < 160. ? 2040100727u : (p.x < 168. ? 858994551u : (p.x < 176. ? 2004043636u : (p.x < 184. ? 2004318071u : (p.x < 192. ? 2576980375u : 2576980377u)))))))))))))))))))))))) : v;
	v = p.y == 91. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2004318089u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2039978393u : (p.x < 64. ? 823202100u : (p.x < 72. ? 2576839475u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2040109465u : (p.x < 96. ? 286331191u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 930576145u : (p.x < 144. ? 286331155u : (p.x < 152. ? 2574730305u : (p.x < 160. ? 2576976281u : (p.x < 168. ? 288638873u : (p.x < 176. ? 823202627u : (p.x < 184. ? 2000975735u : (p.x < 192. ? 2576979831u : 2576980377u)))))))))))))))))))))))) : v;
	v = p.y == 90. ? (p.x < 8. ? 2006555033u : (p.x < 16. ? 2004318071u : (p.x < 24. ? 2004318071u : (p.x < 32. ? 2536719447u : (p.x < 40. ? 2576979833u : (p.x < 48. ? 2543425689u : (p.x < 56. ? 2005367159u : (p.x < 64. ? 286340215u : (p.x < 72. ? 2574455057u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2004326809u : (p.x < 96. ? 286331155u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286488435u : (p.x < 144. ? 1127511345u : (p.x < 152. ? 2576980343u : (p.x < 160. ? 2004326809u : (p.x < 168. ? 2004384153u : (p.x < 176. ? 286488439u : (p.x < 184. ? 876902193u : (p.x < 192. ? 1145324339u : 2541188164u)))))))))))))))))))))))) : v;
	v = p.y == 89. ? (p.x < 8. ? 1148680055u : (p.x < 16. ? 2004309364u : (p.x < 24. ? 860111959u : (p.x < 32. ? 1395732787u : (p.x < 40. ? 2576840519u : (p.x < 48. ? 2541189017u : (p.x < 56. ? 2039977847u : (p.x < 64. ? 326605207u : (p.x < 72. ? 1967198481u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 1467455897u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286470963u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286345075u : (p.x < 144. ? 2004326227u : (p.x < 152. ? 2576979833u : (p.x < 160. ? 2576840567u : (p.x < 168. ? 1433893273u : (p.x < 176. ? 2004318070u : (p.x < 184. ? 1467446612u : (p.x < 192. ? 286475332u : 2000892211u)))))))))))))))))))))))) : v;
	v = p.y == 88. ? (p.x < 8. ? 825456503u : (p.x < 16. ? 1431655763u : (p.x < 24. ? 876959061u : (p.x < 32. ? 1414808628u : (p.x < 40. ? 2004317303u : (p.x < 48. ? 2004318071u : (p.x < 56. ? 1199011703u : (p.x < 64. ? 932812660u : (p.x < 72. ? 1412501777u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 897030553u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1145245969u : (p.x < 120. ? 860095013u : (p.x < 128. ? 856756531u : (p.x < 136. ? 1145320243u : (p.x < 144. ? 2003068228u : (p.x < 152. ? 2004318071u : (p.x < 160. ? 2004322167u : (p.x < 168. ? 1431656311u : (p.x < 176. ? 1431795285u : (p.x < 184. ? 1431655765u : (p.x < 192. ? 1145328981u : 1984189508u)))))))))))))))))))))))) : v;
	v = p.y == 87. ? (p.x < 8. ? 825509751u : (p.x < 16. ? 572662340u : (p.x < 24. ? 572662306u : (p.x < 32. ? 1428300322u : (p.x < 40. ? 2004313973u : (p.x < 48. ? 2004318071u : (p.x < 56. ? 393705369u : (p.x < 64. ? 2005366611u : (p.x < 72. ? 1947275591u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 286476152u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 627519761u : (p.x < 120. ? 1140850688u : (p.x < 128. ? 1145246530u : (p.x < 136. ? 1428309060u : (p.x < 144. ? 1430541653u : (p.x < 152. ? 1717904725u : (p.x < 160. ? 1428313702u : (p.x < 168. ? 2250069u : (p.x < 176. ? 572880160u : (p.x < 184. ? 572859749u : (p.x < 192. ? 572662306u : 1985303893u)))))))))))))))))))))))) : v;
	v = p.y == 86. ? (p.x < 8. ? 825517636u : (p.x < 16. ? 8772u : (p.x < 24. ? 570425344u : (p.x < 32. ? 1073881634u : (p.x < 40. ? 1968514115u : (p.x < 48. ? 2004318070u : (p.x < 56. ? 343373687u : (p.x < 64. ? 2004317267u : (p.x < 72. ? 1947275591u : (p.x < 80. ? 1200200089u : (p.x < 88. ? 1091637556u : (p.x < 96. ? 286475604u : (p.x < 104. ? 286331153u : (p.x < 112. ? 2446097u : (p.x < 120. ? 872415283u : (p.x < 128. ? 817u : (p.x < 136. ? 622854144u : (p.x < 144. ? 1431446018u : (p.x < 152. ? 572522498u : (p.x < 160. ? 572662309u : (p.x < 168. ? 9554u : (p.x < 176. ? 572945664u : (p.x < 184. ? 626349669u : (p.x < 192. ? 1428300288u : 1431655766u)))))))))))))))))))))))) : v;
	v = p.y == 85. ? (p.x < 8. ? 825438993u : (p.x < 16. ? 536880195u : (p.x < 24. ? 8738u : (p.x < 32. ? 2445824u : (p.x < 40. ? 1145176064u : (p.x < 48. ? 2004252533u : (p.x < 56. ? 897013623u : (p.x < 64. ? 2004317251u : (p.x < 72. ? 1091637523u : (p.x < 80. ? 323459479u : (p.x < 88. ? 823202065u : (p.x < 96. ? 286345044u : (p.x < 104. ? 286331153u : (p.x < 112. ? 805315123u : (p.x < 120. ? 1073741827u : (p.x < 128. ? 3u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 512u : (p.x < 176. ? 622993408u : (p.x < 184. ? 1449481573u : (p.x < 192. ? 1431654946u : 878007637u)))))))))))))))))))))))) : v;
	v = p.y == 84. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 1375732260u : (p.x < 24. ? 89478485u : (p.x < 32. ? 39198720u : (p.x < 40. ? 1145315328u : (p.x < 48. ? 1162168148u : (p.x < 56. ? 322113844u : (p.x < 64. ? 1195656563u : (p.x < 72. ? 823202065u : (p.x < 80. ? 286755207u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 1091637521u : (p.x < 112. ? 288358949u : (p.x < 120. ? 1074003968u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 0u : (p.x < 184. ? 1432840034u : (p.x < 192. ? 1414813013u : 860321109u)))))))))))))))))))))))) : v;
	v = p.y == 83. ? (p.x < 8. ? 823202065u : (p.x < 16. ? 1140859443u : (p.x < 24. ? 1735738436u : (p.x < 32. ? 2228261u : (p.x < 40. ? 1074003968u : (p.x < 48. ? 1195656260u : (p.x < 56. ? 286331153u : (p.x < 64. ? 826767219u : (p.x < 72. ? 286331155u : (p.x < 80. ? 286540865u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 1091637521u : (p.x < 112. ? 285212709u : (p.x < 120. ? 279619u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 139264u : (p.x < 176. ? 0u : (p.x < 184. ? 1146443090u : (p.x < 192. ? 1145123925u : 286471236u)))))))))))))))))))))))) : v;
	v = p.y == 82. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 874791697u : (p.x < 24. ? 1127420177u : (p.x < 32. ? 35652980u : (p.x < 40. ? 0u : (p.x < 48. ? 1140868096u : (p.x < 56. ? 286331155u : (p.x < 64. ? 1952937795u : (p.x < 72. ? 286331191u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 1091638065u : (p.x < 112. ? 286261268u : (p.x < 120. ? 16451u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 36000256u : (p.x < 176. ? 139776u : (p.x < 184. ? 286340130u : (p.x < 192. ? 342110515u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 81. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 858993425u : (p.x < 24. ? 286331153u : (p.x < 32. ? 2249777u : (p.x < 40. ? 0u : (p.x < 48. ? 536887296u : (p.x < 56. ? 286331716u : (p.x < 64. ? 1951749185u : (p.x < 72. ? 286331255u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331939u : (p.x < 112. ? 288621379u : (p.x < 120. ? 0u : (p.x < 128. ? 570425344u : (p.x < 136. ? 33562624u : (p.x < 144. ? 2236960u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 626349394u : (p.x < 176. ? 539107874u : (p.x < 184. ? 286331188u : (p.x < 192. ? 341132049u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 80. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331697u : (p.x < 24. ? 286331153u : (p.x < 32. ? 148241u : (p.x < 40. ? 2097664u : (p.x < 48. ? 0u : (p.x < 56. ? 288620544u : (p.x < 64. ? 1146373187u : (p.x < 72. ? 286340180u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1091637521u : (p.x < 104. ? 286343748u : (p.x < 112. ? 1127433251u : (p.x < 120. ? 2097698u : (p.x < 128. ? 572662306u : (p.x < 136. ? 572654114u : (p.x < 144. ? 572662306u : (p.x < 152. ? 33563170u : (p.x < 160. ? 0u : (p.x < 168. ? 2105940u : (p.x < 176. ? 1109393408u : (p.x < 184. ? 286331187u : (p.x < 192. ? 323310353u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 79. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 4469009u : (p.x < 40. ? 572662304u : (p.x < 48. ? 4456994u : (p.x < 56. ? 318767104u : (p.x < 64. ? 71573508u : (p.x < 72. ? 286474242u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 588321041u : (p.x < 104. ? 1127490100u : (p.x < 112. ? 572662306u : (p.x < 120. ? 572662306u : (p.x < 128. ? 572662306u : (p.x < 136. ? 1428300322u : (p.x < 144. ? 1431655717u : (p.x < 152. ? 8738u : (p.x < 160. ? 1073750528u : (p.x < 168. ? 36u : (p.x < 176. ? 8704u : (p.x < 184. ? 286331712u : (p.x < 192. ? 288637713u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 78. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 1430458641u : (p.x < 40. ? 1428300288u : (p.x < 48. ? 67248677u : (p.x < 56. ? 805306368u : (p.x < 64. ? 0u : (p.x < 72. ? 286470144u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 856756497u : (p.x < 104. ? 574890561u : (p.x < 112. ? 572662306u : (p.x < 120. ? 572523042u : (p.x < 128. ? 572662306u : (p.x < 136. ? 1431446050u : (p.x < 144. ? 1431655765u : (p.x < 152. ? 622862885u : (p.x < 160. ? 69358114u : (p.x < 168. ? 35790848u : (p.x < 176. ? 8704u : (p.x < 184. ? 286327040u : (p.x < 192. ? 286339857u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 77. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 1144066321u : (p.x < 40. ? 1431437312u : (p.x < 48. ? 2237781u : (p.x < 56. ? 16384u : (p.x < 64. ? 872415232u : (p.x < 72. ? 286273843u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 572801841u : (p.x < 112. ? 572662306u : (p.x < 120. ? 572662306u : (p.x < 128. ? 1431446050u : (p.x < 136. ? 1431655765u : (p.x < 144. ? 1431655765u : (p.x < 152. ? 1431642709u : (p.x < 160. ? 623203669u : (p.x < 168. ? 576016928u : (p.x < 176. ? 8738u : (p.x < 184. ? 286339328u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 76. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 1091637521u : (p.x < 40. ? 1428161056u : (p.x < 48. ? 35792213u : (p.x < 56. ? 209728u : (p.x < 64. ? 0u : (p.x < 72. ? 322122001u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 572662545u : (p.x < 112. ? 572662306u : (p.x < 120. ? 572662306u : (p.x < 128. ? 1431642658u : (p.x < 136. ? 1717986901u : (p.x < 144. ? 1431655765u : (p.x < 152. ? 1700091221u : (p.x < 160. ? 1449481830u : (p.x < 168. ? 576087653u : (p.x < 176. ? 2228770u : (p.x < 184. ? 286339376u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 75. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 17895697u : (p.x < 40. ? 1431438674u : (p.x < 48. ? 572675413u : (p.x < 56. ? 1127428130u : (p.x < 64. ? 1073750016u : (p.x < 72. ? 286331187u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 572662033u : (p.x < 112. ? 572670498u : (p.x < 120. ? 1144266786u : (p.x < 128. ? 894771747u : (p.x < 136. ? 1717986900u : (p.x < 144. ? 1449551462u : (p.x < 152. ? 1701205333u : (p.x < 160. ? 1717986918u : (p.x < 168. ? 1431725670u : (p.x < 176. ? 2228773u : (p.x < 184. ? 286331155u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 74. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 17895697u : (p.x < 40. ? 1700091221u : (p.x < 48. ? 572872022u : (p.x < 56. ? 606356002u : (p.x < 64. ? 858783744u : (p.x < 72. ? 286331155u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 572662579u : (p.x < 112. ? 539247172u : (p.x < 120. ? 823337506u : (p.x < 128. ? 341123633u : (p.x < 136. ? 1719109734u : (p.x < 144. ? 1432774246u : (p.x < 152. ? 1449547093u : (p.x < 160. ? 1717986917u : (p.x < 168. ? 1700161158u : (p.x < 176. ? 840957990u : (p.x < 184. ? 286475025u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 73. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 17895697u : (p.x < 40. ? 1700091221u : (p.x < 48. ? 572872022u : (p.x < 56. ? 572670498u : (p.x < 64. ? 286457858u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1125191953u : (p.x < 104. ? 860181845u : (p.x < 112. ? 607339569u : (p.x < 120. ? 825442850u : (p.x < 128. ? 874660627u : (p.x < 136. ? 1717995620u : (p.x < 144. ? 1431656038u : (p.x < 152. ? 1717986917u : (p.x < 160. ? 1717986918u : (p.x < 168. ? 1716872806u : (p.x < 176. ? 322109477u : (p.x < 184. ? 286470929u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 72. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 17895697u : (p.x < 40. ? 1717986901u : (p.x < 48. ? 572872022u : (p.x < 56. ? 35791394u : (p.x < 64. ? 286340128u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1125191953u : (p.x < 104. ? 288642389u : (p.x < 112. ? 591671603u : (p.x < 120. ? 1431589954u : (p.x < 128. ? 1163220293u : (p.x < 136. ? 1717995601u : (p.x < 144. ? 1716868454u : (p.x < 152. ? 1753778278u : (p.x < 160. ? 1717986918u : (p.x < 168. ? 1143100758u : (p.x < 176. ? 286339074u : (p.x < 184. ? 286331697u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 71. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 823202065u : (p.x < 40. ? 1717986901u : (p.x < 48. ? 576018006u : (p.x < 56. ? 2236962u : (p.x < 64. ? 286331648u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1393627409u : (p.x < 104. ? 286479701u : (p.x < 112. ? 1093865779u : (p.x < 120. ? 1717916468u : (p.x < 128. ? 1164334422u : (p.x < 136. ? 1717986881u : (p.x < 144. ? 1717982566u : (p.x < 152. ? 2290649224u : (p.x < 160. ? 1717986918u : (p.x < 168. ? 861230438u : (p.x < 176. ? 286339649u : (p.x < 184. ? 286331697u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 70. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1717986901u : (p.x < 48. ? 576018006u : (p.x < 56. ? 139810u : (p.x < 64. ? 286331136u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 823202065u : (p.x < 104. ? 1430537556u : (p.x < 112. ? 823407445u : (p.x < 120. ? 1448362291u : (p.x < 128. ? 1449551462u : (p.x < 136. ? 1717986900u : (p.x < 144. ? 2004309606u : (p.x < 152. ? 1717987464u : (p.x < 160. ? 1717921382u : (p.x < 168. ? 1163220310u : (p.x < 176. ? 823198737u : (p.x < 184. ? 286331171u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 69. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1717986899u : (p.x < 48. ? 572872294u : (p.x < 56. ? 536879650u : (p.x < 64. ? 286331154u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 2254853460u : (p.x < 112. ? 286332008u : (p.x < 120. ? 856756497u : (p.x < 128. ? 1718126726u : (p.x < 136. ? 1717986918u : (p.x < 144. ? 2270516838u : (p.x < 152. ? 1718126728u : (p.x < 160. ? 1431656038u : (p.x < 168. ? 874653013u : (p.x < 176. ? 858854161u : (p.x < 184. ? 286331187u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 68. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1701209137u : (p.x < 48. ? 572876390u : (p.x < 56. ? 1073741856u : (p.x < 64. ? 286331155u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1393627409u : (p.x < 104. ? 2290509414u : (p.x < 112. ? 1091782536u : (p.x < 120. ? 1360073524u : (p.x < 128. ? 1720223878u : (p.x < 136. ? 1717986918u : (p.x < 144. ? 1716938342u : (p.x < 152. ? 1717986918u : (p.x < 160. ? 35804502u : (p.x < 168. ? 1109533218u : (p.x < 176. ? 322113809u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 67. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1717981457u : (p.x < 48. ? 36005478u : (p.x < 56. ? 807534592u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1695617297u : (p.x < 104. ? 2290648678u : (p.x < 112. ? 1683392648u : (p.x < 120. ? 1700091752u : (p.x < 128. ? 1720223880u : (p.x < 136. ? 2288412262u : (p.x < 144. ? 1431725672u : (p.x < 152. ? 1717986918u : (p.x < 160. ? 572675414u : (p.x < 168. ? 1377968640u : (p.x < 176. ? 288559377u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 66. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1431580945u : (p.x < 48. ? 1109747302u : (p.x < 56. ? 268645172u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1697714449u : (p.x < 104. ? 2290649190u : (p.x < 112. ? 2290649224u : (p.x < 120. ? 2290649224u : (p.x < 128. ? 1753778312u : (p.x < 136. ? 1751541350u : (p.x < 144. ? 1449551464u : (p.x < 152. ? 1717986917u : (p.x < 160. ? 572675429u : (p.x < 168. ? 1075978754u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 65. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1414795537u : (p.x < 48. ? 321214053u : (p.x < 56. ? 839979281u : (p.x < 64. ? 286331155u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2254901521u : (p.x < 104. ? 2290649224u : (p.x < 112. ? 2290648680u : (p.x < 120. ? 1451788424u : (p.x < 128. ? 1753778310u : (p.x < 136. ? 1717986917u : (p.x < 144. ? 1717995622u : (p.x < 152. ? 1431655766u : (p.x < 160. ? 572663074u : (p.x < 168. ? 805446178u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 64. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1128468753u : (p.x < 48. ? 324363877u : (p.x < 56. ? 1141969169u : (p.x < 64. ? 286331155u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2290630929u : (p.x < 104. ? 1718126728u : (p.x < 112. ? 2290648678u : (p.x < 120. ? 1183352968u : (p.x < 128. ? 2290648678u : (p.x < 136. ? 1717982533u : (p.x < 144. ? 1717986918u : (p.x < 152. ? 572876390u : (p.x < 160. ? 572653568u : (p.x < 168. ? 318906914u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 63. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 858853649u : (p.x < 48. ? 289760852u : (p.x < 56. ? 856756497u : (p.x < 64. ? 286331188u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2290639633u : (p.x < 104. ? 2290649192u : (p.x < 112. ? 2290649224u : (p.x < 120. ? 1451788424u : (p.x < 128. ? 2290648676u : (p.x < 136. ? 859068023u : (p.x < 144. ? 1449551443u : (p.x < 152. ? 572876390u : (p.x < 160. ? 572522496u : (p.x < 168. ? 858923554u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 62. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 289760577u : (p.x < 56. ? 1127289617u : (p.x < 64. ? 286331188u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2290648337u : (p.x < 104. ? 2290649192u : (p.x < 112. ? 2257094792u : (p.x < 120. ? 1720223880u : (p.x < 128. ? 2290648644u : (p.x < 136. ? 288786568u : (p.x < 144. ? 1431725361u : (p.x < 152. ? 36001109u : (p.x < 160. ? 572522530u : (p.x < 168. ? 856900642u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 61. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 288428305u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 321212977u : (p.x < 56. ? 823211057u : (p.x < 64. ? 286462787u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2290648337u : (p.x < 104. ? 2290649224u : (p.x < 112. ? 2254997640u : (p.x < 120. ? 1753778312u : (p.x < 128. ? 2290648645u : (p.x < 136. ? 288852104u : (p.x < 144. ? 1431651089u : (p.x < 152. ? 1127503189u : (p.x < 160. ? 1107296853u : (p.x < 168. ? 286331715u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 60. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 288428305u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 844244017u : (p.x < 56. ? 286339075u : (p.x < 64. ? 322192145u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2290648081u : (p.x < 104. ? 2290649224u : (p.x < 112. ? 2290649224u : (p.x < 120. ? 2290649224u : (p.x < 128. ? 2290639942u : (p.x < 136. ? 286488712u : (p.x < 144. ? 1431572753u : (p.x < 152. ? 823211093u : (p.x < 160. ? 872423972u : (p.x < 168. ? 286331187u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 59. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 572797201u : (p.x < 56. ? 286339074u : (p.x < 64. ? 286462225u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2288542737u : (p.x < 104. ? 2290649224u : (p.x < 112. ? 2290649224u : (p.x < 120. ? 1753778312u : (p.x < 128. ? 2290504534u : (p.x < 136. ? 286340968u : (p.x < 144. ? 1431572753u : (p.x < 152. ? 286331733u : (p.x < 160. ? 841097764u : (p.x < 168. ? 1125191953u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 58. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 858853649u : (p.x < 56. ? 288637472u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1717986577u : (p.x < 104. ? 2290509414u : (p.x < 112. ? 1753778312u : (p.x < 120. ? 1718126694u : (p.x < 128. ? 1717978214u : (p.x < 136. ? 286331190u : (p.x < 144. ? 1716719889u : (p.x < 152. ? 286331189u : (p.x < 160. ? 575808563u : (p.x < 168. ? 1125191955u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 57. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 289555505u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1700090897u : (p.x < 104. ? 1717986918u : (p.x < 112. ? 1717986950u : (p.x < 120. ? 1449551462u : (p.x < 128. ? 878003557u : (p.x < 136. ? 286331153u : (p.x < 144. ? 1715540241u : (p.x < 152. ? 286331156u : (p.x < 160. ? 572670737u : (p.x < 168. ? 1125191956u : (p.x < 176. ? 286331155u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 56. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 289681681u : (p.x < 64. ? 289686289u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1377977105u : (p.x < 104. ? 1716872533u : (p.x < 112. ? 1432774246u : (p.x < 120. ? 626353766u : (p.x < 128. ? 1145328997u : (p.x < 136. ? 286331155u : (p.x < 144. ? 1429278993u : (p.x < 152. ? 286331156u : (p.x < 160. ? 574828817u : (p.x < 168. ? 823202068u : (p.x < 176. ? 286331187u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 55. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 860033297u : (p.x < 64. ? 1109533235u : (p.x < 72. ? 286331716u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 572797201u : (p.x < 104. ? 572674594u : (p.x < 112. ? 1381324117u : (p.x < 120. ? 626349397u : (p.x < 128. ? 1717986642u : (p.x < 136. ? 286331156u : (p.x < 144. ? 1410404625u : (p.x < 152. ? 286331156u : (p.x < 160. ? 875639057u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331187u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 54. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 858853649u : (p.x < 64. ? 572662340u : (p.x < 72. ? 286330914u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 572723473u : (p.x < 104. ? 572662306u : (p.x < 112. ? 572662306u : (p.x < 120. ? 626139682u : (p.x < 128. ? 1717986642u : (p.x < 136. ? 286331155u : (p.x < 144. ? 856756497u : (p.x < 152. ? 286331188u : (p.x < 160. ? 319897873u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331203u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 53. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 35791393u : (p.x < 72. ? 288555008u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 607195409u : (p.x < 104. ? 608444962u : (p.x < 112. ? 572662306u : (p.x < 120. ? 572662306u : (p.x < 128. ? 1181115682u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331187u : (p.x < 160. ? 286535953u : (p.x < 168. ? 288559377u : (p.x < 176. ? 286331187u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 52. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 2236961u : (p.x < 72. ? 872415744u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 823202065u : (p.x < 104. ? 1091646259u : (p.x < 112. ? 572662308u : (p.x < 120. ? 1428300288u : (p.x < 128. ? 895903061u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 288633649u : (p.x < 168. ? 288370961u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 51. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 548u : (p.x < 72. ? 1073741824u : (p.x < 80. ? 286331155u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 49u : (p.x < 120. ? 1377959936u : (p.x < 128. ? 324359765u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 289686289u : (p.x < 168. ? 288359473u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 50. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 823202065u : (p.x < 64. ? 2u : (p.x < 72. ? 0u : (p.x < 80. ? 286331188u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 49u : (p.x < 120. ? 1377959936u : (p.x < 128. ? 288642661u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 289407761u : (p.x < 168. ? 858783747u : (p.x < 176. ? 286331155u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 49. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 1091637553u : (p.x < 64. ? 2u : (p.x < 72. ? 0u : (p.x < 80. ? 286340162u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 131137u : (p.x < 120. ? 1413611520u : (p.x < 128. ? 286344533u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 318779665u : (p.x < 168. ? 858783747u : (p.x < 176. ? 321982739u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 48. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 1125191953u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 322191872u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 2236977u : (p.x < 120. ? 1414660096u : (p.x < 128. ? 286331989u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 872681745u : (p.x < 168. ? 1125401649u : (p.x < 176. ? 858853649u : (p.x < 184. ? 286339908u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 47. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 588321041u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 841097760u : (p.x < 88. ? 286331155u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 35791889u : (p.x < 120. ? 1431634464u : (p.x < 128. ? 286331685u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 876810513u : (p.x < 168. ? 856895761u : (p.x < 176. ? 856756499u : (p.x < 184. ? 286457892u : (p.x < 192. ? 286331155u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 46. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 1125191953u : (p.x < 64. ? 2u : (p.x < 72. ? 0u : (p.x < 80. ? 1377968674u : (p.x < 88. ? 286331156u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 622994193u : (p.x < 120. ? 1431655762u : (p.x < 128. ? 286331170u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 856756497u : (p.x < 168. ? 286331699u : (p.x < 176. ? 319885585u : (p.x < 184. ? 826277891u : (p.x < 192. ? 286331665u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 45. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 823202065u : (p.x < 64. ? 37u : (p.x < 72. ? 0u : (p.x < 80. ? 626348578u : (p.x < 88. ? 286331156u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1428308241u : (p.x < 120. ? 1431655765u : (p.x < 128. ? 286331170u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 823202065u : (p.x < 168. ? 286340163u : (p.x < 176. ? 286331185u : (p.x < 184. ? 323174403u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 44. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 84u : (p.x < 72. ? 0u : (p.x < 80. ? 626337106u : (p.x < 88. ? 286331155u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1381318929u : (p.x < 120. ? 626336293u : (p.x < 128. ? 286331682u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 858853649u : (p.x < 176. ? 286331185u : (p.x < 184. ? 321991441u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 43. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1363u : (p.x < 72. ? 536870912u : (p.x < 80. ? 844440866u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1381318929u : (p.x < 120. ? 1163206658u : (p.x < 128. ? 286331682u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 876810513u : (p.x < 184. ? 286339345u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 42. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 539317825u : (p.x < 72. ? 572661760u : (p.x < 80. ? 337990946u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 576017169u : (p.x < 120. ? 626336293u : (p.x < 128. ? 1125192738u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 1163146001u : (p.x < 184. ? 286343953u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 41. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 40265009u : (p.x < 72. ? 572661760u : (p.x < 80. ? 321213781u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1431655441u : (p.x < 120. ? 1431655765u : (p.x < 128. ? 1160844322u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 1163220291u : (p.x < 184. ? 286544915u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 40. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449476369u : (p.x < 72. ? 1377959970u : (p.x < 80. ? 321017173u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1431721489u : (p.x < 120. ? 1431655765u : (p.x < 128. ? 1163072324u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 1431655764u : (p.x < 184. ? 288642373u : (p.x < 192. ? 286331153u : 856756497u)))))))))))))))))))))))) : v;
	v = p.y == 39. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449398545u : (p.x < 72. ? 572653602u : (p.x < 80. ? 287462741u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1431725073u : (p.x < 120. ? 1112888661u : (p.x < 128. ? 844370195u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1125191953u : (p.x < 176. ? 1700091221u : (p.x < 184. ? 289756502u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 38. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449333009u : (p.x < 72. ? 572662354u : (p.x < 80. ? 289547554u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1716932881u : (p.x < 120. ? 844453478u : (p.x < 128. ? 877924625u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1430458641u : (p.x < 176. ? 1431655765u : (p.x < 184. ? 341141094u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 37. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449333009u : (p.x < 72. ? 572662354u : (p.x < 80. ? 288637474u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1717977361u : (p.x < 120. ? 1163220326u : (p.x < 128. ? 341119249u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431650577u : (p.x < 176. ? 1431655765u : (p.x < 184. ? 1163224678u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 36. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449398545u : (p.x < 72. ? 572661794u : (p.x < 80. ? 286331712u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1717973265u : (p.x < 120. ? 844453222u : (p.x < 128. ? 324342033u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431655185u : (p.x < 176. ? 1431655765u : (p.x < 184. ? 1431656038u : (p.x < 192. ? 286331156u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 35. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449398545u : (p.x < 72. ? 35791362u : (p.x < 80. ? 286331184u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1717899537u : (p.x < 120. ? 324359510u : (p.x < 128. ? 288428305u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431655185u : (p.x < 176. ? 1431655765u : (p.x < 184. ? 1431656038u : (p.x < 192. ? 286331154u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 34. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1432621329u : (p.x < 72. ? 35792213u : (p.x < 80. ? 286331152u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1717834001u : (p.x < 120. ? 324359510u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431650577u : (p.x < 176. ? 1700091221u : (p.x < 184. ? 1431656038u : (p.x < 192. ? 286331189u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 33. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1432686865u : (p.x < 72. ? 572675413u : (p.x < 80. ? 286331155u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1430327569u : (p.x < 120. ? 289756501u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431646481u : (p.x < 176. ? 1700091221u : (p.x < 184. ? 1431656038u : (p.x < 192. ? 286331186u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 32. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1432686865u : (p.x < 72. ? 1109534037u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1429278993u : (p.x < 120. ? 286545237u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431638289u : (p.x < 176. ? 1431586133u : (p.x < 184. ? 1431655765u : (p.x < 192. ? 286331156u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 31. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1432686865u : (p.x < 72. ? 340927829u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1410404625u : (p.x < 120. ? 286340181u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431572753u : (p.x < 176. ? 1410404660u : (p.x < 184. ? 1431655765u : (p.x < 192. ? 286331155u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 30. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1432760593u : (p.x < 72. ? 289690965u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 856756497u : (p.x < 120. ? 286331185u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 322113809u : (p.x < 176. ? 823202065u : (p.x < 184. ? 1112888660u : (p.x < 192. ? 286331153u : 286462225u)))))))))))))))))))))))) : v;
	v = p.y == 29. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1431712017u : (p.x < 72. ? 286545237u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 807543361u : (p.x < 192. ? 286331153u : 288559377u)))))))))))))))))))))))) : v;
	v = p.y == 28. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1431646481u : (p.x < 72. ? 286340181u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 288637713u : (p.x < 192. ? 286331153u : 318837009u)))))))))))))))))))))))) : v;
	v = p.y == 27. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1431650577u : (p.x < 72. ? 286331205u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 288428305u : (p.x < 192. ? 286331153u : 288559377u)))))))))))))))))))))))) : v;
	v = p.y == 26. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1431651089u : (p.x < 72. ? 286331155u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 289607953u : (p.x < 192. ? 286331153u : 286475025u)))))))))))))))))))))))) : v;
	v = p.y == 25. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1163215121u : (p.x < 72. ? 286331155u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 286331153u : (p.x < 192. ? 286331153u : 286344513u)))))))))))))))))))))))) : v;
	v = p.y == 24. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 878007057u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 286331153u : (p.x < 192. ? 286331153u : 286340163u)))))))))))))))))))))))) : v;
	v = p.y == 23. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 894792721u : 286331153u)))))))) : v;
	v = p.y == 22. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 341144593u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331665u : 286331153u)))))))))))))))))) : v;
	v = p.y == 21. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 290808849u : (p.x < 72. ? 286339345u : 286331153u))))))))) : v;
	v = p.y == 20. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 341066513u : 286331153u)))))))) : v;
	v = p.y == 19. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 339947793u : (p.x < 72. ? 286331153u : (p.x < 80. ? 823202065u : 286331153u)))))))))) : v;
	v = p.y == 18. ? 286331153u : v;
	v = p.y == 17. ? 286331153u : v;
	v = p.y == 16. ? 286331153u : v;
	v = p.y == 15. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286339345u : 286331153u))))))))) : v;
	v = p.y == 14. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286487875u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331185u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331697u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 13. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1949372689u : (p.x < 72. ? 286340471u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 823202065u : (p.x < 136. ? 286476151u : (p.x < 144. ? 286331153u : (p.x < 152. ? 1145324593u : (p.x < 160. ? 1199011700u : (p.x < 168. ? 1127499588u : (p.x < 176. ? 2004318068u : (p.x < 184. ? 286471236u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 12. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 2536575249u : (p.x < 72. ? 286340983u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 856756497u : (p.x < 128. ? 2004304657u : (p.x < 136. ? 2004326809u : (p.x < 144. ? 823211127u : (p.x < 152. ? 2576980343u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2541328793u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 1199012249u : (p.x < 192. ? 286331187u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 11. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 2541187857u : (p.x < 72. ? 286340985u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 856756497u : (p.x < 104. ? 860123955u : (p.x < 112. ? 1145324339u : (p.x < 120. ? 2004300595u : (p.x < 128. ? 2576979831u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2272556953u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 860321656u : 286331187u)))))))))))))))))))))))) : v;
	v = p.y == 10. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 876884241u : (p.x < 56. ? 286331699u : (p.x < 64. ? 2576970803u : (p.x < 72. ? 286332825u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2000892721u : (p.x < 104. ? 2576906137u : (p.x < 112. ? 2040108953u : (p.x < 120. ? 2576971671u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576840569u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2006555033u : 286331767u)))))))))))))))))))))))) : v;
	v = p.y == 9. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 1145254161u : (p.x < 40. ? 322188100u : (p.x < 48. ? 2040099601u : (p.x < 56. ? 1950840695u : (p.x < 64. ? 2576971639u : (p.x < 72. ? 286341017u : (p.x < 80. ? 286331153u : (p.x < 88. ? 856756497u : (p.x < 96. ? 2576840548u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576979832u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2576980377u : 286332025u)))))))))))))))))))))))) : v;
	v = p.y == 8. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 2004304657u : (p.x < 32. ? 2576840567u : (p.x < 40. ? 2004383641u : (p.x < 48. ? 2576971639u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2576980377u : (p.x < 72. ? 286345081u : (p.x < 80. ? 286331153u : (p.x < 88. ? 2000752913u : (p.x < 96. ? 2576980359u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2006555033u : 286331188u)))))))))))))))))))))))) : v;
	v = p.y == 7. ? (p.x < 8. ? 286339891u : (p.x < 16. ? 1145320241u : (p.x < 24. ? 2576980343u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2004318073u : (p.x < 72. ? 860321655u : (p.x < 80. ? 286339891u : (p.x < 88. ? 2575791155u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 932813209u : 858993459u)))))))))))))))))))))))) : v;
	v = p.y == 6. ? (p.x < 8. ? 1145337719u : (p.x < 16. ? 2576840564u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2006555033u : (p.x < 64. ? 1199011703u : (p.x < 72. ? 1145324612u : (p.x < 80. ? 1145337719u : (p.x < 88. ? 2576980343u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2004457881u : 2004326263u)))))))))))))))))))))))) : v;
	v = p.y == 5. ? (p.x < 8. ? 2004318071u : (p.x < 16. ? 2004318071u : (p.x < 24. ? 2576980375u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2308544921u : (p.x < 64. ? 1165457272u : (p.x < 72. ? 1967408196u : (p.x < 80. ? 2004318071u : (p.x < 88. ? 2576975735u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2004457881u : 2004318071u)))))))))))))))))))))))) : v;
	v = p.y == 4. ? (p.x < 8. ? 2004318071u : (p.x < 16. ? 2541188983u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2040109465u : (p.x < 72. ? 2004304999u : (p.x < 80. ? 2574743415u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2004326809u : 2004318071u)))))))))))))))))))))))) : v;
	v = p.y == 3. ? (p.x < 8. ? 2004318071u : (p.x < 16. ? 2576840567u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2576980377u : (p.x < 72. ? 2576840569u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2576980377u : 2004318071u)))))))))))))))))))))))) : v;
	v = p.y == 2. ? (p.x < 8. ? 2004453241u : (p.x < 16. ? 2004318071u : (p.x < 24. ? 2576980375u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2576980377u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2576980377u : 2576979833u)))))))))))))))))))))))) : v;
	v = p.y == 1. ? 2576980377u : v;
	v = p.y == 0. ? 2576980377u : v;
    v = p.x >= 0. && p.x < 200. ? v : 0u;

    float i = float((v >> uint(4. * p.x)) & 15u);
    vec3 color = vec3(0.26, 0.31, 0.16);
    color = i == 1. ? vec3(0.1, 0.24, 0.46) : color;
    color = i == 2. ? vec3(0.4, 0.43, 0.26) : color;
    color = i == 3. ? vec3(0.28, 0.35, 0.47) : color;
    color = i == 4. ? vec3(0.35, 0.38, 0.42) : color;
    color = i == 5. ? vec3(0.51, 0.45, 0.36) : color;
    color = i == 6. ? vec3(0.71, 0.63, 0.53) : color;
    color = i == 7. ? vec3(0.64, 0.66, 0.7) : color;
    color = i == 8. ? vec3(0.88, 0.79, 0.69) : color;
    color = i == 9. ? vec3(0.86, 0.87, 0.89) : color;
    
    return vec4(color, 1.0);
}

// Function 337
vec3 sampledTexture( in vec2 p )
{
    vec3 color;    
    float u = p.x;
    float v = p.y;
    float f, pattern;

        // background
    color = mix(bg, white, crossing(tranScale(u, 0.5/6.75, 6.75), tranScale(v, 0.5/6.75, 6.75)));
    color = mix(color, grey, crossing(tranScale(u, 0.5/6.45, 6.45), tranScale(v, 0.5/6.45, 6.45)));
    
    // centerlines, just add as patterns do not intersect due to filter
    f = 1.0-rect(tranScale(u, 0.5/6.45, 6.45), tranScale(v, 0.5/6.45, 6.45));
    pattern = f * (line(tranScale(u, 1.0/0.2, 0.3)) +
                   line(tranScale(v, 1.0/0.2, 0.3)) +
                   line(tranScale(u, 1.0/0.2, -0.1)) +
                   line(tranScale(v, 1.0/0.2, -0.1)) );
	// dashed, just add as patterns do not intersect due to filter
    f = 1.0-rect(tranScale(u, 0.5/10.7, 10.7), tranScale(v, 0.5/10.7, 10.7));
    pattern += f * (line(tranScale(u, 1.0/0.15, 3.45))*dash(tranScale(v, 1.0/2.0, 1.0), 0.6) +
                    line(tranScale(v, 1.0/0.15, 3.45))*dash(tranScale(u, 1.0/2.0, 1.0), 0.6) +
                    line(tranScale(u, 1.0/0.15, -3.3))*dash(tranScale(v, 1.0/2.0, 1.0), 0.6) +
                    line(tranScale(v, 1.0/0.15, -3.3))*dash(tranScale(u, 1.0/2.0, 1.0), 0.6));
    // stop lines, add again
    pattern += rect(tranScale(u, 1.0/0.4, -9.55), tranScale(v, 1.0/5.55, -0.6)) +
               rect(tranScale(u, 1.0/0.4,  10.05), tranScale(v, 1.0/5.55, -0.6)) +
               rect(tranScale(u, 1.0/0.4, -9.55), tranScale(v, 1.0/5.55, 6.15)) +
               rect(tranScale(u, 1.0/0.4,  10.05), tranScale(v, 1.0/5.55, 6.15)) +
               rect(tranScale(v, 1.0/0.4, -9.55), tranScale(u, 1.0/5.55, -0.6)) +
               rect(tranScale(v, 1.0/0.4,  10.05), tranScale(u, 1.0/5.55, -0.6)) +
               rect(tranScale(v, 1.0/0.4, -9.55), tranScale(u, 1.0/5.55, 6.15)) +
               rect(tranScale(v, 1.0/0.4,  10.05), tranScale(u, 1.0/5.55, 6.15));
    // pedestrian crossing, add again
    pattern += rect(tranScale(v, 1.0/2.0,  9.05), tranScale(u, 1.0/5.55, 6.15))*dash(tranScale(u, 1.0/0.6, 5.925), 0.5) +
               rect(tranScale(v, 1.0/2.0,  9.05), tranScale(u, 1.0/5.55, -0.6))*dash(tranScale(u, 1.0/0.6, -0.825), 0.5) +
               rect(tranScale(v, 1.0/2.0,  -7.05), tranScale(u, 1.0/5.55, 6.15))*dash(tranScale(u, 1.0/0.6, 5.925), 0.5) +
               rect(tranScale(v, 1.0/2.0,  -7.05), tranScale(u, 1.0/5.55, -0.6))*dash(tranScale(u, 1.0/0.6, -0.825), 0.5) +
               rect(tranScale(u, 1.0/2.0,  9.05), tranScale(v, 1.0/5.55, 6.15))*dash(tranScale(v, 1.0/0.6, 5.925), 0.5) +
               rect(tranScale(u, 1.0/2.0,  9.05), tranScale(v, 1.0/5.55, -0.6))*dash(tranScale(v, 1.0/0.6, -0.825), 0.5) +
               rect(tranScale(u, 1.0/2.0,  -7.05), tranScale(v, 1.0/5.55, 6.15))*dash(tranScale(v, 1.0/0.6, 5.925), 0.5) +
               rect(tranScale(u, 1.0/2.0,  -7.05), tranScale(v, 1.0/5.55, -0.6))*dash(tranScale(v, 1.0/0.6, -0.825), 0.5);
    color = mix(color, white, pattern);

    return color;
}

// Function 338
float getSampleDim3(int sampleIndex,vec2 fragCoord)
{
	return fract(getDimensionHash(3,fragCoord) + radicalInverse(sampleIndex, 7));
}

// Function 339
vec4 getSample(in vec2 fragCoord )
{
	vec2 uv = (2.0 * fragCoord.xy - iResolution.xy) / min(iResolution.x, iResolution.y);
    
    vec4 color_mul = vec4(1.0);
    vec4 color_add = vec4(0.0);
    
    float stheta = sin(0.125 * iTime);
    float ctheta = cos(0.125 * iTime);
    float pitch = 0.5;
    mat3 turn_mat = 
        mat3(ctheta, 0.0, stheta,
             0.0, 1.0, 0.0,
             -stheta, 0.0, ctheta) *
        mat3(1.0, 0.0, 0.0,
                         0.0, cos(pitch), sin(pitch),
                         0.0, -sin(pitch), cos(pitch));
    
    vec3 ray_orig = turn_mat * vec3(0.0, 0.0, -1.0);
    
    vec3 ray_dir = normalize(turn_mat * vec3(uv, 1.0));
    
    float bounced = 0.0;
    
	vec3 hit = raymarch(ray_orig, ray_dir);
    if (torus_sdf(hit) < 0.1) {
        float glow;
    	vec3 n = tor_norm(hit, glow);
        ray_dir = normalize(reflect(ray_dir, n));
        ray_orig = hit;
        
        color_add.b = 0.5 * smoothstep(0.3, 0.4, glow);
        bounced = 1.0;
        
    }
    
	return color_mul * background(ray_orig, ray_dir, bounced) + color_add;
}

// Function 340
vec3 sampleWeights(float i) {
	return vec3((1.0 - i) * (1.0 - i), greenWeight() * i * (1.0 - i), i * i);
}

// Function 341
vec3 sample_grad_dist(vec2 uv, float font_size) {
    
    vec3 grad_dist = (textureLod(iChannel0, uv, 0.).yzw - FONT_TEX_BIAS) * font_size;

    grad_dist.y = -grad_dist.y;
    grad_dist.xy = normalize(grad_dist.xy + 1e-5);
    
    return grad_dist;
    
}

// Function 342
vec3 PBR_visitSamples(vec3 V, vec3 N, float roughness, bool metallic, vec3 ior_n, vec3 ior_k )
{
    //Direct relection vector
    vec3 vCenter = reflect(-V,N);
    
    //------------------------------------------------
	//  Randomized Samples : more realistic, but
    //  a lot of samples before it stabilizes 
    //------------------------------------------------
    float randomness_range = 0.75; //Cover only the closest 75% of the distribution. Reduces range, but improves stability.
    float fIdx = 0.0;              //valid range = [0.5-1.0]. Note : it is physically correct at 1.0.
    const int ITER_RDM = 05;
    const float w_rdm = 1.0/float(ITER_RDM);
    vec3 totalRandom = vec3(0.0);
    for(int i=0; i < ITER_RDM; ++i)
    {
        //Random jitter note : very sensitive to hash quality (patterns & artifacts).
        vec2 jitter = hash22(fIdx*10.0+vCenter.xy*100.0);
    	float angularRange = 0.;    
        vec3 sampleDir    = PBR_nudgeSample(vCenter, roughness, jitter.x*randomness_range, jitter.y, angularRange);
        vec3 sampleColor  = PBR_HDRCubemap( sampleDir, angularRange/MIPMAP_SWITCH);
        vec3 contribution = PBR_Equation(V, sampleDir, N, roughness, ior_n, ior_k, metallic, true)*w_rdm;
    	totalRandom += contribution*sampleColor;
		++fIdx;
    }
    
    //------------------------------------------------
	//  Fixed Samples : More stable, but creates
    //  sampling pattern artifacts and the reach is
    //  limited.
    //------------------------------------------------
    fIdx = 0.0;
    const int ITER_FIXED = 15;
    const float w_fixed = 1.0/float(ITER_FIXED); //Sample
    vec3 totalFixed = vec3(0.0);
    for(int i=0; i < ITER_FIXED; ++i)
    {
        //Stable pseudo-random jitter (to improve stability with low sample count)
        //Beware here! second component controls the sampling pattern "swirl", and it must be choosen 
        //             so that samples do not align by doing complete 360deg cycles at each iteration.
        vec2 jitter = vec2( clamp(w_fixed*fIdx,0.0,0.50),
                            fract(w_fixed*fIdx*1.25)+3.14*fIdx);
        float angularRange = 0.;
        vec3 sampleDir    = PBR_nudgeSample(vCenter, roughness, jitter.x, jitter.y, angularRange);
        vec3 sampleColor  = PBR_HDRCubemap( sampleDir, angularRange/MIPMAP_SWITCH);
        vec3 contribution = PBR_Equation(V, sampleDir, N, roughness, ior_n, ior_k, metallic, true)*w_fixed;
        totalFixed += contribution*sampleColor;
		++fIdx;
    }
    
    return (totalRandom*float(ITER_RDM)+totalFixed*float(ITER_FIXED))/(float(ITER_RDM)+float(ITER_FIXED));
}

// Function 343
vec4 grassTexture(vec3 pos, vec3 nor)
{
    
    float g = texture(iChannel1, pos.xz*.5).x;
    float s = texture(iChannel1, pos.xz*.015).x*.2;
    
    
    vec3 flower = texture(iChannel2, pos.xz*.15).xyz;
    float rand = texture(iChannel1, pos.xz*.003).x;
    rand *= rand*rand;
    
    flower =pow(flower,vec3(8, 15, 5)) *10. * rand;
    vec4 mat = vec4(g*.05+s, g*.65, 0, g*.1);
    mat.xyz += flower;

    // Do the red ground lines...
    pos = fract(pos);
    mat = mix(mat, vec4(.2, 0,0,0), smoothstep(.05, .0,min(pos.x, pos.z))
              					  + smoothstep(.95, 1.,max(pos.x, pos.z)));

    
	return min(mat, 1.0);
}

// Function 344
vec3 sampleReflectionMap(vec3 sp, float lodBias){    
    vec3 color = SRGBtoLINEAR(textureLod(reflectTex, sp, lodBias).rgb);
    #if defined (HDR_FOR_POORS)
    	color *= 1.0 + 2.0*smoothstep(0.7, 1.0, dot(LUMA, color)); //HDR for poors
   	#endif
    return color;
}

// Function 345
vec3 LPV_DoubleSample(vec3 p, vec3 d, vec4 Cxp, vec4 Cxn, vec4 Cyp, vec4 Cyn, vec4 Czp, vec4 Czn) {
    vec3 Sqd=d*d;
    vec3 ILP=Cxp.xyz*Sqd.x+Cyp.xyz*Sqd.y+Czp.xyz*Sqd.z;
    vec3 ILN=Cxn.xyz*Sqd.x+Cyn.xyz*Sqd.y+Czn.xyz*Sqd.z;
    return ILP/(Cxp.w*Sqd.x+Cyp.w*Sqd.y+Czp.w*Sqd.z+0.001)+
            ILN/(Cxn.w*Sqd.x+Cyn.w*Sqd.y+Czn.w*Sqd.z+0.001);
}

// Function 346
vec4 textureNoTile( sampler2D samp, in vec2 uv )
{
    vec2 iuv = floor( uv );
    vec2 fuv = fract( uv );

#ifdef USEHASH    
    // generate per-tile transform (needs GL_NEAREST_MIPMAP_LINEARto work right)
    vec4 ofa = texture( iChannel1, (iuv + vec2(0.5,0.5))/256.0 );
    vec4 ofb = texture( iChannel1, (iuv + vec2(1.5,0.5))/256.0 );
    vec4 ofc = texture( iChannel1, (iuv + vec2(0.5,1.5))/256.0 );
    vec4 ofd = texture( iChannel1, (iuv + vec2(1.5,1.5))/256.0 );
#else
    // generate per-tile transform
    vec4 ofa = hash4( iuv + vec2(0.0,0.0) );
    vec4 ofb = hash4( iuv + vec2(1.0,0.0) );
    vec4 ofc = hash4( iuv + vec2(0.0,1.0) );
    vec4 ofd = hash4( iuv + vec2(1.0,1.0) );
#endif
    
    vec2 ddx = dFdx( uv );
    vec2 ddy = dFdy( uv );

    // transform per-tile uvs
    ofa.zw = sign(ofa.zw-0.5);
    ofb.zw = sign(ofb.zw-0.5);
    ofc.zw = sign(ofc.zw-0.5);
    ofd.zw = sign(ofd.zw-0.5);
    
    // uv's, and derivarives (for correct mipmapping)
    vec2 uva = uv*ofa.zw + ofa.xy; vec2 ddxa = ddx*ofa.zw; vec2 ddya = ddy*ofa.zw;
    vec2 uvb = uv*ofb.zw + ofb.xy; vec2 ddxb = ddx*ofb.zw; vec2 ddyb = ddy*ofb.zw;
    vec2 uvc = uv*ofc.zw + ofc.xy; vec2 ddxc = ddx*ofc.zw; vec2 ddyc = ddy*ofc.zw;
    vec2 uvd = uv*ofd.zw + ofd.xy; vec2 ddxd = ddx*ofd.zw; vec2 ddyd = ddy*ofd.zw;
        
    // fetch and blend
    vec2 b = smoothstep(0.25,0.75,fuv);
    
    if( useOld )
    {
        //original approach from iq
        return mix( mix( textureGrad( samp, uva, ddxa, ddya ), 
                         textureGrad( samp, uvb, ddxb, ddyb ), b.x ), 
                    mix( textureGrad( samp, uvc, ddxc, ddyc ),
                         textureGrad( samp, uvd, ddxd, ddyd ), b.x), b.y );

    }
    
    
    // huwb modification - modify blend based on relative brightness
    // to try to preserve intense features (don't add 50% white)
    vec4 A = textureGrad( samp, uva, ddxa, ddya );
    vec4 B = textureGrad( samp, uvb, ddxb, ddyb );
    vec4 C = textureGrad( samp, uvc, ddxc, ddyc );
    vec4 D = textureGrad( samp, uvd, ddxd, ddyd );
    
    vec4 AB = contrastBlend( A, B, b.x );
    vec4 CD = contrastBlend( C, D, b.x );
    return contrastBlend( AB, CD, b.y );
}

// Function 347
vec3 sampleWeights(float i) {
	return vec3(i*i*i, 46.666*pow((1.0-i)*i,1.0), (1.1 - i*0.12) * (0.5 - i*0.5));
    
}

// Function 348
float getDoubleDownsampledColorAt(int row, int column)
{	
	float value = 0.0;
	
	int startRow = row * 2;
	int startColumn = column * 2;
	
	for(int i = 0; i < 100; i++)
	{
		int currentRow = i / 10;
		int currentColumn = i - (currentRow * 10);
		if(currentRow >= startRow && currentRow < (startRow + 4))
		{
			if(currentColumn >= startColumn && currentColumn < (startColumn + 4))
			{
				float multiplier = 1.0;
				if(currentRow != startRow && currentRow != startRow + 3)
				{
					multiplier *= 3.0;
				}
				if(currentColumn != startColumn && currentColumn != startColumn + 3)
				{
					multiplier *= 3.0;
				}
				value += multiplier * subPixelValues[i] / 64.0;
			}
		} 
	}
	
	return value;
}

// Function 349
vec4 textureNoTile_3weights( sampler2D samp, in vec2 uv )
{
    vec4 res = vec4(0.);
    int sampleCnt = 0; // debug vis
    
    // compute per-tile integral and fractional uvs.
    // flip uvs for 'odd' tiles to make sure tex samples are coherent
    vec2 fuv = mod( uv, 2. ), iuv = uv - fuv;
    vec3 BL_one = vec3(0.,0.,1.); // xy = bot left coords, z = 1
    if( fuv.x >= 1. ) fuv.x = 2.-fuv.x, BL_one.x = 2.;
    if( fuv.y >= 1. ) fuv.y = 2.-fuv.y, BL_one.y = 2.;
    
    
    // weight orthogonal to diagonal edge = 3rd texture sample
    vec2 iuv3;
    float w3 = (fuv.x+fuv.y) - 1.;
    if( w3 < 0. ) iuv3 = iuv + BL_one.xy, w3 = -w3; // bottom left corner, offset negative, weight needs to be negated
    else iuv3 = iuv + BL_one.zz; // use transform from top right corner
    
    #if 0
    
    //
    // Original calculation of w3
    //
    
    w3 = smoothstep(BLEND_WIDTH, 1.-BLEND_WIDTH, w3);
    #else
    
    //
    // Modified calculation of w3
    //
    
    w3 = smoothstep(BLEND_WIDTH, 1.-BLEND_WIDTH, pow(1. - min(length(1. - fuv), length(fuv)) , 1.5));
    
    #endif
    
    // if third sample doesnt dominate, take first two
    if( w3 < 0.999 )
    {
        // use weight along long diagonal edge
        float w12 = dot(fuv,vec2(.5,-.5)) + .5;
        w12 = smoothstep(1.125*BLEND_WIDTH, 1.-1.125*BLEND_WIDTH, w12);

        // take samples from texture for each side of diagonal edge
        if( w12 > 0.001 ) res +=     w12  * texture( samp, transformUVs( iuv + BL_one.zy, uv ) ), sampleCnt++;
        if( w12 < 0.999 ) res += (1.-w12) * texture( samp, transformUVs( iuv + BL_one.xz, uv ) ), sampleCnt++;
    }
    
	// first two samples aren't dominating, take third
    if( w3 > 0.001 ) res = mix( res, texture( samp, transformUVs( iuv3, uv ) ), w3 ), sampleCnt++;

    
    // debug vis: colour based on num samples taken for vis purposes
    if( iMouse.z > 0. )
    {
        if( sampleCnt == 1 ) res.rb *= .25;
        if( sampleCnt == 2 ) res.b *= .25;
        if( sampleCnt == 3 ) res.gb *= .25;
    }
    
    return res;
}

// Function 350
vec4 textureFade(sampler2D tex, vec2 uv, vec2 fadeWidth) {
    vec2 offsetuv = uv*vec2(1.0-fadeWidth.x, 1.0 - fadeWidth.y);
    
    vec2 scaling = 1.0 - fadeWidth;
    float hBlend = clamp((uv.y-scaling.y)/fadeWidth.y,0.0,1.0); 
    float vBlend = clamp((uv.x-scaling.x)/fadeWidth.x, 0.0, 1.0);
    
    float q1Blend = hBlend * (1.0-vBlend);
    vec2 q1Sample;
    q1Sample.x = fract(offsetuv.x + fadeWidth.x);
    q1Sample.y = fract(offsetuv.y + (fadeWidth.y * 2.0));
    vec4 tex1 = texture(tex, q1Sample); 
    vec4 q1Col = q1Blend * tex1;

    float q2Blend = hBlend * vBlend;
    vec2 q2Sample;
    q2Sample.x = fract(offsetuv.x + (fadeWidth.x * 2.0));
    q2Sample.y = fract(offsetuv.y + (fadeWidth.y * 2.0));
    vec4 tex2 = texture(tex, q2Sample);
    vec4 q2Col = q2Blend * tex2;
 
    float q3Blend = (1.0-hBlend) * (1.0 - vBlend);
    vec2 q3Sample;
    q3Sample.x = fract(offsetuv.x + fadeWidth.x);
    q3Sample.y = fract(offsetuv.y + fadeWidth.y);
    vec4 tex3 = texture(tex, q3Sample);
	vec4 q3Col = q3Blend * tex3;
    
    float q4Blend = (1.0-hBlend) * vBlend;
    vec2 q4Sample;
    q4Sample.x = fract(offsetuv.x + (fadeWidth.x * 2.0));
    q4Sample.y = fract(offsetuv.y + fadeWidth.y);
    vec4 tex4 = texture(tex, q4Sample);
	vec4 q4Col = q4Blend * tex4;
    
    return q1Col + q2Col + q3Col + q4Col;

}

// Function 351
void TextureEnvBlured3(in vec3 N, in vec3 Rv, out vec3 iblDiffuse, out vec3 iblSpecular) {
    vec3 irradiance = vec3(0.0);   
    
    vec2 ts = vec2(textureSize(reflectTex, 0));
    float maxMipMap = log2(max(ts.x, ts.y));
    float lodBias = maxMipMap-7.0;    
    
    // tangent space calculation from origin point
    vec3 up    = vec3(0.0, 1.0, 0.0);
    vec3 right = cross(up, N);
    up            = cross(N, right);
       
    float sampleDelta = PI / 75.0;
    float nrSamples = 0.0f;
    for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // spherical to cartesian (in tangent space)
            vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N; 

            irradiance += sampleReflectionMap(sampleVec, lodBias) * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    iblDiffuse = PI * irradiance * (1.0 / float(nrSamples));    
}

// Function 352
vec4 sample_biquadratic(sampler2D channel, vec2 res, vec2 uv) {
    vec2 q = fract(uv * res);
    vec2 c = (q*(q - 1.0) + 0.5) / res;
    vec2 w0 = uv - c;
    vec2 w1 = uv + c;
    vec4 s = texture(channel, vec2(w0.x, w0.y))
    	   + texture(channel, vec2(w0.x, w1.y))
    	   + texture(channel, vec2(w1.x, w1.y))
    	   + texture(channel, vec2(w1.x, w0.y));
	return s / 4.0;
}

// Function 353
void sampleSphericalTriangle(in vec3 A, in vec3 B, in vec3 C, in float Xi1, in float Xi2, out vec3 w, out float wPdf) {
	//calculate internal angles of spherical triangle: alpha, beta and gamma
	vec3 BA = orthogonalize(A, B-A);
	vec3 CA = orthogonalize(A, C-A);
	vec3 AB = orthogonalize(B, A-B);
	vec3 CB = orthogonalize(B, C-B);
	vec3 BC = orthogonalize(C, B-C);
	vec3 AC = orthogonalize(C, A-C);
	float alpha = acos(clamp(dot(BA, CA), -1.0, 1.0));
	float beta = acos(clamp(dot(AB, CB), -1.0, 1.0));
	float gamma = acos(clamp(dot(BC, AC), -1.0, 1.0));

	//calculate arc lengths for edges of spherical triangle
	float a = acos(clamp(dot(B, C), -1.0, 1.0));
	float b = acos(clamp(dot(C, A), -1.0, 1.0));
	float c = acos(clamp(dot(A, B), -1.0, 1.0));

	float area = alpha + beta + gamma - PI;

	//Use one random variable to select the new area.
	float area_S = Xi1*area;

	//Save the sine and cosine of the angle delta
	float p = sin(area_S - alpha);
	float q = cos(area_S - alpha);

	// Compute the pair(u; v) that determines sin(beta_s) and cos(beta_s)
	float u = q - cos(alpha);
	float v = p + sin(alpha)*cos(c);

	//Compute the s coordinate as normalized arc length from A to C_s.
	float s = (1.0 / b)*acos(clamp(((v*q - u*p)*cos(alpha) - v) / ((v*p + u*q)*sin(alpha)), -1.0, 1.0));

	//Compute the third vertex of the sub - triangle.
	vec3 C_s = slerp(A, C, s);

	//Compute the t coordinate using C_s and Xi2
	float t = acos(1.0 - Xi2*(1.0 - dot(C_s, B))) / acos(dot(C_s, B));

	//Construct the corresponding point on the sphere.
	vec3 P = slerp(B, C_s, t);

	w = P;
	wPdf = 1.0 / area;
}

// Function 354
vec3 Sample_SphLight_HemiSph(vec3 V, vec3 p, vec3 N, inout uint h, vec3 albedo, float roughness, vec3 F0)
{
    float alpha = GGXAlphaFromRoughness(roughness);
    
    vec3 L;
    {
        float h0 = Hash11(h);
        float h1 = Hash01(h);
        	  
        L = Sample_Sphere(h0, h1, N);
    }

    float t2; vec3 n2; vec3 a2; bool isLight2 = true;
    bool hit = Intersect_Scene(p, L, false, /*out*/ t2, n2, a2, isLight2);

    if(!isLight2) return vec3(0.0);
    
    float NoL = clamp01(dot(N, L));
    
    return Frostbite_R(V, N, L, albedo, roughness, F0) * Radiance * NoL * pi2;
}

// Function 355
vec3 betterTextureSample64(sampler2D tex, vec2 uv) {	
	float textureResolution = 64.0;
	uv = uv*textureResolution + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*(3.0-2.0*fuv); // fuv*fuv*fuv*(fuv*(fuv*6.0-15.0)+10.0);;
	uv = (uv - 0.5)/textureResolution;
	return textureLod(tex, uv, 0.0).rgb;
}

// Function 356
vec4 GetTextureSample(vec2 pos, float freq, vec2 nodePoint)
{
    vec3 hash = hash33(vec3(nodePoint.xy, 0));
    float ang = hash.x * 2.0 * pi;
    mat2 rotation = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
    
    vec2 uv = rotation * pos * freq + hash.yz;
    return texture(iChannel0, uv);
}

// Function 357
vec3 TextureEnvBlured2(in vec3 N) {
    vec3 iblDiffuse = vec3(0.0);
	
    mat3 shR, shG, shB;
    
    CubeMapToSH2(shR, shG, shB);
    
    #if 1
    	shR = shDiffuseConvolution(shR);
    	shG = shDiffuseConvolution(shG);
    	shB = shDiffuseConvolution(shB);
    #endif
    
    #if 0
    	shR = shDiffuseConvolutionPI(shR);
    	shG = shDiffuseConvolutionPI(shG);
    	shB = shDiffuseConvolutionPI(shB);
    #endif    
    
    iblDiffuse = SH2toColor(shR, shG, shB, N);
    return iblDiffuse;
}

// Function 358
vec3 samplef(vec2 tc)
{
	return pow((texture(iChannel0, tc).xyz),vec3(1.00));
}

// Function 359
vec4 cornerSample(vec2 coord)
{
    float t = iTime*0.002;
    return vec4(sample1(coord + vec2(t, t)),
                sample1(coord + vec2(t, t+1.)),
                sample1(coord + vec2(t+1., t+1.)),
                sample1(coord + vec2(t+1., t))); 
}

// Function 360
float sampleVolume(vec3 pos)
{
    float rr = dot(pos,pos);
    rr = sqrt(rr);
    float f = exp(-rr);
    float p = f * _Density;
    
    if (p <= 0.0)
        return p;
    
    p += SpiralNoiseC(512.0 + pos * 8.0) * 0.75;
    pos = rotateY(pos, pos.y * SpiralNoiseC(pos * 4.0)* 2.0);
    p += SpiralNoiseC(200.0 + pos * 3.0) * 1.5;
    p *= rr/_Radius;
        
    p = max(0.0,p);
                
    return p;
}

// Function 361
vec3 getBallTexture(vec2 uv, vec3 color, int num) {
    uv = vec2(1.0 - uv.y, uv.x);
    uv -= vec2(0.5, 0.5);
    uv *= vec2(4.0f, 2.0f);
    uv += vec2(0.5, 0.5);
    uv = min(uv, vec2(0.97));
    uv = max(uv, vec2(0.03));
    
    int px = (num > 7 ? num - 8 : num) * 64 + int(uv.x * 64.0);
    int py = 128 + (num > 7 ? 1 : 0) * 64 + int(uv.y * 64.0);
    uv = vec2(float(px), float(py)) / iResolution.xy;
    return texture(iChannel0,uv).xyz;
}

// Function 362
void Sample_GGX_R(vec2 s, vec3 V, vec3 N, float alpha, vec3 F0, out vec3 L, out vec3 f, out float pdf)
{
    float l = rsqrt((alpha*alpha)/s.y + 1.0 - (alpha*alpha));
    
    vec3 H = Sample_Sphere(s.x * 2.0 - 1.0, l, N);

    L = 2.0 * dot(V, H) * H - V;
    
    float HoV = clamp01(dot(H, V));
    float NoV = clamp01(dot(N, V));
    float NoL = clamp01(dot(N, L));
    float NoH = clamp01(dot(N, H));

    vec3 F = FresnelSchlick(HoV, F0);  
    float G = GGX_G(NoL, NoV, alpha);
    float D = GGX_D(NoH, alpha);
    
    f   = NoV == 0.0 ? vec3(0.0) : (F * G  * D) * 0.25 / NoV;
    pdf = HoV == 0.0 ?      0.0  : (   NoH * D) * 0.25 / HoV;
}

// Function 363
vec3 aaBoxFilteredTexture( in vec2 p, in vec2 ddx, in vec2 ddy )
{
    vec3 color;
    vec2 max_dd = max(ddx, ddy);
    vec2 dual_u = vec2(p.x, max_dd.x);
    vec2 dual_v = vec2(p.y, max_dd.y);
    float f, pattern;
    
    // background
    color = mix(bg, white, crossingBox(tranScale(dual_u, 0.5/6.75, 6.75), tranScale(dual_v, 0.5/6.75, 6.75)));
    color = mix(color, grey, crossingBox(tranScale(dual_u, 0.5/6.45, 6.45), tranScale(dual_v, 0.5/6.45, 6.45)));
    
    // centerlines, just add as patterns do not intersect due to filter
    f = 1.0-rectBox(tranScale(dual_u, 0.5/6.45, 6.45), tranScale(dual_v, 0.5/6.45, 6.45));
    pattern = f * (lineBox(tranScale(dual_u, 1.0/0.2, 0.3)) +
                   lineBox(tranScale(dual_v, 1.0/0.2, 0.3)) +
                   lineBox(tranScale(dual_u, 1.0/0.2, -0.1)) +
                   lineBox(tranScale(dual_v, 1.0/0.2, -0.1)) );
	// dashed, just add as patterns do not intersect due to filter
    f = 1.0-rectBox(tranScale(dual_u, 0.5/10.7, 10.7), tranScale(dual_v, 0.5/10.7, 10.7));
    pattern += f * (lineBox(tranScale(dual_u, 1.0/0.15, 3.45))*dashBox(tranScale(dual_v, 1.0/2.0, 1.0), 0.6) +
                    lineBox(tranScale(dual_v, 1.0/0.15, 3.45))*dashBox(tranScale(dual_u, 1.0/2.0, 1.0), 0.6) +
                    lineBox(tranScale(dual_u, 1.0/0.15, -3.3))*dashBox(tranScale(dual_v, 1.0/2.0, 1.0), 0.6) +
                    lineBox(tranScale(dual_v, 1.0/0.15, -3.3))*dashBox(tranScale(dual_u, 1.0/2.0, 1.0), 0.6));
    // stop lines, add again
    pattern += rectBox(tranScale(dual_u, 1.0/0.4, -9.55), tranScale(dual_v, 1.0/5.55, -0.6)) +
               rectBox(tranScale(dual_u, 1.0/0.4,  10.05), tranScale(dual_v, 1.0/5.55, -0.6)) +
               rectBox(tranScale(dual_u, 1.0/0.4, -9.55), tranScale(dual_v, 1.0/5.55, 6.15)) +
               rectBox(tranScale(dual_u, 1.0/0.4,  10.05), tranScale(dual_v, 1.0/5.55, 6.15)) +
               rectBox(tranScale(dual_v, 1.0/0.4, -9.55), tranScale(dual_u, 1.0/5.55, -0.6)) +
               rectBox(tranScale(dual_v, 1.0/0.4,  10.05), tranScale(dual_u, 1.0/5.55, -0.6)) +
               rectBox(tranScale(dual_v, 1.0/0.4, -9.55), tranScale(dual_u, 1.0/5.55, 6.15)) +
               rectBox(tranScale(dual_v, 1.0/0.4,  10.05), tranScale(dual_u, 1.0/5.55, 6.15));
    // pedestrian crossing, add again
    pattern += rectBox(tranScale(dual_v, 1.0/2.0,  9.05), tranScale(dual_u, 1.0/5.55, 6.15))*dashBox(tranScale(dual_u, 1.0/0.6, 5.925), 0.5) +
               rectBox(tranScale(dual_v, 1.0/2.0,  9.05), tranScale(dual_u, 1.0/5.55, -0.6))*dashBox(tranScale(dual_u, 1.0/0.6, -0.825), 0.5) +
               rectBox(tranScale(dual_v, 1.0/2.0,  -7.05), tranScale(dual_u, 1.0/5.55, 6.15))*dashBox(tranScale(dual_u, 1.0/0.6, 5.925), 0.5) +
               rectBox(tranScale(dual_v, 1.0/2.0,  -7.05), tranScale(dual_u, 1.0/5.55, -0.6))*dashBox(tranScale(dual_u, 1.0/0.6, -0.825), 0.5) +
               rectBox(tranScale(dual_u, 1.0/2.0,  9.05), tranScale(dual_v, 1.0/5.55, 6.15))*dashBox(tranScale(dual_v, 1.0/0.6, 5.925), 0.5) +
               rectBox(tranScale(dual_u, 1.0/2.0,  9.05), tranScale(dual_v, 1.0/5.55, -0.6))*dashBox(tranScale(dual_v, 1.0/0.6, -0.825), 0.5) +
               rectBox(tranScale(dual_u, 1.0/2.0,  -7.05), tranScale(dual_v, 1.0/5.55, 6.15))*dashBox(tranScale(dual_v, 1.0/0.6, 5.925), 0.5) +
               rectBox(tranScale(dual_u, 1.0/2.0,  -7.05), tranScale(dual_v, 1.0/5.55, -0.6))*dashBox(tranScale(dual_v, 1.0/0.6, -0.825), 0.5);
    color = mix(color, white, pattern);

    return color;
}

// Function 364
vec3 SphQuadSample(SphQuad squad, float u, float v) {
    // 1. compute ’cu’
    float au = u * squad.S + squad.k;
    float fu = (cos(au) * squad.b0 - squad.b1) / sin(au);

    float cu = 1./sqrt(fu*fu + squad.b0sq) * sign(fu);
    cu = clamp(cu, -1., 1.); // avoid NaNs

    // 2. compute ’xu’
    float xu = -(cu * squad.z0) / sqrt(1. - cu*cu);
    xu = clamp(xu, squad.x0, squad.x1); // avoid Infs

    // 3. compute ’yv’
    float d = sqrt(xu*xu + squad.z0sq);
    float h0 = squad.y0 / sqrt(d*d + squad.y0sq);
    float h1 = squad.y1 / sqrt(d*d + squad.y1sq);
    float hv = h0 + v * (h1-h0);
    float hv2 = hv*hv;
    float yv = (hv2 < 1.-eps) ? (hv*d)/sqrt(1.-hv2) : squad.y1;

    // 4. transform (xu,yv,z0) to world coords
    return (squad.o + xu*squad.x + yv*squad.y + squad.z0*squad.z);
}

// Function 365
vec4 textureLinear (sampler2D smp, int smpi, vec2 uv)
{
	vec4 col = colorGamma(texture(smp, fixUV(uv, smpi)));
#ifdef PREMULTIPLY_ALPHA
	col.rgb *= col.a;
#endif
	return col;
}

// Function 366
vec3 sampleLightSource(in vec3 x, vec3 ns, float Xi1, float Xi2, out LightSamplingRecord sampleRec) {
    vec3 p_global = light.pos + vec3(1., 0., 0.) * light.size.x * (Xi1 - 0.5) +
        						vec3(0., 0., 1.) * light.size.y * (Xi2 - 0.5);
    vec3 n_global = vec3(0.0, -1.0, 0.0);
    sampleRec.w = p_global - x;
    sampleRec.d = length(sampleRec.w);
    sampleRec.w = normalize(sampleRec.w);
    float cosAtLight = dot(n_global, -sampleRec.w);
    vec3 L = cosAtLight>0.0?getRadiance(vec2(Xi1,Xi2)):vec3(0.0);
    sampleRec.pdf = PdfAtoW(1.0 / (light.size.x*light.size.y), sampleRec.d*sampleRec.d, cosAtLight);
    
	return L;
}

// Function 367
vec3 hsample(vec2 tc)
{
	return highlights(samplef(tc), 0.6);
}

// Function 368
vec3 SampleEnvironment( vec3 vDir )
{
    vec3 vEnvMap = texture(iChannel1, vDir).rgb;
    vEnvMap = vEnvMap * vEnvMap;
    
    float kEnvmapExposure = 0.999;
    vec3 vResult = -log2(1.0 - vEnvMap * kEnvmapExposure);    

    return vResult;
}

// Function 369
void disneyClearCoatSample(out vec3 wi, const in vec3 wo, const in vec2 u, const in SurfaceInteraction interaction, const in MaterialInfo material) {
	float gloss = mix(0.1, 0.001, material.clearcoatGloss);
    float alpha2 = gloss * gloss;
    float cosTheta = sqrt(max(EPSILON, (1. - pow(alpha2, 1. - u[0])) / (1. - alpha2)));
    float sinTheta = sqrt(max(EPSILON, 1. - cosTheta * cosTheta));
    float phi = TWO_PI * u[1];
    
    vec3 whLocal = sphericalDirection(sinTheta, cosTheta, sin(phi), cos(phi));
     
    vec3 tangent = vec3(0.), binormal = vec3(0.);
    createBasis(interaction.normal, tangent, binormal);
    
    vec3 wh = whLocal.x * tangent + whLocal.y * binormal + whLocal.z * interaction.normal;
    
    if(!sameHemiSphere(wo, wh, interaction.normal)) {
       wh *= -1.;
    }
            
    wi = reflect(-wo, wh);   
}

// Function 370
vec4 textureTriPlanar(vec3 P, vec3 N)
{
    float texScale = 2.0;
    // Absolute world normal
    vec3 sharpness = vec3(1.0);
    vec3 Nb = pow(abs(N), sharpness);     
    // Force weights to sum to 1.0
    float b = (Nb.x + Nb.y + Nb.z);
    Nb /= vec3(b);	
    
    vec4 c0 = texture(iChannel0, P.xy * texScale) * Nb.z;
    vec4 c1 = texture(iChannel0, P.yz * texScale) * Nb.x;
    vec4 c2 = texture(iChannel0, P.xz * texScale) * Nb.y;
    
    //vec4 c0 = vec4(1.0,0.0,0.0,1.0) * Nb.z;
    //vec4 c1 = vec4(0.0,1.0,0.0,1.0) * Nb.x;
    //vec4 c2 = vec4(0.0,0.0,1.0,1.0) * Nb.y;
    
    return c0 + c1 + c2;
}

// Function 371
vec3 SphereTexture(sampler2D tex,vec3 normal) {
     float u = atan(normal.z, normal.x) / PI * 2.0;
     float v = asin(normal.y) / PI * 2.0;
     return texture(tex,vec2(u,v)).rgb;
}

// Function 372
float getSphereMappedTexture(in vec3 pointOnSphere)
{
    /* Test to determine which face we are drawing on.
     * Opposing faces are taken care of by the absolute
     * value, leaving us only three tests to perform.
     */
    vec2 st = (
        insideBounds(sphereToCube(pointOnSphere)) +
        insideBounds(sphereToCube(pointOnSphere.zyx)) +
        insideBounds(sphereToCube(pointOnSphere.xzy)));
    
    st *= 12.0;
    float k = GetWaveDisplacement(vec3(st.x,0.0,st.y))*0.5;
    k = clamp(k,0.0,1.0);
	return 1.0-k;
    //return textureFunc(st);
}

// Function 373
vec3 textureBlured(samplerCube tex, vec3 tc) {
   	vec3 r = textureAVG(tex,vec3(1.0,0.0,0.0));
    vec3 t = textureAVG(tex,vec3(0.0,1.0,0.0));
    vec3 f = textureAVG(tex,vec3(0.0,0.0,1.0));
    vec3 l = textureAVG(tex,vec3(-1.0,0.0,0.0));
    vec3 b = textureAVG(tex,vec3(0.0,-1.0,0.0));
    vec3 a = textureAVG(tex,vec3(0.0,0.0,-1.0));
        
    float kr = dot(tc,vec3(1.0,0.0,0.0)) * 0.5 + 0.5; 
    float kt = dot(tc,vec3(0.0,1.0,0.0)) * 0.5 + 0.5;
    float kf = dot(tc,vec3(0.0,0.0,1.0)) * 0.5 + 0.5;
    float kl = 1.0 - kr;
    float kb = 1.0 - kt;
    float ka = 1.0 - kf;
    
    kr = somestep(kr);
    kt = somestep(kt);
    kf = somestep(kf);
    kl = somestep(kl);
    kb = somestep(kb);
    ka = somestep(ka);    
    
    float d;
    vec3 ret;
    ret  = f * kf; d  = kf;
    ret += a * ka; d += ka;
    ret += l * kl; d += kl;
    ret += r * kr; d += kr;
    ret += t * kt; d += kt;
    ret += b * kb; d += kb;
    
    return ret / d;
}

// Function 374
vec2 normToTextureUv(in vec2 normUv, float texIdx, vec2 resolution)
{
    vec2 texMinRC = vec2(0., floor(texIdx * TEXTURES_INV_ROW));
    texMinRC.x = texIdx - texMinRC.y*TEXTURES_ROW;
    vec2 texMin = TEXTURES_INV_ROW * texMinRC;
    vec2 texMax = texMin + vec2(TEXTURES_INV_ROW); 
    
    vec2 btexMin = ceil(texMin * resolution + 2.) / resolution;
    vec2 btexMax = floor(texMax * resolution - 2.) / resolution;
    
    vec2 resUv = normUv * (btexMax - btexMin);
    
    vec2 tuv = btexMin + resUv;

    return tuv;
}

// Function 375
float SampleTexture(in vec2 uv)
{
    return texture(iChannel0, uv).r;
}

// Function 376
vec3 lightSample( const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, float seed, const in MaterialInfo material) {
    vec3 L = (light.position - interaction.point);
    vec3 V = -normalize(interaction.incomingRayDir);
    vec3 r = reflect(V, interaction.normal);
    vec3 centerToRay = dot( L, r ) * r - L;
    vec3 closestPoint = L + centerToRay * clamp( light.radius / length( centerToRay ), 0.0, 1.0 );
    wi = normalize(closestPoint);


    return light.L/dot(L, L);
}

// Function 377
vec4 sampleDof(sampler2D channel, vec2 channelDim, vec2 dir, vec2 u) {
    float screenAperture = channelDim.y*APERTURE;
    float sampleToRad = screenAperture * DOF_CLAMPING / float(DOF_SAMPLES);
    vec4 o = vec4(0);
    float sum = 0.;
    for(int i = -DOF_SAMPLES; i <= DOF_SAMPLES; i++) {
        float sRad = float(i)*sampleToRad;
        vec4 p = texture(channel, (u+dir*sRad)/channelDim);
        float rad = min(abs(p.a-FOCAL_DISTANCE)/p.a, DOF_CLAMPING);
        float influence = clamp((rad*screenAperture - abs(sRad)) + .5, 0., 1.) / (rad*rad+.001);
        o += influence * p;
        sum += influence;
    }
    return o/sum;
}

// Function 378
vec4 gridTexture(in vec2 uv)
{
    if(uv.y < 0.0)
    {
    	return vec4(0.0,0.0,0.0,0.0);
    }
    float thickness = 0.1;
	float speed = 1.5;
    
    float xPhase = mod(6.0*uv.x, 1.0);
    float yPhase = mod(6.0*uv.y-speed*iTime, 1.0);
            
    float xIntensity = max(0.0, 1.0-abs(0.5-xPhase)/thickness);
    float yIntensity = max(0.0, 1.0-abs(0.5-yPhase)/thickness);
    
    vec4 color = vec4(0.3, 0.7, 1.0, 1.0);
    
    vec4 result = (yIntensity+xIntensity)*color;
	return result;
}

// Function 379
vec3 getTexture(vec3 p0, vec3 rd, inout vec3 n, inout vec2 spe, float t,
                sampler2D channel1, sampler2D channel2, sampler2D channel3){ 
    float h = Terrain(p0.xz*.3);
    float elev = p0.y - .002*h;
    
    spe = vec2(0.,1000.);
    
	vec3 p = p0;
   	malaxSpace(p);

    p.x = abs(p.x);
  
    // Texture scale factor.        
    const float tSize1 = 1.; //.5;//1./6.;
	
    // puit
    vec3 pp = p;
    pp.x = abs(pp.x) +.1;// gothique  
  //  pMirror(pp.x, -.1);  
    float rp = length(pp.xz-vec2(3.,3.1));
    
    // arbre
    p.z += .05;
   
    vec3 ph = p;
   
    //vec3 p2 = p, ph = p;

    // Chemin de ronde
    ph.z -= .5;
	pR45(ph.zx);
    ph.z -= 4.6;
    ph.x += 1.;
    pReflect(ph, normalize(vec3(-1.,0,.7)),1.);
    
    vec3 pm = ph;
    pMirrorOctant(pm.xz, vec2(1.5,1.6));

    float ra = length(ph.xz);
    
    int id = rp < .202 ? ID_PUIT :
        //rp<3.1 ? ID_HOUSE_WALL :
        elev<.002 || length(p.xz)>10. ? (abs(p.z+1.9) < 1.9 && abs(p.x) < 2.3 ? ID_GROUND_CASTLE : ID_GROUND) :// sol
        p.y>2.7 ? ID_SMALL_TOUR : // toit tour conique 
        abs(p.z+2.) < 1.55 && abs(p.x) < 2. ? ID_CASTLE :  // chateau
        (length(p.xz-vec2(0,2.)) > 5.83 || (rp>3. && p.z<0.6)) ? ID_STONE :  // rempart 
        //abs(p.x) > 1.8 ? p.y < 2.5 ? vec3(.4,.4,.1) : vec3(.5,.9,.7) : //arbres
        ra < .5 ? (ra < .051 && p.y<.7 ? ID_TREE_1 : ID_TREE_2) :
        p.y < .325 ? ID_HOUSE_WALL : // mur maisonettes   
        ID_HOUSE_ROOF;  // toit maisonettes

    
    vec3 c = vec3(1);
    
    switch(id) {
        case ID_TREE_1 : 
        	n = doBumpMap(channel1, p0.xyz*vec3(1.,.1,1.)*tSize1, n, .07/(1. + t/MAX_DIST));
        	c = vec3(.4,.3,.2); break;
        case ID_TREE_2 :
        	n = doBumpMap(channel1, p0*4.*tSize1, n, .07/(1. + t/MAX_DIST)); 
        	c = vec3(.2,.5,.4); break;
        case ID_PUIT : 
        	n = doBumpMap(channel2, p0*1.95*tSize1, n, .007/(1. + t/MAX_DIST)); 
        	n = doBumpMapBrick(p*30., n, .015); c = .5*vec3(1.,.9,.7); break;
        case ID_GROUND :
        
        
            n = doBumpMap(channel1, p0*tSize1, n, .007/(1. + t/MAX_DIST));//max(1.-length(fwidth(sn)), .001)*hash(sp)/(1.+t/FAR)
			c = NoiseT(1000.*p0.xz)*mix(vec3(.7,.7,.6), vec3(.3,.5,.4), smoothstep(.0,.05, abs(abs(p.x*1.2+.05)-.1)));
       	// test
        	break;
        
        case ID_GROUND_CASTLE :  
        	n = doBumpMapBrick(p0*5., n, .005); 
        	c = vec3(.8,.8,.7); break;
        case ID_SMALL_TOUR : 
        	c = vec3(1.,.7,1); break;
        case ID_CASTLE : 
        	n = doBumpMap(channel3, p0*4.*tSize1, n, .007/(1. + t/MAX_DIST));
        	
        //	c = vec3(.95,.9,.85), smoothstep(0.,.1, sin(10.*p.y))); 
        	c = mix(vec3(1.), vec3(.95,.9,.85), smoothstep(0.,.1, sin(15.*p.y))); 
        	break;
        case ID_STONE : 
        	spe = vec2(.5,99.); 
        	n = doBumpMapBrick(p*8., n, .03);
        	n = doBumpMap(channel1, p0*1.5*tSize1, n, .01/(1. + t/MAX_DIST));
        	c = .5*vec3(1.,.85,.7); break;
        case ID_HOUSE_WALL :
        	//if (length(pm.xz)-.2
        	//n = doBumpMapBrick(p*15., n, .03); 
        	//c = vec3(1.,.9,.7);
            if (abs(pm.x-.0335) <.06 && abs(pm.z+.8) <.2 && pm.y<.285) {
                // porte
                n = doBumpMapBrick(vec3(.3, pm.x+.13, pm.z)*32., n, .03); 
                n = doBumpMap(channel1, 3.*pm.yxz*tSize1, n, .02/(1. + t/MAX_DIST));
                c = .6*vec3(0.,.6,1); 
            } else {	
	        	n = doBumpMap(channel2, p0*1.95*tSize1, n, .007/(1. + t/MAX_DIST)); 
                c = vec3(1.,.95,.9);
            }
                c = c * mix(.4*vec3(.2,.6,.7), vec3(1), 
                      1.-.5*smoothstep(.3,.05, p0.y)*smoothstep(.3, .6,texture(channel2, p0.xy*4.*tSize1).x));
            break;
        case ID_HOUSE_ROOF :
        	spe = vec2(1.,9.); 
        	//n = doBumpMapBrick((p-vec3(0.,.01,0.))*30., n, .03); 
        	n = doBumpMap(channel3, p0*tSize1, n, .025/(1. + t/MAX_DIST));
        	c = vec3(.55,.32,.2) * mix(vec3(1), .7*vec3(.2,.6,.7), 
                  .5*smoothstep(.2,.9,texture(channel2, p0.xy*4.*tSize1).x));
                   //  tex3D(channel2, p0*4.*tSize1, n).x));
        	break;        	
    }
    
    	// prevent normals pointing away from camera (caused by precision errors)
	n = normalize(n - max(.0, dot (n,rd))*rd);
    
    return c;
}

// Function 380
vec3 rayToTexture( vec3 p ) {
    return (p - vec3(0.0,0.5,0.0)) * 0.2 + 0.5;
}

// Function 381
float sampleLightSourcePdf( in vec3 x,
                               in vec3 wi,
                             	float d,
                              	float cosAtLight
                             ) {
    float min_x = objects[0].params_[0];			//min x
    float min_y = objects[0].params_[1];			//min y
    float max_x = objects[0].params_[2];			//max x
    float max_y = objects[0].params_[3];			//max y
    float dim_x = max_x - min_x;
    float dim_y = max_y - min_y;
    float pdfA = 1.0 / (dim_x*dim_y);
    return PdfAtoW(pdfA, d*d, cosAtLight);
}

// Function 382
vec3 samplef(vec2 tc)
{
	return pow(texture(iChannel0, tc).xyz, vec3(2.2, 2.2, 2.2));
}

// Function 383
vec3 textureNormal(vec2 uv) {
    uv = fract(uv) * 3.0 - 1.5;    
        
    vec3 ret;
    ret.xy = sqrt(uv * uv) * sign(uv);
    ret.z = sqrt(abs(1.0 - dot(ret.xy,ret.xy)));
    ret = ret * 0.5 + 0.5;    
    return mix(vec3(0.5,0.5,1.0), ret, smoothstep(1.0,0.98,dot(uv,uv)));
}

// Function 384
float Sample( float u, int row, int range )
{
    float f = 0.;
    for ( int i=0; i < 128; i++ )
    {
        if ( i >= range ) break;
        
        float g = texelFetch(iChannel0,ivec2((int(u*iChannelResolution[0].x)+i-range/2)&int(iChannelResolution[0].x-1.),row),0).r;
        
	    // gamma correct (convert to linear, before we do any blending with other samples)
        // (source texture isn't strictly an image, so this is just a cosmetic tweak)
    	f += pow( g, 2.2 );
    }
    return f/float(range);
}

// Function 385
vec3 disneySubSurfaceSample(out vec3 wi, const in vec3 wo, out float pdf, const in vec2 u, const in vec3 normal, const in MaterialInfo material) {
    
    cosineSample

    vec3 H = normalize(wo+wi);
    float NdotH = dot(normal,H);
    
    pdf = pdfLambertianReflection(wi, wo, normal);
    return vec3(0.);//disneySubsurface(NdotL, NdotV, NdotH, material) * material.subsurface;
}

// Function 386
vec3 TriplannarStarsTexture(vec3 p, vec3 normal)
{
    // the scale of the texture
    float scale = 0.25;
    // the sharpness of the blending between different axises
    float blendSharpness = 2.;
    // finding the different axise's color
    vec3 colX = texture(iChannel3, p.zy * scale).rgb;
    vec3 colY = texture(iChannel3, p.xz * scale).rgb;
    vec3 colZ = texture(iChannel3, p.xy * scale).rgb;
    
    // finding the blending amount for each axis
    vec3 bw = pow(abs(normal), vec3(blendSharpness));
    // making it so the total (x + y + z) is 1
    bw /= dot(bw, vec3(1.));
    
    // finding the final color
    return colX * bw.x + colY * bw.y + colZ * bw.z;
}

// Function 387
float escherTextureX(vec2 p)
{
    vec2 pp = vec2(mod(p.x+0.5, 1.0)-0.5, mod(p.y,2.0));
    
    float d = 1000.0;
    
    for(int i=0; i<19; ++i)
    	if(abs(horizontalDistance(pp, vert[i], vert[i+1])) < abs(d))
        {
            d = horizontalDistance(pp, vert[i], vert[i+1]);
        }
    
    pp = vec2(mod(p.x+0.5, 1.0)-0.5, mod(p.y-1.0,2.0));
    
    for(int i=0; i<19; ++i)
    	if(abs(horizontalDistance(pp, vert[i], vert[i+1])) < abs(d))
        {
            d = horizontalDistance(pp, vert[i], vert[i+1]);
        }
    
    float val = smoothstep(0.0, 1.0, d/0.05);
    val = d;
    
    
    if(mod(p.x-0.5, 2.0)-1.0 > 0.)
        val = -val;
    
    return val;
}

// Function 388
vec3 sampleHemisphereCosWeighted( in vec3 n, in vec2 xi ) {
    return l2w( sampleHemisphereCosWeighted( xi ), n );
}

// Function 389
float digitTexture(in vec2 pos)
{
    return texture(iChannel1, pos).x;
}

// Function 390
vec3 sampleImage(vec2 coord){
    if( PASS2 ){
        return texture(iChannel0,viewport(coord)).rgb;
    } else {
    	return pow3(texture(iChannel0,viewport(coord)).rgb,GAMMA);
    }
}

// Function 391
void texture_uv( inout vec4 fragColor, in vec2 uv)
{
    fragColor = texture(iChannel0, uv);
}

// Function 392
vec3 sampleHemisphere( const vec3 n, in float Xi1, in float Xi2 ) {
    vec2 r = vec2(Xi1,Xi2)*TWO_PI;
	vec3 dr=vec3(sin(r.x)*vec2(sin(r.y),cos(r.y)),cos(r.x));
	return dot(dr,n) * dr;
}

// Function 393
vec3 sampleBSDF(	in vec3 x,
                  	in vec3 ng,
                  	in vec3 ns,
                	in vec3 wi,
                  	in Material mtl,
                  	in bool useMIS,
                  	in int strataCount,
                  	in int strataIndex,
                	out vec3 wo,
                	out float brdfPdfW,
                	out vec3 fr,
                	out bool hitRes,
                	out SurfaceHitInfo hit,
               		out float spec) {
    vec3 Lo = vec3(0.0);
    for(int i=0; i<DL_SAMPLES; i++){
        float Xi1 = rnd();
        float Xi2 = rnd();
        float strataSize = 1.0 / float(strataCount);
        Xi2 = strataSize * (float(strataIndex) + Xi2);
        fr = mtlSample(mtl, ng, ns, wi, Xi1, Xi2, wo, brdfPdfW, spec);
        
        //fr = eval(mtl, ng, ns, wi, wo);

        float dotNWo = dot(wo, ns);
        //Continue if sampled direction is under surface
        if ((dot(fr,fr)>0.0) && (brdfPdfW > EPSILON)) {
            Ray shadowRay = Ray(x, wo);

            //abstractLight* pLight = 0;
            float cosAtLight = 1.0;
            float distanceToLight = -1.0;
            vec3 Li = vec3(0.0);

            {
                float distToHit;

                if(raySceneIntersection( shadowRay, EPSILON, false, hit, distToHit )) {
                    if(hit.mtl_id_>=LIGHT_ID_BASE) {
                        distanceToLight = distToHit;
                        cosAtLight = dot(hit.normal_, -wo);
                        if(cosAtLight > 0.0) {
                            Li = getRadiance(hit.uv_);
                            //Li = lights[0].color_*lights[0].intensity_;
                        }
                    } else {
                        hitRes = true;
                    }
                } else {
                    hitRes = false;
                    //TODO check for infinite lights
                }
            }

            if (distanceToLight>0.0) {
                if (cosAtLight > 0.0) {
                    vec3 contribution = (Li * fr * dotNWo) / brdfPdfW;

                    if (useMIS/* && !(mtl->isSingular())*/) {
                        float lightPickPdf = 1.0;//lightPickingPdf(x, n);
                        float lightPdfW = sampleLightSourcePdf( x, wi, distanceToLight, cosAtLight );
                        //float lightPdfW = sphericalLightSamplingPdf( x, wi );//pLight->pdfIlluminate(x, wo, distanceToLight, cosAtLight) * lightPickPdf;

                        contribution *= misWeight(brdfPdfW, lightPdfW);
                    }

                    Lo += contribution;
                }
            }
        }
    }

    return Lo*(1.0/float(DL_SAMPLES));
}

// Function 394
vec2 Sample_Uniform2(inout float seed) {
    return fract(sin(vec2(seed+=0.1,seed+=0.1))*
                vec2(43758.5453123,22578.1459123));
}

// Function 395
void sampleSphereHemisphereCosinus(vec3 viewer, in Sphere sphere, inout SurfaceLightSample sls){
    
    vec3 main_direction = normalize(viewer - sphere.position);
    sls.normal = randomDirectionHemisphere(main_direction, 1.0f);
    
    sls.point = sphere.position + (sphere.radius ) * sls.normal;
    
    sls.pdf = dot(main_direction, sls.normal) / (PI * sphere.radius2);
}

// Function 396
float SampleDigit(const in float fDigit, const in vec2 vUV)
{
	const float x0 = 0.0 / 4.0;
	const float x1 = 1.0 / 4.0;
	const float x2 = 2.0 / 4.0;
	const float x3 = 3.0 / 4.0;
	const float x4 = 4.0 / 4.0;
	
	const float y0 = 0.0 / 5.0;
	const float y1 = 1.0 / 5.0;
	const float y2 = 2.0 / 5.0;
	const float y3 = 3.0 / 5.0;
	const float y4 = 4.0 / 5.0;
	const float y5 = 5.0 / 5.0;

	// In this version each digit is made of up to 3 rectangles which we XOR together to get the result
	
	vec4 vRect0 = vec4(0.0);
	vec4 vRect1 = vec4(0.0);
	vec4 vRect2 = vec4(0.0);
		
	if(fDigit < 0.5) // 0
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x1, y1, x2, y4);
	}
	else if(fDigit < 1.5) // 1
	{
		vRect0 = vec4(x1, y0, x2, y5); vRect1 = vec4(x0, y0, x0, y0);
	}
	else if(fDigit < 2.5) // 2
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x0, y3, x2, y4); vRect2 = vec4(x1, y1, x3, y2);
	}
	else if(fDigit < 3.5) // 3
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x0, y3, x2, y4); vRect2 = vec4(x0, y1, x2, y2);
	}
	else if(fDigit < 4.5) // 4
	{
		vRect0 = vec4(x0, y1, x2, y5); vRect1 = vec4(x1, y2, x2, y5); vRect2 = vec4(x2, y0, x3, y3);
	}
	else if(fDigit < 5.5) // 5
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x1, y3, x3, y4); vRect2 = vec4(x0, y1, x2, y2);
	}
	else if(fDigit < 6.5) // 6
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x1, y3, x3, y4); vRect2 = vec4(x1, y1, x2, y2);
	}
	else if(fDigit < 7.5) // 7
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x0, y0, x2, y4);
	}
	else if(fDigit < 8.5) // 8
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x1, y1, x2, y2); vRect2 = vec4(x1, y3, x2, y4);
	}
	else if(fDigit < 9.5) // 9
	{
		vRect0 = vec4(x0, y0, x3, y5); vRect1 = vec4(x1, y3, x2, y4); vRect2 = vec4(x0, y1, x2, y2);
	}
	else if(fDigit < 10.5) // '.'
	{
		vRect0 = vec4(x1, y0, x2, y1);
	}
	else if(fDigit < 11.5) // '-'
	{
		vRect0 = vec4(x0, y2, x3, y3);
	}	
	
	float fResult = InRect(vUV, vRect0) + InRect(vUV, vRect1) + InRect(vUV, vRect2);
	
	return mod(fResult, 2.0);
}

// Function 397
vec3 OutlineWhyCantIPassASampler(vec2 fragCoord)
{
	vec2 uv = fragCoord.xy / iResolution.xy;
	uv.y = 1.-uv.y;
  	vec4 lines= vec4(0.30, 0.59, 0.11, 1.0);

	lines.rgb = lines.rgb * LINES*1.5;
 
  	float s11 = dot(texture(iChannel1, uv + vec2(-1.0 / iResolution.x, -1.0 / iResolution.y)), lines);   // LEFT
  	float s12 = dot(texture(iChannel1, uv + vec2(0, -1.0 / iResolution.y)), lines);             // MIDDLE
  	float s13 = dot(texture(iChannel1, uv + vec2(1.0 / iResolution.x, -1.0 / iResolution.y)), lines);    // RIGHT
 

  	float s21 = dot(texture(iChannel1, uv + vec2(-1.0 / iResolution.x, 0.0)), lines);                // LEFT
  	// Omit center
  	float s23 = dot(texture(iChannel1, uv + vec2(-1.0 / iResolution.x, 0.0)), lines);                // RIGHT
 
  	float s31 = dot(texture(iChannel1, uv + vec2(-1.0 / iResolution.x, 1.0 / iResolution.y)), lines);    // LEFT
  	float s32 = dot(texture(iChannel1, uv + vec2(0, 1.0 / iResolution.y)), lines);              // MIDDLE
  	float s33 = dot(texture(iChannel1, uv + vec2(1.0 / iResolution.x, 1.0 / iResolution.y)), lines); // RIGHT
 
  	float t1 = s13 + s33 + (2.0 * s23) - s11 - (2.0 * s21) - s31;
  	float t2 = s31 + (2.0 * s32) + s33 - s11 - (2.0 * s12) - s13;
 
  	vec3 col;
 
	if (((t1 * t1) + (t2* t2)) > 0.04) 
	{
  		col = vec3(-1.,-1.,-1.);
  	}
	else
	{
    		col = vec3(0.,0.,0.);
  	}
 
  	return col;
}

// Function 398
vec4 sampleNormalMap(vec3 N) 
{
    float u = 1.-(atan(N.x, N.z)+PI)/(2.*PI);
	float v = (acos(N.y)/PI);	// 1.- becouse the coordinates origin is in the bottom-left, but I backed from top-left
    return texture(iChannel0, vec2(u, v));   
}

// Function 399
float get_texture_switch_alpha(in sampler2D s)
{
    return texelFetch(s, CTRL_TEXTURE, 0).x;
}

// Function 400
vec4 textured(vec2 p)
{
    return mix(texture(iChannel2, p), texture(iChannel3, p), 0.5 + 0.5 * cos(p.x * tau));
}

// Function 401
void disneyMicrofacetAnisoSample(out vec3 wi, const in vec3 wo, const in vec3 X, const in vec3 Y, const in vec2 u, const in SurfaceInteraction interaction, const in MaterialInfo material) {
    float cosTheta = 0., phi = 0.;
    
    float aspect = sqrt(1. - material.anisotropic*.9);
    float alphax = max(.001, pow2(material.roughness)/aspect);
    float alphay = max(.001, pow2(material.roughness)*aspect);
    
    phi = atan(alphay / alphax * tan(2. * PI * u[1] + .5 * PI));
    
    if (u[1] > .5f) phi += PI;
    float sinPhi = sin(phi), cosPhi = cos(phi);
    float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
    float alpha2 = 1. / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
    float tanTheta2 = alpha2 * u[0] / (1. - u[0]);
    cosTheta = 1. / sqrt(1. + tanTheta2);
    
    float sinTheta = sqrt(max(0., 1. - cosTheta * cosTheta));
    vec3 whLocal = sphericalDirection(sinTheta, cosTheta, sin(phi), cos(phi));
         
    vec3 wh = whLocal.x * X + whLocal.y * Y + whLocal.z * interaction.normal;
    
    if(!sameHemiSphere(wo, wh, interaction.normal)) {
       wh *= -1.;
    }
            
    wi = reflect(-wo, wh);
}

// Function 402
vec3 textureNormal(vec2 uv) {
    vec3 normal = texture( iChannel1, 100.0 * uv ).rgb;
    normal.xy = 2.0 * normal.xy - 1.0;
    normal.z = sqrt(iMouse.x / iResolution.x);
    return normalize( normal );
}

// Function 403
vec2 Sample_Disk(float s0, float s1)
{
    return vec2(cos(Pi * s0), sin(Pi * s0)) * sqrt(s1);
}

// Function 404
vec3 SampleImage2( vec2 vUV, vec2 vScreen, int image )
{
    vec3 a = SampleImage( DistortUV( vUV, 1.0 ), image );
    vec3 b = SampleImage( DistortUV( vUV, 0.0 ), image );
    vec3 c = SampleImage( DistortUV( vUV, -1.0 ), image );
    
    vec3 vResult = vec3(0);
    
    vec3 wa = vec3(1., .5, .1);
    vec3 wb = vec3(.5, 1., .5);
    vec3 wc = vec3(.1, .5, 1.);
    
    vResult += a * wa;
    vResult += b * wb;
    vResult += c * wc;
    
    vResult /= wa + wb + wc;
    
    return vResult;
}

// Function 405
vec3 sampleEnvMap(vec3 rd, float lod)
{
    vec2 uv = vec2(atan(rd.z,rd.x),acos(rd.y));
    uv = fract(uv/vec2(2.0*PI,PI));
    
    vec3 col = vec3(0.,0.05*cos(uv.x)+0.05, .1*sin(uv.y)+.1)*1.;
    
    float r = (1.-pow(lod,.5))*1000.+5.;
    col += vec3(1.)* clamp( pow(1.-roundBox(uv-vec2(.5), vec2(.05,.05),.01),r), 0., 1.);
    col += vec3(1.)* clamp( pow(1.-roundBox(uv-vec2(.67,.5), vec2(.05,.05),.01),r), 0., 1.);
    col += vec3(1.)* clamp( pow(1.-roundBox(uv-vec2(.67,.67), vec2(.05,.05),.01),r), 0., 1.);
    col += vec3(1.)* clamp( pow(1.-roundBox(uv-vec2(.5,.67), vec2(.05,.05),.01),r), 0., 1.);
    col += vec3(1.,.5,.1)*2. * clamp( pow(1.-roundBox(uv-vec2(.3,.7), vec2(.01,.01),.2),r), 0., 1.);
    
    return min(col*(1.-lod*.8),vec3(1.));
}

// Function 406
vec3 sampleLightType( const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, out float visibility, float seed) {
    if( !light.enabled )
        return vec3(0.);
    
    if( light.type == LIGHT_TYPE_SPHERE ) {
        vec3 L = lightSample(light, interaction, wi, lightPdf, seed);
        visibility = visibilityTest(interaction.point + wi * .01, wi);
        return L;
    }
    else if( light.type == LIGHT_TYPE_SUN ) {
        vec3 L = sampleSun(light, interaction, wi, lightPdf, seed);
        visibility = visibilityTestSun(interaction.point + wi * .01, wi);
        return L;
    }
    else {
        return vec3(0.);
    }
}

// Function 407
vec4 baseTexture(in vec2 uv, float depth){    
    float size = 1.;    
    float blur = min(.0005 * (depth * 10.5), .0030);    
    return vec4(.5) * smoothstep(0., blur, smoothmod(uv.x * size, .05, blur * 5.)) * smoothstep(0., blur * RATIO, smoothmod(uv.y * size, .05, blur * RATIO * 3.)) * smoothstep(1., 1. - blur, mod(uv.x * size + .05 * floor(mod(uv.y * size, .1) * 20.), .1) * 20.) + texture(iChannel0, uv * 5.) - vec4(.14, .15, .01, 0.);	
}

// Function 408
vec4 textureFadeHorizontal(sampler2D tex, vec2 uv, float fadeWidth) {
    vec2 offsetuv = uv*vec2(1.0-fadeWidth, 1.0 - fadeWidth);
    
    float scaling = 1.0 - fadeWidth;
    float vBlend = clamp((uv.x-scaling)/fadeWidth, 0.0, 1.0);
    
    float q1Blend = (1.0-vBlend);
    vec2 q1Sample;
    q1Sample.x = fract(offsetuv.x + fadeWidth);
    q1Sample.y = fract(offsetuv.y + fadeWidth);
    vec4 tex1 = texture(tex, q1Sample); 
    vec4 q1Col = q1Blend * tex1;

    float q2Blend = vBlend;
    vec2 q2Sample;
    q2Sample.x = fract(offsetuv.x + (fadeWidth * 2.0));
    q2Sample.y = fract(offsetuv.y + fadeWidth);
    vec4 tex2 = texture(tex, q2Sample);
    vec4 q2Col = q2Blend * tex2;
    
    return q1Col + q2Col;
   
}

// Function 409
float sampleFreq(float freq) {return texture(iChannel0, vec2(freq, 0.25)).x;}

// Function 410
vec4 SampleCharacter( uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vClampedCharUV = clamp(vCharUV, vec2(0.01), vec2(0.99));
    vec2 vUV = (vec2(iChPos) + vClampedCharUV) / 16.0f;

    vec4 vSample;
    
    float l = length( (vClampedCharUV - vCharUV) );

    // Skip texture sample when not in character boundary
    // Ok unless we have big font weight
    if ( l > 0.01f )
    {
        vSample.rgb = vec3(0);
		vSample.w = 2000000.0; 
    }
    else
    {
		vSample = textureLod( iChannelFont, vUV, 0.0 );    
        vSample.gb = vSample.gb * 2.0f - 1.0f;
        vSample.a -= 0.5f + 1.0/256.0;    
    }
        
    return vSample;
}

// Function 411
vec4 SampleMip4(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*4.,sp.z+56.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(4.,0.))*IRES),fract(sp.y));
}

// Function 412
vec4 texture3D(sampler2D tex, vec3 uvw, vec3 vres)
{
    uvw = clamp(floor(uvw * vres), vec3(0), vres-1.0);    
    //XYZ -> Pixel index
    float idx = (uvw.z * (vres.x*vres.y)) + (uvw.y * vres.x) + uvw.x;    
    //Pixel index -> Buffer uv coords
    vec2 uv = vec2(mod(idx, iResolution.x), floor(idx / iResolution.x));    
    return textureLod(tex, (uv + 0.5) / iResolution.xy, 0.0);
}

// Function 413
vec4 texture3Plane(sampler2D tex,vec3 norm, vec3 pos, float mip)
{
    vec4 texel = vec4(0);
    
    texel = mix(texel, texture(tex, pos.yz, mip), abs(norm.x));
    texel = mix(texel, texture(tex, pos.xz, mip), abs(norm.y));
    texel = mix(texel, texture(tex, pos.xy, mip), abs(norm.z));
    
    return texel;
}

// Function 414
vec4 sample_triquadratic_exact(sampler3D channel, vec3 res, vec3 uv) {
    vec3 q = fract(uv * res);
    ivec3 t = ivec3(uv * res);
    ivec3 e = ivec3(-1, 0, 1);
    
    vec3 q0 = (q+1.0)/2.0;
    vec3 q1 = q/2.0;	
    
    vec4 s000 = texelFetch(channel, t + e.xxx, 0);
    vec4 s001 = texelFetch(channel, t + e.xxy, 0);
    vec4 s002 = texelFetch(channel, t + e.xxz, 0);
    vec4 s012 = texelFetch(channel, t + e.xyz, 0);
    vec4 s011 = texelFetch(channel, t + e.xyy, 0);
    vec4 s010 = texelFetch(channel, t + e.xyx, 0);
    vec4 s020 = texelFetch(channel, t + e.xzx, 0);
    vec4 s021 = texelFetch(channel, t + e.xzy, 0);
    vec4 s022 = texelFetch(channel, t + e.xzz, 0);

    vec4 y00 = mix(mix(s000, s001, q0.z), mix(s001, s002, q1.z), q.z);
    vec4 y01 = mix(mix(s010, s011, q0.z), mix(s011, s012, q1.z), q.z);
    vec4 y02 = mix(mix(s020, s021, q0.z), mix(s021, s022, q1.z), q.z);
	vec4 x0 = mix(mix(y00, y01, q0.y), mix(y01, y02, q1.y), q.y);
    
    vec4 s122 = texelFetch(channel, t + e.yzz, 0);
    vec4 s121 = texelFetch(channel, t + e.yzy, 0);
    vec4 s120 = texelFetch(channel, t + e.yzx, 0);
    vec4 s110 = texelFetch(channel, t + e.yyx, 0);
    vec4 s111 = texelFetch(channel, t + e.yyy, 0);
    vec4 s112 = texelFetch(channel, t + e.yyz, 0);
    vec4 s102 = texelFetch(channel, t + e.yxz, 0);
    vec4 s101 = texelFetch(channel, t + e.yxy, 0);
    vec4 s100 = texelFetch(channel, t + e.yxx, 0);

    vec4 y10 = mix(mix(s100, s101, q0.z), mix(s101, s102, q1.z), q.z);
    vec4 y11 = mix(mix(s110, s111, q0.z), mix(s111, s112, q1.z), q.z);
    vec4 y12 = mix(mix(s120, s121, q0.z), mix(s121, s122, q1.z), q.z);
    vec4 x1 = mix(mix(y10, y11, q0.y), mix(y11, y12, q1.y), q.y);
    
    vec4 s200 = texelFetch(channel, t + e.zxx, 0);
    vec4 s201 = texelFetch(channel, t + e.zxy, 0);
    vec4 s202 = texelFetch(channel, t + e.zxz, 0);
    vec4 s212 = texelFetch(channel, t + e.zyz, 0);
    vec4 s211 = texelFetch(channel, t + e.zyy, 0);
    vec4 s210 = texelFetch(channel, t + e.zyx, 0);
    vec4 s220 = texelFetch(channel, t + e.zzx, 0);
    vec4 s221 = texelFetch(channel, t + e.zzy, 0);
    vec4 s222 = texelFetch(channel, t + e.zzz, 0);

    vec4 y20 = mix(mix(s200, s201, q0.z), mix(s201, s202, q1.z), q.z);
    vec4 y21 = mix(mix(s210, s211, q0.z), mix(s211, s212, q1.z), q.z);
    vec4 y22 = mix(mix(s220, s221, q0.z), mix(s221, s222, q1.z), q.z);
    vec4 x2 = mix(mix(y20, y21, q0.y), mix(y21, y22, q1.y), q.y);
    
    return mix(mix(x0, x1, q0.x), mix(x1, x2, q1.x), q.x);
}

// Function 415
vec3 getBallTexture(vec2 uv, vec3 color, int num) {
    vec3 white = vec3(1.0, 1.0, 0.8);
    if(num == 0) {
        return white;
    } else {
        bool solid = (num < 9);
        float edgeBlend = 0.01;
        vec2 dirToCenter = vec2(0.5, 0.5) - vec2(uv.x + (0.5-uv.x)*0.5, uv.y);
        float d = sqrt(dot(dirToCenter, dirToCenter));
        float r = 0.07;
        vec3 res;

        float dirt = texture(iChannel2,uv*1.5).x;

        vec3 non_solid;
        float dd = abs(uv.y - 0.5) * 2.0f;
        if(dd < 0.18 - edgeBlend) {
            non_solid = color;
        } else if(dd > 0.18 + edgeBlend) {
            non_solid = white;
        } else {
            float blende = ((dd - (0.18 - edgeBlend))/(2.0*edgeBlend))*dirt;
            non_solid = mix(color, white, blende);
        }
        vec3 outside_circle = mix(non_solid, color, solid? 1.0 : 0.0);
        vec3 inside_circle = vec3(0.);
        vec2 scale = vec2(5.0, 8.0);
        if(num > 9) {
            vec4 numc1 = char(scale*(uv - 0.5) + vec2(0.3, 0.5), 48.0 + float(num-10));
            vec4 numc2 = char(scale*(uv - 0.5) + vec2(0.7, 0.5), 48.0 + float(num/10));
            numc1.xyz = vec3(1.0) - numc1.xxx*2.0;
            numc2.xyz = vec3(1.0) - numc2.xxx*2.0;
            inside_circle = mix(white, numc1.xyz, numc1.w)*mix(white, numc2.xyz, numc2.w);
        } else {
            vec4 numc = char(scale*(uv - 0.5) + vec2(0.5), 48.0 + float(num));
            numc.xyz = vec3(1.0) - numc.xxx*2.0;
            inside_circle = mix(white, numc.xyz, numc.w);
        }

        bool on_69_mark = (d > 0.047) && (d < 0.057) && (uv.y < 0.5) && (abs(uv.x - 0.5) < 0.03);
        inside_circle *= vec3((num == 6 || num == 9)? (on_69_mark? dirt*dirt : 1.0) : 1.0);
        res = mix(outside_circle, inside_circle, (d < r - edgeBlend)?1.0 : 0.0);

        float blendc = ((d - (r - edgeBlend))/(2.0*edgeBlend))*dirt;
        vec3 on_the_circle = mix(white, color, blendc);
        res = mix(res, on_the_circle, (abs(d - r) < edgeBlend)?1.0 : 0.0);
        return res;
    }
}

// Function 416
vec4 textureImproved( sampler2D tex, in vec2 res, in vec2 uv, in vec2 g1, in vec2 g2 )
{
	uv = uv*res + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*(3.0-2.0*fuv);
	uv = (uv - 0.5)/res;
	return textureGrad( tex, uv, g1, g2 );
}

// Function 417
vec3 sampleLight( 	in vec3 x, in RaySurfaceHit hit, in Material mtl, in bool useMIS ) {
    vec3 Lo = vec3( 0.0 );	//outgoing radiance
    float lightSamplingPdf = 1.0/float(LIGHT_SAMPLES);
   
    for( int i=0; i<LIGHT_SAMPLES; i++ ) {
        //select light uniformly
        float Xi = rnd();
        float strataSize = 1.0 / float(LIGHT_SAMPLES);
        Xi = strataSize * (float(i) + Xi);
        float lightPickPdf;
        int lightId = chooseOneLight(x, Xi, lightPickPdf);

        //Read light info
        vec3 Li;				//incomming radiance
        Sphere lightSphere;
        getLightInfo( lightId, lightSphere, Li );
        
        float Xi1 = rnd();
        float Xi2 = rnd();
        LightSamplingRecord sampleRec;
        sampleSphericalLight( x, lightSphere, Xi1, Xi2, sampleRec );
        
        float lightPdfW = lightPickPdf*sampleRec.pdf;
        vec3 Wi = sampleRec.w;
        
        float dotNWi = dot(Wi,hit.N);

        if ( (dotNWi > 0.0) && (lightPdfW > EPSILON) ) {
            Ray shadowRay = Ray( x, Wi );
            RaySurfaceHit newHit;
            bool visible = true;
#ifdef SHADOWS
            visible = ( raySceneIntersection( shadowRay, EPSILON, newHit ) && EQUAL_FLT(newHit.dist,sampleRec.d,EPSILON) );
#endif
            if(visible) {
                float brdf;
    			float brdfPdfW;			//pdf of choosing Wi with 'bsdf sampling' technique
                
                if( mtl.bsdf_ == BSDF_R_GLOSSY ) {
                    brdf = evaluateBlinn( hit.N, hit.E, Wi, mtl.roughness_ );
                    brdfPdfW = pdfBlinn(hit.N, hit.E, Wi, mtl.roughness_ );	//sampling Pdf matches brdf
                } else {
                    brdf = evaluateLambertian( hit.N, Wi );
                    brdfPdfW = pdfLambertian( hit.N, Wi );	//sampling Pdf matches brdf
                }

                float weight = 1.0;
                if( useMIS ) {
                    weight = misWeight( lightPdfW, brdfPdfW );
                }
                
                Lo += ( Li * brdf * weight * dotNWi ) / lightPdfW;
            }
        }
    }
    
    return Lo*lightSamplingPdf;
}

// Function 418
float SampleDistanceTexture(vec2 texuv,float c)
{
    return Sampx(64u+uint(abs(c)),texuv);
}

// Function 419
float sampleTile(vec2 coord)
{
    vec2 ts = getTileSize();
    
    float dx = ts.x / 4.;
    float dy = ts.y / 4.;
    
    float startx = (coord.x * ts.x) + 0.5 * dx;
    float starty = (coord.y * ts.y) + 0.5 * dy;
    
    float sum = 0.;
    
    for(int i = 0; i < 4; i++)
    {
        float x = startx + float(i) * dx;
        
        for(int j = 0; j < 4; j++)
        {
            float y = starty + float(j) * dy;
            
            vec2 coord = vec2(x, y) / iResolution.xy;
            
            vec4 col = texture(iChannel0, coord);
            
            float l = luminosity(col);
            
            sum += l;
        }
    }
    
    return sum / 16.;
}

// Function 420
vec4 texture3DLinear(sampler2D tex, vec3 uvw, vec3 vres)
{
    vec3 blend = fract(uvw*vres);
    vec4 off = vec4(1.0/vres, 0.0);
    
    //2x2x2 sample blending
    vec4 b000 = texture3D(tex, uvw + off.www, vres);
    vec4 b100 = texture3D(tex, uvw + off.xww, vres);
    
    vec4 b010 = texture3D(tex, uvw + off.wyw, vres);
    vec4 b110 = texture3D(tex, uvw + off.xyw, vres);
    
    vec4 b001 = texture3D(tex, uvw + off.wwz, vres);
    vec4 b101 = texture3D(tex, uvw + off.xwz, vres);
    
    vec4 b011 = texture3D(tex, uvw + off.wyz, vres);
    vec4 b111 = texture3D(tex, uvw + off.xyz, vres);
    
    return mix(mix(mix(b000,b100,blend.x), mix(b010,b110,blend.x), blend.y), 
               mix(mix(b001,b101,blend.x), mix(b011,b111,blend.x), blend.y),
               blend.z);
}

// Function 421
vec4 sampleBlurr(in sampler2D sampler, vec2 uv, vec2 invRes, float a)
{
   	vec4 c = texture(sampler, uv);
    for(int i = -1;i<2; ++i)
    {
    	for(int j = -1;j<2; ++j)
        {
        	if(i==0 && j == 0)
            	continue;
        	vec4 v = (1.-abs(float(i)*a))*(1.-abs(float(j)*a))*texture(sampler, uv+vec2(float(i), float(j))*invRes);
        	c += v;
        }
    }
    //c *= c;
    c *= 0.11;
    return c;
}

// Function 422
vec3 LinearSample(vec2 uv, vec3 CN, out vec3 Moments) {
    float c0,c1,c2,c3; vec3 m0,m1,m2,m3;
    vec2 fuv=floor(uv*HRES-0.499)+0.5;
    vec3 C0=LinearSample0(fuv,c0,m0,CN);
    vec3 C1=LinearSample0(fuv+vec2(1.,0.),c1,m1,CN);
    vec3 C2=LinearSample0(fuv+vec2(0.,1.),c2,m2,CN);
    vec3 C3=LinearSample0(fuv+vec2(1.),c3,m3,CN);
    vec2 fruv=fract(uv*HRES-0.499);
    float mc=mix(mix(c0,c1,fruv.x),mix(c2,c3,fruv.x),fruv.y)+0.01;
    Moments=mix(mix(m0,m1,fruv.x),mix(m2,m3,fruv.x),fruv.y)/mc;
    return mix(mix(C0,C1,fruv.x),mix(C2,C3,fruv.x),fruv.y)/mc;
}

// Function 423
vec3 texturef( in vec2 p )
{
	vec2 q = p;
	p = p*vec2(6.0,128.0);
	float f = 0.0;
    f += 0.500*noise( p ); p = p*2.02;
    f += 0.250*noise( p ); p = p*2.03;
    f += 0.125*noise( p ); p = p*2.01;
	f /= 0.875;
	
	vec3 col = 0.6 + 0.4*sin( f*2.5 + 1.0+vec3(0.0,0.5,1.0) );
	col *= 0.7 + 0.3*noise( 8.0*q.yx );
	col *= 0.8 + 0.2*clamp(2.0*noise(256.0*q.yx ),0.0,1.0);
    col *= vec3(1.0,0.65,0.5) * 0.85;
    return col;

}

// Function 424
void sampleEquiAngular(
	float u,
	float maxDistance,
	vec3 rayOrigin,
	vec3 rayDir,
	vec3 lightPos,
	out float dist,
	out float pdf)
{
	// get coord of closest point to light along (infinite) ray
	float delta = dot(lightPos - rayOrigin, rayDir);
	
	// get distance this point is from light
	float D = length(rayOrigin + delta*rayDir - lightPos);

	// get angle of endpoints
	float thetaA = atan(0.0 - delta, D);
	float thetaB = atan(maxDistance - delta, D);
	
	// take sample
	float t = D*tan(mix(thetaA, thetaB, u));
	dist = delta + t;
	pdf = D/((thetaB - thetaA)*(D*D + t*t));
}

// Function 425
vec3 ImportanceSampleCos(vec3 d) 
{
    vec2 rand = hash2();
    float phi = 6.28318530718*rand.x;
    float xiy = rand.y;
    float r = sqrt(xiy);
    float x = r*cos(phi);
    float y = r*sin(phi);
    float z = sqrt(max(0.0,1.0-x*x-y*y));
	vec3 w = d;
	vec3 u = cross(w.yzx, w);
	vec3 v = cross(w, u);
    return w*z+u*x+v*y;
}

// Function 426
vec3 sample_phone_specular(vec3 n, float roughness, float Xi1, float Xi2)
{
    float theta = acos(pow(Xi1, 1./(roughness + 1.)));
    float phi = 2. * PI * Xi2;
    return local_to_world(spherical_to_cartesian(1., phi, theta), n);
}

// Function 427
float triTexture(in vec2 pos)
{
    vec2 pos2 = pos;
    pos2.x = pos.x  - pos.y * 0.5;
    pos2.y = pos.y / sqrt(3.0) / 0.5;
    
    float ind = 1.0;
    
    if(mod(pos2.x, 1.0) > 0.5)
        ind *= -1.0;
    
    if(mod(pos2.y, 1.0) > 0.5)
        ind *= -1.0;
    
    if(mod(pos2.y, 1.0) > 1.0 - mod(pos2.x, 1.0))
        ind *= -1.0;
    
    ind = max(ind, 0.0);
    
    return ind;
}

// Function 428
vec3 lightSample( const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, float seed ) {
    vec2 u = vec2(random(), random());
    
    vec3 tangent = vec3(0.), binormal = vec3(0.);
    vec3 lightDir = normalize(light.position - interaction.point);
    createBasis(lightDir, tangent, binormal);
    
    float sinThetaMax2 = light.radius * light.radius / distanceSq(light.position, interaction.point);
    float cosThetaMax = sqrt(max(EPSILON, 1. - sinThetaMax2));
    wi = uniformSampleCone(u, cosThetaMax, tangent, binormal, lightDir);
    
    if (dot(wi, interaction.normal) > 0.) {
        lightPdf = 1. / (TWO_PI * (1. - cosThetaMax));
    }
    
	return light.L;
}

// Function 429
float sampleConcreteLine(vec3 position) {
    float line = abs(fract(position.y)-0.5);
    line = 1.0-clamp(line*10.0,0.0,1.0);
    line*=line*line*3.0;
    return 1.0-line;
}

// Function 430
vec3 sampleIndirectLight(vec3 pos, vec3 normal){
    vec3 dir = getCosineWeightedSample(normal);
    vec3 abso = vec3(1.), light = vec3(0.), dc, ec;
    for(int i = 0; i < Bounces; i++){
        if(!trace(pos, dir, normal)) return light + abso*background(dir);
        sdf(pos, dc, ec);
        light += abso * (ec + dc*directLight(pos, normal));
        abso *= dc;
        dir = getCosineWeightedSample(normal);
    }
    return light;
}

// Function 431
float Sample_Uniform(inout float seed) {
    return fract(sin(seed += 0.1)*43758.5453123);
}

// Function 432
vec4 superSample(in vec2 fragCoord) {
  const int sampleCount = antialiasing ? 4 : 1;
  const vec2[] samplePositions = vec2[](                             
    vec2(-0.125, -0.375), vec2(0.375, -0.125),      
    vec2(-0.375,  0.125), vec2(0.125,  0.375)       
  );                                                         
  vec4 result = vec4(0.0);                                    
  float samplesSqrt = sqrt(float(sampleCount));                        
  for (int i = 0; i < sampleCount; i++) {                              
    result += takeSample(fragCoord + samplePositions[i],               
                         1.0 / samplesSqrt);                           
  }                                                                    
                                                                         
  return result / float(sampleCount);                                  
}

// Function 433
vec3 sample_hemisphere_cos_weighted(vec3 n, float Xi1, float Xi2) 
{
    float theta = acos(sqrt(1.0-Xi1));
    float phi = 2. * PI * Xi2;

    return local_to_world(spherical_to_cartesian(1.0, phi, theta), n);
}

// Function 434
vec4 textureBox(vec2 uv) {
    const vec2 RES = vec2(8.0, 8.0);
    vec2 iuv = (floor(uv * RES) + 0.5) / RES;  
    float n = noise1(uv * RES);
    n = max(abs(iuv.x - 0.5), abs(iuv.y - 0.5)) * 2.0;
    n = n * n;
    n = 0.5 + n * 0.4 + noise1(uv * RES) * 0.1;
    return vec4(n, n*0.8, n*0.5, 1.0);
}

// Function 435
vec4 sample3D(sampler2D tex, vec3 uvw, vec3 vres)
{
    uvw = mod(floor(uvw * vres), vres);
    
    //XYZ -> Pixel index
    float idx = (uvw.z * (vres.x*vres.y)) + (uvw.y * vres.x) + uvw.x;
    
    //Pixel index -> Buffer uv coords
    vec2 uv = vec2(mod(idx, iResolution.x), floor(idx / iResolution.x));
    
    return textureLod(tex, (uv + 0.5) / iResolution.xy, 0.0);
}

// Function 436
vec4 texture_Bicubic( sampler2D tex, vec2 t )
{
    vec2 res = iChannelResolution[0].xy;
    vec2 p = res*t - 0.5;
    vec2 f = fract(p);
    vec2 i = floor(p);

    return spline( f.y, spline( f.x, SAM(-1,-1), SAM( 0,-1), SAM( 1,-1), SAM( 2,-1)),
                        spline( f.x, SAM(-1, 0), SAM( 0, 0), SAM( 1, 0), SAM( 2, 0)),
                        spline( f.x, SAM(-1, 1), SAM( 0, 1), SAM( 1, 1), SAM( 2, 1)),
                        spline( f.x, SAM(-1, 2), SAM( 0, 2), SAM( 1, 2), SAM( 2, 2)));
}

// Function 437
vec4 catSample(float time, vec2 uv)
{
    float frame = floor(mod(time * 14.0, 6.0));
    vec2 use_uv = uv * vec2(1.0/6.4, 1) + vec2(frame / 6.4, 0);
    return texture(iChannel0, use_uv);
}

// Function 438
vec3 sTexture(sampler2D smp, vec2 uv) {
 
    vec2 textureResolution = iChannelResolution[1].yy;
	uv = uv*textureResolution + 0.5;
	vec2 iuv = floor( uv );
	uv -= iuv;
	uv = iuv + smoothstep(0., 1., uv); 
    //uv = iuv +  uv*uv*uv*(uv*(uv*6. - 15.) + 10.);
	uv = (uv - .5)/textureResolution;
    return texture(smp, uv).xyz;
    
}

// Function 439
vec3 BRDFLightSample(in Intersection intersecNow,out Intersection intersecNext,out vec3 wi,out float pdf){
	vec3 Li = vec3(0.);
    float x1 = GetRandom(),x2 = GetRandom();
    wi = sample_uniform_hemisphere(intersecNow.normal,x1,x2,pdf);
    Ray shadowRay = Ray(intersecNow.surface,wi);
    SceneIntersect(shadowRay, intersecNext);
    return Li;
}

// Function 440
vec3 sampleHemisphereCosWeighted( in float Xi1, in float Xi2 ) {
    float theta = acos(sqrt(1.0-Xi1));
    float phi = TWO_PI * Xi2;

    return sphericalToCartesian( 1.0, phi, theta );
}

// Function 441
vec3 moonTexture(vec2 uv) {
    float d = length(fract(uv) - .5);
    //return exp(-40. * d * d) * vec3(1.);
    return texture(iChannel0, uv / 16.).rgb;
}

// Function 442
vec3 GroundTexture(vec2 uv)
{   
    uv.x += sin(uv.y*40.)*.01;
    float x = pcurve(fract(uv.x * 10.), .3, .1);
    float y = pcurve(fract(uv.x * 100.), .7, .1);
    x = mix(x,y,.1);
    x -= random(uv) * .1;
    return vec3(1.0, .9, 0.0) * x;
}

// Function 443
vec3  anytexture(vec2 uv){
return vec3(0.7,1,.7)*(mod(floor(uv.x) + floor(uv.y *2. ), 2.));
}

// Function 444
float gridTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )
{
    const float N = 10.0;
    vec2 w = max(abs(ddx), abs(ddy)) + 0.01;
    vec2 a = p + 0.5*w;
    vec2 b = p - 0.5*w;           
    vec2 i = (floor(a)+min(fract(a)*N,1.0)-
              floor(b)-min(fract(b)*N,1.0))/(N*w);
    return (1.0-i.x)*(1.0-i.y);
}

// Function 445
vec3 textureNoTile( in vec2 x, float v )
{
    float k = texture( iChannel1, 0.005*x ).x; // cheap (cache friendly) lookup
    
    vec2 duvdx = dFdx( x );
    vec2 duvdy = dFdx( x );
    
    float l = k*8.0;
    float f = fract(l);
    
#if 1
    float ia = floor(l); // my method
    float ib = ia + 1.0;
#else
    float ia = floor(l+0.5); // suslik's method (see comments)
    float ib = floor(l);
    f = min(f, 1.0-f)*2.0;
#endif    
    
    vec2 offa = sin(vec2(3.0,7.0)*ia); // can replace with any other hash
    vec2 offb = sin(vec2(3.0,7.0)*ib); // can replace with any other hash

    vec3 cola = textureGrad( iChannel0, x + v*offa, duvdx, duvdy ).xyz;
    vec3 colb = textureGrad( iChannel0, x + v*offb, duvdx, duvdy ).xyz;
    
    return mix( cola, colb, smoothstep(0.2,0.8,f-0.1*sum(cola-colb)) );
}

// Function 446
vec4 SampleTextureCatmullRom4Samples(sampler2D tex, vec2 uv, vec2 texSize)
{
    // Based on the standard Catmull-Rom spline: w1*C1+w2*C2+w3*C3+w4*C4, where
    // w1 = ((-0.5*f + 1.0)*f - 0.5)*f, w2 = (1.5*f - 2.5)*f*f + 1.0,
    // w3 = ((-1.5*f + 2.0)*f + 0.5)*f and w4 = (0.5*f - 0.5)*f*f with f as the
    // normalized interpolation position between C2 (at f=0) and C3 (at f=1).
 
    // half_f is a sort of sub-pixelquad fraction, -1 <= half_f < 1.
    vec2 half_f     = 2.0 * fract(0.5 * uv * texSize - 0.25) - 1.0;
 
    // f is the regular sub-pixel fraction, 0 <= f < 1. This is equivalent to
    // fract(uv * texSize - 0.5), but based on half_f to prevent rounding issues.
    vec2 f          = fract(half_f);
 
    vec2 s1         = ( 0.5 * f - 0.5) * f;            // = w1 / (1 - f)
    vec2 s12        = (-2.0 * f + 1.5) * f + 1.0;      // = (w2 - w1) / (1 - f)
    vec2 s34        = ( 2.0 * f - 2.5) * f - 0.5;      // = (w4 - w3) / f
 
    // positions is equivalent to: (floor(uv * texSize - 0.5).xyxy + 0.5 +
    // vec4(-1.0 + w2 / (w2 - w1), 1.0 + w4 / (w4 - w3))) / texSize.xyxy.
    vec4 positions  = vec4((-f * s12 + s1      ) / (texSize * s12) + uv,
                           (-f * s34 + s1 + s34) / (texSize * s34) + uv);
 
    // Determine if the output needs to be sign-flipped. Equivalent to .x*.y of
    // (1.0 - 2.0 * floor(t - 2.0 * floor(0.5 * t))), where t is uv * texSize - 0.5.
    float sign_flip = half_f.x * half_f.y > 0.0 ? 1.0 : -1.0;
 
    vec4 w          = vec4(-f * s12 + s12, s34 * f); // = (w2 - w1, w4 - w3)
    vec4 weights    = vec4(w.xz * (w.y * sign_flip), w.xz * (w.w * sign_flip));
 
    return SampleTextureBilinearlyAndUnpack(tex, positions.xy) * weights.x +
           SampleTextureBilinearlyAndUnpack(tex, positions.zy) * weights.y +
           SampleTextureBilinearlyAndUnpack(tex, positions.xw) * weights.z +
           SampleTextureBilinearlyAndUnpack(tex, positions.zw) * weights.w;
}

// Function 447
vec3 getCosineWeightedSample(vec3 dir) {
	vec3 o1 = normalize(ortho(dir));
	vec3 o2 = normalize(cross(dir, o1));
	vec2 r = vec2(randomFloat(), randomFloat());
	r.x = r.x * 2.0 * Pi;
	r.y = pow(r.y, .5);
	float oneminus = sqrt(1.0-r.y*r.y);
	return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
}

// Function 448
vec3 SampleEnvironment( const in vec3 vDir )
{
	vec3 vSample = textureLod(iChannel0, vDir, 0.0).rgb;
	return (HackHDR(vSample * vSample)) * fSceneIntensity;
}

// Function 449
vec4 sampleLight(int i)
{
    AreaLight light = LIGHTS[i];
    float pdf = 1.0 / (4.0 * light.size.x * light.size.y);
    mat4 S = mat4(light.size.x,            0, 0, 0,
                            0, light.size.y, 0, 0,
                            0,            0, 1, 0,
                            0,            0, 0, 1);
    mat4 M = light.toWorld * S;
    return vec4((M * vec4(vec2(rnd(), rnd()) * 2.0 - 1.0, 0, 1)).xyz, pdf);
}

// Function 450
vec4 getDownsampledColorAt(vec2 uv, mat4 viewMatrix, mat4 inverseViewMatrix)
{		
	//pixel width
	float p = (1.0 / (iResolution.y * 2.0)) - (1.0 / (iResolution.y * 4.0));
	float p2 = (1.0 / iResolution.y) - (1.0 / (iResolution.y * 4.0));
	
	vec4 color = vec4(0,0,0,0);
	color += getSampleAt(uv + vec2(-p2,-p2), viewMatrix, inverseViewMatrix);
	color += 3.0 * getSampleAt(uv + vec2(-p2,-p), viewMatrix, inverseViewMatrix);
	color += 3.0 * getSampleAt(uv + vec2(-p2,p), viewMatrix, inverseViewMatrix);
	color += getSampleAt(uv + vec2(-p2, p2), viewMatrix, inverseViewMatrix);
	color += 3.0 * getSampleAt(uv + vec2(-p,-p2), viewMatrix, inverseViewMatrix);
	color += 9.0 * getSampleAt(uv + vec2(-p, -p), viewMatrix, inverseViewMatrix);
	color += 9.0 * getSampleAt(uv + vec2(-p, p), viewMatrix, inverseViewMatrix);
	color += 3.0 * getSampleAt(uv + vec2(-p, p2), viewMatrix, inverseViewMatrix);
	color += 3.0 * getSampleAt(uv + vec2(p,-p2), viewMatrix, inverseViewMatrix);
	color += 9.0 * getSampleAt(uv + vec2(p, -p), viewMatrix, inverseViewMatrix);
	color += 9.0 * getSampleAt(uv + vec2(p, p), viewMatrix, inverseViewMatrix);
	color += 3.0 * getSampleAt(uv + vec2(p, p2), viewMatrix, inverseViewMatrix);
	color += getSampleAt(uv + vec2(p2, -p2), viewMatrix, inverseViewMatrix);
	color += 3.0 * getSampleAt(uv + vec2(p2, -p), viewMatrix, inverseViewMatrix);
	color += 3.0 * getSampleAt(uv + vec2(p2, p), viewMatrix, inverseViewMatrix);
	color += getSampleAt(uv + vec2(p2, p2), viewMatrix, inverseViewMatrix);
	
	return color;
}

// Function 451
vec4 AntiAliasPointSampleTexture_Linear(vec2 uv, vec2 texsize) {	
	vec2 w=fwidth(uv);
	return texture(iChannel0, (floor(uv)+0.5+clamp((fract(uv)-0.5+w)/w,0.,1.)) / texsize, -99999.0);	
}

// Function 452
void sampleCamera(vec2 fragCoord, vec2 u, out vec3 rayOrigin, out vec3 rayDir)
{
	vec2 filmUv = (fragCoord.xy + u)/iResolution.xy;
	
	float tx = (2.0*filmUv.x - 1.0)*(iResolution.x/iResolution.y);
	float ty = (1.0 - 2.0*filmUv.y);
	float tz = 0.0;
	
	rayOrigin = vec3(0.0, 0.0, 5.0);
	rayDir = normalize(vec3(tx, ty, tz) - rayOrigin);
}

// Function 453
vec4 SampleTextureCatmullRom( vec2 uv, vec2 texSize )
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 samplePos = uv * texSize;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
    vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    
    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 texPos0 = texPos1 - vec2(1.0);
    vec2 texPos3 = texPos1 + vec2(2.0);
    vec2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    vec4 result = vec4(0.0);
    result += sampleLevel0( vec2(texPos0.x,  texPos0.y)) * w0.x * w0.y;
    result += sampleLevel0( vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos0.y)) * w3.x * w0.y;

    result += sampleLevel0( vec2(texPos0.x,  texPos12.y)) * w0.x * w12.y;
    result += sampleLevel0( vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos12.y)) * w3.x * w12.y;

    result += sampleLevel0( vec2(texPos0.x,  texPos3.y)) * w0.x * w3.y;
    result += sampleLevel0( vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos3.y)) * w3.x * w3.y;

    return result;
}

// Function 454
vec3 PBR_visitSamples(vec3 V, vec3 N, float roughness, bool metallic, vec3 ior_n, vec3 ior_k )
{
    const float MIPMAP_SWITCH  = 0.29; //sampling angle delta (rad) equivalent to the lowest LOD.
    const ivec2 SAMPLE_COUNT = ivec2(05,15); //(5 random, 15 fixed) samples
    const vec2 weight = vec2(1./float(SAMPLE_COUNT.x),1./float(SAMPLE_COUNT.y));
    float angularRange = 0.;    
    vec3 vCenter = reflect(-V,N);
    
    //Randomized Samples : more realistic, but jittery
    float randomness_range = 0.75; //Cover only the closest 75% of the distribution. Reduces range, but improves stability.
    float fIdx = 0.0;              //valid range = [0.5-1.0]. Note : it is physically correct at 1.0.
    vec3 totalRandom = vec3(0.0);
    for(int i=0; i < SAMPLE_COUNT[0]; ++i)
    {
        //Random noise from DaveHoskin's hash without sine : https://www.shadertoy.com/view/4djSRW
        vec3 p3 = fract(vec3(fIdx*10.0+vCenter.xyx*100.0) * vec3(.1031,.11369,.13787)); 
    	p3 += dot(p3.zxy, p3.yzx+19.19);
    	vec2 jitter = fract(vec2((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y));
        vec3 sampleDir    = PBR_importanceSampling(vCenter, roughness, jitter.x*randomness_range, jitter.y, angularRange);
        vec3 sampleColor  = PBR_HDRCubemap( sampleDir, angularRange/MIPMAP_SWITCH);
        vec3 contribution = PBR_Equation(V, sampleDir, N, roughness, ior_n, ior_k, metallic, true)*weight[0];
    	totalRandom += contribution*sampleColor;
		++fIdx;
    }
    
    //Fixed Samples : More stable, but can create sampling pattern artifacts (revealing the sampling pattern)
    fIdx = 0.0;
    vec3 totalFixed = vec3(0.0);
    for(int i=0; i < SAMPLE_COUNT[1]; ++i)
    {
        vec2 jitter = vec2( clamp(weight[1]*fIdx,0.0,0.50), fract(weight[1]*fIdx*1.25)+3.14*fIdx); //Fixed sampling pattern.
        vec3 sampleDir    = PBR_importanceSampling(vCenter, roughness, jitter.x, jitter.y, angularRange);
        vec3 sampleColor  = PBR_HDRCubemap( sampleDir, angularRange/MIPMAP_SWITCH);
        vec3 contribution = PBR_Equation(V, sampleDir, N, roughness, ior_n, ior_k, metallic, true)*weight[1];
        totalFixed += contribution*sampleColor;
		++fIdx;
    }
    
    return (totalRandom*weight[1]+totalFixed*weight[0])/(weight[0]+weight[1]);
}

// Function 455
vec3 Sample_ClampedCosineLobe(float s0, float s1, vec3 normal)
{	 
    vec2 d  = Sample_Disk(s0, s1);
    float y = sqrt(clamp01(1.0 - s1));

    vec3 ox, oz;
    OrthonormalBasisRH(normal, ox, oz);

    return (ox * d.x) + (normal * y) + (oz * d.y);
}

// Function 456
vec3 mtlSample(Material mtl, in vec3 Ng, in vec3 Ns, in vec3 E, in float Xi1, in float Xi2, out vec3 L, out float pdf) {
    if(!mtl.metal_ && mtl.specular_weight_ == 0.0) {//pure diffuse
        mat3 trans = mat3FromNormal(Ns);
        vec3 L_local = sampleHemisphereCosWeighted( Xi1, Xi2 );
        L = trans*L_local;
        pdf = pdfDiffuse(L_local);
        return mtl.diffuse_color_ * vec3(INV_PI);
    } else {
        mat3 trans = mat3FromNormal(Ns);
        mat3 inv_trans = mat3Inverse( trans );

        //convert directions to local space
        vec3 E_local = inv_trans * E;
        vec3 L_local;

        if (E_local.z == 0.0) { 
            return vec3(0.);
        } else {
            float alpha = mtl.specular_roughness_;
            float F = mtl.metal_? 1.0 : SchlickFresnel(1.6, E_local.z)* mtl.specular_weight_;
            //Sample specular or diffuse lobe based on fresnel
            if(rnd() < F) {
                // Sample microfacet orientation $\wh$ and reflected direction $\wi$
                vec3 wh = ggx_sample(E_local, alpha, alpha, Xi1, Xi2);
                L_local = reflect(-E_local, wh);
            } else {
                L_local = sampleHemisphereCosWeighted( Xi1, Xi2 );
            }

            if (!sameHemisphere(E_local, L_local)) {
                pdf = 0.0;
            } else {
                // Compute PDF of _wi_ for microfacet reflection
                pdf = 	pdfSpecular(E_local, L_local, alpha) * F +
                        pdfDiffuse(L_local) * (1.0 - F);
            }

            //convert directions to global space
            L = trans*L_local;

            if(!sameHemisphere(Ns, E, L) || !sameHemisphere(Ng, E, L)) {
                pdf = 0.0;
            }

            return mtlEval(mtl, Ng, Ns, E, L);
        }
    }
}

// Function 457
vec3 getHemisphereUniformSample(vec3 n) {
    float cosTheta = getRandom();
    float sinTheta = sqrt(1. - cosTheta * cosTheta);
    
    float phi = 2. * M_PI * getRandom();
    
    // Spherical to cartesian
    vec3 t = normalize(cross(n.yzx, n));
    vec3 b = cross(n, t);
    
	return (t * cos(phi) + b * sin(phi)) * sinTheta + n * cosTheta;
}

// Function 458
vec3 sampleQuad(vec2 uv, float W00, float W01, float W10, float W11) {
    float a = mix(W00, W01, .5);
    float b = mix(W10, W11, .5);
    float u = sampleLinear(uv.x, a, b);
    float c = mix(W00, W10, u);
    float d = mix(W01, W11, u);
    float v = sampleLinear(uv.y, c, d);
    float area = mix(a, b, .5);
    float pdf = mix(c, d, v) / area;
    return vec3(u, v, pdf);
}

// Function 459
void sampleEquiAngular(
	float u,
	float maxDistance,
	vec3 rayOrigin,
	vec3 rayDir,
	vec3 lightPos,
	out float dist,
	out float pdf)
{
	// get coord of closest point to light along (infinite) ray
	float delta = dot(lightPos - rayOrigin, rayDir);

	// get distance this point is from light
	float D = length(rayOrigin + delta*rayDir - lightPos);

	// get angle of endpoints
	float thetaA = atan(0.0 - delta, D);
	float thetaB = atan(maxDistance - delta, D);

	// take sample
	float t = D*tan(mix(thetaA, thetaB, u));
	dist = delta + t;
	pdf = D/((thetaB - thetaA)*(D*D + t*t));
}

// Function 460
void textureParallaxWindow(in vec3 block, inout ray ray, inout rayMarchHit hit, in vec2 uv, in vec3 cell, in vec3 lightColour, in float brightness, inout vec3 colour, in float time) {
    
    if(abs(uv.x)>PARALLAX_INTERROOM_WALL_THICKNESS||abs(uv.y)>PARALLAX_INTERROOM_WALL_THICKNESS) {
        colour=PARALLAX_INTERROOM_WALL_COLOUR;
        return;
    }
    
    vec3 absNormal = abs(hit.surfaceNormal);
    vec3 flatNormal;

    //flaten the normal so we still project axis aligned with no distortion of y axis curve.
    if(absNormal.y > absNormal.x && absNormal.y > absNormal.z) {
        flatNormal = vec3(0.0,1.0,0.0);
    } else {
        flatNormal = normalize(vec3(hit.surfaceNormal.x,0.0,hit.surfaceNormal.z));
    }    
    
    vec3 roomSize = vec3(1.0) ;
    roomSize.z += hash31(cell*16.8736)*2.0;
        
    vec3 tangent = normalize(cross(vec3(0.0,1.0,0.0),flatNormal));
    vec3 bitangent = normalize(cross(flatNormal,tangent));
    mat3 tMatrix = mat3(tangent,bitangent,flatNormal);
    
    vec3 rayDir = normalize(ray.direction*tMatrix)/roomSize;
    vec3 hitPos = vec3(uv.x,uv.y,0.0)/roomSize;
	
    //Room cube, We assume the room is 1 unit cube from -0.5 to +0.5, with a given wall thickness.
    vec3 roomMin = vec3(-PARALLAX_INTERROOM_WALL_THICKNESS,-PARALLAX_INTERROOM_WALL_THICKNESS,-1.0);
    vec3 roomMax = vec3(PARALLAX_INTERROOM_WALL_THICKNESS,PARALLAX_INTERROOM_WALL_THICKNESS,0.0);
    vec3 roomMid = vec3(0.0);
    
    //we only need to interesct 3 planes per ray, looking at the direction of the ray find which 3 its heading towards.
    vec3 planes = mix(roomMin, roomMax, step(0.0, rayDir));
    
    //now do the parallax calcualtion to find the project position 'into' the window
    vec3 planeIntersect = ((planes-hitPos)/rayDir);
    float distance;

    if(planeIntersect.x < planeIntersect.y && planeIntersect.x < planeIntersect.z) {
        //Left/Right wall
        colour=clamp(hash33(cell*48.2270)+0.7,0.0,1.0);
        distance = planeIntersect.x;
    } else if (planeIntersect.y < planeIntersect.x && planeIntersect.y < planeIntersect.z) {
        if(rayDir.y<0.0) {
            //Floor
            colour = clamp(hash33(cell*81.7619)+0.3,0.0,1.0);
        } else {
            //Ceiling
            colour =mix(clamp(hash33(cell*20.9912)+0.3,0.0,1.0),
                        lightColour*6.0,
                	abs(sin((planeIntersect.y*PI*3.0))));
        }
        distance = planeIntersect.y;
    } else if (planeIntersect.z < planeIntersect.x && planeIntersect.z < planeIntersect.y) {
        //Back wall
        colour=clamp(hash33(cell*54.8454)+0.7,0.0,1.0);
        distance = planeIntersect.z;
    } else {
        //error!
        colour=PARALLAX_INTERROOM_WALL_COLOUR;
        distance = 0.0;   
    }
    vec3 intersectionPos = ((hitPos + rayDir * distance) - roomMin);
	//add some distance and height shadow    
    colour*=clamp(intersectionPos.z*(1.0-intersectionPos.y)+0.3,0.0,1.0)*brightness*lightColour;
}

// Function 461
vec3 LPV_Sample(vec3 p, vec3 d, vec4 Cxp, vec4 Cxn, vec4 Cyp, vec4 Cyn, vec4 Czp, vec4 Czn) {
    vec3 Sqd=d*d;
    float InterpW=((d.x<0.)?Cxn.w:Cxp.w)*Sqd.x+((d.y<0.)?Cyn.w:Cyp.w)*Sqd.y+((d.z<0.)?Czn.w:Czp.w)*Sqd.z;
    vec3 InterpL=((d.x<0.)?Cxn.xyz:Cxp.xyz)*Sqd.x+((d.y<0.)?Cyn.xyz:Cyp.xyz)*Sqd.y
                +((d.z<0.)?Czn.xyz:Czp.xyz)*Sqd.z;
    return InterpL/(InterpW+0.001);
}

// Function 462
vec3 sampleBSDF( in vec3 x, in RaySurfaceHit hit, in Material mtl, in bool useMIS ) {
    vec3 Lo = vec3( 0.0 );
    float bsdfSamplingPdf = 1.0/float(BSDF_SAMPLES);
    vec3 n = hit.N * vec3((dot(hit.E, hit.N) < 0.0) ? -1.0 : 1.0);
    
    for( int i=0; i<BSDF_SAMPLES; i++ ) {
        //Generate direction proportional to bsdf
        vec3 bsdfDir;
        float bsdfPdfW;
        float Xi1 = rnd();
        float Xi2 = rnd();
        float strataSize = 1.0 / float(BSDF_SAMPLES);
        Xi2 = strataSize * (float(i) + Xi2);
        float brdf;
        
        if( mtl.bsdf_ == BSDF_R_GLOSSY ) {
            bsdfDir = sampleBlinn( n, hit.E, mtl.roughness_, Xi1, Xi2, bsdfPdfW );
            brdf = evaluateBlinn( n, hit.E, bsdfDir, mtl.roughness_ );
        } else {
            bsdfDir = sampleLambertian( n, Xi1, Xi2, bsdfPdfW );
            brdf = evaluateLambertian( n, bsdfDir );
        }
        
        float dotNWi = dot( bsdfDir, n );

        //Continue if sampled direction is under surface
        if( (dotNWi > 0.0) && (bsdfPdfW > EPSILON) ){
            //calculate light visibility
            RaySurfaceHit newHit;
            if( raySceneIntersection( Ray( x, bsdfDir ), EPSILON, newHit ) && (newHit.obj_id < LIGHT_COUNT) ) {
                //Get hit light Info
                vec3 Li;
                Sphere lightSphere;
                getLightInfo( newHit.obj_id, lightSphere, Li );

                //Read light info
                float weight = 1.0;
				float lightPdfW;
                if ( useMIS ) {
                    lightPdfW = sphericalLightSamplingPdf( x, bsdfDir, newHit.dist, newHit.N, lightSphere );
                    lightPdfW *= lightChoosingPdf(x, newHit.obj_id);
                    weight = misWeight( bsdfPdfW, lightPdfW );
                }

                Lo += brdf*dotNWi*(Li/bsdfPdfW)*weight;
            }
        }
    }

    return Lo*bsdfSamplingPdf;
}

// Function 463
vec4 texture_bilinear( const in sampler2D t, in vec2 uv )
{
  uv -= 0.5 * texelSize.xx;
  // Calculate pixels to sample and interpolating factor
  vec2 f = fract( uv * TEXTURE_SIZE );
  vec2 uvSnapped = uv - texelSize.xx * f + 0.5 * texelSize.xx;

  // As we are sampling snapped pixels need to override
  // the mip-map selection by selecting a large negative
  // bias. Otherwise at boundaries the gradient of
  // uvSnapped is large and an incorrect mip-level is used
  // leading to artifacts  
  float bias = -10.0;
  vec4 tl = texture(t, uvSnapped, bias);
  vec4 tr = texture(t, uvSnapped + texelSize, bias);
  vec4 bl = texture(t, uvSnapped + texelSize.yx, bias);
  vec4 br = texture(t, uvSnapped + texelSize.xx, bias);
    
  vec4 tA = mix( tl, tr, f.x );
  vec4 tB = mix( bl, br, f.x );
  return mix( tA, tB, f.y );
}

// Function 464
StructBuffer TexturesSampler(sampler2D _buffer,vec2 uv){
	return GetStructBuffer(texture(_buffer,uv));
}

// Function 465
vec4 sample_triquadratic(sampler3D channel, vec3 res, vec3 uv) {
    vec3 q = fract(uv * res);
    vec3 c = (q*(q - 1.0) + 0.5) / res;
    vec3 w0 = uv - c;
    vec3 w1 = uv + c;
    vec4 s = texture(channel, vec3(w0.x, w0.y, w0.z))
    	   + texture(channel, vec3(w1.x, w0.y, w0.z))
    	   + texture(channel, vec3(w1.x, w1.y, w0.z))
    	   + texture(channel, vec3(w0.x, w1.y, w0.z))
    	   + texture(channel, vec3(w0.x, w1.y, w1.z))
    	   + texture(channel, vec3(w1.x, w1.y, w1.z))
    	   + texture(channel, vec3(w1.x, w0.y, w1.z))
		   + texture(channel, vec3(w0.x, w0.y, w1.z));
	return s / 8.0;
}

// Function 466
vec3 Sample_Cos_Hemisphere ( float3 wi, float3 N, out float pdf,
                             inout float seed ) {
  vec2 u = Sample_Uniform2(seed);
  float3 wo = Reorient_Hemisphere(
                normalize(To_Cartesian(sqrt(u.y), TAU*u.x)), N);
  pdf = PDF_Cosine_Hemisphere(wo, N);
  return wo;
}

// Function 467
vec4 textureNoTile_3weights_illegible( sampler2D samp, in vec2 uv )
{
    vec2 fuv = mod( uv, 2. ), iuv = uv - fuv;
    vec3 BL_one = vec3(0.,0.,1.);
    if( fuv.x >= 1. ) fuv.x = 2.-fuv.x, BL_one.x = 2.;
    if( fuv.y >= 1. ) fuv.y = 2.-fuv.y, BL_one.y = 2.;
    
    float w12 = smoothstep( 1.125*BLEND_WIDTH, 1.-1.125*BLEND_WIDTH, dot(fuv,vec2(.5,-.5)) + .5 );
    vec4 res = mix( texture( samp, transformUVs( iuv + BL_one.xz, uv ) ), texture( samp, transformUVs( iuv + BL_one.zy, uv ) ), w12 );

    float w3 = (fuv.x+fuv.y) - 1.; vec2 iuv3;
    if( w3 < 0. ) iuv3 = iuv + BL_one.xy, w3 = -w3;
    else iuv3 = iuv + BL_one.zz;
    w3 = smoothstep( BLEND_WIDTH, 1.-BLEND_WIDTH, w3 );
    return mix( res, texture( samp, transformUVs( iuv3, uv ) ), w3 );
}

// Function 468
vec3 getCosineWeightedSample(vec3 dir
){vec3 o1=normalize(ortho(dir))
 ;vec3 o2=normalize(cross(dir,o1))
 ;vec2 r=vec2(randomFloat(),randomFloat())
 ;r.x=r.x*2.0*Pi
 ;r.y=pow(r.y,.5)
 ;float oneminus=sqrt(1.0-r.y*r.y)
 ;return cos(r.x)*oneminus*o1+sin(r.x)*oneminus*o2+r.y*dir;}

// Function 469
vec3 CosineWeightedSampleHemisphere ( vec3 normal, vec2 rnd )
{
   //rnd = vec2(rand(vec3(12.9898, 78.233, 151.7182), seed),rand(vec3(63.7264, 10.873, 623.6736), seed));
   float phi = acos( sqrt(1.0 - rnd.x)) ;
   float theta = 2.0 * 3.14 * rnd.y ;

   vec3 sdir = cross(normal, (abs(normal.x) < 0.5001) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0));
   vec3 tdir = cross(normal, sdir);

   return normalize(phi * cos(theta) * sdir + phi * sin(theta) * tdir + sqrt(1.0 - rnd.x) * normal);
}

// Function 470
void Sample_GGX_R(vec2 s, vec3 V, vec3 N, float alpha, vec3 F0, out vec3 L, out vec3 w)
{
    vec3 H;
    {
    	vec3 ox, oz;
		OrthonormalBasisRH(N, /*out*/ ox, oz);
    	
    	vec3 Vp = vec3(dot(V, ox), dot(V, oz), dot(V, N));
    	
        vec3 Hp = Sample_GGX_VNDF(Vp, alpha, alpha, s.x, s.y);
    	
        H = ox*Hp.x + N*Hp.z + oz*Hp.y;
    }
    
    vec3 F = FresnelSchlick(dot(H, V), F0);

    L = 2.0 * dot(V, H) * H - V;
    
    float NoV = clamp01(dot(N, V));
    float NoL = clamp01(dot(N, L));
    
    float G2 = GGX_G(NoV, NoL, alpha);
    float G1 = GGX_G(NoV, alpha);
    
    w = G1 == 0.0 ? vec3(0.0) : F * G2 / G1;
}

// Function 471
vec3 sampleSun(const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, float seed) {
    vec2 u = vec2(random(), random());
    
    vec3 tangent = vec3(0.), binormal = vec3(0.);
    vec3 lightDir = light.direction;
    createBasis(lightDir, tangent, binormal);
    
    float cosThetaMax = 1. - SUN_SOLID_ANGLE/TWO_PI;
    wi = uniformSampleCone(u, cosThetaMax, tangent, binormal, lightDir);
    
    if (dot(wi, interaction.normal) > 0.) {
        lightPdf = 1. / SUN_SOLID_ANGLE;
    }
    
	return light.L;
}

// Function 472
void TextureEnvBlured2(in vec3 N, in vec3 Rv, out vec3 iblDiffuse, out vec3 iblSpecular) {
    iblDiffuse = vec3(0.0);
    iblSpecular = vec3(0.0);
	
    mat3 shR, shG, shB;
    
    CubeMapToSH2(shR, shG, shB);
    
    #if 1
    	shR = shDiffuseConvolution(shR);
    	shG = shDiffuseConvolution(shG);
    	shB = shDiffuseConvolution(shB);
    #endif
    
    #if 0
    	shR = shDiffuseConvolutionPI(shR);
    	shG = shDiffuseConvolutionPI(shG);
    	shB = shDiffuseConvolutionPI(shB);
    #endif    
    
    iblDiffuse = SH2toColor(shR, shG, shB, N);
}

// Function 473
float Sample_Hex( vec2 hex_pos )
{
	return Sample( FromHex( hex_pos ) );
}

// Function 474
vec3 texture_wood(vec3 pos) {
    pos = quat_mul(quat(vec3(1,0,0),-0.0), pos);
   	//pos.z -= 1.0;
    vec2 core = vec2(cos(pos.z), sin(pos.z))*0.1;
    pos.xy -= core;
    
	float r = length(pos.xy);
    float a = (TAU/2.0 + atan(pos.x,pos.y)) / TAU;
    
    float r_noise = noise(vec2(cos(a*TAU*2.0), sin(a*TAU*2.0)));
    r_noise += noise(vec2(10.0) + vec2(cos(a*TAU*4.0), sin(a*TAU*4.0))) * 0.5; // squigglyness
    r_noise += noise(vec2(100.0) + vec2(cos(a*TAU*8.0), sin(a*TAU*8.0))) * 0.4; // squigglyness
    r_noise += noise(vec2(1000.0) + vec2(cos(a*TAU*16.0), sin(a*TAU*16.0))) * 0.2; // squigglyness
    
    r_noise += noise(pos.z*0.5)*3.0; // knottyness
    
    r_noise *= noise(r*3.0)*5.0; // whorlyness
    r += r_noise*0.05*clamp(r,0.0,1.0); // scale and reduce at center
    
    vec3 col = vec3(1.0,0.8,0.35);
    //float c = 0.5 + 0.5*sin(r*100.0); // 100 rings per meter ~ 1cm rings
    float c = fract(r*5.0);
    //c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, (c-0.15)/0.85);
    c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, sqrt(clamp((c-0.15)/0.85,0.0,1.0)));
    //c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, pow(clamp((c-0.15)/0.85,0.0,1.0), 0.25));
    col = mix(col, vec3(0.5,0.25,0.1)*0.4, c); // ring gradient
    col = mix(col, col*0.8, noise(r*20.0)); // ring-to-ring brightness
    
    return col;
}

// Function 475
vec3 SampleNoiseV3(vec3 p)
{
    // sampling at decreasing scale and height multiple tiles and returning that amount divided by the total possible amount (to normalize it)
    vec3 h = noisedFractal(p).xyz;
    h += noisedFractal(p*2. + 100.).xyz * 0.5;
    h += noisedFractal(p*4. - 100.).xyz * 0.25;
    h += noisedFractal(p*8. + 1000.).xyz * 0.125;
    return h * 0.536193029;
}

// Function 476
bool getSamples(vec3 ro, vec3 rd, out float ratio,
        out float disDistance, out float disDensityIntegral, out float disCoeff,
        out float eqaDistance, out float eqaDensityIntegral, out float eqaCoeff){
   
    float c = 2.*dot(absorbtion+scattering,vec3(0.3333));
    
    //equi-angular
    vec3 lo = vec3(0);
    float delta = dot(lo - ro, rd);
    float D = length(ro + delta*rd - lo);
    float thetaA = atan(0.0 - delta, D);
    float thetaB = PI/2.;
    
    //find coefficients
    vec2 bounds = vec2(0);
    
    float eqaCdfSum = 0.;
    float densitySum = 0.;
    
    int steps = 0;
    while (bounds.x < 10.){
        steps++;
        
    	bounds.x = getIntersection(ro,rd,bounds.y+boundAcc,1.);
        bounds.y = getIntersection(ro,rd,bounds.x+boundAcc,-1.);
        if (bounds.y-bounds.x < 2.*boundAcc) continue;
        
        //t = D*tan(mix(thetaA, thetaB, u))+delta;
        //(1-u)A + uB = A - uA + uB = u(B-A) + A
        //atan((t-delta)/D) = u(B-A) + A  
        eqaCdfSum += 
            (atan((bounds.y-delta)/D) - thetaA)/(thetaB - thetaA) - 
            (atan((bounds.x-delta)/D) - thetaA)/(thetaB - thetaA);
        densitySum += transmittanceBetween(ro, rd, bounds);
    }
    if (steps == 1) return false;
    
    eqaCoeff = 1./eqaCdfSum;
    disCoeff = c/(1.-exp(-1.*densitySum*c));
    ratio = 1.-exp(-1.*densitySum*c);
    
    // calculate sample targets
    float u = rand();
    disDensityIntegral = log(u/(disCoeff/-c)+1.)/-c;
    
    float targetEqaCdf = rand();
    
    // Find bounds to search between
    bounds = vec2(0);
    
    densitySum = 0.;
    float densitySumLast = 0.;
    eqaCdfSum = 0.;
    float eqaCdfSumLast = 0.;
    float eqaTempSum, eqaTempBound;
    vec2 disTempBounds;
    float disTempTarget;
    bool lookingDis = true;
    bool lookingEqa = true;
    
    while (bounds.x < 10.){
    	bounds.x = getIntersection(ro,rd,bounds.y+boundAcc,1.);
        bounds.y = getIntersection(ro,rd,bounds.x+boundAcc,-1.);
        if (bounds.y-bounds.x < 2.*boundAcc) continue;
        
        //eqa
        eqaCdfSumLast = eqaCdfSum;
        float eqaLower = (atan((bounds.x-delta)/D) - thetaA)/(thetaB - thetaA)*eqaCoeff;
        eqaCdfSum += 
            (atan((bounds.y-delta)/D) - thetaA)/(thetaB - thetaA)*eqaCoeff - 
            eqaLower;
            
        if (targetEqaCdf >= eqaCdfSumLast && targetEqaCdf < eqaCdfSum){
            eqaDistance = D*tan(mix(thetaA, thetaB, (
                (targetEqaCdf-eqaCdfSumLast)+eqaLower
            )/eqaCoeff))+delta;
            eqaTempBound = bounds.x;
            eqaTempSum = densitySum;
        }
        
        //dis
        densitySumLast = densitySum;
        float densityLower = evalInt(ro,rd,bounds.x);
        densitySum += evalInt(ro,rd,bounds.y) - densityLower;
        if (disDensityIntegral >= densitySumLast && disDensityIntegral < densitySum){
            disTempBounds = bounds;
            disTempTarget = (disDensityIntegral - densitySumLast)+densityLower;
        }
        
    }
    
    eqaDensityIntegral = eqaTempSum+transmittanceBetween(ro, rd, vec2(eqaTempBound,eqaDistance));
    disDistance = searchDistance(disTempTarget, disTempBounds, ro, rd);
    return true;
}

// Function 477
float getSampleDim0(int sampleIndex,vec2 fragCoord)
{
	return fract(getDimensionHash(0,fragCoord) + radicalInverse(sampleIndex, 2));
}

// Function 478
float circleSample(vec2 dist, vec2 center)
{
    vec2 curRadius = vec2(0.0);
	float returnValue = 0.0;
    
    for (int c = 0; c < CIRCLE_NUMBER; ++c)
    {
    	float normalizedAngle = 0.0;
        curRadius += dist;
        for (int s = 0; s < SAMPLE_PER_CIRCLE; ++s)
        {
            float angle = normalizedAngle * 3.1415 * 2.0;
            vec2 uvToSample = center + vec2(cos(angle), sin(angle)) * curRadius;
            vec3 sampledColor = texture(iChannel0, uvToSample).rgb;
            if (passTest(sampledColor) == false)
                returnValue += 1.0 / float(CIRCLE_NUMBER * SAMPLE_PER_CIRCLE);
            normalizedAngle += 1.0 / float(SAMPLE_PER_CIRCLE);
        }
    }
    return (returnValue);
}

// Function 479
vec4 AntiAliasPointSampleTexture_Smoothstep(vec2 uv, vec2 texsize) {	
	vec2 w=fwidth(uv);
	return texture(iChannel0, (floor(uv)+0.5+smoothstep(0.5-w,0.5+w,fract(uv))) / texsize, -99999.0);	
}

// Function 480
vec4 sampleLevel0(sampler2D sceneTexture, vec2 uv, float mipLevel)
{
    return textureLod(sceneTexture, uv, mipLevel);
}

// Function 481
vec3 getHemisphereGGXSample(vec3 n, vec3 v, float roughness, out float weight) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    
    float epsilon = clamp(getRandom(), 0.001, 1.);
    float cosTheta2 = (1. - epsilon) / (epsilon * (alpha2 - 1.) + 1.);
    float cosTheta = sqrt(cosTheta2);
    float sinTheta = sqrt(1. - cosTheta2);
    
    float phi = 2. * M_PI * getRandom();
    
    // Spherical to cartesian
    vec3 t = normalize(cross(n.yzx, n));
    vec3 b = cross(n, t);
    
	vec3 microNormal = (t * cos(phi) + b * sin(phi)) * sinTheta + n * cosTheta;
    
    vec3 l = reflect(-v, microNormal);
    
    // Sample weight
    float den = (alpha2 - 1.) * cosTheta2 + 1.;
    float D = alpha2 / (M_PI * den * den);
    float pdf = D * cosTheta / (4. * dot(microNormal, v));
    weight = (.5 / M_PI) / (pdf + 1e-6);
    
    if (dot(l, n) < 0.)
        weight = 0.;
    
    return l;
}

// Function 482
vec3 texture_surface(vec3 position, vec3 normal) {
    vec4 noise = texture(iChannel0, position.xz * 1.0 + normal.y * 0.1);
    
    vec3 col = mix(UNDERSIDE_COLOR, SURFACE_COLOR, normal.y + normal.x*normal.z);
    col = mix(col, noise.rgb, 0.5);
    
    return col;
}

// Function 483
vec3 sampleX(vec2 uv, vec2 dispNorm, float disp) {
    vec3 col = vec3(0);
    const float SD = 1.0 / float(SAMPLES);
    float wl = 0.0;
    vec3 denom = vec3(0);
    for(int i = 0; i < SAMPLES; i++) {
        vec3 sw = sampleWeights(wl);
        denom += sw;
        col += sw * texture(iChannel1, uv + dispNorm * disp * wl).xyz;
        wl  += SD;
    }
    
    return col / denom;
}

// Function 484
vec3 sampleIndirectLight(vec3 pos, vec3 normal){
    vec3 dir;
    vec3 abso = vec3(1.), light = vec3(0.), dc, ec;
    for(int i = 0; i < Bounces; i++){
        dir = getCosineWeightedSample(normal);
        if(!trace(pos, dir, normal)) return light + abso*background(dir);
        sdf(pos, dc, ec);
        light += abso * (ec + dc*directLight(pos, normal));
        abso *= dc;
    }
    return light;
}

// Function 485
vec3 ggx_sample(vec3 wi, float alphax, float alphay, float Xi1, float Xi2) {
    //stretch view
    vec3 v = normalize(vec3(wi.x * alphax, wi.y * alphay, wi.z));

    //orthonormal basis
    vec3 t1 = (v.z < 0.9999) ? normalize(cross(v, vec3(0.0, 0.0, 1.0))) : vec3(1.0, 0.0, 0.0);
    vec3 t2 = cross(t1, v);

    //sample point with polar coordinates
    float a = 1.0 / (1.0 + v.z);
    float r = sqrt(Xi1);
    float phi = (Xi2 < a) ? Xi2 / a*PI : PI + (Xi2 - a) / (1.0 - a) * PI;
    float p1 = r*cos(phi);
    float p2 = r*sin(phi)*((Xi2 < a) ? 1.0 : v.z);

    //compute normal
    vec3 n = p1*t1 + p2*t2 + v*sqrt(1.0 - p1*p1 - p2*p2);

    //unstretch
    return normalize(vec3(n.x * alphax, n.y * alphay, n.z));
}

// Function 486
vec4 textureAniso(sampler2D T, vec2 p) {
    mat2 J = inverse(mat2(dFdx(p),dFdy(p)));       // dFdxy: pixel footprint in texture space
    J = transpose(J)*J;                            // quadratic form
    float d = determinant(J), t = J[0][0]+J[1][1], // find ellipse: eigenvalues, max eigenvector
          D = sqrt(t*t-4.*d), 
          V = (t-D)/2., v = (t+D)/2.,                     // eigenvalues 
          M = 1./sqrt(V), m = 1./sqrt(v), l =log2(m*R.y); // = 1./radii^2
  //if (M/m>16.) l = log2(M/16.*R.y);                     // optional
    vec2 A = M * normalize(vec2( -J[0][1] , J[0][0]-V )); // max eigenvector = main axis
    vec4 O = vec4(0);
    for (float i = -7.5; i<8.; i++)                       // sample x16 along main axis at LOD min-radius
        O += textureLod(iChannel0, p+(i/16.)*A, l);
    return O/16.;
}

// Function 487
vec4 textureSmootherstep(sampler2D tex, vec2 uv, vec2 res)
{
	uv = uv*res + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*fuv*(fuv*(fuv*6.0-15.0)+10.0);
	uv = (uv - 0.5)/res;
	return texture( tex, uv );
}

// Function 488
vec3 sample_cos_hemisphere(in vec3 N,float x1, float x2,out float pdf){
    float phi = M_2PI_F * x1;
	float r = sqrt(x2);
	x1 = r * cos(phi);
	x2 = r * sin(phi);
	vec3 T, B;
	frisvad (N, T, B);
	float costheta = sqrt(max(1.0f - x1 * x1 - x2 * x2, 0.0));
	pdf = M_1_PI_F;
	return x1 * T + x2 * B + costheta * N;
}

// Function 489
vec3 sampleBackground(vec3 v)
{
    v = vec3(v.x, -v.z, v.y);
    vec3 rgb = texture(iChannel1, v).xyz;
    return rgb * rgb * 4.; // linearize
}

// Function 490
vec3 disneySheenSample(out vec3 wi, const in vec3 wo, out float pdf, const in vec2 u, const in vec3 normal, const in MaterialInfo material) {
    
    cosineSample

    vec3 H = normalize(wo+wi);
    float LdotH = dot(wo,H);
    
    pdf = pdfLambertianReflection(wi, wo, normal);
    return disneySheen(LdotH, material);
}

// Function 491
vec3 textureNormal(vec2 uv) {
    vec3 normal = texture( iChannel1, 100.0 * uv ).rgb;
    normal.xy = 2.0 * normal.xy - 1.0;
    
    // Adjust n.z scale with mouse to show how flat normals behave
    normal.z = sqrt(iMouse.x / iResolution.x);
    return normalize( normal );
}

// Function 492
vec3 ImportanceSampleGGX( vec2 uniformSamplePos, vec3 N, float alpha2 )
{
	vec2 sphereSamplePos = ImportanceSampleGGXTransform( uniformSamplePos, alpha2 );

	vec3 specSpaceH = SphericalToCartesianDirection( sphereSamplePos );
	
	mat3 specToCubeMat = OrthoNormalMatrixFromZ( N );

	return specToCubeMat * specSpaceH;
}

// Function 493
vec3 sampleSun(const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, float seed) {
    wi = light.direction;
    return light.L;
}

// Function 494
vec3 sample_uniform_hemisphere(in vec3 N,float x1, float x2, out float pdf){
	float z = x1;
	float r = sqrt(max(0., 1. - z*z));
	float phi = M_2PI_F * x2;
	float x = r * cos(phi);
	float y = r * sin(phi);
	vec3 T, B;
	frisvad (N, T, B);
	pdf = 0.5 * M_1_PI_F;
    return x * T + y * B + z * N;
}

// Function 495
vec4 textureAniso(sampler2D T, vec2 p) {
    mat2 J = mat2(dFdx(p),dFdy(p));                 // pixel footprint in texture space
    vec2 A,a; float M,m,l;
    ellips(J, A,a,M,m); 
    A *= M;
    l = log2( m * R.y );                            // MIPmap level corresponding to min radius
    if (M/m>16.) l = log2(M/16.*R.y);               // optional      
    vec4 O = vec4(0);
    for (float i = -7.5; i<8.; i++)                 // sample x16 along main axis at LOD min-radius
        O += textureLod(iChannel0, p+(i/16.)*A, l);
    return O/16.;
}

// Function 496
vec3  anytextureNEW(vec2 uv){
    uv.y *= 2.;
    uv = sin(3.14*uv);
    float a =uv.x*uv.y;
    return vec3(1,.7,.7)* (.5 - .5 * a/fwidth(a));
}

// Function 497
vec3 LightSample(vec3 p,float x1,float x2,out vec3 wo,out float dist,out float pdf){
	vec3 v0v1 = quads[0].v1 - quads[0].v0;
    vec3 v0v3 = quads[0].v3 - quads[0].v0;
    float width  = length(v0v1);
    float height = length(v0v3);
    vec3 O = quads[0].v0 + v0v1*x1 + v0v3*x2;
    wo = O - p;
    dist = length(wo);
    wo = normalize(wo);
    float costhe = dot(-wo,quads[0].normal);
    pdf = PDF_Area2Angle(1./(width*height),dist,clamp(costhe,0.00001,1.));
    return costhe>0. ? GetLightIntensity(): vec3(0.);
}

// Function 498
float sampleDepth(in vec2 uv)
{
	float height = pow(texture(iChannel0,uv).r,2.2);
    return 1.0 - pow(height,1.0/3.0);
}

// Function 499
vec3 sampleLightSource(in vec3 x, in vec3 n, float Xi1, float Xi2, out LightSamplingRecord sampleRec) {
    vec3 s = light.pos - vec3(1., 0., 0.) * light.size.x * 0.5 -
        				 vec3(0., 0., 1.) * light.size.y * 0.5;
    vec3 ex = vec3(light.size.x, 0., 0.);
    vec3 ey = vec3(0., 0., light.size.y);
    
    SphQuad squad;
    SphQuadInit(s, ex, ey, x, squad);
    SphQuadSample(x, squad, Xi1, Xi2, sampleRec);
    
    //we don't have normal for volumetric particles
    if(dot(n,n) < EPSILON) {
        SphQuadSample(x, squad, Xi1,Xi2, sampleRec);
    } else {
        LightSamplingRecord w[CDF_SIZE];
        float ww[CDF_SIZE];
        const float strata = 1.0 / float(CDF_SIZE);
        for(int i=0; i<CDF_SIZE; i++) {
            float xi = strata*(float(i)+rnd());
            SphQuadSample(x, squad, xi, rnd(), w[i]);
            ww[i] = (i == 0)? 0.0 : ww[i-1];
            ww[i] += max(0.0, dot(w[i].w, n));
        }

        float a = Xi1 * ww[CDF_SIZE-1];
        for(int i=0; i<CDF_SIZE; i++) {
            if(ww[i] > a) {
                sampleRec = w[i];
                sampleRec.pdf *= (ww[i] - ((i == 0)? 0.0 : ww[i-1])) / ww[CDF_SIZE-1];
                sampleRec.pdf *= float(CDF_SIZE);
                break;
            }
        }
    }
    
	return getRadiance(vec2(Xi1,Xi2));
}

// Function 500
vec3 uniformSampleHemisphere(const in vec2 u) {
    float z = u[0];
    float r = sqrt(max(EPSILON, 1. - z * z));
    float phi = 2. * PI * u[1];
    return vec3(r * cos(phi), r * sin(phi), z);
}

