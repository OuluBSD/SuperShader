// Reusable Texture Blending Texturing Functions
// Automatically extracted from texturing/mapping-related shaders

// Function 1
vec3 color_dist_mix(vec3 bg, vec3 fg, float dist, float alpha) {
    float d = smoothstep(0.0, 0.75, dist); 
    return mix(bg, fg, alpha*(1.0-d));
}

// Function 2
float mix3d_max(float x, float y, float z) {
	return max(max(x, y), z);
}

// Function 3
d1 mixd(d1 a,d  b,v0 c){return mixd(a,d0td1(b),c);}

// Function 4
float unmix(float a, float b, float x) {
    return (x - a)/(b - a);
}

// Function 5
uint MixHash(uvec3 h)
{
    return ((h.x ^ (h.y >> 16u) ^ (h.z << 15u)) * rPhi3.x) ^ 
           ((h.y ^ (h.z >> 16u) ^ (h.y << 15u)) * rPhi3.y) ^
           ((h.z ^ (h.y >> 16u) ^ (h.x << 15u)) * rPhi3.z);
}

// Function 6
uint MixHash(uvec4 h)
{
    return ((h.x ^ (h.y >> 16u) ^ (h.z << 15u)) * rPhi4.x) ^ 
           ((h.y ^ (h.z >> 16u) ^ (h.w << 15u)) * rPhi4.y) ^
           ((h.z ^ (h.w >> 16u) ^ (h.x << 15u)) * rPhi4.z) ^
           ((h.w ^ (h.x >> 16u) ^ (h.y << 15u)) * rPhi4.w);
}

// Function 7
vec3 blend_burn(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*vec3(blend_burn_f(c1.x, c2.x), blend_burn_f(c1.y, c2.y), blend_burn_f(c1.z, c2.z)) + (1.0-opacity)*c2;
}

// Function 8
GNum2 a_mix(in float a, in GNum2 b, in GNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

// Function 9
float mix3d_add(float x, float y, float z) {
	return min(x+y+z, 1.0);
}

// Function 10
GNum2 a_mix(in GNum2 a, in float b, in float t)
{
    return add(mult(a, 1.0 - t), b*t);
}

// Function 11
vec3 blendNormals(in vec3 norm1, in vec3 norm2)
{
	return normalize(vec3(norm1.xy + norm2.xy, norm1.z));
}

// Function 12
vec4 blend(vec4 old, vec4 new, float i){return (1.-(1./i))*old + (1./i) * new;}

// Function 13
vec3 NormalBlend_UDN(vec3 n1, vec3 n2)
{
    // Unpack
	n1 = n1*2.0 - 1.0;
    n2 = n2*2.0 - 1.0;    
    
	return normalize(vec3(n1.xy + n2.xy, n1.z));
}

// Function 14
vec3 blend_darken_o5229 (vec3 c1, vec3 c2, float opacity) {
	//return min(c1, c2);
	return opacity*min(c1, c2) + (1.0-opacity)*c2;
}

// Function 15
vec3 blend(vec3 old, vec3 new, float i){return (1.-(1./i))*old + (1./i) * new;}

// Function 16
GNum2 a_mix(in float a, in float b, in GNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

// Function 17
d1 mixd(d1 a,d1 b,v0 c){return d1(mix(a.x,b.x,c),mix(a.d,b.d,c));}

// Function 18
vec3 blendOklab(vec2 uv, vec3 lhsRgb, vec3 rhsRgb) {
    vec3 lhs = linearSrgbToOklab(srgbTransferInv(lhsRgb));
    vec3 rhs = linearSrgbToOklab(srgbTransferInv(rhsRgb));
    return oklabToLinearSrgb(mix(lhs, rhs, uv.x));
}

// Function 19
vec2 blend(vec2 c, float s)
{
    c.x *= iResolution.x / iResolution.y;
    vec2 pos = smoothstep(vec2(s), c/4.0, c/5.0*sin(c));
    float i = length(inversesqrt(s));
    vec2 x = pos / i;
    return x * sin(i) * cos(i) * tan(i) + (iTime / 16.0);
}

// Function 20
float Mix(float a, float b)
{
    // Smoothly blends but darkens with each blend
    //return a * b;
    
    // Distance field union
    // Maintains correct intensity regardless of blend count
    // Blends with visual discontinuity
    // My intuition is telling me I shouldn't see the discontinuity!
    return min(a, b);
    
    // Smooth min
    //return smin(a, b, 0.07);
    
    // Looks good but variable under number of blends
    // Effectively "steals" detail from previous blends
    //return sqrt(a * b);
}

// Function 21
float blendedContours (float f, vec2 gradient, float minSpacing, float divisions, float lineWidth, float antialiasing) {
  float screenSpaceLogGrad = hypot(gradient) / f;
  float localOctave = log2(screenSpaceLogGrad * minSpacing) / log2(divisions);
  float contourSpacing = pow(divisions, ceil(localOctave));
  float plotVar = log2(f) / contourSpacing;
  float widthScale = 0.5 * contourSpacing / screenSpaceLogGrad;

  float contourSum = 0.0;
  for(int i = 0; i < octaves; i++) {
    // A weight which fades in the smallest octave and fades out the largest
    float t = float(i + 1) - fract(localOctave);
    float weight = smoothstep(0.0, 1.0, t) * smoothstep(float(octaves), float(octaves) - 1.0, t);

    contourSum += weight * smoothstep(
      0.5 * (lineWidth + antialiasing),
      0.5 * (lineWidth - antialiasing),
      (0.5 - abs(fract(plotVar) - 0.5)) * widthScale
    );

    // Rescale for the next octave
    widthScale *= divisions;
    plotVar /= divisions;
  }

  return contourSum / float(octaves);
}

// Function 22
float blend_dodge_f(float c1, float c2) {
	return (c1==1.0)?c1:min(c2/(1.0-c1),1.0);
}

// Function 23
vec3 blendOklrab(vec2 uv, vec3 lhsRgb, vec3 rhsRgb) {
    vec3 lhs = linearSrgbToOklrab(srgbTransferInv(lhsRgb));
    vec3 rhs = linearSrgbToOklrab(srgbTransferInv(rhsRgb));
    return oklrabToLinearSrgb(mix(lhs, rhs, uv.x));
}

// Function 24
float get_blend_factor(in sampler2D s)
{
    return texelFetch(s, CTRL_BLEND_FACTOR, 0).w;
}

// Function 25
vec3 NormalBlend_Unity(vec3 n1, vec3 n2)
{
    // Unpack
	n1 = n1*2.0 - 1.0;
    n2 = n2*2.0 - 1.0;
    
    mat3 nBasis = mat3(vec3(n1.z, n1.y, -n1.x), // +90 degree rotation around y axis
        			   vec3(n1.x, n1.z, -n1.y), // -90 degree rotation around x axis
        			   vec3(n1.x, n1.y,  n1.z));
	
    return normalize(n2.x*nBasis[0] + n2.y*nBasis[1] + n2.z*nBasis[2]);
}

// Function 26
vec4 VXAADifferentialBlend( vec4 n[4], vec2 w )
{
    vec4 c = vec4( 0.0 );
    c += ( n[ VXAA_W ] + n[ VXAA_E ] ) * w.x;
    c += ( n[ VXAA_N ] + n[ VXAA_S ] ) * w.y;
    return c;
}

// Function 27
float getMixValue(float cycle, inout float offset1, inout float offset2)
{
    // mixval 0..1..0 over full cycle
    float mixval = cycle * 3.0;
    if(mixval > 2.0) mixval = 3.0 - mixval;
    
    // texture phase 1 
    offset1 = cycle;
    // texture phase 2, phase 1 offset by .5
    offset2 = mod(offset1 + .6, 2.0);
    return mixval;
}

// Function 28
uint MixHash(uvec2 h)
{
    return ((h.x ^ (h.y >> 16u)) * rPhi2.x) ^ 
           ((h.y ^ (h.x >> 16u)) * rPhi2.y);
}

// Function 29
float blend(vec2 c, float s)
{
    c.x *= iResolution.x / iResolution.y;
    vec2 pos = smoothstep(vec2(s), c/2.0, c/3.0*sin(c));
    float i = abs(sqrt(s));
    vec2 x = pos / i;
    return length(x) - sin(i) / cos(i) / tan(i);
}

// Function 30
vec4 blend(float a, float b, float m) {
    return vec4(min(min(a, b), a * b - m), (a * colorA + b * colorB) / (a + b));
}

// Function 31
void mixColor(vec4 col, float alpha)
{
    fcol = vec4(mix(fcol.rgb, col.rgb, alpha * col.a), 1.0);
}

// Function 32
vec3 alphaBlend(vec3 c1, vec3 c2, float alpha)
{
    return mix(c1,c2,clamp(alpha,0.0,1.0));
}

// Function 33
vec4 blend_Halftone()
{
    const int xsegs = 24;
    const int ysegs = 24;

    float phaseLim  = phase * 1.5 / float(xsegs);
    const float w = 0.05 / float(xsegs);

    float seg_w = 1.0 / float(xsegs);
    float seg_h = 1.0 / float(ysegs);

    int segnumx = int(uv.x * float(xsegs));
    int segnumy = int(uv.y * float(ysegs));

    vec2 center;
    center.x = float(segnumx) * seg_w + seg_w * 0.5;
    center.y = float(segnumy) * seg_h + seg_h * 0.5;

    float dist_to_center = length(uv - center);

    return interpolateColor(phaseLim, w, dist_to_center);
}

// Function 34
vec3 blend_dodge(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*vec3(blend_dodge_f(c1.x, c2.x), blend_dodge_f(c1.y, c2.y), blend_dodge_f(c1.z, c2.z)) + (1.0-opacity)*c2;
}

// Function 35
vec4 master_blend_o5268 (vec4 c1, vec4 c2, float amount){
	return vec4(blend_darken_o5268(c1.rgb, c2.rgb, c1.a*amount),
	            min(1.0, c2.a+amount*c1.a));
}

// Function 36
d2 mixd(d2 a,d2 b,v0 c){return d2(mix(a.x,b.x,c),mix(a.d,b.d,c));}

// Function 37
vec3 NormalBlend_Linear(vec3 n1, vec3 n2)
{
    // Unpack
	n1 = n1*2.0 - 1.0;
    n2 = n2*2.0 - 1.0;
    
	return normalize(n1 + n2);    
}

// Function 38
float blend (float a, float b) 
{
	const float k = 0.8;    // fusion factor
    float h = clamp (0.5+0.5*(b-a)/k, 0.0, 1.0);
    return mix(b,a,h) - k*h*(1.0-h);
}

// Function 39
vec4 contrastBlend( vec4 A, vec4 B, float alpha )
{
    // brightness
    float Ab = A.x * B.y * A.z;
    float Bb = B.x * B.y * B.z;
    
    // work out blend param based on relative brightness, smoothstep
    // to increase contrast.
    float contrastAlpha = smoothstep(0.,1.,2.*Ab/(Ab+Bb));
    
    // push our alpha towards the contrastAlpha, but still follow alpha to 0 and 1.
    
    // goes to 0 when alpha is near 0 or 1
    float proximityTo01 = min(alpha,1.-alpha);
    // powering it up seems to give better results
    proximityTo01 = pow( proximityTo01, .25 );
    
    // mix between alpha and contrast-aware alpha
    alpha = mix( alpha, contrastAlpha, proximityTo01 );
    
    // blend
    return mix( A, B, alpha );
}

// Function 40
void mixColorPoint(vec2 uv,inout vec3 col,vec2 colPoint,float scale)
{
    //float dist = length(uv - colPoint) * scale;
    //dist = pow(dist,0.25);
    //dist = 1.0 - smoothstep(0.0,1.0,dist);
    
    vec2 uv_ = (uv - colPoint)*scale*24.0;
    float dist = dot(uv_,uv_);
    dist = 1.0 / ( 1.0 + dist );
    
    col = mix(
        col , 
        hash3point(colPoint) ,
        dist
    );
}

// Function 41
vec3 TriPlanarBlendWeightsStandard(vec3 normal) {
	vec3 blend_weights = abs(normal.xyz); 
    
    float blendZone = 0.55;//anything over 1/sqrt(3) or .577 will produce negative values in corner
	blend_weights = blend_weights - blendZone; 

	blend_weights = max(blend_weights, 0.0);     
	float rcpBlend = 1.0 / (blend_weights.x + blend_weights.y + blend_weights.z);
	return blend_weights*rcpBlend;
}

// Function 42
vec4 blendOnto(vec4 cFront, vec4 cBehind) {
    return cFront + (1.0 - cFront.a)*cBehind;
}

// Function 43
vec2
mix_op_9avg( in vec2 p0, in vec2 p1, float max_dist )
{
    bool do_mix = p0.x > 0.0 && p0.x < max_dist &&
                  p1.x > 0.0 && p1.x < max_dist;
    if ( !do_mix && p0.x < p1.x )
    {
        return p0;
    }
    if ( !do_mix && p1.x < p0.x )
    {
        return p1;
    }
    vec2 m1 = mix_op_(p0,p1,vec2(max_dist*0.9, max_dist*0.1));
    vec2 m2 = mix_op_(p0,p1,vec2(max_dist*0.8, max_dist*0.2));
    vec2 m3 = mix_op_(p0,p1,vec2(max_dist*0.7, max_dist*0.3));
    vec2 m4 = mix_op_(p0,p1,vec2(max_dist*0.6, max_dist*0.4));
    vec2 m5 = mix_op_(p0,p1,vec2(max_dist*0.5, max_dist*0.5));
    vec2 m6 = mix_op_(p0,p1,vec2(max_dist*0.4, max_dist*0.6));
    vec2 m7 = mix_op_(p0,p1,vec2(max_dist*0.3, max_dist*0.7));
    vec2 m8 = mix_op_(p0,p1,vec2(max_dist*0.2, max_dist*0.8));
    vec2 m9 = mix_op_(p0,p1,vec2(max_dist*0.1, max_dist*0.9));
    #if 0
    return vec2((m1.x+m2.x+m3.x+m4.x+m5.x+m6.x+m7.x+m8.x+m9.x)/9.0,m9.y);
    #else
    return
    vec2( min(
       vec3( min(vec3(m1.x,m2.x,m3.x)),
             min(vec3(m4.x,m5.x,m6.x)),
             min(vec3(m7.x,m8.x,m9.x)) ) ), m9.y );
    #endif
}

// Function 44
float opUBlend(float d1,float d2,float k)
{
	float h = clamp(.5 + .5*(d2 - d1)/k,0.,1.);
    return mix(d2,d1,h)-k*h*(1.0-h);
    //return d2;
    //return min(d1,d2);
}

// Function 45
float ShapeBlend(float y, float progress) {
    float shapeProgress = clamp(progress * 2. - .5, 0., 1.);
    float shapeBlend = blend(y, .8, shapeProgress);
    return shapeBlend;
}

// Function 46
gia1 gia_mix (gia1 a, gia1 b, gia1 x) {
    return gia_add(a, gia_mul(gia_sub(b,a),x));
}

// Function 47
vec4 Blend(in vec4 fg, in vec4 bg) {
	float a = 1.- fg.a;
	return fg + bg * a;
}

// Function 48
vec3 blend_overlay(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*vec3(blend_overlay_f(c1.x, c2.x), blend_overlay_f(c1.y, c2.y), blend_overlay_f(c1.z, c2.z)) + (1.0-opacity)*c2;
}

// Function 49
float blend(float a, float b, float k) {
    float h = clamp01(mix(1.0, (b-a)/k, 0.5));
    return mix(b, a, h) - k*h*(1.0-h);
}

// Function 50
vec4 blend_Halftone2(float lim1, float lim2)
{
    const int segs = 32;

    float phase = 1.0 - (uv.x - lim1) / (lim2 - lim1);
    float seg_w = 1.0 / float(segs);

    int segnumx = int(uv.x * float(segs));
    int segnumy = int(uv.y * float(segs));

    vec2 center;
    center.x = float(segnumx) * seg_w + seg_w * 0.5;
    center.y = float(segnumy) * seg_w + seg_w * 0.5;

    float dist_to_center = length(uv - center) / seg_w;

    return interpolateColor(phase, 0.05, dist_to_center);
}

// Function 51
float mix3d_xor(float x, float y, float z) {
	float xy = min(x+y, 2.0-x-y);
	return min(xy+z, 2.0-xy-z);
}

// Function 52
float blend_soft_light_f(float c1, float c2) {
	return (c2 < 0.5) ? (2.0*c1*c2+c1*c1*(1.0-2.0*c2)) : 2.0*c1*(1.0-c2)+sqrt(c1)*(2.0*c2-1.0);
}

// Function 53
vec3 blend_rnm_pd(vec3 n1, vec3 n2)
{
	vec3 t = vec3(n1.xy, 1.0);
	vec3 u = vec3(-n2.xy, 1.0);
	float q = dot(t, t);
	float s = sqrt(q);

	if (fract(iTime * 0.6) < 0.5)
	{
		// Least-squares fit of sqrt(x) using (1-a)x + a, over the range [1, 5]

		s = 0.336 * q + 0.664;
	}

	t.z += s;
	vec3 r = t * dot(t, u) - u * (q + s);
    r.z = max(r.z, 0.0);
	return normalize(r);
}

// Function 54
vec3 blend_screen(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*(1.0-(1.0-c1)*(1.0-c2)) + (1.0-opacity)*c2;
}

// Function 55
vec4 alphaBlend(vec4 dest, vec4 source) {
    vec3 blended = (source.rgb * source.a) + (dest.rgb * (1.0 - source.a));
    return vec4(blended, 1.0);
}

// Function 56
vec4 blend(vec4 bg, vec4 fg) {
    vec4 c = vec4(0.);
    c.a = 1.0 - (1.0 - fg.a) * (1.0 - bg.a);
    if(c.a < .00000) return c;
    
    c.r = fg.r * fg.a / c.a + bg.r * bg.a * (1.0 - fg.a) / c.a;
    c.g = fg.g * fg.a / c.a + bg.g * bg.a * (1.0 - fg.a) / c.a;
    c.b = fg.b * fg.a / c.a + bg.b * bg.a * (1.0 - fg.a) / c.a;
    
    return c;
}

// Function 57
float blendingFunction(float t) {
    float f = 1.5 - abs(1.5 - t * 3.0);
    f = clamp(f, 0.0, 1.0);
    f = 1.0 - f;
    f *= f;
    f *= f;
    f = 1.0 - f;
    return f;
}

// Function 58
vec4 BlendFTB(vec4 frontPremul, vec4 backRGBA)
{
    vec4 res;
    
    res.rgb = backRGBA.rgb * (backRGBA.a * (1.0 - frontPremul.a)) + frontPremul.rgb;
    res.a = 1.0 - ((1.0 - backRGBA.a) * (1.0 - frontPremul.a));
    
    return res;
}

// Function 59
vec2 blend(vec2 old, vec2 new, float i){return (1.-(1./i))*old + (1./i) * new;}

// Function 60
float mixColors(float r, float v, float z){
    return clamp(0.5 + 0.5 * (v-r) / z, 0., 1.); 
}

// Function 61
float mix_360(float a, float b, float x) {
    float forward_dist = b-a;
    float rev_dist = 360.0+a-b;
    if (forward_dist > rev_dist) {
    	return mod(mix(360.0+a, b, x), 360.0);
    }
    return mix(a, b, x);
}

// Function 62
vec4 blendOnto(vec4 cFront, vec3 cBehind) {
    return cFront + (1.0 - cFront.a)*vec4(cBehind, 1.0);
}

// Function 63
vec3 blend_normal(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*c1 + (1.0-opacity)*c2;
}

// Function 64
vec3 blend_darken_o5268 (vec3 c1, vec3 c2, float opacity) {
	//return min(c1, c2);
	return opacity*min(c1, c2) + (1.0-opacity)*c2;
}

// Function 65
d2 mixd(d  a,d2 b,v0 c){return mixd(d0td2(a),b,c);}

// Function 66
d2 mixd(d2 a,d  b,v0 c){return mixd(a,d0td2(b),c);}

// Function 67
float hashmix(float x0, float x1, float interp)
{
	x0 = hash(x0);
	x1 = hash(x1);
	#ifdef noise_use_smoothstep
	interp = smoothstep(0.0,1.0,interp);
	#endif
	return mix(x0,x1,interp);
}

// Function 68
vec3 TriPlanarBlendWeightsConstantOverlap(vec3 normal) {

	vec3 blend_weights = normal*normal;//or abs(normal) for linear falloff(and adjust BlendZone)
	float maxBlend = max(blend_weights.x, max(blend_weights.y, blend_weights.z));
 	
    float BlendZone = 0.8f;
	blend_weights = blend_weights - maxBlend*BlendZone;

	blend_weights = max(blend_weights, 0.0);   

	float rcpBlend = 1.0 / (blend_weights.x + blend_weights.y + blend_weights.z);
	return blend_weights*rcpBlend;
}

// Function 69
vec3 blend(vec3 a, vec3 b) {
 
    #if BLEND_MODE == 1
    return 1. - ((1. - a) * (1. - b));
    #elif BLEND_MODE == 0
    return a * b;
    #elif BLEND_MODE == 2
    if(a.x < .5 && a.y < .5 && a.z < .5){
        return 2. * a * b;
    }
    return 1. - 2.*(1.-a)*(1. - b);
    #endif
    
    return a;
    
}

// Function 70
vec4 blend_o5265 (vec4 c1, vec4 c2, float amount) {
	if (c1.a > 0.0 && c2.a > 0.0) return master_blend_o5265 (c1, c2, amount);
	else if (c1.a > 0.0) return c1;
	else return c2;
}

// Function 71
void mixColorLine(vec2 uv,inout vec3 col,vec2 lineA,vec2 lineB,float scale)
{
    col = mix(
        col , 
        hash3point(lineA+lineB) ,
        1.0 - smoothstep(0.0,1.0,sqrt(sqrt( segment(uv,lineA,lineB).x * scale )))
    );
}

// Function 72
vec2 blend(vec2 c, float s)
{
    c.x *= iResolution.x / iResolution.y;
    vec2 pos = smoothstep(vec2(s), c/2.0, c/3.0*sin(c));
    float i = abs(sqrt(s));
    vec2 x = pos / i;
    return x - sin(i) / cos(i) / tan(i);
}

// Function 73
float boundblenduni(float f1, float f2, float f3, float a0, float a1, float a2, float a3)
{
  float r1 = f1*f1/(a1*a1)+f2*f2/(a2*a2);
  float r2 = 0.0;
  if (f3 > 0.0) r2 = f3*f3/(a3*a3);
  float rr = 0.0;
  if (r1 > 0.0) rr = r1/(r1+r2);
  float d = 0.0;
  if (rr < 1.0) d = a0*(1.0-rr)*(1.0-rr)*(1.0-rr)/(1.0+rr);
  return f1 + f2 + sqrt(f1*f1 + f2*f2) + d;
}

// Function 74
vec3 NormalBlend_UnpackedRNM(vec3 n1, vec3 n2)
{
	n1 += vec3(0, 0, 1);
	n2 *= vec3(-1, -1, 1);
	
    return n1*dot(n1, n2)/n1.z - n2;
}

// Function 75
void paint_blend_domain(vec3 color, float stepsize) {
    vec2 u = get_origin();
    float d = max(abs(u.x),abs(u.y));
    if ((d > 0.5) || (d < 0.25))
        return;
    
    vec2 m = mod(u, stepsize);
    vec2 ml = mod(u, stepsize * 2.0);
    vec2 ul = u - ml;
    u = u - m;

    set_source_rgba(vec4(color, 0.5));
    rectangle(u.x, u.y, stepsize, stepsize);
    stroke();
    
    vec2 g = ul + stepsize;
    g.x = (g.x > stepsize)?1.0:((g.x < -stepsize)?-1.0:0.0);
    g.y = (g.y > stepsize)?1.0:((g.y < -stepsize)?-1.0:0.0);
    
    mat3 ql = sample_map(ul + stepsize, stepsize * 2.0);
    mat3 q = sample_map(u + stepsize * 0.5, stepsize);
    m = m / stepsize;
    ml = ml / (stepsize * 2.0);
    
    // sample low-res patch as 4 sub-patches
    mat3 km00 = mat_subdivide(ql);    
    ql = flip_mat_x(ql);
    mat3 km01 = mat_subdivide(ql);
    km01 = flip_mat_x(km01);
    ql = flip_mat_y(ql);
    mat3 km11 = mat_subdivide(ql);
    ql = flip_mat_x(ql);
    km11 = flip_mat_x(km11);
    km11 = flip_mat_y(km11);
    mat3 km10 = mat_subdivide(ql);
    km10 = flip_mat_y(km10);

    // replace interior sub-patches with the hi-res version
    vec2 sp = ul + stepsize * 0.5;
    if (g.x < 0.0) {
        if (g.y > 0.0) {
        	km01 = sample_map(sp + stepsize * vec2(1.0, 0.0), stepsize);
        } else if (g.y < 0.0) {
            km11 = sample_map(sp + stepsize * vec2(1.0, 1.0), stepsize);
        } else {
        	km01 = sample_map(sp + stepsize * vec2(1.0, 0.0), stepsize);
            km11 = sample_map(sp + stepsize * vec2(1.0, 1.0), stepsize);
        }
    } else if (g.x > 0.0) {
        if (g.y > 0.0) {
        	km00 = sample_map(sp + stepsize * vec2(0.0, 0.0), stepsize);
        } else if (g.y < 0.0) {
            km10 = sample_map(sp + stepsize * vec2(0.0, 1.0), stepsize);
        } else {
        	km00 = sample_map(sp + stepsize * vec2(0.0, 0.0), stepsize);
            km10 = sample_map(sp + stepsize * vec2(0.0, 1.0), stepsize);
        }
    } else {
        if (g.y > 0.0) {
        	km00 = sample_map(sp + stepsize * vec2(0.0, 0.0), stepsize);
        	km01 = sample_map(sp + stepsize * vec2(1.0, 0.0), stepsize);
        } else if (g.y < 0.0) {
        	km10 = sample_map(sp + stepsize * vec2(0.0, 1.0), stepsize);
        	km11 = sample_map(sp + stepsize * vec2(1.0, 1.0), stepsize);
        }
    }
    
    // stitch the four patches together
    
    // average the center control point
    float c22 = (km00[1][1] + km01[1][1] + km10[1][1] + km11[1][1]) / 4.0;
    // average cross control points
    float c21 = (km00[1][1] + km10[1][1]) / 2.0;
    float c23 = (km01[1][1] + km11[1][1]) / 2.0;
    float c12 = (km00[1][1] + km01[1][1]) / 2.0;
    float c32 = (km10[1][1] + km11[1][1]) / 2.0;
    // average cross tip control points
    float c20 = (km00[1][0] + km10[1][0]) / 2.0;
    float c02 = (km00[0][1] + km01[0][1]) / 2.0;
    float c24 = (km01[1][2] + km11[1][2]) / 2.0;
    float c42 = (km10[2][1] + km11[2][1]) / 2.0;
    
    km00[2][2] = c22;
    km01[2][0] = c22;
    km10[0][2] = c22;
    km11[0][0] = c22;
    
    km00[2][1] = c21;
    km10[0][1] = c21;
    km01[2][1] = c23;
    km11[0][1] = c23;

    km00[1][2] = c12;
    km01[1][0] = c12;
    km10[1][2] = c32;
    km11[1][0] = c32;
    
    km00[2][0] = c20;
    km10[0][0] = c20;
    km00[0][2] = c02;
    km01[0][0] = c02;
    
    km01[2][2] = c24;
    km11[0][2] = c24;
    km10[2][2] = c42;
    km11[2][0] = c42;
    
    mat3 kout;
    if (ml.x < 0.5) {
        if (ml.y < 0.5) {
            kout = km00;
        } else {
            kout = km10;
        }
    } else {
        if (ml.y < 0.5) {
            kout = km01;
        } else {
            kout = km11;
        }
    }
    
    float k = mat_sample(kout, m);

    #if 0
    set_source_rgba(vec4(g*0.5 + 0.5, 0.0, 0.8));
    rectangle(u.x, u.y, stepsize, stepsize);
    fill();
    #endif
    
    k = clamp((0.008 - abs(k)) * 200.0, 0.0, 1.0);
    set_source_rgba(mix(vec4(color,0.0), vec4(color,1.0), k));
    
    rectangle(u.x, u.y, stepsize, stepsize);
    fill();
    
}

// Function 76
float SpinBlend(float y, float progress) {
    return blend(y, 1.5, progress);
}

// Function 77
vec3 NormalBlend_Whiteout(vec3 n1, vec3 n2)
{
    // Unpack
	n1 = n1*2.0 - 1.0;
    n2 = n2*2.0 - 1.0;
    
	return normalize(vec3(n1.xy + n2.xy, n1.z*n2.z));    
}

// Function 78
GNum2 a_mix(in GNum2 a, in float b, in GNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

// Function 79
vec3 blend(in vec3 src, in vec3 dst, in int mode)
{
    //if (mode == )  return source(src, dst);
    //if (mode == )  return dest(src, dst);
    if (mode == 0)  return screen(src, dst);
    if (mode == 1)  return multiply(src, dst);
    if (mode == 2)  return overlay(src, dst);
    if (mode == 3)  return hardlight(src, dst);
    if (mode == 4)  return softlight(src, dst);
    if (mode == 5)  return colorDodge(src, dst);
    if (mode == 6)  return colorBurn(src, dst);
    if (mode == 7)  return linearDodge(src, dst);
    if (mode == 8)  return linearBurn(src, dst);
    if (mode == 9)  return vividLight(src, dst);
    if (mode == 10) return linearLight(src, dst);
    if (mode == 11) return pinLight(src, dst);
    if (mode == 12) return hardMix(src, dst);
    if (mode == 13) return subtract(src, dst);
    if (mode == 14) return divide(src, dst);
    if (mode == 15) return addition(src, dst);
    if (mode == 16) return difference(src, dst);
    if (mode == 17) return darken(src, dst);
    if (mode == 18) return lighten(src, dst);
    if (mode == 19) return invert(src, dst);
    if (mode == 20) return invertRGB(src, dst);
    if (mode == 21) return hue(src, dst);
    if (mode == 22) return saturation(src, dst);
    if (mode == 23) return color(src, dst);
    if (mode == 24) return luminosity(src, dst);
    if (mode == 25) return exclusion(src, dst);
    return vec3(0.0,0.0,0.0);
}

// Function 80
vec3 blend_hard_light(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*0.5*(c1*c2+blend_overlay(uv, c1, c2, 1.0)) + (1.0-opacity)*c2;
}

// Function 81
SDFRes blendSDF(SDFRes f1, SDFRes f2, float k)
{
    // Branching a lot :( Needs more work
    
   	SDFRes closest  = f1;
    SDFRes furthest = f2;
	
	float diff = float(f1.d > f2.d);
    
	closest.d =  mix(f1.d, f2.d,   diff);
	closest.m =  mix(f1.m, f2.m,   diff);
	closest.m2 = mix(f1.m2, f2.m2, diff);
	closest.b =  mix(f1.b, f2.b,   diff);
	
	furthest.d =  mix(f2.d, f1.d,   diff);
	furthest.m =  mix(f2.m, f1.m,   diff);
	furthest.m2 = mix(f2.m2, f1.m2, diff);
	furthest.b =  mix(f2.b, f1.b,   diff);
    
	// Dominant materials
	float mf1 = mix(closest.m2, closest.m, float(closest.b < 0.5));
	float mf2 = mix(furthest.m2, furthest.m, float(furthest.b < 0.5));
    
    // New distance
    float t  = smin(f1.d, f2.d, k);
    
    // New blend
    float bnew = getBlend(f1.d, f2.d);
    float b = max(closest.b, bnew);
    float bhigher = float(b > bnew);

	float m  = mix(mf1, closest.m,  bhigher);
	float m2 = mix(mf2, closest.m2, bhigher);
    
    return SDFRes(t, m, m2, b);
}

// Function 82
vec2
mix_op_( in vec2 p0, in vec2 p1, vec2 max_dist )
{
    bool do_mix = p0.x > 0.0 && p0.x < max_dist.x &&
                  p1.x > 0.0 && p1.x < max_dist.y;
    if ( !do_mix && p0.x < p1.x )
    {
        return p0;
    }
    if ( !do_mix && p1.x < p0.x )
    {
        return p1;
    }
    vec2 h_max_dist = max_dist * 0.5;
    p0.x = (p0.x-h_max_dist.x)/h_max_dist.x;
    p1.x = (p1.x-h_max_dist.y)/h_max_dist.y;
    float v = mix( p0.x, p1.x, 0.5 );
    vec2 ret = vec2(min(max_dist) * v, p0.y );

    return ret;
}

// Function 83
float hashmix(vec4 p0, vec4 p1, vec4 interp)
{
	float v0 = hashmix(p0.xyz+vec3(p0.w*17.0,0.0,0.0),p1.xyz+vec3(p0.w*17.0,0.0,0.0),interp.xyz);
	float v1 = hashmix(p0.xyz+vec3(p1.w*17.0,0.0,0.0),p1.xyz+vec3(p1.w*17.0,0.0,0.0),interp.xyz);
	#ifdef noise_use_smoothstep
	interp = smoothstep(vec4(0.0),vec4(1.0),interp);
	#endif
	return mix(v0,v1,interp[3]);
}

// Function 84
vec3 blend_darken_o5265 (vec3 c1, vec3 c2, float opacity) {
	return opacity*min(c1, c2) + (1.0-opacity)*c2;
}

// Function 85
vec4 blend_Shift()
{
    float phaseLim = phase * 1.1;
    return interpolateColor(phaseLim, 0.02, uv.x);
}

// Function 86
vec4 master_blend_o5265 (vec4 c1, vec4 c2, float amount){
	return vec4(blend_darken_o5265(c1.rgb, c2.rgb, amount*c1.a),
	            min(1.0, c2.a+amount*c1.a));
}

// Function 87
vec3 hardMixMode (vec3 colorA, vec3 colorB)
{
    float r = hardMixFloat(colorA.r,colorB.r);
    float g = hardMixFloat(colorA.g,colorB.g);
    float b = hardMixFloat(colorA.b,colorB.b);
    return vec3(r,g,b);
}

// Function 88
vec3 blend_multiply(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*c1*c2 + (1.0-opacity)*c2;
}

// Function 89
float opBlend(float d1, float d2, float a)
{
    if (a > 1.0)
        a = 1.0;
    if (a < 0.0)
        a = 0.0;
    return a * d1 + (1.0 - a) * d2;
}

// Function 90
vec2
mix_op_p2( in vec2 p0, in vec2 p1, float max_dist )
{
    bool do_mix = p0.x > 0.0 && p0.x < max_dist &&
                  p1.x > 0.0 && p1.x < max_dist;
    if ( !do_mix && p0.x < p1.x )
    {
        return p0;
    }
    if ( !do_mix && p1.x < p0.x )
    {
        return p1;
    }
    vec2 d;
    d.x = p1.x - pow2( ( max_dist - p0.x ) / max_dist ) * max_dist;
    d.y = p0.x - pow2( ( max_dist - p1.x ) / max_dist ) * max_dist;
    return vec2((d.x+d.y)*0.5,p0.y);
    //return vec2(min(d.x,d.y),p0.y);
}

// Function 91
vec3 blendZcam(vec2 uv, vec3 lhsRgb, vec3 rhsRgb) {
    ZcamViewingConditions cond = getZcamCond();

    Zcam lhs = xyzToZcam(linearSrgbToXyz(srgbTransferInv(lhsRgb)) * cond.whiteLuminance, cond);
    Zcam rhs = xyzToZcam(linearSrgbToXyz(srgbTransferInv(rhsRgb)) * cond.whiteLuminance, cond);

    vec3 lhsJch = vec3(lhs.lightness, lhs.chroma, lhs.hueAngle);
    vec3 rhsJch = vec3(rhs.lightness, rhs.chroma, lhs.hueAngle);
    return clipZcamJchToLinearSrgb(mix(lhsJch, rhsJch, uv.x), cond);
}

// Function 92
vec3 spriteMix(vec3 a, vec3 b) {
    if (b.x == -1.0) {
        return a;
    }
    return b;
}

// Function 93
vec4 master_blend_o5229 (vec4 c1, vec4 c2, float amount){
	return vec4(blend_darken_o5229(c1.rgb, c2.rgb, c1.a*amount),
	            min(1.0, c2.a+amount*c1.a));
}

// Function 94
vec3 blendSrgb(vec2 uv, vec3 lhsRgb, vec3 rhsRgb) {
    return mix(lhsRgb, rhsRgb, uv.x);
}

// Function 95
float GetFontBlend( PrintState state, LayoutStyle style, float size )
{
    float fFeatherDist = 1.0f * length(state.vPixelSize / style.vSize);    
    float f = clamp( (size-state.fDistance + fFeatherDist * 0.5f) / fFeatherDist, 0.0, 1.0);
    return f;
}

// Function 96
float blend(float y, float blend, float progress) {
    float a = (y / modelSize) + .5;
    a -= progress * (1. + blend) - blend * .5;
    a += blend / 2.;
    a /= blend;
    a = clamp(a, 0., 1.);
    a = smoothstep(0., 1., a);
    a = smoothstep(0., 1., a);
    return a;
}

// Function 97
void Blend(inout Shape current, inout float currentD,
           Shape candidate, float candidateD)
{ // Based on Iñigo Quilez's smooth min algorithm:
  // iquilezles.org/www/articles/smin/smin.htm
    float h = clamp(.5+.5*(candidateD-currentD)/candidate.blendStrength,
                    .0, 1.);
    
    currentD       = mix(candidateD, currentD, h) -
                         candidate.blendStrength * h * (1.- h);
    current.color  = mix(candidate.color, current.color, h);
    current.normal = mix(candidate.normal, current.normal, h);
    
    // TODO: Find a better way to interpolate the texture / patterns
    current.type   = (h>=.5) ? current.type   : candidate.type;
	// TODO: Find a better way to interpolate glossiness
    current.glossy = (h>=.5) ? current.glossy : candidate.glossy;
}

// Function 98
vec3 NormalBlend_RNM(vec3 n1, vec3 n2)
{
    // Unpack (see article on why it's not just n*2-1)
	n1 = n1*vec3( 2,  2, 2) + vec3(-1, -1,  0);
    n2 = n2*vec3(-2, -2, 2) + vec3( 1,  1, -1);
    
    // Blend
    return n1*dot(n1, n2)/n1.z - n2;
}

// Function 99
float screenBlend(vec2 va1, vec2 va2){ 
    return 1. - (1. - va1.x*va1.y)*(1. - va2.x*va2.y);
}

// Function 100
void
blend_pma( inout vec4 dst, in vec4 src )
{
    dst.rgb = src.rgb + dst.rgb * ( 1.0 - src.a );
    dst.a = min( 1.0, dst.a + src.a );
}

// Function 101
v1 mixd(v0 a,v1 b,v0 c){return mix(v0tv1(a),b,c);}

// Function 102
float hardMixFloat (float rgbA,float rgbB)
{
    if(rgbA<1.0-rgbB)
    {
        return 0.0;
    }
    else
    {
        return 1.0;
    }
}

// Function 103
float blend_burn_f(float c1, float c2) {
	return (c1==0.0)?c1:max((1.0-((1.0-c2)/c1)),0.0);
}

// Function 104
v0 mixd(v0 a,v0 b,v0 c){return mix(a,b,c);}

// Function 105
vec2
mix_op( in vec2 p0, in vec2 p1, float max_dist )
{
    if ( mod( iTime, 2.0 ) < 1.0 )
    {
        return smin_op(p0,p1,max_dist);
    } else
    {
    	return mix_op_9avg(p0,p1,max_dist);
    }
#if 0
	return mix_op_p2(p0,p1,max_dist);
#endif
}

// Function 106
vec3 NormalBlend_Overlay(vec3 n1, vec3 n2)
{
    vec3 n;
    n.x = overlay(n1.x, n2.x);
    n.y = overlay(n1.y, n2.y);
    n.z = overlay(n1.z, n2.z);

    return normalize(n*2.0 - 1.0);
}

// Function 107
vec3 blendOklch(vec2 uv, vec3 lhsRgb, vec3 rhsRgb) {
    vec3 lhs = labToLch(linearSrgbToOklab(srgbTransferInv(lhsRgb)));
    vec3 rhs = labToLch(linearSrgbToOklab(srgbTransferInv(rhsRgb)));
    rhs.z = lhs.z;
    return clipOklchToLinearSrgb(mix(lhs, rhs, uv.x));
}

// Function 108
float BaseMix(vec2 u){return max(long2(u),max(BaseMainOdd(u),BaseMain(u)));}

// Function 109
GNum2 a_mix(in GNum2 a, in GNum2 b, in GNum2 t)
{
    return add(mult(a, sub(1.0, t)), mult(b, t));
}

// Function 110
vec3 NormalBlend_PartialDerivatives(vec3 n1, vec3 n2)
{	
    // Unpack
	n1 = n1*2.0 - 1.0;
    n2 = n2*2.0 - 1.0;
    
    return normalize(vec3(n1.xy*n2.z + n2.xy*n1.z, n1.z*n2.z));
}

// Function 111
float opBlend( float d1, float d2 ) {
    const float k = 0.1;
    float h = clamp( 0.5+0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}

// Function 112
float mixP(float f0, float f1, float a)
{
    return mix(f0, f1, a*a*(3.0-2.0*a));
}

// Function 113
vec3 FresnelBlenderWithRoughness(vec3 R0, vec3 R90, vec2 envBRDF) {
    return clamp(envBRDF.y * R90 + envBRDF.x * R0, vec3(0.0), vec3(1.0));
}

// Function 114
vec4 blend_Circle()
{
    vec2 texVec;
    texVec.x = (uv.x - 0.5) * 2.0;
    texVec.y = (uv.y - 0.5 / aspectRatio) * 2.0;

    float phaseLim = phase * 1.5;
    float texVecLength = length(texVec);

    return interpolateColor(phaseLim, 0.02, texVecLength);
}

// Function 115
vec3 blend_dissolve(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	if (rand(uv) < opacity) {
		return c1;
	} else {
		return c2;
	}
}

// Function 116
void blend(inout vec4 fragColor, in vec4 color) {
    vec3 rgb = mix(fragColor.rgb, color.rgb, color.a);
    float a = max(fragColor.a, color.a);
    fragColor = vec4(rgb, a);
}

// Function 117
vec3 blend_soft_light(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*vec3(blend_soft_light_f(c1.x, c2.x), blend_soft_light_f(c1.y, c2.y), blend_soft_light_f(c1.z, c2.z)) + (1.0-opacity)*c2;
}

// Function 118
vec2
mix_op_( in vec2 p0, in vec2 p1, float max_dist )
{
    bool do_mix = p0.x > 0.0 && p0.x < max_dist &&
                  p1.x > 0.0 && p1.x < max_dist;
    if ( !do_mix && p0.x < p1.x )
    {
        return p0;
    }
    if ( !do_mix && p1.x < p0.x )
    {
        return p1;
    }
    float h_max_dist = max_dist * 0.5;
    p0.x = (p0.x-h_max_dist)/h_max_dist;
    p1.x = (p1.x-h_max_dist)/h_max_dist;
    float p02 = p0.x*p0.x;
    float p12 = p1.x*p1.x;
    float v = ( p0.x + p1.x )*0.5;
    vec2 ret = vec2(max_dist * v, p0.y );

    return ret;
}

// Function 119
float mix_min(float x, float y, float z) {
	return min(min(x, y), z);
}

// Function 120
vec4 blend_ShiftFancy()
{
    vec4 textureColor;
    float w = 0.2;
    float phaseLim = (phase - 0.2) * 1.4;
    float phaseLim1 = phaseLim - w;
    float phaseLim2 = phaseLim + w;

    if (uv.x < phaseLim1)
    {
        textureColor = textureColor0;
    }
    else
    if (uv.x > phaseLim2)
    {
        textureColor = textureColor1;
    }
    else
    {
        textureColor = blend_Halftone2(phaseLim1, phaseLim2);
    }
    return textureColor;
}

// Function 121
d  mixd(d  a,d  b,v0 c){return d (mix(a.x,b.x,c),mix(a.d,b.d,c));}

// Function 122
vec3 blend_darken(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*min(c1, c2) + (1.0-opacity)*c2;
}

// Function 123
vec3 mixColorLine(vec2 uv,vec3 currentCol,vec3 colLine,vec2 lineA,vec2 lineB,float scale)
{
    return mix(
        currentCol , 
        colLine ,
        1.0 - smoothstep(0.0,1.0,sqrt(sqrt( segment(uv,lineA,lineB).x * scale )))
    );
}

// Function 124
float getBlend(float d1, float d2)
{
    float diff = -abs(d1 - d2);
    float blend = diff / BLEND_SIZE;
    blend = saturate((blend + 1.0) * 0.5);
    return blend;
}

// Function 125
void blend(inout vec4 fragColor, in vec2 fragCoord, in sampler2D channel) {
    vec2 uv = fragCoord.xy/iResolution.xy;
    blend(fragColor, texture(channel, uv));
}

// Function 126
vec4 blend(vec4 a, vec4 b)
{
	return mix(a, b, b.a);   
}

// Function 127
float blend_overlay_f(float c1, float c2) {
	return (c1 < 0.5) ? (2.0*c1*c2) : (1.0-2.0*(1.0-c1)*(1.0-c2));
}

// Function 128
vec2
mix_op_( in vec2 p0, in vec2 p1, float max_dist, float m0 )
{
    bool do_mix = p0.x > 0.0 && p0.x < max_dist &&
                  p1.x > 0.0 && p1.x < max_dist;
    if ( !do_mix && p0.x < p1.x )
    {
        return p0;
    }
    if ( !do_mix && p1.x < p0.x )
    {
        return p1;
    }
    float h_max_dist = max_dist * 0.5;
    p0.x = (p0.x-h_max_dist)/h_max_dist;
    p1.x = (p1.x-h_max_dist)/h_max_dist;
    float v = mix( p0.x, p1.x, m0 );
    vec2 ret = vec2(max_dist * v, p0.y );

    return ret;
}

// Function 129
float blend(float a, float b, float k) {
    float h = clamp(mix(1.0, (b-a)/k, 0.5), 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}

// Function 130
float hashmix(vec3 p0, vec3 p1, vec3 interp)
{
	float v0 = hashmix(p0.xy+vec2(p0.z*43.0,0.0),p1.xy+vec2(p0.z*43.0,0.0),interp.xy);
	float v1 = hashmix(p0.xy+vec2(p1.z*43.0,0.0),p1.xy+vec2(p1.z*43.0,0.0),interp.xy);
	#ifdef noise_use_smoothstep
	interp = smoothstep(vec3(0.0),vec3(1.0),interp);
	#endif
	return mix(v0,v1,interp[2]);
}

// Function 131
vec3 hardMix(in vec3 src, in vec3 dst)
{
    return step(1.0, src + dst);
}

// Function 132
float mix_angle( float angle, float target, float rate )
{    

   	angle = abs( angle - target - 1. ) < abs( angle - target ) ? angle - 1. : angle;
   	angle = abs( angle - target + 1. ) < abs( angle - target ) ? angle + 1. : angle;
	angle = fract(mix(angle, target, rate));   	
   	return bound(angle);
}

// Function 133
void blend(inout vec3 color, vec2 p, vec2 s, float ph) {
    float r = min(s.x, s.y);
    float d = -sdBox(vec3(p,0.0), vec3(s-r,1.0))+r;
    
    d /= r * (0.1 + (sin(iTime)*0.5+0.5)*0.9);
    
    d = smoothstep(0.0, 1.0, d);
    //d = clamp(d, 0.0, 1.0);
    
    // defined in srgb
    vec3 albedo = min(hue2rgb(iTime * 0.01 + ph) + 0.1, 1.0);
    albedo = mix(albedo,vec3(1.0),0.5);
    albedo = srgb2lin(albedo);
    
    if (p.x < m) {
        float eps = 0.001;
        color = pow(color, vec3(1.0 - d)) * pow(albedo, vec3(d));
    } else {
        color = mix(color, albedo, d);
    }
}

// Function 134
v1 mixd(v1 a,v1 b,v0 c){return mix(a,b,c);}

// Function 135
vec3 hardMix( vec3 s, vec3 d )
{
	return floor(s + d);
}

// Function 136
vec3 blendZcam(vec2 uv, vec3 lhsRgb, vec3 rhsRgb) {
    ZcamViewingConditions cond = getZcamCond();

    Zcam lhs = xyzToZcam(linearSrgbToXyz(srgbTransferInv(lhsRgb)) * cond.whiteLuminance, cond);
    Zcam rhs = xyzToZcam(linearSrgbToXyz(srgbTransferInv(rhsRgb)) * cond.whiteLuminance, cond);

    vec3 lhsJch = vec3(lhs.lightness, lhs.chroma, lhs.hueAngle);
    vec3 rhsJch = vec3(rhs.lightness, rhs.chroma, lhs.hueAngle);
    return zcamJchToLinearSrgb(mix(lhsJch, rhsJch, uv.x), cond);
}

// Function 137
vec3 blend3(vec3 x)
{
    vec3 y = 1. - x * x; //Bump function
    y = max(y, vec3(0));
    return y;
}

// Function 138
float opBlend( float p1, float p2 )
{
    float d1 = p1;
    float d2 = p2;
    return smin( d1, d2, 2.0 );
}

// Function 139
void set_source_blend_mode(int mode) {
    _stack.source_blend = mode;
}

// Function 140
float hashmix(vec2 p0, vec2 p1, vec2 interp)
{
	float v0 = hashmix(p0[0]+p0[1]*128.0,p1[0]+p0[1]*128.0,interp[0]);
	float v1 = hashmix(p0[0]+p1[1]*128.0,p1[0]+p1[1]*128.0,interp[0]);
	#ifdef noise_use_smoothstep
	interp = smoothstep(vec2(0.0),vec2(1.0),interp);
	#endif
	return mix(v0,v1,interp[1]);
}

// Function 141
float mix3d_mul(float x, float y, float z) {
	return x*y*z;
}

// Function 142
void mixColorPoint(vec2 uv,inout vec3 col,vec2 colPoint,float scale)
{
    col = mix(
        col , 
        hash3point(colPoint) ,
        1.0 - smoothstep(0.0,1.0,sqrt(sqrt( length(uv - colPoint)* scale )))
    );
}

// Function 143
vec4 mix(vec4 a, vec4 b, float amt)
{
    return ((1.0 - amt) * a) + (b * amt);
}

// Function 144
float mix3d_pow(float x, float y, float z) {
	return pow(pow(x, y), z);
}

// Function 145
vec3 blend_lighten(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*max(c1, c2) + (1.0-opacity)*c2;
}

// Function 146
v2 mixd(v2 a,v2 b,v0 c){return mix(a,b,c);}

// Function 147
d2 mixd(d2 a,d1 b,v0 c){return mixd(a,d1td2(b),c);}

// Function 148
vec2 VXAADifferentialBlendWeight( vec4 n[4] )
{
    float diffWE = VXAALumaDiff( n[ VXAA_W ].rgb, n[ VXAA_E ].rgb );
    float diffNS = VXAALumaDiff( n[ VXAA_N ].rgb, n[ VXAA_S ].rgb );
    return diffWE < diffNS ? vec2( 0.5, 0.0 ) : vec2( 0.0, 0.5 );
}

// Function 149
vec2 mix_custom(vec2 edge0, vec2 edge1, float x) {
    return mix(edge0, edge1, smoothstp(x));
}

// Function 150
void mixColorPoint(vec2 uv,inout vec3 col,vec2 colPoint,float scale)
{
    col = mix(
        col , 
        hash3point(colPoint) ,
        coeffDistPoint(uv,colPoint,scale)
    );
}

// Function 151
vec3 smoothBlend(in vec3 point, in vec3 about, in float radius) {
    point -= about;
    point = mix(-point, point, smoothstep(-radius, radius, point));
    return point + about;
}

// Function 152
vec3 blend(vec4 texture1, float a1, vec4 texture2, float a2, vec4 texture3, float a3)
{
    float d1 = make_depthmap(texture1.rgb);
    float d2 = make_depthmap(texture2.rgb);
    float d3 = make_depthmap(texture3.rgb);
	float ma = max(max(d1 + a1, d2 + a2),d3 + a3) - 0.6;
	float b1 = max(d1 + a1 - ma, 0.0);
	float b2 = max(d2 + a2 - ma, 0.0);
	float b3 = max(d3 + a3 - ma, 0.0);
	return (texture1.rgb * b1 + texture2.rgb * b2 + texture3.rgb * b3) / (b1 + b2 + b3);
}

// Function 153
vec3 blend( vec3 s, vec3 d, int id )
{
	if(id==0)	return darken(s,d);
	if(id==1)	return multiply(s,d);
	if(id==2)	return colorBurn(s,d);
	if(id==3)	return linearBurn(s,d);
	if(id==4)	return darkerColor(s,d);
	
	if(id==5)	return lighten(s,d);
	if(id==6)	return screen(s,d);
	if(id==7)	return colorDodge(s,d);
	if(id==8)	return linearDodge(s,d);
	if(id==9)	return lighterColor(s,d);
	
	if(id==10)	return overlay(s,d);
	if(id==11)	return softLight(s,d);
	if(id==12)	return hardLight(s,d);
	if(id==13)	return vividLight(s,d);
	if(id==14)	return linearLight(s,d);
	if(id==15)	return pinLight(s,d);
	if(id==16)	return hardMix(s,d);
	
	if(id==17)	return difference(s,d);
	if(id==18)	return exclusion(s,d);
	if(id==19)	return subtract(s,d);
	if(id==20)	return divide(s,d);
	
	if(id==21)	return hue(s,d);
	if(id==22)	return color(s,d);
	if(id==23)	return saturation(s,d);
	if(id==24)	return luminosity(s,d);
    
    return vec3(0.0);
}

// Function 154
d1 mixd(d  a,d1 b,v0 c){return mixd(d0td1(a),b,c);}

// Function 155
vec4 mix_custom(vec4 edge0, vec4 edge1, float x) {
    return mix(edge0, edge1, smoothstp(x));
}

// Function 156
vec4 evaluate_blend_operation(int blend_state,vec4 color, vec4 src, vec4 dst)
{
    if(blend_state == BLEND_ONE)
        return color;
    else if(blend_state == BLEND_SRC_ALPHA)
        return color * src.a;
    else if(blend_state == BLEND_ONE_MINUS_SRC_ALPHA)
        return color * (1.0 - src.a);
        
    return vec4(-1.0);//invalid state
}

// Function 157
vec3 blend_difference(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*clamp(c2-c1, vec3(0.0), vec3(1.0)) + (1.0-opacity)*c2;
}

// Function 158
vec3 BlendNormal(vec3 normal){
	vec3 blending = abs(normal);
	blending = normalize(max(blending, 0.00001));
	blending /= vec3(blending.x + blending.y + blending.z);
	return blending;
}

// Function 159
v3 mixd(v3 a,v3 b,v0 c){return mix(a,b,c);}

// Function 160
d2 mixd(d1 a,d2 b,v0 c){return mixd(d1td2(a),b,c);}

// Function 161
GNum2 a_mix(in GNum2 a, in GNum2 b, in float t)
{
    return add(mult(a, 1.0 - t), mult(b, t));
}

// Function 162
vec3 blendLinearSrgb(vec2 uv, vec3 lhsRgb, vec3 rhsRgb) {
    vec3 lhs = srgbTransferInv(lhsRgb);
    vec3 rhs = srgbTransferInv(rhsRgb);

    return srgbTransfer(mix(lhs, rhs, uv.x));
}

// Function 163
float mix_custom(float edge0, float edge1, float x) {
    return mix(edge0, edge1, smoothstp(x));
}

// Function 164
void mixColorLine(vec2 uv,inout vec3 col,vec2 lineA,vec2 lineB,float scale)
{
    col = mix(
        col , 
        vec3(0.0),//hash3point(lineA+lineB) ,
        1.0 - smoothstep(0.0,1.0,sqrt(sqrt( segment(uv,lineA,lineB).x * scale )))
    );
}

// Function 165
vec3 blend(vec3 a, vec3 b, float f, float gamma)
{
    vec3 g = vec3(gamma);
    return pow((1.0 - f) * pow(a, g) + f * pow(b, g), 1.0 / g);
}

// Function 166
vec4 blend(in vec4 under, in vec4 over) {
  vec4 result = mix(under, over, over.a);
  result.a = over.a + under.a * (1.0 - over.a);
    
  return result;
}

// Function 167
GNum2 a_mix(in float a, in GNum2 b, in float t)
{
    return add(a * (1.0 - t), mult(b, t));
}

