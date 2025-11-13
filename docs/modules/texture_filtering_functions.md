# texture_filtering_functions

**Category:** texturing
**Type:** extracted

## Dependencies
texture_sampling, normal_mapping, lighting, raymarching

## Tags
lighting, texturing, color

## Code
```glsl
// Reusable Texture Filtering Texturing Functions
// Automatically extracted from texturing/mapping-related shaders

// Function 1
vec3 SuperFastNormalFilter(sampler2D _tex,vec2 uv,float strength){
    float p00 = GetTextureLuminance(_tex,uv);
    return normalize(vec3(-dFdx(p00),-dFdy(p00),1.-strength));
}

// Function 2
vec3 GetReadMipMapUVW_Dir(vec3 _uvw,vec3 _axis){
    return normalize(
        ((_uvw+3.)*exp2(vec3(5.-10.,8.-10.,2.-10.))-1.)
        * (1.-abs(_axis)) 
        + _axis
    );
}

// Function 3
vec4 drawFilter(vec2 fragCoord) {
    fragCoord = abs(3.0*fragCoord);
    if (fragCoord.x > 1.5 || fragCoord.y > 1.5) {
        return vec4(0.0);
    }
    
    vec2 c = DQCenter(fragCoord);
    vec2 l = DQLobe(fragCoord);
    vec2 f = mix(c, l, greaterThan(fragCoord, vec2(0.5)));
    float o = f.x * f.y;
    return vec4(-20.0*o, o, 0.0, 1.0);
}

// Function 4
float FilterTriangle(vec2 p, vec2 radius)
{
    p /= radius; //TODO: fails at radius0
    return clamp(1.0f - length(p), 0.0, 1.0);
}

// Function 5
vec3 getPreFilteredColour(vec3 N, float roughness, int sampleCount){
    vec3 R = N;
    vec3 V = R;
    
    float totalWeight = 0.0;
    vec3 prefilteredColor = vec3(0.0);    
    
    //Generate sampleCount number of a low discrepancy random directions in the 
    //specular lobe and add the environment map data into a weighted sum.
    for(int i = ZERO; i < sampleCount; i++){
    
        vec2 randomHemisphere = hammersley(i, sampleCount);
        vec3 H  = importanceSampleGGX(randomHemisphere, N, roughness);
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = dot_c(N, L);
        if(NdotL > 0.0){
        
            float level = 0.0;
            
        #if ENV_FILTERING == 1
            // Sample the mip levels of the environment map
            // https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/
            // Vectors to evaluate pdf
            float NdotH = saturate(dot(N, H));
            float VdotH = saturate(dot(V, H));

            // Probability distribution function
            float pdf = D_GGX(NdotH, roughness*roughness) * NdotH / (4.0f * VdotH);

            // Solid angle represented by this sample
            float omegaS = 1.0 / (float(sampleCount) * pdf);
            // An arbitrary value from trial and error
            float envMapSize = 8.0;
            // Solid angle covered by 1 pixel
            float omegaP = 4.0 * PI / (6.0 * envMapSize * envMapSize);
            // Original paper suggests biasing the mip to improve the results
            float mipBias = 1.0;
            level = max(0.5 * log2(omegaS / omegaP) + mipBias, 0.0f);
        #endif
        
            prefilteredColor += getEnvironment(L, level) * NdotL;
            totalWeight      += NdotL;
        }
    }
    prefilteredColor = prefilteredColor / totalWeight;

    return prefilteredColor;
}

// Function 6
float FilterGaussian(vec2 p, vec2 radius )
{
    p /= radius; //TODO: fails at radius0
    
    #if defined( USE_RADIUS_VERSIONS )
    return Gaussian( length(p) );
    #else
	return Gaussian(p.x) * Gaussian(p.y);
    #endif
    
}

// Function 7
vec4 SampleMip2(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*16.,sp.z+32.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(16.,0.))*IRES),fract(sp.y));
}

// Function 8
vec4 SampleMip5(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*4.,sp.z+60.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(2.,0.))*IRES),fract(sp.y));
}

// Function 9
vec3 FastNormalFilter(sampler2D _tex,vec2 uv,float strength,vec2 offset){
	vec3 e = vec3(offset,0.);
    float p00 = GetTextureLuminance(_tex,uv);
    float p10 = GetTextureLuminance(_tex,uv + e.xz);
    float p01 = GetTextureLuminance(_tex,uv + e.zy);
    /* Orgin calculate 
    vec3 ab = vec3(1.,0.,p10-p00);
    vec3 ac = vec3(0.,1.,p01-p00);
    vec3 n = cross(ab,ac);
    n.z *= (1.-strength);
    return normalize(n);
	*/
	vec2 dir = p00-vec2(p10,p01);
    return normalize(vec3(dir,1.-strength));
}

// Function 10
void blindnessFilter( out vec4 myoutput, in vec4 myinput )
{
	if (blindnessType == PROTANOPIA) {
			vec3 opponentColor = RGBtoOpponentMat * vec3(myinput.r, myinput.g, myinput.b);
			opponentColor.x -= opponentColor.y * 1.5; // reds (y <= 0) become lighter, greens (y >= 0) become darker
			vec3 rgbColor = OpponentToRGBMat * opponentColor;
			myoutput = vec4(rgbColor.r, rgbColor.g, rgbColor.b, myinput.a);
	} else if (blindnessType == DEUTERANOPIA) {
			vec3 opponentColor = RGBtoOpponentMat * vec3(myinput.r, myinput.g, myinput.b);
			opponentColor.x -= opponentColor.y * 1.5; // reds (y <= 0) become lighter, greens (y >= 0) become darker
			vec3 rgbColor = OpponentToRGBMat * opponentColor;
			myoutput = vec4(rgbColor.r, rgbColor.g, rgbColor.b, myinput.a);
	} else if (blindnessType == TRITANOPIA) {
			vec3 opponentColor = RGBtoOpponentMat * vec3(myinput.r, myinput.g, myinput.b);
			opponentColor.x -= ((3.0 * opponentColor.z) - opponentColor.y) * 0.25;
			vec3 rgbColor = OpponentToRGBMat * opponentColor;
			myoutput = vec4(rgbColor.r, rgbColor.g, rgbColor.b, myinput.a);
    } else {
			myoutput = myinput;
	}	
}

// Function 11
float FilterMitchell(vec2 p, vec2 r)
{
    p /= r; //TODO: fails at radius0
    #if defined( USE_RADIUS_VERSIONS )
    return Mitchell1D( length(p) ); //note: radius version...
    #else
    return Mitchell1D(p.x) * Mitchell1D(p.y);
    #endif
}

// Function 12
vec3 PrefilterEnvMap( float Roughness,vec3 Dir ) { 
	vec3 N = Dir; 
	vec3 V = Dir;
	vec3 PrefilteredColor = vec3(0.);
    float TotalWeight = 0.;
	const uint NumSamples = 1024u; 
	for( uint i = 0u; i < NumSamples; i++ ) { 
		vec2 Xi = Hammersley( i, NumSamples ); 
		vec3 H = ImportanceSampleGGX( Xi, N, Roughness); 
		vec3 L = reflect(-V,H);//2. * dot( V, H ) * H - V;
		float NoL = dot( N, L ); 
		if( NoL > 0. ) { 
			PrefilteredColor += texture(Cubemap_Texture,L).rgb * NoL; 
			TotalWeight += NoL; 
		}
	}
	return PrefilteredColor / TotalWeight;
}

// Function 13
vec3 getFilterFootprint(vec3 camOrigin, vec2 fragCoord, vec2 iResolution, out vec3 c, out vec3 x, out vec3 y)
{
    //UVs
    vec2 uv = fragCoord / iResolution;
   	vec2 uvX = (fragCoord + vec2(1.,0.)) / iResolution; //
   	vec2 uvY = (fragCoord + vec2(0.,1.)) / iResolution;
    
    //RAYS
   	Ray rayCenter, rayX, rayY;
    rayCenter = setupRay(uv, camOrigin);
    rayX = setupRay(uvX, camOrigin);
    rayY = setupRay(uvY, camOrigin);
    
    //intersectionPositions
    c = findIntersection(camOrigin, rayCenter.direction);
    x = findIntersection(camOrigin, rayX.direction);
    y = findIntersection(camOrigin, rayY.direction);
    
    return rayCenter.direction;
}

// Function 14
float valueNoiseFilter(float x) {
    #if defined(VALUE_NOISE_FILTER_QUINTIC)
    return x*x*x*(x*(x*6.-15.)+10.);
    #elif defined(VALUE_NOISE_FILTER_SMOOTH)
    return smoothstep(0.0,1.0,x);
    #else
    return x;
    #endif
}

// Function 15
float warpFilter(vec2 uv, vec2 pos, float size, float ramp, vec3 iResolution)
{
    return 0.5 + sigmoid( conetip(uv, pos, size, -16., iResolution) * ramp) * 0.5;
}

// Function 16
float VXAATemporalFilterAlpha( float fpsRcp, float convergenceTime )
{
    return exp( -fpsRcp / convergenceTime );
}

// Function 17
float pg2020TriangleFilter(vec2 st){
	float s = st.x * float(PG2020W) - 0.5;
    float t = st.y * float(PG2020H) - 0.5;
    int s0 = int(floor(s));
    int t0 = int(floor(t));
    float ds = s - float(s0);
    float dt = t - float(t0);
    return (1. - ds) * (1. - dt) * texel(s0, t0) +
           (1. - ds) * dt * texel(s0, t0 + 1) +
           ds * (1. - dt) * texel(s0 + 1, t0) +
           ds * dt * texel(s0 + 1, t0 + 1);
}

// Function 18
vec2 PrefilteredEnvApprox(float roughness, float NoV) 
{
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572,  0.022);
    const vec4 c1 = vec4( 1.0,  0.0425,  1.040, -0.040);

    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;

    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Function 19
vec4 SampleMip4(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*4.,sp.z+56.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(4.,0.))*IRES),fract(sp.y));
}

// Function 20
vec4 mipmap(vec2 uv) {
    return texture(iChannel0, uv);
}

// Function 21
float warpFilter(vec2 uv, vec2 pos, float size, float ramp)
{
    return 0.5 + sigmoid( conetip(uv, pos, size, -16.) * ramp) * 0.5;
}

// Function 22
vec3 GetMipMapUVW_Dir(vec3 _uvw,vec3 _axis){
    _uvw = floor((_uvw+1.)*512.)+0.5;
    vec3 a = exp2(floor(log2(_uvw)));
    return normalize((_uvw*2./a - 3.)*(1.-abs(_axis))+_axis);
}

// Function 23
float filterFlowToTexture(float f, vec2 uv) {
    f += smoothstep(.3, .5, f) * .8 * (1. - f);
    f = smoothstep(0., 1., f);
    f = filterEdgesOfLava(f, uv);
    return f;
}

// Function 24
vec4 SampleMip1(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*32.,sp.z);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(32.,0.))*IRES),fract(sp.y));
}

// Function 25
vec4 filterFlake(vec4 color,vec3 pos,vec3 ray,vec3 ray1,vec3 ray2)
{
    vec3 d=distObj(pos,ray,radius,seed);
    vec3 n1=distObj(pos,ray1,radius,seed);
    vec3 n2=distObj(pos,ray2,radius,seed);

    vec3 lq=vec3(dot(d,d),dot(n1,n1),dot(n2,n2));
	if (lq.x<far || lq.y<far || lq.z<far) {
    	vec3 n=normalize(cross(n1-d,n2-d));
        if (lq.x<far && lq.y<far && lq.z<far) {
       		nray = n;//normalize(nray+n);
       		//nray1 = normalize(ray1+n);
       		//nray2 = normalize(ray2+n);
        }
       	float da = pow(abs(dot(n,light)),3.0);
        vec3 cf = mix(vec3(0.0,0.4,1.0),color.xyz*10.0,abs(dot(n,ray)));
       	cf=mix(cf,vec3(2.0),da);
      	color.xyz = mix(color.xyz,cf,mxc*mxc*(0.5+abs(dot(n,ray))*0.5));
    }
    
    return color;
}

// Function 26
vec4 colorFilter(vec4 c)
{
	float g = (c.x + c.y + c.z) / 3.0;
    c = vec4(g,g,g,1.0);
    
    c.x *= 0.3;
    c.y *= 0.5;
    c.z *= 0.7;
    
    return c;
}

// Function 27
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

// Function 28
float Filter01(float b)
{
	float resonanse=200.0*0.00390625; 
	float cutoff=20.0*0.001953125;
	float k3=pi*cutoff; 
	k3=cos(k3)/sin(k3); resonanse=k3*resonanse;	k3=k3*k3; 
	float km=1.0/(1.0+resonanse+k3); 
	resonanse=(1.0-resonanse+k3)/(1.0-k3); k3=2.0*(1.0-k3)*km;

	F0n1=(2.0-k3)*b-k3*F0n1+F0n2; F0n2=(1.0-resonanse)*b+resonanse*(F0n1-F0n2)*0.5; 
	b=km*(F0n1+b)*3.0;
	return b;
}

// Function 29
float filterf( vec4 side, vec4 corner, vec2 tc ) {
	vec4 v = side.xyxy + side.zzww + corner;

	return mix( mix( v.x, v.y, tc.y ), mix( v.z, v.w, tc.y ), tc.x ) * 0.25;
}

// Function 30
vec3 GetMipMapUVW_Dir2(vec3 _uvw,vec3 _axis){
    ivec3 xyz = ID_xyz[int(dot(abs(_axis.yz),vec2(1,2)))];
	vec3 uv = vec3(_uvw[xyz.x],_uvw[xyz.y],_uvw[xyz.z]);
    uv.xy = floor((uv.xy+1.)*512.);
    vec2 a = ceil(log2(uv.xy+1.));
	float max_a = max(a.x,a.y);
    float scale = exp2(max_a-1.);
    vec2 B = step((a.x+a.y)/2.,a);
    ID = max_a*3.-3. + B.y*2. + B.x;
    uv.xy = ((uv.xy+0.5)/scale - B - 0.5)*2.;
    uv = vec3(uv[xyz.x],uv[xyz.y],uv[xyz.z]);
    return normalize(uv+_axis);
}

// Function 31
vec4 getFilters(int x)
{
    float u = (float(x)/iResolution.x);
    return texture(iChannel0, vec2(u,0.0));
}

// Function 32
vec4 filterf(sampler2D tex, vec2 texcoord, vec2 texscale)
{
    float fx = fract(texcoord.x);
    float fy = fract(texcoord.y);
    texcoord.x -= fx;
    texcoord.y -= fy;

    vec4 xcubic = cubic(fx);
    vec4 ycubic = cubic(fy);

    vec4 c = vec4(texcoord.x - 0.5, texcoord.x + 1.5, texcoord.y -
0.5, texcoord.y + 1.5);
    
    vec4 s = vec4(xcubic.x + xcubic.y, xcubic.z + xcubic.w, ycubic.x +
ycubic.y, ycubic.z + ycubic.w);
    
    vec4 offset = c + vec4(xcubic.y, xcubic.w, ycubic.y, ycubic.w) /
s;

    vec4 sample0 = texture(tex, vec2(offset.x, offset.z) *
texscale);
    vec4 sample1 = texture(tex, vec2(offset.y, offset.z) *
texscale);
    vec4 sample2 = texture(tex, vec2(offset.x, offset.w) *
texscale);
    vec4 sample3 = texture(tex, vec2(offset.y, offset.w) *
texscale);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    
    float sxdx = fx*fx-fx-.5;    
    float sydy = fy*fy-fy-.5;

	float x_dir = mix(sample3- sample2 ,sample1- sample0, sy).x*sxdx;
	float y_dir = mix(sample3- sample1 ,sample2- sample0, sx).x*sydy;
	float depth = mix(mix(sample3, sample2,  sx),mix(sample2, sample0, sx), sy).x;

    return vec4(normalize(vec3(x_dir, y_dir, 1.0)), depth);
}

// Function 33
vec4 SampleMip0(vec3 sp) {
    sp.y=sp.y-0.5; float fy=floor(sp.y);
    vec2 cuv1=vec2(sp.x+floor(fy*0.2)*64.,sp.z+mod(fy,5.)*64.);
    vec2 cuv2=vec2(sp.x+floor((fy+1.)*0.2)*64.,sp.z+mod(fy+1.,5.)*64.);
    return mix(texture(iChannel1,cuv1*IRES),
               texture(iChannel1,cuv2*IRES),fract(sp.y));
}

// Function 34
vec4 SampleMip3(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*8.,sp.z+48.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(8.,0.))*IRES),fract(sp.y));
}

// Function 35
vec4 filter(sampler2D tex, vec2 uv)
{
	float f = 25.0;
	vec3 t = texture(tex, uv).rgb;
	t *= 255.0;
	t = floor(t/f)*f;
	t /= 255.0;
	return vec4(t.r,t.g,t.b,1.0);
}

// Function 36
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

// Function 37
float Filter(float inp, float cut_lp, float res_lp)
{
	fb_lp 	= res_lp+res_lp/(1.0-cut_lp + 1e-20);
	n1 		= n1+cut_lp*(inp-n1+fb_lp*(n1-n2))+p4;
	n2		= n2+cut_lp*(n1-n2);
    return n2;
}

// Function 38
float EdgeDirectFilter(vec2 a0,vec2 a1,vec2 a2,vec2 a3,vec2 a4){
    vec4 lum = vec4(a1.x,a2.x,a3.x,a4.x);
    vec4 w = 1.-step(THRESH,abs(lum - a0.x));
    float W = w.x + w.y + w.z + w.w;
    W = (W==0.0) ? W : 1./W;
    return dot(w,vec4(a1.y,a2.y,a3.y,a4.y))*W;
}

// Function 39
float jfigTriangleFilter(vec2 st){
	float s = st.x * float(JFIGW) - 0.5;
    float t = st.y * float(JFIGH) - 0.5;
    int s0 = int(floor(s));
    int t0 = int(floor(t));
    float ds = s - float(s0);
    float dt = t - float(t0);
    return (1. - ds) * (1. - dt) * texel(s0, t0) +
           (1. - ds) * dt * texel(s0, t0 + 1) +
           ds * (1. - dt) * texel(s0 + 1, t0) +
           ds * dt * texel(s0 + 1, t0 + 1);
}

// Function 40
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

// Function 41
vec3 SuperFastNormalFilter(sampler2D _tex,ivec2 iU,float strength){
    float p00 = GetTextureLuminance(_tex,iU);
    return normalize(vec3(-dFdx(p00),-dFdy(p00),1.-strength));
}

// Function 42
float FilterLanczosSinc( vec2 p, vec2 dummy_r )
{
    #if defined( USE_RADIUS_VERSIONS )
    return WindowedSinc( length(p) );
    #else
    return WindowedSinc( p.x ) * WindowedSinc( p.y );
	#endif
}

// Function 43
vec2 PrefilteredDFG_Karis(float roughness, float NoV) {
    // Karis 2014, "Physically Based Material on Mobile"
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572,  0.022);
    const vec4 c1 = vec4( 1.0,  0.0425,  1.040, -0.040);

    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;

    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Function 44
vec3 valueNoiseFilter(vec3 x) {
    #if defined(VALUE_NOISE_FILTER_QUINTIC)
    return x*x*x*(x*(x*6.-15.)+10.);
    #elif defined(VALUE_NOISE_FILTER_SMOOTH)
    return smoothstep(0.0,1.0,x);
    #else
    return x;
    #endif
}

// Function 45
vec2 valueNoiseFilter(vec2 x) {
    #if defined(VALUE_NOISE_FILTER_QUINTIC)
    return x*x*x*(x*(x*6.-15.)+10.);
    #elif defined(VALUE_NOISE_FILTER_SMOOTH)
    return smoothstep(0.0,1.0,x);
    #else
    return x;
    #endif
}

// Function 46
float get_filter_size( int idx )
{
    if ( idx == IDX_GAUSS )
    	return 1.0; //note: 1.25 matches the 2px filters reasonably
    else if ( idx == IDX_MITCHELL )
    	return 2.0;
    else if ( idx == IDX_LANCZOSSINC )
    	return 2.0;
    else
        return 1.0;
}

// Function 47
vec4 getFilters(int x)
{
    float u = (float(x)/iResolution.x);
    return texture(iChannel3, vec2(u,0.0));
}

// Function 48
vec4 edgeFilter(in int px, in int py, in vec2 fragCoord)
{
	vec4 color = vec4(0.0);
	
	for (int y = -EDGE_FILTER_SIZE; y <= EDGE_FILTER_SIZE; ++y)
	{
		for (int x = -EDGE_FILTER_SIZE; x <= EDGE_FILTER_SIZE; ++x)
		{
			color += texture(iChannel0, (fragCoord.xy + vec2(px + x, py + y)) / iResolution.xy);
		}
	}

	color /= float((2 * EDGE_FILTER_SIZE + 1) * (2 * EDGE_FILTER_SIZE + 1));
	
	return color;
}

// Function 49
float filterEdgesOfLava(float f, vec2 uv) {
    //We want the floating stuff to stay out of the edges.
    //There should be a better way to do this though!
    float uvScaler = iResolution.x/iResolution.y;
    float MWF = .003 + .2 * f; 
    float edges = smoothstep(0.0 + .7 * MWF, MWF, uv.x) - 
        smoothstep((1. - MWF) * uvScaler, (1.0 - .3 * MWF) * uvScaler, uv.x);
    f = mix(1.0, f, clamp(edges, .0, 1.));
    return f;
}

// Function 50
float FilterCubic( vec2 p, vec2 radius )
{
    p /= radius; //TODO: fails at radius0
    return smoothstep( 1.0, 0.0, length(p) );
}

// Function 51
vec3 GetReadMipMapUVW_Dir2(vec3 _uvw,vec3 _axis,float ID){
    ivec3 xyz = ID_xyz[int(dot(abs(_axis.yz),vec2(1,2)))];
	vec3 uv = vec3(_uvw[xyz.x],_uvw[xyz.y],_uvw[xyz.z]);
    float group_ID = floor((ID-=1.)/3.);
    float scale = exp2(group_ID-10.);
    float Bound = 1.-exp2(-group_ID);
    uv.xy = scale*(clamp(uv.xy,-Bound,Bound)+SCACLE_COEF[int(mod(ID,3.))])-1.;
    //f(x) = inverse(inverse(f(x)))
    uv = vec3(uv[xyz.x],uv[xyz.y],uv[xyz.z]);
    return normalize(uv + _axis);
}

// Function 52
vec3 FastNormalFilter(sampler2D _tex,ivec2 iU,float strength){
	const ivec2 e = ivec2(1,0);
    float p00 = GetTextureLuminance(_tex,iU);
    float p10 = GetTextureLuminance(_tex,iU + e.xy);
    float p01 = GetTextureLuminance(_tex,iU + e.yx);
    /* Orgin calculate 
    vec3 ab = vec3(1.,0.,p10-p00);
    vec3 ac = vec3(0.,1.,p01-p00);
    vec3 n = cross(ab,ac);
    n.z *= (1.-strength);
    return normalize(n);
	*/
	vec2 dir = p00-vec2(p10,p01);
    return normalize(vec3(dir,1.-strength));
}


```