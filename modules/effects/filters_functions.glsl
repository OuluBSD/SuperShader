// Reusable Filters Effect Functions
// Automatically extracted from effect-related shaders

// Function 1
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

// Function 2
vec3 aBlurFilter(in vec2 fragCoord, int channel, int intensity)
{
    // Since every value in a box blur kernel is the same, we can
    // apply the filter with a single float rather than a whole matrix.
    float percent = 1.0 / float(intensity * intensity);
    
    vec3 sum = vec3(0.0, 0.0, 0.0);
    int j;
    
    for(int i = 0; i < intensity; i++)
    {
        for(j = 0; j < intensity; j++)
        {
        	sum += aSample(i - (intensity/2), j - (intensity/2), fragCoord, channel) * percent;
        }
    }
 
	return sum;
    
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
float GetFilteringWeight(vec2 uv, vec2 focusPos, float targetDimension, float optimizedDimension) {
    float radialExpansion = length(UndistortDerivative(uv, focusPos));
    float resScale = targetDimension / optimizedDimension;
    float contraction = 1. / (radialExpansion * resScale);
    
    float modifiedContraction = contraction - 1. / contraction; // -> ?
    
    return max(modifiedContraction, EPS);
}

// Function 5
vec3 filterf(in vec2 fragCoord)
{
	vec3 hc =samplef(-1,-1, fragCoord) *  1. + samplef( 0,-1, fragCoord) *  2.
		 	+samplef( 1,-1, fragCoord) *  1. + samplef(-1, 1, fragCoord) * -1.
		 	+samplef( 0, 1, fragCoord) * -2. + samplef( 1, 1, fragCoord) * -1.;		

    vec3 vc =samplef(-1,-1, fragCoord) *  1. + samplef(-1, 0, fragCoord) *  2.
		 	+samplef(-1, 1, fragCoord) *  1. + samplef( 1,-1, fragCoord) * -1.
		 	+samplef( 1, 0, fragCoord) * -2. + samplef( 1, 1, fragCoord) * -1.;

	return samplef(0, 0, fragCoord) * pow(luminance(vc*vc + hc*hc), .6);
}

// Function 6
vec4 filter(sampler2D tex, vec2 uv, float time)
{
	float radius = 0.5;
	vec2 center = vec2(0.5 * x_y,0.5);
	vec2 tc = uv - center;
	float dist = length(tc);
	if (dist < radius)
	{
		float percent = (radius - dist) / radius;
		float theta = percent * percent * (2.0 * sin(time)) * 8.0;
		float s = sin(theta);
		float c = cos(theta);
		tc = vec2(dot(tc, vec2(c, -s)), dot(tc, vec2(s, c)));
	}
	tc += center;
	vec3 color = texture(tex, tc).rgb;
	return vec4(color, 1.0);
}

// Function 7
vec3 simpleFilterVenn(vec2 uv, vec2 c) {
    vec3 col = vec3(1.0);
    for (int i=0; i<3; i++) {
	    col *= vec3(circle(uv+rotate(vec2(0.3,0.),float(i*2)*PI/3.)+c,0.5)<0.0? 0.5 : 1.0);
    }
    return col;
}

// Function 8
vec4 getFilters(int x)
{
    float u = (float(x)/iResolution.x);
    return texture(iChannel3, vec2(u,0.0));
}

// Function 9
GtF3 GtFilter(
  // Linear input color
  GtF3 color,
  // Tonemapper constants
  GtF4 tone0,
  GtF4 tone1,
  GtF4 tone2,
  GtF4 tone3
 ){
//--------------------------------------------------------------
  // Peak of all channels
  GtF1 peak=GtMax3F1(color.r,color.g,color.b);
  // Protect against /0
  peak=max(peak,1.0/(256.0*65536.0));
  // Color ratio
  GtF3 ratio=color*GtRcpF1(peak);
//--------------------------------------------------------------
  // Apply tonemapper to peak
  // Contrast adjustment
  peak=pow(peak,tone0.x);
//--------------------------------------------------------------
  // Highlight compression
  #ifdef GT_SHOULDER
   peak=peak/(pow(peak,tone0.y)*tone0.z+tone0.w);
  #else
   // No shoulder adjustment avoids extra pow
   peak=peak/(peak*tone0.z+tone0.w);
  #endif
//--------------------------------------------------------------
  // Convert to non-linear space and saturate
  // Saturation is folded into first transform
  ratio=pow(ratio,tone1.xyz);
  // Move towards white on overexposure
  vec3 white=vec3(1.0,1.0,1.0);     
  ratio=GtLerpF3(ratio,white,pow(GtF3(peak),tone2.xyz));
  // Convert back to linear
  ratio=pow(ratio,tone3.xyz);
//--------------------------------------------------------------
   return ratio*peak;}

// Function 10
float filterSaw(float t, float note, float octave, float cutoff, float q){
    float saw = fract(t*note*exp2(octave-1.))-0.5;
    float sn = cos((t*note*exp2(octave)*PI)+PI*0.5);
    float filt = smoothstep(cutoff-q,cutoff+q,abs(saw)*2.);
    return mix(saw,sn,filt);}

// Function 11
float FilterLanczosSinc( vec2 p, vec2 dummy_r )
{
    #if defined( USE_RADIUS_VERSIONS )
    return WindowedSinc( length(p) );
    #else
    return WindowedSinc( p.x ) * WindowedSinc( p.y );
	#endif
}

// Function 12
vec4 geometricMeanFilter(in vec2 P, float w, float h)
{
   vec2 invR = 1.0 / R.xy;
   vec4 product = vec4(1.0);

   for (float y = -h*0.5; y <h*0.5; y += 1.0)
   {
       for (float x = -w*0.5; x <w*0.5; x += 1.0)
       {
            product *= texture(CH, (P + vec2(x, y)) * invR);
       }
   }
   
   return vec4(
        pow(product.x, 1.0 / float(w*h)),
        pow(product.y, 1.0 / float(w*h)),
        pow(product.z, 1.0 / float(w*h)),
        1.0);
}

// Function 13
vec3 FilterImage( ivec2 vPos )
{
    return SampleImage(vPos);
    
    const int sampleCount = 9;
    
    vec4 vSamples[sampleCount];
    
    for ( int sampleIndex=0; sampleIndex < sampleCount; sampleIndex++)
    {
        int dx = (sampleIndex % 3) -1;
        int dy = (sampleIndex / 3) - 1;
        vec3 vCol = SampleImage(vPos + ivec2(dx,dy) );
        vSamples[sampleIndex] = vec4( vCol, dot( vCol, vec3(0.3) ) );
    }
        
    vec4 vResult = Median9( vSamples );
    
    return vResult.rgb;
}

// Function 14
float filterSaw(float t, float note, float octave, float cutoff, float q){
    float saw = fract(t*note*exp2(octave-1.))-0.5;
    float sn = cos((t*note*exp2(octave)*pi)+pi*0.5);
    float filt = smoothstep(cutoff-q,cutoff+q,abs(saw)*2.);
    return mix(saw,sn,filt);}

// Function 15
float FilterMitchell(vec2 p, vec2 r)
{
    p /= r; //TODO: fails at radius0
    #if defined( USE_RADIUS_VERSIONS )
    return Mitchell1D( length(p) ); //note: radius version...
    #else
    return Mitchell1D(p.x) * Mitchell1D(p.y);
    #endif
}

// Function 16
vec4 sp_spectral_filter(vec4 col, float filmwidth, float cosi)
{
    vec4 retcol = vec4(0.0, 0.0, 0.0, 1.0);
    const float NN = 2001.0;
    float a = 1.0/(nu*nu);
    float cost = sqrt(a*cosi*cosi + (1.0-a));
    float n = 2.0*PI*filmwidth*cost/NN;
    float kn = 0.0;
    mat3 filt = sparsespfiltconst;
    
    for(int i = 0; i < 13; i++)
    {
        kn = (float(i)+6.0f)*n;
        filt += sparsespfilta[i]*cos(kn) + sparsespfiltb[i]*sin(kn);
    }
    
    retcol.xyz = 4.0*(filt*col.xyz)/NN;
    return retcol;
}

// Function 17
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

// Function 18
vec3 texfilter(in vec2 fragCoord)
{
    vec3 sum = texsample(-1, -1, fragCoord) * -1.
             + texsample(-1,  0, fragCoord) * -1.
             + texsample(-1,  1, fragCoord) * -1.
             + texsample( 0, -1, fragCoord) * -1.
             + texsample( 0,  0, fragCoord) *  9.
             + texsample( 0,  1, fragCoord) * -1.
             + texsample( 1, -1, fragCoord) * -1.
             + texsample( 1,  0, fragCoord) * -1.
             + texsample( 1,  1, fragCoord) * -1.;
    
	return sum;
}

// Function 19
float FilterTriangle(vec2 p, vec2 radius)
{
    p /= radius; //TODO: fails at radius0
    return clamp(1.0f - length(p), 0.0, 1.0);
}

// Function 20
vec4 harmonicMeanFilter(in vec2 P, float w, float h)
{
   vec2 invR = 1.0 / R.xy;
   vec4 sum = vec4(1.0);

   for (float y = -h*0.5; y <h*0.5; y += 1.0)
   {
       for (float x = -w*0.5; x <w*0.5; x += 1.0)
       {
            sum += 1.0 / texture(CH, (P + vec2(x, y)) * invR);
       }
   }
   
   return vec4((w*h) / sum.xyz, 1.0);
}

// Function 21
float FilterGaussian(vec2 p, vec2 radius )
{
    p /= radius; //TODO: fails at radius0
    
    #if defined( USE_RADIUS_VERSIONS )
    return Gaussian( length(p) );
    #else
	return Gaussian(p.x) * Gaussian(p.y);
    #endif
    
}

// Function 22
vec3 texfilter(in vec2 uv)
{
    vec3 val = texsample(uv);    
	return gamma(val, GAMMA);
}

// Function 23
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

// Function 24
vec4 getFilters(int x)
{
    float u = (float(x)/iResolution.x);
    return texture(iChannel0, vec2(u,0.0));
}

// Function 25
float warpFilter(vec2 uv, vec2 pos, float size, float ramp, vec3 iResolution)
{
    return 0.5 + sigmoid( conetip(uv, pos, size, -16., iResolution) * ramp) * 0.5;
}

// Function 26
vec3 valueNoiseFilter(vec3 x) {
    #if defined(VALUE_NOISE_FILTER_QUINTIC)
    return x*x*x*(x*(x*6.-15.)+10.);
    #elif defined(VALUE_NOISE_FILTER_SMOOTH)
    return smoothstep(0.0,1.0,x);
    #else
    return x;
    #endif
}

// Function 27
vec3 aKernelFilter(in vec2 fragCoord, mat3 kernel, int channel)
{
    vec3 sum = vec3(0.0, 0.0, 0.0);
    int j;
    
    for(int i = 0; i < 3; i++)
    {
        for(j = 0; j < 3; j++)
        {
        	sum += aSample(i - 1, j -1, fragCoord, channel) * kernel[i][j];
        }
    }
 
	return sum;
}

// Function 28
float VXAATemporalFilterAlpha( float fpsRcp, float convergenceTime )
{
    return exp( -fpsRcp / convergenceTime );
}

// Function 29
vec2 PrefilteredEnvApprox(float roughness, float NoV) 
{
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572,  0.022);
    const vec4 c1 = vec4( 1.0,  0.0425,  1.040, -0.040);

    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;

    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Function 30
float warpFilter(vec2 uv, vec2 pos, float size, float ramp)
{
    return 0.5 + sigmoid( conetip(uv, pos, size, -16.) * ramp) * 0.5;
}

// Function 31
vec4 arithmeticMeanFilter(in vec2 P, float w, float h)
{
   vec2 invR = 1.0 / R.xy;
   vec4 sum = vec4(1.0);

   for (float y = -h*0.5; y <h*0.5; y += 1.0)
   {
       for (float x = -w*0.5; x <w*0.5; x += 1.0)
       {
            sum += texture(CH, (P + vec2(x, y)) * invR);
       }
   }
   
   return vec4(sum.xyz / (w*h), 1.0);
}

// Function 32
F1 Filter(F1 x){
  F1 b=1.0,c=0.0;
  if(abs(x)<1.0)return(1.0/6.0)*((12.0-9.0*b-6.0*c)*x*x*abs(x)+(-18.0+12.0*b+6.0*c)*x*x+(6.0-2.0*b));
  if(abs(x)<2.0)return(1.0/6.0)*((-b-6.0*c)*x*x*abs(x)+(6.0*b+30.0*c)*x*x+(-12.0*b-48.0*c)*abs(x)+(8.0*b+24.0*c));
  return 0.0;}

// Function 33
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

// Function 34
vec3 texfilter(in vec2 fragCoord, in float intensity)
{
    vec3 sum = texsample(-1, -1, fragCoord) * -intensity
             + texsample(-1,  0, fragCoord) * -intensity
             + texsample(-1,  1, fragCoord) * -intensity
             + texsample( 0, -1, fragCoord) * -intensity
             + texsample( 0,  0, fragCoord) * (intensity * 8. + 1.) // sum should always be +1
             + texsample( 0,  1, fragCoord) * -intensity
             + texsample( 1, -1, fragCoord) * -intensity
             + texsample( 1,  0, fragCoord) * -intensity
             + texsample( 1,  1, fragCoord) * -intensity;
    
	return sum;
}

// Function 35
vec3 sharpenFilter(in vec2 fragCoord, float strength){
	vec3 f =
	texSample(-1,-1, fragCoord) *  -1. +                     
	texSample( 0,-1, fragCoord) *  -1. +                    
	texSample( 1,-1, fragCoord) *  -1. +                      
	texSample(-1, 0, fragCoord) *  -1. +                    
	texSample( 0, 0, fragCoord) *   9. +                     
	texSample( 1, 0, fragCoord) *  -1. +                      
	texSample(-1, 1, fragCoord) *  -1. +                     
	texSample( 0, 1, fragCoord) *  -1. +                     
	texSample( 1, 1, fragCoord) *  -1.
	;                                              
	return mix(texSample( 0, 0, fragCoord), f , strength);    
}

// Function 36
vec3 texfilter(in vec2 fragCoord)
{
    vec3 sum = texsample(-1, -1, fragCoord) * 1.
             + texsample(-1,  0, fragCoord) * 2.
             + texsample(-1,  1, fragCoord) * 1.
             + texsample( 0, -1, fragCoord) * 2.
             + texsample( 0,  0, fragCoord) * 4.
             + texsample( 0,  1, fragCoord) * 2.
             + texsample( 1, -1, fragCoord) * 1.
             + texsample( 1,  0, fragCoord) * 2.
             + texsample( 1,  1, fragCoord) * 1.;
    
	return sum / 16.;
}

// Function 37
vec4 colorFilter(vec4 c)
{
	float g = (c.x + c.y + c.z) / 3.0;
    c = vec4(g,g,g,1.0);
    
    c.x *= 0.3;
    c.y *= 0.5;
    c.z *= 0.7;
    
    return c;
}

// Function 38
vec3 texfilter(in vec2 fragCoord)
{
    vec3 sum = texsample(-1, -1, fragCoord)
             + texsample(-1,  0, fragCoord)
             + texsample(-1,  1, fragCoord)
             + texsample( 0, -1, fragCoord)
             + texsample( 0,  0, fragCoord)
             + texsample( 0,  1, fragCoord)
             + texsample( 1, -1, fragCoord)
             + texsample( 1,  0, fragCoord)
             + texsample( 1,  1, fragCoord);
    
	return sum / 9.;
}

// Function 39
float FilterCubic( vec2 p, vec2 radius )
{
    p /= radius; //TODO: fails at radius0
    return smoothstep( 1.0, 0.0, length(p) );
}

// Function 40
vec3 SuperFastNormalFilter(sampler2D _tex,ivec2 iU,float strength){
    float p00 = GetTextureLuminance(_tex,iU);
    return normalize(vec3(-dFdx(p00),-dFdy(p00),1.-strength));
}

// Function 41
vec4 filter(sampler2D tex, vec2 uv)
{
	float f = 25.0;
	vec3 t = texture(tex, uv).rgb;
	t *= 255.0;
	t = floor(t/f)*f;
	t /= 255.0;
	return vec4(t.r,t.g,t.b,1.0);
}

// Function 42
vec3 texfilter(in vec2 fragCoord)
{
    vec3 sum = texsample(-1, -1, fragCoord) *  3.
             + texsample( 0,  0, fragCoord) * -1.
             + texsample( 1,  1, fragCoord) * -1.;
    
	return sum;
}

// Function 43
vec2 valueNoiseFilter(vec2 x) {
    #if defined(VALUE_NOISE_FILTER_QUINTIC)
    return x*x*x*(x*(x*6.-15.)+10.);
    #elif defined(VALUE_NOISE_FILTER_SMOOTH)
    return smoothstep(0.0,1.0,x);
    #else
    return x;
    #endif
}

// Function 44
float smoothFilter(float d)
{
    float v = 2. / iResolution.y;
    return smoothstep(v, -v, d);
}

// Function 45
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

// Function 46
float valueNoiseFilter(float x) {
    #if defined(VALUE_NOISE_FILTER_QUINTIC)
    return x*x*x*(x*(x*6.-15.)+10.);
    #elif defined(VALUE_NOISE_FILTER_SMOOTH)
    return smoothstep(0.0,1.0,x);
    #else
    return x;
    #endif
}

// Function 47
float Filter(float inp, float cut_lp, float res_lp)
{
	fb_lp 	= res_lp+res_lp/(1.0-cut_lp + 1e-20);
	n1 		= n1+cut_lp*(inp-n1+fb_lp*(n1-n2))+p4;
	n2		= n2+cut_lp*(n1-n2);
    return n2;
}

// Function 48
vec3 texfilter(in vec2 fragCoord)
{
    vec3 scc = texsample( 0,  0, fragCoord);
    
    vec3 sum = texsample(-1,  0, fragCoord)
             + texsample( 0, -1, fragCoord)
             + texsample( 0,  1, fragCoord)
             + texsample( 1,  0, fragCoord)
             - scc * 4.;
    
	return scc * pow(luminance(sum * 6.), 1.25);
}

// Function 49
vec3 CRT_Filter(vec3 Colour, vec2 uv)
{
#ifdef CRT_EXTRA_NOISE_ON    
    Colour.r += Hash_From2D(uv * iTime * 911.911 * 4.0) * 0.19;
    Colour.g += Hash_From2D(uv * iTime * 563.577 * 4.0) * 0.19;
    Colour.b += Hash_From2D(uv * iTime * 487.859 * 4.0) * 0.19;
#endif    
    
    vec2 sd_uv = uv;
	vec2 crt_uv = crt_bend_uv_coords(sd_uv, 2.0);    
    vec3 scanline_Colour;
	vec3 slowscan_Colour;
	scanline_Colour.x = scanline(crt_uv, iResolution.y);
    slowscan_Colour.x = slowscan(crt_uv, iResolution.y);
    scanline_Colour.y = scanline_Colour.z = scanline_Colour.x;
	slowscan_Colour.y = slowscan_Colour.z = slowscan_Colour.x;
	Colour = mix(Colour, mix(scanline_Colour, slowscan_Colour, 0.5), 0.04);
    
    // apply the CRT-vignette filter
#ifdef CRT_VIGNETTE_ON
    Colour = CRT_Vignette(Colour, uv);
#endif     
    
    return Colour;
}

// Function 50
void OledFilter(out vec4 fragColor, in vec2 fragCoord )
{
    vec2 display = OLED_DISPLAY;
    vec2 display_coord = display * fragCoord / iResolution.xy;
    vec2 uv = floor(display_coord);
    vec2 upscale = iResolution.xy / display;

#if OLED_BILINEAR
    vec4 ltc, rtc, rbc, lbc, w = vec4(.5);
    mainImage(ltc, uv * upscale);
    mainImage(rtc, (uv + vec2(0., 1.)) * upscale);
    mainImage(rbc, (uv + vec2(1., 0.)) * upscale);
    mainImage(lbc, (uv + vec2(1., 1.)) * upscale);
    vec4 color = mix(mix(ltc, rtc, w), mix(rbc, lbc, w), w);
#else
    vec4 color;
    mainImage(color, (uv + vec2(0.5)) * upscale);
#endif
    float pixel = fract(display_coord.OLED_DIRECTION);
    vec3 rgb_f = vec3(1./3., 2./3., 1.);
    rgb_f = rgb_f.OLED_COLOR_DIRECTION;
    if (pixel < rgb_f.r && pixel >= rgb_f.r - 1./3.) {
        color = vec4(color.r, .0, .0, color.a);
    } else if (pixel < rgb_f.g && pixel >= rgb_f.g - 1./3.) {
        color = vec4(.0, color.g, .0, color.a);
    } else if (pixel < rgb_f.b && pixel >= rgb_f.b - 1./3.) {
        color = vec4(.0, .0, color.b, color.a);
    }
    fragColor = color;
}

// Function 51
vec2 GetFilteringWeight2D(vec2 uv, vec2 focusPos, vec2 targetResolution, vec2 optimizedResolution) {
    float radialExpansion = length(UndistortDerivative(uv, focusPos));
    vec2 resScale = targetResolution / optimizedResolution;
    vec2 contraction = 1. / (radialExpansion * resScale);
    
    vec2 modifiedContraction = contraction - 1. / contraction; // -> ?
    
    return max(modifiedContraction, EPS);
}

// Function 52
vec2 PrefilteredDFG_Karis(float roughness, float NoV) {
    // Karis 2014, "Physically Based Material on Mobile"
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572,  0.022);
    const vec4 c1 = vec4( 1.0,  0.0425,  1.040, -0.040);

    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;

    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Function 53
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

