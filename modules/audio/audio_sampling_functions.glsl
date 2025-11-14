// Reusable Audio Sampling Audio Functions
// Automatically extracted from audio visualization-related shaders

// Function 1
vec4 sample_camera_board(sampler2D tex, vec2 coords){
	float blur_pixel_radius = processor_sharpness_functions.z;
	float sharpness = processor_sharpness_functions.x;
	float denoise = processor_sharpness_functions.y;
	vec2 pixel_size = 2.0 / sensor_resolution * blur_pixel_radius;
	vec4 sample_1 = sample_sensor(tex, coords);
	vec4 blurred = vec4(0.0);
	
	// Four blur samples - with CA this gives 15 samples total
	// Four is the minimum to get a "full" sharpen effect where
	// all sides of an edge are lightened/darkened.
	// In theory you may be able to get it with three, but in
	// practice it still missed some edges.
	blurred += sample_sensor(tex, coords + pixel_size * vec2(1.0, 1.0));
	blurred += sample_sensor(tex, coords + pixel_size * vec2(-1.0, 1.0));
	blurred += sample_sensor(tex, coords + pixel_size * vec2(-1.0, -1.0));
	blurred += sample_sensor(tex, coords + pixel_size * vec2(1.0, -1.0));
	blurred /= 4.0;
	
	// Three blur samples - with CA this gives 12 samples total
	// This is one I wouldn't use. It adds little over two blur samples
	//blurred += sample_sensor(tex, coords + pixel_size * vec2(1.0, 0.0));
	//blurred += sample_sensor(tex, coords + pixel_size * vec2(1.0, 1.0));
	//blurred += sample_sensor(tex, coords + pixel_size * vec2(-1.0, 1.0));
	//blurred /= 3.0;
	
	// Two blur samples - with CA this gives 9 samples total
	//blurred += sample_sensor(tex, coords + pixel_size * vec2(1.0, -1.0));
	//blurred += sample_sensor(tex, coords + pixel_size * vec2(1.0, 1.0));
	//blurred /= 2.0;
	
	// The basic of a sharken is to find the edges and "increase" them
	vec4 edges = sample_1 - blurred;
	vec4 sharpened = mix(sample_1, sample_1 + edges, sharpness);
	
	// This is used to control a selective blur. Where it /isn't/ an
	// edge, blur it. Where it is an edge, sharpen it. This denoises
	// areas of roughly the same color, but leaves edges sharp.
	float sharpen_or_blur = (1.0 - length(abs(edges))) * denoise;
	vec4 processed = mix(sharpened, blurred, sharpen_or_blur);
	
	processed += processor_brightness;
	processed = pow(processed, vec4(1.0/processor_gamma));
	processed = (processed - 0.5) * processor_contrast + 0.5;
	processed = mix(vec4(dot(processed.rgb, vec3(0.2125, 0.7154, 0.0721))), processed, vec4(processor_saturation));
	processed = processed / processor_white_balance_color;
	
	return processed;
}

// Function 2
vec4 LoadMemoryChannel2(in vec2 uv)
{
    return texture(iChannel1, (uv+0.5)/iChannelResolution[2].xy, -100.)*255.;
}

// Function 3
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

// Function 4
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

// Function 5
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

// Function 6
vec4 sample_sensor(sampler2D tex, vec2 coords){
	// Rounding sample positions technically isn't needed because the sensor viewport 
	// is the resolution of the camera. I thought I'd leave this in just in case 
	// something upstream tries to sample in the wrong place.
	vec2 pixel_position = coords;
	pixel_position.x -= mod(coords.x, 2.0 / sensor_resolution.x);
	pixel_position.y -= mod(coords.y, 2.0 / sensor_resolution.y);
	
	// TODO: improve noise type and include hue variation in noise
	vec4 raw_sample = sample_lens(tex, pixel_position);
	raw_sample *= sensor_exposure;
	raw_sample += dot(sensor_pixel_noise.xy, vec2(
		rand(coords) - 0.5,  // Static Noise
		rand(coords+vec2(0.0, iTime*sensor_pixel_noise.z)) - 0.5 // Dynamic noise
	));
	
	// TODO: Should this happen before or after bit limiting? I think it's before
	// because I think it's done in the analog stage of the camera
	raw_sample *= sensor_gain;
	
	// ARRGH Godot!!!!
	//raw_sample = (raw_sample - 0.5) * sensor_dynamic_range + 0.5;
	
	// This set of operations removes the precision from the sensor
	// Most cameras transfer data in RGB565 or YUV422 or some other
	// varient that reduces the 3 bytes of color to two bytes.
	// So this is the stage where HDR is lost
	// In this case, for simplicity I assume all channels have 5 bits (RGB555)
	raw_sample = clamp(raw_sample, 0.0, 1.0);
	raw_sample -= mod(raw_sample, 1.0 / pow(2.0, 5.0));
	
	return raw_sample;
}

// Function 7
vec3 Sample_Cos_Hemisphere ( float3 wi, float3 N, out float pdf,
                             inout float seed ) {
  vec2 u = Sample_Uniform2(seed);
  float3 wo = Reorient_Hemisphere(
                normalize(To_Cartesian(sqrt(u.y), TAU*u.x)), N);
  pdf = PDF_Cosine_Hemisphere(wo, N);
  return wo;
}

// Function 8
float audioAttenuate(float t)
{    
    float lookup = t * 0.95 * float(NUM_AUDIO_SAMPLES-1);

    float signal = 0.;
    for (int i = 0; i < NUM_AUDIO_SAMPLES-1; i += 1) {
        if ( int(lookup) == i )
        {
            signal = mix(g_audioFreqs[i], 
                         g_audioFreqs[i+1], 
                         fract(lookup));
        }
    }
   
    return signal;
}

// Function 9
float get_sample_gain(float offset) {
    return texture(iChannel0, vec2(offset, 0.75)).x;
}

// Function 10
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

// Function 11
vec3 blurSample(sampler2D tex, vec2 uv, float blurAmount) {
    float pct = blurAmount*BLUR_MAX_PCT;
    vec2 sampleRBase = vec2(1.0, iResolution.y/iResolution.x);
    vec3 finalColor = vec3(0.0);

    vec2 sampleR = pct * sampleRBase;
    float ra = rand(uv)*PI_OVER_4;
    float rs = sin(ra);
    float rc = cos(ra);
    finalColor += W_0 * textureLod(tex, uv + vec2( rc, rs)*sampleR, 0.0).rgb;
    finalColor += W_0 * textureLod(tex, uv + vec2(-rs, rc)*sampleR, 0.0).rgb;
    finalColor += W_0 * textureLod(tex, uv + vec2(-rc,-rs)*sampleR, 0.0).rgb;
    finalColor += W_0 * textureLod(tex, uv + vec2( rs,-rc)*sampleR, 0.0).rgb;

    sampleR = 2.0 * pct * sampleRBase;
    ra += PI_OVER_4;
    rs = sin(ra);
    rc = cos(ra);
    finalColor += W_1 * textureLod(tex, uv + vec2( rc, rs)*sampleR, 0.0).rgb;
    finalColor += W_1 * textureLod(tex, uv + vec2(-rs, rc)*sampleR, 0.0).rgb;
    finalColor += W_1 * textureLod(tex, uv + vec2(-rc,-rs)*sampleR, 0.0).rgb;
    finalColor += W_1 * textureLod(tex, uv + vec2( rs,-rc)*sampleR, 0.0).rgb;

    sampleR = 3.0 * pct * sampleRBase;
    ra += PI_OVER_4;
    rs = sin(ra);
    rc = cos(ra);
    finalColor += W_2 * textureLod(tex, uv + vec2( rc, rs)*sampleR, 0.0).rgb;
    finalColor += W_2 * textureLod(tex, uv + vec2(-rs, rc)*sampleR, 0.0).rgb;
    finalColor += W_2 * textureLod(tex, uv + vec2(-rc,-rs)*sampleR, 0.0).rgb;
    finalColor += W_2 * textureLod(tex, uv + vec2( rs,-rc)*sampleR, 0.0).rgb;

    return finalColor;
}

// Function 12
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

// Function 13
float channelstep (out float r, in float pstep ){
r = floor(pstep*r)/pstep;
return r;
}

// Function 14
float3 Sample_Emitter ( int I, float3 O, inout float seed ) {
  float3 lorig = normalize((Sample_Uniform3(seed)-0.5)*2.0);
  lorig *= float3(0.01, lights[I].radius);
  lorig = inverse(Look_At(lights[I].N))*lorig;
  lorig += lights[I].ori;
  return normalize(lorig - O);
}

// Function 15
vec4 SampleWaterNormal( vec2 vUV, vec2 vFlowOffset, float fMag, float fFoam )
{    
    vec2 vFilterWidth = max(abs(dFdx(vUV)), abs(dFdy(vUV)));
  	float fFilterWidth= max(vFilterWidth.x, vFilterWidth.y);
    
    float fScale = (1.0 / (1.0 + fFilterWidth * fFilterWidth * 2000.0));
    float fGradientAscent = 0.25 + (fFoam * -1.5);
    vec3 dxy = FBM_DXY(vUV * 20.0, vFlowOffset * 20.0, 0.75 + fFoam * 0.25, fGradientAscent);
    fScale *= max(0.25, 1.0 - fFoam * 5.0); // flatten normal in foam
    vec3 vBlended = mix( vec3(0.0, 1.0, 0.0), normalize( vec3(dxy.x, fMag, dxy.y) ), fScale );
    return vec4( normalize( vBlended ), dxy.z * fScale );
}

// Function 16
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

// Function 17
vec2 xsample(int i, float t){
 float p=t*INT_PER_STEP*STEP;
 int j=int(p);vec2 s=vec2(0);
  if(i== 2){syy;   s.x=c015(15.-p);
 }if(i==11){       s.x=c015(15.-p);
 }if(i== 4){       s.x=clamp(16.-p,9.,15.);
 }if(i== 5){syy;   s.x=clamp(14.5-p*.5,12.0, 14.0);
                   s.x=mix(s.x,c015(s.x+sin(p*3.1415*.4)*.4),sat(p-17.));
 }if(i== 7){syy;   s.x=mix(c015(10.-abs(14.-p)),c015(16.-p*.85),step(p,11.));//if(p>11.)s.x=c015(10.-abs(14.-p));//else s.x=c015(16.-p*.85);
 }if(i==15){       s.x=mix(0.,c015(17.-p*2.2),step(p,5.));//s.x=p> 5.?0.:c015(17.-p*2.2);
 }if(i==16){s.y=1.;s.x=mix(0.,c015(15.5-p*.5),step(p,10.));//s.x=p> 10.?0.:c015(15.5-p*.5);
 }return s;}

// Function 18
vec3 blurSample(in vec2 uv, in vec2 xoff, in vec2 yoff)
{
    vec3 v11 = texture(iChannel0, uv + xoff).rgb;
    vec3 v12 = texture(iChannel0, uv + yoff).rgb;
    vec3 v21 = texture(iChannel0, uv - xoff).rgb;
    vec3 v22 = texture(iChannel0, uv - yoff).rgb;
    return (v11 + v12 + v21 + v22 + 2.0 * texture(iChannel0, uv).rgb) * 0.166667;
}

// Function 19
float SampleKey(float key)
{
	return step(0.5, texture(iChannel1, vec2(key, 0.25)).x);
}

// Function 20
float doChannel3( float t )
{
  float b = 0.0;
  float n = 0.0;
  float x = 0.0;
  t /= tint;
  D( 48,0)
  D( 0,50)D( 3,50)D( 6,50)D( 6,50)D( 3,50)D( 6,67)D(12,55)D(12,55)D( 9,52)D( 9,48)
  D( 9,53)D( 6,55)D( 6,54)D( 3,53)D( 6,52)D( 4,60)D( 4,64)D( 4,65)D( 6,62)D( 3,64)
  D( 6,60)D( 6,57)D( 3,59)D( 3,55)D( 9,55)D( 9,52)D( 9,48)D( 9,53)D( 6,55)D( 6,54)
  D( 3,53)D( 6,52)D( 4,60)D( 4,64)D( 4,65)D( 6,62)D( 3,64)D( 6,60)D( 6,57)D( 3,59)
  D( 3,55)D( 9,48)D( 9,55)D( 9,60)D( 6,53)D( 9,60)D( 3,60)D( 6,53)D( 6,48)D( 9,52)
  D( 9,55)D( 3,60)D( 6,79)D( 6,79)D( 3,79)D( 6,55)D( 6,48)D( 9,55)D( 9,60)D( 6,53)
  D( 9,60)D( 3,60)D( 6,53)D( 6,48)D( 6,56)D( 9,58)D( 9,60)D( 9,55)D( 3,55)D( 6,48)
  D( 6,48)D( 9,55)D( 9,60)D( 6,53)D( 9,60)D( 3,60)D( 6,53)D( 6,48)D( 9,52)D( 9,55)
  D( 3,60)D( 6,79)D( 6,79)D( 3,79)D( 6,55)D( 6,48)D( 9,55)D( 9,60)D( 6,53)D( 9,60)
  D( 3,60)D( 6,53)D( 6,48)D( 6,56)D( 9,58)D( 9,60)D( 9,55)D( 3,55)D( 6,48)D( 6,44)
  D( 9,51)D( 9,56)D( 6,55)D( 9,48)D( 9,43)D( 6,44)D( 9,51)D( 9,56)D( 6,55)D( 9,48)
  D( 9,43)D( 6,44)D( 9,51)D( 9,56)D( 6,55)D( 9,48)D( 9,43)D( 6,50)D( 3,50)D( 6,50)
  D( 6,50)D( 3,50)D( 6,67)D(12,55)D(12,55)D( 9,52)D( 9,48)D( 9,53)D( 6,55)D( 6,54)
  D( 3,53)D( 6,52)D( 4,60)D( 4,64)D( 4,65)D( 6,62)D( 3,64)D( 6,60)D( 6,57)D( 3,59)
  D( 3,55)D( 9,55)D( 9,52)D( 9,48)D( 9,53)D( 6,55)D( 6,54)D( 3,53)D( 6,52)D( 4,60)
  D( 4,64)D( 4,65)D( 6,62)D( 3,64)D( 6,60)D( 6,57)D( 3,59)D( 3,55)D( 9,48)D( 9,54)
  D( 3,55)D( 6,60)D( 6,53)D( 6,53)D( 6,60)D( 3,60)D( 3,53)D( 6,50)D( 9,53)D( 3,55)
  D( 6,59)D( 6,55)D( 6,55)D( 6,60)D( 3,60)D( 3,55)D( 6,48)D( 9,54)D( 3,55)D( 6,60)
  D( 6,53)D( 6,53)D( 6,60)D( 3,60)D( 3,53)D( 6,55)D( 3,55)D( 6,55)D( 3,55)D( 4,57)
  D( 4,59)D( 4,60)D( 6,55)D( 6,48)D(12,48)D( 9,54)D( 3,55)D( 6,60)D( 6,53)D( 6,53)
  D( 6,60)D( 3,60)D( 3,53)D( 6,50)
  return instr3( note2freq( n ), tint*(t-x) );
}

// Function 21
float SampleFontCharacter( int charIndex, vec2 vCharUV )
{
#if USE_FONT_TEXTURE    
    vec2 vUV;
    
    vCharUV.x = vCharUV.x * 0.6 + 0.25;
    
    vUV.x = (float(charIndex % 16) + vCharUV.x) / 16.0;
    vUV.y = (float(charIndex / 16) + vCharUV.y) / 16.0;
    
	return clamp( ( 0.503 - texture(iChannel1, vUV).w) * 100.0, 0.0, 1.0 );
#else    
	float fCharData = 0.0;
    ivec2 vCharPixel = ivec2(vCharUV * vec2(kCharPixels) );   

    #if !HIRES_FONT
        bool bCharData = CharBitmap12x20( charIndex, vCharPixel );            
        fCharData = bCharData ? 1.0 : 0.0;
    #else
        bool bCharData = CharHiRes( charIndex, vCharUV );
        fCharData = bCharData ? 1.0 : 0.0;
    #endif
    
    return fCharData;
#endif
}

// Function 22
vec3 sampleColor(vec2 uv){
    return texture(iChannel2, uvToTex(uv)).xyz;
}

// Function 23
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

// Function 24
vec4 SampleCharacterTex( uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vUV = (vec2(iChPos) + vCharUV) / 16.0f;
    return textureLod( iChannelFont, vUV, 0.0 );
}

// Function 25
vec3 SingleSample( vec2 vCoord )
{
    vec4 col = vec4(0);

    float fScale = float(kScreenResolution.y) / iResolution.y;
    
    float fScaleX = fScale * ( float(kScreenResolution.x) / float(kScreenResolution.y) ) / (kScreenRatio.x / kScreenRatio.y);
    
    vCoord.x -= (iResolution.x - float(kScreenResolution.x) / fScaleX) * 0.5;
    
    vec2 vScreenUV = vCoord * vec2(fScaleX, fScale);
        
    vScreenUV.y = float(kScreenResolution.y) - 1.0 - vScreenUV.y;
    vScreenUV = vScreenUV / vec2(kScreenResolution.xy);
        
    col = TeletextScreen( vScreenUV );
    
    vec2 vBackgroundUV = vScreenUV;
    vBackgroundUV.y = 1.0 - vBackgroundUV.y;
    //vBackgroundUV += vec2(0.05, 0.1) * iTime;
    vec3 vBackground = texture(iChannelBackgroundData, vBackgroundUV ).rgb;
    
	if ( !Key_IsToggled( iChannelKeyboard, KEY_M ) )
    {
    	vBackground = vec3(0.1);
    }
    
    if ( any( lessThan( vBackgroundUV, vec2(0) ) ) ) vBackground = vec3(0);
    if ( any( greaterThanEqual( vBackgroundUV, vec2(1) ) ) ) vBackground = vec3(0);
    
    vec3 vResult = mix( vBackground, col.rgb, col.a );
    
    // Scanlines
    //vResult *= cos( 3.14 + vScreenUV.y * 3.14 * 2.0 * 625.0 ) * 0.1 + 0.9;
    
    return vResult;
}

// Function 26
vec3 uniformSampleHemisphere(const in vec2 u) {
    float z = u[0];
    float r = sqrt(max(EPSILON, 1. - z * z));
    float phi = 2. * PI * u[1];
    return vec3(r * cos(phi), r * sin(phi), z);
}

// Function 27
vec3 sampleSphereUniform(vec2 uv)
{
	float cosTheta = 2.0*uv.x - 1.0;
	float phi = 2.0*PI*uv.y;
	return unitVecFromPhiCosTheta(phi, cosTheta);
}

// Function 28
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

// Function 29
vec3 uniformSampleCone(vec2 u12, float cosThetaMax, vec3 xbasis, vec3 ybasis, vec3 zbasis) {
    float cosTheta = (1. - u12.x) + u12.x * cosThetaMax;
    float sinTheta = sqrt(1. - cosTheta * cosTheta);
    float phi = u12.y * TWO_PI;
    vec3 samplev = sphericalDirection(sinTheta, cosTheta, sin(phi), cos(phi));
    return samplev.x * xbasis + samplev.y * ybasis + samplev.z * zbasis;
}

// Function 30
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

// Function 31
float sampleField(ivec3 gridIndex, vec2 position)
{
    int level = gridIndex.z;

#if (CurrentInterpolationMode == NearestNeighbor)
    ivec2 gridDimensions = getDimensionsOfLevel(level);

	return getDataValue(gridIndex.x, gridIndex.y, level);
#else
	vec2 fineNodePosition = getNodePosition(gridIndex);
	ivec2 fineBottomLeftOffset = mixIvec2Bool(ivec2(0), ivec2(-1), lessThanEqual(position, fineNodePosition));
	ivec3 fineBottomLeftGridIndex = ivec3(gridIndex.xy + fineBottomLeftOffset, level);

	ivec2 missingQuadrants = getMissingQuadrantsFineAndCoarse(fineBottomLeftGridIndex);
	int fineMissingQuadrants = missingQuadrants.x;

    // Regular interpolation on the first grid

#if (CurrentInterpolationMode == Bilinear)
	float valueFine = interpolateBilinear(position, fineBottomLeftGridIndex);
#elif (CurrentInterpolationMode == Bicubic)
	float valueFine = interpolateBicubic(position, fineBottomLeftGridIndex);
#elif (CurrentInterpolationMode == BSpline)
	float valueFine = interpolateBSpline(position, fineBottomLeftGridIndex);
#endif

	bool hasMissingCoarseQuadrants = (fineMissingQuadrants != 0);
	bool hasMissingFineQuadrants = (level < gNumGrids - 1) && (missingQuadrants.y != 0);
	bool coarseToFine = (!hasMissingCoarseQuadrants) && (hasMissingFineQuadrants);
	bool noBlending = (!hasMissingCoarseQuadrants) && (!hasMissingFineQuadrants);

	if (noBlending)
	{
        // No values have been missing, return regularly interpolated result
        
		return valueFine;
	}
	else
	{
        // Some values have been missing, interpolate at the same position on the second grid as well
        
		ivec3 fineGridIndex = gridIndex;
		ivec3 otherLevelGridIndex;
		float valueCoarse;

		if (coarseToFine)
		{
			valueCoarse = valueFine;

			ivec2 fineCellQuadrantOffset = mixIvec2Bool(ivec2(0), ivec2(1), greaterThan(position, fineNodePosition));

			otherLevelGridIndex = ivec3(gridIndex.xy << 1, level + 1) + ivec3(fineCellQuadrantOffset, 0);
		}
		else
		{
			otherLevelGridIndex = ivec3(gridIndex.xy >> 1, level - 1);
		}

		vec2 otherLevelCenterPosition = getNodePosition(otherLevelGridIndex);
		ivec2 otherLevelBottomLeftOffset = mixIvec2Bool(ivec2(0), ivec2(-1), lessThanEqual(position, otherLevelCenterPosition));
		ivec3 otherLevelBottomLeftGridIndex = ivec3(otherLevelGridIndex.xy + otherLevelBottomLeftOffset, otherLevelGridIndex.z);

		if (coarseToFine)
		{
			fineGridIndex = otherLevelGridIndex;
			fineBottomLeftOffset = otherLevelBottomLeftOffset;
			fineBottomLeftGridIndex = otherLevelBottomLeftGridIndex;
			fineMissingQuadrants = getMissingQuadrants(otherLevelBottomLeftGridIndex);
		}
        
        // Regular interpolation on the second grid

#if (CurrentInterpolationMode == Bilinear)
		float otherValue = interpolateBilinear(position, otherLevelBottomLeftGridIndex);
#elif (CurrentInterpolationMode == Bicubic)
		float otherValue = interpolateBicubic(position, otherLevelBottomLeftGridIndex);
#elif (CurrentInterpolationMode == BSpline)
		float otherValue = interpolateBSpline(position, otherLevelBottomLeftGridIndex);
#endif

		if (coarseToFine)
		{
			valueFine = otherValue;
		}
		else
		{
			valueCoarse = otherValue;
		}
        
        // Smoothly combine the two interpolated values within the transition region

		return smoothCombine(
			position,
			fineGridIndex,
			fineBottomLeftGridIndex,
			fineBottomLeftOffset,
			fineMissingQuadrants,
			valueFine,
			valueCoarse,
			!coarseToFine);
	}
#endif
}

// Function 32
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

// Function 33
vec4 sampleText(in vec2 uv, int start, int count, bool repeat)
{
  float fl = floor(uv + 0.5).x;
  float cursorPos = fl;
  int arrayPos = int(cursorPos);
  if (arrayPos < 0)
  {
    return vec4(0.0, 0.0, 0.0, 1.0);
  }
  if (!repeat && arrayPos >= count)
  {
    return vec4(0.0, 0.0, 0.0, 1.0);
  }

  arrayPos %= count;
  arrayPos += start;

  int letter = letterArray[arrayPos];
  vec2 lp = vec2(letter % 16, 15 - letter/16);
  vec2 uvl = lp + fract(uv+0.5)-0.5;

  // Sample the font texture. Make sure to not use mipmaps.
  // Add a small amount to the distance field to prevent a strange bug on some gpus. Slightly mysterious. :(
  vec2 tp = (uvl+0.5)*(1.0/16.0);
  return texture(iChannel2, tp, -100.0) + vec4(0.0, 0.0, 0.0, 0.000000001);
}

// Function 34
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

// Function 35
float sampleSpriteHead (vec2 uv)
{
    vec2 fracuv = fract(uv);
    int x = int(fracuv.x * 16.0);
    int y = int(fracuv.y * 16.0);
    
    // 16 idx data per row, 1 element & 4 index per 1 element...
    // => 4 element per row
    int indexperelement = 4;
    int elementperrow = 4;
    int bitsperindex = 2;
    int arrayidx = y * elementperrow + x / indexperelement;
    int idx = x % indexperelement;
    int bitoffset = (idx) * bitsperindex;
    int mask = 3 << bitoffset;
    int bits = (sprHead[arrayidx] & mask) >> bitoffset; // test

    float value = float(bits) / 3.0;
    return (value);
}

// Function 36
float getDownsampledValue(ivec3 gridIndex, int nodeIndex)
{
    const int highestLevel = gNumGrids - 1;
    int lowestLevel = gridIndex.z + 1;
    
	int coveredCells = 2 << (highestLevel - lowestLevel);
    ivec3 gridIndexOnLevel = ivec3(gridIndex.xy * coveredCells, highestLevel);
    
    float summedValues = 0.0;
    
    for (int y = 0; y < coveredCells; y++)
    {
    	for (int x = 0; x < coveredCells; x++)
        {
            ivec3 currentGridIndex = gridIndexOnLevel + ivec3(x, y, 0);
        	ivec3 retrievedGridIndex;
    		int nodeIndex = getLeafNodeIndex(currentGridIndex, retrievedGridIndex);
        	int cellIndex = convertNodeToCellIndex(nodeIndex);
            
            summedValues += fetchValue(cellIndex);
        }
    }
    
    return summedValues / float(coveredCells * coveredCells);
}

// Function 37
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

// Function 38
vec3 sampleScene(vec2 ro) {
    vec3 fcol = vec3(0);
    
    for(float i=0.; i < LIGHT_SAMPLES; i++) {
    
        float r = (random(ro + i) + i + fract(iTime)) / LIGHT_SAMPLES * PI * 2.0;

        vec2 rd = vec2(cos(r), sin(r));
        float t = trace(ro, rd);
        vec3 col = vec3(0.);
        
        if(t < 20.) {
            vec2 p = ro + t * rd;
            
            if(materials[id].light) {
                // hit a light
                col = materials[id].emissive * materials[id].intensity;
            }
            else {
                if(t < 0.0001) {
                    // inside object (not light)
                    col = texture(iChannel0, ro * 1.2).rrr;
                }
                else {
                    // hit object; calculate reflection
                    vec2 nor = normal(p);
                    vec2 refl = reflect(rd, nor);
                    int matId = id;
                    float k = trace(p + refl * 0.001, refl);
                    if(k < 20.) {
                        // hit light
                        if(materials[id].light) {
                            col = materials[id].emissive * materials[id].intensity * materials[matId].diffuse;
                        }
                        else {
                            // hit material; calculate second reflection
                            vec2 p2 = p + refl*0.001 + k*refl;
                            nor = normal(p2);
                            refl = reflect(refl, nor);
                            int matId2 = id;
                            float j = trace(p + refl * 0.001, refl);
                            if(j < 20. && materials[id].light) {
                                // hit light
                                col = materials[id].emissive * materials[id].intensity 
                                                             * materials[matId2].diffuse
                                                             * materials[matId].diffuse;
                            }
                        }
                    }
                }
            }
        }
        else col = vec3(0.3); // ambient
        fcol += col;
    }
    
    return fcol / LIGHT_SAMPLES;
}

// Function 39
vec3 samplerectBG (vec2 absuv, vec2 rectUV, float time)
{
    vec2 uvsize = (iResolution.xy / iResolution.x);
    vec2 uvsizeHalf = uvsize * 0.5;
    vec3 colourRect;
    
    // Prepare rect properties
    const float rectScrollPower = 3.0;
    vec2 rectUVOffset = vec2(pow(sin(time * 0.5), rectScrollPower) * 1.0, pow(cos(time * 0.25), rectScrollPower) * 4.0);
    rectUV += rectUVOffset; // / (rectHalfSize * 2.0);

    // Foreground : 10PRINT
    float naejangSDF = samplenaejang(vec3(rectUV, 0.0), time);
    vec3 naejangNormal = getnaejangnormal(rectUV, time); // vec3(clamp(getnaejangnormal(rectUV, time), -1.0, 1.0), 0.5);
	float naejangCenterMix = pow(1.0 - pow(1.0 - naejangSDF, 1.0), 4.0);//smoothstep(0.0, 0.75, naejangSDF - 0.1);
    naejangNormal.xy = mix(naejangNormal.xy, vec2(0.0), naejangCenterMix);
    naejangNormal.z = 1.0;//mix(0.0, 1.0, naejangCenterMix);
    
    // Calculate light
    vec3 viewVector = vec3(0.0, 0.0, 1.0);
    float lightTime = mod(time * 2.0, 6.254);
    vec3 lightPos = vec3(uvsizeHalf + vec2(cos(lightTime), sin(lightTime)) * (uvsizeHalf * 0.75), 1.0);
    vec3 lightDelta = lightPos - vec3(absuv, 0.05 + naejangSDF * 0.35);
    vec3 lightDir = normalize(lightDelta);
    float lightDist = length(lightDelta);
    
    // 1] albedo
    vec3 plasmacolour1 = hsv2rgb(vec3(fract(time * 0.2), 0.5, 1.0));
    vec3 plasmacolour2 = hsv2rgb(vec3(fract(1.0 - time * 0.2), 1.0, 0.5));
    
    vec3 diffuse = mix(plasmacolour2, plasmacolour1, naejangCenterMix);
    //colourRect = diffuse;
    
    // 2] lambert
    float lightAmbient = 0.5;
    float lightDot = dot(naejangNormal, lightDir);
    float lightDistRange = smoothstep(0.3, 0.7, clamp(1.0 / (lightDist * lightDist * 4.0), 0.0, 1.0));
    float lightLit = clamp((lightDot * lightDistRange + lightAmbient), 0.0, 1.0);
    colourRect = diffuse * lightLit;
    
    // 3] Blinn-phong specular reflection
    vec3 phongH = normalize(lightDelta + viewVector);
    float phongDistRange = naejangCenterMix * smoothstep(0.5, 0.7, clamp(1.0 / (lightDist * lightDist * 4.0), 0.0, 1.0));
    float phongDot = dot(naejangNormal, phongH);
    float phongClamped = clamp(phongDot, 0.0, 1.0);
    float phong = pow(phongClamped, 800.0);
    
    colourRect += vec3(phong * phongDistRange);
    
    return colourRect;
}

// Function 40
void UpdateAudio()
{
	float onSound = texture( iChannel0, vec2(KEY_SOUND,0.75) ).x;

	gSound.freqs[0] = texture( iChannel1, vec2( 0.01, 0.25 ) ).x;
	gSound.freqs[1] = texture( iChannel1, vec2( 0.07, 0.25 ) ).x;
	gSound.freqs[2] = texture( iChannel1, vec2( 0.014, 0.25 ) ).x;
	gSound.freqs[3] = texture( iChannel1, vec2( 0.028, 0.25 ) ).x;
	gSound.f = max(onSound,clamp(iTime*K_SOUND_FADE_IN, 0.2,1.)*gSound.freqs[1]*gSound.freqs[2]);
	gSound.fPow = pow( clamp( gSound.f*0.75, 0.0, 1.0 ), 2.0 );
}

// Function 41
float sampleLightSourcePdf(in vec3 x, vec3 ns, in vec3 wi, float d, float cosAtLight) {
    return PdfAtoW(1.0 / (light.size.x*light.size.y), d*d, cosAtLight);
}

// Function 42
vec2 echoChannelB(float t) {
	vec2 s = vec2(0);

    float fb = 0.15, tm = qbeat*1.5, cf = 0.9, ct = tm;
    // tap 1 
    s += makeBells(t) * cf; cf *= fb; 
    // tap 2
    s += makeBells(t - ct) * cf; cf *= fb; ct += tm;
    // tap 3
    s += makeBells(t - ct) * cf; cf *= fb; ct += tm;
    // tap 4
    s += makeBells(t - ct) * cf; cf *= fb; ct += tm;
    // tap 5
    s += makeBells(t - ct) * cf; cf *= fb; ct += tm;
    
    return s;
}

// Function 43
void sampleEquiAngular(
	Ray ray,
	float maxDistance,
	float Xi,
	vec3 lightPos,
	out float dist_to_sample,
	out float pdf)
{
	// get coord of closest point to light along (infinite) ray
	float delta = dot(lightPos - ray.origin, ray.dir);
	
	// get distance this point is from light
	float D = length(ray.origin + delta*ray.dir - lightPos);

	// get angle of endpoints
	float thetaA = atan(-delta, D);
	float thetaB = atan(maxDistance - delta, D);
	
	// take sample
	float t = D*tan(mix(thetaA, thetaB, Xi));
	dist_to_sample = delta + t;
	pdf = D/((thetaB - thetaA)*(D*D + t*t));
}

// Function 44
vec4 CompositeSample(vec2 UV) {
	vec2 InverseRes = 1.0 / iResolution.xy;
	vec2 InverseP = vec2(P, 0.0) * InverseRes;
	
	// UVs for four linearly-interpolated samples spaced 0.25 texels apart
	vec2 C0 = UV;
	vec2 C1 = UV + InverseP * 0.25;
	vec2 C2 = UV + InverseP * 0.50;
	vec2 C3 = UV + InverseP * 0.75;
	vec4 Cx = vec4(C0.x, C1.x, C2.x, C3.x);
	vec4 Cy = vec4(C0.y, C1.y, C2.y, C3.y);

	vec3 Texel0 = texture(iChannel0, C0).rgb;
	vec3 Texel1 = texture(iChannel0, C1).rgb;
	vec3 Texel2 = texture(iChannel0, C2).rgb;
	vec3 Texel3 = texture(iChannel0, C3).rgb;
	
	// Calculated the expected time of the sample.
	vec4 T = A * Cy * vec4(iResolution.x) * Two + B + Cx;

	const vec3 YTransform = vec3(0.299, 0.587, 0.114);
	const vec3 ITransform = vec3(0.595716, -0.274453, -0.321263);
	const vec3 QTransform = vec3(0.211456, -0.522591, 0.311135);

	float Y0 = dot(Texel0, YTransform);
	float Y1 = dot(Texel1, YTransform);
	float Y2 = dot(Texel2, YTransform);
	float Y3 = dot(Texel3, YTransform);
	vec4 Y = vec4(Y0, Y1, Y2, Y3);

	float I0 = dot(Texel0, ITransform);
	float I1 = dot(Texel1, ITransform);
	float I2 = dot(Texel2, ITransform);
	float I3 = dot(Texel3, ITransform);
	vec4 I = vec4(I0, I1, I2, I3);

	float Q0 = dot(Texel0, QTransform);
	float Q1 = dot(Texel1, QTransform);
	float Q2 = dot(Texel2, QTransform);
	float Q3 = dot(Texel3, QTransform);
	vec4 Q = vec4(Q0, Q1, Q2, Q3);

	vec4 W = vec4(Pi2 * CCFrequency * ScanTime);
	vec4 Encoded = Y + I * cos(T * W) + Q * sin(T * W);
	return (Encoded - MinC) / CRange;
}

// Function 45
vec4 sampleImg(vec2 fragCoord, float dt){
    	
#ifndef CONST_SIZE
	float lines = sin((iTime + dt) / 2.2 + 1.0) * 20.0 + 21.0;
	float gradLength = (1.0 / lines);
#endif
	
	vec2 uv = fragCoord.xy;
	vec2 center = iResolution.xy * 0.5;
	vec2 delta = uv - center;
	delta.x = abs(delta.x);
	
	float len = length(delta);
	float gradStep = floor(len * 0.005 * lines) / lines;
	float gradSmooth = len * 0.005;
	float gradCenter = gradStep + (gradLength * 0.5);
	float percentFromCenter = abs(gradSmooth - gradCenter) / (gradLength * 0.5);
	float interpLength = 0.01 * lines;
	float s = 1.0 - smoothstep(0.5 - interpLength, 0.5 + interpLength, percentFromCenter);
	
	float index = gradStep / gradLength;
	vec4 color = vec4(sin(index*0.55), sin(index*0.2), sin(index*0.3), 1)*0.5 + vec4(0.5, 0.5, 0.5, 0);
	
	float angle = atan(delta.x, delta.y);
	float worldAngle = sin(gradStep * 4.0 + (iTime + dt)) * PI * 0.5 + PI * 0.5;
	
    vec4 finalColor;
    
	if(angle < worldAngle){
		vec2 tip = vec2(sin(worldAngle), cos(worldAngle)) * gradCenter * 200.0;
		
		float tipDist = length(delta - tip);
		
		float rad = 50.0 / lines;
		float tipC = 1.0 - smoothstep(rad - 1.0, rad + 1.0, tipDist);
		
		finalColor = vec4(tipC,tipC,tipC,1);	
	}else{
		finalColor = vec4(s,s,s,1);
	}
	finalColor *= color;
    
    return finalColor;
}

// Function 46
float max_channel(vec3 v)
{
	float t = (v.x>v.y) ? v.x : v.y;
	t = (t>v.z) ? t : v.z;
	return t;
}

// Function 47
float sampleHeightfield(vec2 p)
{
    float h = 	textureLod(iChannel0, p / 40. + iTime / 400., 2.).b *
    			textureLod(iChannel1, p / 8., 2.).r * 1.6;
    
    return clamp(h, 0., 1. - 1e-4) * maxHeight;
}

// Function 48
float SampleLuminance (sampler2D tex2D, vec2 uv) {			
	return dot(Sample(tex2D, uv).rgb, vec3(0.3f, 0.59f, 0.11f));
}

// Function 49
float sampleLightSourcePdf(in vec3 x, vec3 ns, in vec3 wi, float d, float cosAtLight) {
    vec3 s = light.pos - vec3(1., 0., 0.) * light.size.x * 0.5 -
        				 vec3(0., 0., 1.) * light.size.y * 0.5;
    vec3 ex = vec3(light.size.x, 0., 0.);
    vec3 ey = vec3(0., 0., light.size.y);
    
    SphQuad squad;
    SphQuadInit(s, ex, ey, x, squad);
    return 1. / squad.S;
}

// Function 50
vec3 sample_grad_dist(vec2 uv, float font_size) {
    
    vec3 grad_dist = (textureLod(iChannel0, uv, 0.).yzw - FONT_TEX_BIAS) * font_size;

    grad_dist.y = -grad_dist.y;
    grad_dist.xy = normalize(grad_dist.xy + 1e-5);
    
    return grad_dist;
    
}

// Function 51
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

// Function 52
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

// Function 53
void sampleCamera(vec2 fragCoord, vec2 u, out vec3 rayOrigin, out vec3 rayDir)
{
	vec2 filmUv = (fragCoord.xy + u)/iResolution.xy;

	float tx = (2.0*filmUv.x - 1.0)*(iResolution.x/iResolution.y);
	float ty = (1.0 - 2.0*filmUv.y);
	float tz = 0.0;

	rayOrigin = vec3(0.0, 0.0, 5.0);
	rayDir = normalize(vec3(tx, ty, tz) - rayOrigin);
}

// Function 54
float get_audio_freq(float t){
    if(t<6.){
        return 1046.5022612024;//C8
    }
    else if(t<12.){
        return 2637.0204553030;//E9
    }
    else{
        return 4186.0090448096;//C10
    }
}

// Function 55
vec4 sample_trails(sampler2D trail_channel, vec2 trail_channel_resolution, vec2 world_coords) {
    vec2 uv = world_coords;
    uv.x *= TRAIL_MAP_SIZE.y / TRAIL_MAP_SIZE.x;
    uv /= MAP_SCALE;
    uv = uv * 0.5 + 0.5;
    
    if (any(lessThan(uv, vec2(0.0))) || any( greaterThan(uv, vec2(1.0)))) {
        return vec4(1.0);
    }
    uv = uv * TRAIL_MAP_SIZE / trail_channel_resolution;
    
    vec4 raw = texture(trail_channel, uv);

    return raw;
}

// Function 56
float doChannel4( float t )
{
  float b = 0.0;
  float x = 0.0;
  t /= tint;
  B(48)
  B(0)B(6)B(3)B(6)B(3)B(6)B(9)B(6)B(3)B(3)B(3)B(6)B(4)B(2)B(6)B(4)
  B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)
  B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)
  B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)
  B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)
  B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)
  B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)
  B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)
  B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)
  B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(3)B(6)B(3)B(6)
  B(9)B(6)B(3)B(3)B(3)B(6)B(3)B(6)B(3)B(6)B(9)B(6)B(3)B(3)B(3)B(6)
  B(3)B(6)B(3)B(6)B(9)B(6)B(3)B(3)B(3)B(6)B(3)B(6)B(3)B(6)B(9)B(6)
  B(3)B(3)B(3)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)
  B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)
  B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)
  B(6)B(4)B(2)B(9)B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)
  B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)
  B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)B(3)B(6)B(6)
  return instr4( note2freq( 42.0 ), tint*(t-x) );
}

// Function 57
float audio_freq( in sampler2D channel, in float f) { return texture( channel, vec2(f, 0.25) ).x; }

// Function 58
vec4 SampleFontTex0(vec2 uv)
{
	vec2 fl = floor(uv + 0.5);
	if (fl.y == 0.0) {
		int charIndex = int(fl.x + 3.0);
		int arrIndex = (charIndex / 4) % 73;
		uint wordFetch = textArr0[arrIndex];
		uint charFetch = (wordFetch >> ((charIndex % 4) * 8)) & 0x000000FFu;
		float charX = float (int (charFetch & 0x0000000Fu)     );
		float charY = float (int (charFetch & 0x000000F0u) >> 4);
		fl = vec2(charX, charY);
	}
	uv = fl + fract(uv+0.5)-0.5;
	return texture(iChannel2, (uv+0.5)*(1.0/16.0), -100.0) + vec4(0.0, 0.0, 0.0, 0.000000001) - 0.5;
}

// Function 59
vec3 sampleHemisphereCosWeighted( in vec3 n, in float Xi1, in float Xi2 ) {
    float theta = acos(sqrt(1.0-Xi1));
    float phi = TWO_PI * Xi2;

    return localToWorld( sphericalToCartesian( 1.0, phi, theta ), n );
}

// Function 60
float GetSource1Sample(float t)
{        
    //return 0.0;
    t = t + 0.5;
    
    return FBM( t * 30.0, 0.5 ) * 2.0 + Saw(220.1*t + 0.5) * 0.5;
    
    //return FBM( t * 30.0, 0.5 );
    //return Square(440.0*fract(t)) * Envelope(fract(t), 0.05, 0.95);
    //return Saw(220.1*t + 0.5);// * Envelope(fract(t), 0.05, 0.95);
    //return Cos(440.0*fract(t)) * Envelope(fract(t), 0.05, 0.95);
    //return Tri(440.0*fract(t)) * Envelope(fract(t), 0.05, 0.95);
}

// Function 61
vec4 downsample(sampler2D sampler, vec2 uv, float pixelSize)
{
    return texture(sampler, uv - mod(uv, vec2(pixelSize) / iResolution.xy));
}

// Function 62
float sampleMusic()
{
	return 0.5 * (
//		texture( iChannel0, vec2( 0.01, 0.25 ) ).x + 
//		texture( iChannel0, vec2( 0.07, 0.25 ) ).x + 
		texture( iChannel0, vec2( 0.15, 0.25 ) ).x + 
		texture( iChannel0, vec2( 0.30, 0.25 ) ).x);
}

// Function 63
float sampleCubeMap(float i, vec3 rd) {
	vec3 col = textureLod(iChannel0, rd * vec3(1.0,-1.0,1.0), 0.0).xyz; 
    return dot(texCubeSampleWeights(i), col);
}

// Function 64
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

// Function 65
LightSample sampleLight(in vec3 p, in Light light)
{
    vec3 difference = light.p - p;
    float distance = length(difference);
    vec3 direction = difference / distance;
    
    float sinTheta = light.r / distance;
	float cosTheta = sqrt(1.0 - sinTheta * sinTheta);
    
    LightSample result;
    
    vec3 hemi = squareToUniformSphereCap(rand2(), cosTheta);
    result.pdf = squareToUniformSphereCapPdf(cosTheta);
    
    vec3 s = normalize(cross(direction, vec3(0.433, 0.433, 0.433)));
    vec3 t = cross(direction, s);
    
    result.d = (direction * hemi.z + s * hemi.x + t * hemi.y);
    return result;
}

// Function 66
vec3 Sample_Uniform3(inout float seed) {
    return fract(sin(vec3(seed+=0.1,seed+=0.1,seed+=0.1))*
                 vec3(43758.5453123,22578.1459123,842582.632592));
}

// Function 67
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

// Function 68
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

// Function 69
float audio_ampl( in sampler2D channel, in float t) { return texture( channel, vec2(t, 0.75) ).x; }

// Function 70
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

// Function 71
float SampleBicubic(sampler2D channel, vec2 uv)
{
    uv -= vec2(0.5);
    
    vec2 uvi = floor(uv);
    vec2 uvf = uv - uvi;

    ivec2 uv0 = ivec2(uvi);
    
    float r = 0.0;
    for(int j = 0; j < 2; ++j)
    for(int i = 0; i < 2; ++i)
    {
        vec4 c = texelFetch(channel, uv0 + ivec2(i, j), 0);
        
        vec2 l = uvf;
        
        if(i != 0) l.x -= 1.0;
        if(j != 0) l.y -= 1.0;
        
        r += dot(c, kern(l));
    }
    
	return r;
}

// Function 72
vec3 importanceSampleGGX(vec2 Xi, float roughness, vec3 N)
{
    float a = roughness * roughness;
    // Sample in spherical coordinates
    float Phi = 2.0 * PI * Xi.x;
    float CosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float SinTheta = sqrt(1.0 - CosTheta * CosTheta);
    // Construct tangent space vector
    vec3 H;
    H.x = SinTheta * cos(Phi);
    H.y = SinTheta * sin(Phi);
    H.z = CosTheta;
    
    // Tangent to world space
    vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.,0.,1.0) : vec3(1.0,0.,0.);
    vec3 TangentX = normalize(cross(UpVector, N));
    vec3 TangentY = cross(N, TangentX);
    return TangentX * H.x + TangentY * H.y + N * H.z;
}

// Function 73
float texelFetchiChannel3(float x) {
    vec2 c = vec2(0);
    if(x-3.>=0.) c += .010805*vec2(texelFetch(iChannel3,ivec2(x-3.,0),0).x,1);
    if(x-2.>=0.) c += .074929*vec2(texelFetch(iChannel3,ivec2(x-2.,0),0).x,1);
    if(x-1.>=0.) c += .238727*vec2(texelFetch(iChannel3,ivec2(x-1.,0),0).x,1);
    if(x+1.<=512.) c += .238727*vec2(texelFetch(iChannel3,ivec2(x+1.,0),0).x,1);
    if(x+2.<=512.) c += .074929*vec2(texelFetch(iChannel3,ivec2(x+2.,0),0).x,1);
    if(x+3.<=512.) c += .010805*vec2(texelFetch(iChannel3,ivec2(x+3.,0),0).x,1);
    c += (1.-c.y)*vec2(texelFetch(iChannel3,ivec2(x,0),0).x,1);
    return c.x;
}

// Function 74
float KarplusStrongSample (float time, float frequency)
{
    // setup
    float numSamples = floor(iSampleRate / frequency);
    float bufferLoopCount = floor(time * iSampleRate / numSamples);
    float sampleIndex = mod(floor(time * iSampleRate), numSamples);    
    
    // calculate our static and sine samples
    #if START_NOISE
    float sample1 = hash11(sampleIndex + bufferLoopCount);
    #else
    float sample1 = sampleIndex / numSamples;
    #endif
    float sample2 = sin(time * frequency * c_twoPi);
        
    // calculate our interpolation values
    float percentTime = pow(0.98, bufferLoopCount);    
    float amplitude = percentTime;
    float lerp = pow(1.0 - percentTime, 0.25) * 0.125 + 0.875;
    
    // lerp from static to sine wave over time
    return mix(sample1, sample2, lerp) * amplitude;    
}

// Function 75
float sampleMusicA() {
	return 0.6 * (
		texture( iChannel1, vec2( 0.15, 0.25 ) ).x + 
		texture( iChannel1, vec2( 0.30, 0.25 ) ).x);
}

// Function 76
float doChannel2( float t )
{
  float b = 0.0;
  float n = 0.0;
  float x = 0.0;
  t /= tint;
  D( 0,66)D( 3,66)D( 6,66)D( 6,66)D( 3,66)D( 6,71)D(12,67)D(12,64)D( 9,60)D( 9,55)
  D( 9,60)D( 6,62)D( 6,61)D( 3,60)D( 6,60)D( 4,67)D( 4,71)D( 4,72)D( 6,69)D( 3,71)
  D( 6,69)D( 6,64)D( 3,65)D( 3,62)D( 9,64)D( 9,60)D( 9,55)D( 9,60)D( 6,62)D( 6,61)
  D( 3,60)D( 6,60)D( 4,67)D( 4,71)D( 4,72)D( 6,69)D( 3,71)D( 6,69)D( 6,64)D( 3,65)
  D( 3,62)D(15,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)D( 6,64)D( 3,65)D( 3,67)D( 6,60)
  D( 3,64)D( 3,65)D( 9,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)D( 6,77)D( 6,77)D( 3,77)
  D(18,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)D( 6,64)D( 3,65)D( 3,67)D( 6,60)D( 3,64)
  D( 3,65)D( 9,68)D( 9,65)D( 9,64)D(30,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)D( 6,64)
  D( 3,65)D( 3,67)D( 6,60)D( 3,64)D( 3,65)D( 9,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)
  D( 6,77)D( 6,77)D( 3,77)D(18,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)D( 6,64)D( 3,65)
  D( 3,67)D( 6,60)D( 3,64)D( 3,65)D( 9,68)D( 9,65)D( 9,64)D(24,68)D( 3,68)D( 6,68)
  D( 6,68)D( 3,70)D( 6,67)D( 3,64)D( 6,64)D( 3,60)D(12,68)D( 3,68)D( 6,68)D( 6,68)
  D( 3,70)D( 3,67)D(27,68)D( 3,68)D( 6,68)D( 6,68)D( 3,70)D( 6,67)D( 3,64)D( 6,64)
  D( 3,60)D(12,66)D( 3,66)D( 6,66)D( 6,66)D( 3,66)D( 6,71)D(12,67)D(12,64)D( 9,60)
  D( 9,55)D( 9,60)D( 6,62)D( 6,61)D( 3,60)D( 6,60)D( 4,67)D( 4,71)D( 4,72)D( 6,69)
  D( 3,71)D( 6,69)D( 6,64)D( 3,65)D( 3,62)D( 9,64)D( 9,60)D( 9,55)D( 9,60)D( 6,62)
  D( 6,61)D( 3,60)D( 6,60)D( 4,67)D( 4,71)D( 4,72)D( 6,69)D( 3,71)D( 6,69)D( 6,64)
  D( 3,65)D( 3,62)D( 9,72)D( 3,69)D( 6,64)D( 9,64)D( 6,65)D( 3,72)D( 6,72)D( 3,65)
  D(12,67)D( 4,77)D( 4,77)D( 4,77)D( 4,76)D( 4,74)D( 4,72)D( 3,69)D( 6,65)D( 3,64)
  D(12,72)D( 3,69)D( 6,64)D( 9,64)D( 6,65)D( 3,72)D( 6,72)D( 3,65)D(12,67)D( 3,74)
  D( 6,74)D( 3,74)D( 4,72)D( 4,71)D( 4,67)D( 3,64)D( 6,64)D( 3,60)D(12,72)D( 3,69)
  D( 6,64)D( 9,64)D( 6,65)D( 3,72)D( 6,72)D( 3,65)D(12,67) 
  return instr2( note2freq( n ), tint*(t-x) );
}

// Function 77
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

// Function 78
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

// Function 79
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

// Function 80
vec3 NearestTextureSample (vec2 P)
{
    vec2 pixel = P * c_textureSize;
    
    vec2 frac = fract(pixel);
    pixel = (floor(pixel) / c_textureSize);
    return texture(iChannel0, pixel + vec2(c_onePixel/2.0)).rgb;
}

// Function 81
float doChannel2( float t )
{
    float x = t;
    float y = 0.0;
    float b = 0.0;
    
    D(704.)
    y += instrument(329.63, tint*(t-x) );

    x = t; b = 0.0;
    D(896.)D(1152.)D(4800.)D(1728.)D(4416.)D(1728.)
    y += instrument(369.99, tint*(t-x) );

    x = t; b = 0.0;
    D(2240.)D(2880.)D(5760.)
    y += instrument(392.0, tint*(t-x) );

    x = t; b = 0.0;
    D(2432.)D(3072.)D(192.)D(960.)D(6144.)
    y += instrument(440.0, tint*(t-x) );

    x = t; b = 0.0;
    D(3584.)D(768.)D(1248.)
    y += instrument(493.88, tint*(t-x) );

    x = t; b = 0.0;
    D(3776.)D(30144.)
    y += instrument(554.37, tint*(t-x) );

    x = t; b = 0.0;
    D(3968.)D(15936.)D(192.)D(9024.)D(4416.)
    y += instrument(587.33, tint*(t-x) );

    x = t; b = 0.0;
    D(25472.)D(768.)D(1920.)D(768.)
    y += instrument(659.26, tint*(t-x) );

    x = t; b = 0.0;
    D(19328.)D(3648.)D(192.)D(5184.)D(384.)D(2496.)D(384.)
    y += instrument(739.99, tint*(t-x) );

    x = t; b = 0.0;
    D(18944.)D(3072.)D(3072.)D(3456.)D(2880.)
    y += instrument(783.99, tint*(t-x) );

    x = t; b = 0.0;
    D(18560.)D(3072.)D(768.)D(2304.)D(6336.)
    y += instrument(880.0, tint*(t-x) );

    x = t; b = 0.0;
    D(18176.)D(3072.)D(3072.)D(6528.)
    y += instrument(987.77, tint*(t-x) );
    
    x = t; b = 0.0;
    D(12416.)
    y += instrument(246.94, tint*(t-x) );
    
    x = t; b = 0.0;
    D(7040.)D(2304.)D(3840.)D(2304.)
    y += instrument(277.18, tint*(t-x) );

    x = t; b = 0.0;
    D(512.)D(9600.)D(1536.)D(4608.)
    y += instrument(293.66, tint*(t-x) );

    return y;
}

// Function 82
float doChannel2( float t )
{
  float x = 0.0;
  float y = 0.0;
  float b = 0.0;
  t /= tint;

  // D0
  x = t; b = 0.0;
  D(24)D(6)D(3)
  y += instrument( 36.0, tint*(t-x) );

  // F0
  x = t; b = 0.0;
  D(66)D(2)D(1)D(2)D(91)D(2)D(1)D(2)
  y += instrument( 43.0, tint*(t-x) );

  // G0
  x = t; b = 0.0;
  D(96)D(2)D(1)D(2)D(91)D(2)D(1)D(2)D(49)D(2)D(1)D(2)D(1)D(2)D(1)D(2)
  y += instrument( 48.0, tint*(t-x) );

  // A0
  x = t; b = 0.0;
  D(48)D(2)D(1)D(2)D(22)D(2)D(43)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(22)D(2)
  D(43)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(1)D(2)D(1)D(2)
  D(37)D(2)D(1)D(2)
  y += instrument( 55.0, tint*(t-x) );

  // A#0
  x = t; b = 0.0;
  D(42)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)
  D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(23)
  y += instrument( 58.0, tint*(t-x) );

  // C1
  x = t; b = 0.0;
  D(41)D(31)D(2)D(63)D(31)D(2)D(56)D(2)D(2)D(52)D(2)D(1)D(2)
  y += instrument( 65.0, tint*(t-x) );

  // D1
  x = t; b = 0.0;
  D(24)D(6)D(3)D(3)D(2)D(1)D(15)D(2)D(1)D(2)D(19)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)
  D(1)D(2)D(7)D(2)D(1)D(2)D(13)D(2)D(1)D(15)D(2)D(1)D(2)D(19)D(2)D(1)D(2)D(1)D(2)D(1)
  D(2)D(13)D(2)D(1)D(2)D(7)D(2)D(1)D(2)D(7)D(2)D(46)D(2)D(1)D(2)D(1)D(2)D(1)D(1)D(1)
  D(13)D(2)D(1)D(2)D(1)D(2)D(1)D(1)D(1)D(7)
  y += instrument( 73.0, tint*(t-x) );

  // F1
  x = t; b = 0.0;
  D(66)D(2)D(1)D(2)D(91)D(2)D(1)D(2)D(121)D(2)D(1)D(1)D(1)
  y += instrument( 87.0, tint*(t-x) );

  // G1
  x = t; b = 0.0;
  D(96)D(2)D(1)D(2)D(91)D(2)D(1)D(2)D(49)D(2)D(1)D(2)D(1)D(2)D(1)D(2)
  y += instrument( 97.0, tint*(t-x) );

  // A1
  x = t; b = 0.0;
  D(48)D(2)D(1)D(2)D(22)D(2)D(43)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(22)D(2)
  D(43)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(1)D(2)D(1)D(2)
  D(37)D(2)D(1)D(2)
  y += instrument( 110.0, tint*(t-x) );

  // A#1
  x = t; b = 0.0;
  D(42)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)
  D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(23)
  y += instrument( 116.0, tint*(t-x) );

  // C2
  x = t; b = 0.0;
  D(41)D(31)D(2)D(63)D(31)D(2)D(56)D(2)D(2)D(52)D(2)D(1)D(2)
  y += instrument( 130.0, tint*(t-x) );

  // D2
  x = t; b = 0.0;
  D(36)D(2)D(1)D(15)D(2)D(1)D(2)D(19)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(7)
  D(2)D(1)D(2)D(13)D(2)D(1)D(15)D(2)D(1)D(2)D(19)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)
  D(1)D(2)D(7)D(2)D(1)D(2)D(7)D(2)D(46)D(2)D(1)D(2)D(1)D(2)D(1)D(1)D(1)D(13)D(2)D(1)
  D(2)D(1)D(2)D(1)D(1)D(1)D(7)
  y += instrument( 146.0, tint*(t-x) );

  // F2
  x = t; b = 0.0;
  D(288)D(2)D(1)D(1)D(1)
  y += instrument( 174.0, tint*(t-x) );
  return y;
}

// Function 83
vec4 GetSamplePos(vec2 uv)
{
    
    rd = vec3(0.,0.,9.);
    rd = Rotate(BuildQuat(vec3(1,0,0),-uv.y*FOV),rd);
    rd = Rotate(BuildQuat(vec3(0,1,0),uv.x*FOV),rd);
    
    vec3 lro = vec3(0.,0.,-.1);
    vec3 lrd = rd;
    
    ro = camPos;
    rd = Rotate(camQuat,rd);
    
    
    vec3 mp=ro;
    lmp=lro;
    
    int i;
    for (i=0;i<maxStepRayMarching;i++){
        dist = map(mp,lmp+dpos*0.005,time-(length(ro-mp)/speedOfLight));
        //if(abs(rayDist)<mix(0.0001,0.1,(mp.z+camDist)*0.005))
        if(dist<0.0001)
            break;
        mp+=rd*dist;
        lmp+=lrd*dist;
    }
    
    float ma=1.-float(i)/80.;
    
    return vec4(mp,ma);
}

// Function 84
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

// Function 85
float doChannel4( float t )
{
  float b = 0.0;
  float x = 0.0;
  t /= tint;
  B(0)B(6)B(3)B(6)B(3)B(6)B(9)B(6)B(3)B(3)B(3)B(6)B(4)B(2)B(6)B(4)
  B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)
  B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)
  B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)
  B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)
  B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)
  B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)
  B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)
  B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)
  B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(3)B(6)B(3)B(6)
  B(9)B(6)B(3)B(3)B(3)B(6)B(3)B(6)B(3)B(6)B(9)B(6)B(3)B(3)B(3)B(6)
  B(3)B(6)B(3)B(6)B(9)B(6)B(3)B(3)B(3)B(6)B(3)B(6)B(3)B(6)B(9)B(6)
  B(3)B(3)B(3)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)
  B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)
  B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)B(6)B(4)B(2)
  B(6)B(4)B(2)B(9)B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)
  B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)
  B(3)B(6)B(6)B(9)B(3)B(6)B(6)B(9)B(3)B(6)B(6)
  return instr4( note2freq( 42.0 ), tint*(t-x) );
}

// Function 86
vec4 SampleTextureBilinearlyAndUnpack(sampler2D tex, vec2 uv)
{
    vec4 sample_color = texture(tex, uv, 0.0);
#ifdef PACK_SIGNED_TO_UNSIGNED
    sample_color = 2.0 * sample_color - 1.0;
#endif // PACK_SIGNED_TO_UNSIGNED
    return sample_color;
}

// Function 87
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

// Function 88
vec3 _sample(vec3 rd)
{
    #ifdef USEENV
	vec3 col = texture(iChannel0,rd).rgb;
    col = pow(col*1.5,vec3(2.2+3.5));
    #else
    vec3 col = vec3(pow(rd.z*0.5+0.5,3.));
    #endif
    
    #if TINTED > 0
    col *= mix(vec3(0.1,0.3,0.9),vec3(0.9,0.27,0.08),rd.x*0.5+0.5);
    #if TINTED > 1
    col *= mix(vec3(0.1,0.9,0.3),vec3(1.,.5,.2),rd.z*0.5+0.5);
    #endif
    #endif
    
    return col;
}

// Function 89
float smootherSample(vec2 uv,float e) 
{
	e*=3.0; 
	return (
		 texture(iChannel0,uv-vec2(e*-0.5,0.0)).x
		+texture(iChannel0,uv-vec2(e*-0.4,0.0)).x
		+texture(iChannel0,uv-vec2(e*-0.3,0.0)).x
		+texture(iChannel0,uv-vec2(e*-0.2,0.0)).x
		+texture(iChannel0,uv-vec2(e*-0.1,0.0)).x
		+texture(iChannel0,uv-vec2(e*+0.0,0.0)).x
		+texture(iChannel0,uv-vec2(e*+0.1,0.0)).x
		+texture(iChannel0,uv-vec2(e*+0.2,0.0)).x
		+texture(iChannel0,uv-vec2(e*+0.3,0.0)).x
		+texture(iChannel0,uv-vec2(e*+0.4,0.0)).x
		+texture(iChannel0,uv-vec2(e*+0.5,0.0)).x
		)/11.0;
}

// Function 90
float doChannel1( float t )
{
  float b = 0.0;
  float n = 0.0;
  float x = 0.0;
  t /= tint;
  D( 48,0)
  D( 0,76)D( 3,76)D( 6,76)D( 6,72)D( 3,76)D( 6,79)D(24,72)D( 9,67)D( 9,64)D( 9,69)
  D( 6,71)D( 6,70)D( 3,69)D( 6,67)D( 4,76)D( 4,79)D( 4,81)D( 6,77)D( 3,79)D( 6,76)
  D( 6,72)D( 3,74)D( 3,71)D( 9,72)D( 9,67)D( 9,64)D( 9,69)D( 6,71)D( 6,70)D( 3,69)
  D( 6,67)D( 4,76)D( 4,79)D( 4,81)D( 6,77)D( 3,79)D( 6,76)D( 6,72)D( 3,74)D( 3,71)
  D(15,79)D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,68)D( 3,69)D( 3,72)D( 6,69)D( 3,72)
  D( 3,74)D( 9,79)D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,84)D( 6,84)D( 3,84)D(18,79)
  D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,68)D( 3,69)D( 3,72)D( 6,69)D( 3,72)D( 3,74)
  D( 9,75)D( 9,74)D( 9,72)D(30,79)D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,68)D( 3,69)
  D( 3,72)D( 6,69)D( 3,72)D( 3,74)D( 9,79)D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,84)
  D( 6,84)D( 3,84)D(18,79)D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,68)D( 3,69)D( 3,72)
  D( 6,69)D( 3,72)D( 3,74)D( 9,75)D( 9,74)D( 9,72)D(24,72)D( 3,72)D( 6,72)D( 6,72)
  D( 3,74)D( 6,76)D( 3,72)D( 6,69)D( 3,67)D(12,72)D( 3,72)D( 6,72)D( 6,72)D( 3,74)
  D( 3,76)D(27,72)D( 3,72)D( 6,72)D( 6,72)D( 3,74)D( 6,76)D( 3,72)D( 6,69)D( 3,67)
  D(12,76)D( 3,76)D( 6,76)D( 6,72)D( 3,76)D( 6,79)D(24,72)D( 9,67)D( 9,64)D( 9,69)
  D( 6,71)D( 6,70)D( 3,69)D( 6,67)D( 4,76)D( 4,79)D( 4,81)D( 6,77)D( 3,79)D( 6,76)
  D( 6,72)D( 3,74)D( 3,71)D( 9,72)D( 9,67)D( 9,64)D( 9,69)D( 6,71)D( 6,70)D( 3,69)
  D( 6,67)D( 4,76)D( 4,79)D( 4,81)D( 6,77)D( 3,79)D( 6,76)D( 6,72)D( 3,74)D( 3,71)
  D( 9,76)D( 3,72)D( 6,67)D( 9,68)D( 6,69)D( 3,77)D( 6,77)D( 3,69)D(12,71)D( 4,81)
  D( 4,81)D( 4,81)D( 4,79)D( 4,77)D( 4,76)D( 3,72)D( 6,69)D( 3,67)D(12,76)D( 3,72)
  D( 6,67)D( 9,68)D( 6,69)D( 3,77)D( 6,77)D( 3,69)D(12,71)D( 3,77)D( 6,77)D( 3,77)
  D( 4,76)D( 4,74)D( 4,72)D(24,76)D( 3,72)D( 6,67)D( 9,68)D( 6,69)D( 3,77)D( 6,77)
  D( 3,69)D(12,71)
  return instr1( note2freq( n ), tint*(t-x) );
}

// Function 91
vec4 multisample( sampler2D tex, vec2 uv, float mip, float offset)
{
	vec4 outcol;
    outcol += texture( tex, uv + vec2(    0.0, 0.0), mip);
    outcol += texture( tex, uv + vec2( offset, 0.0), mip);
    outcol += texture( tex, uv + vec2(-offset, 0.0), mip);
    outcol += texture( tex, uv + vec2( 0.0, offset), mip);
    outcol += texture( tex, uv + vec2( 0.0,-offset), mip);
    return outcol * 0.2;
}

// Function 92
vec3 disneySheenSample(out vec3 wi, const in vec3 wo, out float pdf, const in vec2 u, const in vec3 normal, const in MaterialInfo material) {
    
    cosineSample

    vec3 H = normalize(wo+wi);
    float LdotH = dot(wo,H);
    
    pdf = pdfLambertianReflection(wi, wo, normal);
    return disneySheen(LdotH, material);
}

// Function 93
vec4 sample3D(sampler2D tex, vec3 uvw, vec3 vres)
{
    uvw = mod(floor(uvw * iVResolution), iVResolution);
    float idx = (uvw.z * (iVResolution.x*iVResolution.y)) + (uvw.y * iVResolution.x) + uvw.x;
    vec2 uv = vec2(mod(idx, iResolution.x), floor(idx / iResolution.x));
    
    return texture(tex, (uv + 0.5) / iResolution.xy);
}

// Function 94
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

// Function 95
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

// Function 96
vec3 samplePos3D(in vec3 rp)
{
    vec3 fw = normalize(vec3(rp.x, 0.0, rp.z));
    vec3 pIn = fw * RADIUS;
    vec3 rt =  cross(fw, up);

    vec3 localP = rp-pIn;
    rot[0] = fw; rot[1] = up; rot[2] = rt;
    localP = transpose(rot) * localP; 
    localP = rotz(-ROTATION_T) * localP;
    localP = rot * localP;
    return (localP+pIn);
}

// Function 97
float getSampleDim1(int sampleIndex,vec2 fragCoord)
{
	return fract(getDimensionHash(1,fragCoord) + radicalInverse(sampleIndex, 3));
}

// Function 98
vec3 lightSample( const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, float seed, const in MaterialInfo material) {
    vec3 L = (light.position - interaction.point);
    vec3 V = -normalize(interaction.incomingRayDir);
    vec3 r = reflect(V, interaction.normal);
    vec3 centerToRay = dot( L, r ) * r - L;
    vec3 closestPoint = L + centerToRay * clamp( light.radius / length( centerToRay ), 0.0, 1.0 );
    wi = normalize(closestPoint);


    return light.L/dot(L, L);
}

// Function 99
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

// Function 100
float sampleSpriteBody1 (vec2 uv)
{
    vec2 fracuv = fract(uv);
    int x = int(fracuv.x * 16.0);
    int y = int(fracuv.y * 20.0);
    
    // 16 idx data per row, 1 element & 4 index per 1 element...
    // => 4 element per row
    int indexperelement = 4;
    int elementperrow = 4;
    int bitsperindex = 2;
    int arrayidx = y * elementperrow + x / indexperelement;
    int idx = x % indexperelement;
    int bitoffset = (idx) * bitsperindex;
    int mask = 3 << bitoffset;
    int bits = (sprBody1[arrayidx] & mask) >> bitoffset; // test

    float value = float(bits) / 3.0;
    return (value);
}

// Function 101
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

// Function 102
float getUpsampledValue(ivec3 gridIndex, int cellIndex, int quadrant)
{
	ivec3 quadrantGridIndex = ivec3((gridIndex.xy << 1) + ivec2(quadrant & 1, (quadrant & 2) >> 1), gridIndex.z + 1);

	ivec2 gridExtents = getDimensionsOfLevel(gridIndex.z) - 1;

	ivec3 bottomLeftGridIndex = gridIndex;
	ivec3 bottomRightGridIndex = gridIndex;
	ivec3 topLeftGridIndex = gridIndex;
	ivec3 topRightGridIndex = gridIndex;

	vec2 relativePosition;

	if (quadrant == 0)
	{
		bottomLeftGridIndex.xy = clamp(bottomLeftGridIndex.xy + ivec2(-1, -1), ivec2(0), gridExtents);
		bottomRightGridIndex.xy = clamp(bottomRightGridIndex.xy + ivec2(0, -1), ivec2(0), gridExtents);
		topLeftGridIndex.xy = clamp(topLeftGridIndex.xy + ivec2(-1, 0), ivec2(0), gridExtents);
		topRightGridIndex.xy = clamp(topRightGridIndex.xy + ivec2(0, 0), ivec2(0), gridExtents);

		relativePosition = vec2(0.75, 0.75);
	}
	else if (quadrant == 1)
	{
		bottomLeftGridIndex.xy = clamp(bottomLeftGridIndex.xy + ivec2(0, -1), ivec2(0), gridExtents);
		bottomRightGridIndex.xy = clamp(bottomRightGridIndex.xy + ivec2(1, -1), ivec2(0), gridExtents);
		topLeftGridIndex.xy = clamp(topLeftGridIndex.xy + ivec2(0, 0), ivec2(0), gridExtents);
		topRightGridIndex.xy = clamp(topRightGridIndex.xy + ivec2(1, 0), ivec2(0), gridExtents);

		relativePosition = vec2(0.25, 0.75);
	}
	else if (quadrant == 2)
	{
		bottomLeftGridIndex.xy = clamp(bottomLeftGridIndex.xy + ivec2(-1, 0), ivec2(0), gridExtents);
		bottomRightGridIndex.xy = clamp(bottomRightGridIndex.xy + ivec2(0, 0), ivec2(0), gridExtents);
		topLeftGridIndex.xy = clamp(topLeftGridIndex.xy + ivec2(-1, 1), ivec2(0), gridExtents);
		topRightGridIndex.xy = clamp(topRightGridIndex.xy + ivec2(0, 1), ivec2(0), gridExtents);

		relativePosition = vec2(0.75, 0.25);
	}
	else
	{
		bottomLeftGridIndex.xy = clamp(bottomLeftGridIndex.xy + ivec2(0, 0), ivec2(0), gridExtents);
		bottomRightGridIndex.xy = clamp(bottomRightGridIndex.xy + ivec2(1, 0), ivec2(0), gridExtents);
		topLeftGridIndex.xy = clamp(topLeftGridIndex.xy + ivec2(0, 1), ivec2(0), gridExtents);
		topRightGridIndex.xy = clamp(topRightGridIndex.xy + ivec2(1, 1), ivec2(0), gridExtents);

		relativePosition = vec2(0.25, 0.25);
	}

	float v0 = texelFetch(iChannel1, clamp(bottomLeftGridIndex.xy, ivec2(0), gridExtents), 0).x;
	float v1 = texelFetch(iChannel1, clamp(bottomRightGridIndex.xy, ivec2(0), gridExtents), 0).x;
	float v2 = texelFetch(iChannel1, clamp(topLeftGridIndex.xy, ivec2(0), gridExtents), 0).x;
	float v3 = texelFetch(iChannel1, clamp(topRightGridIndex.xy, ivec2(0), gridExtents), 0).x;

	return mix(mix(v0, v1, relativePosition.x), mix(v2, v3, relativePosition.x), relativePosition.y);
}

// Function 103
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

// Function 104
float sampleDensity(vec3 pos, vec2 uv, float h){
    h = abs(h * 2.0 - 1.0);
   uv.x += iTime * 0.0002;
    uv.y += iTime * 0.0002;
  //  pos *= 0.01;
  //  uv += vec2(supernoise3d(pos), supernoise3d(pos+100.0))*0.01; 
	return smoothstep(h, 1.3, texture(iChannel0, mirrored(uv)).r + texture(iChannel0, mirrored(uv * 0.78 + 34.1235)).r);// * noise2d(uv);
}

// Function 105
vec3 sampleImage(vec2 coord){
   return pow3(texture(iChannel0,viewport(coord)).rgb,GAMMA);
}

// Function 106
vec3 sampleAngle(float u1) {
	float r = sqrt(u1);
	return vec3(-r * -0.809017, -sqrt(1.0 - u1), r * 0.587785);
}

// Function 107
float doChannel1( float t )
{
  float x = 0.0;
  float y = 0.0;
  float b = 0.0;
  t /= tint;

  // F2
  x = t; b = 0.0;
  D(36)D(2)D(2)D(20)D(2)D(16)D(6)D(2)D(226)
  y += instrument( 174.0, tint*(t-x) );

  // G2
  x = t; b = 0.0;
  D(53)D(208)
  y += instrument( 195.0, tint*(t-x) );

  // A2
  x = t; b = 0.0;
  D(34)D(2)D(2)D(2)D(1)D(7)D(2)D(2)D(2)D(1)D(3)D(8)D(2)D(8)D(2)D(4)D(2)D(2)D(2)D(1)
  D(31)D(2)D(4)D(138)D(46)D(2)
  y += instrument( 220.0, tint*(t-x) );

  // A#2
  x = t; b = 0.0;
  D(42)D(2)D(2)D(14)D(2)D(2)D(1)D(25)D(2)D(16)D(2)D(2)
  y += instrument( 233.0, tint*(t-x) );

  // B2
  x = t; b = 0.0;
  D(125)
  y += instrument( 246.0, tint*(t-x) );

  // C3
  x = t; b = 0.0;
  D(35)D(6)D(7)D(2)D(3)D(1)D(5)D(7)D(2)D(2)D(1)D(1)D(2)D(3)D(6)D(199)D(2)D(2)D(2)D(1)
  y += instrument( 261.0, tint*(t-x) );

  // C#3
  x = t; b = 0.0;
  D(120)D(2)D(4)D(132)D(1)D(5)D(42)D(2)
  y += instrument( 277.0, tint*(t-x) );

  // D3
  x = t; b = 0.0;
  D(0)D(2)D(1)D(2)D(1)D(2)D(1)D(1)D(1)D(1)D(2)D(1)D(2)D(1)D(2)D(1)D(1)D(1)D(1)D(2)
  D(1)D(2)D(1)D(2)D(1)D(3)D(2)D(2)D(2)D(2)D(2)D(1)D(5)D(3)D(5)D(2)D(2)D(12)D(2)D(6)
  D(2)D(2)D(2)D(2)D(2)D(1)D(1)D(2)D(5)D(3)D(2)D(2)D(2)D(3)D(3)D(6)D(1)D(136)D(9)D(2)
  D(2)D(2)D(1)D(17)D(2)D(2)D(2)D(1)D(11)
  y += instrument( 293.0, tint*(t-x) );

  // E3
  x = t; b = 0.0;
  D(41)D(7)D(2)D(15)D(7)D(2)D(27)D(6)D(13)D(2)D(4)D(132)D(1)D(23)D(2)D(2)D(2)D(18)D(4)
  y += instrument( 329.0, tint*(t-x) );

  // F3
  x = t; b = 0.0;
  D(42)D(2)D(2)D(20)D(2)D(2)D(19)D(11)D(2)D(6)D(2)D(4)D(5)D(5)D(8)D(2)D(2)D(20)D(2)D(16)
  D(6)D(2)D(82)D(4)D(2)D(2)D(2)D(2)D(1)D(12)D(5)D(2)D(2)D(2)D(1)D(7)
  y += instrument( 349.0, tint*(t-x) );

  // G3
  x = t; b = 0.0;
  D(47)D(24)D(19)D(2)D(2)D(2)D(2)D(3)D(11)D(37)D(120)D(13)D(2)D(2)D(2)D(18)
  y += instrument( 391.0, tint*(t-x) );

  // A3
  x = t; b = 0.0;
  D(95)D(5)D(2)D(12)D(16)D(2)D(2)D(2)D(1)D(7)D(2)D(2)D(2)D(1)D(3)D(8)D(2)D(8)D(2)D(4)
  D(2)D(2)D(2)D(1)D(31)D(2)D(4)D(2)D(2)D(12)D(1)D(1)D(30)D(2)D(2)D(3)D(12)D(5)D(2)D(2)
  D(3)
  y += instrument( 440.0, tint*(t-x) );

  // A#3
  x = t; b = 0.0;
  D(96)D(2)D(40)D(2)D(2)D(14)D(2)D(2)D(1)D(25)D(2)D(16)D(2)D(2)D(24)D(18)D(1)D(1)D(24)D(24)
  y += instrument( 466.0, tint*(t-x) );

  // C4
  x = t; b = 0.0;
  D(131)D(6)D(7)D(2)D(3)D(1)D(5)D(7)D(2)D(2)D(1)D(1)D(2)D(3)D(6)D(47)D(2)
  y += instrument( 523.0, tint*(t-x) );

  // C#4
  x = t; b = 0.0;
  D(216)D(2)D(3)
  y += instrument( 554.0, tint*(t-x) );

  // D4
  x = t; b = 0.0;
  D(132)D(2)D(2)D(2)D(2)D(2)D(1)D(5)D(3)D(5)D(2)D(2)D(12)D(2)D(6)D(2)D(2)D(2)D(2)D(2)
  D(1)D(1)D(2)D(5)D(3)D(2)D(2)D(2)D(3)D(3)D(6)D(2)D(2)D(4)D(4)D(2)D(5)D(7)D(5)
  y += instrument( 587.0, tint*(t-x) );

  // E4
  x = t; b = 0.0;
  D(137)D(7)D(2)D(15)D(7)D(2)D(27)D(6)D(13)D(2)D(8)
  y += instrument( 659.0, tint*(t-x) );

  // F4
  x = t; b = 0.0;
  D(138)D(2)D(2)D(20)D(2)D(2)D(19)D(11)D(2)D(6)D(2)D(4)D(5)D(13)D(2)D(1)D(4)D(3)
  y += instrument( 698.0, tint*(t-x) );

  // G4
  x = t; b = 0.0;
  D(143)D(24)D(19)D(2)D(2)D(2)D(2)D(3)D(11)D(24)D(14)D(4)
  y += instrument( 783.0, tint*(t-x) );

  // A4
  x = t; b = 0.0;
  D(191)D(5)D(2)D(12)D(24)
  y += instrument( 880.0, tint*(t-x) );

  // A#4
  x = t; b = 0.0;
  D(192)D(2)D(52)
  y += instrument( 932.0, tint*(t-x) );

  // C5
  x = t; b = 0.0;
  y += instrument( 1046.0, tint*(t-x) );
  return y;
}

// Function 108
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

// Function 109
vec3 sampleImage(vec2 coord){
    if( PASS2 ){
        return texture(iChannel0,viewport(coord)).rgb;
    } else {
    	return pow3(texture(iChannel0,viewport(coord)).rgb,GAMMA);
    }
}

// Function 110
float sampleSlope(vec2 sp, vec2 uv)
{
	float s = dYdt(sp.y, sp.x);    
 	float theta = atan(s);   
    
    float L = 1.6; // Length of slope thing
    
    vec2 a = vec2(sp.x - L*cos(theta), sp.y - L*sin(theta));
    vec2 b = vec2(sp.x + L*cos(theta), sp.y + L*sin(theta));
    
    float ds = smoothstep(0.058, 0.0581, ln(uv, a, b));
    
    return ds * step(0.15, length(uv-sp));
}

// Function 111
float GetSource0Sample(float t)
{     
    //return 0.0;
    return Square(220.0*fract(t)) * Envelope(fract(t), 0.05, 0.95);

    //return FBM( t * 30.0, 0.5 );
    //return Square(220.0*fract(t)) * Envelope(fract(t), 0.05, 0.95);
    //return Saw(220.0*fract(t)) * Envelope(fract(t), 0.05, 0.95);
    //return Cos(220.0*fract(t)) * Envelope(fract(t), 0.05, 0.95);
    //return Tri(220.0*fract(t)) * Envelope(fract(t), 0.05, 0.95);
}

// Function 112
vec3 getCosineWeightedSample(vec3 dir) {
	vec3 o1 = normalize(ortho(dir));
	vec3 o2 = normalize(cross(dir, o1));
	vec2 r = vec2(randomFloat(), randomFloat());
	r.x = r.x * 2.0 * Pi;
	r.y = pow(r.y, .5);
	float oneminus = sqrt(1.0-r.y*r.y);
	return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
}

// Function 113
vec3 sampleLight( const in vec3 ro, inout float seed ) {
    vec3 n = randomSphereDirection( seed ) * lightSphere.w;
    return lightSphere.xyz + n;
}

// Function 114
vec3 disneySubSurfaceSample(out vec3 wi, const in vec3 wo, out float pdf, const in vec2 u, const in vec3 normal, const in MaterialInfo material) {
    
    cosineSample

    vec3 H = normalize(wo+wi);
    float NdotH = dot(normal,H);
    
    pdf = pdfLambertianReflection(wi, wo, normal);
    return vec3(0.);//disneySubsurface(NdotL, NdotV, NdotH, material) * material.subsurface;
}

// Function 115
vec2 sampleAperture(int nbBlades, float rotation){
    
    float alpha = 2.0*Pi / float(nbBlades);
    float side = sin(alpha/2.0);
    
    int blade = int(randomFloat() * float(nbBlades));
    
    vec2 tri = vec2(randomFloat(), -randomFloat());
    if(tri.x+tri.y > 0.0) tri = vec2(tri.x-1.0, -1.0-tri.y);
    tri.x *= side;
    tri.y *= sqrt(1.0-side*side);
    
    float angle = rotation + float(blade)/float(nbBlades) * 2.0 * Pi;
    
    return vec2(tri.x * cos(angle) + tri.y * sin(angle),
                tri.y * cos(angle) - tri.x * sin(angle));
}

// Function 116
vec4 SampleMip0(vec3 sp) {
    sp.y=sp.y-0.5; float fy=floor(sp.y);
    vec2 cuv1=vec2(sp.x+floor(fy*0.2)*64.,sp.z+mod(fy,5.)*64.);
    vec2 cuv2=vec2(sp.x+floor((fy+1.)*0.2)*64.,sp.z+mod(fy+1.,5.)*64.);
    return mix(texture(iChannel1,cuv1*IRES),
               texture(iChannel1,cuv2*IRES),fract(sp.y));
}

// Function 117
vec4 sampleLevel0( vec2 uv )
{
    return texture( iChannel0, uv, -10.0 );
}

// Function 118
vec2 getBokehTapSampleCoord(const in vec2 o, const in float f, const float n, const in float phiShutterMax){
    vec2 ab = (o * 2.0) - vec2(1.0);    
    vec2 phir = ((ab.x * ab.x) > (ab.y * ab.y)) ? vec2((abs(ab.x) > 1e-8) ? ((PI * 0.25) * (ab.y / ab.x)) : 0.0, ab.x) : vec2((abs(ab.y) > 1e-8) ? ((PI * 0.5) - ((PI * 0.25) * (ab.x / ab.y))) : 0.0, ab.y); 
    phir.x += f * phiShutterMax;
   	phir.y *= (f > 0.0) ? pow((cos(PI / n) / cos(phir.x - ((2.0 * (PI / n)) * floor(((n * phir.x) + PI) / (2.0 * PI))))), f) : 1.0;
    return vec2(cos(phir.x), sin(phir.x)) * phir.y;
}

// Function 119
bool sampleDistComp(vec2 uv, float radius_)
{
    float threshold = 0.005;
    float radius = 0.1 + radius_;
    
    return sqrt(dot(uv, uv)) < radius*(1.0+threshold) && sqrt(dot(uv, uv)) > radius * (1.0-threshold);
}

// Function 120
void downSampled(out sampler res)
{
    
    const int steps = downSampleSteps;
    const int sps = steps / samples;
    res;
    float s = 1.0 / float(steps);
    for(int j = 0; j < samples; j++)
    {
        for(int i = 0; i < sps; i++)
        {
            res.s[j] += texture(iChannel0, vec2(float(j * sps + i) * s, 0.25)).x;
            
        }
        float n = sqrt(float(j) / float(samples));
        float k = (1.0 - bassMalus) + n * bassMalus;
        res.s[j] = (res.s[j] / float(sps)) * k;
        if(res.s[j] < volMin)
            volMin = res.s[j];
        if(res.s[j] > volMax)
            volMax = res.s[j];
    }
    volMin = max(volLo, volMin);
    volMax = max(volHi, volMax);
    for(int j = 0; j < samples; j++)
    {
        res.s[j] = pow(smoothstep(volMin, volMax, res.s[j]), 2.0);
    }
}

// Function 121
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

// Function 122
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

// Function 123
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

// Function 124
vec3 sampleHemisphereCosWeighted( in float Xi1, in float Xi2 ) {
    float theta = acos(clamp(sqrt(1.0-Xi1),-1.0, 1.0));
    float phi = TWO_PI * Xi2;
    return sph2cart( 1.0, phi, theta );
}

// Function 125
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

// Function 126
vec2 reverbChannelB(float t) {
	vec2 s = vec2(0);
    
    vec2 reverb = vec2(0);
    float st = 0.02; float iters = 200.;
    for (float i = 0.; i < iters; i++) {
    	reverb += ((makeBells(t - i*st + random(i)*0.02))/iters) *(1. - i/iters) ;
    }
    reverb *= 2.;
    s += makeBells(t)*0.1;
    s += reverb*1.5;
    s *= 2.;
	return s;
}

// Function 127
vec3 sampleWeights(float i) {
	return vec3((1.0 - i) * (1.0 - i), greenWeight() * i * (1.0 - i), i * i);
}

// Function 128
vec3 SampleEnvironment( vec3 vDir )
{
    vec3 vEnvMap = texture(iChannel1, vDir).rgb;
    vEnvMap = vEnvMap * vEnvMap;
    
    float kEnvmapExposure = 0.999;
    vec3 vResult = -log2(1.0 - vEnvMap * kEnvmapExposure);    

    return vResult;
}

// Function 129
vec3 lambertSample(vec3 n,vec2 r) {
    return normalize(n*Epsilon+pointOnSphere(r)); // 1.001 required to avoid NaN
}

// Function 130
float sampleSong(float x) {
    return texture(iChannel1,vec2(x,0.05)).x*0.2;
}

// Function 131
float Sample_Hex( vec2 hex_pos )
{
	return Sample( FromHex( hex_pos ) );
}

// Function 132
float Sample_Uniform(inout float seed) {
    return fract(sin(seed += 0.1)*43758.5453123);
}

// Function 133
float sampleFreq(float freq){
    return texture(iChannel0, vec2(freq, 0.0)).x;
}

// Function 134
vec3 sampleOnATriangle(float r1, float r2, vec3 corner1, vec3 corner2, vec3 corner3 ){
  return (1. - sqrt(r1))*corner1 + (sqrt(r1)*(1. - r2))*corner2
				+ (r2*sqrt(r1)) * corner3;   
}

// Function 135
void _spirv_std_textures_SampledImage_spirv_std_textures_Image2d_sample(out vec4 _569, sampler2D _570, vec3 _571)
{
    _569.x = 0.0;
    _569.y = 0.0;
    _569.z = 0.0;
    _569.w = 0.0;
    _569 = texture(_570, _571.xy);
}

// Function 136
vec3 getHemisphereUniformSample(vec3 n) {
    float cosTheta = getRandom();
    float sinTheta = sqrt(1. - cosTheta * cosTheta);
    
    float phi = 2. * M_PI * getRandom();
    
    // Spherical to cartesian
    vec3 t = normalize(cross(n.yzx, n));
    vec3 b = cross(n, t);
    
	return (t * cos(phi) + b * sin(phi)) * sinTheta + n * cosTheta;
}

// Function 137
float getSampleDim4(int sampleIndex,vec2 fragCoord)
{
	return fract(getDimensionHash(4,fragCoord) + radicalInverse(sampleIndex, 11));
}

// Function 138
vec3 sampleBuff(vec2 uv)
{
    return texture( iChannel0, uv ).xyz;// + vec3(.1); 
}

// Function 139
float min_channel(vec3 v)
{
	float t = (v.x<v.y) ? v.x : v.y;
	t = (t<v.z) ? t : v.z;
	return t;
}

// Function 140
vec3 texsample(const int x, const int y, in vec2 fragCoord)
{
    vec2 uv = fragCoord.xy / iResolution.xy * iChannelResolution[0].xy;
	uv = (uv + vec2(x, y)) / iChannelResolution[0].xy;
	return texture(iChannel0, uv).xyz;
}

// Function 141
void disneyDiffuseSample(out vec3 wi, const in vec3 wo, out float pdf, const in vec2 u, const in vec3 normal, const in MaterialInfo material) {
    cosineSample
}

// Function 142
float getSample(sampler s, float v)
{
    float at = max(0.0, v * float(samples));
    int k = int(at);
    float f = fract(at);
    float a = 0.0;
    for(int i = 0; i < samples + 1; i++)
    {
        if(i == k)
        {
            a = s.s[i];
        }
        else if(i == k + 1)
        {
            return mix(a, s.s[i], smoothstep(0.0, 1.0, f));
        }
    }
    return s.s[samples-1];
}

// Function 143
vec3 sampleLambertian( in vec3 N, in float r1, in float r2, out float pdf ){
    vec3 L = sampleHemisphereCosWeighted( N, r1, r2 );
    pdf = pdfLambertian(N, L);
    return L;
}

// Function 144
float sampleMusic(float f, float bands)
{
	f = floor(f*bands)/bands;
	float fft = texture( iChannel0, vec2(f,0.0) ).x;
	return fft;
}

// Function 145
float getSampleDim3(int sampleIndex,vec2 fragCoord)
{
	return fract(getDimensionHash(3,fragCoord) + radicalInverse(sampleIndex, 7));
}

// Function 146
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

// Function 147
float sampleFreq(float freq) { return texture(iChannel0, vec2(freq, 0.25)).x;}

// Function 148
vec3 sampleChannel0Pixel(vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    return texture(iChannel0, uv).rgb;
}

// Function 149
float SampleDigit(const in float n, const in vec2 vUV)
{
    if( abs(vUV.x-0.5)>0.5 || abs(vUV.y-0.5)>0.5 ) return 0.0;

    // reference P_Malin - https://www.shadertoy.com/view/4sf3RN
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

// Function 150
LuminanceData SampleLuminanceNeighborhood (sampler2D tex2D, vec2 texSize, vec2 uv) {
	LuminanceData l;
	l.m = SampleLuminance(tex2D, uv);
	l.n = SampleLuminance(tex2D, texSize, uv,  0.0f,  1.0f);
	l.e = SampleLuminance(tex2D, texSize, uv,  1.0f,  0.0f);
	l.s = SampleLuminance(tex2D, texSize, uv,  0.0f, -1.0f);
	l.w = SampleLuminance(tex2D, texSize, uv, -1.0f,  0.0f);

	l.ne = SampleLuminance(tex2D, texSize, uv,  1.0f,  1.0f);
	l.nw = SampleLuminance(tex2D, texSize, uv, -1.0f,  1.0f);
	l.se = SampleLuminance(tex2D, texSize, uv,  1.0f, -1.0f);
	l.sw = SampleLuminance(tex2D, texSize, uv, -1.0f, -1.0f);

	l.highest = max(max(max(max(l.n, l.e), l.s), l.w), l.m);
	l.lowest = min(min(min(min(l.n, l.e), l.s), l.w), l.m);
	l.contrast = l.highest - l.lowest;
	return l;
}

// Function 151
vec4 SampleAudioRaw( float f, float t )
{   
    vec4 r = vec4( 0.0 );
    if ( t <= 0.0 )
    {
        //f = f * f;
        /*
        if ( f > 0.0 )
        {
			f = sqrt( f );
        }*/
        
        r = textureLod( iChannelAudio, vec2(f, 0.0), 0.0);
        
        
        float a = r.r;
        
        float shade = a * (0.75 + f * 0.25);

        shade = pow( shade, 10.0 ) * 50.0;

        r.r = shade;        
        r.g = f * 30.0 + iTime * 5.0;
        r.a = a * a;               
    }
    else
    {
    	r = textureLod( iChannelAudioHistory, vec2(f, t), 0.0);
    }
    
    return r;
}

// Function 152
float doChannel1( float t )
{
    float x = t;
    float y = 0.0;
    float b = 0.0;
     
    D(1280.)D(1340.)D(3076.)D(1536.)D(1536.)D(4608.)D(1536.)
    y += instrument2(369.99, tint*(t-x) );

    x = t; b = 0.0;
    D(4160.)D(6144.)D(1536.)D(4608.)
    y += instrument2(392.0, tint*(t-x) );

    x = t; b = 0.0;
    D(2816.)D(3072.)D(1536.)D(1536.)D(4608.)D(1536.)
    y += instrument2(440.0, tint*(t-x) );

    x = t; b = 0.0;
    D(4352.)D(6144.)D(1536.)D(4608.)
    y += instrument2(493.88, tint*(t-x) );
    
    x = t; b = 0.0;
    D(3008.)D(4608.)D(1536.)D(4608.)D(1536.)
    y += instrument2(554.37, tint*(t-x) );

    x = t; b = 0.0;
    D(1472.)D(3072.)D(1536.)D(4608.)D(1536.)D(4608.)D(14400.)D(384.)
    y += instrument2(587.33, tint*(t-x) );

    x = t; b = 0.0;
    D(31424.)
    y += instrument2(659.26, tint*(t-x) );

    x = t; b = 0.0;
    D(19904.)D(192.)D(10944.)
    y += instrument2(739.99, tint*(t-x) );

    x = t; b = 0.0;
    D(28160.)D(768.)D(1920.)
    y += instrument2(783.99, tint*(t-x) );

    x = t; b = 0.0;
    D(19328.)D(6144.)D(2880.)D(384.)
    y += instrument2(880.0, tint*(t-x) );

    x = t; b = 0.0;
    D(18944.)D(3068.)D(3076.)D(3456.)
    y += instrument2(987.77, tint*(t-x) );

    x = t; b = 0.0;
    D(21632.)D(768.)D(2304.)
    y += instrument2(1108.73, tint*(t-x) );

    x = t; b = 0.0;
    D(18176.)D(384.)D(2688.)D(3072.)D(4800.)
    y += instrument2(1174.66, tint*(t-x) );

    x = t; b = 0.0;
    D(26216.)D(24.)
    y += instrument2(1318.51, tint*(t-x) );

    x = t; b = 0.0;
    D(22976.)D(192.)
    y += instrument2(1479.98, tint*(t-x) );

    x = t; b = 0.0;
    D(896.)
    y += instrument2(246.94, tint*(t-x) );

    x = t; b = 0.0;
    D(2432.)D(4608.)D(1536.)D(4608.)D(1536.)
    y += instrument2(277.18, tint*(t-x) );

    x = t; b = 0.0;
    D(1088.)D(2880.)D(1536.)D(4608.)D(1536.)D(4608.)
    y += instrument2(293.66, tint*(t-x) );

    return y;
}

// Function 153
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

// Function 154
vec3 sampletex( vec2 uv )
{
    float t = fract( 0.1*iTime );
    if ( t < 1.0/3.0)
    	return srgb2lin( texture( iChannel0, uv, -10.0 ).rgb );
    else if ( t < 2.0/3.0 )
        return srgb2lin( texture( iChannel1, uv, -10.0 ).rgb );
    else
	    return srgb2lin( texture( iChannel2, uv, -10.0 ).rgb );    
}

// Function 155
float getSampleDim0(int sampleIndex,vec2 fragCoord)
{
	return fract(getDimensionHash(0,fragCoord) + radicalInverse(sampleIndex, 2));
}

// Function 156
vec3 sampleTexture( in vec3 uvw, in vec3 nor, in float mid )
{
    return mytexture( uvw, nor, mid );
}

// Function 157
vec3 sampleNormal( vec3 p ) {
    vec3 eps = vec3(1e-3,0.,0.);
    float dx = world(p+eps).x-world(p-eps).x;
    float dy = world(p+eps.yxy).x-world(p-eps.yxy).x;
    float dz = world(p+eps.yyx).x-world(p-eps.yyx).x;
    return normalize(vec3(dx,dy,dz));
}

// Function 158
float doChannel3( float t )
{
  float b = 0.0;
  float n = 0.0;
  float x = 0.0;
  t /= tint;
  D( 0,50)D( 3,50)D( 6,50)D( 6,50)D( 3,50)D( 6,67)D(12,55)D(12,55)D( 9,52)D( 9,48)
  D( 9,53)D( 6,55)D( 6,54)D( 3,53)D( 6,52)D( 4,60)D( 4,64)D( 4,65)D( 6,62)D( 3,64)
  D( 6,60)D( 6,57)D( 3,59)D( 3,55)D( 9,55)D( 9,52)D( 9,48)D( 9,53)D( 6,55)D( 6,54)
  D( 3,53)D( 6,52)D( 4,60)D( 4,64)D( 4,65)D( 6,62)D( 3,64)D( 6,60)D( 6,57)D( 3,59)
  D( 3,55)D( 9,48)D( 9,55)D( 9,60)D( 6,53)D( 9,60)D( 3,60)D( 6,53)D( 6,48)D( 9,52)
  D( 9,55)D( 3,60)D( 6,79)D( 6,79)D( 3,79)D( 6,55)D( 6,48)D( 9,55)D( 9,60)D( 6,53)
  D( 9,60)D( 3,60)D( 6,53)D( 6,48)D( 6,56)D( 9,58)D( 9,60)D( 9,55)D( 3,55)D( 6,48)
  D( 6,48)D( 9,55)D( 9,60)D( 6,53)D( 9,60)D( 3,60)D( 6,53)D( 6,48)D( 9,52)D( 9,55)
  D( 3,60)D( 6,79)D( 6,79)D( 3,79)D( 6,55)D( 6,48)D( 9,55)D( 9,60)D( 6,53)D( 9,60)
  D( 3,60)D( 6,53)D( 6,48)D( 6,56)D( 9,58)D( 9,60)D( 9,55)D( 3,55)D( 6,48)D( 6,44)
  D( 9,51)D( 9,56)D( 6,55)D( 9,48)D( 9,43)D( 6,44)D( 9,51)D( 9,56)D( 6,55)D( 9,48)
  D( 9,43)D( 6,44)D( 9,51)D( 9,56)D( 6,55)D( 9,48)D( 9,43)D( 6,50)D( 3,50)D( 6,50)
  D( 6,50)D( 3,50)D( 6,67)D(12,55)D(12,55)D( 9,52)D( 9,48)D( 9,53)D( 6,55)D( 6,54)
  D( 3,53)D( 6,52)D( 4,60)D( 4,64)D( 4,65)D( 6,62)D( 3,64)D( 6,60)D( 6,57)D( 3,59)
  D( 3,55)D( 9,55)D( 9,52)D( 9,48)D( 9,53)D( 6,55)D( 6,54)D( 3,53)D( 6,52)D( 4,60)
  D( 4,64)D( 4,65)D( 6,62)D( 3,64)D( 6,60)D( 6,57)D( 3,59)D( 3,55)D( 9,48)D( 9,54)
  D( 3,55)D( 6,60)D( 6,53)D( 6,53)D( 6,60)D( 3,60)D( 3,53)D( 6,50)D( 9,53)D( 3,55)
  D( 6,59)D( 6,55)D( 6,55)D( 6,60)D( 3,60)D( 3,55)D( 6,48)D( 9,54)D( 3,55)D( 6,60)
  D( 6,53)D( 6,53)D( 6,60)D( 3,60)D( 3,53)D( 6,55)D( 3,55)D( 6,55)D( 3,55)D( 4,57)
  D( 4,59)D( 4,60)D( 6,55)D( 6,48)D(12,48)D( 9,54)D( 3,55)D( 6,60)D( 6,53)D( 6,53)
  D( 6,60)D( 3,60)D( 3,53)D( 6,50)
  return instr3( note2freq( n ), tint*(t-x) );
}

// Function 159
float sampleWt( float wt, bool even )
{
    return even ? (2.-wt) : wt;
}

// Function 160
vec2 echoChannel(float t) {
	vec2 s = vec2(0);

    float fb = 0.55, tm = qbeat*1.5, cf = 0.9, ct = tm;
    // tap 1 
    s += makeAmb(t) * cf; cf *= fb; 
    // tap 2
    s += makeAmb(t - ct) * cf; cf *= fb; ct += tm;
    // tap 3
    s += makeAmb(t - ct) * cf; cf *= fb; ct += tm;
    // tap 4
    s += makeAmb(t - ct) * cf; cf *= fb; ct += tm;
    // tap 5
    s += makeAmb(t - ct) * cf; cf *= fb; ct += tm;
    
    return s;
}

// Function 161
vec4 SampleAudio( float f, float t, float dt )
{
    vec4 sp = SampleAudioRaw( f, t );

    float gradSampleDist = 0.01;
    
    float xofs = SampleAudioRaw( f - gradSampleDist, t ).a - SampleAudioRaw( f + gradSampleDist, t ).a;
    
    vec2 d = vec2( xofs * 5.0, -1.0 ) * dt;
    
    float speed = sp.a * 5.0 * (3.0 * f + 1.4);
    d = normalize(d) * dt * speed;
    f += d.x;
    t += d.y;
    
    t -= dt * 0.1;
    
    //t += -dt * (0.2 + f * f * f * 2.0);
    
    vec4 result = SampleAudioRaw( f, t );
    
    
    float fSpread = 0.005; //* (1.0 - sp.a* sp.a);
    result *= 0.8;
    result += SampleAudioRaw( f - fSpread, t ) * 0.1;
    result += SampleAudioRaw( f + fSpread, t ) * 0.1;
    
    return result;
}

// Function 162
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

// Function 163
float getAudioScalar(float t)
{

    float audioScalar =  10. *  g_audioFreqs[0] * sin(-2.  * 
        (g_beatRate * g_time - PI_OVER_TWO * t));

    audioScalar +=       5. *  g_audioFreqs[1] * sin(-4.  * 
        (g_beatRate * g_time - PI_OVER_TWO * t));

    audioScalar +=       5. *  g_audioFreqs[2] * sin(-8.  * 
        (g_beatRate * g_time - PI_OVER_TWO * t));

    audioScalar +=       5. *  g_audioFreqs[3] * sin(-16. * 
        (g_beatRate * g_time - PI_OVER_TWO * t));

    return (audioScalar + 10.)/20.;
}

// Function 164
vec4 GetSamplePosVR(vec3 origin, vec3 dir)
{
    dir.x = -dir.x;
    rd = normalize(dir) * 9.;
    rd = Rotate(BuildQuat(vec3(0,1,0),3.1415),rd);
    
    vec3 lro = vec3(0.,0.,-.1);
    vec3 lrd = rd;
    
    ro = camPos ;
    rd = Rotate(camQuat,rd);
    
    
    vec3 mp=ro;
    lmp=lro;
    
    int i;
    for (i=0;i<maxStepRayMarching;i++){
        dist = map(mp,lmp+dpos*0.005,time-(length(ro-mp)/speedOfLight));
        //if(abs(rayDist)<mix(0.0001,0.1,(mp.z+camDist)*0.005))
        if(dist<0.0001)
            break;
        mp+=rd*dist;
        lmp+=lrd*dist;
    }
    
    float ma=1.-float(i)/80.;
    
    return vec4(mp,ma);
}

// Function 165
float sampleMusicA() {
	return 0.5 * (
		texture( iChannel0, vec2( 0.15, 0.25 ) ).x + 
		texture( iChannel0, vec2( 0.30, 0.25 ) ).x);
}

// Function 166
vec4 sample3D(sampler2D tex, vec3 uvw, vec3 vres)
{
    uvw = mod(floor(uvw * vres), vres);
    
    //XYZ -> Pixel index
    float idx = (uvw.z * (vres.x*vres.y)) + (uvw.y * vres.x) + uvw.x;
    
    //Pixel index -> Buffer uv coords
    vec2 uv = vec2(mod(idx, iResolution.x), floor(idx / iResolution.x));
    
    return textureLod(tex, (uv + 0.5) / iResolution.xy, 0.0);
}

// Function 167
vec3 sampletex( vec2 uv )
{
    #ifdef SRGBLIN
    	return srgb2lin( texture( iChannel0, uv, -10.0 ).rgb );
    #else
    	return  texture( iChannel0, uv, -10.0 ).rgb ;
    #endif
}

// Function 168
vec4 SampleBicubic2(sampler2D channel, vec2 uv)
{
    uv -= vec2(0.5);
    
    vec2 uvi = floor(uv);
    vec2 uvf = uv - uvi;

    ivec2 uv0 = ivec2(uvi);
    
    vec4 r = vec4(0.0);
    for(int j = 0; j < 2; ++j)
    for(int i = 0; i < 2; ++i)
    {
        vec4 c = texelFetch(channel, uv0 + ivec2(i, j), 0);
        
        vec2 l = uvf;
        
        if(i != 0) l.x -= 1.0;
        if(j != 0) l.y -= 1.0;
        
        r += kern4x4(l) * c;
    }
    
    // r = vec4(df/dx, df/dy, ddf/dxy, f)
	return r;
}

// Function 169
float doChannel1( float t )
{
  float b = 0.0;
  float n = 0.0;
  float x = 0.0;
  t /= tint;
  D( 0,76)D( 3,76)D( 6,76)D( 6,72)D( 3,76)D( 6,79)D(24,72)D( 9,67)D( 9,64)D( 9,69)
  D( 6,71)D( 6,70)D( 3,69)D( 6,67)D( 4,76)D( 4,79)D( 4,81)D( 6,77)D( 3,79)D( 6,76)
  D( 6,72)D( 3,74)D( 3,71)D( 9,72)D( 9,67)D( 9,64)D( 9,69)D( 6,71)D( 6,70)D( 3,69)
  D( 6,67)D( 4,76)D( 4,79)D( 4,81)D( 6,77)D( 3,79)D( 6,76)D( 6,72)D( 3,74)D( 3,71)
  D(15,79)D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,68)D( 3,69)D( 3,72)D( 6,69)D( 3,72)
  D( 3,74)D( 9,79)D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,84)D( 6,84)D( 3,84)D(18,79)
  D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,68)D( 3,69)D( 3,72)D( 6,69)D( 3,72)D( 3,74)
  D( 9,75)D( 9,74)D( 9,72)D(30,79)D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,68)D( 3,69)
  D( 3,72)D( 6,69)D( 3,72)D( 3,74)D( 9,79)D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,84)
  D( 6,84)D( 3,84)D(18,79)D( 3,78)D( 3,77)D( 3,75)D( 6,76)D( 6,68)D( 3,69)D( 3,72)
  D( 6,69)D( 3,72)D( 3,74)D( 9,75)D( 9,74)D( 9,72)D(24,72)D( 3,72)D( 6,72)D( 6,72)
  D( 3,74)D( 6,76)D( 3,72)D( 6,69)D( 3,67)D(12,72)D( 3,72)D( 6,72)D( 6,72)D( 3,74)
  D( 3,76)D(27,72)D( 3,72)D( 6,72)D( 6,72)D( 3,74)D( 6,76)D( 3,72)D( 6,69)D( 3,67)
  D(12,76)D( 3,76)D( 6,76)D( 6,72)D( 3,76)D( 6,79)D(24,72)D( 9,67)D( 9,64)D( 9,69)
  D( 6,71)D( 6,70)D( 3,69)D( 6,67)D( 4,76)D( 4,79)D( 4,81)D( 6,77)D( 3,79)D( 6,76)
  D( 6,72)D( 3,74)D( 3,71)D( 9,72)D( 9,67)D( 9,64)D( 9,69)D( 6,71)D( 6,70)D( 3,69)
  D( 6,67)D( 4,76)D( 4,79)D( 4,81)D( 6,77)D( 3,79)D( 6,76)D( 6,72)D( 3,74)D( 3,71)
  D( 9,76)D( 3,72)D( 6,67)D( 9,68)D( 6,69)D( 3,77)D( 6,77)D( 3,69)D(12,71)D( 4,81)
  D( 4,81)D( 4,81)D( 4,79)D( 4,77)D( 4,76)D( 3,72)D( 6,69)D( 3,67)D(12,76)D( 3,72)
  D( 6,67)D( 9,68)D( 6,69)D( 3,77)D( 6,77)D( 3,69)D(12,71)D( 3,77)D( 6,77)D( 3,77)
  D( 4,76)D( 4,74)D( 4,72)D(24,76)D( 3,72)D( 6,67)D( 9,68)D( 6,69)D( 3,77)D( 6,77)
  D( 3,69)D(12,71)
  return instr1( note2freq( n ), tint*(t-x) );
}

// Function 170
vec2 xsample(int i, float t)
{
    float p = t * INT_PER_STEP * STEP;
    int   ip = int(p);
    vec2  s = vec2(0.0, 0.0);

    if (i == 2)
    {
        s.x = clamp(15.0 - p, 0.0, 15.0);
        s.y = ip == 0 ? 1.0 : 0.0;
    }
    
    if (i == 11)
    {
        s.x = clamp(15.0 - p, 0.0, 15.0);
    }

    if (i == 4)
    {
        s.x = clamp(16.0 - p, 9.0, 15.0);
    }

    if (i == 5)
    {
        s.x = clamp(14.5 - p * 0.5, 12.0, 14.0);
        s.x = mix(s.x,
            clamp(s.x + sin(p * 3.1415 * 0.4) * 0.4, 0.0, 15.0),
            clamp(p - 17.0, 0.0, 1.0));

        s.y = ip == 0 ? 1.0 : 0.0;
    }

    if (i == 7)
    {
        s.x = clamp(16.0 - p * 0.85, 0.0, 15.0);
        if (p > 11.0) s.x = clamp(10.0 - abs(14.0 - p), 0.0, 15.0);
        s.y = ip == 0 ? 1.0 : 0.0;
    }

    if (i == 15)
    {
        s.x = clamp(17.0 - p * 2.2, 0.0, 15.0);
        s.x = p > 5.0 ? 0.0 : s.x;
    }

    if (i == 16)
    {
        s.x = clamp(15.5 - p * 0.5, 0.0, 15.0);
        s.x = p > 10.0 ? 0.0 : s.x;
        s.y = 1.0;
    }

    return s;
}

// Function 171
float doChannel3( float t )
{
    float x = t;
    float y = 0.0;
    float b = 0.0;
    
    D(2240.)
    y += instrument(329.63, tint*(t-x) );

    x = t; b = 0.0;
    D(936.)D(1112.)D(3456.)D(7680.)
    y += instrument(369.99, tint*(t-x) );

    x = t; b = 0.0;
    D(3968.)
    y += instrument(392.0, tint*(t-x) );

    x = t; b = 0.0;
    D(22400.)
    y += instrument(92.5, tint*(t-x) );

    x = t; b = 0.0;
    D(22016.)
    y += instrument(98.0, tint*(t-x) );

    x = t; b = 0.0;
    D(23168.)
    y += instrument(103.83, tint*(t-x) );

    x = t; b = 0.0;
    D(20864.)D(2688.)D(3456.)
    y += instrument(110.0, tint*(t-x) );

    x = t; b = 0.0;
    D(21248.)D(2688.)D(576.)
    y += instrument(123.47, tint*(t-x) );

    x = t; b = 0.0;
    D(24704.)
    y += instrument(138.59, tint*(t-x) );

    x = t; b = 0.0;
    D(10112.)D(10368.)D(4608.)
    y += instrument(146.83, tint*(t-x) );

    x = t; b = 0.0;
    D(25472.)
    y += instrument(164.81, tint*(t-x) );
    
    x = t; b = 0.0;
    D(20096.)
    y += instrument(185.0, tint*(t-x) );
    
    x = t; b = 0.0;
    D(9344.)D(576.)D(5568.)
    y += instrument(220.0, tint*(t-x) );
    
    x = t; b = 0.0;
    D(8576.)D(1152.)D(4992.)D(1152.)
    y += instrument(246.94, tint*(t-x) );
    
    x = t; b = 0.0;
    D(2432.)D(5376.)D(6144.)D(2112.)
    y += instrument(277.18, tint*(t-x) );

    x = t; b = 0.0;
    D(16256.)
    y += instrument(293.66, tint*(t-x) );

    return y;
}

// Function 172
vec4 Sample (sampler2D  tex2D, vec2 uv) {
	return texture(tex2D, uv);
}

// Function 173
float SampleWaterFoam( vec2 vUV, vec2 vFlowOffset, float fFoam )
{
    float f =  FBM_DXY(vUV * 30.0, vFlowOffset * 50.0, 0.8, -0.5 ).z;
    float fAmount = 0.2;
    f = max( 0.0, (f - fAmount) / fAmount );
    return pow( 0.5, f );
}

// Function 174
void sampleCamera(vec2 fragCoord, vec2 u, out vec3 rayOrigin, out vec3 rayDir)
{
	vec2 filmUv = (fragCoord.xy + u)/iResolution.xy;
	
	float tx = (2.0*filmUv.x - 1.0)*(iResolution.x/iResolution.y);
	float ty = (1.0 - 2.0*filmUv.y);
	float tz = 0.0;
	
	rayOrigin = vec3(0.0, 0.0, 5.0);
	rayDir = normalize(vec3(tx, ty, tz) - rayOrigin);
}

// Function 175
float pomSample( in sampler2D t, in vec2 uv )
{
    float r = texture(t, uv*POMSCALE).r;
    return r*r*VOLHEIGHT;
}

// Function 176
float GetSample(float time, float freq, float brightness, float sineMix) {
    float modAmount = 0.15 * brightness + 0.05;
    float modFreq = freq * 1.49;
    float phaseOffset = modAmount * Sine(modFreq * time);
    float triSample = Triangle(freq * time + phaseOffset);
    float sineSample = Sine(freq * time + phaseOffset);
    float amplitude = 0.3 + 0.7 * brightness;
    return mix(triSample, sineSample, sineMix) * amplitude;
}

// Function 177
vec4 sample_lens(sampler2D tex, vec2 coords){
	/* Sample the light at a particular point inside the lens
	This includes all lens artifacts such as vignetting, 
	chromatic aberation, lens distortion, and in the future jello. */
	vec4 outp;
	float r2 = dot(coords, coords);  // u.u = |u|^2
	float r4 = r2 * r2;
	float r6 = r4 * r2;
	vec3 r2r4r6 = vec3(r2, r4, r6);
	
	float vignette_factor = dot(lens_vignette, r2r4r6);
	
	// TODO: add jello
	
	if (length(lens_chromatic_aberation) == 0.0){
		outp = sample_photons(tex, lens_distort_coords(coords));
	} else {
		float abr = dot(lens_chromatic_aberation, r2r4r6);// * rand(UV);
		outp = vec4(
			sample_photons(tex, lens_distort_coords(coords*(1.0 - abr*3.0))).r,
			sample_photons(tex, lens_distort_coords(coords*(1.0 - abr*2.0))).g,
			sample_photons(tex, lens_distort_coords(coords*(1.0 - abr*1.0))).b,
			1.0
		);
	}
	outp.rgb *= (1.0 + vignette_factor);
	return outp;
}

// Function 178
vec4 sampleSelf(vec2 pos)
{
    return textureLod(iChannel2, pos / iChannelResolution[2].xy, 0.0);
}

// Function 179
DistSample MakeDistSample(float d, float s, float m)
{
    DistSample h;
    h.dist = d;
    h.stepRatio = s;
    h.material = m;
    return h;
}

// Function 180
float sample_wavelength(void) {
    return (float(spectrum_width))*sample_uniform() + float(spectrum_start);
}

// Function 181
vec3 sample_grad_dist(vec2 uv, float font_size) {
    
    vec3 grad_dist = (textureLod(iChannel0, uv, 0.).yzw - TEX_BIAS) * font_size;

    grad_dist.y = -grad_dist.y;
    grad_dist.xy = normalize(grad_dist.xy + 1e-5);
    
    return grad_dist;
    
}

// Function 182
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

// Function 183
float sampleDigit( const in float n, const in vec2 vUV )
{		
	if ( vUV.x  < 0.0 ) return 0.0;
	if ( vUV.y  < 0.0 ) return 0.0;
	if ( vUV.x >= 1.0 ) return 0.0;
	if ( vUV.y >= 1.0 ) return 0.0;
	
	float data = 0.0;
	
	     if( n < 0.5 ) data = 7.0 + 5.0 * 16.0 + 5.0 * 256.0 + 5.0 * 4096.0 + 7.0 * 65536.0;
	else if( n < 1.5 ) data = 2.0 + 2.0 * 16.0 + 2.0 * 256.0 + 2.0 * 4096.0 + 2.0 * 65536.0;
	else if( n < 2.5 ) data = 7.0 + 1.0 * 16.0 + 7.0 * 256.0 + 4.0 * 4096.0 + 7.0 * 65536.0;
	else if( n < 3.5 ) data = 7.0 + 4.0 * 16.0 + 7.0 * 256.0 + 4.0 * 4096.0 + 7.0 * 65536.0;
	else if( n < 4.5 ) data = 4.0 + 7.0 * 16.0 + 5.0 * 256.0 + 1.0 * 4096.0 + 1.0 * 65536.0;
	else if( n < 5.5 ) data = 7.0 + 4.0 * 16.0 + 7.0 * 256.0 + 1.0 * 4096.0 + 7.0 * 65536.0;
	else if( n < 6.5 ) data = 7.0 + 5.0 * 16.0 + 7.0 * 256.0 + 1.0 * 4096.0 + 7.0 * 65536.0;
	else if( n < 7.5 ) data = 4.0 + 4.0 * 16.0 + 4.0 * 256.0 + 4.0 * 4096.0 + 7.0 * 65536.0;
	else if( n < 8.5 ) data = 7.0 + 5.0 * 16.0 + 7.0 * 256.0 + 5.0 * 4096.0 + 7.0 * 65536.0;
	else if( n < 9.5 ) data = 7.0 + 4.0 * 16.0 + 7.0 * 256.0 + 5.0 * 4096.0 + 7.0 * 65536.0;
	
	vec2 vPixel = floor( vUV * vec2( 4.0, 5.0 ) );
	float fIndex = vPixel.x + ( vPixel.y * 4.0 );
	
	return mod( floor( data / pow( 2.0, fIndex ) ), 2.0 );
}

// Function 184
vec3 texCubeSampleWeights(float i) {
	return vec3((1.0 - i) * (1.0 - i), 2.0 * i * (1.0 - i), i * i);
}

// Function 185
void downSampled(out sampler res)
{
    
    const int steps = downSampleSteps;
    const int sps = steps / samplesAudio;
    res;
    float s = 1.0 / float(steps);
    for(int j = 0; j < samplesAudio; j++)
    {
        for(int i = 0; i < sps; i++)
        {
            res.s[j] += texture(iChannel0, vec2(float(j * sps + i) * s, 0.25)).x;
            
        }
        float n = sqrt(float(j) / float(samplesAudio));
        float k = (1.0 - bassMalus) + n * bassMalus;
        res.s[j] = (res.s[j] / float(sps)) * k;
        if(res.s[j] < volMin)
            volMin = res.s[j];
        if(res.s[j] > volMax)
            volMax = res.s[j];
    }
    volMin = max(volLo, volMin);
    volMax = max(volHi, volMax);
    for(int j = 0; j < samplesAudio; j++)
    {
        res.s[j] = pow(smoothstep(volMin, volMax, res.s[j]), 2.0);
    }
}

// Function 186
vec4 audioEq() {
    float vol = 0.0;
    
    // bass
    float lows = 0.0;
    for(float i=0.;i<85.; i++){
        float v =  texture(iChannel0, vec2(i/85., 0.0)).x;
        lows += v*v;
        vol += v*v;
    }
    lows /= 85.0;
    lows = sqrt(lows);
    
    // mids
    float mids = 0.0;
    for(float i=85.;i<255.; i++){
        float v =  texture(iChannel0, vec2(i/170., 0.0)).x;
        mids += v*v;
        vol += v*v;
    }
    mids /= 170.0;
    mids = sqrt(mids);
    
    // treb
    float highs = 0.0;
    for(float i=255.;i<512.; i++){
        float v =  texture(iChannel0, vec2(i/255., 0.0)).x;
        highs += v*v;
        vol += v*v;
    }
    highs /= 255.0;
    highs = sqrt(highs);
    
    vol /= 512.;
    vol = sqrt(vol);
    
    return vec4( lows * 1.5, mids * 1.25, highs * 1.0, vol ); // bass, mids, treb, volume
}

// Function 187
float getSampleDim2(int sampleIndex,vec2 fragCoord)
{
	return fract(getDimensionHash(2,fragCoord) + radicalInverse(sampleIndex, 5));
}

// Function 188
float SampleLuminance (sampler2D tex2D, vec2 texSize, vec2 uv, float uOffset, float vOffset) {
	uv += texSize * vec2(uOffset, vOffset);
	return SampleLuminance(tex2D, uv);
}

// Function 189
vec4 sample_lpv_trilin(vec3 p,float c)
{p=clamp(p*lpvsize-.5,vec3(.5),lpvsize-.5)
;vec3 w=fract(p),q=1.-w
;vec2 e=vec2(0,1),h=vec2(q.x,w.x)
;vec4 k=vec4(h*q.y,h*w.y)
,s=k*q.z,t=k*w.z
,p000=fetch(p+e.xxx,c)
,p001=fetch(p+e.xxy,c)
,p010=fetch(p+e.xyx,c)
,p011=fetch(p+e.xyy,c)
,p100=fetch(p+e.yxx,c)
,p101=fetch(p+e.yxy,c)
,p110=fetch(p+e.yyx,c)
,p111=fetch(p+e.yyy,c)
;return p000*s.x+p100*s.y+p010*s.z+p110*s.w
       +p001*t.x+p101*t.y+p011*t.z+p111*t.w;}

// Function 190
vec4 sample_map(sampler2D map_channel, vec2 map_channel_resolution, vec2 world_coords) {
    vec2 uv = world_coords;
    uv.x *= MAP_SIZE.y / MAP_SIZE.x;
    
    uv /= MAP_SCALE;
    uv = uv * 0.5 + 0.5;
    
    if (any(lessThan(uv, vec2(0.0))) || any( greaterThan(uv, vec2(1.0)))) {
        return vec4(1.0);
    }
    
    uv = uv * MAP_SIZE / map_channel_resolution;
    
    vec4 raw = texture(map_channel, uv);
    raw.xyzw -= 0.5;
    
    raw.w *= 3.14 * 4.0;
    
    return raw;
}

// Function 191
void importanceSamplePath(vec3 r0, vec3 r1, out vec3 p0, out vec3 p1)
{
    float score[LASER_PATH];
    float totalscore = 0.0;
    vec3 P0, P1;
    for(int i = 0; i < cpath; i++)
    {
        getClosestPointPair(r0,r1,path[i],path[i+1],P0,P1);
        float s = distance(path[i],path[i+1])/distance(P0,P1);
        totalscore += s;
        score[i] = totalscore;
    }
    
    //target score
    float rscore = rand()*totalscore;
    
    for(int i = 0; i < cpath; i++)
    {
        if(rscore < score[i]) //found score
        {
            p0 = path[i];
            p1 = path[i+1];
            return;
        }
    }
}

// Function 192
vec4 SampleBicubic2(vec2 p)
{
    vec2 p0 = floor(p);
    vec2 l  = p - p0;
    
    vec4 r = vec4(0.0);
	for (float y = 0.0; y < 2.0; ++y)
	for (float x = 0.0; x < 2.0; ++x)
    {
        vec4 n = Map2(p0 + vec2(x, y));
        
        r += kern4x4(l - vec2(x, y)) * n;
    }
    
    return r;
}

// Function 193
vec2 SampleAxes()
{
    vec2 axes;
    axes.x = SampleKey(kKeyRight) - SampleKey(kKeyLeft);
    axes.y = (axes.x == 0.0)? SampleKey(kKeyUp) - SampleKey(kKeyDown) : 0.0;
    return axes;
}

// Function 194
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

// Function 195
vec2 Sample(inout vec2 r)
{
    r = fract(r * vec2(33.3983, 43.4427));
    return r-.5;
    //return sqrt(r.x+.001) * vec2(sin(r.y * TAU), cos(r.y * TAU))*.5; // <<=== circular sampling.
}

// Function 196
Sample sample_mirror(HitQuery hit, Ray ray) {
    
    
    vec2 wi = -ray.dir;
    vec2 normal = hit.normal;
    wi = world2local(normal, wi);
    float cos_theta_i = wi.y;
    
    cos_theta_i = abs(cos_theta_i);
    
    vec2 wo = vec2(-wi.x, wi.y);
    float pdf = 1.0;
    float contrib = 1.0 / cos_theta_i;
    vec2 pert = eps * hit.normal;

    
    wo = local2world(normal, wo);
    
    return Sample(Ray(hit.p + pert, wo, ray.wavelength), contrib, pdf, cos_theta_i);
}

// Function 197
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

// Function 198
void sampleUniform(
	float u,
	float maxDistance,
	out float dist,
	out float pdf)
{
	dist = u*maxDistance;
	pdf = 1.0/maxDistance;
}

// Function 199
vec3 importanceSampleSegmentPoint(vec3 r1, vec3 r2, vec3 x)
{
    vec3 x0 = r1 - x;
    vec3 rd = normalize(r2 - r1);
    float l = length(r2 - r1);
    float a = dot(rd, x0);
    float x02 = dot(x0, x0);
    float sq = sqrt(x02 - a*a); 
    float t = sq*tan(rand(atan(a/sq),atan((l + a)/sq))) - a;
    return r1 + rd*t; //importance sampled point
}

// Function 200
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

// Function 201
vec4 SampleMip3(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*8.,sp.z+48.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(8.,0.))*IRES),fract(sp.y));
}

// Function 202
vec4 SampleBicubic3(vec2 p, out vec4 d2)
{
    vec2 p0 = floor(p);
    vec2 l  = p - p0;
    
    vec4 r = vec4(0.0);
	for (float y = 0.0; y < 2.0; ++y)
	for (float x = 0.0; x < 2.0; ++x)
    {
        vec4 n = Map2(p0 + vec2(x, y));
        
        mat4 mA, mB;
        kern4x4(l - vec2(x, y), /*out*/ mA, mB);
        
        r  += mA * n;
        d2 += mB * n;
    }
    
    return r;
}

// Function 203
vec2 Sample_Uniform2(inout float seed) {
    return fract(sin(vec2(seed+=0.1,seed+=0.1))*
                vec2(43758.5453123,22578.1459123));
}

// Function 204
float doChannel2( float t )
{
  float b = 0.0;
  float n = 0.0;
  float x = 0.0;
  t /= tint;
  D( 48,0)
  D( 0,66)D( 3,66)D( 6,66)D( 6,66)D( 3,66)D( 6,71)D(12,67)D(12,64)D( 9,60)D( 9,55)
  D( 9,60)D( 6,62)D( 6,61)D( 3,60)D( 6,60)D( 4,67)D( 4,71)D( 4,72)D( 6,69)D( 3,71)
  D( 6,69)D( 6,64)D( 3,65)D( 3,62)D( 9,64)D( 9,60)D( 9,55)D( 9,60)D( 6,62)D( 6,61)
  D( 3,60)D( 6,60)D( 4,67)D( 4,71)D( 4,72)D( 6,69)D( 3,71)D( 6,69)D( 6,64)D( 3,65)
  D( 3,62)D(15,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)D( 6,64)D( 3,65)D( 3,67)D( 6,60)
  D( 3,64)D( 3,65)D( 9,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)D( 6,77)D( 6,77)D( 3,77)
  D(18,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)D( 6,64)D( 3,65)D( 3,67)D( 6,60)D( 3,64)
  D( 3,65)D( 9,68)D( 9,65)D( 9,64)D(30,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)D( 6,64)
  D( 3,65)D( 3,67)D( 6,60)D( 3,64)D( 3,65)D( 9,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)
  D( 6,77)D( 6,77)D( 3,77)D(18,76)D( 3,75)D( 3,74)D( 3,71)D( 6,72)D( 6,64)D( 3,65)
  D( 3,67)D( 6,60)D( 3,64)D( 3,65)D( 9,68)D( 9,65)D( 9,64)D(24,68)D( 3,68)D( 6,68)
  D( 6,68)D( 3,70)D( 6,67)D( 3,64)D( 6,64)D( 3,60)D(12,68)D( 3,68)D( 6,68)D( 6,68)
  D( 3,70)D( 3,67)D(27,68)D( 3,68)D( 6,68)D( 6,68)D( 3,70)D( 6,67)D( 3,64)D( 6,64)
  D( 3,60)D(12,66)D( 3,66)D( 6,66)D( 6,66)D( 3,66)D( 6,71)D(12,67)D(12,64)D( 9,60)
  D( 9,55)D( 9,60)D( 6,62)D( 6,61)D( 3,60)D( 6,60)D( 4,67)D( 4,71)D( 4,72)D( 6,69)
  D( 3,71)D( 6,69)D( 6,64)D( 3,65)D( 3,62)D( 9,64)D( 9,60)D( 9,55)D( 9,60)D( 6,62)
  D( 6,61)D( 3,60)D( 6,60)D( 4,67)D( 4,71)D( 4,72)D( 6,69)D( 3,71)D( 6,69)D( 6,64)
  D( 3,65)D( 3,62)D( 9,72)D( 3,69)D( 6,64)D( 9,64)D( 6,65)D( 3,72)D( 6,72)D( 3,65)
  D(12,67)D( 4,77)D( 4,77)D( 4,77)D( 4,76)D( 4,74)D( 4,72)D( 3,69)D( 6,65)D( 3,64)
  D(12,72)D( 3,69)D( 6,64)D( 9,64)D( 6,65)D( 3,72)D( 6,72)D( 3,65)D(12,67)D( 3,74)
  D( 6,74)D( 3,74)D( 4,72)D( 4,71)D( 4,67)D( 3,64)D( 6,64)D( 3,60)D(12,72)D( 3,69)
  D( 6,64)D( 9,64)D( 6,65)D( 3,72)D( 6,72)D( 3,65)D(12,67) 
  return instr2( note2freq( n ), tint*(t-x) );
}

// Function 205
float GetAudio(float fTime)
{
    float fPhase = GetPhase(fTime).x;
	
    float fLoadScreenTime = fTime - vTimeHeader4.y;
    
    float fSignal = 0.0;
	
	if(fPhase == kPhaseBlank)
	{                       
		fSignal = 0.0;           
	}
	else  
	if(fPhase == kPhaseSilent)
	{
		fSignal = 0.0;          
	}
	else
	if(fPhase == kPhaseHeader)
	{
		float fFreq = 3500000.0 / 2168.0;
        fFreq *= 0.5;
		float fBlend = step(fract(fTime * fFreq), 0.5);
		fSignal = fBlend;           
	}
	else
	if(fPhase == kPhaseData)
	{
		float fFreq = 3500000.0 / 1710.0;
        float fWaveTime = fTime * fFreq;

        float fDataHashPos = floor(fWaveTime);
        
        // Attribute loading sounds
        float kAttributeStart = 256.0 * 192.0 / 8.0;    
        float kAttributeEnd = kAttributeStart + (32.0 * 24.0);    
        float fAddressLoaded = fLoadScreenTime * 192.0;
		if( (fAddressLoaded > kAttributeStart) && (fAddressLoaded < kAttributeEnd) )
        {
            fDataHashPos = mod(fDataHashPos, 8.0);
        }

        
        float fValue = hash(fDataHashPos);
        
        
        float fr = fract(fWaveTime);
        if(fValue > 0.5)
        {
            fr = fract(fr * 2.0);
        }
        float fBlend = step(fr, 0.5);

		fSignal = fBlend;                   
	}
	
	return fSignal;
}

// Function 206
vec4 AntiAliasPointSampleTexture_None(vec2 uv, vec2 texsize) {	
	return texture(iChannel0, (floor(uv+0.5)+0.5) / texsize, -99999.0);
}

// Function 207
vec2 sample_biquadratic_gradient_approx(sampler2D channel, vec2 res, vec2 uv) {
    vec2 q = fract(uv * res);
    vec2 cc = 0.5 / res;
    vec2 ww0 = uv - cc;
    vec2 ww1 = uv + cc;
    float nx = texture(channel, vec2(ww1.x, uv.y)).r - texture(channel, vec2(ww0.x, uv.y)).r;
    float ny = texture(channel, vec2(uv.x, ww1.y)).r - texture(channel, vec2(uv.x, ww0.y)).r;
	return vec2(nx, ny);
}

// Function 208
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

// Function 209
vec4 GetSampleColor(ivec2 currentCoord, ivec2 samplePosition, float sampleResolution)
{
    ivec2 sampleOffset = currentCoord - samplePosition;
    ivec2 sampleCoord = ivec2(floor(vec2(sampleOffset) / sampleResolution));
    vec4 sampleColor = texture(iChannel0, vec2(samplePosition + sampleCoord) / iResolution.xy);
    return sampleColor;
}

// Function 210
float sampleMusic()
{
	return 0.25 * (
		texture( iChannel0, vec2( 0.01, 0.25 ) ).x + 
		texture( iChannel0, vec2( 0.07, 0.25 ) ).x + 
		texture( iChannel0, vec2( 0.15, 0.25 ) ).x + 
		texture( iChannel0, vec2( 0.30, 0.25 ) ).x);
}

// Function 211
float SampleBicubic(vec2 p)
{
    vec2 p0 = floor(p);
    vec2 l  = p - p0;
    
    float f = 0.0;
	for (float y = 0.0; y < 2.0; ++y)
	for (float x = 0.0; x < 2.0; ++x)
    {
        vec4 n = Map2(p0 + vec2(x, y));
        
        f += dot(kern(l - vec2(x, y)), n);
    }
    
    return f;
}

// Function 212
vec2 sample_direction_uniform() {
    float theta = two_pi * sample_uniform();
	return vec2(cos(theta), sin(theta));
}

// Function 213
vec4 SampleBicubic3(sampler2D channel, vec2 uv, out vec4 d2)
{
    uv -= vec2(0.5);
    
    vec2 uvi = floor(uv);
    vec2 uvf = uv - uvi;

    ivec2 uv0 = ivec2(uvi);
    
    d2 = vec4(0.0);
    vec4 r = vec4(0.0);
    for(int j = 0; j < 2; ++j)
    for(int i = 0; i < 2; ++i)
    {
        vec4 c = texelFetch(channel, uv0 + ivec2(i, j), 0);
        
        vec2 l = uvf;
        
        if(i != 0) l.x -= 1.0;
        if(j != 0) l.y -= 1.0;
        
        mat4 mA, mB;
        kern4x4(l, /*out*/ mA, mB);
        
        r  += mA * c;
        d2 += mB * c;
    }
    
    // r  = vec4(  df/dx,   df/dy,  ddf/dxy ,         f)
    // d2 = vec4(ddf/dxx, ddf/dyy, dddf/dxxy, dddf/dxyy)
	return r;
}

// Function 214
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

// Function 215
vec4 gaussian7sample(in sampler2D iChannel, in vec2 uv, in float px) {
    return gSmp(0,3.)+gSmp(1,2.)+gSmp(2,1.)+gSmp(3,0.)+gSmp(2,1.)+gSmp(1,2.)+gSmp(0,3.);
}

// Function 216
float3 Sample_Uniform_Cone ( float lobe, out float pdf, inout float seed ) {
  float2 u = Sample_Uniform2(seed);
  float phi = TAU*u.x,
        cos_theta = 1.0 - u.y*(1.0 - cos(lobe));
  pdf = PDF_Cone(lobe);
  return To_Cartesian(cos_theta, phi);
}

// Function 217
float2 Normal_Sampler ( in sampler2D s, in float2 uv ) {
  float2 eps = float2(0.003, 0.0);
  return float2(length(texture(s, uv+eps.xy)) - length(texture(s, uv-eps.xy)),
                length(texture(s, uv+eps.yx)) - length(texture(s, uv-eps.yx)));
}

// Function 218
vec3 sampleEnvironment(vec3 normalizedDir)
{

    vec3 skyColor     = vec3(0.55, 0.45, 0.58);
    vec3 horizonColor = vec3(0.50, 0.35, 0.55);

    float envAmp = 1.0 * (1. + .1 * g_bassBeat);
    return envAmp * mix(horizonColor, skyColor, smoothstep(-.5, 1.0, normalizedDir.y)); 
}

// Function 219
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

// Function 220
vec3 multisample(vec2 pixel) {
 
    vec2 points[4];
    
    points[0] = pixel + vec2(offset_a, offset_b);
    points[1] = pixel + vec2(-offset_a, -offset_b);
    points[2] = pixel + vec2(offset_b, -offset_a);
    points[3] = pixel + vec2(-offset_b, -offset_a);
    
    vec3 color = vec3(0.0);
    
    for (int i= 0; i < 4; i++) {
        color+= calculateLighting(points[i], light0);
        color+= calculateLighting(points[i], light1);
    }
    
    return color / 4.0;
}

// Function 221
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

// Function 222
float sampleMusicA() {
	return 0.6 * (
		texture( iChannel0, vec2( 0.15, 0.25 ) ).x + 
		texture( iChannel0, vec2( 0.30, 0.25 ) ).x);
}

// Function 223
float sampleMusic()
{
	return 0.5 * (
		//texture( iChannel0, vec2( 0.01, 0.25 ) ).x + 
		//texture( iChannel0, vec2( 0.07, 0.25 ) ).x + 
		texture( iChannel0, vec2( 0.15, 0.25 ) ).x + 
		texture( iChannel0, vec2( 0.30, 0.25 ) ).x);
}

// Function 224
float getSample(float time, float tt, float FM){
    tt -= mod(tt,RES);
    float note1 = note(tt);
    float note2 = note(tt+0.5);
    if (note1 <0.0)     return 0.0;    
    float stepper = smoothstep(0.1,0.5,mod(tt,0.5));
    float note = mix(note1,note2,stepper);    
    float angle = PI2*n2f(note)*time;
    return sin(angle+FM*sin(angle*2.033));}

// Function 225
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

// Function 226
vec4 SampleMip2(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*16.,sp.z+32.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(16.,0.))*IRES),fract(sp.y));
}

// Function 227
vec3 sampleHemisphere( const vec3 n, in float Xi1, in float Xi2 ) {
    vec2 r = vec2(Xi1,Xi2)*TWO_PI;
	vec3 dr=vec3(sin(r.x)*vec2(sin(r.y),cos(r.y)),cos(r.x));
	return dot(dr,n) * dr;
}

// Function 228
vec3 sampleSun(const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, float seed) {
    wi = light.direction;
    return light.L;
}

// Function 229
Sample sample_dielectric(HitQuery hit, Ray ray) {

    vec2 wi = -ray.dir;
    vec2 normal = hit.normal;
    wi = world2local(normal, wi);
    float cos_theta_i = wi.y;
    
    
    bool entering = cos_theta_i > 0.0;
    float eta_i = entering ? 1.0: cauchy_ior(ray.wavelength);
    float eta_t = entering ? cauchy_ior(ray.wavelength): 1.0;
    float eta = eta_i / eta_t;
    
    cos_theta_i = abs(cos_theta_i);
    float fresnel = compute_fresnel(cos_theta_i, eta_i, eta_t);
    
    
    vec2 wo, pert;
    float pdf, contrib;
    if (sample_uniform() < fresnel) {
        wo = vec2(-wi.x, wi.y);
        pdf = fresnel;
        contrib = fresnel / cos_theta_i;
        pert = ( entering ? 1. : -1.) * eps * hit.normal;
    } else {
        float sin2_theta_I = max(0.0, 1.0 - cos_theta_i * cos_theta_i);
        float sin2_theta_t = eta * eta * sin2_theta_I;
        float cos_theta_t = sqrt(1.0 - sin2_theta_t);
        wo = -eta * wi + (eta * cos_theta_i - cos_theta_t) * my_faceforward(vec2(0.0,1.0), wi);
        pdf = 1.0 - fresnel;
        contrib = (1.0 - fresnel) / cos_theta_i;
        pert = ( entering ? -1. : 1.) * eps * hit.normal;
        if (false) {
            contrib *= eta * eta;
        }
    }
    
    
    wo = local2world(normal, wo);
    
    // return sampled ray 
    return Sample(Ray(hit.p + pert, wo, ray.wavelength), contrib, pdf, cos_theta_i);
}

// Function 230
vec4 audioeclipse(vec2 fragCoord, float fadein)
{
	vec2 p=(fragCoord.xy-.5*iResolution.xy)/min(iResolution.x,iResolution.y);
    vec3 c=vec3(0,0,0.1);
    for(float i=0.;i<dots; i++){
		float vol =  texture(iChannel2, vec2(i/dots, 0.0)).x;
		float b = vol * brightness;
        float x = radius*cos(iTime*3.14*float(i)/dots);
        float y = radius*sin(iTime*3.14*float(i)/dots);
        vec2 o = vec2(x,y);
		vec3 dotCol = hsv2rgb(vec3((i + iTime*10.)/dots,fadein,1.0));
		c += b/(length(p-o))*dotCol;
    } 
	float dist = distance(p , vec2(0));  
    float shape = smoothstep(0.295, 0.3, dist);
	return vec4(c,shape);
}

// Function 231
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

// Function 232
vec3 GetSampleColor(float sscoc,vec2 uv
){Ray r
 ;r.dir = vec3(0,0,1)
 ;if (fishEye
 ){vec3 crossv=cross(r.dir,vec3(uv,0))
  ;r.dir=qr(aa2q(length(uv)*FOV,normalize(crossv)),r.dir)
  ;}else r.dir = vec3(uv.xy*FOV,1.)
 ;//apply look dir
 ;r.b = objPos[oCam]
 ;r.dir = qr(objRot[oCam],r.dir)
 ;MarchPOV(r,playerTime,sscoc)
 ;return GetDiffuse(sscoc,r);}

// Function 233
float SampleDigit(const in float fDigit, const in vec2 vUV) {
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
	vec4 vRect0 = vec4(0.0);
	vec4 vRect1 = vec4(0.0);
	vec4 vRect2 = vec4(0.0);
	if(fDigit < 0.5) {
		vRect0 = vec4(x0, y0, x3, y5); 
        vRect1 = vec4(x1, y1, x2, y4);
	} else if(fDigit < 1.5) {
		vRect0 = vec4(x1, y0, x2, y5); 
        vRect1 = vec4(x0, y0, x0, y0);
	} else if(fDigit < 2.5) {
		vRect0 = vec4(x0, y0, x3, y5); 
        vRect1 = vec4(x0, y3, x2, y4); 
        vRect2 = vec4(x1, y1, x3, y2);
	} else if(fDigit < 3.5) {
		vRect0 = vec4(x0, y0, x3, y5); 
        vRect1 = vec4(x0, y3, x2, y4); 
        vRect2 = vec4(x0, y1, x2, y2);
	} else if(fDigit < 4.5) {
		vRect0 = vec4(x0, y1, x2, y5); 
        vRect1 = vec4(x1, y2, x2, y5); 
        vRect2 = vec4(x2, y0, x3, y3);
	} else if(fDigit < 5.5) {
		vRect0 = vec4(x0, y0, x3, y5); 
        vRect1 = vec4(x1, y3, x3, y4); 
        vRect2 = vec4(x0, y1, x2, y2);
	} else if(fDigit < 6.5) {
		vRect0 = vec4(x0, y0, x3, y5); 
        vRect1 = vec4(x1, y3, x3, y4); 
        vRect2 = vec4(x1, y1, x2, y2);
	} else if(fDigit < 7.5) {
		vRect0 = vec4(x0, y0, x3, y5); 
        vRect1 = vec4(x0, y0, x2, y4);
	} else if(fDigit < 8.5) {
		vRect0 = vec4(x0, y0, x3, y5); 
        vRect1 = vec4(x1, y1, x2, y2); 
        vRect2 = vec4(x1, y3, x2, y4);
	} else if(fDigit < 9.5) {
		vRect0 = vec4(x0, y0, x3, y5); 
        vRect1 = vec4(x1, y3, x2, y4); 
        vRect2 = vec4(x0, y1, x2, y2);
	} else if(fDigit < 10.5) {
		vRect0 = vec4(x1, y0, x2, y1);
	} else if(fDigit < 11.5) {
		vRect0 = vec4(x0, y2, x3, y3);
	}	
	float fResult = InRect(vUV, vRect0) + InRect(vUV, vRect1) + InRect(vUV, vRect2);
	return mod(fResult, 2.0);
}

// Function 234
vec4 AntiAliasPointSampleTexture_Linear(vec2 uv, vec2 texsize) {	
	vec2 w=fwidth(uv);
	return texture(iChannel0, (floor(uv)+0.5+clamp((fract(uv)-0.5+w)/w,0.,1.)) / texsize, -99999.0);	
}

// Function 235
vec4 SampleMip4(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*4.,sp.z+56.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(4.,0.))*IRES),fract(sp.y));
}

// Function 236
vec3 CosineWeightedSampleHemisphere ( vec3 normal, vec2 rnd )
{
   //rnd = vec2(rand(vec3(12.9898, 78.233, 151.7182), seed),rand(vec3(63.7264, 10.873, 623.6736), seed));
   float phi = acos( sqrt(1.0 - rnd.x)) ;
   float theta = 2.0 * 3.14 * rnd.y ;

   vec3 sdir = cross(normal, (abs(normal.x) < 0.5001) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0));
   vec3 tdir = cross(normal, sdir);

   return normalize(phi * cos(theta) * sdir + phi * sin(theta) * tdir + sqrt(1.0 - rnd.x) * normal);
}

// Function 237
vec3 sampleBuff(float u,float v)
{
    return sampleBuff( vec2(u,v) ); 
}

// Function 238
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

// Function 239
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

// Function 240
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

// Function 241
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

// Function 242
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

// Function 243
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

// Function 244
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

// Function 245
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

// Function 246
float backgroundTerrainSample(vec2 uv){
	float globalX1 = (uv.x + iTime / 12.0 + 100.0);
	float height1 = terrainHeight(globalX1) + 0.3;
	float globalX2 = (uv.x + iTime / 17.0 + 50.0);
	float height2 = terrainHeight(globalX2) + 0.05;
    
    float tree1 = getTree(vec2(uv.x + iTime / 12.0, height1), 59.0, 0.07, 1.8);
    float tree2 = getTree(vec2(uv.x + iTime / 17.0, height2), 23.0, 0.07, 1.2);
    
    float back0 = mix(0.5, 1.0, clamp(-tree2 + smoothstep(height2, height2 + 0.15, uv.y), 0.0, 1.0));
    float back1 = mix(0.2, back0, clamp(-tree1 + smoothstep(height2, height2 + 0.15, uv.y), 0.0, 1.0));
    
    return back1;
}

// Function 247
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

// Function 248
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

// Function 249
float sampleSpriteBody2 (vec2 uv)
{
    vec2 fracuv = fract(uv);
    int x = int(fracuv.x * 16.0);
    int y = int(fracuv.y * 20.0);
    
    // 16 idx data per row, 1 element & 4 index per 1 element...
    // => 4 element per row
    int indexperelement = 4;
    int elementperrow = 4;
    int bitsperindex = 2;
    int arrayidx = y * elementperrow + x / indexperelement;
    int idx = x % indexperelement;
    int bitoffset = (idx) * bitsperindex;
    int mask = 3 << bitoffset;
    int bits = (sprBody2[arrayidx] & mask) >> bitoffset; // test

    float value = float(bits) / 3.0;
    return (value);
}

// Function 250
float samplenaejang (vec3 uv, float time)
{
    // prevent floating point error
    // time = mod(time, 6.285);
    
    // apply wobble
    vec2 wobblyUV = vec2(uv.x, uv.y);
    wobblyUV.x += sin(uv.y * 10.0 + mod(time * 0.82, 6.28) + cos(uv.x * 2.5 + time * 0.5) * 0.15) * 0.065;
    wobblyUV.y += cos(uv.x * 14.2 + mod(time * 0.75, 6.28) + sin(uv.y * 1.5 + time * 0.6) * 0.2) * 0.065;
    
    // calculate 10PRINT
    vec2 uvMult = wobblyUV * 8.0;
   	vec2 uvChunky = floor(uvMult) / 8.0;
    vec2 uvChunkyLocal = fract(uvMult);
    float chunkFlip = sign(floor(noise(uvChunky * 10.0) + 0.5) - 0.5);
    
    vec2 gridDelta = fract(vec2(uvChunkyLocal.x * chunkFlip, uvChunkyLocal.y)) - 0.5;
    float dist1 = min(distance(vec2(0.5), gridDelta), distance(vec2(0.5), -gridDelta));
    float dist2 = abs(0.5 - dist1);
    float thiccness = 0.8 + pow(sin(time), 3.0) * 0.4;
    float shape = dist2 * thiccness;//smoothstep(0.3, 0.75, dist2 * thiccness);
    
    return clamp((1.0 - shape) - uv.z, 0.0, 1.0);
}

// Function 251
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

// Function 252
vec4 sample_neighbors(int i,int d,vec3 p, vec3 o, vec4 f)
{vec3 e=vec3(-.5,.5,.0)
;ivec2 b=ivec2(i,d)
;vec4 g=fetch_lpv(p+o)
;return 
 af(g,b,0,2,p,e.zzx-o,f,gv4[0],bc4[0])
+af(g,b,1,2,p,e.zzy-o,f,gv4[1],bc4[1])
+af(g,b,2,1,p,e.zxz-o,f,gv4[2],bc4[2])
+af(g,b,3,1,p,e.zyz-o,f,gv4[3],bc4[3])
+af(g,b,4,0,p,e.xzz-o,f,gv4[4],bc4[4])
+af(g,b,5,0,p,e.yzz-o,f,gv4[5],bc4[5])
;}

// Function 253
vec4 SampleCol( vec2 vUV )
{
    vec4 vSample = textureLod( iChannel0, vUV, 0.0 );
    
    vec3 vCol = normalize( 0.5 + 0.49 * -cos( vec3(0.1, 0.4, 0.9) * vSample.g + iTime * 2.0));
    
    float shade = vSample.r;
    vSample.rgb = vCol * shade;
    
    return vSample;
}

// Function 254
vec4 getSampleAt(vec2 uv, mat4 viewMatrix, mat4 inverseViewMatrix, int row, int col)
{
	const int gridWidth = 8;
	const int gridHeight = gridWidth;
	float subPixelWidth = 1.0 / (iResolution.y * float(gridWidth));
	float subPixelHeight = subPixelWidth;
	
	float xDisplacement = subPixelWidth * (float(col) - (float(gridWidth - 1) / 2.0));
	float yDisplacement = subPixelHeight * (float(row) - (float(gridHeight - 1) / 2.0));
				
	//This is a vector from the camera to the near plane
	vec3 cameraToNear = vec3(0, 0, (1.0 / tan(verticalFov)));
	
	//Direction of line from camera to near plane in eye coordinates, this is the "ray"
	vec3 lineDirection = vec3(uv.x + xDisplacement, uv.y + yDisplacement, 0) - cameraToNear;
	
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
		float value = 1.0 / 16.0;
		color = vec4(value, value, value, 0);	
	}
	
	return color;
}

// Function 255
vec3 samplescene (vec2 uv, float time)
{
    vec2 uvsize = (iResolution.xy / iResolution.x);
    vec2 uvsizeHalf = uvsize * 0.5;
    vec3 final = vec3(0.0);
    
    // Prepare rect properties
    vec2 rectHalfSize = vec2(0.4, 0.225);
    const float rectUVScale = 1.5;
    vec2 rectUV = (uv - (uvsizeHalf - rectHalfSize)) * rectUVScale;
    
    // Downscale the rectangle's resolution
    const float crunchfactor = 64.0;
    vec2 uvcrunchy = floor(rectUV * crunchfactor) / crunchfactor;
    vec2 uvcrunchylocal = fract(rectUV * crunchfactor);
    
    // Commodore colours
    vec3 colourBG = HEXRGB(0x887ecb);
    vec3 colourRect = HEXRGB(0x50459b);
    
    // Background C64 loading screen-like raster bars
    float rasterScale = 15.0;
    float rasterOff = time * 0.5;
    float rasterMix = floor(fract((uv.y + rasterOff) * rasterScale + (uv.x * sin(time * 3.0)) * 0.5) + 0.5);
    const vec3 colours[3] = vec3[3](HEXRGB(0x6abfc6), HEXRGB(0xa1683c), HEXRGB(0x9ae29b));
    
    colourBG = mix(colours[int(time) % 3], HEXRGB(0xadadad), rasterMix);
    
    // Foreground : 10PRINT
    const float uvdownscaleFactor = 64.0;
    vec2 uvdownscale = (rectUV * uvdownscaleFactor + 0.5);
    vec2 uvdownscaleLocal = fract(uvdownscale);
    uvdownscale = floor(uvdownscale) / uvdownscaleFactor;
    
    vec3 rectBG = samplerectBG(uv, uvdownscale, time);
    float rectBGLuma = clamp(dot(rectBG, rectBG), 0.0, 1.0);
    
    // apply LED light effect to foreground's 10PRINT BG(??)
    float ledDiscRadius = 0.25 * rectBGLuma + 0.20;
    const float ledDiscRadiusSmooth = 0.1;
    float ledDiscDelta = distance(vec2(0.5), uvdownscaleLocal);
    float ledDiscMix = smoothstep(ledDiscRadius + ledDiscRadiusSmooth, ledDiscRadius, ledDiscDelta);
    colourRect = mix(rectBG * 0.5, rectBG, ledDiscMix);
    colourRect = clamp(colourRect + pow(1.0 - ledDiscDelta, 2.0) * 0.2, 0.0, 1.0);

    // Foreground : Sprites
    vec2 sprUV;
    float sprAnimTime = time * 2.0;
    float sprRot = sin(sprAnimTime);
    float sprScale = 8.0;
    vec2 sprOff = vec2(sin(time * 0.5 + cos(time * 0.1) * 0.01) * 0.05, cos(time * 0.5) * 0.025 + sin(time * 0.1) * 0.01);
    
    // body
    float rot = radians(pow(sprRot, 4.0) * 12.0 * 0.1);
    sprUV = (vec2(uv.x, uv.y) - uvsizeHalf + sprOff) * sprScale;
    //sprUV.y -= 0.75;
    sprUV *= mat2(cos(rot), -sin(rot), sin(rot), cos(rot));
    sprUV += 0.5;
    sprUV.y *= -1.0;
    sprUV.y += -0.6;
    colourRect = mixSpriteBody2(colourRect, sprUV);
    
    // body
    rot = radians(pow(sprRot, 3.0) * 12.0 * 0.3);
    sprUV = (vec2(uv.x, uv.y) - uvsizeHalf + sprOff) * sprScale;
    //sprUV.y -= 0.75;
    sprUV *= mat2(cos(rot), -sin(rot), sin(rot), cos(rot));
    sprUV += 0.5;
    sprUV.y *= -1.0;
    sprUV.y += -0.65 + sin(sprAnimTime * 2.0) * 0.05;
    colourRect = mixSpriteBody1(colourRect, sprUV);
    
    // head
    rot = radians(sprRot * 12.0 * -0.5);
    sprUV = (vec2(uv.x, uv.y) - uvsizeHalf + sprOff) * sprScale;
    sprUV *= mat2(cos(rot), -sin(rot), sin(rot), cos(rot));
    sprUV += 0.5;
    sprUV.y *= -1.0;
    sprUV.y += sin(sprAnimTime * 2.0) * 0.1;
    colourRect = mixSpriteHead(colourRect, sprUV);
    
    // debug light
    //float lightCircleMix = smoothstep(0.01, -0.01, length(lightDelta.xy) - 0.01);
    //colourRect = mix(colourRect, vec3(0.0, 1.0, 1.0), lightCircleMix);
    
    // Draw commodore 64-esque screen
    // shadow
    vec2 centerDelta = uvsizeHalf - uv + vec2(0.025, -0.025);
    float rectMinDelta = max(abs(centerDelta.x) - rectHalfSize.x, abs(centerDelta.y) - rectHalfSize.y);
    float rectfactor = 1.0 - ceil(max(rectMinDelta, 0.0));
    vec3 rect = mix(colourBG, colourBG * vec3(0.5), rectfactor);
    
    // screen
    centerDelta = uvsizeHalf - uv;
    rectMinDelta = max(abs(centerDelta.x) - rectHalfSize.x, abs(centerDelta.y) - rectHalfSize.y);
    rectfactor = 1.0 - ceil(max(rectMinDelta, 0.0));
    rect = mix(rect, colourRect, rectfactor);
    
	return rect;
}

// Function 256
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

// Function 257
float sampleFreq(float freq) {return texture(iChannel0, vec2(freq, 0.25)).x;}

// Function 258
vec2 foregroundTerrainSample(vec2 uv){
	float globalX = (uv.x + iTime / 2.0 + 23.0);
	float height = terrainHeight(globalX) - 0.5;
    
	return vec2(0.5, 0.3 - smoothstep(height, height + 1.0, uv.y) * 0.3);
}

// Function 259
vec4 SampleMip1(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*32.,sp.z);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(32.,0.))*IRES),fract(sp.y));
}

// Function 260
float Sample( vec2 pos )
{
	return EvalTestSignal( pos );
	//return texture( iChannel0, pos*0.5 + 0.5 ).x;
}

// Function 261
vec3 sampleLight( const in vec3 ro ) {
    lowp vec3 n = randomSphereDirection() * lightSphere.w;
    return lightSphere.xyz + n;
}

// Function 262
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

// Function 263
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

// Function 264
vec4 sample2D(sampler2D sampler,vec2 resolution, vec2 uv)
{
    return texture(sampler, uv / resolution);
}

// Function 265
vec4 sample_photons(sampler2D tex, vec2 coords){
	/* Samples the photons as they arrive at the enterance
	to the lens. Not quite true because we don't use a 
	real lens model, but close enough.... */
	vec2 texture_coords = uncenter_coords(coords);
	return texture(tex, texture_coords, 1.0);
}

// Function 266
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

// Function 267
vec4 AntiAliasPointSampleTexture_Smoothstep(vec2 uv, vec2 texsize) {	
	vec2 w=fwidth(uv);
	return texture(iChannel0, (floor(uv)+0.5+smoothstep(0.5-w,0.5+w,fract(uv))) / texsize, -99999.0);	
}

// Function 268
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

// Function 269
vec3 chromaticSample(sampler2D s, vec2 position, vec2 warp) {
    return vec3(texture(s, position + 0.8 * warp).r, texture(s, position + warp).g, texture(s, position + 1.2 * warp).b);
}

// Function 270
vec3 cosineSampleHemisphere(const in vec2 u) {
    vec2 d = concentricSampleDisk(u);
    float z = sqrt(max(EPSILON, 1. - d.x * d.x - d.y * d.y));
    return vec3(d.x, d.y, z);
}

// Function 271
vec3 samplef2(vec2 position) {
	float d = sample_biquadratic(iChannel0, iChannelResolution[0].xy, position).r;
    vec2 n = sample_biquadratic_gradient_approx(iChannel0, iChannelResolution[0].xy, position);
    return vec3(n, d);
}

// Function 272
bool sampleDist(vec2 uv)
{
    float radius = 0.1;
    return sqrt(dot(uv, uv)) < radius;
}

// Function 273
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

// Function 274
float sample_uniform(void) {
    return float(rand())/32767.0;
}

// Function 275
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

// Function 276
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

// Function 277
vec3 SampleLPV(vec3 P, vec3 N) {
    vec3 f=fract(P-vec3(0.5,0.5,0.5));
    vec2 pvt1=PToUV(P+vec3(-0.5,-0.5,-0.5));
    vec2 pvt2=PToUV(P+vec3(-0.5,-0.5,0.5));
    vec3 vDS=N.xyz*N.xyz;
    vec3 xC,yC,zC,CC;
    xC=MIX(f,pvt1,pvt2,CC); xC=((N.x>0.)?xC:CC);
    yC=MIX(f,pvt1+vec2(0.,64.),pvt2+vec2(0.,64.),CC); yC=((N.y>0.)?yC:CC);
    zC=MIX(f,pvt1+vec2(0.,128.),pvt2+vec2(0.,128.),CC); zC=((N.z>0.)?zC:CC);
    return xC*vDS.x+yC*vDS.y+zC*vDS.z;
}

// Function 278
vec4 SampleMip5(vec3 sp) {
    sp.y=sp.y-0.5;
    vec2 cuv1=vec2(sp.x+floor(sp.y)*4.,sp.z+60.);
    return mix(texture(iChannel2,cuv1*IRES),
               texture(iChannel2,(cuv1+vec2(2.,0.))*IRES),fract(sp.y));
}

// Function 279
float sampleVolume(in vec3 rp)
{
    float t = map(rp);
    t = -smoothstep(0., -THICKNESS*.5, t);
    float d = noise(SAMPLE(rp)*22.)*.8;
    d += noise(SAMPLE(rp)*70.)*.4;
    d += noise(SAMPLE(rp)*100.)*.2;
    d += noise(SAMPLE(rp)*350.)*.45*d;
    float density = clamp(-t, 0.0, 1.0)*d;
    return clamp((density-0.4)/0.8, 0.0, 1.0);
}

// Function 280
vec4 SampleFlowingNormal( const vec2 vUV, const vec2 vFlowRate, const float fFoam, const float time, out float fOutFoamTex )
{
    float fMag = 2.5 / (1.0 + dot( vFlowRate, vFlowRate ) * 5.0);
    float t0 = fract( time );
    float t1 = fract( time + 0.5 );
    
    float i0 = floor( time );
    float i1 = floor( time + 0.5 );
    
    float o0 = t0 - 0.5;
    float o1 = t1 - 0.5;
    
    vec2 vUV0 = vUV + Hash2(i0);
    vec2 vUV1 = vUV + Hash2(i1);
    
    vec4 sample0 = SampleWaterNormal( vUV0, vFlowRate * o0, fMag, fFoam );
    vec4 sample1 = SampleWaterNormal( vUV1, vFlowRate * o1, fMag, fFoam );

    float weight = abs( t0 - 0.5 ) * 2.0;
    //weight = smoothstep( 0.0, 1.0, weight );

    float foam0 = SampleWaterFoam( vUV0, vFlowRate * o0 * 0.25, fFoam );
    float foam1 = SampleWaterFoam( vUV1, vFlowRate * o1 * 0.25, fFoam );
    
    vec4 result=  mix( sample0, sample1, weight );
    result.xyz = normalize(result.xyz);

    fOutFoamTex = mix( foam0, foam1, weight );

    return result;
}

// Function 281
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

