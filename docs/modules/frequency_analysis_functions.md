# frequency_analysis_functions

**Category:** audio
**Type:** extracted

## Dependencies
texture_sampling, normal_mapping, lighting, raymarching

## Tags
lighting, texturing, color, audio

## Code
```glsl
// Reusable Frequency Analysis Audio Functions
// Automatically extracted from audio visualization-related shaders

// Function 1
vec3 spectrum_to_rgb(in float w){
    float a = 0.;
    float wl = w;

    return lambdatoXYZ(w)*xyz-vec3(0,0,a);
}

// Function 2
void zxspectrum_colors( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 pv = floor(fragCoord.xy/LOWREZ);
    vec2 sv = floor(iResolution.xy/LOWREZ);
                
    vec4 cs=smap(texture(iChannel0,pv/sv).rgb);
    
    if( mod(pv.x+pv.y,2.0)==1.0)
    {
		fragColor = vec4(fmap(vec4(floor(cs.rgb+vec3(0.5+(DITHER*0.3))),cs.a)),1.0);
    }
    else
    {  
		fragColor = vec4(fmap(vec4(floor(cs.rgb+vec3(0.5-(DITHER*0.3))),cs.a)),1.0);
    }

}

// Function 3
float spectrum(float freq, int t, int level)
{
    return spectrum(freq, t, level, iChannel3);
}

// Function 4
float fft(float f,float r){
    return fft(f,r,0.0);
}

// Function 5
vec3 Spectrum(float x) {
    // https://www.shadertoy.com/view/wlSBzD
	float r, g, b;
    
    r = x<.16 ? S(0., .16, x)*.169 :
    	x<.22 ? S(.22, .16, x)*.134+.035 :
    	x<.41 ? S(.22, .41, x)*.098+.035 :
    	x<.64 ? S(.41,.64,x)*.851+.133 :
    			S(1., .64, x)*.984;
    
    g = x<.05 ? 0. :
    	x<.15 ? S(.05, .15, x)*.047 :
    	x<.45 ? S(.15, .45, x)*.882+.047 :
    	x<.70 ? S(.70, .45, x)*.796+.133 :
    			S(1.0, .70, x)*.133;
    
    b = x<.18 ? S(0.0, .18, x)*.5 :
    	x<.22 ? S(.22, .18, x)*.1+.4 :
    	x<.35 ? S(.22, .35, x)*.059+.4 :
    	x<.54 ? S(.54, .35, x)*.334+.125 :
    	x<.60 ? S(.54, .60, x)*.169+.125 :
    	x<.69 ? S(.69, .60, x)*.243+.051 :
    	x<.72 ? S(.69, .72, x)*.043+.051 :
    	x<.89 ? S(.89, .72, x)*.094 : 0.;
    
    return vec3(r,g,b);
}

// Function 6
float getfrequency(float x) {
	return texture(iChannel0, vec2(floor(x * FREQ_RANGE + 1.0) / FREQ_RANGE, 0.0)).x + 0.06;
}

// Function 7
vec2 spectrum( vec2 f, int iters)
{
    
    vec2 c = ((iMouse.xy-R.xy/2.)/100.);
    if(iMouse.x==0.){
        c = (.7+.3*sin(iTime/32.))*vec2(sin(iTime/20.),cos(iTime/20.));
    }
    vec2 spec = vec2(1,0);
    vec2 d = vec2(1,0);
    for(int i = 0; i < iters; i++){
        spec *= (
               cos(dot(f,d))
            )*2.;
        d=mat2(d,-d.y,d.x)*c;
    }
    return spec;
}

// Function 8
vec2 DO_FFT(float u_x,float u_y,inout vec4 c)
{
    //current index
    u_x = floor(u_x);u_y = floor(u_y) - 1.;
    if(u_x >= R_512){return vec2(0.);}
    //store for next shader to visual
    Store_T_F(u_x,u_y,c);
    //FFT.x is real value,y is imagin value
    //Bit Reverse Index Value
    float index = GetBufferValue(u_x,0.).t;
    //get domain value form Audio buffer texture and sort it
    if(u_y == -1.){
    	c.st = vec2(GetTimeDomain(index),0.);
    }
    //Do Buttfly
    //0,1,2,3,4,5,6,7,8 is FFT Step Buttfly
    if(u_y >= 0. && u_y <= 8.)
    {
        //calculate
        float mystep = u_y;
        bool Is_E_data = mod(floor(u_x/exp2(mystep)),2.) == 0.;
        //è®¡ç®—æ—‹è½¬å› å­
        float kn = mod(u_x,exp2(mystep));
        //Cä¸ºeçš„æŒ‡æ•°
        float C = -(_2PI/exp2(mystep+1.))*kn;
        vec2 W = vec2(cos(C),sin(C));
        
        float burb = exp2(mystep);
        
        vec2 buffer = vec2(0.);
        if(Is_E_data){
			//tR[i] = FFT[i].Re+FFT[i+(1<<step)].Re*W_Re-FFT[i+(1<<step)].Im*W_Im;
            //tI[i] = FFT[i].Im+FFT[i+(1<<step)].Re*W_Im+FFT[i+(1<<step)].Im*W_Re;
            //Get Next pixel buffer
 			buffer.x = GetFFT(u_x,u_y).x + GetFFT(u_x+burb,u_y).x*W.x - GetFFT(u_x+burb,u_y).y*W.y;
 			buffer.y = GetFFT(u_x,u_y).y + GetFFT(u_x+burb,u_y).x*W.y + GetFFT(u_x+burb,u_y).y*W.x;
            c.xy = buffer;
        }
        else{
            //tR[i] = FFT[i-(1<<step)].Re-(FFT[i].Re*W_Re-FFT[i].Im*W_Im);
            //tI[i] = FFT[i-(1<<step)].Im-(FFT[i].Re*W_Im+FFT[i].Im*W_Re);
            //Get The Next Pixel Buffer 
            buffer.x = GetFFT(u_x-burb,u_y).x-(GetFFT(u_x,u_y).x*W.x - GetFFT(u_x,u_y).y*W.y);
            buffer.y = GetFFT(u_x-burb,u_y).y-(GetFFT(u_x,u_y).x*W.y + GetFFT(u_x,u_y).y*W.x);
			c.xy = buffer;
        }
        
    }
	return GetFFT(u_x,9.).xy;
}

// Function 9
vec3 hsv2rgb_spectrum(float h, float s, float v) {
	return v* mix(vec3(1),clamp(1.- abs(1.- mod(3.* h+ vec3(1,0,2), 3.)),0.,1.),s);
}

// Function 10
vec4 GetFFT(float x,float y){
	vec4 FFTBuffer = texture(SelfBuffer,vec2(x+0.5,y+0.5)/SelfResolution);
    //++i;
    return FFTBuffer;
}

// Function 11
float fft(vec2 uv)
{
    return cos( uv.y * PI * (2.0*uv.x + 1.0) / 16.0 );
}

// Function 12
void calcFFT(inout vec4 fragColor, vec2 fragCoord)
{
	float x = floor(fragCoord.x);
	float y = floor(fragCoord.y);

	vec2 res = vec2(0.0);

	if (x >= SignalLength) {
		return;
	}

	if (y == ROW_SORTED_SIGNAL) {
        /* Get domain value form Audio buffer texture and sort it */
		
		float index = getBitReverseIndex(x); 
		res.x = getSignal(index) * window(index / SignalLength);

	} else if (y >= ROW_BUTTERFLY_MIN && y <= ROW_BUTTERFLY_MAX) { 
        /* Do Butterfly. */
		
		float st = y - 1.0; // step. (0,1,2,3,4,5,6,7,8)
		res = calcStep(x, st);

	} else if (y == ROW_CPU_SPECTRUM) { 
        /* store for next shader for visualization */
		
		// Store Frequency domain value form cpu
		res.x = texture(SignalChannel, vec2((x + 0.5) / SignalLength, 0.25)).r;

	} else if (y == ROW_FFT) { 
        /* Store calculated power spectrum in logarithmic scale */
        		
        vec2 fftCurr = calcStep(x, LAST_STEP);
        fftCurr /= SignalLength; // normalize
		
        float oldf = read(x, ROW_FFT).r;
        float freq = calcLogPower(fftCurr, oldf);
		
		res = vec2(freq);
	}

	fragColor = vec4(res, 0.0, 0.0);
}

// Function 13
float fftR(float f){
    float sum = 0.0;
    float val = 0.0;
    float coeff = 0.0;
    float k = 0.0;
    for( int i = 0; i < fftSamplesR ; i++ ){
        k = float(i)/float(fftSamplesR-1)-0.5;
        coeff = exp(-k*k/(fftSmooth*fftSmooth)*2.0);
		val += texture(sound, vec2( remapFreq(f + k * fftRadiusR)*fftWidth, 0.0) ).r * coeff;
        sum += coeff;
    }
    return remapIntensity(f,val/sum);
}

// Function 14
float spectrum(float domain, int t, int level)
{
    float sixty_fourth = 1./32.;
    vec2 uv = vec2(float(t)*3.*sixty_fourth + sixty_fourth, domain);
    uv = upper_right(uv); level++;
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(iChannel3, uv).x;
}

// Function 15
float spectrum2D(vec2 uv, float thickness, int level)
{
    float val = spectrum(uv.x, 0, level);
    return (abs(uv.y - val) < thickness/2.) ? (1.-abs(uv.y - val)*2./thickness) : 0.;
}

// Function 16
void CPU_FFT_Visual(vec2 u,inout vec4 c){
	//u.x -= 100.;
    u.x = floor(u.x/iResolution.y*360.)-60.;
    u.y = u.y/iResolution.y*350.-50.;
    if(u.x-0.5 < 256. && u.x>=0. && u.y>0.){
        float enegy = texture(iChannel0,vec2(u.x,10.5)/iChannelResolution[0].xy).y*50.;
        if(u.y < enegy){
            c.b = 1.;
        }
    }
}

// Function 17
vec3 hsv2rgb_fl_spectrum(float hsvhue,float hsvsat,float hsvval) {
	int hsv2hue = scale8(uint8_t(hsvhue),191);
	return hsv2rgb_raw_C(float(hsv2hue)/ 255.,hsvsat,hsvval);
}

// Function 18
float convert_wave_length_to_black_body_spectrum(float wave_length_nm, float temperature_in_kelvin)
{
    float wave_length_in_meters = wave_length_nm * 1e-9;
    float c1                    = 2.0 * pi * plancks_constant * speed_of_light * speed_of_light;
    float c2                    = plancks_constant * speed_of_light / boltzmanns_constant;
    float m                     = c1 / pow(wave_length_in_meters, 5.0) * 1.0 / (exp(c2 / (wave_length_in_meters * temperature_in_kelvin)) - 1.0);
    return m;
}

// Function 19
vec2 GetFFT(int x,int y){
	return texelFetch(DynamicBuffer,ivec2(x,y),0).st;
}

// Function 20
float fft(float t, float resolution) {
    return mix(
        texture(iChannel0, vec2(floor(t * resolution) / resolution, .25)).x,
        texture(iChannel0, vec2(floor(t * resolution + 1.) / resolution, .25)).x,
        fract(t * resolution));
}

// Function 21
vec3 spectrum_to_rgb(in float w){
    float wl = w;

    return lambdatoXYZ(w)*xyz;
}

// Function 22
float getfrequency_blend(float x) {
    return mix(getfrequency(x), getfrequency_smooth(x), 0.5);
}

// Function 23
float ReferenceSpectrum(float x, float D)
{
    return x*tanh(x*D);
}

// Function 24
float FFTBand_amplitude(FFTBand band)
{
    return length(band.f);
}

// Function 25
float spectrum(float domain, int t, int level)
{
    float sixty_fourth = 1./32.;
    vec2 uv = vec2(float(t)*2./sixty_fourth + sixty_fourth, domain);
    uv = upper_right(uv); level++;
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(iChannel3, uv).x;
}

// Function 26
vec2 GetSpectrumUV( vec2 vUV, int index )
{
    vec2 vSpectrumUV = vUV;
    float fMinY = fSpMinY - float(index) * fGraphSpacing;
    float fMaxY = fSpMaxY - float(index) * fGraphSpacing;
    
    vSpectrumUV.x = (vSpectrumUV.x - fSpMinX) / (fSpMaxX - fSpMinX);
    vSpectrumUV.y = (vSpectrumUV.y - fMinY) / (fMaxY - fMinY);
    
    return vSpectrumUV;
}

// Function 27
float fft(float f,float r){
    float sum = 0.0;
    float val = 0.0;
    float coeff = 0.0;
    
    float k = 0.0;
    
     // loop sampling
    for( int i = 0; i < fftSamples ; i++ ){
        k = float(i)/float(fftSamples-1)-0.5;
        // decreasing factor, more important around 0
        coeff = exp(-k*k/(fftSmooth*fftSmooth)*2.0);
        //coeff = 1.0;
        
		val += texture(inputSound, vec2( remapFreq(f + k * r)*fftWidth, 0.0) ).r * coeff;
        
        // simulation for test
        //float freq = ( remapFreq(f + k * r)*fftWidth - 0.5 ) / 0.008;//(iMouse.x/iResolution.x);
		//val += exp(  - freq*freq/2.0 ) * coeff;
        
        sum += coeff;
    }
    
    return remapIntensity(f,val/sum);
    
}

// Function 28
float smoothFrequency(float x, int smoothness)
{
    float f = 0.0;
    int accumulated = 0;
    for(float i = 0.0; i <= float(smoothness) / FREQ; i += 1.0 / FREQ)
    {
        f += frequency(x + i);
        ++accumulated;
    }
    return f / float(accumulated);
}

// Function 29
vec3 spectrum(float n) {
    return pal( n, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.33,0.67) );
}

// Function 30
float FFTBand_angle(FFTBand band)
{
    return degrees(atan(band.f.y, band.f.x));
}

// Function 31
void DrawSpectrum( inout DrawContext drawContext, vec3 vColBG )
{   
    vec2 vSpectrumUV = GetSpectrumUV( drawContext.vUV, 0 );
    
    float fGap = 0.01;

    float fSpread0 = 0.01;
    float x0 = vPrismPoint.x - 0.1 - fSpread0;
    float x1 = vPrismPoint.x - 0.1 + fSpread0;
    
    vec2 v0 = vec2(x0, ProjectPlane( x0, vPrismN1, fPrismD1 ) );
    vec2 v1 = vec2(x1, ProjectPlane( x1, vPrismN1, fPrismD1 ) );
    vec2 v2 = vec2(fSpMaxX, fSpMinY - fGap);
    vec2 v3 = vec2(fSpMinX, fSpMinY - fGap);
    
    vec2 vSpreadUV = invBilinear( drawContext.vUV, v0, v1, v2, v3 );    
    bool inSpreadLight = InUnitSquare( vSpreadUV );

    float fSpread1 = 0.005;
    float x4 = vPrismPoint.x - 0.09 - fSpread1;
    float x5 = vPrismPoint.x - 0.09 + fSpread1;
    
    vec2 v4 = vec2(x4, ProjectPlane( x4, vPrismN2, fPrismD2 ) );
    vec2 v5 = vec2(x5, ProjectPlane( x5, vPrismN2, fPrismD2 ) );

    vec2 vSpreadUV_B = invBilinear( drawContext.vUV, v0, v1, v5, v4 );    
    bool inSpreadLightB = InUnitSquare( vSpreadUV_B );
    

    
    if ( !inSpreadLight )
    {
        if ( inSpreadLightB )
        {
            inSpreadLight = true;
            vSpreadUV = vSpreadUV_B;
            vSpreadUV.y = 0.0;
        }
    }
    
#if !SHOW_SPREAD
    inSpreadLight = false;
#endif    
    
    // Hack convergence color
    vSpreadUV.y = vSpreadUV.y * 0.96 + 0.04;
    
    vec2 vBeamA = (v4 + v5) * 0.5;
    vec2 vBeamB = vec2(0.66 + fGlobalXOffset,0);
    
    float fBeamDist = LineInfo( drawContext.vUV, vBeamA, vBeamB );
    float fBeam = clamp( abs(fBeamDist) * 200.0, 0.0, 1.0 );
    fBeam = sqrt( 1.0 - fBeam * fBeam);
    fBeam += GetGlare( abs( fBeamDist ) ) * 0.2;
    
    float fGlareDist = length( drawContext.vUV - vBeamA );
    float fBeamGlare = GetGlare( fGlareDist );

    
#if !SHOW_BEAM    
    fBeam = 0.0;
    fBeamGlare = 0.0;
#endif    

    bool inSpectrum = InUnitSquare( vSpectrumUV );    

#if SEPARATE_SPECTRUM    
	inSpectrum = inSpectrum || InUnitSquare( GetSpectrumUV( drawContext.vUV, 1 ) ) || InUnitSquare( GetSpectrumUV( drawContext.vUV, 2 ) );
#endif
    
#if !SHOW_SPECTRUM
    inSpectrum = false;
#endif
    
    float fSpreadLightW0 = mix(standardObserver1931_w_min - 20.0, standardObserver1931_w_max + 20.0, vSpreadUV.x);
    float fSpectrumW0 = mix(standardObserver1931_w_min - 20.0, standardObserver1931_w_max + 20.0, vSpectrumUV.x);
    
    
    vec3 vLightColor = vec3(0);
    
    vec3 vTotXYZ = vec3(0);
    for( float w = standardObserver1931_w_min; w < NO_UNROLLF(standardObserver1931_w_max); w += 5.0 )
    {
        vec3 vCurrXYZ = WavelengthToXYZLinear( w );

        float fPower = GetSPD( w );
        
        if ( inSpreadLight )
        {
            float fWeight = UnitGaussian( w, fSpreadLightW0, 0.2 * vSpreadUV.y);
        	vTotXYZ += vCurrXYZ * fWeight * fPower * 0.01;
        }

        float t = (w - standardObserver1931_w_min) / (standardObserver1931_w_max - standardObserver1931_w_min);
        
#if SHOW_SPREAD        
        {
            vec2 vSpPos = vec2( mix( fSpMinX, fSpMaxX, t), fSpMinY - fGap);
            
            vec2 vOffset = vSpPos - drawContext.vUV;
            float d = length( vOffset );
            if ( vOffset.y > 0.0 && d < 0.5 )
            {
	        	vTotXYZ += vCurrXYZ * GetSpectrumGlare( d ) * fPower;
            }
        }
        
        {
            vec2 vPrismPos = mix( v0, v1, t );
            
            vec2 vOffset = vPrismPos - drawContext.vUV;
            float d = length( vOffset );
            if ( d < 0.5 )
            {
	        	vTotXYZ += vCurrXYZ * GetPrismGlare( d ) * fPower;
            }
        }
#endif        
        
        vLightColor += vCurrXYZ * fPower;
    }
    
    vTotXYZ += vLightColor * (fBeam + fBeamGlare) * 0.03;

#if DRAW_PRISM        
    float fPrismShade = PrismShade( drawContext.vUV );    
    vTotXYZ += vLightColor * fPrismShade * 0.1 * vec3( 0.8, 0.9, 1 );
    vTotXYZ += fPrismShade * .3 * vec3( 0.8, 0.9, 1 );
#endif    
    
    if ( inSpectrum )
    {
        vTotXYZ += WavelengthToXYZLinear(fSpectrumW0) * 0.3;
    }
    
    /*if (  drawContext.vUV.y > fSpMinY - fGap )
    {
    	vTotXYZ += 0.5;
    }*/
    
    mat3 cat = GetChromaticAdaptionTransform( mCAT_Bradford, XYZ_D65, XYZ_E );           
	vTotXYZ = vTotXYZ * cat;
        
    vec3 vColor = XYZtosRGB( vTotXYZ );    
    vColor = max( vColor, vec3(0) );
    
    vColor += vColBG;

#if SHOW_LUMINOSITY_BAR    
    vec2 vLuminosityUV = vSpectrumUV;
    vLuminosityUV.y += 1.5;
    if ( InUnitSquare( vLuminosityUV ) )
    {
        float l = WavelengthToLuminosityLinear( fSpectrumW0 ) ;
        vColor += vec3(l);
    }
#endif    
    
    vColor = 1.0 - exp2( vColor * -2.0 ); // Tonemap
    
    vColor = pow( vColor, vec3(1.0 / 2.2) );
        
    drawContext.vResult = vColor;
}

// Function 32
float fft(float p) {
    return texture(iChannel0, vec2(p, 0.25)).x;
}

// Function 33
vec3 demo_spectrum(vec2 uv, int nslits, float spacing) {
  vec4 rnd = paramdither(gfc, 28431);
  vec3 color = vec3(0.);
  for (int nw=370; nw<780; nw+=20) { 
    float lambda = (float(nw)+.5*20.*rnd.x)*1e-9;
    vec2 v = diffraction_pattern(uv, nslits, spacing, lambda);
    vec3 wcol = wavelength_to_srgbl(lambda*1e9);
    color+=wcol*dot(v,v);
  }
  return color*.4;
}

// Function 34
float getfrequency(float x) {
	return texture(iChannel0, vec2(floor(x * FREQ_RANGE + 1.0) / FREQ_RANGE, 0.25)).x + 0.06;
}

// Function 35
float fft(float x) {
    // convert x from uv space [0..1] to fft index [0..511]
    float fft_x = logyscale(x, 0.0, 1.0, 2.0, AUDIO_IN_SIZE-1.0);
    
    // sample before closest previous sample
    float fft_x0 = floor(fft_x) - 1.0;
    float fft_y0 = texelFetch(AUDIO_IN, ivec2(fft_x0, 0), 0).x;    
    
    // closest previous sample
    float fft_x1 = floor(fft_x); 
    float fft_y1 = texelFetch(AUDIO_IN, ivec2(fft_x1, 0), 0).x;

    // closest next sample
    float fft_x2 = ceil(fft_x); 
    float fft_y2 = texelFetch(AUDIO_IN, ivec2(fft_x2, 0), 0).x;
    
    // sample after closest next sample
    float fft_x3 = ceil(fft_x) + 1.0; 
    float fft_y3 = texelFetch(AUDIO_IN, ivec2(fft_x3, 0), 0).x;
    
    vec4 fft_xs = vec4(fft_x0, fft_x1, fft_x2, fft_x3);
    vec4 fft_ys = vec4(fft_y0, fft_y1, fft_y2, fft_y3);

    #if INTERP == 2
    // cubic interpolation (smooth corners)
    float fft_y;
    if (x > 0.5)
        fft_y = linscale(fft_x, fft_x1, fft_x2, fft_y1, fft_y2);
    else
		fft_y = cubic_interp(fft_x, fft_xs, fft_ys);
    
    #elif INTERP == 1
    // linear interpolation (sharp corners)
    float fft_y = linscale(fft_x, fft_x1, fft_x2, fft_y1, fft_y2);
    
    #elif INTERP == 0
    // nearest neighbor interpolation
    float fft_y = fft_y1;
    #endif
    
    return pow(fft_y, FFT_POW);
}

// Function 36
float getfrequency_smooth(float x) {
	float index = floor(x * FREQ_RANGE) / FREQ_RANGE;
    float next = floor(x * FREQ_RANGE + 1.0) / FREQ_RANGE;
	return mix(getfrequency(index), getfrequency(next), smoothstep(0.0, 1.0, fract(x * FREQ_RANGE)));
}

// Function 37
vec3 spectrum(float x) {
    x = x * 2.1 - 0.555;
    vec4 v = vec4(clamp(x, -.6, 0.6), clamp(x, 0.05, 1.05), clamp(x, 0.65, 1.55), clamp(x, 1.16, 1.55));
    v += vec4(0.0, -0.55, -1.1, -1.35);
    v *= vec4(0.8, 1.0, 1.1, 2.5);
    v = (cos(v * v * pi * 4.) * 0.5 + 0.5);
    v.r += v.a * 0.5;
    return v.rgb;
}

// Function 38
float fft(sampler2D channel) {
    const int fftSamples = 1000;
    const float sampleOffset =  (1.0 / float(fftSamples)) / 2.0;
    float totalWeight = 0.0;
    float totalFFT = 0.0;
    
    for(int sampleIndex = 0; sampleIndex < fftSamples; sampleIndex++) {
        // weight: (1/10)^(x-1)
        float samplePointer = sampleOffset + float(sampleIndex) / float(fftSamples);
        float weight = pow(1.0 - samplePointer, 10.0);
        totalWeight += weight;
        totalFFT += texture(channel, vec2(samplePointer, 0.25)).x * weight;
    }
    
	float fft = totalFFT / totalWeight;
    fft = pow(fft, 1.5);
	//float fft = texture(iChannel1, vec2(uv.x, 0.25)).x;
    
    return fft;
}

// Function 39
float fftG(float f){
    float sum = 0.0;
    float val = 0.0;
    float coeff = 0.0;
    float k = 0.0;
    for( int i = 0; i < fftSamplesG ; i++ ){
        k = float(i)/float(fftSamplesG-1)-0.5;
        coeff = exp(-k*k/(fftSmooth*fftSmooth)*2.0);
		val += texture(sound, vec2( remapFreq(f + k * fftRadiusG)*fftWidth, 0.0) ).r * coeff;
        sum += coeff;
    }
    return remapIntensity(f,val/sum)*fftGBGain;
}

// Function 40
vec3 SpectrumPoly(float x) {
    // https://www.shadertoy.com/view/wlSBzD
    return (vec3( 1.220023e0,-1.933277e0, 1.623776e0)
          +(vec3(-2.965000e1, 6.806567e1,-3.606269e1)
          +(vec3( 5.451365e2,-7.921759e2, 6.966892e2)
          +(vec3(-4.121053e3, 4.432167e3,-4.463157e3)
          +(vec3( 1.501655e4,-1.264621e4, 1.375260e4)
          +(vec3(-2.904744e4, 1.969591e4,-2.330431e4)
          +(vec3( 3.068214e4,-1.698411e4, 2.229810e4)
          +(vec3(-1.675434e4, 7.594470e3,-1.131826e4)
          + vec3( 3.707437e3,-1.366175e3, 2.372779e3)
            *x)*x)*x)*x)*x)*x)*x)*x)*x;
}

// Function 41
float FrequencyToTexture(float Frequency){
    return Frequency/440.*ATone;
}

// Function 42
float spectrum2D(vec2 uv, float thickness, int level)
{
    return spectrum2D(uv, thickness, level, iChannel3);
}

// Function 43
float frequency(float x)
{
    return texture(iChannel0, vec2(x, 0)).r;
}

// Function 44
vec2 GPUFFT(ivec2 U,inout vec2 B){
    --U.y;
    int index = BitReverse(U.x,9);
    if(U.y == -1)
    	B.st = vec2(GetTimeDomain(index),0.);
    if(U.y >= 0 && U.y <= 8)
    {
        int burb = 1<<U.y; 
        int kn = U.x & (burb-1);
        float C = -_PI*float(kn)/float(burb);
        vec2 W = sin(vec2(C+_Half_PI,C));
        int E = 1-((U.x>>U.y)&1);
        B.xy = GetFFT(U.x-burb*(1-E),U.y);
        B.xy += mat2(W,-W.y,W.x)*GetFFT(U.x+burb*E,U.y)*float(E*2-1);
    }
	return GetFFT(U.x,9).xy;
}

// Function 45
float fft_sdf(vec2 p) {
    // convert x from uv space [0..1] to fft index [0..511]
    #if LOG_SCALE
    	float fft_x = logyscale(p.x, 0.0, 1.0, 1.0, AUDIO_IN_SIZE-2.0);
	#else
    	float fft_x = linscale(p.x, 0.0, 1.0, 1.0, AUDIO_IN_SIZE-2.0);
    #endif
    
    // closest previous sample
    float fft_x1 = floor(fft_x); 
    float fft_y1 = texelFetch(AUDIO_IN, ivec2(fft_x1, 0), 0).x;

    // closest next sample
    float fft_x2 = ceil(fft_x); 
    float fft_y2 = texelFetch(AUDIO_IN, ivec2(fft_x2, 0), 0).x;
    
    return sd_line(p, vec2(fft_x1, fft_y1), vec2(fft_x2, fft_y2));   
}

// Function 46
float getfrequency(float x) {
	return texture(iChannel0, vec2(floor(x * FREQ_RANGE + 1.0) / FREQ_RANGE, 0.25)).x*gain + 0.06;
}

// Function 47
float spectrum(float domain, int t, int level)
{
    float sixty_fourth = 1./32.;
    vec2 uv = vec2(float(t)*2./sixty_fourth + sixty_fourth, domain);
    uv = upper_right(uv); level++;
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(iChannel3, uv).y;
}

// Function 48
vec3 spectrum_to_xyz(in float w)
{
    w = clamp(w, MIN_WL, MAX_WL) - MIN_WL;
    float n = floor(w / WL_STEP);
    int n0 = min(SPECTRUM_SAMPLES - 1, int(n));
    int n1 = min(SPECTRUM_SAMPLES - 1, n0 + 1);
    float t = w - (n * WL_STEP);
    return mix(cie[n0], cie[n1], t / WL_STEP);
}

// Function 49
float frequencyFromNotenum( float n ) { return 440.0 * pow( 2.0, ( n-49.0) / 12.0 ); }

// Function 50
float spectrum(float domain, int t, int level, sampler2D bufD)
{
    float sixty_fourth = 1./32.;
    vec2 uv = vec2(float(t)*3.*sixty_fourth + sixty_fourth, domain);
    uv = upper_right(uv); level++;
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(bufD, uv).x;
}

// Function 51
vec2 GPUFFT(ivec2 U,inout vec2 B){
    --U.y;
    int index = BitReverse(U.x,BitWidth);
    if(U.y == -1)
    	B.st = vec2(GetTimeDomain(index),0.);
    if(U.y >= 0 && U.y <= 8)
    {
        int burb = 1<<U.y; 
        int kn = U.x & (burb-1);
        float C = -_PI*float(kn)/float(burb);
        vec2 W = sin(vec2(C+_Half_PI,C));
        int E = 1-((U.x>>U.y)&1);
        B.xy = GetFFT(U.x-burb*(1-E),U.y);
        B.xy += mat2(W,-W.y,W.x)*GetFFT(U.x+burb*E,U.y)*float(E*2-1);
    }
	return GetFFT(U.x,9).xy;
}

// Function 52
float FrequencyToTexture(float Frequency){
    if (Frequency>=300.0) {
        return Frequency/440.*ATone;
    } else if ((Frequency>=130.0) && (Frequency<300.0)) {
        return Frequency/440.*ATone2;
    } else{
        return Frequency/440.*ATone3;
    }
}

// Function 53
float fftBand(float start, float end) {
   	float fft = 0.0;
    float st = (end - start) *FFT_SAMPLES_INV;
    float x = start;
    for(int i = 0; i < FFT_SAMPLES; i++) {
 		fft += texture( iChannel0, vec2(x,0.25) ).x;     
        x += st;
    }
    
    fft *= FFT_SAMPLES_INV;
 	return fft;
}

// Function 54
void FFTBand_update(inout FFTBand band, float value)
{
    band.f += band.df * value;
    band.df = vec2(
        band.df.x * band.di.x - band.df.y * band.di.y,
        band.df.y * band.di.x + band.df.x * band.di.y
        );
}

// Function 55
float KeyToFrequency(int n){
    return pow(Semitone,float(n-49))*440.;
}

// Function 56
float spectrum2D(vec2 uv, float thickness, int level, sampler2D bufD)
{
    float val = spectrum(uv.x, 0, level, bufD);
    return (abs(uv.y - val) < thickness/2.) ? (1.-abs(uv.y - val)*2./thickness) : 0.;
}

// Function 57
vec3 fft(float freq,float time){
    return texture(sound,vec2(freq,time)).rgb;
}

// Function 58
vec4 fft(float freq,float time){
    return texture(sound,vec2(freq,time));
}

// Function 59
float frequency(float index)
{
return texture(iChannel0, vec2(index, 0)).x;
}

// Function 60
vec3 spectrum_offset( float t ) {
    float t0 = 3.0 * t - 1.5;
	return clamp( vec3( -t0, 1.0-abs(t0), t0), 0.0, 1.0);
    /*
	vec3 ret;
	float lo = step(t,0.5);
	float hi = 1.0-lo;
	float w = linterp( remap( t, 1.0/6.0, 5.0/6.0 ) );
	float neg_w = 1.0-w;
	ret = vec3(lo,1.0,hi) * vec3(neg_w, w, neg_w);
	return pow( ret, vec3(1.0/2.2) );
*/
}

// Function 61
float fft(float f,float r,float time){
    float sum = 0.0;
    float val = 0.0;
    float coeff = 0.0;
    
    float k = 0.0;
    
     // loop sampling
    for( int i = 0; i < fftSamples ; i++ ){
        k = float(i)/float(fftSamples-1)-0.5;
        // decreasing factor, more important around 0
        coeff = exp(-k*k/(fftSmooth*fftSmooth)*2.0);
        //coeff = 1.0;
        
		val += texture(inputSound, vec2( remapFreq(f + k * r)*fftWidth, time) ).r * coeff;
        
        // simulation for test
        //float freq = ( remapFreq(f + k * r)*fftWidth - 0.5 ) / 0.008;//(iMouse.x/iResolution.x);
		//val += exp(  - freq*freq/2.0 ) * coeff;
        
        sum += coeff;
    }
    
    return remapIntensity(f,val/sum);
    
}

// Function 62
void zxspectrum_clash( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 pv = floor(fragCoord.xy/LOWREZ);
    vec2 bv = floor(pv/8.0)*8.0;
    vec2 sv = floor(iResolution.xy/LOWREZ);
    
    
    vec4 min_cs=vec4(1.0,1.0,1.0,1.0);
    vec4 max_cs=vec4(0.0,0.0,0.0,0.0);
    float bright=0.0;

    
    for(int py=1;py<8;py++)
    {
        for(int px=0;px<8;px++)
        {
		    vec4 cs=bmap( (texture(iChannel0,(bv+vec2(px,py))/sv).rgb) );
            bright+=cs.a;
        	min_cs=min(min_cs,cs);
        	max_cs=max(max_cs,cs);
        }
    }
    
    vec4 c;
    
    if(bright>=24.0)
    {
        bright=1.0;
    }
    else
    {
        bright=0.0;
    }
    
    if( max_cs.rgb==min_cs.rgb )
    {
        min_cs.rgb=vec3(0.0,0.0,0.0);
    }

    if( max_cs.rgb==vec3(0.0,0.0,0.0) )
    {
        bright=0.0;
        max_cs.rgb=vec3(0.0,0.0,1.0);
        min_cs.rgb=vec3(0.0,0.0,0.0);
    }
    
    vec3 c1=fmap(vec4(max_cs.rgb,bright));
    vec3 c2=fmap(vec4(min_cs.rgb,bright));
    
    vec3 cs=texture(iChannel0,pv/sv).rgb;
    
    vec3 d= (cs+cs) - (c1+c2) ;
    float dd=d.r+d.g+d.b;

    if( mod(pv.x+pv.y,2.0)==1.0)
    {
        fragColor=vec4(
                dd>=-(DITHER*0.5) ? c1.r : c2.r,
                dd>=-(DITHER*0.5) ? c1.g : c2.g,
                dd>=-(DITHER*0.5) ? c1.b : c2.b,
                1.0);
    }
    else
    {
        fragColor=vec4(
                dd>=(DITHER*0.5) ? c1.r : c2.r,
                dd>=(DITHER*0.5) ? c1.g : c2.g,
                dd>=(DITHER*0.5) ? c1.b : c2.b,
                1.0);
    }
 
//    fragColor.rgb=c1;
}

// Function 63
float spectrum(float domain, int t, int level)
{
    float sixty_fourth = 1./32.;
    vec2 uv = vec2(float(t)*3.*sixty_fourth + sixty_fourth, domain);
    uv = upper_right(uv); level++;
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(iChannel3, uv).y;
}

// Function 64
float GetSpectrumGlare( float fDist )
{
    float fGlare = 0.0;
    fDist = 1.0f - fDist;
    if ( fDist < 0.0 )
        return 0.0;
    
    fGlare += pow( fDist, 30.0);
    fGlare += pow( fDist, 10.0) * 0.1;
    //return UnitGaussian( fDist, 0.0, 20.0 );
    
    return fGlare * 0.005;
}

// Function 65
float fft(float x) {
    // convert x from uv space [0..1] to fft index [0..511]
    #if LOG_SCALE
    	float fft_x = logyscale(x, 0.0, 1.0, 2.0, AUDIO_IN_SIZE-1.0);
	#else
    	float fft_x = linscale(x, 0.0, 1.0, 2.0, AUDIO_IN_SIZE-1.0);
    #endif
    
    // sample before closest previous sample
    float fft_x0 = floor(fft_x) - 1.0;
    float fft_y0 = texelFetch(AUDIO_IN, ivec2(fft_x0, 0), 0).x;    
    
    // closest previous sample
    float fft_x1 = floor(fft_x); 
    float fft_y1 = texelFetch(AUDIO_IN, ivec2(fft_x1, 0), 0).x;

    // closest next sample
    float fft_x2 = ceil(fft_x); 
    float fft_y2 = texelFetch(AUDIO_IN, ivec2(fft_x2, 0), 0).x;
    
    // sample after closest next sample
    float fft_x3 = ceil(fft_x) + 1.0; 
    float fft_y3 = texelFetch(AUDIO_IN, ivec2(fft_x3, 0), 0).x;
    
    vec4 fft_xs = vec4(fft_x0, fft_x1, fft_x2, fft_x3);
    vec4 fft_ys = vec4(fft_y0, fft_y1, fft_y2, fft_y3);

    #if INTERP == 2
    // cubic interpolation (smooth corners)
    float fft_y;
    if (x > 0.5)
        fft_y = linscale(fft_x, fft_x1, fft_x2, fft_y1, fft_y2);
    else
		fft_y = cubic_interp(fft_x, fft_xs, fft_ys);
    
    #elif INTERP == 1
    // linear interpolation (sharp corners)
    float fft_y = linscale(fft_x, fft_x1, fft_x2, fft_y1, fft_y2);
    
    #elif INTERP == 0
    // nearest neighbor interpolation
    float fft_y = fft_y1;
    #endif
    
    /*
    float xbar = (fft_x1+fft_x2)/2.0;
    if (between(fft_x, vec2(fft_x1, xbar)) > 0.0) {
    	fft_y = linscale(fft_x, fft_x1, xbar, fft_y1, 0.0);
    } else {
    	fft_y = linscale(fft_x, xbar, fft_x2, 0.0, fft_y2);
    }
	*/
    
    return pow(fft_y, FFT_POW);
}

// Function 66
vec2 fft(vec2 uv)
{
    vec2 complex = vec2(0,0);
    
    uv *= float(FFT_SIZE);
    
    float size = float(FFT_SIZE);
    
    for(int x = 0;x < FFT_SIZE;x++)
    {
    	for(int y = 0;y < FFT_SIZE;y++)
    	{
            float a = 2.0 * PI * (uv.x * (float(x)/size) + uv.y * (float(y)/size));
            vec3 samplev = texture(iChannel0,mod(vec2(x,y)/size,1.0)).rgb;
            complex += avg(samplev)*vec2(cos(a),sin(a));
        }
    }
    
    return complex;
}

// Function 67
float fftB(float f){
    float sum = 0.0;
    float val = 0.0;
    float coeff = 0.0;
    float k = 0.0;
    for( int i = 0; i < fftSamplesB ; i++ ){
        k = float(i)/float(fftSamplesB-1)-0.5;
        coeff = exp(-k*k/(fftSmooth*fftSmooth)*2.0);
		val += texture(sound, vec2( remapFreq(f + k * fftRadiusB)*fftWidth, 0.0) ).r * coeff;
        sum += coeff;
    }
    return remapIntensity(f,val/sum)*fftGBGain*fftGBGain;
}

// Function 68
float fft(float x)
{
    return max( texture(iChannel1,vec2(x,.25)).x - .2 , .0 )*2.;
}

// Function 69
void GPU_FFT_Visual(vec2 u,inout vec4 c){
	//u.x -= 100.;
    //u.y -= (0.55*iResolution.y); //128.
    u.x = floor(u.x/iResolution.y*180.)-30.;
    u.y = u.y/iResolution.y*350.-170.;
    //show visual effect
    if(u.x-0.5 < 256. && u.x>=0. && u.y>0.){
        vec2 xy = texture(iChannel0,vec2(u.x+0.5,9.5)/iChannelResolution[0].xy).xy;
        //Energy
        vec4 cc = vec4(length(xy));
        //Really Energy 
    	cc = u.x == 0. ? cc/512. : cc/256.;
        cc *= u.x == 0. ? 2.: 20.;
        cc = 20.*log(cc);
        cc = vec4(clamp(cc.r,1.,120.),0.,0.,0.);
        //show visual effect
        if(floor(u.y) < cc.r ){
    		c.g = 1.;
        }
    }   
}

// Function 70
float fftmul(float i){
    return i*fftH*(i*fftH+0.8)*1.5 + 0.1;
}

// Function 71
FFTBand FFTBand_create(const float n, const int fft_size)
{
    FFTBand band;

    float fi = (float(n) / float(fft_size / 2)) * float(fft_size) * 0.5;
    float angle = 2.0 * PI * fi / float(fft_size);

    band.di = vec2(cos(angle), sin(angle));
    band.df = vec2(1.0 / float(fft_size), 0.0);
    band.f  = vec2(0.0, 0.0);

    return band;
}


```