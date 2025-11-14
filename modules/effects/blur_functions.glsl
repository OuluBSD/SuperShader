// Reusable Blur Effect Functions
// Automatically extracted from effect-related shaders

// Function 1
vec4 blur(vec2 uv) {
	vec4 col = vec4(0.0);
	for(float r0 = 0.0; r0 < 1.0; r0 += ST) {
		float r = r0 * CV;
		for(float a0 = 0.0; a0 < 1.0; a0 += ST) {
			float a = a0 * PI2;
			col += colorat(uv + vec2(cos(a), sin(a)) * r);
		}
	}
	col *= ST * ST;
	return col;
}

// Function 2
vec3 Blur(vec2 uv, float radius)
{
	radius = radius * .04;
    
    vec2 circle = vec2(radius) * vec2((iResolution.y / iResolution.x), 1.0);
    
	// Remove the time reference to prevent random jittering if you don't like it.
	vec2 random = Hash22(uv+iTime);

    // Do the blur here...
	vec3 acc = vec3(0.0);
	for (int i = 0; i < ITERATIONS; i++)
    {
		acc += texture(iChannel0, uv + circle * Sample(random), radius*10.0).xyz;
    }
	return acc / float(ITERATIONS);
}

// Function 3
vec4 naiveBlur(sampler2D input_samp, vec2 center, vec2 sampleDist, float samplesPerDir, bool gammaCorrect){
	vec4 col_samples;
    int noOfSamples = 0;
    for(float rp_i_y = 0.; rp_i_y < samplesPerDir; rp_i_y++){
        for(float rp_i_x = 0.; rp_i_x < samplesPerDir; rp_i_x++){
            vec2 unscaled_rel_pos = vec2( (rp_i_x - (samplesPerDir/2.)),
                                         (rp_i_y - (samplesPerDir/2.)));
            if(length(unscaled_rel_pos) <= (samplesPerDir/2.)){
                noOfSamples++;
                vec4 col_sample = texture(input_samp, center - sampleDist*
                                          unscaled_rel_pos
                                       );
                if(!gammaCorrect)
                    col_samples += col_sample;
                else
                    #ifdef FASTGAMMA
                    col_samples += vec4(col_sample.rgb*col_sample.rgb, col_sample.a);
                    #else
                    col_samples += vec4(pow(col_sample.rgb, vec3(2.2)), col_sample.a);
                    #endif
            }
            
        }
    }
    col_samples /= float(noOfSamples);
    if(!gammaCorrect)
        return col_samples;
    else
        #ifdef FASTGAMMA
        return vec4(pow(col_samples.rgb, vec3(1./2.2)), col_samples.a);
    	#else
    	return vec4(sqrt(col_samples.rgb), col_samples.a);
    	#endif
}

// Function 4
vec3 BlurredPixel (in vec2 uv)
{   
    int c_distX = iMouse.z > 0.0
        ? int(float(c_halfSamplesX+1) * iMouse.x / iResolution.x)
        : int((sin(iTime*2.0)*0.5 + 0.5) * float(c_halfSamplesX+1));
    
	int c_distY = iMouse.z > 0.0
        ? int(float(c_halfSamplesY+1) * iMouse.y / iResolution.y)
        : int((sin(iTime*2.0)*0.5 + 0.5) * float(c_halfSamplesY+1));
    
    float c_pixelWeight = 1.0 / float((c_distX*2+1)*(c_distY*2+1));
    
    vec3 ret = vec3(0);        
    for (int iy = -c_halfSamplesY; iy <= c_halfSamplesY; ++iy)
    {
        for (int ix = -c_halfSamplesX; ix <= c_halfSamplesX; ++ix)
        {
            if (abs(float(iy)) <= float(c_distY) && abs(float(ix)) <= float(c_distX))
            {
                vec2 offset = vec2(ix, iy) * c_pixelSize;
            	ret += texture(iChannel0, uv + offset).rgb * c_pixelWeight;
            }
        }
    }
    return ret;
}

// Function 5
void boxBlur( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    
	int kernel_window_size = BOXRADIUS*2+1;
    int samples = kernel_window_size*kernel_window_size;
    
    highp vec4 color = vec4(0);
    
    highp float wsum = 0.0;
    for (int ry = -BOXRADIUS; ry <= BOXRADIUS; ++ry)
    for (int rx = -BOXRADIUS; rx <= BOXRADIUS; ++rx)
    {
        highp float w = 1.0;
        wsum += w;
    	color += texture(iChannel0, uv+vec2(rx,ry)/iResolution.xy)*w;
    }
    
    fragColor = color/wsum;
}

// Function 6
int DetermineBlurType(float depth, int materialID)
{
    if(depth > DOF_BLUR_START)
    {
        return DOF_BLUR;
    }
    else if(materialID == SNOW_MATERIAL_ID)
    {
        return SNOW_BLUR;
    }
    else if(materialID == ICE_MATERIAL_ID)
    {
        return ICE_BLUR;
    }
    else
    {
        return NO_BLUR;
    }
}

// Function 7
vec3 calcBlur(vec2 texcoord, vec2 pixelSize){
	const int steps = blurSteps;
    
    float totalWeight = 0.0;
    vec3 totalColor = vec3(0.0);
    
    float offsetSize = pixelSize.x * blurOSize;
    
    for (int i = -steps; i <= steps; ++i){
        float offset = float(i);
        float x = abs(offset / blurOSize * ramp);
		float weight = distribution(x);
        
        totalColor += texture(iChannel0, texcoord + vec2(offset * offsetSize, 0.0)).rgb * weight;
        totalWeight += weight;
    }
    
    return decodeColor(totalColor / totalWeight);
}

// Function 8
vec4 BlurV (sampler2D source, vec2 size, vec2 uv, float radius) {

	if (radius >= 1.0)
	{
		vec4 A = vec4(0.0); 
		vec4 C = vec4(0.0); 

		float height = 1.0 / size.y;

		float divisor = 0.0; 
        float weight = 0.0;
        
        float radiusMultiplier = 1.0 / radius;

        for (float y = -20.0; y <= 20.0; y++)
		{
			A = texture(source, uv + vec2(0.0, y * height));
            	
            	weight = SCurve(1.0 - (abs(y) * radiusMultiplier)); 
            
            	C += A * weight; 
            
			divisor += weight; 
		}

		return vec4(C.r / divisor, C.g / divisor, C.b / divisor, 1.0);
	}

	return texture(source, uv);
}

// Function 9
vec4 blur_horizontal(sampler2D channel, vec2 uv, float scale)
{
    float h = scale / iResolution.x;
    vec4 sum = vec4(0.0);

    sum += texture(channel, fract(vec2(uv.x - 4.0*h, uv.y)) ) * 0.0162162162;
    sum += texture(channel, fract(vec2(uv.x - 3.0*h, uv.y)) ) * 0.0540540541;
    sum += texture(channel, fract(vec2(uv.x - 2.0*h, uv.y)) ) * 0.1216216216;
    sum += texture(channel, fract(vec2(uv.x - 1.0*h, uv.y)) ) * 0.1945945946;
    sum += texture(channel, fract(vec2(uv.x + 0.0*h, uv.y)) ) * 0.2270270270;
    sum += texture(channel, fract(vec2(uv.x + 1.0*h, uv.y)) ) * 0.1945945946;
    sum += texture(channel, fract(vec2(uv.x + 2.0*h, uv.y)) ) * 0.1216216216;
    sum += texture(channel, fract(vec2(uv.x + 3.0*h, uv.y)) ) * 0.0540540541;
    sum += texture(channel, fract(vec2(uv.x + 4.0*h, uv.y)) ) * 0.0162162162;

    return sum;
}

// Function 10
vec4 BlurB(vec2 uv, int level)
{
    return BlurB(uv, level, iChannel1, iChannel3);
}

// Function 11
vec4 blur(vec2 uv, vec2 TexelSize, vec2 Direction)
{
    vec4 c = vec4(0.0);
    
    c += texture(iChannel0, uv + (TexelSize*Direction*.5))*0.49;
    c += texture(iChannel0, uv + (TexelSize*Direction*1.5))*0.33;
    c += texture(iChannel0, uv + (TexelSize*Direction*2.5))*0.14;
    c += texture(iChannel0, uv + (TexelSize*Direction*3.5))*9.0;
    c += texture(iChannel0, uv + (TexelSize*Direction*4.5))*0.01;
    c += texture(iChannel0, uv - (TexelSize*Direction*.5))*0.49;
    c += texture(iChannel0, uv - (TexelSize*Direction*1.5))*0.33;
    c += texture(iChannel0, uv - (TexelSize*Direction*2.5))*0.14;
    c += texture(iChannel0, uv - (TexelSize*Direction*3.5))*0.04;
    c += texture(iChannel0, uv - (TexelSize*Direction*4.5))*0.01;
    
    return c/2.0;
}

// Function 12
vec3 blur(sampler2D sp, vec2 uv, vec2 dir, int samples)
{
    float halfSamples = float(samples) * 0.5;
    uv -= dir * halfSamples;
    
    float x = -halfSamples;
    float weight = gaussian(abs(x++), BlurSigma);
    vec4 color = vec4(texture(sp, uv).rgb, 1.0) * weight;
    
    for (int i = 1; i < samples; ++i)
    {
        uv += dir;
        weight = gaussian(abs(x++), BlurSigma);
        color += vec4(texture(sp, uv).rgb, 1.0) * weight;
    }
    
    color.rgb /= color.a;
    return color.rgb;
}

// Function 13
vec3 optimizedBoxBlur(
    in sampler2D tex,
    in vec2 uv,
    in vec2 resolution,
    in vec2 direction,
    in float size,
    in int samples
) {
    float f_samples = float( samples );
    float w = 1. / f_samples;
    vec2 px = 1. / resolution;
    float increment = size / ( f_samples - 1. );
    float halfSize = size * .5;
    vec2 dpx = direction * px;

    vec3 color = vec3( 0. );
    for ( float i = -halfSize; i <= halfSize; i += increment ) {
        vec2 st = uv + dpx * i;
        color += texture( tex, st ).rgb * w;
    }
    
    return color;
}

// Function 14
vec4 boxBlur(vec2 uv, float scale) {
    // Simple box blurring
    const int numSteps = 15;
    
    uv = ((uv * 2. - 1.) *scale) * .5 + .5;
    
    vec4 acc = texture(iChannel0, uv);
    vec2 stepI = 1./iResolution.xy;
    stepI *= scale;
    vec2 offsetU = vec2(0.0);
    vec2 offsetD = vec2(0.0);
    
    for (int j = 0; j < numSteps; j++) {
        offsetU.y += stepI.y;
        offsetU.x = 0.;
        for (int i = 0; i < numSteps; i++) {
            acc += pow(texture(iChannel0, uv + offsetU), vec4(2.2));
            acc += pow(texture(iChannel0, uv - offsetU), vec4(2.2));
            offsetU.x += stepI.x;
        }
    
        offsetD.y -= stepI.y;
        offsetD.x = 0.;
        for (int i = 0; i < numSteps; i++) {
            acc += pow(texture(iChannel0, uv + offsetD), vec4(2.2));
            acc += pow(texture(iChannel0, uv - offsetD), vec4(2.2));
            offsetD.x += stepI.x;
        }
    }
    
    // Gamma correction is added, as it's done by iq here: https://www.shadertoy.com/view/XtsSzH
    return pow(acc / (float(numSteps * numSteps * 4) + 1.), vec4(1. / 2.2));
    
}

// Function 15
vec4 blur(sampler2D sp, vec2 U, vec2 scale) {
    vec4 O = vec4(0);  
    int s = samples/sLOD;
    
    for ( int i = 0; i < s*s; i++ ) {
        vec2 d = vec2(i%s, i/s)*float(sLOD) - float(samples)/2.;
        O += gaussian(d) * textureLod( sp, U + scale * d , float(LOD) );
    }
    
    return O / O.a;
}

// Function 16
vec3 blur(vec3 col, vec2 tc, float offs)
{
	vec4 xoffs = offs * vec4(-2.0, -1.0, 1.0, 2.0) / iResolution.x;
	vec4 yoffs = offs * vec4(-2.0, -1.0, 1.0, 2.0) / iResolution.y;
	
	vec3 color = vec3(0.0, 0.0, 0.0);
	color += hsample(col, tc + vec2(xoffs.x, yoffs.x)) * 0.00366;
	color += hsample(col, tc + vec2(xoffs.y, yoffs.x)) * 0.01465;
	color += hsample(col, tc + vec2(    0.0, yoffs.x)) * 0.02564;
	color += hsample(col, tc + vec2(xoffs.z, yoffs.x)) * 0.01465;
	color += hsample(col, tc + vec2(xoffs.w, yoffs.x)) * 0.00366;
	
	color += hsample(col, tc + vec2(xoffs.x, yoffs.y)) * 0.01465;
	color += hsample(col, tc + vec2(xoffs.y, yoffs.y)) * 0.05861;
	color += hsample(col, tc + vec2(    0.0, yoffs.y)) * 0.09524;
	color += hsample(col, tc + vec2(xoffs.z, yoffs.y)) * 0.05861;
	color += hsample(col, tc + vec2(xoffs.w, yoffs.y)) * 0.01465;
	
	color += hsample(col, tc + vec2(xoffs.x, 0.0)) * 0.02564;
	color += hsample(col, tc + vec2(xoffs.y, 0.0)) * 0.09524;
	color += hsample(col, tc + vec2(    0.0, 0.0)) * 0.15018;
	color += hsample(col, tc + vec2(xoffs.z, 0.0)) * 0.09524;
	color += hsample(col, tc + vec2(xoffs.w, 0.0)) * 0.02564;
	
	color += hsample(col, tc + vec2(xoffs.x, yoffs.z)) * 0.01465;
	color += hsample(col, tc + vec2(xoffs.y, yoffs.z)) * 0.05861;
	color += hsample(col, tc + vec2(    0.0, yoffs.z)) * 0.09524;
	color += hsample(col, tc + vec2(xoffs.z, yoffs.z)) * 0.05861;
	color += hsample(col, tc + vec2(xoffs.w, yoffs.z)) * 0.01465;
	
	color += hsample(col, tc + vec2(xoffs.x, yoffs.w)) * 0.00366;
	color += hsample(col, tc + vec2(xoffs.y, yoffs.w)) * 0.01465;
	color += hsample(col, tc + vec2(    0.0, yoffs.w)) * 0.02564;
	color += hsample(col, tc + vec2(xoffs.z, yoffs.w)) * 0.01465;
	color += hsample(col, tc + vec2(xoffs.w, yoffs.w)) * 0.00366;

	return color;
}

// Function 17
vec3 depthDirectionalBlur(sampler2D tex, float z, float coc, vec2 uv, vec2 blurvec)  
{  
    // z: z at UV  
    // coc: blur radius at UV  
    // uv: initial coordinate  
    // blurvec: smudge direction  
    // numSamples: blur taps  
    vec3 sumcol = vec3(0.);  
  
    for (int i = 0; i < NUM_SAMPLES; ++i)  
    {  
        float r =  (float(i) + hash1(uv + float(i) + 1.) - 0.5)  / (float(NUM_SAMPLES) - 1.) - 0.5;  
        vec2 p = uv + r * coc * blurvec;  
        vec4 smpl = texture(tex, p);  
        if(smpl.w < z) // if sample is closer consider it's CoC  
        {  
            p = uv + r * min(coc, CoC(smpl.w)) * blurvec;  
            p = uv + r * CoC(smpl.w) * blurvec;  
            smpl = texture(tex, p);  
        }  
        sumcol += smpl.xyz;  
    }  
  
    sumcol /= float(NUM_SAMPLES);  
    sumcol = max(sumcol, 0.0);  
  
    return sumcol;  
}

// Function 18
float getBlurSize(float depth, float focusPoint, float focusScale)
{
	float coc = clamp((1.0 / focusPoint - 1.0 / depth)*focusScale, -1.0, 1.0);
	return abs(coc) * MAX_BLUR_SIZE;
}

// Function 19
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

// Function 20
vec4 blur_vertical_left_column(vec2 uv, int depth)
{
    float v = pow(2., float(depth)) / iResolution.y;

    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;
    
    uv1 = fract(vec2(uv.x, uv.y - 4.0*v) * 2.);
    uv2 = fract(vec2(uv.x, uv.y - 3.0*v) * 2.);
    uv3 = fract(vec2(uv.x, uv.y - 2.0*v) * 2.);
    uv4 = fract(vec2(uv.x, uv.y - 1.0*v) * 2.);
    uv5 = fract(vec2(uv.x, uv.y + 0.0*v) * 2.);
    uv6 = fract(vec2(uv.x, uv.y + 1.0*v) * 2.);
    uv7 = fract(vec2(uv.x, uv.y + 2.0*v) * 2.);
    uv8 = fract(vec2(uv.x, uv.y + 3.0*v) * 2.);
    uv9 = fract(vec2(uv.x, uv.y + 4.0*v) * 2.);

    if(uv.x < 0.5)
    {
        if(uv.y > 0.5)
        {
            uv1 = upper_left(uv1);
            uv2 = upper_left(uv2);
            uv3 = upper_left(uv3);
            uv4 = upper_left(uv4);
            uv5 = upper_left(uv5);
            uv6 = upper_left(uv6);
            uv7 = upper_left(uv7);
            uv8 = upper_left(uv8);
            uv9 = upper_left(uv9);
        }
        else
        {
            uv1 = lower_left(uv1);
            uv2 = lower_left(uv2);
            uv3 = lower_left(uv3);
            uv4 = lower_left(uv4);
            uv5 = lower_left(uv5);
            uv6 = lower_left(uv6);
            uv7 = lower_left(uv7);
            uv8 = lower_left(uv8);
            uv9 = lower_left(uv9);
        }
    }
    else
    {
        vec2 uv_s = upper_right(uv*2.)*2.;
        uv1 = clamp(vec2(uv_s.x, uv_s.y - 4.0*v), 0., 1.);
        uv2 = clamp(vec2(uv_s.x, uv_s.y - 3.0*v), 0., 1.);
        uv3 = clamp(vec2(uv_s.x, uv_s.y - 2.0*v), 0., 1.);
        uv4 = clamp(vec2(uv_s.x, uv_s.y - 1.0*v), 0., 1.);
        uv5 = clamp(vec2(uv_s.x, uv_s.y + 0.0*v), 0., 1.);
        uv6 = clamp(vec2(uv_s.x, uv_s.y + 1.0*v), 0., 1.);
        uv7 = clamp(vec2(uv_s.x, uv_s.y + 2.0*v), 0., 1.);
        uv8 = clamp(vec2(uv_s.x, uv_s.y + 3.0*v), 0., 1.);
        uv9 = clamp(vec2(uv_s.x, uv_s.y + 4.0*v), 0., 1.);
        depth--;
        uv1 = upper_right(uv1);
        uv2 = upper_right(uv2);
        uv3 = upper_right(uv3);
        uv4 = upper_right(uv4);
        uv5 = upper_right(uv5);
        uv6 = upper_right(uv6);
        uv7 = upper_right(uv7);
        uv8 = upper_right(uv8);
        uv9 = upper_right(uv9);
    }
    for(int level = 0; level < 8; level++)
    {
        if(level > depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);
    if(uv.x > 0.5 && uv.y > 0.5)
    {
        //return vec4(0);
		sum += texture(iChannel3, uv1) * 0.05;
        sum += texture(iChannel3, uv2) * 0.09;
        sum += texture(iChannel3, uv3) * 0.12;
        sum += texture(iChannel3, uv4) * 0.15;
        sum += texture(iChannel3, uv5) * 0.16;
        sum += texture(iChannel3, uv6) * 0.15;
        sum += texture(iChannel3, uv7) * 0.12;
        sum += texture(iChannel3, uv8) * 0.09;
        sum += texture(iChannel3, uv9) * 0.05;
    }
    else
    {
        sum += texture(iChannel2, uv1) * 0.05;
        sum += texture(iChannel2, uv2) * 0.09;
        sum += texture(iChannel2, uv3) * 0.12;
        sum += texture(iChannel2, uv4) * 0.15;
        sum += texture(iChannel2, uv5) * 0.16;
        sum += texture(iChannel2, uv6) * 0.15;
        sum += texture(iChannel2, uv7) * 0.12;
        sum += texture(iChannel2, uv8) * 0.09;
        sum += texture(iChannel2, uv9) * 0.05;
    }
    return sum/0.98; // normalize
}

// Function 21
vec4 box_blur(sampler2D sp, vec2 uv, vec2 dir) {
	vec4 result = vec4(0.0);
    uv -= dir * BOX_BLUR_OFFSET;
    
    for (int i = 0; i < BOX_BLUR_SAMPLES; ++i) {
        result += texture(sp, uv);
        uv += dir * BOX_BLUR_SCALE;
    }
    result *= vec4(BOX_BLUR_ACCUM);
    
    return result;
}

// Function 22
vec3 blur(vec2 uv, vec2 coords)
{
 
	vec2 noise = rand(uv.xy);
	float tolerance = 0.2;
	float vignette_size = 0.5;
	vec2 powers = pow(abs(vec2(uv.s - 0.5,uv.t - 0.5)),vec2(2.0));
	float radiusSqrd = pow(vignette_size,2.0);
	float gradient = smoothstep(radiusSqrd-tolerance, radiusSqrd+tolerance, powers.x+powers.y);

	vec4 col = vec4(0.0);

	float X1 = coords.x + blurAmount * noise.x*0.004 * gradient;
	float Y1 = coords.y + blurAmount * noise.y*0.004 * gradient;
	float X2 = coords.x - blurAmount * noise.x*0.004 * gradient;
	float Y2 = coords.y - blurAmount * noise.y*0.004 * gradient;
	
	float invX1 = coords.x + blurAmount * ((1.0-noise.x)*0.004) * (gradient * 0.5);
	float invY1 = coords.y + blurAmount * ((1.0-noise.y)*0.004) * (gradient * 0.5);
	float invX2 = coords.x - blurAmount * ((1.0-noise.x)*0.004) * (gradient * 0.5);
	float invY2 = coords.y - blurAmount * ((1.0-noise.y)*0.004) * (gradient * 0.5);

	
	col += texture(iChannel0, vec2(X1, Y1))*0.1;
	col += texture(iChannel0, vec2(X2, Y2))*0.1;
	col += texture(iChannel0, vec2(X1, Y2))*0.1;
	col += texture(iChannel0, vec2(X2, Y1))*0.1;
	
	col += texture(iChannel0, vec2(invX1, invY1))*0.15;
	col += texture(iChannel0, vec2(invX2, invY2))*0.15;
	col += texture(iChannel0, vec2(invX1, invY2))*0.15;
	col += texture(iChannel0, vec2(invX2, invY1))*0.15;
	
	return col.rgb;
}

// Function 23
vec4 blur_linear(sampler2D tex, vec2 texel, vec2 uv, vec2 line)
{
    vec4 total = vec4(0);
    
    float dist = 1.0/SAMPLES;
    for(float i = -0.5; i<=0.5; i+=dist)
    {
        vec2 coord = uv+i*line*texel;
        total += texture(tex,coord);
    }
    
    return total * dist;
}

// Function 24
vec3 simple_blur(sampler2D sp, vec2 uv, vec2 dir) {
    vec3 color = vec3(0.0);
    
    // Initial offset so that the result is later centered.
    uv -= dir * vec2(BlurHalfSamples);
    
    // Explanation:
    //  A: Starting pixel.
    //  B: Ending pixel.
    //  C: Center of the blur kernel.
    //
    // Before offset:
    //               v~~~ The initial coordinate is here.
    // | - | - | - | A | x | x | C | x | x | B |
    //                           ^~~~ The center gets shifted all the way here.
    //
    // After offset:
    //   v~~~ We offset backwards...
    // | A | x | x | C | x | x | B | - | - | - |
    //               ^~~~ ...so that the center remains where it was before the blur.
    
    for (int i = 0; i < BlurSamples; ++i)
        color += texture(sp, uv + dir * float(i)).rgb;
    
    color *= BlurInvSamples;
    return color;
}

// Function 25
vec4 outlineBlur(sampler2D sampler, vec2 uv, float outlineSize)
{
 	vec4 blur = blurTexture(sampler, uv, 4.0, 7.0);
    vec4 col = 1.0 - smoothstep(0.0, 0.5, abs(blur - 0.5));
    
    return col;
}

// Function 26
float calc_aa_blur(float w) {
    vec2 blur = _stack.blur;
    w -= blur.x;
    float wa = clamp(-w*AA, 0.0, 1.0);
    float wb = clamp(-w / blur.x + blur.y, 0.0, 1.0);    
	return wa * wb; //min(wa,wb);    
}

// Function 27
vec4 blur(sampler2D sp, vec2 U, vec2 scale) {
        vec4 O = vec4(0);
        int s = samples / sLOD;

        for (int i = 0; i < s * s; i++) {
            vec2 d = vec2(i % s, i / s) * float(sLOD) - float(samples) / 2.;
            vec4 temp = textureLod(sp, U + scale * d, float(LOD));
            O += vec4(gaussian(d) * temp);
        }

        return O / O.a;
    }

// Function 28
vec4 blur_vertical_upper_left(sampler2D channel, vec2 uv)
{
    float v = 1. / iResolution.y;
    vec4 sum = vec4(0.0);
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 4.0*v)) ) * 0.05;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 3.0*v)) ) * 0.09;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 2.0*v)) ) * 0.12;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 1.0*v)) ) * 0.15;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 0.0*v)) ) * 0.16;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 1.0*v)) ) * 0.15;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 2.0*v)) ) * 0.12;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 3.0*v)) ) * 0.09;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 4.0*v)) ) * 0.05;
    return sum/0.98; // normalize
}

// Function 29
vec3 blurV(vec3 iResolution, sampler2D src, vec2 uv) {
    float blurD = BLUR_D * iResolution.x/iResolution.y;
    return (
        0.006 * CBLUR(vec2(0.0, -3.0*blurD)) +
        0.061 * CBLUR(vec2(0.0, -2.0*blurD)) +
        0.242 * CBLUR(vec2(0.0, -1.0*blurD)) +
        0.383 * CBLUR(vec2(0.0,  0.0*blurD)) +
        0.242 * CBLUR(vec2(0.0,  1.0*blurD)) +
        0.061 * CBLUR(vec2(0.0,  2.0*blurD)) +
        0.006 * CBLUR(vec2(0.0,  3.0*blurD))
    );
}

// Function 30
vec3 direct_blur(sampler2D tex, vec2 uv, vec2 d, int k)
{
    vec2 s = 1.0 / vec2(textureSize(tex, 0));
    d *= s;
    vec2 b = -0.5 * d * float(k - 1);
    vec3 avg = vec3(0.0);
    for(int x = 0; x < k; x++)
        avg += texture(tex, uv + b + d * float(x) * 2.0).rgb;
    return avg / float(k);
}

// Function 31
vec4 gaussianBlur(sampler2D input_samp, vec2 center, vec2 sampleDist, float samplesPerDir, bool gammaCorrect){
	vec4 col_samples = vec4(0.);
    float sigma = samplesPerDir * 0.2;
    
    float cutoffWeight = gaussianWeight(
        sigma,
        vec2(
            ((samplesPerDir-1.)/2.),
            fract(mod(samplesPerDir - 1., 2.)/2. )
        ) 
    );
    float totalWeights = 0.;
    
    for(float rp_i_y = 0.; rp_i_y < samplesPerDir; rp_i_y++){
        for(float rp_i_x = 0.; rp_i_x < samplesPerDir; rp_i_x++){
    		vec2 relPos = vec2( (rp_i_x - ((samplesPerDir-1.)/2.)),
                                (rp_i_y - ((samplesPerDir-1.)/2.)));
            float weight = gaussianWeight(sigma, relPos);
            //only count the sample when the result will significantly affect the
            //end result; unnessecairy passes should be optimized away at compile time
            if(weight > cutoffWeight){
                totalWeights += weight;
                vec4 col_sample = texture(input_samp, center - relPos * sampleDist);
                if(!gammaCorrect)
                    col_samples += weight * col_sample;
                else
                    #ifdef FASTGAMMA
                    col_samples += weight * vec4(col_sample.rgb*col_sample.rgb, col_sample.a);
                    #else
                    col_samples += weight * vec4(pow(col_sample.rgb, vec3(2.2)), col_sample.a);
                    #endif
            }
        }
    }
    
    col_samples /= totalWeights;
    if(!gammaCorrect)
        return col_samples;
    else
        #ifdef FASTGAMMA
        return vec4(pow(col_samples.rgb, vec3(1./2.2)), col_samples.a);
    	#else
    	return vec4(sqrt(col_samples.rgb), col_samples.a);
    	#endif
}

// Function 32
vec4 BlurA(vec2 uv, int level, sampler2D bufA, sampler2D bufD)
{
    if(level <= 0)
    {
        return texture(bufA, fract(uv));
    }

    uv = upper_left(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(bufD, uv);
}

// Function 33
vec4 blur(vec2 uv, vec4 col) 
{
    vec2 dist = 2.0 / iResolution.xy;
    int s = 2; // samples
    for(int x = -s; x <= s; x++) {
    	for(int y = -s; y <= s; y++) {
			col += texture(iChannel2, uv + vec2(x,y) * dist);
        }
    }
    return col *= 0.030;
}

// Function 34
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

// Function 35
vec4 BlurA(vec2 uv, int level)
{
    if(level <= 0)
    {
        return texture(iChannel0, fract(uv));
    }

    uv = upper_left(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(iChannel3, uv)
;
}

// Function 36
vec4 BlurPass(vec2 fragCoords, vec2 resolution, float sampleDistance, sampler2D tex)
{     
    vec2 uv = fragCoords/resolution;
    float v = smoothstep(0.15, 0.5, length(uv - vec2(0.5)));
    vec4 t = vec4(0.0);
    float itter = 0.0;
    
    for(float i = -2.0; i <= 2.0; i++)
    {
        for(float j = -2.0; i <= 2.0; i++)
        {
			t += texture(tex, uv + (vec2(i, j) / resolution) * sampleDistance * v);
            itter += 1.0;
        } 
    }
    
    return t / itter;
}

// Function 37
vec3 fastSurfaceBlur( sampler2D inputColor, sampler2D inputEdge, vec2 uv, vec2 blurStep)
{
	// Normalized gauss kernel
    for (int j = 0; j <= kSize; ++j) {
        kernel[kSize+j] = kernel[kSize-j] = normpdf(float(j), sigma);
    }

	vec3 result = blurEdge*kernel[kSize]*texture(inputColor,uv).xyz;
	float Z = blurEdge*kernel[kSize];
		
	// Right direction
	float weight = blurEdge;
	for(int i = 1; i < kSize; i++){
		vec2 currentPos = uv + float(i)*blurStep;
		
		weight -= texture(inputEdge,currentPos).x;
		if (weight <= 0.0) break;
		
		float coef = weight*kernel[kSize+i];
		result += coef*texture(inputColor,currentPos).xyz;
		Z += coef;
	}
    
	// Left direction
	weight = blurEdge;
	for(int i = 1; i<kSize; i++){
		vec2 currentPos = uv - float(i)*blurStep;
		
		weight -= texture(inputEdge,currentPos).x;
		if (weight <= 0.0) break;
		
		float coef = weight*kernel[kSize-i];
		result += coef*texture(inputColor,currentPos).xyz;
		Z += coef;
	}
    return result / Z;
}

// Function 38
vec4 DirectionalBlur(in vec2 UV, in vec2 Direction, in float Intensity, in sampler2D Texture)
{
    vec4 Color = vec4(0.0);  
    float Noise = texture(iChannel1,UV*NoiseScale).x-0.485;
    
    if (UseNoise==false)
    for (int i=1; i<=Samples/2; i++)
    {
    Color += texture(Texture,UV+float(i)*Intensity/float(Samples/2)*Direction);
    Color += texture(Texture,UV-float(i)*Intensity/float(Samples/2)*Direction);
    }
	else      
    for (int i=1; i<=Samples/2; i++)
    {
    Color += texture(Texture,UV+float(i)*Intensity/float(Samples/2)*(Direction+NoiseStrength*Noise));
    Color += texture(Texture,UV-float(i)*Intensity/float(Samples/2)*(Direction+NoiseStrength*Noise));  
    }    
    return Color/float(Samples);    
}

// Function 39
vec4 blur_vertical_upper_left(sampler2D channel, vec2 uv)
{
    float v = 1. / iResolution.y;
    vec4 sum = vec4(0.0);
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 4.0*v)) ) * 0.0162162162;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 3.0*v)) ) * 0.0540540541;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 2.0*v)) ) * 0.1216216216;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 1.0*v)) ) * 0.1945945946;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 0.0*v)) ) * 0.2270270270;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 1.0*v)) ) * 0.1945945946;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 2.0*v)) ) * 0.1216216216;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 3.0*v)) ) * 0.0540540541;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 4.0*v)) ) * 0.0162162162;
    return sum;
}

// Function 40
blurout blur( sampler2D smpl, vec2 uv, vec2 p, float filtersiz_nm_ch0, vec2 ch0siz, int idx )
{
    float filtersiz = get_filter_size( idx );

    vec2 pq = (floor( p*ch0siz ) + vec2(0.5,0.5) ) / ch0siz;
    
    blurout ret;
    ret.dbgcol = vec4(0);
    vec4 bb_nm = vec4( pq - vec2(filtersiz_nm_ch0),
                       pq + vec2(filtersiz_nm_ch0));
    vec4 bb_px_q = vec4( floor( bb_nm.xy * ch0siz.xy ),
                          ceil( bb_nm.zw * ch0siz.xy ) );
    vec4 bb_nm_q = bb_px_q / ch0siz.xyxy;
    ivec2 bb_px_siz = ivec2( bb_px_q.zw - bb_px_q.xy );

    vec4 sumc = vec4(0.0);
    float sumw = 0.0;
    for ( int y=0; y<bb_px_siz.y; ++y )
    {
        for ( int x=0; x<bb_px_siz.x; ++x )
        {
            vec2 xy_f = (vec2(x,y)+vec2(0.5))/vec2(bb_px_siz);
            vec2 sp = bb_nm_q.xy + (bb_nm_q.zw-bb_nm_q.xy)*xy_f;

            float w = calc_weight( sp-p, vec2(filtersiz_nm_ch0), idx );
            
            #if defined( SIEMENS_PATTERN )
            sumc += w*srgb2lin( vec4(pattern(sp)) );
            #else
            sumc += w*srgb2lin(texture( iChannel0, sp, -10.0 ));
            #endif
            
            sumw += w;

            ret.dbgcol = mix( weight2col(w), ret.dbgcol, dbgp(sp, uv) );
        }
    }

    ret.outcol = sumc / sumw;
    
    return ret;
}

// Function 41
vec4 quickblur(vec2 pos)
{
   vec4 pixval = vec4(0.);
   float csum = 0.;
    
   const int nb = 2*blur_size+1;
   
   for (int y=0; y<nb; y++)
   {
       for (int x=0; x<nb; x++)
       { 
           vec2 ipos = pos + vec2(blur_width*float(x-blur_size)/iResolution.x, blur_width*float(y-blur_size)/iResolution.y);
           float g = 1.; //gauss(distance(ipos, pos), float(blur_size)*blur_fact);
           pixval+= g*texture(iChannel0, ipos);
           csum+= g;
       }
   }
   return pixval/csum;
}

// Function 42
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

// Function 43
vec3 blurH(sampler2D src, vec2 uv) {
    return (
        0.006 * CBLUR(vec2(-3.0*BLUR_D, 0.0)) +
        0.061 * CBLUR(vec2(-2.0*BLUR_D, 0.0)) +
        0.242 * CBLUR(vec2(-1.0*BLUR_D, 0.0)) +
        0.383 * CBLUR(vec2( 0.0*BLUR_D, 0.0)) +
        0.242 * CBLUR(vec2( 1.0*BLUR_D, 0.0)) +
        0.061 * CBLUR(vec2( 2.0*BLUR_D, 0.0)) +
        0.006 * CBLUR(vec2( 3.0*BLUR_D, 0.0))
    );
}

// Function 44
vec4 blur_vertical_left_column(vec2 uv, int depth)
{
    float v = pow(2., float(depth)) / iResolution.y;

    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;

    uv1 = fract(vec2(uv.x, uv.y - 4.0*v) * 2.);
    uv2 = fract(vec2(uv.x, uv.y - 3.0*v) * 2.);
    uv3 = fract(vec2(uv.x, uv.y - 2.0*v) * 2.);
    uv4 = fract(vec2(uv.x, uv.y - 1.0*v) * 2.);
    uv5 = fract(vec2(uv.x, uv.y + 0.0*v) * 2.);
    uv6 = fract(vec2(uv.x, uv.y + 1.0*v) * 2.);
    uv7 = fract(vec2(uv.x, uv.y + 2.0*v) * 2.);
    uv8 = fract(vec2(uv.x, uv.y + 3.0*v) * 2.);
    uv9 = fract(vec2(uv.x, uv.y + 4.0*v) * 2.);

    if(uv.y > 0.5)
    {
        uv1 = upper_left(uv1);
        uv2 = upper_left(uv2);
        uv3 = upper_left(uv3);
        uv4 = upper_left(uv4);
        uv5 = upper_left(uv5);
        uv6 = upper_left(uv6);
        uv7 = upper_left(uv7);
        uv8 = upper_left(uv8);
        uv9 = upper_left(uv9);
    }
    else{
        uv1 = lower_left(uv1);
        uv2 = lower_left(uv2);
        uv3 = lower_left(uv3);
        uv4 = lower_left(uv4);
        uv5 = lower_left(uv5);
        uv6 = lower_left(uv6);
        uv7 = lower_left(uv7);
        uv8 = lower_left(uv8);
        uv9 = lower_left(uv9);
    }

    for(int level = 0; level < 8; level++)
    {
        if(level > depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);

    sum += texture(iChannel2, uv1) * 0.05;
    sum += texture(iChannel2, uv2) * 0.09;
    sum += texture(iChannel2, uv3) * 0.12;
    sum += texture(iChannel2, uv4) * 0.15;
    sum += texture(iChannel2, uv5) * 0.16;
    sum += texture(iChannel2, uv6) * 0.15;
    sum += texture(iChannel2, uv7) * 0.12;
    sum += texture(iChannel2, uv8) * 0.09;
    sum += texture(iChannel2, uv9) * 0.05;

    return sum/0.98; // normalize
}

// Function 45
void set_blur(float b) {
    if (b == 0.0) {
        _stack.blur = vec2(0.0, 1.0);
    } else {
        _stack.blur = vec2(
            b,
            0.0);
    }
}

// Function 46
vec4 Blur(
    sampler2D sp,
    vec2 uv,
    vec2 dir,
    const int samples,
    float bright)
{
    if (samples < 1)
        return vec4(0.0);
    
    if (samples == 1)
        return texture(sp, uv);
    
    uv -= float(samples / 2) * dir;
    
    vec4 color = texture(sp, uv);
    vec4 maxColor = color;
    
    for (int i = 1; i < samples; ++i)
    {
        uv += dir;
        
        vec4 pixel = texture(sp, uv);
        color += pixel;
        maxColor = max(maxColor, pixel);
    }
    
    color /= float(samples);
    
    color = mix(color, maxColor, color * bright);
    
    return color;
}

// Function 47
vec4 BlurA(vec2 uv, int level)
{
    uv = wrap_flip(uv);
    
    if(level <= 0)
    {
        return texture(iChannel0, uv);
    }

    uv = upper_left(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(iChannel3, uv);
}

// Function 48
vec3 blur(sampler2D sp, vec2 uv, vec2 scale) {
    vec3 col = vec3(0.0);
    float accum = 0.0;
    float weight;
    vec2 offset;
    
    for (int x = -samples / 2; x < samples / 2; ++x) {
        for (int y = -samples / 2; y < samples / 2; ++y) {
            offset = vec2(x, y);
            weight = gaussian(offset);
            col += texture(sp, uv + scale * offset).rgb * weight;
            accum += weight;
        }
    }
    
    return col / accum;
}

// Function 49
vec4 blur(in sampler2D sampler, in vec2 fragCoord, in vec2 resolution)
{
    vec2 uv = fragCoord / resolution;
    float blurStrength = distance(uv, vec2(0.5));
    blurStrength = pow(blurStrength, BLUR_RANGE) * (resolution.x / 100.0) * BLUR_STRENGTH;
    vec4 sum = vec4(0.0);
    vec2 pixelSize = vec2(1.0) / resolution;
	for (float x = -1.0; x <= 1.0; x += 1.0)
    {
     	for (float y = -1.0; y <= 1.0; y += 1.0)
        {
            sum += texture(sampler, uv + vec2(x, y) * pixelSize * blurStrength);
        }
    }

    return sum / 9.0;
}

// Function 50
vec3 GaussianBlur(sampler2D image, vec2 centerPixelUV, vec2 uvStep)
{
    vec3 ret = texture(image, centerPixelUV).rgb * gaussianBlurWeights[0];
    for(int i = 1; i < guassianBlurSamples; ++i)
    {
        vec2 offset = uvStep * float(i);
        ret += texture(image, centerPixelUV + offset).rgb * gaussianBlurWeights[i];
        ret += texture(image, centerPixelUV - offset).rgb * gaussianBlurWeights[i];
    }
	return ret;
}

// Function 51
vec4 blur(vec2 fragCoord) {
	const int kSize =  (M_SIZE - 1) / 2;
    float kernel[M_SIZE];
    
    vec4 finalColour = vec4(0.);
    
    float sigma = 7.;
    float Z = 0.;
    
    for (int j = 0; j <= kSize; j++) {
    	kernel[kSize + j] = kernel[kSize - j] = normpdf(float(j), sigma);
    }
    
    for (int j = 0; j < M_SIZE; j++) {
    	Z += kernel[j];
    }
    
    for (int i = -kSize; i <= kSize; i++) {
        for (int j = -kSize; j <= kSize; j++) {
            vec4 tex = texture(iChannel0, (fragCoord + vec2(float(i), float(j)))
                        / iResolution.xy);
            tex.a = (tex.r + tex.g + tex.b) / 3.;
        	finalColour += kernel[kSize + j] * kernel[kSize + i] * tex;
        }
    }
    
    return finalColour / (Z * Z);
}

// Function 52
float rblur( in vec2 uv, in vec2 center, float falloffExp )
{
    // Translate our floating point space to the center of our blur.
    uv -= center;
    
    // Go ahead and precompute the inverse of the number of samples.
    // so we don't have any inner divisions.
    float invSamples = 1.0 / float(RBLUR_SAMPLES);
    
    // Place to accumulate the result.
    float result = 0.0;
    
    // Independent locations to store the results of each inner tap.
    // Why? So each tap doesn't need to write back before the next one
    // can start executing, preventing stalls. Works on x86 and piped
    // MIPS and I think it helps this out too (at least on my old thinkpad).
    float r0=0.0,r1=0.0,r2=0.0,r3=0.0;
    
    // We need to do each tap at a different index/position, so by storing
    // them in a vector we can make incrementation a single op instead of 4.
    vec4 indices = vec4(0,1,2,3);
    
	// Same thing with the scale.
    vec4 scale = vec4(0);
    
    // Go through and and sample the texture.
    for( int i = 0; i < RBLUR_SAMPLES; i+=4 )
    {
        scale = indices*invSamples;
        r0 = texture(iChannel0, uv*scale.x + center).r;
        r1 = texture(iChannel0, uv*scale.y + center).r;
        r2 = texture(iChannel0, uv*scale.z + center).r;
        r3 = texture(iChannel0, uv*scale.w + center).r;
        indices += 4.0;
        result += r0+r1+r2+r3;
    }
    return pow(result * invSamples,falloffExp);
}

// Function 53
vec4 blur(vec2 coord){
    
    vec2 start = vec2(0);
	vec2 uv = vec2(0);
    
    start = coord + vec2(-8, 0) * factor;
    
    for (int i = 0; i < 5; i++){
        uv = start + float(i) * vec2(1,1) * factor;
        addSample(uv/iResolution.xy);
    	}

    start = uv;
    for (int i = 1; i < 9; i++){
        uv = start + float(i) * vec2(1,0) * factor;
        addSample(uv/iResolution.xy);
    	}

    start = uv;
    for (int i = 1; i < 5; i++){
        uv = start + float(i) * vec2(1,-1) * factor;
        addSample(uv/iResolution.xy);
    	}

    start = uv;
    for (int i = 1; i < 5; i++){
        uv = start + float(i) * vec2(-1,-1) * factor;
        addSample(uv/iResolution.xy);
    	}

    start = uv;
    for (int i = 1; i < 9; i++){
        uv = start + float(i) * vec2(-1,0) * factor;
        addSample(uv/iResolution.xy);
    	}

    start = uv;
    for (int i = 1; i < 4; i++){
        uv = start + float(i) * vec2(-1,1) * factor;
        addSample(uv/iResolution.xy);
    	}
    
    return accum / accumW;
    
	}

// Function 54
void apply_motion_blur(inout vec4 fragColor, vec2 fragCoord, vec4 camera_pos)
{
#if !USE_MOTION_BLUR
    return;
#endif

    // not right after teleporting
    float teleport_time = camera_pos.w;
    if (teleport_time > 0. && abs(iTime - teleport_time) < 1e-4)
        return;
    
    vec3 camera_angles = load(ADDR_CAM_ANGLES).xyz;
    vec3 prev_camera_pos = load(ADDR_PREV_CAM_POS).xyz;
    vec3 prev_camera_angles = load(ADDR_PREV_CAM_ANGLES).xyz;
    mat3 view_matrix = rotation(camera_angles.xyz);
    mat3 prev_view_matrix = rotation(prev_camera_angles.xyz);

    vec4 ndc_scale_bias = get_viewport_transform(iFrame, iResolution.xy, g_downscale);
    ndc_scale_bias.xy /= iResolution.xy;
    vec2 actual_res = ceil(iResolution.xy / g_downscale);
    vec4 coord_bounds = vec4(vec2(.5), actual_res - .5);

    vec3 dir = view_matrix * unproject(fragCoord * ndc_scale_bias.xy + ndc_scale_bias.zw);
    vec3 surface_point = camera_pos.xyz + dir * VIEW_DISTANCE * fragColor.w;
    dir = surface_point - prev_camera_pos;
    dir = dir * prev_view_matrix;
    vec2 prev_coord = project(dir).xy;
    prev_coord = (prev_coord - ndc_scale_bias.zw) / ndc_scale_bias.xy;
    float motion = length(prev_coord - fragCoord);

    if (fragColor.w <= 0. || motion * g_downscale < 4.)
        return;
    
    // Simulating a virtual shutter to avoid excessive blurring at lower FPS
    const float MOTION_BLUR_SHUTTER = MOTION_BLUR_AMOUNT / float(MOTION_BLUR_FPS);
    float shutter_fraction = clamp(MOTION_BLUR_SHUTTER/iTimeDelta, 0., 1.);

    vec2 rcp_resolution = 1./iResolution.xy;
    vec4 uv_bounds = coord_bounds * rcp_resolution.xyxy;
    vec2 trail_start = fragCoord * rcp_resolution;
    vec2 trail_end = prev_coord * rcp_resolution;
    trail_end = mix(trail_start, trail_end, shutter_fraction * linear_step(4., 16., motion * g_downscale));

    float mip_level = log2(motion / (float(MOTION_BLUR_SAMPLES) + 1.)) - 1.;
    mip_level = clamp(mip_level, 0., 2.);

    const float INC = 1./float(MOTION_BLUR_SAMPLES);
    float trail_offset = BLUE_NOISE(fragCoord).x * INC - .5;
    float trail_weight = 1.;
    for (float f=0.; f<float(MOTION_BLUR_SAMPLES); ++f)
    {
        vec2 sample_uv = mix(trail_start, trail_end, trail_offset + f * INC);
        if (is_inside(sample_uv, uv_bounds) < 0.)
            continue;
        vec4 s = textureLod(iChannel2, sample_uv, mip_level);
        // Hack: to avoid weapon model ghosting we'll ignore samples landing in that area.
        // This introduces another artifact (sharper area behind the weapon model), but
        // this one is harder to notice in motion...
        float weight = step(0., s.w);
        fragColor.rgb += s.xyz * weight;
        trail_weight += weight;
    }
    
    fragColor.rgb /= trail_weight;
}

// Function 55
vec4 blur9(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3846153846) * direction;
  vec2 off2 = vec2(3.2307692308) * direction;
  color += texture(image, uv) * 0.2270270270;
  color += texture(image, uv + (off1 / resolution)) * 0.3162162162;
  color += texture(image, uv - (off1 / resolution)) * 0.3162162162;
  color += texture(image, uv + (off2 / resolution)) * 0.0702702703;
  color += texture(image, uv - (off2 / resolution)) * 0.0702702703;
  return color;
}

// Function 56
vec3 GaussianBlur( sampler2D tex, ivec2 uv, ivec2 axis, float stddev )
{
    vec3 result = vec3(0);
    int range = int( GaussianRange(.0002,stddev) );
    for ( int i=0; i < 1000; i++ )
    {
        int du = -range + i;
        if ( du > range ) break;
        result += texelFetch( tex, uv + axis*du, 0 ).rgb
            	* GaussianWeight( float(du), stddev );
    }
    return result;
}

// Function 57
vec4 KawaseBlur(vec2 pixelSize, int iteration, vec2 halfSize, vec2 uv)
{
    vec2 dUV = (pixelSize * float(iteration)) + halfSize;
    
    vec2 texcoord = vec2(0);
    
    texcoord.x = uv.x - dUV.x;
    texcoord.y = uv.y + dUV.y;
    
    vec4 color = texture(iChannel0, texcoord);

    texcoord.x = uv.x + dUV.x;
    texcoord.y = uv.y + dUV.y;
    
    color += texture(iChannel0, texcoord);

    texcoord.x = uv.x - dUV.x;
    texcoord.y = uv.y - dUV.y;
    
    color += texture(iChannel0, texcoord);
    
    texcoord.x = uv.x + dUV.x;
    texcoord.y = uv.y - dUV.y;
    
    color += texture(iChannel0, texcoord);

    color.rgb *= 0.25;
    
    return color;
}

// Function 58
vec4 blur(sampler2D sp, vec2 uv, vec2 scale) {
    vec4 col = vec4(0.0);
    float accum, weight, offset;
    
    for (int i = -BLUR_SAMPLES / 2; i < BLUR_SAMPLES / 2; ++i) {
        offset = float(i);
        weight = gaussian(offset, sqrt(float(BLUR_SAMPLES)));
        col += texture(sp, uv + scale * offset) * weight;
        accum += weight;
    }
    
    return col / accum;
}

// Function 59
vec3 backgroundBlurred(vec3 dir){
    return (BackgroundIndex == 1) 	? texture(iChannel0, dir.xzy, 8.).rgb*BackgroundIntensity
        							: texture(iChannel1, dir.xzy, 8.).rgb*BackgroundIntensity;
}

// Function 60
float calc_aa_blur(float w) {
    vec2 blur = _stack.blur;
    w -= blur.x;
    float wa = clamp(-w*AA*uniform_scale_for_aa(), 0.0, 1.0);
    float wb = clamp(-w / blur.x + blur.y, 0.0, 1.0);
	return wa * wb;
}

// Function 61
vec3 blur(vec2 tc, float offs)
{
	vec4 xoffs = offs * vec4(-2.0, -1.0, 1.0, 2.0) / iResolution.x;
	vec4 yoffs = offs * vec4(-2.0, -1.0, 1.0, 2.0) / iResolution.y;
	
	vec3 color = vec3(0.0, 0.0, 0.0);
	color += hsample(tc + vec2(xoffs.x, yoffs.x)) * 0.00366;
	color += hsample(tc + vec2(xoffs.y, yoffs.x)) * 0.01465;
	color += hsample(tc + vec2(    0.0, yoffs.x)) * 0.02564;
	color += hsample(tc + vec2(xoffs.z, yoffs.x)) * 0.01465;
	color += hsample(tc + vec2(xoffs.w, yoffs.x)) * 0.00366;
	
	color += hsample(tc + vec2(xoffs.x, yoffs.y)) * 0.01465;
	color += hsample(tc + vec2(xoffs.y, yoffs.y)) * 0.05861;
	color += hsample(tc + vec2(    0.0, yoffs.y)) * 0.09524;
	color += hsample(tc + vec2(xoffs.z, yoffs.y)) * 0.05861;
	color += hsample(tc + vec2(xoffs.w, yoffs.y)) * 0.01465;
	
	color += hsample(tc + vec2(xoffs.x, 0.0)) * 0.02564;
	color += hsample(tc + vec2(xoffs.y, 0.0)) * 0.09524;
	color += hsample(tc + vec2(    0.0, 0.0)) * 0.15018;
	color += hsample(tc + vec2(xoffs.z, 0.0)) * 0.09524;
	color += hsample(tc + vec2(xoffs.w, 0.0)) * 0.02564;
	
	color += hsample(tc + vec2(xoffs.x, yoffs.z)) * 0.01465;
	color += hsample(tc + vec2(xoffs.y, yoffs.z)) * 0.05861;
	color += hsample(tc + vec2(    0.0, yoffs.z)) * 0.09524;
	color += hsample(tc + vec2(xoffs.z, yoffs.z)) * 0.05861;
	color += hsample(tc + vec2(xoffs.w, yoffs.z)) * 0.01465;
	
	color += hsample(tc + vec2(xoffs.x, yoffs.w)) * 0.00366;
	color += hsample(tc + vec2(xoffs.y, yoffs.w)) * 0.01465;
	color += hsample(tc + vec2(    0.0, yoffs.w)) * 0.02564;
	color += hsample(tc + vec2(xoffs.z, yoffs.w)) * 0.01465;
	color += hsample(tc + vec2(xoffs.w, yoffs.w)) * 0.00366;

	return color;
}

// Function 62
vec4 BlurColor (in vec2 Coord, in sampler2D Tex, in float MipBias)
{
	vec2 TexelSize = MipBias/iChannelResolution[0].xy;
    //    o -= o - length(fwidth(texture(iChannel0,u/iResolution.xy)))*3.;
    vec4  Color = vec4(0.0);//texture(Tex, Coord, MipBias);
    Color += length(fwidth(texture(Tex, Coord + vec2(TexelSize.x,0.0), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(-TexelSize.x,0.0), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(0.0,TexelSize.y), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(0.0,-TexelSize.y), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(TexelSize.x,TexelSize.y), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(-TexelSize.x,TexelSize.y), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(TexelSize.x,-TexelSize.y), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(-TexelSize.x,-TexelSize.y), MipBias)))*3.;    

    return Color/8.;
}

// Function 63
vec4 BlurSpectrogram(vec2 uv, int level, sampler2D bufD)
{
    uv = upper_right(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(bufD, uv); // Buf D in Texture C
}

// Function 64
vec4 quickblur(vec2 pos)
{
   vec4 pixval = vec4(0.);
   float csum = 0.;
    
   const int nb = 2*blur_size+1;
   
   for (int y=0; y<nb; y++)
   {
       for (int x=0; x<nb; x++)
       { 
           vec2 ipos = pos + vec2(blur_width*float(x-blur_size)/iResolution.x, blur_width*float(y-blur_size)/iResolution.y);
           pixval+= texture(iChannel1, ipos);
       }
   }
   return pixval/pow(float(nb), 2.);
}

// Function 65
vec3 calcBlur(vec2 texcoord, vec2 pixelSize){
	const int steps = blurSteps;
    
    float totalWeight = 0.0;
    vec3 totalColor = vec3(0.0);
    
    float offsetSize = pixelSize.y * blurOSize;
    
    for (int i = -steps; i <= steps; ++i){
        float offset = float(i);
        float x = abs(offset / blurOSize * ramp);
		float weight = distribution(x);
        
        totalColor += texture(iChannel0, texcoord + vec2(0.0, offset * offsetSize)).rgb * weight;
        totalWeight += weight;
    }
    
    return decodeColor(totalColor / totalWeight);
}

// Function 66
vec4 blur5(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3333333333333333) * direction;
  color += texture(image, uv) * 0.29411764705882354;
  color += texture(image, uv + (off1 / resolution)) * 0.35294117647058826;
  color += texture(image, uv - (off1 / resolution)) * 0.35294117647058826;
  return color; 
}

// Function 67
vec4 gaussBlurApprox(vec2 uv, float scale) {
    const int numSteps = 15;
    // Strange but const declaration gets an error, 
    // but there is an official way to declare const arrays.
    float gaussCoeff[15]; // 1D gauss kernel, normalized
    gaussCoeff[0] = 0.053917;
    gaussCoeff[1] = 0.053551;
    gaussCoeff[2] = 0.052469;
    gaussCoeff[3] = 0.050713;
    gaussCoeff[4] = 0.048354;
    gaussCoeff[5] = 0.045481;
    gaussCoeff[6] = 0.042201;
    gaussCoeff[7] = 0.038628;
    gaussCoeff[8] = 0.034879;
    gaussCoeff[9] = 0.031068;
    gaussCoeff[10] = 0.027300;
    gaussCoeff[11] = 0.023664;
    gaussCoeff[12] = 0.020235;
    gaussCoeff[13] = 0.017070;
    gaussCoeff[14] = 0.014204;
   
    uv = ((uv * 2. - 1.) *scale) * .5 + .5; // central scaling
    
    vec4 acc = texture(iChannel0, uv) * gaussCoeff[0];
    vec2 stepI = 1./iResolution.xy;
    stepI *= scale;
    vec2 offsetU = vec2(0.0);
    vec2 offsetD = vec2(0.0);
    
    for (int j = 0; j < numSteps; j++) {
        offsetU.y += stepI.y;
        offsetU.x = 0.;
        for (int i = 0; i < numSteps; i++) {
            acc += pow(texture(iChannel0, uv + offsetU), vec4(2.2)) * gaussCoeff[1 + i] * gaussCoeff[1 + j];
            acc += pow(texture(iChannel0, uv - offsetU), vec4(2.2)) * gaussCoeff[1 + i] * gaussCoeff[1 + j];
            offsetU.x += stepI.x;
        }
   
        offsetD.y -= stepI.y;
        offsetD.x = 0.;
        for (int i = 0; i < numSteps; i++) {
            acc += pow(texture(iChannel0, uv + offsetD), vec4(2.2)) * gaussCoeff[1 + i] * gaussCoeff[1 + j];
            acc += pow(texture(iChannel0, uv - offsetD), vec4(2.2)) * gaussCoeff[1 + i] * gaussCoeff[1 + j];
            offsetD.x += stepI.x;
        }
    }
    // Gamma correction is added, as it's done by iq here: https://www.shadertoy.com/view/XtsSzH
    return pow(acc, 1. / vec4(2.2));
    
}

// Function 68
vec3 BlurredPixel (in vec2 uv)
{
    float c_sigmaX      = iMouse.z > 0.0 ? 5.0 * iMouse.x / iResolution.x : (sin(iTime*2.0)*0.5 + 0.5) * 5.0;
	float c_sigmaY      = iMouse.z > 0.0 ? 5.0 * iMouse.y / iResolution.y : c_sigmaX;
    
    float total = 0.0;
    vec3 ret = vec3(0);
        
    for (int iy = 0; iy < c_samplesY; ++iy)
    {
        float fy = Gaussian (c_sigmaY, float(iy) - float(c_halfSamplesY));
        float offsety = float(iy-c_halfSamplesY) * c_pixelSize;
        for (int ix = 0; ix < c_samplesX; ++ix)
        {
            float fx = Gaussian (c_sigmaX, float(ix) - float(c_halfSamplesX));
            float offsetx = float(ix-c_halfSamplesX) * c_pixelSize;
            total += fx * fy;            
            ret += texture(iChannel0, uv + vec2(offsetx, offsety)).rgb * fx*fy;
        }
    }
    return ret / total;
}

// Function 69
vec4 blur(vec2 coord){
    vec2 start = coord + vec2(0, 3) * factor;
    
    for (int i = 0; i < 8; i++){
        vec2 uv = start + float(i) * vec2(0,-1) * factor;
        addSample(uv/iResolution.xy);
    	}
    
    return accum/accumW;
	}

// Function 70
vec4 BlurB(vec2 uv, int level, sampler2D bufB, sampler2D bufD)
{
    uv = fract(uv);
    if(level <= 0)
    {
        return texture(bufB, uv);
    }

    uv = lower_left(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(bufD, uv);
}

// Function 71
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

// Function 72
vec3 blur(vec2 p)
{
    vec3 ite = vec3(0.0);
    for(int i = 0; i < 20; i ++)
    {
        float tc = 0.15;
        ite += pix2(p, globaltime * 3.0 + (hash2(p + float(i)) - 0.5).x * tc, 5.0);
    }
    ite /= 20.0;
    ite += exp(fract(globaltime * 0.25 * 6.0) * -40.0) * 2.0;
    return ite;
}

// Function 73
vec4 blur_horizontal_left_column(vec2 uv, int depth)
{
    float h = pow(2., float(depth)) / iResolution.x;    
    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;

    uv1 = fract(vec2(uv.x - 4.0 * h, uv.y) * 2.);
    uv2 = fract(vec2(uv.x - 3.0 * h, uv.y) * 2.);
    uv3 = fract(vec2(uv.x - 2.0 * h, uv.y) * 2.);
    uv4 = fract(vec2(uv.x - 1.0 * h, uv.y) * 2.);
    uv5 = fract(vec2(uv.x + 0.0 * h, uv.y) * 2.);
    uv6 = fract(vec2(uv.x + 1.0 * h, uv.y) * 2.);
    uv7 = fract(vec2(uv.x + 2.0 * h, uv.y) * 2.);
    uv8 = fract(vec2(uv.x + 3.0 * h, uv.y) * 2.);
    uv9 = fract(vec2(uv.x + 4.0 * h, uv.y) * 2.);

    if(uv.y > 0.5)
    {
        uv1 = upper_left(uv1);
        uv2 = upper_left(uv2);
        uv3 = upper_left(uv3);
        uv4 = upper_left(uv4);
        uv5 = upper_left(uv5);
        uv6 = upper_left(uv6);
        uv7 = upper_left(uv7);
        uv8 = upper_left(uv8);
        uv9 = upper_left(uv9);
    }
    else{
        uv1 = lower_left(uv1);
        uv2 = lower_left(uv2);
        uv3 = lower_left(uv3);
        uv4 = lower_left(uv4);
        uv5 = lower_left(uv5);
        uv6 = lower_left(uv6);
        uv7 = lower_left(uv7);
        uv8 = lower_left(uv8);
        uv9 = lower_left(uv9);
    }

    for(int level = 0; level < 8; level++)
    {
        if(level >= depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);

    sum += texture(iChannel3, uv1) * 0.05;
    sum += texture(iChannel3, uv2) * 0.09;
    sum += texture(iChannel3, uv3) * 0.12;
    sum += texture(iChannel3, uv4) * 0.15;
    sum += texture(iChannel3, uv5) * 0.16;
    sum += texture(iChannel3, uv6) * 0.15;
    sum += texture(iChannel3, uv7) * 0.12;
    sum += texture(iChannel3, uv8) * 0.09;
    sum += texture(iChannel3, uv9) * 0.05;

    return sum/0.98; // normalize
}

// Function 74
vec3 GaussianBlur(in vec2 co, in int dim, in float sigma, in float sep)
{
    vec3 c = vec3(0);
    float z = 0.;
    for (int i=-dim; i < dim; ++i)
    {
		for (int j=-dim; j < dim; ++j)
        {
			float g = GaussianG(float(i), float(j), sigma);
            vec2 p = (co + (vec2(i,j)+.5) * sep);
            vec3 col;
            
            if ( all(greaterThanEqual(ivec2(p), ivec2(0)))
                && all(lessThan(ivec2(p), ivec2(iResolution.xy))) )
            {
            	col = texture(iChannel1,  p / iResolution.xy).rgb;
            }
            else
            {
                col = vec3(1);
            }
            
            c += g * col;
            z += g;
		}
	}
    return c/z;
}

// Function 75
vec4 BlurSpectrogram(vec2 uv, int level)
{
    uv = upper_right(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(iChannel3, uv);
}

// Function 76
vec4 blur_vertical_left_column(vec2 uv, int depth)
{
    float v = pow(2., float(depth)) / iResolution.y;

    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;

    uv1 = fract(vec2(uv.x, uv.y - 4.0*v) * 2.);
    uv2 = fract(vec2(uv.x, uv.y - 3.0*v) * 2.);
    uv3 = fract(vec2(uv.x, uv.y - 2.0*v) * 2.);
    uv4 = fract(vec2(uv.x, uv.y - 1.0*v) * 2.);
    uv5 = fract(vec2(uv.x, uv.y + 0.0*v) * 2.);
    uv6 = fract(vec2(uv.x, uv.y + 1.0*v) * 2.);
    uv7 = fract(vec2(uv.x, uv.y + 2.0*v) * 2.);
    uv8 = fract(vec2(uv.x, uv.y + 3.0*v) * 2.);
    uv9 = fract(vec2(uv.x, uv.y + 4.0*v) * 2.);

    if(uv.x < 0.5)
    {
        if(uv.y > 0.5)
        {
            uv1 = upper_left(uv1);
            uv2 = upper_left(uv2);
            uv3 = upper_left(uv3);
            uv4 = upper_left(uv4);
            uv5 = upper_left(uv5);
            uv6 = upper_left(uv6);
            uv7 = upper_left(uv7);
            uv8 = upper_left(uv8);
            uv9 = upper_left(uv9);
        }
        else
        {
            uv1 = lower_left(uv1);
            uv2 = lower_left(uv2);
            uv3 = lower_left(uv3);
            uv4 = lower_left(uv4);
            uv5 = lower_left(uv5);
            uv6 = lower_left(uv6);
            uv7 = lower_left(uv7);
            uv8 = lower_left(uv8);
            uv9 = lower_left(uv9);
        }
    }
    else
    {
        vec2 uv_s = upper_right(uv*2.)*2.;
        uv1 = clamp(vec2(uv_s.x, uv_s.y - 4.0*v), 0., 1.);
        uv2 = clamp(vec2(uv_s.x, uv_s.y - 3.0*v), 0., 1.);
        uv3 = clamp(vec2(uv_s.x, uv_s.y - 2.0*v), 0., 1.);
        uv4 = clamp(vec2(uv_s.x, uv_s.y - 1.0*v), 0., 1.);
        uv5 = clamp(vec2(uv_s.x, uv_s.y + 0.0*v), 0., 1.);
        uv6 = clamp(vec2(uv_s.x, uv_s.y + 1.0*v), 0., 1.);
        uv7 = clamp(vec2(uv_s.x, uv_s.y + 2.0*v), 0., 1.);
        uv8 = clamp(vec2(uv_s.x, uv_s.y + 3.0*v), 0., 1.);
        uv9 = clamp(vec2(uv_s.x, uv_s.y + 4.0*v), 0., 1.);
        depth--;
        uv1 = upper_right(uv1);
        uv2 = upper_right(uv2);
        uv3 = upper_right(uv3);
        uv4 = upper_right(uv4);
        uv5 = upper_right(uv5);
        uv6 = upper_right(uv6);
        uv7 = upper_right(uv7);
        uv8 = upper_right(uv8);
        uv9 = upper_right(uv9);
    }
    for(int level = 0; level < 8; level++)
    {
        if(level > depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);
    if(uv.x > 0.5 && uv.y > 0.5)
    {
        //return vec4(0);
        sum += texture(iChannel3, uv1) * 0.0162162162;
        sum += texture(iChannel3, uv2) * 0.0540540541;
        sum += texture(iChannel3, uv3) * 0.1216216216;
        sum += texture(iChannel3, uv4) * 0.1945945946;
        sum += texture(iChannel3, uv5) * 0.2270270270;
        sum += texture(iChannel3, uv6) * 0.1945945946;
        sum += texture(iChannel3, uv7) * 0.1216216216;
        sum += texture(iChannel3, uv8) * 0.0540540541;
        sum += texture(iChannel3, uv9) * 0.0162162162;
    }
    else
    {
        sum += texture(iChannel2, uv1) * 0.0162162162;
        sum += texture(iChannel2, uv2) * 0.0540540541;
        sum += texture(iChannel2, uv3) * 0.1216216216;
        sum += texture(iChannel2, uv4) * 0.1945945946;
        sum += texture(iChannel2, uv5) * 0.2270270270;
        sum += texture(iChannel2, uv6) * 0.1945945946;
        sum += texture(iChannel2, uv7) * 0.1216216216;
        sum += texture(iChannel2, uv8) * 0.0540540541;
        sum += texture(iChannel2, uv9) * 0.0162162162;
    }
    return sum; // normalize
}

// Function 77
vec4 blur_box(sampler2D tex, vec2 texel, vec2 uv, vec2 rect)
{
    vec4 total = vec4(0);
    
    float dist = inversesqrt(SAMPLES);
    for(float i = -0.5; i<=0.5; i+=dist)
    for(float j = -0.5; j<=0.5; j+=dist)
    {
        vec2 coord = uv+vec2(i,j)*rect*texel;
        total += texture(tex,coord);
    }
    
    return total * dist * dist;
}

// Function 78
vec4 slopeBlur(vec2 U, float amp, float scale, sampler2D ch) {
    vec4 O = vec4(0);
    float n = amp/scale * Noise2(U*scale).x;
    vec2 dU = vec2(dFdx(n),dFdy(n))/length(fwidth(U*scale)); // may be replaced by finite difference or analytical derivative
  //U += .1*iTime;                                           // for demo
    
    for (float i=-.5; i<.5; i+= 1./float(NBsamples)) {       // blur sum 
        vec2 u = U + i*dU;
        O += texture(ch,u);
    }
    return O / float(NBsamples);
}

// Function 79
vec4 blur(vec2 uv)
{
    vec3 gen = vec3(1.0,1.0,1.0);
   	
    float textureSize = iResolution.x*iResolution.y;    
    float onePixel = 1.0/textureSize;
    
    //
    float tl = NearestTextureSample(uv + vec2(-1.0/iResolution.x,-1.0/iResolution.y)).r;
    float tm = NearestTextureSample(uv + vec2(0.0,-1.0/iResolution.y)).r;
    float tr = NearestTextureSample(uv + vec2(1.0/iResolution.x,-1.0/iResolution.y)).r;
    float ml = NearestTextureSample(uv + vec2(-1.0/iResolution.x,0.0)).r;
    float mm = NearestTextureSample(uv + vec2(0.0,0.0)).r;
    float mr = NearestTextureSample(uv + vec2(1.0/iResolution.x,0.0)).r;
    float bl = NearestTextureSample(uv + vec2(-1.0/iResolution.x,1.0/iResolution.y)).r;
    float bm = NearestTextureSample(uv + vec2(0.0,1.0/iResolution.y)).r;
    float br = NearestTextureSample(uv + vec2(1.0/iResolution.x,1.0/iResolution.y)).r;
	gen *= (tl + tm + tr + ml + mm + mr + bl + bm + br)/9.0;
    
    return vec4(gen,1.0);
}

// Function 80
vec4 BlurA(vec2 uv, int level)
{
    if(level <= 0)
    {
        return texture(iChannel0, fract(uv));
    }

    uv = upper_left(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(iChannel3, uv);
}

// Function 81
vec4 BlurSpectrogram(vec2 uv, int level, sampler2D bufD)
{
    uv = upper_right(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(bufD, uv);
}

// Function 82
vec3 blur(vec2 uv) {
    vec3 col = vec3(0.0,0.0,0.0);
    vec2 d = (vec2(0.5,0.5)-uv)/32.;
    float w = 1.0;
    vec2 s = uv;
    for( int i=0; i<32; i++ )
    {
        vec3 res = surface(vec2(s.x,s.y));
        col += w*smoothstep( 0.0, 1.0, res );
        w *= .985;
        s += d;
    }
    col = col * 4.5 / 32.;
	return blurIntensity*vec3( 0.2*col + 0.8*surface(uv));
}

// Function 83
float getBlurSize(float depth, float focusPoint, float focusScale)
{
	float coc = clamp((1.0 / focusPoint - 1.0 / depth)*focusScale, -1.0, 1.0);
    return abs(coc) * MAX_BLUR_SIZE;
}

// Function 84
vec3 circle_blur(sampler2D sp, vec2 uv, vec2 scale) {
    vec2 ps = (1.0 / iResolution.xy) * scale;
    vec3 col = vec3(0.0);
    float accum = 0.0;
    
    for (int a = 0; a < 360; a += 360 / ANGLE_SAMPLES) {
        for (int o = 0; o < OFFSET_SAMPLES; ++o) {
			col += texture(sp, uv + ps * rot2D(float(o), float(a))).rgb * float(o * o);
            accum += float(o * o);
        }
    }
    
    return col / accum;
}

// Function 85
vec4 blur_vertical_lower_left(sampler2D channel, vec2 uv)
{
    float v = 1. / iResolution.y;
    vec4 sum = vec4(0.0);
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 4.0*v)) ) * 0.0162162162;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 3.0*v)) ) * 0.0540540541;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 2.0*v)) ) * 0.1216216216;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 1.0*v)) ) * 0.1945945946;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 0.0*v)) ) * 0.2270270270;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 1.0*v)) ) * 0.1945945946;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 2.0*v)) ) * 0.1216216216;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 3.0*v)) ) * 0.0540540541;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 4.0*v)) ) * 0.0162162162;
    return sum;
}

// Function 86
vec3 blur(sampler2D tex, vec2 uv, vec2 k, float ss)
{
    vec2 s = 1.0 / vec2(textureSize(tex, 0));
    vec3 avg = vec3(0.0);
    vec2 hk = k * 0.5;
    for(float x = 0.0 - hk.x; x < hk.x; x += 1.0)
    {
        for(float y = 0.0 - hk.y; y < hk.y; y += 1.0)
            avg += texture(tex, uv + s * vec2(x,y) * ss).rgb;
    }
    return avg / (k.x * k.y);
}

// Function 87
vec4 RotateBlur(vec2 fragCoord, float GaussianSize) {
    
    const float total = GaussianDepth * GaussianRingSamples;
    
    vec2 uvCoord = fragCoord/iResolution.xy;
    vec4 avgs = texture(iChannel1, uvCoord);
  
    // start larger, otherwise we tend to miss an angle
    //	(it looked like a small gap at a big enough bokeh)
    float angle = TAU/GaussianRingSamples;
    
    vec2 radStep = GaussianSize/iResolution.xy;
    vec2 rad = radStep;
    
    vec2 uvOffset = vec2(0.);
    
    
    for(float i = 0.; i < total; ++i) {
        
        uvOffset = vec2(cos(angle), sin(angle)) * rad;
      
        
        avgs += texture(iChannel1, uvCoord + uvOffset);
        
        
        // we wrap to zero if we're bigger than 2PI
        angle += (-TAU * float(angle > TAU));
        
        // otherwise we add
        angle += (TAU/GaussianRingSamples);
        
        // we increment if we went full circle
        rad += float(angle > TAU) * (radStep);
    }
    
    
    // tiny adjust seems to fix it, weird 
    // needs adjust based on effect amount
    return avgs / total - GammaAdjust;
    
}

// Function 88
vec3 blur(sampler2D sp, vec2 uv, vec2 scale) {
  vec3 col = vec3(0.0);
  float accum = 0.0;
  float weight;
  vec2 offset;

  for (int x = -samples / 2; x < samples / 2; ++x) {
    for (int y = -samples / 2; y < samples / 2; ++y) {
      offset = vec2(x, y);
      weight = gaussian(offset);
      col += texture(sp, uv + scale * offset).rgb * weight;
      accum += weight;
    }
  }

  return col / accum;
}

// Function 89
vec4 BlurColor (in vec2 Coord, in sampler2D Tex, in float MipBias)
{
	vec2 TexelSize = MipBias/iChannelResolution[0].xy;
    
    vec4  Color = texture(Tex, Coord, MipBias);
    Color += texture(Tex, Coord + vec2(TexelSize.x,0.0), MipBias);    	
    Color += texture(Tex, Coord + vec2(-TexelSize.x,0.0), MipBias);    	
    Color += texture(Tex, Coord + vec2(0.0,TexelSize.y), MipBias);    	
    Color += texture(Tex, Coord + vec2(0.0,-TexelSize.y), MipBias);    	
    Color += texture(Tex, Coord + vec2(TexelSize.x,TexelSize.y), MipBias);    	
    Color += texture(Tex, Coord + vec2(-TexelSize.x,TexelSize.y), MipBias);    	
    Color += texture(Tex, Coord + vec2(TexelSize.x,-TexelSize.y), MipBias);    	
    Color += texture(Tex, Coord + vec2(-TexelSize.x,-TexelSize.y), MipBias);    

    return Color/9.0;
}

// Function 90
vec3 backgroundBlurred(vec3 dir){
    vec3 col;
    if(BackgroundIndex == 0)
    	col = texture(iChannel0, dir.xzy, 7.0).rgb;
    else 
        col = texture(iChannel1, dir.xzy, 7.0).rgb;
    return (col*col + col) * BackgroundIntensity;
}

// Function 91
vec4 BlurB(vec2 uv, int level)
{
    if(level <= 0)
    {
        return texture(iChannel1, fract(uv));
    }

    uv = lower_left(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(iChannel3, uv);
}

// Function 92
float getBlurSize(float depth, float focusPoint, float focusScale) {
    float coc = clamp((1.0 / focusPoint - 1.0 / depth)*focusScale, -1.0, 1.0);
    return abs(coc) * MAX_BLUR_SIZE;
}

// Function 93
vec4 BlurA(vec2 uv, int level)
{
    return BlurA(uv, level, iChannel0, iChannel3);
}

// Function 94
vec4 blur_horizontal_left_column(vec2 uv, int depth)
{
    float h = pow(2., float(depth)) / iResolution.x;    
    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;

    uv1 = fract(vec2(uv.x - 4.0 * h, uv.y) * 2.);
    uv2 = fract(vec2(uv.x - 3.0 * h, uv.y) * 2.);
    uv3 = fract(vec2(uv.x - 2.0 * h, uv.y) * 2.);
    uv4 = fract(vec2(uv.x - 1.0 * h, uv.y) * 2.);
    uv5 = fract(vec2(uv.x + 0.0 * h, uv.y) * 2.);
    uv6 = fract(vec2(uv.x + 1.0 * h, uv.y) * 2.);
    uv7 = fract(vec2(uv.x + 2.0 * h, uv.y) * 2.);
    uv8 = fract(vec2(uv.x + 3.0 * h, uv.y) * 2.);
    uv9 = fract(vec2(uv.x + 4.0 * h, uv.y) * 2.);

    if(uv.y > 0.5)
    {
        uv1 = upper_left(uv1);
        uv2 = upper_left(uv2);
        uv3 = upper_left(uv3);
        uv4 = upper_left(uv4);
        uv5 = upper_left(uv5);
        uv6 = upper_left(uv6);
        uv7 = upper_left(uv7);
        uv8 = upper_left(uv8);
        uv9 = upper_left(uv9);
    }
    else{
        uv1 = lower_left(uv1);
        uv2 = lower_left(uv2);
        uv3 = lower_left(uv3);
        uv4 = lower_left(uv4);
        uv5 = lower_left(uv5);
        uv6 = lower_left(uv6);
        uv7 = lower_left(uv7);
        uv8 = lower_left(uv8);
        uv9 = lower_left(uv9);
    }

    for(int level = 0; level < 8; level++)
    {
        if(level >= depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);

    sum += texture(iChannel3, uv1) * 0.0162162162;
    sum += texture(iChannel3, uv2) * 0.0540540541;
    sum += texture(iChannel3, uv3) * 0.1216216216;
    sum += texture(iChannel3, uv4) * 0.1945945946;
    sum += texture(iChannel3, uv5) * 0.2270270270;
    sum += texture(iChannel3, uv6) * 0.1945945946;
    sum += texture(iChannel3, uv7) * 0.1216216216;
    sum += texture(iChannel3, uv8) * 0.0540540541;
    sum += texture(iChannel3, uv9) * 0.0162162162;
    
    return sum;
}

// Function 95
vec3 radial_blur(sampler2D tex, vec2 uv, int sc, float ss, float p, vec2 c)
{
    int hsc = sc / 2;
    vec3 avg = vec3(0.0);
    for(int x = 0; x < sc; x++)
    {
        float v = float(x - hsc) * p;
        avg += texture(tex, uv + c * v).rgb;
    }
    avg *= 1.0 / float(sc + 1);
    float dist = distance(c, uv);
    return mix(texture(tex, uv).rgb, avg, clamp(dist * ss, 0.0, 1.0));
}

// Function 96
vec4 blur_vertical_lower_left(sampler2D channel, vec2 uv)
{
    float v = 1. / iResolution.y;
    vec4 sum = vec4(0.0);
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 4.0*v)) ) * 0.05;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 3.0*v)) ) * 0.09;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 2.0*v)) ) * 0.12;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 1.0*v)) ) * 0.15;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 0.0*v)) ) * 0.16;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 1.0*v)) ) * 0.15;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 2.0*v)) ) * 0.12;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 3.0*v)) ) * 0.09;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 4.0*v)) ) * 0.05;
    return sum/0.98; // normalize
}

// Function 97
vec4 BlurH (sampler2D source, vec2 size, vec2 uv, float radius) {

	if (radius >= 1.0)
	{
		vec4 A = vec4(0.0); 
		vec4 C = vec4(0.0); 

		float width = 1.0 / size.x;

		float divisor = 0.0; 
        float weight = 0.0;
        
        float radiusMultiplier = 1.0 / radius;
        
        // Hardcoded for radius 20 (normally we input the radius
        // in there), needs to be literal here
        
		for (float x = -20.0; x <= 20.0; x++)
		{
			A = texture(source, uv + vec2(x * width, 0.0));
            
            	weight = SCurve(1.0 - (abs(x) * radiusMultiplier)); 
            
            	C += A * weight; 
            
			divisor += weight; 
		}

		return vec4(C.r / divisor, C.g / divisor, C.b / divisor, 1.0);
	}

	return texture(source, uv);
}

// Function 98
vec4 BlurA(vec2 uv, int level, sampler2D bufA, sampler2D bufD)
{
    uv = fract(uv);
    if(level <= 0)
    {
        return texture(bufA, uv);
    }

    uv = upper_left(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(bufD, uv);
}

// Function 99
vec4 blur ( sampler2D C, vec2 u ) {
	vec2 pxl = (1./iResolution.xy);
    vec4 asp=vec4(0);
    for (float x = -RESOLUTION; x < RESOLUTION; x++) {
        for (float y = -RESOLUTION; y < RESOLUTION; y++) {
        	vec2 uv = u+pxl*vec2(x,y);
            asp+=texture(C, uv);
    	}
    }
    float q = (RESOLUTION*2.);
    q*=q;
    asp/=q;
    return asp;
}

// Function 100
vec4 blur(vec2 uv){

	for (float j = -7.0 + mas; j <= 7.0 - mas; j += 1.0)
		for (float i = -7.0 + mas; i <=7.0 - mas; i += 1.0){
			add(uv, i, j);
			}
    
    for (float i = -5.0 + mas; i <=5.0 - mas; i+=1.0){
        add(uv, i, -8.0 + mas);
        add(uv, i, 8.0 - mas);
    	}
    for (float j = -5.0 + mas; j <=5.0 - mas; j+=1.0){
        add(uv, -8.0 + mas, j);
        add(uv, 8.0 - mas, j);
    	}
    
    for (float i = -3.0 +mas; i <=3.0 -  mas; i+=1.0){
        add(uv, i, -9.0 + mas);
        add(uv, i, 9.0 - mas);
    	}
    for (float j = -3.0 + mas; j <=3.0 - mas; j+=1.0){
        add(uv, -9.0 + mas, j);
        add(uv, 9.0 - mas, j);
    	}

	return accumCol/accumW;
	}

// Function 101
vec3 blurSample(in vec2 uv, in vec2 xoff, in vec2 yoff)
{
    vec3 v11 = texture(iChannel0, uv + xoff).rgb;
    vec3 v12 = texture(iChannel0, uv + yoff).rgb;
    vec3 v21 = texture(iChannel0, uv - xoff).rgb;
    vec3 v22 = texture(iChannel0, uv - yoff).rgb;
    return (v11 + v12 + v21 + v22 + 2.0 * texture(iChannel0, uv).rgb) * 0.166667;
}

// Function 102
vec4 GaussianBlur63(const in sampler2D pTexSource, const in vec2 pCenterUV, const in float pLOD, const in vec2 pPixelOffset){
  return ((texture(pTexSource, pCenterUV + (pPixelOffset * 0.6655479694036902)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 0.6655479694036902))) * 0.0599136953241179)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 2.493712921290222)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 2.493712921290222))) * 0.07758097185631951)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 4.488684594173841)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 4.488684594173841))) * 0.07231852189840038)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 6.483658555984297)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 6.483658555984297))) * 0.06476077451799045)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 8.478635821570533)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 8.478635821570533))) * 0.055711220167732625)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 10.473617403111605)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 10.473617403111605))) * 0.04604064682545439)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 12.468604309302684)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 12.468604309302684))) * 0.03655175467912483)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 14.463597544547007)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 14.463597544547007))) * 0.02787680909208401)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 16.458598108155186)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 16.458598108155186))) * 0.02042423735066097)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 18.453606993553194)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 18.453606993553194))) * 0.014375287044873498)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 20.44862518750038)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 20.44862518750038))) * 0.009719748190692979)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 22.443653669318742)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 22.443653669318742))) * 0.006313369798084606)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 24.43869341013483)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 24.43869341013483))) * 0.0039394453026797525)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 26.43374537213539)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 26.43374537213539))) * 0.0023614374738993053)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 28.428810507838012)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 28.428810507838012))) * 0.001359831963351014)+
         ((texture(pTexSource, pCenterUV + (pPixelOffset * 30.423889759377865)) +
           texture(pTexSource, pCenterUV - (pPixelOffset * 30.423889759377865))) * 0.0007522485145337108);
}

// Function 103
void gaussianBlur( out vec4 fragColor, in vec2 fragCoord, in highp float sigma )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    
	int kernel_window_size = GAUSSIANRADIUS*2+1;
    int samples = kernel_window_size*kernel_window_size;
    
    highp vec4 color = vec4(0);
    
    // precompute this, it is used a few times in the weighting formula below.
    highp float sigma_square = sigma*sigma;
    
    // here we need to keep track of the weighted sum, because if you pick a
    // radius that cuts off most of the gaussian function, then the weights
    // will not add up to 1.0, and will dim the image (i.e it won't be a weighted average,
    // because a weighted average assumes the weights all add up to 1). Therefore,
    // we keep the weights' sum, and divide it out at the end.
    highp float wsum = 0.0;
    for (int ry = -GAUSSIANRADIUS; ry <= GAUSSIANRADIUS; ++ry)
    for (int rx = -GAUSSIANRADIUS; rx <= GAUSSIANRADIUS; ++rx)
    {
        
        // 2d gaussian function, see https://en.wikipedia.org/wiki/Gaussian_blur#Mathematics
        // basically: this is a formula that produces a weight based on distance to the center pixel.
        highp float w = (1.0 / (2.0*PI*sigma_square))* exp(-(float(rx*rx) + float(ry*ry)) / (2.0*sigma_square));
        //highp float w = (1.0 / (2.0*PI*sigma_square)) * exp(-(dot(vec2(0,0), vec2(rx,ry)) / (2.0*sigma_square));
        wsum += w;
        vec4 pixel_color = texture(iChannel0, uv+vec2(rx,ry)/iResolution.xy);
        // degamma
        pixel_color = pow(pixel_color, vec4(2.2));
        
    	color += pixel_color*w;
    }
    
    fragColor = pow(color/wsum, 1.0/vec4(2.2));
}

// Function 104
vec3 naiveBoxBlur(
    in sampler2D tex,
    in vec2 uv,
    in vec2 resolution,
    in float size,
    in int samples
) {
    float f_samples = float( samples );
    vec2 px = 1. / resolution;
    float increment = size / ( f_samples - 1. );
    float halfSize = size * .5;
    
    vec3 color = vec3( 0. );
    float w = 0.;
    for ( float i = -halfSize; i <= halfSize; i += increment ) {
        for ( float j = -halfSize; j <= halfSize; j += increment ) {
            w += 1.;
            vec2 st = uv + vec2( i, j ) * px;
            color += texture( tex, st ).rgb;
        }
    }
    
    return color / w;
}

// Function 105
vec4 blur_angular(sampler2D tex, vec2 texel, vec2 uv, float angle)
{
    vec4 total = vec4(0);
    vec2 coord = uv-0.5;
    
    float dist = 1.0/SAMPLES;
    vec2 dir = vec2(cos(angle*dist),sin(angle*dist));
    mat2 rot = mat2(dir.xy,-dir.y,dir.x);
    for(float i = 0.0; i<=1.0; i+=dist)
    {
        total += texture(tex,coord+0.5);
        coord *= rot;
    }
    
    return total * dist;
}

// Function 106
vec4 blur13(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.411764705882353) * direction;
  vec2 off2 = vec2(3.2941176470588234) * direction;
  vec2 off3 = vec2(5.176470588235294) * direction;
  color += texture(image, uv) * 0.1964825501511404;
  color += texture(image, uv + (off1 / resolution)) * 0.2969069646728344;
  color += texture(image, uv - (off1 / resolution)) * 0.2969069646728344;
  color += texture(image, uv + (off2 / resolution)) * 0.09447039785044732;
  color += texture(image, uv - (off2 / resolution)) * 0.09447039785044732;
  color += texture(image, uv + (off3 / resolution)) * 0.010381362401148057;
  color += texture(image, uv - (off3 / resolution)) * 0.010381362401148057;
  return color;
}

// Function 107
vec4 blur_horizontal(sampler2D channel, vec2 uv, float scale)
{
    float h = scale / iResolution.x;
    vec4 sum = vec4(0.0);

    sum += texture(channel, fract(vec2(uv.x - 4.0*h, uv.y)) ) * 0.05;
    sum += texture(channel, fract(vec2(uv.x - 3.0*h, uv.y)) ) * 0.09;
    sum += texture(channel, fract(vec2(uv.x - 2.0*h, uv.y)) ) * 0.12;
    sum += texture(channel, fract(vec2(uv.x - 1.0*h, uv.y)) ) * 0.15;
    sum += texture(channel, fract(vec2(uv.x + 0.0*h, uv.y)) ) * 0.16;
    sum += texture(channel, fract(vec2(uv.x + 1.0*h, uv.y)) ) * 0.15;
    sum += texture(channel, fract(vec2(uv.x + 2.0*h, uv.y)) ) * 0.12;
    sum += texture(channel, fract(vec2(uv.x + 3.0*h, uv.y)) ) * 0.09;
    sum += texture(channel, fract(vec2(uv.x + 4.0*h, uv.y)) ) * 0.05;

    return sum/0.98; // normalize
}

// Function 108
vec4 blurred( vec2 uv ,float scale){
    
    vec4 sum;
    vec2 center = vec2(iMouse.x/iResolution.x,iMouse.y/iResolution.y);
    float ray = mod(acos(dot(normalize(uv-center),vec2(0.,1.)))/PI/2.,1.);
    if(sign(uv-center).x==-1.) ray=1.-ray;
    scale=scale*(0.003*length(uv-center));
    float dist=length(uv-center)/1.5*2.+scale*rand(uv*16.66+vec2(iTime*137.));
    for (int i = 0; i < SAMPLES; i++) {
    	sum+=vec4(texture(iChannel3,vec2(ray,-float(i)*scale+dist*0.5)));

    };
    
    
	return sum/float(SAMPLES);
}

// Function 109
vec4 blur_radial(sampler2D tex, vec2 texel, vec2 uv, float radius)
{
    vec4 total = vec4(0);
    
    float dist = 1.0/SAMPLES;
    float rad = radius * length(texel);
    for(float i = 0.0; i<=1.0; i+=dist)
    {
        vec2 coord = (uv-0.5) / (1.0+rad*i)+0.5;
        total += texture(tex,coord);
    }
    
    return total * dist;
}

// Function 110
vec4 BlurB(vec2 uv, int level, sampler2D bufB, sampler2D bufD)
{
    if(level <= 0)
    {
        return texture(bufB, fract(uv));
    }

    uv = lower_left(uv);
    for(int depth = 1; depth < 8; depth++)
    {
        if(depth >= level)
        {
            break;
        }
        uv = lower_right(uv);
    }

    return texture(bufD, uv);
}

// Function 111
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

// Function 112
vec4 blur (vec2 uv)
{
    vec4 res;
	for (int x = - 6; x < 6; x ++)
    {
    	for (int y = -6 ; y < 6; y ++)
        {
            res += blurWeights[x+6]*blurWeights[y+6] * texture( iChannel0, ( uv * iResolution.xy + vec2 (x,y) ) / iResolution.xy);
        }
    }
    return res;
}

// Function 113
void gaussianBlur( out vec4 fragColor, in vec2 fragCoord, in highp float sigma )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    
	int kernel_window_size = GAUSSIANRADIUS*2+1;
    int samples = kernel_window_size*kernel_window_size;
    
    highp vec4 color = vec4(0);
    
    // precompute this, it is used a few times in the weighting formula below.
    highp float sigma_square = sigma*sigma;
    
    // here we need to keep track of the weighted sum, because if you pick a
    // radius that cuts off most of the gaussian function, then the weights
    // will not add up to 1.0, and will dim the image (i.e it won't be a weighted average,
    // because a weighted average assumes the weights all add up to 1). Therefore,
    // we keep the weights' sum, and divide it out at the end.
    highp float wsum = 0.0;
    for (int ry = -GAUSSIANRADIUS; ry <= GAUSSIANRADIUS; ++ry)
    for (int rx = -GAUSSIANRADIUS; rx <= GAUSSIANRADIUS; ++rx)
    {
        
        // 2d gaussian function, see https://en.wikipedia.org/wiki/Gaussian_blur#Mathematics
        // basically: this is a formula that produces a weight based on distance to the center pixel.
        highp float w = (1.0 / (2.0*PI*sigma_square))* exp(-(float(rx*rx) + float(ry*ry)) / (2.0*sigma_square));
        //highp float w = (1.0 / (2.0*PI*sigma_square)) * exp(-(dot(vec2(0,0), vec2(rx,ry)) / (2.0*sigma_square));
        wsum += w;
    	color += texture(iChannel0, uv+vec2(rx,ry)/iResolution.xy)*w;
    }
    
    fragColor = color/wsum;
}

