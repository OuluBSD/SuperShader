// Reusable Amplitude Processing Audio Functions
// Automatically extracted from audio visualization-related shaders

// Function 1
float getAvgVolume(float v, int steps)
{
    float sum = 0.0;
    float x = 0.0;
	for (int i = 0; i < steps; i++)
    {
        x = fract(v + float(i) * FREQ_STEP);
        								//pow for non linear spectrum, 2nd pow - soften bass
        sum += texture(iChannel0, vec2(pow(x, 2.0), 0.0)).r * pow(x, 0.08); 
    }
    
    return sum /= float(steps);
}

// Function 2
Ray traceVolumes(Ray ray)
{
    return marchVolume(ray, fog);
}

// Function 3
float GetCornerIntensity(vec2 fragCoord, float checkDist, vec2 corner, float beginSample, float endSample)
{
    float distanceN = 1.0 - (clamp(distance(fragCoord, corner), 0.0, checkDist) / checkDist);
    float intensity = GetIntensityInRange(beginSample, endSample, 0.05);
    float color = distanceN * intensity;
    return color;
}

// Function 4
float intensity(float d) { return smoothstep(15., 0., d); }

// Function 5
void mainVolume( out vec4 voxColor, in vec3 voxCoord)
{
    vec3 uvw = voxCoord / iVResolution;

    vec3 color = vec3(1,0,0);
    
    vec3 p0 = sin(vec3(1.3,0.9,2.1) * iTime + 7.0)*.5+.5;
    vec3 p1 = sin(vec3(0.5,1.6,0.8) * iTime + 4.0)*.5+.5;
    vec3 p2 = sin(vec3(0.9,1.2,1.5) * iTime + 2.0)*.5+.5;
    
    float s0 = cos(length(p0-uvw)*28.0);
    float s1 = cos(length(p1-uvw)*19.0);
    float s2 = cos(length(p2-uvw)*22.0);
    
    float dens = (s0+s1+s2)/3.0;
    
    color = vec3(s0,s1,s2);
    
    dens *= 0.5;
    
    voxColor = vec4(color, dens);
}

// Function 6
void volumetricFog( inout vec3 color, vec3 rayOrigin, vec3 rayDir, float sceneT )
{ // From "[SH16B] Speed Drive 80" by knarkowicz. https://shadertoy.com/view/4ldGz4
    float gFogDensity		= 0.1;
    rayOrigin.z += 2.0 * g_S.playerPos.y;
    
    sceneT = sceneT <= 0.0 ? 100.0 : sceneT;
    
    vec3 seed = vec3( 0.06711056, 0.00583715, 52.9829189 );
    float dither = fract( seed.z * fract( dot( gl_FragCoord.xy, seed.xy ) ) );
    
    float fogAlpha = 0.0;
    for ( int i = ZERO; i < 32; ++i )
    {
        float t = ( float( i ) + 0.5 + dither ) * 5.0;
        if ( t <= sceneT )
        {
        	vec3 p = rayOrigin + t * rayDir;
            float s = densityNoise( p );
            fogAlpha += gFogDensity * t * exp( -gFogDensity * t ) * s;
        }
    }
    fogAlpha = 1.0 - saturate( fogAlpha );
    vec3 fogColor = FOG_COLOR + vec3( 1.0 );
    color = mix( fogColor, color, fogAlpha );
    // color = vec3(0.01)*sceneT;
}

// Function 7
vec4 MarchVolume(vec3 u,vec3 t,vec3 s){
;t=normalize(t);//save>sorry
;vec4 c=vec4(0)//return vaslue
;const vec2 stepn=vec2(40,20)/iterMarchVolume;//2 loop params
;float a=1.,b=110.//diminishing accumulator//absorbtion
;for(float i=.0;i<iterMarchVolume.x;i++)
{;float d=gdVolume(u)
 ;if(d>0.)
 {;d=d/iterMarchVolume.x
  ;a*=1.-d*b
  ;if(a<=.01)break
  ;float Tl=1.
  ;for(float j=.0;j<iterMarchVolume.y; j++)
  {;float l=gdVolume(u+normalize(s)*float(j)*stepn.y)
   //todo, also calculate occlusion of a non-clud distance field.
   ;if(l>0.)
    Tl*=1.-l*b/iterMarchVolume.x
   ;if(Tl<=.01)break;}
  ;c+=clDiff*cloudDark*d*a//light.diffuse
  ;c+=clAmbi*cloudBright*d*a*Tl;//light.ambbience
 ;}
 ;u+=t*stepn.x;}    
;return max(c,(cDiff*cDiff));//;return c
;}

// Function 8
vec3 MarchVolume(vec3 orig, vec3 dir)
{
    //Ray march to find the cube surface.
    float t = 0.0;
    vec3 pos = orig;
    for(int i = 0;i < MAX_MARCH_STEPS;i++)
    {
        pos = orig + dir * t;
        float dist = 100.0;
        
        dist = min(dist, 8.0-length(pos));
        dist = min(dist, max(max(abs(pos.x),abs(pos.y)),abs(pos.z))-1.0);//length(pos)-1.0);
        
        t += dist;
        
        if(dist < MIN_MARCH_DIST){break;}
    }
    
    //Step though the volume and add up the opacity.
    vec4 col = vec4(0.0);
    for(int i = 0;i < MAX_VOLUME_STEPS;i++)
    {
    	t += VOLUME_STEP_SIZE;
        
    	pos = orig + dir * t;
        
        //Stop if the sample becomes completely opaque or leaves the volume.
        if(max(max(abs(pos.x),abs(pos.y)),abs(pos.z))-1.0 > 0.0) {break;}
        
        vec4 vol = Volume(pos);
        vol.rgb *= vol.w;
        
        col += vol;
    }
    
    return col.rgb;
}

// Function 9
vec4 volumeFunc(vec3 p)
{
    //p.xz = rotate(p.xz, p.y*2.0 + iTime);	// firestorm
	float d = distanceFunc(p);
	return shade(d);
}

// Function 10
vec3 intensityToColour(float i) {
	// Algorithm rearranged from http://www.w3.org/TR/css3-color/#hsl-color
	// with s = 0.8 l = 0.5
	float h = 0.666666 - (i * 0.666666);
	
	return vec3(h2rgb(h + 0.333333), h2rgb(h), h2rgb(h - 0.333333));
}

// Function 11
float GetIntensityAverage( vec2 vCoord )
{
	float fDPixel = 1.0;
	
	float fResult 	= SampleBackbuffer( vCoord + vec2(0.0, 0.0) )
			+ SampleBackbuffer( vCoord + vec2( fDPixel, 0.0) )
		      	+ SampleBackbuffer( vCoord + vec2(-fDPixel, 0.0) )
			+ SampleBackbuffer( vCoord + vec2(0.0,  fDPixel) )
			+ SampleBackbuffer( vCoord + vec2(0.0, -fDPixel) );
	
	return fResult / 5.0;       
}

// Function 12
vec3 calcVolumetric(Ray ray, float maxDist) {
 
    vec3 col = vec3(0.);
    
    Light l   = getLight();
    float is  = maxDist / 50.;
    float vrs = maxDist / float(SAMPLES - 1);
    float rs  = rand(gl_FragCoord.xy) + vrs;
    
    Ray volRay = Ray(ray.ori + ray.dir * rs, vec3(0.));
    
    for(int v = 0; v < SAMPLES; v++) {
     
        vec3 lv    = l.p - volRay.ori;
        float ld   = length(lv);
        volRay.dir = lv / ld;
        Hit i      = raymarch(volRay);
        
        if(i.dst > ld) {
         
            col += calcIrradiance(l, volRay.ori) * is;
            
        }
        
        volRay.ori += ray.dir * vrs;
        
    }
    
    return col;
    
}

// Function 13
vec2 VolumeDensity(vec3 aP)
{
    vec2 vDens = vec2(0);
    float vD = length(aP);

    float vHeight = pow(fbm(aP.yxz*0.1 + iTime*CLOUDS_SPEED*0.1), 2.0)*10.0;
    float vTop = CLOUDS_TOP - vHeight;

    if (aP.y >= vTop && aP.y <= CLOUDS_BASE)
    {
        aP.x += vHeight * 10.0;
        float vCov = pow(fbm(aP.yxz*0.05 + iTime*CLOUDS_SPEED*0.5), 1.5)*4.0;
        aP.x -= vHeight * 10.0;

        float vV = (fbm(aP.yxz*1.0 + iTime*CLOUDS_SPEED*4.0)+
                    fbm(aP.xzy*3.0 + iTime*CLOUDS_SPEED*8.0)*0.8) * 0.4;
        vV = pow(vV, 0.5);
        float vT = (fbm(aP*0.2+ iTime*CLOUDS_SPEED))/max(0.00001, CLOUD_COVERAGE*vCov);
		vT /= clamp((CLOUDS_BASE - aP.y)/(abs(CLOUDS_BASE-vTop)*0.3), 0.0, 1.0);
        vT /= clamp((aP.y - vTop)/(abs(CLOUDS_BASE-vTop)*0.5), 0.0, 1.0);

        float vOut;
        if (vV >= vT+CLOUD_SMOOTHNESS)
        {
            vOut = vV;
        }
        else if (vV >= vT)
        {
            vOut = vV * (vV-vT)/(CLOUD_SMOOTHNESS);
        }
        else
        {
            vOut = 0.0;
        }
        
        vDens += vec2(vOut*CLOUD_DENSITY_MULT);
    }
    
    if (vD < 1.0)
    {
        vDens += vec2(1.0);
    }
     
    vDens += ATMO_DENS;
    
    return vDens * vec2(SIGMA_E, SIGMA_S);
}

// Function 14
vec3 intensityToColour(float i) {
	float h = 0.666666 - (i * 0.666666);
	return vec3(h2rgb(h + 0.333333), h2rgb(h), h2rgb(h - 0.333333));
}

// Function 15
float VolumetricExplosion(vec3 p, float r, float t)
{
    R(p.yx, iMouse.x*0.008*M_PI+t*0.1);
	p = p/1.5f;
	//float VolExplosion = VolumetricExplosion(p/0.5)*0.5; // scale
    
    
    float final = sdSphere(p.xy,t-r) * 0.25f;
   // final += noise(p*12.5)*.2;
	//final *= 0.5f-perlin_noise(vec3(p.xy,t));
    final += SpiralNoiseC(p.zxy*0.4132+333., t)*(4.0f-r); //1.25;

    return (1.0f-final) * 0.25f;
}

// Function 16
float volumetricShadow(in vec3 from, in vec3 dir)
{
    float shadow = 1.0;
    float cloud = 0.0;
    float dd = 1.0 / VOL_SHADOW_STEPS;    
    vec3 pos;
    for(float s=0.5; s < VOL_SHADOW_STEPS - 0.1; s+=1.0)// start at 0.5 to sample at center of integral part
    {
        pos = from + dir*(s/VOL_SHADOW_STEPS);
        cloud = atmThickness(pos);
        shadow *= exp(-cloud * dd);
    }
    return shadow;
}

// Function 17
vec3 calculateVolumetricClouds(positionStruct pos, vec3 color, float dither, vec3 sunColor)
{
	const int steps = volumetricCloudSteps;
    const float iSteps = 1.0 / float(steps);
    
    //if (pos.worldVector.y < 0.0)
     //   return color;
    
    float bottomSphere = rsi(vec3(0.0, 1.0, 0.0) * earthRadius, pos.worldVector, earthRadius + cloudMinHeight).y;
    float topSphere = rsi(vec3(0.0, 1.0, 0.0) * earthRadius, pos.worldVector, earthRadius + cloudMaxHeight).y;
    
    vec3 startPosition = pos.worldVector * bottomSphere;
    vec3 endPosition = pos.worldVector * topSphere;
    
    vec3 increment = (endPosition - startPosition) * iSteps;
    vec3 cloudPosition = increment * dither + startPosition;
    
    float stepLength = length(increment);
    
    vec3 scattering = vec3(0.0);
    float transmittance = 1.0;
    
    float lDotW = dot(pos.sunVector, pos.worldVector);
    float phase = phase2Lobes(lDotW);
    
    vec3 skyLight = calcAtmosphericScatterTop(pos);
    
    for (int i = 0; i < steps; i++, cloudPosition += increment)
    {
        float opticalDepth = getClouds(cloudPosition) * stepLength;
        
        if (opticalDepth <= 0.0)
            continue;
        
		scattering += getVolumetricCloudsScattering(opticalDepth, phase, cloudPosition, sunColor, skyLight, pos) * transmittance;
        transmittance *= exp2(-opticalDepth);
    }
    
    return mix(color * transmittance + scattering, color, clamp(length(startPosition) * 0.00001, 0.0, 1.0));
}

// Function 18
float GetLoadingScreenIntensity( vec2 vPos )
{
	vec2 vUV = vPos / kResolution;
	float r = 0.25;
	vec2 vDist = (vUV - 0.5) / r;
	float len = length(vDist);
	vec3 vNormal = vec3(vDist.x, sqrt(1.0 - len * len), vDist.y);
	vec3 vLight = normalize( vec3(1.0, 1.0, -1.0) );
	if(len < 1.0)
	{
		return max(0.0, dot(vNormal, vLight));
	}
	
	return 0.7 - vUV.y * 0.6;
}

// Function 19
vec3 renderVolumetric(Ray ray, float maxDist)
{
    vec3 color = vec3(inverseSquareIntegral(ray, 0.0, maxDist) * OMNI_LIGHT);
    
    for (int i = 0; i < NUM_LIGHTS; i++)
    {
        Range r = cone(lights[i].d, lights[i].a, ray);
        r.end = min(r.end, maxDist);
        
        if (r.end > r.start)
        {
            float boost = mix(1.0, 18.0, insideCone(lights[i].d, lights[i].a, ray.o));
            
            color += inverseSquareIntegral(ray, r.start, r.end) * lights[i].c * boost;
        }
    }
    
    return color;
}

// Function 20
float gridIntensity(vec2 fragCoord, int nodeIndex)
{
    if ((any(lessThanEqual(ivec2(fragCoord), ivec2(0)))) ||
        (any(greaterThanEqual(ivec2(fragCoord), ivec2(iResolution.xy - 1.0)))))
    {
        return 1.0;
    }
    else
    {
        ivec3 temp;
        ivec2 neighborNodeIndices = ivec2(
			getLeafNodeIndex(mix(gOrigin, gUpper, vec2(fragCoord.x - 1.0, fragCoord.y) / iResolution.xy), temp),
            getLeafNodeIndex(mix(gOrigin, gUpper, vec2(fragCoord.x, fragCoord.y - 1.0) / iResolution.xy), temp));
                
        return (any(notEqual(neighborNodeIndices, ivec2(nodeIndex)))) ? 1.0 : 0.0;
    }
}

// Function 21
float punctualLightIntensityToIrradianceFactor(const in float lightDistance,
                                               const in float cutoffDistance,
                                               const in float decayExponent){
    if(decayExponent > 0.0){
     	return pow(saturate(-lightDistance / cutoffDistance + 1.0), decayExponent);   
    }
    
    return 1.0;
}

// Function 22
float volumetricFog(vec3 v, float noiseMod)
{
    float noise = 0.0;
    float alpha = 1.0;
    vec3 point = v;
    for(float i = 0.0; i < NOISE_LAYERS_COUNT; i++)
    {
        noise += getNoiseFromVec3(point) * alpha;
     	point *= NOISE_SIZE_MULTIPLIER;
        alpha *= NOISE_ALPHA_MULTIPLIER;
    }
    
    //noise = noise / ((1.0 - pow(NOISE_ALPHA_MULTIPLIER, NOISE_LAYERS_COUNT))/(1.0 - NOISE_ALPHA_MULTIPLIER));
    noise *= 0.575;

    //edge + bloomy edge
#ifdef MUTATE_SHAPE
    float edge = 0.1 + getNoiseFromVec3(v * 0.5 + vec3(iTime * 0.03)) * 0.8;
#else
    float edge = 0.5;
#endif
    noise = (0.5 - abs(edge * (1.0 + noiseMod * 0.05) - noise)) * 2.0;
    return (smoothstep(1.0 - SHARPNESS * 2.0, 1.0 - SHARPNESS, noise * noise) + (1.0 - smoothstep(1.3, 0.6, noise))) * 0.2;
}

// Function 23
vec4 computeVolumetricLighting(vec4 material, float mediumD, float stepD, vec4 insTrans)
{
    vec3 emissiveColour = blackBodyToRGB(2500.0 + material.y * 2500.0, 3000.0);
    float stepTransmittance = computeVolumetricTransmittance(material, mediumD, stepD);
    insTrans.rgb += insTrans.a *
        (1.0 - stepTransmittance) * emissiveColour;
    insTrans.a *= stepTransmittance;
    
    return insTrans;
}

// Function 24
float VolumetricCloud(vec3 p)
{
    float final = Sphere(p,4.);
    #ifdef TWISTED
    float tnoise = noise(p*0.5);
    //final += tnoise * 1.75;
    final += SpiralNoiseC(p.zxy*0.3132*tnoise+333.)*3.25;
    #else
    final += SpiralNoiseC(p*0.35+333.)*3.0 + fbm(p*50.)*1.25;
    #endif
    return final;
}

// Function 25
vec3 calculateVolumetricLight(positionStruct pos, vec3 color, float dither, vec3 sunColor)
{
    #ifndef VOLUMETRIC_LIGHT
    	return color;
    #endif
    
	const int steps = volumetricLightSteps;
    const float iSteps = 1.0 / float(steps);
    
    vec3 increment = pos.worldVector * cloudMinHeight / clamp(pos.worldVector.y, 0.1, 1.0) * iSteps;
    vec3 rayPosition = increment * dither;
    
    float stepLength = length(increment);
    
    vec3 scattering = vec3(0.0);
    vec3 transmittance = vec3(1.0);
    
    float lDotW = dot(pos.sunVector, pos.worldVector);
    float phase = hgPhase(lDotW, 0.8);
    
    vec3 skyLight = calcAtmosphericScatterTop(pos);
    
    for (int i = 0; i < steps; i++, rayPosition += increment)
    {
        float opticalDepth = getHeightFogOD(rayPosition.y) * stepLength;
        
        if (opticalDepth <= 0.0)
            continue;
        
		scattering += getVolumetricLightScattering(opticalDepth, phase, rayPosition, sunColor, skyLight, pos) * transmittance;
        transmittance *= exp2(-opticalDepth);
    }
    
    return color * transmittance + scattering;
}

// Function 26
float GetExplosionIntensity(Explosion ex)
{
  return mix(1., .0, smoothstep(0., 5.0, distance(ex.life, 5.)));
}

// Function 27
vec3 volumelights( Ray ray )
{   
    float caststep=0.5;    
    vec3 colour = vec3(0.0, 0.0, 0.0);                
    float castdistance = 0.0;
    for (int i=0; i<numlights; i++)
    {
	    castdistance = max(length(g_lights[i].pos-ray.pos)*1.1, castdistance);    
    }
    float castscale=castdistance/caststep;
	float obscurity = 0.0;
    
    for (float t=0.0; t<castdistance; t+=caststep)
    {
        vec3 pos = ray.pos + ray.dir*t; 
        obscurity += fogvalue(mist(pos, 4), pos.z) * 1.2;
		vec3 deltapos;
        float d2;
        
        if (lightning > 0.5)
        {
            vec3 nearest = nearestpointonline(g_lights[0].pos, g_lights[1].pos, pos);
            deltapos = nearest-pos;
            float d2=dot(deltapos, deltapos);
            if (d2<5.0)
            {
                colour.xyz += lightningcolour/(d2*castscale*1.0) * lightning;
            } 
        }
                
        for (int i=0; i<numlights; i++)
        {        
            deltapos = g_lights[i].pos-pos;
            d2=dot(deltapos, deltapos);
            
            if (d2<40.0)
            {
                colour.xyz += g_lights[i].colour/(d2*castscale*0.4);
            }   
            
            //colour.rgb += clamp(mist(pos), 0.0, 1.0)*0.01;
        }
    }
    
    return colour*clamp((1.0-obscurity*0.1), 0.0, 1.0);
}

// Function 28
float remapIntensity(float f, float i){
  //return i;
  // noise level
  i = to01( (i - noiseLevel) / (1.0 - noiseLevel) );
  float k = f-1.0;
  i *= ( fftTrebles - fftBass*k*k ) * fftPreamp;
  // more dynamic
  i *= (i+fftBoost);
    
  return i*fftAmp;
  // limiter, kills dynamic when too loud
  //return 1.0 - 1.0 / ( i*4.0 + 1.0 );
}

// Function 29
void mainVolume( out vec4 voxColor, in vec3 voxCoord)
{
    vec3 uvw = voxCoord / iVResolution;
    
    #ifdef OPENGL_FIX
    	vec3 last = texture(iChannel0,gl_FragCoord.xy/iResolution.xy).xyz;
    #else
    	vec3 last = sample3D(iChannel0, uvw, iVResolution).xyz;
	#endif

    vec3 next = vec3(0);
    vec3 vel = last;
    for(float i = 0.0;i < STEPS;i++)
    {
    	next = Integrate(last, iTimeDelta * SPEED);
        last = next;
    } 
    vel = (next - vel)/(iTimeDelta*SPEED);
	
     //Setup initial conditions.
    if(iFrame <= 30 || KeyPressed(KEY_SPACE))
    {
        uvw = (uvw - 0.5) * 2.0;
        
        startOrig += uvw * startRang;
        
        voxColor = vec4(startOrig, 0);
    }
    else //Save current position.
    {
        voxColor = vec4(next, length(vel));
    } 
}

// Function 30
float veVolumetricExplosion(vec3 p, float radius, float maxRadius, float veProgress) {
    float sdSphere = veSphere(p, radius)*(4. * 1.92 *.25/1.92 * 1.0*        veExplosionRadius(8.0,1.0)/maxRadius);
    float noise1 = (veLOW_QUALITY) ? veNoise(p * 12.5) * .2 : veFBM(p * 50.);
    float age = mix(2.0, -7.5, 1.-pow(1.-veProgress,2.5));
    float noise2 = veSpiralNoiseC(p.zxy * 0.4132 + 0.333*vec3(0,0,1) * (25.*veProgress+max(0.,iMouse.x-15.0) * 0.1));
    float result = min(0.,sdSphere)*.999 +
        (0.25+1.75*veProgress+3.*veProgress*veProgress+4.*veProgress*veProgress*veProgress)*max(0.,sdSphere) +
        0.999*noise1 +
        0.999*age +
        0.999*(noise2+1.0/1.25) * 1.25;
	return result;
}

// Function 31
vec2 GetIntensityGradient(vec2 vCoord)
{
	float fDPixel = 1.0;
	
	float fPX = SampleBackbuffer(vCoord + vec2( fDPixel, 0.0));
	float fNX = SampleBackbuffer(vCoord + vec2(-fDPixel, 0.0));
	float fPY = SampleBackbuffer(vCoord + vec2(0.0,  fDPixel));
	float fNY = SampleBackbuffer(vCoord + vec2(0.0, -fDPixel));
	
	return vec2(fPX - fNX, fPY - fNY);              
}

// Function 32
float intensity(vec3 pixel) {
	return (pixel.r + pixel.g + pixel.b) / 3.0;
}

// Function 33
float getAvgVolume(float v, int steps)
{
    float sum = 0.0;
    float x = 0.0;
	for (int i = 0; i < steps; i++)
    {
        x = fract(v + float(i) * FREQ_STEP);
        								//pow for non linear spectrum
        sum += texture(iChannel0, vec2(pow3(x), 0.0)).r * pow(x, 0.08) * (1.0 + v * 0.5); 
    }
    
    return (sum / float(steps));
}

// Function 34
vec3 DirectLightOnVolume(in Ray ray, in float len) {
    const float strata = 1.0/float(VOLUME_DIRECT_LIGHT_SAMPLES);
    vec3 volumeDirectLight = vec3(0.);
    for(int i=0; i<VOLUME_DIRECT_LIGHT_SAMPLES; i++) {
        float particleDist;
        float particlePdf;
        //avoid sampling other side of the light
        float maxt = -(ray.origin.y - light.pos.y) / ray.dir.y;//doesn't work :(
        len = (ray.dir.y > 0.0) ? min(maxt, len) : len;
        float xi = strata*(float(i)+rnd());
        
        sampleEquiAngular( ray, len, xi, light.pos, particleDist, particlePdf );
        vec3 particlePos = ray.origin + particleDist*ray.dir;
        volumeDirectLight += salmpleLightForParticle(particlePos, ray.time) / particlePdf;
        //volumeDirectLight += samplePhaseForParticle(particlePos, ray.time) / particlePdf;
    }
    return volumeDirectLight / float(VOLUME_DIRECT_LIGHT_SAMPLES);
}

// Function 35
vec3 getVolumetricLightScattering(float opticalDepth, float phase, vec3 p, vec3 sunColor, vec3 skyLight, positionStruct pos)
{
    float intergal = calculateScatterIntergral(opticalDepth, 1.11);
    
	vec3 sunlighting = sunColor * phase * hPi * sunBrightness;
         sunlighting *= getCloudShadow(p, pos);
    vec3 skylighting = skyLight * 0.25 * rPi;
    
    return (sunlighting + skylighting) * intergal * pi;
}

// Function 36
float getAmplitude(float octave)
{
    return 1.0 / pow(2.0, octave);
}

// Function 37
vec3 getVolumetric(vec2 uv, vec2 c) {
    vec3 sum = vec3(0.);
    float  w = 1./float(SAMPLES);
    for(int i = 0; i < SAMPLES; i++) {
        sum += 1.-texture(iChannel0, uv).w;
        uv += (c-uv)*RADIUS;
    }
    return sum * w * INTENSITY;
}

// Function 38
float FFTBand_amplitude(FFTBand band)
{
    return length(band.f);
}

// Function 39
float gdVolume(vec3 p){return.1-length(p)*.05+fbm(p*.3);}

// Function 40
float VolumetricExplosion(vec3 p)
{
    float final = Sphere(p,4.);
    #ifdef LOW_QUALITY
    final += noise(p*12.5)*.2;
    #else
    final += fbm(p*50.);
    #endif
    final += SpiralNoiseC(p.zxy*0.4132+333.)*3.0; //1.25;

    return final;
}

// Function 41
vec3 volume(vec3 p,vec3 rd){
    if(p.x*p.x>1.)return vec3(0);
    if(p.y*p.y>1.)return vec3(0);
    if(p.z*p.z>1.)return vec3(0);
    
    vec3 col=vec3(0);
    col.r=smoothstep(-1.,1.,p.x);
    col.g=smoothstep(-1.,1.,p.y);
    col.b=smoothstep(-1.,1.,p.z);
    //col.r+=0.4/dot2(p-vec3(0.5,0,0));
    //col.g+=0.4/dot2(p-vec3(0,0.5,0));
    //col.b+=0.4/dot2(p-vec3(0,0,0.5));
    
    return col;
}

// Function 42
vec3 GetVolumetricLighting(Ray ray, float maxDist, vec2 fragCoord)
{
	vec3 color = vec3(0,0,0);
	Light light;
	light.p = vec3(sin(iTime*0.3)*2.0,5,cos(iTime*0.3)*2.0+4.0);
	light.color = vec3(1,1,1);
	light.radius = 20.0;
	
	float inscattering = maxDist/200.0;
	float volRayStep = maxDist/float(VOLUMETRIC_SAMPLES-1);
	float randomStep = rand(fragCoord.xy)*volRayStep;
	Ray volRay;
	volRay.o = ray.o + ray.dir*randomStep;
	for(int v = 0; v < VOLUMETRIC_SAMPLES; v++)
	{
		vec3 lightVec = light.p-volRay.o;
		float lightDist = length(lightVec);
		volRay.dir = lightVec/lightDist;
		Intersection i = SceneIntersection(volRay);
		if(i.dist > lightDist)
		{
			color += CalcIrradiance(light, volRay.o)*inscattering;
		}
		volRay.o += ray.dir * volRayStep;
	}
	
	return color;
}

// Function 43
float TweetVolume(float t)
{
    float n = NoiseSlope(t*11.0, .1) * abs(sin(t*14.0))*.5;
    n = (n*smoothstep(0.4, 0.9, NoiseSlope(t*.5+4.0, .1)));
    return n;
}

// Function 44
float intensity(float time) {
    if (time < M1) {
        return time / M1;
    } else if (time < M15) {
        return (time - M1) / (M15 - M1);
    } else if (time < M2) {
        return (time - M15) / (M2 - M15);
    } else if (time < M3) {
        return (time - M2) / (M3 - M2);
    } else {
        return time - M3;
    }
}

// Function 45
float IntersectVolumetric(in vec3 rayOrigin, in vec3 rayDirection, float maxT)
{
    // Precision isn't super important, just want a decent starting point before 
    // ray marching with fixed steps
	float precis = 0.5; 
    float t = 0.0f;
    for(int i=0; i<MAX_SDF_SPHERE_STEPS; i++ )
    {
	    float result = QueryVolumetricDistanceField( rayOrigin+rayDirection*t);
        if( result < (precis) || t>maxT ) break;
        t += result;
    }
    return ( t>=maxT ) ? -1.0 : t;
}

// Function 46
float volumetricShadowLayer(in vec3 from, in float sundotrd ) {
    float dd = CLOUDS_LAYER_SHADOW_MARGE_STEP_SIZE;
    vec3 rd = SUN_DIR;
    float d = dd * .5;
    float shadow = 1.0;

    for(int s=0; s<CLOUD_SELF_SHADOW_STEPS; s++) {
        vec3 pos = from + rd * d;
        float norY = clamp( (pos.y - CLOUDS_LAYER_BOTTOM ) * (1./(CLOUDS_LAYER_TOP - CLOUDS_LAYER_BOTTOM)), 0., 1.);

        if(norY > 1.) return shadow;

        float muE = cloudMapLayer( pos, rd, norY );
        shadow *= exp(-muE * dd);

        dd *= CLOUDS_SHADOW_MARGE_STEP_MULTIPLY;
        d += dd;
    }
    return shadow;
}

// Function 47
float getAmplitude(float octave)
{
    return 1.0 / pow(2.2, octave);
}

// Function 48
vec3 WorldToVolumeCoord(vec3 p, vec3 min_box, vec3 max_box)
{
    return (p - min_box) / (max_box - min_box);
}

// Function 49
vec3 calculateVolumetricLight(vec3 p, vec3 o, vec3 od)
{
	
	vec3 light = getLight(p, o) * selfShadow(p, o);
	vec3 sphere = smoothstep(0.8, 0.81, sphere(p, o)) * lightColor*10.0;
	
	return (light * od + sphere);
}

// Function 50
float getVolume(in float x)
{
    float bar = floor(x * float(BARS));
    float freq = bar * (1./float(BARS));
        
    freq=pow(10.0, freq*2.0-1.0)/10.0; //Logarithmic scale
    
    return texture(iChannel1, vec2(freq, 0.0)).x;
}

// Function 51
vec3 GetLightIntensity(in vec3 pos)
{
    Ray ray;
    ray.org = pos;
    ray.dir = normalize(lightPos - ray.org);
    ray.org += ray.dir*epsilon;
    
    RayTraceSceneResult scene = RayTraceScene(ray, true);
    return scene.hit.t > length(lightPos - ray.org) ? ComputeLightAttenuation(ray.org) : vec3(0.);
}

// Function 52
float intensity(float x){return pow(x,gamma);}

// Function 53
float remapIntensity(float f, float i){
  // noise level
  i = to01( (i - noiseLevel) / (1.0 - noiseLevel) );
  float k = f-1.0;
  // preamp, x2 for trebles -> x1 for bass
  //i *= ( 2.0 - 1.0*k*k ) * fftPreamp;
  //i *= ( 3.0 - 1.5*k*k ) * fftPreamp;
  i *= ( 3.0 - 1.6*k*k ) * fftPreamp;
  // more dynamic
  i *= (i+fftBoost);
  // limiter
  return i*fftAmp;
  // limiter, kills dynamic when too loud
  //return 1.0 - 1.0 / ( i*4.0 + 1.0 );
}

// Function 54
float intensity(vec4 col) {
	return dot(col.rgb, vec3(0.3126, 0.8152, 0.0822));
}

// Function 55
float computeVolumetricTransmittance(vec4 material, float mediumD, float stepD)
{
    float density = max(0.001, -mediumD*40.0);
    return exp(-stepD * density * 1.0);
}

// Function 56
float lightIntensity(vec3 point, Light light) {
	float dist = distance(point, light.pos);

	return light.inten / (dist*dist);
}

// Function 57
float QueryVolumetricDistanceField( in vec3 pos)
{    
    // Fuse a bunch of spheres, slap on some fbm noise, 
    // merge it with ground plane to get some ground fog 
    // and viola! Big cloudy thingy!
    vec3 fbmCoord = (pos + 2.0 * vec3(iTime, 0.0, iTime)) / 1.5f;
    float sdfValue = sdSphere(pos, vec3(-8.0, 2.0 + 20.0 * sin(iTime), -1), 5.6);
    sdfValue = sdSmoothUnion(sdfValue,sdSphere(pos, vec3(8.0, 8.0 + 12.0 * cos(iTime), 3), 5.6), 3.0f);
    sdfValue = sdSmoothUnion(sdfValue, sdSphere(pos, vec3(5.0 * sin(iTime), 3.0, 0), 8.0), 3.0) + 7.0 * fbm_4(fbmCoord / 3.2);
    sdfValue = sdSmoothUnion(sdfValue, sdPlane(pos + vec3(0, 0.4, 0)), 22.0);
    return sdfValue;
}

// Function 58
float VolumeIntensity(vec3 p) {
    // p : [0, 1]
    // try to have no hard edges in intensity here -> would produce "ugly" normals
    
    // simulate sphere in center of volume with soft fall off at the edge of sphere to avoid "ugly" normals
    float i = S(.6, .4, distance(p, vec3(.5)));
    // float i = distance(p, vec3(.5)) < .5 ? 1. : 0.; // sphere without soft edge
    
    return i;
}

// Function 59
float getVolumeAY(float v){
 return sat(exp2(-(31.-(v*31./15.))*.215)-.011);}

// Function 60
float TweetVolume(float t)
{
    float n = NoiseSlope(t*11.0, .1) * abs(sin(t*14.0))*.5;
    n = (n*smoothstep(0.4, 0.9, NoiseSlope(t*.5+4.0, .1)));
    return min(n*n * 2.0, 1.0);
}

// Function 61
vec3 intensity(vec3 x){return pow(x,vec3(gamma));}

// Function 62
float getVolumeAY(float v_0_15)
{
    float vol = exp2(-(31.0 - (v_0_15 * 31.0 / 15.0)) * 0.215) - 0.011;

    return clamp(vol, 0.0, 1.0);
}

// Function 63
vec3 getVolumetricCloudsScattering(float opticalDepth, float phase, vec3 p, vec3 sunColor, vec3 skyLight, positionStruct pos)
{
    float intergal = calculateScatterIntergral(opticalDepth, 1.11);
    
    float beersPowder = powder(opticalDepth * log(2.0));
    
	vec3 sunlighting = (sunColor * getSunVisibility(p, pos) * beersPowder) * phase * hPi * sunBrightness;
    vec3 skylighting = skyLight * 0.25 * rPi;
    
    return (sunlighting + skylighting) * intergal * pi;
}

// Function 64
float projectVolumetric(vec2 uv, inout vec4 backing) {
    vec3 point = vec3(0.,0.,0.);
    vec2 uvh = uv-0.5;
    uvh *= 2.;
    float volume = 0.;
    vec3 increment = vec3(uvh/CUBE_FOV,1.);
    increment = normalize(increment);
    vec3 nrel = point;
    float gi = 0.;
    backing = vec4(0.,0.,0.,1.);
    for (float i=0.; i<=MAX_DIST;i+=0.02) {
        vec3 prel = point-vec3(0.,0.,CUBE_DIST);
        rotateUV(prel, vec2(16.+iTime*24.,24.));
        vec2 wc = prel.xy-0.5;
        wc /= BOX_SIZE;
        float rang = atan(prel.x,prel.y)/radians(360.);
        float fdist = length(prel.xy);
        vec4 ewave = texture(iChannel0,wc);
        vec4 hwave = texture(iChannel0,vec2(rang,fdist/2.));
        float en = BOX_SIZE+ewave.r*WAVE_AMPLI;
        if (prel.x > -en && prel.x < en && prel.y > -en && prel.y < en && prel.z > -en && prel.z < en) {
            volume += 0.0072;
            nrel = prel;
            gi = i;
        }
        point += increment*0.02;
    }
    volume = clamp(volume,0.,1.);
    return volume;
}

// Function 65
float chirp_amplitude()
{
	return 1.0 / exp(10.0 * mod_time());    
}

// Function 66
float brushIntensity(float r)
{
    if(r/RADIUS <0.707)
        return INTENSITY;
    return -INTENSITY;
}

// Function 67
float remapIntensity(float f, float i){
    //return i; // nothing
    float k = to01( trebles(f,bass(f,i))*fftPreamp);
    //return k; // no dynamic
    return k*(k+fftBoost); // more dynamic
}

// Function 68
vec3 intensityToColour(float i) {
	// Algorithm rearranged from http://www.w3.org/TR/css3-color/#hsl-color
	// with s = 0.8, l = 0.5
	float h = 0.666666 - (i * 0.666666);
	
	return vec3(h2rgb(h + 0.333333), h2rgb(h), h2rgb(h - 0.333333));
}

// Function 69
vec3 GetVolumetricShadow(in vec3 p, in vec3 L)
{
	vec3 shadow = vec3(0.36, 0.2, 0.176) * 8.0;
    const float stopValue = 2.0;
    for (float i = 0.5; i < stopValue; i += 1.0)
    {
        vec3 pos = p + L * (i / stopValue);
        float density = GetDensityLOD1(pos);
        shadow *= exp(-density * (i / stopValue));
    }
    return shadow;
}

// Function 70
float VolumetricExplosion(vec3 p)
{
    float final = Sphere(p,4.);
    final += noise(p*20.)*.4;
    final += SpiralNoiseC(p.zxy*fbm(p*10.))*2.5; //1.25;

    return final;
}

// Function 71
vec4 MarchVolume(vec3 orig, vec3 dir)
{
    vec2 hit = IntersectBox(orig, dir, vec3(0), vec3(2));
    
    if(hit.x > hit.y){ return vec4(0); }
    
    //Step though the volume and add up the opacity.
    float t = hit.x;   
    vec4 dst = vec4(0);
    vec4 src = vec4(0);
    
    for(int i = 0;i < MAX_VOLUME_STEPS;i++)
    {
        t += VOLUME_STEP_SIZE;
        
        //Stop marching if the ray leaves the cube.
        if(t > hit.y){break;}
        
    	vec3 pos = orig + dir * t;
        
        vec3 uvw = 1.0 - (pos * 0.5 + 0.5);
        
        #if(LINEAR_SAMPLE == 1)
            src = texture3DLinear(iChannel0, uvw, vres);
        #else
            src = texture3D(iChannel0, uvw, vres);
        #endif
        
        src = clamp(src, 0.0, 1.0);
        
        src.a *= DENSITY_SCALE;
        src.rgb *= src.a;
        
        dst = (1.0 - dst.a)*src + dst;
        
        //Stop marching if the color is nearly opaque.
        if(dst.a > MAX_ALPHA){break;}
    }
    
    return vec4(dst);
}

// Function 72
void TestSceneVolumetric(in vec3 rayPos, out SRayVolumetricInfo info)
{   
    info.scatterProbability = 0.0f;
    info.anisotropy = 0.0f;
    info.emissive = vec3(0.0f, 0.0f, 0.0f);
    info.absorption = vec3(0.0f, 0.0f, 0.0f);
    
    #if SCENE == SCENE_FOG1 || SCENE == SCENE_FOG2
    
    	float dist = BoxDistance(vec3(0.0f, 25.0f, -30.0f), vec3(60.0f, 25.0f, 30.0f), 0.0f, rayPos);
    	info.scatterProbability = step(dist, 0.01f) * 0.02f;
    
    	#if (SCENE == SCENE_FOG2)
    		if (rayPos.x > 0.0f)
                info.anisotropy = 0.5f;
    		else
                info.anisotropy = -0.5f;
        #endif
    
    #elif SCENE == SCENE_ABSORPTION1 || SCENE == SCENE_ABSORPTION2
    	if (step(SphereDistance(vec4( -25.0f, 40.0f, -30.0f, 20.0f), rayPos), 0.01f) > 0.0f)
        {
            #if SCENE == SCENE_ABSORPTION1
            	info.scatterProbability = 0.1f;
            #endif
            info.absorption = vec3(0.0f, 0.4f, 0.9f) * 0.1;
        }
    
    	if (step(SphereDistance(vec4( 25.0f, 40.0f, -30.0f, 20.0f), rayPos), 0.01f) > 0.0f)
        {
            info.scatterProbability = 0.1f;
        }
    
    #elif SCENE == SCENE_EMISSION
    	if (step(SphereDistance(vec4( -25.0f, 40.0f, -30.0f, 20.0f), rayPos), 0.01f) > 0.0f)
        {
           	info.scatterProbability = 0.1f;
            info.emissive = vec3(0.0f, 0.4f, 0.9f) * 0.05f;
        }
    
    	if (step(SphereDistance(vec4( 25.0f, 40.0f, -30.0f, 20.0f), rayPos), 0.01f) > 0.0f)
        {
            info.scatterProbability = 0.1f;
        }
    
    #elif SCENE == SCENE_LIGHTINFOG1
    
    	if (step(SphereDistance(vec4( 0.0f, 40.0f, -30.0f, 20.0f), rayPos), 0.01f) > 0.0f)
        {
           	info.scatterProbability = 0.1f;            
            info.absorption = vec3(1.0f, 1.0f, 1.0f) * 0.05f;
        }
    
    #elif SCENE == SCENE_LIGHTINFOG2
    
 		float dist = length(rayPos - vec3(0.0f, 40.0f, -30.0f));

        float fogAmount = smoothstep(25.0f, 15.0f, dist);
    	info.scatterProbability = fogAmount * 0.1f;
    	info.absorption = vec3(1.0f, 1.0f, 1.0f) * 0.05f * fogAmount;
    
    #elif SCENE == SCENE_MULTIFOG
    
    	float weight = 0.0f;
    
    	if (BezierDistance(rayPos, vec3(10.0f, 0.0f, -30.0f), vec3(10.0f, 40.0f, -30.0f), vec3(50.0f, 50.0f, -30.0f), 7.0f) < 0.0f)
        {
            weight += 1.0f;
            info.scatterProbability += 0.2f;
        }
    
    	if (BezierDistance(rayPos, vec3(-50.0f, 40.0f, 0.0f), vec3(-10.0f, 50.0f, 10.0f), vec3(0.0f, 0.0f, 10.0f), 7.0f) < 0.0f)
        {
            weight += 1.0f;
            info.scatterProbability += 0.05f;
            info.absorption = vec3(0.1f, 0.1f, 0.05f) * 2.0f;
        }    
    
    	if (BezierDistance(rayPos, vec3(-40.0f, 0.0f, -40.0f), vec3(-40.0f, 40.0f, -20.0f), vec3(-20.0f, 60.0f, 0.0f), 7.0f) < 0.0f)
        {
            weight += 1.0f;
            info.scatterProbability += 0.1f;
            info.emissive = vec3(1.0f, 1.0f, 0.125f) * 0.1f;
        }
        
    	if (BoxDistance(vec3(30.0f, 10.0f, -20.0f), vec3(15.0f, 10.0f, 15.0f), 5.0f, rayPos) < 0.0f)
        {
            weight += 1.0f;
            info.absorption = vec3(0.1f, 0.4f, 0.9f) * 0.25f;
        }
    
    	if (weight > 0.0f)
        {
            info.scatterProbability /= weight;
            info.absorption /= weight;
            info.emissive /= weight;
        }
    #elif SCENE == SCENE_ORGANICFOG
    
    	float density = noise((rayPos + vec3(10.0f, 0.0f, 0.0f)) / 25.0f);
    	const float threshold = 0.6f;
    
    	density = clamp((density - threshold) / (1.0f - threshold), 0.0f, 1.0f);
    	info.scatterProbability = density * 0.075f;
    #endif
}

// Function 73
float neuralVolume(in vec3 p) {
vec4 f00=sin(p.x*vec4(.135,-.92,.997,.893)+p.y*vec4(.345,.911,-.651,-1.521)+p.z*vec4(-.825,.656,-.139,.463)+vec4(-.428,.988,-.046,-.945));
vec4 f01=sin(p.x*vec4(.025,1.443,-1.637,.75)+p.y*vec4(-1.56,-1.163,-.861,.431)+p.z*vec4(-1.054,.9,.071,.182)+vec4(-.201,1.218,-.115,-.13));
vec4 f02=sin(p.x*vec4(1.734,-.576,-2.458,1.011)+p.y*vec4(.851,.527,.45,3.14)+p.z*vec4(-1.08,-.05,.364,.691)+vec4(.826,-.068,.883,.768));
vec4 f03=sin(p.x*vec4(.613,-1.078,-.187,-1.355)+p.y*vec4(.051,-2.254,1.874,-.106)+p.z*vec4(1.081,-1.454,2.599,1.164)+vec4(1.106,-.922,-.243,.976));
vec4 f10=sin(mat4(.237,1.184,-.33,-.309,.159,-.064,-1.225,.508,-.225,-.165,.229,-.875,.076,.21,.066,-.435)*f00
    +mat4(.314,1.199,.731,-.684,-.823,-.291,-.164,-.045,-.91,-.19,.737,-.423,-.123,.186,-.421,.414)*f01
    +mat4(-.115,-.24,-.15,-.717,.602,.069,-.02,1.011,.675,.108,.622,-.361,.271,-.1,-.129,.324)*f02
    +mat4(-.423,-.41,.885,-.368,.071,.488,.829,1.093,-.33,-.02,.687,.951,-.645,-.794,1.003,-1.404)*f03
    +vec4(-.314,.42,.682,.317));
vec4 f11=sin(mat4(.623,-.917,-.059,-.484,.052,-.276,.139,-.057,.517,.572,-.709,-.641,-.634,.198,-.309,.73)*f00
    +mat4(.848,-.171,-1.178,-.745,-.204,-.572,-.527,-.213,1.727,.241,.068,.143,-.447,.469,.724,.306)*f01
    +mat4(.76,-.992,.072,-.405,-.568,-.705,.697,.529,-.335,.303,.432,.145,.04,-.056,.288,.171)*f02
    +mat4(-.274,.7,-.054,-.363,.056,-.209,.752,.272,-.394,.361,.048,.035,-.176,.607,-.182,.402)*f03
    +vec4(.498,.69,.346,.271));
vec4 f12=sin(mat4(-.002,-.7,.11,-.593,1.26,-.066,-.275,-.545,-.502,.698,.187,-.258,-.593,-.014,.767,.504)*f00
    +mat4(-.788,-.514,-.677,.159,-.181,.25,1.02,.09,.432,-.637,-.284,-1.395,-.355,.422,.785,.94)*f01
    +mat4(.921,1.168,-.1,-.368,.65,-.37,-.272,-.237,.1,-.382,-1.027,-.366,.283,-.588,.271,.115)*f02
    +mat4(-.317,1.604,.237,.099,.279,-.171,-.186,-.055,-.186,.185,.531,.769,-1.034,-.079,.731,.179)*f03
    +vec4(.315,.32,.234,-.247));
vec4 f13=sin(mat4(.432,.461,-.251,-.908,.118,.238,-.62,.765,.697,.838,.623,-.889,.589,-1.182,.556,.012)*f00
    +mat4(.439,.094,.068,-.764,-.284,-.376,-.227,.695,-.842,.786,.325,.366,.96,-.432,-1.067,-.669)*f01
    +mat4(-1.989,-.098,-.284,-.049,.168,.028,-.32,.836,-.263,.361,.243,-.538,.396,.685,-.115,-.6)*f02
    +mat4(.142,.179,-.491,-.033,.666,-.194,.313,-.496,.953,-.925,.257,.116,-.007,-.351,-.388,1.296)*f03
    +vec4(-.286,.268,.283,.384));
vec4 f20=sin(mat4(1.067,.576,.5,-1.351,1.617,.496,-.806,-.089,-.99,.11,.77,-.887,.573,.052,-1.012,1.272)*f10
    +mat4(-.509,.281,.883,-.813,-.399,-1.383,-.242,1.315,-.499,.01,-.12,1.965,-.121,-.214,-2.116,-1.099)*f11
    +mat4(.901,.547,.566,-2.171,-.461,-.476,.454,.768,-1.299,-.416,.3,-.383,.142,-.181,-.009,.49)*f12
    +mat4(-.097,.154,.28,1.032,-.722,.505,-.243,-.543,-.404,-.341,-1.44,.607,-1.119,-.307,.489,-1.45)*f13
    +vec4(-.528,.114,-1.355,.252));
vec4 f21=sin(mat4(-.557,1.041,-.158,.194,-.752,.698,-.197,-.19,-.001,-.476,-.422,.448,-.181,.869,.953,-.999)*f10
    +mat4(.201,.42,-.784,-.225,.706,-.481,-.053,-.872,.126,-.226,-.153,-.126,-.046,.366,-.939,-.012)*f11
    +mat4(-.417,-.122,-1.034,.703,.524,-.386,.321,-.627,.381,.476,.171,-.402,.319,.038,.652,-.168)*f12
    +mat4(-.266,.401,1.156,-.259,-.405,1.071,.893,.748,-.134,.318,.808,.448,.217,-.316,-.825,.276)*f13
    +vec4(.122,.251,-.471,-.455));
vec4 f22=sin(mat4(-.305,1.162,.973,-.676,.767,-1.133,1.428,-.979,-1.136,1.027,-1.219,.286,.618,.265,1.458,-.905)*f10
    +mat4(.54,-.608,-.982,.567,.95,-.17,-.41,.435,-.303,.344,.365,.071,.476,-1.229,.844,-.176)*f11
    +mat4(.638,-.702,.295,.409,.565,-.454,-1.215,.15,.073,-.058,-1.183,.654,.733,-.836,-.686,-.318)*f12
    +mat4(.014,-.521,.263,-.234,-1.002,.819,.821,-.636,-1.256,-1.13,1.972,-.295,-.522,.21,-1.572,.714)*f13
    +vec4(.185,-.447,.147,.098));
vec4 f23=sin(mat4(-1.731,.056,.409,.383,.686,-.257,-.788,.816,-.286,.455,-.436,-.631,-.877,-.547,-.16,1.115)*f10
    +mat4(.72,.585,1.283,-.416,-.108,.838,.228,-.616,-.188,-.517,-1.421,.339,-.07,.594,.406,1.143)*f11
    +mat4(-.033,.243,1.652,.793,-.308,1.091,.522,-.488,-.253,.923,.113,-.276,-.714,-.292,-.316,-.049)*f12
    +mat4(-.091,-.93,-.056,.757,-.748,-.943,-.746,.662,.657,-.903,-.781,.594,.302,.593,-1.175,-.572)*f13
    +vec4(.231,-.572,.081,.417));
vec4 f30=sin(mat4(.315,.409,.083,.363,-1.403,.808,-.4,-.939,-.884,.19,-.307,-.108,.514,-.2,.185,-.058)*f20
    +mat4(-.128,.494,.029,-.702,.599,-.318,-.25,-.187,1.227,-.095,.094,-.128,.654,-.295,-.511,.316)*f21
    +mat4(.183,.325,.106,.282,-.856,.088,-.24,-.001,.817,-.571,.114,-.437,-.973,.449,-.11,.282)*f22
    +mat4(-.667,-.02,-.083,.162,.015,-.363,-.605,-.683,.239,-.088,.156,.159,.832,-.301,.224,.395)*f23
    +vec4(.043,-.11,-.304,-.07));
vec4 f31=sin(mat4(.473,.023,.568,.182,-.401,-.025,-.402,-1.307,-.338,-.286,.112,-.784,.107,.184,-.12,.415)*f20
    +mat4(.373,-.005,-.018,-.133,.135,.298,-.421,.479,-.151,.395,-.615,.995,-.139,.448,-.313,.08)*f21
    +mat4(.062,.051,.221,-.043,-.012,-.262,.467,-.65,.416,.164,-.003,.844,.042,-.205,-.186,-.835)*f22
    +mat4(-.154,-.283,.331,-.52,-.621,.373,-.804,-.261,.298,-.063,.369,.261,.243,.142,-.225,.765)*f23
    +vec4(-.224,-.446,.427,-.272));
vec4 f32=sin(mat4(1.812,-.311,-.889,-.265,-2.924,-.344,2.515,-1.312,-2.656,.39,1.888,-.736,1.839,-.226,-1.219,.338)*f20
    +mat4(.158,-.342,-.45,-.489,1.195,-.193,-.442,.789,1.664,-.668,-1.055,.804,.585,-.828,.254,-.43)*f21
    +mat4(1.203,-.327,-.743,-.423,-2.251,.4,1.369,-.552,2.255,-.175,-1.347,.844,-.784,.501,.619,-.266)*f22
    +mat4(-2.196,.307,1.213,-.512,-2.076,-.619,1.82,-.405,.963,-.051,-.593,.212,2.068,-.046,-1.471,.712)*f23
    +vec4(1.433,-.302,-1.556,-.575));
vec4 f33=sin(mat4(-1.071,.971,.6,-.127,.287,-.307,-.569,.175,.125,-.304,-.32,.016,-.145,.22,.04,-.074)*f20
    +mat4(.326,-.204,.429,-.039,.521,-.318,-.023,-.054,.328,-.067,-.014,.046,.457,-.352,.354,-.259)*f21
    +mat4(-.581,.496,.36,-.008,.203,-.336,-.024,.028,-.571,.695,-.111,-.28,.407,-.416,-.112,-.023)*f22
    +mat4(.312,-.463,.008,-.018,.911,-.82,-.581,-.062,-.469,.448,.222,-.078,-.069,.165,.629,.03)*f23
    +vec4(-.116,-.003,-.303,.134));
return dot(vec4(-.43,-.112,-.491,.23),f30)+dot(vec4(.378,-.747,-.252,.696),f31)+dot(vec4(-.065,-.318,.122,-.303),f32)+dot(vec4(-.521,-.659,-.376,.469),f33)-.324;
}

// Function 74
vec3 getVolumetricRaymarcher(vec3 p, vec3 o, float dither, vec3 background)
{
	const float isteps = 1.0 / float(steps);
	
	vec3 increment = -p * isteps;
	vec3 marchedPosition = increment * dither + p;
	
	float stepLength = length(increment);
	
	vec3 scatter = vec3(0.0);
	vec3 transMittance = vec3(1.0);
	vec3 currentTransmittence = vec3(1.0);
	
	for (int i = 0; i < steps; i++){
		vec3 od = calculateOD(marchedPosition) * scatterCoeff * stepLength;
		
		marchedPosition += increment;
		
		scatter += calculateVolumetricLight(marchedPosition, o, od) * currentTransmittence;
		
		currentTransmittence *= exp2(od);
		transMittance *= exp2(-od);
	}
	
	return background * transMittance + scatter * transMittance;
}

// Function 75
vec3 MarchVolume(vec3 orig, vec3 dir)
{
    vec2 hit = IntersectBox(orig, dir, vec3(0), vec3(2));
    
    if(hit.x > hit.y){ return vec3(0); }
    
    //Step though the volume and add up the opacity.
    float t = hit.x;
    vec4 col = vec4(0);
    
    for(int i = 0;i < MAX_VOLUME_STEPS;i++)
    {
    	t += VOLUME_STEP_SIZE;
        if(t > hit.y){break;}
        
    	vec3 pos = orig + dir * t;
        
        #if(LINEAR_SAMPLE == 1)
        	vec4 vol = sample3DLinear(iChannel0, pos*0.5+0.5, vres);
        #else
        	vec4 vol = sample3D(iChannel0, pos*0.5+0.5, vres);
        #endif
        
        #if(DISP_MODE == XYZ)
        	col += abs(vol) * 0.001;
        #elif(DISP_MODE == XYZ_STEP)
        	col += smoothstep(6.0, 0.8, abs(vol)) * 0.02;
        #elif(DISP_MODE == SPEED)
        	col += vec4(vol.w*0.000001);
        #endif
    }
    
    #if(DISP_MODE == SPEED)
    	return Grad(1.0-col.r);
    #else
    	return col.rgb;
    #endif
}

// Function 76
float GetSpecularIntensity(vec2 fragCoord, vec3 vEyePos, float fSpecularIntensity, float fSpecularPower, vec3 vLampPos, vec2 dxy)
{
   vec2 uv = fragCoord.xy / iResolution.xy;
   vec2 uv2 = (fragCoord.xy + vec2(DIST_DELTA, 0) + dxy) / iResolution.xy;
   vec2 uv3 = (fragCoord.xy + vec2(0, DIST_DELTA) + dxy) / iResolution.xy;
   float dx = length(texture(iChannel1,uv) - texture(iChannel1,uv2));
   float dy = length(texture(iChannel1,uv) - texture(iChannel1,uv3));
   vec3 vNormal = normalize(vec3(dx, -dy, 1.));
    
   vec3 vReflectedVector = normalize(reflect(normalize(vLampPos), vNormal));
   vec3 vVertexToEyeVector = normalize(vEyePos);
   float fSpecularFactor = clamp(dot(vVertexToEyeVector, vReflectedVector), 0., 1.);
   fSpecularFactor = pow(fSpecularFactor, fSpecularPower);
   float vResult = fSpecularIntensity * fSpecularFactor;
   
   return clamp(vResult, 0., 1.);
}

// Function 77
float getAmplitude(float freq) {
    return texture(iChannel0, vec2(FrequencyToTexture(freq), 0.0)).r;
}

// Function 78
vec2 RayMarchSDFAndVolume(in vec3 ro, in vec3 rd, in vec3 lightDirection, out vec4 scattering)
{
    float t = 0.0;
    float tmax = 120.0;
    const int numSteps = 150;
    float sdfDist = 0.0;
    float stepsTaken = 0.0;
    float density = 0.0;
    float transmittance = 1.0;
    vec3 inscatteredLight = vec3(0.0);
    vec3 L = lightDirection;
    vec3 lightColor = vec3(1.0, 0.8, 0.6) * 10.0;
    
    // --- two LOD levels
    for(int i = 0; i < 100; ++i)
    { 
    	vec3 p = ro + rd * t;
        
        sdfDist = Intersect(p); 
        if (sdfDist < 0.001) { break; }
        
        density = min(GetDensityLOD0(p), sdfDist);
        if (density > 0.99) { break; }
        
        Integrate(p, L, -rd, density, sdfDist, lightColor, transmittance, inscatteredLight);
        t += max(0.05, sdfDist * 0.2); 
        stepsTaken++;
    }
    for(int i = 0; i < 100; ++i)
    { 
    	vec3 p = ro + rd * t;
        
        sdfDist = Intersect(p); 
        if (sdfDist < 0.001) { break; }
        
        density = min(GetDensityLOD1(p), sdfDist);
        if (density > 0.99) { break; }
        
        Integrate(p, L, -rd, density, sdfDist, lightColor, transmittance, inscatteredLight);
        t += max(0.05, sdfDist * 0.2); 
        stepsTaken++;
    }
      
    scattering = vec4(inscatteredLight, transmittance);
    return vec2(t, stepsTaken / float(numSteps));
}

// Function 79
float GetVolumeValue(vec3 pos)
{
    float scale = 96.;//32.0*(2.0+sin(iTime));
    vec3 conner = mBox.Position - mBox.Vertex;
    float value = texture(iChannel0, (pos - conner) / scale).x;//选取3D纹理的一部分进行采样
    return value;
}

// Function 80
float intensity(vec4 col) {
	return dot(col.rgb, vec3(0.2126, 0.7152, 0.0722));
}

// Function 81
float ligIntensity(float t)
{
    return exp(5.*(intensityAtTime(t)-0.5));
}

// Function 82
float GetExplosionIntensity(float life)
{
  return mix(1., .0, smoothstep(0., 5.0, distance(life, 5.)));
}

// Function 83
float edgeIntensity(vec2 uv)
{
	float edgeIntensityX = 1.0;
    if( uv.x < 0.1)
    {
    	edgeIntensityX = 0.7 + 0.3*(uv.x/0.1);
    }
    else if( uv.x > 0.90)   
    {
    	edgeIntensityX = 0.7 + 0.3*((1.0-uv.x)/0.1);
    }
        
    float edgeIntensityY = 1.0;
    if( uv.y < 0.15)
    {
    	edgeIntensityY = 0.6 + 0.4*(uv.y/0.15);
    }
    else if( uv.y > 0.85)   
    {
    	edgeIntensityY = 0.6 + 0.4*((1.0-uv.y)/0.15);
    }        
    return edgeIntensityX*edgeIntensityY;
}

// Function 84
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

// Function 85
vec3 volumetricLight(vec3 p, vec3 ro, vec3 rd, vec2 uv)
{
#ifdef VOLUMETRIC_ACTIVE
    vec3 col = vec3(0.0);
    float val = 0.0;
    
   	p -= rd * noise(9090.0*uv) * 0.6;
    vec3 s = -rd * 2.2 / float(VOLUMETRIC_STEPS);
    
    for (int i = 0; i < VOLUMETRIC_STEPS; i++)
    {
        float v = getVisibility(p, light_pos, 250.0) * .015;
        p += s;
        float t = exp(p.z - 3.0);
        val += v * t;
    }  
    
    return vec3(min(val, .8));
#else
    return vec3(0.0);
#endif
}

// Function 86
float volumetricShadow(in vec3 from, in vec3 to)
{
#if D_VOLUME_SHADOW_ENABLE
    const float numStep = 16.0; // quality control. Bump to avoid shadow alisaing
    float shadow = 1.0;
    float sigmaS = 0.0;
    float sigmaE = 0.0;
    float dd = length(to-from) / numStep;
    for(float s=0.5; s<(numStep-0.1); s+=1.0)// start at 0.5 to sample at center of integral part
    {
        vec3 pos = from + (to-from)*(s/(numStep));
        getParticipatingMedia(sigmaS, sigmaE, pos);
        shadow *= exp(-sigmaE * dd);
    }
    return shadow;
#else
    return 1.0;
#endif
}

// Function 87
vec4 Volume(vec3 pos)
{
    RotateY(pos,iTime);
    RotateZ(pos,-0.5);
    
    float vol = dot(normalize(pos),vec3(1,0,0));
    
    vec3 col = mix(vec3(1.0,0.2,0.2),vec3(0.2,0.2,1.0),step(0.0,vol));
    
    vol = smoothstep(0.6,0.9,abs(vol));
    
	return vec4(col, max(0.0,vol)*0.01);  
}

// Function 88
float GetIntensityFactor(in float t)
{
    return clamp(1.0 - t, 0., 1.);
}

// Function 89
float smoothVolume(in float x, in float volume)
{
    int samples = 1;
    for(int i = -1 * (SMOOTHING / 2); i <= SMOOTHING / 2; i++)
    {
        if(i == 0)
            continue;
        float barX = x + (1./float(BARS)) * float(i);
        if(barX < 0. || barX > 1.)
        {
        	continue;
        }
    	volume += getVolume(barX);
        samples++;
    }
    return clamp(volume / float(samples), 0.0, 1.0);
}

// Function 90
float iVolume( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal )
{
    float d = -log(rand())/density;
    
    if (d < distBound.x || d > distBound.y) 
    {
        return MAX_DIST;
    } 
    else 
    {
    	return d;
    }
}

// Function 91
float ThinFilmAmplitude( float wavelength, float thickness, float cosi )
{
    float ni = N_Air;
    float nt = N_Water;
    
    float cost = GetCosT( ni, nt, cosi );

    // # The wavelength inside a medium is scaled by the index of refraction.
    // wavelength_soap = wavelength / n_soap
    // wavelength_air = wavelength / n_air
    // # First calc phase shift of reflection at rear surface, based on film thickness.
    // phaseDelta = 2 * thickness / math.cos(theta) * 2 * math.pi / wavelength_soap  
    // # There is an additional path to compute, the segment AJ from:
    // # https://www.glassner.com/wp-content/uploads/2014/04/CG-CGA-PDF-00-11-Soap-Bubbles-2-Nov00.pdf
    // phaseDelta -= 2 * thickness * math.tan(theta) * math.sin(incidentAngle) * 2 * math.pi / wavelength_air
    // Simplified to:
    float phaseDelta = 2.0 * thickness * nt * cost * 2.0 * PI / wavelength;
    
    // https://en.wikipedia.org/wiki/Reflection_phase_change
    if (ni < nt)
        phaseDelta -= PI;
    if (ni > nt)
        phaseDelta += PI;

    float front_refl_amp = Fresnel(cosi, cost, ni, nt);
    float front_trans_amp = 1.0 - front_refl_amp;
    float rear_refl_amp = front_trans_amp * Fresnel(cost, cosi, nt, ni);
    
    rear_refl_amp /= front_refl_amp;
    front_refl_amp = 1.0f;
        
    // http://scipp.ucsc.edu/~haber/ph5B/addsine.pdf
    return sqrt(front_refl_amp * front_refl_amp + rear_refl_amp * rear_refl_amp + 2.0 * front_refl_amp * rear_refl_amp * cos(phaseDelta));
}

// Function 92
float pixelIntensity(vec2 uv, vec2 d) {
	vec3 pix = texture(iChannel0, uv + d / iResolution.xy).rgb;
	return intensity(pix);
}

// Function 93
float intensityAtTime(float t)
{
    return fbm1D(t*3.)*0.5 + 0.5;
}

// Function 94
float volumetricShadow(in vec3 from, in float sundotrd ) {
    float dd = CLOUDS_SHADOW_MARGE_STEP_SIZE;
    vec3 rd = SUN_DIR;
    float d = dd * .5;
    float shadow = 1.0;

    for(int s=0; s<CLOUD_SELF_SHADOW_STEPS; s++) {
        vec3 pos = from + rd * d;
        float norY = (length(pos) - (EARTH_RADIUS + CLOUDS_BOTTOM)) * (1./(CLOUDS_TOP - CLOUDS_BOTTOM));

        if(norY > 1.) return shadow;

        float muE = cloudMap( pos, rd, norY );
        shadow *= exp(-muE * dd);

        dd *= CLOUDS_SHADOW_MARGE_STEP_MULTIPLY;
        d += dd;
    }
    return shadow;
}

// Function 95
void RayMarchVolumetric(in vec3 startingRayPos, in vec3 rayDir, inout SRayHitInfo hitInfo, out vec3 absorption, inout uint rngState, in vec2 fragCoord)
{
    float searchDistance = hitInfo.hitAnObject ? min(hitInfo.dist, c_maxDistanceVolumetric) : c_maxDistanceVolumetric;
    float stepSize = searchDistance / float(c_numStepsVolumetric);

    // random starting offset up to a step size for each ray, to make up for lower step count ray marching.
    float t = RandomFloat01(rngState) * stepSize;
    
    float scatterRoll = RandomFloat01(rngState);
    float scatterCum = 1.0f;
    absorption = vec3(1.0f, 1.0f, 1.0f);
    vec3 emissive = vec3(0.0f, 0.0f, 0.0f);

    SRayVolumetricInfo volumetricInfo;
    bool scattered = false;
    
    for (int i = 0; i < c_numStepsVolumetric; ++i)
    {
		vec3 rayPos = startingRayPos + rayDir * t;
        TestSceneVolumetric(rayPos, volumetricInfo);  // we could maybe try averaging the volumetricInfo with the last step or something.
        
        float desiredScatter = scatterRoll / scatterCum;  // this is how much we need to multiply scatterCum by to get to scatterRoll
        
        scatterCum *= exp(-volumetricInfo.scatterProbability * stepSize);               
        if (scatterCum < scatterRoll)
        {
            float lastT = t - stepSize;
            
            // using inverted beer's law to find the time between steps to get the right scatter amount.
            // beer's law is   y = e^(-p*x)
            // inverted, it is x = 1/p * ln(1/y)
            float stepT = (1.0f / volumetricInfo.scatterProbability) * log(1.0f / desiredScatter);
            t = lastT + stepT;
            
            // absorption and emission over distance
            absorption *= exp(-volumetricInfo.absorption * stepT);
            emissive += volumetricInfo.emissive * stepT;
            
            scattered = true;
            break;
        }
        
        // absorption and emission over distance
        absorption *= exp(-volumetricInfo.absorption * stepSize);       
        emissive += volumetricInfo.emissive * stepSize;
        
        // go to next ray position
        t += stepSize;
    }
    
    if (!scattered)
    {
        // emissive over distance should happen even when there's no scattering
        hitInfo.material.emissive += emissive;
        return;
    }
    
    hitInfo.hitAnObject = true;
    hitInfo.objectPass = OBJECTPASS_RAYMARCHVOLUMETRIC;
    hitInfo.dist = t;
    
    // importance sample Henyey Greenstein phase function to get the next ray direction and put it in the normal.
    // http://www.pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Sampling_Volume_Scattering.html
    // https://www.csie.ntu.edu.tw/~cyy/courses/rendering/09fall/lectures/handouts/chap17_volume_4up.pdf
    {
        float g = volumetricInfo.anisotropy;
        
        vec2 rand = vec2(RandomFloat01(rngState), RandomFloat01(rngState));
        
        float cosTheta;
		if (abs(g) < 1e-3)
    		cosTheta = 1.0f - 2.0f * rand.x;
		else
        {
    		float sqrTerm = (1.0f - g * g) /
                    		(1.0f - g + 2.0f * g * rand.x);
    		cosTheta = (1.0f + g * g - sqrTerm * sqrTerm) / (2.0f * g);
		}
        
        float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
		float phi = c_twopi * rand.y;
		vec3 v1, v2;
		CoordinateSystem(rayDir, v1, v2);
		hitInfo.normal = SphericalDirection(sinTheta, cosTheta, phi, v1, v2, -rayDir);
    }
        
    hitInfo.material.diffuse = vec3(0.0f, 0.0f, 0.0f);
    hitInfo.material.specular = vec3(0.0f, 0.0f, 0.0f);
    hitInfo.material.roughness = 0.0f;
    hitInfo.material.emissive = emissive;
}

// Function 96
float volume(float t, vec3 ray_o, vec3 ray_d, float t_refl, vec3 refl_d)
{
    // Uniform sampling on depth range
    float step_size = FOG_DEPTH / float(FOG_SAMPLES);
    // Offset uniform samples to get better coverage
    float frame_offset = step_size * rnd();
    // Beer's law
    // Transmittance is multiplicative so we can do this iteratively
    float step_transmittance = exp(-FOG_DENSITY * step_size);
    // This is not strictly correct when sampling a spotlight close to
    // the scene but it's faster than evaluating per step
	// Idea sparked by loicvdb's cool gi path tracing
	// https://www.shadertoy.com/view/Wt3XRX
    float step_color =
        (1. - step_transmittance) * henyeyGreenstein(-SPOT_D, -ray_d);

    float transmittance = 1.;
    float I = 0.;
    for (int i = 0; i < FOG_SAMPLES; ++i) {
        float s = float(i) * step_size + frame_offset;
        // Break on final hit
        if (s > t && s - t > t_refl)
            break;

        // Check if we should sample on the reflection ray
        vec3 p =
            s > t ? (ray_o + ray_d * t) + refl_d * (s - t) : ray_o + ray_d * s;

        transmittance *= step_transmittance;
        I += step_color * transmittance * evalSpot(p);
    }
    return I;
}

// Function 97
Ray marchVolume(Ray ray, Volume volume)
{   
    float t = sdf(ray.origin, volume);
    
    if(t > ray.t) return ray;
    
    const float MARCH_SIZE = 0.01;
    
    // vec3 lightColor=vec3(1.0,0.5,0.25);
    for (int i = 0; i < 50; ++i)
    {
        vec3 pos = ray.origin + (float(i) * MARCH_SIZE + t) * ray.direction;
        float sdf0 = sdf(pos, volume);
        if (sdf0 < 0.0)
        {
            // float lDist = length(pos - volume.center);
            ray.attenuation *= BeerLambert(volume.absorption * (
            texture(iChannel2, pos * 0.2).x
            + texture(iChannel2, pos * 0.4).x
            + texture(iChannel2, pos * 0.8).x
            + texture(iChannel2, pos * 1.6).x
            ) * 0.25 * abs(sdf0), MARCH_SIZE);
            // ray.attenuation += lightColor / (lDist*lDist)/5000.0;
        }
    }
    
    return ray;
}

// Function 98
float GetIntensityInRange(float beginNorm, float endNorm, float incrementSize)
{
    if (incrementSize == 0.0) return 0.0;
        
    float range = endNorm - beginNorm;
    float incrementCount = range / incrementSize;
    if (incrementCount == 0.0) return 0.0;
    
    float result = 0.0;
    
    for (int i = 0; i < int(incrementCount); i++) {
        float lookupCoord = beginNorm+float(i)*incrementSize;
        result += texture(iChannel1, vec2(lookupCoord, 0.25)).x * 2.0;
    }
    
    result = result / incrementCount;
    
    return result;
}

// Function 99
vec3 intensity(vec3 l, float t) {
    float u = atan2(l.y,l.x);
    float a = pow(cos(t-u),2.0)*length(l.xy) + l.z/2.0;
    return vec3(cos(t)*a,sin(t)*a,0.0);
}

// Function 100
float amplitude(float source_distance, float source_frequency, float source_phase)
{
    float wavelength = LIGHTSPEED / source_frequency;
    float x = (source_distance / wavelength) / (2.0 * PI);
	float y = sin(x + source_phase - iTime);
    return y;
}

