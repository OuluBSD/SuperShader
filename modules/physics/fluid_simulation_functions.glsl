// Reusable Fluid Simulation Physics Functions
// Automatically extracted from particle/physics simulation-related shaders

// Function 1
bool isIntersectingSmokeShape(vec3 u, float e, out float d)
{d=sampleSmokeCap(u);return d<e;}

// Function 2
vec3 smoke_color(float x)
{
    return vec3(.5, .5, .5) * x;
}

// Function 3
vec4 fluidSolver(sampler2D velocityField, vec2 uv, vec2 stepSize, vec4 mouse, vec4 prevMouse)
{
    float k = .2, s = k/dt;
    
    vec4 fluidData = textureLod(velocityField, uv, 0.);
    vec4 fr = textureLod(velocityField, uv + vec2(stepSize.x, 0.), 0.);
    vec4 fl = textureLod(velocityField, uv - vec2(stepSize.x, 0.), 0.);
    vec4 ft = textureLod(velocityField, uv + vec2(0., stepSize.y), 0.);
    vec4 fd = textureLod(velocityField, uv - vec2(0., stepSize.y), 0.);
    
    vec3 ddx = (fr - fl).xyz * .5;
    vec3 ddy = (ft - fd).xyz * .5;
    float divergence = ddx.x + ddy.y;
    vec2 densityDiff = vec2(ddx.z, ddy.z);
    
    // Solving for density
    fluidData.z -= dt*dot(vec3(densityDiff, divergence), fluidData.xyz);
    
    // Solving for velocity
    vec2 laplacian = fr.xy + fl.xy + ft.xy + fd.xy - 4.*fluidData.xy;
    vec2 viscosityForce = viscosityThreshold * laplacian;
    
    // Semi-lagrangian advection
    vec2 densityInvariance = s * densityDiff;
    vec2 was = uv - dt*fluidData.xy*stepSize;
    fluidData.xyw = textureLod(velocityField, was, 0.).xyw;
    
    // Calc external force from mouse input
    vec2 extForce = vec2(0.);
    
    if (mouse.z > 1. && prevMouse.z > 1.)
    {
        vec2 dragDir = clamp((mouse.xy - prevMouse.xy) * stepSize * 600., -10., 10.);
        vec2 p = uv - mouse.xy*stepSize;
        //extForce.xy += .75*.0002/(dot(p, p)+.0001) * (.5 - uv);
        extForce.xy += .001/(dot(p, p)) * dragDir;
    }
    
    fluidData.xy += dt*(viscosityForce - densityInvariance + extForce);
    
    // velocity decay
    fluidData.xy = max(vec2(0.), abs(fluidData.xy) - 5e-6)*sign(fluidData.xy);
    
    // Vorticity confinement
	fluidData.w = (fd.x - ft.x + fr.y - fl.y); // curl stored in the w channel
    vec2 vorticity = vec2(abs(ft.w) - abs(fd.w), abs(fl.w) - abs(fr.w));
    vorticity *= vorticityThreshold/(length(vorticity) + 1e-5)*fluidData.w;
    fluidData.xy += vorticity;

    // Boundary conditions
    fluidData.y *= smoothstep(.5,.48,abs(uv.y - .5));
    fluidData.x *= smoothstep(.5,.49,abs(uv.x - .5));
    
    // density stability
    fluidData = clamp(fluidData, vec4(vec2(-velocityThreshold), 0.5 , -velocityThreshold), vec4(vec2(velocityThreshold), 3.0 , velocityThreshold));
    
    return fluidData;
}

// Function 4
vec4 TexBulletSmokeImpact( vec2 vTexCoord, float fRandom, float fHRandom )
{
    vec4 vResult = vec4(0);
    
    vResult.rgb = vec3(1.0,1.0,1.0) * fRandom;
    
    float fLen = length( (vTexCoord - vec2(4.0, 8.0)) / vec2(6.0, 10.0) ) ;

    fRandom = 1.0 - fRandom;
    
    vResult.a = step(fLen + fRandom * fRandom * 2.0, 1.0);
    
    return vResult;    
}

// Function 5
vec4 primaryRayMarchSmoke(vec3 startPos, vec3 direction, vec3 lightDir) {
    vec3 position = startPos ;
    vec3 stepVector = direction * primSmokeSampleSize ;
    float dist ;
    float extinction = 1.0 ;
    vec3 colour = vec3(0.0) ;
    for (int i = 0 ; i < primSmokeNumSamples ; ++i) {
        if (extinction < 0.05 || !isIntersectingSmokeShape(position,0.005,dist))
            break ;
     	float vertDistFromRocket = abs(position.y - smokeEnd.y) ;
        float deltaYDensityMod = (1.f-(vertDistFromRocket)/(smokeEnd.y-smokeStart.y));
		float density = sampleSmoke(position) * deltaYDensityMod * deltaYDensityMod;
        extinction *= exp(-extinctionCoeff*density*primSmokeSampleSize);
        vec3 scattering = primSmokeSampleSize * density * scatteringCoeff * (ambientCol +  sunCol * getIncidentSunlight(position, lightDir)) ;
        colour += scattering * extinction ;
        position += stepVector ;
    }
    
    return vec4(colour,extinction) ;    
}

// Function 6
vec4 mapSmoke(in vec3 pos)
{
    vec3 pos2 = pos;
    pos2-= chimneyOrig + vec3(5.65, -0.8, 0.);
    
    // Calculating the smoke domain (3D space giving the probability to have smoke inside
    float sw = max(tubeDiam*0.84 + 0.25*pos2.y*(1. + max(0.15*pos2.y, 0.)) + 0.2*windIntensity*(pos.y + chimneyOrig.x - tubeclen - tubeLen2 + 0.3), 0.);
    float smokeDomain = smoothstep(1.2 + sw/4.3, 0.7 - sw*0.5, length(pos2.xz)/sw);
    
    float d;
    vec4 res;
    if (smokeDomain>0.1)
    {           
    	// Space modification in function of the time and wind
        vec3 q = pos2*vec3(1., 1. + 0.5*windIntensity, 1.) + vec3(0.0,-currTime*smokeSpeed + 10.,0.0);
    	q/= smokeScale;
        q.y+= 8.*dWindIntensity + 1.5/(0.7 + dWindIntensity);
        
        // Turbulence of the smoke
        #ifdef smoke_turbulence
        if (smokeTurbulence>0.)
        {
        	float n = smoothstep(4., 0., pos2.y + 3.2)*smokeTurbulence*noise(q*smokeTurbulenceScale)/(currTime + 3.);
        	q.xy = rotateVec(-q.xy, pos.z*n);
        	q.yz = rotateVec(-q.yz, pos.x*n);
        	q.zx = rotateVec(-q.zx, pos.y*n);
        }
        #endif
        
        // Calculation of the noise
        d = clamp(0.6000*noise(q), 0.4, 1.); q = q*2.02;  
        d+= 0.2500*noise(q); q = q*2.03;
        d+= 0.1200*noise(q); q = q*2.08;
        d+= 0.0500*noise(q);
        
        #ifdef heat_refraction
        // Calculation of the refraction due to the temperature difference in the air
        float rrf = smokeDomain*(1. - clamp((pos2.y + 2.8)*0.55, 0., 1.))*smoothstep(0., .3, pos2.y + 3.2);
        rayRef.x+= (smokeRefInt*noise(q*3.27 + q*4.12) - 0.5*smokeRefInt)*rrf;
        rayRef.y+= (smokeRefInt*noise(q*3.37 - q*3.96) - 0.5*smokeRefInt)*rrf;
        rayRef.z+= (smokeRefInt*noise(q*3.11 + q*3.82) - 0.5*smokeRefInt)*rrf;
        #endif

        d = d - 0.3 - smokeBias - 0.04*pos.y + 0.05*(1. + windIntensity);
        d = clamp(d, 0.0, 1.0);
        
 		res = vec4(pow(d*smokeDomain, smokePow));

    	// Some modifications of color and alpha
		res.xyz = mix(smokeCol, 0.2*vec3(0.4, 0.4, 0.4), res.x);
		res.xyz*= 0.2 + 0.2*smoothstep(-2.0, 1.0, pos.y);
    	res.w*= max(smokeDens - 1.8*sqrt(pos.y - 4.), 0.);
    }
    else
    {
        d = 0.;
        res = vec4(0.);
    }
	
	return res;
}

// Function 7
vec3 smoke (vec2 p, float t)
{
    float a = (1.0 - p.y) * cloud (p, t * 3.0);
    
    if (a < 0.3)
    {
        vec3 m = vec3 (0.0, 0.0, 0.0);
        vec3 n = vec3 (0.4, 0.2, 0.15);
        return mix (m, n, a / 0.3);
    }
    else if (a < 0.5)
    {
        vec3 m = vec3 (0.4, 0.2, 0.15);
        vec3 n = vec3 (0.4, 0.3, 0.25);
        return mix (m, n, (a - 0.3) / 0.2);
    }
    else
    {
        vec3 m = vec3 (0.4, 0.3, 0.25);
        vec3 n = vec3 (0.7, 0.5, 0.4);
        return mix (m, n, (a - 0.5) / 0.5);
    }
}

// Function 8
float sampleSmokeCap(vec3 position) {
	return sdCapsule(position,smokeStart,smokeEnd,smokeThickness) ;  
}

// Function 9
vec4 SmokeCol (vec3 ro, vec3 rd, vec3 col)
{
  vec3 clCol, tCol, q;
  float d, dens, atten, sh;
  clCol = vec3 (0.9);
  atten = 0.;
  d = 0.;
  for (int j = 0; j < 150; j ++) {
    q = ro + d * rd;
    dens = SmokeDens (q);
    sh = 0.5 + 0.5 * smoothstep (-0.2, 0.2, dens - SmokeDens (q + 0.1 * szFac * sunDir));
    tCol = mix (vec3 (1., 0.2, 0.), vec3 (0.6, 0.6, 0.), clamp (smoothstep (0.2, 0.8, dens) +
       0.2 * (1. - 2. * Noiseff (10. * tCur)), 0., 1.));
    tCol = mix (mix (tCol, clCol, smkPhs), clCol, smoothstep (-0.15, -0.05,
       (length (vec3 (q.xz * (1. - smkRadEx / length (q.xz)), q.y)) - smkRadIn) / szFac));
    col = mix (col, 4. * dens * tCol * sh, 0.2 * (1. - atten) * dens);
    atten += 0.12 * dens;
    d += szFac * 0.01;
    if (atten > 1. || d > dstFar) break;
  }
  atten *= smoothstep (0.02, 0.04, smkPhs);
  return vec4 (col, atten);
}

// Function 10
vec3 smoke(vec2 p, vec2 o, float t)
{
    const int steps = 10;
    vec3 col = vec3(0.0);
    for (int i = 1; i < steps; ++i)
    {
        //step along a random path that grows in size with time
        p += perlin(p + o) * t * 0.01 / float(i);
        p.y -= t * 0.003; //drift upwards
        
        //sample colour at each point, using mipmaps for blur
        col += texCol(p, float(steps-i) * t * 0.2);
    }
    return col.xyz / float(steps);
}

// Function 11
float smokeBase(vec2 pos)
{
    float v = clamp(pos.x * 1.5, -1.0, 1.0);
    return 1.0 - exp(-cos(v * PI * 0.5) * smoothstep(0.0, -1.0, pos.y) * 3.0);
}

// Function 12
void smoke_trail(float ltime, float duration, vec2 origin, vec2 fp) {
	vec2 var_4 = (vec2(sin(((ltime) + ((gl_FragCoord).x)) * 1187.0), sin(((ltime) + ((gl_FragCoord).y)) * 1447.0))) / 10.0;
	vec2 off = (var_4) * (((sin((((time) * 100.0) + ((var_4) * 211.0)) * ((cos(((gl_FragCoord).x) * 1489.0)) + (cos(((gl_FragCoord).y) * 1979.0))))) + 1.0) / 2.0);
	vec2 lp = closest_point_line(origin, fp, (op) - (off));
	float d = length((lp) - (op));
	float r = ((1.0 - ((length((lp) - (fp))) / (length((origin) - (fp))))) * ((clamp(length((lp) - (fp)), 0.0, 0.1)) * 30.0)) * (((duration) != 0.0) ? ((1.0 - (clamp(((time) - (ltime)) / (duration), 0.0, 1.0))) * 3.0) : 1.0);
	float gn = ((((sin((((time) * 100.0) + ((length(off)) * 211.0)) * ((cos(((gl_FragCoord).x) * 1489.0)) + (cos(((gl_FragCoord).y) * 1979.0))))) + 1.0) / 2.0) * 0.3) + 1.0;
	float grey = clamp((((clamp(0.1 - (d), 0.0, 1.0)) / 5.0) * (r)) * (gn), 0.0, 1.0);
	fragcolor = (vec3(grey, grey, grey)) + (fragcolor);
}

// Function 13
float smoke(vec2 pos)
{
    pos = swirls(pos);
    return smokeBase(pos);
}

// Function 14
vec2 fluid(vec2 uv1){
 vec2 uv = uv1;
 float t = iTime;
 for (float i = 1.; i < 15.; i++)
  {
    uv.x -= (t+sin(t+uv.y*i/1.5))/i;
    uv.y -= cos(uv.x*i/1.5)/i;
  }
  return uv;
}

// Function 15
float SmokeDens (vec3 p)
{
  mat2 rMat;
  vec3 q, u;
  float f;
  f = PrTorusDf (p.xzy, smkRadIn, smkRadEx);
  if (f < 0.) {
    q = p.xzy / smkRadEx;
    u = normalize (vec3 (q.xy, 0.));
    q -= u;
    rMat = mat2 (vec2 (u.x, - u.y), u.yx);
    q.xy = rMat * q.xy;
    q.xz = Rot2D (q.xz, 2.5 * tCur);
    q.xy = q.xy * rMat;
    q += u;
    q.xy = Rot2D (q.xy, 0.1 * tCur);
    f = smoothstep (0., smkRadIn, - f) * Fbm3 (10. * q);
  } else f = 0.;
  return f;
}

// Function 16
vec4 raymarchSmoke(in vec3 ro, in vec3 rd, in vec3 bcol, float tmax, bool isShadow)
{
	vec4 sum = vec4(0.0);
    vec2 windDir = rotateVec(vec2(1., 0.), windAngle);

	float t = isShadow?5.4:abs(0.95*(campos.z - chimneyOrig.z)/rd.z);
	for(int i=0; i<32; i++)
	{
		if(t>tmax || sum.w>1.) break;
		vec3 pos = ro + t*rd;
        
        // Influence of the wind
        pos.xz+= windDir*windIntensity*(pos.y + chimneyOrig.x - tubeclen - tubeLen2 + 0.3);

		vec4 col = mapSmoke(pos);
		
        if (col != vec4(0.))
        {
        	// Color modifications of the smoke
        	col.rgb+= (1. - smokeColPresence)*(1.0 - col.w);
			col.rgb = mix(col.rgb, bcol, smoothstep(smokeColBias, 0.0, col.w));
			col.rgb*= col.a;

			sum = sum + col*(1.0 - sum.a);
        }
		t+= 0.07*(1. + windIntensity)*max(0.1,0.05*t);
	}
	sum.rgb/= (0.001 + sum.a);
	return clamp(sum, 0.0, 1.0);
}

// Function 17
bool isIntersectingSmokeShape(vec3 position, float precis, out float dist) {
    dist = sampleSmokeCap(position) ;
    return dist < precis ;
}

// Function 18
vec4 smoke(vec2 pos){
    vec4 col = vec4(0.0);
    
    // Density
    float d = 0.0;
    
    pos.y += 0.08;
    
    if(pos.y > 0.0){
    	return col;
    }
    
    pos.x += 0.003 * cos(20.0 * pos.y + 4.0 * time * PI2);
    float dd = distance(pos,vec2(0.0,0.0));
    if(dd > 1.0){
    	pos *= 2.2 * pow(1.0 - dd, 2.0);
    }
    
    pos *= 1.9;
    
    d += cos(pos.x * 10.0);
	d += cos(pos.x * 20.0);
	d += cos(pos.x * 40.0);
	
    d += 0.3 * cos(pos.y * 6.0 + 8.0 * time * PI2) - 1.4;
	d += 0.3 * cos(pos.y * 50.0 + 4.0 * time * PI2) ;
	d += 0.3 * cos(pos.y * 10.0 + 2.0 * time * PI2);
    
    if(distance(pos.x, 0.0) < 0.05){
    	d *= 0.2 - distance(pos.x, 0.0);	
    } else {
    	d *= 0.0;
    }
    if( d < 0.0){
    	d = 0.0;
    }
    
    float dy = distance(pos.y, 0.0);
    
    if(dy < 0.3){
        float fac = 1.0 / 0.3 * dy;
    	col.r += 50.0 * pow(1.0 - fac,2.0) * d;
        col.g += 10.0 * pow(1.0 - fac,4.0) * d;
        col.a += 20.0 * (1.0 - fac) * d;
    }
    
    col.rgb += d * 10.0;
    col.a += d;
    
    return col;
}

// Function 19
vec3 Fluid(vec2 uv, float t) {
	float t1 = t*0.5;
	float t2 = t1 + 0.5;
	vec2 uv1 = calcNext(uv, t1);
	vec2 uv2 = calcNext(uv, t2);
	float c1 = getPattern(uv1);
	float c2 = getPattern(uv2);
	float c=mix(c2,c1,t);
    float f=1.5-0.5*abs(t-0.5);
	c=pow(c,f)*f;//correcting the contrast/brightness when sliding
	float h=mix(length(uv-uv2),length(uv-uv1),t);
	return 2.0*c*heatmap(clamp(h*0.5,0.0,1.0));//blue means slow, red = fast
}

// Function 20
float sampleSmokeCap(vec3 u)
{return sdCapsule(u,smokeStart,smokeEnd,smokeThickness);}

// Function 21
vec4 SmokeCol (vec3 ro, vec3 rd, float dstObj)
{
  vec4 col4;
  vec3 smkPos, p;
  float densFac, d, h, xLim;
  smkPos = vec3 (0., 0., 2. * canLen + smkPhs);
  smkPos.yz = Rot2D (smkPos.yz, - canEl);
  smkPos.xz = Rot2D (smkPos.xz, - canAz);
  smkPos.y += 1.4 * whlRad + 2. * bltThk + 0.58 * 1.1 * (0.8 * whlSpc - 1.5 * bltWid);
  smkRadIn = 0.005 + 0.045 * smoothstep (0.02, 0.15, smkPhs);
  smkRadEx = (2.5 + 3. * smoothstep (0.1, 0.4, smkPhs)) * smkRadIn;
  smkRadIn *= 1. - 0.3 * smoothstep (0.7, 1., smkPhs);
  d = 0.;
  for (int j = 0; j < 30; j ++) {
    p = ro + d * rd - smkPos;
    xLim = abs (p.x) - 1.5 * veGap;
    p.x = mod (p.x + 0.5 * veGap, veGap) - 0.5 * veGap;
    p.xz = Rot2D (p.xz, canAz);
    p.yz = Rot2D (p.yz, 0.5 * pi + canEl);
    h = max (PrTorusDf (p.xzy, smkRadIn, smkRadEx), xLim);
    d += h;
    if (h < 0.001 || d > dstFar) break;
  }
  col4 = vec4 (0.);
  if (d < min (dstObj, dstFar)) {
    densFac = 1.5 * max (1.1 - pow (smkPhs, 1.5), 0.);
    for (int j = 0; j < 16; j ++) {
      p = ro + d * rd - smkPos;
      p.x = mod (p.x + 0.5 * veGap, veGap) - 0.5 * veGap;
      p.xz = Rot2D (p.xz, canAz);
      p.yz = Rot2D (p.yz, 0.5 * pi + canEl);
      col4 += densFac * SmokeDens (p) * (1. - col4.w) * vec4 (vec3 (0.9) - col4.rgb, 0.1);
      d += 2.2 * smkRadIn / 16.;
      if (col4.w > 0.99 || d > dstFar) break;
    }
  }
  return col4;
}

// Function 22
vec4 calcSmoke(vec3 u,vec3 t,vec3 s)
{t*=RmSmokSampleSize
;float d,a=1.;vec3 c=vec3(0)
;for(int i=0;i<iterRmSmoke;++i
){if(a<.05||!isIntersectingSmokeShape(u,.005,d))break
 ;float g=sampleSmoke(u)
 ;a*=exp(-oExting*g*RmSmokSampleSize)
 ;c+=a*RmSmokSampleSize*g*oScat*(cAmb+cSun*getIncidentSunlight(u,s));
 ;u+=t;}return vec4(c,a);}

// Function 23
vec4 SmokeCol (vec3 ro, vec3 rd, float dstObj)
{
  vec4 col4;
  vec3 q;
  float densFac, dens, d, h, sh;
  d = 0.;
  for (int j = 0; j < 150; j ++) {
    q = ro + d * rd;
    q.xz = abs (q.xz);
    q -= smkPos;
    h = PrTorusDf (q.xzy, smkRadIn, smkRadEx);
    d += h;
    if (h < 0.001 || d > dstFar) break;
  }
  col4 = vec4 (0.);
  if (d < min (dstObj, dstFar)) {
    densFac = 1.45 * (1.08 - pow (smkPhs, 1.5));
    for (int j = 0; j < 150; j ++) {
      q = ro + d * rd;
      q.xz = abs (q.xz);
      q -= smkPos;
      dens = SmokeDens (q);
      sh = 0.3 + 0.7 * smoothstep (-0.3, 0.1, dens - SmokeDens (q + 0.1 * sunDir));
      col4 += densFac * dens * (1. - col4.w) * vec4 (sh * vec3 (0.9) - col4.rgb, 0.3);
      d += 2.2 * smkRadEx / 150.;
      if (col4.w > 0.99 || d > dstFar) break;
    }
  }
  if (isNite) col4.rgb *= vec3 (0.3, 0.4, 0.3);
  return col4;
}

// Function 24
float smoke(vec2 uv,float param)
{
   float frac = 1.0;
    vec2 z = vec2(0.0,0.0);
    vec2 c = uv;
    
    float timefact = cos(iTime*0.25+param*0.001);
    vec2 constant = (powC(cos(c),vec2(33.8-param*0.01,10.8)));
    for(int i=0; i<20; i++)
    { 
        //this is like the mandelbrot set function but with a bunch of random modifications
        //I think lots of the bits can be removed and it won't look much different!
        z = cmult(sin(z-c),cos(z*z + vec2(param))-c) + timefact*1.0*z 
            + constant;
        
        if(length(z)>2.0)
        {  
            frac = float(i)/20.0;
            break;
        }
    }
    
    return frac;
}

// Function 25
vec3 fluidPos(vec3 p)
{
    float lmin=min(FRes.x,min(FRes.y,FRes.z));
    return p*lmin*.5+FRes*.5;
}

// Function 26
vec3 smokeNormal(vec3 p) {
    vec2 NE = vec2(2., 0.);
    #define s(c) length(texture(iChannel0,w2t(c)).xyz*2.-1.)
	return normalize(vec3( s(p+NE.xyy)-s(p-NE.xyy),
                          s(p+NE.yxy)-s(p-NE.yxy),
                          s(p+NE.yyx)-s(p-NE.yyx) ));
}

// Function 27
vec3 fluid(vec3 uv1,float iters){
 //fake fluid physics
 vec3 uv = uv1;
 for (float i = 1.; i < iters; i++)
  {
    uv.x += sin((iTime-uv.y)*.5)*1.5/i* sin(i * uv.y + iTime * 0.5);
    uv.y += sin((iTime-uv.z)*.5)*1.5/i* sin(i * uv.z + iTime * 0.5 );
    uv.z += sin((iTime-uv.x)*.5)*1.5/i* sin(i * uv.x + iTime * 0.5 );
  }
  return uv;
}

// Function 28
float SmokeParticle(vec2 loc, vec2 pos, float size, float rnd)
{
	loc = loc-pos;
	float d = dot(loc, loc)/size;
	// Outside the circle? No influence...
	if (d > 1.0) return 0.0;

	// Rotate the particles...
	float r= time*rnd*1.85;
	float si = sin(r);
	float co = cos(r);
	// Grab the rotated noise decreasing resolution due to Y position.
	// Also used 'rnd' as an additional noise changer.
	d = noise(hash(rnd*828.0)*83.1+mat2(co, si, -si, co)*loc.xy*2./(pos.y*.16)) * pow((1.-d), 3.)*.7;
	return d;
}

// Function 29
float SmokeDens (vec3 p)
{
  mat2 rMat;
  vec3 q, u;
  p = p.xzy;
  q = p / smkRadEx;
  u = normalize (vec3 (q.xy, 0.));
  q -= u;
  rMat = mat2 (vec2 (u.x, - u.y), u.yx);
  q.xy = rMat * q.xy;
  q.xz = Rot2D (q.xz, 2. * tCur);
  q.xy = q.xy * rMat;
  q += u;
  q.xy = Rot2D (q.xy, 0.1 * tCur);
  return clamp (smoothstep (0., 1., densFac * PrTorusDf (p, smkRadIn, smkRadEx)) *
     Fbm3 (5. * q + 0.01 * tCur) - 0.1, 0., 1.);
}

// Function 30
vec2 fluidnoise(vec3 p) {
    vec2 total = vec2(0);
    float amp = 1.;
    for(int i = 0; i < 1; i++) {
        total += noise(p) * amp;
        p = p*2. + 4.3; amp *= 1.5;
    }
    return total.yx * vec2(-1,1); // divergence-free field
}

// Function 31
float sampleSmoke(vec3 u)
{float n=0.,a=1.,f=3.,l=2.
;for (int i=0;i<octavesSmoke;++i
){n+=a*noise(f*u+vec3(.0,tim*200.,3.*tim));a/=l;f*=l;}
;vec3 d=normalize(smokeEnd-smokeStart)
;u=smokeStart-u;
;return 2.*sat((n*exp(-2.5*length(u-(dot(u,d))*d))-smokeOffset)
                *(1.-exp(-.05*length(u))));}

// Function 32
float SmokeDens (vec3 p)
{
  mat2 rMat;
  vec3 q, u;
  float f;
  f = PrTorusDf (p.xzy, smkRadIn, smkRadEx);
  if (f < 0.) {
    q = p.xzy / smkRadEx;
    u = normalize (vec3 (q.xy, 0.));
    q -= u;
    rMat = mat2 (vec2 (u.x, - u.y), u.yx);
    q.xy = rMat * q.xy;
    q.xz = Rot2D (q.xz, 2.5 * tCur);
    q.xy = q.xy * rMat;
    q += u;
    q.xy = Rot2D (q.xy, 0.2 * tCur);
    f = smoothstep (0., smkRadIn, - f) * Fbm3 (16. * q);
  } else f = 0.;
  return f;
}

// Function 33
float SmokeShellDist (vec3 ro, vec3 rd)
{
  vec3 q;
  float d, h;
  d = 0.;
  for (int j = 0; j < 150; j ++) {
    q = ro + d * rd;
    h = PrTorusDf (q.xzy, smkRadIn, smkRadEx);
    d += h;
    if (h < 0.001 || d > dstFar) break;
  }
  return d;
}

// Function 34
vec4 getSmoke( in vec2 fragCoord )
	{
	  
		vec2 uv =  fragCoord.xy/iResolution.x;
		   
		float cloud = clouds(uv);
		
		return vec4(cloud,cloud,cloud,1.0);

	}

// Function 35
float sampleSmoke(vec3 position) {
  float noiseVal = 0.0 ;
  float amplitude = 1.0 ;
  float freq = 4.5 ;
  float lac = 2.0 ;
  float scaling = 2.0 ;
  for (int i = 0 ; i < octavesSmoke ; ++i) {
    noiseVal += amplitude * noise(freq*position+vec3(0.0,time*200.0,3.0*time)) ;
    amplitude /= lac ;
    freq *= lac ;
  }
    
  vec3 smokeDir = normalize(smokeEnd-smokeStart) ;
  float dist = length((smokeStart - position) - (dot((smokeStart - position),smokeDir))*smokeDir) ;
  noiseVal *= exp(-2.5*dist) ;
  noiseVal -= offset ;
  noiseVal *= (1.0 - exp(-0.05 * length(smokeStart-position))) ;
  noiseVal = clamp(noiseVal,0.0,1.0) ;

  return scaling * noiseVal ;
}

// Function 36
vec4 solveFluid(sampler2D smp, vec2 uv, vec2 w, float time, vec3 mouse, vec3 lastMouse)
{
	const float K = 0.2;
	const float v = 0.55;
    
    vec4 data = textureLod(smp, uv, 0.0);
    vec4 tr = textureLod(smp, uv + vec2(w.x , 0), 0.0);
    vec4 tl = textureLod(smp, uv - vec2(w.x , 0), 0.0);
    vec4 tu = textureLod(smp, uv + vec2(0 , w.y), 0.0);
    vec4 td = textureLod(smp, uv - vec2(0 , w.y), 0.0);
    
    vec3 dx = (tr.xyz - tl.xyz)*0.5;
    vec3 dy = (tu.xyz - td.xyz)*0.5;
    vec2 densDif = vec2(dx.z ,dy.z);
    
    data.z -= dt*dot(vec3(densDif, dx.x + dy.y) ,data.xyz); //density
    vec2 laplacian = tu.xy + td.xy + tr.xy + tl.xy - 4.0*data.xy;
    vec2 viscForce = vec2(v)*laplacian;
    data.xyw = textureLod(smp, uv - dt*data.xy*w, 0.).xyw; //advection
    
    vec2 newForce = vec2(0);
    #ifndef MOUSE_ONLY
    #if 1
    newForce.xy += 0.75*vec2(.0003, 0.00015)/(mag2(uv-point1(time))+0.0001);
    newForce.xy -= 0.75*vec2(.0003, 0.00015)/(mag2(uv-point2(time))+0.0001);
    #else
    newForce.xy += 0.9*vec2(.0003, 0.00015)/(mag2(uv-point1(time))+0.0002);
    newForce.xy -= 0.9*vec2(.0003, 0.00015)/(mag2(uv-point2(time))+0.0002);
    #endif
    #endif
    
    if (mouse.z > 1. && lastMouse.z > 1.)
    {
        vec2 vv = clamp(vec2(mouse.xy*w - lastMouse.xy*w)*400., -6., 6.);
        newForce.xy += .001/(mag2(uv - mouse.xy*w)+0.001)*vv;
    }
    
    data.xy += dt*(viscForce.xy - K/dt*densDif + newForce); //update velocity
    data.xy = max(vec2(0), abs(data.xy)-1e-4)*sign(data.xy); //linear velocity decay
    
    #ifdef USE_VORTICITY_CONFINEMENT
   	data.w = (tr.y - tl.y - tu.x + td.x);
    vec2 vort = vec2(abs(tu.w) - abs(td.w), abs(tl.w) - abs(tr.w));
    vort *= VORTICITY_AMOUNT/length(vort + 1e-9)*data.w;
    data.xy += vort;
    #endif
    
    data.y *= smoothstep(.5,.48,abs(uv.y-0.5)); //Boundaries
    
    data = clamp(data, vec4(vec2(-10), 0.5 , -10.), vec4(vec2(10), 3.0 , 10.));
    
    return data;
}

