// Reusable Effects UI/2D Functions
// Automatically extracted from UI/2D graphics-related shaders

// Function 1
float realSoftShadow( in vec3 ro, in vec3 rd, in float tmin, in float tmax, float w )
{
    vec3 uu = normalize(cross(rd,vec3(0,1,0)));
    vec3 vv = normalize(cross(rd,uu));
    
    float tot = 0.0;
    const int num = 32; // cast 32 rays
	for( int j=0; j<num; j++ )
    {
        // uniform distribution on an disk
        float ra = sqrt(rand());
        float an = 6.283185*rand();
        vec3 jrd = rd + w*ra*(uu*cos(an)+vv*sin(an));
        
        // raycast
        float res = 1.0;
        
        for( int i=0; i<7; i++ ) // 7 objects
        {
            int k = i % 3;
            bool sha = false;
                 if(k==0) sha = shadowBox( ro, jrd, vec3(-4.0 + float(i),0.25,0.0), vec3(0.2,0.5,0.2), tmax);
            else if(k==1) sha = shadowSphere(ro, jrd, vec3(-4.0 + float(i),0.3,0.0), 0.4, tmax);
            else          sha = shadowCylinder( ro - vec3(-4.0 + float(i),0.0,0.0), jrd, 0.8, 0.3, tmax);
            
            if( sha ) { res=0.0; break; }
        }
        
        
        tot += res;
    }
    return tot/float(num);
}

// Function 2
float softShadow(vec3 ro, vec3 rd, float tmin, float tmax)
{
	float res = 1.0;
    float t = tmin;
    for(int i = 0; i < 16; i++)
    {
		float h = getDist(ro + rd*t).dist;
        res = min(res, 1.0 * h/t);
        t += clamp(h, 0.02, .50);
        
        if (h < 0.001 || t > tmax)
            break;
    }
    return clamp(res, 0.0, 1.0);
}

// Function 3
float shadow(in vec3 rp)
{
	float d = 0.05;
	AA aa;
    float h = map(rp + normalize(vec3(0.0, .0, 1.0)) * d, aa, false);
    return clamp(h / d, 0.0, 1.0);
}

// Function 4
float sdf_shadow(float sdf, float size, vec2 light_dir)
{
    vec2 n = sdf_normal(sdf);
    float thresh = size * max(abs(dFdx(sdf)), abs(dFdy(sdf)));
    float mask = clamp(sdf/thresh, 0., 1.);
    return clamp(1. - sdf/size, 0., 1.) * clamp(-dot(light_dir, n), 0., 1.) * mask;
}

// Function 5
float shadowcast_pointlight(in vec3 ro, in vec3 rd, in float light_dist){
    float res = 1.f;
    float t = 0.001f;
    for (int i = 0; i < RAY_STEPS; i++){
        vec3 pos = ro + rd * t;
        float dist = map(pos).x;
        if (res < 0.0000001){
            break;
        }
        if (t > light_dist){
            return res;
        }
        if (dist > FAR_CLIP){
            break;
        }
        res = min(res, 10.f * dist/t);
        t += dist;
    }
    return res;
}

// Function 6
vec2 neighbourEffect(ivec2 p)
{
    vec2 xyN = texelFetch(iChannel0, p+ivec2(1,  1), 0).xy;
    vec2 xyS = texelFetch(iChannel0, p+ivec2(1, -1), 0).xy;
    vec2 xyE = texelFetch(iChannel0, p+ivec2(1, -1), 0).xy;
    vec2 xyW = texelFetch(iChannel0, p+ivec2(-1,-1), 0).xy;
    vec2 add = xyN+xyS+xyE+xyW;
    return add*.002;
}

// Function 7
float soft_shadow(vec3 ro, vec3 rd) {
    float res = 1.;
    float t = .0001;                     
	float h = 1.;
    for(int i = 0; i <20; i++) {         
        h = eval_scene(ro + rd*t).d;
        res = min(res, 4.*h/t);          
		t += clamp(h, .02, 1.);          
    }
    return clamp(res, 0., 1.);
}

// Function 8
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 32; 
    
    ro += n*.0015;
    vec3 rd = lp - ro; // Unnormalized direction ray.
    

    float shade = 1.;
    float t = 0.;//.0015; // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), .0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = min(iFrame, 0); i<maxIterationsShad; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        t += clamp(d, .05, .5); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}

// Function 9
float shadow_march(vec4 pos, vec4 dir, float distance2light, float light_angle, inout vtx co)
{
	float light_visibility = 1.;
	float ph = 1e5;
    float td = dir.w;
	pos.w = map(pos.xyz, dir.xyz, co);
	for (int i = min(0, iFrame); i < 20; i++) 
    {
		dir.w += pos.w;
		pos.xyz += pos.w*dir.xyz;
		pos.w = map(pos.xyz, dir.xyz, co);
		float y = pos.w*pos.w/(2.0*ph);
        float d = (pos.w+ph)*0.5;
		float angle = d/(max(0.00001,dir.w-y-td)*light_angle);
        light_visibility = min(light_visibility, angle);
		ph = pos.w;
		if(dir.w >= distance2light) break;
		if(dir.w > maxd || pos.w < mind*dir.w) return 0.;
    }
	return 0.5 - 0.5*cos(PI*light_visibility);
}

// Function 10
float marchShadow(vec3 ro, vec3 rd)
{
 	float d;
    float sd = 1.0;
    
    float transmittance = 1.0;
    float t = 0.01;
    
    for(uint i = NON_CONST_ZERO_U; i < ITER_SHADOW; ++i)
    {
        vec3 posWS = ro + rd*t;
        float foamAmount = 0.0, taaStrength = 0.0;
        uint materialId = 999u;
        d = fSDF(posWS, kFoamPartId | kWaveFoamPartId | kRockPartId, materialId, 
                 foamAmount, taaStrength);
        d -= t * 0.001;
        
        float stepD = max(0.02+float(i)*0.05, abs(d));
        
        float stepTransmittance = 1.0;
        if(materialId == kFoamMatId)
        {
            stepTransmittance = exp(-2.0*stepD*foamAmount);
        }
        else
        {
            stepTransmittance = 0.0;
        }
        
        float coneWidth = max(0.05, t * 1.0/40.0);
        stepTransmittance = mix(stepTransmittance, 1.0, saturate(d/coneWidth));
        transmittance *= stepTransmittance;
        
        t += stepD;
        
        if(transmittance < 0.05)
        {
            break;
        }
    }
      
    return transmittance;
}

// Function 11
float ObjSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  vec2 gIdP;
  float sh, d, h;
  sh = 1.;
  gIdP = vec2 (-99.);
  d = 0.01;
  for (int j = 0; j < 30; j ++) {
    p = ro + rd * d;
    gId = PixToHex (p.xz);
    if (gId.x != gIdP.x || gId.y != gIdP.y) {
      gIdP = gId;
      SetTrParms ();
    }
    h = ObjDf (p);
    sh = min (sh, smoothstep (0., 0.1 * d, h));
    d += clamp (h, 0.05, 0.5);
    if (sh < 0.05) break;
  }
  return 0.3 + 0.7 * sh;
}

// Function 12
float ObjSShadow (vec3 ro, vec3 rd, float dMax)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = VAR_ZERO; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05 || d > dMax) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 13
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

// Function 14
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for(int i=0; i<50; i++)
    {
    	float h = map(ro + rd*t, false).x;
        res = min(res, 10.0*h/t + 0.02*float(i));
        t += 0.8*clamp(h, 0.01, 0.35);
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 15
float GetShadow(vec3 distances)
{
    distances = clamp((distances+0.5)*5.0,0.0,1.0);
    float shadowAtten = distances.r * 0.33 + distances.g * 0.33 + distances.b * 0.33;
    
    return shadowAtten;
}

// Function 16
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

// Function 17
float castShadowRay( in vec3 ro, in vec3 rd )
{
	vec2 pos = floor(ro.xz);
	vec2 ri = 1.0/rd.xz;
	vec2 rs = sign(rd.xz);
	vec2 ris = ri*rs;
	vec2 dis = (pos-ro.xz+ 0.5 + rs*0.5) * ri;
	float t = -1.0;
	float res = 1.0;
	
    // first step we check noching	
	vec2 mm = step( dis.xy, dis.yx ); 
	dis += mm * ris;
    pos += mm * rs;
	
    // traverse regular grid (2D)	
	for( int i=0; i<16; i++ ) 
	{
		float ma = map(pos);
		
        // test capped cylinder		
		vec3  ce = vec3( pos.x+0.5, 0.0, pos.y+0.5 );
		vec3  rc = ro - ce;
		float a = dot( rd.xz, rd.xz );
		float b = dot( rc.xz, rd.xz );
		float c = dot( rc.xz, rc.xz ) - 0.249;
		float h = b*b - a*c;
		if( h>=0.0 )
		{
			float t = (-b - sqrt( h ))/a;
			if( (ro.y+t*rd.y)<ma )
			{
				res = 0.0;
    			break; 
			}
		}
		mm = step( dis.xy, dis.yx ); 
		dis += mm * ris;
        pos += mm * rs;
	}

	return res;
}

// Function 18
float shadows(in vec3 ro, in vec3 rd, in float start, in float end, in float k){

    float shade = 33.3;
    const int shadIter = 5; 

    float dist = start;
    //float stepDist = end/float(shadIter);

    for (int i=0; i<shadIter; i++){
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.

        dist += clamp(h, 1.32, 1.2);
        
        // There's some accuracy loss involved, but early exits from accumulative distance function can help.
        if ((h)<0.001 || dist > end) break; 
    }
    
    return min(max(shade, 0.3) + 1.4, 1.0); 
}

// Function 19
float shadow_march(vec4 pos, vec4 dir, float distance2light, float light_angle)
{
	float light_visibility = 1.;
	float ph = 1e5;
	float dDEdt = 0.;
	pos.w = map(pos.xyz);
	int i = 0;
	for (; i < shadow_steps; i++) {
	
		dir.w += pos.w;
		pos.xyz += pos.w*dir.xyz;
        vec3 ra =rand3()-0.5;
        
		pos.w = (1. + 0.1*ra.x)*abs(map(pos.xyz));
        dir.xyz = normalize(dir.xyz + 0.01*pos.w*ra/2.5*rayfov*dir.w);
	
		float angle = max((pos.w - 2.5*rayfov*dir.w)/(max(0.0001,dir.w)*light_angle), 0.);
		
        light_visibility = min(light_visibility, angle);
		
		ph = pos.w;
		
        if(dir.w >= distance2light)
		{
			break;
		}
		
		if(dir.w > MAX_DIST || pos.w < max(2.*rayfov*dir.w, MIN_DIST))
		{
			break;
		}
	}
	
	if(i >= shadow_steps)
	{
		light_visibility=0.;
	}
	//return light_visibility; //bad
	light_visibility = clamp(2.*light_visibility - 1.,-1.,1.);
	return  0.5 + (light_visibility*sqrt(1.-light_visibility*light_visibility) + asin(light_visibility))/3.14159265; //looks better and is more physically accurate(for a circular light source)
}

// Function 20
float softShadow(vec3 ro, vec3 rd)
{
    const int ITERS = 30;

    float nearest = 1.0;
    
    float t = 0.1;
    for(int i = 0; i < ITERS; i++)
    {
        vec3 p = ro + rd * t;
        float d = sdWorld(p).x;
        
        float od = d / t;
        
        if(od < nearest) {
            nearest = od;
        }
        if(d <= EPSILON) {
			return 0.0;
        }
        if(d >= INFINITY) {
            break;
        }
        
        t += min(0.5, max(EPSILON, d));
    }
    
    return nearest;
}

// Function 21
float shadow(vec3 lig, vec3 cam){ 
    float l = distance(cam,lig);
    vec3 dir = normalize(lig-cam);
    for(float i = 0.3; i < l;){ //i = length of ray
        vec3 p = cam + i * dir;
    	float h = map(p).Md; //smallest distance from all objects to point
    	if(h < MinDist){
			return(0.0); //successfully hit something at point "point", make dark
        }
        i += h;
    }
    return(1.0); //no collision, make light
}

// Function 22
float softshadow(in vec3 ro, in vec3 rd, float mint, float maxt, float k)
{
    float res = 1.0;
    float dt = 0.1;
    float t = mint;
    for( int i = 0; i < 30; i++){
    	float h = map(ro +rd*t).x;
    	h = max (h , 0.0);
        res = min(res, smoothstep(0.0, 1.0, k*h/t) );
        t += dt; 
        if(h <0.001) break;
    
    }
	return res;

}

// Function 23
float shadow(vec3 ro, vec3 rd){
    float t = 0.01;
    float d = 0.0;
    float shadow = 1.0;
    for(int iter = 0; iter < 128; iter++){
        d = map(ro + rd * t);
        if(d < 0.0001){
            return 0.0;
        }
        if(t > length(ro - lightPos) - 0.5){
            break;
        }
        shadow = min(shadow, 128.0 * d / t);
        t += d;
    }
    return shadow;
}

// Function 24
float TreesSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  vec2 gIdP;
  float sh, d, h;
  sh = 1.;
  gIdP = vec2 (-99.);
  d = 0.01;
  for (int j = 0; j < 20; j ++) {
    p = ro + d * rd;
    gId = PixToHex (p.xz / hgSize);
    if (gId.x != gIdP.x || gId.y != gIdP.y) {
      gIdP = gId;
      SetTrParms ();
    }
    h = TreesDf (p);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += clamp (3. * h, 0.1, 0.2);
    if (sh < 0.05) break;
  }
  return 0.6 + 0.4 * sh;
}

// Function 25
float softShadowSphere( in vec3 ro, in vec3 rd, in vec4 sph )
{
    vec3 oc = sph.xyz - ro;
    float b = dot( oc, rd );
	
    float res = 1.0;
    if( b>0.0 )
    {
        float h = dot(oc,oc) - b*b - sph.w*sph.w;
        res = smoothstep( 0.0, 1.0, 2.0*h/b );
    }
    return res;
}

// Function 26
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
   float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
      float h = renderFunction( ro + rd*t );
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 27
float corner_effect(vec3 pos, vec3 nor, bool inside)
{
    float scale = 0.05;
    
    return min(1.,0.2 + 0.8 *abs(shadow_sdf(pos - scale * nor, inside)) / scale);
}

// Function 28
float shadow(vec3 ro, vec3 rd)
{
    float d = rayIntersect(ro, rd, PRECISION_FACTOR_SHADOW, MIN_DIST_SHADOW, MAX_DIST_SHADOW);
    return (d>0.) ? smoothstep(0., MAX_DIST_SHADOW, d) : 1.;
}

// Function 29
float shadowMarch(vec3 ro, vec3 rd, vec3 lightPos)
{
    float dist = 0.0;
    
    
    float res = 1.0;
    
    float power = 2.0;
    
    for(int i=0; i<MAX_STEPS; i++)
    {
        vec3 pos = ro + dist * rd;
        float d = getDist(pos).x;
        if (d < 0.01)
            return 0.0;
        
		dist += d;
        
        res = min(res, power * d / dist);
        
		if (dist<MIN_DIST || dist>MAX_DIST)
            break;
    }

    return res;
}

// Function 30
float blur(in vec2 p){
    
    // Used to move to adjoining pixels. - uv + vec2(-1, 1)*px, uv + vec2(1, 0)*px, etc.
    vec3 e = vec3(1, 0, -1);
    vec2 px = 1./iResolution.xy;
    
    // Weighted 3x3 blur, or a cheap and nasty Gaussian blur approximation.
	float res = 0.0;
    // Four corners. Those receive the least weight.
	res += tx(p + e.xx*px ).x + tx(p + e.xz*px ).x + tx(p + e.zx*px ).x + tx(p + e.zz*px ).x;
    // Four sides, which are given a little more weight.
    res += (tx(p + e.xy*px ).x + tx(p + e.yx*px ).x + tx(p + e.yz*px ).x + tx(p + e.zy*px ).x)*2.;
	// The center pixel, which we're giving the most weight to, as you'd expect.
	res += tx(p + e.yy*px ).x*4.;
    // Normalizing.
    return res/16.;     
    
}

// Function 31
float ObjSShadow (vec3 ro, vec3 rd, float dMax)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += 0.3;
    if (sh < 0.05 || d > dMax) break;
  }
  return sh;
}

// Function 32
float calcSoftshadow( in vec3 ro, in vec3 rd )
{
    float tmin = 0.001;
    float tmax = 8.0;

    float res = 1.0;
    
    // bounding sphere
    vec2 bs = iSphere(ro,rd,vec4(0.0,0.0,0.0,sqrt(3.0)+kRoundness+0.2));
    if( bs.y>0.0 )
    {
        tmin = max(tmin,bs.x); // clip search space
        tmax = min(tmax,bs.y); // clip search space
        
        float t = tmin;
        for( int i=0; i<64; i++ )
        {
            float h = map( ro + rd*t ).x;
            float s = clamp(8.0*h/t,0.0,1.0);
            res = min( res, s*s*(3.0-2.0*s) );
            t += clamp( h, 0.02, 0.5 );
            if( res<0.005 || t>tmax ) break;
        }
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 33
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax ) {
    // bounding volume
    float tp = (0.8-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

    float res = 1.0;
    float t = mint;
    for( int i=ZERO; i<24; i++ )
    {
		float h = map( ro + rd*t ).x;
        float s = clamp(8.0*h/t,0.0,1.0);
        res = min( res, s*s*(3.0-2.0*s) );
        t += clamp( h, 0.02, 0.2 );
        if( res<0.004 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 34
float traceShadow(vec3 ro, vec3 rd, float dist, int steps)
{
    float rad = 1.5;
    vec3 p = ro;
    float acc = 1.;//rad;//rad*50.;
    for (int i = 0; i < steps && distance(p, ro) < dist; ++i)
    {
        vec2 res = map(p);
        if (res.x < 0.01)
        {
            return 0.;
        }
        float d =min(res.x,1.5);
        acc = min(acc, 30.*d/distance(p, ro));
        p += rd * d;
         // check this https://www.shadertoy.com/view/3tVBRV
        //acc += sat(d/rad*dist*.005);
    }
    return acc;
}

// Function 35
vec3 effect(vec3 p)
{
	return vec3(min(min(func((p*mx).yz), func((p*my).xz)), func((p*mz).xy))/0.57);
}

// Function 36
vec4 pulseDrawShadows(vec2 coord, float time){
    time = mod(time, PULSE_TIMES);
    
    vec4 r = vec4(0, 0, 0, 1.0);
    
    // Pulse variables
    float x = time / PULSE_TIMES, y;// = PULSE_FUNC(x * PI * PULSE_STEP);
    float px = coord.x, py; // = PULSE_FUNC(coord.x * PI * PULSE_STEP);
    vec2 pv, sx, sy;
    
#ifdef PULSE_CALL_GREEN
    if(r == vec4(0, 0, 0, 1.0)){
    	PULSE_FUNC_CALL(PULSE_CALL_GREEN);
        r = pulseDrawShadowsFunc(coord, time, pv, sx, sy, RGBC_GREEN);
    }
#endif
    
#ifdef PULSE_CALL_GREEN
    if(r == vec4(0, 0, 0, 1.0)){
    	PULSE_FUNC_CALL(PULSE_CALL_GREEN);
        r = pulseDrawShadowsFunc(coord, time, pv, sx, sy, RGBC_GREEN);
    }
#endif
    
#ifdef PULSE_CALL_BLUE
    if(r == vec4(0, 0, 0, 1.0)){
    	PULSE_FUNC_CALL(PULSE_CALL_BLUE);
        r = pulseDrawShadowsFunc(coord, time, pv, sx, sy, RGBC_BLUE);
    }
#endif  

#ifdef PULSE_CALL_RED
    if(r == vec4(0, 0, 0, 1.0)){
    	PULSE_FUNC_CALL(PULSE_CALL_RED);
        r = pulseDrawShadowsFunc(coord, time, pv, sx, sy, RGBC_RED);
    }
#endif 
    
#ifdef PULSE_CALL_ORANGE
    if(r == vec4(0, 0, 0, 1.0)){
    	PULSE_FUNC_CALL(PULSE_CALL_ORANGE);
        r = pulseDrawShadowsFunc(coord, time, pv, sx, sy, RGBC_ORANGE);
    }
#endif   
    
#ifdef PULSE_CALL_PURPLE
    if(r == vec4(0, 0, 0, 1.0)){
    	PULSE_FUNC_CALL(PULSE_CALL_PURPLE);
        r = pulseDrawShadowsFunc(coord, time, pv, sx, sy, RGBC_PURPLE);
    }
#endif
    
    return r;
}

// Function 37
float softShadow(vec3 ro, vec3 rd, float k, float mind, float maxd) {
    int id;
    float totalDist = 0.1;
    float res = 1.;
    for (int i = 0; i < MAX_STEPS; ++i) {
        vec3 next = ro + totalDist * rd;
        float d = getDistMinimal(next, id);
        if (id != ID_LASER && id != ID_REFR) {
            res = min(res, k * d / totalDist);
            if (abs(d) < mind) return 0.;
        } else {
            return 1.1;
        }
        totalDist += d;
        if (totalDist > maxd) break;
    }
    return clamp(res, 0., 1.);
}

// Function 38
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    // bounding volume
    float tp = (maxHei-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

    float res = 1.0;
    float t = mint;
    for( int i=ZERO; i<16; i++ )
    {
        float h = map( ro + rd*t ).x;
        float s = clamp(8.0*h/t,0.0,1.0);
        res = min( res, s*s*(3.0-2.0*s) );
        t += clamp( h, 0.02, 0.10 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 39
float blurWeight(
    float radiusFraction)
{
    float result = smoothstep(1.0, 0.0, radiusFraction);
    result = (result * result * result);    
    return result;
}

// Function 40
float
shadow( in vec3 start, in vec3 dir )
{
    float ret = 1.0; return 0.0;  /////// !!!!!!!!!! ABORTED !!!!!!!!!!
    
    
    float c = 1.0;//step( mod( iTime, 4.0 ), 2.0 );
    float t = 0.02, t_max = 16.0;
    MPt mp;
    
    #if DRAW_ITERATIONS_GRADIENT
    int it_;
    #endif
    for ( int it=0; it!=SHADOW_MAX_ITERS; ++it )
    {
	    #if DRAW_ITERATIONS_GRADIENT
	    it_ = it;
    	#endif
        vec3 here = start + dir * t;
        mp = map( here );
        ret = min( ret, 8.0*mp.distance/t);
        if ( mp.distance < ( T_EPS * t ) || t > t_max )
        {
        	break;
        }
        
        float inc;
        // NOTE(theGiallo): this is to sample nicely the twisted things
        inc = c * mp.distance * 0.4;
		inc += ( 1.0 - c ) * clamp( mp.distance, 0.02, 0.1 );
        t += inc;
    }
    #if DRAW_ITERATIONS_GRADIENT
    return float(it_);
    #endif
    if ( t > t_max )
    {
        t = -1.0;
    }
    if ( c == 0.0 ) return 1.0 - clamp( ret, 0.0, 1.0 );

    if ( t < 0.0 )
    {
        return 0.0;
    }
    //return 1.0;
    ret = 1.0 / pow(1.0 - 1e-30 + max( mp.distance, 1e-30 ), 5.0 );
    float th = 0.1;
    return smoothstep( 0.0, 1.0, ( ret*1.1 - th ) / (1.0-th) );
}

// Function 41
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

// Function 42
float shadowRaySphere(in vec3 ro, in vec3 rd, vec4 sphere)
{
	float lambda = dot(-(ro - sphere.xyz),rd);
	float dist = length((ro+rd*lambda)-sphere.xyz)-sphere.w;
	return mix(9999.0,dist,step(0.0,lambda)); 
}

// Function 43
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

// Function 44
float softShadow(in vec3 ro, in vec3 rd )
{
    // real shadows	
    float res = 1.0;
    float t = 0.001;
	for( int i=0; i<80; i++ )
	{
	    vec3  p = ro + t*rd;
        float h = p.y - terrainM( p.xz );
		res = min( res, 16.0*h/t );
		t += h;
		if( res<0.001 ||p.y>(SC*200.0) ) break;
	}
	return clamp( res, 0.0, 1.0 );
}

// Function 45
float shadow(vec3 rpos, vec3 rdir) {
	float t = 1.0;
	float sh = 1.0;

	for (int i = 0; i < SHADOW_ITERS; i++) {
		vec3 pos = rpos + rdir * t;
		float h = scene(pos);
		if (h < 0.01) return 0.0;
		sh = min(sh, h/t*8.0);
		t += max(h, SHADOW_QUALITY);
	}
	
	return sh;
}

// Function 46
bool inShadow(vec3 ro,vec3 rd,float d)
{
	float t;
	bool ret = false;

	if(intersectSphere(ro,rd,spheres[2],d,t)){ ret = true; }
	if(intersectSphere(ro,rd,spheres[3],d,t)){ ret = true; }
	if(intersectSphere(ro,rd,spheres[4],d,t)){ ret = true; }
	if(intersectSphere(ro,rd,spheres[5],d,t)){ ret = true; }
	if(intersectSphere(ro,rd,spheres[6],d,t)){ ret = true; }
	if(intersectSphere(ro,rd,spheres[7],d,t)){ ret = true; }
	if(intersectSphere(ro,rd,spheres[8],d,t)){ ret = true; }

	return ret;
}

// Function 47
float calcShadowFactor(vec3 p, vec3 norm, vec3 lightDir)
{
    float t = .0;
    vec3 rayStart = p + norm + eps*2.0;
        for (int i =0; i<16;i++)
        {
            vec3 sP = rayStart - lightDir * t;
            float dist = scene(sP);
            if(dist < eps)
            {
                return 1.0;
            }
            t+=dist;
        }
    return 0.0;
}

// Function 48
float soft_shadow( vec3 ro, 
                   vec3 rd, 
                   float mint, 
                   float maxt, 
                   float k )
{
    float shadow = 1.0;
    float t = mint;

    for( int i=0; i < SOFTSHADOW_STEPS; i++ )
    {
        if( t < maxt )
        {
            float h = scenedf( ro + rd * t );
            shadow = min( shadow, k * h / t );
            t += SOFTSHADOW_STEPSIZE;
        }
    }
    return clamp( shadow, 0.0, 1.0 );

}

// Function 49
float SoftShadowMissile( in vec3 origin, in vec3 direction, Missile missile )
{
  float res = 2.0, t = 0.02, h;
  for ( int i=0; i<8; i++ )
  {
    h = MapMissile(origin+direction*t, missile);
    res = min( res, 7.5*h/t );
    t += clamp( h, 0.05, 0.2 );
    if ( h<0.001 || t>2.5 ) break;
  }
  return clamp( res, 0.0, 1.0 );
}

// Function 50
void ToggleEffects(inout vec4 fragColor, vec2 fragCoord)
{
   // read and save effect values from buffer  
   vec3 effects =  mix(vec3(-1.0,1.0,1.0), readRGB(ivec2(20, 0)), step(1.0, float(iFrame)));
   effects.x*=1.0+(-2.*float(keyPress(49))); //1-key  LENSDIRT
   effects.y*=1.0+(-2.*float(keyPress(50))); //2-key  GRAINFILTER
   effects.z*=1.0+(-2.*float(keyPress(51))); //3-key  ChromaticAberration
   
   vec3 effects2 =  mix(vec3(1.0,1.0,1.0), readRGB(ivec2(22, 0)), step(1.0, float(iFrame)));
   effects2.y*=1.0+(-2.*float(keyPress(52))); //4-key  AA-pass
   effects2.x*=1.0+(-2.*float(keyPress(53))); //5-key  lens flare

   fragColor.rgb = mix(effects, fragColor.rgb, step(1., distance(fragCoord.xy, vec2(20.0, 0.0))));  
   fragColor.rgb = mix(effects2, fragColor.rgb, step(1., distance(fragCoord.xy, vec2(22.0, 0.0))));  
}

// Function 51
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<14; i++ )
    {
		float h = map( ro + rd*t );
        res = min( res, 8.*h/t );
        t += clamp( h, 0.08, 0.25 );
        if( res<0.001 || t>tmax ) break;
    }
    return max(0.0, res);
}

// Function 52
float calcShadow(vec3 p, vec3 lightPos, float sharpness) {
    vec3 rd = normalize(lightPos - p);
    
    float h;
    float minH = 1.0;
    float d = 0.7;
    for (int i = 0; i < 16; i++) {
        h = map(p + rd * d).x;
        minH = abs(h / d);
        if (minH < 0.01)
            return 0.0;
        d += h;
    }
    
    return minH * sharpness;
}

// Function 53
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

// Function 54
bool intersectShadow( in vec3 ro, in vec3 rd, in float l )
{
    float t;

    bvec4 sss;

    sss.x = esfera2(   fpar00[0], ro, rd, l );
    sss.y = esfera2(   fpar00[1], ro, rd, l );
    sss.z = cylinder2( fpar00[2], ro, rd, l );
    sss.w = cylinder2( fpar00[3], ro, rd, l );

    return any(sss);
}

// Function 55
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<32*1; i++ )
    {
        float h = mapScene( ro + rd*t ).d;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.01, 0.90 );
        if( h<(0.00001) || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 56
float softshadow(const in vec3 ro, const in vec3 rd) {
    float hit = 0.02;
    float res = 1.0;
    for (int i = 0; i < ITER_RAY_SHADOW; i++) {
        float h = sceneDistance(ro + rd * hit);
        res = min(res, 20.0 * h / hit);
        hit += clamp(h, 0.02, 0.10);
        if (h < PRECISION_RAY_SHADOW || hit > BAIL_DISTANCE_SHADOW) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Function 57
float ObjSShadow (vec3 ro, vec3 rd, float dMax)
{
  float sh, d, h;
  sh = 1.;
  d = 0.02;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += clamp (3. * h, 0.04, 0.3);
    if (sh < 0.05 || d > dMax) break;
  }
  return 0.6 + 0.4 * sh;
}

// Function 58
float marchShadow(vec3 ro, vec3 rd, float t, float mt, float tanSourceRadius)
{
 	float d;
    float minVisibility = 1.0;
    
    vec4 material;
    
    for(int i = NON_CONST_ZERO; i < ITER_SHADOW && t < mt; ++i)
    {
        float coneWidth = max(0.0001, tanSourceRadius * t);
        
        vec3 posWS = ro + rd*t;
        d = fSDF(posWS, false, material) + coneWidth*0.5;
        
        minVisibility = min(minVisibility, (d) / max(0.0001, coneWidth*1.0));
        t += d;
        
        if(i >= ITER_SHADOW - 1)
        {
            t = mt;
        }              
        
        if(minVisibility < 0.01)
        {
            minVisibility = 0.0;
        }
    }
      
    return smoothstep(0.0, 1.0, minVisibility);
}

// Function 59
float lightPointDiffuseShadow(vec3 pos, vec3 lightPos, vec3 normal) {
	vec3 lightDir = normalize(lightPos - pos);
	float lightDist = length(lightPos - pos);
	float color = square(dot(normal, lightDir)) / square(lightDist);
	if (color > 0.00) color *= castShadowRay(pos, lightPos, 0.05);
	return max(0.0, color);
}

// Function 60
float calcSoftshadow(in vec3 ro, in vec3 rd)
{
    float res = 1.0;
    float tmax = 12.0;  
    
    float t = 0.02;
    for( int i=0; i<40; i++ )
    {
		float h = map(ro + rd*t);
        res = min( res, 24.0*h/t );
        t += clamp( h, 0.0, 0.80 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 61
float getBlurredTexture(vec2 uv)
{
    float v = 0.;
    for (int j=0;j<btns ;j++)
    {
       float oy = float(j)*btdist/max(float(aasamples-1), 1.);
       for (int i=0;i<btns ;i++)
       {
          float ox = float(i)*btdist/max(float(aasamples-1), 1.);
          v+= dot(texture(iChannel1, uv + vec2(ox, oy)).rgb, vec3(1./3., 1./3., 1./3.));
       }
    }
    return v/float(btns*btns);
}

// Function 62
float shadow(vec3 pos, vec3 lPos, PlantSpace ps)
{   
    vec3 dir = lPos - pos;  // Light direction & disantce
    
    float len = length(dir);
    dir /= len;				// It's normalized now
    
    pos += dir * MIN_DST * 2.0;  // Get out of the surface
    
    float dst = SDF(pos, ps).x; // Get the SDF
    
    // Start casting the ray
    float t = 0.0;
    float obscurance = 1.0;
    
    while (t < len)
    {
        if (dst < MIN_DST) return 0.0; 
        obscurance = min(obscurance, (20.0 * dst / t)); 
        t += dst;
        pos += dst * dir;
        dst = SDF(pos, ps).x;
    }
    return obscurance;     
}

// Function 63
float shadow_march(vec4 pos, vec4 dir, float distance2light, float light_angle, inout object co)
{
	float light_visibility = 1.;
	float ph = 1e5;
	pos.w = map(pos.xyz, co);
	for (int i = min(0, iFrame); i < 32; i++) 
    {
		dir.w += pos.w;
		pos.xyz += pos.w*dir.xyz;
		pos.w = map(pos.xyz, co);
		float y = pos.w*pos.w/(2.0*ph);
        float d = (pos.w+ph)*0.5;
		float angle = d/(max(0.00001,dir.w-y)*light_angle);
        light_visibility = min(light_visibility, angle);
		ph = pos.w;
        if(i >= 31) return 0.;
		if(dir.w >= distance2light) break;
		if(dir.w > maxd || pos.w < max(mind*dir.w, 0.0001)) return 0.;
    }
	light_visibility = clamp(2.*light_visibility - 1.,-1.,1.);
	return  0.5 + (light_visibility*sqrt(1.-light_visibility*light_visibility) + asin(light_visibility))/3.14159265; //looks better and is more physically accurate(for a circular light source)
}

// Function 64
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    // bounding volume
    float tp = (maxHei-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

    float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
        vec2 uv;
		float h = map( ro + rd*t ,rd,uv);
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 65
float intersect_shadow(vec3 p)
{   
    vec3 qq = p;
    qq.xz = mod(qq.xz, 5.0) - 2.5;
    qq.y -= 2.0;
    
    const int mi = 3;
    float sf = 3.0;
    float dcc = 0.0;
    float db = sdf_sbox(qq, vec3(0.5));
    vec3 pp = qq;
    for (int i = 0; i < mi; ++i)
    {
        pp.xy = pp.xy * mat2(0.5, -0.866, 0.866, 0.5);
        vec3 ppm = pp;
       	ppm = mod(ppm, 0.2) - 0.1;
        dcc = sdf_cbox(ppm * sf, 0.1) / sf;
        db = max(db, -dcc);
        sf *= 1.5;      
    }
    
    float d2 = max(db, length(qq) - 0.5);
    return d2;
}

// Function 66
float maskBlurry(vec2 p)
{
    return clamp((shapeDist(p).w + 0.003) * 75.0, 0.0, 1.0);
}

// Function 67
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int iter = 24; 
    
    ro += n*.0015;
    vec3 rd = lp - ro; // Unnormalized direction ray.
    

    float shade = 1.;
    float t = 0.;//.0015; // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = 0; i<iter; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        t += clamp(d, .01, .25); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}

// Function 68
float ts_shadow_sample( TrnSampler ts, sampler2D ch, vec3 x )
{
#if WITH_TRN_SHADOW
    return ts_shadow_eval( ts_shadow_lookup( ts, ch, x ).xy, length(x) );
#else
    return 1.;
#endif
}

// Function 69
vec3 postEffects( in vec3 col, in vec2 uv, in float time )
{
	// gamma correction
	col = pow( clamp(col,0.0,1.0), vec3(0.45) );
	// vigneting
	col *= 0.5+0.5*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.15 );
	return col;
}

// Function 70
float softShadow(vec3 ro, vec3 rd, float maxDist) {
    float total = 0.;
    float s = 1.;
    
    for (int i = 0; i < SHADOW_STEPS; ++i) {
        float d = scene(ro + rd * total);
        if (d < EPS) {
            s = 0.;
            break;
        }
        if (maxDist < total) break;
        s = min(s, SHADOW_SOFTNESS * d / total);
        total += d;
    }
    
    return s;
}

// Function 71
vec3 diskWithMotionBlur( vec3 col, in vec2 uv, in vec3 sph, in vec2 cd, in vec3 sphcol, in float alpha )
{
	vec2 xc = uv - sph.xy;
	float a = dot(cd,cd);
	float b = dot(cd,xc);
	float c = dot(xc,xc) - sph.z*sph.z;
	float h = b*b - a*c;
	if( h>0.0 )
	{
		h = sqrt( h );
		
		float ta = max( 0.0, (-b - h)/a );
		float tb = min( 1.0, (-b + h)/a );
		
		if( ta < tb ) // we can comment this conditional, in fact
		    col = mix( col, sphcol, alpha*clamp(2.0*(tb-ta),0.0,1.0) );
	}

	return col;
}

// Function 72
float shadows( vec3 ro, vec3 rd, float tMax, float k, int octaves ) {
    float res = 1.0;
	float t = 0.1;
	for(int i=0; i<22; i++) {
        if (t<tMax) {
			float h = map(ro + rd*t, octaves).x;
        	res = min( res, k*h/t );
        	t += h;
		}
		else break;
    }
    return clamp(res, 0.2, 1.0);
}

// Function 73
float softshadow(in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k) 
{
    float sh = 1.0;
    float t = mint;
    float h = 0.0;
    for(int i = 0; i < 19; i++) 
	{
        if(t > maxt) continue;
		orbitTrap = vec4(10.0);
        h = map(ro + rd * t);
        sh = min(sh, k * h / t);
        t += h;
    }
    return sh;
}

// Function 74
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

// Function 75
float shadow(in vec3 ro, in vec3 rd, in float mint, in float tmax)
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<15; i++ )
    {
		float h = map(ro + rd*t);
        res = min( res, 4.*h/t );
        t += clamp( h, 0.01, .1 );
        if(h<0.001 || t>tmax) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 76
vec2 ShadowAndAmbient(in vec3 ro, in vec3 rd) {
    vec3 p0 = vec3(0.0), p1 = vec3(0.0);
    
    IRayAABox(ro, rd, 1.0/rd, scmin, scmax, p0, p1);
    
    if (length(ro - p1) < 0.01) return vec2(1.0);
    
    p0 = ro + rd*0.01;
    vec2 dir = normalize(rd.xz);
    float s = rd.y / length(rd.xz);
    
    vec2 mg = vec2(0.0), mr = vec2(0.0), n = vec2(0.0), f = vec2(0.0);
    voronoi_s(p0.xz*2.0, n, f, mg, mr);
    
    float h = map(n + mg);
    
    float a = voronoi_ao(ro.y, n, f, mg);
    vec3 dn = voronoi_n(dir, n, f, mg, mr);
    
    float rh = 0.0, prh = p0.y;
    
  	float dmax = length(p0.xz - p1.xz);
	float mh = 1.0;
    
    const int steps = 8;
    for (int i = steps; i > 0; --i) {
        dn.x *= 0.5;
        rh = p0.y + dn.x*s;
       
        if (dn.x > dmax || h > prh || h > rh) break; 
        
        prh = rh;
       
        h = map(n + mg);
        
        mh = min(mh, 14.0*(rh-h)/(dn.x*s));
        
        dn = voronoi_n(dir, n, f, mg, mr);
    }
    
    if (h > prh || h > rh) return vec2(0.0, a);
    
    return vec2(clamp(mh, 0.0, 1.0), a);
}

// Function 77
float softShadow(vec3 ro, vec3 lp, float k)
{
    float shade = 1.;
    float dist = MIN_DIST;    
    vec3 rd = (lp-ro);
    float end = max(length(rd), MIN_DIST);
    float stepDist = end/25.0;
    rd /= end;
    for (int i=0; i<25; i++)
    {
        float h = map(ro+rd*dist);
        //if (s.mat != BBOX)
            shade = min(shade, k*h/dist);
        dist += clamp(h, 0.02, stepDist*2.0);
        
        if (h < 0.0 || dist > end) break; 
    }

    return min(max(shade, 0.0) + SHADOW, 1.0); 
}

// Function 78
v0 shadow(v2 o,v2 i
){const v0 a=32.//shadow hardnes
 ;v0 r=1.,h =1.,t=.0005//t=(self)intersection avoidance distance
 ;const v0 it=clamp(IterSh,0.,64.)
 ;for(v0 j=0.;j<it;j++
 ){h=dm(o+i*t).x
  ;r=min(r,h*a/t)
  ;t+=clamp(h,.02,2.);}//limit max and min stepping distances
 ;return clamp(r,0.,1.);}

// Function 79
float ShadowFactor(in vec3 ro, in vec3 rd, in vec3 invrd, in vec3 bmin, in vec3 bmax) 
{
    vec3 re = vec3(0.0);
    vec3 pa = vec3(0.0);
    
    IRayAABox(ro, -rd, -invrd, bmin, bmax, re, pa);
    if (dot(re - ro, rd) <= EPS) return 1.0;
    
    vec3 ep = floor(ro + rd*EPS);
    float v = 0.0;
    float ret = 1.0;
    for (float i = 0.0; i < 32.0; ++i) {
        if (map(ep, v)) {
            ret = -i;
        	break;
        }
        
        IRayAABox(ro - rd*2.0, rd, invrd, ep, ep+1.0, pa, ro);
        ep = floor(ro + rd*EPS);
        
        if (dot(re - ro, rd) <= EPS) {
            ret = 1.0;
            break;
        }
    }
    
    return ret;
}

// Function 80
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    // bounding volume
    float tp = (0.8-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );
    
	float res = 1.0;
    float t = mint;
    for( int i=ZERO; i<16; i++ )
    {
		float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 81
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

// Function 82
float calcShadow(in vec3 ro, in vec3 rd ) {
    vec2 eps = vec2( 150.0, 0.0 );
    float height1 = terrain( ro.xz );
    float height2 = terrain( ro.xz );
    float d1 =  5.0, d2 =  50.0, d3 = 150.0;
    float s1 = clamp( 1.0*(height1 + rd.y*d1 - terrain(ro.xz + d1*rd.xz)), 0.0, 1.0 );
    float s2 = clamp( 0.5*(height1 + rd.y*d2 - terrain(ro.xz + d2*rd.xz)), 0.0, 1.0 );
    float s3 = clamp( 0.2*(height2 + rd.y*d3 - terrain(ro.xz + d3*rd.xz)), 0.0, 1.0 );
    return min( min( s1, s2 ), s3 );
}

// Function 83
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

// Function 84
float shadow(in vec3 p, in vec3 rd) { return castRay(p+rd*.001, rd, 1000.).y >= 0. ? 0. : 1.; }

// Function 85
void draw_shadow_box(inout vec4 fragColor, vec2 fragCoord, vec4 box, float border)
{
    fragColor.rgb *= mix(1.-shadow_box(fragCoord, box, border), 1., .5);
}

// Function 86
bool intersectShadow( in vec3 ro, in vec3 rd, in float dist ) {
    float t;
	
	t = iSphere( ro, rd, vec4( 1.5,1.0, 2.7,1.0) );  if( t>eps && t<dist ) { return true; }
    t = iSphere( ro, rd, vec4( 4.0,1.0, 4.0,1.0) );  if( t>eps && t<dist ) { return true; }
	t = iSphere( ro, rd, vec4( 3.3,0.3, 1.3, 0.3) );  if( t>eps && t<dist ) { return true; }
    return false; // optimisation: planes don't cast shadows in this scene
}

// Function 87
float softShadow(vec3 pos, vec3 rayDir, float start, float end, float k ){
    float res = 1.0;
    float depth = start;
    for(int counter = ZERO; counter < 32; counter++){
        float dist = getSDF(pos + rayDir * depth);
        if( abs(dist) < EPSILON){ return 0.0; }       
        if( depth > end){ break; }
        res = min(res, k*dist/depth);
        depth += dist;
    }
    return saturate(res);
}

// Function 88
float shadow( in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    float t = .1;
	float h;
	vec3 p;
    for (int i = 0; i < 12; i++)
	{
        p =  ro + rd*t;
 		
		h = DF(p, false);
      	res = min(5.*h / t, res);
        res = max(res, hitSpecs ? .4: .05);

		t += h;
	}
    return res;
}

// Function 89
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
        float h = map( ro + rd*t, false ).dist;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.00001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 90
vec4 blur(sampler2D sp, vec2 U, vec2 scale) {
    vec4 O = vec4(0);  
    const int s = samples/sLOD;
    
    for ( int i = 0; i < s*s; i++ ) {
        vec2 d = vec2(mod(float(i),float(s)), i/s)*float(sLOD) - float(samples)/2.;
        O += gaussian(d) * texture( sp, U + scale * d , -100. );
    }
    
    return O / O.a;
}

// Function 91
float softShadow( in vec3 o, in vec3 i, float n, float m){
 float r=1.,t=n,k=.05;const float s=100.,u=1.+.78/sqrt(s*.5);
 //not perfect, decent for s range[10.100]
 for(float j=0.;j<s;j++){
  float h=df(o+i*t)/(s+1.);
  if(h<0.)return 0.;
  r=min(r,(k*h/t));
  t=t*u+h;//mod by ollj allows for smaller [n]
     if(t > m)break;}
  return r*s*79./length(i);}

// Function 92
float shadow(vec3 from, vec3 increment)
{
	const float minDist = 10.0;
	
	float res = 1.0;
	float t = 0.1;
	for(int i = 0; i < 80; i++) {
		float m;
        float h = distanceField(from + increment * t,m);
        if(h < minDist)
            return 0.0;
		
		res = min(res, 2.0 * h / t);
        t += h * 0.5;
		//if (t >= 20.0) break;
    }
    return res;
}

// Function 93
float moonShadow(vec3 pos, vec4 moon, vec3 dir) {
    vec3 p = pos - moon.xyz;
	float m = dot(p, dir);
    float n = dot(p, p);
    return step(step(m, 0.0) * moon.w * moon.w, n - m * m);
}

// Function 94
float ShadowCone(vec3 pos, vec3 dir, float FAR, float CR, float Time) {
    float dist=0.; float Occ=1.; float dft,R;
    for (int i=0; i<128; i++) {
        dft=SDF(pos+dir*dist,Time).D;
        if (dist>FAR) break;
        if (dft<eps.x) return 0.;
        R=dist*CR;
        Occ=min(Occ,dft/R);
        dist=dist+dft;
    }
	return max(0.,Occ);
}

// Function 95
bool in_moon_shadow( vec3 p )
{
	return ( dot( p - moon_center, sun_direction ) < 0.0 )
		   && ( lensqr( p - project_on_line1( p, moon_center, sun_direction ) ) < moon_radius * moon_radius );
}

// Function 96
bool shadowCylinder( in vec3 ro, in vec3 rd, in float he, float ra, in float tmax )
{
    float he2 = he*he;
    
    float k2 = 1.0        - rd.y*rd.y;
    float k1 = dot(ro,rd) - ro.y*rd.y;
    float k0 = dot(ro,ro) - ro.y*ro.y - ra*ra;
    
    float h = k1*k1 - k2*k0;
    if( h<0.0 ) return false;
    h = sqrt(h);
    float t = (-k1-h)/k2;

    // body
    float y = ro.y + t*rd.y;
    if( y>0.0 && y<he )
    {
        return t>0.0 && t<tmax;
    }
    
    // caps
    t = ( ((y<0.0) ? 0.0 : he) - ro.y)/rd.y;
    if( abs(k1+k2*t)<h )
    {
        return t>0.0 && t<tmax;
    }

    return false;
}

// Function 97
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

// Function 98
float shadow(in vec3 rp)
{
    const float dist = 0.1;
    float d = 1.0;
    rp += lightDir * dist * 1.5;
    
    for (int i = 1; i < 3; ++i)
    {
        float m = map(rp);
        d = min(d, clamp(m / dist, 0.0, 1.0));
        rp += lightDir * max(m * 2.0, 0.01);
    }
    return d;
}

// Function 99
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, in float cuttingPlane) {
    // bounding volume
    float tp = (maxHei - ro.y) / rd.y;
    if (tp > 0.0) tmax = min(tmax, tp);

    float res = 1.0;
    float t = mint;
    for (int i=0; i < 16; i++) {
		float h = map(ro + rd * t, cuttingPlane).x;
        float s = clamp(4.0 * h / t, 0.0, 1.0);
        res = min(res, s * s * (3.0 - 2.0 * s));
        t += clamp( h, 0.02, 0.10 );
        if (res < 0.005 || t > tmax) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 100
float sphAreaShadow( vec3 P, in vec4 L, vec4 sph )
{
  vec3 ld = L.xyz - P;
  vec3 oc = sph.xyz - P;
  float r = sph.w - BIAS;
  
  float d1 = sqrt(dot(ld, ld));
  float d2 = sqrt(dot(oc, oc));
  
  if (d1 - L.w / 2. < d2 - r) return 1.;
  
  float ls1 = L.w / d1;
  float ls2 = r / d2;

  float in1 = sqrt(1.0 - ls1 * ls1);
  float in2 = sqrt(1.0 - ls2 * ls2);
  
  if (in1 * d1 < in2 * d2) return 1.;
  
  vec3 v1 = ld / d1;
  vec3 v2 = oc / d2;
  float ilm = dot(v1, v2);
  
  if (ilm < in1 * in2 - ls1 * ls2) return 1.0;
  
  float g = length( cross(v1, v2) );
  
  float th = clamp((in2 - in1 * ilm) * (d1 / L.w) / g, -1.0, 1.0);
  float ph = clamp((in1 - in2 * ilm) * (d2 / r) / g, -1.0, 1.0);
  
  float sh = acos(th) - th * sqrt(1.0 - th * th) 
           + (acos(ph) - ph * sqrt(1.0 - ph * ph))
           * ilm * ls2 * ls2 / (ls1 * ls1);
  
  return 1.0 - sh / PI;
}

// Function 101
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

// Function 102
float boxSoftShadow( in vec3 ro, in vec3 rd, in vec3 rad, in float sk ) 
{
    rd += 0.0001 * (1.0 - abs(sign(rd)));
	vec3 rdd = rd;
	vec3 roo = ro;

    vec3 m = 1.0/rdd;
    vec3 n = m*roo;
    vec3 k = abs(m)*rad;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max( max( t1.x, t1.y ), t1.z );
	float tF = min( min( t2.x, t2.y ), t2.z );
	
    if( tN<tF && tF>0.0) return 0.0;
    
    float sh = 1.0;
    sh = segShadow( roo.xyz, rdd.xyz, rad.xyz, sh );
    sh = segShadow( roo.yzx, rdd.yzx, rad.yzx, sh );
    sh = segShadow( roo.zxy, rdd.zxy, rad.zxy, sh );
    sh = clamp(sk*sqrt(sh),0.0,1.0);
    return sh*sh*(3.0-2.0*sh);
}

// Function 103
float softShadow(vec3 ro, vec3 lp, float k, float t){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 32; 
    
    vec3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = .0015; // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        dist += clamp(h, .05, .5); 
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0. || dist > end) break; 
    }

    // I've added a constant to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me. Sometimes, I'll add AO also just for kicks. :)
    return min(max(shade, 0.) + .2, 1.); 
}

// Function 104
float shadow(in vec3 origin, in vec3 direction) {
    float hit = 1.0;
    float t = 0.02;
    
    for (int i = 0; i < 1000; i++) {
        float h = scene(origin + direction * t).x;
        if (h < 0.001) return 0.0;
        t += h;
        hit = min(hit, 10.0 * h / t);
        if (t >= 2.5) break;
    }

    return clamp(hit, 0.0, 1.0);
}

// Function 105
float Shadow( in vec3 ro, in vec3 rd, in float maxt)
{
	float res = 1.0;
    float dt = 0.04;
    float t = .02;
    for( int i=0; i < 20; i++ )
    {
        float h = Map( ro + rd*t ).x;
        res = min( res, 2.0*h/t );
        t += max( 0.15, dt );
    }
    return res;
}

// Function 106
vec4 seeThroughWithShadow(float yc, vec2 p, vec3 point, mat3 rotation, mat3 rrotation)
{
    float shadow = distanceToEdge(point) * 30.0;
    shadow = (1.0 - shadow) / 3.0;
    if (shadow < 0.0)
        shadow = 0.0;
    else
        shadow *= amount;
    vec4 shadowColor = seeThrough(yc, p, rotation, rrotation);
    shadowColor.r -= shadow;
    shadowColor.g -= shadow;
    shadowColor.b -= shadow;
    return shadowColor;
}

// Function 107
float getBlurSize(float depth, float focusPoint, float focusScale)
{
	float coc = clamp((1.0 / focusPoint - 1.0 / depth)*focusScale, -1.0, 1.0);
	return abs(coc) * 10.0;
}

// Function 108
float shadow( in vec3 start, in vec3 n, in vec3 ldir, in float p )
{    
    // Do some quick "is the sun even shining on here" tests.
    // We wait until the sun is just below the horizon before considering
    // it gone.
    if( dot(n,ldir) <= 0.0 || dot(ldir,UP) <= -.25) return 0.0;
    
	float t = EPSILON*40.0;
	float res = 1.0;
    for ( int i = 0; i < S_STEPS; ++i )
    {
        float d = distR( start + ldir * t );
        if ( d < EPSILON*.1 )
            return 0.0;
		
		res = min( res, p * d / t );
        t += d;
		
		if ( t > MAX_DEPTH )
			break;
    }
    return res;
}

// Function 109
float ObjSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  vec2 gIdP;
  float sh, d, h;
  sh = 1.;
  gIdP = vec2 (-99.);
  d = 0.01;
  for (int j = 0; j < 30; j ++) {
    p = ro + d * rd;
    gId = PixToHex (p.xz / hgSize);
    if (length (vec3 (gId.xy, gId.x + gId.y)) <= grLim) {
      if (gId.x != gIdP.x || gId.y != gIdP.y) {
        gIdP = gId;
        SetPngConf ();
      }
      h = ObjDf (p);
      sh = min (sh, smoothstep (0., 0.05 * d, h));
      d += clamp (h, 0.05, 0.3);
    } else d += 0.2;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 110
float Shadow(vec3 p,vec3 l, float d, float r)
{

    float res = 1.0;
    float t = 0.1;

    for (int i = 0; i < MAX_STEPS; ++i) {

        if (res < 0.0 || t > d)
            break;
    
        float h = Object(p+t*l).v;

        res = min(res, r * h / t);
        t += h;    
    }    

    return clamp(res, 0.0, 1.0);
}

// Function 111
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

// Function 112
vec4 texture_blurred(in sampler2D tex, vec2 uv)
{
    return (texture(iChannel0, uv)
		+ texture(iChannel0, vec2(uv.x+1.0, uv.y))
		+ texture(iChannel0, vec2(uv.x-1.0, uv.y))
		+ texture(iChannel0, vec2(uv.x, uv.y+1.0))
		+ texture(iChannel0, vec2(uv.x, uv.y-1.0)))/5.0;
}

// Function 113
float softshadow(in vec3 ro, in vec3 rd)
{
    float res = 1.0;
    float t = 0.0;
    for (int i = 0; i < SOFT_SHADOW_STEPS; ++i)
    {
		vec3 pos = ro + rd * t;
        float h = map(pos).x;
        res = min(res, float(SOFT_SHADOW_STEPS) * h / t);
        if(res < 0.0001)
		{
	    	break;
		}
        t += clamp(h, 0.01, 0.2);
    }
    return saturate(res);
}

// Function 114
bool isInSpheresShadow(Intersect intersection, Light light) {
    float lightDist = length(light.position - intersection.position); //light dist
    vec3 shadowDir = normalize(light.position - intersection.position);  //light dir

    Ray shadowRay =  Ray(intersection.position + 10.0*EPSILON*shadowDir, shadowDir);
    float shadowDist = shortestDistanceToSpheres(shadowRay);

    // if intersected sphere in light direction then shadow before reaching light
    if(shadowDist != MAX_DIST_SHADOW && shadowDist < lightDist) 
        return true;
    return false;

}

// Function 115
float calcShadow(vec3 p, vec3 lightPos, float sharpness) {
	vec3 rd = normalize(lightPos - p);

	float h,
		  minH = 1.,
		  d = .7;
	for (int i = 0; i < 16; i++) {
		h = map(p + rd * d).x;
		minH = abs(h / d);
		if (minH < .01)
			return 0.;
		d += h;
	}

	return minH * sharpness;
}

// Function 116
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = map( ro + rd*t );
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 117
float readShadow(vec2 uv)
{
    vec4 data = texelFetch(iChannel0, ivec2(uv), 0);
    return unpack4(data.x).w;
}

// Function 118
float terrainCalcShadow(in vec3 ro, in vec3 rd ) {
	vec2  eps = vec2(150.0,0.0);
    float h1 = terrainMed( ro.xz );
    float h2 = terrainLow( ro.xz );
    
    float d1 = 10.0;
    float d2 = 80.0;
    float d3 = 200.0;
    float s1 = clamp( 1.0*(h1 + rd.y*d1 - terrainMed(ro.xz + d1*rd.xz)), 0.0, 1.0 );
    float s2 = clamp( 0.5*(h1 + rd.y*d2 - terrainMed(ro.xz + d2*rd.xz)), 0.0, 1.0 );
    float s3 = clamp( 0.2*(h2 + rd.y*d3 - terrainLow(ro.xz + d3*rd.xz)), 0.0, 1.0 );

    return min(min(s1,s2),s3);
}

// Function 119
float shadow(vec3 pos, vec3 normal, vec3 lPos, SceneSetup ps)
{       
#ifdef PROGRESSIVE_RENDERING
    lPos += ps.noise * 8.0; // In progressive mode, the light position is jittered for smooth shadows
#endif
    
    vec3 dir = lPos - pos;  // Light direction & disantce
    
    float len = length(dir);
    dir /= len;				// It's normalized now
    
    pos += normal * MIN_DST * 40.0;
    
    
    vec2 ray =  castRay(pos, dir, MAX_DST, MIN_DST * 10.0, ps, true);
    if (ray.x < MAX_DST) return 0.0; // if it crosses something opage shadow is full
    
    ray =  castRay(pos, dir, MAX_DST, MIN_DST * 10.0, ps, false);
    if (ray.x < MAX_DST) return 0.45; // if it crosses something transparent shadow is partial
    
    // No shadow
    return 1.0;
}

// Function 120
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax ) {
    float res = 1.0;
    float t = mint;
    for( int i=0; i<24; i++ )
    {
        float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.04, 1.0 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 121
bool traceShadow( vec3 roo, vec3 rd, float maxdist ) { 	

	float u, d = MAXDISTANCE;
	vec3 ro = roo - roomoffset;
	vec2 e, s;
	
	for( int i=0; i<4; i++ ) {
			float angle = float(i)*PI*0.5+pillarAngle;
			s = vec2( cos( angle ), sin( angle ) ) + pillarPosition; 
			e = vec2( cos( angle+PI*0.5 ), sin( angle+PI*0.5 ) ) + pillarPosition;
				
			intersectSegment( ro, rd, s, e, d, u );
			if( d < maxdist ) return true;
		}
	return false;
}

// Function 122
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

// Function 123
float shadow(v2 o,v2 i
){const float a=32.//shadow hardnes
 ;float r=1.,h =1.,t=.0005//t=(self)intersection avoidance distance
 ;const float it=clamp(IterSh,0.,64.)
 ;for(float j=0.;j<it;j++
 ){h=dm(o+i*t).x
  ;r=min(r,h*a/t)
  ;t+=clamp(h,.02,2.);}//limit max and min stepping distances
 ;return clamp(r,0.,1.);}

// Function 124
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<48; i++ )
    {
        float h = density(ro + rd*t).x;
		h = max( h, 0.0 );
        res = min( res, k*h/t );
        t += clamp( h, 0.02, 0.5 );
		if( h<0.0001 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 125
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 126
float test_shadow(vec2 xy, float height)
{
    vec3 r0 = vec3(xy, height);
    vec3 rd = normalize( light - r0 );
    
    float hit = 1.0;
    float t   = 0.001;
	
    for (int j=1; j<25; j++)
    {
        vec3 p = r0 + t*rd;
        float h = height_map( p.xy );
        float height_diff = p.z - h;
		
        if (height_diff<0.0)
        {
            return 0.0;
        }
		
        t += 0.01+height_diff*.02;
        hit = min(hit, 2.*height_diff/t); // soft shaddow   
    }
	
    return hit;
}

// Function 127
vec2 shadowMarch(vec3 startPoint, vec3 direction, int iterations, float maxStepDist)
{
    vec3 point = startPoint;
    direction = normalize(direction);
    float dist = 10.0;
    float distSum = 0.0;
    float shadowData = 0.0;
    float shadow = 0.0;
    
    int i;
    for (i = 0; i < SHADOW_RAYS_COUNT && distSum < MAX_SHADOW_DISTANCE && abs(dist) > EPSILON * 0.5; i++)
    {
     	dist = terrainDist(point, direction.xy);
        
        shadow = dot(normalize((point - vec3(0.0, 0.0, dist)) - startPoint), direction);
        if(shadow > shadowData) shadowData = shadow;
        
        dist = min(dist, 1.0);
        distSum += dist;
        point += direction * dist;     
    }
    
    return vec2(smoothstep(MAX_SHADOW_DISTANCE - EPSILON, MAX_SHADOW_DISTANCE, distSum), shadowData);
}

// Function 128
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	rgb = pow(rgb, vec3(0.45));
	// Then saturation...
	rgb = clamp(mix(  vec3(dot(vec3(.2125, .7154, .0721), rgb)), rgb, 1.3), 0.0, 1.0);
	
	// Vignette...
	rgb *= .4+0.4*pow(60.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.3 );	
	return rgb;
}

// Function 129
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

// Function 130
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

// Function 131
float SoftShadow(const in vec3 rayOrigin)
    {
        float res = 1.0;
        float t = 0.1;
        float sceneSample_F = float(SHADOW_SAMPLE);

        for(int i=0; i < SHADOW_SAMPLE; ++i)
        {
            vec3  p = rayOrigin + sunDirection * t;
            float h = SceneDistance(p).x;
            res = min(res, sceneSample_F * h / t);

            if(res < 0.1)
                return 0.1;
            
            t += h;
        }

        return res;
    }

// Function 132
float shadow(in vec3 ro, in vec3 rd, in float mint, in float maxt )
{
	float res = 1.0;
    float t = mint;
    float ph = 1e10;
    
    for( int i=0; i<18; i++ )
    {
		float rz = map(ro + rd*t);
        res = min(res, 4.5*rz/t);
        t += rz;
        if( res<0.0001 || t>maxt ) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Function 133
float ObjSShadow (vec3 ro, vec3 rd)
{
  float dTol = 0.01;
  float sh = 1.;
  float d = 0.07 * szFac;
  for (int i = 0; i < 50; i++) {
    float h = ObjDf (ro + rd * d);
    sh = min (sh, 20. * h / d);
    d += 0.07 * szFac;
    if (h < dTol*(1.0+d)) break;
    dTol *= 1.01;
  }
  return clamp (sh, 0., 1.);
}

// Function 134
float atm_planet_shadow( float coschi, float cosbeta )
{
    return clamp( SCN_RAYCAST_SHADOW_UMBRA * ( coschi + cosbeta ) + .5, 0., 1. );
}

// Function 135
vec3 GetAmbientShadowColor()
{
    return vec3(0, 0, 0.2);
}

// Function 136
float segShadow( in vec3 ro, in vec3 rd, in vec3 pa, float sh )
{
    float dm = dot(rd.yz,rd.yz);
    float k1 = (ro.x-pa.x)*dm;
    float k2 = (ro.x+pa.x)*dm;
    vec2  k5 = (ro.yz+pa.yz)*dm;
    float k3 = dot(ro.yz+pa.yz,rd.yz);
    vec2  k4 = (pa.yz+pa.yz)*rd.yz;
    vec2  k6 = (pa.yz+pa.yz)*dm;
    
    for( int i=0; i<4 + ANGLE_loops; i++ )
    {
        vec2  s = vec2(i&1,i>>1);
        float t = dot(s,k4) - k3;
        
        if( t>0.0 )
        sh = min(sh,dot2(vec3(clamp(-rd.x*t,k1,k2),k5-k6*s)+rd*t)/(t*t));
    }
    return sh;
}

// Function 137
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

// Function 138
float Shadow(vec3 p, vec3 n, vec3 l, float hd, float d, float rnd, int ssteps)
{
    float nl = max(0., dot(n, l))
    , ao = /*sqrt*/(clamp((Scene(p + n * aod).d - hd) / aod, 0., 1.));
    if (nl > 1e-4) {
    	vec3 sht = sunDir * shd
    	, hp = p + n * .002; // self-shadow bias hit position
        int iters = max(1, int(rnd + float(ssteps) / (1.+.002*d)));
        float sh = 1.; // min shadow factor found so far
        for (int i = iters; i-- > 0; ) {
    		float f = (float(i) + 1.) / float(iters)
    		// must distribute the samples nonlinearly
			// to support long shadow trace distances.
			// need more samples close to the receiver.
			, ff = f * f
			, v = max((Scene(p + sht * ff).d - hd + sfuzz * f) * sc / shd / nl / ff, 0.);
			sh = min(sh, v); //sh = min(sh, v * (2.-f)); //sh *= mix(v, 1., f); //sh *= v; //
        }
        // hoisted sqrt and part of clamp out of loop
        sh = /*sqrt*/(min(sh, 1.));
        sh = pow(sh, shfalloff);
	    nl *= sh; // fake soft shadow attenuates direct lighting
    }
    //nl = max(nl, ao);
    //nl *= max(0., mix(ao, 1., .2));  // fake AO
    //nl = mix(nl, 1., ambient);  // HACK ambient floor
    //float af = min(mix(ao, 1., ambient), 1.);
    //nl = mix(nl, 1., af);
    // FIXME I still don't like how I've mixed the factors, really.
    // FIXME ambient light should have a color from the surrounding environment bounces,
    // so should be based on albedo of nearby surfaces
    // as a major HACK can use our *own* albedo since it does contribute somewhat due to interreflections
    float ah = mix(n.y, 1., .5) * ambient; // hemisphere ambient
    nl *= (1.-ambient); // leave room for ambient factor
    nl += ah * ao; // hemi ambient only where not occluded
    //nl /= (1. + ambient); // I like this mixing better
    nl = clamp(nl, 0., 1.);
    // must have some minimum lighting floor to prevent harsh black ao in shadows
    nl = mix(nl, 1., lfloor); // after clamping
//    nl = 1.; // HACK disable lighting
    return nl;
}

// Function 139
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

// Function 140
float softShadow(vec3 ro, vec3 lp, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 16; 
    
    vec3 rd = (lp-ro); // Unnormalized direction ray.

    float shade = 1.0;
    float dist = 0.05;    
    float end = max(length(rd), 0.001);
    float stepDist = end/float(maxIterationsShad);
    
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        //dist += min( h, stepDist ); // So many options here: dist += clamp( h, 0.0005, 0.2 ), etc.
        dist += clamp(h, 0.02, 0.25);
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0.001 || dist > end) break; 
    }

    // I've added 0.5 to the final shade value, which lightens the shadow a bit. It's a preference thing.
    return min(max(shade, 0.) + 0.5, 1.0); 
}

// Function 141
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.02;
  for (int j = VAR_ZERO; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05 || d > dstFar) break;
  }
  return 0.6 + 0.4 * sh;
}

// Function 142
float getDottedShadow(vec2 fragCoord)
{
    vec2 uv = fragCoord;
    uv *= mat2(cos(.8+vec4(0, 11, 33, 0)));
    uv = mod(uv*.225, 1.);
    float res = 0.;
    float shadow = readShadow(fragCoord);
    shadow = max(.65,shadow*.85) + .35*readAO(fragCoord);
    shadow = 1. - shadow;
    res = smoothstep(shadow, shadow+1., pow(length(uv-.5), 4.));
    res = smoothstep(.0, .2, pow(res, .05));
    return res;
}

// Function 143
float My_edge_shadow(vec3 surface, vec3 lig_p,float mintd, float maxdd, float k0,float k1) {
	float start_d = mintd;
	float d = 0.0;
	float mind = 1.0;
	for(int i = 0; i < 20; i++) {		
		d = obj(surface + start_d*lig_p).x;
		mind = min(mind, exp(d*k0)/k1 );
		start_d += d;
		if(start_d > maxdd) break;
	}
	return mind;
}

// Function 144
float shadow_box(vec2 fragCoord, vec4 box, float border)
{
    vec2 clamped = clamp(fragCoord, box.xy, box.xy + box.zw);
    return clamp(1.25 - length(fragCoord-clamped)*(1./border), 0., 1.);
}

// Function 145
vec3 renderShadows( in vec4 gBuffer, in vec3 ro, in vec3 rd )
{ 
    float sh = 1.0;
    
    float t = gBuffer.x;
    float m = gBuffer.w;
    
    vec3 lig = normalize( vec3(-0.6, 0.7, -0.5) );
    
    if( m > 0.0 )
    {
        vec3 pos = ro + t*rd;
        vec3 nor = Sph_To_N(gBuffer.yz);
        
        if(m != MAT_GAS)
        {
            bool unlit = false;
            vec2 uv;
            
            if(m == MAT_RINGS || m == MAT_L_RED || m == MAT_L_GREEN)
            {
                unlit = true;
            }
            
            if(!unlit)
            {
                float dif = saturate( dot( nor, lig ) );
                if( dif>0.02 ) { sh = softshadow( pos, lig, 0.01, FAR_CLIP, 7.0 ); }
            }
        }
	}
 
	return vec3( saturate(sh), sampleNebula(rd), 0.0 );
}

// Function 146
float shadow( in vec3 ro, in vec3 rd )
{
	float res = 1.0;
	for( int i=0; i<NUMSPHEREES; i++ )
        res = min( res, 8.0*sphSoftShadow(ro,rd,sphere[i]) );
    return res;					  
}

// Function 147
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<50; i++ )
    {
        float h = map( ro + rd*t ).x;
        res = min( res, usmoothstep(k*h/t) );
        t += clamp( h, 0.05, 0.2 );
        if( res<0.001 || t>maxt ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 148
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 32; 
    
    ro += n*.0011;
    vec3 rd = lp - ro; // Unnormalized direction ray.
    

    float shade = 1.;
    float t = 0.;//.0015; // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = 0; i<maxIterationsShad; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        
        // Note the ray shortening hack here. It's not entirely accurate, but reduces
        // shadow artifacts slightly for this particular stubborn distance field.
        t += clamp(d*.8, .01, .25); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}

// Function 149
vec2 effect (vec2 uv, vec2 center)
{
    vec2 relPos = uv - center;
    
    float a  = 700.0;
    int   b  = 15;
    float c  = 0.07;
    
    float module  = length(relPos);// * (1. + (fractNoise((vec2(334.,15.) + relPos)*10.)*.005) );
    
    float angle   = getAngle(relPos/module);
    
    module += (fractNoise((vec2(334.,15.) + relPos)*50.)*.002);
    module = float(floor(module*a)/a);
    module += (noise((vec2(3324.,125.) + relPos)*30.)*.0000001);
    
    ////////////////////////////////////
    float seed = rand(vec2(module) + center);
    
    int nExt = int(floor(seed * 20.0 * module))*b;
    
    float nearExtPos = rand( vec2(seed*100.,-1.*20.) ) * 2. * PI;
    for (int i = 0; i < nExt; i++)
    {
        float currentExtPos = rand( vec2(seed*100.,float(i)*20.) ) * 2.*PI;
        if (d_angle(currentExtPos, angle) < d_angle(nearExtPos, angle))
        	nearExtPos = currentExtPos;
    }
    
    //////
    
    float curSize = rand( vec2(nearExtPos*293.) ) * c / module;
    if (d_angle(nearExtPos, angle)  < curSize) 
        angle = nearExtPos;
    
         
    ///////////////////////////////
    
    angle = mod(angle, 2.*PI);
         
    return center + (module * vec2(cos(angle), sin(angle)));
}

// Function 150
float SampleShadowMap(in vec2 shadowCoords)
{
    return texture(iChannel0, shadowCoords).r;
}

// Function 151
bool intersectShadow( in vec3 ro, in vec3 rd, in float dist ) {
    lowp float t;
	
	t = iSphere( ro, rd, vec4( 1.5,1.0, 2.7,1.0) );  if( t>eps && t<dist ) { return true; }
    t = iSphere( ro, rd, vec4( 4.0,1.0, 4.0,1.0) );  if( t>eps && t<dist ) { return true; }

    return false; // optimisation: planes don't cast shadows in this scene
}

// Function 152
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.01;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.03 * d, h));
    d += min (0.05, 3. * h);
    if (sh < 0.001) break;
  }
  return 0.4 + 0.6 * sh;
}

// Function 153
float softshadow(vec3 ro,vec3 rd) 
{
    float sh=1.;
    float t=.02;
    float h=.0;
    for(int i=0;i<22;i++)  
	{
        if(t>20.)continue;
        h=map(ro+rd*t);
        sh=min(sh,4.*h/t);
        t+=h;
    }
    return sh;
}

// Function 154
float shadow(vec3 p, vec3 l) {
    float shad=1.0;    
    float dd=0.;
    for(int i=0;i<50; ++i) {
        float d=map(p);
        shad=min(shad,(abs(d)-.01)*100.);
        if(d<0.01) {
            shad=0.0;
            break;
        }
        if(dd>20.) break;
        p+=l*d;
        dd+=d;
    }
    return shad;
}

// Function 155
float getShadow(vec3 origin, vec3 destination, float soft, float time) {
    float currentDistance = OBJECT_MIN_SURFACE_DISTANCE_SHADOW;
    float maxDistance = distance(destination, origin);
    vec3 direction = normalize(destination-origin);
    rayMarchHit rayMarchHit;
    float light = 1.0;

    for(int i=0; i<RAY_MAX_STEPS_SHADOW; i++) {
        vec3 currentPoint = origin + direction*currentDistance;
        rayMarchHit = GetRayMarchHit(currentPoint, time);
        //Soft Shadows!
        light = min(light,soft * rayMarchHit.distance/currentDistance);
        currentDistance += rayMarchHit.distance;
        if(currentDistance > maxDistance) {
            break;
        } else if(rayMarchHit.distance < OBJECT_MIN_SURFACE_DISTANCE_SHADOW) {
            light = 0.0;
            break;
        }
    }

    return light;
}

// Function 156
float calcSoftshadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.0005;                 // selfintersection avoidance distance
	float h = 1.0;
    for( int i=0; i<80; i++ )         // 40 is the max numnber of raymarching steps
    {
        h = doModel(ro + rd*t).x;
        res = min( res, 64.0*h/t );   // 64 is the hardness of the shadows
        if(h <= 0.0){
            break;
        }
        t +=abs(h);
        if(t > 20.0){
            break;
        }
      //  t += max(h, 0.;
	//	t += clamp( h, 0.001, 2.0 );   // limit the max and min stepping distances
    }
    return clamp(res,0.0,1.0);
}

// Function 157
float shadow(in vec3 ro, in vec3 rd, in float mint)
{
	float res = 1.0;
    
    float t = mint;
    for( int i=0; i<12; i++ )
    {
		float h = map(ro + rd*t);
        res = min( res, 4.*h/t );
        t += clamp( h, 0.1, 1.5 );
            }
    return clamp( res, 0., 1.0 );
}

// Function 158
float castShadowRay( in vec3 ro, in vec3 rd, out vec3 oVos )
{

	vec3 pos = floor(ro);
	vec3 ri = 1.0/rd;
	vec3 rs = sign(rd);
	vec3 dis = (pos-ro + 0.5 + rs*0.5) * ri;
	vec3 ris = ri*rs;
	
	float res = 1.0;

    // detailed raymarching
    
    for( int i=0; i<DETAIL_INTER; i++ ) 
	{
		if( map(pos)>0.5 && i>0 )
		{
            float id = hash1( pos );
            vec3 mini = (pos-ro + 0.5 - 0.5*vec3(rs))*ri;
            float t = max ( mini.x, max ( mini.y, mini.z ) );
            float h = 0.0;
            vec3 ce = pos + 0.5;
            h = map2( ro+rd*t-ce,id ); t += max(0.0,h);
            h = map2( ro+rd*t-ce,id ); t += max(0.0,h);
            h = map2( ro+rd*t-ce,id ); t += max(0.0,h);
            h = map2( ro+rd*t-ce,id ); t += max(0.0,h);
            h = map2( ro+rd*t-ce,id ); t += max(0.0,h);
            h = map2( ro+rd*t-ce,id ); t += max(0.0,h);
            if( h<0.001 )
            {
                return 0.0;
                res=0.0; 
                break; 
			}
		}
		vec3 mi = step( dis.xyz, dis.yzx ); 
		vec3 mm = mi*(1.0-mi.zxy);
		dis += mm * ris;
        pos += mm * rs;
	}
	

    // coarse raymarching

	for( int i=0; i<(16-DETAIL_INTER); i++ ) 
	{
		if( map(pos)>0.5 && i>0 )
		{
            res=0.0; 
            break; 
		}
		vec3 mi = step( dis.xyz, dis.yzx ); 
		vec3 mm = mi*(1.0-mi.zxy);
		dis += mm * ris;
        pos += mm * rs;
	}
	
	oVos = pos;
	
	return res;
}

// Function 159
float shadow( in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    float t = .1;
	float h;
	
    for (int i = 0; i < 15; i++)
	{
        vec3 p =  ro + rd*t;

		h = de(p).x;
		res = min(6.*h / t, res);
		t += h;
	}
    //res += t*t*.08; // Dim over distance
    return clamp(res, .1, 1.0);
}

// Function 160
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

// Function 161
float shadow(vec3 ro, vec3 rd)
{
    float res = 0.0;
    float tmax = 1.5;
    float t = 0.001;
    for(int i=0; i<30; i++ )
    {
        float h = f(ro+rd*t);
        if( h<0.0001 || t>tmax) break;
        t += h;
    }
    if( t>tmax ) res = 1.0;
    return res;
}

// Function 162
float calcSoftshadow(in vec3 ro, in vec3 rd)
{
    float res = 1.0;
    float tmax = 12.0;  
    
    float t = 0.02;
    for( int i=0; i<30; i++ )
    {
		float h = 0.12*map(ro + rd*t, false, true).x;
        res = min( res, 24.0*h/t );
        t += clamp( h, 0.0, 0.80 );
        if( res<0.07 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 163
float SHADOW_MARCH (vec3 p) {
    p = p+sund()*.1;
    float closestDE = 1e3;
    for (float i=0.; i<35.; ++i) {
        float SDFp = SDF(p);
        if (SDFp < 1e-2) {
            return .8;
        }
        p = p+sund()*SDFp*.99;
        closestDE = min(closestDE, SDFp);
        if (SDFp > 7.) {
            break;
        }
    }
    return 1.;
}

// Function 164
float shadowDistanceField(vec3 p) {
	float loxodrome = sdLoxodrome(p - vec3(0.0, 0.0, 0.0), 2.0, 4.0, 0.075);
	float holder = 9001.0;
	#ifndef LOXODROME_ONLY
	holder = sdCylinder(p.xzy - vec3(0.0, 0.0, -1.125), vec2(0.5, 0.25));
	#endif
	return min(holder, loxodrome);
}

// Function 165
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for(int i=0; i<25; i++)
    {
    	float h = map(ro + rd*t, false).x;
        res = min(res, 4.5*h/t);
        t += clamp(h, 0.01, 0.12);
        if( h<0.001 || t>tmax ) break;
    }
    return smoothstep(0.0, 0.8, res);
}

// Function 166
float shadow(vec3 ro, vec3 rd){
    float t = 0.4;
    float d = 0.0;
    float shadow = 1.0;
    for(int iter = 0; iter < 1000; iter++){
        d = map(ro + rd * t);
        if(d < 0.0001){
            return 0.0;
        }
        if(t > length(ro - lightPos) - 0.5){
            break;
        }
        shadow = min(shadow, 128.0 * d / t);
        t += d;
    }
    return shadow;
}

// Function 167
float shadowPCF(vec3 px)
{
    // texture delta
    vec2 delta = 1./iChannelResolution[0].xy;
    
    float factor = 0.;
    // filter size
    const int r = 3;
    for(int y = -r; y <= r; y++)
    {
     	for(int x = -r; x <= r; x++)
        {
         	vec2 offset = delta * vec2(x,y);
            // count the number of shadow hits
			factor += float(texture(iChannel0,px.xy + offset).x > px.z - 0.002);
            
        }
    }
    int size = 2*r +1;
    
    int elements = size*size;
    
    // average of shadow hits
    return factor/float(elements);
}

// Function 168
float TerrainSoftShadow( in vec3 origin, in vec3 direction )
{
  float res = 2.0;
  float t = 0.0;
  float hardness = 6.0;
  for ( int i=0; i<8; i++ )
  {
    float h = TerrainDistance(origin+direction*t);
    res = min( res, hardness*h/t );
    t += clamp( h, 0.02, 0.10 );
    if ( h<0.002 ) break;
  }
  return clamp( res, 0.0, 1.0 );
}

// Function 169
float shadow_falloff( vec3 pa, vec3 pb )
{
	vec3 d = ( pa - pb );
	return 1.0 / ( 1.0 + lensqr( d ) * 0.00005 );
}

// Function 170
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

// Function 171
float segShadow( in vec3 ro, in vec3 rd, in vec3 pa, float sh )
{
    float dm = dot(rd.yz,rd.yz); // dm = 1.0 - rd.x*rd.x
    float k1 = (ro.x-pa.x)*dm;
    float k2 = (ro.x+pa.x)*dm;
    vec2  k5 = (ro.yz+pa.yz)*dm;
    float k3 = dot(ro.yz+pa.yz,rd.yz);
    vec2  k4 = (pa.yz+pa.yz)*rd.yz;
    vec2  k6 = (pa.yz+pa.yz)*dm;
    
    for( int i=0; i<4; i++ )
    {
        vec2  s = vec2(i&1,i>>1);
        float t = dot(s,k4) - k3;
        
        if( t>0.0 )
        sh = min(sh,dot2(vec3(clamp(-rd.x*t,k1,k2),k5-k6*s)+rd*t)/(t*t));
    }
    return sh;
}

// Function 172
vec3 ShadowMaskRGBCols(float x)
{
	return vec3
    (
        ShadowMaskSingleCol(x + SHADOWMASK_RCOL_OFFSET), 
        ShadowMaskSingleCol(x + SHADOWMASK_GCOL_OFFSET), 
        ShadowMaskSingleCol(x + SHADOWMASK_BCOL_OFFSET)
    );    
}

// Function 173
float shadow (in vec3 ro, in vec3 rd)
{
    float result = 1.;
    float t = .1;
    float ph = 1e10;
    for (int i = 0; i < 64; i++) {
        float h = map (ro + t * rd).x;
        if (h < .00001) return .0;
        float y = h*h/(2.*ph);
        float d = sqrt (h*h - y*y);
        result = min (result, 10.*d/max (.0, t - y));
        ph = h;
        t += h*.5;
    }

    return result;
}

// Function 174
vec3 blurSample(in vec2 uv, in vec2 xoff, in vec2 yoff)
{
    vec3 v11 = texture(iChannel0, uv + xoff).rgb;
    vec3 v12 = texture(iChannel0, uv + yoff).rgb;
    vec3 v21 = texture(iChannel0, uv - xoff).rgb;
    vec3 v22 = texture(iChannel0, uv - yoff).rgb;
    return (v11 + v12 + v21 + v22 + 2.0 * texture(iChannel0, uv).rgb) * 0.166667;
}

// Function 175
float calcShadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.01;
    for( int i=0; i<100; i++ )
    {
        vec3 pos = ro + rd*t;
        float h = mapShadow( pos ).x;
        res = min( res, 16.0*max(h,0.0)/t );
        if( h<0.0001 || pos.y>3.0 ) break;
        
        t += clamp(h,0.01,0.2);
    }
    
    return clamp(res,0.0,1.0);
}

// Function 176
float softshadow(in vec3 ro, in vec3 rd, in float k)
{
    float res = 1.0;
    float t = 0.0;
    for (int i = 0; i < SOFT_SHADOW_STEPS; ++i)
    {
		vec3 pos = ro + rd * t;
        float h = map(pos).y;
        res = min(res, k * h / t);
        if(res < 0.0001)
		{
	    	break;
    	}
        t += clamp(h, 0.0, 0.2);
    }
    return saturate(res);
}

// Function 177
float softshadow(in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k) {
    float sh = 1.0;
    float t = mint;
    float h = 0.0;
    for(int i = 0; i < 15; i++) {
        if(t > maxt) continue;
        h = map(ro + rd * t).w;
        sh = min(sh, k * h / t);
        t += h;
    }
    return sh;
}

// Function 178
float shadow( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;
    float t = NEAR;
    for(int i = 0; i<SOFTSHADOW_STEPS;i++) {
        float h = map(ro + rd*t).x;
        if( h<NEAR || t>FAR)
            break;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}

// Function 179
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 1.;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += 0.08 * (1. + 0.1 * d);
    if (sh < 0.05) break;
  }
  return 0.4 + 0.6 * sh;
}

// Function 180
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<48; i++ )
    {
        float h = density(ro + rd*t);
		h = max( h, 0.0 );
        res = min( res, k*h/t );
        t += clamp( h, 0.02, 0.5 );
		if( h<0.0001 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 181
void set_blur(float b) {
    if (b == 0.0) {
        _stack.blur = vec2(0.0, 1.0);
    } else {
        _stack.blur = vec2(
            b,
            0.0);
    }
}

// Function 182
float GrndSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 16; j ++) {
    p = ro + rd * d;
    h = p.y - GrndHt (p.xz);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += max (0.2, 0.1 * d);
    if (sh < 0.05) break;
  }
  return sh;
}

// Function 183
float softshadow(in vec3 ro, in vec3 rd, float mint, float k)
{
	float res = 1.0;
	float t = mint;
	float h = 1.0;
	for (int i = 0; i < 100; i++)
	{
		h = de(ro + rd * t);
		res = min(res, k*h / t);
		t += clamp(h, .005, .1);
	}
	return clamp(res, 0., 1.);
}

// Function 184
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

// Function 185
vec3 postEffects( in vec3 col, in vec2 uv, in float time )
{
	// vigneting
	col *= 0.5+0.5*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.5 );
	return col;
}

// Function 186
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.01;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += min (0.05, 3. * h);
    if (sh < 0.001) break;
  }
  return 0.6 + 0.4 * sh;
}

// Function 187
vec3 postEffects( in vec3 col, in vec2 uv, in float time )
{
	// gamma correction
	//col = pow( clamp(col,0.0,1.0), vec3(0.6) );
	// vigneting
	col *= 0.7+0.3*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.1 );
	return col;
}

// Function 188
float shadowcast(in vec3 ro, in vec3 rd){
    return shadowcast_pointlight(ro, rd, FAR_CLIP);
}

// Function 189
float shadowray(vec3 pos) {
    if (!RENDER_SHADOWS)
        return 1.0;

    float res = 1.0;
    float t = 0.2;
    for (int i = 0; i < 50; i++) {
		float h = sdf(pos + -LIGHT_DIR * t).x;
        res = min(res, 16.0 * h / t);
        t += clamp(h, 0.05, 0.4);
        if (res < 0.05)
            break;
    }
    return clamp(res, 0.0, 1.0);
}

// Function 190
float shadows(in vec3 ro, in vec3 rd){
    float t = 0.1;
    float res = 1.0;
    for( int i = 0; i < 16; i++ )
    {
		float h = map(ro + rd * t).x;
        res = min(res, 2.0 * h / t);
        t += clamp(h, 0.01, 0.012);
        if(t > 2.5 || h < THRESHOLD) break;
    }
    return clamp(res, 0.3, 1.0);
}

// Function 191
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for(int i=0; i<12; i++)
    {
    	float h = map(ro + rd*t).x;
        res = min(res, 10.0*h/t + 0.02*float(i));
        t+= 1.5*clamp(h, 0.01, 0.5);
        if(h<0.001 || t>tmax) break;
    }
    #ifdef show_smoke
    vec4 rsr = raymarchSmoke(ro, rd, vec3(1.), 6., true);
    return clamp(min(res, 1. - rsr.a*0.9), 0.0, 1.0);
    #else
    return clamp(res, 0.0, 1.0);
    #endif
}

// Function 192
vec3 postEffects( in vec3 col, in vec2 uv, in float time )
{
	// gamma correction
	// col = pow( clamp(col,0.0,1.0), vec3(0.45) );
	// vigneting
	col *= 0.7+0.3*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.15 );
	return col;
}

// Function 193
float SampleShadowMap(in vec3 point, in float shadowMapBias)
{
	const float shadowMapMaxDistance = 23.0;
    
    float shadow = 1.0;
    
    vec3 pointToLight = point - gLightPosition;
    float distanceToLight = length(pointToLight);
    
    vec2 shadowCoords = CalculateShadowMapUV(gLightViewMatrix, point, (iResolution.x / iResolution.y));
    
    if((shadowCoords.x > EPSILON && shadowCoords.y > EPSILON) && (shadowCoords.x < (1.0 - EPSILON) && shadowCoords.y < (1.0 - EPSILON)))
    {
#if POISSON_SAMPLING        
        for(int i = 0; i < 4; ++i)
        {
            const float poissonDiskSpread = 0.00125;
            float shadowMap = SampleShadowMap(shadowCoords + (SamplePoissonDisk(i) * poissonDiskSpread));
            
            float shadowMapDistance = (shadowMap * shadowMapMaxDistance);

            if(shadowMapDistance < (distanceToLight - shadowMapBias))
            {
                shadow -= 0.25;
            }   
        }
#else
        float shadowMap = SampleShadowMap(shadowCoords);
        float shadowMapDistance = (shadowMap * shadowMapMaxDistance);

        if(shadowMapDistance < (distanceToLight - shadowMapBias))
        {
            shadow = 0.0;
        }     
#endif
    }
    return shadow;
}

// Function 194
float effectClock( in vec2 p ) {
    float value = 0.0;
    float temp = length(p);
    value += smoothstep(0.75, 0.80, temp) * (1.0-smoothstep(0.80, 0.85, temp));
    vec2 push = p;
    p *= rot(-floor(mod(iDate.w, 60.0)) / 60.0 * 2.0 * PI);
   	float sbar = 0.0;
    if (p.x < 0.02 && p.x > -0.02 && p.y > -0.05 && p.y < 0.7) sbar = 1.0;
    value += sbar;
    p = push;
    p.y += 0.2;
    p.x -= 0.03;
    float minutes = mod(iDate.w / 60.0, 60.0);
    float hour = mod(iDate.w / (60.0*60.0), 24.0);
    if (hour > 13.0) hour -= 12.0;
    p.y *= 0.7;
    p.x += 0.7;
    float print = 0.0;
    if ( hour >= 10.0 )
    	print += SampleDigit(floor(hour/10.0), p*3.3);
    p.x -= 0.3;
    print += SampleDigit(floor(mod(hour, 10.0)), p*3.3);
    push = p;
    p.y = -p.y + 0.22;
    p.x = -p.x + 0.44;
    print += SampleDigit(4.0, p*4.5);
    value = print * 0.5 + value * (0.5+0.5*(1.0-print));
    p = push;
    p.x -= 0.5;
    print = SampleDigit(floor(minutes/10.0), p*3.3);
    p.x -= 0.3;
    print += SampleDigit(floor(mod(minutes, 10.0)), p*3.3);
    value = print * 0.5 + value * (0.5+0.5-0.5*print);
    return value;
}

// Function 195
float calcSoftshadow(in vec3 ro, in vec3 rd, in float mint, in float tmax) {
  float res = 1.0;
  float t = mint;
  for (int i = 0; i < 16; i++) {
    float h = map(ro + rd * t).x;
    res = min(res, 8.0 * h / t);
    t += clamp(h, 0.02, 0.10);
    if (h < 0.001 || t > tmax) break;
  }
  return clamp(res, 0.0, 1.0);
}

// Function 196
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

// Function 197
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = dstScene( ro + rd*t );
        res = min( res, 32.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 198
float softShadow(in vec3 ro, in vec3 rd )
{
    // real shadows	
    float res = 1.0;
    float t = 0.001;
	for( int i=0; i<80; i++ )
	{
	    vec3  p = ro + t*rd;
        float h = p.y - textureLod( iChannel0, fract(p.xz), 0.0).w*.15;
		res = min( res, 20.*h/t );
		t += h;
		if( res<0.001) break;
	}
	return clamp( res, 0.1, 1.0 );
}

// Function 199
float boxSoftShadow( in vec3 ro, in vec3 rd, in mat4 txx, in vec3 rad, in float sk ) 
{
	vec3 rdd = (txx*vec4(rd,0.0)).xyz;
	vec3 roo = (txx*vec4(ro,1.0)).xyz;

    vec3 m = 1.0/rdd;
    vec3 n = m*roo;
    vec3 k = abs(m)*rad;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max( max( t1.x, t1.y ), t1.z );
	float tF = min( min( t2.x, t2.y ), t2.z );
	
    // fake soft shadow
    if( tF<0.0) return 1.0;
    float sh = clamp(0.3*sk*(tN-tF)/tN,0.0,1.0);
    return sh*sh*(3.0-2.0*sh);
}

// Function 200
float softShadow(vec3 pos,vec3 nor,vec3 dir)
{
    float minValue = 0.0001;
    float maxValue = 10.0;
    float t = 0.0;
   
    pos = pos + nor * 0.1;
    
    for(int i = 0; i < 10; ++ i)
    {
        vec2 hit = map(pos + dir * t);
        if(hit.x < minValue) break;
        
        t += hit.x;
        if(t > maxValue) break;
    }
    
    return clamp(t / maxValue,0.0,1.0);
}

// Function 201
float ShadowMaskRows(vec2 uv)
{
    // Stagger rows
    uv.x *= 0.5;
    uv.x -= round(uv.x);
    if(uv.x < 0.0)
        uv.y += 0.5;
    
    return Grille(uv.y, -SHADOWMASK_HORIZGAPWIDTH, SHADOWMASK_HORIZARDNESS);
}

// Function 202
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

// Function 203
float shadow(vec3 rayPos, vec3 lightDir, vec3 normal)
{
    float sVal = 1.0; //initial shadow value.sVal gets mutiplied with diffuse lighting
    float sEPS = 0.01;// our shadow epsilon/precision value
    vec3 ro = rayPos + (normal * sEPS); //reduces self-shadow artifacts since we are starting the march slightly above our surface
    vec3 rd = lightDir; // we are now marching from our surface to light source.
    float sDist; //initializing our shadow distance value
    
    
      for(int i = 0; i < 36; i++)
      {
        sDist = scene(ro); //comparing shadow ray position with our scene
        if(sDist < sEPS)
        {
            sVal = 0.0;
            break;
        }
        ro += rd * sDist;
     }
    
    return sVal;
}

// Function 204
vec3 mapShadow( in vec3 pos )
{
    float h = terrain( pos.xz );
    float d = pos.y - h;
    vec3 res = vec3( d, MAT_GROUND, 0.0 );
    
    res = mapGrass(pos,h,res);
    res = mapMoss(pos,h,res);

    vec3 m1 =  pos - mushroomPos1;
    vec3 m2 = (pos - mushroomPos2).zyx;
    if( length2(m2.xz) < length2(m1.xz) ) m1 = m2;
	res = mapMushroom(m1, res);


    vec3 q = worldToLadyBug(pos);
    vec3 d3 = mapLadyBug(q, res.x*4.0); d3.x/=4.0;
    if( d3.x<res.x ) res = d3;

    return res;
}

// Function 205
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for(int i=0; i<50; i++)
    {
    	float h = map(ro + rd*t).x;
        res = min(res, 10.0*h/t + 0.02*float(i));
        t += 0.8*clamp(h, 0.01, 0.35);
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 206
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

// Function 207
float apprSoftShadow(vec3 ro, vec3 rd, float mint, float tmax, float w)
{
 	float t = mint;
    float res = 1.0;
    for( int i=0; i<256; i++ )
    {
     	float h = map(ro + t*rd);
        res = min( res, h/(w*t) );
    	t += clamp(h, 0.005, 0.50);
        if( res<-1.0 || t>tmax ) break;
    }
    res = max(res,-1.0); // clamp to [-1,1]

    return 0.25*(1.0+res)*(1.0+res)*(2.0-res); // smoothstep
}

// Function 208
float calcSoftshadow(in vec3 ro, in vec3 rd)
{
    float res = 1.0;
    float tmax = 12.0;  
    
    float t = 0.02;
    for( int i=0; i<30; i++ )
    {
		float h = map(ro + rd*t, false).x;
        res = min( res, 27.0*h/t );
        t += clamp( h, 0., 0.80 );
        if( res<0.0005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 209
float shadows(vec3 ro, vec3 rd, float b)
{
	float t = 0.01;
	float res = 1.0;
	for(int i = 0; i < 32; i++)
	{
		float g = geo(ro+rd*t).w;
		res = min(res, b * g / t); //---Traditional soft shadows
		t += g;
		if(g < 0.0005|| g > 3.0)
        {
			break;
        }
	}
	return clamp(res,0.1,1.0);
}

// Function 210
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<50; i++ )
    {
		float h = scene( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.01 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 211
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

// Function 212
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

// Function 213
float groundShadow(vec2 p)
{
	vec2 fp = floor(p);
	vec2 pf = smoothstep(0.0, 1.0, fract(p));
	
	return mix( mix(groundSolidity(fp), groundSolidity(fp + ON), pf.x), 
			   mix(groundSolidity(fp + ON.yx), groundSolidity(fp + ON.xx), pf.x), pf.y);
}

// Function 214
float shadow(in vec3 ro, in vec3 rd, in float tmin, in float tmax) {
    float t = tmin;
    for( int i=0; i<10; ++i )
    {
		float h = map( ro + rd*t ).x;
        if( h<tmin || t>tmax) break;
        t+=h;//float(i)/30.0;
    }
    return clamp(1.0-(tmax-t)/(tmax-tmin), 0.0, 1.0);
}

// Function 215
float softshadow(vec3 ro, vec3 rd, float k )
{ 
    float s=1.0,h=0.0; 
    float t = 0.01;
    for(int i=0; i < 30; ++i)
    { 
        h=f(ro+rd*t); 
        if(h<0.001)return 0.02; 
        s=min(s, k*max(h, 0.0)/t); 
        t+=h; 
    } 
    return s; 
}

// Function 216
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
	float h = 1.0;
    for( int i=0; i<48; i++ )
    {
        h = map(ro + rd*t).x;
        res = min( res, k*h/t );
		t += clamp( h, 0.005, 0.5 );
    }
    return clamp(res,0.0,1.0);
}

// Function 217
float castSoftShadowRay(vec3 pos, vec3 lightPos) {
	const float pi = 3.14159265359;
	const float k = 0.005;
	float res = 1.0;
	vec3 rayDir = normalize(lightPos - pos);
	float maxDist = length(lightPos - pos);
	
	vec3 rayPos = pos + 0.01 * rayDir;
	float distAccum = 0.1;
	
	for (int i = 1; i <= MAX_SECONDARY_RAY_STEPS; i++) {
		rayPos = pos + rayDir * distAccum;
		float dist = shadowDistanceField(rayPos);
		float penumbraDist = distAccum * k;
		res = min(res, inverseMix(-penumbraDist, penumbraDist, dist));
		distAccum += (dist + penumbraDist) * 0.5;
		distAccum = min(distAccum, maxDist);
	}
	res = max(res, 0.0);
	res = res * 2.0 - 1.0;
	return (0.5 * (sqrt(1.0 - res * res) * res + asin(res)) + (pi / 4.0)) / (pi / 2.0);
}

// Function 218
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

// Function 219
void effectDiff(inout vec4 col, vec2 coord)
{
    vec2 g=getGrad(coord,.5);
    col=getCol(coord+g.xy*1.5*iResolution.x/600.);
}

// Function 220
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    #ifdef DISABLE_SHADOWS
        return 1.;
    #endif
    float res = 1.0;
    float t = mint;
    float ph = 1e10;
    
    for( int i=ZERO; i<256; i++ )
    {
        float h = map( ro + rd*t ).d;
        res = min( res, 10.0*h/t );
        t += h;
        if( res<0.0001 || t>tmax ) break;
        
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 221
float maskingAndShadowing(float dotNL, float dotNV, float roughness)
{
    //Smith Joint GGX(Height-Correlated Masking and Shadowing)
    float a = roughness * roughness;
    float a2 = a * a;
    float lambdaNL = (-1.0 + sqrt(1.0 + a2 * (1.0 / (dotNL * dotNL) - 1.0))) / 2.0;
    float lambdaNV = (-1.0 + sqrt(1.0 + a2 * (1.0 / (dotNV * dotNV) - 1.0))) / 2.0;
    return 1.0 / (1.0 + lambdaNL + lambdaNV);
}

// Function 222
float shadow(vec3 ro, vec3 rd)
{
    float res = 0.0;
    float tmax = 1.0;
    float t = 0.001;
    for(int i=0; i<30; i++ )
    {
        float h = f(ro+rd*t);
        if( h<0.0001 || t>tmax) break;
        t += h;
    }
    if( t>tmax ) res = 1.0;
    return res;
}

// Function 223
float shadow(vec3 rayPos, vec3 rayDir, float rayMin, float rayMax, float k)
{
    float attenuation = 1.0;
    
    for(float dist = rayMin; dist < rayMax; )
    {
        vec3 pos = rayPos + rayDir * dist;
        Result mapResult = map(pos, rayDir);
        dist += mapResult.dist;
        
        if(!mapResult.isOvershoot)
            attenuation = min(attenuation, k * mapResult.dist / dist);
        
        if(mapResult.dist < 1e-3)
            break;
    }
    
    return attenuation;
}

// Function 224
float shadow(vec3 o)
{
    float mint=L0_str;
    float maxt=L0_end;
    float k = L0_sft*2.;
    float res = 1.;
    float ph = 1e20;
    float t=mint;
    for( int i=ZERO; i < 64+CHARM; i++)
    {
        float h = sdScene(o + ldir*t).x;
        if(abs(h)<MIN_DIST) return 0.;

        res = min( res, k*h/t);
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.0,t-y));
        ph = h;
        t += h;

        if(t >= maxt) break;
    }
    return smoothstep(.5, .51, res);
}

// Function 225
float My_milk_shadow(vec3 surface, vec3 lig_p,float mintd, float maxdd, float k0,float k1,float k3) {
	float start_d = mintd;
	float d = 0.0;
	float mind = 1.0;
	for(int i = 0; i < 20; i++) {		
		d = obj(surface + start_d*lig_p).x;
		mind = min(mind, abs(log(d*k0+k3))/k1 );
		start_d += d;
		if(start_d > maxdd) break;
	}
	return mind;
}

// Function 226
float ObjSShadow (vec3 ro, vec3 rd)
{
  vec3 p, cIdP;
  float sh, d, h;
  sh = 1.;
  cIdP = vec3 (-99.);
  d = 0.1;
  for (int j = 0; j < 16; j ++) {
    p = ro + d * rd;
    cId = floor (p / bSize);
    if (cId.x != cIdP.x || cId.y != cIdP.y || cId.z != cIdP.z) {
      SetEngConf ();
      cIdP = cId;
    }
    h = ObjDf (p - bSize * (cId + 0.5));
    sh = min (sh, smoothstep (0., 0.1 * d, h));
    d += 0.3;
    if (sh < 0.05) break;
  }
  return sh;
}

// Function 227
vec3 mapShadow(in vec3 pos) {
	float h = terrain(pos.xz);
	float d = pos.y - h;
	vec3 res = vec3(d, MAT_GROUND, 0.0);

	res = mapGrass(pos, h, res);
	res = mapMoss(pos, h, res);

	vec3 m1 = pos - mushroomPos1;
	vec3 m2 = (pos - mushroomPos2).zyx;
	if (length2(m2.xz) < length2(m1.xz)) m1 = m2;
	res = mapMushroom(m1, res);


	vec3 q = worldToLadyBug(pos);
	vec3 d3 = mapLadyBug(q, res.x*4.0); d3.x /= 4.0;
	if (d3.x < res.x) res = d3;

	return res;
}

// Function 228
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	rgb = sqrt(rgb);

	// Then...
	#define CONTRAST 1.1
	#define SATURATION 1.2
	#define BRIGHTNESS 1.15
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);

	// Vignette...

	rgb *= .5 +.5* pow(100.0*xy.x*xy.y*(1.0-xy.x)* (1.0-xy.y), 0.4);	

        
	return clamp(rgb, 0.0, 1.0);
}

// Function 229
float softShadow(in vec3 pos, in vec3 ld, in float ll, float mint, float k)
{
  const float minShadow = 0.25;
  float res = 1.0;
  float t = mint;
  vec3 col;
  float ref;
  float trans;
  vec3 absorb;
  for (int i=0; i<24; i++)
  {
    float distance = distanceField(pos + ld*t, col, ref, trans, absorb);
    res = min(res, k*distance/t);
    if (ll <= t) break;
    if(res <= minShadow) break;
    t += max(mint*0.2, distance);
  }
  return clamp(res,minShadow,1.0);
}

// Function 230
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = 0; j < 25; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 1., 20. * h / d));
    d += min (0.3, 3. * h);
    if (h < 0.001) break;
  }
  return sh;
}

// Function 231
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

// Function 232
float ShadowMaskSingleCol(float x)
{
    return Grille(x, -SHADOWMASK_VERTGAPWIDTH, SHADOWMASK_VERTHARDNESS);
}

// Function 233
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.6 + 0.4 * sh;
}

// Function 234
float DisplayShadowMap(in vec3 cameraPosition, in vec3 cameraDirection, in float glowThreshold)
{
    // Determine our camera info
    Ray cameraRay = Ray(cameraPosition, cameraDirection); 
    return Intersect(cameraRay);
}

// Function 235
float getShadowCoeff(in vec3 p, in vec3 nv) {
    float tHit = 0.0;
    vec3 curPos = p;
    float shadowCoeff = 0.0;

    for (int k = 0; k < RAY_STEPS_SHADOW; k++) {
        float sdStep;
        sdGeometry(curPos, sdStep);

        float curLightPercent = abs(sdStep)/(0.1*tHit);
        shadowCoeff = max(shadowCoeff, 1.0-curLightPercent);

        if (abs(sdStep) < MIN_DIST) {
            shadowCoeff = 1.0;
            break;
        }

        curPos += sdStep * nv;
        tHit += sdStep;
        if (tHit > MAX_DIST) {
            break;
        }
    }

    return clamp(shadowCoeff, 0.0, 1.0);
}

// Function 236
float shadowOther( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;
    float t = NEAR;
    for(int i = 0; i<10;i++) {
        float h = mapOther(ro + rd*t).x;
        if( h<0.001 || t>FAR)
            return 0.0;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}

// Function 237
float calcSoftshadow(in vec3 ro, in vec3 rd)
{
    float res = 5.0;
    float tmax = 6.0;  
    
    float t = 0.2;
    for( int i=0; i<20; i++ )
    {
		vec2 mr = map(ro + rd*t, false, true);
        float h = mr.x;
        res = min(res, 8.4*h/t);
        t += clamp(h*(mr.y==float(SCREEN_OBJ)?0.17:1.), 0.0, 0.36);
        if(res<0.1 || t>tmax) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 238
float softShadowCapsule( in vec3 ro, in vec3 rd, in vec3 a, in vec3 b, in float r )
{
    const float k = 16.0;
    vec3 t = dCapsule( ro, rd, a, b, r );
    return clamp( k*t.x/max(t.z,0.0001), 0.0, 1.0 );
}

// Function 239
float Shadow(Ray ray, vec3 light)
{
    float depth = Raymarch(ray).w;
    if(depth < length(light))
    {
        return 0.1;
    }
    return 1.0;
}

// Function 240
float boxSoftShadow( in vec3 ro, in vec3 rd, in mat4 txx, in vec3 rad, in float sk ) 
{
	vec3 rdd = (txx*vec4(rd,0.0)).xyz;
	vec3 roo = (txx*vec4(ro,1.0)).xyz;

    vec3 m = 1.0/rdd;
    vec3 n = m*roo;
    vec3 k = abs(m)*rad;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max( max( t1.x, t1.y ), t1.z );
	float tF = min( min( t2.x, t2.y ), t2.z );
	
    if( tN>tF || tF<0.0) return 1.0;
	return 0.0;
}

// Function 241
float raymarchShadows(in vec3 ro, in vec3 rd, float tmin, float tmax) {
    float sh = 1.0;
    float t = tmin;
    for(int i = 0; i < 50; i++) {
        vec3 p = ro + rd * t;
        float d = intersectSpheres(p, true).y;
        sh = min(sh, 16.0 * d / t);
        t += 0.5 * d;
        if (d < (0.001 * t) || t > tmax)
            break;
    }
    return sh;
}

// Function 242
float Effect4(vec2 coords, float radius, float cutoffMax, float cutoffMin)
{
    float dist = length(coords);
    if (dist <= cutoffMax && dist >= cutoffMin)
    {
        if (dist < radius)
        {
            return 1.0;
        }
    }
    return 0.0;
}

// Function 243
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

// Function 244
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

// Function 245
vec4 blurSample(sampler2D s, vec2 p, float a)
{
    vec2 u = vec2(+0.0, +0.1)*a;
    vec2 d = vec2(+0.0, -0.1)*a;
    vec2 l = vec2(+0.1, +0.0)*a;
    vec2 r = vec2(-0.1, +0.0)*a;
    vec4 v = texture(s, p);
	v+= texture(s, p+u+l)+texture(s, p+u)+texture(s, p+u+r)+
        texture(s, p+l)+texture(s, p)+texture(s, p+r)+
        texture(s, p+d+l)+texture(s, p+d)+texture(s, p+d+r);
    return v/10.0;
}

// Function 246
float calcSoftshadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.0005;                 // selfintersection avoidance distance
	float h = 1.0;
    for( int i=0; i<40; i++ )         // 40 is the max numnber of raymarching steps
    {
        h = doModel(ro + rd*t);
        res = min( res, 64.0*h/t );   // 64 is the hardness of the shadows
		t += clamp( h, 0.02, 2.0 );   // limit the max and min stepping distances
    }
    return clamp(res,0.0,1.0);
}

// Function 247
float shadow (in vec3 p, in vec3 lPos) {
    float lDist = distance (p, lPos);
    vec3 lDir = normalize (lPos - p);
    int dummy = 0;
    float dist = march (p, lDir, dummy);
    return dist < lDist ? .1 : 1.;
}

// Function 248
vec2 ts_shadow_sample_ao( TrnSampler ts, sampler2D ch, vec3 x )
{
#if WITH_TRN_SHADOW
    vec4 lookup = ts_shadow_lookup( ts, ch, x );
    return vec2( ts_shadow_eval( lookup.xy, length(x) ), lookup.z );
#else
    return vec2(1);
#endif
}

// Function 249
float ObjCSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  vec2 gIdP;
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  gIdP = vec2 (-99.);
  for (int j = 0; j < 30; j ++) {
    p = ro + d * rd;
    gId = PixToHex (p.xz / hgSize);
    if (gId.x != gIdP.x || gId.y != gIdP.y) {
      gIdP = gId;
      SetGrndConf ();
    }
    h = ObjCDf (p);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 250
float sphSoftShadow( in vec3 ro, in vec3 rd, in vec4 sph, in float k )
{
    vec3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    
#if 0
    // physically plausible shadow
    float d = sqrt( max(0.0,sph.w*sph.w-h)) - sph.w;
    float t = -b - sqrt( max(h,0.0) );
    return (t<0.0) ? 1.0 : smoothstep(0.0, 1.0, 2.5*k*d/t );
#else
    // cheap but not plausible alternative
    return (b>0.0) ? step(-0.0001,c) : smoothstep( 0.0, 1.0, h*k/b );
#endif    
}

// Function 251
vec4 ts_shadow_lookup( TrnSampler ts, sampler2D ch, vec3 x )
{
    vec2 uv = ts_uv_centered( ts, x );
    if( ts_is_uv_safe( uv ) )
    {
		vec2 res = vec2( textureSize( ch, 0 ) );
    	vec2 aspect = vec2( res.y / res.x, 1 );
    	float aspect_shadow = min( 0.666666667, res.x / res.y - 1. );
        uv = ( .5 * uv + .5 ) * aspect_shadow * aspect + vec2( aspect.x, 0 );
 		return textureLod( ch, uv, 0. );
    }
    else
        return vec4(0);
}

// Function 252
float shadow (in vec3 p, in vec3 lPos) {
    float lDist = distance (p, lPos);
    vec3 lDir = normalize (lPos - p);
    float dist = march (p, lDir);
    return dist < lDist ? .1 : 1.;
}

// Function 253
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k ) {
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ ) {
		if( t>maxt ) break;
        float h = map( ro + rd*t );
        res = min( res, k*h/t );
        t += 0.03 + h;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 254
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	rgb = pow(rgb, vec3(0.45));

	// Then...
	#define CONTRAST 1.3
	#define SATURATION 1.3
	#define BRIGHTNESS 1.2
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);

	// Vignette...
	rgb *= .5+0.5*pow(180.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.3 );	

	return clamp(rgb, 0.0, 1.0);
}

// Function 255
float shadow(vec3 ro, vec3 rd) {
	// based off of http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
    float res = 1.0;
    float ph = 1e20;
    float tmin = 0.1;
    float tmax = 2.0;
    float k = 8.0;

    for (float t = tmin; t < tmax;) {
        vec2 h = sceneSDF(ro + rd * t);
        if (h.x < EPSILON) {
            return 0.0;
        }
        float y = (h.x * h.x) / (2.0 / ph);
        float d = sqrt((h.x * h.x) - (y * y));
        res = min(res, k * d / max(0.0, t - y));
        ph = h.x;
        t += h.x;
    }
    return res;
}

// Function 256
bool shadow_hit(const in ray r, const in float t_min, const in float t_max) {
    hit_record rec;
    rec.t = t_max;
   
    ray r_ = ray_rotate_y(ray_translate(r, vec3(130,0,65)), -18./180.*3.14159265359);  
    if (hitable_hit(hitable(vec3(82.5), vec3(82.5)),r_,t_min,rec.t,rec)) 
        return true;
    
	r_ = ray_rotate_y(ray_translate(r, vec3(265,0,295)), 15./180.*3.14159265359);  
    if (hitable_hit(hitable(vec3(82.5,165,82.5), vec3(82.5,165,82.5)),r_,t_min,rec.t,rec)) 
        return true;
  
    return false;
}

// Function 257
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<48; i++ )
    {
		float h = dstScene( ro + rd*t );
        res = min( res, 32.0*h/t );
        t += clamp( h, 0.02, 0.40 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 258
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

// Function 259
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = VAR_ZERO; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.03 * d, h));
    d += clamp (h, 0.02, 0.3);
    if (sh < 0.05) break;
  }
  return 0.6  + 0.4 * sh;
}

// Function 260
float SoftShadow( in vec3 origin, in vec3 direction )
{
  float res =1., t = 0.0, h=0.;
  vec3 rayPos = vec3(origin+direction*t);    
  #ifdef HIGH_QUALITY
    for ( int i=0; i<30+min(0, iFrame); i++ )
  {
    h = MapPlanet(rayPos).x;
    #ifdef HIGH_QUALITY_BELT
    h = min(h, MapRocks(rayPos));
    #endif
      
    res = min( res, 3.5*h/t );
    t += clamp( h, 0.01, 250.1);
    if ( h<0.005 ) break;
    rayPos = vec3(origin+direction*t);
  }
  #else
    for ( int i=0; i<10+min(0, iFrame); i++ )
    {
      h = MapPlanet(rayPos).x;
      res = min( res, 8.5*h/t );
      t += clamp( h, 0.01, 100.1);
      if ( h<0.005 ) break;
      rayPos = vec3(origin+direction*t);
    }
  #endif


    return clamp( res, 0.0, 1.0 );
}

// Function 261
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

// Function 262
float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k)
{
    float dt = (maxt - mint) / float(shadowSteps);
    float t = mint;
	t += hash(ro.z*574854.0 + ro.y*517.0 + ro.x)*0.1;
    float res = 1.0;
    for( int i=0; i<shadowSteps; i++ )
    {
        float h = scene(ro + rd*t);
		if (h < hitThreshold) return 0.0;	// hit
        res = min(res, k*h/t);
        //t += h;
		t += dt;
    }
    return clamp(res, 0.0, 1.0);
}

// Function 263
float GrndSShadow (vec3 ro, vec3 rd)
{
  float sh = 1.;
  float d = 0.01;
  for (int i = 0; i < 80; i++) {
    vec3 p = ro + rd * d;
    float h = p.y - GrndHt (p.xz, 0);
    sh = min (sh, 20. * h / d);
    d += 0.5;
    if (h < 0.001) break;
  }
  return clamp (sh, 0., 1.);
}

// Function 264
float shadow(vec3 O, vec3 D) {
    float shaded = 1.;
    
    float L = 0.;
    float d = 0.;
    
    for (int i = 1; i < MAX_STEPS / 3; ++i) {
        d = sceneLow(O + D*L);
        L += d * clamp(L * .6, 0.45, .8);
        
        if (d < THRESHOLD*L)
            return 0.;
        
        shaded = min(shaded, 2. * d / L);
    }
    
    return shaded;
}

// Function 265
float SoftShadow(in vec3 ro, in vec3 rd) {
    float res = 1.0, h, t = .005+hash13(ro)*.02;
    float dt = .01;
    for( int i=0; i<32; i++ ) {
		h = map( ro + rd*t );
		res = min( res, 10.*h/t );
		t += dt;
        dt+=.0025;
        if (h<PRECISION) break;
    }
    return clamp(res, 0., 1.);
}

// Function 266
float castShadowRay( in vec3 ro, in vec3 rd, out vec3 oVos, out vec3 oDir, out vec3 nor)
{
	vec3 pos = floor(ro);
	vec3 ri = 1.0/rd;
	vec3 rs = sign(rd);
	vec3 dis = (pos-ro + 0.5 + rs*0.5) * ri;
	
	float res = -1.0;
	vec3 mm = vec3(0.0);
	for( int i=0; i<52; i++ )  
	{
		if( VoxelMap(pos) ) { res=1.0; break; }
		mm = step(dis.xyz, dis.yxy) * step(dis.xyz, dis.zzx);
		dis += mm * rs * ri;
        pos += mm * rs;
	}

	nor = -mm*rs;
	vec3 vos = pos;
	
    // intersect the cube	
	vec3 mini = (pos-ro + 0.5 - 0.5*vec3(rs))*ri;
	float t = max ( mini.x, max ( mini.y, mini.z ) );
	
	oDir = mm;
	oVos = vos;

	return t*res;
}

// Function 267
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<50; i++ )
    {
        float h = map(ro + rd*t).x;
        res = min( res, k*h/t );
		t += clamp( h, 0.5, 1.0 );
		if( h<0.001 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 268
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

// Function 269
float getShadow(vec3 origin, vec3 sunDir)
{
    _shadowMarch = true;
    _highresTerrain = false;

    float t = 0., s = 1.0;

    for(int i = 0; i < S/2; i++)
    {
        float d = scene(origin + sunDir * t);
       
      	if (t > D) break;
        
        t += d;
        s = min(s,d/t*K);
    }
    
    _highresTerrain = true;
    _shadowMarch = false;

    return clamp(s,0.5,1.);
}

// Function 270
float Shadow(in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    float t = 2.0;
	float h = 0.0;
    
	for (int i = 0; i < 20; i++)
	{
		h = Map(ro + rd * t).x;
		res = min(h / t, res);
		t += h*.02+.35;
	}
	
    return clamp(res, 0.0, 1.0);
}

// Function 271
float eliShadow( in vec3 ro, in vec3 rd, in Ellipsoid sph, in float k )
{
    vec3 oc = ro - sph.cen;
    
    vec3 ocn = oc / sph.rad;
    vec3 rdn = rd / sph.rad;
    
    float a = dot( rdn, rdn );
	float b = dot( ocn, rdn );
	float c = dot( ocn, ocn );

    if( b>0.0 || (b*b-a*(c-1.0))<0.0 ) return 1.0;
    
    return 0.0;
}

// Function 272
float calcShadowArlo( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;
    
    // check bounding volume first
    vec2 bv = sphIntersect( ro, rd, vec4(-0.5,0.5,0.0,3.4) );
    if( bv.y>0.0 )
    {
        float t = 0.01;
        for( int i=0; i<32; i++ )
        {
            float h = mapArloSimple(ro + rd*t );
            res = min( res, smoothstep(0.0,1.0,k*h/t) );
            t += clamp( h, 0.04, 0.5 );
		    if( res<0.01 ) break;
        }
    }
    return clamp(res,0.0,1.0);
}

// Function 273
float shadow( in vec3 ro, in vec3 rd )
{
    float res = 0.0;
    
    float tmax = 12.0;
    
    float t = 0.001;
    for(int i=0; i<80; i++ )
    {
        float h = map(ro+rd*t);
        if( h<0.0001 || t>tmax) break;
        t += h;
    }

    if( t>tmax ) res = 1.0;
    
    return res;
}

// Function 274
float softshadow(vec3 ro, vec3 rd, float k ){ 
     float akuma=1.0,h=0.0; 
	 float t = 0.01;
     for(int i=0; i < 50; ++i){ 
         h=f(ro+rd*t); 
         if(h<0.001)return 0.02; 
         akuma=min(akuma, k*h/t); 
 		 t+=clamp(h,0.01,2.0); 
     } 
     return akuma; 
}

// Function 275
float softshadow(vec3 ro, vec3 rd, float mint, float tmax) {
	float res = 1.0;
	float t = mint;

	for (int i = 0; i < 16; i++) {
		float h = map(ro + rd * t).x;

		res = min(res, 8.0 * h / t);
		t += clamp(h, 0.02, 0.10);

		if (h < 0.001 || t > tmax) {
			break;
		}
	}
	return clamp(res, 0.0, 1.0);
}

// Function 276
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

// Function 277
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

// Function 278
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
	float h = 1.0;
    for( int i=0; i<32; i++ )
    {
        h = map(ro + rd*t).x;
        res = min( res, k*h/t );
		t += clamp( h, 0.005, 0.1 );
    }
    return clamp(res,0.0,1.0);
}

// Function 279
bool MapShadow( sampler2D iData, MapDataInfo mapInfo, vec3 vRO, vec3 vRD, float fMinDist, float fMaxDist, float expand )
{
#if USE_AABB_TREE     
    const int STACK_DEPTH = 10;
    const int STACK_MAX = STACK_DEPTH - 1;    
    int node_stack[STACK_DEPTH];
    int node_stack_ptr = 0;  
    
    node_stack[node_stack_ptr] = 0;
    node_stack_ptr++;
    
    while ( node_stack_ptr > 0 )
    {		
        node_stack_ptr--;
        int iNodeIndex = node_stack[node_stack_ptr];
        AABBNode node = MapData_ReadAABBNode( iData, iNodeIndex );
        
        if ( RayBoxIntersect( node.bounds, vRO, vRD, expand ) )
        {        
            if ( node.childB_brushCount < 0 )
            {
                int iFirstBrushIndex = node.childA_brushIndex;
                int iBrushCount = abs(node.childB_brushCount);
                for ( int iBrushArrayIndex=0; iBrushArrayIndex<iBrushCount; iBrushArrayIndex++ )
                {          
                    int iBrushIndex = iFirstBrushIndex + iBrushArrayIndex;               

                    iBrushIndex = clamp( iBrushIndex, 0, mapInfo.brushCount - 1 );	                                
                    
			        MapDataBrush brushInfo = MapData_ReadBrush( iData, iBrushIndex );
                    
                    C_HitInfo enterInfo = C_HitInfo( 0.0f, 0, 0 );
                    float fExitT = kFarClip;        
                    
                    for ( int iPlaneIndex=0; iPlaneIndex<brushInfo.planeCount; iPlaneIndex++ )
                    {            
                        BrushPlane( iData, vRO, vRD, iBrushIndex, brushInfo.planeStart + iPlaneIndex, expand, enterInfo, fExitT );            
                    }

                    if( enterInfo.fClosestT >= fMinDist && enterInfo.fClosestT <= min(fExitT, fMaxDist) )
                    {
                        return true;
                    }                    
                }
            }
            else
            {

                if ( node_stack_ptr < STACK_MAX )
                {
                    node_stack[node_stack_ptr] = node.childA_brushIndex;
                    node_stack_ptr++;
                }

                if ( node_stack_ptr < STACK_MAX )
                {
                    node_stack[node_stack_ptr] = node.childB_brushCount;
                    node_stack_ptr++;            
                }           
            }
        }
    }
    
    return false;
#else
    for ( int iBrushIndex=0; iBrushIndex<mapInfo.brushCount; iBrushIndex++ )
    {               
        MapDataBrush brushInfo = MapData_ReadBrush( iData, iBrushIndex );
        
		C_HitInfo enterInfo = C_HitInfo( 0.0f, -1, -1 );
		float fExitT = kFarClip;        
        
	    for ( int iPlaneIndex=0; iPlaneIndex<brushInfo.planeCount; iPlaneIndex++ )
    	{            
            BrushPlane( iData, vRO, vRD, iBrushIndex, brushInfo.planeStart + iPlaneIndex, enterInfo, fExitT );            
        }
        
        if( enterInfo.fClosestT >= fMinDist && enterInfo.fClosestT <= min(fExitT, fMaxDist) )
        {
            return true;
        }
    }
    
    return false;    
#endif
}

// Function 280
float scene_raycast_object_shadows( Ray ray )
{
    float result = 1.;
    float t = SCN_ZFAR;
    vec3 albedo, N;
    for( int i = 0, n = int( memload( iChannel0, ADDR_DATASIZES, 0 ).w ); i < n; ++i )
    {
        SceneObj obj = so_load( iChannel0, ADDR_SCENE_OBJECTS + ivec2( i, 0 ) );
        Ray localray = Ray( ( ray.o - obj.r ) * obj.B, ray.d * obj.B );
        switch( int( obj.tybr.x ) )
        {
        case SCNOBJ_TYPE_PRIMITIVE:
        result *= scene_obj_primitive( obj, localray, t, albedo, N );
        break;
        }
    }
    return max( 0., result );
}

// Function 281
float doShadow(vec3 ro, vec3 rd, float len) {
    vec3 p2,n;
    float dmin, dmax;
    bool intersect = box(ro, rd, Bounds, dmin, dmax, n);
    if (intersect) {   
        float val,valId,d=min(.25,dmax); //, d=.5+.5*dd*hash1(ro.x+ro.y+ro.z);
        // suivre cette direction et ajouter un truc a ao si on rtouve un obstacle avant dMax
        for(int k=0;k<40; k++) {
             if (d>=dmax) break;
            p2 = ro + min(d,dmax) *rd;
            val = valueAt(p2);
            if (val >= MinVal && val < MaxVal) {
                // on a rencontrÃ© un obstacle,
                return clamp(mix(0.,1.,(min(d,dmax)/len)),0.,1.); // plus il est pres plus il a de l'influence
            }
            d+=.25*dd;

        }
    }
    return 1.;
}

// Function 282
float Shadow( in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    float t = 0.01;
	float h;
	
    for (int i = 0; i < 8; i++)
	{
		h = Map( ro + rd*t );
		res = min(2.5*h / t, res);
		t += h*.5+.05;
	}
    return max(res, 0.0);
}

// Function 283
float shadow(vec3 o, vec3 d){
    o += norm(o) * 0.001;
	float len = 0.0, lev = 1.0;
	for(float t = 0.0; t < 32.0; t++){
		float di = dist(o + d * len);
		if (di < 0.001){ return 0.5;}
		lev = min(lev, di  * 24.0 / min(len, 1.0));
		len += di;
	}
	return max(0.5, lev) ;
}

// Function 284
float calcSoftshadow( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;
    float t = 0.0;
	float h = 1.0;
    for( int i=0; i<20; i++ )
    {
        h = map(ro + rd*t);
        res = min( res, k*h/t );
		t += clamp( h, 0.01, 1.0 );
		if( h<0.0001 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 285
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

// Function 286
float getShadowTerm(vec3 r)
{
	float light_amount = 1.0;
    
    if( curr_iteration == 0 )
    { 
        for( int i=0; i < 64; ++i )
        {
            vec3 light_pos = g_light.xyz;
			float ofsx = fmod(float(i), SRO);
        	float ofsy = (float(i)*SRO)/SRO;
    		float ofsz = fmod(float(i)*SRO*SRO, SRO);
    		vec3 offset = 1.0 * vec3(ofsx, ofsy, ofsz);
    		light_pos += offset;
        
        	if( shadowRayCast(r, light_pos) ) 
        	{
				light_amount -= 1.0 / 64.0;
        	}
  	  	}
    }
    else
    {
        // LOLHAX!!!!
        light_amount = 0.33;
        if( shadowRayCast(r, g_light.xyz) )
        {
			light_amount = 0.0;
        }
    }
   
    return light_amount;
}

// Function 287
column planetEffect2(float x, float w/*idth*/, float r/*adius*/)
{
    if (x >= w && x <= W - w)
        return column(0., H);
    x -= (W / 2.);
    float x0 = w - (W / 2.);
    float y = sqrt(r * r - x0 * x0) - sqrt(r * r - x * x);
    float R = r + H;
    float h = H + sqrt(R * R - x0 * x0) - sqrt(R * R - x * x);
    return column(y, h - y);
}

// Function 288
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

// Function 289
float calc_soft_shadows(in vec3 ro, in vec3 rd, in float tmin, in float tmax, const float k) {
    float res = 1.0;
    float t = tmin;
    for (int i = 0; i < 50; i++) {
        float h = sdf(ro + rd * t);
        res = min(res, k * h / t);
        t += clamp(h, 0.02, 0.20);
        if (res < 0.005 || t > tmax) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

// Function 290
float GetCloudShadow(const in vec3 pos)
    {
        float cloudyChange = abs(1.0 - (0.1 + (cloudy - 0.15) * (0.8/0.6)));
        vec2 cuv = pos.xz + sunDirection.xz * (100.0 - pos.y) / sunDirection.y;
        float cc = 0.1 + 0.9 * smoothstep(0.0, cloudyChange, 
                                          texture( iChannel1, 0.0008 * cuv 
                                                    + 0.005 * iTime).x);
	
        return cc;
    }

// Function 291
float softShadow(vec3 ro, vec3 lp, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 20; 
    
    vec3 rd = (lp-ro); // Unnormalized direction ray.

    float shade = 1.0;
    float dist = 0.05;    
    float end = max(length(rd), 0.001);
    //float stepDist = end/float(maxIterationsShad);
    
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        //dist += min( h, stepDist ); // So many options here: dist += clamp( h, 0.0005, 0.2 ), etc.
        dist += clamp(h, 0.01, 0.25);
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0.001 || dist > end) break; 
    }

    // I've added 0.5 to the final shade value, which lightens the shadow a bit. It's a preference thing.
    return min(max(shade, 0.) + 0.2, 1.0); 
}

// Function 292
float softShadow(in vec3 o,in vec3 i,float n,float m){
 float r=1.,t=n,k=.05;const float s=100.,u=1.+.78/sqrt(s*.5);
 //not perfect, decent for s range[10.100]
 for(float j=0.;j<s;j++){float h=df(o+i*t)/(s+1.);
  if(h<0.)return 0.;r=min(r,(k*h/t));
  t=t*u+h;//mod by ollj allows for smaller [n]
     if(t > m)break;}return r*s*79./length(i);}

// Function 293
float lightPointDiffuseSpecularShadow(vec3 pos, vec3 lightPos, vec3 cameraPos, vec3 normal) {
	vec3 lightDir = normalize(lightPos - pos);
	float lightDist = length(lightPos - pos);
	float color = dot(normal, lightDir) / square(lightDist);
	if (color > 0.01) {
		vec3 cameraDir = normalize(cameraPos - pos);
		color += dot(cameraDir, lightDir);
		color *= castShadowRay(pos, lightPos, 0.001);
	}
	return max(0.0, color);
}

// Function 294
float castShadowRay(vec3 pos, vec3 lightPos, float treshold) {
	vec3 dir = normalize(pos - lightPos);
	float maxDist = length(lightPos - pos);
	vec3 rayPos = lightPos;
	float distAccum = 0.0;
	for (int i = 0; i < MAX_SECONDARY_RAY_STEPS; i++) {
		float dist = distanceField(rayPos);
		rayPos += dist * dir;
		distAccum += dist;
	}
	if (distAccum > maxDist - treshold) return 1.0;
	else return 0.0;
}

// Function 295
float softshadow(const vec3 origin, in vec3 dir, in float mint, in float tmax, float k)
{
	float res = 1.0;
	float t = mint;
	for( int i=0; i<16; i++ )
	{
		float h = distFunc( origin + dir*t );
		res = min( res, k*h/t );
		t += clamp( h, 0.02, 0.10 );
		if( h<0.001 || t>tmax ) break;
	}
	return clamp( res, 0.0, 1.0 );
}

// Function 296
float LightingShadow(vec3 o, vec3 d, float minDist, float maxDist)
{
    float t = minDist;
        
    for (int i = 0; i < 24; i++)
    {
		float h = Scene(o + d * t).dist;
        t += h;
        
        if (h < 0.005) 
            return 0.0;
        
        if (t > maxDist)
            break;
    }
	
    return 1.0;
}

// Function 297
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test 
    // machine, anyway.
    const int maxIterationsShad = 24; 
    
    ro += n*.0015;
    vec3 rd = lp - ro; // Unnormalized direction ray.
    

    float shade = 1.;
    float t = 0.;//.0015; // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the 
    // lowest number to give a decent shadow is the best one to choose. 
    for (int i = 0; i<maxIterationsShad; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        // Subtle difference. Thanks to IQ for this tidbit.
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); 
        // So many options here, and none are perfect: dist += min(h, .2), 
        // dist += clamp(h, .01, stepDist), etc.
        t += clamp(d, .01, .25); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}

// Function 298
float shadow(vec3 p, vec3 l, float tMax, float k)
{
    float ret = 1.;
    float ph = 1e20;
    for (float t = 0.; t < tMax;) {
        // Fudge around volume artifacts near small objects
        float h = scene(p + l * t) + (rnd() - 0.4) / 10.;
        if (h < .01)
            return .0;
        float y = h * h / (2. * ph);
        float d = sqrt(h * h - y * y);
        ret = min(ret, k * d / max(0., t - y));
        ph = y;
        t += h;
    }
    return ret;
}

// Function 299
float softShadow(vec3 a, vec3 u, float k)
{
    float r = 1.0;
	vec3 p = a;
	float lambda = object_lipschitz();
	float depth = 0.0;
	float step = 0.0;
	for(int i = 0; i < Steps; i++)
	{
		float v = SphereTracedObject(p);
		if (v > 0.0)
			return 0.0;
        r = min(r, k * (v / AmbiantEnergy) / depth);
		
		step = max(abs(v) / lambda, Epsilon);
		depth += step;
		
		if(depth > RayMaxLength)
			return r;
		
		p += step * u;
	}
	return r;
}

// Function 300
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

// Function 301
float card_shadow(vec2 p) {
    float d = 0.;
    d = SS(-0.01, 0.02 + zv, sdBox(p, vec2(0.08, 0.12)));
    return d;
}

// Function 302
float sphShadow( in vec3 ro, in vec3 rd, in vec4 sph )
{
    vec3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    return step( min( -b, min( c, b*b - c ) ), 0.0 );
}

// Function 303
float shadow(vec3 from, vec3 increment)
{
	const float minDist = 1.0;
	
	float res = 1.0;
	float t = 1.0;
	for(int i = 0; i < SHADOW_ITERATIONS; i++) {
        float h = distf(from + increment * t);
        if(h < minDist)
            return 0.0;
		
		res = min(res, SHADOW_SMOOTHNESS * h / t);
        t += SHADOW_STEP;
    }
    return res;
}

// Function 304
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.01;
  for (int j = VAR_ZERO; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 305
float calcShadow( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;

    float t = 0.1;
    for( int i=0; i<32; i++ )
    {
        vec3 pos = ro + rd*t;
        float h = DistanceField(pos, length(pos));
        res = min( res, smoothstep(0.0,1.0,8.0*h/t) );
        t += clamp( h, 0.05, 10.0 );
		if( res<0.01 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 306
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

// Function 307
float ShadowFactor(in vec3 ro, in vec3 rd) {
	vec3 p0 = vec3(0.0);
    vec3 p1 = vec3(0.0);
    
    IRayAABox(ro, rd, 1.0/rd, scmin, scmax, p0, p1);
    vec2 dir = normalize(rd.xz);
    float rs = map(ro.xz, dir).x;
    p0 = ro + rd*0.02;
    
    float sf = rd.y / length(rd.xz);

    float m = -1e5;
    
    vec4 v;
    const int max_steps = 32;
    for (int i = max_steps; i > 0; --i) {
        
        if (dot((p1 - p0), rd) < 0.0) return 1.0;
  
        v = map(p0.xz, dir);
        
        m = v.w;
        if (p0.y < m) break;// return 0.0;
        
        p0 += rd*(length(vec2(v.x, v.x*sf)) + 0.01);
    }
    vec3 i1 = vec3(1,0,0);
    vec3 i2 = vec3(0,1,0);
    vec3 j1 = (p1 - p0);
    return (1.0-smoothstep(1.5,0.1,v.x));
    
}

// Function 308
float shadow(in vec3 ro, in vec3 rd, in float mint, in float tmax)
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<20; i++ )
    {
		float h = map(ro + rd*t);
        res = min( res, 4.*h/t );
        t += clamp( h, 0.2, 1.5 );
            }
    return clamp( res, 0.0, 1.0 );

}

// Function 309
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	

	// Then...
	#define CONTRAST 1.08
	#define SATURATION 1.5
	#define BRIGHTNESS 1.5
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);
	// Noise...
	//rgb = clamp(rgb+Hash(xy*iTime)*.1, 0.0, 1.0);
	// Vignette...
	rgb *= .5 + 0.5*pow(20.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.2);	

    rgb = pow(rgb, vec3(0.47 ));
	return rgb;
}

// Function 310
SurfaceInteraction calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, int technique,out float shadowValue ) {
    vec3 p = ro;
    float res = 1.0;
    float t = mint;
    float ph = 1e10; // big, such that y = 0 on the first iteration
    vec2 obj = vec2(0.);
    SurfaceInteraction interaction = SurfaceInteraction(-1., rd, vec3(0.), vec3(0.), vec3(0.), vec3(0.), -10.);
    
    for( int i=0; i<RAY_MARCH_STEPS; i++ )
    {
        obj = map(p);
        
        if(obj.y == LIGHT_ID) {
           break;
        }
        //obj = map( ro + rd*t );
        float h = obj.x;

        // traditional technique
        if( technique==0 )
        {
            res = min( res, 10.0*h/t );
        }
        // improved technique
        else
        {
            // use this if you are getting artifact on the first iteration, or unroll the
            // first iteration out of the loop
            float y = (i==0) ? 0.0 : h*h/(2.0*ph); 

            //float y = h*h/(2.0*ph);
            float d = sqrt(h*h-y*y);
            res = min( res, 10.0*d/max(0.0,t-y) );
            ph = h;
        }
        
        t += h;
        p += rd * h;
        
        if( res<0.0001 || t>tmax ) break;
        obj.y = 0.;
        
    }
    interaction.id = obj.y;        
    interaction.point = p;
    interaction.normal = calculateNormal(interaction.point);
    interaction.objId = obj.y;
    
    shadowValue = clamp( res, 0.0, 1.0 );
    return interaction;//clamp( res, 0.0, 1.0 );
}

// Function 311
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 312
float ObjSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  vec2 gIdP;
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  gIdP = vec2 (-99.);
  for (int j = 0; j < 30; j ++) {
    p = ro + d * rd;
    gId = PixToHex (p.xz / hgSize);
    if (gId.x != gIdP.x || gId.y != gIdP.y) {
      gIdP = gId;
      SetPngConf ();
    }
    h = ObjDf (p);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += clamp (h, 0.05, 0.3);
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 313
float calcSoftshadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.5;                 // selfintersection avoidance distance
	float h = 1.0;
    for( int i=0; i<40; i++ )         // 40 is the max numnber of raymarching steps
    {
        h = doModel(ro + rd*t);
        res = min( res, 64.0*h/t );   // 64 is the hardness of the shadows
		t += clamp( h, 0.02, 2.0 );   // limit the max and min stepping distances
    }
    return clamp(res,0.0,1.0);
}

// Function 314
float calcSoftshadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.0005;                 // selfintersection avoidance distance
	float h = 1.0;
    for( int i=0; i<5; i++ )         // 40 is the max numnber of raymarching steps
    {
        h = doModel(ro + rd*t);
        res = min( res, 64.0*h/t );   // 64 is the hardness of the shadows
		t += clamp( h, 0.02, 2.0 );   // limit the max and min stepping distances
    }
    return clamp(res,0.0,1.0);
}

// Function 315
float softshadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t=0.01;
    for(int i=0; i<128; i++)
    {
        float h = map(ro + rd*t);
        if( h<0.001 )
            return 0.0;
        res = min( res, 200.*h/t );
        t += h;
        if(t>2.)
            break;
    }
    return res;
}

// Function 316
float softshadow(vec3 pos, vec3 ldir, vec3 playerPos) {
#if USE_SHADOWS
    float res = 1.0;
    float t = 0.01;
    for(int i = 0; i < 16; i++) {
        float h = map(pos - ldir*t, junkMatID, playerPos, true, true);
        res = min(res, 7.0*h/t);
        t += clamp(h, 0.05, 5.0);
        if (h < EPS) break;
    }
    return clamp(res, 0.0, 1.0);
#else
    return 1.0;
#endif
}

// Function 317
float3 getShadowTransmittance(float3 cubePos, float sampledDistance, float stepSizeShadow)
{
    float3 shadow = float3(1.0);
    float3 Ldir = normalize(Lpos-cubePos);
    for(float tshadow=0.0; tshadow<sampledDistance; tshadow+=stepSizeShadow)
    {
        float3 cubeShadowPos = cubePos + tshadow*Ldir;
        float densityShadow = getDensity(cubeShadowPos);
        shadow *= exp(-densityShadow * extinction * stepSizeShadow);
    }
    return shadow;
}

// Function 318
float shadow(vec3 P, vec3 lightPos, float lightRad, vec3 occluderPos, float occluderRad)
{
	float radA = lightRad;
	float radB = occluderRad;
	
	vec3 vecA = lightPos - P;
	vec3 vecB = occluderPos - P;
	
	float dstA = sqrt(dot(vecA, vecA));
	float dstB = sqrt(dot(vecB, vecB));
	
	if (dstA - radA / 2.0 < dstB - radB) return 1.0;
	
	float sinA = radA / dstA;
	float sinB = radB / dstB;
	
	float cosA = sqrt(1.0 - sinA * sinA);
	float cosB = sqrt(1.0 - sinB * sinB);
	
	if (cosA * dstA < cosB * dstB) return 1.0;
	
	vec3 dirA = vecA / dstA;
	vec3 dirB = vecB / dstB;
	
	float cosG = dot(dirA, dirB);
	
	if (cosG < cosA * cosB - sinA * sinB) return 1.0;
	
	float sinG = length(cross(dirA, dirB));
	
	float cscA = dstA / radA;
	float cscB = dstB / radB;
	
	float cosTheta = clamp((cosB - cosA * cosG) * cscA / sinG, -1.0, 1.0);
	float cosPhi = clamp((cosA - cosB * cosG) * cscB / sinG, -1.0, 1.0);
	
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
	float sinPhi = sqrt(1.0 - cosPhi * cosPhi);
	
	float theta = acos(cosTheta);
	float phi = acos(cosPhi);
	
	float unoccluded = theta - cosTheta * sinTheta 
					 + (phi - cosPhi * sinPhi)
					 * cosG * sinB * sinB / (sinA * sinA);
	
	return 1.0 - unoccluded / PI;
}

// Function 319
float contactShadow( vec2 uv, vec4 gbuffer, float radius, vec2 fragCoord, vec2 seed )
{    
    float depth = gbuffer.w;
    
    // calculate the screen space light direction by reverse-projection
    vec2 Ldir = fragCoordForDir(L, iResolution.xy, camera_time_last());
    
    int iseed = iFrame + int(floor(seed.x * 1000.));
    float jitter = halton(iseed, 3);
    
    float shw = 0.0;
    float div = 1.0 / float(COUNT_SHADOW);

    vec2 Ldelta = radius * Ldir * div;

    for(int i = 0; i < COUNT_SHADOW; i++)
    {
        float fi = float(i);
        vec2 off = (fi+jitter) * Ldelta;

        vec2 sampleUv = (fragCoord.xy + off)/iResolution.xy;
        vec4 sampleGBuf = texture(iChannel0, sampleUv);
        float depthDelta = depth - sampleGBuf.w;

        if (depthDelta > 0.0)
        {
            // fade out shadows from contact point
            shw += clamp(1.0 - depthDelta / 8.0, 0.0, 1.0);
        }
    }
    
    return clamp(1.0 - shw, 0.0, 1.0);
}

// Function 320
float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k, bool detail)
{
    float res = 1.0;
    float ph = 1e20;
    for( float t=mint; t<maxt; )
    {
        float h = sdf(ro + rd*t, detail);
        if( h<0.001 )
            return 0.0;
        float y = h*h/(2.5*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.001,t-y) );
        ph = h;
        t += h;
    }
    return res;
}

// Function 321
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

// Function 322
vec3 GetShadowFactor(in vec3 rayOrigin, in vec3 rayDirection, in int maxSteps, in float minMarchSize)
{
    float t = 0.0f;
    vec3 shadowFactor = vec3(1.0f);
    float signedDistance = 0.0;
    bool enteredVolume = false;
    for(int i = min(0, iFrame); i < maxSteps; i++)
    {         
        float marchSize = max(minMarchSize, abs(signedDistance));
        t += marchSize;

        vec3 position = rayOrigin + t*rayDirection;

        signedDistance = QueryVolumetricDistanceField(position, iTime);
        if(signedDistance < 0.0)
        {
            // Soften the shadows towards the edges to simulate an area light
            float softEdgeMultiplier = min(abs(signedDistance / 5.0), 1.0);
            shadowFactor *= BeerLambert(WaterAbsorption * softEdgeMultiplier / WaterColor, marchSize);
            enteredVolume = true;
        }
        else if(enteredVolume)
        {
            // Optimization taking advantage of the shape of the water. The water isn't
            // concave therefore if we've entered it once and are exiting, we're done
            break;
        }
    }
    return shadowFactor;
}

// Function 323
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

// Function 324
float shadow(vec3 ro, vec3 rd, float k) {
    float marchDist = 0.001;
    float boundingVolume = 25.0;
    float darkness = 1.0;
    float threshold = 0.001;

    for(int i = 0; i < 30; i++) {
        if(marchDist > boundingVolume) continue;
        float h = map(ro + rd * marchDist).dist;
        // TODO [Task 7] Modify the loop to implement soft shadows
        darkness = min(darkness, k*h/marchDist);
        marchDist += h * 0.7;
    }
    return darkness;
}

// Function 325
float shadow( in vec3 start, in vec3 ldir, in float p )
{    
	float t = EPSILON;
	float res = 1.0;
    for ( int i = 0; i < 128; ++i )
    {
        float d = dist( start + ldir * t );
        if ( d < EPSILON*.1 )
            return 0.0;
		
		res = min( res, p * d / t );
        t += d;
		
		if ( t > MAX_DEPTH )
			break;
    }
    return res;
}

// Function 326
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for(int i=0; i<20; i++)
    {
    	float h = map(ro + rd*t, false, false).x;
        res = min(res, 4.5*h/t);
        t += clamp(h, 0.01, 0.25);
        if( h<0.001 || t>tmax ) break;
    }
    return smoothstep(0.0, 0.8, res);
}

// Function 327
float shadow(vec2 uv, vec2 ro, vec2 rd) {
    float lim = 0.0005;
    float res = -1.0;
    float inc = lim * 2.0;
    float t = inc;
    float maxt = length(ro - uv);
    
    if (map(uv, ro) < 0.0) return 0.0;
    
    for (int i = 0; i < STEPS; i++) {
        if (t >= maxt) return -1.0;
        float d = map(uv - rd * t, ro);
        if (d <= 0.0) return 0.0;
        
        t = min(t + d * 0.2, maxt);
        res = t;
    }
    
    return res;
}

// Function 328
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
	float h = 1.0;
    for( int i=0; i<32; i++ ) {
        h = 0.15*mapTerrain(ro + rd*t);
        res = min( res, k*h/t );
		t += clamp( h, 0.02, 2.0 );
		
		if( h<0.0001 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 329
float shadows(vec3 _ro, vec3 _rd, float _near, float _far, float t)
{
    float d = _near;
    for(int i = 0;i < SHADOW_STEPS; ++i)
    {
        vec3 p = _ro+_rd*d;
        // eval scene
        vec2 t = raymarch_main_scene_normals(p, t);
        if(abs(t.x)<_near || d>_far)
            break;
        d += t.x*0.5;
    }
    return d;
}

// Function 330
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 331
vec4 gaussianBlur(sampler2D buffer, vec2 uv)
{
    vec4 col = vec4(0.);
    
 	vec2 offsets[9] = vec2[](
        vec2(-texelOffset.x,  texelOffset.y),  // top-left
        vec2( 			0.,   texelOffset.y),  // top-center
        vec2( texelOffset.x,  texelOffset.y),  // top-right
        vec2(-texelOffset.x,  			 0.),  // center-left
        vec2( 			0.,			 	 0.),  // center-center
        vec2( texelOffset.x,  	 		 0.),  // center-right
        vec2(-texelOffset.x,  -texelOffset.y), // bottom-left
        vec2( 			0.,   -texelOffset.y), // bottom-center
        vec2( texelOffset.x,  -texelOffset.y)  // bottom-right    
    );
    
    for(int i = 0; i < 9; i++)
    {
        col += textureLod(buffer, uv + offsets[i], 0.) * kernel[i];
    }
    
    return col;
}

// Function 332
float castShadowRay( in vec3 ro, in vec3 rd, in float time )
{
    float res = 1.0;
    
    ivec2 hid = hexagonID(ro.xz);
    
    const float k3 = 0.866025;
    const vec2 n1 = vec2( 1.0,0.0);
    const vec2 n2 = vec2( 0.5,k3);
    const vec2 n3 = vec2(-0.5,k3);
    
    float d1 = 1.0/dot(rd.xz,n1);
    float d2 = 1.0/dot(rd.xz,n2);
    float d3 = 1.0/dot(rd.xz,n3);
    float d4 = 1.0/rd.y;
    
    float s1 = (d1<0.0)?-1.0:1.0;
    float s2 = (d2<0.0)?-1.0:1.0;
    float s3 = (d3<0.0)?-1.0:1.0;
    float s4 = (d4<0.0)?-1.0:1.0;

    ivec2 i1 = ivec2( 2,0); if(d1<0.0) i1=-i1;
    ivec2 i2 = ivec2( 1,1); if(d2<0.0) i2=-i2;
    ivec2 i3 = ivec2(-1,1); if(d3<0.0) i3=-i3;

    vec2 c1 = (vec2(-s1,s1)-dot(ro.xz,n1))*d1;
    vec2 c2 = (vec2(-s2,s2)-dot(ro.xz,n2))*d2;
    vec2 c3 = (vec2(-s3,s3)-dot(ro.xz,n3))*d3;

    // traverse regular grid (2D)	
	for( int i=0; i<8; i++ ) 
	{
		vec2  ce = hexagonCenFromID( hid );
        float he = 0.5*map(ce, time);
                
        vec2 t1 = c1 + dot(ce,n1)*d1;
        vec2 t2 = c2 + dot(ce,n2)*d2;
        vec2 t3 = c3 + dot(ce,n3)*d3;
        vec2 t4 = (vec2(1.0-s4,1.0+s4)*he-ro.y)*d4;
        
        float tN = max(max(t1.x,t2.x),max(t3.x,t4.x));
        float tF = min(min(t1.y,t2.y),min(t3.y,t4.y));
        if( tN < tF && tF > 0.0)
        {
            res = 0.0;
            break;
		}
        
             if( t1.y<t2.y && t1.y<t3.y ) hid += i1;
        else if( t2.y<t3.y )              hid += i2;
        else                              hid += i3;
	}

	return res;
}

// Function 333
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

// Function 334
float calcShadow(vec3 p, vec3 ld) {
	// Thanks iq.
	float s = 1.,
	      t = 1.;
	for (float i = 0.; i < 30.; i++) {
		float h = sdTies(p + ld * t).d;
		s = min(s, 30. * h / t);
		t += h;
		if (s < .001 || t > 1e2) break;
	}

	return clamp(s, 0., 1.);
}

// Function 335
float computeSoftShadow(vec3 ro, vec3 rd, float tmin, float tmax, float k) { return 1.0; }

// Function 336
float calcShadow( vec3 samplePos, vec3 lightDir, SpotLight light)
{	
	float dist, originDist;
	float result = 1.0;
	float lightDist = length(light.position-samplePos);
	
	vec3 pos = samplePos+(lightDir*(EPSILON+FEATURE_BUMP_FACTOR));
	
	for(int i = 0; i < MAX_SHADOW_STEPS; i++)
	{
		dist = getDist(pos);
		pos+=lightDir*dist;
		originDist = length(pos-samplePos);
		if(dist < EPSILON)
		{
			return 0.0;
		}
		if(originDist >= lightDist || originDist >= MAX_DEPTH)
		{
			return result;
		}
		if( originDist < lightDist )
		{
			result = min( result, lightDist*light.penumbraFactor*dist / originDist );
		}
	}
	return result;
}

// Function 337
float shadows(in vec3 ro, in vec3 rd){
    float t = 0.1;
    float res = 1.0;
    for( int i = 0; i < 32; i++ )
    {
		float h = map(ro + rd * t, false).x;
        res = min(res, 16.0 * h / t);
        t += clamp(h, 0.01, 0.10);
        if(t > 2.5 || h < THRESHOLD) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Function 338
float test_shadow( vec2 xy, float height)
{
    vec3 r0 = vec3(xy, height);
    vec3 rd = normalize( light - r0 );
    
    float hit = 1.0;
    float t   = 0.001;
    for (int j=1; j<25; j++)
    {
        vec3 p = r0 + t*rd;
        float h = height_map( p.xy );
        float height_diff = p.z - h;
        if (height_diff<0.0)
        {
            return 0.0;
        }
        t += 0.01+height_diff*.02;
        hit = min(hit, 2.*height_diff/t); // soft shaddow   
    }
    return hit;
}

// Function 339
float calc_aa_blur(float w) {
    vec2 blur = _stack.blur;
    w -= blur.x;
    float wa = clamp(-w*AA, 0.0, 1.0);
    float wb = clamp(-w / blur.x + blur.y, 0.0, 1.0);    
	return wa * wb; //min(wa,wb);    
}

// Function 340
float
shadow( in vec3 start, in vec3 dir )
{
    float ret = 1.0;
    float c = step( mod( iTime, 4.0 ), 2.0 );
    float t = 0.02, t_max = 16.0;
    MPt mp;
    
    for ( int it=0; it!=16; ++it )
    {
        vec3 here = start + dir * t;
        mp = map( here );
        ret = min( ret, 8.0*mp.distance/t);
        if ( mp.distance < ( T_EPS * t ) || t > t_max )
        {
        	break;
        }
        
        float inc;
        // NOTE(theGiallo): this is to sample nicely the twisted things
        inc = c * mp.distance * 0.4;
		inc += ( 1.0 - c ) * clamp( mp.distance, 0.02, 0.1 );
        t += inc;
    }
    if ( t > t_max )
    {
        t = -1.0;
    }
    if ( c == 0.0 ) return 1.0 - clamp( ret, 0.0, 1.0 );

    if ( t < 0.0 )
    {
        return 0.0;
    }
    //return 1.0;
    ret = 1.0 / pow(1.0 - 1e-30 + max( mp.distance, 1e-30 ), 5.0 );
    float th = 0.1;
    return smoothstep( 0.0, 1.0, ( ret*1.1 - th ) / (1.0-th) );
}

// Function 341
vec4 textureCrtEffect(vec2 uv)
{
    vec2 sz = VIDEO_RES;
    vec2 rsz = 1.0/sz;
    vec2 vy= uv * sz.xy;  
    vec2 ed = (uv-0.5)*2.0;
    float edg = dot(ed,ed);
       
#if WOBBLE
    float s = fract(uv.y - iTime*0.25);
    float q = WOBBLINESS * 0.004 * (0.05 * sin(s) + 0.2 * sin(vy.y*0.1+iTime*5.0) + 0.1*sin(vy.y*0.2+iTime*301.0));

    uv = fract(uv+vec2(q,0));    
#endif

    float tt;
#if BORDER
    tt = clamp(cos((uv.x-0.5)*PI*0.99)*22.0/BORDERNESS,0.0,1.0) * 
         clamp(cos((uv.y-0.5)*PI*0.99)*22.0/BORDERNESS,0.0,1.0);
#else
    tt = 1.0;
#endif
    
    vec4 tex = textureLowRes(uv, sz) * tt;
    tex = vec4((tex.r * 0.2989 + tex.g * 0.5870 + tex.b * 0.1140)); /* to grayscale */

#if COLOR == 2
    vec2 abr = edg * vec2(-1.0,-1.0) * rsz * 0.55 * ABERRATION;
    vec2 abg = edg * vec2( 1.0, 1.0) * rsz * 0.55 * ABERRATION;
    vec2 abb = edg * vec2( 1.0,-1.0) * rsz * 0.55 * ABERRATION;

    vec4 texr = textureLowRes(uv+abr, sz) * tt;
    vec4 texg = textureLowRes(uv+abg, sz) * tt;
    vec4 texb = textureLowRes(uv+abb, sz) * tt;
    
    tex = mix(tex, vec4(texr.r, texg.g, texb.b, 1), SATURATION);
#elif COLOR == 1
    vec4 texr = textureLowRes(uv, sz) * tt;
    vec4 texg = textureLowRes(uv, sz) * tt;
    vec4 texb = textureLowRes(uv, sz) * tt;
    
    tex = mix(tex, vec4(texr.r, texg.g, texb.b, 1), SATURATION);
#endif

    vec4 zero = vec4(0.0);
    vec4 tx = tex;
    
#if NOISE
    tx = mix(tx,temporalNoise(uv, sz), 0.1*NOISYNESS);
#endif
    
#if SCANLINES==2
    float t = sin(iTime*PI*VIDEO_RATE) > 0.0 ? 0.5 : 0.0;
    float yy = 0.5+0.5*sin(TWO_PI*(vy.y + t));
    tx = lerp(tx, zero, yy*0.5);
#elif SCANLINES==1
    float yy = 0.5+0.5*sin(TWO_PI*vy.y);
    tx = lerp(tx, zero, yy*0.5);
#endif
    
#if VSCAN
    tx *= (0.9 + 0.25 * s);
#endif

#if RGBGRID
    float fr = sin(vy.x * TWO_PI) * sin(vy.y * TWO_PI);
    float fg = sin((vy.x+0.5) * TWO_PI) * sin((vy.y-0.25) * TWO_PI);
    float fb = sin((vy.x+0.25) * TWO_PI) * sin((vy.y+0.25) * TWO_PI);
    
    tx.r = tx.r * (0.9 + 0.1 * fr);
    tx.g = tx.g * (0.9 + 0.1 * fg);
    tx.b = tx.b * (0.9 + 0.1 * fb);
#endif

    return tx;
}

// Function 342
vec3 effect(vec2 uv) 
{
    vec2 z = 8.*uv;
    float t = iTime, d = 1./dot(z,z);
   
    vec4 col =
        // color
        vec4(d*3.,.5,0,0)*
        // stripes
        sin(atan(z.y,z.x)*30.+d*99.+4.*t)*
        // rings
        sin(length(z*d)*20.+2.*t)*
        // depth
        max(dot(z,z)*.4-.4,0.);

    
    return col.rgb;
}

// Function 343
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
        float h = map2( ro + rd*t ).d;
        res = min( res, 5.0*h/t );
        t += clamp( h, 0.02, 0.2 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.2, 1.0 );
}

// Function 344
float calcShadowTerrain( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;

    float t = 0.1;
    for( int i=0; i<32; i++ )
    {
        vec3 pos = ro + rd*t;
        float h = mapTerrain(pos, length(pos)).x;
        res = min( res, smoothstep(0.0,1.0,8.0*h/t) );
        t += clamp( h, 0.05, 10.0 );
		if( res<0.01 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 345
float GetShadow( const in vec3 vPos, const in vec3 vNormal, const in vec3 vLightDir, const in float fLightDistance )
{
    #ifdef ENABLE_SHADOWS
		C_Ray shadowRay;
		shadowRay.vDir = vLightDir;
		shadowRay.vOrigin = vPos;
		const float fShadowBias = 0.05;
		shadowRay.fStartDistance = fShadowBias / abs(dot(vLightDir, vNormal));
		shadowRay.fLength = fLightDistance - shadowRay.fStartDistance;
	
		C_HitInfo shadowIntersect;
		Raymarch(shadowRay, shadowIntersect, 32, kNoTransparency);
		
		float fShadow = step(0.0, shadowIntersect.fDistance) * step(fLightDistance, shadowIntersect.fDistance );
		
		return fShadow;          
    #else
    	return 1.0;
    #endif
}

// Function 346
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

// Function 347
void applyEffect(inout vec2 coord, in column col)
{
    coord.y = ((coord.y + col.y) * col.height) / iResolution.y;
}

// Function 348
bool intersectShadow( in vec3 ro, in vec3 rd, in float dist ) {
    float t;
	
	t = iSphere( ro, rd, vec4( 1.5,1.0, 2.7,1.0) );  if( t>eps && t<dist ) { return true; }
    t = iSphere( ro, rd, vec4( 4.0,1.0, 4.0,1.0) );  if( t>eps && t<dist ) { return true; }

    return false; // optimisation: planes don't cast shadows in this scene
}

// Function 349
float shadow(vec3 p)
{
    float r = raymarch(p, L, 96);
    return r > 0. //&& r <= 2.
        ? .0 : 1.;
}

// Function 350
float softShadow(vec3 ro, vec3 lp)
{
	vec3 rd = normalize(lp-ro);
	float tmax = distance(lp,ro);
	float res = 1.0;
    float t = 0.1;
	for(int i = 0; i<256; i++ )
	{
        if(t>=tmax) break;
		float d = map(ro+rd*t).a;
		if(d < 0.001) return 0.0;
		res = min(res, 8.0*d);
		t += d;
	}
	return res;
}

// Function 351
float calcSoftShadow( in vec3 ro, in vec3 rd, float k )
{
    vec4 kk;    
    float res = 1.0;
    float t = 0.01;
    for( int i=ZERO; i<32; i++ )
    {
        float h = mapOpaque(ro + rd*t, kk ).x;
        res = min( res, smoothstep(0.0,1.0,k*h/t) );
        t += clamp( h, 0.04, 0.1 );
		if( res<0.01 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 352
float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
{
    float res = 1.0;
    float ph = 1e20; //
    for( float t=mint; t<maxt; )
    {
        float h = sdf(ro + rd*t);
        if( h<0.001 )
            return 0.0;
        float y = h*h/(2.5*ph); //
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.001,t-y) );
        ph = h;
        t += h;
    }
    return res;
}

// Function 353
void unpackShadow(in vec4 packed, out float depth)
{
    depth = map(packed.r, 0.0, 1.0, MIN_DIST_SHADOW, MAX_DIST_SHADOW);
}

// Function 354
vec4 voronoi_column_trace_shadow(
         vec4 mc,
         vec3 ray_pos,
         vec3 ray_dir,
         float max_h,
         out vec4 hit_pos,
         out vec3 hit_norm )
{
   const int iter = 8;

   vec2 p = ray_pos.xy;
   float s = 1./length(ray_dir.xy);
   vec2 dir = ray_dir.xy*s;
   vec2 n = floor(p);
   vec2 f = fract(p);
   
   mc -= vec4(p, n);
   
   float md;
   
   vec2 mdr = vec2(0,1);
   float dh = 0.;
   float h = 0.;
   
   md = eps;

   for( int k=0; k<iter; ++k )
   {
      // Scan through all Voronoi neighbours in direction of ray.
      
      vec4 kc;
      vec2 kdr;
      float kd = find_neighbour(n, f, dir, mc, kc, kdr)*s;
      
      mc = kc;
      md = kd;
      mdr = kdr;
      
      // Get height of the column
      h = hash12( mc.zw + n )*max_h;
      dh = ray_pos.z + ray_dir.z*md;
      if (dh > max_h || dh < h) break;
   }
   
   if (dh >= h) {
      hit_pos = vec4(ray_pos + ray_dir*max_dist,max_dist);
      hit_norm = vec3(0,0,1);
      return vec4(0);
   }
   
   float d = md;
   hit_norm = vec3(-normalize(mdr),0);
   hit_pos = vec4(ray_pos + ray_dir*d, d);
   return mc + vec4(p, n);
}

// Function 355
void ToggleEffects(inout vec4 fragColor, vec2 fragCoord)
{
  // read and save effect values from buffer  
  vec3 effects =  mix(vec3(-1.0, 1.0, -1.0), readRGB(ivec2(120, 0)), step(1.0, float(iFrame)));
  effects.y*=1.0+(-2.*float(keyPress(50))); //2-key  Grain Filter
  effects.z*=1.0+(-2.*float(keyPress(51))); //3-key  ChromaticAberration

  vec3 effects2 =  mix(vec3(1.0, 1.0, 1.0), readRGB(ivec2(122, 0)), step(1.0, float(iFrame)));
  effects2.y*=1.0+(-2.*float(keyPress(52))); //4-key  God Rays
  effects2.x*=1.0+(-2.*float(keyPress(53))); //5-key  lens flare

  fragColor.rgb = mix(effects, fragColor.rgb, step(1., length(fragCoord.xy-vec2(120.0, 0.0))));  
  fragColor.rgb = mix(effects2, fragColor.rgb, step(1., length(fragCoord.xy-vec2(122.0, 0.0))));
}

// Function 356
float areaShadow( in vec3 P )
{
  float s = 1.0;
  for( int i=0; i<SPH; i++ )
    s = min( s, sphAreaShadow(P, L, sphere[i] ) );
  return s;           
}

// Function 357
float softShadow(vec3 ro, vec3 lp, float k, float t){

    // More would be nicer. More is always nicer, but not really affordable.
    const int maxIterationsShad = 32; 
    
    vec3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = 0.001*(t*.125 + 1.);  // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

         
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        //h = clamp(h, .1, 1.); // max(h, .02);//
        h = max(h, .1);
        dist += h;

        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (shade<0.001 || dist > end) break; 
    }

    // I've added a constant to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me. Sometimes, I'll add AO also, just for kicks. :)
    return min(max(shade, 0.) + .05, 1.); 
}

// Function 358
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.01;
  for (int j = VAR_ZERO; j < 40; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 359
float shadowcast(in vec3 ro, in vec3 rd){
    float res = 1.f;
    float t = 0.001f;
    for (int i = 0; i < RAY_STEPS; i++){
        vec3 pos = ro + rd * t;
        float dist = map(pos).x;
        if (res < 0.0000001){
            break;
        }
        if (dist > FAR_CLIP){
            break;
        }
        res = min(res, 10.f * dist/t);
        t += dist;
    }
    return res;
}

// Function 360
void shadow(vec3 ro, vec3 rd, inout float t, inout int i, bool bl)
{
    float tSphere6 = intersectSphere(ro, rd, sfere[3]);
    if(tSphere6 < t && bl) { t = tSphere6;i=6;}

   	float tSphere = intersectSphere(ro, rd, sfere[2]);
    if(tSphere < t) { t = tSphere;i=2;}
    
    vec2 tRoom = intersectCube(ro, rd, box0);          
   	if(tRoom.x < tRoom.y)   t = tRoom.y; 
    vec3 hit = ro + rd * t;  
    if(hit.y > 0.9999 && hit.x<1.3 && hit.x>-1.3 && hit.z<1.99 && hit.z>1.0) t=10000.0;

}

// Function 361
float ExObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.01;
  for (int j = VAR_ZERO; j < 30; j ++) {
    h = ExObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 362
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, float time )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<128; i++ )
    {
		float h = map( ro + rd*t, time );
        res = min( res, 16.0*h/t );
        t += clamp( h, 0.01, 0.25 );
        if( res<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 363
float softShadow(in ray ray, int maxSteps/*float mint, float k*/)
{
    float k = 4.0;
    float res = 0.0;
    float t = 0.001;
	float h = 1.0;
    
    for( int i=0; i<int(maxSteps); i++ )
    {
        h = sceneDistanceFunction(ray.origin + ray.direction*t, orbitTrap);

        if(res<0.001)
        {
            break;
        }
        t += h;//clamp( h, 0.01, 0.05 );
    }
    return 1.0-saturate(res);
}

// Function 364
bool shadowBox( in vec3 ro, in vec3 rd, in vec3 cen, in vec3 rad, in float tmax ) 
{
    vec3 m = 1.0/rd;
    vec3 n = m*(ro-cen);
    vec3 k = abs(m)*rad;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
	float tN = max( max( t1.x, t1.y ), t1.z );
	float tF = min( min( t2.x, t2.y ), t2.z );
	if( tN > tF || tF < 0.0) return false;
	return tN>0.0 && tN<tmax;
}

// Function 365
float effectBorder( in vec2 p  ) {
    float crossd = crossDist(p);
    return sin(crossd*60.0+iTime*10.0)*0.5+0.5;
}

// Function 366
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for(int i=0; i<16; i++)
    {
    	float h = map(ro + rd*t);
        res = min(res, 10.0*h/t + 0.02*float(i));
        t += 0.8*clamp(h, 0.01, 0.35);
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 367
vec4 gaussianblur(sampler2D tex, vec2 xy, vec2 res, float sizered, float sizegreen, float sizeblue, float sizealpha, vec2 dir) {
    vec4 sigmas = vec4(sizered, sizegreen, sizeblue, sizealpha);

    // Set up state for incremental coefficient calculation, see GPU Gems
    // We use vec4s to store four copies of the state, for different size
    // red/green/blue/alpha blurs
    vec4 gx, gy, gz;
    gx = 1.0 / (sqrt(2.0 * 3.141592653589793238) * sigmas);
    gy = exp(-0.5 / (sigmas * sigmas));
    gz = gy * gy;
    // vec4 a, centre, sample1, sample2 = vec4(0.0);
    vec4 a = vec4(0.0);
    vec4 centre = vec4(0.0);
    vec4 sample1 = vec4(0.0);
    vec4 sample2 = vec4(0.0);

    // First take the centre sample
    centre = texture(tex, xy / res);
    a += gx * centre;
    vec4 energy = gx;
    gx *= gy;
    gy *= gz;

    // Now the other samples
    float support = max(max(max(sigmas.r, sigmas.g), sigmas.b), sigmas.a) * 3.0;
    for(float i = 1.0; i <= support; i++) {
        sample1 = texture(tex, (xy - i * dir) / res);
        sample2 = texture(tex, (xy + i * dir) / res);
        a += gx * sample1;
        a += gx * sample2;
        energy += 2.0 * gx;
        gx *= gy;
        gy *= gz;
    }

    a /= energy;

    if(sizered < 0.1) a.r = centre.r;
    if(sizegreen < 0.1) a.g = centre.g;
    if(sizeblue < 0.1) a.b = centre.b;

    return a;
}

// Function 368
float shadow( in vec3 ro, in vec3 rd, in float maxt)
{
	float res = 1.0;
    float dt = 0.04;
    float t = .02;
    for( int i=0; i < 20; i++ )
    {       
        float h = map(ro + rd*t).x;
        if( h<0.001 )
            return 0.0;
        res = min( res, maxt*h/t );
        t += h;
    }
    return res;
}

// Function 369
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 370
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

// Function 371
float shadow(vec3 ro, vec3 lp, float k, float t){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 32; 
    
    vec3 rd = lp-ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = .001*(t*.125 + 1.);  // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        dist += clamp(h, .01, .2); 
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0.0 || dist > end) break; 
    }

    // I sometimes add a constant to the final shade value, which lightens the shadow a bit. It's a preference 
    // thing. Really dark shadows look too brutal to me. Sometimes, I'll also add AO, just for kicks. :)
    return min(max(shade, 0.) + .0, 1.); 
}

// Function 372
float shadow(vec3 rpos, vec3 rdir) {
	float t = 1.0+SHADOW_QUALITY;
	float sh = 1.0;
	for (int i = 0; i < SHADOW_ITERS; i++) {
		vec3 pos = rpos + rdir * t;
		float h = pos.y - terrain(pos.xz);
		if (h < 0.0) return 0.0;
		sh = min(sh, h/t*8.0);
		t += max(h, SHADOW_QUALITY);
	}
	return sh;
}

// Function 373
float renderRingFarShadow( const in vec3 ro, const in vec3 rd ) {
    // intersect plane
    float d = iPlane( ro, rd, vec4( 0., 0., 1., 0.) );
    
    if( d > 0. ) {
	    vec3 intersection = ro + rd*d;
        float l = length(intersection.xy);
        
        if( l > RING_INNER_RADIUS && l < RING_OUTER_RADIUS ) {
            return .5 + .5 * (.2+.8*noise( l*.07 )) * (.5+.5*noise(intersection.xy));
        } else {
            return 0.;
        }
    } else {
	    return 0.;
    }
}

// Function 374
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax , float io )
{
float res = 1.0;
    float t = mint;
    for( int i=0; i<10; i++ )
    {
float h = map( ro + rd*t , io).x;
        res = min( res, 20.*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 375
float ObjSShadow (vec3 ro, vec3 rd, float dMax)
{
  float sh, d, h;
  doSh = true;
  sh = 1.;
  d = 0.01;
  for (int j = VAR_ZERO; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.001 || d > dMax) break;
  }
  doSh = false;
  return 0.5 + 0.5 * sh;
}

// Function 376
float blur(vec2 uv, float t){
    float col = 0.0;
    const float offAmt = 0.005;
    const float samps = 5.0;
    for(float i=0.0; i<samps; i++){
        float ang = i/samps*6.28;
        col += noiseSource(uv+vec2(cos(ang),sin(ang))*offAmt,t);
    }
    col += noiseSource(uv,t);
    col /= (samps+1.0);
    return col;
}

// Function 377
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

// Function 378
float ExObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.02;
  for (int j = 0; j < 30; j ++) {
    h = ExObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.01 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 379
float softShadowTrace(in vec3 rp, in vec3 rd, in float maxDist, in float penumbraSize, in float penumbraIntensity) {
    vec3 p = rp;
    float sh = 0.;
    float d,s = 0.;
    for (int i = 0; i < ITERATIONS; i++) {
        d = df(rp+rd*s);
        sh += max(0., penumbraSize-d)*float(s>penumbraSize*4.);
        s += d;
        if (d < EPSILON || s > maxDist) break;
    }
    
    if (d < EPSILON) return 0.;
    
    return max(0.,1.-sh/penumbraIntensity);
}

// Function 380
float softshadow( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;
    float t = 0.001;
	float h = 1.0;
    for( int i=0; i<32; i++ )
    {
        h = map(ro + rd*t);
        if(object_id==LIGHT)break;
        res = min( res, k*h/t );
        if( res<0.001 ) break;
        t += h;
    }
    return clamp(res,0.1,1.0);
}

// Function 381
float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<128; i++ )
	{
        float h = map(ro + rd*t);
        if( h<0.001 )
            return 0.0;
        res = min( res, k*h/t );
        t += h;
        if (t >= maxt) {
            break;
        }
	}
    return res;
}

// Function 382
float softShadow(in vec3 ro, in vec3 rd ){
    float res = 1.0;
    float t = 0.001;
	for( int i=0; i<80; i++ )
	{
	    vec3  p = ro + t*rd;
        float h = p.y - terrainM( p.xz );
		res = min( res, 16.0*h/t );
		t += h;
		if( res<0.001 ||p.y>(SC*200.0) ) break;
	}
	return clamp( res, 0.0, 1.0 );
}

// Function 383
vec3 GetApproximateShadowFactor(vec3 position, vec3 rayDirection)
{
    float distanceToPlane = GetApproximateIntersect(position, rayDirection);
	return BeerLambert(WaterAbsorption / WaterColor, distanceToPlane);
}

// Function 384
float softShadow( in vec3 ro, in vec3 rd, float mint, float k )
{
	vec4 dummy;
    float res = 1.0;
    float t = mint;
    for( int i=0; i<45; i++ )
    {
        float h = map(ro + rd*t,dummy).x;
        res = min( res, k*h/t );
        t += h*STEP_REDUCTION;
    }
    return clamp(res,0.0,1.0);
}

// Function 385
void shadowSoft(inout float s,inout vec3 x,inout float j,float t,vec4 m,mat4 B
){float h
 ;for(int i=0;i<64;i++
 ){h=map(x+lightDir*j).x
  ;j+=clamp(h, .032, 1.);
               	s = min(s, h/j);
             	if(j>7.|| h<.001) break;
            } 
}

// Function 386
bool isInShadow(vec3 p, Sphere sphere, Light light)
{  
  float lightDistance = distance(light.position, p);
  vec3 shadowDir = normalize(light.position - p);
  Ray shadowRay = Ray(p + 0.1 * shadowDir, shadowDir);    
  float tShadow = intersect(shadowRay, sphere);
  if(!isinf(tShadow) && tShadow < lightDistance)
	return true;
  
  return false;
}

// Function 387
float
shadow( in vec3 start, in vec3 dir )
{
    float ret = 1.0;
    float c = 1.0;//step( mod( iTime, 4.0 ), 2.0 );
    float t = 0.02, t_max = 16.0;
    MPt mp;
    
    #if DRAW_ITERATIONS_GRADIENT
    int it_;
    #endif
    for ( int it=0; it!=SHADOW_MAX_ITERS; ++it )
    {
	    #if DRAW_ITERATIONS_GRADIENT
	    it_ = it;
    	#endif
        vec3 here = start + dir * t;
        mp = map( here );
        ret = min( ret, 8.0*mp.distance/t);
        if ( mp.distance < ( T_EPS * t ) || t > t_max )
        {
        	break;
        }
        
        float inc;
        // NOTE(theGiallo): this is to sample nicely the twisted things
        inc = c * mp.distance * 0.4;
		inc += ( 1.0 - c ) * clamp( mp.distance, 0.02, 0.1 );
        t += inc;
    }
    #if DRAW_ITERATIONS_GRADIENT
    return float(it_);
    #endif
    if ( t > t_max )
    {
        t = -1.0;
    }
    if ( c == 0.0 ) return 1.0 - clamp( ret, 0.0, 1.0 );

    if ( t < 0.0 )
    {
        return 0.0;
    }
    //return 1.0;
    ret = 1.0 / pow(1.0 - 1e-30 + max( mp.distance, 1e-30 ), 5.0 );
    float th = 0.1;
    return smoothstep( 0.0, 1.0, ( ret*1.1 - th ) / (1.0-th) );
}

// Function 388
float shadow(vec3 p, vec3 dir) {
	float td=.0, sh=1., d=.2;
    for (int i=0; i<60; i++) {
        p-=d*dir;
        d=de(p);
        float dl=de_light(p);
        td+=min(d,dl);
        sh=min(sh,10.*d/td);
        if (sh<.01 || dl<1.) break;
    }
    return clamp(sh,0.,1.);
}

// Function 389
float SoftShadow(in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.001;
    for( int i=0; i<80; i++ )
    {
        vec3  p = ro + t*rd;
        float h = p.y - TerrainH( p.xz );
        res = min( res, 16.0*h/t );
        t += h;
        if( res<0.001 ||p.y>(SC*20.0) ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 390
float	SphereFlakeShadowStack( in vec3 ro, in vec3 rd )
{
    float	result	= 1.0;

    int	idx	= 1;
// Get current element on the stack.
#define	SFEL g_stack[ idx ] 
// Get parent element on the stack
#define	SFPEL g_stack[ idx - 1 ]
// Get parent's parent element on the stack
#define	SFPPEL g_stack[ idx - 2 ]
// Do sphere intersect call on the current element on the stack.
#define SFShadow sphSoftShadow( ro, rd, SFEL.center, SFEL.radius, result )
// Do sphere intersect test call on the current element on the stack.
#define SFIntersectTest sphIntersectTest( ro, rd, SFEL.center, SFEL.radius )
// Check if current element on the stack has a small radius.
#define SFIsRadTooSmall isRadTooSmall( ro, SFEL.center, SFEL.radius )

    if( ! SFIntersectTest )
        return	result;

    SFShadow;

    idx = 2;
    while( idx > 1 )
    {
        if( SFEL.sphereIndex == TOTAL_SPHERES_COUNT )
        {
            SFEL.sphereIndex	= 0;
            idx--;
            continue;
        }

        vec3	perp1	= normalize( cross( SFPEL.direction, SFPPEL.direction ) );
        vec3	perp2	= normalize( cross( SFPEL.direction, perp1 ) );

        vec3	rot		= perp1 * PERP1MOD( SFEL.sphereIndex ) 
            			+ perp2 * PERP2MOD( SFEL.sphereIndex );

        vec3	dirNN	= SFPEL.direction * YAXIS_COS( SFEL.sphereIndex )
            			+ rot;
        
        SFEL.direction	= normalize( dirNN );
        SFEL.center		= SFEL.direction * ( SFEL.radius + SFPEL.radius ) 
            			+ SFPEL.center;

        SFEL.sphereIndex++;
        
        if( SFIsRadTooSmall || ! SFIntersectTest )
            continue;

        SFShadow;

        if( idx == STACK_MAX_SIZE )
            continue;

		idx++;
    }
    
    return	result;
}

// Function 391
float shadow( in vec3 ro, in vec3 rd, float dis)
{
	float res = 1.0;
    float t = hash11(dis)*.5+.2;
	float h;
	
    for (int i = 0; i < 10; i++)
	{
        vec3 p =  ro + rd*t;

		h = map(p,dis).x;
		res = min(10.*h / t*t, res);
		t += h*2.5;
	}
    //res += t*t*.02; // Dim over distance
    return clamp(res, .3, 1.0);
}

// Function 392
float doShadows(in vec3 rp)
{
    float s = 1.0;
    rp+=lightDir*.01;
    
    for (int i = 0; i < 4; ++i)
    {
    	float dist=map(rp);
        rp+=max(dist, 0.001)*lightDir;
        s=min(s,dist);
    }
    return mix(1.0, clamp(s/0.01, 0.0, 1.0), 0.8);
}

// Function 393
column rainEffect(float x, float w/*idth*/, float r/*adius*/)
{
    if (x >= w && x <= W - w)
        return column(0., H);
    x -= (W / 2.);
    float x0 = w - (W / 2.);
    float h = 1. + sqrt(r * r - x0 * x0) - sqrt(r * r - x * x);
    float y = fract(rand(x) + iTime) * h;
    return column(y, H - y);
}

// Function 394
bool IsShadowTrace( int trace_flags ) { return ( trace_flags & TRACE_SHADOW ) != 0; }

// Function 395
void effect(inout vec4 col, vec2 coord)
{
    vec4 col1,col2,col3;
    effectFlow(col1,coord);   // fluid effect
    effectSmear(col2,coord);  // normal to gradient diffusion effect (smearing out a little)
    effectDiff(col3,coord);   // gradient diffusion effect
    //col2=col;
    //float effType=.2;
    //float effType=.5-.5*sin(iTime*.5);
    float effType=smoothstep(.0,.2,-sin(iTime*.3-.3));
    if(iMouse.y>1.) effType=iMouse.y/iResolution.y;
    col=mix(col1,col3,effType);
    col=mix(col,col2,.3);
}

// Function 396
vec3 ringShadowColor( const in vec3 ro ) {
    if( iSphere( ro, SUN_DIRECTION, vec4( 0., 0., 0., EARTH_RADIUS ) ) > 0. ) {
        return vec3(0.);
    }
    return vec3(1.);
}

// Function 397
float shadowSoft( vec3 ro, vec3 rd, float mint, float maxt, float k )
{
	float t = mint;
	float res = 1.0;
    for ( int i = 0; i < 128; ++i )
    {
        float h = scene( ro + rd * t );
        if ( h < 0.001 )
            return 0.0;
		
		res = min( res, k * h / t );
        t += h;
		
		if ( t > maxt )
			break;
    }
    return res;
}

// Function 398
vec3 SkyDomeBlurry(vec3 rayDir, float lod)
{
    rayDir.z = -rayDir.z;
    return AmbianceLight(textureLod(iChannel2, rayDir.xyz,  lod).rgb);
}

// Function 399
float CaveSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 16; j ++) {
    h = CaveDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += max (0.2, 0.1 * d);
    if (sh < 0.05) break;
  }
  return 0.4 + 0.6 * sh;
}

// Function 400
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

// Function 401
float SoftShadow( in vec3 origin, in vec3 direction )
{
  float res = 2.0, t = 0.0, h;
  for ( int i=0; i<16; i++ )
  {
    h = MapTerrain(origin+direction*t);
    res = min( res, 3.5*h/t );
    t += clamp( h, 0.02, 0.8);
    if ( h<0.002 ) break;
  }
  return clamp( res, 0.0, 1.0 );
}

// Function 402
float shadow(in vec3 eye, vec3 marchingDirection, vec3 normal, float end) {
    eye += normal * SHADOW_EPSILON * 2.0;
    float depth = 0.0;

    float shad = 1.0;
    float ph = 1e10;

    for (int i = 0; i < MAX_SHADOW_STEPS; i++) {
        DistMat distMat = sceneSDF(eye + depth * marchingDirection);
        float dist = distMat.dist;

        if (dist < SHADOW_EPSILON) {
	        shad = 0.0;
	        break;
        }

        if (depth > end) {
            break;
        }

        float y = dist * dist / (2.0 * ph);
        float d = sqrt(dist * dist - y*y);
        shad = min(shad, 4.0 * d / max(0.0, depth - y));
        ph = dist;

        depth += dist;
    }

    return clamp(shad, 0.0, 1.0);
}

// Function 403
float getShadow(vec3 ro, vec3 rd)
{
   float d = 0.001; 
   
   for(int i = 0; i < MAXSTEP; i++)
   {
       vec3 pos = ro + d*rd;
       
       float eval = map(pos).x;
       
       if(eval < 0.0) return 0.0;
       d += max(eval,0.04);
       
       if(d > 20.0) return 1.0;
   }
    
   return 1.0;
}

// Function 404
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

// Function 405
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.007;
  for (int j = 0; j < 15; j ++) {
    h = SceneDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 1., 20. * h / d));
    d += min (0.016, 3. * h);
    if (h < 0.001) break;
  }
  return 0.6 + 0.4 * sh;
}

// Function 406
float Shadows(vec3 Pos, vec3 Dir, float MaxDist, float ConeRatio) {
    Info t; float Dist=0.; float Occ=1.; float sD,Dif;
    while (Dist<MaxDist) {
        t=DF(Pos+Dir*Dist,iTime);
        if (t.d<0.001) return 0.;
        sD=Dist*ConeRatio;
        Dif=t.d-sD;
        Occ=Occ*((Dif<0.)?(sD*2.+Dif)/(sD*2.):1.);
        Dist=Dist+t.d;
    }
    return Occ;
}

// Function 407
float softshadow(float tt, vec3 ro, in vec3 rd, float k ,vec4 m,mat4 B) {
    float res=1., z=0.02, h=1.;
    for(int i=0; i<53; i++){
        h = map(ro + rd*z).x;
        res = min( res, k*h/z );
		z += clamp( h, 0.015, 1.0 );
		if( h<0.012 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 408
float getShadowEdge( vec2 fragCoord )
{
    vec2 coord = fragCoord;
    float sha = readShadow(coord);
    sha -= .5;
    sha = abs(sha);
    sha = smoothstep(.1,.2,sha);
    return sha;
}

// Function 409
float getShadow(vec3 pos, vec3 light, vec3 normal){
	vec3 shadowRay = normalize(light - pos);
    pos += 3. * EPS * normal;
    float totDist = 3. * EPS;
    float prevDist = totDist;
    float shadow = 1.;
    float dist, c;

    for(float s = 0.; s < STEPS_SHADOW; s++){
        dist = distScene(pos, c);
        shadow = min(shadow, 4. * dist / totDist);
        if(abs(dist) < EPS){
            shadow = 0.;
            break;
        }
        dist = 0.997 * dist + 0.003 * hash3(pos + sin(iTime));
        pos += shadowRay * dist;
        totDist += dist;
        if(totDist > 2.) break;
    }
    return clamp(shadow, 0., 1.);
}

// Function 410
float calcShadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.01;
    for( int i=ZERO; i<100; i++ )
    {
        vec3 pos = ro + rd*t;
        float h = mapShadow( pos ).x;
        res = min( res, 16.0*max(h,0.0)/t );
        if( h<0.0001 || pos.y>3.0 ) break;
        
        t += clamp(h,0.01,0.2);
    }
    
    return clamp(res,0.0,1.0);
}

// Function 411
float ShadowMarch( vec3 pos, vec3 light )
{
    vec3 ray = normalize(light-pos);
    float e = length(light-pos);
    float t = .02; // step away from the surface
    for ( int i=0; i < 200; i++ )
    {
        float h = Scene(pos+ray*t);
        if ( h < .001 )
        {
            return 0.; // hit something
        }
        if ( t >= e )
        {
            break;
        }
        t += h;
    }
    return 1.; // didn't hit anything
}

// Function 412
float softshadow( in vec3 ro, in vec3 rd, float k ) {
    float res=1., t=.02, h=1.;
    for(int i=0; i<38; i++ ) {
        h = map(ro + rd*t).x;
        res = min( res, k*h/t );
		t += clamp( h, .015, 1. );
		if( h<0.012 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 413
float Shadow( in vec3 ro, in vec3 rd, float dist)
{
	float res = 1.0;
    float t = 0.01;
	float h = 0.0;
    
	for (int i = 0; i < 12; i++)
	{
		if(t < dist)
		{
			h = de(ro + rd * t);
			res = min(4.0*h / t, res);
			t += h + 0.002;
		}
	}
	
    return clamp(res, 0.0, 1.0);
}

// Function 414
float GrndSShadow (vec3 ro, vec3 rd)
{
  float sh = 1.;
  float d = 0.01;
  float eps = 0.001;
  for (int i = 0; i < 80; i++) {
    vec3 p = ro + rd * d;
    float h = p.y - GrndHt (p.xz, 0);
    sh = min (sh, 20. * h / d);
    d += 0.5;
    if (h < eps*(1.0+d)) break;
    eps *= 1.02;
  }
  return clamp (sh, 0., 1.);
}

// Function 415
float calcSoftshadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.0005;                 // selfintersection avoidance distance
	float h = 1.0;
    for( int i=0; i<40; i++ )         // 40 is the max numnber of raymarching steps
    {
        h = doModel(ro + rd*t).x;
        res = min( res, 50.0*h/t );   // 64 is the hardness of the shadows
		t += clamp( h, 0.02, 2.0 );   // limit the max and min stepping distances
    }
    return clamp(res,0.0,1.0);
}

// Function 416
float calcSoftshadow( in vec3 ro, in vec3 rd, float time )
{
    float res = 1.0;

    float tmax = 12.0;
    #if 1
    float tp = (3.5-ro.y)/rd.y; // raytrace bounding plane
    if( tp>0.0 ) tmax = min( tmax, tp );
	#endif    
    
    float t = 0.02;
    for( int i=0; i<50; i++ )
    {
		float h = map( ro + rd*t, time ).x;
        res = min( res, mix(1.0,16.0*h/t, hsha) );
        t += clamp( h, 0.05, 0.40 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 417
float shadowcast_pointlight(in vec3 ro, in vec3 rd, in float light_dist){
    float res = 1.f;
    float t = MIN_CLIP;
    light_dist = min(light_dist, FAR_CLIP);
    for (int i = 0; i < SHADOW_RAY_STEPS; i++){
        t = min(t, light_dist);
        vec3 pos = ro + rd * t;
        float sdf = map(pos).x;
        // Soft shadows were causing heavy banding on the blobs from the point light sources.
        // The reason was that the "distance function" wasn't exact.
        // We can fix this by using a first-order Taylor series approximation for the true distance function.
        // (https://www.iquilezles.org/www/articles/distance/distance.htm)
        // Nevermind, there's something else at play here, too. Also, computing the gradient during the shadow step is expensive.
        // float dist = sdf / length(grad(pos, 0.01));
        float dist = sdf * 50.;
        res = min(res, 0.2 * dist/t);
        t += sdf;
        if (t >= light_dist || res < 0.001) break;
    }
    return res;
}

// Function 418
float Shadow(vec3 pos, float d)
{
    float shadow = 1.0;
    float depth  = mix(1.0, 0.3, clamp(d / 3.0, 0.0, 1.0));
    
    for(int i = 0; i < 64; ++i)
    {
        vec3 p = pos + (SunLightDir * depth);
        vec2 sdf = Scene(p, 1.0);
        
        shadow = min(shadow, (32.0 * sdf.x) / depth);
        depth += sdf.x;
        
        if(sdf.x < 0.001)
        {
            break;
        }
    }
    
    return clamp(shadow, 0.1, 1.0);
}

// Function 419
vec3 effect(vec2 p, vec2 q) { 
  vec3 ro = 0.6*vec3(2.0, 0, 0.2)+vec3(0.0, 0.75, 0.0);
  ro.xz *= ROT(PI/2.0+sin(TIME*0.05));
  ro.yz *= ROT(0.5+0.25*sin(TIME*0.05*sqrt(0.5))*0.5);

  vec3 ww = normalize(vec3(0.0, 0.0, 0.0) - ro);
  vec3 uu = normalize(cross( vec3(0.0,1.0,0.0), ww));
  vec3 vv = normalize(cross(ww,uu));
  float rdd = 2.0;
  vec3 rd = normalize( p.x*uu + p.y*vv + rdd*ww);

  vec3 col = render(ro, rd);
  return col;
}

// Function 420
float ComputeShadow(const vec3 p, const vec3 n, const vec3 L, const float d2l)
{
    float shadow = 1.0;
    
    Ray r;
    r.o = p + n*SHADOW_BIAS; // Without this, the ray doesn't leave the surface
    r.d = L;
    
    RayIntersection ri = CastRay(r, d2l);
    if (ri.shape.type != NO_SHAPE) shadow = 0.0;
    else shadow = clamp(ri.shadow, .0,1.0);

    return shadow;
}

// Function 421
vec4 BlurTri(samplerCube ctx, vec2 p, int dim){
    
    // Initiate the color.
    vec4 col = vec4(0);
    
    int hDim = dim/2; // Half dimension.
    
    float tot = 0.;
    // There's a million boring ways to apply a kernal matrix to a pixel, and this 
    // is one of them. :)
    for (int j=0; j<dim; j++){
        for (int i=0; i<dim; i++){ 
            
            // Alternative distance-based blur. Gaussians are possible too, but
            // I found the simpler formulas better suited to these examples.
            //float ij = length(vec2(hDim - i, hDim - j));
            //ij = max(length(vec2((hDim + 1)*(hDim + 1)))*.75 - ij*ij, 0.);

            // Smoothed triangle blur, of sorts.
            float ij = float(hDim - abs(hDim - i) + 1)*float(hDim - abs(hDim - j) + 1);
            float mDim = float((hDim + 1)*(hDim + 1));
            ij = smoothstep(0., 1., ij/mDim)*mDim;
            
            // Adding the weighted value.
            col += ij*tx(ctx, (p + vec2(i - hDim, j - hDim)/iRes0));
            tot += ij;
        }
    }   
    
    return col/tot;
}

// Function 422
float softShadow(vec3 ro, vec3 rd )
{
    float res = 1.0;
    float t = 0.001;
	
	for(int i = 0; i < 80; i++)
	{
	    vec3  p = ro + t * rd;
        float h = p.y - terrainHeight( p.xz );
		
		res = min( res, 16.0 * h / t );
		t += h;
		
		if( res<0.001 || p.y > (SC * 200.0) ) break;
	}
	
	return clamp( res, 0.0, 1.0 );
}

// Function 423
float calc_aa_blur(float w) {
    vec2 blur = _stack.blur;
    w -= blur.x;
    float wa = clamp(-w*AA*uniform_scale_for_aa(), 0.0, 1.0);
    float wb = clamp(-w / blur.x + blur.y, 0.0, 1.0);
	return wa * wb;
}

// Function 424
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

// Function 425
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = 0; j < 24; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += min (0.08, 3. * h);
    if (sh < 0.001) break;
  }
  return 0.3 + 0.7 * sh;
}

// Function 426
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax ) {
	float res = 1.0;
    float t = mint;
    float ph = 1e10; 
    for( int i=0; i<24; i++ ) {
		float h = map( ro + rd*t );
       	float y = h*h/(2.0*ph);
        float d = sqrt(max(0.,h*h-y*y));
        res = min( res, 8.0*d/max(0.01,t-y) );
        ph = h;
        t += min(h, .2);// clamp( h, 0.02, 0.10 );
        if( res<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 427
float shadow(vec3 p, vec3 n, vec3 lPos)
{
    return shadow(p + n * MIN_DST * 4.0, lPos);
}

// Function 428
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = 0; j < 25; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 1., 20. * h / d));
    d += min (0.1, 3. * h);
    if (h < 0.001) break;
  }
  return sh;
}

// Function 429
float Shadow(vec3 pos)
{
    float shadow = 1.0;
    float depth  = 1.0;
    
#ifdef HIGH_QUALITY
    for(int i = 0; i < 32; ++i)
#else
    for(int i = 0; i < 24; ++i)
#endif
    {
        vec2 sdf = Scene(pos + (SunLightDir * depth));
        
        shadow = min(shadow, (8.0 * sdf.x) / depth);
        depth += sdf.x;
        
        if(sdf.x < 0.001)
        {
            shadow = 0.0;
            break;
        }
    }
    
    return clamp(shadow, 0.0, 1.0);
}

// Function 430
float calcSoftShadow( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;
    float t = 0.01;
    for( int i=0; i<24; i++ )
    {
        float h = mapWithElephants(ro + rd*t );
        res = min( res, smoothstep(0.0,1.0,k*h/t) );
        t += clamp( h, 0.05, 0.5 );
		if( res<0.01 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 431
float intersectsBrickShadow(inout Ray ray, in AABB aabb) {
  float res = 1.0;
  float ph = 1e10;
  float k = 10.0;
    
  float t, t0, t1;
  if (!rayIntersectsAABB(ray, aabb, t, t1)) {
    return res;
  }
  t = max(0.0, t0);
  for (int i = 0; i < steps; i++) {
    vec3 position = positionOnRay(ray, t);
    float sd = map(position);
     
    float y = sd * sd / (2.0 * ph);
    float d = sqrt(sd * sd - y * y);
    res = min(res, k * d / max(0.0, t - y));
    ph = sd;
      
    t += min(sd, 0.005);
      
    if (res < 0.0001 || t > t1) {
      break;
    }
  }
    
  return clamp(res, 0.0, 1.0);
}

// Function 432
void crt_effect(inout vec4 fragColor, vec2 fragCoord, Options options)
{
#if USE_CRT_EFFECT
	if (!test_flag(options.flags, OPTION_FLAG_CRT_EFFECT))
        return;
    
    vec2 uv = fragCoord / iResolution.xy, offset = uv - .5;
    fragColor.rgb *= 1. + sin(fragCoord.y * (TAU/4.)) * (CRT_SCANLINE_WEIGHT);
    fragColor.rgb *= clamp(1.6 - sqrt(length(offset)), 0., 1.);
    
    const float
        MASK_LO = 1. - (CRT_MASK_WEIGHT) / 3.,
        MASK_HI = 1. + (CRT_MASK_WEIGHT) / 3.;

    vec3 mask = vec3(MASK_LO);
    float i = fract((floor(fragCoord.y) * 3. + fragCoord.x) * (1./6.));
    if (i < 1./3.)		mask.r = MASK_HI;
    else if (i < 2./3.)	mask.g = MASK_HI;
    else				mask.b = MASK_HI;

	fragColor.rgb *= mask;
#endif // USE_CRT_EFFECT
}

// Function 433
vec4 pulseDrawShadowsFunc(in vec2 coord, float time, vec2 pv, vec2 sx, vec2 sy, vec4 color){
    // Draw point
    vec2 pixel = vec2(1., 1.);
    
    if(sqrt(pow(abs(pv.y - coord.y * 2. * SCALE_Y + SCALE_Y) / pixel.y * 1., 2.)
       + pow(abs(pv.x - coord.x * 2. * SCALE_X + SCALE_X) / pixel.x * 1., 2.)) < .1
      )
        return color * .7;
    
    return vec4(0, 0, 0, 1.0);
}

// Function 434
float shadowCircle(vec2 p, float r, float i)
    {
        //hacked: moving the shadow for light direction
        p +=vec2(0.02, 0.02)*rot(iTime/10.);
        
        //using polor coordinates to make a cool shape : https://thebookofshaders.com/07/
        float a  =atan( p.y,p.x);
        float shape = sin(a*i+i/1.)/10.;
 	
        //SS is used as smoothstep range
    	//there is a much better way that fabrice told me about.
        //It's in another shader. (Applause)
        float ss = 0.05;
        //shadow created with shape and 1.0-shadow so it returns as black where it should
    	float k = 1.0-smoothstep(r-ss, r+ss, length(p/1.5)+shape);
    	return pow(k,1.2);
}

// Function 435
float Shadows(vec3 Pos, vec3 Dir, float MaxDist, float ConeRatio) {
    Info t; float Dist=0.; float Occ=1.; float sD;
    while (Dist<MaxDist) {
        t=DF(Pos+Dir*Dist,iTime);
        if (t.d<0.001) return 0.;
        sD=Dist*ConeRatio;
        Occ=min(Occ,t.d/sD);
        Dist=Dist+t.d;
    }
    return Occ;
}

// Function 436
float shadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<20; i++ )
    {
		float h = map( ro + rd*t );
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.1, 0.4 );
        if( h<0.001 || t>5. ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 437
float shadowhit( const vec3 ro, const vec3 rd, const float dist) {
    vec3 normal;
    float d = traceSphereGrid( ro, rd, vec2(.3, dist), normal, 4).y;
    d = min(d, iCylinder(ro, rd, vec2(.3, dist), normal, vec3(0,.2,0), 1.5, false));
    return d < dist-0.001 ? 0. : 1.;
}

// Function 438
float Effects(float time)
{
    float effects = 0.0;
    return effects;
}

// Function 439
float softShadow(in vec3 ro, in vec3 rd, in float k)
{
    float res = 1.0;
    float t = 0.0;
    for(int i = 0; i < 64; i++)
    {
        float d = distanceEstimate(ro + rd * t);
        res = min(res, k * d/t);
        if(res < 0.001)
            break;
        t += clamp(d, 0.01, 0.2);
    }
    return (clamp(res, 0.0, 1.0));
}

// Function 440
float marchShadowCheck(VoxelHit hit)
{
    vec3 ro = hit.hitRel + vec3(hit.mapPos) + 0.5;
    vec3 rd = SUN_DIRECTION;
    ro += rd*0.11;
    
    ivec3 mapPos = ivec3(floor(ro));
    vec3 deltaDist = abs(vec3(length(rd)) / rd);
    ivec3 rayStep = ivec3(sign(rd));
    vec3 sideDist = (sign(rd) * (vec3(mapPos) - ro) + (sign(rd) * 0.5) + 0.5) * deltaDist; 
	float fogAccum = 0.0;
    float prevDist = 0.0;
    
    for (int i = 0; i < 16; i++) {

        // check current position for voxel
        vec2 occlusions;  int terrainType;
        getVoxelAndOcclusionsAt(mapPos, terrainType, occlusions);
        
        // if intersected, finish
        if (terrainType != VOXEL_NONE) {
            return 1.0;
        }

        // march forward to next position
        float newDist = min( sideDist.x, min(sideDist.y, sideDist.z ));
        vec3 mi = step( sideDist.xyz, sideDist.yzx ); 
        vec3 mm = mi*(1.0-mi.zxy);
        sideDist += mm * vec3(rayStep) / rd;
        mapPos += ivec3(mm)*rayStep;
        
        // accumulate fog
        fogAccum += occlusions.x * (newDist - prevDist);
        prevDist = newDist;
    }
    
    // no intersection
    return fogAccum / 5.0;
}

// Function 441
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = VAR_ZERO; j < 40; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 442
float GetLight_Diffuse_WithShadows(vec4 p) {
    
    vec4 lightPos = vec4(0, 3, 1, 1);
    
    lightPos.xz += vec2(0., cos(2.0*iTime));
    
    //Ray from the found scene position to the light position, normalized to [0,1]
    vec4 lightRay = normalize(lightPos-p);
    vec4 normalRay = GetNormal_Generic(p);
    
    //Dot product returns [-1:1], so.. "clamp" it
    float dif = clamp(dot(normalRay, lightRay), 0.0, 1.0);
    
    //Let's say we are rendering a point on the plane in the shadow of the sphere.
    //Ray march from this point in the direction of the light to see if we collide
    //with more scenery. If we do, reduce the diffuse lighting.
    //
    //Because 'p' was an output to RayMarch already, p already collides with the scene.
    //So move p a little bit away using the normal we already found.
    float d = RayMarch(p + normalRay*SURF_DIST, lightRay);
    if(d < length(lightPos-p)) dif*= max(abs(d)/(length(lightPos-p)*2.),0.1);
    
    return dif;
}

// Function 443
float Effect2(vec2 coords, float numSegments, float r, float cutoffMax, float cutoffMin, float thickness)
{
    float dist = length(coords);
    float s = abs(sin(r * numSegments * 0.5));
    
    if (dist <= cutoffMax && dist >= cutoffMin)
    {
        if (s < sin(dist) * thickness) 
        {
            return 1.0;
        }
    }
    return 0.0;
}

// Function 444
float
shadow( in vec3 start, in vec3 dir )
{
    float ret = 1.0;
    float c = step( mod( iTime, 4.0 ), 2.0 );
    float t = 0.02, t_max = 16.0;
    MPt mp;
    
    #if DRAW_ITERATIONS_GRADIENT
    int it_;
    #endif
    for ( int it=0; it!=16; ++it )
    {
	    #if DRAW_ITERATIONS_GRADIENT
	    it_ = it;
    	#endif
        vec3 here = start + dir * t;
        mp = map( here );
        ret = min( ret, 8.0*mp.distance/t);
        if ( mp.distance < ( T_EPS * t ) || t > t_max )
        {
        	break;
        }
        
        float inc;
        // NOTE(theGiallo): this is to sample nicely the twisted things
        inc = c * mp.distance * 0.4;
		inc += ( 1.0 - c ) * clamp( mp.distance, 0.02, 0.1 );
        t += inc;
    }
    #if DRAW_ITERATIONS_GRADIENT
    return float(it_);
    #endif
    if ( t > t_max )
    {
        t = -1.0;
    }
    if ( c == 0.0 ) return 1.0 - clamp( ret, 0.0, 1.0 );

    if ( t < 0.0 )
    {
        return 0.0;
    }
    //return 1.0;
    ret = 1.0 / pow(1.0 - 1e-30 + max( mp.distance, 1e-30 ), 5.0 );
    float th = 0.1;
    return smoothstep( 0.0, 1.0, ( ret*1.1 - th ) / (1.0-th) );
}

// Function 445
float calcSoftshadow( in vec3 ro, in vec3 rd, in float k, in float time )
{
    float res = 1.0;
    
    // bounding sphere
    vec2 b = iSphere( ro, rd, 0.535 );
	if( b.y>0.0 )
    {
        // raymarch
        float tmax = b.y;
        float t    = max(b.x,0.001);
        for( int i=0; i<64; i++ )
        {
            float h = map( ro + rd*t, time ).x;
            res = min( res, k*h/t );
            t += clamp( h, 0.012, 0.2 );
            if( res<0.001 || t>tmax ) break;
        }
    }
    
    return clamp( res, 0.0, 1.0 );
}

// Function 446
float softshadow( in vec3 ro, in vec3 rd, float mint, float k, vec3 c )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<80; i++ )
    {
        vec4 kk;
        float h = map(ro + rd*t, c, kk);
        res = min( res, k*h/t );
        if( res<0.001 ) break;
        t += clamp( h, 0.002, 0.1 );
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 447
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

// Function 448
vec4 shadow(ray r){
	float s=1.0;
	float t=MIN_DIST_SHADOW;
	for(int i=0;i<MAX_ITER_SHADOW;i++){
		ray tmp=r;
		tmp.p+=r.d*t;
		float h=dist(tmp);
		if(h<MIN_DIST)return vec4(0.0);
		s=min(s,PENUMBRA*h/t);
		t+=h;
		if(t>MAX_DIST_SHADOW)break;
	}
	return vec4(1.0)*s;
}

// Function 449
float shadow (in vec3 p, in vec3 lPos) {
    float distanceToLight = distance (p, lPos);
    vec3 n = normal (p, EPSILON);
    float distanceToObject = raymarch (p + .01*n, normalize (lPos - p));
    bool isShadowed = distanceToObject < distanceToLight;
    return isShadowed ? .1 : 1.;
}

// Function 450
float calcSoftShadow( vec3 ro, vec3 rd, bool showSurface )
{
    float res = 1.0;
    const float tmax = 2.0;
    float t = 0.001;
    for( int i=0; i<64; i++ )
    {
     	float h = map(ro + t*rd, showSurface).x;
        res = min( res, 64.0*h/t );
    	t += clamp(h, 0.01,0.5);
        if( res<-1.0 || t>tmax ) break;
        
    }
    res = max(res,-1.0);
    return 0.25*(1.0+res)*(1.0+res)*(2.0-res); // smoothstep, in [-1,1]
}

// Function 451
float lightPointDiffuseSoftShadow(vec3 pos, vec3 lightPos, vec3 normal) {
	vec3 lightDir = normalize(lightPos - pos);
	float lightDist = length(lightPos - pos);
	float color = max(dot(normal, lightDir), 0.0) / (lightDist * lightDist);
	if (color > 0.00) color *= castSoftShadowRay(pos, lightPos);
	return max(0.0, color);
}

// Function 452
float shadow(vec3 pi, vec3 l,float k)
{
    float t = 1.0;
    float tmax = 30.0;
    float res = 1.0;
    for(int i=0; i<256; ++i)
    {
        vec3 p = pi + t * l;
        float idx;
        float d = map(p,idx);
        
        res = min(res, k * d/t);
        if(d < 0.0001 || t > tmax) break;
        
        t+=d;
    }
    return clamp(res,0.0,1.0);
}

// Function 453
float shadowSphere(vec3 ro, vec3 rd, vec3 p, float radius)
{
    vec3 d = p-ro;
    float b = dot(d, rd);
    if(b<0.)
        return 1.0;
   	vec3 c = ro+rd*b;
    float s = length(c-p)/radius;
    return max(0.,(min(1.,s)-.7)/.3);
}

// Function 454
float castShadowRay( in vec3 ro, in vec3 rd )
{
    for( uint i=0U; i<NUMBOXES; i++ )
    {
        mat4 ma; vec3 si; getLocation(i, ma, si);

        if( iBox( ro, rd, ma, si )>0.0 )
            return 0.0;
    }
	return 1.0;
}

// Function 455
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

// Function 456
float SoftShadow( in vec3 origin, in vec3 direction )
{
  float res = 2.0, t = 0.02, h;
  for ( int i=0; i<24; i++ )
  {
    h = MapPlane(origin+direction*t);
    res = min( res, 7.5*h/t );
    t += clamp( h, 0.05, 0.2 );
    if ( h<0.001 || t>2.5 ) break;
  }
  return clamp( res, 0.0, 1.0 );
}

// Function 457
vec3 postEffects( in vec3 col, in vec2 uv )
{    
    // gamma correction
	//col = pow( clamp(col,0.0,1.0), vec3(0.6) );
	//vignetting
	col *= 0.5+0.6*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.8 );
    //noise
    col -= snoise((uv*3.+iTime)*1000.)*.1;
    //col = mix(bw(col), col, sound*3.);
    col*=(1.6,.9,.9);
	return col;
}

// Function 458
float softShadow(vec3 pos, vec3 rayDir, float start, float end, float k ){
    float res = 1.0;
    float depth = start;
    int id;
    for(int counter = ZERO; counter < MAX_STEPS; counter++){
        float dist = getSDF(pos + rayDir * depth, id);
        if( abs(dist) < EPSILON){ return 0.0; }       
        if( depth > end){ break; }
        res = min(res, k*dist/depth);
        depth += dist;
    }
    return res;
}

// Function 459
float compute_shadows(in vec3 ro, in vec3 rd)
{
    float res = 1.0;
    float t = 0.01;
    for(int i = 0; i < 16; i++)
    {
        vec3 p = ro + rd * t;
        float d = intersect_shadow(p);
        res = min(res, max(d, 0.0) * 16.0 / t);
        if (res < 0.001)
            break;
        t += d * 0.5;
    }
    
    return res;
}

// Function 460
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

// Function 461
float softshadow(vec3 ro,vec3 rd) 
{
    float sh = 1.;
    float t = .02;
    float h = .0;
    for(int i=0;i<12;i++)  
	{
        if(t>20.)continue;
        h = map(ro+rd*t);
        sh = min(sh,4.*h/t);
        t += h;
    }
    return sh;
}

// Function 462
float shadow( vec3 origin, float min_t) {
    vec3 dir = normalize(lightPos - origin);
    // #define HARD_SHADOW
    #ifdef HARD_SHADOW
    return hardShadow(dir, origin, min_t);
    #else
    return softShadow(dir, origin, min_t, SHADOW_HARDNESS);
    #endif
}

// Function 463
float ShadowRay(vec3 pos, vec3 dir) {
    vec3 IDir=1./dir; vec3 cp,fp; vec4 C;
    float dist=0.;
    float FAR=boxfar(pos,1./dir,vec3(0.),vec3(32.));
    for (int i=0; i<48; i++) {
        if (dist>FAR) break;
        cp=pos+dir*dist;
        fp=floor(cp);
        C=texture(iChannel0,(PToUV(cp)+vec2(0.,1.))*ires);
        if (C.w>0.) return 0.;
        dist+=boxfar(cp,IDir,fp,fp+1.)+0.001;
    }
    return 1.;
}

// Function 464
float shadow(vec3 from, vec3 increment)
{
	const float minDist = 1.0;
	
	float res = 1.0;
	float t = 1.0;
	for(int i = 0; i < SHADOW_ITERATIONS; i++) {
		float m;
        float h = distf(from + increment * t,m);
        if(h < minDist)
            return 0.0;
		
		res = min(res, 4.0 * h / t);
        t += SHADOW_STEP;
    }
    return res;
}

// Function 465
vec3 effect(vec3 p)
{
	p *= mz * mx * my * sin(p.zxy); // sin(p.zxy) is based on iq tech from shader (Sculpture III)
	return vec3(min(min(func(p*mx), func(p*my)), func(p*mz))/.6);
}

// Function 466
float cShadow( in vec3 start, in vec3 ldir, in float md, in float p )
{    
	float t = EPSILON*4.0;
	float res = 1.0;
    for ( int i = 0; i < S_STEPS; ++i )
    {        
        float d = csDist( start + ldir * t );
        if ( d < EPSILON )
            return 0.0;
		
		res = min( res, p * d / t );
        t += d*.25;
		
		if ( t > md)
			break;
    }
    return res;
}

// Function 467
float shadow(v2 o,v2 i){
  const float a=32.;//shadow hardnes
  float r=1.,h =1.,t=.0005;//t=(self)intersection avoidance distance
  for(int j=0;j<IterSh;j++){
   h=dm(o+i*t).x;
   r=min(r,h*a/t);
   t+=clamp(h,.02,2.);}//limit max and min stepping distances
  return clamp(r,0.,1.);}

// Function 468
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

// Function 469
float calcShadow( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;
    
    float t = 0.01;
    for( int i=0; i<128; i++ )
    {
        vec3 pos = ro + t*rd;
        float h = map( pos ).x;
        res = min( res, k*max(h,0.0)/t );
        if( res<0.0001 ) break;
        t += clamp(h,0.01,0.5);
    }

    return res;
}

// Function 470
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = 0; j < 20; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += clamp (3. * h, 0.05, 0.2);
    if (sh < 0.05) break;
  }
  return 0.6 + 0.4 * sh;
}

// Function 471
float calcShadow(in vec3 ro, in vec3 rd) {
	float res = 1.0;
	float t = 0.01;
	for (int i = 0; i < 100; i++) {
		vec3 pos = ro + rd*t;
		float h = mapShadow(pos).x;
		res = min(res, 16.0*max(h, 0.0) / t);
		if (h<0.0001 || pos.y>3.0) break;

		t += clamp(h, 0.01, 0.2);
	}

	return clamp(res, 0.0, 1.0);
}

// Function 472
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

// Function 473
float softShadow( in vec3 ro, in vec3 rd )
{
	float scene = GetShadows(ro,rd);
	float alpha = 1.0 - 0.8*clamp(length(GetTransparency(ro,rd,9999.0)),0.0,1.0);
    return min(alpha,scene);
	
}

// Function 474
float softShadow(ray _r, vec3 _light)
{
    float tmin = 0.02;
    float tmax = 2.5;
        
    ray lr = _r;
    lr.o = _r.hp;
    lr.d = normalize(_light - _r.hp );
    
    float t = tmin;    
    float ss = 1.;
    for(int i=0; i<30; i++)
    {                
        vec3 p = lr.o + lr.d * t;
        float d = map(p).x;
        
        ss = min(ss, 1. * d / t);
        
        if(d < 0.002 || t > tmax)
            break;        
        
        t += clamp(d, .02, .1);
    }
    
    return clamp(ss,0.,1.);
}

// Function 475
bool intersectShadow( in vec3 ro, in vec3 rd, in float dist ) {
    float t;
	
	t = iSphere( ro, rd, movingSphere            );  if( t>eps && t<dist ) { return true; }
    t = iSphere( ro, rd, vec4( 4.0,1.0, 4.0,1.0) );  if( t>eps && t<dist ) { return true; }
#ifdef FULLBOX    
    t = iPlane( ro, rd, vec4( 0.0,-1.0, 0.0,5.49) ); if( t>eps && t<dist && ro.z+rd.z*t < 5.5 ) { return true; }
#endif
    return false; // optimisation: other planes don't cast shadows in this scene
}

// Function 476
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<1; i++ )
    {
		float h = df( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += h*.25;
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0., 1. );
}

// Function 477
float SoftShadow(  vec3 origin,  vec3 direction )
{
  float res = 2.0, t = 0.0, h;
  for ( int i=0; i<32; i++ )
  {
    h = Map(origin+direction*t);
    res = min( res, 6.5*h/t );
    t += clamp( h, 0.07, 0.6 );
    if ( h<0.0025 ) break;
  }
  return clamp( res, 0.0, 1.0 );
}

// Function 478
vec3 blur(vec2 uv,float rad)
{
    mat2 rot = mat2(cos(2.399),sin(2.399),-sin(2.399),cos(2.399));
	vec3 acc=vec3(0.0,0.0,0.0);
    vec2 pixel=vec2(0.002*iResolution.y/iResolution.x,0.002),angle=vec2(0.0,rad);;
    rad=1.0;
	for (int j=0;j<80;j++)
    {  
        rad += 1.0/rad;
	    angle*=rot;
        vec4 col=vec4(surface(uv+pixel*(rad-1.0)*angle),1.0); 
		acc+=col.xyz;
	}
	return acc/80.0;
}

// Function 479
float shadow( vec3 ro, vec3 rd, float mint, float maxt )
{
	float t = mint;
    for ( int i = 0; i < 128; ++i )
    {
        float h = scene( ro + rd * t );
        if ( h < 0.001 )
            return 0.0;
        t += h;
		
		if ( t > maxt )
			break;
    }
    return 1.0;
}

// Function 480
void draw_blur(in Ray ray, float radius, inout TraceResult cur_ctxt)
{
    float accum = 10.0;
    /*vec4 color;
    vec3 normal1;
    float t1 = traceSphere(ray.pos, ray.dir, radius/1.0, normal1);
    vec3 normal2;
    float t2 = traceSphere(ray.pos, ray.dir, radius/2.0, normal2);
    vec3 normal3;
    float t3 = traceSphere(ray.pos, ray.dir, radius/3.0, normal3);
    vec3 normal4;
    float t4 = traceSphere(ray.pos, ray.dir, radius/4.0, normal4);
    vec3 normal5;
    float t5 = traceSphere(ray.pos, ray.dir, radius + 0.04, normal5);
    
    bool b5 = (t5 != INF);
    bool b4 = (t4 != INF);
    bool b3 = (t3 != INF);
    bool b2 = (t2 != INF);
    bool b1 = (t1 != INF);
    
    accum = to_float(b5);*/
    float d = get_dist_ray_point(ray, vec3(0, 0, 0));
    
    vec3 normal5;
    float t5 = traceSphere(ray.pos, ray.dir, radius, normal5);
    
    if (t5 != INF)
    {           
        
        cur_ctxt.color = make_another_blur(cur_ctxt.fragCoord,
                                    max(26.0*d,0.0)).rgb;
        //cur_ctxt.color = vec3(1, 1, 1);
        cur_ctxt.materialType = EMISSION;
        cur_ctxt.alpha = min(1.0 - d/(radius) + 0.1, 0.7);
    }
    vec3 normal6;
    float t6 = traceSphere(ray.pos, ray.dir, radius + 0.04, normal6);
    if (t5 == INF && t6 != INF)
    {
        float dif = d - radius - 0.04;
        
        cur_ctxt.color = make_another_blur(cur_ctxt.fragCoord,
                                    max(16.0 * (1.0 - 25.0*dif),0.0)).rgb;
        
        cur_ctxt.materialType = EMISSION;
        //cur_ctxt.alpha = min(-(0.6/0.04) * dif , 0.7);
    }
    
    //cur_ctxt.alpha = GLOBAL_ALPHA;
    //cur_ctxt.materialType = EMISSION;
     /*//vec3 curPos = ray.pos + t * ray.dir;
*/
}

// Function 481
float softShadow(vec3 ro, vec3 rd, float maxDist) {
    float total = 0.;
    float s = 1.;
    
    for (int i = 0; i < SHADOW_STEPS; ++i) {
        float d = map(ro + rd * total);
        if (d < EPS) {
            s = 0.;
            break;
        }
        if (maxDist < total) break;
        s = min(s, SHADOW_SOFTNESS * d / total);
        total += d;
    }
    
    return s;
}

// Function 482
void DrawBallEffect(vec2 p, vec3 ballColor, inout vec4 color)
{
    vec4 sceneColor = color;
    
    p *= vec2(1.+noise(ballColor.xx), 1.+noise(ballColor.yy));
    p *= vec2(1., 0.75);
    
    p.x *= 1. + max(-p.y*28., 0.)*0.2;
    
    p.y += 0.25;
    float noise = fbm(p*12.+gT*2.);
    float d = max(abs(p.x)-noise*0.06, abs(p.y)-noise*0.20);
    vec4 colorTemp = mix(vec4(ballColor*0.4+color.rgb*0.6, hash(gT*0.1) * 3.0), color, smoothstep(0., noise*0.15, d));
    color.rgb = colorTemp.rgb; color.a = max(color.a, colorTemp.a);
    
    p.y += -0.05;
    noise = fbm(p*34.-gT*4.);
    d = max(abs(p.x)-noise*0.035, abs(p.y)-noise*0.25);
    colorTemp = mix(vec4(ballColor, hash(gT*0.1) * 3.0), color, smoothstep(0., noise*0.15, d));
    color.rgb = colorTemp.rgb; color.a = max(color.a, colorTemp.a);
    
    // fade by height
    color = mix(sceneColor, color, smoothstep(-0.3, 0.3+sin(p.y*20. + color.r*10. + gT*4.)*0.25, p.y));
}

// Function 483
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<80; i++ )
    {
		float h = df( ro + rd*t ).x;
        res = min( res, 8.*h/t );
        t += clamp( h, 0.01, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 484
float shadowMap (vec3 ro, vec3 nor, Celestial o) {
	
	// Light data
	float lRad = sun.radius;
	vec3 lDir = sun.origin - ro;
	float lDis = length(lDir);
	lDir = normalize(lDir);
	
	// Occluder data
	float oRad = o.radius;
	vec3 oDir = o.origin - ro;
	float oDis = length(oDir);
	oDir = normalize(oDir);
	
	// Determine light visible "around" the occluder
	float l = lDis * ( length(cross(lDir, oDir)) - (oRad / oDis) );
	l = smoothstep(-1.0, 1.0, -l / lRad);
    l *= smoothstep(0.0, 0.2, dot(lDir, oDir));
	l *= smoothstep(0.0, oRad, lDis - oDis);
	
	// Return a multiplier representing our softshadow
	return 1.0-l;
}

// Function 485
void sphSoftShadow( in vec3 ro, in vec3 rd, in vec3 sph, in float ra, inout float result )
{
    vec3	oc	= ro - sph;
    float	b	= dot( oc, rd );
    float	c	= dot( oc, oc ) - ra * ra;
    float	h	= b*b - c;

    float	res	= (b>0.0) ? step(-0.0001,c) : smoothstep( 0.0, 1.0, h*k/b );

    result	= min( result, res );
}

// Function 486
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

// Function 487
float ObjSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  vec2 cIdP;
  float sh, d, h;
  sh = 1.;
  cIdP = vec2 (-999.);
  d = 0.02;
  for (int j = 0; j < 40; j ++) {
    p = ro + d * rd;
    cId = floor (p.xz);
    if (cId != cIdP) {
      cIdP = cId;
      SetBldgParms ();
    }
    h = BldgDf (p, dstFar);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += min (0.05, 3. * h);
    if (h < 0.001) break;
  }
  return sh;
}

// Function 488
bool isShadowed(Ray ray,float minDist,float maxDist)
{
	vec3 p;
    return intersect(ray,minDist,maxDist,p)!=OBJ_NONE;
}

// Function 489
float testshadow( vec3 p, float dither )
{
	float shadow = 1.0;
	float s = 0.0; // this causes a problem in chrome: .05*dither;
	for ( int j=0; j < 5; j++ )
	{
		vec3 shadpos = p + s*sunDir;
		shadow = shadow - map(shadpos).a*shadow;
		
		s += .05;
	}
	return shadow;
}

// Function 490
column scrollEffect(float x, float w/*idth*/, float r/*adius*/)
{
    if (x >= w && x <= W - w)
        return column(0., H);
    x -= (W / 2.);
    float x0 = w - (W / 2.);
    r *= 4.;
    float t = sqrt(r * r - x0 * x0) - sqrt(r * r - x * x);
    float y = sin(t + iTime / 2.) * t;
    return column(y, H);
}

// Function 491
float castShadow(vec2 p, vec2 pos, float radius) {
   
    vec2 dir = normalize(pos - p);
    float distanceLight = length(p - pos);
    
    float lightFraction = radius * distanceLight;
    
    float totalDistance = 0.01;
    
    for(int i = 0; i < NUM_RAYS; ++i){
    	float sceneDistance = scene(p + dir * totalDistance);   
        
        if(sceneDistance < -radius) return 0.0;
        
        lightFraction = min(lightFraction, sceneDistance / distanceLight);
        
        //Go ahead
        totalDistance += max(1.0, abs(sceneDistance));
        if(totalDistance > distanceLight) break;
    }
    
    lightFraction = clamp((lightFraction * distanceLight + radius) / (2.0 * radius), 0.0, 1.0);
    lightFraction = smoothstep(0.0, 1.0, lightFraction);
    return lightFraction;
}

// Function 492
float Shadow( in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    float t = 0.06;
	float h;
	
    for (int i = 0; i < 5; i++)
	{
		h = Map( ro + rd*t );
		res = min(4.5*h / t, res);
		t += h+.2;
	}
    return max(res, 0.0);
}

// Function 493
vec4 Blur5(vec2 p) {
    
    vec3 e = vec3(1./iResolution.yy, 0);

	return (tx(p - e.zy) + tx(p - e.xz) + tx(p) + tx(p + e.xz) + tx( p +  e.zy))/5.;
}

// Function 494
float Shadow( in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    float t = 0.05;
	float h;
	
    for (int i = 0; i < 8; i++)
	{
		h = Map( ro + rd*t );
		res = min(6.0*h / t, res);
		t += h;
	}
    return max(res, 0.0);
}

// Function 495
void considerBlurCandidate(
    vec2 selfCoord,    
    vec2 candidateIndexDelta,
    inout vec4 inoutSelfState)
{
	vec2 candidateCoord = (selfCoord + candidateIndexDelta);
	vec4 candidateState = texture(iChannel0, (candidateCoord * sTexelSize));
    
    if (candidateState.w > inoutSelfState.w)
    {
        inoutSelfState = vec4(
            0.0,
            candidateState.y,
            candidateState.z,
            (candidateState.w - (kBlurFadeSteps * (kFadeRate * iTimeDelta))));
    }
}

// Function 496
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

// Function 497
float softshadow(in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k) {
    float res = 1.0, h, t = mint+.1*hash(ro+rd);
    for( int i=0; i<48; i++ ) {
        //  if (t < maxt) {
        h = DE( ro + rd*t ).x;
        res = min( res, k*h/t );
        t += .1;
        //  }
    }
    return clamp(res, 0., 1.);
}

// Function 498
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, in float k)
{
	float res = 1.0;
    float t = mint;
    float ph = 1e10; // big, such that y = 0 on the first iteration
    
    for( int i=0; i<20; i++ )
    {
		float h = map( ro + rd*t ).x;
       	res = min( res, k*h/t );
        t += h;
         if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 499
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

// Function 500
float shadow (in vec3 p, in vec3 n, in vec3 lPos)
{
	float distanceToLight = distance (p, lPos);
	float distanceToObject = raymarch (p + .01*n, normalize (lPos - p)).dist;
	bool isShadowed = distanceToObject < distanceToLight;
	return isShadowed ? .1 : 1.;
}

// Function 501
float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
{
    float ph = 1e10; // big, such that y = 0 on the first iteration
    float res = 1.0;
    for( float t=mint; t < maxt; )
    {
        float h = map(ro + rd*t);
        if( h<0.002 )
            return 0.0;
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, 10.0*d/max(0.0,t-y) );
        t += h;
    }
    return res;
}

// Function 502
vec3 effect(vec2 uv, vec3 col)
{
    float granularity = floor(yVar*20.+10.);
    
    if (mod(granularity, 2.) > 0.) {
        granularity += 1.;
    }
    
    if (granularity > 0.0) 
    {
        float dx = granularity / s.x;
        float dy = granularity / s.y;
        uv = vec2(dx*(floor(uv.x/dx) + 0.5),
                  dy*(floor(uv.y/dy) + 0.5));
        return bg(uv);
    }
    return col;
}

// Function 503
bool traceShadow(in vec2 origin, in vec2 toLight) {
    float maxDist = length(toLight);
    vec2 direction = toLight / maxDist;
    float totalDist = 0.0;
    for(int i = 0; i < STEPS; i++) {
        float dist = lookup(origin).x;
        if(dist < EPSILON) {
            return true;
        }
        origin += direction * dist;
        totalDist += dist;
        if(totalDist >= maxDist) {
            return false;
        }
        /*if(dot(origin, origin) > 10.0) {
            return false;
        }*/
    }
    return true;
}

// Function 504
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, in float hardness )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<32; i++ )
    {
		float h = dstScene( ro + rd*t );
        res = min( res, hardness*h/t );
        t += clamp( h, 0.06, 0.30 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 505
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

// Function 506
float rivet_shadow(vec2 uv, float s) {
	uv /= s;
	uv.y += .06;
	uv.x *= 2.;
	return ls(.3, .0, length(uv));
}

// Function 507
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=ZERO; i<50; i++ )
    {
        float h = map(ro + rd*t).x;
        res = min( res, smoothstep(0.0,1.0,k*h/t) );
		t += clamp( h, 0.01, 0.25 );
		if( res<0.005 || t>10.0 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 508
float shadow(vec3 from, vec3 increment)
{
	const float minDist = 1.0;
	
	float res = 1.0;
	float t = 1.0;
	for(int i = 0; i < SHADOW_ITERATIONS; i++) {
		float m;
        float h = distf2(from + increment * t,m);
        if(h < minDist)
            return 0.0;
		
		res = min(res, 8.0 * h / t);
        t += SHADOW_STEP;
    }
    return res;
}

// Function 509
float eliSoftShadow( in vec3 ro, in vec3 rd, in vec3 sphcen, in vec3 sphrad, in float k )
{
    vec3 oc = ro - sphcen;
    
    vec3 ocn = oc / sphrad;
    vec3 rdn = rd / sphrad;
    
    float a = dot( rdn, rdn );
	float b = dot( ocn, rdn );
	float c = dot( ocn, ocn );
	float h = b*b - a*(c-1.0);

    float t = (-b - sqrt( max(h,0.0) ))/a;

    return (h>0.0) ? step(t,0.0) : smoothstep(0.0, 1.0, -k*h/max(t,0.0) );
}

// Function 510
float shadow(vec3 dir, vec3 origin, float k, int max_steps) {
    float res = 1.0;
    float t = 0.1;
    for(int i = 0; i < max_steps; ++i) {
        float m = sceneMap3D(origin + t * dir);
        if(m < 0.0001) {
            return 0.0;
        }
        //res = min(1.0, (k * m / t));
        t += m;
    }
    return res;
}

// Function 511
float soft_shadow(vec3 ro, vec3 rd) {
    float res = 1.;
    float t = .0001;                     
	float h = 1.;
    for(int i = 0; i < shadow_steps_; i++) {         
        h = eval_scene(ro + rd*t).d;
        res = min(res, 4.*h/t);          
		t += clamp(h, .02, 1.);          
    }
    return clamp(res, 0., 1.);
}

// Function 512
float shadow( Ray r ) {
    float dist = 2.*MinimumDistance;
    float sh = 1.0;
	for( int i = 0; i < MaximumRaySteps; i++ ) {
        vec3 newLocation = r.origin + dist * r.direction;
		float h = DistanceEstimator(newLocation);
        if( h < MinimumDistance || dist > 10.)
            return sh;
        sh = min( sh, 32.*h/dist );
        dist += h;
    }
    return sh;
}

// Function 513
bool TraceShadow(vec3 ro, vec3 rd)
{
	bool hit;
	vec3 pos;
	vec3 hitNormal;
	int mat;
	float dist2 = VoxelTrace(ro+rd*0.6, rd, hit, hitNormal, pos, mat);
	return hit;
}

// Function 514
float shadow(vec3 ro,vec3 rd,float mint,float maxt,float k)
{
    float res = 1.0;
    float ph = 1e20;
    for( float t=mint; t < maxt; )
    {
        float h = world(ro + rd*t);
        if( h<0.001 )
            return 0.0;
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.0,t-y) );
        ph = h;
        t += h;
    }
    return res;
}

// Function 515
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.02;
  for (int j = VAR_ZERO; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.03 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 516
float shadow( in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    float t = .2;
    for (int i = 0; i < 12; i++)
	{
		float h = mapDE( ro + rd*t );
        if (h< -2.) break;
		res = min(10.*h / t, res);
		t += h+.2;
	}
    return max(res, .3);
}

// Function 517
float shadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    float ph = 1e10; // big, such that y = 0 on the first iteration
    
    for(int i=ZERO; i<32; i++)
    {
		float h = map( ro + rd*t ).x;
        // use this if you are getting artifact on the first iteration, or unroll the
        // first iteration out of the loop
        //float y = (i==0) ? 0.0 : h*h/(2.0*ph); 
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, 10.0*d/max(0.0,t-y) );
        ph = h;
        
        t += h;
        
        if( res<0.0001 || t>tmax ) break; 
    }
    return clamp( res, 0.03, 1.0 );
}

// Function 518
float Effect1(vec2 coords, float numSegments, float r, float cutoffMax, float cutoffMin, float thickness)
{
    float s = abs(sin(r * numSegments * 0.5));
    float dist = length(coords);
    
    if (dist <= cutoffMax && dist >= cutoffMin)
    {
        if (s < (dist * (dist * 0.5)) * thickness)
        {
            return 1.0;
        }
    }
    return 0.0;
    
    //return smoothstep(0.12f, 0.08f, abs(s));
}

// Function 519
float GetShadows( in vec3 ro, in vec3 rd)
{
	float shadowSatellite = shadowRaySphere(ro,rd,vec4(ballPos,SATELLITE_RADIUS));
	float shadowPlanet = shadowRaySphere(ro,rd,vec4(0.0,0.0,0.0,PLANET_RADIUS));
	
	return smoothstep(0.0,SOFTSHADOW_BANDWIDTH,min(shadowSatellite,shadowPlanet));
}

// Function 520
float GetShadows( in vec3 ro, in vec3 rd)
{
	float shadowSatellite = shadowRaySphere(ro,rd,vec4(ballPos,SATELLITE_RADIUS));
    
    
	float shadowPlanet = shadowRaySphere(ro,rd,vec4(0.0,0.0,0.0,PLANET_RADIUS));
	
	return smoothstep(0.0,SOFTSHADOW_BANDWIDTH,min(shadowSatellite,shadowPlanet));
}

// Function 521
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

// Function 522
int InShadow(vec3 p, vec3 L, out float t)
{
    int shadowHit = NO_HIT;
    Ray sr;
    sr.Direction = L;
    sr.Orgin = p + vec3(0.05);     
    t =  Trace(sr, shadowHit);
    return shadowHit;
}

// Function 523
vec3 blur(sampler2D sampler, vec2 uv, float sub) {
 
    vec3 sum = vec3(0.);
    float we = 1. / float(BLUR_SAMPLES);
    float an = radians(360. / float(BLUR_SAMPLES));
    for(int i = 0; i < BLUR_SAMPLES; i++) {
        float s = sin(an * float(i)) * BLUR_SIZE;
        float c = cos(an * float(i)) * BLUR_SIZE;
        sum += (texture(sampler, uv+vec2(c,s)).xyz - sub) * we;
    }
    return clamp(sum,0.,1.);
    
}

// Function 524
float SoftShadow( in vec3 origin, in vec3 direction )
{
  float res =1., t = 0.0, h;
  vec3 rayPos = vec3(origin+direction*t);

  for ( int i=0; i<NO_UNROLL(20); i++ )
  {
    h = MapTerrain(rayPos).x;

    res = min( res, 8.5*h/t );
    t += clamp( h, 0.01, 0.25);
    if ( h<0.005 ) break;
    rayPos = vec3(origin+direction*t);
  }
  return clamp( res, 0.0, 1.0 );
}

// Function 525
float Shadow( in vec3 ro, in vec3 rd, float mint)
{
    float res = 1.0;
    float t = .15;
    for( int i=0; i < 15; i++ )
    {
        float h = MapThorns(ro + rd*t);
		h = max( h, 0.0 );
        res = min( res, 4.0*h/t );
        t+= clamp( h*.6, 0.05, .1);
		if(h < .001) break;
    }
    return clamp(res,0.05,1.0);
}

// Function 526
bool shadowed(const in vec3 pos){
	vec3 lightDir = normalize(lightPos - pos);
    Ray toLightRay = Ray(pos + lightDir * .01, lightDir);
    Ray2D toLightRay2D = Ray2D(toLightRay.origin.xz, normalize(toLightRay.dir.xz));
    vec2 modPoint = vec2(floor(toLightRay.origin.x), floor(toLightRay.origin.z)) + .5;
    HitRecord rec;
    for (int i=0; i<10; i++) {
        Box box;
        if(boxAtPos(modPoint, box) && box_hit(box, toLightRay, EPS, rec))
        	return true;
        modPoint += getNextCellAlongVec(toLightRay2D, modPoint);
        //TODO additional break condition should be envolved(based on top plane)
    }
    return false;
}

// Function 527
void effectSmear(inout vec4 col, vec2 coord)
{
    vec2 g=getGrad(coord,.5);
    col=getCol(coord+g.yx*vec2(1,-1)*.7);
}

// Function 528
float SoftShadow(vec3 ro, vec3 rd)
{
    float k = 16.0;
    float res = 1.0;
    float t = 0.1;          // min-t see http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
    for (int i=0; i<48; i++)
    {
        float h = Dist(ro + rd * t);
        res = min(res, k*h/t);
        t += h;
        if (t > 8.0) break; // max-t
    }
    return clamp(res, 0.0, 1.0);
}

// Function 529
float calcShadow(in vec3 ro, in vec3 rd, float tmax) {
    float r = 1.;
    float t = 0.;
    for(int i = 0; i < 128; i++) {
        float h = map(ro + t * rd);
        r = min(r, tmax*h/t);
        if (r < 0.01) break;
        if (t > tmax) break;
        t += h;
    }
    return clamp(r, 0., 1.);
}

// Function 530
float calcshadow(vec3 o, vec3 rd, float tmin, float tmax)
{
    float k = tmin;
    float shadow = 1.;
    for(int i = 0; i < 20; ++i)
    {
        vec4 res = map(o + rd*k);
        shadow = min(shadow, res.x*1.5);
        
        k+=res.x;
        
        if(k > tmax)
        {
            break;
        }
    }
    
    return shadow;
}

// Function 531
float calcShadow(Ray ray, float maxT, float k)
{
    float res = 1.0;
    float ph = 1e20;
    int i = 0;
    for (float t = hitThreshold * 50.; t < maxT; )
    {
        float h = map(rayToPos(ray, t)).t;
        if (h < hitThreshold)
        {
            return 0.;
        }
        float hsqr = pow(h, 2.);
        float y = hsqr/(2. * ph);
        float d = sqrt(hsqr - pow(y, 2.));
        res = min(res, k * d / max(0., t - y));
        ph += h;
        t += h;
        i += 3;
        if (i > maxSteps)
        {
            break;
        }
    }
    return res;
}

// Function 532
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<32; i++ )
    {
		float h = dstScene( ro + rd*t );
        res = min( res, 32.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 533
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

// Function 534
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float res = 1.0;
    float t = mint;
    float ph = 1e10;
    
    for( int i=0; i<32; i++ )
    {
        float h = map( ro + rd*t ).dist;
        res = min( res, 10.0*h/t );
        t += h;
        if( res<0.0001 || t>tmax ) break;
        
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 535
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

// Function 536
float calcShadow( in vec3 ro, in vec3 rd, in vec3 v[16] )
{
    float t = 1.0;
    
    for( int i=0; i<16; i++ ) // for each vertex
    for( int j=0; j< 4; j++ ) // connect it to its 4 neighbors
    {
        int a = i;
        int b = a ^ (1<<j); // change one bit/dimension
        if( a<b )           // skip edge if already visited
        {
            t = min( t, softShadowCapsule( ro, rd, v[a], v[b], rad ) );
        }
    }    

    return t;
}

// Function 537
float calcShadow(vec3 p, vec3 ro, vec3 lightPos) {
    vec3 rd = normalize(lightPos - p);
    
    float res = 1.0;
    float t = 0.05;
    for(float i = 0.0; i < 45.0; i++)
    {
        float h = map(p + rd * t, ro).x;
        if (h < MIN_DIST * t)
            return 0.0; // Hit an object - Full shadow.
        
        res = min(res, 48.0 * h / t);
        t += h;
        
        if (t > 20.0)
            break; // Marched far enough - Stop.
    }
    
    return res;
}

// Function 538
float SoftShadow( in vec3 origin, in vec3 direction )
{
  float res = 2.0;
  float t = 0.0;
  float hardness = 6.50;
  for ( int i=0; i<10; i++ )
  {
    float h = Map(origin+direction*t);
    res = min( res, hardness*h/t );
    t += clamp( h, 0.02, 0.075 );
    if ( h<0.002 ) break;
  }
  return clamp( res, 0.0, 1.0 );
}

// Function 539
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

// Function 540
float march_shadows(vec3 p, vec3 d, float k)
{
    float r = 1.;
    float dep = 0.001+SHADOW_EPS;
    while(dep < FAR)
    {
        vec3 p = p+d*dep;
        float dist = get_terrain(p.xy).w;
        if(p.z < dist+SHADOW_EPS) 0.;
        
        r = min(r, k*(p.z-dist)/dep);
        dep += SHADOW_STEP;
    }
    return r;
}

// Function 541
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

// Function 542
float CalculateShadow(in RayHit hit)
{
    vec3 lightOrigin = hit.surfPos + (vec3(-SunDir.x, SunDir.y, -SunDir.z) * (ShadowDistance + SoftShadowOffset));
    vec3 lightRay    = normalize(hit.surfPos - lightOrigin);
    vec3 point       = vec3(0.0);
    
    vec2  sdf    = vec2(FarClip, 0.0);
    float depth  = 0.1;
    float result = 1.0;
        
    for(int steps = 0; (depth < ShadowDistance) && (steps < 8); ++steps)
    {
    	point  = (hit.surfPos + (-lightRay * depth));
        sdf    = Scene_SDF(point, hit);
        result = min(result, (SoftShadowFactor * sdf.x) / depth);
        depth += sdf.x;
    }
    
    return clamp((sdf.x < Epsilon ? 0.0 : result), 0.3, 1.0);
}

// Function 543
float shadow(i3 o,i3 i){
 const float a=32.;//shadow hardnes
 float r=1.,h =1.,t=.0005;//t=(self)intersection avoidance distance
 for(int j=0;j<IterSh;j++){
  h=dm(o+i*t).x;
  r=min(r,h*a/t);
  t+=clamp(h,.02,2.);}//limit max and min stepping distances
 return clamp(r,0.,1.);}

// Function 544
float shadow(vec3 ro, vec3 rd)
{
    float h = 0.;
    float k =3.5;//shadowSmooth
    float res = 1.;
    float t = 0.2; //bias
    for (int i = 0; t < 15.; i++) // t < shadowMaxDist
    {
        h = map(ro + rd * t).w;
		res = min(res, k*h / t);
        if (h < HITTHRESHOLD)
        {
           break;
        }
        t = t + h;
    }
    return clamp(res+0.05,0.,1.);
}

// Function 545
float calculateShadow(in vec3 pos)
{
    vec3 n = normalize(lightPosition.xyz - pos * lightPosition.w);
    vec3 u = perpendicularVector(n);
	vec3 v = cross(u, n);
    vec3 dir = n;
    
    float result = 0.0;
    for (int s = 0; s < SHADOW_SAMPLES; ++s)
    {
        dir = randomDirection(n, u, v, 0.02, dir + pos + result);
	    for (int i = 0; i < NumSpheres; ++i)
        {
    	    result += float(calculateSphereIntersection(pos, dir, spheres[i]) > 0.0);
        }
    }
    
    return 1.0 - result / float(SHADOW_SAMPLES);
}

// Function 546
float softShadow(vec3 ro, vec3 rd, float mint, float tmax, float power) {
  float res = 1.;
  float t = mint;
  float ph = 1e10;
  for(int i = 0; i < maxShadowIterations; i++) {
    float h = scene(ro + rd * t).x;

    // pattern 1
    // res = min(res, power * h / t);

    // pattern 2
    float y = h * h / (2. * ph);
    float d = sqrt(h * h - y * y);
    res = min(res, power * d / max(0., t - y));
    ph = h;

    t += h;

    float e = EPS;
    if(res < e || t > tmax) break;
  }
  return clamp(res, 0., 1.);
}

// Function 547
float GetShadows(in vec3 ro, in vec3 rd, float k){
    float res = 1.0;
    float d;
    float t = 0.001;
    
    for(int i = 0; i < MAX_STEPS; i++)
    {
        d = map(ro + rd * t).d;
        if(d < MIN_DIST){
            return 0.0;
        }
        res = min(res, k * d / t);
        t += d;
    }
    return res;
}

// Function 548
float calcShadow(vec3 p, vec3 lightPos) {
	// Thanks iq.
	vec3 rd = normalize(lightPos - p);
	float res = 1.,
	      t = .1;
	for (float i = 0.; i < 32.; i++) {
		float h = map(p + rd * t).d;
		res = min(res, 10. * h / t);
		t += h;
		if (res < .001 || t > 3.) break;
	}

	return clamp(res, 0., 1.);
}

// Function 549
float blurWeight(
    float radiusFraction)
{
    float result = smoothstep(1.0, 0.0, radiusFraction);
    result = (result * result * result);
    return result;
}

// Function 550
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
	float h = 1.0;
    for( int i=0; i<32; i++ )
    {
        h = map(ro + rd*t).x;
        res = min( res, k*h/t );
		t += clamp( h, 0.02, 2.0 );
        if( res<0.0001 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 551
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, in float time, in float doDisplace )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<25; i++ )
    {
		float h = map( ro + rd*t, time, doDisplace ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.025, 0.10 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 552
float softShadow(vec3 ro, vec3 lp, float k, float t){

    // More would be nicer. More is always nicer, but not really affordable.
    const int maxIterationsShad = 24; 
    
    vec3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = 0.0015;  // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

         
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        h = clamp(h, .1, .5); // max(h, .02);//
        dist += h;

        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (shade<.001 || dist > end) break; 
    }

    // I've added a constant to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me. Sometimes, I'll add AO also, just for kicks. :)
    return min(max(shade, 0.) + .05, 1.); 
}

// Function 553
float ts_shadow_eval( vec2 lookup, float test )
    { return lookup.x != lookup.y ? parabolstep( lookup.x, lookup.y, test ) : 1.; }

// Function 554
float softshadow( vec3 ro, vec3 rd, float mint, float maxt, float k ){
    float res = 1.0;
    for( float t=mint; t < maxt; ){
        float h = map(ro + rd*t, 0.).x;
        if( h<0.001 ) return 0.2;
        res = min( res, k*h/t );
        t += h;
    }
    return res+0.2;
}

// Function 555
void effect_buf_logic(out vec4 fragColor, in vec2 fragCoord, ivec2 ipx) {
    if ((allData.flag0 != 1.) || (g_time < extime + 0.1)) {
        fragColor = vec4(0., 0., 0., -1.);
        return;
    }
    vec4 retx = vec4(0., 0., 0., -1.);
    float anim_t2 = 1. - get_animstate(clamp((g_time - allData.card_put_anim - 0.5)*2., 0., 1.));
    if (anim_t2 > 0.) {
        retx = load_eff_buf();
    }

    if (allData.last_selected_card >= 0.) {
        float anim_t = get_animstate(clamp((g_time - allData.card_select_anim)*2., 0., 1.));
        if (anim_t >= 1.) {
            if (card_get_hit(allData.mouse_pos) >= 0) {
                vec4 tc = load_board((card_get_hit(allData.mouse_pos) > 9 ? card_get_hit(allData.mouse_pos) - 10 : card_get_hit(allData.mouse_pos)));
                vec4 tc2 = load_card(int(allData.last_selected_card));
                if ((is_c_cr(int(tc.w)))&&(!is_c_cr(int(tc2.w))))
                    fragColor = tc2;
                else
                    fragColor = retx;
                return;
            }
            if (hpmp_get_hit(allData.mouse_pos) > 0) {
                vec4 tc2 = load_card(int(allData.last_selected_card));
                fragColor = tc2;
                return;
            }
        }
    }
    fragColor = retx;
    return;
}

// Function 556
float calcSoftShadowBk( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;
    float t = 0.01;
    for( int i=0; i<16; i++ )
    {
        float h = mapBk(ro + rd*t );
        res = min( res, smoothstep(0.0,1.0,k*h/t) );
        t += clamp( h, 10.0, 100.0 );
		if( res<0.01 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 557
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

// Function 558
float calcShadow(vec3 sp,float imax, int n){
   vec3 dp = normalize(lp-sp);
    float i2 = 0.1;
   
        
        i2 += max(abs(map(sp+dp*i2).x/1.0) , 0.01);
        
        if(abs(map(sp+dp*i2).x) < 0.01){
            
            if(map(sp+dp*i2).y < -1.5){
                
                vec3 norm = calcNormal(sp+dp*i2);
                float fren =pow(1.0 - max(dot(-dp,norm),0.0),5.0);
     	        fren = mix(0.1,1.0,fren);
            return 1.0-fren;
                
                
            }else if(map(sp+dp*i2).y < 2.0){
        
            return 0.0;
            }else{
                
            return 1.0;
                
            }
            
        }else if( i2 > imax) return 1.0;
            
    
    
    return 1.0;
}

// Function 559
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

// Function 560
vec3 shadow(Ray r, float len) {
    for(int i = 0; i < 4; ++i) if(intersect(r, spheres[i]).t < len) return vec3(0);
    return lcolor;
}

// Function 561
bool isInOtherSphereShadow(vec3 p, Sphere thisSphere, Light light)
{
	for(int i=0; i<numberOfSpheres; i++)
	{
		if(isInShadow(p, spheres[i], light))
			return true;
	}
	return false;
}

// Function 562
float Shadow( in vec3 p, in vec3 toLight )
{
    const float softness = 1.;
    toLight = toLight-p;
    float l = length(toLight);
    vec3 ray = toLight/l;
    float epsilon = .001;
    float t = shadowStartDistance;
    float h = 1.;
    float minh = 1e30;
    for ( int i=0; i < shadowLoopCount; i++ )
    {
        if ( h < epsilon || t > l )
            break;
        h = SDF(p+ray*t);
        minh = min(minh,h/max(t*softness,1.));
        t += h;
    }
    return smoothstep(epsilon,.03,minh);
}

// Function 563
vec3 effect(vec2 v) 
{
   	vec2 c0 = vec2(30.,20.);
    vec2 c1 = vec2(10.,40.);
    
    vec2 n=floor(v);
    vec2 f=fract(v);
    
    vec3 col;col.x=10.;
    
    for( float j=-1.; j<=1.; j+=1. )
    {
        for( float i=-1.; i<=1.; i+=1. )
        {
            vec2 g = vec2( i, j);
            
            vec2 ng = n+g;
            float ng0 = dot(ng,c0);
            float ng1 = dot(ng,c1);
            vec2 ng01 = vec2(ng0,ng1);
            vec2 hash = fract(cos(ng01)*iTime*0.2);
            
            vec2 o=sin(m2pi*hash)*.5+.5;
            
            vec2 r=g+o-f;
            
            float d=dot(r,r);
            
            if( d < col.x ) 
                col = vec3 (d, r);
        }
    }
     
    return col.xzz;
}

// Function 564
float MarchShadowRay(vec3 aSamplePos, vec3 aLightPos, float aRand)
{
    vec3 vRD = aLightPos - aSamplePos;
    float vMaxMarchDist = min(SHADOW_FAR, length(vRD));
    vRD = normalize(vRD);
    
    float vTrI = 1.0;
    
    float vNormRandStep = aRand * gcRcpShadowSteps;
    float vP = gcRcpShadowSteps;
    float vD = 0.0;
    float vNextD;
    
    for (int vN = 0; vN < SHADOW_STEPS; ++vN)
    {
        vNextD = pow(vP + vNormRandStep, SHADOW_EXP) * vMaxMarchDist;
        
        float vSS = vNextD - vD;
    	vec3 vPos = aSamplePos + vRD * vD;   

        vec2 vDens = VolumeDensity(vPos);
        float vSampleE = vDens.x + vDens.y;
        float vOpticalDepth = vSampleE * vSS;
        float vTr = exp(-vOpticalDepth);
        if (vPos.y >= 1.5)
        {
            vTr = 0.0;
        }
        
        vTrI *= vTr;
        if (vTrI < 0.00)
        {
            vTrI = 0.0;
            break;
        }
        vD = vNextD; 
        vP += gcRcpShadowSteps;
    }

    return vTrI;
}

// Function 565
float shadow( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;
    float t = NEAR;
    for(int i = 0; i<10;i++) {
        float h = mapActual(ro + rd*t).x;
        if( h<0.001 || t>FAR)
            return 0.0;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}

// Function 566
float shadow(vec3 ro, vec3 rd){
    float t = 0.01;
    float d = 0.0;
    float shadow = 1.0;
    for(int iter = 0; iter < 256; iter++){
        d = map(ro + rd * t);
        if(d < 0.0001){
            return 0.0;
        }
        if(t > length(ro - lightPos) - 0.5){
            break;
        }
        shadow = min(shadow, 128.0 * d / t);
        t += d;
    }
    return shadow;
}

// Function 567
float calcFakeAOAndShadow( in vec3 pos ) { 
    float r = (1.-abs(pos.x)/30.5);
    
    r *= max( min( .35-pos.z / 40., 1.), 0.65);
    r *= .5+.5*smoothstep( -66., -.65, pos.z);
    
    if( pos.y < 25. ) r *= 1.-smoothstep( 18., 25., .5*pos.y+abs(pos.x) ) * (.6+pos.y/25.);
    r *= 1.-smoothstep(5., 8., abs(pos.x) ) * .75 * (smoothstep( 60.,63.,abs(pos.z)));
    
    return clamp(r, 0., 1.);
}

// Function 568
float shadow(vec3 ro, vec3 rd, float id)
{
    float v = 1.;
    for(float i=0.; i<sphereNum;i++)
    {
        if(blinkID[int(i)])
            continue;
        if(i!=id)
        {    
            float tt = mt+i*2.;
            float ds = shadowSphere(ro, rd, spherePos[int(i)], sRadius);
            v*=ds;
            if(ds==0.)
                return 0.;
        }
    }
    return v*v*(3.-2.*v);
}

// Function 569
float ObjSShadow (vec3 ro, vec3 rd, float dLight)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 40; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05 || d > dLight) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 570
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

// Function 571
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

// Function 572
float shadow(vec3 pos, vec3 lPos, MatSpace ps)
{   
    vec3 dir = lPos - pos;  // Light direction & disantce
    
    float len = length(dir);
    dir /= len;				// It's normalized now
    
    pos += dir * MIN_DST * 2.0;  // Get out of the surface
    
    float dst = SDF(pos, ps).x; // Get the SDF
    
    // Start casting the ray
    float t = 0.0;
    float obscurance = 1.0;
    
    while (t < len)
    {
        if (dst < MIN_DST) return 0.0; 
        obscurance = min(obscurance, (20.0 * dst / t)); 
        t += dst;
        pos += dst * dir;
        dst = SDF(pos, ps).x;
    }
    return obscurance;     
}

// Function 573
float shadow(vec3 o)
{
    float mint=.001;
    float maxt=5.;
    float k = 30.;
    float res = 1.;
    // for( float t=mint; t < maxt; )
    float ph = 1e20;
    float t=mint;
    for( int i=ZERO; i < 50; i++)
    {
        float h = sdScene(o + ldir*t).x;
        h *= .5;
        if(abs(h)<MIN_DIST) return 0.;

        res = min( res, k*h/t);
        
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.0,t-y));
        ph = h;
        t += h;

        if(t >= maxt) break;
    }
    return smoothstep(.5, .51, res);
}

// Function 574
float is_shadow(vec3 hit, vec3 lightpos) {
    vec3 rayDir = normalize(lightpos - hit);
    float maxDistance = abs(length(lightpos - hit));
    return shadow(hit, rayDir, 0.1, maxDistance, 32.0);
}

// Function 575
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.02;
  for (int j = 0; j < 40; j ++) {
    h = BldgDf (ro + rd * d, dstFar);
    sh = min (sh, smoothstep (0., 1., 20. * h / d));
    d += min (0.05, 3. * h);
    if (h < 0.001) break;
  }
  return max (sh, 0.);
}

// Function 576
float getDottedShadow(vec2 fragCoord)
{
    vec2 uv = fragCoord;
    uv *= mat2(cos(.8+vec4(0, 11, 33, 0)));
    uv = mod(uv*.3, 1.);
    float res = 0.;
    float shadow = readShadow(fragCoord);
    shadow = saturate(1. - shadow*1.+ .01*readAO(fragCoord));
    res = smoothstep(shadow, shadow+.1, pow(length(uv-.5), 2.));
    return res;
}

// Function 577
float softshadow(vec3 pos, vec3 rayDir, float mint, float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = distfunc( pos + rayDir*t );
        res = min( res, 2.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 578
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

// Function 579
float shadow (in Ray ray, in vec3 lPos)
{
    float distToLight = distance (lPos, ray.ro);
    float dist = .0;

    for (int i = 0; i < MAX_ITER; ++i) {
        float tmp = map (ray.ro + dist * ray.rd);
        if (tmp < EPSILON) {
            if (dist < distToLight)
                return .125;
            else
                return 1.;
        }
        dist += tmp * STEP_SIZE;
    }

    return 1.;
}

// Function 580
column planetEffect(float x, float w/*idth*/, float r/*adius*/)
{
    if (x >= w && x <= W - w)
        return column(0., H);
    x -= (W / 2.);
    float x0 = w - (W / 2.);
    float y = sqrt(r * r - x0 * x0) - sqrt(r * r - x * x);
    return column(y, H);
}

// Function 581
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, in float k )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<64; i++ )
    {
		float h = dstScene( ro + rd*t );
        res = min( res, k*h/t );
        t += clamp( h, 0.07, .5 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 582
vec3 effect(vec2 uv) 
{
    uv/=8.;
    uv+=0.5;
    
	bgColor = textureLod(iChannel2, uv, 0.0).rgga;
    
    curlExtent = (sin((iTime)*0.3)*0.5+0.5);
    
    if (iMouse.z>0.) curlExtent = 1.-iMouse.y/iResolution.y;
        
	amount = curlExtent * (maxAmount - minAmount) + minAmount;
	cylinderCenter = amount;
	cylinderAngle = 2.0 * PI * amount;

    const float angle = 30.0 * PI / 180.0;
    float c = cos(-angle);
    float s = sin(-angle);
    mat3 rotation = mat3(c, s, 0, -s, c, 0, 0.12, 0.258, 1);
    c = cos(angle);
    s = sin(angle);
    mat3 rrotation = mat3(c, s, 0, -s, c, 0, 0.15, -0.5, 1);
    vec3 point = rotation * vec3(uv, 1.0);
    float yc = point.y - cylinderCenter;
    vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
    if (yc < -cylinderRadius) // See through to background
    {
        color = bgColor;
    } 
    else if (yc > cylinderRadius) // Flat surface
    {
        
        color = textureLod(iChannel1, uv, 0.0);
    } 
    else 
    {
        float hitAngle = (acos(yc / cylinderRadius) + cylinderAngle) - PI;
        float hitAngleMod = mod(hitAngle, 2.0 * PI);
        if ((hitAngleMod > PI && amount < 0.5) || (hitAngleMod > PI/2.0 && amount < 0.0)) 
        {
            color = seeThrough(yc, uv, rotation, rrotation);
        } 
        else 
        {
            point = hitPoint(hitAngle, yc, point, rrotation);
            if (point.x < 0.0 || point.y < 0.0 || point.x > 1.0 || point.y > 1.0) 
            {
                color = seeThroughWithShadow(yc, uv, point, rotation, rrotation);
            } 
            else 
            {
                color = backside(yc, point);
                vec4 otherColor;
                if (yc < 0.0) 
                {
                    float shado = 1.0 - (sqrt(pow(point.x - 0.5, 2.0) + pow(point.y - 0.5, 2.0)) / 0.71);
                    shado *= pow(-yc / cylinderRadius, 3.0);
                    shado *= 0.5;
                    otherColor = vec4(0.0, 0.0, 0.0, shado);
                } 
                else 
                {
                    otherColor = textureLod(iChannel1, uv, 0.0);
                }
                color = antiAlias(color, otherColor, cylinderRadius - abs(yc));
            }
        }
    }
    return color.rgb;
}

// Function 583
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 584
float calcSoftShadow( in vec3 ro, in vec3 rd, float k )
{  
    float res = 1.0;
    float t = 0.01;
    for( int i=0; i<32; i++ )
    {
        float h = Scene(ro + rd*t ).x;
        res = min( res, smoothstep(0.0,1.0,k*h/t) );
        t += clamp( h, 0.004, 0.1 );
		if( res<0.001 ) break;
    }
    return clamp(res*res,0.0,1.0);
}

// Function 585
float SoftShadow( in vec3 ro, in vec3 rd, float maxDist, float mint, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<40; i++ )
    {
		if( t > maxDist ) continue; 
        float h = Map(ro + rd*t,1.0).x;
        res = min( res, k*h/t );
        t += h*STEP_REDUCTION;
    }
    return clamp(res,0.0,1.0);
}

// Function 586
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.03 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.7 + 0.3 * sh;
}

// Function 587
void add_effects(inout vec4 fragColor, vec2 fragCoord, bool is_thumbnail)
{
    if (is_demo_mode_enabled(is_thumbnail))
        return;

    vec4 cam_pos = load(ADDR_CAM_POS);
    vec3 fireball = get_fireball_offset(g_animTime) + FIREBALL_ORIGIN;
    float pain = linear_step(80., 16., length(cam_pos.xyz - fireball));
    vec3 lava_delta = abs(cam_pos.xyz - clamp(cam_pos.xyz, LAVA_BOUNDS[0], LAVA_BOUNDS[1]));
    float lava_dist = max3(lava_delta.x, lava_delta.y, lava_delta.z);
    if (lava_dist <= 32.)
        pain = mix(.5, .75, clamp(fract(g_animTime*4.)*2.+-1., 0., 1.));
    if (lava_dist <= 0.)
        pain += .45;
   	fragColor.rgb = mix(fragColor.rgb, vec3(1., .125, .0), sqr(clamp(pain, 0., 1.)) * .75);
    
    if (!print_countdown(fragColor, fragCoord))
    	print_skill_message(fragColor, fragCoord, cam_pos.xyz);
}

// Function 588
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

// Function 589
float SampleShadow(int id, vec2 uv)
{
    float a = atan(uv.y, uv.x)/tau + 0.5;
    float r = length(uv);
    
    float idn = float(id)/iResolution.y;
    
    float s = texture(iChannel0, vec2(a, idn) + hpo).x;
    
    return 1.0-smoothstep(s, s+0.02, length(uv));    
}

// Function 590
float calcShadow(vec3 p, vec3 ld) {
	// Thanks iq.
	float s = 1., t = .1;
	for (float i = Z0; i < 30.; i++)
	{
		float h = tubez(t * ld + p).d;
		s = min(s, 20. * h / t);
		t += h;
		if (s < .01 || t > 25.) break;
	}

	return clamp(s, 0., 1.);
}

// Function 591
vec4 Blur(vec2 p) {
    

	// Kernel matrix dimension, and a half dimension calculation.
    const int mDim = 5, halfDim = (mDim - 1)/2;

    // You can experiment with different Laplacian related setups here. Obviously, when 
    // using the 3x3, you have to change "mDim" above to "3." There are widely varying 
    // numerical variances as well, so that has to be considered also.
    float kernel[mDim*mDim] = float[mDim*mDim](
        
    // Gaussian blur.
    1.,  4., 7.,  4.,  1.,
    4., 16.,  26., 16.,  4.,
    7., 26.,  41., .26, 7.,
    4., 16.,  26., 16.,  4.,
    1.,  4., 7.,  4.,  1.);

    //// Average blur.
    //1.,  1., 1.,  1.,  1.,
    //1.,  1., 1.,  1.,  1.,
    //1.,  1., 1.,  1.,  1.,
    //1.,  1., 1.,  1.,  1.,
    //1.,  1., 1.,  1.,  1.);
     
    
    // Calculating the Gaussian entries.
    //float sigma = 7.;
    //for (int j = 0; j < mDim*mDim; j++) kernel[j] = 0.;
    //int halfSize = (mDim*mDim - 1)/2;
    //for (int j = 0; j <= halfSize; ++j){
    //    kernel[halfSize + j] = kernel[halfSize - j] = nPDF(float(j), sigma);
    //}
 
    
    float total = 0.;
    
    //get the normalization factor (as the gaussian has been clamped)
    //for (int j = 0; j < mDim; j++)  total += kernel[j];
    for (int j = 0; j < mDim*mDim; j++) total += kernel[j];
    
    // Initiate the color. In this example, we only want the XY values, but just
    // in case you'd like to apply this elsewhere, I've included all four texture
    // channnels.
    vec4 col = vec4(0);
    
    // We're indexing neighboring pixels, so make this a pixel width.
    float px = 1./iResolution.y; 

    
    // There's a million boring ways to apply a kernal matrix to a pixel, and this 
    // is one of them. :)
    for (int j=0; j<mDim; j++){
        for (int i=0; i<mDim; i++){ 
            
            col += kernel[j*mDim + i]*tx(fract(p + vec2(i - halfDim, j - halfDim)*px));
        }
    }


    return col/total;
}

// Function 592
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.02;
  for (int j = VAR_ZERO; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05 || d > dstFar) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 593
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh = 1.;
  float d = 0.07 * szFac;
  for (int i = 0; i < 50; i++) {
    float h = ObjDf (ro + rd * d);
    sh = min (sh, 20. * h / d);
    d += 0.07 * szFac;
    if (h < 0.001) break;
  }
  return clamp (sh, 0., 1.);
}

// Function 594
float softshadow(in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    float t = 0.05;
    for(int i = 0; i < 32; i++)
    {
		float h = map(ro + rd * t);
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.02, 0.1);
        if(h < 0.001 || t > 5.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

// Function 595
float shadowmarch(vec3 point, vec3 light){
	vec3 delta = light - point;
	float dmax = length(delta);
	vec3 ray = delta/dmax;
	
	float shadow = 1.0;
	float dsum = 0.1;
	for(int i=0; i<shadow_iterations; i++){
		vec3 p = point + ray*dsum;
		float d = dist(p);
		if(d < 1e-6) return 0.0;
		
		dsum += max(min_step, d*step_fraction);
		shadow = min(shadow, 128.0*d/dsum);
		if(dsum > dmax) return shadow;
	}
	
	return shadow;
}

// Function 596
vec4 HandsShadow(vec2 uv , float l, float w, float t , float b)
{
    vec4 col = vec4(0.);
    
    uv.y += .05;
    uv.x -= .02;
    float h = segmentT(uv,vec2(0,0.),rot(vec2(0.,.8 * l)  ,t * TAU/60.  ) , w);
    uv.y += .02;
    uv *= .45;
    vec4 shape = vec4(ShapeN(rot( uv * 1.5, -t* TAU/60.) , 7,/* Width*/6. * w,/* Height*/ l - .2),1.);
    col += shape;
    col *=  h + shape;
    
    
   
    
    
    col -= length(uv.y) * 2.; // attenuate based on height
    
    
    
    col = max (col,vec4(0.));
    return col * .1;
}

// Function 597
vec3 smokeEffect(vec2 uv) {
    vec3 col = vec3(0.0);
    // time scale
    float v = 0.0002;
    vec3 smoke = vec3(1.0);
    //uv += mo * 10.0; 
   
    vec2 scale = uv * 0.5 ;
    vec2 turbulence = TURBULENCE * vec2(noise(vec2(uv.x * 3.5, uv.y * 3.2)), noise(vec2(uv.x * 2.2, uv.y * 1.5)));
    scale += turbulence;
	float n1 = fbm(vec2(scale.x - abs(sin(iTime * v * 2.0)), scale.y - 50.0 * abs(sin(iTime * v))));
    col =  mix( col, smoke, smoothstep(0.5, 0.9, n1));
    //float y = fragCoord.y/iResolution.y;
    //float fade = exp(-(y*y));
    //col *= fade;
    col = clamp(col, vec3(0.0), vec3(1.0));
    return col;
}

// Function 598
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

// Function 599
vec3 effect(vec2 uv, vec3 col)
{
    float grid = yVar * 10.+5.;
    float step_x = 0.0015625;
    float step_y = step_x * s.x / s.y;
	float offx = floor(uv.x  / (grid * step_x));
    float offy = floor(uv.y  / (grid * step_y));
    vec3 res = bg(vec2(offx * grid * step_x , offy * grid * step_y));
    vec2 prc = fract(uv / vec2(grid * step_x, grid * step_y));
    vec2 pw = pow(abs(prc - 0.5), vec2(2.0));
    float  rs = pow(0.45, 2.0);
    float gr = smoothstep(rs - 0.1, rs + 0.1, pw.x + pw.y);
    float y = (res.r + res.g + res.b) / 3.0;
    vec3 ra = res / y;
    float ls = 0.3;
    float lb = ceil(y / ls);
    float lf = ls * lb + 0.3;
    res = lf * res;
    col = mix(res, vec3(0.1, 0.1, 0.1), gr);
    return col;
}

// Function 600
float shadowTrace(in vec3 pos, in vec3 ray){
    
    vec3 norm = vec3(0.);
    vec2 i;
    float shadow = 0., t = INFINI;
    
    // i.y is the depth reach by the ray.
	// Depth is used to smooth the shadow.
    // From all encountered object, the deeper (biggest i.y) value is kept.

    i = boxImpact(pos, ray, boxO, boxD, norm);
	if(i.x < INFINI) {shadow = max(shadow,i.y), t = min(t,i.x);}
    
    i = coneImpact(pos, ray, conO, conH, conR, norm);
	if(i.x < INFINI) {shadow = max(shadow,i.y), t = min(t,i.x);}
	
    i = sphereImpact(pos,ray,spO,spR, norm);
	if(i.x < INFINI) {shadow = max(shadow,i.y), t = min(t,i.x);}
    
    i = cylinderImpact(pos.yz, ray.yz, cyO, cyR, norm.yz);
	if(i.x < INFINI) {shadow = max(shadow,i.y), t = min(t,i.x);}
    
    i = cylinderImpact(pos.xy, ray.xy, cyO, cyR, norm.xy);
	if(i.x < INFINI) {shadow = max(shadow,i.y), t = min(t,i.x);}
    
    
    #ifdef SPLIT
    if(gl_FragCoord.x < iResolution.x/2.)
    #endif
    
    shadow /= max(20.,t*SH*iResolution.x);	// expanding same way as anti-aliasing
    
    return min(1.,shadow);
}

// Function 601
float shadow(vec3 rayOrigin, vec3 rayDirection) {
    int stepCount = 64;
    float t = 0.03;
    float maximumDistance = 20.0;
    for (int i = 0; i < stepCount; i++) {
        if (t > maximumDistance) {
            break;
        }
        vec3 currentPosition = rayOrigin + rayDirection * t;
        float d = sdf(currentPosition).x;
        if (d < 0.001) {
            return 0.0;
        }
        t += d;
    }
    return 1.0;
}

// Function 602
float shadows(in vec3 ro, in vec3 rd, in float start, in float end, in float k){

    float shade = 1.0;
    const int shadIter = 24; 

    float dist = start;
    //float stepDist = end/float(shadIter);

    for (int i=0; i<shadIter; i++){
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.

        dist += clamp(h, 0.02, 0.2);
        
        // There's some accuracy loss involved, but early exits from accumulative distance function can help.
        if ((h)<0.001 || dist > end) break; 
    }
    
    return min(max(shade, 0.) + 0.0, 1.0); 
}

// Function 603
float softshadow(vec3 ro,vec3 rd) 
{
    float sh = 1.0;
    float t = 0.02;
    float h = 0.0;
    for(int i = 0; i < 22; i++)  
	{
        if(t > 20.) continue;
        h = map(ro + rd * t);
        sh = min(sh, 4.0 * h / t);
        t += h;
    }
    return sh;
}

// Function 604
float calcShadow( vec3 origin, vec3 lightDir, Light light)
{
	float dist;
	float result = 1.0;
	float lightDist = length(light.position-origin);
	
	vec3 pos = vec3(origin)+(lightDir*(EPSILON*15.0+BUMP_FACTOR));
	
	for(int i = 0; i < MAX_SHADOW_STEPS; i++)
	{
		dist = getDist(pos);
		if(dist < EPSILON)
		{
			return 0.0;
		}
		if(length(pos-origin) > lightDist || length(pos-origin) > MAX_DEPTH)
		{
			return result;
		}
		pos+=lightDir*dist;
		if( length(pos-origin) < lightDist )
		{
			result = min( result, light.penumbraFactor*dist / length(pos-origin) );
		}
	}
	return result;
}

// Function 605
bool intersectShadow (in Ray ray, in float dist) {
    Result ball1 = sphereIntersect (ray, vec3 (1.5, -1., 2.0), 1., 0);
    Result ball2 = sphereIntersect (ray, vec3 (1., -1.5, -1.0), .5, 3);
    Result ball3 = sphereIntersect (ray, vec3 (-1., -1.25, -2.0), .75, 6);
    Result ball4 = sphereIntersect (ray, vec3 (-.75, -1.25, .0), .75, 7);
    Result ball5 = sphereIntersect (ray, vec3 (1., -.5, -1.0), .5, 3);
    Result ball6 = sphereIntersect (ray, vec3 (1., .5, -1.0), .5, 3);
    Result ball7 = sphereIntersect (ray, vec3 (1., 1.5, -1.0), .5, 3);

    Result res = minResult (ball1, ball2);
    res = minResult (ball3, res);
    res = minResult (ball4, res);
    res = minResult (ball5, res);
    res = minResult (ball6, res);
    res = minResult (ball7, res);

    if (res.dist > .0001 && res.dist < dist)
        return true;

    return false;
}

// Function 606
vec4 wavesShadowPalette(in float x)
{
    if(x<4.)
    {
        return ARR4(x,  D_BLUE,
			   			WHITE,
			   			L_BLUE,
			   			WHITE);
    }
    else return ARR2(x-4., D_BLUE, L_BLUE);
}

// Function 607
float shadow (vec3 ro, vec3 rd){
    float k = 8.0;
    float res = 1.0;
    
    for(float t = 1.0; t<16.0; t += 0.2){
     	float h = map(ro + rd*t);
        if (h < 0.001)
            return 0.0;
        res = min (res, k*h/t);
    }
    
    return res;
}

// Function 608
float effectIris( in vec2 uv, in float frac ) {
    uv *= 1.5;
    float r = length(uv);
	if (r > 1.0) {
		return 0.0;
	} else {
		vec3 l = normalize(vec3(1, 1, 2));
		vec3 p = vec3(uv, sqrt(1.0 - r*r));
        float angle = cos(iTime*0.02914)*15.115;
        mat2 rotxy = rot(angle);
        mat2 rotxz = rot(sin(iTime*0.447)*0.117);
 		l.xy *= rotxy;
        p.xy *= rotxy;
        l.xz *= rotxz;
        p.xz *= rotxz;
        float d = dot(l, p);
        float theta = atan(p.x, p.y)-angle;
		return (d*0.5+d*d*0.3+0.3)*irisColor(p, frac, theta);
	}
}

// Function 609
vec4 Blur9(vec2 p) {
    
    vec2 px = 1./iResolution.yy;
    // Four spots aren't required in this case, but are when the above isn't aspect correct.
    vec4 e = vec4(px, 0, -px.x);
 
    // Averaging nine pixels.
    return (tx(p - e.xy) +  tx(p - e.zy ) + tx(p - e.wy) + // First row.
			tx(p - e.xz) + tx(p) + tx(p + e.xz) + 		     // Seond row.
			tx(p + e.wy) + tx(p + e.zy) +  tx(p + e.xy))/9.;  // Third row
    
 
}

// Function 610
float ShadowMarch(in vec3 origin, in vec3 rayDirection)
{
	float result = 1.0;
    float t = 0.01;
    for (int i = 0; i < 64; ++i)
    {
        float hit = SdScene(origin + rayDirection * t).x;
        if (hit < 0.001)
            return 0.0;
        result = min(result, 5.0 * hit / t);
        t += hit;
        if (t >= 1.5)
            break;
    }
    
    return clamp(result, 0.0, 1.0);
}

// Function 611
column barrelEffect(float x, float w/*idth*/, float r/*adius*/)
{
    if (x >= w && x <= W - w)
        return column(0., H);
    x -= (W / 2.);
    float x0 = w - (W / 2.);
    float y = sqrt(r * r - x0 * x0) - sqrt(r * r - x * x);
    return column(y, H - y - y);
}

// Function 612
float Scene_TraceShadow( const in vec3 vRayOrigin, const in vec3 vRayDir, const in float fMinDist, const in float fLightDist )
{
    //return 1.0;
    //return ( Scene_Trace( vRayOrigin, vRayDir, 0.1, fLightDist ).fDist < fLightDist ? 0.0 : 1.0;
    
	float res = 1.0;
    float t = fMinDist;
    for( int i=0; i<16; i++ )
    {
		float h = Scene_GetDistance( vRayOrigin + vRayDir * t ).fDist;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.0001 || t>fLightDist ) break;
    }
    return clamp( res, 0.0, 1.0 );    
}

// Function 613
float getFireShadow(vec3 posWS)
{
    float fireLife = max(0.0, s_globalFireLife);
    vec3 fireLightPosWS = vec3(0.0, 0.05 + fireLife*0.35, 0.0);
    
    float lightAnim = s_time * 15.0;
    float lightAnimAmpl = 0.02;
    fireLightPosWS.x += lightAnimAmpl * sin(lightAnim);
    fireLightPosWS.z += lightAnimAmpl * cos(lightAnim);
    
    float fireLightRadius = 0.7 - fireLife * 0.25;
    
    vec3 posToFireWS = fireLightPosWS - posWS;
    float posToFireLength = length(fireLightPosWS - posWS);
    vec3 posToFireDirWS = posToFireWS / max(0.0001, posToFireLength);
    
    float distToLight = posToFireLength-fireLightRadius;
    
    if(distToLight < 0.0)
    {
        return 1.0;
    }
    
    float lightTanAngularRadius = fireLightRadius / max(0.0001, posToFireLength);
    
    return marchShadow(posWS, posToFireDirWS, 0.001, 
                       max(0.0, posToFireLength-fireLightRadius), lightTanAngularRadius);
}

// Function 614
float calcShadow(vec3 p, vec3 ld) {
	// Thanks iq.
	float s = 1., t = .1;
	for (float i = 0.; i < 40.; i++)
	{
		float h = map(p + ld * t);
		s = min(s, 15. * h / t);
		t += h;
		if (s < .001) break;
	}

	return clamp(s, 0., 1.);
}

// Function 615
float softShadow(vec3 ro, vec3 lp, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 20; 
    
    vec3 rd = (lp-ro); // Unnormalized direction ray.

    float shade = 1.0;
    float dist = 0.05;    
    float end = max(length(rd), 0.001);
    //float stepDist = end/float(maxIterationsShad);
    
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        //dist += min( h, stepDist ); // So many options here: dist += clamp( h, 0.0005, 0.2 ), etc.
        dist += clamp(h, 0.01, 0.2);
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0.001 || dist > end) break; 
    }

    // I've added 0.5 to the final shade value, which lightens the shadow a bit. It's a preference thing.
    return min(max(shade, 0.) + 0.2, 1.0); 
}

// Function 616
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

// Function 617
float shadow(in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    
    float t = .0;
    for( int i = 0; i < 12; i++ )
    {
		float h = map(ro + rd*t, 2);
        res = min( res, 3.*h/t );
        t += h*1.5+.2;
    }
    return clamp( res, 0., 1.0 );
}

// Function 618
bool shadowRaySceneIntersection( in Ray ray, in float distMin ) {
    
}

// Function 619
bool in_earth_shadow( vec3 p )
{
	return ( dot( p, sun_direction ) < 0.0 )
		   && ( lensqr( p - project_on_line1( p, earth_center, sun_direction ) ) < earth_radius * earth_radius );
}

// Function 620
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

// Function 621
float softShadow(vec3 ro, vec3 lp, float k, float t){

    // More would be nicer. More is always nicer, but not really affordable.
    const int maxIterationsShad = 48; 
    
    vec3 rd = lp-ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = .0025*(t*.125 + 1.);  // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        dist += clamp(h, .02, .5); 
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0.0 || dist > end) break; 
    }

    // I've added a constant to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me. Sometimes, I'll add AO also just for kicks. :)
    return min(max(shade, 0.) + .15, 1.); 
}

// Function 622
float getShadow(vec3 light, vec3 origin)
{
    _shadowMarch = true;

    float t = 0., s = 1.0;
    float maxt = length(light - origin)-.1;
    vec3 d = normalize(light - origin);

    for(int i = 0; i < S; i++)
    {
        float d = scene(origin + d * t);
        if (t > maxt || t > D) { break; }
        t += d; s = min(s,d/t*K);
    }

    return s;
}

// Function 623
float softShadow(vec3 ro, vec3 rd, float tmin, float tmax, float k) {
    float res = 1.0;
    float t = tmin;
    for (int i = 0; i < 20; i++) {
        float h = map(ro + rd * t).x;
        res = min(res, k * h / t);
        t += clamp(h, 0.001, 0.1);
        if (h < 0.0001 || t > tmax)
            break;
    }
    return clamp(res, 0., 1.0);
}

// Function 624
float softShadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
        float h = map( ro + rd*t ).distance;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 625
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, float softness )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<32; i++ )
    {
		float h = dstScene( ro + rd*t ).dst;
        res = min( res, softness*h/t );
        t += clamp( h, 0.02, 0.40 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 626
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 40; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, 20. * h / d);
    d += 0.1;
    if (h < 0.001) break;
  }
  return clamp (sh, 0., 1.);
}

// Function 627
vec4 blur(vec2 coord){
    vec2 start = coord + vec2(0, 3) * factor;
    
    for (int i = 0; i < 8; i++){
        vec2 uv = start + float(i) * vec2(0,-1) * factor;
        addSample(uv/iResolution.xy);
    	}
    
    return accum/accumW;
	}

// Function 628
float softShadow(in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.001;
	for( int i=0; i<80; i++ )
	{
	    vec3  p = ro + t*rd;
        float h = p.y - terrainM( p.xz );
		res = min( res, 16.0*h/t );
		t += h;
		if( res<0.001 ||p.y>(SC*200.0) ) break;
	}
	return clamp( res, 0.0, 1.0 );
}

// Function 629
float getShadow(vec3 source, vec3 target)
{
	IN_SHADOW_MARCH = true;
	
	float t = 0.01;
	float s = 1.0;
	float r = length(target-source);
	float d;
	
	vec3 dir = normalize(target-source);
	
	for(int i = 0; i < S; i++)
	{
		d = scene(source+dir*t);
		if (d<P) return 0.0;
		if(t>r){break;}
		s = min(s,K*d/clamp(t,0.,1.));
		t += d;
	}
	
	return s;
}

// Function 630
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
    	float h = map(ro + rd*t, false);
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.0001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 631
float softshadow(vec3 pos, vec3 ldir, vec3 playerPos) {
#if USE_SHADOWS
    float res = 1.0;
    float t = 0.01;
    for(int i = 0; i < 25; i++) {
        float h = map(pos - ldir*t, junkMatID, playerPos, true);
        res = min(res, 7.0*h/t);
        t += clamp(h, 0.007, 5.0);
        if (h < EPS) break;
    }
    return clamp01(res);
#else
    return 1.0;
#endif
}

// Function 632
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 60; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, 20. * h / d);
    d += 0.2;
    if (h < 0.001) break;
  }
  return clamp (0.5 + 0.5 * sh, 0., 1.);
}

// Function 633
float ObjSShadow (vec3 ro, vec3 rd, float dLight)
{
  float sh = 1.;
  float d = 0.15;
  for (int i = 0; i < 30; i++) {
    float h = ObjDf (ro + rd * d);
    sh = min (sh, 20. * h / d);
    d += max (0.15, 0.01 * d);
    if (h < 0.01 || d > dLight) break;
  }
  return clamp (sh, 0., 1.);
}

// Function 634
float softshadow(in vec3 ro, in vec3 rd, in float maxt)
{
    float res = 1.0;
    float t = 0.001;
	for( int i=0; i<80; i++ )
	{
	    vec3  p = ro + t*rd;
        float h = dstScene(p).x;
		res = min( res, 16.0*h/t );
		t += h;
		if( res<0.001 ||p.y > maxt) break;
	}
	return clamp( res, 0.0, 1.0 );
}

// Function 635
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for(int i=0; i<50; i++)
    {
    	float h = map(ro + rd*t);
        res = min(res, 10.0*h/t + 0.02*float(i));
        t += 0.8*clamp(h, 0.01, 0.35);
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 636
vec2 CalculateShadowMapUV(in mat4 shadowMapMatrix, in vec3 position, in float aspectRatio)
{
    vec3 lightPosition = vec3(shadowMapMatrix[3][0], shadowMapMatrix[3][1], shadowMapMatrix[3][2]); 
    
    vec3 lightWorldDirection = normalize(position.xyz - lightPosition);
    vec3 shadowMapCameraRayDirection = (vec4(lightWorldDirection.xyz, 1.0) * shadowMapMatrix).xyz;
    shadowMapCameraRayDirection /= shadowMapCameraRayDirection.z;
    
    vec2 textureCoords = shadowMapCameraRayDirection.xy / vec2(aspectRatio, 1.0);
    textureCoords = textureCoords * 0.5 + 0.5;
    
    return textureCoords;
}

// Function 637
float softShadowTrace(in vec3 rp, in vec3 rd, in float maxDist, in float penumbraSize, in float penumbraIntensity) {
    vec3 p = rp;
    float sh = 0.;
    float d,s = 0.;
    for (int i = 0; i < SHADOW_ITERATIONS; i++) {
        d = df(rp+rd*s);
        sh += max(0., penumbraSize-d)*float(s>penumbraSize*4.);
        s += d;
        if (d < EPSILON || s > maxDist) break;
    }
    
    if (d < EPSILON) return 0.;
    
    return max(0.,1.-sh/penumbraIntensity);
}

// Function 638
float shadows(in vec3 ro, in vec3 rd, in float start, in float end, in float k){

    float shade = 1.0;
    const int shadIter = 14; 

    float dist = start;
    //float stepDist = end/float(shadIter);

    for (int i=0; i<shadIter; i++){
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.

        dist += clamp(h, 0.32, 0.2);
        
        // There's some accuracy loss involved, but early exits from accumulative distance function can help.
        if ((h)<0.001 || dist > end) break; 
    }
    
    return min(max(shade, 0.) + 0.0, 1.0); 
}

// Function 639
float shadowtrace(vec3 ro, vec3 rd) {
    int i;
    float t = shadoweps;
    float dist = map(ro+t*rd).x;
    float fac = 1.0;
    for (i=0; i<shadowiters; i++) {
        t += shadowstep;
        dist = map(ro + t*rd).x;
        fac = min(fac, dist * sharpness / t);
    }
    return fac > 0. ? mix(0.5, 1., fac) : mix(0.5, 0., -fac);
}

// Function 640
float shadow(vec3 ro, vec3 rd, float k)
{
	float res = 1.0;
    float t = EPSILON;
    for( int i=0; i<1000; i++)
    {
        float h = distanceField(ro + rd*t);
        if( h<EPSILON )
            return AMBIENT;
        res = min( res, k*h/t );
        t += h;
        if(t>=SHADOWDEPTH) break;
    }
    return res;
}

// Function 641
float calcSoftshadow( in vec3 ro, in vec3 rd, float tmin, float tmax, const float k )
{
	float res = 1.0;
    float t = tmin;
    for( int i=0; i<100; i++ )
    {
		float h = map( ro + rd*t ).x;
        res = min( res, k*h/t );
        t += clamp( h, 0.02, 0.20 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 642
float softShadowSphere( in vec3 ro, in vec3 rd, in vec4 sph )
{
    vec3 oc = sph.xyz - ro;
    float b = dot( oc, rd );
	
    float res = 1.0;
    if( b>0.0 )
    {
        float h = dot(oc,oc) - b*b - sph.w*sph.w;
        res = clamp( 2.0 * h / b, 0.0, 1.0 );
    }
    return res;
}

// Function 643
float shadow( vec3 light, vec3 lv, float len ) {
	float depth = ray_marching( light, lv, 0.0, len );
	
	return step( len - depth, 0.01 );
}

// Function 644
float SoftShadowTower( in vec3 origin, in vec3 direction, float res)
{
  float t = 0.0, h;
  vec3 rayPos = vec3(origin+direction*t);

  for ( int i=0; i<NO_UNROLL(11); i++ )
  {

    h = sdConeSection(rayPos-vec3(-143, 0., 292)-vec3(0., 12., 0.), 10.45, 2.40, 1.40);

    res = min( res, 6.5*h/t );
    t += clamp( h, 0.4, 1.5);
    if ( h<0.005 ) break;
    rayPos = vec3(origin+direction*t);
  }
  return clamp( res, 0.0, 1.0 );
}

// Function 645
float Shadow( in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    float t = 0.05;
	float h;
	
    for (int i = 0; i < 6; i++)
	{
		h = Map( ro + rd*t );
		res = min(7.0*h / t, res);
		t += h+.01;
	}
    return max(res, 0.0);
}

// Function 646
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

// Function 647
float SoftShadow( in vec3 landPoint, in vec3 lightVector, float mint, float maxt, float iterations ){
    float penumbraFactor=1.0;vec3 sphereNormal;float t=mint;
    for(int s=0;s<20;++s){if(t > maxt) break;
        float nextDist = min(
            BuildingsDistance(landPoint + lightVector * t )
            , RedDistance(landPoint + lightVector * t ));
        if( nextDist < 0.001 ){return 0.0;}
        penumbraFactor = min( penumbraFactor, iterations * nextDist / t );
        t += nextDist;}return penumbraFactor;}

// Function 648
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.07 * szFac;
  for (int j = 0; j < 24; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += 0.07 * szFac;
    if (sh < 0.05) break;
  }
  return 0.3 + 0.7 * sh;
}

// Function 649
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not always affordable. :)
    const int maxIterationsShad = 24; 
    
    ro += n*.0015; // Coincides with the hit condition in the "trace" function.  
    vec3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float t = 0.; 
    float end = max(length(rd), .0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, 
    // the lowest number to give a decent shadow is the best one to choose. 
    for (int i = min(iFrame, 0); i<maxIterationsShad; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += clamp(d, .01, stepDist), etc.
        t += clamp(d, .01, .25); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}

// Function 650
float shadow( in vec3 ro, in vec3 rd )
{
    const float k = 2.0;
    
    const int maxSteps = 10;
    float t = 0.0;
    float res = 1.0;
    
    for(int i = 0; i < maxSteps; ++i) {
        
        float d = doModel(ro + rd*t).x;
            
        if(d < INTERSECTION_PRECISION) {
            
            return 0.0;
        }
        
        res = min( res, k*d/t );
        t += d;
    }
    
    return res;
}

// Function 651
void drawMineCraftNotBlur(Ray ray, inout TraceResult cur_ctxt)
{
    TraceResult tr_res;
    float mineT;
    tr_res = TraceMineCraft(ray.pos, ray.dir);

    mineT = sqrt(dot(tr_res.p - ray.pos, tr_res.p - ray.pos)); 
    if (mineT < cur_ctxt.t && tr_res.hit == true)
    {
        cur_ctxt.color = vec3(compute_minecraft_light(tr_res, 
                                    tr_res.sphereIntersect, ray.dir));
        cur_ctxt.n = tr_res.n;
        
        cur_ctxt.t = mineT;
        //cur_ctxt.alpha = max(0.75, 
        //                    sqrt(sqrt(dot(tr_res.p, tr_res.p))));
        cur_ctxt.alpha = GLOBAL_ALPHA;
        cur_ctxt.materialType = EMISSION;
    }
}

// Function 652
float shadows(in vec3 ro, in vec3 rd, in float start, in float end, in float k){

    float shade = 1.0;
    const int shadIter = 14; 

    float dist = start;
    //float stepDist = end/float(shadIter);

    for (int i=0; i<shadIter; i++){
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.

        dist += clamp(h, 2.32, 1.2);
        
        // There's some accuracy loss involved, but early exits from accumulative distance function can help.
        if ((h)<0.001 || dist > end) break; 
    }
    
    return min(max(shade, 0.) + 0.1, 1.0); 
}

// Function 653
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=ZERO; i<16; i++ )
    {
		float h = doModel( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 654
float shadow( in vec3 ro, in vec3 rd, float dis)
{
	float res = 1.0;
    float t = .1;
	float h;
	
    for (int i = 0; i < 15; i++)
	{
        vec3 p =  ro + rd*t;

		h = map(p,dis).x;
		res = min(3.*h / t, res);
		t += h;
	}
    res += t*t*.08; // Dim over distance
    return clamp(res, .6, 1.0);
}

// Function 655
float effectPacman( in vec2 uv, in float frac ) {
    float value = 0.0;
    uv *= 4.0;
    vec2 pacmanCenter = vec2(4.7-frac*12.0, 0.0);
    vec2 delta = uv-pacmanCenter;
    float theta = abs(atan(delta.y, -delta.x));
    float mouth = step(max(0.0, sin(iTime*10.0)*0.4+0.35), theta);
    value += max(0.0, 20.0-distance(uv, pacmanCenter)*20.0)*mouth;
    if (uv.x > pacmanCenter.x+0.5) return value;
    vec2 center = vec2(floor(uv.x)+0.5, 0.0);
    value += max(0.0, 5.0-distance(uv, center)*20.0);
    return value;
}

// Function 656
float softshadow( in vec3 ro, in vec3 rd, in float k )
{
    float res = 1.0;
    float t = 0.0;
    for( int i=0; i<64; i++ )
    {
        vec4 kk;
        float h = map(ro + rd*t, kk);
        res = min( res, k*h/t );
        if( res<0.001 ) break;
        t += clamp( h, 0.01, 0.2 );
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 657
float softshadow(in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k) {
    float res = 1.0, h, t = mint+.1*hash(ro+rd);
    for( int i=0; i<48; i++ ) {
      //  if (t < maxt) {
            h = DE( ro + rd*t ).x;
            res = min( res, k*h/t );
            t += .1;
      //  }
    }
    return clamp(res, 0., 1.);
}

// Function 658
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = 0; j < 16; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += 0.07;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 659
float calcShadow(vec3 ro, vec3 rd) {
    float res = 1.0;
    
    float t = 0.1;
    
    for (int i = 0; i < 16; i++) {
        vec3 pos = ro + t * rd;
        float h = map(pos);
        res = min(res, 10.0 * max(h, 0.0) / t);
        
        if (res < 0.1) break;
        
        t += h;
    }
    
    return max(res, 0.1);
}

// Function 660
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 50; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += min (0.2, 3. * h);
    if (sh < 0.001) break;
  }
  return 0.4 + 0.6 * sh;
}

// Function 661
bool intersectShadowMesh( in vec3 ro, in vec3 rd )
{
    ro.z += 0.5;
    if( !boxIntersect( ro, rd, vec3(0.0), vec3(0.18,0.3,0.52) ) )
        return false;

    bool res = false;
    for( int i=ZERO; i<numFaces; i++ )
    {
		// get the triangle
        vec3 v0 = getVertex(i,0);
        vec3 v1 = getVertex(i,1);
        vec3 v2 = getVertex(i,2);

        vec3 h = triIntersect( ro, rd, v0, v1, v2 );
        if( h.x>0.0 )
        {
            res = true;
            break;
        }
    }
    
    return res;
}

// Function 662
float orbShadow(float rad, vec3 coord)
{
	return 1.0 - smoothstep(0.4, 1.1, distance(coord.xy, frag_coord) / rad) *
		mix(1.0,0.99,orb(rad,coord));
}

// Function 663
int InShadow(vec3 p, vec3 L, out float t)
{
    int shadowHit = NO_HIT;
    Ray sr;
    sr.Direction = L;
    sr.Orgin = p + vec3(0.05);     
    t =  Trace(sr,shadowHit);
    
    return shadowHit;
}

// Function 664
float shadow( in vec3 ro, in vec3 rd, float mint, float maxt )
{
    for( float t=mint; t<maxt; )
    {
        float h = map(ro + rd*t);
        if( h<EPSILON )
            return 0.0;
        t += h;
    }
    return 1.0;
}

// Function 665
vec4 BlurB(vec2 uv, int level)
{
    return BlurB(uv, level, iChannel1, iChannel3);
}

// Function 666
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	rgb = pow(rgb, vec3(0.45));
	
	// Then...
	#define CONTRAST 1.1
	#define SATURATION 1.3
	#define BRIGHTNESS 1.3
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);
	// Vignette...
	rgb *= .4+0.5*pow(40.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.2 );	
	return rgb;
}

// Function 667
float softshadow(in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k) 
{
    float sh = 1.0;
    float t = mint;
    float h = 0.0;
    for(int i = 0; i < 19; i++)  //23 gut!
	{
        if(t > maxt) continue;
		orbitTrap = vec4(10.0);
        h = map(ro + rd * t);
        sh = min(sh, k * h / t);
        t += h;
    }
    return sh;
}

// Function 668
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    // bounding volume
    float tp = (0.8-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

	float res = 1.0;
    float t = mint;
    for( int i=ZERO; i<16; i++ )
    {
		float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 669
float castShadowRay(vec3 pos, vec3 dir, float time, float maxDist)
{
    float totalDist = 0.0;
    float distToSurface = distModel(pos + dir * 0.1, time);
    float maxShadow = 0.0;
    
    for(int i = 0; i < 50 ; ++i)
    {
        totalDist += distToSurface;
        vec3 currentPos = pos + dir * totalDist;
        distToSurface = distModel(currentPos, time);
        float shadowSmooth = mix(0.1, 0.7, smoothstep(0.1, 10.0, totalDist));
        float shadow = smoothstep(shadowSmooth, 0.0, distToSurface);
        shadow *= smoothstep(10.0, 8.0, totalDist);
        maxShadow = max(maxShadow, shadow);
        
        if(distToSurface < 0.00 || totalDist > maxDist)
        {
            return 1.0 - maxShadow;
        }
    }
    
    return  1.0 - maxShadow;
}

// Function 670
float calcSoftshadow(in vec3 ro, in vec3 rd)
{
    float res = 1.0;
    float tmax = 12.0;  
    
    float t = 0.02;
    for( int i=0; i<40; i++ )
    {
		float h = map(ro + rd*t, false, true).x;
        res = min( res, 24.0*h/t );
        t += clamp( h, 0.0, 0.80 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 671
void effectFlow(inout vec4 fragColor, vec2 pos)
{
    float rnd = randS(vec2(float(iFrame)/Res.x,0.5/Res1.y)).x*1.;
    
    vec2 b = vec2(cos(ang*rnd),sin(ang*rnd));
    vec2 v=vec2(0);
    float bbMax=0.7*Res.y; bbMax*=bbMax;
    for(int l=0;l<3;l++)
    {
        if ( dot(b,b) > bbMax ) break;
        vec2 p = b;
        for(int i=0;i<RotNum;i++)
        {
#ifdef SUPPORT_EVEN_ROTNUM
            v+=p.yx*getRot(pos+p,-mh*b);
#else
            // this is faster but works only for odd RotNum
            v+=p.yx*getRot(pos+p,b);
#endif
            p = m*p;
        }
        b*=2.0;
    }
    
    fragColor=getCol(pos+v*vec2(-1,1)*4.);
}

// Function 672
float eliSoftShadow( in vec3 ro, in vec3 rd, in Ellipsoid sph, in float k )
{
    vec3 oc = ro - sph.cen;
    
    vec3 ocn = oc / sph.rad;
    vec3 rdn = rd / sph.rad;
    
    float a = dot( rdn, rdn );
	float b = dot( ocn, rdn );
	float c = dot( ocn, ocn );
	float h = b*b - a*(c-1.0);


    float t = (-b - sqrt( max(h,0.0) ))/a;

    return (h>0.0) ? step(t,0.0) : smoothstep(0.0, 1.0, -k*h/max(t,0.0) );
}

// Function 673
float softshadow( in Ray ray, in float mint, in float maxt, in float k )
{
	float t = mint;
	float res = 1.0;
    for ( int i = 0; i < 128; ++i )
    {
        float h = scene( ray.org + ray.dir * t );
        if ( h < 0.001 )
            return 0.0;
		
		res = min( res, k * h / t );
        t += h;
		
		if ( t > maxt )
			break;
    }
    return res;
}

// Function 674
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

// Function 675
float beamShadow(float x)
{
	x = clamp(x + 0.1, 0.0, 0.5);
	return smoothstep(0.7, 1.0, pow(1.0 - 2.0 * abs(x - 0.5), 0.1));
}

// Function 676
float getShadow(vec3 source, vec3 target)
{
	float r = length(target-source);
	float t = _displayVoxel == true ? 0.05 : 0.01;
	float s = 1.0;
	float d;
	
	vec3 dir = normalize(target-source);
	
	for(int i = 0; i < S; i++)
	{
		d = scene(source+dir*t);
		
		if (d < P) { return 0.0; }
		if (t > r) { break; }
		
		s = min(s,K*d/t);
		t += d/R;
	}
	
	return s;
}

// Function 677
float castShadow( in vec3 ro, vec3 rd, float time )
{
    float res = 1.0;
    float t = 0.00;
    for( int i=0; i< 100; i++ )
    {
        vec3 pos = ro + t*rd;
        float h = map( pos, time ).x;
        res = min( res, 16.0*h/t );
        if ( res<0.001 ) break;
        t += h;
        if( t > 10.0 ) break;
    }

    return clamp(res,0.0,1.0);
}

// Function 678
float shadow(vec2 p, vec2 pos, float radius)
{
	vec2 dir = normalize(pos - p);
	float dl = length(p - pos);
	
	// fraction of light visible, starts at one radius (second half added in the end);
	float lf = radius * dl;
	
	// distance traveled
	float dt = 0.01;

	for (int i = 0; i < 64; ++i)
	{				
		// distance to scene at current position
		float sd = sceneDistance(p + dir * dt);

        // early out when this ray is guaranteed to be full shadow
        if (sd < -radius) 
          return 0.0;
        
		// width of cone-overlap at light
		// 0 in center, so 50% overlap: add one radius outside of loop to get total coverage
		// should be '(sd / dt) * dl', but '*dl' outside of loop
		lf = min(lf, sd / dt);
		
		// move ahead
		dt += max(1.0, abs(sd));
		if (dt > dl) break;
	}
	// multiply by dl to get the real projected overlap (moved out of loop)
	// add one radius, before between -radius and + radius
	// normalize to 1 ( / 2*radius)
	lf = clamp((lf*dl + radius) / (2.0 * radius), 0.0, 1.0);
	lf = smoothstep(0.0, 1.0, lf);
	return lf;
}

// Function 679
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

// Function 680
float shadow_sdf(vec3 pose, bool inside)
{
    
    vec3 pos = pose;
    vec3 rot = pos;
    
    rot.xy -= vec2(0.36,-0.4);

    rot.xy = n_rotate(rot.xy,0.245);

    rot.xy += vec2(0.36,-0.4);
    
    float door_box = box(rot    -vec3(0.18,-0.425,0.82),  vec3(0.18,0.04,0.67));
    
    if(door_box < 0.001)
        pos = rot;
        
        
    
    //use symetries:
        
        //mirror at axes
        vec3 pos_x  = vec3(abs(pos.x),abs(pos.y),pos.z);
        
        //mirror at diagonals
        vec3 pos_xx = pos_x.x < pos_x.y  ?  pos_x.xyz  :  pos_x.yxz;
        
    
    
    
     float d = 1000.;
    
    
    if(pos.z > 1.55)
    {
        float roof = box(pos    -vec3( 0.00, 0.00,1.74),  vec3(0.40,0.40,0.06));//roof2
        roof = max(roof, pln(pos_xx -vec3( 0.00, 0.00,1.80),  vec3(0.00,0.148,0.989)));//roof2 slope
        d = min(d,roof);
        
        d = min(d, box(pos    -vec3( 0.00, 0.00,1.79),  vec3(0.09,0.09,0.03)));//light base
        float tar_rad = length(pos.xy);
        d = min(d, max(tar_rad - 0.05, abs(pos.z - 1.84) - 0.05));//light pole
        d = min(d, max(tar_rad - 0.06, abs(pos.z - 1.91) - 0.05));//light
        d = min(d, max(dot(vec2(0.447,0.894),vec2(tar_rad,pos.z-2.)),1.96-pos.z));//light roof
    }
    else
    {
        if(door_box > 0.001)
            d = min(d,door_box+0.001);
    
        if(pos.z > 0.73)
        {
            d = min(d, box(pos_xx -vec3(0.20,.425,1.30),  vec3(0.15,0.01,.005)));//window bar
            d = min(d, box(vec3(abs(pos_xx.x-.19),pos_xx.yz)-vec3(0.045,0.425,1.30),  vec3(.005,0.01,0.15)));//window vertical bar
            
            
            d = min(d, box(pos_xx -vec3(0.20,0.425,1.30),  vec3(0.15,.005,0.15)) + 0.001+ 0.001*sin(pos_xx.x*400.));//window
            
            if(pos.y < -0.4)
            {
                d = min(d, tor((max(vec3(0.0),abs(pos    + vec3( 0.30, 0.44,-0.98))+vec3(-0.003,0,-0.02))).yxz,  vec2(0.01,0.002)));//phone handle
                
                d = min(d, box(vec3(abs(pos.x+.19)-0.11,pos.yz)    -vec3(0.,-0.435,0.97),  vec3(0.01,0.005,0.13)));//phone sign border vertical
                d = min(d, box(vec3(pos.xy,abs(pos.z-.97)-0.12)    -vec3(-0.19,-0.435,0.00),  vec3(0.10,0.005,0.01)));//phone sign border horizontal
                
                d = min(d, cheap_cyl((rot    + vec3(-0.04,.441,-0.98)).xzy,.017,0.005));//lock
            }
            
            
        }
        else
        {
            float base = box(pos    -vec3( 0.00,0.00,0.06),  vec3(0.50,0.50,0.06));//base
            base = max(base, pln(pos_xx -vec3( 0.00,0.47,0.12),  vec3(0.00,0.514,0.857)));//base chamfer (slope)
            d = min(d,base);
        }
        
        d = min(d, box(vec3(pos_xx.xy,(pos_xx.z-0.15) - round((pos_xx.z-0.15)*3.05)/3.05) -vec3(0.19,0.425,0.0),  vec3(0.13,0.02,0.03)));//horizontal bars
        
        
        d = min(d, box(pos_xx -vec3(0.20,.425,0.65),  vec3(0.15,0.01,0.50)));//panels
        
        d = min(d, box(vec3(abs(pos_xx.x-0.19)+0.19,pos_xx.yz) -vec3(0.35,.425,0.82),  vec3(0.03,0.02,0.70)));//door vertical bar
        
        d = min(d, box(pos_xx -vec3(0.00,0.44,0.82),  vec3(0.02,0.01,0.70)));//center vertical bar
        
        d = min(d, box(pos_xx -vec3(.365,0.45,0.82 ),  vec3(.005,.005,0.70)));//border vertical bar
        d = min(d, box(pos_xx -vec3(0.00,0.45,1.485),  vec3(0.36,.005,.005)));//border horizontal bar
    }
    
    d = min(d, box(pos_xx -vec3( 0.00, 0.45, 1.55),  vec3(0.45,.05,0.06)));//sign
    d = max(d,-box(pos_xx -vec3( 0.00, 0.52, 1.55),  vec3(0.325,0.03,.04)));//sign inside
    
    d = min(d, box(pos_x  -vec3( .42, .42, 0.90),  vec3(0.05,0.05,0.78)));//corner pole
    
    d = min(d, box(pos    -vec3( 0.00, 0.00, 1.655),  vec3(0.425,0.425,0.055)));//roof1
    
    if(inside)
        d = max(d,-box(pos    -vec3(0.,0.45,0.81),  vec3(0.36,.2,0.68)));//open back
    
    d = max(d,-box(pose    -vec3(0.18,-0.425,0.82),  vec3(0.18,0.04,0.67)));//open door
    
    if(door_box < 0.001)
         d = max(d,door_box+0.001);
      
    
    
     
    return (d);
}

// Function 681
float softShadow(vec3 ro, vec3 lp, float k)
{
    const int maxIterationsShad = 32; 
    
    vec3 rd = (lp-ro); 

    float shade = 1.;
    float dist = .005;    
    float end = max(length(rd), 0.001);
    float stepDist = end/float(maxIterationsShad);
    
    rd /= end;

    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
       
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); 
       
        dist += clamp(h, .02, .2);
        
       
        if (h<0.0 || dist > end) break; 
       
    }

   
    return min(max(shade, 0.) + 0.03, 1.0); 
}

// Function 682
void draw_shadow_box(inout vec4 fragColor, vec2 fragCoord, vec4 box)
{
    draw_shadow_box(fragColor, fragCoord, box, DEFAULT_SHADOW_BOX_BORDER);
}

// Function 683
float softShadow(in vec3 ro, in vec3 rd, float mint, float k) {
    float res = 1.0;
    float t = mint;
    for(int i = 0; i < 32; i++) {
    	float h = map(ro + rd * t);
        if (h < 0.001) { return 0.0; }
        res = min(res, k*h/t);
       	t += h;
    }
    return res;
}

// Function 684
float compute_shadows(vec3 origin, vec3 n_light_direction, vec3 light_position)
{
    // TODO: Stop marching once light has been passed.
    // TODO: Use a point light or something!
    
    origin += n_light_direction * 0.005;	// Offset to avoid self-intersection
    
    vec3 point_of_intersection;
    if(find_intersection(origin, n_light_direction, 16, 0.05, 0, point_of_intersection))
    {
        return(0.0);
    }
    
    return(1.0);
}

// Function 685
float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
{
    float res = 1.0;
    float ph = 1e20;
    for( float t=mint; t<maxt; )
    {
        float h = sdf(ro + rd*t, false);
        if( h<0.001 )
            return 0.0;
        float y = h*h/(2.5*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.001,t-y) );
        ph = h;
        t += h;
    }
    return res;
}

// Function 686
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

// Function 687
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

// Function 688
float castShadowRay( in vec3 ro, in vec3 rd )
{
	vec2 pos = floor(ro.xz);
	vec2 ri = 1.0/rd.xz;
	vec2 rs = sign(rd.xz);
	vec2 ris = ri*rs;
	vec2 dis = (pos-ro.xz+ 9.5 + rs*0.5) * ri;
	float t = -1.0;
	float res = 10.0;
	
    // first step we check noching	
	vec2 mm = step( dis.xy, dis.yx ); 
	dis += mm * ris;
    pos += mm * rs;
	
    // traverse regular grid (2D)	
	for( int i=0; i<6; i++ ) 
	{
		float ma = map(pos);
		
        // test capped cylinder		
		vec3  ce = vec3( pos.x+0.5, 0.0, pos.y+0.5 );
		vec3  rc = ro - ce;
		float a = dot( rd.xz, rd.xz );
		float b = dot( rc.xz, rd.xz );
		float c = dot( rc.xz, rc.xz ) - 0.009;
		float h = b*b - a*c;
		if( h>=0.0 )
		{
			float t = (-b - sqrt( h ))/a;
			if( (ro.y+t*rd.y)<ma )
			{
				res = 0.0;
    			break; 
			}
		}
		mm = step( dis.xy, dis.yx ); 
		dis += mm * ris;
        pos += mm * rs;
	}

	return res;
}

// Function 689
float GrndSShadow (vec3 p, vec3 vs)
{
  vec3 q;
  float sh, d;
  sh = 1.;
  d = 0.4;
  for (int j = 0; j <= 25; j ++) {
    q = p + vs * d; 
    sh = min (sh, smoothstep (0., 0.02 * d, q.y - GrndHt (q.xz)));
    d += max (0.4, 0.1 * d);
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 690
float shadows(in vec3 ro, in vec3 rd, in float tmin)
{
    //return 1.0;
    const float tmax = 20.0;
    float res = 1.0;
    float k = 20.0;
    
    float t = tmin;
    for( int i=0; i<50; i++)
    {
        SDFRes h = map(ro + rd*t);
        if( h.d<0.0001 )
            return 0.0;
        if (t > tmax)
            return res;
        
        res = min( res, k*h.d/t );
        t += h.d;
    }
    return res;
}

// Function 691
float softShadows( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.01;
    for( int i=0; i<64; i++ )
    {
        vec3 pos = ro + rd*t;
        float h = map( pos );
        res = min( res, max(h,0.0)*164.0/t );
        if( res<0.001 ) break;
        t += h*0.5;
    }
    
    return res;
}

// Function 692
float shadows( vec3 ro, vec3 rd, float tMax, float k, int octaves ) {
    float res = 1.0;
	float t = 0.001;
	for(int i=0; i<5; i++) {
        if (t<tMax) {
			float h = map(ro + rd*t, octaves).x;
        	res = min( res, k*h/t );
        	t += h;
		}
		else break;
    }
    return clamp(res, 0.0, 1.0);
}

// Function 693
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float tp = (maxHei-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

    float res = 1.0;
    float t = mint;
    for( int i=ZERO; i<16; i++ )
    {
		float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 694
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

// Function 695
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

// Function 696
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

// Function 697
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

// Function 698
float shadow(vec3 p, vec3 n, vec3 lPos, PlantSpace ps)
{
    return shadow(p + n * MIN_DST * 40.0, lPos, ps);
}

// Function 699
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

// Function 700
float hardShadow(ray _r, vec3 _light)
{
    ray r = _r;
    r.o = _r.hp;
    r.d = normalize(_light - _r.hp);
    
    float s = 1.;
    float t = 0.02;    
    for(int i=0; i<30; i++)
    {        
        vec3 p = r.o + r.d * t;
        
        float d = map(p).x;
        
        if(d < 0.002)
            return 0.;
        
        t += d;
    }
    
    return 1.;
}

// Function 701
bool shadowHit(const in Ray ray) 
{
    float _t;
    vec3 _p, _n;
    for (int i = 0; i < SPHERES_NB; i++) {
        if (intersectsSphere(ray, spheres[i], _t, _p, _n)) return true;
    }
    return false;
}

// Function 702
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...


    rgb = rgb*rgb * (3.0-2.0*rgb);
   	rgb = pow(rgb, vec3(0.45));
    rgb  = rgb * 2.;

	// Vignette...
    rgb *= .7+0.5*pow(250.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.3);	


	return clamp(rgb, 0.0, 1.0);
}

// Function 703
float shadow(vec3 px)
{
    // check whether the shadow maps z value is higher than the current one
    // if it's not, the point lies in shadow
     vec4 lookup = texture(iChannel0,px.xy);
     return float(lookup.x  > px.z - 0.002); 
}

// Function 704
float softShadow(vec3 ro, vec3 lp, float k, float t){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 48; 
    
    vec3 rd = lp-ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = .0025*(t*.125 + 1.);  // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        dist += clamp(h, .07, .5); 
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0.0 || dist > end) break; 
    }

    // I've added a constant to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me. Sometimes, I'll add AO also just for kicks. :)
    return min(max(shade, 0.) + .15, 1.); 
}

// Function 705
float volumetric_player_shadow(vec3 p, vec3 rel_cam_pos)
{
#if VOLUMETRIC_PLAYER_SHADOW
    vec3 occluder_p0 = rel_cam_pos;
    vec3 occluder_p1 = occluder_p0 - vec3(0, 0, 48);
#if VOLUMETRIC_PLAYER_SHADOW >= 2
    occluder_p0.z -= 20.;
#endif // VOLUMETRIC_PLAYER_SHADOW >= 2

    float window_dist = p.x * (1. / VOL_SUN_DIR.x);
    float occluder_dist = occluder_p0.x * (1. / VOL_SUN_DIR.x);
    p -= VOL_SUN_DIR * max(0., window_dist - occluder_dist);
    vec3 occluder_point = closest_point_on_segment(p, occluder_p0, occluder_p1);
    float vis = linear_step(sqr(16.), sqr(24.), length_squared(p - occluder_point));

#if VOLUMETRIC_PLAYER_SHADOW >= 2
    vis = min(vis, linear_step(sqr(8.), sqr(12.), length_squared(p - rel_cam_pos)));
#endif // VOLUMETRIC_PLAYER_SHADOW >= 2

    return vis;
#else
    return 1.;
#endif // VOLUMETRIC_PLAYER_SHADOW
}

// Function 706
vec4 texture_blurred2_quantized(in sampler2D tex, vec2 uv, vec3 q)
{
    return (quantize(texture(iChannel0, uv), q)
        + quantize(texture(iChannel0, vec2(uv.x+1.0, uv.y+1.0)), q)
		+ quantize(texture(iChannel0, vec2(uv.x+1.0, uv.y-1.0)), q)
        + quantize(texture(iChannel0, vec2(uv.x-1.0, uv.y+1.0)), q)
		+ quantize(texture(iChannel0, vec2(uv.x-1.0, uv.y-1.0)), q)
		+ quantize(texture(iChannel0, vec2(uv.x+1.0, uv.y)), q)
		+ quantize(texture(iChannel0, vec2(uv.x-1.0, uv.y)), q)
		+ quantize(texture(iChannel0, vec2(uv.x, uv.y+1.0)), q)
		+ quantize(texture(iChannel0, vec2(uv.x, uv.y-1.0)), q))/9.0;
}

// Function 707
bool shadowSphere( in vec3 ro, in vec3 rd, in vec3 cen, in float rad, in float tmax )
{
	vec3 oc = ro - cen;
	float b = dot( oc, rd );
	float c = dot( oc, oc ) - rad*rad;
	float h = b*b - c;
	if( h<0.0 ) return false;
	float t = -b - sqrt( h );
    return t>0.0 && t<tmax;
}

// Function 708
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

// Function 709
float shadow( vec3 ro, vec3 rd )
{
    bool hit = false;
    vec3 p = ro + rd;
    float t = 0.;
    float k = 16.;
    float res = 1.;
    
    for( int i=0;i<32;i++)
    {
       float d = map(p);
        
        t+=d;
        res = min( res, k*d/t );
        
        if(d<EPS)
        {
            hit = true;
            res = 0.;
            break;
        }
        else if(t>15.)
        {
            hit = false;
            break;
        }
      
        p = ro + rd * t * 0.45;
    }
    
    return res;
}

// Function 710
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

// Function 711
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

// Function 712
float shadow(vec3 ro, vec3 rd)
{
    float res = 1.0;
    float t = PRECIS * 30.0;
    for( int i=0; i < 30; i++ )
    {
		float distToSurf = map( ro + rd*t );
        res = min(res, 8.0 * distToSurf / t);
        t += distToSurf;
        if(distToSurf < PRECIS || t > DMAX) break;
    }
    
    return clamp(res, 0.0, 1.0);
}

// Function 713
float softShadow(vec3 pos, vec3 rayDir){
    float res = 1.0;
    float t = 1.0;
    //float ph = 1e10;
    //Start some small distance away from the surface to avoid artifacts
    pos += rayDir * 5.0 * t;
	for(int i = 0; i < SHADOW_STEPS; i++){
	    vec3 p = pos + t * rayDir;
        if(p.y > 2.0*HEIGHT){
        	break;
        }
        float h = p.y - getHeight(p, terrainLimit);
		res = min(res, SHADOW_SHARPNESS * h / t );
        /*
		//An improved shadow approach that didn't quite work
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, SHADOW_SHARPNESS*d/max(0.0,t-y) );
        ph = h;
		*/
		t += h;
        if(res < EPSILON){
            break;
        }
	}
	return clamp(res, 0.0, 1.0);
}

// Function 714
float softshadow(in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k) {
	float res = 1.0, h, t = mint;
    for( int i=0; i<20; i++ ) {
		h = DE( ro + rd*t ).x;
		res = min( res, k*h/t );
                if( res<0.0001 ) break;
		t += 0.02;
    }
    return clamp(res, 0., 1.);
}

// Function 715
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

// Function 716
bool shadowRayCast(vec3 r, vec3 light_pos)
{
    vec3 d = light_pos - r;
    
    float lowest_t = length( d );

    d = normalize(d);
    
    bool hit = false;
    
    for( int j = 0; j < NUM_SPHERES - 1; ++j )
    {
		float t;
        if( isect_sphere( r, d, g_spheres[j].xyz, g_spheres[j].w, t ) )
        {
			if( t < lowest_t )
            {
                lowest_t = t; hit = true;
            }
        }
    }
    
    for( int j = 0; j < NUM_PLANES; ++j )
    {
		vec3 np = g_planes[j].xyz;
        float np_dot_d = dot(np, d);
        if( abs(np_dot_d) > 0.001 )
        {
            float t = (g_planes[j].w - dot(np, r)) / np_dot_d;
            if( t > 0.0 && t < lowest_t )
            {
				lowest_t = t; hit = true;
            }
        }
    }
    
    return hit;
}

// Function 717
float softShadow(vec3 ro, vec3 lp, float k, float t){

    // More would be nicer. More is always nicer, but not really affordable.
    const int maxIterationsShad = 24; 
    
    vec3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = 0.001*(t*.125 + 1.);  // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        dist += clamp(h, .01, .5); 
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0. || dist > end) break; 
    }

    // I've added a constant to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me. Sometimes, I'll add AO also just for kicks. :)
    return min(max(shade, 0.) + .05, 1.); 
}

// Function 718
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float res = 1.0;
    float t = mint;
    float ph = 1e10;
    
    for( int i=0; i<32; i++ )
    {
        float h = map( ro + rd*t );
        res = min( res, 10.0*h/t );
        t += h;
        if( res<0.0001 || t>tmax ) break;
        
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 719
float shadow(in vec3 light, in vec3 eye, in vec3 ray, in vec3 m, in vec3 normal) {
    
    const float wallDist = 42.0;
    vec3 wallPoint = ray / ray.z * wallDist;

    vec3 lightdir = normalize(wallPoint - light);

    const float e = 0.2;
    float nbColl = 1.0;
    
    float i = getFieldIntensity(light, lightdir, m, normal);
    
    
    return 1.0 - smoothstep(7.5,17.5, i);
    
    
    /*
    if(rayMarching(light, lightdir, m, normal)) {
        
        float shadow = max(dot(-lightdir,normal), 0.0);
        
        return 1.0 - shadow;
    }
    else {
	 	return 1.0;   
    }
	*/
}

// Function 720
float SelfShadow(vec3 ro, vec3 rd)
{
    float k = 32.0;
    float res = 1.0;
    float t = 0.1;          // min-t see http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
    for (int i=0; i<16; i++)
    {
        float h = Dist(ro + rd * t);
        res = min(res, k*h/t);
        t += h;
        if (t > 4.0) break; // max-t
    }
    return clamp(res, 0.0, 1.0);
}

// Function 721
vec4 gaussianblur(sampler2D tex, vec2 xy, vec2 res, float sizered, float sizegreen, float sizeblue, float sizealpha, vec2 dir) {
    vec4 sigmas = vec4(sizered, sizegreen, sizeblue, sizealpha);

    // Set up state for incremental coefficient calculation, see GPU Gems
    // We use vec4s to store four copies of the state, for different size
    // red/green/blue/alpha blurs
    vec4 gx = vec4(0.0);
    vec4 gy = vec4(0.0);
    vec4 gz = vec4(0.0);

    gx = 1.0 / (sqrt(2.0 * 3.141592653589793238) * sigmas);
    gy = exp(-0.5 / (sigmas * sigmas));
    gz = gy * gy;
    // vec4 a, centre, sample1, sample2 = vec4(0.0);
    vec4 a = vec4(0.0);
    vec4 centre = vec4(0.0);
    vec4 sample1 = vec4(0.0);
    vec4 sample2 = vec4(0.0);

    // First take the centre sample
    centre = texture(tex, xy / res);
    a += gx * centre;
    vec4 energy = gx;
    gx *= gy;
    gy *= gz;

    // Now the other samples
    float support = max(max(max(sigmas.r, sigmas.g), sigmas.b), sigmas.a) * 3.0;
    for(float i = 1.0; i <= support; i++) {
        sample1 = texture(tex, (xy - i * dir) / res);
        sample2 = texture(tex, (xy + i * dir) / res);
        a += gx * sample1;
        a += gx * sample2;
        energy += 2.0 * gx;
        gx *= gy;
        gy *= gz;
    }
    
    a /= energy;
    
    if(sizered < 0.1) a.r = centre.r;
    if(sizegreen < 0.1) a.g = centre.g;
    if(sizeblue < 0.1) a.b = centre.b;
    
    return a; 
}

// Function 722
float getBlurredAlpha(vec2 uv) {
 
    float sum = 0.;
    int iter = 0;
    
    for(int i = 0; i < BLUR_SAMPLES; i++) {
    
        //float div = float(i) + 1.;
        float div = 1.;
        sum += texture(iChannel0, uv).a;
        sum += texture(iChannel0, uv + vec2(BLUR_SIZE,0.) / div).a;
        sum += texture(iChannel0, uv + vec2(BLUR_SIZE,0.) / 2. / div).a;
        sum += texture(iChannel0, uv + vec2(BLUR_SIZE,0.) / 4. / div).a;
        sum += texture(iChannel0, uv + vec2(BLUR_SIZE,0.) / 6. / div).a;
        sum += texture(iChannel0, uv + vec2(0.,BLUR_SIZE) / div).a;
        sum += texture(iChannel0, uv + vec2(0.,BLUR_SIZE) / 2. / div).a;
        sum += texture(iChannel0, uv + vec2(0.,BLUR_SIZE) / 4. / div).a;
        sum += texture(iChannel0, uv + vec2(0.,BLUR_SIZE) / 6. / div).a;
        
        iter += 9;
        
    }
    
    return sum / float(iter);
    
}

// Function 723
float sphSoftShadow( in vec3 ro, in vec3 rd, in vec4 sph )
{
    float s = 1.0;
    vec2 r = sphDistances( ro, rd, sph );
    if( r.y>0.0 )
        s = max(r.x,0.0)/r.y;
    return s;
}

// Function 724
vec3 effect(vec3 p) 
{
    float time = iTime*0.05;
#ifndef SHAPE_ROTATION
    time = 0.52;
#endif
	mat3 mx = getRotXMat(-7.*(sin(time*2.)*.5+.5));
	mat3 my = getRotYMat(-5.*(sin(time*1.5)*.5+.5));
	mat3 mz = getRotZMat(-3.*(sin(time)*.5+.5));
	
	mat3 m = mx*my*mz;
	
	float d = min(min(pattern(p*m), pattern(p*m*m)), pattern(p*m*m*m));
    
    return vec3(d/0.94); 
}

// Function 725
float softShadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
    {
        float res = 1.0;
        float h = mint;
        for( int i=0; i < 100; i++ )
        {
            ro += rd * h;
            h = map(ro).x;
            if( h<0.01 )
                return 0.0;
            res = min( res, k*h );
            if (h > maxt)
                break;
        }

        return res;
    }

// Function 726
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
	float h = 1.0;
    for( int i=0; i<10; i++ )
    {
        h = map(ro + rd*t).x;
        res = min( res, smoothstep(0.0,1.0,k*h/t) );
		t += clamp( h, 0.02, 2.0 );
		if( res<0.01 || t>10.0 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 727
float effectStar( in vec2 uv ) {
    float theta = atan(uv.y+1.2, uv.x);
    float temp = sin(theta*12.0+iTime*7.45)*0.5+0.5;
    return temp;
}

// Function 728
float shadow(vec3 o, vec3 n){
	float mint=lit0.shadowStart;
	float maxt=lit0.shadowEnd;
	float k = lit0.shadowSoft;
	float res = 1.;
	float t=mint;
	float ph = 1e10; // big, such that y = 0 on the first iteration
	for( int i=0; i < ITERATION; i++){
		float h = sdScene(o + lit0.direction*t, true).x;
#if 1
		res = min( res, k*h/t);
#else
		float y = h*h/(2.0*ph);
		float d = sqrt(h*h-y*y);
		res = min( res, k*d/max(0.0,t-y) );
		ph = h;
#endif
		t += h;
		if( res<0.0001 || t>maxt ) break;
	}
	return sat(res);
}

// Function 729
float calcSoftshadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    float t = 0.0005;                 // selfintersection avoidance distance
	float h = 1.0;
    for( int i=0; i<40; i++ )         // 40 is the max numnber of raymarching steps
    {
        h = doModel(ro + rd*t).x;
        res = min( res, 64.0*h/t );   // 64 is the hardness of the shadows
		t += clamp( h, 0.02, 2.0 );   // limit the max and min stepping distances
    }
    return clamp(res,0.0,1.0);
}

// Function 730
float softshadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
	float res = 1.0;
	for (float t = mint; t < maxt;) {

		float h = map(ro + rd * t).x;

		if (h < 0.001) {
			return 0.0;
		}
		res = min(res, k * h / t);
		t += h;
	}
	return res;
}

// Function 731
float Shadow(vec3 pos)
{
    float shadow = 1.0;
    float depth  = 0.1;
    
    for(int i = 0; i < 32; ++i)
    {
        vec3 p = pos + (SunLightDir * depth);
        vec2 sdf = Scene(p);
        
        shadow = min(shadow, (16.0 * sdf.x) / depth);
        depth += sdf.x;
        
        if(sdf.x < 0.001)
        {
            break;
        }
    }
    
    return shadow;
}

// Function 732
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = fScene( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 733
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
    traceFlame = false;
    float res = 1.0;
    float t = mint;
    for(int i=0; i<14; i++)
    {
    	float h = map(ro + rd*t, false).x;
        res = min(res, 7.0*h/t + 0.01*float(i));
        t += 0.3*clamp(h, 0.01, 0.25);
        if( h<0.001 || t>tmax ) break;
    }
    traceFlame = true;
    return clamp( res, 0.0, 1.0 );
}

// Function 734
float hardShadow(vec3 dir, vec3 origin, float min_t) {
    float t = min_t;
    for(int i = 0; i < RAY_STEPS; ++i) {
        vec3 obj = origin + t * dir;
        float m = sceneMap3D(obj);
        if(dot(obj - lightPos, origin - lightPos) < 0.) 
            return 1.0;
        if(m < 0.0001) {
            // return 0.0;
            return float(i) * 0.01;
        }
        t += m;
    }
    return 1.0;
}

// Function 735
float Shadow( in vec3 ro, in vec3 rd)
{
	float res = 1.0;
    float dt = 0.03;
    float t = .01;
    for( int i=0; i<10; i++ )
    {
		if( t < .15)
		{
			float h = Scene(ro + rd * t).x;
			res = min( res, 2.2*h/t );
			t += .005;
		}
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 736
float shadowcast_pointlight(in vec3 ro, in vec3 rd, in float light_dist){
    light_dist = min(light_dist, FAR_CLIP);
    float res = 1.f;
    float t = MIN_CLIP;
    float ph = 1e20;
    for (int i = 0; i < RAY_STEPS; i++){
        vec3 pos = ro + rd * t;
        float h = map(pos).x;
        
        float y = h*h/(2.*ph);
        float d = sqrt(h*h-y*y);
        res = min(res, 10.f * d/max(0., t-y));
        ph = h;
        t += h;

        if (res < 0.0001 || t > light_dist){
            break;
        }
    }
    return res;
}

// Function 737
float effectSpiral( in vec2 uv ) {
    vec2 polar = vec2(atan(uv.y, uv.x)/PI*5.0, log(length(uv)+1.0)*4.0);
    polar.x -= iTime*0.6;
    polar.y += polar.x*0.5 - iTime * 1.2;
    vec2 f = fract(polar);
    return max(f.x, f.y);
}

// Function 738
float shadow(in ray ray, int maxSteps)
{
    float res = 0.0;

    float t = 0.001;
    
    float k = 8.0;
    
    float h = 0.1;
    
    for(int i=1; i<maxSteps+1; i++ )
    {
        vec3 samplePoint = ray.origin+ray.direction*t;
        if (samplePoint.y >= 1.0 || samplePoint.y <= -1.0 )//|| max(abs(samplePoint.x),abs(samplePoint.z)) >= 6.0)
        {
            return 1.0;
        }
        h = sceneDistanceFunction(ray.origin+ray.direction*t, orbitTrap);
        res = min( res, (k*h)/t );
        //if( h<0.0015*pow(distance(ray.origin, samplePoint), 1.0) || t>tmax) break;
        if(h < 0.00009 || t > maxDepth) 
        {
            break;
        }
        t += h;
    }

    if(t > maxDepth)
    {
        res = 1.0;
    }
    
    return res;
}

// Function 739
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

// Function 740
float calcSoftshadow( in vec3 ro, in vec3 rd, float tmin, float tmax, const float k )
{
    vec2 bound = sphIntersect( ro, rd, 2.1 );
    tmin = max(tmin,bound.x);
    tmax = min(tmax,bound.y);
    
	float res = 1.0;
    float t = tmin;
    for( int i=0; i<50; i++ )
    {
    	vec4 kk;
		float h = map( ro + rd*t, kk, false );
        res = min( res, k*h/t );
        t += clamp( h, 0.02, 0.20 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 741
float calcShadow(vec3 p, vec3 lightPos) {
    // Thanks iq.
    vec3 rd = normalize(lightPos - p);
	float res = 0.9, t = 0.0;
    for (float i = -0.1; i < SHADOW_STEPS; i++)
    {
		float h = map(p + rd * t).d;
        res = min(res, 2.0 * h / t);
        t += h;
        if (res < 0.0009 || t > 15.0) break;
    }
    return clamp(res, -0.1, 0.9);
}

// Function 742
float castShadow(vec3 ro,vec3 rd){
	float res = 1.0;
    float t = 0.001;
    for(int i = 0;i < 70;i++){
    	vec3 pos = ro+rd*t;
        float h = map(pos).x;
        res = min(16.*h/t,res);
        if(res<0.001) break;
        t+=h;
        if(t>20.0) break;
    
    }
    return clamp(res,0.0,1.);

}

// Function 743
vec3 calculate_radial_blur(vec2 uv)
{
    vec3 sample_sum = vec3(0.0);
	float weight_sum = 0.0;
    
    const vec2 screen_center = vec2(0.5);
    vec2 blur_vector = screen_center - uv;
    
    //const float step_size = blur_max_distance / float(blur_sample_count);	// TODO: Importance sampling?
    
    const float step_size = blur_max_distance / float(blur_sample_count);	// TODO: Importance sampling?
    //step_size *= length(blur_vector);

    for(int i=0; i<blur_sample_count; ++i)
    {
        float sample_weight = 1.0 / (10.0 + float(i) * 10.0);
        //float sample_weight = 1.0 - 1.0 / float(blur_sample_count) * float(i);
        //sample_weight = 1.0;
        //sample_weight *= sample_weight;	// Square the weight.
        sample_weight = pow(sample_weight, 1.5);
        
        //sample_weight = 0.005;
        //sample_weight = pow(length(blur_vector), 2.0);
        
        sample_weight = 1.0;
        //sample_weight = pow(length(blur_vector), 2.0);

        sample_weight = pow(1.0 - float(i) / float(blur_sample_count), 1.0);
        
        float offset_scale = abs(uv.x - 0.5) * 2.0;	// TODO: Is this fine? Or should it be radial?       

        vec2 sample_offset = blur_vector * step_size * float(i) * offset_scale;
        
        vec3 s = texture(iChannel0, uv + sample_offset).rgb;
        //sample_weight = max(length(s) - 0.1, 0.0);
            
        sample_sum += s * sample_weight;
        weight_sum += sample_weight;
    }
    
    //bloom /= float((2 * radius + 1) * (2 * radius + 1));
    sample_sum /= weight_sum;
    
    sample_sum += texture(iChannel1, uv).rgb * 0.5;
    
    return(sample_sum);
}

// Function 744
vec4 texture_blurred_quantized(in sampler2D tex, vec2 uv, vec3 q)
{
    return (quantize(texture(iChannel0, uv), q)
		+ quantize(texture(iChannel0, vec2(uv.x+1.0, uv.y)), q)
		+ quantize(texture(iChannel0, vec2(uv.x-1.0, uv.y)), q)
		+ quantize(texture(iChannel0, vec2(uv.x, uv.y+1.0)), q)
		+ quantize(texture(iChannel0, vec2(uv.x, uv.y-1.0)), q))/5.0;
}

// Function 745
vec4 blur9(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
    vec4 color = vec4(0.0);
    vec2 off1 = vec2(1.3846153846) * direction;
    vec2 off2 = vec2(3.2307692308) * direction;
    color += texture(image, uv) * 0.2270270270;
#if BLUR_OUTWARDS
    color += texture(image, uv - (off1 / resolution)) * 0.3162162162 * 2.0;
    color += texture(image, uv - (off2 / resolution)) * 0.0702702703 * 2.0;
#else
    color += texture(image, uv + (off1 / resolution)) * 0.3162162162;
    color += texture(image, uv - (off1 / resolution)) * 0.3162162162;
    color += texture(image, uv + (off2 / resolution)) * 0.0702702703;
    color += texture(image, uv - (off2 / resolution)) * 0.0702702703;
#endif
    return color;
}

// Function 746
float MarchShadow(vec2 orig, vec2 dir)
{
    float d = 0.0;
    
    for(int i = 0;i < MAX_STEPS;i++)
    {
        float ds = Scene(dir * d - orig);
        
        d += ds;
        
        if(ds < EPS)
        {
        	break;   
        }
    }
    
    return d;
}

// Function 747
float softshadow( in vec3 ro, in vec3 rd, float mint, float k, in vec4 c )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<64; i++ )
    {
        vec4 kk;
        float h = map(ro + rd*t, kk, c);
        res = min( res, k*h/t );
        if( res<0.001 ) break;
        t += clamp( h, 0.01, 0.5 );
    }
    return clamp(res,0.0,1.0);
}

// Function 748
bool in_earth_shadow( vec3 p, vec3 sun_direction ) { return ( dot( p, sun_direction ) < 0.0 ) && ( lensqr( p - project_on_line1( p, earth_center, sun_direction ) ) < earth_radius * earth_radius ); }

// Function 749
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

// Function 750
float shadow_march( in vec3 ro, in vec3 rd)
{
    float t=0.01,d;
    
    for(int i=0;i<STEPS;i++)
    {
        d = map(ro + rd*t).d;
        if( d < 0.0001 )
            return 0.0;
        t += d;
    }
    return 1.0;
}

// Function 751
float shadowTrace(vec2 o, vec2 r){
    
    // Raymarching.
    float d, t = 0.;
    
    
    // 96 iterations here: If speed and complilation time is a concern, choose the smallest 
    // number you can get away with. Apparently, swapping the zero for min(0, frame) can
    // force the compliler to not unroll the loop, so that can help sometimes too.
    for(int i=0; i<16;i++){
        
        // Surface distance.
        d = map(o + r*t);
        
        // In most cases, the "abs" call can reduce artifacts by forcing the ray to
        // close in on the surface by the set distance from either side.
        if(d<0. || t>FAR) break;
        
        
        // No ray shortening is needed here, and in an ideal world, you'd never need it, but 
        // sometimes, something like "t += d*.7" will be the only easy way to reduce artifacts.
        t += d*RSF_SHAD;
    }
    
    t = min(t, FAR); // Clipping to the far distance, which helps avoid artifacts.
    
    return t;
    
}

// Function 752
float softshadow(vec3 rayOrigin, vec3 rayDir, float mint, float maxt) {
	float k = 8.0; // how soft the shadow is (a constant)
    float res = 1.0;
    float t = mint;
    for(int i=0; i<10; i++) {
        float h = map(rayOrigin + t*rayDir).x;
        res = min(res, k*h/t);
        t += h; // can clamp how much t increases by for more precision
        if( h < 0.001 ) {
        	break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

// Function 753
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

// Function 754
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
        float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 755
vec3 Blur (vec2 uv)
{
    const int radius = 1;
    
    vec3 color = vec3(0.0);
    float kernel = 0.0;
    vec2 uvScale = 1.75 / iChannelResolution[0].xy;
    
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            float k = 1.0 / pow( 2.0, abs(float(x)) + abs(float(y)) );
            
            color += texture( iChannel0, uv + vec2(x,y) * uvScale ).rgb * k;
            kernel += k;
        }
    }
    
    return color / kernel;
}

// Function 756
float directionalShadow(vec3 cameraTarget, vec3 modelPosition, vec3 p, vec3 dirLightPos)
{
    float shadow = 0.;
    mat3 dirLightView = getView(dirLightPos, cameraTarget);
    vec3 dirLightRayDirection = dirLightView * modelPosition;

    float dt = 0., d = 0.;
    // shadow depth
    float shadowDepth = length((dirLightPos-p));
    dirLightRayDirection = normalize(-dirLightPos+p);
    vec3 shadowPos;
    for( int i = 0; i < maxSteps; ++i )
    {
        vec3 p = dirLightPos + dirLightRayDirection*d;
        int id;	// id of object that is being drawn
        dt = drawScene(p,id);
        if( dt < EPSILON )
        {
            float currentShadowDepth = length((dirLightPos-p));
            // 
            if( currentShadowDepth/shadowDepth < 1.-EPSILON)
                shadow = 1.;	
            break;
        }
        d += dt;
    }
    return shadow;
}

// Function 757
float raymarchShadow(Ray ray)
{
    float shadow = 1.;
	float t = CAMERA_NEAR;
    vec3 p = vec3(0.);
    float h = 0.;
    for(int i = 0; i < 80; ++i)
	{
	    p = ray.origin + t * ray.direction;
        h = p.y - terrainFbm(p.xz, MQ_OCTAVES, iChannel0);
		shadow = min(shadow, 8. * h / t);
		t += h;
		if (shadow < 0.001 || p.z > CAMERA_FAR) break;
	}
	return SAT(shadow);
}

// Function 758
float shadowray(vec3 eye, vec3 dir, float maxd) {
    float d, i, r=1., ph=1e10;
    for(; i<100. && d<maxd; i++){
     	vec3 p = eye + dir * d;
        float ind = space(p).x;
        if (abs(ind) < 0.001 * d)return 0.;
        
        float y = ind*ind/(2.0*ph),
        nd = sqrt(ind*ind-y*y);
        r = min( r, 10.0*nd/max(0.0,d-y) );
        
        d += ind;
    }

    return r;
}

// Function 759
Intersection traceShadow(in Ray ray)
{
    Intersection bestIts;
	bestIts.t = ray.tMax;
    bestIts.isLight = false;
    
    for (int i = 0; i < AMOUNT_LIGHTS; ++i)
    {
        Intersection its = intersectLight(ray, lights[i]);
        if (its.t < bestIts.t)
        {
            bestIts = its;
            bestIts.isLight = true;
            bestIts.light = lights[i];
        }
    }
    
    for (int i = 0; i < AMOUNT_SPHERES; ++i)
    {
        Intersection its = intersectSphere(ray, spheres[i]);
        if (its.t < bestIts.t)
        {
            bestIts = its;
            bestIts.m = spheres[i].m;
            bestIts.isLight = false;
            return bestIts;
        }
    }

    return bestIts;
}

// Function 760
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    // bounding volume
    float tp = (0.8-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

    float res = 1.0;
    float t = mint;
    for( int i=ZERO; i<24; i++ )
    {
		float h = map( ro + rd*t ).x;
        float s = clamp(8.0*h/t,0.0,1.0);
        res = min( res, s*s*(3.0-2.0*s) );
        t += clamp( h, 0.02, 0.2 );
        if( res<0.004 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 761
float softShadow(vec3 ro, vec3 lp, float k){

    // More would be nicer. More is always nicer, but not really affordable.
    const int maxIterationsShad = 32; 
    
    vec3 rd = (lp-ro); // Unnormalized direction ray.

    float shade = 1.0;
    float dist = 0.01;    
    float end = max(length(rd), 0.001);
    float stepDist = end/float(maxIterationsShad);
    
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        //dist += min(h, stepDist); // So many options here, and none are perfect: dist += min( h, 0.2 ), etc
        dist += clamp(h, .02, .25); // So many options here, and none are perfect: dist += min( h, 0.2 ), etc
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0. || dist > end) break; 
    }

    // I've added 0.5 to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me.
    return min(max(shade, 0.) + .05, 1.); 
}

// Function 762
float shadowMarch( vec3 ro, vec3 rd ) {
	float dO = 0.01;
    float res = 1.0;
    
    for (int i = 0; i < 64; i++) {
		float h = getDist( ro + rd * dO );

        res = min( res, 10.0 * h / dO );  
        dO += h;
        
        if( res < 0.0001 || dO > RAYMARCH_MAX_DIST ) break;
    }
    
    return res;//clamp( res, 0.0, 1.0 );
}

// Function 763
float softShadow(vec3 ro, vec3 lp, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 14; 
    
    vec3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = .002;    
    float end = max(length(rd), .001);
    float stepDist = end/float(maxIterationsShad);
    
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = 0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, .2), 
        // clamp(h, .02, stepDist*2.), etc.
        dist += clamp(h, .02, .25);
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0. || dist>end) break; 
        //if (h<.001 || dist > end) break; // If you're prepared to put up with more artifacts.
    }

    // I've added 0.5 to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me.
    return min(max(shade, 0.) + .25, 1.); 
}

// Function 764
float getBlurSize(float depth, float focusPoint, float focusScale)
{
	float coc = clamp((1.0 / focusPoint - 1.0 / depth)*focusScale, -1.0, 1.0);
    return abs(coc) * MAX_BLUR_SIZE;
}

// Function 765
vec4 BlurA(vec2 uv, int level)
{
    return BlurA(uv, level, iChannel0, iChannel3);
}

// Function 766
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<64; i++ )
    {
        float h = mapTerrain(ro + rd*t);
		h = max( h, 0.0 );
        res = min( res, k*h/t );
        t += clamp( h, 0.02, 0.5 );
		if( res<0.001 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 767
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

// Function 768
vec3 performGaussBlur(vec2 pos) 
{
	const float PARA1 = 0.2042, PARA2 = 0.1238, PARA3 = 0.0751;
	vec3 fragColor11 = texture(logTexture, pos - PIXEL_SIZE).xyz;
	vec3 fragColor12 = texture(logTexture, pos - vec2(0.0f, PIXEL_SIZE.y)).xyz;
	vec3 fragColor13 = texture(logTexture, pos + vec2(PIXEL_SIZE.x, -PIXEL_SIZE.y)).xyz;
	vec3 fragColor21 = texture(logTexture, pos - vec2(PIXEL_SIZE.x, 0.0f)).xyz;
	vec3 fragColor22 = texture(logTexture, pos + vec2(0.0f, 0.0f)).xyz;
	vec3 fragColor23 = texture(logTexture, pos + vec2(PIXEL_SIZE.x, 0.0f)).xyz;
	vec3 fragColor31 = texture(logTexture, pos + vec2(-PIXEL_SIZE.x, PIXEL_SIZE.y)).xyz;
	vec3 fragColor32 = texture(logTexture, pos + vec2(0.0f, PIXEL_SIZE.y)).xyz;
	vec3 fragColor33 = texture(logTexture, pos + PIXEL_SIZE).xyz;

	vec3 fragColore1 = texture(logTexture, pos + 2.0*vec2(PIXEL_SIZE.x, 0.0f)).xyz;
	vec3 fragColore2 = texture(logTexture, pos + 2.0*vec2(-PIXEL_SIZE.x, 0.0f)).xyz;
	vec3 fragColore3 = texture(logTexture, pos + 2.0*vec2(0.0f, PIXEL_SIZE.y)).xyz;
	vec3 fragColore4 = texture(logTexture, pos + 2.0*vec2(0.0f, -PIXEL_SIZE.y)).xyz;

	vec3 newColor = PARA3 * (fragColor11 + fragColor13 + fragColor31 + fragColor33) +
		PARA2 * (fragColor12 + fragColor21 + fragColor23 + fragColor32) +
		PARA1 * fragColor22;
	return newColor;
}

// Function 769
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 24; 
    
    ro += n*.0015;
    vec3 rd = lp - ro; // Unnormalized direction ray.
    

    float shade = 1.;
    float t = 0.;//.0015; // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = 0; i<maxIterationsShad; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        t += clamp(d, .01, .25); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}

// Function 770
float shadowVSM(vec3 px,float linearDepth)
{
    // use linear depth for better precision
   
    vec4 lookup = texture(iChannel2,px.xy);
   
    // after the blurring stages, the z and w fields
    // contain estimates of the expected value of
    // depth and squared depth
    float Ex = lookup.z;
    float Ex2 = lookup.w;
    
    // compute variance
    float variance = Ex2 - Ex*Ex;
    
    // temporary for formula
   	float znorm = linearDepth - Ex;
    float znorm2 = znorm*znorm;
    
    // compute upper bounds of probabilty of shading
    // using Chebyshev's inequality
    float p = variance/(variance + znorm2);
    
    // formula is only valid if depth is less than the expected
    // value. The max formulation is taken from an nvidia presentation
    // just called "Variance Shadow Mapping"
    return max( p, float(linearDepth <= Ex) );
}

// Function 771
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh = 1.;
  float d = 0.03;
  for (int i = 0; i < 50; i++) {
    float h = ObjDf (ro + rd * d);
    sh = min (sh, 20. * h / d);
    d += 0.03;
    if (h < 0.001) break;
  }
  return clamp (sh, 0., 1.);
}

// Function 772
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.01;
  for (int j = 0; j < 40; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += min (0.2, 3. * h);
    if (sh < 0.001) break;
  }
  return 0.6 + 0.4 * sh;
}

// Function 773
float shadow(vec3 p) {
	float sh=1.,td=.1;
    for (int i=0; i<50; i++) {
		p+=ldir*td;
        float d=de(p);
		td+=d;
        sh=min(sh,10.*d/td);
        if (sh<.05) break;
    }
    return clamp(sh,0.,1.);
}

// Function 774
float SoftShadow(in vec3 ro, in vec3 rd) {
    float res = 1.0, h, t = .02;
    for( int i=0; i<15; i++ ) {
		h = DE( ro + rd*t ).x;
		res = min( res, 10.*h/t );
		t += .06;
    }
    return clamp(res, 0., 1.);
}

// Function 775
vec4 depthBlur(sampler2D channel, vec2 _uv) {
    vec4 s = tex(channel, _uv), o = s;
	vec3 dst = clamp(s.a*.075-1.25,0.,1.) * 1./iResolution.y * vec3(1.,-1.,0.);
    if (dst.x < .0001) return s;
    //free lvl 0
    vec2 uv = _uv + .5/iResolution.xy;
    //sample lvl 1
    o += .5*tex(channel, uv + dst.xz); o += .5*tex(channel, uv + dst.yz);
	o += .5*tex(channel, uv + dst.zx); o += .5*tex(channel, uv + dst.zy);
    //sample lvl 2
    o += .25*tex(channel, uv + dst.xx); o += .25*tex(channel, uv + dst.xy);
	o += .25*tex(channel, uv + dst.yx); o += .25*tex(channel, uv + dst.yy);
    return o / 4.;
}

// Function 776
float shadow(in vec3 ro, in vec3 rd, float mint, float maxt, float k) {  
    float res = 1.0;
    for (float t = mint; t < maxt;) {
        float h = map(ro + rd * t, false).x;
        if (h < 0.001)
            return 0.0;
        res = min(res, k * h / t);
        t += h;
    }
    return res;    
}

// Function 777
float shadow(v3 o,v3 i){
 const float a=32.;//shadow hardnes
 float r=1.,h =1.,t=.0005;//t=(self)intersection avoidance distance
 for(int j=0;j<IterSh;j++){
  h=dm(o+i*t).x;
  r=min(r,h*a/t);
  t+=clamp(h,.02,2.);}//limit max and min stepping distances
 return clamp(r,0.,1.);}

// Function 778
float soft_shadow_march( in vec3 ro, in vec3 rd, float k)
{
    float res = 1.0;
    float t=0.01;//.0001*sin(PI*fract(iTime));
    float d;
    
    for(int i=0;i<STEPS;i++)
    {
        d = map(ro + rd*t).d;
        if( d < PRECISION )
            return 0.0;
        res = min( res, k*d/t );
        t += d;
    }
    return res;
}

// Function 779
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax) {
	float res = 1.0;
	float t = mint;
	for (int i = 0; i < 32; i++) {
		float h = map(ro + rd * t).x;
		res = min(res, 8.0 * h / t);
		t += clamp(h, 0.02, 0.10);
		if (res < 0.005 || t > tmax) break;
	}
	return clamp(res, 0.0, 1.0);
}

// Function 780
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

// Function 781
void add_teleporter_effect(inout vec4 fragColor, vec2 fragCoordNDC, vec3 camera_pos, float teleport_time)
{
    if (teleport_time <= 0.)
        return;
    
    const float TELEPORT_EFFECT_DURATION = .25;

    // at 144 FPS the trajectories are too obvious/distracting
    const float TELEPORT_EFFECT_FPS = 60.;
    float fraction = floor((iTime - teleport_time)*TELEPORT_EFFECT_FPS+.5) * (1./TELEPORT_EFFECT_FPS);
    
    if (fraction >= TELEPORT_EFFECT_DURATION)
        return;
    fraction = fraction * (1./TELEPORT_EFFECT_DURATION);

    const int PARTICLE_COUNT = 96;
    const float MARGIN = .125;
    const float particle_radius = 12./1080.;
    float aspect = min_component(iResolution.xy) / max_component(iResolution.xy);
    float pos_bias = (-1. + MARGIN) * aspect;
    float pos_scale = pos_bias * -2.;

    // this vignette makes the transition stand out a bit more using just visuals
    // Quake didn't have it, but Quake had sound effects...
    float vignette = clamp(length(fragCoordNDC*.5), 0., 1.);
    fragColor.rgb *= 1. - vignette*(1.-fraction);

    int num_particles = NO_UNROLL(PARTICLE_COUNT);
    for (int i=0; i<num_particles; ++i) // ugh... :(
    {
        vec4 hash = hash4(teleport_time*13.37 + float(i));
        float speed = mix(1.5, 2., hash.z);
        float angle = hash.w * TAU;
        float intensity = mix(.25, 1., fract(float(i)*PHI + .1337));
        vec2 direction = vec2(cos(angle), sin(angle));
        vec2 pos = hash.xy * pos_scale + pos_bias;
        pos += (fraction * speed) * direction;
        pos -= fragCoordNDC;
        float inside = step(max(abs(pos.x), abs(pos.y)), particle_radius);
        if (inside > 0.)
            fragColor = vec4(vec3(intensity), 0.);
    }
}

// Function 782
float calcSoftshadow( in vec3 ro, in vec3 rd, float tmin, float tmax, const float k )
{
	float res = 1.0;
    float t = tmin;
    for( int i=0; i<50; i++ )
    {
		float h = map( ro + rd*t );
        res = min( res, k*h/t );
        t += clamp( h, 0.02, 0.20 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 783
float softShadow(in vec3 o, in vec3 d){float t=.001;
 float res=1.;
 for(int i=0;i<80;i++){
  vec3 p=o+t*d;
  float h=p.y-terrainM(p.xz);
  res=min(res,16.*h/t);t+=h;
  if(res<.001||p.y>(SC*200.))break;}
 return clamp(res,0.,1.);}

// Function 784
vec3 effect(vec2 uv, vec3 col)
{
    float granularity = yVar*20.+10.;
    if (granularity > 0.0) 
    {
        float dx = granularity / s.x;
        float dy = granularity / s.y;
        uv = vec2(dx*(floor(uv.x/dx) + 0.5),
                  dy*(floor(uv.y/dy) + 0.5));
        return bg(uv);
    }
    return col;
}

// Function 785
float terrainShadow(vec3 p, vec3 dir)
{
    vec3 s = p;
    const int n = 32;
    float di = 0.4;
    float dii = 0.002;
    float light = 1.0;
    for(int i = 0; i < n; ++i)
    {
        s += dir * di;
        float h = terrain(s.xz);
        float depth = max(h - s.y,0.0);
        light -= 0.005*exp(depth*2.0);
        di += dii;
    }
    
    return max(light,0.0);
}

// Function 786
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax, int technique )
{
	float res = 1.0;
    float t = mint;
    float ph = 1e10; // big, such that y = 0 on the first iteration
    
    for( int i=0; i<32; i++ )
    {
		float h = map( ro + rd*t );

        // traditional technique
        if( technique==0 )
        {
        	res = min( res, 10.0*h/t );
        }
        // improved technique
        else
        {
            // use this if you are getting artifact on the first iteration, or unroll the
            // first iteration out of the loop
            //float y = (i==0) ? 0.0 : h*h/(2.0*ph); 

            float y = h*h/(2.0*ph);
            float d = sqrt(h*h-y*y);
            res = min( res, 10.0*d/max(0.0,t-y) );
            ph = h;
        }
        
        t += h;
        
        if( res<0.0001 || t>tmax ) break;
        
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 787
vec3 postEffects( in vec3 col, in vec2 uv, in float time )
{
	// vigneting
	col *= 0.7+0.3*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.1 );
	return col;
}

// Function 788
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

// Function 789
float computeSoftShadow(vec3 ro, vec3 rd, float tmin, float tmax, float k)
{
    float res = 1.0;
    float ph  = 1e20;
    for( float t = tmin; t < tmax;)
    {
        float h = mapScene(ro + rd*t).x;
        if( h < 0.001)
            return 0.;
        float y = h * h / (2. * ph);
        float d = sqrt(h*h - y*y);
        res = min(res, k*d / max(0.0,t - y));
        ph  = h;
        t  += h;
    }
    return res;
}

// Function 790
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

// Function 791
void drawMineCraftBlur(Ray ray, inout TraceResult cur_ctxt)
{
    TraceResult tr_res;
    float mineT;
    tr_res = TraceMineCraft(ray.pos, ray.dir);

    mineT = sqrt(dot(tr_res.p - ray.pos, tr_res.p - ray.pos)); 
    if (mineT < cur_ctxt.t && tr_res.hit == true)
    {
        cur_ctxt.color = vec3(compute_minecraft_light(tr_res, 
                                    tr_res.sphereIntersect, ray.dir));
        cur_ctxt.n = tr_res.n;
        
        cur_ctxt.t = mineT;
        //cur_ctxt.alpha = max(0.75, 
        //                   sqrt(sqrt(dot(tr_res.p, tr_res.p))));
        cur_ctxt.alpha = GLOBAL_ALPHA;
        /* blur_effect */
        float ro = abs(dot(tr_res.p, tr_res.p) - RADIUS_MINECRAFT);
        vec3 blur_color = cur_ctxt.color;
        
        blur_color = make_another_blur(cur_ctxt.fragCoord,
                                    abs(8.0*(RADIUS_MINECRAFT - ro))).rgb;
        cur_ctxt.color = blur_color;
        
        cur_ctxt.materialType = EMISSION;
        /* end blur_effect */
    }
}

// Function 792
float softShadow(vec3 ro, vec3 lp, float k, float t){

    // More would be nicer. More is always nicer, but not really affordable.
    const int maxIterationsShad = 24; 
    
    vec3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = 0.001*(t*.125 + 1.);  // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

         
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        h = clamp(h, .1, .5); // max(h, .02);//
        dist += h;

        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (shade<0.001 || dist > end) break; 
    }

    // I've added a constant to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me. Sometimes, I'll add AO also, just for kicks. :)
    return min(max(shade, 0.) + .05, 1.); 
}

// Function 793
float GetShadow(vec3 p, vec3 plig)
{   vec3 lightPos = plig;
    vec3 l = normalize(lightPos-p);
    vec3 n = GetNormal(p);
    float dif = clamp(dot(n, l), 0., 1.);
    float d = RayMarch(p+n*MIN_DIST*2., l , MAX_STEPS/2);
    if(d<length(lightPos-p)) dif *= .1;
    return dif;
}

// Function 794
float effectCapsule( in vec2 uv ) {
    vec3 dir = normalize(vec3(uv * 0.7, 1.0));
    vec3 forward = vec3(0, 0, 1);
    mat2 rotxy = rot(iTime*0.235+0.5);
    mat2 rotzx = rot(iTime*0.412-0.7);
    dir.xy *= rotxy;
    forward.xy *= rotxy;
    dir.zx *= rotzx;
    forward.zx *= rotzx;
    vec3 light = normalize(-forward);
    vec3 from = -forward*5.0;
   	float totdist = 0.0;
    float mindist = 99999.9;
	bool set = false;
    float color = 0.25;
	for (int steps = 0 ; steps < 30 ; steps++) {
		vec3 p = from + totdist * dir;
		float dist = de(p);
        mindist = min(mindist, dist);
		totdist += dist;
		if (dist < 0.04) {
            color = (dot(normal(p), light)*0.5+0.5)*colorCap(p);
            set = true;
            break;
		}
	}
    if ( !set && mindist < 0.25 ) return 0.0;
   	return color;
}

// Function 795
float blurMask(float distanceChange, float dist, float blurAmount) {
    float blurTotal = blurAmount*.01;
    return smoothstep(blurTotal+distanceChange, -distanceChange, dist);
}

// Function 796
vec3 ShadowMask(vec2 uv)
{
    return ShadowMaskRGBCols(uv.x) * ShadowMaskRows(uv);
}

// Function 797
float shadowsoft( vec3 ro, vec3 rd, float k )
{
	float t=.1;
	float res=1.;
    for (int i=0;i<25;++i)
    {
        float h=map(ro+rd*t);
        if (h<0.001) return 0.;
		res=min(res,k*h/t);
        t+=h;
		if (t>0.23) break;
    }
    return res;
}

// Function 798
float GearShadow(vec2 uv, Gear g)
{
    float r = length(uv+vec2(0.1));
    float de = r - g.diskR + 0.0*(g.diskR - g.gearR);
    float eps = 0.4*g.diskR;
    return smoothstep(eps, 0., abs(de));
}

// Function 799
float ObjSShadow (vec3 ro, vec3 rd, float dMax)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 40; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += clamp (2. * h, 0.1, 3.);
    if (sh < 0.05 || d > dMax) break;
  }
  return 0.6 + 0.4 * sh;
}

// Function 800
float getShadow(vec3 RayPos, vec3 RayDir){
    float h = 0.0;
    float c = 0.001;
    float r = 1.0;
    float shadowCoef = 0.5;
    for(float t = 0.0; t < 50.0; t++){
        h = sdAll(RayPos + RayDir * c);
        if(h < 0.001){
            return shadowCoef;
        }
        r = min(r, h * 5.0 / c);
        c += h;
    }
    return 1.0 - shadowCoef + r * shadowCoef;
}

// Function 801
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

// Function 802
float shadow(vec2 p, vec2 pos, float radius)
{
	vec2 dir = normalize(pos - p);
	float dl = length(p - pos);
	
	// fraction of light visible, starts at one radius (second half added in the end);
	float lf = radius * dl;
	
	// distance traveled
	float dt = 0.01;

	for (int i = 0; i < 64; ++i)
	{				
		// distance to scene at current position
		float sd = sceneDist(p + dir * dt);

        // early out when this ray is guaranteed to be full shadow
        if (sd < -radius) 
            return 0.0;
        
		// width of cone-overlap at light
		// 0 in center, so 50% overlap: add one radius outside of loop to get total coverage
		// should be '(sd / dt) * dl', but '*dl' outside of loop
		lf = min(lf, sd / dt);
		
		// move ahead
		dt += max(1.0, abs(sd));
		if (dt > dl) break;
	}

	// multiply by dl to get the real projected overlap (moved out of loop)
	// add one radius, before between -radius and + radius
	// normalize to 1 ( / 2*radius)
	lf = clamp((lf*dl + radius) / (2.0 * radius), 0.0, 1.0);
	lf = smoothstep(0.0, 1.0, lf);
	return lf;
}

// Function 803
float shadow(vec3 ro, vec3 rd, vec3 div, float rand) {
    float md=1.0;
    int steps=MAX_STEPS_SHADOWS;
    float s=5.0/float(steps);
    float t=0.01;//+rand*s;
    for(int i=0; i<steps; ++i) {
        float d=map2(ro+rd*t, div);
        md=min(md,5.0*d/t);
        if(d<-0.0) {
            //md=0.0;
            break;
        } 
        t+=s;
    }

    return clamp(md,0.0,1.0);
}

// Function 804
vec4 blur5(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3333333333333333) * direction;
  color += texture(image, uv) * 0.29411764705882354;
  color += texture(image, uv + (off1 / resolution)) * 0.35294117647058826;
  color += texture(image, uv - (off1 / resolution)) * 0.35294117647058826;
  return color; 
}

// Function 805
float getCloudShadow(vec3 p, positionStruct pos)
{
	const int steps = volumetricLightShadowSteps;
    float rSteps = cloudThickness / float(steps) / abs(pos.sunVector.y);
    
    vec3 increment = pos.sunVector * rSteps;
    vec3 position = pos.sunVector * (cloudMinHeight - p.y) / pos.sunVector.y + p;
    
    float transmittance = 0.0;
    
    for (int i = 0; i < steps; i++, position += increment)
    {
		transmittance += getClouds(position);
    }
    
    return exp2(-transmittance * rSteps);
}

// Function 806
float blur(float x) {return pow(smoothstep(.945,1.0,x),10.);}

// Function 807
float softShadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<45; i++ )
    {
        float h = map(ro + rd*t).x;
        res = min( res, k*h/t );
        t += h*STEP_REDUCTION;
    }
    return clamp(res,0.0,1.0);
}

// Function 808
vec3 LightBoxShadow (in vec3 position) {
    
    float shade = 1.0;

    // calculate whether or not the position is inside of the light box
    vec2 uv;
	vec3 localPos;    
    if (GetMode() >= 1.0)
    {
        // get position in light space and get uv
        localPos = WorldSpaceToDirectionalLightSpace(position);
    	uv = localPos.xz;
        uv.x *= -1.0;
        
        // apply scaling of uv over distance to fake projection
        uv /= (1.0 + localPos.y * directionalLightUVDistanceScale);
        
        // set shade to 1 if it's inside, 0 if it's outside        
    	shade = float(abs(uv.x) < 1.0 && abs(uv.y) < 1.0);
        
        // if it is behind the light source, don't light it!
        shade *= step(0.0, localPos.y);
        
        // apply distance attenuation
        shade *=  1.0 - clamp(directionalLightFalloff * localPos.y * localPos.y, 0.0, 1.0);        
    }
    
    // soften shadows over a distance
	if (GetMode() >= 2.0)
    {
        float softenDistance = clamp(localPos.y * directionalLightSoften, 0.01, 0.99);
    	float softenX = smoothstep(1.0, 1.0 - softenDistance, abs(uv.x));
    	float softenY = smoothstep(1.0, 1.0 - softenDistance, abs(uv.y));
    	shade = shade * softenX * softenY;
    }
    
    // apply texture to light if we should!
    if (GetMode() >= 3.0)
    {
        uv = uv*0.5+0.5;
        return clamp((texture(iChannel0, uv).rgb * directionalLightTextureMADD.x + directionalLightTextureMADD.y) * shade, 0.0, 1.0);
    }
    
    return vec3(shade);
}

// Function 809
float shadow(vec3 p, vec3 n, vec3 lPos, MatSpace ps)
{
    return shadow(p + n * MIN_DST * 40.0, lPos, ps);
}

// Function 810
float shadowMap(vec3 ro, vec3 rd){
	float h = 0.0;
	float c = 0.001;
	float r = 1.0;
	float shadow = 0.5;
	for(float t = 0.0; t < 30.0; t++){
		h = map(ro + rd * c).w;
		if(h < 0.001){
			return shadow;
		}
		r = min(r, h * 16.0 / c);
		c += h;
	}
	return 1.0 - shadow + r * shadow;
}

// Function 811
float Scene_TraceShadow( const in vec3 vRayOrigin, const in vec3 vRayDir, const in float fMinDist, const in float fLightDist )
{
    // Soft Shadow Variation
    // https://www.shadertoy.com/view/lsKcDD    
    // based on Sebastian Aaltonen's soft shadow improvement
    
	float res = 1.0;
    float t = fMinDist;
    float ph = 1e10; // big, such that y = 0 on the first iteration
    
    for( int i=0; i<SHADOW_STEPS; i++ )
    {
		float h = Scene_GetDistance( vRayOrigin + vRayDir*t ).fDist;

        // use this if you are getting artifact on the first iteration, or unroll the
        // first iteration out of the loop
        //float y = (i==0) ? 0.0 : h*h/(2.0*ph); 

        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, 10.0*d/max(0.0,t-y) );
        ph = h;
        
        t += h;
        
        if( res<0.0001 || t>fLightDist ) break;
        
    }
    return clamp( res, 0.0, 1.0 );    
}

// Function 812
float softShadow(vec3 dir, vec3 origin, float min_t, float k) {
    float res = 1.0;
    float t = min_t;
    for(int i = 0; i < RAY_STEPS; ++i) {
        vec3 obj = origin + t * dir;
        float m = sceneMap3D(obj);
        if(dot(obj - lightPos, origin - lightPos) < 0.) 
            return res;
        if(m < 0.0001) {
            return 0.0;
        }
        res = min(res, k * m / t);
        t += m;
    }
    return res;
}

// Function 813
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for(int i=0; i<50; i++)
    {
    	float h = map(ro + rd*t, false).x;
        res = min( res, 10.0*h/t + 0.02*float(i));
        t += 0.8*clamp( h, 0.01, 0.35 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 814
float shadow(in vec3 ro, in vec3 lo, in vec3 ld) {
    float dist = EPSILON;
    float totalDist = 0.0;
    for (int i = 0; i < MAX_RAY_ITERS; i++) {
        if (abs(dist) < EPSILON ||
            totalDist > MAX_RAY_DIST) {
            break;
        }
        
        dist = map(lo);
        totalDist += dist;
        lo += dist * ld;
    }
    
    if (abs(length(ro - lo)) < 0.01) {
        return 1.0;
    } else {
        return 0.25;
    }
}

// Function 815
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k )
{
    float res = 1.0;
    float dt = 0.02;
    float t = mint;
	ro += rd * 0.4;
    for( int i=0; i<32; i++ )
    {
        float h = map( ro + rd*t ).x;
        res = min( res, k*h/t );
        t += max( 0.05, dt );
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 816
float softShadowTrace(in vec3 rp, in vec3 rd, in float maxDist, in float penumbraSize, in float penumbraIntensity) {
    vec3 p = rp;
    float sh = 0.;
    float d,s = 0.;
    for (int i = 0; i < SHADOW_ITERATIONS; i++) {
        d = df(rp+rd*s);
        sh += max(0., penumbraSize-d)*float(s>penumbraSize*2.);
        s += d;
        if (d < EPSILON || s > maxDist) break;
    }
    
    if (d < EPSILON) return 0.;
    
    return max(0.,1.-sh/penumbraIntensity);
}

// Function 817
float computeLightShadow(vec3 _lightPos, vec3 _pointPos, float _scale, float t)
{
    vec3 l2p = _pointPos-_lightPos;
    float far = min(CAM_FAR, length(l2p));
    return exp(-_scale*(far-shadows(_lightPos, normalize(l2p), 0.01, far, t)));
}

// Function 818
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
    	float h = map(ro + rd*t);
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.0001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 819
float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k)
{
    float dt = (maxt - mint) / float(shadowSteps);
    float t = mint;
     //t += hash(ro.z*574854.0 + ro.y*517.0 + ro.x)*0.1;
    float res = 1.0;
    for( int i=0; i<shadowSteps; i++ )
    {
        float h = scene(ro + rd*t);
          if (h < hitThreshold) return 0.0;     // hit
        res = min(res, k*h/t);
        //t += h;
          t += dt;
    }
    return clamp(res, 0.0, 1.0);
}

// Function 820
float sphSoftShadow( in vec3 ro, in vec3 rd, in vec4 sph, in float k )
{
    vec3 oc = sph.xyz - ro;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    return (b<0.0) ? 1.0 : 1.0 - smoothstep( 0.0, 1.0, k*h/b );
}

// Function 821
float effectWave( in vec2 uv, in float frac ) {
    float base = sin(uv.y-frac*2.0*PI+1.5*PI)*0.5+0.5;
    base += sin(uv.x*10.0 - uv.y*4.0-iTime*1.2)*0.025;
    return base;
}

// Function 822
vec3 selfShadow(vec3 p, vec3 o)
{
    #ifndef SELF_SHADOWING
    	return vec3(1.0);
    #endif
    
    const int steps = 8;
    const float iSteps = 1.0 / float(steps);
    
    vec3 increment = o * iSteps;
    vec3 position = p;
    
    vec3 transmittance = vec3(1.0);
    
    for (int i = 0; i < steps; i++)
    {
        float od = calculateOD(position);
		position += increment;
        
        transmittance += od;
    }
    
    return exp2(-transmittance * scatterCoeff * iSteps);
}

// Function 823
float dShadow(vec3 wp, Light l)
{
    vec3 rd = normalize(l.pos-wp);
    /* last param for march(): make it smaller for better performance but less shadows */
    float d = march(wp+(sdist+.02)*rd, rd, 0., 5.).x;
    return clamp(mix(0., .2, d), 0., 1.);
}

// Function 824
vec4 make_another_blur( in vec2 fragCoord, float size)
{
    vec4 fragColor;
    
    float Pi = 6.28318530718; // Pi*2
    
    // GAUSSIAN BLUR SETTINGS {{{
    float Directions = 16.0; // BLUR DIRECTIONS (Default 16.0 - More is better but slower)
    float Quality = 8.0; // BLUR QUALITY (Default 4.0 - More is better but slower)
    float Size = size; // BLUR SIZE (Radius)
    // GAUSSIAN BLUR SETTINGS }}}
   
    vec2 Radius = Size/iResolution.xy;
    
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    // Pixel colour
    vec4 Color = texture(iChannel3, uv);
    
    // Blur calculations
    for( float d=0.0; d<Pi; d+=Pi/Directions)
    {
		for(float i=1.0/Quality; i<=1.0; i+=1.0/Quality)
        {
			Color += texture(iChannel3, uv + 
                                vec2(cos(d), sin(d))*Radius * i);		
        }
 }
    
    // Output to screen
    Color /= Quality * Directions;
   
    //Color = vec4(0, 1, 0, 1);
    fragColor =  Color;
    
    return fragColor;
}

// Function 825
float voxShadow(vec3 ro, vec3 rd, float end){

    float shade = 1.0;
    vec3 p = floor(ro) + .5;

	vec3 dRd = 1./abs(rd);//1./max(abs(rd), vec3(.0001));
	rd = sign(rd);
    vec3 side = dRd*(rd * (p - ro) + 0.5);
    
    vec3 mask = vec3(0);
    
    float d = 1.;
	
	for (int i = 0; i < 16; i++) {
		
        d = map(p);
        
        if (d<0. || length(p-ro)>end) break;
        
        mask = step(side, side.yzx)*(1.-step(side.zxy, side));
		side += mask*dRd;
		p += mask * rd;                
	}

    // Shadow value. If in shadow, return a dark value.
    return shade = step(0., d)*.7 + .3;
    
}

// Function 826
float doShadow( vec3 ro,  vec3 rd,  float dMax) {
    vec3 n;
    float val, dMin = dd*.1;
    vec2 res = rayGround(ro, rd, dMin, dMax, n, val);
    return res.x>dMin && res.x <= dMax ? 1. - clamp((dMax-res.x)/dMax,0.,1.) : 1.;
}

// Function 827
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

// Function 828
vec4 blur(sampler2D sp, vec2 U, vec2 scale) {
    vec4 O = vec4(0);  
    int s = samples/sLOD;
    
    for ( int i = 0; i < s*s; i++ ) {
        vec2 d = vec2(i%s, i/s)*float(sLOD) - float(samples)/2.;
        O += gaussian(d) * textureLod( sp, U + scale * d , float(LOD) );
    }
    
    return O / O.a;
}

// Function 829
vec4 blur(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
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

// Function 830
float boxSoftShadow( in vec3 ro, in vec3 rd, in mat4 txx, in vec3 rad, in float sk ) 
{
	vec3 rdd = (txx*vec4(rd,0.0)).xyz;
	vec3 roo = (txx*vec4(ro,1.0)).xyz;

    vec3 m = 1.0/rdd;
    vec3 n = m*roo;
    vec3 k = abs(m)*rad;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max( max( t1.x, t1.y ), t1.z );
	float tF = min( min( t2.x, t2.y ), t2.z );
	
    if( tN<tF && tF>0.0) return 0.0;
    
    float sh = 1.0;
    sh = segShadow( roo.xyz, rdd.xyz, rad.xyz, sh );
    sh = segShadow( roo.yzx, rdd.yzx, rad.yzx, sh );
    sh = segShadow( roo.zxy, rdd.zxy, rad.zxy, sh );
    sh = clamp(sk*sqrt(sh),0.0,1.0);
    return sh*sh*(3.0-2.0*sh);
}

// Function 831
float shadow(vec3 o,vec3 i){const float minDist=1.;float r=1.,t=.25;
// for(int j=0;j<10;j++){r=min(r,4.*df(o+i*t)/t),t+=.25;}return r;}

// Function 832
float softshadow(vec3 ro, vec3 rd, float mint, float tmax)
{
	float res = 1.0;
    float t = mint;
    for(int i=0; i<16; i++)
    {
    	float h = map(ro + rd*t);
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 833
float GrndSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  float sh, d, h;
  sh = 1.;
  d = 1.;
  for (int j = 0; j < 16; j ++) {
    p = ro + rd * d;
    h = p.y - GrndHt (p.xz);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += 0.3;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 834
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    float res = 1.0;
    for( float t=mint; t<tmax; )
    {
        float h = map(ro + rd*t);
        if( h<0.001 )
            return 0.0;
        res = min( res, 4.*h/t );
        t += h;
    }
    return res;
}

// Function 835
vec4 outlineBlur(sampler2D sampler, vec2 uv, float outlineSize)
{
 	vec4 blur = blurTexture(sampler, uv, 4.0, 7.0);
    vec4 col = 1.0 - smoothstep(0.0, 0.5, abs(blur - 0.5));
    
    return col;
}

// Function 836
float shadow(vec3 pos, vec3 lPos)
{
    lPos.xyz += (vec3(N2(pos.xy), N2(pos.yz), N2(pos.zx)) - 0.5)* 0.03; //jitters the banding away
    
    vec3 dir = lPos - pos;  // Light direction & disantce
    
    float len = length(dir);
    dir /= len;				// It's normalized now
    
    pos += dir * MIN_DST * 10.0;  // Get out of the surface
    
    vec2 dst = SDF(pos); // Get the SDF
    
    // Start casting the ray
    float t = 0.0;
    float obscurance = 1.0;
    
    while (t < len)
    {
        if (dst.x < MIN_DST) return 0.0; 
        
        obscurance = min(obscurance, (20.0 * dst.x / t)); 
        
        t += dst.x;
        pos += dst.x * dir;
        dst = SDF(pos);
    }
    
    return obscurance;     
}

// Function 837
float ObjSShadow (vec3 ro, vec3 rd, float lbDist)
{
  float sh, d, h;
  sh = 1.;
  d = 0.002;
  for (int j = 0; j < 100; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.1 * d, h));
    d += 0.01 * (1. + d);
    if (sh < 0.02 || d > lbDist) break;
  }
  return 0.3 + 0.7 * sh;
}

// Function 838
float softShadow(vec3 ro, vec3 lp, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 20; 
    
    vec3 rd = (lp-ro); // Unnormalized direction ray.

    float shade = 1.0;
    float dist = 0.05;    
    float end = max(length(rd), 0.001);
    //float stepDist = end/float(maxIterationsShad);
    
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        //dist += min( h, stepDist ); // So many options here: dist += clamp( h, 0.0005, 0.2 ), etc.
        dist += clamp(h, 0.01, .5);
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0.001 || dist > end) break; 
    }

    // I've added 0.5 to the final shade value, which lightens the shadow a bit. It's a preference thing.
    return min(max(shade, 0.) + 0.2, 1.0); 
}

// Function 839
float GrndSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  float sh, d, h;
  doSh = true;
  sh = 1.;
  d = 0.05;
  for (int j = 0; j < 30; j ++) {
    p = ro + d * rd;
    h = p.y - GrndHt (p.xz);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 840
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

// Function 841
float calcSoftShadow( in vec3 ro, in vec3 rd, float k )
{
    vec3 kk;
    float res = 1.0;
    float t = 0.01;
    for( int i=0; i<32; i++ )
    {
        float h = map(ro + rd*t, kk ).x;
        res = min( res, smoothstep(0.0,1.0,k*h/t) );
        t += clamp( h, 0.05, 0.5 );
		if( res<0.01 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 842
float shadow(vec3 l) {
    float shad=1.0;
    vec3 p=s+n*0.1+l*0.1;
    float dd=0.;
    for(int i=0;i<50; ++i) {
        float d=map(p);
        //shad=min(shad,(abs(d)-.1)*10.);
        if(d<0.1) {
            shad=0.0;
            break;
        }
        if(dd>20.) break;
        p+=l*d;
        dd+=d;
    }
    return shad;
}

// Function 843
void ToggleEffects(inout vec4 fragColor, vec2 fragCoord)
{
  // read and save effect values from buffer  
  vec3 effects =  mix(vec3(-1.0, -1.0, -1.0), readRGB(ivec2(120, 0)), step(1.0, float(iFrame)));
  effects.x*=1.0+(-2.*float(keyPress(49))); //1-key  color tint mode
  effects.y*=1.0+(-2.*float(keyPress(50))); //2-key  manual / auto camera mode
  effects.z*=1.0+(-2.*float(keyPress(51))); //3-key  chromatic aberration

  vec3 effects2 =  mix(vec3(1.0, 1.0, 1.0), readRGB(ivec2(122, 0)), step(1.0, float(iFrame)));
  effects2.y*=1.0+(-2.*float(keyPress(52))); //4-key  god Rays
  effects2.x*=1.0+(-2.*float(keyPress(53))); //5-key  lens flare

  fragColor.rgb = mix(effects, fragColor.rgb, step(1., length(fragCoord.xy-vec2(120.0, 0.0))));  
  fragColor.rgb = mix(effects2, fragColor.rgb, step(1., length(fragCoord.xy-vec2(122.0, 0.0))));
}

// Function 844
float softshadow(vec3 u,vec3 t,float a,float b,float k){
;float d=a;a=1.0;for(int i=0;i<128;++i ){
 ;float h=scene(u+t*t);if(h<.001 )return 0.
 ;a=min(a,k*h/d);t+=h;if(d>b)break;}return a;}

// Function 845
float SoftShadowRing( in vec3 origin, in vec3 direction )
{
  float res =1., t = 0.0, h=0.;
  vec3 rayPos = vec3(origin+direction*t);    

    for ( int i=0; i<10+min(0, iFrame); i++ )
    {
      h = MapPlanet(rayPos).x;
      res = min( res, 8.5*h/t );
      t += clamp( h, 0.01, 100.1);
      if ( h<0.005 ) break;
      rayPos = vec3(origin+direction*t);
    }
  return clamp( res, 0.0, 1.0 );
}

// Function 846
float calc_masking_shadow_factor(float dot_nl, float dot_nv, float rought_s)
{
    // smith correlated
	float a2 		= rought_s * 0.5;
	float lambda_l 	= calc_smith_lambda(a2, dot_nl);
	float lambda_v 	= calc_smith_lambda(a2, dot_nv);
	return 1.0f / (1.0 + lambda_l + lambda_v);
}

// Function 847
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

// Function 848
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<30; i++ )
    {
		if( t<maxt )
		{
        float h = map( ro + rd*t ).x;
        res = min( res, k*h/t );
        t += 0.2;
		}
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 849
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = VAR_ZERO; j < 50; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.4 + 0.6 * sh;
}

// Function 850
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

// Function 851
float softShadow(in vec3 pos, in vec3 ld, float mint, float k)
{
  float res = 1.0;
  float t = mint;
  for (int i=0; i<32; i++)
  {
    float ref;
    float distance = distanceEstimator(pos + ld*t, ref);
    res = min(res, k*distance/t);
    t += max(distance, mint*0.2);
  }
  return clamp(res,0.25,1.0);
}

// Function 852
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k) {
	float res = 1.0;
    float t = mint;
    for( int i=0; i<26; i++ ) {
		if( t>maxt ) break;
        float h = map( ro + rd*t ).x;
        res = min( res, k*h/t );
        t += h;
    }
    return clamp( res, 0., 1.);
}

// Function 853
float softShadow(vec3 ro, vec3 lp, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 24; 
    
    vec3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = .002;    
    float end = max(length(rd), .001);
    float stepDist = end/float(maxIterationsShad);
    
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = 0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, .2), 
        // clamp(h, .02, stepDist*2.), etc.
        dist += clamp(h, .02, .25);
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0. || dist>end) break; 
        //if (h<.001 || dist > end) break; // If you're prepared to put up with more artifacts.
    }

    // I've added 0.5 to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me.
    return min(max(shade, 0.) + .25, 1.); 
}

// Function 854
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<60; i++ )
    {
		if( t<maxt )
		{
	        float h = map( ro + rd*t ).x;
	        res = min( res, k*h/t );
	        t += 0.02;
		}
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 855
float shadow(vec3 from, vec3 increment)
{
	const float minDist = 1.0;
	
	float res = 1.0;
	float t = 1.0;
	for(int i = 0; i < SHADOW_ITERATIONS; i++) {
		float m = 0.0;
        float h = distf(from + increment * t, m);
        if(h < minDist)
            return 0.0;
		
		res = min(res, SHADOW_SMOOTHNESS * h / t);
        t += SHADOW_STEP;
    }
    return res;
}

// Function 856
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

// Function 857
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for(int i = 0; i < 32; i++)
    {
		float h = scene_dist(ro + rd*t);
        res = min(res, 2.0 * h / t);
        t += clamp(h, 0.02, 0.10);
        if(h < 0.001 || t>tmax ) 
            break;
    }
    return clamp(res, 0.0, 1.0);
}

// Function 858
bool intersectShadow( in vec3 ro, in vec3 rd, in float dist ) {
    lowp float t;
	
	t = iSphere( ro, rd, vec4( 1.5,0.5 + bounce, 2.7,1.0) );  if( t>eps && t<dist ) { return true; }
    t = iSphere( ro, rd, vec4( 4.0,0.5 + bounce2, 4.0,1.0) );  if( t>eps && t<dist ) { return true; }

    return false; // optimisation: planes don't cast shadows in this scene
}

// Function 859
float getBlurSize(float depth, float focusPoint, float focusScale) {
    float coc = clamp((1.0 / focusPoint - 1.0 / depth)*focusScale, -1.0, 1.0);
    return abs(coc) * MAX_BLUR_SIZE;
}

// Function 860
void randExplusionEffect(float t, vec2 xy){
    float ti = ceil(iTime / TIME_PHASE2);
	vec2 seed = vec2(ti, 100.0 / (ti + 1.0));
    int i;
    float r;
    float degree;
    for(i = 0; i < SQUARE_COUNT; ++i){
        r = rand(seed.x, seed.y) * EXPLOSION_RADIUS;
        degree = rand2(seed.x, seed.y) * PI * 2.0;
        
        square[i].x = explosionX + r * cos(degree);
        square[i].y = explosionY + r * sin(degree);
        square[i].z = 
            rand(square[i].x, square[i].y) * SQUARE_MAX_HALF_WIDTH * (
            1.0 - t / (TIME_PHASE2 - TIME_PHASE1));
        
        seed.x = iResolution.x * rand(seed.x + ti, seed.y);
        seed.y = iResolution.y * rand2(seed.y + ti, seed.x);
    }
}

// Function 861
float shadowSoft( vec3 ro, vec3 rd, float mint, float maxt, float k )
{
	float t = mint;
	float res = 1.0;
    for ( int i = 0; i < 64; ++i )
    {
        vec2 h = distance_to_obj( ro + rd * t );
        if ( h.x < 0.001 )
            return 0.1;
		
		res = min( res, k * h.x / t );
        t += h.x;
		
		if ( t > maxt )
			break;
    }
    return res;
}

// Function 862
float shadow( in vec3 start, in vec3 ldir, in float md, in float p )
{    
	float t = EPSILON*4.0;
	float res = 1.0;
    for ( int i = 0; i < S_STEPS; ++i )
    {        
        float d = sDist( start + ldir * t );
        if ( d < EPSILON )
            return 0.0;
		
		res = min( res, p * d / t );
        t += d*.25;
		
		if ( t > md)
			break;
    }
    return res;
}

// Function 863
bool traceSceneShadow(vec3 ro, vec3 rd)
{
    vec3 p = vec3(-.5, -.9, -.5), q = vec3(.5, -.5, .5);
    vec2 b = box(ro, rd, p,q);

    if(b.x > 0. && b.x < b.y)
    {
        return false;
    }

    float a = 3.9;
    mat2 m = mat2(cos(a), sin(a), -sin(a), cos(a));   

    ro.xz *= m;
    rd.xz *= m;

    ro.xy *= m;
    rd.xy *= m;

    ro.yz *= m;
    rd.yz *= m;

    p = vec3(-.26), q = vec3(.26);
    b = box(ro, rd, p, q);

    if(b.x > 0. && b.x < b.y)
    {
        return false;
    }

    return true;
}

// Function 864
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int iter = 24; 
    
    ro += n*.0015; // Bumping the shadow off the hit point.
    
    vec3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float t = 0.; 
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;
    
    //rd = normalize(rd + (hash33R(ro + n) - .5)*.03);
    

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = 0; i<iter; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        t += clamp(d, .01, .25); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}

// Function 865
bool intersectShadow( in vec3 ro, in vec3 rd, in float dist ) {
    lowp float t;
	
	t = iSphere( ro, rd, vec4( 1.5 + sway(),0.5 + bounce(), 2.7,1.0) );  if( t>eps && t<dist ) { return true; }
    t = iSphere( ro, rd, vec4( 4.0,0.5 + bounce2(), 4.0,1.0) );  if( t>eps && t<dist ) { return true; }

    return false; // optimisation: planes don't cast shadows in this scene
}

// Function 866
int TraceShadow(vec3 ro, vec3 rd)
{
	int hit;
	vec3 pos;
	float dist2 = DistanceFields(ro, rd, hit, pos);
	return hit;
}

// Function 867
float castShadowRay(vec3 ro,vec3 rd
){for( int i=0; i<NUMBOXES; i++ //es100 safe
 ){mat4 ma; vec3 si; getLocation(i, ma, si)
  ;if(iBox(ro,rd,ma,si)>0.)return 0.;}
 ;return 1.;}

// Function 868
void packShadow(out vec4 packed, in float depth)
{
    packed = vec4(map(depth, MIN_DIST_SHADOW, MAX_DIST_SHADOW, 0.0, 1.0));
}

// Function 869
float shadow(in vec3 position)
{
    float lightDist;
	vec3 lightPoint = worldPointToLightPoint(position, lightDist);
    
    float shadowDist;
    unpackShadow(texture(SHADOW_BUFFER, lightPoint.xy), shadowDist);
    
    // Compare real distance to light with distance captured by the light "camera".
    // If they differ, this means that something is on the way.
    // This means one thing.
    // Shadow.
    float shadow = shadowDist - lightDist;
    shadow = step(0.0, shadow);

    if (clamp(lightPoint.x, 0.0, 1.0) != lightPoint.x
     || clamp(lightPoint.y, 0.0, 1.0) != lightPoint.y
     || shadowDist >= MAX_DIST_SHADOW - 1.0) // _why_
        shadow = 1.0;
    
    return shadow;
}

// Function 870
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow 
    // test machine, anyway.
    const int maxIterationsShad = 24; 
    
    ro += n*.0015;
    vec3 rd = lp - ro; // Unnormalized direction ray.
    

    float shade = 1.;
    float t = 0.;//.0015; // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = min(iFrame, 0); i<maxIterationsShad; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        t += clamp(d, .01, .25); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}

// Function 871
float evaluateShadows(vec3 origin, vec3 toLight)
{
    float res = 1.0;
    
    for( float t = .05; t < 2.0; )
    {
        float h = sdf_simple(origin + toLight*t);
        if( h < 0.001 )
            return 0.0;
        
        res = min( res, 12.0*h/t );
        t += h * .5 + .001;
    }
    return res;
}

// Function 872
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

// Function 873
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int iter = 24; 
    
    ro += n*.0015; // Bumping the shadow off the hit point.
    
    vec3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float t = 0.; 
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;
    
    //rd = normalize(rd + (hash33R(ro + n) - .5)*.03);
    

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = 0; i<iter; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        t += clamp(d, .01, .25); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        //if (d<0. || t>end) break; 
        // Bounding plane optimization, specific to this example. Thanks to IQ. 
        if (d<0. || t>end || (ro.z + rd.z*t)<-0.11) break;
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}

// Function 874
bool shadowDetect(Ray r, Sphere s)
{        
    vec3 oc = r.origin - s.center;
    float a = dot(r.direction, r.direction);
    float b = 2.0 * dot(oc, r.direction);
    float c = dot(oc, oc) - s.radius * s.radius;
    float det = b*b - 4.0*a*c;
    if (det < 0.0)
    	return false;
    
    
    float t = (-b - sqrt(det)) / (2.0 * a);
    if (r.mint <= t && t < r.maxt) 
        return true;

    return false;
}

// Function 875
vec3 PostEffects(vec3 rgb, vec2 uv)
{
	//#define CONTRAST 1.1
	//#define SATURATION 1.12
	//#define BRIGHTNESS 1.3
	//rgb = pow(abs(rgb), vec3(0.45));
	//rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);
	rgb = (1.0 - exp(-rgb * 6.0)) * 1.0024;
	//rgb = clamp(rgb+hash12(fragCoord.xy*rgb.r)*0.1, 0.0, 1.0);
	return rgb;
}

// Function 876
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

// Function 877
void hitShadow(
    in vec3 startPos, in vec3 nvRayDir,
    in vec3 q00, in vec3 q10, in vec3 q11, in vec3 q01,
    out float lightPercent
){
    lightPercent = 1.0;
    float travel = 0.0;
    vec3 curPos = startPos;

    for (int k = 0; k < RAY_STEPS_SHADOW; k++) {
        float sdCur = sdQuad(q00,q10,q11,q01, BOUNDARY_RADIUS, curPos);

        float curLightPercent = abs(sdCur)/(0.02*travel);
        lightPercent = min(lightPercent, curLightPercent);

        if (sdCur < MIN_DIST) {
            lightPercent = 0.0;
            break;
        }

        curPos += sdCur * nvRayDir;
        travel += sdCur;
        if (travel > MAX_DIST) {
            break;
        }
    }
}

// Function 878
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

// Function 879
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<30; i++ )
    {
		if( t>maxt ) break;

		float h = map( ro + rd*t ).x;
        res = min( res, k*h/t );
        t += 0.02;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 880
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

// Function 881
float Effect3(vec2 coords, float numSegments, float r, float cutoffMax, float cutoffMin, float thickness)
{
    float dist = length(coords);
    float s = abs(sin(r * numSegments * 0.5));
    
    if (dist <= cutoffMax && dist >= cutoffMin)
    {
        if (s < tan(tan(dist)) * thickness) 
        {
            return 1.0;
        }
    }
    return 0.0;
}

// Function 882
float softshadow(in vec3 ro, in vec3 rd) {
#ifdef FAST
	return 1.;
#else

	float res = 1.0, h, t = .02;
    for( int i=0; i<16; i++ ) {
	//	if (t < maxt) {
		h = map0(ro + rd*t, true, true);
		res = min( res, 1.*h/t );
		t += 0.3;
	//	}
    }
    return clamp(res, 0., 1.);
#endif	
}

// Function 883
float SoftShadow( in vec3 origin, in vec3 direction )
{
  float res = 1.0, t = 0.0, h;
  for ( int i=0; i<NO_UNROLL(16); i++ )
  {
    h = Map(origin+direction*t);
    res = min( res, 7.5*h/t );
    t += clamp( h, 0.02, 0.15);
    if ( h<0.002 ) break;
  }
  return clamp( res, 0.0, 1.0 );
}

