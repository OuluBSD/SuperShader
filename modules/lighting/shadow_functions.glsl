// Reusable Shadow Lighting Functions
// Automatically extracted from lighting-related shaders

// Function 1
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

// Function 2
float dir_shadow(in vec3 p, in vec3 l)
{
    float t = 0.15;
    float t_max = 20.0;
    
    float res = 1.0;
    for (int i = 0; i < 256; ++i)
    {
        if (t > t_max) break;
        
        int ignored;
        float d = sdf(p + t*l, ignored);
        if (d < 0.01)
        {
            return 0.0;
        }
        t += d;
        res = min(res, 8.0 * d / t);
    }
    
    return res;
}

// Function 3
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

// Function 4
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

// Function 5
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

// Function 6
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

// Function 7
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

// Function 8
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

// Function 9
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float maxt, in float k )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<32; i++ )
    {
        float h = map( ro + rd*t ).x;
        res = min( res, k*h/t );
        t += h;
		if( t>maxt ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// Function 10
bool TraceShadow(vec3 ro, vec3 rd)
{
	bool hit;
	vec3 pos;
	vec3 hitNormal;
	int mat;
	float dist2 = VoxelTrace(ro+rd*0.6, rd, hit, hitNormal, pos, mat);
	return hit;
}

// Function 11
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

// Function 12
float evaluateBeckmannGeometryShadowingSingleSide(float nlDot, float roughness)
{
    // http://graphicrants.blogspot.jp/2013/08/specular-brdf-reference.html
    float lct = .5 / (roughness * sqrt(1. - nlDot * nlDot) + 0.00001);
    float lc = lct * nlDot;
    float a = 3.353 * lc + 2.181 * lc * lc; // not typo
    float b = 1. + 2.276 * lc + 2.577 * lc * lc;
    return a / b;
}

// Function 13
float ObjSShadow (vec3 ro, vec3 rd)
{
  float d, h, sh;
  sh = 1.;
  d = 0.02;
  for (int i = 0; i < 40; i++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, 20. * h / d);
    d += 0.02;
    if (h < 0.001) break;
  }
  return clamp (sh, 0., 1.);
}

// Function 14
float genShadow(vec3 ro, vec3 rd)
{
    // = distance filed result
    float h = 0.0;

    // current ray position
    float c = 0.001;

    // most nearest distance of scene objects
    float r = 1.0;

    // shadow coef
    float shadowCoef = 0.5;

    // ray marching for shadow
    for (float t = 0.0; t < 50.0; t++)
    {
        h = distFunc(ro + rd * c);

        if (h < EPS)
        {
            return shadowCoef;
        }

        // 現時点の距離関数の結果と係数を掛けたものを
        // レイの現時点での位置で割ったものを利用する
        // 計算結果のうち、もっとも小さいものを採用する
        r = min(r, h * softShadow / c);

        c += h;
    }

    return 1.0 - shadowCoef + (r * shadowCoef);
}

// Function 15
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

// Function 16
float softShadow(vec3 dir, vec3 origin, float min_t, float k) {
    float res = 1.0;
    float t = min_t;
    for(int i = 0; i < RAY_STEPS; ++i) {
        float m = shadowMap3D(origin + t * dir);
        if(m < 0.0001) {
            return 0.0;
        }
        res = min(res, k * m / t);
        t += m;
    }
    return res;
}

// Function 17
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

// Function 18
float SoftShadow( in vec3 landPoint, in vec3 lightVector, float mint, float maxt, float iterations ){
    float penumbraFactor=1.0;vec3 sphereNormal;float t=mint;
    for(int s=0;s<20;++s){if(t > maxt) break;
        float nextDist = min(
            BuildingsDistance(landPoint + lightVector * t )
            , RedDistance(landPoint + lightVector * t ));
        if( nextDist < 0.001 ){return 0.0;}
        penumbraFactor = min( penumbraFactor, iterations * nextDist / t );
        t += nextDist;}return penumbraFactor;}

// Function 19
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

// Function 20
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

// Function 21
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

// Function 22
float orbShadow(float rad, vec3 coord)
{
	return 1.0 - smoothstep(0.4, 1.1, distance(coord.xy, frag_coord) / rad) *
		mix(1.0,0.99,orb(rad,coord));
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
vec2 ts_shadow_sample_ao( TrnSampler ts, sampler2D ch, vec3 x )
{
#if WITH_TRN_SHADOW
    vec4 lookup = ts_shadow_lookup( ts, ch, x );
    return vec2( ts_shadow_eval( lookup.xy, length(x) ), lookup.z );
#else
    return vec2(1);
#endif
}

// Function 25
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

// Function 26
float point_shadow(in vec3 p, in vec3 light_p)
{
    vec3 l = normalize(light_p - p);
        
    float t = 0.15;
    float t_max = distance(light_p, p);
    
    float res = 1.0;
    for (int i = 0; i < 256; ++i)
    {
        if (t > t_max) break;
        
        int ignored;
        float d = sdf(p + t*l, ignored);
        if (d < 0.01)
        {
            return 0.0;
        }
        t += d;
        res = min(res, 64.0 * d / t);
    }
    
    return res;
}

// Function 27
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    // bounding volume
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

// Function 28
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.02 * fusLen;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += 0.03 * fusLen;
    if (sh < 0.05) break;
  }
  return 0.3 + 0.7 * sh;
}

// Function 29
bool shadowHit(const in Ray ray) 
{
    float _t;
    vec3 _p, _n;
    for (int i = 0; i < SPHERES_NB; i++) {
        if (intersectsSphere(ray, spheres[i], _t, _p, _n)) return true;
    }
    return false;
}

// Function 30
float ShadowMaskRows(vec2 uv)
{
    // Stagger rows
    uv.x *= 0.5;
    uv.x -= round(uv.x);
    if(uv.x < 0.0)
        uv.y += 0.5;
    
    return Grille(uv.y, -SHADOWMASK_HORIZGAPWIDTH, SHADOWMASK_HORIZARDNESS);
}

// Function 31
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

// Function 32
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

// Function 33
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

// Function 34
vec3 ShadowMaskRGBCols(float x)
{
	return vec3
    (
        ShadowMaskSingleCol(x + SHADOWMASK_RCOL_OFFSET), 
        ShadowMaskSingleCol(x + SHADOWMASK_GCOL_OFFSET), 
        ShadowMaskSingleCol(x + SHADOWMASK_BCOL_OFFSET)
    );    
}

// Function 35
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

// Function 36
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

// Function 37
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

// Function 38
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

// Function 39
void doShadowColor(in ray primaryRay, inout vec4 col) {
	vec4 returnColor = vec4(0.0);
	vec2 shadowUV;
	vec2 shadowT;
	int shadowId;
	float shadowCheckDelta = light.w;
	ray shadowRay;
	shadowRay.lightColor = primaryRay.lightColor;
	shadowRay.transmittance = primaryRay.transmittance;
	vec3 pos = primaryRay.origin + primaryRay.rayLength*primaryRay.direction;
	shadowRay.origin = pos - 0.001*primaryRay.direction;
	for (int i = 0; i < 7; i++) {
		//soft shadows
		if (i == 0) {
			vec4 tempLight = light;
			tempLight.x += shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else if (i == 1) {
			vec4 tempLight = light;
			tempLight.x -= shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else if (i == 2) {
			vec4 tempLight = light;
			tempLight.y += shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else if (i == 3) {
			vec4 tempLight = light;
			tempLight.y -= shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else if (i == 4) {
			vec4 tempLight = light;
			tempLight.z += shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else if (i == 5) {
			vec4 tempLight = light;
			tempLight.z -= shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else
			shadowRay.direction = normalize(light.xyz-pos);
		shadowId = intersect(shadowRay, shadowT, shadowUV);
		vec3 shadowHit = shadowRay.origin + shadowT.x * shadowRay.direction;
		
		//if we have a non-negative id, we've hit something
		if (shadowId >= 0 && primaryRay.lastHitObject >= 0) {
			vec4 tempColor;
			if (light.y > 0.0) {
				if (primaryRay.lastHitObject != 1
					&& shadowId == 1
					&& length(light.xyz-shadowRay.origin) > length(shadowHit-shadowRay.origin)) {
					//shade objects that are shadowed by the window
					vec3 nor = sceneWindow.xyz;
					shadowRay.lightColor = doLighting(primaryRay.origin, shadowHit, nor, light.xyz);
					shadowRay.lightColor *= 1.0 - vec4(Voronoi(shadowUV),1.0);
					shadowRay.transmittance = primaryRay.transmittance * glassTransmission;
					tempColor = mix(shadowRay.lightColor, col, shadowRay.transmittance);
				}
				else if (primaryRay.lastHitObject == 1) {
					//shade the back side of the window
					vec3 nor = -sceneWindow.xyz;
					shadowRay.lightColor = doLighting(primaryRay.origin, shadowHit, nor, light.xyz);
					shadowRay.lightColor *= vec4(Voronoi(shadowUV),1.0);
					shadowRay.transmittance = primaryRay.transmittance * glassTransmission;
					tempColor = mix(shadowRay.lightColor, col, shadowRay.transmittance);
				}
				if (primaryRay.lastHitObject != 1 && shadowId != 1) {
					//shadows for everything else in the scene
					shadowRay.lightColor = shadowRay.lightColor;
					shadowRay.transmittance = 0.5*primaryRay.transmittance;
					tempColor = mix(shadowRay.lightColor, col, 1.0-shadowRay.transmittance);
				}
			}
			else if (primaryRay.lastHitObject >= 0) {
				//before "sunrise"
				shadowRay.lightColor = shadowRay.lightColor;
				shadowRay.transmittance = 0.5*primaryRay.transmittance;
				tempColor = mix(shadowRay.lightColor, col, shadowRay.transmittance);
			}
			returnColor += tempColor;
		}
	}
	//if we use a number slightly higher than our iteration count,
	//then we get dark, but not black, shadows.  This also washes
	//out the color of the color of the glass, so it's kind of a
	//trade-off.
	col -= returnColor*(1.0/8.5);
}

// Function 40
float shadowMap3D(vec3 pos)
{
    float t = sphere(pos, 4.0, vec3(0.0, 0.0, 0.0));
    t = min(t, sphere(pos, 2.0, greenSpherePos));
    t = min(t, box(pos + vec3(0.0, 3.0, 0.0), vec3(50.0, 1.0, 50.0)));
    return t;
}

// Function 41
float shadow(vec3 dir, vec3 origin, float min_t) {
    return softShadow(dir, origin, min_t, 6.0);
}

// Function 42
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = shipDE( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 43
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

// Function 44
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

// Function 45
float shadow(vec3 p, vec3 n, vec3 lPos, MatSpace ps)
{
    return shadow(p + n * MIN_DST * 40.0, lPos, ps);
}

// Function 46
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

// Function 47
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

// Function 48
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

// Function 49
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

// Function 50
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

// Function 51
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

// Function 52
float softShadows( in vec3 ro, in vec3 rd )
{

    float res = 1.0;
    for( float t = 0.1; t < 8.0; ++t )
    {
    
        float h = map( ro + rd * t ).x;
        if( h < EPS ) return 0.0;
        res = max( res, 8.0 * h / t );
        
    }
    
    return res;
    
}

// Function 53
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

// Function 54
float beamShadow(float x)
{
	x = clamp(x + 0.1, 0.0, 0.5);
	return smoothstep(0.7, 1.0, pow(1.0 - 2.0 * abs(x - 0.5), 0.1));
}

// Function 55
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

// Function 56
float softshadow( in vec3 ro, in vec3 rd )
{
    float res = 1.0;
    for( float t=0.02; t < 2.; )
    {
        float h = map(ro + rd*t).x;
        if( h<0.001 ) return 0.0;
        res = min( res, 8.0*h/t );
        t += h;
    }
    return res;
}

// Function 57
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

// Function 58
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

// Function 59
float shadow (in vec3 ro, in vec3 rd)
{
    float result = 1.;
    float t = .1;
    for (int i = 0; i < MAX_ITER; i++) {
        float h = scene (ro + t * rd).d;
        if (h < .00001) return .0;
        result = min (result, 8. * h/t);
        t += h;
    }

    return result;
}

// Function 60
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

// Function 61
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

// Function 62
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

// Function 63
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

// Function 64
float traceShadow( in vec3 from, in vec3 dir, in vec3 normal, const float sinTheta ) {
    if (dot(dir, normal) < 0.0) return 0.0;
    float minAlpha = 1.0;
    float totdist = 0.0;
    #define SHADOW_STEPS 20
    for (int i = Z ; i < SHADOW_STEPS ; i++) {
        vec3 p = from+dir*totdist;
        if (dot(p, p) > 6.0) return minAlpha;
        float dist = de(p, false, dummy);
        float rad = dist / (totdist*sinTheta);
        float alpha = rad * 0.5 + 0.5;
        if (alpha <= 0.0) {
            return 0.0;
        } else if (alpha < minAlpha) {
            minAlpha = alpha;
        }
        totdist += max(0.01, dist*0.8);
    }
    return minAlpha;
}

// Function 65
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

// Function 66
float shadow (in vec3 p, in vec3 lPos) {
    float lDist = distance (p, lPos);
    vec3 lDir = normalize (lPos - p);
    float dist = march (p, lDir);
    return dist < lDist ? .1 : 1.;
}

// Function 67
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

// Function 68
float GrndSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  float sh, d, h;
  sh = 1.;
  d = 2.;
  for (int i = 0; i < 10; i++) {
    p = ro + rd * d;
    h = p.y - GrndHt (p.xz);
    sh = min (sh, 20. * h / d);
    d += 4.;
    if (h < 0.01) break;
  }
  return clamp (sh, 0., 1.);
}

// Function 69
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

// Function 70
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

// Function 71
float shadow (in vec3 p, in vec3 lPos) {
    float distanceToLight = distance (p, lPos);
    vec3 n = normal (p, EPSILON);
    float distanceToObject = raymarch (p + .01*n, normalize (lPos - p));
    bool isShadowed = distanceToObject < distanceToLight;
    return isShadowed ? .1 : 1.;
}

// Function 72
void packShadow(out vec4 packed, in float depth)
{
    packed = vec4(map(depth, MIN_DIST_SHADOW, MAX_DIST_SHADOW, 0.0, 1.0));
}

// Function 73
float ShadowMaskSingleCol(float x)
{
    return Grille(x, -SHADOWMASK_VERTGAPWIDTH, SHADOWMASK_VERTHARDNESS);
}

// Function 74
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

// Function 75
float atm_planet_shadow( float coschi, float cosbeta )
{
    return clamp( SCN_RAYCAST_SHADOW_UMBRA * ( coschi + cosbeta ) + .5, 0., 1. );
}

// Function 76
vec2 softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    vec2 res = vec2(1.,1.);
    float t = mint;
    for( int i=0; i<32; i++ )
    {
        SDObject hit=mapScene( ro + rd*t ,-1,false);
        float h = hit.d;
        #ifdef ENABLE_TRANSPARENCY
        if(hit.mat.type>0 && hit.mat.trans>0.)
        {
            res.y=min(res.y,1.-hit.mat.trans);
        }
        #endif
        res.x = min( res.x, (32.0*h/t) );
        t += clamp( h, 0.00001, 0.99999 );
        if( h<(0.00001) || t>tmax ) break;
    }
    return clamp( res,1.2-res.y, 1.0 );

}

// Function 77
float GrndSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 1.;
  for (int j = 0; j < 16; j ++) {
    h = GrndDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += 1.;
    if (sh < 0.05) break;
  }
  return sh;
}

// Function 78
bool intersectShadow( in vec3 ro, in vec3 rd, in float dist ) {
    lowp float t;
	
	t = iSphere( ro, rd, vec4( 1.5,0.5 + bounce, 2.7,1.0) );  if( t>eps && t<dist ) { return true; }
    t = iSphere( ro, rd, vec4( 4.0,0.5 + bounce2, 4.0,1.0) );  if( t>eps && t<dist ) { return true; }

    return false; // optimisation: planes don't cast shadows in this scene
}

// Function 79
float lightPointDiffuseSoftShadow(vec3 pos, vec3 lightPos, vec3 normal) {
	vec3 lightDir = normalize(lightPos - pos);
	float lightDist = length(lightPos - pos);
	float color = max(dot(normal, lightDir), 0.0) / (lightDist * lightDist);
	if (color > 0.00) color *= castSoftShadowRay(pos, lightPos);
	return max(0.0, color);
}

// Function 80
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

// Function 81
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

// Function 82
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

// Function 83
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = scene( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );

}

// Function 84
float areaShadow( in vec3 P )
{
  float s = 1.0;
  for( int i=0; i<SPH; i++ )
    s = min( s, sphAreaShadow(P, L, sphere[i] ) );
  return s;           
}

// Function 85
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

// Function 86
float shadow( in vec3 ro, in vec3 rd )
{
	float res = 1.0;
	for( int i=0; i<NUMSPHEREES; i++ )
	{
		float id = float(i);
	    float t = sSphere( ro, rd, sphere[i] ); 
		res = min( t, res );
	}
    return res;					  
}

// Function 87
float shadowDistanceField(vec3 p) {
	float loxodrome = sdLoxodrome(p - vec3(0.0, 0.0, 0.0), 2.0, 4.0, 0.075);
	float holder = 9001.0;
	#ifndef LOXODROME_ONLY
	holder = sdCylinder(p.xzy - vec3(0.0, 0.0, -1.125), vec2(0.5, 0.25));
	#endif
	return min(holder, loxodrome);
}

// Function 88
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

// Function 89
float GrndSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  float sh, d, h;
  sh = 1.;
  d = 0.4;
  for (int i = 0; i < 20; i ++) {
    p = ro + rd * d;
    h = p.y - GrndHt (p.xz);
    sh = min (sh, 20. * h / d);
    d += 0.4;
    if (h < 0.001) break;
  }
  return clamp (sh, 0., 1.);
}

// Function 90
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

// Function 91
float evaluateBeckmannGeometryShadowing(float nlDot, float nvDot, float roughness)
{
    // http://graphicrants.blogspot.jp/2013/08/specular-brdf-reference.html
    float lct = .5 / (roughness * sqrt(1. - nlDot * nlDot) + 0.00001);
    float vct = .5 / (roughness * sqrt(1. - nvDot * nvDot) + 0.00001);
    float lc = lct * nlDot, vc = vct * nvDot;
    float a = 3.353 * lc + 2.181 * lc * lc; // not typo
    a *= 3.353 * vct + 2.181 * vct * vc;
    float b = 1. + 2.276 * lc + 2.577 * lc * lc;
    b *= 1. + 2.276 * vc + 2.577 * vc * vc;
    return a / b;
}

// Function 92
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

// Function 93
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

// Function 94
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

// Function 95
float groundShadow(vec2 p)
{
	vec2 fp = floor(p);
	vec2 pf = smoothstep(0.0, 1.0, fract(p));
	
	return mix( mix(groundSolidity(fp), groundSolidity(fp + ON), pf.x), 
			   mix(groundSolidity(fp + ON.yx), groundSolidity(fp + ON.xx), pf.x), pf.y);
}

// Function 96
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

// Function 97
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

// Function 98
float ts_shadow_sample( TrnSampler ts, sampler2D ch, vec3 x )
{
#if WITH_TRN_SHADOW
    return ts_shadow_eval( ts_shadow_lookup( ts, ch, x ).xy, length(x) );
#else
    return 1.;
#endif
}

// Function 99
float octreeshadow(vec3 ro, vec3 rd, float lightdist) {
    vec3 dummy1;
    float dummy2;
    float proxim;
    float len = octreeray(ro,rd,lightdist,dummy1,dummy1,dummy2,proxim).x;
    if (len >= lightdist) {
        return proxim;
    } else {
        return 0.0;
    }
}

// Function 100
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

// Function 101
float softshadow( in vec3 ro, in vec3 rd, float k )
{
    float res = 1.0;
    float t = 0.001;
	float h = 1.0;
	vec4 kk;
    for( int i=0; i<25; i++ )
    {
        h = map(ro + rd*t,kk);
        res = min( res, smoothstep(0.0,1.0,k*h/t) );
        if( res<0.001 ) break;
		t += clamp( h, 0.02, 2.0 );
    }
    return clamp(res,0.0,1.0);
}

// Function 102
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.02;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + d * rd);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.5 + 0.5 * sh;
}

// Function 103
float Shadow(vec3 p,vec3 d)
{
    float dist = Raymarch(p,d).Dist;
    if(dist < kMaxD - kDelD)
    {
        return 1.0;
    }
    return 0.0;
}

// Function 104
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

// Function 105
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

// Function 106
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

// Function 107
float softshadow(in vec3 ro, in vec3 rd){
    float res = 1.0, t = 0.15; // t=0.15 -> no banding on my stock x.org drivers
    for(int s = 0; s < 16; ++s){
        float h = scene(ro + rd*t);
        if(h < 0.01) return 0.0;
        res = min( res, 2.0*h/t );
        t += h*0.9;
    }
    return res;
}

// Function 108
float shadow(vec3 ro, vec3 rd,float n)
{
    float res = .5;
    float f=0.1;
    for(int i = 0; i <200; i++)
    {
        float h = map(ro+f*rd).x;
            if(h<0.01)
                return 0.;
               	if(f > n)
           break;
        res = min(res,8.*h/f);
            f+=.25*h;
    }
    return res;
}

// Function 109
float shadowMarch(vec3 ro,vec3 l//raypositionOnSurface, LightPosition
){int dum=0;l-=ro//l is now lightDirection
 ;vec3 rd=normalize(l)
 ;float tmax=length(l)
 ;float r = 1.,t=1.//tmax/shadowIters
 ;for(float i=0.;i<shadowIters;i++
 ){if(t>=tmax)break
  ;float d=distanceToClosest(ro+rd*t,dum)
  ;if(d<shadowEps) return 0.
  ;r=min(r, shadowBokeh*d)
  ;t+=d
  ;}return r;}

// Function 110
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

// Function 111
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

// Function 112
float ObjSShadow (vec3 ro, vec3 rd)
{
  vec3 p;
  vec2 gIdP;
  float sh, d, h;
  sh = 1.;
  gIdP = vec2 (-999.);
  d = 0.03;
  for (int j = VAR_ZERO; j < 24; j ++) {
    p = ro + d * rd;
    gId = PixToHex (p.xz / hgSize);
    if (gId.x != gIdP.x || gId.y != gIdP.y) {
      gIdP = gId;
      SetGrdConf ();
    }
    h = ObjDf (p);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += max (0.03, 2. * h);
    if (h < 0.005) break;
  }
  return 0.7 + 0.3 * sh;
}

// Function 113
float softshadow( in vec3 ro, in vec3 rd, float mint, float k )
{
    float res = 1.0;
    float t = mint;
    for( int i=0; i<32; i++ )
    {
        float h = map(ro + rd*t).x;
		h = max( h, 0.0 );
        res = min( res, k*h/t );
        t += clamp( h, 0.001, 0.1 );
		if( res<0.01 || t>6.0 ) break;
    }
    return clamp(res,0.0,1.0);
}

// Function 114
float calcHardshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    for( float t=mint; t < tmax; )
    {
        float h = map(ro + rd*t).x;
        if( h<0.001 )
            return 0.0;
        t += h;
    }
    return 1.0;
}

// Function 115
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

// Function 116
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

// Function 117
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

// Function 118
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

// Function 119
float shadow(vec3 px)
{
    // check whether the shadow maps z value is higher than the current one
    // if it's not, the point lies in shadow
     vec4 lookup = texture(iChannel0,px.xy);
     return float(lookup.x  > px.z - 0.002); 
}

// Function 120
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

// Function 121
bool intersectShadow( in vec3 ro, in vec3 rd, in float dist ) {
    float t;
	
	t = iSphere( ro, rd, vec4( 1.5,1.0, 2.7,1.0) );  if( t>eps && t<dist ) { return true; }
    t = iSphere( ro, rd, vec4( 4.0,1.0, 4.0,1.0) );  if( t>eps && t<dist ) { return true; }

    return false; // optimisation: planes don't cast shadows in this scene
}

// Function 122
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

// Function 123
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

// Function 124
float ts_shadow_eval( vec2 lookup, float test )
    { return lookup.x != lookup.y ? parabolstep( lookup.x, lookup.y, test ) : 1.; }

// Function 125
vec3 ShadowMask(vec2 uv)
{
    return ShadowMaskRGBCols(uv.x) * ShadowMaskRows(uv);
}

// Function 126
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

// Function 127
float shadowhit( const vec3 ro, const vec3 rd, const float dist) {
    vec3 normal;
    float d = traceSphereGrid( ro, rd, vec2(.3, dist), normal, 4).y;
    d = min(d, iCylinder(ro, rd, vec2(.3, dist), normal, vec3(0,.2,0), 1.5, false));
    return d < dist-0.001 ? 0. : 1.;
}

// Function 128
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

// Function 129
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

// Function 130
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

// Function 131
bool intersectShadow( in vec3 ro, in vec3 rd, in float dist ) {
    float t;
	
	t = iSphere( ro, rd, movingSphere            );  if( t>eps && t<dist ) { return true; }
    t = iSphere( ro, rd, vec4( 4.0,1.0, 4.0,1.0) );  if( t>eps && t<dist ) { return true; }
#ifdef FULLBOX    
    t = iPlane( ro, rd, vec4( 0.0,-1.0, 0.0,5.49) ); if( t>eps && t<dist && ro.z+rd.z*t < 5.5 ) { return true; }
#endif
    return false; // optimisation: other planes don't cast shadows in this scene
}

// Function 132
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

// Function 133
vec3 calculateShadow( vec3 currentPixelColor, vec3 lightDir, HitInfo hitInfo )
{
    Ray shadowRay;
    shadowRay.direction = normalize( lightDir );
    shadowRay.origin = hitInfo.position + (shadowRay.direction * Epsilon);
    HitInfo shadowHitInfo = intersect( shadowRay );
    currentPixelColor *= clamp( shadowHitInfo.shadowCasterIntensity, 0.00, 1.0 );
    
    return currentPixelColor;
}

// Function 134
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

// Function 135
float GrndSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 30; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += 0.15;
    if (sh < 0.05) break;
  }
  return 0.6 + 0.4 * sh;
}

// Function 136
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

// Function 137
float GrndSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.1;
  for (int j = 0; j < 16; j ++) {
    h = GrndDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += max (0.2, 0.1 * d);
    if (sh < 0.05) break;
  }
  return 0.3 + 0.7 * sh;
}

// Function 138
void unpackShadow(in vec4 packed, out float depth)
{
    depth = map(packed.r, 0.0, 1.0, MIN_DIST_SHADOW, MAX_DIST_SHADOW);
}

// Function 139
void shadowSoft(inout float s,inout vec3 x,inout float j,float t,vec4 m,mat4 B
){float h
 ;for(int i=0;i<64;i++
 ){h=map(x+lightDir*j).x
  ;j+=clamp(h, .032, 1.);
               	s = min(s, h/j);
             	if(j>7.|| h<.001) break;
            } 
}

// Function 140
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

// Function 141
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

// Function 142
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

// Function 143
vec3 ringShadowColor( const in vec3 ro ) {
    if( iSphere( ro, SUN_DIRECTION, vec4( 0., 0., 0., EARTH_RADIUS ) ) > 0. ) {
        return vec3(0.);
    }
    return vec3(1.);
}

// Function 144
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.05;
  for (int j = 0; j < 16; j ++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, smoothstep (0., 0.05 * d, h));
    d += 0.1;
    if (sh < 0.05) break;
  }
  return sh;
}

// Function 145
float ObjSShadow (vec3 ro, vec3 rd)
{
  float sh, d, h;
  sh = 1.;
  d = 0.02 * fusLen;
  for (int i = 0; i < 50; i++) {
    h = ObjDf (ro + rd * d);
    sh = min (sh, 20. * h / d);
    d += 0.02 * fusLen;
    if (h < 0.001) break;
  }
  return clamp (sh, 0., 1.);
}

// Function 146
float shadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    for (float t = mint; t < maxt;) {
        float h = map(ro + rd * t).w;
        if (h < 0.001)
            return res;
        t += h;
        res = min(res, k * h / t);
    }
    return clamp(res, 0.0, 1.0);
}

