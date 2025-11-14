// Reusable Normal Mapping Texturing Functions
// Automatically extracted from texturing/mapping-related shaders

// Function 1
vec3 drawPathAndNormals( in vec3 col, in vec2 p, in float e, in vec2 path[kNumPoints], in vec3 pathColor, in float normLength )
{
    vec3 d = vec3(1000.0);
    for( int i=0; i<kNumPoints; i++ )
    {
        vec2 a = path[(i-1+kNumPoints)%kNumPoints];
        vec2 b = path[(i+0+kNumPoints)%kNumPoints];
        vec2 c = path[(i+1+kNumPoints)%kNumPoints];

        vec2 n = computeTangent( a, b, c );
        
        n = normLength*normalize(vec2(n.y, -n.x ));

        d = min( d, vec3(sdSegmentSq(p,b,c), 
                         sdPointSq(p,b),
                         sdSegmentSq(p,b,b+n)) );
    }
    d = sqrt(d);

    col = mix( col, pathColor, 1.0-smoothstep(0.0,2.0*e,d.x) );
    col = mix( col, pathColor, 1.0-smoothstep(5.0*e,6.0*e,d.y) );
    col = mix( col, pathColor, 1.0-smoothstep(0.0,2.0*e,d.z) );
    
    return col;
}

// Function 2
vec3 Normal(vec3 p){vec2 e=vec2(.01,0);return normalize(vec3(
 df(p+e.xyy)-df(p-e.xyy),df(p+e.yxy)-df(p-e.yxy),df(p+e.yyx)-df(p-e.yyx)));}

// Function 3
vec3 getnormal( in vec3 p)
			{
				vec2 e = vec2(0.5773,-0.5773)*0.0001;
				vec3 nor = normalize( e.xyy*dist(p+e.xyy) + e.yyx*dist(p+e.yyx) + e.yxy*dist(p+e.yxy ) + e.xxx*dist(p+e.xxx));
				nor = normalize(vec3(nor));
				return nor ;
			}

// Function 4
vec3 calcNormal( in vec3 pos )
{
#if 0    
    vec2 e = vec2(0.002,0.0); 
    return normalize( vec3( map(pos+e.xyy).x - map(pos-e.xyy).x,
                            map(pos+e.yxy).x - map(pos-e.yxy).x,
                            map(pos+e.yyx).x - map(pos-e.yyx).x ) );
#else
    // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+e*0.002).x;
    }
    return normalize(n);
#endif    
}

// Function 5
vec3 normal(vec3 p) {
	vec2 e = vec2(1,0)/1e3;
    p += 0.01 * vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx))/ (2. * length(e));
	return normalize(p);
}

// Function 6
vec2 ct_vfield_normal(
    in vec2 p,
    float npow
){
    vec2 g = vec2(0.0, 0.0);
    
    const int imax = CT_N;
    
    for (int i = 0; i < imax; ++i)
    {
        vec2 dif = g_vfp[i].p - p;
        float sum = dif[0] * dif[0] + dif[1] * dif[1];
        float mass = pow(sum, npow);
        
      	g[0] = g[0] + g_vfp[i].m * dif[0] / mass;
      	g[1] = g[1] + g_vfp[i].m * dif[1] / mass;
        
        
    }
    
    return normalize(g);
}

// Function 7
void computeNormals( out vec2 norm[kNumPoints], in vec2 path[kNumPoints] )
{
    for( int i=0; i<kNumPoints; i++ )
    {
        vec2 n = computeTangent( path[(i-1+kNumPoints)%kNumPoints],
                                 path[(i+0+kNumPoints)%kNumPoints],
                                 path[(i+1+kNumPoints)%kNumPoints] );
        norm[i] = vec2(n.y, -n.x );
    }
}

// Function 8
vec3 genNormal(vec3 p){
	return normalize(p);
}

// Function 9
vec3 getNormal(vec2 uv, int tex ) {
#ifdef NORMAL_MAPS
    float heightScale = 0.004;
    float dHdU, dHdV;
    
    float hpx, hmx, hpy, hmy, h0;
    vec3 c, c1, c2, c3, c4;
    vec2 duv;
    
    if(tex==0){
        vec2 res = iChannelResolution[0].xy;
    	duv = vec2(1.0) / res.xy;
 		c = texture( iChannel0, uv).xyz;
        c1 = texture( iChannel0, uv + vec2(duv.x, 0.0)).xyz;
        c2 = texture( iChannel0, uv - vec2(duv.x, 0.0)).xyz;
        c3 = texture( iChannel0, uv + vec2(0.0, duv.y)).xyz;
        c4 = texture( iChannel0, uv - vec2(0.0, duv.y)).xyz;
    } else if(tex==1) {
        vec2 res = iChannelResolution[1].xy;
    	duv = vec2(1.0) / res.xy;
        c = texture( iChannel1, uv).xyz;
        c1 = texture( iChannel1, uv + vec2(duv.x, 0.0)).xyz;
        c2 = texture( iChannel1, uv - vec2(duv.x, 0.0)).xyz;
        c3 = texture( iChannel1, uv + vec2(0.0, duv.y)).xyz;
        c4 = texture( iChannel1, uv - vec2(0.0, duv.y)).xyz;
        res = iChannelResolution[1].xy;
    } else {
        vec2 res = iChannelResolution[2].xy;
    	duv = vec2(1.0) / res.xy;
        c = texture( iChannel2, uv).xyz;
        c1 = texture( iChannel2, uv + vec2(duv.x, 0.0)).xyz;
        c2 = texture( iChannel2, uv - vec2(duv.x, 0.0)).xyz;
        c3 = texture( iChannel2, uv + vec2(0.0, duv.y)).xyz;
        c4 = texture( iChannel2, uv - vec2(0.0, duv.y)).xyz;
        res = iChannelResolution[2].xy;
    }
    
    h0	= heightScale * dot(c , vec3(1.0/3.0));
    hpx = heightScale * dot(c1, vec3(1.0/3.0));
    hmx = heightScale * dot(c2, vec3(1.0/3.0));
    hpy = heightScale * dot(c3, vec3(1.0/3.0));
    hmy = heightScale * dot(c4, vec3(1.0/3.0));
    dHdU = (hmx - hpx) / (2.0 * duv.x);
    dHdV = (hmy - hpy) / (2.0 * duv.y);
    
    return normalize(vec3(dHdU, dHdV, 1.0));
#else
    return vec3(0.0, 0.0, 1.0);
#endif
}

// Function 10
vec3 getNormal(vec3 p, float eps) {
    vec3 n;
    n.y = map_detailed(p);    
    n.x = map_detailed(vec3(p.x+eps,p.y,p.z)) - n.y;
    n.z = map_detailed(vec3(p.x,p.y,p.z+eps)) - n.y;
    n.y = eps;
    return normalize(n);
}

// Function 11
vec3 get_normal(vec2 uv) {
    // http://web.cs.ucdavis.edu/~amenta/s12/findnorm.pdf
 	uv = uv * (1.0 + 0.6*cos(iTime));
    
    float phi = 2.0*3.1416*uv.x;
    float theta = 2.0*3.1416*uv.y;
    
    vec3 dphi = vec3(-sin(phi), cos(phi), 0.0);
    vec3 dtheta = vec3(-sin(theta)*cos(phi), -sin(theta)*sin(phi), cos(theta));
    
    return normalize(cross(dphi, dtheta));  
}

// Function 12
vec3 normalmap(vec2 p) {
    vec2 e = vec2(1e-3, 0);
    return normalize(vec3(
        heightmap(p - e.xy) - heightmap(p + e.xy),
        heightmap(p - e.yx) - heightmap(p + e.yx),
        2. * e.x));
}

// Function 13
vec3 getNormal(vec3 q, float t){
    float h = .00001
        * sqrt(1./3.)
        * (1. + 128.*t)
        ;
    vec3 n = vec3(0);
    int i = 0;
    i = ZRO;
    for (; i < 4; ++i) {
        vec3 e = vec3((ivec3(i+3, i, i+i)&2) - 1);
        n += map(q + e * h).x * e;
    }
    return normalize(n);
}

// Function 14
float NormalDistributionGGX(float NdotH, float roughness)
{    
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    
    float numerator = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom *= denom * PI;
    
    return numerator / denom;
}

// Function 15
vec3 normal(in vec3 p)
{  
    vec2 e = vec2(-1., 1.)*0.005;   
	return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
					 e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );   
}

// Function 16
vec3 computeNormalBox(vec3 p,Box b)
{
    
	// project on edges

	vec3 center = (b.bmax + b.bmin)/2.;
	vec3 halfSize = (b.bmax - b.bmin)/2.;
	vec3 nx = vec3(1.,0.,0.);
	vec3 ny = vec3(0.,1.,0.);
	vec3 nz = vec3(0.,0.,1.);

	vec3 f1 = center + nx*halfSize.x;
	vec3 f2 = center - nx*halfSize.x;

	vec3 f3 = center + ny*halfSize.y;
	vec3 f4 = center - ny*halfSize.y;

	vec3 f5 = center + nz*halfSize.z;
	vec3 f6 = center - nz*halfSize.z;


	// compute side distance
	if(abs(dot(f1-p,nx)) < 0.00001)
	{
		return nx;
	}
	if(abs(dot(f2-p,nx)) < 0.00001)
	{
		return -nx;
	}
	if(abs(dot(f3-p,ny)) < 0.00001)
	{
		return ny;
	}
	if(abs(dot(f4-p,ny)) < 0.00001)
	{
		return -ny;
	}

	if(abs(dot(f5-p,nz)) < 0.00001)
	{
		return nz;
	}
	if(abs(dot(f6-p,nz)) < 0.00001)
	{
		return -nz;
	}



	return normalize(vec3(1.,1.,1.));
}

// Function 17
vec3 boxNormal(vec3 pos)
{
	return normalize(vec3(distSmoothBox(pos+vec3(EPSILON,0,0)),
						distSmoothBox(pos+vec3(0,EPSILON,0)),
						distSmoothBox(pos+vec3(0,0,EPSILON)))-
						distSmoothBox(pos));
}

// Function 18
vec3 calc_normal(vec3 sample_point) {
    const float h = NORMAL_SAMPLE_SIZE; // replace by an appropriate value
    const vec2 k = vec2(1,-1);
    
    vec3 normal = normalize(
		k.xyy * map( sample_point + k.xyy*h ) + 
		k.yyx * map( sample_point + k.yyx*h ) + 
		k.yxy * map( sample_point + k.yxy*h ) + 
		k.xxx * map( sample_point + k.xxx*h ) );
    normal = normal.zyx;
    return normal;
}

// Function 19
vec3 normalWide(vec3 p) {
	const float epsilon = 0.07;
    return normalize( vec3(scene( vec3(p.x + epsilon, p.y, p.z)) - scene( vec3(p.x - epsilon, p.y, p.z)) , scene( vec3(p.x, p.y + epsilon, p.z)) - scene( vec3(p.x, p.y - epsilon, p.z)), scene( vec3(p.x, p.y, p.z + epsilon)) - scene( vec3(p.x, p.y, p.z - epsilon))));
}

// Function 20
vec3 calcNormal( in vec3 pos ) {
    vec2 e = vec2(1.0,-1.0)*0.0001;
    return normalize( e.xyy*map( pos + e.xyy ) + 
					  e.yyx*map( pos + e.yyx ) + 
					  e.yxy*map( pos + e.yxy ) + 
					  e.xxx*map( pos + e.xxx ) );
}

// Function 21
float ct_normal_mod_pi(
    in vec2 z
){
    float a = atan(z[1], z[0]);
    if (a < 0.0) a += 6.28318;
    return mod(a, 1.0);
}

// Function 22
vec3 getNormal(vec3 p)
{
    const float eps = .001;
    float d0x = scene(vec3(p.x - eps,p.yz));
    float d1x = scene(vec3(p.x + eps,p.yz));
    float d0y = scene(vec3(p.x,p.y - eps,p.z));
    float d1y = scene(vec3(p.x,p.y + eps,p.z));
    float d0z = scene(vec3(p.xy,p.z - eps));
    float d1z = scene(vec3(p.xy,p.z + eps));
    //return vec3(d1x-d0x,d1y-d0y,d1z-d0z)*(.5/eps);
    //return normalize(vec3(d1x-d0x,d1y-d0y,d1z-d0z)*(2.5/eps));
    return normalize(vec3(d1x-d0x,d1y-d0y,d1z-d0z));
    //return vec3(1.0,0.0,0.0);
}

// Function 23
vec3 calcNormal( in vec3 pos )
{
    vec2  eps = vec2(0.001,0.0);
    return normalize( vec3( map(pos+eps.xyy) - map(pos-eps.xyy),
                            map(pos+eps.yxy) - map(pos-eps.yxy),
                            map(pos+eps.yyx) - map(pos-eps.yyx) ) );
}

// Function 24
vec3 SuperFastNormalFilter(sampler2D _tex,vec2 uv,float strength){
    float p00 = GetTextureLuminance(_tex,uv);
    return normalize(vec3(-dFdx(p00),-dFdy(p00),1.-strength));
}

// Function 25
vec3 normal(vec3 p, float t)
{
    float e = MIN_DIST*t;
    vec2 h =vec2(1,-1)*.5773;
    vec3 n = h.xyy * map(p+h.xyy*e).x+
             h.yyx * map(p+h.yyx*e).x+
             h.yxy * map(p+h.yxy*e).x+
             h.xxx * map(p+h.xxx*e).x;
    return normalize(n);
}

// Function 26
vec3 calcNormal( in vec3 pos )
{
    const float eps = 0.002;             // precision of the normal computation

    const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v2 = vec3(-1.0,-1.0, 1.0);
    const vec3 v3 = vec3(-1.0, 1.0,-1.0);
    const vec3 v4 = vec3( 1.0, 1.0, 1.0);

	return normalize( v1*doModel( pos + v1*eps ).x + 
					  v2*doModel( pos + v2*eps ).x + 
					  v3*doModel( pos + v3*eps ).x + 
					  v4*doModel( pos + v4*eps ).x );
}

// Function 27
vec3 calcNormal(in vec3 p) {
    vec2 e = vec2(1.0, -1.0) * 0.0005;
    return normalize(e.xyy * map(p + e.xyy).x + 
					 e.yyx * map(p + e.yyx).x + 
					 e.yxy * map(p + e.yxy).x + 
					 e.xxx * map(p + e.xxx).x);
}

// Function 28
vec3 calculateNormal(float EPSILON, vec3 pos,in vec2 dist)
{
  	vec2 eps = vec2(0.0, EPSILON);
vec3 normal = normalize(vec3(
    distfunc(pos + eps.yxx) - distfunc(pos - eps.yxx),
   distfunc(pos + eps.xyx) - distfunc(pos - eps.xyx),
    distfunc(pos + eps.xxy) - distfunc(pos - eps.xxy)));
        
      	//vec2 eps = vec2(0.0, EPSILON);
//	vec3 normal = normalize(vec3(
  //  distfunc(pos + eps.yxx).x -dist.x,
  // distfunc(pos + eps.xyx).x - dist.x,
  //  distfunc(pos + eps.xxy).x - dist.x));
    
    return normal;
}

// Function 29
vec3 calcNormal(in vec3 p) {
    vec2 e = vec2(0.0001, 0.0);
     
    return normalize(vec3( map(p+e.xyy) - map(p-e.xyy),
                           map(p+e.yxy) - map(p-e.yxy),
                           map(p+e.yyx) - map(p-e.yyx)));
	
}

// Function 30
vec3 box_normal_from_point(vec3 point,vec3 box_extents)
{
    vec3 normal = vec3(0.0);
    float m = FAR;
    float d;

    d = abs(box_extents.x - abs(point.x));
    if (d < m)
    {
        m = d;
        normal = vec3(1.0,0.0,0.0) * sign(point.x);    // Cardinal axis for X
    }

    d = abs(box_extents.y - abs(point.y));
    if (d < m)
    {
        m = d;
        normal = vec3(0.0,1.0,0.0) * sign(point.y);    // Cardinal axis for Y
    }

    d = abs(box_extents.z - abs(point.z));
    if (d < m)
    {
        m = d;
        normal = vec3(0.0,0.0,1.0) * sign(point.z);    // Cardinal axis for Z
    }

    return normal;
}

// Function 31
vec3 getNormal(vec3 p) {
    const vec2 e = vec2(0.0001, 0.);
    return normalize(
        vec3(
            map(p + e.xyy).x - map(p  - e.xyy).x,
            map(p + e.yxy).x - map(p  - e.yxy).x,
            map(p + e.yyx).x - map(p  - e.yyx).x
            )
        );
}

// Function 32
float renormalize(float c)
{
    vec2 nx = texelFetch(iChannel2, ivec2(0, 0), 0).rg;
    float range = nx.y - nx.x;
    float offset = nx.x;
    return (c - nx.x)*1./range;
}

// Function 33
vec3 NormalSinPowWarpTest(vec3 pos, float freq, float amp, float power) {
	return NormalSinPowWarpOffset(pos, GetOffset(), freq, amp, power);
}

// Function 34
vec3 calcNormal( in vec3 pos ){
    vec3 eps = vec3( 0.001, 0.0, 0.0 );
    vec3 nor = vec3(
        map(pos+eps.xyy, false).dist - map(pos-eps.xyy, false).dist,
        map(pos+eps.yxy, false).dist - map(pos-eps.yxy, false).dist,
        map(pos+eps.yyx, false).dist - map(pos-eps.yyx, false).dist );
    return normalize(nor);
}

// Function 35
vec3 sceneNormal(in vec3 pos )
{
    float eps = 0.0001;
    vec3 n;
    float d = scene(pos);
    n.x = scene( vec3(pos.x+eps, pos.y, pos.z) ) - d;
    n.y = scene( vec3(pos.x, pos.y+eps, pos.z) ) - d;
    n.z = scene( vec3(pos.x, pos.y, pos.z+eps) ) - d;
    return normalize(n);
}

// Function 36
vec3 GetSceneNormal( const in vec3 vPos )
{
    const float fDelta = 0.001;

    vec3 vOffset1 = vec3( fDelta, -fDelta, -fDelta);
    vec3 vOffset2 = vec3(-fDelta, -fDelta,  fDelta);
    vec3 vOffset3 = vec3(-fDelta,  fDelta, -fDelta);
    vec3 vOffset4 = vec3( fDelta,  fDelta,  fDelta);

    float f1 = GetSceneDistance( vPos + vOffset1 ).x;
    float f2 = GetSceneDistance( vPos + vOffset2 ).x;
    float f3 = GetSceneDistance( vPos + vOffset3 ).x;
    float f4 = GetSceneDistance( vPos + vOffset4 ).x;

    vec3 vNormal = vOffset1 * f1 + vOffset2 * f2 + vOffset3 * f3 + vOffset4 * f4;

    return normalize( vNormal );
}

// Function 37
vec3 get_normal(vec3 pos)
{
    vec3 eps = vec3(0.0001,0.0,0.0);
	return normalize(vec3(
           f(pos+eps.xyy) - f(pos-eps.xyy),
           f(pos+eps.yxy) - f(pos-eps.yxy),
           f(pos+eps.yyx) - f(pos-eps.yyx)));
}

// Function 38
vec3 getWaterNormal(vec3 p, float d) {
    return normalize(vec3(
        getWaterLevel(vec2(p.x-nEPS,p.z), d) - getWaterLevel(vec2(p.x+nEPS,p.z), d),
        2.0*nEPS,
        getWaterLevel(vec2(p.x,p.z-nEPS), d) - getWaterLevel(vec2(p.x,p.z+nEPS), d)
    ));
}

// Function 39
vec3 getNormal(vec3 p)
{
    const float d = eps;
    return
        normalize
        (
            vec3
            (
                distanceFunction(p+vec3(d,0.0,0.0))-distanceFunction(p+vec3(-d,0.0,0.0)),
                distanceFunction(p+vec3(0.0,d,0.0))-distanceFunction(p+vec3(0.0,-d,0.0)),
                distanceFunction(p+vec3(0.0,0.0,d))-distanceFunction(p+vec3(0.0,0.0,-d))
            )
        );
}

// Function 40
vec3 getNormal(in vec3 pos )
{
    vec3 eps = vec3( 0.001, 0.0, 0.0 );
	vec3 nor = vec3(
	    map(pos+eps.xyy).d - map(pos-eps.xyy).d,
	    map(pos+eps.yxy).d - map(pos-eps.yxy).d,
	    map(pos+eps.yyx).d - map(pos-eps.yyx).d );
	return normalize(nor);
}

// Function 41
vec3 getNormal( vec3 p ) {
    return normalize( getDistance( p ) - 
        getDistances( p - EPZ.yxx, p - EPZ.xyx, p - EPZ.xxy )
    );
}

// Function 42
vec3 GetBumpMapNormal (vec2 uv, mat3 tangentSpace)
{    
	float delta = -1.0/512.0;
	float A = texture(iChannel1, uv + vec2(0.0, 0.0)).x;
	float B = texture(iChannel1, uv + vec2(delta, 0.0)).x;
    float C = texture(iChannel1, uv + vec2(0.0, delta)).x;    
    
	vec3 norm = normalize(vec3(A - B, A - C, 0.15));
	
	return normalize(tangentSpace * norm);
}

// Function 43
float normalize2(float minV, float maxV, float v) {
    return minV + v * (maxV - minV);
}

// Function 44
vec3 normal(in vec3 p, inout float edge) { 
	
    vec2 e = vec2(.034, 0); // Larger epsilon for greater sample spread, thus thicker edges.

    // Take some distance function measurements from either side of the hit point on all three axes.
	float d1 = map(p + e.xyy), d2 = map(p - e.xyy);
	float d3 = map(p + e.yxy), d4 = map(p - e.yxy);
	float d5 = map(p + e.yyx), d6 = map(p - e.yyx);
	float d = map(p)*2.;	// The hit point itself - Doubled to cut down on calculations. See below.
     
    // Edges - Take a geometry measurement from either side of the hit point. Average them, then see how
    // much the value differs from the hit point itself. Do this for X, Y and Z directions. Here, the sum
    // is used for the overall difference, but there are other ways. Note that it's mainly sharp surface 
    // curves that register a discernible difference.
    edge = abs(d1 + d2 - d) + abs(d3 + d4 - d) + abs(d5 + d6 - d);
    //edge = max(max(abs(d1 + d2 - d), abs(d3 + d4 - d)), abs(d5 + d6 - d)); // Etc.
    
    
    // Once you have an edge value, it needs to normalized, and smoothed if possible. How you 
    // do that is up to you. This is what I came up with for now, but I might tweak it later.
    //
    edge = smoothstep(0., 1., sqrt(edge/e.x*8.));
    
    // Curvature. All this, just to take out the inner edges.
    float crv = (d1 + d2 + d3 + d4 + d5 + d6 - d*3.)/e.x;;
    //crv = clamp(crv*32., 0., 1.);
    if (crv<0.) edge = 0.; // Comment out to see what it does.

	
    // Redoing the calculations for the normal with a more precise epsilon value. If you can roll the 
    // edge and normal into one, it saves a lot of map calls. Unfortunately, we want wide edges, so
    // there are six more, making 12 map calls in all. Ouch! :)
    e = vec2(.005, 0);
	d1 = map(p + e.xyy), d2 = map(p - e.xyy);
	d3 = map(p + e.yxy), d4 = map(p - e.yxy);
	d5 = map(p + e.yyx), d6 = map(p - e.yyx); 
    
    // Return the normal.
    // Standard, normalized gradient mearsurement.
    return normalize(vec3(d1 - d2, d3 - d4, d5 - d6));
}

// Function 45
vec3 calcNormal( in vec3 pos )
  {
	vec2 eps = vec2( 0.0001, 0.0 );
	vec3 nor = vec3( map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	                 map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	                 map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
  }

// Function 46
vec3 computeNormalAt(vec3 pos)
{
    vec3 v = vec3(
    	implicitFunc(pos - vec3(DerivStep, 0., 0.)),
    	implicitFunc(pos - vec3(0., DerivStep, 0.)),
    	implicitFunc(pos - vec3(0., 0., DerivStep))
    );
    vec3 v2 = vec3(
    	implicitFunc(pos + vec3(DerivStep, 0., 0.)),
    	implicitFunc(pos + vec3(0., DerivStep, 0.)),
    	implicitFunc(pos + vec3(0., 0., DerivStep))
    );
    return normalize(v - v2);
}

// Function 47
vec3 calcNormal( in vec3 pos, float t )
{
    vec2  eps = vec2( 0.002*t, 0.0 );
    return normalize( vec3( terrainH(pos.xz-eps.xy) - terrainH(pos.xz+eps.xy),
                            2.0*eps.x,
                            terrainH(pos.xz-eps.yx) - terrainH(pos.xz+eps.yx) ) );
}

// Function 48
float getNormal2(vec3 pos, float objnr)
{
    if (int(objnr)==ROOM_OBJ && pos.y<roomSize.y - 0.012)
    {
        #ifdef doors
        if ((abs(pos.z)>0.5*doorSize.x + 2.*dfSize.x || pos.y>doorSize.y + 2.*dfSize.x || abs(pos.x)>roomSize.x*1.5) && pos.y>2.*dfSize.x)
        #endif
        return -0.0022*noise(pos*58.);
    }
}

// Function 49
vec3 NormalBlend_Overlay(vec3 n1, vec3 n2)
{
    vec3 n;
    n.x = overlay(n1.x, n2.x);
    n.y = overlay(n1.y, n2.y);
    n.z = overlay(n1.z, n2.z);

    return normalize(n*2.0 - 1.0);
}

// Function 50
vec3 normal(vec3 p) {
	vec3 n, E = vec3(.005, 0., 0.);

/*
	float n1,n2,n3;

	n1=map(p);
	n2=map(p - 1.0 * E.xyy);
	n3=map(p + 1.0 * E.xyy);
	if (abs(n1-n2)>abs(n3-n1)) {n.x=(n3 - n1)*2.0;  }
	else n.x=(n1 - n2)*2.0;

	n2=map(p - 1.0 * E.yxy);
	n3=map(p + 1.0 * E.yxy);
	if (abs(n1-n2)>abs(n3-n1)) {n.y=(n3 - n1)*2.0;  }
	else n.y=(n1 - n2)*2.0;

	n2=map(p - 1.0 * E.yyx);
	n3=map(p + 1.0 * E.yyx);
	if (abs(n1-n2)>abs(n3-n1)) {n.z=(n3 - n1)*2.0;  }
	else n.z=(n1 - n2)*2.0;
*/


	n.x = map(p + E.xyy) - map(p - E.xyy);
	n.y = map(p + E.yxy) - map(p - E.yxy);
	n.z = map(p + E.yyx) - map(p - E.yyx);
	return normalize(n);
}

// Function 51
vec3 calcNormal( in vec3 pos, float time )
{

    #if 0
    vec2 e = vec2(1.0,-1.0)*0.005773;
    return normalize( e.xyy*map( pos + e.xyy, time ) + 
                     e.yyx*map( pos + e.yyx, time ) + 
                     e.yxy*map( pos + e.yxy, time ) + 
                     e.xxx*map( pos + e.xxx, time ) );
    #else
    // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
    vec3 n = vec3(0.0);
    for( int i=0; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+0.001*e,time);
    }
    return normalize(n);
    #endif    
}

// Function 52
vec3 sphereNormal(vec3 pos)
{
	return normalize(vec3(distSmoothSphere(pos+vec3(EPSILON,0,0)),
						distSmoothSphere(pos+vec3(0,EPSILON,0)),
						distSmoothSphere(pos+vec3(0,EPSILON,0)))-
						distSmoothSphere(pos));
}

// Function 53
vec3 calcNormal( in vec3 pos )
{
#if 0
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
    return normalize( e.xyy*map( pos + e.xyy ).x + 
					  e.yyx*map( pos + e.yyx ).x + 
					  e.yxy*map( pos + e.yxy ).x + 
					  e.xxx*map( pos + e.xxx ).x );
#else
    // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
    vec3 n = vec3(0.0);
    for( int i=min(0,iFrame); i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+0.001*e).x;
    }
    return normalize(n);
#endif    
}

// Function 54
vec3 getTrilinearSurfNormal(in vec3 p, in float a, in float b, in float c, in float d, in float e, in float f, in float g, in float h) {
    float x = p.x, y = p.y, z = p.z;

    float ba = b - a, ca = c - a;
    float q = ba + c - d + e - f - g + h, qx = q * x;
    float dbac = d - ba - c, fbae = f - ba - e, gcae = g - ca - e;

    float dx = ba + dbac * y + (fbae + q * y) * z;
    float dy = ca + dbac * x + (gcae + qx) * z;
    float dz = e - a + fbae * x + (gcae + qx) * y;

    return normalize(vec3(dx, dy, dz));
}

// Function 55
vec3 texNormal(vec2 pos){
    if(abs(pos.x-0.5)>=0.25||abs(pos.y-0.5)>=0.25){
        if(abs(pos.x-0.5)<=0.5&&abs(pos.y-0.5)<=0.5){
            vec3 leftdown=(pos.x>pos.y)?vec3(0,1,1):vec3(1,0,1);
            vec3 rightup=(pos.x>pos.y)?vec3(-1,0,1):vec3(0,-1,1);
            return normalize((pos.x+pos.y>=1.)?rightup:leftdown);
        }
    }
    float dsquare=(pos.x-0.5)*(pos.x-0.5)+(pos.y-0.5)*(pos.y-0.5);
    if(dsquare<=1./16.){
        return normalize(vec3(pos.x-0.5,pos.y-0.5,sqrt(1./16.-dsquare)));
    }
	return vec3(0,0,1);
}

// Function 56
vec3 calcNormal( in vec3 pos )
{
	pos = snap(pos);
    vec2 e = vec2(1.0,-1.0)*(1.0/offset);
    return normalize( e.xyy*dist( pos + e.xyy ) + 
					  e.yyx*dist( pos + e.yyx ) + 
					  e.yxy*dist( pos + e.yxy ) + 
					  e.xxx*dist( pos + e.xxx ) );
   
}

// Function 57
vec3 getNormals(vec2 pos){
	vec3 normals = vec3(0.0);
	
	float delta = 0.01;
    
	float d0 = getHeight(pos);
	float d1 = getHeight(pos + vec2(delta, 0.0));
	float d2 = getHeight(pos + vec2(0.0, delta));
	
	float dx = (d0 - d1) / delta;
	float dy = (d0 - d2) / delta;
	
	normals = normalize(vec3(dx, dy, 1.0 - d0));
	
	return normals;
}

// Function 58
vec3 GetNormal(vec3 p) 
{
    return normalize(p-PLANETCENTER);
}

// Function 59
float3 Normal ( float3 p ) {
  float2 e = float2(1.0, -1.0)*0.001;
  return normalize(
                   e.xyy*Map(p + e.xyy).x +
                   e.yyx*Map(p + e.yyx).x +
                   e.yxy*Map(p + e.yxy).x +
                   e.xxx*Map(p + e.xxx).x);
}

// Function 60
vec3 calcNormal(in vec3 p) {
	const vec2 e = vec2(0.005, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 61
vec3 normal(vec3 p){
    return normalize(vec3(map(vec3(p.x + 0.0001, p.yz)) - map(vec3(p.x - 0.0001, p.yz)),
                          map(vec3(p.x, p.y + 0.0001, p.z)) - map(vec3(p.x, p.y - 0.0001, p.z)),
                	      map(vec3(p.xy, p.z + 0.0001)) - map(vec3(p.xy, p.z - 0.0001))));
}

// Function 62
float normalizedSin(float x) {
    return abs(sin(x));
}

// Function 63
vec3 GetNormal (vec3 p) 
{ 
    vec2 e = vec2(0.01, 0.0); 
    return normalize(vec3(
        map(p+e.xyy)-map(p-e.xyy),
        map(p+e.yxy)-map(p-e.yxy),
        map(p+e.yyx)-map(p-e.yyx)
        )); 
}

// Function 64
vec3 gradNormal(int objId,vec3 p) {
	vec3 g=make_float3(
		dist(objId,p + make_float3(GRADIENT_DELTA, 0, 0)) - dist(objId,p - make_float3(GRADIENT_DELTA, 0, 0)),
		dist(objId,p + make_float3(0, GRADIENT_DELTA, 0)) - dist(objId,p - make_float3(0, GRADIENT_DELTA, 0)),
		dist(objId,p + make_float3(0, 0, GRADIENT_DELTA)) - dist(objId,p - make_float3(0, 0, GRADIENT_DELTA)));

    return normalize(g);
}

// Function 65
vec3 calcNormal( in vec3 pos )
{
  vec3 eps = vec3(0.001, 0.0, 0.0);
  vec3 nor = vec3(map(pos+eps.xyy).x - map(pos-eps.xyy).x,
                  map(pos+eps.yxy).x - map(pos-eps.yxy).x,
                  map(pos+eps.yyx).x - map(pos-eps.yyx).x);
  return normalize(nor);
}

// Function 66
vec3 calcNormal(const vec3 pos, const float t )
{
    vec3 e = (PRECISION_FACTOR * t * .57) * vec3(1, -1, 0);
    return normalize
        (e.xyy*getDistance(pos + e.xyy) +
		 e.yyx*getDistance(pos + e.yyx) +
		 e.yxy*getDistance(pos + e.yxy) +
         e.xxx*getDistance(pos + e.xxx) );
}

// Function 67
vec3 water_normal (vec2 p, float h, float dst)
{
    const float wd = 0.5;
    
    float wx = fwidth(p.x)*wd;
    float wy = fwidth(p.y)*wd;
    
    float t0 =  waterMap(p);
    float tu =  waterMap(p + vec2(0., wy));
    float td =  waterMap(p - vec2(0., wy));
    float tl =  waterMap(p - vec2(wx, 0));
    float tr =  waterMap(p + vec2(wx, 0));
    float tdr = waterMap(p + vec2(wx, wy));
    float tul = waterMap(p - vec2(wx, wy));
    
    vec2 t1 = vec2( t0 - tl, tul - tl );
    vec2 t2 = vec2( tr - t0, tu - t0 );
	
    vec2 rz = (t1 + t2)*0.5;
    t1 = vec2( tdr - td, t0 - td );
    rz = mix(rz, (t1 + t2)*0.5, 0.5);

    h *= pow(dst, 2.);
    return normalize( vec3(rz.x, h*9., rz.y ) );
}

// Function 68
vec3 calcNormal( in vec3 pos, float prec )
{
	vec3 eps = vec3( prec, 0., 0. );
	vec3 nor = vec3(
        map(pos+eps.xyy) - map(pos-eps.xyy),
        map(pos+eps.yxy) - map(pos-eps.yxy),
        map(pos+eps.yyx) - map(pos-eps.yyx) );
	return normalize(nor);
}

// Function 69
vec3 calcNormal( in vec3 pos, in float eps )
{
#if 0    
    vec2 e = vec2(1.0,-1.0)*0.5773*eps;
    return normalize( e.xyy*mapTotal( pos + e.xyy ) + 
					  e.yyx*mapTotal( pos + e.yyx ) + 
					  e.yxy*mapTotal( pos + e.yxy ) + 
					  e.xxx*mapTotal( pos + e.xxx ) );
#else
    // trick by klems, to prevent the compiler from inlining map() 4 times
    vec4 n = vec4(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec4 s = vec4(pos, 0.0);
        s[i] += eps;
        n[i] = mapTotal(s.xyz);
    }
    return normalize(n.xyz-n.w);
#endif    
   
}

// Function 70
vec3 normalized(vec3 a)
{
    return a/length(a);
}

// Function 71
vec3 calcNormal( in vec3 pos )
{
    vec3 e = vec3(0.01,0.0,0.0);
	return normalize( vec3(fbmd(pos.xz-e.xy).x - fbmd(pos.xz+e.xy).x,
                           2.0*e.x,
                           fbmd(pos.xz-e.yx).x - fbmd(pos.xz+e.yx).x ) );
}

// Function 72
vec3 ComputeDetailNormal(vec2 uv)
{
    const vec4 avgRGB0 = vec4(1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0);
    const float scale = 0.02;
    const vec2 du = vec2(1.0/512.0, 0.0);
    const vec2 dv = vec2(0.0, 1.0/512.0);

    float h0  = dot(avgRGB0, texture(iChannel0, uv)) * scale;
    float hpx = dot(avgRGB0, texture(iChannel0, uv + du)) * scale;
    float hmx = dot(avgRGB0, texture(iChannel0, uv - du)) * scale;
    float hpy = dot(avgRGB0, texture(iChannel0, uv + dv)) * scale;
    float hmy = dot(avgRGB0, texture(iChannel0, uv - dv)) * scale;
    
    float dHdU = (hmx - hpx) / (2.0 * du.x);
    float dHdV = (hmy - hpy) / (2.0 * dv.y);
    
    return normalize(vec3(dHdU, dHdV, 1.0)) * 0.5 + 0.5;
}

// Function 73
vec3 terrainNormal(vec2 pos)
{    
    return vec3(0,0,1);
}

// Function 74
vec2 computeTangent( in vec2 a, in vec2 b, in vec2 c )
{
    return (gMethod==0) ? c - a :
           (gMethod==1) ? normalize( c-a ) :
                          normalize( normalize(c-b) + normalize(b-a) );
}

// Function 75
vec3 normal(in vec3 p, in vec3 ray, in float t) {
   // vec2 e = vec2(E.y, -E.y); 
   // return normalize(e.xyy * map(p + e.xyy).x + e.yyx * map(p + e.yyx).x + e.yxy * map(p + e.yxy).x + e.xxx * map(p + e.xxx).x);;
	float pitch = .2 * t / iResolution.x;
    
	vec2 d = vec2(-1,1) * pitch;

	vec3 p0 = p+d.xxx; // tetrahedral offsets
	vec3 p1 = p+d.xyy;
	vec3 p2 = p+d.yxy;
	vec3 p3 = p+d.yyx;
	
	float f0 = map(p0).x;
	float f1 = map(p1).x;
	float f2 = map(p2).x;
	float f3 = map(p3).x;
	
	vec3 grad = p0*f0+p1*f1+p2*f2+p3*f3 - p*(f0+f1+f2+f3);
	//return normalize(grad);	
    // prevent normals pointing away from camera (caused by precision errors)
	return normalize(grad - max(.0,dot (grad,ray ))*ray);
}

// Function 76
vec3 getNormal(vec3 pos)
{
	float d=getDist(pos);
	return normalize(vec3( getDist(pos+vec3(EPSILON,0,0))-d, getDist(pos+vec3(0,EPSILON,0))-d, getDist(pos+vec3(0,0,EPSILON))-d ));
}

// Function 77
vec3 getNormal(in vec3 p) {
    vec3 n=vec3(0.);
    for(int i=gFrame;i<=2;i++){
        vec3 e=  0.001* ((i==0)?vec3(1,0,0):(i==1)?vec3(0,1,0):vec3(0,0,1));
        for(float j=-1.;j<=1.;j+=2.) n+= j*e* mapScene(p + j* e).x ;
    }
    return normalize(n);
 
}

// Function 78
vec3 box_normal_from_point(vec3 point,vec3 box_extents)
{
	vec3 normal = vec3(0.0);
	float m = INF;
	float d;

	d = abs(box_extents.x - abs(point.x));
	if (d < m)
	{
		m = d;
		normal = vec3(1.0,0.0,0.0) * sign(point.x);    // Cardinal axis for X
	}

	d = abs(box_extents.y - abs(point.y));
	if (d < m)
	{
		m = d;
		normal = vec3(0.0,1.0,0.0) * sign(point.y);    // Cardinal axis for Y
	}

	d = abs(box_extents.z - abs(point.z));
	if (d < m)
	{
		m = d;
		normal = vec3(0.0,0.0,1.0) * sign(point.z);    // Cardinal axis for Z
	}

	return normal;
}

// Function 79
vec3 calcNormal( in vec3 pos, in float eps )
{
    vec2 e = vec2(1.0,-1.0)*0.5773*eps;
    return normalize( e.xyy*map( pos + e.xyy ) + 
					  e.yyx*map( pos + e.yyx ) + 
					  e.yxy*map( pos + e.yxy ) + 
					  e.xxx*map( pos + e.xxx ) );
}

// Function 80
vec3 getNormal(vec3 p, float t){
    float e = min_dist * t;
    vec2 h = vec2(1.5,-1.5)*.5773;
    return normalize( h.xyy*map( p + h.xyy*e ).x + 
					  h.yyx*map( p + h.yyx*e ).x + 
					  h.yxy*map( p + h.yxy*e ).x + 
					  h.xxx*map( p + h.xxx*e ).x );
}

// Function 81
vec3 getNormal(vec3 pos, float e, bool inside)
{  
    vec3 n = vec3(0.0);
    for( int i=0; i<4; i++ )
    {
        vec3 e2 = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e2*map_smooth(pos + e*e2);
    }
    return (inside?-1.:1.)*normalize(n);
}

// Function 82
vec3 normal(vec3 p)
{
 	vec3 P = vec3(-4, 4, 0) * PRE;

 	vec3 N = normalize(model(p+P.xyy)*P.xyy+model(p+P.yxy)*P.yxy+
                  model(p+P.yyx)*P.yyx+model(p+P.xxx)*P.xxx);
    
    vec3 B = vec3(t3(iChannel0,p+P.xzz,N).r,t3(iChannel0,p+P.zxz,N).r,
                  t3(iChannel0,p+P.zzx,N).r)-t3(iChannel0,p,N).r;
    B = (B-N*dot(B,N));
    return normalize(N+B*8.0);
}

// Function 83
float sampleNormalDistribution(float U, float mu, float sigma)
{
    float x = sigma * 1.414213f * erfinv(2.0f * U - 1.0f) + mu;
    return x;
}

// Function 84
vec3 calcNormal( in vec3 pos ) {
    const float eps = INTERSECTION_PRECISION;

    const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v2 = vec3(-1.0,-1.0, 1.0);
    const vec3 v3 = vec3(-1.0, 1.0,-1.0);
    const vec3 v4 = vec3( 1.0, 1.0, 1.0);

	return normalize( v1*doModel( pos + v1*eps ) + 
					  v2*doModel( pos + v2*eps ) + 
					  v3*doModel( pos + v3*eps ) + 
					  v4*doModel( pos + v4*eps ) );
}

// Function 85
vec3 getNormal( vec2 uv, vec2 c, vec4 t ) {
	vec2 du = vec2( 1.0 / 1024.0, 0.0 );
    vec2 dv = vec2( 0.0, 1.0 / 1024.0 );
    
    float hpx = sampleHeight( uv + du, c, t );
    float hmx = sampleHeight( uv - du, c, t );
    float hpy = sampleHeight( uv + dv, c, t );
    float hmy = sampleHeight( uv - dv, c, t );
    
    float dHdU = ( hmx - hpx ) / ( 2.0 * du.x );
    float dHdV = ( hmy - hpy ) / ( 2.0 * dv.y );
    
    return vec3( dHdU, dHdV, 1.0 );
}

// Function 86
vec2 ct_vfield_normal(
    in vec2 p,
    float npow
){
    vec2 g = vec2(0.0, 0.0);
    
    const int imax = CT_N * CT_N + 1;
    
    for (int i = 0; i < imax; ++i)
    {
        vec2 dif = g_vfp[i].p - p;
        float sum = dif[0] * dif[0] + dif[1] * dif[1];
        float mass = pow(sum, npow);
        
      	g[0] = g[0] + g_vfp[i].m * dif[0] / mass;
      	g[1] = g[1] + g_vfp[i].m * dif[1] / mass;
    }
    
    return normalize(g);
}

// Function 87
vec2 NormalizeScreenCoords(vec2 screenCoord)
{
    vec2 result = 2.0 * (screenCoord/iResolution.xy - 0.5);
    result.x *= iResolution.x/iResolution.y;
    return result;
}

// Function 88
mat3 OrthoNormalMatrixFromZY( vec3 zDirIn, vec3 yHintDir )
{
	vec3 xDir = normalize( cross( zDirIn, yHintDir ) );
	vec3 yDir = normalize( cross( xDir, zDirIn ) );
	vec3 zDir = normalize( zDirIn );

	mat3 result = mat3( xDir, yDir, zDir );
		
	return result;
}

// Function 89
vec3 normal(vec3 surface){
  vec2 offset = vec2(0.01,0.);
    vec3 nDir = vec3(
      obj(surface+offset.xyy),
        obj(surface+offset.yxy),
        obj(surface+offset.yyx)
    ) - obj(surface);
    return normalize(nDir);
}

// Function 90
vec3 getNormal( in vec3 p ) {
    vec4 n = vec4(0);
    for (int i = Z ; i < 4 ; i++) {
        vec4 s = vec4(p, 0);
        s[i] += 0.0001;
        n[i] = de(s.xyz);
    }
    return normalize(n.xyz-n.w);
}

// Function 91
vec3 GetVolumeNormal( in vec3 pos, float time, int sceneType )
{
    vec3 n = vec3(0.0);
    for( int i=min(0, iFrame); i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*
            ((sceneType == SCENE_TYPE_OCEAN) ?
                QueryVolumetricDistanceField(pos+0.5*e, time) :
                QueryOceanDistanceField(pos+0.5*e, time));
    }
    return normalize(n);
}

// Function 92
vec3 normal(const in vec3 p)
{  
    vec2 e = vec2(-1., 1.)*0.005;   
	return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
					 e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );   
}

// Function 93
vec3 calcNormal (in vec3 pos)
{
	const vec3 eps = vec3 (0.0001, 0, 0);
    float d = distanceFunction(pos);
    return normalize( 
      vec3 (distanceFunction(pos+eps.xyy) - d,
	        distanceFunction(pos+eps.yxy) - d,
	        distanceFunction(pos+eps.yyx) - d));
}

// Function 94
vec3 calcNormal(in vec3 pos)
{
	vec2 eps = vec2( 0.001, 0.0);
	vec3 nor = vec3(
	    scene(pos+eps.xyy) - scene(pos-eps.xyy),
	    scene(pos+eps.yxy) - scene(pos-eps.yxy),
	    scene(pos+eps.yyx) - scene(pos-eps.yyx) );
	return normalize(nor);
}

// Function 95
vec3 getNormal(vec2 uv, int tex ) {
#ifdef NORMAL_MAPS
    float heightScale = 0.004;
    float dHdU, dHdV;
    
    float hpx, hmx, hpy, hmy, h0;
    vec3 c, c1, c2, c3, c4;
    vec2 duv;
    
#if __VERSION__ < 300
    if(tex==0){
        GET_COLORS(0, 0);
    } else if(tex==1) {
        GET_COLORS(1, 1);
    } else {
        GET_COLORS(2, 2);
    }
#else
    switch(tex){
        case 0: {GET_COLORS(0, 0);}
        case 1: {GET_COLORS(1, 1);}
        case 2: {GET_COLORS(2, 2);}
    }
#endif
    
    h0	= heightScale * dot(c , vec3(1.0/3.0));
    hpx = heightScale * dot(c1, vec3(1.0/3.0));
    hmx = heightScale * dot(c2, vec3(1.0/3.0));
    hpy = heightScale * dot(c3, vec3(1.0/3.0));
    hmy = heightScale * dot(c4, vec3(1.0/3.0));
    dHdU = (hmx - hpx) / (2.0 * duv.x);
    dHdV = (hmy - hpy) / (2.0 * duv.y);
    
    return normalize(vec3(dHdU, dHdV, 1.0));
#else
    return vec3(0.0, 0.0, 1.0);
#endif
}

// Function 96
vec3 sphNormal(in vec3 pos, in vec4 sph) {
    return normalize(pos - sph.xyz);
}

// Function 97
vec3 calcNormal(vec3 pos){
    vec3 eps = vec3(.0005,0,0);
    vec3 nor = vec3(0);
    float invert = 1.;
    vec3 npos;
    for (int i = 0; i < NORMAL_STEPS; i++){
        npos = pos + eps * invert;
        nor += map(npos) * eps * invert;
        eps = eps.zxy;
        invert *= -1.;
    }
    return normalize(nor);
}

// Function 98
vec3 normal(vec2 pos, float e, float depth){
    vec2 ex = vec2(e, 0);
    H = getwaves(pos.xy * 0.1, ITERATIONS_NORMAL) * depth;
    vec3 a = vec3(pos.x, H, pos.y);
    return normalize(cross(normalize(a-vec3(pos.x - e, getwaves(pos.xy * 0.1 - ex.xy * 0.1, ITERATIONS_NORMAL) * depth, pos.y)), 
                           normalize(a-vec3(pos.x, getwaves(pos.xy * 0.1 + ex.yx * 0.1, ITERATIONS_NORMAL) * depth, pos.y + e))));
}

// Function 99
vec3 calcNormal( in vec3 pos ) {
    vec3 eps = vec3( 0.001, 0.0, 0.0 );
    vec3 nor = vec3(
        map(pos+eps.xyy).x - map(pos-eps.xyy).x,
        map(pos+eps.yxy).x - map(pos-eps.yxy).x,
        map(pos+eps.yyx).x - map(pos-eps.yyx).x );
    return normalize(nor);
}

// Function 100
vec3 getNormal(in vec3 p, float t) {
	const vec2 e = vec2(.001, 0);
    
    
    //vec3 n = normalize(vec3(map(p + e.xyy) - map(p - e.xyy),
    //map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
    
    float sgn = 1.;
    vec3 n = vec3(0);
    float mp[6];
    vec3[3] e6 = vec3[3](e.xyy, e.yxy, e.yyx);
    for(int i = min(iFrame, 0); i<6; i++){
		mp[i] = map(p + sgn*e6[i/2]);
        sgn = -sgn;
        if(sgn>2.) break;
    }
    
    return normalize(vec3(mp[0] - mp[1], mp[2] - mp[3], mp[4] - mp[5]));
}

// Function 101
vec3 boxNormal(vec3 rp,vec3 p0,vec3 p1)
{
    rp = rp - (p0 + p1) / 2.0;
    vec3 arp = abs(rp) / (p1 - p0);
    return normalize(step(arp.yzx, arp) * step(arp.zxy, arp) * sign(rp));
}

// Function 102
vec3 normal(vec3 sp)
{
    vec3 eps = vec3(.0001, 0.0, 0.0);
    
    vec3 normal = normalize (vec3( map(sp+eps) - map(sp-eps)
                       ,map(sp+eps.yxz) - map(sp-eps.yxz)
                       ,map(sp+eps.yzx) - map(sp-eps.yzx) ));
    
    
 return normal;   
}

// Function 103
vec3 getNormal(vec3 p){
    vec3 n = vec3(0.0);
    int id;
    for(int i = ZERO; i < 4; i++){
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*getSDF(p+e*EPSILON, id);
    }
    return normalize(n);
}

// Function 104
vec3 calcNormalTex(in vec3 pos)
{
    
    float center = luminosity(texture(iChannel0, pos.xy*TEXTURE_TILE_XY).xyz) * NORMAL_STRENGTH;
    float n = luminosity(texture(iChannel0, pos.xy*TEXTURE_TILE_XY + vec2(0.0, TEX_SAMPLE)).xyz) * NORMAL_STRENGTH;
    float s = luminosity(texture(iChannel0, pos.xy*TEXTURE_TILE_XY + vec2(0.0, -TEX_SAMPLE)).xyz) * NORMAL_STRENGTH;
    float e = luminosity(texture(iChannel0, pos.xy*TEXTURE_TILE_XY + vec2(TEX_SAMPLE, 0.0)).xyz) * NORMAL_STRENGTH; 
    float w = luminosity(texture(iChannel0, pos.xy*TEXTURE_TILE_XY + vec2(-TEX_SAMPLE, 0.0)).xyz) * NORMAL_STRENGTH; 
    
    
    float epsilon = 0.001;
    float meshCenter = scene(pos);
    float meshX = scene(pos - vec3(epsilon, 0.0, 0.0));
    float meshY = scene(pos - vec3(0.0, epsilon, 0.0));
    float meshZ = scene(pos - vec3(0.0, 0.0, epsilon));
    
    vec3 meshNorm = normalize(vec3(meshX-meshCenter, meshY-meshCenter, meshZ-meshCenter));
    
    vec3 norm = meshNorm;
    vec3 temp = norm;
    if (norm.x == 1.0)
    {
     	temp.y += 0.5;   
    }
    else
    {
     	temp.x += 0.5;   
    }
    
    vec3 perp1 = normalize(cross(norm, temp));
    vec3 perp2 = normalize(cross(norm, perp1));
    
    vec3 offset = -NORMAL_STRENGTH * (((n-center)-(s-center) * perp1) + ((e-center) - (w-center)) * perp2);
    norm += offset;
    
    return norm;
}

// Function 105
vec3 getNormal(in vec3 p, float t) {
	const vec2 e = vec2(.001, 0);
    
    //vec3 n = normalize(vec3(map(p + e.xyy) - map(p - e.xyy),
    //map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
    
    // So, what's this mess then? Glad you asked. :) Apparently, if there's a break, things
    // won't get unrolled, so the idea is that this will cut down on the number of unrolled
    // map calls which can add to compile time... As to whether it works or not, I can't say,
    // but it seems to cut compile time down on my machine. If you know of something better,
    // feel free to let me know. In case it needs to be said, from a code perspective, I 
    // do not like this. :)
    float sgn = 1.;
    vec3 n = vec3(0);
    float mp[6];
    vec3[3] e6 = vec3[3](e.xyy, e.yxy, e.yyx);
    for(int i = min(iFrame, 0); i<6; i++){
		mp[i] = map(p + sgn*e6[i/2]);
        sgn = -sgn;
        if(sgn>2.) break;
    }
    
    return normalize(vec3(mp[0] - mp[1], mp[2] - mp[3], mp[4] - mp[5]));
}

// Function 106
vec3 calcNormal( in vec3 pos )
{
    vec3 eps = vec3( 0.00001, 0.0, 0.0 );
    vec3 nor = vec3(
        mapScene(pos+eps.xyy).d - mapScene(pos-eps.xyy).d,
        mapScene(pos+eps.yxy).d - mapScene(pos-eps.yxy).d,
        mapScene(pos+eps.yyx).d - mapScene(pos-eps.yyx).d );
    return normalize(nor);
}

// Function 107
vec3 bumpMapNormal( const in vec3 pos, in vec3 nor ) {
    float i = tileId( pos, nor );
    if( i > 0. ) {
        nor+= 0.0125 * vec3( hash(i), hash(i+5.), hash(i+13.) );
        nor = normalize( nor );
    }
    return nor;
}

// Function 108
vec3 calcNormal( in vec3 p ) {
    const float h = 1e-4;
    vec3 n = vec3(0.0);
    for(int i = min(iFrame,0); i<4; i++) {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(p+e*h).x;
    }
    return normalize(n);
}

// Function 109
vec2 normal(in vec2 p) {
    const vec2 NE = vec2(.1,0.);
    return normalize(vec2( height(p+NE)-height(p-NE),
                           height(p+NE.yx)-height(p-NE.yx) ));
}

// Function 110
vec3 Normal( in vec3 pos )
{
    vec2 eps = vec2(PRECISION, 0.0);
	return normalize( vec3(
           MapThorns(pos+eps.xyy) - MapThorns(pos-eps.xyy),
           MapThorns(pos+eps.yxy) - MapThorns(pos-eps.yxy),
           MapThorns(pos+eps.yyx) - MapThorns(pos-eps.yyx) ) );

}

// Function 111
vec3 getNormal(in vec3 pos, in Sphere sph)
{
    return normalize((pos - sph.cen) / sph.rad);
}

// Function 112
vec3 normal(in vec3 p, float ef) {
	vec2 e = vec2(.001*ef, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 113
vec3 getNormal(vec3 p, float d) {
    return normalize(vec3(
        getElevation(vec2(p.x-nEPS,p.z), d) - getElevation(vec2(p.x+nEPS,p.z), d),
        2.0*nEPS,
        getElevation(vec2(p.x,p.z-nEPS), d) - getElevation(vec2(p.x,p.z+nEPS), d)
    ));
}

// Function 114
vec4 GetNormal(vec3 u
){vec2 e = vec2(0.003,0.)
 ;float o=map(u,-1)
 ;return vec4(normalize(vec3(map(u+e.xyy,-1)
                            ,map(u+e.yxy,-1)
                            ,map(u+e.yyx,-1))-o),o);}

// Function 115
vec3 Scene_Normal(vec3 p, float r, inout RayHit hit)
{
    vec2 e = vec2(1.0, -1.0);
    
    vec2 r0 = Scene_SDF(p + e.xyy, hit);
    vec2 r1 = Scene_SDF(p + e.yyx, hit);
    vec2 r2 = Scene_SDF(p + e.yxy, hit);
    vec2 r3 = Scene_SDF(p + e.xxx, hit);
    
    vec3 norm = e.xyy * mix(r0.x, r0.y, r) + 
                e.yyx * mix(r1.x, r1.y, r) + 
                e.yxy * mix(r2.x, r2.y, r) + 
                e.xxx * mix(r3.x, r3.y, r);
    
    return normalize(norm);
}

// Function 116
vec3 surfaceNormal(vec3 pos) {
 	vec3 delta = vec3(0.01, 0.0, 0.0);
    vec3 normal;
    normal.x = combinedDistanceOnly(pos + delta.xyz) - combinedDistanceOnly(pos - delta.xyz);
    normal.y = combinedDistanceOnly(pos + delta.yxz) - combinedDistanceOnly(pos - delta.yxz);
    normal.z = combinedDistanceOnly(pos + delta.zyx) - combinedDistanceOnly(pos - delta.zyx);
    return normalize(normal);
}

// Function 117
vec3 normal(in vec3 p){

    // Note the slightly increased sampling distance, to alleviate artifacts due to hit point inaccuracies.
    vec2 e = vec2(0.005, -0.005); 
    return normalize(e.xyy * map(p + e.xyy) + e.yyx * map(p + e.yyx) + e.yxy * map(p + e.yxy) + e.xxx * map(p + e.xxx));
}

// Function 118
vec3 normal(vec3 p) {
	const vec2 eps = vec2(0.1, 0.0);
	float h = terrain(p.xz);
	return normalize(vec3(
		(terrain(p.xz+eps.xy)-h),
		eps.x,
		(terrain(p.xz+eps.yx)-h)
	));
}

// Function 119
vec3 calcNormal(vec3 ori, vec3 p) {
	vec2 e = vec2(.001,0.);
    vec3 n = vec3(dstScene(ori,p+e.xyy)-dstScene(ori,p-e.xyy),
                  dstScene(ori,p+e.yxy)-dstScene(ori,p-e.yxy),
                  dstScene(ori,p+e.yyx)-dstScene(ori,p-e.yyx));
    return normalize(n);
}

// Function 120
vec3 getNormal(vec3 p, float t, sphere s, float rnd)
{
	vec2 d = vec2(0.01, 0.0);
	float dx = distanceField(p + d.xyy,t,s,rnd)
				- distanceField(p - d.xyy,t,s,rnd);
	float dy = distanceField(p + d.yxy,t,s,rnd)
				- distanceField(p - d.yxy,t,s,rnd);
	float dz = distanceField(p + d.yyx,t,s,rnd)
				- distanceField(p - d.yyx,t,s,rnd);
	return normalize(vec3(dx, dy, dz));
}

// Function 121
vec3 calcNormal( in vec3 p, in float id ) {
        vec3 eps = vec3(0.002,0.0,0.0);

        vec3 nor = normalize( vec3(
            map( p + eps.xyy).x - map(p - eps.xyy).x,
            map( p + eps.yxy).x - map(p - eps.yxy).x,
            map( p + eps.yyx).x - map(p - eps.yyx).x));

        if (id == 2.0) {
            return doBumpMap(p,nor);
        }

        return nor;
    }

// Function 122
vec3 normal(in vec3 pos)
{
  vec3 eps = vec3(.001,0.0,0.0);
  vec3 nor;
  float ref;
  nor.x = distanceEstimator(pos+eps.xyy, ref) - distanceEstimator(pos-eps.xyy, ref);
  nor.y = distanceEstimator(pos+eps.yxy, ref) - distanceEstimator(pos-eps.yxy, ref);
  nor.z = distanceEstimator(pos+eps.yyx, ref) - distanceEstimator(pos-eps.yyx, ref);
  return normalize(nor);
}

// Function 123
vec3 normalAt( vec3 p) {
    vec3 e = vec3 (.001, -.001, 0); 
    return normalize(e.xyy * altitude(p + e.xyy)
                   + e.yyx * altitude(p + e.yyx)
                   + e.yxy * altitude(p + e.yxy)
                   + e.xxx * altitude(p + e.xxx));
}

// Function 124
vec3 normal(vec3 p)
{
	vec3 o = vec3(0.001, 0.0, 0.0);
    return normalize(vec3(map(p+o.xyy) - map(p-o.xyy),
                          map(p+o.yxy) - map(p-o.yxy),
                          map(p+o.yyx) - map(p-o.yyx)));
}

// Function 125
vec3 calcNormal(in vec3 pos, float eps){
    // Central differences approach.
    // 6s compile time because of over-eager compiler inlining.
    /*
    vec2 e = vec2(eps, 0.f);
    return normalize(vec3(map(pos + e.xyy).x - map(pos - e.xyy).x,
                          map(pos + e.yxy).x - map(pos - e.yxy).x,
                          map(pos + e.yyx).x - map(pos - e.yyx).x));
    */
    
    
    // Tetrahedron approach from https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm.
    vec3 n = vec3(0.0);
    for(int i=min(iFrame,0); i<4; i++) {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+eps*e).x;
    }
    return normalize(n);
    
    
    // Below this point are implementations of inlining-resistant central difference algorithms.
    // I started messing with them because the tetahedron approach was causing black artefacts
    // on the underside bridge corners. (But it turns out I had just accidentally set epsilon an
    // order of magnitude higher than intended in one of the calls.)
    // The below implementations aren't used any more, but I think they're interesting enough to keep.
    
    
    // Inlining-resistant central differences.
    // 3s compile time because two map calls get inlined.
    // Follows the sprit of the inlining-resistant tertrahedron approach, but I haven't seen this implementation elsewhere.
    /*
    vec3 n = vec3(0.0);
    for(int i=min(iFrame,0); i<3; i++) {
        vec3 e = vec3(((i+3)>>1)&1,(i>>1)&1,i&1);
        n += e*map(pos+eps*e).x - e*map(pos-eps*e).x;
    }
    return normalize(n);
    */


    // Inlining-resistant central differences mk. 2.
    // 2s compile time with the single inlined map call.
    // This should be equivalent to mk. 1, but it isn't. (I notice it as banding in the water reflection in the smooth part of the water cycle.) I'm not currently sure why.
    /*
    vec3 n = vec3(0.0);
    for(int i=min(iFrame,0); i<6; i++) {
        float signBit = float(i&1);
        int j = i >> 1;
        vec3 e = signBit * vec3(((j+3)>>1)&1,(j>>1)&1,j&1);
        n += e*map(pos+eps*e).x;
    }
    return normalize(n);
    */
    
    
    // It's worth noting that the central differences approaches aren't much more expensive at runtime than the tetrahedron approach.
    // (Not a lot of the total time is spent calculating normals.)
}

// Function 126
vec3 calcNormal( in vec3 pos )
{
    const float eps = 0.001;             // precision of the normal computation

    const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v2 = vec3(-1.0,-1.0, 1.0);
    const vec3 v3 = vec3(-1.0, 1.0,-1.0);
    const vec3 v4 = vec3( 1.0, 1.0, 1.0);

	return normalize( v1*doModel( pos + v1*eps ).x + 
					  v2*doModel( pos + v2*eps ).x + 
					  v3*doModel( pos + v3*eps ).x + 
					  v4*doModel( pos + v4*eps ).x );
}

// Function 127
vec3 surfaceNormal(vec2 coord, float delta)
{
	float diffX = marquetry(vec2(coord.x+diff, coord.y), delta).r - marquetry(vec2(coord.x-diff, coord.y), delta).r;
	float diffY = marquetry(vec2(coord.x, coord.y+diff), delta).r - marquetry(vec2(coord.x, coord.y-diff), delta).r;
	vec2 localDiff = vec2(diffX, diffY);
	localDiff *= -1.0;
	localDiff = (localDiff/2.0)+.5;
	float localDiffMag = length(localDiff);
	float z = sqrt(max(0.,1.0-pow(localDiffMag, 2.0)));
	return vec3(localDiff, z);
}

// Function 128
vec3 GetBoundaryNormal(vec3 ro, vec3 rd, vec3 sz) {
    vec3 m = 1. / rd,
        k = abs(m) * sz,
        a = -m * ro - k * .5, b = a + k;
    return -sign(rd) * step(a.yzx, a.xyz) * step(a.zxy, a.xyz);
}

// Function 129
vec3 getNormal( in vec3 pos )
{
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
    return normalize( e.xyy*distFunc( pos + e.xyy ) + 
					  e.yyx*distFunc( pos + e.yyx ) + 
					  e.yxy*distFunc( pos + e.yxy ) + 
					  e.xxx*distFunc( pos + e.xxx ) ) ;
}

// Function 130
vec3 getNormalLod(const float lod, const vec2 uv){
    
    vec2 om = exp2(lod) / iResolution.xy *0.5;
    vec3 L0 = toViewSpaceLOD(uv+vec2(-1,-1)*om,lod);
    vec3 L1 = toViewSpaceLOD(uv+vec2( 1,-1)*om,lod);
    vec3 L2 = toViewSpaceLOD(uv+vec2(-1, 1)*om,lod);
    vec3 L3 = toViewSpaceLOD(uv+vec2( 1, 1)*om,lod);
    
    return normalize( cross(
        L0 - L3,
        L1 - L2
    ));
}

// Function 131
vec3 calcNormal( in vec3 pos )
{    
  return normalize( vec3(MapTerrain(pos+eps.xyy) - MapTerrain(pos-eps.xyy), 0.5*2.0*eps.x, MapTerrain(pos+eps.yyx) - MapTerrain(pos-eps.yyx) ) );
}

// Function 132
vec3 calcNormal(vec3 pos)
{
    float eps = 0.0001;
	float d = map(pos);
	return normalize(vec3(map(pos+vec3(eps,0,0))-d,map(pos+vec3(0,eps,0))-d,map(pos+vec3(0,0,eps))-d));
}

// Function 133
vec3 terrainNormal( in vec2 pos )
{
#if 1
    return terrainMapD(pos).yzw;
#else    
    vec2 e = vec2(0.03,0.0);
	return normalize( vec3(terrainMap(pos-e.xy).x - terrainMap(pos+e.xy).x,
                           2.0*e.x,
                           terrainMap(pos-e.yx).x - terrainMap(pos+e.yx).x ) );
#endif    
}

// Function 134
vec3 blendNormals(in vec3 norm1, in vec3 norm2)
{
	return normalize(vec3(norm1.xy + norm2.xy, norm1.z));
}

// Function 135
vec3 calcNormal( in vec3 pos )
{
    vec2 e = vec2(1.0,-1.0)*0.5773*precis;
    return normalize( e.xyy*map( pos + e.xyy ).x + 
					  e.yyx*map( pos + e.yyx ).x + 
					  e.yxy*map( pos + e.yxy ).x + 
					  e.xxx*map( pos + e.xxx ).x );
}

// Function 136
vec3 getNormal( in vec3 p )
{
  vec2 e = vec2(0.005, -0.005);
  return normalize(
    e.xyy * Cube(p + e.xyy) +
    e.yyx * Cube(p + e.yyx) +
    e.yxy * Cube(p + e.yxy) +
    e.xxx * Cube(p + e.xxx));
}

// Function 137
vec3 normal(vec3 pos){
	const vec2 e = vec2(0.001,0.);
    return normalize(
    	vec3(
        	map(pos + e.xyy),
            map(pos + e.yxy),
            map(pos + e.yyx)
        )-map(pos)
    );
}

// Function 138
vec3 getNormal(vec3 p)
{
    float e = 0.03;
	float d = map(p);
    vec2 delta = vec2(e,0.);
    return normalize(
    	vec3(
        	map(p+vec3(delta.xyy))-d,
            map(p+vec3(delta.yxy))-d,
            map(p+vec3(delta.yyx))-d
        )
    );
}

// Function 139
vec3 NormalSinPowWarpOffset(vec3 pos, vec3 offset, float freq, float amp, float power) {
	return NormalSinPowWarp(pos + offset, freq, amp, power) - offset;
}

// Function 140
vec3 calcNormal( in vec3 pos ){
    
	vec3 eps = vec3( 0.01, 0.0, 0.0 );
	vec3 nor = vec3(
	    map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	    map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	    map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
}

// Function 141
vec3 normal(vec3 p)
{
	float c = scene(p).d;
	vec2 h = vec2(0.01, 0.0);
	return normalize(vec3(scene(p + h.xyy).d - c, 
						  scene(p + h.yxy).d - c, 
		                  scene(p + h.yyx).d - c));
}

// Function 142
vec3 Normal( vec3 pos, vec3 ray, float t) {

	float pitch = .2 * t / iResolution.x;
    
//#ifdef FAST
//	// don't sample smaller than the interpolation errors in Noise()
	pitch = max( pitch, .005 );
//#endif
	
	vec2 d = vec2(-1,1) * pitch;

	vec3 p0 = pos+d.xxx, // tetrahedral offsets
         p1 = pos+d.xyy,
         p2 = pos+d.yxy,
         p3 = pos+d.yyx;
	
	float f0 = map(p0),
	      f1 = map(p1),
	      f2 = map(p2),
	      f3 = map(p3);
	
	vec3 grad = p0*f0+p1*f1+p2*f2+p3*f3 - pos*(f0+f1+f2+f3);
	// prevent normals pointing away from camera (caused by precision errors)
	return normalize(grad - max(.0,dot (grad,ray ))*ray);
}

// Function 143
vec3 calcNormal(vec3 pos, float ep)
{
    return normalize(vec3(map(pos + vec3(ep, 0, 0)).dist - map(pos - vec3(ep, 0, 0)).dist,
                		map(pos + vec3(0, ep, 0)).dist - map(pos - vec3(0, ep, 0)).dist,
                		map(pos + vec3(0, 0, ep)).dist - map(pos - vec3(0, 0, ep)).dist));                                
}

// Function 144
vec3 normalForSphere(vec3 hit, Sphere s) {   
   return (hit - s.center_radius.xyz) / s.center_radius.w;   
}

// Function 145
vec3 getNormal(vec3 pos, float e, bool inside)
{  
    vec2 q = vec2(0, e);
    return normalize(vec3(map(pos + q.yxx, inside).x - map(pos - q.yxx, inside).x,
                          map(pos + q.xyx, inside).x - map(pos - q.xyx, inside).x,
                          map(pos + q.xxy, inside).x - map(pos - q.xxy, inside).x));
}

// Function 146
vec3 normal(in vec3 p, in vec3 ray, in float t) {
	float pitch = .4 * t / iResolution.x;
    vec2 d = vec2(-1,1) * pitch;
	vec3 p0 = p+d.xxx, p1 = p+d.xyy, p2 = p+d.yxy, p3 = p+d.yyx; // tetrahedral offsets
	float f0 = M(p0), f1 = M(p1), f2 = M(p2), f3 = M(p3);
	vec3 grad = p0*f0+p1*f1+p2*f2+p3*f3 - p*(f0+f1+f2+f3);
	// prevent normals pointing away from camera (caused by precision errors)
	return normalize(grad - max(.0,dot (grad,ray ))*ray);
}

// Function 147
vec3 calcNormal(vec3 p){
  const vec2 eps = vec2(0.0001, 0.0);
  // mathematical procedure.
  vec3 n;
  n.x = map(p + eps.xyy).w - map(p - eps.xyy).w;
  n.y = map(p + eps.yxy).w - map(p - eps.yxy).w;
  n.z = map(p + eps.yyx).w - map(p - eps.yyx).w;
  return normalize(n);
}

// Function 148
vec3 calcNormal( vec3 pos )
{
    float eps = 0.01; // precission
    float gradX = scene( pos + vec3(eps, 0.0, 0.0) ).x - scene( pos - vec3(eps, 0.0, 0.0)).x;
    float gradY = scene( pos + vec3(0.0, eps, 0.0) ).x - scene( pos - vec3(0.0, eps, 0.0)).x;
    float gradZ = scene( pos + vec3(0.0, 0.0, eps) ).x - scene( pos - vec3(0.0, 0.0, eps)).x;
    return normalize( vec3( gradX, gradY, gradZ ) );
}

// Function 149
vec3 normal(in vec3 p)
{
  //tetrahedron normal
  const float n_er=0.01;
  float v1=obj(vec3(p.x+n_er,p.y-n_er,p.z-n_er)).x;
  float v2=obj(vec3(p.x-n_er,p.y-n_er,p.z+n_er)).x;
  float v3=obj(vec3(p.x-n_er,p.y+n_er,p.z-n_er)).x;
  float v4=obj(vec3(p.x+n_er,p.y+n_er,p.z+n_er)).x;
  return normalize(vec3(v4+v1-v3-v2,v3+v4-v1-v2,v2+v4-v3-v1));
}

// Function 150
float normalDistribution1D(float x, float mean, float std_dev) {
    float xMinusMean = x - mean;
    float xMinusMeanSqr = xMinusMean * xMinusMean;
    return exp(-xMinusMeanSqr / (2. * std_dev * std_dev)) /
           (std_dev * 2.506628);
    // 2.506628 \approx sqrt(2 * \pi)
}

// Function 151
vec3 surfaceNormal (in vec3 pos) {
    vec3 normal = vec3(
        map(pos + xDir).x - map(pos - xDir).x,
        map(pos + yDir).x - map(pos - yDir).x,
        map(pos + zDir).x - map(pos - zDir).x
    );
    return normalize(normal);
}

// Function 152
vec3 getNormal(const in vec3 p){  
    vec2 e = vec2(-1., 1.)*0.005;   
	return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );}

// Function 153
vec4 normalShader(vec2 uv)
{
    return texture(iChannel0, uv);
}

// Function 154
float parametric_normal_iteration2(float t, vec2 uv){
	vec2 uv_to_p=parametric(t)-uv;
	vec2 tang=parametric_diff(t);
	vec2 snd_drv=parametric_snd_diff(t);

	float l_tang=dot(tang,tang);

	float fac=dot(tang,snd_drv)/(2.*l_tang);
	float d=-dot(tang,uv_to_p);

	float t2=d/(l_tang+fac*d);

	return t+factor*t2;
}

// Function 155
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

// Function 156
float pack_normal(vec2 n) {
    vec2 s = sign(n);
    return (n.y*s.x - s.y*(n.x + s.x - 2.0)) * 0.25;
}

// Function 157
vec3 calcNormal( in vec3 p, in vec4 c )
{
    vec4 z = vec4(p,0.0);

    // identity derivative
    vec4 J0 = vec4(1,0,0,0);
    vec4 J1 = vec4(0,1,0,0);
    vec4 J2 = vec4(0,0,1,0);
    
  	for(int i=0; i<numIterations; i++)
    {
        vec4 cz = qconj(z);
        
        // chain rule of jacobians (removed the 2 factor)
        J0 = vec4( dot(J0,cz), dot(J0.xy,z.yx), dot(J0.xz,z.zx), dot(J0.xw,z.wx) );
        J1 = vec4( dot(J1,cz), dot(J1.xy,z.yx), dot(J1.xz,z.zx), dot(J1.xw,z.wx) );
        J2 = vec4( dot(J2,cz), dot(J2.xy,z.yx), dot(J2.xz,z.zx), dot(J2.xw,z.wx) );

        // z -> z2 + c
        z = qsqr(z) + c; 
        
        if(qlength2(z)>4.0) break;
    }
    
	vec3 v = vec3( dot(J0,z), 
                   dot(J1,z), 
                   dot(J2,z) );

    return normalize( v );
}

// Function 158
vec3 getNormal(vec3 pos)
{
	return normalize(vec3( getDist(pos+vec3(EPSILON,0,0)), 
					getDist(pos+vec3(0,EPSILON,0)), 
					getDist(pos+vec3(0,0,EPSILON)))-getDist(pos));
}

// Function 159
vec4 GenerateNormalHeightMultipass (sampler2D tex, vec2 uv, vec2 res)
{
    float dist = 0.5;
    vec4 multi;
	multi += GenerateNormalHeight(tex, uv, res, 1. * dist);
	multi += GenerateNormalHeight(tex, uv, res, 2.5 * dist);
	multi += GenerateNormalHeight(tex, uv, res, 5. * dist);
	multi += GenerateNormalHeight(tex, uv, res, 10. * dist);    
    multi *= 0.25;
    return multi;
}

// Function 160
vec3 distMapNormal(vec3 p) {
    return normalize(vec3(
        distMap(vec3(p.x+NORMAL_EPILSON,p.y,p.z))-distMap(vec3(p.x-NORMAL_EPILSON,p.y,p.z)),
        
		distMap(vec3(p.x,p.y+NORMAL_EPILSON,p.z))-distMap(vec3(p.x,p.y-NORMAL_EPILSON,p.z)),
        
        distMap(vec3(p.x,p.y,p.z+NORMAL_EPILSON))-distMap(vec3(p.x,p.y,p.z-NORMAL_EPILSON))
        ));
}

// Function 161
vec3 calcNormal(vec3 p)
{
    float e = 0.3;
    
    vec3 normal;
    vec3 i;
    normal.x = sceneDf(vec3(p.x + e,p.y,p.z), i) - sceneDf(vec3(p.x - e, p.y, p.z), i);
    normal.y = sceneDf(vec3(p.x,p.y + e,p.z), i) - sceneDf(vec3(p.x, p.y - e, p.z), i);
    normal.z = sceneDf(vec3(p.x,p.y,p.z + e), i) - sceneDf(vec3(p.x, p.y, p.z - e), i);
    
    return normalize(normal);
}

// Function 162
vec3 calcNormal(vec3 p) {
	float eps = 0.0001;
	const vec3 v1 = vec3( 1.0,-1.0,-1.0);
	const vec3 v2 = vec3(-1.0,-1.0, 1.0);
	const vec3 v3 = vec3(-1.0, 1.0,-1.0);
	const vec3 v4 = vec3( 1.0, 1.0, 1.0);
	return normalize( v1 * map( p + v1*eps ) +
					  v2 * map( p + v2*eps ) +
					  v3 * map( p + v3*eps ) +
					  v4 * map( p + v4*eps ) );
}

// Function 163
vec3 tweakNormal(vec3 normal, float freq, float blending)
{
    vec2 uv = getUv(normal);
    float s = sin(uv.x * freq);
    float c = cos(uv.y * freq);
    normal.x += blending*s;
    normal.z += blending*c;
    return normalize(normal);
}

// Function 164
vec3 normal(vec2 p) {
	vec2 eps=vec2(0,res*.5);
	float d1=terrain(p+eps.xy)-terrain(p-eps.xy);
	float d2=terrain(p+eps.yx)-terrain(p-eps.yx);
	vec3 n1=(vec3(0.,eps.y*2.,d1));
	vec3 n2=(vec3(eps.y*2.,0.,d2));
	return normalize(cross(n1,n2));
}

// Function 165
vec3 getNormal(in vec3 p) {
    return normalize(vec3(mapScene(p + vec3(0.001, 0.0, 0.0)) - mapScene(p - vec3(0.001, 0.0, 0.0)),
                          mapScene(p + vec3(0.0, 0.001, 0.0)) - mapScene(p - vec3(0.0, 0.001, 0.0)),
                          mapScene(p + vec3(0.0, 0.0, 0.001)) - mapScene(p - vec3(0.0, 0.0, 0.001))));
}

// Function 166
vec3 computeNormalAndSnapPoint(inout vec3 pos) {

    float posLen = length(pos);
    vec2 latlon = _3DToLatLon(pos/posLen);
      
    const float latlonPixel = PIXEL_SIZE;
    
    vec2 discreteLatlon = discretize(latlon, latlonPixel);
    vec3 filteredPos = latLonTo3D(discreteLatlon + 0.5*latlonPixel);
    vec3 normal = normalize(filteredPos);
    
    vec3 nx = normalize(cross(vec3(0,1,0), normal));
    vec3 ny = cross(normal, nx);    

    vec2 dll = (latlon - discreteLatlon) / latlonPixel;
    
    bool TL = dll.x > dll.y;
    bool BL = dll.x < (1.0 - dll.y);
    
    if(TL) {
         if(BL) {
            latlon.y = latlon.y;
            
            normal = -nx;
         }
         else {
            latlon.x = discreteLatlon.x + latlonPixel;         
            normal = +ny;
         }
    }
    else {
         if(BL) {
            latlon.x = discreteLatlon.x;
            normal = -ny;
         }
         else {
                 latlon.y = discreteLatlon.y + latlonPixel;         
         
            normal = +nx;
         }
    }


    // snap
   //pos = latLonTo3D(latlon) * posLen;
    
    return normal;
}

// Function 167
float2 Normal_Sampler ( in sampler2D s, in float2 uv ) {
  float2 eps = float2(0.003, 0.0);
  return float2(length(texture(s, uv+eps.xy)) - length(texture(s, uv-eps.xy)),
                length(texture(s, uv+eps.yx)) - length(texture(s, uv-eps.yx)));
}

// Function 168
vec3 CalcNormal(vec3 p)
{
	vec2 e = vec2(1.0,-1.0) * 0.5773 * 0.0005;
    return normalize( e.xyy * SdScene( p + e.xyy ).x + 
					  e.yyx * SdScene( p + e.yyx ).x + 
					  e.yxy * SdScene( p + e.yxy ).x + 
					  e.xxx * SdScene( p + e.xxx ).x );
}

// Function 169
vec3 normal2(vec3 p){const vec2 e=vec2(.01,0);
 vec3 n=vec3(df(p+e.xyy)-df(p-e.xyy),df(p+e.yxy)-df(p-e.yxy),df(p+e.yyx)-df(p-e.yyx));
 if(length(n)<0.01)return vec3(0);//makes is easier to distinguish if the camera is far from any surface.
 return normalize(n);}

// Function 170
vec3 calcNormal( in vec3 pos, in vec2 tmat)
{
	return normalize(pos-mix(ballPos,vec3(0.0),tmat.y));
}

// Function 171
vec3 calcNormalForTriangle(vec3 a, vec3 b, vec3 c)
{
    vec3 dir = cross(b - a, c - a);
	vec3 normal = normalize(dir);
    return normal;
}

// Function 172
vec3 calcNormal( vec3 p ) 
{
    // We calculate the normal by finding the gradient of the field at the
    // point that we are interested in. We can find the gradient by getting
    // the difference in field at that point and a point slighttly away from it.
    const float h = 0.0001;
    return normalize( vec3(
        			       -distanceField(p)+ distanceField(p+vec3(h,0.0,0.0)),
                           -distanceField(p)+ distanceField(p+vec3(0.0,h,0.0)),
                           -distanceField(p)+ distanceField(p+vec3(0.0,0.0,h)) 
    				 ));
}

// Function 173
void GetBumpNormal(in vec3 pos, inout Material mat, int id)
{
    if(id != 1)
    {
        return;    // We only bump map the wood material
    }
    
    vec2 uv  = pos.xz * 0.25;
    vec2 eps = vec2(0.001, 0.0);

    float sampleU = texture(iChannel0, uv - eps.yx).r;
    float sampleD = texture(iChannel0, uv + eps.yx).r;
    float sampleL = texture(iChannel0, uv - eps.xy).r;
    float sampleR = texture(iChannel0, uv + eps.xy).r;

   	vec3 delta = vec3(
        (sampleL * sampleL - sampleR * sampleR), 
        0.0, 
        (sampleU * sampleU - sampleD * sampleD));

    mat.bump = normalize(mat.bump + (delta * 2.0));
}

// Function 174
vec3 mapRMWaterNormal(vec3 pt, float e) {
    vec3 normal;
    normal.y = sdPlane(pt)+waterDetails(pt, iTime);    
    normal.x = (sdPlane(pt)+waterDetails(vec3(pt.x+e,pt.y,pt.z), iTime)) - normal.y;
    normal.z = (sdPlane(pt)+waterDetails(vec3(pt.x,pt.y,pt.z+e), iTime)) - normal.y;
    normal.y = e;
    return normalize(normal);
}

// Function 175
vec3 calcNormal(in vec3 p){

    // Note the slightly increased sampling distance, to alleviate artifacts due to hit point inaccuracies.
    vec2 e = vec2(0.0025, -0.0025); 
    return normalize(e.xyy * map(p + e.xyy) + e.yyx * map(p + e.yyx) + e.yxy * map(p + e.yxy) + e.xxx * map(p + e.xxx));
}

// Function 176
vec3 calcnormal(vec3 p)
{
    vec2 e = vec2(0.001, 0.);
    return normalize(
        vec3(map(p+e.xyy).x-map(p-e.xyy).x,
        	 map(p+e.yxy).x-map(p-e.yxy).x,
        	 map(p+e.yyx).x-map(p-e.yyx).x)
    );
}

// Function 177
vec3 getNormal(in vec3 p, float t) {
	const vec2 e = vec2(.001, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), 
                          map(p + e.yxy) - map(p - e.yxy),	
                          map(p + e.yyx) - map(p - e.yyx)));
}

// Function 178
float ct_normal_pi(
    in vec2 z
){
    float a = atan(z[1], z[0]);
    if (a < 0.0) a += 6.28318;
    a /= 6.28318;
    return a;
}

// Function 179
vec3 Normal( in vec3 pos )
{
	vec2 eps = vec2( 0.01, 0.0);
	vec3 nor = vec3(
	    MapToScene(pos+eps.xyy) - MapToScene(pos-eps.xyy),
	    MapToScene(pos+eps.yxy) - MapToScene(pos-eps.yxy),
	    MapToScene(pos+eps.yyx) - MapToScene(pos-eps.yyx) );
	return normalize(nor);
}

// Function 180
vec3 getNormal(vec3 p)
{
    vec3 n = v0;
    for (int i=ZERO; i<4; i++)
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(p+e*0.0001);
    }
    return normalize(n);
}

// Function 181
vec3 GetNormal(vec3 p)
{
	float d = GetDst(p);
    vec2 e = vec2(0.01, 0);
    
    vec3 normal = d - vec3(
        GetDst(p - e.xyy),
        GetDst(p - e.yxy),
        GetDst(p - e.yyx));
    
    return normalize(normal);
}

// Function 182
vec3 triplanarNormal(vec3 p, vec3 nor, vec3 w) {
    // compute rotation matrices for the 3 normal maps
    vec3 xrY = cross(nor, vec3(0,1,0));
    vec3 xrX = cross(xrY, nor);
    mat3 xrot = mat3(xrX, sign(nor.x) * xrY, nor);

    vec3 yrY = cross(nor, vec3(0,0,1));
    vec3 yrX = cross(yrY, nor);
    mat3 yrot = mat3(yrX, sign(nor.y) * xrY, nor);

    vec3 zrY = cross(nor, vec3(1,0,0));
    vec3 zrX = cross(zrY, nor);
    mat3 zrot = mat3(zrX, sign(nor.z) * xrY, nor);

    vec3 tnor = vec3(0);
    tnor += w.x * xrot * normalmap(p.yz + 5.);
    tnor += w.y * yrot * normalmap(p.zx + vec2(9., 14.));
    tnor += w.z * zrot * normalmap(p.xy + vec2(12., 7.));
    tnor = normalize(tnor);
    
    return tnor;
}

// Function 183
vec3 calcNormal( in vec3 pos, in float eps )
{
#if 0    
    vec2 e = vec2(1.0,-1.0)*0.5773*eps;
    return normalize( e.xyy*map( pos + e.xyy, kk ).x + 
					  e.yyx*map( pos + e.yyx, kk ).x + 
					  e.yxy*map( pos + e.yxy, kk ).x + 
					  e.xxx*map( pos + e.xxx, kk ).x );
#else
    // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
    vec3 n = vec3(0.0);
    for( int i=0; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+e*eps).x;
    }
    return normalize(n);
#endif    
}

// Function 184
vec3 OceanNormal( vec3 pos )
{
	vec3 norm;
	vec2 d = vec2(.01*length(pos),0);
	
	norm.x = OceanDistanceFieldDetail( pos+d.xyy )-OceanDistanceFieldDetail( pos-d.xyy );
	norm.y = OceanDistanceFieldDetail( pos+d.yxy )-OceanDistanceFieldDetail( pos-d.yxy );
	norm.z = OceanDistanceFieldDetail( pos+d.yyx )-OceanDistanceFieldDetail( pos-d.yyx );

	return normalize(norm);
}

// Function 185
bool RayIntersectAABoxNormal (vec3 boxMin, vec3 boxMax, in vec3 rayPos, in vec3 rayDir, out vec3 hitPos, out vec3 normal, inout float maxTime)
{
    vec3 boxCenter = (boxMin+boxMax)*0.5;
	vec3 roo = rayPos - boxCenter;
    vec3 rad = (boxMax - boxMin)*0.5;

    vec3 m = 1.0/rayDir;
    vec3 n = m*roo;
    vec3 k = abs(m)*rad;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    vec2 time = vec2( max( max( t1.x, t1.y ), t1.z ),
                 min( min( t2.x, t2.y ), t2.z ) );
    
    // if the time is beyond the maximum allowed bail out (we hit somethign else first!)
    if (time.x > maxTime)
        return false;
    
    // if time invalid or we hit from inside, bail out
    if (time.y < time.x || time.x < 0.0)
        return false;
	
    // calculate surface normal
    hitPos = rayPos + rayDir * time.x;   
    vec3 hitPosRelative = hitPos - boxCenter;
    vec3 hitPosRelativeAbs = abs(hitPosRelative);
    vec3 distToEdge = abs(hitPosRelativeAbs - rad);

    float closestDist = 1000.0;
    for(int axis = 0; axis < 3; ++axis)
    {
        if (distToEdge[axis] < closestDist)
        {
            closestDist = distToEdge[axis];
            normal = vec3(0.0);
            if (hitPosRelative[axis] < 0.0)
                normal[axis] = -1.0;
            else
                normal[axis] = 1.0;
        }
    }        

    // store the collision time as the new max time
    maxTime = time.x;
    return true;
}

// Function 186
float distanceOnNormalizedAngle(float angle, float refPoin)
{
  
  float d =abs( angle - refPoin);
  if(d> 0.5) d = 1. - d;
  return d;
}

// Function 187
vec3 calcNormal( in vec3 pos )
{
    // a large epsilon is used here because the underlying data is coarse
    const float eps = 0.01;             // precision of the normal computation

    //const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v2 = vec3(-1.0,-1.0, 1.0);
    const vec3 v3 = vec3(-1.0, 1.0,-1.0);
    const vec3 v4 = vec3( 1.0, 1.0, 1.0);

	return normalize( v1*doModel( pos + v1*eps ).x + 
					  v2*doModel( pos + v2*eps ).x + 
					  v3*doModel( pos + v3*eps ).x + 
					  v4*doModel( pos + v4*eps ).x );
}

// Function 188
vec3 normal(vec3 p)
{
	const vec2 eps = vec2(0.01, 0);
	float nx = distanceField(p + eps.xyy) - distanceField(p - eps.xyy);
	float ny = distanceField(p + eps.yxy) - distanceField(p - eps.yxy);
	float nz = distanceField(p + eps.yyx) - distanceField(p - eps.yyx);
	return normalize(vec3(nx, ny, nz));
}

// Function 189
vec3 getNormal(vec3 p) {
    int id;
    return normalize(vec3(
        getSDF(vec3(p.x + EPSILON, p.y, p.z), id) - 
        getSDF(vec3(p.x - EPSILON, p.y, p.z), id),
        getSDF(vec3(p.x, p.y + EPSILON, p.z), id) - 
        getSDF(vec3(p.x, p.y - EPSILON, p.z), id),
        getSDF(vec3(p.x, p.y, p.z + EPSILON), id) - 
        getSDF(vec3(p.x, p.y, p.z - EPSILON), id)
    ));
}

// Function 190
vec3 hyperNormalizeP(vec3 u) {
    float s = u.z < 0. ? -1. : 1.;
    return u * (s / sqrt(-hyperDot(u, u)));
}

// Function 191
vec3 getNormal(vec3 p){
    
    float d = map(p);
    vec2 e = vec2(0.001,0.);
    
    vec3 n = d - vec3(
            map(p-e.xyy),
            map(p-e.yxy),
            map(p-e.yyx));
            
    return normalize(n);
}

// Function 192
vec2 sdSceneNormal(vec3 pos)
{
	return vec2(sdPlane(pos, -.75), 1.);
}

// Function 193
vec2 estimateNormal(vec2 p) {
    return normalize(vec2(
        mapSeed01(vec2(p.x + 1., p.y)) - mapSeed01(vec2(p.x - 1., p.y)),
        mapSeed01(vec2(p.x, p.y + 1.)) - mapSeed01(vec2(p.x, p.y - 1.))
    ));
}

// Function 194
vec3 calcNormal(in vec3 pos, in vec3 ray, in float t) {

	float pitch = .2 * t / iRes.x;
	pitch = max( pitch, .002 );
	
	vec2 d = vec2(-1,1) * pitch;

	vec3 p0 = pos+d.xxx, p1 = pos+d.xyy, p2 = pos+d.yxy, p3 = pos+d.yyx;
	float f0 = map(p0), f1 = map(p1), f2 = map(p2), f3 = map(p3);
	
	vec3 grad = p0*f0+p1*f1+p2*f2+p3*f3 - pos*(f0+f1+f2+f3);
	// prevent normals pointing away from camera (caused by precision errors)
	return normalize(grad - max(.0, dot (grad,ray))*ray);
}

// Function 195
vec3 GetNormal(vec3 p){
    //distance to point being analyzed
    float d = GetDist(p);
    
    //distance to another point along the objects surface that is closeby
    vec2 e = vec2(0.01,0);
    
    //slope between the two points
    //note: swizzel is the .xxy or .yyx etc
    vec3 n = d - vec3(
         GetDist(p-e.xyy),
         GetDist(p-e.yxy),
         GetDist(p-e.yyx));
         
    return normalize(n);
    
}

// Function 196
vec3 getNormal(vec3 p, float sphereR)
{
	vec2 j = vec2(sphereR, 0.0);
	vec3 nor  	= vec3(0.0,		terrainD(p.xz, sphereR), 0.0);
	vec3 v2		= nor-vec3(j.x,	terrainD(p.xz+j, sphereR), 0.0);
	vec3 v3		= nor-vec3(0.0,	terrainD(p.xz-j.yx, sphereR), -j.x);
	nor = cross(v2, v3);
	return normalize(nor);
}

// Function 197
vec3 getNormal(in vec3 p) {
	//as I explained in the bump function, it get's the gradient
    //around the point of interest, although here instead of a reference we
    //get the slope between a bit above to a bit below the point of interest, etc.
    //They are the slopes of distances so float values, and they are used to 
    //represent components of a vector3 and that vector can be used as the normal.
   
	const float eps = 0.001;
	return normalize(vec3(
		map(vec3(p.x+eps,p.y,p.z))-map(vec3(p.x-eps,p.y,p.z)),
		map(vec3(p.x,p.y+eps,p.z))-map(vec3(p.x,p.y-eps,p.z)),
		map(vec3(p.x,p.y,p.z+eps))-map(vec3(p.x,p.y,p.z-eps))
	));

}

// Function 198
vec3 SuperFastNormalFilter(sampler2D _tex,ivec2 iU,float strength){
    float p00 = GetTextureLuminance(_tex,iU);
    return normalize(vec3(-dFdx(p00),-dFdy(p00),1.-strength));
}

// Function 199
vec3 calcNormal(vec3 p) {
	vec2 e = vec2(.00005, -.00005);
	return normalize(e.xyy * map(p + e.xyy).x +
					 e.yyx * map(p + e.yyx).x +
					 e.yxy * map(p + e.yxy).x +
					 e.xxx * map(p + e.xxx).x);
}

// Function 200
vec3 calcNormalTransparent( in vec3 pos, in float eps )
{
    vec4 kk;
    vec2 e = vec2(1.0,-1.0)*0.5773*eps;
    return normalize( e.xyy*mapTransparent( pos + e.xyy, kk ).x + 
					  e.yyx*mapTransparent( pos + e.yyx, kk ).x + 
					  e.yxy*mapTransparent( pos + e.yxy, kk ).x + 
					  e.xxx*mapTransparent( pos + e.xxx, kk ).x );
}

// Function 201
vec3 normalMap( in vec2 pos )
{
	pos *= 2.0;
	
	float v = texture( iChannel3, 0.015*pos ).x;
	vec3 nor = vec3( texture( iChannel3, 0.015*pos+vec2(1.0/1024.0,0.0)).x - v,
	                 1.0/16.0,
	                 texture( iChannel3, 0.015*pos+vec2(0.0,1.0/1024.0)).x - v );
	nor.xz *= -1.0;
	return normalize( nor );
}

// Function 202
vec3 getNormal (vec3 p) {
    vec2 e = vec2(0.01, 0.0);
    float d = map(p);
    vec3 n = d - vec3( map(p-e.xyy) , map(p-e.yxy) , map(p-e.yyx) );
    return normalize(n);
}

// Function 203
vec3 NormalAtPos( vec2 p )
{
	float eps = 0.01;
    vec3 n = vec3( HeightAtPos(vec2(p.x-eps,p.y)) - HeightAtPos(vec2(p.x+eps,p.y)),
                         2.0*eps,
                         HeightAtPos(vec2(p.x,p.y-eps)) - HeightAtPos(vec2(p.x,p.y+eps)));
    return normalize( n );
}

// Function 204
vec3 getNormal(vec3 p){
    vec3 n = vec3(0.0);
    int id;
    for(int i = ZERO; i < 4; i++){
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*getSDF(p+e*EPSILON);
    }
    return normalize(n);
}

// Function 205
vec3 calcNormal_1245821463(vec3 pos, float eps) {
  const vec3 v1 = vec3( 1.0,-1.0,-1.0);
  const vec3 v2 = vec3(-1.0,-1.0, 1.0);
  const vec3 v3 = vec3(-1.0, 1.0,-1.0);
  const vec3 v4 = vec3( 1.0, 1.0, 1.0);

  return normalize( v1 * mapSolid( pos + v1*eps ).x +
                    v2 * mapSolid( pos + v2*eps ).x +
                    v3 * mapSolid( pos + v3*eps ).x +
                    v4 * mapSolid( pos + v4*eps ).x );
}

// Function 206
vec3 normal(vec3 p, int id)
{
    vec2 q = vec2(0,EPS);
    return normalize(vec3(map(p+q.yxx, id) - map(p-q.yxx, id),
                          map(p+q.xyx, id) - map(p-q.xyx, id),
                          map(p+q.xxy, id) - map(p-q.xxy, id)));
}

// Function 207
vec3 getNormal(vec3 p){
    vec2 e = vec2(0.0, 0.001);
    return normalize((vec3(map(p + e.yxx), map(p + e.xyx), map(p + e.xxy)) - map(p)) / e.y);
}

// Function 208
vec3 calcNormal( in vec3 pos, mat3 rotMat )
{
	vec3 eps = vec3( 0.001, 0.0, 0.0 );
	vec3 nor = vec3(
	    map(pos+eps.xyy, rotMat) - map(pos-eps.xyy, rotMat),
	    map(pos+eps.yxy, rotMat) - map(pos-eps.yxy, rotMat),
	    map(pos+eps.yyx, rotMat) - map(pos-eps.yyx, rotMat) );
	return normalize(nor);
}

// Function 209
vec3 computeSphereNormal(in Sphere s, in Ray r, in float dist) {
    return normalize((r.origin + r.dir * dist) - s.center);
}

// Function 210
void Vector2SphericalNormalized(vec3 v, inout float theta, inout float phi)
{
	theta = acos(v.y) / PI;
	phi = atan(v.x, v.z) / (2.0*PI);
	//phi = (phi < 0.0) ? (phi + 2.0*PI)/(2.0*PI) : (phi/(2.0*PI));
}

// Function 211
vec3 normal(vec3 pos) {
    
    float dx = dist(pos + EPS_VEC.xyy) - dist(pos - EPS_VEC.xyy);
    float dy = dist(pos + EPS_VEC.yxy) - dist(pos - EPS_VEC.yxy);
    float dz = dist(pos + EPS_VEC.yyx) - dist(pos - EPS_VEC.yyx);
    return normalize(vec3(dx,dy,dz));
}

// Function 212
float normalWeight(vec3 normal0, vec3 normal1) {
  const float exponent = 64.0;
  return pow(max(0.0, dot(normal0, normal1)), exponent);
}

// Function 213
vec3 normal(in vec3 p) { float d = df(p).dist; vec2 u = vec2(0.,.001); return normalize(vec3(df(p + u.yxx).dist,df(p + u.xyx).dist,df(p + u.xxy).dist) - d); }

// Function 214
vec3 get_normal(vec3 point) {
	float d0 = get_distance(point).x;
	float dX = get_distance(point-vec3(epsilon, 0.0, 0.0)).x;
	float dY = get_distance(point-vec3(0.0, epsilon, 0.0)).x;
	float dZ = get_distance(point-vec3(0.0, 0.0, epsilon)).x;
		
	return normalize(vec3(dX-d0, dY-d0, dZ-d0));
}

// Function 215
vec3 get_normal( in vec3 p ) {  // sample along 3 axes to get a normal  
    const float o = 0.0009765625; // 1 / 1024
    float c = f(p); // calculate redundantly for now to avoid the bug
    const vec2 h = vec2(o, 0); // transpose instead of making new per axis
    return normalize(vec3( f(p + h.xyy) - c,
                           f(p + h.yxy) - c,
                           f(p + h.yyx) - c) );
}

// Function 216
void transformNormal(inout vec3 normal, in mat4 matrix) {
  normal = normalize((matrix * vec4(normal, 0.0)).xyz);
}

// Function 217
vec3 sphNormal( in vec3 pos, in vec4 sph )
{
    return normalize(pos-sph.xyz);
}

// Function 218
vec3 getDetailNormal(vec3 p, vec3 normal, vec3 idx){
    vec3 tangent;
    vec3 bitangent;
    //Construct orthogonal directions tangent and bitangent to sample detail gradient in
    pixarONB(normal, tangent, bitangent);
    
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);
    
    float EPS = 1e-3;
    vec3 delTangent = 	getDetailExtrusion(p + tangent * EPS, normal, idx) - 
        				getDetailExtrusion(p - tangent * EPS, normal, idx);
    
    vec3 delBitangent = getDetailExtrusion(p + bitangent * EPS, normal, idx) - 
        				getDetailExtrusion(p - bitangent * EPS, normal, idx);
    
    return normalize(cross(delTangent, delBitangent));
}

// Function 219
vec3 normal(vec3 p)
{
	return normalize(vec3(
    	map(p + E.xyy) - map(p - E.xyy),
    	map(p + E.yxy) - map(p - E.yxy),
    	map(p + E.yyx) - map(p - E.yyx)
    ));
}

// Function 220
vec3 calcNormal( in vec3 pos, float e, in vec4 c )
{
    vec3 eps = vec3(e,0.0,0.0);

	return normalize( vec3(
           map(pos+eps.xyy,c).x - map(pos-eps.xyy,c).x,
           map(pos+eps.yxy,c).x - map(pos-eps.yxy,c).x,
           map(pos+eps.yyx,c).x - map(pos-eps.yyx,c).x ) );
}

// Function 221
vec3 tangent(vec3 N) 
{
    vec3 T, B;
	if(N.x == .0 && N.z == .0) T = vec3(0.,0.,.1);
    if(N.z == .0) T = vec3(0.,0.,-1.);
    else 
    {
        float l = sqrt(N.x*N.x+N.z*N.z);
    	T.x = N.z/l;
        T.y = .0;
        T.z = -N.x/l;
    }
    return T;
}

// Function 222
vec3 calcNormal( in vec3 p, in vec4 c )
{
    const vec2 e = vec2(0.001,0.0);
    vec4 za = vec4(p+e.xyy,0.0);
    vec4 zb = vec4(p-e.xyy,0.0);
    vec4 zc = vec4(p+e.yxy,0.0);
    vec4 zd = vec4(p-e.yxy,0.0);
    vec4 ze = vec4(p+e.yyx,0.0);
    vec4 zf = vec4(p-e.yyx,0.0);

  	for(int i=0; i<numIterations; i++)
    {
        za = qsqr(za) + c; 
        zb = qsqr(zb) + c; 
        zc = qsqr(zc) + c; 
        zd = qsqr(zd) + c; 
        ze = qsqr(ze) + c; 
        zf = qsqr(zf) + c; 
    }
    return normalize( vec3(log2(qlength2(za))-log2(qlength2(zb)),
                           log2(qlength2(zc))-log2(qlength2(zd)),
                           log2(qlength2(ze))-log2(qlength2(zf))) );

}

// Function 223
vec3 calculateNormal(in vec3 point)
{
    vec2 h = vec2(EPSILON, 0.0); // Some small value(s)
    return normalize(vec3(scene(point + h.xyy).x - scene(point - h.xyy).x,
                          scene(point + h.yxy).x - scene(point - h.yxy).x,
                          scene(point + h.yyx).x - scene(point - h.yyx).x));
}

// Function 224
vec3 NormalBlend_Whiteout(vec3 n1, vec3 n2)
{
    // Unpack
	n1 = n1*2.0 - 1.0;
    n2 = n2*2.0 - 1.0;
    
	return normalize(vec3(n1.xy + n2.xy, n1.z*n2.z));    
}

// Function 225
vec3 getNormal(vec2 uv, int tex ) {
#ifdef NORMAL_MAPS
    const float heightScale = 0.004;

    vec2 res = getTexRes(tex);
    vec2 duv = vec2(1.0) / res.xy;
    vec3 c  = getColor( uv, tex).xyz;
    vec3 c1 = getColor( uv + vec2(duv.x, 0.0), tex).xyz;
    vec3 c2 = getColor( uv - vec2(duv.x, 0.0), tex).xyz;
    vec3 c3 = getColor( uv + vec2(0.0, duv.y), tex).xyz;
    vec3 c4 = getColor( uv - vec2(0.0, duv.y), tex).xyz;
    
    float h0	= heightScale * dot(c , vec3(1.0/3.0));
    float hpx = heightScale * dot(c1, vec3(1.0/3.0));
    float hmx = heightScale * dot(c2, vec3(1.0/3.0));
    float hpy = heightScale * dot(c3, vec3(1.0/3.0));
    float hmy = heightScale * dot(c4, vec3(1.0/3.0));
    float dHdU = (hmx - hpx) / (2.0 * duv.x);
    float dHdV = (hmy - hpy) / (2.0 * duv.y);
    
    return normalize(vec3(dHdU, dHdV, 1.0));
#else
    return vec3(0.0, 0.0, 1.0);
#endif
}

// Function 226
vec3 calcNormal(vec3 p) {

	vec2 eps = vec2(.001,0.);
	vec3 n = vec3(distScene(p + eps.xyy).dst - distScene(p - eps.xyy).dst,
				  distScene(p + eps.yxy).dst - distScene(p - eps.yxy).dst,
				  distScene(p + eps.yyx).dst - distScene(p - eps.yyx).dst);
	return normalize(n);
	
}

// Function 227
float normalDistribution(in float x, in float sigma, in float mu) {
    const float SQRT_TWO_PI = 2.50662827463;
    float q = (x - mu) / sigma;
    return exp(-0.5 * q * q) / (sigma * SQRT_TWO_PI);
}

// Function 228
vec3 NormalBlend_UnpackedRNM(vec3 n1, vec3 n2)
{
	n1 += vec3(0, 0, 1);
	n2 *= vec3(-1, -1, 1);
	
    return n1*dot(n1, n2)/n1.z - n2;
}

// Function 229
vec3 Scene_GetNormal( vec3 pos )
{
    const float delta = 0.0001;
    
    vec4 samples;
    for( int i=ZERO; i<=4; i++ )
    {
        vec4 offset = vec4(0);
        offset[i] = delta;
        samples[i] = Scene_Distance( pos + offset.xyz );
    }
    
    vec3 normal = samples.xyz - samples.www;    
    return normalize( normal );
}

// Function 230
vec3 GetNormal(in vec3 p) {
    vec2 e = vec2(0.5, -0.5);
    return normalize(
        e.xyy * GetVolumeValue(p + e.xyy)
        + e.yyx * GetVolumeValue(p + e.yyx)
        + e.yxy * GetVolumeValue(p + e.yxy)
        + e.xxx * GetVolumeValue(p + e.xxx)
    );
}

// Function 231
vec3 normal(vec3 p, vec3 dist) 
{
    float eps = dot2(dist) * EPSILON_NRM;
    vec3 n;
    n.y = seaFragmentMap(p); 
    n = vec3(seaFragmentMap(vec3(p.x + eps, p.y, p.z)) - n.y,
	     	 seaFragmentMap(vec3(p.x, p.y, p.z + eps)) - n.y,
	     	 eps);
    return normalize(n);
}

// Function 232
vec3 normal(in vec3 at) {
	vec2 e = vec2(0., NORMAL_EPSILON);
	return normalize(vec3(world(at+e.yxx)-world(at), 
						  world(at+e.xyx)-world(at),
						  world(at+e.xxy)-world(at)));
}

// Function 233
vec3 getNormal(vec3 p) {
    int id;
    float dist = getDist(p, id);
    vec2 e = vec2(MIN_DIST, 0.);
    vec3 n = dist - vec3(
        getDist(p - e.xyy, id),
        getDist(p - e.yxy, id),
        getDist(p - e.yyx, id));
    return normalize(n);
}

// Function 234
vec3 calcNormal(vec3 pos) {
	vec2 eps = vec2(0.001, 0.0);

	vec3 nor = vec3(
			map(pos + eps.xyy).x - map(pos - eps.xyy).x,
			map(pos + eps.yxy).x - map(pos - eps.yxy).x,
			map(pos + eps.yyx).x - map(pos - eps.yyx).x);
	return normalize(nor);
}

// Function 235
vec2 directionalWaveNormal(vec2 p, float amp, vec2 dir, float freq, float speed, float time, float k)
{	
	float a = dot(p, dir) * freq + time * speed;
	float b = 0.5 * k * freq * amp * pow((sin(a) + 1.0) * 0.5, k) * cos(a);
	return vec2(dir.x * b, dir.y * b);
}

// Function 236
vec3 calcNormal( in vec3 pos )
{
	vec3 eps = vec3( 0.001, 0.0, 0.0 );
	vec3 nor = vec3(
	    fScene(pos+eps.xyy).x - fScene(pos-eps.xyy).x,
	    fScene(pos+eps.yxy).x - fScene(pos-eps.yxy).x,
	    fScene(pos+eps.yyx).x - fScene(pos-eps.yyx).x );
	return normalize(nor);
}

// Function 237
Quaternion H_normalize(Quaternion h)
{
    return normalize(h);
}

// Function 238
vec3 ComputeNormals(vec3 p)
{
    vec3 o;
    
    vec3 epsilonX = vec3(EPSILON, 0, 0);
    vec3 epsilonY = vec3(0, EPSILON, 0);
    vec3 epsilonZ = vec3(0, 0, EPSILON);
    
    // To estimate the normal in an axis, from a surface point, we move slightly
    // in that axis and get the changing in the distance to the surface itself.
    // If the change is 0 or really small it means the surface doesn't change in that
    // direction, so its normal in that point won't have that axis component.
    float reference = GetNearestShape(p).distance;
    o.x = GetNearestShape(p+epsilonX).distance - reference;
    o.y = GetNearestShape(p+epsilonY).distance - reference;
    o.z = GetNearestShape(p+epsilonZ).distance - reference;
    
    return normalize(o);
}

// Function 239
vec3 GetOpaqueNormal( in vec3 pos, int objectID )
{
    vec3 n = vec3(0.0);
    for( int i=min(0, iFrame); i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*QueryOpaqueDistanceField(pos+0.5*e, objectID);
    }
    return normalize(n);
}

// Function 240
vec3 calcNormal(vec3 p, float t) {
  // Try and scale epsilon for ray length.
  // This doesn't seem to work too well
  //float eps = t*0.0002;
  float eps = 0.002;
  vec2 e = vec2(eps, 0.0);
  return normalize(vec3(eval(p + e.xyy) - eval(p - e.xyy),
                        eval(p + e.yxy) - eval(p - e.yxy),
                        eval(p + e.yyx) - eval(p - e.yyx)));
}

// Function 241
vec3 hnormalizes(vec3 v){
  //normalization of "space like" vectors (not used).
  return v/hlengths(v);
}

// Function 242
vec3 calcNormal( in vec3 pos )
{
	vec3 eps = vec3( 0.001, 0.0, 0.0 );
    vec3 fuckyPos;
    float minBox, minSphere;
    vec3 adjusted;
    
	vec3 nor = vec3(
	    dist(pos+eps.xyy, fuckyPos, minBox, minSphere, adjusted).x - dist(pos-eps.xyy, fuckyPos, minBox, minSphere, adjusted).x,
	    dist(pos+eps.yxy, fuckyPos, minBox, minSphere, adjusted).x - dist(pos-eps.yxy, fuckyPos, minBox, minSphere, adjusted).x,
	    dist(pos+eps.yyx, fuckyPos, minBox, minSphere, adjusted).x - dist(pos-eps.yyx, fuckyPos, minBox, minSphere, adjusted).x );
	return normalize(nor);
}

// Function 243
vec3 get_normal(in vec3 p, in float t) {
	//https://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
    float h = 0.0002*t; 
    #define ZERO (min(iFrame,0))
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ ){
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(p+e*h).x;
    }
    return normalize(n);
}

// Function 244
vec3 getNormal(vec3 pos, float e)
{  
    vec3 n = vec3(0.0);
    for( int i=0; i<4; i++ )
    {
        vec3 e2 = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e2*map(pos + e*e2, true, false).x;
    }
    return normalize(n);
}

// Function 245
float TangentFunction(in float x, in float y, in float Time)
{
    return x*x - y*y - sin(Time*0.5)*Scale;
    //return y*y - x*x - sin(iTime)*Scale;//Get error if you use this
}

// Function 246
vec3 getNormal(vec3 pos)
{
	float d=getDist(pos);
	// Create the normal vector by comparing the distance near our point.
	return normalize(vec3( getDist(pos+vec3(EPSILON,0,0))-d, getDist(pos+vec3(0,EPSILON,0))-d, getDist(pos+vec3(0,0,EPSILON))-d ));
}

// Function 247
vec3 getNormalMesh(vec3 p){vec2 b=vec2(0,.00001);return Derivative9Tap(Dfm,p,b);}

// Function 248
vec3 ceilingnormal(vec3 p)
{
	float f0 = ceiling(p);    
	float fx = ceiling(p+vec3(0.1, 0.0, 0.0));    
	float fy = ceiling(p+vec3(0.0, 0.1, 0.0));    
	float fz = ceiling(p+vec3(0.0, 0.0, 0.1));
    vec3 norm = vec3(fx-f0, fy-f0, fz-f0);
    norm = normalize(norm);
    return norm;
}

// Function 249
vec3 normal(vec3 pos)
{
    vec2 eps = vec2(0.001, 0.0);
    return normalize(vec3(	map(pos + eps.xyy).x - map(pos - eps.xyy).x,
                    		map(pos + eps.yxy).x - map(pos - eps.yxy).x,
                         	map(pos + eps.yyx).x - map(pos - eps.yyx).x));
}

// Function 250
vec3 calcNormalWater( in vec3 pos )
{
    const vec3 eps = vec3(0.025,0.0,0.0);
    float v = sdWater(pos).x;	
	return normalize( vec3( sdWater(pos+eps.xyy).x - v,
                            eps.x,
                            sdWater(pos+eps.yyx).x - v ) );
}

// Function 251
vec3 AABoxNormal(vec3 bmin, vec3 bmax, vec3 p) 
{
    vec3 n1 = -(1.0 - smoothstep(0.0, 0.03, p - bmin));
    vec3 n2 = (1.0 -  smoothstep(0.0, 0.03, bmax - p));
    
    return normalize(n1 + n2);
}

// Function 252
vec3 CalcNormal(Rectangle A)
{
 vec3 first = A.v1 - A.v2;
 vec3 second = A.v2 - A.v3;
    
 return cross(first,second);
}

// Function 253
vec3 boxNormal( vec3 direction, vec3 point, float radius )
{
    vec3 n = point / direction;
    vec3 s = sign(direction);
    vec3 k = s * radius / direction;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    vec3 normal = -s * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
    
    return normal;
}

// Function 254
vec4 select_plane_normal(vec4 b) {
    float lc = min(min(b.x, b.y), min(b.z, b.w));
    return step(b, vec4(lc));
}

// Function 255
vec3 normal(vec3 p){
	vec2 offset = vec2(0.,0.01);
	vec3 nDir = vec3(
		M(p+offset.yxx),
		M(p+offset.xyx),
		M(p+offset.xxy)
	)-M(p);
	return normalize(nDir);
}

// Function 256
vec3 calcNormal(vec3 pos)
{
    float eps=0.0001;
	float d=map(pos);
	return normalize(vec3(map(pos+vec3(eps,0,0))-d,map(pos+vec3(0,eps,0))-d,map(pos+vec3(0,0,eps))-d));
}

// Function 257
vec3 computeNormal(inout vec3 pos) {
    vec2 latlon = _3DToLatLon(normalize(pos));
    
    const float latlonPixel = PIXEL_SIZE;
    
    vec2 discreteLatlon = discretize(latlon, latlonPixel);
    vec3 filteredPos = latLonTo3D(discreteLatlon + 0.5*latlonPixel);
    
    return normalize(filteredPos);
}

// Function 258
vec2 unpack_normal(float x) {
    float si = fract(x + 0.5)*2.0 - 1.0;
    float cx = 20.0 / (si*si + 4.0) - 4.0;
    float cy = sqrt(1.0 - cx*cx);
    return vec2(cx,cy) * sign(0.5-fract(vec2(0.25,0.0) + x*0.5));
}

// Function 259
vec3 normal(in vec3 p)
{  
    vec2 e = vec2(-1., 1.)*0.001;   
	return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
					 e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );   
}

// Function 260
vec3 calcNormal( in vec3 pos, float t )
{
	float e = 0.001;
	e = 0.0001*t;
    vec3  eps = vec3(e,0.0,0.0);
    vec3 nor;
    nor.x = map(pos+eps.xyy) - map(pos-eps.xyy);
    nor.y = map(pos+eps.yxy) - map(pos-eps.yxy);
    nor.z = map(pos+eps.yyx) - map(pos-eps.yyx);
    return normalize(nor);
}

// Function 261
vec3 Normals (in vec2 UV, in float Strength, in float T, in float MipLvl, in sampler2D height)
{
    float dx = -Strength*(texture(height,UV+vec2(T,0.0),MipLvl).x - texture(height,UV-vec2(T,0.0),MipLvl).x);
    
    float dy = -Strength*(texture(height,UV+vec2(0.0,T),MipLvl).x - texture(height,UV-vec2(0.0,T),MipLvl).x);    
    
    vec3 Normal = normalize(vec3(dx,dy,sqrt(clamp(1.0-dx*dx-dy*dy,0.0,1.0))));

    return Normal;   
}

// Function 262
vec2 stdNormalMap(in vec2 uv) 
{
    float height = texture(heightMap, uv).r;
    return -vec2(dFdx(height), dFdy(height)) * pixelToTexelRatio;
}

// Function 263
vec3 getnormal(vec3 n){
    vec2 e = vec2(1.0,-1.0)*(n.y<0.002-1.?.002:.002);
    return normalize(e.xyy*de(n+e.xyy)+e.yyx*de(n+e.yyx)+e.yxy*de(n+e.yxy)+e.xxx*de(n+e.xxx));
}

// Function 264
vec3 normalHighDetailModel(vec3 P)
{
	vec2 eps = vec2(0.,0.001);
    return normalize(vec3(
        SD_HighDetailModel(P+eps.yxx).d - SD_HighDetailModel(P-eps.yxx).d, 
		SD_HighDetailModel(P+eps.xyx).d - SD_HighDetailModel(P-eps.xyx).d, 
        SD_HighDetailModel(P+eps.xxy).d - SD_HighDetailModel(P-eps.xxy).d));
}

// Function 265
vec3 normal(in vec3 p, in vec3 rd)
{  
    vec2 e = vec2(-1., 1.)*0.01;   
	vec3 n = (e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
					 e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );
    
    //from TekF (error checking)
	float gdr = dot (n, rd );
	n -= max(.0,gdr)*rd;
    return normalize(n);
}

// Function 266
vec3 genNormal(vec3 p){
//	vec3 normal=vec3(
//		sdSphere(p+vec3(EPS,0.0,0.0))-sdSphere(p+vec3(-EPS,0.0,0.0)),
//		sdSphere(p+vec3(0.0,EPS,0.0))-sdSphere(p+vec3(0.0,-EPS,0.0)),
//		sdSphere(p+vec3(0.0,0.0,EPS))-sdSphere(p+vec3(0.0,0.0,-EPS))
//		);
//	return normalize(normal);
//}

// Function 267
vec3 terrainCalcNormalMed( in vec3 pos, float t ) {
	float e = 0.005*t;
    vec2  eps = vec2(e,0.0);
    float h = terrainMed( pos.xz );
    return normalize(vec3( terrainMed(pos.xz-eps.xy)-h, e, terrainMed(pos.xz-eps.yx)-h ));
}

// Function 268
vec3 hnormalizes(vec3 v){//normalization of "space like" vectors.(not used)
	float l=1./hlengths(v);
	return v*l;
}

// Function 269
vec3 normal(vec3 p) {
    const vec2 NE = vec2(MIN_EPSILON, 0.);
    return normalize(vec3(scene(p+NE.xyy,MIN_EPSILON).x-scene(p-NE.xyy,MIN_EPSILON).x,
                          scene(p+NE.yxy,MIN_EPSILON).x-scene(p-NE.yxy,MIN_EPSILON).x,
                          scene(p+NE.yyx,MIN_EPSILON).x-scene(p-NE.yyx,MIN_EPSILON).x));
}

// Function 270
vec4 BakeNormalTangentSpace(float theta, float phi) 
{   
    vec3 dir = vec3(sin(theta)*sin(phi), cos(phi), -sin(phi)*cos(theta));
    vec3 ro = dir*10.;	// The ray starts outside the models
    //vec3 ro = vec3(0.,0.,0.);
    Hit hP = RayMarchHighDetailModel(ro, -dir);
    Hit lP = RayMarchLowDetailModel(ro, -dir);

    vec3 N_ts; 
    mat3 M_ts_ws = mat3(lP.binormal,lP.tangent,lP.normal);
    N_ts = inverse(M_ts_ws)*hP.normal;
    return vec4(normalize(N_ts) * 0.5 + 0.5, .1);
}

// Function 271
vec3 get_water_normal(vec3 water_plane_point_of_intersection)
{
    vec3 noise = texture(iChannel3, water_plane_point_of_intersection.xz * 0.2).rgb;	// TODO: Why can't we use "water_plane_point_of_intersection"???
    
    // TODO: This time-based code has numerical precision issues:
    float water_time = iTime * 9.0;
    
    vec3 offset;
    offset.x = sin(water_time + noise.r * two_pi);
    offset.y = sin(water_time + noise.g * two_pi);
    offset.z = sin(water_time + noise.b * two_pi);
            
    return(normalize(offset + vec3(0.0, 1.0 / 0.03, 0.0)));
}

// Function 272
vec3 getNormal( in vec3 p ){

    // Note the slightly increased sampling distance, to alleviate
    // artifacts due to hit point inaccuracies.
    vec2 e = vec2(0.0025, -0.0025); 
    return normalize(
        e.xyy * map(p + e.xyy) + 
        e.yyx * map(p + e.yyx) + 
        e.yxy * map(p + e.yxy) + 
        e.xxx * map(p + e.xxx));
}

// Function 273
vec3 getNormal(vec3 p, float t){
    float eps = 0.001 * t;

    return normalize(vec3( 
        getHeight(vec3(p.x-eps, p.y, p.z), normalLimit) 
        - getHeight(vec3(p.x+eps, p.y, p.z), normalLimit),
        
        2.0*eps,
        
        getHeight(vec3(p.x, p.y, p.z-eps), normalLimit) 
        - getHeight(vec3(p.x, p.y, p.z+eps), normalLimit) 
    ));
}

// Function 274
vec3 SceneNormal( in vec3 pos, mat3 localToWorld )
{
	vec3 eps = vec3( 0.001, 0.0, 0.0 );
	vec3 nor = vec3(
	    Scene( pos + eps.xyy, localToWorld ) - Scene( pos - eps.xyy, localToWorld ),
	    Scene( pos + eps.yxy, localToWorld ) - Scene( pos - eps.yxy, localToWorld ),
	    Scene( pos + eps.yyx, localToWorld ) - Scene( pos - eps.yyx, localToWorld ) );
	return normalize( -nor );
}

// Function 275
vec3 compute_normal(vec3 pos, float density)
{
	
    float eps = rayStep*2.0;
    vec3 n;
	
    n.x = density_function( vec3(pos.x+eps, pos.y, pos.z) ) - density;
    n.y = density_function( vec3(pos.x, pos.y+eps, pos.z) ) - density;
    n.z = density_function( vec3(pos.x, pos.y, pos.z+eps) ) - density;
    return normalize(n);
}

// Function 276
vec3 getNormal(vec3 pos, float e, float o)
{
    vec2 q = vec2(0., e); //vec2(0.,distance(campos, pos)*0.0005);
    return normalize(vec3(map(pos + q.yxx) - map(pos - q.yxx),
                          map(pos + q.xyx) - map(pos - q.xyx),
                          map(pos + q.xxy) - map(pos - q.xxy)));
}

// Function 277
vec3 getNormal( in vec3 p ){

    vec2 e = vec2(0.5773,-0.5773)*EPS;   //0.001;
    return normalize( e.xyy*map(p+e.xyy ) + e.yyx*map(p+e.yyx ) + 
                      e.yxy*map(p+e.yxy ) + e.xxx*map(p+e.xxx ));
}

// Function 278
vec3 combineNormals0(vec3 n0, vec3 n1) {
    n0 = n0 * 2.0 - 1.0;
    n1 = n1 * 2.0 - 1.0;
    return normalize(n0 + n1) * 0.5 + 0.5;
}

// Function 279
vec3 normal(vec3 p)
{
    vec3 n;
    
    n.x = map(p + vec3(EPS,0.,0.)) - map(p-vec3(EPS,0.,0.));
    n.y = map(p + vec3(0.,EPS,0.)) - map(p-vec3(0.,EPS,0.));
    n.z = map(p + vec3(0.,0.,EPS)) - map(p-vec3(0.,0.,EPS));   
    
    return normalize(n);
}

// Function 280
vec3 calcNormal(in vec3 pos, float camtime)
{
  vec2 eps = vec2(0.002, 0.0);
  vec3 nor;
  nor.x = map(pos+eps.xyy,camtime).x - map(pos-eps.xyy,camtime).x;
  nor.y = map(pos+eps.yxy,camtime).x - map(pos-eps.yxy,camtime).x;
  nor.z = map(pos+eps.yyx,camtime).x - map(pos-eps.yyx,camtime).x;
  return normalize(nor);
}

// Function 281
vec2 tangent_sin(vec2 t) {
    // for
    // sin(a*x + t) = sin(t) + K*a*x
    // cos(a*x + t) = K*a
    // an acceptable approximation for K from t is
    // mix(cos(x%(PI*2)),pow(cos((x%(PI*2)-PI*0.5)/3.0 + PI*0.5),2),step(PI*0.5,x%(PI*2)))
    const float pi = 3.14159265359;
    t = mod(t, 2.0*pi);
    vec2 s = cos((t-pi*0.5)/3.0 + pi*0.5);
    return mix(cos(t),s*s,step(pi*0.5,t));
}

// Function 282
vec3 getNormal (vec3 p) {
  vec2 e = .01 * vec2(-1., 1.);
  vec3 nor = e.xyy*map(p+e.xyy) +
    e.yxy*map(p+e.yxy) +
    e.yyx*map(p+e.yyx);// +
    //e.xxx*map(p+e.xxx); // comment to get white aura
    
  return normalize(nor);
}

// Function 283
vec3 calcNormal(vec3 p) {
  vec3 eps = vec3(1e-4,0,0);
  vec3 n = vec3(map(p + eps.xyy) - map(p - eps.xyy),
                map(p + eps.yxy) - map(p - eps.yxy),
                map(p + eps.yyx) - map(p - eps.yyx));
  return normalize(n);
}

// Function 284
vec3 blend_normal(vec2 uv, vec3 c1, vec3 c2, float opacity) {
	return opacity*c1 + (1.0-opacity)*c2;
}

// Function 285
vec3 calcNormal( in vec3 pos, float t ) 
{
    vec2 e = vec2(0.001, 0.0);
    return normalize( vec3(map(pos+e.xyy,t).x-map(pos-e.xyy,t).x,
                           map(pos+e.yxy,t).x-map(pos-e.yxy,t).x,
                           map(pos+e.yyx,t).x-map(pos-e.yyx,t).x ) );
}

// Function 286
vec3 get_normal(int v) {
   int xyz = get_data(normals_offset, v);
   ivec3 XYZ = (ivec3(xyz) >> ivec3(0,10,20)) & ivec3(1023); 
   return vec3(-1) + vec3(XYZ) / 512.0;    
}

// Function 287
float cotangent(vec3 a, vec3 b, vec3 c) {
    vec3 ba = a - b;
    vec3 bc = c - b;
    return dot(bc, ba) / length(cross(bc, ba));
}

// Function 288
vec3 getNormal(vec3 p, inout float edge, inout float crv) { 
	
    // Roughly two pixel edge spread, regardless of resolution.
    vec2 e = vec2(6./iResolution.y, 0);

	float d1 = map(p + e.xyy), d2 = map(p - e.xyy);
	float d3 = map(p + e.yxy), d4 = map(p - e.yxy);
	float d5 = map(p + e.yyx), d6 = map(p - e.yyx);
	float d = map(p)*2.;

    edge = abs(d1 + d2 - d) + abs(d3 + d4 - d) + abs(d5 + d6 - d);
    //edge = abs(d1 + d2 + d3 + d4 + d5 + d6 - d*3.);
    edge = smoothstep(0., 1., sqrt(edge/e.x*2.));
/*    
    // Wider sample spread for the curvature.
    e = vec2(12./450., 0);
	d1 = map(p + e.xyy), d2 = map(p - e.xyy);
	d3 = map(p + e.yxy), d4 = map(p - e.yxy);
	d5 = map(p + e.yyx), d6 = map(p - e.yyx);
    crv = clamp((d1 + d2 + d3 + d4 + d5 + d6 - d*3.)*32. + .5, 0., 1.);
*/
    
    e = vec2(.0015, 0); //iResolution.y - Depending how you want different resolutions to look.
	d1 = map(p + e.xyy), d2 = map(p - e.xyy);
	d3 = map(p + e.yxy), d4 = map(p - e.yxy);
	d5 = map(p + e.yyx), d6 = map(p - e.yyx);
	
    return normalize(vec3(d1 - d2, d3 - d4, d5 - d6));
}

// Function 289
vec3 sdfNormal(vec3 p, float epsilon)
{
    vec3 eps = vec3(epsilon, -epsilon, 0.0);
    
	float dX = sdf_complex(p + eps.xzz) - sdf_complex(p + eps.yzz);
	float dY = sdf_complex(p + eps.zxz) - sdf_complex(p + eps.zyz);
	float dZ = sdf_complex(p + eps.zzx) - sdf_complex(p + eps.zzy); 

	return normalize(vec3(dX,dY,dZ));
}

// Function 290
mat3 mat3FromNormal(in vec3 n) {
    vec3 x;
    vec3 y;
    basis(n, x, y);
    return mat3(x,y,n);
}

// Function 291
vec3 NormalBlend_UDN(vec3 n1, vec3 n2)
{
    // Unpack
	n1 = n1*2.0 - 1.0;
    n2 = n2*2.0 - 1.0;    
    
	return normalize(vec3(n1.xy + n2.xy, n1.z));
}

// Function 292
vec3 getNormal(vec3 pos, float e)
{
    vec2 q = vec2(0, e);
    return normalize(vec3(map(pos + q.yxx) - map(pos - q.yxx),
                          map(pos + q.xyx) - map(pos - q.xyx),
                          map(pos + q.xxy) - map(pos - q.xxy)));
}

// Function 293
vec3 hnormalizet(vec3 v){//normalization of "time like" vectors.
	float l=1./hlengtht(v);
	return v*l;
}

// Function 294
vec3 calcNormal( in vec3 pos )
{
    vec2 eps = vec2(0.001,0.0);

	return normalize( vec3(
           map(pos+eps.xyy).x - map(pos-eps.xyy).x,
           map(pos+eps.yxy).x - map(pos-eps.yxy).x,
           map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
}

// Function 295
vec3 calculateFloorNormal(const vec3 base ) {
	vec3 A = intersectFloor(base);
	vec3 B = intersectFloor(vec3(base.x + NORM_SEARCH_DIST, base.yz));
	vec3 C = intersectFloor(vec3(base.xy, base.z + NORM_SEARCH_DIST));
	return normalize(cross(C - A, B - A));
}

// Function 296
vec3 calcNormal( in vec3 pos )
{    
  return normalize( vec3(MapPlane(pos+eps.xyy) - MapPlane(pos-eps.xyy), 0.5*2.0*eps.x, MapPlane(pos+eps.yyx) - MapPlane(pos-eps.yyx) ) );
}

// Function 297
vec3 normal(vec3 sp)
{///had to adjust the normal cause I was getting these weird lines on edges.
    vec3 eps = vec3(.0014, 0.0, 0.0);
    
    vec3 normal = normalize (vec3( map(sp+eps) - map(sp-eps)
                       ,map(sp+eps.yxz) - map(sp-eps.yxz)
                       ,map(sp+eps.yzx) - map(sp-eps.yzx) ));
    
    
 return normal;   
}

// Function 298
vec3 calcNormal( in vec3 pos, in float t )
{
    vec2 e = vec2(1.0,-1.0)*surface*t;
    return normalize( e.xyy*map( pos + e.xyy ).x + 
					  e.yyx*map( pos + e.yyx ).x + 
					  e.yxy*map( pos + e.yxy ).x + 
					  e.xxx*map( pos + e.xxx ).x );
}

// Function 299
void normal(inout ray _r)
{
    vec2 eps = vec2(.001,.0);
    float dx = map(_r.hp + eps.xyy).x - map(_r.hp - eps.xyy).x;
    float dy = map(_r.hp + eps.yxy).x - map(_r.hp - eps.yxy).x;
    float dz = map(_r.hp + eps.yyx).x - map(_r.hp - eps.yyx).x; 
    _r.n = normalize(vec3(dx,dy,dz));
}

// Function 300
vec3 calcNormal( in vec3 pos )
{
    // from Paul Malin (4 samples only in a tetrahedron	
    vec2 e = vec2(1.0,-1.0)*0.002;
    return normalize( e.xyy*map( pos + e.xyy ) + 
					  e.yyx*map( pos + e.yyx ) + 
					  e.yxy*map( pos + e.yxy ) + 
					  e.xxx*map( pos + e.xxx ) );
}

// Function 301
vec3 calcNormal(mat3 camMat , vec2 p, vec3 ro  ){
    
    // create view ray
    vec2 eps = vec2( 0.0001, 0.0 );
    
	vec3 l = normalize( camMat * vec3(p.xy - eps.xy,2.0) ); 
    vec3 r = normalize( camMat * vec3(p.xy + eps.xy,2.0) ); 
    vec3 u = normalize( camMat * vec3(p.xy - eps.yx,2.0) ); 
    vec3 d = normalize( camMat * vec3(p.xy + eps.yx,2.0) ); 
    
    // raycast the scene
	float hL = intersect(ro,l);
	vec3 pL = ro + hL * l;
    if( hL == INF ){ return vec3( INF ); }
        
    
     // raycast the scene
	float hR = intersect(ro,r);
	vec3 pR = ro + hR * r;
    if( hR == INF ){ return vec3( INF ); }
    
    // raycast the scene
	float hU = intersect(ro,u);
	vec3 pU = ro + hU * u;
    if( hU == INF ){ return vec3( INF ); }
    
    // raycast the scene
	float hD = intersect(ro,d);
	vec3 pD = ro + hD * d;
    if( hD == INF ){ return vec3( INF ); }
    
    vec3 d1 = pL - pR;
    vec3 d2 = pU - pD;
        
    vec3 nor =cross( d1 , d2 );

	return normalize(nor);
    
}

// Function 302
vec3 calcNormal( in vec3 pos, in float ep )
{
    vec4 kk;
#if 1
    vec2 e = vec2(1.0,-1.0)*0.5773;
    return normalize( e.xyy*map( pos + e.xyy*ep, kk).x + 
					  e.yyx*map( pos + e.yyx*ep, kk).x + 
					  e.yxy*map( pos + e.yxy*ep, kk).x + 
					  e.xxx*map( pos + e.xxx*ep, kk).x );
#else
    // prevent the compiler from inlining map() 4 times
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+e*ep, kk).x;
    }
    return normalize(n);
#endif    
    
}

// Function 303
vec3 getNormal(in vec3 pos) {
	vec2 d = vec2(EPS, 0.0);
	return normalize(vec3(map(pos + d.xyy) - map(pos - d.xyy),
		map(pos + d.yxy) - map(pos - d.yxy),
		map(pos + d.yyx) - map(pos - d.yyx)));
}

// Function 304
vec2 normalPack(vec3 n)
{
    return n.xy*0.5+0.5;
}

// Function 305
vec3 normal(const in vec3 p){vec2 e=vec2(-1.,1.)*eps*2.;
 return normalize(e.yxx*df(p+e.yxx)+e.xxy*df(p+e.xxy)+e.xyx*df(p+e.xyx)+e.yyy*df(p+e.yyy));}

// Function 306
vec3 calcNormal(in vec3 p, int geomId, in float t) {
    vec3 e = vec3(0.001, 0.0, 0.0)*t;
    vec3 n;
	
    if (geomId == 1) {
        //vec2 n1 = fbm(p.xz*.05, 4).yz;
        //n = normalize(vec3(n1.x, 0.001*2., n1.y));
		
        n.x = getMGeomH(p+e.xyy) - getMGeomH(p-e.xyy);
        n.y = 2.*e.x;
        n.z = getMGeomH(p+e.yyx) - getMGeomH(p-e.yyx);
    } else if (geomId == 2) {
        //n = vec3(0., 1., 0.);

        n.x = getWGeomH(p+e.xyy) - getWGeomH(p-e.xyy);
        n.y = 2.*e.x;
        n.z = getWGeomH(p+e.yyx) - getWGeomH(p-e.yyx);

    } else if (geomId == 3) {
        n.x = getSBird(p+e.xyy) - getSBird(p-e.xyy);
        n.y = getSBird(p+e.yxy) - getSBird(p-e.yxy);
        n.z = getSBird(p+e.yyx) - getSBird(p-e.yyx);
    }

    return normalize(n);
}

// Function 307
vec3 getNormal(vec3 p, inout float edge, inout float crv, float ef) { 
	
    // Roughly two pixel edge spread, but increased slightly with larger resolution.
    vec2 e = vec2(ef/mix(450., iResolution.y, .5), 0);

	float d1 = map(p + e.xyy), d2 = map(p - e.xyy);
	float d3 = map(p + e.yxy), d4 = map(p - e.yxy);
	float d5 = map(p + e.yyx), d6 = map(p - e.yyx);
	float d = map(p)*2.;

    edge = abs(d1 + d2 - d) + abs(d3 + d4 - d) + abs(d5 + d6 - d);
    //edge = abs(d1 + d2 + d3 + d4 + d5 + d6 - d*3.);
    edge = smoothstep(0., 1., sqrt(edge/e.x*2.));

    /*
    // Wider sample spread for the curvature.
    e = vec2(12./450., 0);
	d1 = map(p + e.xyy), d2 = map(p - e.xyy);
	d3 = map(p + e.yxy), d4 = map(p - e.yxy);
	d5 = map(p + e.yyx), d6 = map(p - e.yyx);
    crv = clamp((d1 + d2 + d3 + d4 + d5 + d6 - d*3.)*32. + .5, 0., 1.);
	*/
    
    e = vec2(.0015, 0); //iResolution.y - Depending how you want different resolutions to look.
	d1 = map(p + e.xyy), d2 = map(p - e.xyy);
	d3 = map(p + e.yxy), d4 = map(p - e.yxy);
	d5 = map(p + e.yyx), d6 = map(p - e.yyx);
	
    return normalize(vec3(d1 - d2, d3 - d4, d5 - d6));
}

// Function 308
vec3 GetNormal(vec3 p, float sphereR)
{
	vec2 j = vec2(sphereR, 0.0);
	vec3 nor  	= vec3(0.0,		Terrain2(p.xz, sphereR), 0.0);
	vec3 v2		= nor-vec3(j.x,	Terrain2(p.xz+j, sphereR), 0.0);
	vec3 v3		= nor-vec3(0.0,	Terrain2(p.xz-j.yx, sphereR), -j.x);
	nor = cross(v2, v3);
	return normalize(nor);
}

// Function 309
vec3 normal( in vec3 pos )
{
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
    return normalize( e.xyy*map( pos + e.xyy ) + 
					  e.yyx*map( pos + e.yyx ) + 
					  e.yxy*map( pos + e.yxy ) + 
					  e.xxx*map( pos + e.xxx ) );
}

// Function 310
vec4 calculateNormal(vec4 pos, float epsilon)
{
	vec4 epsilonVec = vec4(epsilon, 0.0, 0.0, 0.0);
    vec4 xyyy = vec4(epsilon, 0.0, 0.0, 0.0);
    vec4 yxyy = vec4(0.0, epsilon, 0.0, 0.0);
    vec4 yyxy = vec4(0.0, 0.0, epsilon, 0.0);
    vec4 yyyx = vec4(0.0, 0.0, 0.0, epsilon);
    
	vec4 normal = vec4(
	    map(pos+xyyy).x - map(pos-xyyy).x,
	    map(pos+yxyy).x - map(pos-yxyy).x,
	    map(pos+yyxy).x - map(pos-yyxy).x,
    	map(pos+yyyx).x - map(pos-yyyx).x);
	return normalize(normal);
}

// Function 311
vec3 getDetailNormal(vec3 p, vec3 normal, float t, int id){
    vec3 tangent;
    vec3 bitangent;
    //Construct orthogonal directions tangent and bitangent to sample detail gradient in
    pixarONB(normal, tangent, bitangent);
    
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);
    
    float EPS = DETAIL_EPSILON * 0.2;
    
    vec3 delTangent = vec3(0);
    vec3 delBitangent = vec3(0);
    
    for(int i = ZERO; i < 2; i++){
        
        //i ->  s
        //0 ->  1
        //1 -> -1
        float s = 1.0 - 2.0 * float(i&1);
    
        delTangent += s * getDetailExtrusion(p + s * tangent * EPS, normal, id);

        delBitangent += s * getDetailExtrusion(p + s * bitangent * EPS, normal, id);

    }
    
    return normalize(cross(delTangent, delBitangent));
}

// Function 312
vec4 heightToNormal(sampler2D height, vec3 samplingResolution, vec2 uv, float normalMultiplier) {

    vec2 s = 1.0/samplingResolution.xy;
    
    float p = texture(height, uv).x;
    float h1 = texture(height, uv + s * vec2(1,0)).x;
    float v1 = texture(height, uv + s * vec2(0,1)).x;
       
   	vec2 xy = (p - vec2(h1, v1)) * normalMultiplier;
   
    return vec4(xy + .5, 1., 1.);
}

// Function 313
vec3 GetNormal(vec3 p)
{
	vec3 eps = vec3(0.01,0.0,0.0);
	return normalize(vec3(
		Scene(p+eps.xyy)-Scene(p-eps.xyy),
		Scene(p+eps.yxy)-Scene(p-eps.yxy),
		Scene(p+eps.yyx)-Scene(p-eps.yyx)
		));
}

// Function 314
void geomTangentCurve(vec3 pos1, vec3 pos2, vec3 tan1, vec3 tan2, float r1, float r2, 
                      int rSegNum, int tSegNum, int vIdx, out vec3 pos, out vec3 normal)
{
    float l = length(pos1-pos2);
    l*=.4;
    int i=(vIdx/3/2)%tSegNum;
    //{  // converted some loops into proper vertex index values
        float fact, fact2;
        fact=max(0.,homFact(float(i)/float(tSegNum))); // force >=0 because of sqrt below
        vec3 p1=mix(pos1+tan1*l*sqrt(fact ),pos2-tan2*l*sqrt(1.-fact ),fact );
        fact2=max(0.,homFact(float(i+1)/float(tSegNum))); // force >=0 because of sqrt below
        vec3 p2=mix(pos1+tan1*l*sqrt(fact2),pos2-tan2*l*sqrt(1.-fact2),fact2);

        vec3 ta = mix(tan1,tan2,fact);
        vec3 tn = mix(tan1,tan2,fact2);

        float dph=PI*2./float(rSegNum);
        //vec3 b1=normalize(vec3(ta.x,-ta.y,0));
        vec3 b1=normalize(cross(ta,p1));
        vec3 b2=normalize(cross(ta,b1));
        //vec3 b3=normalize(vec3(tn.x,-tn.y,0));
        vec3 b3=normalize(cross(tn,p2));
        vec3 b4=normalize(cross(tn,b3));
        float r_1 = mix(r1,r2,fact);
        float r_2 = mix(r1,r2,fact2);
        int j=(vIdx/3/2/tSegNum)%rSegNum;
        //{
            float ph  = float(j)*dph;
            float ph2 = ph+dph;
            vec3 v1 = p1+r_1*(b1*cos(ph )+b2*sin(ph ));
            vec3 v2 = p1+r_1*(b1*cos(ph2)+b2*sin(ph2));
            vec3 v3 = p2+r_2*(b3*cos(ph )+b4*sin(ph ));
            vec3 v4 = p2+r_2*(b3*cos(ph2)+b4*sin(ph2));
            vec3 v[4] = vec3[](v1,v2,v3,v4);
            pos = v[triStripIndex[vIdx%6]];
            normal = normalize(cross(v[1]-v[0],v[2]-v[0]));
        //}
    //}
}

// Function 315
vec3 WaterNormal(vec3 p, float e)
{
    vec3 N;
    N.y = Water(p.xz);
    N.x = Water(p.xz-vec2(e, 0.))-N.y;
    N.z = Water(p.xz-vec2(0., e))-N.y;
    return normalize(N);
}

// Function 316
vec3 calcNormal( in vec3 pos ){
    vec3 eps = vec3( 0.001, 0.0, 0.0 );
    vec3 nor = vec3(
        map(pos+eps.xyy).dist - map(pos-eps.xyy).dist,
        map(pos+eps.yxy).dist - map(pos-eps.yxy).dist,
        map(pos+eps.yyx).dist - map(pos-eps.yyx).dist );
    return normalize(nor);
}

// Function 317
vec3 sdfNormal_simple(vec3 p, float epsilon)
{
    vec3 eps = vec3(epsilon, -epsilon, 0.0);
    
	float dX = sdf_simple(p + eps.xzz) - sdf_simple(p + eps.yzz);
	float dY = sdf_simple(p + eps.zxz) - sdf_simple(p + eps.zyz);
	float dZ = sdf_simple(p + eps.zzx) - sdf_simple(p + eps.zzy); 

	return normalize(vec3(dX,dY,dZ));
}

// Function 318
vec3 seaGetNormal(const in vec3 p, const in float eps) {
    vec3 n;
    n.y = seaMapHigh(p);    
    n.x = seaMapHigh(vec3(p.x+eps,p.y,p.z)) - n.y;
    n.z = seaMapHigh(vec3(p.x,p.y,p.z+eps)) - n.y;
    n.y = eps;
    return normalize(n);
}

// Function 319
vec3 getNormal( vec3 center, vec3 p )
{
	vec3 diff = p - center;
	return normalize( diff );
}

// Function 320
vec3 calcNormal(in vec3 p){

    // Note the slightly increased sampling distance, to alleviate artifacts due to hit point inaccuracies.
    vec2 e = vec2(0.001, -0.001); 
    return normalize(e.xyy*map(p + e.xyy) + e.yyx*map(p + e.yyx) + e.yxy*map(p + e.yxy) + e.xxx*map(p + e.xxx));
}

// Function 321
vec2 normal(vec2 p) {
    vec2 d = vec2(0.0001, 0.0);
    return normalize(
        vec2(
        	map(p + d.xy) - map(p - d.xy),
            map(p + d.yx) - map(p - d.yx)) / (2.0 * d.x));
}

// Function 322
vec3 calcNormal(vec3 pos) {
  return calcNormal(pos, 0.002);
}

// Function 323
vec3 ComputeBaseNormal(vec2 uv) 
{
    uv = fract(uv) * 2.0 - 1.0;    
        
    vec3 ret;
    ret.xy = sqrt(uv * uv) * sign(uv);
    ret.z = sqrt(abs(1.0 - dot(ret.xy,ret.xy)));
    
    ret = ret * 0.5 + 0.5;
    return mix(vec3(0.5,0.5,1.0), ret, smoothstep(1.0,0.95,dot(uv,uv)));
}

// Function 324
vec3 terrainCalcNormalHigh( in vec3 pos, float t ) {
    vec2 e = vec2(1.0,-1.0)*0.001*t;

    return normalize( e.xyy*terrainMapH( pos + e.xyy ) + 
					  e.yyx*terrainMapH( pos + e.yyx ) + 
					  e.yxy*terrainMapH( pos + e.yxy ) + 
					  e.xxx*terrainMapH( pos + e.xxx ) );
}

// Function 325
vec3 NormalSinPowWarp(vec3 pos, float freq, float amp, float power) {
    vec3 dir = normalize(pos);
	vec3 warp = abs(sin(dir*freq));
	float mw = max_element(warp);
	mw = pow(mw, power);
	return pos - dir * mw * amp;
}

// Function 326
vec3 calcNormal_3606979787(vec3 pos, float eps) {
  const vec3 v1 = vec3( 1.0,-1.0,-1.0);
  const vec3 v2 = vec3(-1.0,-1.0, 1.0);
  const vec3 v3 = vec3(-1.0, 1.0,-1.0);
  const vec3 v4 = vec3( 1.0, 1.0, 1.0);

  return normalize( v1 * mapRefract( pos + v1*eps ).x +
                    v2 * mapRefract( pos + v2*eps ).x +
                    v3 * mapRefract( pos + v3*eps ).x +
                    v4 * mapRefract( pos + v4*eps ).x );
}

// Function 327
vec3 Normal2(vec3 p){if(df(p)<.03)return Normal(p);return vec3(-.7);}

// Function 328
vec3 genNormal(vec3 p){
	vec3 normal=vec3(
		sdSphere(p+vec3(EPS,0.0,0.0))-sdSphere(p+vec3(-EPS,0.0,0.0)),
		sdSphere(p+vec3(0.0,EPS,0.0))-sdSphere(p+vec3(0.0,-EPS,0.0)),
		sdSphere(p+vec3(0.0,0.0,EPS))-sdSphere(p+vec3(0.0,0.0,-EPS))
		);
	return normalize(normal);
}

// Function 329
vec3 getNormal( in vec3 pos )
{
    vec2 e = vec2(0.002, -0.002);
    return normalize(
        e.xyy * map(pos + e.xyy) + 
        e.yyx * map(pos + e.yyx) + 
        e.yxy * map(pos + e.yxy) + 
        e.xxx * map(pos + e.xxx));
}

// Function 330
vec3 getNormal(in vec3 p) {
	
    const vec2 e = vec2(.001, 0);
    
    //return normalize(vec3(m(p + e.xyy) - m(p - e.xyy), m(p + e.yxy) - m(p - e.yxy),	
    //                      m(p + e.yyx) - m(p - e.yyx)));
    
    // This mess is an attempt to speed up compiler time by contriving a break... It's 
    // based on a suggestion by IQ. I think it works, but I really couldn't say for sure.
    float sgn = 1.;
    float mp[6];
    vec3[3] e6 = vec3[3](e.xyy, e.yxy, e.yyx);
    for(int i = min(iFrame, 0); i<6; i++){
		mp[i] = map(p + sgn*e6[i/2]);
        sgn = -sgn;
        if(sgn>2.) break; // Fake conditional break;
    }
    
    return normalize(vec3(mp[0] - mp[1], mp[2] - mp[3], mp[4] - mp[5]));
}

// Function 331
vec3 getNormal(vec3 hitPos) {
	const float derivDist = 0.00001;
	const float derivDist2 = 2.0 * derivDist;
	float x = hitPos.x;
	float y = hitPos.y;
	float z = hitPos.z;
	vec3 surfaceNormal;
	surfaceNormal.x = distanceField(vec3(x + derivDist, y, z)) 
					- distanceField(vec3(x - derivDist, y, z));
	surfaceNormal.y = distanceField(vec3(x, y + derivDist, z)) 
					- distanceField(vec3(x, y - derivDist, z));
	surfaceNormal.z = distanceField(vec3(x, y, z + derivDist)) 
					- distanceField(vec3(x, y, z - derivDist));
	surfaceNormal = normalize(surfaceNormal / derivDist2);
	return surfaceNormal;
}

// Function 332
ivec4 normals_data(in int i) {
#  ifdef R
#    undef R
#  endif
#  define R(i,a,b,c,d) case i: r=ivec4(a,b,c,d); break;
  ivec4 r;
  switch(i) {
R(0x0000,0x2cfaf3aa,0x29fa97c8,0x316a0798,0x336ad76a)R(0x0001,0x2d4a0fbd,0x307a8397,0x308bb36f,0x303987ac)R(0x0002,0x2e690bc3,0x27a97be6,0x36483f6d,0x38997337)R(0x0003,0x36ab9b14,0x3bca4ecc,0x34295782,0x2fbc3761)R(0x0004,0x2f3b7f85,0x357ccadd,0x302d371f,0x3779734d)R(0x0005,0x39e87729,0x2f38ebbd,0x2bb87fda,0x3197bbaa)R(0x0006,0x3736bf55,0x3ba7fefe,0x3b56a2f8,0x3e386ea1)R(0x0007,0x3a8c0676,0x3f693232,0x3d8aee36,0x2ffb4f84)R(0x0008,0x2c5cf35c,0x2d4dcf16,0x29cc8786,0x332e1e76)R(0x0009,0x37ed421d,0x2dfe6ecc,0x31da2b90,0x3668336b)R(0x000a,0x38072b4b,0x2db873cc,0x32276fa2,0x2d2853d1)R(0x000b,0x37975755,0x3b5636ed,0x3ea6ee7e,0x3df6067c)R(0x000c,0x3fc76223,0x3b8c05ea,0x3fb869ca,0x3e2a85d5)R(0x000d,0x274c9392,0x28bb77b6,0x289e5715,0x26ed5b6a)R(0x000e,0x2a1f0eae,0x240d6772,0x339e45da,0x2eaefe4b)R(0x000f,0x381d15ae,0x24ba67e1,0x28795fe4,0x31d6739c)
R(0x0010,0x3205b78b,0x2a2847e3,0x31d8b3a4,0x2bd8cfd7)R(0x0011,0x36893b60,0x3c089af2,0x3f07266c,0x3f263e13)R(0x0012,0x3ea5c601,0x3f46adc2,0x3b7ba58a,0x3de79d4e)R(0x0013,0x3da9cd68,0x20ec7ba6,0x219ba7c5,0x23cf36d1)R(0x0014,0x223e5335,0x29df8e2f,0x252fb26d,0x1dbe3f3c)R(0x0015,0x1d4d936b,0x348dc57e,0x2e7f19e1,0x301eb59c)R(0x0016,0x383c5d4b,0x1e3a37ea,0x1f6943f8,0x24373ff8)R(0x0017,0x2a93e794,0x260553d7,0x22a8b3fb,0x2e29c7bb)R(0x0018,0x33398b8b,0x2709cbe4,0x37194f56,0x3a79730e)R(0x0019,0x3e095e95,0x3fa91205,0x3fb765d0,0x3e05a5a7)R(0x001a,0x3dc5e181,0x3c660d38,0x3a4ab916,0x3a26d8e2)R(0x001b,0x3b58c0fb,0x1cdc8ba0,0x1dab9bc6,0x207f92a0)R(0x001c,0x1eded707,0x2a9f75be,0x25bfd9ef,0x220fea36)R(0x001d,0x1c9e4f34,0x1b4e0747,0x1a6d5b70,0x341cdd09)R(0x001e,0x304e1935,0x2b5f055d,0x363b20cc,0x1a39efe7)R(0x001f,0x19987ff3,0x18e75bf0,0x1c53a7a8,0x1e82cb82)
R(0x0020,0x1b081bf8,0x1f9a1bec,0x2b0a37ca,0x2fe9e3a9)R(0x0021,0x313b4378,0x245a9fdd,0x2d6e5edc,0x3b2a56df)R(0x0022,0x3dbaea19,0x3f1945af,0x3e180153,0x3af57920)R(0x0023,0x3a364cf0,0x37d5d0c8,0x36c9aca9,0x35367c8d)R(0x0024,0x37a7dca8,0x19fbdfb4,0x1fbfb683,0x1e1f1ee4)R(0x0025,0x25df9d86,0x234fd9b2,0x234ff207,0x197e8315)R(0x0026,0x1b5e1343,0x189db352,0x15fcfb6d,0x314b588a)R(0x0027,0x2f0d0cc5,0x2bbde8df,0x262ef116,0x3189a861)R(0x0028,0x1669bbda,0x186833f0,0x1655a7cf,0x1493e38e)R(0x0029,0x1602b35b,0x17673fe9,0x1c1a97df,0x1e3adfdb)R(0x002a,0x2b19abd2,0x2ca96fcc,0x23cc73a3,0x213eeafd)R(0x002b,0x1cea43e7,0x24694bf3,0x2a4f2a98,0x30ceb246)R(0x002c,0x324e69ac,0x3d2a897d,0x3d68cd3e,0x39d7ccd2)R(0x002d,0x3685a8b5,0x34d67488,0x3405c88b,0x3335c880)R(0x002e,0x30b8444b,0x2e16d83a,0x121b379b,0x2ebdcf03)R(0x002f,0x1c5f62b8,0x25ee0f3e,0x22ef6545,0x256f8d71)
R(0x0030,0x3beb965e,0x1a7e0345,0x160deb2f,0x145e3b04)R(0x0031,0x150d634d,0x128c4f74,0x2e69f448,0x2c3c0874)R(0x0032,0x272da0a7,0x21ae8cdc,0x2e686c37,0x0ea937a6)R(0x0033,0x13d83bd8,0x15c607d3,0x18a35f90,0x15730b6b)R(0x0034,0x15735f7d,0x18a9d7e3,0x1b1b1bd0,0x193a33df)R(0x0035,0x298837e7,0x28184bed,0x170c4397,0x16fe86ff)R(0x0036,0x26ce970a,0x13a92bd1,0x1a584bf6,0x22a713f9)R(0x0037,0x2b4eb2d4,0x2cef462e,0x28cfa1d0,0x31ad50ff)R(0x0038,0x27eec912,0x395910cf,0x3866e4bd,0x33c6cc75)R(0x0039,0x30d5c464,0x2e35e849,0x2f75845c,0x2ce57049)R(0x003a,0x2cc64038,0x2b978c23,0x2cd7082f,0x101a879c)R(0x003b,0x39094332,0x30b9afa5,0x278a9bd3,0x212eccf4)R(0x003c,0x336b94b0,0x3dba0d7a,0x3df98294,0x3f691dc9)R(0x003d,0x1c6ab7dd,0x140c737b,0x162e3319,0x0e6de2c8)R(0x003e,0x184d2373,0x179b97b4,0x2e68d03a,0x2b1a6039)R(0x003f,0x276bdc50,0x208d5c84,0x1e9df8ad,0x30677c4a)
R(0x0040,0x181a97d2,0x1317dfd3,0x18349bbd,0x1e238ba7)R(0x0041,0x18e35f91,0x1bb8fbf6,0x1a5b93bf,0x19796feb)R(0x0042,0x11183fc4,0x25d62be8,0x1f454be0,0x089b5f12)R(0x0043,0x09cd6280,0x166f0aba,0x262ee6ed,0x29de6f01)R(0x0044,0x0b97ef89,0x113737c2,0x18b613e2,0x20f4bbd2)R(0x0045,0x24aeeaf4,0x26af8e7c,0x280fb612,0x185fbde0)R(0x0046,0x22afb280,0x2bbbd869,0x0dcd20f9,0x243bb840)R(0x0047,0x2af8cc21,0x3495e490,0x31655875,0x2ee5d850)R(0x0048,0x2be54845,0x27b56c2a,0x29152038,0x2d340479)R(0x0049,0x3504f0b3,0x34f6ec83,0x3717d09e,0x1b9b4bcc)R(0x004a,0x372c3adf,0x317b3776,0x262c2ba7,0x1a3c9b97)R(0x004b,0x23dcf473,0x34c7a07b,0x3bd86907,0x3b5c1e10)R(0x004c,0x39ab712f,0x0edbe75f,0x091bd703,0x10ae52c2)R(0x004d,0x04bbae7e,0x03abaa1f,0x132c8f6d,0x12ab5b9b)R(0x004e,0x30d8c44f,0x2a19b026,0x26ba7025,0x210bdc40)R(0x004f,0x1e1ce46d,0x1bdd8494,0x21bc2bb3,0x271b9fb9)
R(0x0050,0x2da897cc,0x3205cb8c,0x2eb70bc1,0x1e6b5fce)R(0x0051,0x1ae937f2,0x16b5bfd3,0x0a26ab6b,0x2393d7b0)R(0x0052,0x1fe2db87,0x071452da,0x0207aab2,0x032a8a95)R(0x0053,0x08dcd6a7,0x14fc238f,0x265e6321,0x29ae9eef)R(0x0054,0x26cecaf5,0x05a6df18,0x0a06135e,0x10651795)R(0x0055,0x18a413ad,0x1db2df86,0x299df72e,0x247f46c4)R(0x0056,0x2b9f6a2a,0x2fdee62e,0x126e86c7,0x07bd31fe)R(0x0057,0x2c2d4f48,0x1e6de757,0x27c9fc20,0x0549c8fd)R(0x0058,0x0f480c4b,0x26290c0e,0x24478805,0x29861826)R(0x0059,0x2f943488,0x2be3fc6f,0x3a4624f5,0x2664883d)R(0x005a,0x2035441e,0x22350825,0x24c3a059,0x2c214939)R(0x005b,0x35a32d28,0x3a051918,0x390768c3,0x3118ac51)R(0x005c,0x10cba77f,0x294f923f,0x252ee6f2,0x188ebef5)R(0x005d,0x0e2dc2d1,0x235af427,0x2fc70847,0x2d08742d)R(0x005e,0x32ca307a,0x2d0f0987,0x26ce78e9,0x06dc4e9a)R(0x005f,0x03db7266,0x031b4a3b,0x0119f622,0x00796603)
R(0x0060,0x00084a0a,0x0017cddf,0x0149aa5d,0x0017d5dd)R(0x0061,0x27296c15,0x24d9f415,0x225ad022,0x1eabc43d)R(0x0062,0x1b2c9864,0x170cbc7d,0x134bf78a,0x23ec1bb1)R(0x0063,0x2fdaef91,0x3757ab5b,0x3785df3c,0x2ec863c4)R(0x0064,0x253987ee,0x2194cfd3,0x1652fb6e,0x0565caf5)R(0x0065,0x24926b66,0x1da1631c,0x08034a9a,0x02455654)R(0x0066,0x02b76acb,0x0848e352,0x1627a3e5,0x20baafe1)R(0x0067,0x2afc3b8b,0x2aadbf36,0x307c3359,0x02e65eb8)R(0x0068,0x062542f5,0x0b03f71f,0x12a2df49,0x1511a700)R(0x0069,0x1df1671f,0x3a19bb11,0x393be6bc,0x3dbad22f)R(0x006a,0x3d7a667b,0x0e5dcad2,0x046be1b8,0x3668736a)R(0x006b,0x2c8b4fa3,0x276d0f79,0x1a6d6b6c,0x25dc2054)R(0x006c,0x1dc86c01,0x07383cbc,0x0dd5e070,0x17f47c45)R(0x006d,0x28543451,0x320bcca7,0x2244e029,0x21d51423)R(0x006e,0x2763d45b,0x2b73009c,0x27434873,0x35b65498)R(0x006f,0x22c3e44a,0x1ce4d42c,0x1725f425,0x19e4e032)
R(0x0070,0x1ea10107,0x21a06165,0x2ac1d0ed,0x31246892)R(0x0071,0x2cf7a82c,0x2688900c,0x0028ba1b,0x1fefd199)R(0x0072,0x1bbfd648,0x111eea5b,0x083d463b,0x1e581800)R(0x0073,0x28572c14,0x1d7ac021,0x222cfc71,0x1aaee90f)R(0x0074,0x047c0e06,0x028b0ddf,0x00c961be,0x00083dfa)R(0x0075,0x002895d8,0x00475a36,0x00f6724c,0x01961999)R(0x0076,0x0156b188,0x01d6e562,0x01b70166,0x20989c01)R(0x0077,0x21c96408,0x209a7819,0x1deb9438,0x195c0452)R(0x0078,0x138cf0a3,0x128af457,0x01f9569a,0x181acbce)R(0x0079,0x2bf963d1,0x31d59f8b,0x31347f6f,0x2bcad7b6)R(0x007a,0x2a069bdc,0x32789b9e,0x36846f19,0x2351631a)R(0x007b,0x13915acc,0x03553e90,0x2490b2c2,0x1df03a72)R(0x007c,0x0cb21a9c,0x0683423f,0x0d41d27d,0x071456dc)R(0x007d,0x0c36d38a,0x166557c9,0x1e0593e5,0x2327ebfc)R(0x007e,0x2eb743c2,0x31da0393,0x38584b49,0x0195aa33)R(0x007f,0x0454a68e,0x09034ec0,0x0dd252d5,0x11613e8e)
R(0x0080,0x1ad05a7d,0x3cf77ed5,0x3ef8ba73,0x3fd83a1c)R(0x0081,0x3e069293,0x087cf68a,0x039a3944,0x385452e8)R(0x0082,0x35d73370,0x2a2c039a,0x234e5b30,0x2f0c775b)R(0x0083,0x2eec336a,0x0dcd72ea,0x24ff795e,0x11eddcf7)R(0x0084,0x03b84d10,0x14a42065,0x0a373c8b,0x1124e867)R(0x0085,0x19332477,0x21225896,0x2ca1a514,0x38e4f101)R(0x0086,0x3d1b01ab,0x32ee65d4,0x22d2489b,0x2213146c)R(0x0087,0x1ea31c6a,0x279208c0,0x285260ad,0x2392f076)R(0x0088,0x27e67c19,0x1d039058,0x13951c4d,0x10d59058)R(0x0089,0x15159038,0x15e248bf,0x19c049a1,0x17f09570)R(0x008a,0x1d61d0bd,0x23b49834,0x2287d001,0x01079582)R(0x008b,0x156f69a6,0x0c3e35c0,0x071cc59d,0x12484c31)R(0x008c,0x1928b80d,0x111c2090,0x179d84a6,0x0f3df12c)R(0x008d,0x04ab997e,0x036a414c,0x05297cfa,0x002885d8)R(0x008e,0x001885ff,0x001891eb,0x00e70a69,0x00b68234)R(0x008f,0x00f605f7,0x001765f4,0x0037e5c4,0x01778969)
R(0x0090,0x1d37c802,0x1ca81002,0x1df89402,0x1e690804)R(0x0091,0x1cfa881d,0x167a2c2b,0x117c6c9a,0x0dccb4db)R(0x0092,0x0db9a469,0x00082deb,0x0567af1b,0x21b62ff0)R(0x0093,0x26c3efaa,0x212ba7c6,0x25eb9fbd,0x318a2b93)R(0x0094,0x1ca52fdb,0x3a171f20,0x3b440a4f,0x34b1fa38)R(0x0095,0x22702255,0x12811a8f,0x12c23318,0x04044a49)R(0x0096,0x23501225,0x1f5009cf,0x10b0f9fd,0x1560965c)R(0x0097,0x13c0ee8b,0x1fc072a9,0x1881e332,0x23a2ab77)R(0x0098,0x2873f7a3,0x2f23f770,0x34868b7c,0x37858732)R(0x0099,0x039491a6,0x0493ddf5,0x0822da57,0x0b821a70)R(0x009a,0x0ed13e0d,0x15a07223,0x1d9009e4,0x3cd5eaad)R(0x009b,0x3f77624a,0x3fc765e4,0x3fc73609,0x38c2f20f)R(0x009c,0x03bbb5e7,0x05e850d7,0x32c191cd,0x38f412c3)R(0x009d,0x37a6df4e,0x25cd576f,0x1d2ed305,0x2c9d5343)R(0x009e,0x369af332,0x352b8336,0x2ded4f36,0x0c8e1a6b)R(0x009f,0x2dbf360d,0x1b9f916e,0x079c012f,0x0176618e)
R(0x00a0,0x04f4417b,0x1790bd56,0x0e25ec6c,0x14c48057)R(0x00a1,0x1ef26492,0x256188de,0x3fa80640,0x389cda63)R(0x00a2,0x25c0d131,0x25e13104,0x1cb1d0bf,0x1bd1a8ce)R(0x00a3,0x23b0b138,0x2711111a,0x21d168de,0x1ae214b0)R(0x00a4,0x1f46d005,0x11435ca7,0x0b14ccb6,0x0ed40ca1)R(0x00a5,0x14750849,0x1363ac82,0x1610899f,0x1670a976)R(0x00a6,0x162204d1,0x1b74b432,0x01788d6a,0x091c210f)R(0x00a7,0x09990898,0x082a74d0,0x0d3a407b,0x011999a9)R(0x00a8,0x0a6c08ed,0x003739e2,0x00180220,0x0066c1e9)R(0x00a9,0x0007e9f5,0x0145f1c3,0x0018b1fd,0x00792dd1)R(0x00aa,0x1ac6f40b,0x1a57bc08,0x1b381005,0x1a089c0a)R(0x00ab,0x1518541f,0x123ae858,0x0dcd04f0,0x00c8f19f)R(0x00ac,0x01375685,0x13c48fa2,0x0288febb,0x095a7344)R(0x00ad,0x229927f7,0x3106d3a9,0x34a3c317,0x3471e227)R(0x00ae,0x36d289b1,0x31b159ee,0x24a0cacf,0x2972a75b)R(0x00af,0x1d3102f3,0x0ad20dcb,0x24b0399f,0x16d0e943)
R(0x00b0,0x178061b0,0x24104a76,0x224005ed,0x23a0bacd)R(0x00b1,0x24e08aa6,0x26009eaa,0x27c17309,0x2c32532d)R(0x00b2,0x35f3fb0b,0x07d4151f,0x07635d73,0x095259f2)R(0x00b3,0x0d91cd77,0x160085a5,0x1f306561,0x3e95c225)R(0x00b4,0x39c3c282,0x3d94fa02,0x037a2d47,0x08a66caf)R(0x00b5,0x2a916515,0x354211cf,0x3a644aa3,0x39169f2f)R(0x00b6,0x20ee8327,0x1e2ed308,0x296d8f4d,0x320c633a)R(0x00b7,0x2b4e6ef2,0x0f7ed1db,0x37ecca8f,0x31fe9229)R(0x00b8,0x22cfd9ad,0x125e9d46,0x045ad54a,0x0c91a61d)R(0x00b9,0x00f699a7,0x0d3181f6,0x18404227,0x238015d2)R(0x00ba,0x0fe43c8c,0x1ab3406b,0x21529087,0x2ad1bcf4)R(0x00bb,0x3692d692,0x3c183af1,0x3011558c,0x27609d63)R(0x00bc,0x26706d7e,0x2710599c,0x1f307d4e,0x17b0f931)R(0x00bd,0x10b2150d,0x0a336118,0x1301a11a,0x17b3287e)R(0x00be,0x19b57c23,0x1515583c,0x14d11147,0x18d041c0)R(0x00bf,0x1970b944,0x156214d2,0x15d47851,0x0149a19c)
R(0x00c0,0x05dbd95f,0x002741f1,0x0165a1ef,0x0017e5de)R(0x00c1,0x001851e4,0x005919df,0x00378630,0x00f9f1f0)R(0x00c2,0x0109f1e1,0x15468826,0x16778818,0x16e7dc14)R(0x00c3,0x13c77c27,0x14f77420,0x13d8d029,0x0b6b94c2)R(0x00c4,0x0048ede9,0x00288a20,0x0048763a,0x07b77f4b)R(0x00c5,0x249417b7,0x2931af14,0x2ab08638,0x303145a2)R(0x00c6,0x36023a11,0x34335b06,0x35a4a734,0x32820aa7)R(0x00c7,0x1f9005df,0x25e02615,0x20403a7a,0x2310bace)R(0x00c8,0x27f126e6,0x38a2f9cc,0x28c05211,0x0588e8e5)R(0x00c9,0x0a6510b7,0x3562419e,0x3a13a664,0x3ab69709)R(0x00ca,0x210ef6f8,0x20cf06f1,0x280df33b,0x2dddc712)R(0x00cb,0x280f8671,0x178f455d,0x3a8be687,0x3be7caf7)R(0x00cc,0x359dde19,0x28ef8daa,0x1a6f6555,0x0eedd527)R(0x00cd,0x043ac148,0x13f09a1c,0x17b04a24,0x16a06a42)R(0x00ce,0x02d59171,0x21d05e96,0x11b350a5,0x1a4248a4)R(0x00cf,0x2a11f0da,0x26b18ce4,0x3353330b,0x21d011c3)
R(0x00d0,0x1f1140ed,0x20d3a452,0x2c040070,0x1d138459)R(0x00d1,0x1cf021ae,0x2720599d,0x3002dcd6,0x1a535869)R(0x00d2,0x13744868,0x002899da,0x0116e26f,0x00387632)R(0x00d3,0x001855e1,0x00a931bd,0x013a05cd,0x02182eb4)R(0x00d4,0x021882b2,0x006931ea,0x001895ed,0x1475b439)R(0x00d5,0x1346f42e,0x14f72c22,0x08f8889e,0x0ba7a075)R(0x00d6,0x00084a06,0x0017ba25,0x0ce39b2b,0x1ae0a6b9)R(0x00d7,0x1c700de2,0x30f23ee3,0x26602ddd,0x3514ef49)R(0x00d8,0x39049ae7,0x3892edd7,0x30617975,0x0b66b480)R(0x00d9,0x3311e184,0x38e3021f,0x3b8a42d6,0x25cf46bb)R(0x00da,0x281ebeef,0x2bcefaa1,0x28cfae06,0x1e6e64ce)R(0x00db,0x3c16dee7,0x3c3b4e70,0x32e31309,0x38bcf63f)R(0x00dc,0x2d7f2dc8,0x1b5f694f,0x0ecdf135,0x04aa0117)R(0x00dd,0x1d0009f9,0x1b701606,0x15807623,0x1330ade6)R(0x00de,0x20507aac,0x0f31f137,0x1774e43c,0x1b9220aa)R(0x00df,0x24f174e3,0x30118962,0x2bd2e4a6,0x2e6288d9)
R(0x00e0,0x38f34d94,0x3592916e,0x3973a17c,0x33b410c4)R(0x00e1,0x29b5dc2b,0x1e26b407,0x0038363b,0x00288229)R(0x00e2,0x0218aab1,0x0088e9b8,0x01a9d591,0x0018a1ec)R(0x00e3,0x00786a51,0x00c89a67,0x00286dd8,0x001875ee)R(0x00e4,0x16e72417,0x0d17c063,0x0007c606,0x01958a23)R(0x00e5,0x01d55a16,0x1dc29375,0x2b23fb93,0x3334474a)R(0x00e6,0x35c2565d,0x3251b987,0x3412dd1e,0x3b9415c3)R(0x00e7,0x32ae5259,0x2b1f4274,0x2d4ec558,0x32c2fb04)R(0x00e8,0x38b4ff04,0x3d596eb3,0x1d806298,0x39fbd2a8)R(0x00e9,0x32ce623d,0x23afd1aa,0x166f315e,0x0ced6522)R(0x00ea,0x08f79c9c,0x1d2015c7,0x191035d9,0x0f117589)R(0x00eb,0x0f9121e7,0x19111d12,0x28234c76,0x30e2610b)R(0x00ec,0x2d23848f,0x2b037082,0x2b50d574,0x31e179b3)R(0x00ed,0x35128163,0x2b6260c4,0x28454431,0x20472c02)R(0x00ee,0x1837f00f,0x01c9129a,0x00c9b613,0x01989299)R(0x00ef,0x01ea1591,0x007939d7,0x0018a212,0x01c84959)
R(0x00f0,0x0038d9eb,0x0007edff,0x00a6da46,0x0e93833e)R(0x00f1,0x28a2173b,0x29605e20,0x2050417c,0x2b8240cf)R(0x00f2,0x398c5178,0x2d31caf2,0x3382eaf0,0x3bf7aaf5)R(0x00f3,0x3c5a12c3,0x354d7e90,0x2a3f8e0f,0x1d4fc998)R(0x00f4,0x119e1512,0x13960039,0x19509d59,0x14e0b58d)R(0x00f5,0x0e61f546,0x1310c1b7,0x1be1c4c4,0x25838061)R(0x00f6,0x24624c9e,0x1d803592,0x2a80a98c,0x16614119)R(0x00f7,0x1924c837,0x05f7d0d5,0x0058ca34,0x01da3a5e)R(0x00f8,0x01eab601,0x007935d4,0x0018aa0a,0x00082a05)R(0x00f9,0x0076e632,0x0d31d27d,0x0175a5e3,0x0ee300db)R(0x00fa,0x1890b551,0x3b145686,0x3b2be263,0x3c64d987)R(0x00fb,0x308eca2a,0x249fd9c5,0x199dfcbd,0x2004d429)R(0x00fc,0x1a3164ec,0x13d15d2d,0x21d2748f,0x04b5c11c)R(0x00fd,0x01d5e596,0x012609c6,0x00c7259e,0x0079421b)R(0x00fe,0x005919ef,0x001805da,0x00083df0,0x0007c1f9)R(0x00ff,0x00377dcb,0x1e5158e3,0x36580091,0x2b8d3cac)
R(0x0100,0x00000000,0x00000000,0x00000000,0x00000000)  }
  return r;
}

// Function 333
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

// Function 334
vec3 calcNormal(vec3 p) {
	float eps = 0.001;
	const vec3 v1 = vec3( 1.0,-1.0,-1.0);
	const vec3 v2 = vec3(-1.0,-1.0, 1.0);
	const vec3 v3 = vec3(-1.0, 1.0,-1.0);
	const vec3 v4 = vec3( 1.0, 1.0, 1.0);
	return normalize( v1 * map( p + v1*eps ) +
					  v2 * map( p + v2*eps ) +
					  v3 * map( p + v3*eps ) +
					  v4 * map( p + v4*eps ) );
}

// Function 335
vec3 GetNormal(in vec3 hit_pos, in float delta)
{
	vec3 pre_p = hit_pos - vec3(0,0,delta);
	vec3 p = hit_pos + vec3(0,0,delta);
	
	vec3 t, l, r, b;
	t = l = r = b =  hit_pos + vec3(0.0,0.0, 10000.0);//hit very far away
	vec3 eps = vec3(M_EPSILON, 0.0, 0.0);
	t += eps.yxy;
	l -= eps.xyy;
	r += eps.xyy;
	b -= eps.yxy;
	
	Hit(p + eps.yxy , pre_p + eps.yxy, t);
	Hit(p -eps.xyy , pre_p -eps.xyy, l);
	Hit(p + eps.xyy , pre_p + eps.xyy, r);
	Hit(p -eps.yxy , pre_p -eps.yxy, b);

	return normalize(cross(r-l, t-b));
}

// Function 336
vec3 calcNormal(vec3 p) {
 	   
    vec2 eps = vec2(.001,0.);
    vec3   n = vec3(dstScene(p + eps.xyy) - dstScene(p - eps.xyy),
                    dstScene(p + eps.yxy) - dstScene(p - eps.yxy),
                    dstScene(p + eps.yyx) - dstScene(p - eps.yyx));
    return normalize(n);
    
}

// Function 337
vec3 calcNormal( in vec3 pos )
{
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
    return normalize( e.xyy*map( pos + e.xyy ) + 
					  e.yyx*map( pos + e.yyx ) + 
					  e.yxy*map( pos + e.yxy ) + 
					  e.xxx*map( pos + e.xxx ) );
    /*
	vec3 eps = vec3( 0.0005, 0.0, 0.0 );
	vec3 nor = vec3(
	    map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	    map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	    map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
	*/
}

// Function 338
vec3 normalHeightPlaneZ(vec4 h) {
    return vec3(h.xy,-1);
}

// Function 339
vec3 estimateNormal(in vec3 p) {
	const vec2 e = vec2(EPSILON, 0);
	return normalize(vec3(sceneSDF(p + e.xyy) - sceneSDF(p - e.xyy), sceneSDF(p + e.yxy) - sceneSDF(p - e.yxy),	sceneSDF(p + e.yyx) - sceneSDF(p - e.yyx)));
}

// Function 340
vec3 calcNormal( in vec3 pos )
{
    vec3 eps = vec3(0.0001,0.0,0.0);

    return normalize( vec3(
      map( pos+eps.xyy ) - map( pos-eps.xyy ),
      map( pos+eps.yxy ) - map( pos-eps.yxy ),
      map( pos+eps.yyx ) - map( pos-eps.yyx ) ) );
}

// Function 341
vec3 normalize(vec3 u)     { return (1.0 / length(u)) * u; }

// Function 342
vec3 calcNormal( in vec3 pos, in float id )
{
    vec3 eps = vec3(0.01,0.0,0.0);

	return normalize( vec3(
           map2(pos+eps.xyy,id) - map2(pos-eps.xyy,id),
           map2(pos+eps.yxy,id) - map2(pos-eps.yxy,id),
           map2(pos+eps.yyx,id) - map2(pos-eps.yyx,id) ) );
}

// Function 343
vec3 calcNormal( in vec3 p )
{
    const float eps = 0.008;             

    const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v2 = vec3(-1.0,-1.0, 1.0);
    const vec3 v3 = vec3(-1.0, 1.0,-1.0);
    const vec3 v4 = vec3( 1.0, 1.0, 1.0);

	return normalize( v1*(p.y - dScene( (p + v1*eps) )) + 
					  v2*(p.y - dScene( (p + v2*eps) )) + 
					  v3*(p.y - dScene( (p + v3*eps) )) + 
					  v4*(p.y - dScene( (p + v4*eps) )) );
}

// Function 344
vec3 calcNormal(in vec3 pos)
{
    vec2 e = vec2(1.0, -1.0) * 0.5773 * 0.0005;
    return normalize(e.xyy * map(pos + e.xyy) +
                     e.yyx * map(pos + e.yyx) +
                     e.yxy * map(pos + e.yxy) +
                     e.xxx * map(pos + e.xxx));
}

// Function 345
vec3 normal(in vec3 p)
{
  //tetrahedron normal
  const float n_er=0.01;
  float v1=obj(vec3(p.x+n_er,p.y-n_er,p.z-n_er));
  float v2=obj(vec3(p.x-n_er,p.y-n_er,p.z+n_er));
  float v3=obj(vec3(p.x-n_er,p.y+n_er,p.z-n_er));
  float v4=obj(vec3(p.x+n_er,p.y+n_er,p.z+n_er));
  return normalize(vec3(v4+v1-v3-v2,v3+v4-v1-v2,v2+v4-v3-v1));
}

// Function 346
vec3 normal_unpack(vec2 enc)
{
    vec3 n;
    n.xy = enc*2.0-1.0;
    n.z = sqrt(1.0-dot(n.xy, n.xy));
    return n;
}

// Function 347
vec3 Normal2(vec3 p){if(df(p).w<.03)return Normal(p);return vec3(0);}

// Function 348
vec3 normalforCylinder(vec3 hit,Cylinder cylinder)
{
    vec3 nor;
	nor.xz = hit.xz - cylinder.c.xz;
    nor.y = 0.0;
    nor = nor/cylinder.r;
    //nor.y = 1.0*sign(hit.y-cylinder.c.y);
    return nor;
}

// Function 349
vec3 calculateNormal(vec3 pos, vec3 playerPos) {
    const vec3 e = vec3(EPS, 0.0, 0.0);
	float p = map(pos, junkMatID, playerPos, true);
	return normalize(vec3(map(pos + e.xyy, junkMatID, playerPos, true) - p,
           				  map(pos + e.yxy, junkMatID, playerPos, true) - p,
                          map(pos + e.yyx, junkMatID, playerPos, true) - p));
}

// Function 350
vec3 GetNormal(vec3 p) {
	float d = DE(p);
    vec2 e = vec2(.01, 0);
    
    vec3 n = d - vec3(
        DE(p-e.xyy),
        DE(p-e.yxy),
        DE(p-e.yyx));
    
    return normalize(n);
}

// Function 351
vec3 calcNormal(in vec3 pos) 
{
    int mat;
    vec2 e = vec2(1.0, -1.0) * 0.001;
    return normalize(
        e.xyy * map(pos + e.xyy, mat) +
        e.yyx * map(pos + e.yyx, mat) +
        e.yxy * map(pos + e.yxy, mat) +
        e.xxx * map(pos + e.xxx, mat));
}

// Function 352
vec3 calcNormal(in vec3 pos
){return normalize(vec3(Map(pos+eps.xyy)-Map(pos-eps.xyy),0.5*2.0*eps.x,Map(pos+eps.yyx)-Map(pos-eps.yyx)));}

// Function 353
vec4 BakeNormalWorldSpace(float theta, float phi) 
{   
    vec3 dir = vec3(sin(theta)*sin(phi), cos(phi), -sin(phi)*cos(theta));
    vec3 ro = dir*10.;	// The ray starts outside the models
    Hit hP = RayMarchHighDetailModel(ro, -dir);
    
    return vec4(normalize(hP.normal) * 0.5 + 0.5, .1);
}

// Function 354
vec3 calcNormal( in vec3 pos )
{
    vec3 eps = vec3(0.005,0.0,0.0);
	return normalize( vec3(
           map(pos+eps.xyy).x - map(pos-eps.xyy).x,
           map(pos+eps.yxy).x - map(pos-eps.yxy).x,
           map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
}

// Function 355
vec3 normal(vec3 p) {
 
    const float d = .001;
    
    vec3 left = vec3(p.x - d,p.yz);
    vec3 right = vec3(p.x + d,p.yz);
    vec3 up = vec3(p.x,p.y-d,p.z);
    vec3 down = vec3(p.x,p.y+d,p.z);
    vec3 front = vec3(p.xy,p.z-d);
    vec3 back = vec3(p.xy,p.z+d);
    
    float distLeft = distToScene(left).dist;
    float distRight = distToScene(right).dist;
    float distUp = distToScene(up).dist;
    float distDown = distToScene(down).dist;
    float distFront = distToScene(front).dist;
    float distBack = distToScene(back).dist;
    
    return normalize(vec3(distRight-distLeft,distDown-distUp,distBack-distFront));
    
}

// Function 356
vec3 getNormal(vec3 pos, float e, bool inside)
{
    vec2 q = vec2(0, e);
    return (inside?-1.:1.)*normalize(vec3(map(pos + q.yxx) - map(pos - q.yxx),
                          map(pos + q.xyx) - map(pos - q.xyx),
                          map(pos + q.xxy) - map(pos - q.xxy)));
}

// Function 357
vec3 CalculateNormalMapNormal(in vec2 uv, in float height, in vec3 normal, in vec3 tangent, in vec3 binormal)
{   
    vec3 normalMap = SampleNormalMap(uv, height).rgb;
	return normalize((normal * normalMap.b) + (binormal * normalMap.g) + (tangent * normalMap.r));
}

// Function 358
vec3 getNormal(in vec3 p) {
	const vec2 e = vec2(0.0025, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 359
vec3 mockNormal(vec2 uv) {
    vec3 normal;
    normal.xy = sin( vec2( 318.1, 178.2 ) * uv );
    normal.z = 1.0 + cos( 82.2 * uv.x ); 
    return normalize( normal );
}

// Function 360
vec3 NormalBlend_PartialDerivatives(vec3 n1, vec3 n2)
{	
    // Unpack
	n1 = n1*2.0 - 1.0;
    n2 = n2*2.0 - 1.0;
    
    return normalize(vec3(n1.xy*n2.z + n2.xy*n1.z, n1.z*n2.z));
}

// Function 361
vec3 getNormal(vec3 p) {
	vec2 t = vec2(0.0005,0.);    
    return normalize(vec3(
    	map(p + t.xyy).d - map(p - t.xyy).d,
    	map(p + t.yxy).d - map(p - t.yxy).d,
    	map(p + t.yyx).d - map(p - t.yyx).d
    ));
}

// Function 362
void bumpNormalTexture ( inout Ray ray, in float size, in sampler2D tex, in vec2 texture_size) {
	vec3 uaxis = normalize(cross(normalize(abs(ray.direction)), ray.normal));
	vec3 vaxis = normalize(cross(uaxis, ray.normal));
	mat3 space = mat3( uaxis, vaxis, ray.normal );

	float A = texture2D(tex, ray.object.uv - vec2(0,0) / texture_size).r;
	float B = texture2D(tex, ray.object.uv - vec2(1,0) / texture_size).r;
    float C = texture2D(tex, ray.object.uv - vec2(0,1) / texture_size).r;
	ray.normal = normalize(vec3(B - A, C - A, clamp(1.- size, 0.,1.)));
	ray.normal = normalize(space * ray.normal);
}

// Function 363
float pack_normal(vec2 n) {
    vec2 s = sign(n);
    return s.y*(s.x * (sqrt(5.0/(n.x*s.x + 4.0) - 1.0) - 0.5) + 0.5);
}

// Function 364
vec3 GetNormal(vec3 p) {
    float d = GetDist(p);
    vec2 e = vec2(0.01, 0.0);
    
    vec3 n = d - vec3(
        GetDist(p-e.xyy),
        GetDist(p-e.yxy),
        GetDist(p-e.yyx)
    );
    
    return normalize(n);
}

// Function 365
vec3 calcNormal(in vec3 p) {
	const vec2 e = vec2(.002, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 366
vec3 CalcNormalModification(in vec3 position, in vec3 normal, in float materialId)
{
#ifdef USE_NORMAL_DEFORMATION
	
	vec3 dirX = cross(normal,vec3(0.0,0.0,1.0));
	vec3 dirXalt = cross(normal, vec3(0.0,1.0,0.0)); 
	dirX = mix(dirXalt,dirX,step(0.5,length(dirX)));
	vec3 dirZ = cross(dirX,normal); 	
	
	float isIce = step(1.5,materialId); 
	float isDistortion = step(0.5,materialId);  
	
	position.y += meltHeight*isIce; 
	float dist1 = 0.1*Fbm(vec3(sin(3.0*Fbm(position*2.0)))); 
	float dist2 = 0.05*Fbm(position*3.0);
	float distort = mix(dist2,dist1,isIce);
	
	return isDistortion*(dirX + dirZ)*distort; 
#else 
	return vec3(0.0); 
#endif
}

// Function 367
vec3 normal(in vec3 p) {
	const vec2 e = vec2(0.002, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 368
vec3 calcNormal (in vec3 r)
{
    vec3 eps = vec3(0.001,0.0,0.0);

	return normalize(vec3(
           map(r+eps.xyy) - map(r-eps.xyy),
           map(r+eps.yxy) - map(r-eps.yxy),
           map(r+eps.yyx) - map(r-eps.yyx)
    ));
}

// Function 369
vec3 GetNormal (vec3 p)
{
    float c = map(p).x;
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(map(p+e.xyy).x, map(p+e.yxy).x, map(p+e.yyx).x) - c);
}

// Function 370
vec3 reCalcNormalFast(vec2 uv)
{
    float offsetPixel = 1.0;
    
    vec3 center = reCalcWorldPosition(uv);
    
    // Only sample two points, but vary which ones per frame in the hopes that temporal AA will smooth out artifacts
    if(iFrame % 4 == 0)
    {
        vec3 up = reCalcWorldPosition(uv+vec2(0, offsetPixel/iResolution.y));
        vec3 right = reCalcWorldPosition(uv+vec2(offsetPixel/iResolution.x, 0));
    
        return normalize(cross(up-center, center-right));
    }
    else if(iFrame % 4 == 1)
    {
        vec3 down = reCalcWorldPosition(uv+vec2(0, -offsetPixel/iResolution.y));
        vec3 left = reCalcWorldPosition(uv+vec2(-offsetPixel/iResolution.x, 0));

        return normalize(cross(center-down, left-center));
    }
    else if(iFrame % 4 == 2)
    {
        vec3 up = reCalcWorldPosition(uv+vec2(0, offsetPixel/iResolution.y));
        vec3 left = reCalcWorldPosition(uv+vec2(-offsetPixel/iResolution.x, 0));

        return normalize(cross(up-center, left-center));
    }
    else
    {
        vec3 down = reCalcWorldPosition(uv+vec2(0, -offsetPixel/iResolution.y));
        vec3 right = reCalcWorldPosition(uv+vec2(offsetPixel/iResolution.x, 0));

        return normalize(cross(center-down, center-right));
    }
}

// Function 371
vec3 normal(vec3 p)
{
    vec2 e = vec2(.0001, .0);
    float d = map(p);
    vec3 n = d - vec3(
        map(p - e.xyy*p),
        map(p - e.yxy*p),
        map(p - e.yyx*p));
    return normalize(n);
}

// Function 372
vec3 CubemapNormal(in vec2 tile) 
{   
    float s = (2.0*square((tile.x + 1.0)*0.5) - 1.0);
    
    float x = square(tile.x) * square(tile.y + 1.0) * s;
    float y = square(tile.y) * s;
    float z = square(tile.x + 1.0) * square(tile.y + 1.0) * s;
 
    return vec3(x, y, z);
}

// Function 373
vec3 GetSurfaceNormal(vec3 p)
{
    float d0 = GetDistanceToNearestSurface(p);
    const vec2 epsilon = vec2(.0001,0);
    vec3 d1 = vec3(
        GetDistanceToNearestSurface(p-epsilon.xyy),
        GetDistanceToNearestSurface(p-epsilon.yxy),
        GetDistanceToNearestSurface(p-epsilon.yyx));
    return normalize(d0 - d1);
}

// Function 374
vec3 calcNormal( in vec3 pos )
{
	vec4 dummy;
    vec3 eps = vec3(EPSILON,0.0,0.0);
	return normalize( vec3( map(pos+eps.xyy,dummy).x - map(pos-eps.xyy,dummy).x, map(pos+eps.yxy,dummy).x - map(pos-eps.yxy,dummy).x, map(pos+eps.yyx,dummy).x - map(pos-eps.yyx,dummy).x) );
}

// Function 375
vec3 normal(vec3 p) {
    vec3 d=vec3(0.,det*2.,0.);
	return normalize(vec3(de(p-d.yxx),de(p-d.xyx), de(p-d.xxy))-de(p));
}

// Function 376
float ct_normal_pi(
    in vec2 z
){
    vec2 d = vec2(z[0] * z[0] + z[1] * z[1]);
    float a = atan(d[1], d[0]);
    if (a < 0.0) a += 6.28318;
    a /= 6.28318;
    return a;
}

// Function 377
vec3 tor_tangent(in vec3 point) {
    vec2 ref_point = normalize(point.xz) * tor_rad1;
    return vec3(ref_point, 0.0).zyx * vec3(-1.0, 0.0, 1.0);
}

// Function 378
vec3 getNormal( in vec3 pos )
{
    vec2 e = vec2(0.001, -0.001);
    return normalize(
        e.xyy * map(pos + e.xyy) + 
        e.yyx * map(pos + e.yyx) + 
        e.yxy * map(pos + e.yxy) + 
        e.xxx * map(pos + e.xxx));
}

// Function 379
vec3 calcNormal(in vec3 pos) {
    vec2 e = vec2(1.0, -1.0) * 0.5773 * 0.0005;
    return normalize( e.xyy*map( pos + e.xyy ).x + 
					  e.yyx*map( pos + e.yyx ).x + 
					  e.yxy*map( pos + e.yxy ).x + 
					  e.xxx*map( pos + e.xxx ).x );
}

// Function 380
vec3 getNormal( in vec3 p ){

    // Note the larger than usual sampline distance (epsilon value). It's an old trick to give
    // rounded edges, and with the right objects it gives a slightly blurred antialiased look.
    vec2 e = vec2(0.015, -0.015);
    return normalize( e.xyy*map(p+e.xyy ) + e.yyx*map(p+e.yyx ) + e.yxy*map(p+e.yxy ) + e.xxx*map(p+e.xxx ));
}

// Function 381
vec3 getNormal(vec3 p) {
    vec2 e = vec2(.0001, 0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
	));
}

// Function 382
vec3 normal(in vec3 position) {
    vec3 epsilon = vec3(0.001, 0.0, 0.0);
    vec3 n = vec3(
          scene(position + epsilon.xyy).x - scene(position - epsilon.xyy).x,
          scene(position + epsilon.yxy).x - scene(position - epsilon.yxy).x,
          scene(position + epsilon.yyx).x - scene(position - epsilon.yyx).x);
    return normalize(n);
}

// Function 383
vec3 normal(in Ray ray) {
    vec2 eps = vec2(0.0001, 0);
    float baseDist = sceneDist(ray).d;
 	return normalize(vec3(
        sceneDist(Ray(ray.origin + eps.xyy, ray.dir)).d - 
        sceneDist(Ray(ray.origin - eps.xyy, ray.dir)).d,
        sceneDist(Ray(ray.origin + eps.yxy, ray.dir)).d -
        sceneDist(Ray(ray.origin - eps.yxy, ray.dir)).d,
        sceneDist(Ray(ray.origin + eps.yyx, ray.dir)).d -
        sceneDist(Ray(ray.origin - eps.yyx, ray.dir)).d
        ));
}

// Function 384
vec3 GetNormal(vec3 pos){
 	vec2 e = vec2(1.0,-1.0)*0.5773*0.005;
    return normalize( e.xyy*GetDist( pos + e.xyy ).d + 
					  e.yyx*GetDist( pos + e.yyx ).d + 
					  e.yxy*GetDist( pos + e.yxy ).d + 
					  e.xxx*GetDist( pos + e.xxx ).d );
}

// Function 385
void ColorAndNormal(vec3 hit, inout vec4 mcol, inout vec3 normal, vec2 tRoom, inout vec2 mref, inout float t, const int id)
{
	if(t == tRoom.y)
	{            
		mref = vec2(0.0,0.0);
        normal =-normalForCube(hit, box0);   
        if(normal.x>0.0)
        { 
            mcol.xyz = vec3(0.95,0.05,0.05);
        } 
        else if(normal.x<0.0)
        { 
            mcol.xyz = vec3(0.05,0.95,0.05);
        } 
	}     
	else   
	{
        	 if(id==0) {normal = normalForSphere(hit, sfere[0]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==1) {normal = normalForSphere(hit, sfere[1]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==2) {normal = normalForSphere(hit, sfere[2]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==6) {normal = normalForSphere(hit, sfere[3]); mcol = vec4(0.9,0.9,0.9,10.0); mref = vec2(0.0,0.0);}
    }  
}

// Function 386
vec3 mapNormal(vec3 pt, float e) {
    vec3 normal;
    normal.y = mapDetailed(pt).x;    
    normal.x = mapDetailed(vec3(pt.x+e,pt.y,pt.z)).x - normal.y;
    normal.z = mapDetailed(vec3(pt.x,pt.y,pt.z+e)).x - normal.y;
    normal.y = e;
    return normalize(normal);
}

// Function 387
vec3 calcNormal(in vec3 pos)
{
	vec2 e = vec2(1.0,-1.0)*0.5773;
	const float eps = 0.0005;
	return normalize(
		e.xyy*sdf(pos + e.xyy*eps) + 
		e.yyx*sdf(pos + e.yyx*eps) + 
		e.yxy*sdf(pos + e.yxy*eps) + 
		e.xxx*sdf(pos + e.xxx*eps)
	);
}

// Function 388
vec3 calcNormal( in vec3 pos )
{
    vec3  eps = vec3(precis,0.0,0.0);
    vec3 nor;
    nor.x = map(pos+eps.xyy).x - map(pos-eps.xyy).x;
    nor.y = map(pos+eps.yxy).x - map(pos-eps.yxy).x;
    nor.z = map(pos+eps.yyx).x - map(pos-eps.yyx).x;
    return normalize(nor);
}

// Function 389
vec3 normal(vec2 uv) {
    
    //edge normal, it mean it's the difference between a brick and the white edge
    //higher value mean bigger edge
    float r = 0.03;
    
    float x0 = lum(vec2(uv.x + r, uv.y));
    float x1 = lum(vec2(uv.x - r, uv.y));
    float y0 = lum(vec2(uv.x, uv.y - r));
    float y1 = lum(vec2(uv.x, uv.y + r));
    
    //NOTE: Controls the "smoothness"
    //it also mean how hard the edge normal will be
    //higher value mean smoother normal, lower mean sharper transition
    float s = 1.0;
    vec3 n = normalize(vec3(x1 - x0, y1 - y0, s));

    vec3 p = vec3(uv * 2.0 - 1.0, 0.0);
    vec3 v = vec3(0.0, 0.0, 1.0);

    vec3 l = v - p;
    float d_sqr = l.x * l.x + l.y * l.y + l.z * l.z;
    l *= (1.0 / sqrt(d_sqr));

    vec3 h = normalize(l + v);

    float dot_nl = clamp(dot(n, l), 0.0, 1.0);
    float dot_nh = clamp(dot(n, h), 0.0, 1.0);

    float color = lum(uv) * pow(dot_nh, 14.0) * dot_nl * (1.0 / d_sqr);
    color = pow(color, 1.0 / 2.2);

    return (n * 0.5 + 0.5);
 
}

// Function 390
vec3 getNormal (vec3 pos) {
  vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
  return normalize( e.xyy*map( pos + e.xyy ) + e.yyx*map( pos + e.yyx ) + e.yxy*map( pos + e.yxy ) + e.xxx*map( pos + e.xxx ) );
}

// Function 391
vec3 getNormal(in vec3 p) {
	const vec2 e = vec2(0.005, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 392
vec3 calcNormal( in vec3 p, in vec4 c )
{
#if 1
    vec2 e = vec2(1.0,-1.0)*0.5773*0.001;
    vec4 za=vec4(p+e.xyy,0.0); float mz2a=qlength2(za), md2a=1.0;
    vec4 zb=vec4(p+e.yyx,0.0); float mz2b=qlength2(zb), md2b=1.0;
    vec4 zc=vec4(p+e.yxy,0.0); float mz2c=qlength2(zc), md2c=1.0;
    vec4 zd=vec4(p+e.xxx,0.0); float mz2d=qlength2(zd), md2d=1.0;
  	for(int i=0; i<numIterations; i++)
    {
        md2a *= mz2a; za = qsqr(za)+c; mz2a = qlength2(za);
        md2b *= mz2b; zb = qsqr(zb)+c; mz2b = qlength2(zb);
        md2c *= mz2c; zc = qsqr(zc)+c; mz2c = qlength2(zc);
        md2d *= mz2d; zd = qsqr(zd)+c; mz2d = qlength2(zd);
    }
    return normalize( e.xyy*sqrt(mz2a/md2a)*log2(mz2a) + 
					  e.yyx*sqrt(mz2b/md2b)*log2(mz2b) + 
					  e.yxy*sqrt(mz2c/md2c)*log2(mz2c) + 
					  e.xxx*sqrt(mz2d/md2d)*log2(mz2d) );
#else    
    const vec2 e = vec2(0.001,0.0);
    vec4 za=vec4(p+e.xyy,0.0); float mz2a=qlength2(za), md2a=1.0;
    vec4 zb=vec4(p-e.xyy,0.0); float mz2b=qlength2(zb), md2b=1.0;
    vec4 zc=vec4(p+e.yxy,0.0); float mz2c=qlength2(zc), md2c=1.0;
    vec4 zd=vec4(p-e.yxy,0.0); float mz2d=qlength2(zd), md2d=1.0;
    vec4 ze=vec4(p+e.yyx,0.0); float mz2e=qlength2(ze), md2e=1.0;
    vec4 zf=vec4(p-e.yyx,0.0); float mz2f=qlength2(zf), md2f=1.0;
  	for(int i=0; i<numIterations; i++)
    {
        md2a *= mz2a; za = qsqr(za) + c; mz2a = qlength2(za);
        md2b *= mz2b; zb = qsqr(zb) + c; mz2b = qlength2(zb);
        md2c *= mz2c; zc = qsqr(zc) + c; mz2c = qlength2(zc);
        md2d *= mz2d; zd = qsqr(zd) + c; mz2d = qlength2(zd);
        md2e *= mz2e; ze = qsqr(ze) + c; mz2e = qlength2(ze);
        md2f *= mz2f; zf = qsqr(zf) + c; mz2f = qlength2(zf);
    }
    float da = sqrt(mz2a/md2a)*log2(mz2a);
    float db = sqrt(mz2b/md2b)*log2(mz2b);
    float dc = sqrt(mz2c/md2c)*log2(mz2c);
    float dd = sqrt(mz2d/md2d)*log2(mz2d);
    float de = sqrt(mz2e/md2e)*log2(mz2e);
    float df = sqrt(mz2f/md2f)*log2(mz2f);
    
    return normalize( vec3(da-db,dc-dd,de-df) );
#endif    
}

// Function 393
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

// Function 394
vec3 calcNormal( in vec3 pos, float e, vec3 dir)
{
    vec3 eps = vec3(e,0.0,0.0);

    return normalize(vec3(
           march(pos+eps.xyy, dir).y - march(pos-eps.xyy, dir).y,
           march(pos+eps.yxy, dir).y - march(pos-eps.yxy, dir).y,
           march(pos+eps.yyx, dir).y - march(pos-eps.yyx, dir).y ));
}

// Function 395
vec3 normal(in vec3 rp) {
    return normalize(vec3(df_hq(rp+ne)-df_hq(rp-ne),
                          df_hq(rp+ne.yxz)-df_hq(rp-ne.yxz),
                          df_hq(rp+ne.yzx)-df_hq(rp-ne.yzx)));
}

// Function 396
vec3 estimator_normals (in Ray r, in Scene s)
{
    Isect_data data;
    float t = intersect(r, 0.0, INF, data, s);
    
    /* return black if no intersection */
    if(!(t<INF)) return vec3(0);
    
    
    /* ---- return the normal if there is an intersection ---- */
    
    /* correct normal is in the oppoiste hemisphere of the incident ray */
    vec3 n = -data.n * sign(dot(data.n, r.d));

    /* map from S^2 to [0,1]^3 */
    return 0.5*n+vec3(0.5);
}

// Function 397
TangentView get_tangent_view( vec3 p, vec3 c, float r )
{
	TangentView ret;
	ret.target_vector = p - c;
	float dt = length( ret.target_vector );
	ret.target_vector /= dt;
	float e = ( r * r ) / dt;
	ret.tangent_disk_radius = sqrt( r * r - e * e );
	ret.tangent_disk_center = c + ret.target_vector * e;
	return ret;
}

// Function 398
vec3 getNormal(vec3 pos, float e, bool inside)
{  
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec3 e2 = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        vec2 mr = map(pos + e*e2, inside, false);
        n += e2*mr.x;
        #ifdef wall_bump
        n += e2*getNormal2(pos + e*e2, mr.y);
        #endif        
    }
    return normalize(n);
}

// Function 399
vec2 encodeNormal( in vec3 nor )
{
    nor /= ( abs( nor.x ) + abs( nor.y ) + abs( nor.z ) );
    nor.xy = (nor.z >= 0.0) ? nor.xy : (1.0-abs(nor.yx))*sign(nor.xy);
    vec2 v = 0.5 + 0.5*nor.xy;

    return v;
}

// Function 400
vec3 normal(in vec3 p) {
	const vec2 e = vec2(0.005, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 401
vec3 dNormal(in vec3 p, in float eps)
{
   vec2 e = vec2(eps,0.);
   return normalize(vec3(
      dField(p + e.xyy) - dField(p - e.xyy),
      dField(p + e.yxy) - dField(p - e.yxy),
      dField(p + e.yyx) - dField(p - e.yyx) ));
}

// Function 402
vec3 calcNormalDamage( in vec3 pos, in float eps ) {
    if( pos.y < 0.001 && (mapDamageHigh(pos)-map(pos)) < eps ) {   		
	        return vec3( 0., 1., 0. );
    }
    
    vec2 e = vec2(1.0,-1.0)*(0.5773*eps);
    vec3 n =  normalize( e.xyy*mapDamageHigh( pos + e.xyy ) + 
			     		 e.yyx*mapDamageHigh( pos + e.yyx ) + 
					  	 e.yxy*mapDamageHigh( pos + e.yxy ) + 
					  	 e.xxx*mapDamageHigh( pos + e.xxx ) );
    n = bumpMapNormal( pos, n );
    return n;    
}

// Function 403
vec3 calcNormal_2_5(vec3 pos, float eps) {
  const vec3 v1 = vec3( 1.0,-1.0,-1.0);
  const vec3 v2 = vec3(-1.0,-1.0, 1.0);
  const vec3 v3 = vec3(-1.0, 1.0,-1.0);
  const vec3 v4 = vec3( 1.0, 1.0, 1.0);

  return normalize( v1 * geometry( pos + v1*eps ).x +
                    v2 * geometry( pos + v2*eps ).x +
                    v3 * geometry( pos + v3*eps ).x +
                    v4 * geometry( pos + v4*eps ).x );
}

// Function 404
vec3 normal(vec3 p){
    mat3 e = mat3(0.001);
    return normalize(vec3(map(p+e[0]), map(p+e[1]), map(p+e[2])) - map(p));
}

// Function 405
vec3 getNormal(in vec2 sphereCenter, in float sphereRadius, in vec2 point){
	// Need to figure out how far the current point is from the center to lerp it
	float distFromCenter = distance(point, sphereCenter);
	float weight = distFromCenter/sphereRadius;
	
	// Z component is zero since at the edge the normal will be on the XY plane
	vec3 edgeNormal = vec3(point - sphereCenter, 0);
	edgeNormal = normalize(edgeNormal);
	
	// We know the normal at the center of the sphere points directly at the viewer,
	// so we can use that in our mix/lerp.
	return mix(vec3(0,0,1), edgeNormal, weight); 
}

// Function 406
vec3 normal(vec3 p, vec3 dir) {
    vec3 n = vec3(
        sceneDist(vec3(p.x + NORMAL_EPSILON, p.y, p.z)) - sceneDist(vec3(p.x - NORMAL_EPSILON, p.y, p.z)),
        sceneDist(vec3(p.x, p.y + NORMAL_EPSILON, p.z)) - sceneDist(vec3(p.x, p.y - NORMAL_EPSILON, p.z)),
        sceneDist(vec3(p.x, p.y, p.z  + NORMAL_EPSILON)) - sceneDist(vec3(p.x, p.y, p.z - NORMAL_EPSILON))
    );

    return normalize(n - max(0.0, dot(n, dir)) * dir);
}

// Function 407
vec3 CombineNormal(vec3 n1, vec3 n2, int technique)
{
 	if (technique == TECHNIQUE_RNM)
        return NormalBlend_RNM(n1, n2);
    else if (technique == TECHNIQUE_PartialDerivatives)
        return NormalBlend_PartialDerivatives(n1, n2);
    else if (technique == TECHNIQUE_Whiteout)
        return NormalBlend_Whiteout(n1, n2);
    else if (technique == TECHNIQUE_UDN)
        return NormalBlend_UDN(n1, n2);
    else if (technique == TECHNIQUE_Unity)
        return NormalBlend_Unity(n1, n2);
    else if (technique == TECHNIQUE_Linear)
        return NormalBlend_Linear(n1, n2);
    else
        return NormalBlend_Overlay(n1, n2);
}

// Function 408
vec3 calcNormal(vec3 p){
  const vec2 eps = vec2(0.0001, 0.0);
  // mathematical procedure.
  vec3 n;
  n.x = map(p + eps.xyy) - map(p - eps.xyy);
  n.y = map(p + eps.yxy) - map(p - eps.yxy);
  n.z = map(p + eps.yyx) - map(p - eps.yyx);
  return normalize(n);
}

// Function 409
vec3 normal(vec3 p){return normalize(vec3(cn(xyy),cn(yxy),cn(yyx)));}

// Function 410
vec3 getNormal    (vec3 p){vec2 b=vec2(0,.00001);return Derivative9Tap(Df ,p,b);}

// Function 411
vec3 calcNormal_1245821463(vec3 pos) {
  return calcNormal_1245821463(pos, 0.002);
}

// Function 412
vec3 sdSceneNormal(vec3 p) 
{
    return normalize(vec3(
        sdScene(vec3(p.x + GRAD_STEP, p.y, p.z)) - sdScene(vec3(p.x - GRAD_STEP, p.y, p.z)),
        sdScene(vec3(p.x, p.y + GRAD_STEP, p.z)) - sdScene(vec3(p.x, p.y - GRAD_STEP, p.z)),
        sdScene(vec3(p.x, p.y, p.z  + GRAD_STEP)) - sdScene(vec3(p.x, p.y, p.z - GRAD_STEP))
    ));
}

// Function 413
vec3 NormalBlend_Linear(vec3 n1, vec3 n2)
{
    // Unpack
	n1 = n1*2.0 - 1.0;
    n2 = n2*2.0 - 1.0;
    
	return normalize(n1 + n2);    
}

// Function 414
vec3 calculateNormal(vec3 pos)
{
	vec2 eps = vec2(EPS, 0.);
    return normalize(vec3(sdSceneNormal(pos + eps.xyy).x, 
                          sdSceneNormal(pos + eps.yxy).x, 
                          sdSceneNormal(pos + eps.yyx).x) 
                     - sdSceneNormal(pos).x);
}

// Function 415
vec3 normals(in vec3 p) {
  vec2 e = vec2(EPS, 0.0);
  vec3 gr = vec3( map(p + e.xyy) - map(p - e.xyy),
                  map(p + e.yxy) - map(p - e.yxy),
                  map(p + e.yyx) - map(p - e.yyx));
  return normalize(gr);
}

// Function 416
vec2 estimateNormal(vec2 p) {
    return normalize(vec2(
        mapSeed(vec2(p.x + EPSILON, p.y)) - mapSeed(vec2(p.x - EPSILON, p.y)),
        mapSeed(vec2(p.x, p.y + EPSILON)) - mapSeed(vec2(p.x, p.y - EPSILON))
    ));
}

// Function 417
vec3 normal(vec3 p, float d)
{
	float e = 0.05;
	float dx = scene(vec3(e, 0.0, 0.0) + p) - d;
	float dy = scene(vec3(0.0, e, 0.0) + p) - d;
	float dz = scene(vec3(0.0, 0.0, e) + p) - d;
	return normalize(vec3(dx, dy, dz));
}

// Function 418
vec3 normal(in vec3 p, float ef) {
	vec2 e = vec2(0.001*ef, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 419
vec3 normal2(in vec3 rp) {
    return normalize(vec3(df(rp+ne)-df(rp-ne),
                          df(rp+ne.yxz)-df(rp-ne.yxz),
                          df(rp+ne.yzx)-df(rp-ne.yzx)));
}

// Function 420
vec3 calcNormal(in vec3 p) {
	const vec2 e = vec2(0.002, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 421
vec3 normal(vec3 p) {
    vec2 h = vec2(0.001, 0.0);
    vec3 n = normalize(vec3(
        de(p + h.xyy) - de(p - h.xyy),
        de(p + h.yxy) - de(p - h.yxy),
        de(p + h.yyx) - de(p - h.yyx)
	));
    
    return n;
}

// Function 422
vec3 boxNormal(vec3 rp,vec3 p0,vec3 p1)
{
    rp = rp - (p0 + p1) / 2.;
    vec3 arp = abs(rp) / (p1 - p0);
    return step(arp.yzx, arp) * step(arp.zxy, arp) * sign(rp);
}

// Function 423
vec3 hnormalizet(vec3 v){
  // normalization of "time like" vectors.
  return v/hlengtht(v);
}

// Function 424
vec3 getNormal(vec3 pos, float e)
{  
    vec3 n = vec3(0.0);
    for( int i=0; i<4; i++ )
    {
        vec3 e2 = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e2*map(pos + e*e2).x;
    }
    return normalize(n);
}

// Function 425
vec3 getNormal(in vec2 uv)
{
    //This could probably be more efficient
    //dx
    vec2 dx = vec2(EPS,0.);
    vec3 t = vec3(EPS,rockVoronoi(uv+dx),0.);
    vec3 s = vec3(-EPS,rockVoronoi(uv-dx),0.);
    vec3 uv_dx = normalize((t - s));
    
    //dy
    vec2 dz = vec2(0.,EPS);
    t = vec3(0.,rockVoronoi(uv+dz),EPS);
    s = vec3(0.,rockVoronoi(uv-dz),-EPS);
    vec3 uv_dz = normalize((t - s));
    return cross(uv_dx, uv_dz);
}

// Function 426
vec3 normal(vec3 p) {
    vec2 NE = vec2(EPSILON, 0.);
    return normalize(vec3( df(p+NE.xyy)-df(p-NE.xyy),
                           df(p+NE.yxy)-df(p-NE.yxy),
                           df(p+NE.yyx)-df(p-NE.yyx) ));
}

// Function 427
vec3 getNormal(vec3 p) {
    float d = EPSILON;
    return normalize(vec3(
        sceneSDF(p + vec3(d, 0.0, 0.0)) - sceneSDF(p),
        sceneSDF(p + vec3(0.0, d, 0.0)) - sceneSDF(p),
        sceneSDF(p + vec3(0.0, 0.0, d)) - sceneSDF(p)
    ));
}

// Function 428
vec3 calcNormal(vec3 p) {
 
    vec2 eps = vec2(.001,0.);
    vec3 n   = vec3(dstScene(p+eps.xyy)-dstScene(p-eps.xyy),
                    dstScene(p+eps.yxy)-dstScene(p-eps.yxy),
                    dstScene(p+eps.yyx)-dstScene(p-eps.yyx));
    return normalize(n);
    
}

// Function 429
vec3 sake_normal(vec2 p, vec2 p0) {
    float dfx = sake_displacement(p + vec2(1.0, 0.0) * eps, p0) - sake_displacement(p - vec2(1.0, 0.0) * eps, p0);
    float dfy = sake_displacement(p + vec2(0.0, 1.0) * eps, p0) - sake_displacement(p - vec2(0.0, 1.0) * eps, p0);
    return normalize(vec3(dfx, dfy, 2.0 * eps));
}

// Function 430
vec3 GetNormal(vec3 p) {
	float d = GetDist(p);
    vec2 e = vec2(.001, 0.);
    
    vec3 n = d - vec3(
        GetDist(p - e.xyy),
        GetDist(p - e.yxy),
        GetDist(p - e.yyx));
    
    return normalize(n);
}

// Function 431
vec3 surface_normal( vec3 p )
{
    vec3 epsilon = vec3( 0.001, 0.0, 0.0 );
    vec3 n = vec3(
        scenedf(p + epsilon.xyy) - scenedf(p - epsilon.xyy),
        scenedf(p + epsilon.yxy) - scenedf(p - epsilon.yxy),
        scenedf(p + epsilon.yyx) - scenedf(p - epsilon.yyx) );
    return normalize( n );
}

// Function 432
vec3 getNormal_s(vec3 pos, float e)
{  
    vec2 q = vec2(0, e);
    return normalize(vec3(map_s(pos + q.yxx).x - map_s(pos - q.yxx).x,
                          map_s(pos + q.xyx).x - map_s(pos - q.xyx).x,
                          map_s(pos + q.xxy).x - map_s(pos - q.xxy).x));
}

// Function 433
vec3 normal_o349467(vec3 p){  
  const vec3 e=vec3(0.001,-0.001,0.0);
  float v1=input_o349467(p+e.xyy).x;
  float v2=input_o349467(p+e.yyx).x;
  float v3=input_o349467(p+e.yxy).x;
  float v4=input_o349467(p+e.xxx).x;
  return normalize(vec3(v4+v1-v3-v2,v3+v4-v1-v2,v2+v4-v3-v1));
}

// Function 434
vec3 normal(vec3 pos, float e) {
    vec3 eps = vec3(e, 0.0, 0.0);

    return normalize(vec3(
        map(pos + eps.xyy) - map(pos - eps.xyy),
        map(pos + eps.yxy) - map(pos - eps.yxy),
        map(pos + eps.yyx) - map(pos - eps.yyx)));
}

// Function 435
float shadedNormal( vec3 p, float v ) {
    float epsL = 0.01;
#if 1// centered directional derivative
    float dx = (tweaknoise(p+epsL*lightDir,false)-tweaknoise(p-epsL*lightDir,false))/(2.*epsL);
#else // cheap directional derivative
    float dx = (tweaknoise(p+epsL*lightDir,false)-v)/epsL;
#endif
    return clamp(-dx*grad/scale/v, 0.,1.); // Lambert shading
    
}

// Function 436
vec3 normal(vec3 p){const vec2 e=vec2(.001,0);return normalize(vec3(
 df(p+e.xyy)-df(p-e.xyy),df(p+e.yxy)-df(p-e.yxy),df(p+e.yyx)-df(p-e.yyx)));}

// Function 437
vec3 SampleNormalMap(in vec2 uv, in float height)
{
    const float strength = 40.0;    
    float d0 = SampleTexture(uv.xy);
    float dX = SampleTexture(uv.xy - vec2(EPSILON, 0.0));
    float dY = SampleTexture(uv.xy - vec2(0.0, EPSILON));
    return normalize(vec3((dX - d0) * strength, (dY - d0) * strength, 1.0));
}

// Function 438
vec3 calcNormal( in vec3 pos, in vec4 sh )
{
    vec3 eps = vec3(0.001,0.0,0.0);

	return normalize( vec3(
           map(pos+eps.xyy, sh).x - map(pos-eps.xyy, sh).x,
           map(pos+eps.yxy, sh).x - map(pos-eps.yxy, sh).x,
           map(pos+eps.yyx, sh).x - map(pos-eps.yyx, sh).x ) );
}

// Function 439
vec3 normalHigh(vec3 x) {
	const vec2 eps = vec2(0.1, 0.0);
	float h = sceneHigh(x);
	return normalize(vec3(
		(sceneHigh(x+eps.xyy)-h),
		(sceneHigh(x+eps.yxy)-h),
		(sceneHigh(x+eps.yyx)-h)
	));
}

// Function 440
vec3 normal(vec3 p) {
	vec2 e = vec2(1,0)/1e3;
    p += 0.01 * vec3(
        map(p - e.xyy) - map(p + e.xyy),
        map(p - e.yxy) - map(p + e.yxy),
        map(p - e.yyx) - map(p + e.yyx))/ (2. * length(e));
	return normalize(p);
}

// Function 441
void ColorAndNormal(vec3 hit, inout vec4 mcol, inout vec3 normal, vec2 tRoom, inout vec2 mref, inout float t, const int id)
{
	if(t == tRoom.y)
	{            
		mref = vec2(0.0,0.0);
        normal =-normalForCube(hit, box0);   
        if(abs(normal.x)>0.0)
        { 
            mcol.xyz = vec3(0.95,0.95,0.95);
            mref = vec2(0.0,1.0);
        } 
         else if(normal.y>0.0)
        {
            vec3 tcol = texture(iChannel1,1.0-(hit.xz-vec2(1.5,1.5))/3.5).xyz;
            float s = tcol.y+0.1;//-d
            s = pow(s,3.0)*0.75+0.01;
            mref = vec2((s*0.5+0.1),pow(1.0-s,2.0));
            mcol.xyz = vec3(0.9);//tcol+0.4;
        } 
        else if(abs(normal.z)>0.0)
        {
            mcol.xyz = vec3(0.95,0.15,0.19);
            mref = vec2(0.0,1.0);
            
            if(normal.z<0.0)
			{
            	//cw = vec2(-0.4,0.1);
            	if(	all(lessThanEqual(hit.xy,vec2(-0.05,0.6)+cw)) &&
               		all(greaterThanEqual(hit.xy,vec2(-0.7,-0.6)+cw)) ||
               		all(lessThanEqual(hit.xy,vec2(0.7,0.6)+cw)) &&
               		all(greaterThanEqual(hit.xy,vec2(0.05,-0.6)+cw)))
               		mcol = vec4(vec3(1.1),2.0);
			}
        }
	}     
	else   
	{
        	 if(id==0) {normal = normalForSphere(hit, sfere[0]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==1) {normal = normalForSphere(hit, sfere[1]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==2) {normal = normalForSphere(hit, sfere[2]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==6) {normal = normalForSphere(hit, sfere[3]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
    	else if(id==10) {normal = normalforCylinder(hit, cylinder[0]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,1000.0);}
        else if(id==11) {normal = normalforCylinder(hit, cylinder[1]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,1000.0);}
        else if(id==12) {normal = normalforCylinder(hit, cylinder[2]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,1000.0);}
        else if(id==13) {normal = normalforCylinder(hit, cylinder[3]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,1000.0);}
        else if(id==20) {normal = normalForCube(hit, boxe[0]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==21) {normal = normalForCube(hit, boxe[1]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==22) {normal = normalForCube(hit, boxe[2]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==23) {normal = normalForCube(hit, boxe[3]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,9000.0);}
        else if(id==24) {normal = normalForCube(hit, boxe[4]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,9000.0);}
        else if(id==25) {normal = normalForCube(hit, boxe[5]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,10.0);}
        else if(id==26) {normal = normalForCube(hit, boxe[6]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,10.0);}
        else if(id==27) {normal = normalForCube(hit, boxe[7]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.8,0.8);}
        else if(id==28) {normal = normalForCube(hit, boxe[8]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==29) {normal = normalForCube(hit, boxe[9]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==30) {normal = normalForCube(hit, boxe[10]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==31) {normal = normalForCube(hit, boxe[11]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==32) {normal = normalForCube(hit, boxe[12]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==33) {normal = normalForCube(hit, boxe[13]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==34) {normal = normalForCube(hit, boxe[14]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.05,3.8);}
        
        if(id>19 && id<23)//material for dulap
        {
            vec2 uv = hit.yz;
            uv = abs(normal.y) > 0.0 ? hit.zx : uv;
            uv = abs(normal.z) > 0.0 ? hit.yx : uv; 
            mcol.xyz = texture(iChannel1,1.0-(uv - vec2(1.5,-1.0))/vec2(5.5,0.5)).xyz - vec3(0.35,0.2,0.2);
            mref = vec2(0.0,0.2);// transparent, glossines
            mcol.xyz = vec3(0.1,0.99,0.1);// color
            
            if(id==21)	normal = -normal;
        }
        
        if(id>26 && id<34)//masa scaun
        {
            mcol.xyz = vec3(0.9);
            mref = vec2(0.0,0.7);// transparent, glossines
            //if(id==27) mcol.xyz = vec3(0.9,0.9,0.9);// color
            
            if(id==21)	normal = -normal;
        }
        
        if(id==34)//calorifer
        {
            mcol.xyz = vec3(sin(hit.x*59.0)+2.0-0.2);
            mref = vec2(0.0,0.0);
        }
    }  
}

// Function 442
vec3 calcNormal( in vec3 p )
{
    #ifdef TRAPS
    the code below only works for the actual Julia set, not the traps
    #endif
        
    vec4 z = vec4(p,0.0);

    // identity derivative
    mat4x4 J = mat4x4(1,0,0,0,  
                      0,1,0,0,  
                      0,0,1,0,  
                      0,0,0,1 );

  	for(int i=0; i<kNumIte; i++)
    {
        // f(q) = q³ + c = 
        //   x =  x²x - 3y²x - 3z²x - 3w²x + c.x
        //   y = 3x²y -  y²y -  z²y -  w²y + c.y
        //   z = 3x²z -  y²z -  z²z -  w²z + c.z
        //   w = 3x²w -  y²w -  z²w -  w²w + c.w
		//
        // Jacobian, J(f(q)) =
        //   3(x²-y²-z²-w²)  6xy            6xz            6xw
        //    -6xy           3x²-3y²-z²-w² -2yz           -2yw
        //    -6xz          -2yz            3x2-y²-3z²-w² -2zw
        //    -6xw          -2yw           -2zw            3x²-y²-z²-3w²
        
        float k1 = 6.0*z.x*z.y, k2 = 6.0*z.x*z.z;
        float k3 = 6.0*z.x*z.w, k4 = 2.0*z.y*z.z;
        float k5 = 2.0*z.y*z.w, k6 = 2.0*z.z*z.w;
        float sx = z.x*z.x, sy = z.y*z.y;
        float sz = z.z*z.z, sw = z.w*z.w;
        float mx = 3.0*sx-3.0*sy-3.0*sz-3.0*sw;
        float my = 3.0*sx-3.0*sy-    sz-    sw;
        float mz = 3.0*sx-    sy-3.0*sz-    sw;
        float mw = 3.0*sx-    sy-    sz-3.0*sw;
        
        // chain rule of jacobians
        J = J*mat4x4( mx, -k1, -k2, -k3,
                      k1,  my, -k4, -k5,
                      k2, -k4,  mz, -k6,
                      k3, -k5, -k6,  mw );
        // q = q³ + c
        z = qCube(z) + kC; 
        
        // exit condition
        if(dot2(z)>256.0) break;
    }

    return (p.y>0.0 ) ? vec3(0.0,1.0,0.0) : normalize( (J*z).xyz );
}

// Function 443
vec3 normalEstimation(vec3 pos){
  vec2 k = vec2(MinDist, 0);
  return normalize(vec3(sdf(pos + k.xyy) - sdf(pos - k.xyy),
	  					sdf(pos + k.yxy) - sdf(pos - k.yxy),
  						sdf(pos + k.yyx) - sdf(pos - k.yyx)));
}

// Function 444
vec3 calcNormal( in vec3 pos, float e )
{
    vec3 eps = vec3(e,0.0,0.0);

	return normalize( vec3(
           map(pos+eps.xyy).x - map(pos-eps.xyy).x,
           map(pos+eps.yxy).x - map(pos-eps.yxy).x,
           map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
}

// Function 445
vec3 GetNormal( vec3 p )
{
    vec2 d = vec2(-1,1)*epsilon;
    return normalize(
        	SDF(p+d.xxx).x*d.xxx +
        	SDF(p+d.yyx).x*d.yyx +
        	SDF(p+d.yxy).x*d.yxy +
        	SDF(p+d.xyy).x*d.xyy
        );
}

// Function 446
vec3 calcWallNormal( vec3 p,
                     vec3 rdir )
{
    vec3 epsilon = vec3( EPSILON, 0.0, 0.0 );
    vec3 n = vec3(
        boxinteriortrace(p + epsilon.xyy, rdir, BOXWALL_EXTENT) - 
            boxinteriortrace(p - epsilon.xyy, rdir, BOXWALL_EXTENT),
        boxinteriortrace(p + epsilon.yxy, rdir, BOXWALL_EXTENT) - 
            boxinteriortrace(p - epsilon.yxy, rdir, BOXWALL_EXTENT),
        boxinteriortrace(p + epsilon.yyx, rdir, BOXWALL_EXTENT) - 
            boxinteriortrace(p - epsilon.yyx, rdir, BOXWALL_EXTENT) );
    return normalize( n );
}

// Function 447
vec3 getTriangleNormal(vec3 v0, vec3 v1, vec3 v2) {
    return normalize(cross(v1-v0,v2-v0));
}

// Function 448
vec3 normalForCube(vec3 hit, Box cube)
{  
   if(hit.x < cube.min.x + 0.0001) return vec3(-1.0, 0.0, 0.0);   
   else if(hit.x > cube.max.x - 0.0001) return vec3( 1.0, 0.0, 0.0);   
   else if(hit.y < cube.min.y + 0.0001) return vec3(0.0, -1.0, 0.0);   
   else if(hit.y > cube.max.y - 0.0001) return vec3(0.0, 1.0, 0.0);      
   else if(hit.z < cube.min.z + 0.0001) return vec3(0.0, 0.0, -1.0);   
   else return vec3(0.0, 0.0, 1.0);   
}

// Function 449
mat3 mat3FromNormal(in vec3 n) {
    vec3 x, y;
    branchlessONB(n, x, y);
    return mat3(x,y,n);
}

// Function 450
vec3 treesNormal( in vec3 pos, in float t )
{
    float kk1, kk2, kk3;
#if 0    
    const float eps = 0.005;
    vec2 e = vec2(1.0,-1.0)*0.5773*eps;
    return normalize( e.xyy*treesMap( pos + e.xyy, t, kk1, kk2, kk3 ) + 
                      e.yyx*treesMap( pos + e.yyx, t, kk1, kk2, kk3 ) + 
                      e.yxy*treesMap( pos + e.yxy, t, kk1, kk2, kk3 ) + 
                      e.xxx*treesMap( pos + e.xxx, t, kk1, kk2, kk3 ) );            
#else
    // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*treesMap(pos+0.005*e, t, kk1, kk2, kk3);
    }
    return normalize(n);
#endif    
}

// Function 451
vec3 computeNormalSphere(vec3 p,Sphere s)
{
	return normalize(p - s.center);
}

// Function 452
mat3 orthonormal_basis(in vec3 n)
{
    vec3 f,r;
    if(n.z < -0.999999)
    {
        f = vec3(0 , -1, 0);
        r = vec3(-1, 0, 0);
    }
    else
    {
        float a = 1./(1. + n.z);
        float b = -n.x*n.y*a;
        f = normalize(vec3(1. - n.x*n.x*a, b, -n.x));
        r = normalize(vec3(b, 1. - n.y*n.y*a , -n.y));
    }
    return( mat3(f,r,n) );
}

// Function 453
vec3 mapRMWaterNormal(vec3 pt, float e) {
    vec3 normal;
    normal.y = mapWaterDetailed(pt);    
    normal.x = mapWaterDetailed(vec3(pt.x+e,pt.y,pt.z)) - normal.y;
    normal.z = mapWaterDetailed(vec3(pt.x,pt.y,pt.z+e)) - normal.y;
    normal.y = e;
    return normalize(normal);
}

// Function 454
vec3 getNormal( in vec3 p ) {
	vec3 e = vec3(0.0, 0.03, 0.0);
	return normalize(vec3(
		de(p+e.yxx)-de(p-e.yxx),
		de(p+e.xyx)-de(p-e.xyx),
		de(p+e.xxy)-de(p-e.xxy)));	
}

// Function 455
vec3 Normal(vec3 p)
{
    vec3 off = vec3(NORMAL_OFFS, 0, 0);
    return normalize
    ( 
        vec3
        (
            Scene(p + off.xyz).x - Scene(p - off.xyz).x,
            Scene(p + off.zxy).x - Scene(p - off.zxy).x,
            Scene(p + off.yzx).x - Scene(p - off.yzx).x
        )
    );
}

// Function 456
vec3 getNormal(vec3 pos, float e)
{  
    vec2 q = vec2(0, e);
    return normalize(vec3(map(pos + q.yxx).x - map(pos - q.yxx).x,
                          map(pos + q.xyx).x - map(pos - q.xyx).x,
                          map(pos + q.xxy).x - map(pos - q.xxy).x));
}

// Function 457
vec3 calcNormalOpaque( in vec3 pos, in float eps )
{
    vec4 kk;
#if 0
    vec2 e = vec2(1.0,-1.0)*0.5773*eps;
    return normalize( e.xyy*mapOpaque( pos + e.xyy, kk ).x + 
					  e.yyx*mapOpaque( pos + e.yyx, kk ).x + 
					  e.yxy*mapOpaque( pos + e.yxy, kk ).x + 
					  e.xxx*mapOpaque( pos + e.xxx, kk ).x );
#else
    // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*mapOpaque(pos+eps*e,kk).x;
    }
    return normalize(n);
#endif    
}

// Function 458
vec3 calcNormal( in vec3 pos )
{
   
    vec3 nor = vec3(map(pos+eps.xyy).x - map(pos-eps.xyy).x,map(pos+eps.yxy).x - map(pos-eps.yxy).x,map(pos+eps.yyx).x - map(pos-eps.yyx).x );
    return normalize(nor);
}

// Function 459
vec3 calcNormal(vec3 p, vec3 ro) {
    vec2 e = vec2(1.0, -1.0) * 0.005773;
    return normalize(e.xyy * map(p + e.xyy, ro).x + 
					 e.yyx * map(p + e.yyx, ro).x + 
					 e.yxy * map(p + e.yxy, ro).x + 
					 e.xxx * map(p + e.xxx, ro).x);
}

// Function 460
vec3 NormalBlend_RNM(vec3 n1, vec3 n2)
{
    // Unpack (see article on why it's not just n*2-1)
	n1 = n1*vec3( 2,  2, 2) + vec3(-1, -1,  0);
    n2 = n2*vec3(-2, -2, 2) + vec3( 1,  1, -1);
    
    // Blend
    return n1*dot(n1, n2)/n1.z - n2;
}

// Function 461
vec2 SceneNormal(vec2 p, float h, out float d)
{
    vec2 n = vec2(
          Scene(p + vec2(h,0))
        , Scene(p + vec2(0,h))
      ) - (d = Scene(p));
    if (dot(n,n) < 1e-7) n = vec2(0,1);
    else n = normalize(n);
    return n;
}

// Function 462
vec2 ct_vfield_normal(
    in vec2 p,
    float npow
){
    vec2 g = vec2(0.0, 0.0);
    
    const int imax = CT_N + 1;
    
    for (int i = 0; i < imax; ++i)
    {
        vec2 dif = g_vfp[i].p - p;
        float sum = dif[0] * dif[0] + dif[1] * dif[1];
        float mass = pow(sum, npow);
        
      	g[0] = g[0] + g_vfp[i].m * dif[0] / mass;
      	g[1] = g[1] + g_vfp[i].m * dif[1] / mass;
    }
    
    return normalize(g);
}

// Function 463
vec3 calcNormal( in vec3 pos, in float t )
{
    vec3 eps = vec3( max(0.02,0.001*t),0.0,0.0);
	return normalize( vec3(
           mapTerrain(pos+eps.xyy) - mapTerrain(pos-eps.xyy),
           mapTerrain(pos+eps.yxy) - mapTerrain(pos-eps.yxy),
           mapTerrain(pos+eps.yyx) - mapTerrain(pos-eps.yyx) ) );

}

// Function 464
vec3 normal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)).dist - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)).dist,
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)).dist - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)).dist,
        sceneSDF(vec3(p.x, p.y, p.z + EPSILON)).dist - sceneSDF(vec3(p.x, p.y, p.z - EPSILON)).dist
    ));
}

// Function 465
vec3 heightFieldNormal(vec2 uv)
{
    vec2 e = vec2(1. / 512., 0);
    float h0 = heightField(uv);
    float h1 = heightField(uv + e.xy);
    float h2 = heightField(uv + e.yx);
    float h3 = heightField(uv - e.xy);
    float h4 = heightField(uv - e.yx);
    return normalize(cross(vec3(0, h2 - h4, e.x * 2.), vec3(e.x * 2., h1 - h3, 0)));
}

// Function 466
vec3 get_normal(vec3 p) {
	const vec2 e = vec2(0.002, 0);
	return normalize(vec3(map(p + e.xyy)-map(p - e.xyy), 
                          map(p + e.yxy)-map(p - e.yxy),	
                          map(p + e.yyx)-map(p - e.yyx)));
}

// Function 467
vec3 calculateNormal(vec3 p) {
 
    vec3 epsilon = vec3(0.001, 0., 0.);
    
    vec3 n = vec3(map(p + epsilon.xyy).x - map(p - epsilon.xyy).x,
                  map(p + epsilon.yxy).x - map(p - epsilon.yxy).x,
                  map(p + epsilon.yyx).x - map(p - epsilon.yyx).x);
    
    return normalize(n);
}

// Function 468
vec2 raymarch_main_scene_normals(vec3 _p, float t)
{
    return scene_min(scene_base_sand(_p), scene_pyramids(_p));
}

// Function 469
vec3 calcNormal( in vec3 pos )
{
    vec3 eps = vec3(0.002,0.0,0.0);

    float f = map(pos).x;
	return normalize( vec3(
           map(pos+eps.xyy).x - f,
           map(pos+eps.yxy).x - f,
           map(pos+eps.yyx).x - f ) );
}

// Function 470
vec3 calcNormal( in vec3 p, in vec4 c )
{
    vec4 z = vec4(p,0.0);

    // identity derivative
    mat4x4 J = mat4x4(1,0,0,0,  
                      0,1,0,0,  
                      0,0,1,0,  
                      0,0,0,1 );

  	for(int i=0; i<numIterations; i++)
    {
        // chain rule of jacobians (removed the 2 factor)
        J = J*mat4x4(z.x, -z.y, -z.z, -z.w, 
                     z.y,  z.x,  0.0,  0.0,
                     z.z,  0.0,  z.x,  0.0, 
                     z.w,  0.0,  0.0,  z.x);

        // z -> z2 + c
        z = qsqr(z) + c; 
        
        if(qlength2(z)>4.0) break;
    }

    return normalize( (J*z).xyz );
}

// Function 471
float normalizeColorChannel(in float value, in float min, in float max) {
    return (value - min)/(max-min);
}

// Function 472
vec3 normal(in vec3 p) {
    vec3 eps = vec3(EPSILON, 0.0, 0.0);
    return normalize(vec3(
        	map(p + eps.xyy) - map(p - eps.xyy),
        	map(p + eps.yxy) - map(p - eps.yxy),
        	map(p + eps.yyx) - map(p - eps.yyx)
        ));
}

// Function 473
vec3 escherNormal(vec2 p)
{
    vec2 pp = mod(p,1.0);
    
    float d = 10000000.0;
    int i_min, j_min;
    
    float d_temp;
    
    for(int i=0; i<vert.length()-1; ++i)
    {
        
        for(int j=0; j<textureTiles.length(); ++j)
        {
            d_temp = PointSegDistance2(pp+textureTiles[j], vert[i], vert[i+1]);
            if(d_temp < d)
            {
                d = d_temp;
                i_min = i;
                j_min = j;
            }
        }
    }
    
    d = sqrt(d);
    
    vec2 proj = PointSegProj(pp+textureTiles[j_min], vert[i_min], vert[i_min+1]);
    
    vec2 dir = proj-(pp+textureTiles[j_min]);
    
    //float alpha = M_PI/4.0;
    //vec3 n = vec3(dir/d*cos(alpha), sin(alpha));
    vec3 n = normalize(vec3(dir/d, 1.0));
    
    return n;
}

// Function 474
vec3 calcNormal( in vec3 pos )
{
    // epsilon = a small number
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
    
    return normalize( e.xyy*map( pos + e.xyy ).x + 
					  e.yyx*map( pos + e.yyx ).x + 
					  e.yxy*map( pos + e.yxy ).x + 
					  e.xxx*map( pos + e.xxx ).x );
    /*
	vec3 eps = vec3( 0.0005, 0.0, 0.0 );
	vec3 nor = vec3(
	    map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	    map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	    map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
	*/
}

// Function 475
vec3 llamelGetNormal( in vec3 ro ) {
    vec2 e = vec2(1.0,-1.0)*0.001;

    return normalize( e.xyy*llamelMap( ro + e.xyy ) + 
					  e.yyx*llamelMap( ro + e.yyx ) + 
					  e.yxy*llamelMap( ro + e.yxy ) + 
					  e.xxx*llamelMap( ro + e.xxx ) );
}

// Function 476
vec4 getnormal(vec3 ro) {
    vec2 d = vec2(eps, 0.0);
    float x1 = map(ro+d.xyy).x;
    float x2 = map(ro-d.xyy).x;
    float y1 = map(ro+d.yxy).x;
    float y2 = map(ro-d.yxy).x;
    float z1 = map(ro+d.yyx).x;
    float z2 = map(ro-d.yyx).x;
    return vec4(normalize(vec3(
        x1-x2,
        y1-y2,
        z1-z2)),
        x1+x2+y1+y2+z1+z2-6.*map(ro).x);
}

// Function 477
float ct_normal_pi(
    in vec2 z,
    in float sa
){
    float a = atan(z[1], z[0]) + sa;
    if (a < 0.0) a += 6.28318;
    a /= 6.28318;
    return a;
}

// Function 478
vec3 GetNormal(vec3 p)
{
    int dummy = 0;
    float d = CombineSDF(vec3(p.x , p.y, p.z), dummy);
    return 
    normalize(vec3(
		CombineSDF(vec3(p.x + EPS, p.y, p.z), dummy) - d,
		CombineSDF(vec3(p.x, p.y + EPS, p.z), dummy) - d,
		CombineSDF(vec3(p.x, p.y, p.z + EPS), dummy) - d
		));
}

// Function 479
vec3 normal_o354282(vec3 p) {
	float d = o354282_input_in1(p);
    vec2 e = vec2(.001,0);
    vec3 n = d - vec3(
        o354282_input_in1(p-vec3(e.xyy)),
        o354282_input_in1(p-vec3(e.yxy)),
        o354282_input_in1(p-vec3(e.yyx)));
    return normalize(n);
}

// Function 480
vec3 calcNormal( in vec3 pos, in float time, in float doDisplace )
{
    const float eps = 0.0005;
#if 0    
    vec2 e = vec2(1.0,-1.0)*0.5773;
    return normalize( e.xyy*map( pos + e.xyy*eps,time,doDisplace ).x + 
					  e.yyx*map( pos + e.yyx*eps,time,doDisplace ).x + 
					  e.yxy*map( pos + e.yxy*eps,time,doDisplace ).x + 
					  e.xxx*map( pos + e.xxx*eps,time,doDisplace ).x );
#else
    // trick by klems, to prevent the compiler from inlining map() 4 times
    vec4 n = vec4(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec4 s = vec4(pos, 0.0);
        s[i] += eps;
        n[i] = map(s.xyz, time, doDisplace).x;
    }
    return normalize(n.xyz-n.w);
#endif   
}

// Function 481
vec3 normal(vec3 p){
    const float eps = 0.005;
    return normalize(vec3(dist(p+vec3(eps,0,0))-dist(p-vec3(eps,0,0)),
                          dist(p+vec3(0,eps,0))-dist(p-vec3(0,eps,0)),
                          dist(p+vec3(0,0,eps))-dist(p-vec3(0,0,eps))));
}

// Function 482
vec3 calcNormal( in vec3 pos )
{
    vec3 eps = vec3(0.001,0.0,0.0);

	return normalize( vec3(
           map(pos+eps.xyy).x - map(pos-eps.xyy).x,
           map(pos+eps.yxy).x - map(pos-eps.yxy).x,
           map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
}

// Function 483
vec3 calcNormal( in vec3 pos )
{
	vec3 eps = vec3( 0.01, 0.0, 0.0 );
	vec3 nor = vec3(
   map(pos+eps.xyy).x - map(pos-eps.xyy).x,
   map(pos+eps.yxy).x - map(pos-eps.yxy).x,
   map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
}

// Function 484
vec3 mapUnderWaterNormal(vec3 pt, float e) {
    vec3 normal;
    normal.y = mapUnderWater(pt).x;    
    normal.x = mapUnderWater(vec3(pt.x+e,pt.y,pt.z)).x - normal.y;
    normal.z = mapUnderWater(vec3(pt.x,pt.y,pt.z+e)).x - normal.y;
    normal.y = e;
    return normalize(normal);
}

// Function 485
vec3 getNormalInterp(vec3 lmn, float time) {
    vec3 grad = vec3(
        DENSITY(lmn + vec3(0.1, 0.0, 0.0)) - DENSITY(lmn - vec3(0.1, 0.0, 0.0)),
        DENSITY(lmn + vec3(0.0, 0.1, 0.0)) - DENSITY(lmn - vec3(0.0, 0.1, 0.0)),
        DENSITY(lmn + vec3(0.0, 0.0, 0.1)) - DENSITY(lmn - vec3(0.0, 0.0, 0.1))
    );
    return -grad/(length(grad) + 1e-5);
}

// Function 486
vec3 Normal(in vec3 pos, in float t)
{
	vec2  eps = vec2(.25,0.0);
	vec3 nor = vec3(Map(pos+eps.xyy).x - Map(pos-eps.xyy).x,
					Map(pos+eps.yxy).x - Map(pos-eps.yxy).x,
					Map(pos+eps.yyx).x - Map(pos-eps.yyx).x);
	return normalize(nor);
}

// Function 487
vec3 getNormal( in vec3 pos )
{
	vec2 eps = vec2( EPS_NOR, 0.0 );	
	vec3 nor = vec3(
		map(pos+eps.xyy).d - map(pos-eps.xyy).d,
		map(pos+eps.yxy).d - map(pos-eps.yxy).d,
		map(pos+eps.yyx).d - map(pos-eps.yyx).d );
    
	return normalize(nor);
}

// Function 488
vec3 normal( in vec3 p )
{
	vec3 eps = vec3(0.001, 0.0, 0.0);
	return normalize( vec3(
		map(p+eps.xyy)-map(p-eps.xyy),
		map(p+eps.yxy)-map(p-eps.yxy),
		map(p+eps.yyx)-map(p-eps.yyx)
	) );
}

// Function 489
void OrthonormalBasisRH(vec3 n, out vec3 ox, out vec3 oz)
{
	float sig = n.z < 0.0 ? 1.0 : -1.0;
	
	float a = 1.0 / (n.z - sig);
	float b = n.x * n.y * a;
	
	ox = vec3(1.0 + sig * n.x * n.x * a, sig * b, sig * n.x);
	oz = vec3(b, sig + n.y * n.y * a, n.y);
}

// Function 490
vec3 calcNormal( in vec3 pos )
{
#ifdef ANALYTIC_NORMALS	
	return norMetaBalls( pos );
#else	
    vec3 eps = vec3(precis,0.0,0.0);
	return normalize( vec3(
           map(pos+eps.xyy) - map(pos-eps.xyy),
           map(pos+eps.yxy) - map(pos-eps.yxy),
           map(pos+eps.yyx) - map(pos-eps.yyx) ) );
#endif
}

// Function 491
vec2 safe_normalize(vec2 v) {
	float l = dot(v, v);
	return l > 0. ? v/sqrt(l) : v;
}

// Function 492
vec3 normalLowDetailModel(vec3 P)
{
	vec2 eps = vec2(0.,0.001);
    return normalize(vec3(
        SD_LowDetailModel(P+eps.yxx).d - SD_LowDetailModel(P-eps.yxx).d, 
		SD_LowDetailModel(P+eps.xyx).d - SD_LowDetailModel(P-eps.xyx).d, 
        SD_LowDetailModel(P+eps.xxy).d - SD_LowDetailModel(P-eps.xxy).d));
}

// Function 493
vec2 unpack_normal(float x) {
    float a = x * 2.0;
    vec2 s;
    s.x = sign(a);
    s.y = sign(1.0 - a*s.x);
    vec2 q;
    q.y = fract(a) - 0.5;
    q.x = sqrt(0.5 - q.y*q.y);
    return q*mat2(s.y,-s.x,s.xy);
}

// Function 494
vec3 normals(vec3 p){
  vec3 eps = vec3(1.0/iResolution.x, 0., 0.);
  vec3 n = vec3(
    distf(p+eps.xyy) - distf(p-eps.xyy),
    distf(p+eps.yxy) - distf(p-eps.yxy),
    distf(p+eps.yyx) - distf(p-eps.yyx));
  return normalize(n);
}

// Function 495
vec3 getNormal( in vec3 p )
{
    vec2 e = vec2(0.005, -0.005);
    return normalize(
        e.xyy * Cube(p + e.xyy) +
        e.yyx * Cube(p + e.yyx) +
        e.yxy * Cube(p + e.yxy) +
        e.xxx * Cube(p + e.xxx));
}

// Function 496
vec3 calcNormal( in vec3 pos )
{
    const vec2 e = vec2(1.0,-1.0)*0.5773*kPrecis;
    return normalize( e.xyy*map( pos + e.xyy ).x + 
					  e.yyx*map( pos + e.yyx ).x + 
					  e.yxy*map( pos + e.yxy ).x + 
					  e.xxx*map( pos + e.xxx ).x );
}

// Function 497
vec3 getNormal (vec2 p) {
    return normalize(
        cross(
            vec3(
                epsilon,
                0.,
                y(vec2(p.x+epsilon, p.y))-y(p)
            ),
            vec3(
                0.,
                epsilon,
                y(vec2(p.x, p.y+epsilon))-y(p)
            )
       	)
    );
}

// Function 498
vec2 normal_pack(vec3 n)
{
    return n.xy*0.5+0.5;
}

// Function 499
vec3 calcNormal( in vec3 pos )
{
	vec3 eps = vec3( 0.001, 0.0, 0.0 );
	vec3 nor = vec3(
	    map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	    map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	    map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
}

// Function 500
vec3 ToTangentSpace(vec3 normal,vec3 vector){
	return ToOtherSpaceCoord(CoordBase(normal),vector);
}

// Function 501
vec3 calcNormal( in vec3 pos )
{
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
    return normalize( e.xyy*map( pos + e.xyy ).x + 
					  e.yyx*map( pos + e.yyx ).x + 
					  e.yxy*map( pos + e.yxy ).x + 
					  e.xxx*map( pos + e.xxx ).x );
    /*
	vec3 eps = vec3( 0.0005, 0.0, 0.0 );
	vec3 nor = vec3(
	    map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	    map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	    map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
	*/
}

// Function 502
vec3 GetNormal(vec3 p, vec3 ro, float dO) {
	float d = GetDist(p, ro);
    vec2 e = vec2(.005, 0); // ##
    
    vec3 n = d - vec3(
        GetDist(p-e.xyy, ro),
        GetDist(p-e.yxy, ro),
        GetDist(p-e.yyx, ro));
    
    return normalize(n);
}

// Function 503
vec3 normalEstimation(vec3 pos){
	float dist = distanceEstimation(pos);
	vec3 xDir = vec3(dist, 0, 0);
	vec3 yDir = vec3(0, dist, 0);
	vec3 zDir = vec3(0, 0, dist);
	return normalize(vec3(	distanceEstimation(pos + xDir),
							distanceEstimation(pos + yDir),
							distanceEstimation(pos + zDir))
					- vec3(dist));
}

// Function 504
vec3 getNormal(vec3 p, float dens) {
    vec3 n;
    n.x = map(vec3(p.x+EPSILON,p.y,p.z));
    n.y = map(vec3(p.x,p.y+EPSILON,p.z));
    n.z = map(vec3(p.x,p.y,p.z+EPSILON));
    return normalize(n-dens);
}

// Function 505
vec3 surfaceNormal(vec3 pos)
{
 	vec3 delta = vec3(0.01, 0.0, 0.0);
    vec3 normal;
    normal.x = map(pos + delta.xyz) - map(pos - delta.xyz);
    normal.y = map(pos + delta.yxz) - map(pos - delta.yxz);
    normal.z = map(pos + delta.zyx) - map(pos - delta.zyx);
    return normalize(normal);
}

// Function 506
vec3 eliNormal( vec3 p, vec3 cen, vec3 rad )
{
    return normalize( (p-cen)/rad );
}

// Function 507
vec3 getNormal(vec3 p){
    float c;
    int o;
	return normalize(vec3(distScene(p + vec3(EPSN, 0., 0.), o, c) - distScene(p - vec3(EPSN, 0., 0.), o, c),
    					  distScene(p + vec3(0., EPSN, 0.), o, c) - distScene(p - vec3(0., EPSN, 0.), o, c),
                          distScene(p + vec3(0., 0., EPSN), o, c) - distScene(p - vec3(0., 0., EPSN), o, c)));
}

// Function 508
vec3 normal_decode(vec2 enc)
{
    vec3 n;
    n.xy = enc*2.0-1.0;
    n.z = sqrt(1.0-dot(n.xy, n.xy));
    return n;
}

// Function 509
vec2 NormalizeScreenCoords(vec2 uv, float zoom) {
	uv -= iResolution.xy / 2.;
    uv /= max(iResolution.x, iResolution.y);
    return uv * zoom;
}

// Function 510
vec3 getTangent(vec3 norm)
{
	vec3 tangent;
	vec3 c1 = cross(norm, vec3(0.0, 0.0, 1.0));
	vec3 c2 = cross(norm, vec3(0.0, 1.0, 0.0));
	if (dot(c1, c1) > dot(c2, c2))
		tangent = c1;
	else
		tangent = c2;
	return tangent;
}

// Function 511
vec3 normal(vec2 p)
{
    float s = 0.0025;
    float a = terrain(p + s * vec2( 1.0, 0.0));
    float b = terrain(p + s * vec2( 0.0, 1.0));
    float c = terrain(p + s * vec2(-1.0, 0.0));
    float d = terrain(p + s * vec2( 0.0,-1.0));
    
    vec3 dx = vec3(0.05, 0.0, a - c);
    vec3 dy = vec3(0.0, 0.05, b - d);
    return normalize(cross(dx, dy));
}

// Function 512
vec3 getSurfaceNormal(sampler2D tex, vec2 coord)
{
	// Get the local difference in height about the coordinate given.
	vec2 localDiff = getLocalDiff(tex, coord);
	
	// Remember that the surface normal is a negative reciprocal of
	// the surface tangent (which is what the local difference really is).
	// This step does half that job, negating the local difference.
	localDiff *= -1.0;
	
	// Remember that this is to be stored in a pixel, so we have to
	// fit it to the range [0..1].
	localDiff = (localDiff/2.0)+.5;
	
	// In order to reciprocate the local difference in height--the difference
	// in essentially the Z direction of the material--we consider the localDiff
	// to be the horizontal terms of the normal vector. This leaves one thing
	// left to do.
	// We have to scale the Z term based on the magnitude of the height difference.
	// To do this we consider the normal vector to be the hypotenuse of a triangle,
	// with unit length 1. One side of the triangle is constrained to the XY plane,
	// and is the local height difference. This leaves the Z term easy to solve with
	// the pytheagorean theorem.
	float localDiffMag = length(localDiff);
	float z = sqrt(1.0-pow(localDiffMag, 2.0));
	
	return vec3(localDiff, z);
}

// Function 513
vec3 normal(vec3 p)
{
	float c = scene(p);
	vec3 delta;
	vec2 h = vec2(0.01, 0.0);
	delta.x = scene(p + h.xyy) - c;
	delta.y = scene(p + h.yxy) - c;
	delta.z = scene(p + h.yyx) - c;
	return normalize(delta);
}

// Function 514
vec3 calcNormal(vec3 p) {
 
    vec2 eps = vec2(.001,0.);
    vec3   n = vec3(dstScene(p + eps.xyy).dst - dstScene(p - eps.xyy).dst,
                    dstScene(p + eps.yxy).dst - dstScene(p - eps.yxy).dst,
                    dstScene(p + eps.yyx).dst - dstScene(p - eps.yyx).dst);
    return normalize(n);
    
}

// Function 515
void orthonormalBasis(const vec3 n, out vec3 b1, out vec3 b2) {
  float s = n.z >= 0.0 ? 1.0 : -1.0;
  float a = -1.0 / (s + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
  b2 = vec3(b, s + n.y * n.y * a, -n.y);
}

// Function 516
void calc_binormals(vec3 normal,
                    out vec3 tangent,
                    out vec3 binormal)
{
    if (abs(normal.x) > abs(normal.y))
    {
        tangent = normalize(vec3(-normal.z, 0., normal.x));
    }
    else
    {
        tangent = normalize(vec3(0., normal.z, -normal.y));
    }
    
    binormal = cross(normal, tangent);
}

// Function 517
t normalize(vec3

float d(vec3 p)
{
    return 9. - abs(p.y) - texture(iChannel0, p.xz*.01).x * 7.;
}

// Function 518
vec3 textureNormal(vec2 uv) {
    uv = fract(uv) * 3.0 - 1.5;    
        
    vec3 ret;
    ret.xy = sqrt(uv * uv) * sign(uv);
    ret.z = sqrt(abs(1.0 - dot(ret.xy,ret.xy)));
    ret = ret * 0.5 + 0.5;    
    return mix(vec3(0.5,0.5,1.0), ret, smoothstep(1.0,0.98,dot(uv,uv)));
}

// Function 519
vec3 raymarch_main_normal(vec3 _p, float eps, float t)
{
    vec3 n;
    n.y = raymarch_main_scene_normals(_p, t).x;
    n.x = raymarch_main_scene_normals(_p+vec3(eps, 0., 0.), t).x-n.y;
    n.z = raymarch_main_scene_normals(_p+vec3(0., 0., eps), t).x-n.y;
    n.y = raymarch_main_scene_normals(_p+vec3(0., eps, 0.), t).x-n.y;
    return normalize(n);
}

// Function 520
vec3 superellipsoidNormal(vec3 p, Superellipsoid se)
{
    vec3 e = vec3(vec2(1.0) / se.Exponent.xy, se.Exponent.x / se.Exponent.y);
    vec3 g = 2.0 * e;
    vec3 invr = vec3(1.0) / se.Radius;

    vec3 A = p * invr;
    vec3 B = pow(A * A, e.xxy);
    vec3 C = B / A;

    float E = B.x + B.y;
    float F = pow(E, e.z);
    float G = e.z * (F / E);

    vec3 n = g.xxy * C * invr;
    n.xy *= G;

    n = normalize(n);
    return(n);
}

// Function 521
vec3 getNormal(vec3 pos, float e)
{  
    vec2 q = vec2(0, e);
    return normalize(vec3(map(pos + q.yxx, true).x - map(pos - q.yxx, true).x,
                          map(pos + q.xyx, true).x - map(pos - q.xyx, true).x,
                          map(pos + q.xxy, true).x - map(pos - q.xxy, true).x));
}

// Function 522
vec3 normal(vec3 pos) {
    vec3 e = vec3(0.00001, 0., 0.);
    vec3 nor = normalize( vec3(map(pos+e.xyy).x - map(pos-e.xyy).x,
                               map(pos+e.yxy).x - map(pos-e.yxy).x,
                               map(pos+e.yyx).x - map(pos-e.yyx).x));
    return nor;
}

// Function 523
vec3 Normal( vec3 pos )
{
	const vec2 delta = vec2(0,.01);
	vec3 grad;
	grad.x = DistanceField( pos+delta.yxx )-DistanceField( pos-delta.yxx );
	grad.y = DistanceField( pos+delta.xyx )-DistanceField( pos-delta.xyx );
	grad.z = DistanceField( pos+delta.xxy )-DistanceField( pos-delta.xxy );
	return normalize(grad);
}

// Function 524
vec3 normal(vec3 p, float d){//from dr2 
  vec2 e=vec2(d,-d);vec4 v=vec4(DE(p+e.xxx),DE(p+e.xyy),DE(p+e.yxy),DE(p+e.yyx)); 
  return normalize(2.*v.yzw+vec3(v.x-v.y-v.z-v.w)); 
}

// Function 525
vec3 calcNormal( in vec3 pos) {
	vec2 e = vec2(1.0, -1.0) * 0.5773 * 0.0005;
	return normalize(e.xyy * map(pos + e.xyy).x +
		e.yyx * map(pos + e.yyx).x +
		e.yxy * map(pos + e.yxy).x +
		e.xxx * map(pos + e.xxx).x);
}

// Function 526
vec3 CalcNormal( in vec3 pos, in float mult)
{
    vec2 eps = vec2(EPSILON,0.0);
	return normalize( 
		vec3( Map(pos+eps.xyy,mult).x - Map(pos-eps.xyy,mult).x, 
			  Map(pos+eps.yxy,mult).x - Map(pos-eps.yxy,mult).x, 
			  Map(pos+eps.yyx,mult).x - Map(pos-eps.yyx,mult).x) 
	);
}

// Function 527
vec3 SceneNormal(in vec3 pos, in float depth)
{
	vec2 eps = vec2(0.001 * depth, 0.0);
    return normalize(vec3(Scene(pos + eps.xyy).x - Scene(pos - eps.xyy).x,
                          Scene(pos + eps.yxy).x - Scene(pos - eps.yxy).x,
                          Scene(pos + eps.yyx).x - Scene(pos - eps.yyx).x));
}

// Function 528
vec2 normalize_fragcoord(vec2 frag_coord) {
    return ((frag_coord/iResolution.x) - 0.5 * vec2(1.0, iResolution.y / iResolution.x)) * SCALE;
}

// Function 529
vec3 getNormal(vec3 p) {
    int id;
    float dist = getDistMinimal(p, id);
    vec2 e = vec2(MIN_DIST, 0.);
    vec3 n = dist - vec3(
        getDistMinimal(p - e.xyy, id),
        getDistMinimal(p - e.yxy, id),
        getDistMinimal(p - e.yyx, id));
    return normalize(n);
}

// Function 530
vec3 getNormal(vec3 p, float t){
    // method to prevent compiler inlining map
    float h = t * MIN_DIST;
    #define ZERO (min(iFrame,0))
    vec3 n = vec3(0.0);
    for(int i=ZERO; i<4; i++) {
        vec3 e = 0.5773*(2.*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.);
        n += e*map(p+e*h).x;
    }
    return normalize(n);
    //original tetrahedron normal
    //float e = t*MIN_DIST;
    //vec2 h = vec2(1.,-1.)*.5773;
    //return normalize( h.xyy*map( p + h.xyy*e).x + 
	//				  h.yyx*map( p + h.yyx*e).x + 
	//				  h.yxy*map( p + h.yxy*e).x + 
	// 				  h.xxx*map( p + h.xxx*e).x );
}

// Function 531
vec3 drawNormals( in vec3 col, in vec2 p, in float e, in vec2 path[kNumPoints], in vec2 norm[kNumPoints], in vec3 pathColor, in float normLength )
{
    float d = 1000.0;
    for( int i=0; i<kNumPoints; i++ )
    {
        vec2 b = path[i];
        vec2 n = normLength*normalize(norm[i]);

        d = min( d, sdSegmentSq(p,b,b+n) );
    }
    d = sqrt(d);

    return mix( col, pathColor, 1.0-smoothstep(0.0,2.0*e,d) );
}

// Function 532
vec3 getNormal(vec3 point, vec3 sphere, float s){
    float epsilon = 0.00004; // How far in the plane to grab height from to recalc normals
    // Convert the point into model coordinates.
    vec3 p = point - sphere;
    // Standard mapping of a point on a sphere to a UV
    float u = (.5 + atan(p.z, p.x)
               / (2. * 3.14159)) / s / .5 
        	+ iTime * .1;
    float v = (.5 - asin(p.y) / 3.14159) / s / .5;
    
    // Get the height af (u,v), (u, v + epsilon), and (u + epsilon, v)
    // So that we can construct two vectors to cross product
    float d1 = texture( iChannel0,
                           vec2(u,v)).r;
    float d2 = texture( iChannel0,
                           vec2(u,v + epsilon)).r;
    float d3 = texture( iChannel0,
                           vec2(u + epsilon,v)).r;
    
    // Construct points we can generate the normal from.
    vec3 p1 = vec3(u, v, d1 * 0.1);
    vec3 p2 = vec3(u, v + epsilon, d2 * 0.1);
    vec3 p3 = vec3(u + epsilon, v, d3 * 0.1);
    
    // The normal for a un-displaced sphere
    vec3 sphereNorm = p;
    // A vector between a standard "up" vector and the normal for the displacement map
    vec3 planeNorm = vec3(0., 0., 1.) - normalize(cross((p3-p1), (p2 - p1)));
    
    return normalize(planeNorm + sphereNorm); // Return a corrected normal
}

// Function 533
vec4 sampleNormalMap(vec3 N) 
{
    float u = 1.-(atan(N.x, N.z)+PI)/(2.*PI);
	float v = (acos(N.y)/PI);	// 1.- becouse the coordinates origin is in the bottom-left, but I backed from top-left
    return texture(iChannel0, vec2(u, v));   
}

// Function 534
vec3 combineNormals2(vec3 n0, vec3 n1) {
    n0 = n0 * 2.0 - 1.0;
    n1 = n1 * 2.0 - 1.0;    
	n0  = vec3(n0.xy * n1.z + n1.xy * n0.z, n0.z * n1.z);    
    return normalize(n0) * 0.5 + 0.5;
}

// Function 535
vec3 calcNormal(vec3 p)
{
  float e = 0.01;
  vec3 normal = vec3(sceneDf(vec3(p.x+e,p.y,p.z)) - sceneDf(vec3(p.x-e,p.y,p.z)),
                     sceneDf(vec3(p.x,p.y+e,p.z)) - sceneDf(vec3(p.x,p.y-e,p.z)),
                     sceneDf(vec3(p.x,p.y,p.z+e)) - sceneDf(vec3(p.x,p.y,p.z-e)));
  return normalize(normal);
}

// Function 536
vec3 calcNormal( vec3 pos )
{
    // computes planet in sector
    planet p;
    vec3 sector = floor(pos);
    GetPlanet(sector, p);
    
    // return vector 
    return normalize(pos - (sector + p.center));
}

// Function 537
float GetNormalizedDepth(float x)
{
	return (FAR-x)/(FAR-NEAR);   
}

// Function 538
vec3 BlendNormal(vec3 normal){
	vec3 blending = abs(normal);
	blending = normalize(max(blending, 0.00001));
	blending /= vec3(blending.x + blending.y + blending.z);
	return blending;
}

// Function 539
vec3 normalize_color(vec3 raw) {
    return 2.0 / (exp(-EXPOSURE * raw) + 1.0) - 1.0;
}

// Function 540
vec3 calcNormal( in vec3 pos )
{
    vec3 eps = vec3(0.002,0.0,0.0);

	return normalize( vec3(
           map(pos+eps.xyy).x - map(pos-eps.xyy).x,
           map(pos+eps.yxy).x - map(pos-eps.yxy).x,
           map(pos+eps.yyx).x - map(pos-eps.yyx).x ) );
}

// Function 541
vec3 calcNormal( in vec3 pos )
{
	vec3 eps = vec3( 0.1, 0., 0. );
	vec3 nor = vec3(
	    map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	    map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	    map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
}

// Function 542
vec3 getNormal(vec3 p){
    vec2 e = vec2(0.0035, -0.0035); 
    return normalize(
        e.xyy * map(p + e.xyy) + 
        e.yyx * map(p + e.yyx) + 
        e.yxy * map(p + e.yxy) + 
        e.xxx * map(p + e.xxx));
}

// Function 543
vec3 normal(in vec3 p, in vec3 rd)
{
    vec2 e = vec2(-1., 1.)*2e-5;
	return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
					 e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy));   
}

// Function 544
vec3 textureNormal(vec2 uv) {
    vec3 normal = texture( iChannel1, 100.0 * uv ).rgb;
    normal.xy = 2.0 * normal.xy - 1.0;
    normal.z = sqrt(iMouse.x / iResolution.x);
    return normalize( normal );
}

// Function 545
vec3 getNormal(vec2 p)
{
	vec3 j = vec3(1.0/iChannelResolution[0].xy, 0.0);
	vec3 nor  	= vec3(0.0,		textureLod(iChannel0,fract(p), 0.0).w, 0.0);
	vec3 v2		= nor-vec3(j.x,	textureLod(iChannel0,fract(p+j.xz), 0.0).w, 0.0);
	vec3 v3		= nor-vec3(0.0,	textureLod(iChannel0,fract(p-j.zy), 0.0).w, -j.y);
	nor = cross(v2, v3);
	return normalize(nor);
}

// Function 546
vec3 calcNormal( in vec3 pos, in vec3 v[16] )
{
    return nCapsule( pos, v[objA], v[objB], rad );
}

// Function 547
vec3 GetNormal(in vec3 point) 
{
    IntersectionData d0 = CheckSceneForIntersection(point);
    IntersectionData dX = CheckSceneForIntersection(point - vec3(EPSILON, 0.0, 0.0));
    IntersectionData dY = CheckSceneForIntersection(point - vec3(0.0, EPSILON, 0.0));
    IntersectionData dZ = CheckSceneForIntersection(point - vec3(0.0, 0.0, EPSILON));
    return normalize(vec3(dX.mT - d0.mT, dY.mT - d0.mT, dZ.mT - d0.mT));
}

// Function 548
vec3 calcNormal(in vec3 pos, float eps){
    vec2 e = vec2(eps, 0.f);
    return normalize(vec3(map(pos + e.xyy).x - map(pos - e.xyy).x,
                          map(pos + e.yxy).x - map(pos - e.yxy).x,
                          map(pos + e.yyx).x - map(pos - e.yyx).x));
}

// Function 549
vec3 heightmapNormal(vec2 uv) {
	float xdiff = heightmap(uv) - heightmap(uv+epsi.xy);
	float ydiff = heightmap(uv) - heightmap(uv+epsi.yx);
	return normalize(cross(vec3(epsi.yx, -xdiff), vec3(epsi.xy, -ydiff)));
}

// Function 550
vec3 getNormal(in vec3 p) {
	
	const float eps = 0.001;
	return normalize(vec3(
		map(vec3(p.x+eps,p.y,p.z))-map(vec3(p.x-eps,p.y,p.z)),
		map(vec3(p.x,p.y+eps,p.z))-map(vec3(p.x,p.y-eps,p.z)),
		map(vec3(p.x,p.y,p.z+eps))-map(vec3(p.x,p.y,p.z-eps))
	));

}

// Function 551
vec3 normalFBM(in vec2 uv)
{
    
    float neighbours = .0001;
    
    vec3 neighBX1 = vec3(uv.x+neighbours, uv.y, fbm(uv+vec2(neighbours, 0.0)));
    vec3 neighBX2 = vec3(uv.x-neighbours, uv.y, fbm(uv-vec2(neighbours, 0.0)));
    
    vec3 neighBY1 = vec3(uv.x, uv.y+neighbours, fbm(uv+vec2(0.0, neighbours)));
    vec3 neighBY2 = vec3(uv.x, uv.y-neighbours, fbm(uv-vec2(0.0, neighbours)));
    
    vec3 gX = normalize(neighBX1-neighBX2);
    vec3 gY = normalize(neighBY1-neighBY2);
    vec3 n = cross ( gX, gY );
                         
    return n;// * vec3(1.0,1.0,0.5) + vec3(0.0,0.0,0.5);
}

// Function 552
vec3 getNormalFace( in vec3 pos )
{
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
    return normalize( e.xyy*faceFunc( pos + e.xyy ) + 
					  e.yyx*faceFunc( pos + e.yyx ) + 
					  e.yxy*faceFunc( pos + e.yxy ) + 
					  e.xxx*faceFunc( pos + e.xxx ) ) ;
}

// Function 553
vec3 calNormal(vec3 p)
{
    float h = 0.000001;
    vec2 k = vec2(1.0,-1.0);
    
    return normalize(k.xxx * map(p + k.xxx * h).x 
                   + k.xyy * map(p + k.xyy * h).x
                   + k.yxy * map(p + k.yxy * h).x
                   + k.yyx * map(p + k.yyx * h).x);
}

// Function 554
vec3 calcNormal(vec3 p, float t) {
	float d = .01 * t * .33;
	vec2 e = vec2(1, -1) * .5773 * d;
	return normalize(e.xyy * map(p + e.xyy).d + e.yyx * map(p + e.yyx).d + e.yxy * map(p + e.yxy).d + e.xxx * map(p + e.xxx).d);
}

// Function 555
vec3 mapRMNormal(vec3 pt, float e) {
    vec3 normal;
    normal.y = mapRMDetailed(pt).x;    
    normal.x = mapRMDetailed(vec3(pt.x+e,pt.y,pt.z)).x - normal.y;
    normal.z = mapRMDetailed(vec3(pt.x,pt.y,pt.z+e)).x - normal.y;
    normal.y = e;
    return normalize(normal);
}

// Function 556
float parametric_normal_iteration(float t, vec2 uv){
	vec2 uv_to_p=parametric(t)-uv;
	vec2 tang=parametric_diff(t);

	float l_tang=dot(tang,tang);
	return t-factor*dot(tang,uv_to_p)/l_tang;
}

// Function 557
vec3 calcWaterNormal(vec3 p, float t) {
 
    //vec2 eps = vec2(EPSILON * t, 0.);
    //vec3   n = vec3(getHeight(p.xz - eps.xy) - getHeight(p.xz + eps.xy),
    //                2. * eps.x,
    //                getHeight(p.xz - eps.yx) - getHeight(p.xz + eps.yx));
    //return normalize(n);
    
    float eps = EPSILON * t;
    float h   = getHeight(p.xz);
    vec3  n   = vec3(
        getHeight(p.xz + vec2(eps,0.)) - h,
        eps,
        getHeight(p.xz + vec2(0.,eps)) - h
    );
    return normalize(n);
    
}

// Function 558
vec3 getNormal(in vec3 p) {
	const vec2 e = vec2(.001, 0);
	return normalize(
    vec3(map(p + e.xyy) - map(p - e.xyy), 
    map(p + e.yxy) - map(p - e.yxy),	
    map(p + e.yyx) - map(p - e.yyx)));
}

// Function 559
vec3 calcNormal(in vec3 position)
{
    vec3 eps = vec3(0.0001,0.0,0.0);
    vec4 dummyOrbitTrap;

    return normalize( 
        vec3(
        sceneDistanceFunction(position+eps.xyy, dummyOrbitTrap) - sceneDistanceFunction(position-eps.xyy, dummyOrbitTrap),
        sceneDistanceFunction(position+eps.yxy, dummyOrbitTrap) - sceneDistanceFunction(position-eps.yxy, dummyOrbitTrap),
        sceneDistanceFunction(position+eps.yyx, dummyOrbitTrap) - sceneDistanceFunction(position-eps.yyx, dummyOrbitTrap))
    	);
}

// Function 560
vec3 getNormal(in vec3 p)
{
    vec2 eps = vec2(0.0001, 0.0);
    return normalize(vec3(map(p + eps.xyy).y - map(p - eps.xyy).y,
                          map(p + eps.yxy).y - map(p - eps.yxy).y,
                          map(p + eps.yyx).y - map(p - eps.yyx).y));
}

// Function 561
vec3 calcNormal( in vec3 p, float h)
{
    #define ZERO (min(int(iTime),0)) // non-constant zero
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(p+e*h).x;
    }
    return normalize(n);
}

// Function 562
vec3 normal(vec3 p) {
    vec2 e = vec2(epsilon_, 0.0);
    return normalize(vec3(
        eval_scene(p+e.xyy).d-eval_scene(p-e.xyy).d,
        eval_scene(p+e.yxy).d-eval_scene(p-e.yxy).d,
        eval_scene(p+e.yyx).d-eval_scene(p-e.yyx).d
    ));
}

// Function 563
vec3 calcNormal(vec3 p, float t) {
    vec2 e = vec2(MIN_DIST*t,0.);
    vec3 n = vec3(dstScene(p+e.xyy)-dstScene(p-e.xyy),
                  dstScene(p+e.yxy)-dstScene(p-e.yxy),
                  dstScene(p+e.yyx)-dstScene(p-e.yyx));
    return normalize(n);
}

// Function 564
vec3 getNormal(in vec3 p) {
	const vec2 e = vec2(0.001, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 565
vec3 normal_o354278(vec3 p) {
	float d = input_o354278(p).x;
    vec2 e = vec2(.001,0);
    vec3 n = d - vec3(
        input_o354278(p-vec3(e.xyy)).x,
        input_o354278(p-vec3(e.yxy)).x,
        input_o354278(p-vec3(e.yyx)).x);
    return normalize(n);
}

// Function 566
vec2 lunarSurfaceNormal( in vec2 uv )
{	return normalize( vec2(distLunarSurface(vec2(uv.x+EPSILON,uv.y)),
						   distLunarSurface(vec2(uv.x,uv.y+EPSILON)))-
                      vec2(distLunarSurface(vec2(uv.x-EPSILON,uv.y)),
						   distLunarSurface(vec2(uv.x,uv.y-EPSILON))));
}

// Function 567
vec3 calcNormal( in vec3 pos ){
	vec3 eps = vec3( 0.001, 0.0, 0.0 );
	vec3 nor = vec3(
	    map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	    map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	    map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
}

// Function 568
vec3 normal(vec3 p)
{
	vec3 o = vec3(0.01, 0.0, 0.0);
    return normalize(vec3(map(p+o.xyy) - map(p-o.xyy),
                          map(p+o.yxy) - map(p-o.yxy),
                          map(p+o.yyx) - map(p-o.yyx)));
}

// Function 569
vec3 normalUnpack(vec2 enc)
{
    vec3 n;
    n.xy = enc*2.0-1.0;
    n.z = sqrt(1.0-dot(n.xy, n.xy));
    return n;
}

// Function 570
vec2 GetNormalMap(in sampler2D s, in vec2 resolution, in vec2 uv)
{
	vec3 eps=vec3(1.0/resolution,0.0);
	vec2 norm = vec2(length(texture(s,uv+eps.xz)) - length(texture(s,uv-eps.xz)),
					 length(texture(s,uv+eps.zy)) - length(texture(s,uv-eps.zy)));
	
	return norm;
}

// Function 571
vec3 quadricnormal(Surface surface, vec3 p) {
  p -= surface.p;
  vec3 A = surface.params.xyz;
  return A*p;
}

// Function 572
vec3 Normal(vec3 p){vec2 e=vec2(.01,0)
    ;return normalize(vec3(
 df(p+e.xyy).w-df(p-e.xyy).w
,df(p+e.yxy).w-df(p-e.yxy).w
,df(p+e.yyx).w-df(p-e.yyx).w));}

// Function 573
vec4 segment3d_normal(in vec3 ro, in vec3 rd, in vec3 cc, in vec3 ca, float cr, float ch){
    ca = normalize(ca);
    vec3 oc = ro - cc;
    ch *= .5;

    float card = dot(ca, rd);
    float caoc = dot(ca, oc);

    float a = 1. - card * card;
    float b = dot(oc, rd) - caoc * card;
    float c = dot(oc, oc) - caoc * caoc - cr*cr;
    float h = b * b - a * c;
    if(h < .0)
        return vec4(-1.0);
    float t = (-b-sqrt(h))/a;

    float y = caoc + t * card;  // The ray equation!

    // body
    if(abs(y) < ch)
        return vec4(t, normalize(oc + t * rd - ca * y));

    // caps
    float sy = sign(y);
    oc = ro - (cc + sy * ca * ch);
    b = dot(rd, oc);
    c = dot(oc, oc) - cr * cr;
    h = b * b - c;
    if(h > .0){
        t = -b - sqrt(h);
        return vec4(t, normalize(oc + rd * t));
    }

    return vec4(-1.);
}

// Function 574
vec3 asteroidGetNormal(vec3 p, vec3 id) {
    asteroidTransForm( p, id );
    
    vec3 n;
    n.x = asteroidMapDetailed(vec3(p.x+ASTEROID_EPSILON,p.y,p.z), id);
    n.y = asteroidMapDetailed(vec3(p.x,p.y+ASTEROID_EPSILON,p.z), id);
    n.z = asteroidMapDetailed(vec3(p.x,p.y,p.z+ASTEROID_EPSILON), id);
    n = normalize(n-asteroidMapDetailed(p, id));
    
    asteroidUnTransForm( n, id );
    return n;
}

// Function 575
vec3 normalFunction(vec3 p)
{
	const float eps = 0.01;
	float m;
    vec3 n = vec3( (distf(vec3(p.x-eps,p.y,p.z),m) - distf(vec3(p.x+eps,p.y,p.z),m)),
                   (distf(vec3(p.x,p.y-eps,p.z),m) - distf(vec3(p.x,p.y+eps,p.z),m)),
                   (distf(vec3(p.x,p.y,p.z-eps),m) - distf(vec3(p.x,p.y,p.z+eps),m))
				 );
    return normalize( n );
}

// Function 576
vec3 normal( vec3 p )
{
    const float h = EPSILON;
    const vec2 k = vec2(1,-1);
    return normalize( k.xyy*map( p + k.xyy*h ) +
                      k.yyx*map( p + k.yyx*h ) +
                      k.yxy*map( p + k.yxy*h ) +
                      k.xxx*map( p + k.xxx*h ) );
}

// Function 577
vec3 normal(vec3 p)
{
	float f0 = cavern(p);    
	float fx = cavern(p+vec3(0.1, 0.0, 0.0));    
	float fy = cavern(p+vec3(0.0, 0.1, 0.0));    
	float fz = cavern(p+vec3(0.0, 0.0, 0.1));
    vec3 norm = vec3(fx-f0, fy-f0, fz-f0);
    norm = normalize(norm);
    return norm;
}

// Function 578
vec3 calcNormal_3606979787(vec3 pos) {
  return calcNormal_3606979787(pos, 0.002);
}

// Function 579
vec3 summedWaveNormal(vec2 p)
{
    float time = iTime;
	vec2 sum = vec2(0.0);
	sum += directionalWaveNormal(p, 0.5, normalize(vec2(1, 1)), 5.0, 1.5, time, 1.0);
	sum += directionalWaveNormal(p, 0.25,normalize(vec2(1.4, 1.0)), 11.0, 2.4, time, 1.5);
	sum += directionalWaveNormal(p, 0.125, normalize(vec2(-0.8, -1.0)), 10.0, 2.0, time, 2.0);
	sum += directionalWaveNormal(p, 0.0625, normalize(vec2(1.3, 1.0)), 15.0, 4.0, time, 2.2);
	sum += directionalWaveNormal(p, 0.03125, normalize(vec2(-1.7, -1.0)), 5.0, 1.8, time, 3.0);
	return normalize(vec3(-sum.x, -sum.y, 1.0));
}

// Function 580
vec3 estimate_normal(vec3 p, vec3 ro) {
  return normalize(vec3(
    sdf_scene(vec3(p.x + RENDER_EPSILON, p.y, p.z), ro)
    - sdf_scene(vec3(p.x - RENDER_EPSILON, p.y, p.z), ro),
    sdf_scene(vec3(p.x, p.y + RENDER_EPSILON, p.z), ro)
    - sdf_scene(vec3(p.x, p.y - RENDER_EPSILON, p.z), ro),
    sdf_scene(vec3(p.x, p.y, p.z + RENDER_EPSILON), ro)
    - sdf_scene(vec3(p.x, p.y, p.z - RENDER_EPSILON), ro)
    ));
}

// Function 581
vec3 getNormal(vec2 uv) {
    return textureLod(iChannel0, uv, 0.).xyz;
}

// Function 582
mat3 cotangent_frame( vec3 N, vec3 p, vec2 uv )
{
    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );

    // solve the linear system
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame 
    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
    return mat3( T * invmax, B * invmax, N );
}

// Function 583
vec3 getNormal(in vec3 p) {
	const vec2 e = vec2(0.002, 0);
	return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),	map(p + e.yyx) - map(p - e.yyx)));
}

// Function 584
vec3 normal(vec3 pos) {
	float e = 0.001;
    vec3 n;
    n.x = (field(pos+vec3(e,0,0)) - field(pos-vec3(e,0,0)))/(2.*e);
    n.y = (field(pos+vec3(0,e,0)) - field(pos-vec3(0,e,0)))/(2.*e);
    n.z = (field(pos+vec3(0,0,e)) - field(pos-vec3(0,0,e)))/(2.*e);
    return n;
}

// Function 585
vec3 hyperNormalizeG(vec3 v) {
    return v / sqrt(hyperDot(v, v));
}

// Function 586
vec3 normalEstimation(vec3 pos, float time){
  vec2 k = vec2(MinDist, 0);
  return normalize(vec3(sdf(pos + k.xyy, time) - sdf(pos - k.xyy, time),
	  					sdf(pos + k.yxy, time) - sdf(pos - k.yxy, time),
  						sdf(pos + k.yyx, time) - sdf(pos - k.yyx, time)));
}

// Function 587
vec3 calcNormalFish( in vec3 pos )
{
#if 0    
    const vec3 eps = vec3(0.08,0.0,0.0);
	float v = sdDolphin(pos).x;
	return normalize( vec3(
           sdDolphin(pos+eps.xyy).x - v,
           sdDolphin(pos+eps.yxy).x - v,
           sdDolphin(pos+eps.yyx).x - v ) );
#else
    #define ZERO (min(iFrame,0))

    // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*sdDolphin(pos+0.08*e).x;
    }
    return normalize(n);
#endif    
}

// Function 588
vec3 normal(vec3 p) {
	const float epsilon = 0.01;
    return normalize( vec3(scene( vec3(p.x + epsilon, p.y, p.z)) - scene( vec3(p.x - epsilon, p.y, p.z)) , scene( vec3(p.x, p.y + epsilon, p.z)) - scene( vec3(p.x, p.y - epsilon, p.z)), scene( vec3(p.x, p.y, p.z + epsilon)) - scene( vec3(p.x, p.y, p.z - epsilon))));
}

// Function 589
vec3 sphere_get_normal( in vec3 pos, in vec4 sph )
{
    return normalize(pos-sph.xyz);
}

// Function 590
vec3 normal_from_bary(vec3 n[3], vec4 w) {
    return normalize(n[0] * (w.w - w.x) + n[1] * (w.w - w.y) + n[2] * (w.w - w.z));
}

// Function 591
vec3 calcNormal( in vec3 pos )
{
	vec2 eps = vec2( 0.0001, 0.0 );
	vec3 nor = vec3( map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	                 map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	                 map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
}

// Function 592
vec3 calcMissileNormal( in vec3 pos, Missile missile )
{    
  return normalize( vec3(MapFlyingMissile(pos+eps.xyy, missile) - MapFlyingMissile(pos-eps.xyy, missile), 0.5*2.0*eps.x, MapFlyingMissile(pos+eps.yyx, missile) - MapFlyingMissile(pos-eps.yyx, missile) ) );
}

// Function 593
vec3 donormal(Surface surface, vec3 p) {
  if (surface.type == Quadric) {
    return quadricnormal(surface,p);
  } else {
    assert(false);
  }
}

// Function 594
vec3 getNormal(vec3 pos, float ds)
{

    float c = map(pos, 0.).x;
    // Use offset samples to compute gradient / normal
    vec2 eps_zero = vec2(ds, 0.0);
    return normalize(vec3(map(pos + eps_zero.xyy, 0.0).x, map(pos + eps_zero.yxy, 0.0).x,
                          map(pos + eps_zero.yyx, 0.0).x) - c);
}

// Function 595
vec3 calcNormal_2_5(vec3 pos) {
  return calcNormal_2_5(pos, 0.002);
}

// Function 596
vec2 normalized_polar(vec2 coord) {
    //Centered UV coordinates accounting for aspect ratio
    vec2 uv = (coord - CENTER) / iResolution.y;
    
    //Convert to polar. Normalize the angle component by
    //dividing by a full circle.
    vec2 polar = rect_to_polar(uv);
    polar.y /= TAU;
    
    return polar;
}

// Function 597
vec3 calcTreeNormal( in vec3 pos )
{    
  return normalize( vec3(MapTree(pos+eps.xyy) - MapTree(pos-eps.xyy), 0.5*2.0*eps.x, MapTree(pos+eps.yyx) - MapTree(pos-eps.yyx) ) );
}

// Function 598
vec3 terrainNormal(vec2 p)
{
    float eps = 1.0/1024.0;
    
	highp vec2 dx = vec2(eps,0.0);
    highp vec2 dy = vec2(0.0,eps);
    
    highp vec2 px = p + dx;
    highp vec2 py = p + dy;
    
    highp float h = terrain(p.xy);
    highp vec3 t = vec3(px.x,terrain(px),px.y) - vec3(p.x,h,p.y);
    highp vec3 b = vec3(py.x,terrain(py),py.y) - vec3(p.x,h,p.y);
    
    t = normalize(t);
    b = normalize(b);
    
    return normalize(cross(b,t));        		
}

// Function 599
vec3 getNormal(vec3 p)
{
	float d = 0.0001;
	
    return normalize(vec3(
        distMap(p + vec3(  d, 0.0, 0.0)) - distMap(p + vec3( -d, 0.0, 0.0)),
        distMap(p + vec3(0.0,   d, 0.0)) - distMap(p + vec3(0.0,  -d, 0.0)),
        distMap(p + vec3(0.0, 0.0,   d)) - distMap(p + vec3(0.0, 0.0,  -d))
    ));
}

// Function 600
vec3 calcNormal(vec3 p)
{
    const vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
			  DE(p + e.xyy).x - DE(p - e.xyy).x,
			  DE(p + e.yxy).x - DE(p - e.yxy).x,
			  DE(p + e.yyx).x - DE(p - e.yyx).x));
}

// Function 601
vec3 normal(vec2 p, float td) {
	vec2 eps=vec2(0.,.001);
    return normalize(vec3(map(p+eps.yx)-map(p-eps.yx),2.*eps.y,map(p+eps.xy)-map(p-eps.xy)));
}

// Function 602
vec3 normal( vec3 x )
{
    vec2 e = vec2( .01, 0 );
    return normalize( vec3( map(x+e.xyy).x - map(x-e.xyy).x,
                            map(x+e.yxy).x - map(x-e.yxy).x,
                            map(x+e.yyx).x - map(x-e.yyx).x ) );
}

// Function 603
Complex H_normalize(Complex h)
{
    return normalize(h);
}

// Function 604
vec3
calc_normal( in vec3 pos, in float t )
{
    vec3 eps = vec3( max(0.02,T_EPS*t),0.0,0.0);
	return normalize( vec3(
           map( pos + eps.xyy ).distance - map( pos - eps.xyy ).distance,
           map( pos + eps.yxy ).distance - map( pos - eps.yxy ).distance,
           map( pos + eps.yyx ).distance - map( pos - eps.yyx ).distance ) );
}

// Function 605
vec3 calcNormal(vec3 pos) {
    vec2 e = vec2(MARCH_PERC, 0.0);
    return normalize(vec3(
        modelGeometry(pos + e.xyy) - modelGeometry(pos - e.xyy),
        modelGeometry(pos + e.yxy) - modelGeometry(pos - e.yxy),
        modelGeometry(pos + e.yyx) - modelGeometry(pos - e.yyx)
    ));
}

// Function 606
vec3 sdfNormal(vec3 p, float epsilon)
{
    vec3 eps = vec3(epsilon, -epsilon, 0.0);
    
	float dX = sdf(p + eps.xzz) - sdf(p + eps.yzz);
	float dY = sdf(p + eps.zxz) - sdf(p + eps.zyz);
	float dZ = sdf(p + eps.zzx) - sdf(p + eps.zzy); 

	return normalize(vec3(dX,dY,dZ));
}

// Function 607
vec3 calcNormal(vec3 p) {
    float xCom = map(p + e.xyy).dist - map(p - e.xyy).dist;
    float yCom = map(p + e.yxy).dist - map(p - e.yxy).dist;
    float zCom = map(p + e.yyx).dist - map(p - e.yyx).dist;

    return normalize(vec3(xCom, yCom, zCom));
//    return normalize(vec3(0.5));

}

// Function 608
vec3 Scene_GetNormal(const in vec3 vPos)
{
    const float fDelta = 0.0001;
    vec2 e = vec2( -1, 1 );
    
    vec3 vNormal = 
        Scene_GetDistance( e.yxx * fDelta + vPos ).fDist * e.yxx + 
        Scene_GetDistance( e.xxy * fDelta + vPos ).fDist * e.xxy + 
        Scene_GetDistance( e.xyx * fDelta + vPos ).fDist * e.xyx + 
        Scene_GetDistance( e.yyy * fDelta + vPos ).fDist * e.yyy;
    
    return normalize( vNormal );
}

// Function 609
vec3 normal(vec3 p) {
    float diff = 0.00001;
    DistanceEstimation d = DistanceEstimation(
        vec3(0.0), //p
        vec3(0.0), //s
        vec3(0.0), //d
        0.0, //l
        vec3(0.0), //n
        emat
    );
    d.p = p - diff * I3;
    float nx = distanceEstimation(d);
    d.p = p + diff * I3;
    float px = distanceEstimation(d);
    d.p = p - diff * J3;
    float ny = distanceEstimation(d);
    d.p = p + diff * J3;
    float py = distanceEstimation(d);
    d.p = p - diff * K3;
    float nz = distanceEstimation(d);
    d.p = p + diff * K3;
    float pz = distanceEstimation(d);
    return normalize(vec3(
        px - nx,
        py - ny,
        pz - nz
    ));
}

// Function 610
vec3 BumpNormal(in vec3 norm, in vec2 uv)
{
    vec2 eps = vec2(0.001, 0.0);
    
    float sampleU = texture(iChannel0, uv - eps.yx).r;
    float sampleD = texture(iChannel0, uv + eps.yx).r;
    float sampleL = texture(iChannel0, uv - eps.xy).r;
    float sampleR = texture(iChannel0, uv + eps.xy).r;

   	vec3 delta = vec3(
        (sampleL * sampleL - sampleR * sampleR), 
        0.0, 
        (sampleU * sampleU - sampleD * sampleD));

    return normalize(norm + (delta * 1.0));
}

// Function 611
vec3 calcNormal(vec3 pos) {
	vec2 eps = vec2(0.001, 0.0);

	vec3 nor = vec3(map(pos + eps.xyy).x - map(pos - eps.xyy).x,
			map(pos + eps.yxy).x - map(pos - eps.yxy).x,
			map(pos + eps.yyx).x - map(pos - eps.yyx).x);
	return normalize(nor);
}

// Function 612
vec3 calcNormal( vec3 p )
{
    vec3 eps = vec3(0.002,0.0,0.0);

	return normalize( vec3(
           map(p+eps.xyy) - map(p-eps.xyy),
           map(p+eps.yxy) - map(p-eps.yxy),
           map(p+eps.yyx) - map(p-eps.yyx) ) );  
}

// Function 613
vec3 normal(vec3 pos) {
    vec2 eps = vec2(0.001, 0.0);
    return normalize(vec3(	map(pos + eps.xyy) - map(pos - eps.xyy),
                    		map(pos + eps.yxy) - map(pos - eps.yxy),
                         	map(pos + eps.yyx) - map(pos - eps.yyx)));
}

// Function 614
vec3 TangentVisualiser(in float Angle) {
  Angle = atan(Angle);
  vec3 rgb = clamp(abs(mod(Angle*6.0*PI*0.05+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );
  rgb = rgb*rgb*(3.0-2.0*rgb);
  return max(mix( vec3(1.0), rgb, 1.0), 2.*vec3
    (
        min(smoothstep(-PI, -PI + D3PI, Angle), smoothstep(-PI + D3PI, -PI, Angle)),
    	min(smoothstep(-PI + D3PI, -PI + D3PI + D3PI, Angle),smoothstep(-PI + D3PI + D3PI, -PI + D3PI, Angle)),
    	min(smoothstep(-PI + D3PI + D3PI, -PI + D3PI + D3PI + D3PI, Angle),smoothstep(-PI + D3PI + D3PI + D3PI, -PI + D3PI + D3PI, Angle))
    ));
}

// Function 615
vec3 decodeNormal( in vec2 v )
{
    v = -1.0 + 2.0*v;
    // Rune Stubbe's version, much faster than original
    vec3 nor = vec3(v, 1.0 - abs(v.x) - abs(v.y));
    float t = max(-nor.z,0.0);
    nor.x += (nor.x>0.0)?-t:t;
    nor.y += (nor.y>0.0)?-t:t;
    return normalize( nor );
}

// Function 616
vec3 getCylinderNormal(vec3 pos, vec3 properties)
{
	// Perform modulation to keep in line with our cylinder distance function.
	pos.xz = mod(pos.xz,TREE_REP.xz);
	pos.xz -= vec2(TREE_REP.xz*.5);
	
	// Since we can assume that the cylinder is vertical,
	// the only coordinates that matter are x and z.
	// This speeds up normal generation quite a bit.
	vec2 normal = normalize(pos.xz-properties.xy);
	return vec3(normal.x, 0.0, normal.y);
}

// Function 617
vec3 scene_normal(vec3 pos, float d)
{
    return normalize(
		vec3(dst(vec3(pos.x + EPS, pos.y, pos.z)),
			 dst(vec3(pos.x, pos.y + EPS, pos.z)),
		 	 dst(vec3(pos.x, pos.y, pos.z + EPS))) - d);
}

// Function 618
vec3 getNormal(in vec3 p) {
    vec3 n=vec3(0.);
    vec3 e =vec3(0.001,0,0);
    for(int i=min(iFrame,0);i<=2;i++){       
        for(float j=-1.;j<=1.;j+=2.) n+= j*e* mapVoxel(p + j* e) ;
        e=e.zxy; //TdHopper trick
    }
    return normalize(n);
}

// Function 619
vec3 escherDetailNormal(vec2 p)
{
    vec2 pp = mod(p,1.0);
    
    float d = 10000000.0;
    vec2 dir;
    
    vec2 dir_temp;
    float d_temp;
    
    for(int i=0; i<det.length()-1; ++i)
    {
        
        for(int j=0; j<detailTiles.length(); ++j)
        {
            dir_temp = PointSegDirection(pp+detailTiles[j], det[i], det[i+1]);
            d_temp = length(dir_temp);
            if(d_temp < d)
            {
                d = d_temp;
                dir = dir_temp;
            }
        }
    }
    
    float alpha = M_PI/4.0;
    vec3 n = vec3(dir/d*cos(alpha), sin(alpha));
    
    return n;
}

// Function 620
vec3 GetNormal(vec3 p) {
	float d = GetDist(p);
    vec2 e = vec2(.01, 0);
    
    vec3 n = d - vec3(
        GetDist(p-e.xyy),
        GetDist(p-e.yxy),
        GetDist(p-e.yyx));
    
    return normalize(n);
}

// Function 621
vec3 normal(in vec2 fragCoord)
{
	float R = abs(luminance(texsample( OFFSET_X,0, fragCoord)));
	float L = abs(luminance(texsample(-OFFSET_X,0, fragCoord)));
	float D = abs(luminance(texsample(0, OFFSET_Y, fragCoord)));
	float U = abs(luminance(texsample(0,-OFFSET_Y, fragCoord)));
				 
	float X = (L-R) * .5;
	float Y = (U-D) * .5;

	return normalize(vec3(X, Y, 1. / DEPTH));
}

// Function 622
vec3 calcNormal(vec3 pos, float eps) {
  const vec3 v1 = vec3( 1.0,-1.0,-1.0);
  const vec3 v2 = vec3(-1.0,-1.0, 1.0);
  const vec3 v3 = vec3(-1.0, 1.0,-1.0);
  const vec3 v4 = vec3( 1.0, 1.0, 1.0);

  return normalize( v1 * map( pos + v1*eps ) +
                    v2 * map( pos + v2*eps ) +
                    v3 * map( pos + v3*eps ) +
                    v4 * map( pos + v4*eps ) );
}

// Function 623
vec3 getSmoothNormal(vec3 pos)
{
	return normalize(vec3( getSmoothDist(pos+vec3(EPSILON,0,0)), 
					getSmoothDist(pos+vec3(0,EPSILON,0)), 
					getSmoothDist(pos+vec3(0,0,EPSILON)))-getSmoothDist(pos));
}

// Function 624
mat3 OrthoNormalMatrixFromZ( vec3 zDir )
{
	if ( abs( zDir.y ) < 0.999f )
	{
		vec3 yAxis = vec3( 0.0f, 1.0f, 0.0f );
		return OrthoNormalMatrixFromZY( zDir, yAxis );
	}
	else
	{
		vec3 xAxis = vec3( 1.0f, 0.0f, 0.0f );
		return OrthoNormalMatrixFromZY( zDir, xAxis );
	}
}

// Function 625
vec3 calculateNormal(vec3 pos, vec3 playerPos) {
    const vec3 e = vec3(EPS, 0.0, 0.0);
	float p = map(pos, junkMatID, playerPos, true, true);
	return normalize(vec3(map(pos + e.xyy, junkMatID, playerPos, true, true) - p,
           				  map(pos + e.yxy, junkMatID, playerPos, true, true) - p,
                          map(pos + e.yyx, junkMatID, playerPos, true, true) - p));
}

// Function 626
vec4 GenerateNormalHeight (sampler2D tex, vec2 uv, vec2 res, float width)
{
    vec2 texelSize = 1. / (res * width);
    vec4 h;
    h[0] = dot(GrayscaleWeights, texture(tex, uv + vec2(texelSize * vec2( 0,-1)) ).rgb);
    h[1] = dot(GrayscaleWeights, texture(tex, uv + vec2(texelSize * vec2(-1, 0)) ).rgb);
    h[2] = dot(GrayscaleWeights, texture(tex, uv + vec2(texelSize * vec2( 1, 0)) ).rgb);
    h[3] = dot(GrayscaleWeights, texture(tex, uv + vec2(texelSize * vec2( 0, 1)) ).rgb);
    vec3 n;
    n.y = h[0] - h[3];
    n.x = h[1] - h[2];
    n.z = .25;
    float height = dot(GrayscaleWeights, texture(tex, uv).rgb);
    return vec4(normalize(n), height);
}

// Function 627
vec3 normal( in vec3 p )
{
	vec3 eps = vec3(0.0001, 0.0, 0.0);
	return normalize( vec3(
		map(p+eps.xyy)-map(p-eps.xyy),
		map(p+eps.yxy)-map(p-eps.yxy),
		map(p+eps.yyx)-map(p-eps.yyx)
	) );
}

// Function 628
vec2 getNormal(in vec2 p, in int id) {
    return normalize(vec2(mapEnvironment(p + vec2(0.001, 0.0), id).x - mapEnvironment(p - vec2(0.001, 0.0), id).x,
                          mapEnvironment(p + vec2(0.0, 0.001), id).x - mapEnvironment(p - vec2(0.0, 0.001), id).x));
}

// Function 629
vec3 getNormal( in vec3 p ){

    vec2 e = vec2(0.5773,-0.5773)*0.001;
    return normalize( e.xyy*map(p+e.xyy ) + e.yyx*map(p+e.yyx ) + e.yxy*map(p+e.yxy ) + e.xxx*map(p+e.xxx ));
}

// Function 630
vec3 getNormal(vec3 pos, float e, bool inside)
{  
    vec2 q = vec2(0, e);
    return (inside?-1.:1.)*normalize(vec3(map(pos + q.yxx).x - map(pos - q.yxx).x,
                          map(pos + q.xyx).x - map(pos - q.xyx).x,
                          map(pos + q.xxy).x - map(pos - q.xxy).x));
}

// Function 631
vec3 get_tangent_point( vec3 p, vec3 c, float r, vec3 up )
{
	TangentView tv = get_tangent_view( p, c, r );
	return tv.tangent_disk_center +
		   tv.tangent_disk_radius * normalize( cross( tv.target_vector, cross( up, tv.target_vector ) ) );
}

// Function 632
vec3 normal_color( vec3 x ) {
    return (x-min(x,1.))/(max(x,255.)-min(x,1.));
}

// Function 633
vec3 calcNormal(in vec3 pos) {
	vec3 eps = vec3( 0.01, 0.0, 0.0 );
	vec3 nor = vec3(
	    d(pos+eps.xyy).x - d(pos-eps.xyy).x,
	    d(pos+eps.yxy).x - d(pos-eps.yxy).x,
	    d(pos+eps.yyx).x - d(pos-eps.yyx).x );
	return normalize(nor);
}

// Function 634
vec4 normalMap(vec2 uv) { return heightToNormal(normalChannel, normalSampling, uv, normalStrength); }

// Function 635
vec3 calculate_normal(in vec3 world_point)
{
    const vec3 small_step = vec3(0.0025, 0.0, 0.0);

    float gradient_x = distance_to_closest_object(world_point + small_step.xyy)
        - distance_to_closest_object(world_point - small_step.xyy);
    float gradient_y = distance_to_closest_object(world_point + small_step.yxy) 
        - distance_to_closest_object(world_point - small_step.yxy);
    float gradient_z = distance_to_closest_object(world_point + small_step.yyx) 
        - distance_to_closest_object(world_point - small_step.yyx);

    vec3 normal = vec3(gradient_x, gradient_y, gradient_z);

    return normalize(normal);
}

// Function 636
vec3 calcNormal(vec3 p) {
  vec3 eps = vec3(.0001,0,0);
  vec3 n = vec3(
    map(p + eps.xyy) - map(p - eps.xyy),
    map(p + eps.yxy) - map(p - eps.yxy),
    map(p + eps.yyx) - map(p - eps.yyx)
  );
  return normalize(n);
}

// Function 637
vec2 texNormalMap(in vec2 uv)
{
    vec2 s = 1.0/heightMapResolution.xy;
    
    float p = texture(heightMap, uv).x;
    float h1 = texture(heightMap, uv + s * vec2(textureOffset,0)).x;
    float v1 = texture(heightMap, uv + s * vec2(0,textureOffset)).x;
       
   	return (p - vec2(h1, v1));
}

// Function 638
vec3 calculateNormal(in vec3 position, in float pixelSize) {
  vec2 e = vec2(1.0, -1.0) * pixelSize * 0.1;
  return normalize(
      e.xyy * map(position + e.xyy) + e.yyx * map(position + e.yyx) +
      e.yxy * map(position + e.yxy) + e.xxx * map(position + e.xxx));
}

// Function 639
vec3 getNormal(vec3 p){
    const float d = eps;
    return normalize(vec3(distanceFunction(p+vec3(d,0.0,0.0))-distanceFunction(p+vec3(-d,0.0,0.0)),
                          distanceFunction(p+vec3(0.0,d,0.0))-distanceFunction(p+vec3(0.0,-d,0.0)),
                          distanceFunction(p+vec3(0.0,0.0,d))-distanceFunction(p+vec3(0.0,0.0,-d))));
}

// Function 640
vec3 SceneNormal( in vec3 pos )
{
	vec3 eps = vec3( 0.001, 0.0, 0.0 );
	vec3 normal = vec3(
	    Scene( pos + eps.xyy ) - Scene( pos - eps.xyy ),
	    Scene( pos + eps.yxy ) - Scene( pos - eps.yxy ),
	    Scene( pos + eps.yyx ) - Scene( pos - eps.yyx ) );
	return normalize( normal );
}

// Function 641
vec3 computeNormal(vec3 pos)
{
    vec3 epsilon = vec3(0.0, 0.001, 0.0);
    return normalize( vec3( sceneMap3D(pos + epsilon.yxx) - sceneMap3D(pos - epsilon.yxx),
                            sceneMap3D(pos + epsilon.xyx) - sceneMap3D(pos - epsilon.xyx),
                            sceneMap3D(pos + epsilon.xxy) - sceneMap3D(pos - epsilon.xxy)));
}

// Function 642
vec3 getNormal(vec3 p){
	vec2 t = vec2(0.001, 0.);
    return -normalize(vec3(
        map(p - t.xyy).x - map(p + t.xyy).x,
        map(p - t.yxy).x - map(p + t.yxy).x,
        map(p - t.yyx).x - map(p + t.yyx).x
    ));
}

// Function 643
vec3 calcNormal( in vec3 pos )
{
    vec3 eps = vec3( 0.001, 0.0, 0.0 );
    vec3 nor = vec3(
                    scene(pos+eps.xyy).x - scene(pos-eps.xyy).x,
                    scene(pos+eps.yxy).x - scene(pos-eps.yxy).x,
                    scene(pos+eps.yyx).x - scene(pos-eps.yyx).x );
    return normalize(nor);
}

// Function 644
vec3 getNormal2(vec3 p, float e)
{
    return normalize( vec3( map(p+vec3(e,0.0,0.0), 0.).x - map(p-vec3(e,0.0,0.0), 0.).x,
                            map(p+vec3(0.0,e,0.0), 0.).x - map(p-vec3(0.0,e,0.0), 0.).x,
                            map(p+vec3(0.0,0.0,e), 0.).x - map(p-vec3(0.0,0.0,e), 0.).x));
}

// Function 645
vec3 reCalcNormalSlow(vec2 uv)
{
    float offsetPixel = 1.0;
    vec3 center = reCalcWorldPosition(uv);
    return calcNormal(center);
}

// Function 646
vec3 SceneNormal(vec3 pos, float depth)
{
    vec2 eps = vec2(0.01 * depth, 0.0);
    return normalize(vec3(Scene(pos + eps.xyy) - Scene(pos - eps.xyy),
                          Scene(pos + eps.yxy) - Scene(pos - eps.yxy),
                          Scene(pos + eps.yyx) - Scene(pos - eps.yyx)));
}

// Function 647
vec3 getNormal(vec3 pos)
{
    vec2 eps = vec2(0.0, EPSILON);
	return normalize(vec3(
        distanceField(pos + eps.yxx) - distanceField(pos - eps.yxx),
        distanceField(pos + eps.xyx) - distanceField(pos - eps.xyx),
        distanceField(pos + eps.xxy) - distanceField(pos - eps.xxy)
    ));
}

// Function 648
vec3 combineNormals1(vec3 n0, vec3 n1) {
    n0 = n0 * 2.0 - 1.0;
    n1 = n1 * 2.0 - 1.0;
    n0 = vec3(n0.xy + n1.xy, n0.z * n1.z);
    return normalize(n0) * 0.5 + 0.5;
}

// Function 649
vec2 lunarSurfaceNormal( in vec2 uv )
{	return normalize( vec2(distLunarSurface(vec2(uv.x+EPSILON,uv.y)),
						   distLunarSurface(vec2(uv.x,uv.y+EPSILON)))-
                      vec2(distLunarSurface(vec2(uv.x-EPSILON,uv.y)),
						   distLunarSurface(vec2(uv.x,uv.y-EPSILON))) );
}

// Function 650
vec3 normal(vec3 p)
{
 	vec3 N = vec3(-8,8,0) * PRE;
 	N = normalize(model(p+N.xyy)*N.xyy+model(p+N.yxy)*N.yxy+model(p+N.yyx)*N.yyx+model(p+N.xxx)*N.xxx);
 	return bump(iChannel2,p/4.0,N,0.01);
}

// Function 651
vec3 normal(in vec3 p, in float ds)
{  
    vec2 e = vec2(-1., 1.)*0.0005*pow(ds,1.);
	return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
					 e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );   
}

// Function 652
vec3 calcNormal(vec3 p)
{
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
	return normalize( e.xyy *map(p + e.xyy) + 
					  e.yyx *map(p + e.yyx) + 
					  e.yxy *map(p + e.yxy) + 
				  	  e.xxx *map(p + e.xxx) );
}

// Function 653
vec3 normal( in vec3 pos ){
    vec2 e = vec2(0.002, -0.002);
    return normalize(
        e.xyy * map(pos + e.xyy) + 
        e.yyx * map(pos + e.yyx) + 
        e.yxy * map(pos + e.yxy) + 
        e.xxx * map(pos + e.xxx));
}

// Function 654
vec3 calcNormal( in vec3 pos, float t ){
    vec2  eps = vec2( 0.002*t, 0.0 );
    return normalize( vec3( terrainH(pos.xz-eps.xy) - terrainH(pos.xz+eps.xy),
                            2.0*eps.x,
                            terrainH(pos.xz-eps.yx) - terrainH(pos.xz+eps.yx) ) );
}

// Function 655
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

// Function 656
vec3 calcNormal(vec3 p, float t) {
	vec2 e = vec2(EPSILON*t,0.);
    vec3 n = vec3(dstScene(p+e.xyy)-dstScene(p-e.xyy),
                  dstScene(p+e.yxy)-dstScene(p-e.yxy),
                  dstScene(p+e.yyx)-dstScene(p-e.yyx));
    return normalize(n);
}

// Function 657
vec3 getNormal2(vec3 p)
{
    const float d = EPS;
    return normalize(vec3(map(p+vec3(d,0.0,0.0))-map(p+vec3(-d,0.0,0.0)),
                          map(p+vec3(0.0,d,0.0))-map(p+vec3(0.0,-d,0.0)),
                          map(p+vec3(0.0,0.0,d))-map(p+vec3(0.0,0.0,-d))));
}

// Function 658
vec3 get_normal(in vec3 p)
{
	vec3 eps = vec3(0.001, 0, 0); 
	float nx = scene(p + eps.xyy).x - scene(p - eps.xyy).x; 
	float ny = scene(p + eps.yxy).x - scene(p - eps.yxy).x; 
	float nz = scene(p + eps.yyx).x - scene(p - eps.yyx).x; 
	return normalize(vec3(nx,ny,nz)); 
}

// Function 659
vec3 calc_normal(vec3 sample_point) {
    const float h = 0.01; // replace by an appropriate value
    const vec2 k = vec2(1,-1);
    
    vec3 normal = normalize(
		k.xyy * sample_world( sample_point + k.xyy*h ) + 
		k.yyx * sample_world( sample_point + k.yyx*h ) + 
		k.yxy * sample_world( sample_point + k.yxy*h ) + 
		k.xxx * sample_world( sample_point + k.xxx*h ) );
    normal = normal.zyx;
    return normal;
}

// Function 660
vec3 GetNormal(vec3 p){
	const vec2 e = vec2(0.001,0.);
    vec3 nDir = vec3(
    	scene(p+e.xyy),
        scene(p+e.yxy),
        scene(p+e.yyx)
    )-scene(p);
    return normalize(nDir);
}

// Function 661
vec3 getNormal(vec3 pos, float derivDist) {
	vec3 surfaceNormal;
	surfaceNormal.x = distanceField(vec3(pos.x + derivDist, pos.y, pos.z)) 
					- distanceField(vec3(pos.x - derivDist, pos.y, pos.z));
	surfaceNormal.y = distanceField(vec3(pos.x, pos.y + derivDist, pos.z)) 
					- distanceField(vec3(pos.x, pos.y - derivDist, pos.z));
	surfaceNormal.z = distanceField(vec3(pos.x, pos.y, pos.z + derivDist)) 
					- distanceField(vec3(pos.x, pos.y, pos.z - derivDist));
	return normalize(0.5 * surfaceNormal / derivDist);
}

// Function 662
vec3 getNormal(vec3 pos, float e, int objnr)
{  
    vec2 q = vec2(0, e);
    vec3 b = vec3(map(pos + q.yxx).x - map(pos - q.yxx).x,
                  map(pos + q.xyx).x - map(pos - q.xyx).x,
                  map(pos + q.xxy).x - map(pos - q.xxy).x);
    
    waterPos = getWaterPos(pos);
    b+= vec3(getBumps(pos + q.yxx, objnr) - getBumps(pos - q.yxx, objnr),
             getBumps(pos + q.xyx, objnr) - getBumps(pos - q.xyx, objnr),
             getBumps(pos + q.xxy, objnr) - getBumps(pos - q.xxy, objnr));
        
        
    
    return normalize(b);
}

// Function 663
vec3 SceneNormal(in vec3 p, in float d)
{
    vec2 eps = vec2(0.001 * d, 0.0);
    return normalize(vec3(Scene(p + eps.xyy).x - Scene(p - eps.xyy).x,
                          Scene(p + eps.yxy).x - Scene(p - eps.yxy).x,
                          Scene(p + eps.yyx).x - Scene(p - eps.yyx).x));
}

// Function 664
vec3 textureNormal(vec2 uv) {
    vec3 normal = texture( iChannel1, 100.0 * uv ).rgb;
    normal.xy = 2.0 * normal.xy - 1.0;
    
    // Adjust n.z scale with mouse to show how flat normals behave
    normal.z = sqrt(iMouse.x / iResolution.x);
    return normalize( normal );
}

// Function 665
vec3 calcNormal( in vec3 pos )
{
	vec3 eps = vec3(0.01,0.0,0.0);
	return normalize( vec3(
		map(pos+eps.xyy) - map(pos-eps.xyy),
		map(pos+eps.yxy) - map(pos-eps.yxy),
		map(pos+eps.yyx) - map(pos-eps.yyx) ) );
}

// Function 666
vec3 calculate_normal(in vec3 position) {
    vec3 grad = vec3(
      world(vec3(position.x + eps, position.y, position.z)) - world(vec3(position.x - eps, position.y, position.z)),
      world(vec3(position.x, position.y + eps, position.z)) - world(vec3(position.x, position.y - eps, position.z)),
      world(vec3(position.x, position.y, position.z + eps)) - world(vec3(position.x, position.y, position.z - eps))
    );
    
    return normalize(grad);
  }

// Function 667
vec3 getNormal(vec3 p) {
	vec2 e = vec2(EPS, 0.0);
	return normalize((vec3(map(p+e.xyy), map(p+e.yxy), map(p+e.yyx)) - map(p)) / e.x);
}

// Function 668
vec3 normal(const in vec3 p){  
    vec2 e = vec2(-1., 1.)*0.005;   
	return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );}

// Function 669
vec3 calc_normal(vec3 p)
{
 
    vec3 epsilon = vec3(0.001, 0., 0.);
    
    vec3 n = vec3(map(1., p + epsilon.xyy).x - map(1., p - epsilon.xyy).x,
                  map(1., p + epsilon.yxy).x - map(1., p - epsilon.yxy).x,
                  map(1., p + epsilon.yyx).x - map(1., p - epsilon.yyx).x);
    
    return normalize(n);
}

// Function 670
vec3 normal(vec3 p)
{
    vec2 e = vec2(0.0001, 0.0);
    float d = map(p);
    vec3 n = d - vec3(
        map(p - e.xyy),
        map(p - e.yxy),
        map(p - e.yyx));
    return normalize(n);
}

// Function 671
vec3 getNormal(vec3 pos)
{
    const float e = EPS;
    const vec3 dx = vec3(e, 0, 0);
    const vec3 dy = vec3(0, e, 0);
    const vec3 dz = vec3(0, 0, e);

    float d = distFunc(pos);

    return normalize(vec3(
        d - distFunc(vec3(pos - dx)),
        d - distFunc(vec3(pos - dy)),
        d - distFunc(vec3(pos - dz))
    ));
}

// Function 672
vec3 lambertNoTangent(in vec3 normal, in vec2 uv)
{
    float theta = 6.283185 * uv.x;
    uv.y = 2.0 * uv.y - 1.0;
    vec3 spherePoint = vec3(sqrt(1.0 - uv.y * uv.y) * vec2(cos(theta), sin(theta)), uv.y);
    return normalize(normal + spherePoint);
}

