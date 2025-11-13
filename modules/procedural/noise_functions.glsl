// Reusable Noise Procedural Functions
// Automatically extracted from procedural-related shaders

// Function 1
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 2
float voronoi(vec2 uv, vec2 distProportions, float distRotation, float animationOffset)
{
    vec2 rootUV = floor(uv);
    float deg = degFromRootUV(rootUV, animationOffset);
    vec2 pointUV = voronoiPointFromRoot(rootUV, deg);
    
    vec2 tempRootUV;	//Used in loop only
    vec2 tempPointUV;	//Used in loop only
    vec2 closestPointUV = pointUV;
    float minDist = 2.0;
    float dist = 2.0;
    for (float x = -1.0; x <= 1.0; x+=1.0)
    {
     	for (float y = -1.0; y <= 1.0; y+=1.0)   
        {
         	tempRootUV = rootUV + vec2(x, y);
            deg = degFromRootUV(tempRootUV, animationOffset);
            tempPointUV = voronoiPointFromRoot(tempRootUV, deg);
            tempPointUV = mix(tempPointUV, tempRootUV + 0.5, animationOffset);
            
            dist = length(rotate(uv - tempPointUV, distRotation) * distProportions);
            if(dist < minDist)
            {
             	closestPointUV = tempPointUV;
               	minDist = dist;
            }
        }
    }
    
    return minDist;
}

// Function 3
float snoise(vec3 v)
{
    return length(texture(iChannel0, v.xy));
}

// Function 4
float noise(vec3 pos) {
//    return fract(1.0/sin(length(pos)*3.5+pos[0]*pos[1]*1.0)*0.5+0.5)*0.5+0.5;
//}

// Function 5
float vnoise(vec3 p, int seed)
{
    vec3 p0 = floor(p);
    vec3 s = p - p0;
    //s = s * s * (3.0 - 2.0 * s);
    s = s * s * s * (s * (s * 6.0 - 15.0) + 10.0);
    
    return mix(
        mix(
        	mix(noise(p0, seed), noise(p0 + vec3(1.0, 0.0, 0.0), seed), s.x),
        	mix(noise(p0 + vec3(0.0, 1.0, 0.0), seed), noise(p0 + vec3(1.0, 1.0, 0.0), seed), s.x),
        	s.y),
        mix(
        	mix(noise(p0 + vec3(0.0, 0.0, 1.0), seed), noise(p0 + vec3(1.0, 0.0, 1.0), seed), s.x),
        	mix(noise(p0 + vec3(0.0, 1.0, 1.0), seed), noise(p0 + vec3(1.0), seed), s.x),
        	s.y),
        s.z);
}

// Function 6
float interleavedGradientNoise(vec2 n) {
    float f = 0.06711056 * n.x + 0.00583715 * n.y;
    return fract(52.9829189 * fract(f));
}

// Function 7
float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
    return -1.0+2.0*mix( mix( rand( i + vec2(0.0,0.0) ), 
                     rand( i + vec2(1.0,0.0) ), u.x),
                mix( rand( i + vec2(0.0,1.0) ), 
                     rand( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 8
float noise2(float y, float t) {
    vec2 fl = vec2(floor(y), floor(t));
    vec2 fr = vec2(fract(y), fract(t));
    float a = mix(hash2(fl + vec2(0.0, 0.0)), hash2(fl + vec2(1.0, 0.0)), fr.x);
    float b = mix(hash2(fl + vec2(0.0, 1.0)), hash2(fl + vec2(1.0, 1.0)), fr.x);
    return mix(a, b, fr.y);
}

// Function 9
float noise(vec3 p) {
     p  = fract( p*0.3183099+.1 );
	p *= 17.0;
    return fract(fract( p.x*p.y*p.z*(p.x+p.y+p.z))*4625.3725+iTime);
}

// Function 10
float SmoothNoise(in vec2 o) 
{
	vec2 p = floor(o);
	vec2 f = fract(o);
		
	float n = p.x + p.y*57.0;

	float a = Hash(n+  0.0);
	float b = Hash(n+  1.0);
	float c = Hash(n+ 57.0);
	float d = Hash(n+ 58.0);
	
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;
	
	vec2 t = 3.0 * f2 - 2.0 * f3;
	
	float u = t.x;
	float v = t.y;

	float res = a + (b-a)*u +(c-a)*v + (a-b+d-c)*u*v;
    
    return res;
}

// Function 11
float FractalNoise(vec3 pos)
{
    return Noise(pos)+Noise(pos*2.0)*0.5+Noise(pos*4.0)*0.25+Noise(pos*8.0)*0.125;
}

// Function 12
vec2 smplxNoise2DDeriv(vec2 x, float m, vec2 g)
{
    vec2 dmdxy = min(dot(x, x) - vec2(0.5), 0.0);
	dmdxy = 8.*x*dmdxy*dmdxy*dmdxy;
	return dmdxy*dot(x, g) + m*g;
}

// Function 13
float sunSurfaceNoise(vec3 spos, float time)
{
    float s = 0.28;
    float detail = 3.0;
    for(int i = 0; i < 4; ++i)
    {
        float warp = noise(spos*8.0 * detail + time);
        float n = noise(vec3(spos.xy * detail / spos.z + vec2(warp, 0.0), time * detail / 10.0 + float(i) * 0.618033989));
        n = pow(n, 5.0-float(i));
        s += n / detail;
        detail *= 1.847;
    }
    return s;
}

// Function 14
float getWaveNoise(float ti, float wA, vec2 uv){
    float wN = hash1(ti)/3. + noise1d( (uv.x+ti)*waveCurve) * (max(0.,wA*1.5-.3));
    return wN;
}

// Function 15
vec3 BitangentNoise3D(vec3 p)
{
	const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
	const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

	// First corner
	vec3 i = floor(p + dot(p, C.yyy));
	vec3 x0 = p - i + dot(i, C.xxx);

	// Other corners
	vec3 g = step(x0.yzx, x0.xyz);
	vec3 l = 1.0 - g;
	vec3 i1 = min(g.xyz, l.zxy);
	vec3 i2 = max(g.xyz, l.zxy);

	// x0 = x0 - 0.0 + 0.0 * C.xxx;
	// x1 = x0 - i1  + 1.0 * C.xxx;
	// x2 = x0 - i2  + 2.0 * C.xxx;
	// x3 = x0 - 1.0 + 3.0 * C.xxx;
	vec3 x1 = x0 - i1 + C.xxx;
	vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
	vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

	i = i + 32768.5;
	uvec2 hash0 = _pcg3d16(uvec3(i));
	uvec2 hash1 = _pcg3d16(uvec3(i + i1));
	uvec2 hash2 = _pcg3d16(uvec3(i + i2));
	uvec2 hash3 = _pcg3d16(uvec3(i + 1.0 ));

	vec3 p00 = _gradient3d(hash0.x); vec3 p01 = _gradient3d(hash0.y);
	vec3 p10 = _gradient3d(hash1.x); vec3 p11 = _gradient3d(hash1.y);
	vec3 p20 = _gradient3d(hash2.x); vec3 p21 = _gradient3d(hash2.y);
	vec3 p30 = _gradient3d(hash3.x); vec3 p31 = _gradient3d(hash3.y);

	// Calculate noise gradients.
	vec4 m = clamp(0.5 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0, 1.0);
	vec4 mt = m * m;
	vec4 m4 = mt * mt;

	mt = mt * m;
	vec4 pdotx = vec4(dot(p00, x0), dot(p10, x1), dot(p20, x2), dot(p30, x3));
	vec4 temp = mt * pdotx;
	vec3 gradient0 = -8.0 * (temp.x * x0 + temp.y * x1 + temp.z * x2 + temp.w * x3);
	gradient0 += m4.x * p00 + m4.y * p10 + m4.z * p20 + m4.w * p30;

	pdotx = vec4(dot(p01, x0), dot(p11, x1), dot(p21, x2), dot(p31, x3));
	temp = mt * pdotx;
	vec3 gradient1 = -8.0 * (temp.x * x0 + temp.y * x1 + temp.z * x2 + temp.w * x3);
	gradient1 += m4.x * p01 + m4.y * p11 + m4.z * p21 + m4.w * p31;

	// The cross products of two gradients is divergence free.
	return cross(gradient0, gradient1) * 3918.76;
}

// Function 16
float noise(vec3 p) {
	 /* 1. find current tetrahedron T and its four vertices */
	 /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
	 /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/
	 
	 /* calculate s and x */
	 vec3 s = floor(p + dot(p, vec3(F3)));
	 vec3 x = p - s + dot(s, vec3(G3));
	 
	 /* calculate i1 and i2 */
	 vec3 e = step(vec3(0.0), x - x.yzx);
	 vec3 i1 = e*(1.0 - e.zxy);
	 vec3 i2 = 1.0 - e.zxy*(1.0 - e);
	 	
	 /* x1, x2, x3 */
	 vec3 x1 = x - i1 + G3;
	 vec3 x2 = x - i2 + 2.0*G3;
	 vec3 x3 = x - 1.0 + 3.0*G3;
	 	
	 /* 2. find four surflets and store them in d */
	 vec4 w, d;
	 
	 /* calculate surflet weights */
	 w.x = dot(x, x);
	 w.y = dot(x1, x1);
	 w.z = dot(x2, x2);
	 w.w = dot(x3, x3);
	 
	 /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
	 w = max(0.6 - w, 0.0);
	 
	 /* calculate surflet components */
	 d.x = dot(random3(s), x);
	 d.y = dot(random3(s + i1), x1);
	 d.z = dot(random3(s + i2), x2);
	 d.w = dot(random3(s + 1.0), x3);
	 
	 /* multiply d by w^4 */
	 w *= w;
	 w *= w;
	 d *= w;
	 
	 /* 3. return the sum of the four surflets */
	 return dot(d, vec4(52.0));
}

// Function 17
float PerlinNoise3D(vec3 p)
{
    float surfletSum = 0.0;
    vec3 pXLYLZL = floor(p);

    for(int dx = 0; dx <= 1; ++dx)
    {
        for(int dy = 0; dy <= 1; ++dy)
        {
            for(int dz = 0; dz <= 1; ++dz)
            {
                surfletSum += surflet3D(p, pXLYLZL + vec3(dx, dy, dz));
            }
        }
    }

    return surfletSum * sin(iTime * 0.5);
}

// Function 18
float bnoise(in vec2 p)
{
    float d = sin(p.x*1.5+sin(p.y*.2))*0.1;
    return d += texture(iChannel0,p.xy*0.01+time*0.001).x*0.04;
}

// Function 19
vec3 noiseGrain( vec2 uv )
{
	return vec3(
		texture( iChannel1, uv * 20.0 + vec2( iTime * 100.678, iTime * 100.317 ) ).r
	) * 0.2;
}

// Function 20
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*157.0;

    return mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
               mix( hash(n+157.0), hash(n+158.0),f.x),f.y);
}

// Function 21
float noise ( vec2 uv) {
float a = rand(uv);
float b = rand (uv+vec2(1,0));
 float c  = rand (uv+vec2(0,1));
 float d  = rand ( uv+vec2(1,1));
vec2 u =smoothstep(0.,1.,fract(uv));
return mix (a,b,u.x)+(c-a)*u.y*(1.-u.x)+(d-b)*u.x*u.y;
}

// Function 22
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel1, (uv+0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 23
vec2 noise(vec2 p)
{
    const float kF = 3.1415927;
    
    vec2 i = floor(p);
	vec2 f = fract(p);
    f = f*f*(3.-2.*f);
    float t = (mod(i.x+i.y,2.)*2.-1.) * iTime;       // checkered rotation ( flownoise )
    return mix(mix(CS( t+ kF*dot(p,g(i+vec2(0,0)))), // Gabor kernel (and overlapping neighbors)
               	   CS(-t+ kF*dot(p,g(i+vec2(1,0)))),f.x),
               mix(CS(-t+ kF*dot(p,g(i+vec2(0,1)))),
               	   CS( t+ kF*dot(p,g(i+vec2(1,1)))),f.x),f.y);
}

// Function 24
vec3 noisetile(vec2 uv){
    // clamp probably not (and shouldn't be) needed but anyway
    return vec3(clamp(lerpy(uv), 0.0, 1.0));
}

// Function 25
float noise( vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.-2.*f);
#if FLOWNOISE
    float t = exp2(level)* .4*iTime;
    mat2  R = rot(t);
#else
    mat2  R = mat2(1,0,0,1);
#endif
    if (mod(i.x+i.y,2.)==0.) R=-R;

    return 2.*mix( mix( dot( hash( i + vec2(0,0) ), (f - vec2(0,0))*R ), 
                        dot( hash( i + vec2(1,0) ),-(f - vec2(1,0))*R ), u.x),
                   mix( dot( hash( i + vec2(0,1) ),-(f - vec2(0,1))*R ), 
                        dot( hash( i + vec2(1,1) ), (f - vec2(1,1))*R ), u.x), u.y);
}

// Function 26
vec3 noised(vec2 pos, vec2 scale, float phase, float seed) 
{
    const float kPI2 = 6.2831853071;
    // value noise with derivatives based on Inigo Quilez
    pos *= scale;
    vec4 i = floor(pos).xyxy + vec2(0.0, 1.0).xxyy;
    vec2 f = pos - i.xy;
    i = mod(i, scale.xyxy) + seed;

    vec4 hash = multiHash2D(i);
    hash = 0.5 * sin(phase + kPI2 * hash) + 0.5;
    float a = hash.x;
    float b = hash.y;
    float c = hash.z;
    float d = hash.w;
    
    vec4 udu = noiseInterpolateDu(f);    
    float abcd = a - b - c + d;
    float value = a + (b - a) * udu.x + (c - a) * udu.y + abcd * udu.x * udu.y;
    vec2 derivative = udu.zw * (udu.yx * abcd + vec2(b, c) - a);
    return vec3(value * 2.0 - 1.0, derivative);
}

// Function 27
float HilbertNoise(uvec2 uv)
{
    // Hilbert curve:
    uint C = 0xB4361E9Cu;// cost lookup
    uint P = 0xEC7A9107u;// pattern lookup
    
    #ifdef ANIMATE_NOISE
	uv += uint(iFrame) * uvec2(2447445397u, 3242174893u);
    #endif
    
    uint c = 0u;// accumulated cost
    uint p = 0u;// current pattern

    const uint N = 4u;// tile size = 2^N
    for(uint i = N; --i < N;)
    {
        uvec2 m = (uv >> i) & 1u;// local uv

        uint n = m.x ^ (m.y << 1u);// linearized local uv

        uint o = (p << 3u) ^ (n << 1u);// offset into lookup tables

        c += ((C >> o) & 3u) << (i << 1u);// accu cost (scaled by layer)

        p = (P >> o) & 3u;// update pattern
    }

    // add white noise at tile scale:
    const uint r  = 2654435761u;// prime[(2^32-1) / phi_1  ]
    const uint r0 = 3242174893u;// prime[(2^32-1) / phi_2  ]
    const uint r1 = 2447445397u;// prime[(2^32-1) / phi_2^2]
    
    uv = uv >> N;    
    uint h = ((uv.x * r0) ^ (uv.y * r1)) * r;
    
    c += h;

    // fibonacci hashing (aka 1d Roberts):
    //  2^32-1 = 4294967295
    return float(c * r) * (1.0 / 4294967295.0);
}

// Function 28
vec4 snoise3Dv4S(vec3 texc)
{
    vec3 x=texc*256.0;
    vec3 p = floor(x);
    vec3 f = fract(x);
    // using iq's improved texture filtering (https://www.shadertoy.com/view/XsfGDn)
    f = f*f*(3.0-2.0*f);
    vec2 uv = ((p.xy+vec2(17.0,7.0)*p.z) + 0.5 + f.xy)/256.0;
    vec4 v1 = texture( randSampler, uv, -1000.0);
    vec4 v2 = texture( randSampler, uv+vec2(17.0,7.0)/256.0, -1000.0);
    return mix( v1, v2, f.z )-vec4(0.50);
}

// Function 29
float Voronoi2(vec2 p){
    
	vec2 g = floor(p), o;
	p -= g;// p = fract(p);
	
	vec2 d = vec2(1); // 1.4, etc.
    
	for(int y = -1; y <= 1; y++){
		for(int x = -1; x <= 1; x++){
            
			o = vec2(x, y);
            o += hash22(g + o) - p;
            
			float h = dot(o, o);
            d.y = max(d.x, min(d.y, h)); 
            d.x = min(d.x, h);            
		}
	}
	
	//return sqrt(d.y) - sqrt(d.x);
    return (d.y - d.x); // etc.
}

// Function 30
float cosNoise(in vec2 pos){
	return 0.5*(sin(pos.x) + sin(pos.y));
}

// Function 31
float perlinNoise(float perlinTheta, float r, float time) {
    float sum = 0.0;
    for (int octave=MIN_OCTAVE; octave<MAX_OCTAVE; ++octave) {
        float sf = pow(2.0, float(octave));
        float sf8 = sf*64.0; // I can't remember where this variable name came from
        
		float new_theta = sf*perlinTheta;
        float new_r = sf*r + time; // Add current time to this to get an animated effect
		
        float new_theta_floor = floor(new_theta);
		float new_r_floor = floor(new_r);
		float fraction_r = new_r - new_r_floor;
		float fraction_theta = new_theta - new_theta_floor;
        
        float t1 = seededRandom( new_theta_floor	+	sf8 *  new_r_floor      );
		float t2 = seededRandom( new_theta_floor	+	sf8 * (new_r_floor+1.0) );
        
        new_theta_floor += 1.0;
        float maxVal = sf*2.0;
        if (new_theta_floor >= maxVal) {
            new_theta_floor -= maxVal; // So that interpolation with angle 0-360° doesn't do weird things with angles > 360°
        }
        
        float t3 = seededRandom( new_theta_floor	+	sf8 *  new_r_floor      );
		float t4 = seededRandom( new_theta_floor	+	sf8 * (new_r_floor+1.0) );
        
		float i1 = cosineInterpolate(t1, t2, fraction_r);
		float i2 = cosineInterpolate(t3, t4, fraction_r);
        
        sum += cosineInterpolate(i1, i2, fraction_theta)/sf;
    }
    return sum;
}

// Function 32
float noise(in vec2 p) 
{
    vec2 p00 = floor(p);
    vec2 p10 = p00 + vec2(1.0, 0.0);
    vec2 p01 = p00 + vec2(0.0, 1.0);
    vec2 p11 = p00 + vec2(1.0, 1.0);
    
    vec2 s = p - p00;
    
    float a = dot(hash(p00), s);
	float b = dot(hash(p10), p - p10);
	float c = dot(hash(p01), p - p01);
	float d = dot(hash(p11), p - p11);

    vec2 q = s*s*s*(s*(s*6.0 - 15.0) + 10.0);

    float c1 = b - a;
    float c2 = c - a;
    float c3 = d - c - b + a;

   	return a + q.x*c1 + q.y*c2 + q.x*q.y*c3;
}

// Function 33
float noise( in vec2 p )
{
	const float K1 = 0.366025404;
	const float K2 = 0.211324865;
	
	vec2 i = floor( p + (p.x+p.y)*K1 );
	
	vec2 a = p - i + (i.x+i.y)*K2;
	vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0);
	vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;
	
	vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	
	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
	
	return dot( n, vec3(70.0) );
}

// Function 34
float nestedNoise(vec2 p) {
  float x = movingNoise(p);
  float y = movingNoise(p + 100.);
  return movingNoise(p + vec2(x, y));
}

// Function 35
vec4 noised( in vec3 x )
{
    vec3 i = floor(x);
    vec3 w = fract(x);
    
#if 1
    // quintic interpolation
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
#else
    // cubic interpolation
    vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
#endif    
    
    
    float a = hash(i+vec3(0.0,0.0,0.0));
    float b = hash(i+vec3(1.0,0.0,0.0));
    float c = hash(i+vec3(0.0,1.0,0.0));
    float d = hash(i+vec3(1.0,1.0,0.0));
    float e = hash(i+vec3(0.0,0.0,1.0));
	float f = hash(i+vec3(1.0,0.0,1.0));
    float g = hash(i+vec3(0.0,1.0,1.0));
    float h = hash(i+vec3(1.0,1.0,1.0));
	
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z, 
                 du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                            k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                            k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) );
}

// Function 36
vec2 Noise21(float x)
{
    float p = floor(x);
    float f = fract(x);
    f = f*f*(3.0-2.0*f);
    return  mix( hash21(p), hash21(p + 1.0), f)-.5;
    
}

// Function 37
vec3 noise3(in vec2 uv)
{
    vec3 f = texture(iChannel0, uv/256.0).xyz;
	f = f*f*(3.0-2.0*f);
    return f;
}

// Function 38
float simplexRot (vec2 u,vec2 p){return simplexRot (u,p,0.);}

// Function 39
float noise(vec2 xy) {
    //xy = mod(xy, 1.0);
    
    //xy *= 0.2;
    //xy *= 200.0;
    if (xy[0] == 0.0) xy[0] += 0.00001;
    if (xy[1] == 0.0) xy[1] += 0.00001;
    
    float t = fract(1.0 / (0.0001 + abs(0.001*sin(xy[0]-123.0*xy[0]*xy[1]))));
    t = fract(1.0/(0.00001+fract(t + 200.0*xy[0] / (0.000001+abs(20.0*xy[1])))));
    return abs(t);
}

// Function 40
float fnoise(vec3 p) {
  // random rotation reduces artifacts
  p = mat3(0.28862355854826727, 0.6997227302779844, 0.6535170557707412,
           0.06997493955670424, 0.6653237235314099, -0.7432683571499161,
           -0.9548821651308448, 0.26025457467376617, 0.14306504491456504)*p;
  return dot(vec4(noise(p), noise(p*2.), noise(p*4.), noise(p*8.)),
             vec4(0.5, 0.25, 0.125, 0.06));
}

// Function 41
float noise3d(vec3 pos, sampler3D t) {//add sampler2D t
    float total = 0.;
    for(float i = 0.; i<noise_iterations; i++){
        //total+=nose(pos*pow(2., i))*pow(0.5, i);
        total+=texture(t, pos*pow(2., i)*0.05).r*pow(0.5, i);
    }
    return total*1.0;
}

// Function 42
vec3 snoiseVec3( vec3 x ){

  float s  = simplex(vec3( x ));
  float s1 = simplex(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ));
  float s2 = simplex(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ));
  vec3 c = vec3( s , s1 , s2 );
  return c;

}

// Function 43
float perlin(vec2 p, float dim) {
	
	/*vec2 pos = floor(p * dim);
	vec2 posx = pos + vec2(1.0, 0.0);
	vec2 posy = pos + vec2(0.0, 1.0);
	vec2 posxy = pos + vec2(1.0);
	
	// For exclusively black/white noise
	/*float c = step(rand(pos, dim), 0.5);
	float cx = step(rand(posx, dim), 0.5);
	float cy = step(rand(posy, dim), 0.5);
	float cxy = step(rand(posxy, dim), 0.5);*/
	
	/*float c = rand(pos, dim);
	float cx = rand(posx, dim);
	float cy = rand(posy, dim);
	float cxy = rand(posxy, dim);
	
	vec2 d = fract(p * dim);
	d = -0.5 * cos(d * M_PI) + 0.5;
	
	float ccx = mix(c, cx, d.x);
	float cycxy = mix(cy, cxy, d.x);
	float center = mix(ccx, cycxy, d.y);
	
	return center * 2.0 - 1.0;*/
	return perlin(p, dim, 0.0);
}

// Function 44
float cnoise(vec3 P){
  vec3 Pi0 = floor(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
  Pi0 = mod(Pi0, 289.0);
  Pi1 = mod(Pi1, 289.0);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 / 7.0;
  vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0.0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 / 7.0;
  vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0.0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
  return 2.2 * n_xyz;
}

// Function 45
float Noise(in vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, -100.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 46
float noise(in vec3 p)
{
	vec3 ip = floor(p), fp = fract(p);
    fp = fp*fp*(3.0-2.0*fp);
	vec2 tap = (ip.xy+vec2(37.0,17.0)*ip.z) + fp.xy;
	vec2 cl = textureLod( iChannel0, (tap + 0.5)/256.0, 0.0 ).yx;
	return mix(cl.x, cl.y, fp.z);
}

// Function 47
float noise(vec2 x)
{
	vec2 p = floor(x);
	vec2 f = fract(x);
	return mix(
		mix(
			hash3(p + vec2(0.0, 0.0)).x,
			hash3(p + vec2(1.0, 0.0)).x,
			smoothstep(0.0, 1.0, f.x)
		),
		mix(
			hash3(p + vec2(0.0, 1.0)).x,
			hash3(p + vec2(1.0, 1.0)).x,
			smoothstep(0.0, 1.0, f.x)
		),
		smoothstep(0.0, 1.0, f.y)
	);
}

// Function 48
float noise( in vec3 x )
{
    #if 1
    
    vec3 i = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (i.xy+vec2(37.0,17.0)*i.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
    
    #else
    
    ivec3 i = ivec3(floor(x));
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	ivec2 uv = i.xy + ivec2(37,17)*i.z;
	vec2 rgA = texelFetch( iChannel0, (uv+ivec2(0,0))&255, 0 ).yx;
    vec2 rgB = texelFetch( iChannel0, (uv+ivec2(1,0))&255, 0 ).yx;
    vec2 rgC = texelFetch( iChannel0, (uv+ivec2(0,1))&255, 0 ).yx;
    vec2 rgD = texelFetch( iChannel0, (uv+ivec2(1,1))&255, 0 ).yx;
    vec2 rg = mix( mix( rgA, rgB, f.x ),
                   mix( rgC, rgD, f.x ), f.y );
    return mix( rg.x, rg.y, f.z );
    
    #endif
}

// Function 49
float noise1(float p) {
    float i = floor(p);
    float f = fract(p);
    float u = f * f * (3.0 - 2.0 * f);
    return 1.0 - 2.0 * mix(hash1(i), hash1(i + 1.0), u);
}

// Function 50
float perlin1d(float t) {
    float i0 = floor(t);
    float i1 = ceil(t);
    
    float g0 = noise1d(i0)*4.0 - 2.0;
    float g1 = noise1d(i1)*4.0 - 2.0;
    
    float u = fract(t);
    float su = smoother(u);
    
    float n0 = g0 * u;
    float n1 = g1 * (u - 1.0);
    
    float n = (1.0 - su) * n0 + su * n1;
    
    return n;
    // return (1.0 - su) * g0 + su * g1;
}

// Function 51
float InterpolationNoise(vec2 p)
{
    vec2 fracp = fract(p);    
    vec2 florp = floor(p);
    
    float v1 = smoothNoise(florp);
    float v2 = smoothNoise(florp+vec2(1.0,0.0));
    float v3 = smoothNoise(florp+vec2(0.0,1.0));
    float v4 = smoothNoise(florp+vec2(1.0,1.0));
    
   	float i1 = COSInterpolation(v1,v2,fracp.x);
    float i2 = COSInterpolation(v3,v4,fracp.x);
    
    return COSInterpolation(i1,i2,fracp.y);
    
}

// Function 52
vec3 cellularNoised(vec2 pos, vec2 scale, float jitter, float phase, float seed) 
{       
    const float kPI2 = 6.2831853071;
    pos *= scale;
    vec2 i = floor(pos);
    vec2 f = pos - i;
    
    const vec3 offset = vec3(-1.0, 0.0, 1.0);
    vec4 cells = mod(i.xyxy + offset.xxzz, scale.xyxy) + seed;
    i = mod(i, scale) + seed;
    vec4 dx0, dy0, dx1, dy1;
    multiHash2D(vec4(cells.xy, vec2(i.x, cells.y)), vec4(cells.zyx, i.y), dx0, dy0);
    multiHash2D(vec4(cells.zwz, i.y), vec4(cells.xw, vec2(i.x, cells.w)), dx1, dy1);
    dx0 = 0.5 * sin(phase + kPI2 * dx0) + 0.5;
    dy0 = 0.5 * sin(phase + kPI2 * dy0) + 0.5;
    dx1 = 0.5 * sin(phase + kPI2 * dx1) + 0.5;
    dy1 = 0.5 * sin(phase + kPI2 * dy1) + 0.5;
    
    dx0 = offset.xyzx + dx0 * jitter - f.xxxx; // -1 0 1 -1
    dy0 = offset.xxxy + dy0 * jitter - f.yyyy; // -1 -1 -1 0
    dx1 = offset.zzxy + dx1 * jitter - f.xxxx; // 1 1 -1 0
    dy1 = offset.zyzz + dy1 * jitter - f.yyyy; // 1 0 1 1
    vec4 d0 = dx0 * dx0 + dy0 * dy0; 
    vec4 d1 = dx1 * dx1 + dy1 * dy1; 
    
    vec2 centerPos = (0.5 * sin(phase + kPI2 *  multiHash2D(i)) + 0.5) * jitter - f; // 0 0
    float dCenter = dot(centerPos, centerPos);
    vec4 d = min(d0, d1);
    vec4 less = step(d1, d0);
    vec4 dx = mix(dx0, dx1, less);
    vec4 dy = mix(dy0, dy1, less);

    vec3 t1 = d.x < d.y ? vec3(d.x, dx.x, dy.x) : vec3(d.y, dx.y, dy.y);
    vec3 t2 = d.z < d.w ? vec3(d.z, dx.z, dy.z) : vec3(d.w, dx.w, dy.w);
    t2 = t2.x < dCenter ? t2 : vec3(dCenter, centerPos);
    vec3 t = t1.x < t2.x ? t1 : t2;
    t.x = sqrt(t.x);
    // normalize: 0.75^2 * 2.0  == 1.125
    return  t * vec3(1.0, -2.0, -2.0) * (1.0 / 1.125);
}

// Function 53
float simplex( in vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

	vec2 i = floor( p + (p.x+p.y)*K1 );
	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = step(a.yx,a.xy);    
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;

    vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );

	vec3 n = h*h*h*h*vec3( dot(a,hashz(i+0.0)), dot(b,hashz(i+o)), dot(c,hashz(i+1.0)));

    return dot( n, vec3(80.0) )*0.5+0.5;
	
}

// Function 54
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);

    return mix(mix(mix( hash(p+vec3(0,0,0)),
                        hash(p+vec3(1,0,0)),f.x),
                   mix( hash(p+vec3(0,1,0)),
                        hash(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(p+vec3(0,0,1)),
                        hash(p+vec3(1,0,1)),f.x),
                   mix( hash(p+vec3(0,1,1)),
                        hash(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 55
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*157.0 + 113.0*p.z;

    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 56
vec3 noise_offset(vec3 p, float f, float s)
{
    p += curlnoise(p * f, 1) * s;
    p += curlnoise(p * f, 1) * s;
    p += curlnoise(p * f, 1) * s;
    return p;
}

// Function 57
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);
	return textureLod( iChannel1, (uv+0.5)/256.0, 0.0 ).x;
}

// Function 58
float achnoise(vec2 x){
    vec2 p = floor(x);
    vec2 fr = fract(x);
    vec2 LB = p;
    vec2 LT = p + vec2(0.0, 1.0);
    vec2 RB = p + vec2(1.0, 0.0);
    vec2 RT = p + vec2(1.0, 1.0);

    float LBo = oct(LB);
    float RBo = oct(RB);
    float LTo = oct(LT);
    float RTo = oct(RT);

    float noise1d1 = mix(LBo, RBo, fr.x);
    float noise1d2 = mix(LTo, RTo, fr.x);

    float noise2d = mix(noise1d1, noise1d2, fr.y);

    return noise2d;
}

// Function 59
vec3 DenoiseREF(vec2 uv, vec2 lUV, vec2 aUV, sampler2D attr, sampler2D light, float radius, vec3 CVP, vec3 CN, vec3 CVN,
            vec2 ires, vec2 hres, vec2 asfov) {
    //RGB denoiser
    vec4 L0=texture(light,lUV*ires);
    vec4 Accum=vec4(Read3(L0.x)*0.2,0.2);
    Accum+=(_DenoiseREF(uv,lUV,aUV,vec2(radius,0.),CVP,CN,CVN,attr,light,ires,hres,asfov)+
            _DenoiseREF(uv,lUV,aUV,vec2(0.,radius),CVP,CN,CVN,attr,light,ires,hres,asfov)+
            _DenoiseREF(uv,lUV,aUV,vec2(radius*0.707),CVP,CN,CVN,attr,light,ires,hres,asfov)+
            _DenoiseREF(uv,lUV,aUV,vec2(radius,-radius)*0.707,CVP,CN,CVN,attr,light,ires,hres,asfov)
            )*0.1;
    //Output
    return Accum.xyz/Accum.w;
}

// Function 60
float noiseDist(vec3 p) {
	p = p / NoiseScale;
	return (fbm(p) - NoiseIsoline) * NoiseScale;
}

// Function 61
float Noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    float res = mix(mix( Hash(n+  0.0), Hash(n+  1.0),f.x),
                    mix( Hash(n+ 57.0), Hash(n+ 58.0),f.x),f.y);
    return res;
}

// Function 62
float noise( in vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix( hash(p+vec3(0,0,0)), 
                        hash(p+vec3(1,0,0)),f.x),
                   mix( hash(p+vec3(0,1,0)), 
                        hash(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(p+vec3(0,0,1)), 
                        hash(p+vec3(1,0,1)),f.x),
                   mix( hash(p+vec3(0,1,1)), 
                        hash(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 63
float noise( in vec2 p )
{
	const float K1 = 0.366025404; // (sqrt(3)-1)/2;
	const float K2 = 0.211324865; // (3-sqrt(3))/6;
	
	vec2 i = floor( p + (p.x+p.y)*K1 );
	
	vec2 a = p - i + (i.x+i.y)*K2;
	vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0);
	vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;
	
	vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	
	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
	
	return dot( n, vec3(70.0) );
}

// Function 64
float noise(vec3 p){//Noise function stolen from Virgil who stole it from Shane who I assume understands this shit, unlike me who is too busy throwing toilet paper at my math teacher's house
  vec3 ip=floor(p),s=vec3(7,157,113);
  p-=ip; vec4 h=vec4(0,s.yz,s.y+s.z)+dot(ip,s);
  p=p*p*(3.-2.*p);
  h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
  h.xy=mix(h.xz,h.yw,p.y);
  return mix(h.x,h.y,p.z);
}

// Function 65
float noise( vec2 point )
{
	vec2 p = floor( point );
	vec2 f = fract( point );
	return mix(
		mix( random( p + vec2( 0.0, 0.0 ) ), random( p + vec2( 1.0, 0.0 ) ), f.x ),
		mix( random( p + vec2( 0.0, 1.0 ) ), random( p + vec2( 1.0, 1.0 ) ), f.x ),
		f.y
	);
}

// Function 66
float bnoise(in vec3 p)
{
    float n = sin(triNoise3d(p*.3,0.0)*11.)*0.6+0.4;
    n += sin(triNoise3d(p*1.,0.05)*40.)*0.1+0.9;
    return (n*n)*0.003;
}

// Function 67
float noise(float n,float s,float res)
{
	float a = fract(sin(((floor((n)/s-0.5)*s)/res)*432.6326)*556.6426);
	float b = fract(sin(((floor((n)/s+0.5)*s)/res)*432.6326)*556.6426);
	return mix(a,b,smoothstep(0.0,1.0,+mod(n/s+0.5,1.0)));
}

// Function 68
float noise(float p)
{
    float i = floor(p);
    float f = fract(p);
    f *= f * (3. - 2.*f);
    
    return mix( hash(i-f), hash(i+f), f );
}

// Function 69
float Mnoise(vec3 U ) {
    return noise(U);                      // base turbulence
  //return -1. + 2.* (1.-abs(noise(U)));  // flame like
  //return -1. + 2.* (abs(noise(U)));     // cloud like
}

// Function 70
float noise(vec2 x) { vec2 i = floor(x); vec2 f = fract(x);	float a = hash(i); float b = hash(i + vec2(1.0, 0.0)); float c = hash(i + vec2(0.0, 1.0)); float d = hash(i + vec2(1.0, 1.0)); vec2 u = f * f * (3.0 - 2.0 * f); return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y; }

// Function 71
float Noise101( float x ) { return fract(sin(x)*5346.1764); }

// Function 72
float Mnoise(in vec2 uv ) {
  //return noise(uv);                      // base turbulence
  //return -1. + 2.* (1.-abs(noise(uv)));  // flame like
    return -1. + 2.* (abs(noise(uv)));     // cloud like
}

// Function 73
float noise(in vec3 coord, in int N)
{
    coord *= float(N);
    
    ivec3 c = ivec3(coord);   // current cell
    vec3 unit = fract(coord); // orientation in current cell
   
    vec3 unit_000 = unit;
    vec3 unit_100 = unit - vec3(1, 0, 0);
    vec3 unit_001 = unit - vec3(0, 0, 1);
    vec3 unit_101 = unit - vec3(1, 0, 1);
    vec3 unit_010 = unit - vec3(0, 1, 0);
    vec3 unit_110 = unit - vec3(1, 1, 0);
    vec3 unit_011 = unit - vec3(0, 1, 1);
    vec3 unit_111 = unit - 1.;
    
    // Hash cell coordinates
    int A = p[(c.x  ) % N] + c.y, AA = p[A % N] + c.z, AB = p[(A+1) % N] + c.z,
        B = p[(c.x+1) % N] + c.y, BA = p[B % N] + c.z, BB = p[(B+1) % N] + c.z;

    float x000 = gradient(p[(AA  ) % N], unit_000);
	float x100 = gradient(p[(BA  ) % N], unit_100);
	float x010 = gradient(p[(AB  ) % N], unit_010);
	float x110 = gradient(p[(BB  ) % N], unit_110);
    float x001 = gradient(p[(AA+1) % N], unit_001);
	float x101 = gradient(p[(BA+1) % N], unit_101);
	float x011 = gradient(p[(AB+1) % N], unit_011);
	float x111 = gradient(p[(BB+1) % N], unit_111);

    // Compute fade curves
    vec3 w = fade(unit);
    
    return mix(mix(mix(x000, x100, w.x),
                   mix(x010, x110, w.x),
                   w.y),
               mix(mix(x001, x101, w.x),
                   mix(x011, x111, w.x),
                   w.y),
               w.z);
}

// Function 74
float noise(vec2 x) {
  vec2 i = floor(x);
  vec2 f = fract(x);

  // Four corners in 2D of a tile
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));

  // Simple 2D lerp using smoothstep envelope between the values.
  // return vec3(mix(mix(a, b, smoothstep(0.0, 1.0, f.x)),
  //			mix(c, d, smoothstep(0.0, 1.0, f.x)),
  //			smoothstep(0.0, 1.0, f.y)));

  // Same code, with the clamps in smoothstep and common subexpressions
  // optimized away.
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Function 75
float noise(vec2 uv)
{
    return texture(iChannel0,uv).r;
}

// Function 76
float noise( vec2 p ) 
{
	vec2 pi = floor( p );
	vec2 pf = fract( p );
	
	
	float n = pi.x + 59.0 * pi.y;
	
	pf = pf * pf * (3.0 - 2.0 * pf);
	
	return mix( 
		mix( hash( n ), hash( n + 1.0 ), pf.x ),
		mix( hash( n + 59.0 ), hash( n + 1.0 + 59.0 ), pf.x ),
		pf.y );
		
		
}

// Function 77
float noise21(in vec2 p){
    p.x = fract(p.x);
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash22( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( hash22( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hash22( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( hash22( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 78
void aurorasWithLightningNoise( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 q = fragCoord.xy / iResolution.xy;
    vec2 p = q - 0.5;
	p.x*=iResolution.x/iResolution.y;
    
    vec3 ro = vec3(0,0,-6.7);
    vec3 rd = normalize(vec3(p,1.3));
    vec2 mo = iMouse.xy / iResolution.xy-.5;
    mo = (mo==vec2(-.5))?mo=vec2(-0.1,0.1):mo;
	mo.x *= iResolution.x/iResolution.y;
    rd.yz *= mm2(mo.y);
    rd.xz *= mm2(mo.x + sin(time*0.05)*0.2);
    
    vec3 col = vec3(0.);
    vec3 brd = rd;
    float fade = smoothstep(0.,0.01,abs(brd.y))*0.1+0.9;
    
    col = bg(rd)*fade;
    
    if (rd.y > 0.){
        vec4 aur = smoothstep(0.,1.5,aurora(ro,rd))*fade;
        col += stars(rd);
        col = col*(1.-aur.a) + aur.rgb;
    }
    else //Reflections
    {
        rd.y = abs(rd.y);
        col = bg(rd)*fade*0.6;
        vec4 aur = smoothstep(0.0,2.5,aurora(ro,rd));
        col += stars(rd)*0.1;
        col = col*(1.-aur.a) + aur.rgb;
        vec3 pos = ro + ((0.5-ro.y)/rd.y)*rd;
        float nz2 = domainWarp(pos.xz*vec2(.5,.7));
        col += mix(vec3(0.2,0.25,0.5)*0.08,vec3(0.3,0.3,0.5)*0.7, nz2*0.4);
    }
    
	fragColor = vec4(col, 1.);
}

// Function 79
float Noise2D( vec2 pos )
{
    vec2 baseCorner = floor( pos );
    
    vec4 gradients_01 = vec4( TwoLatticeGradients2D( vec4( baseCorner                         , baseCorner + Noise2DCornerOffsets_1 ) ) );
    vec4 gradients_23 = vec4( TwoLatticeGradients2D( vec4( baseCorner + Noise2DCornerOffsets_2, baseCorner + Noise2DCornerOffsets_3 ) ) );
    
    vec2 frac = pos - baseCorner;
    vec2 fracSmooth = frac * frac * (3.0 - 2.0 * frac);
    
    vec4 vals = vec4(
        dot( frac                         , gradients_01.xy ),
        dot( frac - Noise2DCornerOffsets_1, gradients_01.zw ),
        dot( frac - Noise2DCornerOffsets_2, gradients_23.xy ),
        dot( frac - Noise2DCornerOffsets_3, gradients_23.zw )
    );
    
    vec2 xvals = mix( vals.xy, vals.zw, fracSmooth.x );
    
    return mix( xvals.x, xvals.y, fracSmooth.y );
}

// Function 80
float blobnoises(vec2 uv, float s)
{
    float h = 0.0;
    const float n = 3.0;
    for(float i = 0.0; i < n; i++)
    {
        vec2 p = vec2(0.0, 1.0 * iTime * (i + 1.0) / n) + 1.0 * uv;
    	h += pow(0.5 + 0.5 * cos(pi * clamp(simplegridnoise(p * (i + 1.0)) * 2.0, 0.0, 1.0)), s);
    }
    
    return h / n;
}

// Function 81
float galaxy_noise(float ttime, vec2 p) {
  float s = 1.0;

  p *= tanh(0.1*length(p));
  float tm = ttime;

  float a = cos(p.x);
  float b = cos(p.y);

  float c = cos(p.x*sqrt(3.5)+tm);
  float d = cos(p.y*sqrt(1.5)+tm);

  return a*b*c*d;
}

// Function 82
float simplex2d(vec2 p){vec2 s = floor(p + (p.x+p.y)*F2),x = p - s - (s.x+s.y)*G2; float e = step(0.0, x.x-x.y); vec2 i1 = vec2(e, 1.0-e),  x1 = x - i1 - G2, x2 = x - 1.0 - 2.0*G2; vec3 w, d; w.x = dot(x, x); w.y = dot(x1, x1); w.z = dot(x2, x2); w = max(0.5 - w, 0.0); d.x = dot(random2(s + 0.0), x); d.y = dot(random2(s +  i1), x1); d.z = dot(random2(s + 1.0), x2); w *= w; w *= w; d *= w; return dot(d, vec3(70.0));}

// Function 83
vec2 noise(vec2 tc){
    return hash2(tc);
}

// Function 84
float snoise(vec2 v) {
//
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
//
    
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
  // First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

  // Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

  // Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
    + i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

  // Gradients: 41 points uniformly over a line, mapped onto a diamond.
  // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

  // Normalise gradients implicitly by scaling m
  // Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

  // Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// Function 85
float RustNoise3D(vec3 p)
{
    float n = 0.0;
    float iter = 1.0;
    float pn = noise(p*0.125);
    pn += noise(p*0.25)*0.5;
    pn += noise(p*0.5)*0.25;
    pn += noise(p*1.0)*0.125;
    for (int i = 0; i < 7; i++)
    {
        //n += (sin(p.y*iter) + cos(p.x*iter)) / iter;
        float wave = saturate(cos(p.y*0.25 + pn) - 0.998);
        wave *= noise(p * 0.125)*1016.0;
        n += wave;
        p.xy += vec2(p.y, -p.x) * nudge;
        p.xy *= normalizer;
        p.xz += vec2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        iter *= 1.4733;
    }
    return n;
}

// Function 86
vec3 voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( iTime + 6.2831*o );
        #endif	
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( iTime + 6.2831*o );
        #endif	
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>0.00001 )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec3( md, mr );
}

// Function 87
float fractalNoise(vec2 v) 
{
    // initialize
    float value = 0.0;
    float amplitude = 0.5;
    // loop
    for (int i = 0; i < 4; i++) 
    {
        value += amplitude * gradientNoise(v);
        // double the frequency
        v *= 2.0;
        // half the amplitude
        amplitude *= 0.5;
    }
    return value;
}

// Function 88
float noise2( vec2 p, vec2 cycle )
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = smoothstep(0.,1.,f);
    return mix( mix( hash( mod(i + vec2(0,0),cycle) ), 
                     hash( mod(i + vec2(1,0),cycle) ), u.x),
                mix( hash( mod(i + vec2(0,1),cycle) ), 
                     hash( mod(i + vec2(1,1),cycle) ), u.x), u.y).x;
}

// Function 89
float perlin(vec3 x)
{
    return (noise(x) + noise(x * 2.0) * 0.5 + noise(x * 12.0) * 0.5 + noise(x * 24.0) * 0.25) * 0.25;
}

// Function 90
float voronoi(vec2 p) {
    vec2 n = floor(p);
    vec2 f = fract(p);
    float md = 5.0;
    vec2 m = vec2(0.0);
    for (int i = -1;i<=1;i++) {
        for (int j = -1;j<=1;j++) {
            vec2 g = vec2(i, j);
            vec2 o = hash2(n+g);
            o = 0.5+0.5*sin(iTime+5.038*o);
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d<md) {
              md = d;
              m = n+g+o;
            }
        }
    }
    return 1.0-md;
}

// Function 91
float valueNoise(float i, float p){ return mix(r11(floor(i)),r11(floor(i) + 1.), ss(fract(i), p,0.6));}

// Function 92
vec2 Noise( in vec2 x )
{
    return mix(Hash2(floor(x)), Hash2(floor(x)+1.0), fract(x));
}

// Function 93
vec4 fbmVoronoi(vec2 pos, vec2 scale, int octaves, float shift, float timeShift, float gain, float lacunarity, float octaveFactor, float jitter, float interpolate, float seed) 
{
    float amplitude = gain;
    float time = timeShift;
    vec2 frequency = scale;
    vec2 offset = vec2(shift, 0.0);
    vec2 p = pos * frequency;
    octaveFactor = 1.0 + octaveFactor * 0.12;
    
    vec2 sinCos = vec2(sin(shift), cos(shift));
    mat2 rotate = mat2(sinCos.y, sinCos.x, sinCos.x, sinCos.y);
    
    float n = 1.0;
    vec4 value = vec4(0.0);
    for (int i = 0; i < octaves; i++) 
    {
        vec3 v = voronoi(p / frequency, frequency, jitter, timeShift, seed);
        v.x = v.x * 2.0 - 1.0;
        n *= v.x;
        value += amplitude * vec4(mix(v.x, n, interpolate), hash3D(v.yz));
        
        p = p * lacunarity + offset * float(1 + i);
        frequency *= lacunarity;
        amplitude = pow(amplitude * gain, octaveFactor);
        time += timeShift;
        offset *= rotate;
    }
    value.x = value.x * 0.5 + 0.5;
    return value;
}

// Function 94
float noise( in vec3 x )
{
    vec3 f = fract(x);
    vec3 p = floor(x);
    f = f * f * (3.0 - 2.0 * f);
    
    p.xz += WIND * iTime;
    vec2 uv = (p.xz + vec2(37.0, 17.0) * p.y) + f.xz;
    vec2 rg = texture(iChannel0, (uv + 0.5)/256.0, 0.0).yx;
    return mix(rg.x, rg.y, f.y);
}

// Function 95
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);

	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z);
	vec2 rg1 = texture( iChannel0, (uv+ vec2(0.5,0.5))/256.0, -100.0 ).yx;
	vec2 rg2 = texture( iChannel0, (uv+ vec2(1.5,0.5))/256.0, -100.0 ).yx;
	vec2 rg3 = texture( iChannel0, (uv+ vec2(0.5,1.5))/256.0, -100.0 ).yx;
	vec2 rg4 = texture( iChannel0, (uv+ vec2(1.5,1.5))/256.0, -100.0 ).yx;
	vec2 rg = mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
	
	return mix( rg.x, rg.y, f.z );
}

// Function 96
vec3 voronoi( in vec2 x )
{
    vec2 n = floor( x );
    vec2 f = fract( x );

    vec2 mg, mr;
    
	vec3 m = vec3( 8.0 );
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2  g = vec2( float(i), float(j) );
        vec2  o = hash( n + g );
        
        vec2  r = g - f + o;
        //o = 0.5 + 0.5*sin( iTime + 6.2831*o );
 
	    //vec2  r = g - f; // square
		float d = dot( r, r );

        if( d<m.x ){
			mr = r;
			mg = g;
            m = vec3( d, o.x,o.y);
        }    
    }
    
    
    //distance to
    float md = 8.0 ;
  for( int j=-2; j<=2; j++ ){
	for( int i=-2; i<=2; i++ ){
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash( n + g );
        //o = 0.5 + 0.5*sin( iTime + 6.2831*o );
        
        o =  sin( iTime + 6.2831*o );
        vec2 r =  g - f + o;

        if( dot(mr-r,mr-r)>0.00001 )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }
  }


    return vec3( sqrt(m.x), m.y*m.z, md );
}

// Function 97
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );

    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( hash( i + vec2(0.0,0.0) ),
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ),
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 98
float gnoise(vec2 pos){
    return fract(sin(dot(pos, vec2(12.9898, 78.233))) * 43758.5453);
}

// Function 99
vec2 noise_prng_rand2_1_1(inout noise_prng this_)
{
    return -1.+2.*vec2(noise_prng_uniform_0_1(this_),
                       noise_prng_uniform_0_1(this_));
}

// Function 100
float noise( in vec2 p ){
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

	vec2 i = floor( p + (p.x+p.y)*K1 );
	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = step(a.yx,a.xy);    
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;

    vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );

	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));

    return dot( n, vec3(70.0) );
	
}

// Function 101
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec4 rg = texture( iChannel0, (uv+ 0.5)/256.0, -100.0 );
	return (-1.0+2.0*mix( rg.g, rg.r, f.z ));
}

// Function 102
float tweaknoise( vec3 p , bool step) {
    float d1 = smoothstep(grad/2.,-grad/2.,length(p)-.5),
          d2 = smoothstep(grad/1.,-grad/1.,abs(p.z)-.5),
          d=d1;
#if NOISE==1 // 3D Perlin noise
    float v = fbm(scale*p);
#elif NOISE==2 // Worley noise
    float v = (.9-scale*worley(scale*p).x);
#elif NOISE>=3 // trabeculum 3D
  #if VARIANT==0
    d = (1.-d1)*d2; 
  #elif VARIANT==2
    d=d2;
  #endif
    if (d<0.5) return 0.;
    grad=.8, scale = 10., thresh=.5+.5*(cos(.5*iTime)+.36*cos(.5*3.*iTime))/1.36;
    vec4 w=scale*worley(scale*p-vec3(0.,0.,3.*iTime)); 
    float v=1.-1./(1./(w.z-w.x)+1./(w.a-w.x)); // formula (c) Fabrice NEYRET - BSD3:mention author.
#endif
    
    return (true)? smoothstep(thresh-grad/2.,thresh+grad/2.,v*d) : v*d;
}

// Function 103
float noise(vec2 p)
{
	vec2 fl = floor(p);
	vec2 fr = fract(p);
	
	fr.x = smoothstep(0.0,1.0,fr.x);
	fr.y = smoothstep(0.0,1.0,fr.y);
	
	float a = mix(hash(fl + vec2(0.0,0.0)), hash(fl + vec2(1.0,0.0)),fr.x);
	float b = mix(hash(fl + vec2(0.0,1.0)), hash(fl + vec2(1.0,1.0)),fr.x);
	
	return mix(a,b,fr.y);
}

// Function 104
float noise3D(in vec3 p){
    
	const vec3 s = vec3(7, 157, 113);
	vec3 ip = floor(p); p -= ip; 
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    p = p*p*(3. - 2.*p); //p *= p*p*(p*(p * 6. - 15.) + 10.);
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
    h.xy = mix(h.xz, h.yw, p.y);
    return mix(h.x, h.y, p.z); // Range: [0, 1].
}

// Function 105
float noise1( sampler2D tex, in vec2 x )
{
    return textureLod(tex,(x+0.5)/64.0,0.0).x;
}

// Function 106
v0 noise(in v1 p
){const v0 K1 = 0.366025404 // (sqrt(3)-1)/2;
 ;const v0 K2 = 0.211324865 // (3-sqrt(3))/6;
 ;v1 i=floor(p+(p.x+p.y)*K1)
 ;v1 a=p-i+(i.x+i.y)*K2
 ;v1 o=(a.x>a.y) ? v1(1.0,0.0) : v1(0.0,1.0) //v1 of = 0.5 + 0.5*v1(sign(a.x-a.y), sign(a.y-a.x));
 ;v1 b=a-o+K2
 ;v1 c=a+u2(K2)
 ;v2 h=max( 0.5-v2(dot(a,a), dot(b,b), dot(c,c) ), 0.0 )
 ;v2 n=h*h*h*h*v2( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)))
 ;return dot( n, v2(70.0));}

// Function 107
float achnoise(vec3 x){ 
	vec3 p = floor(x);
	vec3 fr = fract(x);
	vec3 LBZ = p + vec3(0.0, 0.0, 0.0);
	vec3 LTZ = p + vec3(0.0, 1.0, 0.0);
	vec3 RBZ = p + vec3(1.0, 0.0, 0.0);
	vec3 RTZ = p + vec3(1.0, 1.0, 0.0);
	                   
	vec3 LBF = p + vec3(0.0, 0.0, 1.0);
	vec3 LTF = p + vec3(0.0, 1.0, 1.0);
	vec3 RBF = p + vec3(1.0, 0.0, 1.0);
	vec3 RTF = p + vec3(1.0, 1.0, 1.0);
	
	float l0candidate1 = oct(LBZ);
	float l0candidate2 = oct(RBZ);
	float l0candidate3 = oct(LTZ);
	float l0candidate4 = oct(RTZ);
	
	float l0candidate5 = oct(LBF);
	float l0candidate6 = oct(RBF);
	float l0candidate7 = oct(LTF);
	float l0candidate8 = oct(RTF);
	
	float l1candidate1 = mix(l0candidate1, l0candidate2, fr[0]);
	float l1candidate2 = mix(l0candidate3, l0candidate4, fr[0]);
	float l1candidate3 = mix(l0candidate5, l0candidate6, fr[0]);
	float l1candidate4 = mix(l0candidate7, l0candidate8, fr[0]);
	
	
	float l2candidate1 = mix(l1candidate1, l1candidate2, fr[1]);
	float l2candidate2 = mix(l1candidate3, l1candidate4, fr[1]);
	
	
	float l3candidate1 = mix(l2candidate1, l2candidate2, fr[2]);
	
	return l3candidate1;
}

// Function 108
float snoise(vec3 v)
    { 
      const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
      const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
      vec3 i  = floor(v + dot(v, C.yyy) );
      vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
      vec3 g = step(x0.yzx, x0.xyz);
      vec3 l = 1.0 - g;
      vec3 i1 = min( g.xyz, l.zxy );
      vec3 i2 = max( g.xyz, l.zxy );

      //   x0 = x0 - 0.0 + 0.0 * C.xxx;
      //   x1 = x0 - i1  + 1.0 * C.xxx;
      //   x2 = x0 - i2  + 2.0 * C.xxx;
      //   x3 = x0 - 1.0 + 3.0 * C.xxx;
      vec3 x1 = x0 - i1 + C.xxx;
      vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
      vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
      i = mod289(i); 
      vec4 p = permute( permute( permute( 
                 i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
               + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
               + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
      float n_ = 0.142857142857; // 1.0/7.0
      vec3  ns = n_ * D.wyz - D.xzx;

      vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

      vec4 x_ = floor(j * ns.z);
      vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

      vec4 x = x_ *ns.x + ns.yyyy;
      vec4 y = y_ *ns.x + ns.yyyy;
      vec4 h = 1.0 - abs(x) - abs(y);

      vec4 b0 = vec4( x.xy, y.xy );
      vec4 b1 = vec4( x.zw, y.zw );

      //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
      //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
      vec4 s0 = floor(b0)*2.0 + 1.0;
      vec4 s1 = floor(b1)*2.0 + 1.0;
      vec4 sh = -step(h, vec4(0.0));

      vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
      vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

      vec3 p0 = vec3(a0.xy,h.x);
      vec3 p1 = vec3(a0.zw,h.y);
      vec3 p2 = vec3(a1.xy,h.z);
      vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
      vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
      p0 *= norm.x;
      p1 *= norm.y;
      p2 *= norm.z;
      p3 *= norm.w;

    // Mix final noise value
      vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
      m = m * m;
      return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                    dot(p2,x2), dot(p3,x3) ) );
      }

// Function 109
vec3 simplexContour(vec2 p){
    
    
    
    // Scaling constant.
    const float gSc = 8.;
    p *= gSc;
    
    
    // Keeping a copy of the orginal position.
    vec2 oP = p;
    
    // Wobbling the coordinates, just a touch, in order to give a subtle hand drawn appearance.
    p += vec2(n2D3G(p*3.5), n2D3G(p*3.5 + 7.3))*.015;

    
    
    // SIMPLEX GRID SETUP
    
    vec2 s = floor(p + (p.x + p.y)*.36602540378); // Skew the current point.
    
    p -= s - (s.x + s.y)*.211324865; // Use it to attain the vector to the base vertex (from p).
    
    // Determine which triangle we're in. Much easier to visualize than the 3D version.
    float i = p.x < p.y? 1. : 0.; // Apparently, faster than: i = step(p.y, p.x);
    vec2 ioffs = vec2(1. - i, i);
    
    // Vectors to the other two triangle vertices.
    vec2 ip0 = vec2(0), ip1 = ioffs - .2113248654, ip2 = vec2(.577350269); 
    
    
    // Centralize everything, so that vec2(0) is in the center of the triangle.
    vec2 ctr = (ip0 + ip1 + ip2)/3.; // Centroid.
    //
    ip0 -= ctr; ip1 -= ctr; ip2 -= ctr; p -= ctr;
     
     
     
    // Take a function value (noise, in this case) at each of the vertices of the
    // individual triangle cell. Each will be compared the isovalue. 
    vec3 n3;
    n3.x = isoFunction(s);
    n3.y = isoFunction(s + ioffs);
    n3.z = isoFunction(s + 1.);
    
    
    // Various distance field values.
    float d = 1e5, d2 = 1e5, d3 = 1e5, d4 = 1e5, d5 = 1e5; 
  
    
    // The first contour, which separates the terrain (grass or barren) from the beach.
    float isovalue = 0.;
    
    // The contour edge points that the line will run between. Each are passed into the
    // function below and calculated.
    vec2 p0, p1; 
    
    // The isoline. The edge values (p0 and p1) are calculated, and the ID is returned.
    int iTh = isoLine(n3, ip0, ip1, ip2, isovalue, i, p0, p1);
      
    // The minimum distance from the pixel to the line running through the triangle edge 
    // points.
    d = min(d, distEdge(p - p0, p - p1)); 
    
    
    
    //if(iTh == 0) d = 1e5;
    
    // Totally internal, which means a terrain (grass) hit.
    if(iTh == 7){ // 12-20  
 		
        // Triangle.
        //d = min(min(distEdge(p - ip0, p - ip1), distEdge(p - ip1, p - ip2)), 
                  //distEdge(p - ip0, p - ip2));
        
        // Easier just to set the distance to a hit.
        d = 0.;
    } 
    
 
    
    // Contour lines.
    d3 = min(d3, distLine((p - p0), (p - p1))); 
    // Contour points.
    d4 = min(d4, min(length(p - p0), length(p - p1))); 
    
    
    
    
    
    // Displaying the 2D simplex grid. Basically, we're rendering lines between
    // each of the three triangular cell vertices to show the outline of the 
    // cell edges.
    float tri = min(min(distLine(p - ip0, p - ip1), distLine(p - ip1, p - ip2)), 
                  distLine(p - ip2, p - ip0));
    
    // Adding the triangle grid to the d5 distance field value.
    d5 = min(d5, tri);
     
    
    // Dots in the centers of the triangles, for whatever reason. :) Take them out, if
    // you prefer a cleaner look.
    d5 = min(d5, length(p) - .02);   
    
    ////////
    #ifdef TRIANGULATE_CONTOURS
    vec2 oldP0 = p0;
    vec2 oldP1 = p1;

    // Contour triangles: Flagging when the triangle cell contains a contour line, or not.
    float td = (iTh>0 && iTh<7)? 1. : 0.;
    
    // Subdivide quads on the first contour.
    if(iTh==3 || iTh==5 || iTh==6){

        // Grass (non-beach land) only quads.
        vec2 pt = p0;
        if(i==1.) pt = p1;
        d5 = min(d5, distLine((p - pt), (p - ip0))); 
        d5 = min(d5, distLine((p - pt), (p - ip1)));  
        d5 = min(d5, distLine((p - pt), (p - ip2))); 
    }
    #endif
    ////////
    
 
    // The second contour: This one demarcates the beach from the sea.
    isovalue = -.15;
   
    // The isoline. The edge values (p0 and p1) are calculated, and the ID is returned.
    int iTh2 = isoLine(n3, ip0, ip1, ip2, isovalue, i, p0, p1);
   
    // The minimum distance from the pixel to the line running through the triangle edge 
    // points.   
    d2 = min(d2, distEdge(p - p0, p - p1)); 
    
    // Make a copy.
    float oldD2 = d2;
    
    if(iTh2 == 7) d2 = 0.; 
    if(iTh == 7) d2 = 1e5;
    d2 = max(d2, -d);

     
    // Contour lines - 2nd (beach) contour.
    d3 = min(d3, distLine((p - p0), (p - p1)));
    // Contour points - 2nd (beach) contour.
    d4 = min(d4, min(length(p - p0), length(p - p1))); 
                
    d4 -= .075;
    d3 -= .0125;
     
    ////////
    #ifdef TRIANGULATE_CONTOURS
    // Triangulating the contours.
    
    // This logic was put in at the last minute, and isn't my finest work. :)
    // It seems to work, but I'd like to tidy it up later. 

    // Flagging when the triangle contains a second contour line, or not.
    float td2 = (iTh2>0 && iTh2<7)? 1. : 0.;
     
    
    if(td==1. && td2==1.){
        // Both contour lines run through a triangle, so you need to do a little more
        // subdividing. 
        
        // The beach colored quad between the first contour and second contour.
        d5 = min(d5, distLine(p - p0, p - oldP0)); 
        d5 = min(d5, distLine(p - p0, p - oldP1));  
        d5 = min(d5, distLine(p - p1, p - oldP1));
         
        // The quad between the water and the beach.
        if(oldD2>0.){
            vec2 pt = p0;
            if(i==1.) pt = p1;
            d5 = min(d5, distLine(p - pt, p - ip0)); 
            d5 = min(d5, distLine(p - pt, p - ip1));  
            d5 = min(d5, distLine(p - pt, p - ip2)); 
        }
    }   
    else if(td==1. && td2==0.){
        
        // One contour line through the triangle.
        
        // Beach and grass quads.
        vec2 pt = oldP0;
        if(i==1.) pt = oldP1;
        d5 = min(d5, distLine(p - pt, p - ip0)); 
        d5 = min(d5, distLine(p - pt, p - ip1));  
        d5 = min(d5, distLine(p - pt, p - ip2)); 
    }
    else if(td==0. && td2==1.){ 
        
        // One contour line through the triangle.
        
        // Beach and water quads.
        vec2 pt = p0;
        if(i==1.) pt = p1;
        d5 = min(d5, distLine(p - pt, p - ip0)); 
        d5 = min(d5, distLine(p - pt, p - ip1));  
        d5 = min(d5, distLine(p - pt, p - ip2));  
    }
    
    #endif
    ////////
    
    
    // The screen coordinates have been scaled up, so the distance values need to be
    // scaled down.
    d /= gSc;
    d2 /= gSc;
    d3 /= gSc;
    d4 /= gSc;    
    d5 /= gSc; 
    
    
    
    // Rendering - Coloring.
        
    // Initial color.
    vec3 col = vec3(1, .85, .6);
    
    // Smoothing factor.
    float sf = .004; 
   
    // Water.
    if(d>0. && d2>0.) col = vec3(1, 1.8, 3)*.45;
     // Water edging.
    if(d>0.) col = mix(col, vec3(1, 1.85, 3)*.3, (1. - smoothstep(0., sf, d2 - .012)));
    
    // Beach.
    col = mix(col, vec3(1.1, .85, .6),  (1. - smoothstep(0., sf, d2)));
    // Beach edging.
    col = mix(col, vec3(1.5, .9, .6)*.6, (1. - smoothstep(0., sf, d - .012)));
    
    #ifdef GRASS
    // Grassy terrain.
    col = mix(col, vec3(1, .8, .6)*vec3(.7, 1., .75)*.95, (1. - smoothstep(0., sf, d))); 
    #else
    // Alternate barren terrain.
    col = mix(col, vec3(1, .82, .6)*.95, (1. - smoothstep(0., sf, d))); 
    #endif 
    
   
     
 
    // Abstract shading, based on the individual noise height values for each triangle.
    if(d2>0.) col *= (abs(dot(n3, vec3(1)))*1.25 + 1.25)/2.;
    else col *= max(2. - (dot(n3, vec3(1)) + 1.45)/1.25, 0.);
    
    // More abstract shading.
    //if(iTh!=0) col *= float(iTh)/7.*.5 + .6;
    //else col *= float(3.)/7.*.5 + .75;

    
    ////////
    #ifdef TRIANGULATE_CONTOURS
    //if(td==1. || td2==1.) col *= vec3(1, .4, .8); 
    #endif
    ////////
    
    ////////
    #ifdef TRIANGLE_PATTERN
    // A concentric triangular pattern.
    float pat = abs(fract(tri*12.5 + .4) - .5)*2.;
    col *= pat*.425 + .75; 
    #endif
    ////////
    
 
    
    
    // Triangle grid overlay.
    col = mix(col, vec3(0), (1. - smoothstep(0., sf, d5))*.95);
    
     
    
    // Lines.
    col = mix(col, vec3(0), (1. - smoothstep(0., sf, d3)));
    
    
    // Dots.
    col = mix(col, vec3(0), (1. - smoothstep(0., sf, d4)));
    col = mix(col, vec3(1), (1. - smoothstep(0., sf, d4 + .005)));
  
  
    
    // Rough pencil color overlay... The calculations are rough... Very rough, in fact, 
    // since I'm only using a small overlayed portion of it. Flockaroo does a much, much 
    // better pencil sketch algorithm here:
    //
    // When Voxels Wed Pixels - Flockaroo 
    // https://www.shadertoy.com/view/MsKfRw
    //
    // Anyway, the idea is very simple: Render a layer of noise, stretched out along one 
    // of the directions, then mix a similar, but rotated, layer on top. Whilst doing this,
    // compare each layer to it's underlying grey scale value, and take the difference...
    // I probably could have described it better, but hopefully, the code will make it 
    // more clear. :)
    // 
    // Tweaked to suit the brush stroke size.
    vec2 q = oP*1.5;
    // I always forget this bit. Without it, the grey scale value will be above one, 
    // resulting in the extra bright spots not having any hatching over the top.
    col = min(col, 1.);
    // Underlying grey scale pixel value -- Tweaked for contrast and brightness.
    float gr = sqrt(dot(col, vec3(.299, .587, .114)))*1.25;
    // Stretched fBm noise layer.
    float ns = (n2D3G(q*4.*vec2(1./3., 3))*.64 + n2D3G(q*8.*vec2(1./3., 3))*.34)*.5 + .5;
    // Compare it to the underlying grey scale value.
    ns = gr - ns;
    //
    // Repeat the process with a rotated layer.
    q *= rot2(3.14159/3.);
    float ns2 = (n2D3G(q*4.*vec2(1./3., 3))*.64 + n2D3G(q*8.*vec2(1./3., 3))*.34)*.5 + .5;
    ns2 = gr - ns2;
    //
    // Mix the two layers in some way to suit your needs. Flockaroo applied common sense, 
    // and used a smooth threshold, which works better than the dumb things I was trying. :)
    ns = smoothstep(0., 1., min(ns, ns2)); // Rough pencil sketch layer.
    //
    // Mix in a small portion of the pencil sketch layer with the clean colored one.
    col = mix(col, col*(ns + .35), .4);
    // Has more of a colored pencil feel. 
    //col *= vec3(.8)*ns + .5;    
    // Using Photoshop mixes, like screen, overlay, etc, gives more visual options. Here's 
    // an example, but there's plenty more. Be sure to uncomment the "softLight" function.
    //col = softLight(col, vec3(ns)*.75);
    // Uncomment this to see the pencil sketch layer only.
    //col = vec3(ns);
    
    
    /*
    // Just some line overlays.
    vec2 pt = p;
    float offs = -.5;
    if(i<.5) offs += 2.;//pt.xy = -pt.xy;
    pt = rot2(6.2831/3.)*pt;
    float pat2 = clamp(cos(pt.x*6.2831*14. - offs)*2. + 1.5, 0., 1.);
    col *= pat2*.4 + .8;
    */
    
    
    // Cheap paper grain.
    //oP = floor(oP/gSc*1024.);
    //vec3 rn3 = vec3(hash21(oP), hash21(oP + 2.37), hash21(oP + 4.83));
    //col *= .9 + .1*rn3.xyz  + .1*rn3.xxx;

    
    // Return the simplex weave value.
    return col;
 

}

// Function 110
float SmoothNoise(in vec2 o) 
{
	vec2 p = floor(o);
	vec2 f = fract(o);

	float a = Hash12(p);
	float b = Hash12(p+vec2(1,0));
	float c = Hash12(p+vec2(0,1));
	float d = Hash12(p+vec2(1,1));
	
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;
	
	vec2 t = 3.0 * f2 - 2.0 * f3;
	
	float u = t.x;
	float v = t.y;

	float res = a + (b-a)*u +(c-a)*v + (a-b+d-c)*u*v;
    
    return res;
}

// Function 111
float noise(float s){
    // Noise is sampled at every integer s
    // If s = t*f, the resulting signal is close to a white noise
    // with a sharp cutoff at frequency f.
    int si = int(floor(s));
    float sf = fract(s);
    sf = sf*sf*(3.-2.*sf); // smoothstep(0,1,sf)
    return mix(rand(float(si)), rand(float(si+1)), sf)*2.-1.;
}

// Function 112
vec3 Voronoi( vec2 pos )
{
	// find closest & second closest points
	vec3 delta = vec3(-1,0,1);

	// sample surrounding points on the distorted grid
	// could get 2 samples for the price of one using a rotated (17,37) grid...
	vec3 point[9];
	point[0] = VoronoiPoint( pos, delta.xx );
	point[1] = VoronoiPoint( pos, delta.yx );
	point[2] = VoronoiPoint( pos, delta.zx );
	point[3] = VoronoiPoint( pos, delta.xy );
	point[4] = VoronoiPoint( pos, delta.yy );
	point[5] = VoronoiPoint( pos, delta.zy );
	point[6] = VoronoiPoint( pos, delta.xz );
	point[7] = VoronoiPoint( pos, delta.yz );
	point[8] = VoronoiPoint( pos, delta.zz );

	vec3 closest;
	closest.z =
		min(
			min(
				min(
					min( point[0].z, point[1].z ),
					min( point[2].z, point[3].z )
				), min(
					min( point[4].z, point[5].z ),
					min( point[6].z, point[7].z )
				)
			), point[8].z
		);
	
	// find second closest
	// maybe there's a better way to do this
	closest.xy = point[8].xy;
	for ( int i=0; i < 8; i++ )
	{
		if ( closest.z == point[i].z )
		{
			closest = point[i];
			point[i] = point[8];
		}
	}
		
	float t;
	t = min(
			min(
				min( point[0].z, point[1].z ),
				min( point[2].z, point[3].z )
			), min(
				min( point[4].z, point[5].z ),
				min( point[6].z, point[7].z )
			)
		);

	/*slower:
	float t2 = 9.0;
	vec3 closest = point[8];
	for ( int i=0; i < 8; i++ )
	{
		if ( point[i].z < closest.z )
		{
			t2 = closest.z;
			closest = point[i];
		}
		else if ( point[i].z < t2 )
		{
			t2 = point[i].z;
		}
	}*/
	
	return vec3( closest.xy, t-closest.z );
}

// Function 113
float MNoise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( Hash( i + vec2(0.0,0.0) ), 
                     Hash( i + vec2(1.0,0.0) ), u.x),
                mix( Hash( i + vec2(0.0,1.0) ), 
                     Hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 114
float noise21(vec2 uv)
{
    vec2 n = fract(uv* vec2(19.48, 139.9));
    n += sin(dot(uv, uv + 30.7)) * 47.0;
    return fract(n.x * n.y);
}

// Function 115
float noise(vec2 uv){
    vec2 off = vec2(1,0);
    vec2 fuv = floor(uv);
    
    float tl = rand(fuv);
    float tr = rand(fuv + off.xy);
    float bl = rand(fuv + off.yx);
    float br = rand(fuv + off.xx);
    
    vec2 fruv = fract(uv);
    fruv = fruv * fruv * (2.0 - fruv);
    
    return mix(mix(tl,tr,fruv.x), mix(bl,br,fruv.x),fruv.y);
}

// Function 116
float voronoi( in vec2 x ){
    vec2 c = voronoi2d( x ).xy;
	return 0.5 + 0.5*cos( c.y*6.2831 + 0.0 );
}

// Function 117
vec3 atm_cloudnoise1_offs( vec4 r, bool lowfreq )
{
    float lod = log2( r.w );
    const vec2 offs = vec2( 1, 0 );
    vec3 offs_y = vec3(
		textureLod( iChannel3, ( r.xyz + offs.xyy ) / 32., lod ).x + 2. * textureLod( iChannel3, ( r.xyz + offs.xyy ) / 64., lod - 1. ).x,
		textureLod( iChannel3, ( r.xyz + offs.yxy ) / 32., lod ).x + 2. * textureLod( iChannel3, ( r.xyz + offs.yxy ) / 64., lod - 1. ).x,
		textureLod( iChannel3, ( r.xyz + offs.yyx ) / 32., lod ).x + 2. * textureLod( iChannel3, ( r.xyz + offs.yyx ) / 64., lod - 1. ).x ) / 3.;
    if( !lowfreq )
        offs_y = 4. / 5. * offs_y + vec3(
        	textureLod( iChannel3, ( r.xyz + offs.xyy ) / 8., lod + 2. ).x + 2. * textureLod( iChannel3, ( r.xyz + offs.xyy ) / 16., lod + 1. ).x,
			textureLod( iChannel3, ( r.xyz + offs.yxy ) / 8., lod + 2. ).x + 2. * textureLod( iChannel3, ( r.xyz + offs.yxy ) / 16., lod + 1. ).x,
			textureLod( iChannel3, ( r.xyz + offs.yyx ) / 8., lod + 2. ).x + 2. * textureLod( iChannel3, ( r.xyz + offs.yyx ) / 16., lod + 1. ).x ) / 15.;
    return offs_y;
}

// Function 118
float snoise(vec3 v)
      { 
      const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
      const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
      vec3 i  = floor(v + dot(v, C.yyy) );
      vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
      vec3 g = step(x0.yzx, x0.xyz);
      vec3 l = 1.0 - g;
      vec3 i1 = min( g.xyz, l.zxy );
      vec3 i2 = max( g.xyz, l.zxy );

      //   x0 = x0 - 0.0 + 0.0 * C.xxx;
      //   x1 = x0 - i1  + 1.0 * C.xxx;
      //   x2 = x0 - i2  + 2.0 * C.xxx;
      //   x3 = x0 - 1.0 + 3.0 * C.xxx;
      vec3 x1 = x0 - i1 + C.xxx;
      vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
      vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
      i = mod289(i); 
      vec4 p = permute( permute( permute( 
                 i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
               + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
               + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
      float n_ = 0.142857142857; // 1.0/7.0
      vec3  ns = n_ * D.wyz - D.xzx;

      vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

      vec4 x_ = floor(j * ns.z);
      vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

      vec4 x = x_ *ns.x + ns.yyyy;
      vec4 y = y_ *ns.x + ns.yyyy;
      vec4 h = 1.0 - abs(x) - abs(y);

      vec4 b0 = vec4( x.xy, y.xy );
      vec4 b1 = vec4( x.zw, y.zw );

      //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
      //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
      vec4 s0 = floor(b0)*2.0 + 1.0;
      vec4 s1 = floor(b1)*2.0 + 1.0;
      vec4 sh = -step(h, vec4(0.0));

      vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
      vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

      vec3 p0 = vec3(a0.xy,h.x);
      vec3 p1 = vec3(a0.zw,h.y);
      vec3 p2 = vec3(a1.xy,h.z);
      vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
      vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
      p0 *= norm.x;
      p1 *= norm.y;
      p2 *= norm.z;
      p3 *= norm.w;

    // Mix final noise value
      vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
      m = m * m;
      return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                    dot(p2,x2), dot(p3,x3)));
	}

// Function 119
vec2 noise2(vec2 x)
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    return mix(mix( hash2(p),          hash2(p + add.xy),f.x),
                    mix( hash2(p + add.yx), hash2(p + add.xx),f.x),f.y);
    
}

// Function 120
float iqnoise( vec3 x )
{
    // The noise function returns a value in the range -1.0f -> 1.0f
    vec3 p = floor(x);
    vec3 f = fract(x);

    f       = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    return mix(mix(mix( iqhash(n+0.0  ), iqhash(n+1.0  ),f.x),
                   mix( iqhash(n+57.0 ), iqhash(n+58.0 ),f.x),f.y),
               mix(mix( iqhash(n+113.0), iqhash(n+114.0),f.x),
                   mix( iqhash(n+170.0), iqhash(n+171.0),f.x),f.y),f.z);
}

// Function 121
ivec2 voronoiZoom(ivec2 coord, int baseSeed, int size)
{
    ivec2 cell = coord/size;
    vec2 posInCell = vec2(coord-cell*size)/float(size);
    ivec2 newCell = cell;
    float minDist2 = 4.0;
    for (int i=-2; i<=2; ++i)
        for (int j=-2; j<=2; ++j)
        {
            float cx = ivecnoise(cell+ivec2(i, j), baseSeed+0);
            float cy = ivecnoise(cell+ivec2(i, j), baseSeed+1);
            vec2 displacement = vec2(cx+float(i), cy+float(j))-posInCell;
            float dist2 = dot(displacement, displacement);
            if (dist2 < minDist2)
            {
                newCell = cell+ivec2(i, j);
                minDist2 = dist2;
            }
        }
    return newCell;
}

// Function 122
float snoise_1_7(vec3 v)
  {
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D_1_8 = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g_1_9 = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g_1_9;
  vec3 i1 = min( g_1_9.xyz, l.zxy );
  vec3 i2 = max( g_1_9.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D_1_8.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289_1_4(i);
  vec4 p = permute_1_5( permute_1_5( permute_1_5(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D_1_8.wyz - D_1_8.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1_1_10 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0_1_11 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1_1_10.xy,h.z);
  vec3 p3 = vec3(a1_1_10.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt_1_6(vec4(dot(p0_1_11,p0_1_11), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0_1_11 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0_1_11,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
  }

// Function 123
vec3 voronoi_rounder( in vec2 x, in float s, in float e )
{
#if DOMAIN_DEFORM
	x += sin(x.yx*10.)*.07;
#endif

    vec2 n = floor(x);
    vec2 f = fract(x);

    vec2 mr, mg;
    float md = closest(n,f,mr,mg);

    //----------------------------------
    // second pass: distance to edges
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) // skip the same cell
        {
            float d = dot( 0.5*(mr+r), normalize(r-mr) );

            // The whole trick to get continuous function
            // across whole domain and smooth at non-zero distance
            // is to use smooth minimum (as usual)
            // and multiple smoothness factor by distance,
            // so it becomes minimum at zero distance.
            // Simple as that!
            // If you keep smoothness factor constant (i.e. multiple by "s" only),
            // the distance function becomes discontinuous
            // (see https://www.shadertoy.com/view/MdSfzD).
            md = smin(d, md, s*d);
        }
    }

    // Totally empirical compensation for
    // smoothing scaling side-effect.
    md *= .5 + s;

    // At the end do some smooth abs
    // on the distance value.
    // This is really optional, since distance function
    // is already continuous, but we can get extra
    // smoothness from it.
    md = sabs(md, e);

    return vec3( md, mr );
}

// Function 124
float sampleCloudNoise(vec2 cloudUV)
{
    vec3 cloudWind = -s_time * kTimeScale * 0.001 * oz.xxy;
    vec3 cloudDetailWind = s_time * kTimeScale * 0.001 * oz.yxy;
    vec3 uv = vec3(cloudUV * 2.0, 16.5/32.0);
    return (  1.000*textureLod(iChannel1, uv*0.01 + cloudWind * 0.01, 0.0 ).r 
            + 0.500*textureLod(iChannel1, uv*0.03 + cloudWind * 0.50, 0.0 ).r
            + 0.250*textureLod(iChannel1, uv*0.07 + cloudWind * 3.00, 0.0 ).r
            + 0.125*textureLod(iChannel1, uv*0.25 + cloudDetailWind * 15.0, 0.0 ).r
            ) * 0.75 - (0.3 * (1.0 - saturate(dot(cloudUV, cloudUV) / 7.0)));
}

// Function 125
vec3 noised( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    vec2 u = f*f*(3.0-2.0*f);
	float a = textureLod(iChannel0,(p+vec2(0.5,0.5))/256.0,0.0).x;
	float b = textureLod(iChannel0,(p+vec2(1.5,0.5))/256.0,0.0).x;
	float c = textureLod(iChannel0,(p+vec2(0.5,1.5))/256.0,0.0).x;
	float d = textureLod(iChannel0,(p+vec2(1.5,1.5))/256.0,0.0).x;
	return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
				6.0*f*(1.0-f)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));
}

// Function 126
float noise (in vec2 p) 
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 sf = f * f * (3.0 - 2.0 * f);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, sf.x), mix(c, d, sf.x), sf.y);
}

// Function 127
float perlin_noise(vec3 p)
{
    vec3 pi = floor(p);
    vec3 pf = p - pi;
    
    vec3 w = pf * pf * (3.0 - 2.0 * pf);
    
    return 	mix(
        		mix(
                	mix(dot(pf - vec3(0, 0, 0), hash33(pi + vec3(0, 0, 0))), 
                        dot(pf - vec3(1, 0, 0), hash33(pi + vec3(1, 0, 0))),
                       	w.x),
                	mix(dot(pf - vec3(0, 0, 1), hash33(pi + vec3(0, 0, 1))), 
                        dot(pf - vec3(1, 0, 1), hash33(pi + vec3(1, 0, 1))),
                       	w.x),
                	w.z),
        		mix(
                    mix(dot(pf - vec3(0, 1, 0), hash33(pi + vec3(0, 1, 0))), 
                        dot(pf - vec3(1, 1, 0), hash33(pi + vec3(1, 1, 0))),
                       	w.x),
                   	mix(dot(pf - vec3(0, 1, 1), hash33(pi + vec3(0, 1, 1))), 
                        dot(pf - vec3(1, 1, 1), hash33(pi + vec3(1, 1, 1))),
                       	w.x),
                	w.z),
    			w.y);
}

// Function 128
float snoise(vec2 v) {
  float X=.211324865405187, Y=.366025403784439, Z=-.577350269189626, W=.024390243902439;
  vec2 i = floor(v + (v.x+v.y)*Y),
      x0 = v -   i + (i.x+i.y)*X,
       j = step(x0.yx, x0),
      x1 = x0+X-j, 
      x3 = x0+Z; 

  i = mod(i,289.);
  vec3 p = permute( permute( i.y + vec3(0, j.y, 1 ))
                           + i.x + vec3(0, j.x, 1 )   ),

       m = max( .5 - vec3(dot(x0,x0), dot(x1,x1), dot(x3,x3)), 0.),
       x = 2. * fract(p * W) - 1.,
       h = abs(x) - .5,
      a0 = x - floor(x + .5),
       g = a0 * vec3(x0.x,x1.x,x3.x) 
          + h * vec3(x0.y,x1.y,x3.y); 

  m = m*m*m*m* ( 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h ) );  
  return 130. * dot(m, g);
}

// Function 129
float noised_caustics_improveXYPlanes(in vec3 x)
{
    mat3 orthonormalMap = mat3(
        0.788675134594813, -0.211324865405187, -0.577350269189626,
        -0.211324865405187, 0.788675134594813, -0.577350269189626,
        0.577350269189626, 0.577350269189626, 0.577350269189626);
    x = x * orthonormalMap;
    
    vec4 result = noised(x);
    float value = noised(x - 0.125 * result.yzw).x;
    
    return value;
}

// Function 130
float perlin_noise3(vec3 p) {
    vec3 pi = floor(p);
    vec3 pf = p - pi;
    
    vec3 w = pf * pf * (3. - 2. * pf);
    
    return 	mix(
    	mix(
            mix(
                dot(pf - vec3(0, 0, 0), hash3_3(pi + vec3(0, 0, 0))), 
                dot(pf - vec3(1, 0, 0), hash3_3(pi + vec3(1, 0, 0))),
                w.x),
            mix(
                dot(pf - vec3(0, 0, 1), hash3_3(pi + vec3(0, 0, 1))), 
                dot(pf - vec3(1, 0, 1), hash3_3(pi + vec3(1, 0, 1))),
                w.x),
    	w.z),
        mix(
            mix(
                dot(pf - vec3(0, 1, 0), hash3_3(pi + vec3(0, 1, 0))), 
                dot(pf - vec3(1, 1, 0), hash3_3(pi + vec3(1, 1, 0))),
                w.x),
            mix(
                dot(pf - vec3(0, 1, 1), hash3_3(pi + vec3(0, 1, 1))), 
                dot(pf - vec3(1, 1, 1), hash3_3(pi + vec3(1, 1, 1))),
                w.x),
     	w.z),
	w.y);
}

// Function 131
float gaussianNoise(vec2 uv)
{
	vec2 p = floor(uv);
    vec2 f = smoothstep(0.0, 1.0, fract(uv));
    
    f = f*f*(3.0-2.0*f);
    /*float c = cos(uv.x);
    float s = sin(uv.y);
    mat2 R = mat2(c, s, -s, c);*/
    
    return mix(
        	mix(hash2D(p+vec2(0.0, 0.0)), hash2D(p+vec2(1.0, 0.0)), f.x),
        	mix(hash2D(p+vec2(0.0, 1.0)), hash2D(p+vec2(1.0, 1.0)), f.x),
        	f.y);
}

// Function 132
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
    	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 133
float polynoise(vec2 p)
{
    vec2 seed = floor(p);
    vec2 rndv = vec2( rnd(seed.xy), rnd(seed.yx));
    vec2 pt = fract(p);
    float bx = value(pt.x, rndv.x);
    float by = value(pt.y, rndv.y);
    return min(bx, by) * abs(rnd(seed.xy * 0.1));
}

// Function 134
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z);
	vec2 rga = textureLod( iChannel0, (uv+vec2(0.5,0.5))/256.0, 0.0 ).yx;
	vec2 rgb = textureLod( iChannel0, (uv+vec2(1.5,0.5))/256.0, 0.0 ).yx;
	vec2 rgc = textureLod( iChannel0, (uv+vec2(0.5,1.5))/256.0, 0.0 ).yx;
	vec2 rgd = textureLod( iChannel0, (uv+vec2(1.5,1.5))/256.0, 0.0 ).yx;
	
	vec2 rg = mix( mix( rga, rgb, f.x ),
				   mix( rgc, rgd, f.x ), f.y );
	
	return mix( rg.x, rg.y, f.z );
}

// Function 135
vec4 noise(vec4 v){
    // ensure reasonable range
    v = fract(v) + fract(v*1e4) + fract(v*1e-4);
    // seed
    v += vec4(0.12345, 0.6789, 0.314159, 0.271828);
    // more iterations => more random
    v = fract(v*dot(v, v)*123.456);
    v = fract(v*dot(v, v)*123.456);
    return v;
}

// Function 136
float noise(vec3 v)
  { 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
  }

// Function 137
vec4 dnoise(vec3 p) 
{
	 /* 1. find current tetrahedron T and its four vertices */
	 /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
	 /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/
	 
	 vec3 s = floor(p + (p.x+p.y+p.z)*F3);
	 vec3 x = p - s + (s.x+s.y+s.z)*G3;
	 
	 vec3 e = step(vec3(0.0), x - x.yzx);
	 vec3 i1 = e*(1.0 - e.zxy);
	 vec3 i2 = 1.0 - e.zxy*(1.0 - e);
	 	
	 vec3 x1 = x - i1 + G3;
	 vec3 x2 = x - i2 + 2.0*G3;
	 vec3 x3 = x - 1.0 + 3.0*G3;
	 		 
	 /* calculate surflet weights */
	 vec4 w;
	 w.x = dot(x, x);
	 w.y = dot(x1, x1);
	 w.z = dot(x2, x2);
	 w.w = dot(x3, x3);
	 
	 /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
	 w = max(0.6 - w, 0.0);		//aka t0,t1,t2,t3
	 vec4 w2 = w*w;				//aka t20,t21,t22,t23
	 vec4 w4 = w2*w2;			//aka t40,t41,t42,t43
	 
	 /* 2. find four surflets and store them in d */
	 vec3 g0 = random3(s);
	 vec3 g1 = random3(s + i1);
	 vec3 g2 = random3(s + i2);
	 vec3 g3 = random3(s + 1.0);
	 
	 vec4 d;
	 /* calculate surflet components */
	 d.x = dot(g0, x);		//aka graddotp3( gx0, gy0, gz0, x0, y0, z0 )
	 d.y = dot(g1, x1);
	 d.z = dot(g2, x2);
	 d.w = dot(g3, x3);
	 
	 //derivatives as per
	 //http://webstaff.itn.liu.se/~stegu/aqsis/flownoisedemo/srdnoise23.c
	 vec4 w3 = w*w2;
	 vec4 temp = w3*d;
	 vec3 dnoise = temp[0]*x;
	     dnoise += temp[1]*x1;
	     dnoise += temp[2]*x2;
		 dnoise += temp[3]*x3;
		 dnoise *= -8.;
		 dnoise += w4[0]*g0+w4[1]*g1+w4[2]*g2+w4[3]*g3;
		 dnoise *= 52.; //???
		 
	 d *= w4;	//aka n0,n1,n2,n3
	 
	float n = (d.x+d.y+d.z+d.w)*52.;
	
	return vec4(dnoise,n);
}

// Function 138
float perlinNoise1D(float p)
{
	float pi = floor(p), pf = p - pi, w = fade(pf);
    return mix(grad(hash(pi), pf), grad(hash(pi + 1.0), pf - 1.0), w) * 2.0;
}

// Function 139
float fractal_noise(vec3 m) {
    return   0.5333333*simplex3d(m*rot1)
			+0.2666667*simplex3d(2.0*m*rot2)
			+0.1333333*simplex3d(4.0*m*rot3)
			+0.0666667*simplex3d(8.0*m);
}

// Function 140
vec1 noise1(vec2 p){vec2 f=floor(p);p=herm32(fract(p));return mx(p.x,mix(hash2(f),hash2(f+vec2(0,1)),p.y));}

// Function 141
float ddnoise(vec3 v, float f1, float f2)
{
    return snoise(v*f1)/f1 + snoise(v*f2)/f2;
}

// Function 142
vec2 Noise2D( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 res = mix(mix( Hash(p + 0.0), Hash(p + vec2(1.0, 0.0)),f.x),
                   mix( Hash(p + vec2(0.0, 1.0) ), Hash(p + vec2(1.0, 1.0)),f.x),f.y);
    return res-.5;
}

// Function 143
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    return mix(mix( hash1(n+  0.0), hash1(n+  1.0),f.x),
               mix( hash1(n+ 57.0), hash1(n+ 58.0),f.x),f.y);
}

// Function 144
float simplex_noise(vec3 p)
{
    p *= 4.0;
    const float K1 = 0.333333333;
    const float K2 = 0.166666667;
    
    vec3 i = floor(p + (p.x + p.y + p.z) * K1);
    vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);
    
    // thx nikita: https://www.shadertoy.com/view/XsX3zB
    vec3 e = step(vec3(0.0), d0 - d0.yzx);
	vec3 i1 = e * (1.0 - e.zxy);
	vec3 i2 = 1.0 - e.zxy * (1.0 - e);
    
    vec3 d1 = d0 - (i1 - 1.0 * K2);
    vec3 d2 = d0 - (i2 - 2.0 * K2);
    vec3 d3 = d0 - (1.0 - 3.0 * K2);
    
    vec4 h = max(0.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
    vec4 n = h * h * h * h * vec4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0)));
    
    return dot(vec4(31.316), n);
}

// Function 145
float Noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    float res = mix(mix( Hash(p), Hash(p+ vec2(1.0, 0.0)),f.x),
                    mix( Hash(p+ vec2(.0, 1.0)), Hash(p+ vec2(1.0, 1.0)),f.x),f.y);
    return res;
}

// Function 146
float noise(float x, float y, float z)
{  
    // Find the unit cube that contains the point
    int X = int(floor(x)) & 255;
    int Y = int(floor(y)) & 255;
    int Z = int(floor(z)) & 255;

    // Find relative x, y,z of point in cube
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);
    
    // Compute fade curves for each of x, y, z
    float u = fade(x);
    float v = fade(y);
    float w = fade(z);

    // Hash coordinates of the 8 cube corners
    int A = p[X] + Y;
    int AA = p[A] + Z;
    int AB = p[A + 1] + Z;
    int B = p[X + 1] + Y;
    int BA = p[B] + Z;
    int BB = p[B + 1] + Z;

    // Add blended results from 8 corners of cube
    float res = mix(
        mix(
            mix(grad(p[AA], x, y, z),
                 grad(p[BA], x - 1.0f, y, z),
                 u),
            mix(grad(p[AB], x, y - 1.0f, z),
                 grad(p[BB], x - 1.0f, y - 1.0f, z),
                 u),
            v),
        mix(
            mix(grad(p[AA + 1], x, y, z - 1.0f),
                 grad(p[BA + 1], x - 1.0f, y, z - 1.0f),
                 u),
            mix(grad(p[AB + 1], x, y - 1.0f, z - 1.0f),
                 grad(p[BB + 1], x - 1.0f, y - 1.0f, z - 1.0f),
                 u),
            v),
        w);
    return (res + 1.0f) / 2.0f;
}

// Function 147
vec3 noise3(vec3 x) {
	return vec3( noise(x+vec3(123.456,.567,.37)),
				 noise(x+vec3(.11,47.43,19.17)),
				 noise(x) );
}

// Function 148
vec4 noise(float p, float lod){return texture(iChannel0,vec2(p/iChannelResolution[0].x,.0),lod);}

// Function 149
float perlin_noise(vec2 p)
{
	vec2 pi = floor(p);
    vec2 pf = p-pi;
    
    vec2 w = pf*pf*(3.-2.*pf);
    
    float f00 = dot(hash22(pi+vec2(.0,.0)),pf-vec2(.0,.0));
    float f01 = dot(hash22(pi+vec2(.0,1.)),pf-vec2(.0,1.));
    float f10 = dot(hash22(pi+vec2(1.0,0.)),pf-vec2(1.0,0.));
    float f11 = dot(hash22(pi+vec2(1.0,1.)),pf-vec2(1.0,1.));
    
    float xm1 = mix(f00,f10,w.x);
    float xm2 = mix(f01,f11,w.x);
    
    return mix(xm1,xm2,w.y);
}

// Function 150
float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
           -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
    dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// Function 151
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 152
float noise( in vec3 x )
{
    #if 1
    
    vec3 i = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (i.xy+vec2(37.0,17.0)*i.z) + f.xy;
	vec2 rg = textureLod( iChannel2, (uv+0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
    
    #else
    
    ivec3 i = ivec3(floor(x));
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	ivec2 uv = i.xy + ivec2(37,17)*i.z;
	vec2 rgA = texelFetch( iChannel3, (uv+ivec2(0,0))&255, 0 ).yx;
    vec2 rgB = texelFetch( iChannel3, (uv+ivec2(1,0))&255, 0 ).yx;
    vec2 rgC = texelFetch( iChannel3, (uv+ivec2(0,1))&255, 0 ).yx;
    vec2 rgD = texelFetch( iChannel3, (uv+ivec2(1,1))&255, 0 ).yx;
    vec2 rg = mix( mix( rgA, rgB, f.x ),
                   mix( rgC, rgD, f.x ), f.y );
    return mix( rg.x, rg.y, f.z );
    
    #endif
}

// Function 153
float noise(vec3 p){
    vec3 ip=floor(p),s=vec3(7,157,113);p-=ip;
    vec4 h=vec4(0,s.yz,s.y+s.z)+dot(ip,s);
    p=p*p*(3.-2.*p);
    h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
    h.xy=mix(h.xz,h.yw,p.y);return mix(h.x,h.y,p.z);
}

// Function 154
float noise(vec2 x) { vec2 i = floor(x); vec2 f = fract(x); float a = hash(i); float b = hash(i + vec2(1.0, 0.0)); float c = hash(i + vec2(0.0, 1.0)); float d = hash(i + vec2(1.0, 1.0)); vec2 u = f * f * (3.0 - 2.0 * f); return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y; }

// Function 155
vec2 noiseStackUV(vec3 pos,int octaves,float falloff,float diff){
	float displaceA = noiseStack(pos,octaves,falloff);
	float displaceB = noiseStack(pos+vec3(3984.293,423.21,5235.19),octaves,falloff);
	return vec2(displaceA,displaceB);
}

// Function 156
float perlin(vec3 V) {
    float total = 0.0;
    for(int i=2;i<OCTAVES+2;i++) {
        total += (1.0/float(i))*smoothNoise(V);
        V*=2.0+(float(i)/100.0);
    }
    return total;
}

// Function 157
float snoise(float v)
{
    return snoise(vec2(v));
}

// Function 158
vec3 BitangentNoise4D(vec4 p)
{
	const vec4 F4 = vec4( 0.309016994374947451 );
	const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
	                      0.276393202250021,  // 2 * G4
	                      0.414589803375032,  // 3 * G4
	                     -0.447213595499958 ); // -1 + 4 * G4

	// First corner
	vec4 i  = floor(p + dot(p, F4) );
	vec4 x0 = p -   i + dot(i, C.xxxx);

	// Other corners

	// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
	vec4 i0;
	vec3 isX = step( x0.yzw, x0.xxx );
	vec3 isYZ = step( x0.zww, x0.yyz );
	// i0.x = dot( isX, vec3( 1.0 ) );
	i0.x = isX.x + isX.y + isX.z;
	i0.yzw = 1.0 - isX;
	// i0.y += dot( isYZ.xy, vec2( 1.0 ) );
	i0.y += isYZ.x + isYZ.y;
	i0.zw += 1.0 - isYZ.xy;
	i0.z += isYZ.z;
	i0.w += 1.0 - isYZ.z;

	// i0 now contains the unique values 0,1,2,3 in each channel
	vec4 i3 = clamp( i0, 0.0, 1.0 );
	vec4 i2 = clamp( i0 - 1.0, 0.0, 1.0 );
	vec4 i1 = clamp( i0 - 2.0, 0.0, 1.0 );

	// x0 = x0 - 0.0 + 0.0 * C.xxxx
	// x1 = x0 - i1  + 1.0 * C.xxxx
	// x2 = x0 - i2  + 2.0 * C.xxxx
	// x3 = x0 - i3  + 3.0 * C.xxxx
	// x4 = x0 - 1.0 + 4.0 * C.xxxx
	vec4 x1 = x0 - i1 + C.xxxx;
	vec4 x2 = x0 - i2 + C.yyyy;
	vec4 x3 = x0 - i3 + C.zzzz;
	vec4 x4 = x0 + C.wwww;

	i = i + 32768.5;
	uvec2 hash0 = _pcg4d16(uvec4(i));
	uvec2 hash1 = _pcg4d16(uvec4(i + i1));
	uvec2 hash2 = _pcg4d16(uvec4(i + i2));
	uvec2 hash3 = _pcg4d16(uvec4(i + i3));
	uvec2 hash4 = _pcg4d16(uvec4(i + 1.0 ));

	vec4 p00 = _gradient4d(hash0.x); vec4 p01 = _gradient4d(hash0.y);
	vec4 p10 = _gradient4d(hash1.x); vec4 p11 = _gradient4d(hash1.y);
	vec4 p20 = _gradient4d(hash2.x); vec4 p21 = _gradient4d(hash2.y);
	vec4 p30 = _gradient4d(hash3.x); vec4 p31 = _gradient4d(hash3.y);
	vec4 p40 = _gradient4d(hash4.x); vec4 p41 = _gradient4d(hash4.y);

	// Calculate noise gradients.
	vec3 m0 = clamp(0.6 - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0, 1.0);
	vec2 m1 = clamp(0.6 - vec2(dot(x3, x3), dot(x4, x4)             ), 0.0, 1.0);
	vec3 m02 = m0 * m0; vec3 m03 = m02 * m0;
	vec2 m12 = m1 * m1; vec2 m13 = m12 * m1;

	vec3 temp0 = m02 * vec3(dot(p00, x0), dot(p10, x1), dot(p20, x2));
	vec2 temp1 = m12 * vec2(dot(p30, x3), dot(p40, x4));
	vec4 grad0 = -6.0 * (temp0.x * x0 + temp0.y * x1 + temp0.z * x2 + temp1.x * x3 + temp1.y * x4);
	grad0 += m03.x * p00 + m03.y * p10 + m03.z * p20 + m13.x * p30 + m13.y * p40;

	temp0 = m02 * vec3(dot(p01, x0), dot(p11, x1), dot(p21, x2));
	temp1 = m12 * vec2(dot(p31, x3), dot(p41, x4));
	vec4 grad1 = -6.0 * (temp0.x * x0 + temp0.y * x1 + temp0.z * x2 + temp1.x * x3 + temp1.y * x4);
	grad1 += m03.x * p01 + m03.y * p11 + m03.z * p21 + m13.x * p31 + m13.y * p41;

	// The cross products of two gradients is divergence free.
	return cross(grad0.xyz, grad1.xyz) * 81.0;
}

// Function 159
vec4 denoise0( in sampler2D tex, in vec2 pix )
{
    vec2 uv = pix / iResolution.xy;
	return( texture( tex, uv ) );   
}

// Function 160
float noise(in vec3 x){
    vec3 i = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (i.xy+vec2(37.0,17.0)*i.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 161
vec4 texturenoise( vec3 r )
{
    vec3 uvw = r / iChannelResolution[3];
    return texture( iChannel3, uvw ) * 2. - 1.;
}

// Function 162
float noise(vec2 p)
{
    p = floor(p*200.0);
	return rand(p);
}

// Function 163
float noise ( vec2 x)
{
	return iqnoise(x, 0.0, 1.0);
}

// Function 164
float NoiseWrap(in vec2 p)
{
	vec2 f = fract(p);
    p = floor(p);
    f = f*f*(3.0-2.0*f);
    float res = mix(mix(HashWrap(p),
						HashWrap(p + vec2(1.0, 0.0)), f.x),
					mix(HashWrap(p + vec2(0.0, 1.0)),
						HashWrap(p + vec2(1.0, 1.0)), f.x), f.y);
    return res;
}

// Function 165
float voronoi_column_ao( vec2 x, vec4 mc )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    vec2 mr = mc.xy - x;
    vec2 mg = mc.zw - n;
    
    float mh = hash12( n + mg );
    
    // Set center of search based on which half of the cell we are in,
    // since 4x4 is not centered around "n".
    mg = step(.5,f) - 1.;

    float mao = 0.;
    for( int j=-1; j<=2; j++ )
    for( int i=-1; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
        vec2 o = hash22( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>eps ) // skip the same cell
        {
            float d = dot( 0.5*(mr+r), normalize(r-mr) );
            // Get height of the column
            float h = hash12( n + g );
            float ao = clamp((h - mh)*2.,0.,.5)*max(0., 1. - d*4.);
            mao = max(mao, ao);
        }
    }

    return mao;
}

// Function 166
float myNoise(float value)
{
 	value += cos(value * 100.0);
 	value += cos(value * 20.0);
    value = texture(iChannel2, vec2(1.0, 1.0 / 256.0) * value).x - 0.5;
    return value * 2.0;
}

// Function 167
float noise( vec2 p ) {
	return .5 + ( sin( p.x ) + sin( p.y  ) ) / 4.;
}

// Function 168
float Perlin(vec2 uv, vec2 fragCoord)
{
    // Find corner coordinates
    vec4 lwrUpr = vec4(floor(uv), ceil(uv)); 
    mat4x2 crnrs = mat4x2(lwrUpr.xw, lwrUpr.zw,
                          lwrUpr.xy, lwrUpr.zy);
    
    // Generate gradients at each corner
    vec2 mUV = iMouse.zw / iResolution.xy;
    vec2 scTime = abs(vec2(sin(iTime) + mUV.x, cos(iTime) + mUV.y)) + vec2(0.75);
    mat4x2 dirs = mat4x2(hash22(uvec2(floatBitsToUint(crnrs[0]))) * scTime,
                         hash22(uvec2(floatBitsToUint(crnrs[1]))) * scTime,
                         hash22(uvec2(floatBitsToUint(crnrs[2]))) * scTime,
                         hash22(uvec2(floatBitsToUint(crnrs[3]))) * scTime);
    
    // Shift gradients into [-1...0...1]
    dirs *= 2.0f;
    dirs -= mat4x2(vec2(1.0f), vec2(1.0f), 
                   vec2(1.0f), vec2(1.0f));
    
    // Normalize
    dirs[0] = normalize(dirs[0]);
    dirs[1] = normalize(dirs[1]);
    dirs[2] = normalize(dirs[2]);
    dirs[3] = normalize(dirs[3]);
    
    // Find per-cell pixel offset
    vec2 offs = mod(uv, 1.0f);
    
    // Compute gradient weights for each corner; take each offset relative
    // to corners on the square in-line
    vec4 values = vec4(dot(dirs[0], (offs - vec2(0.0f, 1.0f))),
                       dot(dirs[1], (offs - vec2(1.0f))),
                       dot(dirs[2], (offs - vec2(0.0f))),
                       dot(dirs[3], (offs - vec2(1.0f, 0.0f))));
    
    // Return smoothly interpolated values
    vec2 softXY = soften(offs);
    return mix(mix(values.z, 
                   values.w, softXY.x),
               mix(values.x, 
                   values.y, softXY.x),
               softXY.y);
}

// Function 169
vec2 lpnoise(float t, float fq//FM lp-noise for reverb
){t*=fq
 ;float f=fract(t)
 ;vec2 r=floor(t-f+vec2(0,1))/fq
 ;f=smoothstep(0.,1.,f)//;f=step(f,0.)//harder and faster
 ;return mix(noise2(r.x),noise2(r.y),f);}

// Function 170
float mScaleNoise(vec3 texc)
{
    float d=0.0;
    d+=snoise3DS(texc);
    d+=snoise3DS(texc*2.553)*0.5;
    d+=snoise3DS(texc*5.154)*0.25;
    //d+=snoise3DS(texc*400.45)*0.009;
    d+=snoise3DS(texc*400.45*vec3(0.1,0.1,1.0))*0.009;
    //d+=snoise3DS(texc*900.45*vec3(0.1,1.0,0.1))*0.005;
    //d+=snoise3DS(texc*900.45*vec3(1.0,0.1,0.1))*0.005;
    d*=0.5;
    return d;
}

// Function 171
vec4 noiseGra13dx(in vec3 x)
{vec3 p=floor(x),w=fract(x)
#if 1
,u=w*w*w*(w*(w*6.-15.)+10.),v=30.*w*w*(w*(w-2.)+1.)//quintic hermite
#else
,u=w*w*(3.-2.*w),v=6.*w*(1.-w)//cubic hermite
#endif    
//gradients
,G=hash(p+vec3(0)),F=hash(p+vec3(1))
;mat3 D=hash(mat3(p,p,p)+mat3(1)),E=hash(mat3(p,p,p)+1.-mat3(1));
//projections 
;vec3 d=dots(D,w,mat3(1)),e=dots(E,w,1.-mat3(1));
//interpolations
;float g=dot(G,w),f=dot(F,w-vec3(1));
;vec3 h=u.yzx*(g-d.xyx-d.yzz+e.zxy)+d-g,U=u*h,a=d-e
;mat3 S=D-mat3(G,G,G),W=D-E
;a.x=(a.x+a.y+a.z)+f-g;
;float b=u.x*u.y*u.z;
;return vec4(g+U.x+U.y+U.z+a.x*b// value
,G*(1.-b)+b*(W[0]+W[1]+W[2]+F)//https://www.shadertoy.com/view/llByD1
+u.x*(S[0]+u.y*(G-D[0]-D[1]+E[2]))  // derivatives
+u.y*(S[1]+u.z*(G-D[1]-D[2]+E[0]))
+u.z*(S[2]+u.x*(G-D[0]-D[2]+E[1]))
+v*(u.zxy*(g-d.xxy-d.zyz+e.yzx)+h+u.yzx*u.zxy*a.x));}

// Function 172
vec4 noise(vec4 p){float m = mod(p.w,1.0);float s = p.w-m; float sprev = s-1.0;if (mod(s,2.0)==1.0) { s--; sprev++; m = 1.0-m; };return mix(noise(p.xyz+noise(sprev).wyx*3531.123420),	noise(p.xyz+noise(s).wyx*4521.5314),	m);}

// Function 173
float perlin(vec2 uv, int seed) {
    vec2 fl = floor(uv);
    vec2 fr = fract(uv);
    vec2 ij = vec2(0, 1);
    float a = dot(fr - ij.xx, grad(fl + ij.xx, seed));
    float b = dot(fr - ij.xy, grad(fl + ij.xy, seed));
    float c = dot(fr - ij.yx, grad(fl + ij.yx, seed));
    float d = dot(fr - ij.yy, grad(fl + ij.yy, seed));
	return lerp(
    	lerp(a, b, fr.y), lerp(c, d, fr.y), fr.x
    );
}

// Function 174
float Noise2d( in vec2 x )
{
    float xhash = cos( x.x * 37.0 );
    float yhash = cos( x.y * 57.0 );
    return fract( 415.92653 * ( xhash + yhash ) );
}

// Function 175
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    
    float n = p.x + 317.0*p.y + 157.0*p.z;
    
    float a = hash1(n+0.0);
    float b = hash1(n+1.0);
    float c = hash1(n+317.0);
    float d = hash1(n+318.0);
    float e = hash1(n+157.0);
	float f = hash1(n+158.0);
    float g = hash1(n+474.0);
    float h = hash1(n+475.0);

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return -1.0+2.0*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z);
}

// Function 176
float noise( vec3 p )
{
    vec3 i = floor( p ),
         f = fract( p ),
	     u = f*f*(3.-2.*f);
 
    return 2.*mix(
              mix( mix( dot( hash( i + vec3(0,0,0) ), f - vec3(0,0,0) ), 
                        dot( hash( i + vec3(1,0,0) ), f - vec3(1,0,0) ), u.x),
                   mix( dot( hash( i + vec3(0,1,0) ), f - vec3(0,1,0) ), 
                        dot( hash( i + vec3(1,1,0) ), f - vec3(1,1,0) ), u.x), u.y),
              mix( mix( dot( hash( i + vec3(0,0,1) ), f - vec3(0,0,1) ), 
                        dot( hash( i + vec3(1,0,1) ), f - vec3(1,0,1) ), u.x),
                   mix( dot( hash( i + vec3(0,1,1) ), f - vec3(0,1,1) ), 
                        dot( hash( i + vec3(1,1,1) ), f - vec3(1,1,1) ), u.x), u.y), u.z);
}

// Function 177
vec3 Voronoi( vec3 pos )
{
	vec3 d[8];
	d[0] = vec3(0,0,0);
	d[1] = vec3(1,0,0);
	d[2] = vec3(0,1,0);
	d[3] = vec3(1,1,0);
	d[4] = vec3(0,0,1);
	d[5] = vec3(1,0,1);
	d[6] = vec3(0,1,1);
	d[7] = vec3(1,1,1);
	
	const float maxDisplacement = .7;//.518; //tweak this to hide grid artefacts
	
    vec3 pf = floor(pos);

    const float phi = 1.61803398875;

	float closest = 12.0;
	vec3 result;
	for ( int i=0; i < 8; i++ )
	{
        vec3 v = (pf+d[i]);
		vec3 r = fract(phi*v.yzx+17.*fract(v.zxy*phi)+v*v*.03);//Noise(ivec3(floor(pos+d[i])));
		vec3 p = d[i] + maxDisplacement*(r.xyz-.5);
		p -= fract(pos);
		float lsq = dot(p,p);
		if ( lsq < closest )
		{
			closest = lsq;
			result = r;
		}
	}
	return fract(result.xyz);//+result.www); // random colour
}

// Function 178
float simplex3D(vec3 p)
{
	
	float f3 = 1.0/3.0;
	float s = (p.x+p.y+p.z)*f3;
	int i = int(floor(p.x+s));
	int j = int(floor(p.y+s));
	int k = int(floor(p.z+s));
	
	float g3 = 1.0/6.0;
	float t = float((i+j+k))*g3;
	float x0 = float(i)-t;
	float y0 = float(j)-t;
	float z0 = float(k)-t;
	x0 = p.x-x0;
	y0 = p.y-y0;
	z0 = p.z-z0;
	
	int i1,j1,k1;
	int i2,j2,k2;
	
	if(x0>=y0)
	{
		if(y0>=z0){ i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; } // X Y Z order
		else if(x0>=z0){ i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; } // X Z Y order
		else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }  // Z X Z order
	}
	else 
	{ 
		if(y0<z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; } // Z Y X order
		else if(x0<z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; } // Y Z X order
		else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; } // Y X Z order
	}
	
	float x1 = x0 - float(i1) + g3; 
	float y1 = y0 - float(j1) + g3;
	float z1 = z0 - float(k1) + g3;
	float x2 = x0 - float(i2) + 2.0*g3; 
	float y2 = y0 - float(j2) + 2.0*g3;
	float z2 = z0 - float(k2) + 2.0*g3;
	float x3 = x0 - 1.0 + 3.0*g3; 
	float y3 = y0 - 1.0 + 3.0*g3;
	float z3 = z0 - 1.0 + 3.0*g3;	
				 
	vec3 ijk0 = vec3(i,j,k);
	vec3 ijk1 = vec3(i+i1,j+j1,k+k1);	
	vec3 ijk2 = vec3(i+i2,j+j2,k+k2);
	vec3 ijk3 = vec3(i+1,j+1,k+1);	
            
	vec3 gr0 = normalize(vec3(noise3D(ijk0),noise3D(ijk0*2.01),noise3D(ijk0*2.02)));
	vec3 gr1 = normalize(vec3(noise3D(ijk1),noise3D(ijk1*2.01),noise3D(ijk1*2.02)));
	vec3 gr2 = normalize(vec3(noise3D(ijk2),noise3D(ijk2*2.01),noise3D(ijk2*2.02)));
	vec3 gr3 = normalize(vec3(noise3D(ijk3),noise3D(ijk3*2.01),noise3D(ijk3*2.02)));
	
	float n0 = 0.0;
	float n1 = 0.0;
	float n2 = 0.0;
	float n3 = 0.0;

	float t0 = 0.5 - x0*x0 - y0*y0 - z0*z0;
	if(t0>=0.0)
	{
		t0*=t0;
		n0 = t0 * t0 * dot(gr0, vec3(x0, y0, z0));
	}
	float t1 = 0.5 - x1*x1 - y1*y1 - z1*z1;
	if(t1>=0.0)
	{
		t1*=t1;
		n1 = t1 * t1 * dot(gr1, vec3(x1, y1, z1));
	}
	float t2 = 0.5 - x2*x2 - y2*y2 - z2*z2;
	if(t2>=0.0)
	{
		t2 *= t2;
		n2 = t2 * t2 * dot(gr2, vec3(x2, y2, z2));
	}
	float t3 = 0.5 - x3*x3 - y3*y3 - z3*z3;
	if(t3>=0.0)
	{
		t3 *= t3;
		n3 = t3 * t3 * dot(gr3, vec3(x3, y3, z3));
	}
	return 96.0*(n0+n1+n2+n3);
	
}

// Function 179
float perlin( in float x, in float seed ) {
    x += hash11(seed);
    float a = floor(x);
    float b = a + 1.0;
    float f = fract(x);
    a = hash12(vec2(seed, a));
    b = hash12(vec2(seed, b));
    f = f*f*(3.0-2.0*f);
    return mix(a, b, f)*2.0-1.0;
}

// Function 180
float PerlinNoise2D(vec2 p)
{
    float sum = 0.0;
    float frequency =0.0;
    float amplitude = 0.0;
    for(int i=3;i<9;i++)
    {
        frequency = pow(2.0,float(i));
        amplitude = pow(0.6,float(i));
        sum = sum + InterpolationNoise(p*frequency)*amplitude;
    }
    
    return sum;
}

// Function 181
vec2 voronoiPointFromRoot(vec2 root, float deg)
{
  	vec2 point = hash2_2(root) - 0.5;
    float s = sin(deg);
    float c = cos(deg);
    point = mat2x2(s, c, -c, s) * point;
    point += root + 0.5;
    return point;
}

// Function 182
float textureNoise(vec3 uv)
{
	float c = (linearRand(uv * 1.0) * 32.0 +
			   linearRand(uv * 2.0) * 16.0 + 
			   linearRand(uv * 4.0) * 8.0 + 
			   linearRand(uv * 8.0) * 4.0) / 32.0;
	return c * 0.5 + 0.5;
}

// Function 183
float Voronoi(in vec2 p){
    
	vec2 g = floor(p), o; p -= g;
	
	vec3 d = vec3(1); // 1.4, etc.
    
    float r = 0.;
    
	for(int y = -1; y <= 1; y++){
		for(int x = -1; x <= 1; x++){
            
			o = vec2(x, y);
            o += hash22(g + o) - p;
            
			r = dot(o, o);
            
            // 1st, 2nd and 3rd nearest squared distances.
            d.z = max(d.x, max(d.y, min(d.z, r))); // 3rd.
            d.y = max(d.x, min(d.y, r)); // 2nd.
            d.x = min(d.x, r); // Closest.
                       
		}
	}
    
	d = sqrt(d); // Squared distance to distance.
    
    // Fabrice's formula.
    return min(2./(1./max(d.y - d.x, .001) + 1./max(d.z - d.x, .001)), 1.);
    // Dr2's variation - See "Voronoi Of The Week": https://www.shadertoy.com/view/lsjBz1
    //return min(smin(d.z, d.y, .2) - d.x, 1.);
    
}

// Function 184
float fbmNoise(vec2 p)
{
    float rz = 0.;
    float amp = 2.;
    for (int i = 0; i < 6; i++)
    {
		rz += orbitNoise(p)/amp;
        amp *= 2.;
        p *= 2.1;
        p += 12.5;
    }
    return rz;
}

// Function 185
float noise( in vec2 x ) {
    vec2 p = floor(x);
    vec2 f = fract(x);

	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);

	return -1.0 + 2.0*textureLod( iChannel0, (uv+0.5)/256.0, 0.0 ).x;
}

// Function 186
float remap_noise_tri_erp( const float v )
{
    float r2 = 0.5 * v;
    float f1 = sqrt( r2 );
    float f2 = 1.0 - sqrt( r2 - 0.25 );    
    return (v < 0.5) ? f1 : f2;
}

// Function 187
float meuNoise(float p){
    float ts[5];// = {meuHash(p-2.),meuHash(p-1.),meuHash(p),meuHash(p+1.),meuHash(p+2.)};
   ts[0] = meuHash(p-2.);
  ts[1] = meuHash(p-1.);
 ts[2] = meuHash(p-0.);
ts[3] = meuHash(p+1.);
ts[4] = meuHash(p+2.);
    
    return mix(mix(ts[0],ts[1],0.9),mix(ts[3],ts[4],0.1),0.5);
        
       
    
}

// Function 188
vec3  simplexRotD(vec2 u       ){return simplexRotD(u  ,0.);}

// Function 189
float noise2d(vec2 p){
    vec2 ip = floor(p);
    vec2 u = fract(p);
    u = u*u*(3.0-2.0*u);

    float res = mix(
        mix(random2d(ip),random2d(ip+vec2(1.0,0.0)),u.x),
        mix(random2d(ip+vec2(0.0,1.0)),random2d(ip+vec2(1.0,1.0)),u.x),u.y);
    return res*res;
}

// Function 190
float fnoise( vec3 p)
{
    mat3 rot = rotationMatrix( normalize(vec3(0.0,0.0, 1.0)), 0.5*iTime);
    mat3 rot2 = rotationMatrix( normalize(vec3(0.0,0.0, 1.0)), 0.3*iTime);
    float sum = 0.0;
    
    vec3 r = rot*p;
    
    float add = noise(r);
    float msc = add+0.7;
   	msc = clamp(msc, 0.0, 1.0);
    sum += 0.6*add;
    
    p = p*2.0;
    r = rot*p;
    add = noise(r);
 
    add *= msc;
    sum += 0.5*add;
    msc *= add+0.7;
   	msc = clamp(msc, 0.0, 1.0);
    
    p.xy = p.xy*2.0;
    p = rot2 *p;
    add = noise(p);
    add *= msc;
    sum += 0.25*abs(add);
    msc *= add+0.7;
   	msc = clamp(msc, 0.0, 1.0);
 
    p = p*2.0;
  //  p = p*rot;
    add = noise(p);// + vec3(iTime*5.0, 0.0, 0.0));
    add *= msc;
    sum += 0.125*abs(add);
    msc *= add+0.2;
   	msc = clamp(msc, 0.0, 1.0);

    p = p*2.0;
  //  p = p*rot;
    add = noise(p);
    add *= msc;
    sum += 0.0625*abs(add);
    //msc *= add+0.7;
   	//msc = clamp(msc, 0.0, 1.0);

    
    return sum*0.516129; // return msc as detail measure?
}

// Function 191
float SmoothNoise( float t )
{
	float noiset = t * 32.0;
	float tfloor = floor(noiset);
	float ffract = fract(noiset);
	
	float n0 = Hash(tfloor);
	float n1 = Hash(tfloor + 1.0);
	float blend = ffract*ffract*(3.0 - 2.0*ffract);
	return mix(n0, n1, blend) * 2.0 - 1.0;
}

// Function 192
float noiseSDF( in vec3 p )
{
    vec3 i = floor(p);
    vec3 f = fract(p);

    const float G1 = 0.2;
    const float G2 = 0.7;
    
	#define RAD(r) ((r)*(r)*G2)
    #define SPH(i,f,c) length(f-c)-RAD(hash(i+c))
    
    return smin(smin(smin(SPH(i,f,vec3(0,0,0)),
                          SPH(i,f,vec3(0,0,1)),G1),
                     smin(SPH(i,f,vec3(0,1,0)),
                          SPH(i,f,vec3(0,1,1)),G1),G1),
                smin(smin(SPH(i,f,vec3(1,0,0)),
                          SPH(i,f,vec3(1,0,1)),G1),
                     smin(SPH(i,f,vec3(1,1,0)),
                          SPH(i,f,vec3(1,1,1)),G1),G1),G1);
}

// Function 193
vec2 gpuCellNoise3D(const in vec3 xyz)
{
	int xi = int(floor(xyz.x));
	int yi = int(floor(xyz.y));
	int zi = int(floor(xyz.z));

	float xf = xyz.x - float(xi);
	float yf = xyz.y - float(yi);
	float zf = xyz.z - float(zi);

	float dist1 = 9999999.0;
	float dist2 = 9999999.0;
	vec3 cell;

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cell = gpuGetCell3D(xi + x, yi + y, zi + z).xyz;
				cell.x += (float(x) - xf);
				cell.y += (float(y) - yf);
				cell.z += (float(z) - zf);
				float dist = dot(cell, cell);
				if (dist < dist1)
				{
					dist2 = dist1;
					dist1 = dist;
				}
				else if (dist < dist2)
				{
					dist2 = dist;
				}
			}
		}
	}
	return vec2(sqrt(dist1), sqrt(dist2));
}

// Function 194
vec4 smartDeNoise(sampler2D tex, vec2 uv, float sigma, float kSigma, float threshold)
{
    float radius = round(kSigma*sigma);
    float radQ = radius * radius;
    
    float invSigmaQx2 = .5 / (sigma * sigma);      // 1.0 / (sigma^2 * 2.0)
    float invSigmaQx2PI = INV_PI * invSigmaQx2;    // 1.0 / (sqrt(PI) * sigma)
    
    float invThresholdSqx2 = .5 / (threshold * threshold);     // 1.0 / (sigma^2 * 2.0)
    float invThresholdSqrt2PI = INV_SQRT_OF_2PI / threshold;   // 1.0 / (sqrt(2*PI) * sigma)
    
    vec4 centrPx = texture(tex,uv);
    
    float zBuff = 0.0;
    vec4 aBuff = vec4(0.0);
    vec2 size = vec2(textureSize(tex, 0));
    
    for(float x=-radius; x <= radius; x++) {
        float pt = sqrt(radQ-x*x);  // pt = yRadius: have circular trend
        for(float y=-pt; y <= pt; y++) {
            vec2 d = vec2(x,y);

            float blurFactor = exp( -dot(d , d) * invSigmaQx2 ) * invSigmaQx2PI; 
            
            vec4 walkPx =  texture(tex,uv+d/size);

            vec4 dC = walkPx-centrPx;
            float deltaFactor = exp( -dot(dC, dC) * invThresholdSqx2) * invThresholdSqrt2PI * blurFactor;
                                 
            zBuff += deltaFactor;
            aBuff += deltaFactor*walkPx;
        }
    }
    return aBuff/zBuff;
}

// Function 195
void mfnoise(in vec2 x, in float d, in float b, in float e, out float n)
{
    n = 0.;
    float a = 1., nf = 0., buf;
    for(float f = d; f<b; f *= 2.)
    {
        lfnoise(f*x, buf);
        n += a*buf;
        a *= e;
        nf += 1.;
    }
    n *= (1.-e)/(1.-pow(e, nf));
}

// Function 196
float snoise(vec3 v){ 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //  x0 = x0 - 0. + 0.0 * C 
  vec3 x1 = x0 - i1 + 1.0 * C.xxx;
  vec3 x2 = x0 - i2 + 2.0 * C.xxx;
  vec3 x3 = x0 - 1. + 3.0 * C.xxx;

// Permutations
  i = mod(i, 289.0 ); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients
// ( N*N points uniformly over a square, mapped onto an octahedron.)
  float n_ = 1.0/7.0; // N=7
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

// Function 197
float noise3f( in vec3 p )
{
	ivec3 ip = ivec3(floor(p));
    vec3  fp = fract(p);
	vec3 u = fp*fp*(3.0-2.0*fp);

    int n = ip.x + ip.y*57 + ip.z*113;

	float res = mix(mix(mix(fhash(n+(0+57*0+113*0)),
                            fhash(n+(1+57*0+113*0)),u.x),
                        mix(fhash(n+(0+57*1+113*0)),
                            fhash(n+(1+57*1+113*0)),u.x),u.y),
                    mix(mix(fhash(n+(0+57*0+113*1)),
                            fhash(n+(1+57*0+113*1)),u.x),
                        mix(fhash(n+(0+57*1+113*1)),
                            fhash(n+(1+57*1+113*1)),u.x),u.y),u.z);

    return 1.0 - res*(1.0/1073741824.0);
}

// Function 198
float Noisefv2 (vec2 p)
{
  vec2 i = floor (p);
  vec2 f = fract (p);
  f = f * f * (3. - 2. * f);
  vec4 t = Hashv4f (dot (i, cHashA3.xy));
  return mix (mix (t.x, t.y, f.x), mix (t.z, t.w, f.x), f.y);
}

// Function 199
float bnoise( in float x )
{
    // setup    
    float i = floor(x);
    float f = fract(x);
    float s = sign(fract(x/2.0)-0.5);
    
    // use some hash to create a random value k in [0..1] from i
    //float k = hash(uint(i));
  	//float k = 0.5+0.5*sin(i);
  	float k = fract(i*.1731);

    // quartic polynomial
    return s*f*(f-1.0)*((16.0*k-4.0)*f*(f-1.0)-1.0);
}

// Function 200
vec2 lpnoise(float t, float fq)
{
    t *= fq;

    float tt = fract(t);
    float tn = t - tt;
    tt = smoothstep(0.0, 1.0, tt);

    vec2 n0 = noise(floor(tn + 0.0) / fq);
    vec2 n1 = noise(floor(tn + 1.0) / fq);

    return mix(n0, n1, tt);
}

// Function 201
vec4 ThornVoronoi( vec3 p)
{
    
    vec2 f = fract(p.xz);
    p.xz = floor(p.xz);
	float d = 1.0e10;
    vec3 id = vec3(0.0);
    
	for (int xo = -1; xo <= 1; xo++)
	{
		for (int yo = -1; yo <= 1; yo++)
		{
            vec2 g = vec2(xo, yo);
            vec2 n = texture(iChannel3,(p.xz + g+.5)/256.0, -100.0).xy;
            n = n*n*(3.0-2.0*n);
            
			vec2 tp = g + .5 + sin(p.y + 1.2831 * (n * gTime*.5)) - f;
            float d2 = dot(tp, tp);
			if (d2 < d)
            {
                // 'id' is the colour code for each thorn
                d = d2;

                id = vec3(tp.x, p.y, tp.y);
            }
		}
	}

    return vec4(id, 1.35-pow(d, .17));
}

// Function 202
float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p); f = f*f*(3.-2.*f); // smoothstep

    float v= mix( mix(hash21(i+vec2(0,0)),hash21(i+vec2(1,0)),f.x),
                  mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x), f.y);
	return   MOD==0 ? v
	       : MOD==1 ? 2.*v-1.
           : MOD==2 ? abs(2.*v-1.)
                    : 1.-abs(2.*v-1.);
}

// Function 203
float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  // First corner
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  // Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy; // -1.0+3.0*C.x = -0.5 = -D.y

  // Permutations
  i = mod289(i);
  vec4 p = permute(
      permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x +
      vec4(0.0, i1.x, i2.x, 1.0));

  // Gradients: 7x7 points over a square, mapped onto an octahedron.
  // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z); //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_); // mod(j,N)

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  // vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  // vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  // Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  // Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// Function 204
float perlinNoise(vec2 pos, vec2 scale, float rotation, float seed) 
{
    vec2 sinCos = vec2(sin(rotation), cos(rotation));
    return perlinNoise(pos, scale, mat2(sinCos.y, sinCos.x, sinCos.x, sinCos.y), seed);
}

// Function 205
float noise(vec2 p)
{
    vec2 i = floor(p), f = fract(p); 
	f *= f*f*(3.-2.*f);
    
    vec2 c = vec2(0,1);
    
    return mix(mix(hash(i + c.xx), 
                   hash(i + c.yx), f.x),
               mix(hash(i + c.xy), 
                   hash(i + c.yy), f.x), f.y);
}

// Function 206
float bnoise (vec2 uv)
{
    return texture(iChannel3, uv).x * 2. - 1.;
}

// Function 207
vec4 noise(in vec2 uv ) {
    return
        texture(iChannel0, uv * 0.125)  *  0.50    +
        texture(iChannel0, uv * 0.25)   *  0.25    +
        texture(iChannel0, uv * 0.5)    *  0.125   +
        texture(iChannel0, uv * 1.0)    *  0.0625  +
        texture(iChannel0, uv * 2.0)    *  0.03125 +
        texture(iChannel0, uv * 4.0)    *  0.03125;
}

// Function 208
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel3, (uv+ 0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 209
float noiseW(in vec2 p) {
    return texture(iChannel0, (p - 16.0) / 256.0, -100.0).x;
}

// Function 210
float achnoise(float x){
	float p = floor(x);
	float fr = fract(x);
	float L = p;
	float R = p + 1.0;
	
	float Lo = oct(L);
	float Ro = oct(R);
	
	return mix(Lo, Ro, fr);
}

// Function 211
float noisePattern(vec3 pos)
{
    return noise(normalize(pos)*2.5);
}

// Function 212
float snoise(vec2 v){
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                     + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
                            dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g = vec3(0.0);
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Function 213
uint noise_prng_rand(inout noise_prng this_)
{
    return this_.x_ *= 3039177861u;
}

// Function 214
float Noise(in vec3 p)
{
    vec3 i = floor(p);
	vec3 f = fract(p); 
	f *= f * (3.0-2.0*f);

    return mix(
		mix(mix(Hash(i + vec3(0.,0.,0.)), Hash(i + vec3(1.,0.,0.)),f.x),
			mix(Hash(i + vec3(0.,1.,0.)), Hash(i + vec3(1.,1.,0.)),f.x),
			f.y),
		mix(mix(Hash(i + vec3(0.,0.,1.)), Hash(i + vec3(1.,0.,1.)),f.x),
			mix(Hash(i + vec3(0.,1.,1.)), Hash(i + vec3(1.,1.,1.)),f.x),
			f.y),
		f.z);
}

// Function 215
vec3 TerrainNoiseHQ(vec2 p) { return TerrainNoise(p, 15); }

// Function 216
float noise( vec2 p )
{
    vec2 i = floor( p ),
         f = fract( p ),
	     u = f*f*(3.-2.*f);

#define P(x,y) dot( hash( i + vec2(x,y) ), f - vec2(x,y) )
    return mix( mix( P(0,0), P(1,0), u.x),
                mix( P(0,1), P(1,1), u.x), u.y);
}

// Function 217
float noise( in vec3 x )
{
  vec3 p  = floor(x);
  vec3 f  = smoothstep(0.0, 1.0, fract(x));
  float n = p.x + p.y*57.0 + 113.0*p.z;

  return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
    mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
    mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

// Function 218
and noises (messy namespace function stuff, instrument lib specific, oh well:
float noise1(float t, float seed){return 1.0+sin(t * 0.02 * seed+sin(t * 0.05 * seed)) * 0.25;}

// Function 219
vec3 noise1v( float n )
{   
    
    vec2 coords = vec2(mod(n,NOISE_DIMENSION)/NOISE_DIMENSION, 
                       floor(n/NOISE_DIMENSION)/NOISE_DIMENSION);
    
    return texture(iChannel0, coords).rgb;
}

// Function 220
float snoise(vec3 v){ 
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //  x0 = x0 - 0. + 0.0 * C 
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1. + 3.0 * C.xxx;

    // Permutations
    i = mod(i, 289.0 ); 
    vec4 p = permute( permute( permute( 
        i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                              + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
                     + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0/7.0; // N=7
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                 dot(p2,x2), dot(p3,x3) ) );
}

// Function 221
float noise( in int i, in int j )
{
  vec2 n = vec2(i,j); 
  vec2 p = floor(n);
  vec2 f = fract(n);
  f = f*f*(3.0-2.0*f);
  vec2 uv = (p+vec2(37.0,17.0)) + f;
  vec2 rg = hash( uv/256.0 ).yx;
  return 0.5*mix( rg.x, rg.y, 0.5 );
}

// Function 222
float noise1(float p)
{
    float fl = floor(p);
    float fc = fract(p);
    return mix(rand(fl), rand(fl + 1.0), fc);
}

// Function 223
float textureNoise(vec3 uv)
{
	float c = (linearRand(uv * 1.0) * 32.0 +
			   linearRand(uv * 2.0) * 16.0 + 
			   linearRand(uv * 4.0) * 8.0 + 
			   linearRand(uv * 8.0) * 4.0 + 
			   linearRand(uv * 16.0) * 2.0 + 
			   linearRand(uv * 32.0) * 1.0) / 32.0;
	return c * 0.5 + 0.5;
}

// Function 224
float noiseOld( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel1, (uv+0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 225
float pNoise(vec2 p, int res){
	float persistance = .5;
	float n = 0.;
	float normK = 0.;
	float f = 4.;
	float amp = 1.;
	int iCount = 0;
	for (int i = 0; i<50; i++){
		n+=amp*noise(p, f);
		f*=2.;
		normK+=amp;
		amp*=persistance;
		if (iCount == res) break;
		iCount++;
	}
	float nf = n/normK;
	return nf*nf*nf*nf;
}

// Function 226
float noise3( vec3 x ) {
    vec3 p = floor(x),f = fract(x);

    f = f*f*(3.-2.*f);  // or smoothstep     // to make derivative continuous at borders

#define hash3(p)  fract(sin(1e3*dot(p,vec3(1,57,-13.7)))*4375.5453)        // rand
    
    return mix( mix(mix( hash3(p+vec3(0,0,0)), hash3(p+vec3(1,0,0)),f.x),       // triilinear interp
                    mix( hash3(p+vec3(0,1,0)), hash3(p+vec3(1,1,0)),f.x),f.y),
                mix(mix( hash3(p+vec3(0,0,1)), hash3(p+vec3(1,0,1)),f.x),       
                    mix( hash3(p+vec3(0,1,1)), hash3(p+vec3(1,1,1)),f.x),f.y), f.z);
}

// Function 227
float noiseOld( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel1, (uv+0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 228
float noise(vec3 p){//Noise function stolen from Virgil who stole it from Shane who I assume understands this shit, unlike me who is too busy argueing about the poetry of football hooliganism
  vec3 ip=floor(p),s=vec3(7,157,113);
  p-=ip; vec4 h=vec4(0,s.yz,s.y+s.z)+dot(ip,s);
  p=p*p*(3.-2.*p);
  h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
  h.xy=mix(h.xz,h.yw,p.y);
  return mix(h.x,h.y,p.z);
}

// Function 229
float noise3D(in vec3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
	const vec3 s = vec3(7, 157, 113);
	
	vec3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride vector for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    
	p -= ip; // Cell's fractional component.
	
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: fract(sin(x)*largeNumber).
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
	
    // Interpolating along Y.
    h.xy = mix(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return mix(h.x, h.y, p.z); // Range: [0, 1].
	
}

// Function 230
float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);
    float a = hash21(i);
    float b = hash21(i + vec2(1., 0.));
    float c = hash21(i + vec2(0., 1.));
    float d = hash21(i + vec2(1., 1.));
    vec2 u = f * f * (3.-2.*f);
    return mix(a, b, u.x) + (c - a)* u.y * (1. - u.x) + (d - b)* u.x * u.y;
}

// Function 231
float perlin(vec2 p, float dim, float time) {
	vec2 pos = floor(p * dim);
	vec2 posx = pos + vec2(1.0, 0.0);
	vec2 posy = pos + vec2(0.0, 1.0);
	vec2 posxy = pos + vec2(1.0);
	
	float c = rand(pos, dim, time);
	float cx = rand(posx, dim, time);
	float cy = rand(posy, dim, time);
	float cxy = rand(posxy, dim, time);
	
	vec2 d = fract(p * dim);
	d = -0.5 * cos(d * M_PI) + 0.5;
	
	float ccx = mix(c, cx, d.x);
	float cycxy = mix(cy, cxy, d.x);
	float center = mix(ccx, cycxy, d.y);
	
	return center * 2.0 - 1.0;
}

// Function 232
float SpiralNoiseC(vec3 p, float t)
{
    float n = -mod(t * 0.2,-2.); // noise amount
    float iter = 2.0;
    for (int i = 0; i < 8; i++)
    {
        // add sin and cos scaled inverse with the frequency
        n += -abs(sin(p.y*iter) + cos(p.x*iter)) / iter;	// abs for a ridged look
        // rotate by adding perpendicular and scaling down
        p.xy += vec2(p.y, -p.x) * nudge;
        p.xy *= normalizer;
        // rotate on other axis
        p.xz += vec2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        // increase the frequency
        iter *= 1.733733;
    }
    return n;
}

// Function 233
float perlin(vec3 v) {
    vec3 p = floor(v), f = fract(v);
	f = f*f*(3.-2.*f);
	float n = p.x + dot(p.yz,vec2(157,113));
	vec4 s1 = mix(permute(n+vec4(0,157,113,270)),permute(n+vec4(1,158,114,271)),f.x);
	return mix(mix(s1.x,s1.y,f.y), mix(s1.z,s1.w,f.y), f.z);
}

// Function 234
float triNoise3d(in vec3 p)
{
    float z=1.5;
	float rz = 0.;
    vec3 bp = p;
	for (float i=0.; i<=3.; i++ )
	{
        vec3 dg = tri3(bp*2.)*1.;
        p += (dg+time*0.25);

        bp *= 1.8;
		z *= 1.5;
		p *= 1.1;
        p.xz*= m2;
        
        rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
        bp += 0.14;
	}
	return rz;
}

// Function 235
vec2 noise2(float t){return hash2(vec2(t, t * 1.423)) * 2.0-1.0;}

// Function 236
float bluenoise(vec2 uv)
{
    #if defined( ANIMATED )
    uv += 1337.0*fract(iTime);
    #endif
    float v = texture( iChannel1 , (uv + 0.5) / iChannelResolution[1].xy, 0.0).x;
    return v;
}

// Function 237
float noise( vec2 uv, float detail){
	float n = 0.;
    float m = 0.;

    for(float i = 0.; i < detail; i++){
    	float x = pow(2., i);
        float y = 1./x;
        
        n += noise(uv*x+y)*y;
        m += y;
    }
    
    return n/m;
    
}

// Function 238
float InterleavedGradientNoise(vec2 pixel, int frame) 
{
    pixel += (float(frame) * 5.588238f);
    return fract(52.9829189f * fract(0.06711056f*float(pixel.x) + 0.00583715f*float(pixel.y)));  
}

// Function 239
float snoise(vec2 p) {
  	vec2 inter = smoothstep(0., 1., fract(p));
  	float s = mix(noise(sw(p)), noise(se(p)), inter.x);
  	float n = mix(noise(nw(p)), noise(ne(p)), inter.x);
  	return mix(s, n, inter.y);
}

// Function 240
float polarNoise0(vec3 x)
{   
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix( hash(p+vec3(0,0,0)), 
                        hash(p+vec3(1,0,0)),f.x),
                   mix( hash(p+vec3(0,1,0)), 
                        hash(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(p+vec3(0,0,1)), 
                        hash(p+vec3(1,0,1)),f.x),
                   mix( hash(p+vec3(0,1,1)), 
                        hash(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 241
float Noise13(float x, float seed)
{
    float res = Noise1(x, seed);
    res += Noise1(x * 2.0, seed) * 0.5;
    res += Noise1(x * 4.0, seed) * 0.25;
    return res;
}

// Function 242
vec4 noise( in vec2 p ) {
    return texture(iChannel1, p, 0.0);
}

// Function 243
float snoise(in lowp vec2 v) {
  lowp vec2 i = floor((v.x+v.y)*.36602540378443 + v),
      x0 = (i.x+i.y)*.211324865405187 + v - i;
  lowp float s = step(x0.x,x0.y);
  lowp vec2 j = vec2(1.0-s,s),
      x1 = x0 - j + .211324865405187, 
      x3 = x0 - .577350269189626; 
  i = mod(i,289.);
  lowp vec3 p = permute( permute( i.y + vec3(0, j.y, 1 ))+ i.x + vec3(0, j.x, 1 )   ),
       m = max( .5 - vec3(dot(x0,x0), dot(x1,x1), dot(x3,x3)), 0.),
       x = fract(p * .024390243902439) * 2. - 1.,
       h = abs(x) - .5,
      a0 = x - floor(x + .5);
  return .5 + 65. * dot( pow(m,vec3(4.))*(- 0.85373472095314*( a0*a0 + h*h )+1.79284291400159 ), a0 * vec3(x0.x,x1.x,x3.x) + h * vec3(x0.y,x1.y,x3.y));
}

// Function 244
float SpiralNoiseC(vec3 p)
{
    float n = 0.0;	// noise amount
    float iter = 1.0;
    for (int i = 0; i < 7; i++)
    {
        // add sin and cos scaled inverse with the frequency
        n += -abs(sin(p.y*iter) + cos(p.x*iter)) / iter;	// abs for a ridged look
        // rotate by adding perpendicular and scaling down
        p.xy += vec2(p.y, -p.x) * nudge;
        p.xy *= normalizer;
        // rotate on other axis
        p.xz += vec2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        // increase the frequency
        iter *= 1.733733;
    }
    return n;
}

// Function 245
float value_noise(vec2 p)
{
    vec2 pi = floor(p);
    vec2 pf = p - pi;
    
    vec2 w = pf * pf * (3.0 - 2.0 * pf);
    
    return mix(mix(hash21(pi + vec2(0.0, 0.0)), hash21(pi + vec2(1.0, 0.0)), w.x),
               mix(hash21(pi + vec2(0.0, 1.0)), hash21(pi + vec2(1.0, 1.0)), w.x),
               w.y);
}

// Function 246
float snoise(vec2 P){
  const vec2 C = vec2 (0.211324865405187134,  // (3.0-sqrt ( 3 . 0 ) ) / 6 . 0 ;
  0.366025403784438597) ;  //  0.5*( sqrt ( 3 . 0 )-1.0) ;
  //  First  corner
  vec2 i = floor(P+ dot (P,C.yy)  ) ;
  vec2 x0=P-i+ dot (i,C.xx) ;// Other  corners
  vec2 i1;
  i1.x = step (x0.y,x0.x) ;  //  1.0 if(x0.x > x0.y ,  e l s e  0.0
  i1.y = 1.0-i1.x;
  // x1 = x0-i1 + 1.0*C. xx ;  x2 = x0-1.0 + 2.0*C. xx ;
  vec4 x12 = x0.xyxy + vec4 (C.xx,C.xx*2.0-1.0) ;x12.xy-=i1;//  Permutations
  i = mod(i,  289.0);  // Avoid  truncation  in  polynomial  evaluation
  vec3 p = permute(permute(i.y+ vec3 (0.0 ,i1.y,  1.0  ) )+i.x+ vec3 (0.0 ,i1.x,  1.0  ) ) ;//  Circularly  symmetric  blending  kernel
  vec3 m = max(0.5-vec3 ( dot (x0,x0) ,  dot (x12.xy,x12.xy) ,dot (x12.zw,x12.zw) ) ,  0.0) ;m=m*m;m=m*m;//  Gradients  from 41  points  on a  line ,  mapped onto a diamond
  vec3 x= fract(p*(1.0  /  41.0) )*2.0-1.0  ;vec3 gy= abs (x)-0.5  ;vec3 ox= floor(x+ 0.5) ;  // round (x)  i s  a GLSL 1.30  feature
  vec3 gx = x-ox;//  Normalise  gradients  i m p l i c i t l y  by  s c a l i n g m
  m *= taylorInvSqrt(gx*gx+gy*gy) ;// Compute  f i n a l  noise  value  at P
  vec3 g;g.x=gx.x*x0.x+gy.x*x0.y;g.yz=gx.yz*x12.xz+gy.yz*x12.yw;//  Scale  output  to  span  range  [-1 ,1]//  ( s c a l i n g  f a c t o r  determined by  experiments )
  return  130.0*dot (m,g) ;
}

// Function 247
float noise(vec2 x) {
	return iqnoise(x,0.0,1.0);
}

// Function 248
float boxNoise( in vec2 p, in float z )
{
    vec2 fl = floor(p);
    vec2 fr = fract(p);
    fr = smoothstep(0.0, 1.0, fr);    
    float res = mix(mix( hash13(vec3(fl, z)),             hash13(vec3(fl + vec2(1,0), z)),fr.x),
                    mix( hash13(vec3(fl + vec2(0,1), z)), hash13(vec3(fl + vec2(1,1), z)),fr.x),fr.y);
    return res;
}

// Function 249
float bnoise(in vec3 p)
{
    float n = sin(triNoise3d(p*3.)*7.)*0.4;
    n += sin(triNoise3d(p*1.5)*7.)*0.2;
    return (n*n)*0.01;
}

// Function 250
float SpiralNoiseC(vec3 p)
{
    float n = 1.-mod(iTime * 0.1,-1.); // noise amount
    float iter = 2.0;
    for (int i = 0; i < 8; i++)
    {
        // add sin and cos scaled inverse with the frequency
        n += -abs(sin(p.y*iter) + cos(p.x*iter)) / iter;	// abs for a ridged look
        // rotate by adding perpendicular and scaling down
        p.xy += vec2(p.y, -p.x) * nudge;
        p.xy *= normalizer;
        // rotate on other axis
        p.xz += vec2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        // increase the frequency
        iter *= 1.733733;
    }
    return n;
}

// Function 251
float noise3D(in vec3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
	const vec3 s = vec3(27, 57, 113);
	
	vec3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride vector for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    
	p -= ip; // Cell's fractional component.
	
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: fract(sin(x)*largeNumber).
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
	
    // Interpolating along Y.
    h.xy = mix(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return mix(h.x, h.y, p.z); // Range: [0, 1].
	
}

// Function 252
float noise(vec2 p) {
    float A = rand(vec2(floor(p.x), floor(p.y)));
    float B = rand(vec2(floor(p.x) + 1.0, floor(p.y)));
    float C = rand(vec2(floor(p.x), floor(p.y) + 1.0));
    float D = rand(vec2(floor(p.x) + 1.0, floor(p.y) + 1.0));

    float fc = fract(p.x);
    float bicubicc = fc * fc * (3.0 - 2.0 * fc);

    float fr = fract(p.y);
    float bicubicr = fr * fr * (3.0 - 2.0 * fr);

    float AB = mix(A, B, bicubicc);
    float CD = mix(C, D, bicubicc);

    float final = mix(AB, CD, bicubicr);

    return final;
}

// Function 253
float smoothNoise(vec2 p)
{
    float n0=noise(floor(p)+vec2(0,0));
    float n1=noise(floor(p)+vec2(1,0));
    float n2=noise(floor(p)+vec2(0,1));
    float n3=noise(floor(p)+vec2(1,1));
    float u=fract(p.x),v=fract(p.y);
    return mix(mix(n0,n1,u),mix(n2,n3,u),v);
}

// Function 254
vec2 perlin(vec2 p)
{
   vec2 pi = floor(p);
   vec2 pf = p - pi;
   vec2 a = vec2(0.,1.);
   return hash22(pi+a.xx)*(1.-pf.x)*(1.-pf.y) +
          hash22(pi+a.xy)*(1.-pf.x)*pf.y +
          hash22(pi+a.yx)*pf.x*(1.-pf.y) +
          hash22(pi+a.yy)*pf.x*pf.y;   
}

// Function 255
float noise(vec2 p){
    vec2 ip = floor(p); vec2 u = fract(p);u=u*u*(3.0-2.0*u);
    float res = mix(
        mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
        mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
    return res;
}

// Function 256
float noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f *= f * (3.0 - 2.0*f);
    
    return mix( mix( hash(i+vec2(0.0, 0.0)), hash(i+vec2(1.0, 0.0)), f.x ),
                mix( hash(i+vec2(0.0, 1.0)), hash(i+vec2(1.0, 1.0)), f.x ), f.y );
}

// Function 257
vec3 noise3(vec2 p){vec2 f=fract(p),g=f*f,u=g*(3.-2.*f);vec4 h=hash4(dot(floor(p),hSeed.yzw.xy))
 ;return vec3(h.x+(h.y-h.x)*u.x+(h.z-h.x)*u.y+(h.x-h.y-h.z+h.w)*u.x*u.y,30.*g*(g-2.*f+1.)*(vec2(h.y-h.x,h.z-h.x)+(h.x-h.y-h.z+h.w)*u.yx));}

// Function 258
float tweaknoise( vec2 p) {
    float d=0.;
    for (float i=0.; i<5.; i++) {
        float a0 = hash(i+5.6789), a=1.*a0*iTime+2.*PI*a0; vec2 dp=vec2(cos(a),sin(a)); 
                
        vec2 ip = hash12(i+5.6789)+dp;
        float di = smoothstep(grad/2.,-grad/2.,length(p-ip)-.5);
        d += (1.-d)*di;
    }
    //float d = smoothstep(grad/2.,-grad/2.,length(p)-.5);
#if NOISE==1 // 3D Perlin noise
    float v = fbm(vec3(scale*p,.5));
#elif NOISE==2 // Worley noise
    float v = 1. - scale*worley(scale*p).x;
#elif NOISE>=3 // trabeculum 2D
    if (d<0.5) return 0.;
    grad=.8, scale = 5.;
	vec3 w = scale*worley(scale*p);
    float v;
    if (false) // keyToggle(32)) 
        v =  2.*scale*worleyD(scale*p);
    else
 	v= w.y-w.x;	 //  v= 1.-1./(w.y-w.x);
#endif
    
    return v*d;
    //return smoothstep(thresh-grad/2.,thresh+grad/2.,v*d);
}

// Function 259
float modifiedVoronoiNoise12(vec2 uv)
{
 	vec2 rootCell = floor(uv);

    float value = 0.0;

    for (float x = -1.0; x <= 1.0; x++)
    {
     	for(float y = -1.0; y <= 1.0; y++)
        {
         	vec2 cell = rootCell + vec2(x, y);
            vec2 cellPoint = getCellPoint(cell);
            float cellValue = getCellValue(cell);
            float cellDist = distance(uv, cellPoint);
            value += makeSmooth(clamp(1.0 - cellDist, 0.0, 1.0)) * cellValue;
        }
    }

    return value * 0.5;
}

// Function 260
vec4 snoise(vec3 v)
{
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);

    // First corner
    vec3 i  = floor(v + dot(v, vec3(C.y)));
    vec3 x0 = v   - i + dot(i, vec3(C.x));

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.x;
    vec3 x2 = x0 - i2 + C.y;
    vec3 x3 = x0 - 0.5;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    vec4 p =
      permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0))
                            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                            + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    vec4 j = p - 49.0 * floor(p / 49.0);  // mod(p,7*7)

    vec4 x_ = floor(j / 7.0);
    vec4 y_ = floor(j - 7.0 * x_); 

    vec4 x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    vec4 y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;

    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 g0 = vec3(a0.xy, h.x);
    vec3 g1 = vec3(a0.zw, h.y);
    vec3 g2 = vec3(a1.xy, h.z);
    vec3 g3 = vec3(a1.zw, h.w);

    // Normalize gradients
    vec4 norm = taylorInvSqrt(vec4(dot(g0, g0), dot(g1, g1), dot(g2, g2), dot(g3, g3)));
    g0 *= norm.x;
    g1 *= norm.y;
    g2 *= norm.z;
    g3 *= norm.w;

    // Compute noise and gradient at P
    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    vec4 m2 = m * m;
    vec4 m3 = m2 * m;
    vec4 m4 = m2 * m2;
    vec3 grad =
      -6.0 * m3.x * x0 * dot(x0, g0) + m4.x * g0 +
      -6.0 * m3.y * x1 * dot(x1, g1) + m4.y * g1 +
      -6.0 * m3.z * x2 * dot(x2, g2) + m4.z * g2 +
      -6.0 * m3.w * x3 * dot(x3, g3) + m4.w * g3;
    vec4 px = vec4(dot(x0, g0), dot(x1, g1), dot(x2, g2), dot(x3, g3));
    return 42.0 * vec4(grad, dot(m4, px));
}

// Function 261
vec3 voronoiSphereMapping(vec3 n){
	vec2 uv=vec2(atan(n.x,n.z),acos(n.y));
   	return getVoronoi(1.5*uv);}

// Function 262
float snoise(vec2 v)
  {
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
// First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                + i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// Function 263
F1 Noise(F2 n,F1 x){n+=x;return fract(sin(dot(n.xy,F2(12.9898, 78.233)))*43758.5453)*2.0-1.0;}

// Function 264
float InterleavedGradientNoise( vec2 uv )
{
    const vec3 magic = vec3( 0.06711056, 0.00583715, 52.9829189 );
    return fract( magic.z * fract( dot( uv, magic.xy ) ) );
}

// Function 265
float noise1D(float x) {
    float p = floor(x);
    float n = fract(x);
    float f = n*n*(3.0-2.0*n);
    float winx = 1.0;
    float winy = 2.0;
    
    return mix(
                mix(hash1D(p)     , hash1D(p+winx), f),
                mix(hash1D(p+winy), hash1D(p+winx+winy), f),
                f);
}

// Function 266
vec4 voronoi( in vec2 x, float w )
{
    vec2 n = floor( x );
    vec2 f = fract( x );

	vec4 m = vec4( 8.0, 0.0, 0.0, 0.0 );
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = vec2( float(i),float(j) );
        vec2 o = hash2( n + g );
		
		// animate
        o = 0.5 + 0.5*sin( iTime + 6.2831*o );

        // distance to cell		
		float d = length(g - f + o);
		
        // do the smoth min for colors and distances		
		vec3 col = 0.5 + 0.5*sin( hash1(dot(n+g,vec2(7.0,113.0)))*2.5 + 3.5 + vec3(2.0,3.0,0.0));
		float h = smoothstep( 0.0, 1.0, 0.5 + 0.5*(m.x-d)/w );
		
	    m.x   = mix( m.x,     d, h ) - h*(1.0-h)*w/(1.0+3.0*w); // distance
		m.yzw = mix( m.yzw, col, h ) - h*(1.0-h)*w/(1.0+3.0*w); // color
    }
	
	return m;
}

// Function 267
float noise(float t){return textureLod(iChannel0,vec2(t,.0)/iChannelResolution[0].xy,0.0).x;}

// Function 268
float voronoi_ao(in float h, inout vec2 n, inout vec2 f, inout vec2 mg) {
    float a = 0.0;
    
    for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i) 
    {
    	vec2 g = mg + vec2(float(i), float(j));
        a += max(0.0, map(n + g) - h);
    }
    
    return exp(-a*0.5) + 0.2;
    //return max(0.0, 1.0 - a*0.2) + 0.2;
}

// Function 269
float Noisefv2 (vec2 p){
  vec4 t=Hashv4f(dot(floor(p),cHashA4.yz));
  p=fract(p);p*=p*(3.-2.*p);
  return mix (mix(t.x,t.y,p.x),
               mix(t.z,t.w,p.x),p.y);}

// Function 270
float noise( in float p )
{
    return noise(vec2(p, 0.0));        
}

// Function 271
void noise_prng_srand(inout noise_prng this_, in uint s)
{
    this_.x_ = s;
}

// Function 272
float TileableNoise(in vec3 p, in float numCells )
{
	vec3 f, i;
	
	p *= numCells;

	
	f = fract(p);		// Separate integer from fractional
    i = floor(p);
	
    vec3 u = f*f*(3.0-2.0*f); // Cosine interpolation approximation

    return mix( mix( mix( dot( Hash( i + vec3(0.0,0.0,0.0), numCells ), f - vec3(0.0,0.0,0.0) ), 
                          dot( Hash( i + vec3(1.0,0.0,0.0), numCells ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( Hash( i + vec3(0.0,1.0,0.0), numCells ), f - vec3(0.0,1.0,0.0) ), 
                          dot( Hash( i + vec3(1.0,1.0,0.0), numCells ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( Hash( i + vec3(0.0,0.0,1.0), numCells ), f - vec3(0.0,0.0,1.0) ), 
                          dot( Hash( i + vec3(1.0,0.0,1.0), numCells ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( Hash( i + vec3(0.0,1.0,1.0), numCells ), f - vec3(0.0,1.0,1.0) ), 
                          dot( Hash( i + vec3(1.0,1.0,1.0), numCells ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// Function 273
float interpNoise3D1(vec3 p) {
  vec3 pFract = fract(p);
  float llb = random1(floor(p));
  float lrb = random1(floor(p) + vec3(1.0,0.0,0.0));
  float ulb = random1(floor(p) + vec3(0.0,1.0,0.0));
  float urb = random1(floor(p) + vec3(1.0,1.0,0.0));

  float llf = random1(floor(p) + vec3(0.0,0.0,1.0));
  float lrf = random1(floor(p) + vec3(1.0,0.0,1.0));
  float ulf = random1(floor(p) + vec3(0.0,1.0,1.0));
  float urf = random1(floor(p) + vec3(1.0,1.0,1.0));

  float lerpXLB = mySmootherStep(llb, lrb, pFract.x);
  float lerpXHB = mySmootherStep(ulb, urb, pFract.x);
  float lerpXLF = mySmootherStep(llf, lrf, pFract.x);
  float lerpXHF = mySmootherStep(ulf, urf, pFract.x);

  float lerpYB = mySmootherStep(lerpXLB, lerpXHB, pFract.y);
  float lerpYF = mySmootherStep(lerpXLF, lerpXHF, pFract.y);

  return mySmootherStep(lerpYB, lerpYF, pFract.z);
}

// Function 274
vec2 multiNoise(vec4 pos, vec4 scale, float phase, float seed) 
{
    const float kPI2 = 6.2831853071;
    pos *= scale;
    vec4 i = floor(pos);
    vec4 f = pos - i;
    vec4 i0 = mod(i.xyxy + vec2(0.0, 1.0).xxyy, scale.xyxy) + seed;
    vec4 i1 = mod(i.zwzw + vec2(0.0, 1.0).xxyy, scale.xyxy) + seed;

    vec4 hash0 = multiHash2D(i0);
    hash0 = 0.5 * sin(phase + kPI2 * hash0) + 0.5;
    vec4 hash1 = multiHash2D(i1);
    hash1 = 0.5 * sin(phase + kPI2 * hash1) + 0.5;
    vec2 a = vec2(hash0.x, hash1.x);
    vec2 b = vec2(hash0.y, hash1.y);
    vec2 c = vec2(hash0.z, hash1.z);
    vec2 d = vec2(hash0.w, hash1.w);

    vec4 u = noiseInterpolate(f);
    vec2 value = mix(a, b, u.xz) + (c - a) * u.yw * (1.0 - u.xz) + (d - b) * u.xz * u.yw;
    return value * 2.0 - 1.0;
}

// Function 275
vec3 noise(float p){return texture(iChannel0,vec2(p/iChannelResolution[0].x,.0)).xyz;}

// Function 276
float noise1t(vec3 p,float spd//triangle-interpolation noise.
){float z=1.4,r=0.
 ;p=p*9.+vec3(7,13,21)//optionally evade the strong [y=x mirror] that afo3() has
 ;vec3 b=p
 ;for(float i=0.;i<4.;i++//multi-octaves,but the afo3(()function also implies a sqivel-rotation.
 ){vec3 dg=afo3(b*2.)
  ;p+=(dg+      spd);b*=1.8;z*=1.5;p*=1.2
//;p+=(dg+iTime*spd);b*=1.8;z*=1.5;p*=1.2
  ;r+=perm3(u5cos2,p)/z //a weird way of using define,deal with it
  ;b+=.14;};return r;}

// Function 277
vec3 Voronoi(in vec3 p, in vec3 rd){
    
    // One of Tomkh's snippets that includes a wrap to deal with
    // larger numbers, which is pretty cool.

 
    vec3 n = floor(p);
    p -= n + .5;
 
    
    // Storage for all sixteen hash values. The same set of hash values are
    // reused in the second pass, and since they're reasonably expensive to
    // calculate, I figured I'd save them from resuse. However, I could be
    // violating some kind of GPU architecture rule, so I might be making 
    // things worse... If anyone knows for sure, feel free to let me know.
    //
    // I've been informed that saving to an array of vectors is worse.
    //vec2 svO[3];
    
    // Individual Voronoi cell ID. Used for coloring, materials, etc.
    cellID = vec3(0); // Redundant initialization, but I've done it anyway.

    // As IQ has commented, this is a regular Voronoi pass, so it should be
    // pretty self explanatory.
    //
    // First pass: Regular Voronoi.
	vec3 mo, o;
    
    // Minimum distance, "smooth" distance to the nearest cell edge, regular
    // distance to the nearest cell edge, and a line distance place holder.
    float md = 8., lMd = 8., lMd2 = 8., lnDist, d;
    
    // Note the ugly "gIFrame" hack. The idea is to force the compiler not
    // to unroll the loops, thus keep the program size down... or something. 
    // GPU compilation is not my area... Come to think of it, none of this
    // is my area. :D
    for( int k=min(-2, gIFrame); k<=2; k++ ){
    for( int j=min(-2, gIFrame); j<=2; j++ ){
    for( int i=min(-2, gIFrame); i<=2; i++ ){
    
        o = vec3(i, j, k);
        o += hash33(n + o) - p;
        // Saving the hash values for reuse in the next pass. I don't know for sure,
        // but I've been informed that it's faster to recalculate the had values in
        // the following pass.
        //svO[j*3 + i] = o; 
  
        // Regular squared cell point to nearest node point.
        d = dot(o, o); 

        if( d<md ){
            
            md = d;  // Update the minimum distance.
            // Keep note of the position of the nearest cell point - with respect
            // to "p," of course. It will be used in the second pass.
            mo = o; 
            cellID = vec3(i, j, k) + n; // Record the cell ID also.
        }
       
    }
    }
    }
    
    // Second pass: Distance to closest border edge. The closest edge will be one of the edges of
    // the cell containing the closest cell point, so you need to check all surrounding edges of 
    // that cell, hence the second pass... It'd be nice if there were a faster way.
    for( int k=min(-3, gIFrame); k<=3; k++ ){
    for( int j=min(-3, gIFrame); j<=3; j++ ){
    for( int i=min(-3, gIFrame); i<=3; i++ ){
        
        // I've been informed that it's faster to recalculate the hash values, rather than 
        // access an array of saved values.
        o = vec3(i, j, k);
        o += hash33(n + o) - p;
        // I went through the trouble to save all sixteen expensive hash values in the first 
        // pass in the hope that it'd speed thing up, but due to the evolving nature of 
        // modern architecture that likes everything to be declared locally, I might be making 
        // things worse. Who knows? I miss the times when lookup tables were a good thing. :)
        // 
        //o = svO[j*3 + i];
        
        // Skip the same cell... I found that out the hard way. :D
        if( dot(o - mo, o - mo)>.00001 ){ 
            
            // This tiny line is the crux of the whole example, believe it or not. Basically, it's
            // a bit of simple trigonometry to determine the distance from the cell point to the
            // cell border line. See IQ's article for a visual representation.
            lnDist = dot(0.5*(o + mo), normalize(o - mo));
            
            // Abje's addition. Border distance using a smooth minimum. Insightful, and simple.
            //
            // On a side note, IQ reminded me that the order in which the polynomial-based smooth
            // minimum is applied effects the result. However, the exponentional-based smooth
            // minimum is associative and commutative, so is more correct. In this particular case, 
            // the effects appear to be negligible, so I'm sticking with the cheaper polynomial-based 
            // smooth minimum, but it's something you should keep in mind. By the way, feel free to 
            // uncomment the exponential one and try it out to see if you notice a difference.
            //
            // // Polynomial-based smooth minimum.
            //lMd = smin(lMd, lnDist, lnDist*.75); //lnDist*.75
            //
            // Exponential-based smooth minimum. By the way, this is here to provide a visual reference 
            // only, and is definitely not the most efficient way to apply it. To see the minor
            // adjustments necessary, refer to Tomkh's example here: Rounded Voronoi Edges Analysis - 
            // https://www.shadertoy.com/view/MdSfzD
            lMd = sminExp(lMd, lnDist, 16.); 
            
            // Minimum regular straight-edged border distance. If you only used this distance,
            // the web lattice would have sharp edges.
            lMd2 = min(lMd2, lnDist);
        }

    }
    }
    }

    // Return the smoothed and unsmoothed distance. I think they need capping at zero... but 
    // I'm not positive.
    return max(vec3(lMd, lMd2, md), 0.);
}

// Function 278
vec3 sdVoronoiCell(in vec2 p, in vec2 cellId ) {
    float md = 8.0;    
    vec2 mr = hash2(cellId);
    for(int i=0; i<8; i++) {
        vec2 g = ID_POS(i),
             r = g + hash2(cellId + g);
        md = min(md, dot(.5*(mr+r)-p, normalize(r-mr)));
    }
    return vec3(-md, mr-p);
}

// Function 279
float noise(in vec2 p) {
	vec2 F = floor(p), f = fract(p);
	f = f * f * (3. - 2. * f);
	return mix(
		mix(hash(F), 			 hash(F+vec2(1.,0.)), f.x),
		mix(hash(F+vec2(0.,1.)), hash(F+vec2(1.)),	  f.x), f.y);
}

// Function 280
float noise(vec3 x) {
    const vec3 step = vec3(110, 241, 171);
    vec3 i = floor(x);
    vec3 f = fract(x);
    float n = dot(i, step);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

// Function 281
float fractalNoise(vec2 vl) {
    float persistance = 2.0;
    float amplitude = 1.2;
    float rez = 0.0;
    vec2 p = vl;
    
    for (float i = 0.0; i < OCTAVES; i++) {
        rez += amplitude * valueNoiseSimple(p);
        amplitude /= persistance;
        p *= persistance; // Actually the size of the grid and noise frequency
        //frequency *= persistance;
    }
    return rez;
}

// Function 282
float voronoi(vec3 x) {
	vec3 p = floor(x);
	vec3 f = fract(x);

	vec2 res = vec2(8.0);

	for(int i = -1; i <= 1; i++)
	for(int j = -1; j <= 1; j++)
	for(int k = -1; k <= 1; k++) {
		vec3 g = vec3(float(i), float(j), float(k));
		vec3 r = g + hash(p + g) - f;

		float d = max(abs(r.x), max(abs(r.y), abs(r.z)));

		if(d < res.x) {
			res.y = res.x;
			res.x = d;
		} else if(d < res.y) {
			res.y = d;
		}
	}

	return res.y - res.x;
}

// Function 283
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);
	return textureLod( iChannel1, (uv+118.4)/256.0, 0.0 ).x;
}

// Function 284
float valueNoise2D(vec2 p) {
	
	vec2 f = fract(p); // Fractional cell position.
    
    f *= f*(3. -2.*f);// Smooth step
    //f = f*f*f*(10. + f*(6.*f - 15.)); // Smoother smooth step.
    //f = (1. - cos(f*3.14159265))*.5; // Cos smooth step
	
    // Random values for all four cell corners.
	vec4 h = fract(sin(vec4(0, 41, 289, 330) + dot(floor(p), vec2(41, 289)))*43758.5453);
	h = sin(h*6.283 + iTime)*.5 + .5; // Animation.
	//h = abs(fract(h+iTime*.125) - .5)*2.; // More linear animation.
	
    // Interpolating the random values to produce the final value.
	return dot(vec2(1. - f.y, f.y), vec2(1. - f.x, f.x)*mat2(h));
    
}

// Function 285
float Pseudo3dNoise(vec3 pos) {
    vec2 i = floor(pos.xy);
    vec2 f = pos.xy - i;
    vec2 blend = f * f * (3.0 - 2.0 * f);
    float noiseVal = 
        mix(
            mix(
                dot(GetGradient(i + vec2(0, 0), pos.z), f - vec2(0, 0)),
                dot(GetGradient(i + vec2(1, 0), pos.z), f - vec2(1, 0)),
                blend.x),
            mix(
                dot(GetGradient(i + vec2(0, 1), pos.z), f - vec2(0, 1)),
                dot(GetGradient(i + vec2(1, 1), pos.z), f - vec2(1, 1)),
                blend.x),
        blend.y
    );
    return noiseVal / 0.7; // normalize to about [-1..1]
}

// Function 286
vec4 texture_denoise(in sampler2D tex, vec2 uv, float threshold, vec3 q)
{
    vec4 col = texture(tex, uv),
        blurred = texture_blurred_quantized(tex, uv, q);
    
    if (length(col-blurred) <= threshold)
        return blurred;
    else
        return col;
}

// Function 287
float noise(vec2 p) {
    return texture(iChannel0, p * 0.05 ).x;
}

// Function 288
vec4 voronoi( in vec2 x, float mode )
{
    vec2 n = floor( x );
    vec2 f = fract( x );

	vec3 m = vec3( 8.0 );
	float m2 = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = vec2( float(i),float(j) );
        vec2 o = hash2( n + g );

		// animate
        o = 0.5 + 0.5*sin( iTime + 6.2831*o );

		vec2 r = g - f + o;

        // euclidean		
		vec2 d0 = vec2( sqrt(dot(r,r)), 1.0 );
        // manhattam		
		vec2 d1 = vec2( 0.71*(abs(r.x) + abs(r.y)), 1.0 );
        // triangular		
		vec2 d2 = vec2( max(abs(r.x)*0.866025+r.y*0.5,-r.y), 
				        step(0.0,0.5*abs(r.x)+0.866025*r.y)*(1.0+step(0.0,r.x)) );

		vec2 d = d0; 
		if( mode<3.0 ) d=mix( d2, d0, fract(mode) );
		if( mode<2.0 ) d=mix( d1, d2, fract(mode) );
		if( mode<1.0 ) d=mix( d0, d1, fract(mode) );
		
        if( d.x<m.x )
        {
			m2 = m.x;
            m.x = d.x;
            m.y = hash1( dot(n+g,vec2(7.0,113.0) ) );
			m.z = d.y;
        }
		else if( d.x<m2 )
		{
			m2 = d.x;
		}

    }
    return vec4( m, m2-m.x );
}

// Function 289
vec2 noise(vec2 tc) {
  return (2.*texture(iChannel0, tc.xy ).xy - 1.).xy;
}

// Function 290
vec4 noised( in vec3 x, out mat3 dd )
{
    vec3 p = floor(x);
    vec3 w = fract(x);

    // cubic interpolation vs quintic interpolation
#if 0
    vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
    vec3 ddu = 6.0 - 12.0*w;
#else
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
    vec3 ddu = 60.0*w*(1.0+w*(-3.0+2.0*w));
#endif
    
    
    float n = p.x + p.y*157.0 + 113.0*p.z;
    
    float a = hash(n+  0.0);
    float b = hash(n+  1.0);
    float c = hash(n+157.0);
    float d = hash(n+158.0);
    float e = hash(n+113.0);
	float f = hash(n+114.0);
    float g = hash(n+270.0);
    float h = hash(n+271.0);
	
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    dd = mat3( ddu.x*(k1 + k4*u.y + k6*u.z + k7*u.y*u.z), 
               du.x*(k4+k7*u.z)*du.y,
               du.x*(k6+k7*u.y)*du.z,
              
               du.y*(k4+k7*u.z)*du.x,
               ddu.y*(k2 + k5*u.z + k4*u.x + k7*u.z*u.x),
               du.y*(k5+k7*u.x)*du.z,
              
               du.z*(k6+k7*u.y)*du.x,
               du.z*(k5+k7*u.x)*du.y,
               ddu.z*(k3 + k6*u.x + k5*u.y + k7*u.x*u.y) );


    return vec4( k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z, 
                 du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                            k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                            k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) );
}

// Function 291
float noise(  vec2 p )
{
    vec2 i = floor( p ), f = fract( p ), u = f*f*(3.-2.*f);
    float v = mix( mix( dot( hash( i + vec2(0,0) ), f - vec2(0,0) ), 
                     dot( hash( i + vec2(1,0) ), f - vec2(1,0) ), u.x),
                mix( dot( hash( i + vec2(0,1) ), f - vec2(0,1) ), 
                     dot( hash( i + vec2(1,1) ), f - vec2(1,1) ), u.x), u.y);
    return v;
    //return 1.-abs(2.*v-1.);
    //return abs(2.*v-1.);
}

// Function 292
float noise(float x) 
{
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3.0 - 2.0 * f);
    float result = mix(hash(i), hash(i + 1.0), u);
    result = remap( result, 0.0, 1.0, 0.25, 1.5);
    return result*1.5;
}

// Function 293
float custom_perlin_hash(vec2 uv)
{
 	vec2 lower	= floor(uv);
    vec2 frac 	= fract(uv);
    vec2 f = frac*frac*(3.0-2.0*frac);
    
    return mix(
        	mix(hash(lower+vec2(0.0, 0.0)), hash(lower+vec2(1.0, 0.0)), f.x),
        	mix(hash(lower+vec2(0.0, 1.0)), hash(lower+vec2(1.0, 1.0)), f.x),
        	f.y);
}

// Function 294
float complicatedNoise(float t, vec2 uv){
    float f1 = float(hash12(uv+floor(t)*10.) < THRESHOLD);
    float f2 = float(hash12(uv+floor(t+1.)*10.) < THRESHOLD);
    float f3 = float(hash12(uv+floor(t+2.)*10.) < THRESHOLD);
    return ((f1+f2)*.5+f3)*.5;
}

// Function 295
vec4 texNoise(vec2 uv){ float f = 0.; f+=texture(iChannel0, uv*.125).r*.5; f+=texture(iChannel0,uv*.25).r*.25; //MERCURTY SDF LIBRARY IS HERE OFF COURSE: http://mercury.sexy/hg_sdf/
     f+=texture(iChannel0,uv*.5).r*.125; f+=texture(iChannel0,uv*1.).r*.125; f=pow(f,1.2);return vec4(f*.45+.05);}

// Function 296
float pnoise12(vec2 p, float scl) {
    vec2 i = floor(p*scl),
     f = fract(p*scl),
     u = vec2(fade(f.x), fade(f.y)),
     o = vec2(0., 1.),
     g00 = hash22(i).xy, g01 = hash22(i + o.xy).xy,
     g11 = hash22(i + o.yy).xy, g10 = hash22(i + o.yx).xy,
     d00 = f, d01 = f - o.xy,
     d11 = f - o.yy, d10 = f - o.yx;
    float s00 = dot(g00, d00), s01 = dot(g01, d01),
     s11 = dot(g11, d11), s10 = dot(g10, d10),
     x1 = mix(s01,s11, u.x), x2 = mix(s00,s10, u.x);
    return mix(x2, x1, u.y);
}

// Function 297
vec2 cellularNoise(vec2 pos, vec2 scale, float jitter, float phase, uint metric, float seed) 
{       
    const float kPI2 = 6.2831853071;
    pos *= scale;
    vec2 i = floor(pos);
    vec2 f = pos - i;
    
    const vec3 offset = vec3(-1.0, 0.0, 1.0);
    vec4 cells = mod(i.xyxy + offset.xxzz, scale.xyxy) + seed;
    i = mod(i, scale) + seed;
    vec4 dx0, dy0, dx1, dy1;
    multiHash2D(vec4(cells.xy, vec2(i.x, cells.y)), vec4(cells.zyx, i.y), dx0, dy0);
    multiHash2D(vec4(cells.zwz, i.y), vec4(cells.xw, vec2(i.x, cells.w)), dx1, dy1);
    dx0 = 0.5 * sin(phase + kPI2 * dx0) + 0.5;
    dy0 = 0.5 * sin(phase + kPI2 * dy0) + 0.5;
    dx1 = 0.5 * sin(phase + kPI2 * dx1) + 0.5;
    dy1 = 0.5 * sin(phase + kPI2 * dy1) + 0.5;
    
    dx0 = offset.xyzx + dx0 * jitter - f.xxxx; // -1 0 1 -1
    dy0 = offset.xxxy + dy0 * jitter - f.yyyy; // -1 -1 -1 0
    dx1 = offset.zzxy + dx1 * jitter - f.xxxx; // 1 1 -1 0
    dy1 = offset.zyzz + dy1 * jitter - f.yyyy; // 1 0 1 1
    vec4 d0 = distanceMetric(dx0, dy0, metric);
    vec4 d1 = distanceMetric(dx1, dy1, metric);
    
    vec2 centerPos = (0.5 * sin(phase + kPI2 *  multiHash2D(i)) + 0.5) * jitter - f; // 0 0
    vec4 F = min(d0, d1);
    // shuffle into F the 4 lowest values
    F = min(F, max(d0, d1).wzyx);
    // shuffle into F the 2 lowest values 
    F.xy = min(min(F.xy, F.zw), max(F.xy, F.zw).yx);
    // add the last value
    F.zw = vec2(distanceMetric(centerPos, metric), 1e+5);
    // shuffle into F the final 2 lowest values 
    F.xy = min(min(F.xy, F.zw), max(F.xy, F.zw).yx);
    
    vec2 f12 = vec2(min(F.x, F.y), max(F.x, F.y));
    // normalize: 0.75^2 * 2.0  == 1.125
    return (metric == 0u ? sqrt(f12) : f12) * (1.0 / 1.125);
}

// Function 298
float waterNoise(vec2 x) {
    vec2 p = floor(x);
    vec2 n = fract(x);
    vec2 f = n*n*(3.0-2.0*n);
    float winx = 1.0;
    float winy = 1.0;
    
    return mix(
                mix(hash2D(p)     , hash2D(p+vec2(winx, 0.0)), f.x),
                mix(hash2D(p+vec2(0.0, winy)), hash2D(p+vec2(winx, winy)), f.x),
                f.y);
}

// Function 299
float Noise21 (vec2 p, float ta, float tb) {
    return fract(sin(p.x*ta+p.y*tb)*5678.);
}

// Function 300
float Noise2D(vec2 uv){
    //uv+=iTime;
    vec2 st = fract(uv);
    vec2 id = floor(uv);
    st = st*st*(3.0-2.0*st);
    float c=mix(mix(N21(id),N21(id+vec2(1.0,0.0)),st.x),mix(N21(id+vec2(0.0,1.0)),N21(id+vec2(1.0,1.0)),st.x),st.y);
	return c;
}

// Function 301
float noise( in vec4 x )
{
    vec4 p = floor(x);
    vec4 f = fract(x);
	f = f*f*(3.0-2.0*f);
    
	vec2 uv = (p.xy + p.z*zOffset + p.w*wOffset) + f.xy;
    
   	vec4 s = tex(uv);
	return mix(mix( s.x, s.y, f.z ), mix(s.z, s.w, f.z), f.w);
}

// Function 302
float NoiseT( in vec2 x ) {
    vec2 p = floor(x), f = fract(x);
    f = f*f*(3.-2.*f);
    float n = p.x + p.y*57.0;
    return mix(mix( HashT(n     ), HashT(n+  1.),f.x),
               mix( HashT(n+ 57.), HashT(n+ 58.),f.x),f.y);
}

// Function 303
float Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel2, (uv+ 0.5)/256.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 304
float noise( in vec3 x )
{
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix( hash(i+vec3(0,0,0)), 
                        hash(i+vec3(1,0,0)),f.x),
                   mix( hash(i+vec3(0,1,0)), 
                        hash(i+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(i+vec3(0,0,1)), 
                        hash(i+vec3(1,0,1)),f.x),
                   mix( hash(i+vec3(0,1,1)), 
                        hash(i+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 305
float FractalNoise(in vec2 xy)
{
	float w = .8;
	float f = 0.0;

	for (int i = 0; i < 4; i++)
	{
		f += CloudNoise(xy) * w;
		w = w*0.5;
		xy = rotate2D * xy;
	}
	return f;
}

// Function 306
float noiseTexture(vec2 p, float t)
{
    p = .5*p + t;
    float val = tex(p);
    //return val;
    
    if keypress(32)
    {
        val *= .5;
        val += tex(p*2.) *.25;
        val += tex(p*4.) *.125;
        val += tex(p*8.) *.0625;
    }
    
    return val;
}

// Function 307
vec2 Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
//	vec3 f2 = f*f; f = f*f2*(10.0-15.0*f+6.0*f2);

	// there's an artefact because the y channel almost, but not exactly, matches the r channel shifted (37,17)
	// this artefact doesn't seem to show up in chrome, so I suspect firefox uses different texture compression.
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec4 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0 );
	return mix( rg.yw, rg.xz, f.z );
}

// Function 308
float voronoiBorder(vec2 x, float seed, float phase) {
	vec2 xi = floor(x);
	vec2 xf = fract(x);
	
	vec2 res = vec2(10.);
	for(int i=-1; i<=1; i++) {
		for(int j=-1; j<=1; j++) {
			vec2 b = vec2(i,j);
			vec2 rv = rand2f(xi+b+seed*1.3412);
			rv = sin(rv*pi2 + phase)*.5+.5;
			rv *= .75;
			vec2 r = b+rv - xf;
			float d = dot(r,r);
			
			if(d<res.x) {
				res.y = res.x;
				res.x = d;
			} else if(d<res.y) {
				res.y = d;
			}
		}
	}
    
	res = sqrt(res);
	return 1.-smoothstep(-.1, .1, res.y-res.x);
}

// Function 309
float positiveSimplexNoise(vec2 uv) {return abs(simplexNoise(uv));}

// Function 310
float Noise11(float x)
{
    float p = floor(x);
    float f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix( hash11(p), hash11(p + 1.0), f);

}

// Function 311
float smoothNoise(int x,int y)
{
    return noise(x,y)/4.0+(noise(x+1,y)+noise(x-1,y)+noise(x,y+1)+noise(x,y-1))/8.0+(noise(x+1,y+1)+noise(x+1,y-1)+noise(x-1,y+1)+noise(x-1,y-1))/16.0;
}

// Function 312
float getNoise(vec2 uv, float t){
    
    //given a uv coord and time - return a noise val in range 0 - 1
    //using ashima noise
    
    //add time to y position to make noise field move upwards
    
    float TRAVEL_SPEED = 1.5;
    
    //octave 1
    float SCALE = 2.0;
    float noise = snoise( vec3(uv.x*SCALE ,uv.y*SCALE - t*TRAVEL_SPEED , 0));
    
    //octave 2 - more detail
    SCALE = 6.0;
    noise += snoise( vec3(uv.x*SCALE + t,uv.y*SCALE , 0))* 0.2 ;
    
    //move noise into 0 - 1 range    
    noise = (noise/2. + 0.5);
    
    return noise;
    
}

// Function 313
float noise(vec2 U) {
    return hash(uint(U.x+iResolution.x*U.y));
}

// Function 314
float triNoise(in vec3 p)
{
    float z=1.4;
	float rz = 0.;
    vec3 bp = p;
	for (float i=0.; i<=4.; i++ )
	{
        vec3 dg = tri3(bp*2.);
        p += dg;

        bp *= 1.8;
		z *= 1.5;
		p *= 1.2;
           
        rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
        bp += 0.14;
	}
	return rz;
}

// Function 315
float Noisefv3 (vec3 p)
{
  vec4 t;
  vec3 ip, fp;
  ip = floor (p);
  fp = fract (p);
  fp *= fp * (3. - 2. * fp);
  t = mix (Hashv4v3 (ip), Hashv4v3 (ip + vec3 (0., 0., 1.)), fp.z);
  return mix (mix (t.x, t.y, fp.x), mix (t.z, t.w, fp.x), fp.y);
}

// Function 316
vec2 Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
//	vec3 f2 = f*f; f = f*f2*(10.0-15.0*f+6.0*f2);

	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z);

	vec4 rg = textureLod( iChannel0, (uv+f.xy+0.5)/256.0, 0.0 );

	return mix( rg.yw, rg.xz, f.z );
}

// Function 317
float noise(vec3 x) 
{ 
    vec3 s = noiseSpeed*iTime;
    x *= noiseScale.xyz;
    x.xy *= rot(x.z*distortionTwist);
    float texZ = texture(iChannel0, vec2(x.z*noiseZDistortionScale, s.z)).x;
    float tex = texture(iChannel0, x.xy + vec2(texZ-0.5)*noiseZDistortion + s.xy).x;   
    return tex;
}

// Function 318
vec4 noised( in vec3 x )
{
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);

    float n = p.x + 317.0*p.y + 157.0*p.z;
    
    float a = hash1(n+0.0);
    float b = hash1(n+1.0);
    float c = hash1(n+317.0);
    float d = hash1(n+318.0);
    float e = hash1(n+157.0);
	float f = hash1(n+158.0);
    float g = hash1(n+474.0);
    float h = hash1(n+475.0);

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( -1.0+2.0*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z), 
                      2.0* du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                                      k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                                      k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) );
}

// Function 319
float voronoi( vec2 p )
{
     vec2 g = floor( p );
     vec2 f = fract( p );
    
     float distanceFromPointToCloestFeaturePoint = 1.0;
     for( int y = -1; y <= 1; ++y )
     {
          for( int x = -1; x <= 1; ++x )
          {
               vec2 latticePoint = vec2( x, y );
               float h = distance( latticePoint + hash( g + latticePoint), f );
		  
		distanceFromPointToCloestFeaturePoint = min( distanceFromPointToCloestFeaturePoint, h ); 
          }
     }
    
     return 1.0 - sin(distanceFromPointToCloestFeaturePoint);
}

// Function 320
v0 noise(in vec2 p
){const v0 K1 = 0.366025404 // (sqrt(3)-1)/2;
 ;const v0 K2 = 0.211324865 // (3-sqrt(3))/6;
 ;vec2 i = floor( p + (p.x+p.y)*K1)
 ;vec2 a = p - i + (i.x+i.y)*K2
 ;vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0) //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
 ;vec2 b = a - o + K2
 ;vec2 c = a - 1.0 + 2.0*K2
 ;vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 )
 ;vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)))
 ;return dot( n, vec3(70.0));}

// Function 321
float noise( in vec3 p )
{
    vec3 fl = floor( p );
    vec3 fr = fract( p );
    fr = fr * fr * ( 3.0 - 2.0 * fr );

    float n = fl.x + fl.y * 157.0 + 113.0 * fl.z;
    return mix( mix( mix( hash( n +   0.0), hash( n +   1.0 ), fr.x ),
                     mix( hash( n + 157.0), hash( n + 158.0 ), fr.x ), fr.y ),
                mix( mix( hash( n + 113.0), hash( n + 114.0 ), fr.x ),
                     mix( hash( n + 270.0), hash( n + 271.0 ), fr.x ), fr.y ), fr.z );
}

// Function 322
float perlin_2d(vec2 p)
{
    vec2 a = floor(p);
    vec2 d = p - a.xy;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = vec4(a.x, a.x + 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.yyyy;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));

    vec4 o3 = o2 * d.y + o1 * (1.0 - d.y);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return(o4.y * 0.0 + o4.x * 1.0);
}

// Function 323
float noise (vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p); f = f*f*(3.-2.*f); // smoothstep

    float v= mix( mix( mix(hash(i+vec3(0,0,0)),hash(i+vec3(1,0,0)),f.x),
                       mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x), f.y), 
                  mix( mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),
                       mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x), f.y), f.z);
	return   MOD==0 ? v
	       : MOD==1 ? 2.*v-1.
           : MOD==2 ? abs(2.*v-1.)
                    : 1.-abs(2.*v-1.);
}

// Function 324
vec3 voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

	float id, le;

    float md = 10.0;
    
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g1 = n + vec2(float(i),float(j));
        vec3 rr = rand3( g1 );
		vec2 o = g1 + rr.xy;
        vec2 r = x - o;
        float d = dot(r,r);
        float z = rr.z;
        
        #if LEVEL>0
        if( z<0.75 )
        #endif            
        {
            if( d<md )
            {
                md = d;
                id = z + g1.x + g1.y*7.0;
                le = 0.0;
            }
        }
        #if LEVEL>0
        else
        {
            for( int l=ZERO; l<2; l++ )
            for( int k=ZERO; k<2; k++ )
            {
                vec2 g2 = g1 + vec2(float(k),float(l))/2.0;
                rr = rand3( g2 );
                o = g2 + rr.xy/2.0;
                r = x - o;
                d = dot(r,r);
                z = rr.z;
                #if LEVEL>1
                if( z<0.8 )
                #endif                    
                {
                    if( d<md )
                    {
                        md = d;
                        id = z + g2.x + g2.y*7.0;
                        le = 1.0;
                    }
                }
                #if LEVEL>1
                else
                {
                    for( int n=ZERO; n<2; n++ )
                    for( int m=ZERO; m<2; m++ )
                    {
                        vec2 g3 = g2 + vec2(float(m),float(n))/4.0;
                        rr = rand3( g3 );
                        o = g3 + rr.xy/4.0;
                        r = x - o;
                        d = dot(r,r);
                        z = rr.z;

                        #if LEVEL>2
                        if( z<0.8 )
                        #endif                    
                        {
                            if( d<md )
                            {
                                md = d;
                                id = z + g3.x + g3.y*7.0;
                                le = 2.0;
                            }
                        }
                        #if LEVEL>2
                        else
                        {
                            for( int t=ZERO; t<2; t++ )
                            for( int s=ZERO; s<2; s++ )
                            {
                                vec2 g4 = g3 + vec2(float(s),float(t))/8.0;
                                rr = rand3( g4 );
                                o = g4 + rr.xy/8.0;
                                r = x - o;
                                d = dot(r,r);
                                z = rr.z;

                                if( d<md )
                                {
                                    md = d;
                                    id = z + g4.x + g4.y*7.0;
                                    le = 3.0;
                                }
                            }
                        }
                        #endif
                    }
                }
                #endif
            }
        }
        #endif        
    }

    return vec3( md, le, id );
}

// Function 325
vec4 texNoise(vec2 uv,sampler2D tex ){ float f = 0.; f+=texture(tex, uv*.125).r*.5; f+=texture(tex,uv*.25).r*.25; //This function does q perlon equivalent to the texNoise perlin texsture we get in bonzomatic shader editor, thankx to yx for this
                       f+=texture(tex,uv*.5).r*.125; f+=texture(tex,uv*1.).r*.125; f=pow(f,1.2);return vec4(f*.45+.05);}

// Function 326
float noise3(vec3 v){
    v *= 64.; // emulates 64x64 noise texture
    return ( perlin(v) + perlin(v+.5) )/2.;
}

// Function 327
float noise(vec2 p) {
  float a = sin(p.x);
  float b = sin(p.y);
  float c = 0.5 + 0.5*cos(p.x + p.y);
  float d = mix(a, b, c);
  return d;
}

// Function 328
float noiseMulti( in vec3 pos)
{
    vec3 q = pos;
    float f  = 0.5000*noise( q ); q = m*q*2.01;
    f += 0.2500*noise( q ); q = m*q*2.02;
    f += 0.1250*noise( q ); q = m*q*2.03;
    f += 0.0625*noise( q ); q = m*q*2.01;

    float w = 0.5000+0.25+0.125+0.0625;
    return f/w;
}

// Function 329
vec3 noise(vec3 p){float m = mod(p.z,1.0);float s = p.z-m; float sprev = s-1.0;if (mod(s,2.0)==1.0) { s--; sprev++; m = 1.0-m; };return mix(texture(iChannel0,p.xy/iChannelResolution[0].xy+noise(sprev).yz).xyz,texture(iChannel0,p.xy/iChannelResolution[0].xy+noise(s).yz).xyz,m);}

// Function 330
vec1 noise1(vec1 p){return mx(herm32(fract(p)),hash2(floor(p)));}

// Function 331
float fractalblobnoise(vec2 v, float s)
{
    float val = 0.;
    const float n = 4.;
    for(float i = 0.; i < n; i++)
    	val += pow(0.5, i+1.) * blobnoise(exp2(i) * v + vec2(0, T), s);
    return val;
}

// Function 332
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                        mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                        mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
    return res;
}

// Function 333
float blugausnoise2(vec2 c1) {
    float nrand1 = n4rand_ss(c1);
    float nrand0 = n4rand_ss(vec2(c1.x- 1.,c1.y));
    float nrand2 = n4rand_ss(vec2(c1.x+ 1.,c1.y));
    return 2.0* nrand1- 0.5* (nrand0+ nrand2);
}

// Function 334
float tweaknoise( vec3 p , bool step) {
    float d1 = smoothstep(grad/2.,-grad/2.,length(p)-.5),
          d2 = smoothstep(grad/2.,-grad/2.,abs(p.z)-.5),
          d=d1;
#if NOISE==1 // 3D Perlin noise
    float v = fbm(scale*p);
#elif NOISE==2 // Worley noise
    float v = (.9-scale*worley(scale*p).x);
#elif NOISE>=3 // trabeculum 3D
  #if !VARIANT
    d = (1.-d1)*d2; 
  #endif
    if (d<0.5) return 0.;
    grad=.8, scale = 10., thresh=.7+.3*(cos(.5*iTime)+.36*cos(.5*3.*iTime))/1.36;
    vec4 w=scale*worley(scale*p); 
    float v=1.-1./(1./(w.z-w.x)+1./(w.a-w.x)); // formula (c) Fabrice NEYRET - BSD3:mention author.
#endif
    
    return (true)? smoothstep(thresh-grad/2.,thresh+grad/2.,v*d) : v*d;
}

// Function 335
float noise1d(float n) {
    return fract(cos(n*89.42)*343.42);
}

// Function 336
float valueNoiseSimple(vec2 vl) {

   const vec2 helper = vec2(0., 1.);
    vec2 interp = smoothstep(vec2(0.), vec2(1.), fract(vl));
    vec2 grid = floor(vl);

    float rez = mix(mix(rand2(grid + helper.xx),
                        rand2(grid + helper.yx),
                        interp.x),
                    mix(rand2(grid + helper.xy),
                        rand2(grid + helper.yy),
                        interp.x),
                    interp.y);
#if SHARP_MODE==1    
    return abs(rez*2. -1.);
#else
    return rez;
#endif
}

// Function 337
vec3 voronoi(vec2 pos, vec2 scale, float jitter, float phase, float seed)
{
     // voronoi based on Inigo Quilez: https://archive.is/Ta7dm
    const float kPI2 = 6.2831853071;
    pos *= scale;
    vec2 i = floor(pos);
    vec2 f = pos - i;

    // first pass
    vec2 minPos, tilePos;
    float minDistance = 1e+5;
    for (int y=-1; y<=1; y++)
    {
        for (int x=-1; x<=1; x++)
        {
            vec2 n = vec2(float(x), float(y));
            vec2 cPos = hash2D(mod(i + n, scale) + seed) * jitter;
            cPos = 0.5 * sin(phase + kPI2 * cPos) + 0.5;
            vec2 rPos = n + cPos - f;

            float d = dot(rPos, rPos);
            if(d < minDistance)
            {
                minDistance = d;
                minPos = rPos;
                tilePos = cPos;
            }
        }
    }

    // second pass, distance to edges
    minDistance = 1e+5;
    for (int y=-2; y<=2; y++)
    {
        for (int x=-2; x<=2; x++)
        { 
            vec2 n = vec2(float(x), float(y));
            vec2 cPos = hash2D(mod(i + n, scale) + seed) * jitter;
            cPos = 0.5 * sin(phase + kPI2 * cPos) + 0.5;
            vec2 rPos = n + cPos - f;
            
            vec2 v = minPos - rPos;
            if(dot(v, v) > 1e-5)
                minDistance = min(minDistance, dot( 0.5 * (minPos + rPos), normalize(rPos - minPos)));
        }
    }

    return vec3(minDistance, tilePos);
}

// Function 338
vec3 simplexRotD(vec2 u, vec2 p, float rot
){vec3 x,y;vec2 d0,d1,d2,p0,p1,p2;NoiseHead(u,x,y,d0,d1,d2,p0,p1,p2)
 ;x=mod(vec3(p0.x,p1.x,p2.x),p.x)
 ;y=mod(vec3(p0.y,p1.y,p2.y),p.y);x=x+.5*y//with    TilingPeriod (p)
 ;return NoiseDer(x,y,rot,d0,d1,d2);}

// Function 339
float noise(in vec2 p){

    float res=0.;
    float f=1.;
	for( int i=0; i< 3; i++ ) 
	{		
        p=m2*p*f+.6;     
        f*=1.2;
        res+=sin(p.x+sin(2.*p.y));
	}        	
	return res/3.;
}

// Function 340
float tnoise(vec2 p)
{
    return textureLod(iChannel3, p, 0.0).x;
}

// Function 341
float layeredNoise12(vec2 x)
{
 	float sum = 0.0;
    float maxValue = 0.0;

    for (float i = 1.0; i <= 2.0; i *= 2.0)
    {
        float noise = modifiedVoronoiNoise12(x * i) / i;
     	sum += noise;
        maxValue += 1.0 / i;
    }

    return sum / maxValue;
}

// Function 342
float valueNoiseSimple(vec2 vl, out vec4 der) {
   const vec2 minStep = vec2(1., 0. );

   vec2 grid = floor(vl);

    float s = rand2(grid);
    float t = rand2(grid + minStep);
    float u = rand2(grid + minStep.yx);
    float v = rand2(grid + minStep.xx);
    
    float fractX = fract(vl.x);
    float x1 = interFunc2(fractX);
    
    float fractY = fract(vl.y);
    float y = interFunc2(fractY);
    
    float k3 = s - t - u + v;
    float k2 = t - s;
    float k1 = u - s;
    
    float interpY = s + k2 * x1  + k1 * y + k3 * x1 * y;
    
    /* Inspired by https://www.iquilezles.org/www/articles/morenoise/morenoise.htm */
    
    der.z = (k2 + k3 * y) * (120.0 * fractX * fractX - 180.0 * fractX + 60.) * fractX;
    der.w = (k1 + k3 * x1) * (120.0 * fractY * fractY - 180.0 * fractY + 60.) * fractY;
    
    der.x = (k2 + k3 * y) * (30.0 * fractX * fractX - 60.0 * fractX + 30.) * fractX * fractX;
    der.y = (k1 + k3 * x1) * (30.0 * fractY * fractY - 60.0 * fractY + 30.) * fractY * fractY;
    
    return interpY;//s + k2 * x1  + k1 * y + k3 * x1 * y;
}

// Function 343
float noiseblend(vec3 p)
{
    vec2 off = vec2(1.,0.);
    return mix(	mix(	mix(noise(p), noise(p+off.xyy),fract(p.x)),
           				mix(noise(p+off.yxy), noise(p+off.xxy),fract(p.x)),fract(p.y)),
               	mix(	mix(noise(p+off.yyx), noise(p+off.xyx),fract(p.x)),
           				mix(noise(p+off.yxx), noise(p+off.xxx),fract(p.x)),fract(p.y))
               ,fract(p.z));
    
}

// Function 344
float fractalNoiseLow(vec2 vl) {
    float persistance = 2.;
    float amplitude = 1.2;
    float rez = 0.0;
    vec2 p = vl;
    float norm = 0.0;
    for (int i = 0; i < OCTAVES - 3; i++) {
        norm += amplitude;
        rez += amplitude * valueNoiseSimpleLow(p);
        amplitude /= persistance;
        p *= persistance;
    }
    return rez / norm;
}

// Function 345
float simplex3d(vec3 p) {
	 /* 1. find current tetrahedron T and it's four vertices */
	 /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
	 /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/
	 
	 /* calculate s and x */
	 vec3 s = floor(p + dot(p, vec3(F3)));
	 vec3 x = p - s + dot(s, vec3(G3));
	 
	 /* calculate i1 and i2 */
	 vec3 e = step(vec3(0.0), x - x.yzx);
	 vec3 i1 = e*(1.0 - e.zxy);
	 vec3 i2 = 1.0 - e.zxy*(1.0 - e);
	 	
	 /* x1, x2, x3 */
	 vec3 x1 = x - i1 + G3;
	 vec3 x2 = x - i2 + 2.0*G3;
	 vec3 x3 = x - 1.0 + 3.0*G3;
	 
	 /* 2. find four surflets and store them in d */
	 vec4 w, d;
	 
	 /* calculate surflet weights */
	 w.x = dot(x, x);
	 w.y = dot(x1, x1);
	 w.z = dot(x2, x2);
	 w.w = dot(x3, x3);
	 
	 /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
	 w = max(0.6 - w, 0.0);
	 
	 /* calculate surflet components */
	 d.x = dot(random3(s), x);
	 d.y = dot(random3(s + i1), x1);
	 d.z = dot(random3(s + i2), x2);
	 d.w = dot(random3(s + 1.0), x3);
	 
	 /* multiply d by w^4 */
	 w *= w;
	 w *= w;
	 d *= w;
	 
	 /* 3. return the sum of the four surflets */
	 return dot(d, vec4(52.0));
}

// Function 346
float y_noise(float y)
{
    y += texture(iChannel0, vec2(y * 0.05, 0.0)).x * 0.08;
    y += texture(iChannel0, vec2(y * 0.1, 0.0)).x * 0.04;
    y += texture(iChannel0, vec2(y * 0.2, 0.0)).x * 0.02;
    return y;
}

// Function 347
float Voronoi(vec2 p){
    
	vec2 g = floor(p), o;
	p -= g;// p = fract(p);
	
	vec2 d = vec2(1); // 1.4, etc.
    
	for(int y = -1; y <= 1; y++){
		for(int x = -1; x <= 1; x++){
            
			o = vec2(x, y);
            o += hash22(g + o) - p;
            
			float h = dot(o, o);
            d.y = max(d.x, min(d.y, h)); 
            d.x = min(d.x, h);            
		}
	}
    
    
	
	//return sqrt(d.y) - sqrt(d.x);
    return (d.y - d.x); // etc.
}

// Function 348
float simplex3d_fractal(vec3 m) {
    return   0.5333333*simplex3d(m*rot1)
			+0.2666667*simplex3d(2.0*m*rot2)
			+0.1333333*simplex3d(4.0*m*rot3)
			+0.0666667*simplex3d(8.0*m);
}

// Function 349
float trigNoise3D(in vec3 p){

    // 3D transformation matrix.
    const mat3 m3RotTheta = mat3(0.25, -0.866, 0.433, 0.9665, 0.25, -0.2455127, -0.058, 0.433, 0.899519 )*1.5;
  
	float res = 0.;

    float t = trig3(p*PI);
	p += (t - iTime*0.25);
    p = m3RotTheta*p;
    //p = (p+0.7071)*1.5;
    res += t;
    
    t = trig3(p*PI); 
	p += (t - iTime*0.25)*0.7071;
    p = m3RotTheta*p;
     //p = (p+0.7071)*1.5;
    res += t*0.7071;

    t = trig3(p*PI);
	res += t*0.5;
	 
	return res/2.2071;
}

// Function 350
float noiseSpace(vec3 ray,vec3 pos,float r,mat3 mr,float zoom,vec3 subnoise)
{
  	float b = dot(ray,pos);
  	float c = dot(pos,pos) - b*b;
    
    vec3 r1=vec3(0.0);
    
    float s=0.0;
    float d=0.0625*1.5;
    float d2=zoom/d;

	float rq=r*r;
    float l1=sqrt(abs(r-c));
    r1= (ray*(b-l1)-pos)*mr;

    r1*=d2;
    s+=abs(noise3(vec3(r1+subnoise))*d);
    s+=abs(noise3(vec3(r1*0.5+subnoise))*d*2.0);
    s+=abs(noise3(vec3(r1*0.25+subnoise))*d*4.0);
    return s;
}

// Function 351
float noise3D(vec3 p)
{
	return fract(sin(dot(p ,vec3(12.9898,78.233,128.852))) * 43758.5453)*2.0-1.0;
}

// Function 352
float Voronoi(in vec2 p){
    
	vec2 g = floor(p), o; p -= g; // Cell ID, offset variable, and relative cell postion.
	
	vec3 d = vec3(1); // 1.4, etc. "d.z" holds the distance comparison value.
    
	for(int y=-1; y<=1; y++){
		for(int x=-1; x<=1; x++){
            
			o = vec2(x, y); // Grid cell ID offset.
            o += hash22(g + o) - p; // Random offset.
			
            // Regular squared Euclidean distance.
            d.z = dot(o, o); 
            // Adding some radial variance as we sweep around the circle. It's an old
            // trick to draw flowers and so forth. Three petals is reminiscent of a
            // triangle, which translates roughly to a blocky appearance.
            d.z *= cos(atan(o.y, o.x)*3. - 3.14159/2.)*.333 + .667;
            //d.z *= (1. -  tri(atan(o.y, o.x)*3./6.283 + .25)*.5); // More linear looking.
            
            d.y = max(d.x, min(d.y, d.z)); // Second order distance.
            d.x = min(d.x, d.z); // First order distance.
                      
		}
	}

    // A bit of science and experimentation.
    return d.y*.5 + (d.y-d.x)*.5; // Range: [0, 1]... Although, I'd check. :)
    
    //return d.y; // d.x, d.y - d.x, etc.
    
    
}

// Function 353
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
#ifndef HIGH_QUALITY_NOISE
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0. ).yx;
#else
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z);
	vec2 rg1 = textureLod( iChannel0, (uv+ vec2(0.5,0.5))/256.0, 0. ).yx;
	vec2 rg2 = textureLod( iChannel0, (uv+ vec2(1.5,0.5))/256.0, 0. ).yx;
	vec2 rg3 = textureLod( iChannel0, (uv+ vec2(0.5,1.5))/256.0, 0. ).yx;
	vec2 rg4 = textureLod( iChannel0, (uv+ vec2(1.5,1.5))/256.0, 0. ).yx;
	vec2 rg = mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
#endif	
	return mix( rg.x, rg.y, f.z );
}

// Function 354
float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
		+ i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// Function 355
float noise(vec3 n)
{
 	return snoise(n)*.6+snoise(n*2.)*.4;
}

// Function 356
vec4 bccNoiseDerivatives_XYZ(vec3 X) {
    X = dot(X, vec3(2.0/3.0)) - X;
    
    vec4 result = bccNoiseDerivativesPart(X) + bccNoiseDerivativesPart(X + 144.5);
    
    return vec4(dot(result.xyz, vec3(2.0/3.0)) - result.xyz, result.w);
}

// Function 357
float iqnoise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
		
	float k = 1.0+63.0*pow(1.0-v,6.0);
	
	float va = 0.0;
	float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = vec2( float(i),float(j) );
		vec3 o = hash3( p + g )*vec3(u,u,1.0);
		vec2 r = g - f + o.xy;
		float d = dot(r,r);
		float ww = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
		va += o.z*ww;
		wt += ww;
    }
	
    return va/wt;
}

// Function 358
vec4 AchNoise3D(vec3 x)
{
    vec3 p = floor(x);
    vec3 fr = smoothstep(0.0, 1.0, fract(x));

    vec4 L1C1 = mix(hash43(p+nbs[0]), hash43(p+nbs[2]), fr.x);
    vec4 L1C2 = mix(hash43(p+nbs[1]), hash43(p+nbs[3]), fr.x);
    vec4 L1C3 = mix(hash43(p+nbs[4]), hash43(p+nbs[6]), fr.x);
    vec4 L1C4 = mix(hash43(p+nbs[5]), hash43(p+nbs[7]), fr.x);
    vec4 L2C1 = mix(L1C1, L1C2, fr.y);
    vec4 L2C2 = mix(L1C3, L1C4, fr.y);
    return mix(L2C1, L2C2, fr.z);
}

// Function 359
float simplex(vec3 v){
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

  float n_ = 0.142857142857;
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
  }

// Function 360
float EyeNoise( in float x )
{
    float p = floor(x);
    float f = fract(x);
	f = clamp(pow(f, 7.0), 0.0,1.0);
	//f = f*f*(3.0-2.0*f);
    return mix(Hash(p), Hash(p+1.0), f);
}

// Function 361
float triNoise3D(in vec3 p, in float spd)
{
    float z=1.4;
	float rz = 0.;
    vec3 bp = p;
	for (float i=0.; i<=3.; i++ )
	{
        vec3 dg = tri3(bp*2.);
        p += (dg+iTime*.1*spd);

        bp *= 1.8;
		z *= 1.5;
		p *= 1.2;
        //p.xz*= m2;
        
        rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
        bp += 0.14;
	}
	return rz;
}

// Function 362
float GetWaterNoise(vec3 position, float time)
{
    return WaterTurbulence * fbm_4(position / 15.0 + time / 3.0);
}

// Function 363
float WaveletNoise(vec2 p, float phase, float scaleFactor) {
    float d = 0.;
    float scale = 1.;
    float mag=0.;
    for(float i=0.; i<4.; i++) {
        d += GetWavelet(p, phase, scale);
        p = p*mat2(.54,-.84, .84, .54)+i;
        mag += 1./scale;
        scale *= scaleFactor; 
    }
    d /= mag;
    return d;
}

// Function 364
vec3 Noisev3v2 (vec2 p)
{
  vec4 h;
  vec3 g;
  vec2 ip, fp, ffp;
  ip = floor (p);
  fp = fract (p);
  ffp = fp * fp * (3. - 2. * fp);
  h = Hashv4f (dot (ip, cHashA3.xy));
  g = vec3 (h.y - h.x, h.z - h.x, h.x - h.y - h.z + h.w);
  return vec3 (h.x + dot (g.xy, ffp) + g.z * ffp.x * ffp.y,
     30. * fp * fp * (fp * fp - 2. * fp + 1.) * (g.xy + g.z * ffp.yx));
}

// Function 365
vec2 Voronoi(const in vec2 mPos)
        {
            vec2 n = floor(mPos);
            vec2 f = fract(mPos);

            vec3 m = vec3(8.0);
            for(int j = -1; j <= 1; ++j)
            for(int i = -1; i <= 1; ++i)
            {
                vec2  g = vec2( float(i), float(j));
                vec2  o = vec2(Hash2D(n + g));
                vec2  r = g - f + o;
                float d = dot(r, r);
                if(d < m.x)
                    m = vec3(d, o);
            }

            return vec2(sqrt(abs(m.x)), m.y + m.z);
        }

// Function 366
vec4 voronoi(vec2 p, float roundness, out vec2 smallestp) {
  
  int C = 2;
  
  float offset = 1.5;
  
  vec2 closest;
  float mindist = float(C*C);
  float secondmin = mindist;
  vec2 secclosest;
  
  
  for(int x = -C; x <= C; x++)
    for(int y = -C; y <= C; y++) {
      vec2 vp = point(floor(p) + vec2(x,y)) * offset;
      float d = dist(vp + floor(p) + vec2(x,y),p);
      
      float size = pow(random(vp) / 4. + 0.5,roundness);
      float comp = d * size;
      
      if(mindist > comp) {
        secclosest = closest;
        secondmin = mindist;
        
        closest = vp + vec2(x+y);
        smallestp = floor(p) + vec2(x, y);
        mindist = comp;
      }
      else if(secondmin > comp) {
        secclosest = vp + vec2(x+y);
        secondmin = comp;
      }
    }
    
  return vec4(fract(closest.x),fract(closest.y),mindist,(secondmin - mindist));
}

// Function 367
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

	f =  f*f*(3.0-2.0*f);

	float a = textureLod( iChannel0, (p+vec2(0.5,0.5))/256.0, 0.0 ).x;
	float b = textureLod( iChannel0, (p+vec2(1.5,0.5))/256.0, 0.0 ).x;
	float c = textureLod( iChannel0, (p+vec2(0.5,1.5))/256.0, 0.0 ).x;
	float d = textureLod( iChannel0, (p+vec2(1.5,1.5))/256.0, 0.0 ).x;

	return mix( mix( a, b, f.x ), mix( c, d, f.x ), f.y );

}

// Function 368
float noise( in vec2 p )
{
    ivec2 i = ivec2(floor( p ));
    vec2 f = fract( p );
	vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( dot( grad( i+ivec2(0,0) ), f-vec2(0.0,0.0) ), 
                     dot( grad( i+ivec2(1,0) ), f-vec2(1.0,0.0) ), u.x),
                mix( dot( grad( i+ivec2(0,1) ), f-vec2(0.0,1.0) ), 
                     dot( grad( i+ivec2(1,1) ), f-vec2(1.0,1.0) ), u.x), u.y);
}

// Function 369
vec3 voronoi( in vec2 x )
{
    ivec2 p = ivec2(floor( x ));
    vec2 f = fract(x);

    ivec2 mb = ivec2(0);
    vec2 mr = vec2(0.0);
    vec2 mg = vec2(0.0);

    float md = 8.0;
    for(int j=-1; j<=1; ++j)
    for(int i=-1; i<=1; ++i)
    {
        ivec2 b = ivec2( i, j );
        vec2  r = vec2( b ) + noise( vec2(p + b) ) - f;
        vec2 g = vec2(float(i),float(j));
		vec2 o = vec2(noise( vec2(p) + g ));
        float d = length(r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    md = 8.0;
    for(int j=-2; j<=2; ++j)
    for(int i=-2; i<=2; ++i)
    {
        ivec2 b = ivec2( i, j );
        vec2 r = vec2( b ) + noise( vec2(p + b) ) - f;


        if( length(r-mr)>0.00001 )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }
    return vec3( md, mr );
}

// Function 370
float noise( float x )
{    
	return fract(sin(1371.1*x)*43758.5453);
}

// Function 371
float noise( in vec3 x )
{
    vec3 p = floor( x );
    vec3 f = fract( x );

    float res = 1.0;
    for( int k=-1; k<=1; k++ )
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec3 b = vec3( float(i), float(j), float(k) );
        vec3 r = vec3( b ) - f + random3f( p + b );
        float d = dot( r, r );
		res = min(d, res);			
    }

    return res;
}

// Function 372
float trigNoise3D(in vec3 p){

    // 3D transformation matrix.
    const mat3 m3RotTheta = mat3(0.25, -0.866, 0.433, 0.9665, 0.25, -0.2455127, -0.058, 0.433, 0.899519 )*1.5;
  
	float res = 0.;

    float t = trig3(p*3.14159265);
	p += (t);
    p = m3RotTheta*p;
    //p = (p+0.7071)*1.5;
    res += t;
    
    t = trig3(p*3.14159265); 
	p += (t)*0.7071;
    p = m3RotTheta*p;
     //p = (p+0.7071)*1.5;
    res += t*0.7071;

    t = trig3(p*3.14159265);
	res += t*0.5;
	 
	return res/2.2071;
}

// Function 373
float noise( vec2 co ){
    return fract( sin( dot( co.xy, vec2( 12.9898, 78.233 ) ) ) * 43758.5453 );
}

// Function 374
vec4 noised( vec3 x )
{
	// http://www.iquilezles.org/www/articles/gradientnoise/gradientnoise.htm
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);

    float a = hash( p+vec3(0,0,0) );
    float b = hash( p+vec3(1,0,0) );
    float c = hash( p+vec3(0,1,0) );
    float d = hash( p+vec3(1,1,0) );
    float e = hash( p+vec3(0,0,1) );
    float f = hash( p+vec3(1,0,1) );
    float g = hash( p+vec3(0,1,1) );
    float h = hash( p+vec3(1,1,1) );

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( -1.0+2.0*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z), 
                      2.0* du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                                      k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                                      k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) ).yzwx;
}

// Function 375
float craterNoise3D(in vec3 p){
	
    
    const float radius = 0.25;
    const float slope = .095;
    const float frequency = 2.35;
    const float depth = -0.982;
    const float rimWidth = 0.125;
    
	float fractal = fbm(p * frequency * 2.0, 4) * 0.07;
	float cell = gpuCellNoise3D((p * frequency) + fractal ).x;
	float r = radius + fractal;
	float crater = smoothstep(slope, r, cell);
	  	  crater = mix(depth, crater, crater);
	float rim = 1.0 - smoothstep(r, r + rimWidth, cell);
	      crater = rim - (1.0 - crater);
	return crater * 0.08;
}

// Function 376
vec4 ThornVoronoi( vec3 p, out float which)
{
    
    vec2 f = fract(p.xz);
    p.xz = floor(p.xz);
	float d = 1.0e10;
    vec3 id = vec3(0.0);
    
	for (int xo = -1; xo <= 1; xo++)
	{
		for (int yo = -1; yo <= 1; yo++)
		{
            vec2 g = vec2(xo, yo);
            vec2 n = textureLod(iChannel3,(p.xz + g+.5)/256.0, 0.0).xy;
            n = n*n*(3.0-2.0*n);
            
			vec2 tp = g + .5 + sin(p.y + 1.2831 * (n * time*.5)) - f;
            float d2 = dot(tp, tp);
			if (d2 < d)
            {
                // 'id' is the colour code for each thorn
                d = d2;
                which = n.x+n.y*3.0;
                id = vec3(tp.x, p.y, tp.y);
            }
		}
	}

    return vec4(id, 1.35-pow(d, .17));
}

// Function 377
vec4 getNoise(in vec2 uv, in vec2 noiseUVscale, in float noiseBrightness, float zoffset)
{
     vec4 noise = texture(iChannel1, vec3(uv * noiseUVscale,zoffset + iTime * 0.1)) * noiseBrightness;
     return noise;
}

// Function 378
float SimplexNoiseRaw(vec3 pos)
{
    const float K1 = 0.333333333;
    const float K2 = 0.166666667;
    
    vec3 i = floor(pos + (pos.x + pos.y + pos.z) * K1);
    vec3 d0 = pos - (i - (i.x + i.y + i.z) * K2);
    
    vec3 e = step(vec3(0.0), d0 - d0.yzx);
	vec3 i1 = e * (1.0 - e.zxy);
	vec3 i2 = 1.0 - e.zxy * (1.0 - e);
    
    vec3 d1 = d0 - (i1 - 1.0 * K2);
    vec3 d2 = d0 - (i2 - 2.0 * K2);
    vec3 d3 = d0 - (1.0 - 3.0 * K2);
    
    vec4 h = max(0.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
    vec4 n = h * h * h * h * vec4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0)));
    
    return dot(vec4(31.316), n);
}

// Function 379
float iqnoise( vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
		
	float k = 1.0+63.0*pow(1.0-v,4.0);
	
	float va = 0.0;
	float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = vec2( float(i),float(j) );
		vec3 o = hash3( p + g )*vec3(u,u,1.0);
		vec2 r = g - f + o.xy;
		float d = dot(r,r);
		float ww = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
		va += o.z*ww;
		wt += ww;
    }
	
    return va/wt;
}

// Function 380
float noise( float x, float y )
{
    
    vec2 p = vec2(x,y);
    float f = 0.0;
    f += 0.500000*(0.5+0.5*noise( p )); p = m*p*2.02;
    f += 0.250000*(0.5+0.5*noise( p )); p = m*p*2.03;
    f += 0.125000*(0.5+0.5*noise( p )); p = m*p*2.01;
    f += 0.062500*(0.5+0.5*noise( p )); p = m*p*2.04;
    f += 0.031250*(0.5+0.5*noise( p )); p = m*p*2.01;
    f += 0.015625*(0.5+0.5*noise( p ));
    return f/0.96875;
}

// Function 381
vec3 blobnoisenrm(vec2 v, float s)
{
    vec2 e = vec2(.01,0);
    return normalize(
           vec3(blobnoise(v + e.xy, s) - blobnoise(v -e.xy, s),
                blobnoise(v + e.yx, s) - blobnoise(v -e.yx, s),
                //e.x));
                1.0));
}

// Function 382
float FoamNoise(vec3 pos)
{
    vec2 s = iTime * vec2(0.01);
    vec2 t1 = texture(iChannel1, pos.xz*0.03 + s).xz-0.5;
    float t2 = texture(iChannel1, pos.xz*0.06 - s + t1*0.1).y;
    
    return t2;
}

// Function 383
float snoise(vec2 v)
  {
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
// First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
		+ i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// Function 384
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 385
vec4 bccNoiseDerivatives_PlaneFirst(vec3 X) {
    
    // Not a skew transform.
    mat3 orthonormalMap = mat3(
        0.788675134594813, -0.211324865405187, -0.577350269189626,
        -0.211324865405187, 0.788675134594813, -0.577350269189626,
        0.577350269189626, 0.577350269189626, 0.577350269189626);
    
    X = orthonormalMap * X;
    vec4 result = bccNoiseDerivativesPart(X) + bccNoiseDerivativesPart(X + 144.5);
    
    return vec4(result.xyz * orthonormalMap, result.w);
}

// Function 386
float valuenoise(vec2 uv) {
    vec2 iuv = floor(uv);
    vec2 offset = vec2(0.,1.);
    float v00 = rand(iuv);
    float v01 = rand(iuv+offset.xy);
    float v10 = rand(iuv+offset.yx);
    float v11 = rand(iuv+offset.yy);
    vec2 disp = fract(uv);
    float v0 = cubemix(v00, v01, disp.y);
    float v1 = cubemix(v10, v11, disp.y);
    return cubemix(v0, v1, disp.x) - 0.5;
}

// Function 387
float noise( in vec3 x )
{
	vec3 p = floor(x);
	vec3 f = fract(x);
	
	f = f*f*(3.0-2.0*f);
	float n = p.x + p.y*57.0 + 113.0*p.z;
	return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
				   mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
			   mix(mix( hash(n+113.0), hash(n+114.0),f.x),
				   mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

// Function 388
float noised( in vec2 x ){
    vec2 f = fract(x);
    vec2 u = f*f*(3.0-2.0*f);
  
    vec2 p = floor(x);
	float a = textureLod( iChannel1, (p+vec2(0.5,0.5))*0.00390625, 0.0 ).x;
	float b = textureLod( iChannel1, (p+vec2(1.5,0.5))*0.00390625, 0.0 ).x;
	float c = textureLod( iChannel1, (p+vec2(0.5,1.5))*0.00390625, 0.0 ).x;
	float d = textureLod( iChannel1, (p+vec2(1.5,1.5))*0.00390625, 0.0 ).x;
    
	float res = (a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y);
    res = res - 0.5;
    return res;
}

// Function 389
float smoothnoise(const in vec3 o) 
{
	vec3 p = floor(o);
	vec3 fr = fract(o);
		
	float n = p.x + p.y*101.0 + p.z * 4001.0;

	float a = hash(n+   0.0);
	float b = hash(n+   1.0);
	float c = hash(n+ 101.0);
	float d = hash(n+ 102.0);
	float e = hash(n+4001.0);
	float f = hash(n+4002.0);
	float g = hash(n+4102.0);
	float h = hash(n+4103.0);
	
	vec3 fr2 = fr * fr;
	vec3 fr3 = fr2 * fr;
	
	vec3 t = 3.0 * fr2 - 2.0 * fr3;
		
	return mix(
			    mix( mix(a,b, t.x),
		             mix(c,d, t.x), t.y),
			    mix( mix(e,f, t.x),
		             mix(g,h, t.x), t.y),
			t.z);
}

// Function 390
float noise( in vec2 p ) {
    vec2 i = floor(p), f = fract(p);
	vec2 u = f*f*f*(6.*f*f - 15.*f + 10.);
;
    return mix( mix( dot( hash( i + vec2(0.,0.) ), f - vec2(0.,0.) ), 
                     dot( hash( i + vec2(1.,0.) ), f - vec2(1.,0.) ), u.x),
                mix( dot( hash( i + vec2(0.,1.) ), f - vec2(0.,1.) ), 
                     dot( hash( i + vec2(1.,1.) ), f - vec2(1.,1.) ), u.x), u.y);
}

// Function 391
float triNoise2d(in vec2 p)
{
    float z=2.;
    float z2=1.5;
	float rz = 0.;
    vec2 bp = p;
    rz+= (tri(-time*0.5+p.x*(sin(-time)*0.3+.9)+tri(p.y-time*0.2)))*.7/z;
	for (float i=0.; i<=2.; i++ )
	{
        vec2 dg = tri2(bp*2.)*.8;
        dg *= mm2(time*2.);
        p += dg/z2;

        bp *= 1.7;
        z2 *= .7;
		z *= 2.;
		p *= 1.5;
        p*= m2;
        
        rz+= (tri(p.x+tri(p.y)))/z;
	}
	return rz;
}

// Function 392
vec3 voronoi( in vec2 x )
{
#if 1
    // slower, but better handles big numbers
    vec2 n = floor(x);
    vec2 f = fract(x);
    vec2 h = step(.5,f) - 2.;
    n += h; f -= h;
#else
    vec2 n = floor(x - 1.5);
    vec2 f = x - n;
#endif

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mr;

    float md = 8.0;
    for( int j=0; j<=3; j++ )
    for( int i=0; i<=3; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=0; j<=3; j++ )
    for( int i=0; i<=3; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) // skip the same cell
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec3( md, mr );
}

// Function 393
float gold_noise(in vec2 xy, in float seed)
{
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}

// Function 394
vec2 noiseInterpolate(const in vec2 x) 
{ 
    vec2 x2 = x * x;
    return x2 * x * (x * (x * 6.0 - 15.0) + 10.0); 
}

// Function 395
float pnoise2(vec2 uv, float sz) {
    float u = uv[0]/sz, v = uv[1]/sz, fu = floor(u), fv = floor(v);
    
    float c1 = noise(vec2(fu, fv)*sz); 
    float c2 = noise(vec2(fu, fv+1.0)*sz); 
    float c3 = noise(vec2(fu+1.0, fv+1.0)*sz); 
    float c4 = noise(vec2(fu+1.0, fv)*sz); 
    
    u -= fu; v -= fv;
    
    u = smoothstep(0.0, 1.0, u);
    v = smoothstep(0.0, 1.0, v);
    
    float r1 = c1 + (c2 - c1)*v;
    float r2 = c4 + (c3 - c4)*v;
    
    float r = r1 + (r2 - r1)*u;
    
    return r;
}

// Function 396
float SmoothNoise2D(vec2 uv) {
    vec2 lv = fract(uv);
    vec2 id = floor(uv);
    
    lv = lv*lv*(3.0-2.0*lv);
    
    float bl = HASH21(id + vec2(0, 0));
    float br = HASH21(id + vec2(1, 0));
    float b = mix(bl, br, lv.x);
    
    float tl = HASH21(id + vec2(0, 1));
    float tr = HASH21(id + vec2(1, 1));
    float t = mix(tl, tr, lv.x);
    
    return mix(b, t, lv.y);
}

// Function 397
float perlin(vec2 pos)
{
    return noise(pos.x, pos.y, 0.0f);
}

// Function 398
vec3 sandNoiser(vec2 p, float ti){
    vec3 e = vec3(0.);
    for(int j=1; j<3; j++){
        e += texture(iChannel1, p * (float(j)*1.79) + vec2(ti*7.89541) ).rgb ;
    }
    e /= 3.;
    return e;
}

// Function 399
float noise( float x ){return fract(sin(1371.1*x)*43758.5453);}

// Function 400
float perlin(vec2 uv)
{
    
    float c = smoothNoise(uv * 4.* sin(1. * 0.01 + .113));
    c += smoothNoise(uv * 8. * sin(1. * 0.12)) * 0.5;
    c += smoothNoise(uv * 16. * sin(1. * 0.1 + 1.213)) * 0.25;
    c += smoothNoise(uv * 32.* sin(1. * 0.042 + .213)) * 0.125;
    c += smoothNoise(uv * 64. * sin(1. * 0.0037 + .113)) * 0.0625;
    c /= 2.;
    return c;
}

// Function 401
vec3 gradientNoised(vec2 pos, vec2 scale, float seed) 
{
    // gradient noise with derivatives based on Inigo Quilez
    pos *= scale;
    vec4 i = floor(pos).xyxy + vec2(0.0, 1.0).xxyy;
    vec4 f = (pos.xyxy - i.xyxy) - vec2(0.0, 1.0).xxyy;
    i = mod(i, scale.xyxy) + seed;
    
    vec4 hashX, hashY;
    smultiHash2D(i, hashX, hashY);
    vec2 a = vec2(hashX.x, hashY.x);
    vec2 b = vec2(hashX.y, hashY.y);
    vec2 c = vec2(hashX.z, hashY.z);
    vec2 d = vec2(hashX.w, hashY.w);
    
    vec4 gradients = hashX * f.xzxz + hashY * f.yyww;

    vec4 udu = noiseInterpolateDu(f.xy);
    vec2 u = udu.xy;
    vec2 g = mix(gradients.xz, gradients.yw, u.x);
    
    vec2 dxdy = a + u.x * (b - a) + u.y * (c - a) + u.x * u.y * (a - b - c + d);
    dxdy += udu.zw * (u.yx * (gradients.x - gradients.y - gradients.z + gradients.w) + gradients.yz - gradients.x);
    return vec3(mix(g.x, g.y, u.y) * 1.4142135623730950, dxdy);
}

// Function 402
float noise(vec3 p)
{
	vec3 ip=floor(p); p-=ip; 
    vec3 s=vec3(7,157,113);
    vec4 h=vec4(0.,s.yz,s.y+s.z)+dot(ip,s);
    p=p*p*(3.-2.*p); 
    h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
    h.xy=mix(h.xz,h.yw,p.y);
    return mix(h.x,h.y,p.z); 
}

// Function 403
float Voronoi(in vec2 p){
    
	vec2 g = floor(p), o; p -= g;
	
	vec3 d = vec3(1); // 1.4, etc. "d.z" holds the distance comparison value.
    
	for(int y = -1; y <= 1; y++){
		for(int x = -1; x <= 1; x++){
            
			o = vec2(x, y);
            o += hash22(g + o) - p;
            
			d.z = dot(o, o); 
            // More distance metrics.
            //o = abs(o);
            //d.z = max(o.x*.8666 + o.y*.5, o.y);// 
            //d.z = max(o.x, o.y);
            //d.z = (o.x*.7 + o.y*.7);
            
            d.y = max(d.x, min(d.y, d.z));
            d.x = min(d.x, d.z); 
                       
		}
	}
	
    return max(d.y/1.2 - d.x*1., 0.)/1.2;
    //return d.y - d.x; // return 1.-d.x; // etc.
    
}

// Function 404
float Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel0, (uv+ 0.5)/256.0, -99.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 405
vec2 Noise22(vec2 x)
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    vec2 res = mix(mix( hash22(p),          hash22(p + add.xy),f.x),
                    mix( hash22(p + add.yx), hash22(p + add.xx),f.x),f.y);
    return res-.5;
}

// Function 406
float noiseOct(vec2 p, float r)
{float k=.0,o=1.
;for (float z=.0;z<iterNoiseOctaves;++z
){k+=Simplex12(p/o,r,z)*o;o*=2.;}return k;}

// Function 407
float noise(float t, float seed) {
    float i = floor(t), f = fract(t);
    float u = 0.5 - cos(3.14159*f)*0.5;
    return mix(hash11(i + seed),  hash11(i + 1. + seed), u);
}

// Function 408
float noise3( vec3 x , out vec2 g) {
    vec3 p = floor(x),f = fract(x),
        F = f*f*(3.-2.*f);  // or smoothstep     // to make derivative continuous at borders

#define hash3(p)  fract(sin(1e3*dot(p,vec3(1,57,-13.7)))*4375.5453)        // rand
    
    float v000 = hash3(p+vec3(0,0,0)), v100 = hash3(p+vec3(1,0,0)),
          v010 = hash3(p+vec3(0,1,0)), v110 = hash3(p+vec3(1,1,0)),
          v001 = hash3(p+vec3(0,0,1)), v101 = hash3(p+vec3(1,0,1)),
          v011 = hash3(p+vec3(0,1,1)), v111 = hash3(p+vec3(1,1,1));
    
    g.x = 6.*f.x*(1.-f.x)                        // gradients
          * mix( mix( v100 - v000, v110 - v010, F.y),
                 mix( v101 - v001, v111 - v011, F.y), F.z);
    g.y = 6.*f.y*(1.-f.y)
          * mix( mix( v010 - v000, v110 - v100, F.x),
                 mix( v011 - v001, v111 - v101, F.x), F.z);
    
    return mix( mix(mix( v000, v100, F.x),       // triilinear interp
                    mix( v010, v110, F.x),F.y),
                mix(mix( v001, v101, F.x),       
                    mix( v011, v111, F.x),F.y), F.z);
}

// Function 409
float polarNoiseN(vec3 pos1)
{
    vec3 q = 8.0*pos1;
    float f = 0.0;
    f  = 0.5000*polarNoise0(q); q = m*q*2.;
    f += 0.2500*polarNoise0(q); q = m*q*2.;
    
    return f;
}

// Function 410
float get_octave_noise(vec2 pos)
{
    float rows = float(IMAGE_ROWS);
    pos *= rows;
    float columns = rows * (iResolution.x / iResolution.y);
    float scale = float(SCALE);
    if(scale <= 0.0f)
    {
        scale = 0.001f;
    }
    
    int octaves = int(OCTAVES);
    float lacunarity = max(LACUNARITY, 1.0f);
    float persistence = min(PERSISTANCE, 1.0f);
    
    float halfX = 0.0f;
    float halfY = 0.0f;
#if SCALE_FROM_CENTER
    halfX = columns / 2.0f;
    halfY = rows / 2.0f;
#endif

    float amplitude = 1.0f;
    float frequency = 1.0f;
    float noiseVal = 0.0f;
    
    // Add LODs
#if LEVEL_OF_DETAIL
    pos /= float(LEVEL_OF_DETAIL);
    pos = vec2(floor(pos.x), floor(pos.y));
    pos *= float(LEVEL_OF_DETAIL);
#endif

    vec2 offset = 0.1f * vec2(iTime * 1.25f, iTime * 1.25f);
    
    for (int i = 0; i < octaves; i++)
    {
#if NORMALIZE_OFFSET
        float sampleX = (((pos.x-halfX) / scale) * frequency) + offset.x;
        float sampleY = (((pos.y-halfY) / scale) * frequency) + offset.y;
#else
        float sampleX = (((pos.x-halfX + offset.x*scale) / scale) * frequency);
        float sampleY = (((pos.y-halfY + offset.y*scale) / scale) * frequency);
#endif
        float noise = (perlin(vec2(sampleX, sampleY)) * 2.0f) - 1.0f;
        noiseVal += noise * amplitude;
        // Decrease A and increase F
        amplitude *= persistence;
        frequency *= lacunarity;
    }    

    // Inverser lerp so that noiseval lies between 0 and 1 
#if SMOOTH_INVERSE_LERP
    noiseVal = smoothstep(-0.95f, 1.1f, noiseVal);
#else
    noiseVal = linear_step(-0.7f,0.85f,noiseVal);
#endif
    return noiseVal;
}

// Function 411
vec3 perlin31(float p, float n)
{
    float frq = 1., amp = 1., norm = 0.;
    vec3 res = vec3(0.);
    for(float i = 0.; i < n; i++)
    {
        res += amp*perlin31(frq*p);
        norm += amp;
        frq *= 2.;
       // amp *= 1;
    }
    return res/norm;
}

// Function 412
float noiseMask(vec2 uv, int layer)
{
    vec4 p = vec4(uv * 3., float(layer) * DIVERGENCE, iTime * 0.5);
    float f = twistedSineNoise(p);
    f += length(uv) * SHAPE_SIMPLICITY;
    return step(SHAPE_SIMPLICITY, f + float(layer));
}

// Function 413
float fractal_noise(vec3 p)
{
    float f = 0.0;
    // add animation
    p = p - vec3(1.0, 1.0, 0.0) * iTime * 0.1;
    p = p * 3.0;
    f += 0.50000 * noise(p); p = 2.0 * p;
	f += 0.25000 * noise(p); p = 2.0 * p;
	f += 0.12500 * noise(p); p = 2.0 * p;
	f += 0.06250 * noise(p); p = 2.0 * p;
    f += 0.03125 * noise(p);
    
    return f;
}

// Function 414
float TrigNoise(vec3 x, float a, float b)
{
    vec4 u = vec4(dot(x, vec3( 1.0, 1.0, 1.0)), 
                  dot(x, vec3( 1.0,-1.0,-1.0)), 
                  dot(x, vec3(-1.0, 1.0,-1.0)),
                  dot(x, vec3(-1.0,-1.0, 1.0))) * a;

    return dot(sin(x     + cos(u.xyz) * b), 
               cos(x.zxy + sin(u.zwx) * b));
}

// Function 415
vec4 bluenoise(vec2 u){//U=floor(U/8.); 
 vec4 n=8./9.*noise(u)-1./9.*(V(-1,-1)+V(0,-1)+V(1,-1)+V(-1,0)+V(1,0)+V(-1,1)+V(0,1)+V(1,1));  
 return n*2.+.5;}

// Function 416
vec4 bccNoiseDerivativesPart(vec3 X) {
    vec3 b = floor(X);
    vec4 i4 = vec4(X - b, 2.5);
    
    // Pick between each pair of oppposite corners in the cube.
    vec3 v1 = b + floor(dot(i4, vec4(.25)));
    vec3 v2 = b + vec3(1, 0, 0) + vec3(-1, 1, 1) * floor(dot(i4, vec4(-.25, .25, .25, .35)));
    vec3 v3 = b + vec3(0, 1, 0) + vec3(1, -1, 1) * floor(dot(i4, vec4(.25, -.25, .25, .35)));
    vec3 v4 = b + vec3(0, 0, 1) + vec3(1, 1, -1) * floor(dot(i4, vec4(.25, .25, -.25, .35)));
    
    // Gradient hashes for the four vertices in this half-lattice.
    vec4 hashes = permute(mod(vec4(v1.x, v2.x, v3.x, v4.x), 289.0));
    hashes = permute(mod(hashes + vec4(v1.y, v2.y, v3.y, v4.y), 289.0));
    hashes = mod(permute(mod(hashes + vec4(v1.z, v2.z, v3.z, v4.z), 289.0)), 48.0);
    
    // Gradient extrapolations & kernel function
    vec3 d1 = X - v1; vec3 d2 = X - v2; vec3 d3 = X - v3; vec3 d4 = X - v4;
    vec4 a = max(0.75 - vec4(dot(d1, d1), dot(d2, d2), dot(d3, d3), dot(d4, d4)), 0.0);
    vec4 aa = a * a; vec4 aaaa = aa * aa;
    vec3 g1 = grad(hashes.x); vec3 g2 = grad(hashes.y);
    vec3 g3 = grad(hashes.z); vec3 g4 = grad(hashes.w);
    vec4 extrapolations = vec4(dot(d1, g1), dot(d2, g2), dot(d3, g3), dot(d4, g4));
    
    // Derivatives of the noise
    vec3 derivative = -8.0 * mat4x3(d1, d2, d3, d4) * (aa * a * extrapolations)
        + mat4x3(g1, g2, g3, g4) * aaaa;
    
    // Return it all as a vec4
    return vec4(derivative, dot(aaaa, extrapolations));
}

// Function 417
float perlinFBM(vec3 st){

    const float initScale = 5.0;
    st*=initScale;
    const int octaves = 5;
    float a = 0.5;
    float f = 1.0;
    float tot = 0.0;
    for(int i = 0; i <= octaves; ++i){
        tot += a*InterpolatedNoise(f*st.x, f*st.y, f*st.z);
     	a *= a;
        f *= 2.0;
    }
    return tot;
    
}

// Function 418
float noise_itself(vec2 p)
{
    return noise(p * 8.0);
}

// Function 419
float perlin(vec2 uv)
{
    float a,b,c,d, coef1,coef2, t, p;

    t = _PerlinPrecision;					// Precision
    p = 0.0;								// Final heightmap value

    for(float i=0.0; i<_PerlinOctaves; i++)
    {
        a = rnd(vec2(floor(t*uv.x)/t, floor(t*uv.y)/t));	//	a----b
        b = rnd(vec2(ceil(t*uv.x)/t, floor(t*uv.y)/t));		//	|    |
        c = rnd(vec2(floor(t*uv.x)/t, ceil(t*uv.y)/t));		//	c----d
        d = rnd(vec2(ceil(t*uv.x)/t, ceil(t*uv.y)/t));

        if((ceil(t*uv.x)/t) == 1.0)
        {
            b = rnd(vec2(0.0, floor(t*uv.y)/t));
            d = rnd(vec2(0.0, ceil(t*uv.y)/t));
        }

        coef1 = fract(t*uv.x);
        coef2 = fract(t*uv.y);
        p += inter(inter(a,b,coef1), inter(c,d,coef1), coef2) * (1.0/pow(2.0,(i+0.6)));
        t *= 2.0;
    }
    return p;
}

// Function 420
float noise_perlin(vec2 p)
{
	vec2 ni = floor(p);
    vec2 nf = fract(p);
    
    vec2 w = nf*nf*(3.-2.*nf);
    
    float f1 = dot(hash22(ni),nf);
    float f2 = dot(hash22(ni+vec2(1.,0.)),nf-vec2(1.,0.));
    float f3 = dot(hash22(ni+vec2(1.,1.)),nf-vec2(1.,1.));
    float f4 = dot(hash22(ni+vec2(0.,1.)),nf-vec2(0.,1.));
    
    float f12 = mix(f1,f2,w.x);
    
    float f34 = mix(f4,f3,w.x);
    
    float f=mix(f12,f34,w.y);
    
    return f;
    
}

// Function 421
vec3 voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
        }
    }
    
    // Set center of search based on which half of the cell we are in,
    // since 4x4 is not centered around "n".
    vec2 mg = step(.5,f) - 1.;

    //----------------------------------
    // second pass: distance to borders,
    // visits two neighbours to the right/down
    //----------------------------------
    md = 8.0;
    for( int j=-1; j<=2; j++ )
    for( int i=-1; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) // skip the same cell
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec3( md, mr );
}

// Function 422
float value_noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    vec2 s = smoothstep(0.0, 1.0, f);
    float nx0 = mix(rand(i + vec2(0.0, 0.0)), rand(i + vec2(1.0, 0.0)), s.x);
    float nx1 = mix(rand(i + vec2(0.0, 1.0)), rand(i + vec2(1.0, 1.0)), s.x);
    return mix(nx0, nx1, s.y);
}

// Function 423
float gaussianNoise3D(vec3 uv)
{
	vec3 p = floor(uv);
    vec3 f = fract(uv);
    
    f = f*f*(3.0-2.0*f);
    /*float c = cos(uv.x);
    float s = sin(uv.y);
    mat2 R = mat2(c, s, -s, c);*/
    
    return
        mix(
        	mix(
                mix(hash3D(p+vec3(0.0, 0.0, 0.0)), hash3D(p+vec3(1.0, 0.0, 0.0)), f.x),
                mix(hash3D(p+vec3(0.0, 1.0, 0.0)), hash3D(p+vec3(1.0, 1.0, 0.0)), f.x),
            f.y),
        	mix(
                mix(hash3D(p+vec3(0.0, 0.0, 1.0)), hash3D(p+vec3(1.0, 0.0, 1.0)), f.x),
                mix(hash3D(p+vec3(0.0, 1.0, 1.0)), hash3D(p+vec3(1.0, 1.0, 1.0)), f.x),
            f.y),
        f.z);
}

// Function 424
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 425
float DenoiseCoeff(vec4 att, float CD, vec3 CN, vec3 deltaL, float LCoeff) {
    if (att.w>99990.) return 0.;
    return WeightVar(CD-att.w,dot(CN,Read(att.y).xyz*2.-1.),deltaL,LCoeff);
}

// Function 426
float Noise3D(in vec3 p) { float z  = 1.4; float rz = 0.0; vec3  bp = p; for(float i = 0.0; i <= 2.0; i++) { vec3 dg = tri3(bp); p += (dg); bp *= 2.0; z  *= 1.5; p  *= 1.3; rz += (tri(p.z+tri(p.x+tri(p.y))))/z; bp += 0.14; } return rz; }

// Function 427
float interleavedGradientNoise(vec2 pos)
{
  float f = 0.06711056 * pos.x + 0.00583715 * pos.y;
  return fract(52.9829189 * fract(f));
}

// Function 428
vec2 stepnoise(vec2 p, float size) {
    p += 10.0;
    float x = floor(p.x/size)*size;
    float y = floor(p.y/size)*size;
    
    x = fract(x*0.1) + 1.0 + x*0.0002;
    y = fract(y*0.1) + 1.0 + y*0.0003;
    
    float a = fract(1.0 / (0.000001*x*y + 0.00001));
    a = fract(1.0 / (0.000001234*a + 0.00001));
    
    float b = fract(1.0 / (0.000002*(x*y+x) + 0.00001));
    b = fract(1.0 / (0.0000235*b + 0.00001));
    
    return vec2(a, b);
    
}

// Function 429
vec2 voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

	vec3 m = vec3(8.0);
    for(int j=-1; j<=1; j++)
    for(int i=-1; i<=1; i++)
    {
        vec2  g = vec2(float(i), float(j));
        vec2  o = hash(n + g) * (cos(iTime) + 1.0) / 2.0;
        
        if (mod(n.x, 2.0) - float(i) == -1.0 || mod(n.x, 2.0) - float(i) == 1.0)
            o.y += 0.5;
        
       	vec2  r = g - f + o;
        
		float d = dot(r, r);
        if(d < m.x)
            m = vec3(d, o);
    }

    return vec2( sqrt(m.x), m.y+m.z );
}

// Function 430
float triNoise3d(in vec3 p, in float spd)
{
    float z=1.4;
	float rz = 0.;
    vec3 bp = p;
	for (float i=0.; i<=3.; i++ )
	{
        vec3 dg = tri3(bp*2.);
        p += (dg+time*spd);

        bp *= 1.8;
		z *= 1.5;
		p *= 1.2;
        //p.xz*= m2;
        
        rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
        bp += 0.14;
	}
	return rz;
}

// Function 431
vec3 noise3( in vec3 x )
{
    return textureLod(iChannel0, x.xy/x.z, 0.0).xyz;   
}

// Function 432
float noise( vec2 uv ){
    vec3 x = vec3(uv, 0);

    vec3 p = floor(x);
    vec3 f = fract(x);
    
    f       = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    
    return mix(mix(mix( hash(n+0.0), hash(n+1.0),f.x),
                   mix( hash(n+57.0), hash(n+58.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

// Function 433
vec3 noise(float p, float lod){return texture(iChannel0,vec2(p/iChannelResolution[0].x,.0),lod).xyz;}

// Function 434
float noiseValueFbm2D(vec2 p, float s, float speed, int octaves, float amplitude)
{
    float o, mx = 0.0;
    for(int i = 0; i >= 0; i++)
    {
        if(i >= octaves)	break;
        float a = pow(amplitude, float(i));
        o += a * noiseValue2D(p, s * exp2(float(i)), speed);
        mx += a;
    }
    return o / mx;
}

// Function 435
vec3 noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return 1.0 - 2.0 * mix(mix(mix(hash3(i + vec3(0.0, 0.0, 0.0)), 
                                   hash3(i + vec3(1.0, 0.0, 0.0)), u.x),
                               mix(hash3(i + vec3(0.0, 1.0, 0.0)), 
                                   hash3(i + vec3(1.0, 1.0, 0.0)), u.x), u.y),
                           mix(mix(hash3(i + vec3(0.0, 0.0, 1.0)), 
                                   hash3(i + vec3(1.0, 0.0, 1.0)), u.x),
                               mix(hash3(i + vec3(0.0, 1.0, 1.0)), 
                                   hash3(i + vec3(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

// Function 436
float noise( in vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
    vec2 rg = texture( iChannel0, (uv+0.5) / 256.0 ).yx;
    return mix( rg.x, rg.y, f.z );
}

// Function 437
float Noise3D( vec4 pos CACHEARG )
{
    vec3 baseCorner = floor( pos.xyz );
    
    vec3 frac = pos.xyz - baseCorner;
    vec3 fracSmooth = frac * frac * (3.0 - 2.0 * frac);
    
    vec4 vals[2];
	// Manually unrolling this loop helped compile times a lot.
	/*for ( int i = 0; i < 4; i++ )
    {
        for ( int j = 0; j < 2; j++ )
        {
            vec3 offset = frac - Noise3DCornerOffsets[i * 4 + j];
            vals[i][j] = dot( offset, cache.gradients[i * 4 + j] );
        }
    }*/
    vals[0] = vec4(
        dot( frac                         , LatticeGradient3D( baseCorner                         , pos.w ) ),
    	dot( frac - Noise3DCornerOffsets_1, LatticeGradient3D( baseCorner + Noise3DCornerOffsets_1, pos.w ) ),
    	dot( frac - Noise3DCornerOffsets_2, LatticeGradient3D( baseCorner + Noise3DCornerOffsets_2, pos.w ) ),
    	dot( frac - Noise3DCornerOffsets_3, LatticeGradient3D( baseCorner + Noise3DCornerOffsets_3, pos.w ) )
    );
    vals[1] = vec4(
        dot( frac - Noise3DCornerOffsets_4, LatticeGradient3D( baseCorner + Noise3DCornerOffsets_4, pos.w ) ),
    	dot( frac - Noise3DCornerOffsets_5, LatticeGradient3D( baseCorner + Noise3DCornerOffsets_5, pos.w ) ),
    	dot( frac - Noise3DCornerOffsets_6, LatticeGradient3D( baseCorner + Noise3DCornerOffsets_6, pos.w ) ),
    	dot( frac - Noise3DCornerOffsets_7, LatticeGradient3D( baseCorner + Noise3DCornerOffsets_7, pos.w ) )
    );
    
    vec4 xvals = mix( vals[0], vals[1], fracSmooth.x );
    
    vec2 yvals = mix( xvals.xy, xvals.zw, fracSmooth.y );
    
    return mix( yvals.x, yvals.y, fracSmooth.z );
}

// Function 438
vec3 voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>0.00001 )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec3( md, mr );
}

// Function 439
float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) +
        (c - a)* u.y * (1.0 - u.x) +
        (d - b) * u.x * u.y;
}

// Function 440
float noise1D(float p){
    float fl = floor(p);
    float fc = fract(p);
    return (mix(rand(fl), rand(fl + 1.0), fc)-.5)*2.;
}

// Function 441
float NebulaNoise(vec3 p)
{
    float final = Disk(p.xzy,vec3(2.0,1.8,1.25));
    final += fbm(p*90.);
    final += SpiralNoiseC(p.zxy*0.5123+100.0)*3.0;

    return final;
}

// Function 442
float fractalNoise(vec2 vl, out float mainWave) {
    
#if SHARP_MODE==1
    const float persistance = 2.4;
    float frequency = 2.2;
    const float freq_mul = 2.2;
    float amplitude = .4;
#else
    const float persistance = 3.0;
    float frequency = 2.3;
    const float freq_mul = 2.3;
    float amplitude = .7;
#endif
    
    float rez = 0.0;
    vec2 p = vl;
    
    float mainOfset = (iTime + 40.)/ 2.;
    
    vec2 waveDir = vec2(p.x+ mainOfset, p.y + mainOfset);
    float firstFront = amplitude + 
			        (valueNoiseSimple(p) * 2. - 1.);
    mainWave = firstFront * valueNoiseSimple(p + mainOfset);
    
    rez += mainWave;
    amplitude /= persistance;
    p *= unique_transform;
    p *= frequency;
    

    float timeOffset = iTime / 4.;

    
    for (int i = 1; i < OCTAVES; i++) {
        waveDir = p;
        waveDir.x += timeOffset;
        rez += amplitude * sin(valueNoiseSimple(waveDir * frequency) * .5 );
        amplitude /= persistance;
        p *= unique_transform;
        frequency *= freq_mul;

        timeOffset *= 1.025;

        timeOffset *= -1.;
    }

    return rez;
}

// Function 443
vec4 simplexWeaveBump(vec3 q){   
    
    
    
    // SIMPLEX GRID SETUP
    
    vec2 p = q.xy; // 2D coordinate on the XY plane    
    
    vec2 s = floor(p + (p.x + p.y)*.36602540378); // Skew the current point.
    
    p -= s - (s.x + s.y)*.211324865; // Use it to attain the vector to the base vertex (from p).
    
    // Determine which triangle we're in. Much easier to visualize than the 3D version.
    float i = p.x < p.y? 1. : 0.; // Apparently, faster than: i = step(p.y, p.x);
    vec2 ioffs = vec2(1. - i, i);
    
    // Vectors to the other two triangle vertices.
    vec2 p1 = p - ioffs + .2113248654, p2 = p - .577350269; 

    
    // THE WEAVE PATTERN
    
    // A random value -- based on the triangle vertices, and ranging between zero and one.
    float dh = hash21((s*3. + ioffs + 1.));
    
 
    // Based on the unique random value for the triangle cell, rotate the tile.
    // In effect, we're swapping vertex positions... I figured it'd be cheaper,
    // but I didn't think about it for long, so there could be a faster way. :)
    //
    if(dh<1./3.) { // Rotate by 60 degrees.
        vec2 ang = p;
        p = p1, p1 = p2, p2 = ang;
        
    }
    else if(dh<2./3.){ // Rotate by 120 degrees.
        vec2 ang = p;
        p = p2, p2 = p1, p1 = ang;
    }

     
    
    // Angles subtended from the current position to each of the three vertices... There's probably a 
    // symmetrical way to make just one "atan" call. Anyway, you can use these angular values to create 
    // patterns that follow the contours. In this case, I'm using them to create some cheap repetitious lines.
    vec3 a = vec3(atan(p.y, p.x), atan(p1.y, p1.x), atan(p2.y, p2.x));
 
    // The torus rings. 
    // Toroidal axis width. Basically, the weave pattern width.
    float tw = .2;
    // For symmetry, we want the middle of the torus ring to cut dirrectly down the center
    // of one of the equilateral triangle sides, which is half the distance from one of the
    // vertices to the other. Add ".1" to it to see that it's necessary.
    float mid = dist((p2 - p))*.5;
    // The three distance field functions: Stored in cir.x, cir.y and cir.z.
    vec3 cir = vec3(dist(p), dist(p1), dist(p2));
    // Equivalent to: vec3 tor =  cir - mid - tw; tor = max(tor, -(cir - mid + tw));
    vec3 tor =  abs(cir - mid) - tw;
        
 
    

    
    // RENDERING
    // Applying the layered distance field objects.
    
    // The background. I tried non-greyscale colors, but they didn't fit.
    vec3 bg = vec3(.75);
    // Floor pattern.
    //bg *= clamp(cos((q.x - q.y)*6.2831*16.) - .25, 0., 1.)*.2 + .9;
   
    // The scene color. Initialized to the background.
    vec3 col = bg;
    
    // Outer torus ring color. 
    vec3 rimCol = vec3(.5);
    
    // Use the angular component to create lines running perpendicular 
    // to the curves.
    
    // Angular floor ridges.
    vec3 ridges = smoothstep(.05, -.05, -cos(a*12.) + .93)*.5;
    
    // Ridges pattern on the toroidal rails. Set it to zero, if you're not sure
    // what it does.
    vec3 ridges2 = smoothstep(.05, -.05, sin(a*18.) + .93);
    
    // Using the angle to create the height of the toroidal curves.
    a = sin(a*3. - 6.283/9.)*.5 + .5; 
     

    
         
    // Smoothing factor and line width.
    const float sf = .015, lw = .02;
   
    // Rendering the the three ordered (random or otherwise) objects:
    //
    // This is all pretty standard stuff. If you're not familiar with using a 2D
    // distance field value to mix a layer on top of another, it's worth learning.
    // On a side note, "1. - smoothstep(a, b, c)" can be written in a more concise
    // form (smoothstep(b, a, c), I think), but I've left it that particular way
    // for readability. You could also reverse the first two "mix" values, etc.
    // By readability, I mean the word "col" is always written on the left, the
    // "0." figure is always on the left, etc. If this were a more GPU intensive
    // exercise, then I'd rewrite things.
    
    
    // Bottom toroidal segment.
    //
    // Outer dark edges.
    //col = mix(col, vec3(.1), 1. - smoothstep(0., sf, tor.z));
    // The main toroidal face.
    //col = mix(col, rimCol*(1. - ridges.z), 1. - smoothstep(0., sf, tor.z + lw));
    
    // Same layering routine for the middle toroidal segment.
    col = mix(col, vec3(.1), 1. - smoothstep(0., sf, tor.y));
    col = mix(col, rimCol*(1. - ridges.y), 1. - smoothstep(0., sf, tor.y + lw));
  
    // The final toroidal segment last.
	col = mix(col, vec3(.1), (1. - smoothstep(0., sf, tor.x))*1.);
    col = mix(col, rimCol*(1. - ridges.x), (1. - smoothstep(0., sf, tor.x + lw))*1.);
  
    //col = vec3(1);
    
    // Hexagon centers.
    //col = mix(col, vec3(.5), (1. - smoothstep(0., sf, hole0)));

    /*
    float shp = min(tor.x, tor.y);
    shp =  -shp + .035;//abs(shp - .075);
    col = mix(col, vec3(.1), (1. - smoothstep(0., sf, shp)));
    col = mix(col, vec3(rimCol), (1. - smoothstep(0., sf, shp + lw)));
    */

    
    #ifdef SIMPLEX_GRID
    // Displaying the 2D simplex grid. Basically, we're rendering lines between
    // each of the three triangular cell vertices to show the outline of the 
    // cell edges.
    vec3 c = vec3(distLine(p, p1), distLine(p1, p2), distLine(p2, p));
    c.x = min(min(c.x, c.y), c.z);
    //col = mix(col, vec3(.1), (1. - smoothstep(0., sf*2., c.x - .0)));
    col = mix(col, vec3(.1), (1. - smoothstep(0., sf*2., c.x - .02)));
    col = mix(col, vec3(1, .8, .45)*1.35, (1. - smoothstep(0., sf*1.25, c.x - .005)));
    #endif
    
    
    // Arc tile calculations: This is a shading and bump mapping routine. However, 
    // we're using it in a 3D setting, so we need to determine which 3D tube is closest, 
    // in order to know where to apply the bump pattern.
    vec2 d1 = vec2(1e5), d2 = vec2(1e5), d3 = vec2(1e5);
    
    // Moving the Z component to the right depth, in order to match the 3D
    // pattern. I forgot to do this, and it led to all kinds of confusion. :)
    q.z -= .5;
     
    // The three arc sweeps. By the way, you can change the define above the
    // "tubeOuter" function to produce a hexagonal sweep, etc.
    float depth = .125;
    vec2 v1 = vec2(tubeOuter(p.xy) - mid, q.z - depth); // Bottom level arc.
    // This arc sweeps between the bottom and top levels.
    vec2 v2 = vec2(tubeOuter(p1.xy) - mid, q.z + depth*a.y - depth); 
    vec2 v3 = vec2(tubeOuter(p2.xy) - mid, q.z); // Top level arc.
    
    // Shaping the poloidal coordinate vectors above into rails.
    d1 = tube(v1);
    d2 = tube(v2);
    d3 = tube(v3);

    
    // Based on which tube we've hit, apply the correct anular bump pattern.
    // Try mixing this up, and you'll see why it's necessary.
    float bump = d1.x<d2.x && d1.x<d3.x? ridges2.x : d2.x<d3.x? ridges2.y : ridges2.z;
    
     
    
    // Return the simplex weave value.
    return vec4(col, 1. - bump);
 

}

// Function 444
float mnoise(vec3 pos) {
    float intArg = floor(pos.z);
    float fracArg = fract(pos.z);
    vec2 hash = mBBS(intArg * 3.0 + vec2(0, 3), modulus);
    vec4 g = vec4(
        texture(iChannel0, vec2(pos.x, pos.y + hash.x) / modulus).xy,
        texture(iChannel0, vec2(pos.x, pos.y + hash.y) / modulus).xy) * 2.0 - 1.0;
    return mix(g.x + g.y * fracArg,
               g.z + g.w * (fracArg - 1.0),
               smoothstep(0.0, 1.0, fracArg));
  }

// Function 445
vec4 Noise( in ivec2 x )
{
	return texture( iChannel0, (vec2(x)+0.5)/256.0, -100.0 );
}

// Function 446
float noise(vec2 n) {
    const vec2 d = vec2(0.0, 1.0);
    vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
    return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

// Function 447
float noise(float g) {
	float p = floor(g);
	float f = fract(g);

	return mix(hash(p), hash(p + 1.0), f);
}

// Function 448
vec3 noised( in vec2 x ) {
    vec2 f = fract(x);
    vec2 u = f*f*(3.0-2.0*f);
    
    vec2 p = vec2(floor(x));
    float a = hash12( (p+vec2(0,0)) );
	float b = hash12( (p+vec2(1,0)) );
	float c = hash12( (p+vec2(0,1)) );
	float d = hash12( (p+vec2(1,1)) );
    
	return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
				6.0*f*(1.0-f)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));
}

// Function 449
float noise( in vec3 x )
{
	float  z = x.z*64.0;
	vec2 offz = vec2(0.217,0.123);
	vec2 uv1 = x.xy + offz*floor(z); 
	vec2 uv2 = uv1  + offz;
	return mix(texture( iChannel0, uv1 ,-1000.0).x,texture( iChannel0, uv2 ,-1000.0).x,fract(z))-0.5;
}

// Function 450
float worleyNoise(vec3 uv, float freq, bool tileable)
{    
    vec3 id = floor(uv);
    vec3 p = fract(uv);
    float minDist = 10000.;
    
    for (float x = -1.; x <= 1.; ++x)
    {
        for(float y = -1.; y <= 1.; ++y)
        {
            for(float z = -1.; z <= 1.; ++z)
            {
                vec3 offset = vec3(x, y, z);
                vec3 h = vec3(0.);
                if (tileable)
                    h = hash33(mod(id + offset, vec3(freq))) * .4 + .3; // [.3, .7]
				else
                    h = hash33(id + offset) * .4 + .3; // [.3, .7]
    			h += offset;
            	vec3 d = p - h;
           		minDist = min(minDist, dot(d, d));
            }
        }
    }
    
    // inverted worley noise
    return 1. - minDist;
}

// Function 451
float noise(float x) { float i = floor(x); float f = fract(x); float u = f * f * (3.0 - 2.0 * f); return mix(hash(i), hash(i + 1.0), u); }

// Function 452
void noiseCracks(out vec4 fragColor, in vec2 fragCoord) {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    float value = fbm2(uv + vec2(0.5, 0.5));
    vec3 col = vec3(value);
    float cracks = abs( -1. + 2. * fbm(uv * 6.));
    cracks = clamp(pow(cracks, 2.3) * 10., 0., 1.);
    cracks = 0.5 + sin(value * 20. * 3.14) * 0.5;
    /*if (cracks <= 0.5) {
        cracks = mix(1., 0.5, clamp((cracks - 0.45) * 20., 0., 1.));
    }
    else {
        cracks = smoothstep(0.25, 1., clamp((cracks - 0.5) * 20., 0., 1.));
    }*/
    col = mix(col, vec3(0.4), cracks);
    
    // Output to screen
    fragColor = vec4(col,1.0);
}

// Function 453
float snoise(vec3 p)
{
    const float K1 = 0.333333333;
    const float K2 = 0.166666667;

    vec3 i = floor(p + (p.x + p.y + p.z) * K1);
    vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);

    vec3 e = step(vec3(0.0), d0 - d0.yzx);
    vec3 i1 = e * (1.0 - e.zxy);
    vec3 i2 = 1.0 - e.zxy * (1.0 - e);

    vec3 d1 = d0 - (i1 - 1.0 * K2);
    vec3 d2 = d0 - (i2 - 2.0 * K2);
    vec3 d3 = d0 - (1.0 - 3.0 * K2);

    vec4 h = max(0.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
    vec4 n = h * h * h * h * vec4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0)));

    return dot(vec4(31.316), n);
}

// Function 454
float noise_prng_uniform_0_1(inout noise_prng this_)
{
    return float(noise_prng_rand(this_)) / float(4294967295u);
}

// Function 455
float pnoise(vec3 P, vec3 rep)
{
  vec3 Pi0 = mod(floor(P), rep); // Integer part, modulo period
  vec3 Pi1 = mod(Pi0 + vec3(1, 1, 1), rep); // Integer part + 1, mod period
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1, 1, 1); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(.5, .5, .5, .5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0, 0, 0, 0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(.5, .5, .5, .5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0, 0, 0, 0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
  return 2.2 * n_xyz;
}

// Function 456
float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187,0.366025403784439,-0.577350269189626,0.024390243902439);
  vec2 i = floor(v + dot(v, C.yy) );
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i,289.);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 )) + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  vec3 x = 2. * fract(p * C.www) - 1.;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m = m*m*m*m*(1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h ));
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130. * dot(m, g);
}

// Function 457
float noise(vec3 x){
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix(mix(mix(hash(i+vec3(0, 0, 0)), 
                       hash(i+vec3(1, 0, 0)),f.x),
                   mix(hash(i+vec3(0, 1, 0)), 
                       hash(i+vec3(1, 1, 0)),f.x),f.y),
               mix(mix(hash(i+vec3(0, 0, 1)), 
                       hash(i+vec3(1, 0, 1)),f.x),
                   mix(hash(i+vec3(0, 1, 1)), 
                       hash(i+vec3(1, 1, 1)),f.x),f.y),f.z);
}

// Function 458
float gradientNoise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    float rz =  mix( mix( dot( hashg( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( hashg( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hashg( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( hashg( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
    
    //return rz*0.75+0.5;
    return smoothstep(-.9,.9,rz);
}

// Function 459
vec2 noise(float t)
{
    return hash22(vec2(t, t * 1.423)) * 2.0 - 1.0;
}

// Function 460
float noise(vec2 p)
{
    return hash(p.x + p.y*57.0);
}

// Function 461
float mynoise(vec2 u) {
    return noise(vec3(u,0));                // use procedural noise
 // return texture(iChannel0, u/256.).x;  // use image noise
}

// Function 462
float snoise(in vec2 v)
  {
  const vec4 C = vec4(0.211324865405187, // (3.0-sqrt(3.0))/6.0
                      0.366025403784439, // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626, // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
// First corner
  vec2 i = floor(v + dot(v, C.yy) );
  vec2 x0 = v - i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
+ i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// Function 463
float triNoise3d(in vec3 p)
{
    float z=1.4;
	float rz = 0.;
    vec3 bp = p;
	for (float i=0.; i<=3.; i++ )
	{
        vec3 dg = tri3(bp);
        p += (dg);

        bp *= 2.;
		z *= 1.5;
		p *= 1.2;
        //p.xz*= m2;
        
        rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
        bp += 0.14;
	}
	return rz;
}

// Function 464
vec4 denoise1( in sampler2D tex, in vec2 pix )
{
    //not bothering with 2-pass separable approach
    mat3 avg_sampler;
    avg_sampler[0] = vec3( 1.0, 2.0, 1.0 );
    avg_sampler[2] = avg_sampler[0];
    avg_sampler[1] = avg_sampler[0] * 2.0;
    avg_sampler /= 16.0;
    
    vec4 accum = vec4(0.0);
    for( int i = 0; i < 3; i++ )
    {
        for( int j = 0; j < 3; j++ )
        {
			vec2 uv = pix;
            uv += vec2( i,j);
            uv += vec2( -1.0, -1.0 );
            uv /= iResolution.xy;
            accum += texture( tex, uv ) * avg_sampler[i][j];
        }
    }
    
    return accum;
}

// Function 465
float noise(vec2 p)
{
    vec2 i = floor(p), f = fract(p); 
	f *= f*f*(3.-2.*f);
    return mix(mix(hash(i + vec2(0.,0.)), 
                   hash(i + vec2(1.,0.)), f.x),
               mix(hash(i + vec2(0.,1.)), 
                   hash(i + vec2(1.,1.)), f.x), f.y);
}

// Function 466
vec3 noise(vec3 p, float lod){float m = mod(p.z,1.0);float s = p.z-m; float sprev = s-1.0;if (mod(s,2.0)==1.0) { s--; sprev++; m = 1.0-m; };return mix(texture(iChannel0,p.xy/iChannelResolution[0].xy+noise(sprev,lod).yz,lod).xyz,texture(iChannel0,p.xy/iChannelResolution[0].xy+noise(s,lod).yz,lod).xyz,m);}

// Function 467
float noise3d( in vec3 p ) {
    vec3 idx = vec3(fract(sin(dot(p / 0.3, vec3(2.5,3.46,1.29))) * 12394.426),
                   fract(sin(dot(p / 0.17, vec3(3.987,2.567,3.76))) * 52422.82465),
                   fract(sin(dot(p / 0.44, vec3(6.32,3.87,5.24))) * 34256.267));
    //p.z *= p.z;
    //p.y *= p.y;
    //p.z = mix(p.y, p.z, idx.x * (1.0 - idx.y));
    p.xz = mod(p.xz - 0.5 * CELL_SIZE, vec2(CELL_SIZE));
    p.xz = rot(fract(sin(dot(idx.xz, vec2(3.124,1.75)))) * 312.2) * p.xz;
    float s = hash3d(1e4 * p + idx);
    return s;
}

// Function 468
float mynoise ( vec3 p)
{
	 return noise(p);
	 //return .5+.5*sin(50.*noise(p));
}

// Function 469
float NoiseGen(vec3 p) {
    // This is a bit faster if we use 2 accumulators instead of 1.
    // Timed on Linux/Chrome/TitanX Pascal
    float wave0 = 0.0;
    float wave1 = 0.0;
    wave0 += sin(dot(p, vec3(-1.316, 0.918, 1.398))) * 0.0783275458;
    wave1 += sin(dot(p, vec3(0.295, -0.176, 2.167))) * 0.0739931495;
    wave0 += sin(dot(p, vec3(-0.926, 1.445, 1.429))) * 0.0716716966;
    wave1 += sin(dot(p, vec3(-1.878, -0.174, 1.258))) * 0.0697839187;
    wave0 += sin(dot(p, vec3(-1.995, 0.661, -0.908))) * 0.0685409863;
    wave1 += sin(dot(p, vec3(-1.770, 1.350, -0.905))) * 0.0630152419;
    wave0 += sin(dot(p, vec3(2.116, -0.021, 1.161))) * 0.0625361712;
    wave1 += sin(dot(p, vec3(0.405, -1.712, -1.855))) * 0.0567751048;
    wave0 += sin(dot(p, vec3(1.346, 0.945, 1.999))) * 0.0556465603;
    wave1 += sin(dot(p, vec3(-0.397, -0.573, 2.495))) * 0.0555747667;
    wave0 += sin(dot(p, vec3(0.103, -2.457, -1.144))) * 0.0516322279;
    wave1 += sin(dot(p, vec3(-0.483, -1.323, 2.330))) * 0.0513093320;
    wave0 += sin(dot(p, vec3(-1.715, -1.810, -1.164))) * 0.0504567036;
    wave1 += sin(dot(p, vec3(2.529, 0.479, 1.011))) * 0.0500811899;
    wave0 += sin(dot(p, vec3(-1.643, -1.814, -1.437))) * 0.0480875812;
    wave1 += sin(dot(p, vec3(1.495, -1.905, -1.648))) * 0.0458268348;
    wave0 += sin(dot(p, vec3(-1.874, 1.559, 1.762))) * 0.0440084357;
    wave1 += sin(dot(p, vec3(1.068, -2.090, 2.081))) * 0.0413624154;
    wave0 += sin(dot(p, vec3(-0.647, -2.197, -2.237))) * 0.0401592830;
    wave1 += sin(dot(p, vec3(-2.146, -2.171, -1.135))) * 0.0391682940;
    wave0 += sin(dot(p, vec3(2.538, -1.854, -1.604))) * 0.0349588163;
    wave1 += sin(dot(p, vec3(1.687, 2.191, -2.270))) * 0.0342888847;
    wave0 += sin(dot(p, vec3(0.205, 2.617, -2.481))) * 0.0338465332;
    wave1 += sin(dot(p, vec3(3.297, -0.440, -2.317))) * 0.0289423448;
    wave0 += sin(dot(p, vec3(1.068, -1.944, 3.432))) * 0.0286404261;
    wave1 += sin(dot(p, vec3(-3.681, 1.068, 1.789))) * 0.0273625684;
    wave0 += sin(dot(p, vec3(3.116, 2.631, -1.658))) * 0.0259772492;
    wave1 += sin(dot(p, vec3(-1.992, -2.902, -2.954))) * 0.0245830241;
    wave0 += sin(dot(p, vec3(-2.409, -2.374, 3.116))) * 0.0245592756;
    wave1 += sin(dot(p, vec3(0.790, 1.768, 4.196))) * 0.0244078334;
    wave0 += sin(dot(p, vec3(-3.289, 1.007, 3.148))) * 0.0241328015;
    wave1 += sin(dot(p, vec3(3.421, -2.663, 3.262))) * 0.0199736126;
    wave0 += sin(dot(p, vec3(3.062, 2.621, 3.649))) * 0.0199230290;
    wave1 += sin(dot(p, vec3(4.422, -2.206, 2.621))) * 0.0192399437;
    wave0 += sin(dot(p, vec3(2.714, 3.022, 4.200))) * 0.0182510631;
    wave1 += sin(dot(p, vec3(-0.451, 4.143, -4.142))) * 0.0181293526;
    wave0 += sin(dot(p, vec3(-5.838, -0.360, -1.536))) * 0.0175114826;
    wave1 += sin(dot(p, vec3(-0.278, -4.565, 4.149))) * 0.0170799341;
    wave0 += sin(dot(p, vec3(-5.893, -0.163, -2.141))) * 0.0167655258;
    wave1 += sin(dot(p, vec3(4.855, -4.153, 0.606))) * 0.0163155335;
    wave0 += sin(dot(p, vec3(4.498, 0.987, -4.488))) * 0.0162770287;
    wave1 += sin(dot(p, vec3(-1.463, 5.321, -3.315))) * 0.0162569125;
    wave0 += sin(dot(p, vec3(-1.862, 4.386, 4.749))) * 0.0154338176;
    wave1 += sin(dot(p, vec3(0.563, 3.616, -5.751))) * 0.0151952226;
    wave0 += sin(dot(p, vec3(-0.126, 2.569, -6.349))) * 0.0151089405;
    wave1 += sin(dot(p, vec3(-5.094, 4.759, 0.186))) * 0.0147947096;
    wave0 += sin(dot(p, vec3(1.319, 5.713, 3.845))) * 0.0147035221;
    wave1 += sin(dot(p, vec3(7.141, -0.327, 1.420))) * 0.0140573910;
    wave0 += sin(dot(p, vec3(3.888, 6.543, 0.547))) * 0.0133309850;
    wave1 += sin(dot(p, vec3(-1.898, -3.563, -6.483))) * 0.0133171360;
    wave0 += sin(dot(p, vec3(1.719, 7.769, 0.340))) * 0.0126913718;
    wave1 += sin(dot(p, vec3(-2.210, -7.836, 0.102))) * 0.0123746071;
    wave0 += sin(dot(p, vec3(6.248, -5.451, 1.866))) * 0.0117861898;
    wave1 += sin(dot(p, vec3(1.627, -7.066, -4.732))) * 0.0115417453;
    wave0 += sin(dot(p, vec3(4.099, -7.704, 1.474))) * 0.0112591564;
    wave1 += sin(dot(p, vec3(7.357, 3.788, 3.204))) * 0.0112252325;
    wave0 += sin(dot(p, vec3(-2.797, 6.208, 6.253))) * 0.0107206906;
    wave1 += sin(dot(p, vec3(6.130, -5.335, -4.650))) * 0.0105693992;
    wave0 += sin(dot(p, vec3(5.276, -5.576, -5.438))) * 0.0105139072;
    wave1 += sin(dot(p, vec3(9.148, 2.530, -0.383))) * 0.0103996383;
    wave0 += sin(dot(p, vec3(3.894, 2.559, 8.357))) * 0.0103161113;
    wave1 += sin(dot(p, vec3(-6.604, 8.024, -0.289))) * 0.0094066875;
    wave0 += sin(dot(p, vec3(-5.925, 6.505, -6.403))) * 0.0089444733;
    wave1 += sin(dot(p, vec3(9.085, 10.331, -0.451))) * 0.0069245599;
    wave0 += sin(dot(p, vec3(-8.228, 6.323, -9.900))) * 0.0066251015;
    wave1 += sin(dot(p, vec3(10.029, -3.802, 12.151))) * 0.0058122824;
    wave0 += sin(dot(p, vec3(-10.151, -6.513, -11.063))) * 0.0057522358;
    wave1 += sin(dot(p, vec3(-1.773, -16.284, 2.828))) * 0.0056578101;
    wave0 += sin(dot(p, vec3(11.081, 8.687, -9.852))) * 0.0054614334;
    wave1 += sin(dot(p, vec3(-3.941, -4.386, 16.191))) * 0.0054454253;
    wave0 += sin(dot(p, vec3(-6.742, 2.133, -17.268))) * 0.0050050132;
    wave1 += sin(dot(p, vec3(-10.743, 5.698, 14.975))) * 0.0048323955;
    wave0 += sin(dot(p, vec3(-9.603, 12.472, 14.542))) * 0.0043264378;
    wave1 += sin(dot(p, vec3(13.515, 14.345, 8.481))) * 0.0043208884;
    wave0 += sin(dot(p, vec3(-10.330, 16.209, -9.742))) * 0.0043013736;
    wave1 += sin(dot(p, vec3(-8.580, -6.628, 19.191))) * 0.0042005922;
    wave0 += sin(dot(p, vec3(-17.154, 10.620, 11.828))) * 0.0039482427;
    wave1 += sin(dot(p, vec3(16.330, 14.123, -10.420))) * 0.0038474789;
    wave0 += sin(dot(p, vec3(-21.275, 10.768, -3.252))) * 0.0038320501;
    wave1 += sin(dot(p, vec3(1.744, 7.922, 23.152))) * 0.0037560829;
    wave0 += sin(dot(p, vec3(-3.895, 21.321, 12.006))) * 0.0037173885;
    wave1 += sin(dot(p, vec3(-22.705, 2.543, 10.695))) * 0.0036484394;
    wave0 += sin(dot(p, vec3(-13.053, -16.634, -13.993))) * 0.0036291121;
    wave1 += sin(dot(p, vec3(22.697, -11.230, 1.417))) * 0.0036280459;
    wave0 += sin(dot(p, vec3(20.646, 14.602, 3.400))) * 0.0036055008;
    wave1 += sin(dot(p, vec3(5.824, -8.717, -23.680))) * 0.0035501527;
    wave0 += sin(dot(p, vec3(6.691, 15.499, 20.079))) * 0.0035029508;
    wave1 += sin(dot(p, vec3(9.926, -22.778, 9.144))) * 0.0034694278;
    wave0 += sin(dot(p, vec3(-9.552, -27.491, 2.197))) * 0.0031359281;
    wave1 += sin(dot(p, vec3(21.071, -17.991, -11.566))) * 0.0030453280;
    wave0 += sin(dot(p, vec3(9.780, 1.783, 28.536))) * 0.0030251754;
    wave1 += sin(dot(p, vec3(8.738, -18.373, 22.725))) * 0.0029960272;
    wave0 += sin(dot(p, vec3(14.105, 25.703, -8.834))) * 0.0029840058;
    wave1 += sin(dot(p, vec3(-24.926, -17.766, -4.740))) * 0.0029487709;
    wave0 += sin(dot(p, vec3(1.060, -1.570, 32.535))) * 0.0027980099;
    wave1 += sin(dot(p, vec3(-24.532, -19.629, -16.759))) * 0.0025538949;
    wave0 += sin(dot(p, vec3(28.772, -21.183, -9.935))) * 0.0024494819;
    wave1 += sin(dot(p, vec3(-28.413, 22.959, 8.338))) * 0.0024236674;
    wave0 += sin(dot(p, vec3(-27.664, 22.197, 13.301))) * 0.0023965996;
    wave1 += sin(dot(p, vec3(-27.421, 20.643, 18.713))) * 0.0023203498;
    wave0 += sin(dot(p, vec3(18.961, -7.189, 35.907))) * 0.0021967023;
    wave1 += sin(dot(p, vec3(-23.949, 4.885, 33.762))) * 0.0021727461;
    wave0 += sin(dot(p, vec3(35.305, 8.594, 20.564))) * 0.0021689816;
    wave1 += sin(dot(p, vec3(30.364, -11.608, -27.199))) * 0.0021357139;
    wave0 += sin(dot(p, vec3(34.268, 26.742, 0.958))) * 0.0020807976;
    wave1 += sin(dot(p, vec3(-26.376, -17.313, -32.023))) * 0.0020108850;
    wave0 += sin(dot(p, vec3(31.860, -32.181, -2.834))) * 0.0019919601;
    wave1 += sin(dot(p, vec3(25.590, 32.340, 21.381))) * 0.0019446179;
    wave0 += sin(dot(p, vec3(-17.771, -23.941, 37.324))) * 0.0018898258;
    wave1 += sin(dot(p, vec3(-38.699, 19.953, -22.675))) * 0.0018379538;
    wave0 += sin(dot(p, vec3(-46.284, 11.672, -15.411))) * 0.0017980056;
    wave1 += sin(dot(p, vec3(-32.023, -43.976, -7.378))) * 0.0016399251;
    wave0 += sin(dot(p, vec3(-42.390, -21.165, -31.889))) * 0.0015752176;
    wave1 += sin(dot(p, vec3(-18.949, -40.461, 39.107))) * 0.0015141244;
    wave0 += sin(dot(p, vec3(-21.507, -5.939, -58.531))) * 0.0014339601;
    wave1 += sin(dot(p, vec3(-51.745, -43.821, 9.651))) * 0.0013096306;
    wave0 += sin(dot(p, vec3(39.239, 25.971, -52.615))) * 0.0012701774;
    wave1 += sin(dot(p, vec3(-49.669, -35.051, -36.306))) * 0.0012661695;
    wave0 += sin(dot(p, vec3(-49.996, 35.309, 38.460))) * 0.0012398870;
    wave1 += sin(dot(p, vec3(27.000, -65.904, -36.267))) * 0.0011199347;
    wave0 += sin(dot(p, vec3(-52.523, -26.557, 57.693))) * 0.0010856391;
    wave1 += sin(dot(p, vec3(-42.670, 0.269, -71.125))) * 0.0010786551;
    wave0 += sin(dot(p, vec3(-9.377, 64.575, -68.151))) * 0.0009468199;
    wave1 += sin(dot(p, vec3(14.571, -29.160, 106.329))) * 0.0008019719;
    wave0 += sin(dot(p, vec3(-21.549, 103.887, 36.882))) * 0.0007939609;
    wave1 += sin(dot(p, vec3(-42.781, 110.966, -9.070))) * 0.0007473261;
    wave0 += sin(dot(p, vec3(-112.686, 18.296, -37.920))) * 0.0007409259;
    wave1 += sin(dot(p, vec3(71.493, 33.838, -96.931))) * 0.0007121903;
    return wave0+wave1;
}

// Function 470
vec2 interpNoise2D(vec2 uv) {
    vec2 uvFract = fract(uv);
    vec2 ll = random2(floor(uv));
    vec2 lr = random2(floor(uv) + vec2(1,0));
    vec2 ul = random2(floor(uv) + vec2(0,1));
    vec2 ur = random2(floor(uv) + vec2(1,1));

    vec2 lerpXL = mySmoothStep(ll, lr, uvFract.x);
    vec2 lerpXU = mySmoothStep(ul, ur, uvFract.x);

    return mySmoothStep(lerpXL, lerpXU, uvFract.y);
}

// Function 471
float cnoise(vec2 p)
{
  vec4 pi = floor(p.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 pf = fract(p.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  pi = mod289(pi); // To avoid truncation effects in permutation
  vec4 ix = pi.xzxz;
  vec4 iy = pi.yyww;
  vec4 fx = pf.xzxz;
  vec4 fy = pf.yyww;

  vec4 i = permute(permute(ix) + iy);

  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
  vec4 gy = abs(gx) - 0.5 ;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;

  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);

  vec4 norm = taylor_inv_sqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;  
  g01 *= norm.y;  
  g10 *= norm.z;  
  g11 *= norm.w;  

  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));

  vec2 fade_xy = fade(pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

// Function 472
float stepNoise(vec2 p) {
    return noise(floor(p));
}

// Function 473
float NebulaNoise(vec3 p)
{
   float final = p.y + 4.5;
    final -= SpiralNoiseC(p.xyz);   // mid-range noise
    final += SpiralNoiseC(p.zxy*0.5123+100.0)*2.0;   // large scale features
    final -= SpiralNoise3D(p);   // more large scale features, but 3d

    return final;
}

// Function 474
float noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f *= f*(3.-2.*f);
    
    return mix( mix( hash(i+vec2(0., 0.)), hash(i+vec2(1., 0.)), f.x ),
           mix( hash(i+vec2(0., 1.)), hash(i+vec2(1., 1.)), f.x ), f.y);
}

// Function 475
vec3 fbmdPerlin(vec2 pos, vec2 scale, int octaves, vec2 shift, float axialShift, float gain, vec2 lacunarity, float slopeness, float octaveFactor, bool negative, float seed) 
{
    vec2 cosSin = vec2(cos(axialShift), sin(axialShift));
    mat2 transform = mat2(cosSin.x, cosSin.y, -cosSin.y, cosSin.x) * mat2(0.8, -0.6, 0.6, 0.8);
    return fbmdPerlin(pos, scale, octaves, shift, transform, gain, lacunarity, slopeness, octaveFactor, negative, seed);
}

// Function 476
vec4 getMultiNoise(in vec2 uv, in vec2 noiseUVscale, in float noiseBrightness)
{
    int octaves = 4;
    vec2 sScale = noiseUVscale.xy;
    float brightness = noiseBrightness;
    float brightMul = 0.2;
    float freqMul = 20.4;
    float zOffset = iTime * 0.1;
    vec4 anoise = vec4(0,0,0,0);
    for (int i = 0; i < octaves; ++i)
    {
     	vec4 noise = texture(iChannel1, vec3(uv * sScale,zOffset)) * brightness;
    	anoise += noise;
        sScale *= freqMul;
        brightness *= brightMul;
        zOffset += 0.1;
    }
    //anoise /= float(octaves);
    return anoise;
}

// Function 477
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
    vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
    return mix( rg.x, rg.y, f.z );
}

// Function 478
float valueNoise( in vec3 x, float tile ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix( valueHash(mod(p+vec3(0,0,0),tile)), 
                        valueHash(mod(p+vec3(1,0,0),tile)),f.x),
                   mix( valueHash(mod(p+vec3(0,1,0),tile)), 
                        valueHash(mod(p+vec3(1,1,0),tile)),f.x),f.y),
               mix(mix( valueHash(mod(p+vec3(0,0,1),tile)), 
                        valueHash(mod(p+vec3(1,0,1),tile)),f.x),
                   mix( valueHash(mod(p+vec3(0,1,1),tile)), 
                        valueHash(mod(p+vec3(1,1,1),tile)),f.x),f.y),f.z);
}

// Function 479
float planetNoise( in vec3 x )
{
   vec3 p = floor(x);
   vec3 f = fract(x);
f = f*f*(3.0-2.0*f);

vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
return mix( rg.x, rg.y, f.z );
}

// Function 480
float Noise(float time, float pitch)
{
    float ret = Hash(floor(time * pitch));
	return ret;
}

// Function 481
float voronoi3D(vec3 uv)
{
    vec3 fl = floor(uv);
    vec3 fr = fract(uv);
    float res = 1.0;
    for(int k=-1;k<=1;k++)
    for( int j=-1; j<=1; j++ ) {
        for( int i=-1; i<=1; i++ ) {
            vec3 p = vec3(i, j, k);
            #if defined(ENABLE_UINT_HASH)
            float h = random(fl+p);
            #else
            float h = hash3D(fl+p);
            #endif
            vec3 vp = p-fr+h;
            float d = dot(vp, vp);
            
            res +=1.0/pow(d, 16.0);
        }
    }
    return pow( 1.0/res, 1.0/16.0 );
}

// Function 482
vec2 noise2(in vec3 uv, in float shift_by) {
    return vec2(simple_noise(uv, shift_by),
                simple_noise(uv + vec3(0.0, 0.0, 101.0), shift_by));
}

// Function 483
float noise3f( in vec3 p, in int sem )
{
    ivec3 i = ivec3( floor(p) );
    vec3  f = p - vec3(i);

    // quintic smoothstep
    vec3 w = f*f*f*(f*(f*6.0-15.0)+10.0);

    int n = i.x + i.y * 57 + 113*i.z + sem;

	return 1.0 - 2.0*mix(mix(mix(hash(n+(0+57*0+113*0)),
                                 hash(n+(1+57*0+113*0)),w.x),
                             mix(hash(n+(0+57*1+113*0)),
                                 hash(n+(1+57*1+113*0)),w.x),w.y),
                         mix(mix(hash(n+(0+57*0+113*1)),
                                 hash(n+(1+57*0+113*1)),w.x),
                             mix(hash(n+(0+57*1+113*1)),
                                 hash(n+(1+57*1+113*1)),w.x),w.y),w.z);
}

// Function 484
float whangHashNoise(uint u, uint v, uint s)
{
    //return fract(sin(float(s + (u*1080u + v)%10000u) * 78.233) * 43758.5453);
    
    uint seed = (u*1664525u + v) + s;
    
    seed  = (seed ^ 61u) ^(seed >> 16u);
    seed *= 9u;
    seed  = seed ^(seed >> 4u);
    seed *= uint(0x27d4eb2d);
    seed  = seed ^(seed >> 15u);
    
    float value = float(seed) / (4294967296.0);
    return value;
}

// Function 485
float noise(in vec3 x){
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 486
float gavoronoi3(in vec2 p) {
  vec2 ip = floor(p);
  vec2 fp = fract(p);
  float f = 2.*PI;//frequency
  float v = .8;//cell variability <1.
  float dv = .4;//direction variability <1.
  vec2 dir = ga_m;// direction scale
  float va = 0.0;
  float wt = 0.0;
  for (int i=-1; i<=1; i++) {
    for (int j=-1; j<=1; j++) {
      vec2 o = vec2(i, j)-.5;
      vec2 h = hash2_ga(ip - o);
      vec2 pp = fp +o  -h;
      float d = dot(pp, pp);
      float w = exp(-d*4.);
      wt +=w;
      h = dv*h+dir;//h=normalize(h+dir);
      va += cos(dot(pp,h)*f/v)*w;
    }
  }
  return va/wt;
}

// Function 487
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0. ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 488
float getLayeredNoise(vec3 seed)
{
	return (0.5 * getNoise(seed * 0.05)) +
           (0.25 * getNoise(seed * 0.1)) +
           (0.125 * getNoise(seed * 0.2)) +
           (0.0625 * getNoise(seed * 0.4));
}

// Function 489
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    float n = mix( mix( dot( -1.0+2.0*hash( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                        dot( -1.0+2.0*hash( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                   mix( dot( -1.0+2.0*hash( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                        dot( -1.0+2.0*hash( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
	return 0.5 + 0.5*n;
}

// Function 490
float Noise(vec3 pos)
{
    pos *=2.0;
    vec3 p = floor(pos);
    vec3 f = fract(pos);
    vec2 uv = p.xy + vec2(37.0, 17.0)*p.z + f.xy;
    
    vec2 rg = texture(iChannel0, (uv+0.5)/256.0).yx;
    return mix(rg.x, rg.y, f.z)*2.0-1.0;
}

// Function 491
vec4 noise(ivec2 p){
    return noise(ivec3(p, 0));
}

// Function 492
float fractalNoise(vec2 p) {
  float total = 0.0;
  total += smoothNoise(p);
  total += smoothNoise(p*2.) / 2.;
  total += smoothNoise(p*4.) / 4.;
  total += smoothNoise(p*8.) / 8.;
  total += smoothNoise(p*16.) / 16.;
  total /= 1. + 1./2. + 1./4. + 1./8. + 1./16.;
  return total;
}

// Function 493
vec4 atm_cloudnoise1_d( vec4 r, float scale, bool lowfreq )
{
    float invscale = 1. / scale;
    float y = atm_cloudnoise1( invscale * r, lowfreq );
    return vec4( invscale * ( atm_cloudnoise1_offs( invscale * r, lowfreq ) - y ), y );
}

// Function 494
float valueNoise(vec2 p)
{
    vec2 ip = floor(p);
    vec2 fp = fract(p);
	vec2 ramp = fp*fp*(3.0-2.0*fp);

    float rz= mix( mix( hash12(ip + vec2(0.0,0.0)), hash12(ip + vec2(1.0,0.0)), ramp.x),
                   mix( hash12(ip + vec2(0.0,1.0)), hash12(ip + vec2(1.0,1.0)), ramp.x), ramp.y);
    
    return rz;
}

// Function 495
vec2 noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p); f = f*f*(3.-2.*f); // smoothstep
    vec2 v= mix( mix(hash22(i+vec2(0,0)),hash22(i+vec2(1,0)),f.x),
                 mix(hash22(i+vec2(0,1)),hash22(i+vec2(1,1)),f.x), f.y);
    return 2.*v-1.;
}

// Function 496
vec3 SmoothNoise_DXY(in vec2 o) 
{
	vec2 p = floor(o);
	vec2 f = fract(o);
		
	float n = p.x + p.y*57.0;

	float a = Hash(n+  0.0);
	float b = Hash(n+  1.0);
	float c = Hash(n+ 57.0);
	float d = Hash(n+ 58.0);
	
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;
	
	vec2 t = 3.0 * f2 - 2.0 * f3;
	vec2 dt = 6.0 * f - 6.0 * f2;
	
	float u = t.x;
	float v = t.y;
	float du = dt.x;	
	float dv = dt.y;	

	float res = a + (b-a)*u +(c-a)*v + (a-b+d-c)*u*v;
    
	float dx = (b-a)*du + (a-b+d-c)*du*v;
	float dy = (c-a)*dv + (a-b+d-c)*u*dv;    
    
    return vec3(dx, dy, res);
}

// Function 497
float cnoise(vec2 P)
{
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod289(Pi); // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;

  vec4 i = permute(permute(ix) + iy);

  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
  vec4 gy = abs(gx) - 0.5 ;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;

  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);

  vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;  
  g01 *= norm.y;  
  g10 *= norm.z;  
  g11 *= norm.w;  

  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));

  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

// Function 498
vec3 voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( iTime + 6.2831*o );
        #endif
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( iTime + 6.2831*o );
        #endif	
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>0.00001 )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec3( md, mr );
}

// Function 499
float perlinWorley(vec2 st){
    
    vec2 pixel_coords = st;
    
    float perlin = perlinFBM(pixel_coords/float(iResolution));
	float worley = worleyFBM(pixel_coords/float(iResolution));
	perlin = clamp(perlin, 0.0, 1.0);
	worley = clamp(worley, 0.0, 1.0);
    float worley2 = remap(worley, 0.0, 1.0, 0.0, 0.45);
	return remap( 1. - worley2, 0.0, 1.0, perlin, 1.0);
    
}

// Function 500
float vnoise(vec2 x) {
  vec2 i = floor(x);
  vec2 w = fract(x);

#if 1
  // quintic interpolation
  vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);
#else
  // cubic interpolation
  vec2 u = w*w*(3.0-2.0*w);
#endif

  float a = hash(i+vec2(0.0,0.0));
  float b = hash(i+vec2(1.0,0.0));
  float c = hash(i+vec2(0.0,1.0));
  float d = hash(i+vec2(1.0,1.0));

  float k0 =   a;
  float k1 =   b - a;
  float k2 =   c - a;
  float k3 =   d - c + a - b;

  float aa = mix(a, b, u.x);
  float bb = mix(c, d, u.x);
  float cc = mix(aa, bb, u.y);

  return k0 + k1*u.x + k2*u.y + k3*u.x*u.y;
}

// Function 501
float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = rand(i);
    float b = rand(i + vec2(1.0, 0.0));
    float c = rand(i + vec2(0.0, 1.0));
    float d = rand(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Function 502
float perlinFBM(vec2 st){
 	return perlinFBM( vec3(st + iTime/20.0*vec2(1.,0),iTime/50.0));   
}

// Function 503
float valueNoise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash22( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( hash22( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hash22( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( hash22( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 504
float noise( in vec2 x ) {
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(1.5-f)*2.0;
    
    float res = mix(mix( hash12(p), hash12(p + add.xy),f.x),
                    mix( hash12(p + add.yx), hash12(p + add.xx),f.x),f.y);
    return res;
}

// Function 505
vec2 waveNoise(float time)
{
    return textureLod(iChannel0, vec2(time, time + kPI * 0.5), 0.0).rg*2.0 - oz.xx;
}

// Function 506
vec4 Noise( in vec2 x )
{
    vec2 p = floor(x.xy);
    vec2 f = fract(x.xy);
	f = f*f*(3.0-2.0*f);
//	vec3 f2 = f*f; f = f*f2*(10.0-15.0*f+6.0*f2);

	// there's an artefact because the y channel almost, but not exactly, matches the r channel shifted (37,17)
	// this artefact doesn't seem to show up in chrome, so I suspect firefox uses different texture compression.
	vec2 uv = p.xy + f.xy;
	return textureLod( iChannel0, (uv+0.5)/256.0, 0.0 );
}

// Function 507
float noise( in vec3 p )
{
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);

    return mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// Function 508
float noise( in vec3 x )
{
    #if 0

    // 3D texture
    return texture(iChannel2, x*.03).x*1.05;
    
	#else
    
    // Use 2D texture...
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);

	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel0, (uv+ 0.5)/256.0, -99.0).yx;
	return mix( rg.x, rg.y, f.z );
    
    #endif
}

// Function 509
vec2 lpnoise(float t, float f
){vec2 g=fr(t*f)
 ;return mix(noise(g.y/f),noise((g.y+1.)/f),smoothstep(0.,1.,g.x));}

// Function 510
float valueNoiseSimpleLow(vec2 vl) {

   const vec2 helper = vec2(0., 1.);
    vec2 interp = smoothstep(vec2(0.), vec2(1.), fract(vl));
    vec2 grid = floor(vl);

    return mix(mix(rand2(grid + helper.xx),
                   rand2(grid + helper.yx),
                   interp.x),
               mix(rand2(grid + helper.xy),
                   rand2(grid + helper.yy),
                   interp.x),
               interp.y);
}

// Function 511
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);

    return mix(mix( hash(p+vec2(0,0)), 
                   hash(p+vec2(1,0)),f.x),
               mix( hash(p+vec2(0,1)), 
                   hash(p+vec2(1,1)),f.x),f.y);
}

// Function 512
float noise_value( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( hash21( i + vec2(0.0,0.0) ), 
                     hash21( i + vec2(1.0,0.0) ), u.x),
                mix( hash21( i + vec2(0.0,1.0) ), 
                     hash21( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 513
float noise(vec2 x)
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    vec2 i = f*f*(3.0-2.0*f);
    return mix(mix(hash(i+vec2(0.0, 0.0)),
                   hash(i+vec2(0.0, 1.0)), x.x),
               mix(hash(i+vec2(1.0, 0.0)),
                   hash(i+vec2(1.0, 1.0)), x.x),x.y);
                   
                   
    
}

// Function 514
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);
    float t = pow(2.,level)* .2*iTime;
    mat2 R = mat2(cos(t),-sin(t),sin(t),cos(t));
    if (mod(i.x+i.y,2.)==0.) R=-R;

    return 2.*mix( mix( dot( hash( i + vec2(0,0) ), (f - vec2(0,0))*R ), 
                     dot( hash( i + vec2(1,0) ),-(f - vec2(1,0))*R ), u.x),
                mix( dot( hash( i + vec2(0,1) ),-(f - vec2(0,1))*R ), 
                     dot( hash( i + vec2(1,1) ), (f - vec2(1,1))*R ), u.x), u.y);
}

// Function 515
vec2 noise2(vec2 xy) {
    float a = noise(xy), b = noise(vec2(a+xy[1], a-xy[0]));
    
	return normalize(vec2(a, b))*0.35;
}

// Function 516
float noise(vec3 uv)
{
    vec3 fr = fract(uv.xyz);
    vec3 fl = floor(uv.xyz);
    float h000 = Hash3d(fl);
    float h100 = Hash3d(fl + zeroOne.yxx);
    float h010 = Hash3d(fl + zeroOne.xyx);
    float h110 = Hash3d(fl + zeroOne.yyx);
    float h001 = Hash3d(fl + zeroOne.xxy);
    float h101 = Hash3d(fl + zeroOne.yxy);
    float h011 = Hash3d(fl + zeroOne.xyy);
    float h111 = Hash3d(fl + zeroOne.yyy);
    return mixP(
        mixP(mixP(h000, h100, fr.x), mixP(h010, h110, fr.x), fr.y),
        mixP(mixP(h001, h101, fr.x), mixP(h011, h111, fr.x), fr.y)
        , fr.z);
}

// Function 517
float noise01(vec2 p){return clamp((noise(p)+.5)*.5, 0.,1.);}

// Function 518
vec4 trn_noise_d( vec3 x, out vec4 p, out vec4 q )
{
	vec3 xf = fract(x);
	vec3 xi = floor(x);
    ivec3 ix = ivec3( xi );
    p = vec4( rand( ix + ivec3( 0, 0, 0 ) ), rand( ix + ivec3( 1, 0, 0 ) ),
			  rand( ix + ivec3( 0, 1, 0 ) ), rand( ix + ivec3( 1, 1, 0 ) ) );
    q = vec4( rand( ix + ivec3( 0, 0, 1 ) ), rand( ix + ivec3( 1, 0, 1 ) ),
			  rand( ix + ivec3( 0, 1, 1 ) ), rand( ix + ivec3( 1, 1, 1 ) ) );
    vec3 t = xf - .5;
    vec3 u = .5 - 2. * ( abs(t) * t - t );
    vec3 v = 2. - 4. * abs(t);
    vec4 dpq = q - p;
    vec4 pqz = mix( p, q, u.z );
    return vec4(
		mix( pqz.yz - pqz.xx, pqz.ww - pqz.zy, u.yx ) * v.xy,
        mix( mix( dpq.x, dpq.y, u.x ), mix( dpq.z, dpq.w, u.x ), u.y ) * v.z,
    	mix( mix( pqz.x, pqz.y, u.x ), mix( pqz.z, pqz.w, u.x ), u.y ) );
}

// Function 519
vec2 perlin(vec2 p)
{
    vec2 x = vec2(0.0);
    for (int i = 0; i < 6; ++i)
    {
        float j = pow(2.0, float(i));
        x += (texture(iChannel1, p * j * 0.001).xy-0.5) / j;
    }
    return x;
}

// Function 520
float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p); f = f*f*(3.-2.*f); // smoothstep

    float v= mix( mix(hash21(i+vec2(0,0)),hash21(i+vec2(1,0)),f.x),
                  mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x), f.y);
	return 2.*v-1.;
}

// Function 521
vec2 Noisev2v4 (vec4 p)
{
  vec4 ip, fp, t1, t2;
  ip = floor (p);  fp = fract (p);
  fp = fp * fp * (3. - 2. * fp);
  t1 = Hashv4f (dot (ip.xy, cHashA3.xy));
  t2 = Hashv4f (dot (ip.zw, cHashA3.xy));
  return vec2 (mix (mix (t1.x, t1.y, fp.x), mix (t1.z, t1.w, fp.x), fp.y),
               mix (mix (t2.x, t2.y, fp.z), mix (t2.z, t2.w, fp.z), fp.w));
}

// Function 522
float snoise(vec3 v
){const vec2 C=vec2(1,2)/6.
 ;const vec4 D=vec4(0,.5,1,2)
 ;vec3 i=floor(v+dot(v,C.yyy))
 ;vec3 x0=v-i+dot(i,C.xxx)
 ;vec3 g=step(x0.yzx,x0.xyz)
 ;vec3 l=1.-g
 ;vec3 j1=min(g.xyz,l.zxy)
 ;vec3 i2=max(g.xyz,l.zxy)
 ;vec3 x1=x0-j1+C.xxx
 ;vec3 x2=x0-i2+C.yyy
 ;vec3 x3=x0-D.yyy
 ;i=mod289(i)
 ;vec4 p=vec4(0)
 ;p=permute(p+i.z+vec4(0,j1.z,i2.z,1))
 ;p=permute(p+i.y+vec4(0,j1.y,i2.y,1)) 
 ;p=permute(p+i.x+vec4(0,j1.x,i2.x,1))
 ;//Gradients: 7x7 points over a square, mapped onto an octahedron.
 ;//ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
 ;float n_=1./7.
 ;vec3 ns=n_*D.wyz-D.xzx
 ;vec4 j=p-49.*floor(p*ns.z*ns.z)
 ;vec4 x_=floor(j*ns.z);
 ;vec4 y_=floor(j-7.*x_)
 ;vec4 x =x_*ns.x+ns.yyyy
 ;vec4 y =y_*ns.x+ns.yyyy
 ;vec4 h =1.-abs(x)-abs(y)
 ;vec4 b0=vec4(x.xy,y.xy)
 ;vec4 b1=vec4(x.zw,y.zw)
 ;vec4 s0=u3(floor(b0))
 ;vec4 s1=u3(floor(b1))
 ;vec4 sh=-step(h,vec4(0))
 ;vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy
 ;vec4 a1=b1.xzyw+s1.xzyw*sh.zzww
 ;vec3 p0=vec3(a0.xy,h.x)
 ;vec3 p1=vec3(a0.zw,h.y)
 ;vec3 p2=vec3(a1.xy,h.z)
 ;vec3 p3=vec3(a1.zw,h.w)
 ;a0=taylorInvSqrt(vec4(dd(p0),dd(p1),dd(p2),dd(p3)))
 ;p0*=a0.x;
 ;p1*=a0.y;
 ;p2*=a0.z;
 ;p3*=a0.w
 ;vec4 m=max(.6-vec4(dd(x0),dd(x1),dd(x2),dd(x3)),0.)
 ;m*=m
 ;return 42.*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));}

// Function 523
ivec2 perlinNoiseWarp(ivec2 coord, int baseSeed, float strength, float scale)
{
    vec2 warpCoord = vec2(coord)/scale;
    vec2 warpDisplacement = vec2(perlinNoise(warpCoord, baseSeed+0), perlinNoise(warpCoord, baseSeed+1));
    return coord+ivec2(strength*warpDisplacement);
}

// Function 524
float texNoise(vec2 uv){ float f = 0.; f+=texture(iChannel0, uv*.125).r*.5;
    f+=texture(iChannel0,uv*.25).r*.25;f+=texture(iChannel0,uv*.5).r*.125;
    f+=texture(iChannel0,uv*1.).r*.125;f=pow(f,1.2);return f*.45+.05;
}

// Function 525
float mul_noise(vec3 x) {
    float n = 2.*noise(x);  x *= 2.1; // return n/2.;
         n *= 2.*noise(x);  x *= 1.9;
         n *= 2.*noise(x);  x *= 2.3;
         n *= 2.*noise(x);  x *= 1.9;
         n *= 2.*noise(x);
    return n/2.; 
}

// Function 526
vec3 noise( in float x )
{
    float p = floor(x);
    float f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix( hash3(p+0.0), hash3(p+1.0),f);
}

// Function 527
void UpdateVoronoi(inout particle U, in vec3 p)
{
    //check neighbours 
    CheckRadius(U, p, 1);
    CheckRadius(U, p, 2);
    CheckRadius(U, p, 3);
    CheckRadius(U, p, 4);
}

// Function 528
float snoise(vec3 v)
  {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i = floor(v + dot(v, C.yyy) );
  vec3 x0 = v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  // x0 = x0 - 0.0 + 0.0 * C.xxx;
  // x1 = x0 - i1 + 1.0 * C.xxx;
  // x2 = x0 - i2 + 2.0 * C.xxx;
  // x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy; // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z); // mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ ); // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
  }

// Function 529
float floorNoise(vec2 x)
{
    float sum = 0.0;
    float fd = float(DEPTH);
    x += hash(x + iTime) * 0.5;
    for(int i = 1; i < DEPTH; ++i)
    {
        x += hash(x) * 0.01;
        float pdepth = (fd/float(i));
        float sx = floor(mod(x.x - pdepth, 2.0)) + floor(mod(x.x + pdepth, 2.0));
        float sy = floor(mod(x.y - pdepth, 2.0)) + floor(mod(x.y + pdepth, 2.0));
        sum += sx + sy;
    }
    // the max height will approch (pi - 3)
	// this will roughly aprox sin(x*pi)(pi-3)
    return min((sum/(fd * 2.0)) - 0.5, 1.0);
}

// Function 530
vec3 blobnoisenrms(vec2 uv, float s)
{
    float d = 0.01;
    return normalize(
           vec3(blobnoises(uv + vec2(  d, 0.0), s) - blobnoises(uv + vec2( -d, 0.0), s),
                blobnoises(uv + vec2(0.0,   d), s) - blobnoises(uv + vec2(0.0,  -d), s),
                d));
}

// Function 531
float Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy + vec2(37.0, 17.0) * p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+.5)/256., 0.).yx;
	return mix(rg.x, rg.y, f.z);
}

// Function 532
vec3 voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec3( md, mr );
}

// Function 533
float snoise(vec3 p){
    
    const vec3 s=vec3(7,157,113);	
    vec3 ip=floor(p);    
    vec4 h=vec4(0.,s.yz,s.y+s.z)+dot(ip,s);    
    p-=ip;
    p=p*p*(3.-2.*p);
    h=mix(fract(sin(h)*43758.5453),fract(sin(h+s.x)*43758.5453),p.x);
    h.xy=mix(h.xz,h.yw,p.y);
    return mix(h.x,h.y,p.z);	
}

// Function 534
float Noise(in vec2 x) {
    vec2 p=floor(x), f=fract(x);
    f *= f*(3.-2.*f);
    float n = p.x + p.y*57.;
    return mix(mix(hash(n+ 0.), hash(n+ 1.),f.x),
               mix(hash(n+57.), hash(n+58.),f.x),f.y);

}

// Function 535
float noise( in vec3 x )
{
    // grid
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);

    
    // gradients
    vec3 ga = hash( p+vec3(0.0,0.0,0.0) );
    vec3 gb = hash( p+vec3(1.0,0.0,0.0) );
    vec3 gc = hash( p+vec3(0.0,1.0,0.0) );
    vec3 gd = hash( p+vec3(1.0,1.0,0.0) );
    vec3 ge = hash( p+vec3(0.0,0.0,1.0) );
    vec3 gf = hash( p+vec3(1.0,0.0,1.0) );
    vec3 gg = hash( p+vec3(0.0,1.0,1.0) );
    vec3 gh = hash( p+vec3(1.0,1.0,1.0) );
    
    // projections
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
	
    // interpolation
    return va + 
           u.x*(vb-va) + 
           u.y*(vc-va) + 
           u.z*(ve-va) + 
           u.x*u.y*(va-vb-vc+vd) + 
           u.y*u.z*(va-vc-ve+vg) + 
           u.z*u.x*(va-vb-ve+vf) + 
           u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);
}

// Function 536
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 537
vec3 voronoi(in vec2 x)
{
	vec2 n = floor(x);
	vec2 f = fract(x);

	vec2 mg, mr;
	
	float md = 8.0;
	for(int j = -1; j <= 1; j ++)
	{
		for(int i = -1; i <= 1; i ++)
		{
			vec2 g = vec2(float(i),float(j));
			vec2 o = hash(n + g);
			vec2 r = g + o - f;
			float d = max(abs(r.x), abs(r.y));
			
			if(d < md)
			{
				md = d;
				mr = r;
				mg = g;
			}
		}
	}
	
	return vec3(n + mg, mr);
}

// Function 538
float noise(vec2 p){
    vec2 ip = floor(p);
    vec2 u = fract(p);
    u = u*u*(3.0-2.0*u);

    float res = mix(
        mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
        mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
    return res*res;
}

// Function 539
float snoise(vec3 uv, float res)
{
	const vec3 s = vec3(1e0, 1e2, 1e3);
	
	uv *= res;
	
	vec3 uv0 = floor(mod(uv, res))*s;
	vec3 uv1 = floor(mod(uv+vec3(1.), res))*s;
	
	vec3 f = fract(uv); f = f*f*(3.0-2.0*f);

	vec4 v = vec4(uv0.x+uv0.y+uv0.z, uv1.x+uv0.y+uv0.z,
		      	  uv0.x+uv1.y+uv0.z, uv1.x+uv1.y+uv0.z);

	vec4 r = fract(sin(v*1e-1)*1e3);
	float r0 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
	
	r = fract(sin((v + uv1.z - uv0.z)*1e-1)*1e3);
	float r1 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
	
	return mix(r0, r1, f.z)*2.-1.;
}

// Function 540
float noise( in vec2 p )
{
	return sin(p.x)*sin(p.y);
}

// Function 541
float sunSurfaceNoise2(vec3 spos, float time)
{
    float s = 0.28;
    float detail = 3.0;
    for(int i = 0; i < 2; ++i)
    {
        float warp = noise(spos*8.0 * detail + time);
        float n = noise(vec3(spos.xy * detail / spos.z + vec2(warp, 0.0), time * detail / 10.0 + float(i) * 0.618033989));
        n = pow(n, 5.0-float(i));
        s += n / detail;
        detail *= 1.847;
    }
    return s;
}

// Function 542
float valueNoise(vec3 x, float freq)
{
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3. - 2. * f);
	
    return mix(mix(mix(hash13(mod(i + vec3(0, 0, 0), freq)),  
                       hash13(mod(i + vec3(1, 0, 0), freq)), f.x),
                   mix(hash13(mod(i + vec3(0, 1, 0), freq)),  
                       hash13(mod(i + vec3(1, 1, 0), freq)), f.x), f.y),
               mix(mix(hash13(mod(i + vec3(0, 0, 1), freq)),  
                       hash13(mod(i + vec3(1, 0, 1), freq)), f.x),
                   mix(hash13(mod(i + vec3(0, 1, 1), freq)),  
                       hash13(mod(i + vec3(1, 1, 1), freq)), f.x), f.y), f.z);
}

// Function 543
float Noise(vec2 uv) {
    vec2 id = floor(uv);
    vec2 m = fract(uv);
    m = 3.* m * m - 2.* m * m * m;
    
    float top = mix(N2(id.x, id.y), N2(id.x+1., id.y), m.x);
    float bot = mix(N2(id.x, id.y+1.), N2(id.x+1., id.y+1.), m.x);
    
    return mix(top, bot, m.y);
}

// Function 544
vec4 bccNoiseBase(vec3 X) {
    
    // First half-lattice, closest edge
    vec3 v1 = round(X);
    vec3 d1 = X - v1;
    vec3 score1 = abs(d1);
    vec3 dir1 = step(max(score1.yzx, score1.zxy), score1);
    vec3 v2 = v1 + dir1 * sign(d1);
    vec3 d2 = X - v2;
    
    // Second half-lattice, closest edge
    vec3 X2 = X + 144.5;
    vec3 v3 = round(X2);
    vec3 d3 = X2 - v3;
    vec3 score2 = abs(d3);
    vec3 dir2 = step(max(score2.yzx, score2.zxy), score2);
    vec3 v4 = v3 + dir2 * sign(d3);
    vec3 d4 = X2 - v4;
    
    // Gradient hashes for the four points, two from each half-lattice
    vec4 hashes = permute(mod(vec4(v1.x, v2.x, v3.x, v4.x), 289.0));
    hashes = permute(mod(hashes + vec4(v1.y, v2.y, v3.y, v4.y), 289.0));
    hashes = mod(permute(mod(hashes + vec4(v1.z, v2.z, v3.z, v4.z), 289.0)), 48.0);
    
    // Gradient extrapolations & kernel function
    vec4 a = max(0.5 - vec4(dot(d1, d1), dot(d2, d2), dot(d3, d3), dot(d4, d4)), 0.0);
    vec4 aa = a * a; vec4 aaaa = aa * aa;
    vec3 g1 = grad(hashes.x); vec3 g2 = grad(hashes.y);
    vec3 g3 = grad(hashes.z); vec3 g4 = grad(hashes.w);
    vec4 extrapolations = vec4(dot(d1, g1), dot(d2, g2), dot(d3, g3), dot(d4, g4));
    
    // Derivatives of the noise
    vec3 derivative = -8.0 * mat4x3(d1, d2, d3, d4) * (aa * a * extrapolations)
        + mat4x3(g1, g2, g3, g4) * aaaa;
    
    // Return it all as a vec4
    return vec4(derivative, dot(aaaa, extrapolations));
}

// Function 545
float noise( in vec3 x )
{
	vec3 p = floor(x);
	vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 546
float tetraNoise(in vec3 p)
{
    vec3 i = floor(p + dot(p, vec3(0.333333)) );  p -= i - dot(i, vec3(0.166666)) ;
    vec3 i1 = step(p.yzx, p), i2 = max(i1, 1.0-i1.zxy); i1 = min(i1, 1.0-i1.zxy);    
    vec3 p1 = p - i1 + 0.166666, p2 = p - i2 + 0.333333, p3 = p - 0.5;
    vec4 v = max(0.5 - vec4(dot(p,p), dot(p1,p1), dot(p2,p2), dot(p3,p3)), 0.0);
    vec4 d = vec4(dot(p, hash33(i)), dot(p1, hash33(i + i1)), dot(p2, hash33(i + i2)), dot(p3, hash33(i + 1.)));
    return clamp(dot(d, v*v*v*8.)*1.732 + .5, 0., 1.); // Not sure if clamping is necessary. Might be overkill.
}

// Function 547
float craterNoise3D(in vec3 p){
	
    
    float radius = 0.30;
    float slope = .32;
    float frequency = 1.0;
    float depth = -0.22;
    float rimWidth = 0.2;
    
	float fractal = fbm(p * frequency * 2.0) * 0.17;
	float cell = gpuCellNoise3D((p * frequency) + fractal ).x;
	float r = radius + fractal;
	float crater = smoothstep(slope, r, cell);
	  	  crater = mix(depth, crater, crater);
	float rim = 1.0 - smoothstep(r, r + rimWidth, cell);
	      crater = rim - (1.0 - crater);
return crater * 0.175;
}

// Function 548
float noise(in vec3 position)
{
        return texture(iChannel1, position.xy).r;
}

// Function 549
float tilingNoise(vec2 position, float size) {
    float value = snoise(vec3(position * size, 0.0));
    
    float wrapx = snoise(vec3(position * size - vec2(size, 0.0), 0.0));    
    value = mix(value, wrapx, max(0.0, position.x * size - (size - 1.0)));

    float wrapy = snoise(vec3(position * size - vec2(0.0, size), 0.0));
    float wrapxy = snoise(vec3(position * size - vec2(size, size), 0.0)); 
    wrapy = mix(wrapy, wrapxy, max(0.0, position.x * size - (size - 1.0)));
	return mix(value, wrapy, max(0.0, position.y * size - (size - 1.0)));
}

// Function 550
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
                              hash( i + vec2(1.0,0.0) ), u.x),
                         mix( hash( i + vec2(0.0,1.0) ), 
                              hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 551
vec3 simplexRotD(vec2 u,float rot
){vec2 d0,d1,d2,p0,p1,p2;vec3 x,y;NoiseHead(u,x,y,d0,d1,d2,p0,p1,p2)
 ;x=x+.5*y;x=mod289(x);y=mod289(y)        //without TilingPeriod 
 ;return NoiseDer(x,y,rot,d0,d1,d2);}

// Function 552
2D noise (lerp between grid point noise values
float noise(vec2 uv) {
    vec2 fuv = floor(uv);
    vec4 cell = vec4(
        hash(fuv + vec2(0, 0)),
        hash(fuv + vec2(0, 1)),
        hash(fuv + vec2(1, 0)),
        hash(fuv + vec2(1, 1))
    );
    vec2 axis = mix(cell.xz, cell.yw, fract(uv.y));
    return mix(axis.x, axis.y, fract(uv.x));
}

// Function 553
vec3 gradientNoised(vec2 pos, vec2 scale, mat2 transform, float seed) 
{
    // gradient noise with derivatives based on Inigo Quilez
    pos *= scale;
    vec4 i = floor(pos).xyxy + vec2(0.0, 1.0).xxyy;
    vec4 f = (pos.xyxy - i.xyxy) - vec2(0.0, 1.0).xxyy;
    i = mod(i, scale.xyxy) + seed;
    
    vec4 hashX, hashY;
    smultiHash2D(i, hashX, hashY);

    // transform gradients
    vec4 m = vec4(transform);
    vec4 rh = vec4(hashX.x, hashY.x, hashX.y, hashY.y);
    rh = rh.xxzz * m.xyxy + rh.yyww * m.zwzw;
    hashX.xy = rh.xz;
    hashY.xy = rh.yw;

    rh = vec4(hashX.z, hashY.z, hashX.w, hashY.w);
    rh = rh.xxzz * m.xyxy + rh.yyww * m.zwzw;
    hashX.zw = rh.xz;
    hashY.zw = rh.yw;
    
    vec2 a = vec2(hashX.x, hashY.x);
    vec2 b = vec2(hashX.y, hashY.y);
    vec2 c = vec2(hashX.z, hashY.z);
    vec2 d = vec2(hashX.w, hashY.w);
    
    vec4 gradients = hashX * f.xzxz + hashY * f.yyww;

    vec4 udu = noiseInterpolateDu(f.xy);
    vec2 u = udu.xy;
    vec2 g = mix(gradients.xz, gradients.yw, u.x);
    
    vec2 dxdy = a + u.x * (b - a) + u.y * (c - a) + u.x * u.y * (a - b - c + d);
    dxdy += udu.zw * (u.yx * (gradients.x - gradients.y - gradients.z + gradients.w) + gradients.yz - gradients.x);
    return vec3(mix(g.x, g.y, u.y) * 1.4142135623730950, dxdy);
}

// Function 554
vec3 TileableCurlNoise(vec3 p, in float numCells, int octaves)
{
  const float e = .1;
  vec3 dx = vec3( e   , 0.0 , 0.0 );
  vec3 dy = vec3( 0.0 , e   , 0.0 );
  vec3 dz = vec3( 0.0 , 0.0 , e   );

  vec3 p_x0 = snoiseVec3( p - dx, numCells, octaves );
  vec3 p_x1 = snoiseVec3( p + dx, numCells, octaves );
  vec3 p_y0 = snoiseVec3( p - dy, numCells, octaves );
  vec3 p_y1 = snoiseVec3( p + dy, numCells, octaves );
  vec3 p_z0 = snoiseVec3( p - dz, numCells, octaves );
  vec3 p_z1 = snoiseVec3( p + dz, numCells, octaves );

  float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
  float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
  float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

  const float divisor = 1.0 / ( 2.0 * e );
  return normalize( vec3( x , y , z ) * divisor );
  // technically incorrect but I like this better...
  //return normalize(vec3( x , y , z ));
}

// Function 555
vec2 noiseTex2(in vec3 x)
{
    vec3 fl = floor(x);
    vec3 fr = fract(x);
	fr = fr * fr * (3.0 - 2.0 * fr);
	vec2 uv = (fl.xy + vec2(37.0, 17.0) * fl.z) + fr.xy;
	vec4 rgba = textureLod(iChannel0, (uv + 0.5) * 0.00390625, 0.0 ).xyzw;
	return mix(rgba.yw, rgba.xz, fr.z);
}

// Function 556
vec4 TurbulenceNoise( in vec3 p )
{
	float f=noise(p);
	vec3 n=GrNoise(p);
	return vec4(-sign(f)*n,-abs(f));
}

// Function 557
float noise( in vec2 x ) {
    vec2 p = floor(x);
    vec2 w = fract(x);
    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);

    float a = hash2(p+vec2(0,0));
    float b = hash2(p+vec2(1,0));
    float c = hash2(p+vec2(0,1));
    float d = hash2(p+vec2(1,1));

    return -1.0+2.0*( a + (b-a)*u.x + (c-a)*u.y + (a - b - c + d)*u.x*u.y );
}

// Function 558
vec3 voronoi( in vec2 x ){
    vec2 n = floor( x );
    vec2 f = fract( x );

	float dis          = 4.;
    vec2  voronoipoint = vec2(0);
    
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {	
        //cell around
        vec2  g = vec2(i,j);
        
        //o MUST be a point between (0,0) and (1,1) for the algorithm to work
      	vec2  o = hash( n + g );//random number between (0,0) and (1,1) corresponding to cellid = n+g
        
              o = -cos(o*iTime)*.5+.5;//animation(notice how it keeps o between (0,0) and (1,1))
        									//it also maps (0,0) back to (0,0) on purpose
        
        vec2  delta = g+o-f;//delta between (n+g+o = voronoi_point) and (n+f = x)
        
      	
      //float d = max(abs(delta.x)*.866025403785+delta.y*.5,-delta.y);;//triangular distance metric
	  //float d = abs(delta.x)+abs(delta.y);//manhattan distance metric
        float d = sqrt(dot(delta,delta));//euclidian distance metric
      
        
        if( d<dis ){
            dis          = d;
            voronoipoint = n+g+o;
        }

    }

    return vec3( dis, voronoipoint );
}

// Function 559
float noise( in vec3 x )
{
	float  z = x.z*64.0;
	vec2 offz = vec2(0.317,0.123);
	vec2 uv1 = x.xy + offz*floor(z); 
	vec2 uv2 = uv1  + offz;
	return mix(textureLod( iChannel0, uv1 ,0.0).x,textureLod( iChannel0, uv2 ,0.0).x,fract(z))-0.5;
}

// Function 560
vec3 voronoi(in vec2 uv, inout vec3 col)
{
    vec2 x = uv + vec2(1.0, 0.0) * iTime;
    
    vec2 n = floor(x);
    vec2 f = x - n;

	float md = ZOOM;
    float td = 0.0;
    
    #if DEBUG
	float aspectAdjust = max(iResolution.x, iResolution.y);
    vec2 click = iMouse.xy / aspectAdjust * ZOOM;
    click += vec2(1.0, 0.0) * iTime;
    #endif
    
    mat2 rot = rotate2D(iTime);
    
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2  g = vec2( float(i), float(j) );
        vec2 h = hash(n + g);
        vec2 lo = h - 0.5;
        lo = rot * lo;
        
        vec2  o = vec2(0.5) + lo;
      	vec2  r = g - f + o;
		float d = dot( r, r );
        float potential = 1.0 / d;
        
        if (d < md)
            md = d;
        
        td += potential;
        
        #if DEBUG
        {
            vec2 n = floor(click);
            vec2 f = click - n;
            
			vec2 h = hash(n + g);
        	vec2 lo = h - 0.5;
        	lo = rot * lo;      
            
			vec2  o = vec2(0.5) + lo;
      		vec2  r = g - f + o;            
            float d = dot( r, r );
            
            float t = line(x, click, n + g + o);
            
            vec3 lineCol = vec3(1.0/d);
            col = mix(col, lineCol, 1.0-smoothstep(0.0, 25.0/aspectAdjust, abs(t)));
        }
        #endif
    }
    return vec3(td, sqrt(md), hash(n));
}

// Function 561
float noise (in vec3 p) {
	vec3 f = fract (p);
	p = floor (p);
	f = f * f * (3. - 2. * f);
	f.xy += p.xy + p.z * vec2 (37., 17.);
	f.xy = texture (iChannel0, (f.xy + .5) / 256., -256.).yx;
	return mix (f.x, f.y, f.z);
}

// Function 562
float noiseStep(float i, float a, float b, vec2 x, float h) {
    float d = 0.2*(b-a);
	return 1.0-i+(smoothstep(a-d, b+d, noise(vec3(x,h))*(i)));
}

// Function 563
vec2 noise(vec2 c)
{
    return mod(sin(c * mat2(3., 100., 4., 102.)) * 1e6, 1.);
}

// Function 564
float noise( in vec2 p ) {                     // noise in [-1,1]
    // p+= iTime;
    vec2 i = floor(p), f = fract(p);
	vec2 u = f*f*(3.-2.*f);
    return mix( mix( dot( hash( i + vec2(0.,0.) ), f - vec2(0.,0.) ), 
                     dot( hash( i + vec2(1.,0.) ), f - vec2(1.,0.) ), u.x),
                mix( dot( hash( i + vec2(0.,1.) ), f - vec2(0.,1.) ), 
                     dot( hash( i + vec2(1.,1.) ), f - vec2(1.,1.) ), u.x), u.y);
}

// Function 565
float noise(vec3 x) 
{
  vec3 p = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);

  float n = p.x + p.y * 157.0 + 113.0 * p.z;
  return mix(
    mix(mix(hash(n + 0.0), hash(n + 1.0), f.x), 
    mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y), 
    mix(mix(hash(n + 113.0), hash(n + 114.0), f.x), 
    mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
}

// Function 566
float noise2D(vec2 p, float seed){
    p /= 20.f;
    return fract(5.f * sin(dot(p, p) * seed) - p.y * cos(435.324 * seed * p.x));;
}

// Function 567
float noise(vec3 p){
    vec3 b=floor(p),f=fract(p);return mix(
        mix(mix(rand(b+O.yyy),rand(b+O.xyy),f.x),mix(rand(b+O.yxy),rand(b+O.xxy),f.x),f.y),
        mix(mix(rand(b+O.yyx),rand(b+O.xyx),f.x),mix(rand(b+O.yxx),rand(b+O.xxx),f.x),f.y),f.z);
}

// Function 568
float noise(vec3 v)
{
	const vec2  C = vec2(1.0/6.0, 1.0/3.0);
	const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
	// First corner
	vec3 i  = floor(v + dot(v, C.yyy));
	vec3 x0 = v - i + dot(i, C.xxx);
	// Other corners
	vec3 g = step(x0.yzx, x0.xyz);
	vec3 l = 1.0 - g;
	vec3 i1 = min(g.xyz, l.zxy);
	vec3 i2 = max(g.xyz, l.zxy);
	vec3 x1 = x0 - i1 + C.xxx;
	vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
	vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y
	// Permutations
	i = mod289(i);
	vec4 p = permute( permute( permute( i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
	// Gradients: 7x7 points over a square, mapped onto an octahedron.
	// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
	float n_ = 0.142857142857; // 1.0/7.0
	vec3  ns = n_ * D.wyz - D.xzx;
	vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)
	vec4 x_ = floor(j * ns.z);
	vec4 y_ = floor(j - 7.0 * x_);    // mod(j,N)
	vec4 x = x_ *ns.x + ns.yyyy;
	vec4 y = y_ *ns.x + ns.yyyy;
	vec4 h = 1.0 - abs(x) - abs(y);
	vec4 b0 = vec4(x.xy, y.xy);
	vec4 b1 = vec4(x.zw, y.zw);
	vec4 s0 = floor(b0) * 2.0 + 1.0;
	vec4 s1 = floor(b1) * 2.0 + 1.0;
	vec4 sh = -step(h, vec4(0.0));
	vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
	vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
	vec3 p0 = vec3(a0.xy, h.x);
	vec3 p1 = vec3(a0.zw, h.y);
	vec3 p2 = vec3(a1.xy, h.z);
	vec3 p3 = vec3(a1.zw, h.w);
	//Normalise gradients
	vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;
	// Mix final noise value
	vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
	m = m * m;
	return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

// Function 569
void NoiseHead(vec2 u,inout vec3 x,inout vec3 y
       ,inout vec2 d0,inout vec2 d1,inout vec2 d2
       ,inout vec2 p0,inout vec2 p1,inout vec2 p2
){u.y+=.001
 ;vec2 uv=vec2(u.x + u.y*0.5, u.y)
 ;vec2 j0=floor(uv)
 ;vec2 f0=fract(uv)
 ;vec2 j1=(f0.x>f0.y)?vec2(1,0):vec2(0,1)
 ;p0=vec2(j0.x-j0.y*.5,j0.y)
 ;p1=vec2(p0.x+j1.x- j1.y*.5,p0.y+j1.y)
 ;p2=vec2(p0.x+.5,p0.y +1.)
 ;j1=j0+j1
 ;vec2 i2 = j0 + vec2(1)
 ;d0=u-p0;d1=u-p1;d2=u-p2
 ;x=vec3(p0.x, p1.x, p2.x)
 ;y=vec3(p0.y, p1.y, p2.y);}

// Function 570
float atm_cloudnoise2( vec4 r, bool lowfreq )
{
    float lod = log2( r.w );
    float y = textureLod( iChannel3, r.xyz / 64., lod - 1. ).x;
 	if( !lowfreq )
    	y = 2. / 3. * y + textureLod( iChannel3, r.xyz / 32., lod ).x / 3.;
    return y;
}

// Function 571
float InterpolationNoise(float x, float y)
{
    int ix = int(x);
    int iy = int(y);
    float fracx = x-float(int(x));
    float fracy = y-float(int(y));
    
    float v1 = smoothNoise(ix,iy);
    float v2 = smoothNoise(ix+1,iy);
    float v3 = smoothNoise(ix,iy+1);
    float v4 = smoothNoise(ix+1,iy+1);
    
   	float i1 = COSInterpolation(v1,v2,fracx);
    float i2 = COSInterpolation(v3,v4,fracx);
    
    return COSInterpolation(i1,i2,fracy);
    
}

// Function 572
float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
           -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
    dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// Function 573
float FractalNoise(in vec2 xy)
{
	float w = .7;
	float f = 0.0;

	for (int i = 0; i < 3; i++)
	{
		f += Noise(xy) * w;
		w = w*0.6;
		xy = 2.0 * xy;
	}
	return f;
}

// Function 574
VoronoiData voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mr;
    vec2 mi;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mi = n + g;
        }
    }
    
    // Set center of search based on which half of the cell we are in,
    // since 4x4 is not centered around "n".
    vec2 mg = step(.5,f) - 1.;

    //----------------------------------
    // second pass: distance to borders,
    // visits two neighbours to the right/down
    //----------------------------------
    md = 8.0;
    for( int j=-1; j<=2; j++ )
    for( int i=-1; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) // skip the same cell
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return VoronoiData( md, mr, mi );
}

// Function 575
vec4 ValueSimplex3D(vec3 p)
{
	vec4 a = AchNoise3D(p);
	vec4 b = AchNoise3D(p + 120.5);
	return (a + b) * 0.5;
}

// Function 576
float noise( vec3 p )
{
    //  simplex math constants
    const float SKEWFACTOR = 1.0/3.0;
    const float UNSKEWFACTOR = 1.0/6.0;
    const float SIMPLEX_CORNER_POS = 0.5;
    const float SIMPLEX_TETRAHADRON_HEIGHT = 0.70710678118654752440084436210485;    // sqrt( 0.5 )

    //  establish our grid cell.
    p *= SIMPLEX_TETRAHADRON_HEIGHT;    // scale space so we can have an approx feature size of 1.0
    vec3 Pi = floor( p + dot( p, vec3( SKEWFACTOR) ) );

    //  Find the vectors to the corners of our simplex tetrahedron
    vec3 x0 = p - Pi + dot(Pi, vec3( UNSKEWFACTOR ) );
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 Pi_1 = min( g.xyz, l.zxy );
    vec3 Pi_2 = max( g.xyz, l.zxy );
    vec3 x1 = x0 - Pi_1 + UNSKEWFACTOR;
    vec3 x2 = x0 - Pi_2 + SKEWFACTOR;
    vec3 x3 = x0 - SIMPLEX_CORNER_POS;

    //  pack them into a parallel-friendly arrangement
    vec4 v1234_x = vec4( x0.x, x1.x, x2.x, x3.x );
    vec4 v1234_y = vec4( x0.y, x1.y, x2.y, x3.y );
    vec4 v1234_z = vec4( x0.z, x1.z, x2.z, x3.z );

    // clamp the domain of our grid cell
    Pi.xyz = Pi.xyz - floor(Pi.xyz * ( 1.0 / 69.0 )) * 69.0;
    vec3 Pi_inc1 = step( Pi, vec3( 69.0 - 1.5 ) ) * ( Pi + 1.0 );

    //  generate the random vectors
    vec4 Pt = vec4( Pi.xy, Pi_inc1.xy ) + vec2( 50.0, 161.0 ).xyxy;
    Pt *= Pt;
    vec4 V1xy_V2xy = mix( Pt.xyxy, Pt.zwzw, vec4( Pi_1.xy, Pi_2.xy ) );
    Pt = vec4( Pt.x, V1xy_V2xy.xz, Pt.z ) * vec4( Pt.y, V1xy_V2xy.yw, Pt.w );
    const vec3 SOMELARGEFLOATS = vec3( 635.298681, 682.357502, 668.926525 );
    const vec3 ZINC = vec3( 48.500388, 65.294118, 63.934599 );
    vec3 lowz_mods = vec3( 1.0 / ( SOMELARGEFLOATS.xyz + Pi.zzz * ZINC.xyz ) );
    vec3 highz_mods = vec3( 1.0 / ( SOMELARGEFLOATS.xyz + Pi_inc1.zzz * ZINC.xyz ) );
    Pi_1 = ( Pi_1.z < 0.5 ) ? lowz_mods : highz_mods;
    Pi_2 = ( Pi_2.z < 0.5 ) ? lowz_mods : highz_mods;
    vec4 hash_0 = fract( Pt * vec4( lowz_mods.x, Pi_1.x, Pi_2.x, highz_mods.x ) ) - 0.49999;
    vec4 hash_1 = fract( Pt * vec4( lowz_mods.y, Pi_1.y, Pi_2.y, highz_mods.y ) ) - 0.49999;
    vec4 hash_2 = fract( Pt * vec4( lowz_mods.z, Pi_1.z, Pi_2.z, highz_mods.z ) ) - 0.49999;

    //  evaluate gradients
    vec4 grad_results = inversesqrt( hash_0 * hash_0 + hash_1 * hash_1 + hash_2 * hash_2 ) * ( hash_0 * v1234_x + hash_1 * v1234_y + hash_2 * v1234_z );

    //  Normalization factor to scale the final result to a strict 1.0->-1.0 range
    //  http://briansharpe.wordpress.com/2012/01/13/simplex-noise/#comment-36
    const float FINAL_NORMALIZATION = 37.837227241611314102871574478976;

    //  evaulate the kernel weights ( use (0.5-x*x)^3 instead of (0.6-x*x)^4 to fix discontinuities )
    vec4 kernel_weights = v1234_x * v1234_x + v1234_y * v1234_y + v1234_z * v1234_z;
    kernel_weights = max(0.5 - kernel_weights, 0.0);
    kernel_weights = kernel_weights*kernel_weights*kernel_weights;

    //  sum with the kernel and return
    return dot( kernel_weights, grad_results ) * FINAL_NORMALIZATION;
}

// Function 577
float voronoi(vec2 uv, float t) {
    float minDist = 100.0;
    float cellIndex = 0.0;
    
    uv *= 3.0;

    vec2 gv = fract(uv)-0.5;
    vec2 id = floor(uv);
    vec2 cid = vec2(0.0);

    for(float y = -1.0; y <= 1.0; y++)
    {
        for(float x = -1.0; x <= 1.0; x++)
        {
            vec2 offs = vec2(x, y);

            vec2 n = h22(id+offs);
            vec2 p = offs+sin(n*t)*0.5;
            float d = length(gv-p);

            if(d<minDist)
            {
                minDist = d;
                cid = id+offs;
            }
        }
    }
    
    return minDist;
}

// Function 578
float hashNoise(vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix(hash13(p+vec3(0,0,0)), 
                       hash13(p+vec3(1,0,0)),f.x),
                   mix(hash13(p+vec3(0,1,0)), 
                       hash13(p+vec3(1,1,0)),f.x),f.y),
               mix(mix(hash13(p+vec3(0,0,1)), 
                       hash13(p+vec3(1,0,1)),f.x),
                   mix(hash13(p+vec3(0,1,1)), 
                       hash13(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 579
bool isInsideVoronoiCell(in vec2 p, in vec2 cellId) {
    float dm = length(hash2(cellId)-p);
    for(int i=0; i<8; i++) {
        vec2 g = ID_POS(i),
             r = g + hash2(cellId + g);
        if (length(r-p) < dm) return false;
    }
    return true;
}

// Function 580
float noise(vec2 n) {
	const vec2 d = vec2(0.0, 1.0);
	vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
	return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

// Function 581
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(2.9-1.9*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                        mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                        mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
    return res;
}

// Function 582
float voronoi(vec2 uv, float t, float seed, float size) {
    
    float minDist = 100.;
    
    float gridSize = size;
    
    vec2 cellUv = fract(uv * gridSize) - 0.5;
    vec2 cellCoord = floor(uv * gridSize);
    
    for (float x = -1.; x <= 1.; ++ x) {
        for (float y = -1.; y <= 1.; ++ y) {
            vec2 cellOffset = vec2(x,y);
            
            // Random 0-1 for each cell
            vec2 rand01Cell = rand01(cellOffset + cellCoord + seed);
			
            // Get position of point
            vec2 point = cellOffset + sin(rand01Cell * (t+10.)) * .5;
            
			// Get distance between pixel and point
            float dist = distFn(cellUv, point);
    		minDist = min(minDist, dist);
        }
    }
    
    return minDist;
}

// Function 583
float cnoise(vec3 p)
{
    vec3 size = 1.0 / vec3(textureSize(iChannel1, 0));
    return (
        noise(p * size * 1.0 + vec3(0.52, 0.78, 0.43)) * 0.5 + 
        noise(p * size * 2.0 + vec3(0.33, 0.30, 0.76)) * 0.25 + 
        noise(p * size * 4.0 + vec3(0.70, 0.25, 0.92)) * 0.125) * 1.14;
}

// Function 584
vec2 eval_noise(vec2 uv, float b)       
{   
	float cellsz = 2. *_kernelRadius;
	vec2  _ij = uv / cellsz,
           ij = floor(_ij),
	      fij = fract(_ij ),
	    noise = vec2(0);
    
	for (int j = -2; j <= 2; j++)
		for (int i = -2; i <= 2; i++) {
			vec2 nij = vec2(i,j);
			noise += cell( ivec2(ij + nij) , fij - nij, b );
		}

    return noise;
}

// Function 585
float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3. - 2.0 * f);

    return mix(a, b, u.x) + 
            (c - a)* u.y * (1. - u.x) + 
            (d - b) * u.x * u.y;
}

// Function 586
float noise(vec3 p)
{
    return textureLod(iChannel1, p, 0.0).x;
}

// Function 587
float noise( in vec3 x )
{

    vec3 p = floor( x );
    vec3 k = fract( x );
    
    k *= k * k * ( 3.0 - 2.0 * k );
    
    float n = p.x + p.y * 57.0 + p.z * 113.0; 
    
    float a = hash( n );
    float b = hash( n + 1.0 );
    float c = hash( n + 57.0 );
    float d = hash( n + 58.0 );
    
    float e = hash( n + 113.0 );
    float f = hash( n + 114.0 );
    float g = hash( n + 170.0 );
    float h = hash( n + 171.0 );
    
    float res = mix( mix( mix ( a, b, k.x ), mix( c, d, k.x ), k.y ),
                     mix( mix ( e, f, k.x ), mix( g, h, k.x ), k.y ),
                     k.z
    				 );
    
    return res;
    
}

// Function 588
float fnoise(in vec2 p) {
	return .5 * noise(p) + .25 * noise(p*2.03) + .125 * noise(p*3.99);
}

// Function 589
float voronoise( in vec2 p, float u, float v )
{
	float k = 1.0+63.0*pow(1.0-v,6.0);

    vec2 i = floor(p);
    vec2 f = fract(p);
    
	vec2 a = vec2(0.0,0.0);
    for( int y=-2; y<=2; y++ )
    for( int x=-2; x<=2; x++ )
    {
        vec2  g = vec2( x, y );
		vec3  o = hash3( i + g )*vec3(u,u,1.0);
		vec2  d = g - f + o.xy;
		float w = pow( 1.0-smoothstep(0.0,1.414,length(d)), k );
		a += vec2(o.z*w,w);
    }
	
    return a.x/a.y;
}

// Function 590
float newnoise(in ivec3 ix, in vec3 fx){
    // grid
    uvec3 p = uvec3(ix + ivec3(floor(fx)) );
    vec3 w = fract(fx);
    vec3 u = w*w*(3.0-2.0*w);
    return mix( mix( mix( dot( inthash( p  ), w  ), 
                      dot( inthash( p + uvec3(1,0,0) ), w - vec3(1.0,0.0,0.0) ), u.x),
                 mix( dot( inthash( p + uvec3(0,1,0) ), w - vec3(0.0,1.0,0.0) ), 
                      dot( inthash( p + uvec3(1,1,0) ), w - vec3(1.0,1.0,0.0) ), u.x), u.y),
            mix( mix( dot( inthash( p + uvec3(0,0,1) ), w - vec3(0.0,0.0,1.0) ), 
                      dot( inthash( p + uvec3(1,0,1) ), w - vec3(1.0,0.0,1.0) ), u.x),
                 mix( dot( inthash( p + uvec3(0,1,1) ), w - vec3(0.0,1.0,1.0) ), 
                      dot( inthash( p + uvec3(1,1,1) ), w - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// Function 591
float noise( in vec2 p ) 
{
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
    return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 592
float NebulaNoise(vec3 p)
{
   float final = p.y + 4.5;
    final -= SpiralNoiseC(p.xyz);   // mid-range noise
    final += SpiralNoiseC(p.zxy*0.5123+100.0)*4.0;   // large scale features
    final -= SpiralNoise3D(p);   // more large scale features, but 3d

    return final;
}

// Function 593
float Noisefv3 (vec3 p)
{
  vec4 t1, t2;
  vec3 ip, fp;
  float q;
  ip = floor (p);  fp = fract (p);
  fp = fp * fp * (3. - 2. * fp);
  q = dot (ip, cHashA3);
  t1 = Hashv4f (q);
  t2 = Hashv4f (q + cHashA3.z);
  return mix (mix (mix (t1.x, t1.y, fp.x), mix (t1.z, t1.w, fp.x), fp.y),
              mix (mix (t2.x, t2.y, fp.x), mix (t2.z, t2.w, fp.x), fp.y), fp.z);
}

// Function 594
float noise(vec3 p){//Noise function stolen from Virgil who stole it from Shane who I assume understands this shit, unlike me who is far too busy trading pokemon cards
  vec3 ip=floor(p),s=vec3(7,157,113);
  p-=ip;
  vec4 h=vec4(0,s.yz,s.y+s.z)+dot(ip,s);
  p=p*p*(3.-2.*p);
  h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
  h.xy=mix(h.xz,h.yw,p.y);
  return mix(h.x,h.y,p.z);//Ah, yes I understand this bit: it draws a shape which, if you are drunk enough, looks like a penis
}

// Function 595
float noise(vec2 p) {
#ifdef Use_Perlin
    return perlin_noise(p);
#elif defined Use_Value
    return value_noise(p);
#elif defined Use_Simplex
    return simplex_noise(p);
#endif
    
    return 0.0;
}

// Function 596
vec3 Voronoi( vec2 pos ) {
	vec2 d[8];
	d[0] = vec2(0);
	d[1] = vec2(1,0);
	d[2] = vec2(0,1);
	d[3] = vec2(1);
	
	const float maxDisplacement = .7;//.518; //tweak this to hide grid artefacts
	
	float closest = 12.0;
	vec4 result;
	for ( int i=0; i < 8; i++ )
	{
		vec4 r = Rand(ivec2(floor(pos+d[i])));
		vec2 p = d[i] + maxDisplacement*(r.xy-.5);
		p -= fract(pos);
		float lsq = dot(p,p);
		if ( lsq < closest )
		{
			closest = lsq;
			result = r;
		}
	}
	return fract(result.xyz+result.www); // random colour
}

// Function 597
float Noisefv2 (vec2 p)
{
  vec4 t;
  vec2 ip, fp;
  ip = floor (p);
  fp = fract (p);
  fp = fp * fp * (3. - 2. * fp);
  t = Hashv4f (dot (ip, cHashA3.xy));
  return mix (mix (t.x, t.y, fp.x), mix (t.z, t.w, fp.x), fp.y);
}

// Function 598
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float a = textureLod(iChannel0,(p+vec2(0.5,0.5))/256.0,0.0).x;
	float b = textureLod(iChannel0,(p+vec2(1.5,0.5))/256.0,0.0).x;
	float c = textureLod(iChannel0,(p+vec2(0.5,1.5))/256.0,0.0).x;
	float d = textureLod(iChannel0,(p+vec2(1.5,1.5))/256.0,0.0).x;
    return mix(mix( a, b,f.x), mix( c, d,f.x),f.y);
}

// Function 599
vec2 Noise2( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y * 57.0;
   vec2 res = mix(mix( Hash22(p),          Hash22(p + add.xy),f.x),
                  mix( Hash22(p + add.yx), Hash22(p + add.xx),f.x),f.y);
    return res;
}

// Function 600
float noise( in vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

	vec2  i = floor( p + (p.x+p.y)*K1 );
    vec2  a = p - i + (i.x+i.y)*K2;
    float m = step(a.y,a.x); 
    vec2  o = vec2(m,1.0-m);
    vec2  b = a - o + K2;
	vec2  c = a - 1.0 + 2.0*K2;
    vec3  h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3  n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot( n, vec3(70.0) );
}

// Function 601
vec4 noisergba(vec2 uv){
 uv -=floor(uv/289.0)*289.;
 uv +=vec2(223.35734, 550.56781);
 //uv *=uv;
 float xy = uv.x * uv.y;    
 return vec4(sin(fract(xy*.000000064)),
             sin(fract(xy*.000000543)),
             sin(fract(xy*.000000192)),
             sin(fract(xy*.000000423)));}

// Function 602
float simplex3d(vec3 p) {
    const float F3 =  0.3333333;
    const float G3 =  0.1666667;
    
    vec3 s = floor(p + dot(p, vec3(F3)));
    vec3 x = p - s + dot(s, vec3(G3));
    
    vec3 e = step(vec3(0.0), x - x.yzx);
    vec3 i1 = e*(1.0 - e.zxy);
    vec3 i2 = 1.0 - e.zxy*(1.0 - e);
    
    vec3 x1 = x - i1 + G3;
    vec3 x2 = x - i2 + 2.0*G3;
    vec3 x3 = x - 1.0 + 3.0*G3;
    
    vec4 w, d;
    
    w.x = dot(x, x);
    w.y = dot(x1, x1);
    w.z = dot(x2, x2);
    w.w = dot(x3, x3);
    
    w = max(0.6 - w, 0.0);
    
    d.x = dot(random3(s), x);
    d.y = dot(random3(s + i1), x1);
    d.z = dot(random3(s + i2), x2);
    d.w = dot(random3(s + 1.0), x3);
    
    w *= w;
    w *= w;
    d *= w;
    
    return dot(d, vec4(52.0));
}

// Function 603
vec3 calcNoiseNormal(  vec3 pos )
{    
  return normalize( vec3(fbm(pos+eps.xyy) - fbm(pos-eps.xyy), 0.5*2.0*eps.x, fbm(pos+eps.yyx) - fbm(pos-eps.yyx) ) );
}

// Function 604
float noise(vec2 p)
{
    return hash(p.x + p.y * 57.0);
}

// Function 605
float voronoi(vec3 p){

	vec3 b, r, g = floor(p);
	p = fract(p); // "p -= g;" works on some GPUs, but not all, for some annoying reason.
	
	// Maximum value: I think outliers could get as high as "3," the squared diagonal length 
	// of the unit cube, with the mid point being "0.75." Is that right? Either way, for this 
	// example, the maximum is set to one, which would cover a good part of the range, whilst 
	// dispensing with the need to clamp the final result.
	float d = 1.; 
     
    // I've unrolled one of the loops. GPU architecture is a mystery to me, but I'm aware 
    // they're not fond of nesting, branching, etc. My laptop GPU seems to hate everything, 
    // including multiple loops. If it were a person, we wouldn't hang out. 
	for(int j = -1; j <= 1; j++) {
	    for(int i = -1; i <= 1; i++) {
    		
		    b = vec3(i, j, -1);
		    r = b - p + hash33(g+b);
		    d = min(d, dot(r,r));
    		
		    b.z = 0.0;
		    r = b - p + hash33(g+b);
		    d = min(d, dot(r,r));
    		
		    b.z = 1.;
		    r = b - p + hash33(g+b);
		    d = min(d, dot(r,r));
    			
	    }
	}
	
	return d; // Range: [0, 1]
}

// Function 606
float noises( in vec3 p){
	float a = 0.0;
	for(float i=1.0;i<4.0;i++){
		a += noise(p)/i;
		p = p*2.5 + vec3(i*a*0.02,a*0.01,a*0.1);
	}
	return a;
}

// Function 607
float noise(float a){return sin(mix(rand(floor(a)),rand(floor(a+1.)),fract(a)));}

// Function 608
float fractal_noise(vec3 p)
{
    float f = 0.0;
    // add animation
    //p = p - vec3(1.0, 1.0, 0.0) * iTime * 0.1;
    p = p * 3.0;
    f += 0.50000 * noise(p); p = 2.0 * p;
	f += 0.25000 * noise(p); p = 2.0 * p;
	f += 0.12500 * noise(p); p = 2.0 * p;
	f += 0.06250 * noise(p); p = 2.0 * p;
    f += 0.03125 * noise(p);
    
    return f;
}

// Function 609
float simplex3D(vec3 p)
{
	float f3 = 1.0/3.0;
	float s = (p.x+p.y+p.z)*f3;
	int i = int(floor(p.x+s));
	int j = int(floor(p.y+s));
	int k = int(floor(p.z+s));
	
	float g3 = 1.0/6.0;
	float t = float((i+j+k))*g3;
	float x0 = float(i)-t;
	float y0 = float(j)-t;
	float z0 = float(k)-t;
	x0 = p.x-x0;
	y0 = p.y-y0;
	z0 = p.z-z0;
	int i1,j1,k1;
	int i2,j2,k2;
	if(x0>=y0)
	{
		if		(y0>=z0){ i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; } // X Y Z order
		else if	(x0>=z0){ i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; } // X Z Y order
		else 			{ i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; } // Z X Z order
	}
	else 
	{ 
		if		(y0<z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; } // Z Y X order
		else if	(x0<z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; } // Y Z X order
		else 			{ i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; } // Y X Z order
	}
	float x1 = x0 - float(i1) + g3; 
	float y1 = y0 - float(j1) + g3;
	float z1 = z0 - float(k1) + g3;
	float x2 = x0 - float(i2) + 2.0*g3; 
	float y2 = y0 - float(j2) + 2.0*g3;
	float z2 = z0 - float(k2) + 2.0*g3;
	float x3 = x0 - 1.0 + 3.0*g3; 
	float y3 = y0 - 1.0 + 3.0*g3;
	float z3 = z0 - 1.0 + 3.0*g3;			 
	vec3 ijk0 = vec3(i,j,k);
	vec3 ijk1 = vec3(i+i1,j+j1,k+k1);	
	vec3 ijk2 = vec3(i+i2,j+j2,k+k2);
	vec3 ijk3 = vec3(i+1,j+1,k+1);	     
	vec3 gr0 = normalize(vec3(noise3D(ijk0),noise3D(ijk0*2.01),noise3D(ijk0*2.02)));
	vec3 gr1 = normalize(vec3(noise3D(ijk1),noise3D(ijk1*2.01),noise3D(ijk1*2.02)));
	vec3 gr2 = normalize(vec3(noise3D(ijk2),noise3D(ijk2*2.01),noise3D(ijk2*2.02)));
	vec3 gr3 = normalize(vec3(noise3D(ijk3),noise3D(ijk3*2.01),noise3D(ijk3*2.02)));
	float n0 = 0.0;
	float n1 = 0.0;
	float n2 = 0.0;
	float n3 = 0.0;
	float t0 = 0.5 - x0*x0 - y0*y0 - z0*z0;
	if(t0>=0.0)
	{
		t0*=t0;
		n0 = t0 * t0 * dot(gr0, vec3(x0, y0, z0));
	}
	float t1 = 0.5 - x1*x1 - y1*y1 - z1*z1;
	if(t1>=0.0)
	{
		t1*=t1;
		n1 = t1 * t1 * dot(gr1, vec3(x1, y1, z1));
	}
	float t2 = 0.5 - x2*x2 - y2*y2 - z2*z2;
	if(t2>=0.0)
	{
		t2 *= t2;
		n2 = t2 * t2 * dot(gr2, vec3(x2, y2, z2));
	}
	float t3 = 0.5 - x3*x3 - y3*y3 - z3*z3;
	if(t3>=0.0)
	{
		t3 *= t3;
		n3 = t3 * t3 * dot(gr3, vec3(x3, y3, z3));
	}
	return 96.0*(n0+n1+n2+n3);
}

// Function 610
float SimplexNoise(
    vec3  pos,
    float octaves,
    float scale,
    float persistence)
{
    float final        = 0.0;
    float amplitude    = 1.0;
    float maxAmplitude = 0.0;
    
    for(float i = 0.0; i < octaves; ++i)
    {
        final        += SimplexNoiseRaw(pos * scale) * amplitude;
        scale        *= 2.0;
        maxAmplitude += amplitude;
        amplitude    *= persistence;
    }
    
    return (final / maxAmplitude);
}

// Function 611
float PerlinNoise3D(vec3 uv, int octaves) {
    float c = 0.0;
    float s = 0.0;
    for (float i = 0.0; i < float(octaves); i++) {
        c += SmoothNoise3D(uv * pow(2.0, i)) * pow(0.5, i);
        s += pow(0.5, i);
    }
    
    return c /= s;
}

// Function 612
float noise( in float x )
{
    float p = floor(x);
    float f = fract(x);

    f = f*f*(3.0-2.0*f);

    return mix( hash(p+0.0), hash(p+1.0),f);
}

// Function 613
vec4 noise(vec4 c,vec2 px)
{
    vec2 uv = px / iResolution.xy;
    
    vec4 r = texture(iChannel0,uv+vec2(sin(iTime*10.0),sin(iTime*20.0)));
    
    c += r * 0.2; 

    return c;
}

// Function 614
float noise(float x)
{  
    return texture(iChannel1, vec2(x,x)).x;
}

// Function 615
float noiseLayers(in vec3 p) {

    // Normally, you'd just add a time vector to "p," and be done with 
    // it. However, in this instance, time is added seperately so that 
    // its frequency can be changed at a different rate. "p.z" is thrown 
    // in there just to distort things a little more.
    vec3 t = vec3(0., 0., p.z + iTime*1.5);

    const int iter = 5; // Just five layers is enough.
    float tot = 0., sum = 0., amp = 1.; // Total, sum, amplitude.

    for (int i = 0; i < iter; i++) {
        tot += voronoi(p + t) * amp; // Add the layer to the total.
        p *= 2.; // Position multiplied by two.
        t *= 1.5; // Time multiplied by less than two.
        sum += amp; // Sum of amplitudes.
        amp *= .5; // Decrease successive layer amplitude, as normal.
    }
    
    return tot/sum; // Range: [0, 1].
}

// Function 616
float orbitNoise(vec2 p)
{
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float rz = 0.;
    float orbitRadius = .75;

    //16 taps
    for (int j = -1; j <= 2; j++)
    for (int i = -1; i <= 2; i++)
    {
        vec2 dp = vec2(j,i);
        vec4 rn = hash42(dp + ip) - 0.5;
        vec2 op = fp - dp + rn.zw*orbitRadius;
        rz += nuttall(length(op),1.85)*dot(rn.xy*1.7, op);
    }
    return rz*0.5+0.5;
}

// Function 617
float voronoi(in vec2 x)
{
	vec2 p = floor(x);
	vec2 f = fract(x);
	
	vec2 res = vec2(8.0);
	for(int j = -1; j <= 1; j ++)
	{
		for(int i = -1; i <= 1; i ++)
		{
			vec2 b = vec2(i, j);
			vec2 r = vec2(b) - f + rand2(p + b);
			
			// chebyshev distance, one of many ways to do this
			float d = max(abs(r.x), abs(r.y));
			
			if(d < res.x)
			{
				res.y = res.x;
				res.x = d;
			}
			else if(d < res.y)
			{
				res.y = d;
			}
		}
	}
	return res.y - res.x;
}

// Function 618
float noise( in vec2 p ) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
        vec2 i = floor(p + (p.x+p.y)*K1);       
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
        vec2 c = a - 1.0 + 2.0*K2;
    vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
        vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot(n, vec3(70.0));  
}

// Function 619
v0 noise01(vec2 p){return clamp((noise(p)+.5)*.5, 0.,1.);}

// Function 620
float noiseFloat(vec2 p)
{
    p = fract(p * vec2(1000. * 0.21353, 1000. * 0.97019));
    p = p + dot(p, p + 1000. * 0.54823);
    return fract(p.x * p.y);
}

// Function 621
float noise( in vec3 x )
{
	vec3 p = floor(x);
	vec3 f = fract(x);

	f = f*f*(3.0-2.0*f);
	float n = p.x + p.y*57.0 + 113.0*p.z;
	return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
				   mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
			   mix(mix( hash(n+113.0), hash(n+114.0),f.x),
				   mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

// Function 622
float snoise(vec4 v, out vec4 gradient)
  {
  const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
                        0.276393202250021,  // 2 * G4
                        0.414589803375032,  // 3 * G4
                       -0.447213595499958); // -1 + 4 * G4

// First corner
  vec4 i  = floor(v + dot(v, vec4(F4)) );
  vec4 x0 = v -   i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
  vec4 i0;
  vec3 isX = step( x0.yzw, x0.xxx );
  vec3 isYZ = step( x0.zww, x0.yyz );
//  i0.x = dot( isX, vec3( 1.0 ) );
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;
//  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;
  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;

  // i0 now contains the unique values 0,1,2,3 in each channel
  vec4 i3 = clamp( i0, 0.0, 1.0 );
  vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
  vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

  //  x0 = x0 - 0.0 + 0.0 * C.xxxx
  //  x1 = x0 - i1  + 1.0 * C.xxxx
  //  x2 = x0 - i2  + 2.0 * C.xxxx
  //  x3 = x0 - i3  + 3.0 * C.xxxx
  //  x4 = x0 - 1.0 + 4.0 * C.xxxx
  vec4 x1 = x0 - i1 + C.xxxx;
  vec4 x2 = x0 - i2 + C.yyyy;
  vec4 x3 = x0 - i3 + C.zzzz;
  vec4 x4 = x0 + C.wwww;

// Permutations
  i = mod289(i); 
  float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute( permute( permute( permute (
             i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));

// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

  vec4 p0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);

// Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x; 
  p1 *= norm.y; 
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= taylorInvSqrt(dot(p4,p4));

// Mix contributions from the five corners
  vec3 m0 = max(0.5 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.5 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
  vec3  m02 = m0 * m0;
  vec2 m12 = m1 * m1; 
  vec3 m04 = m02 * m02;
  vec2 m14 = m12 * m12; 
  vec3 pdotx0 = vec3(dot(p0,x0), dot(p1,x1), dot(p2,x2));
  vec2 pdotx1 = vec2(dot(p3,x3), dot(p4,x4));

  // Determine noise gradient;
  vec3 temp0 = m02 * m0 * pdotx0;
  vec2 temp1 = m12 * m1 * pdotx1;
  gradient = -8.0 * (temp0.x * x0 + temp0.y * x1 + temp0.z * x2 + temp1.x * x3 + temp1.y * x4);
  gradient += m04.x * p0 + m04  .y * p1 + m04.z * p2 + m14.x * p3 + m14.y * p4;
  gradient *= 109.319;
 
  return 109.319 * (  dot(m02*m02, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
                + dot(m12*m12, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;
 
  }

// Function 623
float noise(vec2 p, float freq ){
	float unit = 1./freq;
	vec2 ij = floor(p/unit);
	vec2 xy = mod(p,unit)/unit;
	xy = .5*(1.-cos(PI*xy));
	float a = rand((ij+vec2(0.,0.)));
	float b = rand((ij+vec2(1.,0.)));
	float c = rand((ij+vec2(0.,1.)));
	float d = rand((ij+vec2(1.,1.)));
	float x1 = mix(a, b, xy.x);
	float x2 = mix(c, d, xy.x);
	return mix(x1, x2, xy.y);
}

// Function 624
vec2 noise2_3(vec3 coord)
{
    //vec3 f = fract(coord);
    vec3 f = smoothstep(0.0, 1.0, fract(coord));
 	
    vec3 uv000 = floor(coord);
    vec3 uv001 = uv000 + vec3(0,0,1);
    vec3 uv010 = uv000 + vec3(0,1,0);
    vec3 uv011 = uv000 + vec3(0,1,1);
    vec3 uv100 = uv000 + vec3(1,0,0);
    vec3 uv101 = uv000 + vec3(1,0,1);
    vec3 uv110 = uv000 + vec3(1,1,0);
    vec3 uv111 = uv000 + vec3(1,1,1);
    
    vec2 v000 = hash2_3(uv000);
    vec2 v001 = hash2_3(uv001);
    vec2 v010 = hash2_3(uv010);
    vec2 v011 = hash2_3(uv011);
    vec2 v100 = hash2_3(uv100);
    vec2 v101 = hash2_3(uv101);
    vec2 v110 = hash2_3(uv110);
    vec2 v111 = hash2_3(uv111);
    
    vec2 v00 = mix(v000, v001, f.z);
    vec2 v01 = mix(v010, v011, f.z);
    vec2 v10 = mix(v100, v101, f.z);
    vec2 v11 = mix(v110, v111, f.z);
    
    vec2 v0 = mix(v00, v01, f.y);
    vec2 v1 = mix(v10, v11, f.y);
    vec2 v = mix(v0, v1, f.x);
    
    return v;
}

// Function 625
float noise(in vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = smoothstep(0.0, 1.0, f);
	
	vec2 uv = (p.xy + vec2(37.0, 17.0) * p.z) + f.xy;
	vec2 rg = texture(iChannel1, (uv + 0.5) / 256.0, -100.0).yx;
	return mix(rg.x, rg.y, f.z) * 2.0 - 1.0;
}

// Function 626
float simplexNoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
        + i.x + vec3(0.0, i1.x, 1.0 ));

    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Function 627
float Voronoi3Tap(vec2 p){
    
	// Simplex grid stuff.
    //
    vec2 s = floor(p + (p.x + p.y)*.3660254); // Skew the current point.
    p -= s - (s.x + s.y)*.2113249; // Use it to attain the vector to the base vertice (from p).

    // Determine which triangle we're in -- Much easier to visualize than the 3D version. :)
    // The following is equivalent to "float i = step(p.y, p.x)," but slightly faster, I hear.
    float i = p.x<p.y? 0. : 1.;
    
    
    // Vectors to the other two triangle vertices.
    vec2 p1 = p - vec2(i, 1. - i) + .2113249, p2 = p - .5773502; 

    // Add some random gradient offsets to the three vectors above.
    p += hash22(s)*.125;
    p1 += hash22(s +  vec2(i, 1. - i))*.125;
    p2 += hash22(s + 1.)*.125;
    
    // Determine the minimum Euclidean distance. You could try other distance metrics, 
    // if you wanted.
    float d = min(min(dot(p, p), dot(p1, p1)), dot(p2, p2))/.425;
   
    // That's all there is to it.
    return sqrt(d); // Take the square root, if you want, but it's not mandatory.

}

// Function 628
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f*f*(3.0-2.0*f);
    float a = textureLod(iChannel1,(p+vec2(0.5,0.5))/64.0, 0.0).x;
	float b = textureLod(iChannel1,(p+vec2(1.5,0.5))/64.0, 0.0).x;
	float c = textureLod(iChannel1,(p+vec2(0.5,1.5))/64.0, 0.0).x;
	float d = textureLod(iChannel1,(p+vec2(1.5,1.5))/64.0, 0.0).x;
    float res = mix(mix( a, b,f.x), mix( c, d,f.x),f.y);

	return 2.0*res;
}

// Function 629
float snoise(in vec3 v)
{
  const vec2 C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i = floor(v + dot(v, C.yyy) );
  vec3 x0 = v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  // x0 = x0 - 0.0 + 0.0 * C.xxx;
  // x1 = x0 - i1 + 1.0 * C.xxx;
  // x2 = x0 - i2 + 2.0 * C.xxx;
  // x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy; // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z); // mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ ); // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
}

// Function 630
float valueNoiseSimple(vec2 vl) {
   float minStep = 1.0 ;

   vec2 grid = floor(vl); // Left-bottom corner of the grid
   vec2 gridPnt1 = grid;
   vec2 gridPnt2 = vec2(grid.x, grid.y + minStep);
   vec2 gridPnt3 = vec2(grid.x + minStep, grid.y);
   vec2 gridPnt4 = vec2(gridPnt3.x, gridPnt2.y);

    // Removed perlinNoise, Value noise is much faster and has good enough result
    float s = rand2(grid); // 0,0
    float t = rand2(gridPnt3); // 1,0
    float u = rand2(gridPnt2); // 0,1
    float v = rand2(gridPnt4); // 1,1
    
    float x1 = smoothstep(0., 1., fract(vl.x));
    float interpX1 = mix(s, t, x1);
    float interpX2 = mix(u, v, x1);
    
    float y = smoothstep(0., 1., fract(vl.y));
    float interpY = mix(interpX1, interpX2, y);
    
    return interpY;
}

// Function 631
float noise3( vec3 p ) {
    vec3 noise = fract(sin(vec3(dot(p,vec3(127.1, 311.7, 191.999)),
                          dot(p,vec3(269.5, 183.3, 765.54)),
                          dot(p, vec3(420.69, 631.2,109.21))))
                 *43758.5453);
    return max(noise.x, max(noise.y, noise.z));
}

// Function 632
float getNoise(vec3 v) {
    //  make it curl
    for (int i=0; i<noiseSwirlSteps; i++) {
    	v.xy += vec2(fbm3(v), fbm3(vec3(v.xy, v.z + 1000.))) * noiseSwirlStepValue;
    }
    //  normalize
    return fbm5(v) / 2. + 0.5;
}

// Function 633
float PhiNoise(uvec2 uv)
{
    // flip every other tile to reduce anisotropy
    if(((uv.x ^ uv.y) & 4u) == 0u) uv = uv.yx;
	//if(((uv.x       ) & 4u) == 0u) uv.x = -uv.x;// more iso but also more low-freq content
    
    // constants of 2d Roberts sequence rounded to nearest primes
    const uint r0 = 3242174893u;// prime[(2^32-1) / phi_2  ]
    const uint r1 = 2447445397u;// prime[(2^32-1) / phi_2^2]
    
    // h = high-freq dither noise
    uint h = (uv.x * r0) + (uv.y * r1);
    
    // l = low-freq white noise
    uv = uv >> 2u;// 3u works equally well (I think)
    uint l = ((uv.x * r0) ^ (uv.y * r1)) * r1;
    
    // combine low and high
    return float(l + h) * (1.0 / 4294967296.0);
}

// Function 634
float noise(vec2 st) {
        vec2 i = floor(st);
        vec2 f = fract(st);

        vec2 u = f*f*(3.0-2.0*f);

        return mix( mix( dot( random2(i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                         dot( random2(i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                    mix( dot( random2(i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                         dot( random2(i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
    }

// Function 635
float fractalNoise(in vec3 vl) {
    const float persistance = 2.;
    const float persistanceA = 2.;
    float amplitude = .5;
    float rez = 0.0;
    float rez2 = 0.0;
    vec3 p = vl;
    
    for (int i = 0; i < OCTAVES / 2; i++) {
        rez += amplitude * valueNoiseSimple3D(p);
        amplitude /= persistanceA;
        p *= persistance;
    }
    
    float h = smoothstep(0., 1., vl.y*.5 + .5 );
    if (h > 0.01) { // small optimization, since Hermit polynom has low front at the start
        // God is in the details
        for (int i = OCTAVES / 2; i < OCTAVES; i++) {
            rez2 += amplitude * valueNoiseSimple3D(p);
            amplitude /= persistanceA;
            p *= persistance;
        }
        rez += mix(0., rez2, h);
    }
    
    return rez;
}

// Function 636
float snoise(in vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453)-.1;
}

// Function 637
float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
    return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ),  u.x), u.y);
}

// Function 638
float vnoise( in vec2 p ){
    vec2 i = floor( p );
    vec2 f = fract( p );
    vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( rand1D(i + vec2(0.0,0.0)), rand1D(i + vec2(1.0,0.0)), u.x), mix( rand1D(i + vec2(0.0,1.0)), rand1D(i + vec2(1.0,1.0)), u.x), u.y);
    }

// Function 639
float noise_3(in vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);	
	vec3 u = f*f*(3.0-2.0*f);
    
    
    vec2 ii = i.xy + i.z * vec2(5.0);
    float a = hash12( ii + vec2(0.0,0.0) );
	float b = hash12( ii + vec2(1.0,0.0) );    
    float c = hash12( ii + vec2(0.0,1.0) );
	float d = hash12( ii + vec2(1.0,1.0) ); 
    float v1 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
    
    ii += vec2(5.0);
    a = hash12( ii + vec2(0.0,0.0) );
	b = hash12( ii + vec2(1.0,0.0) );    
    c = hash12( ii + vec2(0.0,1.0) );
	d = hash12( ii + vec2(1.0,1.0) );
    float v2 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
        
    return max(mix(v1,v2,u.z),0.0);
}

// Function 640
float perlin3( vec3 coord, float x  ) {

        float n = 0.0;
        n = x * abs( snoise( coord ));
        return n;
    
    }

// Function 641
float noise(vec2 p)
{
    vec2 i = floor(p);
	vec2 f = fract(p); 
	f *= f * (3.0-2.0*f);

    return mix(
			mix(hash1(i + vec2(0.,0.)), hash1(i + vec2(1.,0.)),f.x),
			mix(hash1(i + vec2(0.,1.)), hash1(i + vec2(1.,1.)),f.x),
			f.y);
}

// Function 642
vec3 getVoronoi(vec2 x, float time){
    vec2 n = floor(x),
         f = fract(x),
         mr;
    float md=5.;
    for( int j=-1; j<=1; j++ ){
        for( int i=-1; i<=1; i++ ){
            vec2 g=vec2(float(i),float(j));
            vec2 o=0.5+0.5*sin(time + 6.2831*hash(n+g));
            vec2 r=g+o-f;
            float d=dot(r,r);
            if( d<md ) {md=d;mr=r;}
		}
    }
    return vec3(md,mr);
}

// Function 643
vec2 noise(vec2 tc) {
  return hash(tc);
}

// Function 644
float Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 645
float cnoise(vec2 P)
{
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod289(Pi); // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;

  vec4 i = permute(permute(ix) + iy);

  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
  vec4 gy = abs(gx) - 0.5 ;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;

  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);

  vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;  
  g01 *= norm.y;  
  g10 *= norm.z;  
  g11 *= norm.w;  

  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));

  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return n_xy;
}

// Function 646
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash2d( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( hash2d( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hash2d( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( hash2d( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 647
vec2 voronoi( in vec2 x )
{
    vec2 n = floor( x );
    vec2 f = fract( x );

	vec3 m = vec3( 8.0 );
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2  g = vec2( float(i), float(j) );
        vec2  o = hash( n + g );
      //vec2  r = g - f + o;
	    vec2  r = g - f + (0.5+0.5*sin(iTime+6.2831*o));
		float d = dot( r, r );
        if( d<m.x )
            m = vec3( d, o );
    }

    return vec2( sqrt(m.x), m.y+m.z );
}

// Function 648
float terrain_noise(vec2 p,float shft) {
	const int loops = 11;
	float a = 0.;
	float s = 0.;
	for (int i = 0; i < loops; i++) {
		float fi = float(i);
		a += smooth_random((p*(fi+1.))+vec2(random(fi),random(fi+16.245)),fi+shft)*(1./(fi+1.)); //Just because of this â†’ (1/(n+1)) â† I found out about Harmonic number.
		s += (1./(fi+1.));
	}
	return a/s;
}

// Function 649
float noise3DTex( in vec3 p ){
    
    vec3 i = floor(p); p -= i; p *= p*(3. - 2.*p);
	p.xy = texture(iChannel0, (p.xy + i.xy + vec2(37, 17)*i.z + .5)/256., -100.).yx;
	return mix(p.x, p.y, p.z);
}

// Function 650
float BandNoiseOpti(float freqband, float ltime)
{
    float result = 0.0;

    float freqCount = 60.0;
    for (float i=0.; i<freqCount; i++)
    {
        float stepf = freqband/5000. + 0.1*(i/freqCount) * (Noise((i)*0.0001)*2.0-1.0);
        float volume = pow(max(0.0,1.0-(i/freqCount)),2.0);//1.0-(i/120.0); // //QFilter(stepf,freqband/5000.0,20.0)
        result += volume*sin(ltime*(stepf * 5000.0) * TAU);
    }

    return result * 0.5;
}

// Function 651
f4 noised( in f3 x )
{
    f3 p = floor(x);
    f3 w = fract(x);
    f3 u = w*w*(3.-2.*w);
    f3 du = 6.*w*(1.-w);
    
    f1 n = p.x + p.y*157. + 113.*p.z;
    
    f1 a = hash(n);
    f1 b = hash(n+  1.);
    f1 c = hash(n+157.);
    f1 d = hash(n+158.);
    f1 e = hash(n+113.);
    f1 f = hash(n+114.);
    f1 g = hash(n+270.);
    f1 h = hash(n+271.);
    
    f1 k0 =   a;
    f1 k1 =   b - a;
    f1 k2 =   c - a;
    f1 k3 =   e - a;
    f1 k4 =   a - b - c + d;
    f1 k5 =   a - c - e + g;
    f1 k6 =   a - b - e + f;
    f1 k7 = - a + b + c - d + e - f - g + h;

    return f4( k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z, 
                 du * (f3(k1,k2,k3) + u.yzx*f3(k4,k5,k6) + u.zxy*f3(k6,k4,k5) + k7*u.yzx*u.zxy ));
}

// Function 652
float snoise(vec3 v)
	{ 
	const vec2	C = vec2(1.0/6.0, 1.0/3.0) ;
	const vec4	D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
	vec3 i	= floor(v + dot(v, C.yyy) );
	vec3 x0 =	 v - i + dot(i, C.xxx) ;

// Other corners
	vec3 g = step(x0.yzx, x0.xyz);
	vec3 l = 1.0 - g;
	vec3 i1 = min( g.xyz, l.zxy );
	vec3 i2 = max( g.xyz, l.zxy );

	//	 x0 = x0 - 0.0 + 0.0 * C.xxx;
	//	 x1 = x0 - i1	+ 1.0 * C.xxx;
	//	 x2 = x0 - i2	+ 2.0 * C.xxx;
	//	 x3 = x0 - 1.0 + 3.0 * C.xxx;
	vec3 x1 = x0 - i1 + C.xxx;
	vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
	vec3 x3 = x0 - D.yyy;			// -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
	i = mod289(i); 
	vec4 p = permute( permute( permute( 
						 i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
					 + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
					 + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
	float n_ = 0.142857142857; // 1.0/7.0
	vec3	ns = n_ * D.wyz - D.xzx;

	vec4 j = p - 49.0 * floor(p * ns.z * ns.z);	//	mod(p,7*7)

	vec4 x_ = floor(j * ns.z);
	vec4 y_ = floor(j - 7.0 * x_ );		// mod(j,N)

	vec4 x = x_ *ns.x + ns.yyyy;
	vec4 y = y_ *ns.x + ns.yyyy;
	vec4 h = 1.0 - abs(x) - abs(y);

	vec4 b0 = vec4( x.xy, y.xy );
	vec4 b1 = vec4( x.zw, y.zw );

	//vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
	//vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
	vec4 s0 = floor(b0)*2.0 + 1.0;
	vec4 s1 = floor(b1)*2.0 + 1.0;
	vec4 sh = -step(h, vec4(0.0));

	vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
	vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

	vec3 p0 = vec3(a0.xy,h.x);
	vec3 p1 = vec3(a0.zw,h.y);
	vec3 p2 = vec3(a1.xy,h.z);
	vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
	//vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
	vec4 norm = inversesqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;

// Mix final noise value
	vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
	m = m * m;
	return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
																dot(p2,x2), dot(p3,x3) ) );
	}

// Function 653
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel1, (uv+0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 654
float noise(vec2 uv, float baseres)
{
    float n = 0.0;
    for (int i = 0; i < 7; i++)
    {
        float v = pow(2.0, float(i));
        n += (1.5 / v) * snoise(vec3(uv + vec2(1.,1.) * (float(i) / 17.), 1), v * baseres);
    }
    
    
    return clamp((1.0 - n) * .5, 0., 1.) * 2.0;
}

// Function 655
float noiseLayer(vec2 uv){    
    float t = (iTime+iMouse.x)/5.;
    uv.y -= t/60.; // clouds pass by
    float e = 0.;
    for(float j=1.; j<9.; j++){
        // shift each layer in different directions
        float timeOffset = t*mod(j,2.989)*.02 - t*.015;
        e += 1.-texture(iChannel0, uv * (j*1.789) + j*159.45 + timeOffset).r / j ;
    }
    e /= 3.5;
    return e;
}

// Function 656
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
    // I appreciate the slight artifacting not having a 0..1 lookup
    // induces. (Using the 64x texture). It creates walls of
    // boxes instead of regions. This means I can get a decent
    // looking image without shadows.
	vec2 rg = texture( iChannel0, (uv+0.5)/256.0, -100.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 657
float voronoi(vec2 p, float s)
{
    vec2 i = floor(p*s);
    vec2 current = i + fract(p*s);
    float min_dist = 1.;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            vec2 neighbor = i + vec2(x, y);
            vec2 point = r2D(neighbor);
            point = 0.5 + 0.5*sin(iTime*.5 + 6.*point);
            float dist = polygon(neighbor+point - current, 3.);
            min_dist = min(min_dist, dist);
        }
    }
    return min_dist;
}

// Function 658
float noise(in vec2 xy)
{
    vec2 ij = floor(xy);
    vec2 uv = xy-ij;
    uv = uv*uv*(3.0-2.0*uv);
   

    float a = random(vec2(ij.x, ij.y ));
    float b = random(vec2(ij.x+1., ij.y));
    float c = random(vec2(ij.x, ij.y+1.));
    float d = random(vec2(ij.x+1., ij.y+1.));
    float k0 = a;
    float k1 = b-a;
    float k2 = c-a;
    float k3 = a-b-c+d;
    return (k0 + k1*uv.x + k2*uv.y + k3*uv.x*uv.y);
}

// Function 659
vec3 noise3(vec3 x) {
	return vec3( noise(x+vec3(123.456,.567,.37)),
				noise(x+vec3(.11,47.43,19.17)),
				noise(x) );
}

// Function 660
float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) + 
            (c - a)* u.y * (1.0 - u.x) + 
            (d - b) * u.x * u.y;
}

// Function 661
float Noiseff (float p)
{
  float i = floor (p);
  float f = fract (p);
  f = f * f * (3. - 2. * f);
  vec2 t = Hashv2f (i);
  return mix (t.x, t.y, f);
}

// Function 662
float noise(vec3 p){
    vec3 ip=floor(p);p-=ip;
    vec3 s=vec3(7,157,113);
    vec4 h=vec4(0,s.yz,s.y+s.z)+dot(ip,s);
    p=p*p*(3.-2.*p);
    h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
    h.xy=mix(h.xz,h.yw,p.y);
    return mix(h.x,h.y,p.z);
}

// Function 663
float tullynoise(in vec2 uv)
{
	vec2 pixel = uv * iChannelResolution[1].xy;
    return( texture(iChannel1, rndC(pixel) / iChannelResolution[1].xy).r );
}

// Function 664
vec3 voronoi( vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
		for( int i=-1; i<=1; i++ )
		{
			vec2 g = vec2(float(i),float(j));
			vec2 o = hash( n + g );
			o = 0.5 + 0.5*sin( iTime + 6.2831*o );
			vec2 r = g + o - f;
			
			//Euclidian distance
			float d = dot(r,r);

			if( d<md )
			{
				md = d;
				mr = r;
				mg = g;
			}
		}

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
		for( int i=-2; i<=2; i++ )
		{
			vec2 g = mg + vec2(float(i),float(j));
			vec2 o = hash( n + g );
			o = 0.5 + 0.5*sin( iTime + 6.2831*o );
			vec2 r = g + o - f;

			if( length(mr-r) >= 0.0001 )
			{
				// distance to line		
				float d = dot( 0.5*(mr+r), normalize(r-mr) );

				md = min( md, d );
			}
		}

    return vec3( md, mr );
}

// Function 665
float n_noise(in vec2 p)
{
    return 0.5 + 0.5 * noise(p);
}

// Function 666
float simplex2d(vec2 p) {
	return texture(iChannel0, p*0.015).x;
}

// Function 667
float gold_noise(in vec2 coordinate, in float seed){
    return fract(tan(distance(coordinate*(seed+PHI), vec2(PHI, PI)))*SQ2);
}

// Function 668
float noiseStack(vec3 pos,int octaves,float falloff){
	float noise = snoise(vec3(pos));
	float off = 1.0;
	if (octaves>1) {
		pos *= 2.0;
		off *= falloff;
		noise = (1.0-off)*noise + off*snoise(vec3(pos));
	}
	if (octaves>2) {
		pos *= 2.0;
		off *= falloff;
		noise = (1.0-off)*noise + off*snoise(vec3(pos));
	}
	if (octaves>3) {
		pos *= 2.0;
		off *= falloff;
		noise = (1.0-off)*noise + off*snoise(vec3(pos));
	}
	return (1.0+noise)/2.0;
}

// Function 669
float noise3D(vec3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
	const vec3 s = vec3(1, 57, 113);
	
	vec3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride vector for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    
	p -= ip; // Cell's fractional component.
	
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: fract(sin(x)*largeNumber).
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
	
    // Interpolating along Y.
    h.xy = mix(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return mix(h.x, h.y, p.z); // Range: [0, 1].
	
}

// Function 670
float noise2(vec3 pos)
{
    #ifdef bumped_metal
    vec3 q = pos;
    float f = 0.7000*noise(q); q = m*q*8.;
    f += 0.070*noise(q); q = m*q*2.75;
    f += 0.008*noise(q); //q = m*q*2.35;
    //f += 0.005*noise(q);
    return f;
    #else
    return 0.3;
    #endif
}

// Function 671
float noiseLayer(vec2 p, float ti){
    float e =0.;
    for(float j=1.; j<9.; j++){
        e += texture(iChannel0, p * float(j) + vec2(ti*7.89541) + vec2(j*159.78) ).r / (j/2.);
    }
    e /= 8.5;
    return e;
}

// Function 672
float orbitNoise3D(vec3 p)
{
    vec3 ip = floor(p);
    vec3 fp = fract(p);
    float rz = 0.;
    float orbitRadius = 0.75; //Zero value for standard coherent/gradient/perlin noise

    for (int k = -1; k <= 2; k++)
    for (int j = -1; j <= 2; j++)
    for (int i = -1; i <= 2; i++)
    {
            vec3 dp = vec3(k,j,i);
            uint base = baseHash(floatBitsToUint(dp + ip));
        	vec3 rn1 = vec3(uvec3(base, base*16807U, base*48271U) & uvec3(0x7fffffffU))/float(0x7fffffff);
        	vec3 rn2 = vec3(base*1664525U, base*134775813U, base*22695477U) * (1.0/float(0xffffffffU)); //(2^32 LCGs)
        	vec3 op = fp - dp - (rn1.xyz-0.5)*orbitRadius;
        	rz += nuttall(length(op),1.85)*dot(rn2.xyz*1.0, op);
    }
    
    return rz*0.5 + 0.5;
}

// Function 673
float noise2D( in vec2 pos )
{
return noise2D(pos,0.0);
}

// Function 674
float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
    return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);}

// Function 675
float pnoise(vec2 P, vec2 rep)
{
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod(Pi, rep.xyxy); // To create noise with explicit period
  Pi = mod289(Pi);        // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;

  vec4 i = permute(permute(ix) + iy);

  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
  vec4 gy = abs(gx) - 0.5 ;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;

  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);

  vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;  
  g01 *= norm.y;  
  g10 *= norm.z;  
  g11 *= norm.w;  

  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));

  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

// Function 676
float perlin(vec2 pos, float time)
{
    return (perlin(vec3(pos, time))+1.)*.5;
}

// Function 677
float noise_perlin1(vec3 p)
{
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);
	
    vec2 zo = vec2(0.,1.);
    float f000 = dot(hash33(i),f-zo.xxx);
    float f010 = dot(hash33(i+zo.yxx),f-zo.yxx);
    float f001 = dot(hash33(i+zo.xyx),f-zo.xyx);
    float f011 = dot(hash33(i+zo.yyx),f-zo.yyx);
    
    float hx1 = mix(f000,f010,u.x);
    float hx2 = mix(f001,f011,u.x);
    float hy1 = mix(hx1,hx2,u.y);
    
    float f100 = dot(hash33(i+zo.xxy),f-zo.xxy);
    float f110 = dot(hash33(i+zo.yxy),f-zo.yxy);
    float f101 = dot(hash33(i+zo.xyy),f-zo.xyy);
    float f111 = dot(hash33(i+zo.yyy),f-zo.yyy);
    
    hx1 = mix(f100,f110,u.x);
    hx2 = mix(f101,f111,u.x);
    float hy2 = mix(hx1,hx2,u.y);
    
    float h = mix(hy1,hy2,u.z);
    	
    return h;
}

// Function 678
float interpolate_noise(vec2 pos)
{
	float	a, b, c, d;
	
	a = smooth_noise(floor(pos));	
	b = smooth_noise(vec2(floor(pos.x+1.0), floor(pos.y)));
	c = smooth_noise(vec2(floor(pos.x), floor(pos.y+1.0)));
	d = smooth_noise(vec2(floor(pos.x+1.0), floor(pos.y+1.0)));
		
	a = mix(a, b, fract(pos.x));
	b = mix(c, d, fract(pos.x));
	a = mix(a, b, fract(pos.y));
	
	return a;				   	
}

// Function 679
float noise (in vec2 p)
{
    const float K1 = .366025404;
    const float K2 = .211324865;

	vec2 i = floor (p + (p.x + p.y)*K1);
	
    vec2 a = p - i + (i.x + i.y)*K2;
    vec2 o = step (a.yx, a.xy);    
    vec2 b = a - o + K2;
	vec2 c = a - 1. + 2.*K2;

    vec3 h = max (.5 - vec3 (dot (a, a), dot (b, b), dot (c, c) ), .0);

	vec3 n = h*h*h*h*vec3 (dot (a, hash (i + .0)),
						   dot (b, hash (i + o)),
						   dot (c, hash (i + 1.)));

    return dot (n, vec3 (70.));
}

// Function 680
float noise(float t){float f=fract(t);t=floor(t);return mix(rand(t),rand(t+1.0),f);}

// Function 681
vec4 Denoise(vec2 lUV, vec2 aUV, sampler2D light, sampler2D attr,
             float radius, float CD, vec3 CN) {
    vec4 L0,L1,L2,L3,L4,L5,L6,L7,L8;
    //Light fetching
    L0=texture(light,lUV*IRES); L1=texture(light,(lUV+vec2(radius,0.))*IRES);
    L2=texture(light,(lUV+vec2(-radius,0.))*IRES); L3=texture(light,(lUV+vec2(0.,radius))*IRES);
    L4=texture(light,(lUV+vec2(0.,-radius))*IRES);
    L5=texture(light,(lUV+vec2(radius))*IRES); L6=texture(light,(lUV+vec2(-radius,radius))*IRES);
    L7=texture(light,(lUV+vec2(radius,-radius))*IRES); L8=texture(light,(lUV+vec2(-radius))*IRES);
    //Variance
    vec2 Moments=(Read(L0.w).yz*0.25
        		+(Read(L1.w).yz+Read(L2.w).yz+Read(L3.w).yz+Read(L4.w).yz)*0.125
        		+(Read(L5.w).yz+Read(L6.w).yz+Read(L7.w).yz+Read(L8.w).yz)*I16)*16.;
    float Variance=abs(Moments.y-Moments.x*Moments.x);
    Moments=clamp(Moments,vec2(0.),vec2(0.99));
    //SVGF filter
    //float Lc=1./(Coeff_L*sqrt(Variance)+0.0001);
    	float Lc=1./(pow(0.5,radius-1.)*Coeff_L*sqrt(Variance)+0.0001);
    vec4 Accum=vec4(L0.xyz*0.25,0.25);
    Accum+=(vec4(L1.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(radius,0.))*IRES),CD,CN,L1.xyz-L0.xyz,Lc)+
            vec4(L2.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(-radius,0.))*IRES),CD,CN,L2.xyz-L0.xyz,Lc)+
            vec4(L3.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(0.,radius))*IRES),CD,CN,L3.xyz-L0.xyz,Lc)+
            vec4(L4.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(0.,-radius))*IRES),CD,CN,L4.xyz-L0.xyz,Lc)
            )*0.125;
    Accum+=(vec4(L5.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(radius))*IRES),CD,CN,L5.xyz-L0.xyz,Lc)+
            vec4(L6.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(-radius,radius))*IRES),CD,CN,L6.xyz-L0.xyz,Lc)+
            vec4(L7.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(radius,-radius))*IRES),CD,CN,L7.xyz-L0.xyz,Lc)+
            vec4(L8.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(-radius))*IRES),CD,CN,L8.xyz-L0.xyz,Lc)
            )*I16;
    //Output
    return vec4(Accum.xyz/Accum.w,Write(vec4(Read(L0.w).x,Moments,0.)));
}

// Function 682
vec2 noise2(in vec2 p) {
	vec2 F = floor(p), f = fract(p);
	f = f * f * (3. - 2. * f);
	return mix(
		mix(hash22(F), 			  hash22(F+vec2(1.,0.)), f.x),
		mix(hash22(F+vec2(0.,1.)), hash22(F+vec2(1.)),	f.x), f.y);
}

// Function 683
float perlin_a (vec3 n)
{
    vec3 x = floor(n * 64.0) * 0.015625;
    vec3 k = vec3(0.015625, 0.0, 0.0);
    float a = noise(x);
    float b = noise(x + k.xyy);
    float c = noise(x + k.yxy);
    float d = noise(x + k.xxy);
    vec3 p = (n - x) * 64.0;
    float u = mix(a, b, p.x);
    float v = mix(c, d, p.x);
    return mix(u,v,p.y);
}

// Function 684
float noise(vec2 p){
    p *= 4.;
	float a = 1., r = 0., s=0.;
    
    for (int i=0; i<9; i++) {
      r += a*perlin_noise(p); s+= a; p *= 2.00; a*=.57;
    }
    
    return r/s;///(.1*3.);
}

// Function 685
float triNoise2d(in vec2 p, float spd)
{
    float z=1.8;
    float z2=2.5;
	float rz = 0.;
    p *= mm2(p.x*0.06);
    vec2 bp = p;
	for (float i=0.; i<5.; i++ )
	{
        vec2 dg = tri2(bp*1.85)*.75;
        dg *= mm2(time*spd);
        p -= dg/z2;

        bp *= 1.3;
        z2 *= .45;
        z *= .42;
		p *= 1.21 + (rz-1.0)*.02;
        
        rz += tri(p.x+tri(p.y))*z;
        p*= -m2;
	}
    return clamp(1./pow(rz*29., 1.3),0.,.55);
}

// Function 686
float noise(vec2 uv) {    
    vec2 iuv = floor(uv);
    vec2 fuv = fract(uv);
    
    vec4 i = vec4(iuv, iuv + 1.0);
    vec4 f = vec4(fuv, fuv - 1.0);        
    
    i = (i + 0.5) / iChannelResolution[0].xyxy;
        
    vec2 grad_a = 2.0 * texture(iChannel0, i.xy, -100.0).rg - 1.0;
    vec2 grad_b = 2.0 * texture(iChannel0, i.zy, -100.0).rg - 1.0;
    vec2 grad_c = 2.0 * texture(iChannel0, i.xw, -100.0).rg - 1.0;
    vec2 grad_d = 2.0 * texture(iChannel0, i.zw, -100.0).rg - 1.0;
    
    float a = dot(f.xy, grad_a);
    float b = dot(f.zy, grad_b);
    float c = dot(f.xw, grad_c);
    float d = dot(f.zw, grad_d);
        
    fuv = fuv*fuv*fuv*(fuv*(fuv*6.0 - 15.0) + 10.0);    
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}

// Function 687
vec2 noise(vec2 p){
	vec2 co = floor(p);
	vec2 mu = fract(p);
	mu = 3.*mu*mu-2.*mu*mu*mu;
	vec2 a = rand((co+vec2(0.,0.)));
	vec2 b = rand((co+vec2(1.,0.)));
	vec2 c = rand((co+vec2(0.,1.)));
	vec2 d = rand((co+vec2(1.,1.)));
	return mix(mix(a, b, mu.x), mix(c, d, mu.x), mu.y);
}

// Function 688
float valueNoise(vec2 p) {
    
	vec2 i = floor(p);
    vec2 f = fract(p);
    
    f = f*f*f*(f*(f*6.0-15.0)+10.0);
    
    vec2 add = vec2(1.0,0.0);
    float res = mix(
        mix(hash12(i), hash12(i + add.xy), f.x),
        mix(hash12(i + add.yx), hash12(i + add.xx), f.x),
        f.y);
    return res;
        
}

// Function 689
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0 + 113.0*p.z;

    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                        mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                        mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
    return 1.0 - sqrt(res);
}

// Function 690
vec3 voronoiSphereMapping(vec3 n){
	vec2 uv=vec2(atan(n.x,n.z),acos(n.y));
    return getVoronoi(1.5*uv);}

// Function 691
float simplex_noise(vec2 p)
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
    
    vec2 i = floor(p + (p.x + p.y) * K1);
    
    vec2 a = p - (i - (i.x + i.y) * K2);
    vec2 o = (a.x < a.y) ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    vec2 b = a - (o - K2);
    vec2 c = a - (1.0 - 2.0 * K2);
    
    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    vec3 n = h * h * h * h * vec3(dot(a, hash22(i)), dot(b, hash22(i + o)), dot(c, hash22(i + 1.0)));
    
    return dot(vec3(70.0, 70.0, 70.0), n);
}

// Function 692
float noise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.-2.*f);
	
    float n = p.x + p.y*157. + 113.*p.z;
    
    vec4 v1 = fract(753.5453123*sin(n + vec4(0., 1., 157., 158.)));
    vec4 v2 = fract(753.5453123*sin(n + vec4(113., 114., 270., 271.)));
    vec4 v3 = mix(v1, v2, f.z);
    vec2 v4 = mix(v3.xy, v3.zw, f.y);
    return mix(v4.x, v4.y, f.x);
}

// Function 693
float snoise(in vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453)-.25;
}

// Function 694
vec4 snoise3Dv4(vec3 texc)
{
    vec3 x=texc*256.0;
    vec3 p = floor(x);
    vec3 f = fract(x);
    //f = f*f*(3.0-2.0*f);
    vec2 uv;
    uv = (p.xy+vec2(17,7)*p.z) + 0.5 + f.xy;
    vec4 v1 = texture( randSampler, uv/256.0, -1000.0);
    vec4 v2 = texture( randSampler, (uv+vec2(17,7))/256.0, -1000.0);
    return mix( v1, v2, f.z )-vec4(0.50);
}

// Function 695
float noise31(in vec3 x){
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix( hash31(p+vec3(0,0,0)), 
                        hash31(p+vec3(1,0,0)),f.x),
                   mix( hash31(p+vec3(0,1,0)), 
                        hash31(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash31(p+vec3(0,0,1)), 
                        hash31(p+vec3(1,0,1)),f.x),
                   mix( hash31(p+vec3(0,1,1)), 
                        hash31(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 696
vec3 curlNoise(vec3 p)
{
  const float e = .1;
  vec3 dx = vec3( e   , 0.0 , 0.0 );
  vec3 dy = vec3( 0.0 , e   , 0.0 );
  vec3 dz = vec3( 0.0 , 0.0 , e   );

  vec3 p_x0 = snoiseVec3( p - dx );
  vec3 p_x1 = snoiseVec3( p + dx );
  vec3 p_y0 = snoiseVec3( p - dy );
  vec3 p_y1 = snoiseVec3( p + dy );
  vec3 p_z0 = snoiseVec3( p - dz );
  vec3 p_z1 = snoiseVec3( p + dz );

  float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
  float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
  float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

  const float divisor = 1.0 / ( 2.0 * e );
  return normalize( vec3( x , y , z ) * divisor );
}

// Function 697
float simplexRot(vec2 u, float rot
){vec3 x,y;vec2 d0,d1,d2,p0,p1,p2;NoiseHead(u,x,y,d0,d1,d2,p0,p1,p2)
 ;x=x+.5*y;x=mod289(x);y=mod289(y)       //without TilingPeriod
 ;return NoiseNoDer(x,y,rot,d0,d1,d2);}

// Function 698
vec4 noised( in vec3 x )
{
    vec3 p = floor(x);
    vec3 w = fract(x);
    
#if 0
    // quintic interpolation
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
#else
    // cubic interpolation
    vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
#endif    
    
    
    float a = noise(p+vec3(0.0,0.0,0.0));
    float b = noise(p+vec3(1.0,0.0,0.0));
    float c = noise(p+vec3(0.0,1.0,0.0));
    float d = noise(p+vec3(1.0,1.0,0.0));
    float e = noise(p+vec3(0.0,0.0,1.0));
	float f = noise(p+vec3(1.0,0.0,1.0));
    float g = noise(p+vec3(0.0,1.0,1.0));
    float h = noise(p+vec3(1.0,1.0,1.0));
	
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z, 
                 du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                            k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                            k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) );
}

// Function 699
float snoise(vec3 v)
  { 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
  }

// Function 700
float perlinNoise3D(vec3 p) {
    float surfletSum = 0.f;
    // Iterate over the four integer corners surrounding uv
    for(int dx = 0; dx <= 1; ++dx) {
        for(int dy = 0; dy <= 1; ++dy) {
            for(int dz = 0; dz <= 1; ++dz) {
                surfletSum += surflet(p, floor(p) + vec3(dx, dy, dz));
            }
        }
    }
    return surfletSum;
}

// Function 701
float snoise(vec2 co)
{ // BUGFIX: 12.98... -> 129.8...
    return fract(sin(dot(co.xy ,vec2(129.898,78.233))) * 43758.5453);
}

// Function 702
float noise(in vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+.5)/256., 0.).yx;
	return mix(rg.x, rg.y, f.z);
}

// Function 703
float Noisefv2 (vec2 p)
{
  vec2 t, ip, fp;
  ip = floor (p);  
  fp = fract (p);
  fp = fp * fp * (3. - 2. * fp);
  t = mix (Hashv2v2 (ip), Hashv2v2 (ip + vec2 (0., 1.)), fp.y);
  return mix (t.x, t.y, fp.x);
}

// Function 704
float noise(in vec2 p) {
    vec2 pi = floor(p);
    vec2 pf = fract(p);

    float r00 = rand(vec2(pi.x    ,pi.y    ));
    float r10 = rand(vec2(pi.x+1.0,pi.y    ));
    float r01 = rand(vec2(pi.x    ,pi.y+1.0));
    float r11 = rand(vec2(pi.x+1.0,pi.y+1.0));

    return mix(mix(r00, r10, pf.x), mix(r01, r11, pf.x), pf.y);
}

// Function 705
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

// Function 706
float noise(vec2 st,float travel) {
    st += vec2(0.,0.);
    vec3 i = floor(vec3(st,travel));
    vec3 f = fract(vec3(st,travel));
    

    vec3 u = smoothstep(0.,1.,f);

    float base00 = dot( random3(i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) );
    float base10 = dot( random3(i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) );
    float base01 = dot( random3(i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) );
    float base11 = dot( random3(i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) );
    float top00 = dot( random3(i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) );
    float top10 = dot( random3(i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) );
    float top01 = dot( random3(i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) );
    float top11 = dot( random3(i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) );
    float base = mix(mix(base00,base10,u.x),mix(base01,base11,u.x),u.y);
    float top = mix(mix(top00,top10,u.x),mix(top01,top11,u.x),u.y);
    return mix(base,top,u.z);
    
}

// Function 707
float noise_sum_abs_sin(vec3 p)
{
    float f = noise_sum_abs(p);
    f = sin(f * 2.5 + p.x * 5.0 - 1.5);
    
    return f ;
}

// Function 708
float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(a, b, u.x) + 
            (c - a)* u.y * (1.0 - u.x) + 
            (d - b) * u.x * u.y;
}

// Function 709
float noise(vec2 p)
{
  return textureLod(iChannel0,p*vec2(1./256.),0.0).x;
}

// Function 710
vec2 Voronoi(in vec2 p){
    
    // One of Tomkh's snippets that includes a wrap to deal with
    // larger numbers, which is pretty cool.

#if 1
    // Slower, but handles big numbers better.
    vec2 n = floor(p);
    p -= n;
    vec2 h = step(.5, p) - 1.5;
    n += h; p -= h;
#else
    vec2 n = floor(p - 1.);
    p -= n;
#endif
    
    // Storage for all sixteen hash values. The same set of hash values are
    // reused in the second pass, and since they're reasonably expensive to
    // calculate, I figured I'd save them from resuse. However, I could be
    // violating some kind of GPU architecture rule, so I might be making 
    // things worse... If anyone knows for sure, feel free to let me know.
    //
    // I've been informed that saving to an array of vectors is worse.
    //vec2 svO[3];
    
    // Individual Voronoi cell ID. Used for coloring, materials, etc.
    cellID = vec2(0); // Redundant initialization, but I've done it anyway.

    // As IQ has commented, this is a regular Voronoi pass, so it should be
    // pretty self explanatory.
    //
    // First pass: Regular Voronoi.
	vec2 mo, o;
    
    // Minimum distance, "smooth" distance to the nearest cell edge, regular
    // distance to the nearest cell edge, and a line distance place holder.
    float md = 8., lMd = 8., lMd2 = 8., lnDist, d;
    
    for( int j=0; j<3; j++ )
    for( int i=0; i<3; i++ ){
    
        o = vec2(i, j);
        o += hash22(n + o) - p;
        // Saving the hash values for reuse in the next pass. I don't know for sure,
        // but I've been informed that it's faster to recalculate the had values in
        // the following pass.
        //svO[j*3 + i] = o; 
  
        // Regular squared cell point to nearest node point.
        d = dot(o, o); 

        if( d<md ){
            
            md = d;  // Update the minimum distance.
            // Keep note of the position of the nearest cell point - with respect
            // to "p," of course. It will be used in the second pass.
            mo = o; 
            cellID = vec2(i, j) + n; // Record the cell ID also.
        }
       
    }
    

    // Second pass: Distance to closest border edge. The closest edge will be one of the edges of
    // the cell containing the closest cell point, so you need to check all surrounding edges of 
    // that cell, hence the second pass... It'd be nice if there were a faster way.
    for( int j=0; j<3; j++ )
    for( int i=0; i<3; i++ ){
        
        // I've been informed that it's faster to recalculate the hash values, rather than 
        // access an array of saved values.
        o = vec2(i, j);
        o += hash22(n + o) - p;
        // I went through the trouble to save all sixteen expensive hash values in the first 
        // pass in the hope that it'd speed thing up, but due to the evolving nature of 
        // modern architecture that likes everything to be declared locally, I might be making 
        // things worse. Who knows? I miss the times when lookup tables were a good thing. :)
        // 
        //o = svO[j*3 + i];
        
        // Skip the same cell... I found that out the hard way. :D
        if( dot(o-mo, o-mo)>.00001 ){ 
            
            // This tiny line is the crux of the whole example, believe it or not. Basically, it's
            // a bit of simple trigonometry to determine the distance from the cell point to the
            // cell border line. See IQ's article for a visual representation.
            lnDist = dot( 0.5*(o+mo), normalize(o-mo));
            
            // Abje's addition. Border distance using a smooth minimum. Insightful, and simple.
            //
            // On a side note, IQ reminded me that the order in which the polynomial-based smooth
            // minimum is applied effects the result. However, the exponentional-based smooth
            // minimum is associative and commutative, so is more correct. In this particular case, 
            // the effects appear to be negligible, so I'm sticking with the cheaper polynomial-based 
            // smooth minimum, but it's something you should keep in mind. By the way, feel free to 
            // uncomment the exponential one and try it out to see if you notice a difference.
            //
            // // Polynomial-based smooth minimum.
            lMd = smin(lMd, lnDist, .15); 
            //
            // Exponential-based smooth minimum. By the way, this is here to provide a visual reference 
            // only, and is definitely not the most efficient way to apply it. To see the minor
            // adjustments necessary, refer to Tomkh's example here: Rounded Voronoi Edges Analysis - 
            // https://www.shadertoy.com/view/MdSfzD
            //lMd = sminExp(lMd, lnDist, 20.); 
            
            // Minimum regular straight-edged border distance. If you only used this distance,
            // the web lattice would have sharp edges.
            lMd2 = min(lMd2, lnDist);
        }

    }

    // Return the smoothed and unsmoothed distance. I think they need capping at zero... but 
    // I'm not positive.
    return max(vec2(lMd, lMd2), 0.);
}

// Function 711
float noise( in vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

	vec2 i = floor( p + (p.x+p.y)*K1 );
	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = step(a.yx,a.xy);    
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;

    vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );

	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));

    return dot( n, vec3(70.0) );
	
}

// Function 712
float cyclicAddNoise( vec2 uv, int octaves, vec2 base_cycle){
    float f=0.;
    vec2 cycle = base_cycle*FirstDiv;
    uv *= FirstDiv;
    for(int i=0;i<octaves; i++){
        f += 1./(pow(2.,float(i+1)))*noise2( uv,cycle );
        uv *= 2.;
        cycle *= 2.;
    }

    return (f+1.)/2.;
}

// Function 713
vec4 Mnoise( vec4 N ) {   // apply non-linearity 1 (per scale) after blending
#  if MODE==0
    return N;                      // base turbulence
#elif MODE==1
    return -1. + 2.* (1.-abs(N));  // flame like
#elif MODE==2
    return -1. + 2.* (abs(N));     // cloud like
#endif
}

// Function 714
float noise( in vec2 x )
{
    #ifdef TEXTURE_NOISE

    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 uv = p.xy + f.xy;
    vec2 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0).yx;
    return mix( rg.x, rg.y, 1.0);

    #else

    ivec2 p = ivec2(floor(x));
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    ivec2 uv = p.xy;
    vec2 rgA = texelFetch( iChannel0, (uv+ivec2(0,0))&255, 0 ).yx;
    vec2 rgB = texelFetch( iChannel0, (uv+ivec2(1,0))&255, 0 ).yx;
    vec2 rgC = texelFetch( iChannel0, (uv+ivec2(0,1))&255, 0 ).yx;
    vec2 rgD = texelFetch( iChannel0, (uv+ivec2(1,1))&255, 0 ).yx;
    vec2 rg = mix( mix( rgA, rgB, f.x ),
                  mix( rgC, rgD, f.x ), f.y );
    return mix( rg.x, rg.y, 1.0 );

    #endif
}

// Function 715
float noise(in vec2 p)
{
    p *= 0.45;
    const float K1 = 0.366025404;
    const float K2 = 0.211324865;

	vec2 i = floor( p + (p.x+p.y)*K1 );
	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0);
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;

    vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );

	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));

    return dot( n, vec3(38.0) );
	
}

// Function 716
float noise(vec3 p) {
	vec3 ip = floor(p);
    p -= ip; 
    vec3 s = vec3(7, 157, 113);
    vec4 h = vec4(0, s.yz, s.y + s.z) + dot(ip, s);
    p *= p * (3.-2.*p); 
    h = mix(hash(h), hash(h + s.x), p.x);
    h.xy = mix(h.xz, h.yw, p.y);
    return mix(h.x, h.y, p.z); 
}

// Function 717
float noise(vec3 x) {
    const vec3 step = vec3(110, 241, 171);

    vec3 i = floor(x);
    vec3 f = fract(x);

 
    float n = dot(i, step);

    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

// Function 718
float noise(vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
	
    float n = p.x + p.y * 157.0 + 113.0 * p.z;
    return mix(mix(mix(hash(n +   0.0), hash(n +   1.0), f.x),
                   mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y),
               mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                   mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
}

// Function 719
float Noise(in vec3 p){
    vec3 i = floor(p);
        vec3 f = fract(p); 
        f *= f * (3.0-2.0*f);

    return mix(
                mix(mix(hash13(i + vec3(0.,0.,0.)), hash13(i + vec3(1.,0.,0.)),f.x),
                        mix(hash13(i + vec3(0.,1.,0.)), hash13(i + vec3(1.,1.,0.)),f.x),
                        f.y),
                mix(mix(hash13(i + vec3(0.,0.,1.)), hash13(i + vec3(1.,0.,1.)),f.x),
                        mix(hash13(i + vec3(0.,1.,1.)), hash13(i + vec3(1.,1.,1.)),f.x),
                        f.y),
                f.z);
}

// Function 720
float smoothnoise(in vec2 o) 
{
	vec2 p = floor(o);
	vec2 f = fract(o);
		
	float n = p.x + p.y*57.0;

	float a = hash(n+  0.0);
	float b = hash(n+  1.0);
	float c = hash(n+ 57.0);
	float d = hash(n+ 58.0);
	
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;
	
	vec2 t = 3.0 * f2 - 2.0 * f3;
	vec2 dt = 6.0 * f - 6.0 * f2;
	
	float u = t.x;
	float du = dt.x;	
	float v = t.y;
	float dv = dt.y;	

	float res = a + (b-a)*u +(c-a)*v + (a-b+d-c)*u*v;
	
	//float dx = (b-a)*du + (a-b+d-c)*du*v;
	//float dy = (c-a)*dv + (a-b+d-c)*u*dv;
	
	return res;
}

// Function 721
float pnoise3D(in vec3 p)
{
	vec3 pi = permTexUnit*floor(p)+permTexUnitHalf; // Integer part, scaled so +1 moves permTexUnit texel
	// and offset 1/2 texel to sample texel centers
	vec3 pf = fract(p);     // Fractional part for interpolation

	// Noise contributions from (x=0, y=0), z=0 and z=1
	float perm00 = rnm(pi.xy).a ;
	vec3  grad000 = rnm(vec2(perm00, pi.z)).rgb * 4.0 - 1.0;
	float n000 = dot(grad000, pf);
	vec3  grad001 = rnm(vec2(perm00, pi.z + permTexUnit)).rgb * 4.0 - 1.0;
	float n001 = dot(grad001, pf - vec3(0.0, 0.0, 1.0));

	// Noise contributions from (x=0, y=1), z=0 and z=1
	float perm01 = rnm(pi.xy + vec2(0.0, permTexUnit)).a ;
	vec3  grad010 = rnm(vec2(perm01, pi.z)).rgb * 4.0 - 1.0;
	float n010 = dot(grad010, pf - vec3(0.0, 1.0, 0.0));
	vec3  grad011 = rnm(vec2(perm01, pi.z + permTexUnit)).rgb * 4.0 - 1.0;
	float n011 = dot(grad011, pf - vec3(0.0, 1.0, 1.0));

	// Noise contributions from (x=1, y=0), z=0 and z=1
	float perm10 = rnm(pi.xy + vec2(permTexUnit, 0.0)).a ;
	vec3  grad100 = rnm(vec2(perm10, pi.z)).rgb * 4.0 - 1.0;
	float n100 = dot(grad100, pf - vec3(1.0, 0.0, 0.0));
	vec3  grad101 = rnm(vec2(perm10, pi.z + permTexUnit)).rgb * 4.0 - 1.0;
	float n101 = dot(grad101, pf - vec3(1.0, 0.0, 1.0));

	// Noise contributions from (x=1, y=1), z=0 and z=1
	float perm11 = rnm(pi.xy + vec2(permTexUnit, permTexUnit)).a ;
	vec3  grad110 = rnm(vec2(perm11, pi.z)).rgb * 4.0 - 1.0;
	float n110 = dot(grad110, pf - vec3(1.0, 1.0, 0.0));
	vec3  grad111 = rnm(vec2(perm11, pi.z + permTexUnit)).rgb * 4.0 - 1.0;
	float n111 = dot(grad111, pf - vec3(1.0, 1.0, 1.0));

	// Blend contributions along x
	vec4 n_x = mix(vec4(n000, n001, n010, n011), vec4(n100, n101, n110, n111), fade(pf.x));

	// Blend contributions along y
	vec2 n_xy = mix(n_x.xy, n_x.zw, fade(pf.y));

	// Blend contributions along z
	float n_xyz = mix(n_xy.x, n_xy.y, fade(pf.z));

	// We're done, return the final noise value.
	return n_xyz;
}

// Function 722
float TrigNoise(vec3 x)
{
    return TrigNoise(x, 2.0, 1.0);
}

// Function 723
vec3 simplexWeave(vec2 p){
    
    // Keeping a copy of the orginal position.
    vec2 oP = p;
    
    // Scaling constant.
    const float gSc = 5.;
    p *= gSc;
    
    
    // SIMPLEX GRID SETUP
    
    vec2 s = floor(p + (p.x + p.y)*.36602540378); // Skew the current point.
    
    p -= s - (s.x + s.y)*.211324865; // Use it to attain the vector to the base vertex (from p).
    
    // Determine which triangle we're in. Much easier to visualize than the 3D version.
    float i = p.x < p.y? 1. : 0.; // Apparently, faster than: i = step(p.y, p.x);
    vec2 ioffs = vec2(1. - i, i);
    
    // Vectors to the other two triangle vertices.
    vec2 p1 = p - ioffs + .2113248654, p2 = p - .577350269; 
 
    
    
    ////////////
    // SIMPLEX NOISE... or close enough.
    //
    // We already have the triangle points, so we may as well take the last few steps to
    // produce some simplex noise.
    //
    // Vector to hold the falloff value of the current pixel with respect to each vertice.
    vec3 d = max(.5 - vec3(dot(p, p), dot(p1, p1), dot(p2, p2)), 0.); // Range [0, 0.5]
    //
    // Determining the weighted contribution of each random gradient vector for each point...
    // Something to that effect, anyway. I could save three hash calculations below by using 
    // the following line, but it's a relatively cheap example, and I wanted to keep the noise 
    // seperate. By the way, if you're after a cheap simplex noise value, the calculations 
    // don't have to be particularly long. From here to the top, there's only a few lines, and 
    // the quality is good enough.
    vec3 w = vec3(dot(hash22(s), p), dot(hash22((s + ioffs)), p1), dot(hash22(s + 1.), p2));
    //
    // Combining the above to achieve a rough simplex noise value.
    float noise = clamp(0.5 + dot(w, d*d*d)*12., 0., 1.);    
    ////////////
    
    
    // THE WEAVE PATTERN
    
    // Three random values -- taken at each of the triangle vertices, and ranging between zero 
    // and one. Since neighboring triangles share vertices, the segments are guaranteed to meet
    // at edge boundaries, provided the right shape is chosen, etc.
    vec3 h = vec3(hash21(s), hash21((s + ioffs)), hash21(s + 1.));
    //vec3 h = vec3(length(hash22(s)), length(hash22(s - vec2(i, sc - vec2(i, 1. + i)))), length(hash22(s + 1.)))*.35;

    
    #ifdef NO_WEAVE
    // To draw the stacked circle version, the layers need to be have a lighting range from zero to one,
    // but have to be distinct (not equal) for ordering purposes. To ensure that, I've spaced the layers
    // out by a set amount, then with a little hack, seperated X from Y and Z, then Y from Z... I think
    // the logic is sound? Either way, I'm not noticing any random flipping, so it'll do.
    h = floor(h*15.999)/15.;
    if(h.x == h.y) h.y += .0001;
    if(h.x == h.z) h.z += .0001;
    if(h.y == h.z) h.z += .0001;
    #endif
    
     
    
    // Angles subtended from the current position to each of the three vertices... There's probably a 
    // symmetrical way to make just one "atan" call. Anyway, you can use these angular values to create 
    // patterns that follow the contours. In this case, I'm using them to create some cheap repetitious lines.
    vec3 a = vec3(atan(p.y, p.x), atan(p1.y, p1.x), atan(p2.y, p2.x));
 
    // The torus rings. 
    // Toroidal axis width. Basically, the weave pattern width.
    float tw = .25;
    // With the hexagon shape, the pattern width needs to be decreased.
    #if SHAPE == 1 
    tw = .19;
    #endif
    // For symmetry, we want the middle of the torus ring to cut dirrectly down the center
    // of one of the equilateral triangle sides, which is half the distance from one of the
    // vertices to the other. Add ".1" to it to see that it's necessary.
    float mid = dist((p2 - p))*.5;
    // The three distance field functions: Stored in cir.x, cir.y and cir.z.
    vec3 cir = vec3(dist(p), dist(p1), dist(p2));
    // Equivalent to: vec3 tor =  cir - mid - tw; tor = max(tor, -(cir - mid + tw));
    vec3 tor =  abs(cir - mid) - tw;

    
    // It's not absolutely necessary to scale the distance values by the scaling factor, but I find
    // it helps, since it allows me scale up and down without having to change edge widths, smoothing
    // factor variables, and so forth.
    tor /= gSc;
    cir /= gSc;

    
    
    

    #ifdef NO_WEAVE
    // Front to back ordering:
    //
    // Specifically ordering the torus rings based on their individual heights -- as
    // opposed to randomly ordering them -- will create randomly stacked rings, which
    // I thought was interesting enough to include... But at the end of the day, it's
    // probably not all that interesting. :D
    
    // I'm not sure how fond I am of the following hacky logic block, but it's easy 
    // enough to follow, plus it gets the job done:
    //
    // If the torus assoicated with the X vertex is lowest, render it first, then
    // check to see whether Y or Z should be rendered next. Swap the rendering order --
    // via swizzling -- accordingly. Repeat the process for the other vertices.
    //
    if(h.x<h.y && h.x<h.z){ // X vertex is lowest.
        
        // If you reorder one thing, you usually have to reorder everything else.
        // Forgetting to do this, which I often do, sets me up for a lot of debugging. :)
        if(h.z<h.y) { tor = tor.xzy; h = h.xzy; a = a.xzy; }
        else {  tor = tor.xyz; h = h.xyz; a = a.xyz; }
    }
    else if(h.y<h.z && h.y<h.x) {  // Y vertex is lowest.
        
         if(h.z<h.x) { tor = tor.yzx; h = h.yzx; a = a.yzx; }
         else { tor = tor.yxz; h = h.yxz; a = a.yxz; }
    }
    else { // Z vertex is lowest.
        
        if(h.y<h.x) { tor = tor.zyx; h = h.zyx; a = a.zyx; }
        else { tor = tor.zxy; h = h.zxy; a = a.zxy;}
    }
    
    #else
    // Random order logic to create the weave pattern: Use the unique
    // ID for this particular simplex grid cell to generate a random
    // number, then use it to randomly mix the order via swizzling
    // combinations. For instance, "c.xyz" will render layer X, Y then
    // Z, whereas the swizzled combination "c.zyx" will render them
    // in reverse order. There are six possible order combinations.
    // The order in which you render the tori surrounding the three
    // vertices will result in the spaghetti-like pattern you see.
    //
    // On a side note, including all six ordering possibilities 
    // guarantees that the pattern randomization is maximized, but
    // there's probably a simpler way to achieve the same result.
    
    // Random value -- unique to each grid cell.
    float dh = hash21((s + s + ioffs + s + 1.));
    if(dh<1./6.){ tor = tor.xzy; a = a.xzy; }
    else if(dh<2./6.){ tor = tor.yxz; a = a.yxz; }
    else if(dh<3./6.){ tor = tor.yzx; a = a.yzx; }
    else if(dh<4./6.){ tor = tor.zxy; a = a.zxy; }
    else if(dh<5./6.){ tor = tor.zyx; a = a.zyx; } 
   
    #endif

    
    // RENDERING
    // Applying the layered distance field objects.
    
    // The background. This one barely shows, so is very simple.
    vec3 bg = vec3(.075, .125, .2)*noise;
    bg *= clamp(cos((oP.x - oP.y)*6.2831*128.)*1., 0., 1.)*.15 + .925;
   
    // The scene color. Initialized to the background.
    vec3 col = bg;
    
    // Outer torus ring color. Just a bit of bronze.
    vec3 rimCol = vec3(1, .7, .5);
    // Applying some contrasty noise for a fake shadowy lighting effect. Since the 
    // noise is simplex based, the shadows tend to move in a triangular motion that
    // matches the underlying grid the pattern was constucted with.
    rimCol *= (smoothstep(0., .75, noise - .1) + .5);
    
    // Toroidal segment color. The angle is being used to create lines run perpendicular 
    // to the curves.
    vec3 torCol = vec3(.2, .4, 1);
    a = clamp(cos(a*48. + iTime*0.)*1. + .5, 0., 1.)*.25 + .75;
    
    // Using the tori's distance field to produce a bit of faux poloidal curvature.
    // The value has also been repeated to create the line pattern that follows the
    // pattern curves... Set it to "vec3(1)" to see what it does. :)
    vec3 cc = max(.05 - tor*32., 0.);
    cc *= clamp(cos(tor*6.2831*80.)*1. + .5, 0., 1.)*.25 + .75;
    #ifdef NO_WEAVE
    // If not using a weave pattern, you end up with some distinct, stacked tori,
    // which means you can use the random height values to shade them and introduce 
    // some depth information.
    cc *= (h*.9 + .1);
    #endif
        
    // Smoothing factor and line width.
    const float sf = .005, lw = .005;
   
    // Rendering the the three ordered (random or otherwise) objects:
    //
    // This is all pretty standard stuff. If you're not familiar with using a 2D
    // distance field value to mix a layer on top of another, it's worth learning.
    // On a side note, "1. - smoothstep(a, b, c)" can be written in a more concise
    // form (smoothstep(b, a, c), I think), but I've left it that particular way
    // for readability. You could also reverse the first two "mix" values, etc.
    // By readability, I mean the word "col" is always written on the left, the
    // "0." figure is always on the left, etc. If this were a more GPU intensive
    // exercise, then I'd rewrite things.
    
    // Bottom toroidal segment.
    //
    // Drop shadow with 50% transparency.
    col = mix(col, vec3(0), (1. - smoothstep(0., sf*4., tor.x - .00))*.5);
    // Outer dark edges.
    col = mix(col, vec3(0), 1. - smoothstep(0., sf, tor.x));
    // The bronze toroidal outer rim color.
    col = mix(col, rimCol*cc.x, 1. - smoothstep(0., sf, tor.x + lw));
    // The main blueish toroidal face with faux round shading and pattern.
    col = mix(col, torCol*col.x*a.x, 1. - smoothstep(0., sf, tor.x + .015));
    // Some inner dark edges. Note the "abs." This could be rendered before
    // the later above as a thick dark strip...
    col = mix(col, vec3(0), 1. - smoothstep(0., sf, abs(tor.x + .015)));
    
    // Same layring routine for the middle toroidal segment.
    col = mix(col, vec3(0), (1. - smoothstep(0., sf*4., tor.y - .00))*.5);
    col = mix(col, vec3(0), 1. - smoothstep(0., sf, tor.y));
    col = mix(col, rimCol*cc.y, 1. - smoothstep(0., sf, tor.y + lw)); 
    col = mix(col, torCol*col.x*a.y, 1. - smoothstep(0., sf, tor.y + .015));
    col = mix(col, vec3(0), 1. - smoothstep(0., sf, abs(tor.y + .015)));

    // Render the top toroidal segment last.
    col = mix(col, vec3(0), (1. - smoothstep(0., sf*4., tor.z - .00))*.5);
	col = mix(col, vec3(0), 1. - smoothstep(0., sf, tor.z));
    col = mix(col, rimCol*cc.z, 1. - smoothstep(0., sf, tor.z + lw));
    col = mix(col, torCol*col.x*a.z, 1. - smoothstep(0., sf, tor.z + .015));
    col = mix(col, vec3(0), 1. - smoothstep(0., sf, abs(tor.z + .015)));
    

    
    #ifdef SHOW_SIMPLEX_GRID
    // Displaying the 2D simplex grid. Basically, we're rendering lines between
    // each of the three triangular cell vertices to show the outline of the 
    // cell edges.
    vec3 c = vec3(distLine(p, p1), distLine(p1, p2), distLine(p2, p));
    c /= gSc;
    c.x = min(min(c.x, c.y), c.z);
    torCol = col;
    col = mix(col, vec3(0), (1. - smoothstep(0., sf*2., c.x - .005))*.65);
    col = mix(col, torCol*3., (1. - smoothstep(0., sf/2., c.x - .0015))*.75);
    #endif
    
   
    // Just the simplex noise, for anyone curious.
    //return vec3(noise);
    
    // Return the simplex weave value.
    return col;
 

}

// Function 724
fractal noise (4 octaves)
    else	
	{
        mat2 m = mat2(2); // mat2( 1.6,  1.2, -1.2,  1.6 ); // sqrt2 not floor-friendly ;-)
#define N(uv,s) noise( floor(scale*uv/s)*s/sampling ); // floor: optional (without = curvy)
		n  = 0.5000*N( uv, 1.); uv = m*uv;
		n += 0.2500*N( uv, 2.); uv = m*uv;
		n += 0.1250*N( uv, 4.); uv = m*uv;
		n += 0.0625*N( uv, 8.); uv = m*uv;
	}

// Function 725
float cnoise(vec2 P)
{
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod289(Pi); // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;

  vec4 i = permute(permute(ix) + iy);

  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0;
  vec4 gy = abs(gx) - 0.5;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;

  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);

  vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;

  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));

  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

// Function 726
float NoiseFBM(in vec3 p, float numCells, int octaves)
{
	float f = 0.0;
    
	// Change starting scale to any integer value...
    p = mod(p, vec3(numCells));
	float amp = 0.5;
    float sum = 0.0;
	
	for (int i = 0; i < octaves; i++)
	{
		f += Noise(p, numCells) * amp;
        sum += amp;
		amp *= 0.5;

		// numCells must be multiplied by an integer value...
		numCells *= 2.0;
	}

	return f / sum;
}

// Function 727
float voronoi(in vec3 p) {
  vec3 ip = floor(p);
  vec3 fp = fract(p);
  float rid = -1.;
  vec2 r = vec2(2.);
  for (int i=-1; i<=0; i++)
    for (int j=-1; j<=0; j++)
      for (int k=-1; k<=0; k++) {
        vec3 g = vec3(i, j, k);
        vec3 pp = fp +g +hash3(ip - g)*.6;
        float d = dot(pp, pp);

        if (d < r.x) {
          r.y = r.x;
          r.x = d;
        } else if(d < r.y) {
          r.y = d;
        }
      }
  return r.x;
}

// Function 728
float noise(vec3 p, int seed)
{
    return float(hash(int(p.x), int(p.y), int(p.z), seed)) / 4294967296.0;
}

// Function 729
float noise4d(vec4 x){
	vec4 p=floor(x);
	vec4 f=smoothstep(0.,1.,fract(x));
	float n=p.x+p.y*157.+p.z*113.+p.w*971.;
	return mix(mix(mix(mix(hash(n),hash(n+1.),f.x),mix(hash(n+157.),hash(n+158.),f.x),f.y),
	mix(mix(hash(n+113.),hash(n+114.),f.x),mix(hash(n+270.),hash(n+271.),f.x),f.y),f.z),
	mix(mix(mix(hash(n+971.),hash(n+972.),f.x),mix(hash(n+1128.),hash(n+1129.),f.x),f.y),
	mix(mix(hash(n+1084.),hash(n+1085.),f.x),mix(hash(n+1241.),hash(n+1242.),f.x),f.y),f.z),f.w);
}

// Function 730
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel0, (uv+0.5)/256.0, -100.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 731
float smoothNoise(in vec3 q){
	float f  = .5000*noise(q); q=msun*q*2.01;
          f += .2500*noise(q); q=msun*q*2.02;
          f += .1250*noise(q); //q=msun*q*2.03;
       //   f += .0625*noise(q);
	return f;
}

// Function 732
float fNoise(vec2 p){
    p+=.1*vec2(sin(2.1*iTime),cos(3.6*iTime));
    p+=.5*vec2(sin(iTime/5.),cos(iTime/3.2));
    mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
    float f=0.;
    float str=.5;
    for(int i=0;i<4;i++){
        f += str*noise(p);
    	str/=2.;
        p = m*p;
    }
	return 0.5 + 0.5*f;
}

// Function 733
vec3 voronoi_n(inout vec2 rd, inout vec2 n,  inout vec2 f, 
                              inout vec2 mg, inout vec2 mr) {
    float md = 1e5;
    vec2 mmg = mg;
    vec2 mmr = mr;
    vec2 ml = vec2(0.0);
    
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {   
        vec2 g = mmg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;

    	vec2 l = r-mmr;
 		if((dot(l, l) * dot(rd, l)) > 1e-5) {
            float d = dot(0.5*(mmr+r), l)/dot(rd, l);
            if (d < md) {
                md = d;
                mg = g;
                mr = r;
                ml = l;
            }
        }
    }
    
    return vec3(md, ml);
}

// Function 734
float noise( in vec2 x )
{
	#ifdef PROCEDURAL_NOISE
	x *= 1.75;
    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0;

    float res = mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                    mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);
    return res;
	#else
	return texture(iChannel0, x*.01).x;
	#endif
}

// Function 735
float perlin(float p){
	float fr= fract(p);
	float frn= fr-1.;
	float f= floor(p);
	float c= ceil(p);
	float a= nmaps(rand(f));
	float b= nmaps(rand(c));
	return lerp(a,b,smooth(fr));
}

// Function 736
float fnoisePerlin(float amp, float freq, float x, float y, float z){
	x=x*freq;
	y=y*freq;
	float fx=floor(x);
	float fy=floor(y);
	float fz=floor(z);
	float cx=ceil(x);
	float cy=ceil(y);
	float cz=ceil(z);
    
	vec3 v000=hash(vec3(fx, fy, fz));
	vec3 v100=hash(vec3(cx, fy, fz));
	vec3 v010=hash(vec3(fx, cy, fz));
	vec3 v110=hash(vec3(cx, cy, fz));
	vec3 v001=hash(vec3(fx, fy, cz));
	vec3 v101=hash(vec3(cx, fy, cz));
	vec3 v011=hash(vec3(fx, cy, cz));
	vec3 v111=hash(vec3(cx, cy, cz));

    
	float a000=dot(v000, vec3(x-fx,y-fy, z-fz));
	float a100=dot(v100, vec3(x-cx,y-fy, z-fz));
	float a010=dot(v010, vec3(x-fx,y-cy, z-fz));
	float a110=dot(v110, vec3(x-cx,y-cy, z-fz));
	
    float a001=dot(v001, vec3(x-fx,y-fy, z-cz));
	float a101=dot(v101, vec3(x-cx,y-fy, z-cz));
	float a011=dot(v011, vec3(x-fx,y-cy, z-cz));
	float a111=dot(v111, vec3(x-cx,y-cy, z-cz));
    
    
    float mx=blend(x-fx);
    float my=blend(y-fy);
    float mz=blend(z-fz);
    
    
    float ix00=mix(a000, a100, mx);
    float ix10=mix(a010, a110, mx);
    float ix01=mix(a001, a101, mx);
    float ix11=mix(a011, a111, mx);
    
    float iy0=mix(ix00,ix10, my);
    float iy1=mix(ix01,ix11, my);
    
    float iz=mix(iy0, iy1, mz);
    
    
    /*
	float sx=blend(x-fx);
	float a=s+sx*(t-s);
	float b=u+sx*(v-u);

	float sy=blend(y-fy);

	float r=a+sy*(b-a);*/
    
	return cos(amp*iz);
}

// Function 737
float noise( in vec2 x, in float scale )
{
    x *= scale;
    x+=iTime+20.;
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(1.5-f)*2.0;
    
    float res = mix(mix( hash12(p, scale), hash12(p + add.xy, scale),f.x),
                    mix( hash12(p + add.yx, scale), hash12(p + add.xx, scale),f.x),f.y);
    return res;
}

// Function 738
vec3 dnoise2f(vec2 p){
    float i = floor(p.x), j = floor(p.y);
    float u = p.x-i, v = p.y-j;
    float du = 30.*u*u*(u*(u-2.)+1.);
    float dv = 30.*v*v*(v*(v-2.)+1.);
    u=u*u*u*(u*(u*6.-15.)+10.);
    v=v*v*v*(v*(v*6.-15.)+10.);
    float a = r(i,     j    );
    float b = r(i+1.0, j    );
    float c = r(i,     j+1.0);
    float d = r(i+1.0, j+1.0);
    float k0 = a;
    float k1 = b-a;
    float k2 = c-a;
    float k3 = a-b-c+d;
    return vec3(k0 + k1*u + k2*v + k3*u*v,
                du*(k1 + k3*v),
                dv*(k2 + k3*u));
}

// Function 739
float pnoise(vec2 co, float freq, int steps, float persistence)
{
  float value = 0.0;
  float ampl = 1.0;
  float sum = 0.0;
  for(int i=0 ; i<steps ; i++)
  {
    sum += ampl;
    value += noise(co, freq) * ampl;
    freq *= 2.0;
    ampl *= persistence;
  }
  return value / sum;
}

// Function 740
float noise(in vec2 p)
{
	vec2 i = floor(p);
	vec2 f = fract(p);

	vec2 u = f * f * (3.0 - 2.0 * f);

	return mix(mix(dot(hash(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
		dot(hash(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
		mix(dot(hash(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
			dot(hash(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x),
		u.y);
}

// Function 741
vec3  simplexRotD(vec2 u,vec2 p){return simplexRotD(u,p,0.);}

// Function 742
float noise(vec2 n) {
	const vec2 d = vec2(0.0, 1.0);
  vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
	return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

// Function 743
float fnoise(float x) {
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3.0 - 2.0 * f);
    return mix(hash(i), hash(i + 1.0), u);
}

// Function 744
fixed noise4q(fixed4 x)
			{
				fixed4 n3 = fixed4(0,0.25,0.5,0.75);
				fixed4 p2 = floor(x.wwww+n3);
				fixed4 b = floor(x.xxxx +n3) + floor(x.yyyy +n3)*157.0 + floor(x.zzzz +n3)*113.0;
				fixed4 p1 = b + frac(p2*0.00390625)*fixed4(164352.0, -164352.0, 163840.0, -163840.0);
				p2 = b + frac((p2+1)*0.00390625)*fixed4(164352.0, -164352.0, 163840.0, -163840.0);
				fixed4 f1 = frac(x.xxxx+n3);
				fixed4 f2 = frac(x.yyyy+n3);
				
				fixed4 n1 = fixed4(0,1.0,157.0,158.0);
				fixed4 n2 = fixed4(113.0,114.0,270.0,271.0);		
				fixed4 vs1 = lerp(hash4(p1), hash4(n1.yyyy+p1), f1);
				fixed4 vs2 = lerp(hash4(n1.zzzz+p1), hash4(n1.wwww+p1), f1);
				fixed4 vs3 = lerp(hash4(p2), hash4(n1.yyyy+p2), f1);
				fixed4 vs4 = lerp(hash4(n1.zzzz+p2), hash4(n1.wwww+p2), f1);	
				vs1 = lerp(vs1, vs2, f2);
				vs3 = lerp(vs3, vs4, f2);
				
				vs2 = lerp(hash4(n2.xxxx+p1), hash4(n2.yyyy+p1), f1);
				vs4 = lerp(hash4(n2.zzzz+p1), hash4(n2.wwww+p1), f1);
				vs2 = lerp(vs2, vs4, f2);
				vs4 = lerp(hash4(n2.xxxx+p2), hash4(n2.yyyy+p2), f1);
				fixed4 vs5 = lerp(hash4(n2.zzzz+p2), hash4(n2.wwww+p2), f1);
				vs4 = lerp(vs4, vs5, f2);
				f1 = frac(x.zzzz+n3);
				f2 = frac(x.wwww+n3);
				
				vs1 = lerp(vs1, vs2, f1);
				vs3 = lerp(vs3, vs4, f1);
				vs1 = lerp(vs1, vs3, f2);
				
				return dot(vs1,0.25);
			}

// Function 745
vec4 noise(vec2 v) {
	vec4 c = vec4(0.0,0.0,0.0,0.0);
	float s = 0.0;
	for(float i=1.0;i<16.0;i++) {
		float q = pow(2.0,i);
		c+=texture(iChannel0,v*pow(0.5,i))*q;
		s+=q;
	}
	return c/s;
}

// Function 746
float perlinNoise(vec2 x) {
    vec2 id = floor(x);
    vec2 f = fract(x);

	float a = hash1(id);
    float b = hash1(id + vec2(1.0, 0.0));
    float c = hash1(id + vec2(0.0, 1.0));
    float d = hash1(id + vec2(1.0, 1.0));
	// Same code, with the clamps in smoothstep and common subexpressions
	// optimized away.
    vec2 u = hermiteInter(f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Function 747
vec4 _DenoiseGI(vec2 uv, vec2 aUV, vec2 lUV, vec2 UVoff, vec3 CVP, vec3 CN, vec3 CVN, sampler2D ta, sampler2D tl,
            vec2 ires, vec2 hres, vec2 asfov, vec3 L0, float LCoeff) {
    //Denoiser help function
    vec4 a0=texture(ta,(aUV+UVoff)*ires);
    vec4 a1=texture(ta,(aUV-UVoff)*ires);
    vec4 L=vec4(0.); vec3 SP,L1;
    if (a0.w<9990. && Box2(uv+UVoff,hres)<-0.5) {
        vec4 l0=texture(tl,(lUV+UVoff)*ires);
        SP=normalize(vec3(((uv+UVoff)*ires*4.-1.)*asfov,1.))*a0.w;
        L1=l0.xyz;
        //L+=vec4(L1,1.)*WeightVar(dot(SP-CVP,CVN),dot(CN,Read3(a0.y)*2.-1.),L1-L0,LCoeff);
        L+=vec4(L1,1.)*Weight(dot(SP-CVP,CVN),dot(CN,Read3(a0.y)*2.-1.));
    }
    if (a1.w<9990. && Box2(uv-UVoff,hres)<-0.5) {
        vec4 l1=texture(tl,(lUV-UVoff)*ires);
        SP=normalize(vec3(((uv-UVoff)*ires*4.-1.)*asfov,1.))*a1.w;
        L1=l1.xyz;
        //L+=vec4(L1,1.)*WeightVar(dot(SP-CVP,CVN),dot(CN,Read3(a1.y)*2.-1.),L1-L0,LCoeff);
        L+=vec4(L1,1.)*Weight(dot(SP-CVP,CVN),dot(CN,Read3(a1.y)*2.-1.));
    }
    return L;
}

// Function 748
float flatNoise( vec3 p ) {
    p *= vec3(4.5, 3.0, 1.78);
    float f;
    f  = 0.5000*noise( p.xz ); p = m*p*2.02;
    f += 0.2500*noise( p.xz ); p = m*p*2.03;
    f += 0.1250*noise( p.xz ); p = m*p*2.01;
    f += 0.0625*noise( p.xz );
    
    float freq = 0.5;
    return smoothstep(freq + 0.08, freq + 0.1, f);
}

// Function 749
float noise(vec2 p)
{
  	vec2 f  = smoothstep(0.0, 1.0, fract(p));
  	p  = floor(p);
  	float n = p.x + p.y*57.0;
  	return mix(mix(rand(n+0.0), rand(n+1.0),f.x), mix( rand(n+57.0), rand(n+58.0),f.x),f.y);
}

// Function 750
vec3 voronoi( in vec3 x )
{
    vec3 p = floor( x );
    vec3 f = fract( x );

	float id = 0.0;
    vec2 res = vec2( 100.0 );
    for( int k=-1; k<=1; k++ )
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec3 b = vec3( float(i), float(j), float(k) );
        vec3 r = vec3( b ) - f + random3f( p + b );
        float d = dot( r, r );

        if( d < res.x )
        {
			id = dot( p+b, vec3(1.0,57.0,113.0 ) );
            res = vec2( d, res.x );			
        }
        else if( d < res.y )
        {
            res.y = d;
        }
    }

    return vec3( sqrt( res ), abs(id) );
}

// Function 751
float goldNoise(vec2 coord, float seed) {
	float phi = 1.6180339887498 * 0000.1;
  	float pi2 = PI * 0.1;
  	float sq2 = 1.4142135623730 * 10000.;
  	float temp = fract(
    	sin(dot(coord*(seed+phi), vec2(phi, pi2))) * sq2
  	);
  	return temp;
}

// Function 752
float noise( in vec2 p ){
   
    vec2 i = floor(p); p -= i; 
    p *= p*p*(p*(p*6. - 15.) + 10.);
    //p *= p*(3. - p*2.);  

    return mix( mix( hash21(i + vec2(0, 0)), 
                     hash21(i + vec2(1, 0)), p.x),
                mix( hash21(i + vec2(0, 1)), 
                     hash21(i + vec2(1, 1)), p.x), p.y);
}

// Function 753
float voronoi(vec2 uv, float t, float seed, float size) {
    
    float minDist = 100.;
    
    float gridSize = size;
    
    vec2 cellUv = fract(uv * gridSize) - 0.5;
    vec2 cellCoord = floor(uv * gridSize);
    
    for (float x = -1.; x <= 1.; ++ x) {
        for (float y = -1.; y <= 1.; ++ y) {
            vec2 cellOffset = vec2(x,y);
            
            // Random 0-1 for each cell
            vec2 rand01Cell = rand01(cellOffset + cellCoord + seed);
			
            // Get position of point in cell
            vec2 point = cellOffset + sin(rand01Cell * (t+10.)) * .5;
            
			// Get distance between pixel and point
            float dist = distFn(t, cellUv, point);
    		minDist = min(minDist, dist);
        }
    }
    
    return minDist;
}

// Function 754
float Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel2, (uv+0.5)/256.0, -100.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 755
float smoothNoise(vec3 x) {
    vec3 p = floor(x);
    vec3 n = fract(x);
    vec3 f = n*n*(3.0-2.0*n);
    float winy = 157.0;
    float winz = 113.0;
    
    float wx = p.x+winy*p.y+winz*p.z;
    return mix(
        	mix(
                mix(hash2(wx+0.0)     , hash2(wx+1.0), f.x),
                mix(hash2(wx+0.0+winy), hash2(wx+1.0+winy), f.x),
                f.y),
        	mix(
                mix(hash2(wx+0.0+winz)     , hash2(wx+1.0+winz), f.x),
                mix(hash2(wx+0.0+winy+winz), hash2(wx+1.0+winy+winz), f.x),
                f.y)
        , f.z);
}

// Function 756
float cloudNoise3D(vec3 uv, vec3 _wind)
{
    float v = 1.0-voronoi3D(uv*20.0+_wind);
    float fs = fbm3Dsimple(uv*40.0+_wind);
    float mask = fbm3Dsimple(uv*0.1+_wind);
    return clamp(v*fs*mask, 0.0, 1.0);
}

// Function 757
float Noise2(vec2 uv)
{
    vec2 corner = floor(uv);
	float c00 = N2(corner + vec2(0.0, 0.0));
	float c01 = N2(corner + vec2(0.0, 1.0));
	float c11 = N2(corner + vec2(1.0, 1.0));
	float c10 = N2(corner + vec2(1.0, 0.0));
    
    vec2 diff = fract(uv);
    
    diff = diff * diff * (vec2(3) - vec2(2) * diff);
    //diff = smoothstep(vec2(0), vec2(1), diff);
    
    return mix(mix(c00, c10, diff.x), mix(c01, c11, diff.x), diff.y);
}

// Function 758
float noise(in vec2 co){
    float a = fract(co.x * 10.5 + co.y * 7.5 + fract(iDate.a));
    a = fract(715.5 * a * a + 57.1 * co.x);
    a = fract(1371.5 * a * a + 757.1 * co.y);
    return a;
}

// Function 759
float noise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    vec2 uv = (p.xy + vec2(37.0, 17.0) * p.z) + f.xy;
    vec2 rg = textureLod( iChannel0, (uv+.5)/256., 0.).yx;
    return mix(rg.x, rg.y, f.z);
}

// Function 760
float noise( in vec3 x, float step1, float step2)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    float n = mix(mix(mix( hash(p+vec3(0,0,0)), 
                        hash(p+vec3(1,0,0)),f.x),
                   mix( hash(p+vec3(0,1,0)), 
                        hash(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(p+vec3(0,0,1)), 
                        hash(p+vec3(1,0,1)),f.x),
                   mix( hash(p+vec3(0,1,1)), 
                        hash(p+vec3(1,1,1)),f.x),f.y),f.z);
    n = smoothstep(step1, step2, n);
    return n;
}

// Function 761
float perlinNoise(vec3 x) {
    const vec3 step = vec3(110., 241., 171.);

    vec3 id = floor(x);
    vec3 f = fract(x);
 
    // For performance, compute the base input to a 1D hash from the integer part of the argument and the 
    // incremental change to the 1D based on the 3D -> 1D wrapping
    float n = dot(id, step);

    vec3 u = hermiteInter(f);
    return mix(mix(mix( hash1(n + dot(step, vec3(0, 0, 0))), hash1(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash1(n + dot(step, vec3(0, 1, 0))), hash1(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash1(n + dot(step, vec3(0, 0, 1))), hash1(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash1(n + dot(step, vec3(0, 1, 1))), hash1(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

// Function 762
vec2 CenterOfVoronoiCell (vec2 local, vec2 global, float time)
{
    vec2 point = local + global;
    return local +
        	vec2( IsEven( int(point.y) ) ? 0.5 : 0.0, 0.0 ) +		// hex
        	(sin( time * Hash22( point ) * 0.628 ) * 0.5 + 0.5);	// animation
}

// Function 763
vec4 noise(vec2 p, float lod){return texture(iChannel0,p/iChannelResolution[0].xy,lod);}

// Function 764
float noise3D(vec3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
	const vec3 s = vec3(1, 113, 57);
	
	vec3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride vector for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    
	p -= ip; // Cell's fractional component.
	
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: fract(sin(x)*largeNumber).
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
	
    // Interpolating along Y.
    h.xy = mix(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return mix(h.x, h.y, p.z); // Range: [0, 1].
	
}

// Function 765
float bicubicNoise(in vec2 p)
{
    vec2 fp = fract(p);
    vec2 ip = floor(p);
    
    float s99 = hash12(ip+vec2(-1,-1)), s19 = hash12(ip+vec2(1,-1));
    float s00 = hash12(ip+vec2(0,0)),   s20 = hash12(ip+vec2(2,0));
    float s91 = hash12(ip+vec2(-1, 1)), s11 = hash12(ip+vec2(1, 1));
    float s02 = hash12(ip+vec2(0,2)),   s22 = hash12(ip+vec2(2,2));
    float s09 = hash12(ip+vec2(0,-1)),  s29 = hash12(ip+vec2(2,-1));
    float s90 = hash12(ip+vec2(-1,0)),  s10 = hash12(ip+vec2(1,0));
    float s01 = hash12(ip+vec2(0,1)),   s21 = hash12(ip+vec2(2,1));
    float s92 = hash12(ip+vec2(-1,2)),  s12 = hash12(ip+vec2(1,2));
    
    float rz =  eval(eval(s99, s09, s19, s29, fp.x), eval(s90, s00, s10, s20, fp.x),
                eval(s91, s01, s11, s21, fp.x), eval(s92, s02, s12, s22, fp.x), fp.y);
    
    //return rz;
    return smoothstep(0.0,1.,rz);
}

// Function 766
float noiseT(in vec2 p) {
    return texture(iChannel0, p / 256.0, -100.0).x * 2.0 - 1.0;
}

// Function 767
float noise(vec3 x) {
  const vec3 step = vec3(110, 241, 171);

  vec3 i = floor(x);
  vec3 f = fract(x);

  // For performance, compute the base input to a 1D hash from the integer part of the argument and the
  // incremental change to the 1D based on the 3D -> 1D wrapping
  float n = dot(i, step);

  vec3 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                 mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
             mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                 mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

// Function 768
float noise(vec3 uv)
{
    vec3 fr = fract(uv.xyz);
    vec3 fl = floor(uv.xyz);
    float h000 = Hash3d(fl);
    float h100 = Hash3d(fl + zeroOne.yxx);
    float h010 = Hash3d(fl + zeroOne.xyx);
    float h110 = Hash3d(fl + zeroOne.yyx);
    float h001 = Hash3d(fl + zeroOne.xxy);
    float h101 = Hash3d(fl + zeroOne.yxy);
    float h011 = Hash3d(fl + zeroOne.xyy);
    float h111 = Hash3d(fl + zeroOne.yyy);
    return mixP(
        mixP(mixP(h000, h100, fr.x),
             mixP(h010, h110, fr.x), fr.y),
        mixP(mixP(h001, h101, fr.x),
             mixP(h011, h111, fr.x), fr.y)
        , fr.z);
}

// Function 769
float fractalNoiseLow(vec2 vl, out float mainWave) {
    #if SHARP_MODE==1
    const float persistance = 2.4;
    float frequency = 2.2;
    const float freq_mul = 2.2;
    float amplitude = .4;
#else
    const float persistance = 3.0;
    float frequency = 2.3;
    const float freq_mul = 2.3;
    float amplitude = .7;
#endif
    
    float rez = 0.0;
    vec2 p = vl;
    
    float mainOfset = (iTime + 40.)/ 2.;
    
    vec2 waveDir = vec2(p.x+ mainOfset, p.y + mainOfset);
    float firstFront = amplitude + 
			        (valueNoiseSimple(p) * 2. - 1.);
    mainWave = firstFront * valueNoiseSimple(p + mainOfset);
    
    rez += mainWave;
    amplitude /= persistance;
    p *= unique_transform;
    p *= frequency;
    

    float timeOffset = iTime / 4.;

    
    for (int i = 1; i < OCTAVES - 5; i++) {
        waveDir = p;
        waveDir.x += timeOffset;
        rez += amplitude * sin(valueNoiseSimple(waveDir * frequency) * .5 );
        amplitude /= persistance;
        p *= unique_transform;
        frequency *= freq_mul;

        timeOffset *= 1.025;

        timeOffset *= -1.;
    }

    return rez;
}

// Function 770
vec2 noise2(vec2 n){ return vec2(noise(vec2(n.x+0.2, n.y-0.6)), noise(vec2(n.y+3., n.x-4.)));}

// Function 771
vec4 SqrNoise( in vec3 p )
{
	float f=noise(p);
	vec3 n=GrNoise(p);
	return vec4(2.*f*n,f*f);
}

// Function 772
float noise(in vec2 uv) {
    vec2 f = fract(uv);
    vec2 i = floor(uv);
    
    float a = rand(i);
    float b = rand(i + vec2(0.0, 1.0));
    float c = rand(i + vec2(1.0, 0.0));
    float d = rand(i + vec2(1.0, 1.0));
    
    vec2 u = -2. * f * f * f + 3. * f * f;
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.x);
}

// Function 773
float noise( in vec2 p ) {	
    p *= rot(1.941611);
    return sin(p.x) * .25 + sin(p.y) * .25 + .50;
}

// Function 774
vec2 noise2(float x)
{
  return vec2(noise(x), noise(fract(noise(x * 5.0) * 11.0)));
}

// Function 775
float octNoise(vec2 uv, int octaves){
    float sum = 0.0;
    float f = 1.0;
    for (int i = 0; i < octaves; ++i){
        sum += (noise(uv*f) - 0.5) * 1. /f;
        f *= 2.0;
    }
    return sum;
}

// Function 776
vec3 voronoi( in vec2 x, in vec2 dir)
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 1e5;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;

    
 		if( dot(r-mr,r-mr) > 1e-5 ) {
            vec2 l = r-mr;
            
            if (dot(dir, l) > 1e-5) {
            	md = min(md, dot(0.5*(mr+r), l)/dot(dir, l));
            }
        }
        
    }
    
    return vec3( md, n+mg);
}

// Function 777
vec2 perlin (vec2 p)
{
    vec2 a = texture(iChannel1, vec2(floor(p.x), ceil (p.y))/64.).xy;
    vec2 b = texture(iChannel1, vec2(ceil (p.x), ceil (p.y))/64.).xy;
    vec2 c = texture(iChannel1, vec2(floor(p.x), floor(p.y))/64.).xy;
    vec2 d = texture(iChannel1, vec2(ceil (p.x), floor(p.y))/64.).xy;
    
    vec2 m = smoothstep(0.,1.,fract(p));
    return mix(mix(c, a, m.y), mix(d, b, m.y), m.x) - .5;
}

// Function 778
float Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 779
float perlin_noise(vec3 p)
{
    vec3 pi = floor(p);
    vec3 pf = p - pi;

    vec3 w = pf * pf * (3.0 - 2.0 * pf);

    return  mix(
                mix(
                    mix(dot(pf - vec3(0, 0, 0), hash33(pi + vec3(0, 0, 0))),
                        dot(pf - vec3(1, 0, 0), hash33(pi + vec3(1, 0, 0))),
                        w.x),
                    mix(dot(pf - vec3(0, 0, 1), hash33(pi + vec3(0, 0, 1))),
                        dot(pf - vec3(1, 0, 1), hash33(pi + vec3(1, 0, 1))),
                        w.x),
                    w.z),
                mix(
                    mix(dot(pf - vec3(0, 1, 0), hash33(pi + vec3(0, 1, 0))),
                        dot(pf - vec3(1, 1, 0), hash33(pi + vec3(1, 1, 0))),
                        w.x),
                    mix(dot(pf - vec3(0, 1, 1), hash33(pi + vec3(0, 1, 1))),
                        dot(pf - vec3(1, 1, 1), hash33(pi + vec3(1, 1, 1))),
                        w.x),
                    w.z),
                w.y);
}

// Function 780
float noise( in vec3 x )
{
    ivec3 p = ivec3(floor(x));
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
    
	ivec2 uv = p.xy + ivec2(37,17)*p.z;

	vec2 rgA = texelFetch( iChannel0, (uv+ivec2(0,0))&255, 0 ).yx;
    vec2 rgB = texelFetch( iChannel0, (uv+ivec2(1,0))&255, 0 ).yx;
    vec2 rgC = texelFetch( iChannel0, (uv+ivec2(0,1))&255, 0 ).yx;
    vec2 rgD = texelFetch( iChannel0, (uv+ivec2(1,1))&255, 0 ).yx;

    vec2 rg = mix( mix( rgA, rgB, f.x ),
                   mix( rgC, rgD, f.x ), f.y );
    return mix( rg.x, rg.y, f.z );
}

// Function 781
vec3 noised( in vec2 x )
{
    vec2 f = fract(x);
    vec2 u = f*f*(3.0-2.0*f);

#if 1
    // texel fetch version
    ivec2 p = ivec2(floor(x));
    float a = texelFetch( iChannel0, (p+ivec2(0,0))&255, 0 ).x;
	float b = texelFetch( iChannel0, (p+ivec2(1,0))&255, 0 ).x;
	float c = texelFetch( iChannel0, (p+ivec2(0,1))&255, 0 ).x;
	float d = texelFetch( iChannel0, (p+ivec2(1,1))&255, 0 ).x;
#else    
    // texture version    
    vec2 p = floor(x);
	float a = textureLod( iChannel0, (p+vec2(0.5,0.5))/256.0, 0.0 ).x;
	float b = textureLod( iChannel0, (p+vec2(1.5,0.5))/256.0, 0.0 ).x;
	float c = textureLod( iChannel0, (p+vec2(0.5,1.5))/256.0, 0.0 ).x;
	float d = textureLod( iChannel0, (p+vec2(1.5,1.5))/256.0, 0.0 ).x;
#endif
    
	return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
				6.0*f*(1.0-f)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));
}

// Function 782
float noise1( in vec2 x )
{
  vec2 p  = floor(x);
  vec2 f  = smoothstep(0.0, 1.0, fract(x));
  float n = p.x + p.y*57.0;
  return mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
    mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);
}

// Function 783
float fast_voronoi( in vec2 x )
{
    x -= .5;
    vec2 n = floor(x);
    vec2 f = fract(x) + .5;
    
    float md = MAX_D;
    for( int j=0; j<=1; j++ )
    for( int i=0; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec3 o = hash32( n + g );
        vec2 r = g + o.xy - f;
        md = min(md, dot(r,r));
    }
    
    return md;
}

// Function 784
float simplex(vec2 p) {
	float f2 = 1.0/ 2.0;
	float s = (p.x+ p.y)* f2;
	int i = int(floor(p.x+ s));
	int j = int(floor(p.y+ s));

	float g2 = 1.0/ 4.0;
	float t = float((i+ j))* g2;
	float x0 = float(i)- t;
	float y0 = float(j)- t;
	x0 = p.x- x0;
	y0 = p.y- y0;
	int i1,j1,k1;
	int i2,j2,k2;
	if(x0>= y0) {
		i1 = 1;
		j1 = 0;
		k1 = 0;
		i2 = 1;
		j2 = 1;
		k2 = 0;
	} else {
		i1 = 0;
		j1 = 1;
		k1 = 0;
		i2 = 1;
		j2 = 1;
		k2 = 0;
	}
	float x1 = x0- float(i1)+ g2;
	float y1 = y0- float(j1)+ g2;
	float x2 = x0- float(i2)+ 2.0* g2;
	float y2 = y0- float(j2)+ 2.0* g2;
	float x3 = x0- 1.0+ 3.0* g2;
	float y3 = y0- 1.0+ 3.0* g2;
	vec2 ijk0 = vec2(i,j);
	vec2 ijk1 = vec2(i+ i1,j+ j1);
	vec2 ijk2 = vec2(i+ i2,j+ j2);
	vec2 ijk3 = vec2(i+ 1,j+ 1);
	vec2 gr0 = normalize(vec2(noise(ijk0),noise(ijk0* 2.01)));
	vec2 gr1 = normalize(vec2(noise(ijk1),noise(ijk1* 2.01)));
	vec2 gr2 = normalize(vec2(noise(ijk2),noise(ijk2* 2.01)));
	vec2 gr3 = normalize(vec2(noise(ijk3),noise(ijk3* 2.01)));
	float n0 = 0.0;
	float n1 = 0.0;
	float n2 = 0.0;
	float n3 = 0.0;
	float t0 = 0.5- x0* x0- y0* y0;
	if(t0>= 0.0) {
		t0 *= t0;
		n0 = t0* t0* dot(gr0,vec2(x0,y0));
	}
	float t1 = 0.5- x1* x1- y1* y1;
	if(t1>= 0.0) {
		t1 *= t1;
		n1 = t1* t1* dot(gr1,vec2(x1,y1));
	}
	float t2 = 0.5- x2* x2- y2* y2;
	if(t2>= 0.0) {
		t2 *= t2;
		n2 = t2* t2* dot(gr2,vec2(x2,y2));
	}
	float t3 = 0.5- x3* x3- y3* y3;
	if(t3>= 0.0) {
		t3 *= t3;
		n3 = t3* t3* dot(gr3,vec2(x3,y3));
	}
	return 96.0* (n0+ n1+ n2+ n3);
}

// Function 785
float noise( in vec2 x )
{
	//return texture( iChannel0, (x+0.5)/256.0 ).x;

	vec2 p = floor(x);
    vec2 f = fract(x);

	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);

	return textureLod( iChannel0, (uv+0.5)/256.0, -0.0 ).x;
}

// Function 786
float power_noise( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);
    #if ACCURACY == 2
    // shift by half-cell
    vec2 h = step(.5,f) - 2.;
    n += h; f -= h;
    #endif

    float md = MAX_D;
    #if ACCURACY == 2
    for( int j=0; j<=3; j++ )
    for( int i=0; i<=3; i++ )
    #else
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    #endif
    {
        vec2 g = vec2(float(i),float(j));
        vec3 o = hash32( n + g );
        vec2 r = g + o.xy - f;
        md = min( md, dot(r,r) - o.z*o.z );
    }
    
    return md;
}

// Function 787
float noiseo(vec2 st)
{
    vec2 f = fract(st);
    vec2 i = floor(st);
    
    vec2 u = f * f * f * (f * (f * 6. - 15.) + 10.);
    
    float r = mix( mix( dot( random(i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ),
                     dot( random(i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( random(i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ),
                     dot( random(i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
    return r * .5 + .5;
}

// Function 788
float noise(vec2 p)
{
	vec2 f = fract(p);
    p = floor(p);
    f = f*f*(3.0-2.0*f);
    float res = mix(mix(hash(p),
						hash(p + vec2(1.0, 0.0)), f.x),
					mix(hash(p + vec2(0.0, 1.0)),
						hash(p + vec2(1.0, 1.0)), f.x), f.y);
    return res;
}

// Function 789
float perlin_noise(vec2 p)
{
	vec2 pi = floor(p);
    vec2 pf = p-pi;
    
    vec2 w = pf*pf*(3.-2.*pf);
    
    float f00 = dot(hash22(pi+vec2(.0,.0)),pf-vec2(.0,.0));
    float f01 = dot(hash22(pi+vec2(.0,1.)),pf-vec2(.0,1.));
    float f10 = dot(hash22(pi+vec2(1.0,0.)),pf-vec2(1.0,0.));
    float f11 = dot(hash22(pi+vec2(1.0,1.)),pf-vec2(1.0,1.));
    
    float xm1 = mix(f00,f10,w.x);
    float xm2 = mix(f01,f11,w.x);
    
    float ym = mix(xm1,xm2,w.y); 
    return ym;
   
}

// Function 790
float lightningNoise (vec2 forPos)
{
    forPos *= 4.0;
    forPos.y *= 0.85;
    float wobbleAmount1 = sin(forPos.y) * 0.5 + sin(forPos.y * 2.0) * 0.25 + sin(forPos.y * 4.0) * 0.125 + sin(forPos.y * 8.0) * 0.0625;
    float wobbleAmount2 = sin(forPos.x) * 0.5 + sin(forPos.x * 2.0) * 0.25 + sin(forPos.x * 4.0) * 0.125 + sin(forPos.x * 8.0) * 0.0625;
    float horizontalStrike = 1.0 - abs(sin(forPos.x + wobbleAmount1 * 1.1));
    float verticalStrike = 1.0 - abs(cos(forPos.y + wobbleAmount2 * 1.1));
    return (horizontalStrike + verticalStrike) * 0.35;
}

// Function 791
float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p); f = f*f*(3.-2.*f); // smoothstep

    float v= mix( mix( mix(hash31(i+vec3(0,0,0)),hash31(i+vec3(1,0,0)),f.x),
                       mix(hash31(i+vec3(0,1,0)),hash31(i+vec3(1,1,0)),f.x), f.y), 
                  mix( mix(hash31(i+vec3(0,0,1)),hash31(i+vec3(1,0,1)),f.x),
                       mix(hash31(i+vec3(0,1,1)),hash31(i+vec3(1,1,1)),f.x), f.y), f.z);
	return   MOD==0 ? v
	       : MOD==1 ? 2.*v-1.
           : MOD==2 ? abs(2.*v-1.)
                    : 1.-abs(2.*v-1.);
}

// Function 792
vec4 Noise( in ivec2 x )
{
	return textureLod( iChannel0, (vec2(x)+0.5)/256.0, 0.0 );
}

// Function 793
float smoothNoise(float x, float y) {

   //get fractional part of x and y
   float fractX = x - floor(x);
   float fractY = y - floor(y);
   
   //wrap around
   float x1 = mod((floor(x) + iResolution.x), iResolution.x);
   float y1 = mod((floor(y) + iResolution.y), iResolution.y);
   
   //neighbor values
   float x2 = mod((x1 + iResolution.x - 1.0), iResolution.x);
   float y2 = mod((y1 + iResolution.y - 1.0), iResolution.y);

   //smooth the noise with bilinear interpolation
   float value = 0.0;
   value += fractX       * fractY       * snoise(vec2(x1, y1));
   value += fractX       * (1.0 - fractY) * snoise(vec2(x1, y2));
   value += (1.0 - fractX) * fractY       * snoise(vec2(x2, y1));
   value += (1.0 - fractX) * (1.0 - fractY) * snoise(vec2(x2, y2));

   return value;
}

// Function 794
float snoise(vec3 v)
  { 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

// Function 795
float noise1f( sampler2D tex, in vec2 x )
{
    return texture(tex,(x+0.5)/64.0).x;
}

// Function 796
float cosNoise( in vec2 pos)
{
	return 0.5 * ( sin(pos.x) * sin(pos.y));   
}

// Function 797
float fractalNoise(float freq, float lacunarity, float decay, vec2 threshold, vec3 p)
{  
  float res=0.0;
  float currentFreq=freq;
  float weight=1.0;
  float maxValue=0.0;
  // Always 5 octaves because the condition in the loop must be a constant.
  for(int i=0;i<5;i++)
  {

    res+=weight*cnoise(currentFreq*p);
    if(threshold.x==0.0||(res>threshold.x*float(i)&&res<threshold.y*float(i)))
          weight/=decay;
    else weight/=3.0;
    

    currentFreq*=lacunarity;
  }
  return res/5.0;
}

// Function 798
float SmoothNoise3D(vec3 uv) {
    vec3 lv = fract(uv);
    vec3 id = floor(uv);
    
    lv = smoothstep(0.0, 1.0, lv);
    
    float fbl = HASH31(id + vec3(0, 0, 0));
    float fbr = HASH31(id + vec3(1, 0, 0));
    float fb = mix(fbl, fbr, lv.x);
    float ftl = HASH31(id + vec3(0, 1, 0));
    float ftr = HASH31(id + vec3(1, 1, 0));
    float ft = mix(ftl, ftr, lv.x);
    
    float bbl = HASH31(id + vec3(0, 0, 1));
    float bbr = HASH31(id + vec3(1, 0, 1));
    float bb = mix(bbl, bbr, lv.x);
    float btl = HASH31(id + vec3(0, 1, 1));
    float btr = HASH31(id + vec3(1, 1, 1));
    float bt = mix(btl, btr, lv.x);
    
    return mix(mix(fb, ft, lv.y), mix(bb, bt, lv.y), lv.z);
}

// Function 799
float noise(vec2 p)
{
    vec2 f = fract( p ), i = p-f, u = (3.-2.*f)*f*f;

    return mix( mix( dot( hash2( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( hash2( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hash2( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( hash2( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 800
vec2 trinoise(vec2 uv){
    const float sq = sqrt(3./2.);
    uv.x *= sq;
    uv.y -= .5*uv.x;
    vec2 d = fract(uv);
    uv -= d;
    if(dot(d,vec2(1))<1.){
        float n1 = hash(uv);
        float n2 = hash(uv+vec2(1,0));
        float n3 = hash(uv+vec2(0,1));
        float nmid = mix(n2,n3,d.y);
        float ng = mix(n1,n3,d.y);
        float dx = d.x/(1.-d.y);
        return vec2(mix(ng,nmid,dx),min(min((1.-dx)*(1.-d.y),d.x),d.y));
	}else{
    	float n2 = hash(uv+vec2(1,0));
        float n3 = hash(uv+vec2(0,1));
        float n4 = hash(uv+1.);
        float nmid = mix(n2,n3,d.y);
        float nd = mix(n2,n4,d.y);
        float dx = (1.-d.x)/(d.y);
        return vec2(mix(nd,nmid,dx),min(min((1.-dx)*d.y,1.-d.x),1.-d.y));
	}
    return vec2(0);
}

// Function 801
float noise(vec3 m) {
    return   0.5333333*simplex3d(m)
			+0.2666667*simplex3d(2.0*m)
			+0.1333333*simplex3d(4.0*m)
			+0.0666667*simplex3d(8.0*m);
}

// Function 802
float InterferenceNoise( vec2 uv )
{
	float displayVerticalLines = 483.0;
    float scanLine = floor(uv.y * displayVerticalLines); 
    float scanPos = scanLine + uv.x;
	float timeSeed = fract( iTime * 123.78 );
    
    return InterferenceSmoothNoise1D( scanPos * 234.5 + timeSeed * 12345.6 );
}

// Function 803
fixed noiseSpere(float zoom, float3 subnoise, float anim)
			{
				fixed s = 0.0;

				if (sphere <sqRadius) {
					if (_Detail>0.0) s = noise4q(fixed4(surfase*zoom*3.6864 + subnoise, fragTime*_SpeedHi))*0.625;
					if (_Detail>1.0) s =s*0.85+noise4q(fixed4(surfase*zoom*61.44 + subnoise*3.0, fragTime*_SpeedHi*3.0))*0.125;
					if (_Detail>2.0) s =s*0.94+noise4q(fixed4(surfase*zoom*307.2 + subnoise*5.0, anim*5.0))*0.0625;//*0.03125;
					if (_Detail>3.0) s =s*0.98+noise4q(fixed4(surfase*zoom*600.0 + subnoise*6.0, fragTime*_SpeedLow*6.0))*0.03125;
					if (_Detail>4.0) s =s*0.98+noise4q(fixed4(surfase*zoom*1200.0 + subnoise*9.0, fragTime*_SpeedLow*9.0))*0.01125;
				}
				return s;
			}

// Function 804
float Noise(in vec3 p)
{
    vec3 i = floor(p);
	vec3 f = fract(p); 
	f *= f * (3.0-2.0*f);

    return mix(
		mix(mix(Hash(i  			), 	Hash(i + add.xyy),f.x),
			mix(Hash(i + add.yxy),		Hash(i + add.xxy),f.x),
			f.y),
		mix(mix(Hash(i + add.yyx),    	Hash(i + add.xyx),f.x),
			mix(Hash(i + add.yxx), 		Hash(i + add.xxx),f.x),
			f.y),
		f.z);
}

// Function 805
vec4 voronoi( in vec2 x )
{
    vec2 mouse = get_mouse();
    x.x += mouse.x*.5;
    x.y += mouse.y*.5;
    
	float wave = (time*.7) + mouse.x; // mouse.x; // time;
    
    vec2 n = floor(x);
    vec2 f = fract(x);
	float ox = 0.;
	vec2 mg, mr;

    float md = 8.;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
        ox = o.x;
        o = .5 + .5 *sin(o * wave + PI2);
        //o = .5 + .5 * sin( time + PI2 * o );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    md = 8.;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
        ox = o.x;
        o = .5 + .5 *sin(o * wave + PI2);
        //o = .5 + .5 * sin( time + PI2 * o );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>0.00001 )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec4( md, mr, ox );
}

// Function 806
float snoise( vec3 v ) {
	const vec2 C = vec2(1.0/6.0, 1.0/3.0) ;
	const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

	// First corner
	vec3 i = floor(v + dot(v, C.yyy) );
	vec3 x0 = v - i + dot(i, C.xxx);

	// Other corners
  	vec3 g = step(x0.yzx, x0.xyz);
  	vec3 l = 1.0 - g;
  	vec3 i1 = min( g.xyz, l.zxy );
  	vec3 i2 = max( g.xyz, l.zxy );

	vec3 x1 = x0 - i1 + C.xxx;
	vec3 x2 = x0 - i2 + C.yyy;
	vec3 x3 = x0 - D.yyy;

	// Permutations
 	i = mod289(i);
  	vec4 p = permute( permute( permute( 
		i.z + vec4(0.0, i1.z, i2.z, 1.0 )) +
		i.y + vec4(0.0, i1.y, i2.y, 1.0 )) + 
		i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
	
	//p = permute(p + seed); // optional seed value

	// Gradients: 7x7 points over a square, mapped onto an octahedron.
	// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  	float n_ = 0.142857142857; // 1.0/7.0
 	vec3 ns = n_ * D.wyz - D.xzx;

	vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
	
	vec4 x_ = floor(j * ns.z);
	vec4 y_ = floor(j - 7.0 * x_ );
	
	vec4 x = x_ *ns.x + ns.yyyy;
	vec4 y = y_ *ns.x + ns.yyyy;
	vec4 h = 1.0 - abs(x) - abs(y);
	
	vec4 b0 = vec4( x.xy, y.xy );
	vec4 b1 = vec4( x.zw, y.zw );

  	vec4 s0 = floor(b0)*2.0 + 1.0;
  	vec4 s1 = floor(b1)*2.0 + 1.0;
  	vec4 sh = -step(h, vec4(0.0));

  	vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  	vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  	vec3 p0 = vec3(a0.xy,h.x);
  	vec3 p1 = vec3(a0.zw,h.y);
  	vec3 p2 = vec3(a1.xy,h.z);
  	vec3 p3 = vec3(a1.zw,h.w);

	// Normalize gradients
  	vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  	p0 *= norm.x;
  	p1 *= norm.y;
  	p2 *= norm.z;
  	p3 *= norm.w;

	// Mix final noise value
  	vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  	m = m * m;
  	return 42.0 * dot( m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)) );
}

// Function 807
vec2 voronoi(in vec3 x){
    vec3 p = floor( x );
    vec3 f = fract( x );

	float id;
    vec2 res = vec2( 100.0 );
    for( int k=-1; k<=1; k++ )
    	for( int j=-1; j<=1; j++ )
    		for( int i=-1; i<=1; i++ ){
                vec3 b = vec3( float(i), float(j), float(k) );
                vec3 r = vec3( b ) - f + hash33( p + b );
                float d = dot( r, r );

                if( d < res.x ){
                    id = dot(p + b, vec3(20.31, 517., 113.));
                    res = vec2(d, res.x);
                }else if(d < res.y){
                    res.y = d;
                }
    }
    return vec2(res.x, abs(id));
}

// Function 808
float noise(vec2 p){
	return fract(sin(fract(sin(p.x) * (43.13311)) + p.y) * 31.0011);
}

// Function 809
float noise(in vec3 p)
{
#ifdef ANIMATE
	p.z += iTime * .75;
#endif
	
    vec3 i = floor(p);
	vec3 f = fract(p); 
	f *= f * (3.-2.*f);

    vec2 c = vec2(0,1);

    return mix(
		mix(mix(hash(i + c.xxx), hash(i + c.yxx),f.x),
			mix(hash(i + c.xyx), hash(i + c.yyx),f.x),
			f.y),
		mix(mix(hash(i + c.xxy), hash(i + c.yxy),f.x),
			mix(hash(i + c.xyy), hash(i + c.yyy),f.x),
			f.y),
		f.z);
}

// Function 810
float noise( in vec2 p )
{
    ivec2 i = ivec2(floor( p ));
     vec2 f =       fract( p );
	
	vec2 u = f*f*(3.0-2.0*f); // feel free to replace by a quintic smoothstep instead

    return mix( mix( dot( grad( i+ivec2(0,0) ), f-vec2(0.0,0.0) ), 
                     dot( grad( i+ivec2(1,0) ), f-vec2(1.0,0.0) ), u.x),
                mix( dot( grad( i+ivec2(0,1) ), f-vec2(0.0,1.0) ), 
                     dot( grad( i+ivec2(1,1) ), f-vec2(1.0,1.0) ), u.x), u.y);
}

// Function 811
float noise( in vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

	vec2 i = floor( p + (p.x+p.y)*K1 );
	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;

#if 1	
	//even more extra rotations for more flow!
	float t = iTime*.5;
	float co = cos(t); float si = sin(t);	
	a = RotCS(a,co,si);
	b = RotCS(b,co,si);
	c = RotCS(c,co,si);
#endif
	
    vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );

	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));

    return dot( n, vec3(70.0) );
	
}

// Function 812
vec3 TerrainNoiseRaw(vec2 p)
{
    vec2 pfract = fract(p);
    vec2 pfloor = floor(p);
    
    // Quintic interpolation factor (6x^5-15x^4+10x^3) and it's derivative (30x^4-60x^3+30x^2)
    vec2 lfactor = pfract * pfract * pfract * (pfract * (pfract * 6.0 - 15.0) + 10.0);
    vec2 lderiv  = 30.0 * pfract * pfract * (pfract * (pfract - 2.0) + 1.0);
    
    /** 
     * Noise LUT sample points (p = pfloor):
     *
     *      ┌───┬───┐
     *      │ c │ d │
     *      ├───┼───┤
     *      │ a │ b │
     *      p───┴───┘
	 */
    
    // Use textureLod instead of texture so that Angle (Windows Firefox) doesn't choke.
    float a = textureLod(iChannel0, (pfloor + vec2(0.5, 0.5)) * TexDim, 0.0).r;
    float b = textureLod(iChannel0, (pfloor + vec2(1.5, 0.5)) * TexDim, 0.0).r;
    float c = textureLod(iChannel0, (pfloor + vec2(0.5, 1.5)) * TexDim, 0.0).r;
    float d = textureLod(iChannel0, (pfloor + vec2(1.5, 1.5)) * TexDim, 0.0).r;
    
    /**
     * For the value (.r) we perform a bilinear interpolation with a 
     * quintic factor (biquintic?) over the four sampled points.
     *
     * .r could be written as:
     * 
     *    mix(mix(a, b, lfactor.x), mix(c, d, lfactor.x), lfactor.y)
     *
     * The mixes are factored out so that the individual components
     * (k0, k1, k2, k4) can be used in finding the derivative (for the normal).
     */
    
    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;
    
	return vec3(
        k0 + (k1 * lfactor.x) + (k2 * lfactor.y) + (k4 * lfactor.x * lfactor.y),  // Heightmap value
        lderiv * vec2(k1 + k4 * lfactor.y, k2 + k4 * lfactor.x));                 // Value derivative
}

// Function 813
float Noise( in vec2 x ) {
    vec2 p = floor(x);
    vec2 f = fract(x);
	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);
	return textureLod( iChannel0, (uv+118.4)/256.0, -100.0 ).x;
}

// Function 814
vec4 noise(float p){return texture(iChannel0,vec2(p/iChannelResolution[0].x,.0));}

// Function 815
float noise( vec3 P )
{
    //  https://github.com/BrianSharpe/Wombat/blob/master/Value3D.glsl

    // establish our grid cell and unit position
    vec3 Pi = floor(P);
    vec3 Pf = P - Pi;
    vec3 Pf_min1 = Pf - 1.0;

    // clamp the domain
    Pi.xyz = Pi.xyz - floor(Pi.xyz * ( 1.0 / 69.0 )) * 69.0;
    vec3 Pi_inc1 = step( Pi, vec3( 69.0 - 1.5 ) ) * ( Pi + 1.0 );

    // calculate the hash
    vec4 Pt = vec4( Pi.xy, Pi_inc1.xy ) + vec2( 50.0, 161.0 ).xyxy;
    Pt *= Pt;
    Pt = Pt.xzxz * Pt.yyww;
    vec2 hash_mod = vec2( 1.0 / ( 635.298681 + vec2( Pi.z, Pi_inc1.z ) * 48.500388 ) );
    vec4 hash_lowz = fract( Pt * hash_mod.xxxx );
    vec4 hash_highz = fract( Pt * hash_mod.yyyy );

    //	blend the results and return
    vec3 blend = Pf * Pf * Pf * (Pf * (Pf * 6.0 - 15.0) + 10.0);
    vec4 res0 = mix( hash_lowz, hash_highz, blend.z );
    vec4 blend2 = vec4( blend.xy, vec2( 1.0 - blend.xy ) );
    return dot( res0, blend2.zxzx * blend2.wwyy );
}

// Function 816
float snoise( vec2 uv )
{
    return noise( uv ) * 2.0 - 1.0;
}

// Function 817
float pnoise(vec2 co, int oct)
{
	float total = 0.0;
	float m = 0.0;
	
	for(int i=0; i<oct; i++)
	{
		float freq = pow(2.0, float(i));
		float amp  = pow(0.5, float(i));
		
		total += noise(freq * co) * amp;
		m += amp;
	}
	
	return total/m;
}

// Function 818
float noise(vec2 p){return hash(mod(p.x+p.y*57.,1024.0));}

// Function 819
float noise1d( float p ) {
    float i = floor( p );
    float f = fract( p );
    float u = f*f*f*(f*(f*6.0-15.0)+10.0);
    float v = mix( hash1(i), hash1(i + 1.), u);              
    return v;
}

// Function 820
float vnoise2(vec2 p) {
    vec2 i = floor(p);
	vec2 f = fract(p);
    
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    
    float c1 = b - a;
    float c2 = c - a;
    float c3 = d - c - b + a;
	
    vec2 u = f * f * (3.0 - 2.0 * f);
    
   	return a + u.x*c1 + u.y*c2 + u.x*u.y*c3;
}

// Function 821
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*157.0;
    return mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
               mix( hash(n+157.0), hash(n+158.0),f.x),f.y);
}

// Function 822
float fbm_noise(vec2 uv, int steps) {
	float v = 0.;
    for(int i = 0; i < steps; i++){
        float factor = pow(2.,float(i + 1)) / 2.;
    	v += snoise(uv * factor) / factor;
    }
    return v / ((pow(.5,float(steps))- 1.) / -.5);
}

// Function 823
float Noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 824
float noise3f( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel0, (uv+0.5)/256.0, -100.0 ).yx;
	return mix( rg.x, rg.y, f.z )*2.-1.;
}

// Function 825
float noise( in vec3 x )
{

    vec3 p = floor( x );
    vec3 z = fract( x );
    
    z *= z * z * ( 3.0 - 2.0 * z );
    
    float n = p.x + p.y * 57.0 + p.z * 113.0; 
    
    float a = hash( n );
    float b = hash( n + 1.0 );
    float c = hash( n + 57.0 );
    float d = hash( n + 58.0 );
    
    float e = hash( n + 113.0 );
    float f = hash( n + 114.0 );
    float g = hash( n + 170.0 );
    float h = hash( n + 171.0 );
    
    float i = hash( n + 210.0 );
    float j = hash( n + 211.0 );
    float k = hash( n + 250.0 );
    float l = hash( n + 251.0 );
    
    float res = mix( mix( mix ( a, b, z.x ), mix( c, d, z.x ), z.y ),
                     mix( mix ( e, f, z.x ), mix( g, h, z.x ), z.y ), 
                     z.z
    				 );
    
    return res;
    
}

// Function 826
vec4 simplex_noise(vec3 p, float r) {
    
    const float K1 = .333333333;
    const float K2 = .166666667;
    
    vec3 i = floor(p + (p.x + p.y + p.z) * K1);
    vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);
    
    vec3 e = step(vec3(0.), d0 - d0.yzx);
	vec3 i1 = e * (1. - e.zxy);
	vec3 i2 = 1. - e.zxy * (1. - e);
    
    vec3 d1 = d0 - (i1 - 1. * K2);
    vec3 d2 = d0 - (i2 - 2. * K2);
    vec3 d3 = d0 - (1. - 3. * K2);
    
    vec4 h = max(.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.);
    vec4 n = h * h * h * h * vec4(dot(d0, hash33(i, r)), dot(d1, hash33(i + i1, r)), dot(d2, hash33(i + i2, r)), dot(d3, hash33(i + 1., r)));
    
    return 70.*n;
}

// Function 827
float perlinfbm(vec3 p, float freq, int octaves)
{
    float G = exp2(-.85);
    float amp = 1.;
    float noise = 0.;
    for (int i = 0; i < octaves; ++i)
    {
        noise += amp * gradientNoise(p * freq, freq);
        freq *= 2.;
        amp *= G;
    }
    
    return noise;
}

// Function 828
vec4 noiseInterpolateDu(const in vec2 x) 
{ 
    vec2 x2 = x * x;
    vec2 u = x2 * x * (x * (x * 6.0 - 15.0) + 10.0); 
    vec2 du = 30.0 * x2 * (x * (x - 2.0) + 1.0);
    return vec4(u, du);
}

// Function 829
float perlinNoiseWarp(vec2 pos, vec2 scale, float strength, float phase, float factor, float spread, float seed)
{
    vec2 offset = vec2(spread, 0.0);
    strength *= 32.0 / max(scale.x, scale.y);
    
    vec4 gp;
    gp.x = perlinNoise(pos - offset.xy, scale, phase, seed);
    gp.y = perlinNoise(pos + offset.xy, scale, phase, seed);
    gp.z = perlinNoise(pos - offset.yx, scale, phase, seed);
    gp.w = perlinNoise(pos + offset.yx, scale, phase, seed);
    gp = pow(gp, vec4(factor));
    vec2 warp = vec2(gp.y - gp.x, gp.w - gp.z);
    return pow(perlinNoise(pos + warp * strength, scale, phase, seed), factor);
}

// Function 830
float Noiseff (float p)
{
  vec2 t;
  float ip, fp;
  ip = floor (p);
  fp = fract (p);
  fp = fp * fp * (3. - 2. * fp);
  t = Hashv2f (ip);
  return mix (t.x, t.y, fp);
}

// Function 831
vec4 voronoi(vec2 x, float rp)
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;
    float id = 0.0;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( mod(n + g, rp) );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            id = dot(mod(n + g, rp), vec2(7.,41.));
            md = d;
            mr = r;
            mg = g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( mod(n + g, rp) );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>0.00001 )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec4(md, mr, id);
}

// Function 832
vec3 getVoronoi(vec2 x){
    vec2 n=floor(x),f=fract(x),mr;
    float md=5.;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ ){
        vec2 g=vec2(float(i),float(j));
		vec2 o=0.5+0.5*sin(TIME_RATIO+6.2831*getHash2BasedProc(n+g));//animated
        vec2 r=g+o-f;
        float d=dot(r,r);
        if( d<md ) {md=d;mr=r;} }
    return vec3(md,mr);}

// Function 833
float voronoi(in vec2 p){
    
	vec2 g = floor(p), o; p -= g;
	
	vec3 d = vec3(1); // 1.4, etc. "d.z" holds the distance comparison value.
    
	for(int y = -1; y <= 1; y++){
		for(int x = -1; x <= 1; x++){
            
			o = vec2(x, y);
            o += hash22(g + o) - p;
            
			d.z = dot(o, o);            
            d.y = max(d.x, min(d.y, d.z));
            d.x = min(d.x, d.z); 
                       
		}
	}
	
	
    return d.y - d.x;
    // return d.x;
    // return max(d.y*.91 - d.x*1.1, 0.)/.91;
    // return sqrt(d.y) - sqrt(d.x); // etc.
}

// Function 834
vec4 voronoi (in vec2 x)
{
    // from https://www.shadertoy.com/view/ldl3W8
    // The MIT License
    // Copyright © 2013 Inigo Quilez
    
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = Hash22( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( iTime * ANIMATE + 6.2831*o );
        #endif	
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = Hash22( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( iTime * ANIMATE + 6.2831*o );
        #endif	
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>0.00001 )
        	md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec4( x - (n + mr + f), md, Hash21( mg + n ) );
}

// Function 835
float PerlinNoise2D(float x,float y)
{
    float sum = 0.0;
    float frequency =0.0;
    float amplitude = 0.0;
    for(int i=firstOctave;i<octaves + firstOctave;i++)
    {
        frequency = pow(2.0,float(i));
        amplitude = pow(persistence,float(i));
        sum = sum + InterpolationNoise(x*frequency,y*frequency)*amplitude;
    }
    
    return sum;
}

// Function 836
float orbitNoise1D(float p)
{
    float ip = floor(p);
    float fp = fract(p);
    float rz = 0.;
    float orbitRadius = .75;

    for (int i = -1; i <= 2; i++)
    {
        vec2 rn = hash21(float(i) + ip) - 0.5;
        float op = fp - float(i) + rn.y*orbitRadius;
        rz += nuttall(abs(op), 1.85)*rn.x*3.0*op;
    }
    return rz*0.5+0.5;
}

// Function 837
float perlin( vec3 P )
{
    // establish our grid cell and unit position
    vec3 Pi = floor(P);
    vec3 Pf = P - Pi;
    vec3 Pf_min1 = Pf - 1.0;

    // clamp the domain
    Pi.xyz = Pi.xyz - floor(Pi.xyz * ( 1.0 / 69.0 )) * 69.0;
    vec3 Pi_inc1 = step( Pi, vec3( 69.0 - 1.5 ) ) * ( Pi + 1.0 );

    // calculate the hash
    vec4 Pt = vec4( Pi.xy, Pi_inc1.xy ) + vec2( 50.0, 161.0 ).xyxy;
    Pt *= Pt;
    Pt = Pt.xzxz * Pt.yyww;
    const vec3 SOMELARGEFLOATS = vec3( 635.298681, 682.357502, 668.926525 );
    const vec3 ZINC = vec3( 48.500388, 65.294118, 63.934599 );
    vec3 lowz_mod = vec3( 1.0 / ( SOMELARGEFLOATS + Pi.zzz * ZINC ) );
    vec3 highz_mod = vec3( 1.0 / ( SOMELARGEFLOATS + Pi_inc1.zzz * ZINC ) );
    vec4 hashx0 = fract( Pt * lowz_mod.xxxx );
    vec4 hashx1 = fract( Pt * highz_mod.xxxx );
    vec4 hashy0 = fract( Pt * lowz_mod.yyyy );
    vec4 hashy1 = fract( Pt * highz_mod.yyyy );
    vec4 hashz0 = fract( Pt * lowz_mod.zzzz );
    vec4 hashz1 = fract( Pt * highz_mod.zzzz );

    // calculate the gradients
    vec4 grad_x0 = hashx0 - 0.49999;
    vec4 grad_y0 = hashy0 - 0.49999;
    vec4 grad_z0 = hashz0 - 0.49999;
    vec4 grad_x1 = hashx1 - 0.49999;
    vec4 grad_y1 = hashy1 - 0.49999;
    vec4 grad_z1 = hashz1 - 0.49999;
    vec4 grad_results_0 = inversesqrt( grad_x0 * grad_x0 + grad_y0 * grad_y0 + grad_z0 * grad_z0 ) * ( vec2( Pf.x, Pf_min1.x ).xyxy * grad_x0 + vec2( Pf.y, Pf_min1.y ).xxyy * grad_y0 + Pf.zzzz * grad_z0 );
    vec4 grad_results_1 = inversesqrt( grad_x1 * grad_x1 + grad_y1 * grad_y1 + grad_z1 * grad_z1 ) * ( vec2( Pf.x, Pf_min1.x ).xyxy * grad_x1 + vec2( Pf.y, Pf_min1.y ).xxyy * grad_y1 + Pf_min1.zzzz * grad_z1 );

    // Classic Perlin Interpolation
    vec3 blend = Pf * Pf * Pf * (Pf * (Pf * 6.0 - 15.0) + 10.0);
    vec4 res0 = mix( grad_results_0, grad_results_1, blend.z );
    vec4 blend2 = vec4( blend.xy, vec2( 1.0 - blend.xy ) );
    float final = dot( res0, blend2.zxzx * blend2.wwyy );
    return ( final * 1.1547005383792515290182975610039 );  // scale things to a strict -1.0->1.0 range  *= 1.0/sqrt(0.75)
}

// Function 838
float iqnoise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
		
	float k = 1.0+63.0*pow(1.0-v,4.0);
	
	float va = 0.0;
	float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = vec2( float(i),float(j) );
		vec3 o = hash3( p + g )*vec3(u,u,1.0);
		vec2 r = g - f + o.xy;
		float d = dot(r,r);
		float ww = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
		va += o.z*ww;
		wt += ww;
    }
	
    return va/wt;
}

// Function 839
float noise(const float time)
{
    return fract(sin(time * 1e4 * 12.9898) * 43758.5453) * 2. - 1.;
}

// Function 840
float noise( const in vec2 x ) {
    vec2 p = floor(x);
    vec2 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy) + f.xy;
	return textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).x;
}

// Function 841
vec3 cnoise(vec3 p)
{
    vec3 size = 1.0 / vec3(textureSize(iChannel0, 0));
	vec3 n0 = noise(p * size * 1.0);
    vec3 n1 = noise(p * size * 2.0);
    vec3 n2 = noise(p * size * 4.0);
    vec3 n3 = noise(p * size * 8.0);
    return (
                       r(n0) * 0.5 +
        n0 *           r(n1) * 0.25 +
        n0 * n1 *      r(n2) * 0.125 +
        n0 * n1 * n2 * r(n3) * 0.0625) * 1.06667;
}

// Function 842
float cosNoise( in vec2 p )
{
    return 0.5*( sin(p.x) + sin(p.y) );
}

// Function 843
float voronoi3D(vec3 n, float time){
    float dis = 0.9;
    for(float y = 0.0; y <= 1.0; y++){
        for(float x = 0.0; x <= 1.0; x++){
            for(float z = -1.0; z <= 1.0; z++){
                // Neighbor place in the grid
                vec3 p = floor(n) + vec3(x, y, z);
                float d = length((0.27 * sin(rand3D(p) * intensity + time * 2.0)) + vec3(x, y, z) - fract(n));
                dis = min(dis, d);
                }
            }
        }
    return dis;
    }

// Function 844
float perlinTexture(in vec2 pos)
{
    const float N = 5.0;
    
    vec2 pos2 = pos;
    
    float ind = 0.0;
    float ampl = 1.0;
    
    for(float n = 0.0; n < N; n ++)
    {
    	ind += texture(iChannel0, pos / iChannelResolution[0].x).x * ampl;
    
        ampl *= 0.5;
        pos *= 2.0;
    }
    
    return ind / (1.0 - pow(0.5, N+1.0)) * 0.5;
}

// Function 845
float noise(vec2 uv, float t){
    return pow(blur(uv,t),2.0);
}

// Function 846
float snoise(vec2 v
){const vec4 C=vec4(.211324865405187 //(3.-sqrt(3.))/6.
                  , .366025403784439 //sqrt(3.0)*.5-.5
                  ,-.577350269189626 //C.x*2.-1.
                  , .024390243902439)//1./41.
 ;vec2 i=floor(v+dot(v,C.yy))
 ;vec2 x0=v-i+dot(i,C.xx)
 ;vec2 ii1=(x0.x>x0.y)?vec2(1,0):vec2(0,1)
 ;vec4 x12=x0.xyxy+C.xxzz
 ;x12.xy-=ii1
 ;i=mod289(i)
 ;vec3 p=permute(permute(i.y+vec3(0,ii1.y,1))+i.x+vec3(0, ii1.x,1))
 ;vec3 m=max(.5-vec3(dd(x0),dd(x12.xy),dd(x12.zw)),0.)
 ;m*=m;m*=m
 ;// Gradients: 41 points uniformly over a line, mapped onto a diamond.
 ;// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)
 ;vec3 x=2.*fract(p*C.www)-1.
 ;vec3 h=abs(x)-.5
 ;vec3 ox=floor(x+.5)
 ;vec3 a0=x-ox
 ;//Normalise gradients implicitly by scaling m
 ;//Approximation of: m *= inversesqrt( a0*a0 + h*h )
 ;m*=1.79284291400159-.85373472095314*(a0*a0+h*h);
 ;a0=vec3(a0.x*x0.x+h.x*x0.y,a0.yz*x12.xz+h.yz*x12.yw)
 ;return 130.*dot(m,a0);}

// Function 847
float VolumeFogNoise(vec3 p)
{
    float time = iTime;
    
    p.x -= time * 220.0;
    p.z -= time * 40.0;
    p.y -= time * 25.5;
    
    return Noise3D(p * 0.002);
}

// Function 848
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel1, (uv+ 0.5)/256.0, -100.0 ).yx;
	return -1.0+2.0*mix( rg.x, rg.y, f.z );
}

// Function 849
float noise12(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);
    return 1.0 - 2.0 * mix(mix(hash12(i + vec2(0.0, 0.0)), 
                               hash12(i + vec2(1.0, 0.0)), u.x),
                           mix(hash12(i + vec2(0.0, 1.0)), 
                               hash12(i + vec2(1.0, 1.0)), u.x), u.y);
}

// Function 850
float Noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0;

    float res = mix(mix( Hash(n+  0.0), Hash(n+  1.0),f.x),
                    mix( Hash(n+ 57.0), Hash(n+ 58.0),f.x),f.y);

    return res;
}

// Function 851
float triNoise3d(in vec3 p, in float spd)
{
    float z = 1.45;
	float rz = 0.;
    vec3 bp = p;
	for (float i=0.; i<4.; i++ )
	{
        vec3 dg = tri3(bp);
        p += (dg+time*spd+10.1);
        bp *= 1.65;
		z *= 1.5;
		p *= .9;
        p.xz*= m2;
        
        rz+= (tri2(p.z+tri2(p.x+tri2(p.y))))/z;
        bp += 0.9;
	}
	return rz;
}

// Function 852
float Noise (vec2 p)
{
   	vec2 lv = fract (p);
    vec2 index = floor (p);
    
    vec2 sm = 6.0 * lv * lv * lv * lv * lv -
       	 15.0 * lv * lv * lv * lv + 
         10.0 * lv * lv * lv; //smooth function
    
    float bl = Hash (index);
    float br = Hash (index + vec2 (1.0, 0.0));
    float b = mix (bl, br, sm.x);
    float tl = Hash (index + vec2 (0.0, 1.0));
    float tr = Hash (index + vec2 (1.0, 1.0));
    float t = mix (tl, tr, sm.x);
    
    return mix (b, t, sm.y);   
}

// Function 853
float noise(in vec2 x) {
    vec2 p = floor(x);
    vec2 w = fract(x);
    
    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    
    float a = hash1(p+vec2(0,0));
    float b = hash1(p+vec2(1,0));
    float c = hash1(p+vec2(0,1));
    float d = hash1(p+vec2(1,1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;

    return k0 + k1*u.x + k2*u.y + k4*u.x*u.y;
}

// Function 854
float noise(vec2 p)
{
	vec2 ip = floor(p);
	vec2 u = fract(p);
	u = u*u*(3.0-2.0*u);
    float a = hash(ip+vec2(0.0,0.0));
    float b = hash(ip+vec2(1.0,0.0));
    float c = hash(ip+vec2(0.0,1.0));
    float d = hash(ip+vec2(1.0,1.0));
	float res = mix(mix(a,b,u.x),mix(c,d,u.x),u.y);
	return res*res;
}

// Function 855
float mfperlin1d(float x, float seed, float fmin, float fmax, float phi)
{
    float sum = 0.;
    float a = 1.;
    
    for(float f = fmin; f<fmax; f *= 2.)
    {
        sum += a*p(f*x, seed);
        a *= phi;
    }
    
    return sum;
}

// Function 856
float noise( in vec2 p ) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
	vec2 i = floor(p + (p.x+p.y)*K1);	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;
    vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot(n, vec3(70.0));	
}

// Function 857
float Mnoise( vec2 uv ) {
#  if MODE==0
    return noise(uv);                      // base turbulence
#elif MODE==1
    return -1. + 2.* (1.-abs(noise(uv)));  // flame like
#elif MODE==2
    return -1. + 2.* (abs(noise(uv)));     // cloud like
#endif
}

// Function 858
vec3 drawVoronoiLines(in vec2 p, in vec2 cellId) {
    vec2 rm = hash2(cellId); // center of cell
    vec3 d = vec3(999.);
    for(int i=0; i<8; i++) {
        vec2 g = ID_POS(i),                     // relative pos of neigbourg cell
             r = g + hash2(cellId + g),         // center of the neigbourg cell
             n = r - rm,                     
             c = rm + n*.5;                     // ref pt between cells
        n = normalize(vec2(-n.y,n.x));          // normal on the edge
        d = min(d, vec3(length(c-p),            // distance to the 
                        sdSegment(p,r,rm),      // distance to segment between cells 
                        sdSegment(p,c+n,c-n))); // distance to edge of cell
    }
    return d;
}

// Function 859
float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( random2(i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( random2(i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( random2(i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( random2(i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 860
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z);
	vec2 rg1 = textureLod( iChannel0, (uv+ vec2(0.5,0.5))/256.0, 0. ).yx;
	vec2 rg2 = textureLod( iChannel0, (uv+ vec2(1.5,0.5))/256.0, 0. ).yx;
	vec2 rg3 = textureLod( iChannel0, (uv+ vec2(0.5,1.5))/256.0, 0. ).yx;
	vec2 rg4 = textureLod( iChannel0, (uv+ vec2(1.5,1.5))/256.0, 0. ).yx;
	vec2 rg = mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
	return mix( rg.x, rg.y, f.z );
}

// Function 861
vec2 VecNoise(vec2 point)
{
    vec2 res;
    res.x = Noise(point,0.0);
    res.y = Noise(point + vec2(iTime),0.0);
    return res;
}

// Function 862
vec4 CostasNoise(ivec2 u,int iFrame,vec4 m){
 ;u/=2
 ;float r=0.;
 //last 3 entries are better set to 0, they are very small and too symmetrical costas arrays (cause diagonal bands)
 //;float f[]=float[7](1.,0.,0.,0.,0.,0.,0.)//singleton    
 ;float f[]=float[7](1.,1.,1.,1.,1.,1.,1.)//flat (strong banding)
 //;float f[]=float[7](4.,3.,2.,1.,0.,0.,0.)//valley
;float blue[]=float[7](1.,2.,4.,8.,16.,32.,0.)//blue (most banding?)
 //;float f[]=float[7](1.,2.,2.,1.,0.,0.,0.)//windowed 
 ;float yellow[]=float[7](64.,32.,16.,8.,4.,2.,0.)//anti-blue /least banding)
     //dissappointingly, even small prime tiles as small as 19*19 salready have too stron giagonal banding
     //so i guess, i just need larger tiles and larger primes.
     //i take a bet that it is a bad idea to repeat prime-gaps (espoecially short ones), which may result in the banding
 ;m=clamp(m,0.,1.)
 
 ;for(int i=0;i<7;i++){
      blue[i]=mix(blue[i],yellow[i],m.x);
 }    
;for(int i=0;i<7;i++){
      f[i]=mix(blue[i],f[i],m.y);//mix to flat
 }
 ;r+=float(Cs(u,0,iFrame))/float(gpo(0))*f[0];
 ;r+=float(Cs(u,2,iFrame))/float(gpo(2))*f[2];
 ;u=u.yx//addition, to make half of the arrays diagonally flipped
 ;r+=float(Cs(u,1,iFrame))/float(gpo(1))*f[1];
 ;r+=float(Cs(u,3,iFrame))/float(gpo(3))*f[3];
 //large above, small below
 ;r+=float(Cs(u,4,iFrame))/float(gpo(4))*f[4];    
 ;u=u.yx//addition, to make half of the arrays diagonally flipped
 ;r+=float(Cs(u,5,iFrame))/float(gpo(5))*f[5];
 ;r+=float(Cs(u,6,iFrame))/float(gpo(6))*f[6]; 
    
 ;float a=r/(f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6])//divide by sum of weights
 ;return vec4(a,a,a,1);
}

// Function 863
float movingNoise(vec2 p) {
 
    float x = fractalNoise(p + iTime);
    float y = fractalNoise(p - iTime);
    return fractalNoise(p + vec2(x, y));   
    
}

// Function 864
float aBitBetterNoise(vec4 x){
    // this trick here works for all IQ noises, it makes cost 2x but makes it look a lot lot better
    // it is not required of course
    float a = noise4d(x);
    float b = noise4d(x + .5);                      
    return (a+b)*.5;
}

// Function 865
void noiseInterpolateDu(const in vec3 x, out vec3 u, out vec3 du) 
{ 
    vec3 x2 = x * x;
    u = x2 * x * (x * (x * 6.0 - 15.0) + 10.0); 
    du = 30.0 * x2 * (x * (x - 2.0) + 1.0);
}

// Function 866
float cellnoise(vec2 p)
{
    vec2 fp=fract(p);
    vec2 ip=p-fp;
    float nd=1e3;
    vec2 nc=p;
    for(int i=-1;i<2;i+=1)
        for(int j=-1;j<2;j+=1)
        {
            vec2 c=ip+vec2(i,j)+vec2(noise(ip+vec2(i,j)),noise(ip+vec2(i+10,j)));
            float d=distance(c,p);
            if(d<nd)
            {
                nd=d;
                nc=c;
            }
        }

    return nd;
}

// Function 867
float noise (in vec2 p)
{
    const float K1 = .366025404;
    const float K2 = .211324865;

    vec2 i = floor (p + (p.x + p.y)*K1);
    
    vec2 a = p - i + (i.x + i.y)*K2;
    vec2 o = step (a.yx, a.xy);    
    vec2 b = a - o + K2; 
    vec2 c = a - 1. + 2.*K2;

    vec3 h = max (.5 - vec3 (dot (a, a), dot (b, b), dot (c, c) ), .0);

    vec3 n = h*h*h*h*vec3 (dot (a, hash (i + .0)),
                           dot (b, hash (i + o)),
                           dot (c, hash (i + 1.)));

    return dot (n, vec3 (70.));
}

// Function 868
float noise(vec2 co, float frequency)
{
  vec2 v = vec2(co.x * frequency, co.y * frequency);

  float ix1 = floor(v.x);
  float iy1 = floor(v.y);
  float ix2 = floor(v.x + 1.0);
  float iy2 = floor(v.y + 1.0);

  float fx = hermite(fract(v.x));
  float fy = hermite(fract(v.y));

  float fade1 = mix(rand(vec2(ix1, iy1)), rand(vec2(ix2, iy1)), fx);
  float fade2 = mix(rand(vec2(ix1, iy2)), rand(vec2(ix2, iy2)), fx);

  return mix(fade1, fade2, fy);
}

// Function 869
float noiseMod(vec2 u,vec2 s){u*=s;vec2 f=fract(u);u=floor(u);	f=f*f*(3.-2.*f);//any interpolation is fine for f
	return mix(mix(h12s(mod(u          ,s)),h12s(mod(u+vec2(1,0),s)),f.x),
					    mix(h12s(mod(u+vec2(0,1),s)),h12s(mod(u+vec2(1,1),s)),f.x),f.y);}

// Function 870
float simplexTruchet(in vec3 p)
{
    
    // Breaking space into tetrahedra and obtaining the four verticies. The folowing three code lines
    // are pretty standard, and are used for all kinds of things, including 3D simplex noise. In this
    // case though, we're constructing tetrahedral Truchet tiles.
    
    // Skewing the cubic grid, then determining relative fractional position.
    vec3 i = floor(p + dot(p, vec3(1./3.)));  p -= i - dot(i, vec3(1./6.)) ;
    
    // Breaking the skewed cube into tetrahedra with partitioning planes, then determining which side of 
    // the intersecting planes the skewed point is on. Ie: Determining which tetrahedron the point is in.
    vec3 i1 = step(p.yzx, p), i2 = max(i1, 1. - i1.zxy); i1 = min(i1, 1. - i1.zxy);    
    
    
    // Using the above to produce the four vertices for the tetrahedron.
    vec3 p0 = vec3(0), p1 = i1 - 1./6., p2 = i2 - 1./3., p3 = vec3(.5);

    
    
    // Using the verticies to produce a unit random value for the tetrahedron, which in turn is used 
    // to determine its rotation.
    float rnd = hash31(i*57.31 + i1*41.57 + i2*27.93);
    
    // This is a cheap way (there might be cheaper, though) to rotate the tetrahedron. Basically, we're
    // rotating the vertices themselves, depending on the random number generated.
    vec3 t0 = p1, t1 = p2, t2 = p3, t3 = p0;
    if (rnd > .66){ t0 = p2, t1 = p3; t2 = p0; t3 = p1; }
    else if (rnd > .33){ t0 = p3, t1 = p0; t2 = p1; t3 = p2; } 
    

    
    // Threading two torus segments through each pair of faces on the tetrahedron.
    
    // Used to hold the distance field values for the tori segments and the bolts.
    // v.xy holds the first torus and bolt values, and v.zw hold the same for the second torus.
    vec4 v;
    
    // Axial point of the torus segment, and the normal from which the orthonormal bais is derived.
    vec3 q, bn; 
 
    
    // I remember reasoning that the outer torus radius had to be this factor (sqrt(6)/8), but I 
    // can't for the life of me remember why. A lot of tetrahedral lengths involve root six. I 
    // think it's equal to the tetrahedral circumradius... I'm not happy with that explanation either, 
    // so I'll provide a proper explanation later. :D Either way, it's the only value that fits.
    float rad = .306186218; // Equal to sqrt(6)/8.
    float rad2 = .025; // The smaller cross-sectional torus radius.


    // Positioning the center of each torus at the corresponding edge mid-point, then aligning with
    // the direction of the edge running through the midpoint. One of the ways to align an object is to
    // determine a face normal, construct an orthonormal basis from it, then multiply the object by it
    // relative to its position. On a side note, orientation could probably be achieved with a few 
    // matrix rotations instead, which may or may not be cheaper, so I'll look into it later.
    
    // First torus. Centered on the line between verticies t0 and t1, and aligned to the face that
    // the edge runs through.
    bn = (t0 - t1)*1.1547005; // Equivalent to normalize(t0 - t1);
    q = basis(bn)*(p - mix(t0, t1, .5)); // Applying Nimitz's basis formula to the point to realign it.
    v.xy = tor(q, rad, rad2); // Obtain the first torus distance.

    // Second torus. Centered on the line between verticies t2 and t3, and aligned to the face that
    // the edge runs through.
    bn = (t2 - t3)*1.1547005; // Equivalent to normalize(t2 - t3);
    q = basis(bn)*(p - mix(t2, t3, .5)); // Applying Nimitz's basis formula to the point to realign it.
    v.zw = tor(q, rad, rad2); // Obtain the second torus distance.

    // Determine the minium torus value, v.x, and the minimum bolt value, v.y.
    v.xy = min(v.xy, v.zw);
    
 
    // Object ID. It's either the ribbed torus itself or a bolt.
    objID = step(v.x, v.y);

    // Return the minimum surface point.
    return min(v.x, v.y);
    
    
}

// Function 871
float noise(in float _x) {
    return mix(rand_step(_x-1.), rand_step(_x), smoothstep(0.,1.,fract(_x)));
}

// Function 872
float noise( vec3 x ) { const vec3 step = vec3( 110, 241, 171 ); vec3 i = floor( x ); vec3 f = fract( x ); float n = dot( i, step ); vec3 u = f * f * ( 3.0 - 2.0 * f ); return mix( mix( mix( hash( n + dot( step, vec3( 0, 0, 0 ) ) ), hash( n + dot( step, vec3( 1, 0, 0 ) ) ), u.x ), mix( hash( n + dot( step, vec3( 0, 1, 0 ) ) ), hash( n + dot( step, vec3( 1, 1, 0 ) ) ), u.x ), u.y ), mix( mix( hash( n + dot( step, vec3( 0, 0, 1 ) ) ), hash( n + dot( step, vec3( 1, 0, 1 ) ) ), u.x ), mix( hash( n + dot( step, vec3( 0, 1, 1) ) ), hash( n + dot( step, vec3( 1, 1, 1 ) ) ), u.x ), u.y ), u.z ); }

// Function 873
float voronoi(vec2 u, float i) {
	#define l(i) length(fract(abs(u*.01+fract(i*vec2(1,8))+sin(u.yx*.2+i*8.)*.02+sin(u*.06+1.6+i*6.)*.1))-.5)
	return l(i);
}

// Function 874
float noise(vec2 p,float r2,float z 
){vec2 q=skew(p),r=fract(q);q=floor(q);return 
  noise(p,r2,z,q)
+noise(p,r2,z,q+1.)
+noise(p,r2,z,q+mix(vec2(0,1),vec2(1,0),vec2(step(r.y,r.x))));}

// Function 875
float cloudNoise2D(vec2 uv, vec2 _wind)
{
    float v = 1.0-voronoi2D(uv*10.0+_wind);
    float fs = fbm2Dsimple(uv*20.0+_wind);
    return clamp(v*fs, 0.0, 1.0);
}

// Function 876
float simpleNoise(vec2 p)
{
    float s = sin(p.x), c = cos(p.y);
    p = mat2(c, -s, s, c)*p;
    vec2 f = floor(p);
    vec2 f_ = smoothstep(0.0, 1.0, fract(p));
    float x = p.x+13.0*p.y;
    
    return mix(
        	mix(hash1D(x+0.0), hash1D(x+13.0), f.x),
        	mix(hash1D(x+13.0), hash1D(x+42.0), f.x),
        	f.y);
}

// Function 877
float noise(in vec2 p) {
	vec2 e=vec2(1.,0.), F = floor(p), f = fract(p), k = (3. - 2.*f) * f * f;
	return mix(mix(hash(F),      hash(F+e.xy), k.x),
			   mix(hash(F+e.yx), hash(F+e.xx), k.x), k.y);
}

// Function 878
float perlin_noise(vec2 pos)
{
    float x = pos.x / 32.0;
    float y = pos.y / 32.0;
    
	float value = 0.0;
    float x0 = floor(x);    
    float y0 = floor(y);
	float x1 = x0 + 1.0;	
    float y1 = y0 + 1.0;
    
    vec2 a = GenerateGradientVector(x0, y0);
    vec2 b = GenerateGradientVector(x1, y0);
    vec2 c = GenerateGradientVector(x0, y1);
    vec2 d = GenerateGradientVector(x1, y1);
    
    vec2 aD = vec2(x - x0, y - y0);    
    vec2 bD = vec2(x - x1, y - y0);
    vec2 cD = vec2(x - x0, y - y1);
    vec2 dD = vec2(x - x1, y - y1);
    
    float aDot = dot(a, aD);
    float bDot = dot(b, bD);
    float cDot = dot(c, cD);
    float dDot = dot(d, dD);
    
    float dx = x - x0;    
    float dy = y - y0;
	float sX = 3.0 * dx * dx - 2.0 * dx * dx * dx;
	float sY = 3.0 * dy * dy - 2.0 * dy * dy * dy;
    float firstAverage = mix(aDot, bDot, sX);
    float secondAverage = mix(cDot, dDot, sX);
    value = mix(firstAverage, secondAverage, y - y0);
	
	return ( value + 1.0 ) / 2.0;
}

// Function 879
vec4 noiseVcDx(vec3 x){vec3 p=floor(x),w=fract(x)   
#if 1
 ;vec3 u=w*w*w*(w*(w*6.-15.)+10.)// quintic hermite interpolation (3*1*1 binomial)  
 ,n=30.*w*w*(w*(w-2.)+1.)
//higher degree hermites can have FASTER gradient descents
//for newer hardware with longer pipelines, with more iterations.
#else
 ;vec3 u=w*w*(3.-2.*w)//cubic hermite interpolation (2*1 binomial)
 ,n=6.*w*(1.-w)//likely faster on older hardware, with less iterations
#endif   
;float a=hash(p+vec3(0))//smallest corner coefficient
,      h=hash(p+vec3(1))//largest  corner coefficient
;vec3 b=hash(   mat3(1)+mat3(p,p,p))//3 corners that are adjacent to smallest corner
,     c=hash(1.-mat3(1)+mat3(p,p,p))//3 corners that are adjacent to largest  corner//8 cube corners
,k=b-a,l=a-b.xyx-b.yzz+c.zxy,v=((l*u).zxy+k)*u//+9 coefficients in 3 vectors.
;b-=c;h+=b.x+b.y+b.z-a//modify b to modify h
;return vec4(a+v.x+v.y+v.z+h*u.x*u.y*u.z, n*(l*u.yzx+l.zxy*u.zxy+h*u.yzx*u.zxy+k) );}

// Function 880
float noise (in vec3 x)
{
	vec3 p = floor(x);
	vec3 f = fract(x);

	f = f*f*(3.0-2.0*f);

	float n = p.x + p.y*57.0 + 113.0*p.z;

	float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
						mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
					mix(mix( hash(n+113.0), hash(n+114.0),f.x),
						mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
	return res;
}

// Function 881
vec3 noise3( in vec3 x)
{
	return vec3( noise(x+vec3(123.456,.567,.37)),
				noise(x+vec3(.11,47.43,19.17)),
				noise(x) );
}

// Function 882
vec3 fbmdPerlin(vec2 pos, vec2 scale, int octaves, vec2 shift, mat2 transform, float gain, vec2 lacunarity, float slopeness, float octaveFactor, bool negative, float seed) 
{
    // fbm implementation based on Inigo Quilez
    float amplitude = gain;
    vec2 frequency = floor(scale);
    vec2 p = pos * frequency;
    octaveFactor = 1.0 + octaveFactor * 0.3;

    vec3 value = vec3(0.0);
    vec2 derivative = vec2(0.0);
    for (int i = 0; i < octaves; i++) 
    {
        vec3 n = perlinNoised(p / frequency, frequency, transform, seed);
        derivative += n.yz;
        n.x = negative ? n.x : n.x * 0.5 + 0.5;
        n *= amplitude;
        value.x += n.x / (1.0 + mix(0.0, dot(derivative, derivative), slopeness));
        value.yz += n.yz; 
        
        p = (p + shift) * lacunarity;
        frequency *= lacunarity;
        amplitude = pow(amplitude * gain, octaveFactor);
        transform *= transform;
    }

    return clamp(value,-1.,1.);
}

// Function 883
float noise(vec2 p) {
	vec2 i = ceil(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3. - 2. * f);
   	float a = random(i);
    float b = random(i + vec2(1., 0.));
    float c = random(i + vec2(0., 1.));
    float d = random(i + vec2(1., 1.));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Function 884
vec2 voronoi(in vec2 st)
{
    vec2 id = floor(st);
    vec2 gv = fract(st);
    vec3 mx = vec3(8.0);
    vec2 p = vec2(0.);

    for(int y =- 1; y <= 1; y ++ )
    for(int x =- 1; x <= 1; x ++ )
    {
        vec2 offset = vec2(float(x), float(y));
        vec2 p = N23(id + offset);
        vec2 r = offset - gv ;
        r += (.5 + .5 * sin(iTime*5. + 6.2831 * p));
        float d = dot(r, r);

        if (d < mx.x)
        mx = vec3(d, p);

    }

    // balls molecules?
    return vec2(sqrt(mx.x)*VRN_BALL, length(mx.yz)); //balls
    
    // germs?
    //return vec2(sqrt(mx.x)*VRN_BALL, length(p)); // germs
    
    // sun hydrogen fusion-fission?
    //return vec2(sqrt(mx.x)*VRN_BALL, log(1. + .1*dot(mx.xz,mx.xz)));

}

// Function 885
float noise( in vec3 x )
{
    x *= 5.0;
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0. ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 886
float Voronoi(in vec2 p){
    
	vec2 g = floor(p), o; p -= g;
	
    // I'm not sure what the largest conceivable closest squared-distance would be, but I think 
    // values as high as 2.5 (center to non-diagonal outer square corner) are possible. Statistically, 
    // it's unlikely, so when scaling back the final value to the zero-to-one range, I divide by 
    // something less than my maximum and cap it to one... It's all a matter of what look you're 
    // trying to achieve.
	vec3 d = vec3(2.5);
    
	for(int y=-1; y<=1; y++){
		for(int x=-1; x<=1; x++){
            
			o = vec2(x, y);
            o += hash22(g + o) - p;
            
			d.z = dot(o, o);
            d.y = max(d.x, min(d.y, d.z));
            d.x = min(d.x, d.z); 
                       
		}
	}

    // Final value, with rough scaling.
    return min((d.y - d.x)*.6, 1.); // Scale: [0, 1].
    
}

// Function 887
vec2 fluidnoise(vec3 p) {
    vec2 total = vec2(0);
    float amp = 1.;
    for(int i = 0; i < 1; i++) {
        total += noise(p) * amp;
        p = p*2. + 4.3; amp *= 1.5;
    }
    return total.yx * vec2(-1,1); // divergence-free field
}

// Function 888
float Noise( float x )
{
	return Hash( floor(x * 32.0) ) * 2.0 - 1.0;
}

// Function 889
float noise1( sampler3D tex, in vec3 x )
{
    return textureLod(tex,(x+0.5)/32.0,0.0).x;
}

// Function 890
float noise(in vec3 x) {
    vec3 p = floor(x), f = fract(x);
	f = f*f*(3.-2.*f);
	vec2 rg = textureLod(iChannel0, (((p.xy+vec2(37.,17.)*p.z) + f.xy)+.5)/256., -100.).yx;
	return mix(rg.x, rg.y, f.z);
}

// Function 891
float perlin2(vec2 x)
{
	return (
         16.0 * texture(iChannel0, x * 0.3) + 
         6.0 * texture(iChannel0, x) + 
         3.0 * texture(iChannel0, x * 3.0) + 
         0.3 * texture(iChannel0, x * 12.0) + 
         0.2 * texture(iChannel0, x * 25.0)).x * 0.1;
}

// Function 892
float simplegridnoise(vec2 v)
{
    float s = 1. / 256.;
    vec2 fl = floor(v), fr = fract(v);
    float mindist = 1e9;
    for(int y = -1; y <= 1; y++)
        for(int x = -1; x <= 1; x++)
        {
            vec2 offset = vec2(x, y);
            vec2 pos = .5 + .5 * cos(2. * pi * (T*.1 + hash(fl+offset)) + vec2(0,1.6));
            mindist = min(mindist, length(pos+offset -fr));
        }
    
    return mindist;
}

// Function 893
vec2 Voronoi( in vec2 x )
{
	vec2 p = floor( x );
	vec2 f = fract( x );
	float res=100.0,id;
	for( int j=-1; j<=1; j++ )
	for( int i=-1; i<=1; i++ )
	{
		vec2 b = vec2( float(i), float(j) );
		vec2 r = b - f  + Hash( p + b );
		float d = dot(r,r);
		if( d < res )
		{
			res = d;
			id  = Hash(p+b);
		}			
    }
	return vec2(max(.4-sqrt(res), 0.0),id);
}

// Function 894
float perlin_noise(vec2 p)
{
    vec2 pi = floor(p);
    vec2 pf = p - pi;
    
    vec2 w = pf * pf * (3.0 - 2.0 * pf);
    
    return mix(mix(dot(hash22(pi + vec2(0.0, 0.0)), pf - vec2(0.0, 0.0)), 
                   dot(hash22(pi + vec2(1.0, 0.0)), pf - vec2(1.0, 0.0)), w.x), 
               mix(dot(hash22(pi + vec2(0.0, 1.0)), pf - vec2(0.0, 1.0)), 
                   dot(hash22(pi + vec2(1.0, 1.0)), pf - vec2(1.0, 1.0)), w.x),
               w.y);
}

// Function 895
float Noisefv2 (vec2 p)
{
  vec4 t;
  vec2 ip, fp;
  ip = floor (p);  fp = fract (p);
  fp = fp * fp * (3. - 2. * fp);
  t = Hashv4f (dot (ip, cHashA3.xy));
  return mix (mix (t.x, t.y, fp.x), mix (t.z, t.w, fp.x), fp.y);
}

// Function 896
float noise2D( in vec2 pos , float lod)
{
  vec2 f = fract(pos);
  f = f*f*(3.0-2.0*f);
  vec2 rg = textureLod( iChannel1, (((floor(pos).xy+vec2(37.0, 17.0)) + f.xy)+ 0.5)/256.0, lod).yx;  
  return -1.0+2.0*mix( rg.x, rg.y, 0.5 );
}

// Function 897
float noiseValue2D(vec2 p, float s, float speed)
{
    p *= s;
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * f * (f * (6.0 * f - 15.0) + 10.0);
    float r1 = anim(hash12(i + vec2(0.0, 0.0)), speed);
    float r2 = anim(hash12(i + vec2(1.0, 0.0)), speed);
    float r3 = anim(hash12(i + vec2(0.0, 1.0)), speed);
    float r4 = anim(hash12(i + vec2(1.0, 1.0)), speed);
    return mix(mix(r1, r2, f.x), mix(r3, r4, f.x), f.y);
}

// Function 898
vec2 Noise22(vec2 x)
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    vec2 res = mix(mix( hash22(p),          hash22(p + add.xy),f.x),
                   mix( hash22(p + add.yx), hash22(p + add.xx),f.x),f.y);
    return res-.5;
}

// Function 899
float snoise(float t){float f=fract(t);t=floor(t);return mix(rand(t),rand(t+1.0),f*f*(3.0-2.0*f));}

// Function 900
vec4 snoise( in vec3 uv ) {
    const float textureResolution = 32.0;
	uv = uv*textureResolution + 0.5;
	vec3 iuv = floor( uv );
	vec3 fuv = fract( uv );
	uv = iuv + fuv*fuv*(3.0-2.0*fuv);
	uv = (uv - 0.5)/textureResolution;
	return texture( iChannel3, uv );
}

// Function 901
vec4 noise(vec2 u){u=mod(u,M.x)+M.y; // mod() avoids grid glitch
 u-=.5;return vec4(hash(uint(u.x+iResolution.x*u.y)));}

// Function 902
float tetraNoise(in vec3 p)
{
    // Skewing the cubic grid, then determining the first vertice and fractional position.
    vec3 i = floor(p + dot(p, vec3(0.333333)) );  p -= i - dot(i, vec3(0.166666)) ;
    
    // Breaking the skewed cube into tetrahedra with partitioning planes, then determining which side of the 
    // intersecting planes the skewed point is on. Ie: Determining which tetrahedron the point is in.
    vec3 i1 = step(p.yzx, p), i2 = max(i1, 1.0-i1.zxy); i1 = min(i1, 1.0-i1.zxy);    
    
    // Using the above to calculate the other three vertices. Now we have all four tetrahedral vertices.
    // Technically, these are the vectors from "p" to the vertices, but you know what I mean. :)
    vec3 p1 = p - i1 + 0.166666, p2 = p - i2 + 0.333333, p3 = p - 0.5;
  

    // 3D simplex falloff - based on the squared distance from the fractional position "p" within the 
    // tetrahedron to the four vertice points of the tetrahedron. 
    vec4 v = max(0.5 - vec4(dot(p,p), dot(p1,p1), dot(p2,p2), dot(p3,p3)), 0.0);
    
    // Dotting the fractional position with a random vector generated for each corner -in order to determine 
    // the weighted contribution distribution... Kind of. Just for the record, you can do a non-gradient, value 
    // version that works almost as well.
    vec4 d = vec4(dot(p, hash33(i)), dot(p1, hash33(i + i1)), dot(p2, hash33(i + i2)), dot(p3, hash33(i + 1.)));
    
     
    // Simplex noise... Not really, but close enough. :)
    return clamp(dot(d, v*v*v*8.)*1.732 + .5, 0., 1.); // Not sure if clamping is necessary. Might be overkill.

}

// Function 903
float noise(  vec2 p )
{
    vec2 i = floor( p ), f = fract( p ), u = f*f*(3.-2.*f);
    return mix( mix( dot( hash( i + vec2(0,0) ), f - vec2(0,0) ), 
                     dot( hash( i + vec2(1,0) ), f - vec2(1,0) ), u.x),
                mix( dot( hash( i + vec2(0,1) ), f - vec2(0,1) ), 
                     dot( hash( i + vec2(1,1) ), f - vec2(1,1) ), u.x), u.y);
}

// Function 904
float perlin_noise(vec2 pos)
{
	float	n;
	
	n = interpolate_noise(pos*0.0625)*0.5;
	n += interpolate_noise(pos*0.125)*0.25;
	n += interpolate_noise(pos*0.025)*0.225;
	n += interpolate_noise(pos*0.05)*0.0625;
	n += interpolate_noise(pos)*0.03125;
	return n;
}

// Function 905
float dnoise(vec3 v)
{
    return snoise(v) > 0.0 ? 1.0 : 0.0;
}

// Function 906
vec3 curlnoise(in vec3 p, int seed)
{
    vec4 x = gnoised(p, seed);
    vec4 y = gnoised(p, seed+3);
    vec4 z = gnoised(p, seed+6);
    return vec3(z.z - x.w, x.w - z.y, y.y - x.z);
}

// Function 907
float noise(vec2 p) {
    vec2 pi = floor(p);
    vec2 pf = fract(p);

    float r00 = rand(vec2(pi.x    ,pi.y    ));
    float r10 = rand(vec2(pi.x+1.0,pi.y    ));
    float r01 = rand(vec2(pi.x    ,pi.y+1.0));
    float r11 = rand(vec2(pi.x+1.0,pi.y+1.0));

    vec2 m = pf*pf*(3.0-2.0*pf);
    return mix(mix(r00, r10, m.x), mix(r01, r11, m.x), m.y);
}

// Function 908
float noise (in vec3 p) {
	vec3 f = fract (p);
	p = floor (p);
	f = f * f * (3.0 - 2.0 * f);
	f.xy += p.xy + p.z * vec2 (37.0, 17.0);
	f.xy = texture (iChannel0, (f.xy + 0.5) / 256.0, -256.0).yx;
	return mix (f.x, f.y, f.z);
}

// Function 909
float noiseValue(vec3 uv)
{
    vec3 fr = fract(uv.xyz);
    vec3 fl = floor(uv.xyz);
    float h000 = Hash3d(fl);
    float h100 = Hash3d(fl + zeroOne.yxx);
    float h010 = Hash3d(fl + zeroOne.xyx);
    float h110 = Hash3d(fl + zeroOne.yyx);
    float h001 = Hash3d(fl + zeroOne.xxy);
    float h101 = Hash3d(fl + zeroOne.yxy);
    float h011 = Hash3d(fl + zeroOne.xyy);
    float h111 = Hash3d(fl + zeroOne.yyy);
    return mixP(
        mixP(mixP(h000, h100, fr.x),
             mixP(h010, h110, fr.x), fr.y),
        mixP(mixP(h001, h101, fr.x),
             mixP(h011, h111, fr.x), fr.y)
        , fr.z);
}

// Function 910
float voronoi( in vec2 x )
{
    float t = iTime/SWITCH_TIME;
	float function 			= mod(t,4.0);
    bool  multiply_by_F1	= mod(t,8.0)  >= 4.0;
	bool  inverse			= mod(t,16.0) >= 8.0;
	float distance_type		= mod(t/16.0,4.0);
    
    vec2 n = floor( x );
    vec2 f = fract( x );

	float F1 = 8.0;
	float F2 = 8.0;
	
	
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(i,j);
        vec2 o = hash( n + g );

        o = 0.5 + 0.41*sin( iTime + 6.2831*o ); // animate

		vec2 r = g - f + o;

		float d = 	distance_type < 1.0 ? dot(r,r)  :				// euclidean^2
				  	distance_type < 2.0 ? sqrt(dot(r,r)) :			// euclidean
					distance_type < 3.0 ? abs(r.x) + abs(r.y) :		// manhattan
					distance_type < 4.0 ? max(abs(r.x), abs(r.y)) :	// chebyshev
					0.0;

		if( d<F1 ) 
		{ 
			F2 = F1; 
			F1 = d; 
		}
		else if( d<F2 ) 
		{
			F2 = d;
		}
    }
    
	
	float c = function < 1.0 ? F1 : 
			  function < 2.0 ? F2 : 
			  function < 3.0 ? F2-F1 :
			  function < 4.0 ? (F1+F2)/2.0 : 
			  0.0;
		
	if( multiply_by_F1 )	c *= F1;
	if( inverse )			c = 1.0 - c;
	
    return c;
}

// Function 911
float noiseSpere(vec3 ray,vec3 pos,float r,mat3 mr,float zoom,vec3 subnoise,float anim)
{
  	float b = dot(ray,pos);
  	float c = dot(pos,pos) - b*b;
    
    vec3 r1=vec3(0.0);
    
    float s=0.0;
    float d=0.03125;
    float d2=zoom/(d*d); 
    float ar=5.0;
   
    for (int i=0;i<3;i++) {
		float rq=r*r;
        if(c <rq)
        {
            float l1=sqrt(rq-c);
            r1= ray*(b-l1)-pos;
            r1=r1*mr;
            s+=abs(noise4q(vec4(r1*d2+subnoise*ar,anim*ar))*d);
        }
        ar-=2.0;
        d*=4.0;
        d2*=0.0625;
        r=r-r*0.02;
    }
    return s;
}

// Function 912
float noise4q(vec4 x)
{
	vec4 n3 = vec4(0,0.25,0.5,0.75);
	vec4 p2 = floor(x.wwww+n3);
	vec4 b = floor(x.xxxx+n3) + floor(x.yyyy+n3)*157.0 + floor(x.zzzz +n3)*113.0;
	vec4 p1 = b + fract(p2*0.00390625)*vec4(164352.0, -164352.0, 163840.0, -163840.0);
	p2 = b + fract((p2+1.0)*0.00390625)*vec4(164352.0, -164352.0, 163840.0, -163840.0);
	vec4 f1 = fract(x.xxxx+n3);
	vec4 f2 = fract(x.yyyy+n3);
	f1=f1*f1*(3.0-2.0*f1);
	f2=f2*f2*(3.0-2.0*f2);
	vec4 n1 = vec4(0,1.0,157.0,158.0);
	vec4 n2 = vec4(113.0,114.0,270.0,271.0);	
	vec4 vs1 = mix(hash4(p1), hash4(n1.yyyy+p1), f1);
	vec4 vs2 = mix(hash4(n1.zzzz+p1), hash4(n1.wwww+p1), f1);
	vec4 vs3 = mix(hash4(p2), hash4(n1.yyyy+p2), f1);
	vec4 vs4 = mix(hash4(n1.zzzz+p2), hash4(n1.wwww+p2), f1);	
	vs1 = mix(vs1, vs2, f2);
	vs3 = mix(vs3, vs4, f2);
	vs2 = mix(hash4(n2.xxxx+p1), hash4(n2.yyyy+p1), f1);
	vs4 = mix(hash4(n2.zzzz+p1), hash4(n2.wwww+p1), f1);
	vs2 = mix(vs2, vs4, f2);
	vs4 = mix(hash4(n2.xxxx+p2), hash4(n2.yyyy+p2), f1);
	vec4 vs5 = mix(hash4(n2.zzzz+p2), hash4(n2.wwww+p2), f1);
	vs4 = mix(vs4, vs5, f2);
	f1 = fract(x.zzzz+n3);
	f2 = fract(x.wwww+n3);
	f1=f1*f1*(3.0-2.0*f1);
	f2=f2*f2*(3.0-2.0*f2);
	vs1 = mix(vs1, vs2, f1);
	vs3 = mix(vs3, vs4, f1);
	vs1 = mix(vs1, vs3, f2);
	float r=dot(vs1,vec4(0.25));
	//r=r*r*(3.0-2.0*r);
	return r*r*(3.0-2.0*r);
}

// Function 913
void lfnoise(in vec2 t, out float n)
{
    vec2 i = floor(t);
    t = fract(t);
    t = smoothstep(c.yy, c.xx, t);
    vec2 v1, v2;
    rand(i, v1.x);
    rand(i+c.xy, v1.y);
    rand(i+c.yx, v2.x);
    rand(i+c.xx, v2.y);
    v1 = c.zz+2.*mix(v1, v2, t.y);
    n = mix(v1.x, v1.y, t.x);
}

// Function 914
vec3 Voronoi(vec2 p){
    
    // Convert to the hexagonal grid.
    vec2 pH = pixToHex(p); // Map the pixel to the hex grid.

    // There'd be a heap of ways to get rid of this array and speed things up. The
    // most obvious, would be unrolling the loops, but there'd be more elegant ways.
    // Either way, I've left it this way just to make the code easier to read.
 
    // Hexagonal grid offsets. "vec2(0)" represents the center, and the other offsets effectively circle it.
    // Thanks, Abje. Hopefully, the compiler will know what to do with this. :)
	const vec2 hp[7] = vec2[7](vec2(-1), vec2(0, -1), vec2(-1, 0), vec2(0), vec2(1), vec2(1, 0), vec2(0, 1)); 
    
    
    // Voronoi cell ID containing the minimum offset point distance. The nearest
    // edge will be one of the cells edges.
    vec2 minCellID = vec2(0); // Redundant initialization, but I've done it anyway.

    // As IQ has commented, this is a regular Voronoi pass, so it should be
    // pretty self explanatory.
    //
    // First pass: Regular Voronoi.
	vec2 mo, o;
    
    // Minimum distance, "smooth" distance to the nearest cell edge, regular
    // distance to the nearest cell edge, and a line distance place holder.
    float md = 8., lMd = 8., lMd2 = 8., lnDist, d;
    
    for (int i=0; i<7; i++){
    
        // Determine the offset hexagonal point.
        vec2 h = hexPt(pH + hp[i]) - p;
        // Determine the distance metric to the point.
    	d = dot(h, h);
    	if( d<md ){ // Perform updates, if applicable.
            
            md = d;  // Update the minimum distance.
            // Keep note of the position of the nearest cell point - with respect
            // to "p," of course. It will be used in the second pass.
            mo = h; 
            //cellID = h + p; // For cell coloring.
            minCellID = hp[i]; // Record the minimum distance cell ID.
        }
    }
    
    // Second pass: Point to nearest cell-edge distance.
    //
    // With the ID of the cell containing the closest point, do a sweep of all the
    // surrounding cell edges to determine the closest one. You do that by applying
    // a standard distance to a line formula.
    for (int i=0; i<7; i++){
    
         // Determine the offset hexagonal point in relation to the minimum cell offset.
        vec2 h = hexPt(pH + hp[i] + minCellID) - p - mo; // Note the "-mo" to save some operations. 
        
        // Skip the same cell.
        if(dot(h, h)>.00001){
            
            // This tiny line is the crux of the whole example, believe it or not. Basically, it's
            // a bit of simple trigonometry to determine the distance from the cell point to the
            // cell border line. See IQ's article (link above) for a visual representation.            
            lnDist = dot(mo + h*.5, normalize(h));
            
            // Abje's addition. Border distance using a smooth minimum. Insightful, and simple.
            //
            // On a side note, IQ reminded me that the order in which the polynomial-based smooth
            // minimum is applied effects the result. However, the exponentional-based smooth
            // minimum is associative and commutative, so is more correct. In this particular case, 
            // the effects appear to be negligible, so I'm sticking with the cheaper polynomial-based
            // smooth minimum, but it's something you should keep in mind. By the way, feel free to 
            // uncomment the exponential one and try it out to see if you notice a difference.
            //
            // Polynomial-based smooth minimum. The last factor controls the roundness of the 
            // edge joins. Zero gives you sharp joins, and something like ".25" will produce a
            // more rounded look.
            lMd = smin(lMd, lnDist, .1);
            // Exponential-based smooth minimum.
            //lMd = sminExp(lMd, lnDist, 20.); 
            
            // Minimum regular straight-edged border distance. If you only used this distance,
            // the web lattice would have sharp edges.
            lMd2 = min(lMd2, lnDist);
            
        }

    }

    // Return the smoothed and unsmoothed distance. I think they need capping at zero... but I'm not 
    // positive. Although not used here, the standard minimum point distance is returned also.
    return max(vec3(lMd, lMd2, md), 0.);
    
    
}

// Function 915
float snoise(vec3 p) {
	 /* 1. find current tetrahedron T and it's four vertices */
	 /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
	 /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/
	 
	 /* calculate s and x */
	 vec3 s = floor(p + dot(p, vec3(F3)));
	 vec3 x = p - s + dot(s, vec3(G3));
	 
	 /* calculate i1 and i2 */
	 vec3 e = step(vec3(0.0), x - x.yzx);
	 vec3 i1 = e*(1.0 - e.zxy);
	 vec3 i2 = 1.0 - e.zxy*(1.0 - e);
	 	
	 /* x1, x2, x3 */
	 vec3 x1 = x - i1 + G3;
	 vec3 x2 = x - i2 + 2.0*G3;
	 vec3 x3 = x - 1.0 + 3.0*G3;
	 
	 /* 2. find four surflets and store them in d */
	 vec4 w, d;
	 
	 /* calculate surflet weights */
	 w.x = dot(x, x);
	 w.y = dot(x1, x1);
	 w.z = dot(x2, x2);
	 w.w = dot(x3, x3);
	 
	 /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
	 w = max(0.6 - w, 0.0);
	 
	 /* calculate surflet components */
	 d.x = dot(random3(s), x);
	 d.y = dot(random3(s + i1), x1);
	 d.z = dot(random3(s + i2), x2);
	 d.w = dot(random3(s + 1.0), x3);
	 
	 /* multiply d by w^4 */
	 w *= w;
	 w *= w;
	 d *= w;
	 
	 /* 3. return the sum of the four surflets */
	 return dot(d, vec4(52.0));
}

// Function 916
vec4 voronoi_column_trace(
         vec3 ray_pos,
         vec3 ray_dir,
         float max_h,
         out vec4 hit_pos,
         out vec3 hit_norm,
         out vec3 hit_dh )
{
   const int iter = 32;
   
   vec2 p = ray_pos.xy;
   float s = 1./length(ray_dir.xy);
   vec2 dir = ray_dir.xy*s;
   vec2 n = floor(p);
   vec2 f = fract(p);
   
   vec4 mc;
   float md;

   // Find closest Voronoi cell to ray starting position.
   md = 8.;

   // This is fast. approx of closest point search,
   // to make it error free in all possible cases,
   // we would have to use 4x4 scan.
   for( int j=-1; j<=1; j++ )
   for( int i=-1; i<=1; i++ )
   {
      vec2 g = vec2(float(i),float(j));
      vec2 o = hash22( n + g );
      vec2 r = g + o - f;
      float d = dot(r,r);

      if( d<md )
      {
         md = d;
         mc = vec4(r, g);
      }
   }
   
   vec2 mdr = vec2(0,1);
   float dh = 0.;
   float prev_h = 0.;
   float h = 0.;
   
   md = eps;

   for( int k=0; k<iter; ++k )
   {
      // Get height of the column
      h = hash12( mc.zw + n )*max_h;
      if (ray_dir.z >= 0.) {
         dh = ray_pos.z + ray_dir.z*md;
         if (dh < h || dh > max_h) break; // ray goes to inifnity or hits
      }
      
      vec4 kc;
      vec2 kdr;
      float kd = find_neighbour(n, f, dir, mc, kc, kdr)*s;
      
      if (ray_dir.z < 0.) {
         dh = ray_pos.z + ray_dir.z*kd;
         if (dh < h) break; // hit!
      }
      
      mc = kc;
      md = kd;
      mdr = kdr;
      prev_h = h;
   }
   
   if (dh >= h) {
      hit_pos = vec4(ray_pos + ray_dir*max_dist,max_dist);
      hit_norm = vec3(0,0,1);
      hit_dh = vec3(1,1,h);
      return vec4(0);
   }
   
   float d;
   if (ray_dir.z >= 0.) {
      d = md;
      hit_norm = vec3(-normalize(mdr),0);
      hit_dh = vec3(vec2(ray_pos.z + ray_dir.z*d - prev_h, h-prev_h)/max_h,h);
   }
   else {
      d = (h - ray_pos.z)/ray_dir.z;
      if (md > d) {
         d = md;
         hit_norm = vec3(-normalize(mdr),0);
         hit_dh = vec3(vec2(ray_pos.z + ray_dir.z*d - prev_h, h-prev_h)/max_h,h);
      } else {
         hit_norm = vec3(0,0,1);
         hit_dh = vec3(1,1,h);
      }
   }
   hit_pos = vec4(ray_pos + ray_dir*d, d);
   return mc + vec4(p, n);
}

// Function 917
vec4 noised( in vec3 x )
{
    vec3 p = floor(x);
    vec3 w = fract(x);
	vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
    
    float n = p.x + p.y*157.0 + 113.0*p.z;
    
    float a = hash(n+  0.0);
    float b = hash(n+  1.0);
    float c = hash(n+157.0);
    float d = hash(n+158.0);
    float e = hash(n+113.0);
	float f = hash(n+114.0);
    float g = hash(n+270.0);
    float h = hash(n+271.0);
	
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z, 
                 du * (vec3(k1,k2,k3) + u.yzx*vec3(k4,k5,k6) + u.zxy*vec3(k6,k4,k5) + k7*u.yzx*u.zxy ));
}

// Function 918
float NOISE(vec2 x) { float v = 0.0; float a = 0.5; vec2 shift = vec2(100); mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50)); for (int i = 0; i < NUM_OCTAVES; ++i) { v += a * noise(x); x = rot * x * 2.0 + shift; a *= 0.5; } return v; }

// Function 919
vec4 texNoise(vec2 uv){ float f = 0.; f+=texture(iChannel0, uv*.125).r*.5; f+=texture(iChannel0,uv*.25).r*.25;
                       f+=texture(iChannel0,uv*.5).r*.125; f+=texture(iChannel0,uv*1.).r*.125; f=pow(f,1.2);return vec4(f*.45+.05);}

// Function 920
vec3 atm_cloudnoise2_offs( vec4 r, bool lowfreq )
{
    float lod = log2( r.w );
    const vec2 offs = vec2( 1, 0 );
    vec3 offs_y = vec3(
        textureLod( iChannel3, ( r.xyz + offs.xyy ) / 64., lod - 1. ).x,
		textureLod( iChannel3, ( r.xyz + offs.yxy ) / 64., lod - 1. ).x,
        textureLod( iChannel3, ( r.xyz + offs.yyx ) / 64., lod - 1. ).x );
	if( !lowfreq )
    {
        offs_y = 2. / 3. * offs_y + vec3(
        	textureLod( iChannel3, ( r.xyz + offs.xyy ) / 32., lod ).x,
			textureLod( iChannel3, ( r.xyz + offs.yxy ) / 32., lod ).x,
			textureLod( iChannel3, ( r.xyz + offs.yyx ) / 32., lod ).x ) / 3.;
    }
    return offs_y;
}

// Function 921
vec4 snoise(vec2 texc) { return  2.0*texture(randSampler,texc, -1000.0)-vec4(1.0); }

// Function 922
float iqsVoronoiDistance( vec2 x ) {
    vec2 p = vec2(floor(x));
    vec2 f = fract(x);

    vec2 mb;
    vec2 mr;

    float res = 8.0;
    for(int j = -1; j <= 1; j++) {
        for(int i = -1; i <= 1; i++) {
            vec2 b = vec2(float(i), float(j));
            vec2 o = random2(p + b);
            o = 0.5 + 0.5 * sin(iTime + 6.2831 * o); // 0 to 1 range
            vec2 r = vec2(b) + o - f;
            float d = dot(r,r);

            if( d < res ) {
                res = d;
                mr = r;
                mb = b;
            }
    	}
    }

    res = 8.0;
    for(int j = -2; j <= 2; j++) {
        for(int i = -2; i <= 2; i++) {
            vec2 b = mb + vec2(float(i), float(j));
            vec2 o = random2(p + b);
            o = 0.5 + 0.5 * sin(iTime + 6.2831 * o); // 0 to 1 range
            vec2 r = vec2(b) + o - f;
            float d = dot(0.5 * (mr + r), normalize(r - mr));

            res = min( res, d );
        }
    }

    return res;
}

// Function 923
float voronoi_tile(vec2 p) {
    vec2 g = floor(p);
    vec2 f = fract(p);
    vec2 k = f*f*f*(6.0*f*f - 15.0*f + 10.0);

    f -= vec2(0.5);
    g -= vec2(0.5);
    float res = 1.0;
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {
            vec2 b = vec2(i, j);
            float d = length(hash2(g + b) - abs(f) + b);
            res = min(res, d);
        }
    }
    return res;
}

// Function 924
float perlinFbm(vec3 p, float freq, int octaves)
{
    float G = exp2(-.85);
    float amp = 1.;
    float noise = 0.;
    for (int i = 0; i < octaves; ++i)
    {
        noise += amp * valueNoise(p * freq, freq);
        freq *= 2.;
        amp *= G;
    }
    
    return noise;
}

// Function 925
float voronoi3D(vec3 uv) {
    vec3 fl = floor(uv);
    vec3 fr = fract(uv);
    float res = 1.0;
    for(int k=-1;k<=1;k++)
    for( int j=-1; j<=1; j++ ) {
        for( int i=-1; i<=1; i++ ) {
            vec3 p = vec3(i, j, k);
            float h = hash3D(fl+p);
            vec3 vp = p-fr+h;
            float d = dot(vp, vp);
            
            res +=1.0/pow(d, 8.0);
        }
    }
    return pow( 1.0/res, 1.0/16.0 );
}

// Function 926
float noise (vec3 p) {
	vec3 ip = floor(p),
         s = vec3(7.0, 157.0, 113.0);
    p-= ip;
    vec4 h = vec4(0.0, s.yz, s.y+s.z) + dot(ip, s);
    h = mix(fract(sin(h)*43758.5), fract(sin(h+s.x)*43758.5), p.x);
    h.xy = mix(h.xz, h.yw, p.y);
    return mix(h.x, h.y, p.z);
}

// Function 927
float snoise(vec2 v)
{
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    // First corner
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);

    // Other corners
    vec2 i1;
    //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
    //i1.y = 1.0 - i1.x;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    // x0 = x0 - 0.0 + 0.0 * C.xx ;
    // x1 = x0 - i1 + 1.0 * C.xx ;
    // x2 = x0 - 1.0 + 2.0 * C.xx ;
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
    + i.x + vec3(0.0, i1.x, 1.0 ));

    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

    // Compute final noise value at P
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Function 928
vec4 Noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
	f = f*f*(3.0-2.0*f);
//	vec2 f2 = f*f; f = f*f2*(10.0-15.0*f+6.0*f2);

	vec2 uv = p + f;

#if (1)
	vec4 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0 );
#else
	// on some hardware interpolation lacks precision
	vec4 rg = mix( mix(
				texture( iChannel0, (floor(uv)+0.5)/256.0, -100.0 ),
				texture( iChannel0, (floor(uv)+vec2(1,0)+0.5)/256.0, -100.0 ),
				fract(uv.x) ),
				  mix(
				texture( iChannel0, (floor(uv)+vec2(0,1)+0.5)/256.0, -100.0 ),
				texture( iChannel0, (floor(uv)+1.5)/256.0, -100.0 ),
				fract(uv.x) ),
				fract(uv.y) );
#endif			  

	return rg;
}

// Function 929
float iqnoise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
		
	float k = 1.0+63.0*pow(1.0-v,4.0);
    
    //vec2 noiseTexelSize = (vec2(1.0) / iChannelResolution[0].xy);
	
	float va = 0.0;
	float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {            
        vec2 g = vec2( float(i),float(j) );
        
        /* Original static-params.
        vec3 cellParams = hash3(p + g);
		*/
        
        /* Aborted buggy attempt to use a noise texture.
        float animationScalar = (0.5 * iTime);
        float animationFract = fract(animationScalar);
        vec2 paramsOrigin = (13.0 * (p + g));
        vec2 baseParamsIndex = (paramsOrigin + vec2(floor(animationScalar), 0.0));                            
        vec3 startSeedSample = texture(iChannel0, (baseParamsIndex + vec2(0.5)) * noiseTexelSize).xyz;
        vec3 endSeedSample = texture(iChannel0, ((baseParamsIndex + vec2(1.0, 0.0)) + vec2(0.5)) * noiseTexelSize).xyz;
        vec3 cellParams = smoothstep(startSeedSample, endSeedSample, vec3(animationFract));
        //cellParams = texture(iChannel0, ((paramsOrigin + vec2(animationScalar, 0.0)) + vec2(0.5)) * noiseTexelSize).xyz;
		*/
        
        // Straightforward sin waves.
        vec3 cellParamSpeeds = mix(vec3(0.05), vec3(0.6), hash3(p + g));
        vec3 cellParams = smoothstep(-1.0, 1.0, sin(cellParamSpeeds * (iTime + 200.0)));
        cellParams.xy = mix(vec2(0.1), vec2(0.9), cellParams.xy); // Avoid overly sharp edges my ensuring a minimum distance between the cell-centers.
		
		vec3 o = cellParams*vec3(u,u,1.0);
		vec2 r = g - f + o.xy;
		float d = dot(r,r);
		float ww = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
		va += o.z*ww;
		wt += ww;
    }
	
    return va/wt;
}

// Function 930
vec3 noised( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
#if 1
    // quintic interpolation
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0);
#else
    // cubic interpolation
    vec2 u = f*f*(3.0-2.0*f);
    vec2 du = 6.0*f*(1.0-f);
#endif    
    
    float va = hash( i + vec2(0.0,0.0) );
    float vb = hash( i + vec2(1.0,0.0) );
    float vc = hash( i + vec2(0.0,1.0) );
    float vd = hash( i + vec2(1.0,1.0) );
    
    float k0 = va;
    float k1 = vb - va;
    float k2 = vc - va;
    float k4 = va - vb - vc + vd;

    return vec3( va+(vb-va)*u.x+(vc-va)*u.y+(va-vb-vc+vd)*u.x*u.y, // value
                 du*(u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va) );     // derivative                
}

// Function 931
float gnoise(in vec3 x, int seed)
{
    // grid
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    #if 1
    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
    vec3 iu = 1.0 - u;
    #else
    // cubic interpolant
    vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
    #endif    
    
    // gradients
    vec3 ga = grad( p+vec3(0.0,0.0,0.0), seed );
    vec3 gb = grad( p+vec3(1.0,0.0,0.0), seed );
    vec3 gc = grad( p+vec3(0.0,1.0,0.0), seed );
    vec3 gd = grad( p+vec3(1.0,1.0,0.0), seed );
    vec3 ge = grad( p+vec3(0.0,0.0,1.0), seed );
	vec3 gf = grad( p+vec3(1.0,0.0,1.0), seed );
    vec3 gg = grad( p+vec3(0.0,1.0,1.0), seed );
    vec3 gh = grad( p+vec3(1.0,1.0,1.0), seed );
    
    // projections
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
	
    // interpolations
    return 
        va + u.x*(vb-va) + u.y*(vc-va) + u.z*(ve-va) + 
        u.x*u.y*(va-vb-vc+vd) + 
        u.y*u.z*(va-vc-ve+vg) + 
        u.z*u.x*(va-vb-ve+vf) + 
        (-va+vb+vc-vd+ve-vf-vg+vh)*u.x*u.y*u.z;
}

// Function 932
vec3 noise_deriv(in vec2 p) {
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
    
    float a = hash( i + vec2(0.0,0.0) );
	float b = hash( i + vec2(1.0,0.0) );    
    float c = hash( i + vec2(0.0,1.0) );
	float d = hash( i + vec2(1.0,1.0) );    
    float h1 = mix(a,b,u.x);
    float h2 = mix(c,d,u.x);
                                  
    return vec3(abs(mix(h1,h2,u.y)),
               6.0*f*(1.0-f)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));
}

// Function 933
float noise(vec2 p) {
  	return random(p.x + p.y*1e5);
}

// Function 934
float noise(in vec2 uv)
{
    return sin(uv.x)+cos(uv.y);
}

// Function 935
vec3 Voronoi(vec2 p){
    
    // Convert to the hexagonal grid.
    vec2 pH = pixToHex(p); // Map the pixel to the hex grid.

    // There'd be a heap of ways to get rid of this array and speed things up. The
    // most obvious, would be unrolling the loops, but there'd be more elegant ways.
    // Either way, I've left it this way just to make the code easier to read.
 
    // Hexagonal grid offsets. "vec2(0)" represents the center, and the other offsets effectively circle it.
    // Thanks, Abje. Hopefully, the compiler will know what to do with this. :)
	const vec2 hp[7] = vec2[7](vec2(-1), vec2(0, -1), vec2(-1, 0), vec2(0), vec2(1), vec2(1, 0), vec2(0, 1)); 
    
    
    // Voronoi cell ID containing the minimum offset point distance. The nearest
    // edge will be one of the cell's edges.
    vec2 minCellID = vec2(0); // Redundant initialization, but I've done it anyway.

    // As IQ has commented, this is a regular Voronoi pass, so it should be
    // pretty self explanatory.
    //
    // First pass: Regular Voronoi.
	vec2 mo, o;
    
    // Minimum distance, "smooth" distance to the nearest cell edge, regular
    // distance to the nearest cell edge, and a line distance place holder.
    float md = 8., lMd = 8., lMd2 = 8., lnDist, d;
    
    for (int i=0; i<7; i++){
    
        // Determine the offset hexagonal point.
        vec2 h = hexPt(pH + hp[i]) - p;
        // Determine the distance metric to the point.
    	d = dot(h, h);
    	if( d<md ){ // Perform updates, if applicable.
            
            md = d;  // Update the minimum distance.
            // Keep note of the position of the nearest cell point - with respect
            // to "p," of course. It will be used in the second pass.
            mo = h; 
            //cellID = h + p; // For cell coloring.
            minCellID = hp[i]; // Record the minimum distance cell ID.
        }
    }
    
    // Second pass: Point to nearest cell-edge distance.
    //
    // With the ID of the cell containing the closest point, do a sweep of all the
    // surrounding cell edges to determine the closest one. You do that by applying
    // a standard distance to a line formula.
    for (int i=0; i<7; i++){
    
         // Determine the offset hexagonal point in relation to the minimum cell offset.
        vec2 h = hexPt(pH + hp[i] + minCellID) - p - mo; // Note the "-mo" to save some operations. 
        
        // Skip the same cell.
        if(dot(h, h)>.00001){
            
            // This tiny line is the crux of the whole example, believe it or not. Basically, it's
            // a bit of simple trigonometry to determine the distance from the cell point to the
            // cell border line. See IQ's article (link above) for a visual representation.            
            lnDist = dot(mo + h*.5, normalize(h));
            
            // Abje's addition. Border distance using a smooth minimum. Insightful, and simple.
            //
            // On a side note, IQ reminded me that the order in which the polynomial-based smooth
            // minimum is applied effects the result. However, the exponentional-based smooth
            // minimum is associative and commutative, so is more correct. In this particular case, 
            // the effects appear to be negligible, so I'm sticking with the cheaper polynomial-based
            // smooth minimum, but it's something you should keep in mind. By the way, feel free to 
            // uncomment the exponential one and try it out to see if you notice a difference.
            //
            // Polynomial-based smooth minimum. The last factor controls the roundness of the 
            // edge joins. Zero gives you sharp joins, and something like ".25" will produce a
            // more rounded look. Tomkh noticed that a variable smoothing factor - based on the
            // line distance - produces continuous isolines.
            lMd = smin2(lMd, lnDist, (lnDist*.5 + .5)*.15);
            //lMd = smin2(lMd, lnDist, .1);
            // Exponential-based smooth minimum.
            //lMd = sminExp(lMd, lnDist, 20.); 
            //lMd = sminExp(lMd, lnDist, (lnDist*.5 + .5)*50.);
            
            // Minimum regular straight-edged border distance. If you only used this distance,
            // the web lattice would have sharp edges.
            lMd2 = min(lMd2, lnDist);
            
        }

    }

    // Return the smoothed and unsmoothed distance. I think they need capping at zero... but I'm not 
    // positive. Although not used here, the standard minimum point distance is returned also.
    return max(vec3(lMd, lMd2, md), 0.);
    
    
}

// Function 936
float noise3D(vec3 x) {
    vec3 p = floor(x);
    vec3 n = fract(x);
    vec3 f = n*n*(3.0-2.0*n);
    float winx = 1.0;
    float winy = 1.0;
    float winz = 1.0;
    
    return mix(
        	mix(
                mix(hash3D(p)     				  , hash3D(p+vec3(winx, 0.0, 0.0)), f.x),
                mix(hash3D(p+vec3(0.0, winy, 0.0)), hash3D(p+vec3(winx, winy, 0.0)), f.x),
                f.y),
        	mix(
                mix(hash3D(p+vec3(0.0, 0.0, winz)), hash3D(p+vec3(winx, 0.0, winz)), f.x),
                mix(hash3D(p+vec3(0.0, winy, winz)), hash3D(p+vec3(winx, winy, winz)), f.x),
                f.y),
        	f.z);
}

// Function 937
float simplex2d(vec2 p) {
	 vec2 s = floor(p + (p.x+p.y)*F2);
	 vec2 x = p - s - (s.x+s.y)*G2;
	 
     float e = step(0.0, x.x-x.y);
	 vec2 i1 = vec2(e, 1.0-e);
	 	 
	 vec2 x1 = x - i1 - G2;
	 vec2 x2 = x - 1.0 - 2.0*G2;
	 
	 vec3 w, d;
	 	 
	 w.x = dot(x, x);
	 w.y = dot(x1, x1);
	 w.z = dot(x2, x2);
	 	 
	 w = max(0.5 - w, 0.0);
	 
	 d.x = dot(random2(s + 0.0), x);
	 d.y = dot(random2(s +  i1), x1);
	 d.z = dot(random2(s + 1.0), x2);
	 
	 w *= w;
	 w *= w;
	 d *= w;
	 	 
	 return 0.5+dot(d, vec3(70.0));
}

// Function 938
float Voronoi(vec2 p){
    
	vec2 g = floor(p), o;
	p -= g;// p = fract(p);
	
	vec2 d = vec2(2); // 1.4, etc.
    
	for(int y = -1; y <= 1; y++){
		for(int x = -1; x <= 1; x++){
            
			o = vec2(x, y);
            o += hash22(g + o) - p;
            
			float h = dot(o, o);
            d.y = max(d.x, min(d.y, h)); 
            d.x = min(d.x, h);            
		}
	}
	
	return min(sqrt(d.x), 1.);
    //return min(d.y - d.x, 1.); // etc.
}

// Function 939
float noise(in vec3 x) {
	vec3 p = floor(x);
	vec3 f = fract(x);
	f = f * f * (3. - 2. * f);
	vec2 uv = (p.xy + vec2(37., 17.) * p.z) + f.xy;
	vec2 rg = textureLod(iChannel0, (uv + .5) / 256., 0.).yx;
	return -1. + 2.4 * mix(rg.x, rg.y, f.z);
}

// Function 940
float voronoi(in vec2 uv) {
    // split in squares
    const float space = 0.4;
    vec2 rf = vec2(1.0, 0.5);
    vec2 rs = vec2(0.5, 1.6);
      
    // take n sample in each square
    vec2 uvi = vec2(floor(uv / space - 0.5));
    vec2 p1 = uvi * space;
    vec2 p2 = (uvi + vec2(0, 1)) * space;
    vec2 p3 = (uvi + vec2(1, 0)) * space;
    vec2 p4 = (uvi + vec2(1, 1)) * space;
    float m = 10000.0;
    
    for (int i = 0; i < N; i++) {
        p1 = (uvi + rand2(p1 * rf + rs)) * space;
        m = min(m, distance(p1, uv));
        p2 = (uvi + vec2(0, 1) + rand2(p2 * rf + rs)) * space;
        m = min(m, distance(p2, uv));
        p3 = (uvi + vec2(1, 0) + rand2(p3 * rf + rs)) * space;
        m = min(m, distance(p3, uv));
        p4 = (uvi + vec2(1, 1) + rand2(p4 * rf + rs)) * space;
        m = min(m, distance(p4, uv));
    }
    
    return 1. - pow(m, 0.5) / space * sqrt(float(N)) * 0.2;
}

// Function 941
float treeNoise(vec2 p)
{
    p = vec2(ivec2(p * 1.8));
    vec2 o = vec2(0.12, 0.08);
    return (hash12(p + o.xy) + hash12(p + o.yx) + hash12(p + o.xx) + hash12(p + o.yy)) * 0.25;
}

// Function 942
vec2 Noisev2v4 (vec4 p)
{
  vec4 i, f, t1, t2;
  i = floor (p);
  f = fract (p);
  f = f * f * (3. - 2. * f);
  t1 = Hashv4f (dot (i.xy, cHashA3.xy));
  t2 = Hashv4f (dot (i.zw, cHashA3.xy));
  return vec2 (mix (mix (t1.x, t1.y, f.x), mix (t1.z, t1.w, f.x), f.y),
               mix (mix (t2.x, t2.y, f.z), mix (t2.z, t2.w, f.z), f.w));
}

// Function 943
vec3 noise3d(in vec3 x)
{
    return texture(iChannel1, x / 32.0).xyz;
}

// Function 944
float voronoi( vec3 x, float tile ) {
    vec3 p = floor(x);
    vec3 f = fract(x);

    float res = 100.;
    for(int k=-1; k<=1; k++){
        for(int j=-1; j<=1; j++) {
            for(int i=-1; i<=1; i++) {
                vec3 b = vec3(i, j, k);
                vec3 c = p + b;

                if( tile > 0. ) {
                    c = mod( c, vec3(tile) );
                }

                vec3 r = vec3(b) - f + hash13( c );
                float d = dot(r, r);

                if(d < res) {
                    res = d;
                }
            }
        }
    }

    return 1.-res;
}

// Function 945
float noise( in vec2 x )
{
    ivec2 p = ivec2(floor(x));
    vec2 f = fract(x);
	f = f*f*(3.0-2.0*f);
	ivec2 uv = p.xy;
	float rgA = texelFetch( iChannel1, (uv+ivec2(0,0))&255, 0 ).x;
    float rgB = texelFetch( iChannel1, (uv+ivec2(1,0))&255, 0 ).x;
    float rgC = texelFetch( iChannel1, (uv+ivec2(0,1))&255, 0 ).x;
    float rgD = texelFetch( iChannel1, (uv+ivec2(1,1))&255, 0 ).x;
    return mix( mix( rgA, rgB, f.x ),
                mix( rgC, rgD, f.x ), f.y );
}

// Function 946
float noise( in vec3 p )
{
    vec3 f = fract(p);
    p = floor(p);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel3, (uv+ 0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 947
vec4 FlowNoise(vec3 uvw, vec2 uv)
{
	vec4 n = vec4(0.);

	float f = 1.;
	float a = 1.;
			
	float lac = 2.13;
	
#if 0	
	for (int i=0; i<5; i++)
	{	
		//offsetting swirl angle relative to position seems to flow along the gradient
		float ang = iTime*.4;//+uv.y*0.5;
		
		ang *= Checker2(uvw.xy*0.0125);
		
		vec3 ax = normalize(vec3(1,1,1)); 
//		vec3 ax = texture(iChannel0,vec2(float(i)*0.1,0.)).xyz*2.-1.;
		quat = quat_rotation( ang*2.*f, normalize(ax) );

		float e = 0.1;//*f;
		
		//advect by going back in domain along noise gradient
		vec4 dn = dnoise(uvw);
		uvw -= 0.01*dn.xyz;
		
		n += abs(a*dn);
		uvw *= lac;
		f *= lac;
		a *= (1./lac);
	}
#else
	vec3 ax = normalize(vec3(1,1,1)); 
	float e = 0.1;//*f;
	float ang;
	vec4 dn;
		ang = iTime*.4+uv.y*0.5;
		quat = quat_rotation( ang*2.*f, normalize(ax) );
		dn = dnoise(uvw);
		uvw -= 0.01*dn.xyz;
		n += abs(a*dn);
		uvw *= lac;
		f *= lac;
		a *= (1./lac);
	
		ang = iTime*.4+uv.y*0.5;
		quat = quat_rotation( ang*2.*f, normalize(ax) );
		dn = dnoise(uvw);
		uvw -= 0.01*dn.xyz;
		n += abs(a*dn);
		uvw *= lac;
		f *= lac;
		a *= (1./lac);

		ang = iTime*.4+uv.y*0.5;
		quat = quat_rotation( ang*2.*f, normalize(ax) );
		dn = dnoise(uvw);
		uvw -= 0.01*dn.xyz;
		n += abs(a*dn);
		uvw *= lac;
		f *= lac;
		a *= (1./lac);

		ang = iTime*.4+uv.y*0.5;
		quat = quat_rotation( ang*2.*f, normalize(ax) );
		dn = dnoise(uvw);
		uvw -= 0.01*dn.xyz;
		n += abs(a*dn);
		uvw *= lac;
		f *= lac;
		a *= (1./lac);

		ang = iTime*.4+uv.y*0.5;
		quat = quat_rotation( ang*2.*f, normalize(ax) );
		dn = dnoise(uvw);
		uvw -= 0.01*dn.xyz;
		n += abs(a*dn);
		uvw *= lac;
		f *= lac;
		a *= (1./lac);
	
#endif
	
	return n;
}

// Function 948
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return -1.0+1.7*mix( rg.x, rg.y, f.z );
}

// Function 949
float noise(vec2 n) {
	const vec2 d = vec2(0.0, 1.0);
	vec2 b = floor(n);
	vec2 f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
	return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

// Function 950
float noise(in vec3 x, in float base_scale, in float space_decay, in float height_decay,
           in float shift_by) {
	float h = 1.0;
    float s = base_scale;

    float summation = 0.0;
    
    for (int i = 0; i < 5; ++i) {
    	summation = summation + h * noise_term(x + vec3(0.0, 0.0, s * shift_by), s);
        s *= space_decay;
        h *= height_decay;
    }
    return summation;
}

// Function 951
float noise(in vec2 x)
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0;
    float res = mix(mix(hash(n+  0.0), hash(n+  1.0), f.x),
                    mix(hash(n+ 57.0), hash(n+ 58.0), f.x), f.y);
    return res;
}

// Function 952
float map5_perlin(vec2 p)  
{	
    float r = 0.;
    float s = 1.;
    float f = 1.;
    
    float w = 0.;
    for(int i = 0;i < 5;i++){
        r += s*noise_perlin(p*f); w += s;
        s /= 2.;
        f *= 2.;
    }
    return r/w;
}

// Function 953
vec3 blobnoisenrm(vec2 v, float s)
{
    vec2 e = vec2(.01,0);
    return normalize(
           vec3(blobnoise(v + e.xy, s) - blobnoise(v -e.xy, s),
                blobnoise(v + e.yx, s) - blobnoise(v -e.yx, s),
                1.0));
}

// Function 954
vec2 rectvoronoi(in vec3 x, mat3 d, vec4 s) {
    ivec3 p = ivec3(floor( x ));
    vec3 f = fract( x );

    ivec3 mb;
    vec3 mr;
    float id = 1.0e20;
    const int range = 3;
    for( int k=-VORORANGE; k<=VORORANGE; k++ )
    for( int j=-VORORANGE; j<=VORORANGE; j++ )
    for( int i=-VORORANGE; i<=VORORANGE; i++ )
    {
        ivec3 b = ivec3( i, j, k );
        vec3 B = vec3(p + b);
        vec3 rv = rand3(B, d, s );
        //vec3 rv = RAND(B, 0.0, 1.0, s);
        vec3 r = vec3(b) - f + rv;
        float dis = length( r );

        if(dis < id) {
            mb = b;
            mr = r;
            id = dis;
        }
    }
    float bd = 1.0e20;
    for( int k=-VORORANGE; k<=VORORANGE; k++ )
    for( int j=-VORORANGE; j<=VORORANGE; j++ )
    for( int i=-VORORANGE; i<=VORORANGE; i++ )
    {
        ivec3 b = mb + ivec3( i, j, k );
        vec3 B = vec3(p + b);
        vec3 rv = rand3(B, d, s );
        //vec3 rv = RAND(B, 0.0, 1.0, s);
        vec3 r = vec3(b) - f + rv;
        float dis = dot( 0.5*(mr+r), normalize(r-mr) );

        bd = min(bd, dis);
    }
    return vec2(id, bd);
}

// Function 955
float noise1(float p)
{
	float fl = floor(p);
	float fc = fract(p);
	return mix(rand(fl), rand(fl + 1.0), fc);
}

// Function 956
float mul_noise(vec3 x) {
    float n = 2.*noise(x);  x *= 2.1; // return n/2.;
         n *= 2.*noise(x);  x *= 1.9;
         n *= 2.*noise(x);  x *= 2.3;
         n *= 2.*noise(x);  x *= 1.9;
      //   n *= 2.*noise(x);
    return n/2.; 
}

// Function 957
VoronoiData voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;
    vec2 mi;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
            mi = n + g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return VoronoiData( md, mr, mi );
}

// Function 958
float PerlinNoise2D(vec2 uv, int octaves) {
    float c = 0.0;
    float s = 0.0;
    for (float i = 0.0; i < float(octaves); i++) {
        c += SmoothNoise2D(uv * pow(2.0, i)) * pow(0.5, i);
        s += pow(0.5, i);
    }
    
    return c /= s;
}

// Function 959
float noise1( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
#ifndef HIGH_QUALITY_NOISE
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel2, (uv+ 0.5)/256.0, 0.0 ).yx;
#else
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z);
	vec2 rg1 = textureLod( iChannel2, (uv+ vec2(0.5,0.5))/256.0, 0.0 ).yx;
	vec2 rg2 = textureLod( iChannel2, (uv+ vec2(1.5,0.5))/256.0, 0.0 ).yx;
	vec2 rg3 = textureLod( iChannel2, (uv+ vec2(0.5,1.5))/256.0, 0.0 ).yx;
	vec2 rg4 = textureLod( iChannel2, (uv+ vec2(1.5,1.5))/256.0, 0.0 ).yx;
	vec2 rg = mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
#endif	
	return mix( rg.x, rg.y, f.z );
}

// Function 960
float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = r2d(i);
    float b = r2d(i + vec2(1.0, 0.0));
    float c = r2d(i + vec2(0.0, 1.0));
    float d = r2d(i + vec2(1.0, 1.0));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    vec2 u = f*f*(3.0-2.0*f);
    // u = smoothstep(0.,1.,f);

    // Mix 4 coorners percentages
    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Function 961
vec4 snoise(vec3 v) {

    vec3 i  = floor(v + dot(v, vec3(third)));
    vec3 p1 = v - i + dot(i, vec3(sixth));

    vec3 g = step(p1.yzx, p1.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 p2 = p1 - i1 + sixth;
    vec3 p3 = p1 - i2 + third;
    vec3 p4 = p1 - 0.5;
    
    vec4 ix = i.x + vec4(0.0, i1.x, i2.x, 1.0);
    vec4 iy = i.y + vec4(0.0, i1.y, i2.y, 1.0);
    vec4 iz = i.z + vec4(0.0, i1.z, i2.z, 1.0);

    vec4 p = permute(permute(permute(iz)+iy)+ix);

    vec4 r = mod(p, 49.0);

    vec4 x_ = floor(r / 7.0);
    vec4 y_ = floor(r - 7.0 * x_); 

    vec4 x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    vec4 y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;

    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 g1 = vec3(a0.xy, h.x);
    vec3 g2 = vec3(a0.zw, h.y);
    vec3 g3 = vec3(a1.xy, h.z);
    vec3 g4 = vec3(a1.zw, h.w);

    vec4 n = taylor(vec4(dot(g1,g1),dot(g2,g2),dot(g3,g3),dot(g4,g4)));    

    vec3 n1 = g1 * n.x;
    vec3 n2 = g2 * n.y;
    vec3 n3 = g3 * n.z;
    vec3 n4 = g4 * n.w;

    vec4 m = vec4(dot(p1,p1),dot(p2,p2),dot(p3,p3),dot(p4,p4));
    
    vec4 m1 = max(0.6 - m, 0.0);
    vec4 m2 = m1 * m1;
    vec4 m3 = m2 * m1;
    vec4 m4 = m2 * m2;
    
    vec3 q1 = -6.0 * m3.x * p1 * dot(p1, n1) + m4.x * n1;
    vec3 q2 = -6.0 * m3.y * p2 * dot(p2, n2) + m4.y * n2;
    vec3 q3 = -6.0 * m3.z * p3 * dot(p3, n3) + m4.z * n3;
    vec3 q4 = -6.0 * m3.w * p4 * dot(p4, n4) + m4.w * n4;
     
    vec3 q = q1+q2+q3+q4;
    
    vec4 t = vec4(dot(p1,n1),dot(p2,n2),dot(p3,n3),dot(p4,n4));
    
    return (42.0 * vec4(q, dot(m4, t)));
    
}

// Function 962
float noise1( in float x )
{
    float p = floor(x);
    float f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix( hash1(p+0.0), hash1(p+1.0), f );
}

// Function 963
float WorleyNoise(vec2 uv)
{
    // Tile the space
    vec2 uvInt = floor(uv);
    vec2 uvFract = fract(uv);

    float minDist = 1.0; // Minimum distance initialized to max.

    // Search all neighboring cells and this cell for their point
    for(int y = -1; y <= 1; y++)
    {
        for(int x = -1; x <= 1; x++)
        {
            vec2 neighbor = vec2(float(x), float(y));

            // Random point inside current neighboring cell
            vec2 point = random2(uvInt + neighbor);

            // Animate the point
            point = 0.5 + 0.5 * sin(iTime + 6.2831 * point); // 0 to 1 range

            // Compute the distance b/t the point and the fragment
            // Store the min dist thus far
            vec2 diff = neighbor + point - uvFract;
            float dist = length(diff);
            minDist = min(minDist, dist);
        }
    }
    return minDist;
}

// Function 964
vec2 cellNoise(vec2 point )
{
    float d = 1e30, daux;
    vec2 v, vaux;
    vec2 o, offset;
    vec2 noise;    
    
    // 1st Worley pass (Inside cells)
    vec2 pi = floor(point); // Integer part of the point
    vec2 pf = fract(point); // Decimal part of the point

    for( int i=-1; i<=1; i++ )
    {
        for( int j=-1; j<=1; j++ )
        {
            offset = vec2(i,j); 
            noise = hash2( pi + offset ); // noise for the int point+offset
            #ifdef ANIMATE
           	 	noise = animateCell1(noise); // lets animate the cells!
            #endif	
            vaux = offset + noise - pf;
            float daux = euclideanDist2(vaux); // Compute square distance from the point to this cell

            if( daux<d )
            {
                d = daux; // keep min distance
                v = vaux; // keep v of the cell with min dist
                o = offset; // keep offset of the cell with min dist
            }
        }
    }

    // 2nd Worley pass (cell borders) 
    d=1e30;
    for( int i=-2; i<=2; i++ )
    {
        for( int j=-2; j<=2; j++ )
        {
            offset = o + vec2(i,j); // Get global offset (Old offset + new (borders) offset)
            noise = hash2( pi + offset ); // Get the noise for the int point + global offset
            #ifdef ANIMATE
            	noise = animateCell1(noise); // lets move these cells!
            #endif	
            vaux = offset + noise - pf; 
			
            daux = dot( 0.5*(v+vaux), normalize(vaux-v)); // Compute square distance
            d = min(d,daux); // Keep the minimun distance
        }
    }
	// Return the minimun distance and a lineal combination of the noise for coloring purposes
    // Adding a sin that depends on the time we get the turning on and off effect of the texture
    return vec2(d*2.0*(0.5*sin(iTime*1.6)+0.9),8.0*noise.x+5.0*noise.y);
}

// Function 965
vec3 VoronoiFactal (in vec2 coord, float time)
{
    const float freq = 4.0;
    const float freq2 = 6.0;
    const int iterations = 4;
    
    vec2 uv = coord * freq;
    
    vec3 color = vec3(0.0);
    float alpha = 0.0;
    float value = 0.0;
    
    for (int i = 0; i < iterations; ++i)
    {
    	vec4 v = voronoi( uv );
    	
        uv = ( v.xy * 0.5 + 0.5 ) * freq2 + Hash12( v.w );
        
        float f = pow( 0.01 * float(iterations - i), 3.0 );
        float a = 1.0 - smoothstep( 0.0, 0.08 + f, v.z );
        
        vec3 c = Rainbow( Hash11( float(i+1) / float(iterations) + value * 1.341 ), i > 1 ? 0.0 : a );
        
        color = color * alpha + c * a;
        alpha = max( alpha, a );
        value = v.w;
    }
    
    #ifdef COLORED
    	return color;
    #else
    	return vec3( alpha ) * Rainbow( 0.06, alpha );
    #endif
}

// Function 966
float snoise(float p){
	return snoise(vec2(p,0.));
}

// Function 967
float perlin(float x, float h) {
	float a = floor(x);
	return blerp(mod(x, 1.0),
		rand(vec2(a-1.0, h)), rand(vec2(a-0.0, h)),
		rand(vec2(a+1.0, h)), rand(vec2(a+2.0, h)));
}

// Function 968
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( hash( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hash( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( hash( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 969
float snoise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 970
float noise2d(vec2 uv)
{
    vec2 fr = fract(uv.xy);
    vec2 fl = floor(uv.xy);
    float h00 = Hash2d(fl);
    float h10 = Hash2d(fl + zeroOne.yx);
    float h01 = Hash2d(fl + zeroOne);
    float h11 = Hash2d(fl + zeroOne.yy);
    return mixP(mixP(h00, h10, fr.x), mixP(h01, h11, fr.x), fr.y);
}

// Function 971
float noise31(vec3 p8, float pro, float st) {
    float v=0.0;
    float s3=0.5;
    for(float g=min(0.,float(iFrame)); g<st; ++g) {
        v+=srnd31(p8+g*72.3)*s3;
        p8*=pro;
        s3*=0.5;
    }
    return v;
}

// Function 972
float worleyNoiseInefficient(vec2 uv) {
    float minDist = 1.0;
    
    for(float i = 0.0; i < 100.0; ++i) {
        vec2 bufB_UV = vec2(i / float(iResolution.x), 0.0);
        vec2 particlePos = texture(iChannel0, bufB_UV).xy;
        
        float dist = length(particlePos - uv);
        minDist = min(dist, minDist);
    }
    return minDist;
}

// Function 973
float noiseHigh(in vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0 + 113.0*p.z;
    float res = mix(mix(mix(hash(n+  0.0), hash(n+  1.0), f.x),
                        mix(hash(n+ 57.0), hash(n+ 58.0), f.x), f.y),
                    mix(mix(hash(n+113.0), hash(n+114.0), f.x),
                        mix(hash(n+170.0), hash(n+171.0), f.x), f.y), f.z);
    return res;
}

// Function 974
float noise(vec3 x) {  // By default, simple 3D value noise with cubic interpolation
    vec3 i = floor(x), // Switch to gradient noise above if you wish, but little differences
         F = fract(x), e = vec3(1,0,0),
         f = smoothstep(0.,1.,F );
    vec4 T = mix ( vec4(T(e.zzz),T(e.zxz), T(e.zzx), T(e.zxx) ),
                   vec4(T(e.xzz),T(e.xxz), T(e.xzx), T(e.xxx) ),
                   f.x );
    vec2 v = mix( T.xz, T.yw, f.y);
    return mix(v.x,v.y,f.z);
        }

// Function 975
vec2 noise3d2(vec3 pos)
{
    vec3  f = floor(pos);
    ivec3 i = ivec3(f);
    
    vec2 r000 = hash2(i + ivec3(0,0,0));
    vec2 r100 = hash2(i + ivec3(1,0,0));
    vec2 r010 = hash2(i + ivec3(0,1,0));
    vec2 r110 = hash2(i + ivec3(1,1,0));
    vec2 r001 = hash2(i + ivec3(0,0,1));
    vec2 r101 = hash2(i + ivec3(1,0,1));
    vec2 r011 = hash2(i + ivec3(0,1,1));
    vec2 r111 = hash2(i + ivec3(1,1,1));
    
    f = pos - f;
    
    f = f * f * (3. - 2. * f);
    
    return mix(mix(mix(r000,r100,f.x), mix(r010,r110,f.x), f.y),  mix(mix(r001,r101,f.x), mix(r011,r111,f.x), f.y),  f.z);
}

// Function 976
float FBMNoise(in vec2 st)
{
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = FBMRandom(i);
    float b = FBMRandom(i + vec2(1.0, 0.0));
    float c = FBMRandom(i + vec2(0.0, 1.0));
    float d = FBMRandom(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
           (c - a) * u.y * (1.0 - u.x) +
           (d - b) * u.x * u.y;
}

// Function 977
float BandNoise(float freqband)
{
    float result = 0.0;

    float freqCount = 160.0;
    for (float i=0.; i<freqCount; i++)
    {
        float stepf = (i/freqCount) + Noise((i)*0.0001)*0.253;
        float volume = max(1.-10.*abs(stepf-freqband/5000.),0.0);//1.0-(i/120.0); // //QFilter(stepf,freqband/5000.0,20.0)
        result += volume*sin(gTime*(stepf * 5000.0) * TAU);
    }

    return result * 0.5;
}

// Function 978
float simplex_noise(vec3 p)
{
    const float K1 = 0.333333333;
    const float K2 = 0.166666667;

    vec3 i = floor(p + (p.x + p.y + p.z) * K1);
    vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);

    // thx nikita: https://www.shadertoy.com/view/XsX3zB
    vec3 e = step(vec3(0.0), d0 - d0.yzx);
    vec3 i1 = e * (1.0 - e.zxy);
    vec3 i2 = 1.0 - e.zxy * (1.0 - e);

    vec3 d1 = d0 - (i1 - 1.0 * K2);
    vec3 d2 = d0 - (i2 - 2.0 * K2);
    vec3 d3 = d0 - (1.0 - 3.0 * K2);

    vec4 h = max(0.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
    vec4 n = h * h * h * h * vec4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0)));

    return dot(vec4(31.316), n);
}

// Function 979
float smooth_noise(vec2 pos)
{
	return   ( noise(pos + vec2(1,1)) + noise(pos + vec2(1,1)) + noise(pos + vec2(1,1)) + noise(pos + vec2(1,1)) ) / 16.0 		
		   + ( noise(pos + vec2(1,0)) + noise(pos + vec2(-1,0)) + noise(pos + vec2(0,1)) + noise(pos + vec2(0,-1)) ) / 8.0 		
    	   + noise(pos) / 4.0;
}

// Function 980
vec3 snoise_swirl( vec3 noisept, float timeMult, int octaves, float dimension, float lacunarity,  float xscl, float yscl, float zscl)
{
    float amp, ampsum;

    vec4 np = vec4(noisept.x, noisept.y,noisept.z, 0.0);
    np.x *= xscl;
    np.y *= yscl;
    np.z *= zscl;
    np.w = iTime * timeMult;

    vec3 sum = vec3(0,0,0);
    ampsum = 0.0;
    amp = 1.0;
    
    float H = 1.0f - dimension;
    float ampfactor = 1.0f / pow(lacunarity, H);
   

    for (int j=0; j<octaves; j++, amp*ampfactor)
    {
        if (j > 0)
        {
            np.x *= lacunarity;
            np.y *= lacunarity;
            np.z *= lacunarity;
        }

        float nx = snoise( vec4(np.x, 0.0, 0.0, np.w ));
        float ny = snoise( vec4(0.0, np.y, 0.0, np.w ));
        float nz = snoise( vec4(0.0, 0.0, np.z, np.w ));

        vec3 ret = vec3( nx, ny, nz );             
                          
        sum += ret * amp;

        ampsum += amp;
    }
    sum.x /= ampsum;
    sum.y /= ampsum;
    sum.z /= ampsum;
                          
    return sum;
}

// Function 981
float noise_value1( in vec3 p )
{
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);
	
    vec2 zo = vec2(0.,1.);
    float f000 = hash31(i);
    float f010 = hash31(i+zo.yxx);
    float f001 = hash31(i+zo.xyx);
    float f011 = hash31(i+zo.yyx);
    
    float hx1 = mix(f000,f010,u.x);
    float hx2 = mix(f001,f011,u.x);
    float hy1 = mix(hx1,hx2,u.y);
    
    float f100 = hash31(i+zo.xxy);
    float f110 = hash31(i+zo.yxy);
    float f101 = hash31(i+zo.xyy);
    float f111 = hash31(i+zo.yyy);
    
    
     hx1 = mix(f100,f110,u.x);
     hx2 = mix(f101,f111,u.x);
    float hy2 = mix(hx1,hx2,u.y);
    
    float h = mix(hy1,hy2,u.z);
    	
    return h;
}

// Function 982
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;

    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                        mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                        mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
    return 1.0 - sqrt(res);
}

// Function 983
float noises( in vec3 p){
	float a = 0.0;
	for(float i=1.0;i<6.0;i++){
		a += noise(p)/i;
		p = p*2.0 + vec3(0.0,a*0.001/i,a*0.0001/i);
	}
	return a;
}

// Function 984
vor voronoi( in vec2 x )
{
    
    
	vor res;
    // slower, but better handles big numbers
    vec2 n = floor(x);
    vec2 f = fract(x);
    vec2 h = step(.5,f) - 2.;
    n += h; f -= h;

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mr;

    float md = 8.0;
    for( int j=0; j<=3; j++ )
    for( int i=0; i<=3; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            res.p1 = g+o+n;
            res.cell1 = g+n;
        }
        res.pointDistances[i][j]=d;
        
        
        
        
        
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=0; j<=3; j++ )
    for( int i=0; i<=3; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON )
        {
            float newD = dot( 0.5*(mr+r), normalize(r-mr) );
            if (newD < md)
            {
                md = newD;
                res.p2 = g+o+n;
                res.cell2 = g+n;
            }
        }// skip the same cell
    }
    res.distToBorder = md;
    
    
    
    
    
   
    return res;
}

// Function 985
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
    vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
    return mix( rg.x, rg.y, f.z );
}

// Function 986
float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    // Four corners in 2D of a tile
    float a = hash2(i);
    float b = hash2(i + vec2(1.0, 0.0));
    float c = hash2(i + vec2(0.0, 1.0));
    float d = hash2(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Function 987
float simplex3d(vec3 p)
{
	 vec3 s = floor(p + dot(p, vec3(F3)));
	 vec3 x = p - s + dot(s, vec3(G3));

	 vec3 e = step(vec3(0.0), x - x.yzx);
	 vec3 i1 = e*(1.0 - e.zxy);
	 vec3 i2 = 1.0 - e.zxy*(1.0 - e);
	 
	 vec3 x1 = x - i1 + G3;
	 vec3 x2 = x - i2 + 2.0*G3;
	 vec3 x3 = x - 1.0 + 3.0*G3;
	 
	 vec4 w, d;
	 
	 w.x = dot(x, x);
	 w.y = dot(x1, x1);
	 w.z = dot(x2, x2);
	 w.w = dot(x3, x3);
	 
	 w = max(0.6 - w, 0.0);
	 
	 d.x = dot(random3(s), x);
	 d.y = dot(random3(s + i1), x1);
	 d.z = dot(random3(s + i2), x2);
	 d.w = dot(random3(s + 1.0), x3);
	 
	 w *= w;
	 w *= w;
	 d *= w;
	 
	 return dot(d, vec4(52.0));
}

// Function 988
float plasma_noise3( vec3 x ) 
{
    vec3 p = floor(x),f = fract(x);

    f = f*f*(3.-2.*f);  // or smoothstep     // to make derivative continuous at borders

#define plasma_hash3(p)  fract(sin(1e3*dot(p,vec3(1,57,-13.7)))*4375.5453)        // rand
    
    return mix( mix(mix( plasma_hash3(p+vec3(0,0,0)), plasma_hash3(p+vec3(1,0,0)),f.x),       // triilinear interp
                    mix( plasma_hash3(p+vec3(0,1,0)), plasma_hash3(p+vec3(1,1,0)),f.x),f.y),
                mix(mix( plasma_hash3(p+vec3(0,0,1)), plasma_hash3(p+vec3(1,0,1)),f.x),       
                    mix( plasma_hash3(p+vec3(0,1,1)), plasma_hash3(p+vec3(1,1,1)),f.x),f.y), f.z);
}

// Function 989
vec3 Voronoi(vec2 p){
    
    // Convert to the hexagonal grid.
    vec2 pH = pixToHex(p); // Map the pixel to the hex grid.

    // There'd be a heap of ways to get rid of this array and speed things up. The
    // most obvious, would be unrolling the loops, but there'd be more elegant ways.
    // Either way, I've left it this way just to make the code easier to read.
 
    // Hexagonal grid offsets. "vec2(0)" represents the center, and the other offsets effectively circle it.
    // Thanks, Abje. Hopefully, the compiler will know what to do with this. :)
	const vec2 hp[7] = vec2[7](vec2(-1), vec2(0, -1), vec2(-1, 0), vec2(0), vec2(1), vec2(1, 0), vec2(0, 1)); 
    
    
    // Voronoi cell ID containing the minimum offset point distance. The nearest
    // edge will be one of the cells edges.
    vec2 minCellID = vec2(0); // Redundant initialization, but I've done it anyway.

    // As IQ has commented, this is a regular Voronoi pass, so it should be
    // pretty self explanatory.
    //
    // First pass: Regular Voronoi.
	vec2 mo, o;
    
    // Minimum distance, "smooth" distance to the nearest cell edge, regular
    // distance to the nearest cell edge, and a line distance place holder.
    float md = 8., lMd = 8., lMd2 = 8., lnDist, d;
    
    for (int i=0; i<7; i++){
    
        // Determine the offset hexagonal point.
        vec2 h = hexPt(pH + hp[i]) - p;
        // Determine the distance metric to the point.
    	d = dot(h, h);
    	if( d<md ){ // Perform updates, if applicable.
            
            md = d;  // Update the minimum distance.
            // Keep note of the position of the nearest cell point - with respect
            // to "p," of course. It will be used in the second pass.
            mo = h; 
            //cellID = h + p; // For cell coloring.
            minCellID = hp[i]; // Record the minimum distance cell ID.
        }
    }
    
    // Second pass: Point to nearest cell-edge distance.
    //
    // With the ID of the cell containing the closest point, do a sweep of all the
    // surrounding cell edges to determine the closest one. You do that by applying
    // a standard distance to a line formula.
    for (int i=0; i<7; i++){
    
         // Determine the offset hexagonal point in relation to the minimum cell offset.
        vec2 h = hexPt(pH + hp[i] + minCellID) - p - mo; // Note the "-mo" to save some operations. 
        
        // Skip the same cell.
        if(dot(h, h)>.00001){
            
            // This tiny line is the crux of the whole example, believe it or not. Basically, it's
            // a bit of simple trigonometry to determine the distance from the cell point to the
            // cell border line. See IQ's article (link above) for a visual representation.            
            lnDist = dot(mo + h*.5, normalize(h));
            
            // Abje's addition. Border distance using a smooth minimum. Insightful, and simple.
            //
            // On a side note, IQ reminded me that the order in which the polynomial-based smooth
            // minimum is applied effects the result. However, the exponentional-based smooth
            // minimum is associative and commutative, so is more correct. In this particular case, 
            // the effects appear to be negligible, so I'm sticking with the cheaper polynomial-based
            // smooth minimum, but it's something you should keep in mind. By the way, feel free to 
            // uncomment the exponential one and try it out to see if you notice a difference.
            //
            // Polynomial-based smooth minimum. The last factor controls the roundness of the 
            // edge joins. Zero gives you sharp joins, and something like ".25" will produce a
            // more rounded look. Tomkh noticed that a variable smoothing factor - based on the
            // line distance - produces continuous isolines.
            lMd = smin2(lMd, lnDist, (lnDist*.5 + .5)*mix(0.0,0.4,sin(iTime * 0.5)*0.5+0.5));
            //lMd = smin2(lMd, lnDist, .1);
            // Exponential-based smooth minimum.
            //lMd = sminExp(lMd, lnDist, 20.); 
            //lMd = sminExp(lMd, lnDist, (lnDist*.5 + .5)*50.);
            
            // Minimum regular straight-edged border distance. If you only used this distance,
            // the web lattice would have sharp edges.
            lMd2 = min(lMd2, lnDist);
            
        }

    }

    float t = iTime * 5.;
    d = lMd * 25.;
    mo -= vec2(cos(d + t),sin(d + t)) / d;
    lMd2 = length(mo);
    //variation
    //lMd2 = length(mo*sin(mo.yx * 10.) * 2.);
    
    // Return the smoothed and unsmoothed distance. I think they need capping at zero... but I'm not 
    // positive. Although not used here, the standard minimum point distance is returned also.
    return max(vec3(lMd, lMd2, md), 0.);
    
    
}

// Function 990
float Noiseff (float p)
{
  float i, f;
  i = floor (p);  f = fract (p);
  f = f * f * (3. - 2. * f);
  vec2 t = Hashv2f (i);
  return mix (t.x, t.y, f);
}

// Function 991
vec4 voronoi( in vec2 x, out vec2 resUV, out float resOcc )
{
    vec2 n = floor( x );
    vec2 f = fract( x );

	vec2 uv = vec2(0.0);
	vec4 m = vec4( 8.0 );
	float m2 = 9.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = vec2( float(i),float(j) );
        vec2 o = hash2( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( 0.5*iTime + 6.2831*o );
        #endif	
		vec2 r = g - f + o;

        // distance and tex coordinates		
        vec2 u = vec2( dot( r, vec2(0.5, 0.866) ), 
					   dot( r, vec2(0.5,-0.866) ) );
		vec2 d = vec2( -r.y, 1.0 );
		float h = 0.5*abs(r.x)+0.866*r.y;
		if( h > 0.0 ) 
		{ 
			u = vec2( h, r.x );
			d = vec2( 0.866*abs(r.x)+0.5*r.y, 0.5*step(0.0,r.x) ); 
		}
		
        if( d.x<m.x )
        {
			m2 = m.x;
            m.x = d.x;
            m.y = dot(n+g,vec2(7.0,113.0) );
			m.z = d.y;
			m.w = max(r.y,0.0);
			uv = u;
        }
        else if( d.x<m2 )
		{
			m2 = d.x;
        }
			
    }
	resUV = uv;
	resOcc = m2-m.x;
    return m;
}

// Function 992
float noise( vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 993
float noise (vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = smoothstep(0.,1.,f);

    return mix(a, b, u.x) + 
        (c - a)* u.y * (1.0 - u.x) + 
        (d - b) * u.x * u.y;
}

// Function 994
vec3 snoiseVec3( vec3 x, in float numCells, int octaves )
{
   
  float s  = TileableNoiseFBM(vec3( x ), numCells, octaves);
  float s1 = TileableNoiseFBM(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ), numCells, octaves);
  float s2 = TileableNoiseFBM(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ), numCells, octaves);
  vec3 c = vec3( s , s1 , s2 );
  return c;

}

// Function 995
float coloredNoise(float t, float fc, float df)
{
    // Noise peak centered around frequency fc
    // containing frequencies between fc-df and fc+df.
    // Modulate df-wide noise by an fc-frequency sinusoid
    return sin(TAU*fc*mod(t,1.))*noise(t*df);
}

// Function 996
vec2 voronoi( in vec2 x )
{
    vec2 n = floor( x );
    vec2 f = fract( x );

	vec3 m = vec3( 8.0 );
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2  g = vec2( float(i), float(j) );
        vec2  o = hash( n + g );
        vec2  r = g - f + o;
		float d = dot( r, r );
        if( d<m.x )
            m = vec3( d, o );
    }

    return vec2( sqrt(m.x), m.y+m.z );
}

// Function 997
float get_octave_noise(vec2 pos)
{
    float rows = float(IMAGE_ROWS);
    pos *= rows;
    float columns = rows * (iResolution.x / iResolution.y);
    float scale = float(SCALE);
    if(scale <= 0.0f)
    {
        scale = 0.001f;
    }
    
    int octaves = int(OCTAVES);
    float lacunarity = max(LACUNARITY, 1.0f);
    float persistence = min(PERSISTANCE, 1.0f);
    
    float halfX = 0.0f;
    float halfY = 0.0f;
#if SCALE_FROM_CENTER
    halfX = columns / 2.0f;
    halfY = rows / 2.0f;
#endif

    float amplitude = 1.0f;
    float frequency = 1.0f;
    float noiseVal = 0.0f;
    
    // Add LODs
#if LEVEL_OF_DETAIL
    pos /= float(LEVEL_OF_DETAIL);
    pos = vec2(floor(pos.x), floor(pos.y));
    pos *= float(LEVEL_OF_DETAIL);
#endif

    vec2 offset = 0.1f * vec2(iTime * -1.0f, iTime * -1.25f);
    
    for (int i = 0; i < octaves; i++)
    {
#if NORMALIZE_OFFSET
        float sampleX = (((pos.x-halfX) / scale) * frequency) + offset.x;
        float sampleY = (((pos.y-halfY) / scale) * frequency) + offset.y;
#else
        float sampleX = (((pos.x-halfX + offset.x*scale) / scale) * frequency);
        float sampleY = (((pos.y-halfY + offset.y*scale) / scale) * frequency);
#endif
        float noise = (perlin(vec2(sampleX, sampleY)) * 2.0f) - 1.0f;
        noiseVal += noise * amplitude;
        // Decrease A and increase F
        amplitude *= persistence;
        frequency *= lacunarity;
    }    

    // Inverser lerp so that noiseval lies between 0 and 1 
#if SMOOTH_INVERSE_LERP
    noiseVal = smoothstep(-0.95f, 1.1f, noiseVal);
#else
    noiseVal = linear_step(-0.7f,0.85f,noiseVal);
#endif
    return noiseVal;
}

// Function 998
float snoiseFbm(int octaves,float persistence,float freq,vec3 coords){
 float amp=1.,maxamp=0.,sum=0.;for(int i=0;i<octaves;++i){
  sum+=amp*snoise(coords*freq);freq*=2.;maxamp+=amp;amp*=persistence;}
 return(sum/maxamp)*.5+.5;}

// Function 999
float noise( in vec3 p )
{
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);

    return mix( mix( mix( dot( hash3( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash3( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( hash3( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash3( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( hash3( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash3( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( hash3( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash3( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// Function 1000
float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Function 1001
float noise (vec3 x)
{
    //smoothing distance to texel http://www.iquilezles.org/www/articles/texture/texture.htm
    x*=32.;
    x += 0.5;
    
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f*f*f*(f*(f*6.0-15.0)+10.0);
	x = f+i;    
    x-=0.5;
    
    return texture( iChannel0, x/32.0 ).x;
}

// Function 1002
float snoise(vec2 v)
{
    const vec4 C = vec4(0.211324865405187, // (3.0-sqrt(3.0))/6.0
                        0.366025403784439, // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626, // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    // First corner
    vec2 i = floor(v + dot(v, C.yy) );
    vec2 x0 = v - i + dot(i, C.xx);

    // Other corners
    vec2 i1;
    //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
    //i1.y = 1.0 - i1.x;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    // x0 = x0 - 0.0 + 0.0 * C.xx ;
    // x1 = x0 - i1 + 1.0 * C.xx ;
    // x2 = x0 - 1.0 + 2.0 * C.xx ;
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                     + i.x + vec3(0.0, i1.x, 1.0 ));

    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

    // Compute final noise value at P
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Function 1003
float noise(vec2 uv)
{
	float v=0.;
	float a=-SPEED*t,	co=cos(a),si=sin(a); 
	mat2 M = mat2(co,-si,si,co);
	const int L = 7;
	float s=1.;
	for (int i=0; i<L; i++)
	{
		uv = M*uv;
		float b = tex(uv*s);
		v += 1./s* pow(b,RETICULATION); 
		s *= 2.;
	}
	
    return v/2.;
}

// Function 1004
float triNoise3D(in vec3 p, float spd) {
    float z = 1.4;
	float rz = 0.0;
    vec3 bp = p;
	for (float i = 0.0; i <= 3.0; i++) {
        vec3 dg = tri3(bp * 2.0);
        p += (dg + iTime * .3 * spd);
        bp *= 1.8;
		z *= 1.5;
		p *= 1.2;    
        rz += tri(p.z + tri(p.x + tri(p.y))) / z;
        bp += 0.14;
	}
	return rz;
}

// Function 1005
vec3 _DenoiseSHAD(vec2 uv, vec2 lUV, vec2 aUV, vec2 UVoff, vec3 CVP,
            vec3 CN, vec3 CVN, sampler2D ta, sampler2D tl, vec2 ires, vec2 hres, vec2 asfov) {
    //Denoiser help function
    vec4 l0=texture(tl,(lUV+UVoff)*ires);
    vec4 a0=texture(ta,(aUV+UVoff)*ires);
    vec4 l1=texture(tl,(lUV-UVoff)*ires);
    vec4 a1=texture(ta,(aUV-UVoff)*ires);
    vec3 L=vec3(0.); vec3 SP;
    if (a0.w<9990. && Box2(uv+UVoff,hres)<-0.5) {
        SP=normalize(vec3(((uv+UVoff)*ires*4.-1.)*asfov,1.))*a0.w;
    	L+=vec3(l0.zw,1.)*Weight(dot(SP-CVP,CVN),dot(CN,Read3(a0.y)*2.-1.));
    }
    if (a1.w<9990. && BoxC2(uv-UVoff,hres)<-0.5) {
        SP=normalize(vec3(((uv-UVoff)*ires*4.-1.)*asfov,1.))*a1.w;
    	L+=vec3(l1.zw,1.)*Weight(dot(SP-CVP,CVN),dot(CN,Read3(a1.y)*2.-1.));
    }
    return L;
}

// Function 1006
float noise( in vec2 x ){return texture(iChannel0, x*.01).x;}

// Function 1007
vec3 noised(in vec2 p){//noise with derivatives
	float res=0.;
    vec2 dres=vec2(0.);
    float f=1.;
    mat2 j=m2;
	for( int i=0; i< 3; i++ ) 
	{		
        p=m2*p*f+.6;     
        f*=1.2;
        float a=p.x+sin(2.*p.y);
        res+=sin(a);
        dres+=cos(a)*vec2(1.,2.*cos(2.*p.y))*j;
        j*=m2*f;
        
	}        	
	return vec3(res,dres)/3.;
}

// Function 1008
float noise(in vec2 p
){const float K1 = 0.366025404 // (sqrt(3)-1)/2;
 ;const float K2 = 0.211324865 // (3-sqrt(3))/6;
 ;vec2 i = floor( p + (p.x+p.y)*K1)
 ;vec2 a = p - i + (i.x+i.y)*K2
 ;vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0) //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
 ;vec2 b = a - o + K2
 ;vec2 c = a - 1.0 + 2.0*K2
 ;vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 )
 ;vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)))
 ;return dot( n, vec3(70.0));}

// Function 1009
float snoise( vec3 v ) {

    const vec2 C = vec2( 1.0 / 6.0, 1.0 / 3.0 );
    const vec4 D = vec4( 0.0, 0.5, 1.0, 2.0 );

    // First corner

    vec3 i  = floor( v + dot( v, C.yyy ) );
    vec3 x0 = v - i + dot( i, C.xxx );

    // Other corners

    vec3 g = step( x0.yzx, x0.xyz );
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //  x0 = x0 - 0. + 0.0 * C
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1. + 3.0 * C.xxx;

    // Permutations

    i = mod( i, 289.0 );
    vec4 p = permute( permute( permute(
        i.z + vec4( 0.0, i1.z, i2.z, 1.0 ) )
                              + i.y + vec4( 0.0, i1.y, i2.y, 1.0 ) )
                     + i.x + vec4( 0.0, i1.x, i2.x, 1.0 ) );

    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)

    float n_ = 1.0 / 7.0; // N=7

    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor( p * ns.z *ns.z );  //  mod(p,N*N)

    vec4 x_ = floor( j * ns.z );
    vec4 y_ = floor( j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs( x ) - abs( y );

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor( b0 ) * 2.0 + 1.0;
    vec4 s1 = floor( b1 ) * 2.0 + 1.0;
    vec4 sh = -step( h, vec4( 0.0 ) );

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3( a0.xy, h.x );
    vec3 p1 = vec3( a0.zw, h.y );
    vec3 p2 = vec3( a1.xy, h.z );
    vec3 p3 = vec3( a1.zw, h.w );

    // Normalise gradients

    vec4 norm = taylorInvSqrt( vec4( dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3) ) );
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value

    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3) ), 0.0 );
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                 dot(p2,x2), dot(p3,x3) ) );

}

// Function 1010
vec4 noise3Dv4(vec3 texc)
{
    vec3 x=texc*256.0;
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 uv;
    uv = (p.xy+vec2(17,7)*p.z) + 0.5 + f.xy;
    vec4 v1 = textureLod( randSampler, uv/256.0, 0.0);
    vec4 v2 = textureLod( randSampler, (uv+vec2(17,7))/256.0, 0.0);
    return mix( v1, v2, f.z );
}

// Function 1011
float Noise3D(const in vec3 mPos)
	{
		vec3 p = floor(mPos);
		vec3 f = fract(mPos);
		f = f * f * (2.0 - 1.0 * f); //EaseInExpo
		
		vec2 noiseUV = (p.xy + vec2(37.0,17.0) * p.z) + f.xy;
		vec2 rg = texture(iChannel0, (noiseUV + 0.5)/256.0, -100.0).yx;
		return mix(rg.x, rg.y, f.z);
	}

// Function 1012
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0;

    float res = mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                    mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);
    return res;
}

// Function 1013
float noise2( vec2 x ) {
    vec2 p = floor(x),f = fract(x);

    f = f*f*(3.-2.*f);  // or smoothstep     // to make derivative continuous at borders

#define hash2(p)  fract(sin(1e3*dot(p,vec2(1,57)))*4375.5453)        // rand
    
    return mix(mix( hash2(p+vec2(0,0)), hash2(p+vec2(1,0)),f.x),     // triilinear interp
               mix( hash2(p+vec2(0,1)), hash2(p+vec2(1,1)),f.x),f.y);
}

// Function 1014
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
   

    f = f*f*(3.1-2.0*f);

    float n = p.x + p.y*57.0 + 113.0*p.z;

    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                        mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                        mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
    return res;
}

// Function 1015
float noise( vec2 x )
{
    x *= iResolution.y;
    vec2 p = floor(x),f = fract(x);

    f = f*f*(3.-2.*f);                       // to make derivative continuous at borders

#define hash(p)  fract(sin(1e3*dot(p,vec2(1,57)))*43758.5453)        // rand
    
    return mix(mix( hash(p+vec2(0,0)), hash(p+vec2(1,0)),f.x),       // bilinear interp
               mix( hash(p+vec2(0,1)), hash(p+vec2(1,1)),f.x),f.y);
}

// Function 1016
float snoise (vec2 v)
{
	const vec4 C = vec4(0.211324865405187,	// (3.0-sqrt(3.0))/6.0
				0.366025403784439,	// 0.5*(sqrt(3.0)-1.0)
				-0.577350269189626,	// -1.0 + 2.0 * C.x
				0.024390243902439);	// 1.0 / 41.0

	// First corner
	vec2 i  = floor(v + dot(v, C.yy) );
	vec2 x0 = v -   i + dot(i, C.xx);

	// Other corners
	vec2 i1;
	i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
	vec4 x12 = x0.xyxy + C.xxzz;
	x12.xy -= i1;

	// Permutations
	i = mod289(i); // Avoid truncation effects in permutation
	vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
		+ i.x + vec3(0.0, i1.x, 1.0 ));

	vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
	m = m*m ;
	m = m*m ;

	// Gradients: 41 points uniformly over a line, mapped onto a diamond.
	// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

	vec3 x = 2.0 * fract(p * C.www) - 1.0;
	vec3 h = abs(x) - 0.5;
	vec3 ox = floor(x + 0.5);
	vec3 a0 = x - ox;

	// Normalise gradients implicitly by scaling m
	// Approximation of: m *= inversesqrt( a0*a0 + h*h );
	m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

	// Compute final noise value at P
	vec3 g;
	g.x  = a0.x  * x0.x  + h.x  * x0.y;
	g.yz = a0.yz * x12.xz + h.yz * x12.yw;
	return 130.0 * dot(m, g);
}

// Function 1017
float PhiNoise2(uvec2 uv)
{
    uvec2 uv0 = uv;
    // flip every other tile to reduce anisotropy
    if(((uv.x ^ uv.y) & 4u) == 0u) uv = uv.yx;
	if(((uv.x       ) & 4u) == 0u) uv.x = -uv.x;// more iso but also more low-freq content
    
    // constants of 2d Roberts sequence rounded to nearest primes
    const uint r0 = 3242174893u;// prime[(2^32-1) / phi_2  ]
    const uint r1 = 2447445397u;// prime[(2^32-1) / phi_2^2]
    
    // h = high-freq dither noise
    uint h = (uv.x * r0) + (uv.y * r1);
    
    uint l;
    {
        uv = uv0 >> 2u;
        //uv.x = -uv.x;
        
        if(((uv.x ^ uv.y) & 4u) == 0u) uv = uv.yx;
        if(((uv.x       ) & 4u) == 0u) uv.x = -uv.x;

        uint h = (uv.x * r0) + (uv.y * r1);
			 h = h ^ 0xE2E17FDCu;
        
        l = h;
        
        {
            uv = uv0 >> 4u;
            if(((uv.x ^ uv.y) & 4u) == 0u) uv = uv.yx;
            if(((uv.x       ) & 4u) == 0u) uv.x = -uv.x;

            uint h = (uv.x * r0) + (uv.y * r1);
                 h = h ^ 0x1B98264Du;

            l += h;
    	}
    }
    
    // combine low and high
    return float(l + h*1u) * (1.0 / 4294967296.0);
}

// Function 1018
vec2 NoisePrecise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
//	vec3 f2 = f*f; f = f*f2*(10.0-15.0*f+6.0*f2);

	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z);

	vec4 rg = mix( mix(
				textureLod( iChannel0, (uv+0.5)/256.0, 0.0 ),
				textureLod( iChannel0, (uv+vec2(1,0)+0.5)/256.0, 0.0 ),
				f.x ),
				  mix(
				textureLod( iChannel0, (uv+vec2(0,1)+0.5)/256.0, 0.0 ),
				textureLod( iChannel0, (uv+1.5)/256.0, 0.0 ),
				f.x ),
				f.y );
				  

	return mix( rg.yw, rg.xz, f.z );
}

// Function 1019
float simplex(vec3 v)
  {
  const vec2  C = vec2(1.0/6.00, 1.0/3.00) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 3.22);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 200.0 *sin(iTime * 0.1) * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
  }

// Function 1020
vec3 Voronoi(in vec3 p, in vec3 rd){
    
    // One of Tomkh's snippets that includes a wrap to deal with
    // larger numbers, which is pretty cool.

 
    vec3 n = floor(p);
    p -= n + .6;
 
    
    // Storage for all sixteen hash values. The same set of hash values are
    // reused in the second pass, and since they're reasonably expensive to
    // calculate, I figured I'd save them from resuse. However, I could be
    // violating some kind of GPU architecture rule, so I might be making 
    // things worse... If anyone knows for sure, feel free to let me know.
    //
    // I've been informed that saving to an array of vectors is worse.
    //vec2 svO[3];
    
    // Individual Voronoi cell ID. Used for coloring, materials, etc.
    cellID = vec3(0); // Redundant initialization, but I've done it anyway.

    // As IQ has commented, this is a regular Voronoi pass, so it should be
    // pretty self explanatory.
    //
    // First pass: Regular Voronoi.
	vec3 mo, o;
    
    // Minimum distance, "smooth" distance to the nearest cell edge, regular
    // distance to the nearest cell edge, and a line distance place holder.
    float md = 9., lMd = 9., lMd2 = 9., lnDist, d;
    
    // Note the ugly "gIFrame" hack. The idea is to force the compiler not
    // to unroll the loops, thus keep the program size down... or something. 
    // GPU compilation is not my area... Come to think of it, none of this
    // is my area. :D
    for( int k=min(-3, gIFrame); k<=3; k++ ){
    for( int j=min(-3, gIFrame); j<=3; j++ ){
    for( int i=min(-3, gIFrame); i<=3; i++ ){
    
        o = vec3(i, j, k);
        o += hash33(n + o) - p;
        // Saving the hash values for reuse in the next pass. I don't know for sure,
        // but I've been informed that it's faster to recalculate the had values in
        // the following pass.
        //svO[j*3 + i] = o; 
  
        // Regular squared cell point to nearest node point.
        d = dot(o, o); 

        if( d<md ){
            
            md = d;  // Update the minimum distance.
            // Keep note of the position of the nearest cell point - with respect
            // to "p," of course. It will be used in the second pass.
            mo = o; 
            cellID = vec3(i, j, k) + n; // Record the cell ID also.
        }
       
    }
    }
    }
    
    // Second pass: Distance to closest border edge. The closest edge will be one of the edges of
    // the cell containing the closest cell point, so you need to check all surrounding edges of 
    // that cell, hence the second pass... It'd be nice if there were a faster way.
    for( int k=min(-4, gIFrame); k<=4; k++ ){
    for( int j=min(-4, gIFrame); j<=4; j++ ){
    for( int i=min(-4, gIFrame); i<=4; i++ ){
        
        // I've been informed that it's faster to recalculate the hash values, rather than 
        // access an array of saved values.
        o = vec3(i, j, k);
        o += hash33(n + o) - p;
        // I went through the trouble to save all sixteen expensive hash values in the first 
        // pass in the hope that it'd speed thing up, but due to the evolving nature of 
        // modern architecture that likes everything to be declared locally, I might be making 
        // things worse. Who knows? I miss the times when lookup tables were a good thing. :)
        // 
        //o = svO[j*3 + i];
        
        // Skip the same cell... I found that out the hard way. :D
        if( dot(o - mo, o - mo)>.00002 ){ 
            
            // This tiny line is the crux of the whole example, believe it or not. Basically, it's
            // a bit of simple trigonometry to determine the distance from the cell point to the
            // cell border line. See IQ's article for a visual representation.
            lnDist = dot(0.6*(o + mo), normalize(o - mo));
            
            // Abje's addition. Border distance using a smooth minimum. Insightful, and simple.
            //
            // On a side note, IQ reminded me that the order in which the polynomial-based smooth
            // minimum is applied effects the result. However, the exponentional-based smooth
            // minimum is associative and commutative, so is more correct. In this particular case, 
            // the effects appear to be negligible, so I'm sticking with the cheaper polynomial-based 
            // smooth minimum, but it's something you should keep in mind. By the way, feel free to 
            // uncomment the exponential one and try it out to see if you notice a difference.
            //
            // // Polynomial-based smooth minimum.
            //lMd = smin(lMd, lnDist, lnDist*.75); //lnDist*.75
            //
            // Exponential-based smooth minimum. By the way, this is here to provide a visual reference 
            // only, and is definitely not the most efficient way to apply it. To see the minor
            // adjustments necessary, refer to Tomkh's example here: Rounded Voronoi Edges Analysis - 
            // https://www.shadertoy.com/view/MdSfzD
            lMd = sminExp(lMd, lnDist, 26.); 
            
            // Minimum regular straight-edged border distance. If you only used this distance,
            // the web lattice would have sharp edges.
            lMd2 = min(lMd2, lnDist);
        }

    }
    }
    }

    // Return the smoothed and unsmoothed distance. I think they need capping at zero... but 
    // I'm not positive.
    return max(vec3(lMd, lMd2, md), 0.);
}

// Function 1021
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture(iChannel0, (uv+0.5)/256.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 1022
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return 1. - 0.82*mix( rg.x, rg.y, f.z );
}

// Function 1023
float noise3D( in vec3 x )
{
    x*= 800.;
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy + vec2(37.0, 17.0) * p.z) + f.xy;
	vec2 rg = textureLod( iChannel1, (uv+ 0.5) / 256.0, 0.0).yx;
	return mix(rg.x, rg.y, f.z);
}

// Function 1024
float Noise1(float x, float seed)
{
    vec2 uv = vec2(x, seed);
    vec2 corner = floor(uv);
	float c00 = N2(corner + vec2(0.0, 0.0));
	float c10 = N2(corner + vec2(1.0, 0.0));
    
    float diff = fract(uv.x);
    
    diff = diff * diff * (3.0 - 2.0 * diff);
    
    return mix(c00, c10, diff) - 0.5;
}

// Function 1025
float noise( in vec3 f )
{
    vec3 p = floor(f);
    f = fract(f);
    f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 1026
float noise (in vec2 uv) {
    vec2 i = floor(uv);
    vec2 f = fract(uv);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Function 1027
float noise(vec2 p,float r2,float z,vec2 q){vec2 h=hash(q,z),d=unskew(q)-p;z=d.x*d.x+d.y*d.y
 ;return pow(max(0.,r2-z),4.)*dot(d,h);}

// Function 1028
float noisetex(in vec2 x){
    vec2 p = floor(x);
    vec2 f = fract(x);
	f = f*f*(3.-2.*f);    
    if(x.y/iResolution.y<.5){
        float a =  noisergba((p+f+.5)/256.).y;
        return a*8.;}
//replaces
	return texture( iChannel0, (p + f + 0.5)/256.0, -100.0 ).x;
    
}

// Function 1029
float noise3d (vec3 p)
{
    vec3 u = floor (p);
    vec3 v = fract (p);
    
    v = v * v * (3. - 2. * v);

    float n = u.x + u.y * 57. + u.z * 113.;
    float a = hash (n);
    float b = hash (n + 1.);
    float c = hash (n + 57.);
    float d = hash (n + 58.);

    float e = hash (n + 113.);
    float f = hash (n + 114.);
    float g = hash (n + 170.);
    float h = hash (n + 171.);

    float result = mix (mix (mix (a, b, v.x),
                             mix (c, d, v.x),
                             v.y),
                        mix (mix (e, f, v.x),
                             mix (g, h, v.x),
                             v.y),
                        v.z);

    return result;
}

// Function 1030
float noise_3(in vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);	
	vec3 u = f*f*(3.0-2.0*f);
    
    vec2 ii = i.xy + i.z * vec2(5.0);
    float a = hash12( ii + vec2(0.0,0.0) );
	float b = hash12( ii + vec2(1.0,0.0) );    
    float c = hash12( ii + vec2(0.0,1.0) );
	float d = hash12( ii + vec2(1.0,1.0) ); 
    float v1 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
    
    ii += vec2(5.0);
    a = hash12( ii + vec2(0.0,0.0) );
	b = hash12( ii + vec2(1.0,0.0) );    
    c = hash12( ii + vec2(0.0,1.0) );
	d = hash12( ii + vec2(1.0,1.0) );
    float v2 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
        
    return max(mix(v1,v2,u.z),0.0);
}

// Function 1031
float snoise(vec3 uv, float res)
{
  const vec3 s = vec3(1e0, 1e2, 1e3);
  
  uv *= res;
  
  vec3 uv0 = floor(mod(uv, res))*s;
  vec3 uv1 = floor(mod(uv+vec3(1.), res))*s;
  
  vec3 f = fract(uv); f = f*f*(3.0-2.0*f);

  vec4 v = vec4(uv0.x+uv0.y+uv0.z, uv1.x+uv0.y+uv0.z,
              uv0.x+uv1.y+uv0.z, uv1.x+uv1.y+uv0.z);

  vec4 r = fract(sin(v*1e-1)*1e3);
  float r0 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
  
  r = fract(sin((v + uv1.z - uv0.z)*1e-1)*1e3);
  float r1 = mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y);
  
  return mix(r0, r1, f.z)*2.-1.;
}

// Function 1032
float value_noise(vec3 p)
{
    vec3 pi = floor(p);
    vec3 pf = p - pi;
    
    vec3 w = pf * pf * (3.0 - 2.0 * pf);
    
    return 	mix(
        		mix(
        			mix(hash31(pi + vec3(0, 0, 0)), hash31(pi + vec3(1, 0, 0)), w.x),
        			mix(hash31(pi + vec3(0, 0, 1)), hash31(pi + vec3(1, 0, 1)), w.x), 
                    w.z),
        		mix(
                    mix(hash31(pi + vec3(0, 1, 0)), hash31(pi + vec3(1, 1, 0)), w.x),
        			mix(hash31(pi + vec3(0, 1, 1)), hash31(pi + vec3(1, 1, 1)), w.x), 
                    w.z),
        		w.y);
}

// Function 1033
float noise(vec3 p){
  	vec3 ip=floor(p);p-=ip; 
    vec3 s=vec3(7,157,113);
    vec4 h=vec4(0.,s.yz,s.y+s.z)+dot(ip,s);
    p=p*p*(3.-2.*p); 
    h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
    h.xy=mix(h.xz,h.yw,p.y);
    return mix(h.x,h.y,p.z); 
}

// Function 1034
float NoiseFBM(in vec3 p, float numCells, int octaves)
{
	float f = 0.0;
    
	// Change starting scale to any integer value...
    p = mod(p, vec3(numCells));
	float amp = 0.5;
    float sum = 0.0;
	
	for (int i = 0; i < octaves; i++)
	{
		f += CellNoise(p, numCells) * amp;
        sum += amp;
		amp *= 0.5;

		// numCells must be multiplied by an integer value...
		numCells *= 2.0;
	}

	return f / sum;
}

// Function 1035
vec3 sampleVoronoi(vec2 uv, float size)
{	
	float nbPoints = size*size;
	float m = floor(uv.x*size);
	float n = floor(uv.y*size);			
	
	vec3 voronoiPoint = vec3(0.);;			
	float dist2Max = 3.;
	const float _2PI = 6.28318530718;
	
	for (int i=-1; i<2; i++)
	{ 
		for (int j=-1; j<2; j++)
		{
			vec2 coords = vec2(m+float(i),n+float(j));																
			float phase = _2PI*(size*coords.x+coords.y)/nbPoints;
			vec2 delta = 0.25*vec2(sin(iTime+phase), cos(iTime+phase));
			vec2 point = (coords +vec2(0.5) + delta)/size;
			vec2 dir = uv-point;
			float dist2 = Manhatan(dir);										
			float t = 0.5*(1.+sign(dist2Max-dist2));
			vec3 tmp = vec3(coords/size,dist2);
			dist2Max = mix(dist2Max,dist2,t);
			voronoiPoint = mix(voronoiPoint,tmp,t);				
		}
	}	
	return voronoiPoint;		
}

// Function 1036
float perlin_b (vec3 n)
{
    vec3 base = vec3(n.x, n.y, floor(n.z * 64.0) * 0.015625);
    vec3 dd = vec3(0.015625, 0.0, 0.0);
    vec3 p = (n - base) *  64.0;
    float front = perlin_a(base + dd.yyy);
    float back = perlin_a(base + dd.yyx);
    return mix(front, back, p.z);
}

// Function 1037
vec4 noise(vec2 p){return texture(iChannel0,p/iChannelResolution[0].xy);}

// Function 1038
vec3 voronoi( in vec2 x, float m) {
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mr;

    float md = 8.0;
    for( float j=-1.; j<=1.; j++ )
        for( float i=-1.; i<=1.; i++ ) {
            vec2 g = vec2(i,j);
	    	vec2 seed = hash2( n + g );
            seed = .5 + .3 * cos(iTime*m * 6.283 * seed);
            vec2 r = g + seed - f;
            float d = dot(r,r);
            if( d<md ) {
                md = d;
                mr = r;
            }
    }
    //----------------------------------
    // second pass: distance to borders,
    //----------------------------------
    md = 8.0;
    for( float j=-1.; j<=1.; j++ )
        for( float i=-1.; i<=1.; i++ ) {
            vec2 g = vec2(i,j);
	    	vec2 seed = hash2( n + g );
            seed = .5 + .3 * cos(iTime*m * 6.283 * seed);
		    vec2 r = g + seed - f;

            if( dot(mr-r,mr-r)>.0001 ) { // skip the same cell 
                // smooth minimum for rounded borders
                // apparently I need the max(,.0) to filter out weird values
                md = max(smin(md,dot( 0.5*(mr+r), normalize(r-mr) ),.25),.0);
            }
        }
    

    return vec3( mr, sqrt(md));
}

// Function 1039
float noise_func( vec2 xy )
{
    float x = xy.x;
    float y = xy.y;
    float rx = x + sin(y/43.0)*43.0;
    float ry = y + sin(x/37.0)*37.0;
    
    float f = sin(rx/11.2312) + sin(ry/14.4235);
    
    f = f*0.5 + sin(rx/24.0) * sin(ry/24.0);
    
    rx += sin(y/210.23)*210.23;
    ry += sin(x/270.0)*270.0;
    
    f = f*0.5 + sin(rx/65.0) * sin(ry/65.0);
    f = f*0.5 + sin(rx/165.0) * sin(ry/165.0);
    
    return f / 1.0;
}

// Function 1040
float noise3(vec3 p)
{
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.zzzz;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));

    vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}

// Function 1041
float noise(vec3 p){
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.zzzz;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));

    vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}

// Function 1042
vec4 voronoi(vec2 x) {
  vec2 n = floor(x);
  vec2 f = fract(x);

  vec4 m = vec4(8.0);
  for(int j=-1; j<=1; j++)
  for(int i=-1; i<=1; i++)
  {
    vec2  g = vec2(float(i), float(j));
    vec2  o = hash2(n + g);
    vec2  r = g - f + o;
    float d = dot(r, r);
    if(d<m.x)
    {
      m = vec4(d, o.x + o.y, r);
    }
  }

  return vec4(sqrt(m.x), m.yzw);
}

// Function 1043
float perlinNoise(vec2 pos, vec2 scale, float seed)
{
    // based on Modifications to Classic Perlin Noise by Brian Sharpe: https://archive.is/cJtlS
    pos *= scale;
    vec4 i = floor(pos).xyxy + vec2(0.0, 1.0).xxyy;
    vec4 f = (pos.xyxy - i.xyxy) - vec2(0.0, 1.0).xxyy;
    i = mod(i, scale.xyxy) + seed;

    // grid gradients
    vec4 gradientX, gradientY;
    multiHash2D(i, gradientX, gradientY);
    gradientX -= 0.49999;
    gradientY -= 0.49999;

    // perlin surflet
    vec4 gradients = inversesqrt(gradientX * gradientX + gradientY * gradientY) * (gradientX * f.xzxz + gradientY * f.yyww);
    // normalize: 1.0 / 0.75^3
    gradients *= 2.3703703703703703703703703703704;
    vec4 lengthSq = f * f;
    lengthSq = lengthSq.xzxz + lengthSq.yyww;
    vec4 xSq = 1.0 - min(vec4(1.0), lengthSq); 
    xSq = xSq * xSq * xSq;
    return dot(xSq, gradients);
}

// Function 1044
float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );	

    // bteitler: This is equivalent to the "smoothstep" interpolation function.
    // This is a smooth wave function with input between 0 and 1
    // (since it is taking the fractional part of <p>) and gives an output
    // between 0 and 1 that behaves and looks like a wave.  This is far from obvious, but we can graph it to see
    // Wolfram link: http://www.wolframalpha.com/input/?i=plot+x*x*%283.0-2.0*x%29+from+x%3D0+to+1
    // This is used to interpolate between random points.  Any smooth wave function that ramps up from 0 and
    // and hit 1.0 over the domain 0 to 1 would work.  For instance, sin(f * PI / 2.0) gives similar visuals.
    // This function is nice however because it does not require an expensive sine calculation.
    vec2 u = f*f*(3.0-2.0*f);

    // bteitler: This very confusing looking mish-mash is simply pulling deterministic random values (between 0 and 1)
    // for 4 corners of the grid square that <p> is inside, and doing 2D interpolation using the <u> function
    // (remember it looks like a nice wave!) 
    // The grid square has points defined at integer boundaries.  For example, if <p> is (4.3, 2.1), we will 
    // evaluate at points (4, 2), (5, 2), (4, 3), (5, 3), and then interpolate x using u(.3) and y using u(.1).
    return -1.0+2.0*mix( 
                mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), 
                        u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), 
                        u.x), 
                u.y);
}

// Function 1045
float Noisefv3 (vec3 p)
{
  vec4 t1, t2;
  vec3 ip, fp;
  float q;
  ip = floor (p);
  fp = fract (p);
  fp = fp * fp * (3. - 2. * fp);
  q = dot (ip, cHashA3);
  t1 = Hashv4f (q);
  t2 = Hashv4f (q + cHashA3.z);
  return mix (mix (mix (t1.x, t1.y, fp.x), mix (t1.z, t1.w, fp.x), fp.y),
              mix (mix (t2.x, t2.y, fp.x), mix (t2.z, t2.w, fp.x), fp.y), fp.z);
}

// Function 1046
float valueNoiseSimple3D(in vec3 vl) {
    const vec2 helper = vec2(0., 1.);
    vec3 grid = floor(vl);
    vec3 interp = smoothstep(vec3(0.), vec3(1.), fract(vl));
    
    float interpY0 = mix(mix(rand3(grid),
                         	 rand3(grid + helper.yxx),
                         	 interp.x),
                        mix(rand3(grid + helper.xyx),
                         	rand3(grid + helper.yyx),
                         	interp.x),
                        interp.y);
    
    
    float interpY1 = mix(mix(rand3(grid + helper.xxy),
                         	 rand3(grid + helper.yxy),
                         	interp.x),
                        mix(rand3(grid + helper.xyy),
                         	rand3(grid + helper.yyy),
                         	interp.x),
                        interp.y);
    
    return -1. + 2.*mix(interpY0, interpY1, interp.z);
}

// Function 1047
float noise(in vec3 p)
{
    vec3 ip = floor(p);
    vec3 fp = fract(p);
	fp = fp*fp*(3.0-2.0*fp);
	vec2 tap = (ip.xy+vec2(37.0,17.0)*ip.z) + fp.xy;
	vec2 rz = textureLod( iChannel0, (tap+0.5)/256.0, 0.0 ).yx;
	return mix( rz.x, rz.y, fp.z );
}

// Function 1048
vec2 noise(in vec2 uv)
{
    return vec2(sin(uv.x),cos(uv.y));
}

// Function 1049
float layeredPerlin(vec2 uv, int seed) {
	int l = amplitudes.length();
    float result = 0.;
    float total = 0.;
    float octave = float(firstOctave);
    for (int i = 0; i < l; i++) {
        seed = hash(seed);
    	result += perlin(uv * pow(2., -octave), seed) * amplitudes[i];
        total += amplitudes[i];
        octave -= 1.;
    }
    return -result/total;
}

// Function 1050
float fNoise(vec2 p){
    mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
    float f=0.;
    float str=.5;
    //change the number of iterations in this loop to make the contours more or less wobbly
    for(int i=0;i<4;i++){
        f += str*noise(p);
    	str/=2.;
        p = m*p;
    }
	return 0.5 + 0.5*f;
}

// Function 1051
v0 noise01(v1 p){return clamp((noise(p)+.5)*.5, 0.,1.);}

// Function 1052
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    float res = mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                    mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);
    return -1.0 + 2.0*res;
}

// Function 1053
vec2 Noise( in vec3 x ){return x.xy;return mix(x.yx,x.xz, x.y);}

// Function 1054
vec3 noiseq(float x)
{
    return (noise(x)+noise(x+10.25)+noise(x+20.5)+noise(x+30.75))*0.25;
}

// Function 1055
float FbmNoise(vec2 p)
{
  const float octaves = 8.0;
  const float lacunarity = 2.0;
  const float H = 0.5;

  float value = 0.0, k = 0.0;
  for (float i = 0.0; i < octaves; ++ i) {
    value += Noise(p,0.0) * pow(lacunarity, -H * i);
    p *= lacunarity;
    ++k;
  }

  float remainder = fract(octaves);
  if (remainder >= 0.0) {
    value -= remainder * Noise(p,0.0) - pow(lacunarity, -H * k);
  }
  return value;
}

// Function 1056
float voronoi(vec2 p) {
    vec2 n = floor(p);
    vec2 f = fract(p);
    float md = 5.0;
    vec2 m = vec2(0.0);
    for (int i = -1;i<=1;i++) {
        for (int j = -1;j<=1;j++) {
            vec2 g = vec2(i, j);
            vec2 o = hash2(n+g);
            o = 0.5+0.5*sin(iTime+5.038*o);
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d<md) {
              md = d;
              m = n+g+o;
            }
        }
    }
    return md;
}

// Function 1057
vec4 gnoised(in vec3 x, int seed)
{
    // grid
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    #if 1
    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
    vec3 iu = 1.0 - u;
    #else
    // cubic interpolant
    vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
    #endif    
    
    // gradients
    vec3 ga = grad( p+vec3(0.0,0.0,0.0), seed );
    vec3 gb = grad( p+vec3(1.0,0.0,0.0), seed );
    vec3 gc = grad( p+vec3(0.0,1.0,0.0), seed );
    vec3 gd = grad( p+vec3(1.0,1.0,0.0), seed );
    vec3 ge = grad( p+vec3(0.0,0.0,1.0), seed );
	vec3 gf = grad( p+vec3(1.0,0.0,1.0), seed );
    vec3 gg = grad( p+vec3(0.0,1.0,1.0), seed );
    vec3 gh = grad( p+vec3(1.0,1.0,1.0), seed );
    
    // projections
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
	
    // interpolations
    return vec4( va + u.x*(vb-va) + u.y*(vc-va) + u.z*(ve-va) + u.x*u.y*(va-vb-vc+vd) + u.y*u.z*(va-vc-ve+vg) + u.z*u.x*(va-vb-ve+vf) + (-va+vb+vc-vd+ve-vf-vg+vh)*u.x*u.y*u.z,    // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.z*(ge-ga) + u.x*u.y*(ga-gb-gc+gd) + u.y*u.z*(ga-gc-ge+gg) + u.z*u.x*(ga-gb-ge+gf) + (-ga+gb+gc-gd+ge-gf-gg+gh)*u.x*u.y*u.z +   // derivatives
                 du * (vec3(vb,vc,ve) - va + u.yzx*vec3(va-vb-vc+vd,va-vc-ve+vg,va-vb-ve+vf) + u.zxy*vec3(va-vb-ve+vf,va-vb-vc+vd,va-vc-ve+vg) + u.yzx*u.zxy*(-va+vb+vc-vd+ve-vf-vg+vh) ));
}

// Function 1058
float noise2d(vec2 p)
{
    vec2 f = fract(p);
    p = floor(p);
    f = f*f*(3.0-2.0*f);
    
    float res = mix(mix( hash12(p),  		    hash12(p + vec2(1,0)),f.x),
                    mix( hash12(p + vec2(0,1)), hash12(p + vec2(1,1)),f.x),f.y);
    return res;
}

// Function 1059
float noise (vec3 n) 
{ 
	return fract(sin(dot(n, vec3(95.43583, 93.323197, 94.993431))) * 65536.32);
}

// Function 1060
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    return mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
               mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);
}

// Function 1061
float Noiseff (float p)
{
  vec2 t;
  float ip, fp;
  ip = floor (p);  fp = fract (p);
  fp = fp * fp * (3. - 2. * fp);
  t = Hashv2f (ip);
  return mix (t.x, t.y, fp);
}

// Function 1062
float noise( float x )
{
    return fract(sin(1371.1*x)*43758.5453);
}

// Function 1063
vec4 texNoise(vec2 uv,sampler2D tex ){ float f = 0.; f+=texture(tex, uv*.125).r*.5; f+=texture(tex,uv*.25).r*.25; //Funciton simulating the perlin noise texture we have in Bonzomatic shader editor, written by yx
                       f+=texture(tex,uv*.5).r*.125; f+=texture(tex,uv*1.).r*.125; f=pow(f,1.2);return vec4(f*.45+.05);}

// Function 1064
float simple_noise(in vec3 uv, in float shift_by) {
  return noise(uv * 10.0, 5.0, 0.75, 0.75, shift_by);
}

// Function 1065
float noise(vec3 pos){
    return texture(iChannel1, pos).r;
}

// Function 1066
float terrain_surface_noise(vec2 p, int iter) {
  float res = 0.0;

  // scale up the domain first (i.e. start terrain on low frequency)
  p *= 0.125; // * 1/8

  // amplitude, amp total, accumulative result
  float amp = 1.0, amp_sum = 0.0, accum_res = 0.0;

  for (int i=0; i<iter; i++) {
    accum_res += n2d(p)*amp;

    // scaling and skewing
    p = mat2(1.0, -0.95, 0.60, 1.0) * p*3.0;

    // adding up the amp
    amp_sum += amp;

    // scaling down the amp
    amp *= 0.4 - (float(i)*0.05);
  }

  res = accum_res / amp_sum;

  return res;
}

// Function 1067
float snoise(vec3 v){
  const vec2  C = vec2(0.166666667, 1./3.) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 = v - i + dot(i, C.xxx) ;
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod(i,289.);
  vec4 p = permute( permute( permute(
	  i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
	+ i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
	+ i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
  vec3 ns = 0.142857142857 * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = floor(j - 7.0 * x_ ) *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);
  vec4 norm = inversesqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m * m;
  return .5 + 12.0 * dot( m, vec4( dot(p0,x0), dot(p1,x1),dot(p2,x2), dot(p3,x3) ) );
}

// Function 1068
vec4 simplexWeave(vec3 q){
    
   
    // SIMPLEX GRID SETUP
    
    vec2 p = q.xy; // 2D coordinate on the XY plane.
   
    vec2 s = floor(p + (p.x + p.y)*.36602540378); // Skew the current point.
    
    p -= s - (s.x + s.y)*.211324865; // Use it to attain the vector to the base vertex (from p).
    
    // Determine which triangle we're in. Much easier to visualize than the 3D version.
    float i = p.x < p.y? 1. : 0.; // Apparently, faster than: i = step(p.y, p.x);
    vec2 ioffs = vec2(1. - i, i);
    
    // Vectors to the other two triangle vertices.
    vec2 p1 = p - ioffs + .2113248654, p2 = p - .577350269; 
    

    
    // THE WEAVE PATTERN
    
    // A random value -- based on the triangle vertices, and ranging between zero 
    // and one.
    float dh = hash21((s*3. + ioffs + 1.));
    
 
    // Based on the unique random value for the triangle cell, rotate the tile.
    // In effect, we're swapping vertex positions... I figured it'd be cheaper,
    // but I didn't think about it for long, so there could be a faster way. :)
    //
    if(dh<1./3.) { // Rotate by 60 degrees.
        vec2 ang = p;
        p = p1, p1 = p2, p2 = ang;
        
    }
    else if(dh<2./3.){ // Rotate by 120 degrees.
        vec2 ang = p;
        p = p2, p2 = p1, p1 = ang;
    }
     
     
    
    // Angles subtended from the current position to each of the three vertices... There's probably a 
    // symmetrical way to make just one "atan" call. Anyway, you can use these angular values to create 
    // patterns that follow the contours. In this case, I'm using them to create some cheap repetitious lines.
    vec3 a = vec3(atan(p.y, p.x), atan(p1.y, p1.x), atan(p2.y, p2.x));

    // The torus rings. 

    // For symmetry, we want the middle of the torus ring to cut dirrectly down the center
    // of one of the equilateral triangle sides, which is half the distance from one of the
    // vertices to the other. Add ".1" to it to see that it's necessary.
    float mid = dist((p2 - p))*.5;
 
    // Interspercing some pylons and bolts around the arcs using
    // a standard fixed repeat partitioning of angular components.
    
    const float aNum = 6.;
    vec3 ia = (floor(a/6.283*aNum) + .5)/aNum;
    ia += .25/aNum; // A hack to move the objects to a better place.
    
    vec2 px = rot2(ia.x*6.283)*p;
    px.x -= mid;

    vec2 py = rot2(ia.y*6.283)*p1;
    py.x -= mid;

    vec2 pz = rot2(ia.z*6.283)*p2;
    pz.x -= mid;
    
    px = abs(px);
    py = abs(py);
    pz = abs(pz);
   
    // A repeat trick to move the radial component of an object to a set distance
    // on both sides of the radial origin... A bit hard to decribe, but if you comment
    // the following lines out, you'll see what I mean. :)
    px.x = abs(px.x - .08);
    py.x = abs(py.x - .08);
    pz.x = abs(pz.x - .08);

    // Bolts.
    float cdx = tube4(px) - .02;
    float cdy = tube4(py) - .02;
    float cdz = tube4(pz) - .02;
 
    // Pylons.
    float bdx = pylon(px);
    float bdy = pylon(py);
    float bdz = pylon(pz);
     
///
    

    // Relative arc heights -- Based on the angle, and arranged to vary between
    // zero and one.
    a = sin(a*3. - 6.283/9.)*.5 + .5;

    q.z -= .5;
    
    
    // Arc tile calculations. 
    vec2 d1 = vec2(1e5), d2 = vec2(1e5), d3 = vec2(1e5);
    
    // The three arc sweeps. By the way, you can change the define above the
    // "tubeOuter" function to produce a hexagonal sweep, etc.
    float depth = .125;
    vec2 v1 = vec2(tubeOuter(p.xy) - mid, q.z - depth); // Bottom level arc.
    // This arc sweeps between the bottom and top levels.
    vec2 v2 = vec2(tubeOuter(p1.xy) - mid, q.z + depth*a.y - depth); 
    vec2 v3 = vec2(tubeOuter(p2.xy) - mid, q.z); // Top level arc.
    
    // Shaping the poloidal coordinate vectors above into rails.
    d1 = tube(v1);
    d2 = tube(v2);
    d3 = tube(v3);
    
    
    // Bolts.
    cdx = max(cdx, abs(q.z) - .03);
    cdy = max(cdy, abs(q.z) - .03);
    cdz = max(cdz, abs(q.z) - .03);
   
    q.z -= .475;//.465;
   
    // Pylons.
    bdx = max(bdx, abs(q.z) - .45);
    bdy = max(bdy, abs(q.z) - .45);
    bdz = max(bdz, abs(q.z) - .45);
    bdx = min(min(bdx, bdy), bdz);

    


    
    //////
    // The three distance field functions: Stored in cir.x, cir.y and cir.z.
    vec3 cir = vec3(dist(p), dist(p1), dist(p2));
    // Equivalent to: vec3 tor =  cir - mid - tw; tor = max(tor, -(cir - mid + tw));
    vec3 tor =  abs(cir - mid) - .21;
    
    // Optional floor holes that match the bump pattern. I like the extra detail, but 
    // figured it distracted from weave pattern itself, so left it as an option.
    float hole0 = 0.;
    #ifdef FLOOR_HOLES
    hole0 = -min(tor.x, tor.y);
    #endif
    
    // Bolts.
    d1.x = min(d1.x, cdx);
    d2.x = min(d2.x, cdy);
    d3.x = min(d3.x, cdz);
    
   
    // Obtaining the minimum of the center and outside rail objects.
    d1.xy = min(min(d1.xy, d2.xy), d3.xy);
    
   
    // Return the individual simplex weave object values:
    // Holes, pylons, rails and bolts, and center strip.
    return vec4(hole0, bdx, d1.xy);
 

}

// Function 1069
float TileableNoiseFBM(in vec3 p, float numCells, int octaves)
{
	float f = 0.0;
    
	// Change starting scale to any integer value...
    p = mod(p, vec3(numCells));
	float amp = 0.5;
    float sum = 0.0;
	
	for (int i = 0; i < octaves; i++)
	{
		f += TileableNoise(p, numCells) * amp;
        sum += amp;
		amp *= 0.5;

		// numCells must be multiplied by an integer value...
		numCells *= 2.0;
	}

	return f / sum;
}

// Function 1070
float FractalNoise(in vec2 xy)
{
	float w = .7;
	float f = 0.0;

	for (int i = 0; i < 4; i++)
	{
		f += Noise(xy) * w;
		w *= 0.5;
		xy *= 2.7;
	}
	return f;
}

// Function 1071
vec3 noised( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );

#if 1
    // quintic interpolation
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0);
#else
    // cubic interpolation
    vec2 u = f*f*(3.0-2.0*f);
    vec2 du = 6.0*f*(1.0-f);
#endif    
    
    vec2 ga = hash( i + vec2(0.0,0.0) );
    vec2 gb = hash( i + vec2(1.0,0.0) );
    vec2 gc = hash( i + vec2(0.0,1.0) );
    vec2 gd = hash( i + vec2(1.0,1.0) );
    
    float va = dot( ga, f - vec2(0.0,0.0) );
    float vb = dot( gb, f - vec2(1.0,0.0) );
    float vc = dot( gc, f - vec2(0.0,1.0) );
    float vd = dot( gd, f - vec2(1.0,1.0) );

    return vec3( va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd),   // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +  // derivatives
                 du * (u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va));
}

// Function 1072
float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );
    
    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( hash( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hash( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( hash( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1073
vec3 noised2( in vec2 x ) {
    vec2 p = floor(x);
    vec2 w = fract(x);

    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec2 du = 30.0*w*w*(w*(w-2.0)+1.0);

    float a = hash2(p+vec2(0,0));
    float b = hash2(p+vec2(1,0));
    float c = hash2(p+vec2(0,1));
    float d = hash2(p+vec2(1,1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;

    return vec3( -1.0+2.0*(k0 + k1*u.x + k2*u.y + k4*u.x*u.y), 
                      2.0* du * vec2( k1 + k4*u.y,
                                      k2 + k4*u.x ) );
}

// Function 1074
float noise(float p) {
    float i = floor(p);
    float f = fract(p);
    float t = f * f * (3.0 - 2.0 * f);
    return lerp(f * hash11(i), (f - 1.0) * hash11(i + 1.0), t);
}

// Function 1075
float triangleNoise(in vec2 p)
{
    float z=1.5;
    float z2=1.5;
	float rz = 0.;
    vec2 bp = 2.*p;
	for (float i=0.; i<=4.; i++ )
	{
        vec2 dg = tri2(bp*2.)*.8;
        dg *= rotate(0.314);
        p += dg/z2;

        bp *= 1.6;
        z2 *= .6;
		z *= 1.8;
		p *= 1.2;
        p*= trinoisemat;
        
        rz+= (tri(p.x+tri(p.y)))/z;
	}
	return rz;
}

// Function 1076
float noise3D(vec3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
	const vec3 s = vec3(7, 157, 113);
	
	vec3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride vector for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    
	p -= ip; // Cell's fractional component.
	
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: fract(sin(x)*largeNumber).
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
	
    // Interpolating along Y.
    h.xy = mix(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return mix(h.x, h.y, p.z); // Range: [0, 1].
	
}

// Function 1077
float noiseTex(vec2 uv, float seed, float octaves) {
    float v = 0.;
    uv += N21(seed);
    
    for (float i = 1.; i <= 11.; i++) {
    	v += Noise(uv) / i;
        uv *= 2.;
        
        if (i > octaves) break;
    }
    
    return v * .5;
}

// Function 1078
float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(4.0-3.0*f);
    return -2.0+3.0*mix( mix( hash( i + vec2(0.1,0.1) ), 
                     hash( i + vec2(2.0,0.1) ), u.x),
                mix( hash( i + vec2(0.1,2.0) ), 
                     hash( i + vec2(2.0,2.0) ), u.x), u.y);
}

// Function 1079
float noise( in vec2 p ) {
    vec2 i = floor(p), f = fract(p);
	vec2 u = f*f*(3.-2.*f);
    return mix( mix( dot( hash( i + vec2(0.,0.) ), f - vec2(0.,0.) ), 
                     dot( hash( i + vec2(1.,0.) ), f - vec2(1.,0.) ), u.x),
                mix( dot( hash( i + vec2(0.,1.) ), f - vec2(0.,1.) ), 
                     dot( hash( i + vec2(1.,1.) ), f - vec2(1.,1.) ), u.x), u.y);
}

// Function 1080
vec4 texNoise(vec2 uv){ float f = 0.; f+=texture(iChannel0, uv*.125).r*.5; f+=texture(iChannel0,uv*.25).r*.25; //MERCURTY SDF LIBRARY IS HERE OFF COURSE: http://mercury.sexy/hg_sdf/
                       f+=texture(iChannel0,uv*.5).r*.125; f+=texture(iChannel0,uv*1.).r*.125; f=pow(f,1.2);return vec4(f*.45+.05);}

// Function 1081
float tetraNoise(in vec3 p){
    
    // Skewing the cubic grid, then determining the first vertice and fractional position.
    vec3 i = floor(p + dot(p, vec3(0.333333)) );  p -= i - dot(i, vec3(0.166666)) ;
    
    // Breaking the skewed cube into tetrahedra with partitioning planes, then determining which side of the 
    // intersecting planes the skewed point is on. Ie: Determining which tetrahedron the point is in.
    vec3 i1 = step(p.yzx, p), i2 = max(i1, 1.0-i1.zxy); i1 = min(i1, 1.0-i1.zxy);    
    
    // Using the above to calculate the other three vertices. Now we have all four tetrahedral vertices.
    vec3 p1 = p - i1 + 0.166666, p2 = p - i2 + 0.333333, p3 = p - 0.5;
  

    // 3D simplex falloff.
    vec4 v = max(0.5 - vec4(dot(p,p), dot(p1,p1), dot(p2,p2), dot(p3,p3)), 0.0);
    
    // Dotting the fractional position with a random vector generated for each corner -in order to determine 
    // the weighted contribution distribution... Kind of. Just for the record, you can do a non-gradient, value 
    // version that works almost as well.
    vec4 d = vec4(dot(p, hash33(i)), dot(p1, hash33(i + i1)), dot(p2, hash33(i + i2)), dot(p3, hash33(i + 1.)));
    
     
    // Simplex noise... Not really, but close enough. :)
    return clamp(dot(d, v*v*v*8.)*1.732 + .5, 0., 1.); // Not sure if clamping is necessary. Might be overkill.

}

// Function 1082
vec2 rectvoronoi(in vec3 x, RTriplet3 trip) {
    ivec3 p = ivec3(floor( x ));
    vec3 f = fract( x );

    ivec3 mb;
    vec3 mr;
    float id = 1.0e20;
    const int range = 3;
    for( int k=-VORORANGE; k<=VORORANGE; k++ )
    for( int j=-VORORANGE; j<=VORORANGE; j++ )
    for( int i=-VORORANGE; i<=VORORANGE; i++ )
    {
        ivec3 b = ivec3( i, j, k );
        vec3 B = vec3(p + b);
        //vec3 rv = rand3(B, d, s );
        vec3 rv = rvec3(B, trip);
        //vec3 rv = RAND(B, 0.0, 1.0, s);
        vec3 r = vec3(b) - f + rv;
        float dis = length( r );

        if(dis < id) {
            mb = b;
            mr = r;
            id = dis;
        }
    }
    float bd = 1.0e20;
    for( int k=-VORORANGE; k<=VORORANGE; k++ )
    for( int j=-VORORANGE; j<=VORORANGE; j++ )
    for( int i=-VORORANGE; i<=VORORANGE; i++ )
    {
        ivec3 b = mb + ivec3( i, j, k );
        vec3 B = vec3(p + b);
        //vec3 rv = rand3(B, d, s );
        vec3 rv = rvec3(B, trip);
        //vec3 rv = RAND(B, 0.0, 1.0, s);
        vec3 r = vec3(b) - f + rv;
        float dis = dot( 0.5*(mr+r), normalize(r-mr) );

        bd = min(bd, dis);
    }
    return vec2(id, bd);
}

// Function 1083
vec2 fractalNoise2d(vec2 p, float t, float scale)
{
    return vec2(map5_value1(vec3(p*scale, t)), 
                map5_value1(vec3(p*scale+ vec2(100.0*scale), t)));   
}

// Function 1084
float mnoise(vec2 p)
{
    return noise(p) + 0.7*noise(p*1.34) + 0.6*noise(p*2.46) + 0.4*noise(p*3.75);
}

// Function 1085
float Voronoi(in vec2 p){
    
    // Partitioning the grid into unit squares and determining the fractional position.
	vec2 g = floor(p), o; p -= g;
	
    // "d.x" and "d.y" represent the closest and second closest distances
    // respectively, and "d.z" holds the distance comparison value.
	vec3 d = vec3(8); // 8., 2, 1.4, etc. 
    
     
    
    // A 4x4 grid sample is required for the smooth minimum version.
	for(int j = -1; j <= 2; j++){
		for(int i = -1; i <= 2; i++){
            
			o = vec2(i, j); // Grid reference.
             // Note the offset distance restriction in the hash function.
            o += hash22(g + o) - p; // Current position to offset point vector.
            
            // Distance metric. Unfortunately, the Euclidean distance needs
            // to be used for clean equidistant-looking cell border lines.
            // Having said that, there might be a way around it, but this isn't
            // a GPU intensive example, so I'm sure it'll be fine.
			d.z = length(o); 
            
            // Hacked in random ID. There'd be smarter ways to do this.
            #ifdef RAND_CELL_COLOR
            if(d.z<d.x) id = g + vec2(i, j);
            #endif
            
            // Up until this point, it's been a regular Voronoi example. The only
            // difference here is the the mild smooth minimum's to round things
            // off a bit. Replace with regular mimimum functions and it goes back
            // to a regular second order Voronoi example.
            d.y = max(d.x, smin(d.y, d.z, .4)); // Second closest point with smoothing factor.
            
            d.x = smin(d.x, d.z, .2); // Closest point with smoothing factor.
            
            // Based on IQ's suggestion - A commutative exponential-based smooth minimum.
            // This algorithm is just an approximation, so it doesn't make much of a difference,
            // but it's here anyway.
            //d.y = max(d.x, sminExp(d.y, d.z, 10.)); // Second closest point with smoothing factor.
            //d.x = sminExp(d.x, d.z, 20.); // Closest point with smoothing factor.

                       
		}
	}    
	
    // Return the regular second closest minus closest (F2 - F1) distance.
    return d.y - d.x;
    
}

// Function 1086
vec4 bluenoise( vec2 fc )
{
    return texture( iChannel2, fc / iChannelResolution[2].xy );
}

// Function 1087
float smoothNoise(vec2 p)
{
    return noise(p)/4.0+(noise(p+vec2(1.0,0.0))+noise(p-vec2(1.0,0.0))+noise(p+vec2(0.0,1.0))+noise(p-vec2(0.0,1.0)))/8.0+(noise(p+vec2(1.0,1.0))+noise(p+vec2(1.0,-1.0))+noise(p-vec2(1.0,-1.0))+noise(p-vec2(1.0,1.0)))/16.0;
}

// Function 1088
float fractalNoise(in vec3 loc) {
	float n = 0.0 ;
	for (int octave=1; octave<=NUM_OCTAVES; octave++) {	
		n = n + snoise(loc/float(octave*8)) ; 
	}
	return n ;
}

// Function 1089
float gnoise(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    float a = dot(hash22(i), f);
	float b = dot(hash22(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
	float c = dot(hash22(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
	float d = dot(hash22(i + vec2(1.0)), f - vec2(1.0));

    float c1 = b - a;
    float c2 = c - a;
    float c3 = d - c - b + a;

    vec2 q = f*f*f*(f*(f*6.0 - 15.0) + 10.0);
    
   	return a + q.x*c1 + q.y*c2 + q.x*q.y*c3;
}

// Function 1090
float snoise(vec3 v, bool showBug)
{ 
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g;
    
    if (showBug) {
        g = step(x0.yzx, x0.xyz);
    }
    else {
        // Old fix to show where the problem lies:
        // g = vec3(1.0, 1.0, 1.0);
        // if (x0.x < x0.y)
        //     g.x = 0.0;
        // if (x0.y < x0.z)
        //     g.y = 0.0;
        // if (x0.z <= x0.x) // <-- The fix is to have less-or-equal here. TODO: Optimize!
        //     g.z = 0.0;
        
        // Suggested fix from iq:
        // g.z = (x0.z == x0.x) ? 0.0 : g.z;
    
        // Suggested fix from Stefan Gustavson
        g = step(x0.yzz, x0.xyx);
        g.z = 1.0-g.z; // Ugly fix
    }
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    i = mod289(i); 
    vec4 p = permute( permute( permute( 
        i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
      + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);

    return 42.0 * dot( m*m*m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                     dot(p2,x2), dot(p3,x3) ) );
}

// Function 1091
float snoise(vec2 v)
{
  const vec4 C = vec4(0.211324865405187,0.366025403784439,-0.577350269189626,0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                	+ i.x + vec3(0.0, i1.x, 1.0 ));
  
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  
  return 130.0 * dot(m, g);		
}

// Function 1092
float noise(vec3 x, out vec2 g) {     // pseudoperlin improvement from foxes idea 
    vec2 g0,g1;
    float n = (noise3(x,g0)+noise3(x+11.5,g1)) / 2.;
    g = (g0+g1)/2.;
    return n;
}

// Function 1093
float bnoise( in float x )
{
    float i = floor(x);
    float f = fract(x);
    float s = sign(fract(x/2.0)-0.5);
    float k = 0.5+0.5*sin(i);
    return s*f*(f-1.0)*((16.0*k-4.0)*f*(f-1.0)-1.0);
}

// Function 1094
float vonronoise(vec2 x) {
	return iqnoise(x,1.0,1.0);
}

// Function 1095
float noise(vec2 p,float r2){float k=0.,o=1.;for(float z=.0;z<L;++z){k+=noise(p/o,r2,z)*o;o*=2.;}return k;}

// Function 1096
vec2 Noise( in ivec3 x )
{
	vec2 uv = vec2(x.xy)+vec2(37.0,17.0)*float(x.z);
	return textureLod( iChannel0, (uv+0.5)/256.0, 0.0 ).xz;
}

// Function 1097
float tweaknoise( vec3 p , bool step) {
    float d1 = smoothstep(grad/2.,-grad/2.,length(p)-.5),
          d2 = smoothstep(grad/1.,-grad/1.,abs(p.z)-.5),
          d=d1;
#if NOISE==1 // 3D Perlin noise
    float v = fbm(scale*p);
#elif NOISE==2 // Worley noise
    float v = (.9-scale*worley(scale*p).x);
#elif NOISE>=3 // trabeculum 3D
  #if VARIANT==0
    d = (1.-d1)*d2; 
  #elif VARIANT==2
    d=1.; //d=d2;
  #endif
    if (d<0.5) return 0.;
    grad=.8, scale = 7., thresh=.7+.2*(cos(.5*time)+.36*cos(.5*3.*time))/1.36;
    vec4 w=scale*worley(scale*p-vec3(0.,0.,3.*time)); 
    float v=1.-1./(1./(w.z-w.x)+1./(w.a-w.x)); // formula (c) Fabrice NEYRET - BSD3:mention author.
#  if SPIKES    
    v *= 1.-.9*(2.*fbm(10.*scale*p)-1.);
#  endif
#endif
    
    return (true)? smoothstep(thresh-grad/2.,thresh+grad/2.,v*d) : v*d;
}

// Function 1098
vec3 perlin31 (float p)
{
   float pi = floor(p);
   float pf = p - pi;
   return hash31(pi)*(1.-pf) +
          hash31(pi + 1.)*pf; 
}

// Function 1099
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);
	return textureLod( iChannel0, (uv+118.4)/256.0, 0.0 ).x;
}

// Function 1100
float PerlinSeries(vec2 pos)
{
    return perlin_noise(pos) + 0.4 * perlin_noise(pos * 2.0) + 0.2 * perlin_noise(pos * 4.0);
}

// Function 1101
float polarNoise(vec3 pos1)
{
    vec3 q = 8.0*pos1;
    float f = 0.0;
    f  = 0.5000*polarNoise0(q); q = m*q*2.;
    f += 0.2500*polarNoise0(q); q = m*q*2.;
    f += 0.1250*polarNoise0(q); q = m*q*2.;
    f += 0.0625*polarNoise0(q); q = m*q*2.;
    
    return f;
}

// Function 1102
float noise(vec2 n) {
    return fract(sin(dot(n, vec2(12.9898, 78.233)))* 43758.5453)* 2.0- 1.0;
}

// Function 1103
float noise(float t, float hiRes) {
	return bass(t, 1. - .35 * sectionSoundOsc(t, .05)) * .25 + fbm(t, .5 + .2 * sectionSoundOsc(t, .05), 7.25 + .5 * sectionSoundOsc(t, .025), (mix(1., 10. - 9. * sectionSoundOsc(t, .05), hiRes)));
}

// Function 1104
float noise( vec2 p, vec2 cycle ) { // -> [-1,1]
    vec2 i = floor(p),
         f = fract(p);
	vec2 u = smoothstep(0.,1.,f);
#if TILEABLE_NOISE
    return mix( mix( dot( hash( mod(i + vec2(0,0),cycle) ), f - vec2(0,0) ), 
                     dot( hash( mod(i + vec2(1,0),cycle) ), f - vec2(1,0) ), u.x),
                mix( dot( hash( mod(i + vec2(0,1),cycle) ), f - vec2(0,1) ), 
                     dot( hash( mod(i + vec2(1,1),cycle) ), f - vec2(1,1) ), u.x),
                u.y);
#else
    return mix( mix( dot( hash( i + vec2(0,0) ), f - vec2(0,0) ), 
                     dot( hash( i + vec2(1,0) ), f - vec2(1,0) ), u.x),
                mix( dot( hash( i + vec2(0,1) ), f - vec2(0,1) ), 
                     dot( hash( i + vec2(1,1) ), f - vec2(1,1) ), u.x),
                u.y);
#endif
}

// Function 1105
float gradientNoise(vec3 x, float freq)
{
    // grid
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    // quintic interpolant
    vec3 u = w * w * w * (w * (w * 6. - 15.) + 10.);

    
    // gradients
    vec3 ga = hash33(mod(p + vec3(0., 0., 0.), freq));
    vec3 gb = hash33(mod(p + vec3(1., 0., 0.), freq));
    vec3 gc = hash33(mod(p + vec3(0., 1., 0.), freq));
    vec3 gd = hash33(mod(p + vec3(1., 1., 0.), freq));
    vec3 ge = hash33(mod(p + vec3(0., 0., 1.), freq));
    vec3 gf = hash33(mod(p + vec3(1., 0., 1.), freq));
    vec3 gg = hash33(mod(p + vec3(0., 1., 1.), freq));
    vec3 gh = hash33(mod(p + vec3(1., 1., 1.), freq));
    
    // projections
    float va = dot(ga, w - vec3(0., 0., 0.));
    float vb = dot(gb, w - vec3(1., 0., 0.));
    float vc = dot(gc, w - vec3(0., 1., 0.));
    float vd = dot(gd, w - vec3(1., 1., 0.));
    float ve = dot(ge, w - vec3(0., 0., 1.));
    float vf = dot(gf, w - vec3(1., 0., 1.));
    float vg = dot(gg, w - vec3(0., 1., 1.));
    float vh = dot(gh, w - vec3(1., 1., 1.));
	
    // interpolation
    return va + 
           u.x * (vb - va) + 
           u.y * (vc - va) + 
           u.z * (ve - va) + 
           u.x * u.y * (va - vb - vc + vd) + 
           u.y * u.z * (va - vc - ve + vg) + 
           u.z * u.x * (va - vb - ve + vf) + 
           u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);
}

// Function 1106
float noise( in vec3 x ) { // in [0,1]
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.-2.*f);

    float n = p.x + p.y*57. + 113.*p.z;

    float res = mix(mix(mix( hash(n+  0.), hash(n+  1.),f.x),
                        mix( hash(n+ 57.), hash(n+ 58.),f.x),f.y),
                    mix(mix( hash(n+113.), hash(n+114.),f.x),
                        mix( hash(n+170.), hash(n+171.),f.x),f.y),f.z);
    return res;
}

// Function 1107
float Get3DNoise(vec3 u){return texture(iChannel0,u*noiseResInverse).x;}

// Function 1108
float smoothNoise(vec2 p) {
  vec2 inter = smoothstep(0., 1., fract(p));
  float s = mix(noise(sw(p)), noise(se(p)), inter.x);
  float n = mix(noise(nw(p)), noise(ne(p)), inter.x);
  return mix(s, n, inter.y);
  return noise(nw(p));
}

// Function 1109
float ringRayNoise(vec3 ray,vec3 pos,float r,float size,mat3 mr,float anim)
{
  	float b = dot(ray,pos);
    vec3 pr=ray*b-pos;
       
    float c=length(pr);

    pr*=mr;
    
    pr=normalize(pr);
    
    float s=max(0.0,(1.0-size*abs(r-c)));
    
    float nd=noise4q(vec4(pr*1.0,-anim+c))*2.0;
    nd=pow(nd,2.0);
    float n=0.4;
    float ns=1.0;
    if (c>r) {
        n=noise4q(vec4(pr*10.0,-anim+c));
        ns=noise4q(vec4(pr*50.0,-anim*2.5+c*2.0))*2.0;
    }
    n=n*n*nd*ns;
    
    return pow(s,4.0)+s*s*n;
}

// Function 1110
float smoothNoise2(vec2 p)
{
    vec2 p0 = floor(p + vec2(0.0, 0.0));
    vec2 p1 = floor(p + vec2(1.0, 0.0));
    vec2 p2 = floor(p + vec2(0.0, 1.0));
    vec2 p3 = floor(p + vec2(1.0, 1.0));
    vec2 pf = fract(p);
    return mix( mix(noise(p0), noise(p1), pf.x), 
              	mix(noise(p2), noise(p3), pf.x), pf.y);
}

// Function 1111
vec3 noise3(float x)
{
    return vec3(noise(x), fract(noise(x * 5.0) * 11.0), fract(noise(x * 3.0) * 7.0));
}

// Function 1112
float noise3D(vec3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
	const vec3 s = vec3(7, 157, 113);
	
	vec3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride vector for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    
	p -= ip; // Cell's fractional component.
	
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Smoother version of the above. Weirdly, the extra calculations can sometimes
    // create a surface that's easier to hone in on, and can actually speed things up.
    // Having said that, I'm sticking with the simpler version above.
	//p = p*p*p*(p*(p * 6. - 15.) + 10.);
    
    // Even smoother, but this would have to be slower, surely?
	//vec3 p3 = p*p*p; p = ( 7. + ( p3 - 7. ) * p ) * p3;	
	
    // Cosinusoidal smoothing. OK, but I prefer other methods.
    //p = .5 - .5*cos(p*3.14159);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: fract(sin(x)*largeNumber).
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
	
    // Interpolating along Y.
    h.xy = mix(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return mix(h.x, h.y, p.z); // Range: [0, 1].
	
}

// Function 1113
vec4 Denoise(vec2 lUV, vec2 aUV, sampler2D light, sampler2D attr,
             float radius, float CD, vec3 CN) {
    vec4 L0,L1,L2,L3,L4,L5,L6,L7,L8;
    //Light fetching
    L0=texture(light,lUV*IRES); L1=texture(light,(lUV+vec2(radius,0.))*IRES);
    L2=texture(light,(lUV+vec2(-radius,0.))*IRES); L3=texture(light,(lUV+vec2(0.,radius))*IRES);
    L4=texture(light,(lUV+vec2(0.,-radius))*IRES);
    L5=texture(light,(lUV+vec2(radius))*IRES); L6=texture(light,(lUV+vec2(-radius,radius))*IRES);
    L7=texture(light,(lUV+vec2(radius,-radius))*IRES); L8=texture(light,(lUV+vec2(-radius))*IRES);
    //Variance
    vec2 Moments=(Read(L0.w).yz*0.25
        		+(Read(L1.w).yz+Read(L2.w).yz+Read(L3.w).yz+Read(L4.w).yz)*0.125
        		+(Read(L5.w).yz+Read(L6.w).yz+Read(L7.w).yz+Read(L8.w).yz)*I16)*16.;
    float Variance=abs(Moments.y-Moments.x*Moments.x);
    Moments=clamp(Moments,vec2(0.),vec2(0.99));
    //SVGF filter
    	float Lc=1./(pow(0.5,radius-1.)*Coeff_L*sqrt(Variance)+0.0001);
    vec4 Accum=vec4(L0.xyz*0.25,0.25);
    Accum+=(vec4(L1.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(radius,0.))*IRES),CD,CN,L1.xyz-L0.xyz,Lc)+
            vec4(L2.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(-radius,0.))*IRES),CD,CN,L2.xyz-L0.xyz,Lc)+
            vec4(L3.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(0.,radius))*IRES),CD,CN,L3.xyz-L0.xyz,Lc)+
            vec4(L4.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(0.,-radius))*IRES),CD,CN,L4.xyz-L0.xyz,Lc)
            )*0.125;
    Accum+=(vec4(L5.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(radius))*IRES),CD,CN,L5.xyz-L0.xyz,Lc)+
            vec4(L6.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(-radius,radius))*IRES),CD,CN,L6.xyz-L0.xyz,Lc)+
            vec4(L7.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(radius,-radius))*IRES),CD,CN,L7.xyz-L0.xyz,Lc)+
            vec4(L8.xyz,1.)*DenoiseCoeff(texture(attr,(aUV+vec2(-radius))*IRES),CD,CN,L8.xyz-L0.xyz,Lc)
            )*I16;
    //Output
    return vec4(Accum.xyz/Accum.w,Write(vec4(Read(L0.w).x,Moments,0.)));
}

// Function 1114
float noise_sum_abs3(vec3 p) {
    float f = 0.;
    p = p * 3.;
    f += 1.0000 * abs(perlin_noise3(p)); p = 2. * p;
    f += 0.5000 * abs(perlin_noise3(p)); p = 3. * p;
	f += 0.2500 * abs(perlin_noise3(p)); p = 4. * p;
	f += 0.1250 * abs(perlin_noise3(p)); p = 5. * p;
	f += 0.0625 * abs(perlin_noise3(p)); p = 6. * p;
    
    return f;
}

// Function 1115
float gradientNoise(vec2 v)
{
    return fract(52.9829189 * fract(dot(v, vec2(0.06711056, 0.00583715))));
}

// Function 1116
float Noise(in vec3 p, in float numCells )
{
	vec3 f, i;
	
	p *= numCells;

	
	f = fract(p);		// Separate integer from fractional
    i = floor(p);
	
    vec3 u = f*f*(3.0-2.0*f); // Cosine interpolation approximation

    return mix( mix( mix( dot( Hash( i + vec3(0.0,0.0,0.0), numCells ), f - vec3(0.0,0.0,0.0) ), 
                          dot( Hash( i + vec3(1.0,0.0,0.0), numCells ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( Hash( i + vec3(0.0,1.0,0.0), numCells ), f - vec3(0.0,1.0,0.0) ), 
                          dot( Hash( i + vec3(1.0,1.0,0.0), numCells ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( Hash( i + vec3(0.0,0.0,1.0), numCells ), f - vec3(0.0,0.0,1.0) ), 
                          dot( Hash( i + vec3(1.0,0.0,1.0), numCells ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( Hash( i + vec3(0.0,1.0,1.0), numCells ), f - vec3(0.0,1.0,1.0) ), 
                          dot( Hash( i + vec3(1.0,1.0,1.0), numCells ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// Function 1117
float fractalNoise(vec2 vl, out vec4 der) {
    float persistance = 2.;
    float amplitude = 1.2;
    float rez = 0.0;
    vec2 p = vl;
    vec4 temp;
    float norm = 0.;
    der = vec4(0.);
    for (int i = 0; i < OCTAVES + 2; i++) {
        norm += amplitude;
        rez += amplitude * valueNoiseSimple(p, temp);
        // to use as normals, we need to take into account whole length,
        // we can either normalize vector here or don't apply the amplitude
        der += temp;
        amplitude /= persistance;
        p *= persistance;
    }
    return rez / norm;
}

// Function 1118
float Noise3D_Dust(in vec3 mPos)
{
    // Just some random figures, analogous to stride. You can change this, if you want.
	const vec3 s = vec3(113, 157, 1);
	
	vec3 ip = floor(mPos); // Unique unit cell ID.
    
    // Setting up the stride vector for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    vec4 h = vec4(0.0, s.yz, s.y + s.z) + dot(ip, s);
    
	mPos -= ip; // Cell's fractional component.
	
    // A bit of cubic smoothing, to give the noise that rounded look.
    mPos = mPos * mPos * (3.0 - 2.0 * mPos);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: fract(sin(x)*largeNumber).
    h = mix(fract(sin(h) * 43758.5453), fract(sin(h + s.x) * 43758.5453), mPos.x);
	
    // Interpolating along Y.
    h.xy = mix(h.xz, h.yw, mPos.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return mix(h.x, h.y, mPos.z); // Range: [0, 1].
}

// Function 1119
float Noisefv3a (vec3 p)
{
  vec4 t1, t2;
  vec3 ip, fp;
  ip = floor (p);
  fp = fract (p);
  fp = fp * fp * (3. - 2. * fp);
  t1 = Hashv4v3 (ip);
  t2 = Hashv4v3 (ip + vec3 (0., 0., 1.));
  return mix (mix (mix (t1.x, t1.y, fp.x), mix (t1.z, t1.w, fp.x), fp.y),
              mix (mix (t2.x, t2.y, fp.x), mix (t2.z, t2.w, fp.x), fp.y), fp.z);
}

// Function 1120
float noise(vec2 p, sampler2D s, vec2 r, float sc, float bl)
{
    float data = 0.;
    for(int i = 0; i < 8; i++)
    {
        data+=texture(s, p*sc+SAMPLE[i]*(bl/r.x)).x/8.;
    }
    return data;
}

// Function 1121
float NOISE(float x) { float v = 0.0; float a = 0.5; float shift = float(100); for (int i = 0; i < NUM_OCTAVES; ++i) { v += a * noise(x); x = x * 2.0 + shift; a *= 0.5; } return v; }

// Function 1122
vec3 voronoi( in vec2 x, out vec2 cpId )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash( n + g );
		#ifdef ANIMATE
        o = animbias + animscale*sin( iTime*0.5 + 6.2831*o );
        #endif	
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash( n + g );
		#ifdef ANIMATE
        o = animbias + animscale*sin( iTime*0.5 + 6.2831*o );
        #endif	
        vec2 r = g + o - f;

		
        if( dot(mr-r,mr-r)>0.000001 )
		{
        // distance to line		
        float d = dot( 0.5*(mr+r), normalize(r-mr) );

        md = min( md, d );
		}
    }
	
	cpId = n+mg;

    return vec3( md, mr );
}

// Function 1123
float bilerpNoise(vec2 uv) {
    vec2 uvFract = fract(uv);
    float ll = random1(floor(uv));
    float lr = random1(floor(uv) + vec2(1,0));
    float ul = random1(floor(uv) + vec2(0,1));
    float ur = random1(floor(uv) + vec2(1,1));

    float lerpXL = mySmootherStep(ll, lr, uvFract.x);
    float lerpXU = mySmootherStep(ul, ur, uvFract.x);

    return mySmootherStep(lerpXL, lerpXU, uvFract.y);
}

// Function 1124
float TriangularNoise(vec2 n,float time){
    float t = fract(time);
	float nrnd0 = nrand( n + 0.07*t );
	float nrnd1 = nrand( n + 0.11*t );
	return (nrnd0+nrnd1) / 2.0;
}

// Function 1125
float oldnoise( in vec3 p ){
    vec3 i = floor( p );
    vec3 f = fract( p );
	vec3 u = f*f*(3.0-2.0*f);
    return mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// Function 1126
float noise3D(vec3 p){
	return fract(sin(dot(p ,vec3(12.9898,78.233,128.852))) * 43758.5453)*2.0-1.0;
}

// Function 1127
float octaveNoise(vec2 p){
    float f;
    p *= 1.0;
    //mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
    mat2 m = mat2( 2.8,  2.6, -2.6,  2.8 );
    f  = 0.3333*noise( p ); p = m*p;
    f += 0.1111*noise( p ); p = m*p;
    f += 0.0370*noise( p ); p = m*p;
    f += 0.0123*noise( p ); p = m*p;
    return f;
}

// Function 1128
float gnoise(float p) {
    float i = floor(p);
	float f = fract(p);
    
    float a = hash11(i) * f;
    float b = hash11(i + 1.0) * (f - 1.0);
    
    float u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u);
}

// Function 1129
float noise2D(vec2 uv)
{
    uv = fract(uv)*1e3;
    vec2 f = fract(uv);
    uv = floor(uv);
    float v = uv.x+uv.y*1e3;
    vec4 r = vec4(v, v+1., v+1e3, v+1e3+1.);
    r = fract(1e5*sin(r*1e-2));
    f = f*f*(3.0-2.0*f);
    return (mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y));    
}

// Function 1130
vec4 noised(vec3 x){
    vec3 p = floor(x);
    vec3 w = fract(x);
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
    
    vec3 ga = hash( p+vec3(0.0,0.0,0.0) );
    vec3 gb = hash( p+vec3(1.0,0.0,0.0) );
    vec3 gc = hash( p+vec3(0.0,1.0,0.0) );
    vec3 gd = hash( p+vec3(1.0,1.0,0.0) );
    vec3 ge = hash( p+vec3(0.0,0.0,1.0) );
	vec3 gf = hash( p+vec3(1.0,0.0,1.0) );
    vec3 gg = hash( p+vec3(0.0,1.0,1.0) );
    vec3 gh = hash( p+vec3(1.0,1.0,1.0) );
    
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
	
    return vec4( va + u.x*(vb-va) + u.y*(vc-va) + u.z*(ve-va) + u.x*u.y*(va-vb-vc+vd) + u.y*u.z*(va-vc-ve+vg) + u.z*u.x*(va-vb-ve+vf) + (-va+vb+vc-vd+ve-vf-vg+vh)*u.x*u.y*u.z,    // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.z*(ge-ga) + u.x*u.y*(ga-gb-gc+gd) + u.y*u.z*(ga-gc-ge+gg) + u.z*u.x*(ga-gb-ge+gf) + (-ga+gb+gc-gd+ge-gf-gg+gh)*u.x*u.y*u.z +   // derivatives
                 du * (vec3(vb,vc,ve) - va + u.yzx*vec3(va-vb-vc+vd,va-vc-ve+vg,va-vb-ve+vf) + u.zxy*vec3(va-vb-ve+vf,va-vb-vc+vd,va-vc-ve+vg) + u.yzx*u.zxy*(-va+vb+vc-vd+ve-vf-vg+vh) ));
}

// Function 1131
float perlin(float t, float seed)
{
    float r = 0.f;
    float f = 1.f;
    float a = 1.f;
    for(float i = 0.; i < 5.; i++)
    {
        r += noise(f*t, f+seed)*a;
        f *= 1.4;
        a *= 0.6;
    }
    return r;
}

// Function 1132
v0 brushNoise(v1 v,v2 r//https://www.shadertoy.com/view/ltj3Wc
){v+=(noise01(v)-.5)*.02
 ;v+=cos(v.y*3.)*.009
 ;v+=(noise01(v*5.)-.5)*.005
 ;v+=(noise01(v*min(r.y,r.x)*.18)-.5)*.0035
 ;return v.x;}

// Function 1133
float bnoise(in vec3 p)
{
    float n = ssin(fbm(p*21.)*40.)*0.003;
    return n;
}

// Function 1134
float gradientNoise(in vec2 uv)
{
    const vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(uv, magic.xy)));
}

// Function 1135
float mnoiser(vec2 p)
{
    return mnoise(p) - mnoise(p + vec2(0.1, -0.1));
}

// Function 1136
float Perlin(vec2 uv, vec2 params)
{
    float p = 0.;
    float t = params.x;
    for (float i = 0.; i < params.y; i++)
    {
        float a = FBMRandom(vec2(floor(t * uv.x) / t, floor(t * uv.y) / t));	   
        float b = FBMRandom(vec2(ceil (t * uv.x) / t, floor(t * uv.y) / t));		
        float c = FBMRandom(vec2(floor(t * uv.x) / t, ceil (t * uv.y) / t));		
        float d = FBMRandom(vec2(ceil (t * uv.x) / t, ceil (t * uv.y) / t));
        if ((ceil(t * uv.x) / t) == 1.)
        {
            b = FBMRandom(vec2(0., floor(t * uv.y) / t));
            d = FBMRandom(vec2(0., ceil(t * uv.y) / t));
        }
        float coef1 = fract(t * uv.x);
        float coef2 = fract(t * uv.y);
        p += SmoothCos(
                SmoothCos(a, b, coef1),
                SmoothCos(c, d, coef1),
                coef2
                ) * (1. / pow(2., (i + 0.6)));
        t *= 2.;
    }
    return p;
}

// Function 1137
float supernoise3dX(vec3 p){

	float a =  noise3d(p);
	float b =  noise3d(p + 10.5);
	return (a * b);
}

// Function 1138
float HilbertNoise(uvec2 uv)
{
    // Hilbert curve:
    uint C = 0xB4361E9Cu;// cost lookup
    uint P = 0xEC7A9107u;// pattern lookup

    uint c = 0u;// accumulated cost
    uint p = 0u;// current pattern

    const uint N = 10u;// tile size = 2^N
    for(uint i = N; --i < N;)
    {
        uvec2 m = (uv >> i) & 1u;// local uv

        uint n = m.x ^ (m.y << 1u);// linearized local uv

        uint o = (p << 3u) ^ (n << 1u);// offset into lookup tables

        c += ((C >> o) & 3u) << (i << 1u);// accu cost (scaled by layer)

        p = (P >> o) & 3u;// update pattern
    }

    // Noise (via fibonacci hashing):
    #ifdef ANIMATE_NOISE
    c += uint(iFrame);
    #endif
    
    //  2^32-1        = 4294967295
    // (2^32-1) / phi = 2654435769
    return float(c * 2654435769u) * (1.0 / 4294967295.0);
}

// Function 1139
float noise(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);

	vec3 u = f * f * (3.0 - 2.0 * f);

	float n0 = dot(hash3(i + vec3(0.0, 0.0, 0.0)), f - vec3(0.0, 0.0, 0.0));
	float n1 = dot(hash3(i + vec3(1.0, 0.0, 0.0)), f - vec3(1.0, 0.0, 0.0));
	float n2 = dot(hash3(i + vec3(0.0, 1.0, 0.0)), f - vec3(0.0, 1.0, 0.0));
	float n3 = dot(hash3(i + vec3(1.0, 1.0, 0.0)), f - vec3(1.0, 1.0, 0.0));
	float n4 = dot(hash3(i + vec3(0.0, 0.0, 1.0)), f - vec3(0.0, 0.0, 1.0));
	float n5 = dot(hash3(i + vec3(1.0, 0.0, 1.0)), f - vec3(1.0, 0.0, 1.0));
	float n6 = dot(hash3(i + vec3(0.0, 1.0, 1.0)), f - vec3(0.0, 1.0, 1.0));
	float n7 = dot(hash3(i + vec3(1.0, 1.0, 1.0)), f - vec3(1.0, 1.0, 1.0));

	float ix0 = mix(n0, n1, u.x);
	float ix1 = mix(n2, n3, u.x);
	float ix2 = mix(n4, n5, u.x);
	float ix3 = mix(n6, n7, u.x);

	float ret = mix(mix(ix0, ix1, u.y), mix(ix2, ix3, u.y), u.z) * 0.5 + 0.5;
	return ret * 2.0 - 1.0;
}

// Function 1140
float lfnoise(float y)
{
    vec2 t = y*c.xx;
    vec2 i = floor(t);
    t = smoothstep(c.yy, c.xx, fract(t));
    vec2 v1 = vec2(hash12(i), hash12(i+c.xy)),
    v2 = vec2(hash12(i+c.yx), hash12(i+c.xx));
    v1 = c.zz+2.*mix(v1, v2, t.y);
    return mix(v1.x, v1.y, t.x);
}

// Function 1141
float noise( in vec2 p )
{
    // from https://www.shadertoy.com/view/lsf3WH
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1142
float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Function 1143
float noise( vec3 x )
{
	// http://www.iquilezles.org/www/articles/gradientnoise/gradientnoise.htm
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);

    float a = hash( p+vec3(0,0,0) );
    float b = hash( p+vec3(1,0,0) );
    float c = hash( p+vec3(0,1,0) );
    float d = hash( p+vec3(1,1,0) );
    float e = hash( p+vec3(0,0,1) );
    float f = hash( p+vec3(1,0,1) );
    float g = hash( p+vec3(0,1,1) );
    float h = hash( p+vec3(1,1,1) );

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;
    return -1.0+2.0*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z);
}

// Function 1144
float noise(vec2 x){
    vec2 f = fract(x)*fract(x)*(3.0-2.0*fract(x));
	return mix(mix(hash12(floor(x)),
                   hash12(floor(x)+vec2(1,0)),f.x),
               mix(hash12(floor(x)+vec2(0,1)),
                   hash12(floor(x)+vec2(1)),f.x),f.y);
}

// Function 1145
vec2 noise2_2( vec2 p )     
{
	vec3 pos = vec3(p,.5);
	if (ANIM) pos.z += time;
	pos *= m;
    float fx = noise(pos);
    float fy = noise(pos+vec3(1345.67,0,45.67));
    return vec2(fx,fy);
}

// Function 1146
float noise(vec3 x) {
    const vec3 step = vec3(110, 241, 171);

    vec3 i = floor(x);
    vec3 f = fract(x);
 
    // For performance, compute the base input to a 1D hash from the integer part of the argument and the 
    // incremental change to the 1D based on the 3D -> 1D wrapping
    float n = dot(i, step);

    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

// Function 1147
vec3 gradientNoised(vec2 pos, vec2 scale, float rotation, float seed) 
{
    vec2 sinCos = vec2(sin(rotation), cos(rotation));
    return gradientNoised(pos, scale, mat2(sinCos.y, sinCos.x, sinCos.x, sinCos.y), seed);
}

// Function 1148
vec4 _DenoiseREF(vec2 uv, vec2 lUV, vec2 aUV, vec2 UVoff, vec3 CVP,
            vec3 CN, vec3 CVN, sampler2D ta, sampler2D tl, vec2 ires, vec2 hres, vec2 asfov) {
    //Denoiser help function
    vec4 l0=texture(tl,(lUV+UVoff)*ires);
    vec4 a0=texture(ta,(aUV+UVoff)*ires);
    vec4 l1=texture(tl,(lUV-UVoff)*ires);
    vec4 a1=texture(ta,(aUV-UVoff)*ires);
    vec4 L=vec4(0.); vec3 SP;
    if (a0.w<9990. && Box2(uv+UVoff,hres)<-0.5) {
        SP=normalize(vec3(((uv+UVoff)*ires*4.-1.)*asfov,1.))*a0.w;
    	L+=vec4(Read3(l0.x),1.)*Weight(dot(SP-CVP,CVN),dot(CN,Read3(a0.y)*2.-1.));
    }
    if (a1.w<9990. && Box2(uv-UVoff,hres)<-0.5) {
        SP=normalize(vec3(((uv-UVoff)*ires*4.-1.)*asfov,1.))*a1.w;
    	L+=vec4(Read3(l1.x),1.)*Weight(dot(SP-CVP,CVN),dot(CN,Read3(a1.y)*2.-1.));
    }
    return L;
}

// Function 1149
vec3 dnoise3(vec3 p) {  // --- grad(noise)
    vec3 i = floor(p);
    vec3 g, f0 = fract(p), f = f0*f0*(3.-2.*f0), df = 6.*f0*(1.-f0); // smoothstep
#define dmix(a,b,x) x*(b-a)
    g.x = mix(  mix(dmix(hash31(i+vec3(0,0,0)),hash31(i+vec3(1,0,0)),df.x),
                    dmix(hash31(i+vec3(0,1,0)),hash31(i+vec3(1,1,0)),df.x), f.y), 
                mix(dmix(hash31(i+vec3(0,0,1)),hash31(i+vec3(1,0,1)),df.x),
                    dmix(hash31(i+vec3(0,1,1)),hash31(i+vec3(1,1,1)),df.x), f.y), f.z);
    g.y = mix( dmix( mix(hash31(i+vec3(0,0,0)),hash31(i+vec3(1,0,0)),f.x),
                     mix(hash31(i+vec3(0,1,0)),hash31(i+vec3(1,1,0)),f.x), df.y), 
               dmix( mix(hash31(i+vec3(0,0,1)),hash31(i+vec3(1,0,1)),f.x),
                     mix(hash31(i+vec3(0,1,1)),hash31(i+vec3(1,1,1)),f.x), df.y), f.z);
    g.z = dmix( mix( mix(hash31(i+vec3(0,0,0)),hash31(i+vec3(1,0,0)),f.x),
                     mix(hash31(i+vec3(0,1,0)),hash31(i+vec3(1,1,0)),f.x), f.y), 
                mix( mix(hash31(i+vec3(0,0,1)),hash31(i+vec3(1,0,1)),f.x),
                     mix(hash31(i+vec3(0,1,1)),hash31(i+vec3(1,1,1)),f.x), f.y), df.z);
	return 2.*g; // <><><>only MOD=1 <><><>
}

// Function 1150
float Noise11(float x)
{
    float p = floor(x);
    float f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix( hash11(p), hash11(p + 1.0), f)-.5;

}

// Function 1151
float Noise(in vec2 p)
{
	vec2 f = fract(p);
    p = floor(p);
    f = f*f*(3.0-2.0*f);
    float res = mix(mix(Hash(p),
						Hash(p + vec2(1.0, 0.0)), f.x),
					mix(Hash(p + vec2(0.0, 1.0)),
						Hash(p + vec2(1.0, 1.0)), f.x), f.y);
    return res;
}

// Function 1152
float perlin2d(vec2 co) {
    float i0 = floor(co.x);
    float j0 = floor(co.y);
    float i1 = ceil(co.x);
    float j1 = ceil(co.y);

    vec2 g00 = normalize(hash22(vec2(i0, j0)) * 2.0 - 1.0);
    vec2 g10 = normalize(hash22(vec2(i1, j0)) * 2.0 - 1.0);
    vec2 g01 = normalize(hash22(vec2(i0, j1)) * 2.0 - 1.0);
    vec2 g11 = normalize(hash22(vec2(i1, j1)) * 2.0 - 1.0);
    
    vec2 uv = fract(co);
    
    float n00 = dot(g00, vec2(uv.x, uv.y));
    float n10 = dot(g10, vec2(uv.x - 1.0, uv.y));
    float n01 = dot(g01, vec2(uv.x, uv.y - 1.0));
    float n11 = dot(g11, vec2(uv.x - 1.0, uv.y - 1.0));
    
    float su = smoother(uv.x);
    float sv = smoother(uv.y);

    float nx0 = (1.0 - su) * n00 + n10 * su;
    float nx1 = (1.0 - su) * n01 + n11 * su;
    
    float n = (1.0 - sv) * nx0 + nx1 * sv;

    return n;
}

// Function 1153
float star_noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0;

    float res = mix(mix( star_hash(n+  0.0), star_hash(n+  1.0),f.x),
                    mix( star_hash(n+ 57.0), star_hash(n+ 58.0),f.x),f.y);

    return res;
}

// Function 1154
float noise(vec3 x){
    vec3 p  = floor(x);
    vec3 f  = fract(x);
    f       = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    return mix(mix(mix( h(n+0.0), h(n+1.0),f.x),
                   mix( h(n+57.0), h(n+58.0),f.x),f.y),
               mix(mix( h(n+113.0), h(n+114.0),f.x),
                   mix( h(n+170.0), h(n+171.0),f.x),f.y),f.z);
}

// Function 1155
float perlinNoise(vec2 coord, int seed)
{
    ivec2 cell = ivec2(coord);
    if (coord.x < 0.0) cell.x--;
    if (coord.y < 0.0) cell.y--;
    vec2 posInCell = coord-vec2(cell);
    float val00 = dot(circle(ivecnoise(cell+ivec2(0, 0), seed)), posInCell-vec2(0.0, 0.0));
    float val01 = dot(circle(ivecnoise(cell+ivec2(0, 1), seed)), posInCell-vec2(0.0, 1.0));
    float val10 = dot(circle(ivecnoise(cell+ivec2(1, 0), seed)), posInCell-vec2(1.0, 0.0));
    float val11 = dot(circle(ivecnoise(cell+ivec2(1, 1), seed)), posInCell-vec2(1.0, 1.0));
    vec2 mixAmount = posInCell*posInCell*posInCell*(posInCell*(posInCell*6.0-15.0)+10.0);
    return mix(mix(val00, val10, mixAmount.x), mix(val01, val11, mixAmount.x), mixAmount.y);
}

// Function 1156
float simplexRot(vec2 u, vec2 p, float rot
){vec3 x,y;vec2 d0,d1,d2,p0,p1,p2;NoiseHead(u,x,y,d0,d1,d2,p0,p1,p2)
 ;x=mod(vec3(p0.x,p1.x,p2.x),p.x)
 ;y=mod(vec3(p0.y,p1.y,p2.y),p.y);x=x+.5*y//with    TilingPeriod (p)
 ;return NoiseNoDer(x,y,rot,d0,d1,d2);}

// Function 1157
float perlin(vec3 p) {
    vec3 i = floor(p);
    vec4 a = dot(i, vec3(1., 57., 21.)) + vec4(0., 57., 21., 78.);
    vec3 f = cos((p-i)*PI)*(-.5)+.5;
    a = mix(sin(cos(a)*a),sin(cos(1.+a)*(1.+a)), f.x);
    a.xy = mix(a.xz, a.yw, f.y);
    return mix(a.x, a.y, f.z);
}

// Function 1158
float metaNoiseRaw(vec2 uv, float density)
{
    float v = 0.99;
    float r0 = hash(2015.3548);
    float s0 = iTime*(r0-0.5)*rotationSpeed;
    vec2 f0 = iTime*moveSpeed*r0;
    vec2 c0 = vec2(hash(31.2), hash(90.2)) + s0;   
    vec2 uv0 = rotuv(uv*(1.0+r0*v), r0*360.0 + s0, c0) + f0;    
    float metaball0 = saturate1(metaBall(uv0)*density);
    
    for(int i = 0; i < 25; i++)
    {
        float inc = float(i) + 1.0;
    	float r1 = hash(2015.3548*inc);
        float s1 = iTime*(r1-0.5)*rotationSpeed;
        vec2 f1 = iTime*moveSpeed*r1;
    	vec2 c1 = vec2(hash(31.2*inc), hash(90.2*inc))*100.0 + s1;   
    	vec2 uv1 = rotuv(uv*(1.0+r1*v), r1*360.0 + s1, c1) + f1 - metaball0*distortion;    
    	float metaball1 = saturate1(metaBall(uv1)*density);
        
        metaball0 *= metaball1;
    }
    
    return pow(metaball0, metaPow);
}

// Function 1159
vec3 divfreenoise( vec3 q ) { // fluid-like noise = div-free -> curl
    vec2 e = vec2(1./16.,0);
    q += .1*iTime;            // animated flow
    vec3 v = noise(q); 
 // return 2.*(v -.5);             // regular
    return vec3( noise(q+e.yxy).z-v.z - v.y+noise(q+e.yyx).y, // curl
                 noise(q+e.yyx).x-v.x - v.z+noise(q+e.xyy).z,
                 noise(q+e.xyy).y-v.y - v.x+noise(q+e.yxy).x
                ) *1.;
}

// Function 1160
float noise(vec2 x)
{
    vec2 p = floor(x), f = fract(x);
    float va = 0., wt = 0.;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ ) {
        vec2 g = vec2(i,j);
		vec3 o = hash3( p + g )*vec3(0,0,1);
		vec2 r = g - f + o.xy;
		float d = dot(r,r),
		ww = 1.-smoothstep(0.,1.414,sqrt(d));
		va += o.z*ww;
		wt += ww;
    }
	
    return va/wt;
}

// Function 1161
vec2 rectvoronoi(in vec3 x, RSet3 r1, RSet3 r2, RSet3 r3) {
    ivec3 p = ivec3(floor( x ));
    vec3 f = fract( x );

    ivec3 mb;
    vec3 mr;
    float id = 1.0e20;
    const int range = 3;
    for( int k=-VORORANGE; k<=VORORANGE; k++ )
    for( int j=-VORORANGE; j<=VORORANGE; j++ )
    for( int i=-VORORANGE; i<=VORORANGE; i++ )
    {
        ivec3 b = ivec3( i, j, k );
        vec3 B = vec3(p + b);
        //vec3 rv = rand3(B, d, s );
        vec3 rv = rvec3(B, r1, r2, r3);
        //vec3 rv = RAND(B, 0.0, 1.0, s);
        vec3 r = vec3(b) - f + rv;
        float dis = length( r );

        if(dis < id) {
            mb = b;
            mr = r;
            id = dis;
        }
    }
    float bd = 1.0e20;
    for( int k=-VORORANGE; k<=VORORANGE; k++ )
    for( int j=-VORORANGE; j<=VORORANGE; j++ )
    for( int i=-VORORANGE; i<=VORORANGE; i++ )
    {
        ivec3 b = mb + ivec3( i, j, k );
        vec3 B = vec3(p + b);
        //vec3 rv = rand3(B, d, s );
        vec3 rv = rvec3(B, r1, r2, r3);
        //vec3 rv = RAND(B, 0.0, 1.0, s);
        vec3 r = vec3(b) - f + rv;
        float dis = dot( 0.5*(mr+r), normalize(r-mr) );

        bd = min(bd, dis);
    }
    return vec2(id, bd);
}

// Function 1162
float noise21(vec2 p, float pro, float st) {
    float v=0.0;
    float s=0.5;
    for(float i=0.; i<st; ++i) {
        v+=srnd21(p+i*72.3)*s;
        p*=pro;
        s*=0.5;
    }
    return v;
}

// Function 1163
float noise( in vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

// Function 1164
vec2 noise(vec3 p) {
    uvec3 i = uvec3(ivec3(floor(p)));
     vec3 f =             fract(p)  ,
          u = f*f*f*(f*(f*6.-15.)+10.),
         du = 30.*f*f*(f*(f-2.)+1.);

#define g(x,y,z) sin( tau* ( hash3i1f(i+uvec3(x,y,z)) + vec2(0,.25) ) ) // SC(rand angle)
    vec2 ga = g(0,0,0),
         gb = g(1,0,0),
         gc = g(0,1,0),
         gd = g(1,1,0),
         ge = g(0,0,1),
         gf = g(1,0,1),
         gg = g(0,1,1),
         gh = g(1,1,1);
 
#define v(g,i,j)  dot(g, f.xy - vec2(i,j))
    float va = v(ga,0,0),
          vb = v(gb,1,0),
          vc = v(gc,0,1),
          vd = v(gd,1,1),
          ve = v(ge,0,0),
          vf = v(gf,1,0),
          vg = v(gg,0,1),
          vh = v(gh,1,1);
    
    return mix(mix(mix(ga, gb, u.x), mix(gc, gd, u.x), u.y),
               mix(mix(ge, gf, u.x), mix(gg, gh, u.x), u.y), u.z)
         + du.xy * mix(u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va,
                       u.yx*(ve-vf-vg+vh) + vec2(vf,vg) - ve, u.z);
}

// Function 1165
float noise2(vec3 pos)
{
    vec3 q = 8.0*pos;
    float f  = 0.5000*noise(q) ; q = m*q*2.01;
    f+= 0.2500*noise(q); q = m*q*2.02;
    f+= 0.1250*noise(q); q = m*q*2.03;
    f+= 0.0625*noise(q); q = m*q*2.01;
    return f;
}

// Function 1166
float Voronoi(vec3 p)
{
	vec3 n = floor(p);
	vec3 f = fract(p);

	float shortestDistance = 1.0;
	for (int x = -1; x < 1; x++) {
		for (int y = -1; y < 1; y++) {
			for (int z = -1; z < 1; z++) {
				vec3 o = vec3(x,y,z);
				vec3 r = (o - f) + 1.0 + sin(Hash3(n + o)*50.0)*0.2;
				float d = dot(r,r);
				if (d < shortestDistance) {
					shortestDistance = d;
				}
			}
		}
	}
	return shortestDistance;
}

// Function 1167
float noise( in vec3 x ) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 uv = (i.xy+vec2(37.0,17.0)*i.z) + f.xy;
    vec2 rg = textureLod( iChannel1, (uv+0.5)/256.0, 0.0).yx;
    return mix( rg.x, rg.y, f.z );
}

// Function 1168
float triNoise2d(in vec2 p, float spd)
{
    float z=1.8;
    float z2=2.5;
	float rz = 0.;
    p *= mm2(p.x*0.06);
    vec2 bp = p;
	for (float i=0.; i<5.; i++ )
	{
        vec2 dg = tri2(bp*1.85)*.75;
        dg *= mm2(iTime*spd);
        p -= dg/z2;

        bp *= 1.3;
        z2 *= .45;
        z *= .42;
		p *= 1.21 + (rz-1.0)*.02;
        
        rz += tri(p.x+tri(p.y))*z;
        p*= -m2;
	}
    return clamp(1./pow(rz*29., 1.0),0.,.55);
}

// Function 1169
float noise(vec2 x){
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float res = mix(mix(hash(p),          hash(p + add.xy),f.x),
                    mix(hash(p + add.yx), hash(p + add.xx),f.x),f.y);
    return res;
}

// Function 1170
vec2 stepnoise(vec2 p, float size) { 
    p = floor((p+10.)/size)*size;          // is p+10. useful ?   
    p = fract(p*.1) + 1. + p*vec2(2,3)/1e4;    
    p = fract( 1e5 / (.1*p.x*(p.y+vec2(0,1)) + 1.) );
    p = fract( 1e5 / (p*vec2(.1234,2.35) + 1.) );      
    return p;    
}

// Function 1171
float noise_term(in vec3 x, in float scale_val) {
    vec3 s = vec3(scale_val);
    vec3 x000 = x - mod(x, s);
    vec3 x001 = x + vec3(0.0, 0.0, scale_val);
    x001 = x001 - mod(x001, s);
    vec3 x010 = x + vec3(0.0, scale_val, 0.0);
    x010 = x010 - mod(x010, s);
    vec3 x011 = x + vec3(0.0, s.xy);
    x011 = x011 - mod(x011, s);
    vec3 x100 = x + vec3(scale_val, 0.0, 0.0);
    x100 = x100 - mod(x100, s);
    vec3 x101 = x + vec3(scale_val, 0.0, scale_val);
    x101 = x101 - mod(x101, s);
    vec3 x110 = x + vec3(s.xy, 0.0);
    x110 = x110 - mod(x110, s);
    vec3 x111 = x + s;
    x111 = x111 - mod(x111, s);
    
    float v000 = hash13(x000);
    float v001 = hash13(x001);
    float v010 = hash13(x010);
    float v011 = hash13(x011);
    float v100 = hash13(x100);
    float v101 = hash13(x101);
    float v110 = hash13(x110);
    float v111 = hash13(x111);
    
    vec3 uvw = mod(x, s) / s;
    
    float zweight = smoothstep(0.0, 1.0, uvw.z);
    float v00 = mix(v000, v001, zweight);
    float v01 = mix(v010, v011, zweight);
    float v10 = mix(v100, v101, zweight);
    float v11 = mix(v110, v111, zweight);
    
    float yweight = smoothstep(0.0, 1.0, uvw.y);
    float v1 = mix(v10, v11, yweight);
    float v0 = mix(v00, v01, yweight);
    
    float xweight = smoothstep(0.0, 1.0, uvw.x);
    
    return mix(v0, v1, xweight);
}

// Function 1172
vec3 TerrainNoiseMQ(vec2 p) { return TerrainNoise(p, 9); }

// Function 1173
float Simplex12(vec2 p, float r, float z)
{vec2 q=skew(p),a=floor(q),i=q-a
;vec2 c=a+mix(vec2(0,1),vec2(1,0),step(i.y,i.x))
;z=z+10.*floor(iTime/40.)
;return h12Skew(a   ,p,r,z)
       +h12Skew(a+1.,p,r,z)
       +h12Skew(c   ,p,r,z);}

// Function 1174
float noise3D(in vec3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
	const vec3 s = vec3(113, 157, 1);
	
	vec3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride vector for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    
	p -= ip; // Cell's fractional component.
	
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: fract(sin(x)*largeNumber).
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
	
    // Interpolating along Y.
    h.xy = mix(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return mix(h.x, h.y, p.z); // Range: [0, 1].
	
}

// Function 1175
vec4 fractalNoise(vec2 coord) {
    vec4 value = vec4(0.0);
    float scale = 0.5;
    for (int i = 0; i < 5; i += 1) {
     	value += texture(iChannel0, coord) * scale;
        coord *= 2.0;
        scale *= 0.6;
    }
    return value;
}

// Function 1176
float add_noise(vec3 x) {
    float n = noise(x)/2.;  x *= 2.1; // return n*2.;
         n += noise(x)/4.;  x *= 1.9;
         n += noise(x)/8.;  x *= 2.3;
         n += noise(x)/16.; x *= 1.9;
      //   n += noise(x)/32.;
    return n; 
}

// Function 1177
float noise(in vec2 p) {
    vec2 ip = floor(p);
    vec2 fp = fract(p);
	vec2 u = fp*fp*(3.0-2.0*fp);
    return -1.0+2.0*mix( mix( nmzHash( ip + vec2(0.0,0.0) ), nmzHash( ip + vec2(1.0,0.0) ), u.x),
                mix(nmzHash( ip + vec2(0.0,1.0) ), nmzHash( ip + vec2(1.0,1.0)), u.x), u.y);
}

// Function 1178
float osc_noise(float p){p *= 20000.;float F = floor(p), f = fract(p);
 return mix(hash(F), hash(F+1.), f);}

// Function 1179
float noise_sum_abs_sin(vec2 p)
{
    float f = noise_sum_abs(p);
    f = sin(f * 1.5 + p.x * 7.0);
    
    return f * f;
}

// Function 1180
vec2 Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);

	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec4 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0 );
	return mix( rg.yw, rg.xz, f.z );
}

// Function 1181
vec3 noise(vec3 p)
{
    return 1.0 - 2.0 * abs(0.5 - textureLod(iChannel0, p, 0.0).xyz);
}

// Function 1182
float smplxNoise2D(vec2 p, out vec2 deriv, float randKey, float roffset)
{
    //i is a skewed coordinate of a bottom vertex of a simplex where p is in.
    vec2 i0 = floor(p + vec2( (p.x + p.y)*skewF(2.0) ));
    //x0, x1, x2 are unskewed displacement vectors.
    float unskew = unskewG(2.0);
    vec2 x0 = p - (i0 + vec2((i0.x + i0.y)*unskew));

    vec2 ii1 = x0.x > x0.y ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec2 ii2 = vec2(1.0);

//  vec2 i1 = i0 + ii1;
//  vec2 x1 = p - (i1 + vec2((i1.x + i1.y)*unskew));
//          = p - (i0 + ii1 + vec2((i0.x + i0.y + 1.0)*unskew));
//          = p - (i0 + vec2((i0.x + i0.y)*unskew)) - ii1 - vec2(1.0)*unskew;
    vec2 x1 = x0 - ii1 - vec2(unskew);
//  vec2 i2 = i0 + ii2;
//  vec2 x2 = p - (i2 + vec2((i2.x + i2.y)*unskew));
//          = p - (i0 + ii2 + vec2((i0.x + i0.y + 2.0)*unskew));
//          = p - (i0 + vec2((i0.x + i0.y)*unskew)) - ii2 - vec2(2.0)*unskew;
    vec2 x2 = x0 - ii2 - vec2(2.0*unskew);

    vec3 m = max(vec3(0.5) - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0);
    m = m*m;
    m = m*m;

    float r0 = 3.1416*2.0*rand(vec3(mod(i0, 16.0)/16.0, randKey));
    float r1 = 3.1416*2.0*rand(vec3(mod(i0 + ii1, 16.0)/16.0, randKey));
    float r2 = 3.1416*2.0*rand(vec3(mod(i0 + ii2, 16.0)/16.0, randKey));

    float randKey2 = randKey + 0.01;
    float spmin = 0.5;
    float sps = 2.0;
    float sp0 = spmin + sps*rand(vec3(mod(i0, 16.0)/16.0, randKey2));
    float sp1 = spmin + sps*rand(vec3(mod(i0 + ii1, 16.0)/16.0, randKey2));
    float sp2 = spmin + sps*rand(vec3(mod(i0 + ii2, 16.0)/16.0, randKey2));

    r0 += iTime*sp0 + roffset;
    r1 += iTime*sp1 + roffset;
    r2 += iTime*sp2 + roffset;
    //Gradients;
    vec2 g0 = vec2(cos(r0), sin(r0));
    vec2 g1 = vec2(cos(r1), sin(r1));
    vec2 g2 = vec2(cos(r2), sin(r2));

    deriv = smplxNoise2DDeriv(x0, m.x, g0) + smplxNoise2DDeriv(x1, m.y, g1) + smplxNoise2DDeriv(x2, m.z, g2);
    return dot(m*vec3(dot(x0, g0), dot(x1, g1), dot(x2, g2)), vec3(1.0));
//    return dot(m*vec3(length(x0), length(x1), length(x2)), vec3(1.0));
}

// Function 1183
float noise2d(vec2 x){
    vec2 p = floor(x);
    vec2 fr = fract(x);
    vec2 LB = p;
    vec2 LT = p + vec2(0.0, 1.0);
    vec2 RB = p + vec2(1.0, 0.0);
    vec2 RT = p + vec2(1.0, 1.0);

    float LBo = oct(LB);
    float RBo = oct(RB);
    float LTo = oct(LT);
    float RTo = oct(RT);

    float noise1d1 = mix(LBo, RBo, fr.x);
    float noise1d2 = mix(LTo, RTo, fr.x);

    float noise2d = mix(noise1d1, noise1d2, fr.y);

    return noise2d;
}

// Function 1184
float map_noise(vec3 p) {
    p *= 0.5;    
    float ret = noise_3(p);
    ret += noise_3(p * 2.0) * 0.5;
    ret = (ret - 1.0) * 5.0;
    return saturate(ret * 0.5 + 0.5);
}

// Function 1185
float noise3(in vec3 u)
{vec3 p=floor(u);u=fract(u)
;u=herm2(u)
;u.xy+=p.xy+vec2(37,17)*p.z

//;u=mod289(u);
//;u.xy=cellular2x2x2(u);
    //vec2(sin(u.x)+cos(u.x)),sin(u.x)
;u.xy=textureLod(iChannel0,(u.xy+.5)/256.,.0).yx
;return u2(mix(u.x,u.y,u.z));
}

// Function 1186
float densityNoise( vec3 pos )
{
    vec2 bent = getBent();
    
    float noise = 1.0;
    float noiseDetail = textureLod( iChannel1, vec2( 1.0, 1.0 ) * pos.xz / 64.0, 0.0 ).x;        
    pos.x -= pos.y;
    vec2 uv1 = vec2( 0.2, 1.5 ) * pos.xz / 64.0 + iTime * vec2( 0.01, 0.1 );
    float noiseBase = textureLod( iChannel1, uv1, 0.0 ).y;
    noise = step( 0.6, noiseBase );
    noise *= noiseDetail * 0.5 + 0.5;
    noise *= smoothstep( 1.5, 0.0, pos.y ); // height falloff    
    noise *= ( 1.0 - bent.y ); // disable on hills    
	return noise;
}

// Function 1187
bool iSimplex3(vec3 p[4], vec3 ro, vec3 rd, 
               out float near, out float far, 
               out vec4 bnear, out vec4 bfar) {
    // convert ray endpoints to barycentric basis
    // this can be optimized further by caching the determinant
    vec4 r0 = to_bary(p, ro);
    vec4 r1 = to_bary(p, ro + rd);

    // build barycentric ray direction from endpoints
    vec4 brd = r1 - r0;
    // compute ray scalars for each plane
    vec4 t = -r0/brd;
    
    near = -1.0 / 0.0;
    far = 1.0 / 0.0;    
    for (int i = 0; i < 4; ++i) {
        // equivalent to checking dot product of ray dir and plane normal
        if (brd[i] < 0.0) {
            far = min(far, t[i]);
        } else {
            near = max(near, t[i]);
        }
    }
    
    bnear = r0 + brd * near;
    bfar = r0 + brd * far;
    
    return (far > 0.0) && (far > near);
}

// Function 1188
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    float res = mix(mix( hash12(p), hash12(p+ vec2(1.0, 0.0)),f.x),
                    mix( hash12(p+ vec2(.0, 1.0)), hash12(p+ vec2(1.0, 1.0)),f.x),f.y);
    return res;
}

// Function 1189
float noise_itself(vec3 p)
{
    return noise(p * 8.0);
}

// Function 1190
vec2 noise( in float time ) {
	vec2 audio = vec2(0.0);
    for (int t = 0; t < ITERATIONS; t++) {
        float v = float(t)*3.21239;
		audio += hash22(vec2(time + v, time*1.423 + v)) * 2.0 - 1.0;
    }
    audio /= float(ITERATIONS);
    return audio;
}

// Function 1191
vec2 noise(float t){return hash22(vec2(t,t*1.423))*2.-1.;}

// Function 1192
vec3 noised( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );

    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0); 
    
    vec2 ga = hash( i + vec2(0.0,0.0) );
    vec2 gb = hash( i + vec2(1.0,0.0) );
    vec2 gc = hash( i + vec2(0.0,1.0) );
    vec2 gd = hash( i + vec2(1.0,1.0) );
    
    float va = dot( ga, f - vec2(0.0,0.0) );
    float vb = dot( gb, f - vec2(1.0,0.0) );
    float vc = dot( gc, f - vec2(0.0,1.0) );
    float vd = dot( gd, f - vec2(1.0,1.0) );

    return vec3( va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd),   // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +  // derivatives
                 du * (u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va));
}

// Function 1193
float trigNoise3D(in vec3 p){

    p /= 2.;
    float res = 0., sum = 0.;
    
    // IQ's cheap, texture-lookup noise function. Very efficient, but still 
    // a little too processor intensive for multiple layer usage in a largish 
    // "for loop" setup. Therefore, just one layer is being used here.
    float n = pn(p*8. + iTime*2.);


    // Two sinusoidal layers. I'm pretty sure you could get rid of one of 
    // the swizzles (I have a feeling the GPU doesn't like them as much), 
    // which I'll try to do later.
    
    vec3 t = sin(p.yzx*3.14159265 + cos(p.zxy*3.14159265+1.57/2.))*0.5 + 0.5;
    p = p*1.5 + (t - 1.5); //  + iTime*0.1
    res += (dot(t, vec3(0.333)));

    t = sin(p.yzx*3.14159265 + cos(p.zxy*3.14159265+1.57/2.))*0.5 + 0.5;
    res += (dot(t, vec3(0.333)))*0.7071;    
	 
	return ((res/1.7071))*0.85 + n*0.15;
}

// Function 1194
float getNoise(vec2 uv, float t){
    
    //given a uv coord and iTime - return a noise val in range 0 - 1
    //using ashima noise
    
    //octave 1
    float SCALE = 1.5;
    float noise = snoise( vec3(uv.x*SCALE, uv.y*SCALE, t));
    
    //octave 2 - more detail
     SCALE = 3.0;
    noise += snoise( vec3(uv.x*SCALE + t,uv.y*SCALE , 0))* 0.2 ;

    // //octave 3 - more detail
    SCALE = 5.0;
     noise += snoise( vec3(uv.x*SCALE + t,uv.y*SCALE , 0))* 0.2 ;
    
    //move noise into 0 - 1 range    
    noise = (noise/2. + 0.5);
       
    return noise;
    
}

// Function 1195
float noise( in float p )
{
    return noise(vec2(p, 0.0));
}

// Function 1196
float nnoise( in vec2 uv ){return 0.5 + 0.5*snoise(uv);}

// Function 1197
float noise(vec3 x) {
    const vec3 step = vec3(110, 241, 171);

    vec3 i = floor(x);
    vec3 f = fract(x);

   
    float n = dot(i, step);

    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

// Function 1198
float sNoise(vec2 p) {
  	vec2 inter = smoothstep(0., 2., fract(p));
    
  	float s = mix(noise(sw(p)), noise(se(p)), inter.x);
    
  	float n = mix(noise(nw(p)), noise(ne(p)), inter.x);
    
  	return mix(s, n, inter.y);
}

// Function 1199
float add_noise(vec3 x) {
    float n = noise(x)/2.;  x *= 2.1; // return n*2.;
         n += noise(x)/4.;  x *= 1.9;
         n += noise(x)/8.;  x *= 2.3;
         n += noise(x)/16.; x *= 1.9;
         n += noise(x)/32.;
    return n; 
}

// Function 1200
vec4 texNoise(vec2 uv,sampler2D tex ){ float f = 0.; f+=texture(tex, uv*.125).r*.5; f+=texture(tex,uv*.25).r*.25;
                       f+=texture(tex,uv*.5).r*.125; f+=texture(tex,uv*1.).r*.125; f=pow(f,1.2);return vec4(f*.45+.05);}

// Function 1201
float noise(float x)
{
    return fract((sin(x * 102.1) * 1002.1 + sin(x * 53.0)* 3023.7));
}

// Function 1202
float noise2dT(vec2 uv)
{
    vec2 fr = fract(uv);
    vec2 smoothv = fr*fr*(3.0-2.0*fr);
    vec2 fl = floor(uv);
    uv = smoothv + fl;
    return textureLod(iChannel0, (uv + 0.5)/iChannelResolution[0].xy, 0.0).y;	// use constant here instead?
}

// Function 1203
float perlinbetween(float a, float b, float min, float max)
{
	return min + (perlin(a,b))*(max-min);
}

// Function 1204
vec4 denoise2( in sampler2D tex, in vec2 pix )
{
    const float size = 1.0;
    
    vec2 offsets[4];
    offsets[0] = vec2( -1.0, -1.0) * size;
    offsets[1] = vec2( 1.0, -1.0) * size;
    offsets[2] = vec2( -1.0, 1.0) * size;
    offsets[3] = vec2( 1.0, 1.0) * size;
    
    float cur_dev = 9999999999.0;
    vec4 result = vec4(0.5);
    result.a = 1.0;
    
    for( int quad = 0; quad < 4; quad++)
    {
		vec2 sp = pix + offsets[quad];
        vec2 uv = sp / iResolution.xy;
        vec4 smpl = texture( iChannel1, uv );
        float deviation = smpl.a;
        if( deviation < cur_dev )
        {
            cur_dev = deviation;
            result.rgb = smpl.rgb;
        }
    }
    
    
    
    return result;
    
}

// Function 1205
float Voronoi(vec2 p)
{	
    // Partitioning the 2D space into repeat cells.
    vec2 ip = floor(p); // Analogous to the cell's unique ID.
    p -= ip; // Fractional reference point within the cell (fract(p)).

    // Set the minimum distance (squared distance, in this case, because it's 
    // faster) to a maximum of 1. Outliers could reach as high as 2 (sqrt(2)^2)
    // but it's being capped to 1, because it covers a good portion of the range
    // (basically an inscribed unit circle) and dispenses with the need to 
    // normalize the final result.
    //
    // If you're finding that your Voronoi patterns are a little too contrasty,
    // you could raise "d" to something like "1.5." Just remember to divide
    // the final result by the same amount, clamp, or whatever.
    float d = 1.;
    
    // Put a "unique" random point in the cell (using the cell ID above), and it's 8 
    // neighbors (using their cell IDs), then check for the minimum squared distance 
    // between the current fractional cell point and these random points.
    for (int i = -1; i <= 1; i++){
	    for (int j = -1; j <= 1; j++){
	    
     	    vec2 cellRef = vec2(i, j); // Base cell reference point.
            
            vec2 offset = hash22(ip + cellRef); // 2D offset.
            
            // Vector from the point in the cell to the offset point.
            vec2 r = cellRef + offset - p; 
            float d2 = dot(r, r); // Squared length of the vector above.
            
            d = min(d, d2); // If it's less than the previous minimum, store it.
        }
    }
    
    // In this case, the distance is being returned, but the squared distance
    // can be used too, if preferred.
    return sqrt(d); 
}

// Function 1206
float noise(vec3 p){//Noise function stolen from Virgil who stole it from Shane who I assume understands this shit, unlike me who is too busy trying to fit these round pegs in squared holes
  vec3 ip=floor(p),s=vec3(7,157,113);
  p-=ip;
  vec4 h=vec4(0,s.yz,s.y+s.z)+dot(ip,s);
  p=p*p*(3.-2.*p);
  h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
  h.xy=mix(h.xz,h.yw,p.y);
  return mix(h.x,h.y,p.z);//Ah, yes I understand this bit: it draws a shape which, if you have enough imagination, looks like a penis
}

// Function 1207
vec4 Noise4( vec4 x ) { return fract(sin(x)*5346.1764)*2. - 1.; }

// Function 1208
vec4 noiseSpace(vec3 ray,vec3 pos,float r,mat3 mr,float zoom,vec3 subnoise,float anim)
{
  	float b = dot(ray,pos);
  	float c = dot(pos,pos) - b*b;
    
    vec3 r1=vec3(0.0);
    
    float s=0.0;
    float d=0.0625*1.5;
    float d2=zoom/d;

	float rq=r*r;
    float l1=sqrt(abs(rq-c));
    r1= (ray*(b-l1)-pos)*mr;

    r1*=d2;
    s+=abs(noise4q(vec4(r1+subnoise,anim))*d);
    s+=abs(noise4q(vec4(r1*0.5+subnoise,anim))*d*2.0);
    s+=abs(noise4q(vec4(r1*0.25+subnoise,anim))*d*4.0);
    //return s;
    return vec4(s*2.0,abs(noise4q(vec4(r1*0.1+subnoise,anim))),abs(noise4q(vec4(r1*0.1+subnoise*6.0,anim))),abs(noise4q(vec4(r1*0.1+subnoise*13.0,anim))));
}

// Function 1209
float snoise(vec3 v) {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i = floor(v + dot(v, C.yyy) );
  vec3 x0 = v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  // x0 = x0 - 0.0 + 0.0 * C.xxx;
  // x1 = x0 - i1 + 1.0 * C.xxx;
  // x2 = x0 - i2 + 2.0 * C.xxx;
  // x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy; // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z); // mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ ); // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
}

// Function 1210
vec3 DenoiseGI(vec2 uv, vec2 aUV, vec2 lUV, sampler2D attr, sampler2D light, float radie,
                vec3 CVP, vec3 CN, vec3 CVN, vec2 ires, vec2 hres, vec2 asfov) {
    //Diffuse denoiser
    vec4 L0=texture(light,lUV*ires);
    //Variance
    vec2 Moments=Read3(L0.w).xy*16.;
    float Variance=abs(Moments.y-Moments.x*Moments.x);
    float LCoeff=1.; ///(Coeff_L*pow(4.*sqrt(Variance),radius)+0.0001);
    float radius=min(radie,sqrt(Variance/(PI*0.1)));
        radius=min(radie,sqrt((4096./(Read3(L0.w).z*64.))/PI));
    
    //Wavelet
    vec3 LL0=L0.xyz;
    vec4 Accum=vec4(LL0*0.2,0.2);
    Accum+=(_DenoiseGI(uv,aUV,lUV,vec2(radius,0.),CVP,CN,CVN,attr,light,ires,hres,asfov,LL0,LCoeff)+
            _DenoiseGI(uv,aUV,lUV,vec2(0.,radius),CVP,CN,CVN,attr,light,ires,hres,asfov,LL0,LCoeff)+
            _DenoiseGI(uv,aUV,lUV,vec2(radius*0.707),CVP,CN,CVN,attr,light,ires,hres,asfov,LL0,LCoeff)+
            _DenoiseGI(uv,aUV,lUV,vec2(radius,-radius)*0.707,CVP,CN,CVN,attr,light,ires,hres,asfov,LL0,LCoeff)
            )*0.1;
    //Output
    return vec3(Accum.xyz/Accum.w);
}

// Function 1211
vec3 VoronoiPoint(vec2 pos, vec2 delta )
{
	const float randScale = .8; // reduce this to remove axis-aligned hard edged errors
	
	vec2 p = floor(pos)+delta;
	vec2 r = (Rand(p)-.5)*randScale;
	vec2 c = p+.5+r;
	
	// various length calculations for different patterns
	//float l = length(c-pos);
	//float l = length(vec3(c-pos,.1));
	float l = abs(c.x-pos.x)+abs(c.y-pos.y); // more interesting shapes
	
	return vec3(c,l);
}

// Function 1212
vec3 valueNoiseDerivative(vec2 x, sampler2D smp)
{
    vec2 f = fract(x);
    vec2 u = f * f * (3. - 2. * f);

#if 1
    // texel fetch version
    ivec2 p = ivec2(floor(x));
    float a = texelFetch(smp, (p + ivec2(0, 0)) & 255, 0).x;
	float b = texelFetch(smp, (p + ivec2(1, 0)) & 255, 0).x;
	float c = texelFetch(smp, (p + ivec2(0, 1)) & 255, 0).x;
	float d = texelFetch(smp, (p + ivec2(1, 1)) & 255, 0).x;
#else    
    // texture version    
    vec2 p = floor(x);
	float a = textureLod(smp, (p + vec2(.5, .5)) / 256., 0.).x;
	float b = textureLod(smp, (p + vec2(1.5, .5)) / 256., 0.).x;
	float c = textureLod(smp, (p + vec2(.5, 1.5)) / 256., 0.).x;
	float d = textureLod(smp, (p + vec2(1.5, 1.5)) / 256., 0.).x;
#endif
    
	return vec3(a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y,
				6. * f * (1. - f) * (vec2(b - a, c - a) + (a - b - c + d) * u.yx));
}

// Function 1213
float noiseNew( in vec3 x )
{
	x += 0.5;
	vec3 fx = fract( x );
	x = floor( x ) + fx*fx*(3.0-2.0*fx);
    return texture( iChannel0, (x-0.5)/32.0 ).x;

}

// Function 1214
vec3 noised( in vec2 x )
{
    vec2 p = floor(x);
    vec2 w = fract(x);
    
    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec2 du = 30.0*w*w*(w*(w-2.0)+1.0);
    
    float a = hash1(p+vec2(0,0));
    float b = hash1(p+vec2(1,0));
    float c = hash1(p+vec2(0,1));
    float d = hash1(p+vec2(1,1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;

    return vec3( -1.0+2.0*(k0 + k1*u.x + k2*u.y + k4*u.x*u.y), 
                      2.0* du * vec2( k1 + k4*u.y,
                                      k2 + k4*u.x ) );
}

// Function 1215
float noise_scale()
{
    return 0.8;
}

// Function 1216
float noise(vec2 pos, float dist, float rotation, float time)
{
    time *= 1.+dist/100.;
    pos += vec2(time*rotation, 0.)*.5;
    pos -= vec2(iMouse.x/iResolution.x, iMouse.y/iResolution.y);
    return perlin(pos*dist + vec2(dist), time*2.);
}

// Function 1217
float noise(vec3 pos) {
    vec2 i = floor(pos.xy);
    vec2 f = pos.xy - i;
    vec2 blend = f * f * (3.0 - 2.0 * f);
    float noiseVal = 
        mix(
            mix(
                dot(GetGradient(i + vec2(0, 0), pos.z), f - vec2(0, 0)),
                dot(GetGradient(i + vec2(1, 0), pos.z), f - vec2(1, 0)),
                blend.x),
            mix(
                dot(GetGradient(i + vec2(0, 1), pos.z), f - vec2(0, 1)),
                dot(GetGradient(i + vec2(1, 1), pos.z), f - vec2(1, 1)),
                blend.x),
        blend.y
    );
    return noiseVal / 0.7; // normalize to about [-1..1]
}

// Function 1218
float getNoiseValue(vec2 p, float time)
{
    vec3 p3 = vec3(p.x, p.y, 0.0) + vec3(0.0, 0.0, time*0.025);
    float noise = simplex3d(p3*32.0);// simplex3d_fractal(p3*8.0+8.0);
	return 0.5 + 0.5*noise;
}

// Function 1219
float noise(in vec3 p) {
    float j = iTime * 0.045;
    float v = (sin3((p+vec3(j*7.0, j*2.3, j*1.0)) * 10.0) * freqs.w +
               sin3((p+vec3(j*3.0, j*1.2, j*0.4)) * 8.0) * freqs.z +
               sin3((p+vec3(j*2.4, j*0.6, j*2.6)) * 6.0) * freqs.y +
               sin3((p+vec3(j*1.4, j*5.8, j*1.9)) * 4.0) * freqs.x) * 0.25;
    //return 0.0;
    
    v = abs(v);
    float f = floor(v*10.0);
    
    v = clamp((smoothstep(0.0, 1.0, mix(0.1, 0.2, v*10.0-f)) + f)* 0.1, 0.0, 1.0);
    return v;
}

// Function 1220
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 1221
vec3 snoiseVec3( vec3 x )
{
  float numCells = 6.0;
  int octaves = 3;
   
  float s  = TileableNoiseFBM(vec3( x ), numCells, octaves);
  float s1 = TileableNoiseFBM(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ), numCells, octaves);
  float s2 = TileableNoiseFBM(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ), numCells, octaves);
  vec3 c = vec3( s , s1 , s2 );
  return c;

}

// Function 1222
float noise(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f*f*(3. -2.*f);
    
    return mix(mix(hash12(i + vec2(0, 0)), 
                   hash12(i + vec2(1, 0)), u.x), 
               mix(hash12(i + vec2(0, 1)), 
                   hash12(i + vec2(1, 1)), u.x), u.y);
}

// Function 1223
float noise(const in vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix( hash(p+vec3(0,0,0)), 
                        hash(p+vec3(1,0,0)),f.x),
                   mix( hash(p+vec3(0,1,0)), 
                        hash(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(p+vec3(0,0,1)), 
                        hash(p+vec3(1,0,1)),f.x),
                   mix( hash(p+vec3(0,1,1)), 
                        hash(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 1224
vec2 noise2( vec2 location, vec2 delta ) {
    const vec2 c = vec2(12.9898, 78.233);
    const float m = 43758.5453;
    return vec2(
        fract(sin(dot(location +      delta            , c)) * m),
        fract(sin(dot(location + vec2(delta.y, delta.x), c)) * m)
        );
}

// Function 1225
vec4 SNoise( in vec3 p )
{
	float f=noise(p);
	vec3 n=GrNoise(p);
	return vec4(n,f);
}

// Function 1226
vec4 snoiseProc( in vec3 x ) {
    x *= 32.0;
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix(mix(mix( hash43(p+vec3(0,0,0)), 
                        hash43(p+vec3(1,0,0)),f.x),
                   mix( hash43(p+vec3(0,1,0)), 
                        hash43(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash43(p+vec3(0,0,1)), 
                        hash43(p+vec3(1,0,1)),f.x),
                   mix( hash43(p+vec3(0,1,1)), 
                        hash43(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 1227
vec3 VoronoiColor(float density, vec2 uv, out float distance2border, out vec2 featurePt, out bool noTiles)
{
	float XYRatio = iResolution.x / iResolution.y;
	vec2 p = uv;
	p.x *= XYRatio;
	
    vec3 v = voronoi( density*p );
    distance2border = v.x;
    featurePt = v.yz;
	featurePt.x /= (density * XYRatio);
	featurePt.y /= density;
    
    //tile color = color at feature-point location
    vec2 uvCenter = uv;
    uvCenter.x += featurePt.x;
    uvCenter.y += featurePt.y;
  	
	vec3 color = vec3(0.0);

	//compute margin where no tiles are allowed
	if (abs(uvCenter.x)*XYRatio < MARGIN/density || abs(uvCenter.y) < MARGIN/density || abs(1.0 - uvCenter.x)*XYRatio < MARGIN/density || abs(1.0 - uvCenter.y) < MARGIN/density)
	{
		color = texture(iChannel0, uv).rgb;
		noTiles = true;
	}
    else
	{
		color = texture(iChannel0, uvCenter).rgb;
		noTiles = false;
	}
        
    return color;
}

// Function 1228
float noise( vec2 uv )
{
    
    vec2 lv = fract( uv );
    lv = lv * lv * ( 3.0 - 2.0 * lv );
    vec2 id = floor( uv );
    
    float bl = hash( id );
    float br = hash( id + vec2( 1, 0 ) );
    float b = mix( bl, br, lv.x );
    
    float tl = hash( id + vec2( 0, 1 ) );
    float tr = hash( id + vec2( 1 ) );
    float t = mix( tl, tr, lv.x );
    
    float c = mix( b, t, lv.y );
    
    return c;

}

// Function 1229
vec4 noised( in vec3 x )
{
    vec3 p = floor(x);
    vec3 w = fract(x);
    
#if 1
    // quintic interpolation
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
#else
    // cubic interpolation
    vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
#endif    
    
    
    float a = iqhash(p+vec3(0.0,0.0,0.0));
    float b = iqhash(p+vec3(1.0,0.0,0.0));
    float c = iqhash(p+vec3(0.0,1.0,0.0));
    float d = iqhash(p+vec3(1.0,1.0,0.0));
    float e = iqhash(p+vec3(0.0,0.0,1.0));
	float f = iqhash(p+vec3(1.0,0.0,1.0));
    float g = iqhash(p+vec3(0.0,1.0,1.0));
    float h = iqhash(p+vec3(1.0,1.0,1.0));
	
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y +
                 k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z,
                 du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                            k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                            k3 + k6*u.x + k5*u.y + k7*u.x*u.y ));
}

// Function 1230
vec3 voronoi( in vec3 uv, in vec3 no, inout float rough ) {
    
    vec3 center = floor(uv) + 0.5;
    vec3 bestCenterOffset = vec3(0);
    float bestDist = 9e9;
    vec3 bestCenterOffset2 = vec3(0);
    float bestDist2 = 9e9;
    
    for (float x = -0.5 ; x < 1.0 ; x+=1.0)
    for (float y = -0.5 ; y < 1.0 ; y+=1.0)
    for (float z = -0.5 ; z < 1.0 ; z+=1.0) {
		vec3 offset = vec3(x, y, z);
        vec3 newCenter = center + offset;
        vec3 newCenterOffset = hash(newCenter);
        vec3 temp = newCenterOffset - uv;
        float distSq = dot(temp, temp);
        if (distSq < bestDist) {
    		bestCenterOffset2 = bestCenterOffset;
    		bestDist2 = bestDist;
            bestCenterOffset = newCenterOffset;
            bestDist = distSq;
        } else if (distSq < bestDist2) {
            bestCenterOffset2 = newCenterOffset;
            bestDist2 = distSq;
        }
    }
    
    vec3 n1 = normalize(no + hash33(bestCenterOffset)-0.5);
    vec3 n2 = normalize(no + hash33(bestCenterOffset2)-0.5);
    float d = (sqrt(bestDist)-sqrt(bestDist2));
    float aad = 0.02;
    return mix(n1, n2, smoothstep(-aad, +aad, d*2.0));
}

// Function 1231
vec4 sNoise(float T,vec2 p
){T=(T-10.)/2.
 ;float N=floor(T),pT=T/10.,k=0.,o=1.
 ;p/=pow(2.,6.64385618977*(1.-pT))
 ;T-=floor(T)
 ;for(float z=0.;z<N;++z){k+=noise(p/o,.5,z)*o;o*=2.;}
 ;float m=T
 ;if(N<1.)m=1.
 ;k+=m*noise(p/o,0.5,N)*o
 ;k=.5+pow(2.,(1.-pT)*8.96578428466)*0.1*k
 ;return vec4(k,k,k,1.);}

// Function 1232
float metaNoise(vec2 uv)
{ 
    float density = mix(densityMin,densityMax,sin(iTime*densityEvolution)*0.5+0.5);
    return 1.0 - smoothstep(ballradius, ballradius+smoothing, metaNoiseRaw(uv, density));
}

// Function 1233
float noise(vec3 p) {
	const vec3 s = vec3(7.0, 157.0, 113.0);
	vec3 ip = floor(p);
    vec4 h = vec4(0.0, s.yz, s.y + s.z) + dot(ip, s);
	p -= ip;
	
    h = mix(fract(sin(h) * 43758.5453), fract(sin(h + s.x) * 43758.5453), p.x);
	
    h.xy = mix(h.xz, h.yw, p.y);
    return mix(h.x, h.y, p.z);
}

// Function 1234
float voronoi( in vec2 x )
{
	   
	vec2 n = floor( x );
	vec2 f = fract( x );

	vec3 m = vec3( 8.0 );
	for( int j=-1; j<=1; j++ )
	for( int i=-1; i<=1; i++ )
	{
		vec2  g = vec2( float(i), float(j) );
		vec2  o = hash( n + g );
		vec2  r = g - f + (0.5+0.5*sin(6.2831*o));
	float d = dot( r, r );
		if( d<m.x )
			m = vec3( d, o );
	}

	 vec2 c = vec2( sqrt(m.x), m.y+m.z );
	
	return 0.5 + 0.5*cos( c.y*6.2831 + 0.0 );	
}

// Function 1235
float noise12(vec2 p)
{
	float a = 0.0, b = a;
    for (int t = 0; t < ITERATIONS; t++)
    {
        float v = float(t+1)*.152;
        vec2 pos = (p * v + iTime * 1500. + 50.0);
        a += hash12(pos);
    }
    return a / float(ITERATIONS);
}

// Function 1236
float noise(vec2 p) {
  return random(p.x + p.y*10000.);
}

// Function 1237
vec4 texNoise(vec2 uv){ float f = 0.; f+=texture(iChannel0, uv*.125).r*.5;
    f+=texture(iChannel0,uv*.25).r*.25;f+=texture(iChannel0,uv*.5).r*.125;f+=texture(iChannel0,uv*1.).r*.125;f=pow(f,1.2);return vec4(f*.45+.05);
}

// Function 1238
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel0, (uv+ 0.5)/256.0).yx;
	return mix( rg.x, rg.y, f.z )*2.-1.;
}

// Function 1239
float map5_perlin1(vec3 p)  
{	
    float r = 0.;
    float s = 1.;
    float f = 1.;
    
    float w = 0.;
    for(int i = 0;i < 5;i++){
        r += s*noise_perlin1(p*f); w += s;
        s /= 2.;
        f *= 2.;
    }
    return r/w;
}

// Function 1240
vec3 TerrainNoise(in vec2 p, int octaves)
{
    // Samples the terrain heightmap.
    // There are certain parts of the terrain that could be better (which
    // I wont point out so hopefully you don't notice it too), but I just
    // really like the first mountain at the start of the scene.
    
    float sratio = DistanceRatioCamera(p);   // Ratio [0, 1] to the camera start position.
    float cratio = DistanceRatioCity(p);     // Ratio [0, 1] to the city position.
    
    float amplitude = 1.0;
    float value     = 0.0;                   // Cumulative noise value
    
	vec2  deriv     = vec2(0.0);             // Cumulative noise derivative
	vec2  samplePos = (p * TerrainTexScale);
    
    for(int i = 0; i < octaves; ++i)
    {
        // For each iteration we accumulate the noise value, derivative, and
        // scale/adjust the sample position.
        
        vec3 noise = TerrainNoiseRaw(samplePos);
        
        // 'noise.x * amplitude': Reduce the contribution of each successive iteration
        // '* (sratio + 0.1)': Flatten out the terrain at camera start/end
        // '/ (...)': Sharpen the mountain slopes
        
        deriv += noise.yz;
        value += (noise.x * amplitude * (sratio + 0.1)) / (0.9 + dot(deriv, deriv));
        
        amplitude *= TerrainPersistence;
        samplePos  = m2 * samplePos * 1.9;
    }
    
    // Here we compose the height range for the terrain.
    // sratio and cratio are used to form the 'craters' or terrain openings
    // that the camera starts at (sratio) and the city resides in (cratio).
    // Without this, the entire terrain would be mountains and unusable.
    // The '+ p.y * 0.01' just accentuates the peaks.
    
    float height = mix(100.0, TerrainMaxHeight + p.y * 0.01, min(sratio, cratio));
    
	return vec3(height * value, deriv);
}

// Function 1241
vec3 voronoi(vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);

	float id = 0.0;
    vec3 mg;
    vec3 mr;
    float res = 10.0;
    for(int k=0; k<=1; k++)
    for(int j=0; j<=1; j++)
    for(int i=0; i<=1; i++)
    {
        vec3 b = vec3(float(i), float(j), float(k));
        vec3 r = b - f + 0.45*hash(p + b);
        float d = dot(r, r);

        if(d < res)
        {
			id = dot(p + b, vec3(1.0,57.0,113.0));
            res = d;	
            mg = b;
            mr = r;
        }
    }
    
    float md2 = 8.0;
    for(int k=-1; k<=1; k++)
    for(int j=-1; j<=1; j++)
    for(int i=-1; i<=2; i++)
    {
        vec3 b = mg + vec3(float(i),float(j),float(k));;
        vec3 r = b - f + 0.45*hash(p + b);

        if(dot(mr-r,mr-r)>0.00001)
            md2 = 1./(1./md2 + 0.53/dot((mr + r*1.07), normalize(r*1.07 - mr)));
    }

    return vec3(res, md2, abs(id));
}

// Function 1242
vec2 voronoi( in vec2 x )
{
    vec2 n = floor( x );
    vec2 f = fract( x );

	vec3 m = vec3( 8.0 );
    for( float j=-1.; j<=1.; j++ )		// iterate cell neighbors
    for( float i=-1.; i<=1.; i++ )
    {
        vec2  g = vec2( i, j );			// vector holding offset to current cell
        vec2  o = hash2( n + g );		// unique random offset per cell
      	o.y*=.1;
        vec2  r = g - f + o;			// current pixel pos in local coords
	   
		float d = dot( r, r );			// squared dist from center of local coord system
        
        if( d<m.x )						// if dist is smallest...
            m = vec3( d, o );			// .. save new smallest dist and offset
    }

    return vec2( sqrt(m.x), m.y+m.z );
}

// Function 1243
vec8 perlin4x4 (vec4 p)
{
    // Address and interpolation values
    vec4 f = fract(p),
    m = f*f*f*(f*f*6. - f*15. + 10.);
    p -= f;

    // Interpolating the gradients for noise
    vec4 noise = mix(mix(mix(
        mix(dott(hash(p + vec4(0,0,0,0)), f - vec4(0,0,0,0)), 
            dott(hash(p + vec4(1,0,0,0)), f - vec4(1,0,0,0)), m.x), 
        mix(dott(hash(p + vec4(0,1,0,0)), f - vec4(0,1,0,0)), 
            dott(hash(p + vec4(1,1,0,0)), f - vec4(1,1,0,0)), m.x), m.y), mix(
        mix(dott(hash(p + vec4(0,0,1,0)), f - vec4(0,0,1,0)), 
            dott(hash(p + vec4(1,0,1,0)), f - vec4(1,0,1,0)), m.x), 
        mix(dott(hash(p + vec4(0,1,1,0)), f - vec4(0,1,1,0)), 
            dott(hash(p + vec4(1,1,1,0)), f - vec4(1,1,1,0)), m.x), m.y), m.z), mix(mix(
        mix(dott(hash(p + vec4(0,0,0,1)), f - vec4(0,0,0,1)), 
            dott(hash(p + vec4(1,0,0,1)), f - vec4(1,0,0,1)), m.x), 
        mix(dott(hash(p + vec4(0,1,0,1)), f - vec4(0,1,0,1)), 
            dott(hash(p + vec4(1,1,0,1)), f - vec4(1,1,0,1)), m.x), m.y), mix(
        mix(dott(hash(p + vec4(0,0,1,1)), f - vec4(0,0,1,1)), 
            dott(hash(p + vec4(1,0,1,1)), f - vec4(1,0,1,1)), m.x), 
        mix(dott(hash(p + vec4(0,1,1,1)), f - vec4(0,1,1,1)), 
            dott(hash(p + vec4(1,1,1,1)), f - vec4(1,1,1,1)), m.x), m.y), m.z), m.w);
    // Interpolating the values of the gradients for the first derivative normal
    // (of the alpha channel)
    // It's faster to recalculate the hashes by the way
    vec4 derivative = mix(mix(mix(
        mix(hash(p + vec4(0,0,0,0)).w, 
            hash(p + vec4(1,0,0,0)).w, m.x), 
        mix(hash(p + vec4(0,1,0,0)).w, 
            hash(p + vec4(1,1,0,0)).w, m.x), m.y), mix(
        mix(hash(p + vec4(0,0,1,0)).w, 
            hash(p + vec4(1,0,1,0)).w, m.x), 
        mix(hash(p + vec4(0,1,1,0)).w, 
            hash(p + vec4(1,1,1,0)).w, m.x), m.y), m.z), mix(mix(
        mix(hash(p + vec4(0,0,0,1)).w, 
            hash(p + vec4(1,0,0,1)).w, m.x), 
        mix(hash(p + vec4(0,1,0,1)).w, 
            hash(p + vec4(1,1,0,1)).w, m.x), m.y), mix(
        mix(hash(p + vec4(0,0,1,1)).w, 
            hash(p + vec4(1,0,1,1)).w, m.x), 
        mix(hash(p + vec4(0,1,1,1)).w, 
            hash(p + vec4(1,1,1,1)).w, m.x), m.y), m.z), m.w);
    return vec8(noise, derivative);
}

// Function 1244
float rnoise( in vec2 uv ){return 1. - abs(snoise(uv));}

// Function 1245
float noise( in vec3 uvt ) {
    vec2 p = uvt.xy;
    vec2 ft = fract(uvt.z * vec2(1.0, 1.0));
    vec2 i = floor(p+ft) + floor(uvt.z);
    vec2 f = fract( p +ft );
	vec2 u = f*f*(3.0-2.0*f);
    return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1246
vec3 voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
        }
    }

    //----------------------------------
    // second pass: distance to borders,
    // visits only neighbouring cells
    //----------------------------------
    md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) // skip the same cell
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec3( md, mr );
}

// Function 1247
float noise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);

	vec2 u = f * f * (3.0 - 2.0 * f);

	vec2 a = vec2(0.0, 0.0);
	vec2 b = vec2(1.0, 0.0);
	vec2 c = vec2(0.0, 1.0);
	vec2 d = vec2(1.0, 1.0);

	float n0 = hash(i + a);
	float n1 = hash(i + b);
	float n2 = hash(i + c);
	float n3 = hash(i + d);

	float ix0 = mix(n0, n1, u.x);
	float ix1 = mix(n2, n3, u.x);

	return mix(ix0, ix1, u.y);
}

// Function 1248
vec3 noise3(vec3 p) {
	return vec3(noise(p + vec3(0.268, 0.920, 0.015)),
                noise(p + vec3(0.143, 0.920, 0.578)),
                noise(p + vec3(0.229, 0.793, 0.670)));
}

// Function 1249
float smoothNoise(vec2 p) {

    vec2 interp = smoothstep(0., 1., fract(p));
    float s = mix(noise(sw(p)), noise(se(p)), interp.x);
    float n = mix(noise(nw(p)), noise(ne(p)), interp.x);
    return mix(s, n, interp.y);
        
}

// Function 1250
float gradientNoise(vec2 v) 
{
    vec2 i = floor(v);
    vec2 f = fract(v);
	
	vec2 u = smoothstep(0.0, 1.0, f);
	
	// random vectors at square corners
	vec2 randomA = random2(i + vec2(0.0,0.0));
	vec2 randomB = random2(i + vec2(1.0,0.0));
	vec2 randomC = random2(i + vec2(0.0,1.0));
	vec2 randomD = random2(i + vec2(1.0,1.0));
	
	// direction vectors from square corners to v
    //  directionN = v - (i + vec2(x,y)) = f - vec2(x,y)
	vec2 directionA = f - vec2(0.0,0.0);
	vec2 directionB = f - vec2(1.0,0.0);
	vec2 directionC = f - vec2(0.0,1.0);
	vec2 directionD = f - vec2(1.0,1.0);
	
	// "influence values"
	float a = dot(randomA, directionA);
	float b = dot(randomB, directionB);
	float c = dot(randomC, directionC);
	float d = dot(randomD, directionD);
	
    // final result: interpolate from corner values
	return mix( mix(a, b, u.x), mix(c, d, u.x), u.y );
}

// Function 1251
float voronoi_noise2(vec2 p){
	vec2 g = floor(p), o; p -= g;
	vec3 d = vec3(1.); 
    
	for(int y = -2; y <= 2; y++){
		for(int x = -2; x <= 2; x++){
            
			o = vec2(x, y);
            o += hash2_2(g + o) - p;
            
			d.z = max(dot(o.x, o.x), dot(o.y, o.y));    
            d.y = max(d.x, min(d.y, d.z));
            d.x = min(d.x, d.z); 
                       
		}
	}
    return max(d.y/1.2 - d.x*1., 0.)/1.2;  
}

// Function 1252
fractal noise (4 octaves)
    else	
	{
		uv *= 8.0;
        mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
		f  = 0.5000*noise( uv ); uv = m*uv;
		f += 0.2500*noise( uv ); uv = m*uv;
		f += 0.1250*noise( uv ); uv = m*uv;
		f += 0.0625*noise( uv ); uv = m*uv;
	}

// Function 1253
float simplexRot (vec2 u       ){return simplexRot (u  ,0.);}

// Function 1254
float noise(in float x) {
	float p = floor(x);
	float f = fract(x);
		
	f = f*f*(3.0-2.0*f);	
	return mix( hash(p+  0.0), hash(p+  1.0),f);
}

// Function 1255
float voronoi( vec2 pos, ivec2 ic )
{
	float mind = 1e6;
	for( int i = -1; i <= 1; i++ )
	{
		for( int j = -1; j <= 1; j++ )
		{
			vec2 pc = pointInCell( ic + ivec2( i, j ) );
			vec2 vd = ( pc - pos );
			float d = dot( vd, vd );
			mind = min( d, mind );
		}
	}
	return sqrt( mind );
}

// Function 1256
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0 + 113.0*p.z;

    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                        mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                        mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
    return res;
}

// Function 1257
float perlinNoise( vec2 uv )
{   
    float n = 		noise( uv * 1.0 ) 	* 128.0 +
        		noise( uv * 2.0 ) 	* 64.0 +
        		noise( uv * 4.0 ) 	* 32.0 +
        		noise( uv * 8.0 ) 	* 16.0 +
        		noise( uv * 16.0 ) 	* 8.0 +
        		noise( uv * 32.0 ) 	* 4.0 +
        		noise( uv * 64.0 ) 	* 2.0 +
        		noise( uv * 128.0 ) * 1.0;
    
    float noiseVal = n / ( 1.0 + 2.0 + 4.0 + 8.0 + 16.0 + 32.0 + 64.0 + 128.0 );
    noiseVal = abs(noiseVal * 2.0 - 1.0);
	
    return 	noiseVal;
}

// Function 1258
float noise_sum(vec2 p)
{
    float f = 0.0;
    p = p * 4.0;
    f += 1.0000 * noise(p); p = 2.0 * p;
    f += 0.5000 * noise(p); p = 2.0 * p;
	f += 0.2500 * noise(p); p = 2.0 * p;
	f += 0.1250 * noise(p); p = 2.0 * p;
	f += 0.0625 * noise(p); p = 2.0 * p;
    
    return f;
}

// Function 1259
float noise(vec3 p)
{
    p = floor(p);
    p = fract(p*vec3(283.343, 251.691, 634.127));
    p += dot(p, p+23.453);
    return fract(p.x*p.y);
}

// Function 1260
vec3 noise3( vec3 p )
{
	if (ANIM) p += iTime;
    float fx = noise(p);
    float fy = noise(p+vec3(1345.67,0,45.67));
    float fz = noise(p+vec3(0,4567.8,-123.4));
    return vec3(fx,fy,fz);
}

// Function 1261
float WaveletNoise(vec2 p, float z, float k) {
    // https://www.shadertoy.com/view/wsBfzK
    float d=0.,s=1.,m=0., a;
    for(float i=0.; i<4.; i++) {
        vec2 q = p*s, g=fract(floor(q)*vec2(123.34,233.53));
    	g += dot(g, g+23.234);
		a = fract(g.x*g.y)*1e3;// +z*(mod(g.x+g.y, 2.)-1.); // add vorticity
        q = (fract(q)-.5)*mat2(cos(a),-sin(a),sin(a),cos(a));
        d += sin(q.x*10.+z)*smoothstep(.25, .0, dot(q,q))/s;
        p = p*mat2(.54,-.84, .84, .54)+i;
        m += 1./s;
        s *= k; 
    }
    return d/m;
}

// Function 1262
float snoise(vec2 v) {

    // Precompute values for skewed triangular grid
    const vec4 C = vec4(0.211324865405187,
                        // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,
                        // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,
                        // -1.0 + 2.0 * C.x
                        0.024390243902439);
                        // 1.0 / 41.0

    // First corner (x0)
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);

    // Other two corners (x1, x2)
    vec2 i1 = vec2(0.0);
    i1 = (x0.x > x0.y)? vec2(1.0, 0.0):vec2(0.0, 1.0);
    vec2 x1 = x0.xy + C.xx - i1;
    vec2 x2 = x0.xy + C.zz;

    // Do some permutations to avoid
    // truncation effects in permutation
    i = mod289(i);
    vec3 p = permute(
            permute( i.y + vec3(0.0, i1.y, 1.0))
                + i.x + vec3(0.0, i1.x, 1.0 ));

    vec3 m = max(0.5 - vec3(
                        dot(x0,x0),
                        dot(x1,x1),
                        dot(x2,x2)
                        ), 0.0);

    m = m*m ;
    m = m*m ;
    // m /=m/0.02;
    m *=p*abs(sin(iTime/PI));

    // Gradients:
    //  41 pts uniformly over a line, mapped onto a diamond
    //  The ring size 17*17 = 289 is close to a multiple
    //      of 41 (41*7 = 287)

    vec3 x = 2.0 * fract(p * C.www) - 1.0*abs(sin(iTime*0.05));
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt(a0*a0 + h*h);
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0+h*h);

    // Compute final noise value at P
    vec3 g = vec3(0.0);
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * vec2(x1.x,x2.x) + h.yz * vec2(x1.y,x2.y);
    return 130.0 * dot(m, g);
}

// Function 1263
float triangleNoise(in vec2 p)
{
    float z=1.5;
    float z2=1.5;
	float rz = 0.;
    vec2 bp = p;
	for (float i=0.; i<=3.; i++ )
	{
        vec2 dg = tri2(bp*2.)*.8;
        dg *= mm2(time*.3);
        p += dg/z2;

        bp *= 1.6;
        z2 *= .6;
		z *= 1.8;
		p *= 1.2;
        p*= m2;
        
        rz+= (tri(p.x+tri(p.y)))/z;
	}
	return rz;
}

// Function 1264
float noise(vec3 p)
{
	vec3 ip=floor(p);
    p-=ip; 
    vec3 s=vec3(7,157,113);
    vec4 h=vec4(0.,s.yz,s.y+s.z)+dot(ip,s);
    p=p*p*(3.-2.*p); 
    h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
    h.xy=mix(h.xz,h.yw,p.y);
    return mix(h.x,h.y,p.z); 
}

// Function 1265
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
    vec2 uv = p.xy + f.xy;
	vec2 rg = vec2(texture( iChannel0, (uv+vec2(37.0,17.0)*p.z+0.5)/256.0, -100.0 ).x,
                   texture( iChannel0, (uv+vec2(37.0,17.0)*(p.z+1.0)+0.5)/256.0, -100.0 ).x );
	return mix( rg.x, rg.y, f.z );
}

// Function 1266
float noiseTex(in vec3 x)
{
    vec3 fl = floor(x);
    vec3 fr = fract(x);
	fr = fr * fr * (3.0 - 2.0 * fr);
	vec2 uv = (fl.xy + vec2(37.0, 17.0) * fl.z) + fr.xy;
	vec2 rg = textureLod(iChannel0, (uv + 0.5) * 0.00390625, 0.0 ).xy;
	return mix(rg.y, rg.x, fr.z);
}

// Function 1267
float noise_sum(vec3 p)
{
    float f = 0.0;
    p = p * 4.0;
    f += 1.0000 * noise(p); p = 2.0 * p;
    f += 0.5000 * noise(p); p = 2.0 * p;
	f += 0.2500 * noise(p); p = 2.0 * p;
	f += 0.1250 * noise(p); p = 2.0 * p;
	f += 0.0625 * noise(p); p = 2.0 * p;
    
    return f;
}

// Function 1268
vec4 Noise( in vec2 x )
{
    vec2 p = floor(x.xy);
    vec2 f = fract(x.xy);
	f = f*f*(3.0-2.0*f);
//	vec3 f2 = f*f; f = f*f2*(10.0-15.0*f+6.0*f2);

	vec2 uv = p.xy + f.xy;
	return textureLod( iChannel0, (uv+0.5)/256.0, 0.0 );
}

// Function 1269
vec4 texturenoiseLod( vec3 r, float lod )
{
    vec3 uvw = r / iChannelResolution[3];
    return textureLod( iChannel3, uvw, lod ) * 2. - 1.;
}

// Function 1270
vec3 GrNoiseK( in vec3 p ){
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = interp(f);
	vec3 du= Dinterp(f);

    vec3 v=mix( mix( mix( hash( i + vec3(0.0,0.0,0.0) ), 
                          hash( i + vec3(1.0,0.0,0.0) ), u.x),
                     mix( hash( i + vec3(0.0,1.0,0.0) ), 
                          hash( i + vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( hash( i + vec3(0.0,0.0,1.0) ), 
                          hash( i + vec3(1.0,0.0,1.0) ), u.x),
                     mix( hash( i + vec3(0.0,1.0,1.0) ), 
                          hash( i + vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
	
	v.x+=  mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), du.x),
                     mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), du.x), u.y),
                mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), du.x),
                     mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), du.x), u.y), u.z );
	v.x-=  mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                     dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), u.y),
                mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                     dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), u.y), u.z);

	v.y+=  mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), du.y),
                mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), du.y), u.z );
	v.y-=  mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                     dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                     dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x), u.z);
	
	v.z+=  mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), du.z );
	v.z-=  mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                     dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                     dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y);


	return v;
}

// Function 1271
float snoise(vec4 v)
  {
  const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
                        0.276393202250021,  // 2 * G4
                        0.414589803375032,  // 3 * G4
                       -0.447213595499958); // -1 + 4 * G4

// First corner
  vec4 i  = floor(v + dot(v, vec4(F4)) );
  vec4 x0 = v -   i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
  vec4 i0;
  vec3 isX = step( x0.yzw, x0.xxx );
  vec3 isYZ = step( x0.zww, x0.yyz );
//  i0.x = dot( isX, vec3( 1.0 ) );
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;
//  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;
  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;

  // i0 now contains the unique values 0,1,2,3 in each channel
  vec4 i3 = clamp( i0, 0.0, 1.0 );
  vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
  vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

  //  x0 = x0 - 0.0 + 0.0 * C.xxxx
  //  x1 = x0 - i1  + 1.0 * C.xxxx
  //  x2 = x0 - i2  + 2.0 * C.xxxx
  //  x3 = x0 - i3  + 3.0 * C.xxxx
  //  x4 = x0 - 1.0 + 4.0 * C.xxxx
  vec4 x1 = x0 - i1 + C.xxxx;
  vec4 x2 = x0 - i2 + C.yyyy;
  vec4 x3 = x0 - i3 + C.zzzz;
  vec4 x4 = x0 + C.wwww;

// Permutations
  i = mod289(i);
  float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute( permute( permute( permute (
             i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));

// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

  vec4 p0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);

// Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= taylorInvSqrt(dot(p4,p4));

// Mix contributions from the five corners
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
  m0 = m0 * m0;
  m1 = m1 * m1;
  return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
               + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;

}

// Function 1272
float achnoise(float x){
    float p = floor(x);
    float fr = fract(x);
    float L = p;
    float R = p + 1.0;

    float Lo = oct(L);
    float Ro = oct(R);

    return mix(Lo, Ro, fr);
}

// Function 1273
float noise(vec3 x) {
	vec3 p = floor(x);
	vec3 f = fract(x);
	f = f * f * (3.0 - 2.0 * f);

	float n = p.x + p.y * 157.0 + 113.0 * p.z;
	return mix(
			mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
					mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y),
			mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
					mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
}

// Function 1274
float fnoise(vec2 v) {
	return fract(sin(dot(v, vec2(12.9898, 78.233))) * 43758.5453) * 0.55;
}

// Function 1275
float noise(vec3 x)
{
    //x.x = mod(x.x, 0.4);
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix(hash(n+  0.0), hash(n+  1.0),f.x),
                   mix(hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix(hash(n+113.0), hash(n+114.0),f.x),
                   mix(hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 1276
float noise_2( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( hash12( i + vec2(0.0,0.0) ), 
                     hash12( i + vec2(1.0,0.0) ), u.x),
                mix( hash12( i + vec2(0.0,1.0) ), 
                     hash12( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1277
float noise( in vec3 x )
{
    vec3 p = floor(x);
	vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 1278
float noise3( vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + dot(p.yz,vec2(157.0,113.0));
    vec4 s1=mix(hash4(vec4(n)+NC0),hash4(vec4(n)+NC1),vec4(f.x));
    return mix(mix(s1.x,s1.y,f.y),mix(s1.z,s1.w,f.y),f.z);
}

// Function 1279
float noise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);

	// Four corners in 2D of a tile
	float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    // Simple 2D lerp using smoothstep envelope between the values.
	// return vec3(mix(mix(a, b, smoothstep(0.0, 1.0, f.x)),
	//			mix(c, d, smoothstep(0.0, 1.0, f.x)),
	//			smoothstep(0.0, 1.0, f.y)));

	// Same code, with the clamps in smoothstep and common subexpressions
	// optimized away.
    vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Function 1280
vec4 noised( in vec3 x )
{
    vec3 i = floor(x);
    vec3 w = fract(x);

/*// cubic interpolation used because it doesn't rush to extreme values too fast
#if 0
    // quintic interpolation
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
#else
    // cubic interpolation
    vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
#endif*/

// MODIFIED!
// Cosine interpolation seems to produce the fewest artifacts actually
vec3 u = 0.5-0.5*cos(3.1416 * w);
vec3 du = 0.5*3.1416*sin(3.1416 * w);
    
    
    float a = hash(i+vec3(0.0,0.0,0.0));
    float b = hash(i+vec3(1.0,0.0,0.0));
    float c = hash(i+vec3(0.0,1.0,0.0));
    float d = hash(i+vec3(1.0,1.0,0.0));
    float e = hash(i+vec3(0.0,0.0,1.0));
	float f = hash(i+vec3(1.0,0.0,1.0));
    float g = hash(i+vec3(0.0,1.0,1.0));
    float h = hash(i+vec3(1.0,1.0,1.0));
	
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return vec4( k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z, 
                 du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                            k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                            k3 + k6*u.x + k5*u.y + k7*u.x*u.y ) );
}

// Function 1281
float noise(vec3 p){//Noise function stolen from Virgil who stole it from Shane who I assume understands this shit, unlike me who is too busy contemplating if this cloud is indeed shaped like a penis
  vec3 ip=floor(p),s=vec3(7,157,113);
  p-=ip; vec4 h=vec4(0,s.yz,s.y+s.z)+dot(ip,s);
  p=p*p*(3.-2.*p);
  h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
  h.xy=mix(h.xz,h.yw,p.y);
  return mix(h.x,h.y,p.z);
}

// Function 1282
float smoothNoise2(vec2 p)
{
    vec2 p0 = floor(p + vec2(0.0, 0.0));
    vec2 p1 = floor(p + vec2(1.0, 0.0));
    vec2 p2 = floor(p + vec2(0.0, 1.0));
    vec2 p3 = floor(p + vec2(1.0, 1.0));
    vec2 pf = fract(p);
    return mix( mix(noise(p0), noise(p1), pf.x),mix(noise(p2), noise(p3), pf.x), pf.y);
}

// Function 1283
float iqsVoronoiDistanceInefficient( vec2 x ) {
    float minDist = 1.0;
    
    vec2 mb;
    vec2 mr;
    
    vec2 a, b;
    
    vec2 res = vec2(8.0);
    for(float i = 0.0; i < 100.0; ++i) {
        vec2 bufB_UV = vec2(i / float(iResolution.x), 0.0);
        vec2 particlePos = texture(iChannel0, bufB_UV).xy;
        
        vec2 r = particlePos - x;
        float dist = dot(r, r);
        
        if(dist < res.x) {
            res.y = dist;
            mr = r;
            a = r;
        }
        else if(dist < res.y) {
            res.y = dist;
            b = r;
        }
    }
    
    /*
    for(float i = 0.0; i < 100.0; ++i) {
        vec2 bufB_UV = vec2(i / float(iResolution.x), 0.0);
        vec2 particlePos = texture(iChannel0, bufB_UV).xy;
        vec2 r = particlePos - x;
        float dist = dot(0.5 * (mr + r), normalize(r - mr));
        res = min(res, dist);
    }

    return res;
	*/
    return dot(0.5 * (a + b), normalize(b - a));
}

// Function 1284
float fnoise(vec2 uv){
    // thanks iq
    float f = 0.0;
	mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
	f  = 0.5000*noise( uv ); uv = m*uv;
	f += 0.2500*noise( uv ); uv = m*uv;
	f += 0.1250*noise( uv ); uv = m*uv;
	f += 0.0625*noise( uv ); uv = m*uv;
    return f;
}

// Function 1285
float SpiralNoiseC(vec3 p)
{
    float n = 0.0;	// noise amount
    float iter = 2.0;
    for (int i = 0; i < 8; i++)
    {
        // add sin and cos scaled inverse with the frequency
        n += -abs(sin(p.y*iter) + cos(p.x*iter)) / iter;	// abs for a ridged look
        // rotate by adding perpendicular and scaling down
        p.xy += vec2(p.y, -p.x) * nudge;
        p.xy *= normalizer;
        // rotate on other axis
        p.xz += vec2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        // increase the frequency
        iter *= 1.733733;
    }
    return n;
}

// Function 1286
float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
    return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1287
float Noise(in vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel2, (uv+ 0.5)/256.0, 0.0 ).yx;

	return mix( rg.x, rg.y, f.z );
}

// Function 1288
vec3 voronoi_n(in vec2 rd, in vec2 n,  in vec2 f, 
               inout vec2 mg, inout vec2 mr) {
    float md = 1e5;
    vec2 mmg = mg;
    vec2 mmr = mr;
    vec2 ml = vec2(0.0, 0.0);
    
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {   
        vec2 g = mmg + vec2(i, j);
		vec2 o = hash2(n + g);
		vec2 r = g + o - f;

    	vec2 l = r - mmr;
        if (dot(rd, l) > 1e-5) {
            float d = dot(0.5*(mmr + r), l)/dot(rd, l);
            if (d < md) {
                md = d;
                mg = g;
                mr = r;
                ml = l;
            }
        }
    }
    
    return vec3(md, normalize(ml));
}

// Function 1289
float WorleyNoise(vec2 uv) {
    // Tile the space
    uv *= 1.0;
    vec2 uvInt = floor(uv);
    vec2 uvFract = fract(uv);
    float minDist = 1.0; // Minimum distance initialized to max.

    // Search all neighboring cells and this cell for their point
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));

            // Random point inside current neighboring cell
            vec2 point = random2(uvInt + neighbor);

            // Animate the point
            point = 0.5 + 0.5 * sin(iTime + 6.2831 * point); // 0 to 1 range

            // Compute the distance b/t the point and the fragment
            // Store the min dist thus far
            vec2 diff = neighbor + point - uvFract;
            float dist = length(diff);
            minDist = min(minDist, dist);
        }
    }
    return minDist;
}

// Function 1290
vec4 noise(ivec4 p){
    const float scale = pow(2., -32.);
    uvec4 h = hash(uvec4(p));
    return vec4(h)*scale;
}

// Function 1291
float noise3lin(in vec3 uvx) {
    vec3 f = fract(uvx);
    vec3 i = floor(uvx);
    
    float a1 = rand3(i);
    float b1 = rand3(i + vec3(0.0, 1.0, 0.0));
    float c1 = rand3(i + vec3(1.0, 0.0, 0.0));
    float d1 = rand3(i + vec3(1.0, 1.0, 0.0));
    float a2 = rand3(i + vec3(0.0, 0.0, 1.0));
    float b2 = rand3(i + vec3(0.0, 1.0, 1.0));
    float c2 = rand3(i + vec3(1.0, 0.0, 1.0));
    float d2 = rand3(i + vec3(1.0, 1.0, 1.0));
    
    vec3 u = -2. * f * f * f + 3. * f * f;
    
    float a = mix(a1, a2, f.z);
    float b = mix(b1, b2, f.z);
    float c = mix(c1, c2, f.z);
    float d = mix(d1, d2, f.z);
    
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.x);
}

// Function 1292
float noise(vec3 x){
    vec3 p = floor(x);
    vec3 w = fract(x);
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 ga = hash(p+vec3(0.0,0.0,0.0));
    vec3 gb = hash(p+vec3(1.0,0.0,0.0));
    vec3 gc = hash(p+vec3(0.0,1.0,0.0));
    vec3 gd = hash(p+vec3(1.0,1.0,0.0));
    vec3 ge = hash(p+vec3(0.0,0.0,1.0));
    vec3 gf = hash(p+vec3(1.0,0.0,1.0));
    vec3 gg = hash(p+vec3(0.0,1.0,1.0));
    vec3 gh = hash(p+vec3(1.0,1.0,1.0));
    float va = dot(ga, w-vec3(0.0,0.0,0.0));
    float vb = dot(gb, w-vec3(1.0,0.0,0.0));
    float vc = dot(gc, w-vec3(0.0,1.0,0.0));
    float vd = dot(gd, w-vec3(1.0,1.0,0.0));
    float ve = dot(ge, w-vec3(0.0,0.0,1.0));
    float vf = dot(gf, w-vec3(1.0,0.0,1.0));
    float vg = dot(gg, w-vec3(0.0,1.0,1.0));
    float vh = dot(gh, w-vec3(1.0,1.0,1.0));
    return va+
    u.x*(vb-va)+
    u.y*(vc-va)+
    u.z*(ve-va)+
    u.x*u.y*(va-vb-vc+vd)+
    u.y*u.z*(va-vc-ve+vg)+
    u.z*u.x*(va-vb-ve+vf)+
    u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);
}

// Function 1293
vec3 getVoronoi(vec2 x){
    vec2 n=floor(x),f=fract(x),mr;
    float md=5.;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ ){
        vec2 g=vec2(float(i),float(j));
		vec2 o=0.5+0.5*sin(iTime+6.2831*getHash2BasedProc(n+g));//animated
        vec2 r=g+o-f;
        float d=dot(r,r);
        if( d<md ) {md=d;mr=r;} }
    return vec3(md,mr);}

// Function 1294
float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
           -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
    dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 0.5+0.5*(130.0 * dot(m, g));
}

// Function 1295
float getNoise(vec2 noiseCoord, float angleOffset, float noiseSeed, float r) {
    float angle = noise_period * time;
    angle += angleOffset;
    r *= space;
    noiseCoord += vec2(
        floor(r * sin(angle)),
        floor(r * cos(angle))
    );
    
    noiseCoord.x += noiseSeed;
    
    return rand(noiseCoord);    
}

// Function 1296
vec3 voronoi( in vec2 x, in vec2 dir)
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 1e5;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;

    
 		if( dot(r-mr,r-mr) > 1e-5 ) {
            
            vec2 f = r-mr;
            
            if (dot(dir, f) > 1e-5) {
            	vec2 m = 0.5*(mr+r);
   	 			float c = m.x*(m.y + f.x) - m.y*(m.x - f.y);
            	float d = 1.0 / dot(dir, f);
                
            	md = min(md, dot(dir, dir*c*d));
            }
        }
        
    }
    
    return vec3( md, n+mg);
}

// Function 1297
float InterferenceSmoothNoise1D( float x )
{
    float f0 = floor(x);
    float fr = fract(x);

    float h0 = InterferenceHash( f0 );
    float h1 = InterferenceHash( f0 + 1.0 );

    return h1 * fr + h0 * (1.0 - fr);
}

// Function 1298
float noise2(vec3 pos)
{
    vec3 q = 8.0*pos;
    float f  = 0.5000*noise( q ); q = m*q*2.01;
    f += 0.2500*noise( q ); q = m*q*2.02;
    f += 0.1250*noise( q ); q = m*q*2.03;
    f += 0.0625*noise( q ); q = m*q*2.01;
    return f;
}

// Function 1299
float SpiralNoiseC(vec3 p)
{
    float n = -mod(iTime * 0.2,-2.); // noise amount
    float iter = 2.0;
    for (int i = 0; i < 8; i++)
    {
        // add sin and cos scaled inverse with the frequency
        n += -abs(sin(p.y*iter) + cos(p.x*iter)) / iter;	// abs for a ridged look
        // rotate by adding perpendicular and scaling down
        p.xy += vec2(p.y, -p.x) * nudge;
        p.xy *= normalizer;
        // rotate on other axis
        p.xz += vec2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        // increase the frequency
        iter *= 1.733733;
    }
    return n;
}

// Function 1300
float noise( in vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 1301
float smoothVoronoi(vec2 p, float falloff) {

    vec2 ip = floor(p); p -= ip;
	
	float d = 1., res = 0.;
	
	for(int i=-1; i<=2; i++) {
		for(int j=-1; j<=2; j++) {
            
			vec2 b = vec2(i, j);
            
			vec2 v = b - p + hash22(ip + b);
            
			d = max(dot(v,v), 1e-8);
			
			res += 1./pow(d, falloff );
            //res += exp( -16.*d ); // Alternate version.
		}
	}

	return pow(1./res, .5/falloff);
    //return clamp((-(1./16.)*log(res) + .1)/1.1, 0., 1.); // Alternate version.
}

// Function 1302
float noise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 1303
vec2 TriangularNoise2DShereRay(vec2 n,float time){
	float theta = TWO_PI*GetRandom();
    float r = TriangularNoise(n,time);
    return vec2(cos(theta),sin(theta))*(1.-r);
}

// Function 1304
float noise(in vec3 x) {
    vec3 p = floor(x);
    vec3 f = x - p;
    f = f * dot(f, vec3(3.0,3.0,3.0)-2.0*f);
    float n = dot(p, vec3(1.0, 57.0, 113.0));
    return mix(mix(mix(hash(n +  0.0), hash(n +  1.0), clamp(f.x,0.,1.)),
                   mix(hash(n + 57.0), hash(n + 58.0), clamp(f.x,0.,1.)),
                   clamp(f.y,0.,1.)),
               mix(mix(hash(n + 113.0), hash(n + 114.0), clamp(f.x,0.,1.)),
                   mix(hash(n + 170.0), hash(n + 171.0), clamp(f.x,0.,1.)),
                   clamp(f.y,0.,1.)),
               clamp(f.z,0.,1.));
}

// Function 1305
vec3 divfreenoise( vec3 q ) { // fluid-like noise = div-free -> curl
    vec2 e = vec2(1./16.,0);
 // q += .1*iTime;            // animated flow
    vec3 v = noise(q); 
 // return v -.5;             // regular
    return vec3( noise(q+e.yxy).z-v.z - v.y+noise(q+e.yyx).y, // curl
                 noise(q+e.yyx).x-v.x - v.z+noise(q+e.xyy).z,
                 noise(q+e.xyy).y-v.y - v.x+noise(q+e.yxy).x
                ) *1.;
}

// Function 1306
F1 Noise(F2 n,F1 x){n+=x;return fract(sin(dot(n.xy,F2(12.9898, 78.233)))*43758.5453);}

// Function 1307
float noise(vec2 p)
{
    vec2 pi = floor(p);
    vec2 pf = smoothstep(0., 1., p - pi);
    return mix(
        mix(hash(pi), 	   hash(pi+o), pf.x), 
        mix(hash(pi+o.yx), hash(pi+o.xx), pf.x), 
        pf.y);
}

// Function 1308
vec4 noise(vec3 p, float lod){float m = mod(p.z,1.0);float s = p.z-m; float sprev = s-1.0;if (mod(s,2.0)==1.0) { s--; sprev++; m = 1.0-m; };return mix(texture(iChannel0,p.xy/iChannelResolution[0].xy + noise(sprev,lod).yz,lod*21.421),texture(iChannel0,p.xy/iChannelResolution[0].xy + noise(s,lod).yz,lod*14.751),m);}

// Function 1309
float Noise(in vec2 p, in float scale )
{
	vec2 f;
	
	p *= scale;

	
	f = fract(p);		// Separate integer from fractional
    p = floor(p);
	
    f = f*f*(3.0-2.0*f);	// Cosine interpolation approximation
	
    float res = mix(mix(Hash(p, 				 scale),
						Hash(p + vec2(1.0, 0.0), scale), f.x),
					mix(Hash(p + vec2(0.0, 1.0), scale),
						Hash(p + vec2(1.0, 1.0), scale), f.x), f.y);
    return res;
}

// Function 1310
float noise( in vec3 x )
{
    vec3 f = fract(x);
    vec3 p = floor(x);
    f = f * f * (3.0 - 2.0 * f);
     
    vec2 uv = (p.xy + vec2(37.0, 17.0) * p.z) + f.xy;
    vec2 rg = texture(iChannel0, (uv + 0.5)/256.0, -100.0).yx;
    return mix(rg.x, rg.y, f.z);
}

// Function 1311
float PhiNoise(uvec3 uvw)
{
    // flip every other tile to reduce anisotropy
    if(((uvw.x ^ uvw.y ^ uvw.z) & 4u) == 0u) uvw = uvw.yzx;
    
    // constants of 3d Roberts sequence rounded to nearest primes
    const uint r0 = 3518319149u;// prime[(2^32-1) / phi_3  ]
    const uint r1 = 2882110339u;// prime[(2^32-1) / phi_3^2]
    const uint r2 = 2360945581u;// prime[(2^32-1) / phi_3^3]
    
    // h = high-freq dither noise
    uint h = (uvw.x * r0) + (uvw.y * r1) + (uvw.z * r2);
    
    // l = low-freq white noise
    uvw = uvw >> 2u;// 3u works equally well (I think)
    uint l = ((uvw.x * r0) ^ (uvw.y * r1) ^ (uvw.z * r2)) * r1;
    
    // combine low and high
    return float(l + h) * (1.0 / 4294967296.0);
}

// Function 1312
vec4 bluenoise(int s, vec2 fc) {
    vec2 blue_noise_res = iChannelResolution[0].xy;
    vec2 tileRes = vec2(256.0, 256.0);
	int tilex = s % 4;
    int tiley = s / 4;
    
    vec2 coord = mod(fc, tileRes);
    
    return texture( iChannel0, (vec2(tilex, tiley) * tileRes + coord) / blue_noise_res);
}

// Function 1313
float noise(vec3 x) { const vec3 step = vec3(110, 241, 171); vec3 i = floor(x); vec3 f = fract(x); float n = dot(i, step); vec3 u = f * f * (3.0 - 2.0 * f); return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x), mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y), mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x), mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z); }

// Function 1314
float uniformNoise(vec2 n){
    // uniformly distribued, normalized in [0..1[
    return fract(sin(dot(n, vec2(12.9898, 78.233))) * 43758.5453);
}

// Function 1315
float simplex3d(vec3 p) {
	 vec3 s = floor(p + dot(p, vec3(F3)));
	 vec3 x = p - s + dot(s, vec3(G3));
	 
	 vec3 e = step(vec3(0.0), x - x.yzx);
	 vec3 i1 = e*(1.0 - e.zxy);
	 vec3 i2 = 1.0 - e.zxy*(1.0 - e);
	 
	 vec3 x1 = x - i1 + G3;
	 vec3 x2 = x - i2 + 2.0*G3;
	 vec3 x3 = x - 1.0 + 3.0*G3;
	 
	 vec4 w, d;
	 
	 w.x = dot(x, x);
	 w.y = dot(x1, x1);
	 w.z = dot(x2, x2);
	 w.w = dot(x3, x3);
	 
	 w = max(0.6 - w, 0.0);
	 
	 d.x = dot(random3(s), x);
	 d.y = dot(random3(s + i1), x1);
	 d.z = dot(random3(s + i2), x2);
	 d.w = dot(random3(s + 1.0), x3);
	 
	 w *= w;
	 w *= w;
	 d *= w;
	 
	 return dot(d, vec4(52.0));
}

// Function 1316
vec2 Noise2( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 res = mix(mix( Hash2(p), Hash2(p + vec2(1.0,0.0)),f.x),
                   mix( Hash2(p + vec2(0.0,1.0)), Hash2(p + vec2(1.0,1.0)),f.x),f.y);
    return res-vec2(.5);
}

// Function 1317
float SeaNoise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    float res = mix(mix( Hash(n+  0.0), Hash(n+  1.0),f.x),
                    mix( Hash(n+ 57.0), Hash(n+ 58.0),f.x),f.y);
    return res;
}

// Function 1318
float snoise(vec2 v)
		{
				const vec4 C = vec4(0.211324865405187,0.366025403784439,-0.577350269189626,0.024390243902439);
				vec2 i  = floor(v + dot(v, C.yy) );
				vec2 x0 = v -   i + dot(i, C.xx);
				
				vec2 i1;
				i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
				vec4 x12 = x0.xyxy + C.xxzz;
				x12.xy -= i1;
				
				i = mod289(i); // Avoid truncation effects in permutation
				vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
					+ i.x + vec3(0.0, i1.x, 1.0 ));
				
				vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
				m = m*m ;
				m = m*m ;
				
				vec3 x = 2.0 * fract(p * C.www) - 1.0;
				vec3 h = abs(x) - 0.5;
				vec3 ox = floor(x + 0.5);
				vec3 a0 = x - ox;
				
				m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
				
				vec3 g;
				g.x  = a0.x  * x0.x  + h.x  * x0.y;
				g.yz = a0.yz * x12.xz + h.yz * x12.yw;

				return 130.0 * dot(m, g);		
		}

// Function 1319
float noise(in vec3 p)
{
    return noise(p.xy);
}

// Function 1320
vec3 perlinNoised(vec2 pos, vec2 scale, mat2 transform, float seed)
{
    // based on Modifications to Classic Perlin Noise by Brian Sharpe: https://archive.is/cJtlS
    pos *= scale;
    vec4 i = floor(pos).xyxy + vec2(0.0, 1.0).xxyy;
    vec4 f = (pos.xyxy - i.xyxy) - vec2(0.0, 1.0).xxyy;
    i = mod(i, scale.xyxy) + seed;

    // grid gradients
    vec4 gradientX, gradientY;
    multiHash2D(i, gradientX, gradientY);
    gradientX -= 0.49999;
    gradientY -= 0.49999;

    // transform gradients
    vec4 mt = vec4(transform);
    vec4 rg = vec4(gradientX.x, gradientY.x, gradientX.y, gradientY.y);
    rg = rg.xxzz * mt.xyxy + rg.yyww * mt.zwzw;
    gradientX.xy = rg.xz;
    gradientY.xy = rg.yw;

    rg = vec4(gradientX.z, gradientY.z, gradientX.w, gradientY.w);
    rg = rg.xxzz * mt.xyxy + rg.yyww * mt.zwzw;
    gradientX.zw = rg.xz;
    gradientY.zw = rg.yw;
    
    // perlin surflet
    vec4 gradients = inversesqrt(gradientX * gradientX + gradientY * gradientY) * (gradientX * f.xzxz + gradientY * f.yyww);
    vec4 m = f * f;
    m = m.xzxz + m.yyww;
    m = max(1.0 - m, 0.0);
    vec4 m2 = m * m;
    vec4 m3 = m * m2;
    // compute the derivatives
    vec4 m2Gradients = -6.0 * m2 * gradients;
    vec2 grad = vec2(dot(m2Gradients, f.xzxz), dot(m2Gradients, f.yyww)) + vec2(dot(m3, gradientX), dot(m3, gradientY));
    // sum the surflets and normalize: 1.0 / 0.75^3
    return vec3(dot(m3, gradients), grad) * 2.3703703703703703703703703703704;
}

// Function 1321
vec3 TextureNoise(vec2 uvs)
{
    return textureLod(iChannel3, uvs, 0.0).rgb;
}

// Function 1322
float noise(float x) {
    float i = floor(x), f = fract(x);
    float u =         
        // 0.5; // Constant
        // f; // Linear
        f * f * (3.0 - 2.0 * f); // Cubic
    
    return 2.0 * mix(hash(i), hash(i + 1.0), u) - 1.0;
}

// Function 1323
float snoise(vec4 v){
  const vec2  C = vec2( 0.138196601125010504,  // (5 - sqrt(5))/20  G4
                        0.309016994374947451); // (sqrt(5) - 1)/4   F4
// First corner
  vec4 i  = floor(v + dot(v, C.yyyy) );
  vec4 x0 = v -   i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
  vec4 i0;

  vec3 isX = step( x0.yzw, x0.xxx );
  vec3 isYZ = step( x0.zww, x0.yyz );
//  i0.x = dot( isX, vec3( 1.0 ) );
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;

//  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;

  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;

  // i0 now contains the unique values 0,1,2,3 in each channel
  vec4 i3 = clamp( i0, 0.0, 1.0 );
  vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
  vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

  //  x0 = x0 - 0.0 + 0.0 * C 
  vec4 x1 = x0 - i1 + 1.0 * C.xxxx;
  vec4 x2 = x0 - i2 + 2.0 * C.xxxx;
  vec4 x3 = x0 - i3 + 3.0 * C.xxxx;
  vec4 x4 = x0 - 1.0 + 4.0 * C.xxxx;

// Permutations
  i = mod(i, 289.0); 
  float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute( permute( permute( permute (
             i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));
// Gradients
// ( 7*7*6 points uniformly over a cube, mapped onto a 4-octahedron.)
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.

  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

  vec4 p0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);

// Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= taylorInvSqrt(dot(p4,p4));

// Mix contributions from the five corners
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
  m0 = m0 * m0;
  m1 = m1 * m1;
  return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
               + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;

}

// Function 1324
float noise( in vec2 x ) {
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    return mix(mix( hash(n + 0.0), hash(n + 1.0), f.x), mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y);
}

// Function 1325
float noise(in vec2 p)
{
    vec2 f = fract(p);
    p = floor(p);
    float v = p.x + p.y * 1000.0;
    vec4 r = vec4(v, v + 1.0, v + 1000.0, v + 1001.0);
    r = fract(100000.0 * sin(r * 0.001));
    f = f * f * (3.0 - 2.0 * f);
    return 2.0 * (mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y)) - 1.0;
}

// Function 1326
float smoothNoise2(vec2 p){vec2 pf=fract(p);
 return mix(mix(noise(floor(p          )),noise(floor(p+vec2(1,0))),pf.x),
             mix(noise(floor(p+vec2(0,1))),noise(floor(p+vec2(1,1))),pf.x),pf.y);}

// Function 1327
float vnoise1(float p) {
    float i = floor(p);
	float f = fract(p);
    
    float a = hash11(i);
    float b = hash11(i + 1.0);
    
    float u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u);
}

// Function 1328
float noise2D(vec2 uv) {
    vec2 st = 0.1 * uv; 
    vec2 i = floor(st);
    vec2 f = fract(st);
    
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    
    vec2 u = f*f*(3.0-2.0*f);
    float a1 = mix(a, b, u.x);
    float a2 = mix(c, d, u.x);
    float a3 = mix(a1, a2, u.y);
    return clamp(a3, 0.0, 1.0); 
}

// Function 1329
float valueNoise(float t){       
    float i = fract(t);
    float f = floor(t);
	float a = rand(f);
    float b = rand(f + 1.);
    return mix(a, b, i);
}

// Function 1330
float noise2( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	//f = f*f*(3.0-2.0*f);
    f = (f*f*(3.0-2.0*f)+f)*0.5;
#ifndef HIGH_QUALITY_NOISE
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0. ).yx;
#else
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z);
	vec2 rg1 = textureLod( iChannel0, (uv+ vec2(0.5,0.5))/256.0, 0. ).yx;
	vec2 rg2 = textureLod( iChannel0, (uv+ vec2(1.5,0.5))/256.0, 0. ).yx;
	vec2 rg3 = textureLod( iChannel0, (uv+ vec2(0.5,1.5))/256.0, 0. ).yx;
	vec2 rg4 = textureLod( iChannel0, (uv+ vec2(1.5,1.5))/256.0, 0. ).yx;
	vec2 rg = mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
#endif	
	return mix( rg.x, rg.y, f.z );
}

// Function 1331
float noise(float x)
{
    float p=floor(x);
    float f=fract(x);
    f=f*f*(3.-2.*f);
    return mix(hash11(p),hash11(p+1.0),f);
}

// Function 1332
float noise(vec2 p) {
    vec2 n = floor(p);
    vec2 f = fract(p);
    f = f * f * f * (3.0 - 2.0 * f);
    vec2 add = vec2(1.0, 0.0);
    float h = mix( mix(hash(n+add.yy), hash(n+add.xy), f.x), 
                   mix(hash(n+add.yx), hash(n+add.xx), f.x), f.y);
        
    return h;
}

// Function 1333
float perlinNoise(vec2 pos, vec2 scale, mat2 transform, float seed)
{
    // based on Modifications to Classic Perlin Noise by Brian Sharpe: https://archive.is/cJtlS
    pos *= scale;
    vec4 i = floor(pos).xyxy + vec2(0.0, 1.0).xxyy;
    vec4 f = (pos.xyxy - i.xyxy) - vec2(0.0, 1.0).xxyy;
    i = mod(i, scale.xyxy) + seed;

    // grid gradients
    vec4 gradientX, gradientY;
    multiHash2D(i, gradientX, gradientY);
    gradientX -= 0.49999;
    gradientY -= 0.49999;

    // transform gradients
    vec4 m = vec4(transform);
    vec4 rg = vec4(gradientX.x, gradientY.x, gradientX.y, gradientY.y);
    rg = rg.xxzz * m.xyxy + rg.yyww * m.zwzw;
    gradientX.xy = rg.xz;
    gradientY.xy = rg.yw;

    rg = vec4(gradientX.z, gradientY.z, gradientX.w, gradientY.w);
    rg = rg.xxzz * m.xyxy + rg.yyww * m.zwzw;
    gradientX.zw = rg.xz;
    gradientY.zw = rg.yw;

    // perlin surflet
    vec4 gradients = inversesqrt(gradientX * gradientX + gradientY * gradientY) * (gradientX * f.xzxz + gradientY * f.yyww);
    // normalize: 1.0 / 0.75^3
    gradients *= 2.3703703703703703703703703703704;
    f = f * f;
    f = f.xzxz + f.yyww;
    vec4 xSq = 1.0 - min(vec4(1.0), f); 
    return dot(xSq * xSq * xSq, gradients);
}

// Function 1334
vec2 Noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 res = mix(mix( Hash(p + 0.0), Hash(p + vec2(1.0, 0.0)),f.x),
                   mix( Hash(p + vec2(0.0, 1.0) ), Hash(p + vec2(1.0, 1.0)),f.x),f.y);
    return res-.5;
}

// Function 1335
float FractalNoise(in vec2 xy)
{
	float w = 1.5;
	float f = 0.0;
    xy *= .08;

	for (int i = 0; i < 5; i++)
	{
		f += texture(iChannel2, .5+xy * w, -99.0).x / w;
		w += w;
	}
	return f*.8;
}

// Function 1336
vec3 valueNoiseD(in vec2 p)
{
    vec2 X = floor(p);
    vec2 x = fract(p);
    
    vec2 fn = x * x * x * (6.0 * x * x - 15.0 * x + 10.0);
    vec2 dfn = 30.0 * x * x * (x * x - 2.0 * x + 1.0);
    float u = fn.x;
    float v = fn.y;
    float du = dfn.x;
    float dv = dfn.y;
    
    float a = rand(X + vec2(0.0, 0.0));
    float b = rand(X + vec2(1.0, 0.0));
    float c = rand(X + vec2(0.0, 1.0));
    float d = rand(X + vec2(1.0, 1.0));
    
    float n = 2.0 * ((b - a) * u + (c - a) * v + (a - b - c + d) * u * v + a) - 1.0;
    float dnu = 2.0 * du * ((b - a) + (a - b - c + d) * v);
    float dnv = 2.0 * dv * ((c - a) + (a - b - c + d) * u);
    return vec3(n, dnu, dnv);
}

// Function 1337
float voronoi2D(vec2 n, float time){
    float dis = 0.9;
    for(float y = 0.0; y <= 1.0; y++){
        for(float x = 0.0; x <= 1.0; x++){
            // Neighbor place in the grid
            vec2 p = floor(n) + vec2(x, y);
            float d = length((0.27 * sin(rand2D(p) * intensity + time * 2.0)) + vec2(x, y) - fract(n));
            dis = min(dis, d);
            }
        }
    return dis;
    }

// Function 1338
float ipnoise(vec3 p)
{
  vec3 P  = mod(floor(p), 256.0);
//       p -= floor(p);
  p = fract(p);
  vec3 f  = fade(p);

  // HASH COORDINATES FOR 6 OF THE 8 CUBE CORNERS
  float A  = perm(P.x      ) + P.y;
  float AA = perm(A        ) + P.z;
  float AB = perm(A   + 1.0) + P.z;
  float B  = perm(P.x + 1.0) + P.y;
  float BA = perm(B        ) + P.z;
  float BB = perm(B   + 1.0) + P.z;

  // AND ADD BLENDED RESULTS FROM 8 CORNERS OF CUBE
  return mix(
    mix(mix( grad(perm(AA     ), p),
             grad(perm(BA     ), p + vec3(-1.,  0.,  0.)), f.x),
         mix(grad(perm(AB     ), p + vec3( 0., -1.,  0.)),
             grad(perm(BB     ), p + vec3(-1., -1.,  0.)), f.x), f.y),
    mix(mix( grad(perm(AA + 1.), p + vec3( 0.,  0., -1.)),
             grad(perm(BA + 1.), p + vec3(-1.,  0., -1.)), f.x),
         mix(grad(perm(AB + 1.), p + vec3( 0., -1., -1.)),
             grad(perm(BB + 1.), p + vec3(-1., -1., -1.)), f.x), f.y),
    f.z);
}

// Function 1339
void voronoi_s(in vec2 x, inout vec2 n,  inout vec2 f, 
                          inout vec2 mg, inout vec2 mr) {

    n = floor(x);
    f = fract(x);

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if (d < md) {
            md = d;
            mr = r;
            mg = g;
        }
    }   
}

// Function 1340
float noiseSource(vec2 uv, float t){
    
    float waveA = sin(
        uv.x*150.0 +
        sin(uv.x*15.1)*(1.5+sin(uv.y*1.1)*0.3) +
        t*(1.0+sin(t*0.25)*0.005) +
        sin(uv.y*100.0 + sin(uv.x*110.0) + t*5.0) +
        sin(uv.y*8.0 + t*0.2)*15.0 +
        sin(uv.y* (150.0+sin(uv.y*20.0)*4.0+sin(uv.x*10.0)*3.5) + t*1.0)
    );
    
    float waveB = sin(
        uv.y*140.0 +
        sin(uv.y*15.1)*(1.5+sin(uv.x*1.1)*0.3) +
        t*(1.1+sin(t*0.2)*0.004) +
        sin(uv.x*110.0 + sin(uv.y*120.0) + t*4.0) +
        sin(uv.x*9.0 + t*0.1)*14.0 +
        sin(uv.x* (160.0+sin(uv.x*25.0)*5.0+sin(uv.y*12.0)*2.5) + t*2.0 )
    );
    
    return pow((waveA*0.5+0.5)*(waveB*0.5+0.5),0.5);
    
}

// Function 1341
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix( hash(p+vec3(0,0,0)), 
                        hash(p+vec3(1,0,0)),f.x),
                   mix( hash(p+vec3(0,1,0)), 
                        hash(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(p+vec3(0,0,1)), 
                        hash(p+vec3(1,0,1)),f.x),
                   mix( hash(p+vec3(0,1,1)), 
                        hash(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 1342
float noise( in vec3 x )
{
	float  z = x.z*64.0;
	vec2 offz = vec2(0.317,0.123);
	vec2 uv1 = x.xy + offz*floor(z); 
	vec2 uv2 = uv1  + offz;
	return mix(texture( iChannel0, uv1 ,-100.0).x,texture( iChannel0, uv2 ,-100.0).x,fract(z))-0.5;
}

// Function 1343
bool iSimplexPrim(vec3 p[4], 
	vec4 c, vec3 m1, vec3 m2, 
	vec3 ro, vec3 rd, 
	out Hit near, out Hit far) {
    
    vec3 sn[3];
    compute_planes(p, sn);
    
    // convert ray endpoints to barycentric basis
    // this can be optimized further by caching the determinant
    vec4 r0 = to_bary(p[3], sn, ro);
    vec4 r1 = to_bary(p[3], sn, ro + rd);

    // build barycentric ray direction from endpoints
    vec4 brd = r1 - r0;
    // compute ray scalars for each plane
    vec4 t = -r0/brd;
    
    // valid since GL 4.1
    near.t = -1.0 / 0.0;
    far.t = 1.0 / 0.0;    
#if 0
    for (int i = 0; i < 4; ++i) {
        // equivalent to checking dot product of ray dir and plane normal
        if (brd[i] < 0.0) {
            far.t = min(far.t, t[i]);
        } else {
            near.t = max(near.t, t[i]);
        }
    }
#else
    // loopless, branchless alternative
    // equivalent to checking dot product of ray dir and plane normal    
    bvec4 comp = lessThan(brd, vec4(0.0));
    vec4 far4 = mix(vec4(far.t), t, comp);
    vec4 near4 = mix(t, vec4(near.t), comp);
    far.t = min(min(far4.x,far4.y),min(far4.z,far4.w));
    near.t = max(max(near4.x,near4.y),max(near4.z,near4.w));
#endif
    
    if ((far.t <= 0.0) || (far.t <= near.t))
        return false;
    near.b = r0 + brd * near.t;
    far.b = r0 + brd * far.t;

#ifdef HIT_TET_PLANES
    vec4 n0 = select_plane_normal(near.b);
    vec4 n1 = select_plane_normal(far.b);
#else
    vec4 n0;
    vec4 n1;
#endif
        
#if 1
    // reconstruct 1D quadratic coefficients from three samples
    float c0 = eval_quadratic(near.b, c, m1, m2);
    float c1 = eval_quadratic(r0 + brd * (near.t + far.t) * 0.5, c, m1, m2);
    float c2 = eval_quadratic(far.b, c, m1, m2);

    float A = 2.0*(c2 + c0 - 2.0*c1);
    float B = 4.0*c1 - 3.0*c0 - c2;
    float C = c0;
    
    if (A == 0.0) return false;
    // solve quadratic
    float k = B*B - 4.0*A*C;
    if (k < 0.0)
        return false;
    k = sqrt(k);
    float d0 = (-B - k) / (2.0*A);
    float d1 = (-B + k) / (2.0*A);
    
    //if (min(B,C) > 0.0) return false;
    
    if (d0 > 1.0) return false;
    // for a conic surface, d1 can be smaller than d0
    if ((d1 <= d0)||(d1 > 1.0))
        d1 = 1.0;
    else if (d1 < 0.0) return false;
    if (d0 > 0.0) {
        near.t = near.t + (far.t - near.t)*d0;
    }
    far.t = near.t + (far.t - near.t)*d1;
    near.b = r0 + brd * near.t;
    far.b = r0 + brd * far.t;
#ifdef HIT_TET_PLANES
    if ((d0 > 0.0) && (d0 < 1.0)) {
        n0 = -eval_quadratic_diff(near.b, c, m1, m2);
    }
    if ((d1 > 0.0) && (d1 < 1.0)) {
        n1 = -eval_quadratic_diff(far.b, c, m1, m2);
    }
#else
    if ((d0 > 0.0) && (d0 < 1.0)) {
        n0 = -eval_quadratic_diff(near.b, c, m1, m2);
    } else if ((d1 > 0.0) && (d1 < 1.0)) {
        n0 = eval_quadratic_diff(far.b, c, m1, m2);
    }  else {
        return false;
    }
#endif
    
#else
#endif
    near.n = normal_from_bary(sn, n0);
    far.n = normal_from_bary(sn, n1);
    return true;
}

// Function 1344
vec4 CostasNoise(ivec2 u,int iFrame){
 ;u/=2
 /*
 ;int[arrLen] m=int[arrLen](0,0,0,0,0,0,0);
 ;m[0]=Cs(u,0);
 ;m[1]=Cs(u,1);
 ;m[2]=Cs(u,2);
 ;m[3]=Cs(u,3);
 ;m[4]=Cs(u,4);
 ;m[5]=Cs(u,5);
 ;m[6]=Cs(u,6);
 ;float a=float(mixedPrimeBase(m))/float(3212440750)//first attempt failed*/
     //VERY exponential weights seem silly mow, but the precison would be neat.
     
 
 ;float r=0.;
    
    
 //;float f[]=float[7](0.,1.,0.,0.,0.,0.,0.)//singleton
    
 //;float f[]=float[7](1.,1.,1.,1.,1.,1.,1.)//flat (strong banding)
 //;float f[]=float[7](4.,3.,2.,1.,2.,3.,4.)//valley
 //;float f[]=float[7](1.,2.,4.,8.,16.,32.,64.)//blue (most banding?)
 //;float f[]=float[7](1.,2.,3.,4.,3.,2.,1.)//windowed 
 ;float f[]=float[7](64.,32.,16.,8.,4.,2.,1.)//anti-blue /least banding)
     //dissappointingly, even small prime tiles as small as 19*19 salready have too stron giagonal banding
     //so i guess, i just need larger tiles and larger primes.
     //i take a bet that it is a bad idea to repeat prime-gaps (espoecially short ones), which may result in the banding
 ;r+=float(Cs(u,0,iFrame))/float(gpo(0))*f[0];
 ;r+=float(Cs(u,1,iFrame))/float(gpo(1))*f[1];
 ;r+=float(Cs(u,2,iFrame))/float(gpo(2))*f[2];
 ;r+=float(Cs(u,3,iFrame))/float(gpo(3))*f[3];
 ;r+=float(Cs(u,4,iFrame))/float(gpo(4))*f[4];
 ;r+=float(Cs(u,5,iFrame))/float(gpo(5))*f[5];
 ;r+=float(Cs(u,6,iFrame))/float(gpo(6))*f[6];
 ;float a=r/(f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6])
 ;return vec4(a,a,a,1);
}

// Function 1345
float noise(vec2 p) { vec2 pm = mod(p,1.0); vec2 pd = p-pm; return hashmix(pd,(pd+vec2(1.0,1.0)), pm); }

// Function 1346
float noise( in vec2 p )
{
  return -1.0+2.0*textureGood( iChannel0, p-0.5 );
}

// Function 1347
float achnoise(vec2 x){
	vec2 p = floor(x);
	vec2 fr = fract(x);
	vec2 LB = p;
	vec2 LT = p + vec2(0.0, 1.0);
	vec2 RB = p + vec2(1.0, 0.0);
	vec2 RT = p + vec2(1.0, 1.0);
	
	float LBo = oct(LB);
	float RBo = oct(RB);
	float LTo = oct(LT);
	float RTo = oct(RT);
	
	float noise1d1 = mix(LBo, RBo, fr.x);
	float noise1d2 = mix(LTo, RTo, fr.x);
	
	float noise2d = mix(noise1d1, noise1d2, fr.y);
	
	return noise2d;
}

// Function 1348
vec4 texNoise(vec2 uv){ float f = 0.; f+=texture(iChannel0, uv*.125).r*.5;
    f+=texture(iChannel0,uv*.25).r*.25;f+=texture(iChannel0,uv*.5).r*.125;
    f+=texture(iChannel0,uv*1.).r*.125;f=pow(f,1.2);return vec4(f*.45+.05);
}

// Function 1349
float smoothVoronoi( in vec2 x )
{
    ivec2 p = ivec2(floor( x ));
    vec2  f = fract( x );

    float res = 0.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        ivec2 b = ivec2( i, j );
        vec2  r = vec2( b ) - f + noise( vec3(p + b, 0.0) );
        float d = length( r );

        res += exp( -32.0*d );
    }
    return -(1.0/32.0)*log( res );
}

// Function 1350
float snoise(vec3 v, out vec3 gradient, float time)
{
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    
// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;
    
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// add rotation
  x0.xy *= rot(time*checkersign(a0.xy));
  x1.xy *= rot(time*checkersign(a0.zw));
  x2.xy *= rot(time*checkersign(a1.xy));
  x3.xy *= rot(time*checkersign(a1.zw));
    
// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  vec4 m2 = m * m;
  vec4 m4 = m2 * m2;
  vec4 pdotx = vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3));

// Determine noise gradient
  vec4 temp = m2 * m * pdotx;
  gradient = -8.0 * (temp.x * x0 + temp.y * x1 + temp.z * x2 + temp.w * x3);
  gradient += m4.x * p0 + m4.y * p1 + m4.z * p2 + m4.w * p3;
  gradient *= 42.0;

  return 42.0 * dot(m4, pdotx);
}

// Function 1351
vec4 Noise( in vec2 x )
{
    vec2 p = floor(x.xy);
    vec2 f = fract(x.xy);
	f = f*f*(3.0-2.0*f);
//	vec3 f2 = f*f; f = f*f2*(10.0-15.0*f+6.0*f2);

	// there's an artefact because the y channel almost, but not exactly, matches the r channel shifted (37,17)
	// this artefact doesn't seem to show up in chrome, so I suspect firefox uses different texture compression.
	vec2 uv = p.xy + f.xy;
	return texture( iChannel0, (uv+0.5)/256.0, -100.0 );
}

// Function 1352
float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    float h1 = rand2(i)*sin(iTime+100000.0*rand2(i));
    float h2 = rand2(i + vec2(1.0, 0.0))*sin(iTime+100000.0*rand2(i + vec2(1.0, 0.0)));
    float h3 = rand2(i + vec2(1.0, 1.0))*sin(iTime+100000.0*rand2(i + vec2(1.0, 1.0)));
    float h4 = rand2(i + vec2(0.0, 1.0))*sin(iTime+100000.0*rand2(i + vec2(0.0, 1.0)));

    vec2 u = smoothstep(0.0, 1.0, f);

    return mix(mix(h1, h2, u.x), mix(h4, h3, u.x), u.y);
}

// Function 1353
float noise(vec2 pos, vec2 scale, float phase, float seed) 
{
    const float kPI2 = 6.2831853071;
    pos *= scale;
    vec4 i = floor(pos).xyxy + vec2(0.0, 1.0).xxyy;
    vec2 f = pos - i.xy;
    i = mod(i, scale.xyxy) + seed;

    vec4 hash = multiHash2D(i);
    hash = 0.5 * sin(phase + kPI2 * hash) + 0.5;
    float a = hash.x;
    float b = hash.y;
    float c = hash.z;
    float d = hash.w;

    vec2 u = noiseInterpolate(f);
    float value = mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
    return value * 2.0 - 1.0;
}

// Function 1354
fractal noise (4 octaves)
    else	
	{
		uv *= 5.0;
        mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
		f  = 0.5000*noise( uv ); uv = m*uv;
		f += 0.2500*noise( uv ); uv = m*uv;
		f += 0.1250*noise( uv ); uv = m*uv;
		f += 0.0625*noise( uv ); uv = m*uv;
	}

// Function 1355
float noise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    float k = 1.0 + 63.0*pow(1.0-v,4.0);
    float va = 0.0;
    float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2  g = vec2( float(i), float(j) );
        vec3  o = hash3( p + g )*vec3(u,u,1.0);
        vec2  r = g - f + o.xy;
        float d = dot(r,r);
        float w = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
        va += w*o.z;
        wt += w;
    }

    return va/wt;
}

// Function 1356
float noise1_2(in vec2 uv)
{
    vec2 f = fract(uv);
    //vec2 f = smoothstep(0.0, 1.0, fract(uv));
    
 	vec2 uv00 = floor(uv);
    vec2 uv01 = uv00 + vec2(0,1);
    vec2 uv10 = uv00 + vec2(1,0);
    vec2 uv11 = uv00 + 1.0;
    
    float v00 = hash1_2(uv00);
    float v01 = hash1_2(uv01);
    float v10 = hash1_2(uv10);
    float v11 = hash1_2(uv11);
    
    float v0 = mix(v00, v01, f.y);
    float v1 = mix(v10, v11, f.y);
    float v = mix(v0, v1, f.x);
    
    return v;
}

// Function 1357
vec4 voronoi2d( in vec2 x )
{
	   
	vec2 n = floor( x );
	vec2 f = fract( x );

	vec3 m = vec3( 8.0 );
	for( int j=-1; j<=1; j++ )
	for( int i=-1; i<=1; i++ )
	{
		vec2  g = vec2( float(i), float(j) );
		vec2  o = hash( n + g );
		vec2  r = g - f + (0.5+0.5*sin(6.2831*o));
		float d = dot( r, r );
		if( d<m.x )
			m = vec3( d, o );
	}

	vec4 c = vec4( sqrt(m.x), m.y+m.z, vec2(m.xy) );
	
	return c;
}

// Function 1358
float triNoise2d(in vec2 p, float spd){
    float z=1.8;
    float z2=2.5;
	float rz = 0.;
    p *= mm2(p.x*0.06);
    vec2 bp = p;
	for (float i=0.; i<5.; i++ )
	{
        vec2 dg = tri2(bp*1.85)*.75;
        dg *= mm2(time*spd);
        p -= dg/z2;

        bp *= 1.3;
        z2 *= .45;
        z *= .42;
		p *= 1.21 + (rz-1.0)*.02;
        
        rz += tri(p.x+tri(p.y))*z;
        p*= -m2;
	}
    return clamp(1./pow(rz*29., 1.3),0.,.55);
}

// Function 1359
float Noisefv2 (vec2 p)
{
  vec2 i, f;
  i = floor (p);  f = fract (p);
  f = f * f * (3. - 2. * f);
  vec4 t = Hashv4f (dot (i, cHashA3.xy));
  return mix (mix (t.x, t.y, f.x), mix (t.z, t.w, f.x), f.y);
}

// Function 1360
float noise( float a, float b )
{
    return snoise(vec2(a, b));
}

// Function 1361
float NoiseNoDer(vec3 iuw,vec3 ivw,float rot,vec2 a,vec2 b,vec2 c
){vec3 w=vec3(dot(rgrad2(vec2(iuw.x,ivw.x),rot),a)
             ,dot(rgrad2(vec2(iuw.y,ivw.y),rot),b)
             ,dot(rgrad2(vec2(iuw.z,ivw.z),rot),c))     
 ;vec3 t=.8-vec3(dd(a),dd(b),dd(c));t*=t
 ;return 11.*dot(t*t,w);}

// Function 1362
float snoise(in lowp vec2 v) {
  lowp vec2 i = floor((v.x+v.y)*.36602540378443 + v),
      x0 = (i.x+i.y)*.211324865405187 + v - i;
  lowp float s = step(x0.x,x0.y);
  lowp vec2 j = vec2(1.0-s,s),
      x1 = x0 - j + .211324865405187, 
      x3 = x0 - .577350269189626; 
  i = mod(i,289.);
  lowp vec3 p = permute( permute( i.y + vec3(0, j.y, 1 ))+ i.x + vec3(0, j.x, 1 )   ),
       m = max( .5 - vec3(dot(x0,x0), dot(x1,x1), dot(x3,x3)), 0.),
       x = fract(p * .024390243902439) * 2. - 1.,
       h = abs(x) - .5,
      a0 = x - floor(x + .5);
  return -0.278 + .5 + 65. * dot( pow(m,vec3(4.))*(- 0.85373472095314*( a0*a0 + h*h )+1.79284291400159 ), a0 * vec3(x0.x,x1.x,x3.x) + h * vec3(x0.y,x1.y,x3.y));
}

// Function 1363
float noisefloor(vec2 uv,float t)
{
    float n = texNoise(uv*.1).r*4.-1.;
    return texNoise(vec2(n+iTime*t)).r;
}

// Function 1364
float noise(vec2 pos){
    return texture(iChannel2, pos).r;
}

// Function 1365
float blugausnoise(vec2 c1) {
    c1 += 0.07* fract(iiTime);
    //vec2 c0 = vec2(c1.x- 1.,c1.y);
    //vec2 c2 = vec2(c1.x+ 1.,c1.y);
    vec3 cx = c1.x+ vec3(-1,0,1);
    vec4 f0 = fract(vec4(cx* 9.1031,c1.y* 8.1030));
    vec4 f1 = fract(vec4(cx* 7.0973,c1.y* 6.0970));
	vec4 t0 = vec4(f0.xw,f1.xw);//fract(c0.xyxy* vec4(.1031,.1030,.0973,.0970));
	vec4 t1 = vec4(f0.yw,f1.yw);//fract(c1.xyxy* vec4(.1031,.1030,.0973,.0970));
	vec4 t2 = vec4(f0.zw,f1.zw);//fract(c2.xyxy* vec4(.1031,.1030,.0973,.0970));
    vec4 p0 = t0+ dot(t0,t0.wzxy+ 19.19);
    vec4 p1 = t1+ dot(t1,t1.wzxy+ 19.19);
    vec4 p2 = t2+ dot(t2,t2.wzxy+ 19.19);
	vec4 n0 = fract(p0.zywx* (p0.xxyz+ p0.yzzw));
	vec4 n1 = fract(p1.zywx* (p1.xxyz+ p1.yzzw));
	vec4 n2 = fract(p2.zywx* (p2.xxyz+ p2.yzzw));
    return dot(0.5* n1- 0.125* (n0+ n2),vec4(1));
}

// Function 1366
float noise(vec3 p) {
#ifdef Use_Perlin
    return perlin_noise(p * 2.0);
#elif defined Use_Value
    return value_noise(p * 2.0);
#elif defined Use_Simplex
    return simplex_noise(p);
#endif
    
    return 0.0;
}

// Function 1367
vec4 noised_improveXYPlanes(in vec3 x)
{
    mat3 orthonormalMap = mat3(
        0.788675134594813, -0.211324865405187, -0.577350269189626,
        -0.211324865405187, 0.788675134594813, -0.577350269189626,
        0.577350269189626, 0.577350269189626, 0.577350269189626);
    x = x * orthonormalMap;
    
    vec4 result = noised(x);
    result.yzw = orthonormalMap * result.yzw;
    return result;
}

// Function 1368
vec3 noised( in vec2 x ){
    x-= 15.;
    vec2 p = floor(x);
    vec2 f = fract(x);
    vec2 u = f*f*(3.0-2.0*f);
	float a = rand((p+vec2(0.5,0.5))/256.0);
	float b = rand((p+vec2(1.5,0.5))/256.0);
	float c = rand((p+vec2(0.5,1.5))/256.0);
	float d = rand((p+vec2(1.5,1.5))/256.0);
	return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
				6.0*f*(1.0-f)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));
}

// Function 1369
float smoothNoise(vec2 uv)
{
    vec2 lv = fract(uv);
    vec2 id = floor(uv);
    
    lv = lv * lv * (3. - 2.*lv);
    
    float bl = hash12(id);
    float br = hash12(id + vec2(1., 0.));
    
    float b = mix(bl, br, lv.x);
    
    
    float tl = hash12(id + vec2(0., 1.));
    float tr = hash12(id + vec2(1., 1.));
    
    float t = mix(tl, tr, lv.x);
    
    float c = mix(b,t, lv.y);
    return c;
}

// Function 1370
float voronoiClassic(in vec2 st){
    // Tile the space
    vec2 i_st = floor(st);
    vec2 f_st = fract(st);

    float m_dist = 1.0;  // minimun distance
    vec2 m_point ;
    
    for (int j=-1; j<=1; j++ ) {
        for (int i=-1; i<=1; i++ ) {
            vec2 neighbor = vec2(float(i),float(j));
            vec2 point = random2(i_st + neighbor);
            
             point = 0.5 + 0.5*sin(time + 0.4 *sin(time*0.1) * point);
             
          
            vec2 diff = neighbor + point - f_st;
            float dist = length(diff);
            
           
             m_dist = (min(m_dist, dist) + min(m_dist, m_dist*dist)) /2.;
            
        }
    }
    
    

    return m_dist ;
   
}

// Function 1371
float snoise(vec2 v){
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                     + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
                            dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Function 1372
float Noise( in vec2 p ) 
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

    vec2  i = floor( p + (p.x+p.y)*K1 );
    vec2  a = p - i + (i.x+i.y)*K2;
    float m = step(a.y,a.x); 
    vec2  o = vec2(m,1.0-m);
    vec2  b = a - o + K2;
    vec2  c = a - 1.0 + 2.0*K2;
    vec3  h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
    vec3  n = h*h*h*h*vec3( dot(a,Hash2(i+0.0)), dot(b,Hash2(i+o)), dot(c,Hash2(i+1.0)));
    return dot( n, vec3(70.0) );
}

// Function 1373
vec4 backgroundNoise(in vec2 uv)
{
    vec4 bottomColor = vec4(0.8, 0.2, 1.0, 1.0); 
    vec4 topColor = vec4(1.0, 0.0, 0.5, 1.0);
    float intensity = 0.25;
    
    uv = uv*3.0;
    uv.x += iTime*13.0;
    uv.y += iTime*7.0;
	float noiseSample = texture(iChannel0, uv).r;
    
    vec4 result = bottomColor + noiseSample * topColor;
    result.rgb *= intensity;
 	return result;   
}

// Function 1374
float cnoise(vec3 P)
{
  vec3 Pi0 = floor(P); // Integer part for indexing
  vec3 Pi1 = Pi0 + vec3(1, 1, 1); // Integer part + 1
  Pi0 = mod289(Pi0);
  Pi1 = mod289(Pi1);
  vec3 Pf0 = fract(P); // Fractional part for interpolation
  vec3 Pf1 = Pf0 - vec3(1, 1, 1); // Fractional part - 1.0
  vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  vec4 iy = vec4(Pi0.yy, Pi1.yy);
  vec4 iz0 = Pi0.zzzz;
  vec4 iz1 = Pi1.zzzz;

  vec4 ixy = permute(permute(ix) + iy);
  vec4 ixy0 = permute(ixy + iz0);
  vec4 ixy1 = permute(ixy + iz1);

  vec4 gx0 = ixy0 * (1.0 / 7.0);
  vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  vec4 gz0 = vec4(.5, .5, .5, .5) - abs(gx0) - abs(gy0);
  vec4 sz0 = step(gz0, vec4(0, 0, 0, 0));
  gx0 -= sz0 * (step(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step(0.0, gy0) - 0.5);

  vec4 gx1 = ixy1 * (1.0 / 7.0);
  vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  vec4 gz1 = vec4(.5, .5, .5, .5) - abs(gx1) - abs(gy1);
  vec4 sz1 = step(gz1, vec4(0, 0, 0, 0));
  gx1 -= sz1 * (step(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step(0.0, gy1) - 0.5);

  vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
  vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
  vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
  vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
  vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
  vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
  vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
  vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

  vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
  float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
  float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
  float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
  float n111 = dot(g111, Pf1);

  vec3 fade_xyz = fade(Pf0);
  vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
  vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x); 
  return 2.2 * n_xyz;
}

// Function 1375
float noise( in vec2 p ) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f*f*(3.-2.*f);
    return mix( mix( dot( hash( i + vec2(0.,0.) ), f - vec2(0.,0.) ), 
                     dot( hash( i + vec2(1.,0.) ), f - vec2(1.,0.) ), u.x),
                mix( dot( hash( i + vec2(0.,1.) ), f - vec2(0.,1.) ), 
                     dot( hash( i + vec2(1.,1.) ), f - vec2(1.,1.) ), u.x), u.y);
}

// Function 1376
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 w = fract(x);
    vec2 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    
#if 0
    p *= 0.3183099;
    float kx0 = 50.0*fract( p.x );
    float kx1 = 50.0*fract( p.x+0.3183099 );
    float ky0 = 50.0*fract( p.y );
    float ky1 = 50.0*fract( p.y+0.3183099 );

    float a = fract( kx0*ky0*(kx0+ky0) );
    float b = fract( kx1*ky0*(kx1+ky0) );
    float c = fract( kx0*ky1*(kx0+ky1) );
    float d = fract( kx1*ky1*(kx1+ky1) );
#else
    float a = hash1(p+vec2(0,0));
    float b = hash1(p+vec2(1,0));
    float c = hash1(p+vec2(0,1));
    float d = hash1(p+vec2(1,1));
#endif
    
    return -1.0+2.0*( a + (b-a)*u.x + (c-a)*u.y + (a - b - c + d)*u.x*u.y );
}

// Function 1377
float snoise3DS(vec3 texc)
{
    return snoise3Dv4S(texc).x;
}

// Function 1378
float noise_sum_abs(vec2 p)
{
    float f = 0.0;
    p = p * 7.0;
    f += 1.0000 * abs(noise(p)); p = 2.0 * p;
    f += 0.5000 * abs(noise(p)); p = 2.0 * p;
	f += 0.2500 * abs(noise(p)); p = 2.0 * p;
	f += 0.1250 * abs(noise(p)); p = 2.0 * p;
	f += 0.0625 * abs(noise(p)); p = 2.0 * p;
    
    return f;
}

// Function 1379
float snoise( vec2 p ) {
	vec2 f = fract(p);
	p = floor(p);
	float v = p.x+p.y*1000.0;
	vec4 r = vec4(v, v+1.0, v+1000.0, v+1001.0);
	r = fract(100000.0*sin(r*.001));
	f = f*f*(3.0-2.0*f);
	return 2.0*(mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y))-1.0;
}

// Function 1380
vec4 voronoi(in vec2 x){
    x.x += .6;
    vec2 n = floor(x);
    vec2 f = fract(x);
	
    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr, closest, cellId;
	
    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash22(n + g);
		vec2 r = g + o - f;
        float d = dot(r,r);

        if(d<md){
            md = d;
            mr = r;
            mg = g;
            closest = n + g + o;
        }
    }
    
    if(abs(n.y + mg.y) >= 3. || length(n.x + mg.x) >= 6.)
        return vec4(0.);
    
    bool notRock = abs(n.y + mg.y) == 2.
                || length(n.x + mg.x) >= 5.;
    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 4.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash22( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>0.00001 )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return vec4( md, closest, notRock? .5: 1.);
	
}

// Function 1381
float perlinNoise(vec2 uv, float frequency, int octaves, float lacunarity, float persistence)
{
    float amplitude = 1.;
    float pixelValue = 0.;
    float maxValue = 0.;	//used to normalize the final value
    
    for (int octave = 0; octave < octaves; octave ++)
    {
        //Get Pixel's Position Within the Cell && Cell's Position Within the Grid
        vec2 pixelPosition = fract(uv * frequency);
        vec2 cellPosition = floor(uv * frequency);

        //Get Gradient Vectors of the Cell's Points
        vec2 gradientVector1 = randomLimitedVector(cellPosition);
        vec2 gradientVector2 = randomLimitedVector(vec2(cellPosition.x + 1., cellPosition.y));
        vec2 gradientVector3 = randomLimitedVector(vec2(cellPosition.x, cellPosition.y + 1.));
        vec2 gradientVector4 = randomLimitedVector(vec2(cellPosition.x + 1., cellPosition.y + 1.));

        //Calculate Distance Vectors from the Cell's Points to the Pixel
        vec2 distanceVector1 = vec2(pixelPosition.x, - pixelPosition.y);
        vec2 distanceVector2 = vec2(- (1. - pixelPosition.x), - pixelPosition.y);
        vec2 distanceVector3 = vec2(pixelPosition.x, 1. - pixelPosition.y);
        vec2 distanceVector4 = vec2(- (1. - pixelPosition.x), 1. - pixelPosition.y);

        //Calculate Dot Product of the Gradient && Distance Vectors
        float dotProduct1 = dot(gradientVector1, distanceVector1);
        float dotProduct2 = dot(gradientVector2, distanceVector2);
        float dotProduct3 = dot(gradientVector3, distanceVector3);
        float dotProduct4 = dot(gradientVector4, distanceVector4);

        //Apply Smootherstep Function on the Pixel Position for Interpolation
        vec2 pixelPositionSmoothed = vec2(smootherstep(pixelPosition.x), smootherstep(pixelPosition.y));

        //Interpolate Between the Dot Products
        float interpolation1 = mix(dotProduct1, dotProduct2, pixelPositionSmoothed.x);
        float interpolation2 = mix(dotProduct3, dotProduct4, pixelPositionSmoothed.x);
        float interpolation3 = mix(interpolation1, interpolation2, pixelPositionSmoothed.y);
        
        pixelValue += (interpolation3 * 0.5 + 0.5) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return pixelValue / maxValue;
}

// Function 1382
vec1 noise1(vec3 p){vec3 f=floor(p);p=herm32(fract(p));return bilin(mix(hash4(f),hash4(f+vec3(0,0,1)),p.z),p.xy);}

// Function 1383
vec2 noise(float time)
{
    return vec2(clamp(fract(sin(time*8.3)*1e8)-.4, 0.0, 0.2)); 
}

// Function 1384
float noise(in vec3 p) {
    const vec3 step = vec3(110.0, 241.0, 171.0);

    vec3 i = floor(p);
    vec3 f = fract(p);

    // For performance, compute the base input to a
    // 1D random from the integer part of the
    // argument and the incremental change to the
    // 1D based on the 3D -> 1D wrapping
    float n = dot(i, step);

    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix( mix(mix(random(n + dot(step, vec3(0,0,0))),
                        random(n + dot(step, vec3(1,0,0))),
                        u.x),
                    mix(random(n + dot(step, vec3(0,1,0))),
                        random(n + dot(step, vec3(1,1,0))),
                        u.x),
                u.y),
                mix(mix(random(n + dot(step, vec3(0,0,1))),
                        random(n + dot(step, vec3(1,0,1))),
                        u.x),
                    mix(random(n + dot(step, vec3(0,1,1))),
                        random(n + dot(step, vec3(1,1,1))),
                        u.x),
                u.y),
            u.z);
}

// Function 1385
float worleyNoise(vec3 uv, float freq)
{    
    vec3 id = floor(uv);
    vec3 p = fract(uv);
    
    float minDist = 10000.;
    for (float x = -1.; x <= 1.; ++x)
    {
        for(float y = -1.; y <= 1.; ++y)
        {
            for(float z = -1.; z <= 1.; ++z)
            {
                vec3 offset = vec3(x, y, z);
            	vec3 h = hash33(mod(id + offset, vec3(freq))) * .5 + .5;
    			h += offset;
            	vec3 d = p - h;
           		minDist = min(minDist, dot(d, d));
            }
        }
    }
    
    // inverted worley noise
    return 1. - minDist;
}

// Function 1386
float smoothnoise(const in float o) 
{
	float p = floor(o);
	float f = fract(o);
		
	float n = p;

	float a = hash(n+  0.0);
	float b = hash(n+  1.0);
	
	float f2 = f * f;
	float f3 = f2 * f;
	
	float t = 3.0 * f2 - 2.0 * f3;
	
	return mix(a, b, t);
}

// Function 1387
vec3 bluenoise(vec2 coord) {
    return texture(iChannel1, coord / 1024.0f).xyz;
}

// Function 1388
float valueNoiseStepped(float i, float p, float steps){ return mix(  floor(r11(floor(i))*steps)/steps, floor(r11(floor(i) + 1.)*steps)/steps, ss(fract(i), p,0.6));}

// Function 1389
float Noise( in vec3 x )
{
    #if 0
    return texture(iChannel2, x*0.05).x;
    #else
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
    #endif
}

// Function 1390
vec2 noise2(vec2 x)
{
  return vec2(noise(x.x), noise(x.y));
}

// Function 1391
float smoothNoise2(vec2 p)
{
    p=fract(p/256.0);
    return textureLod(iChannel0, p, 0.0).r;
}

// Function 1392
float remappedNoise(in vec3 p){
	return .5 + .5 * (noise(p)/.6);
}

// Function 1393
float noise (vec3 n)
{
    vec3 base = floor(n * 64.0) * 0.015625;
    vec3 dd = vec3(0.015625, 0.0, 0.0);
    float a = hash(base);
    float b = hash(base + dd.xyy);
    float c = hash(base + dd.yxy);
    float d = hash(base + dd.xxy);
    vec3 p = (n - base) * 64.0;
    float t = mix(a, b, p.x);
    float tt = mix(c, d, p.x);
    return mix(t, tt, p.y);
}

// Function 1394
float gnoise( in float p )
{
    int   i = int(floor(p));
    float f = fract(p);
	float u = f*f*(3.0-2.0*f);
    return mix( hash(i+0)*(f-0.0), 
                hash(i+1)*(f-1.0), u);
}

// Function 1395
float Noise(vec4 formants)
{
    float result = BandNoiseOpti(formants.x,gTime);
    result += BandNoiseOpti(formants.y,gTime);
    result += BandNoiseOpti(formants.z,gTime);
    result += BandNoiseOpti(formants.w,gTime);
    return result * 0.01;
}

// Function 1396
vec2 getNoiseColor(vec3 dir, int type){
    vec2 res;
    float a = 2. * PI * pow(smoothstep(0., .75, bap.phase), .5);
    dir.xz *= mat2(cos(a), -sin(a), sin(a), cos(a));
    vec4 noises = cubeMap(dir);
    res.x = noises[type];
    if(type == AP_VORONOI_NOISE){
        float phase = smoothstep(.25, .5, bap.phase);
        float craterSize = mix(.01, .1, noises.a * .001) + (1.-phase);
        res.x = mix(res.x, smoothstep(.025 * craterSize * 4., .075 * craterSize * 4., distance(res.x, craterSize)), phase);
    }
    res.y = smoothstep(0., .25, bap.phase) * smoothstep(1., .5, bap.phase);
    return res;
}

// Function 1397
float snoise(vec3 v)
{
    return length(texture(iChannel0, v.xy * 0.1));
}

// Function 1398
float twistedSineNoise(vec4 q)
{
    float a = 1.;
    float sum = 0.;
    for(int i = 0; i <4 ; i++){
        q = m4 * q;
        vec4 s = sin(q.ywxz / a) * a;
        q += s;
        sum += s.x;
        a *= 0.7;
    }
    return sum;
}

// Function 1399
vec2 noise2(in vec2 p) {
	vec2 F = floor(p), f = fract(p);
	f = f * f * (3. - 2. * f);
	return mix(
		mix(hash2(F), 			  hash2(F+vec2(1.,0.)), f.x),
		mix(hash2(F+vec2(0.,1.)), hash2(F+vec2(1.)),	f.x), f.y);
}

// Function 1400
float snoise(in mediump vec3 v){
  const lowp vec2 C = vec2(0.16666666666,0.33333333333);
  const lowp vec4 D = vec4(0,.5,1,2);
  lowp vec3 i  = floor(C.y*(v.x+v.y+v.z) + v);
  lowp vec3 x0 = C.x*(i.x+i.y+i.z) + (v - i);
  lowp vec3 g = step(x0.yzx, x0);
  lowp vec3 l = (1. - g).zxy;
  lowp vec3 i1 = min( g, l );
  lowp vec3 i2 = max( g, l );
  lowp vec3 x1 = x0 - i1 + C.x;
  lowp vec3 x2 = x0 - i2 + C.y;
  lowp vec3 x3 = x0 - D.yyy;
  i = mod(i,289.);
  lowp vec4 p = permute( permute( permute(
	  i.z + vec4(0., i1.z, i2.z, 1.))
	+ i.y + vec4(0., i1.y, i2.y, 1.))
	+ i.x + vec4(0., i1.x, i2.x, 1.));
  lowp vec3 ns = .142857142857 * D.wyz - D.xzx;
  lowp vec4 j = -49. * floor(p * ns.z * ns.z) + p;
  lowp vec4 x_ = floor(j * ns.z);
  lowp vec4 x = x_ * ns.x + ns.yyyy;
  lowp vec4 y = floor(j - 7. * x_ ) * ns.x + ns.yyyy;
  lowp vec4 h = 1. - abs(x) - abs(y);
  lowp vec4 b0 = vec4( x.xy, y.xy );
  lowp vec4 b1 = vec4( x.zw, y.zw );
  lowp vec4 sh = -step(h, vec4(0));
  lowp vec4 a0 = b0.xzyw + (floor(b0)*2.+ 1.).xzyw*sh.xxyy;
  lowp vec4 a1 = b1.xzyw + (floor(b1)*2.+ 1.).xzyw*sh.zzww;
  lowp vec3 p0 = vec3(a0.xy,h.x);
  lowp vec3 p1 = vec3(a0.zw,h.y);
  lowp vec3 p2 = vec3(a1.xy,h.z);
  lowp vec3 p3 = vec3(a1.zw,h.w);
  lowp vec4 norm = inversesqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  lowp vec4 m = max(.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.);
  return .5 + 12. * dot( m * m * m, vec4( dot(p0,x0), dot(p1,x1),dot(p2,x2), dot(p3,x3) ) );
}

// Function 1401
float noise(vec3 x)
{
    //x.x = mod(x.x, 0.4);
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 1402
float noise (vec2 p) {
	return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Function 1403
float noise( in vec3 p )
{
    vec3 f = fract(p);
    p = floor(p);
	f = f*f*(3.0-2.0*f);
	 
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel1, (uv+ 0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 1404
float simplex(vec3 v)
  {
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
  }

// Function 1405
float noise(in vec2 p) {
	vec2 e=vec2(1.,0.), F = floor(p), f = fract(p), k = (3. - 2.*f) * f * f;
	return mix(mix(hash2(F),      hash2(F+e.xy), k.x),
			   mix(hash2(F+e.yx), hash2(F+e.xx), k.x), k.y);
}

// Function 1406
float noise(float x) {
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3.0 - 2.0 * f);
    return mix(hash(i), hash(i + 1.0), u);
}

// Function 1407
float fractalblobnoise(vec2 v, float s)
{
    float val = 0.;
    const float n = 4.;
    for(float i = 0.; i < n; i++)
        //val += 1.0 / (i + 1.0) * blobnoise((i + 1.0) * v + vec2(0.0, iTime * 1.0), s);
    	val += pow(0.5, i+1.) * blobnoise(exp2(i) * v + vec2(0, T), s);

    return val;
}

// Function 1408
float InterpolatedNoise(float x, float y, float z) {
	int integer_X = int(floor(x));
	float fractional_X = fract(x);
	int integer_Y = int(floor(y));
	float fractional_Y = fract(y);
    int integer_Z = int(floor(z));
    float fractional_Z = fract(z);
    
	vec3 randomInput = vec3(integer_X, integer_Y, integer_Z);
	float v1 = Random2D(randomInput + vec3(0.0, 0.0, 0.0));
	float v2 = Random2D(randomInput + vec3(1.0, 0.0, 0.0));
	float v3 = Random2D(randomInput + vec3(0.0, 1.0, 0.0));
	float v4 = Random2D(randomInput + vec3(1.0, 1.0, 0.0));
    
    float v5 = Random2D(randomInput + vec3(0.0, 0.0, 1.0));
	float v6 = Random2D(randomInput + vec3(1.0, 0.0, 1.0));
	float v7 = Random2D(randomInput + vec3(0.0, 1.0, 1.0));
	float v8 = Random2D(randomInput + vec3(1.0, 1.0, 1.0));
    
    
	float i1 = Interpolate(v1, v2, fractional_X);
	float i2 = Interpolate(v3, v4,  fractional_X);
    
    float i3 = Interpolate(v5, v6, fractional_X);
    float i4 = Interpolate(v7, v8, fractional_X);
    
    float y1 = Interpolate(i1, i2, fractional_Y);
    float y2 = Interpolate(i3, i4, fractional_Y);
    
    
	return Interpolate(y1, y2, fractional_Z);
}

// Function 1409
vec3 TerrainNoiseLQ(vec2 p) { return TerrainNoise(p, 3); }

// Function 1410
vec3 curlNoise(vec3 p)
{
    const float e = .1;
  vec3 dx = vec3( e   , 0.0 , 0.0 );
  vec3 dy = vec3( 0.0 , e   , 0.0 );
  vec3 dz = vec3( 0.0 , 0.0 , e   );

  vec3 p_x0 = snoiseVec3( p - dx );
  vec3 p_x1 = snoiseVec3( p + dx );
  vec3 p_y0 = snoiseVec3( p - dy );
  vec3 p_y1 = snoiseVec3( p + dy );
  vec3 p_z0 = snoiseVec3( p - dz );
  vec3 p_z1 = snoiseVec3( p + dz );

  float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
  float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
  float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

  //const float divisor = 1.0 / ( 2.0 * e );
  //return normalize( vec3( x , y , z ) * divisor );
  // technically incorrect but I like this better...
  return vec3( x , y , z );
}

// Function 1411
float noise2d(vec2 co) {
  return fract(sin(dot(co.xy ,vec2(1.0,73))) * 43758.5453);
}

// Function 1412
float noise3d(in vec3 p, in int si)
{
    float z=1.4;
	float rz = 0.;
    vec3 bp = p;
	for (int i=0; i<= si; i++ )
	{
        vec3 dg = tri3(bp);
        p += (dg);

        bp *= 1.8;
		z *= 1.4;
		p *= 1.3;
        
        rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
        bp += 0.2;
	}
	return rz;
}

// Function 1413
vec4 noise(int p){
 	return noise(ivec3(p, 0, 0));   
}

// Function 1414
vec3 perlin31(float p, float n)
{
    float frq = 1., amp = 1., norm = 0.;
    vec3 res = vec3(0.);
    for(float i = 0.; i < n; i++)
    {
        res += amp*perlin31(frq*p);
        norm += amp;
        frq *= 1.1;
        amp *= 0.95;
    }
    return res/norm;
}

// Function 1415
vec4 voronoi (vec2 p, float roundness) {
  vec2 temp;
  return voronoi(p,roundness,temp);
}

// Function 1416
float perlin(vec2 uv)
{
    vec2 relco = fract(uv);
    vec2 inco = floor(uv);
    
    vec2 grad1 = grad(inco);
    vec2 grad2 = grad(inco+vec2(1,0));
    vec2 grad3 = grad(inco+vec2(1,1));
    vec2 grad4 = grad(inco+vec2(0,1));
    
    float s = dot(grad1,relco);
    float t = dot(grad2, relco-vec2(1,0));
    float u = dot(grad3, relco-1.);
    float v = dot(grad4, relco-vec2(0,1));
    
    float n1 = mix(s,t,smoothstep(0.,1.,relco.x));
    float n2 = mix(v,u,smoothstep(0.,1.,relco.x));

    return mix(n1,n2,smoothstep(0.,1.,relco.y));
}

// Function 1417
float RustleNoise(float t, float iter) {
    float s;
    for (float i = 0.0; i < iter; i++)
    {
		s += Hash(iSampleRate * (t + i)) * 2.0 - 1.0;
    }
    return s *= 1.0 / iter;
}

// Function 1418
float fractalnoise(vec2 uv, float mag) {
    float d = valuenoise(uv);
    int i;
    float fac = 1.;
    vec2 disp = vec2(0., 1.);
    for (i=0; i<3; i++) {
        uv += mag * iTime * disp * fac;
        disp = mat2(.866, 0.5, -0.5, .866) * disp; //rotate each moving layer
        fac *= 0.5;
        d += valuenoise(uv/fac)*fac;
    }
    return d;
}

// Function 1419
float noise(vec2 p )
{
    return fract(sin(fract(sin(p.x)*(41.13311))+ p.y)*31.0011);
}

// Function 1420
float perlinNoise(vec2 uv) {
    vec2 PT  = floor(uv);
    vec2 pt  = fract(uv);
    vec2 mmpt= ssmooth(pt);

    vec4 grads = vec4(
        dot(hash(PT + vec2(.0, 1.)), pt-vec2(.0, 1.)),   dot(hash(PT + vec2(1., 1.)), pt-vec2(1., 1.)),
        dot(hash(PT + vec2(.0, .0)), pt-vec2(.0, .0)),   dot(hash(PT + vec2(1., .0)), pt-vec2(1., 0.))
    );

    return 5.*mix (mix (grads.z, grads.w, mmpt.x), mix (grads.x, grads.y, mmpt.x), mmpt.y);
}

// Function 1421
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
    vec2 u = smoothstep(vec2(0), vec2(1), f); // similar to u = f*f*(3.0-2.0*f);

    return mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1422
float noise(in vec3 p)
{
	vec3 ip = floor(p), fp = fract(p);
    fp = fp*fp*(3.0 - 2.0*fp); //Cubic smoothing
	vec2 tap = (ip.xy+vec2(37.0,17.0)*ip.z) + fp.xy;
	vec2 rz = textureLod( iChannel0, (tap + 0.5)/256.0, 0.0 ).yx;
	return mix(rz.x, rz.y, fp.z);
}

// Function 1423
vec4 texNoise(vec2 uv)
{
    float f = 0.;
    f += texture(iChannel0, uv*.125).r*.5;
    f += texture(iChannel0, uv*.25).r*.25;
    f += texture(iChannel0, uv*.5).r*.125;
    f += texture(iChannel0, uv*1.).r*.125;
    f=pow(f,1.2);
    return vec4(f*.45+.05);
}

// Function 1424
vec4 CostasNoise(ivec2 u,int iFrame,vec4 m){
 ;u/=2

 ;float r=0.;
    
 //last 3 entries are better set to 0, they are very small and too symmetrical costas arrays (cause diagonal bands)
 //;float f[]=float[7](1.,0.,0.,0.,0.,0.,0.)//singleton    
 ;float f[]=float[40](
  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.)
     
 //;float f[]=float[arrLen](1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.)
     
    // float[7](1.,1.,1.,1.,1.,1.,1.)//flat (strong banding)
 //;float f[]=float[7](4.,3.,2.,1.,0.,0.,0.)//valley
;float blue[]=f//float[7](1.,2.,4.,8.,16.,32.,0.)//blue (most banding?)
 //;float f[]=float[7](1.,2.,2.,1.,0.,0.,0.)//windowed 
 ;float yellow[]=f//float[7](64.,32.,16.,8.,4.,2.,0.)//anti-blue /least banding)
     //dissappointingly, even small prime tiles as small as 19*19 salready have too stron giagonal banding
     //so i guess, i just need larger tiles and larger primes.
     //i take a bet that it is a bad idea to repeat prime-gaps (espoecially short ones), which may result in the banding

 ;m=clamp(m,0.,1.)
 
 ;for(int i=0;i<40;i++){
      blue[i]=mix(blue[i],yellow[i],m.x);
 }    
;for(int i=0;i<40;i++){
      f[i]=mix(blue[i],f[i],m.y);//mix to flat
 }
    
    ;float s=0.;
    ;int ass=8
  // for(int i=0;i<2;i++ 
   ;for(int i=0;i<40+1;i++

       ){  
       
       if(
           i==7
           //i==20
          // i==21
       //    i<8
       ){
       s+=f[i]
        ;r+=float(Cs(u,i,iFrame))/float(gpo(i))*f[i];
           ;}}
    
 //;r+=float(Cs(u,3,iFrame))/float(gpo(3))*f[0];
 //;r+=float(Cs(u,2,iFrame))/float(gpo(2))*f[2];
// ;u=u.yx//addition, to make half of the arrays diagonally flipped
 //;r+=float(Cs(u,1,iFrame))/float(gpo(1))*f[1];
 //;r+=float(Cs(u,3,iFrame))/float(gpo(3))*f[3];
 //large above, small below
 //;r+=float(Cs(u,4,iFrame))/float(gpo(4))*f[4];    
 //;u=u.yx//addition, to make half of the arrays diagonally flipped
 //;r+=float(Cs(u,5,iFrame))/float(gpo(5))*f[5];
 //;r+=float(Cs(u,6,iFrame))/float(gpo(6))*f[6]; 
     
    
 ;float a=r/s//(f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6])//divide by sum of weights
 ;return vec4(a,a,a,1);
}

// Function 1425
float noise(vec2 v) {
    // stretched texture noise
	return texture(iChannel0, v/vec2(128.0, 32.0)).x;
}

// Function 1426
float noise( in vec3 p )
{
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = interp(f);

    return mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// Function 1427
float noise( vec2 p )
{
	return sin(p.x)*sin(p.y);
}

// Function 1428
float noise_3(in vec3 p) {
    vec3 i = floor( p );
    vec3 f = fract( p );	
	vec3 u = f*f*(3.0-2.0*f);
    
    float a = hash3( i + vec3(0.0,0.0,0.0) );
	float b = hash3( i + vec3(1.0,0.0,0.0) );    
    float c = hash3( i + vec3(0.0,1.0,0.0) );
	float d = hash3( i + vec3(1.0,1.0,0.0) ); 
    float v1 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
    
    a = hash3( i + vec3(0.0,0.0,1.0) );
	b = hash3( i + vec3(1.0,0.0,1.0) );    
    c = hash3( i + vec3(0.0,1.0,1.0) );
	d = hash3( i + vec3(1.0,1.0,1.0) );
    float v2 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
        
    return abs(mix(v1,v2,u.z));
}

// Function 1429
float cnoise(vec2 p) {
	vec4 pi = floor(p.xyxy)+ vec4(0.0, 0.0, 1.0, 1.0);
	vec4 pf = fract(p.xyxy)- vec4(0.0, 0.0, 1.0, 1.0);
	pi = mod289(pi); // To avoid truncation effects in permutation
	vec4 ix = pi.xzxz;
	vec4 iy = pi.yyww;
	vec4 fx = pf.xzxz;
	vec4 fy = pf.yyww;

	vec4 i = permute(permute(ix)+ iy);

	vec4 gx = fract(i* (1.0/ 41.0))* 2.0- 1.0;
	vec4 gy = abs(gx)- 0.5;
	vec4 tx = floor(gx+ 0.5);
	gx = gx- tx;

	vec2 g00 = vec2(gx.x, gy.x);
	vec2 g10 = vec2(gx.y, gy.y);
	vec2 g01 = vec2(gx.z, gy.z);
	vec2 g11 = vec2(gx.w, gy.w);

	vec4 norm = taylor_inv_sqrt(
			vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
	g00 *= norm.x;
	g01 *= norm.y;
	g10 *= norm.z;
	g11 *= norm.w;

	float n00 = dot(g00, vec2(fx.x, fy.x));
	float n10 = dot(g10, vec2(fx.y, fy.y));
	float n01 = dot(g01, vec2(fx.z, fy.z));
	float n11 = dot(g11, vec2(fx.w, fy.w));

	vec2 fade_xy = smootherstep(pf.xy);
	vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
	float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
	return 2.3* n_xy;
}

// Function 1430
float noise3(in vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f * f * (3.0 - 2.0 * f);
	vec2 uv = (p.xy + vec2(37.0, 17.0+smoothstep(0.8,0.99, fft(0.))) * p.z) + f.xy;
	vec2 rg = texture(iChannel0, (uv + 0.5) / 256.0, -100.0).yx;
	rg += texture(iChannel0, (uv + iTime) / 64.0, -100.0).yx/10.0;
	rg += texture(iChannel0, (uv + iTime/3.2 + 0.5) / 100.0, -100.0).zx/5.0;
	return mix(rg.x, rg.y, f.z);
}

// Function 1431
float noise(vec2 p) {
    return random(p.x + p.y*10000.0);
}

// Function 1432
vec3 NoiseDer(vec3 iuw,vec3 ivw,float rot,vec2 a,vec2 b,vec2 c
){vec2 d=rgrad2(vec2(iuw.x,ivw.x),rot)
 ;vec2 e=rgrad2(vec2(iuw.y,ivw.y),rot)
 ;vec2 f=rgrad2(vec2(iuw.z,ivw.z),rot)
 ;vec3 w=vec3(dot(d,a),dot(e,b),dot(f,c))
 ;vec3 t=.8-vec3(dd(a),dd(b),dd(c))
 ;if(t.x<0.){a.x=0.;a.y=0.;t.x=0.;}
 ;if(t.y<0.){b.x=0.;b.y=0.;t.y=0.;}
 ;if(t.z<0.){c.x=0.;c.y=0.;t.z=0.;}
 ;vec3 t3=t*t*t;t*=t;t*=t
 ;a=a*8.*t3.x;a=t.x*d-a*w.x
 ;b=b*8.*t3.y;b=t.y*e-b*w.y
 ;c=c*8.*t3.z;c=t.z*f-c*w.z
 ;return 11.*vec3(dot(t,w),a+b+c);;}

// Function 1433
vec3 gerstner_noise_normal(vec3 g, float eps)
{   
    // noise normal
    vec3 n;
    n.y = map_detailed(g);    
    n.x = map_detailed(vec3(g.x+eps,g.y,g.z)) - n.y;
    n.z = map_detailed(vec3(g.x,g.y,g.z+eps)) - n.y;
    n.y = eps;
    n = normalize(n);
    
    // gerstner normal
    vec3 gn = gerstner_normal(g).xzy;
    
    // mix normals
    n.x += gn.x;
    n.y += gn.y;
    n.z *= gn.z;
    
    return normalize(n);
}

// Function 1434
vec3 snoise( in vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

	vec2  i = floor( p + (p.x+p.y)*K1 );
    vec2  a = p - i + (i.x+i.y)*K2;
    float m = step(a.y,a.x); 
    vec2  o = vec2(m,1.0-m);
    vec2  b = a - o + K2;
	vec2  c = a - 1.0 + 2.0*K2;
    vec3  h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3  n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return 1e2*n; //return full vector
}

// Function 1435
vec4 bccNoisePlaneFirst(vec3 X) {
    
    // Rotate so Z points down the main diagonal. Not a skew transform.
    mat3 orthonormalMap = mat3(
        0.788675134594813, -0.211324865405187, -0.577350269189626,
        -0.211324865405187, 0.788675134594813, -0.577350269189626,
        0.577350269189626, 0.577350269189626, 0.577350269189626);
    
    vec4 result = bccNoiseBase(orthonormalMap * X);
    return vec4(result.xyz * orthonormalMap, result.w);
}

// Function 1436
vec2 cell_noise( in vec3 p )
{
    float offset = (iChannelResolution[1].x * iChannelResolution[1].y) * texelFetch(iChannel2, ivec2(p.xy), 0).x; 
    vec2 x = vec2(mod(offset, iChannelResolution[1].x),
                  offset/iChannelResolution[1].y);
    vec2 lookup = vec2(mod(x.x + p.z, iChannelResolution[1].x), 
                       x.y + (x.x + p.z)/iChannelResolution[1].x);
    return textureLod( iChannel1, (lookup+ 0.5)/iChannelResolution[1].x, 0.0 ).xy;
}

// Function 1437
float noise( in vec3 x )
{    
    vec3 p = floor(x);
    vec3 f = fract(x);
	//f = f*f*(3.0-2.0*f);
	vec2 uv = p.xy + f.xy;
	vec2 rg = textureLod( iChannel0, (uv.xy + p.z*37.0 + 0.5)/256.0, 0. ).yx;
	
    //return rg.x*0.9;
    //return mix( rg.x, rg.y, 0.5 );
	return mix( rg.x, rg.y, f.z );
}

// Function 1438
float Noisefv3 (vec3 p)
{
  vec3 i = floor (p);
  vec3 f = fract (p);
  f = f * f * (3. - 2. * f);
  float q = dot (i, cHashA3);
  vec4 t1 = Hashv4f (q);
  vec4 t2 = Hashv4f (q + cHashA3.z);
  return mix (mix (mix (t1.x, t1.y, f.x), mix (t1.z, t1.w, f.x), f.y),
     mix (mix (t2.x, t2.y, f.x), mix (t2.z, t2.w, f.x), f.y), f.z);
}

// Function 1439
vec4 noiseInterpolate(const in vec4 x) 
{ 
    vec4 x2 = x * x;
    return x2 * x * (x * (x * 6.0 - 15.0) + 10.0); 
}

// Function 1440
float voronoi2D(vec2 uv)
{
    vec2 fl = floor(uv);
    vec2 fr = fract(uv);
    float res = 1.0;
    for(int k=-1;k<=1;k++)
    for(int j=-1; j<=1; j++ )
    {
        vec2 p = vec2(j, k);
        #if defined(ENABLE_UINT_HASH)
        float h = random(fl+p);
        #else
        float h = hash2D(fl+p);
        #endif
        vec2 vp = p-fr+h;
        float d = dot(vp, vp);

        res +=1.0/pow(d, 16.0);
    }
    return pow( 1.0/res, 1.0/16.0 );
}

// Function 1441
float noise(vec2 p) {
  vec2 g = floor(p);
  vec2 f = fract(p);
  vec2 k = f*f*f*(6.0*f*f - 15.0*f + 10.0);

  float lb = dot(shash2(g + vec2(0.0, 0.0)), vec2(0.0, 0.0) - f);
  float rb = dot(shash2(g + vec2(1.0, 0.0)), vec2(1.0, 0.0) - f);
  float lt = dot(shash2(g + vec2(0.0, 1.0)), vec2(0.0, 1.0) - f);
  float rt = dot(shash2(g + vec2(1.0, 1.0)), vec2(1.0, 1.0) - f);

  float b = mix(lb, rb, k.x);
  float t = mix(lt, rt, k.x);
  return 0.5 + 0.5 * mix(b, t, k.y);
}

// Function 1442
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
    
#if 1
	vec2 uv = (p.xy+vec2(37.0,239.0)*p.z) + f.xy;
    vec2 rg = textureLod(iChannel0,(uv+0.5)/256.0,0.0).yx;
#else
    ivec3 q = ivec3(p);
	ivec2 uv = q.xy + ivec2(37,239)*q.z;

	vec2 rg = mix(mix(texelFetch(iChannel0,(uv           )&255,0),
				      texelFetch(iChannel0,(uv+ivec2(1,0))&255,0),f.x),
				  mix(texelFetch(iChannel0,(uv+ivec2(0,1))&255,0),
				      texelFetch(iChannel0,(uv+ivec2(1,1))&255,0),f.x),f.y).yx;
#endif    
	return -1.0+2.0*mix( rg.x, rg.y, f.z );
}

// Function 1443
float EvalWhiteNoise(uvec2 uv)
{
    const uint r  = 2654435761u;
    const uint r0 = 3242174893u;
    const uint r1 = 2447445397u;
    uint h = ((uv.x * r0 + uv.y) ^ (uv.y * r1 + uv.x)) * r;
    return float(h) * (1.0 / 4294967295.0);
}

// Function 1444
float noise(in vec3 x){
	vec3 p = floor(x);
	vec3 f = fract(x);
	f = f*f*(3.0 - 2.0*f);
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 1445
vec3 noise(vec2 p){return texture(iChannel0,p/iChannelResolution[0].xy).xyz;}

// Function 1446
vec2 noiseVector(vec2 p)
{
    float n = noiseFloat(p);
    return vec2(n, noiseFloat(p + n));
}

// Function 1447
float achnoise(vec3 x){
    vec3 p = floor(x);
    vec3 fr = fract(x);
    vec3 LBZ = p + vec3(0.0, 0.0, 0.0);
    vec3 LTZ = p + vec3(0.0, 1.0, 0.0);
    vec3 RBZ = p + vec3(1.0, 0.0, 0.0);
    vec3 RTZ = p + vec3(1.0, 1.0, 0.0);

    vec3 LBF = p + vec3(0.0, 0.0, 1.0);
    vec3 LTF = p + vec3(0.0, 1.0, 1.0);
    vec3 RBF = p + vec3(1.0, 0.0, 1.0);
    vec3 RTF = p + vec3(1.0, 1.0, 1.0);

    float l0candidate1 = oct(LBZ);
    float l0candidate2 = oct(RBZ);
    float l0candidate3 = oct(LTZ);
    float l0candidate4 = oct(RTZ);

    float l0candidate5 = oct(LBF);
    float l0candidate6 = oct(RBF);
    float l0candidate7 = oct(LTF);
    float l0candidate8 = oct(RTF);

    float l1candidate1 = mix(l0candidate1, l0candidate2, fr[0]);
    float l1candidate2 = mix(l0candidate3, l0candidate4, fr[0]);
    float l1candidate3 = mix(l0candidate5, l0candidate6, fr[0]);
    float l1candidate4 = mix(l0candidate7, l0candidate8, fr[0]);


    float l2candidate1 = mix(l1candidate1, l1candidate2, fr[1]);
    float l2candidate2 = mix(l1candidate3, l1candidate4, fr[1]);


    float l3candidate1 = mix(l2candidate1, l2candidate2, fr[2]);

    return l3candidate1;
}

// Function 1448
float noiseNew( in vec3 x )
{
    return texture( iChannel0, x/32.0 ).x;    // <---------- Sample a 3D texture!
}

// Function 1449
float Voronoi(in vec2 p){
    
    // Partitioning the grid into unit squares and determining the fractional position.
	vec2 g = floor(p), o; p -= g;
	
    // "d.x" and "d.y" represent the closest and second closest distances
    // respectively, and "d.z" holds the distance comparison value.
	vec3 d = vec3(2); // 8., 2, 1.4, etc. 
    
    // A 4x4 grid sample is required for the smooth minimum version.
	for(int j = -1; j <= 2; j++){
		for(int i = -1; i <= 2; i++){
            
			o = vec2(i, j); // Grid reference.
             // Note the offset distance restriction in the hash function.
            o += hash22(g + o) - p; // Current position to offset point vector.
            
            // Distance metric. Unfortunately, the Euclidean distance needs
            // to be used for clean equidistant-looking cell border lines.
            // Having said that, there might be a way around it, but this isn't
            // a GPU intensive example, so I'm sure it'll be fine.
			d.z = length(o); 
            
            // Up until this point, it's been a regular Voronoi example. The only
            // difference here is the the mild smooth minimum's to round things
            // off a bit. Replace with regular mimimum functions and it goes back
            // to a regular second order Voronoi example.
            d.y = max(d.x, smin(d.y, d.z, .4)); // Second closest point with smoothing factor.
            d.x = smin(d.x, d.z, .2); // Closest point with smoothing factor.
            
            // Based on IQ's suggestion - A commutative exponential-based smooth minimum.
            // This algorithm is just an approximation, so it doesn't make much of a difference,
            // but it's here anyway.
            //d.y = max(d.x, sminExp(d.y, d.z, 10.)); // Second closest point with smoothing factor.
            //d.x = sminExp(d.x, d.z, 20.); // Closest point with smoothing factor.

                       
		}
	}    
	
    // Return the regular second closest minus closest (F2 - F1) distance.
    return d.y - d.x;
    
}

// Function 1450
float achnoise(vec4 x){ 
	vec4 p = floor(x);
	vec4 fr = fract(x);
	vec4 LBZU = p + vec4(0.0, 0.0, 0.0, 0.0);
	vec4 LTZU = p + vec4(0.0, 1.0, 0.0, 0.0);
	vec4 RBZU = p + vec4(1.0, 0.0, 0.0, 0.0);
	vec4 RTZU = p + vec4(1.0, 1.0, 0.0, 0.0);
	                 
	vec4 LBFU = p + vec4(0.0, 0.0, 1.0, 0.0);
	vec4 LTFU = p + vec4(0.0, 1.0, 1.0, 0.0);
	vec4 RBFU = p + vec4(1.0, 0.0, 1.0, 0.0);
	vec4 RTFU = p + vec4(1.0, 1.0, 1.0, 0.0);
	                 
	vec4 LBZD = p + vec4(0.0, 0.0, 0.0, 1.0);
	vec4 LTZD = p + vec4(0.0, 1.0, 0.0, 1.0);
	vec4 RBZD = p + vec4(1.0, 0.0, 0.0, 1.0);
	vec4 RTZD = p + vec4(1.0, 1.0, 0.0, 1.0);
	                 
	vec4 LBFD = p + vec4(0.0, 0.0, 1.0, 1.0);
	vec4 LTFD = p + vec4(0.0, 1.0, 1.0, 1.0);
	vec4 RBFD = p + vec4(1.0, 0.0, 1.0, 1.0);
	vec4 RTFD = p + vec4(1.0, 1.0, 1.0, 1.0);
	
	float l0candidate1  = oct(LBZU);
	float l0candidate2  = oct(RBZU);
	float l0candidate3  = oct(LTZU);
	float l0candidate4  = oct(RTZU);
	
	float l0candidate5  = oct(LBFU);
	float l0candidate6  = oct(RBFU);
	float l0candidate7  = oct(LTFU);
	float l0candidate8  = oct(RTFU);
	
	float l0candidate9  = oct(LBZD);
	float l0candidate10 = oct(RBZD);
	float l0candidate11 = oct(LTZD);
	float l0candidate12 = oct(RTZD);
	
	float l0candidate13 = oct(LBFD);
	float l0candidate14 = oct(RBFD);
	float l0candidate15 = oct(LTFD);
	float l0candidate16 = oct(RTFD);
	
	float l1candidate1 = mix(l0candidate1, l0candidate2, fr[0]);
	float l1candidate2 = mix(l0candidate3, l0candidate4, fr[0]);
	float l1candidate3 = mix(l0candidate5, l0candidate6, fr[0]);
	float l1candidate4 = mix(l0candidate7, l0candidate8, fr[0]);
	float l1candidate5 = mix(l0candidate9, l0candidate10, fr[0]);
	float l1candidate6 = mix(l0candidate11, l0candidate12, fr[0]);
	float l1candidate7 = mix(l0candidate13, l0candidate14, fr[0]);
	float l1candidate8 = mix(l0candidate15, l0candidate16, fr[0]);
	
	
	float l2candidate1 = mix(l1candidate1, l1candidate2, fr[1]);
	float l2candidate2 = mix(l1candidate3, l1candidate4, fr[1]);
	float l2candidate3 = mix(l1candidate5, l1candidate6, fr[1]);
	float l2candidate4 = mix(l1candidate7, l1candidate8, fr[1]);
	
	
	float l3candidate1 = mix(l2candidate1, l2candidate2, fr[2]);
	float l3candidate2 = mix(l2candidate3, l2candidate4, fr[2]);
	
	float l4candidate1 = mix(l3candidate1, l3candidate2, fr[3]);
	
	return l4candidate1;
}

// Function 1451
float noise (vec2 v) {
    vec4 n = vec4(floor(v),ceil(v));
    vec4 h = vec4(hash(n.xy),hash(n.zy),hash(n.xw),hash(n.zw));
    return mix(mix(h.x,h.y,v.x-n.x),mix(h.z,h.w,v.x-n.x),v.y-n.y);
}

// Function 1452
float snoise(vec3 v)
{ 
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    i = mod289(i); 
    vec4 p = 
        permute
        (
            permute
            ( 
                permute
                (
                    i.z + vec4(0.0, i1.z, i2.z, 1.0)
                )
                + i.y + vec4(0.0, i1.y, i2.y, 1.0 )
            )
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 )
        );

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
    //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
}

// Function 1453
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);
	return textureLod( iChannel0, (uv+118.4)/256.0, 0. ).x;
}

// Function 1454
float fbmPerlin(vec2 pos, vec2 scale, int octaves, float shift, float axialShift, float gain, float lacunarity, uint mode, float factor, float offset, float seed) 
{
    float amplitude = gain;
    vec2 frequency = floor(scale);
    float angle = axialShift;
    float n = 1.0;
    vec2 p = fract(pos) * frequency;

    float value = 0.0;
    for (int i = 0; i < octaves; i++) 
    {
        float pn = perlinNoise(p / frequency, frequency, angle, seed) + offset;
        if (mode == 0u)
        {
            n *= abs(pn);
        }
        else if (mode == 1u)
        {
            n = abs(pn);
        }
        else if (mode == 2u)
        {
            n = pn;
        }
        else if (mode == 3u)
        {
            n *= pn;
        }
        else if (mode == 4u)
        {
            n = pn * 0.5 + 0.5;
        }
        else
        {
            n *= pn * 0.5 + 0.5;
        }
        
        n = pow(n < 0.0 ? 0.0 : n, factor);
        value += amplitude * n;
        
        p = p * lacunarity + shift;
        frequency *= lacunarity;
        amplitude *= gain;
        angle += axialShift;
    }
    return value;
}

// Function 1455
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0 ).yx;
	
	return mix( rg.x, rg.y, f.z );
}

// Function 1456
vec2 voronoi(vec2 p){
    //initialize distance to a very high number
    float d = 9e9;
    //declare an integer for kepping tack of which point 'p' is closest to
    int index = 0;
    //usually when renderring voronoi diagrams you only loop through the nearest cells,
    //but since we're using such an unordinary metric that would proably lead to lots of artifacts
    for(int i = 0; i < numCells; i++){
        //declare 'o', which will be the center of the cell
        vec2 c;
        //for the first cell, we make o depend on mouse position
        if(i == 0) c=(iMouse.xy==vec2(0))?vec2(0):(2.0*iMouse.xy-iResolution.xy)/iResolution.y;
        else {
            //otherwise we generate a random number and do some stuff with it to make center move 
            //around inside the disk
            vec3 h = hash31(float(i));
            c = .5 * h.xy - .25;
            c = mix(c, -c, .5 + .5 * sin(h.z * iTime));
            c = rot(c, (h.z - h.x - h.y) * iTime);
            c *= inversesqrt(sqrt(dot(c, c)));
        }
        //calculate the distance from the center to p using the Poincare metric
        float d0 = metric(p, c);
        //if that distance is less than the current minimum distance, reset the minimum distance to
        //it and do the same for the index of the cell
        if(d0 < d){index = i; d = d0;}
        //if all we wanted was distance data, i.e. for making Worley noise, we could just say
        //d=min(d,d0) which would be nice since its less branching, but we want the index of the cell
        //as well so we need the conditional.
    }
    return vec2(index, d);
}

// Function 1457
float snoise(vec3 x) {
	vec3 p = floor(x);
	vec3 f = fract(x);
	f = f * f * (3.0 - 2.0 * f);

	float n = p.x + p.y * 157.0 + 113.0 * p.z;
	return mix(
			mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
					mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y),
			mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
					mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
}

// Function 1458
float CellNoise(in vec3 p, in float numCells)
{
	p *= numCells;
	float d = 1.0e10;
	for (int xo = -1; xo <= 1; xo++)
	{
		for (int yo = -1; yo <= 1; yo++)
		{
            for (int zo = -1; zo <= 1; zo++)
		    {
			    vec3 tp = floor(p) + vec3(xo, yo, zo);
			    tp = p - tp - HashTex(tp, numCells);
			    d = min(d, dot(tp, tp));
            }
		}
	}
    float r = 1.0 - sqrt(d);
    //float r = 1.0 - d;// ...Bubbles
	return -1.0 + 2.0 * r;
}

// Function 1459
float perlin(vec3 p){
	vec3 fr= fract(p);
	vec3 frn= fr-1.;
	vec3 f= floor(p);
	vec3 c= ceil(p);
	vec3 nnn= nmaps(rand(vec3(f.x,f.y,f.z)));
	vec3 nnp= nmaps(rand(vec3(f.x,f.y,c.z)));
	vec3 npn= nmaps(rand(vec3(f.x,c.y,f.z)));
	vec3 npp= nmaps(rand(vec3(f.x,c.y,c.z)));
	vec3 pnn= nmaps(rand(vec3(c.x,f.y,f.z)));
	vec3 pnp= nmaps(rand(vec3(c.x,f.y,c.z)));
	vec3 ppn= nmaps(rand(vec3(c.x,c.y,f.z)));
	vec3 ppp= nmaps(rand(vec3(c.x,c.y,c.z)));
	float d_nnn= dot(nnn, vec3(fr .x, fr .y, fr .z));
	float d_nnp= dot(nnp, vec3(fr .x, fr .y, frn.z));
	float d_npn= dot(npn, vec3(fr .x, frn.y, fr .z));
	float d_npp= dot(npp, vec3(fr .x, frn.y, frn.z));
	float d_pnn= dot(pnn, vec3(frn.x, fr .y, fr .z));
	float d_pnp= dot(pnp, vec3(frn.x, fr .y, frn.z));
	float d_ppn= dot(ppn, vec3(frn.x, frn.y, fr .z));
	float d_ppp= dot(ppp, vec3(frn.x, frn.y, frn.z));
	vec4 zn= vec4(
		d_nnn,
		d_npn,
		d_pnn,
		d_ppn
	);
	vec4 zp= vec4(
		d_nnp,
		d_npp,
		d_pnp,
		d_ppp
	);
	vec4 lx= lerp(zn,zp, smooth(fr.zzzz));
	vec2 ly= lerp(lx.xz, lx.yw, smooth(fr.yy));
	return nmapu(lerp(ly.x,ly.y, smooth(fr.x)));
}

// Function 1460
float noise(in vec2 x) {
	vec2 p = floor(x);
	vec2 f = fract(x);
		
	f = f*f*(3.0-2.0*f);	
	float n = p.x + p.y*57.0;
	
	float res = mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
					mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);
	return res;
}

// Function 1461
vec4 noised( in vec3 x, in float scale )
{
    x *= scale;

    // grid
    vec3 i = floor(x);
    vec3 w = fract(x);
    
    #if 1
    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
    #else
    // cubic interpolant
    vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
    #endif    
    
    // gradients
    vec3 ga = hash( i+vec3(0.0,0.0,0.0), scale );
    vec3 gb = hash( i+vec3(1.0,0.0,0.0), scale );
    vec3 gc = hash( i+vec3(0.0,1.0,0.0), scale );
    vec3 gd = hash( i+vec3(1.0,1.0,0.0), scale );
    vec3 ge = hash( i+vec3(0.0,0.0,1.0), scale );
	vec3 gf = hash( i+vec3(1.0,0.0,1.0), scale );
    vec3 gg = hash( i+vec3(0.0,1.0,1.0), scale );
    vec3 gh = hash( i+vec3(1.0,1.0,1.0), scale );
    
    // projections
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
	
    // interpolations
    return vec4( va + u.x*(vb-va) + u.y*(vc-va) + u.z*(ve-va) + u.x*u.y*(va-vb-vc+vd) + u.y*u.z*(va-vc-ve+vg) + u.z*u.x*(va-vb-ve+vf) + (-va+vb+vc-vd+ve-vf-vg+vh)*u.x*u.y*u.z,    // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.z*(ge-ga) + u.x*u.y*(ga-gb-gc+gd) + u.y*u.z*(ga-gc-ge+gg) + u.z*u.x*(ga-gb-ge+gf) + (-ga+gb+gc-gd+ge-gf-gg+gh)*u.x*u.y*u.z +   // derivatives
                 du * (vec3(vb,vc,ve) - va + u.yzx*vec3(va-vb-vc+vd,va-vc-ve+vg,va-vb-ve+vf) + u.zxy*vec3(va-vb-ve+vf,va-vb-vc+vd,va-vc-ve+vg) + u.yzx*u.zxy*(-va+vb+vc-vd+ve-vf-vg+vh) ));
}

// Function 1462
vec3 addnoise( in vec2 uv )
{
    vec3 x = texture( iChannel0, uv*3.0 ).rgb;
    return (x - 0.5);
}

// Function 1463
float layeredNoise( vec3 p )
{
    float f;
    f  = 0.5000* textureLod( iChannel1, p, 0.0 ).x; p = p*2.0;
    f += 0.2500* textureLod( iChannel1, p, 0.0 ).x; p = p*2.0;
    f += 0.1250* textureLod( iChannel1, p, 0.0 ).x; p = p*2.0;
    f += 0.0800* textureLod( iChannel1, p, 0.0 ).x; p = p*2.0;
    f += 0.0625* textureLod( iChannel1, p, 0.0 ).x; p = p*2.0;
    
    return f;
}

// Function 1464
vec3 dotsNoise(vec2 pos, vec2 scale, float density, float size, float sizeVariation, float roundness, float seed) 
{
    pos *= scale;
    vec4 i = floor(pos).xyxy + vec2(0.0, 1.0).xxyy;
    vec2 f = pos - i.xy;
    i = mod(i, scale.xyxy);
    
    vec4 hash = hash4D(i.xy + seed);
    if (hash.w > density)
        return vec3(0.0);

    float radius = clamp(size + (hash.z * 2.0 - 1.0) * sizeVariation * 0.5, 0.0, 1.0);
    float value = radius / size;  
    radius = 2.0 / radius;
    f = f * radius - (radius - 1.0);
    f += hash.xy * (radius - 2.0);
    f = pow(abs(f), vec2((mix(20.0, 1.0, sqrt(roundness)))));

    float u = 1.0 - min(dot(f, f), 1.0);
    return vec3(clamp(u * u * u * value, 0.0, 1.0), hash.w, hash.z);
}

// Function 1465
float noise(int x,int y)
{   
    float fx = float(x);
    float fy = float(y);
    
    return 2.0 * fract(sin(dot(vec2(fx, fy) ,vec2(12.9898,78.233))) * 43758.5453) - 1.0;
}

// Function 1466
float getNoise( vec3 x )
{
    x *= 50.0;
    // The noise function returns a value in the range -1.0f -> 1.0f

    vec3 p = floor(x);
    vec3 f = fract(x);

    f       = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;

    return mix(mix(mix( hash(n+0.0), hash(n+1.0),f.x),
                   mix( hash(n+57.0), hash(n+58.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

// Function 1467
float noise(vec3 p)
{
	vec3 ip=floor(p);
	p-=ip;
	vec3 s=vec3(7, 157, 113);
	vec4 h=vec4(0., s.yz, s.y+s.z)+dot(ip, s);
	p=p*p*(3.-2.*p);
	h=mix(fract(sin(h)*43758.5), fract(sin(h+s.x)*43758.5), p.x);
	h.xy=mix(h.xz, h.yw, p.y);
	return mix(h.x, h.y, p.z);
}

// Function 1468
vec3 warpedNoise(in vec2 q) {

	vec2 o = vec2(0.0);
    o.x = 0.5*fbm6(vec2(2.0*q));
    o.y = 0.5*fbm6(vec2(2.0*q));
    
    vec2 n = vec2(0.0);
    n.x = fbm6(vec2(7.0*o+vec2(19.2)));
    n.y = fbm6(vec2(7.0*o+vec2(15.7)));

    vec2 p = 4.0*q + 4.0*n;
    
    float f = 0.5 + 0.5 * fbm4(p);
    
    // Rendering the value in different ways
    float time = mod(iTime - 13.0, 4.0 * 4.0);
    
    // Option 1: Grayscale rendering
    if (time < 4.0) {
    	return 0.8 * vec3(f);
    }
    
    // Option 2: Rendering with steps
    if (time < 8.0) {
    	float steps = 10.0;
    	return 0.8 * vec3(floor(f * steps) / steps);
    }
    
    // Option 3: Dirt-like Colours
    if (time < 12.0) {
    	float steps = 10.0;
    	float val = floor(f * steps) / steps;
        val = min(max(val, 0.6), 0.8);
    	return 0.45 + 0.35*sin(vec3(0.05,0.08,0.10)*(val*1583343.0));
    }
        
    // Option 4: Camo-like Colours
    else {
    	float steps = 10.0;
    	float val = floor(f * steps) / steps;
        val = min(max(val, 0.6), 0.8);
    	return 0.25 + 0.15*sin(vec3(0.05,0.08,0.10)*(val*1234.1));
    }
        
    // Option 5: Time-based Colours (unused)
    //float steps = 10.0;
    //float val = floor(f * steps) / steps;
    // Color function from iq: https://www.shadertoy.com/view/Xds3zN
    //return 0.45 + 0.35*sin(vec3(0.05,0.08,0.10)*(val*iTime*10.0));

}

// Function 1469
float noise(vec2 uv){
	vec2 i = floor(uv);
    vec2 f = fract(uv);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    vec2 u = f*f*(3.0-2.0*f);
    // u = smoothstep(0.,1.,f);

    // Mix 4 coorners percentages
    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Function 1470
vec3 noise(vec3 pos, vec3 k, vec3 p, vec3 s)
{
    vec3 t = repeatTime(s);
    float X1 = cos(dot(pos.yz,k.yz) + p.x - t.x);
    float Y1 = cos(dot(pos.zx,k.zx) + p.y - t.y);
    float Z1 = cos(dot(pos.xy,k.xy) + p.z - t.z);

    float X2 = cos(dot(vec2(Y1, Z1),k.yz) + p.x + t.x);
    float Y2 = cos(dot(vec2(Z1, X1),k.zx) + p.y + t.y);
    float Z2 = cos(dot(vec2(X1, Y1),k.xy) + p.z + t.z);
 	return vec3(X2, Y2, Z2); 
}

// Function 1471
float Noise(vec2 p, float x)
{
    vec2 i = sin(floor(p))*x;
    i += floor(p)*(1.0-x);
    vec2 f = fract(p);
    vec2 u = (f * f * (3.0 - 2.0 * f));
    
    return mix(mix(Hash(i + vec2(0.0, 0.0)),
                   Hash(i + vec2(1.0, 0.0)), u.x),
               mix(Hash(i + vec2(0.0, 1.0)),
                   Hash(i + vec2(1.0, 1.0)), u.x), u.y);
 
}

// Function 1472
float noise(vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix(hash(n+  0.0), hash(n+  1.0),f.x),
                   mix(hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix(hash(n+113.0), hash(n+114.0),f.x),
                   mix(hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 1473
float noise( float p )
{
    float i = floor( p );
    float f = fract( p );
	float u = f*f*(3.0-2.0*f);
    return mix( hash( i ), hash( i + 1.0 ), u);
}

// Function 1474
float octaveNoise (vec2 st) {
    float total = 0.0;
    vec2 freq = FREQUENCY;
    float amp = AMPLITUDE;
    // loop from large frequency to small to add smaller scale of noise
    for(int i=0;i<OCTAVE;i++) {
        total += noise(st*freq-iTime*SPEED) * amp;
        amp *= PERSISTENCE;
        freq *= vec2(2.1,2.1);
    }

    return total;
}

// Function 1475
float supernoise3d(vec3 p){

	float a =  noise3d(p);
	float b =  noise3d(p + 10.5);
	return (a + b) * 0.5;
}

// Function 1476
float iqnoise(in vec2 pos, float irregular, float smoothness)
{
	vec2 cell = floor(pos);
	vec2 cellOffset = fract(pos);

	float sharpness = 1.0 + 63.0 * pow(1.0-smoothness, 4.0);
	
	float value = 0.0;
	float accum = 0.0;
	// Sample the surrounding cells, from -2 to +2
	// This is necessary for the smoothing as well as the irregular grid.
	for(int x=-2; x<=2; x++ )
	for(int y=-2; y<=2; y++ )
	{
		vec2 samplePos = vec2(float(y), float(x));

  		// Center of the cell is not at the center of the block for irregular noise.
  		// Note that all the coordinates are in "block"-space, 0 is the current block, 1 is one block further, etc
		vec2 center = rand2(cell + samplePos) * irregular;
		float centerDistance = length(samplePos - cellOffset + center);

		// High sharpness = Only extreme values = Hard borders = 64
		// Low sharpness = No extreme values = Soft borders = 1
		float samplex = pow(1.0 - smoothstep(0.0, 1.414, centerDistance), sharpness);

		// A different "color" (shade of gray) for each cell
		float color = rand(cell + samplePos);
		value += color * samplex;
		accum += samplex;
	}

	return value/accum;
}

// Function 1477
float organicNoise(vec2 pos, vec2 scale, float density, vec2 phase, float contrast, float highlights, float shift, float seed)
{
    vec2 s = mix(vec2(1.0), scale - 1.0, density);
    float nx = perlinNoise(pos + phase, scale, seed);
    float ny = perlinNoise(pos, s, seed);

    float n = length(vec2(nx, ny) * mix(vec2(2.0, 0.0), vec2(0.0, 2.0), shift));
    n = pow(n, 1.0 + 8.0 * contrast) + (0.15 * highlights) / n;
    return n * 0.5;
}

// Function 1478
float noise2D( vec2 p ) {
	vec2 f = fract(p);
	p = floor(p);
	float v = p.x+p.y*1000.0;
	vec4 r = vec4(v, v+1.0, v+1000.0, v+1001.0);
	r = fract(100000.0*sin(r*.001));
	f = f*f*(3.0-2.0*f);
	return 2.0*(mix(mix(r.x, r.y, f.x), mix(r.z, r.w, f.x), f.y))-1.0;
}

// Function 1479
float noise(vec2 p) {

    return random(p.x + p.y * 10000.);
            
}

// Function 1480
vec3 noise3(in vec3 uv, in float shift_by) {
    return vec3(simple_noise(uv, shift_by),
                simple_noise(uv + vec3(0.0, 0.0, 101.0), shift_by),
               simple_noise(uv + vec3(0.0, 101.0, 0.0), shift_by));
}

// Function 1481
float noise(vec2 p) {
    return snoise(p);// + snoise(p + snoise(p));
//    return (snoise(p) * 64.0 + snoise(p * 2.0) * 32.0 + snoise(p * 4.0) * 16.0 + snoise(p * 8.0) * 8.0 + snoise(p * 16.0) * 4.0 + snoise(p * 32.0) * 2.0 + snoise(p * 64.0)) / (1.0 + 2.0 + 4.0 + 8.0 + 16.0 + 32.0 + 64.0);
}

// Function 1482
float noise(vec3 p)
{
	vec3 ip=floor(p);
    p-=ip; 
    vec3 s=vec3(7,157,113);
    vec4 h=vec4(0.,s.yz,s.y+s.z)+dot(ip.xyz,s);
    p=p*p*(3.-2.*p); 
    h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
    h.xy=mix(h.xz,h.yw,p.y);
    return mix(h.x,h.y,p.z); 
}

// Function 1483
float SpiralNoiseC(vec3 p)
{
    float n = 0.0;	// noise amount
    float iter = 1.0;
    for (int i = 0; i < 6; i++)
    {
        // add sin and cos scaled inverse with the frequency
        n += -abs(sin(p.y*iter) + cos(p.x*iter)) / iter;	// abs for a ridged look
        // rotate by adding perpendicular and scaling down
        p.xy += vec2(p.y, -p.x) * nudge;
        p.xy *= normalizer;
        // rotate on other axis
        p.xz += vec2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        // increase the frequency
        iter *= 1.733733;
    }
    return n;
}

// Function 1484
float noise(in vec2 p){vec2 i=floor(p);vec2 f=fract(p);vec2 u=f*f*(3.0-2.0*f);return mix(mix(hash(i+vec2(0.0,0.0)),hash(i+vec2(1.0,0.0)),
u.x),mix(hash(i+vec2(0.0,1.0)),hash(i+vec2(1.0,1.0)),u.x),u.y);}

// Function 1485
vec4 Noise( in vec2 x )
{
    x = x*sqrt(3./4.) + x.yx*vec2(1,-1)*sqrt(1./4.); // tilt the grid so it's not aligned to the flag to make it less visible

    vec2 p = floor(x);
    vec2 f = fract(x);
	f = f*f*(3.0-2.0*f);
//	vec2 f2 = f*f; f = f*f2*(10.0-15.0*f+6.0*f2);

	vec2 uv = p + f;

#if (0)
	vec4 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0 );
#else
	// on some hardware interpolation lacks precision
    ivec2 iuv = ivec2(floor(uv));
    vec2 fuv = uv - vec2(iuv);
    
	vec4 rg = mix( mix(
				texelFetch( iChannel0, iuv&255, 0 ),
				texelFetch( iChannel0, (iuv+ivec2(1,0))&255, 0 ),
				fuv.x ),
				  mix(
				texelFetch( iChannel0, (iuv+ivec2(0,1))&255, 0 ),
				texelFetch( iChannel0, (iuv+ivec2(1,1))&255, 0 ),
				fuv.x ),
				fuv.y );
#endif			  

	return rg;
}

// Function 1486
float polarNoise2N(vec3 pos)
{
    float a = 2.*atan(pos.y, pos.x);
    vec3 pos1 = vec3(pos.z, length(pos.yx) + iTime*txtSpeed, a);
    vec3 pos2 = vec3(pos.z, length(pos.yx) + iTime*txtSpeed, a+12.57);    
    
    float f1 = polarNoiseN(pos1);
    float f2 = polarNoiseN(pos2);
    float f = mix(f1, f2, smoothstep(-5., -6.285, a));
    
    //f = smoothstep(0.01, 0.2, f)-smoothstep(0.2, 0.52, f)+smoothstep(0.45, 0.63, f);
    
    return f;
}

// Function 1487
float noise( vec2 p )
{
    vec2 i = floor( p ),
         f = fract( p ),
	  // u = f*f*(3.-2.*f);
         u = f*f*f* ( 10. + f * ( -15. + 6.* f ) ); // better for derivative. from http://staffwww.itn.liu.se/~stegu/TNM022-2005/perlinnoiselinks/paper445.pdf

#define P(x,y) dot( hash( i + vec2(x,y) ), f - vec2(x,y) )
    return mix( mix( P(0,0), P(1,0), u.x),
                mix( P(0,1), P(1,1), u.x), u.y);
}

// Function 1488
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix( hash13(p+vec3(0,0,0)), 
                        hash13(p+vec3(1,0,0)),f.x),
                   mix( hash13(p+vec3(0,1,0)), 
                        hash13(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash13(p+vec3(0,0,1)), 
                        hash13(p+vec3(1,0,1)),f.x),
                   mix( hash13(p+vec3(0,1,1)), 
                        hash13(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 1489
float getPerlin(vec2 coord)
{
	int xi0 = int(floor(coord.x)); 
 	int yi0 = int(floor(coord.y)); 
 	int xi1 = xi0 + 1; 
 	int yi1 = yi0 + 1;
 	float tx = coord.x - floor(coord.x); 
 	float ty = coord.y - floor(coord.y); 	
 	float u = smoothFloat(tx);
 	float v = smoothFloat(ty);
 	// gradients at the corner of the cell
 	vec2 c00 = getGradient(vec2(xi0, yi0));
 	vec2 c10 = getGradient(vec2(xi1, yi0)); 
 	vec2 c01 = getGradient(vec2(xi0, yi1)); 
 	vec2 c11 = getGradient(vec2(xi1, yi1));
 	// generate vectors going from the grid points to p
 	float x0 = tx, x1 = tx - 1.;
 	float y0 = ty, y1 = ty - 1.;
 	vec2 p00 = vec2(x0, y0); 
 	vec2 p10 = vec2(x1, y0); 
 	vec2 p01 = vec2(x0, y1); 
 	vec2 p11 = vec2(x1, y1);	
 	// linear interpolation
 	float a = mix(dot(c00, p00), dot(c10, p10), u); 
 	float b = mix(dot(c01, p01), dot(c11, p11), u);
 	return abs(mix(a, b, v)); // g 
}

// Function 1490
float cnoise2(vec2 P)
{
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod289(Pi); // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;

  vec4 i = permute(permute(ix) + iy);

  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
  vec4 gy = abs(gx) - 0.5 ;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;

  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);

  vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;

  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));

  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

// Function 1491
float SpiralNoiseC(vec3 p)
{
    float n = 0.0;	// noise amount
    float iter = 1.0;
    for (int i = 0; i < 8; i++)
    {
        // add sin and cos scaled inverse with the frequency
        n += -abs(sin(p.y*iter) + cos(p.x*iter)) / iter;	// abs for a ridged look
        // rotate by adding perpendicular and scaling down
        p.xy += vec2(p.y, -p.x) * nudge;
        p.xy *= normalizer;
        // rotate on other axis
        p.xz += vec2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        // increase the frequency
        iter *= 1.733733;
    }
    return n;
}

// Function 1492
vec3 voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( iTime + 6.2831*o );
        #endif
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    vec2 ml = vec2(0.0);
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( iTime + 6.2831*o );
        #endif	
        vec2 r = g + o - f;

        if( dot(r-mr,r-mr)>0.00001 ) {
            vec2 l = normalize(r-mr);
            float d = dot( 0.5*(mr+r), l );
            if (md > d) { 
        		md = d;
                ml = l;
            }
        }           
    }

    return vec3(md, ml);
}

// Function 1493
float smoothVoronoi(vec2 p, float falloff) {

    vec2 ip = floor(p); p -= ip;
	
	float d = 1., res = 0.0;
	
	for(int i = -1; i <= 2; i++) {
		for(int j = -1; j <= 2; j++) {
            
			vec2 b = vec2(i, j);
            
			vec2 v = b - p + hash22(ip + b);
            
			d = max(dot(v,v), 1e-4);
			
			res += 1.0/pow( d, falloff );
		}
	}

	return pow( 1./res, .5/falloff );
}

// Function 1494
float movingNoise(vec2 p) {
  float total = 0.0;
  total += smoothNoise(p     - time);
  total += smoothNoise(p*2.  + time) / 2.;
  total += smoothNoise(p*4.  - time) / 4.;
  total += smoothNoise(p*8.  + time) / 8.;
  total += smoothNoise(p*16. - time) / 16.;
  total /= 1. + 1./2. + 1./4. + 1./8. + 1./16.;
  return total;
}

// Function 1495
float perlinNoise( in float x ) {
    float lower = floor(x);
    float upper = lower + 1.0;
    float lowerV = hash(lower);
    float upperV = hash(upper);
    return smoothstep(lower, upper, x) * (upperV - lowerV) + lowerV;
}

// Function 1496
float Noise( in vec2 f )
{
    vec2 p = floor(f);
    f = fract(f);
    f = f*f*(3.0-2.0*f);
    float res = textureLod(iChannel0, (p+f+.5)/256.0, 0.0).x;
    return res;
}

// Function 1497
vec3 noise3( in vec3 x)
{
	return vec3( noise(x+vec3(123.456,.567,.37)),
				noise(x+vec3(.11,47.43,19.17)),
				noise(x) );
}

// Function 1498
float voronoi(in vec2 x,float t)
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    
    vec2 res = vec2(8.0);
    for(int j = -1; j <= 1; j ++)
    {
        for(int i = -1; i <= 1; i ++)
        {
            vec2 b = vec2(i, j);
            vec2 r = vec2(b) - f + rand2(p + b,t);
            
            // chebyshev distance, one of many ways to do this
            float d = sqrt(abs(r.x*r.x) + abs(r.y*r.y));
            
            if(d < res.x)
            {
                res.y = res.x;
                res.x = d;
            }
            else if(d < res.y)
            {
                res.y = d;
            }
        }
    }
    return res.y - res.x;
}

// Function 1499
vec2 Noise2(vec2 x) {     // pseudoperlin improvement from foxes idea 
    return (noise2(x)+noise2(x+11.5)) / 2.;
}

// Function 1500
float valueNoise( vec2 p )
{
    // i is an integer that allow us to move along grid points.
    vec2 i = floor( p );
    // f will be used as an offset between the grid points.
    vec2 f = fract( p );
    
    // Hermite Curve.
    // The formula 3f^2 - 2f^3 generates an S curve between 0.0 and 1.0.
    // If we factor out the variable f, we get f*f*(3.0 - 2.0*f)
    // This allows us to smoothly interpolate along an s curve between our grid points.
    // To see the S curve graph, go to the following url.
    // https://www.desmos.com/calculator/mnrgw3yias
    f = f*f*(3.0 - 2.0*f);
    
    // Interpolate the along the bottom of our grid.
    float bottomOfGrid =    mix( hash( i + vec2( 0.0, 0.0 ) ), hash( i + vec2( 1.0, 0.0 ) ), f.x );
    // Interpolate the along the top of our grid.
    float topOfGrid =       mix( hash( i + vec2( 0.0, 1.0 ) ), hash( i + vec2( 1.0, 1.0 ) ), f.x );

    // We have now interpolated the horizontal top and bottom grid lines.
    // We will now interpolate the vertical line between those 2 horizontal points
    // to get our final value for noise.
    float t = mix( bottomOfGrid, topOfGrid, f.y );
    
    return t;
}

// Function 1501
float noise( const in vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 1502
float blobnoise(vec2 v, float s)
{
    return pow(.5 + .5 * cos(pi * clamp(simplegridnoise(v)*2., 0., 1.)), s);
}

// Function 1503
vec4 temporalNoise(vec2 uv, vec2 sz)
{
    vec2 tz = iChannelResolution[2].xy;
    vec2 tv = uv + fract(vec2(iTime * 257.0, iTime * 359.0));
    vec2 tx = tv * sz;
    vec4 cl = texture(iChannel2, tx/tz);
    return clamp((cl - 0.5) * 2.0 + 0.5,0.0,1.0);
}

// Function 1504
float achnoise(vec4 x){
    vec4 p = floor(x);
    vec4 fr = fract(x);
    vec4 LBZU = p + vec4(0.0, 0.0, 0.0, 0.0);
    vec4 LTZU = p + vec4(0.0, 1.0, 0.0, 0.0);
    vec4 RBZU = p + vec4(1.0, 0.0, 0.0, 0.0);
    vec4 RTZU = p + vec4(1.0, 1.0, 0.0, 0.0);

    vec4 LBFU = p + vec4(0.0, 0.0, 1.0, 0.0);
    vec4 LTFU = p + vec4(0.0, 1.0, 1.0, 0.0);
    vec4 RBFU = p + vec4(1.0, 0.0, 1.0, 0.0);
    vec4 RTFU = p + vec4(1.0, 1.0, 1.0, 0.0);

    vec4 LBZD = p + vec4(0.0, 0.0, 0.0, 1.0);
    vec4 LTZD = p + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 RBZD = p + vec4(1.0, 0.0, 0.0, 1.0);
    vec4 RTZD = p + vec4(1.0, 1.0, 0.0, 1.0);

    vec4 LBFD = p + vec4(0.0, 0.0, 1.0, 1.0);
    vec4 LTFD = p + vec4(0.0, 1.0, 1.0, 1.0);
    vec4 RBFD = p + vec4(1.0, 0.0, 1.0, 1.0);
    vec4 RTFD = p + vec4(1.0, 1.0, 1.0, 1.0);

    float l0candidate1  = oct(LBZU);
    float l0candidate2  = oct(RBZU);
    float l0candidate3  = oct(LTZU);
    float l0candidate4  = oct(RTZU);

    float l0candidate5  = oct(LBFU);
    float l0candidate6  = oct(RBFU);
    float l0candidate7  = oct(LTFU);
    float l0candidate8  = oct(RTFU);

    float l0candidate9  = oct(LBZD);
    float l0candidate10 = oct(RBZD);
    float l0candidate11 = oct(LTZD);
    float l0candidate12 = oct(RTZD);

    float l0candidate13 = oct(LBFD);
    float l0candidate14 = oct(RBFD);
    float l0candidate15 = oct(LTFD);
    float l0candidate16 = oct(RTFD);

    float l1candidate1 = mix(l0candidate1, l0candidate2, fr[0]);
    float l1candidate2 = mix(l0candidate3, l0candidate4, fr[0]);
    float l1candidate3 = mix(l0candidate5, l0candidate6, fr[0]);
    float l1candidate4 = mix(l0candidate7, l0candidate8, fr[0]);
    float l1candidate5 = mix(l0candidate9, l0candidate10, fr[0]);
    float l1candidate6 = mix(l0candidate11, l0candidate12, fr[0]);
    float l1candidate7 = mix(l0candidate13, l0candidate14, fr[0]);
    float l1candidate8 = mix(l0candidate15, l0candidate16, fr[0]);


    float l2candidate1 = mix(l1candidate1, l1candidate2, fr[1]);
    float l2candidate2 = mix(l1candidate3, l1candidate4, fr[1]);
    float l2candidate3 = mix(l1candidate5, l1candidate6, fr[1]);
    float l2candidate4 = mix(l1candidate7, l1candidate8, fr[1]);


    float l3candidate1 = mix(l2candidate1, l2candidate2, fr[2]);
    float l3candidate2 = mix(l2candidate3, l2candidate4, fr[2]);

    float l4candidate1 = mix(l3candidate1, l3candidate2, fr[3]);

    return l4candidate1;
}

// Function 1505
float perlinNoise(float x) {
    float id = floor(x);
    float f = fract(x);
    float u = f;
    return mix(hash1(id), hash1(id + 1.0), u);
}

// Function 1506
float lcnoise(vec3 p)
{
    float f = 0.0;
    ivec3 iv = ivec3(floor(p));
    vec3 fv = fract(p) + 1.0;
    
    int m = 0; // initialization for Knuth's "algorithm H"
    ivec3 di = ivec3(1), ki = ivec3(0);
    ivec4 fi = ivec4(0, 1, 2, 3);
    
    // instead of writing a triply nested loop (!!)
    // generate the indices for the neighbors in Gray order (Knuth's "algorithm H")
    // see section 7.2.1.1 of TAOCP
    
	for (int k = 0; k < 64; k++) // loop through all neighbors
    { 
		 // seeding
        int s = permp(permp(permp(0, iv.z + ki.z), iv.y + ki.y), iv.x + ki.x) + 531;
        
        // L'Ecuyer simple LCG
        s = (2469 * s) % 65521;
        float c = 2.0 * (float(s)/65521.0) - 1.0;
    
        // Mitchell-Netravali cubic, https://doi.org/10.1145/54852.378514
        float r = length(vec3(ki) - fv);
        f += c * (( r < 1.0 ) ? (16.0 + r * r * (21.0 * r - 36.0)) : (( r < 2.0 ) ? (32.0 + r * ((36.0 - 7.0 * r) * r - 60.0)) : 0.0))/18.0;
        
        // updating steps for Knuth's "algorithm H"
        m = fi[0]; fi[0] = 0; ki[2 - m] += di[m];
        if ((ki[2 - m] % 3) == 0) {
            di[m] = -di[m];
            fi[m] = fi[m + 1]; fi[m + 1] = m + 1;
        }
	}
        
    return f;
}

// Function 1507
float pnoise(vec3 uvz, float sz) {
    float zu = floor(uvz[2]/sz)*sz;
    float a = pnoise2(uvz.xy+zu*4.0, sz);
    float b = pnoise2(uvz.xy+(zu+sz)*4.0, sz);
    
    float z = 1.0-fract(abs(uvz[2]/sz));
    z = smoothstep(0.0, 1.0, z);
    
    return a + (b - a)*z;
}

// Function 1508
float trigNoise3D(in vec3 p){

    
    float res = 0., sum = 0.;
    
    // IQ's cheap, texture-lookup noise function. Very efficient, but still 
    // a little too processor intensive for multiple layer usage in a largish 
    // "for loop" setup. Therefore, just one layer is being used here.
    float n = n3D(p*8. + iTime*2.);


    // Two sinusoidal layers. I'm pretty sure you could get rid of one of 
    // the swizzles (I have a feeling the GPU doesn't like them as much), 
    // which I'll try to do later.
    
    vec3 t = sin(p.yzx*3.14159265 + cos(p.zxy*3.14159265+1.57/2.))*0.5 + 0.5;
    p = p*1.5 + (t - 1.5); //  + iTime*0.1
    res += (dot(t, vec3(0.333)));

    t = sin(p.yzx*3.14159265 + cos(p.zxy*3.14159265+1.57/2.))*0.5 + 0.5;
    res += (dot(t, vec3(0.333)))*0.7071;    
	 
	return ((res/1.7071))*0.85 + n*0.15;
}

// Function 1509
float noise12(vec2 pi)
{
	vec3 p = vec3(pi,0);
	vec3 ip=floor(p);
    p-=ip; 
    vec3 s=vec3(7,157,113);
    vec4 h=vec4(0.,s.yz,s.y+s.z)+dot(ip,s);
    p=p*p*(3.-2.*p); 
    h=mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
    h.xy=mix(h.xz,h.yw,p.y);
    return mix(h.x,h.y,p.z); 
}

// Function 1510
float noise(in vec2 uv)
{
    
    return sin(uv.x*1.25)+cos(uv.y/1.25);
}

// Function 1511
float tilableVoronoi( vec3 p, const int octaves, float tile ) {
    float f = 1.;
    float a = 1.;
    float c = 0.;
    float w = 0.;

    if( tile > 0. ) f = tile;

    for( int i=0; i<octaves; i++ ) {
        c += a*voronoi( p * f, f );
        f *= 2.0;
        w += a;
        a *= 0.5;
    }

    return c / w;
}

// Function 1512
float Get3DNoise(vec3 pos) 
{
    float p = floor(pos.z);
    float f = pos.z - p;
    
    const float invNoiseRes = 1.0 / 64.0;
    
    float zStretch = 17.0 * invNoiseRes;
    
    vec2 coord = pos.xy * invNoiseRes + (p * zStretch);
    
    vec2 noise = vec2(texture(iChannel0, coord).x,
					  texture(iChannel0, coord + zStretch).x);
    
    return mix(noise.x, noise.y, f);
}

// Function 1513
vec2 Noisev2v4 (vec4 p)
{
  vec4 i, f, t1, t2;
  i = floor (p);
  f = fract (p);
  f = f * f * (3. - 2. * f);
  t1 = Hashv4f (dot (i.xy, vec2 (1., 57.)));
  t2 = Hashv4f (dot (i.zw, vec2 (1., 57.)));
  return vec2 (mix (mix (t1.x, t1.y, f.x), mix (t1.z, t1.w, f.x), f.y),
               mix (mix (t2.x, t2.y, f.z), mix (t2.z, t2.w, f.z), f.w));
}

// Function 1514
float Noisefv3a (vec3 p)
{
  vec3 i, f;
  i = floor (p);  f = fract (p);
  f *= f * (3. - 2. * f);
  vec4 t1 = Hashv4v3 (i);
  vec4 t2 = Hashv4v3 (i + vec3 (0., 0., 1.));
  return mix (mix (mix (t1.x, t1.y, f.x), mix (t1.z, t1.w, f.x), f.y),
              mix (mix (t2.x, t2.y, f.x), mix (t2.z, t2.w, f.x), f.y), f.z);
}

// Function 1515
float noise(
	in vec3 x
){
	vec3 p = floor(x);
	vec3 f = fract(x);
	f = f*f*(3.0 - 2.0*f);
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 1516
float simplex3d(vec3 p) {
	 vec3 s = floor(p + dot(p, vec3(F3)));
	 vec3 x = p - s + dot(s, vec3(G3));
	 
	 vec3 e = step(vec3(0.0), x - x.yzx);
	 vec3 i1 = e*(1.0 - e.zxy);
	 vec3 i2 = 1.0 - e.zxy*(1.0 - e);
	 	
	 vec3 x1 = x - i1 + G3;
	 vec3 x2 = x - i2 + 2.0*G3;
	 vec4 w, d;
	 
	 vec3 x3 = x - 1.0 + 3.0*G3;
	 
	 w.x = dot(x, x);
	 w.y = dot(x1, x1);
	 w.z = dot(x2, x2);
	 w.w = dot(x3, x3);
	 
	 w = max(0.6 - w, 0.0);
	 
	 /* calculate surflet components */
	 d.x = dot(random3(s), x);
	 d.y = dot(random3(s + i1), x1);
	 d.z = dot(random3(s + i2), x2);
	 d.w = dot(random3(s + 1.0), x3);
	 
	 w *= w;
	 w *= w;
	 d *= w;
	 
	 return dot(d, vec4(52.0));
}

// Function 1517
float ivecnoise(ivec2 coord, int seed)
{
    return hash13(vec3(vec2(coord), seed));
}

// Function 1518
float noise(in vec2 p) {
    const float K1 = (sqrt(3.) - 1.) / 2.;
    const float K2 = (3. - sqrt(3.)) / 6.;
    vec2 i = floor(p + (p.x + p.y) * K1);
    vec2 a = p - i + (i.x + i.y) * K2;
    vec2 o = (a.x > a.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec2 b = a - o + K2;
    vec2 c = a - 1.0 + 2.0 * K2;
    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    vec3 n = h * h * h * h * vec3(dot(a, hash(i + 0.0)), dot(b, hash(i + o)), dot(c, hash(i + 1.0)));
    return dot(n, vec3(50.0));
}

// Function 1519
float noise(vec2 p) {return random(p.x + p.y*NoiseVar.y);}

// Function 1520
float noise( in vec2 p ) 
{
    vec2 i = floor((p)), f = fract((p));
    vec2 u = f*f*(3.-2.*f);
    return mix( mix( dot( hash( i + vec2(0.,0.) ), f - vec2(0.,0.) ), 
                     dot( hash( i + vec2(1.,0.) ), f - vec2(1.,0.) ), u.x),
                mix( dot( hash( i + vec2(0.,1.) ), f - vec2(0.,1.) ), 
                     dot( hash( i + vec2(1.,1.) ), f - vec2(1.,1.) ), u.x), u.y);
}

// Function 1521
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0;

    float res = mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                    mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);

    return res;
}

// Function 1522
vec2 VoronoiCircles (in vec2 coord, float freq, float time, float radiusScale)
{
    const int radius = 1;
    
    vec2 point = coord * freq;
    vec2 ipoint = floor( point );
    vec2 fpoint = fract( point );
    
    vec2 center = fpoint;
    vec2 icenter = vec2(0);
    
    float md = 2147483647.0;
	float mr = 2147483647.0;
    
	// find nearest circle
	for (int y = -radius; y <= radius; ++y)
	for (int x = -radius; x <= radius; ++x)
	{
        vec2 cur = vec2(x, y);
		vec2 c = CenterOfVoronoiCell( vec2(cur), ipoint, time );
		float d = dot( c - fpoint, c - fpoint );

		if ( d < md )
		{
			md = d;
			center = c;
			icenter = cur;
		}
	}
    
	// calc circle radius
	for (int y = -radius; y <= radius; ++y)
	for (int x = -radius; x <= radius; ++x)
	{
        if ( x == 0 && y == 0 )
            continue;
        
        vec2 cur = icenter + vec2(x, y);
		vec2 c = CenterOfVoronoiCell( vec2(cur), ipoint, time );
		float d = dot( c - fpoint, c - fpoint );
		
		if ( d < mr )
			mr = d;
	}
    
    md = sqrt( md );
	mr = sqrt( mr ) * 0.5 * radiusScale;
    
	if ( md < mr )
		return vec2( md / mr, ValueOfVoronoiCell( icenter + ipoint ) );

	return vec2( 0.0, -2.0 );
}

// Function 1523
float noise(vec3 x) {
	vec3 p = floor(x);
	vec3 f = fract(x);
	f = f * f * (3.0 - 2.0 * f);

	float n = p.x + p.y * 157.0 + 113.0 * p.z;
	return mix(
			mix(mix(hash(n + 0.0), hash(n + 1.0), f.x),
					mix(hash(n + 200.0), hash(n + 200.0), f.x), f.y),
			mix(mix(hash(n + 11.0), hash(n + 11.0), f.x),
					mix(hash(n + 27.0), hash(n + 27.0), f.x), f.y), f.z);
}

// Function 1524
float noise(in vec2 p) {
    vec2 pi = floor(p);
    vec2 pf = fract(p);

    float r00 = rand(vec2(pi.x    ,pi.y    ));
    float r10 = rand(vec2(pi.x+1.0,pi.y    ));
    float r01 = rand(vec2(pi.x    ,pi.y+1.0));
    float r11 = rand(vec2(pi.x+1.0,pi.y+1.0));

    vec2 m = pf*pf*(3.0-2.0*pf);
    return mix(mix(r00, r10, m.x), mix(r01, r11, m.x), m.y);
}

// Function 1525
float noiseDist(vec3 p) {
	p = p / NoiseScale;
	return (FBM(p) - NoiseIsoline) * NoiseScale;
}

// Function 1526
float simplex_noise(vec3 p)
{
    const float K1 = 0.333333333;
    const float K2 = 0.166666667;
    
    vec3 i = floor(p + (p.x + p.y + p.z) * K1);
    vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);
        
    vec3 e = step(vec3(0.0), d0 - d0.yzx);
	vec3 i1 = e * (1.0 - e.zxy);
	vec3 i2 = 1.0 - e.zxy * (1.0 - e);
    
    vec3 d1 = d0 - (i1 - 1.0 * K2);
    vec3 d2 = d0 - (i2 - 2.0 * K2);
    vec3 d3 = d0 - (1.0 - 3.0 * K2);
    
    vec4 h = max(0.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
    vec4 n = h * h * h * h * vec4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0)));
    
    return dot(vec4(31.316), n);
}

// Function 1527
float noise( vec3 p ) 
{
	vec3 pi = floor( p );
	vec3 pf = fract( p );

	
	float n = pi.x + 59.0 * pi.y + 256.0 * pi.z;

	pf.x = pf.x * pf.x * (3.0 - 2.0 * pf.x);
	pf.y = pf.y * pf.y * (3.0 - 2.0 * pf.y);
	pf.z = sin( pf.z );

	float v1 = 	
		mix(
			mix( hash( n ), hash( n + 1.0 ), pf.x ),
			mix( hash( n + 59.0 ), hash( n + 1.0 + 59.0 ), pf.x ),
			pf.y );
	
	float v2 = 	
		mix(
		mix( hash( n + 256.0 ), hash( n + 1.0 + 256.0 ), pf.x ),
			mix( hash( n + 59.0 + 256.0 ), hash( n + 1.0 + 59.0 + 256.0 ), pf.x ),
			pf.y );

	return mix( v1, v2, pf.z );
}

// Function 1528
float simplex_noise(vec3 p)
{
    const float K1 = 0.333333333;
    const float K2 = 0.166666667;
    
    vec3 i = floor(p + (p.x + p.y + p.z) * K1);
    vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);
    
    // thx nikita: https://www.shadertoy.com/view/XsX3zB
    vec3 e = step(vec3(0.0), d0 - d0.yzx);
	vec3 i1 = e * (1.0 - e.zxy);
	vec3 i2 = 1.0 - e.zxy * (1.0 - e);
    
    vec3 d1 = d0 - (i1 - 1.0 * K2);
    vec3 d2 = d0 - (i2 - 2.0 * K2);
    vec3 d3 = d0 - (1.0 - 3.0 * K2);
    
    vec4 h = max(0.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
    vec4 n = h * h * h * h * vec4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0)));
    
    return dot(vec4(31.316), n);
}

// Function 1529
float smooth_noise(in vec3 position)
{
        vec3 integer = floor(position);
        vec3 fractional = fract(position);

        fractional = fractional * fractional * (3. - 2. * fractional);

        float seed = integer.x + integer.y * 57. + 113. * integer.z;
        return mix(mix(
                        mix(grain(seed), grain(seed + 1.),
                                fractional.x),
                        mix(grain(seed + 57.), grain(seed + 58.),
                                fractional.x),
                        fractional.y),
                mix(mix(grain(seed + 113.), grain(seed + 114.),
                                fractional.x),
                        mix(grain(seed + 170.), grain(seed + 171.),
                                fractional.x),
                        fractional.y),
                    fractional.z);
}

// Function 1530
float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,
                        0.366025403784439,
                        -0.577350269189626,
                        0.024390243902439);
    
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    
    vec2 i1 = vec2(0.0);
    i1 = (x0.x > x0.y)? vec2(1.0, 0.0):vec2(0.0, 1.0);
    vec2 x1 = x0.xy + C.xx - i1;
    vec2 x2 = x0.xy + C.zz;
    
    i = mod289(i);
    vec3 p = permute(
            permute( i.y + vec3(0.0, i1.y, 1.0))
                + i.x + vec3(0.0, i1.x, 1.0 ));
    
    vec3 m = max(0.5 - vec3(
                        dot(x0,x0),
                        dot(x1,x1),
                        dot(x2,x2)
                        ), 0.0);
    
    m = m*m ;
    m = m*m ;
    
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0+h*h);
    
    vec3 g = vec3(0.0);
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * vec2(x1.x,x2.x) + h.yz * vec2(x1.y,x2.y);
    
    return 130.0 * dot(m, g);
}

// Function 1531
float snoise(vec2 v)
  {
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
		+ i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// Function 1532
vec3 voronoi( in vec2 x, out vec2 gradient, out vec2 cell)
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mg, mr, mr_2;

    float md = 10.0;
    
    //we need to extend the search domain to two cells for voronoi generation (as opposed to one).
    //this is due to the fact that the manhattan metric is not a straight distance
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( iTime * .1 + 6.2831*o );
        #endif	
        vec2 r = g + o - f;
        
        //manhattan distance (L1 Norm)
        float d = abs(r.x) +abs(r.y);

        if( d<md )
        {
            md = d;
            mr = r;
            mg = g;
            cell = o;
        }
    }
	//cell = mg;
    //----------------------------------
    // second pass: distance to borders
    //---------------------------------
    
    md = 10.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2;i<=2; i++ )
    {
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		#ifdef ANIMATE
        o = 0.5 + 0.5*sin( iTime * .1 + 6.2831*o );
        #endif	
        vec2 r = g + o - f;
		
        //call our custom edge distance function
        if( dot(mr-r,mr-r)>1e-4 ){
            vec2 grad = vec2(0.);
            float man_dist = man_dist(f - mr, f - r, f, grad);
        	md = min(md, man_dist);
            gradient = md == man_dist ? grad : gradient;
        } 
            
    }
	
    return vec3( md, mr );
}

// Function 1533
float noise_3(in vec3 p) {
        vec3 i = floor(p);
        vec3 f = fract(p);	
        vec3 u = f*f*(3.0-2.0*f);

        vec2 ii = i.xy + i.z * vec2(5.0);
        float a = hash12( ii + vec2(0.0,0.0) );
        float b = hash12( ii + vec2(1.0,0.0) );    
        float c = hash12( ii + vec2(0.0,1.0) );
        float d = hash12( ii + vec2(1.0,1.0) ); 
        float v1 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);

        ii += vec2(5.0);
        a = hash12( ii + vec2(0.0,0.0) );
        b = hash12( ii + vec2(1.0,0.0) );    
        c = hash12( ii + vec2(0.0,1.0) );
        d = hash12( ii + vec2(1.0,1.0) );
        float v2 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);

        return max(mix(v1,v2,u.z),0.0);
    }

// Function 1534
vec4 noise(vec3 p){float m = mod(p.z,1.0);float s = p.z-m; float sprev = s-1.0;if (mod(s,2.0)==1.0) { s--; sprev++; m = 1.0-m; };return mix(texture(iChannel0,p.xy/iChannelResolution[0].xy + noise(sprev).yz*21.421),texture(iChannel0,p.xy/iChannelResolution[0].xy + noise(s).yz*14.751),m);}

// Function 1535
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
}

// Function 1536
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    float a = textureLod( iChannel0, x.xy/256.0 + (p.z+0.0)*120.7123, 0.0 ).x;
    float b = textureLod( iChannel0, x.xy/256.0 + (p.z+1.0)*120.7123, 0.0 ).x;
	return mix( a, b, f.z );
}

// Function 1537
vec2 noise2(float time)
{
    return vec2(clamp(fract(sin(time*8.3)*1e6)-.4, 0.0, 0.09)); 
}

// Function 1538
float SpiralNoise3D(vec3 p)
{
    float n = 0.0;
    float iter = 1.0;
    for (int i = 0; i < 5; i++)
    {
        n += (sin(p.y*iter) + cos(p.x*iter)) / iter;
        p.xz += vec2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        iter *= 1.33733;
    }
    return n;
}

// Function 1539
float perlin (vec2 uv) {
    vec2 id = floor(uv);
    vec2 gv = fract(uv);

    // Four corners in 2D of a tile
    float a = hash12(id);
    float b = hash12(id + vec2(1.0, 0.0));
    float c = hash12(id + vec2(0.0, 1.0));
    float d = hash12(id + vec2(1.0, 1.0));

    vec2 u = gv * gv * (3.0 - 2.0 * gv);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Function 1540
VoronoiData voronoi( in vec2 x )
{
    vec2 n = floor(x);
    vec2 f = fract(x);

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mr;
    vec2 mi;

    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mi = n + g;
        }
    }

    //----------------------------------
    // second pass: distance to borders,
    // visits only neighbouring cells
    //----------------------------------
    md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2( n + g );
		vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) {// skip the same cell
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
        }
    }

    return VoronoiData( md, mr, mi );
}

// Function 1541
vec3 noised( in vec2 p )
{
    vec2 i = floor( p ),
         f = fract( p );

#if 1 // quintic interpolation
    vec2 u = f*f*f*(f*(f*6.-15.)+10.),
        du = 30.*f*f*(f*(f-2.)+1.);
#else // cubic interpolation
    vec2 u = f*f*(3.-2.*f),
        du = 6.*f*(1.-f);
#endif    
    
    vec2 ga = hash( i + vec2(0,0) ),
         gb = hash( i + vec2(1,0) ),
         gc = hash( i + vec2(0,1) ),
         gd = hash( i + vec2(1,1) );
    
    float va = dot( ga, f - vec2(0,0) ),
          vb = dot( gb, f - vec2(1,0) ),
          vc = dot( gc, f - vec2(0,1) ),
          vd = dot( gd, f - vec2(1,1) );

    return vec3( va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd),   // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +  // derivatives
                 du * (u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va));
}

// Function 1542
float noise(in vec3 p){
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);

    return mix( mix( mix( dot( hash( i + vec3(0.0,0.0,0.0) ), f - vec3(0.0,0.0,0.0) ), 
                          dot( hash( i + vec3(1.0,0.0,0.0) ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,0.0) ), f - vec3(0.0,1.0,0.0) ), 
                          dot( hash( i + vec3(1.0,1.0,0.0) ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( hash( i + vec3(0.0,0.0,1.0) ), f - vec3(0.0,0.0,1.0) ), 
                          dot( hash( i + vec3(1.0,0.0,1.0) ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( hash( i + vec3(0.0,1.0,1.0) ), f - vec3(0.0,1.0,1.0) ), 
                          dot( hash( i + vec3(1.0,1.0,1.0) ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// Function 1543
float snoise(in vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

// Function 1544
vec3 noise(vec2 p, float lod){return texture(iChannel0,p/iChannelResolution[0].xy,lod).xyz;}

// Function 1545
float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( random2(i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ),
                     dot( random2(i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( random2(i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ),
                     dot( random2(i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1546
float noise(in float x)
{
    float i = floor(x);
    float f = fract(x);
    //f = f*f*(3. - 2.*f);
    return mix(rand(i), rand(i+1.), f);
}

// Function 1547
float Noise(vec2 uv)
{
    vec2 corner = floor(uv);
	float c00 = N2(corner + vec2(0.0, 0.0));
	float c01 = N2(corner + vec2(0.0, 1.0));
	float c11 = N2(corner + vec2(1.0, 1.0));
	float c10 = N2(corner + vec2(1.0, 0.0));
    
    vec2 diff = fract(uv);
    
    diff = diff * diff * (vec2(3) - vec2(2) * diff);
    
    return mix(mix(c00, c10, diff.x), mix(c01, c11, diff.x), diff.y);
}

// Function 1548
float noise(vec2 p) {
	return fract(sin(dot(p.xy ,vec2(12.9898,78.233))) * 456367.5453);
}

// Function 1549
vec2 sinNoise(vec2 p, vec4 s)
{
    vec4 ps = p.xyxy + s * iTime;
    vec2 p1 = ps.xy;
    vec2 p2 = ps.xy * rot2 * 0.4;
    vec2 p3 = ps.zw * rot6 * 0.7;
    vec2 p4 = ps.zw * rot10 * 1.5;
	vec4 s1 = sin(vec4(p1.x, p1.y, p2.x, p2.y));
    vec4 s2 = sin(vec4(p3.x, p3.y, p4.x, p4.y));
    
    return ((s1.xy + s1.zw + s2.xy + s2.zw)*0.25);
}

// Function 1550
float tetraNoise(in vec3 p)
{
    // Skewing the cubic grid, then determining the first vertice and fractional position.
    vec3 i = floor(p + dot(p, vec3(.333333)) );  p -= i - dot(i, vec3(.166666)) ;
    
    // Breaking the skewed cube into tetrahedra with partitioning planes, then determining which side of the 
    // intersecting planes the skewed point is on. Ie: Determining which tetrahedron the point is in.
    vec3 i1 = step(p.yzx, p), i2 = max(i1, 1. - i1.zxy); i1 = min(i1, 1. - i1.zxy);    
    
    // Using the above to calculate the other three vertices -- Now we have all four tetrahedral vertices.
    // Technically, these are the vectors from "p" to the vertices, but you know what I mean. :)
    vec3 p1 = p - i1 + .166666, p2 = p - i2 + .333333, p3 = p - .5;
  

    // 3D simplex falloff - based on the squared distance from the fractional position "p" within the 
    // tetrahedron to the four vertice points of the tetrahedron. 
    vec4 v = max(.5 - vec4(dot(p, p), dot(p1, p1), dot(p2, p2), dot(p3, p3)), 0.);
    
    // Dotting the fractional position with a random vector, generated for each corner, in order to determine 
    // the weighted contribution distribution... Kind of. Just for the record, you can do a non-gradient, value 
    // version that works almost as well.
    vec4 d = vec4(dot(p, hash33(i)), dot(p1, hash33(i + i1)), dot(p2, hash33(i + i2)), dot(p3, hash33(i + 1.)));
     
     
    // Simplex noise... Not really, but close enough. :)
    return clamp(dot(d, v*v*v*8.)*1.732 + .5, 0., 1.); // Not sure if clamping is necessary. Might be overkill.

}

// Function 1551
vec4 Noise401( vec4 x ) { return fract(sin(x)*5346.1764); }

// Function 1552
float fractalNoise(vec2 p) {

    float x = 0.;
    x += smoothNoise(p      );
    x += smoothNoise(p * 2. ) / 2.;
    x += smoothNoise(p * 4. ) / 4.;
    x += smoothNoise(p * 8. ) / 8.;
    x += smoothNoise(p * 16.) / 16.;
    x /= 1. + 1./2. + 1./4. + 1./8. + 1./16.;
    return x;
            
}

// Function 1553
float FractalVoronoi(vec3 p)
{
	float n = 0.0;
	float f = 0.5, a = 0.5;
	mat2 m = mat2(0.8, 0.6, -0.6, 0.8);
	for (int i = 0; i < FBM_ITERATIONS; i++) {
		n += Voronoi(p * f) * a;
		f *= FBM_FREQUENCY_GAIN;
		a *= FBM_AMPLITUDE_GAIN;
		p.xy = m * p.xy;
	}
	return n;
}

// Function 1554
float CloudNoise( in vec2 uv )
{
	vec2 iuv = floor(uv);
	vec2 fuv = fract(uv);
	uv = (iuv + fuv*fuv*(3.0-2.0*fuv)) / 1024.0;
    return texture(iChannel0, uv, -100.0 ).z;
}

// Function 1555
float billowedNoise( vec3 p )
{
   return abs( noise( p ) );
}

// Function 1556
vec2 Noise1(vec2 x)
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    vec2 res = mix(mix( hash22(p),          hash22(p + add.xy),f.x),
                   mix( hash22(p + add.yx), hash22(p + add.xx),f.x),f.y);
    return res;
}

// Function 1557
float noise(vec3 x)
{
    //x.x = mod(x.x, 0.4);
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0 - 2.0*f);
	
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix(hash(n + 0.0), hash(n + 1.0),f.x),
                   mix(hash(n + 157.0), hash(n + 158.0),f.x),f.y),
               mix(mix(hash(n + 113.0), hash(n + 114.0),f.x),
                   mix(hash(n + 270.0), hash(n + 271.0),f.x),f.y),f.z);
}

// Function 1558
float triNoise3d(in vec3 p, in float spd) {
    float z = 1.45;
    float rz = 0.;
    vec3 bp = p;
    for (float i = 0.; i < 4.; i++) {
        vec3 dg = tri3(bp);
        p += (dg + time * spd + 10.1);
        bp *= 1.65;
        z *= 1.5;
        p *= .9;
        p.xz *= m2;

        rz += (tri2(p.z + tri2(p.x + tri2(p.y)))) / z;
        bp += 0.9;
    }
    return rz;
}

// Function 1559
float Noise( float x )
{
    return fract( sin( 123523.9898 * x ) * 43758.5453 );
}

// Function 1560
float Noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    float res = mix(mix( Hash12(p),          Hash12(p + add.xy),f.x),
                    mix( Hash12(p + add.yx), Hash12(p + add.xx),f.x),f.y);
    return res;
}

// Function 1561
float noised(in vec2 p){
    vec2 i = floor( p );
    vec2 f = fract( p );

    vec2 u = f*f*(3.0-2.0*f);
    vec2 du = 6.0*f*(1.0-f);
    
    vec2 ga = hash( i + vec2(0.0,0.0) );
    vec2 gb = hash( i + vec2(1.0,0.0) );
    vec2 gc = hash( i + vec2(0.0,1.0) );
    vec2 gd = hash( i + vec2(1.0,1.0) );
    
    float va = dot( ga, f - vec2(0.0,0.0) );
    float vb = dot( gb, f - vec2(1.0,0.0) );
    float vc = dot( gc, f - vec2(0.0,1.0) );
    float vd = dot( gd, f - vec2(1.0,1.0) );

    return (va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd));
}

// Function 1562
float Voronoi(vec2 uv)
{
	vec2 i = floor(uv);
	vec2 f = fract(uv);

	float d = 64.;
	for(float y = -1.; y <= 1.; ++y)
	{
		for (int x = -1; x <= 1; ++x)
		{
			vec2 b = vec2(x, y);
            float vNoise = fract(sin(
                dot(i + b, vec2(101.9364, 96.45418))
                ) * 100000.0);
			vec2 c = b + vNoise - f;
			d = min(d, dot(c, c));
		}
	}
	return sqrt(d);
}

// Function 1563
float triNoise3d(in vec3 p)
{
    p.y *= 0.57;
    float z=1.5;
	float rz = 0.;
    vec3 bp = p;
	for (float i=0.; i<2.; i++ )
	{
        vec3 dg = tri3(bp*.5);
        p += (dg+0.1);

        bp *= 2.2;
		z *= 1.4;
		p *= 1.2;
        p.xz*= m2;
        
        rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
        bp += 0.9;
	}
	return rz;
}

// Function 1564
float snoise(in float x) {
    return mix(Hash11(floor(x)), Hash11(ceil(x)), smoothstep(0.0, 1.0, fract(x)));
}

// Function 1565
VoronoiData voronoi( in vec2 x )
{
#if 1
    // slower, but better handles big numbers
    vec2 n = floor(x);
    vec2 f = fract(x);
    vec2 h = step(.5,f) - 2.;
    n += h; f -= h;
#else
    vec2 n = floor(x - 1.5);
    vec2 f = x - n;
#endif

    //----------------------------------
    // first pass: regular voronoi
    //----------------------------------
	vec2 mr;
    vec2 mi;

    float md = 8.0;
    for( int j=0; j<=3; j++ )
    for( int i=0; i<=3; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;
        float d = dot(r,r);

        if( d<md )
        {
            md = d;
            mr = r;
            mi = n + g;
        }
    }

    //----------------------------------
    // second pass: distance to borders
    //----------------------------------
    md = 8.0;
    for( int j=0; j<=3; j++ )
    for( int i=0; i<=3; i++ )
    {
        vec2 g = vec2(float(i),float(j));
        vec2 o = hash2( n + g );
        vec2 r = g + o - f;

        if( dot(mr-r,mr-r)>EPSILON ) // skip the same cell
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }

    return VoronoiData( md, mr, mi );
}

// Function 1566
float noise(vec2 pos)
{
	return fract( sin( dot(pos*0.001 ,vec2(24.12357, 36.789) ) ) * 12345.123);	
}

// Function 1567
float triangleNoise(const vec2 n) {
    // triangle noise, in [-0.5..1.5[ range
    vec2 p = fract(n * vec2(5.3987, 5.4421));
    p += dot(p.yx, p.xy + vec2(21.5351, 14.3137));

    float xy = p.x * p.y;
    // compute in [0..2[ and remap to [-1.0..1.0[
    float noise = (fract(xy * 95.4307) + fract(xy * 75.04961) - 1.0);
    //noise = sign(noise) * (1.0 - sqrt(1.0 - abs(noise)));
	return noise;
}

// Function 1568
float iqNoise(vec3 x) {
    vec3 p = floor(x );
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
   // f = f*f*f*(f*(f*6.0-15.0)+10.0);
    float c1 = 883.0;
    float c2 = 971.0;
    float c3 = 1.0;//127.0;
    float n = p.x*c3 + p.y*c2+ c1*p.z;
    return mix(
        mix(
            mix(HashMe(n+0.0),HashMe(n+ c3),f.x),
            mix(HashMe(n+c2),HashMe(n+c2+ c3),f.x),
            f.y),
        mix(
            mix(HashMe(n+c1),HashMe(n+c1+ c3),f.x),
            mix(HashMe(n+c1+c2),HashMe(n+c1+c2+ c3),f.x),
            f.y),
        f.z);
}

// Function 1569
float noise( in vec2 p )
{
    p*=noiseIntensity;
    vec2 i = floor( p );
    vec2 f = fract( p );
	vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( random( i + vec2(0.0,0.0) ), 
                     random( i + vec2(1.0,0.0) ), u.x),
                mix( random( i + vec2(0.0,1.0) ), 
                     random( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1570
float noise( in vec2 p )
{
    ivec2 i = ivec2(floor( p ));
     vec2 f =       fract( p );
	vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( dot( grad( i+ivec2(0,0) ), f-vec2(0.0,0.0) ), 
                     dot( grad( i+ivec2(1,0) ), f-vec2(1.0,0.0) ), u.x),
                mix( dot( grad( i+ivec2(0,1) ), f-vec2(0.0,1.0) ), 
                     dot( grad( i+ivec2(1,1) ), f-vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1571
float Voronoi(in vec3 p)
{
	float d = 1.0e10;
	for (int zo = -1; zo <= 1; zo++)
	{
		for (int xo = -1; xo <= 1; xo++)
		{
			for (int yo = -1; yo <= 1; yo++)
			{
				vec3 tp = floor(p) + vec3(xo, yo, zo);
				d = min(d, length(p - tp - Noise(p)));
			}
		}
	}
	return .72 - d*d*d;
}

// Function 1572
float snoise(vec3 v)
    {
        const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
        const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

        // First corner
        vec3 i  = floor(v + dot(v, C.yyy) );
        vec3 x0 =   v - i + dot(i, C.xxx) ;

        // Other corners
        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min( g.xyz, l.zxy );
        vec3 i2 = max( g.xyz, l.zxy );

        //   x0 = x0 - 0.0 + 0.0 * C.xxx;
        //   x1 = x0 - i1  + 1.0 * C.xxx;
        //   x2 = x0 - i2  + 2.0 * C.xxx;
        //   x3 = x0 - 1.0 + 3.0 * C.xxx;
        vec3 x1 = x0 - i1 + C.xxx;
        vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
        vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

        // Permutations
        i = mod289(i);
        vec4 p = permute( permute( permute(
        i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
        + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
        + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

        // Gradients: 7x7 points over a square, mapped onto an octahedron.
        // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
        float n_ = 0.142857142857; // 1.0/7.0
        vec3  ns = n_ * D.wyz - D.xzx;

        vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

        vec4 x = x_ *ns.x + ns.yyyy;
        vec4 y = y_ *ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);

        vec4 b0 = vec4( x.xy, y.xy );
        vec4 b1 = vec4( x.zw, y.zw );

        //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
        //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
        vec4 s0 = floor(b0)*2.0 + 1.0;
        vec4 s1 = floor(b1)*2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));

        vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
        vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

        vec3 p0 = vec3(a0.xy,h.x);
        vec3 p1 = vec3(a0.zw,h.y);
        vec3 p2 = vec3(a1.xy,h.z);
        vec3 p3 = vec3(a1.zw,h.w);

        //Normalise gradients
        vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;

        // Mix final noise value
        vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
        m = m * m;
        return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
        dot(p2,x2), dot(p3,x3) ) );
    }

// Function 1573
float noiseText(in vec3 x) {
    vec3 p = floor(x), f = fract(x);
	f = f*f*(3.-f-f);
	vec2 uv = (p.xy+vec2(37.,17.)*p.z) + f.xy,
	     rg = textureLod( iChannel0, (uv+.5)/256., -100.).yx;
	return mix(rg.x, rg.y, f.z);
}

// Function 1574
vec4 atm_cloudnoise2_d( vec4 r, float scale, bool lowfreq )
{
    float invscale = 1. / scale;
    float y = atm_cloudnoise2( invscale * r, lowfreq );
    return vec4( invscale * ( atm_cloudnoise2_offs( invscale * r, lowfreq ) - y ), y );
}

// Function 1575
float noise3D(vec3 p)
{
	return fract(sin(dot(p ,vec3(12.,78.333,126.))) * 43758.);
}

// Function 1576
float RustNoise3D(vec3 p)
{
    float n = 0.0;
    float iter = 1.0;
    float pn = noise(p*0.125);
    pn += noise(p*0.25)*0.5;
    pn += noise(p*0.5)*0.25;
    pn += noise(p*1.0)*0.125;
    for (int i = 0; i < 7; i++)
    {
        //n += (sin(p.y*iter) + cos(p.x*iter)) / iter;
        float wave = saturate(cos(p.y*0.25 + pn) - 0.998);
       // wave *= noise(p * 0.125)*1016.0;
        n += wave;
        p.xy += vec2(p.y, -p.x) * nudge;
        p.xy *= normalizer;
        p.xz += vec2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        iter *= 1.4733;
    }
    return n*500.0;
}

// Function 1577
float VoroNoise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    float k = 1.0 + 63.0*pow(1.0-v,4.0);
    float va = 0.0;
    float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2  g = vec2( float(i), float(j) );
        vec3  o = hash3( p + g )*vec3(u,u,1.0);
        vec2  r = g - f + o.xy;
        float d = dot(r,r);
        float w = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
        va += w*o.z;
        wt += w;
    }

    return va/wt;
}

// Function 1578
vec4 noised( in vec3 x )
{
    // grid
    vec3 i = floor(x);
    vec3 w = fract(x);
    
    #if 1
    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    vec3 du = 30.0*w*w*(w*(w-2.0)+1.0);
    #else
    // cubic interpolant
    vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
    #endif    
    
    // gradients
    vec3 ga = hash( i+vec3(0.0,0.0,0.0) );
    vec3 gb = hash( i+vec3(1.0,0.0,0.0) );
    vec3 gc = hash( i+vec3(0.0,1.0,0.0) );
    vec3 gd = hash( i+vec3(1.0,1.0,0.0) );
    vec3 ge = hash( i+vec3(0.0,0.0,1.0) );
	vec3 gf = hash( i+vec3(1.0,0.0,1.0) );
    vec3 gg = hash( i+vec3(0.0,1.0,1.0) );
    vec3 gh = hash( i+vec3(1.0,1.0,1.0) );
    
    // projections
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
	
    // interpolations
    return vec4( va + u.x*(vb-va) + u.y*(vc-va) + u.z*(ve-va) + u.x*u.y*(va-vb-vc+vd) + u.y*u.z*(va-vc-ve+vg) + u.z*u.x*(va-vb-ve+vf) + (-va+vb+vc-vd+ve-vf-vg+vh)*u.x*u.y*u.z,    // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.z*(ge-ga) + u.x*u.y*(ga-gb-gc+gd) + u.y*u.z*(ga-gc-ge+gg) + u.z*u.x*(ga-gb-ge+gf) + (-ga+gb+gc-gd+ge-gf-gg+gh)*u.x*u.y*u.z +   // derivatives
                 du * (vec3(vb,vc,ve) - va + u.yzx*vec3(va-vb-vc+vd,va-vc-ve+vg,va-vb-ve+vf) + u.zxy*vec3(va-vb-ve+vf,va-vb-vc+vd,va-vc-ve+vg) + u.yzx*u.zxy*(-va+vb+vc-vd+ve-vf-vg+vh) ));
}

// Function 1579
vec4 trilinearNoiseDerivative(vec3 p)
{
    p /= DOMAIN_SCALING;
    const float TEXTURE_RES = 256.0; //Noise texture resolution
    vec3 pixCoord = floor(p);//Pixel coord, integer [0,1,2,3...256...]
    vec2 layer_translation = -pixCoord.z*vec2(37.0,17.0)/TEXTURE_RES; //noise volume stacking trick : g layer = r layer shifted by (37x17 pixels -> this is no keypad smashing, but the actual translation embedded in the noise texture).
    
    vec2 c1 = texture(iChannel0,layer_translation+(pixCoord.xy+vec2(0,0)+0.5)/TEXTURE_RES,-100.0).xy;
    vec2 c2 = texture(iChannel0,layer_translation+(pixCoord.xy+vec2(1,0)+0.5)/TEXTURE_RES,-100.0).xy; //+x
    vec2 c3 = texture(iChannel0,layer_translation+(pixCoord.xy+vec2(0,1)+0.5)/TEXTURE_RES,-100.0).xy; //+z
    vec2 c4 = texture(iChannel0,layer_translation+(pixCoord.xy+vec2(1,1)+0.5)/TEXTURE_RES,-100.0).xy; //+x+z
    
    vec3 x = p-pixCoord;     //Pixel interpolation position, linear range [0-1] (fractional part)
    vec3 t = (3.0 - 2.0 * x) * x * x;
    
    //Lower quad corners
    float a = c1.x; //(x+0,y+0,z+0)
    float b = c2.x; //(x+1,y+0,z+0)
    float c = c3.x; //(x+0,y+1,z+0)
    float d = c4.x; //(x+1,y+1,z+0)
    
    //Upper quad corners
    float e = c1.y; //(x+0,y+0,z+1)
    float f = c2.y; //(x+1,y+0,z+1)
    float g = c3.y; //(x+0,y+1,z+1)
    float h = c4.y; //(x+1,y+1,z+1)
    
    //Trilinear noise interpolation : (1-t)*v1+(t)*v2, repeated along the 3 axis of the interpolation cube.
    float za = ((a+(b-a)*t.x)*(1.-t.y)
               +(c+(d-c)*t.x)*(   t.y));
    float zb = ((e+(f-e)*t.x)*(1.-t.y)
               +(g+(h-g)*t.x)*(   t.y));
    float value = (1.-t.z)*za+t.z*zb;
    
    //Derivative scaling
    float sx =  ((b-a)+t.y*(a-b-c+d))*(1.-t.z)
               +((f-e)+t.y*(e-f-g+h))*(   t.z);
    float sy =  ((c-a)+t.x*(a-b-c+d))*(1.-t.z)
               +((g-e)+t.x*(e-f-g+h))*(   t.z);
    float sz =  zb-za;
    
    //Ease-in ease-out derivative : (3x^2-2x^3)' = 6x-6x^2
    vec3 dxyz = 6.*x*(1.-x);
    
    return vec4(value*DOMAIN_SCALING,
	            vec3(dxyz.x*sx,
                     dxyz.y*sy,
                     dxyz.z*sz));
}

// Function 1580
float noise2f( in vec2 p )
{
	ivec2 ip = ivec2(floor(p));
    vec2  fp = fract(p);
	vec2 u = fp*fp*(3.0-2.0*fp);

    int n = ip.x + ip.y*113;

	float res = mix(mix(fhash(n+(0+113*0)),
                        fhash(n+(1+113*0)),u.x),
                    mix(fhash(n+(0+113*1)),
                        fhash(n+(1+113*1)),u.x),u.y);

    return 1.0 - res*(1.0/1073741824.0);
}

// Function 1581
float nestedNoise(vec2 p) {
    
    float x = movingNoise(p);
    float y = movingNoise(p + 100.);
    return movingNoise(p + vec2(x, y));
    
}

// Function 1582
float noise1d(float p){
	float fl = floor(p);
	float fc = fract(p);
	return mix(random1d(fl), random1d(fl + 1.0), fc);
}

// Function 1583
float atm_cloudnoise1( vec4 r, bool lowfreq )
{
    float lod = log2( r.w );
	float y = ( textureLod( iChannel3, r.xyz / 32., lod ).x + 2. * textureLod( iChannel3, r.xyz / 64., lod - 1. ).x ) / 3.;
    if( !lowfreq )
		y = 4. / 5. * y + ( textureLod( iChannel3, r.xyz / 8., lod + 2. ).x + 2. * textureLod( iChannel3, r.xyz / 16., lod + 1. ).x ) / 15.;
	return y;
}

// Function 1584
float noise(
    vec2 uv, 
    float s1, 
    float s2, 
    float t1, 
    float t2, 
    float c1) 
{
	return clamp(hash33(vec3(uv.xy * s1, t1)).x +
		hash33(vec3(uv.xy * s2, t2)).y, c1, 1.);
}

// Function 1585
float voronoi2D(vec2 uv) {
    vec2 fl = floor(uv);
    vec2 fr = fract(uv);
    float res = 1.0;
    for( int j=-1; j<=1; j++ ) {
        for( int i=-1; i<=1; i++ ) {
            vec2 p = vec2(i, j);
            float h = hash2D(fl+p);
            vec2 vp = p-fr+h;
            float d = dot(vp, vp);
            
            res +=1.0/pow(d, 8.0);
        }
    }
    return pow( 1.0/res, 1.0/16.0 );
}

// Function 1586
vec4 Dither_TriangleNoise(vec4 rgba, float levels) {
    // Gjøl 2016, "Banding in Games: A Noisy Rant"
#if RGB_TRIANGLE_NOISE == 1
    vec3 noise = vec3(
        triangleNoise(gl_FragCoord.xy / iResolution.xy         ) / (levels - 1.0),
        triangleNoise(gl_FragCoord.xy / iResolution.xy + 0.1337) / (levels - 1.0),
        triangleNoise(gl_FragCoord.xy / iResolution.xy + 0.3141) / (levels - 1.0)
    );
#else
    vec3 noise = vec3(triangleNoise(gl_FragCoord.xy / iResolution.xy) / (levels - 1.0));
#endif
    return vec4(rgba.rgb + noise, rgba.a);
}

// Function 1587
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( hash2d( i + vec2(0.0,0.0) ), 
                     hash2d( i + vec2(1.0,0.0) ), u.x),
                mix( hash2d( i + vec2(0.0,1.0) ), 
                     hash2d( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1588
float perlinNoise3D(int gridWidth, int gridHeight, int gridDepth, vec3 position) {
	
	// Takes input position in the interval [0, 1] in all axes, outputs noise in the range [0, 1].
	vec3 gridDimensions = vec3(gridWidth, gridHeight, gridDepth);
	position *= gridDimensions;
	
	// Get corners,
	vec3 lowerBoundPosition = floor(position);
	
	// Calculate gradient values!
	float gradientValues[8];
	for (float z=0.0; z<2.0; z++) {
		for (float y=0.0; y<2.0; y++) {
			for (float x=0.0; x<2.0; x++) {
				vec3 currentPointPosition = lowerBoundPosition + vec3(x, y, z);
				
				vec3 displacementVector = (currentPointPosition - position);
				vec3 gradientVector = gradient(mod(currentPointPosition.x + permutation(mod(currentPointPosition.y + permutation(currentPointPosition.z), 256.0)), 256.0));
				
				gradientValues[int((z*4.0) + (y*2.0) + x)] = (0.0 + dot(gradientVector, displacementVector)) * 2.0;
			}
		}
	}
	
	
	
	// Interpolate using Hermit,
	vec3 interpolationRatio = position - lowerBoundPosition;
	float finalNoise = 0.0;
	finalNoise += gradientValues[7] * hermit3D(interpolationRatio);
	finalNoise += gradientValues[6] * hermit3D(vec3(1.0 - interpolationRatio.x, interpolationRatio.y, interpolationRatio.z));
	finalNoise += gradientValues[5] * hermit3D(vec3( interpolationRatio.x, 1.0 - interpolationRatio.y, interpolationRatio.z));
	finalNoise += gradientValues[4] * hermit3D(vec3(1.0 - interpolationRatio.x, 1.0 - interpolationRatio.y, interpolationRatio.z));
	
	finalNoise += gradientValues[3] * hermit3D(vec3( interpolationRatio.x, interpolationRatio.y, 1.0 - interpolationRatio.z));
	finalNoise += gradientValues[2] * hermit3D(vec3(1.0 - interpolationRatio.x, interpolationRatio.y, 1.0 - interpolationRatio.z));
	finalNoise += gradientValues[1] * hermit3D(vec3( interpolationRatio.x, 1.0 - interpolationRatio.y, 1.0 - interpolationRatio.z));
	finalNoise += gradientValues[0] * hermit3D(vec3(1.0 - interpolationRatio.x, 1.0 - interpolationRatio.y, 1.0 - interpolationRatio.z));
	
	
	
	return finalNoise;
}

// Function 1589
vec4 bccNoiseClassic(vec3 X) {
    
    // Rotate around the main diagonal. Not a skew transform.
    vec4 result = bccNoiseBase(dot(X, vec3(2.0/3.0)) - X);
    return vec4(dot(result.xyz, vec3(2.0/3.0)) - result.xyz, result.w);
}

// Function 1590
float noise3D(in vec3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
	const vec3 s = vec3(113, 157, 1);
	
	vec3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride vector for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    
	p -= ip; // Cell's fractional component.
	
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: fract(sin(x)*largeNumber).
    h = mix(fract(sin(h)*43758.5453), fract(sin(h + s.x)*43758.5453), p.x);
	
    // Interpolating along Y.
    h.xy = mix(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    float n = mix(h.x, h.y, p.z); // Range: [0, 1].
	
    return n;//abs(n - .5)*2.;
}

// Function 1591
void blueNoise(vec2 p, out vec4 blue1, out vec4 blue2)
{
    blue1 = texture(iChannel1, p / vec2(1024.0));
    blue2 = texture(iChannel1, (p+vec2(137.0, 189.0)) / vec2(1024.0));
    
    #if ANIMATE_NOISE
    const float c_goldenRatioConjugate = 0.61803398875;
    blue1 = fract(blue1 + float(iFrame%256) * c_goldenRatioConjugate);
    blue2 = fract(blue2 + float(iFrame%256) * c_goldenRatioConjugate);
    #endif        
}

// Function 1592
float noise(vec3 p){
    
    float t = iTime;
    vec3 np = normalize(p);
    
    // kind of bi-planar mapping
    float a = texture(iChannel0,t/20.+np.xy).x;      
    float b = texture(iChannel0,t/20.+.77+np.yz).x;
    
    a = mix(a,.5,abs(np.x));
    b = mix(b,.5,abs(np.z));
    
    float noise = a+b-.4;    
    noise = mix(noise,.5,abs(np.y)/2.);
        
    return noise;
}

// Function 1593
vec3 voronoi( in vec3 x )
{
    vec3 p = floor( x );
    vec3 f = fract( x );

	//float id = 0.0;
    vec2 res = vec2( 100.0 );
    for( int k=-1; k<=1; k++ )
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec3 b = vec3( float(i), float(j), float(k) );
        vec3 r = vec3( b ) - f + hash( p + b );
        float d = dot( r, r );

        if( d < res.x )
        {
			//id = dot( p+b, vec3(1.0,57.0,113.0 ) );
            res = vec2( d, res.x );			
        }
        else if( d < res.y )
        {
            res.y = d;
        }
    }

    return vec3(  res , 0.0 );//abs(id)
}

// Function 1594
float noise(vec2 p){return texture(iChannel1,p/64.).r;}

// Function 1595
vec4 noise(ivec3 p){
    const float scale = 1.0/float(0xffffffffU);
    uvec4 h = hash(uvec3(p));
    return vec4(h)*scale;
}

// Function 1596
float cyclicMultNoise( vec2 uv, vec2 resolution, vec2 base_cycle ) {

	float d = 1.; // initial density
    
	float n_tiles_level_1 = pow(2.,FirstDivision+round(ZoomDistance/log(2.)));
    
    uv *= n_tiles_level_1;
	vec2 cycle = base_cycle*n_tiles_level_1;
    
	// computation of the multiplicative noise
	float q = n_tiles_level_1;
	for (float i = 0.; i < NbScales; i++) {
        // Stop if the value is too low (and we assume it will thus stay low)
		if (d<1e-2) continue;
		
		// compute only the visible scales
		float crit = q - length(resolution)/LimitDetails;
		if (crit < SmoothZone) {
            
            float n = noise(uv + 10.7*i*i, cycle); // n : [-1,1]

            // Sharpen the noise
            for (int j = 0; j < GazConcentration; j++) {
                n = sin(n*PI/2.); // n : [-1,1] -> [-1,1]
            }

            n = n+1.; // n : [-1,1] -> [0,2]
            
            if (crit>0.) {
                // avoid aliasing by linear interpolation
				float t = crit/SmoothZone;
				n = mix(n,1.,t);
			}
            
			d *= n;
		}
		
        // go to the next octave
		uv *= fRatio;
        cycle *= fRatio;
        q*= fRatio;
	}
	
	d = max(d,0.);
    return d;
}

// Function 1597
vec2 noise2_2(vec2 uv)
{
    vec2 f = smoothstep(0.0, 1.0, fract(uv));
    
 	vec2 uv00 = floor(uv);
    vec2 uv01 = uv00 + vec2(0,1);
    vec2 uv10 = uv00 + vec2(1,0);
    vec2 uv11 = uv00 + 1.0;
    vec2 v00 = hash2_2(uv00);
    vec2 v01 = hash2_2(uv01);
    vec2 v10 = hash2_2(uv10);
    vec2 v11 = hash2_2(uv11);
    
    vec2 v0 = mix(v00, v01, f.y);
    vec2 v1 = mix(v10, v11, f.y);
    vec2 v = mix(v0, v1, f.x);
    
    return v;
}

// Function 1598
float orbitNoise(vec2 p)
{
    //return wrap(p,iFrame,iTime);
    //return fract(p.x-p.y);
    
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float rz = 0.;
    float orbitRadius = .75;

    //16 taps
    for (int j = -1; j <= 2; j++)
    for (int i = -1; i <= 2; i++)
    {
        vec2 dp = vec2(j,i);
        vec4 rn = hash42(dp + ip) - 0.5;
        vec2 op = fp - dp + rn.zw*orbitRadius;
        rz += nuttall(length(op),1.85)*dot(rn.xy*1.7, op);
    }
    return rz*0.5+0.5;
    //return smoothstep(-1.0, 1.0,rz);
    /**/
}

// Function 1599
float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    vec2 u = smoothstep(0.,1.,f);

    return mix( mix( dot( random2(i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ),
                     dot( random2(i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( random2(i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ),
                     dot( random2(i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 1600
float noise(vec2 n, float t) {
    
    vec3 p = vec3(n.x, n.y, t);
    const float K1 = 0.333333333;
    const float K2 = 0.166666667;
    
    vec3 i = floor(p + (p.x + p.y + p.z) * K1);
    vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);
    
    vec3 e = step(vec3(0.0), d0 - d0.yzx);
	vec3 i1 = e * (1.0 - e.zxy);
	vec3 i2 = 1.0 - e.zxy * (1.0 - e);
    
    vec3 d1 = d0 - (i1 - 1.0 * K2);
    vec3 d2 = d0 - (i2 - 2.0 * K2);
    vec3 d3 = d0 - (1.0 - 3.0 * K2);
    
    vec4 h = max(0.6 - vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), 0.0);
    vec4 q = h * h * h * h * vec4(dot(d0, hash(i)), dot(d1, hash(i + i1)), dot(d2, hash(i + i2)), dot(d3, hash(i + 1.0)));
    
    return dot(vec4(50.), q);

}

// Function 1601
float SmoothNoise(vec2 uv){
    vec2 lv = fract(uv * 10.);
    vec2 id = floor(uv * 10.);
    
    lv = lv * lv * (3. -2. * lv);
    
    float bl = N21(id);
    float br = N21(id + vec2(1,0));
    float b = mix(bl, br, lv.x);
    
    float tl = N21(id+vec2(0,1));
    float tr = N21(id+vec2(1,1));
    float t = mix(tl, tr, lv.x);
    
    return mix(b, t, lv.y);
}

// Function 1602
float snoise(vec2 v) {
  const vec4 C = vec4(.211324865405187,.366025403784439,-.577350269189626,.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1., 0.) : vec2(0., 1.);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod289(i);
  vec3 p = permute( permute( i.y + vec3(0., i1.y, 1. )) + i.x + vec3(0., i1.x, 1. ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.);
  m = m*m;
  m = m*m;
  vec3 x = 2. * fract(p * C.www) - 1.;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - .85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130. * dot(m, g);
}

// Function 1603
float noise(vec2 co)
{
	vec2 pos  = floor(co);
	vec2 fpos = co - pos;
	
	fpos = (3.0 - 2.0*fpos)*fpos*fpos;
	
	float c1 = shash(pos);
	float c2 = shash(pos + vec2(0.0, 1.0));
	float c3 = shash(pos + vec2(1.0, 0.0));
	float c4 = shash(pos + vec2(1.0, 1.0));
	
	float s1 = mix(c1, c3, fpos.x);
	float s2 = mix(c2, c4, fpos.x);
	
	return mix(s1, s2, fpos.y);
}

// Function 1604
vec2 DenoiseSHAD(vec2 uv, vec2 lUV, vec2 aUV,  sampler2D attr, sampler2D light, float radius, vec3 CVP, vec3 CN, vec3 CVN,
            vec2 ires, vec2 hres, vec2 asfov) {
    //R denoiser
    vec4 L0=texture(light,lUV*ires);
    vec3 Accum=vec3(L0.zw*0.2,0.2);
    Accum+=(_DenoiseSHAD(uv,lUV,aUV,vec2(radius,0.),CVP,CN,CVN,attr,light,ires,hres,asfov)+
            _DenoiseSHAD(uv,lUV,aUV,vec2(0.,radius),CVP,CN,CVN,attr,light,ires,hres,asfov)+
            _DenoiseSHAD(uv,lUV,aUV,vec2(radius*0.707),CVP,CN,CVN,attr,light,ires,hres,asfov)+
            _DenoiseSHAD(uv,lUV,aUV,vec2(radius,-radius)*0.707,CVP,CN,CVN,attr,light,ires,hres,asfov)
            )*0.1;
    //Output
    return Accum.xy/Accum.z;
}

// Function 1605
float Voronoi( in vec2 x, float u)
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    float va = 1000.0;
    
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2  g = vec2( float(i), float(j) );
        vec3  o = hash3( p + g )*vec3(u,u,1.0);
        vec2  r = g - f + o.xy;
        float d = dot(r,r);
        float w = sqrt(d);
        va = min(va, w);
    }

    return va;
}

// Function 1606
vec4 texNoise(vec2 uv){ float f = 0.; f+=texture(iChannel0, uv*.125).r*.5; //Rough shadertoy approximation of the bonzomatic noise texture by yx - https://www.shadertoy.com/view/tdlXW4
    f+=texture(iChannel0,uv*.25).r*.25;f+=texture(iChannel0,uv*.5).r*.125;f+=texture(iChannel0,uv*1.).r*.125;f=pow(f,1.2);return vec4(f*.45+.05);
}

// Function 1607
vec3 voronoi3(in vec2 x, out vec4 cellCenters)
{
	vec2 p = floor(x);
	vec2 f = fract(x);

	vec2 i1 = vec2(0.0);
	vec2 i2 = vec2(0.0);
	vec3 res = vec3(8.0);
	for(int j = -1; j <= 1; j ++)
	{
		for(int i = -1; i <= 1; i ++)
		{
			vec2 b = vec2(i, j);
			vec2 r = vec2(b) - f + rand2(p + b) * voronoiRandK;

			//float d = max(abs(r.x), abs(r.y));
			float d = (abs(r.x) + abs(r.y));

			if (d < res.x)
			{
				res.z = res.y;
				res.y = res.x;
				res.x = d;
				i2 = i1;
				i1 = p + b;
			}
			else if (d < res.y)
			{
				res.z = res.y;
				res.y = d;
				//r2 = r;
				i2 = p + b;
			}
			else if (d < res.z)
			{
				res.z = d;
			}
		}
	}
	cellCenters = vec4(i1,i2);
	return res;
}

// Function 1608
vec3 voronoi( in vec3 x )
{
    vec3 p = floor( x );
    vec3 f = fract( x );

	float id = 0.0;
    vec2 res = vec2( 100.0 );
    for( int k=-1; k<=1; k++ )
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec3 b = vec3( float(i), float(j), float(k) );
        vec3 r = vec3( b ) - f + hash( p + b );
        float d = dot( r, r );

        if( d < res.x )
        {
			id = dot( p+b, vec3(1.0,57.0,113.0 ) );
            res = vec2( d, res.x );			
        }
        else if( d < res.y )
        {
            res.y = d;
        }
    }

    return vec3( sqrt( res ), abs(id) );
}

// Function 1609
float ValueOfVoronoiCell (vec2 coord)
{
    return Hash21( coord );
}

// Function 1610
float valueNoise(in vec2 p)
{
    vec2 X = floor(p);
    vec2 x = fract(p);
    
    vec2 fn = x * x * x * (6.0 * x * x - 15.0 * x + 10.0);
    float u = fn.x;
    float v = fn.y;
    
    float a = rand(X + vec2(0.0, 0.0));
    float b = rand(X + vec2(1.0, 0.0));
    float c = rand(X + vec2(0.0, 1.0));
    float d = rand(X + vec2(1.0, 1.0));
    
    float n = (b - a) * u + (c - a) * v + (a - b - c + d) * u * v + a;
    return 2.0 * n - 1.0;
}

// Function 1611
vec3 Voronoi(in vec3 p, in vec3 rd){
    
    // One of Tomkh's snippets that includes a wrap to deal with
    // larger numbers, which is pretty cool.

 
    vec3 n = floor(p);
    p -= n + .5;
 
    
    // Storage for all sixteen hash values. The same set of hash values are
    // reused in the second pass, and since they're reasonably expensive to
    // calculate, I figured I'd save them from resuse. However, I could be
    // violating some kind of GPU architecture rule, so I might be making 
    // things worse... If anyone knows for sure, feel free to let me know.
    //
    // I've been informed that saving to an array of vectors is worse.
    //vec2 svO[3];
    
    // Individual Voronoi cell ID. Used for coloring, materials, etc.
    cellID = vec3(0); // Redundant initialization, but I've done it anyway.

    // As IQ has commented, this is a regular Voronoi pass, so it should be
    // pretty self explanatory.
    //
    // First pass: Regular Voronoi.
	vec3 mo, o;
    
    // Minimum distance, "smooth" distance to the nearest cell edge, regular
    // distance to the nearest cell edge, and a line distance place holder.
    float md = 8., lMd = 8., lMd2 = 8., lnDist, d;
    
    for( int k=-2; k<=2; k++ ){
    for( int j=-2; j<=2; j++ ){
    for( int i=-2; i<=2; i++ ){
    
        o = vec3(i, j, k);
        o += hash33(n + o) - p;
        // Saving the hash values for reuse in the next pass. I don't know for sure,
        // but I've been informed that it's faster to recalculate the had values in
        // the following pass.
        //svO[j*3 + i] = o; 
  
        // Regular squared cell point to nearest node point.
        d = dot(o, o); 

        if( d<md ){
            
            md = d;  // Update the minimum distance.
            // Keep note of the position of the nearest cell point - with respect
            // to "p," of course. It will be used in the second pass.
            mo = o; 
            cellID = vec3(i, j, k) + n; // Record the cell ID also.
        }
       
    }
    }
    }

    // Second pass: Distance to closest border edge. The closest edge will be one of the edges of
    // the cell containing the closest cell point, so you need to check all surrounding edges of 
    // that cell, hence the second pass... It'd be nice if there were a faster way.
    for( int k=-3; k<=3; k++ ){
    for( int j=-3; j<=3; j++ ){
    for( int i=-3; i<=3; i++ ){
        
        // I've been informed that it's faster to recalculate the hash values, rather than 
        // access an array of saved values.
        o = vec3(i, j, k);
        o += hash33(n + o) - p;
        // I went through the trouble to save all sixteen expensive hash values in the first 
        // pass in the hope that it'd speed thing up, but due to the evolving nature of 
        // modern architecture that likes everything to be declared locally, I might be making 
        // things worse. Who knows? I miss the times when lookup tables were a good thing. :)
        // 
        //o = svO[j*3 + i];
        
        // Skip the same cell... I found that out the hard way. :D
        if( dot(o - mo, o - mo)>.00001 ){ 
            
            // This tiny line is the crux of the whole example, believe it or not. Basically, it's
            // a bit of simple trigonometry to determine the distance from the cell point to the
            // cell border line. See IQ's article for a visual representation.
            lnDist = dot(0.5*(o + mo), normalize(o - mo));
            
            // Abje's addition. Border distance using a smooth minimum. Insightful, and simple.
            //
            // On a side note, IQ reminded me that the order in which the polynomial-based smooth
            // minimum is applied effects the result. However, the exponentional-based smooth
            // minimum is associative and commutative, so is more correct. In this particular case, 
            // the effects appear to be negligible, so I'm sticking with the cheaper polynomial-based 
            // smooth minimum, but it's something you should keep in mind. By the way, feel free to 
            // uncomment the exponential one and try it out to see if you notice a difference.
            //
            // // Polynomial-based smooth minimum.
            //lMd = smin2(lMd, lnDist, lnDist*.75); //lnDist*.75
            //
            // Exponential-based smooth minimum. By the way, this is here to provide a visual reference 
            // only, and is definitely not the most efficient way to apply it. To see the minor
            // adjustments necessary, refer to Tomkh's example here: Rounded Voronoi Edges Analysis - 
            // https://www.shadertoy.com/view/MdSfzD
            lMd = sminExp(lMd, lnDist, 10.); 
            
            // Minimum regular straight-edged border distance. If you only used this distance,
            // the web lattice would have sharp edges.
            lMd2 = min(lMd2, lnDist);
        }

    }
    }
    }

    // Return the smoothed and unsmoothed distance. I think they need capping at zero... but 
    // I'm not positive.
    return max(vec3(lMd, lMd2, md), 0.);
}

// Function 1612
float Noise(in vec3 p)
{
    vec3 i = floor(p);
        vec3 f = fract(p); 
        f *= f * (3.0-2.0*f);

    return mix(
                mix(mix(hash13(i + vec3(0.,0.,0.)), hash13(i + vec3(1.,0.,0.)),f.x),
                        mix(hash13(i + vec3(0.,1.,0.)), hash13(i + vec3(1.,1.,0.)),f.x),
                        f.y),
                mix(mix(hash13(i + vec3(0.,0.,1.)), hash13(i + vec3(1.,0.,1.)),f.x),
                        mix(hash13(i + vec3(0.,1.,1.)), hash13(i + vec3(1.,1.,1.)),f.x),
                        f.y),
                f.z);
}

// Function 1613
float Noise(in vec3 p)
{
    vec3 i = floor(p);
	vec3 f = fract(p); 
	f *= f * (3.0-2.0*f);

    return mix(
		mix(mix(hash13(i + vec3(0.,0.,0.)), hash13(i + vec3(1.,0.,0.)),f.x),
			mix(hash13(i + vec3(0.,1.,0.)), hash13(i + vec3(1.,1.,0.)),f.x),
			f.y),
		mix(mix(hash13(i + vec3(0.,0.,1.)), hash13(i + vec3(1.,0.,1.)),f.x),
			mix(hash13(i + vec3(0.,1.,1.)), hash13(i + vec3(1.,1.,1.)),f.x),
			f.y),
		f.z);
}

// Function 1614
float noise3(vec3 v){
    v *= 64./2.; // emulates 64x64 noise texture
  const vec2 C = 1./vec2(6,3);
  const vec4 D = vec4(0,.5,1,2);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0);
  vec3 l = 1. - g.zxy;
  vec3 i1 = min( g, l );
  vec3 i2 = max( g, l );
  vec3 x1 = x0 - i1 + C.x;
  vec3 x2 = x0 - i2 + C.y;
  vec3 x3 = x0 - D.yyy;
  i = mod(i,289.);
  vec4 p = permute( permute( permute(
	  i.z + vec4(0., i1.z, i2.z, 1.))
	+ i.y + vec4(0., i1.y, i2.y, 1.))
	+ i.x + vec4(0., i1.x, i2.x, 1.));
  vec3 ns = .142857142857 * D.wyz - D.xzx;
  vec4 j = p - 49. * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = floor(j - 7. * x_ ) *ns.x + ns.yyyy;
  vec4 h = 1. - abs(x) - abs(y);
  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );
  vec4 sh = -step(h, vec4(0));
  vec4 a0 = b0.xzyw + (floor(b0)*2.+ 1.).xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + (floor(b1)*2.+ 1.).xzyw*sh.zzww ;
  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);
  vec4 norm = inversesqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  vec4 m = max(.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.);
  return .5 + 12. * dot( m * m * m, vec4( dot(p0,x0), dot(p1,x1),dot(p2,x2), dot(p3,x3) ) );
}

// Function 1615
float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = rand(i);
    float b = rand(i + vec2(1.0,0.0));
    float c = rand(i + vec2(0.0, 1.0));
    float d = rand(i + vec2(1.0, 1.0));

    vec2 u = smoothstep(0.0,1.0, f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Function 1616
float snoise3D(vec3 texc)
{
    return snoise3Dv4(texc).x;
}

// Function 1617
float polarNoise2(vec3 pos)
{
    float a = 2.*atan(pos.y, pos.x);
    vec3 pos1 = vec3(pos.z, length(pos.yx) + time2*txtSpeed, a);
    vec3 pos2 = vec3(pos.z, length(pos.yx) + time2*txtSpeed, a+12.57);    
    
    float f1 = polarNoise(pos1);
    float f2 = polarNoise(pos2);
    float f = mix(f1, f2, smoothstep(-5., -6.285, a));
    
    f = smoothstep(0.01, 0.2, f)-smoothstep(0.2, 0.52, f)+smoothstep(0.45, 0.63, f);
    f = 0.8-pati+f*pati;
    
    return f;
}

// Function 1618
float noise_sum_abs(vec3 p)
{
    float f = 0.0;
    p = p * 3.0;
    f += 1.0000 * abs(noise(p)); p = 2.0 * p;
    f += 0.5000 * abs(noise(p)); p = 2.0 * p;
	f += 0.2500 * abs(noise(p)); p = 2.0 * p;
	f += 0.1250 * abs(noise(p)); p = 2.0 * p;
	f += 0.0625 * abs(noise(p)); p = 2.0 * p;
    
    return f;
}

// Function 1619
float noise(vec3 uv) {
    vec3 fuv = floor(uv);
    vec4 cell0 = vec4(
        hash(fuv + vec3(0, 0, 0)),
        hash(fuv + vec3(0, 1, 0)),
        hash(fuv + vec3(1, 0, 0)),
        hash(fuv + vec3(1, 1, 0))
    );
    vec2 axis0 = mix(cell0.xz, cell0.yw, fract(uv.y));
    float val0 = mix(axis0.x, axis0.y, fract(uv.x));
    vec4 cell1 = vec4(
        hash(fuv + vec3(0, 0, 1)),
        hash(fuv + vec3(0, 1, 1)),
        hash(fuv + vec3(1, 0, 1)),
        hash(fuv + vec3(1, 1, 1))
    );
    vec2 axis1 = mix(cell1.xz, cell1.yw, fract(uv.y));
    float val1 = mix(axis1.x, axis1.y, fract(uv.x));
    return mix(val0, val1, fract(uv.z));
}

// Function 1620
vec2 Voronoi(in vec2 p, in vec2 dim){
    
    p *= dim;
    
    // One of Tomkh's snippets that includes a wrap to deal with
    // larger numbers, which is pretty cool.

#if 1
    // Slower, but handles big numbers better.
    vec2 n = floor(p);
    p -= n;
    vec2 h = step(.5, p) - 1.5;
    n += h; p -= h;
#else
    vec2 n = floor(p - 1.);
    p -= n;
#endif
    
    // Storage for all sixteen hash values. The same set of hash values are
    // reused in the second pass, and since they're reasonably expensive to
    // calculate, I figured I'd save them from resuse. However, I could be
    // violating some kind of GPU architecture rule, so I might be making 
    // things worse... If anyone knows for sure, feel free to let me know.
    //
    // I've been informed that saving to an array of vectors is worse.
    //vec2 svO[3];
    
    // Individual Voronoi cell ID. Used for coloring, materials, etc.
    cellID = vec2(0); // Redundant initialization, but I've done it anyway.

    // As IQ has commented, this is a regular Voronoi pass, so it should be
    // pretty self explanatory.
    //
    // First pass: Regular Voronoi.
	vec2 mo, o;
    
    // Minimum distance, "smooth" distance to the nearest cell edge, regular
    // distance to the nearest cell edge, and a line distance place holder.
    float md = 8., lMd = 8., lMd2 = 8., lnDist, d;
    
    for( int j=0; j<3; j++ )
    for( int i=0; i<3; i++ ){
    
        o = vec2(i, j);
        o += hash22B(n + o, dim) - p;
        // Saving the hash values for reuse in the next pass. I don't know for sure,
        // but I've been informed that it's faster to recalculate the had values in
        // the following pass.
        //svO[j*3 + i] = o; 
  
        // Regular squared cell point to nearest node point.
        d = dot(o, o); 

        if( d<md ){
            
            md = d;  // Update the minimum distance.
            // Keep note of the position of the nearest cell point - with respect
            // to "p," of course. It will be used in the second pass.
            mo = o; 
            cellID = vec2(i, j) + n; // Record the cell ID also.
        }
       
    }
    
    #ifdef FAST_EXP
    lMd = 0.;
    #endif

    // Second pass: Distance to closest border edge. The closest edge will be one of the edges of
    // the cell containing the closest cell point, so you need to check all surrounding edges of 
    // that cell, hence the second pass... It'd be nice if there were a faster way.
    for( int j=0; j<3; j++ )
    for( int i=0; i<3; i++ ){
        
        // I've been informed that it's faster to recalculate the hash values, rather than 
        // access an array of saved values.
        o = vec2(i, j);
        o += hash22B(n + o, dim) - p;
        // I went through the trouble to save all sixteen expensive hash values in the first 
        // pass in the hope that it'd speed thing up, but due to the evolving nature of 
        // modern architecture that likes everything to be declared locally, I might be making 
        // things worse. Who knows? I miss the times when lookup tables were a good thing. :)
        // 
        //o = svO[j*3 + i];
        
        // Skip the same cell... I found that out the hard way. :D
        if( dot(o-mo, o-mo)>.00001 ){ 
            
            // This tiny line is the crux of the whole example, believe it or not. Basically, it's
            // a bit of simple trigonometry to determine the distance from the cell point to the
            // cell border line. See IQ's article for a visual representation.
            lnDist = dot( 0.5*(o+mo), normalize(o-mo));
            
            // Abje's addition. Border distance using a smooth minimum. Insightful, and simple.
            //
            // On a side note, IQ reminded me that the order in which the polynomial-based smooth
            // minimum is applied effects the result. However, the exponentional-based smooth
            // minimum is associative and commutative, so is more correct. In this particular case, 
            // the effects appear to be negligible, so I'm sticking with the cheaper polynomial-based 
            // smooth minimum, but it's something you should keep in mind. By the way, feel free to 
            // uncomment the exponential one and try it out to see if you notice a difference.
            //
            // // Polynomial-based smooth minimum.
            //lMd = smin(lMd, lnDist, lnDist*.5); 
            //
            // Exponential-based smooth minimum. By the way, this is here to provide a visual reference 
            // only, and is definitely not the most efficient way to apply it. To see the minor
            // adjustments necessary, refer to Tomkh's example here: Rounded Voronoi Edges Analysis - 
            // https://www.shadertoy.com/view/MdSfzD
            
            #ifdef FAST_EXP
            lMd += sExp(lnDist, (lnDist)*30.);
            #else
            lMd = sminExp(lMd, lnDist, (lnDist)*30.); 
            #endif
            
            // Minimum regular straight-edged border distance. If you only used this distance,
            // the web lattice would have sharp edges.
            lMd2 = min(lMd2, lnDist);
        }

    }
    
    #ifdef FAST_EXP
    lMd = -log(lMd)/((lMd)*30.);
    #endif

    // Return the smoothed and unsmoothed distance. I think they need capping at zero... but 
    // I'm not positive.
    return max(vec2(lMd, lMd2), 0.);
}

// Function 1621
vec3 noise(float x)
{
    float p = fract(x); x-=p;
    return mix(hash(x),hash(x+1.0),p);
}

// Function 1622
float cloudNoise(float scale,in vec3 p, in vec3 dir)
{
	vec3 q = p + dir; 
    float f;
	f  = 0.50000*noise( q ); q = q*scale*2.02 + dir;
    f += 0.25000*noise( q ); q = q*2.03 + dir;
    f += 0.12500*noise( q ); q = q*2.01 + dir;
    f += 0.06250*noise( q ); q = q*2.02 + dir;
    f += 0.03125*noise( q );
    return f;
}

