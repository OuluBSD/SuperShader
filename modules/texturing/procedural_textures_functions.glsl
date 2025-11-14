// Reusable Procedural Textures Texturing Functions
// Automatically extracted from texturing/mapping-related shaders

// Function 1
float vnoise(in vec2 p) {
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 2
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

// Function 3
float Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    return mix(mix(mix( Hash(n+  0.0), Hash(n+  1.0),f.x),
                   mix( Hash(n+ 57.0), Hash(n+ 58.0),f.x),f.y),
               mix(mix( Hash(n+113.0), Hash(n+114.0),f.x),
                   mix( Hash(n+170.0), Hash(n+171.0),f.x),f.y),f.z);
}

// Function 4
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

// Function 5
float worleyNoise(vec2 uv, float freq, float t, bool curl)
{
    uv *= freq;
    uv += t + (curl ? curlNoise(uv*2.) : vec2(0.)); // exaggerate the curl noise a bit
    
    vec2 id = floor(uv);
    vec2 gv = fract(uv);
    
    float minDist = 100.;
    for (float y = -1.; y <= 1.; ++y)
    {
        for(float x = -1.; x <= 1.; ++x)
        {
            vec2 offset = vec2(x, y);
            vec2 h = hash22(id + offset) * .8 + .1; // .1 - .9
    		h += offset;
            vec2 d = gv - h;
           	minDist = min(minDist, dot(d, d));
        }
    }
    
    return minDist;
}

// Function 6
vec3 noise3( in vec3 x)
{
	return vec3( noise(x+vec3(123.456,.567,.37)),
				noise(x+vec3(.11,47.43,19.17)),
				noise(x) );
}

// Function 7
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

// Function 8
float smoothNoise(vec3 q){
	float f = 0.5000*noise( q ); q = q*2.01;
    f += 0.2500*noise( q ); q = q*2.02;
    f += 0.1250*noise( q ); q = q*2.03;
    f += 0.0625*noise( q ); q = q*2.01;
	return f;
}

// Function 9
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

// Function 10
vec2 Noisev2v4 (vec4 p)
{
  vec4 i, f, t1, t2;
  i = floor (p);  f = fract (p);  f = f * f * (3. - 2. * f);
  t1 = Hashv4f (dot (i.xy, cHashA3.xy));  t2 = Hashv4f (dot (i.zw, cHashA3.xy));
  return vec2 (mix (mix (t1.x, t1.y, f.x), mix (t1.z, t1.w, f.x), f.y),
               mix (mix (t2.x, t2.y, f.z), mix (t2.z, t2.w, f.z), f.w));
}

// Function 11
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

// Function 12
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

// Function 13
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

// Function 14
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

// Function 15
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

// Function 16
float treeNoise(vec2 p)
{
    p = vec2(ivec2(p * 1.8));
    vec2 o = vec2(0.12, 0.08);
    return (hash12(p + o.xy) + hash12(p + o.yx) + hash12(p + o.xx) + hash12(p + o.yy)) * 0.25;
}

// Function 17
float noise(vec2 p){
    vec2 ip = floor(p); vec2 u = fract(p);u=u*u*(3.0-2.0*u);
    float res = mix(
        mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
        mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
    return res;
}

// Function 18
float noise(vec2 p){
    return texture(iChannel0,sFract(p)).r;
}

// Function 19
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

// Function 20
float noise(in vec2 x){return texture(iChannel0, x*.01).x;}

// Function 21
float noise(in float x) {
	float p = floor(x);
	float f = fract(x);
		
	f = f*f*(3.0-2.0*f);	
	return mix( hash(p+  0.0), hash(p+  1.0),f);
}

// Function 22
float perlinNoise2D(vec2 xy, float freq, float amp, int octaves, int seed){
    float total = 0.0;
    float totalScale = 0.0;
    // current freq, amp, scale
    vec3 currFAS = vec3(freq, amp, amp);
    for(int i = 0; i < 5; i++){
        total += interpolatedNoise2D(abs(xy) * currFAS.x, seed) * currFAS.y;
        totalScale += currFAS.z;
        currFAS *= vec3(2.0, 0.5, 0.5);
        if (i >= octaves) break;
    }
    return amp * (total / totalScale);
}

// Function 23
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

// Function 24
float skyNoise( in vec2 p ) {
    const float K1 = 0.366025404; 
    const float K2 = 0.211324865; 
    vec2 i = floor(p + (p.x+p.y)*K1);	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;
    vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3 n = 2.*h*h*h*h*vec3( dot(a,hash(i)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot(n, vec3(70.));	
}

// Function 25
float vnoise(in float p) {
    float i = floor( p );
    float f = fract( p );	
	float u = f*f*(3.0-2.0*f);
    return mix( hash( i ), hash( i + 1.0 ), u);
}

// Function 26
float tetraNoise(in vec3 p)
{
    vec3 i = floor(p + dot(p, vec3(0.333333)) );  p -= i - dot(i, vec3(0.166666)) ;
    
    vec3 i1 = step(p.yzx, p), i2 = max(i1, 1.0-i1.zxy); i1 = min(i1, 1.0-i1.zxy);    
    
    vec3 p1 = p - i1 + 0.166666, p2 = p - i2 + 0.333333, p3 = p - 0.5;
  
    vec4 v = max(0.5 - vec4(dot(p,p), dot(p1,p1), dot(p2,p2), dot(p3,p3)), 0.0);
    vec4 d = vec4(dot(p, hash33(i)), dot(p1, hash33(i + i1)), dot(p2, hash33(i + i2)), dot(p3, hash33(i + 1.)));
    
    return clamp(dot(d, v*v*v*8.)*1.732 + .5, 0., 1.); 
}

// Function 27
vec4 noised( in vec3 x ) {
    
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
    
    
    float a = hash13(p+vec3(0.0,0.0,0.0));
    float b = hash13(p+vec3(1.0,0.0,0.0));
    float c = hash13(p+vec3(0.0,1.0,0.0));
    float d = hash13(p+vec3(1.0,1.0,0.0));
    float e = hash13(p+vec3(0.0,0.0,1.0));
	float f = hash13(p+vec3(1.0,0.0,1.0));
    float g = hash13(p+vec3(0.0,1.0,1.0));
    float h = hash13(p+vec3(1.0,1.0,1.0));
	
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

// Function 28
vec2 valueNoiseFilter(vec2 x) {
    #if defined(VALUE_NOISE_FILTER_QUINTIC)
    return x*x*x*(x*(x*6.-15.)+10.);
    #elif defined(VALUE_NOISE_FILTER_SMOOTH)
    return smoothstep(0.0,1.0,x);
    #else
    return x;
    #endif
}

// Function 29
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

// Function 30
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

// Function 31
vec3 gradientNoised(vec2 pos, vec2 scale, float rotation, float seed) 
{
    vec2 sinCos = vec2(sin(rotation), cos(rotation));
    return gradientNoised(pos, scale, mat2(sinCos.y, sinCos.x, sinCos.x, sinCos.y), seed);
}

// Function 32
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

// Function 33
vec3 value_noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f*f*f*(f*(f*6.0 - 15.0) + 10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0); //iq

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = a - b - c + d;

    vec3 nres;
    
    nres.x = mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    nres.yz = vec2(k1+k3*u.y, k2+k3*u.x)*du;

    return nres;
}

// Function 34
float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p); f = f*f*(3.-2.*f); // smoothstep

    float v= mix( mix( mix(hash31(i+vec3(0,0,0)),hash31(i+vec3(4,0,0)),f.x),
                       mix(hash31(i+vec3(0,1,0)),hash31(i+vec3(1,1,0)),f.y), f.z), 
                  mix( mix(hash31(i+vec3(0,3,1)),hash31(i+vec3(1,0,5)),f.z),
                       mix(hash31(i+vec3(0,1,1)),hash31(i+vec3(1,1,1)),f.x), f.y), f.z);
	return   MOD==0 ? v
	       : MOD==1 ? 2.*v-1.
           : MOD==2 ? abs(2.*v-1.)
                    : 1.-abs(2.*v-1.);
}

// Function 35
void noise_prng_srand(inout noise_prng this_, in uint s)
{
    this_.x_ = s;
}

// Function 36
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

// Function 37
float interpolatedNoise3D(vec3 xyz, int seed) {
    vec3 p = floor(xyz);
    vec3 f = fract(xyz);
    f = f*f*(3.0-2.0*f);

    float n = dot(vec4(p, seed), vec4(1, 433, 157, 141));
    return mix(mix(mix(hash(n+  0.0), hash(n+  1.0),f.x),
                   mix(hash(n+157.0), hash(n+158.0),f.x),f.z),
               mix(mix(hash(n+433.0), hash(n+434.0),f.x),
                   mix(hash(n+590.0), hash(n+591.0),f.x),f.z), f.y);
}

// Function 38
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

// Function 39
float SmoothNoise( vec2 p )
{
    p *= 0.0013;

    float s = 1.0;
	float t0 = 0.0;
	for( int i=0; i<1; i++ )
	{
        t0 += s*Noise( p );
		s *= 0.5 + 0.1*t0;
        p = 0.97*smm2*p + (t0-0.5)*0.2;
	}
	return t0;
}

// Function 40
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

// Function 41
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

// Function 42
float noise(vec2 p){
	return fract(sin(fract(sin(p.x) * (4313.13311)) + p.y) * 3131.0011);
}

// Function 43
float noise(vec2 p)
{
  return textureLod(iChannel0,p*vec2(1./256.),0.0).x;
}

// Function 44
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

// Function 45
vec3 noise33(  vec3 p )
{
    p=p+10000.;
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u;
    u.x = f.x*f.x*(3.0-2.0*f.x);
    u.y = f.y*f.y*(3.0-2.0*f.y);
    u.z = f.z*f.z*(3.0-2.0*f.z);

    return mix( 
        mix(
                mix( hash33( i + vec3(0.0,0.0,0.0) ), 
                     hash33( i + vec3(1.0,0.0,0.0) ), u.x),
                mix( hash33( i + vec3(0.0,1.0,0.0) ), 
                     hash33( i + vec3(1.0,1.0,0.0) ), u.x)
               , u.y),
              mix(
                mix( hash33( i + vec3(0.0,0.0,1.0) ), 
                     hash33( i + vec3(1.0,0.0,1.0) ), u.x),
                mix( hash33( i + vec3(0.0,1.0,1.0) ), 
                     hash33( i + vec3(1.0,1.0,1.0) ), u.x)
               , u.y)
        
         , u.z)
        ;
}

// Function 46
vec4 noise(ivec3 p){
    const float scale = 1.0/float(0xffffffffU);
    uvec4 h = hash(uvec3(p));
    return vec4(h)*scale;
}

// Function 47
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

// Function 48
float noise(vec2 x){
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float res = mix(mix(hash(p),          hash(p + add.xy),f.x),
                    mix(hash(p + add.yx), hash(p + add.xx),f.x),f.y);
    return res;
}

// Function 49
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

// Function 50
float smooth_noise(vec2 pos)
{
	return   ( noise(pos + vec2(1,1)) + noise(pos + vec2(1,1)) + noise(pos + vec2(1,1)) + noise(pos + vec2(1,1)) ) / 16.0 		
		   + ( noise(pos + vec2(1,0)) + noise(pos + vec2(-1,0)) + noise(pos + vec2(0,1)) + noise(pos + vec2(0,-1)) ) / 8.0 		
    	   + noise(pos) / 4.0;
}

// Function 51
float InterleavedGradientNoise(vec2 pixel, int frame) 
{
    pixel += (float(frame) * 5.588238f);
    return fract(52.9829189f * fract(0.06711056f*float(pixel.x) + 0.00583715f*float(pixel.y)));  
}

// Function 52
float noise(vec3 v) {
    vec3 V = floor(v);v-=V;
    return mix(mix(
    	mix(hash(V),      hash(V+E.xyy),v.x),
        mix(hash(V+E.yxy),hash(V+E.xxy),v.x),v.y),
        mix(mix(hash(V+E.yyx),hash(V+E.xyx),v.x),
        mix(hash(V+E.yxx),hash(V+E.xxx),v.x),v.y),v.z);
}

// Function 53
vec2 Noise( in vec2 x ) {
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 res = mix(mix( Hash(p + 0.0), Hash(p + vec2(1.0, 0.0)),f.x),
                   mix( Hash(p + vec2(0.0, 1.0) ), Hash(p + vec2(1.0, 1.0)),f.x),f.y);
    return res-.5;
}

// Function 54
float noise(float s){
    // Noise is sampled at every integer s
    // If s = t*f, the resulting signal is close to a white noise
    // with a sharp cutoff at frequency f.
    int si = int(floor(s));
    float sf = fract(s);
    sf = sf*sf*(3.-2.*sf); // smoothstep(0,1,sf)
    return mix(rand(float(si)), rand(float(si+1)), sf)*2.-1.;
}

// Function 55
float snoise3(vec3 st){
  vec3 p = st + (st.x + st.y + st.z) / 3.0;
  vec3 f = fract(p);
  vec3 i = floor(p);
  vec3 g0, g1, g2, g3;
  vec4 wt;
  g0 = i;
  g3 = i + u_111;
  if(f.x >= f.y && f.x >= f.z){
    g1 = i + u_100;
    g2 = i + (f.y >= f.z ? u_110 : u_101);
    wt = (f.y >= f.z ? vec4(1.0 - f.x, f.x - f.y, f.y - f.z, f.z) : vec4(1.0 - f.x, f.x - f.z, f.z - f.y, f.y));
  }else if(f.y >= f.x && f.y >= f.z){
    g1 = i + u_010;
    g2 = i + (f.x >= f.z ? u_110 : u_011);
    wt = (f.x >= f.z ? vec4(1.0 - f.y, f.y - f.x, f.x - f.z, f.z) : vec4(1.0 - f.y, f.y - f.z, f.z - f.x, f.x));
  }else{
    g1 = i + u_001;
    g2 = i + (f.x >= f.y ? u_101 : u_011);
    wt = (f.x >= f.y ? vec4(1.0 - f.z, f.z - f.x, f.x - f.y, f.y) : vec4(1.0 - f.z, f.z - f.y, f.y - f.x, f.x));
  }
  float value = 0.0;
  wt = wt * wt * wt * (wt * (wt * 6.0 - 15.0) + 10.0);
  value += wt.x * dot(p - g0, random3(g0));
  value += wt.y * dot(p - g1, random3(g1));
  value += wt.z * dot(p - g2, random3(g2));
  value += wt.w * dot(p - g3, random3(g3));
  return value;
}

// Function 56
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

// Function 57
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

// Function 58
float snoise(vec3 p) {

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

// Function 59
float cloudNoise2D(vec2 uv, vec2 _wind)
{
    float v = 1.0-voronoi2D(uv*10.0+_wind);
    float fs = fbm2Dsimple(uv*20.0+_wind);
    return clamp(v*fs, 0.0, 1.0);
}

// Function 60
float noise( const in vec2 x ) {
    vec2 p = floor(x);
    vec2 f = fract(x);
	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);
	return textureLod( iChannel0, (uv+118.4)/256.0, 0.0 ).x;
}

// Function 61
float valueNoise2du(vec2 samplePoint) {
    vec2 pointI = floor(samplePoint);
    vec2 pointF = fract(samplePoint);
    vec2 u = valueNoiseFilter(pointF);

    vec2 m = mix(
        vec2(
            hash21(pointI), //bl
            hash21(pointI + vec2(0.0,1.0)) //fl
        ),
        vec2(
            hash21(pointI + vec2(1.0,0.0) ),//br
            hash21(pointI + vec2(1.0,1.0) ) //fr
        ),u.x);

    return mix(m.x,m.y,u.y);
}

// Function 62
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

// Function 63
float valueNoise(vec2 p)
{
    vec2 ip = floor(p);
    vec2 fp = fract(p);
	vec2 ramp = fp*fp*(3.0-2.0*fp);

    float rz= mix( mix( hash12(ip + vec2(0.0,0.0)), hash12(ip + vec2(1.0,0.0)), ramp.x),
                   mix( hash12(ip + vec2(0.0,1.0)), hash12(ip + vec2(1.0,1.0)), ramp.x), ramp.y);
    
    return rz;
}

// Function 64
float Noisefv2 (vec2 p){
  vec4 t=Hashv4f(dot(floor(p),cHashA4.yz));
  p=fract(p);p*=p*(3.-2.*p);
  return mix (mix(t.x,t.y,p.x),
               mix(t.z,t.w,p.x),p.y);}

// Function 65
float noise(in vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    f *= f * (3.0 - 2.0 * f);
    
    return mix(mix(hash(i + vec2(0., 0.)), hash(i + vec2(1., 0.)), f.x),
               mix(hash(i + vec2(0., 1.)), hash(i + vec2(1., 1.)), f.x),
               f.y);
}

// Function 66
float Noise2d( in vec2 x )
{
    float xhash = cos( x.x * 37.0 );
    float yhash = cos( x.y * 57.0 );
    return fract( 415.92653 * ( xhash + yhash ) );
}

// Function 67
float noise(vec2 n){
	vec2 d = vec2(0.0, 1.0);
	vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
	return mix(mix(rand(b + d.xx), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

// Function 68
float noise(vec3 x) {
    vec3 p = floor(x), f = fract(x);
	vec2 rg = texture(iChannel1, (p.xy+vec2(37.0,17.0)*p.z + f.xy)/256.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 69
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

// Function 70
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

// Function 71
vec3 noise(vec3 p, float lod){float m = mod(p.z,1.0);float s = p.z-m; float sprev = s-1.0;if (mod(s,2.0)==1.0) { s--; sprev++; m = 1.0-m; };return mix(texture(iChannel0,p.xy/iChannelResolution[0].xy+noise(sprev,lod).yz,lod).xyz,texture(iChannel0,p.xy/iChannelResolution[0].xy+noise(s,lod).yz,lod).xyz,m);}

// Function 72
float noise11(in float x) {
	return fract(sin(x)*35746.1764); 
}

// Function 73
float mynoise(vec2 u) {
    return noise(vec3(u,0));                // use procedural noise
 // return texture(iChannel0, u/256.).x;  // use image noise
}

// Function 74
float iqnoise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
	float k = 1.+63.*pow(1.-v,4.);
	float va = 0.;
	float wt = 0.;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ ) {
        vec2 g = vec2(i,j);
		vec3 o = hash3( p + g )*vec3(u,u,1.);
		vec2 r = g - f + o.xy;
		float d = dot(r,r);
		float ww = pow( 1.-smoothstep(0.,1.414,sqrt(d)), k );
		va += o.z*ww;
		wt += ww;
    }
	
    return va/wt;
}

// Function 75
float interleavedGradientNoise(vec2 pos)
{
  float f = 0.06711056 * pos.x + 0.00583715 * pos.y;
  return fract(52.9829189 * fract(f));
}

// Function 76
float noise( vec2 x )
{
    x *= iResolution.y;
    vec2 p = floor(x),f = fract(x);

    f = f*f*(3.-2.*f);                       // to make derivative continuous at borders

#define hash(p)  fract(sin(1e3*dot(p,vec2(1,57)))*43758.5453)        // rand
    
    return mix(mix( hash(p+vec2(0,0)), hash(p+vec2(1,0)),f.x),       // bilinear interp
               mix( hash(p+vec2(0,1)), hash(p+vec2(1,1)),f.x),f.y);
}

// Function 77
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

// Function 78
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

// Function 79
float triNoise3d(in vec3 p, in float spd)
{
    float z=1.4;
	float rz = 0.;
    vec3 bp = p;
	for (float i=0.; i<=3.; i++ )
	{
        vec3 dg = tri3(bp*2.);
        p += (dg+time*.1*spd);

        bp *= 1.8;
		z *= 1.5;
		p *= 1.2;
        //p.xz*= m2;
        
        rz+= (tri(p.z+tri(p.x+tri(p.y))))/z;
        bp += 0.14;
	}
	return rz;
}

// Function 80
float noise( in vec4 x )
{
    vec4 p = floor(x);
    vec4 f = fract(x);
	f = f*f*(3.0-2.0*f);
    
	vec2 uv = (p.xy + p.z*zOffset + p.w*wOffset) + f.xy;
    
   	vec4 s = tex(uv);
	return mix(mix( s.x, s.y, f.z ), mix(s.z, s.w, f.z), f.w);
}

// Function 81
float rainNoise(in vec2 x)
{
    vec2 p = floor(x);
    vec2 f = _smoothstep(x);
    float n = p.x + p.y * 57.0;
    return mix(mix(rainHash(n +  0.0), rainHash(n +  1.0), f.x),
               mix(rainHash(n + 57.0), rainHash(n + 58.0), f.x), f.y);
}

// Function 82
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

// Function 83
float smoothNoise2(vec2 p){vec2 pf=fract(p);
 return mix(mix(noise(floor(p          )),noise(floor(p+vec2(1,0))),pf.x),
             mix(noise(floor(p+vec2(0,1))),noise(floor(p+vec2(1,1))),pf.x),pf.y);}

// Function 84
float noise(vec2 p){
	vec2 ip = floor(p);
	vec2 u = fract(p);
	u = u*u*(3.0-2.0*u);
	
	float res = mix(
		mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
		mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
	return res*res;
}

// Function 85
float noise(float n,float s,float res)
{
	float a = fract(sin(((floor((n)/s-0.5)*s)/res)*432.6326)*556.6426);
	float b = fract(sin(((floor((n)/s+0.5)*s)/res)*432.6326)*556.6426);
	return mix(a,b,smoothstep(0.0,1.0,+mod(n/s+0.5,1.0)));
}

// Function 86
vec4 bluenoise( vec2 fc )
{
    return texture( iChannel2, fc / iChannelResolution[2].xy );
}

// Function 87
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 88
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

// Function 89
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

// Function 90
float noise(vec2 p, float r2, float z) {
    vec2 q = skew(p);
    vec2 qa = floor(q);
    vec2 qi = q - qa;
    vec2 qb = qa + vec2(1.0, 1.0);
    vec2 qc;
    
    if (qi.x < qi.y) {
		qc = qa + vec2(0.0, 1.0);
    } else {
		qc = qa + vec2(1.0, 0.0);
    }
    
    float ka = noise(qa, p, r2, z);
    float kb = noise(qb, p, r2, z);
    float kc = noise(qc, p, r2, z);
    return ka + kb + kc;
}

// Function 91
float noise1(float p)
{
	float fl = floor(p);
	float fc = fract(p);
	return mix(rand(fl), rand(fl + 1.0), fc);
}

// Function 92
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

// Function 93
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

// Function 94
float SimplexNoise(vec2 xy) {
    const float K1 = 0.3660254038;  // (sqrt(3)-1)/2
    const float K2 = 0.2113248654;  // (-sqrt(3)+3)/6
    vec2 p = xy + (xy.x + xy.y)*K1;
    vec2 i = floor(p);
    vec2 f1 = xy - (i - (i.x + i.y)*K2);
    vec2 s = f1.x < f1.y ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    vec2 f2 = f1 - s + K2;
    vec2 f3 = f1 - 1.0 + 2.0*K2;
    vec2 n1 = 2.0 * hash22(i) - 1.0;
    vec2 n2 = 2.0 * hash22(i + s) - 1.0;
    vec2 n3 = 2.0 * hash22(i + 1.0) - 1.0;
    vec3 v = vec3(dot(f1, n1), dot(f2, n2), dot(f3, n3));
    vec3 w = max(-vec3(dot(f1, f1), dot(f2, f2), dot(f3, f3)) + 0.5, vec3(0.0));
    return dot((w*w*w*w) * v, vec3(32.0));
}

// Function 95
float blockyNoise(vec2 uv, float threshold, float scale, float seed)
{
	float scroll = floor(iTime + sin(11.0 *  iTime) + sin(iTime) ) * 0.77;
    vec2 noiseUV = uv.yy / scale + scroll;
    float noise2 = texture(iChannel1, noiseUV).r;
    
    float id = floor( noise2 * 20.0);
    id = noise(id + seed) - 0.5;
    
  
    if ( abs(id) > threshold )
        id = 0.0;

	return id;
}

// Function 96
float noise( in vec2 x ){
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 57.0;
    return mix(mix( hash(n + 0.0), hash(n + 1.0), f.x), mix(hash(n + 57.0), hash(n + 58.0), f.x), f.y);
}

// Function 97
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

// Function 98
float bnoise(in vec2 p){ return fbm(p*5.); }

// Function 99
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

// Function 100
float noiseLayers(in vec3 p) {
    vec3 t = vec3(0., 0., p.z+iTime*1.5);

    const int iter = 5;
    float tot = 0., sum = 0., amp = 1.;

    for (int i = 0; i < iter; i++) {
        tot += xvoronoi(p + t) * amp;
        p *= 2.0;
        t *= 1.5;
        sum += amp;
        amp *= 0.5;
    }
    return tot/sum;
}

// Function 101
float noise(vec2 n) {
	const vec2 d = vec2(0.0, 1.0);
  	vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
	return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

// Function 102
float iqNoise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
                              hash( i + vec2(1.0,0.0) ), u.x),
                         mix( hash( i + vec2(0.0,1.0) ), 
                              hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 103
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

// Function 104
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

// Function 105
float snoise(vec3 v)
{
    return length(texture(iChannel0, v.xy * 0.1));
}

// Function 106
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

// Function 107
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

// Function 108
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

// Function 109
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

// Function 110
vec4 noise(float t){return texture(iChannel1,vec2(floor(t), floor(t))/256.);}

// Function 111
float noise3D(vec3 p)
{
	return fract(sin(dot(p ,vec3(12.9898,78.233,128.852))) * 43758.5453)*2.0-1.0;
}

// Function 112
vec2 noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p); f = f*f*(3.-2.*f); // smoothstep
    vec2 v= mix( mix(hash22(i+vec2(0,0)),hash22(i+vec2(1,0)),f.x),
                  mix(hash22(i+vec2(0,1)),hash22(i+vec2(1,1)),f.x), f.y);
    return 2.*v-1.;
}

// Function 113
float nnoise( in vec2 uv ){return 0.5 + 0.5*snoise(uv);}

// Function 114
vec2 noise( in vec3 x ) {

    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(27.0,11.0)*p.z) + f.xy;
	vec4 t = textureLod( iChannel2, (uv+0.5)/256.0, 0.0);
	return mix( t.xz, t.yw, f.z );
    
}

// Function 115
float noise(const vec3 x)
{
	vec3 p=floor(x);
	vec3 f=fract(x);

    	f=f*f*(3.0-2.0*f);

    	float n=p.x+p.y*57.0+p.z*43.0;

    	float r1=mix(mix(hash(n+0.0),hash(n+1.0),f.x),mix(hash(n+57.0),hash(n+57.0+1.0),f.x),f.y);
    	float r2=mix(mix(hash(n+43.0),hash(n+43.0+1.0),f.x),mix(hash(n+43.0+57.0),hash(n+43.0+57.0+1.0),f.x),f.y);

	return mix(r1,r2,f.z);
}

// Function 116
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

// Function 117
float noise(in float _x) {
    return mix(rand_step(_x-1.), rand_step(_x), smoothstep(0.,1.,fract(_x)));
}

// Function 118
float noise( in vec2 x )
{
  vec2 p = floor(x);
  vec2 f = fract(x);
  f = f*f*(3.0-2.0*f);

  float res = mix(mix( terrainNoise(p), terrainNoise(p + addPos.xy), f.x), 
    mix( terrainNoise(p + addPos.yx), terrainNoise(p + addPos.xx), f.x), f.y);
  return res;
}

// Function 119
float Mnoise( vec2 uv ) {
#  if MODE==0
    return noise(uv);                      // base turbulence
#elif MODE==1
    return -1. + 2.* (1.-abs(noise(uv)));  // flame like
#elif MODE==2
    return -1. + 2.* (abs(noise(uv)));     // cloud like
#endif
}

// Function 120
float PerlinNoise3D(vec3 uv, int octaves) {
    float c = 0.0;
    float s = 0.0;
    for (float i = 0.0; i < float(octaves); i++) {
        c += SmoothNoise3D(uv * pow(2.0, i)) * pow(0.5, i);
        s += pow(0.5, i);
    }
    
    return c /= s;
}

// Function 121
float noise(in vec3 x) {
	vec3 p = floor(x);
	vec3 f = fract(x);
	f = f * f * (3. - 2. * f);
	vec2 uv = (p.xy + vec2(37., 17.) * p.z) + f.xy;
	vec2 rg = textureLod(iChannel0, (uv + .5) / 256., 0.).yx;
	return -1. + 2.4 * mix(rg.x, rg.y, f.z);
}

// Function 122
float Noise2d( in vec2 x )
{
    float xhash = Hash( x.x * 37.0 );
    float yhash = Hash( x.y * 57.0 );
    return fract( xhash + yhash );
}

// Function 123
vec3 noise3( in vec3 x)
{
	return vec3( noise(x+vec3(123.456,.567,.37)),
				noise(x+vec3(.11,47.43,19.17)),
				noise(x) );
}

// Function 124
float noise(vec3 pos, uint octave)
{
    // Octave
    float t = float(octave);

    // Store the Fractional Part of the Coordinate, for Interpolation later
    vec3 f = fract(pos);

    // Floor-ify the Position (so the hash gets nice whole numbers)
    pos = floor(pos);

    // Sample
    float t0 = hash44(vec3(0.0, 0.0, 0.0)+pos, t).x;
    float t1 = hash44(vec3(1.0, 0.0, 0.0)+pos, t).x;
    float t2 = hash44(vec3(0.0, 1.0, 0.0)+pos, t).x;
    float t3 = hash44(vec3(1.0, 1.0, 0.0)+pos, t).x;
    float t4 = hash44(vec3(0.0, 0.0, 1.0)+pos, t).x;
    float t5 = hash44(vec3(1.0, 0.0, 1.0)+pos, t).x;
    float t6 = hash44(vec3(0.0, 1.0, 1.0)+pos, t).x;
    float t7 = hash44(vec3(1.0, 1.0, 1.0)+pos, t).x;

    // Return Interpolated Value
    return mix(mix(mix(t0, t1, f.x), mix(t2, t3, f.x), f.y), mix(mix(t4, t5, f.x), mix(t6, t7, f.x), f.y), f.z);
}

// Function 125
float noise(vec3 p) {
	const vec3 s = vec3(7, 157, 113);
	vec3 ip = floor(p);
	vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
	p -= ip;

	h = mix(fract(sin(h) * 43758.5453), fract(sin(h + s.x) * 43758.5453), p.x);

	h.xy = mix(h.xz, h.yw, p.y);
	return mix(h.x, h.y, p.z);
}

// Function 126
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

// Function 127
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

// Function 128
float noise(vec2 x) { vec2 i = floor(x); vec2 f = fract(x); float a = hash(i); float b = hash(i + vec2(1.0, 0.0)); float c = hash(i + vec2(0.0, 1.0)); float d = hash(i + vec2(1.0, 1.0)); vec2 u = f * f * (3.0 - 2.0 * f); return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y; }

// Function 129
float noise(in vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    
    f = f * f * (3.0 - 2.0 * f);
    
    float n = p.x + p.y * 57.0 + 113.0 * p.z;
    
    float res = mix(mix(mix(hash(n +   0.0), hash(n +   1.0), f.x),
                        mix(hash(n +  57.0), hash(n +  58.0), f.x), f.y),
                    mix(mix(hash(n + 113.0), hash(n + 114.0), f.x),
                        mix(hash(n + 170.0), hash(n + 171.0), f.x), f.y), f.z);
    return res;
}

// Function 130
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return -1.0+2.0*mix(mix(mix( hash(p+vec3(0,0,0)), 
                        hash(p+vec3(1,0,0)),f.x),
                   mix( hash(p+vec3(0,1,0)), 
                        hash(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(p+vec3(0,0,1)), 
                        hash(p+vec3(1,0,1)),f.x),
                   mix( hash(p+vec3(0,1,1)), 
                        hash(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 131
float gnoise(float p) {
    float i = floor(p);
	float f = fract(p);
    
    float a = hash11(i) * f;
    float b = hash11(i + 1.0) * (f - 1.0);
    
    float u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u);
}

// Function 132
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

// Function 133
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

// Function 134
float noise( in vec3 x )
{


    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);

    vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
    vec2 rg = textureLod( iChannel1, (uv+ 0.5)/256.0, 0.0 ).yx;
    return mix( rg.x, rg.y, f.z );
    
}

// Function 135
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

// Function 136
float noise(vec4 p)
{
	vec4 pm = mod(p,1.0);
	vec4 pd = p-pm;
	return hashmix(pd,(pd+vec4(1.0,1.0,1.0,1.0)), pm);
}

// Function 137
vec3 NoiseD( in vec2 x )
{
	x+=4.2;
    vec2 p = floor(x);
    vec2 f = fract(x);

    vec2 u = f*f*(3.0-2.0*f);
	//vec2 u = f*f*f*(6.0*f*f - 15.0*f + 10.0);
    float n = p.x + p.y*57.0;

    float a = Hash(n+  0.0);
    float b = Hash(n+  1.0);
    float c = Hash(n+ 57.0);
    float d = Hash(n+ 58.0);
	return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
				6.0*f*(f-1.0)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));
}

// Function 138
float Noise( in vec2 x ) {
    vec2 p = floor(x);
    vec2 f = fract(x);
	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);
	return textureLod( iChannel0, (uv+118.4)/256.0, -100.0 ).x;
}

// Function 139
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel3, (uv+ 0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 140
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

// Function 141
vec4 noisergba(vec2 uv){
 uv -=floor(uv/289.0)*289.;
 uv +=vec2(223.35734, 550.56781);
 //uv *=uv;
 float xy = uv.x * uv.y;    
 return vec4(sin(fract(xy*.000000064)),
             sin(fract(xy*.000000543)),
             sin(fract(xy*.000000192)),
             sin(fract(xy*.000000423)));}

// Function 142
float Noise(vec3 p, float o)
{
    float result = 0.0f;
    float a = 1.0f;
    float t= 0.0;
    float f = 0.5;
    float s= 2.0f;
    
    p.x += o;
    result += SmoothNoise3d(p) * a; t+= a; p = m3 * p * s; a = a * f;
    p.x += o;
    result += SmoothNoise3d(p) * a; t+= a; p = m3 * p * s; a = a * f;
    p.x += o;
    result += SmoothNoise3d(p) * a; t+= a; p = m3 * p * s; a = a * f;
    p.x += o;
    result += SmoothNoise3d(p) * a; t+= a; p = m3 * p * s; a = a * f;
    result = result / t;
    
    return result;
}

// Function 143
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

// Function 144
float noise(vec2 p){
	return fract(sin(fract(sin(p.x) * (43.13311)) + p.y) * 31.0011);
}

// Function 145
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);
	return textureLod( iChannel1, (uv+118.4)/256.0, 0.0 ).x;
}

// Function 146
float noise2(vec3 pos)
{
    vec3 q = 8.0*pos;
    float f  = 0.5000*noise(q) ; q = m*q*2.01;
    f+= 0.2500*noise(q); q = m*q*2.02;
    f+= 0.1250*noise(q); q = m*q*2.03;
    f+= 0.0625*noise(q); q = m*q*2.01;
    return f;
}

// Function 147
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

// Function 148
vec3 noised(in vec2 x){
 vec2 p=floor(x),f=fract(x),u=f*f*(3.-2.*f);
 float n=p.x+p.y*57.,a=h11(n+ 0.),b=h11(n+ 1.),c=h11(n+57.),d=h11(n+58.);
 return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
  30.*f*f*(f*(f-2.0)+1.0)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));}

// Function 149
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

// Function 150
vec4 bluenoise(vec2 u){//U=floor(U/8.); 
 vec4 n=8./9.*noise(u)-1./9.*(V(-1,-1)+V(0,-1)+V(1,-1)+V(-1,0)+V(1,0)+V(-1,1)+V(0,1)+V(1,1));  
 return n*2.+.5;}

// Function 151
float InterferenceSmoothNoise1D( float x )
{
    float f0 = floor(x);
    float fr = fract(x);

    float h0 = InterferenceHash( f0 );
    float h1 = InterferenceHash( f0 + 1.0 );

    return h1 * fr + h0 * (1.0 - fr);
}

// Function 152
vec3 sunNoise(vec2 uv) {
    float bend = 30.0 * sin(0.1 * iTime) * pow(dot(uv, uv), 2.0);
    vec2 ruv = vec2((abs(fract(atan(uv.x, uv.y) / TAU * 4.0) - 0.5) * 4.0 - 1.0 + bend) * 6.0, length(uv) * 30.0 - iTime);
    float s = (1.0 + 10.0 * noise(ruv, 0.5)) / dot(uv, uv) / 15.0;
    return clamp(vec3(s - 0.5, 0.5 * s - 0.4, s - 3.0), 0.0, 3.0);
}

// Function 153
float noise (vec3 pos){
    float a = rand (pos);
    float b = rand (pos+vec3(1.,0.,0.));
    float c = rand (pos+vec3(0.,1.,0.));
    float d = rand (pos+vec3(1.,1.,0.));
    float e = rand (pos+vec3(0.,0.,1.));
    float f = rand (pos+vec3(1.,0.,1.));
    float g = rand (pos+vec3(0.,1.,1.));
    float h = rand (pos+vec3(1.,1.,1.));
    vec3 u = smoothstep(0.,1.,fract(pos));
                   
    float m1 = mix(mix(a,c,u.y),mix(b,d,u.y),u.x);
	float m2 = mix(mix(e,g,u.y),mix(f,h,u.y),u.x);
    return  mix(m1,m2,u.z);
}

// Function 154
float noise(in vec2 p) 
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 sp = _smoothstep(f);
    return -1.0 + 2.0 * mix(mix(hash(i + vec2(0.0, 0.0)), 
                                hash(i + vec2(1.0, 0.0)), sp.x),
                            mix(hash(i + vec2(0.0, 1.0)), 
                                hash(i + vec2(1.0, 1.0)), sp.x), sp.y);
}

// Function 155
float noise(in vec2 p)
{
    return n(p/32.) * 0.58 +
           n(p/16.) * 0.2  +
           n(p/8.)  * 0.1  +
           n(p/4.)  * 0.05 +
           n(p/2.)  * 0.02 +
           n(p)     * 0.0125;
}

// Function 156
float noise2D(vec2 p, float seed){
    p /= 20.f;
    return fract(5.f * sin(dot(p, p) * seed) - p.y * cos(435.324 * seed * p.x));;
}

// Function 157
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

// Function 158
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

// Function 159
vec3 noise(float p){return texture(iChannel0,vec2(p/iChannelResolution[0].x,.0)).xyz;}

// Function 160
F1 Noise(F2 n,F1 x){n+=x;return fract(sin(dot(n.xy,F2(12.9898, 78.233)))*43758.5453)*2.0-1.0;}

// Function 161
vec4 noise( in vec2 p ) {
    return texture(iChannel1, p, 0.0);
}

// Function 162
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

// Function 163
vec3 NOISE_volumetricRoughnessMap(vec3 p, float rayLen)
{
    vec4 sliderVal = vec4(0.5,0.85,0,0.5);
    ROUGHNESS_MAP_UV_SCALE *= 0.1*pow(10.,2.0*sliderVal[0]);
    
    float f = iTime;
    const mat3 R1  = mat3(0.500, 0.000, -.866,
	                     0.000, 1.000, 0.000,
                          .866, 0.000, 0.500);
    const mat3 R2  = mat3(1.000, 0.000, 0.000,
	                      0.000, 0.500, -.866,
                          0.000,  .866, 0.500);
    const mat3 R = R1*R2;
    p *= ROUGHNESS_MAP_UV_SCALE;
    p = R1*p;
    vec4 v1 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021;
    vec4 v2 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021+1.204*v1.xyz;
    vec4 v3 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021+0.704*v2.xyz;
    vec4 v4 = NOISE_trilinearWithDerivative(p);
    
    return (v1
	      +0.5*(v2+0.25)
	      +0.4*(v3+0.25)
	      +0.6*(v4+0.25)).yzw;
}

// Function 164
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

// Function 165
float noise(in vec3 x)
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = smoothstep(0.0, 1.0, f);
	
	vec2 uv = (p.xy + vec2(37.0, 17.0) * p.z) + f.xy;
	vec2 rg = texture(iChannel1, (uv + 0.5) / 256.0, -100.0).yx;
	return mix(rg.x, rg.y, f.z) * 2.0 - 1.0;
}

// Function 166
uint noise_prng_rand(inout noise_prng this_)
{
    return this_.x_ *= 3039177861u;
}

// Function 167
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

// Function 168
float fractalNoise(in vec3 loc) {
	float n = 0.0 ;
	for (int octave=1; octave<=NUM_OCTAVES; octave++) {	
		n = n + snoise(loc/float(octave*8)) ; 
	}
	return n ;
}

// Function 169
float noise(vec4 v){
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

// Function 170
float tnoise(vec2 p)
{
    return textureLod(iChannel3, p, 0.0).x;
}

// Function 171
vec2 Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
//	vec3 f2 = f*f; f = f*f2*(10.0-15.0*f+6.0*f2);

	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;

// hardware interpolation lacks precision
//	vec4 rg = textureLod( iChannel0, (uv+0.5)/256.0, 0.0 );
	vec4 rg = mix( mix(
				texture( iChannel0, (floor(uv)+0.5)/256.0, -100.0 ),
				texture( iChannel0, (floor(uv)+vec2(1,0)+0.5)/256.0, -100.0 ),
				fract(uv.x) ),
				  mix(
				texture( iChannel0, (floor(uv)+vec2(0,1)+0.5)/256.0, -100.0 ),
				texture( iChannel0, (floor(uv)+1.5)/256.0, -100.0 ),
				fract(uv.x) ),
				fract(uv.y) );
				  

	return mix( rg.yw, rg.xz, f.z );
}

// Function 172
float InterleavedGradientNoise( vec2 uv )
{
    const vec3 magic = vec3( 0.06711056, 0.00583715, 52.9829189 );
    return fract( magic.z * fract( dot( uv, magic.xy ) ) );
}

// Function 173
float bluenoise(vec2 U) {
#define V(i,j)  noise(U+vec2(i,j))
    float N = 8./9.* noise( U ) 
           - 1./9.*( V(-1,-1)+V(0,-1)+V(1,-1) +V(-1,0)+V(1,0) +V(-1,1)+V(0,1)+V(1,1) );  
    return N*2.0+0.5;
}

// Function 174
float interleavedGradientNoise(vec2 n) {
    float f = 0.06711056 * n.x + 0.00583715 * n.y;
    return fract(52.9829189 * fract(f));
}

// Function 175
vec2 curlNoise(vec2 uv)
{
    vec2 eps = vec2(0., 1.);
    
    float n1, n2, a, b;
    n1 = perlinNoise(uv + eps);
    n2 = perlinNoise(uv - eps);
    a = (n1 - n2) / (2. * eps.y);
    
    n1 = perlinNoise(uv + eps.yx);
    n2 = perlinNoise(uv - eps.yx);
    b = (n1 - n2)/(2. * eps.y);
    
    return vec2(a, -b);
}

// Function 176
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

// Function 177
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

// Function 178
float noise( in vec2 p )
{
	return sin(p.x)*sin(p.y);
}

// Function 179
float noise(in vec3 x){vec3 p=floor(x),f=fract(x);
 f=f*f*(3.-2.*f);
 float n=p.x+p.y*57.+113.*p.z;
 return mix(mix(mix(h11(n+  0.),h11(n+  1.),f.x),
                mix(h11(n+ 57.),h11(n+ 58.),f.x),f.y),
            mix(mix(h11(n+113.),h11(n+114.),f.x),
                mix(h11(n+170.),h11(n+171.),f.x),f.y),f.z);}

// Function 180
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

// Function 181
float noise(vec2 uv, float blockiness)
    {
        vec2 lv = fract(uv);
        vec2 id = floor(uv);

        float n1 = random(id);
        float n2 = random(id+vec2(1,0));
        float n3 = random(id+vec2(0,1));
        float n4 = random(id+vec2(1,1));

        vec2 u = smoothstep(0.0, 1.0 + blockiness, lv);

        return mix(mix(n1, n2, u.x), mix(n3, n4, u.x), u.y);
    }

// Function 182
float noise(vec3 x)
{
    //x.x = mod(x.x, 0.4);
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix(hash(n +   0.0), hash(n +   1.0),f.x),
                   mix(hash(n + 157.0), hash(n + 158.0),f.x),f.y),
               mix(mix(hash(n + 113.0), hash(n + 114.0),f.x),
                   mix(hash(n + 270.0), hash(n + 271.0),f.x),f.y),f.z);
}

// Function 183
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

// Function 184
float noise3D01(vec3 p) {
    return fract(sin(dot(p ,vec3(12.9898,78.233,128.852))) * 43758.5453);
}

// Function 185
float noise(in vec3 x)
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

// Function 186
float P_procedural(float x, int i, int level) {
    
    // We use even functions
    x = abs(x);
    // After 4 standard deviation sigma, we consider that the distribution equals zero
    float sigma_dist_4 = 4. * 0.353535; // alpha_dist = 0.5 so sigma_dist \approx 0.3535 (0.5 / sqrt(2))
    if(x >= sigma_dist_4) return 0.;
    
    int nMicrofacetsCurrentLevel = int(pow(2., float(level)));
    float density = 0.;
    // Dictionary should be precomputed, but we cannot use memory with Shadertoy
    // So we generate it on the fly with a very limited number of lobes
    nMicrofacetsCurrentLevel = min(16, nMicrofacetsCurrentLevel);
    
    for (int n = 0; n < nMicrofacetsCurrentLevel; ++n) {
        
        float U_n = hashIQ(uint(i*7333+n*5741));
        // alpha roughness equals sqrt(2) * RMS roughness
        //     ALPHA_DIC     =   1.414214 * std_dev
        // std_dev = ALPHA_DIC / 1.414214 
        float currentMean = sampleNormalDistribution(U_n, 0., ALPHA_DIC / 1.414214);
        density += normalDistribution1D(x, currentMean, 0.05) +
                   normalDistribution1D(-x, currentMean, 0.05);
    }
    return density / float(nMicrofacetsCurrentLevel);
}

// Function 187
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

// Function 188
float Noise11(float x)
{
    float p = floor(x);
    float f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix( hash11(p), hash11(p + 1.0), f)-.5;

}

// Function 189
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

// Function 190
float noiseFunction(vec2 pos)
{
    return texture(iChannel0, pos - 0.5).x * 2.0 - 1.0;
}

// Function 191
float remap_noise_tri_erp( const float v )
{
    float r2 = 0.5 * v;
    float f1 = sqrt( r2 );
    float f2 = 1.0 - sqrt( r2 - 0.25 );    
    return (v < 0.5) ? f1 : f2;
}

// Function 192
float noise(float p)
{
	float pm = mod(p,1.0);
	float pd = p-pm;
	return hashmix(pd,pd+1.0,pm);
}

// Function 193
vec3 noise3( in float x )
{
    float p = floor(x);
    float f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix( hash3(p+0.0), hash3(p+1.0), f );
}

// Function 194
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

// Function 195
float noise(vec2 p)
{
    vec2 f = fract( p ), i = p-f, u = (3.-2.*f)*f*f;

    return mix( mix( dot( hash2( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( hash2( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hash2( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( hash2( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 196
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

// Function 197
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

// Function 198
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

// Function 199
float noised( in vec2 x ){
    vec2 f = fract(x);
    vec2 u = f*f*(3.0-2.0*f);
  
    vec2 p = floor(x);
	float a = textureLod( iChannel0, (p+vec2(0.5,0.5))/256.0, 0.0 ).x;
	float b = textureLod( iChannel0, (p+vec2(1.5,0.5))/256.0, 0.0 ).x;
	float c = textureLod( iChannel0, (p+vec2(0.5,1.5))/256.0, 0.0 ).x;
	float d = textureLod( iChannel0, (p+vec2(1.5,1.5))/256.0, 0.0 ).x;
    
	float res = (a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y);
    res = res - 0.5;
    return res;
}

// Function 200
float noise(  vec2 p )
{
    vec2 i = floor( p ), f = fract( p ), u = f*f*(3.-2.*f);
    return mix( mix( dot( hash( i + vec2(0,0) ), f - vec2(0,0) ), 
                     dot( hash( i + vec2(1,0) ), f - vec2(1,0) ), u.x),
                mix( dot( hash( i + vec2(0,1) ), f - vec2(0,1) ), 
                     dot( hash( i + vec2(1,1) ), f - vec2(1,1) ), u.x), u.y);
}

// Function 201
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

// Function 202
float perlinNoise(vec2 pos, vec2 scale, float rotation, float seed) 
{
    vec2 sinCos = vec2(sin(rotation), cos(rotation));
    return perlinNoise(pos, scale, mat2(sinCos.y, sinCos.x, sinCos.x, sinCos.y), seed);
}

// Function 203
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

// Function 204
float rnoise( in vec2 uv ){return 1. - abs(snoise(uv));}

// Function 205
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

// Function 206
vec4 TurbulenceNoise( in vec3 p )
{
	float f=noise(p);
	vec3 n=GrNoise(p);
	return vec4(-sign(f)*n,-abs(f));
}

// Function 207
float noise(vec2 uv) {
	return clamp(texture(iChannel1, uv.xy + iTime*6.0).r +
		texture(iChannel1, uv.xy - iTime*4.0).g, 0.96, 1.0);
}

// Function 208
vec3 noise(vec2 uv) {
    // http://obge.paradice-insight.us/wiki/Includes_%28Effects%29
    vec2 pos = floor(uv * 16.0);
    float noise = fract(sin(iTime+dot(pos ,vec2(12.9898,78.233)*2.0)) * 43758.5453);
	float r = abs(noise);
    return vec3(r, r, r);
}

// Function 209
vec3 noised( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    vec2 u = f*f*(3.0-2.0*f);

    float n = p.x + p.y*57.0;

    float a = hash(n+  0.0);
    float b = hash(n+  1.0);
    float c = hash(n+ 57.0);
    float d = hash(n+ 58.0);
	return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
				30.0*f*f*(f*(f-2.0)+1.0)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));

}

// Function 210
float valueNoise(vec3 uv){
    vec3 id = floor(uv);
    vec3 fd = fract(uv);
    fd = smoothstep(0.,1., fd);
    
    float ibl = hash13(id + vec3(0,-1,0));
    float ibr = hash13(id + vec3(1,-1,0));
    float itl = hash13(id + vec3(0));
    float itr = hash13(id + vec3(1,0,0));
    
    
    float jbl = hash13(id + vec3(0,-1,1));
    float jbr = hash13(id + vec3(1,-1,1));
    float jtl = hash13(id + vec3(0,0, 1));
    float jtr = hash13(id + vec3(1,0, 1));
    
    
    float ibot = mix(ibl, ibr, fd.x); 
    float iup = mix(itl, itr, fd.x);
    float jbot = mix(jbl, jbr, fd.x);
    float jup = mix(jtl, jtr, fd.x);
    
    float i = mix(ibot, iup, fd.y);
    float j = mix(jbot, jup, fd.y);
    
    return mix(i, j, fd.z); 
}

// Function 211
float noise(in vec3 p)
{
	vec3 ip = floor(p);
    vec3 f = fract(p);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (ip.xy+vec2(37.0,17.0)*ip.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix(rg.x, rg.y, f.z);
}

// Function 212
float SampleNoiseFractal(vec3 p)
{
    float h = noisedFractal(p).x;
    h += noisedFractal(p*2. + 100.).x * 0.5;
    h += noisedFractal(p*4. - 100.).x * 0.25;
    h += noisedFractal(p*8. + 1000.).x * 0.125;
    return h / (1.865);
}

// Function 213
float cosNoise( in vec2 pos)
{
	return 0.5 * ( sin(pos.x) * sin(pos.y));   
}

// Function 214
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

// Function 215
float noise(vec2 p){return hash(mod(p.x+p.y*57.,1024.0));}

// Function 216
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);

	vec2 uv = (p.xy+vec2(37.0,239.0)*p.z) + f.xy;
    vec2 rg = textureLod(iChannel1,(uv+0.5)/256.0,0.0).yx;
	return mix( rg.x, rg.y, f.z )*2.0-1.0;
}

// Function 217
float valueNoiseFilter(float x) {
    #if defined(VALUE_NOISE_FILTER_QUINTIC)
    return x*x*x*(x*(x*6.-15.)+10.);
    #elif defined(VALUE_NOISE_FILTER_SMOOTH)
    return smoothstep(0.0,1.0,x);
    #else
    return x;
    #endif
}

// Function 218
float noise(vec2 p) {
    return snoise(p);// + snoise(p + snoise(p));
//    return (snoise(p) * 64.0 + snoise(p * 2.0) * 32.0 + snoise(p * 4.0) * 16.0 + snoise(p * 8.0) * 8.0 + snoise(p * 16.0) * 4.0 + snoise(p * 32.0) * 2.0 + snoise(p * 64.0)) / (1.0 + 2.0 + 4.0 + 8.0 + 16.0 + 32.0 + 64.0);
}

// Function 219
vec3 noised( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    vec2 u = f*f*(3.0-2.0*f);
	float a = texture(iChannel0,(p+vec2(0.5,0.5))/256.0,-100.0).x;
	float b = texture(iChannel0,(p+vec2(1.5,0.5))/256.0,-100.0).x;
	float c = texture(iChannel0,(p+vec2(0.5,1.5))/256.0,-100.0).x;
	float d = texture(iChannel0,(p+vec2(1.5,1.5))/256.0,-100.0).x;
	return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
				6.0*f*(1.0-f)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));
}

// Function 220
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

// Function 221
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

// Function 222
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

// Function 223
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel1, (uv+ 0.5)/256.0, -100.0 ).yx;
	return -1.0+2.0*mix( rg.x, rg.y, f.z );
}

// Function 224
float noise(vec2 p) {
    return random(p.x + p.y*10000.0);
}

// Function 225
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

// Function 226
float noise (vec2 co) {
  return length (texture (iChannel0, co));
}

// Function 227
float Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 228
vec4 noiseInterpolateDu(const in vec2 x) 
{ 
    vec2 x2 = x * x;
    vec2 u = x2 * x * (x * (x * 6.0 - 15.0) + 10.0); 
    vec2 du = 30.0 * x2 * (x * (x - 2.0) + 1.0);
    return vec4(u, du);
}

// Function 229
float noise( in vec2 p ){
   
    vec2 i = floor(p); p -= i; 
    p *= p*p*(p*(p*6. - 15.) + 10.);
    //p *= p*(3. - p*2.);  

    return mix( mix( hash21(i + vec2(0, 0)), 
                     hash21(i + vec2(1, 0)), p.x),
                mix( hash21(i + vec2(0, 1)), 
                     hash21(i + vec2(1, 1)), p.x), p.y);
}

// Function 230
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

// Function 231
vec3 noise(float p, float lod){return texture(iChannel0,vec2(p/iChannelResolution[0].x,.0),lod).xyz;}

// Function 232
float noise3(vec3 v){
    v *= 64.; // emulates 64x64 noise texture
    return ( perlin(v) + perlin(v+.5) )/2.;
}

// Function 233
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

// Function 234
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

// Function 235
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
	vec2 uv = p.xy + f.xy*f.xy*(3.0-2.0*f.xy);
	return textureLod( iChannel0, (uv+118.4)/256.0, 0. ).x;
}

// Function 236
vec4 noisedFractal( in vec3 x )
{
    // grid
    vec3 i = floor(x);
    vec3 w = fract(x);
    
    // cubic interpolant
    vec3 u = w*w*(3.0-2.0*w);
    vec3 du = 6.0*w*(1.0-w);
    
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

// Function 237
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

// Function 238
vec2 Noise21(float x)
{
    float p = floor(x);
    float f = fract(x);
    f = f*f*(3.0-2.0*f);
    return  mix( hash21(p), hash21(p + 1.0), f)-.5;
    
}

// Function 239
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

// Function 240
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

// Function 241
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

// Function 242
void staticNoise(inout vec2 p, vec2 groupSize, float grainSize, float contrast) {
    GlitchSeed seedA = glitchSeed(glitchCoord(p, groupSize), 5.);
    seedA.prob *= .5;
    if (shouldApply(seedA) == 1.) {
        GlitchSeed seedB = glitchSeed(glitchCoord(p, vec2(grainSize)), 5.);
        vec2 offset = vec2(rand(seedB.seed), rand(seedB.seed + .1));
        offset = round(offset * 2. - 1.);
        offset *= contrast;
        p += offset;
    }
}

// Function 243
float noise(vec3 x) { const vec3 step = vec3(110, 241, 171); vec3 i = floor(x); vec3 f = fract(x); float n = dot(i, step); vec3 u = f * f * (3.0 - 2.0 * f); return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x), mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y), mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x), mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z); }

// Function 244
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

// Function 245
float noise(vec3 p)
{
    return textureLod(iChannel1, p, 0.0).x;
}

// Function 246
float noise3( mediump vec3 x )
{
    mediump vec3 p = floor(x);
    lowp vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    mediump float n = p.x + dot(p.yz,vec2(157.0,113.0));
    lowp vec4 s1 = mix(hash4(vec4(n)+NC0),hash4(vec4(n)+NC1),f.xxxx);
    return mix(mix(s1.x,s1.y,f.y),mix(s1.z,s1.w,f.y),f.z);
}

// Function 247
float noise( in vec3 x ) { // base noise in [0,1]; 
    x += 2.*time*(1.,2.,1.);
    vec3 p = floor(x), f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0 + 113.0*p.z;
    float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                        mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                    mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                        mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
#if NOISE_TYPE==1
	return res;
#elif NOISE_TYPE==2
	return abs(2.*res-1.);
#elif NOISE_TYPE==3
	return 1.-abs(2.*res-1.);
#endif
}

// Function 248
vec4 noise( in vec2 p ) {
	return texture(iChannel1, p, 0.0);
}

// Function 249
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

// Function 250
float pnoise1D(in float p) {
    float pi = permTexUnit*floor(p) + permTexUnitHalf;
    float pf = fract(p);
    float grad00 = texture(iChannel0, vec2(pi, 0.0), -10.).r * 4.0 - 1.0;
    float n00 = dot(grad00, pf);
    float grad10 = texture(iChannel0, pi + vec2(permTexUnit, 0.0), -10.).r * 4.0 - 1.0;
    float n10 = dot(grad10, pf - 1.0);
    float n = mix(n00, n10, fade(pf));

    return n;
}

// Function 251
float noiseTex(in vec3 x)
{
    vec3 fl = floor(x);
    vec3 fr = fract(x);
	fr = fr * fr * (3.0 - 2.0 * fr);
	vec2 uv = (fl.xy + vec2(37.0, 17.0) * fl.z) + fr.xy;
	vec2 rg = textureLod(iChannel0, (uv + 0.5) * 0.00390625, 0.0 ).xy;
	return mix(rg.y, rg.x, fr.z);
}

// Function 252
float perlinNoise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);

	float a = hash12(i);
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Function 253
float noise3d(in vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    vec2 uv = (p.xy + vec2(37.0, 17.0) * p.z) + f.xy;
    vec2 rg = texture(iChannel1, (uv + 0.5) / 256.0, -100.0).yx;
    return mix(rg.x, rg.y, f.z);
}

// Function 254
float noise222( mediump vec2 x, mediump vec2 y, mediump vec2 z )
{
    mediump vec4 lx = vec4(x*y.x,x*y.y);
    mediump vec4 p = floor(lx);
    lowp vec4 f = fract(lx);
    f = f*f*(3.0-2.0*f);
    mediump vec2 n = p.xz + p.yw*157.0;
    lowp vec4 h = mix(hash4(n.xxyy+NC0.xyxy),hash4(n.xxyy+NC1.xyxy),f.xxzz);
    return dot(mix(h.xz,h.yw,f.yw),z);
}

// Function 255
float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( random2(i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( random2(i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( random2(i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( random2(i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 256
float noise(vec3 x) { 
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

  return -1.0+2.0*l3candidate1;
}

// Function 257
float bnoise(in vec3 p)
{
    float n = sin(triNoise3d(p*.3,0.0)*11.)*0.6+0.4;
    n += sin(triNoise3d(p*1.,0.05)*40.)*0.1+0.9;
    return (n*n)*0.003;
}

// Function 258
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

// Function 259
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

// Function 260
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

// Function 261
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

// Function 262
float noise(vec2 v) {
    vec2 V = floor(v);v-=V;
    return mix(mix(hash(V),hash(V+E.xy),v.x),mix(hash(V+E.yx),hash(V+E.xx),v.x),v.y);
}

// Function 263
float noise(vec2 U) {
    U-=.5; return hash(uint(U.x+iResolution.x*U.y));
}

// Function 264
float noise(vec2 n)
{
	vec2 d = vec2(0.0, 1.0);
	vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
	return mix(mix(rand(b + d.xx), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

// Function 265
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

// Function 266
float SampleNoise(vec3 p)
{
    float h = noise(p) * 1.1 - 0.1;
    h *= noise(p / 2.);
    return h;
}

// Function 267
float noiseS(vec2 p){
    vec2 s = vec2(0.123,0.213);
    vec2 c = vec2(0.736,0.564);
    return (noise(p)+noise(p+s)+noise(p+c)+noise(p+s.yx)+noise(p+c.yx))/5.;
}

// Function 268
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

// Function 269
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

// Function 270
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

// Function 271
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

// Function 272
float noise( in vec2 x ){return texture(iChannel0, x*.01).x;}

// Function 273
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

// Function 274
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

// Function 275
vec2 noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p); f = f*f*(3.-2.*f); // smoothstep
    vec2 v= mix( mix(hash22(i+vec2(0,0)),hash22(i+vec2(1,0)),f.x),
                 mix(hash22(i+vec2(0,1)),hash22(i+vec2(1,1)),f.x), f.y);
    return 2.*v-1.;
}

// Function 276
float snoise(in vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

// Function 277
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

// Function 278
float noise( in float p )
{
    return noise(vec2(p, 0.0));        
}

// Function 279
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( hash12( i + vec2(0.0,0.0) ), 
                     hash12( i + vec2(1.0,0.0) ), u.x),
                mix( hash12( i + vec2(0.0,1.0) ), 
                     hash12( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 280
float noise( in vec2 p ,in vec2 md)
{
    
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash( p + vec2(0.0,0.0) ,md), f - vec2(0.0,0.0) ), 
                     dot( hash( p + vec2(1.0,0.0),md ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hash( p + vec2(0.0,1.0) ,md), f - vec2(0.0,1.0) ), 
                     dot( hash( p + vec2(1.0,1.0) ,md), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 281
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

// Function 282
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

// Function 283
vec3 noise(vec2 p){return texture(iChannel0,p/iChannelResolution[0].xy).xyz;}

// Function 284
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

// Function 285
float tnoise(in vec2 c)
{   
  c = c*fract(c*vec2(W0,W1));
  float p  = c.x*c.y;
  float i  = floor(p);
  float u0 = p-i;
  float u1 = fract(W3*i);
  return u0-u1;
}

// Function 286
vec2 noise2_2( vec2 p )     
{
	vec3 pos = vec3(p,.5);
	if (ANIM) pos.z += time;
	pos *= m;
    float fx = noise(pos);
    float fy = noise(pos+vec3(1345.67,0,45.67));
    return vec2(fx,fy);
}

// Function 287
float noise( in float p )
{
    return noise(vec2(p, 0.0));
}

// Function 288
float correlated_bluenoise(vec2 U) {
    return mix(bluenoise(U),
               bluenoise(mod(U,64.)),
               0.5)/0.75;
}

// Function 289
float gold_noise(in vec2 xy, in float seed)
{
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}

// Function 290
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

// Function 291
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

// Function 292
vec4 noise(vec4 c,vec2 px)
{
    vec2 uv = px / iResolution.xy;
    
    vec4 r = texture(iChannel0,uv+vec2(sin(iTime*10.0),sin(iTime*20.0)));
    
    c += r * 0.2; 

    return c;
}

// Function 293
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

// Function 294
float noise( in vec2 x ) {
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*157.0;
    return mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
               mix( hash(n+157.0), hash(n+158.0),f.x),f.y);
}

// Function 295
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

// Function 296
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

// Function 297
float cnoise(vec3 p)
{
    vec3 size = 1.0 / vec3(textureSize(iChannel1, 0));
    return (
        noise(p * size * 1.0 + vec3(0.52, 0.78, 0.43)) * 0.5 + 
        noise(p * size * 2.0 + vec3(0.33, 0.30, 0.76)) * 0.25 + 
        noise(p * size * 4.0 + vec3(0.70, 0.25, 0.92)) * 0.125) * 1.14;
}

// Function 298
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

// Function 299
float noise2D(vec2 x) {
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

  return -1.0+2.0*noise2d;
}

// Function 300
float noise1s( in float x )
{
	x -= 0.5;

	float x0 = floor( x );
	float y0 = hash11( x0 );
	float y1 = hash11( x0 + 1.0 );

	return mix( y0, y1, smoothstep_unchecked( x - x0 ) );
}

// Function 301
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

// Function 302
float noise( vec2 p ) {
	vec2 i = floor( p );
	vec2 f = fract( p );
	
	f = f * f * ( 3.0 - 2.0 * f );
	
	return mix(
		mix( hash( i + vec2( 0.0, 0.0 ) ), hash( i + vec2( 1.0, 0.0 ) ), f.x ),
		mix( hash( i + vec2( 0.0, 1.0 ) ), hash( i + vec2( 1.0, 1.0 ) ), f.x ),
		f.y
	);
}

// Function 303
float noise( vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel0, (uv + 0.5)/256.0, -100.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 304
vec3 valueNoiseFilter(vec3 x) {
    #if defined(VALUE_NOISE_FILTER_QUINTIC)
    return x*x*x*(x*(x*6.-15.)+10.);
    #elif defined(VALUE_NOISE_FILTER_SMOOTH)
    return smoothstep(0.0,1.0,x);
    #else
    return x;
    #endif
}

// Function 305
float GetNoise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    return mix(mix(mix( GetHash(p+vec3(0,0,0)), 
                        GetHash(p+vec3(1,0,0)),f.x),
                   mix( GetHash(p+vec3(0,1,0)), 
                        GetHash(p+vec3(1,1,0)),f.x),f.y),
               mix(mix( GetHash(p+vec3(0,0,1)), 
                        GetHash(p+vec3(1,0,1)),f.x),
                   mix( GetHash(p+vec3(0,1,1)), 
                        GetHash(p+vec3(1,1,1)),f.x),f.y),f.z);
}

// Function 306
float noise (vec3 n) 
{ 
	return fract(sin(dot(n, vec3(95.43583, 93.323197, 94.993431))) * 65536.32);
}

// Function 307
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

// Function 308
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

// Function 309
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

// Function 310
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

// Function 311
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

// Function 312
float noise( in vec2 p ) {

    ivec2 i = ivec2(floor( p ));
    vec2 f = fract( p );
	
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);

    return mix( mix( dot( grad( i+ivec2(0,0) ), f-vec2(0.0,0.0) ), 
                     dot( grad( i+ivec2(1,0) ), f-vec2(1.0,0.0) ), u.x),
                mix( dot( grad( i+ivec2(0,1) ), f-vec2(0.0,1.0) ), 
                     dot( grad( i+ivec2(1,1) ), f-vec2(1.0,1.0) ), u.x), u.y);

}

// Function 313
vec2 noiseInterpolate(const in vec2 x) 
{ 
    vec2 x2 = x * x;
    return x2 * x * (x * (x * 6.0 - 15.0) + 10.0); 
}

// Function 314
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

// Function 315
float bluenoise(vec2 uv)
{
    uv += 1337.0*fract(iTime);
    float v = texture( iChannel2 , (uv + 0.5) / iChannelResolution[2].xy, 0.0).x;
    return v;
}

// Function 316
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

// Function 317
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

// Function 318
vec4 valueNoise(vec2 t, float w){
    vec2 fr = fract(t);
	return 
        mix(
            mix( 
                texture(iChannel0,vec2(floor(t.x), floor(t.y))/256.),
                texture(iChannel0,vec2(floor(t.x), floor(t.y) + 1.)/256.),
            	smoothstep(0.,1.,fr.y)
            ),
            mix( 
                texture(iChannel0,vec2(floor(t.x) + 1.,floor(t.y))/256.),
                texture(iChannel0,vec2(floor(t.x) + 1.,floor(t.y) + 1.)/256.),
            	smoothstep(0.,1.,fr.y)
            ),
            smoothstep(0.,1.,pow(fr.x, w)));
}

// Function 319
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

// Function 320
float noise2(vec3 pos)
{
    vec3 q = 8.0*pos;
    float f  = 0.5000*noise( q ); q = m*q*2.01;
    f += 0.2500*noise( q ); q = m*q*2.02;
    f += 0.1250*noise( q ); q = m*q*2.03;
    f += 0.0625*noise( q ); q = m*q*2.01;
    return f;
}

// Function 321
float noise(vec3 p)
{
	vec3 ip = floor(p);
    p -= ip; 
    vec3 s = vec3(7,157,113);
    vec4 h = vec4(0.,s.yz,s.y+s.z)+dot(ip,s);
    p = p*p*(3.-2.*p); 
    h = mix(fract(sin(h)*43758.5),fract(sin(h+s.x)*43758.5),p.x);
    h.xy = mix(h.xz,h.yw,p.y);
    return mix(h.x,h.y,p.z); 
}

// Function 322
float noise(float x)
{  
    return texture(iChannel1, vec2(x,x)).x;
}

// Function 323
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

// Function 324
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

// Function 325
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

// Function 326
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

// Function 327
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

// Function 328
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

// Function 329
float interpolatedNoise2D(vec2 xy, int seed) {
    vec2 p = floor(xy);
    vec2 f = fract(xy);
    f = f*f*(3.0-2.0*f);

    float n = dot(vec3(p.xy, seed), vec3(1, 157, 141));
    return mix(mix(hash(n+  0.0), hash(n+  1.0),f.x),
               mix(hash(n+157.0), hash(n+158.0),f.x),f.y);
}

// Function 330
vec2 Noise2D( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec2 res = mix(mix( Hash(p + 0.0), Hash(p + vec2(1.0, 0.0)),f.x),
                   mix( Hash(p + vec2(0.0, 1.0) ), Hash(p + vec2(1.0, 1.0)),f.x),f.y);
    return res-.5;
}

// Function 331
vec4 noise(float t){return texture(iChannel0,vec2(floor(t), floor(t))/256.);}

// Function 332
float noise21(vec2 p)
{
	vec2 ip = floor(p);
	vec2 u  = fract(p);
	u = u * u * (3.0 - 2.0 * u);
	
	float res = mix(
		mix(rand21(ip), rand21(ip + vec2(1.0, 0.0)), u.x),
		mix(rand21(ip + vec2(0.0, 1.0)), rand21(ip + vec2(1.0, 1.0)), u.x), u.y);

	return res * res;
}

// Function 333
vec3 noise(vec3 p){float m = mod(p.z,1.0);float s = p.z-m; float sprev = s-1.0;if (mod(s,2.0)==1.0) { s--; sprev++; m = 1.0-m; };return mix(texture(iChannel0,p.xy/iChannelResolution[0].xy+noise(sprev).yz).xyz,texture(iChannel0,p.xy/iChannelResolution[0].xy+noise(s).yz).xyz,m);}

// Function 334
float meuNoise(float p){
    float ts[5];// = {meuHash(p-2.),meuHash(p-1.),meuHash(p),meuHash(p+1.),meuHash(p+2.)};
   ts[0] = meuHash(p-2.);
  ts[1] = meuHash(p-1.);
 ts[2] = meuHash(p-0.);
ts[3] = meuHash(p+1.);
ts[4] = meuHash(p+2.);
    
    return mix(mix(ts[0],ts[1],0.9),mix(ts[3],ts[4],0.1),0.5);
        
       
    
}

// Function 335
float terrainNoise(vec2 p)
{
  vec3 p3  = fract(vec3(p.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 19.19);
  return fract((p3.x + p3.y) * p3.z);
}

// Function 336
float noise1(vec2 p) {
    #ifndef BILINEAR
		return hash(floor(p));
    #else    
        vec2 i = floor(p);
        vec2 f = fract(p);
    	vec2 tx = mix(vec2(hash(i),hash(i+vec2(0.,1.))) ,
                      vec2(hash(i+vec2(1.,0.)),hash(i+vec2(1.))),f.x);
        return mix(tx.x,tx.y,f.y);
    #endif
}

// Function 337
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

// Function 338
float noise1( in vec2 x )
{
  vec2 p  = floor(x);
  vec2 f  = smoothstep(0.0, 1.0, fract(x));
  float n = p.x + p.y*57.0;
  return mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
    mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);
}

// Function 339
float GetWaterNoise(vec3 position, float time)
{
    return WaterTurbulence * fbm_4(position / 15.0 + time / 3.0);
}

// Function 340
vec3 SampleNoiseV3(vec3 p)
{
    // sampling at decreasing scale and height multiple tiles and returning that amount divided by the total possible amount (to normalize it)
    vec3 h = noisedFractal(p).xyz;
    h += noisedFractal(p*2. + 100.).xyz * 0.5;
    h += noisedFractal(p*4. - 100.).xyz * 0.25;
    h += noisedFractal(p*8. + 1000.).xyz * 0.125;
    return h * 0.536193029;
}

// Function 341
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

// Function 342
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

// Function 343
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

// Function 344
vec2 noise2(vec2 location, vec2 delta) 
{
    const vec2 c = vec2(12.9898, 78.233);
    const float m = 43758.5453;
    return vec2(
        fract(sin(dot(location +      delta            , c)) * m),
        fract(sin(dot(location + vec2(delta.y, delta.x), c)) * m)
        );
}

// Function 345
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

// Function 346
float noise2D( in vec2 pos )
{
  return noise2D(pos, 0.0);
}

// Function 347
float noise(vec2 n) {
    const vec2 d = vec2(0.0, 1.0);
    vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
    return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

// Function 348
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

// Function 349
float noise( in vec3 x )
{
	return texture( iChannel0, (x.xy + x.z*37.0 + 0.5)/256.0, 0.0 ).x*0.75;
}

// Function 350
float noise( in vec3 x )
{
  vec3 p = floor(x);
  vec3 f = fract(x);

  float a = textureLod( iChannel1, x.xy/64.0 + (p.z+0.0)*120.7123, 0.1 ).x;
  float b = textureLod( iChannel1, x.xy/64.0 + (p.z+1.0)*120.7123, 0.1 ).x;
  return mix( a, b, f.z );
}

// Function 351
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

// Function 352
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

// Function 353
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

// Function 354
float noise(in vec2 x) {
	vec2 p = floor(x);
	vec2 f = fract(x);
		
	f = f*f*(3.0-2.0*f);	
	float n = p.x + p.y*57.0;
	
	float res = mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
					mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);
	return res;
}

// Function 355
vec4 Noise( in ivec2 x )
{
	return texture( iChannel0, (vec2(x)+0.5)/256.0, -100.0 );
}

// Function 356
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*57.0;
    return mix(mix( hash1(n+  0.0), hash1(n+  1.0),f.x),
               mix( hash1(n+ 57.0), hash1(n+ 58.0),f.x),f.y);
}

// Function 357
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

// Function 358
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

// Function 359
vec2 noise2(vec2 x)
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    return mix(mix( hash2(p),          hash2(p + add.xy),f.x),
                    mix( hash2(p + add.yx), hash2(p + add.xx),f.x),f.y);
    
}

// Function 360
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

// Function 361
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

// Function 362
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

// Function 363
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

// Function 364
float gradientNoise(in vec2 uv)
{
    const vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(uv, magic.xy)));
}

// Function 365
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

// Function 366
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

// Function 367
vec2 noiseStackUV(vec3 pos,int octaves,float falloff,float diff){
	float displaceA = noiseStack(pos,octaves,falloff);
	float displaceB = noiseStack(pos+vec3(3984.293,423.21,5235.19),octaves,falloff);
	return vec2(displaceA,displaceB);
}

// Function 368
float noise(vec2 p)
{
	vec2 pm = mod(p,1.0);
	vec2 pd = p-pm;
	return hashmix(pd,(pd+vec2(1.0,1.0)), pm);
}

// Function 369
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

// Function 370
float snoise(vec3 x)
{
    //float n=dot(x,vec3(1.0,23.0,244.0));
    //return fract(sin(n)*1399763.5453123);
    return fract((x.x+x.y)*0.5);
}

// Function 371
float mul_noise(vec3 x) {
    float n = 2.*noise(x);  x *= 2.1; // return n/2.;
         n *= 2.*noise(x);  x *= 1.9;
         n *= 2.*noise(x);  x *= 2.3;
         n *= 2.*noise(x);  x *= 1.9;
         n *= 2.*noise(x);
    return n/2.; 
}

// Function 372
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

// Function 373
float noise (vec2 st){
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = R(i);
    float b = R((i + vec2(1.0, 0.0)));
    float c = R((i + vec2(0.0, 1.0)));
    float d = R((i + vec2(1.0, 1.0)));

    vec2 u = smoothstep(0.,1.,f);

    return (mix(a, b, u.x) +
            (c - a) * u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y) * 2.0 - 1.0;
}

// Function 374
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

// Function 375
float iqnoise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    float inv_v = 1.0-v;
    inv_v *= inv_v;
    inv_v *= inv_v;
		
	//float k = 63.0*pow(1.0-v,4.0) + 1.0;
    float k = 63.0*inv_v + 1.0;
	
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

// Function 376
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

// Function 377
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

// Function 378
float noise( const in vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 379
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

// Function 380
float Noisefv2 (vec2 p)
{
  vec4 t;
  vec2 i, f;
  i = floor (p);  f = fract (p);  f = f * f * (3. - 2. * f);
  t = Hashv4f (dot (i, cHashA3.xy));
  return mix (mix (t.x, t.y, f.x), mix (t.z, t.w, f.x), f.y);
}

// Function 381
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

// Function 382
float noise( in vec2 p ) {                     // noise in [-1,1]
    // p+= iTime;
    vec2 i = floor(p), f = fract(p);
	vec2 u = f*f*(3.-2.*f);
    return mix( mix( dot( hash( i + vec2(0.,0.) ), f - vec2(0.,0.) ), 
                     dot( hash( i + vec2(1.,0.) ), f - vec2(1.,0.) ), u.x),
                mix( dot( hash( i + vec2(0.,1.) ), f - vec2(0.,1.) ), 
                     dot( hash( i + vec2(1.,1.) ), f - vec2(1.,1.) ), u.x), u.y);
}

// Function 383
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

// Function 384
float fogNoise(in vec3 p)
{
    float z = 2.1;
    p += tri3(p);
    return tri(p.z + tri(p.x + tri(p.y))) / z;
}

// Function 385
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

// Function 386
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

// Function 387
vec2 Noise( in ivec3 x )
{
	vec2 uv = vec2(x.xy)+vec2(37.0,17.0)*float(x.z);
	return texture( iChannel0, (uv+0.5)/256.0, -100.0 ).xz;
}

// Function 388
vec4 Mnoise( vec4 N ) {   // apply non-linearity 1 (per scale) after blending
#  if MODE==0
    return N;                      // base turbulence
#elif MODE==1
    return -1. + 2.* (1.-abs(N));  // flame like
#elif MODE==2
    return -1. + 2.* (abs(N));     // cloud like
#endif
}

// Function 389
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

// Function 390
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

// Function 391
float noise( in vec2 p ) {
    vec2 i = floor(p), f = fract(p);
	vec2 u = f*f*(3.-2.*f);
    return mix( mix( dot( hash( i + vec2(0.,0.) ), f - vec2(0.,0.) ), 
                     dot( hash( i + vec2(1.,0.) ), f - vec2(1.,0.) ), u.x),
                mix( dot( hash( i + vec2(0.,1.) ), f - vec2(0.,1.) ), 
                     dot( hash( i + vec2(1.,1.) ), f - vec2(1.,1.) ), u.x), u.y);
}

// Function 392
vec2 noise3_2( mediump vec3 x ) { return vec2(noise3(x),noise3(x+100.0)); }

// Function 393
float LineNoise(float x, float t)
{
    float n = Noise2(vec2(x * 0.6, t * 0.2));
    //n += Noise2(vec2(x * 0.8, t * 0.2 + 34.8)) * 0.5;
    //n += Noise2(vec2(x * 1.2, t * 0.3 + 56.8)) * 0.25;
    
    return n - (1.0) * 0.5;
}

// Function 394
float noise( in vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

	vec2 i = floor( p + (p.x+p.y)*K1 );
	
    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
	vec2 c = a - 1.0 + 2.0*K2;

    vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );

	vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));

    return dot( n, vec3(70.0) );
	
}

// Function 395
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

// Function 396
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

// Function 397
float stepNoise(vec2 p) {
    return noise(floor(p));
}

// Function 398
float proceduralSplatter(vec2 st, float radius, float numCircles){
    float pct = 0.;
    st.x -= .5;
    for (float i = 1.; i < numCircles; i++){
        st.y -=(.3/ (i+1.));
        pct +=smoothstep(radius * 1./i, radius * 1./i - .1, length(st));
    }
    return pct;
}

// Function 399
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

// Function 400
float noise2d(vec2 p)
{
    float t = texture(iChannel0, p).x;
    t += 0.5 * texture(iChannel0, p * 2.0).x;
    t += 0.25 * texture(iChannel0, p * 4.0).x;
    return t / 1.75;
}

// Function 401
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

// Function 402
float noise(vec2 p, float r2) {
    float k = 0.0;
    float o = 1.0;
    for (float z = 0.0; z < L; ++z) {
        k += noise(p / o, r2, z) * o;
        o *= 2.0;
    }
    return k;
}

// Function 403
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

// Function 404
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

// Function 405
float noise(vec2 uv) {
    return fract(sin(dot(uv,vec2(32.23,365.07))) * 17.0125);
}

// Function 406
float noise(vec2 p)
{
    vec2 i = floor(p), f = fract(p); 
	f *= f*(3.-2.*f);
    
    vec2 c = vec2(0,1);
    
    return mix(mix(hash(i + c.xx), 
                   hash(i + c.yx), f.x),
               mix(hash(i + c.xy), 
                   hash(i + c.yy), f.x), f.y);
}

// Function 407
float noise3( vec3 p ) {
    vec3 noise = fract(sin(vec3(dot(p,vec3(127.1, 311.7, 191.999)),
                          dot(p,vec3(269.5, 183.3, 765.54)),
                          dot(p, vec3(420.69, 631.2,109.21))))
                 *43758.5453);
    return max(noise.x, max(noise.y, noise.z));
}

// Function 408
float noise(vec3 p){
	vec3 f = fract(p);
    f = f * f * (3. - 2. * f);
    vec3 c = floor(p);
  
    return mix(mix(mix(hash(c), hash(c + vec3(1., 0., 0.)), f.x),
               	   mix(hash(c + vec3(0., 1., 0.)), hash(c + vec3(1., 1., 0.)), f.x),
               	   f.y),
               mix(mix(hash(c + vec3(0., 0., 1.)), hash(c + vec3(1., 0., 1.)), f.x),
               	   mix(hash(c + vec3(0., 1., 1.)), hash(c + vec3(1., 1., 1.)), f.x),
               	   f.y),
               f.z);  
}

// Function 409
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

// Function 410
float noise(vec2 q, vec2 p, float r2, float z) {
    vec2 h = hash(q, z);
    vec2 d = unskew(q) - p;
    float d2 = d.x * d.x + d.y * d.y;
    return pow(max(0.0, r2 - d2), 4.0) * dot(d, h);
}

// Function 411
vec3 colNoise(vec2 st, vec3 c, float amp) {
  float x = Random2d(st) * amp;
  float y = Random2d(st) * (amp-x);
  float z = Random2d(st) * (amp-x-y);
  x -= amp/2.;
  y-=amp/2.;
  z-=amp/2.;
  float tmp;
  // Shuffle
  for (int i=0; i<3; i++) {
    switch (int(floor(Random2d(st) * 3.))) {
      case 0: tmp=x; x=y; y=tmp;
              break;
      case 1: tmp=x; x=z; z=tmp;
              break;
      case 2: tmp=y; y=z; z=tmp;
              break;
      default: break;
    }
  }
  //return vec3(red(c)+x, green(c)+y, blue(c)+z);
    return c + vec3(x,y,z);
}

// Function 412
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

// Function 413
vec2 Noise2(vec2 x) {     // pseudoperlin improvement from foxes idea 
    return (noise2(x)+noise2(x+11.5)) / 2.;
}

// Function 414
float Noise(in vec2 p)
{
    vec2 n = floor(p);
    vec2 f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(mix(Hash(n), Hash(n + vec2(1.0, 0.0)), u.x),
               mix(Hash(n + vec2(0.0, 1.0)), Hash(n + vec2(1.0)), u.x), u.y);
}

// Function 415
vec2 GaborNoise(vec2 uv, float freq, float dir) {
    vec2 f=vec2(0.); float fa=0.;
	for (float i=0.; i<NB; i++) { 
		vec2 pos = vec2(1.8*rndi(i,0.),rndi(i,1.));
        float a = dir + SPREAD *PI*(2.*i/NB-1.);
		f += Gabor(uv-pos, freq, a);
	}
	return f *sqrt(200./NB); // /6.;
}

// Function 416
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

// Function 417
float noise( in vec2 p )
{
    return fract((sin(p.x)+cos(p.y))*43758.5453123);
    // my tests showed that another ver. has a tiny speed difference:
    // - a little bit faster on Intel chips,
    // - but a little bit slower on Nvidia:
    //  p  = fract(p * vec2(0.3247122237916646, 0.134707348285849));
    //  p += dot(p.xy, p.yx+19.19);
    //  return fract(p.x * p.y);
}

// Function 418
float noise ( vec2 x)
{
	return iqnoise(x, 0., 1.);
}

// Function 419
float blue_noise( vec2 uv )
{
    return texture( iChannel1, uv ).r;
}

// Function 420
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

// Function 421
float FractalNoise(in vec2 p)
{
    p *= 5.0;
    mat2 m = mat2(1.6,  1.2, -1.2,  1.6);
	float f = 0.5000 * Noise(p); p = m * p;
	f += 0.2500 * Noise(p); p = m * p;
	f += 0.1250 * Noise(p); p = m * p;
	f += 0.0625 * Noise(p); p = m * p;
    
    return f;
}

// Function 422
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

// Function 423
vec4 noised(in vec3 x)
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

// Function 424
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z )*2.0-1.0;
}

// Function 425
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

// Function 426
vec2 noise( in vec3 x )
{
    vec3 ip = floor(x);
    vec3 fp = fract(x);
	fp = fp*fp*(3.0-2.0*fp);
	vec2 tap = (ip.xy+vec2(37.0,17.0)*ip.z) + fp.xy;
	vec4 rz = textureLod( iChannel0, (tap+0.5)/256.0, 0.0 );
	return mix( rz.yw, rz.xz, fp.z );
}

// Function 427
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

// Function 428
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

// Function 429
float noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f*f*(3.0-2.0*f);

    float n = p.x + p.y*157.0;

    return mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
               mix( hash(n+157.0), hash(n+158.0),f.x),f.y);
}

// Function 430
float noiseMod(vec2 u,vec2 s){u*=s;vec2 f=fract(u);u=floor(u);	f=f*f*(3.-2.*f);//any interpolation is fine for f
	return mix(mix(h12s(mod(u          ,s)),h12s(mod(u+vec2(1,0),s)),f.x),
					    mix(h12s(mod(u+vec2(0,1),s)),h12s(mod(u+vec2(1,1),s)),f.x),f.y);}

// Function 431
vec4 noise(vec2 u){u=mod(u,M.x)+M.y; // mod() avoids grid glitch
 u-=.5;return vec4(hash(uint(u.x+iResolution.x*u.y)));}

// Function 432
float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(4.0-3.0*f);
    return -2.0+3.0*mix( mix( hash( i + vec2(0.1,0.1) ), 
                     hash( i + vec2(2.0,0.1) ), u.x),
                mix( hash( i + vec2(0.1,2.0) ), 
                     hash( i + vec2(2.0,2.0) ), u.x), u.y);
}

// Function 433
float noise (in vec2 uv)
{
  vec2 b = floor(uv);
  return mix(mix(rand(b),rand(b+1.),.5),mix(rand(b+0.1),rand(b+1.),.5),.5);
}

// Function 434
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

// Function 435
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

// Function 436
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

// Function 437
float noise2D( in vec2 pos, float lod)
{   
  vec2 f = fract(pos);
  f = f*f*(3.0-2.0*f);
  vec2 rg = textureLod( iChannel1, (((floor(pos).xy+vec2(37.0, 17.0)) + f.xy)+ 0.5)/64.0, lod).yx;  
  return -1.0+2.0*mix( rg.x, rg.y, 0.5 );
}

// Function 438
float noise( const in  float p ) {    
    float i = floor( p );
    float f = fract( p );	
	float u = f*f*(3.0-2.0*f);
    return -1.0+2.0* mix( hash( i + 0. ), hash( i + 1. ), u);
}

// Function 439
float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = rand(i);
    float b = rand(i + vec2(1.0, 0.0));
    float c = rand(i + vec2(0.0, 1.0));
    float d = rand(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Function 440
vec2 Noise22(vec2 x)
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    vec2 res = mix(mix( hash22(p),          hash22(p + add.xy),f.x),
                    mix( hash22(p + add.yx), hash22(p + add.xx),f.x),f.y);
    return res-.5;
}

// Function 441
float value_noise(in vec3 p) {
    vec3 ip = floor(p);
    vec3 fp = fract(p);

    float q = random(ip);
    float w = random(ip + vec3(1.,0.,0.));
    float e = random(ip + vec3(0.,1.,0.));
    float r = random(ip + vec3(1.,1.,0.));
    
    float a = random(ip + vec3(0.,0.,1.));
    float s = random(ip + vec3(1.,0.,1.));
    float d = random(ip + vec3(0.,1.,1.));
    float f = random(ip + vec3(1.,1.,1.));

    vec3 u = 3.*fp*fp - 2.*fp*fp*fp;
    
    float v1 = mix(mix(q,w,u.x),
                    mix(e,r,u.x), u.y);
    float v2 = mix(mix(a,s,u.x), 
                    mix(d,f,u.x), u.y);
    float v = mix(v1, v2, u.z);
    return v;
}

// Function 442
float noise_prng_uniform_0_1(inout noise_prng this_)
{
    return float(noise_prng_rand(this_)) / float(4294967295u);
}

// Function 443
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

// Function 444
float noise(vec2 x){
    vec2 f = fract(x);
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0);
    
    vec2 p = floor(x);
	float a = texture(iChannel0, (p+vec2(0.0, 0.0))/1024.0).x;
	float b = texture(iChannel0, (p+vec2(1.0,0.0))/1024.0).x;
	float c = texture(iChannel0, (p+vec2(0.0,1.0))/1024.0).x;
	float d = texture(iChannel0, (p+vec2(1.0,1.0))/1024.0).x;

    
	return a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y;
}

// Function 445
vec3 NoiseD( in vec2 p )
{
    vec2 f = fract(p);
    p = floor(p);
    vec2 u = f*f*(1.5-f)*2.0;
    vec4 n;
	n.x = textureLod( iChannel0, (p+vec2(0.5,0.5))*STEP, 0.0 ).x;
	n.y = textureLod( iChannel0, (p+vec2(1.5,0.5))*STEP, 0.0 ).x;
	n.z = textureLod( iChannel0, (p+vec2(0.5,1.5))*STEP, 0.0 ).x;
	n.w = textureLod( iChannel0, (p+vec2(1.5,1.5))*STEP, 0.0 ).x;

    // Normally you can make a texture out of these 4 so
    // you don't have to do any of it again...
    n.yzw = vec3(n.x-n.y-n.z+n.w, n.y-n.x, n.z-n.x);
    vec2 d = 6.0*f*(f-1.0)*(n.zw+n.y*u.yx);
    
	return vec3(n.x + n.z * u.x + n.w * u.y + n.y * u.x * u.y, d.x, d.y);
}

// Function 446
float Noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);

	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( iChannel0, (uv+ 0.5)/256.0, -100.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 447
float noise(vec2 p){
    vec2 ip = floor(p);
    vec2 u = fract(p);
    u = u*u*(3.0-2.0*u);

    float res = mix(
        mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
        mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
    return res*res;
}

// Function 448
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

// Function 449
vec2 noiseTex2(in vec3 x)
{
    vec3 fl = floor(x);
    vec3 fr = fract(x);
	fr = fr * fr * (3.0 - 2.0 * fr);
	vec2 uv = (fl.xy + vec2(37.0, 17.0) * fl.z) + fr.xy;
	vec4 rgba = textureLod(iChannel0, (uv + 0.5) * 0.00390625, 0.0 ).xyzw;
	return mix(rgba.yw, rgba.xz, fr.z);
}

// Function 450
float valueNoise3du(vec3 samplePoint) {
    vec3 pointI = floor(samplePoint);
    vec3 pointF = fract(samplePoint);
    vec3 u = valueNoiseFilter(pointF);

    //Slight Optimisation
    vec4 m = mix(
        vec4(
            hash31(pointI ),//bbl,
            hash31(pointI + vec3(0.0,1.0,0.0) ),//btl,
            hash31(pointI + vec3(0.0,0.0,1.0) ),//fbl,
            hash31(pointI + vec3(0.0,1.0,1.0) )//ftl
        ),vec4(
            hash31(pointI + vec3(1.0,0.0,0.0) ),//bbr,
            hash31(pointI + vec3(1.0,1.0,0.0) ),//btr,
            hash31(pointI + vec3(1.0,0.0,1.0) ),//fbr,
            hash31(pointI + vec3(1.0,1.0,1.0) )//ftr
        ),u.x);

    vec2 n = mix(m.xz, m.yw, u.y);
    return mix(n.x,n.y,u.z);
}

// Function 451
float myNoise(float value)
{
 	value += cos(value * 100.0);
 	value += cos(value * 20.0);
    value = texture(iChannel2, vec2(1.0, 1.0 / 256.0) * value).x - 0.5;
    return value * 2.0;
}

// Function 452
vec4 noise(int p){
 	return noise(ivec3(p, 0, 0));   
}

// Function 453
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

// Function 454
vec3 noise1v( float n )
{   
    
    vec2 coords = vec2(mod(n,NOISE_DIMENSION)/NOISE_DIMENSION, 
                       floor(n/NOISE_DIMENSION)/NOISE_DIMENSION);
    
    return texture(iChannel0, coords).rgb;
}

// Function 455
float noise(vec2 p)
{
    vec2 pi = floor(p);
    vec2 pf = smoothstep(0., 1., p - pi);
    return mix(
        mix(hash(pi), 	   hash(pi+o), pf.x), 
        mix(hash(pi+o.yx), hash(pi+o.xx), pf.x), 
        pf.y);
}

// Function 456
float noise(vec2 pos)
{
	return fract( sin( dot(pos*0.001 ,vec2(24.12357, 36.789) ) ) * 12345.123);	
}

// Function 457
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

// Function 458
float noise( const in  vec2 p ) {    
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
    return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 459
float NoiseT( in vec2 x ) {
    vec2 p = floor(x), f = fract(x);
    f = f*f*(3.-2.*f);
    float n = p.x + p.y*57.0;
    return mix(mix( HashT(n     ), HashT(n+  1.),f.x),
               mix( HashT(n+ 57.), HashT(n+ 58.),f.x),f.y);
}

// Function 460
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

// Function 461
vec3 noised( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    
    // cubic interpolation vs quintic interpolation
#if 1 
    vec2 u = f*f*(3.0-2.0*f);
    vec2 du = 6.0*f*(1.0-f);
    vec2 ddu = 6.0 - 12.0*f;
#else
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0);
    vec2 ddu = 60.0*f*(1.0+f*(-3.0+2.0*f));
#endif
    
	float a = textureLod(iChannel0,(p+vec2(0.5,0.5))/256.0,0.0).x;
	float b = textureLod(iChannel0,(p+vec2(1.5,0.5))/256.0,0.0).x;
	float c = textureLod(iChannel0,(p+vec2(0.5,1.5))/256.0,0.0).x;
	float d = textureLod(iChannel0,(p+vec2(1.5,1.5))/256.0,0.0).x;
	
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k4 =   a - b - c + d;


    // value
    float va = a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y;
    // derivative                
    vec2  de = du*(vec2(b-a,c-a)+(a-b-c+d)*u.yx);
    // hessian (second derivartive)
    mat2  he = mat2( ddu.x*(k1 + k4*u.y),   
                     du.x*k4*du.y,
                     du.y*k4*du.x,
                     ddu.y*(k2 + k4*u.x) );
    
    return vec3(va,de);

}

// Function 462
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

// Function 463
vec4 noise4(vec2 x) {
    return vec4(
        noise(x+vec2(0,0)),
        noise(x+vec2(0,.333)),
        noise(x+vec2(.333,0)),
        noise(x+vec2(0,.666))
    );
}

// Function 464
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

// Function 465
float noise(float x) {
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3.0 - 2.0 * f);
    return mix(hash(i), hash(i + 1.0), u);
}

// Function 466
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

// Function 467
float valueNoise1du(float samplePoint) {
    float pointI = floor(samplePoint);
    return mix(hash11(pointI),hash11(pointI + 1.0 ),valueNoiseFilter(fract(samplePoint)));
}

// Function 468
float Noise(in vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, -100.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 469
float valueNoise(float i, float p){ return mix(r11(floor(i)),r11(floor(i) + 1.), ss(fract(i), p,0.6));}

// Function 470
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

// Function 471
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
    vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
    return mix( rg.x, rg.y, f.z );
}

// Function 472
float noise(vec2 p, sampler2D s, vec2 r, float sc, float bl)
{
    float data = 0.;
    for(int i = 0; i < 8; i++)
    {
        data+=texture(s, p*sc+SAMPLE[i]*(bl/r.x)).x/8.;
    }
    return data;
}

// Function 473
vec4 noise(vec2 U) {
    U-=.5; 
    return vec4( hash(uint(U.x+iResolution.x*U.y)) );  // white noise
}

// Function 474
float noise(vec2 xz)
{
	vec2 f = fract(xz);
	xz = floor(xz);
	vec2 u = f * f * (3.0 - 2.0 * f);
	return	mix(mix(hash12(xz), hash12(xz + vec2(1.0, 0.0)), u.x),
	mix(hash12(xz + vec2(0.0, 1.0)), hash12(xz + vec2(1.0, 1.0)), u.x), u.y);
}

// Function 475
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

// Function 476
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

// Function 477
float iqNoiseLayered( vec2 p )
{
    vec2 q = 0.05*p;
	float f = 0.0;
    f += 0.50000*almostAbs(iqNoise( q )); q = m2*q*2.02; q -= 0.1*iTime;
    f += 0.25000*almostAbs(iqNoise( q )); q = m2*q*2.03; q += 0.2*iTime;
    f += 0.12500*almostAbs(iqNoise( q )); q = m2*q*2.01; q -= 0.4*iTime;
    f += 0.06250*almostAbs(iqNoise( q )); q = m2*q*2.02; q += 1.0*iTime;
    f += 0.03125*almostAbs(iqNoise( q ));
    return 3.7-4.0*f;
}

// Function 478
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

// Function 479
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

// Function 480
float Noisefv2 (vec2 p)
{
  vec2 t, ip, fp;
  ip = floor (p);  
  fp = fract (p);
  fp = fp * fp * (3. - 2. * fp);
  t = mix (Hashv2v2 (ip), Hashv2v2 (ip + vec2 (0., 1.)), fp.y);
  return mix (t.x, t.y, fp.x);
}

// Function 481
float noise(sampler2D randSrc, vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (i.xy+vec2(37.0,17.0)*i.z) + f.xy;
	vec2 rg = textureLod( randSrc, (uv+0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 482
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

// Function 483
vec3 noise(vec2 p, float lod){return texture(iChannel0,p/iChannelResolution[0].xy,lod).xyz;}

// Function 484
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

// Function 485
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.-2.*f);
	vec2 uv = (p.xy+vec2(37.,17.)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ .5)/256., -100.).yx;
	return -1.+2.*mix( rg.x, rg.y, f.z );
}

// Function 486
float noise11(float p)
{
	float fl = floor(p);
	float fc = fract(p);
	return mix(rand11(fl), rand11(fl + 1.0), fc);
}

// Function 487
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

// Function 488
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

// Function 489
float bnoise(in vec3 p)
{
    p*= 2.5;
    float n = noise(p*10.)*0.8;
    n += noise(p*25.)*0.5;
    n += noise(p*45.)*0.25;
    return (n*n)*0.004;
}

// Function 490
vec3 NOISE_volumetricRoughnessMap(vec3 p, float rayLen)
{
    float ROUGHNESS_MAP_UV_SCALE = 6.00;//Valid range : [0.1-100.0]
    vec4 sliderVal = vec4(0.5,0.85,0,0.5);
    ROUGHNESS_MAP_UV_SCALE *= 0.1*pow(10.,2.0*sliderVal[0]);
    
    float f = iTime;
    const mat3 R1  = mat3(0.500, 0.000, -.866,
	                     0.000, 1.000, 0.000,
                          .866, 0.000, 0.500);
    const mat3 R2  = mat3(1.000, 0.000, 0.000,
	                      0.000, 0.500, -.866,
                          0.000,  .866, 0.500);
    const mat3 R = R1*R2;
    p *= ROUGHNESS_MAP_UV_SCALE;
    p = R1*p;
    vec4 v1 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021;
    vec4 v2 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021+1.204*v1.xyz;
    vec4 v3 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021+0.704*v2.xyz;
    vec4 v4 = NOISE_trilinearWithDerivative(p);
    
    return (v1
	      +0.5*(v2+0.25)
	      +0.4*(v3+0.25)
	      +0.6*(v4+0.25)).yzw;
}

// Function 491
InterpNodes2 GetNoiseInterpNodes(float smoothNoise)
{
    vec2 globalPhases = vec2(smoothNoise * 0.5) + vec2(0.5, 0.0);
    vec2 phases = fract(globalPhases);
    vec2 seeds = floor(globalPhases) * 2.0 + vec2(0.0, 1.0);
    vec2 weights = min(phases, vec2(1.0f) - phases) * 2.0;
    return InterpNodes2(seeds, weights);
}

// Function 492
float valueNoiseStepped(float i, float p, float steps){ return mix(  floor(r11(floor(i))*steps)/steps, floor(r11(floor(i) + 1.)*steps)/steps, ss(fract(i), p,0.6));}

// Function 493
vec4 NOISE_trilinearWithDerivative(vec3 p)
{
    //Trilinear extension over noise derivative from (Elevated), & using the noise stacking trick from (Clouds).
	//Inspiration & Idea from :
    //https://www.shadertoy.com/view/MdX3Rr (Elevated)
    //https://www.shadertoy.com/view/XslGRr (Clouds)
    
    //For more information, see also:
    //NoiseVolumeExplained : https://www.shadertoy.com/view/XsyGWz
	//2DSignalDerivativeViewer : https://www.shadertoy.com/view/ldGGDR
    
    const float TEXTURE_RES = 256.0; //Noise texture resolution
    vec3 pixCoord = floor(p);//Pixel coord, integer [0,1,2,3...256...]
    //noise volume stacking trick : g layer = r layer shifted by (37x17 pixels)
    //(37x17)-> this value is the actual translation embedded in the noise texture, can't get around it.
	//Note : shift is different from g to b layer (but it also works)
    vec2 layer_translation = -pixCoord.z*vec2(37.0,17.0)/TEXTURE_RES; 
    
    vec2 c1 = texture(iChannel2,layer_translation+(pixCoord.xy+vec2(0,0)+0.5)/TEXTURE_RES,-100.0).rg;
    vec2 c2 = texture(iChannel2,layer_translation+(pixCoord.xy+vec2(1,0)+0.5)/TEXTURE_RES,-100.0).rg; //+x
    vec2 c3 = texture(iChannel2,layer_translation+(pixCoord.xy+vec2(0,1)+0.5)/TEXTURE_RES,-100.0).rg; //+z
    vec2 c4 = texture(iChannel2,layer_translation+(pixCoord.xy+vec2(1,1)+0.5)/TEXTURE_RES,-100.0).rg; //+x+z
    
    vec3 x = p-pixCoord; //Pixel interpolation position, linear range [0-1] (fractional part)
    
    vec3 x2 = x*x;
    vec3 t = (6.*x2-15.0*x+10.)*x*x2; //Quintic ease-in/ease-out function.
    vec3 d_xyz = (30.*x2-60.*x+30.)*x2; //dt/dx : Ease-in ease-out derivative.
    
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
    
    //Derivative scaling (texture lookup slope, along interpolation cross sections).
    //This could be factorized/optimized but I fear it would make it cryptic.
    float sx =  ((b-a)+t.y*(a-b-c+d))*(1.-t.z)
               +((f-e)+t.y*(e-f-g+h))*(   t.z);
    float sy =  ((c-a)+t.x*(a-b-c+d))*(1.-t.z)
               +((g-e)+t.x*(e-f-g+h))*(   t.z);
    float sz =  zb-za;
    
    return vec4(value,d_xyz*vec3(sx,sy,sz));
}

// Function 494
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

// Function 495
float Noise2(vec2 uv)
{
    vec2 corner = floor(uv);
	float c00 = N2(corner + vec2(0.0, 0.0));
	float c01 = N2(corner + vec2(0.0, 1.0));
	float c11 = N2(corner + vec2(1.0, 1.0));
	float c10 = N2(corner + vec2(1.0, 0.0));
    
    vec2 diff = fract(uv);
    
    return CosineInterpolate(CosineInterpolate(c00, c10, diff.x), CosineInterpolate(c01, c11, diff.x), diff.y);
}

// Function 496
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

// Function 497
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

// Function 498
float noise( in vec3 f )
{
    vec3 p = floor(f);
    f = fract(f);
    f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel0, (uv+ 0.5)/256.0, 0.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 499
float noise(vec3 p) {
	const vec3 s = vec3(7.0, 157.0, 113.0);
	vec3 ip = floor(p);
    vec4 h = vec4(0.0, s.yz, s.y + s.z) + dot(ip, s);
	p -= ip;
	
    h = mix(fract(sin(h) * 43758.5453), fract(sin(h + s.x) * 43758.5453), p.x);
	
    h.xy = mix(h.xz, h.yw, p.y);
    return mix(h.x, h.y, p.z);
}

// Function 500
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

// Function 501
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

// Function 502
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

// Function 503
float PerlinNoise2D(vec2 uv, int octaves) {
    float c = 0.0;
    float s = 0.0;
    for (float i = 0.0; i < float(octaves); i++) {
        c += SmoothNoise2D(uv * pow(2.0, i)) * pow(0.5, i);
        s += pow(0.5, i);
    }
    
    return c /= s;
}

// Function 504
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

// Function 505
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

// Function 506
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

// Function 507
float noise(float x)
{
    float p = floor(x);
    float f = fract(x);
    f = f*f*(3.0-2.0*f);
	
    vec2 n = vec2(p, p+1000.);
    return mix(rand2(n), rand2(n + 1.0),f);
}

// Function 508
float noise(vec3 x) 
{
  vec3 p = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);

  float n = p.x + p.y * 157.0 + 113.0 * p.z;
  return -1.0+2.0*mix(
    mix(mix(hash(n + 0.0), hash(n + 1.0), f.x), 
    mix(hash(n + 157.0), hash(n + 158.0), f.x), f.y), 
    mix(mix(hash(n + 113.0), hash(n + 114.0), f.x), 
    mix(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
}

// Function 509
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

// Function 510
vec3 noiseD( in vec2 x )
{
	x+=4.2;
    vec2 p = floor(x);
    vec2 f = fract(x);

    vec2 u = f*f*(1.5-f)*2.0;;
    
    float a = hash12(p);
    float b = hash12(p + add.xy);
    float c = hash12(p + add.yx);
    float d = hash12(p + add.xx);
	return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y,
				6.0*f*(f-1.0)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));
}

// Function 511
float noise (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    vec2 u = f*f*(3.0-2.0*f);
    // u = smoothstep(0.,1.,f);

    // Mix 4 coorners porcentages
    float value = mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
    return value;
}

// Function 512
vec4 SqrNoise( in vec3 p )
{
	float f=noise(p);
	vec3 n=GrNoise(p);
	return vec4(2.*f*n,f*f);
}

// Function 513
float noiseNew( in vec3 x )
{
    return texture( iChannel0, x/32.0 ).x;    // <---------- Sample a 3D texture!
}

// Function 514
float noise( vec2 p )
{
	return sin(p.x)*sin(p.y);
}

// Function 515
float noise( in vec2 x )

{
	return texture(iChannel1, x ).r;
}

// Function 516
float noise(in vec2 uv)
{
    return sin(uv.x)+cos(uv.y);
}

// Function 517
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

// Function 518
void noiseInterpolateDu(const in vec3 x, out vec3 u, out vec3 du) 
{ 
    vec3 x2 = x * x;
    u = x2 * x * (x * (x * 6.0 - 15.0) + 10.0); 
    du = 30.0 * x2 * (x * (x - 2.0) + 1.0);
}

// Function 519
F1 Noise(F2 n,F1 x){n+=x;return fract(sin(dot(n.xy,F2(12.9898, 78.233)))*43758.5453);}

// Function 520
float noise(vec2 xz)
{
	vec2 f = fract(xz);
	xz = floor(xz);
	vec2 u = f * f * (3.0 - 2.0 * f);
	return	mix(mix(hash12(xz), hash12(xz + add.xy), u.x),
	mix(hash12(xz + vec2(0.0, 1.0)), hash12(xz + add.xx), u.x), u.y);
}

// Function 521
float cloudNoise3D(vec3 uv, vec3 _wind)
{
    float v = 1.0-voronoi3D(uv*20.0+_wind);
    float fs = fbm3Dsimple(uv*40.0+_wind);
    float mask = fbm3Dsimple(uv*0.1+_wind);
    return clamp(v*fs*mask, 0.0, 1.0);
}

// Function 522
float nestedNoise(vec2 p) {
  float x = movingNoise(p);
  float y = movingNoise(p + 100.);
  return movingNoise(p + vec2(x, y));
}

// Function 523
float noiseNew( in vec3 x )
{
	x += 0.5;
	vec3 fx = fract( x );
	x = floor( x ) + fx*fx*(3.0-2.0*fx);
    return texture( iChannel0, (x-0.5)/32.0 ).x;

}

// Function 524
float cosNoise( in vec2 p )
{
    return 0.3*( sin(p.x) + sin(p.y) );
}

// Function 525
vec4 noise(ivec4 p){
    const float scale = pow(2., -32.);
    uvec4 h = hash(uvec4(p));
    return vec4(h)*scale;
}

// Function 526
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

// Function 527
float noiseOld( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel1, (uv+0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 528
float noise(in vec2 x){vec2 p=floor(x),f=fract(x);f=f*f*(3.0-2.0*f);
 float n=p.x+p.y*57.;
 return mix(mix(h11(n+ 0.),h11(n+ 1.),f.x),
            mix(h11(n+57.),h11(n+58.),f.x),f.y);}

// Function 529
vec2 noise(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * HASHSCALE3);
    p3 += dot(p3, p3.yzx+19.19);
    return fract(vec2((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y));
}

// Function 530
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

// Function 531
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

// Function 532
float add_noise(vec3 x) {
    float n = noise(x)/2.;  x *= 2.1; // return n*2.;
         n += noise(x)/4.;  x *= 1.9;
         n += noise(x)/8.;  x *= 2.3;
         n += noise(x)/16.; x *= 1.9;
         n += noise(x)/32.;
    return n; 
}

// Function 533
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

// Function 534
float noise(vec2 p) {
    vec2 ip = floor(p);
    float c00 = rand(ip);
    float c01 = rand(ip + vec2(1., 0.));
    float c10 = rand(ip + vec2(0., 1.));
    float c11 = rand(ip + vec2(1.));
    
    vec2 fp = fract(p);
    vec2 uf = smoothstep(vec2(0.), vec2(1.), fp);
    float r0 = mix(c00, c01, uf.x);
    float r1 = mix(c10, c11, uf.x);
    return mix(r0, r1, uf.y);
}

// Function 535
float Noisefv3 (vec3 p)
{
  vec4 t1, t2;
  vec3 i, f;
  float q;
  i = floor (p);  f = fract (p);  f = f * f * (3. - 2. * f);
  q = dot (i, cHashA3);
  t1 = Hashv4f (q);  t2 = Hashv4f (q + cHashA3.z);
  return mix (mix (mix (t1.x, t1.y, f.x), mix (t1.z, t1.w, f.x), f.y),
     mix (mix (t2.x, t2.y, f.x), mix (t2.z, t2.w, f.x), f.y), f.z);
}

// Function 536
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

// Function 537
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

// Function 538
float polynoise(vec2 p, float sharpness)
{
    vec2 seed = floor(p);
    vec2 rndv = vec2(rnd(seed.xy), rnd(seed.yx));
    vec2 pt = fract(p);
    float bx = value(pt.x, rndv.x, rndv.y * sharpness);
    float by = value(pt.y, rndv.y, rndv.x * sharpness);
    return min(bx, by) * (0.3 + abs(rand(seed.xy * 0.01)) * 0.7);
}

// Function 539
float fnoise( vec3 p, in float t )
{
	p *= .25;
    float f;

	f = 0.5000 * Noise(p); p = p * 3.02; p.y -= t*.2;
	f += 0.2500 * Noise(p); p = p * 3.03; p.y += t*.06;
	f += 0.1250 * Noise(p); p = p * 3.01;
	f += 0.0625   * Noise(p); p =  p * 3.03;
	f += 0.03125  * Noise(p); p =  p * 3.02;
	f += 0.015625 * Noise(p);
    return f;
}

// Function 540
float InterferenceNoise( vec2 uv )
{
	float displayVerticalLines = 483.0;
    float scanLine = floor(uv.y * displayVerticalLines); 
    float scanPos = scanLine + uv.x;
	float timeSeed = fract( iTime * 123.78 );
    
    return InterferenceSmoothNoise1D( scanPos * 234.5 + timeSeed * 12345.6 );
}

// Function 541
float noiseMask(vec2 uv, int layer)
{
    vec4 p = vec4(uv * 3., float(layer) * DIVERGENCE, iTime * 0.5);
    float f = twistedSineNoise(p);
    f += length(uv) * SHAPE_SIMPLICITY;
    return step(SHAPE_SIMPLICITY, f + float(layer));
}

// Function 542
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

// Function 543
float noise(float x) { float i = floor(x); float f = fract(x); float u = f * f * (3.0 - 2.0 * f); return mix(hash(i), hash(i + 1.0), u); }

// Function 544
float proceduralSplatter(vec2 st, float radius, float numCircles){
    float pct = 0.;
    st.x -= .5;
    for (float i = 1.; i < numCircles; i++){
        st.y -=(.3 / (i+1.));
        pct +=smoothstep(radius * 1./i, radius * 1./i - .1, length(st));
    }
    return pct;
}

// Function 545
float noise(float p){
    float fl = floor(p);
    float fc = fract(p);
    return mix(rand(fl), rand(fl + 1.0), fc);
}

// Function 546
float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = texture( noiseTexture, (uv+0.5)/256.0, -100.0 ).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 547
float noise(vec3 p)
{
	vec3 pm = mod(p,1.0);
	vec3 pd = p-pm;
	return hashmix(pd,(pd+vec3(1.0,1.0,1.0)), pm);
}

// Function 548
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

// Function 549
float noise(vec2 p) {
    const float K1 = .366025404;
    const float K2 = .211324865;
    vec2 i = floor(p + (p.x+p.y)*K1);
    vec2 a = p - i + (i.x + i.y)*K2;
    vec2 o = step(a.yx, a.xy);
    vec2 b = a - o + K2;
    vec2 c = a - 1. + 2.*K2;
    vec3 h = max(.5 - vec3(dot(a,a), dot(b,b), dot(c,c)), 0.);
    vec3 n = h*h*h*h*vec3(dot(a, hash(i)),
                          dot(b, hash(i+o)),
                          dot(c, hash(i+1.)));
    return dot(n, vec3(70));
}

// Function 550
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

// Function 551
float noise2D( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f * f * (3.0 - 2.0 * f);

    return mix( mix( dot( hash( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( hash( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hash( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( hash( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// Function 552
float noise (vec2 v) {
    vec4 n = vec4(floor(v),ceil(v));
    vec4 h = vec4(hash(n.xy),hash(n.zy),hash(n.xw),hash(n.zw));
    return mix(mix(h.x,h.y,v.x-n.x),mix(h.z,h.w,v.x-n.x),v.y-n.y);
}

// Function 553
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

// Function 554
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

// Function 555
float Simple360HDR_noise(vec2 v){
  vec2 v1=floor(v);
  vec2 v2=smoothstep(0.0,1.0,fract(v));
  float n00=Simple360HDR_hash12(v1);
  float n01=Simple360HDR_hash12(v1+vec2(0,1));
  float n10=Simple360HDR_hash12(v1+vec2(1,0));
  float n11=Simple360HDR_hash12(v1+vec2(1,1));
  return mix(mix(n00,n01,v2.y),mix(n10,n11,v2.y),v2.x);
}

// Function 556
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

// Function 557
vec4 NOISE_trilinearWithDerivative(vec3 p)
{
    //Trilinear extension over noise derivative from (Elevated), & using the noise stacking trick from (Clouds).
	//Inspiration & Idea from :
    //https://www.shadertoy.com/view/MdX3Rr (Elevated)
    //https://www.shadertoy.com/view/XslGRr (Clouds)
    
    //For more information, see also:
    //NoiseVolumeExplained : https://www.shadertoy.com/view/XsyGWz
	//2DSignalDerivativeViewer : https://www.shadertoy.com/view/ldGGDR
    
    const float TEXTURE_RES = 256.0; //Noise texture resolution
    vec3 pixCoord = floor(p);//Pixel coord, integer [0,1,2,3...256...]
    //noise volume stacking trick : g layer = r layer shifted by (37x17 pixels)
    //(37x17)-> this value is the actual translation embedded in the noise texture, can't get around it.
	//Note : shift is different from g to b layer (but it also works)
    vec2 layer_translation = -pixCoord.z*vec2(37.0,17.0)/TEXTURE_RES; 
    
    vec2 c1 = texture(iChannel0,layer_translation+(pixCoord.xy+vec2(0,0)+0.5)/TEXTURE_RES,-100.0).rg;
    vec2 c2 = texture(iChannel0,layer_translation+(pixCoord.xy+vec2(1,0)+0.5)/TEXTURE_RES,-100.0).rg; //+x
    vec2 c3 = texture(iChannel0,layer_translation+(pixCoord.xy+vec2(0,1)+0.5)/TEXTURE_RES,-100.0).rg; //+z
    vec2 c4 = texture(iChannel0,layer_translation+(pixCoord.xy+vec2(1,1)+0.5)/TEXTURE_RES,-100.0).rg; //+x+z
    
    vec3 x = p-pixCoord; //Pixel interpolation position, linear range [0-1] (fractional part)
    
    vec3 x2 = x*x;
    vec3 t = (6.*x2-15.0*x+10.)*x*x2; //Quintic ease-in/ease-out function.
    vec3 d_xyz = (30.*x2-60.*x+30.)*x2; //dt/dx : Ease-in ease-out derivative.
    
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
    
    //Derivative scaling (texture lookup slope, along interpolation cross sections).
    //This could be factorized/optimized but I fear it would make it cryptic.
    float sx =  ((b-a)+t.y*(a-b-c+d))*(1.-t.z)
               +((f-e)+t.y*(e-f-g+h))*(   t.z);
    float sy =  ((c-a)+t.x*(a-b-c+d))*(1.-t.z)
               +((g-e)+t.x*(e-f-g+h))*(   t.z);
    float sz =  zb-za;
    
    return vec4(value,d_xyz*vec3(sx,sy,sz));
}

// Function 558
float noise2( mediump vec2 x )
{
    vec2 p = floor(x);
    lowp vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*157.0;
    lowp vec4 h = hash4(vec4(n)+vec4(NC0.xy,NC1.xy));
    lowp vec2 s1 = mix(h.xy,h.zw,f.xx);
    return mix(s1.x,s1.y,f.y);
}

// Function 559
vec2 Noise( in vec2 x )
{
    return mix(Hash2(floor(x)), Hash2(floor(x)+1.0), fract(x));
}

// Function 560
float noise(vec2 n) {
	const vec2 d = vec2(0.0, 1.0);
	vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
	return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

// Function 561
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

// Function 562
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

// Function 563
vec2 noise_prng_rand2_1_1(inout noise_prng this_)
{
    return -1.+2.*vec2(noise_prng_uniform_0_1(this_),
                       noise_prng_uniform_0_1(this_));
}

// Function 564
float uniformNoise(vec2 n){
    // uniformly distribued, normalized in [0..1[
    return fract(sin(dot(n, vec2(12.9898, 78.233))) * 43758.5453);
}

// Function 565
vec3 noisetile(vec2 uv){
    // clamp probably not (and shouldn't be) needed but anyway
    return vec3(clamp(lerpy(uv), 0.0, 1.0));
}

// Function 566
float noise( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );	
	vec2 u = f*f*(3.0-2.0*f);
    return -1.0+2.0*mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Function 567
vec4 SNoise( in vec3 p )
{
	float f=noise(p);
	vec3 n=GrNoise(p);
	return vec4(n,f);
}

// Function 568
vec3 noise(vec3 p)
{
    return 1.0 - 2.0 * abs(0.5 - textureLod(iChannel0, p, 0.0).xyz);
}

// Function 569
float noise(float p){
	float fl = floor(p);
    float fc = fract(p);
	return mix(rand(fl), rand(fl + 1.0), fc);
}

// Function 570
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

// Function 571
float polynoise(vec2 p)
{
    vec2 seed = floor(p);
    vec2 rndv = vec2( rnd(seed.xy), rnd(seed.yx));
    vec2 pt = fract(p);
    float bx = value(pt.x, rndv.x);
    float by = value(pt.y, rndv.y);
    return min(bx, by) * abs(rnd(seed.xy * 0.2));
}

// Function 572
float coloredNoise(float t, float fc, float df)
{
    // Noise peak centered around frequency fc
    // containing frequencies between fc-df and fc+df.
    // Modulate df-wide noise by an fc-frequency sinusoid
    return sin(TAU*fc*mod(t,1.))*noise(t*df);
}

// Function 573
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

// Function 574
float snoise( in vec3 x, const in float lod ) {
    float dim = 32.0 / exp2(lod);
    x = x * dim;
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    x = (p+f+0.5) / dim;
    return textureLod(iChannel0, x, lod).r;
}

// Function 575
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

// Function 576
vec4 Noise( in vec2 x )
{
    vec2 p = floor(x.xy);
    vec2 f = fract(x.xy);
	f = f*f*(3.0-2.0*f);
//	vec3 f2 = f*f; f = f*f2*(10.0-15.0*f+6.0*f2);

	vec2 uv = p.xy + f.xy;
	return texture( iChannel0, (uv+0.5)/256.0, -100.0 );
}

// Function 577
vec4 noiseInterpolate(const in vec4 x) 
{ 
    vec4 x2 = x * x;
    return x2 * x * (x * (x * 6.0 - 15.0) + 10.0); 
}

// Function 578
float noise(vec3 p){
    vec3 a = floor(p);
    vec3 d = fract(p);
    
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
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x) ;

    return o4.y * d.y + o4.x * (1.0 - d.y);
}

// Function 579
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

// Function 580
float warpedNoise(in vec2 uv){
    float scale = .5;
    float timeScale = 4.;
    return noise(uv*scale + noise(uv*scale + noise(uv*1.) +iTime/timeScale ));
}

// Function 581
float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1. + 3.0 * C.xxx;

    i = mod(i, 289.0);
    vec4 p = permute(permute(permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0)) +
            i.y + vec4(0.0, i1.y, i2.y, 1.0)) +
        i.x + vec4(0.0, i1.x, i2.x, 1.0));

    float n_ = 1.0 / 7.0; // N=7
    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z); //  mod(p,N*N)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_); // mod(j,N)

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
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
    vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1),
        dot(p2, x2), dot(p3, x3)));
}

// Function 582
float organicNoise(vec2 pos, vec2 scale, float density, vec2 phase, float contrast, float highlights, float shift, float seed)
{
    vec2 s = mix(vec2(1.0), scale - 1.0, density);
    float nx = perlinNoise(pos + phase, scale, seed);
    float ny = perlinNoise(pos, s, seed);

    float n = length(vec2(nx, ny) * mix(vec2(2.0, 0.0), vec2(0.0, 2.0), shift));
    n = pow(n, 1.0 + 8.0 * contrast) + (0.15 * highlights) / n;
    return n * 0.5;
}

// Function 583
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

// Function 584
vec2 noise(vec2 p)
{
    return texture(iChannel1,p,-100.0).xy;
}

// Function 585
float noise(vec2 pos) 
{
	return abs(fract(sin(dot(pos ,vec2(19.9*pos.x,28.633*pos.y))) * 1341.9453*pos.x));
}

// Function 586
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

// Function 587
float noiseSS(vec2 p){
    vec2 s = vec2(0.437,0.325);
    vec2 c = vec2(0.112,0.942);
    return (noiseS(p)+noiseS(p+s)+noiseS(p+c)+noiseS(p+s.yx)+noiseS(p+c.yx))/5.;
}

// Function 588
float noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash( i + vec2(0,0) ), f - vec2(0,0) ), 
                     dot( hash( i + vec2(1,0) ), f - vec2(1,0) ), u.x),
                mix( dot( hash( i + vec2(0,1) ), f - vec2(0,1) ), 
                     dot( hash( i + vec2(1,1) ), f - vec2(1,1) ), u.x), u.y);
}

// Function 589
vec4 noise(ivec2 p){
    return noise(ivec3(p, 0));
}

// Function 590
float noiseOld( in vec3 x )
{
    vec3 p = floor(x);
    vec3 f = fract(x);
	f = f*f*(3.0-2.0*f);
	vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;
	vec2 rg = textureLod( iChannel1, (uv+0.5)/256.0, 0.0).yx;
	return mix( rg.x, rg.y, f.z );
}

// Function 591
float smoothNoise(vec2 p) {
  vec2 inter = smoothstep(0., 1., fract(p));
  float s = mix(noise(sw(p)), noise(se(p)), inter.x);
  float n = mix(noise(nw(p)), noise(ne(p)), inter.x);
  return mix(s, n, inter.y);
  return noise(nw(p));
}

// Function 592
vec4 valueNoise(vec2 t, float w){
    vec2 fr = fract(t);
	return 
        mix(
            mix( 
                texture(iChannel1,vec2(floor(t.x), floor(t.y))/256.),
                texture(iChannel1,vec2(floor(t.x), floor(t.y) + 1.)/256.),
            	smoothstep(0.,1.,fr.y)
            ),
            mix( 
                texture(iChannel1,vec2(floor(t.x) + 1.,floor(t.y))/256.),
                texture(iChannel1,vec2(floor(t.x) + 1.,floor(t.y) + 1.)/256.),
            	smoothstep(0.,1.,fr.y)
            ),
            smoothstep(0.,1.,pow(fr.x, w)));
}

// Function 593
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

// Function 594
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

// Function 595
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

// Function 596
float noise( in vec2 p ) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f*f*(3.-2.*f);
    return mix( mix( dot( hash( i + vec2(0.,0.) ), f - vec2(0.,0.) ), 
                     dot( hash( i + vec2(1.,0.) ), f - vec2(1.,0.) ), u.x),
                mix( dot( hash( i + vec2(0.,1.) ), f - vec2(0.,1.) ), 
                     dot( hash( i + vec2(1.,1.) ), f - vec2(1.,1.) ), u.x), u.y);
}

// Function 597
float noise(vec2 x) {
    //https://thebookofshaders.com/11/
    vec2 i = floor(x);
    vec2 f = fract(x);
    float y = rand(vec2(.5));
    float a = rand(i);
    float b = rand(i + vec2(1., 0.0));
    float c = rand(i + vec2(0.0, 1.));
    float d = rand(i + vec2(1., 1.));
    vec2 u = smoothstep(0.0, 1.0, f);
    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// Function 598
float Noise(in vec2 p)
{
	vec2 f;
	f = fract(p);			// Separate integer from fractional
    p = floor(p);
    f = f*f*(3.0-2.0*f);	// Cosine interpolation approximation
    float res = mix(mix(Hash(p),
						Hash(p + vec2(1.0, 0.0)), f.x),
					mix(Hash(p + vec2(0.0, 1.0)),
						Hash(p + vec2(1.0, 1.0)), f.x), f.y);
    return res;
}

// Function 599
float osc_noise(float p) {
    p *= 20000.;
    float F = floor(p), f = fract(p);
    return mix(hash(F), hash(F+1.), f);
}

// Function 600
float gradientNoise(vec2 v)
{
    return fract(52.9829189 * fract(dot(v, vec2(0.06711056, 0.00583715))));
}

// Function 601
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

// Function 602
float SmoothNoise3d(vec3 p)
{
    vec3 fl = floor(p);
    vec3 fr = p - fl;
    
    vec3 ot = fr*fr*(3.0-2.0*fr);
    vec3 zt = 1.0f - ot;
    
    
    float result = 0.0f;
    
    result += hash13(fl + vec3(0,0,0)) * (zt.x * zt.y * zt.z);
    result += hash13(fl + vec3(1,0,0)) * (ot.x * zt.y * zt.z);

    result += hash13(fl + vec3(0,1,0)) * (zt.x * ot.y * zt.z);
    result += hash13(fl + vec3(1,1,0)) * (ot.x * ot.y * zt.z);

    result += hash13(fl + vec3(0,0,1)) * (zt.x * zt.y * ot.z);
    result += hash13(fl + vec3(1,0,1)) * (ot.x * zt.y * ot.z);

    result += hash13(fl + vec3(0,1,1)) * (zt.x * ot.y * ot.z);
    result += hash13(fl + vec3(1,1,1)) * (ot.x * ot.y * ot.z);

    return result;
}

// Function 603
float SmoothNoise(in vec2 o) 
{
	vec2 p = floor(o);
	vec2 f = fract(o);
		
	//float n = p.x + p.y*57.0;

	float a = hash12(p);
	float b = hash12(p+vec2(1,0));
	float c = hash12(p+vec2(0,1));
	float d = hash12(p+vec2(1,1));
	
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;
	
	vec2 t = 3.0 * f2 - 2.0 * f3;
	
	float u = t.x;
	float v = t.y;

	float res = a + (b-a)*u +(c-a)*v + (a-b+d-c)*u*v;
    
    return res;
}

// Function 604
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

// Function 605
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

// Function 606
float snoise(float x)
{
    float n = floor(x);
    float f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(hash1(n), hash1(n + 1.0), f);
}

// Function 607
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

// Function 608
vec3 Noisev3v2 (vec2 p)
{
  vec4 h;
  vec3 g;
  vec2 ip, fp, ffp;
  ip = floor (p);
  fp = fract (p);
  ffp = fp * fp * (3. - 2. * fp);
  h = Hashv4f (dot (ip, vec2 (1., 57.)));
  g = vec3 (h.y - h.x, h.z - h.x, h.x - h.y - h.z + h.w);
  return vec3 (h.x + dot (g.xy, ffp) + g.z * ffp.x * ffp.y,
     30. * fp * fp * (fp * fp - 2. * fp + 1.) * (g.xy + g.z * ffp.yx));
}

// Function 609
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

// Function 610
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

// Function 611
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

// Function 612
float noise( const in  vec3 x ) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.0-2.0*f);
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                   mix( hash(n+157.0), hash(n+158.0),f.x),f.y),
               mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                   mix( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

// Function 613
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

// Function 614
float noise(float p){
	float fl = floor(p);
  	float fc = fract(p);
	return mix(rand(fl), rand(fl + 1.0), fc);
}

// Function 615
float Simple360HDR_noiseOct(vec2 p){
  return
    Simple360HDR_noise(p)*0.5+
    Simple360HDR_noise(p*2.0+13.0)*0.25+
    Simple360HDR_noise(p*4.0+23.0)*0.15+
    Simple360HDR_noise(p*8.0+33.0)*0.10+
    Simple360HDR_noise(p*16.0+43.0)*0.05;
}

// Function 616
float noise(float p)
{
	float i = floor(p);
    float f = fract(p);
    
    float t = f * f * (3.0 - 2.0 * f);
    
    return lerp(f * hash11(i), (f - 1.0) * hash11(i + 1.0), t);
}

// Function 617
float noise(vec2 uv) {
    return fract(sin(uv.x*32.23 + uv.y*365.07) * 17.0125);
}

// Function 618
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

// Function 619
float noise( in vec2 p ) {
    vec2 i = floor(p), f = fract(p);
	vec2 u = f*f*f*(6.*f*f - 15.*f + 10.);
;
    return mix( mix( dot( hash( i + vec2(0.,0.) ), f - vec2(0.,0.) ), 
                     dot( hash( i + vec2(1.,0.) ), f - vec2(1.,0.) ), u.x),
                mix( dot( hash( i + vec2(0.,1.) ), f - vec2(0.,1.) ), 
                     dot( hash( i + vec2(1.,1.) ), f - vec2(1.,1.) ), u.x), u.y);
}

// Function 620
float gold_noise(in vec2 coordinate, in float seed){
    return fract(tan(distance(coordinate*(seed+GOLD_PHI), vec2(GOLD_PHI, GOLD_PI)))*GOLD_SQ2);
}

// Function 621
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

// Function 622
float mul_noise(vec3 x) {
    float n = 2.*noise(x);  x *= 2.1; // return n/2.;
         n *= 2.*noise(x);  x *= 1.9;
         n *= 2.*noise(x);  x *= 2.3;
         n *= 2.*noise(x);  x *= 1.9;
      //   n *= 2.*noise(x);
    return n/2.; 
}

// Function 623
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

// Function 624
vec4 bluenoise(vec2 U) {                               // blue noise
#define V(i,j)  noise(U+vec2(i,j))
  //U=floor(U/8.); 
    vec4 N = 8./9.* noise( U ) 
           - 1./9.*( V(-1,-1)+V(0,-1)+V(1,-1) +V(-1,0)+V(1,0) +V(-1,1)+V(0,1)+V(1,1) );  
    return N*2. + .5;   // or *1 to avoid saturation at the price of low contrast
}

// Function 625
float noised(vec2 uv)
{
    uv*=9.;
    return sin(uv.x)+cos(uv.y+dot(uv.x,uv.y));
    //return texture(iChannel0,uv).r;
}

// Function 626
float add_noise(vec3 x) {
    float n = noise(x)/2.;  x *= 2.1; // return n*2.;
         n += noise(x)/4.;  x *= 1.9;
         n += noise(x)/8.;  x *= 2.3;
         n += noise(x)/16.; x *= 1.9;
      //   n += noise(x)/32.;
    return n; 
}

// Function 627
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

// Function 628
float gnoise( in float p )
{
    int   i = int(floor(p));
    float f = fract(p);
	float u = f*f*(3.0-2.0*f);
    return mix( hash(i+0)*(f-0.0), 
                hash(i+1)*(f-1.0), u);
}

// Function 629
float perlinNoise3D(vec3 xyz, float freq, float amp, int octaves, int seed){
    float total = 0.0;
    float totalScale = 0.0;
    // current freq, amp, scale
    vec3 currFAS = vec3(freq, amp, amp);
    for(int i = 0; i < 5; i++){
        total += interpolatedNoise3D(abs(xyz) * currFAS.x, seed) * currFAS.y;
        totalScale += currFAS.z;
        currFAS *= vec3(2.0, 0.5, 0.5);
        if (i >= octaves) break;
    }
    return amp * (total / totalScale);
}

// Function 630
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

// Function 631
float noise13( in vec3 p )
{
    vec3 i = floor( p );
    vec3 f = fract( p );
	
	vec3 u = f*f*(3.0-2.0*f);

    return mix( 
        mix(
                mix( hash13( i + vec3(0.0,0.0,0.0) ), 
                     hash13( i + vec3(1.0,0.0,0.0) ), u.x),
                mix( hash13( i + vec3(0.0,1.0,0.0) ), 
                     hash13( i + vec3(1.0,1.0,0.0) ), u.x)
               , u.y),
              mix(
                mix( hash13( i + vec3(0.0,0.0,1.0) ), 
                     hash13( i + vec3(1.0,0.0,1.0) ), u.x),
                mix( hash13( i + vec3(0.0,1.0,1.0) ), 
                     hash13( i + vec3(1.0,1.0,1.0) ), u.x)
               , u.y)
        
         , u.z)
        ;
}

// Function 632
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

// Function 633
float Noise( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    float res = mix(mix( Hash12(p),          Hash12(p + add.xy),f.x),
                    mix( Hash12(p + add.yx), Hash12(p + add.xx),f.x),f.y);
    return res;
}

// Function 634
float Noise( in vec2 p )
{
    return 0.5*(cos(6.2831*p.x) + cos(6.2831*p.y));
}

// Function 635
float noise(vec2 p) {
  return random(p.x + p.y*10000.);
}

// Function 636
float valueNoise1duw(float samplePoint, float wrap) {
    float pointI = floor(samplePoint);
    return mix(hash11(mod(pointI, wrap)),hash11(mod(pointI + 1.0 ,wrap)),valueNoiseFilter(fract(samplePoint)));
}

// Function 637
float vnoise1(float p) {
    float i = floor(p);
	float f = fract(p);
    
    float a = hash11(i);
    float b = hash11(i + 1.0);
    
    float u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u);
}

// Function 638
vec3 noise( in float x )
{
    float p = floor(x);
    float f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix( hash3(p+0.0), hash3(p+1.0),f);
}

// Function 639
float noise( in vec2 x ) {
    vec2 f = fract(x);
    vec2 u = f*f*(3.0-2.0*f);
    
    vec2 p = vec2(floor(x));
    float a = hash12( (p+vec2(0,0)) );
	float b = hash12( (p+vec2(1,0)) );
	float c = hash12( (p+vec2(0,1)) );
	float d = hash12( (p+vec2(1,1)) );
    
	return a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y;
}

// Function 640
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

