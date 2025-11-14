// Reusable Distortion Effect Functions
// Automatically extracted from effect-related shaders

// Function 1
vec2 Distort(vec2 uv, vec2 focusPos) {
    // the bounds can be calculated on the CPU
    vec2 boundStart = GetBoundStart(focusPos);
    vec2 distortedSize = GetDistortedSize(focusPos);
    
    return (RadialDistortion(uv * distortedSize + boundStart)) + focusPos;
}

// Function 2
vec2 lensDistortion(vec2 uv) {
    // radial distortion coefficients
    float k1 = 0.126;
    float k2 = 0.004;
    float k3 = 0.001;
    float k4 = 0.;
    float k5 = 0.;
    float k6 = 0.;
    // tangential distortion coefficients
    float p1 = 0.001;
    float p2 = 0.001;
    
    float x = uv.x;
    float y = uv.y;
    float r2 = x*x + y*y;
    
    float num = 1. + (k1 + (k2 + k3*r2)*r2)*r2;
    float den = 1. + (k4 + (k5 + k6*r2)*r2)*r2;
    float rd = num / den;
    
    vec2 td;
    td.x = 2.*p1*x*y + p2*(r2 + 2.*x*x);
    td.y = p1*(r2 + 2.*y*y) + 2.*p2*x*y;
    
    return uv*rd + td;
}

// Function 3
vec2 BarrelDistortion (vec2 uv)
{
    //angle 
    float theta = atan(uv.y, uv.x);
    //ammount
    float anim_effect_1 = ((sin (iTime + 0.3))*0.005);
    float radius = length(uv) + anim_effect_1;
    //contrast
    radius = pow(radius, barrelContrast);
    
    uv.x = radius * cos(theta);
    uv.y = radius * sin(theta);
    color.r = 0.5 * (uv.y +1.0);
    color.b = radius;
    return 0.5 * (uv + 1.0);
}

// Function 4
vec2 warp(vec2 uv, vec2 warp_uvs, float strength, sampler2D tex)
{
#ifdef INVERT_WARP_TEX
    vec4 w = 1.0 - texture(tex, warp_uvs);
#else
    vec4 w = texture(tex, warp_uvs);
#endif
    return uv + strength * w.xy;
}

// Function 5
vec2 easyBarrelDistortion(vec2 uv)
{
    float demoScale = 1.1;
    uv *= demoScale;
    float th = atan(uv.x, uv.y);
    float barrelDistortion = 1.2;
    float r = pow(sqrt(uv.x*uv.x + uv.y*uv.y), barrelDistortion);
    uv.x = r * sin(th);
    uv.y = r * cos(th);
    return uv;
}

// Function 6
vec3 simulate_lightmap_distortion(vec3 surface_point)
{
    surface_point = floor(surface_point);
    surface_point *= 1./LIGHTMAP_SCALE;
    vec3 f = fract(surface_point + .5);
    return (surface_point + f - smoothen(f)) * LIGHTMAP_SCALE;
}

// Function 7
vec2 barrelDistortion(vec2 loc, float magnitude, float radius)
{
    float theta = atan(loc.y,loc.x);
    radius = pow(radius, magnitude);
    loc.x = radius * cos(theta);
    loc.y = radius * sin(theta);
    
    return 0.5 * (loc + 1.0);
}

// Function 8
vec2 distort( vec2 uv, float t, vec2 min_distort, vec2 max_distort )
{
    vec2 dist = mix( min_distort, max_distort, t );
    
    #ifdef conrady
    return brownConradyDistortion( uv, 75.0 * dist.x );
    #endif
    
    #ifdef barrel
    return barrelDistortion( uv,1.75*dist );
    #endif

}

// Function 9
vec2 barrelDistort(in vec2 p, in vec2 alpha) {
    return p / (1.0 - alpha * dot(p, p));
}

// Function 10
vec2 vortex_pair_warp(vec2 uv, vec2 pos, vec2 vel)
{
    vec2 aspect = vec2(1.,iResolution.y/iResolution.x);
    float ramp = 4.;

    float d = 0.125;

    float l = length(vel);
    vec2 p1 = pos;
    vec2 p2 = pos;

    if(l > 0.){
        vec2 normal = normalize(vel.yx * vec2(-1., 1.))/aspect;
        p1 = pos - normal * d / 2.;
        p2 = pos + normal * d / 2.;
    }

    float w = l / d * 2.;

    // two overlapping rotations that would annihilate when they were not displaced.
    vec2 circle1 = vortex_warp(uv, p1, d, ramp, vec2(cos(w),sin(w)));
    vec2 circle2 = vortex_warp(uv, p2, d, ramp, vec2(cos(-w),sin(-w)));
    return (circle1 + circle2) / 2.;
}

// Function 11
vec2 distort(float x, float x2, float x3, float x4, float y, float y2, float y3, float y4) {
    return vec2(
        	        0.027935f +
            x*      1.095561f +
            y*      0.012740f +
            x2*    -0.039869f +
            x*y*    0.124914f +
            y2*     0.002292f +
            x3*    -0.292835f +
            x2*y*  -0.034783f +
            x*y2*  0.012176f + // un fisheye
            y3*     0.004875f +
            x4*   -0.036597f + // unsquish towards eyes
            x3*y*  -0.110877f +
            x2*y2* -0.043108f +
            x*y3*  -0.062561f +
            y4*     0.019749f
            ,       0.016951f +
            x*     0.042731f + // sad face
            y*      1.076121f +
            x2*     0.185902f +
            x*y*   0.060663f + // like looking at a plane side on
            y2*     0.029832f +
            x3*    -0.044401f +
            x2*y*  -0.349245f +
            x*y2*  -0.008731f +
            y3*    -0.212708f +
            x4*    -0.175194f +
            x3*y*  -0.064730f +
            x2*y2* -0.232783f +
            x*y3*   0.054503f +
            y4*    -0.127740f
	);
}

// Function 12
vec2 vortex_warp(vec2 uv, vec2 pos, float size, float ramp, vec2 rot, vec3 iResolution)
{
    vec2 aspect = vec2(1.,iResolution.y/iResolution.x);

    vec2 pos_correct = 0.5 + (pos - 0.5);
    vec2 rot_uv = pos_correct + complex_mul((uv - pos_correct)*aspect, rot)/aspect;
    float _filter = warpFilter(uv, pos_correct, size, ramp, iResolution);
    return mix(uv, rot_uv, _filter);
}

// Function 13
vec2 inverseBarrelDistortion(vec2 uv, float distortion) {    
    uv -= .5;
    
    float b = distortion;
    float l = length(uv);
    
    float x0 = pow(9.*b*b*l + sqrt(3.) * sqrt(27.*b*b*b*b*l*l + 4.*b*b*b), 1./3.);
    float x = x0 / (pow(2., 1./3.) * pow(3., 2./3.) * b) - pow(2./3., 1./3.) / x0;
       
    return uv * (x / l) + .5;
}

// Function 14
vec2 warp(vec2 uv, vec2 p, float offset)
{
	uv -= p;
	float minkpow = WARP_ORDER;
	float d = pow(pow(abs(uv.x),minkpow)+pow(abs(uv.y),minkpow),1./minkpow);
	uv /= pow(d,2.)*1.-offset;
	uv += p;
	return uv;
}

// Function 15
vec2 distort( vec2 uv, float t, vec2 min_distort, vec2 max_distort )
{
    vec2 dist = mix( min_distort, max_distort, t );
    //return radialdistort( uv, 2.0 * dist );
    //return barrelDistortion( uv, 1.75 * dist ); //distortion at center
    return brownConradyDistortion( uv, 75.0 * dist.x );
}

// Function 16
float warpnoise3(vec3 p) {
    float f = 0.0;
    const float c1 = 0.06;
    const float tc = 0.05;
    vec3 q = vec3(fBm(p + tc*iTime), 
                  fBm(p + vec3(5.1, 1.3, 2.2) + tc*iTime), 
                  fBm(p + vec3(3.4, 4.8, 5.5) + tc*iTime));
    
    return 1.2*fBm(p + c1*q);
}

// Function 17
vec2 brownConradyDistortion(in vec2 uv, in float k1)
{
    uv = uv * 2.0 - 1.0;	// brown conrady takes [-1:1]

    // positive values of K1 give barrel distortion, negative give pincushion
    float r2 = uv.x*uv.x + uv.y*uv.y;
    uv *= 1.0 + k1 * r2;
    
    // tangential distortion (due to off center lens elements)
    // is not modeled in this function, but if it was, the terms would go here
    
    uv = (uv * .5 + .5);	// restore -> [0:1]
    return uv;
}

// Function 18
vec2 radialdistort(vec2 coord, vec2 amt)
{
	vec2 cc = coord - 0.5;
	return coord + 2.0 * cc * amt;
}

// Function 19
vec2 vortex_warp(vec2 uv, vec2 pos, float size, float ramp, vec2 rot)
{
    vec2 aspect = vec2(1.,iResolution.y/iResolution.x);

    vec2 pos_correct = 0.5 + (pos - 0.5);
    vec2 rot_uv = pos_correct + complex_mul((uv - pos_correct)*aspect, rot)/aspect;
    float _filter = warpFilter(uv, pos_correct, size, ramp);
    return mix(uv, rot_uv, _filter);
}

// Function 20
vec2 brownConradyDistortion(vec2 uv, float scalar)
{
// AH!!!    uv = uv * 2.0 - 1.0;
    uv = (uv - 0.5 ) * 2.0;
    
    if( true )
    {
        // positive values of K1 give barrel distortion, negative give pincushion
        float barrelDistortion1 = -0.02 * scalar; // K1 in text books
        float barrelDistortion2 = 0.0 * scalar; // K2 in text books

        float r2 = dot(uv,uv);
        uv *= 1.0 + barrelDistortion1 * r2 + barrelDistortion2 * r2 * r2;
        //uv *= 1.0 + barrelDistortion1 * r2;
    }
    
    // tangential distortion (due to off center lens elements)
    // is not modeled in this function, but if it was, the terms would go here
//    return uv * 0.5 + 0.5;
   return (uv / 2.0) + 0.5;
}

// Function 21
vec2 barrelDistortion( vec2 coord, float amt, float zoom )
{ // based on gtoledo3 (XslGz8)
  // added zoomimg
	vec2 cc = coord - 0.6;
    vec2 p = cc * zoom;
    coord = p + 0.6;
	float dist = dot( cc, cc );
	return coord + cc * dist * amt;
}

// Function 22
vec2 HmdWarp(vec2 texIn, vec2 LensCenter)
{
   vec2 theta = (texIn - LensCenter) * ScaleIn;
   float rSq = theta.x * theta.x + theta.y * theta.y;
   vec2 theta1 = theta * (HmdWarpParam.x + HmdWarpParam.y * rSq + HmdWarpParam.z * rSq * rSq + HmdWarpParam.w * rSq * rSq * rSq);
   return LensCenter + Scale * theta1;
}

// Function 23
vec2 barrel_distort_inv( vec2 ec, float a ) { float ec2 = dot( ec, ec ); float u = a * ( a + 1. ) * ec2; return u < 1. / 4096. ? ec * ( a + 1. ) : ec / ( 2. * a * ec2 ) * ( sqrt( 4. * u + 1. ) - 1. ); }

// Function 24
vec2 lens_distort_coords(vec2 input_coords){
	// Radial distortion. See https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
	// This code is based on modules/calib3d/src/undistort.cpp in the function
	// cvUndistortPointsInternal where they do an iterative undistort that compares the
	// real values with computed distorted ones. The changes are mostly to try make
	// it obvious to the GLSL compiler what we are doing in the hope it can optimize the
	// multiplies more effectively. This should be disassembled/tested at some point.
	float x = input_coords.x;
	float y = input_coords.y;
	
	float r2 = dot(input_coords, input_coords);  // u.u = |u|^2
	float r4 = r2 * r2;
	float r6 = r2 * r4;
	
	float a1 = 2.0 * x * y;
	float a2 = r2 + 2.0 * x * x;
	float a3 = r2 + 2.0 * y * y;
	float cdist = 1.0 + dot(lens_radial_distortion, vec3(r2, r4, r6));
	vec2 tangental_dist = vec2(
		dot(lens_tangental_distortion, vec2(a1, a2)),
		dot(lens_tangental_distortion, vec2(a3, a1))
	);
	vec2 distort_amt = vec2(cdist) + tangental_dist;

	return input_coords * distort_amt * lens_zoom_factor;
}

// Function 25
float fbmWarp2(in vec2 st, out vec2 q, out vec2 r)  {
  q.x = fbm(st + vec2(0.0,0.0));
  q.y = fbm(st + vec2(5.2,1.3));

  r.x = fbm( st + 4.0*q + vec2(1.7,9.2) + 0.7*iTime);
  r.y = fbm( st + 4.0*q + vec2(8.3,2.8) + 0.7*iTime);

  return fbm( st + 4.0*r);
}

// Function 26
vec2 DistortUV( vec2 vUV, float f )
{
    vUV -= 0.5;

    float fScale = 0.0005;
    
    float r1 = 1. + f * fScale;
    
    vec3 v = vec3(vUV, sqrt( r1 * r1 - dot(vUV, vUV) ) );
    
    v = normalize(v);
    vUV = v.xy;
    
    
    vUV += 0.5;
    
    return vUV;
}

// Function 27
vec2 Undistort(vec2 uv, vec2 focusPos) {
    vec2 boundStart = GetBoundStart(focusPos);
    vec2 distortedSize = GetDistortedSize(focusPos);

    return (InverseRadialDistortion(uv - focusPos) - boundStart) / distortedSize;
}

// Function 28
vec2 barrelDistortion(vec2 uv)
{   
    float distortion = 0.2;
    float r = uv.x*uv.x + uv.y*uv.y;
    uv *= 1.6 + distortion * r + distortion * r * r;
    return uv;
}

// Function 29
vec2 barrelDistortion(vec2 coord, float amt) {
	vec2 cc = coord - 0.5;
	float dist = dot(cc, cc);
	//return coord + cc * (dist*dist)  * amt;
	return coord + cc * dist * amt;

    }

// Function 30
vec2 distort(in vec2 coord, float mag, float phase, float freq)
{
    vec2 noiseSampleDirection = vec2(1.0, 0.319);
    vec2 sampleLocation1 = noiseSampleDirection * phase;
    vec2 sampleLocation2 = vec2(1.0, 0.8) - noiseSampleDirection * phase;
    vec3 noise1 = texture(iChannel0, sampleLocation1).rgb;
    vec3 noise2 = texture(iChannel0, sampleLocation2).rgb;    
    
    vec3 n1 = hash31(phase);
    vec3 n2 = hash31(1.0+phase);

    // Would rather distort relative to world center rather than fixed directions
    	// (you'll notice the waves will lean/move toward the bottom left corner
    float dx = 0.0 + 0.6 * waves(coord,
                                    vec2(1.9 + 0.4 * n1.r, 1.9 + 0.4 * n1.g) * 3.3,
                                    vec2(5.7 + 1.4 * n1.b, 5.7 + 1.4 * n2.r) * 2.8,
                                    vec2(n1.r - n2.r, n1.g + n2.b) * 5.0,
                                    vec2(1.1), iTime*freq);
    float dy = 0.5 + 0.7 * waves(coord,
                                    vec2(-1.7 - 0.9 * n2.g, 1.7 + 0.9 * n2.b) * 3.1,
                                    vec2(5.9 + 0.8 * n1.g, -5.9 - 0.8 * n1.b) * 3.7,
                                    vec2(n1.g + n2.g, n1.b - n2.r) * 5.0,
                                    vec2(-0.9), iTime*freq);
   
   float amt = dx*dy;

   return coord + normalize(vec2(dx,dy))*amt*mag;
}

// Function 31
void distort(inout vec2 p)
{
    float theta = atan(p.y, p.x);
    float radius = pow(length(p), BARREL_POWER);
    p = radius * vec2(cos(theta), sin(theta));
    p = (p + 1.0) * 0.5;
}

// Function 32
vec2 distort(vec2 uv){
	float ang=atan(uv.x,uv.y)/4.;
	return uv*mat2(cos(ang),-sin(ang),sin(ang),cos(ang));
    //return vec2(ang/3.14,1./length(uv));
}

// Function 33
void BarrelDistortion(inout vec3 r,float d){//rey,degree
;r.z/=d;
;r.z=r.z*r.z-dot(r.xy,r.xy)//fisheye-lens
;r.z=d*sqrt(r.z);}

// Function 34
vec2 distort(vec2 p)
{
    float theta  = atan(p.y, p.x);
    float radius = length(p);
    radius = pow(radius, barrelDistortionPower);
    p.x = radius * cos(theta);
    p.y = radius * sin(theta);
    return 0.5 * (p + 1.0);
}

// Function 35
vec2 UndistortDerivative(vec2 uv, vec2 focusPos) {
    vec2 distortedSize = GetDistortedSize(focusPos);
    
    return InverseRadialDistortionDerivative(uv - focusPos) / distortedSize;
}

// Function 36
vec2 brownConradyDistortion(vec2 uv, float dist)
{
    uv = uv * 2.0 - 1.0;
    float barrelDistortion1 = 0.1 * dist; // K1 in text books
    float barrelDistortion2 = -0.025 * dist; // K2 in text books

    float r2 = dot(uv,uv);
    uv *= 1.0 + barrelDistortion1 * r2 + barrelDistortion2 * r2 * r2;
    
    return uv * 0.5 + 0.5;
}

// Function 37
vec2 undistort(vec2 u) {
    float r2 = dot(u,u);
    float r4 = r2*r2;
    float r6 = r4*r2;
    vec2 d = u*(1.0 + K1*r2 + K2*r4 + K3*r6);
    d.x = d.x + P2*(r2 + 2.0*u.x*u.x) + 2.0*P1*u.x*u.y;
	d.y = d.y + P1*(r2 + 2.0*u.y*u.y) + 2.0*P2*u.x*u.y;
    return d;
}

// Function 38
void BarrelDistortion( inout vec3 ray, float degree )
{
	ray.z /= degree;
	ray.z = ( ray.z*ray.z - dot(ray.xy,ray.xy) );
	ray.z = degree*sqrt(ray.z);
}

// Function 39
vec2 distort(vec2 vector)
{
	float theta = atan(vector.y, vector.x);
	float radius = length(vector);
	radius = pow(radius, barrelPower);
	vector.x = radius * cos(theta);
	vector.y = radius * sin(theta);
	return 0.5 * ((mouse - vector) + 1.0);
}

// Function 40
vec2 barrelDistortion( vec2 p, vec2 amt )
{
    p = 2.0 * p - 1.0;
    const float maxBarrelPower = 5.0;
    float theta  = atan(p.y, p.x);
    vec2 radius = vec2( length(p) );
    radius = pow(radius, 1.0 + maxBarrelPower * amt);
    p.x = radius.x * cos(theta);
    p.y = radius.y * sin(theta);

    return p * 0.5 + 0.5;
}

// Function 41
vec2 getDistortion(in float LRValue, in float isFunctionForRightEye, in vec2 uv) {
	//sample at c,r or -c,r depending on which eye (LRValue) and which the function was designed for (isFunctionForRightEye)
    float lrx = uv.x * LRValue * isFunctionForRightEye;
    float x2 = lrx * lrx, x3 = x2 * lrx, x4 = x3 * lrx,
          y2 = uv.y * uv.y, y3 = y2 * uv.y, y4 = y3 * uv.y;
    vec2 res = vec2(lrx, uv.y) - distort(lrx, x2, x3, x4, uv.y, y2, y3, y4);
    res.x *= LRValue * isFunctionForRightEye;
    //texCoords are c, r from -1 to 1
    return vec2(uv.x, uv.y) + res;
}

// Function 42
void BarrelDistortion( inout vec3 ray, float degree )
{
	ray.z /= degree;
	ray.z = ( ray.z*ray.z - dot(ray.xy,ray.xy) ); // fisheye
	ray.z = degree*sqrt(ray.z);
}

// Function 43
float DistortDomain (vec2 p, out vec2 fDstr, out vec2 sDstr)
{
	fDstr = vec2 (
    	FBM (p + vec2 (.5, 0.3)),
        FBM (p + vec2 (2.2, 0.1)  * iTime * 0.03)
    );
             
    sDstr = vec2 (
        FBM (p + 5.0 * fDstr + vec2 (2.5, 0.2)  * iTime * 0.1),
        FBM (p + 5.0 * fDstr + vec2 (-0.2, .3)  * iTime * 0.5)
    );
             
   	return FBM (p + 5.0 * sDstr);
}

// Function 44
float dowarp ( in vec2 q, out vec2 a, out vec2 b )
{
	float ang=0.;
	ang = 1.2345 * sin (33.33); //0.015*iTime);
	mat2 m1 = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
	ang = 0.2345 * sin (66.66); //0.021*iTime);
	mat2 m2 = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));

	a = vec2( marble(m1*q), marble(m2*q+vec2(1.12,0.654)) );

	ang = 0.543 * cos (13.33); //0.011*iTime);
	m1 = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
	ang = 1.128 * cos (53.33); //0.018*iTime);
	m2 = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));

	b = vec2( marble( m2*(q + a)), marble( m1*(q + a) ) );
	
	return marble( q + b +vec2(0.32,1.654));
}

// Function 45
vec2 barrelDistortion( vec2 p, vec2 amt )
{
    p = 2.0 * p - 1.0;

    /*
    const float maxBarrelPower = 5.0;
	//note: http://glsl.heroku.com/e#3290.7 , copied from Little Grasshopper
    float theta  = atan(p.y, p.x);
    vec2 radius = vec2( length(p) );
    radius = pow(radius, 1.0 + maxBarrelPower * amt);
    p.x = radius.x * cos(theta);
    p.y = radius.y * sin(theta);

	/*/
    // much faster version
    //const float maxBarrelPower = 5.0;
    //float radius = length(p);
    float maxBarrelPower = sqrt(5.0);
    float radius = dot(p,p); //faster but doesn't match above accurately
    p *= pow(vec2(radius), maxBarrelPower * amt);
	/* */

    return p * 0.5 + 0.5;
}

// Function 46
vec2 barrelDistortion( vec2 coord, float amt, float zoom )
{ // based on gtoledo3 (XslGz8)
  // added zoomimg
	vec2 cc = coord - 0.5;
    vec2 p = cc * zoom;
    coord = p + 0.5;
	float dist = dot( cc, cc );
	return coord + cc * dist * amt;
}

// Function 47
vec2 brownConradyDistortion(vec2 uv)
{
    // positive values of K1 give barrel distortion, negative give pincushion
    float barrelDistortion1 = 0.15; // K1 in text books
    float barrelDistortion2 = 0.0; // K2 in text books
    float r2 = uv.x*uv.x + uv.y*uv.y;
    uv *= 1.0 + barrelDistortion1 * r2 + barrelDistortion2 * r2 * r2;
    
    // tangential distortion (due to off center lens elements)
    // is not modeled in this function, but if it was, the terms would go here
    return uv;
}

// Function 48
vec2 warpSpeed(float time, float gravity, vec2 pos)
{
	// Do some things to stretch out the timescale based on 2D position and actual time.
	return vec2(-time*5.55 + sin(pos.x*10.0*sin(time*.2))*.4, 
		time*gravity+sin(pos.y*10.0*sin(time*.4))*.4);}

// Function 49
vec3 distort(sampler2D sampler, vec2 uv, float edgeSize)
{
    vec2 pixel = vec2(1.0) / iResolution.xy;
    vec3 field = rgb2hsv(edge(sampler, uv, edgeSize));
    vec2 distort = pixel * sin((field.rb) * PI * 2.0);
    float shiftx = noise(vec2(quantize(uv.y + 31.5, iResolution.y / TILE_SIZE) * iTime, fract(iTime) * 300.0));
    float shifty = noise(vec2(quantize(uv.x + 11.5, iResolution.x / TILE_SIZE) * iTime, fract(iTime) * 100.0));
    vec3 rgb = texture(sampler, uv + (distort + (pixel - pixel / 2.0) * vec2(shiftx, shifty) * (50.0 + 100.0 * Amount)) * Amount).rgb;
    vec3 hsv = rgb2hsv(rgb);
    hsv.y = mod(hsv.y + shifty * pow(Amount, 5.0) * 0.25, 1.0);
    return posterize(hsv2rgb(hsv), floor(mix(256.0, pow(1.0 - hsv.z - 0.5, 2.0) * 64.0 * shiftx + 4.0, 1.0 - pow(1.0 - Amount, 5.0))));
}

// Function 50
vec2 InverseRadialDistortion(vec2 xy) {
    vec2 scaledXY = xy * DISTORTION_STRENGTH / FOV_REGION_SCALE;
    float scaledRadius = length(scaledXY);
    return inverse_distortion_fn(scaledRadius) * scaledXY / scaledRadius;
}

// Function 51
vec2 vortex_warp(vec2 uv, vec2 pos, float size, float ramp, vec2 rot)
{
    vec2 aspect = vec2(1.,iResolution.y/iResolution.x);

    vec2 pos_correct = 0.5 + (pos - 0.5);
    vec2 rot_uv = pos_correct + complex_mul((uv - pos_correct)*aspect, rot)/aspect;
    float filtered = warpFilter(uv, pos_correct, size, ramp);
    return mix(uv, rot_uv, filtered);
}

// Function 52
vec2 barrelDistort(vec2 pos, float power)
{
	float t = atan(pos.y, pos.x);
	float r = pow(length(pos), power);
	pos.x   = r * cos(t);
	pos.y   = r * sin(t);
	return 0.5 * (pos + 1.0);
}

// Function 53
vec2 BarrelDistortion(vec2 uv, float mouse, float k){
    vec2 center = uv - 0.5;

    float radius = length(center);
    
    float radiusp2 = (k + pow(radius, 2.0)) * radius;
    
    float f = mix(radius, radiusp2, mouse);
    
   	float a = atan(center.y, center.x);
    
    return vec2(cos(a), sin(a)) * f + 0.5;
}

// Function 54
vec2 barrelDistortion(vec2 uv, float distortion) {    
    uv -= .5;
    uv *= 1. + dot(uv, uv) * distortion;
    return uv + .5;
}

// Function 55
float vfbmWarp (vec2 p) {
  vec2 q = vec2(0);
  vec2 s = vec2(0);
  vec2 r = vec2(0);

  return vfbmWarp(p, q, s, r);
}

// Function 56
vec2 GetDistortedSize(vec2 focusPos) {
    return GetBoundEnd(focusPos) - GetBoundStart(focusPos);
}

// Function 57
vec2 vortex_warp(vec2 uv, vec2 pos, float size, float ramp, vec2 rot)
{
    vec2 aspect = vec2(1.,iResolution.y/iResolution.x);

    vec2 pos_correct = 0.5 + (pos - 0.5);
    vec2 rot_uv = pos_correct + complex_mul((uv - pos_correct)*aspect, rot)/aspect;
    float filterv = warpFilter(uv, pos_correct, size, ramp);
    return mix(uv, rot_uv, filterv);
}

// Function 58
vec2 distort(vec2 p, float power) {
    // Convert to polar coords:
    float theta  = atan(p.y, p.x);
    float radius = length(p);

    // Distort:
    radius = pow(radius, power);

    // Convert back to Cartesian:
    p.x = radius * cos(theta);
    p.y = radius * sin(theta);

    return p;
}

// Function 59
vec2 GetDistortionTexelOffset(vec2 offsetDirection, float offsetDistance, float time)
{
    float progress = mod(time, EffectDuration) / EffectDuration;
    
    float halfWidth = EffectWidth / 2.0;
    float lower = 1.0 - smoothstep(progress - halfWidth, progress, offsetDistance);
    float upper = smoothstep(progress, progress + halfWidth, offsetDistance);
    
    float band = 1.0 - (upper + lower);
    
    
    float strength = 1.0 - progress;
    float fadeStrength = smoothstep(0.0, EffectFadeInTimeFactor, progress);
    
    float distortion = band * strength * fadeStrength;
    
    
    return distortion * offsetDirection * EffectMaxTexelOffset;
}

// Function 60
void WarpSpace(inout vec3 eyevec, inout vec3 raypos)
{
    vec3 origin = vec3(0.0, 0.0, 0.0);

    float singularityDist = distance(raypos, origin);
    float warpFactor = 1.0 / (pow(singularityDist, 2.0) + 0.000001);

    vec3 singularityVector = normalize(origin - raypos);
    
    float warpAmount = 5.0;

    eyevec = normalize(eyevec + singularityVector * warpFactor * warpAmount / float(ITERATIONS));
}

// Function 61
vec2 barrelDistortion(vec2 coord, float amt, float zoom)
{ // based on gtoledo3 (XslGz8)
  // added zoomimg
	vec2 cc = coord-0.5;
    vec2 p = cc*zoom;
    coord = p+0.5;
	float dist = dot(cc, cc);
	return coord +cc*dist*amt;
}

// Function 62
vec2 screenDistort(vec2 uv)
{
	uv -= vec2(.5,.5);
	uv = uv*1.2*(1./1.2+2.*uv.x*uv.x*uv.y*uv.y);
	uv += vec2(.5,.5);
	return uv;
}

// Function 63
vec2 distort(vec2 p)
{
    vec2 d = p - center;
    float theta  = atan(d.y, d.x);
    float rad = length(d);
    rad = pow(rad, 1. + barrelPower);
    d.x = rad * cos(theta);
    d.y = rad * sin(theta);
    return d;
}

// Function 64
void BarrelDistortion( inout vec3 ray, float degree )
{
	// would love to get some disperson on this, but that means more rays
	ray.z /= degree;
	ray.z = ( ray.z*ray.z - dot(ray.xy,ray.xy) ); // fisheye
	ray.z = degree*sqrt(ray.z);
}

// Function 65
vec4 textureDistorted(const in sampler2D tex, const in vec2 texCoord, const in vec2 direction, const in vec3 distortion) {
  return vec4(textureLimited(tex, (texCoord + (direction * distortion.r))).r,
              textureLimited(tex, (texCoord + (direction * distortion.g))).g,
							textureLimited(tex, (texCoord + (direction * distortion.b))).b,
              1.0);
}

// Function 66
float Distort(vec2 uv) 
{
    float p1 = sin(uv.x);
    float p2 = sin(uv.y);
    float p3 = sin(uv.x - uv.y);
    float p4 = sin(length(uv));
    return p1 + p2 + p3 + p4;
}

// Function 67
vec2 scandistort(vec2 uv) 
{
	float scan1 = clamp(cos(uv.y * 2.0 + iTime), 0.0, 1.0);
	float scan2 = clamp(cos(uv.y * 2.0 + iTime + 4.0) * 10.0, 0.0, 1.0);
	float amount = scan1 * scan2 * uv.x; 
	uv.x -= 0.05 * mix( texture(iChannel1, vec2(uv.x, amount)).x * amount, amount, 0.9 );
    
	return uv;
}

// Function 68
vec2 GetOptimalDistortedResolution(vec2 focusPos, vec2 sourceResolution) {
    vec2 gradientOnFocus = vec2(UndistortDerivative(focusPos + vec2(EPS, 0.), focusPos).x,
                                UndistortDerivative(focusPos + vec2(0., EPS), focusPos).y);
    return sourceResolution / gradientOnFocus;
}

// Function 69
float vfbmWarp (vec2 p, out vec2 q, out vec2 s, vec2 r) {
  const float scale = 4.0;
  const float angle = 0.01 * PI;
  const float si = sin(angle);
  const float c = cos(angle);
  const mat2 rot = mat2(c, si, -si, c);

  q = vec2(
        vfbm4(p + vec2(0.0, 0.0)),
        vfbm4(p + vec2(3.2, 34.5)));
  q *= rot;

  s = vec2(
        vfbm4(p + scale * q + vec2(23.9, 234.0)),
        vfbm4(p + scale * q + vec2(7.0, -232.0)));
  s *= rot;

  return vfbm6(p + scale * s);
}

// Function 70
vec2 lensDistort(vec2 c, float factor)
{
    // [0;1] -> [-1;1]
    c = (c - 0.5) * 2.0;
    // [-1;1] -> film frame size
    c.y *= 3.0/4.0;
    // distort
    c /= 1.0 + dot(c, c) * -factor + 1.6 * factor;
    // film frame size -> [-1;1]
    c.y *= 4.0/3.0;
    // [-1;1] -> [0;1]
    c = c * 0.5 + 0.5;
    return c;
}

// Function 71
vec2 lens_distortion(vec2 r, float alpha) {
    return r * (1.0 - alpha * dot(r, r));
    
}

// Function 72
float barrel_distort_rate( float ec2, float a ) { return ( a + 1. + a * ec2 ) / square( a + 1. - a * ec2 ); }

// Function 73
float DistortingFormula(in float hueIN){
    vec4 abcd = ReturnABCD(hueIN); 
    return abcd.z + ((hueIN - abcd.x) * ((abcd.w - abcd.z)/(abcd.y - abcd.x) ));
}

// Function 74
vec2 vortex_pair_warp(vec2 uv, vec2 pos, vec2 vel)
{
    vec2 aspect = vec2(1.,iResolution.y/iResolution.x);
    float ramp = 5.;

    float d = 0.2;

    float l = length(vel);
    vec2 p1 = pos;
    vec2 p2 = pos;

    if(l > 0.){
        vec2 normal = normalize(vel.yx * vec2(-1., 1.))/aspect;
        p1 = pos - normal * d / 2.;
        p2 = pos + normal * d / 2.;
    }

    float w = l / d * 2.;

    // two overlapping rotations that would annihilate when they were not displaced.
    vec2 circle1 = vortex_warp(uv, p1, d, ramp, vec2(cos(w),sin(w)));
    vec2 circle2 = vortex_warp(uv, p2, d, ramp, vec2(cos(-w),sin(-w)));
    return (circle1 + circle2) / 2.;
}

// Function 75
vec2 distortUV(vec2 uv, vec2 nUV)
{
    float intensity = 0.01;
    float scale = 0.01;
    float speed = 0.25;
    
    
    nUV.x += (iTime)*speed;
    nUV.y += (iTime)*speed;
    vec2 noise= texture( iChannel0, nUV*scale).xy;
    
    uv += (-1.0+noise*2.0) * intensity;
    
    return uv;
}

// Function 76
vec2 warp(vec2 pos)
{
	pos = pos * 2.0 - 1.0;
	pos *= vec2(
		1.0 + (pos.y * pos.y) * kWarp.x,
		1.0 + (pos.x * pos.x) * kWarp.y
	);
	return pos * 0.5 + 0.5;
}

// Function 77
vec2 barrel_distort( vec2 ec, float a ) { return ec / max( 0., 1. + a * ( 1. - dot( ec, ec ) ) ); }

// Function 78
vec2 vortex_pair_warp(vec2 uv, vec2 pos, vec2 vel, vec3 iResolution)
{
    vec2 aspect = vec2(1.,iResolution.y/iResolution.x);
    float ramp = 20.;

    float d = 0.075;

    vel *= aspect;
    float l = length(vel);
    vec2 p1 = pos;
    vec2 p2 = pos;

    if(l > 0.){
        vec2 normal = normalize(rot90(vel))/aspect;
        p1 = pos + normal * d / 2.;
        p2 = pos - normal * d / 2.;
    }

    float w = l*32.;

    // two overlapping rotations that would annihilate if they were not displaced.
    vec2 circle1 = vortex_warp(uv, p1, d, ramp, vec2(cos(w),sin(w)), iResolution);
    vec2 circle2 = vortex_warp(uv, p2, d, ramp, vec2(cos(-w),sin(-w)), iResolution);
    return (circle1 + circle2) / 2.;
}

// Function 79
void applyDistortion(inout vec2 uv, vec2 pos, float power){
    float noiseX = texture(iChannel1, pos.xy / 768. + vec2(iTime * .01)).x;
    float noiseY = texture(iChannel1, pos.xy / 4096. + vec2(iTime * .01)).x;
 	uv += vec2((noiseX - 0.5) * power, (noiseY - 0.5) * power);   
}

// Function 80
float warpFilter(vec2 uv, vec2 pos, float size, float ramp)
{
    return 0.5 + sigmoid( conetip(uv, pos, size, -16.) * ramp) * 0.5;
}

// Function 81
vec2 lens_distortion(vec2 r, float alpha) {
    return r * (1.0 - alpha * dot(r, r));
}

// Function 82
vec2 brownConradyDistortion(vec2 uv, float dist)
{
    uv = uv * 2.0 - 1.0;
    // positive values of K1 give barrel distortion, negative give pincushion
    float barrelDistortion1 = 0.1 * dist; // K1 in text books
    float barrelDistortion2 = -0.025 * dist; // K2 in text books

    float r2 = dot(uv,uv);
    uv *= 1.0 + barrelDistortion1 * r2 + barrelDistortion2 * r2 * r2;
    //uv *= 1.0 + barrelDistortion1 * r2;
    
    // tangential distortion (due to off center lens elements)
    // is not modeled in this function, but if it was, the terms would go here
    return uv * 0.5 + 0.5;
}

// Function 83
float warpFilter(vec2 uv, vec2 pos, float size, float ramp, vec3 iResolution)
{
    return 0.5 + sigmoid( conetip(uv, pos, size, -16., iResolution) * ramp) * 0.5;
}

// Function 84
vec2 brownConradyDistortion(in vec2 uv, in float k1, in float k2)
{
    uv = uv * 2.0 - 1.0;	// brown conrady takes [-1:1]

    // positive values of K1 give barrel distortion, negative give pincushion
    float r2 = uv.x*uv.x + uv.y*uv.y;
    uv *= 1.0 + k1 * r2 + k2 * r2 * r2;
    
    // tangential distortion (due to off center lens elements)
    // is not modeled in this function, but if it was, the terms would go here
    
    uv = (uv * .5 + .5);	// restore -> [0:1]
    return uv;
}

// Function 85
vec2 warp(vec2 uv, vec2 mo, float force, float radius) 
{
    vec2 mouv = mo-uv;
    
    // electro static formula
	return uv - force * exp(-dot( mouv, mouv)/abs(radius)) * mouv;
}

// Function 86
vec2 RadialDistortion(vec2 xy) {
    float radius = length(xy);
    return (distortion_fn(radius) * xy / radius) * FOV_REGION_SCALE / DISTORTION_STRENGTH;
}

// Function 87
void WarpSpace(inout vec3 eyevec, inout vec3 raypos)
{
    vec3 origin = vec3(0.0, 0.0, 0.0);

    float singularityDist = distance(raypos, origin);
    float warpFactor = 1.0 / (pow(singularityDist, 2.0) + 0.005);

    vec3 singularityVector = normalize(origin - raypos);
    
    float warpAmount = 5.0;

    eyevec = normalize(eyevec + singularityVector * warpFactor * warpAmount / float(ITERATIONS));
}

// Function 88
vec2 distort(vec2 d) {
    vec2 u = d;
    vec2 t;
    for(int i=0; i<500; i++) {
        float r2 = dot(u,u);
    	float r4 = r2*r2;
    	float r6 = r4*r2;
        t = u;
        u.x = d.x - P2*(r2 + 2.0*u.x*u.x) - 2.0*P1*u.x*u.y;
        u.y = d.y - P1*(r2 + 2.0*u.y*u.y) - 2.0*P2*u.x*u.y;
		u = u/(1.0 + K1*r2 + K2*r4 + K3*r6);
        
        vec2 v = u-t;
        if(dot(v,v) < 0.0001) {
            break;
        }
    }
    return u;
}

// Function 89
vec2 Distort(vec2 p)
{
    float theta  = atan(p.y, p.x);
    float radius = length(p);
    radius = pow(radius, 1.3);
    p.x = radius * cos(theta);
    p.y = radius * sin(theta);
    return 0.5 * (p + 1.0);
}

// Function 90
vec2 scandistort(vec2 uv) {
	float scan1 = clamp(cos(uv.y * 2.0 + iTime), 0.0, 1.0);
	float scan2 = clamp(cos(uv.y * 2.0 + iTime + 4.0) * 10.0, 0.0, 1.0) ;
	float amount = scan1 * scan2 * uv.x; 
	
	uv.x -= 0.05 * mix(texture(iChannel1, vec2(uv.x, amount)).r * amount, amount, 0.9);

	return uv;
	 
}

// Function 91
vec2 Warp(vec2 pos){
  pos=pos*2.0-1.0;    
  pos*=vec2(1.0+(pos.y*pos.y)*warp.x,1.0+(pos.x*pos.x)*warp.y);
  return pos*0.5+0.5;}

// Function 92
vec3 warp(vec2 u, float ph1, float ph2){

    // Initializing the warped UV coordinates. This gives it a bit 
    // of a worm hole quality. There are infinitly other mutations.
    vec2 v = u - log(1./max(length(u), .001))*vec2(-1, 1);
    
    // Scene color.
    vec3 col = vec3(0.);
    
    // Number of iterations.
    const int n = 5;
    
    for (int i = 0; i<n; i++){
    
        // Warp function.
        v = cos(v.y - vec2(0, 1.57))*exp(sin(v.x + ph1) + cos(v.y + ph2));
        v -= u;
        
        // Color via IQ's cosine palatte and shading.
        vec3 d = (.5 + .45*cos(vec3(i)/float(n)*3. + vec3(0, 1, 2)*1.5))/max(length(v), .001);
        // Accumulation.
        col += d*d/32.;
        
        // Adding noise for that fake path traced look. 
        // Also, to hide speckling in amongst noise. :)
        //col += fract(sin(u.xyy*.7 + u.yxx + dot(u + fract(iTime), 
        //             vec2(113.97, 27.13)))*45758.5453)*.01 - .005;
    }
    
    return col;
}

// Function 93
vec2 distortUV(vec2 uv) {
    float r = length (uv);
    uv = normalize(uv) * pow(r, mix(1.0,0.025, g_speed));
    
    r = length (uv);
    float rr = r*r;
    float k1 = mix(-0.2, 0.0, g_speed);
    float k2 = mix(-0.1, 0.0, g_speed);
    
    return uv * (1.0 + k1*rr + k2*rr*rr);
}

// Function 94
vec2 InverseRadialDistortionDerivative(vec2 xy) {
    vec2 scaledXY = xy * DISTORTION_STRENGTH / FOV_REGION_SCALE;
    float scaledRadius = length(scaledXY);
    return (inv_dist_derivative(scaledRadius) * DISTORTION_STRENGTH / FOV_REGION_SCALE) * 
        scaledXY / scaledRadius;
}

// Function 95
float distortedCapsule(vec3 p){
    float dtime = 1.8*p.z-time-1.; // mix time with space to create wave
    float dt = sin((dtime)-0.8*sin(dtime)); // distorted time, asymmetric sine wave
    p.x += 0.2*(p.z)*dt;
   	float d = sdVerticalCapsule(p-vec3(0.9,0,0.), 2.0,0.05*(4.0-1.5*p.z));
    float d2 = sdSphere(p-vec3(0.9,0,2.0),0.2);
    d = sdUnion_s(d,d2,0.1);
	return d;
}

