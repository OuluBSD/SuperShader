// Reusable Utilities UI/2D Functions
// Automatically extracted from UI/2D graphics-related shaders

// Function 1
vec4 hexCoords(vec2 uv)
{
    vec2 r = vec2(1., 1.73);
    vec2 h = r * 0.5;
    vec2 a = mod(uv, r) - h;
    vec2 b = mod(uv - r * 0.5, r) - h;
    vec2 gv = length(a) < length(b) ? a : b;
    
    vec2 id = gv - uv;
    
    float x = atan(gv.x, gv.y);
    float y = 0.5 - hexDist(gv);
    
    return vec4(x, y, id);
}

// Function 2
int coordToIndex(vec2 coord)
{
    ivec2 viewport = ivec2(iResolution + 0.5);
    
    return int(coord.y) * viewport.x + int(coord.x);
}

// Function 3
vec2 normalizeScreenCoords(vec2 fragCoord) {
    // Get coordinate in the range -1, 1.
	vec2 result = 2.0 * (fragCoord / iResolution.xy - 0.5);
    
    // Correct for the aspect ratio.
    result.x *= iResolution.x / iResolution.y;
	
    return result;
}

// Function 4
float getPolarCoord(vec2 q, float dir){
    
    // The actual animation. You perform that before polar conversion.
    q = r2(iTime*dir)*q;
    
    // Polar angle.
    const float aNum = 1.;
    float a = atan(q.y, q.x);
   
    // Wrapping the polar angle.
    return mod(a/3.14159, 2./aNum) - 1./aNum;
   
    
}

// Function 5
vec4 transform(vec4 X)
{
    float alpha1	= iTime*TWOPI/151.0;
    float alpha2	= iTime*TWOPI/145.0;
    float beta1		= iTime*TWOPI/131.0;
    float beta2		= iTime*TWOPI/137.0;
    float beta3		= iTime*TWOPI/143.0;
    
    vec4 Y = X;
    Y.xw *= ROTATION(beta1);
    Y.yw *= ROTATION(beta2);
    Y.zw *= ROTATION(beta3);
    Y.xy *= ROTATION(alpha1);
    Y.zx *= ROTATION(alpha2);
	return Y;
}

// Function 6
vec3 fragCoordToProjectionPlane(vec2 fragUv)
{
	vec2 uv = pixelToNormalizedspace(fragUv);
    return projectionCenter + projectionRight * uv.x + projectionUp * uv.y;
}

// Function 7
vec2 calcCoordsID(vec2 uv, int ID, float rotation)
{
    vec2 cellSize = vec2(PI / 16.0, PI / 20.0);
    
	if(ID == 0)
	{
		uv = vec2(length(uv), atan(uv.y/uv.x) * 0.2);
	}
	else if(ID == 2)
	{
		uv = vec2(log(length(uv) + 0.001) * 2.0, atan(uv.y, uv.x)) * 0.2;
	}
	else if(ID == 3)
	{
		uv = vec2(uv.x*uv.y, 0.5*(uv.y*uv.y - uv.x*uv.x)) * 2.5; // Parabolic coordinates? But reversed (parabolic to carthesian)
	}
	else if(ID == 4)
	{
		uv = exp(uv.x)*vec2(cos(uv.y), sin(uv.y));
	}
	else if(ID == 5)
	{
		float ff = length(uv) * 3.5;
		uv =  2.5 * uv * atan(ff) / (ff + 0.01);
	}
	else if(ID == 6)
	{
		uv = vec2(log(length(uv) + 0.001), atan(uv.y/uv.x));
        uv = rotate(uv, PI*0.25);
        cellSize *= SQRT2_OVER2 * vec2(1.0, 1.0) * 2.0;
        //uv.y += 1.0 * uv.x;
	}
	else if(ID == 7)
	{
		uv.x /= (1.0 + 2.0 * abs(uv.y));
	}
	    
    vec2 uvIntMod2 = mod(floor((uv) / cellSize), 2.0);
	uv = mod(uv, cellSize);
    if(abs(uvIntMod2.x) < 0.1 || abs(2.0-uvIntMod2.x) < 0.1) uv.x = cellSize.x - uv.x;
    if(abs(uvIntMod2.y) < 0.1 || abs(2.0-uvIntMod2.y) < 0.1) uv.y = cellSize.y - uv.y;
    
    uv -= cellSize*0.5;

	return uv;
}

// Function 8
vec3 TransformPosition(vec3 pos)
{
    pos.yz *= Rot((pos.z + 2.0)*sin(iTime*0.3)*0.2);
    pos.xy *= Rot(pos.z*sin(iTime*0.1)*0.25);
    pos.y -= 0.5 + sin(iTime*0.5)*0.2; 
    
    return pos;
}

// Function 9
mat4 zup_spherical_coords_to_matrix( vec2 theta, vec2 phi )
{
	vec3 z = zup_spherical_coords_to_vector( theta, phi );
	vec3 x = zup_spherical_coords_to_vector( perp( theta ), phi ); // note: perp(theta) = unit_vector2(theta+PI*0.5)
	vec3 y = cross( z, x );
	return ( mat4( vec4( x, 0.0 ), vec4( y, 0.0 ), vec4( z, 0.0 ), vec4( 0.0, 0.0, 0.0, 1.0 ) ) );
}

// Function 10
void transform(mat3 mtx) {
    mtx = mat2x3_invert(mtx);
    _stack.position.xy = (mtx * vec3(_stack.position.xy,1.0)).xy;
    _stack.position.zw = (mtx * vec3(_stack.position.zw,1.0)).xy;
    _stack.scale *= vec2(length(mtx[0].xy), length(mtx[1].xy));
}

// Function 11
vec2 vector_to_zup_spherical_coords( vec3 n )
{
	float theta = safe_acos( n.z ); // note: vectors normalized with normalize() are not immune to -1,1 overflow which cause nan in acos
	float phi = calc_angle( n.xy  );
	return vec2( theta, phi );
}

// Function 12
vec3 yup_spherical_coords_to_vector( float theta, float phi ) { return zup_spherical_coords_to_vector( theta, phi ).yzx; }

// Function 13
void transform(mat3 mtx) {
    mtx = mat2x3_invert(mtx);
    _stack.position.xy = (mtx * vec3(_stack.position.xy,1.0)).xy;
    _stack.position.zw = (mtx * vec3(_stack.position.zw,1.0)).xy;
    vec2 u = vec2(mtx[0].x, mtx[1].x);
    _stack.scale *= length(u);
}

// Function 14
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,theta);
    p.zx = rotate(p.zx,-phi);
  }
  if (!keypress(CHAR_R)) {
    p.yz = rotate(p.yz,0.123*iTime);
    //p.xy = rotate(p.xy,0.5*PI);
  }
  return p;
}

// Function 15
vec3 transformPoint(vec3 p, float t)
{
    mat3 m = rotX(cos(t) * 3.) * rotY(sin(t / 2.) * 3.) * rotZ(sin(t / 3.) * 3.);
    return m * p * (1. + cos(t * .7) * .2);
}

// Function 16
vec2 textureCoordinates(in vec3 position, in float ringRadius) {
  vec2 q = vec2(length(position.xz) - ringRadius, position.y);
  float u = (atan(position.x, position.z) + pi) / (2.0 * pi);
  float v = (atan(q.x, q.y) + pi) / (2.0 * pi);
  return vec2(u, v);
}

// Function 17
mat4 quat_to_transform(vec4 quat, vec3 translation) {
    float qx = quat.x;
    float qy = quat.y;
    float qz = quat.z;
    float qw = quat.w;
    float qx2 = qx * qx;
    float qy2 = qy * qy;
    float qz2 = qz * qz;
    
 	return mat4(
        1.0 - 2.0*qy2 - 2.0*qz2,	2.0*qx*qy - 2.0*qz*qw,	2.0*qx*qz + 2.0*qy*qw, 0.0,
    	2.0*qx*qy + 2.0*qz*qw,	1.0 - 2.0*qx2 - 2.0*qz2,	2.0*qy*qz - 2.0*qx*qw, 0.0,
    	2.0*qx*qz - 2.0*qy*qw,	2.0*qy*qz + 2.0*qx*qw,	1.0 - 2.0*qx2 - 2.0*qy2, 0.0,
        translation, 0.0
    );
}

// Function 18
float coordinateGrid(vec2 r) {
	vec3 axesCol = vec3(0.0, 0.0, 1.0);
	vec3 gridCol = vec3(0.5);
	float ret = 0.0;
	
	// Draw grid lines
	const float tickWidth = 0.1;
	for(float i=-2.0; i<2.0; i+=tickWidth) {
		// "i" is the line coordinate.
		ret += 1.-smoothstep(0.0, 0.005, abs(r.x-i));
		ret += 1.-smoothstep(0.0, 0.01, abs(r.y-i));
	}
	// Draw the axes
	ret += 1.-smoothstep(0.001, 0.005, abs(r.x));
	ret += 1.-smoothstep(0.001, 0.005, abs(r.y));
	return ret;
}

// Function 19
vec2 coord(vec2 xy)
{
    //normalize to -1,+1
    vec2 uv = 2.0 * xy.xy / iResolution.xy - 1.0;
    //fix ar such that y=[-1,1], x=[-1*ar,1*ar] (usually >1)
    uv.x = uv.x * iResolution.x / iResolution.y;
    return uv;
}

// Function 20
vec3 colorFromCoord(vec2 p){
    float t=hash12(p);
    return pal(t, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.10,0.20) );
    //return pal(t, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.33,0.67) );
}

// Function 21
vec3 UVToFisheyeCoord(float U, float V, float MinCos)
{
    float NdcX = U * 2.0 - 1.0;
    float NdcY = V * 2.0 - 1.0;
    NdcX *= iResolution.x / iResolution.y;
    
    float R = Sqrt(NdcX * NdcX + NdcY * NdcY);
    
    // x - sin(theta), z - cos(theta)
    vec3 Dir = vec3(NdcX / R, 0.0, NdcY / R);
    
    float Phi = clamp(R, 0.0, 1.0) * kPi * 0.75;
    
    Dir.y   = clamp(cos(Phi), MinCos, 1.0);
	Dir.xz *= Sqrt(1.0 - Dir.y * Dir.y);
    return Dir;
}

// Function 22
void getEyeCoords(out vec3 right, out vec3 fwd, out vec3 up, out vec3 eye, out vec3 dir, float aspect, vec2 spos)
{
    float speed=1.0;
    float elev = float(iMouse.y/iResolution.y)*0.5*1.0+0.7*sin(time*0.3*speed);
    float azim = float(iMouse.x/iResolution.x)*0.5*1.0+0.5*time*speed;
    right = vec3(sin(azim),cos(azim),0);
    fwd   = vec3(vec2(-1,1)*right.yx*cos(elev),sin(elev));
    up    = cross(right,fwd);
    eye = -(20.0+4.0*sin(1.0*time*speed)+4.0*sin(0.65264*time*speed))*fwd+vec3(0,0,10);
    dir = normalize(spos.x*right+spos.y*aspect*up+1.5*fwd);
}

// Function 23
vec2 transform(vec2 p, vec2 circ) {
    
    
    // The following is a standard polar repeat operation. It works
    // the same in hyperbolic space as it does in Euclidian space.
    // If you didn't do this, you'd reflect across just the one
    // edge. Set "ia" to ".5/float(N)" to see what I mean.
     
    float ia = (floor(atan(p.x, p.y)/TAU*float(N)) + .5)/float(N);
    // Start with a point on the boundary of the circle, then use 
    // polar repetition to put it on all the edge boundaries...
    // right in the middle of the edge, which makes sense.
    vec2 vert = rot2(ia*TAU)*vec2(0, circ.x);
   
    // The radius squared of the circle domain you're reflecting to. 
    float rSq = circ.y*circ.y;
    
    // Circle inversion, which relates back to an inverse Mobius
    // transformation. There are a lot of topics on just this alone, but 
    // the bottom line is, if you perform this operation on a point within
    // the Poincare disk, it will be reflected. It's similar to the
    // "p /= dot(p, p)" move that some may have used before.
    vec2 pc = p - vert;
    float lSq = dot(pc, pc);
    
    // If the distance (we're squaring for speed) from the current
    // point to any of the edge vertex points is within the limits, 
    // hyperbolically reflect it.
    if(lSq<rSq){
         
        p = pc*rSq/lSq + vert;
        
        // Attempting to add some extra randomness. Normally,
        // you wouldn't have this here.
        p = rot2(TAU/float(N)*(count + float(Q)))*p;
        
        
        // If we have a hit, increase the counter. This value can be useful
        // for coloring, and other things.
        count++; 
       
    }
     
    return p;
}

// Function 24
vec3 zup_spherical_coords_to_vector( vec2 theta_vec, vec2 phi_vec ) { return vec3( theta_vec.y * phi_vec, theta_vec.x ); }

// Function 25
vec2 pixel_coord(vec2 fg) {
    return ((fg / iResolution.xy)*2.0-1.0)*aspect;
}

// Function 26
vec3 yup_spherical_coords_to_vector( vec2 theta, vec2 phi ) { return zup_spherical_coords_to_vector( theta, phi ).yzx; }

// Function 27
vec2 voxToTexCoord(vec3 p) {
 	p = floor(p);
    return swizzleChunkCoord(p.xy) * packedChunkSize + vec2(mod(p.z, packedChunkSize.x), floor(p.z / packedChunkSize.x));
}

// Function 28
float GetBayerFromCoordLevel(vec2 pixelpos)
{ivec2 p=ivec2(pixelpos);int a=0
;for(int i=0; i<iterBayerMat; i++
){a+=bayer2x2(p>>(iterBayerMat-1-i)&1)<<(2*i);
}return float(a)/float(2<<(iterBayerMat*2-1));}

// Function 29
vec3 genLightCoords()
{
	// Use simple trig to rotate the light position around a point.
	vec3 lightCoords = vec3(lightPathCenter.x + (sin(iTime*timeScale)*lightPathRadius), 
				lightPathCenter.y + (cos(iTime*timeScale)*lightPathRadius),
				lightPathCenter.z);
	return lightCoords;
}

// Function 30
vec2 ProjectCoordsWave(vec2 normCoords)
{
	const float MAX_RADIUS = 1.0;
    float rad = sqrt(dot(normCoords, normCoords));
    if(rad > MAX_RADIUS)
        return normCoords;
    
    const float MIN_DEPTH = 0.4;
    const float WAVE_INV_FREQ = 20.0;
    const float WAVE_VEL = -10.0;
    float z = MIN_DEPTH + 
        (MAX_RADIUS - MIN_DEPTH) 
        * 0.5 * (1.0 + sin(WAVE_INV_FREQ * rad + iTime * WAVE_VEL));
//    if(z > 0.2)
//        return normCoords;
    return normCoords / z;
}

// Function 31
vec2 clip_coord(vec2 fragCoord)
{
	vec2 aspect = vec2(iResolution.x/iResolution.y, 1.0);
	return 2.0*aspect*fragCoord.xy/iResolution.xy - aspect;
}

// Function 32
mat4 quat_to_transform(vec4 quat, vec3 translation) {
    vec4 q = quat;
    vec4 q2 = quat * quat;
    
 	return mat4(
        1.0 - 2.0*(q2.y + q2.z), 2.0*(q.x*q.y - q.z*q.w), 2.0*(q.x*q.z + q.y*q.w), 0.0,
    	2.0*(q.x*q.y + q.z*q.w), 1.0 - 2.0*(q2.x + q2.z), 2.0*(q.y*q.z - q.x*q.w), 0.0,
    	2.0*(q.x*q.z - q.y*q.w), 2.0*(q.y*q.z + q.x*q.w),1.0 - 2.0*(q2.x + q2.y), 0.0,
        translation, 0.0
    );
}

// Function 33
vec3 barycentricCoordinate(vec2 P,Equerre T)
{
    vec2 PA = P - T.A;
    vec2 PB = P - T.B;
    vec2 PC = P - T.C;
    
    vec3 r = vec3(
        det22(PB,PC),
        det22(PC,PA),
        det22(PA,PB)
    );
    
    return r / (r.x + r.y + r.z);
}

// Function 34
vec3 transform_vector( mat4 m, vec3 v ) { return ( m * vec4( v, 0.0 ) ).xyz ; }

// Function 35
vec2 coord(in vec2 p) {
  p = p / iResolution.xy;
  // correct aspect ratio
  if (iResolution.x > iResolution.y) {
    p.x *= iResolution.x / iResolution.y;
    p.x += (iResolution.y - iResolution.x) / iResolution.y / 2.0;
  } else {
    p.y *= iResolution.y / iResolution.x;
    p.y += (iResolution.x - iResolution.y) / iResolution.x / 2.0;
  }
  // centering
  p -= 0.5;
  p *= vec2(-1.0, 1.0);
  return p;
}

// Function 36
vec2 fragCoordToXY(vec2 fragCoord) {
  vec2 relativePosition = fragCoord.xy / iResolution.xy;
  float aspectRatio = iResolution.x / iResolution.y;

  vec2 cartesianPosition = (relativePosition - vec2(0.38, 0.5)) * 3.0;
  cartesianPosition.x *= aspectRatio;

  return cartesianPosition;
}

// Function 37
vec2 transform_forward(vec2 P)
{
    return P;
}

// Function 38
vec3 getRayDirByCoord(vec2 coord){
	vec3 pointV3 = getPointV3ByFragCoord(coord);
    vec3 ray = pointV3 - cameraPosition;
    return normalize(ray);
}

// Function 39
vec4 get_viewport_transform(int frame, vec2 resolution, float downscale)
{
    vec2 ndc_scale = vec2(downscale);
    vec2 ndc_bias = vec2(0);//ndc_scale * taa_jitter(frame) / resolution.xy;
    ndc_scale *= 2.;
    ndc_bias  *= 2.;
    ndc_bias  -= 1.;
    ndc_scale.y *= resolution.y / resolution.x;
    ndc_bias.y  *= resolution.y / resolution.x;
    return vec4(ndc_scale, ndc_bias);
}

// Function 40
vec2 texcoords(vec3 p)
{   
	return p.xy * 0.25;
}

// Function 41
vec3 GetSceneCoordsTile(vec2 loc)
{
    return vec3(floor(loc), GetSceneTileIndex(loc.y));
}

// Function 42
vec4 swapCoords(vec2 seed, vec2 groupSize, vec2 subGrid, vec2 blockSize) {
    vec2 rand2 = vec2(rand(seed), rand(seed+.1));
    vec2 range = subGrid - (blockSize - 1.);
    vec2 coord = floor(rand2 * range) / subGrid;
    vec2 bottomLeft = coord * groupSize;
    vec2 realBlockSize = (groupSize / subGrid) * blockSize;
    vec2 topRight = bottomLeft + realBlockSize;
    topRight -= groupSize / 2.;
    bottomLeft -= groupSize / 2.;
    return vec4(bottomLeft, topRight);
}

// Function 43
vec2 UVToFragCoord(const in vec2 aUV)
{
	vec2 vFragCoord = aUV * 2.0 - 1.0;
	vFragCoord.x *= iResolution.x / iResolution.y;
	return vFragCoord;	
}

// Function 44
vec2 uvTransform(vec2 coord, vec2 resolution, vec2 tilling, vec2 offset, inout vec2 index)
{   
    vec2 result ;
    
    // screen ratio
    float ratio = resolution.x/resolution.y;
    	
    // This will normalize (0 to 1) (based on Y AXIS, x will be bigger on most screens)
    // x -> [0 , ratio]
    // y -> [0 , 1]
    result = coord/resolution.xy;
    
    // remaps coordinates to fit these intervals
    result.x = mapToRange(result.x, -ratio, ratio);
    result.y = mapToRange(result.y, -1., 1.);

    result.x = tileCoordinate( result.x, tilling.x, 2.*vec2(-1., 1.), offset.x, index.x);
    result.y = tileCoordinate( result.y, tilling.y, 2.*vec2(-1., 1.), offset.y, index.y);
  
    return result;
}

// Function 45
vec3 primTransform(vec3 p, prim_t b) {
	p-=b.pos;
    p.xy*=rotate(b.rxy);
    p.yz*=rotate(b.ryz);
    return p;
}

// Function 46
vec2 texCoordFromNormal(in vec3 n) {
	vec2 c = n.xy * abs(n.z) + n.xz * abs(n.y) + n.yz * abs(n.x);
    return c * 0.5 + 0.5;
//    c.x = mix(n.x, n.y, abs(n.y) - abs(n.x));
 //   c.y = mix(n.y, n.z, abs(n.z) - abs(n.y));
 //   return c * 0.5 + 0.5;
}

// Function 47
vec2 fragCoordForWorldPos(in vec3 worldPos, in vec2 resolution, in float time)
{
    vec3 pos = camera(time);
    
    float viewDist = distance(pos, CamTarget);
    
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 dir = (CamTarget - pos) / viewDist;
    vec3 right = normalize(cross(dir, up));
    up = normalize(cross(right, dir));

    float imgU = tan(camera_fov()) * viewDist;
    float imgV = imgU * resolution.y / resolution.x;

    float dWorld = dot(worldPos - pos, dir);
    float dProj = viewDist;
    vec3 projPos = (worldPos - pos) * (dProj / dWorld) + pos - CamTarget;
    
    vec2 uv = vec2(dot(projPos, right) / imgU,
                   dot(projPos, up)    / imgV) * 0.5 + 0.5;
    return uv * resolution.xy;
}

// Function 48
float coordMask(ivec2 coord, ivec4 frame)
{
    return ((coord.x >= frame.x) && (coord.x <= frame.z) && (coord.y >= frame.y) && (coord.y <= frame.w)) ? 1.0 : 0.0;
}

// Function 49
vec2 hitCoordsNormalized(int index, vec2 xy) {
    vec4 bbox = controls[index].bboxUV;
    vec2 midpoint = vec2(bbox[0] + bbox[2], bbox[1] + bbox[3]) * 0.5;
    vec2 span = abs(vec2(bbox[0] - bbox[2], bbox[1] - bbox[3]));
    vec2 uv = xyToUV(xy);
 
    return (uv - midpoint) / (0.5 * span);
}

// Function 50
mat4 zup_spherical_coords_to_matrix( float theta, float phi ) { return zup_spherical_coords_to_matrix( unit_vector2( theta ), unit_vector2( phi ) ); }

// Function 51
mat4 transform(vec3 translation, vec3 rotation, vec3 scale) {
	float cx = cos(radians(rotation.x));
	float sx = sin(radians(rotation.x));
	float cy = cos(radians(rotation.y));
	float sy = sin(radians(rotation.y));
	float cz = cos(radians(rotation.z));
	float sz = sin(radians(rotation.z));

	return mat4(
			(cy*cz+sy*sx*sz) * scale.x,
			(cx*sz) * scale.x,
			(cy*sx*sz-cz*sy) * scale.x,
			0,
			(cz*sy*sx-cy*sz) * scale.y,
			(cx*cz) * scale.y,
			(cy*cz*sx+sy*sz) * scale.y,
			0,
			(cx*sy) * scale.z,
			(-sx) * scale.z,
			(cy*cx) * scale.z,
			0,
			translation.x,
			translation.y,
			translation.z,
			1
	);
}

// Function 52
vec3 rotationCoord(vec3 n, in float t, float paramRotate)
{
 vec3 result;
 //--------------------------------------------
   vec2 sc = vec2(sin(t), cos(t));
   mat3 rotate;
   if(paramRotate <= 0.1)
   {

      rotate = mat3(  1.0,  0.0,  0.0,
                      0.0,  1.0,  0.0,
                      0.0,  0.0,  1.0);   
   }
   else if(paramRotate <= 1.0)
   {
      rotate = mat3(  1.0,  0.0,  0.0,
                      0.0, sc.y,-sc.x,
                      0.0, sc.x, sc.y);
   }
   else if(paramRotate <= 2.0)
   {
       rotate = mat3(  1.0,  0.0,  0.0,
                       0.0, sc.y,sc.x,
                       0.0, -sc.x, sc.y);  
   }
   else if (paramRotate <= 3.0)
   {
      rotate = mat3( sc.y,  0.0, -sc.x,
                     0.0,   1.0,  0.0,
                     sc.x,  0.0, sc.y);   
   }
   else if (paramRotate <= 4.0)
   {
      rotate = mat3( sc.y,  0.0, sc.x,
                     0.0,   1.0,  0.0,
                    -sc.x,  0.0, sc.y);   
   }   
   else if (paramRotate <= 5.0)
   {
       rotate = mat3( sc.y,sc.x,  0.0,
                     -sc.x, sc.y, 0.0,
                      0.0,  0.0,  1.0);  
   }   
   else if (paramRotate <= 6.0)
   {
       rotate = mat3( sc.y,-sc.x, 0.0,
                      sc.x, sc.y, 0.0,
                      0.0,  0.0,  1.0);  
   }     
   else
   {
   mat3 rotate_x = mat3(  1.0,  0.0,  0.0,
                          0.0, sc.y,-sc.x,
                          0.0, sc.x, sc.y);
   mat3 rotate_y = mat3( sc.y,  0.0, -sc.x,
                         0.0,   1.0,  0.0,
                         sc.x,  0.0,  sc.y);
   mat3 rotate_z = mat3( sc.y, sc.x,  0.0,
                        -sc.x, sc.y,  0.0,
                         0.0,  0.0,   1.0);
   rotate = rotate_z * rotate_y * rotate_z;                
   }
  result = n * rotate;
  return result;
}

// Function 53
mat3 createCoordinateSystem(in vec3 n)
{
    vec3 b, t;
    if (abs(n.x) > abs(n.y)) {
        b = normalize(vec3(n.z, 0.0, -n.x));
    } else {
        b = normalize(vec3(0.0, n.z, -n.y));
    }
    t = cross(n, b);
    return mat3(t,n,b);
}

// Function 54
vec3 UVToEquirectCoord(float U, float V, float MinCos)
{
    float Phi = kPi - V * kPi;
    float Theta = U * 2.0 * kPi;
    vec3 Dir = vec3(cos(Theta), 0.0, sin(Theta));
	Dir.y   = clamp(cos(Phi), MinCos, 1.0);
	Dir.xz *= Sqrt(1.0 - Dir.y * Dir.y);
    return Dir;
}

// Function 55
vec4 Transform(int i,vec4 p
){p.xyz=objPos[i]-p.xyz
 ;p.xyz=qr(objRot[i],p.xyz)
 ;p.xyz/=objSca[i]
 ;p.w*=dot(vec3(1),abs(objSca[i]))/3.//distance field dilation approx
 ;return p;}

// Function 56
vec3 ntransform( in mat4 mat, in vec3 v ) { return (mat*vec4(v,0.0)).xyz; }

// Function 57
vec3 seaUntransform( in vec3 x ) {
    x.yz = rotate( -0.8, x.yz );
    return x;
}

// Function 58
vec3 worldToPawnCoords(vec3 pos, float time)
{
    float id_x = floor(pos.x / SquareSize + 0.5);
    time += noise(id_x - 1.0) * 3.0;
    pos.z -= time * SquareSize;
    float id_z = floor(pos.z / (SquareSize * 4.0) + 0.5);
    //Store this for later, remove saw_wave3(time_offset) so the pieces stay within squares
    float time_offset = noise(145.0 * id_z + 23.0) * 5.0;
    time += time_offset;
    
    vec3 squareCoord = pos;
    squareCoord.x = (fract(pos.x / SquareSize + 0.5) - 0.5) * SquareSize;
    squareCoord.z = (fract(pos.z / (SquareSize * 4.0) + 0.5) - 0.5) * SquareSize * 4.0;
    squareCoord.z += (saw_wave3(time) - saw_wave3(time_offset) - 1.0) * SquareSize;
    
    return squareCoord;
}

// Function 59
vec3 yup_spherical_coords_to_vector( float theta, float phi )
{
	return zup_spherical_coords_to_vector( theta, phi ).yzx;
}

// Function 60
vec2 toCanonicalCoordinates(vec2 pixel) {
    return (pixel -.5 * iResolution.xy) / iResolution.y;
}

// Function 61
float coordinateGrid(vec2 r) {
	vec3 axesCol = vec3(0.0, 0.0, 1.0);
	vec3 gridCol = vec3(0.5);
	float ret = 0.0;
	
	// Draw grid lines
	const float tickWidth = 0.1;
	for(float i=-2.0; i<2.0; i+=tickWidth) {
		// "i" is the line coordinate.
		ret += 1.-smoothstep(0.0, 0.008, abs(r.x-i));
		ret += 1.-smoothstep(0.0, 0.008, abs(r.y-i));
	}
	// Draw the axes
	ret += 1.-smoothstep(0.001, 0.015, abs(r.x));
	ret += 1.-smoothstep(0.001, 0.015, abs(r.y));
	return ret;
}

// Function 62
vec2 unpackfragcoord2 (float p, vec2 s) {
    float x = mod(p, s.x);
    float y = (p - x) / s.x + 0.5;
    return vec2(x,y);
}

// Function 63
vec2 GetProjectedCoord( vec3 vPos, vec3 vNormal )
{
    int iAxis = GetProjectionAxis ( vNormal );
    return GetProjectedCoord( vPos, iAxis );
}

// Function 64
void transformray (vec3 ro, vec3 rd, mat2 rotationY, vec3 offset, out vec3 outro, out vec3 outrd)
{
	outro = ro + offset;
	outro = vec3(rotationY * outro.xz, outro.y).xzy;
	outrd = vec3(rotationY * rd.xz, rd.y).xzy;
}

// Function 65
vec2 fragCoordForDir(in vec3 worldDir, in vec2 resolution, in float time)
{
    vec3 pos = camera(time);
    float viewDist = distance(pos, CamTarget);
    
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 dir = normalize(CamTarget - pos);
    vec3 right = normalize(cross(dir, up));
    up = normalize(cross(right, dir));
    
    float imgU = tan(camera_fov()) * viewDist;
    float imgV = imgU * resolution.y / resolution.x;

    float dProj = distance(pos, CamTarget);
    
    //vec3 worldPosA = CamTarget;
    vec3 worldPosB = CamTarget + worldDir;
    
    
    //float dWorldA = dot(worldPosA - pos, dir);
    float dWorldB = dot(worldPosB - pos, dir);
    
    
    //vec3 projPosA = (worldPosA - pos) * (dProj / dWorldA) + pos - CamTarget;
    vec3 projPosB = (worldPosB - pos) * (dProj / dWorldB) + pos - CamTarget;
    
    //vec2 uvA = vec2(dot(projPosA, right) / imgU,
    //                dot(projPosA, up)    / imgV) * 0.5 + 0.5;
                    
    vec2 uvB = vec2(dot(projPosB, right) / imgU,
                    dot(projPosB, up)    / imgV) * 0.5 + 0.5;
                    
    
    // since CamTarget always stay at [0.5, 0.5]
    return normalize((uvB * resolution.xy) - (0.5 * resolution.xy));
}

// Function 66
vec2 unpackCoord(float f) 
{
    vec2 coord;
    coord.x = floor(mod(f, 512.0));
    coord.y = floor(f / 512.0);
    return coord * INV_SCALE;
}

// Function 67
vec3 unpackfragcoord3 (float p, vec3 s) {
    float x = mod(p, s.x);
    float y = mod((p - x) / s.x, s.y);
    float z = (p - x - floor(y) * s.x) / (s.x * s.y);
    return vec3(x,y+0.5,z+0.5);
}

// Function 68
vec2 coordToUv(vec2 coord) {
  return (coord - (iResolution.xy * 0.5)) / iResolution.y;
}

// Function 69
vec3 yup_spherical_coords_to_vector( float theta, float phi ) { return yup_spherical_coords_to_vector( unit_vector2( theta ), unit_vector2( phi ) ); }

// Function 70
vec3 calcCubeCoordsInGnomonicProjection(in vec2 screenCoord, in vec2 centralPoint, in vec2 FoVScale) {
	return sphericalToCubemap( calcSphericalCoordsInGnomonicProjection(screenCoord, centralPoint, FoVScale) );
}

// Function 71
hexCoordsOuts HexCoords(vec2 uv) {
	
    vec2 r = vec2(1, sqrt3); // 1, sqrt3
    vec2 h = r*.5;
    
    vec2 a = mod(uv, r)-h; // tile UV coords and move 0,0 to center of tile
    vec2 b = mod(uv-h, r)-h; // move uv coords, tile and move 0,0 to centre of tile
    
    float isB = dot(a, a) < dot(b,b) ? 0. : 1.;
    vec2 gv = (1.-isB)*a + isB*b; // select which tile are in based on distance from the origin of the tile
    
    
    // fix UVs back to square
    gv.y = gv.y*h.y;
    gv += 0.5;
    
    //ids
    float x_id = floor(uv.x+isB*0.5/1.);
    float y_id = isB * 2.*floor((uv.y+ half_sqrt3)/sqrt3) + (1.-isB)* (2.*floor((uv.y)/sqrt3)+1.);
        
    return hexCoordsOuts(gv, vec2(x_id,y_id));
}

// Function 72
float convertCoords(vec2 coord, vec3 xyr) {
    float r = xyr.z / 2.;
    vec2 absolute = vec2(xyr.x - width / 2., height / 2. - xyr.y) + vec2(r, -r);
    return sdCircle(coord - absolute * ratio / height * 2., r * ratio / height * 2.);
}

// Function 73
vec4 coordinate2(vec2 uv,vec2 uvStep){
	vec4 color = backgroundColor;
    
    //uv.y += sin(uv.x);
    
    for(float i = -2.;i < +2.;i += .1){
    	color = mix(gridColor,color,step(.003,abs(uv.x - i)));
        color = mix(gridColor,color,step(.003,abs(uv.y - i)));
    }
    
    color = mix(axisColor,color,step(.003,abs(uv.x)));
    color = mix(axisColor,color,step(.003,abs(uv.y)));
    
    return color;
}

// Function 74
mat4 Transform(vec3 scale, vec3 rot, vec3 trans) 
{
    return mat4(Scaling(scale) * Rotation(rot)) * Translation(trans);
}

// Function 75
ivec2 indexToCoords( int i, float xResolution )
{
    int ixResolution = int(xResolution);
    int x = i % ixResolution;
    int y = i / ixResolution;
    
    return ivec2(x, y);
}

// Function 76
vec3 GetGliderTransform(vec3 p, Ray camera)
{
    vec3 steadyPos = p;
    vec3 introPos = p;
    
    introPos -= (camera.origin + camera.forward * .85);
    introPos -= vec3(.5, 0.0, .2) * smoothstep(20.0, .0, iTime);
    introPos.y += cos(iTime * .2) * .2 + .3 - smoothstep(1.5, 4.5, iTime) * .4;
    introPos.y += smoothstep(10.0, 30.0, iTime) * 1.5;
    
    pR(introPos.xy, sin(iTime*.3));
    
    float t = iTime * .3;
    vec2 offset = vec2(cos(t), sin(t));
    
    float circleTime = cos(t * 1.3) * 0.5 + 0.5;
    steadyPos.xz += offset * mix(2.35, 3.5, sin(t*.75) * 0.5 + 0.5);
    pR(steadyPos.xz, t);    
    steadyPos.y -= .1 + mix(-.9, 1.6, circleTime);
    pR(steadyPos.yx, cos(t * .2));
    
    p = introPos;
    p = mix(introPos, steadyPos, smoothstep(15.0, 35.0, iTime));
    
    return p * mix(2.0, 1.75, smoothstep(15.0, 35.0, iTime));
}

// Function 77
vec2 sdBoxTexcoord(vec3 p, vec3 b)
{
    vec3 d = p - b;
    vec2 uv = vec2(0.5, -0.5) * d.xy / b.xy;
    return uv;
}

// Function 78
vec2 calcSphericalCoordsInGnomonicProjection(in vec2 screenCoord, in vec2 centralPoint, in vec2 FoVScale) {
	return calcSphericalCoordsFromProjections(screenCoord, centralPoint, FoVScale, false); 
}

// Function 79
vec2 GetProjectedCoord( vec3 vPos, int iAxis )
{
    vec2 vTexCoord;
    switch( iAxis )
    {
        default:
        case 0:
        	vTexCoord = vPos.yz;
			break;        

        case 1:
        	vTexCoord = vPos.xz;
			break;        

        case 2:
        	vTexCoord = vPos.xy;
			break;        
    }
    
    return vTexCoord;
}

// Function 80
vec3 WorldToVolumeCoord(vec3 p, vec3 min_box, vec3 max_box)
{
    return (p - min_box) / (max_box - min_box);
}

// Function 81
vec3 UnprojectCoord( vec2 vTexCoord, vec4 vPlane, int iAxis )
{
    vec3 vPos = vec3(0.0);
    
    switch( iAxis )
    {
        default:
        case 0:
        	vPos.yz = vTexCoord;
        	vPos.x = -( vPos.y * vPlane.y + vPos.z * vPlane.z + vPlane.w ) / vPlane.x;
			break;        

        case 1:
			vPos.xz = vTexCoord;
    		vPos.y = -( vPos.x * vPlane.x + vPos.z * vPlane.z + vPlane.w ) / vPlane.y;
			break;        

        case 2:
			vPos.xy = vTexCoord;            
            vPos.z = -(vPos.x * vPlane.x + vPos.y * vPlane.y + vPlane.w) / vPlane.z;
			break;        
    }
    
    return vPos;    
}

// Function 82
vec3 TextureCoordsToPoint(in ivec2 coords){
    int size = 32;
    const int squareSide = 4;
    vec3 p;
    p.xz = mod(vec2(coords.xy),float(size));
    
    ivec2 index = coords.xy/size;
    p.y = float(index.y*squareSide+index.x);
    return p;
}

// Function 83
bool isBlockCoord(vec3 coord) {
    return (int(coord.z)== 0);
}

// Function 84
vec3 InvTransformHeadDir( vec3 vPos )
{
    return g_sceneState.mHeadRot * vPos;
}

// Function 85
vec2 swizzleChunkCoord(vec2 chunkCoord) {
    vec2 c = chunkCoord;
    float dist = max(abs(c.x), abs(c.y));
    vec2 c2 = floor(abs(c - 0.5));
    float o = max(c2.x, c2.y);
    float neg = step(c.x + c.y, 0.) * -2. + 1.;
    return (neg * c) + o;
}

// Function 86
float GetSceneSnowTileTreeCoord(float coords, float treeIdx)
{
    return floor(-kSceneWidth + 2.0 * kSceneWidth * Hash(vec2(coords, treeIdx)).y);
}

// Function 87
float square_anim_transform(vec2 p, float initial_scale, float angle,
                            float scale, vec2 offset, float t) {
    angle *= t;
    scale = mix(initial_scale, scale, t);
    offset *= t;
    p = iR(angle) * iS(scale) * (p - offset);
    
    return hollow_square(p, .5, .03 / scale);
}

// Function 88
int uvToLinearCoord(vec2 uv) 
{
	return int(((uv.y-0.5) * iResolution.x) + uv.x-0.5);
}

// Function 89
mat3 UTIL_axisRotationMatrix( vec3 u, float ct, float st )
{
    return mat3(  ct+u.x*u.x*(1.-ct),     u.x*u.y*(1.-ct)-u.z*st, u.x*u.z*(1.-ct)+u.y*st,
	              u.y*u.x*(1.-ct)+u.z*st, ct+u.y*u.y*(1.-ct),     u.y*u.z*(1.-ct)-u.x*st,
	              u.z*u.x*(1.-ct)-u.y*st, u.z*u.y*(1.-ct)+u.x*st, ct+u.z*u.z*(1.-ct) );
}

// Function 90
vec3 rotate_cube_coord(vec3 cube_coord) {
    return vec3(
        cube_coord.y-cube_coord.z,
        cube_coord.z-cube_coord.x,
        cube_coord.x-cube_coord.y
    );
}

// Function 91
vec2 ScaleCoord(vec2 coord, vec2 scale, vec2 pivot)
{
    return (coord - pivot) * scale + pivot;
}

// Function 92
mat4 yup_spherical_coords_to_matrix( float theta, float phi ) {  return yup_spherical_coords_to_matrix( unit_vector2( theta ), unit_vector2( phi ) ); }

// Function 93
vec2 getUVCoords(in vec2 fragCoords)
{
    return ((fragCoords - 0.5) / (iResolution.xy - 1.0) - 0.5) * vec2(getAspectRatio(), 1.0);
}

// Function 94
vec2 PolarCoords(vec2 uv) {
	// carthesian coords in polar coords out
    return vec2(atan(uv.x, uv.y), length(uv));
}

// Function 95
vec3 transformCube(vec3 p) {
	p.x -= ROTATION_DIST;

	p = rotateY(p, getAngle(tween(iTime)));
	p.x += ROTATION_DIST;
	return p;
}

// Function 96
complex transform(complex z) {
	return mul(mul(z, thetransform[0]) + thetransform[1],
	    invert(mul(z, thetransform[2]) + thetransform[3]));
}

// Function 97
mat3 GetChromaticAdaptionTransform( mat3 M, vec3 XYZ_w, vec3 XYZ_wr )
{
    //return inverse(CA_A_to_D65_VonKries);    
    //return inverse(CA_A_to_D65_Bradford);
        
    //return mat3(1,0,0, 0,1,0, 0,0,1); // do nothing
    
	//mat3 M = mCAT_02;
    //mat3 M = mCAT_Bradford;
    //mat3 M = mCAT_VonKries;
    //mat3 M = mat3(1,0,0,0,1,0,0,0,1);
    
    vec3 w = XYZ_w * M;
    vec3 wr = XYZ_wr * M;
    vec3 s = w / wr;
    
    mat3 d = mat3( 
        s.x,	0,		0,  
        0,		s.y,	0,
        0,		0,		s.z );
        
    mat3 cat = M * d * inverse(M);
    return cat;
}

// Function 98
void normalized_coords(vec2 frag_coord,
  out vec2 p, out vec2 cursor)
{
  float m = min(iResolution.x, iResolution.y);
  p = (frag_coord.xy - 0.5 * iResolution.xy)/m;
  cursor = (iMouse.xy - 0.5 * iResolution.xy)/m;
}

// Function 99
vec4 Transform(Object infos, vec4 pos)
{
    pos.xyz = infos.pos-pos.xyz;
    pos.xyz = Rotate(infos.rot,pos.xyz);
    pos.xyz /= infos.scale;
    
    pos.w *= dot(vec3(1),abs(infos.scale))/3.; //distance field dilation approx
    
    return pos;
}

// Function 100
vec3 fan_transform_xz(in vec3 pos, in float center, in float range) {
    center *= 0.017453292519943295;
    range *= 0.017453292519943295;
    float start = (center - range/2.),
          ang = atan(pos.x, pos.z),
          len = length(pos.xz);
    ang = mod(ang-start, range) - range/2. + center;
    pos.xz = len * vec2(sin(ang), cos(ang));
    return pos;
}

// Function 101
vec2 transform_inverse(vec2 P)
{
    return P;
}

// Function 102
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,-theta);
    p.zx = rotate(p.zx,phi);
  }
  //p.yz = rotate(p.yz,iTime * 0.125);
  //p.zx = rotate(p.zx,iTime * 0.2);
  return p;
}

// Function 103
vec2 Cam_GetUVFromWindowCoord( const in vec2 vWindow )
{
    vec2 vScaledWindow = vWindow;
    vScaledWindow.x *= iResolution.y / iResolution.x;

    return vScaledWindow * 0.5 + 0.5;
}

// Function 104
vec3 seaTransform( in vec3 x ) {
    x.yz = rotate( 0.8, x.yz );
    return x;
}

// Function 105
vec3 repeat_transform(in vec3 pos, in vec3 repeat) {
    if (repeat.x > 0.) pos.x = mod(pos.x + repeat.x/2., repeat.x) - repeat.x/2.;
    if (repeat.y > 0.) pos.y = mod(pos.y + repeat.y/2., repeat.y) - repeat.y/2.;
    if (repeat.z > 0.) pos.z = mod(pos.z + repeat.z/2., repeat.z) - repeat.z/2.;
    return pos;
}

// Function 106
vec2 fragCoord2UV(vec2 fc)
{
    vec2 fitNum = floor(iResolution.xy / TIT_RES_F);
    float scale = max(min(fitNum.x, fitNum.y), 1.0);
    vec2 fc2 = remap(vec2(0.0), TIT_RES_F * vec2(float(scale)), vec2(0.0), TIT_RES_F, fc);
    fc2 -= floor((remap(vec2(0.0), TIT_RES_F * vec2(float(scale)), vec2(0.0), TIT_RES_F, iResolution.xy) - TIT_RES_F) / vec2(2.0));
    return fc2 / iResolution.xy;
}

// Function 107
mat4 inverseTransform(vec3 translate, vec3 rotate, vec3 scale) {
    mat4 s;
    // matrices indexed to columns!
    s[0] = vec4(1./scale.x, 0., 0., 0.);
    s[1] = vec4(0., 1./scale.y, 0., 0.);
    s[2] = vec4(0., 0., 1./scale.z, 0.);
	s[3] = vec4(0., 0., 0., 1.);                                     
                                    
    rotate.x = radians(rotate.x);
    rotate.y = radians(rotate.y);
    rotate.z = radians(rotate.z);
      
    mat4 r_x;
    r_x[0] = vec4(1., 0., 0., 0.);
    r_x[1] = vec4(0., cos(rotate.x), -sin(rotate.x), 0.);
    r_x[2] = vec4(0., sin(rotate.x), cos(rotate.x), 0.);
    r_x[3] = vec4(0., 0., 0., 1.);
                                                                   
    mat4 r_y;
    r_y[0] = vec4(cos(rotate.y), 0., sin(rotate.y), 0.);
    r_y[1] = vec4(0., 1, 0., 0.);
    r_y[2] = vec4(-sin(rotate.y), 0., cos(rotate.y), 0.);
    r_y[3] = vec4(0., 0., 0., 1.);

    mat4 r_z;
    r_z[0] = vec4(cos(rotate.z), -sin(rotate.z), 0., 0.);
    r_z[1] = vec4(sin(rotate.z), cos(rotate.z), 0., 0.);
    r_z[2] = vec4(0., 0., 1., 0.);
    r_z[3] = vec4(0., 0., 0., 1.);
    
    mat4 t;
    t[0] = vec4(1., 0., 0., 0.);
    t[1] = vec4(0., 1., 0., 0.);
    t[2] = vec4(0., 0., 1., 0.);
    t[3] = vec4(-translate.x, -translate.y, -translate.z, 1.); 
    
    return s * r_z * r_y * r_x * t;
}

// Function 108
vec3 terrainTransformRo( const in vec3 ro ) {
    vec3 rom = terrainTransform(ro);
    rom.y -= EARTH_RADIUS - 100.;
    rom.xz *= 5.;
    rom.xz += vec2(-170.,50.)+vec2(-4.,.4)*time;    
    rom.y += (terrainLow( rom.xz ) - 86.)*clamp( 1.-1.*(length(ro)-EARTH_RADIUS), 0., 1.);
    return rom;
}

// Function 109
void asteroidUnTransForm(inout vec3 ro, const in vec3 id ) {
    float yzangle = (id.y-.5)*time*2.;
    ro.yz = rotate( -yzangle, ro.yz );

    float xyangle = (id.x-.5)*time*2.;
    ro.xy = rotate( -xyangle, ro.xy );  
}

// Function 110
vec3 texCoords( in vec3 p )
{
	return 64.0*p;
}

// Function 111
vec3 transform(vec3 inp, mat4 offset) {
    return (offset * vec4(inp, 1.0)).xyz;
}

// Function 112
vec2 texCoords( in vec3 pos, int mid )
{
    vec2 matuv;
    
    if( mid==0 )
    {
        matuv = pos.xz;
    }
    else if( mid==1 )
    {
        vec3 q = normalize( pos - sc0.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc0.w;
    }
    else if( mid==2 )
    {
        vec3 q = normalize( pos - sc1.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc1.w;
    }
    else if( mid==3 )
    {
        vec3 q = normalize( pos - sc2.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc2.w;
    }
    else if( mid==4 )
    {
        vec3 q = normalize( pos - sc3.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc3.w;
    }

	return 200.0*matuv;
}

// Function 113
mat4 zup_spherical_coords_to_matrix_rollx( vec2 theta, vec2 phi, vec2 rollx )
{
	vec3 z = zup_spherical_coords_to_vector( theta, phi );
	vec3 x = zup_spherical_coords_to_vector( perp( theta ), phi ); // note: perp(theta) = unit_vector2(theta+PI*0.5)
	vec3 y = cross( z, x );
	vec3 ry = y * rollx.x + z * rollx.y;
	vec3 rz = -y * rollx.y + z * rollx.x;
	y = ry;
	z = rz;
	return ( mat4( vec4( x, 0.0 ), vec4( y, 0.0 ), vec4( z, 0.0 ), vec4( 0.0, 0.0, 0.0, 1.0 ) ) );
}

// Function 114
bool getGridCoord(in vec2 fragCoord, out int x, out int y, out int z) {
 	x = int(fragCoord.x);
    z = int(fragCoord.y) - BLD_GRID_H;
    y = x / GRID_W;
    x = x % GRID_W;
    return x >= 0 && x < GRID_W && y >= 0 && y < GRID_H && z >= 0 && z < GRID_L;
}

// Function 115
vec2 FragCoordToUV(const in vec2 aFragCoord)
{
    vec2 vScaledFragCoord = aFragCoord;
    vScaledFragCoord.x *= iResolution.y / iResolution.x;
    return (vScaledFragCoord * 0.5 + 0.5);
}

// Function 116
vec2 GetWindowCoord( vec2 uv )
{
	vec2 window = uv * 2.0 - 1.0;
	window.x *= iResolution.x / iResolution.y;

	return window;	
}

// Function 117
vec2 screenToPolygonTransform(vec2 p, float t)
{
    float a = t * 4.;
    mat2 m = mat2(cos(a), sin(a), -sin(a), cos(a));
    return m * (p - vec2(cos(t * 14.)*.5, 0.));
}

// Function 118
vec2 glitchCoord(vec2 p, vec2 gridSize) {
	vec2 coord = floor(p / gridSize) * gridSize;;
    coord += (gridSize / 2.);
    return coord;
}

// Function 119
vec3 Custom_InvOutputTransform(vec3 aces, float Y_MIN, float Y_MID, float Y_MAX, int OT_Display, int OT_Limit, int EOTF, int SURROUND, bool STRETCH_BLACK, bool D60_SIM, bool LEGAL_RANGE)
{
Chromaticities OT_DISPLAY_PRI, OT_LIMIT_PRI;
OT_DISPLAY_PRI = REC2020_PRI;
OT_LIMIT_PRI = REC2020_PRI;
if(OT_Display == 1)
OT_DISPLAY_PRI = P3D60_PRI;
if(OT_Display == 2)
OT_DISPLAY_PRI = P3D65_PRI;
if(OT_Display == 3)
OT_DISPLAY_PRI = P3DCI_PRI;
if(OT_Display == 1)
OT_DISPLAY_PRI = REC709_PRI;

if(OT_Limit == 1)
OT_LIMIT_PRI = P3D60_PRI;
if(OT_Limit == 2)
OT_LIMIT_PRI = P3D65_PRI;
if(OT_Limit == 3)
OT_LIMIT_PRI = P3DCI_PRI;
if(OT_Limit == 1)
OT_LIMIT_PRI = REC709_PRI;

return invOutputTransform(aces, Y_MIN, Y_MID, Y_MAX, OT_DISPLAY_PRI, OT_LIMIT_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE);
}

// Function 120
vec2 transform(vec2 p, mat3 m){ return (vec3(p,1)*m).xy;}

// Function 121
vec4 coordinate1(vec2 uv,vec2 uvStep){//More effective
	vec4 color = backgroundColor;
    
    //uv.y += sin(uv.x + iTime);
    
    color = mix(gridColor,color,smoothstep(0.,uvStep.x * 2.,abs(mod(uv.x,.05))));
    color = mix(gridColor,color,smoothstep(0.,uvStep.y * 2.,abs(mod(uv.y,.05))));
    
    color = mix(axisColor,color,abs(smoothstep(0.,uvStep.x,abs(uv.x))));
    color = mix(axisColor,color,abs(smoothstep(0.,uvStep.y,abs(uv.y))));
    
    //vec2 dotdot = step(.006,mod(uv,0.05));
    //float dotdotdot = dotdot.x * dotdot.y;
    //color = mix(red,color,dotdotdot);
    
    return color;
}

// Function 122
vec2 getLevelCoords(vec2 coord, int level, inout vec2 frameCoord)
{

    ivec2 dir = ivec2(1,0);
    ivec2 s = ivec2(Res1)/(dir+1);
    ivec2 sp = s;
    ivec2 o = ivec2(0);
    ivec2 op = o;
    for(int i=0;i<level;i++) {
        op=o; o+=s*dir;
        dir=(dir+1)&1;
        sp=s; s/=dir+1;
    }

    vec2 c = coord*vec2(s)+vec2(o);
    frameCoord=fract(c);
    return (floor(c)+.5)/Res1;
}

// Function 123
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,-theta);
    p.zx = rotate(p.zx,phi);
  }
  if (dorotate) {
    float t = iTime;
    p.zx = rotate(p.zx,t * 0.2);
  }
  return p;
}

// Function 124
vec2 Cam_WorldToWindowCoord(const in vec3 vWorldPos, const in CameraState cameraState )
{
    vec3 vOffset = vWorldPos - cameraState.vPos;
    vec3 vCameraLocal;

    vCameraLocal = vOffset * Cam_GetWorldToCameraRotMatrix( cameraState );
	
    vec2 vWindowPos = vCameraLocal.xy / (vCameraLocal.z * tan( radians( cameraState.fFov ) ));
    
    return vWindowPos;
}

// Function 125
vec3 transform(in vec3 p) {
	// camera ray rotation   
	if (iMouse.x > 0.0) {
		float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
		float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;   
		p.yz = rotate(p.yz,theta);
		p.zx = rotate(p.zx,-phi);
    }
	p.yz = rotate(p.yz, 0.1*iTime);
	p.zx = rotate(p.zx, 0.2*iTime);
	return p;
}

// Function 126
vec2 toRectCoords(vec2 polar)
{
    float x = polar.x * cos(polar.y);
    float y = polar.x * sin(polar.y);
    
    return vec2(x,y);
}

// Function 127
vec3 coord2to3(vec2 c)
{
    ivec2 N=ivec2(Res0/FRes.xy);
    vec2 cr=c/FRes.xy;
    vec2 indXY=floor(cr);
    vec2 cm=(cr-indXY)*FRes.xy;
    //vec2 cm=mod(c,FRes.xy);
    return vec3(cm,indXY.x+indXY.y*float(N.x)+.5);
}

// Function 128
void rayForFragCoord(in vec2 fragCoord, in vec2 resolution, in float time, out vec3 ro, out vec3 rd)
{
    vec3 pos = camera(time);
    float viewDist = distance(pos, CamTarget);

    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 dir = (CamTarget - pos) / viewDist;
    vec3 right = normalize(cross(dir, up));
    up = normalize(cross(right, dir));
    
    float imgU = tan(camera_fov()) * viewDist;
    float imgV = imgU * resolution.y / resolution.x;

    vec2 uv = fragCoord / resolution.xy * 2.0 - 1.0;
    
    dir = normalize(CamTarget + uv.x * imgU * right + uv.y * imgV * up - pos);
    
    ro = pos;
    rd = dir;
}

// Function 129
vec2 calcSphericalCoordsFromProjections(in vec2 screenCoord, in vec2 centralPoint, in vec2 FoVScale, in bool stereographic) {
    vec2 cp = (centralPoint * 2.0 - 1.0) * vec2(PI, PI_2);  // [-PI, PI], [-PI_2, PI_2]
    
    // Convert screen coord in gnomonic mapping to spherical coord in [PI, PI/2]
    vec2 convertedScreenCoord = (screenCoord * 2.0 - 1.0) * FoVScale * vec2(PI, PI_2); 
    float x = convertedScreenCoord.x, y = convertedScreenCoord.y;
    
    float rou = sqrt(x * x + y * y), c = stereographic ? 2.0 * atan(rou / localRadius / 2.0) : atan(rou); 
	float sin_c = sin( c ), cos_c = cos( c );  
    
    float lat = asin(cos_c * sin(cp.y) + (y * sin_c * cos(cp.y)) / rou);
	float lon = cp.x + atan(x * sin_c, rou * cos(cp.y) * cos_c - y * sin(cp.y) * sin_c);
    
	lat = (lat / PI_2 + 1.0) * 0.5; lon = (lon / PI + 1.0) * 0.5; //[0, 1]

    // uncomment the following if centralPoint ranges out of [0, PI/2] [0, PI]
	// while (lon > 1.0) lon -= 1.0; while (lon < 0.0) lon += 1.0;
	// while (lat > 1.0) lat -= 1.0; while (lat < 0.0) lat += 1.0;
    
    // convert spherical coord to cubemap coord
   return (bool(keyPressed(KEY_ENTER)) ? screenCoord : vec2(lon, lat)) * vec2(PI2, PI);
}

// Function 130
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,-theta);
    p.zx = rotate(p.zx,phi);
  }
  if (dorotate) {
    p.yz = rotate(p.yz,iTime * 0.125);
    p.zx = rotate(p.zx,iTime * 0.2);
  }
  return p;
}

// Function 131
vec3 round_cube_coord(vec3 barycentric) {
    vec3 rounded = round(barycentric);
    vec3 diff = abs(rounded-barycentric);
    if(diff.x > diff.y && diff.x > diff.z)
        rounded.x = -rounded.y-rounded.z;
    else if(diff.y > diff.z)
        rounded.y = -rounded.x-rounded.z;
    else
        rounded.z = -rounded.x-rounded.y;
    return rounded;
}

// Function 132
mat3 UTIL_axisRotationMatrix( vec3 u, float t )
{
    float c = cos(t);
    float s = sin(t);
    //  _        _   _           _     _                    _ 
    // |_px py pz_| | m11 m21 m31 |   | px*m11+py*m21+pz*m31 |
    //              | m12 m22 m32 | = | px*m12+py*m22+pz*m32 |
    //              |_m13 m23 m33_|   |_px*m13+py*m23+pz*m33_|
    return mat3(  c+u.x*u.x*(1.-c),     u.x*u.y*(1.-c)-u.z*s, u.x*u.z*(1.-c)+u.y*s,
	              u.y*u.x*(1.-c)+u.z*s, c+u.y*u.y*(1.-c),     u.y*u.z*(1.-c)-u.x*s,
	              u.z*u.x*(1.-c)-u.y*s, u.z*u.y*(1.-c)+u.x*s, c+u.z*u.z*(1.-c) );
}

// Function 133
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    p.yz = rotate(p.yz,-theta);
    p.zx = rotate(p.zx,phi);
  }
  if (dorotate) {
    p.yz = rotate(p.yz,iTime * 0.125);
    p.zx = rotate(p.zx,iTime * 0.1);
  }
  return p;
}

// Function 134
vec3 barycentricCoordinate(vec2 P,Pinwheel T)
{
    vec2 PA = P - T.A;
    vec2 PB = P - T.B;
    vec2 PC = P - T.C;
    
    vec3 r = vec3(
        det22(PB,PC),
        det22(PC,PA),
        det22(PA,PB)
    );
    
    return r / (r.x + r.y + r.z);
}

// Function 135
vec2 unswizzleChunkCoord(vec2 storageCoord) {
 	vec2 s = floor(storageCoord);
    float dist = max(s.x, s.y);
    float o = floor(dist / 2.);
    float neg = step(0.5, mod(dist, 2.)) * 2. - 1.;
    return neg * (s - o);
}

// Function 136
vec2 triangleCoord( vec3 v2, vec3 v0, vec3 v1 )
{
    float dot00=dot(v0,v0);
    float dot01=dot(v0,v1);
    float dot02=dot(v0,v2);
    float dot11=dot(v1,v1);
    float dot12=dot(v1,v2);
    float denom = dot00*dot11-dot01*dot01;
    if(denom<0.00001) return vec2(-1,-1);
    vec2  rval;
    rval.x = (dot11 * dot02 - dot01 * dot12) / denom;
    rval.y = (dot00 * dot12 - dot01 * dot02) / denom;
    return rval;
}

// Function 137
float UTIL_distanceToLineSeg(vec2 p, vec2 a, vec2 b)
{
    //       p
    //      /
    //     /
    //    a--e-------b
    vec2 ap = p-a;
    vec2 ab = b-a;
    //Scalar projection of ap in the ab direction = dot(ap,ab)/|ab| : Amount of ap aligned towards ab
    //Divided by |ab| again, it becomes normalized along ab length : dot(ap,ab)/(|ab||ab|) = dot(ap,ab)/dot(ab,ab)
    //The clamp provides the line seg limits. e is therefore the "capped orthogogal projection", and length(p-e) is dist.
    vec2 e = a+clamp(dot(ap,ab)/dot(ab,ab),0.0,1.0)*ab;
    return length(p-e);
}

// Function 138
vec3 UnprojectCoord( vec2 vTexCoord, vec4 vPlane )
{
    int iAxis = GetProjectionAxis( vPlane.xyz );
	return UnprojectCoord( vTexCoord, vPlane, iAxis );
}

// Function 139
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,theta);
    p.zx = rotate(p.zx,-phi);
  }
  if (dorotate) {
    float t = iTime;
    p.yz = rotate(p.yz, 0.1*t);
    p.zx = rotate(p.zx, 0.222*t);
  }
  return p;
}

// Function 140
vec2 cameraTransformation(vec2 p, float t)
{
    float a = t * 2. + sin(-t) * 1.5;
    mat2 r = mat2(cos(a), sin(a), -sin(a), cos(a));
    return (r * (p - vec2(cos(t / 2.), sin(t / 3.))) * 5. / pow(2., 1. + cos(t)));
}

// Function 141
bool persisting_coord(vec2 fragCoord) {
    if(fragCoord.x <= _MAX_INDEX && fragCoord.y == 0.5)
        return true;
    return false;
}

// Function 142
vec3 zup_spherical_coords_to_vector( float theta, float phi ) { vec2 theta_vec = unit_vector2( theta ); return vec3( theta_vec.y * unit_vector2( phi ), theta_vec.x ); }

// Function 143
int coordToCaveStateArrIndex(ivec2 coord)
{
    return coord.y * CAV_SIZ.x + coord.x;
}

// Function 144
UIDrawContext UIDrawContext_TransformChild( UIDrawContext parentContext, UIDrawContext childContext )
{
    UIDrawContext result;
    
    // The child canvas size is unmodified
    result.vCanvasSize = childContext.vCanvasSize;

    // Child viewport positions are in the parent's canvas
    // Transform them to screen co-ordinates    
    result.viewport.vPos = UIDrawContext_CanvasPosToScreenPos( parentContext, childContext.viewport.vPos );
    vec2 vMax = childContext.viewport.vPos + childContext.viewport.vSize;
    vec2 vScreenMax = UIDrawContext_CanvasPosToScreenPos( parentContext, vMax );
    result.viewport.vSize = vScreenMax - result.viewport.vPos;
    result.vOffset = childContext.vOffset;
    
    // Now clip the view so that it is within the parent view
    vec2 vViewMin = max( result.viewport.vPos, parentContext.clip.vPos );
    vec2 vViewMax = min( result.viewport.vPos + result.viewport.vSize, parentContext.clip.vPos + parentContext.clip.vSize );

    // Clip view to current canvas
    vec2 vCanvasViewMin = result.viewport.vPos - result.vOffset;
    vec2 vCanvasViewMax = vCanvasViewMin + result.vCanvasSize;
    
    vViewMin = max( vViewMin, vCanvasViewMin );
	vViewMax = min( vViewMax, vCanvasViewMax );
    
    result.clip = Rect( vViewMin, vViewMax - vViewMin );
    
    return result;
}

// Function 145
vec4 sunGlareCoords( mat4 cam, vec3 v, vec3 l )
{
	vec3 sy = normalize( cross( cam[0].xyz, l ) );
	vec3 sx = normalize( cross( l, sy ) );
	return vec4( normalize( vec2( dot( v, sx ), dot( v, sy ) ) ), dot( v, l ), -cam[2].z );
}

// Function 146
vec2 transformPos(vec2 pos) {
    pos = (pos - 0.5) * 4.0 + 0.5;
    pos = mod(pos, 1.0);
    return pos;
}

// Function 147
vec3 yup_spherical_coords_to_vector( vec2 theta, vec2 phi ) { return zup_spherical_coords_to_vector( theta, phi ).yzx ; }

// Function 148
vec2 transform_inverse(vec2 P)
{
    float x = P.x;
    float y = P.y;
    float lambda = atan(_sinh(x/(k0*a)),cos(y/(k0*a)));
    float phi    = asin(sin(y/(k0*a))/_cosh(x/(k0*a)));
    return vec2(lambda,phi);
}

// Function 149
vec4 transform(vec4 X)
{
    float alpha1	= iTime*TWOPI/51.0;
    float alpha2	= iTime*TWOPI/45.0;
    float beta1		= iTime*TWOPI/31.0;
    float beta2		= iTime*TWOPI/37.0;
    float beta3		= iTime*TWOPI/43.0;
    
    vec4 Y = X;
    Y.xw *= ROTATION(beta1);
    Y.yw *= ROTATION(beta2);
    Y.zw *= ROTATION(beta3);
    Y.xy *= ROTATION(alpha1);
    Y.zx *= ROTATION(alpha2);
	return Y;
}

// Function 150
Ray transform_ray( mat4 m, Ray ray ) { return mkray( transform_point( m, ray.o ), transform_vector( m, ray.d ) ); }

// Function 151
vec3 Transform(vec3 scale, vec3 rot, vec3 trans, vec3 p)
{
    p = Scale(scale, p);
    p = Rotate(rot, p);
    p = Translate(trans, p);

    return p;
}

// Function 152
vec2 calcSphericalCoordsInGnomonicProjection(in vec2 screenCoord, in vec2 centralPoint, in vec2 FoVScale) {
    vec2 cp = (centralPoint * 2.0 - 1.0) * vec2(PI, PI_2);  // [-PI, PI], [-PI_2, PI_2]
    
    // Convert screen coord in gnomonic mapping to spherical coord in [PI, PI/2]
    vec2 convertedScreenCoord = (screenCoord * 2.0 - 1.0) * FoVScale * vec2(PI, PI_2); 
    float x = convertedScreenCoord.x, y = convertedScreenCoord.y;
    
    float rou = sqrt(x * x + y * y), c = atan(rou); 
	float sin_c = sin( c ), cos_c = cos( c );  
    
    float lat = asin(cos_c * sin(cp.y) + (y * sin_c * cos(cp.y)) / rou);
	float lon = cp.x + atan(x * sin_c, rou * cos(cp.y) * cos_c - y * sin(cp.y) * sin_c);
    
	lat = (lat / PI_2 + 1.0) * 0.5; lon = (lon / PI + 1.0) * 0.5; //[0, 1]

    // uncomment the following if centralPoint ranges out of [0, PI/2] [0, PI]
	// while (lon > 1.0) lon -= 1.0; while (lon < 0.0) lon += 1.0;
	// while (lat > 1.0) lat -= 1.0; while (lat < 0.0) lat += 1.0;
    
    // convert spherical coord to cubemap coord
   return (bool(keyPressed(KEY_SPACE)) ? screenCoord : vec2(lon, lat)) * vec2(PI2, PI);
}

// Function 153
bool CalcWindowCoordinate(
	in vec2 inFragCoord,
	in vec2 inResolution,
	in vec2 inUVPosition,
	in vec2 inUVSize,
	out vec2 outWndFragCoord,
	out vec2 outWndResolution
	)
{
	outWndFragCoord	= inFragCoord - inUVPosition * inResolution;
	outWndResolution= inUVSize * inResolution;
	return all(bvec4(lessThanEqual(vec2(0.0), outWndFragCoord), lessThanEqual(outWndFragCoord, outWndResolution)));
}

// Function 154
vec2 transformUVs( in vec2 iuvCorner, in vec2 uv )
{
    // random in [0,1]^4
	vec4 tx = hash4( iuvCorner );
    // scale component is +/-1 to mirror
    tx.zw = sign( tx.zw - 0.5 );
    // debug vis
    #if JIGGLE
    tx.xy *= .05*sin(5.*iTime+iuvCorner.x+iuvCorner.y);
    #endif
    // random scale and offset
	return tx.zw * uv + tx.xy;
}

// Function 155
vec4 coord3to2(vec3 p)
{
    p.z-=.5;
    p=mod(p+FRes+FRes,FRes);
    p.xy=clamp(p.xy,vec2(.5),FRes.xy-.5);
    ivec2 N=ivec2(Res0/FRes.xy);
    int z1=int(p.z)%int(FRes.z);
    int z2=(z1+1)%int(FRes.z);
    vec2 xy1 = p.xy + vec2( float(z1%N.x)*FRes.x, float(z1/N.x)*FRes.y );
    vec2 xy2 = p.xy + vec2( float(z2%N.x)*FRes.x, float(z2/N.x)*FRes.y );
    return vec4(xy1,xy2);
}

// Function 156
vec2 perspectiveTransform(vec3 x, int seed){
    //x.z+=iMouse.x/iResolution.x-.5;
    float cam_pitch=1.0471975511966-3.14159/2.;
    float cam_yaw=3.14159/4.;//-3.14159/2.;
    if(iMouse.x>0.){
        cam_yaw = pi*2.*iMouse.x/iResolution.x;
        cam_pitch = pi*2.*(iMouse.y/iResolution.y-.5);
    }
    //float cam_perspective=0.1505;
    vec3 forward = normalize(vec3(cos(cam_pitch)*cos(cam_yaw),cos(cam_pitch)*sin(cam_yaw),sin(cam_pitch)));
    vec3 right = normalize(cross(forward,vec3(0,0,-1)));
    vec3 up = cross(right,forward);
    vec3 c0 = vec3(0,0,0);
    float a = dot(x-c0,forward);
    float b = dot(x-c0,-right);
    float c = dot(x-c0,up);
    return 25.*(vec2(b,c)-.2*center)+randc(rand2(seed))*a*0.0;///cam_perspective ;
}

// Function 157
vec3 GetCubeMapCoord(vec3 origin, vec3 direction)
{        
    float b = dot(2.0 * direction, origin);
    float c = dot(origin, origin) - pow(CubeDist, 2.0);        
    float z = (-b + sqrt(b*b - 4.0*c)) * 0.5;
    return normalize(origin + direction * z);
}

// Function 158
vec2 getCoords(vec2 c,vec2 r){
    return (c*2.-r)/min(r.x,r.y);
}

// Function 159
vec3 Transform(vec3 p, float angle)
{
    p.xz *= Rot(angle);
    p.xy *= Rot(angle *.7);
    return p;
}

// Function 160
vec2 pixelIndexToFragcoord( vec2 pixel_indexf, vec3 aResolution )
{
	// note that pixelIndexToFragcoord(floor(fragCoord))==fragCoord
	return aResolution.xy * ( ( vec2( 0.5 ) + pixel_indexf ) / aResolution.xy );
}

// Function 161
vec3 UVToViewSpaceCoord(float U, float V, float MinCos)
{
    float HalfFovV = kPi / 6.0;
    float AspRatio = iResolution.x / iResolution.y;

    float yScale = cos(HalfFovV) / sin(HalfFovV);
    float xScale = yScale / AspRatio;

    vec3 Dir;
    Dir.z = 10.0 / kKilometersToMeters;
    Dir.x = (U * 2.0 - 1.0) / xScale * Dir.z;
    Dir.y = (V * 2.0 - 1.0) / yScale * Dir.z;
    Dir = normalize(Dir);
    // clamp cosine of zenith angle
    Dir.xz /= Sqrt(1.0 - Dir.y * Dir.y);
    Dir.y   = clamp(Dir.y, MinCos, 1.0);
    Dir.xz *= Sqrt(1.0 - Dir.y * Dir.y);
    return Dir;
}

// Function 162
vec3 texCoords( in vec3 p )
{
	return 5.0*p;
}

// Function 163
vec3 transformForReflection(vec3 p, vec3 dir, vec3 refl) {
    float dist = sceneDist(p);

    return p + 2.0 * dist * (dir + refl);
}

// Function 164
ivec2 PointToTextureCoords(in vec3 p){
	int size = 32;
    const float squareSide = 4.;
    p = floor(p.xyz);
    p.xz = mod(p.xz,float(size));
    return ivec2(p.xz)+ivec2(int(mod(p.y,squareSide))*size,int(floor(p.y/squareSide))*size);
}

// Function 165
vec2 unswizzleChunkCoord(vec2 storageCoord) {
 	vec2 s = floor(storageCoord);
    float dist = max(s.x, s.y);
    float offset = floor(dist / 2.);
    float neg = step(0.5, mod(dist, 2.)) * 2. - 1.;
    return neg * (s - offset);
}

// Function 166
vec3 calcCubeCoordsInStereographicProjection(in vec2 screenCoord, in vec2 centralPoint, in vec2 FoVScale) {
	return sphericalToCubemap( calcSphericalCoordsInStereographicProjection(screenCoord, centralPoint, FoVScale) );
}

// Function 167
vec2 unswizzleChunkCoord(vec2 storageCoord) {
 	vec2 s = storageCoord;
    float dist = max(s.x, s.y);
    float offset = floor(dist / 2.);
    float neg = step(0.5, mod(dist, 2.)) * 2. - 1.;
    return neg * (s - offset);
}

// Function 168
mat3 CoordBase(vec3 n){
	vec3 x,y;
    frisvad(n,x,y);
    return mat3(x,y,n);
}

// Function 169
vec3 texToVoxCoord(vec2 textelCoord, vec3 offset) {
	vec3 voxelCoord = offset;
    voxelCoord.xy += unswizzleChunkCoord(textelCoord / packedChunkSize);
    voxelCoord.z += mod(textelCoord.x, packedChunkSize.x) + packedChunkSize.x * mod(textelCoord.y, packedChunkSize.y);
    return voxelCoord;
}

// Function 170
vec3 transform_point( mat4 m, vec3 p ) { return ( m * vec4( p, 1.0 ) ).xyz; }

// Function 171
vec3 outputTransform
(
vec3 In,
float Y_MIN,
float Y_MID,
float Y_MAX,
Chromaticities DISPLAY_PRI,
Chromaticities LIMITING_PRI,
int EOTF,
int SURROUND,
bool STRETCH_BLACK,
bool D60_SIM,
bool LEGAL_RANGE
)
{
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
TsParams PARAMS_DEFAULT = init_TsParams( Y_MIN, Y_MAX, 0.0);
float expShift = log2(inv_ssts(Y_MID, PARAMS_DEFAULT)) - log2(0.18);
TsParams PARAMS = init_TsParams( Y_MIN, Y_MAX, expShift);
vec3 rgbPre = rrt_sweeteners(In);
vec3 rgbPost = ssts_f3(rgbPre, PARAMS);
vec3 linearCV = Y_2_linCV_f3( rgbPost, Y_MAX, Y_MIN);
vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
if (SURROUND == 0) {
} else if (SURROUND == 1) {
if ((EOTF == 1) || (EOTF == 2) || (EOTF == 3)) {
XYZ = dark_to_dim( XYZ);
}
} else if (SURROUND == 2) {
}
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
if (D60_SIM == false) {
if ((DISPLAY_PRI.white.x != AP0.white.x) && (DISPLAY_PRI.white.y != AP0.white.y)) {
XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;
}
}
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
if (D60_SIM == true) {
float SCALE = 1.0;
if ((DISPLAY_PRI.white.x == 0.3127) && (DISPLAY_PRI.white.y == 0.329)) {
SCALE = 0.96362;
}
else if ((DISPLAY_PRI.white.x == 0.314) && (DISPLAY_PRI.white.y == 0.351)) {
linearCV.x = roll_white_fwd( linearCV.x, 0.918, 0.5);
linearCV.y = roll_white_fwd( linearCV.y, 0.918, 0.5);
linearCV.z = roll_white_fwd( linearCV.z, 0.918, 0.5);
SCALE = 0.96;
}
linearCV = linearCV * SCALE;
}
linearCV = max( linearCV, 0.0);
vec3 outputCV;
if (EOTF == 0) {
if (STRETCH_BLACK == true) {
outputCV = Y_2_ST2084_f3( max( linCV_2_Y_f3(linearCV, Y_MAX, 0.0), 0.0) );
} else {
outputCV = Y_2_ST2084_f3( linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN) );
}
} else if (EOTF == 1) {
outputCV = bt1886_r_f3( linearCV, 2.4, 1.0, 0.0);
} else if (EOTF == 2) {
outputCV = moncurve_r_f3( linearCV, 2.4, 0.055);
} else if (EOTF == 3) {
outputCV = pow_f3( linearCV, 1.0/2.6);
} else if (EOTF == 4) {
outputCV = linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN);
} else if (EOTF == 5) {
if (STRETCH_BLACK == true) {
outputCV = Y_2_ST2084_f3( max( linCV_2_Y_f3(linearCV, Y_MAX, 0.0), 0.0) );
}
else {
outputCV = Y_2_ST2084_f3( linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN) );
}
outputCV = ST2084_2_HLG_1000nits_f3( outputCV);
}
if (LEGAL_RANGE == true) {
outputCV = fullRange_to_smpteRange_f3( outputCV);
}
return outputCV;
}

// Function 172
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    p.yz = rotate(p.yz,theta);
    p.zx = rotate(p.zx,-phi);
  }
  if (!key(CHAR_R)) p.zx = rotate(p.zx,iTime * 0.2);
  return p;
}

// Function 173
vec2 rotateCoord(vec2 uv, float rads) {
    uv *= mat2(cos(rads), sin(rads), -sin(rads), cos(rads));
	return uv;
}

// Function 174
void stepTransform(inout vec3 p, inout float t) {
    p -= bloomPosition;
    p /= stepScale;
    globalScale *= stepScale;
    p *= bloomRotate;
    t -= delay;
}

// Function 175
void transformRay(inout Ray ray, mat4 matrix) {
  ray.origin = (matrix * vec4(ray.origin, 1.0)).xyz;
  ray.direction = normalize(matrix * vec4(ray.direction, 0.0)).xyz;
}

// Function 176
vec2 normalize_fragcoord(vec2 frag_coord) {
    return ((frag_coord/iResolution.x) - 0.5 * vec2(1.0, iResolution.y / iResolution.x)) * SCALE;
}

// Function 177
vec3 terrainTransform( in vec3 x ) {
    x.zy = rotate( -.83, x.zy );
    return x;
}

// Function 178
vec3 getPointV3ByFragCoord(vec2 coord){
    
    vec3 pointScreen = vec3(coord.x, coord.y, 0.5);
	vec3 pointV3 = unproject(pointScreen);

    return pointV3;
}

// Function 179
vec3 ToOtherSpaceCoord(mat3 otherSpaceCoord,vec3 vector){
	return vector * otherSpaceCoord;
}

// Function 180
vec2 fix_fcoord_for_vr( vec2 fcoord )
{
    if( g_vrmode )
    {
        vec2 xy = vr_unproject( vec3( 1.35, -1, +iResolution.y / iResolution.x ) );
        vec2 zw = vr_unproject( vec3( 1.35, +1, -iResolution.y / iResolution.x ) );
        fcoord = ( fcoord - unViewport.xy - xy ) * iResolution.xy / ( zw - xy );
    }
    return fcoord;
}

// Function 181
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float theta = -(2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = -(2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,theta);
    p.zx = rotate(p.zx,-phi);
  }
  if (true) {
    p.zx = rotate(p.zx,iTime * 0.2);
    p.xy = rotate(p.xy,iTime * 0.125);
  }
  return p;
}

// Function 182
vec2 ImportanceSampleGGXTransform( const vec2 uniformSamplePos, const in float alpha2 )
{
	// [Karis2013]  Real Shading in Unreal Engine 4
	// http://blog.tobias-franke.eu/2014/03/30/notes_on_importance_sampling.html

	float theta = acos( sqrt( (1.0f - uniformSamplePos.y) /
							( (alpha2 - 1.0f) * uniformSamplePos.y + 1.0f )
							) );

	float phi = 2.0f * PI * uniformSamplePos.x;

	return vec2( theta, phi );
}

// Function 183
CameraData GetCameraTransform( float aTime, vec4 aDate, vec3 aResolution, vec4 aMouse )
{
	CameraData data;

	data.tan_half_fovy = 0.6;

#ifdef EXTRA_3D_CAMERA
	data.camera = mat4( iCamera[0], iCamera[1], iCamera[2], iCamera[3] );
	data.tan_half_fovy = iTanHalfFovy;
#else
	data.camera = walkAndFlyCamera( data.tan_half_fovy, aTime, aDate, aResolution, aMouse );
#endif
	return data;
}

// Function 184
vec2 seeCoords(vec2 p)
{
    return p.xy;
}

// Function 185
vec2 ViewportTransform(vec3 p) 
{
	return vec2(
        (viewport_width / 2.) * p.x + (viewport_x + viewport_width / 2.),
        (viewport_eighth / 2.) * p.y + (viewport_y + viewport_eighth / 2.)
        );
}

// Function 186
vec3 transform(in vec3 p, in vec2 uv) {
    p.yz = crot(p.yz,pi * uv.y);
    p.zx = crot(p.zx,-pi * uv.x);
	return p;}

// Function 187
vec2 screen_coord(vec2 xy, vec2 dim) {
    return (xy - 0.5*dim) / min(dim.x, dim.y);
}

// Function 188
vec4 HexCoords(vec2 uv) {
    vec2 s = vec2(1, R3);
    vec2 h = .5*s;

    vec2 gv = s*uv;
    
    vec2 a = mod(gv, s)-h;
    vec2 b = mod(gv+h, s)-h;
    
    vec2 ab = dot(a,a)<dot(b,b) ? a : b;
    vec2 st = ab;
    vec2 id = gv-ab;
    
   // ab = abs(ab);
    //st.x = .5-max(dot(ab, normalize(s)), ab.x);
	st = ab;
    return vec4(st, id);
}

// Function 189
float tileCoordinate( float coord, float tilling, vec2 window, float offset, out float index) 
{   
    // Zoom
    float value = coord*tilling + offset;
    
    // Set tile index
    index = floor((value - window.x)/(window.y - window.x));
    
    // Chop in to windows
    value = modExpanded(value, window.x, window.y);
    
    return value;
}

// Function 190
vec2 get_coord(float i, float r, float np){
    float th = PI+(i*2.*PI/np);
    return vec2(r*cos(th), r*sin(th));
}

// Function 191
vec3 zup_spherical_coords_to_vector( float theta, float phi ) { return zup_spherical_coords_to_vector( unit_vector2( theta ), unit_vector2( phi ) ); }

// Function 192
int coordsToIndex( vec2 coords, float xResolution )
{
    return int((coords.y - 0.5) * xResolution + (coords.x - 0.5));
}

// Function 193
vec3 ptransform( in mat4 mat, in vec3 v ) { return (mat*vec4(v,1.0)).xyz; }

// Function 194
int CubeFaceCoords(vec3 p){

    // Elegant cubic space stepping trick, as seen in many voxel related examples.
    vec3 f = abs(p); f = step(f.zxy, f)*step(f.yzx, f); 
    
    ivec3 idF = ivec3(p.x<.0? 0 : 1, p.y<.0? 2 : 3, p.z<0.? 4 : 5);
    
    return f.x>.5? idF.x : f.y>.5? idF.y : idF.z; 
}

// Function 195
vec3 transform(in vec3 p) {
  if (halfspace) {
    p.yz = rotate(p.yz, -0.6);
  }
  if (iMouse.x > 0.0) {
    float theta = -(2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = -(2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,-theta);
    p.zx = rotate(p.zx,phi);
  }
  if (dorotate) {
    float t = iTime;
    if (centre) t *= 0.3;
    if (!halfspace) p.yz = rotate(p.yz,t*0.125);
    p.zx = rotate(p.zx,t * 0.2);
  }
  return p;
}

// Function 196
vec2 uvcoords(vec2 p) {
	vec2 uv = p / iResolution.xy;
    uv = uv * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;
    return uv;
}

// Function 197
vec3 ShipTransform( vec3 pos )
{
    pos -= vec3(2,-.5,3);
    
    vec3 rot = vec3(-.4,-.2,-.1); //pitch, yaw, roll (radians) - applied roll then pitch then yaw - left hand screw rule
    
/*this is the inverse, because we're transforming rays not the object
    vec2 d = vec2(-1,1);
    pos.xy = pos.xy*cos(rot.z)+sin(rot.z)*d*pos.yx;
    pos.yz = pos.yz*cos(rot.x)+sin(rot.x)*d*pos.zy;
    pos.zx = pos.zx*cos(rot.y)+sin(rot.y)*d*pos.xz;*/
    
    vec2 d = vec2(1,-1);
    pos.zx = pos.zx*cos(rot.y)+sin(rot.y)*d*pos.xz;
    pos.yz = pos.yz*cos(rot.x)+sin(rot.x)*d*pos.zy;
    pos.xy = pos.xy*cos(rot.z)+sin(rot.z)*d*pos.yx;
    
    return pos;
}

// Function 198
vec2 getSymmetricCoords(vec2 coords) {
	if (coords.x < 0.) coords.x = -coords.x;
    if (coords.y < 0.) coords.y = -coords.y;
    if (coords.y > coords.x) {
        float swp = coords.y;
        coords.y = coords.x;
        coords.x = swp;
    }
    return coords;
}

// Function 199
vec3 sphercoord(vec2 p) {
  float l1 = acos(p.x);
  float l2 = acos(-1.)*p.y;
  return vec3(cos(l1), sin(l1)*sin(l2), sin(l1)*cos(l2));
}

// Function 200
void rayForFragCoordViewSpace(in vec2 fragCoord, in vec2 resolution, in float time, out vec3 rd)
{
    vec3 pos = camera(time);
    float viewDist = distance(pos, CamTarget);

    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 dir = vec3(0.0, 0.0, 1.0);
    vec3 right = vec3(1.0, 0.0, 0.0);
    
    float imgU = tan(camera_fov()) * viewDist;
    float imgV = imgU * resolution.y / resolution.x;

    vec2 uv = fragCoord / resolution.xy * 2.0 - 1.0;
    
    dir = normalize(vec3(0.0, 0.0, viewDist) + uv.x * imgU * right + uv.y * imgV * up);
    
    rd = dir;
}

// Function 201
float FourierTransform(vec2 u,float scaleY){
 u.x=floor((u.x+iTime/scrollspeedX)*DFTitterations)/DFTitterations;
 //rasterizing by DFTitterations makes pixels square?
 u.x=(u.x-1.)*scrollspeedX;
 float n=DFTitterations;
 float a=2.*PI*floor(u.y*n*.5)/n;
 vec2 j=vec2(cos(a),sin(a));
 vec2 d=vec2(1,0),f=vec2(0);
 for(int i=0;i<int(n);i++){
  float x=float(i)/n,t=x*scaleY+u.x;
  vec2  w=synthWave(t);
  x=(w.x+w.y)*.25*(1.-cos(2.*PI*x));//Hann window
  f+=d*x;
  d=d.xy*j.x+vec2(-1,1)*d.yx*j.y;}
 float y=.5*length(f);
 return sat(sqrt(y/(1.+y)));}

// Function 202
void texcoords(vec2 fragCoord, vec2 resolution,
               out vec2 v_rgbNW, out vec2 v_rgbNE,
               out vec2 v_rgbSW, out vec2 v_rgbSE,
               out vec2 v_rgbM) {
    vec2 inverseVP = 1.0 / resolution.xy;
    v_rgbNW = (fragCoord + vec2(-1.0, -1.0)) * inverseVP;
    v_rgbNE = (fragCoord + vec2(1.0, -1.0)) * inverseVP;
    v_rgbSW = (fragCoord + vec2(-1.0, 1.0)) * inverseVP;
    v_rgbSE = (fragCoord + vec2(1.0, 1.0)) * inverseVP;
    v_rgbM = vec2(fragCoord * inverseVP);
}

// Function 203
vec4 getTriangleCoords(vec2 uv) {
    uv.y /= triangleScale;
    uv.x -= uv.y / 2.0;
    vec2 center = floor(uv);
    vec2 local = fract(uv);
    
    center.x += center.y / 2.0;
    center.y *= triangleScale;
    
    if (local.x + local.y > 1.0) {
    	local.x -= 1.0 - local.y;
        local.y = 1.0 - local.y;
        center.y += 0.586;
        center.x += 1.0; 
    } else {
        center.y += 0.287;
    	center.x += 0.5;
    }
    
    return vec4(center, local);
}

// Function 204
vec3 QTransform(vec3 v, vec4 q)
{
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Function 205
vec2 getCoord(vec2 coord)
{
    vec2 ts = getTileSize();
    
    int row = int(floor(coord.x / ts.x));
    int col = int(floor(coord.y / ts.y));
    
    return vec2(row, col);
}

// Function 206
vec2 unpackCoord(float f) 
{
    return vec2((mod(f, 512.0)), floor(f / 512.0)) * INV_SCALE;
}

// Function 207
bool IsCoord(ivec2 iU,ivec2 coord){
	return all(equal(iU,coord));
}

// Function 208
vec3 qtransform(vec4 q, vec3 v)
{ 
	return v + 2.0*cross(cross(v, q.xyz) + q.w*v, q.xyz);
}

// Function 209
vec2 icosahedronFaceCoordinates(vec3 p) {
    vec3 pn = normalize(p);
    vec3 i = intersection(pn, facePlane, -1.);
    return vec2(dot(i, uPlane), dot(i, vPlane));
}

// Function 210
vec2 rotateCoords(vec2 uv)
{
    float angle = 0.0;
    float time = mod(iTime, 23.0);
    if(time < 8.0)
    {
    	angle = smoothstep(0.0, 1.0, time * 0.125) * PI ;   
    }
	uv = rotate(uv, angle);
    
    return uv;
}

// Function 211
bool IsCoord(ivec2 iU,ivec2 coord){return all(equal(iU,coord));}

// Function 212
float GetBayerFromCoordLevel(vec2 pixelpos//https://www.shadertoy.com/view/XtV3RG
){ivec2 p=ivec2(pixelpos);int a=0
 ;for(int i=0;i<iterBayerMat;i++
){a+=bayer2x2(p>>(iterBayerMat-1-i)&1)<<(2*i)
 ;}return float(a)/float(2<<(iterBayerMat*2-1));}

// Function 213
GridCoords grid_coords(vec2 uv, vec2 grid_dims) {
    vec2 scaled = uv * grid_dims;
    GridCoords gridded;
    gridded.id = floor(scaled);
    gridded.uv = fract(scaled);
    
    //Flip coordinates back and forth every cell (around the circle only)
    //for added symmetry
    vec2 flipped = vec2(0.0, 1.0) - gridded.uv;
    float odd = mod(gridded.id.y, 2.0);
    gridded.flipped = mix(gridded.uv, flipped, odd);
        
    return gridded;
}

// Function 214
vec2 voxToTexCoord(vec3 voxCoord) {
    vec3 p = floor(voxCoord);
    return swizzleChunkCoord(p.xy) * packedChunkSize + vec2(mod(p.z, packedChunkSize.x), floor(p.z / packedChunkSize.x));
}

// Function 215
vec2 arrangeCoords(vec2 p)
{
    vec2 q = p.xy/iResolution.xy;
    vec2 r = -1.0+2.0*q;
	r.x *= iResolution.x/iResolution.y;
    return r;
}

// Function 216
vec2 NormalizeScreenCoords(vec2 screenCoord)
{
    vec2 result = 2.0 * (screenCoord/iResolution.xy - 0.5);
    result.x *= iResolution.x/iResolution.y;
    return result;
}

// Function 217
vec2 Cam_GetViewCoordFromUV( const in vec2 vUV )
{
	vec2 vWindow = vUV * 2.0;
	vWindow.x *= iResolution.x / iResolution.y;

	return vWindow;	
}

// Function 218
vec3 texToVoxCoord(vec2 textelCoord, vec3 offset,int bufferId) {

    vec2 packedChunkSize= packedChunkSize_C;

	vec3 voxelCoord = offset;
    voxelCoord.xy += unswizzleChunkCoord(textelCoord / packedChunkSize);
    voxelCoord.z += mod(textelCoord.x, packedChunkSize.x) + packedChunkSize.x * mod(textelCoord.y, packedChunkSize.y);
    return voxelCoord;
}

// Function 219
vec3 peano_transform(vec3 p){
    const float blockiness = ITERS>0 ? .8 : 1.;
    const float gap=pow(2.,float(-ITERS))/(1.-pow(2.,float(-ITERS))+blockiness);
    const float final_piece=2.*(pow(2.,float(-ITERS))-gap*(1.-pow(2.,float(-ITERS)))) + PI*.5*gap;
    
    float sublen=total_len=pow(8.,float(ITERS))*final_piece;
    float s=1.;
	float add=0.;
    float rev=1.;
    float flip=1.;
    for(int i=0;i<ITERS;++i){
        sublen*=1./8.;
        float a=(s+gap)*.5;
        vec3 sp=sign(p);
        if(sp==vec3(-1.,-1.,-1.)){p=vec3(+p.z+a,+p.y+a,+p.x+a); add-=rev*sublen*3.5; flip*=-1.; }
        if(sp==vec3(+1.,-1.,-1.)){p=vec3(+p.z+a,+p.x-a,+p.y+a); add-=rev*sublen*2.5; }
        if(sp==vec3(+1.,+1.,-1.)){p=vec3(+p.z+a,+p.x-a,-p.y+a); add-=rev*sublen*1.5; rev*=-1.; }
        if(sp==vec3(-1.,+1.,-1.)){p=vec3(-p.y+a,-p.x-a,+p.z+a); add-=rev*sublen*0.5; flip*=-1.; }
        if(sp==vec3(-1.,+1.,+1.)){p=vec3(-p.y+a,-p.x-a,-p.z+a); add+=rev*sublen*0.5; flip*=-1.; rev*=-1.; }
        if(sp==vec3(+1.,+1.,+1.)){p=vec3(-p.z+a,+p.x-a,-p.y+a); add+=rev*sublen*1.5; }
        if(sp==vec3(+1.,-1.,+1.)){p=vec3(-p.z+a,+p.x-a,+p.y+a); add+=rev*sublen*2.5; rev*=-1.; }
        if(sp==vec3(-1.,-1.,+1.)){p=vec3(+p.y+a,-p.z+a,+p.x+a); add+=rev*sublen*3.5; rev*=-1.; }
        s=(s-gap)*.5;
    }
    if(p.y>-gap && p.z<gap) p=vec3(p.x,length(p.yz-vec2(-gap,gap))-gap,(atan(p.y+gap,gap-p.z)-PI*.25)*gap);
    else if(-p.y>p.z) p=vec3(p.x,-p.z,p.y+(1.-PI*.25)*gap);
    else p.z+=(PI*.25-1.)*gap;
    p.z=p.z*rev+add;
    p.x*=flip;
    return p;
}

// Function 220
ivec2 indexToCoord(int index)
{
    ivec2 viewport = ivec2(iResolution + 0.5);
    
    return ivec2(index % viewport.x, index / viewport.x);
}

// Function 221
void transform(inout vec3 p){
  p = rotateX(p, pi * iTime * 0.3);
  p = rotateY(p, pi * iTime * 0.15);
}

// Function 222
vec2 ProjectCoordsSphere(vec2 normCoords)
{
    const float SPHERE_RADIUS_SQ = 1.0;
    //z^2 = R^2 - (x^2 + y^2).
    float z2 = SPHERE_RADIUS_SQ - dot(normCoords, normCoords);
    if(z2 <= 0.0)
        return normCoords;

    //Project the 3D point(normCoords.x, normCoords.y, sqrt(z2)) onto the screen
    //to emulate the sphere-like refraction.
    vec2 outProjectedCoords = normCoords / sqrt(z2);

    //Add an antialiasing step to avoid jagged edges on the lens.
    const float AA_EDGE = 0.2;
    if(z2 < AA_EDGE)
    {
        //Smooth transition of the squared z from 0 to AA_EDGE.
        //Normalize this transition factor to the [0,1] interval.
        float aaFactor = smoothstep(0.0, 1.0, z2 / AA_EDGE);
        
        //Perform another smooth transition between the projected coordinates and the original ones.
        //When z is very small, the projected coordinates are very big and tend to opint to the same position,
        //thus giving the edge of the lens a jagged appearance.
        outProjectedCoords = mix(
            normCoords, 
            outProjectedCoords,
        	aaFactor);
    }
    
    return outProjectedCoords;
}

// Function 223
vec2 CoordCross(vec3 u
){vec3 o=vec3(2,.1,2)
 ;vec2 b=vec2(box(u-o*vec3(1,-1,1),o),-1)
 ;float e=ma(ab(u))-9.*u5(cos(iTime))//distance from vec3(0) where coordinate crosses are drawn
 ;u=fract(u/2.-.5)-.5
 ;float c=ma(ab(u))+.125*e//width of each coordinate cross
 ;float y=step(abs(u.z),abs(u.x))
 ;u.xz=mix(u.xz,vec2(-u.z,u.x),step(abs(u.z),abs(u.x)))//;if(abs(u.y)<abs(u.x))u=vec2(-u.y,u.x)//cross mirror
 ;y+=step(abs(u.z),abs(u.y))*2.
 ;u.yz=mix(u.yz,vec2(-u.z,u.y),step(abs(u.z),abs(u.y)))//;if(abs(u.y)<abs(u.x))u=vec2(-u.y,u.x)//cross mirror
 ;float d=ma(ab(u.xy))+.003*e//thickness of coordinate crosses
 ;d=ma(d,c) 
 ;d=ma(d,e)
 ;vec2 r=vec2(d,y)
 ;r=minx(r,b)
 ;return r
 ;}

// Function 224
vec2 GetWindowCoord( const in vec2 vUV )
{
	vec2 vWindow = vUV * 2.0 - 1.0;
	vWindow.x *= iResolution.x / iResolution.y;

	return vWindow;	
}

// Function 225
vec3 Transform(in vec3 xyz, in STransform trans)
{
	mat3 r = Euler2Matrix(trans.m_euler);
	return r * xyz * trans.m_scale + trans.m_off; 
}

// Function 226
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,-theta);
    p.zx = rotate(p.zx,phi);
  }
  if (!key(CHAR_R)) {
    p.yz = rotate(p.yz,iTime * 0.125);
    p.zx = rotate(p.zx,iTime * 0.2);
  }
  return p;
}

// Function 227
vec4 transformVecByQuat( vec4 v, vec4 q )
{
    return vec4( transformVecByQuat( v.xyz, q ), v.w );
}

// Function 228
vec2 transform_inverse(vec2 P)
{
    float rho = length(P);
    float theta = atan(P.y,P.x);
    if( theta < 0.0 )
        theta = 2.0*M_PI+theta;
    return vec2(rho,theta);
}

// Function 229
vec2 icosahedronFaceCoordinates(vec3 p) {
    vec3 i = intersection(normalize(p), facePlane, -1.);
    return vec2(dot(i, uPlane), dot(i, vPlane));
}

// Function 230
vec2 applyTransform(in vec2 p)
{
    float t = time*.05;
#ifdef DEBUG
    if (iMouse.z > .001) t = iMouse.x/iResolution.x * numPhases;
#endif
    float pct = smoothstep(0., 1., mod(t, 1.));
    return mix(getTransform(p, t), getTransform(p, t+1.), pct);
}

// Function 231
vec2 getBokehTapSampleCoord(const in vec2 o, const in float f, const float n, const in float phiShutterMax){
    vec2 ab = (o * 2.0) - vec2(1.0);    
    vec2 phir = ((ab.x * ab.x) > (ab.y * ab.y)) ? vec2((abs(ab.x) > 1e-8) ? ((PI * 0.25) * (ab.y / ab.x)) : 0.0, ab.x) : vec2((abs(ab.y) > 1e-8) ? ((PI * 0.5) - ((PI * 0.25) * (ab.x / ab.y))) : 0.0, ab.y); 
    phir.x += f * phiShutterMax;
   	phir.y *= (f > 0.0) ? pow((cos(PI / n) / cos(phir.x - ((2.0 * (PI / n)) * floor(((n * phir.x) + PI) / (2.0 * PI))))), f) : 1.0;
    return vec2(cos(phir.x), sin(phir.x)) * phir.y;
}

// Function 232
vec2 voxToTexCoord(vec3 voxCoord) {
    vec2 packedChunkSize= packedChunkSize;
    vec3 p = floor(voxCoord);
    return swizzleChunkCoord(p.xy) * packedChunkSize + vec2(mod(p.z, packedChunkSize.x), floor(p.z / packedChunkSize.x));
}

// Function 233
vec2 transform(float time, vec2 offset)
{
	vec2 mid = vec2(0.5, 0.5);
	return rotate(vec2(cos(time*1.07)*0.2, sin(time)*0.2) + offset - mid, sin(time/10.0)*5.00)*(cos(time*0.87)/2.0+1.0) + mid;
}

// Function 234
vec3 zup_spherical_coords_to_vector( float theta, float phi )
{
	vec2 theta_vec = unit_vector2( theta );
	vec2 phi_vec = unit_vector2( phi );
	return vec3( theta_vec.y * phi_vec, theta_vec.x );
}

// Function 235
vec2 voxToTexCoord(vec3 voxCoord,int bufferId) {

    vec2 packedChunkSize= packedChunkSize_C;

    vec3 p = floor(voxCoord);
    return swizzleChunkCoord(p.xy) * packedChunkSize + vec2(mod(p.z, packedChunkSize.x), floor(p.z / packedChunkSize.x));
}

// Function 236
vec2 swizzleChunkCoord(vec2 chunkCoord) {
    vec2 c = chunkCoord;
    float dist = max(abs(c.x), abs(c.y));
    vec2 c2 = floor(abs(c - 0.5));
    float offset = max(c2.x, c2.y);
    float neg = step(c.x + c.y, 0.) * -2. + 1.;
    return (neg * c) + offset;
}

// Function 237
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    p.yz = rotate(p.yz,theta);
    p.zx = rotate(p.zx,-phi);
  }
  if (dorotate) {
    p.yz = rotate(p.yz,iTime * 0.125);
    p.zx = rotate(p.zx,iTime * 0.1);
  }
  return p;
}

// Function 238
vec3 invOutputTransform
(
vec3 In,
float Y_MIN,
float Y_MID,
float Y_MAX,
Chromaticities DISPLAY_PRI,
Chromaticities LIMITING_PRI,
int EOTF,
int SURROUND,
bool STRETCH_BLACK,
bool D60_SIM,
bool LEGAL_RANGE
)
{
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI);
TsParams PARAMS_DEFAULT = init_TsParams( Y_MIN, Y_MAX, 0.0);
float expShift = log2(inv_ssts(Y_MID, PARAMS_DEFAULT)) - log2(0.18);
TsParams PARAMS = init_TsParams( Y_MIN, Y_MAX, expShift);
vec3 outputCV = In;
if (LEGAL_RANGE == true) {
outputCV = smpteRange_to_fullRange_f3( outputCV);
}
vec3 linearCV;
if (EOTF == 0) {
if (STRETCH_BLACK == true) {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, 0.0);
} else {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, Y_MIN);
}
} else if (EOTF == 1) {
linearCV = bt1886_f_f3( outputCV, 2.4, 1.0, 0.0);
} else if (EOTF == 2) {
linearCV = moncurve_f_f3( outputCV, 2.4, 0.055);
} else if (EOTF == 3) {
linearCV = pow_f3( outputCV, 2.6);
} else if (EOTF == 4) {
linearCV = Y_2_linCV_f3( outputCV, Y_MAX, Y_MIN);
} else if (EOTF == 5) {
outputCV = HLG_2_ST2084_1000nits_f3( outputCV);
if (STRETCH_BLACK == true) {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, 0.0);
} else {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, Y_MIN);
}
}
if (D60_SIM == true) {
float SCALE = 1.0;
if ((DISPLAY_PRI.white.x == 0.3127) && (DISPLAY_PRI.white.y == 0.329)) {
SCALE = 0.96362;
linearCV = linearCV * (1.0 / SCALE);
}
else if ((DISPLAY_PRI.white.x == 0.314) && (DISPLAY_PRI.white.y == 0.351)) {
SCALE = 0.96;
linearCV.x = roll_white_rev( linearCV.x / SCALE, 0.918, 0.5);
linearCV.y = roll_white_rev( linearCV.y / SCALE, 0.918, 0.5);
linearCV.z = roll_white_rev( linearCV.z / SCALE, 0.918, 0.5);
}
}
vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
if (D60_SIM == false) {
if ((DISPLAY_PRI.white.x != AP0.white.x) && (DISPLAY_PRI.white.y != AP0.white.y)) {
XYZ = invert_f33(calculate_cat_matrix(AP0.white, REC709_PRI.white)) * XYZ;
}
}
if (SURROUND == 0) {
} else if (SURROUND == 1) {

if ((EOTF == 1) || (EOTF == 2) || (EOTF == 3)) {
XYZ = dim_to_dark( XYZ);
}
} else if (SURROUND == 2) {
}
linearCV = XYZtoRGB(AP1) * XYZ;
vec3 rgbPost = linCV_2_Y_f3( linearCV, Y_MAX, Y_MIN);
vec3 rgbPre = inv_ssts_f3( rgbPost, PARAMS);
vec3 aces = inv_rrt_sweeteners( rgbPre);
return aces;
}

// Function 239
vec2 coordToWorld(vec2 fragCoord){
    vec2 uv = fragCoord/iResolution.xy;
    uv.y /= (iResolution.x/iResolution.y);
    uv.y = 1.0-uv.y;
    return uv;
}

// Function 240
vec2 transform(vec3 p) {
    float o = 20.0;
    vec2 dx = anglevec(radians(o));
    vec2 dy = anglevec(radians(o+120.0));
    vec2 dz = vec2(0.0, 0.33);
    
    return p.x * dx + p.y * dy + p.z * dz;
}

// Function 241
vec3 buffer2coord(vec2 b) {  
    return vec3(
        floor(b.x/BLOCK_BUFFER.x),
        floor((b.y-1.)/BLOCK_BUFFER.y),
        floor(mod(b.x, BLOCK_BUFFER.x))* BLOCK_BUFFER.x +floor(mod((b.y-1.), BLOCK_BUFFER.y))
     );

}

// Function 242
vec3 fan_transform_xy(in vec3 pos, in float center, in float range) {
    center *= 0.017453292519943295;
    range *= 0.017453292519943295;
    float start = (center - range/2.),
          ang = atan(pos.x, pos.y),
          len = length(pos.xy);
    ang = mod(ang-start, range) - range/2. + center;
    pos.xy = len * vec2(sin(ang), cos(ang));
    return pos;
}

// Function 243
vec3 transformVecByQuat( vec3 v, vec4 q )
{
    return v + 2.0 * cross( q.xyz, cross( q.xyz, v ) + q.w*v );
}

// Function 244
void invtransform(vec2 p, vec2 bmin, vec2 bdelta, vec2 bmax, float unitScale,
                  float lineThickness, float aspect, int index, float blend,
                  out vec2 q, out mat2 J) {

  // center of current box
  vec2 center = screen2World(bmin + 0.5 * bdelta, bmin, unitScale, aspect);

  // create gradient variable
  GNum2 x = varG2x(p.x);
  GNum2 y = varG2y(p.y);

  // coordinate system in center
  GNum2 cx = sub(x, center.x);
  GNum2 cy = sub(y, center.y);

  // 5 transforms
  GNum2 tx[5];
  GNum2 ty[5];

  // polar coordinates
  {
    // scaling to better showcase distortion
    GNum2 sx = mult(cx, 0.75);
    GNum2 sy = mult(cy, 0.75);
    tx[0] = invRPolar(sx, sy);
    ty[0] = invThetaPolar(sx, sy);
  }

  // y' = y/(1.5 - sin(2pix))
  {

    tx[1] = cx;
    ty[1] = mult(cy, sub(1.5, a_sin(mult(2.0 * pi, cx))));
  }

  // y' = y*(1 + (2x)^2)
  {
    GNum2 xm = mult(x, 2.0);

    tx[2] = x;
    ty[2] = div(y, add(1.0, mult(xm, xm)));
  }

  // x' = exp(x)
  {
    tx[3] = a_log(x);
    ty[3] = y;
  }

  // identity
  {
    tx[4] = x;
    ty[4] = y;
  }

  // interpolate one function with the next with the given factor
  // -> nice transitions
  int idxp = int(mod(float(index + 1), 5.0));
  GNum2 rx = a_mix(tx[index], tx[idxp], blend);
  GNum2 ry = a_mix(ty[index], ty[idxp], blend);

  // coordinate in original system
  q.x = rx.val;
  q.y = ry.val;

  // Jacobian
  // since we need extra parameters we have to create the Jacobian by ourself
  // instead of using the JACOBIAN2 macro
  J = transpose(mat2(rx.g, ry.g));
}

// Function 245
vec3 transformframe(vec3 p) {
  if (iMouse.x > 0.0) {
    // Full range of rotation across the screen.
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    p.yz = rotate(p.yz,theta);
    p.zx = rotate(p.zx,-phi);
  }
  // autorotation
  if (dorotate) {
    p.yz = rotate(p.yz,-iTime*0.125);
    p.zx = rotate(p.zx,iTime*0.1);
  }
  return p;
}

// Function 246
vec2 GetScreenPixelCoord( vec2 vScreenUV )
{
        vec2 vPixelPos = floor(vScreenUV * kResolution);
        vPixelPos.y = 192.0 - vPixelPos.y;
       
        return vPixelPos;
}

// Function 247
vec2 wrapFragCoord(vec2 fragCoord) {
    // Simulate wrap: mirror
    return abs(1023.0 - mod(fragCoord + 1023.0, 2046.0));
}

// Function 248
vec2 texCoords( in vec3 pos, int mid )
{
    vec2 matuv;
    
    if( mid==0 )
    {
        matuv = pos.xz;
    }
    else if( mid==1 )
    {
        vec3 q = normalize( pos - sc0.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc0.w;
    }
    else if( mid==2 )
    {
        vec3 q = normalize( pos - sc1.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc1.w;
    }
    else if( mid==3 )
    {
        vec3 q = normalize( pos - sc2.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc2.w;
    }
    else if( mid==4 )
    {
        vec3 q = normalize( pos - sc3.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc3.w;
    }

	return 8.0*matuv;
}

// Function 249
vec3 transform( in vec4 p )
{
    p.xw *= rot(iTime*0.41);
    p.yw *= rot(iTime*0.23);
    p.xy *= rot(iTime*0.73);
    p.wz *= rot(iTime*0.37);
    
    // orthogonal projection
    #if PROJECTION==0
    return p.xyz;
    #else
    // perspective projection
	return 2.5*p.xyz/(3.0+p.w);
    #endif
}

// Function 250
vec3 transform_vector( mat4 m, vec3 v ) { return ( m * vec4( v, 0.0 ) ).xyz; }

// Function 251
vec3 TransformHeadPos( vec3 vPos )
{
    return (vPos + g_sceneState.vNeckOffset) * g_sceneState.mHeadRot - g_sceneState.vNeckOffset;
}

// Function 252
void fragCoordFromVCube(in vec3 vcube, out int page, out vec2 fragCoord)
{
    vec2 p;
    if (abs(vcube.x) > abs(vcube.y) && abs(vcube.x) > abs(vcube.z)) {
        if (vcube.x > 0.0) { page = 1; } else { page = 2; }
        p = vcube.yz/vcube.x;
    } else if (abs(vcube.y) > abs(vcube.z)) {
        if (vcube.y > 0.0) { page = 3; } else { page = 4; }
        p = vcube.xz/vcube.y;
    } else {
        if (vcube.z > 0.0) { page = 5; } else { page = 6; }
        p = vcube.xy/vcube.z;
    }

    fragCoord = floor((0.5 + 0.5*p)*1024.0);
}

// Function 253
vec3 sphereTransformation(vec3 pt){
    // Rotate around the Z axis, slowly accelerating:
    float r0 = iTime*exp(-1.0/pow(0.01+0.01*iTime,2.0));
    float cr0 = cos(r0);
    float sr0 = sin(r0);
    pt = vec3(pt.x*cr0-pt.y*sr0, pt.x*sr0+pt.y*cr0, pt.z);
    
    // Rotate around the X axis:
    float r1 = iTime*1.0;
    float sr1 = sin(r1);
    float cr1 = cos(r1);
    pt=vec3(pt.x, pt.y*cr1-pt.z*sr1, pt.y*sr1+pt.z*cr1);
    
    return pt;
}

// Function 254
vec2 transform_inverse(vec2 P)
{
    const float B = 2.0;
    float x = P.x;
    float y = P.y;
    float z = 1.0 - (x*x/16.0) - (y*y/4.0);
    if (z < 0.0)
        discard;
    z = sqrt(z);
    float lon = 2.0*atan( (z*x),(2.0*(2.0*z*z - 1.0)));
    float lat = asin(z*y);
    return vec2(lon,lat);
}

// Function 255
vec2 texCoords( in vec3 pos, int mid )
{
    vec2 matuv;
    
    if( mid==0 )
    {
        matuv = pos.xz;
    }
    else if( mid==1 )
    {
        vec3 q = normalize( pos - sc0.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc0.w;
    }
    else if( mid==2 )
    {
        vec3 q = normalize( pos - sc1.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc1.w;
    }
    else if( mid==3 )
    {
        vec3 q = normalize( pos - sc2.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc2.w;
    }
    else if( mid==4 )
    {
        vec3 q = normalize( pos - sc3.xyz );
        matuv = vec2( atan(q.x,q.z), acos(q.y ) )*sc3.w;
    }

	return 12.0*matuv;
}

// Function 256
vec3 Custom_OutputTransform(vec3 aces, float Y_MIN, float Y_MID, float Y_MAX, int p_OT_Display, int p_OT_Limit, int EOTF, int SURROUND, bool STRETCH_BLACK, bool D60_SIM, bool LEGAL_RANGE)
{
Chromaticities OT_DISPLAY_PRI, OT_LIMIT_PRI;
OT_DISPLAY_PRI = REC2020_PRI;
OT_LIMIT_PRI = REC2020_PRI;

if(p_OT_Display == 1)
OT_DISPLAY_PRI = P3D60_PRI;
if(p_OT_Display == 2)
OT_DISPLAY_PRI = P3D65_PRI;
if(p_OT_Display == 3)
OT_DISPLAY_PRI = P3DCI_PRI;
if(p_OT_Display == 1)
OT_DISPLAY_PRI = REC709_PRI;

if(p_OT_Limit == 1)
OT_LIMIT_PRI = P3D60_PRI;
if(p_OT_Limit == 2)
OT_LIMIT_PRI = P3D65_PRI;
if(p_OT_Limit == 3)
OT_LIMIT_PRI = P3DCI_PRI;
if(p_OT_Limit == 1)
OT_LIMIT_PRI = REC709_PRI;

return outputTransform(aces, Y_MIN, Y_MID, Y_MAX, OT_DISPLAY_PRI, OT_LIMIT_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE);
}

// Function 257
v3 Transform(int i,v3 p
){p.xyz=objPos[i]-p.xyz
 ;p.xyz=qr(objRot[i],p.xyz)
 ;p.xyz/=objSca[i]
 ;p.w*=dot(v2(1),abs(objSca[i]))/3.//distance field dilation approx
 ;return p;}

// Function 258
void coordinateSystem(const vec3 v1, out vec3 v2, out vec3 v3) {
    if (abs(v1.x) > abs(v1.y)) {
        v2 = vec3(-v1.z, 0, v1.x) / length(v1.xz);
    }
    else {
        v2 = vec3(0, v1.z, -v1.y) / length(v1.yz);
    }
    v3 = cross(v1, v2);
}

// Function 259
vec2 screenToSquareTransform(vec2 p, float t, float i)
{
    float a = t * 2. + cos(t + i) + i / 3.;
    mat2 m = mat2(cos(a), sin(a), -sin(a), cos(a));
    vec2 o = vec2(cos(t * 2. * 2. + i * 5.5) / 2., sin(t * 2.9 * 2. + i * 1.27) * .25);
    o.y += cos(t / 4. + i) * .1;
    return m * (p - o) * (3. + sin(t - i) * 1.5) * 8.;
}

// Function 260
mat3 polarTransformation(float longitude, float latitude)
{
    vec3 X = vec3(1,0,0);
    vec3 Y = vec3(0,1,0);
    vec3 Z = vec3(0,0,1);
    
    mat3 m = rotationMatrix(Z, longitude);
    Y = m * Y;
    
    m = rotationMatrix(Y, latitude) * m;
    X = m * X;
    Z = m * Z;
    
    return mat3(X, Y, Z);
}

// Function 261
float normCoordsToClickboxVal(vec2 dr) {
    return clamp(dr[0] * 0.5 + 0.5, 0., 1.) + round(500. * clamp(dr[1] * 0.5 + 0.5, 0., 1.));
}

// Function 262
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float theta = -(2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = -(2.0*iMouse.x-iResolution.x)/iResolution.y*PI;
    p.yz = rotate(p.yz,phi);
    p.zx = rotate(p.zx,phi);
  }
  //p.yz = rotate(p.yz,0.1*iTime * 0.125);
  p.zx = rotate(p.zx,0.1*iTime);
  return p;
}

// Function 263
vec2 pixelCoord(vec2 fragCoord) { 
	return ((fragCoord - (iResolution.xy / 2.0)) / min(iResolution.x, iResolution.y)); 
}

// Function 264
mat4 yup_spherical_coords_to_matrix( float theta, float phi )
{
 vec3 y = yup_spherical_coords_to_vector( theta, phi );
 vec3 z = yup_spherical_coords_to_vector( theta + 3.141592654 * 0.5, phi );
 vec3 x = cross( y, z );
 return mat4( vec4( x, 0.0 ), vec4( y, 0.0 ), vec4( z, 0.0 ), vec4( 0, 0, 0, 1 ) );
}

// Function 265
vec3 genLightCoords()
{
	vec3 lightCoords = vec3(lightPathCenter.x + (sin(iTime*timeScale)*lightPathRadius), 
				lightPathCenter.y + (cos(iTime*timeScale)*lightPathRadius),
				lightPathCenter.z);
	return lightCoords;
}

// Function 266
vec4 HexCoords(vec2 uv, out vec2 gv) {
	vec2 r = vec2(1, 01.-(0.502*(cos(iTime/4.))));
    vec2 h = r*.5;
    vec2 a = mod(uv, r)-h;
    vec2 b = mod(uv-h, r)-h;
    gv = dot(a, a) < dot(b,b) ? a : b;
    float x = atan(gv.x, gv.y);
    float y = .5-HexDist(gv);
    vec2 id = uv-gv;
    return vec4(x, y, id.x,id.y);
}

// Function 267
vec2 normalizeScreenCoords(vec2 screenCoord)
{
    vec2 result = 2.0*(screenCoord/iResolution.xy - 0.5);
    result.x *= iResolution.x/iResolution.y;
    return result;
}

// Function 268
vec2 GetNextCoordinates(vec2 loc)
{
    float behav = GetSceneTileBehaviour(loc);
    vec4 vss = GetSceneTileVss(loc);
    
    float lsloc = vss.w == 0.0 && behav != kBehavWater? loc.x : GetSceneTileLocalSpaceLoc(loc.x, vss);
    float lsloc0 = floor(lsloc) + 0.5;
    
    loc.y = floor(loc.y) + 0.5;
    loc.x += (lsloc0 - lsloc);    
    return loc;
}

// Function 269
vec2 convertCoord(vec2 p){ return mod(p.xy, dSize)/dSize; }

// Function 270
vec4
color_from_polar_coordinates(vec2 origin, vec2 uv)
{
    vec2 uv_recentered = origin - uv;

    float r = sqrt(
		uv_recentered.x * uv_recentered.x 
		/ (ASPECT_RATIO * ASPECT_RATIO) 
            + uv_recentered.y* uv_recentered.y);
	
    float theta = atan(uv_recentered.y / uv_recentered.x);

    vec4 return_color;

    if (about_equal(r, 0.2))
    {
        return_color = vec4(0,0,1,1);
    }
    else
    {
        return_color = vec4(r, theta, 0, 1.0);
    }

    if (about_equal(theta, -0.5))
    {
        return_color = vec4(1.0,1.0,1.0,1.0);
    }

    return return_color;
}

// Function 271
mat3 Transform(vec3 direction, vec3 up)
{
    vec3 z = direction;
    vec3 x = normalize(cross(up, z));
    vec3 y = normalize(cross(z, x));
    return mat3(x, y, z);
}

// Function 272
vec2 Cam_GetViewCoordFromUV( vec2 vUV, float fAspectRatio )
{
	vec2 vWindow = vUV * 2.0 - 1.0;
	vWindow.x *= fAspectRatio;

	return vWindow;	
}

// Function 273
vec3 transformCheckerboardScene(
    in vec2 uv,
    in float aspect, 
    int storyState, in float t0, in float lengthState)
{
    vec3 colRaster = vec3(0);
    vec3 colTextured = vec3(0);

    if (isOnAfigLogo(uv, iResolution.x / iResolution.y)) {
        vec2 rasterUV = uv;

        vec3 col = vec3(0.);
        rasterUV = 2.* rasterUV - vec2(1.);
        rasterUV.x *= aspect;

        if (!downgrade(rasterUV, rasterUV)) {
            colRaster = vec3(0.);
        } else {
            colRaster = green_phosphore;
        }

        rasterUV.x /= aspect;
        rasterUV = rasterUV / 2. + vec2(0.5);

    }

    float t = t0 / lengthState;

    vec2 texCoords;

    if (isInAfigLogo(uv, aspect, texCoords)) {
        colTextured = checkerboardTexture(texCoords);
    }

    return mix(colRaster, colTextured, t);
}

// Function 274
vec3 texToVoxCoord(vec2 textelCoord, vec3 offset) {

    vec2 packedChunkSize= packedChunkSize;
	vec3 voxelCoord = offset;
    voxelCoord.xy += unswizzleChunkCoord(textelCoord / packedChunkSize);
    voxelCoord.z += mod(textelCoord.x, packedChunkSize.x) + packedChunkSize.x * mod(textelCoord.y, packedChunkSize.y);
    return voxelCoord;
}

// Function 275
vec3 vcubeFromFragCoord(int page, vec2 fragCoord)
{
    vec2 p = (wrapFragCoord(fragCoord) + 0.5)*(2.0/1024.0) - 1.0;

    vec3 fv;
    if (page == 1) {
        fv = vec3(1.0, p);
    } else if (page == 2) {
        fv = -vec3(1.0, p);
    } else if (page == 3) {
        fv = vec3(p.x, 1.0, p.y);
    } else if (page == 4) {
        fv = -vec3(p.x, 1.0, p.y);
    } else if (page == 5) {
        fv = vec3(p, 1.0);
    } else if (page == 6) {
        fv = -vec3(p, 1.0);
    }
    return fv;
}

// Function 276
vec2 toPolarCoords(vec2 pos)
{
    float radius = length(pos);
    float angle = atan(pos.y, pos.x);
    return vec2(radius, angle);
}

// Function 277
vec2 Cam_GetViewCoordFromUV( vec2 vUV, vec2 res )
{
	vec2 vWindow = vUV * 2.0 - 1.0;
	vWindow.x *= res.x / res.y;

	return vWindow;	
}

// Function 278
vec2 particleCoord(int idx, sampler2D s)
{
    ivec2 ires=textureSize(s,0);
    return vec2(idx%ires.x,idx/ires.x)+.5;
}

// Function 279
vec2 sphereCoords(vec2 _st, float _scale) {
  float maxFactor = sin(1.570796327);
  vec2 uv = vec2(0.0);
  vec2 xy = 2.0 * _st.xy - 1.0;
  float d = length(xy);
  if (d < (2.0 - maxFactor)) {
    d = length(xy * maxFactor);
    float z = sqrt(1.0 - d * d);
    float r = atan(d, z) / 3.1415926535 * _scale;
    float phi = atan(xy.y, xy.x);
    uv.x = r * cos(phi) + 0.5;
    uv.y = r * sin(phi) + 0.5;
  } else {
    uv = _st.xy;
  }
  return uv;
}

// Function 280
vec2 transformPos(vec2 pos) {
    pos = (pos - 0.5) * 4.0 + 0.5;
    pos = mod(pos, 1.0);
	//pos = clamp(pos, 0.0, 1.0);
    pos = vec2(0.5);
    return pos;
}

// Function 281
vec3 texCoords( in vec3 p )
{
	return 3.0*p;
}

// Function 282
vec3 makelinecoords(float t, vec2 centre, vec3 mobius) {
  float A = mobius.x, B = mobius.y, C = mobius.z;
  t -= 0.05*iTime;
  t = (tan(t*PI)-C)/A;
  // Infinities!
  if (abs(t) > 1e4) t = sign(t)*1e4;
  t = (t-B)/(1.0+t*B);
  return join(vec3(centre,1),vec3(centre+vec2(1,t),1));
}

// Function 283
vec2 ProjectCoordsLogLens(vec2 normCoords)
{
    float z = -log(dot(normCoords, normCoords));
    if(z <= 0.0)
        return normCoords;

    //Project the 3D point(normCoords.x, normCoords.y, sqrt(z2)) onto the screen
    //to emulate the sphere-like refraction.
    vec2 outProjectedCoords = normCoords / z;

    //Add an antialiasing step to avoid jagged edges on the lens.
    const float AA_EDGE = 0.2;
    if(z < AA_EDGE)
    {
        //Smooth transition of the squared z from 0 to AA_EDGE.
        //Normalize this transition factor to the [0,1] interval.
        float aaFactor = smoothstep(0.0, 1.0, z / AA_EDGE);
        
        //Perform another smooth transition between the projected coordinates and the original ones.
        //When z is very small, the projected coordinates are very big and tend to opint to the same position,
        //thus giving the edge of the lens a jagged appearance.
        outProjectedCoords = mix(
            normCoords, 
            outProjectedCoords,
        	aaFactor);
    }
    
    return outProjectedCoords;
}

// Function 284
vec3 transform(vec3 p) {
  if (iMouse.x > 0.0) {
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    //float phi = 0.1*iTime; //(2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,-theta);
    p.zx = rotate(p.zx,phi);
  }
  p.yz = rotate(p.yz,-0.8);
  return p;
}

// Function 285
float packfragcoord2 (vec2 p, vec2 s) {
    return floor(p.y) * s.x + p.x;
}

// Function 286
vec2 coord(vec2 c, vec2 r)
{
    return vec2
    (
        (c.x - ((r.x - r.y) / 2.0)) / r.y,
        c.y / r.y
    );
}

// Function 287
float UTIL_distanceToLineSeg(vec2 p, vec2 a, vec2 b)
{
    //Scalar projection of ap in the ab direction = dot(ap,ab)/|ab| : Amount of ap aligned towards ab
    //Divided by |ab| again, it becomes normalized along ab length : dot(ap,ab)/(|ab||ab|) = dot(ap,ab)/dot(ab,ab)
    //The clamp provides the line seg limits. e is therefore the "capped orthogogal projection".
    //       p
    //      /
    //     /
    //    a--e-------b
    vec2 ap = p-a;
    vec2 ab = b-a;
    vec2 e = a+clamp(dot(ap,ab)/dot(ab,ab),0.0,1.0)*ab;
    return length(p-e);
}

// Function 288
vec2 GetCoord( float offset )
{
    float y = floor(offset / iResolution.x);
    return vec2( offset - y * iResolution.x, y ) + .5;
}

// Function 289
vec2 deformCoords(vec2 uv)
{
    float t = mod(iTime, 20.0);
    float t01 = 0.0;
    if(t > 17.0) t01 = (t-17.0) / 3.0;
    
    int id0 = int(floor(mod(iTime/20.0, 8.0)));
    int id1 = int(floor(mod(iTime/20.0 + 1.0, 8.0)));
    
	vec2 uv1 = calcCoordsID(uv, id0, 0.0);
	vec2 uv2 = calcCoordsID(uv, id1, 0.0);
	uv = mix(uv1, uv2, t01);

    return uv;
}

// Function 290
vec2 Cam_GetUVFromWindowCoord( vec2 vWindow, vec2 res )
{
    vec2 vScaledWindow = vWindow;
    vScaledWindow.x *= res.y / res.x;

    return (vScaledWindow * 0.5 + 0.5);
}

// Function 291
vec2 fragCoordToXY(vec2 fragCoord) {
  vec2 relativePosition = fragCoord.xy / iResolution.xy;
  float aspectRatio = iResolution.x / iResolution.y;

  vec2 cartesianPosition = (relativePosition - 0.5) * 4.0;
  cartesianPosition.x *= aspectRatio;

  return cartesianPosition;
}

// Function 292
vec2 clickboxCoordsNorm(float value) {
    return 2. * vec2(fract(value) - 0.5, floor(value) / 500. - 0.5);
}

// Function 293
vec4 LineSegCoord(vec2 p1, vec2 p2, vec2 uv, out float segmentLength){
    

    vec2 vector = p2 - p1;                         // Find the vector between the two lines
          uv   -= p1;                              // Move the entire coord system so that the point 1 sits on the origin, it is either that or always adding point 1 when you want to find your actual point
    float len   = max(length(vector), 0.01);                  // Find the ditance between the two points
       vector  /= len;                             // normalize the vector 
    float vUv   = dot(vector, uv);                 // Find out how far the projection of the current pixel on the line goes along the line using dot product
    vec2  p     = vector * clamp(vUv, 0.,len) ;    // since vector is normalized, the if you multiplied it with the projection amount, you will get to the coordinate of where the current uv has the shortest distance on the line. The clamp there ensures that this point always remains between p1 and p2, take this out if you want an infinite line
    vec2 ToLine = p - uv;                       
    float d     = length(ToLine);                  // the actual distance between the current pixel and its projection on the line
    
    vec2 ortho    = vec2(vector.y, -vector.x);     // For 3D you would have to use cross product or something
    float signedD = dot(ortho, ToLine);            // this gives you a signed distance between the current pixel and the line. in contrast to the value d, first this value is signed, so different on the different sides of the line, and second, for a line segment with finite ends, beyond the finit end, the magnitude of this value and d start to differ. This value will continue to get smaller, as you go around the corner on the finit edge and goes into negative
    segmentLength = len;
    
                                                   // fourth component is used for drawing the branch thickness, is a noramlized value stating how far the pixel is between p1 nad p2
    return vec4(vUv, d, signedD, clamp(vUv, 0.,len)/ len); 
}

// Function 294
vec3 nextCoord(cell c, int d, int btype){
    
    vec2 dir=DIRS[d];
    if(btype==5){
       	int npos=-1;
        if(d==0) switch (c.pos){
			case 19: npos=25;break;
            case 25: npos=28;break;
            case 28: npos=22;break;
            case 10: npos=16;break;
            case 7: npos=10;break;
            case 13: npos=7;break;
            default: break;

        }else if(d==2) switch (c.pos){
        	case 25: npos=19;break;
            case 28: npos=25;break;
            case 22: npos=28;break;
            case 16: npos=10;break;
            case 10: npos=7;break;
            case 7: npos=13;break;
            default: break;
        }
        
        if((d%2)==0 &&
            (c.pos==9 || c.pos==14 ||c.pos==15 ||c.pos==20 ||c.pos==21 ||c.pos==26 )    
                ) npos=5;

        if(npos!=-1) return vec3(c.block_pos,float(npos));
    }
	return buffer2coord(coord2buffer(vec3(c.block_pos,c.pos)) +dir);
}

// Function 295
vec3 zup_spherical_coords_to_vector( vec2 theta_phi ) { return zup_spherical_coords_to_vector( theta_phi.x, theta_phi.y ); }

// Function 296
vec2 getTransform(in vec2 p, float t)
{
    int which = int(mod(t, numPhases));

    if (which == 0) {
        p = mapSquare(p);
        p = pow(vec2(.3), abs(p));
        p = rotate(time*.1)*p;
        p += .1*sin(time*.2);
        p = dupSquares(p);
        p -= .1*sin(time*.2);
        p = dupSquares(p);
    } else if (which == 1) {
        p = pow(abs(p), vec2(.5));
        p = mapSquare(p);
        p = pow(abs(p), vec2(3.));
        p += .1*sin(time*.2);
        p = dupSquares(p);
        p = rotate(time*.1)*p;
        p = dupGrid(p);
        p -= .1;
        p = rotate(time*.1)*p;
    } else if (which == 2) {
        p = mapSquare(p);
        p = dupGrid(p*.5);
        p += .2 + .1*sin(time*.2);
        p = dupSquares(p);
        p = rotate(time*.1)*p;
        p = dupSquares(p);
    } else if (which == 3) {
        p = mapSquare(p);
        p = dupGrid(p*.7);
        p = dupSquaresConcentric(p);
        p = rotate(time*.1)*p;
        p = dupSquares(p);
        p += .3*sin(time*.2);
        p = pow(abs(p), vec2(.5));
        p = dupSquares(p);
    } else if (which == 4) {
        p = pow(vec2(.3), abs(p));
        p = mapSquare(p);
        p = dupGrid(p);
        p = dupSquaresConcentric(p);
        p = rotate(time*.1)*p;
        p = dupSquares(p);
        p += .3*sin(time*.2);
        p = pow(abs(p), vec2(.5));
        p = dupSquares(p);
    } else if (which == 5) {
        p = pow(vec2(.3), abs(p));
        p = mapSquare(p);
        p = dupGrid(p);
        p = dupSquaresConcentric(p);
        p += .3*sin(time*.2);
        p = rotate(time*.1)*p;
        p = dupSquares(p);
        p = pow(abs(p), vec2(.5));
        p = dupSquares(p);
    }
#if 0  // REJECTS
    }

// Function 297
vec2 transform_forward(vec2 P)
{
    const float B = 2.0;
    float longitude = P.x;
    float latitude  = P.y;
    float cos_lat = cos(latitude);
    float sin_lat = sin(latitude);
    float cos_lon = cos(longitude/B);
    float sin_lon = sin(longitude/B);
    float d = sqrt(1.0 + cos_lat * cos_lon);
    float x = (B * M_SQRT2 * cos_lat * sin_lon) / d;
    float y =     (M_SQRT2 * sin_lat) / d;
    return vec2(x,y);
}

// Function 298
vec2 fragCoordForViewPos(in vec3 viewPos, in vec2 resolution, in float time)
{
    vec3 pos = camera(time);
    
    float viewDist = distance(pos, CamTarget);
    
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 dir = vec3(0.0, 0.0, 1.0);//(CamTarget - pos) / viewDist;
    vec3 right = vec3(1.0, 0.0, 1.0);

    float imgU = tan(camera_fov()) * viewDist;
    float imgV = imgU * resolution.y / resolution.x;

    float dView = dot(viewPos, dir);
    float dProj = viewDist;
    vec3 projPos = (viewPos) * (dProj / dView) - vec3(0.0, 0.0, viewDist);
    
    vec2 uv = vec2(dot(projPos, right) / imgU,
                   dot(projPos, up)    / imgV) * 0.5 + 0.5;
    return uv * resolution.xy;
}

// Function 299
vec2 transform_forward(vec2 P)
{
    float lambda = P.x;
    float phi = P.y;
    float x = 0.5*k0*log((1.0+sin(lambda)*cos(phi))
            / (1.0 - sin(lambda)*cos(phi)));
    float y = k0*a*atan(tan(phi), cos(lambda));
    return vec2(x,y);
}

// Function 300
vec3 transform(in vec3 p) {
  const float PI = 3.14159;
  if (iMouse.x > 0.0) {
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,-theta);
    p.zx = rotate(p.zx,phi);
  }
  return p;
}

// Function 301
vec2 TransformFromCanvasTextureToFramedTexture(
	vec2 canvasTextureCoord,
	vec2 canvasTextureSize,
	vec2 framedTextureSize)
{	
	vec2 result = (canvasTextureCoord / canvasTextureSize);

	float canvasAspectRatio = (canvasTextureSize.x / canvasTextureSize.y);
	float framedAspectRatio = (framedTextureSize.x / framedTextureSize.y);

	if (framedAspectRatio < canvasAspectRatio)
	{
		float relativeAspectRatio = (canvasAspectRatio / framedAspectRatio);

		result.x *= relativeAspectRatio;
		result.x -= (0.5 * (relativeAspectRatio - 1.0));
	}
	else
	{
		float relativeAspectRatio = (framedAspectRatio / canvasAspectRatio);

		result.y *= relativeAspectRatio;
		result.y -= (0.5 * (relativeAspectRatio - 1.0));
	}

	return result;
}

// Function 302
vec3 cameraTransform(vec3 vector) {
    return rotateVector(rotateVector(vector, vec3(1,0,0), PI/6.0*0.5), vec3(0,1,0), PI/8.0*iTime*1.0) + 1.0*vec3(0, cos(iTime)*80.0+100.0, -200);
}

// Function 303
void asteroidTransForm(inout vec3 ro, const in vec3 id ) {
    float xyangle = (id.x-.5)*time*2.;
    ro.xy = rotate( xyangle, ro.xy );
    
    float yzangle = (id.y-.5)*time*2.;
    ro.yz = rotate( yzangle, ro.yz );
}

// Function 304
vec3 transformframe(vec3 p) {
  if (iMouse.x > 0.0) {
    // Full range of rotation across the screen.
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    p.yz = rotate(p.yz,theta);
    p.zx = rotate(p.zx,-phi);
  }
  // autorotation - we always rotate a little as otherwise nothing can
  // be seen (since the z-axis is part of the model).
  float t = 1.0;
  if (dorotate) t += iTime;
  p.yz = rotate(p.yz,-t*0.125);
  p.zx = rotate(p.zx,-t*0.1);
  return p;
}

// Function 305
vec2 normalizeScreenCoords(vec2 screenCoord)
{
    float fov=1.9;
    vec2 result = fov*(screenCoord/iResolution.xy - 0.5);
    result.x *= iResolution.x/iResolution.y;
    return result;
}

// Function 306
vec2 Cam_GetUVFromWindowCoord( const in vec2 vWindow, float fAspectRatio )
{
    vec2 vScaledWindow = vWindow;
    vScaledWindow.x /= fAspectRatio;

    return (vScaledWindow * 0.5 + 0.5);
}

// Function 307
vec2 calcSphericalCoordsInStereographicProjection(in vec2 screenCoord, in vec2 centralPoint, in vec2 FoVScale) {
	return calcSphericalCoordsFromProjections(screenCoord, centralPoint, FoVScale, true); 
}

// Function 308
mat4 yup_spherical_coords_to_matrix( vec2 theta, vec2 phi )
{
	vec3 y = yup_spherical_coords_to_vector( theta, phi );
	vec3 z = yup_spherical_coords_to_vector( perp( theta ), phi ); // note: perp(theta) = unit_vector2(theta+PI*0.5)
	vec3 x = cross( y, z );
	return ( mat4( vec4( x, 0.0 ), vec4( y, 0.0 ), vec4( z, 0.0 ), vec4( 0, 0, 0, 1 ) ) );
}

// Function 309
uint maxPowerFromCoord (in ivec3 coord, in bool isAboveOpaque)
{
    uint maxPower = 0u;
    
    Voxel voxel = readVoxel(coord);
	if (voxel.type == VOXEL_TYPE_REDSTONE_TORCH)
    {
        maxPower = max(maxPower, voxel.energy);
    }
    else if (voxel.type == VOXEL_TYPE_REDSTONE_DUST
             && voxel.energy > 0u)
    {
        maxPower = max(maxPower, voxel.energy - 1u);
    }
    else if (voxel.type == VOXEL_TYPE_VOID)
    {
        Voxel voxelMinusZ = readVoxel(coord + ivec3(0,0,-1)),
            voxelPlusZ = readVoxel(coord + ivec3(0,0,1));
        
        if (voxelMinusZ.type == VOXEL_TYPE_REDSTONE_DUST
            && voxelMinusZ.energy > 0u)
        {
            maxPower = max(maxPower, voxelMinusZ.energy - 1u);
        }
        if (voxelPlusZ.type == VOXEL_TYPE_REDSTONE_DUST
            && voxelPlusZ.energy > 0u)
        {
            maxPower = max(maxPower, voxelPlusZ.energy - 1u);
        }
    }
    else if (voxel.type == VOXEL_TYPE_STONE && !isAboveOpaque)
    {
    	Voxel voxelPlusZ = readVoxel(coord + ivec3(0,0,1));
        if (voxelPlusZ.type == VOXEL_TYPE_REDSTONE_DUST
            && voxelPlusZ.energy > 0u)
        {
            maxPower = max(maxPower, voxelPlusZ.energy - 1u);
        }
    }
    
    return maxPower;
}

// Function 310
bool getBldGridCoord(in vec2 fragCoord, out int x, out int y, out int n) {
	x = int(fragCoord.x);
    y = int(fragCoord.y);
    n = x % BLD_GRID_MAX;
    x = x / BLD_GRID_MAX;    
    return x < BLD_GRID_W && y < BLD_GRID_H;
}

// Function 311
vec3 transformVecByQuat( vec3 v, vec4 q )
{
    return (v + 2.0 * cross( q.xyz, cross( q.xyz, v ) + q.w*v ));
}

// Function 312
mat3 calculateEyeRayTransformationMatrix( in vec3 ro, in vec3 ta, in float roll )
{
    vec3 ww = normalize( ta - ro );
    vec3 uu = normalize( cross(ww,vec3(sin(roll),cos(roll),0.0) ) );
    vec3 vv = normalize( cross(uu,ww));
    return mat3( uu, vv, ww );
}

// Function 313
vec2 calculate_coord(vec2 c){
    vec2 normalized_coord = c/iResolution.xy;
    float t = iTime;
    normalized_coord -= vec2(0.5f) + vec2(sin(t*0.1f)*0.3f, cos(t*0.1f)*0.3f);
    normalized_coord *= 3.0 - cos(t*0.2)*2.0f;
    return normalized_coord;
}

// Function 314
vec3 QuatTransformVec(vec3 v, vec4 q)
{
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Function 315
vec2 arrowTileCenterCoord(vec2 pos) {
	return (floor(pos / ARROW_TILE_SIZE) + 0.5) * ARROW_TILE_SIZE;
}

// Function 316
vec2 transform_forward(vec2 P)
{
    float x = P.x * cos(P.y);
    float y = P.x * sin(P.y);
    return vec2(x,y);
}

// Function 317
vec3 InvTransformHeadPos( vec3 vPos )
{
    return  g_sceneState.mHeadRot * (vPos + g_sceneState.vNeckOffset) - g_sceneState.vNeckOffset;
}

// Function 318
int GetCoord(int x, int y)
{
   return (int(cMapSize.y)*y)+x;
}

// Function 319
vec3 terrainUntransform( in vec3 x ) {
    x.zy = rotate( .83, x.zy );
    return x;
}

// Function 320
vec2 coordAdjust(vec4 box, vec2 coords){
    coords.xy /= iResolution.xy;
    coords.x *= box.z - box.x;
    coords.y *= box.w - box.y;
    coords.xy += box.xy; 
    return coords;
}

// Function 321
repeatInfo UTIL_repeat(vec2 p, float interval)
{
    repeatInfo rInfo;
    rInfo.pRepeated = p / interval; //Normalize
    rInfo.pRepeated = fract(rInfo.pRepeated+0.5)-0.5; //centered fract
    rInfo.pRepeated *= interval; //Rescale
    rInfo.anchor = p-rInfo.pRepeated;
    return rInfo;
}

// Function 322
vec3 transform(in vec3 p) {
  if (iMouse.x > 0.0) {
    float theta = (2.0*iMouse.y-iResolution.y)/iResolution.y*PI;
    float phi = (2.0*iMouse.x-iResolution.x)/iResolution.x*PI;
    p.yz = rotate(p.yz,theta);
    p.zx = rotate(p.zx,-phi);
  }
  return p;
}

// Function 323
mat4 calc_transform(inout KIFS kifs) {
    float angle = kifs.angle * DEGREES_TO_RADIANS;

    float c = cos(angle);
    float s = sin(angle);

    vec3 t = (1.0-c) * kifs.axis;

    return mat4(
        vec4(c + t.x * kifs.axis.x, t.y * kifs.axis.x - s * kifs.axis.z, t.z * kifs.axis.x + s * kifs.axis.y, 0.0) * kifs.scale,
        vec4(t.x * kifs.axis.y + s * kifs.axis.z, (c + t.y * kifs.axis.y), t.z * kifs.axis.y - s * kifs.axis.x, 0.0) * kifs.scale,
        vec4(t.x * kifs.axis.z - s * kifs.axis.y, t.y * kifs.axis.z + s * kifs.axis.x, c + t.z * kifs.axis.z, 0.0) * kifs.scale,
        vec4(kifs.offset, 1.0)
    );
}

// Function 324
void transformNormal(inout vec3 normal, in mat4 matrix) {
  normal = normalize((matrix * vec4(normal, 0.0)).xyz);
}

// Function 325
float packfragcoord3 (vec3 p, vec3 s) {
    return floor(p.z) * s.x * s.y + floor(p.y) * s.x + p.x;
}

// Function 326
vec2 coord2buffer(vec3 c){
	return vec2(
        floor(c.x) * BLOCK_BUFFER.x  + floor(c.z/BLOCK_BUFFER.x),
        floor(c.y) * BLOCK_BUFFER.y  + floor(mod(c.z,BLOCK_BUFFER.x)) +1.
    );
}

// Function 327
vec3 InvTransform(in vec3 xyz, in STransform trans)
{
	vec3 re = xyz - trans.m_off;
	mat3 r = Euler2Matrix(-trans.m_euler);
	return r * re / trans.m_scale; 
}

// Function 328
vec2 SpiralCoords(vec2 st, float turns) {
	// polar coords in... spiral coords out. Spiral coordinates are neat!
    st.x = st.x/twopi +.5;
    st.y *= turns;
    float s = st.y+st.x;
    float l = (floor(s)-st.x);
    float d = fract(s);
    return vec2(l, d);
}

// Function 329
vec2 unpackCoord(float f) 
{
    return vec2(mod(f, 512.0),floor(f / 512.0)) / 511.0;
}

// Function 330
GridCoords grid_coords(vec2 uv, vec2 grid_dims) {
    vec2 scaled = uv * grid_dims;
    GridCoords gridded;
    gridded.cell_id = floor(scaled);
    gridded.cell_uv = fract(scaled);
    return gridded;
}

// Function 331
vec2 GetScreenSpaceCoord(const in vec2 fragCoord)
{
	vec2 mUV = (fragCoord.xy / iResolution.xy) * 2.0 - 1.0;
	mUV.x *= iResolution.x / iResolution.y;

	return mUV;	
}

// Function 332
vec2 unpackCoord(float f) 
{
    // Needs a C++ 'modf', but sadly no...
    return vec2(mod(f, 512.0),floor(f / 512.0)) * INV_SCALE;
}

