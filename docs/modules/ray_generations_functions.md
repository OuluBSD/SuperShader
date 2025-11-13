# ray_generations_functions

**Category:** raymarching
**Type:** extracted

## Dependencies
texture_sampling, normal_mapping, lighting, raymarching

## Tags
lighting, texturing, color, raymarching

## Code
```glsl
// Reusable Ray Generations Raymarching Functions
// Automatically extracted from raymarching/raytracing-related shaders

// Function 1
vec3 rayDir(vec3 ro, vec3 at, vec2 uv) {
	vec3 f = normalize(at - ro),
		 r = normalize(cross(vec3(0, 1, 0), f));
	return normalize(f + r * uv.x + cross(f, r) * uv.y);
}

// Function 2
vec3 GetRayColour( const in vec3 vRayOrigin, const in vec3 vRayDir )
{
	Intersection intersection;
    return GetRayColour( vRayOrigin, vRayDir, intersection );
}

// Function 3
float getRayleigMultiplier(vec2 p, vec2 lp)
{
    float dist = greatCircleDist(p, lp)/pi*5.;
	return 1.0 + pow(1.0 - clamp(dist, 0.0, 1.0), 2.0) * pi * 0.5;
}

// Function 4
vec3 rayDirection(vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(CAMERA_FOV) / 2.0);
    return normalize(vec3(xy, -z));
}

// Function 5
Ray getRay(Camera cam, vec2 uv)
{
    vec2 rd = cam.lensRadius * randomInUnitDisk(gSeed);
    vec3 offset = cam.u * rd.x + cam.v * rd.y;
    float time = cam.time0 + hash1(gSeed) * (cam.time1 - cam.time0);
    return createRay(
        cam.origin + offset,
        normalize(cam.lowerLeftCorner + uv.x * cam.horizontal + uv.y * cam.vertical - cam.origin - offset),
        time);
}

// Function 6
vec3 rayDir(vec3 ro, vec3 lookAt, vec2 uv) {
	vec3 f = normalize(lookAt - ro),
	     r = normalize(cross(vec3(0, 1, 0), f));
	return normalize(f + r * uv.x + cross(f, r) * uv.y);
}

// Function 7
vec3 GetRayDir(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l-p),
        r = normalize(cross(vec3(0,1,0), f)),
        u = cross(f,r),
        c = f*z,
        i = c + uv.x*r + uv.y*u,
        d = normalize(i);
    return d;
}

// Function 8
vec3 rayDirection(float fieldOfView, vec2 fragCoord, vec2 resolution) {
    vec2 xy = fragCoord - resolution.xy / 2.0;
    float z = (0.5 * resolution.y) / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

// Function 9
vec3 getRay(vec3 rayDir, float rot) {
	rayDir = normalize(rayDir);
    float cosVal = cos(rot);
    float sinVal = sin(rot);    
    return vec3((rayDir.x * cosVal) + (rayDir.z * sinVal), rayDir.y, (rayDir.z * cosVal) - (rayDir.x * sinVal));
}

// Function 10
Ray getRay(vec3 lookFrom, vec3 targetDir, vec3 upVector, float FOV, float aspect, float aperture, float focusDist, vec2 fragCoord) {
    float halfHeight = tan(radians(FOV*0.5));
    float halfWidth = aspect * halfHeight;
    vec3 origin = lookFrom;
    vec3 w = targetDir;
    vec3 u = normalize(cross(upVector, w));
    vec3 v = cross(w, u);
    vec3 lowerLeftCorner = origin
        - halfWidth * focusDist * u
        - halfHeight * focusDist * v
        - focusDist * w;
    vec3 horizontal = 2.0 * halfWidth * focusDist * u;
    vec3 vertical = 2.0 * halfHeight * focusDist * v;

    vec2 rd = 0.5*aperture * randomInUnitDisk(randomSeed);
    vec3 offset = u * rd.x + v * rd.y;
    return Ray(origin + offset, normalize(lowerLeftCorner 
               + (fragCoord.x/iResolution.x) * horizontal 
               + (fragCoord.y/iResolution.y) * vertical
               - origin - offset));
}

// Function 11
ray GetRay(vec2 uv, vec3 camPos, vec3 lookat, float zoom){
        ray a;
        a.o = camPos;
        
        vec3 f = normalize(lookat - camPos);
        vec3 r = cross(vec3(0.,1,0.), f);
        vec3 u = cross(f, r);//zoom is the distance of camera to screen,.
        vec3 c = a.o +  f* zoom;    //center of screen point, f is normalized
        vec3 i = c + uv.x * r + uv.y * u;
        a.d = normalize(i - a.o);//distance of ray from focus to screen.
        return a;
    
    }

// Function 12
vec3 getRay_870892966(mat3 camMat, vec2 screenPos, float lensLength) {
  return normalize(camMat * vec3(screenPos, lensLength));
}

// Function 13
vec3 rayDir( float fov, vec2 size, vec2 pos )
{
	vec2 xy = pos - size * 0.5;

	float cot_half_fov = tan( ( 90.0 - fov * 0.5 ) * DEG_TO_RAD );	
	float z = size.y * 0.5 * cot_half_fov;
	
	return normalize( vec3( xy, z ) );
}

// Function 14
vec4 RayDir(vec2 pixPos,
            vec2 viewSizes,
            inout uint randVal,
            int frameCtr)
{
    viewSizes *= float(MAX_SPP_PER_AXIS);
    pixPos *= float(MAX_SPP_PER_AXIS);
    vec2 sampleXY = vec2(frameCtr % MAX_SPP_PER_AXIS,
                   		 frameCtr / MAX_SPP_PER_AXIS); // Offset the current ray within the sampling grid
    sampleXY += AAJitter(randVal); // Jitter the given offset
    pixPos += sampleXY; // Apply offset to the scaled pixel position
    vec3 dir = vec3(pixPos - (viewSizes / 2.0),
                	viewSizes.y / tan(1.62 / 2.0)); // Generate ray direction
    return vec4(normalize(dir), // Normalize
                BlackmanHarris(sampleXY)); // Generate filter value
}

// Function 15
vec3 getRay_7_2(vec3 origin, vec3 target, vec2 screenPos, float lensLength) {
  mat3 camMat = calcLookAtMatrix_8_1(origin, target, 0.0);
  return getRay_7_2(camMat, screenPos, lensLength);
}

// Function 16
vec3 getRayColor(Ray ray) {


    float d = mix(DENSITY_MIN, DENSITY_MAX, (ray.eta - ETA)/(1./ETA-ETA));
    vec3 matColor = mix(AIR_COLOR, MATERIAL_COLOR, (ray.eta - ETA)/(1./ETA-ETA));
    vec3 col = getColor(ray);

    float q = exp(-d*ray.cp.dist);
    col = col*q+matColor*(1.-q);
    return col*ray.share;
}

// Function 17
vec3 getRay(vec2 pos)
{
    mat3 rmat = getRot(angles);
    vec2 uv = FOV*(pos - R*0.5)/R.x;
    return normalize(rmat[0]*uv.x + rmat[1]*uv.y + rmat[2]);
}

// Function 18
vec3 GetRay( vec3 dir, float zoom, vec2 uv )
{
	uv = uv - .5;
	uv.x *= iResolution.x/iResolution.y;
	
	dir = zoom*normalize(dir);
	vec3 right = normalize(cross(vec3(0,1,0),dir));
	vec3 up = normalize(cross(dir,right));
	
	return normalize(dir + right*uv.x + up*uv.y);
}

// Function 19
vec3 rayDir(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l-p),
        r = normalize(cross(vec3(0,1,0), f)),
        u = cross(f,r),
        c = p+f*z,
        i = c + uv.x*r + uv.y*u,
        d = normalize(i-p);
    return d;
}

// Function 20
vec3 GetRay( vec3 dir, float zoom, vec2 uv )
{
	uv = uv - .5;
	uv.x *= iResolution.x/iResolution.y;
	
	dir = zoom*normalize(dir);
	vec3 right = normalize(cross(vec3(0,1,0),dir));
	vec3 up = normalize(cross(dir,right));
	
	return dir + right*uv.x + up*uv.y;
}

// Function 21
vec3 GetCameraRayDir( const in vec2 vWindow, const in vec3 vCameraPos, const in vec3 vCameraTarget )
{
	vec3 vForward = normalize(vCameraTarget - vCameraPos);
	vec3 vRight = normalize(cross(vec3(0.0, 1.0, 0.0), vForward));
	vec3 vUp = normalize(cross(vForward, vRight));
							  
	vec3 vDir = normalize(vWindow.x * vRight + vWindow.y * vUp + vForward * 2.0);

	return vDir;
}

// Function 22
vec3 GetRayDir(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l-p), // forward
        r = normalize(cross(vec3(0,1,0), f)), // right
        u = cross(f,r), // up
        c = f*z, // 
        i = c + uv.x*r + uv.y*u,
        d = normalize(i);
    return d;
}

// Function 23
vec3 getRayDir(vec3 camDir, vec2 fragCoord) {
  vec3 yAxis = vec3(0, 1, 0);
  vec3 xAxis = normalize(cross(camDir, yAxis));
  vec2 q = fragCoord / iResolution.xy;
  vec2 p = 2.0 * q - 1.0;
  p.x *= iResolution.x / iResolution.y;
  return normalize(p.x * xAxis + p.y * yAxis + 5.0 * camDir);
}

// Function 24
ray getRay(vec2 uv, camera camera) {
    ray ray;
    ray.origin = camera.origin;
    vec3 center = ray.origin + camera.forward * camera.zoom;
    vec3 intersection = center + (uv.x * camera.right) + ( uv.y * camera.up );
    ray.direction = normalize(intersection - ray.origin);
    return ray;   
}

// Function 25
void camRay(vec2 uv, inout vec3 ray_o, inout vec3 ray_d)
{
    ray_o =
        cb(0) * vec3(0, 0, -22.5 + sin(TIME_S / 30.) * 12.) // Global z pan
      + cb(1) * vec3(
          vec2(-11., 4.) + vec2(2, 3) * mod(TIME_S, 100.) / 40.,
          -6.)                                             // Close outside dutch
      + cb(2) * vec3(-4.3, .4, -4.2 + SCENE_PROGRESS)      // Inside dutch
      + cb(3) * vec3(0, 0, -40. + TIME_S / 4.)             // Scene z pan
      + cb(12) * vec3(7.7 - SCENE_PROGRESS * 2., .75, -.7) // "Modified camera"
        ;

    uv -= .5;                                      // origin at center
    uv /= vec2(iResolution.y / iResolution.x, 1.); // fix aspect ratio
    ray_d = normalize(vec3(uv, .7));               // pull ray

    vec3 camRot =
        cb(2) * vec3(8. + SCENE_PROGRESS * 2., 1.7 + SCENE_PROGRESS, -4.)
        / 10.                          // Inside dutch
      + cb(12) * vec3(-1.2, -.52, .25) // "Modified camera"
        ;
    pR(ray_d.yz, camRot.y);
    pR(ray_d.xz, camRot.x);
    pR(ray_d.yx, camRot.z);
}

// Function 26
void getRays(inout Ray ray, out Ray r1, out Ray r2) {
     vec3 p = ray.cp.p;
    float cs = dot(ray.cp.normal, ray.rd);
    // simple approximation
    float fresnel = 1.0-abs(cs);
    vec3 normal = sign(cs)*ray.cp.normal;
    vec3 refr = refract(ray.rd, -normal, ray.eta);
    vec3 refl = reflect(ray.rd, ray.cp.normal);
    vec3 z = normal*DIST_EPSILON*2.;
    p += z;
    r1 = Ray(refr, findIntersection(p, refr),  vec3(0),1.-fresnel, 1./ray.eta);
    p -= 2.*z;
    r2 = Ray( refl, findIntersection(p, refl), vec3(0),fresnel, ray.eta);
}

// Function 27
vec3 getRayColor( sRay ray ) {
    sHit i;
    march(ray, i, MARCHSTEPS, kTt); //256

    vec3 color;
    if(i.oid.x < 0.5) {
        color = getBackground(ray.rd);
    } else  {
        sSurf s;
        s.nor  = normal(i.hp, kTt);
        sMat m = getMaterial( i );
        s.ref  = getReflection(ray, i, s);
        if(m.trs > 0.0) s.tra = getTransparency(ray, i, s, m);
        color  = setColor(ray, i, s, m);
    }

    getFog(color, ray, i); // BUG? Is this intentional that color is not updated??
    return color;
}

// Function 28
vec3 getRayDir(vec3 ro, vec3 lookAt, vec2 uv) {
	vec3 forward = normalize(lookAt - ro),
	     right = normalize(cross(vec3(0, 1, 0), forward));
	return normalize(forward + right * uv.x + cross(forward, right) * uv.y);
}

// Function 29
bool getRay(vec2 uv, out vec3 ro, out vec3 rd)
{
    mat3 cam = getCam(get(CamA));
    vec2 apert_cent = -0.*uv; 
    vec2 ap = aperture();  
    if(!(distance(ap, apert_cent) < 1.0)) return false;  
    float apd = length(ap);  
    vec3 daperture = ap.x*cam[0] + ap.y*cam[1]; 
    ro = get(CamP).xyz + aperture_size*daperture;
    float focus =2.5 + 0.8*pow(apd,5.0);
    rd = normalize(focus*(cam*vec3(FOV*uv, 1.0)) - aperture_size*daperture);
    return true;
}

// Function 30
vec3 raydir(vec2 uv, vec3 ro, vec3 lookat, float zoom) {
    vec3 forward = normalize(lookat - ro);
    vec3 temp = cross(vec3(0.0, 1.0, 0.0), forward);
    vec3 up = normalize(cross(forward, temp));
    vec3 right = cross(up, forward);
    vec3 screen_center = ro + forward * zoom;
    vec3 i = screen_center + uv.x * right + uv.y * up;
    vec3 rd = i-ro;
    return rd;

}

// Function 31
v22 getRay(vec2 u//uU is not normalized
){u=(u-iR.xy*.5)/iR.y
 ;mat4 ct=q2m(tf(camA0),tf(camP0).xyz)
 ;mat3 m=m42Rot(ct) //;mat3 m=q2m(tf(camA0))
 ;vec3 rd=normalize(m*vec3(0,0,1)   //up
                   +(m*vec3(1,0,0)*u.x//right+forward...
                   +m*vec3(0,1,0)*u.y)*pi/FieldOfView)
 ;return v22(ct[3].xyz,rd);}

// Function 32
vec3 camray(vec2 uv)
{
	uv -= vec2(0.5);
	uv.x *= (iResolution.x/iResolution.y);
	vec2 vv = uv*0.8;
	return normalize( camdir + camX*vv.x + camY*vv.y );
}

// Function 33
vec3 NetXYToRayDir(vec2 p){
    vec2 major = floor(p/1024.);
    vec2 minor = 1024.-mod(p,vec2(1024.));
    
    int face=-1;
    if(major==vec2(0,1)){         face = 0;    
    } else if(major==vec2(1,1)){  face = 2;    
    } else if(major==vec2(2,1)){  face = 3;    
    } else if(major==vec2(3,1)){  face = 5;    
    } else if(major==vec2(1,0)){  face = 4;    
    } else if(major==vec2(1,2)){  face = 1;
    }
    vec2 xy = minor - .5;
    return XYFaceToRayDir(ivec3(xy,face));
}

// Function 34
vec3 GetCameraRayDir( const in vec2 vWindow, const in vec3 vCameraPos, const in vec3 vCameraTarget )
{
	vec3 vForward = normalize(vCameraTarget - vCameraPos);
	vec3 vRight = normalize(cross(vec3(0.0, 1.0, 0.0), vForward));
	vec3 vUp = normalize(cross(vForward, vRight));
	
    const float kFOV = 1.8;
    
	vec3 vDir = normalize(vWindow.x * vRight + vWindow.y * vUp + vForward * kFOV);

	return vDir;
}

// Function 35
vec3 getRayDirection(vec2 pos, vec2 res, float fov)
{
	float fx = tan(radians(fov) * 0.5) / res.x;
	vec2 d = (2.0 * pos - res) * fx;
	return normalize(vec3(d, 1.0));
}

// Function 36
vec3 CAM_getRay(Cam cam,vec2 uv)
{
    uv = cam.lens*uv/(cam.lens-length(uv)*length(uv));
    uv *= cam.zoom;
    return normalize(uv.x*cam.R+uv.y*cam.U+cam.D);
}

// Function 37
vec3 getRay(vec2 pos)
{
    rmat = getRot(angles);
    vec2 uv = FOV*(pos - R*0.5)/R.x;
    return normalize(rmat[0]*uv.x + rmat[1]*uv.y + rmat[2]);
}

// Function 38
vec3 getRay(in vec2 st, in vec3 pos, in vec3 camTarget){
    float 	focal = 1.;
    vec3 ww = normalize( camTarget - pos);
    vec3 uu = normalize( cross(ww,vec3(0.0,1.0,0.0)) ) ;
    vec3 vv = cross(uu,ww);
	// create view ray
	return normalize( st.x*uu + st.y*vv + focal*ww );
}

// Function 39
vec3 rayDirection(vec3 cameraDir, vec2 uv){
    
    vec3 cameraPlaneU = vec3(normalize(vec2(cameraDir.y, -cameraDir.x)), 0);
    vec3 cameraPlaneV = cross(cameraPlaneU, cameraDir) ;
	return normalize(cameraDir + uv.x * cameraPlaneU + uv.y * cameraPlaneV);

}

// Function 40
vec3 GetRayColour( const in vec3 vRayOrigin, const in vec3 vRayDir, out Intersection intersection )
{
    RaymarchScene( vRayOrigin, vRayDir, intersection );        

    if ( intersection.m_objId == OBJ_ID_SKY )
    {
        return GetSkyColour( vRayDir, 1.0 );
    }
    
    Surface surface;
    GetSurfaceInfo( intersection, surface );

    vec3 vIgnore = vec3(0.0);
    vec3 vResult = vec3(0.0);
    float fSunShadow = 1.0;
    AddSunLight( surface, -vRayDir, fSunShadow, vResult, vIgnore );
    AddSkyLight( surface, vResult, vIgnore);
    return vResult * surface.m_albedo;
}

// Function 41
ray getRay(float u, float v, camera cam) { return ray(cam.o,cam.llc + u*cam.h + v*cam.v); }

// Function 42
vec3 castCamRay(vec2 xy){
    float ratio = Cam.dir.z/length(Cam.dir.xy);
    vec3 VertPerp = vec3(-Cam.dir.x*ratio, -Cam.dir.y*ratio, length(Cam.dir.xy));
    vec3 yComp = VertPerp * xy.y;
    
    vec3 HoriPerp = vec3(Cam.dir.y,-Cam.dir.x,0.0)/length(vec3(Cam.dir.y,-Cam.dir.x,0.0));
    vec3 xComp = HoriPerp * xy.x;
    
    vec3 ray = (Cam.dir*3.0)+xComp+yComp;
    vec3 rayDir = (ray/length(ray));

    vec3 P = Cam.pos;
    vec3 N = vec3(0.0,0.0,0.0);
    vec3 Collision = checkCollide(P,rayDir);
    if(Collision.x>0.0){
        if(int(Collision.y)==0){ //Sphere
            Sphere tempS = objects.spheres[int(Collision.z)];
			
            if(bounceCount>=1){
                P = P + (rayDir*Collision.x);
                N = (tempS.pos-P)/length(tempS.pos-P);
                P = P + N*(0.0001);
                vec3 tempRef = reflect(rayDir,N)/length(reflect(rayDir,N));
                return b1(P,tempRef,-N,tempS.col,tempS.gTd,tempS.sTr,tempS.fres);
            }
            else{
                return tempS.col;
            }
        }
        if(int(Collision.y)==1){ //Triangle
            Triangle tempT = objects.triangles[int(Collision.z)];

            if(bounceCount>=1){
                P = P + (rayDir*Collision.x);
                N = cross((tempT.Vert1-tempT.Vert2),(tempT.Vert1-tempT.Vert3));
                N = -N/length(N);
                P = P + N*(0.00001);
                vec3 tempRef = reflect(rayDir,N)/length(reflect(rayDir,N));
                return b1(P,tempRef,N,tempT.col,tempT.gTd,tempT.sTr,tempT.fres);
            }
            else{
                return tempT.col;
            }
        }

        if(int(Collision.y)==2){ //Light
            Light tempL = objects.lights[int(Collision.z)];
            return tempL.col;

        }

    }
    else{
		if(eviLight){
			return texture(iChannel0,vec3(rayDir.x,rayDir.z,rayDir.y)).xyz;
        }
    }
    
    
    

}

// Function 43
vec3 getRay(in vec2 st, in vec3 pos, in vec3 camTarget){
    float 	focal = .5;
    vec3 ww = normalize( camTarget - pos);
    vec3 uu = normalize( cross(ww,vec3(0.0,1.0,0.0)) ) ;
    vec3 vv = cross(uu,ww);
	return normalize( st.x*uu + st.y*vv + focal*ww );
}

// Function 44
Ray cameraGetRay(in Camera cam,in float2 uv)
{
    float3 rd = cam.lens_radius*random_in_unit_disk();
    float3 offset = cam.u * rd.x + cam.v * rd.y;
	return Ray(cam.origin + offset,cam.lower_left_corner + uv.x*cam.horizontal + uv.y*cam.vertical - cam.origin - offset);
}

// Function 45
Ray getRay(Camera cam, vec2 uv)
{
    return createRay(
        cam.origin,
        normalize(cam.lowerLeftCorner + uv.x * cam.horizontal + uv.y * cam.vertical - cam.origin));
}

// Function 46
vec3 getRayDirection(vec2 fragCoord, vec3 cameraDirection) {
    vec2 uv = fragCoord.xy / iResolution.xy;
	
    const float screenWidth = 1.0;
    float originToScreen = screenWidth / 2.0 / tan(FIELD_OF_VIEW / 2.0);
    
    vec3 screenCenter = originToScreen * cameraDirection;
    vec3 baseX = normalize(cross(screenCenter, vec3(0, -1.0, 0)));
    vec3 baseY = normalize(cross(screenCenter, baseX));
    
    return normalize(screenCenter + (uv.x - 0.5) * baseX + (uv.y - 0.5) * iResolution.y / iResolution.x * baseY);
}

// Function 47
vec3 getRayDir(vec3 ro, vec3 lookAt, vec2 uv) {
    vec3 forward = normalize(lookAt - ro),
         right = normalize(cross(vec3(-0.1, 0.9, -0.1), forward)),
         up = cross(forward, right);
    return normalize(forward + right * uv.x + up * uv.y);
}

// Function 48
vec3 rayDirection(vec2 angle, vec2 uv, vec2 renderResolution){
    vec3 cameraDir = vec3(sin(angle.y) * cos(angle.x), sin(angle.y) * sin(angle.x), cos(angle.y));
    vec3 cameraPlaneU = vec3(normalize(vec2(cameraDir.y, -cameraDir.x)), 0);
    vec3 cameraPlaneV = cross(cameraPlaneU, cameraDir) * renderResolution.y / renderResolution.x;
	return normalize(cameraDir + uv.x * cameraPlaneU + uv.y * cameraPlaneV);

}

// Function 49
Ray CameraGetRay(Camera cam, vec2 point)
{
    float halfAngle = radians(cam.fov) / 2.;
	float hSize = tan(halfAngle);
	float d = 1.0;
	float left = -hSize;
	float right = hSize;
	float top = hSize;
	float bottom = -hSize;
	
	float aspectRatio = iResolution.x / iResolution.y;
	
	float x = bottom+(top-bottom)*point.y ;
	float y = (left+(right-left)*point.x)* aspectRatio;
	vec3 dir = (d*cam.w) + x*cam.v + y*cam.u;

	Ray ray = Ray(cam.origin, normalize(dir));
    
    return ray;
}

// Function 50
void getRay()
{
    vec2 uv = (p - R*0.5)/R.x;
    ray = normalize(vec3(FOV*uv, 1.));
}

// Function 51
vec3 getRayDirection(vec2 uv)
{
    vec2 mouseUV = getMouseUV();
    rayOrigin.yz *= rotation2d(mix(-PI/2.0, PI/2.0, mouseUV.y));
    rayOrigin.xz *= rotation2d(mix(-PI, PI, mouseUV.x));
    
    vec3 cameraForward = normalize(cameraTarget - rayOrigin);
    vec3 cameraRight = normalize(cross(cameraForward, vec3(0.0, 1.0, 0.0)));
    vec3 cameraUp = normalize(cross(cameraRight, cameraForward));
    
    vec3 rayDirection = normalize(uv.x * cameraRight + uv.y * cameraUp + cameraForward);
    return rayDirection;    
}

// Function 52
rayMarchHit GetRayMarchHit(vec3 position, float time) {
    rayMarchHit hit;
    #if defined(DEBUG_USE_SQUARE_DISTANCE)
    hit.distance = RAY_MAX_DISTANCE*RAY_MAX_DISTANCE;
    #else
    hit.distance = RAY_MAX_DISTANCE;
    #endif

    float newDistance = hit.distance;

    vec3 pBlockCenter = position + (0.5*blockSize);
    // position of the cube (-0.5*blockSize to +0.5*blockSize) > 0.0 to 1.0
    vec3 pBlockI = floor(pBlockCenter/blockSize);
    // fraction of distance withinteh grid, -0.5 to + 0.5
    vec3 pBlockF = ((pBlockCenter/blockSize)-pBlockI)-0.5;
    //wold position of the center of the grid, positionGI*blockSize
    vec3 pBlockWI = pBlockI*blockSize;
    // -0.5*blockSize to +0.5*blockSize
    vec3 pBlockWF = pBlockF*blockSize; 

    vec3 pCellWF = mod(pBlockWF+(gridSizeH),gridSize)-gridSizeH;

    //on north/south - -2.5 < mod(x,blockSize) < 2.5
    bool onNorthSouth = (pBlockWF.x >= -gridSizeH && pBlockWF.x <= gridSizeH && (pBlockWF.z < -gridSizeH || pBlockWF.z > gridSizeH) );
    bool onEastWest = (pBlockWF.z >= -gridSizeH && pBlockWF.z <= gridSizeH && (pBlockWF.x < -gridSizeH || pBlockWF.x > gridSizeH) );

    #if defined(DEBUG_RENDER_RAILS)
    //Road Main Beams
    //FIXME: the length of beams *0.8?!?!
    newDistance = min(newDistance,sdXAlignedCylinder(abs(pBlockWF)-vec3(2.0,1.0, roadLengthQ - gridSize*0.8 ), roadLength , 0.05 ));
    newDistance = min(newDistance,sdXAlignedCylinder(abs(pBlockWF.zyx)-vec3(2.0,1.0, roadLengthQ - gridSize*0.8 ), roadLength , 0.05 ));

    //Intersection Main Beams
    newDistance = min(newDistance,sdCircle( abs(pBlockWF.xz) - vec2(2.0,2.2), 0.05 ));
    newDistance = min(newDistance,sdCircle( abs(pBlockWF.zx) - vec2(2.0,2.2), 0.05 ));

    //dont crossbrase the road
    if (pBlockWF.y < -gridSizeH || pBlockWF.y > gridSizeH) {
        newDistance = min(newDistance,sdCapsule( abs(vec3(pBlockWF.x, pCellWF.y, pBlockWF.z)) , vec3(2.0,1.5,2.2), vec3(0.0,0.0,2.2),  0.03));
        newDistance = min(newDistance,sdCapsule( abs(vec3(pBlockWF.z, pCellWF.y, pBlockWF.x)) , vec3(2.0,1.5,2.2), vec3(0.0,0.0,2.2),  0.03));
    }

    newDistance = min(newDistance,sdCapsule( abs(vec3(pBlockWF.x, pCellWF.y, pBlockWF.z)) , vec3(0.0,2.5,2.2), vec3(2.0,1.5,2.2), 0.03));
    newDistance = min(newDistance,sdCapsule( abs(vec3(pBlockWF.z, pCellWF.y, pBlockWF.x)) , vec3(0.0,2.5,2.2), vec3(2.0,1.5,2.2), 0.03));

    if (onNorthSouth) {
        newDistance = min(newDistance,sdCapsule( abs(vec3(pCellWF.x, pBlockWF.y, pCellWF.z)) , vec3(2.0,1.0,2.5), vec3(2.0,0.0,0), 0.03));
        newDistance = min(newDistance,sdCapsule( abs(vec3(pCellWF.x, pBlockWF.y, pCellWF.z)) , vec3(2.0,1.0,2.5), vec3(0.0,1.0,0), 0.03));
    }

    if (onEastWest) {
        newDistance = min(newDistance,sdCapsule( abs(vec3(pCellWF.z, pBlockWF.y, pCellWF.x)) , vec3(2.0,1.0,2.5), vec3(2.0,0.0,0), 0.03));
        newDistance = min(newDistance,sdCapsule( abs(vec3(pCellWF.z, pBlockWF.y, pCellWF.x)) , vec3(2.0,1.0,2.5), vec3(0.0,1.0,0), 0.03));
    }

    if (newDistance < hit.distance) {
        hit.origin = position;
        hit.distance = newDistance;
        hit.materialId = MAT_RAIL;
        hit.neon = vec3(0.0);
    }
    #endif

    // Lights
    newDistance = min(newDistance,sdCapsule( abs(vec3(pBlockWF.x, pCellWF.y, pBlockWF.z)) , vec3(2.2,1.25,2.0), vec3(2.0,1.25,2.2), 0.1));

    if (onNorthSouth) {
        newDistance = min(newDistance,sdXAlignedCylinder(abs(vec3(pCellWF.x, pBlockWF.y, pCellWF.z)) -vec3(2.0,1.0,0.0), 0.4, 0.1 ));
    }

    if (onEastWest) {
        newDistance = min(newDistance,sdXAlignedCylinder(abs(vec3(pCellWF.z, pBlockWF.y, pCellWF.x)) -vec3(2.0,1.0,0.0), 0.4, 0.1 ));
    }

    if (newDistance < hit.distance) {
        hit.origin = position;
        hit.distance = newDistance;
        hit.materialId = MAT_NEON;
        hit.neon = roadLightColour*4.0;
    }

    //Buildings.
    //rework the grids so 0,0,0 is the center of the buildings
    pBlockCenter = position;// + vec3(0.0,blockSize*0.5,0.0);

    // position of the cube (-0.5*blockSize to +0.5*blockSize) > 0.0 to 1.0
    pBlockI = floor(pBlockCenter/blockSize);
    // fraction of distance withinteh grid, -0.5 to + 0.5
    pBlockF = ((pBlockCenter/blockSize)-pBlockI)-0.5;

    //wold position of the center of the grid, positionGI*blockSize
    pBlockWI = pBlockI*blockSize;
    // -0.5*blockSize to +0.5*blockSize
    pBlockWF = pBlockF*blockSize; 

    float floorCount=1.0;
    
    float blockHash = hash21(pBlockI.xz)*20.0;
    float buildingFloor = floor(position.y)+floor(blockHash);;
    float round = clamp(fract(blockHash)-0.5,0.05,0.5);
    
    for(float i=-floorCount;i<=floorCount;i++) {
		//pick a random point where the width and width-round are both within a single world space unit,
        //the texturing later snaps to the world grid and if the ends of hte window cross the boundy we get artifacts.
        float windowWigle=valueNoise1du(buildingFloor + i + blockHash)*2.0;
        float baseWidth=floor((roadLength-2.0-windowWigle)*0.5);
        float buildingfloorSize=baseWidth-map(fract(windowWigle)*(1.0-round*2.0), 0.0, 1.0, 0.01, 0.5)-round;
        newDistance = min(newDistance, sdBox(vec3(pBlockWF.x,fract(pBlockWF.y)-i,pBlockWF.z), vec3(buildingfloorSize,0.5-round,buildingfloorSize))-round);
    }

    if (newDistance < hit.distance) {
        hit.origin = position;
        hit.distance = newDistance;
        hit.materialId = MAT_WALL;
        hit.neon = vec3(0.0);
    }

    //Corners
    newDistance = min(newDistance, sdBox( abs(pBlockWF.xz)-(roadLength*0.5)+(gridSize*0.4), vec2(0.5) ) - 0.5 );

    //Mid Beams
    if (abs(pBlockWF.z) < (roadLength*0.5)) {
        newDistance = min(newDistance, sdBox( vec2(abs(pBlockWF.x)-(roadLength*0.5)+(gridSize*0.2),pCellWF.z), vec2(0.375) ) - 0.125);
    }
    if (abs(pBlockWF.x) < (roadLength*0.5)) {
        newDistance = min(newDistance, sdBox( vec2(abs(pBlockWF.z)-(roadLength*0.5)+(gridSize*0.2),pCellWF.x), vec2(0.375) ) - 0.125);
    }
    if (newDistance < hit.distance) {
        hit.origin = position;
        hit.distance = newDistance;
        hit.materialId = MAT_SOLID;
        hit.neon = vec3(0.0);
    }

    return hit;
}

// Function 53
vec3 getRayDir(vec3 ro, vec3 lookAt, vec2 uv) {
    vec3 forward = normalize(lookAt - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    return normalize(forward + right * uv.x + up * uv.y);
}

// Function 54
Ray GetRayFromCamera(Camera cam)
{
    const vec3 up = vec3(0,1,0);
    
    vec3 dirF = normalize(cam.dir);
    vec3 dirT = -normalize(cross(dirF, up));
    vec3 dirU = normalize(cross(dirF, dirT));
	
	vec2 jitteredUV = fragUV.xy + vec2(EDGE_SIZE, cam.aspect * EDGE_SIZE) * 2.0 * vec2(Rand() - 0.5, Rand() - 0.5);
	
	vec3 sensor = GetSensorPoint(cam, jitteredUV);	
	vec3 focalPoint = normalize(sensor) * cam.focusDist;
	
	vec3 lensPoint = GetAperturePoint(cam);
	vec3 refractLocalRay = normalize(focalPoint - lensPoint);
	
	vec3 rayPos = cam.pos + (dirT * lensPoint.x + dirU * lensPoint.y + dirF * lensPoint.z);
	vec3 rayDir = normalize(dirT * refractLocalRay.x + dirU * refractLocalRay.y + dirF * refractLocalRay.z);
	
    return Ray(rayPos, rayDir, 0, vec3(1,1,1), 1.0);
}

// Function 55
vec3 getRayDir(vec2 fragCoord)
{
  vec2 uv = (fragCoord.xy / iResolution.xy )*2.0 - 1.0;
  uv.x *= iResolution.x/iResolution.y;                   
  return normalize(uv.x * cam.right + uv.y * cam.up + cam.forward);
}

// Function 56
void getRays(inout Ray ray, out Ray r1, out Ray r2) {
     vec3 p = ray.cp.p;
    float cs = dot(ray.cp.normal, ray.rd);
    // simple approximation
    float fresnel = 1.0-abs(cs);
//	fresnel = mix(0.1, 1., 1.0-abs(cs));
    float r = ray.cp.mat - ID_FLOOR;
     vec3 normal = sign(cs)*ray.cp.normal;
    vec3 refr = refract(ray.rd, -normal, ray.eta);
    vec3 refl = reflect(ray.rd, ray.cp.normal);
    vec3 z = normal*DIST_EPSILON*2.;
    p += z;
    r1 = Ray(refr, findIntersection(p, refr),  vec3(0),(1.-fresnel)*r, 1./ray.eta);
    p -= 2.*z;
    r2 = Ray( refl, findIntersection(p, refl), vec3(0),r*fresnel, ray.eta);
}

// Function 57
float GetRayFirstStep( const in C_Ray ray )
{
    return ray.fStartDistance;  
}

// Function 58
void getRay(out vec3 ro, out vec3 rd, vec2 pos, ivec2 p)
{
    vec2 angles = texelFetch(iChannel3,  ivec2(ANGLE_INDX,0), 0).xy;
 	mat3 camera = getCamera(angles);
    int kk = int(iMouse.z);
  	//dither position on the aperture
    vec4 blue = texture(iChannel2, vec2(p)/1024. + PI*iTime);
    int I = int(blue.x * float(Nsamp));
    
    pos += (blue.yz - 0.5)/iResolution.x;
    vec2 delta = 2.*aperture*fcircle(I, Nsamp);
    vec3 ro0 = vec3(delta.x, 0., delta.y);
    vec3 rd0 = focal_plane*vec3(FOV*pos.x, 1, FOV*pos.y) - ro0;
    
    ro = texelFetch(iChannel3,  ivec2(POS_INDX,0), 0).xyz + transpose(camera)*ro0;
    rd = normalize(transpose(camera)*rd0);
}

// Function 59
vec3 GetCameraRayDir( const in vec2 vWindow, const in mat3 mCameraRot )
{
	vec3 vDir = normalize( CameraToWorld(vec3(vWindow.x, vWindow.y, 2.0), mCameraRot) );

	return vDir;
}

// Function 60
vec3 getRays(vec2 coord, out float mask){
    
    const int numSteps = int(Num_Steps * Density);
    int weight = 0;
    
    vec3 gr = vec3(0.0);
    
    vec2 mouseCoord = coord.xy - (iMouse.xy / iResolution.xy);
    
    vec2 delta = (mouseCoord) * (Density / float(numSteps));
    vec2 customTexcoord = coord.st;
    
    float visibility = 1.0-sqrt(pow(mouseCoord.x,2.0) + pow(mouseCoord.y,2.0));
    visibility = pow(visibility,1.0);
    visibility += pow(visibility,2.0);
    visibility = clamp(visibility,0.0,1.0);
    visibility = 1.0;
	
    for(int i = 0; i < numSteps; i++){
        
        customTexcoord -= delta;
    	float noise = getNoiseTexture(customTexcoord);
        
        gr += getTexure(customTexcoord.st + delta * noise);
        
        weight++;
    }
    
    gr /= float(weight);
    gr *= visibility;
    mask = (gr.r + gr.g + gr.b) / 3.0;
    mask = pow(mask,2.2);
    gr *= mask;
    gr = max(gr,0.0);
    
    return gr;
    
}

// Function 61
vec3 getRay_9_4(mat3 camMat, vec2 screenPos, float lensLength) {
  return normalize(camMat * vec3(screenPos, lensLength));
}

// Function 62
vec3 getCameraRayDir(vec2 uv, vec3 camPos, vec3 camTarget)
{
    // Calculate camera's "orthonormal basis", i.e. its transform matrix components
    vec3 camForward = normalize(camTarget - camPos);
    vec3 camRight = normalize(cross(vec3(0.0, 1.0, 0.0), camForward));
    vec3 camUp = normalize(cross(camForward, camRight));
     
    float fPersp = 2.0;
    vec3 vDir = normalize(uv.x * camRight + uv.y * camUp + camForward * fPersp);
 
    return vDir;
}

// Function 63
Ray Camera_getRay(Camera cam, vec2 uv) {
    return Ray(cam.origin, 
               normalize(cam.ll + uv.x * cam.width * cam.u + uv.y * cam.height * cam.v - cam.origin));
}

// Function 64
vec3 CAM_getRay(Cam cam,vec2 uv)
{
    uv *= 2.0*iResolution.x/iResolution.y;;
    return normalize(uv.x*cam.R+uv.y*cam.U+cam.D*2.5);
}

// Function 65
vec3 GetRayColour( const in vec3 vRayOrigin, const in vec3 vRayDir, out Intersection intersection )
{
    RaymarchScene( vRayOrigin, vRayDir, intersection );        

    if ( intersection.m_objId == OBJ_ID_SKY )
    {
        return GetSkyColour( vRayDir );
    }
    
    Surface surface;
    GetSurfaceInfo( intersection, surface );

    vec3 vIgnore = vec3(0.0);
    vec3 vResult = vec3(0.0);
    float fSunShadow = 1.0;
    AddSunLight( surface, -vRayDir, fSunShadow, vResult, vIgnore );
    AddSkyLight( surface, vResult, vIgnore);
    return vResult * surface.m_albedo;
}

// Function 66
vec3 rayDir(vec3 camFwd, float fov, vec2 uv)
{
    // In what direction to shoot?
    vec3 camUp = vec3(0.,0.,1.);
    camUp = normalize(camUp - camFwd*dot(camFwd, camUp)); // Orthonormalize
    vec3 camRight = cross(camFwd, camUp);
    return normalize(camFwd + (uv.x * camRight + uv.y * camUp)*fov);
}

// Function 67
vec3 GetRayDir(vec2 uv) {
	vec3 up = vec3(0., 1., 0.);
    vec3 vdir = normalize(at - eye);
    vec3 xdir = normalize(cross(vdir, up));
    up = normalize(cross(xdir, vdir));
    vec3 center = eye + vdir * near;
    vec2 zuv = (uv - vec2(.5)) * iResolution.xy * zoom;
    vec3 pix = center + zuv.x * xdir + zuv.y * up;
    return pix - eye;
}

// Function 68
vec3 CreateRayDir(const vec2 vFragCoord, const vec2 aCamRot)
{  
    float vAspectRatio = iResolution.x / iResolution.y;
    float vTan = tan(0.5 * radians(FOV));
    vec3 vRD = vec3(
        (2.0 * ((vFragCoord.x+0.5) / iResolution.x) - 1.0) * vTan * vAspectRatio,
        (1.0 - 2.0 * ((vFragCoord.y+0.5) / iResolution.y)) * vTan,
        -1.0);
    vRD = normalize(vRD);
   	mat3 vMatView = CreateCameraRotationMatrix(aCamRot);
    return vMatView * vRD;
}

// Function 69
ivec3 RayDirToXYFace(vec3 dir){
    
    if        (dir.x>max(abs(dir.y),abs(dir.z))){
        dir /= dir.x/512.;
        return ivec3(-dir.z+512.0, -dir.y+512., 0);
    } else if (dir.y>max(abs(dir.z),abs(dir.x))){
        dir /= dir.y/512.;
        return ivec3(dir.x+512., dir.z+512.,1);
    } else if (dir.z>max(abs(dir.x),abs(dir.y))){
        dir /= dir.z/512.;
        return ivec3(dir.x+512.,-dir.y+512.,2);
    } else if (-dir.x>max(abs(dir.y),abs(dir.z))){
        dir /=-dir.x/512.;
        return ivec3(dir.z+512.,-dir.y+512.,3);
    } else if (-dir.y>max(abs(dir.z),abs(dir.x))){
        dir /=-dir.y/512.;
        return ivec3(dir.x+512.,-dir.z+512.,4);
    } else if (-dir.z>max(abs(dir.x),abs(dir.y))){
        dir /=-dir.z/512.;
        return ivec3(-dir.x+512.,-dir.y+512.,5);
    } else return ivec3(0,0,-1);
}

// Function 70
vec3 GetRayDirection(float fov, vec2 res, vec2 fragCoord) {
    vec2 xy = fragCoord - res/2.;
    float z = res.y / tan(radians(fov/2.));
    return normalize(vec3(xy, -z));
}

// Function 71
ray_t camera_getRay( camera_t c, vec2 uv )
{
    ray_t ray;
    ray.o = c.pos;
    
    // Rotate camera according to mouse position
    float ca = cos(mouse.x), sa = sin(mouse.x);
    mat3 rotX = mat3(ca, 0.0, sa, 0.0, 1.0, 0.0, -sa, 0.0, ca);
    ca = cos(mouse.y), sa = sin(mouse.y);
    mat3 rotY = mat3(1.0, 0.0, 0.0, 0.0, ca, -sa, 0.0, sa, ca);
    mat3 rotM = rotX * rotY;
    
	ray.o = rotM*c.pos;
    ray.d = rotM*normalize( vec3( uv, -1.0 ) ); // should be -1! facing into scene
    
	return ray;
}

// Function 72
ray_ getRay(vec2 pixel, camera_ camera)
{
	vec3 right = cross(camera.front, camera.up);
	vec3 up = cross(right, camera.front);
	float fovScale = tan(camera.fov * 0.5 * 3.1415926535 / 180.0) * 2.0;
	vec2 point = pixel;
	vec3 r = right * point.x * fovScale;
	vec3 u = up * point.y * fovScale;
	return ray_(camera.eye, normalize(camera.front + r + u));
}

// Function 73
vec3 getRayColor(vec3 origin, vec3 ray, out vec3 p, out vec3 normal, out bool hit)
{
    vec3 col = vec3(0.0);
    
    float dist, depth;
    depth = 0.0;

    const int maxsteps = 128;

    for (int i = 0; i < maxsteps; i++)
    {
        p = origin + ray * depth;

        dist = distFunc(p);

        if (dist < EPS)
        {
            break;
        }

        depth += dist;
    }

    float shadow = 1.0;
    
    if (dist < EPS)
    {
        float ao = 1.0;
        normal = getNormal(p);

        float diff = dot(normal, normalize(lightDir));
        float spec = pow(clamp(dot(reflect(lightDir, normal), ray), 0.0, 1.0), 10.0);

        shadow = genShadow(p + normal * EPS, lightDir);

        ao = AmbientOcclusion(p + normal * EPS, normal);

        float u = 1.0 - floor(mod(p.x, 2.0));
        float v = 1.0 - floor(mod(p.z, 2.0));
        
        const float width = 0.2;
        
        if ((u + width >= 1.0) || (v + width >= 1.0))
        {
            diff *= 0.5;
        }
        else
        {
            diff *= 0.7;
        }
        //if ((u == 1.0 && v < 1.0) || (v == 1.0 && u < 1.0))
        //{
        //    diff *= 0.7;
        //}

        col = sceneColor(p).rgb * vec3(diff) + vec3(0.5) * spec;
        col *= max(0.7, ao);
        col *= max(0.5, shadow);
        
        hit = true;
    }
    else
    {
        vec4 cube = texture(iChannel0, ray);
    	col = cube.xyz;
        hit = false;
    }
    
    return col;
}

// Function 74
vec3 RayDirection(float fov, vec2 size, vec2 fragCoord)
{
    vec2 xy = fragCoord - (size / 2.0);
    float z = size.y / tan(radians(fov) / 2.0);
    return normalize(vec3(xy, -z));
}

// Function 75
vec3 getRay(vec2 angles, vec2 pos)
{
    mat3 camera = getCamera(angles);
    return normalize(transpose(camera)*vec3(FOV*pos.x, 1., FOV*pos.y));
}

// Function 76
Ray Camera_GetRay(in Camera camera, vec2 uv) 
{ 
    uv    = (uv * 2.0) - 1.0; 
    uv.x *= (iResolution.x / iResolution.y); 
    
    Ray ray; 
    ray.o = camera.origin; 
    ray.d = normalize((uv.x * camera.right) + (uv.y * camera.up) + (camera.forward * 2.5)); 
    
    return ray; 
}

// Function 77
vec3 rayDirection(vec2 angle, vec2 uv, vec2 renderResolution){
    vec3 cameraDir = vec3(sin(angle.y) * cos(angle.x), sin(angle.y) * sin(angle.x), cos(angle.y));
    vec3 cameraPlaneU = vec3(normalize(vec2(cameraDir.y, -cameraDir.x)), 0);
    vec3 cameraPlaneV = cross(cameraPlaneU, cameraDir) * renderResolution.y / renderResolution.x;
    float fish = 1.0 - dot(uv, uv) * FISHEYE;
	return normalize(cameraDir*fish + uv.x * cameraPlaneU + uv.y * cameraPlaneV);

}

// Function 78
void getRay(in vec2 fragCoord, out vec3 ro, out vec3 rd, out float fade)
{
    // misc
    float r = EARTH_R;
    vec3 top = vec3(0, r * 1.0001f, 0);

    // time / segment
    const float SEGMENT_DURATION = 5.5;
    float t = iTime;
    float segment = trunc(t / SEGMENT_DURATION);
    float segT = mod(t, SEGMENT_DURATION) / SEGMENT_DURATION;
    
    // fade
    float fadePercent = 0.12;
    fade = smoothstep(0.0, fadePercent, segT) * (1.0 - smoothstep(1.0 - fadePercent, 1.0, segT));
    if (t/SEGMENT_DURATION<0.5)
        fade = 1.0;
    
    //     
    float s = (segment+10.0)/202.0;
    float rand1 = clamp(texture(iChannel0, vec2(s, s)).x, 0.0, 1.0);
    float rand2 = clamp(texture(iChannel0, vec2(s, rand1)).x, 0.0, 1.0);
    float rand3 = clamp(texture(iChannel0, vec2(rand1+s, rand2)).x, 0.0, 1.0);
    
    // source
    float thetaCoef = rand1;
    float thetaSign = sign(rand2-0.5);
    float theta = -PI*0.5 + thetaSign * mix(0.4, 1.1, thetaCoef);
    float phi = PI*0.5 + PI*0.27 * mix(-1.0, 1.0, rand3); 
    
    float maxH = mix(r*6.0, r*3.0, thetaCoef); 
    float h = mix(r*1.1, maxH, rand2);
    
    vec3 start = vec3(theta, phi, h);    
    
    vec3 end = start;
    end.x += thetaSign * PI * 0.35 / (1.0+thetaCoef);
    
    float maxZChange = r*2.5;
    float minZ = r * 1.0001f;
    float tgtZ = max(start.z * 0.3, minZ);
    float change = min(start.z - tgtZ, maxZChange);
    end.z = start.z - change;
    
    vec3 cur = mix(start, end, segT);       
    ro = sph(cur.x, cur.y, cur.z);
    
    // target
    vec3 dirTop = normalize(top - ro);
    vec3 dirSun = sunDir();
    float coef = length(ro - top) / r;
    vec3 dir = mix(dirSun, dirTop, 0.4);
    vec3 tgt = ro + dir * r;
    
    // final ray
    mat3 cam = setCamera(ro, tgt, 0.0);    
	vec2 p = (-iResolution.xy + 2.0*fragCoord) / iResolution.y;
    rd = cam * normalize(vec3(p.xy, FOCAL));        
}

// Function 79
vec2 getScreenspaceUvFromRayDirectionWS(
    vec3 rayDirectionWS,
	vec3 cameraForwardWS,
	vec3 cameraUpWS,
	vec3 cameraRightWS,
	float aspectRatio)
{
    vec3 eyeToCameraPlaneCenterWS = cameraForwardWS * kCameraPlaneDist;
    // project rayDirectionWs onto camera forward
    float projDist                 = dot(rayDirectionWS, cameraForwardWS);
    vec3  eyeToPosOnCameraPlaneWS = (rayDirectionWS / projDist) * kCameraPlaneDist;
    vec3  vecFromPlaneCenterWS       = eyeToPosOnCameraPlaneWS - eyeToCameraPlaneCenterWS;

    float xDist = dot(vecFromPlaneCenterWS, cameraRightWS);
    float yDist = dot(vecFromPlaneCenterWS, cameraUpWS);
    
    xDist /= aspectRatio;
    xDist = xDist * 0.5 + 0.5;
    yDist = yDist * 0.5 + 0.5;

    return vec2(xDist, yDist);
}

// Function 80
vec3 GetCameraRayDir(const in vec2 mUV, const in vec3 camPosition, const in vec3 camTarget)
{
	vec3 forwardVector = normalize(camTarget - camPosition);
	vec3 rightVector = normalize(cross(vec3(0.0, 1.0, 0.0), forwardVector));
	vec3 upVector = normalize(cross(forwardVector, rightVector));	
    
	vec3 camDirection = normalize(mUV.x * rightVector * FOV
                                  + mUV.y * upVector * FOV
                                  + forwardVector);

    
	sunPos = vec2(dot(sunDirection, rightVector), 
                  dot(sunDirection, upVector));
    
    #ifndef DEBUG_NO_RAIN
    	//Raining
    	#ifndef DEBUG_NO_WATERDROPLET
			float t = floor(mUV.x * MAX_DROPLET_NMBR);
            float r = Hash1D(t);
    
            //used for radius of droplet, smaller float == bigger drop
            float fRadiusSeed = fract(r * 40.0);
            float radius = fRadiusSeed * fRadiusSeed * 0.02 + 0.001;
    
            float fYpos = r * r - clamp(mod(iTime * radius * 2.0, 10.2) - 0.2, 0.0, 1.0);
            radius *= rainingValue;
            vec2 vPos = vec2((t + 0.5) * (1.0 / MAX_DROPLET_NMBR), fYpos * 3.0 - 1.0);
            vec2 vDelta = mUV - vPos;

            const float fInvMaxRadius = 1.0 / (0.02 + 0.001);
            vDelta.x /= (vDelta.y * fInvMaxRadius) * -0.15 + 1.85; // big droplets tear shaped

            vec2 vDeltaNorm = normalize(vDelta);
            float l = length(vDelta);
            if(l < radius)
            {		
                l = l / radius;

                float lz = sqrt(abs(1.0 - l * l));
                vec3 vNormal = l * vDeltaNorm.x * rightVector 
                               + l* vDeltaNorm.y * upVector 
                               - lz * forwardVector;
                vNormal = normalize(vNormal);
                camDirection = refract(camDirection, vNormal, 0.7);
            }
        #endif
    #endif
	return camDirection;
}

// Function 81
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

// Function 82
vec3 getRaysSecondPass(vec2 coord, out float mask){
    
    const int numSteps = int(Num_Steps * Density);
    int weight = 0;
    
    vec3 gr = vec3(0.0);
    
    vec2 mouseCoord = coord.xy - (iMouse.xy / iResolution.xy);
    
    vec2 delta = (mouseCoord) * (Density / float(numSteps));
    vec2 customTexcoord = coord.st;
    
    float visibility = 1.0-sqrt(pow(mouseCoord.x,2.0) + pow(mouseCoord.y,2.0));
    visibility = pow(visibility,1.0);
    visibility += pow(visibility,2.0);
    visibility = clamp(visibility,0.0,1.0);
    visibility = 1.0;
	
    for(int i = 0; i < numSteps; i++){
        
        customTexcoord -= delta;
    	float noise = getNoiseTexture(customTexcoord);
        
        gr += getNoiseTexture(customTexcoord.st + delta * noise);
        
        weight++;
    }
    
    gr /= float(weight);
    gr *= visibility;
    mask = (gr.r + gr.g + gr.b) / 3.0;
    mask = pow(mask,2.2);
    gr *= mask;
    gr = max(gr,0.0);
    
    return gr;
    
}

// Function 83
vec3 getRay(vec2 uv){
   uv = (uv * 2.0 - 1.0)* vec2(resolution.x / resolution.y, 1.0);
	vec3 proj = normalize(vec3(uv.x, uv.y, 1.5));	
    if(resolution.y < 400.0) return proj;
	vec3 ray = rotmat(vec3(0.0, -1.0, 0.0), 3.0 * (mouse.x * 2.0 - 1.0)) * rotmat(vec3(1.0, 0.0, 0.0), 1.5 * (mouse.y * 2.0 - 1.0)) * proj;
    return ray;
}

// Function 84
vec3 GetRayDir(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l - p),
        r = normalize(cross(vec3(0., 1., 0.), f)),
        u = cross(f, r),
        c = f * z,
        i = c + uv.x * r + uv.y * u,
        d = normalize(i);
    return d;
}

// Function 85
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.;
    float z = size.y / tan(radians(fieldOfView) / 2.);
    return normalize(vec3(xy, -z));
}

// Function 86
vec3 rayDirection(vec2 fragCoord){
    float zoom=1.2;
    vec2 offset=fragCoord/iResolution.xy-0.5;
    offset.y/=iResolution.x/iResolution.y;
    vec3 rawDir=vec3(offset/zoom,-1.);
    vec2 mouseAngles=iMouse.xy/iResolution.xy-0.5;
    vec4 qUpDown=quaternion(vec3(1,0,0),radians(45.+mouseAngles.y*30.0));
    vec4 qLeftRight=quaternion(vec3(0,0,1),radians(-45.-mouseAngles.x*30.0));
    vec4 q=quaternionMultiply(qLeftRight,qUpDown);
    return normalize(rotation(q,rawDir));
}

// Function 87
vec3 GetRayDir(vec2 uv, vec3 ro) {
	vec3 f = normalize(vec3(0)-ro),
        r = normalize(cross(vec3(0,1,0), f)),
        u = cross(f, r),
        c = ro + f,
        i = c + uv.x*r + uv.y*u,
        rd = normalize(i-ro);
    return rd;
}

// Function 88
Ray Camera_GetRay(in Camera camera, vec2 uv)
{
    Ray ray;
    
    uv    = (uv * 2.0) - 1.0;
    uv.x *= (iResolution.x / iResolution.y);
    
    ray.origin    = camera.origin;
    ray.direction = normalize((uv.x * camera.right) + (uv.y * camera.up) + (camera.forward * 2.5));

    return ray;
}

// Function 89
vec3 GetCameraRayDir(vec2 vWindow, vec3 vCameraDir, float fov)
{
	vec3 vForward = normalize(vCameraDir);
	vec3 vRight = normalize(cross(vec3(0.0, 1.0, 0.0), vForward));
	vec3 vUp = normalize(cross(vForward, vRight));
    
	vec3 vDir = normalize(vWindow.x * vRight + vWindow.y * vUp + vForward * fov);

	return vDir;
}

// Function 90
vec3 rayDirection(float fov, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fov) / 2.0);
    return normalize(vec3(xy, -z));
}

// Function 91
vec2 RayDirToNetXY(vec3 rayDir){
    ivec3 XYFace = RayDirToXYFace(rayDir);
    if(XYFace.z==0){         return 1024.-vec2(XYFace.xy) + vec2(0,1)*1024.;
    } else if(XYFace.z==1){  return 1024.-vec2(XYFace.xy) + vec2(1,2)*1024.;
    } else if(XYFace.z==2){  return 1024.-vec2(XYFace.xy) + vec2(1,1)*1024.;
    } else if(XYFace.z==3){  return 1024.-vec2(XYFace.xy) + vec2(2,1)*1024.;
    } else if(XYFace.z==4){  return 1024.-vec2(XYFace.xy) + vec2(1,0)*1024.;
    } else if(XYFace.z==5){  return 1024.-vec2(XYFace.xy) + vec2(3,1)*1024.;
    } else return vec2(0);
}

// Function 92
vec3 RayDir(float fovRads, vec2 viewSizes, vec2 pixID)
{
    vec2 xy = pixID - (viewSizes / 2.0);
    float z = viewSizes.y / tan(fovRads / 2.0);
    return normalize(vec3(xy, z));
}

// Function 93
vec3 getRayDirByCoord(vec2 coord){
	vec3 pointV3 = getPointV3ByFragCoord(coord);
    vec3 ray = pointV3 - cameraPosition;
    return normalize(ray);
}

// Function 94
vec3 getCameraRayDir(vec2 uv, vec3 camPos, vec3 camTarget)
{
    vec3 camForward = normalize(camTarget - camPos);
    vec3 camRight = normalize(cross(vec3(0.,1.,0.), camForward));
    vec3 camUp = normalize(cross(camForward, camRight));
    return normalize(uv.x * camRight + uv.y * camUp + camForward * 2.0);
}

// Function 95
vec3 rayDirection(float fov, vec2 size, vec2 fragCoord) {
  vec2 xy = fragCoord - size * 0.5;
  float z = size.y / tan(radians(fov) * 0.5);
  return normalize(vec3(xy, -z));
}

// Function 96
float getRayleighPhase( float fCos2 ) {
    return 0.75 * ( 2.0 + 0.5 * fCos2 );
}

// Function 97
vec3 getRay_7_2(mat3 camMat, vec2 screenPos, float lensLength) {
  return normalize(camMat * vec3(screenPos, lensLength));
}

// Function 98
vec3 rayDirection(float fov, vec2 size, vec2 fragCoords) {
    // Center coordinate system 
    // Define x and y coordinates 
    vec2 xy = fragCoords - size / 2.0;
    // Find z
    float z = size.y / tan(radians(fov) / 2.0);
    // Return normalized direction to march 
    return normalize(vec3(xy, -z));
}

// Function 99
vec3 XYFaceToRayDir(ivec3 p){
    vec2 x = vec2(p-512) + 0.5;
           if (p.z==0){     return vec3( 512,-x.y,-x.x);
    } else if (p.z==1){     return vec3( x.x, 512, x.y);
    } else if (p.z==2){     return vec3( x.x,-x.y, 512);
    } else if (p.z==3){     return vec3(-512,-x.y, x.x);
    } else if (p.z==4){     return vec3( x.x,-512,-x.y);
    } else if (p.z==5){     return vec3(-x.x,-x.y,-512);
    } else return vec3(0);
}

// Function 100
vec3 getRay(in vec2 st, in vec3 pos, in vec3 camTarget){
    float 	focal = 1.;
    vec3 ww = normalize( camTarget - pos );
    vec3 uu = normalize( cross(ww,vec3(0.0,1.0,0.0)) ) ;
    vec3 vv = cross(uu,ww);
	// create view ray
	return normalize( st.x*uu + st.y*vv + focal*ww );
}

// Function 101
vec3 rayDirection(float fieldOfView, vec2 fragCoord) {
    vec2 xy = fragCoord - iResolution.xy / 2.0;
    float z = iResolution.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

// Function 102
vec3 CAM_getRay(Cam cam,vec2 uv)
{
    uv *= CAM_FOV;
    return normalize(uv.x*cam.R+uv.y*cam.U+cam.D);
}

// Function 103
vec3 RayDir(vec2 pixPos,
            vec2 viewSizes)
{
    vec3 dir = vec3(pixPos - (viewSizes / 2.0),
                	viewSizes.y / tan(1.62 / 2.0));
    return normalize(dir);
}

// Function 104
vec3 cubemapRayDir(in vec2 fragCoord, vec2 bufferSize) 
{     
    bufferSize.y = min(bufferSize.y, bufferSize.x*0.66667 + 4.0);
    
    float ts = (bufferSize.y - 2.0) * 0.5;
    
    fragCoord = min(fragCoord, 
                    vec2(ts*3.0 - 1.0, 2.0*ts + 1.0));
    
    vec2 tc = vec2(fragCoord.x / ts, 
                   fragCoord.y*2.0 / bufferSize.y); 
    
    vec2 ti = floor(tc) - vec2(1.0, 0.0);
    vec3 n = -vec3((1.0 - abs(ti.x))*(ti.y*2.0 - 1.0), 
                   ti.x*ti.y, ti.x*(1.0 - ti.y));

    float bpy = min(0.9999, fragCoord.y / ts);
    float tpy = max(1.0, (fragCoord.y - 2.0) / ts);

    vec2 p = fract(vec2(tc.x, (bpy * (1.0 - floor(tc.y)) 
                               + tpy * floor(tc.y)))) - 0.5;
    
    vec3 px = vec3(0.5*n.x, p.y, -p.x*n.x) * step(0.5, n.x)
              + vec3(0.5*n.x, -p.x, -p.y*n.x) * step(n.x, -0.5);
    vec3 py = vec3(-p.x*n.y, 0.5*n.y, p.y) * abs(n.y);
    vec3 pz = vec3(p.x*n.z, p.y, 0.5*n.z) * abs(n.z);
    
    return normalize(px + py + pz);
}

// Function 105
vec3 getRayColor(Ray ray) {

    vec3 p =  ray.cp.p;
    float d = mix(DENSITY_MIN, DENSITY_MAX, (ray.eta - ETA)/(1./ETA-ETA));
    vec3 matColor = mix(AIR_COLOR, MATERIAL_COLOR, (ray.eta - ETA)/(1./ETA-ETA));
    vec3 col = getColor(ray,p);

    float q = exp(-d*ray.cp.dist);
    col = col*q+matColor*(1.-q);
    return col*ray.share;
}

// Function 106
Ray getRay(Camera c, vec2 ndc)
{
    Ray r;
    r.o = c.pos;
    r.d = normalize(c.z + c.x*ndc.x + c.y*ndc.y);
    return r;
}

// Function 107
vec3 GetRayDir(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l-p),
        r = normalize(cross(f,vec3(0,1,0))),
        u = cross(r,f),
        c = f*z,
        i = c + uv.x*r + uv.y*u,
        d = normalize(i);
    return d;
}

// Function 108
Ray getRay(vec2 uv)
{
    float dis=1./tan(.5*cam.fov*PI/180.);
    return Ray(cam.pos,normalize(uv.x*cam.r+uv.y*cam.u+cam.f*dis));
}

// Function 109
ray GetRay(vec2 uv, vec3 camPos, vec3 lookat, float zoom) {
        ray a;
        a.o = camPos;
        vec3 f = normalize(lookat-camPos);
        vec3 r = cross(vec3(0, 1., 0), f);
        vec3 u = cross(f, r);
        vec3 c = a.o + f * zoom;
        vec3 i = c + uv.x * r + uv.y * u;

        a.d = normalize(i - a.o);

        return a;
    }

// Function 110
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
        vec2 xy = fragCoord - size / 2.0;
        float z = size.y / tan(radians(fieldOfView) / 2.0);
        return normalize(vec3(xy, -z));
    }

// Function 111
vec3 rayDirection(float fieldofView,vec2 size,vec2 fragCoord) {
  vec2 xy = fragCoord - size / 2.0;
  float z = size.y / tan(radians(fieldofView) / 2.0 );
  return normalize(vec3(xy,-z));
}

// Function 112
vec3 getRayDir(vec3 ro, vec2 uv) {
	vec3 forward = normalize(-ro),
	     right = normalize(cross(vec3(0, 1, 0), forward));
	return normalize(forward + right * uv.x + cross(forward, right) * uv.y);
}

// Function 113
vec3 RayDirection(float fieldOfView, vec2 fragCoord, vec2 size){
	vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

// Function 114
float getRayleigMultiplier(vec2 p, vec2 lp){
	return 1.0 + pow(1.0 - clamp(distance(p, lp), 0.0, 1.0), 2.0) * pi * 0.5;
}

// Function 115
Ray Camera_getRay(Camera camera, float s, float t)
{
    vec3 rd = camera.lensRadius * random_in_unit_disk();
    vec3 offset = camera.u * rd.x + camera.v * rd.y;

    Ray ray;

    ray.origin = camera.origin + offset;
    ray.direction = camera.lowerLeftCorner + s * camera.horizontal + t * camera.vertical - camera.origin - offset;

    return ray;
}

// Function 116
vec3 rayDir(float fov, vec2 size, vec2 pos)
{
    vec2 xy = pos - size * 0.5;
    
    float cotHalfFOV = tan((90.0 - fov * 0.5) * DEG_TO_RAD);
    float z = size.y * 0.5 * cotHalfFOV;
    
    return normalize(vec3(xy, -z));
}

// Function 117
vec3 getRayDir(vec3 cameraDir, vec2 screenPos) {
	vec3 planeU = vec3(1.0, 0.0, 0.0);
	vec3 planeV = vec3(0.0, iResolution.y / iResolution.x * 1.0, 0.0);
	return normalize(cameraDir + screenPos.x * planeU + screenPos.y * planeV);
}

// Function 118
vec3 getRay(vec2 angles, vec2 pos)
{
    mat3 camera = getRot(angles);
    return normalize(transpose(camera)*vec3(FOV*pos.x, FOV*pos.y, 1.));
}

// Function 119
vec3 getRay_870892966(vec3 origin, vec3 target, vec2 screenPos, float lensLength) {
  mat3 camMat = calcLookAtMatrix_1460171947(origin, target, 0.0);
  return getRay_870892966(camMat, screenPos, lensLength);
}

// Function 120
vec3 getRay(vec2 uv){
    uv = (uv * 2.0 - 1.0) * vec2(Resolution.x / Resolution.y, 1.0);
	vec3 proj = normalize(vec3(uv.x, uv.y, 1.0) + vec3(uv.x, uv.y, -1.0) * pow(length(uv), 2.0) * 0.05);	
    if(Resolution.x < 400.0) return proj;
	vec3 ray = rotmat(vec3(0.0, -1.0, 0.0), 3.0 * (Mouse.x * 2.0 - 1.0)) * rotmat(vec3(1.0, 0.0, 0.0), 1.5 * (Mouse.y * 2.0 - 1.0)) * proj;
    return ray;
}

// Function 121
vec3 calcCameraRayDir(float fov, vec2 fragCoord, vec2 resolution)
{
	float tanFov = tan(fov / 2.0 * 3.14159 / 180.0) / resolution.x;
	vec2 p = tanFov * (fragCoord * 2.0 - resolution.xy);
	vec3 rayDir = normalize(vec3(p.x, p.y, 1.0));
	rotateAxis(rayDir.yz, iCamRotX);
	rotateAxis(rayDir.xz, iCamRotY);
	rotateAxis(rayDir.xy, iCamRotZ);
	return rayDir;
}

// Function 122
vec3 rayDirection(float fieldOfView, vec2 fragCoord) {
    vec2 xy = fragCoord - iResolution.xy / 2.0;
    float z = (0.5 * iResolution.y) / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

// Function 123
vec3 GetCameraRayDir( const in vec2 vWindow, const in vec3 vCameraPos, const in vec3 vCameraTarget )
{
	vec3 vForward = normalize(vCameraTarget - vCameraPos);
	vec3 vRight = normalize(cross(vec3(0.0, 1.0, 0.0), vForward));
	vec3 vUp = normalize(cross(vForward, vRight));
							  
    float fPersp = 3.0;
	vec3 vDir = normalize(vWindow.x * vRight + vWindow.y * vUp + vForward * fPersp);

	return vDir;
}

// Function 124
vec3 getRay_9_4(vec3 origin, vec3 target, vec2 screenPos, float lensLength) {
  mat3 camMat = calcLookAtMatrix_7_1(origin, target, 0.0);
  return getRay_9_4(camMat, screenPos, lensLength);
}

// Function 125
Ray GetRay(vec2 uv, vec3 camPos, vec3 dir, float zoom){
    Ray r;
    r.o = camPos;
    vec3 f = normalize(dir);
    vec3 right = cross(vec3(.0,1.,.0), f);
    vec3 u = cross(f,right);
    
    vec3 c = r.o + f*zoom;
    vec3 i = c + uv.x *right + uv.y *u;
    r.d = normalize(i -r.o);
    return r;
}

// Function 126
vec3 getRayDir(vec3 ro, vec3 lookAt, vec2 uv) {
	vec3 f = normalize(lookAt - ro),
	     r = normalize(cross(vec3(0, 1, 0), f));
	return normalize(f + r * uv.x + cross(f, r) * uv.y);
}

// Function 127
vec3 getRayDir(in vec2 uv)
{
    return normalize(vec3(uv, -1.0));
}

// Function 128
vec3 setupRayDirection(float camFov)
{
	vec2 coord = vec2(gl_FragCoord.xy);
    vec2 v = vec2(coord / iResolution.xy) * 2.0 - 1.0;
    float camAspect = iResolution.x/iResolution.y;
    float fov_y_scale = tan(camFov/2.0);
    vec3 raydir = vec3(v.x*fov_y_scale*camAspect, v.y*fov_y_scale, -1.0);
    return normalize(raydir);
}

// Function 129
vec3 getRay(vec3 ro, vec3 look, vec2 uv){
    vec3 f = normalize(look - ro);
    vec3 r = normalize(vec3(f.z,0,-f.x));
    vec3 u = cross (f,r);
    return normalize(f + uv.x * r + uv.y * u);
}

// Function 130
float getRayleighPhase(float cosTheta) {
    const float k = 3.0/(16.0*PI);
    return k*(1.0+cosTheta*cosTheta);
}

// Function 131
vec3 CubemapRayDir(in vec2 fragCoord) 
{
    vec2 t = fragCoord.xy*vec2(4.0, 2.0) / iResolution.xy;
    vec3 n = CubemapNormal(floor(t));
    
    float g = 4.0 / iResolution.x;
    float vo = iResolution.x*0.5 - iResolution.y;
    
    vec2 xzp = fract(min(vec2(4.0, 0.99999), fragCoord.xy * g));
    
    vec2 ypp = vec2(min(0.99999, fragCoord.x * g), max(1.0, (fragCoord.y + vo) * g));
    vec2 ypn = vec2(max(3.0,     fragCoord.x * g), max(1.0, (fragCoord.y + vo) * g));
    vec2 yp = fract(ypp * step(-0.5, n.y) + ypn * (1.0 - step(-0.5, n.y)));
    
    vec2 p = (xzp * (1.0 - abs(n.y)) + yp * abs(n.y)) - 0.5;
    
    vec3 px = vec3(0.5*n.x, p.y, -p.x*n.x) * abs(n.x);
    vec3 py = vec3(p.x*n.y, 0.5*n.y, -p.y) * abs(n.y);
    vec3 pz = vec3(p.x*n.z, p.y, 0.5*n.z) * abs(n.z);
    
   	vec3 rd = px + py + pz; 
    return normalize(rd);
}

// Function 132
vec3 getRay(vec2 uv){
    uv = (uv * 2.0 - 1.0) * vec2(Resolution.x / Resolution.y, 1.0);
	vec3 proj = normalize(vec3(uv.x, uv.y, 1.0) + vec3(uv.x, uv.y, -1.0) * pow(length(uv), 2.0) * 0.1);	
    if(Resolution.x < 400.0) return proj;
	vec3 ray = axisangle(vec3(0.0, -1.0, 0.0), 3.0 * (Mouse.x * 2.0 - 1.0)) * axisangle(vec3(1.0, 0.0, 0.0), 1.5 * (Mouse.y * 2.0 - 1.0)) * proj;
    return ray;
}

// Function 133
vec3 uvToRayDir( vec2 uv ) {
    vec2 v = PI * ( vec2( 1.5, 1.0 ) - vec2( 2.0, 1.0 ) * uv );
    return vec3(
        sin( v.y ) * cos( v.x ),
        cos( v.y ),
        sin( v.y ) * sin( v.x )
    );
}

// Function 134
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
	vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

// Function 135
vec3 getRay(vec3 pos, vec3 dir, vec3 up, vec2 fragCoord)
{
	vec2 xy = fragCoord.xy / iResolution.xy - vec2(0.5);
	xy.y *= -iResolution.y / iResolution.x;

	vec3 eyed = normalize(dir);
	vec3 ud = normalize(cross(vec3(0.0, -1.0, 0.0), eyed));
	vec3 vd = normalize(cross(eyed, ud));

	float f = FOV * length(xy);
	return normalize(normalize(xy.x * ud + xy.y * vd) + (1.0 / tan(f)) * eyed);
}

// Function 136
vec3 getRay(vec2 uv, vec3 eye, vec3 up, vec3 right)
{
    float nearWidth = tan(fov/2.0) * near * 2.0;
    float nearHeight = iResolution.y/iResolution.x * nearWidth;
    vec3 front = normalize(cross(up, right));
    right = normalize(cross(front, up));
    vec3 p = eye + front * near + right * uv.x * nearWidth + up * uv.y * nearHeight;
    return normalize(p - eye);
}

// Function 137
vec3 getRay(vec3 origin, vec3 target, vec2 screenPos, float lensLength) {
  mat3 camMat = calcLookAtMatrix(origin, target, 0.0);
  return normalize(camMat * vec3(screenPos, lensLength));
}


```