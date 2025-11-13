// Reusable Optimizations Raymarching Functions
// Automatically extracted from raymarching/raytracing-related shaders

// Function 1
vec4 march(vec3 p, vec3 d)
{
    vec4 m = vec4(p,0);
    for(int i = 0; i<99; i++)
    {
        float s = dist(m.xyz);
        m += vec4(d,1)*s;
        
        if (s<.01 || m.w>20.) break;
    }
    return m;
}

// Function 2
vec3 raymarch(vec3 raydir, vec3 rayori){
    // Start our ray (moved slightly forward for dithering)
    vec3 raypos = rayori+(raydir*rand()*stepsize);

    for(uint i = 0U; i < maxmarches; i++){
        // Check if we reached our Source
        if(raypos.z > 0.0){return vec3(signal(raypos.xy));}

        // Check if the Ray is Outside an Acceptable Position
        if(abs(raypos.z) < scenesize || length(raypos.xy) > transmissionlanesize){break;}

        // "Distort" the Ray
        raydir = normalize(raydir+(fbm3(raypos*2.0)*0.025));

        // March the Ray
        raypos += raydir*stepsize;
    }

    // If the ray reached an unacceptable position or never hit anything, output 0.0
    return vec3(0.0);
}

// Function 3
void marchThroughField(inout vec3 pos, vec3 dir, vec3 eye)
{
	float dist;
	for(int i = 0; i < MAX_VIEW_STEPS; i++)
	{
		dist = getDist(pos);
		if(dist < EPSILON || length(pos-eye) > MAX_DEPTH-EPSILON)			
			return;
		else	
			pos += dir*dist;
	}
	return;
}

// Function 4
vec2 march(in vec3 ro, in vec3 rd) {
    const int maxSteps = 100;

    vec2 Q = vec2(1e9);

    vec3 p = ro;
    float t = 0.0;
    for (int n = 1; n <= maxSteps; ++n) {
        Q = map(ro + rd * t);
        float closeEnoughEps = (n == maxSteps ? 0.2 : closeEps);
        if (Q.x < closeEnoughEps) {
            return vec2(t, Q.y);
        }
        t += Q.x;
        if (t > 200.0) {
            return vec2(t, Q.y);
        }
    }
    return vec2(t, Q.y);
}

// Function 5
vec3 march(vec3 ro, vec3 rd){
    float total = 0.;
    float dist = 0.;
    for(int i = 0; i<300; i++){
        dist = map(ro+rd*total);
        total+=dist;
        if(dist < 0.01){
        	break;
        }
    }
    if(dist>0.01){
    	matID = 1;
    }
   
    return (ro+rd*total);
    
}

// Function 6
float RayMarch(vec3 ro, vec3 rd, int PMaxSteps)
{   float t = 0.; 
    vec3 dS=vec3(9999.0,-1.0,-1.0);
    float marchCount = 0.0;
    vec3 p;
    float minDist = 9999.0; 
    
    for(int i=0; i <= PMaxSteps; i++) 
    {  	p = ro + rd*t;
        dS = GetDist(p);
        t += dS.x;
        if ( abs(dS.x)<MIN_DIST  || i == PMaxSteps)
            {mObj.hitbln = true; minDist = abs(t); break;}
        if(t>MAX_DIST)
            {mObj.hitbln = false;    minDist = t;    break; } 
        marchCount++;
    }
    mObj.dist = minDist;
    mObj.id_color = dS.y;
    mObj.marchCount=marchCount;
    mObj.id_material=dS.z;
    mObj.normal=GetNormal(p);
    mObj.phit=p;
    return t;
}

// Function 7
vec3 ray_march_diffuse(vec3 rayOrigin, vec3 rayDir, vec3 diffuse) {
        float distanceTravelled = 0.0;
        const int NUMBER_OF_STEPS = 64;
        const float MINIMUM_HIT_DISTANCE = 0.001;
        const float MAXIMUM_TRACE_DISTANCE = 1000.0;
        const int BOUNCE_AMOUNTS = 3;
        vec3 finalDiffuse = diffuse;
        bool done = false;
        for(int r = 0; r < BOUNCE_AMOUNTS; r++) {
            if (done) {
                break;
            }
            for(int i = 0; i < NUMBER_OF_STEPS; i++) {
                vec3 currPos = rayOrigin + distanceTravelled * rayDir;
                vec4 sceneData = scene_dist(currPos);
                float sceneDist = sceneData.x;
                vec3 sceneDiffuse = sceneData.yzw;
                if (sceneDist < MINIMUM_HIT_DISTANCE) {
                    float addWeight = pow(0.5, float(r + 1));
                    finalDiffuse = (1.0 - addWeight) * finalDiffuse + addWeight * sceneDiffuse;
                    vec3 normal = calculate_normal(currPos);
                    rayOrigin = currPos;
                    rayDir = reflect(rayDir, normal);
                    break;
                }
                if (sceneDist > MAXIMUM_TRACE_DISTANCE) {
                    done = true;
                    break;
                }
                distanceTravelled += sceneDist;
            }
        }
        finalDiffuse = 0.6 * Sky(rayOrigin, rayDir) + 0.4 * finalDiffuse;
        return finalDiffuse;
    }

// Function 8
vec4 rayMarching(vec3 viewVec, vec3 eyePos, out bool isHit, out vec3 normal, float epsilon, out float AO)
{
	isHit = false;
	float depth = 0.1;

	int count = 0;

	vec3 endPoint;

	float radius = 1.0;
	vec3 c = vec3(10.0);

	int maxRayStep = 128;

	for(int i=0; i<maxRayStep; i++)
	{
		endPoint = eyePos + depth * viewVec;

		vec2 result = SDF( endPoint, eyePos, viewVec);

		float dist = result.x;

		if(dist < epsilon * depth) 
		{
			isHit = true;       

			normal = getSurfaceNormal(endPoint, epsilon, eyePos, viewVec);
			AO = getAO(endPoint, normal);

			return vec4(endPoint, result.y);
		}

		depth += dist * STEP_SIZE_SCALER;// + epsilon * log(float(i) + 1.0);

		if(depth >= MAX_RAYDISTANCE)
		{			
			return vec4(endPoint, -1.0);
		}
	}

	return vec4(endPoint, -1.0);
}

// Function 9
vec2 marchVxl(in vec3 ro, in vec3 rd, float near, float far, out vec3 alig, out vec3 vPos)
{
    float lastD = 0.0001;
    float travel = near;
    
    float gridStride = 0.;
    vec3 ip = vec3(0);
    vec3 invRd = 1./rd;
    vec2 bxNfo = vec2(0.);
    
    for( int i=0; i<ITR; i++ )
    {
        travel += lastD*.8 + gridStride;
        if(travel > far) break;
        vec3 pos = ro + rd*travel;
        float mapD = map(pos).x;
        
        if (mapD < (scl*1.2))
        {
            travel -= lastD*0.6;
            pos = ro + rd*travel;
            ip = (floor(pos/scl) + 0.5)*scl;
        	bxNfo = map(ip);
            if (bxNfo.x < 0.0) break;
            vec3 q  = fract(pos/scl)*scl - hscl;
            gridStride = dBox(q, invRd, hscl + 1e-6);
            mapD = 0.;
        }
        else gridStride= 0.;
        lastD = mapD;
    }
    
    vec3 intc = -(fract((ro + rd*travel)/scl)*scl - hscl)*invRd - abs(invRd)*hscl;
    alig = step(intc.yzx, intc.xyz)*step(intc.zxy, intc.xyz);
    vPos = ip;
    
	return vec2(travel, bxNfo.y);
}

// Function 10
vec4 trace_steps(vec3 start_point, vec3 direction, int steps, float dist) {
    vec3 end_vec = direction * dist;
    for (int i=0; i<steps; i++) {
        float percent = float(i) / float(steps);
        vec3 point = start_point + percent * end_vec;
        
        float df = map(point);
        if (df < 0.0) {
            return vec4(point, percent);
        }
    }
    return vec4(start_point + direction*dist, 1.0);
}

// Function 11
Result march (in vec3 ro, in vec3 rd, out int iter)
{
    Result res = Result (.0, 0);
    for (int i = 0; i < MAX_ITER; ++i) {
        iter = i;
        vec3 p = ro + res.d * rd;
        Result tmp = scene (p);
        //if (tmp.d < EPSILON) return res;
        if (abs (tmp.d) < EPSILON * (1. + .125*tmp.d)) return res;
        res.d += tmp.d * STEP_SIZE;
        res.id = tmp.id;
    }

    return res;
}

// Function 12
float stepsize(vec3 p)
{
    float md = sqrt(min(min(
        sq(p - sphere1.xyz), 
        sq(p - sphere2.xyz)), 
        sq(p - sphere3.xyz)));
    return max(min_stepsize, (md - 1.0) * 0.667);
}

// Function 13
float rayMarch(in vec3 ro, in vec3 rd) {
	float dO = 0.0;
    
    for (int i = 0; i < RAYMARCH_MAX_STEPS; i++) {
		vec3 p = ro + rd * dO;
        float dS = getDist(p);
        dO += dS;
        if (dO > RAYMARCH_MAX_DIST) break;
        if (dS < RAYMARCH_SURFACE_DIST) {
            break;
        }
    }
    
    return dO;
}

// Function 14
vec2 raymarch(vec3 ro, vec3 rd, float t) {
    float id = -1.;
    for(int i = 0; i < MAX_ITER; i++) {
        vec2 scn = dstScene(ro+rd*t);
        if(scn.x < MIN_DIST*t || t > MAX_DIST) {
            id = scn.y;
            break;
        }
        t += scn.x * .5;
    }
    return vec2(t,id);
}

// Function 15
void marchObjects(vec3 eye, vec3 ray, float wDepth, inout vec4 color) {
    float dist = 0.0;
    int id;
    vec3 rayPos = eye + ray * dist;
    vec3 c;
    float depth = CAM_FAR;
    vec3 glowColor = vec3(0.0);
    for (int i = 0; i < 100; i++) {
        dist = objects(rayPos, color.rgb, id);
        depth = distance(rayPos, eye);
        if (depth > wDepth) {
            break;
        }
        if (dist < 0.01) {
            vec3 normal = objectsNormal(rayPos, 0.01);
            color = vec4(objectsColor(id, normal, ray), depth);
            return;
        }
        rayPos += ray * dist;
    }
}

// Function 16
vec2	march(vec3 pos, vec3 dir)
{
    vec2	dist = vec2(0.0, 0.0);
    vec3	p = vec3(0.0, 0.0, 0.0);
    vec2	s = vec2(0.0, 0.0);

	    for (float i = -1.; i < I_MAX; ++i)
	    {
	    	p = pos + dir * dist.y;
	        dist.x = scene(p);
	        dist.y += dist.x;
	        if (dist.x < E || dist.y > FAR)
            {
                break;
            }
	        s.x++;
    }
    s.y = dist.y;
    return (s);
}

// Function 17
vec2 RayMarch(vec3 ro, vec3 rd){
    // distance from origin
    vec2 dO=vec2(0.,0.);
    // march until max steps is achieved or object hit
    for(int i=0; i <MAX_STEPS; i++){
        // current point being evaluated
        vec3 p = ro + dO.x*rd;
        
        // get distance to seam
        vec2 ds = GetDist(p);
        //move origin to new point
        
        /*if(ds.y==7.){
            dO+=ds.x*.4;
        }
        else{
            dO+=ds.x*.7;
        }*/
         dO+=ds.x*.8;
        if(ds.x < SURFACE_DIST){
            dO.y = ds.y;
            break;
        }
        else if( dO.x > MAX_DIST){
            dO.y= -1.;
            break;
        }
    }
    return dO;
}

// Function 18
vec2 rayMarching(vec3 ro, vec3 rd) {
    float tmax = MAX_DIST;
    float t = 0.0;
    vec2 result = vec2(-1.0);
    
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        vec2 res = sceneSDF(p);
        if (res.x < EPSILON || t > tmax) break;
       
        t += res.x;
        result.x = t;
        result.y = res.y;
    }
    
    if (t > tmax) result.x = tmax;
    return result;
}

// Function 19
float march(vec3 ro,vec3 rd, int iter){
	float maxd=10.;
    float tmpDist=1.;
    float finalDist;
    for(int i=0;i<ITERATIONS_MAX;i++){
        if(i>iter)break;
        if( tmpDist<0.0001||finalDist>maxd) break;
	    tmpDist=map(ro+rd*finalDist);
        finalDist+=tmpDist; }
    if(finalDist>maxd) finalDist=-1.;
	return finalDist; }

// Function 20
vec2 Step (int sId)
{
  vec4 dv;
  vec2 sv, c, del2c;
  c = Loadv4 (sId).xy;
  sv = vec2 (mod (float (sId), gSize), floor (float (sId) / gSize));
  dv = mod (vec4 (sv.xx, sv.yy) + vec4 (1., -1., 1., -1.), gSize);
  del2c = Loadv4 (int (dv.x + gSize * sv.y)).xy +
          Loadv4 (int (dv.y + gSize * sv.y)).xy +
          Loadv4 (int (sv.x + gSize * dv.z)).xy +
          Loadv4 (int (sv.x + gSize * dv.w)).xy - 4. * c;
  c += delT * (difC * del2c - constKF.y * c +
     vec2 (constKF.y, - constKF.x * c.y) + c.x * c.y * c.y * vec2 (-1., 1.));
  return c;
}

// Function 21
float march(vec3 ro, vec3 rd, out float d, int id)
{
    float t = .0;
    for(int i = 0; i < 64; i++)
    {
        d = map(ro+t*rd, id);
        if(d < EPS || t > MAX) break;
        t += d;
    }
    return t;
}

// Function 22
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

// Function 23
float raymarcher( in vec3 ro, in vec3 rd )
{
	const float maxd = 50.0;
	const float precis = 0.01;
    float h = precis*2.0;
    float t = 0.0;
	float res = -1.0;
    for( int i=0; i<100; i++ )
    {
        if( h<precis||t>maxd ) break;
	    h = scene( ro+rd*t );
        t += h * 1.0;
    }

    if( t<maxd ) res = t;
    return res;
}

// Function 24
float rayMarch(in vec3 ro, in vec3 rd, out int mat, out int iter)
{
  float t = 0.0;
  float distance;
  for (int i = 0; i < MAX_RAY_MARCHES; i++)
  {
    iter = i;
    distance = map(ro + rd*t, mat);
    if (distance < TOLERANCE || t > MAX_RAY_LENGTH) break;
    t += distance;
  }
  
  if (abs(distance) > 100.0*TOLERANCE) return MAX_RAY_LENGTH;
  
  return t;
}

// Function 25
vec2 Step (vec2 sv)
{
  vec4 dv;
  vec2 c, del2c;
  c = Loadv4 (sv).xy;
  dv = mod (vec4 (sv.xx, sv.yy) + vec4 (1., -1., 1., -1.), gSize);
  del2c = Loadv4 (vec2 (dv.x, sv.y)).xy +
          Loadv4 (vec2 (dv.y, sv.y)).xy +
          Loadv4 (vec2 (sv.x, dv.z)).xy +
          Loadv4 (vec2 (sv.x, dv.w)).xy - 4. * c;
  c += delT * (difC * del2c - constKF.y * c +
     vec2 (constKF.y, - constKF.x * c.y) + c.x * c.y * c.y * vec2 (-1., 1.));
  return c;
}

// Function 26
float RayMarch(vec3 ro, vec3 rd)
{
    float dO = 0.;
    
    for(int i = 0 ; i < MAX_STEPS; i++){
        vec3 p = ro + rd * dO;
        float dS = GetDist(p);
        dO += dS;
        if(dO > MAX_DIST || dS < SURF_DIST) break;
    }
    return dO;
}

// Function 27
vec3 raymarch(Ray inputray)
{
    const float exposure = 1e-2;
    const float gamma = 2.2;
    const float intensity = 100.0;
    vec3 ambient = vec3(0.2, 0.3, 0.6) *6.0* intensity / gamma;

    vec3 prevcolour = vec3(0.0, 0.0, 0.0);
    vec3 colour = vec3(0.0, 0.0, 0.0);
    vec3 mask = vec3(1.0, 1.0, 1.0);
    vec3 fresnel = vec3(1.0, 1.0, 1.0);
    
    Ray ray=inputray;
        
    vec3 lightpos = g_light.pos;
    
    for (int i=0; i<REFLECT_ITERATIONS; i++)
    {
        Result result = raymarch_query(ray, 10.0);

        vec3 tolight = lightpos - result.pos;
        tolight = normalize(tolight);
                
        if (result.t > NOT_CLOSE)
        {
            vec3 spotlight = drawlights(ray)*3000.0;
            
//          ambient = texture(iChannel1, ray.dir).xyz*100.0;
            ambient = vec3(0.0);
//            ambient = environment(ray.dir);
                       
            colour += mask * (ambient + spotlight);                             
            break;
        }
        else
        {   
			prevcolour = result.mat.colour.rgb;
            
            vec3 r0 = result.mat.colour.rgb * result.mat.specular;
            float hv = clamp(dot(result.normal, -ray.dir), 0.0, 1.0);
            fresnel = r0 + (1.0 - r0) * pow(1.0 - hv, 5.0);
            mask *= fresnel;            
            
            vec3 possiblelighting = clamp(dot(result.normal, tolight), 0.0, 1.0) * g_light.colour
                    * result.mat.colour.rgb * result.mat.diffuse
                    * (1.0 - fresnel) * mask / fresnel;
            
            possiblelighting += environment(reflect(ray.dir, result.normal)) * dot(result.normal, -ray.dir)*0.35;
            
            float falloff = 1.0 - clamp(length(ray.pos)*0.1, 0.0, 1.0);
            possiblelighting *= falloff;
            
            if (length(possiblelighting) > 0.01f)
            {
                Ray shadowray = Ray(result.pos+result.normal*0.01, tolight);
                Result shadowresult = raymarch_query(shadowray, length(lightpos - result.pos)*0.9);
#ifdef SOFTSHADOWS                
                colour += possiblelighting*clamp(shadowresult.mint*4.0, 0.0, 1.0);
#else
                if (shadowresult.travelled >= length(lightpos - result.pos)*0.9)
                	colour += possiblelighting;
#endif
            }
            
            float falloff2 = 1.0 - clamp(length(ray.pos)*0.3, 0.0, 1.0);
            vec3 selfillum = result.mat.selfillum*result.mat.colour.rgb*40.0*clamp(dot(result.normal, -ray.dir), 0.0, 1.0);            
            colour += selfillum*falloff2;
            
            Ray reflectray;
            reflectray.pos = result.pos + result.normal*0.02f;
            reflectray.dir = reflect(ray.dir, result.normal);
            ray = reflectray;
        }
    }
        
    colour.xyz = vec3(pow(colour * exposure, vec3(1.0 / gamma)));    
    return colour;    
}

// Function 28
vec4 raymarch(vec3 p, vec3 d)
{
    float S = 0.0;
    float L = S;
    vec3 D = normalize(d);
    vec3 P = p+D*S;
    for(int i = 0;i<240;i++)
    {
        S = model(P);
        L += S;
        P += D*S;
        if ((L>MAX) || (S<PRE)) {break;}
    }
    return vec4(P,min(L/MAX,1.0));
}

// Function 29
void Raymarch( const in C_Ray ray, out C_HitInfo result, const float fMaxDist, const int maxIter )
{          
	result.fDistance = kRaymarchStartDistance;
	result.vObjectId.x = 0.0;
				    
	for(int i=0;i<=kRaymarchMatIter;i++)                
	{
		result.vPos = ray.vOrigin + ray.vDir * result.fDistance;
		vec4 vSceneDist = GetDistanceScene( result.vPos );
		result.vObjectId = vSceneDist.yzw;
		
		// abs allows backward stepping - should only be necessary for non uniform distance functions
		if((abs(vSceneDist.x) <= kRaymarchEpsilon) || (result.fDistance >= fMaxDist) || (i > maxIter))
		{
			break;
		}                          	
		
		result.fDistance = result.fDistance + vSceneDist.x;      
	}
	
	
	if(result.fDistance >= fMaxDist)
	{
		result.vPos = ray.vOrigin + ray.vDir * result.fDistance;
		result.vObjectId.x = 0.0;
		result.fDistance = 1000.0;
	}
}

// Function 30
vec2 raymarch_main_scene(vec3 _p, float t)
{
    vec2 sand = scene_base_sand(_p);
    #if defined(ENABLE_HD_SAND_DEPTH)
    sand.x = max(sand.x, -(length(_p-vec3(30., 10., 140.))-90.));
    #else
    sand.x = max(sand.x, -(length(_p-vec3(-20., 2., 53.))-16.));
    #endif
    #if defined(ENABLE_REEDS)
 	return scene_min(scene_reeds(_p-vec3(1., 0., -3.), sand.x, t), scene_min(sand, scene_pyramids(_p)));
    #else
        return scene_min(sand, scene_pyramids(_p));
    #endif
}

// Function 31
float march(vec3 ro, vec3 rd) {
  float t = 0.0;
  for(int i = 0; i < maxsteps; i++) {
    vec3 p = ro + t*rd;
    float d = getdist(p);
    t += d;
    if (t > maxdist || abs(d) < t*precis) break;
  }
  return t;
}

// Function 32
float march(vec3 ro, vec3 rd){
  float t = E;
  float d = 0.0;

  float omega = 1.0;//muista testata eri arvoilla! [1,2]
  float prev_radius = 0.0;

  float candidate_t = t;
  float candidate_error = 1000.0;
  float sg = sgn(scene(ro));

  vec3 p = ro;

	for(int i = 0; i < STEPS; ++i){
		float sg_radius = sg*scene(p);
		float radius = abs(sg_radius);
		d = sg_radius;
		bool fail = omega > 1. && (radius+prev_radius) < d;
		if(fail){
			d -= omega * d;
			omega = 1.;
		}
		else{
			d = sg_radius*omega;
		}
		prev_radius = radius;
		float error = radius/t;

		if(!fail && error < candidate_error){
			candidate_t = t;
			candidate_error = error;
		}

		if(!fail && error < PIXELR || t > FAR){
			break;
		}
		t += d;
    p = rd*t+ro;
	}
  //discontinuity reduction
  float er = candidate_error;
  for(int j = 0; j < 6; ++j){
    float radius = abs(sg*scene(p));
    p += rd*(radius-er);
    t = length(p-ro);
    er = radius/t;

    if(er < candidate_error){
      candidate_t = t;
      candidate_error = er;
    }
  }
	if(t <= FAR || candidate_error <= PIXELR){
		t = candidate_t;
	}
	return t;
}

// Function 33
vec3 raymarch(const in vec3 origin, const in mat3 view, const in vec2 uv, const in vec2 invSize) {
  vec2 p = -1.0 + 2.0 * uv;
  p.x *= invSize.y / invSize.x;
  vec3 rd = normalize(view * vec3(p, 2.0));
  return render(origin, rd);
}

// Function 34
vec2 raymarch_main_scene_normals(vec3 _p, float t)
{
    return scene_min(scene_base_sand(_p), scene_pyramids(_p));
}

// Function 35
vec3 vmarch(in vec3 ro, in vec3 rd, in float j, in vec3 orig)
{   
    vec3 p = ro;
    vec2 r = vec2(0.);
    vec3 sum = vec3(0);
    float w = 0.;
    for( int i=0; i<VOLUMETRIC_STEPS; i++ )
    {
        r = map(p,j);
        p += rd*.03;
        float lp = length(p);
        
        vec3 col = sin(vec3(1.02,2.5,1.52)*3.94+r.y)*.85+0.4;
        col.rgb *= smoothstep(1.0,2.09,-r.x);
        col *= smoothstep(0.02,.2,abs(lp-1.1));
        col *= smoothstep(0.1,.34,lp);
        sum += abs(col)*5. * (1.2-noise(lp*2.+j*13.+time*5.)*1.1) / (log(distance(p,orig)-2.)+.75);
    }
    return sum;
}

// Function 36
float raymarch(vec3 ro, vec3 rd) {
    float d = 0., t = 0.0;
    for (int i = 0; i < STEPS; ++i) {
        d = map(ro + t*rd);
        if (d < EPS*t || t > FAR)
            break;
        t += max(0.35*d, 2.*EPS*t);
    }
   
    return d < EPS*t ? t : -1.;
}

// Function 37
vec4 rayMarch(in vec3 from, in vec3 dir, in vec2 fragCoord) {
	// Add some noise to prevent banding
	float totalDistance = Jitter*rand(fragCoord.xy+vec2(iTime));
	vec3 dir2 = dir;
	float distance;
	int steps = 0;
	vec3 pos;
	for (int i=0; i <= MaxSteps; i++) {
		pos = from + totalDistance * dir;
		distance = DE(pos)*FudgeFactor;
		totalDistance += distance;
		if (distance < MinimumDistance) break;
		steps = i;
	}
	
	// 'AO' is based on number of steps.
	// Try to smooth the count, to combat banding.
	float smoothStep =   float(steps) ;
		float ao = 1.0-smoothStep/float(MaxSteps);
	
	// Since our distance field is not signed,
	// backstep when calc'ing normal
	vec3 normal = getNormal(pos-dir*normalDistance*3.0);
	vec3 bg = vec3(0.2);
	if (steps == MaxSteps) {
		return vec4(bg,1.0);
	}
	vec3 color = getColor(normal, pos);
	vec3 light = getLight(color, normal, dir);
	
	color = mix(color*Ambient+light,bg,1.0-ao);
	return vec4(color,1.0);
}

// Function 38
vec4 ray_march(inout vec4 p, vec4 ray, float sharpness) {
	//March the ray
	float d = DE(p);
	if (d < 0.0 && sharpness == 1.0) {
		vec3 v;
		if (abs(iMarblePos.x) >= 999.0f) {
			v = (-20.0 * iMarbleRad) * iMat[2].xyz;
		} else {
			v = iMarblePos.xyz - iMat[3].xyz;
		}
		d = dot(v, v) / dot(v, ray.xyz) - iMarbleRad;
	}
	float s = 0.0;
	float td = 0.0;
	float min_d = 1.0;
	for (; s < float(MAX_MARCHES); s += 1.0) {
		//if the distance from the surface is less than the distance per pixel we stop
		float min_dist = max(FOVperPixel*td, MIN_DIST);
		if (d < min_dist) {
			s += d / min_dist;
			break;
		} else if (td > MAX_DIST) {
			break;
		}
		td += d;
		p += ray * d;
		min_d = min(min_d, sharpness * d / td);
		d = DE(p);
	}
	return vec4(d, s, td, min_d);
}

// Function 39
Obj rayMarch(vec3 ro, vec3 rd) {
	
    float t = 0.;
    
    Obj hitObj = Obj(SKY, t, ro + rd);
    for (int i = 0; i < MAX_STEPS; i++) {
    	
        vec3 p = ro + rd*t;
        
        hitObj = getDistance(p);
        t += hitObj.d*0.35;
        
        // the ray has marched far enough but hit nothing. 
        // Render the pixel as a part of the sky.
        if (t > MAX_DIST) {
        	hitObj = Obj(SKY, t, p);
            break;
        }

        // the ray has marched close enough to an object
        if (abs(hitObj.d) < SURF_DIST) {
            hitObj = Obj(hitObj.type, t, p);
        	break;
        }
        
        hitObj.d = t;
    }
    
    return hitObj;
}

// Function 40
Hit marching(vec3 ro, vec3 rd, float signInd) 
{
    float tmax = MAX_DISTANCE;
    float t = EPSILON;
    Hit result = Hit(-1.0, -1);
    
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        Hit res = sceneSDF(p); 
        float dist = res.dist * signInd;
        
        if (dist < 0.0 )
        {
            return result;
        }
        else if (t > tmax)
        {
            result.matIndex = -1;
            result.dist = tmax;
            break;
        }
        
        t += max(dist, EPSILON); //faster than abs()
        result.dist = t;
        result.matIndex = res.matIndex;
    }
    
    return result;
}

// Function 41
void Step (int mId, out vec3 rm, out vec3 vm, out vec4 qm, out vec3 wm)
{
  mat3 mRot, mRotN;
  vec4 drw4;
  vec3 rmN, vmN, wmN, dr, dv, rts, rtsN, rms, vms, fc, am, wam, dSp, drw;
  float farSep, rSep, grav, dt;
  grav = 2. * gravSgn;
  dt = 0.01;
  rm = Loadv4 (4 + 4 * mId).xyz;
  vm = Loadv4 (4 + 4 * mId + 1).xyz;
  qm = Loadv4 (4 + 4 * mId + 2);
  wm = Loadv4 (4 + 4 * mId + 3).xyz;
  mRot = QtToRMat (qm);
  farSep = length (blkGap * (blkSph - 1.)) + 1.;
  am = vec3 (0.);
  wam = vec3 (0.);
  for (int n = 0; n < nBlock; n ++) {
    rmN = Loadv4 (4 + 4 * n).xyz;
    if (n != mId && length (rm - rmN) < farSep) {
      vmN = Loadv4 (4 + 4 * n + 1).xyz;
      mRotN = QtToRMat (Loadv4 (4 + 4 * n + 2));
      wmN = Loadv4 (4 + 4 * n + 3).xyz;
      for (int j = 0; j < nSiteBk; j ++) {
        rts = mRot * RSite (j);
        rms = rm + rts;
        vms = vm + cross (wm, rts);
        dv = vms - vmN;
        fc = vec3 (0.);
        for (int jN = 0; jN < nSiteBk; jN ++) {
          rtsN = mRotN * RSite (jN);
          dr = rms - (rmN + rtsN);
          rSep = length (dr);
          if (rSep < 1.) fc += FcFun (dr, rSep, dv - cross (wmN, rtsN));
        }
        am += fc;
        wam += cross (rts, fc);
      }
    }
  }
  if (length (rm) > spRad - 0.5 * (farSep + 1.)) {
    for (int j = 0; j < nSiteBk; j ++) {
      rts = mRot * RSite (j);
      dr = rm + rts - spRad * normalize (rm + rts);
      rSep = length (dr);
      if (rSep < 1.) {
        fc = FcFun (dr, rSep, vm + cross (wm, rts));
        am += fc;
        wam += cross (rts, fc);
      }
    }
  }
  am.y -= grav;
  dSp = blkGap * blkSph;
  wam = mRot * (wam * mRot / (0.5 * (vec3 (dot (dSp, dSp)) - dSp * dSp) + 1.));
  vm += dt * am;
  rm += dt * vm;
  wm += dt * wam;
  qm = normalize (QtMul (RMatToQt (LpStepMat (0.5 * dt * wm)), qm));
}

// Function 42
bool ray_march_hit(vec3 rayOrigin, vec3 rayDir) {
        float distanceTravelled = 0.0;
        const int NUMBER_OF_STEPS = 32;
        const float MINIMUM_HIT_DISTANCE = 0.001;
        const float MAXIMUM_TRACE_DISTANCE = 1000.0;
        for(int i = 0; i < NUMBER_OF_STEPS; i++) {
            vec3 currPos = rayOrigin + distanceTravelled * rayDir;
            float sceneDist = scene_dist(currPos).x;
            if (sceneDist < MINIMUM_HIT_DISTANCE) {
                return true;
            }
            if (sceneDist > MAXIMUM_TRACE_DISTANCE) {
                break;
            }
            distanceTravelled += sceneDist;
        }
        return false;
    }

// Function 43
vec4 rayMarch(vec3 rayDir, vec3 cameraOrigin)
{
    const int maxItter = 200;
	const float maxDist = 70.0;
    
    float totalDist = 0.0;
	vec3 pos = cameraOrigin;
	vec3 dist = vec3(epsilon, 0.0, 0.0);
    
    for(int i = 0; i < maxItter; i++)
	{
       	dist = distfunc(pos);
        
		totalDist += dist.x; 
        
		pos += dist.x * rayDir;
        
        if(dist.x < epsilon || totalDist > maxDist)
		{
			break;
		}
	}
    
    return vec4(dist.x, totalDist, dist.y, dist.z);
}

// Function 44
void marchRay(inout Ray ray, inout vec3 colour) {
    bool inside = false; // are we inside or outside the glass object
    vec3 impact = vec3(1); // This decreases each time the ray passes through glass, darkening colours

    vec3 startpoint = ray.origin;
    
#ifdef DEBUG   
vec4 debugColour = vec4(1, 0, 0, 1);
#endif
    
    SDResult result;
    vec3 n;
    vec3 glassStartPos;
    
    for (int i=0; i<kMAXITERS; i++) {
        // Get distance to nearest surface
        result = sceneDist(ray);
        
        // Step half that distance along ray (helps reduce artefacts)
        float stepDistance = inside ? abs(result.d) : result.d;
        ray.origin += ray.dir * stepDistance;
        
        float f = fog(ray, stepDistance);
        
        colour = mix(colour, vec3(0.7), clamp(f, 0., 1.));
        impact *= 1. - f;
        
       // if (length(ray.origin) > 40.0) { break; }
        
        if (stepDistance < eps) {
            // colision
            // normal
            // Get the normal, then clamp the intersection to the surface
    		n = normal(ray);
            //clampToSurface(ray, stepDistance, n);
#ifdef DEBUG
#endif
            
            if (result.material == kFLOORMATERIAL) {
            	// ray hit floor
              
                // Add some noise to the normal, since this is pretending to be grit...
                vec3 randomNoise = texture(iChannel2, ray.origin.xz * .050, 0.0).rgb;
                randomNoise += texture(iChannel2, ray.origin.xz * .10, 0.0).rgb;
                n = mix(vec3(0,1,0), randomNoise-1.0, 0.25);
                colour = mix(colour, kMATTECOLOUR * light(ray, n) * vec3(occlusion(ray, n)), impact);
                /*
				n = mix(n, normalize(vec3(randomNoise.x, 1, randomNoise.y)), randomNoise.z);
                
                // Colour is just grey with crappy fake lighting...
                colour += mix(
                    kFLOORCOLOUR, 
                    vec3(0), 
                    pow(max((-n.x+n.y) * 0.5, 0.0), 2.0)
                ) * impact;
				*/
                impact *= 0.;
                break;
            }
            
            if ( result.material == kMATTEMATERIAL ) {
                // ray hit thing
                
                // tex coord from normal
                vec2 coord = texCoordFromNormal(n);
                	
                float fresnel = fresnelTerm(ray, n, 2.0);
                
                // Add some noise to the normal, since this is pretending to be grit...
                vec3 randomNoise = texture(iChannel2, n.xy * .50, fresnel * 4.0).rgb;
                randomNoise += texture(iChannel2, n.xz * .50, fresnel * 4.0).rgb;
                randomNoise += texture(iChannel2, n.yz * .50, fresnel * 4.0).rgb;
                randomNoise /= 3.0;
                n = mix(n, randomNoise-0.5, 0.3);
                colour = mix(colour, kMATTECOLOUR * light(ray, n) * vec3(occlusion(ray, n)), impact);
                /*
				n = mix(n, normalize(vec3(randomNoise.x, 1, randomNoise.y)), randomNoise.z);
                
                // Colour is just grey with crappy fake lighting...
                colour += mix(
                    kFLOORCOLOUR, 
                    vec3(0), 
                    pow(max((-n.x+n.y) * 0.5, 0.0), 2.0)
                ) * impact;
				*/
                impact *= 0.;
                break;
            }
            
            // check what material it is...
            
            if (result.material == kMIRRORMATERIAL) {
                
                // handle interior glass / other intersecion
                if (inside) {
                     float glassTravelDist =  min(distance(glassStartPos, ray.origin) / 16.0, 1.);
    				glassStartPos = ray.origin;
                    // mix in the colour
                	impact *= mix(kGLASSCOLOUR, kGLASSCOLOUR * 0.1, glassTravelDist);
                    
                }
                
                // it's a mirror, reflect the ray
                ray.dir = reflect(ray.dir, n);
                    
                // Step 2x epsilon into object along normal to ensure we're beyond the surface
                // (prevents multiple intersections with same surface)
                ray.origin += n * eps * 2.0;
                
                // Mix in the mirror colour
                impact *= kMIRRORCOLOUR;
                
            } else {
                // glass material
            
                if (inside) {
                	// refract glass -> air
                	ray.dir = refract(ray.dir, -n, 1.0/kREFRACT);
                    
                    // Find out how much to tint (how far through the glass did we go?)
                    float glassTravelDist =  min(distance(glassStartPos, ray.origin) / 16.0, 1.);
    
                    // mix in the colour
                	impact *= mix(kGLASSCOLOUR, kGLASSCOLOUR * 0.1, glassTravelDist);
                    
#ifdef DEBUG
debugValue += glassTravelDist / 2.0;
#endif
      
                
              	} else {
               		// refract air -> glass
                	glassStartPos = ray.origin;
                    
              	  	// Mix the reflection in, according to the fresnel term
                	float fresnel = fresnelTerm(ray, n, 1.0);
                    fresnel = fresnel;
    				/*
                    colour = mix(
                    	colour, 
                    	texture(iChannel1, reflect(ray.dir, n)), 
                    	vec4(fresnel) * impact);
*/
                    colour = mix(
                        colour,
                        backgroundColour(ray, 0.0),
                        vec3(fresnel) * impact);
                    impact *= 1.0 - fresnel;
    			
                	// refract the ray
            		ray.dir = refract(ray.dir, n, kREFRACT);
                    
#ifdef DEBUG
//debugValue += 0.5;
#endif
                }
            
            	// Step 2x epsilon into object along normal to ensure we're beyond the surface
                ray.origin += (inside ? n : -n) * eps * 2.0;
                
                // Flip in/out status
                inside = !inside;
            }
        }
        
        // increase epsilon
        eps += divergence * stepDistance;
    }
    
    // So far we've traced the ray and accumulated reflections, now we need to add the background.
   // colour += texture(iChannel0, ray.dir) * impact;
    ray.origin = startpoint;
    colour.rgb += backgroundColour(ray, 0.0) * impact;
    
#ifdef DEBUG
//debugColour.rgb = ray.dir;
debugColour.rgb = vec3(float(debugStep)/2.0);
colour = debugColour;
#endif
}

// Function 45
float RayMarchOut(vec3 ro, vec3 rd) 
{
	float dO=0.;
    
    for(float i=0.0; i<1.0; i+=0.05) 
	{
    	vec3 p = ro + rd*i;
        float dS = GetDist(p);
		dO += 0.05 * step(dS, 0.0);
    }
	return exp(-dO*1.1);
}

// Function 46
float maxStep( in vec3 pos ,in vec3 rd,in vec3 gridFloorPos)
{
    //return gridSize*0.2;
    float res = 1e10;
    /*
        a b
		c d
       比如射线依次穿过cdb，cb里的球等高，d里球特高，相交d球时可能返回很大的跨步
	*/
    if( rd.x>0.)
    {
        vec3 planeNormal = vec3(-1.,0.,0.);
        vec3 planePos = gridFloorPos + vec3(gridSize,0.,0.);
        float t = (dot(planeNormal,planePos) - dot(planeNormal,pos)) / dot(planeNormal,rd);
        res = min(t+0.001,res);
    }
    else if( rd.x<0.)
    {
        vec3 planeNormal = vec3(1.,0.,0.);
        vec3 planePos = gridFloorPos;
        float t = (dot(planeNormal,planePos) - dot(planeNormal,pos)) / dot(planeNormal,rd);
        res = min(t+0.001,res);
    }
    
    if( rd.y>0.)
    {
        vec3 planeNormal = vec3(0.,-1.,0.);
        vec3 planePos = gridFloorPos + vec3(0.,gridSize,0.);
        float t = (dot(planeNormal,planePos) - dot(planeNormal,pos)) / dot(planeNormal,rd);
        res = min(t+0.001,res);
    }
    else if( rd.y<0.)
    {
        vec3 planeNormal = vec3(0.,1.,0.);
        vec3 planePos = gridFloorPos;
        float t = (dot(planeNormal,planePos) - dot(planeNormal,pos)) / dot(planeNormal,rd);
        res = min(t+0.0001,res);
    }
    
    if( rd.z>0.)
    {
        vec3 planeNormal = vec3(0.,0.,-1.);
        vec3 planePos = gridFloorPos + vec3(0.,0.,gridSize);
        float t = (dot(planeNormal,planePos) - dot(planeNormal,pos)) / dot(planeNormal,rd);
        res = min(t+0.001,res);
    }
    else if( rd.z<0.)
    {
        vec3 planeNormal = vec3(0.,0.,1.);
        vec3 planePos = gridFloorPos;
        float t = (dot(planeNormal,planePos) - dot(planeNormal,pos)) / dot(planeNormal,rd);
        res = min(t+0.001,res);
    }
    return res;
}

// Function 47
RayHit MarchOneRay(const in vec3 start, const in vec3 rayVec,
                 out vec3 posI, const in int maxIter)
{
    RayHit rh = NewRayHit();
    rh.material = 256.0;
	float dist = 1000000.0;
	float t = 0.0;
	vec3 pos = vec3(0.0);
    vec3 signRay = max(vec3(0.0), sign(rayVec));
	// ray marching time
    pos = start;
    posI = floor(start);
    vec3 delta = signRay - fract(pos);
    vec3 hit = (delta/rayVec);
    vec3 signRayVec = sign(rayVec);
    vec3 invAbsRayVec = abs(1.0 / rayVec);
    // This is the highest we can ray march before early exit.
    float topBounds = max(10.0, start.y);
    for (int i = 0; i < maxIter; i++)	// This is the count of the max times the ray actually marches.
    {
#ifdef DISTANCE_FIELD
        //dist = DistanceToTerrain(posI);
		dist = posI.y - texelFetch(iChannel0, ivec2(posI.xz+gameGridCenter),0).x;
#else
		vec4 terrainTex = texelFetch(iChannel0, ivec2(posI.xz+gameGridCenter),0);
        dist = terrainTex.x;
#endif
        if ((terrainTex.y != 0.0) && (dist >= posI.y)) {
            if (round(posI.y) == 1.0)
            {
                float inGameGrid = max(abs(posI.x), abs(posI.z));
                float tex = mod(terrainTex.y, 16.0);
                // Only draw yellow trail walls if we're inside the game grid.
                if (inGameGrid <= gameGridRadius) {
                    vec3 boxSizeX = vec3(0.02, 0.25, 0.02);
                    vec3 boxSizeZ = vec3(0.02, 0.25, 0.02);
                    boxSizeX.x += sign(float(int(tex)&1))*0.25;
                    boxSizeX.x += sign(float(int(tex)&2))*0.25;
                    boxSizeZ.z += sign(float(int(tex)&4))*0.25;
                    boxSizeZ.z += sign(float(int(tex)&8))*0.25;
                    vec3 boxOffsetX = posI + 0.5 - vec3(0,0.25,0);
                    vec3 boxOffsetZ = posI + 0.5 - vec3(0,0.25,0);
                    // Shrink the blue trail when bad guy dies
                    if (terrainTex.y >= 16.0) {
                        float anim = saturate(animA.x);
                        boxSizeX.y *= anim;
                        boxSizeZ.y *= anim;
                        boxOffsetX.y -= (1.0 - anim) * 0.25;
                        boxOffsetZ.y -= (1.0 - anim) * 0.25;
                    }
                    boxOffsetX.x -= sign(float(int(tex)&1))*0.25;
                    boxOffsetX.x += sign(float(int(tex)&2))*0.25;
                    boxOffsetZ.z -= sign(float(int(tex)&4))*0.25;
                    boxOffsetZ.z += sign(float(int(tex)&8))*0.25;
                    RayHit rh = BoxIntersect(pos, rayVec, boxOffsetX, boxSizeX, 1.0);
                    rh = Union(rh, BoxIntersect(pos, rayVec, boxOffsetZ, boxSizeZ, 1.0));
                    if (rh.tMin == bignum) dist = smallVal;
                    else {
                        rh.material = 128.0;
                        if (terrainTex.y >= 16.0) rh.material = 129.0;
                        return rh;
                    }
                }
            }
            if (terrainTex.y == 256.0) {
                vec3 bpos = pos - (posI + 0.5);
                bpos = RotateY(bpos, localTime*8.0);
                vec3 rayVecR = RotateY(rayVec, localTime*8.0);
                bpos += (posI + 0.5);
                RayHit rh = BoxIntersect(bpos, rayVecR, posI + 0.5, vec3(0.28), 8.0);
                if (rh.tMin == bignum) dist = smallVal;  // Missed powerup. Keep ray marching.
                else return rh;  // Hit powerup
            }
            //vec4 hitS = SphereIntersect(pos, rayVec, posI + 0.5, 0.5);
            //if (hitS.w == bignum) dist = smallVal;
        }
        // || (t > maxDepth)
#ifdef DISTANCE_FIELD
        if ((dist < smallVal) || (posI.y > topBounds)) break;
#else
        if ((dist >= posI.y) || (posI.y > topBounds)) break;
#endif

        vec3 absHit = abs(hit);
        t = min(absHit.x, min(absHit.y, absHit.z));
        vec3 walk = step(absHit, vec3(t));
        hit += walk * invAbsRayVec;
        posI += walk * signRayVec;
        /*if (t == absHit.x) {
            hit.x += invAbsRayVec.x;
            posI.x += signRayVec.x;
        }
        if (t == absHit.y) {
            hit.y += invAbsRayVec.y;
            posI.y += signRayVec.y;
        }
        if (t == absHit.z) {
            hit.z += invAbsRayVec.z;
            posI.z += signRayVec.z;
        }*/
    }
#ifdef DISTANCE_FIELD
    if (dist >= smallVal) return rh;
#else
    if (dist < posI.y) return rh;
#endif
    // Hit the voxel terrain
    pos = t * rayVec + start;
	//vec4 tex = texelFetch(iChannel0, ivec2(pos.xz+gameGridCenter),0);
    //if (abs(pos.y - tex.x) > 2.5) rh.material = 1.0;
    rh.tMin = t;
    rh.hitMin = pos;
    rh.normMin = CalcNormal(pos, rayVec);
    rh.tMax = rh.tMin + 1.0;
    return rh;
}

// Function 48
float raymarching(
  in vec3 prp,
  in vec3 scp,
  in int maxite,
  in float precis,
  in float startf,
  in float maxd,
  out int objfound)
{ 
  const vec3 e=vec3(0.1,0,0.0);
  float s=startf;
  vec3 c,p,n;
  float f=startf;
  objfound=1;
  for(int i=0;i<256;i++){
    if (abs(s)<precis||f>maxd||i>maxite) break;
    f+=s;
    p=prp+scp*f;
    s=obj(p);
  }
  if (f>maxd) objfound=-1;
  return f;
}

// Function 49
float RayMarch(vec3 ro, vec3 rd) {
    float dO = 0.;
    

    for (int i = 0; i<MAX_STEPS; i++) {
        vec3 p = ro+dO*rd;
        float dS = GetDist(p);
        dO+=dS;
        if(dS<SURF_DIST || dO>MAX_DISTANCE) break;
    }
        return dO;    
}

// Function 50
vec4 StepJFA (in vec2 fragCoord, in float level)
{
    level = clamp(level-1.0, 0.0, c_maxSteps);
    float stepwidth = floor(exp2(c_maxSteps - level)+0.5);
    
    float bestDistance = 9999.0;
    vec2 bestCoord = vec2(0.0);
    vec3 bestColor = vec3(0.0);
    
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 sampleCoord = fragCoord + vec2(x,y) * stepwidth;
            
            vec4 data = texture( iChannel0, sampleCoord / iChannelResolution[0].xy);
            vec2 seedCoord;
            vec3 seedColor;
            DecodeData(data, seedCoord, seedColor);
            float dist = length(seedCoord - fragCoord);
            if ((seedCoord.x != 0.0 || seedCoord.y != 0.0) && dist < bestDistance)
            {
                bestDistance = dist;
                bestCoord = seedCoord;
                bestColor = seedColor;
            }
        }
    }
    
    return EncodeData(bestCoord, bestColor);
}

// Function 51
vec2 March(in Ray ray, int maxSteps)
{
    float depth = NearClip;
    float id = 1.0;
    
    for(int i = 0; i < maxSteps; ++i)
    {
        vec2 sdf = Scene(ray.o + (ray.d * depth));
        
        if(sdf.x < Epsilon)
        {
            id = sdf.y;
            break;
        }
        
        if(sdf.x >= FarClip)
        {
            break;
        }
        
        depth += sdf.x;
    }
    
    return vec2(clamp(depth, NearClip, FarClip), id);
}

// Function 52
float ramp_step(float x,float a,float ea)
{
    return clamp((x-a)/ea + 0.5,0.0,1.0);
}

// Function 53
vec3 raymarch(inout Ray ray, vec3 playerPos, bool drawPlayer, bool drawRefractive, vec3 SUN_DIR) {
    float h = 1.0;
    float t = 0.0;
    for(int i = 0; i < MAX_STEPS; i++) {
        h = map(ray.src + t*ray.dir, ray.matID, playerPos, drawPlayer, drawRefractive);
        t += h;
        ray.iter = i;
        if (t > TMAX) break;
    }
    int missed = int(step(TMAX, t));
    ray.matID = (1- missed) * ray.matID;
    ray.t = float(missed)*TMAX + float(1-missed)*t;
    ray.pos = ray.src + ray.t*ray.dir;
    ray.nor = calculateNormal(ray.pos, playerPos);
    if (texture(iChannel1, vec2(KEY_N, 0.25)).x > 0.5)			// Color with normals
        return normalize(0.5*(ray.nor+1.0));
    return computeColor(ray, playerPos, SUN_DIR);
}

// Function 54
vec2 _smoothstep(in vec2 p)
{
    vec2 f = fract(p);
    return f * f * (3.0 - 2.0 * f);
}

// Function 55
float raymarchShadow(in vec3 ro, in vec3 rd, float tmin, float tmax) {
    float sh = 1.0;
    float t = tmin;
    float breakOut = 0.0;
    int i = 0;
    while (i < 40 && breakOut != 1.0) {
        vec3 p = ro + rd * t;
        float d = p.y - fBm(p.xz);
        sh = min(sh, 16.0 * d / t);
        t += 0.5 * d;
        if (d < (0.001 * t) || t > tmax)
            breakOut = 1.0;
        i++;
    }
    return sh;
}

// Function 56
void Step (ivec2 iv, out vec3 r, out vec3 v)
{
  vec3 dr, f, wDir;
  vec2 s;
  float fOvlap, fBend, fGrav, fDamp, dt;
  IdNebs ();
  fOvlap = 1000.;
  fBend = 50.;
  fGrav = 0.25;
  fDamp = 0.5;
  wDir = vec3 (Rot2D (vec2 (-1., 0.), -0.8 * pi * (2. * Fbm1 (0.5 * tCur) - 1.)), 0.).xzy;
  r = GetR (vec2 (iv));
  v = GetV (vec2 (iv));
  f = fOvlap * PairForce (r) + SpringForce (iv, r, v) + fBend * BendForce (iv, r) +
     (1. + 8. * Fbm1 (tCur)) * NormForce (iv, r, wDir);
  f -= fDamp * v;
  f.y -= fGrav;
  dt = 0.02;
  if (iv.x != 0 || mod (float (iv.y), float ((nBallE.y - 1) / 8)) != 0. && iv.y != nBallE.y - 2) {
    v += dt * f;
    r += dt * v;
  }
}

// Function 57
float overshootstep2( float x, float df0, float a, vec3 args )
{
	float y0 = df0 / a; // calculate y0 such that the derivative at x=0 becomes df0
	float y = x > 0.0 ? overshoot( x, args, df0 ) : -( 1.0 - exp( x * a ) ) * y0; // look there is a smiley in that calculation
	return ( y + y0 ) / ( 1.0 + y0 ); // the step goes from y0 to 1, normalize so it is 0 to 1
}

// Function 58
vec4 raymarch(float world, vec3 from, vec3 increment)
{
	const float maxDist = 200.0;
	const float minDist = 0.1;
	const int maxIter = RAYMARCH_ITERATIONS;
	
	float dist = 0.0;
	
	float material = 0.0;
	
	float glow = 1000.0;
	
	for(int i = 0; i < maxIter; i++) {
		vec3 pos = (from + increment * dist);
		float distEval = distf(int(world), pos, material);
		
		if (distEval < minDist) {
			break;
		}
		
		#ifdef GLOW
		if (material == 3.0) {
			glow = min(glow, distEval);
		}
		#endif
		
		
		if (length(pos.xz) < 12.0 && 
			pos.y > 0.0 &&
			(from + increment * (dist + distEval)).y <= 0.0) {
			if (world == 0.0) {
				world = 1.0;
			} else {
				world = 0.0;
			}
		}
		dist += distEval;
	}
	
	
	if (dist >= maxDist) {
		material = 0.0;
	}
	
	return vec4(dist, material, world, glow);
}

// Function 59
float rayMarcher(vec3 ro, vec3 rd){
	float tot = 0.;
    for(int i=0;i<MAX_DIST;i++){
    	vec3 p = ro+rd*tot;
        float diff = SDF(p);
        tot+=diff;
        if(diff<EPSI || tot>float(MAX_DIST)){
        	tot = float(i)/float(MAX_DIST-500);
            break;
        }
    }
    return tot;
}

// Function 60
vec4 put_text_step_count(vec4 col, vec2 uv, vec2 pos, float scale, int count)
{
	float unit = asp * scale * 0.1;
    float h = 0.;
    vec2 sc = vec2(unit, unit*0.8);
    
    int d = count % 10;
    int t = count / 10;
    
    h = max(h, word_map(uv, pos+vec2(unit*0.35, 0.), 48+d, sc));
    
    if(t > 0)
    {
    	h = max(h, word_map(uv, pos, 48+t, sc));
    }
    
    col = mix(col, vec4(1.-vec3(h), 1.), h);
    
    return col;
}

// Function 61
vec3 raymarch(vec3 ro, vec3 rd) {
    float rl = PHI;
    vec2 sdmat = vec2(0.0);
    
    for (int i = 0; i < MAX_ITER; i++) {
        vec3 p = ro + rl * rd;
        sdmat = sdmat_scene(p);
        if (sdmat.x < PHI) { break; }
        rl += sdmat.x;
        if (rl >= MAX_DEPTH) { break; }
    }
    
    return vec3(rl, sdmat);
}

// Function 62
float raymarchwater(vec3 camera, vec3 start, vec3 end, float depth){
    vec3 pos = start;
    float h = 0.0;
    float hupper = depth;
    float hlower = 0.0;
    vec2 zer = vec2(0.0);
    vec3 dir = normalize(end - start);
    for(int i=0;i<318;i++){
        h = getwaves(pos.xz, ITERATIONS_RAYMARCH) * depth - depth;
        if(h + 0.01 > pos.y) {
            return distance(pos, camera);
        }
        pos += dir * (pos.y - h);
    }
    return -1.0;
}

// Function 63
void marcher(vec3 ro, vec3 rd, inout float d, inout float m, inout vec3 p, int steps, float sg) {
    for(int i=0;i<steps;i++)
    {
        p=ro+rd*d;
        vec2 ray = map(p, sg);
        if(abs(ray.x)<MIN_DIST||d>MAX_DIST)break;
        d+=i<64?ray.x*.5:ray.x;
        m =ray.y;
    }
}

// Function 64
vec4 ray_march(in vec3 ray_origin, in vec3 ray_direction)
{
    float total_distance_traveled = 0.0;
    const int NUMBER_OF_STEPS = 64;
    const float MINIMUM_HIT_DISTANCE = 0.001;
    const float MAXIMUM_TRACE_DISTANCE = 1000.0;

    for (int i = 0; i < NUMBER_OF_STEPS; ++i)
    {
        vec3 current_position = ray_origin + total_distance_traveled * ray_direction;
		float distance_to_closest = distance_to_closest_object(current_position);
        if (distance_to_closest < MINIMUM_HIT_DISTANCE) 
        {
            vec3 normal = calculate_normal(current_position);
            
            vec3 light_positions[3];
            light_positions[0] = vec3(1.0+sin(iTime)*5.0, -3.0+3.0*cos(iTime/3.0), 4.0 + 1.0 *sin(iTime/5.0));
            light_positions[1] = vec3(1.0-sin(iTime/2.0)*2.0, -1.0-cos(iTime/2.0), 7.0 + 1.0 -sin(iTime/4.0));
            light_positions[2] = vec3(2.0-sin(iTime/2.0)*2.0, -5.0-sin(iTime/4.0), 2.0 + 1.0 -sin(iTime/1.0));
            vec3 light_intensities[3];
            light_intensities[0] = vec3(0.8, 0.4, 0.4);
            light_intensities[1] = vec3(0.04, 0.9, 0.2);
            light_intensities[2] = vec3(0.1, 0.2, 0.8);
            vec3 direction_to_view = normalize(current_position - ray_origin);float fresnel_base = 1.0 + dot(direction_to_view, normal);
            float fresnel_intensity = 0.04*pow(fresnel_base, 2.0);
            float fresnel_shadowing = pow(fresnel_base, 8.0);            
            float fresnel_supershadowing = pow(fresnel_base, 40.0);      
            float fresnel_antialiasing = 4.0*pow(fresnel_base, 8.0);
            float attenuation =  pow(total_distance_traveled,2.0)/150.0;
            
            vec3 col = vec3(0.0);
            
            for (int j = 0; j < 3; j++)
            {
                vec3 direction_to_light = normalize(current_position - light_positions[j]);
                vec3 light_reflection_unit_vector =
                	 reflect(direction_to_light ,normal);                

                float diffuse_intensity = 0.6*pow(max(0.0, dot(normal, direction_to_light)),5.0);            
                float ambient_intensity = 0.2;            
                float specular_intensity = 
                    1.15* pow(clamp(dot(direction_to_view, light_reflection_unit_vector), 0.0,1.0), 90.0);
                float backlight_specular_intensity =             
                    0.01* pow(clamp(dot(direction_to_light, light_reflection_unit_vector),0.0,1.0), 3.0); 
                
                
            	vec3 colFromLight = vec3(0.0);
                colFromLight += vec3(0.89, 0.35, 0.15) * diffuse_intensity;
                colFromLight += vec3(0.3, 0.1, 0.1) * ambient_intensity;
                colFromLight += vec3(1.0) * specular_intensity;            
                colFromLight += vec3(1.0,0.5,0.5) * backlight_specular_intensity;            
                colFromLight += vec3(1.0, 0.1, 0.2) * fresnel_intensity;
                colFromLight -= vec3(0.0, 1.0, 1.0) * fresnel_shadowing ;
                colFromLight -= vec3(0.0, 1.0, 1.0) * fresnel_supershadowing ;
                colFromLight += vec3(.3, 0.1, 0.1) - attenuation ; 
               //	colFromLight *= 1.6;
               // colFromLight *= sqrt(light_intensities[j]);
                col += colFromLight;
            }
            return vec4(col, 1.0-fresnel_antialiasing);
        }

        if (total_distance_traveled > MAXIMUM_TRACE_DISTANCE)
        {
            break;
        }
        total_distance_traveled += distance_to_closest;
    }
    return vec4(0.0);
}

// Function 65
vec3 raymarch_visualize(vec2 ro, vec2 rd, vec2 uv)
{
	//ro - ray origin
	//rd - ray direction

	vec3 ret_col = vec3(0.0);

	const float epsilon = 0.001;
	float t = 0.0;
	for(int i = 0; i < 10; ++i)
	{
		vec2 coords = ro + rd * t;
		float dist = field(coords);
		ret_col += visualize_region(uv, dist, coords);
		t += dist;
		if(abs(dist) < epsilon)
			break;
	}

	//return t;
	return ret_col;
}

// Function 66
float ramp_step(float x, float a, float ea)
{
    return clamp((a - x) / ea + 0.5, 0.0, 1.0);
}

// Function 67
float raymarch(vec3 cam_pos, vec3 march_dir, float t) {
    float depth = MIN_DIST;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = scene(cam_pos + depth * march_dir, t);
        if (dist < EPS) {
            return depth;
        }
        depth += dist;
        if (depth >= MAX_DIST) {
            return MAX_DIST;
        }
    }
    return MAX_DIST;
}

// Function 68
vec3 RayMarchCloud(Ray ray, vec3 sunDir, vec3 bgColor)
{
    vec3 rayPos = ray.o;
    rayPos += ray.dir * (kCloudHeight - rayPos.y) / ray.dir.y;
    
    float dl = 1.0;
    float scatter = 0.0;
    vec3 t = bgColor;
    for(int i = 0; i < RAYMARCH_CLOUD_ITER; i++) {
        rayPos += dl * ray.dir;
        float dens = SmoothNoise(vec3(0.05, 0.001 - 0.001 * iTime, 0.1) * rayPos - vec3(0,0, 0.2 * iTime)) * 
            SmoothNoise(vec3(0.01, 0.01, 0.01) * rayPos);
        t -= 0.01 * t * dens * dl;
        t += 0.02 * dens * dl;
	}
    return t;
}

// Function 69
float march(vec3 ro, vec3 rd, out vec4 outMaterial, out vec4 outInscatterTransmittance)
{
    float t = 0.001;
 	float d = 0.0;
    
    outInscatterTransmittance = oz.yyyx;
    
    for(int i = NON_CONST_ZERO; i < ITER; ++i)
    {
        float coneWidth = kPixelConeWithAtUnitLength * t;
        
        vec3 posWS = ro + rd*t;
        d = fSDF(posWS, true, outMaterial);
        
        if(outMaterial.x == kMatFire && d < coneWidth)
        {
            float mediumD = d;
            d = max(0.01, abs(d))*(s_pixelRand*0.5 + 0.75) + coneWidth;
            
            outInscatterTransmittance = 
                computeVolumetricLighting(outMaterial, mediumD, d, outInscatterTransmittance);
        }
        
        t += d;
        
        if(i >= ITER - 1)
        {
            t = kMaxDist;
        }              
        

        if(d < coneWidth || t >= kMaxDist)
        {
            break;
        }
    }
      
    return t;
}

// Function 70
float expStep( float x, float k, float n )
{
    return exp( -k*pow(x,n) );
}

// Function 71
vec3 raymarch(in vec3 from, in vec3 dir) 

{
	float ey=mod(t*.5,1.);
	float glow,eglow,ref,sphdist,totdist=glow=eglow=ref=sphdist=0.;
	vec2 d=vec2(1.,0.);
	vec3 p, col=vec3(0.);
	vec3 origdir=dir,origfrom=from,sphNorm;
	
    for (int i=0; i<RAY_STEPS; i++) {
		if (d.x>det && totdist<6.0) {
			p=from+totdist*dir;
			d=de(p);
			det=detail*(1.+totdist*60.)*(1.+ref*5.);
			totdist+=max(detail,d.x); 
			if (d.y<.5) glow+=max(0.,.02-d.x)/.02;
		}
	}
	vec3 ov=normalize(vec3(1.,.5,1.));
	vec3 sol=dir+lightdir;
    float l=pow(max(0.,dot(normalize(-dir*ov),normalize(lightdir*ov))),1.5)+sin(atan(sol.x,sol.y)*20.+length(from)*50.)*.002;
    totdist=min(5.9,totdist);
    p=from+dir*(totdist-detail);
    vec3 backg=.4*(1.2-l)+LIGHT_COLOR*l*.75;
	backg*=AMBIENT_COLOR*(1.-max(0.2,dot(normalize(dir),vec3(0.,1.,0.)))*.2);
	float fondo=0.;
	vec3 pp=p*.5+sin(t*2.)*.5;
    for (int i=0; i<15; i++) {
        fondo+=clamp(0.,1.,texture1(pp+dir*float(i)*.02))*max(0.,1.-exp(-.03*float(i)));
    }
    vec3 backg2=backg*(1.+fondo*(LIGHT_COLOR)*.75);
    if (d.x<.01) {
        vec3 norm=normal(p);
		col=mix(light(p-abs(d.x-det)*dir, dir, norm, d.y),backg,1.-exp(-.3*totdist*totdist)); 
		col = mix(col, backg2, 1.0-exp(-.02*pow(abs(totdist),2.)));
	} else { 
		col=backg2;
	}
	vec3 lglow=LIGHT_COLOR*pow(abs(l),30.)*.5;
    col+=glow*(.3+backg+lglow)*.005;
	col+=lglow*min(1.,totdist*totdist*.03)*1.2;
	return col; 
}

// Function 72
vec3 march(vec3 eye, vec3 dir, float maxd){
    
    float d, i; vec2 ind;
    for (; i<100. && d < maxd; i++){
        vec3 p = eye + dir * d;
        ind = street(p);
        if (abs(ind.x) < 0.001 * d)break;
        d += ind.x;
    }
    return vec3(d, ind.y, pow(i/60., 2.5));
}

// Function 73
vec2 RayMarch(vec3 ro, vec3 rd, inout vec3 gc) {
	float dO=0.;
    vec2 dm;
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        dm = GetDist(p);
        // Ripped from Shaw
        float at = .03 / (1. + dm.x * dm.x * 100.);
        vec3 gcc = vec3(1., .5, 0.); // hsv2rgb(vec3(Hash(dm.y),0.8,1.)); // 
        gc += gcc * at;       
        float dS = dm.x;
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }
    
    return vec2(dO,dm.y);
}

// Function 74
float RayMarch(vec3 ro, vec3 rd, int PMaxSteps)
{   float t = 0.; 
    vec3 dS=vec3(9999.0,-1.0,-1.0);
    float marchCount = 0.0;
    vec3 p;
    
    #define DISTANCE_BIAS 0.75
    float minDist = 9999.0; 
    
    for(int i=0; i <= PMaxSteps; i++) 
    {  	p = ro + rd*t;
        dS = GetDist(p);
        t += dS.x;
        if ( abs(dS.x)<MIN_DIST  || i == PMaxSteps)
            {mObj.hitbln = true; minDist = abs(t); break;}
        if(t>MAX_DIST)
            {mObj.hitbln = false;    minDist = t;    break; } 
        marchCount++;
    }
    mObj.dist = minDist;
    mObj.id_color = dS.y;
    mObj.marchCount=marchCount;
    mObj.id_material=dS.z;
    mObj.normal=GetNormal(p);
    mObj.phit=p;
    return t;
}

// Function 75
vec3 marchScene(vec3 ro, vec3 rd
){vec2 m=1.-iMouse.xy/iResolution.xy;//m could be identical to ScreenSpace.xy
 //[l] lazily coutneracts overstepping for higher precision in its gradient descent.
 //     by intentional understepping, assuming lipschits constant >1,
 float l=m.x;//set by iMouse.x
 //dynamic number of iterations lacks backwards compatibility.
 float iterMax=450.*m.y;//set by Mouse.y
 //float EPS=0.001; ish.
 //loop accumulators:
 float t=.0,     // t=distanceToCamera (without epsilon)
       g=.0;     // VolumeMarched smoothstep glow: +=lerp((exp(-SquaredDistanceToGlowCenter)))
 vec2  r=vec2(0);// .x=distanceToSurface .y=MaterialID
 for (float i = 0.; i < iterMax; i++){
     vec3 p=ro + rd * t;//pointOnRay
     vec2 r = map(p);   //shortest euclideanPointDistance to distanceField of pointOnRay.
 #ifdef DynamicEps
     if (t>FAR|| log(t*t*EPS/r.x)>0.) break;//zFar || logEps        exits
     //above is very basic logeps, IFF (scene is scaled properly) it relpaces the line below.
 #else
     if (t>FAR||         EPS>r.x    ) break;//zFar || zNearSurface  exits
 #endif
     g += smoothstep(0.,1.,1.2*exp(-dot(p,p)));//increment glow
     t += r.x*l ;}//march along ray
 return vec3(t,g,r.y);}

// Function 76
bool MarchReflectionRay( vec3 start, vec3 dir, out vec3 pos CACHEARG )
{
    // same as MarchCameraRay except lower tolerances because artifacts won't be very noticeable.
    // no bounds checking because reflection rays always start near the surface
    
    // assumes dir is normalized
    pos = start + dir * REFLECTION_EPSILON;
    
    float prevMarchDist = REFLECTION_EPSILON;
    float prevSurfaceDist = BlobDist( start CACHE );
    
    for ( int i = 0; i < MAX_REFLECTION_RAYMARCH_STEPS; i++ )
    {
        float surfaceDist = BlobDist( pos CACHE );
        if ( surfaceDist < EPSILON )
        {
            if ( surfaceDist < 0.0 )
            	pos = RefineSurfacePos( pos, dir, surfaceDist, prevSurfaceDist, prevMarchDist, REFLECTION_EPSILON CACHE );
            return true;
        }
        
        float gradientAlongRay = (prevSurfaceDist - surfaceDist) / prevMarchDist;
        float safeGradient = max( gradientAlongRay, MIN_GRADIENT_FOR_REFLECTION_RAYS );
        
        float addDist = (surfaceDist + REFLECTION_EPSILON) / safeGradient;
        prevMarchDist = addDist;
        
        prevSurfaceDist = surfaceDist;
        pos += dir * addDist;
        
        vec3 relPos = pos - BLOB_BOUNDING_CENTER;
        relPos *= BLOB_BOUNDING_SCALE;
        if ( dot( relPos, relPos ) > BLOB_BOUNDING_RADIUS_SQR )
            return false;
    }
    
    return true;
}

// Function 77
float StepThruEmptyVoxels(vec3 ro, vec3 rd, float dlimit)
{
    ivec3 n;
//    if (Condition(voxid(ro + rd * 1e-3))) 
//        return 0.;
    float t = ScanDDA3(ro, ro + rd * dlimit, n);
    //vec3 hp = ro + rd * t;
    return t;
}

// Function 78
vec3 march(vec2 uv, float t) {
    origin = vec3(uv - vec2(0.5, 0.25), 1.0);
    origin.xy*=0.7;
    
    #define ni(x) smootherstep(0.0, 1.0, max(0.0, min(1.0, x)))
    
    orbit = 1.0 - ni((t-50.0)*0.04);
    sea = ni((t-100.0)*0.05);
	seabrite = ni((t-113.0)*0.3);
    buildings2 = 1.0-orbit; // TODO simplify?
    buildings = buildings2-sea;
	startmove = ni((t-7.0)*0.045);
	surface = ni((t-140.0)*0.1);
	spaceboost = ni((t-31.0)*0.18);
	
    //float back = ni((t-142.0)*0.08);
    back = ni((t-159.0)*0.1); //147
    end = ni((t-185.0)*0.2);
    sea -= back; //ni((t-150.0)*0.1);
    intro = ni((t-8.0)*0.1);
    intro -= back;
	surface -= back*0.9;
	
	struckfinal = ni((t-175.0)*0.15);
    
    bust = ni((t-30.0)*0.2);
	
	zofs = t*0.5 + t*buildings2*0.5      - pow(0.015*t, 4.0) - startmove*6.0 - buildings*40.0;;
    
    //intro
    ryz = orbit*0.4 - 0.2 + buildings*0.15 + 0.5;
    rxz = orbit*0.7 + sin(t*0.1)*0.1*sea + sea*0.1;
    
    ryz += buildings*(sin(t*0.1)*0.2-0.1) + 0.1 * surface;
   
    rxz += buildings*0.8;
    rxz += cos(t*0.1)*0.3*sea  - 0.2*sea;
    ryz -= 0.4 + 0.4*sea + cos(t*0.1)*0.4*sea;
    rxy = -0.2*sea;
    
    pR(origin.xy, rxy * intro);
    pR(origin.yz, ryz * intro);
    pR(origin.xz, rxz * intro);
    
    vec3 dir = normalize(origin);
    
    origin.x += 0.25;
    origin.y += 5.0 - sea*7.0;
    
    p = origin;
    accum = vec3(0.);
	//int lim = 40 + int(spaceboost*spaceboost)*40;
    for (iii=0;iii<80;iii++) {
		//if (iii > lim) break;
        d = field(p, t);
        accum += d;
		//accum *= mix(0.85, 1.0, spaceboost);
            
    	p += dir * 1.0e-3 * max(0.005, 1.0/length(d));
    }
	
	float dist = length(p-origin);
	//accum *= 6.0 - 5.0*spaceboost;
    
    accum *= mix(max(0., 1.0 - 0.1*sqrt(dist)), 1.0, buildings);
	accum *= 1.0 + surface*abs(sin(dist * 0.1))*2.0*cos(p.y*0.6);
	//accum *= pow(min(1.0, max(0.0, dist*(0.15))), 2.0);
	//accum *= min(1.0, max(0.0, pow(dist - 10.0, 1.0)*0.05))*4.0;
	//accum = mix(max(vec3(0.0), accum - vec3(1.0) * max(0.0, sqrt(dist)*5e-2))*1.5, accum, spaceboost);
    accum /= 1.0+buildings*0.4;
    return accum + vec3(pow(max(accum.x, max(accum.y, accum.z)), 2.0)); // boost the saturated colors to white
}

// Function 79
float raymarch(vec3 ori, vec3 dir) {
    float t = 0.;
    for(int i = 0; i < 256; i++) {
    	float dst = dstScene(ori+dir*t);
        if(dst < .001 || t > 256.)
            break;
        t += dst * .75;
    }
    return t;
}

// Function 80
void marchRay(inout Ray ray, inout vec4 colour, inout int steps, in int maxSteps) {
    bool inside = false; // are we inside or outside the glass object
    vec4 impact = vec4(1.0); // This decreases each time the ray passes through glass, darkening colours
    bool hit = false;
#ifdef DEBUG   
vec4 debugColour = vec4(1, 0, 0, 1);
#endif
    for (int i=0; i<kMAXITERS; i++) {
        // Get distance to nearest surface
        float d = sceneDist(ray);
        
        // Step half that distance along ray (helps reduce artefacts)
        ray.origin += ray.dir * abs(d);  
        
        if (abs(d) < kEPSILON) {
            // colision
    
            hit = true;
            /*
            if ( ray.origin.y < kEPSILON ) {
                // ray hit floor
                impact *= vec4(0.6,0.6,0.6,1.0);
                ray.dir = reflect(ray.dir, vec3(0,1,0));
            	// Intersection count inc, break if over limit else next iteration
                if (nextStepIsFinal(steps, maxSteps)) { break; } else { continue; }
            }
            */
            // Get the normal, then clamp the intersection to the surface
    		vec3 n = normal(ray);
            clampToSurface(ray, d, n);
#ifdef DEBUG
debugColour.rgb = n;
//break;
#endif
            
            if (inside) {
                // refract glass -> air
            	ray.dir = refract(-n, -ray.dir, 1.0/kREFRACT);
            	impact *= vec4(0.4, 0.5, 0.9, 1.0);
          
                
            } else {
                // refract air -> glass
                // Calulcate a fresnel term for reflections
                float fresnel = min(1., dot(ray.dir, n) + 1.0);
       			fresnel = pow(fresnel, 2.);
                
                // Mix the reflection in, according to the fresnel term
    			colour = mix(
                    colour, 
                    texture(iChannel1, reflect(ray.dir, n) * kFLIPY), 
                    vec4(fresnel) * impact);
    			
                // refract the ray
            	ray.dir = refract(ray.dir, n, kREFRACT);
            }
            
            // Intersection count inc, break if over limit
            if (nextStepIsFinal(steps, maxSteps)) { break; }
            
            // Step 2x epsilon into object along normal to ensure we're beyond the surface
            // (prevents multiple intersections with same surface)
            ray.origin += (inside ? n : -n) * kEPSILON * 2.0;
            
            // Flip in/out status
            inside = !inside;
        }
    }
    
    // So far we've traced the ray and accumulated reflections, now we need to add the background.
    colour += texture(iChannel0, ray.dir * kFLIPY) * impact;// / float(steps+1);
#ifdef DEBUG
colour = debugColour;
//colour.rgb = vec3(float(steps)/8.);
//colour.rgb = ray.dir;
#endif
}

// Function 81
vec2 mySmoothStep(vec2 a, vec2 b, float t) {
    t = smoothstep(0.0, 1.0, t);
    return mix(a, b, t);
}

// Function 82
float Marches()
{
	return 50.0 + 40.0 * cos(iTime / 1.0);
}

// Function 83
vec2 smoothstepd( float a, float b, float x)
{
	if( x<a ) return vec2( 0.0, 0.0 );
	if( x>b ) return vec2( 1.0, 0.0 );
    float ir = 1.0/(b-a);
    x = (x-a)*ir;
    return vec2( x*x*(3.0-2.0*x), 6.0*x*(1.0-x)*ir );
}

// Function 84
float smootherstep(float a, float b, float r) {
    r = clamp(r, 0.0, 1.0);
    return mix(a, b, r * r * r * (r * (6.0 * r - 15.0) + 10.0));
}

// Function 85
void rayMarch(inout ray _r)
{
	// ray start point
	vec3 p = _r.o;
	vec2 d = vec2(0.);
	float ld = 0.;
		
	// traverse frustum
	for(float s=ncp; s<fcp; s+=step)
	{
		// step in front
		p += _r.d*step;
		
		// check hit
		d = map(p);
		if(!_r.h && d.x<0.)
		{
			_r.h = true;
			
			// interpolate p
			float i = step*ld/(ld-d.x);
			_r.hp = p;//_r.o + _r.d * (s-step+i);            
            break;
		}
		
		//last d
		ld = d.x;
	}
}

// Function 86
vec2 RAYMARCH_distanceField( vec3 o, vec3 dir)
{
    //From Inigo Quilez DF ray marching :
    //http://www.iquilezles.org/www/articles/raymarchingdf/raymarchingdf.htm
    float tmax = GEO_MAX_DIST;
    float t = 0.0;
    float dist = GEO_MAX_DIST;
    for( int i=0; i<50; i++ )
    {
	    dist = DF_composition( o+dir*t ).d;
        if( abs(dist)<0.0001 || t>GEO_MAX_DIST ) break;
        t += dist;
    }
    
    return vec2( t, dist );
}

// Function 87
float Steps(vec3 p)
{
    float s = FarClip;
    vec3  o = vec3(0.0);
    float w = 14.0;
    
    for(int i = 0; i < 10; ++i)
    {
    	s = min(s, Box(p + o + vec3(0.0, 0.0,- w), vec3(8.0, 0.25, w)));  
        o += vec3(0.0, -0.35, -1.0);
        w -= 1.0;
    }
    
    float g = HexPrism(RotY(p + vec3(-8.5, 0.0, -14.0), PI * 0.5), vec2(3.4, 0.5), 0.2);
    float b = Box(p - vec3(8.5, 0.0, -1.5), vec3(0.5, 1.0, 1.5));
    
    return min(s, min(g, b));
}

// Function 88
vec4 raymarch( vec3 ro, vec3 rd, vec3 bgcol, ivec2 px )
{
	vec4 sum = vec4(0);
	float dt = .01,
         den = 0., _den, lut,
           t = intersect_sphere( ro, rd, vec3(0), BR );
    if ( t == -1. ) return vec4(0); // the ray misses the object 
    t += 1e-5;                      // start on bounding sphere
    
    for(int i=0; i<500; i++) {
        vec3 pos = ro + t*rd;
        if(   sum.a > .99               // end if opaque or...
           || length(pos) > BR ) break; // ... exit bounding sphere
                                    // --- compute deltaInt-density
        _den = den; den = map(pos); // raw density
        float _z = z;               // depth in object
        lut = LUTs( _den, den );    // shaped through transfer function
        if( lut > .0                // optim
          ) {                       // --- compute shading                  
#if 0                               // finite differences
            vec2 e = vec2(.3,0);
            vec3 n = normalize( vec3( map(pos+e.xyy) - den,
                                      map(pos+e.yxy) - den,
                                      map(pos+e.yyx) - den ) );
         // see also: centered tetrahedron difference: https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
            float dif = clamp( -dot(n, sundir), 0., 1.);
#else                               // directional difference https://www.iquilezles.org/www/articles/derivative/derivative.htm
         // float dif = clamp((lut - LUTs(_den, map(pos+.3*sundir)))/.6, 0., 1. ); // pseudo-diffuse using 1D finite difference in light direction 
            float dif = clamp((den - map(pos+.3*sundir))/.6, 0., 1. );             // variant: use raw density field to evaluate diffuse
#endif
/*
            vec3  lin = vec3(.65,.7,.75)*1.4 + vec3(1,.6,.3)*dif,          // ambiant + diffuse
                  col = vec3(.2 + dif);
            col = mix( col , bgcol, 1.-exp(-.003*t*t) );   // fog
*/            
            vec3 col = exp(- vec3(3,3,2) *(1.-z));     // dark with shadow
         // vec3 col =   exp(- vec3(3,3,2) *(.8-_z));  // dark with depth
                   //      *  exp(- 1.5 *(1.-z));
            sum += (1.-sum.a) * vec4(col,1)* (lut* dt*5.); // --- blend. Original was improperly just den*.4;
        }
        t += dt;  // stepping
    }

    return sum; 
}

// Function 89
vec4 rayMarch(inout vec3 p, in vec3 rd, out vec3 dists) {
  float dS = 99., d = 0., minDS = dS, steps = 0.;
  for (int i = 0; i < MAX_STEPS; i++) {
    steps += 1.;
    dS = mapWDists(p, dists);
    minDS = min(minDS, abs(dS));
    d += dS;
    p = p + rd * dS;
    if ((0. <= dS && dS < SURF_DIST) || d > MAX_DIST) break;
  }
  return vec4(d, dS, minDS, steps);
}

// Function 90
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

// Function 91
Hit raymarch(Ray ray) {
 
    vec3 p = ray.ori;
    float t = 0.;
    int id = -1;
    
    for(int i = 0; i < MAX_ITERATIONS; i++) {
     
        Dist d = distScene(p);
        p += ray.dir * d.dist;
        
        if(d.dist <= MIN_DISTANCE) {
         
            t = d.dist;
            id = d.id;
            
            break;
            
        }
        
    }
    
    return Hit(p,Dist(t,id));
    
}

// Function 92
void Step (int mId, out vec3 rm, out vec3 vm, out vec4 qm, out vec3 wm)
{
  mat3 mRot, mRotN;
  vec3 rmN, vmN, wmN, dr, dv, rts, rtsN, rms, vms, fc, am, wam, dSp;
  float farSep, rSep, grav, dt;
  grav = 20.;
  dt = 0.01;
  rm = Loadv4 (4 + 4 * mId).xyz;
  vm = Loadv4 (4 + 4 * mId + 1).xyz;
  qm = Loadv4 (4 + 4 * mId + 2);
  wm = Loadv4 (4 + 4 * mId + 3).xyz;
  if (nStep < 50.) return;
  mRot = QtToRMat (qm);
  farSep = length (blkGap * (blkSph - 1.)) + 1.;
  am = vec3 (0.);
  wam = vec3 (0.);
  for (int n = 0; n < nBlock; n ++) {
    rmN = Loadv4 (4 + 4 * n).xyz;
    if (n != mId && length (rm - rmN) < farSep) {
      vmN = Loadv4 (4 + 4 * n + 1).xyz;
      mRotN = QtToRMat (Loadv4 (4 + 4 * n + 2));
      wmN = Loadv4 (4 + 4 * n + 3).xyz;
      for (int j1 = 0; j1 < nSiteBk; j1 ++) {
        rts = mRot * RSite (j1);
        rms = rm + rts;
        vms = vm + cross (wm, rts);
        dv = vms - vmN;
        fc = vec3 (0.);
        for (int j2 = 0; j2 < nSiteBk; j2 ++) {
          rtsN = mRotN * RSite (j2);
          dr = rms - (rmN + rtsN);
          rSep = length (dr);
          if (rSep < 1.) fc += FcFun (dr, rSep, dv - cross (wmN, rtsN));
        }
        am += fc;
        wam += cross (rts, fc);
      }
    }
  }
  for (int j = 0; j < nSiteBk; j ++) {
    rts = mRot * RSite (j);
    dr = rm + rts;
    rSep = abs (dr.y);
    if (rSep < 1.) {
      fc = FcFun (vec3 (0., dr.y, 0.), rSep, vm + cross (wm, rts));
      am += fc;
      wam += cross (rts, fc);
    }
  }
  dSp = blkGap * blkSph;
  wam = mRot * (wam * mRot / (0.5 * (vec3 (dot (dSp, dSp)) - dSp * dSp) + 1.));
  am.y -=  grav;
  vm += dt * am;
  rm += dt * vm;
  wm += dt * wam;
  qm = normalize (QtMul (RMatToQt (LpStepMat (0.5 * dt * wm)), qm));
}

// Function 93
Object rayMarching(vec3 origin, vec3 direction, out vec3 p)
{
    float currentDistance = 0.0;

    Object res=Object(0.0,vec3(0.0,0.0,0.0));

    float lowerThreshold=0.001;
    float upperThreshold=30.0; 
    for(int i=0;i<maxIterations;i++)
    {
        p=origin+direction*currentDistance;
	Object obj=map(p);
	float distanceToClosestSurface = obj.distance;
        currentDistance += distanceToClosestSurface;
        if(distanceToClosestSurface<lowerThreshold)
	{
	  return obj;
	  break;
	}
        
        if(distanceToClosestSurface>upperThreshold)
        {
            currentDistance=0.0;
            break;
        }
    }

    return res;
}

// Function 94
void march(vec3 origin, vec3 dir, out float t, out int hitObj) {
    t = 0.001;
    for(int i = 0; i < RAY_STEPS; ++i) {
        vec3 pos = origin + t * dir;
    	float m;
        sceneMap3D(pos, m, hitObj);
        if(primitives[hitObj].primitiveId == CUBE)
        {
        	if(abs(m) < 0.01) 
                return;
        }
        else if(primitives[hitObj].primitiveId == SQUARE_PLANE)
        {
            if(abs(m) < 0.01) 
            {
            	return;
            }
        	
        }
        t += m;
    }
    t = -1.0;
    hitObj = -1;
}

// Function 95
vec4 march(vec3 ro, vec3 rd) {
    vec3 r = ro;
    bool hit = false;
    float t = 0.;
    float q = 0.;
   	float tf = 0.;
    int j = 0;
    vec3 cdl = vec3(0);
    bool cht = false;
    for ( int i = 0; i < 90; ++i ) {
        float df = map(r);
        t= ((r-ro)/rd).r;
        if (df < pr*10.) {
        	q += df;
        }
        if ((df < pr&&df>-.5)||t>50.) {
            if (df < pr) {
                
            hit = true;
            }
            break;
        }
        if (df<-.5) {
			df = -df/2.;
        }
        float cl = cld(r);
        tf += cl;    
        if (cl < pr) {
            j++;
            vec3 a = tnt(iChannel0, (r.zy+r.xz+r.xy+vec2(iTime*0.)+float(j))/10., 1.).rrr*smoothstep(6., 0., clamp(tf, 0., 6.)/6.);
        	cdl += a*sqrt(float(j))*1.5*pow(smoothstep(0., 1., clamp(float(j)/15., 0., 1.)), 1./2.5);
            cht = true;
        }
        r+=rd*df;
    }
    vec3 O = vec3(1.2, 1.1, 1);
    vec3 COL = O;
    
    if (hit) {
   		vec3 sand = .5*(tnt(iChannel1, r.xz+noise(r)/2., 1.)+tnt(iChannel2, r.xz, 2.).rrr*vec3(1.2, 1.1, 0)/2.)*vec3(1.1, 1.2, 1);
        COL = sand*.8/(1.-height(vec3(r.xz/10.,0))*.6);
        
    }
    COL += max(vec3(0.),cdl/float(j)*.2)*clamp((t-tf), 0., 1.);
    float fog = clamp((50.-t)/(50.-30.), 0., 1.);
    return vec4(mix(COL,O, 1.-fog), clamp(q/4., 0., 1.));
}

// Function 96
vec3 raymarchClouds( const in vec3 ro, const in vec3 rd, const in vec3 bgc, const in vec3 fgc, const in float startdist, const in float maxdist, const in float ani ) {
    // dithering	
	float t = startdist+CLOUDSCALE*0.02*hash(rd.x+35.6987221*rd.y+time);//0.1*texture( iChannel0, fragCoord.xy/iChannelResolution[0].x ).x;
	
    // raymarch	
	vec4 sum = vec4( 0.0 );
	for( int i=0; i<64; i++ ) {
		if( sum.a > 0.99 || t > maxdist ) continue;
		
		vec3 pos = ro + t*rd;
		float a = cloudMap( pos, ani );

        // lighting	
		float dif = clamp(0.1 + 0.8*(a - cloudMap( pos + lig*0.15*CLOUDSCALE, ani )), 0., 0.5);
		vec4 col = vec4( (1.+dif)*fgc, a );
		// fog		
	//	col.xyz = mix( col.xyz, fgc, 1.0-exp(-0.0000005*t*t) );
		
		col.rgb *= col.a;
		sum = sum + col*(1.0 - sum.a);	

        // advance ray with LOD
		t += (0.03*CLOUDSCALE)+t*0.012;
	}

    // blend with background	
	sum.xyz = mix( bgc, sum.xyz/(sum.w+0.0001), sum.w );
	
	return clamp( sum.xyz, 0.0, 1.0 );
}

// Function 97
vec4 ray_march_vol(vec3 p, vec3 r)
{
    vec4 color = vec4(0., 0., 0., 0.);
    vec4 background = vec4(0.1);
    for(int i = 0; i < 90; i++)
    {
        vec3 tcolor = COL(p);
        float density = length(tcolor);
        float noise  =(1.+0.5*rand());
        float d = 4.*DX*exp(-2.*min(density,2.))*noise;
        float opacity = 1.-exp(-OPACITY*density*d);
        float newa = max(color.w + (1. - color.w)*opacity,0.0001);
        color.xyz = (color.xyz*color.w + (1.-color.w)*opacity*tcolor)/newa;
        color.w = newa;
        if(1. - newa < 0.02) break;
        p += r*d;
    }
    return background + color;
}

// Function 98
float stepize(float a, int nb)
{
 	a = float(int(a*float(nb)))/float(nb);   
    return a;
}

// Function 99
float grid_step(float t)
{
  t = min(t, 1.0-t);
  return smoothstep(0.0, GRID_K, t);
}

// Function 100
Hit raymarch(vec3 rayOrigin, vec3 rayDirection){

    float currentDist = INTERSECTION_PRECISION * 2.0;
    float rayLength = 0.;
    Model model;

    for(int i = 0; i < NUM_OF_TRACE_STEPS; i++){
        if (currentDist < INTERSECTION_PRECISION || rayLength > MAX_TRACE_DISTANCE) {
            break;
        }
        model = map(rayOrigin + rayDirection * rayLength);
        currentDist = model.dist;
        rayLength += currentDist * (1. - model.underStep);
    }

    bool isBackground = false;
    vec3 pos = vec3(0);
    vec3 normal = vec3(0);

    if (rayLength > MAX_TRACE_DISTANCE) {
        isBackground = true;
    } else {
        pos = rayOrigin + rayDirection * rayLength;
        normal = calcNormal(pos);
    }

    return Hit(
        model,
        pos,
        isBackground,
        normal,
        rayOrigin,
        rayLength,
        rayDirection
    );
}

// Function 101
vec2 RayMarch(vec3 ro, vec3 rd) {
	float dO=0.;    
    vec2 m;
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        m = GetDist(p);
        float dS = m.x;
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }    
    return vec2(dO,m.y);
}

// Function 102
Hit march(vec3 origin, vec3 rayDir, float maxDist) {
    vec3 p;
    float len = 0.;
    float dist = 0.;
    Model model;

    for (float i = 0.; i < 100.; i++) {
        len += dist;
        p = origin + len * rayDir;
        model = map(p);
        dist = model.d;
        if (abs(model.d) / len < .0002) {
            break;
        }
        if (len >= maxDist) {
            len = maxDist;
            model.id = 0;
            break;
        }
    }   

    return Hit(model, p, len);
}

// Function 103
float march(vec3 rayOrigin, vec3 rd) {
    const float minDist = .05;
	float t = 0.0, dt;
	for(int i=0; i<128; i++){
		dt = map(rayOrigin + rd*t);
		if(dt<minDist || t>150.){ break; } 
		t += dt*0.75;
	}
    return (dt < minDist) ? t : -1.;
}

// Function 104
Hit raymarch(Ray ray, bool shadow) {
 
    vec3 p = ray.ori;
    int id = -1;
    
    for(int i = 0; i < MAX_ITERATIONS; i++) {
     
        Dst scn = dstScene(p, shadow);
        p += ray.dir * scn.dst * .75;
        
        if(scn.dst < MIN_DISTANCE) {
         
            id = scn.id;
            break;
            
        }
        
    }
    
    return Hit(p,id);
    
}

// Function 105
vec4 MarchLight(in Ray ray) {//remaining, total, weighted min, steps
    ray.pos += LIGHT_eps*Normal(ray.pos);
    ray.vel = LIGHT_dir(ray);//point towards light
    float d;
    bool exit = false;
    
	vec4 dist = vec4(-1, 0, MARCH_maxl, 0);//remaining, total, weighted min, steps
    for (int i = 0; i < LIGHT_itr /*&& d > MARCH_eps && dist.x < MARCH_maxl*/; i++) {
    	d = Map(ray.pos).x;
        if (d > LIGHT_dist(ray)) {//touches light
            d = LIGHT_dist(ray);
            exit = true;
        }
        dist.y += d;
        dist.z = min(d/dist.y, dist.z);
        dist.w++;
        
        ray.pos += d*ray.vel;
        if (exit || !(d > MARCH_eps && dist.y < MARCH_maxl)) break;
    }
    dist.x = LIGHT_dist(ray);
	return dist/MARCH_norm.yywz;
}

// Function 106
void Step (int mId, out vec3 r, out vec3 v, out vec3 a, out float grp)
{
  vec4 p;
  vec3 dr, rSum, vSum;
  float nNeb, rLen, vMag, rMarg;
  p = Loadv4 (3 * mId);
  r = p.xyz;
  grp = p.w;
  v = Loadv4 (3 * mId + 1).xyz;
  a = vec3 (0.);
  vSum = vec3 (0.);
  rSum = vec3 (0.);
  nNeb = 0.;
  for (int n = 0; n < nBoid; n ++) {
    if (n != mId) {
      p = Loadv4 (3 * n);
      dr = r - p.xyz;
      rLen = length (dr);
      if (rLen < 1.) a += fSep * (1. / rLen - 1.) * dr;
      if (rLen < rFlok && grp == p.w) {
        rSum += p.xyz;
        vSum += Loadv4 (3 * n + 1).xyz;
        ++ nNeb;
      }
    }
  }
  if (nNeb > 0.) a -= fFlok * (r - rSum / nNeb) + fAln * (v - vSum / nNeb);
  dr = r - rLd;
  rLen = length (dr);
  if (rLen < rAttr) {
    a += ((1. - 2. * smoothstep (2., 3., rLen)) * fLead / max (rLen * rLen, 0.001)) * dr;
  }
  rMarg = 1.;
  dr = r;
  dr.xy -= vec2 ((hoopSz - hoopThk) * sign (r.x), hoopHt);
  dr = max (abs (dr) - vec3 (hoopThk, hoopSz, hoopThk), 0.) * sign (dr);
  rLen = length (dr);
  if (rLen < hoopThk + rMarg) a += fSep * ((hoopThk + rMarg) / rLen - 1.) * dr;
  dr = r;
  dr.y -= hoopHt + (hoopSz - hoopThk) * sign (r.y);
  dr = max (abs (dr) - vec3 (hoopSz, hoopThk, hoopThk), 0.) * sign (dr);
  rLen = length (dr);
  if (rLen < hoopThk + rMarg) a += fSep * ((hoopThk + rMarg) / rLen - 1.) * dr;
  dr = r;
  dr.y -= 0.25 * hoopHt;
  dr = max (abs (dr) - vec3 (hoopThk, 0.25 * hoopHt, hoopThk), 0.) * sign (dr);
  rLen = length (dr);
  if (rLen < hoopThk + rMarg) a += fSep * ((hoopThk + rMarg) / rLen - 1.) * dr;
  a += 0.05 * (vFly - length (v)) * normalize (v);
  v += dt * a;
  r += dt * v;
  rLen = length (r);
  if (rLen > regSz) {
    if (dot (r, v) > 0.) v = 0.9 * reflect (v, r / rLen);
    r *= (regSz - 0.05) / rLen;
  }
  if (r.y < 0.) {
    r.y = 0.05;
    if (v.y < 0.) v = 0.9 * reflect (v, vec3 (0., 1., 0.));
  }
}

// Function 107
float march(in vec3 ro, in vec3 rd)
{
	float precis = 0.005;
    float h=precis*2.0;
    float d = 0.;
    for( int i=0; i<ITR; i++ )
    {
        if( abs(h)<precis || d>FAR ) break;
        d += h;
	    float res = map(ro+rd*d);
        h = res;
        #ifdef SHOW_CELLS 
        rd.xy *= d*0.00001+.996;
        #endif
    }
	return d;
}

// Function 108
vec3 march(vec2 uv, vec3 camPos) {
    mat4 vm = viewMatrix(camFwd, camUp);
    vec3 ray = (vm * vec4(calcRay(uv, 80.0, iResolution.x / iResolution.y), 1.0)).xyz;
    vec4 color = vec4(BACKGROUND, CAM_FAR);
    vec3 waterColor;
    marchWater(camPos, ray, color);
    marchObjects(camPos, ray, color.w, color);
    return color.rgb;
}

// Function 109
vec3 smoothstep_unchecked( vec3 x ) { return ( x * x ) * ( 3.0 - x * 2.0 ); }

// Function 110
vec4 raymarch(vec2 resolution, vec2 uv, vec4 start_data, mat4 camera_transform) {
    // Convert to range (-1, 1) and correct aspect ratio
    vec2 screen_coords = (uv - 0.5) * 2.0;
    screen_coords.x *= resolution.x / resolution.y;
    
    
    vec3 ray_start_position = camera_transform[3].xyz;
    
    vec3 ray_direction = normalize(vec3(screen_coords * LENS, 1.0));
    ray_direction = (camera_transform * vec4(ray_direction, 0.0)).xyz;
    
    
    float dist = start_data.a * 0.9;
    vec3 sample_point = ray_start_position + dist * ray_direction;
    
    float results = sample_world(sample_point);
    
    float tolerance = 0.0;
    
    for (int i=0; i<steps; i += 1) {
        dist += results*1.125;
        sample_point += ray_direction * results;
        results = sample_world(sample_point);
        
        // TODO: Derive from resolution, camera lens and distance
        ;
    	tolerance=4.*LENS/resolution.x*dist*dist;
        
        if (results < tolerance || dist > 5.0) {
        	break; 
        }
    }
    
    
    
    return vec4(sample_point, dist);
}

// Function 111
float smStep(float x)
{
    return 3.0*x*x-2.0*x*x*x;
}

// Function 112
float stepsize(vec3 p)
{
    float md = sqrt(min(min(
        sq(p - sphere1.xyz), 
        sq(p - sphere2.xyz)), 
        sq(p - sphere3.xyz)));
    return max(min_stepsize, abs(md - 1.0) * 0.667);
}

// Function 113
vec2 RayMarch(vec3 ro, vec3 rd)
{
    float dO = 0.0; //distance from origin
    vec2 retval = vec2(0);
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + dO*rd;	//march the ray
        retval = GetDist(p);
        float dS = retval.y;	//minimum distance from surface relative to p
        dO += dS;				//it is safe to march at least the distance dS without intersecting
        
        //approached surface close enough OR reached "infinity"
		if (dS < SURFACE_DIST || dO > MAX_DIST)
            break;
    }
    
    retval.y = dO;				//retval.x is the detected object type
    return retval;
}

// Function 114
vec2 rayMarch( in vec3 origin, in vec3 direction ) {
    vec2 total = vec2( .0 );
    for ( int i = 0 ; i < RAY_MARCH_STEPS ; i++ ) {
        vec3 point = origin + direction * total.x;
        vec2 current = sceneDistance( point, total.x );

#ifdef THANKS_SHANE_FOR_THE_RAY_SHORTENING_SUGGESTION
        if ( total.x > RAY_MARCH_TOO_FAR || current.x < RAY_MARCH_CLOSE ) {
            break;
        }
        // Note: Ray advancement occurs after checking for a surface hit.
        //
        // Ray shortening: Shorter for the first few iterations.
        total.x += i<32? current.x*.35 : current.x*.85; 
        total.y = current.y;
#else
        total.x += current.x;
        total.y = current.y;

        if ( total.x > RAY_MARCH_TOO_FAR || current.x < RAY_MARCH_CLOSE ) {
            break;
        }     
#endif

    }
    return total;
}

// Function 115
uint lcgStep(uint z, uint A, uint C) { return A * z + C; }

// Function 116
float raymarch(vec3 ro, vec3 rd) {
  float d = 0.;
  float t = 0.;
  for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
    d = map(ro + t * rd);
    if (d < EPSILON * t || t > MAX_DIST) break;
    t += 0.5 * d;
  }

  return d < EPSILON * t ? t : -1.;
}

// Function 117
RayHit March( vec3 origin,  vec3 direction)
{
  RayHit result;
  float maxDist = 70.0;
  float t = 0.0, glassDist = 10000.0, dist = 0.0;
  vec3 rayPos;

  for ( int i=0; i<200; i++ )
  {
    rayPos =origin+direction*t;
    dist = Map( rayPos);
    glassDist=min(glassDist, MapGlass( rayPos));

    if (abs(dist)<0.001 || t>maxDist )
    {             
      result.hit=!(t>maxDist);
      result.depth = t; 
      result.dist = dist;                              
      result.hitPos = origin+((direction*t));   
      result.steps = float(i);
      result.winDist = winDist;
      result.glassDist = glassDist;
      result.dekoDist = dekoDist;
      result.steelDist = steelDist;
      break;
    }
    t += dist;
  }    
   

  return result;
}

// Function 118
float march(vec3 rayOrigin, vec3 rayDirection){
	float minimumHit = 0.0;
    
    for(int i = 0; i < 100; i++){
    	vec3 hit = rayOrigin + rayDirection * minimumHit;
        float dist = intersect(hit);
        minimumHit += dist;
        if(dist > 100.0 || dist < 0.01) break;
    }
    return minimumHit;
}

// Function 119
float raymarchTerrain(Ray ray)
{
	float t = CAMERA_NEAR, h = 0.;
    for (int i = 0; i < 200; ++i)
    {
    	vec3 pos = ray.origin + ray.direction * t;
        h = pos.y - terrainFbm(pos.xz, MQ_OCTAVES, iChannel0);
        if (abs(h) < (t * .002) || t > CAMERA_FAR)
            break;
        t += h * .5;
    }
    return t;
}

// Function 120
bool raymarch_to_light(vec3 ray_start, vec3 ray_dir, float maxDist, float maxY, out float dist, out vec3 p, out int iterations, out float light_intensity
){dist = 0.0 + 10.1*hash1(gl_FragCoord.xy + time)
 ;float minStep = 0.01
 ;light_intensity = 1.
 ;float mapDist
 ;for (int i = 1; i <= MAX_RAYMARCH_ITER_SHADOWS; i++
 ){p = ray_start + ray_dir * dist
  ;mapDist = mapBlocks(p, ray_dir).y
  ;if (mapDist < MIN_RAYMARCH_DELTA
  ){iterations = i
   ;return true;}
  ;light_intensity = min(light_intensity, SOFT_SHADOWS_FACTOR * mapDist / dist)
  ;dist += max(mapDist, minStep)
  ;if(dist>=maxDist||p.y>maxY)break
 ;}
 ;return false;}

// Function 121
float smoothstep_unchecked( float x ) { return ( x * x ) * ( 3.0 - x * 2.0 ); }

// Function 122
float rayMarch(in float dmod, in vec3 ro, inout vec3 rd, float mint, float minstep, out int rep, out vec3 col, out float ref, out float trans, out vec3 absorb)
{
  float t = mint;
  for (int i = 0; i < MAX_RAY_MARCHES; i++)
  {
    float distance_ = distanceField(ro + rd*t, col, ref, trans, absorb);
    float distance = dmod*distance_;
    if (distance < TOLERANCE*t || t > MAX_RAY_LENGTH) break;
    t += max(distance, minstep);
    rep = i;
  }
  return t;
}

// Function 123
float rand_step(in float _x) {
    return rand(floor(_x));
}

// Function 124
vec3 marchRay(in vec3 ro, in vec3 rd) {
    const int MAX_ITERS = 100;
    float dist = EPSILON;
    float totalDist = 0.0;
    
    for (int i = 0; i < MAX_ITERS; i++) {
        if (abs(dist) < EPSILON || totalDist > MAX_RAY_DIST) {
            break;
        }
        
        dist = map(ro);
        ro += rd * dist;
        totalDist += dist;
    }
    
    if (abs(dist) < EPSILON) {
        return lighting(ro, rd);
    } else {
        return BG_COLOR;
    }
}

// Function 125
vec3 raymarch(vec2 uv) {
    vec3 col = vec3(0);
    
    vec3 s=vec3((curve(time, 0.7)-.5)*5.0,0,-150.0);
    vec3 t=vec3(0,0,0);
    
    s -= tunnel(s);
    t -= tunnel(t);
    
    vec3 cz=normalize(t-s);
    vec3 cx=normalize(cross(cz, vec3(sin(time*0.1)*0.1,1,0)));
    vec3 cy=normalize(cross(cz, cx));
    
    float fov = 0.3 + pulse(bpm*0.5,20.0)*0.2;
    vec3 r=normalize(uv.x*cx + uv.y*cy + fov*cz);
    
    float maxdist=300.0;
        
    vec3 p=s;
    float at=0.0;
    float dd=0.0;
    for(int i=0; i<200; ++i) {
        float d=map(p);
        if(d<0.001) {
            if(tra>0.5) {
                //vec3 n=getnorm(p);
                float didi = 1.0-length(p)/60.0;
                col += vec3(0.0002*float(i)*didi,0,0);
                d=0.2;
            } else {
                break;
            }
        }
        if(dd>maxdist) break;
        p+=r*d;
        dd+=d;
        at += (1.0-tra)*1.0/(1.0+d);
    }

    float fog = 1.0-clamp(dd/maxdist,0.0,1.0);
    
    vec3 n=getnorm(p);
    vec3 l=normalize(vec3(1,3,-2));
    vec3 h=normalize(l-r);
    float spec=max(0.0,dot(h,n));
    float fres=pow(1.0-abs(dot(r,n)), 3.0);
    
    vec3 col1 = vec3(0.7,0.8,0.6);
    vec3 col2 = vec3(0.8,0.8,0.5)*3.0;
    float iter = pow(abs(r.z), 7.0);
    vec3 atmocol = mix(col1, col2, iter);
    
    float ao=1.0;//getao(p,n,3.0) * getao(p,n,1.5) * 3.0;
    float sss=getsss(p,r,2.0) + getsss(p,r,10.0);
    
    float fade = fog * ao;
    col += (max(0.0,dot(n,l)) * .5+.5) * 0.7 * fade * atmocol * 0.2;
    col += max(0.0,dot(n,l)) * (0.3 + 0.6*pow(spec,4.0) + 0.9*pow(spec,30.0)) * fade * atmocol*0.7;
    col += pow(1.0-fog,5.0) * vec3(0.7,0.5,0.2);
    col += pow(oo*0.15,0.7)*vec3(0.5,0.7,0.3);
    
    col += pow(at*0.035,0.4) * atmocol;

    col += key(fract(length(p)*0.02)) * vec3(0.2,1.0,fract(trp.x*0.1)) * 10.0 * fog;
    col *= 1.8,

    col = tweakcolor(col);
    
    col *= 1.2-length(uv);
    
    return col;
}

// Function 126
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

// Function 127
vec3 marchCrystal(vec3 ori, vec3 dir, vec2 t_range, vec3 cell_origin, vec2 cell, out vec3 p) {
    float t = t_range.x;
    float d = 0.0;
    for(int i = 0; i < NUM_STEPS; i++) {
        p = ori + dir * t;
        d = crystalSDF(p - cell_origin, cell);
        if(abs(d) <= TRESHOLD || t >= t_range.y) break;
        t += d - TRESHOLD;
    } 
    return vec3(d, t, step(t,t_range.y-EPSILON));
}

// Function 128
float linearstep(float start, float end, float x)
{
    float range = end - start;
    return saturate((x - start) / range);
}

// Function 129
vec3 raymarchTerrain( const in vec3 ro, const in vec3 rd, const in vec3 bgc, const in float startdist, inout float dist ) {
	float t = startdist;

    // raymarch	
	vec4 sum = vec4( 0.0 );
	bool hit = false;
	vec3 col = bgc;
	
	for( int i=0; i<80; i++ ) {
		if( hit ) break;
		
		t += 8. + t/300.;
		vec3 pos = ro + t*rd;
		
		if( pos.y < terrainMap(pos) ) {
			hit = true;
		}		
	}
	if( hit ) {
		// binary search for hit		
		float dt = 4.+t/400.;
		t -= dt;
		
		vec3 pos = ro + t*rd;	
		t += (0.5 - step( pos.y , terrainMap(pos) )) * dt;		
		for( int j=0; j<2; j++ ) {
			pos = ro + t*rd;
			dt *= 0.5;
			t += (0.5 - step( pos.y , terrainMap(pos) )) * dt;
		}
		pos = ro + t*rd;
		
		vec3 dx = vec3( 100.*EPSILON, 0., 0. );
		vec3 dz = vec3( 0., 0., 100.*EPSILON );
		
		vec3 normal = vec3( 0., 0., 0. );
		normal.x = (terrainMap(pos + dx) - terrainMap(pos-dx) ) / (200. * EPSILON);
		normal.z = (terrainMap(pos + dz) - terrainMap(pos-dz) ) / (200. * EPSILON);
		normal.y = 1.;
		normal = normalize( normal );		

		col = vec3(0.2) + 0.7*texture( iChannel2, pos.xz * 0.01 ).xyz * 
				   vec3(1.,.9,0.6);
		
		float veg = 0.3*fbm(pos*0.2)+normal.y;
					
		if( veg > 0.75 ) {
			col = vec3( 0.45, 0.6, 0.3 )*(0.5+0.5*fbm(pos*0.5))*0.6;
		} else 
		if( veg > 0.66 ) {
			col = col*0.6+vec3( 0.4, 0.5, 0.3 )*(0.5+0.5*fbm(pos*0.25))*0.3;
		}
		col *= vec3(0.5, 0.52, 0.65)*vec3(1.,.9,0.8);
		
		vec3 brdf = col;
		
		float diff = clamp( dot( normal, -lig ), 0., 1.);
		
		col = brdf*diff*vec3(1.0,.6,0.1);
		col += brdf*clamp( dot( normal, lig ), 0., 1.)*vec3(0.8,.6,0.5)*0.8;
		col += brdf*clamp( dot( normal, vec3(0.,1.,0.) ), 0., 1.)*vec3(0.8,.8,1.)*0.2;
		
		dist = t;
		t -= pos.y*3.5;
		col = mix( col, bgc, 1.0-exp(-0.0000005*t*t) );
		
	}
	return col;
}

// Function 130
vec3 raymarchScattering(vec3 pos, 
                              vec3 rayDir, 
                              vec3 sunDir,
                              float tMax,
                              float numSteps) {
    float cosTheta = dot(rayDir, sunDir);
    
	float miePhaseValue = getMiePhase(cosTheta);
	float rayleighPhaseValue = getRayleighPhase(-cosTheta);
    
    vec3 lum = vec3(0.0);
    vec3 transmittance = vec3(1.0);
    float t = 0.0;
    for (float i = 0.0; i < numSteps; i += 1.0) {
        float newT = ((i + 0.3)/numSteps)*tMax;
        float dt = newT - t;
        t = newT;
        
        vec3 newPos = pos + t*rayDir;
        
        vec3 rayleighScattering, extinction;
        float mieScattering;
        getScatteringValues(newPos, rayleighScattering, mieScattering, extinction);
        
        vec3 sampleTransmittance = exp(-dt*extinction);

        vec3 sunTransmittance = getValFromTLUT(iChannel0, iChannelResolution[0].xy, newPos, sunDir);
        vec3 psiMS = getValFromMultiScattLUT(iChannel1, iChannelResolution[1].xy, newPos, sunDir);
        
        vec3 rayleighInScattering = rayleighScattering*(rayleighPhaseValue*sunTransmittance + psiMS);
        vec3 mieInScattering = mieScattering*(miePhaseValue*sunTransmittance + psiMS);
        vec3 inScattering = (rayleighInScattering + mieInScattering);

        // Integrated scattering within path segment.
        vec3 scatteringIntegral = (inScattering - inScattering * sampleTransmittance) / extinction;

        lum += scatteringIntegral*transmittance;
        
        transmittance *= sampleTransmittance;
    }
    return lum;
}

// Function 131
float march(vec3 ro, vec3 rd, float rmPrec, float maxd, float mapPrec)
{
    float s = rmPrec;
    float d = 0.;
    for(int i=0;i<100;i++)
    {      
        if (s<rmPrec||s>maxd) break;
        s = map(ro+rd*d).x*mapPrec;
        d += s;
    }
    return d;
}

// Function 132
float smoothstep_unchecked_6( float x ) { return x * x * x * x * x * ( 6.0 - 5.0 * x ); }

// Function 133
vec3 raymarch(vec3 eye, vec3 dir)
{
    vec3 depth = MIN_VEC;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++)
    {
        vec3 p = eye + dir * depth.z;
        
        vec3 dist = scene(p);
        if (dist.z < EPSILON * length(p)) {
			return vec3(dist.xy, depth.z);
        }
        depth.z += dist.z * (dist.z/(MAX_DIST) + 1.0);
        if (depth.z > MAX_DIST) {
            return MAX_VEC;
        }
    }
    return MAX_VEC;
}

// Function 134
vec3 Raymarch(Ray r, float startT)
{
    float t = startT;
    float d = 0.0;
    float iterations = 0.0;
    
	for(int j = 0; j < MAX_ITERATIONS; j++)
	{
		d = sdf(r.origin + r.direction * t, false);

		if(d < EPSILON)
            break;
        
		t += d * .4;
        
        if(t > MAX_DISTANCE)
            break;
        
        iterations += 1.0;
	}
    
    t = min(t, MAX_DISTANCE);
    
    return vec3(t, iterations / float(MAX_ITERATIONS), d);
}

// Function 135
dfObject march(in vec3 rayStart, in vec3 rayDir) {
    
    // Near Clipping
    float t = 0.01;

    // Far Clipping
    float tmax= 50.0;
    
    Material obj = Material(vec3(0.), vec3(0.), MAT_NOTHING);
    
    for (int i = 0; i < 50; i++) {
    	
        float prec = EPSILON*t;
        
        // map on Distance Field
        vec3 p = rayStart + t*rayDir;
        dfObject res = map(p);
        

        // If we hit something, or out of far clipping, exit
        if ((res.d < EPSILON) || (t > tmax))
        	break;
       	
        t += res.d;
        obj = res.mat;
        
        
    }
    if (t > tmax) obj.type = -1;
    return dfObject(t, obj);
    
}

// Function 136
t_object march(vec3 ro, vec3 rd) {
	float d = 1.;
    float curd = 0.;
    int id;
    
    t_object obj;
    for(int i=0; i < march_steps_; i++) {
        if(d < epsilon_ || curd > march_range_) break;
        obj = eval_scene(ro + curd*rd);
        d = obj.d*dist_perc_;
        curd += d;
    }
    
    obj.d = curd;
    return obj;
}

// Function 137
float colGradStep(vec3 pt, vec3 normal)
{
    float dotted = dot(pt, normal) / 2.0 + 1.0;
    dotted = pow(dotted, colGradPow);

    return 1.0 - dotted;
}

// Function 138
vec2 rayMarching(in vec3 camPos, in vec3 rayDirection)
{
    float dMin = 1.;
    float dMax = 50.;
    float precis = 0.002;
    float traveledDistance = dMin;
    float color = -1.;
    
    for( int i = 0 ; i < 50 ; i++ )
    {
        vec2 res = scene( camPos + (rayDirection * traveledDistance) );
        
        if( res.x<precis || traveledDistance>dMax )
        {
            break;
        }
        
        traveledDistance += res.x;
        color = res.y;
    }
    
    if( traveledDistance > dMax )
    {
        color = -1.0;
    }
    return vec2( traveledDistance, color );
}

// Function 139
vec3 Raymarch(Ray r, float startT)
{
    float t = startT;
    float d = 0.0;
    float iterations = 0.0;
    
	for(int j = 0; j < MAX_ITERATIONS; j++)
	{
		d = sdf(r.origin + r.direction * t, false);

		if(d < EPSILON)
            break;
        
		t += d;
        
        if(t > MAX_DISTANCE)
            break;
        
        iterations += 1.0;
	}
    
    t = min(t, MAX_DISTANCE);
    
    return vec3(t, iterations / float(MAX_ITERATIONS), d);
}

// Function 140
vec4 rayMarch(in vec3 from, in vec3 dir, in vec2 fragCoord) {
	// Add some noise to prevent banding
	float totalDistance = Jitter*rand(fragCoord.xy+vec2(iTime));
	vec3 dir2 = dir;
	float distance;
	int steps = 0;
	vec3 pos;
	for (int i=0; i <= MaxSteps; i++) {
		pos = from + totalDistance * dir;
		distance = DE(pos)*FudgeFactor;
		
		totalDistance += distance;
		if (distance < MinimumDistance) break;
		steps = i;
	}
	
	// 'AO' is based on number of steps.
	float ao = 1.0-float(steps)/float(MaxSteps);
	
	// Since our distance field is not signed,
	// backstep when calc'ing normal
	vec3 normal = getNormal(pos-dir*normalDistance*3.0);
	vec3 b = bg(dir);
	b = mix(b, vec3(0.0,0.0,0.0), 0.2);
	if (steps == MaxSteps) {
		return vec4(b,1.0);
	}
	vec3 color = getColor(normal, pos);
	vec3 light = getLight(color, normal, dir);
	color =(color*Ambient+light)*(ao);
	return vec4(color,1.0);
}

// Function 141
vec3 RayMarch(vec3 origin,vec3 direction)
{
    float hitDist = 0.0;
    for(int i = 0;i < MAX_STEPS;i++)
    {
        float sceneDist = Scene(origin + direction * hitDist);
        
        hitDist += sceneDist;
        
        if(sceneDist < MIN_DIST)
        {
            break;
        }
    }
    
    return origin + direction * hitDist;
}

// Function 142
float usmoothstep( in float x )
{
    x = clamp(x,0.0,1.0);
    return x*x*(3.0-2.0*x);
}

// Function 143
vec3 raymarch(Ray inputray)
{
    const float exposure = 1e-2;
    const float gamma = 2.2;
    const float intensity = 100.0;
    vec3 ambient = vec3(0.2, 0.3, 0.6) *6.0* intensity / gamma;

    vec3 prevcolour = vec3(0.0, 0.0, 0.0);
    vec3 colour = vec3(0.0, 0.0, 0.0);
    vec3 mask = vec3(1.0, 1.0, 1.0);
    vec3 fresnel = vec3(1.0, 1.0, 1.0);
    
    Ray ray=inputray;
        
#ifdef LIGHTBULB    
    vec3 lightpos = g_light.pos;
#else
    vec3 lightpos = -g_light.pos*200000000000.0;	// 'directional'
#endif
    
    for (int i=0; i<REFLECT_ITERATIONS; i++)
    {
        Result result = raymarch_query(ray, 10.0);

        vec3 tolight = lightpos - result.pos;
        tolight = normalize(tolight);
                
        if (result.t > NOT_CLOSE)
        {
#ifdef LIGHTBULB            
            vec3 spotlight = drawlights(ray)*600.0;
#else            
            vec3 spotlight = vec3(1e4) * pow(clamp(dot(ray.dir, tolight),0.0,1.0), 75.0);
#endif //LIGHTBULB           
            
//          ambient = texture(iChannel1, ray.dir).xyz*100.0;
            ambient = mix(vec3(1.0, 1.0, 1.0), vec3(0.2, 0.2, 0.5), pow(abs(ray.dir.y), 0.5))*300.0;
                       
            colour += mask * (ambient + spotlight);                             
            break;
        }
        else
        {   
            //result.mat.colour.rgb *= noise(ray.pos);
			prevcolour = result.mat.colour.rgb;
            
            vec3 r0 = result.mat.colour.rgb * result.mat.specular;
            float hv = clamp(dot(result.normal, -ray.dir), 0.0, 1.0);
            fresnel = r0 + (1.0 - r0) * pow(1.0 - hv, 5.0);
            mask *= fresnel;            
            
            vec3 possiblelighting = clamp(dot(result.normal, tolight), 0.0, 1.0) * g_light.colour
                    * result.mat.colour.rgb * result.mat.diffuse
                    * (1.0 - fresnel) * mask / fresnel;
            
            if (length(possiblelighting) > 0.01f)
            {
                Ray shadowray = Ray(result.pos+result.normal*0.01, tolight);
                Result shadowresult = raymarch_query(shadowray, length(lightpos - result.pos)*0.9);
#ifdef SOFTSHADOWS                
                colour += possiblelighting*clamp(shadowresult.mint*4.0, 0.0, 1.0);
#else
                if (shadowresult.travelled >= length(lightpos - result.pos)*0.9)
                	colour += possiblelighting;
#endif
            }
            
            Ray reflectray;
            reflectray.pos = result.pos + result.normal*0.02f;
            reflectray.dir = reflect(ray.dir, result.normal);
            ray = reflectray;
        }
    }
        
    colour.xyz = vec3(pow(colour * exposure, vec3(1.0 / gamma)));    
    return colour;    
}

// Function 144
vec3 march(vec3 from,vec3 dir) {
	vec3 p, col=vec3(0.);
    float td=.5, k=0.;
    vec2 h;
    for (int i=0;i<600;i++) {
    	p=from+dir*td;
        h=hit(p);
        if (h.x>.5||td>maxdist) break;
        td+=st;
    }
    if (h.x>.5) {
        p=bsearch(from,dir,td);
    	col=shade(p,dir,h.y,td);
    } else {
    }
	col=mix(col,2.*vec3(mod(gl_FragCoord.y,4.)*.1),pow(td/maxdist,3.));
    return col*vec3(.9,.8,1.);
}

// Function 145
bool hitDrostePicture( vec2 uv ) {
	vec3 ro, rd;
	getRoAndRd( uv, ro, rd );	
	
	vec2 res = fastCastRay(ro,rd,200.0);
	return (res.y == 2. );
}

// Function 146
vec4 raymarch(vec3 org, vec3 dir)
{
	float d = 0.0, glow = 0.0, eps = 0.02;
	vec3  p = org;
	bool glowed = false;
	
	for(int i=0; i<64; i++)
	{
		d = scene(p) + eps;
		p += d * dir;
		if( d>eps )
		{
			if(flame(p) < .0)
				glowed=true;
			if(glowed)
       			glow = float(i)/64.;
		}
	}
	return vec4(p,glow);
}

// Function 147
vec4 vmarch(in vec3 ro, in vec3 rd)
{
	vec4 rz = vec4(0);
	float t = 2.5;
    t += 0.03*hash21(gl_FragCoord.xy);
	for(int i=0; i<STEPS; i++)
	{
		if(rz.a > 0.99 || t > 6.)break;
		vec3 pos = ro + t*rd;
        vec4 col = map(pos);
        float den = col.a;
        col.a *= ALPHA_WEIGHT;
		col.rgb *= col.a*1.7;
		rz += col*(1. - rz.a);
        t += BASE_STEP - den*(BASE_STEP-BASE_STEP*0.015);
	}
    return rz;
}

// Function 148
vec4 raymarch (vec3 ro, vec3 rd)
{
	for (int i=0; i<256; i++)
	{
		float t = map(ro);
		if ( t<0.001 ) return color(ro);
		ro+=t*rd;
	}
	return vec4(0.0);
}

// Function 149
vec3 RayMarchTerrial(vec3 ro,vec3 rd,float rz){
    vec3 col = vec3(1.,1.,1.);
    vec3 pos = ro + rz * rd;
    vec3 nor = CalcTerrianNormal(pos,rz);

    vec3 ref = reflect( rd, nor );
    float fre = clamp( 2.0+dot(rd,nor), 0.1, 2.0 );
    vec3 hal = normalize(lightDir-rd);
	col = vec3(0.09,0.06,0.04);
    // lighting     
    float amb = clamp(0.6+0.6*nor.y,0.1,2.0);
    float dif = clamp( dot( lightDir, nor ), 0.1, 2.0 );
    float bac = clamp( 0.3 + 0.9*dot( normalize( vec3(-lightDir.x, 0.1, lightDir.z ) ), nor ), 0.1, 2.0 );

    //shadow
    float sh = 2.0; 
  
    vec3 lin  = vec3(0.1,0.1,0.1);
    lin += dif*vec3(8.00,6.00,4.00)*2.3;
    lin += amb*vec3(0.50,0.70,2.00)*2.2;
    lin += bac*vec3(0.50,0.60,0.70);
    col *= lin;
  
    // fog
    float fo = 1.2-exp(-pow(0.002*rz/SC,2.5));
    vec3 fco = 0.75*vec3(0.5,0.75,2.0);// + 0.1*vec3(1.0,0.8,0.5)*pow( sundot, 4.0 );
    col = mix( col, fco, fo );
  return col;
}

// Function 150
float march(vec3 ro, vec3 rd){
    
    vec3 p = ro;
    float t = 0.0;
    float n = 0.0;
    
    for(float i = 0.0; i < STEPS; ++i){
        ++n;
        float d = dist(p);
        t += d;
        p += d*rd;
        
        if(d < EPSILON || t > FAR){
            break;
        }
    }
    return 1.0-n/STEPS;
}

// Function 151
vec4	march(vec3 pos, vec3 dir)
{
    vec2	dist = vec2(0.0, 0.0);
    vec3	p = vec3(0.0, 0.0, 0.0);
    vec4	s = vec4(0.0, 0.0, 0.0, 0.0);

    for (int i = -1; i < I_MAX; ++i)
    {
    	p = pos + dir * dist.y;
        dist.x = scene(p);
        dist.y += dist.x;
        if (dist.x < E || dist.y > 30.)
        {
            s.y = 1.;
            break;
        }
        s.x++;
    }
    s.w = dist.y;
    return (s);
}

// Function 152
vec4 raymarch(vec3 eye, vec3 dir)
{
    vec4 depth = vec4(0);
    for (int i = 0; i < 128; i++)
    {
        vec3 p = eye + dir * depth.w;
        
        vec4 dist = scene(p);
        if (dist.w < EPSILON * length(p)) {
			return vec4(saturate(dist.xyz), depth.w);
        }
        depth.w += dist.w;
        if (depth.w > MAX_DIST) {
            return MAX_VEC;
        }
    }
    return MAX_VEC;
}

// Function 153
vec2 smootherstep(vec2 t) {
	return t* t* t* (t* (t* 6.0- 15.0)+ 10.0);
}

// Function 154
float raymarch()
{
  float d = 0.0, t = 0.0;
  for(int i = 0; i < 32; ++i)
  {
    d = sphere(camera.origin + t * camera.direction);
    if(abs(d) < 0.01) return t;
    if(t > 100.0) return -1.0;
    t += d;
  }
  return -1.0;
}

// Function 155
Hit raymarch(CastRay castRay){

    float currentDist = INTERSECTION_PRECISION * 2.0;
    Model model;

    Ray ray = Ray(castRay.origin, castRay.direction, 0.);

    for( int i=0; i< NUM_OF_TRACE_STEPS ; i++ ){
        if (currentDist < INTERSECTION_PRECISION || ray.len > MAX_TRACE_DISTANCE) {
            break;
        }
        model = map(ray.origin + ray.direction * ray.len);
        currentDist = model.dist;
        ray.len += currentDist * FUDGE_FACTOR;
    }

    bool isBackground = false;
    vec3 pos = vec3(0);
    vec3 normal = vec3(0);
    vec3 color = vec3(0);

    if (ray.len > MAX_TRACE_DISTANCE) {
        isBackground = true;
    } else {
        pos = ray.origin + ray.direction * ray.len;
        normal = calcNormal(pos);
    }

    return Hit(ray, model, pos, isBackground, normal, color);
}

// Function 156
vec3 raymarch( vec3 ro, vec3 rd, const vec2 nf, const float eps ) {
    glowAcc = vec2(999.);
    vec3 p = ro + rd * nf.x;
    float l = 0.;
    for(int i=0; i<128; i++) {
		float d = worldSafe(p);
        l += d;
        p += rd * d;
        
        if(d < eps || l > nf.y)
            break;
    }
    
    return p;
}

// Function 157
vec4 raymarch(vec3 from, vec3 increment)
{
	const float maxDist = 200.0;
	const float minDist = 0.1;
	const int maxIter = RAYMARCH_ITERATIONS;
	
	float dist = 0.0;
	
	float material = 0.0;
	
	for(int i = 0; i < maxIter; i++) {
		vec3 pos = (from + increment * dist);
		float distEval = distf(pos, material);
		
		if (distEval < minDist) {
			break;
		}
		
		dist += distEval * RAYMARCH_DOWNSTEP;
	}
	
	
	if (dist >= maxDist) {
		material = 0.0;
	}
	
	return vec4(dist, material, 0.0, 0.0);
}

// Function 158
float raymarch (in vec3 ro, in vec3 rd)
{
    float t = .0;
    float d = .0;
    for (int i = 0; i < 64; ++i) {
        vec3 p = ro + d * rd;
        t = scene (p);
        if (abs(t) < EPSILON*(1. + .125*t)) break;
        d += t * .75;
    }

    return d;
}

// Function 159
float overshootstep1( float x, vec3 args )
{
	float df0 = 6.0;
	float s = 0.5;
	if ( x > 0.0 ) return 1.0 - ( 1.0 - overshoot( x, args, df0 ) ) * s;
	return 1.0 - ( 1.0 + ( 1.0 - cubicstep( max( x, -1.0 ) + 1.0, 0.0, df0 ) ) ) * s;
}

// Function 160
float lightMarch(vec3 ro, vec3 lightPos)
{
    vec3 rd = lightPos-ro;
    float d = length (rd);
    rd = rd/d;
    float t = 0.;
    float stepLength = d/ float(nStepLight);
    float densitySum = 0.;
    float sampleNoise;
    int i = 0;
    for (; i < nStepLight; i++)
    {
    	sampleNoise = map ( ro + t * rd);
       
        densitySum += sampleNoise;
        
        t += stepLength;
    }
    
    return exp(- d * (densitySum / float(i)));
}

// Function 161
MarchRes marchRay(vec3 pos, vec3 dir, float speed)
{
 	MarchRes res;
    res.minDist = 100000.0;
    res.glowAmt = 1.0;
    Object o;
    
    res.totalDist = 0.001;

    for(int x=0; x<200; x++)
    {
 		res.curRay = pos + (dir*res.totalDist);
        
        o = map(res.curRay);
        res.glowAmt = min(res.glowAmt, 0.5*o.dist/(0.02+abs(0.02*sin(iTime))*res.totalDist));
        if(abs(o.dist) < 0.00001)
        {
            res.minDist = o.dist;
            res.obj = o;
            break;
        }
        else if(res.totalDist >= VIEW_DIST) break;
            
        if(o.dist < res.minDist)
            res.minDist = o.dist;
        res.totalDist += o.dist*speed; // repalce 0.8 w/ this for trippy mode ;p => (0.3+0.2*(sin(iTime))); //couldn't handle the hair :' (
    }
    
    if(res.totalDist < VIEW_DIST)
    {
        o.normal = calcNormal(res.curRay, o.normEps);
        res.obj = o;
    }
    	
    res.glowAmt = max(res.glowAmt, 0.0);
    res.glowAmt = smoothstep(0.0, 1.0, res.glowAmt);// res.glowAmt*res.glowAmt*(3.0-2.0*res.glowAmt);
    return res;
}

// Function 162
vec2 marcher(vec3 ro, vec3 rd, float sg,  int maxstep){
	float d =  .0,
     	  m =  -1.;
    	int i = 0;
        for(i=0;i<maxstep;i++){
        	vec3 p = ro + rd * d;
            vec2 t = map(p, sg);
            if(abs(t.x)<d*MINDIST||d>MAXDIST)break;
            d += t.x*.95;
            m  = t.y;
        }
    return vec2(d,m);
}

// Function 163
float linstep(in float mn, in float mx, in float x)
{
	return clamp((x - mn)/(mx - mn), 0., 1.);
}

// Function 164
RayHit RaymarchScene(in Ray ray)
{
    RayHit hit;
    
    hit.hit      = false;
    hit.material = 0.0;
    
    float sdf   = FarClip;
    int   steps = 0;
    
    for(float depth = NearClip; (depth < FarClip) && (steps < MarchSteps); ++steps)
    {
    	vec3 pos = ray.origin + (ray.direction * depth);
        
        sdf = Scene_SDF(pos, hit);
        
        if(sdf < Epsilon)
        {
            hit.hit      = true;
            hit.surfPos  = pos;
            hit.surfNorm = Scene_Normal(pos);
            
            return hit;
        }
        
        depth += sdf;
    }
    
    return hit;
}

// Function 165
MarchResult march(vec3 p0, vec3 ray, bool withWater) {
    float type = SKY;
    float d = 0.0;
    int stp = 0;
    vec3 p = p0;
    while (type==SKY && d<(withWater?maxDist:maxDist*0.125) && (stp++<(withWater?maxStep:maxStep/3))) {
        p = p0 + d*ray;
        float waterLevel = withWater ? /*getWaterLevel(p.xz, d)*/ meanWaterLevel : -9999.9;
        float stpSize = estDistToTrn(p,d) * (withWater?1.0:2.0);
        // TODO fix this mess
        if (p.y<=waterLevel) {
            type = WATER;
            d = (waterLevel-p0.y)/ray.y;
            p = p0+d*ray;
        }
        else if (stpSize<d*0.001) type = LAND;
        else d+= stpSize;
    }
    d = min(d, maxDist);
    return MarchResult(d, p, type);
}

// Function 166
float MarchThruSolidVoxels(vec3 ro, vec3 rd, float dlimit)
{
    vec2 mr = March(ro, rd, 1.04, 128 + IZERO);
    //vec3 hp = ro + rd * t;
    return mr.x;
}

// Function 167
float rayMarch(in vec3 ro, in vec3 rd) {
  float mn=0., mx=300.;
  float thr = 1e-6;

  float d=mn;
  for(int i=0;i<100;i++) {
    vec3 pos = ro + rd*d;
    float tmp = map(pos);
    if(tmp<thr || mx<tmp) break;
    d += tmp * 0.3;
  }
  return d;
}

// Function 168
float linearstep(float mn, float mx, float v)
{
    return clamp((v - mn) / (mx - mn), 0.0, 1.0);
}

// Function 169
vec4 raymarch( in vec3 rayo, in vec3 rayd, in float expInter, in vec2 fragCoord )
{
    vec4 sum = vec4( 0.0 );
     
    float step = 0.075;
     
    // dither start pos to break up aliasing
	vec3 pos = rayo + rayd * (expInter + step*texture( iChannel0, fragCoord.xy/iChannelResolution[0].x ).x);
	
    for( int i=0; i<25; i++ )
    {
        if( sum.a > 0.99 ) continue;
		
		float radiusFromExpCenter = length(pos - expCenter);
		
		if( radiusFromExpCenter > expRadius+0.01 ) continue;
		
		float dens, rawDens;
		
        dens = densityFn( pos, radiusFromExpCenter, rawDens, sum.a );
		
		vec4 col = vec4( computeColour(dens,radiusFromExpCenter), dens );
		
		// uniform scale density
		col.a *= 0.6;
		
		// colour by alpha
		col.rgb *= col.a;
		
		// alpha blend in contribution
		sum = sum + col*(1.0 - sum.a);  
		
		// take larger steps through negative densities.
		// something like using the density function as a SDF.
		float stepMult = 1. + 2.5*(1.-clamp(rawDens+1.,0.,1.));
		
		// step along ray
		pos += rayd * step * stepMult;
    }
	
    return clamp( sum, 0.0, 1.0 );
}

// Function 170
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

// Function 171
vec2 dist_march( vec3 ro, vec3 rd, float maxd )
{
    
    float epsilon = 0.001;
    float dist = 10. * epsilon;
    float t = 0.;
    for (int i=0; i < DISTMARCH_STEPS; i++) 
    {
        if ( abs(dist) < epsilon || t > maxd ) break;
        t += dist;
        dist = scenedf( ro + t * rd );
    }


    float objhit = 0.;
    if( t < maxd ) 
    {
        objhit = 1.;
    }

    return vec2(objhit, t);
}

// Function 172
void TestBoxMarch(in vec3 rayPos, inout SRayHitInfo info, in vec3 boxPos, in vec3 boxRadius, in float width, in SMaterial material)
{
    float dist = BoxDistance(boxPos, boxRadius, width, rayPos);
    if (dist < info.dist)
    {
        info.objectPass = OBJECTPASS_RAYMARCH;
        info.dist = dist;
        
        vec3 relPos = max(abs(rayPos - boxPos) - boxRadius, 0.0f);
        int maxIndex = MaxCompIndex(relPos);
        if (maxIndex == 0)
        {
            info.normal = (rayPos.x < boxPos.x) ? vec3(-1.0f, 0.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
        }
        else if(maxIndex == 1)
        {
            info.normal = (rayPos.y < boxPos.y) ? vec3(0.0f, -1.0f, 0.0f) : vec3(0.0f, 1.0f, 0.0f);
        }
        else
        {
            info.normal = (rayPos.z < boxPos.z) ? vec3(0.0f, 0.0f, -1.0f) : vec3(0.0f, 0.0f, 1.0f);
        }
        
        info.material = material;
    }    
}

// Function 173
vec2 raymarchScene(in vec3 ro, in vec3 rd, in float tmin, in float tmax, bool refrSph) {
    vec3 res = vec3(ID_NONE);
    float t = tmin;
    for (int i = 0; i < 250; i++) {
        vec3 p = ro + rd * t;
        res = vec3(intersect(p, refrSph), t);
        float d = res.y;
        if (d < (0.001 * t) || t > tmax)
            break;
        t += 0.5 * d;
    }
    return res.xz;
}

// Function 174
float Raymarch(Camera camera, vec2 uv)
{    
    float totalDistance = 0.0;
    
    int bounceFrame = BounceFrame(iFrame);
    
    float maxDistance = MAX_DISTANCE;
    
    if(bounceFrame > 0)
		maxDistance = 15.0;
    
	for(int j = 0; j < MAX_STEPS; ++j)
	{
        vec3 p = camera.origin + camera.direction * totalDistance;
		float d = max(0.0, sdf(p));

		totalDistance += d;
                
		if(d < EPSILON || totalDistance > maxDistance)
            break;
	}
    
    return totalDistance;
}

// Function 175
vec3 ray_march(in vec3 ray_origin, in vec3 ray_direction)
{
    float total_distance_traveled = 0.0;
    const int NUMBER_OF_STEPS = 64;
    const float MINIMUM_HIT_DISTANCE = 0.001;
    const float MAXIMUM_TRACE_DISTANCE = 1000.0;

    for (int i = 0; i < NUMBER_OF_STEPS; ++i)
    {
        vec3 current_position = ray_origin + total_distance_traveled * ray_direction;

		float distance_to_closest = map_the_world(current_position);

        if (distance_to_closest < MINIMUM_HIT_DISTANCE) 
        {
            vec3 normal = calculate_normal(current_position);
            
            vec3 light_positions[3];
            light_positions[0] = vec3(1.0+sin(iTime)*5.0, -3.0+3.0*cos(iTime/3.0), 4.0 + 1.0 *sin(iTime/5.0));
            light_positions[1] = vec3(1.0-sin(iTime/2.0)*2.0, -1.0-cos(iTime/2.0), 7.0 + 1.0 -sin(iTime/4.0));
            light_positions[2] = vec3(2.0-sin(iTime/2.0)*2.0, -5.0-sin(iTime/4.0), 2.0 + 1.0 -sin(iTime/1.0));
            float light_intensities[3];
            light_intensities[0] = 0.8;
            light_intensities[1] = 0.4;
            light_intensities[2] = 0.7;
            vec3 direction_to_view = normalize(current_position - ray_origin);
            
            vec3 col = vec3(0.0);
            
            for (int j = 0; j < 3; j++)
            {
                vec3 direction_to_light = normalize(current_position - light_positions[j]);
                vec3 light_reflection_unit_vector =
                	 reflect(direction_to_light ,normal);                

                float diffuse_intensity = 0.6*pow(max(0.0, dot(normal, direction_to_light)),5.0);            
                float ambient_intensity = 0.2;            
                float specular_intensity = 
                    1.15* pow(clamp(dot(direction_to_view, light_reflection_unit_vector), 0.0,1.0), 50.0);
                float backlight_specular_intensity =             
                    0.2* pow(clamp(dot(direction_to_light, light_reflection_unit_vector),0.0,1.0), 3.0); 
                float fresnel_base = 1.0 + dot(direction_to_view, normal);
                float fresnel_intensity = 0.10*pow(fresnel_base, 0.3);
                float fresnel_shadowing = pow(fresnel_base, 5.0);            
                float fresnel_supershadowing = pow(fresnel_base, 50.0);
                float attenuation =  pow(total_distance_traveled,2.0)/180.0;

                
            	vec3 colFromLight = vec3(0.0);
                colFromLight += vec3(0.89, 0.0, 0.0) * diffuse_intensity;
                colFromLight += vec3(0.3, 0.1, 0.1) * ambient_intensity;
                colFromLight += vec3(1.0) * specular_intensity;            
                colFromLight += vec3(1.0,0.5,0.5) * backlight_specular_intensity;            
                colFromLight += vec3(1.0, 0.1, 0.2) * fresnel_intensity;
                colFromLight -= vec3(0.0, 1.0, 1.0) * fresnel_shadowing ;
                colFromLight -= vec3(0.0, 1.0, 1.0) * fresnel_supershadowing * col * col;
                colFromLight += vec3(.3, 0.1, 0.1) - attenuation ; 
               	colFromLight /= 1.2;
                colFromLight *= light_intensities[j];
                col += colFromLight;
            }
            return col;
        }


        if (total_distance_traveled > MAXIMUM_TRACE_DISTANCE)
        {
            break;
        }
        total_distance_traveled += distance_to_closest;
    }
    return vec3(0.0);
}

// Function 176
RayHit MarchReflection( vec3 origin,  vec3 direction)
{
  RayHit result;
  float maxDist = 90.0;
  float t = 0.0, dist = 0.0;
  vec3 rayPos;
 
  for ( int i=0; i<32; i++ )
  {
    rayPos =origin+direction*t;
    dist = Map( rayPos);
 

    if (abs(dist)<0.05 || t>maxDist )
    {             
      result.hit=!(t>maxDist);
      result.depth = t; 
      result.dist = dist;                              
      result.hitPos = origin+((direction*t));   
      result.steelDist = steelDist;
      result.platformDist = platformDist;
      result.terrainDist =terrainDist;
      result.waterDist =waterDist;
      break;
    }
    t += dist;
  }

  return result;
}

// Function 177
vec3 rayMarchReflection(vec3 rayDir, vec3 cameraOrigin)
{
    const int maxItter = 30;
	const float maxDist = 20.0;
    
    float totalDist = 0.0;
	vec3 pos = cameraOrigin;
	vec4 dist = vec4(epsilon);

    for(int i = 0; i < maxItter; i++)
	{
		dist = distfunc(pos);
		totalDist += dist.x;
		pos += dist.x * rayDir;
        
        if(dist.x < epsilon || totalDist > maxDist)
		{
			break;
		}
	}
    
    return vec3(dist.x, totalDist, dist.y);
}

// Function 178
float shared_smoothstep(float _3522, float _3523, float _3524)
{
    float _3529 = shared_saturate((_3524 - _3522) / (_3523 - _3522));
    return (_3529 * _3529) * (3.0 - (2.0 * _3529));
}

// Function 179
void StepB (int mId, out vec3 r, out vec3 v)
{
  vec3 f;
  float fDamp, dt, biMass;
  fDamp = 0.5;
  biMass = 0.02;
  r = GetRB (mId);
  v = GetVB (mId);
  f = biMass * (BBForce (mId, r, v) + WallForceB (r) + BShForceB (r) -
     2. * fDamp * v) - gravVec;
  dt = 0.02;
  v += dt * f;
  r += dt * v;
}

// Function 180
Hit raymarch(vec3 eye, vec3 ray){
	float dsum = 0.0;
	for(int i=0; i<iterations; i++){
		vec3 p = eye + dsum*ray;
		float dmin = dist(p);
		if(dmin < threshold){
			return Hit(p, grad(p), dsum);
		} else {
			dsum += max(min_step, dmin*step_fraction);
		}
	}
	
	vec3 p = eye + dsum*ray;
	return Hit(p, vec3(0), dsum);
}

// Function 181
void ScatteringStep(float R, float V, float L, float VoL, float Di, vec3 OptLenV, float UseOzoneAbsorption, out vec3 RayS, out vec3 MieS)
{
	float Ri = Sqrt((Di + 2.0 * V * R) * Di + R * R);
    float Vi = (R * V + Di) / Ri;
    float Li = (R * L + Di * L) / Ri;
    
    if (Li > AtmHorizonCos(Ri))
    {
        // Opitcal length from a current point to the atmoshere bound 
        // in view direction and in direction to the light source
        vec3 OptLenVi = OpticalLength(Ri, Vi, ATM_NUM_TRANSMITTANCE_STEPS);
        vec3 OptLenLi = OpticalLength(Ri, Li, ATM_NUM_TRANSMITTANCE_STEPS);
        
        // Compute total optical length of the path and compute transmittance from it
        vec3 Ti = Transmittance(max(OptLenV - OptLenVi, 0.0) + OptLenLi, UseOzoneAbsorption);
        
        float Hi = Ri - R;
        
        // Multiply by corresponding particle density
        RayS = Ti * exp(-Hi * kAtmRayHeightScale);
        MieS = Ti * exp(-Hi * kAtmMieHeightScale);
    }
}

// Function 182
float march(vec3 ro,vec3 rd,float t){
    float r;
    for(int i=0;i<100;i++){
        vec3 p=ro+rd*r;
        float dS=map(p,t);
        r+=dS;
        if(r>100.||abs(dS)<.001)break;
    }
    
    return r;
}

// Function 183
float SmoothStep5 (float xLo, float xHi, float x)
{
  x = clamp ((x - xLo) / (xHi - xLo), 0., 1.);
  return x * x * x * (x * (6. * x - 15.) + 10.);
}

// Function 184
float sstep(float a, float b) {
    return smoothstep(a - .005, a + .005, b);
}

// Function 185
float march(vec3 p, vec3 nv) {
    float lightAmount = 1.0;

    vec2 tRange;
    float didHitBox;
    boxClip(BOX_MIN, BOX_MAX, p, nv, tRange, didHitBox);
    tRange.s = max(0.0, tRange.s);

    if (didHitBox < 0.5) {
        return 0.0;
    }

    float t = tRange.s + min(tRange.t-tRange.s, RAY_STEP_L)*hash13(100.0*p);
    int i=0;
    for (; i<150; i++) { // Theoretical max steps: (BOX_MAX-BOX_MIN)*sqrt(3)/RAY_STEP_L
        if (t > tRange.t || lightAmount < 1.0-QUIT_ALPHA_L) { break; }
        
        vec3 rayPos = p + t*nv;
        vec3 lmn = lmnFromWorldPos(rayPos);

        float density = getPage1(lmn).s;
        float calpha = clamp(density * MAX_ALPHA_PER_UNIT_DIST * RAY_STEP_L, 0.0, 1.0);

        lightAmount *= 1.0 - calpha;

        t += RAY_STEP_L;
    }

    return lightAmount;
}

// Function 186
RayResult raymarch(in vec3 ro, in vec3 rd)
{
    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; ++i)
    {
        vec3 p = ro + rd * t;
        SceneDist scene = map(p);
        if (scene.dist < MIN_DIST)
        {
            return RayResult(t, p, normal(p), scene.material);
        }
        t += scene.dist;
        if (t > MAX_DIST)
        {
            break;
        }
    }
    return RayResult(MAX_DIST, vec3(0.0), vec3(0.0), kMatNone);
}

// Function 187
vec3 RayMarchGlass(vec2 uv, vec3 ro, vec3 dir) {
    float traveled = 0.0;
    float everythingTraveled = 0.0;
    
    vec2 distAndMaterial = vec2(0);
    vec2 dnmEverythingElse = vec2(0);
    
    vec3 impact = vec3(1.0);
    bool goneTooFar = false;
    bool everythingsGoneTooFar = false;
    bool hitGlass = false;
    bool hitOtherThings = false;
    
    for (int i=ZERO; i < 100; ++i){
        dnmEverythingElse = sceneWithMaterials(ro + dir * everythingTraveled);
        everythingTraveled += dnmEverythingElse.x;
        
#ifndef WITHOUTKODOS
        distAndMaterial = sceneWithGlassMaterials(ro + dir * traveled);
        traveled += distAndMaterial.x;
#endif
        
        if (dnmEverythingElse.x < .01) {
            hitOtherThings = true;
        }
        
#ifndef WITHOUTKODOS
        if (distAndMaterial.x < .01) {
            hitGlass = true;
        }
        
        
        if (distAndMaterial.x > MAXDISTANCE) {
            goneTooFar = true;
        }
#endif
        
        if (dnmEverythingElse.x > MAXDISTANCE) {
            everythingsGoneTooFar = true;
        }
        
#ifndef WITHOUTKODOS
        if (hitGlass && (hitOtherThings || everythingsGoneTooFar)) {
            break;
        }
        
        if (hitOtherThings && (goneTooFar || hitGlass)) {
            break;
        }
        
        if (goneTooFar && everythingsGoneTooFar) {
            break;
        }
#else
        if (hitOtherThings) {
            break;
        }
        
        if (everythingsGoneTooFar) {
            break;
        }
#endif
    }
    
    vec3 color = starryBackground(dir);
    
    if (hitGlass && hitOtherThings) {
        vec3 scenePoint = ro + dir * everythingTraveled;
        vec3 glassPoint = ro + dir * traveled;
        
        float sceneDistance = distance(ro, scenePoint);
        float glassDistance = distance(ro, glassPoint);
        
        if (glassDistance >= sceneDistance) {
            hitGlass = false;
        } else {
            hitOtherThings = false;
        }
    }
    
    if (hitOtherThings) {
        vec3 hitPoint = ro + dir * everythingTraveled;
        color = GetColor(dnmEverythingElse, hitPoint, everythingTraveled, dir);
        return color;
    }
    
    if (hitGlass) {
        // Reflective/Refractive Surface
        color = vec3(71.0/255.0, 142.0/255.0, 232.0/255.0);;
        vec3 hitPoint = ro + dir * traveled;
        vec3 hitNormal = GetNormal1(hitPoint);
        
        vec3 refractionDirection = normalize(refract(hitNormal, -dir,1.0/1.6));
        
        bool outside = dot(dir, hitNormal) < 0.0;
        
        float surfaceWidth = 0.001;
        vec3 bias = surfaceWidth * hitNormal;
        
#ifdef ALLOWREFLECTIONS
        vec3 reflectionDirection = normalize(reflect(dir, hitNormal));
        vec3 reflectionRayOrig = outside ? hitPoint + bias : hitPoint - bias;
        vec3 reflectionColor = RayMarchReflection(reflectionRayOrig, reflectionDirection);
        float reflection = 0.04 + 0.96 * pow(1.0 - dot(-dir, hitNormal), 5.0);
#endif
        
        vec3 refractionRayOrig = outside ? hitPoint - bias : hitPoint + bias;
        vec4 refractionDandM = RayMarchInteriorGlass(refractionRayOrig, refractionDirection);
        vec3 newHit =  refractionRayOrig + refractionDirection * refractionDandM.w;
        vec3 newNormal = GetNormal2(hitPoint);
        
        float light = dot(newNormal, normalize( vec3(0,5,-5)))*.5+.5;
        float glassTravelDist =  1.-clamp(distance(refractionRayOrig, newHit) / 16.0, 0., 1.);
        
#ifdef ALLOWREFLECTIONS
        color = mix(reflectionColor*(1.0-reflection)+refractionDandM.xyz, color, 0.3);
        
        //color = reflectionColor*(1.0-reflection)+refractionDandM.xyz;
        //color = mix(reflectionColor*(1.0-reflection), color, 0.5);
#else
        color = refractionDandM.xyz;
#endif
        
        float glassLight = dot(hitNormal, normalize( vec3(0,5,-5)))*.5+.5;
        
        float specularStrength = 0.5;
        vec3 lightColor = vec3(1);
        vec3 viewDir = normalize(ro - hitPoint);
        vec3 reflectDir = reflect(-normalize( vec3(0,5,-5)), hitNormal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64.);
        vec3 specular = specularStrength * spec * lightColor;
        
        color = color * (glassLight + specular);
        return color;
    }
    
    return color;
}

// Function 188
float march(inout vec3 p, vec3 dir)
{
    //if(!intersectBox(p-bbpos,dir,bbsize)) { enable_car=false; }
    //if(!(intersectBox(p-bbpos1,dir,bbsize1)||intersectBox(p-bbpos2,dir,bbsize2))) { enable_car=false; }
    vec3 p0=p;
    float eps=.0003;
    float dmin=1000.;
    bool findmin=false;
    float d=dist(p);
    vec3 pmin=p;
    for(int i=0;i<500;i++)
    {
        float dp=d;
        d=dist(p);
        p+=dir*d*1.;
#ifdef SHADOW
        if (d<dp) findmin=true;
        if (findmin && d<dmin) { dmin=d; pmin=p; }
#endif
        if (d<eps) return 0.;
    }
    return clamp(dmin/length(pmin-p0)/.05,0.,1.);
}

// Function 189
void ray_march_scene(Ray r, float k, inout vec3 c)
{
    float uniform_step = k;
    float jit = 1.;
    //jit = 50.*fract(1e4*sin(1e4*dot(r.dir, vec3(1., 7.1, 13.3))));
   
    float t_gen = 1.;

    float param_t = intersect_sphere(r, m.pos, RADIUS);
    if(param_t <= -1.)
        return;
    vec3 p = ray_interpolation(r, k*jit);        
     
    //rgb transparency               
    
    vec3 t_acc = vec3(1.);	// accumulated parameters for transparency
    float t_loc = transp(uniform_step, 14., ( clamp(smoothstep(.2, 3.*RADIUS, (RADIUS-length(p))) - abs( 2.*(fbm(p/8.)) ), 0., 1.)  ) );
    
    for(int s = 10; s <110; s++)
    {               
        float dist_dist = dot(p-cam.pos, p-cam.pos);
        float dist_center = length(m.pos-cam.pos);
        vec3 center = p-m.pos;

        float d = length(center)-RADIUS-.5-jit*k;
        float size = length(center)/RADIUS;

        if(sdf_rounded_cylinder( center, 1.20, 0.2, 5. ) < 0.)
        {
            
            #if COLOR            
            #if ANIM      
            	anim = iTime/5.;
            
            #endif
            
            
            float n = fbm( ( 
                (p)/( clamp(0., RADIUS+1., length(center)) + cos(PI-Psnoise(p/(30.)) )- 1./size*anim) //shockwave stuff
            			) )  ;; ;
            // 1./size is the speed of the wave propgation 
            /////////////////////

            float mask = smoothstep(0.,
                                   	70.*RADIUS,
                                    RADIUS/length(center));

            
            //Optical density/depth : dens for density
            float dens = ( clamp( mask,
                               	  0.,
                                  1.) *n);
                        
           if(length(p-cam.pos) >(dist_center+m.radius) 
              )//|| (k*dens  < -9.9))
        	{
         	break;
        	}
            //How colors (rgb) are absorbed at point p in the current iteration
            //k is the step size          
             vec3 rgb_t = exp(-vec3(
                		k * 25. * f(p.x) * dens, 
                      	k * 10. * dens,
              	      	k * 15. * f(p.z) * dens ));    
                
    		//blending
   			c += t_acc*vec3(1.)*(1.-rgb_t);
                        t_acc *= (rgb_t);           

            #endif
        }

        //if it will never be in the shape anymore, return;        
        
        p += r.dir*k;

        k = uniform_step;
    }
    

    //c =float(s)/vec3(50,150,20); return;

    #if COLOR

    #else
    c = vec3(t_gen); return;
    #endif
}

// Function 190
vec3 MarchRay(vec3 orig, vec3 dir)
{
    vec3 mPos = orig;
    vec3 lPos = orig;
    for(int i = 0;i < 96;i++)
    {
        float dMap = Map(mPos);
        if(dMap < eps)
        {
            if(dMap < 0.0)
            {
                mPos = (lPos + mPos)*0.5;
            }
            else
            {
                break;
            }
        }
        else
        {
            lPos = mPos;
            mPos += dir*dMap;
        }
        
    }
    
    return mPos;
}

// Function 191
vec3 raymarch(vec3 raydir, vec3 raypos){
    float distest;
    for(uint i = 0U; i < maxmarches; i++){
        distest = DE(raypos);
        if(distest < collisiondist){return vec3(1.0);}
        raypos += raydir*distest;
        if(length(raypos) > scenesize){break;}
    }
    return vec3(0.0);
}

// Function 192
mat3 LpStepMat (vec3 a)
{
  mat3 m1, m2;
  vec3 t, c, s;
  float b1, b2;
  t = 0.25 * a * a;
  c = (1. - t) / (1. + t);
  s = a / (1. + t);
  m1[0][0] = c.y * c.z;  m2[0][0] = c.y * c.z;
  b1 = s.x * s.y * c.z;  b2 = c.x * s.z;
  m1[0][1] = b1 + b2;  m2[1][0] = b1 - b2;
  b1 = c.x * s.y * c.z;  b2 = s.x * s.z;
  m1[0][2] = - b1 + b2;  m2[2][0] = b1 + b2;
  b1 = c.y * s.z;
  m1[1][0] = - b1;  m2[0][1] = b1;  
  b1 = s.x * s.y * s.z;  b2 = c.x * c.z;
  m1[1][1] = - b1 + b2;  m2[1][1] = b1 + b2; 
  b1 = c.x * s.y * s.z;  b2 = s.x * c.z;
  m1[1][2] = b1 + b2;  m2[2][1] = b1 - b2;
  m1[2][0] = s.y;  m2[0][2] = - s.y;
  b1 = s.x * c.y;
  m1[2][1] = - b1;  m2[1][2] = b1;
  b1 = c.x * c.y;
  m1[2][2] = b1;  m2[2][2] = b1;
  return m1 * m2;
}

// Function 193
void TestPlaneMarch(in vec3 rayPos, inout SRayHitInfo info, in vec4 plane, in SMaterial material)
{
    float dist = PlaneDistance(plane, rayPos);
    if (dist < info.dist)
    {
        info.rayMarchedObject = true;
        info.dist = dist;        
        info.normal = plane.xyz;
        info.material = material;
    }    
}

// Function 194
vec3 march(in vec3 ro, in vec3 rd) {
  	float d = 0.;
  	vec3 p = ro;
  	float li=0.;
  	for(float i=0.; i<200.; i++) {
    	float h = scene(p)*.5; // undershoot the march by half
    	if(abs(h)<.001*d) return vec3(d,i,1);
    	if(d>100.) return vec3(d,i,0);
    	d+=h;
    	p+=rd*h;
        li = i;
  	}
  	return vec3(d, li, 0);
}

// Function 195
vec2 march(vec3 ro, vec3 rd){
    if (bounding(ro, rd)){
        float t = 0.72, d;
        for (int i = 0; i < 96; i++){
            d = dist(ro + rd * t);
            t += d;

            if (d < 0.002) return vec2(t, d);
            if (d > 0.4) return vec2(-1.0);
        }
    }

    return vec2(-1.0);
}

// Function 196
vec3 MarchRay(vec3 origin,vec3 dir)
{
    vec3 marchPos = origin;
    for(int i = 0;i < MAX_STEPS;i++)
    {
        float sceneDist = Scene(marchPos);
        
        marchPos += dir * sceneDist * STEP_MULT;
        
        if(sceneDist < MIN_DIST)
        {
            break;
        }
    }
    
    return marchPos;
}

// Function 197
vec2 calcStep(float x, float st) 
{
	float ex = exp2(st);
	float kn = mod(x, ex);
	float an = -(TWO_PI / exp2(st + 1.0)) * kn;

	vec2 cs = vec2(cos(an), sin(an));

	bool isEven = mod(floor(x / ex), 2.0) == 0.0;

	vec2 res = vec2(0.0);

	if (isEven) {
		res.x = read(x, st).x + read(x + ex, st).x * cs.x - read(x + ex, st).y * cs.y;
		res.y = read(x, st).y + read(x + ex, st).x * cs.y + read(x + ex, st).y * cs.x;
	} else {
		res.x = read(x - ex, st).x - (read(x, st).x * cs.x - read(x, st).y * cs.y);
		res.y = read(x - ex, st).y - (read(x, st).x * cs.y + read(x, st).y * cs.x);
	}
	return res;
}

// Function 198
vec3 rayMarch(vec3 rayDir, vec3 cameraOrigin, vec3 lightPos)
{
    const int MAX_ITER = 50;
	const float MAX_DIST = 10.0;
    
    float totalDist = 0.0;
	vec3 pos = cameraOrigin;
	vec2 dist = vec2(EPSILON, 0.0);
    
    for(int i = 0; i < MAX_ITER; i++)
	{
		dist = distfunc(pos, lightPos - pos);
		totalDist += dist.x;
		pos += dist.x*rayDir;
        
        if(dist.x < EPSILON || totalDist > MAX_DIST)
		{
			break;
		}
	}
    
    return vec3(dist.x, totalDist, dist.y);
}

// Function 199
vec2 smoothstep_unchecked( vec2 x ) { return ( x * x ) * ( 3.0 - x * 2.0 ); }

// Function 200
void TestBoxMarch(in vec3 rayPos, inout SRayHitInfo info, in vec3 boxPos, in vec3 boxRadius, in float width, in SMaterial material)
{
    float dist = BoxDistance(boxPos, boxRadius, width, rayPos);
    if (dist < info.dist)
    {
        info.rayMarchedObject = true;
        info.dist = dist;
        
        vec3 relPos = max(abs(rayPos - boxPos) - boxRadius, 0.0f);
        int maxIndex = MaxCompIndex(relPos);
        if (maxIndex == 0)
        {
            info.normal = (rayPos.x < boxPos.x) ? vec3(-1.0f, 0.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
        }
        else if(maxIndex == 1)
        {
            info.normal = (rayPos.y < boxPos.y) ? vec3(0.0f, -1.0f, 0.0f) : vec3(0.0f, 1.0f, 0.0f);
        }
        else
        {
            info.normal = (rayPos.z < boxPos.z) ? vec3(0.0f, 0.0f, -1.0f) : vec3(0.0f, 0.0f, 1.0f);
        }
        
        info.material = material;
    }    
}

// Function 201
vec4 raymarchTrees( in vec3 ro, in vec3 rd, float tmax, vec3 bgcol, out float resT )
{
	vec4 sum = vec4(0.0);
    float t = tmax;
	for( int i=0; i<512; i++ )
	{
		vec3 pos = ro + t*rd;
		if( sum.a>0.99 || pos.y<0.0  || t>20.0 ) break;
		
		vec4 col = mapTrees( pos, rd );

		col.xyz = mix( col.xyz, bgcol, 1.0-exp(-0.0018*t*t) );
        
		col.rgb *= col.a;

		sum = sum + col*(1.0 - sum.a);	
		
		t += 0.0035*t;
	}
    
    resT = t;

	return clamp( sum, 0.0, 1.0 );
}

// Function 202
vec4 raymarch(vec3 rayOrigin, vec3 rayDir)
{
	float totalDist = 0.0;
	for(int j=0; j<MAXSTEPS; j++)
	{
		vec3 p = rayOrigin + totalDist*rayDir;
		float dist = distanceField(p);
		if(abs(dist)<EPSILON)	//if it is near the surface, return an intersection
		{
			return vec4(p, 1.0);
		}
		totalDist += dist;
		if(totalDist>=MAXDEPTH) break;
	}
	return vec4(0);
}

// Function 203
float ray_marching( vec3 origin, vec3 dir, float start, float end ) {
	float depth = start;
	for ( int i = 0; i < max_iterations; i++ ) {
        vec3 p = origin + dir * depth;
		float dist = dist_field( p ) / length( gradient( p ) );
		if ( abs( dist ) < stop_threshold ) {
			return depth;
		}
		depth += dist * 0.9;
		if ( depth >= end) {
			return end;
		}
	}
	return end;
}

// Function 204
vec3 march (vec3 origin, vec3 direction) {
    bool hit = false;
    float compoundedd = 0.;
    float closestSDF = 1e10;
    float rcount = 0.;
    vec3 finalTX = vec3(1.);
    vec3 objcol;
    for (int i=0; i<200; ++i) {
        if ((compoundedd>100. || origin.y>16.) && rcount == 0.) {
            return bg(direction, closestSDF);
        }
        float SDFp = SDF(origin);
        closestSDF = min(closestSDF, SDFp);
        float DE = SDF(origin);
        if (DE > .1 && DE < 100.) {
            DE *= .7;
        }
        if (SDFp < 1e-2) {
            origin += direction*SDFp*.99;
            objcol = TEXcolor(origin, direction);
            if (TEXrindex(origin) == 0. && rcount == 0.) {
            	return objcol;
            }
            if (rcount > 0.) {
                break;
            }
            float shiny = pow(.7, rcount+1.);
            direction = reflect(direction, RFX(origin));
            origin = origin+direction*.1;
            objcol = objcol*(1.-shiny)+bg(direction, closestSDF)*shiny;
            rcount++;
        }
        origin += direction*DE;
        compoundedd += DE;
    }
    if (rcount > 0.) {
        float shiny = pow(.7, rcount+1.);
        return objcol*(1.-shiny)+bg(direction, closestSDF)*shiny;
    }
    return bg(direction, closestSDF);
}

// Function 205
float march(vec3 s, vec3 d)
{
    mat3 rotMat = rotationMatrix(vec3(1,1,0), iTime*0.2);
    float dist = 1.0;	// distance
    float distd = 0.1;
    for(int i = 0; i < MARCHLIMIT; i++)
    {
        distd = map(s + d*dist, rotMat)*MARCHSTEPFACTOR;
        if(distd < 0.00001 || dist > 10.0)
            break;
        dist += distd;
    }
    
	return min(dist, 100.0);
}

// Function 206
float aaStep(in float compValue, in float gradient){
  float halfChange = fwidth(gradient) * 0.5f;
  //base the range of the inverse lerp on the change over one pixel
  float lowerEdge = compValue - halfChange;
  float upperEdge = compValue + halfChange;
  //do the inverse interpolation
  return( clamp((gradient - lowerEdge) / (upperEdge - lowerEdge), 0.0f, 1.0f) );
}

// Function 207
float stepNoise(vec2 p) {
    return noise(floor(p));
}

// Function 208
vec4 raymarch (vec3 ro, vec3 rd)
{
	for (int i=0;i<16;i++)
	{
		float t = map(ro);
		if (t<0.001) return ProceduralSkybox(ro,rd);     
		ro+=t*rd;
	}
	return vec4(0,0,0,1);
}

// Function 209
float bumpstep(float edge0, float edge1, float x)
{
    return 1.0-abs(clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)-.5)*2.0;
}

// Function 210
void TestPlaneMarch(in vec3 rayPos, inout SRayHitInfo info, in vec4 plane, in SMaterial material)
{
    float dist = PlaneDistance(plane, rayPos);
    if (dist < info.dist)
    {
        info.objectPass = OBJECTPASS_RAYMARCH;
        info.dist = dist;        
        info.normal = plane.xyz;
        info.material = material;
    }    
}

// Function 211
void vs_pace_halfstep( inout VehicleState vs, VehicleInputs vi,
                       float dt, bool localphysics, bool pausemode )
{
    vs_pace_FSG( vs, vi, dt );
    vs_pace_throttle( vs, vi, dt );
    vs_pace_EAR_and_trim( vs, vi, dt );

    if( localphysics )
    {
        if( !pausemode )
        {
		    vs.localr_diff += dt * vs.localv;
            vs.localr = vs.localr_base + vs.localr_diff;
        }
    }
    else
    {
		if( !pausemode )
        	vs.orbitr += dt * vs.orbitv;
    }
}

// Function 212
vec4 raymarch( in vec3 ro, in vec3 rd, in vec3 bgcol, in ivec2 px )
{
	vec4 sum = vec4(0.0);

	float t = 0.0;//0.05*texelFetch( iChannel0, px&255, 0 ).x;

    MARCH(40,map5);
    MARCH(40,map4);
    MARCH(30,map3);
    MARCH(30,map2);

    return clamp( sum, 0.0, 1.0 );
}

// Function 213
vec2 rayMarching(vec3 ro, vec3 rd) {
    float tmax = MAX_DIST;
    float t = 0.0;
    vec2 result = vec2(-1.0);
    
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        vec2 res = sceneSDF(p);
        if (res.x < EPSILON || t > tmax) break;
        result.x = t;
        result.y = res.y;
        t += res.x;
    }
    if (t > tmax) {
        result.x = tmax;
        result.y = -1.0;
    }
    return result;
}

// Function 214
vec4 march(vec3 p, vec3 nv, vec2 fragCoord) {
    vec2 tRange;
    float didHitBox;
    boxClip(BOX_MIN, BOX_MAX, p, nv, tRange, didHitBox);
    tRange.s = max(0.0, tRange.s);

    vec4 color = vec4(0.0);
    if (didHitBox < 0.5) {
        return color;
    }
    
    bool noColor = texelFetch(iChannel2, ivec2(KEY_A,0), 0).x > 0.5;
	bool noDensity = texelFetch(iChannel2, ivec2(KEY_D,0), 0).x > 0.5;
    
    float t = tRange.s + min(tRange.t-tRange.s, RAY_STEP)*hash12(fragCoord);
    int i=0;
    for (; i<150; i++) { // Theoretical max steps: (BOX_MAX-BOX_MIN)*sqrt(3)/RAY_STEP
        if (t > tRange.t || color.a > QUIT_ALPHA) { break; }

        vec3 rayPos = p + t*nv;
        vec3 lmn = lmnFromWorldPos(rayPos);

        float density;
        float lightAmount;
        readLMN(lmn, density, lightAmount);

        vec3 cfrag = noColor ? vec3(1.0) : colormap(0.7*density+0.8);
        density = noDensity ? 0.1 : density;

        float calpha = density * MAX_ALPHA_PER_UNIT_DIST * RAY_STEP;
        vec4 ci = clamp( vec4(cfrag * lightAmount, 1.0)*calpha, 0.0, 1.0);
        color = blendOnto(color, ci);

        t += RAY_STEP;
    }

    float finalA = clamp(color.a/QUIT_ALPHA, 0.0, 1.0);
    color *= (finalA / (color.a + 1e-5));

    bool showSteps = texelFetch(iChannel2, ivec2(KEY_F,0), 0).x > 0.5;
    return showSteps ? vec4(vec3(float(i)/150.0), 1.0) : color;
}

// Function 215
float inv_sstep2(float x)
{
	x = clamp(x, 0.0, 1.0);
	float ix = (1.0 - x);
    x = sqrt(x);
    ix = sqrt(ix);
    return x / (x + ix);
}

// Function 216
uint tausStep(uint z, uint S1, uint S2, uint S3, uint M) {
  uint b = ((z << S1) ^ z) >> S2;
  return ((z & M) << S3) ^ b;
}

// Function 217
vec4 march(inout vec3 pos, vec3 dir)
{
    // cull the sphere
    if(length(pos-dir*dot(dir,pos))>1.05) 
    	return vec4(0,0,0,1);
    
    float eps=0.001;
    float bg=1.0;
    for(int cnt=0;cnt<64;cnt++)
    {
        float d = dist(pos);
        pos+=d*dir;
        if(d<eps) { bg=0.0; break; }
    }
    vec3 n = getGrad(pos,.001);
    return vec4(n,bg); // .w=1 => background
}

// Function 218
vec3 march(vec3 ro, vec3 rd, float dither, float var)
{
    float value = 0.;
    float t = dither;
    float densitySum = 0.;

    float stepLength = maxDist / float(nStep);
    vec3 color = vec3(0.01,0.02,0.05)*1.;
    for (int i = 0; i < nStep; i++)
    {
        
        vec3 samplePos = ro + t * rd ; 
    	float sampleNoise = map (samplePos);
        densitySum += sampleNoise;
    	
        //light1
        vec3 lightPos1 = vec3 (-18,1.8,0);         
        vec3 light1 = calculateLight(samplePos, lightPos1, vec3 (0.6,0.25,0.15), 250.);
 		
        //light2
        vec3 lightPos2 = vec3 (0.,0.,-15.);
        vec3 light2 = calculateLight(samplePos, lightPos2, vec3 (0.1 ,0.2,0.6), 200.);
     	
        //light3
        float n = 1. * (noise(0.7*samplePos.y)-0.5)- 0.2*samplePos.y;
        vec3 lightPos3 = vec3 (n,samplePos.y,10.*(hash(floor(0.1*iTime))-0.5));     
        float storm =  mix (1.2,0., sign(fract(-0.1+0.1*iTime)-0.15 )) * noise (20.*iTime);
        vec3 light3 = calculateLight(samplePos, lightPos3, vec3 (1.,1.,1.), storm);

        vec3 ambientColor = vec3 (.0,0.025,0.025);
        
        color += exp(- t*(densitySum/float(i+1)))  * sampleNoise * (ambientColor + light1 + light2 + light3);
        
        t +=  stepLength * var;
    }
    
   
    return color;
}

// Function 219
float smootherstep(float edge0, float edge1, float x) {
    float t = (x - edge0)/(edge1 - edge0);
    float t1 = t*t*t*(t*(t*6. - 15.) + 10.);
    return clamp(t1, 0.0, 1.0);
}

// Function 220
float raymarch( vec3 o, vec3 target, float start_time, float timeDir, out vec4 objPos )
{
    objPos = vec4(0.,0.,0.,1.);
    vec3 delta = target - o;
    float dist = length( delta );
   
    vec3 d = delta / dist;
    
    float eps = 0.001;
    
    float x = 0.;
    float t = start_time;
    
    for( int i = 0; i < 150; i++ )
    {
        float dx = sdf( o + x * d, t, objPos );
        
        if( abs(dx) < eps )
        {
            return x;
        }

        dx *= .7;
        x += dx;
        if( x >= dist )
            break;
        
        // progress time as ray advances to simulate flight time of photon
        t += timeDir * light_time_per_m() * dx;
    }
    
    return ZFAR;
}

// Function 221
Hit raymarch(Ray ray) {
 
    float d  = 0.;
    int iter = 0;
    
    for(int i = 0; i < MAX_ITERATIONS; i++) {
     
        d += dstScene(ray.ori + ray.dir * d) * .75;
        
        if(d <= MIN_DISTANCE || d > FAR_PLANE) {
         
            iter = i;
            break;
            
        }
        
    }
    
    return Hit(d,iter);
    
}

// Function 222
float animStep(float t, float stepIndex, float delay) {
    return animStep(t - delay, stepIndex);
}

// Function 223
float oversteer_step_cubic( float x, float s )
{
	s = -s;
	x = saturate( x );
	if ( x > 0.5 ) {
		x = 1.0 - x;
		return 1.0 + ( ( ( -6.0 - s * 2.0 ) + ( 8.0 + s * 4.0 ) * x ) * x ) * x;
	}
	return -( ( ( -6.0 - s * 2.0 ) + ( 8.0 + s * 4.0 ) * x ) * x ) * x;
}

// Function 224
MarchResult MarchRay(vec3 orig,vec3 dir)
{
    float steps = 0.0;
    float dist = 0.0;
    float id = 0.0;
    
    for(int i = 0;i < MAX_STEPS;i++)
    {
        vec2 object = Scene(orig + dir * dist);
        //Add the sky dome and have it follow the camera.
        object = opU(object, -sdSphere(dir * dist, MAX_DIST, SKYDOME));
        
        dist += abs(object.x) * STEP_MULT;
        
        id = object.y;
        
        steps++;
        
        if(abs(object.x) < MIN_DIST * dist)
        {
            break;
        }
    }
    
    MarchResult result;
    
    result.position = orig + dir * dist;
    result.normal = Normal(result.position);
    result.dist = dist;
    result.steps = steps;
    result.id = id;
    
    return result;
}

// Function 225
vec4 raymarch( in vec3 ro, in vec3 rd )
{
    vec4 sum = vec4(0, 0, 0, 0);
    
    // setup sampling - compute intersection of ray with 2 sets of planes
    float2 t, dt, wt;
	SetupSampling( t, dt, wt, ro, rd );
    
    // fade samples at far extent
    float f = .45; // magic number - TODO justify this
    float endFade = f*float(SAMPLE_COUNT)*PERIOD;
    float startFade = .99*endFade;
    
    for(int i=0; i<SAMPLE_COUNT; i++)
    {
        if( sum.a > 0.99 ) continue;

        // data for next sample
        vec4 data = t.x < t.y ? vec4( t.x, wt.x, dt.x, 0. ) : vec4( t.y, wt.y, 0., dt.y );
        // somewhat similar to: https://www.shadertoy.com/view/4dX3zl
        //vec4 data = mix( vec4( t.x, wt.x, dt.x, 0. ), vec4( t.y, wt.y, 0., dt.y ), float(t.x > t.y) );
        vec3 pos = ro + data.x * rd;
        pos *= 1.1;
        float w = data.y;
        t += data.zw;
        
        // fade samples at far extent
        w *= smoothstep( endFade, startFade, data.x );
        
        vec4 col = map( pos );
        
        // iqs goodness
        float dif = clamp((col.w - map(pos+0.6*sundir).w)/0.6, 0.0, 1.0 );
        vec3 lin = vec3(0.54, 0.55, 0.60)*1.79 + 0.35*vec3(0.85, 0.57, 0.3)*dif;
        col.xyz *= lin;
        
        col.xyz *= col.xyz;
        
        col.a *= col.a;
        col.rgb *= col.a;

        // integrate
        sum += col * (1.2 - sum.a) * w * 0.66;
    }

    sum.xyz /= (0.001+sum.w);
     vec3 hsv = rgb2hsv(sum.xyz);
    hsv.x -= 0.06;
    hsv.y -= 0.02;
    hsv.z *=  1.2;
    hsv.z += 0.063;
    sum.xyz = hsv2rgb(hsv);

    return clamp( sum, 0.0, 1.0 );
}

// Function 226
endif
raymarch( in vec3 start, in vec3 dir, inout float t, in float t_max )
{
    MPt mp;
    for ( int it=0; it!=120; ++it )
    {
        vec3 here = start + dir * t;
        mp = map( here );
        if ( mp.distance < ( T_EPS * t ) || t > t_max )
        {
        	break;
        }
        #if 1
        // NOTE(theGiallo): this is to sample nicely the twisted things
        t += mp.distance * 0.4;
        #else
        t += mp.distance;
        #endif
    }
    if ( t > t_max )
    {
        t = -1.0;
    }
    return mp;
}

// Function 227
vec3 ray_marching( vec3 origin, vec3 dir, float start, float end ) {
		
		float depth = start;
		vec3 salida = vec3(end);
		vec3 dist = vec3(2800.0);
		
		for ( int i = 0; i < max_iterations; i++ ) 		{
			if ( dist.x < stop_threshold || depth > end ) break;
                dist = map( origin + dir * depth );
                depth += dist.x;
				dist.y = float(i);
		}
		
		salida = vec3(depth, dist.y, dist.z);
		return salida;
	}

// Function 228
float raymarch(vec3 ray_start, vec3 ray_dir, out float dist, out vec3 p, out int iterations) {
    dist = 0.0;
    float minStep = 0.1;
	vec2 mapRes;
    for (int i = 1; i <= MAX_RAYMARCH_ITER; i++) {
        p = ray_start + ray_dir * dist;
        mapRes = map(p, ray_dir);
        if (mapRes.y < MIN_RAYMARCH_DELTA) {
           iterations = i;
           return mapRes.x;
        }
        dist += max(mapRes.y, minStep);
    }
    return -1.;
}

// Function 229
vec2 ray_march(vec3 cam_pos, vec3 cam_dir){
    float t_near = 0.0;
    for(int i = 0; i < STEP_MAX; i++){
        vec3 p = cam_pos + cam_dir * t_near; // t_near is how far we can go along ray without hitting object
        vec2 dist = scene_sdf(p);
        t_near += dist.x;
        // Check if we missed entirely or hit something
        // > DIST_MAX then we missed all objects, less than EPSILON, we hit an object 
        if(t_near > DIST_MAX){ 
            return vec2(-1., -1);
        }else if(dist.x < EPSILON){
            return vec2(t_near, dist.y); 
        }
    }
    
    return vec2(-1., -1);
}

// Function 230
float rayMarch(vec3 eye, vec3 marchingDirection, 
               float start, float end) {
	// Define starting depth 
    float depth = start;
    // March until maxMarchSteps is reached
    for (int i = 0; i < maxMarchSteps; i++) {
        // Obtain distance from closest surface 
        float dist = sceneSDF(eye + depth *
                              marchingDirection);
        // Determine if marched inside surface
        if (dist < epsilon) {
            // Inside scene surface
            return depth; 
        }
        // Update depth 
        depth += dist;
        // Determine if marched too far 
        if (depth >= end) {
            // Return farthest allowable
            return end;
        }
    }
    // Return distance if marched more than max steps
    return end;
}

// Function 231
hitInfo rayMarchArray(vec3 origin, vec3 dir)
{
    vec3 size = vec3(0.95,0.95,0.95);
    float t=1.0;
    hitInfo info;
    for(int i=0; i < 100; ++i)
    {
        //Cube position
        vec3 p = origin+t*dir;
        vec3 cubePos = floor(p/CELL_SIZE)*CELL_SIZE+0.5*CELL_SIZE;
        
        //rotation values
        float yaw = sin(iTime+cubePos.x+cubePos.y+cubePos.z);
        float pitch = iTime/1.0+sin(iTime+cubePos.x+cubePos.y+cubePos.z);
    	
        //rotated ray origin and direction
        vec3 rd = rotate(dir,yaw,pitch);
		vec3 ro = rotate(origin-cubePos,yaw,pitch);
        
        //ray-cube intersection function
        info = rayCubeIntersec(ro,rd,size);
        
        //check for hit : stop or continue.
        if(info.dist<MAX_DIST)
        	break;
        
        //Step into the next cell.
        t = t+CELL_SIZE; 
    }
    return info;
}

// Function 232
float raymarch(vec3 ray_origin, vec3 ray_direction) {
	float d = 0.0;
	
	for (int i = 0; i < max_steps; i++) {
		vec3 new_point = ray_origin + ray_direction*d;
		float s = get_distance(new_point);
		if (s < epsilon) return d;
		d += s;
		if (d > max_distance) return max_distance;
	}
	return max_distance;
}

// Function 233
float RAYMARCH_DFSS( vec3 o, vec3 L, float coneWidth )
{
    //Variation of the Distance Field Soft Shadow from : https://www.shadertoy.com/view/Xds3zN
    //Initialize the minimum aperture (angle tan) allowable with this distance-field technique
    //(45deg: sin/cos = 1:1)
    float minAperture = 1.0; 
    float t = 0.0; //initial travel distance, from geometry surface (usually, pretty close)
    float dist = 10.0;
    for( int i=0; i<7; i++ )
    {
        vec3 p = o+L*t; //Sample position = ray origin + ray direction * travel distance
        float dist = DF_composition( p ).d;
        dist = min(dist,t);
        float curAperture = dist/t; //Aperture ~= cone angle tangent (sin=dist/cos=travelDist)
        minAperture = min(minAperture,curAperture);
        //Step size : limit range (0.02-0.42)
        t += 0.02+min(dist,0.4);
    }
    
    //The cone width controls shadow transition. The narrower, the sharper the shadow.
    return saturate(minAperture/coneWidth); //Should never exceed [0-1]. 0 = shadow, 1 = fully lit.
}

// Function 234
vec4 raymarch( in vec3 ro, in vec3 rd, in vec3 bgcol )
{
	vec4 sum = vec4(0.0);

	float t = (50.0-ro.y)/rd.y;
    
	if( t>0.0 )
    {
        for(int i=0; i<512; i++ )
        { 
            vec3  pos = ro + t*rd;
            float den = map( pos );
            if( den>0.001 )
            { 
                // lighting
                float dif = clamp( (den - map(pos+50.0*sundir))*1.0, 0.0, 1.0 ); 
                vec3 lin = vec3(0.5,0.7,0.9)*0.5 + vec3(1.0, 0.7, 0.5)*dif*5.0;
                vec4 col = vec4( mix( vec3(1.0,0.95,0.8), vec3(0.1,0.2,0.3), sqrt(clamp(den,0.0,1.0) )), den );
                col.xyz *= lin;
                col.xyz = mix( col.xyz, bgcol, 1.0-exp(-0.0005*t) );
                col.a *= 0.1;
                // front to back blending    
                col.rgb *= col.a;
                sum = sum + col*(1.0-sum.a);
                // early skip
                if( sum.a > 0.99 || pos.y>300.0) break;
            }
            t += 2.0;
        }
    }
    
    return clamp( sum, 0.0, 1.0 );
}

// Function 235
SRayHit 	rayMarchBlobs(SBlob blob1, SBlob blob2, vec3 rayOrigin, vec3 rayDirection, float farDist)
{
    SRayHit	hit;
    vec3 	currentPos = rayOrigin;
    float 	currentDist = 3.0f;

    while (currentDist < farDist)
    {
        currentPos += rayDirection * rayMarchDist;
        currentDist += rayMarchDist;
        
        float 	sumInfluence = 0.0f;
        
        vec3 	curToBlob1 = blob1.m_Position - currentPos;
        float 	blobInfluence1 = blob1.m_Intensity / dot(curToBlob1, curToBlob1);

		sumInfluence += blobInfluence1;
        
    	vec3 	curToBlob2 = blob2.m_Position - currentPos;
        float 	blobInfluence2 = blob2.m_Intensity / dot(curToBlob2, curToBlob2);

		sumInfluence += blobInfluence2;

        if (sumInfluence > solidSurfThreshold)
        {
            hit.m_Distance = currentDist;
            hit.m_Position = currentPos;
            return hit;
        }
	}
    hit.m_Distance = infinity;
	hit.m_Position = vec3(infinity);
    return hit;
}

// Function 236
Moon MarchMoon(vec3 ro, vec3 rd, vec3 sun_dir, vec3 sun_col)
{
    // the moons postion (sense it dosent change mid frame)
    vec3 mp = vec3(sin(iTime) * 10.5, sin(iTime) * 0.1, cos(iTime) * 10.5);
    float dst;
    vec3 p = ro;
    float dfs = 0.;
    bool collided = false;
    // setting into the scene
    for (int s = 0; s < 35; s++)
    {
        // getting the distance to the moon
        dst = length(p - mp) - 1.;
        p += rd * dst;
        dfs += dst;
        
        // checking if the ray has collided
        if (dst < 0.1)
        {
            collided = true;
            break;
        }
        // checking if the ray has passed the moon
        else if (dfs > 40.) break;
    }
    
    // getting the normal
    float d = gd(p, mp);
    vec2 e = vec2(0.01, 0);
    
    vec3 normal = d - vec3(
        gd(p - e.xyy, mp),
        gd(p - e.yxy, mp),
        gd(p - e.yyx, mp));
    
    normal = normalize(normal);
    // bump mapping some noise onto the normal to make the moon less smooth
    vec3 surface_noise = SampleNoiseV3(p - mp);  // getting the noise at object coords not world so the noise texture dosent move with the moon
    normal = normalize(normal + normal * (surface_noise * 0.75));
    
    // coloring the moon
    vec3 color = vec3(0., 0., 0.);
    if (collided)
    {
        // setting the moons color the noise value
        color = vec3(surface_noise.x * 0.25 + 0.5);  // add color and shading and bump mapping
        // lighting and enshadoing the moon
        color *= max(dot(normal, sun_dir), 0.);
        // adding specular lighting to the moon
        SpecularLight spec = Specular(0.8, normal, rd, sun_dir);
        color = color * spec.diffuse + sun_col * spec.highlight;
    }
    // returning the data
    Moon moon = Moon(dfs, p, normal, color, collided);
    return moon;
}

// Function 237
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

// Function 238
vec3 distmarch( vec3 ro, vec3 rd, float maxd )
{
    
    float dist = 6.;
    float t = 0.;
    float material = 0.;
    float heat = 0.;
    for (int i=0; i < DISTMARCH_STEPS; i++) 
    {
        if (( abs(dist) < EPSILON || t > maxd) && material >= 0. ) 
            continue;

        t += dist;
        vec3 dfresult = scenedf( ro + t * rd, rd );
        dist = dfresult.x;
        material = dfresult.y;
        heat = dfresult.z;
    }

    if( t > maxd ) material = -1.0; 
    return vec3( t, material, heat );
}

// Function 239
float rayMarch(vec3 rayOrig, vec3 rayDir) {
	float dist = 0.;
    
    for (int i = 0; i < MAX_STEPS; ++i) {
		vec3 p = rayOrig + dist * rayDir;
		float sceneDist = sceneDist(p);
        
        dist += sceneDist;
       
        if (dist > MAX_DIST || sceneDist < SURF_DIST) {
        	break;
        }
	}
    
    return dist;
}

// Function 240
vec4 raymarch( in vec3 rayo, in vec3 rayd, in vec2 expInter, in float t, out float d )
{
    vec4 sum = vec4( 0.0 );
    
    float step = 1.5 / float(steps);
     
    // start iterating on the ray at the intersection point with the near half of the bounding sphere
	//vec3 pos = rayo + rayd * (expInter.x + step*texture( iChannel2, gl_FragCoord.xy/iChannelResolution[0].x ).x);		// dither start pos to break up aliasing
	//vec3 pos = rayo + rayd * (expInter.x + 1.0*step*fract(0.5*(gl_FragCoord.x+gl_FragCoord.y)));	// regular dither
	vec3 pos = rayo + rayd * (expInter.x);	// no dither

    float march_pos = expInter.x;
    d = 4000.0;
    
    // t goes from 0 to 1 + mult delay. that is 0 to 1 is for one explosion ball. the delay for time distribution of the multiple explosion balls.
    // t_norm is 0 to 1 for the whole animation (incl mult delay).
    float t_norm = t / tmax;
    float smooth_t = sin(t_norm*2.1);	//sin(t*2.);

	//float bright = 6.1;
	float t1 = 1.0 - t_norm;	// we use t_norm instead of t so that final color is reached at end of whole animation and not already at end of first explosion ball.
    //float bright = 3.1 + 18.0 * t1*t1;
	//float bright = 3.1 + 1.4 * t1*t1;
	//float bright = 3.1 + 4.4 * t1*t1;
	float bright = brightness.x + brightness.y * t1*t1;
	//float bright = smoothstep(0.0, 30.1, 1.0);
	//float bright = smoothstep(20.0, 3.1, 1.0);
    //float bright = 10.;

    for( int i=0; i<steps; i++ )
    {
        if( sum.a >= 0.98 ) { d = march_pos; break; }
        if ( march_pos >= expInter.y ) break;
        
        float rad, r, rawDens;
        vec4 col;
        calcDens( pos, rad, r, rawDens, t, smooth_t, col, bright );

        if ( rawDens <= 0.0 )
        {
            float s = step * 2.0;
            pos += rayd * s;
            march_pos += s;
            continue;
        }
        
#ifdef OLD_COLORING
        contributeDens( rad, r, rawDens, bright, col, sum );
#else
        contributeColor( col, sum );
#endif
		
		// take larger steps through low densities.
		// something like using the density function as a SDF.
		float stepMult = 1.0 + (1.-clamp(rawDens+col.a,0.,1.));
		// step along ray
		pos += rayd * step * stepMult;
        march_pos += step * stepMult;

		//pos += rayd * step;
	}

#ifdef SHOW_BOUNDS
    if ( sum.a < 0.1 )
        sum = vec4(0.,0.,.5,0.1);
#endif
	
    return clamp( sum, 0.0, 1.0 );
}

// Function 241
float marching(vec3 ro, vec3 rd) 
{
    float tmax = MAX_DISTANCE;
    float t = 0.001;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        float h = sceneSDF(p);
        if (t < EPSILON || t > tmax) break;
        t += h;
    }
    return t;
}

// Function 242
float linear_step(float low, float high, float value)
{
    return clamp((value-low)*(1./(high-low)), 0., 1.);
}

// Function 243
float march(vec3 ro, vec3 rd, float tMax)
{
    float t = MIN_DIST;
    for (int i = 0; i < MAX_STEPS; ++i) {
        float h = scene(ro + rd * t);
        if (h < MIN_DIST || h > tMax)
            break;
        t += h;
    }
    if (t > tMax)
        t = 1e38; // WebGL doesn't like the divide-by-zero INF-trick
        // t = 1. / 0.; // INF
    return t;
}

// Function 244
vec4 Raymarch_GetColor( vec3 vRayOrigin, vec3 vRayDir )
{
    vec4 vColor = vec4(0);
    
    return vColor;
}

// Function 245
float rayMarch(Ray r) {
    float t = EPSILON;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(r.o + t * r.d);
        if (dist < EPSILON) {
			return t;
        }
        t += dist;
        if (t >= MAX_DIST || i == MAX_MARCHING_STEPS - 1) {
            return MAX_DIST;
        }
    }
    return t;
}

// Function 246
float RayMarchDist(vec3 ray_origin, vec3 ray_dir, float initial_dist, float iTime)
{
    float dist = initial_dist;
    
    for (int i = 0; i < MAX_STEPS; ++i)
    {
        vec3 p = ray_origin + ray_dir * dist;
        
        float nearest = GetSceneDist(p, iTime);
        
        if (nearest < MIN_DIST)
            break;
        
        dist += nearest;
        
        if (dist > MAX_DIST)
            break;
    }
    
    return dist;
}

// Function 247
void Raymarch( const in C_Ray ray, out C_HitInfo result, const int maxIter, const float fTransparentScale )
{        
    result.fDistance = GetRayFirstStep( ray );
    result.vObjectId.x = 0.0;
        
    for(int i=0;i<=kRaymarchMaxIter;i++)              
    {
        result.vPos = ray.vOrigin + ray.vDir * result.fDistance;
        vec4 vSceneDist = GetDistanceScene( result.vPos, fTransparentScale );
        result.vObjectId = vSceneDist.yzw;
        
        // abs allows backward stepping - should only be necessary for non uniform distance functions
        if((abs(vSceneDist.x) <= kRaymarchEpsilon) || (result.fDistance >= ray.fLength) || (i > maxIter))
        {
            break;
        }                        

        result.fDistance = result.fDistance + vSceneDist.x; 
    }


    if(result.fDistance >= ray.fLength)
    {
        result.fDistance = 1000.0;
        result.vPos = ray.vOrigin + ray.vDir * result.fDistance;
        result.vObjectId.x = 0.0;
    }
}

// Function 248
vec4 raymarchGas(vec3 ro, vec3 rd,
                  in float tmin, in float tmax,
                  in vec2 uv,
                  in vec3 Lp,
                  in vec3 Lcolor,
                  out float blendAlpha)
{
    if(tmax<0. || tmin<0.) return vec4(1., 0., 0., 0.);
    float maxTargetTravel = tmax-tmin;
    float cStep = 0.08;
    float vStep = maxTargetTravel*cStep;
    vec3 totalRadiance = vec3(0.);
    float Tr = 1.;
    float screenShift = 1.0*hash2D(rd.xy+fract(iTime));
    tmin -= vStep*screenShift;
    float d = tmin;
    #if defined(USE_SKY) && defined(CLOUDS_RAYMARCH_MIE)
    vec3 miec;
    vec3 Pa = ro+rd*tmax;
    Lcolor = getSkyLight(Pa, RotXV3(normalize(Lp-Pa), 90.*(-1.+2.*hash2D(10.*uv+fract(iTime)))), Lp, SKY_BETA_RAY, SKY_BETA_MIE, miec, SKY_MIE_HEIGHT);
    Lcolor = miec;
    #endif
    for(d=tmin; d<tmax; d+=vStep)
	{
        vec3 P = ro+rd*d;        
        vec3 L = normalize(Lp-P);
        // map
        #ifdef USE_TEXTURE_MAP
        float map = 1.;
        #else
        float map = max(0., length(P-TARGETPOS)); // not 1-x, magic
        #endif
        // density
        float den = map*density(P, TARGETPOS);
        if(den>0.)
        {
            float sumLuminance = 0.;
            float transmittance = 1.;
            float phase = 0.;
            {
            	float mu = dot(rd, L);
                float cosAngle = 0., g = 0.;
                cosAngle = mu; g = mix(-.30, .50, .5);//.5*(1.+cosAngle));
                phase = mix(phaseHenyeyGreenstein(-.2, mu), phaseHenyeyGreenstein(.8, mu), .5);
            }
            sumLuminance += phase * transmittance;
            float accDensity = 0., accLength = 0.;
            #ifdef USE_TEXTURE_MAP
            float stepL = (CLOUDEND-CLOUDSTART)*0.01;
            #else
            float stepL = 0.1;
            #endif
            for(int i = 0;i < 4; ++i)
            {
                vec3 Pl = P+L*float(i)*stepL;
        		#ifdef USE_TEXTURE_MAP
                	float lmap = 1.0;
                #else
                	float lmap = max(0., TARGETSIZE-length(Pl-TARGETPOS))/TARGETSIZE;
                #endif
                float det = lmap*density(Pl, TARGETPOS);

                #ifdef USE_TEXTURE_MAP
                transmittance *= BeersPowder(det, stepL, 1.);
                #else
                transmittance *= exp(-det*float(i)*stepL);
                accDensity += det;
                accLength += float(i)*stepL;
                #endif
                stepL *= 1.6;
                if(transmittance<=0.05) break;
            }
            #ifdef USE_TEXTURE_MAP
            // Use computed transmittance for the shadowing, then overwrite it to compute the proper
            // transmittance
            sumLuminance = phase * SUNINTENSITY * transmittance;
            //
            transmittance = BeersPowder(den, vStep, 1.);
            #else
            sumLuminance = phase * SUNINTENSITY * exp(-accDensity*accLength);
            #endif
            float nPy = length(P-EARTHPOS);
            float amb = mix(0.05, 0.2, remap01(nPy, CLOUDSTART, CLOUDEND));
            vec3 radiance = (amb + sumLuminance*Lcolor)*den;
            // scattering
            totalRadiance += Tr * (radiance-transmittance*radiance) / den;
            Tr *= transmittance;
            if(Tr<=0.05) break;
            vStep = maxTargetTravel*cStep*.5;
        }
        else
           vStep = maxTargetTravel*cStep;
	}
    {
        vec3 Pmax = ro+rd*(tmax-screenShift);
        float mu = dot(rd, normalize(Lp-Pmax));
        //d = 1.;
        float phase = 0.;
        {
            float cosAngle = 0., g = 0.;
            cosAngle = mu; g = mix(-.30, .50, 0.5);
            phase = phaseHenyeyGreenstein(g, cosAngle);
        }
        vec2 P2d = 0.5*(Pmax.xz/Pmax.y);
        P2d.y = 1.-P2d.y;
        float tex = remap01(texture(iChannel3, 100.1*P2d
    	#if !defined(CLOUDS_FIXED_POS)
                                    +.001*iTime
    	#endif
                                   ).r, .4, 1.);
        tex = pow(tex, 1.);
        float mainForm = clamp(tex*tex*(3.-2.*tex), 0., 1.);
        float den = max(0., mainForm-.2*fbm_hash(20.*P2d+iTime));
        if(den>0.)
        {
            float transmittance = BeersPowder(den, 2., 1.);
            float lum = phase * transmittance;
            vec3 radiance = (0.08+lum*Lcolor)*den;
            totalRadiance += Tr * (radiance-transmittance*radiance) / den;
            Tr *= transmittance;
        }
    }   
    float rdy = remap01(tmin, 1e5, 6e5);
    rdy = 1.-(exp(-8.*rdy));
    
    blendAlpha = clamp(rdy, 0., 1.);
	return vec4(Tr, totalRadiance);
}

// Function 249
void MarchPOV(inout RayInfo r, float startTime
){//dpos = vec3(-2.2,0,0)
 ;//lorentzF = LorentzFactor(length(dpos))
 ;float speedC = length(dpos)/SpeedOfLight
 ;vec3 nDpos = vec3(1,0,0)
 ;if(length(dpos)>0.)nDpos = normalize(dpos)
 ;//shrink space along vel axis (length contraction of field of view)
 ;float cirContraction = dot(nDpos,r.dir)*(LorentzFactor(length(LgthContraction*dpos)))
 ;vec3 newDir = (r.dir - nDpos*dot(nDpos,r.dir)) + cirContraction*nDpos
 ;r.dir = normalize(newDir)
 ;float dDirDpos = dot(dpos,r.dir)
 ;// Aberration of light, at high speed (v) photons angle of incidence (a) vary with lorenz factor (Y) :
 ;// tan(a') = sin(a)/(Y*(v/c + cos(a)))
 ;// velComponentOfRayDir' = Y*(velComponentOfRayDir+v/c)
 ;float lightDistortion = lorentzF*(dot(-nDpos,r.dir)+speedC)
 ;r.dir=mix(r.dir
           ,normalize((r.dir-nDpos*dot(nDpos,r.dir))-lightDistortion*nDpos)
           ,FOVAberrationOfLight)
 ;//Classical Newtown Mechanic instead would be
 ;//r.dir = normalize(r.dir-dpos/SpeedOfLight)
 ;for (r.iter=0;r.iter<maxStepRayMarching;r.iter++
 ){float camDist = length(r.b - objPos[oCam])//es100 error , no array of class allowed
  ;float photonDelay = -photonLatency*camDist/SpeedOfLight
  //takes dilated distance x/Y and find the time in map frame with :
  // v = -dDirDpos (dot product of direction & velocity, because we want to transform from cam frame to map frame)
  // Y = lorentzFactor
  //
  // t' = Y(t-v*(x/Y)/c²)
  // t' = Y(0-v*(x/Y)/c²)
  // t' = Y(v*x/Y)/c²
  // t' = vx/c²
  ;float relativeInstantEvents = SimultaneousEvents*dDirDpos*camDist/(SpeedOfLight*SpeedOfLight)
  ;r.time = startTime
  ;r.time += mix(relativeInstantEvents,photonDelay,photonLatency)
  ;SetTime(r.time)
  ;r.dist = map(r.b,-1)
  ;//Gravitational lens
  ;vec3 blackHoleDirection = (objPos[oBlackHole]-r.b)//es100 error , no array of class allowed
  ;r.dir+=(1./RayPrecision)*r.dist*normalize(blackHoleDirection)*BlackHoleMassFactor/(length(blackHoleDirection)*SpeedOfLight*SpeedOfLight)
  ;r.dir = normalize(r.dir)
  ;if(abs(r.dist)<rayEps)break
  ;r.b+= (1./RayPrecision)*(r.dist)*(r.dir);}
 ;//r.b = origin + r.dir*min(length(r.b-origin),maxDist)
 ;r.surfaceNorm = GetNormal(r.b).xyz;}

// Function 250
void marchRay(inout Ray ray, inout vec3 colour) {
    bool inside = false; // are we inside or outside the glass object
    vec3 impact = vec3(1); // This decreases each time the ray passes through glass, darkening colours

    vec3 startpoint = ray.origin;
    
#ifdef DEBUG   
vec3 debugColour = vec3(1, 0, 0);
#endif
    
    SDResult result;
    vec3 n;
    vec3 glassStartPos;
    
    //float glow = 0.0;
    
    for (int i=0; i<kMAXITERS; i++) {
        // Get distance to nearest surface
        result = sceneDist(ray);
        
        //glow += result.material == kGLOWMATERIAL ? 
        //    pow(max(0.0, (80.0 - result.d) * 0.0125), 4.0) * result.d * 0.01
        //    : 0.0;
        
        // Step half that distance along ray (helps reduce artefacts)
        float stepDistance = (inside ? abs(result.d) : result.d) * 0.25;
        ray.origin += ray.dir * stepDistance;
        //if (length(ray.origin) > 40.0) { break; }
        
        if (stepDistance < eps) {
            // colision
            // normal
            // Get the normal, then clamp the intersection to the surface
    		n = normal(ray);
            //clampToSurface(ray, stepDistance, n);
#ifdef DEBUG
//debugColour = n;
//break;
#endif
            
            if ( result.material == kFLOORMATERIAL ) {
                // ray hit floor
                
                // Add some noise to the normal, since this is pretending to be grit...
                vec3 randomNoise = texrand(ray.origin.xz * 0.4, 0.0);
                randomNoise.xz = randomNoise.xz * 2. - 1.;
                n = mix(n, normalize(vec3(randomNoise.x, 1, randomNoise.y)), randomNoise.z * 0.3);
                
                // Colour is just grey with crappy fake lighting...
                float o = occlusion(ray, n);
                colour += vec3(1) * o * impact;
                impact *= 0.;
                break;
            }
            
            if ( result.material == kMATTEMATERIAL ) {
                // ray hit floor
                
                // Add some noise to the normal, since this is pretending to be grit...
                //vec3 randomNoise = texrand(n.xz * 0.5 + 0.5, 0.0);
                //randomNoise.xz = randomNoise.xz * 2. - 1.;
               // n = mix(n, normalize(vec3(randomNoise.x, 1, randomNoise.y)), randomNoise.z * 0.1);
                
                // Colour is just grey with crappy fake lighting...
                float o = occlusion(ray, n);
                o = pow(o, 2.0);
               // o = 1.0;
                vec3 tex = surfaceColour(ray, n);
                //tex = mod(ray.origin, 1.0);
                //tex = vec3(1);
                vec3 l = texture(iChannel3, n).rgb * 0.5 + 0.5;
                colour += tex * l * o * impact;
                impact *= 0.;
                break;
            }
            
            if (result.material == kGLOWMATERIAL) {
             	colour = mix(colour, kGLOWCOLOUR, impact);
                impact *= 0.;
                break;
            }
            
            // check what material it is...
            
            if (result.material == kMIRRORMATERIAL) {
                
                // handle interior glass / other intersecion
                if (inside) {
                     float glassTravelDist =  min(distance(glassStartPos, ray.origin) / 16.0, 1.);
    				glassStartPos = ray.origin;
                    // mix in the colour
                	impact *= mix(kGLASSCOLOUR, kGLASSCOLOUR * 0.1, glassTravelDist);
                    
                }
                
                // it's a mirror, reflect the ray
                ray.dir = reflect(ray.dir, n);
                    
                // Step 2x epsilon into object along normal to ensure we're beyond the surface
                // (prevents multiple intersections with same surface)
                ray.origin += n * eps * 4.0;
                
                // Mix in the mirror colour
                colour += highlight(ray, n);
                impact *= kMIRRORCOLOUR;
                float o = occlusion(ray, n);
                impact *= o;
#ifdef DEBUG
debugColour = vec3(o);
break;
#endif
                
            } else {
                // glass material
            
                if (inside) {
                	// refract glass -> air
                	ray.dir = refract(ray.dir, -n, 1.0/kREFRACT);
                    
                    // Find out how much to tint (how far through the glass did we go?)
                    float glassTravelDist =  min(distance(glassStartPos, ray.origin) / 16.0, 1.);
    
                    // mix in the colour
                	impact *= mix(kGLASSCOLOUR, kGLASSCOLOUR * 0.1, glassTravelDist);
                    
#ifdef DEBUG
debugValue += glassTravelDist / 2.0;
#endif
      
                
              	} else {
               		// refract air -> glass
                	glassStartPos = ray.origin;
                    
              	  	// Mix the reflection in, according to the fresnel term
                	float fresnel = fresnelTerm(ray, n, 1.0);
                    fresnel = fresnel;
    				/*
                    colour = mix(
                    	colour, 
                    	texture(iChannel1, reflect(ray.dir, n)), 
                    	vec4(fresnel) * impact);
*/
                    colour = mix(
                        colour,
                        backgroundColour(ray, 0.0),
                        vec3(fresnel) * impact);
                	colour += n.x * 0.1;//highlight(ray, n);
                    impact *= 1.0 - fresnel;
    			
                	// refract the ray
            		ray.dir = refract(ray.dir, n, kREFRACT);
                    
#ifdef DEBUG
//debugValue += 0.5;
#endif
                }
            
            	// Step 2x epsilon into object along normal to ensure we're beyond the surface
                ray.origin += (inside ? n : -n) * eps * 2.0;
                
                // Flip in/out status
                inside = !inside;
            }
        }
        
        // increase epsilon
        eps += divergence * stepDistance;
    }
    
    // So far we've traced the ray and accumulated reflections, now we need to add the background.
   // colour += texture(iChannel0, ray.dir) * impact;
    ray.origin = startpoint;
    colour.rgb += backgroundColour(ray, 0.0) * impact; // + glow * kGLOWCOLOUR;
    
#ifdef DEBUG
//debugColour.rgb = ray.dir;
//debugColour = vec3(float(debugStep)/2.0);
colour = debugColour;
#endif
}

// Function 251
float some_step(float t) {
    return pow(t, 4.0);
}

// Function 252
vec4 march(in vec3 ro, in vec3 rd)
{
	float precis = 0.001;
    float h=precis*2.0;
    vec2 d = vec2(0.,10000.);
    float md = 1.;
    float id = 0.;;
    bool stp = false;
    for( int i=0; i<ITR; i++ )
    {
        if( abs(h)<precis || d.x>=FAR ) break;
        d.x += h;
	   	vec2 res = map(ro+rd*d.x);
        if (!stp) 
        {
            md = min(md,res.x);
            if (h < EDGE_SIZE && h < res.x && i>0)
            {
                stp = true;
                d.y = d.x;
            }
        }
        h = res.x;
        id = res.y;
    }
    
    if (stp) md = smoothstep(EDGE_SIZE-SMOOTH, EDGE_SIZE+0.01, md);
    else md = 1.;
	return vec4(d, md, id);
}

// Function 253
float raymarchAO(in vec3 ro, in vec3 rd, float tmin) {
    float ao = 0.0;
    for (float i = 0.0; i < 5.0; i++) {
        float t = tmin + pow(i / 5.0, 2.0);
        vec3 p = ro + rd * t;
        float d = p.y - fBm(p.xz);
        ao += max(0.0, t - 0.5 * d - 0.05);
    }
    return 1.0 - 0.4 * ao;
}

// Function 254
float RayMarch(vec3 ro,vec3 rd) { float cd = 0.;float pd = MID*1.1;for(int s = 0; s < MS && cd < MAD && pd > MID;s++){pd = getDist(ro+rd*cd);cd += pd;}return cd;}

// Function 255
vec3 raymarche( in vec3 ro, in vec3 rd, in vec2 nfplane )
{
	vec3 p = ro+rd*nfplane.x;
	float t = 0.;
	for(int i=0; i<1256; i++)
	{
        float d = map(p);
        t += d;
        p += rd*d;
		if( t > nfplane.y )
            break;
            
	}
	
	return p;
}

// Function 256
float raymarching(
  in vec3 prp,
  in vec3 scp,
  in int maxite,
  in float precis,
  in float startf,
  in float maxd,
  out float objid)
{ 
  const vec3 e=vec3(0.1,0,0.0);
  vec2 s=vec2(startf,0.0);
  vec3 c,p,n;
  float f=startf;
  for(int i=0;i<256;i++){
    if (abs(s.x)<precis||f>maxd||i>maxite) break;
    f+=s.x;
    p=prp+scp*f;
    s=obj(p);
    objid=s.y;
  }
  if (f>maxd) objid=-1.0;
  return f;
}

// Function 257
float raymarch(vec3 ray_origin, vec3 ray_direction) {
  float d = 0.0;

  for (float i = 0.; i < 70.; i++) {
    vec3 new_point = ray_origin + ray_direction*d;
    float s = get_distance(new_point);
    if (s < 0.001) return d;
    d += s;
    if (d > 70.) return 300.;
  }
  return 300.;
}

// Function 258
vec3 StepValue3(float a, float b, vec3 ra, vec3 rb)
{
    return mix(ra, rb, step(a, b));
}

// Function 259
float ray_marching(vec3 origin, vec3 dir, float start, float end) {
	float depth = start;
	for (int i = 0; i < max_iterations; i++) {
		float dist = dist_field(origin + dir * depth);
		if (dist < stop_threshold) {
			return depth;
		}
		depth += dist;
		if (depth >= end) {
			return end;
		}
	}
	return end;
}

// Function 260
bool rayMarching(in vec3 origin, in vec3 ray, out vec3 m, out vec3 normal) {
    float testingHeight = MAX_ELEVATION;

    if(!intersectSphere(origin, ray, Sphere(sphere_center, sphere_radius + testingHeight), m)) {
        return false;
    }
    
    normal = normalize(m);

    // Start slow scan
    for(int i = 0; i<400; i++) {
        float r = length(m - sphere_center) - sphere_radius;
        
        // m exits sphere
        if(r > MAX_ELEVATION+STEP) {
            return false;
        }
        
        float dist = r - getHeight(m);
        
        // side collision
        if(dist < -STEP) {
            normal = computeNormalAndSnapPoint(m);
            return true;
        }
        if(dist < STEP) {
            normal = computeNormal(m);
            m += ray * dist;
            return true;
        }
        m += ray * STEP;    
    }
    
    return false;
}

// Function 261
float RayMarch(vec3 startPos, vec3 dir) {
	float depth = 0.0;
    for (int i = 0; i < 64; i++) {
        vec3 pos = startPos + dir * depth;
        float dist = SceneDistance(pos);
        if (dist < 0.0001) {
        	return depth;
        }
        depth += 0.6 * dist;
        if (depth >= MAX_DEPTH) {
            return MAX_DEPTH;
        }
    }
    return MAX_DEPTH;
}

// Function 262
float rayMarch(Ray r, Sphere sphere) {
	float d0 = 0.;
    
    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = r.origin + d0*r.direction;
        float dS = getDist(p, sphere);
        d0 += dS;
        
        if (dS < SURFACE_DIST || d0 > MAX_DIST)
            break;
    }
    
    return d0;
}

// Function 263
float RayMarch(vec3 rayOrigin, vec3 rayDirection, out Hit result)
{
    const int MAX_STEPS = 200;
    float totalDistance = 0.0;
    for (int i = 0; i < MAX_STEPS; ++i)
    {
        vec3 rayPoint = rayOrigin + rayDirection * totalDistance;
        vec2 mapResult = MapScene(rayPoint);
        if (mapResult.x < MIN_DIST) 
        {
            result.point = rayPoint;
            result.normal = GetNormal(rayPoint);
            result.type = mapResult.y;
            return totalDistance;
        }
        totalDistance += mapResult.x;
        if (totalDistance > MAX_DIST) break;
    }
    result.type = TYPE_NONE;
    return MAX_DIST;
}

// Function 264
float vonronoiStep(float i, float a, float b, vec2 x) {
    float d = 0.2*(b-a);
	return 1.0-i+(smoothstep(a-d, b+d, vonronoi(x))*(i));
}

// Function 265
void rayMarch(vec3 worldVector, out float depth, out vec3 rayPosition){
    const float rMarchSteps = 1. / float(marchSteps);
    
    const float startDepth = 0.;
    const float endDepth = farPlane;
    
    vec3 increment = (endDepth - startDepth) * worldVector * rMarchSteps;
    // increment is 10 * pixelPos * 0.001953125
    rayPosition = increment + startDepth * worldVector;    
    // rayposition is increment (-0.0195325 <> 0.0195325) + 0 * -1<>1
        
    float os = 0.0;
    
    depth = 0.0;
    
    for (int i = 0; i < marchSteps; ++i, rayPosition += increment){
        float rayLength = length(rayPosition);
        calculateShape(rayPosition, os);
        
        depth = rayLength;
        
        if (calculateRayHit(os, rayLength)) break;
    }
}

// Function 266
vec4 rayMarch(vec3 startPoint, vec3 direction, int iterations, float maxStepDist)
{
 	vec3 point = startPoint;
    direction = normalize(direction);
    float distSum = 0.0;
    float shadowData = 1.0;
    float dist = 10.0;
    
    int i;
    for (i = 0; i < iterations && distSum < MAX_VIEW_DISTANCE && abs(dist) > EPSILON; i++)
    {
     	dist = terrainDist(point, direction.xy);
        dist = min(dist, maxStepDist) * 0.4;
        distSum += dist;
        point += direction * dist;
    }
    
    return vec4(point.xyz, distSum);
}

// Function 267
float march(in vec3 ro, in vec3 rd) {
  const float maxd = 5.0;
  const float precis = 0.001;
  float t = 0.0;
  float res = -1.0;
  for (int i = 0; i < 200; i++) {
    //assert(i < 30);
    if (t > maxd) return -1.0;
    float h = map(ro+rd*t);
    t += h;
    if (h < precis) return t;
  }
  return t;
}

// Function 268
vec2 raymarch(in vec3 ro, in vec3 rd)
{
    float t = MIN_TRACE_DIST;
    vec2 h;
    for(int i=0; i<MAX_TRACE_STEPS; i++)
    {
        h = map(ro + t * rd);
        if (h.x < PRECISION * (t*0.125+1.))
            return vec2(t, h.y);

        if (t > MAX_TRACE_DIST)
            break;

        t += h.x;
    }
    return vec2(-1.0);
}

// Function 269
float gainStep(float x, float e) {
    return gainStep(0., 1., x, e);
}

// Function 270
vec4 StepJFA (in vec2 fragCoord, in float level)
{
    level = clamp(level, 0.0, c_maxSteps);
    float stepwidth = floor(exp2(c_maxSteps - level)+0.5);
    
    float bestDistance = 9999.0;
    vec2 bestCoord = vec2(0.0);
    vec3 bestColor = vec3(0.0);
    
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 sampleCoord = fragCoord + vec2(x,y) * stepwidth;
            
            vec4 data = texture( iChannel0, sampleCoord / iChannelResolution[0].xy);
            vec2 seedCoord;
            vec3 seedColor;
            DecodeData(data, seedCoord, seedColor);
            float dist = length(seedCoord - fragCoord);
            if ((seedCoord.x != 0.0 || seedCoord.y != 0.0) && dist < bestDistance)
            {
                bestDistance = dist;
                bestCoord = seedCoord;
                bestColor = seedColor;
            }
        }
    }
    
    return EncodeData(bestCoord, bestColor);
}

// Function 271
float iqStep(float i, float a, float b, float u, float v, vec2 x) {
    float d = 0.2*(b-a);
	return 1.0-i+(smoothstep(a-d, b+d, iqnoise(x, u, v))*(i));
}

// Function 272
vec2 rayMarch(vec3 ro, vec3 rd)
{
    float t = 0.0;
    vec3 p;
    vec2 obj;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        p = ro + t * rd;
       	
        obj = map(p);
        
        if (obj.x < SURFACE_DIST || t > MAX_DIST) break;
        
        t += obj.x;
    }
    
    obj.x = t;
    return obj;
}

// Function 273
float marchToLight(vec3 p, vec3 sunDir, float sunDot, float scatterHeight)
{
    float lightRayStepSize = 11.;
	vec3 lightRayDir = sunDir * lightRayStepSize;
    vec3 lightRayDist = lightRayDir * .5;
    float coneSpread = length(lightRayDir);
    float totalDensity = 0.;
    for(int i = 0; i < CLOUD_LIGHT_STEPS; ++i)
    {
        // cone sampling as explained in GPU Pro 7 article
     	vec3 cp = p + lightRayDist + coneSpread * noiseKernel[i] * float(i);
        float y = cloudHeightFract(length(p));
        if (y > .95 || totalDensity > .95) break; // early exit
        totalDensity += getCloudDensity(cp, y, false) * lightRayStepSize;
        lightRayDist += lightRayDir;
    }
    
    return 32. * exp(-totalDensity * mix(CLOUD_ABSORPTION_BOTTOM,
				CLOUD_ABSORPTION_TOP, scatterHeight)) * (1. - exp(-totalDensity * 2.));
}

// Function 274
MarchRes marchRay(vec3 pos, vec3 dir, float speed)
{
 	MarchRes res;
    Object o;
    
    res.totalDist = 0.001;
    res.glowAmt = 1.0;
    res.minDist = 1000.0;

    for(int x=0; x<150; x++)
    {
 		res.curRay = pos + (dir*res.totalDist);
        
        o = map(res.curRay);
        
        if(abs(o.dist) < 0.00001)
        {
            res.obj = o;
            break;
        }
        else if(res.totalDist >= VIEW_DIST) break;
           
        
        if(o.dist < res.minDist) res.minDist = o.dist;
        res.totalDist += o.dist*speed;
    }
    
    if(res.totalDist < VIEW_DIST)
    {
        o.normal = calcNormal(res.curRay, o.normEps);
        
        res.obj = o;
    }
    	
    
    return res;
}

// Function 275
float March(vec3 eye, vec3 dir, float start, float end) {
    
	float depth = start;
   
    int i = 0;
    do {

     	float dist = SceneSDF(eye + depth * dir);
        
        if(dist < EPSILON) return depth;
        
        depth += dist;
        
        if(depth >= end) return end;

    }
    while(i++ < MAX_STEPS);

    return end;
}

// Function 276
bool raymarchW(out vec3 pos)
{
	float t = .0;
	float d =.0;
	int i = 0;
	do
	{
		d = map(ro + rd * t);
		t += d;
	} while (i++ < maxSteps && t < drawDist && d > epSI);

	pos = ro + rd * t;
	return (t < drawDist);
}

// Function 277
vec4 raymarch(vec2 resolution, vec2 uv, vec4 start_data, mat4 camera_transform) {
    int steps = 16;
    
    // Convert to range (-1, 1) and correct aspect ratio
    vec2 screen_coords = (uv - 0.5) * 2.0;
    screen_coords.x *= resolution.x / resolution.y;
    
    
    vec3 ray_start_position = camera_transform[3].xyz;
    
    vec3 ray_direction = normalize(vec3(screen_coords * LENS, 1.0));
    ray_direction = (camera_transform * vec4(ray_direction, 0.0)).xyz;
    
    
    float dist = start_data.a * 0.9;
    vec3 sample_point = ray_start_position + dist * ray_direction;
    
    vec4 results = sample_world(sample_point);
    
    float tolerance = 0.0;
    
    for (int i=0; i<steps; i += 1) {
        dist += results.a;
        sample_point += ray_direction * results.a;
        results = sample_world(sample_point);
        
        // TODO: Derive from resolution, camera lens and distance
    	tolerance = LENS / resolution.x * dist;
        
        if (results.a < tolerance) {
        	break; 
        }
    }
    
    vec4 data = vec4(
        vec3(results) * 0.5 + 0.5,
        dist
    );
    
    return data;
}

// Function 278
Hit raymarch(Ray ray) {
 
    vec3 p = ray.ori;
    float t = 0.;
    int id = -1;
    
    for(int i = 0; i < MAX_ITERATIONS; i++) {
     
        Dist d = distToScene(p);
        p += ray.ori + (ray.dir * d.dist);
        
        if(d.dist <= MIN_DISTANCE) {
         
            t = d.dist;
            id = d.id;
            
            break;
            
        }
        
    }
    
    return Hit(p,Dist(t,id));
    
}

// Function 279
float raymarchwater(vec3 camera, vec3 start, vec3 end, float depth){
    vec3 pos = start;
    float h = 0.0;
    float hupper = depth;
    float hlower = 0.0;
    vec2 zer = vec2(0.0);
    vec3 dir = normalize(end - start);
    float eps = 0.01;
    for(int i=0;i<318;i++){
        h = getwaves(pos.xz * 0.1, ITERATIONS_RAYMARCH/int(1.0+length(pos/100.0))) * depth - depth;
        float dist_pos = distance(pos, camera);
        if(h + eps*dist_pos > pos.y) {
            return dist_pos;
        }
        pos += dir * (pos.y - h);
        eps *= 1.01;
    }
    return -1.0;
}

// Function 280
void RayMarchScene(in vec3 startingRayPos, in vec3 rayDir, inout SRayHitInfo oldHitInfo)
{
    SMaterial dummyMaterial = SMaterial(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), 0.0f, vec3(0.0f, 0.0f, 0.0f));
    
    float rayDistance = c_minimumRayHitTime;
    float lastRayDistance = c_minimumRayHitTime;
    
    float lastHitInfoDist = 0.0f;
    
    SRayHitInfo newHitInfo = oldHitInfo;
    newHitInfo.hitAnObject = false;
    
    for (int stepIndex = 0; stepIndex < c_numSteps; ++stepIndex)
    {
        vec3 rayPos = startingRayPos + rayDistance * rayDir;
        
        newHitInfo = TestSceneMarch(rayPos);
        
        // these two lines are so that the material code goes away when the test functions are inlined
        newHitInfo.normal = vec3(0.0f, 0.0f, 0.0f);
        newHitInfo.material = dummyMaterial;
        
        newHitInfo.hitAnObject = newHitInfo.dist < 0.0f;
        if (newHitInfo.hitAnObject)
            break;
        
        lastRayDistance = rayDistance;
        rayDistance += max(newHitInfo.dist, c_minStepDistance);

        lastHitInfoDist = newHitInfo.dist;
        
        if (rayDistance > oldHitInfo.dist)
            break;
    }
    
    if (newHitInfo.hitAnObject)
    {
		float refinedHitPercent = lastHitInfoDist / (lastHitInfoDist - newHitInfo.dist);
        newHitInfo.dist = mix(lastRayDistance, rayDistance, refinedHitPercent);
        
        if (newHitInfo.dist < oldHitInfo.dist)
            oldHitInfo = newHitInfo;
    }
}

// Function 281
vec2	march(vec3 pos, vec3 dir)
{
    vec2	dist = vec2(0.0);
    vec3	p = vec3(0.0);
    vec2	s = vec2(0.0);

    vec3	dirr;
    for (int i = 0; i < I_MAX; ++i)
    {
        dirr = dir;
    	rotate(dirr.zx, .05*dist.y*sin(t*1.5));
        #ifdef	SPIRAL
        rotate(dirr.yx, .8*dist.y*sin(t*.5));
        #endif
    	p = pos + dirr * dist.y;
        dist.x = scene(p);
        dist.y += dist.x;
        if (dist.x < E || dist.y > 20.)
        {
            p=ss;
            g = p.y;
	        g += (step(sin(5.*p.x), .5) 
             + step(sin(20.*p.x), .5) );
           break;
        }
        s.x++;
    }
    s.y = dist.y;
    return (s);
}

// Function 282
vec4 raymarch(vec3 p, vec3 d)
{
    float S = 0.0;
    float T = S;
    vec3 D = normalize(d);
    vec3 P = p+D*S;
    for(int i = 0;i<240;i++)
    {
        S = model(P);
        T += S;
        P += D*S;
        if ((T>MAX) || (S<PRE)) {break;}
    }
    return vec4(P,min(T/MAX,1.0));
}

// Function 283
vec3 march(vec3 origin, vec3 dir)
{
    const int ITER = 50;
    vec3 p = origin;
    vec3 d = normalize(dir);
    for (int i = 0; i < ITER; i++) {
        p = p + df(p) * d;
    }
    return p;
}

// Function 284
RMResult raymarch(vec3 ro, vec3 rd, out float t)
{
	t = 0.;
    vec3 p = ro + t * rd;
    RMResult s = map(p);
    float isInside = sign(s.dist);
    for(int i = 0; i < I_MAX; i++)
    {
        float inc = isInside * s.dist;
        if (t + inc < FAR && abs(s.dist) > EPS) 
        {
			t += inc;
	        p = ro + t * rd;
            s = map(p);
        }
        else
        {
            if (t + inc > FAR)
            {
               s.id = -1.;
            }
            break;
        }
    }
    return s;
}

// Function 285
float rayMarch(vec3 rayDir, vec3 cameraOrigin)
{
    const int MAX_ITER = 50;
	const float MAX_DIST = 30.0;
    
    float totalDist = 0.0;
    float totalDist2 = 0.0;
	vec3 pos = cameraOrigin;
	float dist = EPSILON;
    vec3 col = vec3(0.0);
    float glow = 0.0;
    
    for(int j = 0; j < MAX_ITER; j++)
	{
		dist = distfunc(pos);
		totalDist = totalDist + dist;
		pos += dist*rayDir;
        
        if(dist < EPSILON || totalDist > MAX_DIST)
		{
			break;
		}
	}
    
    return totalDist  ;
}

// Function 286
float march(in vec3 ro, in vec3 rd) {
  float maxd = length(ro) + 1.0;
  float precis = 0.001;
  float t = 0.0;
  float res = -1.0;
  for (int i = 0; i < 100; i++) {
    //assert(i < 20);
    float h = map(ro+rd*t);
    if (abs(h) < precis) return t;
    t += h;
    if (t < 0.0 || t > maxd) break;
  }
  return -1.0;
}

// Function 287
float march (in vec3 ro, in vec3 rd, out int iter) {
    float t = .0;
    float d = .0;
    iter = 0;
    for (int i = 0; i < 64; ++i) {
        iter++;
        vec3 p = ro + d * rd;
        t = map (p);
        if (abs (t) < .0001 * (1. + .125*t)) break;
        d += t*.7;
    }

    return d;
}

// Function 288
vec3 rayMarch(vec3 origin, vec3 dir) {
    vec3 p = origin;
    while(true) {
        float d = sceneSDE(p);
        if (d < 0.01) { return p; }
        if (length(p - origin) > 100.) { return p; }
        p += dir * d;
    }
}

// Function 289
float smootheststep(float edge0, float edge1, float x)
{
    x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0) * 3.14159265;
    return 0.5 - (cos(x) * 0.5);
}

// Function 290
float march(vec3 ro, vec3 rd, float rmPrec, float maxd, float mapPrec)
{
    float s = rmPrec;
    float d = 0.;
    for(int i=0;i<250;i++)
    {      
        if (s<rmPrec||s>maxd) break;
        s = map(ro+rd*d).x*mapPrec;
        d += s;
    }
    return d;
}

// Function 291
float linearstep(float begin, float end, float t) {  return clamp((t - begin) / (end - begin), 0.0, 1.0);  }

// Function 292
float somestep(float t)
{
    return pow(t, 4.0);
}

// Function 293
vec2 linstep( vec2 x0, vec2 x1, vec2 a )
{
    vec2 o;
    o = clamp((a - x0)/(x1 - x0), 0.0, 1.0);
    return o;
}

// Function 294
float RayMarch(vec3 ro, vec3 rd) 
{
	
    // The extra distance might force a near-plane hit, so
    // it's set back to zero.
    float dO = 0.; 
    //Determines size of shadow
    for(int i=0; i<MAX_STEPS; i++) 
    {
    	vec3 p = ro + rd*dO;
        float dS = GetDist(p);
        
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
        dO += dS;
        
    }
    
    return dO;
}

// Function 295
void MarchLight(inout Ray r, float startTime, float maxDist
){float totalDist = 0.0
 ;vec3 origin = r.b
 ;for (r.iter=0;r.iter<maxStepRayMarching;r.iter++
 ){r.time = startTime
  ;SetTime(r.time)
  ;r.dist = map(r.b,-1)
  ;totalDist += r.dist
  ;r.b+= r.dir*(r.dist)
  ;if(abs(r.dist)<rayEps||totalDist>maxDist)break;}}

// Function 296
vec2 RayMarch(vec3 ro, vec3 rd, float side, int stepnum) {
    vec2 dO = vec2(0.0);
    
    float lastDistEval = 1e10; 
    float dist;
    for(int i=0; i<stepnum; i++) {
        vec3 p = ro + rd*dO.x;
        vec2 dS = GetDist(p);
        dist = dS.x;
        dO.x += dS.x*side;
        dO.y = dS.y;
        
        if(stepnum == MAX_STEPS){
            if (lastDistEval < EDGE_WIDTH && dist > lastDistEval + 0.0005) {
                edge = 1.0;
            }
            if (dist < lastDistEval) lastDistEval = dist;
        }
        
        if(dO.x>MAX_DIST || abs(dS.x)<SURF_DIST) break;
    }
    
    return dO;
}

// Function 297
float expstep(float x, float k) {
  return exp(((k * x) - k));
}

// Function 298
vec2 RayMarch(vec3 ro, vec3 rd){
    // distance from origin
    vec2 dO=vec2(0.,0.);
    // march until max steps is achieved or object hit
    for(int i=0; i <MAX_STEPS; i++){
        // current point being evaluated
        vec3 p = ro + dO.x*rd;
        
        // get distance to seam
        vec2 ds = GetDist(p);
        //move origin to new point
        dO+=ds.x*.5;
        if(ds.x < SURFACE_DIST){
            dO.y = ds.y;
            break;
        }
        else if( dO.x > MAX_DIST){
            dO.y= -1.;
            break;
        }
    }
    return dO;
}

// Function 299
vec4 March(vec3 rayOrigin, vec3 rayStep)
{
	vec3 position = rayOrigin;
	float distance;
	float displacement;
	for(int step = MarchSteps; step >=0  ; --step)
	{
		displacement = RenderScene(position, distance);
		if(distance < 0.05) break;
		position += rayStep * distance;
	}
	return mix(Shade(displacement), Background, float(distance >= 0.5));
}

// Function 300
vec3 ShadeSteps(int n)
{
   float t=float(n)/(float(MAX_STEPS-1));
   return 0.5+mix(vec3(0.05,0.05,0.5),vec3(0.65,0.39,0.65),t);
}

// Function 301
vec2 rayMarch(vec3 ro, vec3 rd)
{
    vec2 dist 	= vec2(0.0);
    vec2 d 		= vec2(0.0);
    vec3 pos 	= vec3(0.0);
    
    for(int i=0; i<MAX_STEPS; i++)
    {
        pos = ro + dist.x * rd;
        d = getDist(pos);
		dist.x += d.x;
        dist.y = d.y;
		if (dist.x<MIN_DIST)
            break;
        
        if (dist.x>MAX_DIST)
            return vec2(0.0, 0.0);
    }
    
    return dist;
}

// Function 302
Hit raymarch(Ray ray) {
 
    vec3 p = ray.ori;
    int id = -1;
    
    for(int i = 0; i < MAX_ITERATIONS; i++) {
     
        Dst scn = dstScene(p);
        p += ray.dir * scn.dst * .75;
        
        if(scn.dst < MIN_DISTANCE) {
         
            id = scn.id;
            break;
            
        }
        
    }
    
    return Hit(p,id);
    
}

// Function 303
vec4 put_text_step(vec4 col, vec2 uv, vec2 pos, float scale)
{
	float unit = asp * scale * 0.1;
    float h = 0.;
    vec2 sc = vec2(unit, unit*0.8);
    
    // S
    h = max(h, word_map(uv, pos, 83, sc));
    // t
    h = max(h, word_map(uv, pos+vec2(unit*0.35, 0.), 116, sc));
    // e
    h = max(h, word_map(uv, pos+vec2(unit*0.7, 0.), 101, sc));
    // p
    h = max(h, word_map(uv, pos+vec2(unit*1.05, 0.), 112, sc));
    
    col = mix(col, vec4(1.-vec3(h), 1.), h);
    
    return col;
}

// Function 304
float march (in vec3 ro, in vec3 rd) {
    float t = .0;
    float d = .0;
    for (int i = 0; i < MAX_STEPS; ++i) {
        vec3 p = ro + d * rd;
        t = map (p);
        if (t < EPSILON) break;
        d += t*STEP_SIZE;
    }

    return d;
}

// Function 305
MarchResult raymarchScene(in vec3 ro, in vec3 rd, out vec4 scattering, in vec2 fragCoord)
{
    float tmin = 0.0;
    float tmax = 50.0;
    float t = tmin;
    float prevt = t;
    float d = 0.0, eps = 0.0;
    float prevd = d;
    float density = 0.0;
    float transmittance = 1.0;
    float stp = 0.0;
    vec3 inscatteredLight = vec3(0.0);
    vec3 inscatteredLightFromEquiAngularSampling = vec3(0.0);
    MarchResult result;
    int stepsEquiAngularSampling = 0;
    
    for (int i = 0; i < 300; ++i)
    {
        eps = t * 0.001;
        vec3 p = ro + t * rd;    
          
        density = (densityVoumetricFog(p * 253.0));
        integrateVolumetricFog(p, rd, density, stp, transmittance, inscatteredLight, fragCoord);
        if (stepsEquiAngularSampling < 32)
        {
            float u = rand(rand(fragCoord.x * iResolution.y + fragCoord.y + d * 1.5) + iTime + 1234.32598);
            float v = rand(rand(fragCoord.y * iResolution.x + fragCoord.x + d * 2.5) + iTime * 2.0 + 6234.32598);
            vec3 lightPos;
            vec3 lightCol;
            sampleAreaLight(vec2(u, v), lightPos, lightCol);

            float x0 = rand(rand(fragCoord.y * iResolution.y + fragCoord.x + d * 3.5 + iTime) + 236526.436346);
            // --- equi-angular sampling
            float DT = dot(lightPos - ro, rd);
            float D = length(ro + DT * rd - lightPos);
            float tha = atan(0.0 - DT, D);
            float thb = atan(length(tmax - ro) - DT, D);
            float tsampled = D * tan(mix(tha, thb, x0));
            float pdf = D / ((thb - tha) * (D * D + tsampled * tsampled));
            float x = DT + tsampled;
            vec3 sampledPos = ro + x * rd;
            
        	float densityFromSampledPos = (densityVoumetricFog(sampledPos * 253.0));
        	integrateVolumetricFogFromSampledPosition(sampledPos, rd, densityFromSampledPos, x, inscatteredLightFromEquiAngularSampling, lightPos, lightCol, pdf); 
        	stepsEquiAngularSampling++;        
        }
        
        SceneData res = intersectTerrain(p.xz);
        float h = res.sdf;
        d = p.y - h;
        
        vec3 prot = getRotation() * (p - AreaLightPosition);
        SceneData lightPlane = intersectBox(prot, AreaLightSize);
        SceneData terrain;
        terrain.sdf = d;
        terrain.materialID = res.materialID;
        SceneData scene = unite(terrain, lightPlane);
        d = scene.sdf;
        result.materialID = scene.materialID;
        if (d < eps)
            break;        
        
        stp = d * 0.32;
        prevt = t;
        prevd = d;
        t += stp;
        if (t > tmax)
            break;
    }
    if (t > tmax)
        t = -1.0;
    else
        t = mix(prevt, t, d/prevd);
    
    inscatteredLightFromEquiAngularSampling *= 1.0 / float(stepsEquiAngularSampling); 
    scattering = vec4(inscatteredLight + inscatteredLightFromEquiAngularSampling, transmittance);
    result.t = t;
    return result;
}

// Function 306
float shadow_march(vec4 pos, vec4 dir, float distance2light, float light_angle, inout object co)
{
	float light_visibility = 1.;
	float ph = 1e5;
    float td = dir.w;
	pos.w = map(pos.xyz, co);
	for (int i = min(0, iFrame); i < 32; i++) 
    {
		dir.w += pos.w;
		pos.xyz += pos.w*dir.xyz;
		pos.w = map(pos.xyz, co);
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

// Function 307
RayHit SkyMarch(in vec3 origin, in vec3 direction)
{
  RayHit result;
  result.treeHit = false;
  float maxDist = 10.10, precis = 0.004;
  float t = 0.0, dist = 0.0, distStep = 0.75;
  vec3 rayPos =vec3(0);

  for ( int i=0; i<64; i++ )
  {
    rayPos =origin+direction*t;

    dist = max(ellipsoid(  rayPos-vec3(0.0, 0., -2.0), vec3( 10.0, 0.5, 10.0)), 
      -ellipsoid( rayPos-vec3(0.0, 0., -2.0), vec3(9.0, 0.49, 9.0)));

    if (abs(dist)<precis || t>maxDist )
    {        
      result.hit=true;
      result.depth = t; 
      result.dist = dist;                              
      result.hitPos = origin+((direction*t)*0.99);   
      result.steps = float(i);
      result.id=1.0;
      break;
    }
    t += dist*distStep;
  }
  if (t>maxDist) {
    result.hit=false;
  }

  return result;
}

// Function 308
vec4 textureSmootherstep(sampler2D tex, vec2 uv, vec2 res)
{
	uv = uv*res + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*fuv*(fuv*(fuv*6.0-15.0)+10.0);
	uv = (uv - 0.5)/res;
	return texture( tex, uv );
}

// Function 309
RayHit TerrainMarch(in vec3 origin, in vec3 direction)
{
  RayHit result;
  result.treeHit = false;
  float maxDist = 1.0, precis = 0.007;
  float t = 0.0, dist = 0.0, distStep = 0.1;
  vec3 rayPos =vec3(0);

  for ( int i=0; i<maxRaySteps; i++ )
  {
    rayPos =origin+direction*t;
    dist = TerrainDistance( rayPos);

    if (abs(dist)<precis || t>maxDist )
    {        
      result.hit = !(t>maxDist);
      result.depth = t; 
      result.dist = dist;                              
      result.hitPos = origin+((direction*t)*0.99);   
      result.steps = float(i);
      result.unclampedHeight=unclampedHeight;
      result.treeHit = treeHit;
      result.terrainHeight = terrainHeight;
      result.treeHeight = treeHeight;
      break;
    }
    t += dist*distStep;
  }

  return result;
}

// Function 310
void Step (int mId, out vec3 r, out vec3 v)
{
  vec3 rn, vn, dr, f;
  float fOvlap, fDamp, grav, rSep, dt;
  fOvlap = 1000.;
  fDamp = 7.;
  grav = 50.;
    
  vec4 ll0 = Loadv4 (2 * mId);
  float rtot;  
  r = ll0.xyz;
  v = Loadv4 (2 * mId + 1).xyz;
  f = vec3 (0.);
 
  for (int n = 0; n < SPH; n ++) {
    vec4 ll = Loadv4 (2 * n);
    rn = ll.xyz;
    dr = r - rn;
    rSep = length (dr);
    rtot = ll0.w + ll.w;
    if (n != mId && rSep < rtot) f += fOvlap * (rtot / rSep - 1.) * dr;
  }
    
  dr = hbLen - abs(r) - ll0.w;
  f -= step (dr, vec3 (1.)) * fOvlap * sign (r) * (1. / abs (dr) - 1.) * dr +
      vec3 (0., grav, 0.) * QToRMat (qtVu) + fDamp * v;
  dt = .02; //iTimeDelta; //*/0.02;
  v += dt * f;
  r += dt * v;
}

// Function 311
void march( inout vec3 p, vec3 d )
{

    
    float r = distGrid(p+d*EPSILON);// reset r
 	
    if (int(fract(d.x)*50.)== 0 || int(fract(d.y)*50.) == 0)
        flag = 1;
    float rr = 0.;
    ivec3 id = ivec3(0,0,0);
    for(int j = 0; j < 15; j++)  // arbitrary number of gridboxes to search; this
        // would have a max based on number of gridboxes per large cube
    {
    	r = distGrid(p-d*10.*EPSILON);
        if (j == 0) r = distGrid(p+d);
        rr = 0.;
        for(int i = 0; i < 20/*V_STEPS*/; i++)
        {
            if(abs(r) < EPSILON/*|| r > MAX_DEPTH*/)
            { 
                /*ivec3 id = getCellID(p, 2., ivec3(0,0,0));
                if (id.z >-3 && id.z < -2){
                   /* flag = 1;*/ break;
            	//}
        	}
            p += d*r;

            r = distGrid(p);
        }

        ivec3 id = getCellID(p, .5, ivec3(0,0,0));
       // id = ivec3(0,2,1);
       if (id.z >8 && id.z < 13 && id.x >-3 && id.x < 3 && id.y > -3&& id.y <  3)
        gColor += normalize(mod(vec3(float(id.x),float(id.y),float(id.z)),10.))/(float(j+1)*2.);
	

        //if (id == ivec3(0,0,-3))
        //    flag = 1;
        //    break;
    if (id.x >0 && id.x < 2)
    	{
			flag = 0;
            for(int i = 0; i < V_STEPS; i++)
            {
                rr = dist(p+d*EPSILON);
                if(rr < EPSILON || r > MAX_DEPTH)
                    return;
                p += d*rr;

                rr = dist(p);
        	}    
        }}

    
    

	return;
}

// Function 312
float2 March ( in Ray ray ) {
  float dist = 0.0;
  float2 cur;
  for ( int i = 0; i != 128; ++ i ) {
    cur = Map(ray.ori + ray.dir*dist);
    if ( cur.x <= 0.0005 || dist > 256.0 ) break;
    dist += cur.x;
  }
  if ( dist > 256.0 || dist < 0.0 ) return float2(-1.0);
  return float2(dist, cur.y);
}

// Function 313
vec2 raymarchVoxel(vec3 ro, vec3 rd, out vec3 nor, vec2 beats) {
  vec3 pos = floor(ro);
  vec3 ri = 1.0 / rd;
  vec3 rs = sign(rd);
  vec3 dis = (pos - ro + 0.5 + rs * 0.5) * ri;
  
  float res = -1.0;
  vec3 mm = vec3(0.0);
  
  for (int i = 0; i < 38; i++) {
    float k = voxelModel(pos, ro, beats);
    if (k > 0.5) {
      res = k;
      break;
    }
     
    mm = step(dis.xyz, dis.yxy) * step(dis.xyz, dis.zzx);
		dis += mm * rs * ri;
    pos += mm * rs;
  }
  
  if (res < -0.5) {
    return vec2(-1.0);
  }
  
  nor = -mm * rs;
  
  vec3 vpos = pos;
  vec3 mini = (pos-ro + 0.5 - 0.5*vec3(rs))*ri;
  float t = max(mini.x, max(mini.y, mini.z));
  
  return vec2(t, 0.0);
}

// Function 314
Result march (in Ray ray)
{
    Result res = nullResult;

    for (int i = 0; i < MAX_ITER; ++i) {
        res.iter = i;
        float tmp = map (ray.ro + res.dist * ray.rd);
        if (tmp < EPSILON) break;
        res.dist += tmp * STEP_SIZE;
    }

    res.point = ray.ro + res.dist * ray.rd;
    res.normal = normal (res.point);
    //res.id = tmp.id;

    return res;
}

// Function 315
float smootherstep(float edge0, float edge1, float x)
{
    x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    return x*x*x*(x*(x*6.0 - 15.0) + 10.0);
}

// Function 316
vec2 raymarch(vec3 ro, vec3 rd) {
	float tmin = 0.0;
	float tmax = Far;

	float precis = 0.0002;
	float t = tmin;
	float m = MAT_NONE;

	for (int i = 0; i < MaxSteps; i++) {
		vec2 res = abs(map(ro + rd * t));

		if (res.x < precis || t > tmax) {
			break;
		}
		t += res.x * MarchDumping;
		m = res.y;
	}

	if (t > tmax) {
		m = MAT_NONE;
	}
	return vec2(t, m);
}

// Function 317
vec4 march(in vec3 p, in vec3 dir){
    float d = 0.0;
    vec4 result = vec4(0.0);
    vec3 cp = p;
    
    for(int i = 0; i < steps; i++){
        cp = p + dir * d;
        result = scene(cp);
        d += result.w;
       
        
        if(result.w < delta)
        {
            break;
        }
        
        if(d > maxDist){
            result.rgb = vec3(1,0,0);
            break;
        }
    }
    
    return vec4(result.rgb, d);
}

// Function 318
vec4 raymarch(vec3 start_point, vec3 direction, int steps, float max_dist) {
    vec3 position = start_point;
    
    float dist = 0.0;
    
    for (int i=0; i<steps; i++) {
        float df = map(position);
        
        float threshold = 0.002 * dist;
        float step_size = df * 0.9; // This helps because it isn't a true SDF
        
        if ((df < threshold)) {
            return vec4(position, dist / max_dist);
        }
        if  (dist > max_dist) {
            return vec4(position, 1.0);
        }
        dist += step_size;
        position += direction * step_size;
    }
    return vec4(position, dist/max_dist);
}

// Function 319
vec4 RayMarch(vec3 coord, vec3 rayVector)
{
	vec3 marchCoord = coord;
	
	for (float t = 0.0; t < MAX_T; t++)
	{
		float dist = DELT * pow(t, LOD);
		marchCoord += rayVector * dist;
		
		if (InFloor(marchCoord))
		{
			float fog = (MAX_T - t) / MAX_T;
			return fog * FloorColor(marchCoord);
		}
	}
	
	return BackgroundColor();
}

// Function 320
vec3 march( vec3 _ro, vec3 _rd) {
    vec3 ro = _ro, rd = _rd, nrm;
    //start at starting loc
    float t = 0., dt = 0., ld = 0.;
    //sdf results
    mat4 c = mat4(0.);
    //march vars
    vec3 col = vec3(0.), pos = ro;
    //the march loop
    for( int i=0; i<steps; i++ ) {        
        //step the ray and update position
        t += dt;
        //position to test depth functions
        pos = ro+rd*t;
        //get sdf results
        c = map(pos, ro, rd, t, dt);
        //update light
        col += c[1].rgb;
        ro = c[2].xyz;
        rd = c[3].xyz;
        //control step size
        ld = dt;
        //step slower closer to the camera
        dt = min(.2, c[0].x);
    }
    
    //final color
    return col / float(steps) * 5.;
}

// Function 321
void march(vec3 origin, vec3 dir, out float t, out int hitObj)
{
    t = 0.001;
    for(int i = 0; i < RAY_STEPS; ++i)
    {
        vec3 pos = origin + t * dir;
    	float m;
        sceneMap3D(pos, m, hitObj, LIGHT_POS);
        if(m < 0.01)
        {
            return;
        }
        t += m;
    }
    t = -1.0;
    hitObj = -1;
}

// Function 322
void march( inout vec3 p, in vec3 d, in vec3 e ,bool Refl)
{
	float r = dist(p+d*EPSILON,Refl);
    vec3 dir = Refl?d:d*.5;
	for(int i = 0; i < V_STEPS; i++)
	{
		if(r < EPSILON || length(p-e) > MAX_DEPTH)
			return;
		p += dir*r; // The higher levels of noise in the rock may be skipped
        			  // if the steps are too large.
//        r = min(dist(p),sceneDist(p,sd));
  	    r = dist(p,Refl);
	}
	return;
}

// Function 323
float march(vec3 p, vec3 nv) {
    float lightAmount = 1.0;

    vec2 tRange;
    bool didHitBox;
    boxClip(BOX_MIN, BOX_MAX, p, nv, tRange, didHitBox);
    tRange.s = max(0.0, tRange.s);

    if (!didHitBox) {
        return 0.0;
    }

    float t = tRange.s;
    for (int i = 0; i < 150; i++) { // Theoretical max steps: (BOX_MAX-BOX_MIN)*sqrt(3)/RAY_STEP_L
        if (t > tRange.t || lightAmount < 1.0-QUIT_ALPHA_L) { break; }

        vec3 rayPos = p + t*nv;
        vec3 lmn = lmnFromWorldPos(rayPos);

        float density = getPage1(lmn).s;
        float calpha = clamp(density * MAX_ALPHA_PER_UNIT_DIST * RAY_STEP_L, 0.0, 1.0);

        lightAmount *= 1.0 - calpha;

        t += RAY_STEP_L;
    }

    return lightAmount;
}

// Function 324
vec4 raymarch (vec3 ro, vec3 rd)
{
    vec3 color = vec3(0.0);
	for (int i = 0; i < 512; i++)
	{
 		bool hit = InsideTetrahedronII(vec3(0.943, 0, -0.333 ),vec3( -0.471, 0.816, -0.333), vec3( -0.471, -0.816, -0.333), vec3(0, 0, 1 ),ro,color);
 		if (hit) return vec4(color,1.0); 
    	ro += rd * 0.01;
     }
	return vec4(0,0,0,1);
}

// Function 325
vec2 rayMarch(vec3 origin, vec3 direct)
{
    float res = 0.0;
    
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 tmp = origin + direct * res;
        vec2 d = getDist(tmp);
        res += d.x;
        
        if (d.x < EPSILON)
        	return vec2(res, d.y);
        
        if (res >= MAX_DIST)
            return vec2(MAX_DIST, 0);
    }

    return vec2(MAX_DIST, 0);
}

// Function 326
float march(vec3 eye, vec3 marchingDirection, float start, float end, out float outClr) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = scene(eye + depth * marchingDirection);
        if (dist < MIN_FLOAT) {
            return depth;
        }
        depth += dist * .75;
        outClr = 1. - float(i)/float(MAX_MARCHING_STEPS);
        if (depth >= end) {
            return end;
        }
    }
    return end;
}

// Function 327
MarchResult MarchRay(vec3 orig,vec3 dir)
{
    float steps = 0.0;
    float dist = 0.0;
    float id = 0.0;
    
    for(int i = 0;i < MAX_STEPS;i++)
    {
        vec2 object = Scene(orig + dir * dist);
        
        //Add the sky dome and have it follow the camera.
        object = opU(object, -sdSphere(dir * dist, MAX_DIST, SKYDOME));
        
        dist += object.x * STEP_MULT;
        
        id = object.y;
        
        steps++;
        
        if(abs(object.x) < MIN_DIST * dist)
        {
            break;
        }
    }
    
    MarchResult result;
    
    result.position = orig + dir * dist;
    result.normal = Normal(result.position);
    result.dist = dist;
    result.steps = steps;
    result.id = id;
    
    return result;
}

// Function 328
float RayMarch(vec4 rayOrigin, vec4 rayDistance)
{
    float distanceOrigin = 0.0018;
    
    for(int i = 0; i < MAX_STEPS; i++)
    {
        //Get next point
        vec4 p = rayOrigin + rayDistance*distanceOrigin;
        
        //Get distance from new point to the scene
        float dS  = GetDist(p);
        
        //Add distance from new point to total distance from origin
        distanceOrigin += dS;
        
        //If we have marched too far or if we are sufficiently close to scene, we're done marching
        if(distanceOrigin > MAX_DIST || dS < SURF_DIST) break;
    }
    return distanceOrigin;
}

// Function 329
vec2 march( vec3 eye, vec3 direction ) {
    vec2 total = vec2( .0, -1. );
    vec3 p = eye;
    for ( float i = .0 ; i < FAR ; i++ ) {
        vec2 current = map( p );
        total.x += current.x;
        total.y = current.y;
        if ( total.x > FAR || abs( current.x ) < NEAR ) break;
        p += current.x * direction;
    }
    total.y = mix( -1., total.y, step( total.x, FAR ) );
    return total;
}

// Function 330
vec2 RayMarch(in vec3 origin, in vec3 rayDirection, inout vec3 mtl)
{
    float material = -1.0;
    float t = 0.01;
	for(int i = 0; i < 64; ++i)
    {
        vec3 p = origin + rayDirection * t;
        vec2 hit = SdScene(p, mtl);
        if (hit.x < 0.001 * t || t > 50.0)
			break;
        t += hit.x;
        material = hit.y;
    }
    
    if (t > 50.0)
    {
     	material = -1.0;   
    }
    return vec2(t, material);
}

// Function 331
float lightMarch(vec3 sun_dir, vec3 start)
{
    // find dst to edge of sphere
    float lp;
    float dst;
    vec3 p = start;
    float dstThroughCloud = 0.;
    // stepping through the cloud using ray marching to find the distance through the sphere in the direction of the sun
    for (int s = 0; s < 25; s++)
    {
        lp = length(p);
        dst = -(lp - 6.);
        dstThroughCloud += dst;
        p += sun_dir * dst;
    }
    dstThroughCloud = min(dstThroughCloud, 1.25);
    // find total density along ray
    float density;
    p = start;
    float total_density = 0.;
    float step_size = dstThroughCloud / 21.;
    vec3 step_size_v3 = sun_dir * step_size;
    // stepping through the cloud and adding up the density
    for (int s = 0; s < 20; s++)
    {
        p += step_size_v3;
        density = max(SampleNoise(p), 0.) * step_size * 4. * HeightScale(p);
        total_density += density;
    }
    // returning the total desnsity acumulated on the rays journey
    return total_density;
}

// Function 332
vec3 march(vec3 from, vec3 dir) {
	vec3 p, col=vec3(0.);
    float totdist=0., d;
    for (int i=0; i<100; i++) {
    	p=from+totdist*dir;
        d=de(p);
    	totdist+=max(det,d);
        if (totdist>maxdist||length(col)>.3) break;
        col+=max(0.,det-d)*l;
    }
	col=.96-col*2.5*vec3(3.,2.,1.);
    return col;
}

// Function 333
vec2	march(vec3 pos, vec3 dir)
{
    vec2	dist = vec2(0.0, 0.0);
    vec3	p = vec3(0.0, 0.0, 0.0);
    vec2	s = vec2(0.0, 0.0);

	    for (float i = -1.; i < I_MAX; ++i)
	    {
	    	p = pos + dir * dist.y;
	        dist.x = scene(p);
	        dist.y += dist.x*.2; // makes artefacts disappear
            // log trick by aiekick
	        if (log(dist.y*dist.y/dist.x/1e5) > .0 || dist.x < E || dist.y > FAR)
            {
                break;
            }
	        s.x++;
    }
    s.y = dist.y;
    return (s);
}

// Function 334
vec3 raymarch(vec3 p, vec3 dir) {
    float contrib = 1.;
    vec3 color = vec3(0., 0., 0.);

    for (int i = 0; i < STEPS; i++) {
        Thing thing = scene(p);

        if (abs(thing.dist) < EPS) {
            vec3 n = normal(p);
            vec3 light_dir = normalize(vec3(0., 1., 0.));

            // Ambient lighting
            color += contrib * (1. - thing.refl) * thing.color;

            contrib *= thing.refl;

            p -= 2. * EPS * dir; // back up a bit

            // Reflection
            dir -= 2. * dot(n, dir) * n;
        }

        p += thing.dist * dir;
    }

    return color;
}

// Function 335
float rayMarchToAreaLight(in vec3 ro, in vec3 rd)
{
    const float numStep = 4.0;
    float shadow = 1.0;
  	float stepDist = length(rd - ro) / numStep;
    vec3 dir = normalize(rd - ro);
    for(float i = 0.5; i < numStep; i += 1.0)
    {
        vec3 pos = ro + dir * (i / (numStep));
        shadow *= exp(-densityVoumetricFog(pos * 253.0) * stepDist);
    }
    return shadow;
}

// Function 336
float march(vec3 ro, vec3 rd) {
	float tmin = 0.0;
	float t = tmin;
    float d;

	for (int i = 0; i < MaxSteps; i++) {
		d = map(ro + rd * t);
		if (d < precis || t > Far) {
			break;
		}
		t += d * MarchDumping;
	}
	return t;
}

// Function 337
float march(inout vec3 pos, vec3 dir)
{
    float rval=0.0;
    float R = 1.01;
    float d0=length(pos);

    // do some bounding check for better performance
    bool inside = false;
    // bounding sphere
    vec3 pn = pos-dir*dot(pos,dir);
    float d=length(pn);
    inside = inside || (d<R);
    inside=true;
    if(!inside) return 0.0;
    
    float eps=.001;
    for(int i=0;i<80;i++)
    {
       	float d=dist(pos);
        if(d<eps) { rval=mdist(pos).y; break; }
        if(d>d0+R) { rval=0.0; break; }
        pos+=dir*d*1.;
    }
    return rval;
}

// Function 338
void stepState()
{
	rand = rand ^ (rand << 13u);
	rand = rand ^ (rand >> 17u);
	rand = rand ^ (rand << 5u);
	rand *= 1685821657u;
}

// Function 339
float smoothStep(float edge0, float edge1, float x)
{
    x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0); 
    return x*x*(3.0- 2.0*x);
}

// Function 340
float smoothstep_c( float x, float c, float r ) { return smoothstep( c - r, c + r, x ); }

// Function 341
void SimulationStep(inout particle U)
{
    vec4 border = border_grad(U.pos.xyz);
    vec3 cvec = -U.pos.xyz*vec3(0,0,1);
    vec3 G = 0.15*normalize(cvec)/size3d;
   
    vec3 force =calc_force(U.pos.xyz);
    vec3 bound =1.*normalize(border.xyz)*exp(-0.4*border.w*border.w);
    float cooling = 1. - (1.-exp(-0.3*length(U.vel.xyz)))*(0.01*exp(-0.05*dot(cvec,cvec)) + 0.01*exp(-0.4*border.w*border.w) + 0.08*exp(-0.1*dot(force,force)));
    U.vel.xyz =  U.vel.xyz*cooling + dt*(bound+force+G);
    U.pos.xyz += dt*U.vel.xyz;
}

// Function 342
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

// Function 343
float linearstep( float a, float b, float x ) { return saturate( ( x - a ) / ( b - a ) ); }

// Function 344
float march(inout vec3 p, vec3 dir)
{
    //if(!intersectBox(p-bbpos,dir,bbsize)) { enable_car=false; }
    vec3 pc=carTrafo(p);
    vec3 pdir=carTrafo(dir,0.);
    //enable_car=true;
    if(!intersectBox(pc-bbpos,pdir,bbsize)) { enable_car=false; }
    if(terrbbsize!=vec3(0) && (!intersectBox(p-terrbbpos,dir,terrbbsize))) { enable_terr=false; }
    //if(!(intersectBox(pc-bbpos1,pdir,bbsize1)||intersectBox(pc-bbpos2,pdir,bbsize2))) { enable_car=false; }
    vec3 p0=p;
    float eps=.001;
    float dmin=100000.;
    bool findmin=false;
    float d=dist(p);
    vec3 pmin=p;
    for(int i=min(0,iFrame);i<150;i++)  // min(0,iFrame) avoids unrolling of loop - thx Dave_Hoskins
    {
        float dp=d;
        d=dist(p);
        p+=dir*d*.7;
#ifdef SHADOW
        if (d<dp) findmin=true;
        if (findmin && d<dmin) { dmin=d; pmin=p; }
#endif
        if (d<eps) return 0.;
#ifndef SHADOW
        if (dmin<50. && d>50.) break;
#else
        if (dmin<150. && d>150.) break;
#endif
        //if (length(p)>100.) break;
    }
    float lmin=length(pmin-p0);
    return clamp(dmin/lmin/.05,0.,1.);
}

// Function 345
float expstep(float x,float k,float n){
	return exp(-k*pow(x,n));
}

// Function 346
float GetRayFirstStep( const in C_Ray ray )
{
    return ray.fStartDistance;  
}

// Function 347
vec3 march(vec3 o, vec3 dir){
    vec3 p = o;
    float e = 0.0;
    for(int i = 0; i < 100; ++i){
        float d = 0.5*map(p);
        e += d;
        if(d < 0.005 || e > 12.0)
            break;
        p += d*dir;
    }
    
    return p;
}

// Function 348
uint tausStep(uint z,int s1,int s2,int s3,uint M){
    uint b=(((z << s1) ^ z) >> s2);
    return (((z & M) << s3) ^ b);
}

// Function 349
float raymarch(vec3 ray_start, vec3 ray_dir, out float dist, out vec3 p, out int iterations
){dist = 0.0 + 110.1*hash1(gl_FragCoord.xy + time)
 ;float minStep = .01
 ;vec2 mapRes
 ;for (int i = 1; i <= MAX_RAYMARCH_ITER; i++
 ){p = ray_start + ray_dir * dist
  ;mapRes = map(p, ray_dir)
  ;if (mapRes.y < MIN_RAYMARCH_DELTA
  ){iterations = i
   ;return mapRes.x
   ;}
  ;dist += max(mapRes.y, minStep)
  ;}
 ;return -1.;}

// Function 350
raymarch_result raymarch(
    vec3 eye, vec3 direction, float start, float end)
{
    raymarch_result result;
    float depth = start;
    int stepCount = 0;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        stepCount = i;
        float dist = sceneSDF(eye + depth * direction);
        if (dist < EPSILON) {
            result.depth = depth;
            result.steps = stepCount;
            return result;
        }
        depth += dist;
        if (depth >= end) {
            result.depth = end;
            result.steps = stepCount;
            return result;
        }
    }
    result.depth = end;
    result.steps = stepCount;
    return result;
}

// Function 351
vec3 march(in vec3 ro, in vec3 rd) {
    float dist = EPSILON;
    float totalDist = 0.0;
    for (int i = 0; i < MAX_RAY_ITERS; i++) {
        if (abs(dist) < EPSILON ||
            totalDist > MAX_RAY_DIST) {
            break;
        }
        
        dist = map(ro);
        totalDist += dist;
        ro += dist * rd;
    }
    
    if (abs(dist) < EPSILON) {
        return lighting(ro, rd, totalDist);
    } else {
        return GLOBAL_AMBIENT;
    }
}

// Function 352
void march(
    in vec3 p, in vec3 nv,
    out float didHit, out vec3 hitPos,
    out vec3 nvHitNormal, out vec3 hitUV, out int hitBranchDepth
) {
    // Update range for root sphere
    vec2 tRangeRoot;
    float didHitSphere;
    sphereClip(SPHERE_CENTER, SPHERE_RADIUS, p, nv, tRangeRoot, didHitSphere);
    if (didHitSphere < 0.5) {
        didHit = 0.0;
        return;
    }

    // transform is p \mapsto M(p - O)
    mat3 transMInv = ID_3X3;
    mat3 transM = ID_3X3;
    vec3 transO = vec3(0.0);

    vec3 pTransRay = p;
    vec3 nvTransRay = nv;
    vec2 tRangeCur = tRangeRoot;

    int branchDepth = 0;
    int branch = -1;

    for (int i=0; i<100; i++) { // TODO

        vec2 hitData = vec2(tRangeCur.t, 0.0);

        // Check object hit within sphere
        float didHitObject;
        float tHitObject;
        vec3 uvHitObject;
        hitObject(
            pTransRay, nvTransRay, vec2(max(0.0, tRangeCur.s), tRangeCur.t),
            didHitObject, tHitObject, uvHitObject
        );
        if (didHitObject > 0.5) {
            hitData = minHitData(tHitObject, hitData, tRangeCur, 0.1);
        }

        // Check "left" subsphere hit
        vec2 tRangeSubsphereL;
        float didHitL;
        sphereClip(
            SUBSPHERE_CENTER_L, SUBSPHERE_RADIUS, pTransRay, nvTransRay,
            tRangeSubsphereL, didHitL
        );
        if (branchDepth < MAX_BRANCH_DEPTH && didHitL > 0.5) {
            hitData = minHitData(tRangeSubsphereL.s, hitData, tRangeCur, 0.2);
        }

        // Check "right" subsphere hit
        vec2 tRangeSubsphereR;
        float didHitR;
        sphereClip(
            SUBSPHERE_CENTER_R, SUBSPHERE_RADIUS, pTransRay, nvTransRay,
            tRangeSubsphereR, didHitR
        );
        if (branchDepth < MAX_BRANCH_DEPTH && didHitR > 0.5) {
            hitData = minHitData(tRangeSubsphereR.s, hitData, tRangeCur, 0.3);
        }

        if (hitData.y < 0.05) {

            // Exiting sphere: "pop" transform to parent sphere

            if (branchDepth == 0) {
                break;
            }
            popBranch(branch, branchDepth);
            makeT(branch, branchDepth, transM, transMInv, transO);

            pTransRay = transM * (p - transO);
            nvTransRay = normalize( transM * nv );

            vec2 tRangeParent;
            float didHitSphereParent;
            sphereClip(
                SPHERE_CENTER, SPHERE_RADIUS, pTransRay, nvTransRay,
                tRangeParent, didHitSphereParent
            );
            tRangeCur = vec2(tRangeCur.t/SUBSPHERE_ZOOM, tRangeParent.t);

        } else if (hitData.y < 0.15) {

            // Hit object--done!

            didHit = 1.0;
            vec3 hitPosTrans = pTransRay + tHitObject*nvTransRay;
            hitPos = transMInv*hitPosTrans + transO;

            vec3 hitNormal;
            objNormal(hitPosTrans, hitNormal);
            nvHitNormal = normalize(transMInv*hitNormal);

            hitUV = uvHitObject;
            hitBranchDepth = branchDepth;

            break;

        } else if (hitData.y < 0.25) {

            // Entered "left" subsphere; push transform and continue

            pushBranch(branch, branchDepth, 0);
            makeT(branch, branchDepth, transM, transMInv, transO);

            pTransRay = transM * (p - transO);
            nvTransRay = normalize( transM * nv );
            tRangeCur = tRangeSubsphereL * SUBSPHERE_ZOOM;

        } else if (hitData.y < 0.35) {

            // Entered "right" subsphere; push transform and continue

            pushBranch(branch, branchDepth, 1);
            makeT(branch, branchDepth, transM, transMInv, transO);

            pTransRay = transM * (p - transO);
            nvTransRay = normalize( transM * nv );
            tRangeCur = tRangeSubsphereR * SUBSPHERE_ZOOM;

        }
    }

}

// Function 353
void march(vec3 origin, vec3 dir, out float t, out int hitObj, vec3 lightPos)
{
    t = 0.001;
    for(int i = 0; i < RAY_STEPS; ++i)
    {
        vec3 pos = origin + t * dir;
    	float m;
        sceneMap3D(pos, m, hitObj);
        if(m < 0.01)
        {
            return;
        }
        t += m;
    }
    t = -1.0;
    hitObj = -1;
}

// Function 354
float gradientStep(float edge0, float edge1, float x)
{
    return smoothstep(edge0, edge1, x);

    // smootherstep
    //x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    //return x*x*x*(x*(x*6. - 15.) + 10.);
    
    // linear
    //x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    //return x;
}

// Function 355
vec4 March(vec3 start, vec3 dir, float maxd, float dd) {
    vec4 ddir = vec4(dir, 1.) * dd;
    vec4 pos = vec4(start, 0.);
    vec4 color = vec4(0.);
    prevval = vec4(-pos.y);
    for (int i = 0; i < kMarchSteps; ++i) {
        float density = 1.;
        
#if HEIGHT_FIELD
        // 2D height field.
        vec4 c = Fxy(pos.xyz);
#else 
        // 3D function volume rendering
        vec4 c = Fxyz(pos.xyz);
#endif        
        
        // axis clipping
        //c = pos.z > 0. ? vec4(0.) : c; 
        
        c = (pos.w > maxd ? vec4(0.) : c); // reject samples outside box
        c.a = clamp(c.a, 0., 1.);  // insure we have good alpha values.
        color = color + c * (1. - color.a);  // blending
        pos += ddir;
    }
    return color;
}

// Function 356
float dda_march( vec3 ro, vec3 rd,
                 float maxdist,
                 out vec4 scene_rgba )
{
 
    vec3 cell_coord = floor(ro/g_cellsize); cell_coord.y = 0.;
    vec3 rs = sign(rd);
    
    vec2 deltaDist = g_cellsize/rd.xz;
    vec2 sideDist = ((cell_coord.xz - ro.xz)/g_cellsize + 0.5 + rs.xz*0.5) * deltaDist;    

    float res = 0.0;
    vec3 mm = vec3(0.0);
    
    scene_rgba = vec4(0.);
    
    float t = 0.;
    
    vec3 pos = ro;
    vec3 cell_pos = mod(ro, g_cellsize) - .5 * g_cellsize;
    
    for( int i=0; i<32; i++ ) 
    {
        //if (scene_rgba.a > .95 || t >= maxdist) { break; }

        // DDA march along the xz boundaries, ignoring the y plane boundaries
        mm.xz = step(sideDist.xy, sideDist.yx);

        vec3 normal = vec3(0.); 
        normal.xz = mm.xz * rs.xz;
        cell_coord += mm * rs * vec3(1., 0., 1.);
        
        vec3 ddn = rd * -rs;
        vec3 po = .5 * g_cellsize * rs;
        vec3 plane_t = (rs * (cell_pos - po))/ddn;
        float cell_extent = min(plane_t.x, plane_t.z);        
        pos += cell_extent * rd;
        
        cell_pos = pos - g_cellsize * cell_coord - .5 * g_cellsize;

        vec4 cell_res = shade_cell(cell_pos, rd, cell_coord);

        t = length(pos - ro);

        // composite
        scene_rgba.rgb += cell_res.rgb * cell_res.a * exp(-.05 * t + 1.);
        scene_rgba.a += (1. - scene_rgba.a) * cell_res.a;

        sideDist += mm.xz * rs.xz * deltaDist;  
    }    
    
    return t;
}

// Function 357
void StepM (int mId, out vec3 rm, out vec3 vm, out vec4 qm, out vec3 wm)
{
  mat3 mRot;
  vec3 dr, rs, am, wam, rMom;
  float rSep, grav, dt;
#if ! PAR_SPH
  mat3 mRotN;
  vec3 rmN, vmN, wmN, rsN, dv, rms, vms, fc;
#endif
  grav = 10.;
  dt = 0.01;
  rm = GetR (mId);
  vm = GetV (mId);
  qm = GetQ (mId);
  wm = GetW (mId);
  mRot = QtToRMat (qm);
  am = vec3 (0.);
  wam = vec3 (0.);
#if ! PAR_SPH
  for (int n = VAR_ZERO; n < nObj; n ++) {
    rmN = GetR (n);
    if (n != mId && length (rm - rmN) < farSep) {
      vmN = GetV (n);
      mRotN = QtToRMat (GetQ (n));
      wmN = GetW (n);
      for (int j1 = VAR_ZERO; j1 < nSphObj; j1 ++) {
        rs = mRot * RSph (float (j1));
        rms = rm + rs;
        vms = vm + cross (wm, rs);
        dv = vms - vmN;
        fc = vec3 (0.);
        for (int j2 = VAR_ZERO; j2 < nSphObj; j2 ++) {
          rsN = mRotN * RSph (float (j2));
          dr = rms - (rmN + rsN);
          rSep = length (dr);
          if (rSep < 1.) fc += FcFun (dr, rSep, dv - cross (wmN, rsN));
        }
        am += fc;
        wam += cross (rs, fc);
      }
    }
  }
  for (int j = VAR_ZERO; j < nSphObj; j ++) {
    rs = RSph (float (j));
    rs = mRot * rs;
    dr = rm + rs;
    rSep = abs (dr.y);
    if (rSep < 1.) {
      fc = FcFun (vec3 (0., dr.y, 0.), rSep, vm + cross (wm, rs));
      am += fc;
      wam += cross (rs, fc);
    }
  }
#else
  for (int j = VAR_ZERO; j < nSphObj; j ++) {
    am += GetAS (mId * nSphObj + j);
    wam += GetWAS (mId * nSphObj + j);
  }
#endif
  rMom = vec3 (0.);
  for (int j = VAR_ZERO; j < nSphObj; j ++) {
    rs = RSph (float (j));
    rMom += dot (rs, rs) - rs * rs + 1./6.;
  }
  rMom /= float (nSphObj);
  wam = mRot * (wam * mRot / rMom);
  am.y -=  grav;
  vm += dt * am;
  rm += dt * vm;
  wm += dt * wam;
  qm = normalize (QtMul (RMatToQt (LpStepMat (0.5 * dt * wm)), qm));
}

// Function 358
Result raymarchHF(Ray ray, float mindist, float maxdist, float stepsize, bool water)
{
    Result result;
    float dist = mindist;
    float h = 1.0;
    float dh = 1.0;
    float lastdh = -1.0;
    float lastdist = 0.0;
    float fracstep = 0.5;
    
    float maxy = water?WAVEHEIGHT:0.0;
    dist = max(dist, planeIntersect(ray.pos, ray.dir, vec4(0.0, 1.0, 0.0, -maxy)));
        
    for (float i=0.0; i<90.0; i++)
    {
        vec3 pos = ray.pos + ray.dir*dist;
        h = terrain(pos, water);
        dh = pos.y - h;
        if (dh<(0.001*i) || dist>maxdist)
        {
            break;    
        }
        lastdh = dh;
        lastdist = dist;
        dist+=stepsize;
        stepsize+=(0.0004*i);
    }
    
    if (dist<maxdist)
    {
        fracstep = lastdh/(lastdh-dh);    
        dist = mix(lastdist, dist, fracstep);
    }
    result.dist = dist;
    result.travelled = dist;
    result.pos = ray.pos + ray.dir*dist; 
    const float dn = 0.1;	
    result.normal.x = terrain(result.pos, water) - terrain(result.pos+XAXIS*dn, water);
    result.normal.y = dn;
    result.normal.z = terrain(result.pos, water) - terrain(result.pos+ZAXIS*dn, water);
    result.normal = normalize(result.normal);
    result.mat = g_basic;
    return result;
}

// Function 359
void raymarch(vec3 from, vec3 increment)
{
	const int   kMaxIterations = 60;
	const float kHitDistance   = 0.01;
	
	marchDistance    = -15.0;
	marchMaterial    = -1.0;
	
	for(int i = 0; i < kMaxIterations; i++)
	{
		distanceField(from + increment * marchDistance);
		if (fieldDistance > kHitDistance)
		{
			marchDistance += fieldDistance;
			marchMaterial  = fieldMaterial;
		}
	}
	
	marchPosition = from + increment * marchDistance;
    if (marchDistance > 32.0)
        marchMaterial = -1.0;
}

// Function 360
float march(inout vec3 p, vec3 dir)
{
    //if(!intersectBox(p-bbpos,dir,bbsize)) { enable_car=false; }
    vec3 pc=carTrafo(p);
    vec3 pdir=carTrafo(dir,0.);
    //enable_car=true;
    if(!(intersectBox(pc-bbpos1,pdir,bbsize1)||intersectBox(pc-bbpos2,pdir,bbsize2))) { enable_car=false; }
    vec3 p0=p;
    float eps=.001;
    float dmin=1000.;
    bool findmin=false;
    float d=dist(p);
    vec3 pmin=p;
    for(int i=0;i<150;i++)
    {
        float dp=d;
        d=dist(p);
        p+=dir*d*.8;
#ifdef SHADOW
        if (d<dp) findmin=true;
        if (findmin && d<dmin) { dmin=d; pmin=p; }
#endif
        if (d<eps) return 0.;
        if (d>300.) break;
    }
    return clamp(dmin/length(pmin-p0)/.05,0.,1.);
}

// Function 361
float RayMarch(vec3 ro, vec3 rd){
    // distance from origin
    float dO=0.;
    // march until max steps is achieved or object hit
    for(int i=0; i <MAX_STEPS; i++){
        // current point being evaluated
        vec3 p = ro + dO*rd;
        
        // get distance to seam
        float ds = GetDist(p);
        //move origin to new point
        dO+=ds;
        if(ds < SURFACE_DIST || dO > MAX_DIST){
            break;
        }
    }
    return dO;
}

// Function 362
vec3 march( in vec3 p, in vec3 d, in vec3 e )
{
    float c;	// Current distance.
    for(int i = 0; i < MAX_V_STEPS; ++i)
    {
        // Get the current distance.
        c = dist(p);
        
        // If we're close enough, return the current position.
        if( c<EPSILON ) return p;
        
        // If the current march will take us further than the clip
        // distance, return the current position plus the clip distance.
        else if( c>MAX_DEPTH ) return e+d*MAX_DEPTH;
            
		// If we've gone far enough, return the current position.
        else if( length(p-e) > MAX_DEPTH ) return p;
            
        // Otherwise we march on!
        else p += d*c;
    }
    return p;
}

// Function 363
float march(vec3 ro, vec3 rd, float rmPrec, float maxd, float mapPrec)
{
    float s = rmPrec;
    float d = 0.;
    for(int i=0;i<150;i++)
    {      
        if (s<rmPrec||s>maxd) break;
        s = map(ro+rd*d).x*mapPrec;
        d += s;
    }
    return d;
}

// Function 364
vec4 RayMarch(vec3 ro, vec3 rd) {
	float dO=0.;
    float dS=0.;
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        dS = GetDist(p, ro);
        dO += dO>MAX_DIST || dS<SURF_DIST ? 0.00 : dS;
    }
    
    return vec4( ro + rd*dO,dO); // ##
}

// Function 365
vec4 march(vec3 from, vec3 dir, vec3 camdir) {
    // variable declarations
	vec3 p=from, col=vec3(0.1), backcol=col;
    float totdist=0., d=0.,sdet, glow=0., lhit=1.;
	// the detail value is smaller towards the end as we are closer to the fractal boundary
   	det*=1.-fin*.7;
    // raymarching loop to obtain an occlusion value of the sun at the camera direction
    // used for the lens flare
    for (int i=0; i<70; i++) {
    	p+=d*ldir; // advance ray from camera pos to light dir
        d=de(p)*2.; // distance estimation, doubled to gain performance as we don't need too much accuracy for this
        lhit=min(lhit,d); // occlusion value based on how close the ray pass from the surfaces and very small if it hits 
        if (d<det) { // ray hits the surface, bye
            break;
        }
    }
    // main raymarching loop
    for (int i=0; i<150; i++) {
    	p=from+totdist*dir; // advance ray
        d=de(p); // distance estimation to fractal surface
        sdet=det*(1.+totdist*.1); // makes the detail level lower for far hits 
        if (d<sdet||totdist>maxdist) break; // ray hits the surface or it reached the max distance defined
    	totdist+=d; // distance accumulator  
        glow++; // step counting used for glow
    }
    float sun=max(0.,dot(dir,ldir)); // the dot product of the cam direction and the light direction using for drawing the sun
    if (d<.2) { // ray most likely hit a surface
    	p-=(sdet-d)*dir; // backstep to correct the ray position
        vec3 c=fcol; // saves the color set by the de function to not get altered by the normal calculation
        vec3 n=normal(p); // calculates the normal at the ray hit point
        col=shade(p,dir,n,c); // sets the color and lighting
    } else { // ray missed any surface, this is the background
        totdist=maxdist; 
    	p=from+dir*maxdist; // moves the ray to the max distance defined
        // Kaliset fractal for stars and cosmic dust near the sun. 
        vec3 st = (dir * 3.+ vec3(1.3,2.5,1.25)) * .3;
        for (int i = 0; i < 10; i++) st = abs(st) / dot(st,st) - .8;
        backcol+=length(st)*.015*(1.-pow(sun,3.))*(.5+abs(st.grb)*.5);
        sun-=length(st)*.0017;
        sun=max(0.,sun);
		backcol+=pow(sun,100.)*.5; // adds sun light to the background
    }
    backcol+=pow(sun,20.)*suncol*.8; // sun light
    float normdist=totdist/maxdist; // distance of the ray normalized from 0 to 1
    col=mix(col,backcol,pow(normdist,1.5)); // mix the surface with the background in the far distance (fog)
    col=max(col,col*vec3(sqrt(glow))*.13); // adds a little bit of glow
	// lens flare
    vec2 pflare=dir.xy-ldir.xy;
    float flare=max(0.,1.0-length(pflare))-pow(abs(1.-mod(camdir.x-atan(pflare.y,pflare.x)*5./3.14,2.)),.6);
	float cflare=pow(max(0.,dot(camdir,ldir)),20.)*lhit;
    col+=pow(max(0.,flare),3.)*cflare*suncol;
	col+=pow(sun,30.)*cflare;
    // "only glow" part (at sec. 10)
    col.rgb=mix(col.rgb,glow*suncol*.01+backcol,1.-smoothstep(0.,.8,abs(time-10.5)));
    return vec4(col,normdist); // returns the resulting color and a normalized depth in alpha
}

// Function 366
vec2 raymarch_main(vec3 _ro, vec3 _rd, float _near, float _far, float t)
{
    vec2 close;
    close.x = 0.;
    for(int i = 0;i < MAIN_SCENE_MAX_STEPS; ++i)
    {
        vec3 p = _ro+_rd*close.x;
        // eval scene
        vec2 t = raymarch_main_scene(p, t);
        if(t.x<_near || close.x>_far)
            break;
        close.x += t.x*0.5;
        close.y = t.y;
    }
    return close;
}

// Function 367
vec4 raymarche( in vec3 org, in vec3 dir, in vec2 nfplane )
{
	float d = 1.0, g = 0.0, t = 0.0;
	vec3 p = org+dir*nfplane.x;
	
	for(int i=0; i<64; i++)
	{
		if( d > 0.001 && t < nfplane.y )
		{
			d = map(p);
			t += d;
			p += d * dir;
			g += 1./64.;
		}
	}
	
	return vec4(p,g);
}

// Function 368
void ray_march_scene(Ray r, float k, inout vec3 c)
{
    float uniform_step = k;
    float jit = 1.;
    //jit = 50.*fract(1e4*sin(1e4*dot(r.dir, vec3(1., 7.1, 13.3))));
   
    float t_gen = 1.;

    float param_t = intersect_sphere(r, m.pos, RADIUS);
    if(param_t <= -1.)
        return;
    vec3 p = ray_interpolation(r, k*jit);        
     
    //rgb transparency               
    
    vec3 t_acc = vec3(1.);	// accumulated parameters for transparency
    float t_loc = transp(uniform_step, 14., ( clamp(smoothstep(.2, 3.*RADIUS, (RADIUS-length(p))) - abs( 2.*(fbm(p/8.)) ), 0., 1.)  ) );
    int s = 0;
    
    for(s; s <90; s++)
    {               
        float dist_dist = dot(p-cam.pos, p-cam.pos);
        float dist_center = length(m.pos-cam.pos);
        vec3 center = p-m.pos;

        float d = length(center)-RADIUS-.5-jit*k;
        float size = length(center)/RADIUS;

        if(length(center)-RADIUS < 0.)
        {
            
            #if COLOR            
            #if ANIM      
            	anim = iTime/10.;
            
            #endif
            float n = fbm( ( 
                p/( clamp(0., RADIUS, length(center)) + cos(PI+snoise(p)) - 1./size*anim ) //shockwave stuff
            			) )  ;
            

            float mask = smoothstep(1.,
                                   	20.*RADIUS,
                                  	RADIUS/length(center));

            
            //Optical density/depth : dens for density
            float dens = ( clamp( mask,
                               	  0.,
                                  1.) *n);
            
           if(length(p-cam.pos) >(dist_center+m.radius) || 
           (k*dens  < -9.9))
        	{
         	break;
        	}
            //How colors (rgb) are absorbed at point p in the current iteration
            //k is the step size          
             vec3 rgb_t = exp(-vec3(
                		k * 15. * dens, 
                      	k * 10. * dens,
              	      	k * 15. * 1./size * dens));    
            
            t_acc *= (rgb_t);           
    
    		//blending
   			c += t_acc*vec3(1.)*(1.-rgb_t);
            #endif
        }

        //if it will never be in the shape anymore, return;        
        
        p += r.dir*k;

        k = uniform_step;
    }
    

    //c =float(s)/vec3(50,150,20); return;

    #if COLOR

    #else
    c = vec3(t_gen); return;
    #endif
}

// Function 369
vec4 RayMarcher(in vec2 fragCoord)
{
    vec2 p = -1.0 + 2.0 * gl_FragCoord.xy / iResolution.xy; // screenPos can range from -1 to 1
	p.x *= iResolution.x / iResolution.y; // Correct aspect ratio
    
    vec2 mo = iMouse.xy/iResolution.xy;
		 
	float time = 15.0 + iTime;

    //BackGround
    vec3 col = background;
    
    
	// camera	
	vec3 cameraOrigin 	= vec3( 0.0+mo.x*0.5, 4.0+mo.y, 4.0 ); //camera origin
	vec3 cameraTarget 	= vec3( 0.0, 0.0, 0.0 ); //Camera target
    vec3 upDirection 	= vec3(	0.0, -1.0, 0.0 );
	
	// camera-to-world transformation
    mat3 ca = setCamera(cameraOrigin, cameraTarget, upDirection );
    
    vec3 rayDir = normalize(ca[0] * p.x + ca[1] * p.y +ca[2]);
    
    render(col, rayDir, cameraOrigin);
    
    return vec4( col, 1.0 );
    
}

// Function 370
float march(in vec3 ro, in vec3 rd) {
  float maxd = 20.0;
  const float precis = 0.001;
  float t = 0.01;
  float res = 1e8;
  float K = 0.2;
  for (int i = 0; i < 200; i++) {
    vec3 p = ro + rd * t;
    float d = eval(p);
    //d *= 0.6;       // Fudge factor
    //d = min(0.5,d); // Try and avoid overstepping
    t += d/(1.0+K*d);
    if (d < precis) return t;
    if (t > maxd) break;
  }
  return 1e8;
}

// Function 371
vec3 vmarch(Ray ray, float dist)
{   
    vec3 p = ray.ro;
    vec2 r = vec2(0.);
    vec3 sum = vec3(0);
    vec3 c = hue(vec3(0.,0.,1.),5.5);
    for( int i=0; i<20; i++ )
    {
        r = map(p);
        if (r.x > .01) break;
        p += ray.rd*.015;
        vec3 col = c;
        col.rgb *= smoothstep(.0,0.15,-r.x);
        sum += abs(col)*.5;
    }
    return sum;
}

// Function 372
float periodicsmoothstep( float x )
{
    float mx = mod(x, 4.);
    return smoothstep(0., .2, mx) - smoothstep(2., 2.2, mx);
}

// Function 373
vec2 march(vec3 origin, vec3 ray){
    float rayLength = MARCH_MIN;
    float material = MAT_BG+.0678;
    bool hit = false;
    for(float i=0.;i<80.;i++){
        vec2 m = map(origin+ray*rayLength);
    	float dist = m.x;
        float mat = m.y;
        rayLength += dist*.99;
        if(rayLength>=MARCH_MAX) break;
        if(dist<.01) {
        	hit = true;
            material = mat;
            break;
        }
    }
    if(hit==false) rayLength = MARCH_MAX;
    return vec2(rayLength,material);
}

// Function 374
vec3 ray_marching( vec3 origin, vec3 dir, float start, float end ) {
		
		float depth = start;
		vec3 salida = vec3(end);
		vec3 dist = vec3(0.1);
		
		for ( int i = 0; i < max_iterations; i++ ) 		{
			if ( dist.x < stop_threshold || depth > end ) break;
                dist = map( origin + dir * depth );
                depth += dist.x;
		}
		
		salida = vec3(depth, dist.y, dist.z);
		return salida;
	}

// Function 375
Result raymarch_query(Ray ray, float maxdist)
{
    float mint=TOO_FAR;
    float maxt=0.0;
	float travelled=0.0;
    for (int i=0; i<MARCH_ITERATIONS; i++)
    {
    	SDFResult res = sceneSDF(ray.pos);
        
      	maxt = max(maxt, res.dist);    
       	if (res.dist<maxt)    
        {
	        mint = min(mint, res.dist);            
        }
                
        ray.pos += res.dist*ray.dir; 
        travelled += res.dist;
        
        if (travelled>maxdist)
            break;
    }     
    
    Result result = resultSDF(ray.pos);
    result.mint = mint;
    result.travelled=travelled;
    return result;
}

// Function 376
vec3 march(vec3 ro, vec3 rd){
    
    vec3 p = ro;
    float t = 0.0;
    
    for(int i = 0; i < STEPS; ++i){
        float d = dist(p);
        t += d;
        p += rd*d;
        
        if(d < EPSILON || t > FAR){
            break;
        }
    }
    
    vec3 col = vec3(0.0);
    if(t <= FAR){
        col = vec3(1.0);
    }
    
    return col;
}

// Function 377
float raymarch_againstphotons( vec3 o, vec3 target, float start_time, out vec4 objPos )
{
    return raymarch( o, target, start_time, -1., objPos );
}

// Function 378
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

// Function 379
float aastep(float x) {
    return ( x<0.001 ? 0. : 1.0-sin(pi*x)/(pi*x) );
}

// Function 380
float march (vec3 o, vec3 r){
 	float t = 0.0;
    for(int i = 0; i < nsteps; i++){
		vec3 p = o + r * t;
        float d = map (p);
        t += d * 0.5;
    }
   return t;
}

// Function 381
vec4 raymarch(vec3 from, vec3 increment)
{
	const float maxDist = 200.0;
	const float minDist = 0.1;
	const int maxIter = 120;
	
	float dist = 0.0;
	
	float material = 0.0;
	
	for(int i = 0; i < maxIter; i++) {
		float distEval = distanceField(from + increment * dist, material);
		
		if (distEval < minDist) {
			break;
		}
		
		dist += distEval;
	}
	
	
	if (dist >= maxDist) {
		material = 0.0;
	}
	
	return vec4(dist, 0.0, 0.0, material);
}

// Function 382
vec3 march( in vec2 p )
{
    vec4 h=vec4(0.0);
	for( int i=0; i<32; i++ )
    {
        h = map(p);
        if( h.x<0.001 ) break;
        p = p + h.x*randomInCircle();
    }
    return h.yzw;
}

// Function 383
Intersection Raymarch(Camera camera)
{    
    Intersection outData;
    outData.sdf = 0.0;
    outData.density = 0.0;
    outData.totalDistance = MIN_DISTANCE;
        
	for(int j = 0; j < MAX_STEPS; ++j)
	{
        vec3 p = camera.origin + camera.direction * outData.totalDistance;
		outData.sdf = sdf(p);

		outData.totalDistance += outData.sdf;
                
		if(outData.sdf < EPSILON || outData.totalDistance > MAX_DISTANCE)
            break;
	}
    
    return outData;
}

// Function 384
float4 rayMarch(float3 rayOrigin, float3 rayStep, out float3 pos)
{
	float4 sum = float4(0, 0, 0, 0);
	pos = rayOrigin;
	for(int i=0; i<_VolumeSteps; i++) {
		float4 col = volumeFunc(pos);
		col.a *= _Density;
		col.a = min(col.a, 1.0);
		
		// pre-multiply alpha
		col.rgb *= col.a;
		sum = sum + col*(1.0 - sum.a);	
#if 0
		// exit early if opaque
        	if (sum.a > _OpacityThreshold)
            		break;
#endif		
		pos += rayStep;
	}
	return sum;
}

// Function 385
void TestLineMarch(in vec3 rayPos, inout SRayHitInfo info, in vec3 A, in vec3 B, in float width, in SMaterial material)
{   
    vec3 normal;
    float dist = LineDistance(A, B, width, rayPos, normal);
    if (dist < info.dist)
    {
        info.rayMarchedObject = true;
        info.dist = dist;        
        info.normal = normal;
        info.material = material;
    }    
}

// Function 386
float marchingCubePath(vec3 p, float t) {
	if (t < 0.0) 
		return sdBox(p - vec3(2.5, max(-0.5 - 3.0 * (t + 1.0), 0.5), 0.5), vec3(0.40)) - 0.05;
	else if (t < 2.0) 
		return marchingCube(p - vec3(2.5, 0.5, 0.5), t);
	else if (t < 5.0) 
		return marchingCube(p.zyx * vec3(1.0, 1.0, -1.0) - vec3(2.5, 0.5, -2.5), t - 2.0);
	else if (t < 8.0) 
		return marchingCube(p.zxy * vec3(1.0, -1.0, -1.0) - vec3(2.5, 0.5, -0.5), t - 5.0);
	else if (t < 11.0) 
		return marchingCube(p.yxz * vec3(1.0, -1.0, -1.0) - vec3(-2.5, 0.5, -2.5), t - 8.0);
	else if (t < 14.0)
		return marchingCube(p.yzx * vec3(1.0, -1.0, 1.0) - vec3(-2.5, 0.5, -0.5), t - 11.0);
	else if (t < 16.0)
		return marchingCube(p.xzy * vec3(1.0, -1.0, 1.0) - vec3(2.5, 0.5, -2.5), t - 14.0);
	else if (t < 18.0)
		return marchingCube(p.yzx * vec3(1.0, -1.0, -1.0) - vec3(-0.5, 0.5, -2.5), t - 16.0);
	else
		return tumblingCube(p, t - 18.0);
}

// Function 387
vec3 rayMarch( in vec3 ro, in vec3 rd, out float idObj, out vec3 ptCol ) {

        float dist = 0.0;
        vec3 vdist;
        vec3 np = ro;
        for( int i = 0; i < MAX_ITERATIONS; i++ ) {
            
            vdist = map(np);
            dist = vdist.x;
            if (dist < 0.001)
                break;
            np += rd * dist;

        }

        idObj = vdist.y;
        ptCol = np;

        if (dist < 0.01) {
            return colorize(np, vdist.y);
        }

        return vec3( 0.6, .6, 0.8);    

    }

// Function 388
vec4 raymarch(vec3 from, vec3 increment)
{
	const float maxDist = 200.0;
	const float minDist = 0.001;
	const int maxIter = RAYMARCH_ITERATIONS;
	
	float dist = 0.0;
	
	float lastDistEval = 1e10;
#ifdef TRADITIONAL
	float edge = 1.0;
#else
	float edge = 0.0;
#endif
	
	for(int i = 0; i < maxIter; i++) {
		vec3 pos = (from + increment * dist);
		float distEval = distf(pos);
		
#ifdef TRADITIONAL
		if (distEval < minDist) {
			if (i > RAYMARCH_ITERATIONS - 5) edge = 0.0;
			// Probably should put a break here, but it's not working with GL ES...
		}
#else
	#ifdef PERSPECTIVE_FIX
		// Could not figure out the math :P
		if (lastDistEval < (EDGE_WIDTH / dist) * 20.0 && distEval > lastDistEval + 0.001) {
			edge = 1.0;
		}
	#else
		if (lastDistEval < EDGE_WIDTH && distEval > lastDistEval + 0.001) {
			edge = 1.0;
			// Also should put a break here, but it's not working with GL ES...
		}
	#endif
		if (distEval < minDist) {
			break;
		}
#endif
		
		dist += distEval;
		if (distEval < lastDistEval) lastDistEval = distEval;
	}
	
	return vec4(dist, 0.0, edge, 0);
}

// Function 389
void marchThroughField(inout vec3 pos, vec3 dir)
{
	float dist;
	
	for(int i = 0; i < MAX_VIEW_STEPS; i++)
	{
		dist = getDist(pos);
		if(dist < EPSILON || dist > MAX_DEPTH)
			return;
		else	
			pos += dir*dist*.75;
	}
	return;
}

// Function 390
void MarchLight(inout RayInfo r, float startTime, float maxDist
){float totalDist = 0.0
 ;vec3 origin = r.b
 ;for (r.iter=0;r.iter<maxStepRayMarching;r.iter++
 ){r.time = startTime
  ;SetTime(r.time)
  ;r.dist = map(r.b,-1)
  ;totalDist += r.dist
  ;r.b+= r.dir*(r.dist)
  ;if(abs(r.dist)<rayEps||totalDist>maxDist)break;}}

// Function 391
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

// Function 392
vec3 RayMarch(vec3 ro, vec3 rd) {
	vec3 color = BACKGROUND_COLOR;
    float st = 0.0;
    vec3 p = vec3(0.0);
    
    for (int i = 0; i < MAX_STEPS; i++) {
    	p = ro + rd*st;
        vec4 mesh = MAP_Scene(p, ro, 0);
        
        // Scene Color
        if (mesh.w <= SURFACE_DISTANCE) {
            mesh = MAP_Scene(p, ro, 1);
            vec3 sceneColor = mesh.xyz;
        	
             if (st>=FOG_START) {
            	float nrmlz = st - FOG_START;
                nrmlz /= FOG_END - FOG_START;
                sceneColor = mix(sceneColor, FOG_COLOR, pow(nrmlz, 1.0));
            }
            
            color = sceneColor;
            break;
        }
        
        if (st >= MAX_RENDER_DISTANCE) {
        	break;
        }
        
        st += mesh.w;
    }
    
    // Sky(Background) Color
    if (st >= MAX_RENDER_DISTANCE) {
    	color = mix(SKY_DOWN_COLOR, SKY_UP_COLOR, pow(abs(p.y), 0.1));
    }
    
    return color;
}

// Function 393
float RayMarch(vec3 ro, vec3 rd) {
    float dO = 0.0;
    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd*dO;
        float dS = GetDist(p);
        dO += dS;
        if(dO>MAX_DIST || dS<SURF_DIST) break;
    }
    
    return dO;
}

// Function 394
vec4 Raymarch(vec3 ro, vec3 rd, out float emission) 
{
	float dO=0.;
    vec3 color = vec3(0.);
    
    for(int i=0; i<MAX_STEPS; i++) 
    {
    	vec3 p = ro + rd*dO;
        float dS = Scene(p);
        dO += dS;
        
        if(dO>MAX_DIST || dS<SURF_DIST) break;

        float tunnel = 1. - saturate(abs(dS - ObjTunnel(p)));

        color = vec3(.2,.05,.05); // ground
        color = mix(color, vec3(1,0,0), smoothstep(0.2, 1., tunnel));
    }
    
    return vec4(dO, color);
}

// Function 395
vec3 fogMarch(vec3 rayStart, vec3 rayDirection, float time, float disMod)
{
    float stepLength = RENDER_DISTANCE / float(RAYS_COUNT);
 	vec3 fog = vec3(0.0);   
    vec3 point = rayStart;
    
    for(int i = 0; i < RAYS_COUNT; i++)
    {
     	point += rayDirection *stepLength;
        fog += volumetricFog(point, disMod) //intensity
            * mix(COLOR1, COLOR2 * (1.0 + disMod * 0.5), getNoiseFromVec3((point + vec3(12.51, 52.167, 1.146)) * 0.5)) //coloring
            * mix(1.0, getNoiseFromVec3(point * 40.0) * 2.0, DITHER)	//Dithering
            * getNoiseFromVec3(point * 0.2 + 20.0) * 2.0;	//Cutting big holes
        
        stepLength *= STEP_MODIFIER;
    }
    
    //There is a trick
    //Cutting mask in result, it will fake dynamic fog change, cover imperfections and add more 3D feeling
   	fog = (fog / float(RAYS_COUNT)) * (pow(getNoiseFromVec3((rayStart + rayDirection * RENDER_DISTANCE)), 2.0) * 3.0 + disMod * 0.5);
	
    return fog;
}

// Function 396
vec4 raymarchSea(vec3 ro, vec3 rd, vec3 up, vec3 fwd, in mat3 cam,
                  in float tmin, in float tmax,
                  in vec3 Lo,
                  in vec3 Lcolor)
{
    if(tmax<0. || tmin<0.) return vec4(1., 0., 0., 0.);
    float maxTargetTravel = tmax-tmin;
    float cStep = 0.08;
    float vStep = maxTargetTravel*cStep;

    float t = 0.;
    if((length(ro+rd*tmin-EARTHPOS)-EARTHRADIUS)<tmin)
        return vec4(1., 0., 0., 1.);
    for(int i = 0;i<60;++i)
    {
        vec3 p = ro+rd*t;
        float depth = mapSea(p, up);
        if(depth<tmin || t>tmax) break;
        t += 0.5*depth;
    }

    float rdy = clamp((tmax-t)/(tmax-tmin), 0., 1.);
    
    rdy = exp(-10.*(t-tmin)/(tmax-tmin));
    
    vec4 c = vec4(1., 0., 0., clamp(rdy*rdy, 0., 1.));
    
    // lighting
    #if 1
    vec3 Pw = ro+rd*t;
    vec3 Nw = normalSea(Pw, up, fwd, 0.01);
    vec3 Lw = normalize(Lo-Pw);
    #if 0
    float tsmin = -1.0, tsmax = -1.0;
    float de = IntersectSphere(
        ro, rd, Lp, SUNRADIUS,
        tsmin, tsmax);
    vec3 H = normalize(-rd+normalize((ro+rd*tsmin)-Pw));
    #else
    vec3 H = normalize(-rd+Lw);
    #endif
    float NoL = dot(Nw, Lw);
    float NoV = dot(Nw, -rd);
    
    float HoN = max(0., dot(H, Nw));
    float F = clamp(Fresnel_Schlick(1., 1.033, NoV), 0., 1.);
    
    vec3 diffuse = mix(vec3(0.02, 0.25, 0.45), vec3(0.46, 0.71, 0.76), F);
    
    c.rgb = SUNINTENSITY * (
        diffuse*max(0., NoL)
        + diffuse*.2*max(0., dot(Nw, up))
        + Lcolor * pow(max(0., HoN), 64.)
        );
    c.a *= F;
    #else
        vec3 Pw = ro+rd*t;
        vec3 Nw = normalSea(Pw, up, fwd, 0.01);
    	c.rgb = oceanLighting(ro, rd, Lo, Pw, Nw, fwd, cross(fwd, up));
    //vec3 pouet;
    //c.rgb = getSkyLight(Pw, reflect(rd, vec3(0., 1., 0.)), Lo, SKY_BETA_RAY, SKY_BETA_MIE, pouet, SKY_MIE_HEIGHT);
    #endif
    
	return c;
}

// Function 397
vec3 raymarch(vec3 pathdir, vec3 pathorig){
    vec3 pathpos = pathorig;
    pathpos += pathdir*14.0;
    vec3 surfnormal;
    float distest, lightdistest;
    int bounces = 0;
    int object = 0;
    vec3 closestpos = pathpos;
    vec3 outCol = vec3(1.0);
    for(int i = 0; i < maxmarches; i++){
        // Check if the path is done
        if(length(pathpos) > scenesize || pathpos.z < -3.9 || bounces > maxbounces){break;}
        if(light(pathpos) < collisiondist){return outCol*vec3(1.0);}

        // Find the distance to the scene
        distest = DE(pathpos);
        lightdistest = light(pathpos);

        // Michael0884: Closest Non-Colliding Position
        if(distest > min(collisiondist, lightdistest)){closestpos = pathpos;}

        // Bounce the Path if it hits something
        if(distest < collisiondist){
            int object = getmat(pathpos);
            vec4 matprops = materialproperties(pathpos, object);
            outCol *= matprops.rgb;
            surfnormal = normal(pathpos);
            pathpos = closestpos;
            pathdir = reflect(pathdir, normalize(nrand3(matprops.w, surfnormal)));
            bounces++;
        }

        // Otherwise just keep going
        else{pathpos += pathdir*min(distest, lightdistest);}
    }
    return vec3(0.0);
}

// Function 398
vec4 raymarch(float end, vec3 pos, vec3 dir, float h)
{
    float total = 0.0;
    for (int i = 0; i < MARCHMAX; i++)
    {
        vec3 current = pos + total * dir;
        // NOTE: form must fit inside in 0..1 grid-cell!
        current -= 0.5; // center in 0..1 cube
        float dist = mix(sphereSDF(current), cubeSDF(current), h);
        if (dist < MARCHEPS) // We're inside the scene surface!
        {
            vec3 n = mix(sphereNRM(current), cubeNRM(current), h);

            return vec4(n * 0.5 + 0.5, 1.0);
        }

        // Move along the view ray
        //total += dist; // XXX artifacts on cube-edges when using full step-size (possibly due to GRIDEPS?) XXX
        total += 0.5 * dist; // XXX half step-size hides (or solves?) this XXX

        if (total >= end) // Gone too far; give up
        {
            return vec4(0.0);
        }
    }

    return vec4(0.0);
}

// Function 399
float rayMarching(vec3 eye, vec3 dir) {
    float full_dist = MIN_DIST;
    
    for (int i = 0; i < MARCH_STEPS; i++) {
        float current_dist = sceneSDF(eye + dir * full_dist);
        
        if (current_dist < EPSILON) {
            return full_dist;
        }
        
        full_dist += current_dist;
        if (full_dist >= MAX_DIST) {
            return MAX_DIST;
        }
    }
    
     return MAX_DIST;
}

// Function 400
float marchScene(in vec3 rO, in vec3 rD, vec2 co)
{
	float t = 5.+10.*hash12(co);
    float oldT = 0.;
	vec2 dist = vec2(1000);
	vec3 p;
    bool hit = false;
    
    #ifdef MOVIE

    for( int j=0; j < 1000; j++ )
    #else
    for( int j=0; j < 200; j++ )
    #endif
	{
		if (t >= FAR) break;
		p = rO + t*rD;

		float h = map(p, t*0.002);
 		if(h < 0.01)
		{
            dist = vec2(oldT, t);
            break;
	     }
        oldT = t;
        #ifdef MOVIE
        t += h * .2;
        #else
        t += h * .35 + t*.001;
        #endif
	}
    if (t < FAR) 
    {
       t = BinarySubdivision(rO, rD, dist);
    }
    return t;
}

// Function 401
vec4 RayMarch(Ray initialRay)
{
	if (TerrainMiss(initialRay))
	{
		return SKY;
	}
	
	// raycast directly to MAX_H if above MAX_H and casting downwards
	if (initialRay.origin.z > MAX_H && initialRay.direction.z < 0.0)
	{
		initialRay = CastRay(initialRay, (initialRay.origin.z - MAX_H) / abs(initialRay.direction.z));
	}
	
	float delt = MAX_H / MARCHES / abs(initialRay.direction.z);
	
	for(float t = 0.0; t <= MARCHES;  t++)
	{				
		float dist = delt * t;
		Ray ray = CastRay(initialRay, dist);
		
		// We marched our way right out of the terrain bounds...
		if (TerrainMiss(ray))
		{
			return SKY;
		}
		
        float depthBelowTerrain = MixedTerrainHeight(ray.origin) - ray.origin.z;
        
		if (depthBelowTerrain >= 0.1)
		{
            // ray backtracking
            ray = CastRay(initialRay, dist - (depthBelowTerrain / ray.direction.z));
            
			return MixedTerrainColor(ray.origin);
		}
	}
	
	return RED;
}

// Function 402
vec3 rayMarch(ray marcher){
    float epsilon = 0.001;
    float t = 0., d, d2;
    vec3 targetSphere = sphere1Pos;
    float targetRad = sphere1Rad;
    vec3 point;
    
    for (float i = 0.; i < maxIterations; i++){
        point = marcher.position + marcher.direction * t;
        
        d = distFunc(point);
        if (d < epsilon){
            // Calc phong illumination
            vec3 normal = getNormal(point);
            vec3 light1Dir = normalize(lightPos - point);
            vec3 viewDir = normalize(eye - point);
            vec3 reflection1 = reflect(light1Dir, normal);
            // Add the ambient component
            float Ip = test.amb;
            // Add the diffuse component
            Ip += max(0., test.diff * dot(light1Dir, normal));
            // Add the specular component
            Ip += max(0., pow(test.spec * dot(reflection1, viewDir), test.shiny));
            return Ip * vec3(.816, 0.26, .96); //getNormal(point);
        }
        
        d2 = faceFunc(point);
        if (d2 < epsilon){
            // Calc phong illumination
            vec3 normal = getNormalFace(point);
            vec3 light1Dir = normalize(lightPos - point);
            vec3 viewDir = normalize(eye - point);
            vec3 reflection1 = reflect(light1Dir, normal);
            // Add the ambient component
            float Ip = test.amb;
            // Add the diffuse component
            Ip += max(0., test.diff * dot(light1Dir, normal));
            // Add the specular component
            Ip += max(0., pow(test.spec * dot(reflection1, viewDir), test.shiny));
            return Ip * vec3(1.);
        }
        
        t+=min(d, d2);
    }
    return vec3(0.);
}

// Function 403
vec4 rayMarch(vec3 p, vec3 d, inout object co)
{
    if(co.id < 0)
    {
        co = getObject(0);
    }
    
    float td = 0.; float DE = 1e10;
    for(int i = min(0, iFrame); i < maxs; i++)
    {
        //march
        DE = map(p, co);
        
        p += DE*d;
        td += DE;
        
        //outide of the scene
        if(td > maxd) return vec4(p, -1.);
        //has hit the surface
        if(DE < mind*td) break;
    }
    return vec4(p, DE);
}

// Function 404
void Step (inout vec4 s)
{
  vec4 k1, k2, k3, k4;
  k1 = EvalRhs (s);
  k2 = EvalRhs (s + k1 / 2.);
  k3 = EvalRhs (s + k2 / 2.);
  k4 = EvalRhs (s + k3);
  s += (k1 + k4) / 6. + (k2 + k3) / 3.;
  s.xy = mod (s.xy + pi, 2. * pi) - pi;
}

// Function 405
vec4 cloud_march(Ray ry, in vec3 bg_clr)
{
	vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
	float t = 0.0;
    
    // point to the sun so we can shade the portion of the clouds facing the sun as it moves across the sky
    vec3 int_pt, n;
    int_plane(ry, cloud_plane, int_pt, n);
    vec3 sun_dir = normalize(sun_light.pos - int_pt);

    
	// march through the point cloud of data to make clouds
    for(int i=0; i<CLOUD_RAY_MARCH_STEPS; ++i)
    {
        vec3 pos = ry.pos + (t * ry.dir);
        float den = cloud_map(pos);
        // Clouds are naturally fluffy, and itegration will smooth the clouds
        // out. The higher this threshold, the more jagged the look and the clouds
        // start to look like procedurally generated mountains, more than they do hills
        if (den > 0.25)
        {
            float diff = clamp((den - cloud_map(pos + (0.3 * sun_dir))) / 0.2, 0.0, 1.0 );
            sum = integrate(sum, diff, den, bg_clr, t);
        }
        t += max(0.05, 0.02 * t);
    }
    
    return clamp(sum, 0.0, 1.0);
}

// Function 406
float rayMarch(vec3 origin, vec3 dir)
{
    float dist = 0.0;
    for (int i =0; i < 256; ++i)
    {
        vec3 pos = origin + dir*dist;
        float h = scene(pos);
        if (h < 0.001) break;
        dist += h;
    }
    return dist;
}

// Function 407
void raymarching( out vec4 fragColor, in vec3 prp, in vec3 scp){

  //Raymarching
  const vec3 e=vec3(0.1,0,0);
  float maxd=48.0; //Max depth

  vec2 s=vec2(0.1,0.0);
  vec3 c,p,n;

  float f=1.0;
  for(int i=0;i<128;i++){
    if (abs(s.x)<.001||f>maxd) break;
    f+=s.x;
    p=prp+scp*f;
    s=inObj(p);
  }
  
  if (f<maxd){
    if (s.y==0.0)
      c=obj0_c(p);
    else if (s.y==1.0)
      c=obj1_c(p);
    else
      c=obj2_c(p);
 
    n=objNormal(p);
    float b=abs(dot(n,normalize(prp-p)));
    vec3 objColor=b*c+pow(b,8.0);
  
    //reflect
    float f0=f;
    if (s.y==2.0){
      prp=p-0.02*scp;
      scp=reflect(scp,n);
      f=0.0;
      s=vec2(0.1,0.0);
      maxd=16.0;
      for(int i=0;i<64;i++){
        if (abs(s.x)<.01||f>maxd) break;
        f+=s.x;
        p=prp+scp*f;
        s=inObj(p);
      }
      if (f<maxd){
        if (s.y==0.0)
          c=obj0_c(p);
        else if (s.y==1.0)
          c=obj1_c(p);
        else
          c=obj2_c(p);
      
        n=objNormal(p);
        b=abs(dot(n,normalize(prp-p)));
        vec3 objColor2=b*c+pow(b,8.0);
        fragColor=vec4((objColor*0.8+objColor2*0.2)*(1.0-f0*.03),1.0);
      } else fragColor=vec4(objColor*0.8*(1.0-f0*.03),1.0);

    } else fragColor=vec4(objColor*(1.0-f0*.03),1.0);
    
  } else fragColor=vec4(0,0,0,1);

}

// Function 408
vec3 Raymarche(vec3 org, vec3 dir, int step)
{
	float d=0.0;
	vec3 p=org;
	
	for(int i=0; i<64; i++)
	{
		d = scene(p);
		p += d * dir;
	}
	
	return p;
}

// Function 409
float march(vec2 ro, vec2 rd) {
	float t = 0.;
    for(float i=0.; i < MaxSteps; i++) {
		vec2 p = ro + t * rd;
        float dt = scene(p);
        if(dt < MinDistance) return t+0.00001;
        t += dt;
    }       
    return 0.;
}

// Function 410
vec2 raymarchUnderWater(vec3 ro, vec3 rd, in float tmin, in float tmax) {
    vec2 m = vec2(-1.0, -1.0);
    vec2 res = vec2(tmin, -1.0);
    res.x = tmin;
	for( int i=0; i<REFR_NUM_STEPS; i++ )
	{
        m = mapUnderWater(ro + res.x*rd);
		if( m.x<tmin || res.x>tmax ) break;
		res.x += /*0.5**/m.x*log(1.0+float(i));
        res.y = m.y;
	}
    if( res.x>tmax ) res.y=-1.0;
	return res;
}

// Function 411
float inverse_smoothstep( float x )
{
    return 0.5 - sin(asin(1.0-2.0*x)/3.0);
}

// Function 412
float RayMarch(vec3 rayOrigin, vec3 rayDirection)
{
    float distanceFromOrigin = 0.;
    for(int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = rayOrigin + distanceFromOrigin * rayDirection;
        float distanceToScene = GetDistanceToNearestSurface(p);
        distanceFromOrigin += distanceToScene;
        bool foundSurface = distanceToScene < SURFACE_DIST;
        bool exceededMax = distanceFromOrigin > MAX_DIST;
        if (foundSurface || exceededMax)
            break;
    }
    return distanceFromOrigin;
}

// Function 413
vec3 march (vec3 p, vec3 d) {
    float compoundedD = 0.;
    float rxcount = 0.;
    float rxindex = .3;
    vec3 finalcol = vec3(0.);
    for (float i=0.; i<60.; ++i) {
        float SDFp = SDF(p);
        float DE = SDFp;
        DE *= .9;
        if (SDFp < 1e-2) {
            p = p+d*SDFp*.999;
            int idSDFp = idSDF(p);
            vec3 TEXpd = TEX(p, d);
            if (idSDFp == 0) {
                finalcol = TEX(p, d);
                d = reflect(d, dSDF(p));
                p = p+d*.1;
                ++rxcount;
                continue;
            }
            if (rxcount == 0.) {
                return TEX(p, d);
            }
            return finalcol*(1.-rxindex)+rxindex*TEX(p, d);
        }
        p = p+d*DE;
        compoundedD += DE;
        if (compoundedD > 20. || SDFp > 7.) {
            break;
        }
    }
    if (rxcount > 0.) {
        finalcol = (1.-rxindex)*finalcol+rxindex*skycol(d);
        return finalcol;
    }
    return skycol(d);
}

// Function 414
vec4 marchRay(vec3 ro, vec3 rd) {

    float d = 10.0; //distance marched
    vec4 pc = vec4(0.); //pixel colour

    for (int i = 0; i < MAXIMUM_STEPS; ++i) {
        vec3 rp = ro + rd * d;
        
        
        float ns = nearestSurface(rp);
        d += ns;
        
        if (ns < DISTANCE_THRESHOLD) {
            vec3 sunPos = vec3(sin(iTime)*-32., 12., cos(iTime)*-32.);
            vec3 norm = normal(rp);
            
            float diffuse = dot(normalize(sunPos), norm);
            
            vec3 reflection = reflect(normalize(sunPos), normalize(norm));
            float specularAngle = max(0.0, dot(reflection, vec3(0,0,1.)));
            vec4 illuminationSpecular = clamp(pow(specularAngle, 0.01), 0., 0.01) * vec4(1.);
            
            vec4 clr = vec4(diffuse) * 0.2;
            pc = BASE_COLOR + clr + illuminationSpecular;
            break;
        }
        
        if (d > FAR_CLIP) {
            break;
        }
    }

    return pc;
}

// Function 415
float aastep(float threshold, float value) 
{
    float afwidth = length(
        vec2(dFdx(value), dFdy(value))) * 0.70710678118654757;
    return smoothstep(
        threshold - afwidth, threshold+afwidth, value); 
}

// Function 416
float sinstep( float x )
{
    return (sin(TWO_PI * x-PI) + (TWO_PI * x - PI) + PI)/TWO_PI;
}

// Function 417
RayHit March(in vec3 origin, in vec3 direction, float maxDist, float precis, int maxSteps)
{
  RayHit result;

  float t = 0.0, dist = 0.0, distStep = 1.0;
  vec3 rayPos =vec3(0);

  for ( int i=0; i<maxSteps; i++ )
  {
    rayPos =origin+direction*t;
    dist = Map( rayPos);

    if (abs(dist)<precis || t>maxDist )
    {    
        result.hitID =10;
        
        if(d == dist){  result.hitID =1;}
        else if(d2 == dist){  result.hitID =2;}
        else if(d3 == dist){  result.hitID =3;}
        else if(d4 == dist){  result.hitID =4;}
        else if(d5 == dist){  result.hitID =5;}
        else if(d6 == dist){  result.hitID =6;}
        else if(d7 == dist){  result.hitID =7;}
        else if(d8 == dist){  result.hitID =8;}
        else if(d9 == dist){  result.hitID =9;}
   
      result.depth = t; 
      result.dist = dist;                              
      result.hitPos = origin+((direction*t)*0.99);   
      result.steps = float(i);
      break;
    }
    t += dist*distStep;
  }


  return result;
}

// Function 418
RayHit RaymarchScene(in Ray ray)
{
    RayHit hit;
    
    hit.hit      = false;
    hit.material = 0.0;
    
    float sdf = FarClip;
    
    for(float depth = NearClip; depth < FarClip; )
    {
    	vec3 pos = ray.origin + (ray.direction * depth);
        
        sdf = Scene_SDF(pos, hit);
        
        if(sdf < Epsilon)
        {
            hit.hit      = true;
            hit.surfPos  = pos;
            hit.surfNorm = Scene_Normal(pos);
            
            return hit;
        }
        
        depth += sdf;
    }
    
    return hit;
}

// Function 419
bool rayMarchTrans(vec3 startPos, vec3 direction, out float rayDist) {
    vec3 position = startPos ;
    bool intersected = false ;
    rayDist = 0.0 ;
    float delta = minPrimStepSize ;
    float precis = 0.0005 ;
    
    for (int i = 0 ; i < primNumSamples ; ++i) {
		if (isIntersectingSmokeShape(position,precis,delta)) {
            return true ;
        } else {
            precis = 0.00005 * rayDist ;
		    rayDist += delta ;
            position = (rayDist)*direction + startPos ;
        }
    }
    
    return false ;
}

// Function 420
vec2 rayMarch(Ray ray)
{
	float dist = 0.;
    vec2 result = vec2(-1.);
    for(int i = 0; i < 128; ++i)
    {  
        result = sdScene(ray.origin + ray.direction * dist);
        if (result.x < EPS * dist || dist >= cameraFar) break;
        dist += result.x;
    }

    if (dist >= cameraFar) result.y = -1.;
    return vec2(dist, result.y);
}

// Function 421
Hit march(vec3 eye, vec3 marchingDirection) {
	const float precis = 0.001;
    float t = 0.0;
	float l = 0.0;
    for(int i=0; i<MAX_MARCHING_STEPS; i++){
	    Hit hit = world( eye + marchingDirection * t );
        if( hit.dist < precis ) return Hit(t, hit.matID, hit.normal);
        t += hit.dist * .5;
    }
    return miss;
}

// Function 422
void march(in state s, in vec3 pRay, in vec3 nvRayIn, out vec4 color, out vec3 nvRayOut)
{
    bool skipOpacity = texelFetch(KEY_SAMPLER, ivec2(KEY_A,0), 0).x > 0.5;
    bool skipRefraction = texelFetch(KEY_SAMPLER, ivec2(KEY_S,0), 0).x > 0.5;
    bool debugSteps = texelFetch(KEY_SAMPLER, ivec2(KEY_D,0), 0).x > 0.5;
    bool debugNormal = texelFetch(KEY_SAMPLER, ivec2(KEY_F,0), 0).x > 0.5;

    // Light (in world coordinates)
    vec3 pLightO = pRay + vec3(0.0, 10.0, 0.0);

    // Light and camera (in object coordinates)
    mat3 prMatInv = qToMat(qConj(s.pr));
    vec3 pCam = prMatInv*(pRay - s.p) + s.p;
    vec3 pLight = prMatInv*(pLightO - s.p) + s.p;

    // Ray while marching (in object coordinates)
    vec3 pCur = pCam;
    vec3 nvRayCur = prMatInv*nvRayIn;

    color = vec4(0.0);
    int curSubstance = SUBSTANCE_AIR;

    int i=0;
    for (; i<STEPS; i++) {

        // Quick exits
        // ----------------
        vec3 centerToCur = pCur - s.p;
        if (
            (length(centerToCur) > BOUNDING_SPHERE_RADIUS) &&
            (dot(nvRayCur, centerToCur) > 0.0)
        ) { break; }

        if (color.a > 0.95) { break; }
		// ----------------

        float sdGlass = sdfGlass(pCur, s);
        float sdWater = sdfWater(pCur, s);
        vec3 dpStep = abs(min(sdGlass, sdWater))*nvRayCur;

        vec3 nvGlass = SDF_NORMAL(sdfGlass, pCur, s);
        vec3 nvWater = SDF_NORMAL(sdfWater, pCur, s);

        if (curSubstance == SUBSTANCE_AIR) {

            if (sdGlass < SDF_EPS && dot(nvGlass,nvRayCur) < 0.0) {

                curSubstance = SUBSTANCE_GLASS;

                vec4 sColor = computeSpecular(
                    0.8, 80.0, nvGlass, normalize(pLight-pCur), normalize(pCam-pCur)
                );
                color = blendOnto(color, sColor);

                // Schlick approximation
                float cosHitAngle = clamp(dot(nvGlass, -nvRayCur), 0.0, 1.0);
                float r0 = pow((IR_GLASS-IR_AIR)/(IR_GLASS+IR_AIR), 2.0);
                float valRefl = mix(r0, 1.0, pow(clamp(1.0 - cosHitAngle, 0.0, 1.0), 3.0)); // Modified exponent 5 -> 3

                vec3 nvRefl = reflect(nvRayCur, nvGlass);
                color = blendOnto(color, valRefl*vec4(SKYBOX(nvRefl), 1.0));

                dpStep = sdGlass*nvRayCur;
                dpStep += -DSTEP_ADJUST_EPS*nvGlass;
                if (!skipRefraction) {
                    nvRayCur = refractFix(nvRayCur, nvGlass, IR_AIR/IR_GLASS);
                }

            } else if (sdWater < SDF_EPS && dot(nvWater,nvRayCur) < 0.0) {

                curSubstance = SUBSTANCE_WATER;

                vec4 sColor = computeSpecular(
                    1.0, 40.0, nvWater, normalize(pLight-pCur), normalize(pCam-pCur)
                );
                color = blendOnto(color, sColor);

                // Schlick approximation
                float cosHitAngle = clamp(dot(nvWater, -nvRayCur), 0.0, 1.0);
                float r0 = pow((IR_WATER-IR_AIR)/(IR_WATER+IR_AIR), 2.0);
                float valRefl = mix(r0, 1.0, pow(clamp(1.0 - cosHitAngle, 0.0, 1.0), 5.0));

                vec3 nvRefl = reflect(nvRayCur, nvWater);
                color = blendOnto(color, valRefl*vec4(SKYBOX(nvRefl), 1.0));

                dpStep = sdWater*nvRayCur;
                dpStep += -DSTEP_ADJUST_EPS*nvWater;
                if (!skipRefraction) {
                    nvRayCur = refractFix(nvRayCur, nvWater, IR_AIR/IR_WATER);
                }

            }

        } else if (curSubstance == SUBSTANCE_GLASS) {

            float sdGlassInv = -sdGlass;
            vec3 nvGlassInv = -nvGlass;

            dpStep = abs(sdGlassInv)*nvRayCur;

            if (!skipOpacity) {
                color = blendOnto(color, clamp(GLASS_OPACITY*sdGlassInv,0.0,1.0)*vec4(GLASS_COLOR, 1.0));
            }

            if (sdGlassInv < SDF_EPS && dot(nvGlassInv,nvRayCur) < 0.0) {

                curSubstance = SUBSTANCE_AIR;

                dpStep = sdGlassInv*nvRayCur;
                dpStep += -DSTEP_ADJUST_EPS*nvGlassInv;
                if (!skipRefraction) {
                    nvRayCur = refractFix(nvRayCur, nvGlassInv, IR_GLASS/IR_AIR);
                }

            }

        } else if (curSubstance == SUBSTANCE_WATER) {

            float sdWaterInv = -sdWater;
            vec3 nvWaterInv = -nvWater;

            dpStep = abs(sdWaterInv)*nvRayCur;

            if (!skipOpacity) {
                color = blendOnto(color, clamp(WATER_OPACITY*sdWaterInv,0.0,1.0)*vec4(WATER_COLOR, 1.0));
            }

            if (sdWaterInv < SDF_EPS && dot(nvWaterInv,nvRayCur) < 0.0) {

                curSubstance = SUBSTANCE_AIR;

                dpStep = sdWaterInv*nvRayCur;
                dpStep += -DSTEP_ADJUST_EPS*nvWaterInv;
                if (!skipRefraction) {
                    nvRayCur = refractFix(nvRayCur, nvWaterInv, IR_WATER/IR_AIR);
                }

            }

        }

        pCur += dpStep;

    }

    // Convert ray direction from object to world coordinates
    nvRayOut = qToMat(s.pr)*nvRayCur;

    if (debugSteps) {
        color = vec4( vec3(float(i)/float(STEPS)), 1.0 );
    } else if (debugNormal) {
        color = vec4( 0.5 + 0.5*nvRayOut, 1.0 );
    }
}

// Function 423
float linstep(float a,float b,float x)
{
    return clamp((x-a)/(b-a),0.,1.);
}

// Function 424
float wstep(float w, float thr, float x)
{
    return smoothstep(thr-w*.5,thr+w*.5,x);
}

// Function 425
marchRes march(in vec3 rayStart, in vec3 rayDir) {
    
    // first step
    float t = 100.0*EPSILON;//0.05;

    // Far Clipping
    float tmax= 50.0;
    
    Material obj = Material(vec3(0.), vec3(0.), MAT_NOTHING);
    
    vec3 p;
    
    for (int i = 0; i < 256; i++) {
        // map on Distance Field
        p = rayStart + t*rayDir;
        dfObject res = map(p);
        

        // If we hit something, exit
        if (abs(res.d) < EPSILON)
        	break;
       	
        t += abs(res.d)*1.0;
        obj = res.mat;
    }
    
    return marchRes(p, obj);
    
}

// Function 426
float sstep(float x, float p)
{
	x = clamp(x, 0.0, 1.0);
	float ix = 1.0 - x;
    x = pow(x, p);
    ix = pow(ix, p);
    return x / (x + ix);
}

// Function 427
float march(in vec3 ro, in vec3 rd) {
  float maxd = 25.0;
  const float precis = 0.001;
  float t = 0.25;
  float res = 1e8;
  for (int i = 0; i < 200; i++) {
    vec3 p = ro + rd * t;
    float d = eval(p);
    //d = max(d,-p.z);
    //d = max(d,length(p)-2.0);
    d *= 0.5;       // Fudge factor
    d = min(0.5,d); // Try and avoid overstepping
    t += d;
    if (d < precis) return t;
    if (t > maxd) break;
  }
  return 1e8;
}

// Function 428
float March(in vec3 P, in vec3 D)
{
    float t = 0.01;
    float m = 0.0;
    for (int i = 0; i < 72; ++i) {
        float d = Map(P + D*t);
        if (d <= 0.008) {
            break;
        }
        t += d + 0.004;
    }
    return t;
}

// Function 429
float sstep(float t) {
	return sin(t * PI - PI / 2.) * .5 + .5;
}

// Function 430
void march(inout vec3 pos, vec3 dir)
{
    float eps=.001;
    for(int i=0;i<80;i++)
    {
        float d=dist(pos);
        pos+=dir*d*.7;
        if (d<eps) break;
    }
}

// Function 431
vec3 raymarche( in vec3 ro, in vec3 rd, in vec2 nfplane )
{
	vec3 p = ro+rd*nfplane.x;
	float t = 0.;
	for(int i=0; i<64; i++)
	{
        float d = map(p);
        t += d;
        p += rd*d;
		if( d < 0.001 || t > nfplane.y )
            break;
            
	}
	
	return p;
}

// Function 432
RayHit March( vec3 origin, vec3 direction, float maxDist)
{
  RayHit result;
  float t = 0.0, dist = 0.0, glassDist=100000.0;
  vec3 rayPos = vec3(0.);
    float td=0.;
  float precis=.0;
  for ( int i=0; i<NO_UNROLL(120); i++ )
  {
    rayPos =origin+direction*t;
    dist = Map( rayPos);
    #ifdef HIGH_QUALITY
    if(glassDist>0.05)
    { 
      glassDist = min(glassDist, MapGlass(rayPos));
    }
    #else
    glassDist =MapGlass(rayPos);
    dist=min(dist,glassDist); 
    #endif
    precis = 0.001*t;
    if (dist<precis || t>maxDist )
    {
      result.hit=!(t>maxDist);
      result.depth = t; 
      result.dist = dist;                              
      result.hitPos = origin+direction*(t-td);   
      result.winDist = winDist;
      result.dekoDist = dekoDist;
      result.glassDist = glassDist;
      result.steelDist = steelDist;
      result.lampDist = lampDist;
      result.doorDist = doorDist;
      break;
    }
    td= dist*0.65;
      t+=td;
  }    


  return result;
}

// Function 433
float march(in vec3 ro, in vec3 rd)
{
	const float maxd = 5.0;
	const float precis = 0.001;
    float h = precis * 2.0;
    float t = 0.0;
	float res = -1.0;
    for(int i = 0; i < 100; i++)
    {
        if(h < precis || t > maxd) break;
	    h =min( map(ro + rd * t),0.1); // rewrite 2018/01/10
        t += h;
    }
    if(t < maxd) res = t;
    return res;
}

// Function 434
raymarchResult raymarch(vec3 at, vec3 normal) {
    for(int iteration = 0; iteration < 128; iteration++) {
        shapeResult sam = scene(at);
        if(sam.distance < 0.1)
            return raymarchResult(at, sam.materialId);
        at += normal * sam.distance * 0.5;
    }
    return raymarchResult(vec3(0.0), MATERIAL_SKY);
}

// Function 435
float march(vec3 eye, vec3 dir) {
   float depth = 0.0f;
   for (int s = 0; s < MAXSTEP; ++s) {
      float dist = sdf(eye + depth * dir);
      if (dist < EPSILON) {
         return depth;
      }
      depth += dist;
   }
   return 0.0;
}

// Function 436
void Step (int mId, out vec3 rm, out vec3 vm, out vec4 qm, out vec3 wm)
{
  mat3 mRot, mRotN;
  vec3 rmN, vmN, wmN, dr, dv, rts, rtsN, rms, vms, fc, am, wam;
  vec3 rMom;
  float farSep, rSep, grav, fDamp, fAttr, dt;
  const vec2 e = vec2 (0.1, 0.);
  grav = 5.;
  fDamp = 0.1;
  fAttr = 0.1;
  dt = 0.01;
  rm = Loadv4 (4 * mId).xyz;
  vm = Loadv4 (4 * mId + 1).xyz;
  qm = Loadv4 (4 * mId + 2);
  wm = Loadv4 (4 * mId + 3).xyz;
  mRot = QtToRMat (qm);
  farSep = sphGap * nSphObj / 3. + 2.;
  am = vec3 (0.);
  wam = vec3 (0.);
  for (int n = 0; n < nObj; n ++) {
    rmN = Loadv4 (4 * n).xyz;
    if (n != mId && length (rm - rmN) < farSep) {
      vmN = Loadv4 (4 * n + 1).xyz;
      mRotN = QtToRMat (Loadv4 (4 * n + 2));
      wmN = Loadv4 (4 * n + 3).xyz;
      for (float j = 0.; j < nSphObj; j ++) {
        rts = mRot * RSph (j);
        rms = rm + rts;
        vms = vm + cross (wm, rts);
        dv = vms - vmN;
        fc = vec3 (0.);
        for (float jN = 0.; jN < nSphObj; jN ++) {
          rtsN = mRotN * RSph (jN);
          dr = rms - (rmN + rtsN);
          rSep = length (dr);
          if (rSep < 1.) fc += FcFun (dr, rSep, dv - cross (wmN, rtsN));
        }
        am += fc;
        wam += cross (rts, fc);
      }
    }
  }
  rMom = vec3 (0.);
  for (float j = 0.; j < nSphObj; j ++) {
    rts = RSph (j);
    rMom += dot (rts, rts) - rts * rts + 1./6.;
    rts = mRot * rts;
    dr = rm + rts;
    dr.xz = -0.55 * GrndNf (dr).xz;
    dr.y += 0.55 - GrndHt (rm.xz - dr.xz);
    rSep = length (dr);
    if (rSep < 1.) {
      fc = FcFun (dr, rSep, vm + cross (wm, rts));
      am += fc;
      wam += cross (rts, fc);
    }
  }
  rMom /= nSphObj;
  am -= fDamp * vm;
  am.y -= grav;
  am += fAttr * (rLead - rm);
  wam = mRot * (wam * mRot / rMom);
  vm += dt * am;
  rm += dt * vm;
  wm += dt * wam;
  qm = normalize (QtMul (RMatToQt (LpStepMat (0.5 * dt * wm)), qm));
}

// Function 437
vec4 raymarch(vec3 campos, vec3 raydir, vec4 o1) {
	float totaldist = 0.0;
	vec3 pos = campos;
	float dist = EPS;
	
	for (int i = 0; i < MAX_ITER; i++) {
    	if (dist < EPS || totaldist > MAX_DIST)
        	break;
	
    	dist = distfunc(pos, o1);
	    totaldist += dist;
    	pos += dist * raydir;
    }
    
    return vec4(pos, dist);
}

// Function 438
bool gridStep(inout vec3 pos, inout vec3 n, vec3 grid, vec3 dir)
{
    float l,lmin=10000.;
    vec3 s = sign(dir);
    // find next nearest cube border (.00001 -> step a tiny bit into next cube)
    vec3 next=floor(pos/grid+s*(.5+.00001)+.5)*grid; // assuming floor(x+1.)==ceil(x)
    l=(next.x-pos.x)/dir.x; if (l>0. && l<lmin) { lmin=l; n=-vec3(1,0,0)*s; }
    l=(next.y-pos.y)/dir.y; if (l>0. && l<lmin) { lmin=l; n=-vec3(0,1,0)*s; }
    l=(next.z-pos.z)/dir.z; if (l>0. && l<lmin) { lmin=l; n=-vec3(0,0,1)*s; }
    
    pos+=dir*lmin;
    return checkSolid((floor((pos-.5*n*grid)/grid)+.5)*grid);
}

// Function 439
float RayMarch(vec3 ro, vec3 rd){
    float dO = 0.;
    for (int i =0; i<MAX_STEPS; i++){
        vec3 p = ro+dO*rd;
        float dS = GetDist(p);
        dO +=dS;
        if (dS<SURF_DIST|| dO>MAX_DIST) break;
    }
    return dO;
}

// Function 440
float calculateSteps(float orig, float dist)
{
    // Plane hit: xy = [0, 0.5] modulo range on each integer z
    if (orig < 0.5) return 0.0;
    
    // How many steps to hit (modulo) 0.0 or 1.0?
    float steps = (dist < 0.0) ? 
        (orig - 0.5) / -dist :
    	(1.0 - orig) / dist;
    
    return ceil(steps);
}

// Function 441
float vonronoiseStep(float i, float a, float b, vec2 x) {
    float d = 0.2*(b-a);
	return 1.0-i+(smoothstep(a-d, b+d, vonronoise(x))*(i));
}

// Function 442
vec2 smoothstepd(float e0, float e1, float x)
{
    if (x < e0)
        return vec2(0.0, 0.0);
    if (x > e1)
        return vec2(1.0, 0.0);
    
    float t = (x - e0) / (e1 - e0);
    float td = 1.0 / (e1 - e0);
    return vec2(3.0 * t * t - 2.0 * t * t * t, 6.0 * t * (1.0 - t) * td);
}

// Function 443
vec3 marchAA(in vec3 ro, in vec3 rd, in vec3 bgc, in float px)
{
    float precis = px*.01;
    float prb = precis;
    float t=map(ro);
	float dm=100.0,tm=0.0,df=100.0,tf=0.0,od=1000.0,d=0.;
	for(int i=0;i<ITR;i++)
    {
		d=map(ro+rd*t);
		if(df==100.0)
        {
			if(d>od)
            {
				if(od<px*(t-od))
                {
					df=od; tf=t-od;
                    t += .05;
				}
			}
			od=d;
		}
		if(d<dm){tm=t;dm=d;}
		t+=d;
		if(t>FAR || d<precis)break;
	}
	vec3 col=bgc;
    
	if(dm<px*tm)
        col=mix(shade((ro+rd*tm) - rd*(px*(tm-dm)) ,rd, tm),col,clamp(dm/(px*tm),0.0,1.0));
	
    float qq=0.0;
    if((df==100.0 || tm==tf) && t < FAR)
    {
        ro+=vec3(0.5,0.5,0.)*px*tm*1.;
        tf=tm;
        df=dm;
        qq=.01;
	}
    dfog = tm;
    return mix(shade((ro+rd*tf) - rd*(px*tf-df),rd, tf),col,clamp(qq+df/(px*tf),0.0,1.0));
    
}

// Function 444
float raymarchGateShadow(in vec3 ro, in vec3 rd, float tmin, float tmax) {
    float sh = 1.0;
    float t = tmin;
    float breakOut = 0.0;
    int i = 0;
    while (i < 80 && breakOut != 1.0) {
        vec3 p = ro + rd * t;
        float d = map(p, false).y;
        sh = min(sh, 16.0 * d / t);
        t += 0.5 * d;
        if (d < (0.001 * t) || t > tmax)
            breakOut = 1.0;
        i++;
    }
    return sh;
}

// Function 445
float rayMarch( in vec3 origin, in vec3 direction ) {
    float total = .0;
    for ( int i = 0 ; i < RAY_MARCH_STEPS ; i++ ) {
        vec3 point = origin + direction * total;
                
        float current = sceneDistance( point );
        total += current;
        if ( total > RAY_MARCH_TOO_FAR || abs(current) < RAY_MARCH_CLOSE ) {
            break;
        }
    }
    return total;
}

// Function 446
vec3 OpticalLengthStep(float R, float V, float Di)
{
    // Re-compute radius at distance Di using cosine theorem
    float Ri = Sqrt((Di + 2.0 * V * R) * Di + R * R);
    float Hi = Ri - kAtmRadiusMin;
    
    // Standard Rayleigh / Mie density profiles
    float RayDensity = exp(-Hi * kAtmRayHeightScale);
    float MieDensity = exp(-Hi * kAtmMieHeightScale);
    
    // Piecewise linear approximation of the ozone profile from (Page 10) :
    // ftp://es-ee.tor.ec.gc.ca/pub/ftpcm/!%20for%20Jacob/Introduction%20to%20atmospheric%20chemistry.pdf
    // Density linearly increases from 0 at 15Km to 1.0 at 25Km and decreases back to 0.0 at 40.0Km
    float OznDensity = Hi < 25.0 ? clamp( Hi / 15.0 - 2.0 / 3.0, 0.0, 1.0)
                                 : clamp(-Hi / 15.0 + 8.0 / 3.0, 0.0, 1.0);
    
    return vec3(RayDensity, MieDensity, OznDensity * RayDensity);
}

// Function 447
vec3 march(vec3 ro, vec3 rd, float x) {
	// Raymarch.
	vec3 p = ro,
		 col = vec3(0);

	float md = -1., d = .01, o = 1.;

	if (inFluid) {
		float bubbleO = 1.;
		col = BLUE;
		for (float i = Z0; i < 40.; i++) {
			p = d * rd + ro;
			Hit h = Hit(bblz(p), 1);

			if (d > 15.)
				break;

			if (abs(h.d) < .0015) {
				// Not physically accurate, but looks ok.
				d++;
				col += lights(p, rd, d, h) * .3 * bubbleO;
				bubbleO *= .25;
			}

			d += h.d; // No hit, so keep marching.
		}

		return col;
	}

	bool inGlass = false;
	float shine = 0.;
	for (float i = Z0; i < 140.; i++) {
		p = d * rd + ro;
		Hit gh, h = map(p);

		if (abs(h.d) < .0015) {
			col += (lights(p, rd, d, h) + shine) * o;
			if (md < 0.)
				md = d;
			else
				break;

			o *= h.id == 6 ? .02 : .1;
			ro = p;
			rd = reflect(rd, calcN(ro, d));
			d = .5;
		} else if (!inGlass && (gh = glass(p)).d < .0015) {
			inGlass = true;
			shine = lights(p, rd, d, gh).r;
			ro = p + gh.d * rd;
			rd = refract(rd, glassN(ro, d), .8);
			d = .5;
			continue;
		}

		d += h.d; // No hit, so keep marching.
	}

	return col;
}

// Function 448
bool marching(out vec3 pos,vec3 dir){
    for(int i = 0;i != 128;i++){
        float d = surface_distance(pos);
        if(d <= 0.) return true;
        float st = max(d * 0.1,0.02);
        pos += st * dir;
    }
    return false;
}

// Function 449
float RayMarch(vec3 ro, vec3 rd)
{
    float d0 = 0.;
    for(int i = 0; i < Max_Steps; i++)
    {
        vec3 p = ro + rd*d0;
        float ds = GetDist(p);
        d0+=ds;
        if(d0>Max_Dist || ds < Surf_Dist) 
            break;           
    }
                     
    return d0;     
}

// Function 450
vec2 marcher(vec3 ro, vec3 rd, int maxsteps) {
	float d = 0.;
    float m = -1.;
    for(int i=0;i<maxsteps;i++){
    	vec2 t = map(ro + rd * d);
        if(abs(t.x)<d*MIN_DIST||d>MAX_DIST) break;
        d += t.x*.5;
        m  = t.y;
    }
	return vec2(d,m);
}

// Function 451
vec3 stepspace(
  in vec3 p,
  in float s)
{
  return p-mod(p-s/2.0,s);
}

// Function 452
float rayMarch( in vec3 ro, in vec3 rd, float tmax, in vec3 samples123 )
{
    float t = 0.0;
    
    // bounding plane
    float h = (1.0-ro.y)/rd.y;
    if( h>0.0 ) t=h;

    // raymarch
    for( int i=0; i<20; i++ )    
    {        
        vec3 pos = ro + t*rd;
        float h = map( pos, samples123.x );
        if( h<0.001 || t>tmax ) break;
        t += h;
    }
    return t;    
}

// Function 453
float raymarchTerrain( in vec3 ro, in vec3 rd )
{
	float maxd = 50.0;
	float precis = 0.001;
    float h = 1.0;
    float t = 0.0;
    for( int i=0; i<80; i++ )
    {
        if( abs(h)<precis||t>maxd ) break;
        t += h;
	    h = mapTerrain( ro+rd*t );
    }

    if( t>maxd ) t=-1.0;
    return t;
}

// Function 454
vec3 RayMarchReflection(vec3 ro, vec3 dir) {
    float traveled = 0.0;
    vec2 distAndMaterial = vec2(0);
    
    for (int i=ZERO; i < 30; ++i){
        distAndMaterial = sceneWithMaterials(ro + dir * traveled);
        traveled += distAndMaterial.x;
        if (distAndMaterial.x < .01 || distAndMaterial.x > MAXDISTANCE) {
            break;
        }
    }
    
    vec3 hitPoint = ro + dir * traveled;
    
    vec3 color = vec3(1);
    color = GetColor(distAndMaterial, hitPoint, traveled, dir);
    return color;
}

// Function 455
vec3 march(inout vec3 rayOrigin, inout vec3 rayDir, in vec3 lightPos, in int stage, inout int isOut){
    vec3 col = vec3(0.0);
    vec3 rayPos = rayOrigin;
    float trvlDist = 0.1;
    
    for(int i = 0; i < 256; i++){
        rayPos = rayOrigin + rayDir * trvlDist;
        float dist = sdf(rayPos);
        if(dist < EP){
            vec3 norm = getNormal(rayPos);
            vec3 lightDir = normalize(lightPos - rayPos);
            //float diff = max(0.0, dot(norm, lightDir));
            float fresnel = max(0.0, 1.0 - dot(-rayDir, norm));
            fresnel = pow(fresnel, 0.2);
            
            //col = sampleRandomHemisphere(rayPos * float(i+1), norm);
            //col = norm;
            //col = texture(iChannel0, norm).xyz;
            
            rayOrigin = rayPos;
            rayDir = mix(sampleRandomHemisphere(rayPos * float(stage+1), norm), reflect(rayDir, norm), fresnel);
            //rayDir = mix(reflect(rayDir, norm), sampleRandomHemisphere(rayPos * float(stage+1), norm), fresnel);
            rayDir = normalize(rayDir);
            //rayDir = reflect(rayDir, norm);
            
            col = texture(iChannel0, rayDir).xyz * shadow(rayOrigin, rayDir);
            
            return col;
        }
        trvlDist += dist;
        if(trvlDist > 16.0){
            break;
            isOut = 1;
        }
    }
    return texture(iChannel0, rayDir).xyz;
}

// Function 456
bool raymarch(vec3 o, vec3 dir, inout float depth, inout vec3 n) {
	float t = 0.0;
    float d = 10000.0;
    float dt = 0.0;

    for (int i = 0; i < 128; i++)
    {
        vec3 v = o + dir * t;
        d = world(v);

        if (d < 0.001)
        {
            break;
        }

        dt = min(abs(d), 0.1);
        t += dt;

        if (t > depth)
        {
            break;
        }
    }
    
    if (d >= 0.001)
    {
        return false;
    }
    
    t -= dt;

    for (int i = 0; i < 4; i++)
    {
        dt *= 0.5;
        vec3 v = o + dir * (t + dt);

        if (world(v) >= 0.001)
        {
            t += dt;
        }
    }
    
    depth = t;
    n = normalize(calcNormal(o + dir * t));
    
    return true;
}

// Function 457
float marchGodRay(vec3 ro, vec3 rd, vec3 light, float hitDist) {
    // March through the scene, accumulating god rays.
    vec3 p = ro;
    vec3 st = rd * hitDist / 96.0;
    float god = 0.0;
    for (int i = 0; i < 96; i++) {
        float distFromGodLight = 1.0 - godLight(p, light);
        god += godLight(p, light);
        p += st;
    }
    
    god /= 96.0;

    return smoothstep(0.0, 1.0, min(god, 1.0));
}

// Function 458
vec2 shortestPathStepDist(vec4 uvs)
{
    vec2 stepDist = uvs.zw - uvs.xy;
    
    // Modulo range [-0.5, +0.5]
    stepDist = fract(stepDist + 0.5) - 0.5;
    return stepDist;
}

// Function 459
void SphereMarchDiagram( inout vec3 o, vec2 uv )
{
    float size = iResolution.y*.55; // don't scale uv, it will break the anti aliasing
    
    if ( abs(uv.x) > .8*size || abs(uv.y) > .27*size )
        return;
    
	vec2 a = size*vec2(-.3,.1);
    vec2 b = size*vec2(.5,.0 );
    float l = length(b-a);
    
    float tt = fract(iTime/10.)*10.;
    float t[10];
    for( int i=0; i < t.length(); i++ )
    {
        t[i] = smoothstep(.0,1.,tt-float(i));
    }
    
    vec2 p0 = a;
    vec2 p1 = mix(a,b,.5);
    vec2 p2 = mix(a,b,.74);
    vec2 p3 = mix(a,b,.92);
    vec2 p4 = mix(a,b,.98);
    
    Circle( o, p4, l*.025*t[8], vec3(1,.5,.5), uv );
    Circle( o, p4, l*.015*t[8], vec3(1,0,0), uv );
    Circle( o, p3, l*.065*t[6], vec3(1,.5,.5), uv );
    Circle( o, p3, l*.055*t[6], vec3(1,0,0), uv );
    Circle( o, p2, l*.185*t[4], vec3(1,.5,.5), uv );
    Circle( o, p2, l*.175*t[4], vec3(1,0,0), uv );
    Circle( o, p1, l*.245*t[2], vec3(1,.5,.5), uv );
    Circle( o, p1, l*.235*t[2], vec3(1,0,0), uv );
    Circle( o, p0, l*.505*t[0], vec3(1,.5,.5), uv );
    Circle( o, p0, l*.495*t[0], vec3(1,0,0), uv );
    
    vec2 arrowhead = normalize(b-a)*.3*size+a;
    Line( o, a, arrowhead, size*.01, vec3(0), uv );
    Line( o, arrowhead+vec2(-.04,.06)*size, arrowhead, size*.01, vec3(0), uv );
    Line( o, arrowhead+vec2(-.06,-.05)*size, arrowhead, size*.01, vec3(0), uv );
         
    Line( o, p0, mix(p0,p1,t[1]), size*.005, vec3(1), uv );
    Line( o, p1, mix(p1,p2,t[3]), size*.005, vec3(1), uv );
    Line( o, p2, mix(p2,p3,t[5]), size*.005, vec3(1), uv );
    Line( o, p3, mix(p3,p4,t[7]), size*.005, vec3(1), uv );
    Line( o, p4, mix(p4,b,t[7]), size*.005, vec3(1), uv );
    Circle( o, p0, size*.017, vec3(1), uv );
    Circle( o, p1, size*.017*t[1], vec3(1), uv );
    Circle( o, p2, size*.017*t[3], vec3(1), uv );
    Circle( o, p3, size*.017*t[5], vec3(1), uv );
    Circle( o, p4, size*.017*t[7], vec3(1), uv );
    Circle( o, b, size*.017*t[9], vec3(1), uv );

    Line( o, size*vec2(.7,.1), size*vec2(.43,-.08), size*.02, vec3(0), uv );
    Line( o, size*vec2(.29,-.18), size*vec2(.43,-.08), size*.02, vec3(0), uv );
    Line( o, size*vec2(.29,-.18), size*vec2(.04,-.165), size*.02, vec3(0), uv );
    Line( o, size*vec2(-.1,-.3), size*vec2(.04,-.165), size*.02, vec3(0), uv );
}

// Function 460
vec4 StepJFA (in vec2 fragCoord, in int step, bool xAxis)
{
    float level = clamp(float(step), 0.0, c_maxSteps);
    float stepwidth = floor(exp2(c_maxSteps - level)+0.5);
    
    float bestDistance = 9999.0;
    vec2 bestCoord = vec2(0.0);
    vec3 bestColor = vec3(0.0);
    
    if (xAxis)
    {
        // jfa step x axis
        for (int x = -1; x <= 1; ++x) {
            vec2 sampleCoord = fragCoord + vec2(x,0) * stepwidth;

            vec4 data = texture( iChannel0, sampleCoord / iChannelResolution[0].xy);
            vec2 seedCoord;
            vec3 seedColor;
            DecodeData(data, seedCoord, seedColor);
            float dist = length(seedCoord - fragCoord);
            if ((seedCoord.x != 0.0 || seedCoord.y != 0.0) && dist < bestDistance)
            {
                bestDistance = dist;
                bestCoord = seedCoord;
                bestColor = seedColor;
            }
        }
    }
    else
    {
        // jfa step y axis
        for (int y = -1; y <= 1; ++y) {
            vec2 sampleCoord = fragCoord + vec2(0,y) * stepwidth;

            vec4 data = texture( iChannel0, sampleCoord / iChannelResolution[0].xy);
            vec2 seedCoord;
            vec3 seedColor;
            DecodeData(data, seedCoord, seedColor);
            float dist = length(seedCoord - fragCoord);
            if ((seedCoord.x != 0.0 || seedCoord.y != 0.0) && dist < bestDistance)
            {
                bestDistance = dist;
                bestCoord = seedCoord;
                bestColor = seedColor;
            }
        }        
    }
    
    return EncodeData(bestCoord, bestColor);
}

// Function 461
vec2 raymarch(vec3 ori, vec3 dir, int iter) {
    float  t = 0.;
    float id = -1.;
    for(int i = 0; i < MAX_ITERATIONS; i++) {
        if(t >= MAX_DISTANCE || i >= iter) {
        	break;   
        }
    	vec2 scn = dstScene(ori+dir*t);
        if(scn.x < EPSILON) {
        	id = scn.y;
            break;   
        }
        t += scn.x * .75;
    }
    return vec2(t,id);
}

// Function 462
vec3 vmarch(in vec3 ro, in vec3 rd)
{   
    vec3 p = ro;
    vec2 r = vec2(0.);
    vec3 sum = vec3(0);
    float tot = 0.;
    for( int i=0; i<200; i++ )
    {
        r = map(p);
        if (r.x > .5)break;
        vec3 col = sin(vec3(1.5,2.,1.8)*r.y*1.3+0.4)*.9+0.15;
        col.rgb *= smoothstep(fz,intsfz,-r.x);
        sum += abs(col) * (1.8-noise(p.x*1.+p.z*10.+time*16.)*1.3);
        //"hybrid" step
        p += rd*max(.015, max(r.x,0.)*3.);
    }
    return clamp(sum,0.,1.);
}

// Function 463
float min_step(){return 1.25/density();}

// Function 464
vec3 ray_march(vec3 rayOrigin, vec3 rayDir) {
        float distanceTravelled = 0.0;
        const int NUMBER_OF_STEPS = 2048;
        const float MINIMUM_HIT_DISTANCE = 0.001;
        const float MAXIMUM_TRACE_DISTANCE = 1000.0;
        for(int i = 0; i < NUMBER_OF_STEPS; i++) {
            vec3 currPos = rayOrigin + distanceTravelled * rayDir;
            vec4 sceneData = scene_dist(currPos);
            float sceneDist = sceneData.x;
            vec3 sceneDiffuse = sceneData.yzw;
            if (sceneDist < MINIMUM_HIT_DISTANCE) {
                vec3 normal = calculate_normal(currPos);
                vec3 lightPos = vec3(200.0, -500.0, 150.0);
                vec3 dirToLight = normalize(currPos - lightPos);
                float lightIntensity = max(0.2, dot(normal, dirToLight));
                vec3 reflectDir = reflect(rayDir, normal);
                sceneDiffuse = ray_march_diffuse(currPos + reflectDir * 0.002, reflectDir, sceneDiffuse);
                if (ray_march_hit(currPos + dirToLight * 0.01, dirToLight)) {
                    lightIntensity = 0.15;
                }
                return ((sceneDiffuse) / 2.0) * lightIntensity + 0.2 * vec3(1.0, 1.0, 1.0);
            }
            if (sceneDist > MAXIMUM_TRACE_DISTANCE) {
                break;
            }
            distanceTravelled += sceneDist;
        }
        return Sky(rayOrigin, rayDir);
        
    }

// Function 465
vec4 raymarch(vec3 from, vec3 increment)
{
	const float maxDist = 200.0;
	const float minDist = 0.001;
	const int maxIter = RAYMARCH_ITERATIONS;
	
	float dist = 0.0;
	float material = 0.0;
	
	for(int i = 0; i < maxIter; i++) {
		vec3 pos = (from + increment * dist);
		float distEval = distf(pos, material);
		
		if (distEval < minDist) {
			break;
		}
		
		dist += distEval;
	}
	
	if (dist >= maxDist) material = 0.0;
	
	return vec4(dist, material, 0, 0);
}

// Function 466
float march(in vec3 ro, in vec3 rd, out float itrc)
{
    float t = 0.;
    float d = map(rd*t+ro);
    float precis = 0.0001;
    for (int i=0;i<=ITR;i++)
    {
        if (abs(d) < precis || t > FAR) break;
        precis = t*0.0001;
        float rl = max(t*0.02,1.);
        t += d*rl;
        d = map(rd*t+ro)*0.7;
        itrc++;
    }

    return t;
}

// Function 467
float bisectmarch(vec3 p, vec3 d, float D, in object o)
{
    p += d*D*0.005;
    float DE = 1e10; float td = 0.;
    for(int i = 0; i < 5; i++)
    {
        float sd = sdObj(p, o);
        p += sd*d;
        DE = min(sd, DE);
        td+= sd;
        if(td > D || sd < 0.001) break;
    }
    return DE;
}

// Function 468
void rayMarch()
{
    
}

// Function 469
VOX_MAR sdf_MarchIntoVoxel(
        VOX_000 vox_000
    ,   V_3     xyz
    ,   V_3     rwN
    )
    {

        VOX_MAR vox_mar;
        vox_mar.exit=U32( 1 );


        return( vox_mar );
    }

// Function 470
vec4 raymarch(vec3 eye, vec3 dir)
{
    vec3 info = vec3(0);
    float depth = 0.0, i;
    for (i=0.0; i<256.0 && depth<MAX_DIST; i++){
        vec3 p = eye + depth * dir;
        info = thing(p);
        if (abs(info.x) < EPSILON * depth)break;
        depth += info.x * remap(i,0.0,256.0,0.5,1.0);
    }
    return vec4(depth, info.yz, i/256.0);
}

// Function 471
SRayHitInfo TestSceneMarch(in vec3 rayPos)
{
    SRayHitInfo hitInfo;
    hitInfo.hitAnObject = false;
    hitInfo.dist = c_superFar;
    
    // glowing triangles
    {

        vec3 A = vec3(0.0f, 0.0f, 0.0f);
        vec3 B = vec3(1.5f, 3.0f, 0.0f);
        vec3 C = vec3(3.0f, 0.0f, 0.0f);
        float lineWidth = 0.1f;
        
        vec3 center = (A + B + C) / 3.0f;
        A -= center;
        B -= center;
        C -= center;
        
        A *= 3.0f;
        B *= 3.0f;
        C *= 3.0f;
        
        // foreground purple one
        SMaterial material;
        material.diffuse = vec3(0.0f, 0.0f, 0.0f);
        material.specular = vec3(0.0f, 0.0f, 0.0f);
        material.roughness = 0.0f;
        material.emissive = pow(vec3(0.73f, 0.06f, 0.99f), vec3(2.2f, 2.2f, 2.2f)) * 10.0f;            

        TestLineMarch(rayPos, hitInfo, A, B, lineWidth, material);
        TestLineMarch(rayPos, hitInfo, B, C, lineWidth, material);
        TestLineMarch(rayPos, hitInfo, C, A, lineWidth, material);
        
        // blue one slightly behind
        material.emissive = pow(vec3(0.3f, 0.15f, 1.0f), vec3(2.2f, 2.2f, 2.2f)) * 10.0f;
        A += vec3(0.0f, 0.0f, 5.0f);
        B += vec3(0.0f, 0.0f, 5.0f);
        C += vec3(0.0f, 0.0f, 5.0f);
        TestLineMarch(rayPos, hitInfo, A, B, lineWidth, material);
        TestLineMarch(rayPos, hitInfo, B, C, lineWidth, material);
        TestLineMarch(rayPos, hitInfo, C, A, lineWidth, material);        
        
        // red one behind more
        material.emissive = pow(vec3(1.0f, 0.15f, 0.3f), vec3(2.2f, 2.2f, 2.2f)) * 10.0f;
        A += vec3(0.0f, 0.0f, 5.0f);
        B += vec3(0.0f, 0.0f, 5.0f);
        C += vec3(0.0f, 0.0f, 5.0f);
        TestLineMarch(rayPos, hitInfo, A, B, lineWidth, material);
        TestLineMarch(rayPos, hitInfo, B, C, lineWidth, material);
        TestLineMarch(rayPos, hitInfo, C, A, lineWidth, material);              
	}    

    // a neon cactus
    {
    	SMaterial material;
        material.diffuse = vec3(0.0f, 0.0f, 0.0f);
        material.specular = vec3(0.0f, 0.0f, 0.0f);
        material.roughness = 0.0f;
        material.emissive = pow(vec3(0.73f, 0.06f, 0.99f), vec3(2.2f, 2.2f, 2.2f)) * 10.0f;
        
        vec3 cactusOffset = vec3(0.0f, 0.0f, 50.0f);
        
        // main body section
        {
            vec3 A = vec3(-40.0f, -10.0f, 0.0f) + cactusOffset;
            vec3 B = vec3(-40.0f, 5.0f, 1.0f) + cactusOffset;
            vec3 C = vec3(-40.0f, 20.0f, 0.0f) + cactusOffset;
            TestBezierMarch(rayPos, hitInfo, A, B, C, 2.0f, material);
        }
        
        // Arm going to left
        {
            vec3 A = vec3(-40.0f, 5.0f, 1.0f) + cactusOffset;
            vec3 B = vec3(-32.5f, 10.0f, 0.0f) + cactusOffset;
            vec3 C = vec3(-32.5f, 15.0f, -1.0f) + cactusOffset;
            TestBezierMarch(rayPos, hitInfo, A, B, C, 1.0f, material);
        }
        
        // Arm going to right
        {
            vec3 A = vec3(-40.0f, 2.0f, 1.0f) + cactusOffset;
            vec3 B = vec3(-47.5f, 7.0f, 2.0f) + cactusOffset;
            vec3 C = vec3(-47.5f, 13.0f, 4.0f) + cactusOffset;
            TestBezierMarch(rayPos, hitInfo, A, B, C, 1.0f, material);
        }        
        
    }

    return hitInfo;
}

// Function 472
vec4 doMarch(vec3 pos,vec3 cam){
    int i = 0;
    vec3 march = vec3(0);
    float d = 1.;
    
    for(;i<100&&d>0.02&&length(march)<100.;i++){
        d = de(pos+march);
        march += d*cam;
    }
    return vec4(march+pos,i);
}

// Function 473
float raymarch(vec3 ori, vec3 dir, int maxIter) {
 
    float d = 0.;
    for(int i = 0; i < 256; i++) {
        if(d > 1000. || i >= maxIter) {
			break;            
        }
        vec3    p = ori+dir*d;
        float dst = dstScene(p);
        d += dst * .75;
        if(dst < .001) {
            break;
        }
    }
    return d;
    
}

// Function 474
vec4 RayMarch(Ray initialRay)
{
	if (TerrainMiss(initialRay))
	{
		return SKY;
	}
	
	// raycast directly to MAX_H if above MAX_H and casting downwards
	if (initialRay.origin.z > MAX_H && initialRay.direction.z < 0.0)
	{
		initialRay = CastRay(initialRay, (initialRay.origin.z - MAX_H) / abs(initialRay.direction.z));
	}
	
	float marches = Marches();
	float delt = MAX_H / marches / abs(initialRay.direction.z);
	
	for(float t = 0.0; t <= INF;  t++)
	{
		if (t > marches)
		{
			break;
		}
				
		float dist = delt * t;
		Ray ray = CastRay(initialRay, dist);
		
		// We marched our way right out of the terrain bounds...
		if (TerrainMiss(ray))
		{
			return SKY;
		}
		
		if (ray.origin.z < TerrainHeight(ray.origin))
		{
			// todo: ray backtracing
			return TerrainColor(ray.origin);
		}
	}
	
	return RED;
}

// Function 475
void march( sRay ray, out sHit res, int maxIter, float fts ) {
    res.hd = ray.sd;
    res.oid.x = 0.0;

    for( int i=0;i<=MARCHSTEPS;i++ ) {
        res.hp = ray.ro + ray.rd * res.hd;
        vec4 r = DE( res.hp, fts );
        res.oid = r.yzw;
        if((abs(r.x) <= 0.01) || (res.hd >= ray.rl) || (i > maxIter))
            break;
        res.hd = res.hd + r.x;
    }
    if(res.hd >= ray.rl) {
        res.hd = MAX_DIST;
        res.hp = ray.ro + ray.rd * res.hd;
        res.oid.x = 0.0;
    }
}

// Function 476
float ray_marching( vec3 origin, vec3 dir, float start, float end ) {
	float depth = start;
	for ( int i = 0; i < max_iterations; i++ ) {
		float dist = dist_field( origin + dir * depth );
		if ( dist < stop_threshold ) {
			return depth;
		}
		depth += dist;
		if ( depth >= end) {
			return end;
		}
	}
	return end;
}

// Function 477
vec3 raymarch(SAMPLERTYPE sampler, vec3 ray_start, vec3 step, float max_dist)
{
	float step_dist = length(step);
	float dist = 0.0;
	vec3 color = vec3(0.0);
	vec3 sdt = vec3(1.0);

	vec3 p = ray_start;
	/*if (si.x <= 0) */{
		for (int i = 0; i < samples && dist < max_dist; i++, dist += step_dist, p += step) {

			// float r = d/2.0;
			vec3 p2 = p - spiral_origin;
			if (in_galaxy(p2)) {
				float lp2 = length(p2);

				float bulge_density = pow(max(2.0 - lp2/bulge.y, 0.0), bulge.z);
				float density = galaxy_density(sampler, p2, bulge_density, tweaks1.y, tweaks1.z);
				float clamped_density = clamp(density, 0.0, 1.0);
				float eps = tweaks2.y;
				float dif = clamp(
					(galaxy_density(sampler, p2 - eps*normalize(p2), bulge_density, tweaks1.y, tweaks1.z) - density) / eps,
					0.0, 1.0);

				sdt *= transmittance.xyz*(1.0 - clamped_density);
				// float clamped_bulge_density = clamp(bulge_density, 0.0, 1.0);
				float bulge_light_intensity = bulge.y*bulge.y * 2.0 * PI / (10.0 * lp2); //Roughly solid angle of bulge from p2's vantage point.

#ifdef BACK_TO_FRONT
				color =
					(
						color + //Color from previous steps
						mix(color_ramp(clamped_density * step_dist * 30.0), bulge_color * bulge_density, bulge_density) * tweaks2.z + //Emissivity
						bulge_light_intensity * tweaks2.x * dif*bulge_color //Diffuse lighting
					) *
					(vec3(1.0) - transmittance.xyz*clamped_density); //Attenuation
#else
				if (length(sdt) < 0.005) {
					// samples = i;
					// color = vec3(1.0, 0.0, 0.0);
					break;
				}
				color +=
					(
						mix(color_ramp(clamped_density * step_dist * 30.0), bulge_color * bulge_density, bulge_density) + //Emissivity
						bulge_light_intensity * tweaks2.x * dif*bulge_color //Diffuse lighting
					) * sdt; //Attenuation
#endif
			}
		}
	}
	return clamp(color * brightness, 0.0, 100000.0);
}

// Function 478
float linearstep(float a, float b, in float x )
{
    return clamp( (x-a)/(b-a), 0.0, 1.0 );
}

// Function 479
float aaStep(float threshold, float x)
{
    float afwidth = clamp(length(vec2(dFdx(x), dFdy(x))) * 0.70710678118654757, 0. ,0.05);
    return smoothstep(threshold-afwidth, threshold+afwidth, x);
}

// Function 480
float rayMarching(vec3 ro, vec3 rd) {
    float d = 0.;
    for(int i = 0; i < 50; i++) {
        float dScene = getDistance(ro + rd * d);
        if( d > MAX_DIST || dScene < SURF_DIST*0.99) break;
        d += dScene;
    }
    return d;
}

// Function 481
vec2
RayMarch(vec3 ro, vec3 rd)
{
    vec2 res = vec2(-1.0, -1.0);
    float t = 0.00;
    vec2 hit;

    for(int i =0 ; i < MAX_STEPS && t < MAX_DIST; ++i)
    {
        hit = Map(ro + t *rd);
        if(abs((hit.x)) < t * MIN_DIST)
        {
            res = vec2(t, hit.y);
            break;
        }

        t += hit.x;
    }
    
    return res;
}

// Function 482
vec3 col_step_flame( float d, vec2 uv )
{    
    vec2 sunpos = vec2(.0,-.45);
    vec3 hot = vec3(1.,1.,1.);
    vec3 orange = vec3(1.,.65,0.);
    vec3 yellow = vec3(1.,1.,0.);
    vec3 red = vec3(1.,0.,0.);
    
    float ds = length(uv-sunpos);
    float f = clamp(pow(1.3-ds,3.),.0,1.);

    float x = .75 * ds;
    float mp = .3;
    vec3 suncol =
    	(1.-step(mp,x))*mix(hot, yellow, x/mp) +
    	step(mp,x)*mix(yellow, red, .5*(x-mp))
        ;
    //suncol = mix(orange, hot, f);
    
    // border blend
    float bw = .05;
    vec3 bcol = vec3(0.);
    vec3 outcol = vec3(0.5);
    vec3 col =
        vec3
        (
        mix(suncol,bcol,smoothstep(-0.01,0.01,d))
      + mix(bcol,outcol,smoothstep(bw,2.*bw,d))
        );

    return col;
}

// Function 483
float RayMarch (vec3 or, vec3 dir){
    for(float i=0.;i<renderDist;){
    
    vec3 pos= i*dir+or;
    float DE=DE(pos);
    if(DE<SUR_ACC){
    return i;
    
    }else{
    i+=DE*0.9;
    }
    }
    return renderDist;
}

// Function 484
float march (in vec3 p) {
    vec2 h = vec2(0.0);
	for (int i = 0; i < 32; i++) {
        h = interiorMap(p);
        if (h.x < 0.001) break;
        p = p + h.x * randomOnSphere();
    }
    return h.y;
}

// Function 485
float ray_marching( vec3 origin, vec3 dir, float start, float end ) {
	const int max_iterations = 255;
	const float stop_threshold = 0.001;
	float depth = start;
	for ( int i = 0; i < max_iterations; i++ ) {
		float dist = DE_atlogo( origin + dir * depth );
		if ( dist < stop_threshold ) {
			return depth;
		}
		depth += dist;
		if ( depth >= end) {
			return end;
		}
	}
	return end;
}

// Function 486
vec2 aaa_step2( vec2 K, vec2 u )
	{ return vec2( aaa_step( K.x, u.x ), aaa_step( K.y, u.y ) ); }

// Function 487
vec4 raymarch(vec3 raydirection, vec3 rayorigin){
    vec3 rayposition = rayorigin;
    float distanceestimate;
    float distancetravelled = 0.0;
    bool hit = false;
    for(int i = 0; i < maxmarches; i++){
        distanceestimate = distanceestimator(rayposition);
        if(distanceestimate < collisiondistance){hit = true; break;}
        rayposition += raydirection*distanceestimate;
        distancetravelled += distanceestimate;
        if(distancetravelled > maxdistance){break;}
    }
    if(hit){return vec4(rayposition, distancetravelled);}
    else{return vec4(-1.0);}
}

// Function 488
float march(in vec3 ro, in vec3 rd) {
  const float maxd = 30.0;
  const float precis = 0.0001;
  float h = precis * 2.0;
  float t = 0.0;
  float res = -1.0;
  for(int i = 0; i < 64; i++) {
      if (h < precis || t > maxd) break;
      h = map(ro + rd * t);
      t += h;
    }
  if (t < maxd) res = t;
  return res;
}

// Function 489
vec4 raymarch( in vec3 ro, in vec3 rd )
{
	vec4 sum = vec4(0, 0, 0, 0);

	float t = 0.0;
	vec3 pos = vec3(0.0, 0.0, 0.0);
	for(int i=0; i<100; i++)
	{
		if (sum.a > 0.8 || pos.y > 9.0 || pos.y < -2.0) continue;
		pos = ro + t*rd;

		vec4 col = map( pos );
		
		// Accumulate the alpha with the colour...
		col.a *= 0.08;
		col.rgb *= col.a;

		sum = sum + col*(1.0 - sum.a);	
    	t += max(0.1,0.04*t);
	}
	sum.xyz /= (0.003+sum.w);

	return clamp( sum, 0.0, 1.0 );
}

// Function 490
float raymarchTerrain( in vec3 ro, in vec3 rd )
{
	float maxd = 30.0;
    float t = 0.1;
    for( int i=0; i<256; i++ )
    {
	    float h = mapTerrain( ro+rd*t );
        if( h<(0.001*t) || t>maxd ) break;
        t += h*0.8;
    }

    if( t>maxd ) t=-1.0;
    return t;
}

// Function 491
float valueNoiseStepped(float i, float p, float steps){ return mix(  floor(r11(floor(i))*steps)/steps, floor(r11(floor(i) + 1.)*steps)/steps, ss(fract(i), p,0.6));}

// Function 492
float aaa_step( float K, float u )
	{ return aaa_cov( .5 + u / K ); }

// Function 493
void march( inout vec3 p, vec3 d )
{
	float r = dist(p+d*EPSILON);
	for(int i = 0; i < V_STEPS; i++)
	{
		if(r < EPSILON || r > MAX_DEPTH)
			return;
		p += d*r*.5;
        r = dist(p);
	}
	return;
}

// Function 494
float rayMarch(vec3 ro, vec3 rd, float start)
{
    float t = start;
    for (int i = 0; i < 256; ++i)
    {
        float d = sdf(ro + rd * t);
        if (d < VERY_SMOL)
            return t;
        t += d;
        if (t >= MAX_DIST) return MAX_DIST;
    }
    return -t;
}

// Function 495
vec3 march(vec2 p) {
    for(int i = 0; i < 32; ++i) {
    	float d = scene(p).x;
        p += d * onCircle();
    }
    return scene(p).yzw;
}

// Function 496
float march(vec3 ro, vec3 rd )
{
	float maxd = 1.5;
    float t = 0.001;
    for( int i=0; i<1400; i++ )
    {
        vec3 p = ro+rd*t;
	    float h = map(p.xz);
        bool b = p.x<0.0||p.x>1.0||p.z>1.0;
        if (b) t=2.0;
        
        if( h>p.y || t>maxd) break;
        t+=0.001;
    }

    if( t>maxd ) t=-1.0;
    return t;
}

// Function 497
void rayMarch(vec3 pos, vec3 dir)
{
    // Efficiently start the ray just in front of the drone...
    float l = max(length(drone-pos)-14.2, .0);
    float d =  l;
    l+=23.;// ...and end it just after
    int hits = 0;
	// Collect 4 of the closest scrapes on the tracing sphere...
    for (int i = 0; i < 55; i++)
    {
        // Leave if it's gone past the drone or when it's found 7 stacks points...
        if(d > l || hits == 6) break;
        vec3 p = pos + dir * (d);
		float r= SphereRadius(d);
		float de = mapDE(p);
        // Only store the closest ones (roughly), which means we don't
        // have to render the 8 stack points, just the most relavent ones.
        // This also prevents the banding seen when using small stacks.
        if(de < r &&  de < eStack.x)
        {
            // Rotate the stack and insert new value!...
			dStack = dStack.wxyz; dStack.x = d; 
            eStack = eStack.wxyz; eStack.x = de;
			hits++;    
        }
		d +=de*.9;
    }
    return;
}

// Function 498
float march(vec3 ro,vec3 rd){
	float maxd=10.;
    float tmpDist=1.;
    float finalDist=0.;
    for(int i=0;i<50;i++){
        if( tmpDist<0.001||finalDist>maxd) break;
	    tmpDist=map(ro+rd*finalDist);
        finalDist+=tmpDist; }
    if(finalDist>maxd) finalDist=-1.;
	return finalDist; }

// Function 499
float mySmootherStep(float a, float b, float t) {
    t = t*t*t*(t*(t*6.0 - 15.0) + 10.0);
    return mix(a, b, t);
}

// Function 500
float march(vec3 ro, vec3 rd)
{
    float t = 0.0;
    for(int i = 0; i < ITR && t < MAX_DIST; ++i)
    {
        float dS = scene(ro + t*rd);
        if (abs(dS) < SURF_DIST )break;
        t += dS;
    }
        return  t ;
    }

// Function 501
vec4 raymarching(vec3 ro, vec3 rd)
{
    float t = 0.0;
    for (int i = 0; i < 50; i++) {
       	float distToSurf = map(ro + t * rd);
        t += distToSurf;
        if (distToSurf < PRECIS || t > DMAX) break; 
    }
    
    vec4 col = vec4(0.0);
    if (t <= DMAX) {
        vec3 nor = normal(ro + t * rd);
        col.z = 1.0 - abs((t * rd) * camMat).z / DMAX; // Depth
        col.xy = (nor * camMat * 0.5 + 0.5).xy;	// Normal
        col.w = dot(lightDir, nor) * 0.5 + 0.5; // Diff
        col.w *= shadow(ro + t * rd, lightDir);
    }
    
    return col;
}

// Function 502
vec4 raymarch(vec2 resolution, vec2 uv, vec4 start_data, mat4 camera_transform) {
    // Convert to range (-1, 1) and correct aspect ratio
    vec2 screen_coords = (uv - 0.5) * 2.0;
    screen_coords.x *= resolution.x / resolution.y;
    
    
    vec3 ray_start_position = camera_transform[3].xyz;
    
    vec3 ray_direction = normalize(vec3(screen_coords * LENS, 1.0));
    ray_direction = (camera_transform * vec4(ray_direction, 0.0)).xyz;
    
    
    float dist = start_data.a * 0.9;
    vec3 sample_point = ray_start_position + dist * ray_direction;
    
    float results = sample_world(sample_point);
    
    float tolerance = 0.0;
    
    for (int i=0; i<steps; i += 1) {
        dist += results;
        sample_point += ray_direction * results;
        results = sample_world(sample_point);
        
        // TODO: Derive from resolution, camera lens and distance
    	tolerance = LENS / resolution.x * dist;
        
        if (results < tolerance || dist > 5.0) {
        	break; 
        }
    }
    
    
    
    return vec4(sample_point, dist);
}

// Function 503
vec3 marchIt(vec3 origin, vec3 dir) {

    float rad = 0.0;
    float traveled = 0.0;
    for(float i = 0.0; i < MAX_STEPS; i += 1.0) {
        vec3 pos = origin + (dir * traveled);
        vec2 result = map(pos, rad);
        
        if(traveled > MIN_DISTANCE && result.y < EPSILON) {
			return vec3(traveled, i / MAX_STEPS, rad);
        }
        
        traveled += max(MIN_STEP_SIZE, result.y);
    }
    
    return vec3(0.0,1.0,0.0);
}

// Function 504
float smoothstep4(float e1, float e2, float e3, float e4, float val)
{
    return min(smoothstep(e1,e2,val), 1.-smoothstep(e3,e4,val));
}

// Function 505
Result raymarch (in vec3 ro, in vec3 rd)
{
    Result res = Result (.0, 0);

    for (int i = 0; i < MAX_ITER; i++)
    {
        vec3 p = ro + res.d * rd;
        Result tmp = scene (p);
        if (abs (tmp.d) < EPSILON*(1. + .125*tmp.d)) return res;
        res.d += tmp.d * STEP_SIZE;
        res.id = tmp.id;
    }

    return res;
}

// Function 506
float marchWorld(
    inout vec3 pos, inout vec3 dir,
    inout float dist, in float maxDist, in float minDist, out float nearest,
    inout int numSteps, in int maxNumSteps,
    inout vec3 color, out vec3 normal, out int returnCode
){
    float colorFrac = 1.;
    float transparency = 0.75;
    vec3 backgroundColor = vec3(0.1,0.2,0.5);
    vec3 sphereColor = vec3(0,0,0);
    vec3 waterColor = vec3(0,0,0.5);
    vec3 lightDir = normalize(vec3(1,1,1));
    nearest = maxDist;
    
    vec3 spherePosition = vec3(0,0,20);
    spherePosition.y = waterFunction(spherePosition, iTime-0.5)+5.;
    float sphereRadius = 4.;
    for(int i=0; i<maxNumSteps; i++) {
        float sdToWater = sdWater(pos);
        float sdToSphere = sdSphere(pos, spherePosition, sphereRadius);
        float sd = min(sdToWater, sdToSphere);
        if(sd < nearest){
        	nearest = sd;
        }
        
        numSteps++;
        if(dist + sd + minDist > maxDist){
            // Fill the remaining color.
    		color = mix(color, backgroundColor, colorFrac);
            sd = maxDist-dist-sd-minDist;
            dist += sd;
            pos += dir*sd;
            
            returnCode = TOO_FAR;
        	return sd;
        }
        if(sd <= minDist){
            if(sdToWater < sdToSphere){
            	sdWaterNormal(/*in vec3 pos=*/pos, /*inout vec3 normal=*/normal, /*inout float sd=*/sd);
                color = mix(color, waterColor*dot(lightDir, normal), colorFrac);
                colorFrac *= transparency;
                if(dot(normal, dir) < 0.){
                    dir = reflect(dir, normal);
                    sd = max(sd, minDist*2.);
                }
            }else{
            	sdSphereNormal(
                    /*in vec3 pos=*/pos, /*in vec3 center=*/spherePosition, /*in float radius=*/sphereRadius,
                    /*inout vec3 normal=*/normal, /*out float sd=*/sd
                );
                color = mix(color, sphereColor*dot(lightDir, normal), colorFrac);
                colorFrac *= transparency;
                
                if(dot(normal, dir) < 0.){
                    dir = reflect(dir, normal);
                    sd = max(sd, minDist*2.);
                }
            }
        }
        dist += sd;
        pos += dir*sd;
    }
    
    // Fill the remaining color.
    color = mix(color, backgroundColor, colorFrac);
    
    //
    returnCode = TOO_MANY_STEPS;
    return -1.;
}

// Function 507
float raymarch (in vec3 ro, in vec3 rd, inout int iter) {
    float t = .0;
    float d = .0;
    for (int i = 0; i < MAX_ITER; ++i) {
        vec3 p = ro + d * rd;
        t = scene (p);
        if (abs (t) < EPSILON * (1. + .125*t)) break;
        d += t*STEP_BIAS;
        iter = i;
    }

    return d;
}

// Function 508
vec4 RayMarch(vec3 ro,vec3 rd,vec3 p)
{
    float dO=0.0;
    int k=0;
    int i;
    for(i=0; i<MAX_STEPS; i++)
    {
        float ds=GetDist(p);
        float fds=ds;
        float mindist=2.0,maxdist=15.0;
        fds=min(fds, length(p)-mindist);
        fds=min(fds, maxdist-length(p));
        //dO+=fds*pow(maxdist/mindist,float(k));
        dO+=fds;
        p+=fds*rd;
        if(length(p)-mindist<=SURF_DIST)
        {
            p=normalize(p)*(maxdist-SURF_DIST*1.2);
            k--;
            
        }
        if(length(p)>=maxdist-SURF_DIST)
        {
            p=normalize(p)*(mindist+SURF_DIST*1.2);
            k++;
            fck-=0.3;
        }

        
        if(MAX_DIST<dO || k>2 || k<-3)
        {
        fck=0.0;
        break;
        }

        if(MAX_DIST<dO || ds<=SURF_DIST)
        break;
    }

    return vec4(p,dO);
}

// Function 509
float raymarch(vec3 ro, vec3 rd) {
    float t = 0.;
    for(int i = 0; i < MAX_ITER; i++) {
        float d = dstScene(ro+rd*t);
        if(d < MIN_DIST || t > MAX_DIST) {
            break;
        }
        t += d * .75;
    }
    return t;
}

// Function 510
bool marchInv( inout vec3 p, in vec3 d, in vec3 e)
{
	float t = 0.;//distInv(p+d*0.0001);
    vec3 dir = d;
	for(int i = 0; i < 32; i++)
	{
        float d= distInv(p+dir*t);
        if( d < 0.01 || t > 10. )
        {
            p+=dir*t;
			return  d < 0.01;
        }
		t+=d*.5;        
	}
    return false;
}

// Function 511
vec2 march(vec3 ro, vec3 rd, float initd, float maxd)
{
    float d = initd;
    for (int i=0; i<maxsteps; i++){
        vec2 n = map(ro + rd*d);
        d += n.x;
        if (n.x < sdist)
            return vec2(d, n.y);
        if (d > maxd)
            return vec2(1000., 0.);
    }
    return vec2(1000., 0.);
}

// Function 512
float smoothstepOsc(float x, float w){    
    SMOOTH_QUANTISE_UNIT(16., w)
}

// Function 513
v0 smoothstep4(v0 e1, v0 e2, v0 e3, v0 e4, v0 val
){return min(smoothstep(e1,e2,val),1.-smoothstep(e3,e4,val));}

// Function 514
vec4 raymarch( vec3 ro, vec3 rd, vec3 bgcol, ivec2 px )
{
	vec4 sum = vec4(0);
	float  t = 0., //.05*texelFetch( iChannel0, px&255, 0 ).x; // jitter ray start
          dt = 0.,
         den = 0., _den, lut, dv;
    for(int i=0; i<150; i++) 
    {
        vec3 pos = ro + t*rd;
        if( pos.y < -3. || pos.y > 3. || sum.a > .99 ) break;
        _den = den; den = map(pos); // raw density
        if( abs(pos.x) > .5 )       // cut a slice 
        {
            for (float ofs=0.; ofs<7.; ofs++) 
            {
                dv = (ofs/3.5-1.)*.4; // draw 7 isovalues
                lut = LUTs( _den+dv, den+dv ); // shaped through transfer function
                if (lut>.01)          // not empty space
                { 
                    vec3  col = hue(ofs/8.);
                    col = mix( col , bgcol, 1.-exp(-.003*t*t) ); // fog
                    sum += (1.-sum.a) * vec4(col,1)* (lut* dt*3.); // blend. Original was improperly just den*.4;
            }  }  }
        t += dt = max(.05,.02*t);     // stepping
    }

    return sum; 
}

// Function 515
vec4 rayMarch(inout vec4 p, in vec4 rd, out vec4 dists) {
  float dS = 99., d = 0., minDS = dS, steps = 0.;
  for (int i = 0; i < MAX_STEPS; i++) {
    steps++;
    dS = mapWDists(p, dists);
    minDS = min(minDS, abs(dS));
    d += dS;
    p = p + rd * dS;
    if ((0. <= dS && dS < SURF_DIST) || d > MAX_DIST) break;
  }
  return vec4(d, dS, minDS, steps);
}

// Function 516
vec2
raymarch( in vec3 start, in vec3 dir, inout float t, in float t_max )
{
    MPt mp;
    for ( int it=0; it!=RAYM_MAX_ITERS; ++it )
    {
        vec3 here = start + dir * t;
        mp = map( here );
        #if DRAW_ITERATIONS_GRADIENT
        mp.y = mp.y * 10000.0 + float(it);
        #endif
        if ( mp.distance < ( T_EPS * t ) || t > t_max )
        {
        	break;
        }
        #if 1
        // NOTE(theGiallo): this is to sample nicely the twisted things
        t += mp.distance * 0.4;
        #else
        t += mp.distance;
        #endif
    }
    if ( t > t_max )
    {
        t = -1.0;
    }
    return mp;
}

// Function 517
float linearstep(float begin, float end, float t) {
    return clamp((t - begin) / (end - begin), 0.0, 1.0);
}

// Function 518
RMResult raymarch(float time, vec3 ro, vec3 rd, out float t)
{
	t = 0.;
    vec3 p = ro + t * rd;
    RMResult s = map(p, time);
    float isInside = sign(s.dist);
    for(int i = 0; i < I_MAX; i++)
    {
        float inc = isInside * s.dist;
        if (min(abs(p.z - 3. * BOARD_UNIT), abs(p.z)) < 0.5 * BOARD_UNIT)
        {
            inc *= 0.7;         // dirty hack to fix domain discontinuity near z=0 and z=3*BOARD_UNIT plane 
            if (abs(p.x - 1.5 * BOARD_UNIT) < 0.5 * BOARD_UNIT)
                inc *= 0.6;     // fix annoying artefacts on the moving bishop that I couldn't remove otherwise
        }
        if (t + inc < FAR && abs(s.dist) > EPS) 
        {
			t += inc;
	        p = ro + t * rd;
            s = map(p, time);
        }
        else
        {
            if (t + inc > FAR)
            {
               s.id = -1.;
            }
            break;
        }
    }
    return s;
}

// Function 519
vec3 Raymarch( vec3 ro, vec3 rd, vec2 p )
{
    vec3 fragColor = vec3(0.);
    
    // this intersects the ray with a set of planes (shown as lines in the diagram).
    // these calculations could be moved outside the pixel shader in normal scenarios.
    vec2 t, dt, wt;
    SetupSampling( t, dt, wt, ro, rd );
    
    if( wt.x >= 0.01 )
    {
        float march = MarchAgainstPlanes( t.x, dt.x, wt.x, ro, rd, p );
        fragColor = max( fragColor, march * .6*vec3(1.2,.2,.2) );
    }
    if( wt.y >= 0.01 )
    {
        float march = MarchAgainstPlanes( t.y, dt.y, wt.y, ro, rd, p );
        fragColor = max( fragColor, march * .6*vec3(.2,1.2,.2) );
    }
    
    return fragColor;
}

// Function 520
float rayMarch(in vec3 ro, in vec3 rd, out int mr) {
	float dO = 0.0;
    
    for (int i = 0; i < RAYMARCH_MAX_STEPS; i++) {
		vec3 p = ro + rd * dO;
        float dS = getDist(p);
        dO += dS;
        if (dO > RAYMARCH_MAX_DIST) break;
        if (dS < RAYMARCH_SURFACE_DIST) {
            mr = 1;
            break;
        }
    }
    
    return dO;
}

// Function 521
vec4 raymarch(vec3 org, vec3 dir)
{
        float d = 0.0, glow = 0.0, eps = 0.02;
        vec3  p = org;
        bool glowed = false;

        for(int i=0; i<64; i++)
        {
                d = scene(p) + eps;
                p += d * dir;
                if( d>eps )
                {
                        if(flame(p) < .0)
                                glowed=true;
                        if(glowed)
                        glow = float(i)/64.;
                }
        }
        return vec4(p,glow);
}

// Function 522
vec3 Raymarch(Ray r)
{
    float t = 0.0;
    float d = 0.0;
    
	for(int j = 0; j < MAX_ITERATIONS_FOREGROUND; j++)
	{
		d = sdf(r.origin + r.direction * t, r);

		if(d < EPSILON)
            break;
        
		t += d;
        
        if(t > MAX_DISTANCE)
            break;
	}
    
    t = min(t, MAX_DISTANCE);
    
    return vec3(t, 0.0, d);
}

// Function 523
RMResult raymarch(vec3 ro, vec3 rd, out float t)
{
	t = 0.;
    vec3 p = ro + t * rd;
    RMResult s = RMResult(-1., FAR);
    for(int i = 0; i < I_MAX; i++)
    {
		s = map(p);
        if (t + s.dist < FAR && abs(s.dist) > EPS) 
        {
			t += s.dist;
	        p = ro + t * rd;

        }
        else
        {
            if (t + s.dist > FAR)
            {
               s.id = -1.;
            }
            break;
        }
    }
    return s;
}

// Function 524
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

// Function 525
vec3 raymarch( in vec3 ro, vec3 rd, vec2 tminmax ) {
    //start at starting loc
    float t = tminmax.x;
    //small delta
    float dt = (tminmax.y - tminmax.x) / float(vol_steps);
    //output color
    vec3 col= vec3(0.);
    vec3 pos = ro;
    //current sample
    float c = 0.;
    for( int i=0; i<vol_steps; i++ ) {
        //this steps through empty space faster
        t += (.7 + t*t * 0.007) * dt*exp(-c*c);
        pos = ro+t*rd;
        //get plasma density
        c = map(pos*size);
		//adjusted sumation
        col += c*c*normalize(abs(pos.zyx));
    }
    return col * 0.008;
}

// Function 526
void march(vec3 origin, vec3 dir, out float t, out int hitObj)
{
    t = 0.001;
    for(int i = 0; i < RAY_STEPS; ++i)
    {
        vec3 pos = origin + t * dir;
        float m;
        sceneMap3D(pos, m, hitObj);
        if(m < 0.01)
        {
            return;
        }
        t += m;
    }
    t = -1.0;
    hitObj = -1;
}

// Function 527
raymarchResult raymarchFast(vec3 at, vec3 normal) {
    for(int iteration = 0; iteration < 24; iteration++) {
        shapeResult sam = scene(at);
        if(sam.distance < 0.1)
            return raymarchResult(at, sam.materialId);
        at += normal * sam.distance;
    }
    return raymarchResult(vec3(0.0), MATERIAL_SKY);
}

// Function 528
vec4 raymarchClouds( in vec3 ro, in vec3 rd, in vec3 bcol, float tmax )
{
	vec4 sum = vec4( 0.0 );

	float sun = pow( clamp( dot(rd,lig), 0.0, 1.0 ),6.0 );
	float t = 0.0;
	for( int i=0; i<60; i++ )
	{
		if( t>tmax || sum.w>0.95 ) break;//continue;
		vec3 pos = ro + t*rd;
		vec4 col = mapClouds( pos );
		
        col.xyz += vec3(1.0,0.7,0.4)*0.4*sun*(1.0-col.w);
		col.xyz = mix( col.xyz, bcol, 1.0-exp(-0.00006*t*t*t) );
		
		col.rgb *= col.a;

		sum = sum + col*(1.0 - sum.a);	

		t += max(0.1,0.05*t);
	}

	sum.xyz /= (0.001+sum.w);

	return clamp( sum, 0.0, 1.0 );
}

// Function 529
vec3 ShadeSteps(int n)
{
    const vec3 a = vec3(97, 130, 234) / vec3(255.0);
    const vec3 b = vec3(220, 94, 75) / vec3(255.0);
    const vec3 c = vec3(221, 220, 219) / vec3(255.0);
    float t = float(n) / float(StepsMax);   
    if (t < 0.5)
        return mix(a, c, 2.0 * t);
    else
        return mix(c, b, 2.0 * t - 1.0);
}

// Function 530
float March(vec3 origin, vec3 direction, float start, float stop, inout float edgeLength)
{
    float depth = start;
    
    for	(int i = 0; i < MAX_STEPS; i++)
    {
        float dist = SceneSDF(origin + (depth * direction)); // Grab min step
        edgeLength = min(dist, edgeLength);
        
        if (dist < EPSILON) // Hit
            return depth;
        
        if (dist > edgeLength && edgeLength <= EDGE_THICKNESS ) // Edge hit
            return 0.0;
        
        depth += dist; // Step
        
        if (depth >= stop) // Reached max
            break;
    }
    
    return stop;
}

// Function 531
Marched March(vec3 ro, vec3 rd, float thresh, float dmax, int iters)
{
    Marched c = Marched(dmax, mSky, dmax);
    int i = iters;
    float t = 0.;
    while (i-- > 0) {
        vec3 mp = ro + rd * t;
        Hit h = Scene(mp);
        float d = h.d, ad = abs(d);
        t += d;
        //if (ad < abs(c.nmd)) {
            c.m = h.m, c.nmd = h.d;
            if (rd.y >= 0. && (ad > dmax
            	|| mp.y > hmax))
                break; //t = dmax;
        	if (ad < thresh * t || t >= dmax)
        		break;
        //}
    }
    c.t = t = clamp(t, 0., dmax);
    if (abs(c.nmd) > thresh * 2. * t) 
        c.m = mSky;
    if (c.m == mSky)
        c.t = dmax; // caller won't be able to tell how far it got though
    return c;
}

// Function 532
float RayMarch(vec3 ro, vec3 rd){
    // distance from origin
    float dO=0.;
    // march until max steps is achieved or object hit
    for(int i=0; i <MAX_STEPS; i++){
        // current point being evaluated
        vec3 p = ro + dO*rd;
        
        // get distance to seam
        float ds = GetDist(p);
        //move origin to new point
        dO+=ds*.1;
        if(ds < SURFACE_DIST || dO > MAX_DIST){
            break;
        }
    }
    return dO;
}

// Function 533
vec3 march(vec3 eye, vec3 dir, float cone) {
    /*
    float s = STEP_SIZE;
    vec3 col = vec3(0.0);
    vec3 pos = eye;
	// main loop
    for (int i = 0; i < STEPS; ++i) {
        // get the color
        col += get_color(eye, dir, pos, s);
        // if the color is white, break (optimization)
        if (col.x >= 1.0 && col.y >= 1.0 && col.z >= 1.0) {
        	break;
        }
        pos += dir*s;
        s *= STEP_MULTIPLIER;
    }
    return col;
	*/
    // raymarching for the stars
    // the big box step
    for (int i = 0, box_size = 1; i < MAJOR_STEPS; ++i, box_size *= BOX_INCREMENT) {
        // start
        vec3 pos = eye;
        // minor step
        for (int j = 0; j < MINOR_STEPS; ++j) {
    
            // distance to the next point
            float dist = get_dist_to_next(float(box_size), pos, dir);
        	// get the color
            //col += get_color(eye, dir, pos, dist, cone);
            // if the star was hit, break and return
            if (was_star_hit(eye, dir, pos, cone)) {
            	return vec3(1.0);
            }
            // get the next point
            pos = get_next_point(dist, pos, dir);
        }
    }
    // raymarching for the cloud
    vec3 col = vec3(0.0);
    vec3 pos = eye+dir*STEP_SIZE;
    float s = STEP_SIZE;
    float tot = 0.0;
    // loop
    for (int i = 0; i < STEPS; ++i) {
        // get the color
        col += get_color(eye, dir, pos, s);
        // increment step size
        s *= STEP_MULTIPLIER;
        // increase step
        pos += dir*s;
        // increase total
        tot += s;
    }
    return col / tot;
}

// Function 534
Hit RayMarchHighDetailModel(vec3 ro, vec3 dir) 
{
    vec3 P = vec3(0.,0.,0.);
    float t = 0.;
    while(t < MAX_MARCHING_DISTANCE) 
    {
        P = ro + t*dir;
        Hit hit = SD_HighDetailModel(P);
        if((hit.d)<0.01) 
    	{
            hit.normal = normalHighDetailModel(P);
            hit.tangent = tangent(hit.normal);
            hit.binormal = normalize(cross(hit.normal, hit.tangent));
			return hit;
        }
        t+=hit.d;
    }
    return Hit(-1, MAX_MARCHING_DISTANCE,vec3(0.),vec3(0.),vec3(0.),vec3(0.));
}

// Function 535
vec4 doMarch(vec3 pos,vec3 cam){
    int i = 0;
    vec3 march = vec3(0);
    float d = 1.;
    
    for(;i<200&&d>0.1&&length(march)<50.;i++){
        d = de(pos+march);
        march += d*cam;
    }
    return vec4(march+pos,i);
}

// Function 536
void march(inout vec3 pos, vec3 dir, inout float dmin)
{
    float eps=.001;
    float dtot=0.;
    dmin=10000.;
    float dp=dist(pos);
    for(int i=0;i<100;i++)
    {
        float d=dist(pos);
        if(d<dp) dmin=min(d,dmin);
        dp=d;
        d*=.8;
        pos+=d*dir;
        dtot+=d;
        if(d<eps) break;
        if(dtot>4.) { pos-=(dtot-4.)*dir; break; }
    }
}

// Function 537
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

// Function 538
vec3 raymarch(CastRay castRay){

    float currentDist = INTERSECTION_PRECISION * 2.0;
    float lastDist = currentDist;
    vec3 pos, lastPos = vec3(0);    
    vec4 outline = vec4(0);
    
    Model model;
    Ray ray = Ray(castRay.origin, castRay.direction, 0.);

    for (int i = 0; i < NUM_OF_TRACE_STEPS; i++) {

        lastPos = pos;
        pos = ray.origin + ray.direction * ray.len;

        if (ray.len > MAX_TRACE_DISTANCE) {
            break;
        }

        if (currentDist < INTERSECTION_PRECISION) {
            break;
        }

        if (currentDist > lastDist && currentDist < OUTLINE) {

            float t = lastDist / OUTLINE;
            
            vec4 newOutline = shadeOutline(pos, t);
            float contribution = 1. - outline.a;
            outline.rgb = mix(outline.rgb, newOutline.rgb, contribution);
            outline.a += newOutline.a * contribution;

            if (t < OUTLINE_BOUNDRY) {
                pos = lastPos;
                break;
			}
        }

        model = map(pos);
        lastDist = currentDist;
        currentDist = model.dist;
        ray.len += currentDist * FUDGE_FACTOR;
    }
	
    vec3 color = mix(
        shadeSurface(pos, ray),
       	outline.rgb,
        outline.a
	);
    
    return color;
}

// Function 539
float raymarch_cube(vec3 origin, vec3 dir, float start, float end) {
	const int max_iterations = 64;
	const float stop_threshold = 0.01;
	float depth = start;
	for (int i=0; i<max_iterations; i++) {
		float dist = DE_cube(origin + dir*depth);
		if (dist < stop_threshold) return depth;
		depth += dist;
		if (depth >= end) return end;
	}
	return end;
}

// Function 540
bool raymarch(vec3 ro, vec3 rd, inout float t, float t1, float step_size, float eps) {
    for (int i=int(ZERO); i<100; i++) {
        float dt = step_size*mapDist(ro+rd*t);
        t += dt;
        if (abs(dt) < eps) break;
        if (t > t1) return false;
    }
    return true;
}

// Function 541
IntersectInfo raymarch(vec3 pos,vec3 dir){
	IntersectInfo info;
    info.distance = 1.;
    float d = 0.;
    for(int i=0;i<64;i++){
		info.surface = pos+dir*info.distance;
        d = map(info.surface);
		info.distance += d;
        if(d < 0.02||info.distance>31.) break;
    }
    info.surface += d*dir;
	info.normal = normal(info.surface);
    return info;
}

// Function 542
vec4 raymarch( vec3 ro, vec3 rd, vec3 bgcol, ivec2 px )
{
	vec4 sum = vec4(0);
	float  t = 0., //.05*texelFetch( iChannel0, px&255, 0 ).x; // jitter ray start
          dt = 0.,
         den = 0., _den, lut;
    for(int i=0; i<150; i++) {
        vec3 pos = ro + t*rd;
        if( pos.y < -3. || pos.y > 3. || sum.a > .99 ) break;
        _den = den; den = map(pos); // raw density
        lut = LUTs( _den, den );    // shaped through transfer function
        
        if( lut > .01               // optim
            && abs(pos.x) > .5      // cut a slice 
          ) {
            float dif = clamp((lut - LUTs(_den, map(pos+.3*sundir)))/.6, 0., 1. ); // pseudo-diffuse using 1D finite difference in light direction 
         // float dif = clamp((den - map(pos+.3*sundir))/.6, 0., 1. );             // variant: use raw density field to evaluate diffuse
            vec3  lin = vec3(.65,.7,.75)*1.4 + vec3(1,.6,.3)*dif,          // ambiant + diffuse
                  col = lin * mix( vec3(1,.95,.8), vec3(.25,.3,.35), lut );// pseudo- shadowing with in-cloud depth ? 
            col = mix( col , bgcol, 1.-exp(-.003*t*t) );   // fog

            sum += (1.-sum.a) * vec4(col,1)* (lut* dt*5.); // blend. Original was improperly just den*.4;
        }
        t += dt = max(.05,.02*t); // stepping
    }

    return sum; // clamp( sum, 0., 1. );
}

// Function 543
vec4 march(inout vec3 pos, vec3 dir)
{
    // cull the sphere
    if(length(pos-dir*dot(dir,pos))>1.07) 
    	return vec4(0,0,0,1);
    
    float eps=0.003;
    float bg=1.0;
    float d=10000., dp;
    for(int cnt=0;cnt<50;cnt++)
    {
        dp=d;
        d = dist(pos);
        pos+=d*.8*dir;
        if(d<eps) break;
    }
    bg = (d<dp)?0.0:1.0;
    vec3 n = getGrad(pos,eps*.1);
    return vec4(n,bg); // .w=1 => background
}

// Function 544
float rayMarch( in vec3 ro, in vec3 rd, float tmax )
{
    float t = 0.0;
    
    // bounding plane
    float h = (1.0-ro.y)/rd.y;
    if( h>0.0 ) t=h;

    // raymarch 30 steps    
    for( int i=0; i<30; i++ )    
    {        
        vec3 pos = ro + t*rd;
        float h = map( pos );
        if( h<0.001 || t>tmax ) break;
        t += h;
    }
    return t;    
}

// Function 545
float march(vec3 ro, vec3 rd, float mx) {
	float t = 0.0;

	for(int i = 0; i < 200; i++) {
		float d = map(ro + rd*t);
		if(d < 0.001 || t >= mx) break;
		t += d*0.75;
	}

	return t;
}

// Function 546
void rayMarch (inout Ray ray) {
	//Отступ
	ray.origin += ray.near * ray.direction;
	
	ray.distance 	= ray.far;
	ray.normal 		= vec3(0);
	ray.object.uv   = vec2(0);
	ray.object.id	= -1;
	ray.hit 		= false;
	
	float d;
	vec3 normal;
	vec2 uv;
	
	d = intersectSphere(ray, sphere_light, normal, uv);
	if (ray.distance > d) {
		ray.distance 	= d;
		ray.normal		= normal;
		ray.object.uv	= uv;
		ray.object.id	= 1;
		ray.hit 		= true;
	}
	
	d = intersectSphere(ray, sphere1, normal, uv);
	if (ray.distance > d) {
		ray.distance 	= d;
		ray.normal		= normal;
		ray.object.uv	= uv;
		ray.object.id	= 2;
		ray.hit 		= true;
	}
	
	d = intersectBox(ray, box1, normal, uv);
	if (ray.distance > d) {
		ray.distance 	= d;
		ray.normal		= normal;
		ray.object.uv	= uv;
		ray.object.id	= 3;
		ray.hit 		= true;
	}
	
	d = intersectBox(ray, box2, normal, uv);
	if (ray.distance > d) {
		ray.distance 	= d;
		ray.normal		= normal;
		ray.object.uv	= uv;
		ray.object.id	= 4;
		ray.hit 		= true;
	}

	d = intersectSphere(ray, sphere2, normal, uv);
	if (ray.distance > d) {
		ray.distance 	= d;
		ray.normal		= normal;
		ray.object.uv	= uv;
		ray.object.id	= 5;
		ray.hit 		= true;
	}

	d = intersectCylinder(ray, cylinder1, normal, uv);
	if (ray.distance > d) {
		ray.distance 	= d;
		ray.normal		= normal;
		ray.object.uv	= uv;
		ray.object.id	= 6;
		ray.hit 		= true;
	}

	ray.position = ray.origin + ray.direction * ray.distance;
}

// Function 547
vec3 RayMarch(vec3 ro, vec3 rd){
    float dt = 0.0;
    for (int i = 0; i < MAX_STEPS; i++){
        vec3 p = ro + dt * rd;
        Hit hit = map(p);
        float dist = hit.d;
        if(dist < MIN_DIST){
            isHit = true;
            break;
        }
        if(isHit == true || dist > MAX_DIST){
            return vec3(0.0, 0.0, 0.2);
            break;
        }
        dt += dist*0.925; // Multiplied by 0.925 to stop weird things from happening
    }
    return ro + dt * rd;
}

// Function 548
raymarchResult raymarch(vec3 direction, vec3 camera, float maxD, float outline){ //ray from camera to direction (extended into infinity)
    direction = normalize(direction);
    
    raymarchResult res;
    res.type = 0.;
    res.position = vec3(0.);
    
    float minH = 999.;
    float setI = (outline == 2. ? outlineSize + 1. : 0.3);
    
    for(float i = setI; i < maxD;){ //i = length of ray
        vec3 p = camera + i * direction;
    	mapResult hMap = map(p); //smallest distance from all objects to point
        float h = hMap.Md;
        if(h > minH && minH > MinDist && minH < MinDist + outlineSize && outline != 0.){ //if distance increases from smallest distance
            res.position = p;
            res.type = 2.;
            return(res);
        }
        minH = h;
        
        
    	if(h < MinDist){
            res.position = p;
            res.type = 1.;
			return(res); //successfully hit something at point "point"
        }
        i += h;
    }

    return(res);
}

// Function 549
void march(vec3 origin, vec3 dir, out float t, out int hitObj)
{
    t = 0.001;
    for(int i = 0; i < RAY_STEPS; ++i)
    {
        vec3 pos = origin + t * dir;
        float m;
        sceneMap3D(pos, m, hitObj);
        if(m < 0.01)
        {
            return;
        }
        t += m;
    }
    // If there is no object in 256 steps
    t = -1.0;
    hitObj = -1;
}

// Function 550
void MarchPOV(inout Ray r, float startTime,float sscoc
){//dpos = vec3(-2.2,0,0)
 ;//lorentzF = LorentzFactor(length(dpos))
 ;float speedC = length(dpos)/cSpe
 ;vec3 nDpos = vec3(1,0,0)
 ;if(length(dpos)>0.)nDpos = normalize(dpos)
 ;//shrink space along vel axis (length contraction of field of view)
 ;float cirContraction = dot(nDpos,r.dir)*(LorentzFactor(length(LgthContraction*dpos)))
 ;vec3 newDir = (r.dir - nDpos*dot(nDpos,r.dir)) + cirContraction*nDpos
 ;r.dir = normalize(newDir)
 ;float dDirDpos = dot(dpos,r.dir)
 ;// Aberration of light, at high speed (v) photons angle of incidence (a) vary with lorenz factor (Y) :
 ;// tan(a') = sin(a)/(Y*(v/c + cos(a)))
 ;// velComponentOfRayDir' = Y*(velComponentOfRayDir+v/c)
 ;float lightDistortion = lorentzF*(dot(-nDpos,r.dir)+speedC)
 ;r.dir=mix(r.dir
           ,normalize((r.dir-nDpos*dot(nDpos,r.dir))-lightDistortion*nDpos)
           ,FOVAberrationOfLight)
 ;//Classical Newtown Mechanic instead would be
 ;//r.dir = normalize(r.dir-dpos/cSpe)
 ;for (r.iter=0;r.iter<maxStepRayMarching;r.iter++
 ){float camDist = length(r.b - objPos[oCam])
  //;float photonDelay = -camDist*cLag/cSpe
  //takes dilated distance x/Y and find the time in map frame with :
  // v = -dDirDpos (dot product of direction & velocity, because we want to transform from cam frame to map frame)
  // Y = lorentzFactor
  // t' = Y(t-v*(x/Y)/c²)
  // t' = Y(0-v*(x/Y)/c²)
  // t' = Y(v*x/Y)/c²
  // t' = vx/c²
  ;r.time = startTime
  ;r.time += mix(simultaneouity*dDirDpos*camDist/(cSpe*cSpe),-camDist*cLag/cSpe,cLag)
  ;SetTime(r.time)
  ;r.dist = map(r.b,-1)
  ;//Gravitational lens
  ;vec3 blackHoleDirection=(objPos[oBlackHole]-r.b)
  ;r.dir+=(1./RayPrecision)*r.dist*reciprocalLipschitz
      *normalize(blackHoleDirection)*BlackHoleMassFactor/(length(blackHoleDirection)*cSpe*cSpe)
  ;r.dir = normalize(r.dir)
  ;if(abs(r.dist)<getEpsToFps(sscoc))break
  ;r.b+= (1./RayPrecision)*(r.dist)*reciprocalLipschitz*(r.dir);}
 ;//r.b = origin + r.dir*min(length(r.b-origin),maxDist)
 ;r.surfaceNorm = GetNormal(r.b).xyz;}

// Function 551
MarchResult MarchRay(vec3 orig,vec3 dir)
{
    float steps = 0.0;
    float dist = 0.0;
    
    for(int i = 0;i < MAX_STEPS;i++)
    {
        float sceneDist = Scene(orig + dir * dist);
        
        dist += sceneDist * STEP_MULT;
        
        steps++;
        
        if(abs(sceneDist) < MIN_DIST)
        {
            break;
        }
    }
    
    MarchResult result;
    
    result.position = orig + dir * dist;
    result.normal = Normal(result.position);
    result.dist = dist;
    result.steps = steps;
    return result;
}

// Function 552
vec4 rayMarchCycle(vec3 rayOrigin, vec3 rayDir,vec2 fragCoord,out vec3 outPhit)
{
    vec3 col=vec3(0.);
	vec3 L = normalize( vec3(.1, .9, -.23 ));

    vec3 finalColor=vec3(0.0);
    vec3 cloudColor;
    vec2 rayHit = rayMarch(rayOrigin, rayDir,cloudColor,fragCoord,0);
    float dist=rayHit[0];
    float mat=rayHit[1];
    vec3 pHit=rayOrigin+rayDir*dist;
    outPhit=pHit;
    
    vec3 N=calcNormal(pHit);
    float NoL = max(dot(N, L), 0.0);
    float ao=calcAO(pHit,N);

    vec3 skyColor=Sky(pHit,(mat==2.0)?reflect(rayDir,N):rayDir,iTime,L);
    vec3 darkSkyColor=nightSky(pHit,(mat==2.0)?reflect(rayDir,N):rayDir,iTime,L);
    vec3 finSkyCol=skyColor;
    
    if (mat==1.0) // paper plane
    {
        finalColor=vec3(NoL)*vec3(0.58,0.12,0.12);
    }
    else if (mat==2.0) // sea plane sea scene/night
    {
        if ((iTime>=60.0)&&(iTime<65.0)) finSkyCol=mix(darkSkyColor,skyColor,(65.0-iTime)/5.0);
        else if (iTime>=65.0) finSkyCol=darkSkyColor;
		//vec3 reflektCol=reflekkt(rayOrigin,rayDir,fragCoord);        
        vec3 col=mix(vec3(pow(NoL,8.0))*vec3(0.01,0.01,0.02),finSkyCol,0.6);
        finalColor=clamp(col,vec3(0.0),vec3(1.0));
        // final fadeout
        if (iTime>=65.0) finalColor=mix(finalColor,vec3(0.0),clamp((iTime-65.0)/3.0,0.0,1.0));
    }
    else if (mat==3.0) // cage bars
    {
        vec3 barCol=vec3(0.74);
        finalColor=barCol.xyz*ao;
        if (iTime>=70.0) finalColor=mix(vec3(0.0),finalColor,clamp((iTime-70.0),0.0,1.0));
    }
    else if (mat==4.0) // cage base
    {
        vec3 barCol=vec3(0.22);
        //float d=distance(pHit,vec3(0.0));
        //vec3 fgcol=fog(barCol,d,vec3(0.8));
        finalColor=barCol.xyz*ao;
        if (iTime>=70.0) finalColor=mix(vec3(0.0),finalColor,clamp((iTime-70.0),0.0,1.0));
    }
    else if (mat==5.0) // cage floor
    {
        float d=distance(pHit,vec3(0.0));
        vec3 floorCol=vec3(0.01);
        vec3 fgcol=fog(floorCol,d,vec3(0.8));
        finalColor=fgcol.xyz;
        if (iTime>=70.0) finalColor=mix(vec3(0.0),finalColor,clamp((iTime-70.0),0.0,1.0));
    }
    else if ((iTime<20.0)||(iTime>=70.0)) // cage scene
    {
        finalColor=vec3(0.8);
    }
    else
    {
        if ((iTime>=60.0)&&(iTime<65.0)) finSkyCol=mix(darkSkyColor,skyColor,(65.0-iTime)/5.0);
        else if (iTime>=65.0) finSkyCol=darkSkyColor;
        
        finalColor=finSkyCol+cloudColor*0.85;

        // final fadeout
        if (iTime>=65.0) finalColor=mix(finalColor,vec3(0.0),clamp((iTime-65.0)/3.0,0.0,1.0));
    }

    // ghost mode
    if ((iTime>12.0)&&(iTime<20.0))
    {
    	vec2 rayHit = rayMarch(rayOrigin, rayDir,cloudColor,fragCoord,1);
        if (rayHit[1]==1.0)
        {
            finalColor+=vec3(NoL)*vec3(0.58,0.12,0.12);
        }
    }
    
    return vec4(finalColor,-999999.0);    
}

// Function 553
vec3 marchRay(vec3 eyePos,vec3 dir)
{
    vec3 color = vec3(0);
    vec3 currentPoint = eyePos;
    while(length(currentPoint) < MAX_DISTANCE)
    {
        vec3 sceneCol1, sceneCol2;
        float dist = sceneDf(currentPoint, sceneCol1, sceneCol2);
        if(dist <= MIN_DISTANCE)
        {
            color = processLighting(sceneCol1, sceneCol2, eyePos, currentPoint);
            float absz = -currentPoint.z;
            color *= 1.0-vec3(max(absz - 40.0, 0.0)/10.0);
            break;
        }
        currentPoint += (dist * dir);
    }
    return color;
}

// Function 554
float rayMarching (vec3 ro, vec3 rd, out vec3 pos)
{
	float totalDist = 0.0;
	pos = ro;
	float dist = EPSILON;
	
	for (int i = 0; i < MAX_ITER; i++) {
		if (dist < EPSILON || totalDist > MAX_DIST) break;
		dist = distanceFunction(pos);
		totalDist += dist*0.98;
		pos = ro + totalDist * rd;
	}		
	return dist;
}

// Function 555
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

// Function 556
vec3 march( in vec3 ro, in vec3 rd)
{
    float t=0.0,d;
    
    for(int i=0;i<STEPS;i++)
    {
        d=map(ro+rd*t).d;
        if(abs(d)<PRECISION){hit=true;}
        if(hit==true||t>DEPTH){break;}
        t+=d;
    }
    
    return ro+rd*t;
}

// Function 557
vec4 internalmarchconservative(vec3 atm, vec3 p1, vec3 p2, float noisestrength){
    int stepcount = CLOUDS_STEPS;
    float stepsize = 1.0 / float(stepcount);
    float rd = fract(rand2sTime(UV)) * stepsize;
    float c = 0.0;
    float w = 0.0;
    float coverageinv = 1.0;
    vec3 pos = vec3(0);
    float clouds = 0.0;
    vec3 color = vec3(0.0);
	float colorw = 1.01;                      
    float godr = 0.0;
    float godw = 0.0;
    float depr = 0.0;
    float depw = 0.0;
    float iter = 0.0;
    vec3 lastpos = p1;
    //depr += distance(CAMERA, lastpos);
    depw += 1.0;
    float linear = distance(p1, mix(p1, p2, stepsize));
    float limit = distance(p1, p2);
    for(int i=0;i<CLOUDS_STEPS;i++){
        if(coverageinv <= 0.01) break;
        pos = mix(p1, p2, iter + rd);
        vec2 as = cloudsDensity3D(pos);
		vec3 timev = vec3(time*0.01, time * 0.01, 0.0);
        clouds = as.x*1.3;
        float W = clouds * max(0.0, coverageinv);
        color += W * vec3(as.y * as.y);
        colorw += W;

        coverageinv -= clouds;
        depr += step(0.99, coverageinv) * distance(lastpos, pos);
        lastpos = pos;
        iter += stepsize;
        //rd = fract(rd + iter * 124.345345);
    }
    if(coverageinv > 0.99) depr = 0.0;
    float cv = 1.0 - clamp(coverageinv, 0.0, 1.0);
    color *= getSunColorDirectly(0.0) * 3.0;
	vec3 sd = rotmat(vec3(1.0, 1.0, 0.0), time * 0.25) * normalize(vec3(0.0, 1.0, 0.0)); 
    color *= 0.5 + 0.5 * (1.0 / (0.2 + 1.0 * (1.0 - max(0.0, dot(sd, normalize(p2 - p1))))));
    return vec4(sqrt(max(0.0, (dot(sd, VECTOR_UP)))) * mix((color / colorw) + atm * min(0.6, 0.01 * depr), atm * 0.41, min(0.99, 0.00001 * depr)), cv);
}

// Function 558
float linstep(float x0, float x1, float xn)
{
	return (xn - x0) / (x1 - x0);
}

// Function 559
vec4 raymarch( in vec3 ro, in vec3 rd, in vec3 bgcol, in ivec2 px )
{
	vec4 sum = vec4(0.0);

	float t = 0.0;//0.05*texelFetch( iChannel0, px&255, 0 ).x;
    for(int i=0; i<256; i++)
    {
       vec3 pos = ro + t*rd;
       if( /*pos.y<-3.0 || pos.y>2.0 ||*/ sum.a>0.99 ) break;
       float den = map( pos );
       if( den>0.01 )
       {
         float dif = clamp((den - map(pos+0.3*sundir))/0.6, 0.0, 1.0 );
         vec3  lin = vec3(0.65,0.7,0.75)*1.4 + vec3(1.0,0.6,0.3)*dif;
         vec4  col = vec4( mix( vec3(1.0,0.95,0.8), vec3(0.25,0.3,0.35), den ), den );
         col.xyz *= lin;
         col.xyz = mix( col.xyz, bgcol, 1.0-exp(-0.005*t*t) );
         col.w *= 0.4;
         
         col.rgb *= col.a;
         sum += col*(1.0-sum.a);
       }\
       t += max(0.05,0.01*t);\
    }

    return clamp( sum, 0.0, 1.0 );
}

// Function 560
vec3 raymarch( in vec3 ro, in vec3 rd, in vec2 pixel )
{
	vec4 sum = vec4( 0.0 );

	float t=dither(pixel);
	
	for( int i=0; i<100; i++ )
	{
		if( sum.a > 0.99 ) break;
		
		vec3 pos = ro + t*rd;
        float d= map( pos );
		vec4 col = vec4(mix( vec3(1.0,1.0,1.23), vec3(0.1,0.0,0.10), d ),1.);

		col *= d*3.;

		sum +=  col*(1.0 - sum.a);	

		t += 0.05;
	}

	return clamp( sum.xyz, 0.0, 1.0 );
}

// Function 561
vec2 rayMarch(vec3 ro, vec3 rd, vec2 uv)
{
    float l = hash12(uv)*.5;
    int i;
    vec2 d2;
    mist = 0.0;
    
    #ifdef OFF_LINE
    for (i = 0; i < 800; i++)
    #else
    for (i = 0; i < 300; i++)
    #endif
    {
        vec3 p = ro + rd * l;
        d2 = de(p);
        mist += smoothstep(.12,.03,d2.x);
        
        if (abs(d2.x) < .03 || l > 1000.0) break;
        #ifdef OFF_LINE
        l += d2.x*.5;
        #else
        l += d2.x*.85;
        #endif
	}
    mist = pow(mist, 3.) * .0001;
    return vec2(l, d2.y);
}

// Function 562
float my_smoothstep( float x )
{
    return x*x*(3.0-2.0*x);
}

// Function 563
vec2 march(vec3 rd, vec3 ro){
 	float t = 0.;   
    vec2 d = vec2(0);
    
    for(int i = 0; i < 80; i++){
    	d = map(ro + rd*t); 	   
        if(abs(d.x) < .002 || t > 90.){
            break;
        }
        t += d.x * .75;
    }   
    return vec2(t, d.y);
}

// Function 564
vec3 raymarch(in vec3 orig, in vec3 dir) {
	float result = 0.0;
    vec3 p = orig;
    for (int i = 0; i < 32; ++i) {
    	float d = torus_sdf(p);
        result = result + d;
        p = p + d * dir;
        if (d < 1.0e-4) {
            return p;
        }
    }
    return p;
}

// Function 565
void march(vec3 origin, vec3 dir, out float t, out int hitObj, vec3 lightPos)
{
    t = 0.001;
    for(int i = 0; i < RAY_STEPS; ++i)
    {
        vec3 pos = origin + t * dir;
    	float m;
        sceneMap3D(pos, m, hitObj, lightPos);
        if(m < 0.01)
        {
            return;
        }
        t += m;
    }
    t = -1.0;
    hitObj = -1;
}

// Function 566
bool raymarch( Ray ray, out vec3 hitPos, out vec3 hitNrm )
{
	const int maxSteps = 128;
	const float hitThreshold = 0.0001;

	bool hit = false;
	hitPos = ray.org;

	vec3 pos = ray.org;

	for ( int i = 0; i < maxSteps; i++ )
	{
		float d = scene( pos );

		if ( d < hitThreshold )
		{
			hit = true;
			hitPos = pos;
			hitNrm = sceneNormal( pos, d );
			break;
		}
		pos += d * ray.dir;
	}
	return hit;
}

// Function 567
vec2 soundStep(in float t) {
    float o = 0.2*noise(vec2(t,0.));
    float i = fract(t*1.23+o);
    
    return Wind(t*0.025) * clamp(i*10.,0.,1.) * clamp(1.-i*6., 0., 1.);
}

// Function 568
float march(in vec3 ro, in vec3 rd)
{
	float precis = 0.001;
    float h=precis*2.0;
    float d = 0.;
    for( int i=0; i<ITR; i++ )
    {
        if( abs(h)<precis || d>FAR ) break;
        d += h;
	    float res = map(ro+rd*d);
        h = res;
    }
	return d;
}

// Function 569
vec3 ShadeSteps(int n)
{
   float t=float(n)/(float(Steps-1));
   return vec3(t,0.25+0.75*t,0.5-0.5*t);
}

// Function 570
vec4 Raymarch(Ray ray)
{
    float depth = 0.0;
    vec3 position = vec3(0.0);
    for(int i = 0; i < RAYMARCHING_MAX_STEPS; ++i)
    {
        position = ray.origin + (depth * ray.direction);
        float distance = SceneSD(position);
        
        if(distance < RAYMARCHING_THRESHOLD || distance >= RAYMARCHING_MAX_DEPTH)
        {
            break;
        }
        
        depth += distance;
    }
    
    return vec4(position, depth);
}

// Function 571
vec4 intersectSphere_raymarch(vec3 o, vec3 l, float r)
{
	int max_steps = 200;
	float max_delta = r * 1e3f;
	float min_delta = r / 1e5f;
	float end_delta = r / 1e4f;

	vec4 NO_HIT = vec4(0.0f, 0.0f, 0.0f, -1.0f);
	vec3 rd = l;
	vec3 p = o;

	int i;
	for (i = 0; i < max_steps; i++)
	{
		float d = sdSphere(p, r);
		if (d > max_delta)
			return NO_HIT;
		if (d < end_delta)
			break;
		p += rd * max(d, min_delta);
	}

	if (i == max_steps)
		return NO_HIT;
	else
		return vec4(p, 1.0f);
}

// Function 572
vec2 raymarch(vec3 position, vec3 direction)
{
    /*
	This function iteratively analyses the scene to approximate the closest ray-hit
	*/
    // We track how far we have moved so we can reconstruct the end-point later
    float total_distance = NEAR_CLIPPING_PLANE;
    vec2 result;
    for(int i = 0 ; i < NUMBER_OF_MARCH_STEPS ; ++i)
    {
        result = scene(position + direction * total_distance);
        // If our ray is very close to a surface we assume we hit it
        // and return it's material
        if(result.x < EPSILON)
        	break;
        
        // Accumulate distance traveled
        // The result.x contains closest distance to the world
        // so we can be sure that if we move it that far we will not accidentally
        // end up inside an object. Due to imprecision we do increase the distance
        // by slightly less... it avoids normal errors especially.
        total_distance += result.x * DISTANCE_BIAS;
        
        // Stop if we are headed for infinity
        if(total_distance > FAR_CLIPPING_PLANE)
            break;
    }
    // By default we return no material and the furthest possible distance
    // We only reach this point if we didn't get close to a surface during the loop above
    return vec2(total_distance, result.y);
}

// Function 573
void Step (ivec2 iv, out vec3 r, out vec3 v)
{
  vec3 f;
  float fDamp, dt;
#if SCYL
  if (iv.x == nBallEx) iv.x = 0;
#endif
  IdNebs ();
  fDamp = 0.5;
  r = GetR (vec2 (iv));
  v = GetV (vec2 (iv));
  f = PairForce (iv, r) + SpringForce (iv, r, v) + BendForce (iv, r) +
     WallForce (r) + BShForce (r) - gravVec - fDamp * v;
  dt = 0.02;
  v += dt * f;
  r += dt * v;
}

// Function 574
ModelSpec specForStep(vec3 p, float x, float scale) {
    float move = moveAnim(x) * stepMove;
    float sizeScale = mix(1., stepScale, wobbleScaleAnim(x));
    float sizeScaleCore = mix(1., stepScale, scaleAnim(x));
    float bounds = boundsForStep(p, move, sizeScale, scale);
    float level = levelStep(p / scale, move, sizeScale * ballSize, x);
    return ModelSpec(move, sizeScale, sizeScaleCore, bounds, level);
}

// Function 575
float RayMarch(vec3 ro, vec3 rd) 
{
	float dO=0.;
    
    for(int i=0; i<MAX_STEPS; i++) 
	{
    	vec3 p = ro + rd*dO;
        float dS = GetDist(p);
        dO += dS;
        if(dO>MAX_DIST || dS<SURF_DIST) break;
    }
    
    return dO;
}

// Function 576
vec3 ray_march(vec3 p, vec3 r)
{
    float td = 0.;
    int i;
    for(i = 0; i < MAX_MARCHES; i++)
    {
    	float de = map(p);
        if(de < rayfov*td || td > MAX_DIST)
        {
            break;
        }
        p += de*r;
        td += de;
    }
    vec3 col =sdcolor(p);
    if(td > MAX_DIST || i > MAX_MARCHES)
    {
        col = sky_color(r);
    }
    else
    {
        vec4 norm = calcNormal(p,MIN_DIST);
        p += r*(norm.w-rayfov*td);
        #ifdef SHADOWS
      		float shad = shadow_march(vec4(p+norm.xyz*0.05,0.), vec4(light,0.), 400.,0.03);
        #else
        	float shad = 1.;
        #endif
        
        col = lighting(vec4(col,0.), vec2(0.2,0.3), vec4(p, 0.), vec4(r, td), norm, shad);
    }
   
    return col;

}

// Function 577
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

// Function 578
bool nextStepIsFinal(inout int steps, in int maxSteps) {
    steps++;
    return steps >= maxSteps;
}

// Function 579
vec3 marchVol( in vec3 ro, in vec3 rd, in float t, in float mt )
{
	vec4 rz = vec4(0);
    #if 1
    t -= (dot(rd, vec3(0,1,0))+1.);
    #endif
	float tmt = t +15.;
	for(int i=0; i<25; i++)
	{
		if(rz.a > 0.99)break;

		vec3 pos = ro + t*rd;
        float r = mapVol( pos,.1 );
        float gr =  clamp((r - mapVol(pos+ vec3(.0,.7,0.0),.1))/.3, 0., 1. );
        vec3 lg = vec3(0.72,0.28,.0)*1.2 + 1.3*vec3(0.55, .77, .9)*gr;
        vec4 col = vec4(lg,r*r*r*2.5); //Could increase this to simulate entry
        col *= smoothstep(t-0.0,t+0.2,mt);
        
        pos.y *= .7;
        pos.zx *= ((pos.y-5.)*0.15 - 0.4);
        float z2 = length(vec3(pos.x,pos.y*.75 - .5,pos.z))-.75;
        col.a *= smoothstep(.4,1.2,.7-map2(vec3(pos.x,pos.y*.17,pos.z)));
		col.rgb *= col.a;
		rz = rz + col*(1. - rz.a);
		
        t += abs(z2)*.1 + 0.12;
        if (t>mt || t > tmt)break;
	}
	
    rz.g *= rz.w*0.9+0.12;
    rz.r *= rz.w*0.5+0.48;
	return clamp(rz.rgb, 0.0, 1.0);
}

// Function 580
vec4 march(vec3 cam, vec3 n)
{
    
    float len = 1.0;
    vec4 ret;
    
    for(int i = 0; i < MARCHLIMIT; i++)
    {
        ret = range(camPos + len*n)*0.5;
		len += ret.w;
    }
    
	return vec4(ret.xyz, len);
}

// Function 581
vec2 linearStep2(vec2 mi, vec2 ma, vec2 v)
{
    return clamp((v - mi) / (ma - mi), 0.0 ,1.0);
}

// Function 582
vec2 rayMarch(in vec3 ro, in vec3 rd, float tol, float tmin, float tmax) {
	float t = tmin;
	float m = -1.0;
    
    for (int i=0; i<60; i++) {
		vec2 res = map(ro+rd*t);
        m = res.y;
		if (res.x < precis || t > tmax)  break;
		t += res.x*tol;
	}

	if (t > tmax) {
		m = -1.0;
	}
	return vec2(t, m);
}

// Function 583
bool marchRay(vec3 startPos, vec3 dir, out vec3 color) {
  vec3 p = startPos;
  bool checksteps = keypress(CHAR_Q);
  for (int i = 0; i < MAXSTEPS; i++) {
    assert(i < MAXSTEPS-1);
    if (checksteps) assert(i < 200);
    if (length(p) > MAX_DISTANCE) return false;
    float dist = sceneDf(p);
    if (dist <= MIN_DISTANCE) break;
    // Proceed cautiously when "close" to surface
    if (dist > limit) dist = dist-limit+slow;
    else dist = slow*dist;
    p += dist*dir;
  }
  vec3 baseColor = vec3(0.8);
  if (!keypress(CHAR_U)) baseColor = triplanar(normalize(p),sampler).xyz;
  if (keypress(CHAR_A)) {
    // Show axes
    float d = min(abs(p.x),min(abs(p.y),abs(p.z)));
    if (d < 0.02) baseColor = vec3(0.5,0,0);
  }
  color = processLighting(baseColor,dir,p);
  return true;
}

// Function 584
vec3 raymarch(in vec3 from, in vec3 dir) 

{
	float ey=mod(t*.5,1.);
	float glow,eglow,ref,sphdist,totdist=glow=eglow=ref=sphdist=0.;
	vec2 d=vec2(1.,0.);
	vec3 p, col=vec3(0.);
	vec3 origdir=dir,origfrom=from,sphNorm;
	
	//FAKING THE SQUISHY BALL BY MOVING A RAY TRACED BALL
	vec3 wob=cos(dir*500.0*length(from-pth1)+(from-pth1)*250.+iTime*10.)*0.0005;
	float t1=Sphere(from-pth1+wob,dir,0.015);
	float tg=Sphere(from-pth1+wob,dir,0.02);
	if(t1>0.){
		ref=1.0;from+=t1*dir;sphdist=t1;
		sphNorm=normalize(from-pth1+wob);
		dir=reflect(dir,sphNorm);
	} 
	else if (tg>0.) { 
		vec3 sphglowNorm=normalize(from+tg*dir-pth1+wob);
		glow+=pow(max(0.,dot(sphglowNorm,-dir)),5.);
	};
	
	for (int i=0; i<RAY_STEPS; i++) {
		if (d.x>det && totdist<3.0) {
			p=from+totdist*dir;
			d=de(p);
			det=detail*(1.+totdist*60.)*(1.+ref*5.);
			totdist+=d.x; 
			energy=ENERGY_COLOR*(1.5+sin(iTime*20.+p.z*10.))*.25;
			if(d.x<0.015)glow+=max(0.,.015-d.x)*exp(-totdist);
			if (d.y<.5 && d.x<0.03){//ONLY DOING THE GLOW WHEN IT IS CLOSE ENOUGH
				float glw=min(abs(3.35-p.y-ey),abs(3.35-p.y+ey));//2 glows at once
				eglow+=max(0.,.03-d.x)/.03*
				(pow(max(0.,.05-glw)/.05,5.)
				+pow(max(0.,.15-abs(3.35-p.y))/.15,8.))*1.5;
			}
		}
	}
	float l=pow(max(0.,dot(normalize(-dir.xz),normalize(lightdir.xz))),2.);
	l*=max(0.2,dot(-dir,lightdir));
	vec3 backg=.5*(1.2-l)+LIGHT_COLOR*l*.7;
	backg*=AMBIENT_COLOR;
	if (d.x<=det) {
		vec3 norm=normal(p-abs(d.x-det)*dir);//DO THE NORMAL CALC OUTSIDE OF LIGHTING (since we already have the sphere normal)
		col=light(p-abs(d.x-det)*dir, dir, norm, d.y)*exp(-.2*totdist*totdist); 
		col = mix(col, backg, 1.0-exp(-1.*pow(totdist,1.5)));
	} else { 
		col=backg;
	}
	vec3 lglow=LIGHT_COLOR*pow(l,30.)*.5;
	col+=glow*(backg+lglow)*1.3;
	col+=pow(eglow,2.)*energy*.015;
	col+=lglow*min(1.,totdist*totdist*.3);
	if (ref>0.5) {
		vec3 sphlight=light(origfrom+sphdist*origdir,origdir,sphNorm,2.);
		col=mix(col*.3+sphlight*.7,backg,1.0-exp(-1.*pow(sphdist,1.5)));
	}
	return col; 
}

// Function 585
bool rayMarching(in vec3 origin, in vec3 ray, out vec3 m, out int material) {
    
    float	marchingDist = 0.0;
    float 	nbIter 		 = 0.0;

    for(int i = 0; i<1000; i++) {
        
        m = origin + ray * marchingDist;    
        
    	float dist = map(m, material);
        
        if(dist < 0.000001) {
            return true;
        }
        else {
            marchingDist += dist;
            
            if(marchingDist >= 500.) {
                break;
            }
        }
    }
    
	return false;    
}

// Function 586
vec3 raymarch_flopine(vec3 ro, vec3 rd, vec2 uv)
{
	vec3 col;
	float dither = random(uv);
	float t = 0.;
	vec3 p;// = ro;
	for (float i = 0.; i < 80.; i++)
	{
		p = ro + t * rd;
		float d = SDF(p);
		if (d < 0.001)
		{
			col = vec3(i / 80.);
			break;
		}
		d *= 1. + dither * 0.1;

		t += d * .8;
	}

	float g2_force = mix(0., 0.8, smoothstep(10., 14., time) * (1. - smoothstep(116., 120., time)));
	col += g1 * vec3(0.2, 0.4, 0.);
	col += (g2* g2_force) * vec3(0., 0.5, 0.5);
	col = mix(col, vec3(0., 0.3, 0.4), 1. - exp(-0.001*t*t));

	return col;
}

// Function 587
bool rayMarch(vec3 ro, vec3 rd, vec2 iMouse, float iTime, out float t) {
    t = 1e-4;
    for(int i = 0; i < 18; i++) {
        vec3 p = ro + t*rd;
        float h = map(p, iMouse, iTime);
        if( abs(h) < 1e-5) {
            return true;
        }

        if (t > maxT) {
            return false;
        }
        t += h;
    }
    return true;
}

// Function 588
void SimulationStep(inout particle U)
{
    vec4 border = border_grad(U.pos.xyz);
    vec3 cvec = -U.pos.xyz*vec3(0,0,1);
    vec3 G = 0.15*normalize(cvec)/size3d;
   
    vec3 force =calc_force(U.pos.xyz);
    vec3 bound =1.*normalize(border.xyz)*exp(-0.4*border.w*border.w);
    float cooling = 1. - (1.-exp(-0.3*length(U.vel.xyz)))*(0.01*exp(-0.05*dot(cvec,cvec)) + 0.01*exp(-0.4*border.w*border.w) + 0.04*exp(-0.1*dot(force,force)));
    U.vel.xyz =  U.vel.xyz*cooling + dt*(bound+force+G);
    U.pos.xyz += dt*U.vel.xyz;
}

// Function 589
vec2 raymarch(vec3 ro, vec3 rd, int iter) {
    float  t = 0.;
    float id = -1.;
    for(int i = 0; i < MAX_ITERATIONS; i++) {
        if(t >= FAR_PLANE || i >= iter) {
            id = -1.;
            break;
        }
        vec2 scn = dstScene(ro+rd*t);
        if(scn.x < EPSILON) {
            id = scn.y;
            break;
        }
        t += scn.x * .75;
    }
    return vec2(t,id);
}

// Function 590
vec3 raymarch_main_normal(vec3 _p, float eps, float t)
{
    vec3 n;
    n.y = raymarch_main_scene_normals(_p, t).x;
    n.x = raymarch_main_scene_normals(_p+vec3(eps, 0., 0.), t).x-n.y;
    n.z = raymarch_main_scene_normals(_p+vec3(0., 0., eps), t).x-n.y;
    n.y = raymarch_main_scene_normals(_p+vec3(0., eps, 0.), t).x-n.y;
    return normalize(n);
}

// Function 591
vec2 rayMarch(in vec3 u,vec3 t,float l,float m,float zFar)
{float g = -1.//matrrial
;for(float i=.0;i<iterRm;i++
){vec2 r=gd(u+t*m);g=r.y
 ;if(r.x<eps||m>zFar)break
 ;m+=r.x*l
;}
;g=mix(-1.,g,step(m,zFar));
;return vec2(m,g);}

// Function 592
Hit ray_march(vec3 rayfrom, vec3 raydir) {
    // begin at ray origin
    float t = 0.0;
    Hit hit;
    // ray march loop
    for(int i=0; i<MAX_STEPS; ++i) {
        // compute next march point
        vec3 p = rayfrom+t*raydir;
        // get the distance to the closest surface
        hit = map(p);
        // hit a surface
        if(abs(hit.d) < (SURF_DIST*t))
            break;
        // increase the distance to the closest surface
        t += hit.d;
        if(t > MAX_DIST) {
            hit.material = 0;
            break;
        }
    }
    // return the distance to `rayfrom`
    hit.d = t;
    return hit;
}

// Function 593
HitInfo rayMarch(vec3 o, vec3 d)
{
    const float tMax = 75.0;
    float t = 0.0;
    for(int i=0; i < 40; ++i)
    {
        itCount += 1.0;
        float d = map(o+t*d).d;
        t += d>0.?d:0.75*d;
        if(abs(d)<0.001 || t > tMax)
            break;
    }
    
    HitInfo info = map(o+t*d);
    info.matID = (t>tMax)?MAT_SKY:info.matID;
    info.d = min(t,tMax);
    return info;
}

// Function 594
float march(in vec3 ro, in vec3 rd, out float drift, in vec2 scUV)
{
	float precis = 0.01;
    float h=precis*2.0;
    float d = hash12(scUV);
    drift = 0.0;
    for( int i=0; i<ITR; i++ )
    {
        vec3 p = ro+rd*d;
        if(h < precis || d > FAR) break;
        h = map(p);
        drift +=  fogmap(p, d);
        d += min(h*.65 + d * .002, 8.0);
	 }
    drift = min(drift, 1.0);
	return d;
}

// Function 595
float stepUp(float t, float len, float smo)
{
  float tt = mod(t += smo, len);
  float stp = floor(t / len) - 1.0;
  return smoothstep(0.0, smo, tt) + stp;
}

// Function 596
float rayMarchMushrooms(in vec3 iOrigin, in vec3 iRay) {
    
 	float t = 0.0;
    vec3 p = iOrigin;
    for (int i = 0; i < MAX_STEPS; i++) {
        p = iOrigin + t*iRay;
     	float sdist = allMushroomsSDF(p);
        t += sdist;
        if (abs(sdist) < EPSILON) {
            break;
        }
    }
    return t;
    
}

// Function 597
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

// Function 598
float levelStep(vec3 p, float move, float size, float x) {
    float transition = smoothstep(0., .1, x);
    float blend = hardstep(move + size, size, length(p));
    blend = mix(0., blend, transition);
    return blend;
}

// Function 599
vec3 raymarch( in vec3 ro, vec3 rd, vec2 tminmax )
{
    float t = tminmax.x;
    float dt = 0.02;
    if (animate_pattern)
      dt = 0.02 + 0.01*cos(time*0.5);

    vec3 col= vec3(0.);
    float c = 0.;
    for( int i=0; i<64; i++ )
    {
        t+=dt*exp(-2.*c);
        if(t>tminmax.y)break;
        vec3 pos = ro+t*rd;
        c = 0.45 * map(ro+t*rd);
        col = 0.98*col + 0.08*vec3(c*c, c, c*c*c);  // green
        col = 0.99*col + 0.08*vec3(c*c*c, c*c, c);  // blue
        col = 0.99*col + 0.08*vec3(c, c*c*c, c*c);  // red
    }
    return col;
}

// Function 600
vec3 marchRay (vec3 ro, vec3 rs)
{
    float d = .0;
    float m;
    vec3 r;
    for (int i = 0; i <= 150 && d <= 10.; i++)
    {
        vec3 r = ro + rs*d;
        m = dist3D(r);
        if (m <= 0.001) return r;
        d += m*.7;
    }
}

// Function 601
void MarchPOV(inout RayInfo r, float startTime)
{
    //dpos = vec3(-2.2,0,0);
    //lorentzF = LorentzFactor(length(dpos));
    
    float speedC = length(dpos)/SpeedOfLight;
    vec3 nDpos = vec3(1,0,0);
    
    if (length(dpos) > 0.)
    	nDpos = normalize(dpos);
    
    //shrink space along vel axis (length contraction of field of view)
    float cirContraction = dot(nDpos,r.dir)*(LorentzFactor(length(LgthContraction*dpos)));
    vec3 newDir = (r.dir - nDpos*dot(nDpos,r.dir)) + cirContraction*nDpos;
    r.dir = normalize(newDir);
    
    float dDirDpos = dot(dpos,r.dir);
    
    // Aberration of light, at high speed (v) photons angle of incidence (a) vary with lorenz factor (Y) :
    // tan(a') = sin(a)/(Y*(v/c + cos(a)))
    // velComponentOfRayDir' = Y*(velComponentOfRayDir+v/c)
    float lightDistortion = lorentzF*(dot(-nDpos,r.dir)+speedC);
    r.dir = mix(
        r.dir,
        normalize((r.dir - nDpos*dot(nDpos,r.dir)) - lightDistortion*nDpos),
        FOVAberrationOfLight);
    
    //Classical Newtown Mechanic instead would be :
    //r.dir = normalize(r.dir-dpos/SpeedOfLight);
    
    
    for (r.iter=0;r.iter<maxStepRayMarching;r.iter++){
        
        float camDist = length(r.pos - objects[o_cam].pos);
        
        float photonDelay = -photonLatency*camDist/SpeedOfLight;
        
        //takes dilated distance x/Y and find the time in map frame with :
        // v = -dDirDpos (dot product of direction & velocity, because we want to transform from cam frame to map frame)
        // Y = lorentzFactor
        //
        // t' = Y(t-v*(x/Y)/c²)
        // t' = Y(0-v*(x/Y)/c²)
        // t' = Y(v*x/Y)/c²
        // t' = vx/c²
        float relativeInstantEvents = SimultaneousEvents*dDirDpos*camDist/(SpeedOfLight*SpeedOfLight);
        
    	r.time = startTime;
        r.time += mix(relativeInstantEvents,photonDelay,photonLatency);
        
        
        
    	SetTime(r.time);
        r.dist = map(r.pos,-1);
        
        
        //blackhole Gravitational lens effect
        vec3 blackHoleDirection = (objects[o_blackHole].pos-r.pos);
        r.dir += (1./RayPrecision)*r.dist*normalize(blackHoleDirection)*BlackHoleMassFactor/(length(blackHoleDirection)*SpeedOfLight*SpeedOfLight);
        r.dir = normalize(r.dir);
        
        if(abs(r.dist)<rayEps)
            break;
        r.pos+= (1./RayPrecision)*(r.dist)*(r.dir); //
    }
    
    //r.pos = origin + r.dir*min(length(r.pos-origin),maxDist);
    
    r.surfaceNorm = GetNormal(r.pos).xyz;
}

// Function 602
float boundsForStep(vec3 p, float move, float sizeScale, float scale) {
    float overfit = .3;
    p /= scale;
    float d = (length(p) - move - ballSize * sizeScale - overfit);
    d *= scale;
    return d;
}

// Function 603
vec3 march(inout vec3 p, vec3 rd) {
	float i,
	      d = .01;
	bool addOrb = true;
	Hit h, orb;
	vec3 orbP, c;
	g = 0.;
	for (i = Z0; i < 128.; i++) {
		h = map(p, addOrb);
		if (abs(h.d) < .0015) {
			if (h.id == 0) {
				orb = h;
				orbP = p;
				addOrb = false;
			}
			else break;
		}

		d += h.d;
		if (d > 64.) break;
		p += h.d * rd;
	}

	c = (d < 64. ? lights(p, rd, d, h) : sky(rd)) + g;
	if (!addOrb) c += lights(orbP, rd, d, orb);
	return c;
}

// Function 604
vec4 rayMarch(vec3 p, vec3 d, inout vtx co)
{    
    float td = 0.; float DE = 1e10;
    for(int i = min(0, iFrame); i < maxs; i++)
    {
        //march
        DE = map(p, d, co);
        
        p += DE*d;
        td += DE;
        
        //outide of the scene
        if(td > maxd) return vec4(p, -1.);
        //has hit the surface
        if(DE < mind*td)
        {
            p += - mind*td*d;
            break;
        }
    }
    return vec4(p, DE);
}

// Function 605
float rayMarch(in vec3 iOrigin, in vec3 iRay, inout int ioObjectHit) {
    
 	float t = 0.0;
    vec3 p = iOrigin;
	
    for (int i = 0; i < MAX_STEPS; i++) {
        p = iOrigin + t*iRay;
     	float m = allMushroomsSDF(p);
     	float o = allOrbsSDF(p);
     	float g = groundSDF(p);
		
		float depth = min(min(o, m), g);
		// We hit an orb
		if (o < m && o < g) {
			depth = o;
			ioObjectHit = 2;
		} 
		// We hit a mushroom
		if (m < o && m < g) {
			depth = m;
			ioObjectHit = 1;
		}
		// We hit the ground
		if (g < o && g < m) {
			depth = g;
			ioObjectHit = 0;
		}		
        t += depth; 
        if (abs(depth) < EPSILON) {
            break;
        }
    }
	
    return t;
    
}

// Function 606
vec3 Raymarch(Ray r, float startT)
{
    float t = startT;
    float d = 0.0;
    float iterations = 0.0;
    
	for(int j = 0; j < MAX_ITERATIONS; j++)
	{
		d = sdf(r.origin + r.direction * t, true);

		if(d < EPSILON)
            break;
        
		t += d;
        
        if(t > MAX_DISTANCE)
            break;
        
        iterations += 1.0;
	}
    
    t = min(t, MAX_DISTANCE);
    
    return vec3(t, iterations / float(MAX_ITERATIONS), d);
}

// Function 607
void marchRay(inout Ray ray, inout vec4 colour) {
    bool inside = false; // are we inside or outside the glass object
    vec4 impact = vec4(1.0); // This decreases each time the ray passes through glass, darkening colours

#ifdef DEBUG   
vec4 debugColour = vec4(1, 0, 0, 1);
#endif
    
    SDResult result;
    vec3 n;
    vec3 glassStartPos;
    
    for (int i=0; i<kMAXITERS; i++) {
        // Get distance to nearest surface
        result = sceneDist(ray);
        
        // Step half that distance along ray (helps reduce artefacts)
        float stepDistance = inside ? abs(result.d) : result.d;
            //result.material == kGLASSMATERIAL ? abs(result.d) : result.d;
        ray.origin += ray.dir * stepDistance * 0.5;
        if (length(ray.origin) > 30.0) { break; }
        
        if (stepDistance < eps) {
            // colision
            // normal
            // Get the normal, then clamp the intersection to the surface
    		n = normal(ray);
            clampToSurface(ray, stepDistance, n);
#ifdef DEBUG
//debugColour = vec4(-n*1.0, 1);
//debugStep++;
//if (debugStep == 3) break;
          //  break;
//if (mod(ray.origin.y, 1.0) > 0.5) break;
//debugValue += 0.25;
#endif
            
            if ( result.material == kFLOORMATERIAL ) {
                // ray hit floor
                
                // Add some noise to the normal, since this is pretending to be grit...
                vec3 randomNoise = texrand(ray.origin.xz * 0.4);
                n = mix(n, normalize(vec3(randomNoise.x, 1, randomNoise.y)), randomNoise.z);
                
                // Colour is just grey with crappy fake lighting...
                colour += mix(
                    kFLOORCOLOUR, 
                    vec4(0,0,0,1), 
                    pow(max((-n.x+n.y) * 0.5, 0.0), 2.0)
                ) * impact;
                impact *= 0.;
                break;
            }
            
            // check what material it is...
            
            if (result.material == kMIRRORMATERIAL) {
                // it's a mirror, reflect the ray
                ray.dir = reflect(ray.dir, n);
                    
                // Step 2x epsilon into object along normal to ensure we're beyond the surface
                // (prevents multiple intersections with same surface)
                ray.origin += n * eps * 2.0;
                
                // Mix in the mirror colour
                impact *= kMIRRORCOLOUR;
                
            } else {
                // glass material
            
                if (inside) {
                	// refract glass -> air
                	ray.dir = refract(ray.dir, -n, 1.0/kREFRACT);
                    
                    // Find out how much to tint (how far through the glass did we go?)
                    float glassTravelDist =  clamp(distance(glassStartPos, ray.origin) / 1.0, 0., 1.);
                    
                    // Get a random colour
                	impact *= mix(vec4(1), kGLASSCOLOUR, glassTravelDist);
                    
#ifdef DEBUG
debugValue += glassTravelDist / 2.0;
#endif
      
                
              	} else {
               		// refract air -> glass
                	glassStartPos = ray.origin;
                    
              	  	// Mix the reflection in, according to the fresnel term
                	float fresnel = fresnelTerm(ray, n, 2.0);
    				colour = mix(
                    	colour, 
                    	texture(iChannel1, reflect(ray.dir, n)), 
                    	vec4(fresnel) * impact);
                    impact *= 1.0 - fresnel;
    			
                	// refract the ray
            		ray.dir = refract(ray.dir, n, kREFRACT);
                    
#ifdef DEBUG
//debugValue += 0.5;
#endif
                }
            
            	// Step 2x epsilon into object along normal to ensure we're beyond the surface
                ray.origin += (inside ? n : -n) * eps * 2.0;
                
                // Flip in/out status
                inside = !inside;
            }
        }
        
        // increase epsilon
        eps += divergence * stepDistance;
    }
    
    // So far we've traced the ray and accumulated reflections, now we need to add the background.
    colour += texture(iChannel0, ray.dir) * impact;
    
    
#ifdef DEBUG
//debugColour.rgb = ray.dir;
debugColour.rgb = vec3(float(debugStep)/2.0);
colour = debugColour;
#endif
}

// Function 608
vec4 raymarch( vec3 ro, vec3 rd, vec3 bgcol, ivec2 px )
{
	vec4 sum = vec4(0);
	float dt = .01,
         den = 0., _den, lut,
           t = intersect_sphere( ro, rd, BS.xyz, BS.w );
    if ( t == -1. ) return vec4(0); // the ray misses the object 
    t += 1e-5;                      // start on bounding sphere
    
    for(int i=0; i<500; i++) {
        vec3 pos = ro + t*rd;
        if(   sum.a > .99               // end if opaque or...
           || length(pos-BS.xyz) > BS.w ) break; // ... exit bounding sphere
                                    // --- compute deltaInt-density
        lod = 1.*abs( pos.z - sin(4.*iTime) ); // *** tune Depth of Field $DOF ***
        dt = max(.01, .05*lod);     // *** use larger steps where $DOF blurry ***
        _den = den; den = map(pos); // raw density
        float _z = z;               // depth in object
        lut = LUTs( _den, den );    // shaped through transfer function
        if( lut > .0                // optim
          ) {                       // --- compute shading                  
#if 0                               // finite differences
            vec2 e = vec2(.3,0);
            vec3 n = normalize( vec3( map(pos+e.xyy) - den,
                                      map(pos+e.yxy) - den,
                                      map(pos+e.yyx) - den ) );
         // see also: centered tetrahedron difference: https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
            float dif = clamp( -dot(n, sundir), 0., 1.);
#else                               // directional difference https://www.iquilezles.org/www/articles/derivative/derivative.htm
         // float dif = clamp((lut - LUTs(_den, map(pos+.3*sundir)))/.6, 0., 1. ); // pseudo-diffuse using 1D finite difference in light direction 
            float dif = clamp((den - map(pos+.3*sundir))/.6, 0., 1. );             // variant: use raw density field to evaluate diffuse
#endif
/*
            vec3  lin = vec3(.65,.7,.75)*1.4 + vec3(1,.6,.3)*dif,          // ambiant + diffuse
                  col = vec3(.2 + dif);
            col = mix( col , bgcol, 1.-exp(-.003*t*t) );   // fog
*/            
            vec3 S = pos.x < 0. ? vec3(3,3,2) : vec3(2,3,3);
            vec3 col = exp(-S *(1.-z));                   // dark with shadow
         // vec3 col =   exp(- vec3(3,3,2) *(.8-_z));     // dark with depth
                   //      *  exp(- 1.5 *(1.-z));
            sum += (1.-min(1.,sum.a)) * vec4(col,1)* (lut* dt*5.); // --- blend. Original was improperly just den*.4;
        }
        t += dt;  // stepping
    }

    return sum; 
}

// Function 609
float reflections_ray_marching( vec3 origin, vec3 dir, float start, float end ) {
	
    float depth = start;
	for ( int i = 0; i < max_iterations; i++ ) {        
		float dist = coolThing( ( origin + dir * depth )/2.0, 1.7)*2.0;
        dist -= dist*0.95;
		if ( dist < stop_threshold ) {
			return depth;
		}
		depth += dist;
		if ( depth >= end) {
			return end;
		}
	}
	return end;
}

// Function 610
vec3 march( in vec2 p )
{
    vec4 h = vec4(0.0);
	for( int i=0; i<24; i++ )
    {
        h = map(p);
        if( h.x<0.001 ) break;
        p = p + h.x*randomInCircle();
    }
    return h.yzw;
}

// Function 611
float march (in vec3 ro, in vec3 rd) {
    float t = 0.01;   
    for (int i = 0; i < RAY_STEPS; i++) {
        vec3 pos = ro + t * rd;
        float m = map(pos);
        if (m < 0.01) {
            return t;
        }
        t += m;
    }
    
	return -1.0;    
}

// Function 612
vec3 march(vec3 ro, vec3 rd) {
	// Raymarch.
	vec3 p, c;
	float gg,
	      d = .01;
	Hit h;
	for (float i = 0.; i < 120.; i++) {
		p = ro + rd * d;
		h = map(p);
		if (abs(h.d) < .0015 || d > 6e2) break;
		d += h.d; // No hit, so keep marching.
	}

	gg = g; // Cache the 'glow'.
	if (d > 6e2) c = vec3(.85, .9, 1);
	else c = mix(lights(p, rd, d, h), vec3(1), smoothstep(2e2, 540., d));

	c += gg * vec3(0, 1, 0);
	if (h.id == 3 || h.id == 1) {
		// Reflections applied to cockpit glass and tie metal.
		rd = reflect(rd, calcN(p, d));
		float alpha = (h.id == 3 ? .4 : .2) * smoothstep(0., 1., -rd.y);
		if (alpha < .001) return c; // Only reflect downwards.
		d = .01;
		ro = p;
		for (float i = 0.; i < 40.; i++) {
			p = ro + rd * d;
			h = sdTerrain(p);
			if (abs(h.d) < .0015 || d > 3e2) break;
			d += h.d; // No hit, so keep marching.
		}

		// Combine a % of the reflected color.
		c = mix(c, d > 3e2 ? vec3(1) : lights(p, rd, d, h), alpha);
	}

	return c;
}

// Function 613
vec4 raymarch( inout vec3 p, inout vec3 dir, out int out_steps, out float dmin )
{
    int iter = 0;
	vec4 d;
	float rdt = 0.0;
    dmin = 100000.0;
	for ( int i=0; i<NUM_ITERATIONS; i++ )
	{
        iter += 1;
		d = scene( p );

        dmin = min( dmin, d.x );
        
		if ( (d.x < 0.0 ) || (rdt > FAR_CLIP) ) {
			break;
		}
		else
		{
			float dt = 0.01 + STEP_MULT * d.x; //note: constant-multiply to compensate for distorted space, actual dist < dist - could use gradient-approximation instead? (see iq)
			p += dir * dt;
			rdt += dt;
		}
	}

	out_steps = iter;
	return d;

}

// Function 614
void march(vec3 origin, vec3 dir, out float t, out int objId, vec3 lightPos)
{
    t = 0.001;
    for(int i = 0; i < RAY_STEPS; ++i)
    {
        vec3 pos = origin + t * dir;
        float m;
        sceneMap3D(pos, m, objId, lightPos);
        if(m < 0.01)
        {
            return;
        }
        t += m;
    }

    t = -1.0;
    objId = -1;
}

// Function 615
float smootherstep(float x)
{
	x = clamp(x, 0.0, 1.0); // optional
	return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

// Function 616
float smoothStep(float x) {
    return x * x * (3. - 2. * x);
}

// Function 617
Hit march_geometry(const in Ray ray, float start, float end, const in WORLD w) {
    float t = start;
    for(int i=0; i<MAX_GEOMETRY_MARCHING_STEPS; i++){
        Hit hit = world(ray.origin + t * ray.direction, ray, w);
        if( hit.dist < EPS ) return Hit(t, hit.matID, hit.normal);
        t += hit.dist * .75;
    }
    return miss;
}

// Function 618
float RayMarch( in vec3 ro, in vec3 rd )
{
	float precis = 0.0005;
    float h		 = precis*.2;
    float t		 = 0.01;
	float res	 = 2000.0;
	bool hit	 = false;

    for( int i=0; i< 150; i++ )
    {
		if (!hit && t < 8.0)
		{
			h = de(ro + rd * t);
			t += h * .8;
			if (h < precis)
			{
				res = t;
				hit = true;;
			}
			precis *= 1.03;
		}
    }
	
    return res;
}

// Function 619
vec3 raymarch(in vec3 from, in vec3 dir) 

{
	hitcar=0.;
	ref=0.;
	float totdist=0.;
	float glow=0.;
	float d=1000.;
	vec3 p=from, col=vec3(0.5);

	float deta=DETAIL*(1.+backcam); // lower detail for HUD cam
	vec3 carp=vec3(0.); // coordinates for car hit
	vec3 carn=vec3(0.); // normal for car
	float cardist=0.; // ray length for car
	vec3 odir=dir; // save original ray direction

	for (int i=0; i<RAY_STEPS; i++) {
		if (d>det && totdist<MAX_DIST) {
			d=de(p);
			p+=d*dir;
			det=max(deta,deta*totdist*.5*(1.+ref)); // scale detail with distance or reflect
			totdist+=d; 
			float gldist=det*8.; // background glow distance 
			if(d<gldist&&totdist<20.) glow+=max(0.,gldist-d)/gldist*exp(-.1*totdist); //accum glow
#ifndef LOW_QUALITY
			if (hitcar>0. && ref<1.) { // hit car, bounce ray (only once)
				p=p-abs(d-det)*dir; // backstep
				carn=normal(p); // save car normal
				carp=p; // save car hit pos
				dir=reflect(dir,carn); // reflect ray
				p+=det*dir*10.; // advance ray
				d=10.; cardist=totdist;
				ref=1.;
			}
#endif
		} 
#ifdef LOOP_BREAKS		
		else break;
#endif
	}

	tubeinterval=abs(1.+cos(p.z*3.14159*.5))*.5; // set light tubes interval
	float cglow=1./(1.0+minL*minL*5000.0); // car glow
	float tglow=1./(1.0+minT*minT*5000.0); // tubes glow
	float l=max(0.,dot(normalize(-dir),normalize(LIGHTDIR))); // lightdir gradient
	vec3 backg=AMBIENT_COLOR*.4*max(0.1,pow(l,5.)); // background
	float lglow=pow(l,50.)*.5+pow(l,200.)*.5; // sun glow

	if (d<.5) { // hit surface
		vec3 norm=normal(p); // get normal
		p=p-abs(d-det)*dir; // backstep
		col=shade(p, dir, norm); // get shading 
		col+=tglow*TUBE_COLOR*pow(tubeinterval,1.5)*2.; // add tube glow
		col = mix(backg, col, exp(-.015*pow(abs(totdist),1.5))); // distance fading

	} else { // hit background
		col=backg; // set color to background
		col+=lglow*SUN_COLOR; // add sun glow
		col+=glow*pow(l,5.)*.035*LIGHT_COLOR; // borders glow
		
#ifdef LOW_QUALITY
		vec3 st = (dir * 3.+ vec3(1.3,2.5,1.25)) * .3;
		for (int i = 0; i < 14; i++) st = abs(st) / dot(st,st) - .9;

		col+= min( 1., pow( min( 5., length(st) ), 3. ) * .0025 ); // add stars
#else
		float planet=Sphere(planetpos,dir, 2.); // raytrace planet

		// kaliset formula - used for stars and planet surface 
		float c;
		if (planet>0.) c=1.; else c=.9; // different params for planet and stars
		vec3 st = (dir * 3.+ vec3(1.3,2.5,1.25)) * .3;
		for (int i = 0; i < 14; i++) st = abs(st) / dot(st,st) - c;

		col+= min( 1., pow( min( 5., length(st) ), 3. ) * .0025 ); // add stars
		
		// planet atmosphere
		col+=PLANET_COLOR*pow(max(0.,dot(dir,normalize(-planetpos))),100.)*150.*(1.-dir.x);
		// planet shading
		if (planet>0.) col=shadeplanet(planet*dir,st);
#endif
		
	}
	// car shading

		// add turbine glows
	
#ifdef LOW_QUALITY
	cglow*=1.15;
#else
	if (ref>0.) {
		ref=0.;
		col=shade(carp,odir,carn)+col*.3; // car shade + reflection
		// I wanted a lighter background for backward reflection
		l=max(0.,dot(normalize(-odir),normalize(LIGHTDIR)));
		backg=AMBIENT_COLOR*.4*max(0.1,pow(l,5.)); 
		col = mix(backg, col,exp(-.015*pow(abs(cardist),1.5))); // distance fading
	}
#endif 

	
	col+=TURBINES_COLOR*pow(abs(cglow),2.)*.4;
	col+=TURBINES_COLOR*cglow*.15;


	return col; 
}

// Function 620
float RayMarch(vec3 ro, vec3 rd, float fft) {
	float dO=-333.;
    float SURF_DIST = fft * 13.;
    if(iTime>9.8){
        SURF_DIST = 0.;
    }
    if(iTime>22.95){
        SURF_DIST = fft;
    }

    if(iTime>70.06){
        SURF_DIST = sin(iTime)*10.;
    }
    
    if(iTime>78.06){
        SURF_DIST = fft;
    }
    if(iTime>104.){
        SURF_DIST = 2.0 + sin(iTime * 0.5);
    }
    if(iTime>218.){
        SURF_DIST = 100.;
    }
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        float dS = GetDist(p);
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }
    return dO;
}

// Function 621
float smoothstepLine(float lower, float upper, float value, float width){
    width *= 0.5;
    return smoothstep(lower - width, lower, value) * (1.0 - smoothstep(upper, upper + width, value));
}

// Function 622
vec4 march(ray r)
{
    vec3 c = vec3(0.);
    for(int i = 0; i < ITER; i++)
    { 
        if(r.l > MAXD ) 
        {
            c = BGC * 2.;
            break;
        }
        vec3 p = r.o + r.d * r.l;
        vec4 g = geo(p);
        if(g.w < SURD) 
        {
            c = g.rgb * 2.5;
            break;
        }
        r.l += g.w;
    }
    return vec4(c, r.l);
}

// Function 623
float RayMarchStepCount(vec3 ray_origin, vec3 ray_dir, float initial_dist, float iTime)
{
    float dist = initial_dist;
    
    int i = 0;
    for (; i < MAX_STEPS; ++i)
    {
        vec3 p = ray_origin + ray_dir * dist;
        
        float nearest = GetSceneDist(p, iTime);
        
        if (nearest < MIN_DIST)
            break;
        
        dist += nearest;

        if (dist > MAX_DIST)
            break;
    }
    
    return float(i);
}

// Function 624
vec3 rayMarch(ray marcher){
    float epsilon = 0.001;
    float t = 0.;
    for (float i = 0.; i < maxIterations; i++){
        vec3 point = marcher.position + marcher.direction * t;
        float d = distFunc(point);
        if (d < epsilon){
            // UV wrapping so we can texture the sphere
            float u = (.5 + atan(point.z, point.x)
               / (2. * 3.14159)) / sphere1Rad / .5;
   	 		float v = (.5 - asin(point.y) / 3.14159) / sphere1Rad / .5;
      
            // Calc phong illumination
            vec3 normal = getNormal(point, sphere1Pos, sphere1Rad);
            vec3 lightDir = normalize(lightPos - point);
            vec3 viewDir = normalize(eye - point);
            vec3 reflection = reflect(lightDir, normal);
            // Add the ambient component
            float Ip = test.amb;
            // Add the diffuse component
            Ip += test.diff * dot(lightDir, normal);
            // Add the specular component
            Ip += pow(test.spec * dot(reflection, viewDir), test.shiny);
            return Ip * texture(iChannel1, vec2(u + iTime * 0.1, v)).xyz;
        }
        t+=d;
    }
    return vec3(0.);
}

// Function 625
RMInfo Raymarch(vec3 from, vec3 to)
{
    float t = 0.;
    int objId = BACKGROUND_ID;
    vec3 pos;
    vec3 normal;
    float dist;
    
  	for (int i = 0; i < NUM_STEPS; ++i)
    {
    	pos = from + to * t;
        dist = Ring(pos);

        if (dist > MAX_DIST || abs(dist) < MIN_DIST)
            break;

        t += dist * 0.43;
        objId = RING_ID;
  	}
    
    if (t < MAX_DIST)
    {
        normal = SurfaceNormal(pos);
    }
    else
    {
        objId = BACKGROUND_ID;
    }

    return RMInfo(pos, normal, objId);
}

// Function 626
float gainStep(float start, float end, float x, float e) {
    return gain(hardstep(start, end, x), e);
}

// Function 627
float march(vec3 s, vec3 d)
{
    float dist = 1.0;	// distance
    for(int i = 0; i < MARCHLIMIT; i++)
    {
        dist += map(s + d*dist)*MARCHSTEPFACTOR;
    }
    
	return min(dist, MAXDIST);
}

// Function 628
vec3 ray_march(vec3 p, vec3 r)
{
    float td = 0.;
    int i;
    for(i = 0; i < MAX_MARCHES; i++)
    {
    	float de = map(p);
        if(de < rayfov*td || td > MAX_DIST)
        {
            break;
        }
        p += de*r;
        td += de;
    }
    vec3 col = vec3(0);
    if(td > MAX_DIST || i > MAX_MARCHES)
    {
        col = sky_color(r);
    }
    else
    {
        vec4 norm = calcNormal(p,MIN_DIST);
        //sampling color at the closest point
        col = sdcolor(p - norm.w*norm.xyz);
        p += r*(norm.w-rayfov*td);
        #ifdef SHADOWS
      		float shad = shadow_march(vec4(p+norm.xyz*0.05,0.), vec4(light,0.), 400.,0.03);
        #else
        	float shad = 1.;
        #endif
       
        col = lighting(vec4(col,0.), vec2(0.2,0.3), vec4(p, 0.), vec4(r, td), norm, shad);
    }
   
    return col;

}

// Function 629
vec3 TexStep6( vec2 vTexCoord, float fRandom, float fHRandom )
{
	vec3 col = mix( vec3(87.0, 67.0, 51.0), vec3(119.0, 95.0, 75.0), fRandom) / 255.0;

	col *= Indent( vTexCoord, vec2(-1.0, 3.0), vec2(32.0, 1.0), 1.3, 0.7);
	col *= Indent( vTexCoord, vec2(-1.0, 8.0), vec2(32.0, 0.0), 1.3, 0.9);

	float fStreak = clamp((vTexCoord.y / 16.0) * 1.5 - fHRandom, 0.0, 1.0);

	col *= fStreak * 0.3 + 0.7;
	
	return col;
}

// Function 630
float hardstep(float a, float b, float t) {
    float s = 1. / (b - a);
    return clamp((t - a) * s, 0., 1.);
}

// Function 631
void MarchLight(float sscoc,inout Ray r, float startTime, float f//relkativistic raymarcher
){float acc = 0.
 ;float eps=getEpsToFps(sscoc)
 ;float lip=getReLipschitzToFps(sscoc)
 ;vec3 origin = r.b
 ;for (r.iter=0;r.iter<maxStepRayMarching;r.iter++
 ){r.time = startTime
  ;SetTime(r.time)
  ;r.dist=map(r.b,-1)
  ;acc+=r.dist*lip
  ;r.b+=r.dir*(r.dist)
  ;if(abs(r.dist)<eps||acc>f)break;}}

// Function 632
float linearstep(float edge0, float edge1, float x) {
	float t = (x - edge0)/(edge1 - edge0);
	return clamp(t, 0.0, 1.0);
}

// Function 633
float rayMarch(vec3 ro,vec3 rd, bool setMat){
    float tot = 0.;
    float dst = 0.;
    for(int i = 0; i < MAX_DST; i++){
        vec3 p = ro + rd*tot;
        dst = map(p,setMat);
        tot+=dst;
        if(dst<EPSI||tot>float(MAX_DST)){
        	tot = float(i)/float(MAX_DST);
            break;
        }
     }
    if(dst>EPSI && setMat){//sky
    	mat = 1;
    }
	return tot;
}

// Function 634
float overshootstep3( float x, float df0, float a, vec3 args )
{
	float y0 = df0 / a; // calculate y0 such that the derivative at x=0 becomes df0
	float y = x > 0.0 ? overshoot( x, args, df0 ) : 1.0 - exp( -df0 * x );
	return ( y + y0 ) / ( 1.0 + y0 ); // the step goes from y0 to 1, normalize so it is 0 to 1
}

// Function 635
void step2(inout lic_t s) {
    vec2 t = texture(iChannel0, s.p).xy;
    if (dot(t, s.t) < 0.0) t = -t;
    s.t = t;

    s.dw = (abs(t.x) > abs(t.y))? 
        abs((fract(s.p.x) - 0.5 - sign(t.x)) / t.x) : 
        abs((fract(s.p.y) - 0.5 - sign(t.y)) / t.y);

    s.p += t * s.dw / img_size;
    s.w += s.dw;
}

// Function 636
float linearStep(float a, float b, float x)
{
    float t = clamp((x - a) / (b - a), 0.0, 1.0);
    return t;
}

// Function 637
bool raymarch(in vec3 ro, in vec3 rd, inout float t, in float tmax, in surface paras,
           in bool useFirstOrder, in bool usePowerMethod,
           out float dist, out int steps) {      
      steps = 0;
      dist = 10000.0;
      vec3 pos;
      float lastT = 0.;  
      for (int i = 0; i < maxSteps; i++) {
        pos = ro + t * rd;
        float d = DE(pos, paras, useFirstOrder, usePowerMethod);
        // hit surface
        if (d < eps) {
          dist = 0.;
          t = lastT;
          return true;
        }
        // update step
        lastT = t;
        t += d;
        steps++;
        // ray outside rang
        if (t > tmax)
          return false;
      }
      return false; // i >= maxSteps      
}

// Function 638
float RayMarching(vec2 screen_pos, float near, float far)
{
	const int steps = 128;
	float delta = 1.0/float(steps);
	
	
	for (int i=0; i<steps; ++i)
	{
		vec3 pos = vec3(screen_pos, float(i)*delta);
		
		//ortho inv trans
		pos += vec3(0.0, 0.0, near/(far-near));//obj pos
		pos.z *= (far-near);
		
		//view inv trans
		//pass
		
		
		
		//world inv trans
		vec3 pos1 = pos - vec3(-0.5, 0, 1.0);
		float scale = 0.4;
		pos1 = Transform(pos1, vec3(iTime, iTime*0.2, iTime*0.1), 1.0/scale);
		if (HitC(pos1, 0.5, 1.0,0.9))
		//if (HitUnitCircle(pos))
		{
			return 1.0-float(i)*delta;
		}
		
		
		//world inv trans
		vec3 pos2 = pos - vec3(0.5, 0, 1.0);
		float scale2 = 0.4;
		pos2 = Transform(pos2, vec3(0.0, iTime*0.5, 0.0), 1.0/scale);
		if (HitY(pos2, 0.5, 0.2, 1.0))
		{
			return 1.0-float(i)*delta;
		}
	}
	return -1.0;
}

// Function 639
bool rayMarching(in vec3 origin, in vec3 ray, out vec3 m, out vec3 normal) {
    
    float	marchingDist = 39.5;
    float 	nbIter 		 = 0.0;

 	m = origin;   
    
    float lastestIntensity = -1000.;
    
    for(int i=0; i<200; i++) {
        
    	float fieldIntensity = getFieldIntensity(m, normal);
        
      if(lastestIntensity>fieldIntensity) {
            break;
        }
        
        if(fieldIntensity > THRESHOLD-0.01) {
            return true;
        }
        else {
            float delta = THRESHOLD - fieldIntensity;
            marchingDist += 0.01;//sqrt(delta)*0.5;
        	m = origin + ray * marchingDist;    
        }
    }
    
	return false;    
}

// Function 640
vec4 Raymarch_Sprite( vec2 fragCoord, vec4 vSpriteInfo )
{
    vec4 vResult = vec4( 0 );
    
    vec2 vSpritePos = fragCoord - vSpriteInfo.xy;
    float fSpriteX = vSpritePos.x - vSpriteInfo.z * 0.5;
    
    float fRotation = g_scene.fCameraRotation;
    
    //fRotation += iTime;
    
    vec3 vCameraPos;
    vCameraPos.x = cos(fRotation) * fSpriteX;
    vCameraPos.y = vSpritePos.y;
    vCameraPos.z = sin(fRotation) * fSpriteX;
    
    vec3 vCameraDir = vec3(-sin(fRotation), 0, cos(fRotation));
    
    vCameraPos -= vCameraDir * 200.0;

	SceneResult sceneResult = Scene_Trace( vCameraPos, vCameraDir, 1000.0 );
    
    if ( sceneResult.fDist > 400.0 )
    {
        return vResult;
    }

    vec3 vHitPos = vCameraPos + vCameraDir * sceneResult.fDist;
    
    vec3 vNormal = Scene_GetNormal( vHitPos );
    
    float fShade = max( 0.0, dot( vNormal, g_scene.vLightDir ) );
    
    float fSpecIntensity = 1.0;
    
    float fFBM1 = fbm( sceneResult.vUVW.xy * 30.0 * vec2(1.0, 0.4), 0.2 );    
    float fFBM2 = fbm( sceneResult.vUVW.xy * 30.0 * vec2(1.0, 0.4) + 5.0, 0.5 );
    vec3 vDiffuseCol = vec3(1.);
    if ( sceneResult.fObjectId == MAT_CHARACTER )
    {
        float fUniformBlend = smoothstep( 0.5, 0.6, fFBM1 );
        
        vDiffuseCol = mix( g_scene.charDef.vUniformColor0, 
                          g_scene.charDef.vUniformColor1, 
                          fUniformBlend );
        
        vDiffuseCol = mix( vDiffuseCol, g_scene.charDef.vSkinColor, step(2.2,sceneResult.vUVW.x) );

        float fBootBlend = step(sceneResult.vUVW.x, .4);
        
        fBootBlend = max( fBootBlend, step( abs(2.5 - sceneResult.vUVW.x), 0.2 ) ); // arm thing
        
        vDiffuseCol = mix( vDiffuseCol, g_scene.charDef.vBootsColor, fBootBlend );
                
        float fGoreBlend = smoothstep( 0.6, 0.7, fFBM2 );
        fGoreBlend = max( fGoreBlend, step(2.9,sceneResult.vUVW.x) ); // bloody hands
        
        vDiffuseCol = mix( vDiffuseCol, 
                          vec3(1,0,0), 
                          fGoreBlend );
        
        //vDiffuseCol = fract(sceneResult.vUVW);//g_scene.charDef.vCol;
        //vDiffuseCol = sceneResult.vUVW.xxx / 5.0;//g_scene.charDef.vCol;
    }
    else if ( sceneResult.fObjectId == MAT_SHOTGUN )
    {
        vDiffuseCol = vec3( 0.2 );
    }
    else if ( sceneResult.fObjectId == MAT_WOOD )
    {
        vDiffuseCol = vec3( 0.4, 0.2, .1 );
    }
    else if ( sceneResult.fObjectId == MAT_HEAD )
    {
        vDiffuseCol = g_scene.charDef.vSkinColor;
        float fHairBlend = step( sceneResult.vUVW.x + fFBM1 * 0.5, 0.1 );
        vDiffuseCol = mix( vDiffuseCol, g_scene.charDef.vHairColor, fHairBlend );
    }
    else if ( sceneResult.fObjectId == MAT_EYE )
    {
        vDiffuseCol = g_scene.charDef.vEyeColor;
    }
    else if ( sceneResult.fObjectId == MAT_GREY )
    {
        vDiffuseCol = vec3( 0.2 );
        fSpecIntensity = 0.1;
    }
    
    vec3 vDiffuseLight = g_scene.vAmbientLight + fShade * g_scene.vLightColor;
    vResult.rgb = vDiffuseCol * vDiffuseLight;
    
    vec3 vRefl = reflect( vec3(0, 0, 1), vNormal );
    float fDot = max(0.0, dot( vRefl, g_scene.vLightDir )) * fShade;
    float fSpec = pow( fDot, 5.0 );
    vResult.rgb += fSpec * fSpecIntensity;
    
    vResult.rgb = 1.0 - exp2( vResult.rgb * -2.0 );
    vResult.rgb = pow( vResult.rgb, vec3(1.0 / 1.5) );
    
    vResult.a = 1.0;
    
    return vResult;
}

// Function 641
vec3 MarchRay(vec3 origin,vec3 dir)
{
    bool inWarp = false;
    
    float dist = 0.0;
    
    //Distance to the "warp zone".
    float warpDist = MarchWarp(origin,dir);
    
    for(int i = 0;i < MAX_STEPS;i++)
    {
        float sceneDist = Scene(origin + dir * dist);
        
        //Reset the march distance, set the ray origin to the surface of the "warp zone", scale the map and ray origin.
        #ifndef DISABLE_WARP
        if(warpDist < dist && !inWarp)
    	{
            scale.x = 4.0;
            
            dist = 0.0;
            origin = origin + dir * warpDist;
            origin /= scale;
            
            inWarp = true;
    	}
        #endif
        
        dist += sceneDist * STEP_MULT;
        
        if(abs(sceneDist) < MIN_DIST || sceneDist > MAX_DIST)
        {
            if(sceneDist < 0.0)
            {
                dist += MIN_DIST;
            }
            
            break;
        }
    }
    
    return origin + dir * dist;
}

// Function 642
vec4 march(vec3 p,vec3 r)
{
    vec4 m = vec4(p+r,1);
    for(int i = 0;i<99;i++)
    {
        float s = dist(m.xyz);
        m += vec4(r,1)*s;
        
        if (s<EPS || m.w>MAX) return m;
    }
    return m;
}

// Function 643
vec4 ray_march( in vec3 ro, in vec3 rd, int maxstep ) {
    float t = 0.0001;
    vec3 m = vec3(0.);
    for( int i=0; i<maxstep; i++ ) {
        vec4 d = map(ro + rd * t);
        m = d.yzw;
        if(d.x<.001*t||t>MAX_DIST) break;
        t += d.x*.5;
    }
    return vec4(t,m);
}

// Function 644
RayHit march(Ray ray)
{
	float dist = rayAABB(ray, worldbox, 0.0, 1e20);
	//if(dist == 0.0)
	//	return RayHit(0.0, vec3(0), vec3(0));
	vec3 start = (ray.o) + ray.d * (floor(dist));
	vec3 pos = raycast(start, ray.d);
	return RayHit(dist = length(pos), normalize(pos), vec3(1.0,1.0,1.0));
}

// Function 645
float march(in vec3 ro, in vec3 rd)
{
	float precis = 0.001;
    float h=precis*2.0;
    float d = 0.;
    for( int i=0; i<ITR; i++ )
    {
        if( abs(h)<precis || d>FAR ) break;
        d += h;
	    float res = map(ro+rd*d);
        h = res;
    }   
	return d;
}

// Function 646
void RaymarchScene( vec3 vRayOrigin, vec3 vRayDir, out Intersection intersection )
{
    float stepScale = 1.0;
#ifdef ENABLE_CONE_STEPPING
    vec2 vRayProfile = vec2( sqrt(dot(vRayDir.xz, vRayDir.xz) ), vRayDir.y );
    vec2 vGradVec = normalize( vec2( 1.0, 2.0 ) ); // represents the biggest gradient in our heightfield
    vec2 vGradPerp = vec2( vGradVec.y, -vGradVec.x );

    float fRdotG = dot( vRayProfile, vGradPerp );
    float fOdotG = dot( vec2(0.0, 1.0), vGradPerp );

    stepScale = -fOdotG / fRdotG;

    if ( stepScale < 0.0 )
    {
        intersection.m_objId = OBJ_ID_SKY;
        intersection.m_dist = k_fFarClip;
        return;
    }
#endif
    
    intersection.m_dist = 0.01;
    intersection.m_objId = OBJ_ID_SKY;
    
    float fSceneDist = 0.0;
    
    float oldT = 0.01;
    for( int iter = 0; iter < k_raymarchSteps; iter++ )
    {
        vec3 vPos = vRayOrigin + vRayDir * intersection.m_dist;
      
        // into sky - early out
        if ( vRayDir.y > 0.0 )
        {
            if( vPos.y > 1.0 )
            {
                intersection.m_objId = OBJ_ID_SKY;
                intersection.m_dist = k_fFarClip;
                break;
            }
        }

      
        fSceneDist = GetSceneDistance( vPos );

        oldT = intersection.m_dist;
        intersection.m_dist += fSceneDist * stepScale;
                
        intersection.m_objId = OBJ_ID_GROUND;
        if ( fSceneDist <= 0.01 )
        {
            break;
        }

        if ( intersection.m_dist > k_fFarClip )
        {
            intersection.m_objId = OBJ_ID_SKY;
            intersection.m_dist = k_fFarClip;
            break;
        }        

        
    }    
    
    intersection.m_pos = vRayOrigin + vRayDir * intersection.m_dist;
}

// Function 647
vec2 raymarch(vec3 ro, vec3 rd, float tmin, float tmax)
{
    vec2 nearest = vec2(tmin, TYPE_DEFAULT);
    for(int i=0;i<RAYMARCH_ITERATIONS;i++)
    {
        vec3 p = ro+rd*nearest.x;
        vec2 res = map(p);
        if(res.x<tmin || nearest.x>tmax)
            break;
        nearest.x += 0.5*res.x*res.x; // res square solves a lot of temporal depth noise
        nearest.y = res.y;
    }
    return nearest;
}

// Function 648
bool march(out int steps, out vec3 point, out float smallest_dist, in vec3 r0, in vec3 rd) {
    float total_dist = 0.0;
    float cur_dist;
    smallest_dist = 10000000.0;
    
    point = r0;
    for (steps=0; steps<100; steps++) {
        cur_dist = dist(point);
        if(cur_dist < smallest_dist) {
            smallest_dist = cur_dist;
        }
        if(cur_dist < threshold) {
            return true;
        }
        total_dist += cur_dist;
    	point = r0 + rd * total_dist;
    }
    return false;
}

// Function 649
vec3 march(vec3 pos, vec3 dir) {
    float t = time();
    float d = 0.;
    float step_frac = .9;
    int max_marches = MARCHES;
    for(int i = 0; i < MARCHES; i++) {
        d = dist(pos);
        if (d < EPS) {
            return shade(pos,dir,t);
        }
        step_frac = max(0.2,min(1.,d));
    	pos.xyz += dir * d * step_frac;
    }
    return vec3(0.);
}

// Function 650
vec4 raymarch(in vec3 ro, in vec3 rd)
{
    vec4 acc = vec4(0.);
    float t = 0.0;
    for (int i = 0; i < 32 && acc.a < 0.95; ++i)
    {
        vec3 pos = ro + t * rd;
        float d = map(pos);
        float a = clamp(d * -30., 0.0, 0.2);
        float s = map(pos + 0.3 * sundir);
        float diff = clamp((s - d) * 0.4, 0.0, 1.0);
        vec3 brdf = vec3(0.65,0.68,0.7)* 0.2 + 3.*vec3(0.7, 0.5, 0.3)*diff;
        acc.w += (1. - acc.w) * a;
        acc.xyz += a * brdf;
        t += max(d * 0.5, 0.02);
    }
    
    acc.xyz /= (0.001 + acc.w);
    return acc;
}

// Function 651
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

// Function 652
vec2 march(Ray ray) 
{
    const int steps = 50;
    const float prec = 0.001;
    vec2 res = vec2(0.);
    
    for (int i = 0; i < steps; i++) 
    {        
        vec2 s = map(ray.ro + ray.rd * res.x);
        
        if (res.x > MAXDIST || s.x < prec) 
        {
        	break;    
        }
        
        res.x += s.x;
        res.y = s.y;
        
    }
   
    return res;
}

// Function 653
vec3 raymarch(in vec3 ro, in vec3 rd, in vec2 seed)
{
    float t = 0.;
    for (int i=0; i<INIT_ITERS; i++)
    {
    	float dist = map(ro + t*rd);
        t += dist;
    }
    float transmission = 1.0;
    float threshold = noise(seed)*0.7 + 0.2;
    for (int i=0; i<ITERS; i++)
    {
        float st = STEP * (1.+(noise(seed)-0.5)*0.8);
    	transmission *= (1.0 - density(ro + t*rd));
        if (transmission < threshold)
        {
            break;
        }
        seed = noise2D(seed);
        t += st;
    }
    
    //secondary ray
    ro = ro + t*rd;
    vec3 reflectance = vec3(step(ro.y, 0.), step(ro.x, 0.), step(-ro.y, 0.))*smoothstep(length(ro),5., 6.);
    t = 0.;
    vec3 light_dir = normalize(vec3(noise2D(seed), noise(seed)) - 0.5);
    float intensity = pow(light_dir.y + 1., 2.);
    for (int i=0; i<ITERS; i++)
    {
        t += STEP;
    	transmission *= (1.0 - density(ro + t*light_dir));
    }
    
    vec3 n = normal(ro);
    vec3 h = normalize(light_dir - rd);
    float spec = pow(max(0., dot(h, n)), 10.);
    float diffuse = max(0., dot(n, light_dir));
    return reflectance*(spec + diffuse)*intensity;
}

// Function 654
float march(in vec3 rayOrigin, in vec3 rayDirection) {
    float dist = 0.1;
    for (int i = 0; i < MAX_ITER; i++) {
        float t = map(rayOrigin + rayDirection * dist);
        dist += t;
        if (abs(t) < PRECISION || dist > MAX_DISTANCE) {
            break;
        }
    }
	return dist;
}

// Function 655
vec2 raymarch(in vec3 origin, in vec3 ray)
{
    float t = 0.0; // t is the clipping plane where anything below this is removed.
    float id = -1.0; // For the Background
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec2 dist = scene(origin + ray * t);
        if (dist.x < EPSILON)
        {
            break; // We are inside the surface.
        }
        // Move along the ray in constant steps
        t += dist.x;
        // Since each element has an ID, we want that there too!
        id = dist.y;

        if (t >= MAX_DIST)
        {
            id = BACKGROUND;
            return vec2(MAX_DIST, id); //We are too far away!
        }
    }
    return vec2(t, id);
}

// Function 656
float parabolstep( float a, float b, float x ) { float t = clamp( ( x - a ) / ( b - a ), 0., 1. ) - .5; return .5 - 2. * ( abs( t ) * t - t ); }

// Function 657
float march(inout vec3 p, vec3 dir)
{
	vec2 r = distID(p+dir*EPSILON);
	for(int i = 0; i < V_STEPS; i++)
	{
		if(r.s < EPSILON)
			return r.t;
		p += dir*r.s;
        r = distID(p);
	}
	return r.t;
}

// Function 658
vec2 raymarch(inout vec3 pos, inout vec3 dir)
{
    vec2 result;	//the scene is contained in this variable
    
    for(float i = 0.0; i < EVALCNT; i++)
    {
        result = scene(pos);
        pos += dir * result.x;
        if(result.x < AEPSILON)
        {
            return result;
        }
    }
 	return vec2(99999.0, 99999.0);   //here should be some sky color computations, but now, I cannot do such
    						 //thing with my n00b ass...nonetheless, cornflower blue rulz
}

// Function 659
vec3 RayMarchScene(vec3 rayOrigin, inout rayIntersect rIntersec)
{
    float t = NEARCLIP;
    vec2 res = vec2(NEARCLIP, -1.0);
    
	for(int i=0; i < SCENE_SAMPLE; ++i)
	{
        vec3 pos = rayOrigin + t * rIntersec.rd;
        res = SceneDistance(pos);
        if(res.x < (EPSILON * t) || t > FARCLIP)
            break;
        t += res.x * 0.5;
	}
    
    vec3 pos = rayOrigin + t * rIntersec.rd;
    rIntersec.mPos = pos;
    rIntersec.dist = t;
	
    material mat;
    mat.albedo = vec3(244.0, 164.0, 96.0)/255.0;
    mat.reflectivity = 0.8;
    rIntersec.mat = mat;
    
    #if DEBUG_PASS == 0
        if (t > FARCLIP)
        {
            rIntersec.dist = FARCLIP;
            return GetSkybox(rayOrigin, rIntersec);
        }
        else
        {
            #ifndef DEBUG_NO_FOG
                float sundot = clamp(dot(rIntersec.rd, sunDirection), 0.0, 1.0);
            	vec3 sky = CheapSkyBox(rayOrigin, rIntersec);
            	ColorScene(rIntersec);
            
            	float fogFactor = EaseOutSine(rIntersec.dist / FARCLIP);
            	return mix(Shading(rayOrigin, rIntersec), sky, fogFactor);
            #else
            	ColorScene(rIntersec);
            	return Shading(rayOrigin, rIntersec);
            #endif
        }
    
    #elif DEBUG_PASS == 1
    	if(t < FARCLIP)
            ColorScene(rIntersec);
    	return rIntersec.nor;
    
    #elif DEBUG_PASS == 2
        if (t > FARCLIP)
            rIntersec.dist = FARCLIP;
        return vec3(rIntersec.dist) / FARCLIP;
    
    #elif DEBUG_PASS == 3
    	return rIntersec.mPos;
    
    #elif DEBUG_PASS == 4
    	if(t < FARCLIP)
            ColorScene(rIntersec);
        vec3 sunLightPos = normalize(sunDirection);
        float NdotL = clamp(dot(rIntersec.nor, sunLightPos), 0.0, 1.0);
        return vec3(NdotL);
    
    #elif DEBUG_PASS == 5
    	if(t < FARCLIP)
            ColorScene(rIntersec);
        float NdotV = clamp(dot(rIntersec.nor, -rIntersec.rd), 0.0, 1.0);
        return vec3(NdotV);
    
    #elif DEBUG_PASS == 6
    	if(t < FARCLIP)
            ColorScene(rIntersec);
        vec3 sunLightPos = normalize(sunDirection);
        vec3 HalfAngleV = normalize(-rIntersec.rd + sunLightPos);
        float NdotH = clamp(dot(rIntersec.nor, HalfAngleV), 0.0, 1.0);
        return vec3(NdotH);
    
    #elif DEBUG_PASS == 7
    	if(t < FARCLIP)
            ColorScene(rIntersec);
        vec3 sunLightPos = normalize(sunDirection);
        vec3 HalfAngleV = normalize(-rIntersec.rd + sunLightPos);
        float VdotH = clamp(dot(-rIntersec.rd, HalfAngleV), 0.0, 1.0);
        return vec3(VdotH);
    
    #elif DEBUG_PASS == 8
    	if (t > FARCLIP)
        {
            rIntersec.dist = FARCLIP;
            return GetSkybox(rayOrigin, rIntersec);
        }
        else
	    	return CheapSkyBox(rayOrigin, rIntersec);
    
    #elif DEBUG_PASS == 9
    	if(t < FARCLIP)
            ColorScene(rIntersec);
    
    	vec3 amb = (abs(sun_Color.w - 1.0) * 0.03 + AMBIENT_POW) * AMBIENT_COLOR;
    	return amb;

    #elif DEBUG_PASS == 10
   		float shadow = SoftShadow(rIntersec.mPos + sunDirection);
    	return vec3(shadow);
    
    #endif
}

// Function 660
float march(vec3 q, vec3 r) {
  float t = 0.1;
  int numsteps = 200;
  float precis = 1e-3;
  for (int i = 0; i < numsteps; i++) {
    //assert(i < 20);
    vec3 p = q+t*r;
    float d = eval(p);
    //assert(d >= 0.0);
    if (abs(d) < precis) return t;
    d = min(0.5,d);
    t += d;
    if (t < 0.0 || t > maxdist) break;
  }
  return -1.0;
}

// Function 661
float RAYMARCH_isosurface( vec3 o, vec3 d, float isoSurfaceValue)
{
    //Learned from Inigo Quilez DF ray marching :
    //http://www.iquilezles.org/www/articles/raymarchingdf/raymarchingdf.htm
    //Original articles (interesting read) :
    //Sphere Tracing: A Geometric Method for the Antialiased Ray Tracing of Implicit Surfaces (1989)
    //http://mathinfo.univ-reims.fr/IMG/pdf/hart94sphere.pdf
    //John C. Hart Sphere Tracing: A Geometric Method for the Antialiased Ray Tracing of Implicit Surfaces (1994)
    //http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.48.3825 p. 5.75-5.85
    
    const float tolerance = 0.0001;
    float t = 0.0;
    float dist = MAX_DIST;
    #if OVERSTEP_COMPENSATION
    for( int i=0; i<30; i++ )
    {
        dist = DF_composition( o+d*t ).d;
        dist -= isoSurfaceValue;
        
        if( abs(dist)<tolerance*100.0 ) break;
        t += dist;
    }
    
    t -= Z_REPEAT_DIST/2.0;
    
    for( int i=0; i<30; i++ )
    {
        dist = DF_composition( o+d*t ).d;
        dist -= isoSurfaceValue;
        
        if( abs(dist)<tolerance ) break;
        
        t += min(dist,Z_REPEAT_DIST/5.0);
    }
    #else
    for( int i=0; i<70; i++ )
    {
        dist = DF_composition( o+d*t ).d;
        dist -= isoSurfaceValue;
        
        if( abs(dist)<tolerance ) break;
        t += dist;
    }
    #endif
    
    return t;
}

// Function 662
raysample RayMarch(vec3 rO, vec3 rD, int bounces, vec3 col) 
{
    //	Set Distance to Origin
    raysample dO = raysample(0., rO, 0, col);
    
    for (int i=0; i<MAX_STEPS; i++) 
    {
        //	Step Forward
        vec3 p = rO+dO.dist*rD;
        raysample dS = GetDist(p);
        dO.dist += dS.dist;
        dO.point = p;
        dO.material = dS.material;
        
        //	On surface? Can we stop?
        if (dS.dist<SURF_DIST) {
            //	Surface has been hit.
            //	Is it a mirror? Can we bounce?
            if (dO.material == 0 && bounces > 0) {
                //	Reflect ray
                rO = rO+dO.dist*rD;
                rD = reflect(rD, GetNormal(rO));
                rO = rO-dO.dist*rD;
                //	Nudge ray away from surface
                dO.dist += 2.0*SURF_DIST;
                //	Pick up some color
                dO.color *= M[dO.material].specular + M[dO.material].color * M[dO.material].diffuse;
                bounces--;
            } else {
                break;
            }
        }
        //	Tired of marching? Can we stop?
        if (dO.dist>MAX_DIST) break;
    }
    
    return dO;
}

// Function 663
RayHit raymarching(inout RayHit ray)
{
    RayHit nohit;
    ray.uv = ray.dir;
    
    //ray
    for (int i = 0; i < 400; i++)
    {        
      	Hit hit = mapDistance(ray.pos);
        ray.hit = hit;
        ray.distanceTotal	+= hit.distanceRadius;           // total
        ray.pos 			+= hit.distanceRadius * ray.dir; // walk point to direction
        
        if (hit.distanceRadius < .01) break;						// so close... performance otimization...
        if (hit.distanceRadius > 500.) { ray = nohit; break; }	// so far..... performance otimization...
 
    }
    if (ray.pos.x != 0.){
        ray.clo = close3d(ray);
        ray.nor = normal3d(ray);
        ray.refl = reflect(ray.dir, ray.nor );
    }
    
    return ray;
}

// Function 664
vec3 raymarchFast( vec3 ro, vec3 rd, const vec2 nf, const float eps ) {
    glowAcc = vec2(999.);
    vec3 p = ro + rd * nf.x;
    float l = 0.;
    for(int i=0; i<64; i++) {
		float d = world(p);
        l += d;
        p += rd * d*1.2;
        
        if(d < eps || l > nf.y)
            break;
    }
    
    return p;
}

// Function 665
vec2 march_ray(vec3 ray_origin, vec3 ray_direction, float tmax) {
	float t = 0.0;//current depth
    float m = 0.0;
    for( int i=0; i<256; i++ )
    {
        vec3 pos = ray_origin + t*ray_direction;
        //get dist to nearest surface
        vec2 h = map_the_world(pos);
        m = h.y;
        //if we hit something break
        if( h.x<0.0001 || t>tmax ) return vec2(t,m);
        //step forward
        t += h.x;
    }
    return vec2(t,m);
}

// Function 666
float sstep(float x) {
    return ( x<0. ? 0. : 1.0 );
}

// Function 667
float march(vec3 ro, vec3 rd){
    float t = 0.001;//EPSILON;
    float step = 0.0;

    float omega = 1.3;//muista testata eri arvoilla! [1,2]
    float prev_radius = 0.0;

    float candidate_t = t;
    float candidate_error = 1000.0;
    float sg = sgn(sdf(ro));

    vec3 p = vec3(0.0);

	for(int i = 0; i < STEPS; ++i){
		p = rd*t+ro;
		float sg_radius = sg*sdf(p);
		float radius = abs(sg_radius);
		step = sg_radius;
		bool fail = omega > 1. && (radius+prev_radius) < step;
		if(fail){
			step -= omega * step;
			omega = 1.;
		}
		else{
			step = sg_radius*omega;
		}
		prev_radius = radius;
		float error = radius/t;

		if(!fail && error < candidate_error){
			candidate_t = t;
			candidate_error = error;
		}

		if(!fail && error < PIXELR || t > FAR){
			break;
		}
		t += step;
	}
    //discontinuity reduction
    float er = candidate_error;
    for(int j = 0; j < 6; ++j){
        float radius = abs(sg*sdf(p));
        p += rd*(radius-er);
        t = length(p-ro);
        er = radius/t;

        if(er < candidate_error){
            candidate_t = t;
            candidate_error = er;
        }
    }
	if(t <= FAR || candidate_error <= PIXELR){
		t = candidate_t;
	}
	return t;
}

// Function 668
float rayMarch(vec3 ro, vec3 rd) {
    // Distance to the object
    float dO = 0.0; 
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + (dO * rd);
        float dS = getDistance(p);
        dO += dS;
        if (dS <= SURFACE_DIST || dS > MAX_DISTANCE) break;
    }

    return dO;
}

// Function 669
void setStep(float num) {
    if(num < stepping) {
    	stepping = num;
    }
    if(stepping < 0.1) {
    	stepping = 0.1;
    }
}

// Function 670
float march(in vec3 ro, in vec3 rd)
{
	float precis = 0.002;
    float h=precis*2.0;
    float d = 0.;
    for( int i=0; i<ITR; i++ )
    {
        if( abs(h)<precis || d>FAR ) break;
        d += h;
	    float res = map(ro+rd*d);
        h = res;
    }
	return d;
}

// Function 671
float steps(in float x, in float k) {
	float fr = fract(x);
	return floor(x)+(fr<k?0.:(fr-k)/(1.-k));
}

// Function 672
vec2 relaxedMarch(vec3 ro, vec3 rd) {

    const float PR = EPS;

    float om = 1.3,
          t = EPS,
          ce = FAR,
          ct = EPS,
          pr = 0.0,
          sl = 0.0,
          fs = 1.0, //map(ro).x < 0.0 ? -1.0 : 1.0,
          id = 0.0;

    for (int i = ZERO; i < 120; ++i) {
        vec2 si = map(ro + rd * t);
        float sr = fs * si.x;
        float r = abs(sr);
        bool fail = om > 1.0 && (r + pr) < sl;
        if (fail) {
            sl -= om * sl;
            om = 1.0;
        } else {
            sl = sr * om;
        }
        pr = r;
        float err = r / t;
        if (!fail && err < ce) {
            ct = t;
            ce = err;
        }
        if (!fail && err < PR || t > FAR) {
            id = si.y;
            break;
        }
        t += sl;
    }

    if (t>FAR || ce>PR) ct=-1.;

    return vec2(ct,id);
}

// Function 673
float sstep2(float x)
{
	x = clamp(x, 0.0, 1.0); // optional
	float ix = (1.0 - x);
    x = x * x;
    ix = ix * ix;
    return x / (x + ix);
}

// Function 674
void MarchLight(float sscoc,inout Ray r, float startTime, float maxDist//relkativistic raymarcher
){float totalDist = 0.0
 ;vec3 origin = r.b
 ;for (r.iter=0;r.iter<maxStepRayMarching;r.iter++
 ){r.time = startTime
  ;SetTime(r.time)
  ;r.dist = map(r.b,-1)
  ;totalDist += r.dist*getReLipschitzToFps(sscoc)
  ;r.b+=r.dir*(r.dist)
  ;if(abs(r.dist)<getEpsToFps(sscoc)||totalDist>maxDist)break;}}

// Function 675
float raymarch(vec3 ro,vec3 rd)
{
	float t = 0.;
    vec3 p ;
    for(float i = 0.;i < MAXSTEP;i++)
    {
    	p = ro+rd*t;
        float d = map(p);
        t += d*.1;
        if(abs(d) < EPS || t > FAR) break;
        count += 1./(1.+d*d);
    }
    return t;
}

// Function 676
vec2 RayMarch(vec3 r_origin, vec3 r_direction)
{
    float d_origin = 0.;
    float min_dist = 1000000.0;
    
    for(int i=0; i<MAX_STEPS; i++) 
    {
        vec3 p = r_origin + d_origin*r_direction;
        float d_surf = GetDist(p);
        min_dist = min(d_surf, min_dist);
        d_origin += d_surf;
        if(d_surf<MIN_DIST || d_origin>MAX_DIST) break;
    }
    
    return vec2(d_origin,min_dist);
}

// Function 677
vec2	march(vec3 pos, vec3 dir)
{
    vec2	dist = vec2(.0);
    vec3	p = vec3(0.0);
    vec2	s = vec2(0.0);
	first = 0.;
    dynamiceps = E*1.;
    for (int i = 1; i < I_MAX; ++i)
    {
    	p = pos + dir * dist.y;
        dist.x = real_scene(p);
        dist.y += dist.x;
        dynamiceps = 1./1e4+( +.001+(dynamiceps)/max(1.,min(dist.y, dist.x)) );
        evaluateLight(p)*.051125;
        if (log(dist.y*dist.y/dist.x/1e5)>0. || dist.x <= dynamiceps || dist.y > FAR)
        {
            if (mind == minf && first <= PORTAL_DEPTH)
            {
                first++;
                float	e = .001;
				vec3 eps = vec3(e,0.0,0.0);
                //pos.z -=26.;
            	vec3	norm = normalize(vec3(
                real_scene((pos+eps.xyy) + (dir) * dist.y) - real_scene((pos-eps.xyy) + (dir) * dist.y ),
                real_scene((pos+eps.yxy) + (dir) * dist.y) - real_scene((pos-eps.yxy) + (dir) * dist.y ),
                real_scene((pos+eps.yyx) + (dir) * dist.y) - real_scene((pos-eps.yyx) + (dir) * dist.y ) ));
                //pos.z += 26.;
                dir = refract( (dir), norm, .05);
                normalize(dir);
                dist.y=0.;
                rotate(dir.yx, 1.57+iTime);
                continue;
			}
           break;
        }
        s.x++;
    }
    s.x = s.x + first;
    s.y = dist.y;
    return (s);
}

// Function 678
void TestBezierMarch(in vec3 rayPos, inout SRayHitInfo info, in vec3 A, in vec3 B, in vec3 C, in float width, in SMaterial material)
{
    float dist = BezierDistance(rayPos, A, B, C, width);
    if (dist < info.dist)
    {
        info.rayMarchedObject = true;
        info.dist = dist;    
        info.normal = vec3(1.0f, 0.0f, 0.0f); // TODO: normal, if you ever need it!
        info.material = material;
    }    
}

// Function 679
Hit rayMarch(vec3 ro, vec3 rd) {
    Hit resultHit;
    resultHit.dist = 0.0;

    for(int i = 0; i < MAX_STEPS; ++i) {
        vec3 p = ro + resultHit.dist * rd;
        Hit hit = sdf(p);
        resultHit.dist += hit.dist;
        resultHit.color = hit.color;
        if(hit.dist < EPSILON || resultHit.dist > MAX_DIST) {
            break;
        }
    }
    return resultHit;
}

// Function 680
float march(vec3 eye, vec3 dir) {
    float depth = 0.0;
    for (int i = 0; i < marchIter; ++i) {
        float dist = scene(eye + depth * dir);
        depth += dist;
        if (dist < epsilon || depth >= marchDist)
			break;
    }
    return depth;
}

// Function 681
void Step (int mId, out vec3 rm, out vec3 vm, out vec4 qm, out vec3 wm)
{
  mat3 mRot, mRotN;
  vec3 rmN, vmN, wmN, dr, dv, rts, rtsN, rms, vms, fc, am, wam, dSp;
  float farSep, rSep, grav, fDamp, fAttr, dt;
  const vec2 e = vec2 (0.1, 0.);
  grav = 5.;
  fDamp = 0.1;
  fAttr = 0.1;
  dt = 0.01;
  rm = Loadv4 (4 * mId).xyz;
  vm = Loadv4 (4 * mId + 1).xyz;
  qm = Loadv4 (4 * mId + 2);
  wm = Loadv4 (4 * mId + 3).xyz;
  mRot = QtToRMat (qm);
  farSep = length (blkGap * (blkSph - 1.)) + 1.;
  am = vec3 (0.);
  wam = vec3 (0.);
  for (int n = 0; n < nBlock; n ++) {
    rmN = Loadv4 (4 * n).xyz;
    if (n != mId && length (rm - rmN) < farSep) {
      vmN = Loadv4 (4 * n + 1).xyz;
      mRotN = QtToRMat (Loadv4 (4 * n + 2));
      wmN = Loadv4 (4 * n + 3).xyz;
      for (int j = 0; j < nSiteBk; j ++) {
        rts = mRot * RSite (j);
        rms = rm + rts;
        vms = vm + cross (wm, rts);
        dv = vms - vmN;
        fc = vec3 (0.);
        for (int jN = 0; jN < nSiteBk; jN ++) {
          rtsN = mRotN * RSite (jN);
          dr = rms - (rmN + rtsN);
          rSep = length (dr);
          if (rSep < 1.) fc += FcFun (dr, rSep, dv - cross (wmN, rtsN));
        }
        am += fc;
        wam += cross (rts, fc);
      }
    }
  }
  for (int j = 0; j < nSiteBk; j ++) {
    rts = mRot * RSite (j);
    dr = rm + rts;
    dr.xz = -0.55 * GrndNf (dr).xz;
    dr.y += 0.55 - GrndHt (rm.xz - dr.xz);
    rSep = length (dr);
    if (rSep < 1.) {
      fc = FcFun (dr, rSep, vm + cross (wm, rts));
      am += fc;
      wam += cross (rts, fc);
    }
  }
  am -= fDamp * vm;
  am.y -= grav;
  am += fAttr * (rLead - rm);
  dSp = blkGap * blkSph;
  wam = mRot * (wam * mRot / (0.25 * (vec3 (dot (dSp, dSp)) - dSp * dSp) + 1.));
  vm += dt * am;
  rm += dt * vm;
  wm += dt * wam;
  qm = normalize (QtMul (RMatToQt (LpStepMat (0.5 * dt * wm)), qm));
}

// Function 682
vec3 raymarch_o349467(vec2 uv) {
    uv-=0.5;
	vec3 cam=vec3((sin(sin(iTime*0.2)*0.5+0.5)*4.0),p_o349467_CamY,(sin(sin(iTime*0.3)*0.5+0.5)*4.0))*p_o349467_CamZoom;
	vec3 lookat=vec3(p_o349467_LookAtX,p_o349467_LookAtY,p_o349467_LookAtZ);
	vec3 ray=normalize(lookat-cam);
	vec3 cX=normalize(cross(vec3(0.0,1.0,0.0),ray));
	vec3 cY=normalize(cross(cX,ray));
	vec3 rd = normalize(ray*p_o349467_CamD+cX*uv.x+cY*uv.y);
	vec3 ro = cam;
	
	float d=0.;
	vec3 p=vec3(0);
	vec2 dS=vec2(0);
	march_o349467(d,p,dS,ro,rd);
	
    vec3 color=vec3(0.0);
	vec3 objColor=(dS.y<0.5)?o349467_input_tex3d_a(vec4(p,1.0)):o349467_input_tex3d_b(vec4(p,1.0));
	vec3 light=normalize(vec3(p_o349467_SunX,p_o349467_SunY,p_o349467_SunZ));
	if (d<50.0) {
	    vec3 n=normal_o349467(p);
		float l=clamp(dot(-light,-n),0.0,1.0);
		vec3 ref=normalize(reflect(rd,-n));
		float r=clamp(dot(ref,light),0.0,1.0);
		float cAO=mix(1.0,calcAO_o349467(p,n),p_o349467_AmbOcclusion);
		float shadow=mix(1.0,calcSoftshadow_o349467(p,light,0.05,5.0),p_o349467_Shadow);
		color=min(vec3(max(shadow,p_o349467_AmbLight)),max(l,p_o349467_AmbLight))*max(cAO,p_o349467_AmbLight)*objColor+pow(r,p_o349467_Pow)*p_o349467_Specular;
		//reflection
		d=0.01;
		march_o349467(d,p,dS,p,ref);
		vec3 objColorRef=vec3(0);
		if (d<50.0) {
			objColorRef=(dS.y<0.5)?o349467_input_tex3d_a(vec4(p,1.0)):o349467_input_tex3d_b(vec4(p,1.0));
			n=normal_o349467(p);
			l=clamp(dot(-light,-n),0.0,1.0);
			objColorRef=max(l,p_o349467_AmbLight)*objColorRef;
		} else {
			objColorRef=o349467_input_hdri(equirectangularMap(ref.xzy)).xyz;
		}
		color=mix(color,objColorRef,p_o349467_Reflection);
	} else {
		color=o349467_input_hdri(equirectangularMap(rd.xzy)).xyz;
	}
	return color;
}

// Function 683
vec3 raymarching(vec3 ro, vec3 rd, float t, vec3 backCol)
{   
    vec4 sum = vec4(0.0);
    vec3 pos = ro + rd * t;
    for (int i = 0; i < 40; i++) {
        if (sum.a > 0.99 || 
            pos.y < (MIN_HEIGHT-1.0) || 
            pos.y > (MAX_HEIGHT+1.0)) break;
        
        float den = density(pos);
        
        if (den > 0.01) {
            float dif = clamp((den - density(pos+0.3*sundir))/0.6, 0.0, 1.0);

            vec3 lin = vec3(0.65,0.7,0.75)*1.5 + vec3(1.0, 0.6, 0.3)*dif;        
            vec4 col = vec4( mix( vec3(1.0,0.95,0.8)*1.1, vec3(0.35,0.4,0.45), den), den);
            col.rgb *= lin;

            // front to back blending    
            col.a *= 0.5;
            col.rgb *= col.a;

            sum = sum + col*(1.0 - sum.a); 
        }
        
        t += max(0.05, 0.02 * t);
        pos = ro + rd * t;
    }
    
    sum = clamp(sum, 0.0, 1.0);
    
    float h = rd.y;
    sum.rgb = mix(sum.rgb, backCol, exp(-20.*h*h) );
    
    return mix(backCol, sum.xyz, sum.a);
}

// Function 684
float raymarch(vec3 rayOrigin, vec3 rayDirection)
{
    float t = 0.0;
    for(int i = 0; i < maxSteps; ++i)
    {
        //pick a point p from origin along rayDirection
        //at distance step t. (addition of 2 vectors)
        vec3 p = rayOrigin + rayDirection * t; 
        
        //calculate distance of p from sphere
        float d = distSphere(p, sphereRadius);
        
        
        //increment the step value by distance
        t += d;
        
        //if we are close enough to target pixel. break
        if(d < epsilon)
        {
            break;
        }
    }
    return t;
}

// Function 685
float march( vec3 a, vec3 ab ) {
    float d = .0;
    for ( int i = 0 ; i < STEPS ; i++ ) {
        vec3 b = a + d * ab;
        float n = getDistance( b );
        d += n;
        if ( abs( n ) < CLOSE || d > FAR ) break;
    }
    return d;
}

// Function 686
vec4 Raymarcher(vec3 p,vec3 dir,bool A)
{
    float tmin = 1.8;// rFloor(p,dir,0.035);
    
    if(A)
    {
   		 float h=rSphere(p,dir,pins(0)+vec3(0,-0.21,0),BOUNDS);
  		  tmin=min(tmin,h);
    }
    float tmax = MAX_DIST;
    float t = tmin;
    vec4  dist = vec4(MAX_DIST,0.,0.,0.);
    for( int i=0; i<MAX_ITERATIONS; i++ )
    {
	    dist = sdScene( p+dir*t ,A);
        if( (dist.x)<MIN_DIST || t>MAX_DIST ) break;
        t += dist.x;
    }
    
    return vec4( t, dist.yzw );

}

// Function 687
Hit raymarch(vec3 rayOrigin, vec3 rayDirection){

    float currentDist = INTERSECTION_PRECISION * 2.0;
    float rayLength = 0.;
    Model model;

    for(int i = 0; i < NUM_OF_TRACE_STEPS; i++){
        if (currentDist < INTERSECTION_PRECISION || rayLength > MAX_TRACE_DISTANCE) {
            break;
        }
        model = map(rayOrigin + rayDirection * rayLength);
        currentDist = model.dist;
        rayLength += currentDist * (1. - FUDGE_FACTOR);
    }

    bool isBackground = false;
    vec3 pos = vec3(0);
    vec3 normal = vec3(0);

    if (rayLength > MAX_TRACE_DISTANCE) {
        isBackground = true;
    } else {
        pos = rayOrigin + rayDirection * rayLength;
        normal = calcNormal(pos);
    }

    return Hit(model, pos, isBackground, normal, rayOrigin, rayDirection);
}

// Function 688
vec3 RayMarchCloud(vec3 ro,vec3 rd){
    vec3 col = vec3(0.1,0.1,0.1);  
    float sundot = clamp(dot(rd,lightDir),0.1,2.0);
    
     // sky      
    col = vec3(0.3,0.6,0.95)*2.1 - rd.y*rd.y*0.6;
    col = mix( col, 0.95*vec3(0.8,0.85,0.95), pow( 2.0-max(rd.y,0.1), 5.0 ) );
    // sun
    col += 0.35*vec3(2.0,0.8,0.5)*pow( sundot,6.0 );
    col += 0.35*vec3(2.0,0.9,0.7)*pow( sundot,74.0 );
    col += 0.5*vec3(2.0,0.9,0.7)*pow( sundot,612.0 );
    // clouds
    col = Cloud(col,ro,rd,vec3(2.0,1.05,2.0),2.);
            // .
    col = mix( col, 0.78*vec3(0.5,0.75,2.0), pow( 2.0-max(rd.y,0.1), 26.0 ) );
    return col;
}

// Function 689
float linstep(float a, float b, float t)
{
    float v=(t-a)/(b-a);
    return clamp(v,0.,1.);
}

// Function 690
void StepRS (int sId, out vec3 rms, out vec3 vms)
{
  vec3 rs;
  int mId;
  mId = sId / nSphObj;
  rs = QtToRMat (GetQ (mId)) * RSph (float (sId - mId * nSphObj));
  rms = GetR (mId) + rs;
  vms = GetV (mId) + cross (GetW (mId), rs);
}

// Function 691
vec3 rayMarch(vec3 rayDir, vec3 cameraOrigin)
{
    const int maxItter = 100;
	const float maxDist = 30.0;
    
    float totalDist = 0.0;
	vec3 pos = cameraOrigin;
	vec4 dist = vec4(epsilon);
    
    for(int i = 0; i < maxItter; i++)
	{
		dist = distfunc(pos);
		totalDist += dist.x;
		pos += dist.x * rayDir;
        
        if(dist.x < epsilon || totalDist > maxDist)
		{
			break;
		}
	}
    
    return vec3(dist.x, totalDist, dist.y);
}

// Function 692
vec3 marchAA(in vec3 ro, in vec3 rd, in vec3 bgc, in float px, in mat3 cam)
{
    float precis = px*.1;
    float prb = precis;
    float t=map(ro);
	vec3 col = vec3(0);
	float dm=100.0,tm=0.0,df=100.0,tf=0.0,od=1000.0,d=0.;
	for(int i=0;i<ITR;i++) {
		d=map(ro+rd*t);
		if(df==50.0) {
			if(d>od) {
				if(od<px*(t-od)) {
					df=od;tf=t-od;
				}
			}
			od=d;
		}
		if(d<dm){tm=t;dm=d;}
		t+=d;
		if(t>FAR || d<precis)break;
	}
	col=bgc;
    
	if(dm<px*tm)
        col=mix(shade((ro+rd*tm) - rd*(px*(tm-dm)) ,rd),col,clamp(dm/(px*tm),0.0,1.0));
    
	float qq=0.0;
	
    if((df==100.0 || tm==tf) && t < FAR) {
        ro+=cam*vec3(0.5,0.5,0.)*px*tm*1.;
        tf=tm;
        df=dm;
        qq=.01;
	}
    return mix(shade((ro+rd*tf) - rd*(px*tf-df),rd),col,clamp(qq+df/(px*tf),0.0,1.0));
}

// Function 693
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

// Function 694
mat3 RayMarch(vec3 ro, vec3 rd) {
	float dO = 0.;
    float world = 0.;
    
    for(int i=0; i<MAX_STEPS; i++) {
        // Calculate step.
    	vec3 p = ro + rd*dO;
        float dS = GetDist(p);
        
        // Determine if we passed through a portal.
        float t = intersectPlane(p, rd, vec3(0, 0, 1), vec3(0));
        float portal = intersectPortal(p, rd, vec3(0, 0, 1), vec3(0), t);
        if (portal != 0. && t < dS) {
          ro += PORTAL_TRANSLATE;
          dS = t + EPSILON;
          world += sign(dot(rd, vec3(0, 0, 1)));
        }
        
        // Determine if we should stop.
        dO += dS;
        if (dO>MAX_DIST) {
            dO = MAX_DIST;
            break;
        } else if (dS<EPSILON) {
            break;
        }
    }
    
    mat3 result;
    result[0] = vec3(ro + rd * dO);
    result[1] = vec3(dO, 0, 0);
    result[2] = vec3(world, 0, 0);
    return result;
}

// Function 695
Hit raymarch(Ray ray) {

    vec3 p = ray.ori;
    bool h = false;
    
    for(int i = 0; i < 64; i++) {
        
        float dst = dstScene(p);
        p += ray.dir * dst * .5;
        
        if(dst < .001) {
         
            h = true;
            break;
            
        }
        
    }
    
    return Hit(p,h);
    
}

// Function 696
float raymarchAO(in vec3 ro, in vec3 rd, float tmin) {
    float ao = 0.0;
    for (float i = 0.0; i < 5.0; i++) {
        float t = tmin + pow(i / 5.0, 2.0);
        vec3 p = ro + rd * t;
        float d = intersect(p).y;
        ao += max(0.0, t - 0.5 * d - 0.05);
    }
    return 1.0 - 0.00125 * ao;
}

// Function 697
float march(inout vec3 p, vec3 cam, float tol, float maxd, int niter, out bool hit)
{
    //vec3 p = init;
    hit = false;
    float t = 0.;
    for (int i = 0; i < niter && !hit; ++i) {
        float dist = scene(p);
        hit = dist*dist < tol*tol;
        p += dist * cam;
        t += dist;
        if (t > maxd) break; //t = distance(p,init)
    }
    return t;
}

// Function 698
void TestSphereMarch(in vec3 rayPos, inout SRayHitInfo info, in vec4 sphere, in SMaterial material)
{
    float dist = SphereDistance(sphere, rayPos);
    if (dist < info.dist)
    {
        info.objectPass = OBJECTPASS_RAYMARCH;
        info.dist = dist;        
        info.normal = normalize(rayPos - sphere.xyz);
        info.material = material;
    }    
}

// Function 699
Hit RayMarchLowDetailModel(vec3 ro, vec3 dir) 
{
    vec3 P = vec3(0.,0.,0.);
    float t = 0.;
    while(t < MAX_MARCHING_DISTANCE) 
    {
        P = ro + t*dir;
        Hit hit = SD_LowDetailModel(P);
        if((hit.d)<0.01) 
    	{
            hit.normal = normalLowDetailModel(P);
            hit.tangent = tangent(hit.normal);
            hit.binormal = normalize(cross(hit.normal, hit.tangent));
			return hit;
        }
        t+=hit.d;
    }
    return Hit(-1, MAX_MARCHING_DISTANCE,vec3(0.),vec3(0.),vec3(0.),vec3(0.));
}

// Function 700
vec3 march (vec3 ro, vec3 rd)
{
    float pixelSize = 1. / iResolution.x;
    bool forceHit = true;
    float infinity = 10000000.0;
    float t_min = .0000001;
    float t_max = 1000.0;
    float t = t_min;
    vec3 candidate = vec3 (t_min, .0, .0);
    vec3 candidate_error = vec3 (infinity, .0, .0);
    float w = LARGE_STEP;
    float lastd = .0;
    float stepSize = .0;
    float sign = map (ro).x < .0 ? -1. : 1.;

    for (int i = 0; i < MAX_STEPS; i++)
	{
        float signedd = sign * map (ro + rd * t).x;
        float d = abs (signedd);
        bool fail = w > 1. && (d + lastd) < stepSize;

        if (fail) {
            stepSize -= w * stepSize;
            w = SMALL_STEP;
        } else {
            stepSize = signedd * w;
        }

		lastd = d;

        float error = d / t;
        if (!fail && error < candidate_error.x) {
            candidate_error.x = error;
            candidate.x = t;
        }

        if (!fail && error < pixelSize || t > t_max) {
        	break;
		}

        candidate_error.y = map (ro + rd * t).y;
        candidate.y = candidate_error.y;

        candidate_error.z = float (i);
        candidate.z = candidate_error.z;

        t += stepSize;
 
	}

    if ((t > t_max || candidate_error.x > pixelSize) && !forceHit) {
        return vec3 (infinity, .0, .0);
    }

	return candidate;
}

// Function 701
vec3 stepspace(
  in vec3 p,
  in float s)
{
  return p-mod(p-s/2.0,s);
}

// Function 702
vec4 rayMarch(vec3 rayOrigin, vec3 rayStep, out vec3 pos)
{
	vec4 sum = vec4(0, 0, 0, 0);
	pos = rayOrigin;
	for(int i=0; i<_VolumeSteps; i++) {
		vec4 col = volumeFunc(pos);
		col.a *= _Density;
		// pre-multiply alpha
		col.rgb *= col.a;
		sum = sum + col*(1.0 - sum.a);	
		pos += rayStep;
	}
	return sum;
}

// Function 703
vec4 RayMarchInteriorGlass(vec3 ro, vec3 dir) {
    float traveled = 0.0;
    vec2 distAndMaterial = vec2(0);
    
    for (int i=ZERO; i < 50; ++i){
        distAndMaterial = sceneInsideGlassMaterials(ro + dir * traveled);
        traveled += distAndMaterial.x;
        if (distAndMaterial.x < .01 || distAndMaterial.x > MAXDISTANCE) {
            break;
        }
    }
    
    vec3 hitPoint = ro + dir * traveled;
    
    vec3 color = vec3(1);
    color = GetColor2(distAndMaterial, hitPoint, traveled);
    return vec4(color, traveled);
}

// Function 704
float march(vec3 o, vec3 r){
 	float t = 0.1;
    float precis = 1e-4;
    for(int i=0;i<200;++i){
        vec3 p = o+r*t;
        float d = map(p);
        if (abs(d) < t*precis) return t;
        t += d/(1.0+0.3*d);
        if(t>limit || t < 0.0){
            return limit;
        }
    }
    return t;
}

// Function 705
Result raymarchDE(Ray ray, float distfactor, float maxtravel)
{
	float travelled=0.0;
    for (int i=0; i<DE_ITERATIONS; i++)
    {
    	vec2 res = sceneSDF(ray.pos);
                        
        ray.pos += res.x*ray.dir*distfactor; 
        travelled += res.x*distfactor;
        
        if (travelled>maxtravel)
            break;
    }     
    
    Result result = resultsDE(ray.pos);
    result.travelled=travelled;
    return result;
}

// Function 706
vec3 raymarch(samplerCube s, vec3 ro, vec2 auv, vec3 lp, float dn, float fb, float fp, vec3 col)
{
    vec3 rd = getCamera(ro, vec3(auv, -0.45));
    float t = 0.0;
    float d = 0.0;
    vec3 color = vec3(0.0);
    for(int x = 0; x < MAX_STEPS_MARCH; x++)
    {
        vec3 pos = ro + t * rd;
        float d = length(pos) - SPHERE_RADIUS;
        if(d < 0.001)
        {
            vec3 n = normalize(pos);

            vec3 l = normalize(-lp - ro);
			vec3 v = normalize(pos - ro);
			vec3 r = normalize(reflect(-l, n));
            d = clamp(abs(ro.z - pos.z), 0.0, 1.0);
			float spec = 0.7 * clamp(pow(max(dot(r, v), 0.0), 128.0), 0.0, 1.0);
            float diff = dot(normalize(lp), n) * 0.4;
            
            vec3 a = texture(s, reflect(rd, n * dn)).rgb;
            vec3 col = mix(a, col, 0.70) + vec3(diff + spec);
            
            // like a Fresnel coefficient
            float fc = min(pow(distance(n, ro), fp) * 2.0, fb);
            
            col = mix(col, a, fc);
            return col;
        }
        
        t += d;
    }
    
    return texture(s, rd).rgb;
}

// Function 707
vec2 march(vec3 ro, vec3 rd){
	float t=0. , d = far, it = 0.;
    for (int i=0;i<iter;i++){
     	t += (d = DE(ro+t*rd));
        if(d<eps || t> far) break;
        it += 1.;
    }
    return vec2(t,it/float(iter));
}

// Function 708
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

// Function 709
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

// Function 710
float shadow_march(vec4 pos, vec4 dir, float distance2light, float light_angle, inout object co)
{
	float light_visibility = 1.;
	float ph = 1e5;
	pos.w = map(pos.xyz, co);
	int i = min(0, iFrame);
	for (; i < shadow_steps; i++) {
	
		dir.w += pos.w;
		pos.xyz += pos.w*dir.xyz;
		pos.w = map(pos.xyz, co);
		
		float y = pos.w*pos.w/(2.0*ph);
        float d = (pos.w+ph)*0.5;
		float angle = d/(max(0.00001,dir.w-y)*light_angle);
		
        light_visibility = min(light_visibility, angle);
		
		ph = pos.w;
		
		if(dir.w >= distance2light)
		{
			break;
		}
		
		if(dir.w > maxd || pos.w < max(mind*dir.w, 0.0001))
		{
			return 0.;
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

// Function 711
RayPayLoad RayMarch(vec3 marchingPosition, vec3 rayDirection, float marchedDistance){
    RayPayLoad result;
    result.minDistToScene = FAR_CLIP;
    result.marchedDistance = marchedDistance;
    for (int i = 0; i < 1024; i++) {
        SceneSamplePayLoad scenePayLoad = GetClosestObject(marchingPosition);
        marchingPosition = marchingPosition + rayDirection * scenePayLoad.distToScene;
        result.marchedDistance = result.marchedDistance + scenePayLoad.distToScene;
        result.minDistToScene = min(result.minDistToScene, scenePayLoad.distToScene);
        if (abs(scenePayLoad.distToScene) < EPSILON){
            // Found object
            result.hitScene = true;
            result.obj = scenePayLoad.closestObj;
            result.hitPosition = marchingPosition;
            result.normal = Normal(scenePayLoad.closestObj, marchingPosition);
            break;
        }
        if (result.marchedDistance > FAR_CLIP){
            // Went too far
            result.hitScene = false;
            break;
        }
    }
    return result;
}

// Function 712
vec2 raymarch(vec3 ro, vec3 rd, in float tmin, in float tmax) {
    vec2 m = vec2(-1.0, -1.0);
    vec2 res = vec2(tmin, -1.0);
    res.x = tmin;
	for( int i=0; i<NUM_STEPS; i++ )
	{
        m = map(ro + res.x*rd);
		if( m.x<tmin || res.x>tmax ) break;
		res.x += m.x;
        res.y = m.y;
	}
    if( res.x>tmax ) res.y=-1.0;
	return res;
}

// Function 713
float march(in vec3 ro, in vec3 rd, in float startf, in float maxd)
{
	float precis = 0.01;
    float h=precis*2.0;
    float d = startf;
    for( int i=0; i<MAX_ITER; i++ )
    {
        if( abs(h)<precis||d>maxd ) break;
        d += h;
	    float res = map(ro+rd*d).x;
        h = res;
    }
	return d;
}

// Function 714
float march(in vec3 ro, in vec3 rd)
{
    float precis = .3;
    float h= 1.;
    float d = 0.;
    for( int i=0; i<17; i++ )
    {
        if( abs(h)<precis || d>70. ) break;
        d += h;
        vec3 pos = ro+rd*d;
        pos.y += .5;
	    float res = map(pos)*7.;
        h = res;
    }
	return d;
}

// Function 715
float raymarch(vec3 ori, vec3 dir) {
    float t = 0.;
    for(int i = 0; i < MAX_ITERATIONS; i++) {
    	float scn = dstScene(ori+dir*t);
        if(scn < EPSILON*t || t > MAX_DISTANCE)
            break;
        t += scn * .75;
    }
    return t;
}

// Function 716
void MarchLight(inout RayInfo r, float startTime, float maxDist)
{
    float totalDist = 0.0;
    vec3 origin = r.pos;
    
    for (r.iter=0;r.iter<maxStepRayMarching;r.iter++){
        r.time = startTime; 
    	SetTime(r.time);
        r.dist = map(r.pos,-1);
        totalDist += r.dist;
        
        
        r.pos+= r.dir*(r.dist);
        if (abs(r.dist)<rayEps || totalDist > maxDist)
        {
            break;
        }
    }
}

// Function 717
Hit RayMarch(vec3 ro, vec3 rd, int bounce){
	float dO = 0.;
    Hit hit;
    
    int steps = MAX_STEPS;
    
    if(bounce > 0){
        steps /= bounce*4;
    }
    
    for(int i = 0; i < steps; i++){
        vec3 p = dO*rd+ro;
        hit = GetDist(p);
        dO += hit.d;
        if(hit.d < SURFACE_DIST || dO > MAX_DIST){
            break;
        }
    }
    return Hit(dO, hit.ID);
}

// Function 718
void TestBezierMarch(in vec3 rayPos, inout SRayHitInfo info, in vec3 A, in vec3 B, in vec3 C, in float width, in SMaterial material)
{
    float dist = BezierDistance(rayPos, A, B, C, width);
    if (dist < info.dist)
    {
        info.objectPass = OBJECTPASS_RAYMARCH;
        info.dist = dist;    
        info.normal = vec3(1.0f, 0.0f, 0.0f); // TODO: figure out the normal, if you ever need it! finite differences? dunno.
        info.material = material;
    }    
}

// Function 719
vec3 raymarch(vec3 ro, vec3 rd, inout vec3 finalPos, vec3 eye) {
	float t = 0.0;
	const int maxIter = 100;
	const float maxDis = 300.0;
	float d = 0.0;
	vec3 p = vec3(-1.0, -1.0, -1.0);
	vec3 col = vec3(0);
	const int jumps = 3;
	float ref = 1.0;
	vec3 scatteredLight = vec3(0.0);
	float transmittance = 1.0;
	for (int j = 0; j < jumps; j++) {
		for (int i = 0; i < maxIter; i++) {
			p = ro + rd * t;

			vec2 res = map(p, rd);
			d = res.x;
			float fogAmount = 0.01;
			float lightDis = -1.0;
			vec3 light = evaluateLight(p, lightDis);
			d = min(min(d, 1.0), max(lightDis, 0.05));
			vec3 lightIntegrated = light - light * exp(-fogAmount * d);
			scatteredLight += transmittance * lightIntegrated;
			transmittance *= exp(-fogAmount * d);

			t += d;
			float m = res.y;
			bool end = i == maxIter - 1 ||t > maxDis;
			if (d < 0.01 || end) {
				vec3 c = vec3(1);
				vec3 normal = getNormal(p, rd);
				if (m == MAT_WALL) {
					c = vec3(1,0,0);
				} else if (m == MAT_SPIN) {
					c = vec3(0.5);
				} else if (m == MAT_GROUND) {
					vec3 q = floor(p);
					c = vec3(0.3,0.3,1);
				}

				c *= occlusion(p, normal, rd);
				addLightning(c, normal, eye, p);
				if (end) {
					transmittance = 0.0;
				}
				col = mix(col, transmittance * c + scatteredLight, ref);
				if (m == MAT_SPIN) {
					ref *= 0.8;
				} else {
					ref = 0.0;
				}
				rd = reflect(rd, getNormal(p, rd));
				ro = p + rd*0.05;
				t = 0.0;
				break;
			}
			if (t > maxDis) {
				break;
			}
		}

		if (ref < 0.1) {
			break;
		}
	}
	finalPos = p;
	return col;
}

// Function 720
float rayMarch(vec3 pos, vec3 dir, vec2 uv, float d)
{
    d = max(d,.0);
    float oldD = d;
    bool hit = false;
    for (int i = 0; i < 80; i++)
    {
		float de = mapDE(pos + dir * d);

		if(de < sphereSize(d) || d > 2000.0) break;
		oldD = d;

		d += 0.5*de;
   
    }
	if (d > 2000.0)
		oldD = 2000.0;

    
    return oldD;
}

// Function 721
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

// Function 722
float March(vec3 Dir, vec3 eye) {
    float depth = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        float dist = sceneSDF(eye + (Dir*depth));
        if (dist < EPSILON) {
            return depth;
        }
        depth += dist;
        if (depth >= MAX_DIST) {
            return MAX_DIST;
        }
    }
    return MAX_DIST;
}

// Function 723
bool RayMarch(vec3 org, vec3 dir, out vec3 p)
{
	p=org;
	bool hit = false;
	float dist = .0;
	// 'Break'less ray march...
	for(int i = 0; i < 120; i++)
	{
		if (!hit && dist < 25.0)
		{
			p = org + dir*dist;
			float d = Scene(p);
			if (d < 0.05)
			{
				hit = true;
			}
			dist += d*.5;
		}
	}
	return hit;
}

// Function 724
float sinstep(float x)
{
    return (1. - cos(PI * x)) * .5;
}

// Function 725
float marchingCube(vec3 p, float t) {
	p.z -= floor(t);
	p.y += 0.5;
	p.z -= 0.5;
	p.yz = rotate2d(p.yz, -mod(t, 1.0) * pi / 2.0);
	p.y -= 0.5;
	p.z += 0.5;
	return sdBox(p, vec3(0.40)) - 0.05;
}

// Function 726
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

// Function 727
float march(in vec3 ro, in vec3 rd, in float startf, in float maxd, in float j)
{
	float precis = 0.001;
    float h=0.5;
    float d = startf;
    for( int i=0; i<MAX_ITER; i++ )
    {
        if( abs(h)<precis||d>maxd ) break;
        d += h*1.2;
	    float res = map(ro+rd*d, j).x;
        h = res;
    }
	return d;
}

// Function 728
void MarchPOV(inout Ray r, float startTime
){//dpos = vec3(-2.2,0,0)
 ;//lorentzF = LorentzFactor(length(dpos))
 ;float speedC = length(dpos)/cSpe
 ;vec3 nDpos = vec3(1,0,0)
 ;if(length(dpos)>0.)nDpos = normalize(dpos)
 ;//shrink space along vel axis (length contraction of field of view)
 ;float cirContraction = dot(nDpos,r.dir)*(LorentzFactor(length(LgthContraction*dpos)))
 ;vec3 newDir = (r.dir - nDpos*dot(nDpos,r.dir)) + cirContraction*nDpos
 ;r.dir = normalize(newDir)
 ;float dDirDpos = dot(dpos,r.dir)
 ;// Aberration of light, at high speed (v) photons angle of incidence (a) vary with lorenz factor (Y) :
 ;// tan(a') = sin(a)/(Y*(v/c + cos(a)))
 ;// velComponentOfRayDir' = Y*(velComponentOfRayDir+v/c)
 ;float lightDistortion = lorentzF*(dot(-nDpos,r.dir)+speedC)
 ;r.dir=mix(r.dir
           ,normalize((r.dir-nDpos*dot(nDpos,r.dir))-lightDistortion*nDpos)
           ,FOVAberrationOfLight)
 ;//Classical Newtown Mechanic instead would be
 ;//r.dir = normalize(r.dir-dpos/cSpe)
 ;for (r.iter=0;r.iter<maxStepRayMarching;r.iter++
 ){float camDist = length(r.b - objPos[oCam])//es100 error , no array of class allowed
  ;float photonDelay = -camDist*cLag/cSpe
  //takes dilated distance x/Y and find the time in map frame with :
  // v = -dDirDpos (dot product of direction & velocity, because we want to transform from cam frame to map frame)
  // Y = lorentzFactor
  //
  // t' = Y(t-v*(x/Y)/c²)
  // t' = Y(0-v*(x/Y)/c²)
  // t' = Y(v*x/Y)/c²
  // t' = vx/c²
  ;float relativeInstantEvents = SimultaneousEvents*dDirDpos*camDist/(cSpe*cSpe)
  ;r.time = startTime
  ;r.time += mix(relativeInstantEvents,photonDelay,cLag)
  ;SetTime(r.time)
  ;r.dist = map(r.b,-1)
  ;//Gravitational lens
  ;vec3 blackHoleDirection = (objPos[oBlackHole]-r.b)//es100 error , no array of class allowed
  ;r.dir+=(1./RayPrecision)*r.dist*normalize(blackHoleDirection)*BlackHoleMassFactor/(length(blackHoleDirection)*cSpe*cSpe)
  ;r.dir = normalize(r.dir)
  ;if(abs(r.dist)<rayEps)break
  ;r.b+= (1./RayPrecision)*(r.dist)*(r.dir);}
 ;//r.b = origin + r.dir*min(length(r.b-origin),maxDist)
 ;r.surfaceNorm = GetNormal(r.b).xyz;}

// Function 729
MarchResult raymarch(const in Camera cam)
{
	MarchResult march;
	
	float depth = 0.0;
	for(int i = 0; i < MAX_ITERATIONS; ++i)
	{
		march.position = cam.position + cam.rayDir * depth;
		march.hit = map(march.position);
		
		
		if(march.hit.dist <= EPSILON || depth >= MAX_DEPTH)
		{
			break;
		}
		
		depth += march.hit.dist;
	}
	
	if(depth < MAX_DEPTH)
	{
		vec3 eps = vec3(EPSILON, 0, 0);
		march.normal=normalize(
			   vec3(march.hit.dist - map(march.position-eps.xyy).dist,
					march.hit.dist - map(march.position-eps.yxy).dist,
					march.hit.dist - map(march.position-eps.yyx).dist));
		
	}
	
	return march;
}

// Function 730
vec4 marchV(in vec3 ro, in vec3 rd, in float t, in vec3 bgc)
{
	vec4 rz = vec4( 0.0 );
	
	for( int i=0; i<150; i++ )
	{
		if(rz.a > 0.99 || t > 200.) break;
		
		vec3 pos = ro + t*rd;
        float den = mapV(pos);
        
        vec4 col = vec4(mix( vec3(.8,.75,.85), vec3(.0), den ),den);
        col.xyz *= mix(bgc*bgc*2.5,  mix(vec3(0.1,0.2,0.55),vec3(.8,.85,.9),moy*0.4), clamp( -(den*40.+0.)*pos.y*.03-moy*0.5, 0., 1. ) );
        col.rgb += clamp((1.-den*6.) + pos.y*0.13 +.55, 0., 1.)*0.35*mix(bgc,vec3(1),0.7); //Fringes
        col += clamp(den*pos.y*.15, -.02, .0); //Depth occlusion
        col *= smoothstep(0.2+moy*0.05,.0,mapV(pos+1.*lgt))*.85+0.15; //Shadows
        
		col.a *= .95;
		col.rgb *= col.a;
		rz = rz + col*(1.0 - rz.a);

        t += max(.3,(2.-den*30.)*t*0.011);
	}

	return clamp(rz, 0., 1.);
}

// Function 731
vec3 march(vec3 from, vec3 dir) {
	float td, d, g = 0.;
    vec3 c = vec3(0.), p;
    for (int i = 0; i < 60; i++) {
    	p = from + dir * td;
        d = de(p);
        td += d;
        if (td > 50. || d < det) break;
		g += smoothstep(-4.,1.,p.x);
    }
    if (d < det) {
    	p -= det * dir * 2.;
        vec3 col = color(objid, p);
        vec3 n = normal(p);
        c = shade(p, dir, n, col, objid);
        //cl1 = clamp(cl1, 0., 1.);
        float cl1 = clouds(p, dir);
		vec3 nc = normal_clouds(p, dir);
        c = mix(c, .1 + cloud_color * max(0., dot(normalize(ldir), nc)), clamp(cl1,0.,1.));
    }
    else
    {
        vec2 pp = dir.xy + vec2(.434, .746);
        float m1 = 100., m2 = m1;
        for (int i=0; i < 6; i++) {
        	pp = abs(pp) / dot(pp, pp) - .9;
        	m1 = min(m1, length(pp * vec2(4.,1.)));
        	m2 = min(m2, length(pp * vec2(1.,4.)));
        }
		c += pow(max(0., 1. - m1), 30.) * .5;		
		c += pow(max(0., 1. - m2), 30.) * .5;		
    }
    g /= 60.;
    return c + (pow(g, 1.3) + pow(g,1.7) * .5) * atmo_color * .5;
}

// Function 732
SceneData raymarch(in vec3 ro, in vec3 rd)
{
    float t = 0.0;
    SceneData sceneOutput = SceneData(MAX_DIST, kInvMat);
    for (int i = 0; i < 90; ++i)
    {
        SceneData scene = map(ro + rd * t);
        if (scene.dist < MIN_DIST)
        {
            sceneOutput = SceneData(t, scene.material);
            break;
        }
        t += scene.dist;
        if (t > MAX_DIST)
        {
            break;
        }
    }
    return sceneOutput;
}

// Function 733
void StepAS (int sId, out vec3 am, out vec3 wam)
{
  vec3 dr, rm, rms, vms, fc;
  float rSep;
  int mId, sIdN;
  mId = sId / nSphObj;
  rm = GetR (mId);
  rms = GetRS (sId);
  vms = GetVS (sId);
  am = vec3 (0.);
  wam = vec3 (0.);
  for (int mIdN = VAR_ZERO; mIdN < nObj; mIdN ++) {
    if (mIdN != mId && length (rms - GetR (mIdN)) < farSep) {
      for (int j = VAR_ZERO; j < nSphObj; j ++) {
        sIdN = mIdN * nSphObj + j;
        dr = rms - GetRS (sIdN);
        rSep = length (dr);
        if (rSep < 1.) {
          fc = FcFun (dr, rSep, vms - GetVS (sIdN));
          am += fc;
          wam += cross (rms - rm, fc);
        }
      }
    }
  }
  rSep = abs (rms.y);
  if (rSep < 1.) {
    fc = FcFun (vec3 (0., rms.y, 0.), rSep, vms);
    am += fc;
    wam += cross (rms - rm, fc);
  }
}

// Function 734
float rayMarch(vec3 ro, vec3 rd){
 	float R0 = 0.;
    for(int i = 0; i < MAX_COUNT; i++){
     	vec3 p = ro + R0 * rd;
        float d = getDist(p);
        R0 += d;
        if(d < MIN_DIST || R0 > MAX_DIST) break;
    }
    return R0;
}

// Function 735
float march(in vec3 ro, in vec3 rd) {
  const float maxd = 25.0;
  const float precis = 0.0002;
  float h = precis * 2.0;
  float t = 0.0;
  float res = 1e8;
  for (int i = 0; i < 200; i++) {
    if (h < precis || t > maxd) break;
    h = eval(ro + rd * t);
    // When inverting, limit step size to prevent
    // overshoot when coming in from a distance.
    if (invert) h = min(h,0.5);
    t += h;
  }
  if (t < maxd) res = t;
  return res;
}

// Function 736
float scaleForStep(float step) {
    return pow(1./stepScale, step);
}

// Function 737
vec2 _smoothstep(in vec2 f)
{
    return f * f * (3.0 - 2.0 * f);
}

// Function 738
float lerpstep(float a, float b, float x)
{
    return (a == b) ? step(x, a) : clamp((x - a) / (b - a), 0.0, 1.0);
}

// Function 739
float ramp_step(float steppos, float t) {
    return clamp(t-steppos+0.5, 0.0, 1.0);
}

// Function 740
vec2 rayMarch(vec3 rayDir, vec3 cameraOrigin)
{
    const int MAX_ITER = 100;
	const float MAX_DIST = 40.0;
    
    float totalDist = 0.0;
	vec3 pos = cameraOrigin;
	float dist = EPSILON;
    
    for(int i = 0; i < MAX_ITER; i++)
	{
		dist = distfunc(pos);
		totalDist += dist;
		pos += dist*rayDir;
        
        if(dist < EPSILON || totalDist > MAX_DIST)
		{
			break;
		}
	}
    
    return vec2(dist, totalDist);
}

// Function 741
void Step (ivec2 iv, out vec3 r, out vec3 v)
{
  vec3 dr, f;
  vec2 s;
  float fDamp, dt, rSep, w, t;
  IdNebs ();
  fDamp = 0.5;
  r = GetR (vec2 (iv));
  v = GetV (vec2 (iv));
  f = PairForce (r) + SpringForce (iv, r, v) + BendForce (iv, r) +
     NormForce (iv, r, v);
  dr.xz = -0.55 * GrndNf (r).xz;
  dr.y = r.y + 0.55 - GrndHt (r.xz - dr.xz);
  rSep = length (dr);
  if (rSep < 1.) f += fOvlap * (1. / rSep - 1.) * dr;
  f -= fDamp * v;
  f.y -= grav;
  s = float (nBallE) - vec2 (iv);
  w = max (1. - dot (s, s) / 8., 0.);
  f += 100. * w * (rLead - r);
  t = mod (ntStep / 500., 8.);
  f.y += 200. * w * ((t < 6.) ? t : 3. * (8. - t));
  dt = 0.02;
  v += dt * f;
  r += dt * v;
}

// Function 742
vec3 rayMarch(vec3 rd, vec3 ro, out float d){
    vec3 p = ro;
    float s = prec;
    for(int i=0;i<rayMarchSteps;i++)
	{      
		if (s<prec||s>maxd) break;
		s = map(p) * marchPrecision;
		d += s;
		p = ro+rd*d;
	}
    return p;
}

// Function 743
bool rayMarch(vec3 r0, vec3 rd, inout float d)
{
    d = 0.0;
    for(int i = 0; i < 100; ++i)
        {
            vec3 p = r0 + d * rd;
            float t = map(p);
            d += t;
            if(abs(t) < 0.001)
            {
                return true;
            }
            if(d > 200.0) break;
        }
    return false;
}

// Function 744
float Linstep(float a, float b, float t)
{
	return clamp((t-a)/(b-a),0.,1.);
}

// Function 745
vec3 raymarchTerrain(in vec3 ro, in vec3 rd, in float tmin, in float tmax) {
    float t = tmin;
    vec3 res = vec3(-1.0);
    float breakOut = 0.0;
    int i = 0;
    while (i < 10 && breakOut != 1.0) {
        vec3 p = ro + rd * t;
        res = vec3(map(p), t);
        float d = res.y;
        if (d < (0.001 * t) || t > tmax)
            breakOut = 1.0;
        t += 0.5 * d;
        i++;
    }
    return res;
}

// Function 746
vec2 march( vec3 ro, vec3 rd ) {
    vec2 t = vec2(0);
    
    for (int i = 0; i < 64; ++i) {	
        vec3 ra = ro+rd*t.x;
		vec2 h = map( ra );
        if (h.x < prc) {
        	t.y = h.y;
        }
        t.x += h.x;
    }
    return t;
}

// Function 747
float march (vec3 ro, vec3 rd, float time) {
 	float d = EPSILON;
    float t = 0.0;
    
    for (int i = 0; i < MAXSTEPS; ++i) {
     	vec3 p = ro + rd * d;
       	t = map (p, time);
        if (t < EPSILON || d >= MAXDIST) 
            break;
        d += t;
    }
    return d;
    
}

// Function 748
float march(inout vec3 pos, inout vec3 dir)
{
    float eps = .0001;
    float mat=-1.;
    for(int i=0;i<50;i++)
    {
        vec2 dm=getDistM(pos);
        float d=dm.x;
        pos+=dir*d*.9;
        if(d<eps) { mat=dm.y; break; }
    }
    return mat;
}

// Function 749
void Step (int mId, out vec3 r, out vec3 v)
{
  vec3 rn, vn, dr, f;
  ivec3 iv, ivN;
  float fOvlap, fSpring, fDamp, grav, rSep, spLenD, dt, fm, fmId;
  int nbId;
  IdNebs ();
  fOvlap = 1000.;
  fSpring = 50.;
  fDamp = 0.2;
  grav = 3.;
  r = Loadv4 (2 * mId).xyz;
  v = Loadv4 (2 * mId + 1).xyz;
  f = vec3 (0.);
  for (int n = 0; n < nBall; n ++) {
    rn = Loadv4 (2 * n).xyz;
    dr = r - rn;
    rSep = length (dr);
    if (n != mId && rSep < 1.) f += fOvlap * (1. / rSep - 1.) * dr;
  }
  fm = float (nBallE);
  fmId = float (mId);
  iv = ivec3 (mod (fmId, fm), mod (floor (fmId / fm), fm),
     floor (fmId / (fm * fm)));
  for (int n = 0; n < nNeb; n ++) {
    ivN = iv + idNeb[n];
    if (InLatt (ivN.x) && InLatt (ivN.y) && InLatt (ivN.z)) {
      nbId = (ivN.z * nBallE + ivN.y) * nBallE + ivN.x;
      rn = Loadv4 (2 * nbId).xyz;
      vn = Loadv4 (2 * nbId + 1).xyz;
      dr = r - rn;
      rSep = length (dr);
      f += fSpring * (spLen - rSep) * normalize (dr) - fDamp * (v - vn);
    }
  }
  spLenD = spLen * sqrt (3.);
  for (int n = 0; n < nNebD; n ++) {
    ivN = iv + idNebD[n];
    if (InLatt (ivN.x) && InLatt (ivN.y) && InLatt (ivN.z)) {
      nbId = (ivN.z * nBallE + ivN.y) * nBallE + ivN.x;
      rn = Loadv4 (2 * nbId).xyz;
      vn = Loadv4 (2 * nbId + 1).xyz;
      dr = r - rn;
      rSep = length (dr);
      f += fSpring * (spLenD - rSep) * normalize (dr) - fDamp * (v - vn);
    }
  }
  dr = hbLen - abs (r);
  f -= step (dr, vec3 (1.)) * fOvlap * sign (r) * (1. / abs (dr) - 1.) * dr +
      vec3 (0., grav, 0.) * QtToRMat (qtVu) + fDamp * v;
  dt = 0.02;
  v += dt * f;
  r += dt * v;
}

// Function 750
vec4 march(vec2 ro, vec2 rd)
{
    float t = 0., s = float(S), d;

    for(float i = 0.; i < S; i++)
    {
        d = scene(ro+rd*t);

        if (d < P || t > D || i > floor(T))
        {
            s = float(i);
            break;
        }

		visualize(ro,rd,t,d,i);
        
        t += d;
    }
    
    return vec4(ro+rd*t,d,s);
}

// Function 751
float march(vec3 ro, vec3 rd){
    float t = 0.;
    for(int i=0; i<ITR; i++){
        vec3 p = ro + rd * t;
        float d = map(p);
        if(d<=MIN_DIST) break;
        t += d;
        if(t < MIN_DIST) break;
        if(t >= MAX_DIST) return MAX_DIST;
        //EX: Marching Cost
        //dbg_1F( float(i)*(1./float(ITR)) );
        //st_assert(i<90, 0);
    }
    return t;
}

// Function 752
float marchRay(vec3 ro, vec3 rd) {
  float t = 0.0;
  for(int i = 0; i < 300; i++) {
    vec3 p = ro + t * rd;
    float h = map(p);
    if (h < 0.0001) return t;
    t += h;
    if (t > farClip) return 0.0;
  }
  return t;
}

// Function 753
float sinStep(float x) {
    return sin(x-0.5)*PI * 0.5 + 0.5;
}

// Function 754
MarchRes marchRay(vec3 pos, vec3 dir, float speed)
{
 	MarchRes res;
    Object o;
    
    res.totalDist = 0.001;
    res.minDist = 1000.0;

    for(int x=0; x<250; x++)
    {
 		res.curRay = pos + (dir*res.totalDist);
        
        o = map(res.curRay);
        
        if(abs(o.dist) < 0.001)
        {
            res.obj = o;
            break;
        }
        else if(res.totalDist >= VIEW_DIST) break;
           
        
        if(o.dist < res.minDist) res.minDist = o.dist;
        res.totalDist += o.dist*speed;
    }
    
    if(res.totalDist < VIEW_DIST)
    {
        o.normal = calcNormal(res.curRay, o.normEps);
        
        res.obj = o;
    }
    	
    
    return res;
}

// Function 755
void Step (int mId, out vec3 rm, out vec3 vm, out float rad)
{
  vec4 p;
  vec3 dr, am;
  float fOvlap, fBond, fCent, rSep, radAv, dt, s;
  vec4 drw;
  fOvlap = 1000.;
  fBond = 0.005;
  fCent = 1.;
  p = Loadv4 (2 * mId);
  rm = p.xyz;
  rad = p.w;
  vm = Loadv4 (2 * mId + 1).xyz;
  am = vec3 (0.);
  for (int n = 0; n < nBall; n ++) {
    p = Loadv4 (2 * n);
    dr = rm - p.xyz;
    rSep = length (dr);
    radAv = 0.5 * (rad + p.w);
    if (n != mId) {
      s = radAv / rSep - 1.;
      am += ((s > 0.) ? fOvlap : fBond) * s * dr;
    }
  }
  radAv = 0.5 * (rad + 1.);
  drw = vec4 ((hbSize - abs (rm)) * (1. - 2. * step (0., rm)), 0.);
  for (int nf = 0; nf < 3; nf ++) {
    dr = (nf == 1) ? drw.wyw : ((nf == 0) ? drw.xww : drw.wwz);
    rSep = length (dr);
    if (rSep < radAv) am += fOvlap * (radAv / rSep - 1.) * dr;
  }
  am -= fCent * rm;
  dt = 0.02;
  vm += dt * am;
  rm += dt * vm;
}

// Function 756
float march(in vec3 ro, in vec3 rd, out float itrc, float noise)
{
    float t = noise * 0.01;
    float d = map(rd*t+ro);
    float precis = 0.0001;
    
    for (int i=0;i<=ITR;i++)
    {
        if (abs(d) < precis || t > FAR) break;
        precis = t*0.0001;
        float rl = max(t*0.02,1.);
        t += d*rl;
        d = map(rd*t+ro)*0.7;
        itrc++;
    }

    return t;
}

// Function 757
float smootherstep(float edge0, float edge1, float x) {
	float t = (x - edge0)/(edge1 - edge0);
	float t1 = t*t*t*(t*(t*6. - 15.) + 10.);
	return clamp(t1, 0.0, 1.0);
}

// Function 758
float RayMarch(vec3 ro, vec3 rd, float side) {
	float dO=0.;
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        float dS = GetDist(p)*side;
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }
    return dO;
}

// Function 759
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

// Function 760
vec4 march(vec3 ro, vec3 rd)
{
	float t = 0.0;
	for (int i=0; i<256; i++)
	{
		vec3 p = ro + rd * t;
		float d = dist(p);
		if (abs(d) < 0.01)
			return vec4(p,1.0);
		t += d;
		if (t >= 100.0)
			break;
	}
	return vec4(ro + rd * t, 0.0);
}

// Function 761
void TestSphereMarch(in vec3 rayPos, inout SRayHitInfo info, in vec4 sphere, in SMaterial material)
{
    float dist = SphereDistance(sphere, rayPos);
    if (dist < info.dist)
    {
        info.rayMarchedObject = true;
        info.dist = dist;        
        info.normal = normalize(rayPos - sphere.xyz);
        info.material = material;
    }    
}

// Function 762
bool raymarch_to_light(vec3 ray_start, vec3 ray_dir, float maxDist, float maxY, out float dist, out vec3 p, out int iterations, out float light_intensity) {
    dist = 0.; 
    float minStep = 1.0;
    light_intensity = 1.0;
	float mapDist;
    for (int i = 1; i <= MAX_RAYMARCH_ITER_SHADOWS; i++) {
        p = ray_start + ray_dir * dist;
        mapDist = mapBlocks(p, ray_dir).y;
        if (mapDist < MIN_RAYMARCH_DELTA) {
            iterations = i;
            return true;
        }
		light_intensity = min(light_intensity, SOFT_SHADOWS_FACTOR * mapDist / dist);
		dist += max(mapDist, minStep);
        if (dist >= maxDist || p.y > maxY) { break; }
    }
    return false;
}

// Function 763
float RAYMARCH_DFSS( vec3 o, vec3 L, float coneWidth )
{
    //Variation of the Distance Field Soft Shadow from : https://www.shadertoy.com/view/Xds3zN
    //Initialize the minimum aperture (angle tan) allowable with this distance-field technique
    //(45deg: sin/cos = 1:1)
    float minAperture = 1.0; 
    float t = 0.0;
    float dist = GEO_MAX_DIST;
    for( int i=0; i<6; i++ )
    {
        vec3 p = o+L*t; //Sample position = ray origin + ray direction * travel distance
        float dist = map( p ).d;
        float curAperture = dist/t; //Aperture ~= cone angle tangent (sin=dist/cos=travelDist)
        minAperture = min(minAperture,curAperture);
        t += 0.03+dist; //0.03 : min step size.
    }
    
    //The cone width controls shadow transition. The narrower, the sharper the shadow.
    return saturate(minAperture/coneWidth); //Should never exceed [0-1]. 0 = shadow, 1 = fully lit.
}

// Function 764
vec3 raymarch_o354278(vec2 uv) {
    vec3 cam=vec3(1.250000000+sin(iTime*0.25)*0.5,1.400000000+cos(iTime*0.2)*0.5,1.500000000);
	vec3 lookat=vec3(0.000000000,0.000000000,0.000000000);
	vec3 ray=normalize(lookat-cam);
	vec3 cX=normalize(cross(vec3(0.0,-1.0,0.0),ray));
	vec3 cY=normalize(cross(cX,ray));
	vec3 rd = normalize(ray*1.000000000+cX*uv.x+cY*uv.y);
	vec3 ro = cam;
	
	float d=0.;
	vec3 p=vec3(0);
	vec2 dS=vec2(0);
	march_o354278(d,p,dS,ro,rd);
	
    vec3 color=vec3(0.0);
	vec3 objColor=(dS.y<0.5)?o354278_input_tex3d_a(p):o354278_input_tex3d_b(p);
	float fog=max(1.0-(d/50.0),0.0);
	vec3 light=normalize(vec3(0.950000000,1.200000000,0.400000000));
	if (d<50.0) {
	    vec3 n=normal_o354278(p);
		float l=clamp(dot(-light,-n),0.0,1.0);
		float r=clamp(dot(reflect(rd,-n),light),0.0,1.0);
		float cAO=calcAO_o354278(p,n);
		float shadow=calcSoftshadow_o354278(p,light,0.05,5.0);
		color=min(vec3(max(shadow,0.200000000)),max(l,0.200000000))*max(cAO,0.200000000)*objColor+pow(r,200.000000000)*0.850000000;
	} else {
	    color=o354278_p_SkyColor_gradient_fct(rd.y).xyz;
	}
    return color*(fog)+o354278_p_SkyColor_gradient_fct(rd.y).xyz*(1.0-fog);
}

// Function 765
vec3 march (vec3 p, vec3 d) {
    float rxcount = 0.;
    float shiny = 1.;
    vec3 finalcol = vec3(0., 0., 0.);
    float distfromcam = 0.;
    for (int i=0; i<100; ++i) {
        float SDFp = SDF(p);
        if (SDFp < 1e-1) {
            p = p+d*SDFp*.995;
            vec3 TEXpd = TEX(p, d);
            float rxindexp = rxindex(p);
            if (rxindexp == 0. || rxcount > 4.) {
                // hits solid object, final color determined
                finalcol = finalcol+TEXpd*shiny*(1.-rxindexp);
                return finalcol;
            }
            if (rxcount > 3.) {
                // waaaay to many reflections reflect background col
                break;
            }
            finalcol = finalcol+TEXpd*shiny*(1.-rxindexp);
            shiny = shiny*rxindexp;
            d = reflect(d, dSDF(p));
            p = p+d*.2;
            ++rxcount;
        }
        float DE = SDFp;
        if (0. < DE && DE < 10.) {
            DE *= .7;
        }
        p = p+d*DE;
        distfromcam += DE;
        if (distfromcam > 155.) {
            break;
        }
    }
    // diverges waaaay out into the sky. reflect sky color
    if (rxcount > 0.) {
        return finalcol+bg(d)*shiny;
    }
    return bg(d);
}

// Function 766
vec2 ray_marching( vec3 origin, vec3 dir, float start, float end ) {
	
    float depth = start;
	for ( int i = 0; i < max_iterations; i++ ) {
        vec2 distResult = map( origin + dir * depth );
		float dist = distResult.x;
        dist -= dist*0.95;
		if ( dist < stop_threshold ) {
			return vec2(depth,distResult.y);
		}
		depth += dist;
		if ( depth >= end) {
			return vec2(end,-1.0);
		}
	}
	return vec2(end,-1.0);
}

// Function 767
float Raymarch( const in C_Ray ray )
{        
    float fDistance = 0.1;
        
    for(int i=0;i<=kMaxIterations;i++)              
    {
        float fSceneDist = GetDistanceScene( ray.vOrigin + ray.vDir * fDistance );
        
        if((fSceneDist <= 0.01) || (fDistance >= 1000.0))
        {
            break;
        }                        

        fDistance = fDistance + fSceneDist; 
    }

	fDistance = min(fDistance, 1000.0);
	
	return fDistance;
}

// Function 768
bool rayMarch(vec3 eye, vec3 dir, float minDistance, float maxDistance,
              out float totDist, out bool isWater)
{
    totDist = minDistance;
    vec3 pos = eye;
	for (int i = 0; i < numMarches; i++)
    {
        pos = eye + totDist * dir;
        float dist = sceneSDF(pos, isWater);
        if (dist < epsilon)
        {
            return true;
        }
        else if (dist > maxDistance)
        {
            return false;
        }
        totDist += dist * .25;
    }
    
    return false;
}

// Function 769
vec3 raymarch(vec3 pathdir, vec3 pathorig){
    vec3 pathpos = pathorig;
    pathpos += pathdir*6.0;
    vec3 surfnormal;
    float distest, lightdistest;
    int bounces = 0;
    int object = 0;
    vec3 closestpos = pathpos;
    vec3 outCol = vec3(1.0);
    for(int i = 0; i < maxmarches; i++){
        // Check if the path is done
        if(length(pathpos) > scenesize || pathpos.z < -4.0 || bounces > maxbounces){break;}
        if(light(pathpos) < collisiondist){return outCol*vec3(1.0);}

        // Find the distance to the scene
        distest = DE(pathpos);
        lightdistest = light(pathpos);

        // Michael0884: Closest Non-Colliding Position
        if(distest > min(collisiondist, lightdistest)){closestpos = pathpos;}

        // Bounce the Path if it hits something
        if(distest < collisiondist){
            int object = getmat(pathpos);
            vec4 matprops = materialproperties(pathpos, object);
            outCol *= matprops.rgb;
            surfnormal = normal(pathpos);
            pathpos = closestpos;
            pathdir = reflect(pathdir, normalize(nrand3(matprops.w, surfnormal)));
            bounces++;
        }

        // Otherwise just keep going
        else{pathpos += pathdir*min(distest, lightdistest);}
    }
    return vec3(0.0);
}

// Function 770
vec4 raymarchTerrain( in vec3 ro, in vec3 rd )
{
	float maxd = 20.0;
    float precis = 0.0001;
	float h = 1.0;
	float t = 0.1;
	
	vec4 res = vec4(0.0);
	for( int i=0; i<200; i++ )
	{
		if( abs(h)<precis||t>maxd ) break;

		res = mapTerrain( ro+rd*t );
		h = res.w*0.08;
		t += h;
	}
	if( t>maxd ) t=-1.0;
	return vec4(res.xyz,t);
}

// Function 771
bool RayMarch(
	const in Ray r, 
	const float startT, const float endT, 
	const float stp, 
	const int N,
	out float t, out float v, out int i)
{
	float t0=startT;
	t=t0;
	v=Value(r.p+r.d*t);

	if(v<0.)
		return true;

	i=0;
	for(int j=0;j<1;j+=0)
	{
		t+=max(v*.85, stp);
		float v1=Value(r.p+r.d*t);
		if(v1<0.)
		{
			// Linear interpolation between two last steps
			t=t0+(t-t0)*v/(v-v1);
			v=Value(r.p+r.d*t);
			return true;
		}
		if(t>endT)
			return false;
		i++;
		if(i>N)
			return false;
		v=v1;
		t0=t;
	}
	return false;
}

// Function 772
vec4 MarchMap(inout Ray ray) {//total, min, steps, mat
	vec4 dist = vec4(0, MARCH_maxl, 0, 0);//total, min, steps, mat
    vec2 d = vec2(1, 0);
    
    for (int i = 0; i < MARCH_itr; i++) {
        if (!(d.x > MARCH_eps && dist.x < MARCH_maxl)) break;
    	d = Map(ray.pos);
        dist.x += d.x;
        if (dist.y > d.x)
            dist.yw = d;
        dist.z++;
        
        ray.pos += d.x*ray.vel;
    }
    dist.x = min(dist.x, MARCH_maxl);
	return dist/MARCH_norm.yyzw;
}

// Function 773
vec4 raymarch( in vec3 ro, in vec3 rd, in vec2 nfplane )
{
    float glow = 0.;
	vec3 p = ro+rd*nfplane.x;
	float t = 0.;
	for(int i=0; i<256; i++)
	{
        float d = map(p)*.8;
        t += d;
        p += rd*d;
		glow += 1./256.;
		if( d < 0.0001 || t > nfplane.y )
            break;
            
	}
	
	return vec4(p,glow);
}

// Function 774
void march_o354278(out float d,out vec3 p,out vec2 dS, vec3 ro, vec3 rd){
    for (int i=0; i < 500; i++) {
    	p = ro + rd*d;
        dS = input_o354278(p);
        d += dS.x;
        if (d > 50.0 || abs(dS.x) < 0.0001) break;
    }
}

// Function 775
float RayMarch(vec3 ro, vec3 rd) {
	float dO=0.;
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        float dS = GetDist(p);
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }
    return dO;
}

// Function 776
vec3 RayMarch(in vec3 position, in vec3 direction, out int mtl){
  vec3 hitColor;
  vec3 r;
  float sg=1.;
  float nextDistance= 1.;
  float shadowReflections=MAX_SHADOW_REFLECTIONS;
  float maxDist =MAX_DIST;
  float maxSteps=MAX_STEPS;
  int hardLimit =MAX_HARD;
  float eps     =.01;
  for(int ever=3;ever!=0;++ever){//the ride (n)ever ends! unless [hardlimit<0] triggers a return. this lets me drain hardlöimit in the loop-
    nextDistance=df(position,mtl);
	position+=direction*nextDistance;
    if(nextDistance<eps){//if we hit a surface.
      vec3 n=ComputeNormal(position,mtl,eps);n=normalize(n);
      vec3 col=MaterialColor(mtl);
	  position+=n;
      if(shadowReflections>0.0){//if we stil calculate shadows (for this reflection)
		vec3 lightpos = vec3(250.0*sin(time*.005), 400.0 + 40.0 *cos(time*.002), 250.0*cos(time*.005));
        lightpos=lightpos-position;vec3 lightVector=normalize(lightpos);float lightdist=length(lightpos);
		float shadow = SoftShadow(position, lightVector, 0.3, lightdist,shadowHardness);
        if(mtl==BUILDINGS_MTL){col=mix(shadowColor,col,clamp(position.y/7.0,0.0,1.0));}
		float attenuation=clamp(dot(n,lightVector),0.0,1.0);
		shadow=min(shadow,attenuation);col=mix(shadowColor,col,shadow);
        float AO=AmbientOcclusion(position,n, 1.0, 7.0);col=mix(shadowColor,col,AO);shadowReflections-=1.0;}
      float refl=.45;//surface reflectiveness
      if (mtl==GROUND_MTL)refl=.3;//ground has other reflectiveness
      r=mix(col,r,1.-sg);//mix the color of the current ray (reflection) with the accumulated total color to be returned.
      sg=sg*refl;if(sg<.01)return r;
      direction=direction-(n*1.5*(dot(direction,n)));//direction gets reflected at surface normal. 
        //*1.5 factor means we WILL likely overstp within reflections, accumulatively for each reflection.
        //this is a reasonable fps booster for less quality in reflections.
    }hardLimit--;
    epsmod1 
    epsmod2 
    epsmod3 	
    if(maxSteps<0.||maxDist<0.||hardLimit<0){
	  if (direction.y<0.)return mix(groundColor,r,1.-sg);
                         return mix(skyColor   ,r,1.-sg);}}return vec3(3,3,3);}

// Function 777
vec2 raymarchVoxel(vec3 ro, vec3 rd, out vec3 nor) {
  vec3 pos = floor(ro);
  vec3 ri = 1.0 / rd;
  vec3 rs = sign(rd);
  vec3 dis = (pos - ro + 0.5 + rs * 0.5) * ri;
  
  float res = -1.0;
  vec3 mm = vec3(0.0);
  
  for (int i = 0; i < 96; i++) {
    float k = voxelModel(pos);
    if (k > 0.5) {
      res = k;
      break;
    }
     
    mm = step(dis.xyz, dis.yxy) * step(dis.xyz, dis.zzx);
		dis += mm * rs * ri;
    pos += mm * rs;
  }
  
  if (res < -0.5) {
    return vec2(-1.0);
  }
  
  nor = -mm * rs;
  
  vec3 vpos = pos;
  vec3 mini = (pos-ro + 0.5 - 0.5*vec3(rs))*ri;
  float t = max(mini.x, max(mini.y, mini.z));
  
  return vec2(t, 0.0);
}

// Function 778
void MarchPOV(inout Ray r, float startTime,float sscoc
){//dpos = vec3(-2.2,0,0)
 ;//lorentzF = LorentzFactor(length(dpos))
 ;float speedC = length(dpos)/cSpe
 ;vec3 nDpos = vec3(1,0,0)
 ;if(length(dpos)>0.)nDpos = normalize(dpos)
 ;//shrink space along vel axis (length contraction of field of view)
 ;float cirContraction = dot(nDpos,r.dir)*(LorentzFactor(length(LgthContraction*dpos)))
 ;vec3 newDir = (r.dir - nDpos*dot(nDpos,r.dir)) + cirContraction*nDpos
 ;r.dir = normalize(newDir)
 ;float dDirDpos = dot(dpos,r.dir)
 ;// Aberration of light, at high speed (v) photons angle of incidence (a) vary with lorenz factor (Y) :
 ;// tan(a') = sin(a)/(Y*(v/c + cos(a)))
 ;// velComponentOfRayDir' = Y*(velComponentOfRayDir+v/c)
 ;float lightDistortion = lorentzF*(dot(-nDpos,r.dir)+speedC)
 ;r.dir=mix(r.dir
           ,normalize((r.dir-nDpos*dot(nDpos,r.dir))-lightDistortion*nDpos)
           ,FOVAberrationOfLight)
 ;//Classical Newtown Mechanic instead would be
 ;//r.dir = normalize(r.dir-dpos/cSpe)
 ;for (r.iter=0;r.iter<maxStepRayMarching;r.iter++
 ){float camDist = length(r.b - objPos[oCam])//es100 error , no array of class allowed
  ;float photonDelay = -camDist*cLag/cSpe
  //takes dilated distance x/Y and find the time in map frame with :
  // v = -dDirDpos (dot product of direction & velocity, because we want to transform from cam frame to map frame)
  // Y = lorentzFactor
  //
  // t' = Y(t-v*(x/Y)/c²)
  // t' = Y(0-v*(x/Y)/c²)
  // t' = Y(v*x/Y)/c²
  // t' = vx/c²
  ;float relativeInstantEvents = SimultaneousEvents*dDirDpos*camDist/(cSpe*cSpe)
  ;r.time = startTime
  ;r.time += mix(relativeInstantEvents,photonDelay,cLag)
  ;SetTime(r.time)
  ;r.dist = map(r.b,-1)
  ;//Gravitational lens
  ;vec3 blackHoleDirection=(objPos[oBlackHole]-r.b)//es100 error , no array of class allowed
  ;r.dir+=(1./RayPrecision)*r.dist*reciprocalLipschitz
      *normalize(blackHoleDirection)*BlackHoleMassFactor/(length(blackHoleDirection)*cSpe*cSpe)
  ;r.dir = normalize(r.dir)
  ;if(abs(r.dist)<getEpsToFps(sscoc))break
  ;r.b+= (1./RayPrecision)*(r.dist)*reciprocalLipschitz*(r.dir);}
 ;//r.b = origin + r.dir*min(length(r.b-origin),maxDist)
 ;r.surfaceNorm = GetNormal(r.b).xyz;}

// Function 779
vec4 march(inout vec3 pos, vec3 dir, out vec2 uv)
{
    // cull the sphere
    if(length(pos-dir*dot(dir,pos))>1.07) 
    	return vec4(0,0,0,1);
    
    float eps=0.001;
    float bg=1.0;
    for(int cnt=0;cnt<52;cnt++)
    {
        float d = dist(pos);
        pos+=d*dir*.8;
        if(d<eps) { bg=0.0; break; }
    }
    vec3 n = normalize(getGrad(pos,.001));
    uv=distUV(pos).yz;
    return vec4(n,bg); // .w=1 => background
}

// Function 780
vec4 raymarchClouds( in vec3 ro, in vec3 rd, in vec3 bcol, float tmax, out float rays, ivec2 px )
{
	vec4 sum = vec4(0, 0, 0, 0);
	rays = 0.0;
    
	float sun = clamp( dot(rd,lig), 0.0, 1.0 );
	float t = 0.1*texelFetch( iChannel0, px&ivec2(255), 0 ).x;
	for(int i=0; i<64; i++)
	{
		if( sum.w>0.99 || t>tmax ) break;
		vec3 pos = ro + t*rd;
		vec4 col = mapClouds( pos );

		float dt = max(0.1,0.05*t);
		float h = (2.8-pos.y)/lig.y;
		float c = fbm( (pos + lig*h)*0.35 );
		//kk += 0.05*dt*(smoothstep( 0.38, 0.6, c ))*(1.0-col.a);
		rays += 0.02*(smoothstep( 0.38, 0.6, c ))*(1.0-col.a)*(1.0-smoothstep(2.75,2.8,pos.y));
	
		
		col.xyz *= vec3(0.4,0.52,0.6);
		
        col.xyz += vec3(1.0,0.7,0.4)*0.4*pow( sun, 6.0 )*(1.0-col.w);
		
		col.xyz = mix( col.xyz, bcol, 1.0-exp(-0.0018*t*t) );
		
		col.a *= 0.5;
		col.rgb *= col.a;

		sum = sum + col*(1.0 - sum.a);	

		t += dt;//max(0.1,0.05*t);
	}
    rays = clamp( rays, 0.0, 1.0 );

	return clamp( sum, 0.0, 1.0 );
}

// Function 781
vec4 raymarch( vec3 ro, vec3 rd, vec3 bgcol, ivec2 px )
{
	vec4 sum = vec4(0);
	float  t = 0., //.05*texelFetch( iChannel0, px&255, 0 ).x; // jitter ray start
          dt = 0.,
         den = 0., _den, lut, dv;
    for(int i=0; i<150; i++) 
    {
        vec3 pos = ro + t*rd;
        if( pos.y < -3. || pos.y > 3. || sum.a > .99 ) break;
        _den = den; den = map(pos);  // raw density
        if( abs(pos.x) > .5 )        // cut a slice 
        {
            dv = -.6+sin(2.*iTime);  // explore isovalues
            lut = LUTs( _den+dv, den+dv ); // shaped through transfer function
            if (lut>.01)             // not empty space
            { 
                vec3  col = mix(hue(.1*pos.z) , vec3(1), .8);
                col = mix( col , bgcol, 1.-exp(-.003*t*t) ); // fog
                sum += (1.-sum.a) * vec4(col,1)* (lut* dt*10.); // blend. Original was improperly just den*.4;
            }  }
        t += dt = max(.05,.02*t);     // stepping
    }

    return sqrt(1.-sum);              // black on white + sRGB
}

// Function 782
void march(vec3 origin, vec3 dir, out float t, out int objectHit) {
    t = 0.001;
    for (int i = 0; i < RAY_STEPS; i++) {
        vec3 pos = origin + t * dir;
        float min;
        sceneMap3D(pos, min, objectHit);
        if (min < 0.001) {
            return;
        }
        t += min;
    }
    t = -1.;
    objectHit = -1;
}

// Function 783
vec2 March(in Ray ray)
{
    float depth    = NearClip;
    float material = -1.0;
    
    for( ; depth < FarClip; )
    {
        vec3 pos = ray.o + (ray.d * depth);
        vec2 sdf = Scene(pos, 0.0);
        
        if(sdf.x < Epsilon)
        {
            material = sdf.y;
            break;
        }
        
        depth += sdf.x;
    }
    
    return vec2(depth, material);
}

// Function 784
float StepValue(float a, float b, float ra, float rb)
{
    return mix(ra, rb, step(a, b));
}

// Function 785
void TestLineMarch(in vec3 rayPos, inout SRayHitInfo info, in vec3 A, in vec3 B, in float width, in SMaterial material)
{   
    vec3 normal;
    float dist = LineDistance(A, B, width, rayPos, normal);
    if (dist < info.dist)
    {
        info.objectPass = OBJECTPASS_RAYMARCH;
        info.dist = dist;        
        info.normal = normal;
        info.material = material;
    }    
}

// Function 786
Hit march(Ray r)
{
    float t = 0., d, s;
    vec3 p;
    
    for(int i = 0; i < S; i++)
    {
        d = scene(p = r.o + r.d*t);

        if (d < P || t > D)
        {
            s = float(i);
            break;
        }

        t += d/max(R+1.,1.);
    }

    return Hit(p, t, d, s);
}

// Function 787
void stepTransform(inout vec3 p, inout float t) {
    p -= bloomPosition;
    p /= stepScale;
    globalScale *= stepScale;
    p *= bloomRotate;
    t -= delay;
}

// Function 788
vec2 raymarch(vec3 ro, vec3 rd, in float tmin, in float tmax) {
    vec2 m = vec2(-1.0, -1.0);
    vec2 res = vec2(tmin, -1.0);
    res.x = tmin;
	for( int i=0; i<NUM_STEPS; i++ )
	{
        m = mapRM(ro + res.x*rd);
		if( m.x<tmin || res.x>tmax ) break;
		res.x += 0.5*m.x*log(1.0+float(i));
        res.y = m.y;
	}
    if( res.x>tmax ) res.y=-1.0;
	return res;
}

// Function 789
vec4 raymarch () {
  vec2 uv = (gl_FragCoord.xy-.5*iResolution.xy)/iResolution.y;
  float dither = rng(uv+fract(time));
  vec3 eye = vec3(0,5,-4.5);
  vec3 ray = getCamera(eye, uv);
  vec3 pos = eye;
  float shade = 0.;
  for (float i = 0.; i <= 1.; i += 1./STEPS) {
    float dist = map(pos);
		if (dist < VOLUME) {
			shade = 1.-i;
			break;
		}
    dist *= .5 + .1 * dither;
    pos += ray * dist;
  }

  vec4 color = vec4(shade);
  color *= getLight(pos, eye);
  color = smoothstep(.0, .5, color);
  color = sqrt(color);
  return color;
}

// Function 790
float raymarch()
{
  float d = 0.0, t = 0.0;
  for(int i = 0; i < 32; ++i)
  {
    d = sphere(camera.origin + t * camera.direction);
    if(abs(d) < 0.01) return t;
    if(t > 100.0) return -1.0;
    t += d;
  }
  return -1.0;  // no intersection
}

// Function 791
vec4 rayMarch1d(inout vec4 p, in vec4 rd, in vec4 rd2, out vec4 dists) {
  vec4 dists2, p2;
  float dS = 99., dSx = 99., dSy, dSz, dSw, d = 0., minDS = dS, steps = 0.;
  for (int i = 0; i < MAX_STEPS; i++) {
    steps++;
    dS = mapWDists(p, dists);
    if (dS > SURF_DIST) {  // we are receeding,
      p2 = p - rd2 * 1.41;
      vec4 innerRmd = rayMarch(p2, rd2 * vec4(1,1,1,1), dists2);
      steps += innerRmd.w;
      if (innerRmd.y < SURF_DIST) { p = p2; }
      /* minWith 4d */
      // float ddS = dS - innerRmd.y;
      // dists = (max(0., ddS) * dists + max(0., -ddS) * dists2) * -1. / abs(ddS);
      /* -- end minWith */
      dists = minWith14(dS, dists, innerRmd.y, dists2);
      dS = min(dS, innerRmd.y);
      minDS = min(minDS, innerRmd.z);
    }
    minDS = min(minDS, (dS));
    mat4 chBasis = mat4(0, 0, 0, 0, 0, 0, 0, 0, rd.x, rd.y, rd.z, rd.w, rd2.x, rd2.y, rd2.z, rd2.w);
    vec4 components = normalize(dists * chBasis);
    d += dS;
    /* first, take the largest of the components we got back from the change
    of basis (it breaks for some angles with less than this - I'm not really
    sure why) */
    float dsFactor = maxOf(vec4(components.zw, -components.zw));
    #ifdef SLOW_THRU
    // then, slow down the ray whenever it has a coordinate near 0 for some axis.
    dsFactor *= min(1., max(minOf(abs(p)) + SPOOK_BE_GONE, .01));
    #endif
    p = p + rd * dS * dsFactor;
    // * sin(atan(dSw, dSz)) * (min(1., max(minOf(abs(p)) + SPOOK_BE_GONE, .01))); // move slower inside;
    if ((0. <= dS && dS < SURF_DIST) || d > MAX_DIST) break;
  }
  return vec4(d, dS, minDS, steps);
}

// Function 792
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

// Function 793
float lerpStep(float t, float a, float b)
{
    return floor(t/b) + mix(0.0, 1.0, clamp(mod(t, b)/a, 0.0, 1.0));
}

// Function 794
float linearstep(float edge0, float edge1, float x) {
    float t = (x - edge0)/(edge1 - edge0);
    return clamp(t, 0.0, 1.0);
}

// Function 795
void RayMarchScene(in vec3 startingRayPos, in vec3 rayDir, inout SRayHitInfo oldHitInfo)
{
    SMaterial dummyMaterial = SMaterial(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), 0.0f, vec3(0.0f, 0.0f, 0.0f));
    
    float rayDistance = c_minimumRayHitTime;
    float lastRayDistance = c_minimumRayHitTime;
    
    float lastHitInfoDist = 0.0f;
    
    SRayHitInfo newHitInfo = oldHitInfo;
    newHitInfo.hitAnObject = false;
    
    for (int stepIndex = 0; stepIndex < c_numSteps; ++stepIndex)
    {
        vec3 rayPos = startingRayPos + rayDistance * rayDir;
        
        newHitInfo = TestSceneMarch(rayPos);
        
        // these two lines are so that the material code goes away when the test functions are inlines
        newHitInfo.normal = vec3(0.0f, 0.0f, 0.0f);
        newHitInfo.material = dummyMaterial;
        
        newHitInfo.hitAnObject = newHitInfo.dist < 0.0f;
        if (newHitInfo.hitAnObject)
            break;
        
        lastRayDistance = rayDistance;
        rayDistance += max(newHitInfo.dist, c_minStepDistance);

        lastHitInfoDist = newHitInfo.dist;
        
        if (rayDistance > oldHitInfo.dist)
            break;
    }
    
    if (newHitInfo.hitAnObject)
    {
		float refinedHitPercent = lastHitInfoDist / (lastHitInfoDist - newHitInfo.dist);
        newHitInfo.dist = mix(lastRayDistance, rayDistance, refinedHitPercent);
        
        if (newHitInfo.dist < oldHitInfo.dist)
            oldHitInfo = newHitInfo;
    }
}

// Function 796
float raymarch_withphotons( vec3 o, vec3 target, float start_time, out vec4 objPos )
{
    return raymarch( o, target, start_time, 1., objPos );
}

// Function 797
float March(vec3 eye, vec3 dir, float start, float end) {
    
	float depth = start;
    
    int i = 0;
    do {

        // check against scene
     	float dist = SceneSDF(eye + depth * dir);
        
        // smol distance, we found a hit
        if(dist < EPSILON) return depth;
        
        // incr
        depth += dist;
        
        // too big, hit the void
        if(depth >= end) return end;

    }
    while(i++ < MAX_STEPS);

    return end;
}

// Function 798
vec4 MarchBlackHole(vec3 u,vec3 d){
//;vec2 m=(iMouse.xy/iResolution.xy);
;float h=u.x;//camera height modifies zFar
;float l=0.;//[matte used], every reflection increases [matte used]
;vec3 sun=normalize(vec3(sin(iTime),.4,cos(0.43*iTime)))
;vec4 o=vec4(0,0,0,1)
;float dLast=0.;//last distance to an object
;float t=0.; //distance to camera
//;vec3 hPos=vec3(5.*sin(.6*iTime),116,15.+5.*iTime+4.*cos(.2*iTime)) //position of black hole
;vec3 hPos=vec3(5.*sin(.6*iTime),5  ,15.+5.*iTime+4.*cos(.2*iTime)) //position of black hole
;const vec4 cPlane =vec4(.1,.7,.0,1);
;const vec4 cSphere=vec4(.9,.2,.1,1);
;float force,mdist;
;for(float i=0.;i<iterRm;++i
){
 ;float distS=sphere1(u);
 ;float distP=distFromPlane(u)
 ;vec3 hPosd=(u-hPos)
 ;float mdist2=dot(hPosd,hPosd);mdist=sqrt(mdist2)
 //;force=22.*(.02+.5*((cos(0.1*iTime)+1.)))/mdist2 //mass of black hole
 //;force=22.*(.02+.5*((cos(0.1*iTime)+1.)))/mdist2 //mass of black hole
 ;force=.3/mdist2
 #ifdef blackHoleDimReflections
 //;o*=smoothstep(-force,0.,.1*mdist*force);//black hole has a darker reflection 
 #endif
 #ifdef blackHoleIsInvisibleInside
 ;if(mdist<force) return o;//if raymarched inside event horizon, break.
 #endif
 //distances to 2 shapes
 ;if(eMode(t,distS)
 ){accCol(.5,o,u,d,l,sun,cSphere,normalize(vec3(0,0,8)-repeat(u)));//explicit normal of sphere1()
 ;if(l>.99)break;}//out if reflectiveness
 ;if(eMode(t,distP)
 ){accCol(.8,o,u,d,l,sun,cPlane,vec3(0,1,0));
 ;if(l>.99)break;}//out fof reflectiveness
 ;dLast=min(min(distS,distP),mdist)
 ;t+=dLast
 //artificial blue horizon IS a BV that can look like a globe while the black hole is stronger.
 ;if((mdist)>zFar)break//enforce horizon for a blue sky line
  #ifndef doBlackHole
 // ;d=normalize(d-dLast*force)//distortion by black hole
 ;d=normalize(d-dLast*force*(u-hPos)*sqrt(mdist*.001))
 ;u+=d*dLast*sqrt(mdist*.001) //u moves along the ray
      
 //;d=normalize(d-dLast)//black hole bends ray (and shoortens ray)
  #endif
// ;u+=d*dLast//u moves along the ray
;}
;u+=d*force
;o.xyz=(o.xyz*l+vec3(.2,.5,.9)*(1.-l));//remaining [l] becomes blue sky
;float angle = dot(sun,d)
//below reflects a modulo light on the floor that does not exist
//still okay.
//;o += pow((abs(angle)),180.)*.8;
;return o
;}

// Function 799
float RayMarch(vec3 ro, vec3 rd) {
	float dO = 0.;
    
    for(int i = 0; i < MAX_STEPS; i++) {
    	vec3 p = ro + rd * dO;
        float dS = GetDist(p);
        dO += abs(dS);
        if(dO > MAX_DIST || abs(dS) < SURF_DIST) break;
    }
    
    return dO;
}

// Function 800
float MarchWarp(vec3 origin,vec3 dir)
{
    float dist = 0.0;
    
    for(int i = 0;i < MAX_STEPS;i++)
    {
        float sceneDist = Warp(origin + dir * dist);
        
        dist += sceneDist * STEP_MULT;
        
        if(abs(sceneDist) < MIN_DIST || sceneDist > MAX_DIST)
        {
            break;
        }
    } 
    return dist;
}

// Function 801
vec4 rayMarch1d(inout vec3 p, in vec3 rd, in vec3 rd2, out vec3 dists) {
  float dS = 99., dSx = 99., dSy, dSz, d = 0., minDS = dS, steps = 0.;
  vec3 dists2;
  for (int i = 0; i < MAX_STEPS; i++) {
    steps += 1.;
    dS = 99.;
    dS = mapWDists(p, dists);
    minDS = min(minDS, abs(dS));
    if (dS > SURF_DIST) {
      vec3 p2 = p - rd2 * 2.;
      vec4 innerRmd = rayMarch(p2, rd2, dists2);
      // steps += innerRmd.w;
      if (innerRmd.x < MAX_DIST && innerRmd.y < SURF_DIST) {
        p = p2;
        return vec4(length(vec2(innerRmd.x, d)), innerRmd.yz, steps);
      } else {
        dS = min(dS, min(innerRmd.y, 99.));
        minDS = min(minDS, abs(innerRmd.z));
      }
    }
    // backported chBasis + slow_thru from 4d version
    mat3 chBasis = mat3(rd.x, rd.y, rd.z, 0, 0, 0, rd2.x, rd2.y, rd2.z);
    vec3 components = normalize(dists * chBasis);
    d += dS;
    float dsFactor = maxOf(vec4(components.xz, -components.xz));
    dsFactor *= min(1., max(minOf(abs(p)) + .08, .01));
    p = p + rd * dS * dsFactor;
    if ((0. <= dS && dS < SURF_DIST) || d > MAX_DIST) break;
  }
  return vec4(d, dS, minDS, steps);
}

// Function 802
vec4 raymarch( in vec3 ro, in vec3 rd )
{
    vec4 sum = vec4(0, 0, 0, 0);
    
    // setup sampling - compute intersection of ray with 2 sets of planes
    float2 t, dt, wt;
	SetupSampling( t, dt, wt, ro, rd );
    
    // fade samples at far extent
    float f = .6; // magic number - TODO justify this
    float endFade = f*float(SAMPLE_COUNT)*PERIOD;
    float startFade = .8*endFade;
    
    for(int i=0; i<SAMPLE_COUNT; i++)
    {
        if( sum.a > 0.99 ) continue;

        // data for next sample
        vec4 data = t.x < t.y ? vec4( t.x, wt.x, dt.x, 0. ) : vec4( t.y, wt.y, 0., dt.y );
        // somewhat similar to: https://www.shadertoy.com/view/4dX3zl
        //vec4 data = mix( vec4( t.x, wt.x, dt.x, 0. ), vec4( t.y, wt.y, 0., dt.y ), float(t.x > t.y) );
        vec3 pos = ro + data.x * rd;
        float w = data.y;
        t += data.zw;
        
        // fade samples at far extent
        w *= smoothstep( endFade, startFade, data.x );
        
        vec4 col = map( pos );
        
        // iqs goodness
        float dif = clamp((col.w - map(pos+0.6*sundir).w)/0.6, 0.0, 1.0 );
        vec3 lin = vec3(0.51, 0.53, 0.63)*1.35 + 0.55*vec3(0.85, 0.57, 0.3)*dif;
        col.xyz *= lin;
        
        col.xyz *= col.xyz;
        
        col.a *= 0.75;
        col.rgb *= col.a;

        // integrate. doesn't account for dt yet, wip.
        sum += col * (1.0 - sum.a) * w;
    }

    sum.xyz /= (0.001+sum.w);

    return clamp( sum, 0.0, 1.0 );
}

// Function 803
float march(vec3 ro, vec3 rd){
    float t = 0.;
    for(int i=0; i<200; i++){
        vec3 p = ro + rd*t;
        float d = dist(p);
        if(d<.001 || t > MAX_D)break;
        t += d;
    }
    return t;
}

// Function 804
Marched March(vec3 ro, vec3 rd, float thresh, float dmax, int iters)
{
	Marched c = Marched(dmax, mSky, dmax);
	int i = iters;
	float t = 0.;
	while (i-- > 0) {
		vec3 mp = ro + rd * t;
		Hit h = Scene(mp);
		float d = h.d, ad = abs(d);
		t += d;
		c.m = h.m, c.nmd = h.d;
		if (rd.y >= 0. && (ad > dmax
			|| mp.y > hmax))
			break; //t = dmax;
		if (ad < thresh * t || t >= dmax)
			break;
	}
	c.t = t = clamp(t, 0., dmax);
	if (abs(c.nmd) > thresh * 2. * t) 
		c.m = mSky;
	if (c.m == mSky)
		c.t = dmax; // caller won't be able to tell how far it got though
	return c;
}

// Function 805
HitResult march(vec3 o, vec3 r) {
    HitResult result;
    
    result.col = vec3(1.,1.,1.);
    result.dist = 0.;
    
    vec3 p = o;
    
    for(int i = 0; i < ITER; i++) {
        float d = dist(p)*.7;
        p += r*d;
        result.dist += d;
    }
    vec3 normal = calcNormal(p);
    float light = max(0., dot(normal, normalize(vec3(1.,10., 1.))));
    result.col = mix(vec3(.9, 0.86, 0.87), vec3(0.1,0.5,0.1), (-p.y)*2.);
    if(p.y < -0.58) result.col = vec3(0.,0.,1.);
    result.col *= light;
    
    return result;
}

// Function 806
float animStep(float t, float stepIndex) {
    float x = t;
    x *= MODEL_STEPS;
    x -= stepIndex;
    x *= transitionPoint;
    x = tweakAnim(x);
    return x;
}

// Function 807
vec3 raymarch( in vec3 ro, in vec3 rd, in vec2 pixel )
{
	vec4 sum = vec4( 0.0 );

	float t = 0.0;

    // dithering	
	t += 0.05*textureLod( iChannel0, pixel.xy/iChannelResolution[0].x, 0.0 ).x;
	
	for( int i=0; i<100; i++ )
	{
		if( sum.a > 0.99 ) break;
		
		vec3 pos = ro + t*rd;
		vec4 col = map( pos );
		
		col.xyz *= mix( 3.1*vec3(1.0,0.5,0.05), vec3(0.48,0.53,0.5), clamp( (pos.y-0.2)/2.0, 0.0, 1.0 ) );
		
		col.a *= 0.6;
		col.rgb *= col.a;

		sum = sum + col*(1.0 - sum.a);	

		t += 0.05;
	}

	return clamp( sum.xyz, 0.0, 1.0 );
}

// Function 808
void march( inout vec3 p, vec3 d )
{
	float r = dist(p+d*EPSILON);
	for(int i = 0; i < V_STEPS; i++)
	{
		if(r < EPSILON || r > MAX_DEPTH)
			return;
		p += d*r*.45;
        r = dist(p);
	}
	return;
}

// Function 809
vec4 RayMarch(in Ray ray)
{
    float depth   = NearClip;
    float nearest = FarClip;          // Keeps track of nearest pass for SDF Near-Miss outline
    float edge    = 0.0;              // Keeps track of if edge for SDF Edge outline
    float lastSDF = FarClip;          // Keeps track of last SDF value for SDF Edge outline
    
    for(int i = 0; i < MaxSteps; ++i)
    {
    	vec3 pos = ray.o + (ray.d * depth);
        vec2 sdf = Scene(pos);
        
        nearest = min(sdf.x, nearest);
        
        if((lastSDF < EdgeThresold) && (sdf.x > lastSDF))
        {
            edge = 1.0;
        }
        
        if(sdf.x < Epsilon)
        {
            return vec4(clamp(depth, NearClip, FarClip), nearest, edge, sdf.y);
        }
        
        depth  += sdf.x * 0.35;        // Note: Modifying the '* 0.35' affects the outlines in subtle ways.
        lastSDF = sdf.x;
    }
    
    return vec4(FarClip, nearest, edge, 0.0);
}

// Function 810
vec3 raymarch( in vec3 ro, in vec3 rd, in vec2 ani, in vec2 pixel )
{
    // background color	
	vec3 bgc = vec3(0.6,0.7,0.7) + 0.3*rd.y;
    bgc *= 0.2;
	

    // dithering	
	float t = 0.03*texture( iChannel0, pixel.xy/iChannelResolution[0].x ).x;

    // raymarch	
	vec4 sum = vec4( 0.0 );
	for( int i=0; i<150; i++ )
	{
		if( sum.a > 0.99 ) continue;
		
		vec3 pos = ro + t*rd;
		vec4 col = map( pos, ani );

        // lighting		
		float dif = 0.1 + 0.4*(col.w - map( pos + lig*0.15, ani ).w);
		col.xyz += dif;

        // fog		
		col.xyz = mix( col.xyz, bgc, 1.0-exp(-0.005*t*t) );
		
		col.rgb *= col.a;
		sum = sum + col*(1.0 - sum.a);	

        // advance ray with LOD
		t += 0.03+t*0.012;
	}

    // blend with background	
	sum.xyz = mix( bgc, sum.xyz/(sum.w+0.0001), sum.w );
	
	return clamp( sum.xyz, 0.0, 1.0 );
}

// Function 811
vec3 RayMarch(vec3 ro, vec3 rd) {
	float dO=MIN_DIST;
    float dS;
    float matId=0.;
    
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        vec2 g = GetDist(p);
        dS = g.x;
        dO += dS;
        matId = g.y;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }
    
    return vec3(dO, abs(dS), matId);
}

// Function 812
Hit raymarch(vec3 eye, vec3 ray){
	float dsum = 0.0;
	for(int i=0; i<iterations; i++){
		vec3 p = eye + dsum*ray;
		float dmin = dist(p);
		if(dmin < threshold){
			return Hit(p, grad(p), dsum);
		} else {
			dsum += max(min_step, dmin*step_fraction);
		}
	}
	
	vec3 p = eye + dsum*ray;
	return Hit(p, grad(p), dsum);
}

// Function 813
vec4 RayMarch(vec3 origin, vec3 direction) {
	
    float distance = 0.;
    float closest = FLOAT_MAX;
    vec2 closestPoint = vec2(0.);
    
    for(int i = 0; i < MAX_STEPS; i++) {
        
    	vec3 p = origin + direction * distance;        
        float sphereDistance = GetSphereDist(p);
        
        distance += sphereDistance;
        
        // If the calculated distance to the closest sphere
        // is smaller than what it was, update it and
        // update the hit point as well.
        if (sphereDistance < closest ) {
            closest = sphereDistance;
            closestPoint = p.xy;
        }
        
        if (distance > MAX_DIST) {
            // No hit
            return vec4(-1, closest, closestPoint);
        }
        
        if (sphereDistance < SURF_DIST) {
            // Sphere hit
            return vec4(distance, 0, closestPoint);
        }
    }
    
    // No hit
    return vec4(-1, closest, closestPoint);
}

// Function 814
vec3 raymarch(vec3 rayorig, vec3 raydir) {
  vec3 pos = rayorig;
  float d = getSdfWithPlane(pos);
  int work = 0;

  for (int step = 0; step < renderDepth; step++) {
    work++;
    pos = pos + raydir * d;
    d = getSdfWithPlane(pos);
    if (abs(d) < 0.001) {
      break;
    }
  }

  return showRenderDepth
    ? vec3(float(work) / float(renderDepth))
    : (abs(d) < 0.001) 
      ? illuminate(pos)
      : background;
}

// Function 815
float march(in vec3 ro, in vec3 rd)
{
    float t=0.,stp=0.0,os=0.0,pd=10.0, d =0.;
	for(int i=0;i<ITR;i++)
    {
        t+=stp;
        d=map(ro+rd*t);
        if (t>FAR || abs(d) <0.0005)break;
        if(d>=os)
        {		
            os=.9*d*d/pd;
            stp=d+os;
            pd=d;
        }
        else
        {
            stp=-os;
            pd=1.;
            os=.001;
        }
    }
    return t;
}

// Function 816
void march(inout vec3 pos, vec3 dir)
{
    float eps=.001;
    for(int i=0;i<80;i++)
    {
        float d=triDist(pos);
        pos+=dir*d*.9;
        if (d<eps) break;
    }
}

// Function 817
vec4 march(inout vec3 pos, vec3 dir)
{
    // cull the sphere
    if(length(pos-dir*dot(dir,pos))>1.3) 
    	return vec4(0,0,1,1);
    
    float eps=0.001;
    float bg=1.0;
    for(int cnt=0;cnt<100;cnt++)
    {
        float d = dist(pos);
        pos+=d*dir*.7;
        if(d<eps) { bg=0.0; break; }
    }
    vec3 n = getGrad(pos,.001);
    if(dot(n,n)<.0001) n=vec3(0,0,1);
    return vec4(n,bg); // .w=1 => background
}

// Function 818
float rxStep(float a, float b, float x, float k)
{
    return (a == b) ? step(x, a) : rxEase((x - a) / (b - a), k);
}

// Function 819
float ray_march(vec3 ro, vec3 rd) {
  float t = 0.0;
  for(int i = 0; i < 128; i++) {
    vec3 p = (ro + (t * rd));
    float d = map(p);
    if(d <= 0.0001) {
      return t;
    }
    t = (t + d);
    if(t > 10.0) {
      return -1.0;
    }
  }
  return -1.0;
}

// Function 820
vec2 marchRay(vec3 ro, vec3 rd) 
{
    float mint = 0.01;
    float t = 0.0;
    while(t < MAX_VIEW_DISTANCE+10.0)
    {	
        vec3 p = ro + rd*t;
        if(p.y < 0.0 && p.y > 2.0) break;
        vec2 h = sdScene(p);
        if( h.x < 0.01)
        {
            return vec2(t, h.y);
        }
        t+=h.x;
    }
    return vec2(MAX_VIEW_DISTANCE, -1.0); // No hit
}

// Function 821
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

// Function 822
void Step (int mId, out vec4 p, out vec4 qt)
{
  vec2 r, rn, dr, f, v;
  float fOvlap, fric, rSep, vm;
  fOvlap = 1000.;
  fric = 0.015;
  p = Loadv4 (2 * mId);
  r = p.xy;
  v = p.zw;
  qt = Loadv4 (2 * mId + 1);
  if (r.x < 2. * hbLen) {
    f = vec2 (0.);
    for (int n = 0; n < nBall; n ++) {
      rn = Loadv4 (2 * n).xy;
      if (rn.x < 2. * hbLen) {
        dr = r - rn;
        rSep = length (dr);
        if (n != mId && rSep < 1.) f += fOvlap * (1. / rSep - 1.) * dr;
      }
    }
    dr = hbLen * vec2 (1., 1.75) - abs (r);
    f -= step (dr, vec2 (1.)) * fOvlap * sign (r) * (1. / abs (dr) - 1.) * dr;
    f -= fric * v;
    if (runState == 2.) {
      v += dt * f;
      r += dt * v;
    }
    if (length (abs (r) - hbLen * vec2 (1., 1.75) + 0.6) < 0.9 ||
       length (abs (r) - hbLen * vec2 (1., 0.) + 0.6) < 0.9) r.x = 100. * hbLen;
    if (runState == 2.) {
      vm = length (v);
      if (vm > 1e-6) qt = RMatToQ (QToRMat (qt) *
         VToRMat (normalize (vec3 (v.y, 0., - v.x)), vm * dt / 0.5));
    }
  }
  p = vec4 (r, v);
}

// Function 823
vec3 march (vec3 p, vec3 d) {
    float steelSDFp;
    float iceSDFp;
    float SDFp;
    vec3 dsteelSDFp;
    vec3 diceSDFp;
    vec3 TEXp;
    vec3 finalcol = vec3(0.);
    float shiny = 1.;
    float rxcount = 0.;
    for (int i=0; i<100; ++i) {
        steelSDFp = steelSDF(p);
        iceSDFp = abs(iceSDF(p));
        SDFp = min(steelSDFp, iceSDFp);
        if (SDFp < 1e-2) {
            p = p+d*SDFp*.95;
            if (steelSDFp < iceSDFp) {
            	dsteelSDFp = dsteelSDF(p);
                dsteelSDFp = vec3(0., 1., 0.);
                TEXp = steelTEX(p, d, dsteelSDFp);
                finalcol = finalcol+clamp(TEXp*.2+steelSPEX(p, d, dsteelSDFp), 0., 1.)*shiny;
                shiny *= .8;
                d = reflect(d, dsteelSDFp);
                p = p+d*.2;
                ++rxcount;
            }
            else {
            	diceSDFp = diceSDF(p);
                TEXp = iceTEX(p, d, diceSDFp);
                finalcol = finalcol+clamp(TEXp*.4+iceSPEX(p, d, diceSDFp), 0., 1.)*shiny;
                shiny *= .6;
                // d = reflect(d, diceSDFp);
                p = p+d*.2;
                ++rxcount;
            }
            if (rxcount > 2.) {
                break;
            }
            if (p.y < -10. || p.y > 20.) {
                break;
            }
        }
        p = p+d*SDFp*.7;
        if (length(p) > 10.) {
            p = p+d*SDFp*.2;
        }
    }
    // add the sky color
    finalcol = finalcol+shiny*vec3(0., 0., 0.);
    return finalcol;
}

// Function 824
MarchResult march(in vec3 ro, in vec3 rd, inout float t){

    MarchResult m;
    m.p = ro+rd;
    for(int i = 0; i < 40; ++i){
        vec2 d = sdf(m.p);
        t += d.x;
        m.p += rd*d.x;
        m.id = d.y;
        
        if(d.x < 0.01 || t > 100.){
            break;
        }
        
    }
    
    return m;
}

// Function 825
vec2 raymarchTerrain( in vec3 ro, in vec3 rd, float tmin, float tmax )
{
    //float tt = (150.0-ro.y)/rd.y; if( tt>0.0 ) tmax = min( tmax, tt );
    
    float dis, th;
    float t2 = -1.0;
    float t = tmin; 
    float ot = t;
    float odis = 0.0;
    float odis2 = 0.0;
    for( int i=ZERO; i<400; i++ )
    {
        th = 0.001*t;

        vec3  pos = ro + t*rd;
        vec2  env = terrainMap( pos.xz );
        float hei = env.x;

        // tree envelope
        float dis2 = pos.y - (hei+kMaxTreeHeight*1.1);
        if( dis2<th ) 
        {
            if( t2<0.0 )
            {
                t2 = ot + (th-odis2)*(t-ot)/(dis2-odis2); // linear interpolation for better accuracy
            }
        }
        odis2 = dis2;
        
        // terrain
        dis = pos.y - hei;
        if( dis<th ) break;
        
        ot = t;
        odis = dis;
        t += dis*0.8*(1.0-0.75*env.y); // slow down in step areas
        if( t>tmax ) break;
    }

    if( t>tmax ) t = -1.0;
    else t = ot + (th-odis)*(t-ot)/(dis-odis); // linear interpolation for better accuracy
    return vec2(t,t2);
}

// Function 826
vec4 march(vec3 from, vec3 dir, vec3 camdir) {
    dir += hash33(dir)*sin(iTime/2.0)/25.0;
    // variable declarations
	vec3 p=from, col=vec3(0.1), backcol=col;
    float totdist=0., d=0.,sdet, glow=0., lhit=1.;
	// the detail value is smaller towards the end as we are closer to the fractal boundary
   	det*=1.-fin*.7;
    // raymarching loop to obtain an occlusion value of the sun at the camera direction
    // used for the lens flare
    for (int i=0; i<70; i++) {
    	p+=d*ldir; // advance ray from camera pos to light dir
        d=de(p)*2.; // distance estimation, doubled to gain performance as we don't need too much accuracy for this
        lhit=min(lhit,d); // occlusion value based on how close the ray pass from the surfaces and very small if it hits 
        if (d<det) { // ray hits the surface, bye
            break;
        }
    }
    // main raymarching loop
    for (int i=0; i<150; i++) {
    	p=from+totdist*dir; // advance ray
        d=de(p); // distance estimation to fractal surface
        sdet=det*(1.+totdist*.1); // makes the detail level lower for far hits 
        if (d<sdet||totdist>maxdist) break; // ray hits the surface or it reached the max distance defined
    	totdist+=d; // distance accumulator  
        glow++; // step counting used for glow
    }
    float sun=max(0.,dot(dir,ldir)); // the dot product of the cam direction and the light direction using for drawing the sun
    if (d<.2) { // ray most likely hit a surface
    	p-=(sdet-d)*dir; // backstep to correct the ray position
        vec3 c=fcol; // saves the color set by the de function to not get altered by the normal calculation
        vec3 n=normal(p); // calculates the normal at the ray hit point
        col=shade(p,dir,n,c)+sin(p)*sin(iTime/2.0+hash33(p))/5.0; // sets the color and lighting
    } else { // ray missed any surface, this is the background
        totdist=maxdist; 
    	p=from+dir*maxdist; // moves the ray to the max distance defined
        // Kaliset fractal for stars and cosmic dust near the sun. 
        vec3 st = (dir * 3.+ vec3(1.3,2.5,1.25)) * .3;
        for (int i = 0; i < 10; i++) st = abs(st) / dot(st,st) - .8;
        backcol+=length(st)*.015*(1.-pow(sun,3.))*(.5+abs(st.grb)*.5);
        sun-=length(st)*.0017;
        sun=max(0.,sun);
		backcol+=pow(sun,100.)*.5; // adds sun light to the background
    }
    backcol+=pow(sun,20.)*suncol*.8; // sun light
    float normdist=totdist/maxdist; // distance of the ray normalized from 0 to 1
    col=mix(col,backcol,pow(normdist,1.5)); // mix the surface with the background in the far distance (fog)
    col=max(col,col*vec3(sqrt(glow))*.13); // adds a little bit of glow
	// lens flare
    vec2 pflare=dir.xy-ldir.xy;
    float flare=max(0.,1.0-length(pflare))-pow(abs(1.-mod(camdir.x-atan(pflare.y,pflare.x)*5./3.14,2.)),.6);
	float cflare=pow(max(0.,dot(camdir,ldir)),20.)*lhit;
    col+=pow(max(0.,flare),3.)*cflare*suncol;
	col+=pow(sun,30.)*cflare;
    // "only glow" part (at sec. 10)
    col.rgb=mix(col.rgb,glow*suncol*.01+backcol,1.-smoothstep(0.,.8,abs(time-10.5)));
    return vec4(col,normdist); // returns the resulting color and a normalized depth in alpha
}

// Function 827
void rayMarch(vec3 ro, vec3 rd, out float t, out float d, in float maxd)
{
    t = 0.;
    d = 0.;
    vec3 cp = ro;
    for(int i=0;i<200;++i)
    {
        d = map(cp);
        t += d;
        cp = ro+rd*t;
        if (d < .001 || d > maxd || abs(cp.y) > 35.)
            break;
    }
}

// Function 828
float raymarch(vec3 pos, vec3 dir) {
	float dist = 0.0;
	float dscene;

	for (int i = 0; i < RM_ITERS; i++) {
		dscene = scene(pos + dist * dir);
		if (abs(dscene) < 0.1)
			break;
		dist += RM_FACTOR * dscene;
	}

	return dist;
}

// Function 829
HitData rayMarch(vec4 point, vec4 dir){
    HitData hd;
	float marched = 0.;
    float epsilon = 0.1;
    float lastDistance = 0.;
    while(marched < 10000.){
    	float distance = getDistance(point);
        marched += distance;
        point += dir*distance;
        if(distance < epsilon){
            return HitData(marched, 1.-distance/lastDistance);
        }
        lastDistance = distance;
        
    }return HitData(1000.,1.);
}

// Function 830
Hit raymarch(Ray ray) {
 
    vec3 p = ray.ori;
    int id = -1;
    
    for(int i = 0; i < MAX_ITERATIONS; i++) {
     
        Dst dst = dstScene(p);
        p += ray.dir * dst.dst * .75;
        
        if(dst.dst <= MIN_DISTANCE) {
         
            id = dst.id;
            break;
            
        }
        
    }
    
    return Hit(p,id);
    
}

// Function 831
float rxStep2(float a, float b, float x, float k)
{
    return (a == b) ? step(x, a) : rxEase2((x - a) / (b - a), k);
}

// Function 832
float Raymarch(in vec3 from, in vec3 to, in mat3 matrix)
{
    float dist  = MIN_DIST;
    float depth = 0.0;

    for(int i = 0; i < NUM_STEPS; ++i)
    {
        if(dist < MIN_DIST || depth > MAX_DIST) 
            break;

        dist  = Pentagram((from + to * depth) * matrix);
        depth += dist;

        // past maximum raymarch distance - no surface hit
        if(depth > MAX_DIST)
            return 0.0;
    }

    return depth;
}

// Function 833
void Step (ivec2 iv, out vec3 r, out vec3 v)
{
  vec3 f;
  float fDamp, grav, dt;
  IdNebs ();
  fOvlap = 1000.;
  fDamp = 0.5;
  grav = 2.;
  r = GetR (vec2 (iv));
  v = GetV (vec2 (iv));
  f = vec3 (0.);
  PairForce (iv, r, f);
  SpringForce (iv, r, v, f);
  BendForce (iv, r, f);
  WallForce (r, f);
  ObsForce (r, f);
  f -= vec3 (0., grav, 0.) * QtToRMat (qtVu) + fDamp * v;
  dt = 0.02;
  v += dt * f;
  r += dt * v;
}

// Function 834
void Step (int mId, out vec3 rm, out vec3 vm, out vec4 qm, out vec3 wm,
   out float sz)
{
  vec4 p;
  vec3 rmN, vmN, wmN, dr, dv, drw, am, wam, vn;
  float fOvlap, fricN, fricT, fricS, fricSW, fDamp, fAttr, grav, rSep, szN, szAv,
     fc, ft, ms, drv, dt;
  const vec2 e = vec2 (0.1, 0.);
  fOvlap = 1000.;
  fricN = 10.;
  fricS = 1.;
  fricSW = 10.;
  fricT = 0.5;
  fAttr = 0.2;
  fDamp = 0.05;
  grav = 10.;
  p = Loadv4 (4 * mId);
  rm = p.xyz;
  sz = p.w;
  vm = Loadv4 (4 * mId + 1).xyz;
  qm = Loadv4 (4 * mId + 2);
  wm = Loadv4 (4 * mId + 3).xyz;
  ms = sz * sz * sz;
  am = vec3 (0.);
  wam = vec3 (0.);
  for (int n = 0; n < nBall; n ++) {
    p = Loadv4 (4 * n);
    rmN = p.xyz;
    szN = p.w;
    dr = rm - rmN;
    rSep = length (dr);
    szAv = 0.5 * (sz + szN);
    if (n != mId && rSep < szAv) {
      fc = fOvlap * (szAv / rSep - 1.);
      vmN = Loadv4 (4 * n + 1).xyz;
      wmN = Loadv4 (4 * n + 3).xyz;
      dv = vm - vmN;
      drv = dot (dr, dv) / (rSep * rSep);
      fc = max (fc - fricN * drv, 0.);
      am += fc * dr;
      dv -= drv * dr + cross ((sz * wm + szN * wmN) / (sz + szN), dr);
      ft = min (fricT, fricS * abs (fc) * rSep / max (0.001, length (dv)));
      am -= ft * dv;
      wam += (ft / rSep) * cross (dr, dv);
    }
  }
  vn = normalize (vec3 (GrndHt (rm.xz + e.xy) - GrndHt (rm.xz - e.xy), 2. * e.x,
     GrndHt (rm.xz + e.yx) - GrndHt (rm.xz - e.yx)));
  dr.xz = -0.5 * sz * vn.xz;
  dr.y = rm.y + 0.55 * sz - GrndHt (rm.xz - dr.xz);
  rSep = length (dr);
  if (rSep < sz) {
    fc = fOvlap * (sz / rSep - 1.);
    dv = vm;
    drv = dot (dr, dv) / (rSep * rSep);
    fc = max (fc - fricN * drv, 0.);
    am += fc * dr;
    dv -= drv * dr + cross (wm, dr);
    ft = min (fricT, fricSW * abs (fc) * rSep / max (0.001, length (dv)));
    am -= ft * dv;
    wam += (ft / rSep) * cross (dr, dv);
  }
  am += fAttr * (rLead - rm);
  am.y -= grav * ms;
  am -= fDamp * vm;
  dt = 0.01;
  vm += dt * am / ms;
  rm += dt * vm;
  wm += dt * wam / (0.1 * ms * sz);
  qm = normalize (QtMul (RMatToQt (LpStepMat (0.5 * dt * wm)), qm));
}

// Function 835
vec3 march(in vec3 origin, in vec3 dir, in float maxlen) {
	float dist = 0.0f;
    vec3 pos = origin;
    vec3 d = dir;
    
    while (dist < maxlen) {
    	float t = samp(pos);
        if (t < 0.001f) {
        	float fx = samp(vec3(pos.x + 0.0001f, pos.y, pos.z)) - samp(vec3(pos.x - 0.0001f, pos.y, pos.z));
			float fy = samp(vec3(pos.x, pos.y + 0.0001f, pos.z)) - samp(vec3(pos.x, pos.y - 0.0001f, pos.z));
			float fz = samp(vec3(pos.x, pos.y, pos.z + 0.0001f)) - samp(vec3(pos.x, pos.y, pos.z - 0.0001f));
			vec3 normal = normalize(vec3(fx, fy, fz));
            if (dot(-d, normal) < 0.0f) normal = -normal;
            return vec3(max(normal.y, 0.4f));
        }
        
        d = normalize(d + vec3(0, -0.0025f, 0));
        
        dist += 0.01f;
        pos += 0.01f * d;
    }
    
    return vec3(0.0);
}

// Function 836
vec4 cloudMarch(vec3 p, vec3 ray)
{
    float density = 0.;

    float stepLength = VOLUME_LENGTH / float(MAX_STEPS);
    float shadowStepLength = SHADOW_LENGTH / float(SHADOW_STEPS);
    vec3 light = normalize(vec3(1.0, 2.0, 1.0));

    vec4 sum = vec4(0., 0., 0., 1.);
    
    vec3 pos = p + ray * jitter * stepLength;
    
    for (int i = 0; i < MAX_STEPS; i++)
    {
        if (sum.a < 0.1) {
        	break;
        }
        float d = map(pos);
    
        if( d > 0.001)
        {
            vec3 lpos = pos + light * jitter * shadowStepLength;
            float shadow = 0.;
    
            for (int s = 0; s < SHADOW_STEPS; s++)
            {
                lpos += light * shadowStepLength;
                float lsample = map(lpos);
                shadow += lsample;
            }
    
            density = clamp((d / float(MAX_STEPS)) * 20.0, 0.0, 1.0);
            float s = exp((-shadow / float(SHADOW_STEPS)) * 3.);
            sum.rgb += vec3(s * density) * vec3(1.1, 0.9, .5) * sum.a;
            sum.a *= 1.-density;

            sum.rgb += exp(-map(pos + vec3(0,0.25,0.0)) * .2) * density * vec3(0.15, 0.45, 1.1) * sum.a;
        }
        pos += ray * stepLength;
    }

    return sum;
}

// Function 837
float march( in vec3 ro, in vec3 rd )
{
	const float maxd = 10.0;
	const float precis = 0.001;
    float h = precis * 2.0;
    float t = 0.0;
	float res = -1.0;
    for( int i = 0; i < 64; i++ )
    {
        if( h < precis || t > maxd ) break;
	    h = map( ro + rd * t );
        t += h;
    }
    if( t < maxd ) res = t;
    return res;
}

// Function 838
vec4 raymarch(in vec3 pos, in vec3 dir, in float maxL) {
	float l = 0.;
	for (int i = 0; i < TRACE_STEPS; ++i) {
		float d = world(pos + dir * l);
		if (d < TRACE_EPSILON*l) break; // if we return here, browser will crash on mac os x, lols
		l += d;
		if (l > maxL) break;
	}
	return vec4(pos + dir * l, l);
}

// Function 839
vec2 raymarch(vec3 ro, vec3 rd, in float tmin, in float tmax) {
    vec2 m = vec2(-1.0, -1.0);
    vec2 res = vec2(tmin, -1.0);
    res.x = tmin;
	for( int i=0; i<NUM_STEPS; i++ )
	{
        m = map(ro + res.x*rd);
		if( m.x<tmin || res.x>tmax ) break;
		res.x += 0.5*m.x;
        res.y = m.y;
	}
    if( res.x>tmax ) res.y=-1.0;
	return res;
}

// Function 840
vec3 march (vec3 p, vec3 d) {
    float compoundedD = 0.;
    float rxcount = 0.;
    float rxindex = .3;
    vec3 finalcol = vec3(0.);
    for (float i=0.; i<60.; ++i) {
        float SDFp = SDF(p);
        float DE = SDFp;
        DE *= .999;
        if (SDFp < 1e-2) {
            p = p+d*SDFp*.99;
            int idSDFp = idSDF(p);
            vec3 TEXpd = TEX(p, d);
            if (idSDFp == 0) {
                finalcol = TEX(p, d);
                d = reflect(d, dSDF(p));
                p = p+d*.1;
                ++rxcount;
                continue;
            }
            if (rxcount == 0.) {
                return TEX(p, d);
            }
            return finalcol*(1.-rxindex)+rxindex*TEX(p, d);
        }
        p = p+d*DE;
        compoundedD += DE;
        if (compoundedD > 20. || SDFp > 7.) {
            break;
        }
    }
    if (rxcount > 0.) {
        finalcol = (1.-rxindex)*finalcol+rxindex*skycol(d);
        return finalcol;
    }
    return skycol(d);
}

// Function 841
float raymarch(vec3 ori, vec3 dir) {
 
    float t = 0.;
    for(int i = 0; i < MAX_ITER; i++) {
    	vec3  p = ori + dir * t;
        float d = dstScene(p);
        if(d < EPSILON || t > MAX_DIST)
            break;
        t += d * .75;
    }
    return t;
    
}

// Function 842
vec2 march(vec3 ro, vec3 rd)
{
   float t = 0.0;

   for(int steps =0; steps < 200; steps++)
   {
      vec2 d = f(ro  + t * rd); 
      if(d.x < 0.0001*t) return vec2(t, d.y);
      t += d.x;
      if(t>50.) break;
   }
   return vec2(-1);
}

// Function 843
MarchResult MarchRay(vec3 orig,vec3 dir)
{
    float steps = 0.0;
    float dist = 0.0;
    
    for(int i = 0;i < MAX_STEPS;i++)
    {
        float sceneDist = Scene(orig + dir * dist);
        
        dist += sceneDist * STEP_MULT;
        
        steps++;
        
        if(abs(sceneDist) < MIN_DIST)
        {
            break;
        }
    }
    
    MarchResult result;
    
    result.position = orig + dir * dist;
    result.normal = Normal(result.position);
    result.dist = dist;
    result.steps = steps;
    
    return result;
}

// Function 844
void Step (int mId, out vec3 rm, out vec3 vm, out vec4 qm, out vec3 wm)
{
  vec4 p;
  vec3 rmN, rmN1, rmN2, vmN, wmN, dr, dr1, dr2, dv, drw, am, wam;
  float fOvlap, fricN, fricT, fricS, fricSW, fDamp, fPull, grav, rSep,
     fc, ft, drv, dt;
  fOvlap = 500.;
  fricN = 10.;
  fricS = 0.1;
  fricSW = 1.;
  fricT = 0.5;
  fPull = 0.5;
  fDamp = 0.5;
  grav = 10.;
  p = Loadv4 (4 * mId);
  rm = p.xyz;
  vm = Loadv4 (4 * mId + 1).xyz;
  qm = Loadv4 (4 * mId + 2);
  wm = Loadv4 (4 * mId + 3).xyz;
  am = vec3 (0.);
  wam = vec3 (0.);
  for (int n = VAR_ZERO; n < nBall; n ++) {
    p = Loadv4 (4 * n);
    rmN = p.xyz;
    dr = rm - rmN;
    rSep = length (dr);
    if (n != mId && rSep < 1.) {
      fc = fOvlap * (1. / rSep - 1.);
      vmN = Loadv4 (4 * n + 1).xyz;
      wmN = Loadv4 (4 * n + 3).xyz;
      dv = vm - vmN;
      drv = dot (dr, dv) / (rSep * rSep);
      fc = max (fc - fricN * drv, 0.);
      am += fc * dr;
      dv -= drv * dr + cross (0.5 * (wm + wmN), dr);
      ft = min (fricT, fricS * abs (fc) * rSep / max (0.001, length (dv)));
      am -= ft * dv;
      wam += (ft / rSep) * cross (dr, dv);
    }
  }
  dr.xz = 0.5 * SurfNf (rm).xz;
  dr.y = rm.y + 0.5 - SurfHt (rm.xz - dr.xz);
  rSep = length (dr);
  if (rSep < 1.) {
    fc = fOvlap * (1. / rSep - 1.);
    dv = vm;
    drv = dot (dr, dv) / (rSep * rSep);
    fc = max (fc - fricN * drv, 0.);
    am += fc * dr;
    dv -= drv * dr + cross (wm, dr);
    ft = min (fricT, fricSW * abs (fc) * rSep / max (0.001, length (dv)));
    am -= ft * dv;
    wam += (ft / rSep) * cross (dr, dv);
  }
  am += vec3 (Rot2D (vec2 (fPull, 0.), pi * (0.25 + 0.1 * sin (0.001 * nStep))),
     - grav).xzy - fDamp * vec3 (1., 5., 1.) * vm;
  dt = (fBall < 0) ? 0.02 : 0.005;
  vm += dt * am ;
  rm += dt * vm;
  wm += dt * wam / 0.1;
  qm = normalize (QtMul (RMatToQt (LpStepMat (0.5 * dt * wm)), qm));
}

// Function 845
float rayMarch(vec3 ro, vec3 rd) {
	vec3 pos = ro;
    float inc = 0.;
    float d = 0.;
    for(int i=0;i<ITERATIONS;i++) {
    	pos = ro + d*rd;
        inc = sdScene(pos);
        d += inc;
        if(inc < MIN_D || d > MAX_D) break;
    }
    return d;
}

// Function 846
float RaymarchScene(in Ray ray, inout float nearest)
{
    float sdf = FarClip;
    
    vec3 o = (ray.origin) + (ray.direction * 2.0);
    
    for(float depth = NearClip; depth < FarClip; )
    {
    	vec3 pos = o + (ray.direction * depth);
        
        sdf = Scene_SDF(pos);
        nearest = min(sdf, nearest);
        
        if(sdf < Epsilon)
        {
            return depth;
        }
        
        depth += sdf;
    }
    
    return FarClip;
}

// Function 847
void marchWater(vec3 eye, vec3 ray, inout vec4 color) {
    const vec3 planeNorm = vec3(0.0, 1.0, 0.0);
    const float depth = 3.0;
    float ceilDist = intersectPlane(eye, ray, vec3(0.0, 0.0, 0.0), planeNorm);
    vec3 normal = vec3(0.0);
    if (dot(planeNorm, ray) > -0.05) {
        normal = vec3(0.0);
        color = vec4(vec3(0.0), CAM_FAR);
        return;
    }
    float height = 0.0;
    vec3 rayPos = eye + ray * ceilDist;
    for (int i = 0; i < 80; i++) {
        height = heightmap(rayPos.xz, WATER_MARCH_ITERATIONS) * depth - depth;
        if (rayPos.y - height < 0.1) {
            color.w = distance(rayPos, eye);
            vec3 normPos = (eye + ray * color.w);
            normal = waterNormal(normPos.xz, 0.005);
            color.rgb = waterColor(ray, normal, normPos);
            return;
        }
        rayPos += ray * (rayPos.y - height);
    }
}

// Function 848
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

// Function 849
bool MarchCameraRay( vec3 start, vec3 dir, out vec3 pos CACHEARG )
{
    vec3 boundsStart;
    if ( !RayHitsBlobBounds( start, dir, boundsStart ) )
	    return false;
    
    // assumes dir is normalized
    pos = boundsStart + dir * EPSILON;
    
    float prevMarchDist = EPSILON;
    float prevSurfaceDist = BlobDist( boundsStart CACHE );
    
    #if DEBUG_BOUNDS
    	if ( prevSurfaceDist <= 0.0 )
        {
            pos = boundsStart;
            return true;
        }
    #endif
    
    for ( int i = 0; i < MAX_RAYMARCH_STEPS; i++ )
    {
        float surfaceDist = BlobDist( pos CACHE );
        if ( surfaceDist <= EPSILON )
        {
            if ( surfaceDist < 0.0 )
            	pos = RefineSurfacePos( pos, dir, surfaceDist, prevSurfaceDist, prevMarchDist, EPSILON CACHE );
            return true;
        }
        
        // calculate the gradient of the function along the ray.
        // we're hoping that the gradient doesn't get suddenly steeper ahead of us.
        // to protect against that, we don't go lower than MIN_GRADIENT.
        // we want MIN_GRADIENT as low as possible without artifacts.
        float gradientAlongRay = (prevSurfaceDist - surfaceDist) / prevMarchDist;
        float safeGradient = max( gradientAlongRay, MIN_GRADIENT );
        
        float addDist = (surfaceDist + EPSILON) / safeGradient;
        prevMarchDist = addDist;
        
        prevSurfaceDist = surfaceDist;
        pos += dir * addDist;
        
        vec3 relPos = pos - BLOB_BOUNDING_CENTER;
        relPos *= BLOB_BOUNDING_SCALE;
        if ( dot( relPos, relPos ) > BLOB_BOUNDING_RADIUS_SQR )
            return false;
    }
    
    return true;
}

// Function 850
float raymarchWater(Ray ray, float tmin, float tmax) {
 
    float t = tmin;
    for(int i = 0; i < 256; i++) {
     
        vec3 p  = ray.ori + ray.dir * t;
        float h = p.y - getHeight(p.xz);
        if(h < EPSILON * t || t > tmax) break;
        t += h * .5;
        
    }
    return t;
    
}

// Function 851
vec3 _smoothstep(in vec3 p)
{
     return p * p * 3.0 - 2.0 * mul3x(p);
}

// Function 852
float raymarch(vec3 ori, vec3 dir, int iter) {
    float t = 0.;
    for(int i = 0; i < MAX_ITERATIONS; i++) {
        if(i >= iter) {
            t = MAX_DISTANCE;
        	break;
        }
        vec3  p = ori+dir*t;
        float d = dstScene(p);
        if(d < EPSILON)
            break;
        t += d * .75;
    }
    return t;
}

// Function 853
float rayMarch( in vec3 ro, in vec3 rd, float tmax )
{
    float t = 0.0;
    
    // bounding plane
    float h = (1.0-ro.y)/rd.y;
    if( h>0.0 ) t=h;

    // raymarch
    for( int i=0; i<20; i++ )    
    {        
        vec3 pos = ro + t*rd;
        float h = map( pos );
        if( h<0.001 || t>tmax ) break;
        t += h;
    }
    return t;    
}

// Function 854
bool raymarch(float3 start, float3 d, float t0, float t1,float stp, const int N, out float t)
{
    t=t0;
    
    int i=0;
    for(int j=0;j<1;j+=0)
    {
	    float3 p=start+d*t;
        float v=value(p);
        if(v<0.0)
            return true;
        i++;
        if(i>N)
            break;
        t+=min(0.01+v*0.25,stp);
    }
    return false;
}

// Function 855
float Raymarch( const in C_Ray ray )
{        
    float fDistance = .1;
    bool hit = false;
    for(int i=0;i < 50; i++)
    {
			float fSceneDist = MapToScene( ray.vOrigin + ray.vDir * fDistance );
			if(fSceneDist <= 0.01 || fDistance >= 150.0)
			{
				hit = true;
                break;
			} 

        	fDistance = fDistance + fSceneDist;
	}
	
	return fDistance;
}

// Function 856
float RayMarch(vec3 ro, vec3 rd,out float glowCumul) {
	float dO=0.0;  
    float dS;
    float dC; // distance to cell boundaries
    float glowDist;
    glowCumul=0.0;
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        dS = GetDist(p,glowDist,dC);
        dO += min(dS*.9,dC+0.05); 
        float at = 1.0 / (1. + pow(glowDist*20.,3.0) );
        glowCumul+=at;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }    
    return dO;
}

// Function 857
bool ray_marching( vec3 o, vec3 dir, inout float depth, inout vec3 n ) {
	float t = 0.0;
    float d = 10000.0;
    float dt = 0.0;
    for ( int i = 0; i < 128; i++ ) {
        vec3 v = o + dir * t;
        d = dist_field( v );
        if ( d < 0.001 ) {
            break;
        }
        dt = min( abs(d), 0.1 );
        t += dt;
        if ( t > depth ) {
            break;
        }
    }
    
    if ( d >= 0.001 ) {
        return false;
    }
    
    t -= dt;
    for ( int i = 0; i < 4; i++ ) {
        dt *= 0.5;
        
        vec3 v = o + dir * ( t + dt );
        if ( dist_field( v ) >= 0.001 ) {
            t += dt;
        }
    }
    
    depth = t;
    n = normalize( gradient( o + dir * t ) );
    return true;
    
    return true;
}

// Function 858
vec3 raymarch(in vec3 from, in vec3 dir) 

{
	float ey=mod(t*.5,1.);
	float glow,eglow,ref,sphdist,totdist=glow=eglow=ref=sphdist=0.;
	vec2 d=vec2(1.,0.);
	vec3 p, col=vec3(0.);
	vec3 origdir=dir,origfrom=from,sphNorm;
	
    for (int i=0; i<RAY_STEPS; i++) {
		if (d.x>det && totdist<6.0) {
			p=from+totdist*dir;
			d=de(p);
			det=detail*(1.+totdist*60.)*(1.+ref*5.);
			totdist+=max(detail,d.x); 
			if (d.y<.5) glow+=max(0.,.02-d.x)/.02;
		}
	}
	vec3 ov=normalize(vec3(1.,.5,1.));
	vec3 sol=dir+lightdir;
    float l=pow(max(0.,dot(normalize(-dir*ov),normalize(lightdir*ov))),1.5)+sin(atan(sol.x,sol.y)*20.+length(from)*50.)*.0015;
    totdist=min(5.9,totdist);
    p=from+dir*(totdist-detail);
    vec3 backg=.4*(1.2-l)+LIGHT_COLOR*l*.75;
	backg*=AMBIENT_COLOR*(1.-max(0.2,dot(normalize(dir),vec3(0.,1.,0.)))*.2);
	float fondo=0.;
	vec3 pp=p*.5+sin(t*2.)*.5;
    for (int i=0; i<10; i++) {
        fondo+=clamp(0.,1.,textur(pp+dir*float(i)*.01))*max(0.,1.-exp(-.05*float(i)))*2.;
    }
    vec3 backg2=backg*(1.+fondo*(FLOOR_COLOR)*2.);
    if (d.x<.01) {
        vec3 norm=normal(p);
		col=mix(light(p-abs(d.x-det)*dir, dir, norm, d.y),backg,1.-exp(-.3*totdist*totdist)); 
		col = mix(col, backg2, 1.0-exp(-.02*pow(abs(totdist),2.)));
	} else { 
		col=backg2;
	}
	vec3 lglow=LIGHT_COLOR*pow(abs(l),30.)*.5;
    col+=glow*(.3+backg+lglow)*.007;
	col+=lglow*min(1.,totdist*totdist*.2)*1.5;
    
	return min(vec3(1.),col); 
}

// Function 859
Hit raymarch(Ray ray)
{
	vec3 pos;
	Hit hit;
	hit.dist = 0.;
	Hit curHit;
	for (int i = 0; i < 40; i++)
	{
		pos = ray.org + hit.dist * ray.dir;
		curHit = scene(pos);
		hit.dist += curHit.dist;
		glowAmt += clamp(pow(curHit.dist+0.1, -8.),0.,0.15)*glows(curHit.index);
	}
	hit.index = curHit.index;
	hit.index = curHit.dist < 0.01 ? hit.index : -1.;
	return hit;
}

// Function 860
void march(
    in vec3 p, in vec3 nv,
    out vec4 color
) {
    vec2 tRange;
    float didHitBox;
    boxClip(BOX_MIN, BOX_MAX, p, nv, tRange, didHitBox);

    color = vec4(0.0);
    if (didHitBox < 0.5) {
        return;
    }

    float t = tRange.s;
    for (int i=0; i<800; i++) {
		// Get voxel data
        vec3 lmn = lmnFromWorldPos( p + (t+EPS)*nv );
        vec4 data = readLMN(lmn);

        vec3 curlV = readCurlAtLMN(lmn);

        float normalizedDensity = unmix(0.5, 3.0, data.w);
        float normalizedSpeed = pow(unmix(0.0, 10.0, length(data.xyz)), 0.5);
        float normalizedVorticity = clamp(pow(length(curlV),0.5), 0.0, 1.0);

        #ifdef VORTICITY_CONFINEMENT
        vec3 cbase = colormapInferno( normalizedVorticity );
        float calpha = pow(normalizedSpeed, 3.0);
        #else
        vec3 cbase = colormapInferno( normalizedSpeed );
        float calpha = pow(normalizedDensity, 3.0);
        #endif

        vec4 ci = vec4(cbase, 1.0)*calpha;

        // Determine path to next voxel
        vec3 curBoxMin, curBoxMax;
        boxFromLMN(lmn, curBoxMin, curBoxMax);

        vec2 curTRange;
        float curDidHit;
        boxClip(curBoxMin, curBoxMax, p, nv, curTRange, curDidHit);

        // Adjust alpha for distance through the voxel
        ci *= clamp((curTRange.t - curTRange.s)*15.0, 0.0, 1.0);

        // Accumulate color
        color = vec4(
            color.rgb + (1.0-color.a)*ci.rgb,
            color.a + ci.a - color.a*ci.a
        );

        // Move up to next voxel
        t = curTRange.t;
        if (t+EPS > tRange.t || color.a > 1.0) { break; }
    }
}

// Function 861
vec3 RayMarch(vec3 ro,vec3 rd)
{
    float hd = 0.0;
    id=0;
    for(int i = 0;i < 128;i++)
    {
        float d = Scene(ro + rd * hd);
        hd += d;
        if(d < 0.0001) {id=1;break;}
    }   
    return ro + rd * hd;
}

// Function 862
void marchRay(inout Ray ray, inout vec3 colour) {
    bool inside = false; // are we inside or outside the glass object
    vec3 impact = vec3(1); // This decreases each time the ray passes through glass, darkening colours

    vec3 startpoint = ray.origin;
    
#ifdef DEBUG   
vec3 debugColour = vec3(1, 0, 0);
#endif
    
    SDResult result;
    vec3 n;
    vec3 glassStartPos;
    
    //float glow = 0.0;
    
    for (int i=0; i<kMAXITERS; i++) {
        // Get distance to nearest surface
        result = sceneDist(ray);
        
        //glow += result.material == kGLOWMATERIAL ? 
        //    pow(max(0.0, (80.0 - result.d) * 0.0125), 4.0) * result.d * 0.01
        //    : 0.0;
        
        // Step half that distance along ray (helps reduce artefacts)
        float stepDistance = (inside ? abs(result.d) : result.d) * 0.65;
        ray.origin += ray.dir * stepDistance;
        //if (length(ray.origin) > 40.0) { break; }
        
        if (stepDistance < eps) {
            // colision
            // normal
            // Get the normal, then clamp the intersection to the surface
    		n = normal(ray);
            //clampToSurface(ray, stepDistance, n);
#ifdef DEBUG
//debugColour = n;
//break;
#endif
            
            if ( result.material == kFLOORMATERIAL ) {
                // ray hit floor
                
                // Add some noise to the normal, since this is pretending to be grit...
                vec3 randomNoise = texrand(ray.origin.xz * 0.4, 0.0);
                n = mix(n, normalize(vec3(randomNoise.x, 1, randomNoise.y)), randomNoise.z);
                
                // Colour is just grey with crappy fake lighting...
                colour += mix(
                    kFLOORCOLOUR, 
                    vec3(0), 
                    pow(max((-n.x+n.y) * 0.5, 0.0), 2.0)
                ) * impact;
                float o = occlusion(ray, n);
#ifdef DEBUG
debugColour = vec3(o);
break;
#endif
                colour *= o;
                impact *= 0.;
                break;
            }
            
            if (result.material == kGLOWMATERIAL) {
             	colour = mix(colour, kGLOWCOLOUR, impact);
                impact *= 0.;
                break;
            }
            
            // check what material it is...
            
            if (result.material == kMIRRORMATERIAL) {
                
                // handle interior glass / other intersecion
                if (inside) {
                     float glassTravelDist =  min(distance(glassStartPos, ray.origin) / 16.0, 1.);
    				glassStartPos = ray.origin;
                    // mix in the colour
                	impact *= mix(kGLASSCOLOUR, kGLASSCOLOUR * 0.1, glassTravelDist);
                    
                }
                
                // it's a mirror, reflect the ray
                ray.dir = reflect(ray.dir, n);
                    
                // Step 2x epsilon into object along normal to ensure we're beyond the surface
                // (prevents multiple intersections with same surface)
                ray.origin += n * eps * 4.0;
                
                // Mix in the mirror colour
                colour += highlight(ray, n);
                impact *= kMIRRORCOLOUR;
                float o = occlusion(ray, n);
                impact *= o;
#ifdef DEBUG
debugColour = vec3(o);
break;
#endif
                
            } else {
                // glass material
            
                if (inside) {
                	// refract glass -> air
                	ray.dir = refract(ray.dir, -n, 1.0/kREFRACT);
                    
                    // Find out how much to tint (how far through the glass did we go?)
                    float glassTravelDist =  min(distance(glassStartPos, ray.origin) / 16.0, 1.);
    
                    // mix in the colour
                	impact *= mix(kGLASSCOLOUR, kGLASSCOLOUR * 0.1, glassTravelDist);
                    
#ifdef DEBUG
debugValue += glassTravelDist / 2.0;
#endif
      
                
              	} else {
               		// refract air -> glass
                	glassStartPos = ray.origin;
                    
              	  	// Mix the reflection in, according to the fresnel term
                	float fresnel = fresnelTerm(ray, n, 1.0);
                    fresnel = fresnel;
    				/*
                    colour = mix(
                    	colour, 
                    	texture(iChannel1, reflect(ray.dir, n)), 
                    	vec4(fresnel) * impact);
*/
                    colour = mix(
                        colour,
                        backgroundColour(ray, 0.0),
                        vec3(fresnel) * impact);
                	colour += n.x * 0.1;//highlight(ray, n);
                    impact *= 1.0 - fresnel;
    			
                	// refract the ray
            		ray.dir = refract(ray.dir, n, kREFRACT);
                    
#ifdef DEBUG
//debugValue += 0.5;
#endif
                }
            
            	// Step 2x epsilon into object along normal to ensure we're beyond the surface
                ray.origin += (inside ? n : -n) * eps * 2.0;
                
                // Flip in/out status
                inside = !inside;
            }
        }
        
        // increase epsilon
        eps += divergence * stepDistance;
    }
    
    // So far we've traced the ray and accumulated reflections, now we need to add the background.
   // colour += texture(iChannel0, ray.dir) * impact;
    ray.origin = startpoint;
    colour.rgb += backgroundColour(ray, 0.0) * impact; // + glow * kGLOWCOLOUR;
    
#ifdef DEBUG
//debugColour.rgb = ray.dir;
//debugColour = vec3(float(debugStep)/2.0);
colour = debugColour;
#endif
}

// Function 863
vec4 raymarch(vec3 origin, vec3 dir) {
	float t = 0.0;
	const int max_steps = 256;
	float epsilon = 0.00075;
	vec3 fog_color = vec3(1.0);
	vec4 color = vec4(fog_color,1);
	int num_reflections = 1;
	int reflections_left = num_reflections;
	for(int i = 0; i < max_steps; ++i) {
		vec3 p = origin + dir * t;
		float d = total_distance(p);
		if(d < epsilon) {
			vec3 n = normal(p);
			vec3 sample_color = vec3(0.8);
			// distance fade
			sample_color = mix(sample_color, fog_color, vec3(clamp(1.0-exp(-length(p)/8.0), 0.0, fog_color.x))); // Sphere color
			// front light
			//color.xyz += 0.1 * vec3(1) * clamp(dot(-dir, n),0,1);
			// ambient occlusion
			sample_color *= ambient_occlusion(p, n, num_reflections - reflections_left);

			if (false) {
				float step_intensity = 1.0-(float(i))/float(max_steps);
				color.yz *= vec2(step_intensity);
			}
            float f = 1.0;
            if (num_reflections != reflections_left)
            	f = 0.075;
			color.xyz = mix(color.xyz, sample_color, f);
            break;/*
			if (reflections_left == 0) {
				break;
			} else {
				// restart loop starting from new reflection point
				reflections_left -= 1;
				epsilon *= 10.0;
				origin = p + n * epsilon * 2.0;
				dir = normalize(dir-2.0*n*dot(dir,n));
				i = 0;
				t = 0.0;
				max_steps /= 4;
				continue;
			}*/
		}
		t += d * 1.0;
	}
	return vec4(pow(color.xyz, vec3(2.2)), color.w);
}

// Function 864
float Linstep(float a, float b, float t)
{
	return clamp((t-a)/(b-a),0.,1.);

}

// Function 865
void march(inout vec3 pos, vec3 dir)
{
    float dmin;
    march(pos,dir,dmin);
}

// Function 866
HitInfo raymarch(const in Ray ray, float start, float end) {
    float depth = start;
    HitInfo hit;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        hit = sceneH(ray.eye + depth * ray.dir);
        if (hit.dist < EPSILON) {
            return HitInfo(hit.object, depth);
        }
        depth += hit.dist;
        if (depth >= end) {
            return HitInfo(NO_HITS, 0.);
        }
    }
    return HitInfo(NO_HITS, 0.);
}

// Function 867
vec2 raymarch(vec3 ro, vec3 rd) {
	float tmin = .01;
    float tmax = 80.;
    float m = -1.;
    float t = tmin;
    for(int i = 0; i < 99; i++) {
		vec3 pos = ro + rd * t;
        vec2 h = scene(pos);
        m = h.y;
        if(abs(h.x) < (stopThreshold * t)) break;
        t += h.x;
        if(t > tmax) break;
    }
    if(t > tmax) m = -1.;
    return vec2(t, m);
}

// Function 868
float rayMarch(vec3 pos, vec3 dir, vec2 uv, float d)
{
    d = max(d,.0);
    float oldD = d;
    bool hit = false;
    for (int i = 0; i < 80; i++)
    {
		float de = mapDE(pos + dir * d);

       if(de < sphereSize(d) || d > 2000.0) break;

		oldD = d;
		d += 0.5*de;

        
   
    }
	if (d > 2000.0)
		oldD = 2000.0;

    
    return oldD;
}

// Function 869
SRayHitInfo TestSceneMarch(in vec3 rayPos)
{
    SRayHitInfo hitInfo;
    hitInfo.hitAnObject = false;
    hitInfo.dist = c_superFar;
    return hitInfo;
}

// Function 870
vec3 marcher(vec3 ro, vec3 rd, int maxstep, float sol){
	float d =  .00001,
     	  m = -1.;
    	float glowDist = 1e9;
    	int i = 0;
        for(i=0;i<maxstep;i++){
        	vec3 p = ro + rd * d;
            vec2 t = map(p, sol);
            if(abs(t.x)<MINDIST)break;
            d += t.x*.85;
            m  = t.y;
            if(d>MAXDIST)break;
            glowDist = min(glowDist, d);
        }
    return vec3(d,m,glowDist);
}

// Function 871
RayMarchObject RayMarch(vec3 ro, vec3 rd) 
{
	float dO=0.;
    int materialID;
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        RayMarchObject object = GetScene(p);
        float dS = object.dist;
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) {
            materialID = object.materialID;
            break;
        }
    }
    
    return RayMarchObject(materialID, dO);
}

// Function 872
float rayMarch(vec3 pos, vec3 dir, vec2 uv, float d)
{
    d = max(d,.0);
    bool hit = false;
	float de = 0.0, od = 0.0;
    for (int i = 0; i < 150; i++)
    {
        de = mapDE(pos + dir * d);

       if(de < sphereSize(d)  || d > 2000.0) break;

        od = d;
		d += 0.5*de;

   
    }
	if (d < 2000.0)
        d = binarySubdivision(pos, dir, vec2(d, od));
	else
		d = 2000.0;
    
    return d;
}

// Function 873
SurfaceInteraction rayMarch(vec3 ro, vec3 rd) {
    
    SurfaceInteraction interaction = SurfaceInteraction(-1., rd, vec3(0.), vec3(0.), vec3(0.), vec3(0.), -10.);
    
    float t = 0.;
    vec3 p = ro;    
    vec2 obj = vec2(0.);
    float d = INFINITY;
    
    for (int i = 0; i < RAY_MARCH_STEPS; i++) {
        
        obj = map(p);
        d = obj.x;
        
        t += d;
        p += rd * d;
        
        if (d < .001) { break; }
        obj.y = 0.;
        
    }

    interaction.id = obj.y;        
    interaction.point = p;
    interaction.normal = calculateNormal(interaction.point);
    interaction.objId = obj.y;
    return interaction;
}

// Function 874
vec4 march(vec3 origin, vec3 direction) {
    
    const int mainSteps = 30;
    const int shadowSteps = 10;
    const vec3 toLight = normalize(vec3(1.0,1.0,0.));
    const float mainDensityScale = 1.5;
    
    const float shadowingThreshold = 0.001;
    const float shadowDensityScale = 3.;
    
    vec3 light = vec3(0.);
    float transmittance = 1.;
    
    vec3 samplePosition = origin;
   
    const float mainStepLength = 2. / float(mainSteps); // why does lowering this below 2 change the appearance?
    const float shadowStepLength = 1. / float(shadowSteps);
    
    const vec3 scaledShadowDensity = shadowDensityScale * shadowStepLength / vec3(0.8,0.7,1.0);
    
    const float shadowConstant = -log(shadowingThreshold) / scaledShadowDensity.z;
    
    const vec3 mainLightColor = vec3(0.6,0.8,1.);
    const vec3 innerLightColor = vec3(0.7,0.4,1.) * 4.;
    
    vec3 mainStepAmount = direction * mainStepLength;
    
    vec3 shadowStepAmount = toLight * shadowStepLength;
    
    vec4 innerLight = innerLightPositionAndIntensity();
    
    for(int i = 0; i < mainSteps; i++) {
        float localDensity = min(1.0, density(samplePosition) * mainDensityScale);
        if (localDensity > 0.001) {
            
            // - main light (directional)
            
            vec3 shadowSamplePosition = samplePosition;
            float shadowAccumulation = 0.;
            for(int j = 0; j < shadowSteps; j++) {
                shadowSamplePosition += shadowStepAmount;
                
                shadowAccumulation += min(1.0, density(shadowSamplePosition) * shadowDensityScale);
                if (shadowAccumulation > shadowConstant || dot(shadowSamplePosition, shadowSamplePosition) > 1.) break;
            }
            
            vec3 shadowTerm = exp(-shadowAccumulation * scaledShadowDensity);
            float stepDensity = min(1.,localDensity * mainStepLength);
            vec3 absorbedLight = shadowTerm * stepDensity;
            
            // accumulate directional light
            light += absorbedLight * transmittance * mainLightColor;
            
            
            // - inner light (point)
            
            shadowSamplePosition = samplePosition;
            shadowAccumulation = 0.;
            vec3 toInnerLight = innerLight.xyz - samplePosition;
            vec3 innerLightShadowStepAmount = normalize(toInnerLight) * shadowStepLength;
            
            for(int j = 0; j < shadowSteps; j++) {
                shadowSamplePosition += innerLightShadowStepAmount;
                
                shadowAccumulation += min(1.0, density(shadowSamplePosition) * shadowDensityScale);
                
                // bail out if we’ve accumulated enough or if we’ve gone outside the bounding sphere (squared length of the sample position > 1)
                if (shadowAccumulation > shadowConstant || dot(shadowSamplePosition, shadowSamplePosition) > 1.) break;
            }
            
            shadowTerm = exp(-shadowAccumulation * scaledShadowDensity);
            stepDensity = min(1.,localDensity * mainStepLength);
            absorbedLight = shadowTerm * stepDensity;
            
            // inverse-squared fade of the inner point light
            float attenuation = min(1.0, 1.0 / (dot(toInnerLight, toInnerLight) * 2. + 0.0001)) * innerLight.w;
            
            // accumulate point light
            light += absorbedLight * (transmittance * attenuation) * innerLightColor;
            
            // -
            
            transmittance *= (1. - stepDensity);

            if (transmittance < 0.01) {
                break;
            }
        }
        
        samplePosition += mainStepAmount;
    }
    
    return vec4(vec3(light), transmittance);
}

// Function 875
float linstep(in float mn, in float mx, in float x){
	return clamp((x - mn)/(mx - mn), 0., 1.);
}

// Function 876
float aastep(float frequency, float threshold, float value)
{
	float afwidth = frequency / 512.;
	return smoothstep(threshold-afwidth, threshold+afwidth, value);
}

// Function 877
float march(vec3 ro, vec3 rd) {
  float t = 0.1;
  for(int i = 0; i < 200; i++) {
    //assert(i < 100);
    float h = de(ro + rd * t);
    if (h < t*precis) return t;
    if (doinvert) h *= 0.5;
    t += h;
    if (t < 0.0 || t > maxdist) break;
  }
  return -1.0;
}

// Function 878
vec4 raymarche( in vec3 org, in vec3 dir, in vec2 nfplane )
{
	float d = 1.0, g = 0.0, t = 0.0;
	vec3 p = org+dir*nfplane.x;
	
	for(int i=0; i<42; i++)
	{
		if( d > 0.001 && t < nfplane.y )
		{
			d = map(p);
			t += d;
			p += d * dir;
			g += 1./42.;
		}
	}
	
	return vec4(p,g);
}

// Function 879
float RAYMARCH_DFAO( vec3 o, vec3 N, float isoSurfaceValue)
{
    //Variation of DFAO from : https://www.shadertoy.com/view/Xds3zN
    //Interesting reads:
    //https://docs.unrealengine.com/latest/INT/Engine/Rendering/LightingAndShadows/DistanceFieldAmbientOcclusion/index.html#howdoesitwork?
    //Implementation notes:
    //-Doubling step size at each iteration
    //-Allowing negative distance field values to contribute, making cracks much darker
    //-Not reducing effect with distance (specific to this application)
    float MaxOcclusion = 0.0;
    float TotalOcclusion = 0.0;
    const int nSAMPLES = 4;
    float stepSize = 0.11/float(nSAMPLES);
    for( int i=0; i<nSAMPLES; i++ )
    {
        float t = 0.01 + stepSize;
        //Double distance each iteration (only valid for small sample count, e.g. 4)
        stepSize = stepSize*2.0;
        float dist = DF_composition( o+N*t ).d-isoSurfaceValue;
        //Occlusion factor inferred from the difference between the 
        //distance covered along the ray, and the distance from other surrounding geometry.
        float occlusion = zclamp(t-dist);
        TotalOcclusion += occlusion;//Not reducing contribution on each iteration
        MaxOcclusion += t;
    }
    
    //Here, TotalOcclusion can actually exceed MaxOcclusion, where the rays
    //get inside the shape and grab negative occlusion values. It does look good
    //that way IMHO (much darker in the cracks), therefore the maximum occlusion is bumped
    //25% to allow those cracks to get darker.
    return saturate(1.0-TotalOcclusion/(MaxOcclusion*1.25));
}

// Function 880
float smootherstep(float edge0, float edge1, float x)
{
	x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
	return x*x*x*(x*(x*6.0 - 15.0) + 10.0);
}

// Function 881
bool starshipMarch(vec3 eye, vec3 ray, out vec3 pos) {
    float dist = 0.0;
    for (int i = 0; i < marchIter; ++i) {
        pos = eye + dist * ray;
        float sdf = starshipSdf(pos);
        dist += sdf;
        if (sdf < epsilon)
            return true;
        if (dist >= marchDist)
            return false;
    }
    return true;
}

// Function 882
Hit raymarch(Ray ray) {
 
    vec3 p = ray.ori;
    bool b = false;
    
    for(int i = 0; i < 128; i++) {
     
        float dst = dstScene(p);
        p += ray.dir * dst * .75;
        
        if(dst < .001) {
         
            b = true;
            break;
            
        }
        
    }
    
    return Hit(p,b);
    
}

// Function 883
vec2 rayMarch(vec3 ro, vec3 rd) {
	float dO=0.;
    float matId = -1.;
    
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        vec2 res = getDist(p);
        float dS = res.x;
        dO += dS;
        matId = res.y;
        
        if(dO>MAX_DIST || dS<SURF_DIST) break;
    }
    
    return vec2(dO, matId);
}

// Function 884
float raymarch(vec3 ray_ori, vec2 uv) {
	vec3 target = vec3(0);
	vec3 ray_dir = camera(ray_ori, target, uv);
	vec3 pos = ray_ori;

	// local density/distance
	float ldensity = 0.;

	// accumulation color & density
	vec4 sum = vec4(0.);

	float tmax = 25.;
	float tdist = 0., dist = 0.;

	for (float i = 0.; (i < 1.); i += 1. / 64.) {

		if (dist < tdist * .001 || tdist > tmax || sum.a > .95)
			break;

		// evaluate distance function
		dist = de(pos) * .59;

		// check whether we are close enough (step)
		// compute local density and weighting factor 
		const float h = .05;
		ldensity = (h - dist) * step(dist, h);

		vec4 col = vec4(1);
		col.a = ldensity;

		// pre-multiply alpha
		// checkout: https://www.shadertoy.com/view/XdfGz8
		// http://developer.download.nvidia.com/assets/gamedev/files/gdc12/GDC2012_Mastering_DirectX11_with_Unity.pdf
		col.rgb *= col.a;
		sum += (1. - sum.a) * col;

		// from duke/las
		sum.a += .004;

		// enforce minimum stepsize
		dist = max(dist, .03);

		// step forward
		pos += dist * ray_dir; // sphere-tracing
		tdist += dist;
	}

	// from duke/las
	// simple scattering approximation
	sum *= 1. / exp(ldensity * 3.) * 1.25;

	sum.r = pow(sum.r, 2.15);
	sum.r -= texture(iChannel0, uv * 6.).r * .18;

	return sum.r;
}

// Function 885
bool RayMarchPerturbedSphere(in vec3 ro, in vec3 rd, in vec3 c, in float r, in float br, 
                             out vec3 n, out vec3 sd) {
    n = vec3(0.0);
    sd = vec3(0.0);
    
    vec3 bp0 = vec3(0.0);
    vec3 bp1 = vec3(0.0);
    bool bres = RaySphereIntersection(ro, rd, c, br, bp0, bp1);
    if (!bres) return false;
    
    vec3 p0 = vec3(0.0); 
    vec3 p1 = vec3(0.0);
    bool res = RaySphereIntersection(ro, rd, c, r, p0, p1); 
    
    float dist = float(res)*length(p0 - bp0) + (1.0-float(res)) * length(bp0 - bp1);
	//float dist = length(bp0 - bp1);
    const float sc = 128.0;
    const float invsc = 1.0 / sc;
    float s = dist * invsc;
    
    bool ret = false;
    vec3 pn = vec3(0.0);
    for (float d = 0.0; d < sc; ++d) {
    	pn = (bp0 + d*s*rd) - c;
		
        sd = normalize(pn) * r;
        float h = length(pn) - r - s;
        
        float h0 = noise(sd);
        if (h0 > h) {
            ret = true;
            break;
        } 
    }
    
    n = SphereNormal(normalize(pn), r, s);
    return ret;
}

// Function 886
float march(vec3 ro, vec3 rd){
    float t = 0.001;
    float step = 0.0;

    float omega = 1.0;//muista testata eri arvoilla! [1,2]
    float prev_radius = 0.0;

    float candidate_t = t;
    float candidate_error = 1000.0;
    float sg = sgn(dist(ro));

    vec3 p = vec3(0.0);

	for(int i = 0; i < STEPS; ++i){
		p = rd*t+ro;
		float sg_radius = sg*dist(p);
		float radius = abs(sg_radius);
		step = sg_radius;
		bool fail = omega > 1. && (radius+prev_radius) < step;
		if(fail){
			step -= omega * step;
			omega = 1.;
		}
		else{
			step = sg_radius*omega;
		}
		prev_radius = radius;
		float error = radius/t;

		if(!fail && error < candidate_error){
			candidate_t = t;
			candidate_error = error;
		}

		if(!fail && error < EPSILON || t > FAR){
			break;
		}
		t += step;
	}
    //discontinuity reduction
    float er = candidate_error;
    for(int j = 0; j < 6; ++j){
        float radius = abs(sg*dist(p));
        p += rd*(radius-er);
        t = length(p-ro);
        er = radius/t;

        if(er < candidate_error){
            candidate_t = t;
            candidate_error = er;
        }
    }
	if(t <= FAR || candidate_error <= EPSILON){
		t = candidate_t;
	}
	return t;
}

// Function 887
float RayMarch(vec3 ro, vec3 rd){
    // distance from origin
    float dO=0.;
    // march until max steps is achieved or object hit
    for(int i=0; i <MAX_STEPS; i++){
        // current point being evaluated
        vec3 p = ro + dO*rd;
        
        // get distance to seam
        float ds = GetDist(p);
        //move origin to new point
        dO+=ds*.7;
        if(ds < SURFACE_DIST || dO > MAX_DIST){
            break;
        }
    }
    return dO;
}

// Function 888
float rayMarch( in vec3 origin, in vec3 direction ) {
    float total = .0;
    for ( int i = 0 ; i < RAY_MARCH_STEPS ; i++ ) {
        vec3 point = origin + direction * total;
                
        float current = sceneDistance( point );
        total += current;
        if ( total > RAY_MARCH_TOO_FAR || current < RAY_MARCH_CLOSE ) {
            break;
        }
    }
    return total;
}

// Function 889
RayHit March( vec3 origin,  vec3 direction)
{
  RayHit result;
  float maxDist = 380.0;
  float t = 0.0, dist = 0.0;
  vec3 rayPos;
 
  for ( int i=0; i<200; i++ )
  {
    rayPos =origin+direction*t;
    dist = Map( rayPos);
 

    if (dist<0.01 || t>maxDist )
    {             
      result.hit=!(t>maxDist);
      result.depth = t; 
      result.dist = dist;                              
      result.hitPos = origin+((direction*t));   
      result.steelDist = steelDist;
      result.platformDist = platformDist;
      result.terrainDist =terrainDist;
      result.waterDist =waterDist;
      break;
    }
    t += dist;
  }

  return result;
}

// Function 890
vec3 raymarching(vec3 ro, vec3 rd)
{
    vec3 bg = vec3(0.0);
    
    vec2 res = render(ro, rd);
    vec3 col;
    if (res.y > -0.5) {
        vec3 pos = ro + rd * res.x;
        vec3 nor = normal(pos);
        vec3 view = -rd;
        float dif = dot(nor, view);
        dif = dif * 0.5 + 0.5;
        
        float h = clamp(height(pos) + 0.2, 0.0, 1.0) * HEIGHT;
        vec3 light = vec3(0.3 + hash11(res.y * 0.2), 0.3 + hash11(res.y * 0.4), 0.3 + hash11(res.y * 0.6));
        col = vec3(0.8, 0.85, 1.0);
        light = palette(pos.y * 0.2, vec3(0.5, 0.5, 0.5),vec3(0.5, 0.5, 0.5),vec3(1.0, 1.0, 0.5), light);
        col = mix(light, col, smoothstep(h * 0.8, h, pos.y)) * dif;
    	col = mix(bg, col, smoothstep(0.0, 0.3, pos.y));
    }
    
    // distance fog
    col = mix(col, bg, smoothstep(DMIN, DMAX, res.x));
    
    //return vec3(res.y + 1.0);
    return col;
}

// Function 891
vec4 raymarch(vec4 P,vec3 R)
{
    P = vec4(P.xyz+R*2.,2);
    float E = 1.;
 	for(int i = 0;i<300;i++)
    {
        P += vec4(R,1)*E;
        float H = height(P.xy);
        E = clamp(E+(H-P.z)-.5,E,1.);
        if (H-E*.6<P.z)
        {
        	P -= vec4(R,1)*E;
            E *= .7;
            if (E<PRE*P.w/FOG) break;
        }
    }
    return P;
}

// Function 892
float aaStep(float a, float b, float x)
{
    // lerp step, make sure that a != b
    x = clamp(x, a, b);
    return (x - a) / (b - a);
}

// Function 893
float noiseStep(float i, float a, float b, vec2 x) {
    float d = 0.2*(b-a);
	return 1.0-i+(smoothstep(a-d, b+d, noise(x))*(i));
}

// Function 894
float shared_smoothstep(float _551, float _552, float _553)
{
    float _558 = shared_saturate((_553 - _551) / (_552 - _551));
    return (_558 * _558) * (3.0 - (2.0 * _558));
}

// Function 895
vec2 rayMarch(vec3 ro, vec3 rd)
{
    float t = 0.0001;
    vec2 res;
    for(int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + t*rd;
        res = map(p);
        if(res.x < SURF_DIST || res.x > MAX_DIST) break;
        t += res.x;
    }

    if (t > MAX_DIST) t = -1.0;
    
    return vec2(t, res.y);
}

// Function 896
Hit raymarch(CastRay castRay){

    float currentDist = INTERSECTION_PRECISION * 2.0;
    Model model;
    
    Ray ray = Ray(castRay.origin, castRay.direction, 0.);

    for( int i=0; i< NUM_OF_TRACE_STEPS ; i++ ){
        if (currentDist < INTERSECTION_PRECISION || ray.len > MAX_TRACE_DISTANCE) {
            break;
        }
        model = map(ray.origin + ray.direction * ray.len);
        currentDist = model.dist;
        ray.len += currentDist * FUDGE_FACTOR;
    }
    
    bool isBackground = false;
    vec3 pos = vec3(0);
    vec3 normal = vec3(0);
    vec3 color = vec3(0);
    
    if (ray.len > MAX_TRACE_DISTANCE) {
        isBackground = true;
    } else {
        pos = ray.origin + ray.direction * ray.len;
        normal = calcNormal(pos);
    }

    return Hit(ray, model, pos, isBackground, normal, color);
}

// Function 897
float continuousSmoothstep(float x, float w){
    float v = x - w * .5;
    float b = floor(v);
	return b + smoothstep(b + 1. - w, b + 1., v);
}

// Function 898
vec3 raymarch( in vec3 ro, vec3 rd, vec2 tminmax )
{
    float t = tminmax.x;
    float dt = .02;
    //float dt = .2 - .195*cos(iTime*.05);//animated
    vec3 col= vec3(0.);
    vec3 c = vec3( 0.);
    for( int i=0; i<INNER_DEPTH; i++ )
	{
        t+=dt*exp(-2.*length(c));
        if(t>tminmax.y)break;
        //vec3 pos = ro+t*rd;
        vec3 pos = refract( ro, (ro+t*rd)/.7,-.012);
        c = map(pos);//map(ro+t*rd);               
        
        float gr= MILKY_LIGHT*0.013824/float(INNER_DEPTH);//.01*(2.0-float(i)/64.0);
        
        col = .995*col+ .09*c+vec3(gr);//vec3(c*c, c, c*c*c);//green	
        //col = .99*col+ .08*vec3(c*c*c, c*c, c);//blue
    }    
    return col;
}

// Function 899
void march_o349467(inout float d,inout vec3 p,inout vec2 dS, vec3 ro, vec3 rd){
    for (int i=0; i < 500; i++) {
    	p = ro + rd*d;
        dS = input_o349467(p);
        d += dS.x;
        if (d > 50.0 || abs(dS.x) < 0.0001) break;
    }
}

// Function 900
void RaymarchScene(in Ray ray, inout RayHit hit)
{
    hit.hit = false;
    
    vec2 sdf   = vec2(FarClip, 0.0);
    vec3 hNorm = vec3(0.0);
    
    float depth = NearClip;
    
    for(int steps = 0; (depth < FarClip) && (steps < 80); ++steps)
    {
    	vec3 pos = ray.origin + (ray.direction * depth);
        
        hit.steepness = 0.0;
        hNorm         = Scene_Normal(pos, 0.0, hit);
        hit.steepness = smoothstep(0.75, 1.0, hNorm.y);
        
        sdf = Scene_SDF(pos, hit);
        
        if(sdf.x < Epsilon)
        {
            hit.hit       = true;
            hit.surfPos   = pos;
            hit.surfNorm  = Scene_Normal(pos, 1.0, hit);
            hit.depth     = depth;
            hit.heightmap = sdf.y;
            
            break;
        }
        
        depth += sdf.x;
    }
}

// Function 901
bool RayMarching(in vec2 screen_pos, in float near, in float far, out vec3 hit_pos, out float cal_delta)
{
	const float delta = 1.0/float(K_MACRO_STEP);
	
	float z_param1 = far-near;
	float z_param2 = near/z_param1;
	
	for (int i=0;i<K_MACRO_STEP; ++i)
	{
		vec3 pos = vec3(screen_pos, float(i)*delta);
		//ortho inv trans
		pos.z += z_param2;
		pos.z *= z_param1;
		
		//view inv trans
		//pass
		
		vec3 pre_pos = vec3(pos.xy, pos.z - delta * z_param1);
		
		vec3 re = pre_pos;
		if( Hit(pos, pre_pos, re))
		{
			hit_pos = re;
			cal_delta = delta*z_param1;
			return true;
		}
	}
	return false;
}

// Function 902
vec3 rayMarching(vec3 start,vec3 dir,float maxDepth){
    vec3 steps=-dir/dir.z*maxDepth/float(iteration);
    vec3 pos=start;
    for(int i=0;i<iteration+1;i++){
        vec3 next=pos+steps;
        if(next.z+texDepth(next.xy)<0.){
            for(int j=0;j<binaryPass;j++){
                vec3 mid=(pos+next)/2.;
                if(mid.z+texDepth(mid.xy)<0.){
                    next=mid;
                }else{
                	pos=mid;
                }
            }
            return pos;
        }
        pos=next;
    }
    return pos;
}

// Function 903
void march(vec3 origin, vec3 dir, out float t, out int hitObj)
{
    t = 0.001;
    for(int i = 0; i < RAY_STEPS; ++i)
    {
        vec3 pos = origin + t * dir;
    	float m;
        sceneMap3D(pos, m, hitObj);
        if(m < 0.01)
        {
            return;
        }
        t += m;
    }
    t = -1.0;
    hitObj = -1;
}

// Function 904
vec3 raymarch_lsdlive(vec3 ro, vec3 rd, vec2 uv) {
	vec3 p;
	float t = 0., ri;

	float dither = random(uv);

	for (float i = 0.; i < 1.; i += .02) {// 50 iterations to keep it "fast"
		ri = i;
		p = ro + rd * t;
		float d = de(p);
		d *= 1. + dither * .05; // avoid banding & add a nice "artistic" little noise to the rendering (leon gave us this trick)
		d = max(abs(d), .002); // phantom mode trick from aiekick https://www.shadertoy.com/view/MtScWW
		t += d * .5;
	}

	// Shading: uv, iteration & glow:
	vec3 c = mix(vec3(.9, .8, .6), vec3(.1, .1, .2), length(uv) + ri);
	c.r += sin(p.z * .1) * .2;
	c += g * .035; // glow trick from balkhan https://www.shadertoy.com/view/4t2yW1

	return c;
}

// Function 905
vec3 path_march(vec3 p, vec3 ray, float t, float i, float angle, float seed)
{
    vec3 fincol = vec3(1.), finill = vec3(0.);
    vec4 res = vec4(0.);
    for(float b = 0.; (b < MAX_BOUNCE); b++)
    {
        if(b < 1.)
        {
            float h = map(p).w;
            if (h < angle*t || t > MAX_DIST)
            {
                 res = vec4(p, h);
            }
        }
       
        if(res.xyz != p)
        {
            //march next ray
       		res = trace(p, ray, t, i, angle);
        }
         
        if(t > MAX_DIST || (i >= MAX_STEPS && res.w > 5.*angle*t))
        {
            finill += sky(ray)*fincol;
            break;
        }
        
        /// Surface interaction
        vec3 norm = calcNormal(res.xyz, res.w);    
        //discontinuity correction
        p = res.xyz - (res.w - 2.*angle*t)*norm;
        
        vec3 refl = reflect(ray, norm);
        
        float refl_prob = hash(seed*SQRT2);
       
        //random diffusion, random distr already samples cos(theta) closely
        if(refl_prob < reflection)
        {
            vec3 rand = clamp(pow(1.-reflection,4.)*randn(seed*SQRT3),-1.,1.);
        	ray = normalize(refl + rand);
        }
        else
        {
            vec3 rand = random_sphere(seed*SQRT3);
            ray = normalize(norm + rand);
        }
      

        //color and illuminaition
        vec4 colp = map(p);
        fincol = fincol*clamp(colp.xyz,0.,1.);
        
        //add fractal glow
        finill += 5.*light_distr(p)*fincol;
        finill += vec3(1.)*exp(-300.*clamp(pow(abs(length(colp.xyz-vec3(0.2,0.3+0.01*cos(iTime),0.75+0.01*sin(iTime)))),2.),0.,1.))*fincol;
        finill += vec3(0.6)*exp(-300.*clamp(pow(abs(length(colp.xyz-vec3(0.3,0.8,0.3))),2.),0.,1.))*fincol;
        
        angle *= 1.15;
    }
    
    return finill;
}

// Function 906
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

// Function 907
float MarchAgainstPlanes( float t0, float dt, float wt, vec3 ro, vec3 rd, vec2 p )
{
    float t = t0;
    float res = 0.;
    vec3 pos;
    
    float firstWt = t0/dt;
    
    // first sample - blend in
    pos = ro + t * rd;
    res = max( res, firstWt*wt*smoothstep( 4., 2., length( pos.xz - p ) ) );
    t += dt;
    
    // interior samples
    for( int i = 1; i < SAMPLE_CNT-1; i++ )
    {
        pos = ro + t * rd;
        
        // render - in this case draw dots at each sample
        res = max( res, wt*smoothstep( 4., 2., length( pos.xz - p ) ) );
        
        t += dt;
    }
    
    // last sample - blend out
    pos = ro + t * rd;
    res = max( res,(1.-firstWt)*wt*smoothstep( 4., 2., length( pos.xz - p ) ) );
    t += dt;
    
    return res;
}

// Function 908
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

// Function 909
vec2 ray_march(in vec3 ro, in vec3 rd)
{
    float t = 0.0;
    float tmax = 120.0;
    const int smax = 200;
    float st = 0.0;
    
    for (int i = 0; i < smax; ++i)
    {
        vec3 p = ro + rd * t;
        float res = intersect(p);
        if (res < 0.001 || t > tmax)
            break;
        t += res * 0.5;
        st++;
    }
    
    if (t < tmax)
        return vec2(t, st / float(smax));
    else
        return vec2(-1.0, st / float(smax));
}

// Function 910
float ray_marching( vec3 origin, vec3 dir, float start, float end ) {
	float depth = start;
	for ( int i = 0; i < max_iterations; i++ ) {
		float dist = map( origin + dir * depth );
		if ( dist < stop_threshold ) {
			return depth;
		}
		depth += dist * 0.3;
		if ( depth >= end) {
			return end;
		}
	}
	return end;
}

// Function 911
vec3 marchVol2( in vec3 ro, in vec3 rd, in float t, in float mt )
{
    
    vec3 bpos = ro +rd*t;
    t += length(vec3(bpos.x,bpos.y,bpos.z))-1.;
    t -= dot(rd, vec3(0,1,0));
	vec4 rz = vec4(0);
	float tmt = t +1.5;
	for(int i=0; i<25; i++)
	{
		if(rz.a > 0.99)break;

		vec3 pos = ro + t*rd;
        float r = mapVol2( pos,.01 );
        vec3 lg = vec3(0.7,0.3,.2)*1.5 + 2.*vec3(1,1,1)*0.75;
        vec4 col = vec4(lg,r*r*r*3.);
        col *= smoothstep(t-0.25,t+0.2,mt);
        
        float z2 = length(vec3(pos.x,pos.y*.9,pos.z))-.9;
        col.a *= smoothstep(.7,1.7, 1.-map2(vec3(pos.x*1.1,pos.y*.4,pos.z*1.1)));
		col.rgb *= col.a;
		rz = rz + col*(1. - rz.a);
		
        t += z2*.015 + abs(.35-r)*0.09;
        if (t>mt || t > tmt)break;
        
	}
	
	return clamp(rz.rgb, 0.0, 1.0);
}

// Function 912
vec3 march(vec2 fc) {
    vec2 p = (-iResolution.xy + 2. * fc.xy) / iResolution.y;
    inside = 1.;
    vec3 camPos = vec3(0,0,2.5);
    vec3 rayDirection = normalize(vec3(p,-4));
    vec3 rayPosition = camPos;
    float distance = 0.;
    vec3 c = vec3(0);
    vec3 n;
    bool rf = false;
    vec2 m;
    float ss;

    for (int i = 0; i < 300; i++) {

        rayPosition += rayDirection * distance * .8;
        m = map(rayPosition);
        distance = m.x;

        if (abs(distance) < .005) {
            if (m.y == 0.) {
                n = calcNormal(rayPosition);
                rayDirection = refract(rayDirection, n, 1. / 2.222);
                rayPosition -= n * .001;
                inside *= -1.;
            } else {
                break;
            }
        }
    }

    if (m.y == 1.) {
        ss = rayPosition.y / 20. + .5;
        ss = saturate(ss);
        c = spectrum(ss/ 2.);
    }
    if (m.y == 2.) {
        n = calcNormal(rayPosition);
        c = n * .5 + .5;
        ss = dot(n, .5*vec3(.5,1,1));
        ss = saturate(ss);
        c = spectrum(ss / 2. - .8);
        c *= mix(1., saturate(ss), .8);
    }

    c = pow(c, vec3(1./2.2));
    return c;
}

// Function 913
float raymarch(vec3 ori, vec3 dir) {
 
    float t = 0.;
    for(int i = 0; i < MAX_ITERATIONS; i++) {
    	vec3  p = ori + dir * t;
        float d = dstScene(p);
        if(d < EPSILON || t > MAX_DISTANCE) {
            break;
        }
        t += d * .75;
    }
    return t;
    
}

// Function 914
float cubicstep( float x, float df0, float df1 ) { float b = 3.0 - df1 - 2.0 * df0; float a = 1.0 - df0 - b; return ( ( a * x + b ) * x + df0 ) * x; }

// Function 915
float RayMarch(vec3 ro, vec3 rd) {
	float dO=0.;
    
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        float dS = GetDist(p);
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }
    
    return dO;
}

// Function 916
vec3 raymarch(Ray ray)
{
    vec3 glowBase = vec3(1., .0, .3) * .002;
    vec3 glow = vec3(0.);
    float t = 0.;
    for (int i = 0; i < maxSteps && t <= maxDistance; i++)
    {
        vec3 currentPos = rayToPos(ray, t);
        Hit closestHit = map(currentPos);
        
        if (closestHit.t < hitThreshold)
        {
            vec3 normal = calcNormal(currentPos);
            vec3 color = closestHit.color * calcLight(currentPos, ray.dir, normal);
            return color + glow;
        }
        t += closestHit.t;
        glow += glowBase;
    }

    return vec3(0.01, 0.02, 0.03) + glow;
}

// Function 917
bool rayMarchSolids(vec3 startPos, vec3 direction, out float rayDist) {
    vec3 position = startPos ;
    bool intersected = false ;
    rayDist = 0.0 ;
    float delta = minPrimStepSize ;
    float precis = 0.0005 ;
    
    for (int i = 0 ; i < primNumSamples ; ++i) {
		if (isIntersectingRocket(position,precis,delta)) {
            return true ;
        } else {
            precis = 0.0005 * rayDist ;
		    rayDist += delta ;
            position = (rayDist)*direction + startPos ;
        }
    }
    
    return false ;
}

// Function 918
float march(vec3 eye, vec3 marchingDirection){
    const float precis = .001;
    float t = 0.0;
	float l = 0.0;
    for(int i=0; i<MAX_MARCHING_STEPS; i++){
	    vec3 p = eye + marchingDirection * t;
        float hit = world(p);
        if(hit < precis) return t;
        t += hit * .25;
    }
    return -1.;
}

// Function 919
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

// Function 920
float smoothstep1(float x) {
    return smoothstep(0., 1., x);
}

// Function 921
vec2 rayMarch(vec3 rayOrigin, vec3 rayDir, out vec3 cloudColor,vec2 fragCoord, int traceMode)
{
    float t = 0.0;
    vec3 pixelColor;
    vec3 skyColor = vec3(0.9);

    if ((iTime>20.0)&&(iTime<52.0))
    {
    	float adder=((iTime>=36.0)&&(iTime<44.0))?(fract(hash12(rayDir.xz))*0.1):0.0;
        for (int rayStep = 0; rayStep < NUM_STEPS; ++rayStep)
        {
            vec3 position = adder+0.05 * float(NUM_STEPS - rayStep) * rayDir;
            position.z+=iTime;
            float noiseScale=0.75;
            float posMulty=((iTime>=28.0)&&(iTime<36.0))?0.35:1.25;
            float signedCloudDistance = position.y+posMulty;
            for (int octaveIndex = 0; octaveIndex < NUM_NOISE_OCTAVES; ++octaveIndex)
            {
                position *= 2.0;
                noiseScale *= 2.0;
                signedCloudDistance -= RandomNumber(position) / noiseScale;
            }
            if (signedCloudDistance < 0.0)
                pixelColor+=(pixelColor-1.-signedCloudDistance * skyColor.zyx)*signedCloudDistance*0.15;
        }

        cloudColor=pixelColor;
    }
    
    for (int i = 0; i < 128; i++)
    {
        vec2 res = (iTime<20.0)?SDFCageScene(rayOrigin + rayDir * t,traceMode):
        	(iTime<52.0)?SDFscene(rayOrigin + rayDir * t):
        	(iTime<70.0)?SDFSeaScene(rayOrigin + rayDir * t):
        	SDFCageScene(rayOrigin + rayDir * t,traceMode);
        if (res[0] < (0.0001*t))
        {
            return vec2(t,res[1]);
        }
        t += res[0];
    }
     
    return vec2(-1.0,-1.0);
}

// Function 922
vec3 vmarch(in vec3 ro, in vec3 rd, in float j, in vec3 orig)
{   
    vec3 p = ro;
    vec2 r = vec2(0.);
    vec3 sum = vec3(0);
    float w = 0.;
    for( int i=0; i<VOLUMETRIC_STEPS; i++ )
    {
        r = map(p,j);
        p += rd*.03;
        float lp = length(p);
        
        vec3 col = sin(vec3(1.05,2.5,1.52)*3.94+r.y)*.85+0.4;
        col.rgb *= smoothstep(.0,.015,-r.x);
        col *= smoothstep(0.04,.2,abs(lp-1.1));
        col *= smoothstep(0.1,.34,lp);
        sum += abs(col)*5. * (1.2-noise(lp*2.+j*13.+time*5.)*1.1) / (log(distance(p,orig)-2.)+.75);
    }
    return sum;
}

// Function 923
vec2 Step (int sId)
{
  vec2 sv, c, del2c;
  sv = vec2 (mod (float (sId), gSize), floor (float (sId) / gSize));
  c = Loadv4 (sId).xy;
  del2c = Loadv4 (int (mod (sv.x + 1., gSize) + gSize * sv.y)).xy +
          Loadv4 (int (mod (sv.x - 1., gSize) + gSize * sv.y)).xy +
          Loadv4 (int (sv.x + gSize * mod (sv.y + 1., gSize))).xy +
          Loadv4 (int (sv.x + gSize * mod (sv.y - 1., gSize))).xy - 4. * c;
  c += delT * (difC * del2c - constKF.y * c +
     vec2 (constKF.y, - constKF.x * c.y) + c.x * c.y * c.y * vec2 (-1., 1.));
  return c;
}

// Function 924
vec4 raymarch( vec3 ro, vec3 rd, vec3 bgcol, ivec2 px )
{
	vec4 sum = vec4(0);
	float  t = 0., //.05*texelFetch( iChannel0, px&255, 0 ).x; // jitter ray start
          dt = 0.,
         den = 0., _den, lut;
    for(int i=0; i<550; i++) {
        vec3 pos = ro + t*rd;
        if( pos.y < -3. || pos.y > 3. || sum.a > .99 ) break;
                                    // --- compute deltaInt-density
        _den = den; den = map(pos); // raw density
        lut = LUTs( _den, den );    // shaped through transfer function
        
        if( lut > .0                // optim
       //   && abs(pos.x) > .5      // cut a slice 
          ) {                       // --- compute shading
            
#if 0                               // finite differences
            vec2 e = vec2(.3,0);
            vec3 n = normalize( vec3( map(pos+e.xyy) - den,
                                      map(pos+e.yxy) - den,
                                      map(pos+e.yyx) - den ) );
         // see also: centered tetrahedron difference: https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
            float dif = clamp( -dot(n, sundir), 0., 1.);
#else                               // directional difference https://www.iquilezles.org/www/articles/derivative/derivative.htm
         // float dif = clamp((lut - LUTs(_den, map(pos+.3*sundir)))/.6, 0., 1. ); // pseudo-diffuse using 1D finite difference in light direction 
            float dif = clamp((den - map(pos+.3*sundir))/.6, 0., 1. );             // variant: use raw density field to evaluate diffuse
#endif
          // dif = 1.;
           vec3  lin = vec3(.65,.7,.75)*1.4 + vec3(1,.6,.3)*dif,          // ambiant + diffuse
          //     col = lin * mix( vec3(1,.95,.8), vec3(.25,.3,.35), lut );// pseudo- shadowing with in-cloud depth ? 
          //     col = lin * clamp(1.-lut,0.,1.);
                 col = vec3(.2 + dif);
             col = mix( col , bgcol, 1.-exp(-.003*t*t) );   // fog
             sum += (1.-sum.a) * vec4(col,1)* (lut* dt*5.); // --- blend. Original was improperly just den*.4;
        }
        t += dt = .06; // max(.05,.02*t); // stepping
    }

    return sum; // clamp( sum, 0., 1. );
}

// Function 925
vec3 REFLECTION_RayMarch(vec3 ro, vec3 rd) {
	vec3 color = vec3(0.5);
    float st = 1.0/float(MAX_REFLECTION_STEPS);
    vec3 p = vec3(0.0);
    
    for (int i = 0; i < MAX_REFLECTION_STEPS; i++) {
    	p = ro + rd*st;
        vec4 mesh = MAP_REFLECTION_Scene(p, ro, 0);
        
        // Scene Color
        if (mesh.w <= SURFACE_DISTANCE) {
            mesh = MAP_REFLECTION_Scene(p, ro, 1);
            vec3 sceneColor = mesh.xyz;
        	
             if (st>=FOG_START) {
            	float nrmlz = st - FOG_START;
                nrmlz /= FOG_END - FOG_START;
                sceneColor = mix(sceneColor, FOG_COLOR, pow(nrmlz, 1.0));
            }
            
            color = sceneColor;
            break;
        }
        
        if (st >= MAX_RENDER_DISTANCE) {
        	break;
        }
        
        st += mesh.w;
    }
    
    // Sky(Background) Color
    if (st >= MAX_RENDER_DISTANCE) {
    	color = mix(SKY_DOWN_COLOR, SKY_UP_COLOR, pow(abs(p.y), 0.1));
    }
    
    return color;
}

// Function 926
bool MarchShadowRay( vec3 start, vec3 dir, out vec3 pos CACHEARG )
{
    // Same as MarchCameraRay except lower tolerances because artifacts will barely be noticeable,
    // and we don't care about hit pos being accurate.
    // Caller should check bounds because sometimes we want to call this when we already know we're inside the bounds.
    
    pos = start + dir * SHADOW_EPSILON;
    
    float prevMarchDist = SHADOW_EPSILON;
    float prevSurfaceDist = BlobDist( start CACHE );
    
    for ( int i = 0; i < MAX_SHADOW_RAYMARCH_STEPS; i++ )
    {
        float surfaceDist = BlobDist( pos CACHE );
        if ( surfaceDist <= EPSILON )
            return true;
        
        float gradientAlongRay = (prevSurfaceDist - surfaceDist) / prevMarchDist;
        float safeGradient = max( gradientAlongRay, MIN_GRADIENT_FOR_SHADOW_RAYS );
        
        float addDist = (surfaceDist + SHADOW_EPSILON) / safeGradient;
        prevMarchDist = addDist;
        
        prevSurfaceDist = surfaceDist;
        pos += dir * addDist;

        vec3 relPos = pos - BLOB_BOUNDING_CENTER;
        relPos *= BLOB_BOUNDING_SCALE;
        if ( dot( relPos, relPos ) > BLOB_BOUNDING_RADIUS_SQR )
            return false;
	}
    
    return true;
}

