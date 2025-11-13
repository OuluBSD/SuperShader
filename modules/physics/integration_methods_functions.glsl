// Reusable Integration Methods Physics Functions
// Automatically extracted from particle/physics simulation-related shaders

// Function 1
float IntegrateOuter(float x, float d)
{
    return x / sqrt(d*d + x*x) + 1.0;
}

// Function 2
vec4 VerletIntegral(in ivec2 iU){
	vec4 P = GetP(iU);
    if(iU.y == GridSize.y-1 ){
        vec2 mouse = (iMouse.xy*2.-R)/R.y;
        if(iMouse.x == 0. && iMouse.y == 0.)
            mouse = vec2(0.);
            
    	P.xy = vec2(iU.x+12,iU.y+5)*R1 + mouse;	
    }
    vec2 PreviousP = P.zw;
    vec2 CurrentP = Constraint(iU,P.xy);
    vec2 NextP = CurrentP + (CurrentP - PreviousP)*(1.-Friction) + Gravity*iTimeDelta*iTimeDelta/2.;
    
    PreviousP = CurrentP;
    
    return vec4(NextP,PreviousP);
}

// Function 3
void forwardEuler(inout vec4 current, in float h) {
  //Adding Gravity
  current.zw += g * h;

  // Forward Euler
  current.xy += current.zw * h;
}

// Function 4
void BodyIntegrate( inout Body body, float dT )
{
#ifdef ENABLE_GRAVITY_TOGGLE    
    if( !KeyIsToggled( KEY_G ) )
#endif // ENABLE_GRAVITY_TOGGLE        
    {
    	BodyApplyGravity( body, dT );
    }
    
    body.vMomentum += body.vForce * dT;
    body.vAngularMomentum += body.vTorque * dT;
    
    vec3 vVel = body.vMomentum / body.fMass;
    vec3 vAngVel = body.vAngularMomentum / body.fIT;

    body.vPos += vVel * dT;
    vec4 qAngDelta = QuatFromVec3( vAngVel * dT );
    body.qRot = QuatMul( qAngDelta, body.qRot );

    body.qRot = normalize( body.qRot );
}

// Function 5
float Integrate(float x, float dd)
{
    return x / sqrt(dd + x*x) + 1.0;
}

// Function 6
float integrate(vec2 xy) {
    float tx = 0.0;
    float ty = 0.0;
    int n = 2000;
    vec2 last_p = path(0.0);

    float wavelength = 0.02+0.02*iMouse.x/iResolution.x;

    for (int i = 0; i < 2000; ++i) {
        float t = float(i)/float(n);
        vec2 p = path(t);
        float d = hypot(xy-p);

        float dt = p.y-last_p.y;

        if (dt < 0.0 && cross2(p-last_p, xy-p) > 0.0) {
            float path_length = d-p.x;
            float s = 2.0*pi/wavelength*path_length;
            tx += cos(s)*dt/d;
            ty += sin(s)*dt/d;
        }

//        path_length = d+p.x;
//        s = -2.0*pi/wavelength*path_length;
//        tx += cos(s)*dt/d;
//        ty += sin(s)*dt/d;

        last_p = p;
    }
    return hypot(vec2(tx, ty));
}

// Function 7
vec3 IntegrateAcceleration(vec3 ro, vec3 rd, float mass)
{
    float proj = dot(ro, rd);
    vec3 dp = ro - proj * rd;
    
    //d = length(dp)  distance between star and ray 恒星到射线的距离
    //intergrate = G*M/d * x/sqrt(d*d+x*x)  star outer acceleration integration 外部射线垂直方向积分
    //limit(x/sqrt(d*d+x*x)) = +-1 function limit 积分函数极限

    float d = length(dp);
    if (d == 0.0) return vec3(0);
    float g = CONST_G * mass / d * Integrate(proj, d*d);
    return g / d * dp;
}

// Function 8
float integratePolygon(vec2 sa, vec2 sb)
{
    vec2 sd = sb - sa;
    float sum = 0.;
    int e = 0;
    
    vec2 pa = polygonCorner(0);
    
    for(int i = 1; i <= numPolygonCorners; ++i)
    {
        vec2 pb = polygonCorner(i);
        vec2 d = pb - pa;
        vec2 n = vec2(d.y, -d.x);
        float dotsdn = dot(sd, n);
        float t = dot(pa - sa, n) / dotsdn;
        float u = dot(sa + sd * t - pa, d);
        
        if(u > 0. && u <= dot(d, d))
        {
            if(t > 0. && t <= 1.)
                sum += t * sign(dotsdn);
            
            if(t > 1.)
                e ^= 1;
        }
        
        pa = pb;
    }
    
    if(e != 0)
        sum += 1.;
    
    return sum;
}

// Function 9
float Integrate(float x1, float x2, float dd)
{
    return x2 / sqrt(dd + x2*x2) - x1 / sqrt(dd + x1*x1);
}

// Function 10
float integrate_L1(float A, float B, float C, float D, float x) {
    return 0.5*((A*x + B)*abs(A*x + B)/A + (C*x + D)*abs(C*x + D)/C)*bias;
}

// Function 11
vec4 IntegrateRK4(vec4 state, float h)
{         
    vec4 k1 = h*dAdt_dVdt( state );
    vec4 k2 = h*dAdt_dVdt( state + k1/2.0 );
    vec4 k3 = h*dAdt_dVdt( state + k2/2.0 );
    vec4 k4 = h*dAdt_dVdt( state + k3 );

    state += (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
    
    return state;
}

// Function 12
float integrateCheckerboard(vec2 uv0, vec2 uv1)
{
  	vec2 rd = uv1 - uv0;
    
    vec2 dt = abs(vec2(1) / rd);
    vec2 t = (floor(uv0) + max(sign(rd), 0.) - uv0) * dt * sign(rd);
    int e = int(floor(uv0.x) + floor(uv0.y)) & 1;
    
    float mt = 0., pt, a = 0.;
    
    for(int i = 0; i < 8; ++i)
    {
        pt = mt;
        mt = min(t.x, t.y);
        
        if((i & 1) == e)
        	a += min(1., mt) - pt;

        if(mt > 1.)
            break;
        
        t += step(t, t.yx) * dt;
    }
    
    return a;
}

// Function 13
vec2 IntegrateBRDF(float NdotV, float roughness) {
    vec3 V;
    V.x = sqrt(1.0 - NdotV*NdotV);
    V.y = 0.0;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0;

    vec3 N = vec3(0.0, 0.0, 1.0);

    for(int i = 0; i < BRDF_SAMPLE_COUNT; ++i)
    {
        vec2 Xi = Hammersley(i, BRDF_SAMPLE_COUNT);
        vec3 H  = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if(NdotL > 0.0)
        {
            float G = GeometrySmith(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(BRDF_SAMPLE_COUNT);
    B /= float(BRDF_SAMPLE_COUNT);
    return vec2(A, B);
}

// Function 14
vec3 integrate( vec3 x0, float sign_, int steps )
{
  vec3 x = x0;
  float h = 1.0 / float(steps);   // stepsize
  for( int k=0; k < steps; k++ )  // Euler steps
  {
    x = x + h * sign_ * svf( x );
  }
  return (x-x0);
}

// Function 15
float euler( vec2 ic, float tmax, vec2 p )
{
    float h = tmax/float(MAXSTEPS); // stepsize

	float t = 0.0;
    vec2 a = ic, b = a;
    float d = dot( p - a, p - a );
    
    for( int i = 0; i < MAXSTEPS; i++ )
    {
        b = b + h * rhs(t, b);

        d = min( d, sdSegment( p, a, b ) );        
		t += h; a = b;
	}
    
	return sqrt(d);
}

// Function 16
Body Integrate3D(float mass, mat3 mInverseBodyInertiaTensor, 
                 Body obj, vec3 vCMForce, vec3 vTorque, float dt) {
    // compute auxiliary quantities
    mat3 mInverseWorldInertiaTensor = obj.mOrientation * mInverseBodyInertiaTensor * transpose(obj.mOrientation);
    vec3 vAngularVelocity = mInverseWorldInertiaTensor * obj.vAngularMomentum;
       		
    vCMForce -= obj.vCMVelocity * dKdl/dt; // Air friction
    vTorque -= vAngularVelocity * dKda/dt;
    
    obj.vCMVelocity	+= dt * vCMForce /mass;
    
    obj.mOrientation	 += skewSymmetric(vAngularVelocity) * obj.mOrientation  * dt;
    obj.vAngularMomentum += vTorque * dt;
    obj.mOrientation = orthonormalize(obj.mOrientation);

    // integrate primary quantities
    obj.vCMPosition	+= obj.vCMVelocity * dt;
    
    return obj;
}

// Function 17
void VerletIntegrate (inout vec4 point, in vec2 acceleration)
{
	vec2 currentPos = point.xy;
    vec2 lastPos = point.zw;

    vec2 newPos = currentPos + currentPos - lastPos + acceleration * c_tickDeltaTimeSq;
    
    point.xy = newPos;
    point.zw = currentPos;
}

// Function 18
mat3 fromEuler(vec3 ang) {
	vec2 a1 = vec2(sin(ang.x),cos(ang.x));
    vec2 a2 = vec2(sin(ang.y),cos(ang.y));
    vec2 a3 = vec2(sin(ang.z),cos(ang.z));
    mat3 m;
    m[0] = vec3(a1.y*a3.y+a1.x*a2.x*a3.x,a1.y*a2.x*a3.x+a3.y*a1.x,-a2.y*a3.x);
	m[1] = vec3(-a2.y*a1.x,a1.y*a2.y,a2.x);
	m[2] = vec3(a3.y*a1.x*a2.x+a1.y*a3.x,a1.x*a3.x-a1.y*a3.y*a2.x,a2.y*a3.y);
	return m;
}

// Function 19
float integratePolygon(vec2 sa, vec2 sb)
{
    vec2 sd = sb - sa;
    float sum = 0.;
    int e = 0;
    
    float ts[numPolygonCorners];

    int endPointInOut = 0;
    int startPointInOut = 0;

    vec2 pa = polygonCorner(0);
    for(int i = 1; i <= numPolygonCorners; ++i)
    {
        vec2 pb = polygonCorner(i);
        vec2 d = pb - pa;
        vec2 n = vec2(d.y, -d.x);
        
        float dotsdn = dot(sd, n);
        float t = dot(pa - sa, n) / dotsdn;
        float u = dot(sa + sd * t - pa, d);
        
        ts[i - 1] = -1.;
        
        if(u > 0. && u <= dot(d, d))
        {
            if(t > 0. && t <= 1.)
                ts[i - 1] = t;
            if(t > 1.)
                endPointInOut ^= 1;
            if(t < 0.)
                startPointInOut ^= 1;
        }
        
        pa = pb;
    }
    
    // The signs of the intersection distances to be added together
    // unfortunately are order-dependent. So all intersections are needed first
    // and then their relative order defines the sign.
    for(int i = 0; i < numPolygonCorners; ++i)
    {
        int e = 0;
        float ti = ts[i];
        
        for(int j = 0; j < numPolygonCorners; ++j)
        {
            float tj = ts[j];
            
            if(tj > 0. && tj < ti)
            	e ^= 1;
        }
        
        if(ts[i] > 0.)
            sum += e != 0 ? ts[i] : -ts[i];
    }
    
    if(startPointInOut != 0)
        sum = -sum;

    if(endPointInOut != 0)
        sum += 1.;
    
    return sum;
}

// Function 20
float inv_integrate_sqrt_poly2(float A, float B, float C, float t) {
    // there is to my knowledge no closed form solution.
    
    // the simplest root finding approach is bisection, but it 
    // needs many iterations to achieve sufficient precision.
    
    // a faster method is to use newton-raphson, which conveniently
    // requires the derivative which we started from.
    
    // our only hope for a closed form solution is the discovery 
    // of an invertible approximation of the integral.
    float x = 0.5;
    for (int k = 0; k < 4; ++k) {
        vec2 q = integrate_sqrt_poly2(A, B, C, x);
        float d = (t-q.x)/q.y;
        x = clamp(x + d, 0.0, 1.0);
    }
    return x;
}

// Function 21
vec2 Euler(vec2 posUV){
    vec2 AspectRatio = iResolution.xy/iResolution.y;
    return dt*GetVelocityUV(mod(posUV,vec2(1.0)))/AspectRatio;
}

// Function 22
vec2 integrate_sqrt_poly2(float A, float B, float C, float x) {
    float xx = x*x;
    float T = sqrt(A*xx + B*x + C);
    float P = 2.0*A*x + B;
    float Q = P * T;
    float R = (B*B - 4.0*A*C)*log(2.0*sqrt(A)*T + P);
    return vec2(Q/(4.0*A) - R/(8.0*pow(A,1.5)), T); 
}

// Function 23
float IntegrateOuter(float x1, float x2, float d)
{
    return x2 / sqrt(d*d + x2*x2) - x1 / sqrt(d*d + x1*x1);
}

// Function 24
vec2 euler(sampler2D g, vec2 p){
    vec2 middle    = getT(g,p).xy;   	
    vec2 up        = getT(g,p+vec2( 0, int(grid_level))).xy;
    vec2 down      = getT(g,p+vec2( 0,-int(grid_level))).xy;
    vec2 right     = getT(g,p+vec2( int(grid_level), 0)).xy;
    vec2 left      = getT(g,p+vec2(-int(grid_level), 0)).xy;
    vec2 upright   = getT(g,p+vec2( int(grid_level), int(grid_level))).xy;	
    vec2 upleft    = getT(g,p+vec2(-int(grid_level), int(grid_level))).xy;
    vec2 downright = getT(g,p+vec2( int(grid_level),-int(grid_level))).xy;
    vec2 downleft  = getT(g,p+vec2(-int(grid_level),-int(grid_level))).xy;
	
    vec2 gradDiv\
   	= vec2((0.25*(upleft.y + up.y)  - 0.25*(downleft.y + down.y)   + left.x   -middle.x\
                -(0.25*(upright.y + up.y) - 0.25*(downright.y+ down.y)  + middle.x     -right.x))/(pow(grid_level,2.)*dx*dx),\
                (up.y    - middle.y     + 0.25*(upleft.x + left.x)     -0.25*(upright.x + right.x)\
                -(middle.y  - down.y   + 0.25*(downleft.x + left.x)   -0.25*(downright.x+right.x)))/(pow(grid_level,2.)*dx*dx));
    
    //laplacian(u) = div(grad(u))
    //Gijs ninepoint stensil.
    vec2 laplacian\
    = (-8.*middle + up + left + right + down + upright + upleft + downright + downleft)/(3.*pow(grid_level,2.)*dx*dx);  
    
    //(uxdx + uydy)ux, (uxdx + uydy)uy
    vec2 convect\
   	= vec2((middle.x*(left.x - right.x) + middle.y*(up.x - down.x))/(2.*grid_level*dx),\
               (middle.x*(left.y - right.y) + middle.y*(up.y - down.y))/(2.*grid_level*dx));
    
    //"Navier-Stokes"
    return  middle.xy + dt*(viscosity*laplacian + nonlin_strength*convect + 1.0/epsilon*gradDiv);
}

// Function 25
void BodyIntegrate( inout Body body, float dT )
{


    body.vMomentum += body.vForce * dT;
    body.vAngularMomentum += body.vTorque * dT;

    vec3 vVel = body.vMomentum / body.fMass;
    vec3 vAngVel = body.vAngularMomentum / body.fIT;

    body.vPos += vVel * dT;
    vec4 qAngDelta = QuatFromVec3( vAngVel * dT );
    body.qRot = multQuat( qAngDelta, body.qRot );

    body.qRot = normalize( body.qRot );
}

// Function 26
void Integrate(in vec3 F, in vec3 T, inout mat4 s) {
	vec4 q = s[0];
    vec3 x = s[1].xyz;
    vec3 P = s[2].xyz;
    vec3 L = s[3].xyz;

    vec3 v = P*invMass;
    mat3 R = quatMat3Cast(q);
    mat3 invI = R*invI_lamp*transpose(R);
    vec3 omega = invI*L;
    
    x += v*iTimeDelta;
    q += 0.5 * quatMult(vec4(omega*iTimeDelta, 0.0), q);
    q = normalize(q);

    CollisionImpulse(x, q, v, omega, invI, P, L);
    
    P += F*iTimeDelta;
	L += T*iTimeDelta;
    
    s = mat4(q, vec4(x, 1.0), vec4(P, 0.0), vec4(L, 0.0));
}

// Function 27
vec3 IntegrateSurface(vec3 col, vec3 pos, vec3 n, float matId, vec3 rayDir, SceneSetup setup)
{ 
     PBRMat mat;
     
     GetMaterial(matId, pos, setup, mat, false);
     
     mat.albedo.rgb *= mat.albedo.rgb; // Convert albedo to linear space

     vec3 ambient = SkyDomeBlurry(n, 5.0);
     ambient *= mat.occlusion * 0.5;
    
     col = mix(col, mat.albedo.rgb * ambient, mat.albedo.a);
     
     // Fresnel
     float fresnel = pow(1.0 - sat(dot(n, -rayDir)), 1.0);

     // Add both light contributions
	 vec3 key_LightPos = vec3(10.0, 24.0, -13.0);
     col += PBRLight(pos, n, rayDir, mat, key_LightPos, vec3(1.0), 1000.0, fresnel, setup, true);
                  
     vec3 fill_LightPos = vec3(-20.0, 15.0, 20.0);
     col += PBRLight(pos, n, rayDir, mat, fill_LightPos, vec3(1.0), 1000.0, fresnel, setup, false);

	return col;
}

// Function 28
mat3  euler(float h, float p, float r){float a=sin(h),b=sin(p),c=sin(r),d=cos(h),e=cos(p),f=cos(r);return mat3(f*e,c*e,-b,f*b*a-c*d,f*d+c*b*a,e*a,c*a+f*b*d,c*b*d-f*a,e*d);}

// Function 29
void integrateSpring(inout Point p1, inout Point p2, in float rest_length, in float stepsize)
{
    float dist = distance(p1.pos, p2.pos);
    
    float factor = clamp((rest_length - dist)/dist, -1., 1.);
    
    vec3 dir = (p1.pos - p2.pos) * factor * stepsize;
    
    p1.vel += dir;
    p2.vel -= dir;
}

// Function 30
float integrateSquare(vec2 pa, vec2 pb)
{
    vec2 d = pb - pa, sd = sign(d);
    
    vec2 t0 = (-sd - pa) / d;
    vec2 t1 = (+sd - pa) / d;
    
    vec2 i = clamp(vec2(max(t0.x, t0.y), min(t1.x, t1.y)), 0., 1.);
    
    return max(0., i.y - i.x);
}

// Function 31
vec2 IntegrateBRDF(float roughness, float NdotV) {
    vec3 V;
    V.x = sqrt(1.0 - NdotV*NdotV);
    V.y = 0.0;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0;

    const int SAMPLE_COUNT = 128;

    vec3 N = vec3(0.0, 0.0, 1.0);
    vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 TangentX = normalize(cross(UpVector, N));
    vec3 TangentY = cross(N, TangentX);

    for(int i = 0; i < SAMPLE_COUNT; ++i)  {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 HTangent = ImportanceSampleGGX(Xi, roughness);
        
        vec3 H = normalize(HTangent.x * TangentX + HTangent.y * TangentY + HTangent.z * N);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if(NdotL > 0.0) {
            float G = GeometryGGX_Smith(NdotV, NdotL, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return vec2(A, B);
}

// Function 32
vec3 Integrate(vec3 cur, float dt)
{
	vec3 next = vec3(0);
    
    next.x = O * (cur.y - cur.x);
    next.y = cur.x * (P - cur.z) - cur.y;
    next.z = cur.x*cur.y - B*cur.z;
    
    return cur + next * dt;
}

// Function 33
void BodyIntegrate(inout Body body, float dt ) {
    body.vMomentum += body.vForce * dt; 
    body.vAngularMomentum += body.vTorque*dt;
    
    vec3 vVel = body.vMomentum * body.invM;
    vec3 vAngVel = getAngularVelocityWorld(body);
    
    if (length(vVel) > MAX_VEL) {
        vVel = normalize(vVel)*MAX_VEL;
    }
    if (length(vAngVel) > MAX_AVEL) {
        vAngVel = normalize(vAngVel)*MAX_AVEL;
    }
    body.vPos += vVel * dt;
    vec4 qAngDelta = QuatFromVec3(vAngVel * dt);
    body.qRot = normalize(QuatMul(qAngDelta, body.qRot));
    body.mRot = QuatToMat3(body.qRot);  // update rot
    body.vMomentum *= pow(.91,10.*DT);
    body.vAngularMomentum *= pow(.93,10.*DT); 
}

// Function 34
void integrateCameraRig(inout CameraRig rig, in float steps) {
    float stepsize = steps;
	integrateSpring(rig.p1, rig.p2, cam_size, stepsize);
    integrateSpring(rig.p1, rig.p3, cam_size, stepsize);
    integrateSpring(rig.p1, rig.p4, cam_size, stepsize);
    integrateSpring(rig.p1, rig.p5, cam_size, stepsize);
    
    integrateSpring(rig.p3, rig.p2, sqrt(2.) * cam_size, stepsize);
    integrateSpring(rig.p3, rig.p4, sqrt(2.) * cam_size, stepsize);
    integrateSpring(rig.p3, rig.p5, sqrt(2.) * cam_size, stepsize);
    
    integrateSpring(rig.p5, rig.p2, sqrt(2.) * cam_size, stepsize);
    integrateSpring(rig.p5, rig.p4, sqrt(2.) * cam_size, stepsize);
    
    float de = DE(rig.p1.pos);
    rig.p5.vel += stepsize * DE_norm(rig.p5.pos, 0.1) * smoothstep(.7, .0, de) * thrust_along_normal;
    
    vec3 thrust = normalize(rig.p5.pos - rig.p1.pos) * constant_thrust * steps;
    rig.p1.vel += thrust;
    rig.p2.vel += thrust;
    rig.p3.vel += thrust;
    rig.p4.vel += thrust;
    rig.p5.vel += thrust;
    
    stepsize = steps;
    integratePoint(rig.p1, stepsize);
    integratePoint(rig.p2, stepsize);
    integratePoint(rig.p3, stepsize);
    integratePoint(rig.p4, stepsize);
    integratePoint(rig.p5, stepsize);
}

// Function 35
float inv_integrate_sqrt_poly2_fast(float B, float C, float t) {
    // there is to my knowledge no closed form solution.
    
    // the simplest root finding approach is bisection, but it 
    // needs many iterations to achieve sufficient precision.
    
    // a faster method is to use newton-raphson, which conveniently
    // requires the derivative which we started from.
    
    // our only hope for a closed form solution is the discovery 
    // of an invertible approximation of the integral.
    float x = 0.5;
    for (int k = 0; k < 4; ++k) {
        vec2 q = integrate_sqrt_poly2_fast(B, C, x);
        float d = (t-q.x)/q.y;
        x = clamp(x + d, 0.0, 1.0);
    }
    return x;
}

// Function 36
void integrateVolumetricFog(in vec3 p, 
                           in vec3 V,
                           in float density,
                           in float d,
                           inout float transmittance, 
                           inout vec3 inscatteredLight,
                           in vec2 fragCoord)
{
    // --- sample a random position on the area light
    transmittance *= exp(-density * d);
    float g = 0.2;
    float u = rand(rand(fragCoord.x * p.x + fragCoord.y + d * 1.5) + 46984.4363);
    float v = rand(rand(fragCoord.y * p.y + fragCoord.x + d * 2.5) + 3428.532546);
    vec3 lightPos;
    vec3 lightCol;
    sampleAreaLight(vec2(u, v), lightPos, lightCol);
    vec3 L = (lightPos - p);
    float G = (dot(normalize(L), transpose(getRotation()) * AreaLightNormal)) / dot(L, L);
    float areaPdf = 1.0 / (AreaLightSize.x * AreaLightSize.y);
    float shadow = rayMarchToAreaLight(p, lightPos);
    float phaseHG = (1.0 / (4.0 * 3.14)) * ((1.0 - g * g) / (pow((1.0 + g * g - 2.0 * g * max(dot(normalize(L), V), 0.0)), 3.0 / 2.0)));
    inscatteredLight += density * transmittance * lightCol * G * phaseHG * d * (1.0 / areaPdf) * shadow;
}

// Function 37
vec3 IntegrateAcceleration(vec3 ro, vec3 rd, float mass)
{
    float proj = dot(ro, rd);
    vec3 dp = ro - proj * rd;
    
    //l = length(dp)  distance between star and ray 恒星到射线的距离
    //inner = G*M/V*l * x  star inner acceleration integration 内部射线垂直方向积分
    //outer = G*M/l * x/sqrt(l*l+x*x)  star outer acceleration integration 外部射线垂直方向积分
    //limit(x/sqrt(l*l+x*x)) = +-1.0 function limit 积分函数极限

    float l = length(dp);
    if (l == 0.0) return vec3(0);
    float g = CONST_G * mass / l * IntegrateOuter(proj, l);
    float r = GetStarRadius(mass);
#ifdef INTERGRATE_STAR_CENTER
    if (l < r)
    {
        float in0 = -sqrt(r * r - l * l);
        if (proj > in0)
        {
            float in1 = min(proj, -in0);
            g -= CONST_G * mass / d * IntegrateOuter(in0, in1, l);
            //inner
        #ifdef RIGID_STAR
            g += CONST_G * mass / (r * r * r) * l * (in1 - in0);
        #else
            g += CONST_G * mass / (r * d) * (in1 - in0);
        #endif
        }
    }
#endif
    return g / l * dp;
}

// Function 38
vec2 integrate_sqrt_poly2_fast(float B, float C, float x) {
    float xx = x*x;
    float T = sqrt(xx + B*x + C);
    float P = 2.0*x + B;
    float Q = P * T;
    float R = (B*B - 4.0*C)*log(2.0*T + P);
    return vec2((Q - R*0.5)*0.25, T);
}

// Function 39
vec2 Euler(vec2 posUV){
    vec2 AspectRatio = iResolution.xy/iResolution.y;
    return dt*GetVelocityUV(posUV)/AspectRatio;
}

// Function 40
Quaternion euler_to_quat(vec3 angles)
{
    Quaternion
        q0 = axis_angle(vec3(0, 0, 1), angles.x),
        q1 = axis_angle(vec3(1, 0, 0), angles.y),
        q2 = axis_angle(vec3(0, 1, 0),-angles.z);
    return mul(q0, mul(q1, q2));
}

// Function 41
void euler(inout vec4 current, in float timeDelta) {
  current.zw += g * timeDelta;
  current.xy += current.zw * timeDelta;
}

// Function 42
mat3 EulerToMat(vec3 p){vec3 s=sin(p),c=cos(p);
 return mat3(c.z*c.x,-c.z*s.x,c.z*s.x*s.y+s.z*c.y
 ,s.x ,c.x*c.y ,-c.x*s.y
 ,-s.z*c.x ,s.z*s.x*c.y+c.z*s.y ,-s.z*s.x*s.y+c.z*c.y);}

// Function 43
vec4 Euler2AxisAngle(vec3 p){p*=.5;//.xyz=heading,altitude,bank
 vec3 s=sin(p),c=cos(p);
 vec4 r;
 r.w=2.*acos(mulC(c)-mulC(s));
 r.x=s.x*s.y*c.z+c.x*c.y*s.z
// x=s1  s2  c3 +c1  c2  s3
 r.y=s.x*c.y*c.z+c.x*s.y*s.z
// y=s1  c2  c3 +c1  s2  s3
 r.z=c.x*s.y*c.z-s.x*c.y*c.z
// z=c1  s2  c3 -s1  c2  s3
 return r;}

// Function 44
vec2 eulerInte(vec2 p) {
    vec2 v = texture(iChannel0, p / iResolution.xy).xy;
    return p - v * iTimeDelta;
}

// Function 45
void integratePoint(inout Point p, in float stepsize) {
	// advance by velocity
    p.pos += stepsize * p.vel;
    // air friction
    p.vel *= 1. - stepsize * friction_air;
    
    float de = DE(p.pos);
    
    //p.vel += stepsize * DE_norm(p.pos, 0.1) * smoothstep(.2, .0, de) * thrust_along_normal;
    
    if (de < 0.) {
    	// move point to surface
        for (int i=0; i<3; ++i) {
        	vec3 n = DE_norm(p.pos, 0.01);
        	p.pos += n * -de * .7;
        }
        vec3 n = DE_norm(p.pos, 0.01);
        // reflect at surface
        n = DE_norm(p.pos, 0.01);
        p.vel = reflect(p.vel, n);
        p.vel *= 1. - stepsize * friction_surface;
    }
}

// Function 46
float inv_integrate_L1(float A, float B, float C, float D, float t) {
    // even for the L1 form there is no closed form solution.
    // see inv_integrate_sqrt_poly2 for more information
    float x = 0.5;
    for (int k = 0; k < 4; ++k) {
        float q = integrate_L1(A, B, C, D, x);
        float d = (t-q)/(bias*(abs(A*x + B) + abs(C*x + D)));
        x = clamp(x + d, 0.0, 1.0);
    }
    return x;
}

// Function 47
void update_trajectory_euler(in int body, out vec4 XYZW, out vec3 velXYZ)
{
    vec4 bodyXYZW;
    vec3 bodyVelXyz;
    getBodyPosVel(body, bodyXYZW, bodyVelXyz);
    
    vec3 accelVec = calculate_gravity_accel(body, bodyXYZW, 0.0);
    
    velXYZ = bodyVelXyz + GRAVITY_COEFF * accelVec;
    XYZW = bodyXYZW + UPDATE_STEP * vec4(velXYZ.xyz, 0.0);
}

// Function 48
vec4 Integrate(vec4 body, vec2 accel, float delta)
{
    return vec4(2.0*body.xy - body.zw + accel * delta*delta, body.xy);
}

// Function 49
vec4 integrate( in vec4 sum, in float dif, in float den, in vec3 bgcol, in float t )
{
    // lighting
    vec3 lin = vec3(0.9,0.95,1.0) + 0.5*vec3(0.7, 0.5, 0.3)*dif * smoothstep(-0.3, 0.3, v3sunDir.y);
    vec4 col = vec4( mix( 1.15*vec3(1.0,0.95,0.8), vec3(0.65), den ), den );
    col.xyz *= lin;
    //col.xyz = mix( col.xyz, bgcol, 1.0-exp(-0.003*t*t) );
    // front to back blending    
    col.a *= 0.4;
    col.rgb *= col.a;
    return sum + col*(1.0-sum.a);
}

// Function 50
void integrateVolumetricFogFromSampledPosition(in vec3 p, 
                           in vec3 V,
                           in float density,
                           in float d,
                           inout vec3 inscatteredLight,
                           in vec3 lightPos,
                           in vec3 lightCol,
                           in float xPdf)
{
    // same integration as above, but position on the ray is given by equi-angular sampling
    float trans = exp(-density * d);
    float g = 0.2;
    vec3 L = (lightPos - p);
    float G = (dot(normalize(L), transpose(getRotation()) * AreaLightNormal)) / dot(L, L);
    float areaPdf = 1.0 / (AreaLightSize.x * AreaLightSize.y);
    float shadow = rayMarchToAreaLight(p, lightPos);
    float phaseHG = (1.0 / (4.0 * 3.14)) * ((1.0 - g * g) / (pow((1.0 + g * g - 2.0 * g * max(dot(normalize(L), V), 0.0)), 3.0 / 2.0)));
    inscatteredLight += density * trans * lightCol * G * phaseHG  * (1.0 / areaPdf) * (1.0 / xPdf) * shadow;
}

