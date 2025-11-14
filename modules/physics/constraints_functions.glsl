// Reusable Constraints Physics Functions
// Automatically extracted from particle/physics simulation-related shaders

// Function 1
vec3 bezier_solve(float a, float b, float c) {
    float p = b - a*a / 3.0, p3 = p*p*p;
    float q = a * (2.0*a*a - 9.0*b) / 27.0 + c;
    float d = q*q + 4.0*p3 / 27.0;
    float offset = -a / 3.0;
    if(d >= 0.0) { 
        float z = sqrt(d);
        vec2 x = (vec2(z, -z) - q) / 2.0;
        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        return vec3(offset + uv.x + uv.y);
    }
    float v = acos(-sqrt(-27.0 / p3) * q / 2.0) / 3.0;
    float m = cos(v), n = sin(v)*1.732050808;
    return vec3(m + m, -n - m, n - m) * sqrt(-p / 3.0) + offset;
}

// Function 2
vec3 resolveAdjacentCorner(in Cam perspectiveCam, vec3 P1, vec2 p1_resolved, vec2 p2_adjacent, vec2 parallel_a, vec2 parallel_b)
{
    //screen space intersection (vanishing point on the projection plane)
    vec2 ssIntersec = lineLineIntersection(p1_resolved,p2_adjacent,parallel_a,parallel_b);
    //Vanishing point direction, from camera, in world space.
    vec3 dirVanishingPoint = ray(ssIntersec, perspectiveCam);
    vec3 p1_to_p2 = dirVanishingPoint; //Since vanishing point is at "infinity", p1_to_p2 == dirVanishingPoint
    vec3 r2 = ray(p2_adjacent, perspectiveCam);//Ray from camera to p2, in world space
    
    //<Line3D intersection : where p1_to_p2 crosses r2>
    //(Note : this could probably be made simpler with a proper 3D line intersection formula)
    //Find (rb,p1_to_p2) intersection:
    vec3 n_cam_p1_p2 = cross(p1_to_p2,r2); //normal to the triangle formed by point p1, point p2 and the camera origin
    vec3 n_plane_p2 = cross(n_cam_p1_p2,r2); //normal to the plane which is crossed by line p1-p2 at point p2
    float t = rayPlaneIntersec(P1,p1_to_p2,perspectiveCam.o,n_plane_p2);
    vec3 p2_ws = P1+t*p1_to_p2;
    //</Line3D intersection>
    return p2_ws;
}

// Function 3
v2 SolveQuad(v2 a){v0 e=-a.x/3.;v0 p=a.y+a.x*e,t=p*p*p,
 q=-(2.*a.x*a.x-9.*a.y)*e/9.+a.z,d=q*q+4.*t/27.;if(d>.0){v1 x=(v1(1,-1)*sqrt(d)-q)*.5;
 return v2(suv(sign(x)*pow(abs(x),v1(1./3.)))+e);}v1 m=cs(acos(-sqrt(-27./t)*q*.5)/3.)
  *v1(1,sqrt(3.));return v2(m.x+m.x,-suv(m),m.y-m.x)*sqrt(-p/3.)+e;}

// Function 4
v2 SolveQuad(v2 a){float e=-a.x/3.;v0 p=a.y+a.x*e,t=p*p*p,
 q=-(2.*a.x*a.x-9.*a.y)*e/9.+a.z,d=q*q+4.*t/27.;if(d>.0){v1 x=(v1(1,-1)*sqrt(d)-q)*.5;
 return v2(suv(sign(x)*pow(abs(x),v1(1./3.)))+e);}v1 m=cs(acos(-sqrt(-27./t)*q*.5)/3.)
  *v1(1,sqrt(3.));return v2(m.x+m.x,-suv(m),m.y-m.x)*sqrt(-p/3.)+e;}

// Function 5
bool solveQuadratic(float a, float b, float c, out float t0, out float t1){
	float discrim = b * b - 4.0 * a * c;
	if (discrim < 0.0) return false;
	float rootDiscrim = sqrt(discrim);
	float q = (b > 0.0) ? -0.5 * (b + rootDiscrim) : -0.5 * (b - rootDiscrim); 
	t1 = q / a; 
	t0 = c / q;
	return true;
}

// Function 6
vec3 bezier_solve(float a, float b, float c) {
    float p = b - a*a / 3.0, p3 = p*p*p;
    float q = a * (2.0*a*a - 9.0*b) / 27.0 + c;
    float d = q*q + 4.0*p3 / 27.0;
    float offset = -a / 3.0;
    if(d >= 0.0) {
        float z = sqrt(d);
        vec2 x = (vec2(z, -z) - q) / 2.0;
        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        return vec3(offset + uv.x + uv.y);
    }
    float v = acos(-sqrt(-27.0 / p3) * q / 2.0) / 3.0;
    float m = cos(v), n = sin(v)*1.732050808;
    return vec3(m + m, -n - m, n - m) * sqrt(-p / 3.0) + offset;
}

// Function 7
void solve(sampler2D data, inout Body b0, int id, vec2 ires) {
    vec2 displace = vec2(0.0);
    
    // collision detection
    for(int i = 0; i < NUM_OBJECTS; i++) {
        if(i == id) continue;
        
        Body b1 = getBody(data, ires, i);
        displace += collisionWithBody(b0,b1);
    }
    
    // walls
    displace += collisionWithPlane(b0, vec3(0.0,1.0,FRAME_SIZE.y));
    displace += collisionWithPlane(b0, vec3(0.0,-1.0,FRAME_SIZE.y));
    displace += collisionWithPlane(b0, vec3(1.0,0.0,FRAME_SIZE.x));
    displace += collisionWithPlane(b0, vec3(-1.0,-.0,FRAME_SIZE.x));

    b0.pos += displace;
}

// Function 8
vec2 solveCubic2b(vec1 a,vec1 b,vec1 c//https://www.shadertoy.com/view/XtdyDn
){vec2 p=vec2(b-a*a/3.,a)
 ;vec1 q=a*(2.*a*a-9.*b)/27.+c
 ,s=p.x*p.x*p.x
 ;c=q*q+4.*s/27.//determinant seperates cases where a root repeats
 ;if(q*q+4.*s/27.>0.)return root23(vec2(a),b,c)//both return values are identical
 ;v0 v=acos(-sqrt(-27./s)*q*.5)/3.,m=cos(v),n=sin(v)*sqrt(3.);p/=3.//...does not care for 3rd (middle) root, intended as subroutine for bezier/parabola
 ;return vec2(m+m,-n-m)*sqrt(-p.x)-p.y;}

// Function 9
vec2 Constraint(in ivec2 iU,in vec2 localP){
    //Structure Constraint
    if(iU.x > 0){
        localP += Simulation(localP,iU,ivec2(-1, 0),K,R1);
    }
    if(iU.y > 0){
        localP += Simulation(localP,iU,ivec2( 0,-1),K,R1);
    }
    if(iU.x < GridSize.x - 1){
    	localP += Simulation(localP,iU,ivec2( 1, 0),K,R1);
    }
    if(iU.y < GridSize.y - 1){
    	localP += Simulation(localP,iU,ivec2( 0, 1),K,R1);
    }
    //Shear Constraint
    if(iU.x>0 && iU.y>0){
    	localP += Simulation(localP,iU,ivec2(-1,-1),K,R2);
    }
    if(iU.x>0 && iU.y<GridSize.y - 1){
    	localP += Simulation(localP,iU,ivec2(-1, 1),K,R2);
    }
    if(iU.y>0 && iU.x<GridSize.x - 1){
    	localP += Simulation(localP,iU,ivec2( 1,-1),K,R2);
    }
    if(iU.x<GridSize.x - 1 && iU.y<GridSize.y - 1){
    	localP += Simulation(localP,iU,ivec2( 1, 1),K,R2);
    }
	//Blend Constraint
    /**/
    if(iU.x>1){
    	localP += Simulation(localP,iU,ivec2(-2, 0),K,R3);
    }
    if(iU.y>1){
    	localP += Simulation(localP,iU,ivec2( 0,-2),K,R3);
    }
    if(iU.x<GridSize.x-2){
    	localP += Simulation(localP,iU,ivec2( 2, 0),K,R3);
    }
    if(iU.y<GridSize.y-2){
    	localP += Simulation(localP,iU,ivec2( 0, 2),K,R3);
    }
    /*
    if(iU.x>1 && iU.y>1){
    	localP += Simulation(localP,iU,ivec2(-2,-2),K,R4);
    }
    if(iU.x>1 && iU.y<GridSize.y-2){
    	localP += Simulation(localP,iU,ivec2(-2, 2),K,R4);
    }
    if(iU.y>1 && iU.x<GridSize.x-2){
    	localP += Simulation(localP,iU,ivec2( 2,-2),K,R4);
    }
    if(iU.x<GridSize.x-2 && iU.y<GridSize.y-2){
    	localP += Simulation(localP,iU,ivec2( 2, 2),K,R4);
    }
    
    /**/
    return localP;
}

// Function 10
bool solveQuadraticIntersection(float a, float b, float c, out float t)
{
    if(abs(a) < eps32)
    {
        t = -c / b;
        return t > 0.0;
    }

	float discriminant = b * b - 4.0 * a * c;

    if(abs(discriminant) < eps32)
    {
        t = - b / (2.0 * a);
        return true;
    }
    else if(discriminant < 0.0)
    {
        return false;
    }
    else
	{
        float sqrtd = sqrt(discriminant);

        float t0 = (-b + sqrtd) / (2.0 * a);
        float t1 = (-b - sqrtd) / (2.0 * a);

        if(t1 < t0)
        {
            float tt = t0;
            t0 = t1;
            t1 = tt;
        }

        if(t0 > 0.0)
        {
            t = t0;
            return true;
        }

        if(t1 > 0.0)
        {
            t = t1;
            return true;
        }

        return false;
	}
}

// Function 11
void resolve_axial_intersection(vec3 campos, vec3 rcp_delta, inout Intersection result, int best_index)
{
    if (best_index == -1)
        return;

    vec3 mins = get_axial_point(best_index);
    vec3 maxs = get_axial_point(best_index+1);
    vec3 t0 = (mins - campos) * rcp_delta;
    vec3 t1 = (maxs - campos) * rcp_delta;
    vec3 tmin = min(t0, t1);
    float t = max_component(tmin);
    int axis =
        (t == tmin.x) ? 0 :
    	(t == tmin.y) ? 1 :
    	2;
    bool side = rcp_delta[axis] > 0.;
    int face = (axis << 1) + int(side);

    result.plane		= (best_index + (best_index<<1)) + face;
    result.material		= get_axial_brush_material(best_index>>1, face);
    result.normal		= vec3(0);
    result.normal[axis]	= side ? -1. : 1.;
    result.uv_axis		= axis;
    result.mips			= fwidth(float(result.plane)) < 1e-4;
}

// Function 12
void resolvePerspective(in Cam perspectiveCam, in screenSpaceQuad ssQuad, out worldSpaceQuad wsQuad)
{
    vec3 ra = ray(ssQuad.a, perspectiveCam); //Find the direction of the ray passing by point a in screen space.
	                                      //For the sake of simplicity, screenspace [uv.x,uv.y] = worldspace [x,y]. Z = depth.
    //Let's place point a in an arbitrary position along the ray ra. 
    //It does not matter at which distance exactly, as it is the relationship between
    //the corners that is important. The first corner distance simply defines the scaling of the 3D scene.
    wsQuad.a = perspectiveCam.o + 4.5*ra; //5.5 = arbitrary scaling. Projective geometry does not preserve world space scaling.
    wsQuad.b = resolveAdjacentCorner(perspectiveCam, wsQuad.a, ssQuad.a, ssQuad.b, ssQuad.c, ssQuad.d);
    wsQuad.c = resolveAdjacentCorner(perspectiveCam, wsQuad.b, ssQuad.b, ssQuad.c, ssQuad.a, ssQuad.d);
    wsQuad.d = resolveAdjacentCorner(perspectiveCam, wsQuad.a, ssQuad.a, ssQuad.d, ssQuad.b, ssQuad.c);
}

// Function 13
bool solveQuadratic(float A, float B, float C, out float t0, out float t1){
	float discrim = B*B-4.0*A*C;
	if ( discrim < 0.0 )
        	return false;
	float rootDiscrim = sqrt(discrim);
	float Q = (B > 0.0) ? -0.5 * (B + rootDiscrim) : -0.5 * (B - rootDiscrim); 
	float t_0 = Q / A; 
	float t_1 = C / Q;
	t0 = min( t_0, t_1 );
	t1 = max( t_0, t_1 );
	return true;
}

// Function 14
vec2 solveCubic2b(vec3 a){return solveCubic2b(a.x,a.y,a.z);}

// Function 15
float solve( vec2 p ) {
	float time = iTime + time_offset;

	float r = length( p );
	float t = atan( p.y, p.x ) - time * 0.1;
	
	float v = 1000.0;
	for ( int i = 0; i < 32; i++ ) {
		if ( t > time * 1.5 ) {
			continue;
		}
		v = min( v, abs( function( r, t ) ) );
		t += PI * 2.0;
	}
	return v;
}

// Function 16
vec4 solveConstraints(ivec2 id, vec4 p)
{
    // TODO vector compare
    if (id.x > 0   )  p = react(p, id + ivec2(-1, 0), spacing);
    if (id.x < limx)  p = react(p, id + ivec2( 1, 0), spacing);
    if (id.y > 0   )  p = react(p, id + ivec2( 0,-1), spacing);
    if (id.y < limy)  p = react(p, id + ivec2( 0, 1), spacing);
	float r2s = spacing * sqrt(2.);
    if (id.x > 0    && id.y > 0   )  p = react(p, id + ivec2(-1, -1), r2s);
    if (id.x > 0    && id.y < limy)  p = react(p, id + ivec2(-1,  1), r2s);
    if (id.x < limx && id.y > 0   )  p = react(p, id + ivec2( 1, -1), r2s);
    if (id.x < limx && id.y < limy)  p = react(p, id + ivec2( 1,  1), r2s);

    return p;
}

// Function 17
vec3 solveCubic(float a, float b, float c){
    float p = b - a*a / 3.0, p3 = p*p*p;
    float q = a * (2.0*a*a - 9.0*b) / 27.0 + c;
    float d = q*q + 4.0*p3 / 27.0;
    float offset = -a / 3.0;
    if(d >= 0.0) { 
        float z = sqrt(d);
        vec2 x = (vec2(z, -z) - q) / 2.0;
        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        return vec3(offset + uv.x + uv.y);
    }
    float v = acos(-sqrt(-27.0 / p3) * q / 2.0) / 3.0;
    float m = cos(v), n = sin(v)*1.732050808;
    return vec3(m + m, -n - m, n - m) * sqrt(-p / 3.0) + offset;
}

// Function 18
vec2 solveJoint(in vec2 a, in vec2 b, in float ra, in float rb, in float config) {
    vec2 ba = b - a;
    float d = dot(ba, ba), l = sqrt(d);
    float q = (d + ra * ra - rb * rb) / (2.0 * ra * l);
    return a + (ba * q + vec2(-ba.y, ba.x) * sqrt(1.0 - q * q) * config) * ra / l;
}

// Function 19
float solve_black_body_fraction_below_wavelength(float wavelength, float temperature){ 
	const float iterations = 2.;
	const float h = PLANCK_CONSTANT;
	const float k = BOLTZMANN_CONSTANT;
	const float c = SPEED_OF_LIGHT;

	float L = wavelength;
	float T = temperature;

	float C2 = h*c/k;
	float z = C2 / (L*T);
	
	return 15.*(z*z*z + 3.*z*z + 6.*z + 6.) * exp(-z)/(PI*PI*PI*PI);
}

// Function 20
vec2 implicitSolveV(vec2 pos)
{
    vec2 posInit = pos;
    for(int i=0; i<NITER; i++)
    {
        pos = posInit - Dt*texture(iChannel1,  world2uv(pos)).xy;
    }
    
    return pos;
}

// Function 21
vec3 solveContrainsts( in vec2 id, in vec3 p )
{
    if( id.x > 0.5 )  p = react( p, id + vec2(-1.0, 0.0), 0.1 );
    if( id.x < 38.5 )  p = react( p, id + vec2( 1.0, 0.0), 0.1 );
    if( id.y > 0.5 )  p = react( p, id + vec2( 0.0,-1.0), 0.1 );
    if( id.y < 38.5 )  p = react( p, id + vec2( 0.0, 1.0), 0.1 );

    if( id.x > 0.5 && id.y > 0.5)  p = react( p, id + vec2(-1.0, -1.0), 0.14142 );
    if( id.x > 0.5 && id.y < 38.5)  p = react( p, id + vec2(-1.0,  1.0), 0.14142 );
    if( id.x < 38.5 && id.y > 0.5)  p = react( p, id + vec2( 1.0, -1.0), 0.14142 );
    if( id.x < 38.5 && id.y < 38.5)  p = react( p, id + vec2( 1.0,  1.0), 0.14142 );

    return p;
}

// Function 22
vec3 solveCubic(float a, float b, float c)
{
    float p = b - a*a / 3.0, p3 = p*p*p;
    float q = a * (2.0*a*a - 9.0*b) / 27.0 + c;
    float d = q*q + 4.0*p3 / 27.0;
    float offset = -a / 3.0;
    if(d >= 0.0) { 
        float z = sqrt(d);
        vec2 x = (vec2(z, -z) - q) / 2.0;
        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        return vec3(offset + uv.x + uv.y);
    }
    float v = acos(-sqrt(-27.0 / p3) * q / 2.0) / 3.0;
    float m = cos(v), n = sin(v)*1.732050808;
    return vec3(m + m, -n - m, n - m) * sqrt(-p / 3.0) + offset;
}

// Function 23
vec2 solveCubic2b(vec1 a,vec1 b,vec1 c//https://www.shadertoy.com/view/XtdyDn
){vec2 p=vec2(b-a*a/3.,a)
 ;vec1 q=a*(2.*a*a-9.*b)/27.+c
 ,s=p.x*p.x*p.x
 ;c=q*q+4.*s/27.//determinant seperates cases where a root repeats
 ;if(q*q+4.*s/27.>0.)return root23(vec2(a),b,c)//both return values are identical
 ;float v=acos(-sqrt(-27./s)*q*.5)/3.,m=cos(v),n=sin(v)*sqrt(3.);p/=3.//...does not care for 3rd (middle) root, intended as subroutine for bezier/parabola
 ;return vec2(m+m,-n-m)*sqrt(-p.x)-p.y;}

// Function 24
bool SolveSquare(float A,float B,float C,out vec2 x
){float D=B*B-4.0*A*C
 ;if(D<0.0)return false
 ;x.x=(-B-sqrt(D))/(2.0*A)
 ;x.y=(-B+sqrt(D))/(2.0*A)
 ;return true;}

// Function 25
vec4 solveFluid(sampler2D smp, vec2 uv, vec2 w, float time, vec3 mouse, vec3 lastMouse)
{
	const float K = 0.2;
	const float v = 0.55;
    
    vec4 data = textureLod(smp, uv, 0.0);
    vec4 tr = textureLod(smp, uv + vec2(w.x , 0), 0.0);
    vec4 tl = textureLod(smp, uv - vec2(w.x , 0), 0.0);
    vec4 tu = textureLod(smp, uv + vec2(0 , w.y), 0.0);
    vec4 td = textureLod(smp, uv - vec2(0 , w.y), 0.0);
    
    vec3 dx = (tr.xyz - tl.xyz)*0.5;
    vec3 dy = (tu.xyz - td.xyz)*0.5;
    vec2 densDif = vec2(dx.z ,dy.z);
    
    data.z -= dt*dot(vec3(densDif, dx.x + dy.y) ,data.xyz); //density
    vec2 laplacian = tu.xy + td.xy + tr.xy + tl.xy - 4.0*data.xy;
    vec2 viscForce = vec2(v)*laplacian;
    data.xyw = textureLod(smp, uv - dt*data.xy*w, 0.).xyw; //advection
    
    vec2 newForce = vec2(0);
    #ifndef MOUSE_ONLY
    #if 1
    newForce.xy += 0.75*vec2(.0003, 0.00015)/(mag2(uv-point1(time))+0.0001);
    newForce.xy -= 0.75*vec2(.0003, 0.00015)/(mag2(uv-point2(time))+0.0001);
    #else
    newForce.xy += 0.9*vec2(.0003, 0.00015)/(mag2(uv-point1(time))+0.0002);
    newForce.xy -= 0.9*vec2(.0003, 0.00015)/(mag2(uv-point2(time))+0.0002);
    #endif
    #endif
    
    if (mouse.z > 1. && lastMouse.z > 1.)
    {
        vec2 vv = clamp(vec2(mouse.xy*w - lastMouse.xy*w)*400., -6., 6.);
        newForce.xy += .001/(mag2(uv - mouse.xy*w)+0.001)*vv;
    }
    
    data.xy += dt*(viscForce.xy - K/dt*densDif + newForce); //update velocity
    data.xy = max(vec2(0), abs(data.xy)-1e-4)*sign(data.xy); //linear velocity decay
    
    #ifdef USE_VORTICITY_CONFINEMENT
   	data.w = (tr.y - tl.y - tu.x + td.x);
    vec2 vort = vec2(abs(tu.w) - abs(td.w), abs(tl.w) - abs(tr.w));
    vort *= VORTICITY_AMOUNT/length(vort + 1e-9)*data.w;
    data.xy += vort;
    #endif
    
    data.y *= smoothstep(.5,.48,abs(uv.y-0.5)); //Boundaries
    
    data = clamp(data, vec4(vec2(-10), 0.5 , -10.), vec4(vec2(10), 3.0 , 10.));
    
    return data;
}

// Function 26
void solveCollisions(inout vec2 particlePrevPosition, inout vec2 particleCurrPosition)
{
    vec2 particleInertia = (particleCurrPosition - particlePrevPosition);
    
	if(particleCurrPosition.x < particlesSize || particleCurrPosition.x > iResolution.x - particlesSize)
    {
    	particleCurrPosition.x = clamp(particleCurrPosition.x, particlesSize, iResolution.x - particlesSize);
        particlePrevPosition.x = particleCurrPosition.x + particleInertia.x * collisionDamping;
    }
    
    if(particleCurrPosition.y < particlesSize || particleCurrPosition.y > iResolution.y - particlesSize)
    {
    	particleCurrPosition.y = clamp(particleCurrPosition.y, particlesSize, iResolution.y - particlesSize);
        particlePrevPosition.y = particleCurrPosition.y + particleInertia.y * collisionDamping;
    }
}

// Function 27
float solve_quadratic(poly2 f) {
    return solve_quadratic0(f.a);
}

// Function 28
void ResolveDistanceConstraint (inout vec2 pointA, inout vec2 pointB, float distance)
{
    // calculate how much we need to adjust the distance between the points
    // and cut it in half since we adjust each point half of the way
    float halfDistanceAdjust = (distance - length(pointB-pointA)) * 0.5;
    
    // calculate the vector we need to adjust along
    vec2 adjustVector = normalize(pointB-pointA);
    
    // adjust each point half of the adjust distance, along the adjust vector
    pointA -= adjustVector * halfDistanceAdjust;
    pointB += adjustVector * halfDistanceAdjust;
}

// Function 29
vec2 solveGrayScott(float feed, float kill, vec2 uv, vec4 uplus, vec4 ucross, vec4 vplus, vec4 vcross) 
{

    vec2 duv;

    #ifdef USE_NINE
    duv.x = diff.x*lapnine(uv.x, uplus, ucross);
    duv.y = diff.y*lapnine(uv.y, vplus, vcross);
    #else 
    duv.x = diff.x*lap(uv.x, uplus);
    duv.y = diff.y*lap(uv.y, vplus);
    #endif
    duv.x += - uv.x*uv.y*uv.y + feed*(1.0 - uv.x);
    duv.y += uv.x*uv.y*uv.y - (feed+kill)*uv.y;

    return uv + dt*duv;;
}

// Function 30
float solveP3(float a, float c, float d) {
	c /= a; d /= a;
	float C = -d/2.*(1.+sqrt(1.+(c*c*c)/(d*d)*4./27.));  
	C = sign(C)*pow(abs(C),1./3.);
    return C-c/(3.*C);
}

// Function 31
void resolve_nonaxial_intersection(inout Intersection result, int best_index)
{
    if (best_index == -1)
        return;
    
    vec4 plane = get_nonaxial_plane(best_index);
    
    result.normal = plane.xyz;
    result.uv_axis = dominant_axis(plane.xyz);
    result.plane = best_index + NUM_MAP_AXIAL_PLANES;
    result.material = get_plane_material(result.plane);

    // pixel quad straddling geometric planes? no mipmaps for you!
    float plane_hash = dot(plane, vec4(17463.12, 25592.53, 15576.84, 19642.77));
    result.mips = fwidth(plane_hash) < 1e-4;
}

// Function 32
float solve_black_body_fraction_between_wavelengths(float lo, float hi, float temperature){
	return 	solve_black_body_fraction_below_wavelength(hi, temperature) - 
			solve_black_body_fraction_below_wavelength(lo, temperature);
}

// Function 33
vec2 solveCubic2(vec3 a)
{
	float p  = a.y - a.x*a.x/3.,
	      p3 = p*p*p,
	      q  = a.x* ( 2.*a.x*a.x - 9.*a.y ) /27. + a.z,
	      d  = q*q + 4.*p3/27.;
    
	if(d>0.) {
		vec2 x = ( vec2(1,-1)*sqrt(d) -q ) *.5;
        x = sign(x) * pow( abs(x) , vec2(1./3.) );
  		return vec2( x.x+x.y -a.x/3. );
  	}
    
 	float v = acos( -sqrt(-27./p3)*q*.5 ) / 3.,
 	      m = cos(v),
 	      n = sin(v)*sqrt(3.);

	return vec2(m+m,-n-m) * sqrt(-p/3.) - a.x/3.;
}

// Function 34
bvec4 solve_quartic(in vec4 coeffs,
                    out vec4 roots) {
        
    float p = coeffs[0];
    float q = coeffs[1]; 
    float r = coeffs[2];
    float s = coeffs[3];
    
    ////////////////////////////////////////////////////////////
	// form resolvent cubic and solve it to obtain one real root
        
    float i = -q;
    float j = p*r - 4.*s;
    float k = 4.*q*s - r*r - p*p*s;
    
    // coefficients of normal form
    float a = (3.*j - i*i) / 3.;
    float b = (2.*i*i*i - 9.*i*j + 27.*k) / 27.;
    
    float delta1 = b*b / 4.;
    float delta2 = a*a*a / 27.;
    
    float delta = delta1 + delta2;
    
    float z1;
    
    if (delta >= 0.) {
        vec2 AB = -0.5*b + vec2(1,-1) * sqrt(max(delta, 0.));
        AB = sign(AB) * pow(abs(AB), vec2(1.0/3.0));
        z1 = AB.x + AB.y;
    } else {
        float phi = acos( -sign(b) * sqrt(delta1/-delta2) );
        z1 = 2. * sqrt(-a/3.) * cos( phi / 3.);
    }
    
    // shift back from normal form to root of resolvent cubic
    z1 -= i/3.;
    
    ////////////////////////////////////////////////////////////
	// now form quartic roots from resolvent cubic root

    float R2 = p*p/4. - q + z1; 
        
    bool R_ok = (R2 >= 0.);

    float R = sqrt(max(R2, 0.));
    
    float foo, bar;
    
    if (R == 0.) { 
        float z124s = z1*z1 - 4.*s;
        R_ok = R_ok && (z124s >= 0.);
        foo = 3.*p*p / 4. - 2.*q;
        bar = 2.*sqrt(max(z124s, 0.));
    } else {
        foo = 3.*p*p / 4. - R2 - 2.*q;
        bar = (4.*p*q - 8.*r - p*p*p) / (4.*R);
    }
    
    bool D_ok = R_ok && (foo + bar >= 0.);
    bool E_ok = R_ok && (foo - bar >= 0.);
    
    float D = sqrt(max(foo + bar, 0.));
    float E = sqrt(max(foo - bar, 0.));
    
    roots = vec4(-p/4.) + 0.5 * vec4(R+D, R-D, -(R-E), -(R+E));
    return bvec4(D_ok, D_ok, E_ok, E_ok);

}

// Function 35
vec4 solve_quartic(vec4 p){
 ;float quadrant=sign(p.x),s=p.w// form resolvent cubic and solve it to obtain one real root
 ,j=p.x*p.z-4.*p.w,k=4.*p.y*p.w-p.z*p.z-p.x*p.x*p.w,b=(-2.*p.y*p.y*p.y+9.*p.y*j+27.*k)/27.//coefficients of normal form
 ,delta1=b*b/4.,a=(3.*j-p.y*p.y)/3.,delta2=a*a*a/27.,z1
 ;if(delta1+delta2<0.)z1=2.*sqrt(-a/3.)*cos(acos(-sign(b)*sqrt(delta1/-delta2))/3.)
 ;else    z1=suv(pow(abs(-.5*b+vec2(1,-1)*sqrt(max(delta1+delta2,0.))),vec2(1./3.)))//sum of 2 cubic roots
 ;z1+=p.y/3. // shift back from normal form to root of resolvent cubic
 ;float R2=p.x*p.x/4.-p.y+z1//form quartic roots from resolvent cubic root
 ;bool R_ok=(R2>=0.);float R=sqrt(max(R2,0.)),foo,bar;if(R==0.//i do not call this elegant!
 ){float z124s=z1*z1-4.*p.w;R_ok=R_ok &&(z124s>=0.);foo=3.*p.x*p.x/4.-2.*p.y   ;bar=2.*sqrt(max(z124s,0.))
 ;}else{           ;foo=3.*p.x*p.x/4.-R2-2.*p.y;bar=(4.*p.x*p.y-8.*p.z-p.x*p.x*p.x)/(4.*R);}
 ;float D=sqrt(max(foo+bar,0.)),E=sqrt(max(foo-bar,0.));vec4 roots=vec4(-p.x/4.)+.5*vec4(R+D,R-D,-R+E,-R-E)
 ;roots=mix(roots,roots.xzyw,step(sign(p.x),0.))//optional root sorting within homotopy
 ;return roots;}

// Function 36
bool solveQuadratic(float A, float B, float C, out float t0, out float t1) {
	float discrim = B*B-4.0*A*C;
    
    if ( discrim <= 0.0 ){
        return false;
    } else {
        float rootDiscrim = sqrt( discrim );

        float t_0 = (-B-rootDiscrim)/(2.0*A);
        float t_1 = (-B+rootDiscrim)/(2.0*A);

        t0 = min( t_0, t_1 );
        t1 = max( t_0, t_1 );

        return true;
    }
}

// Function 37
void ResolveGroundCollision (inout vec2 point, inout bool pointTouchingGround)
{
    vec2 gradient;
    float dist = EstimatedDistanceFromPointToGround (point, 1.0, 1.0, gradient) * -1.0;
    if (dist < c_wheelRadius)
    {
        float distanceAdjust = c_wheelRadius - dist;
        point -= normalize(gradient) * distanceAdjust;
        pointTouchingGround = true;
    }
}

// Function 38
float solve_quadratic(vec3 fa, float x) {
    float a = fa[2];
    float b = fa[1];
    float c = fa[0];

    // the quadratic solve doesn't work for a=0
    // so we need a branch here.
    if (a == 0.0) {
        return -c / b;
    } else { 
        // (-b +- sqrt(b*b - 4.0*a*c)) / 2.0*a
        float k = -0.5*b/a;
        float q = sqrt(k*k - c/a);
        float q0 = k - q;
        float q1 = k + q;
        
        // pick the root right of x
		return (q0 <= x)?q1:q0;
    }
}

// Function 39
vec4 fluidSolver(sampler2D velocityField, vec2 uv, vec2 stepSize, vec4 mouse, vec4 prevMouse)
{
    float k = .2, s = k/dt;
    
    vec4 fluidData = textureLod(velocityField, uv, 0.);
    vec4 fr = textureLod(velocityField, uv + vec2(stepSize.x, 0.), 0.);
    vec4 fl = textureLod(velocityField, uv - vec2(stepSize.x, 0.), 0.);
    vec4 ft = textureLod(velocityField, uv + vec2(0., stepSize.y), 0.);
    vec4 fd = textureLod(velocityField, uv - vec2(0., stepSize.y), 0.);
    
    vec3 ddx = (fr - fl).xyz * .5;
    vec3 ddy = (ft - fd).xyz * .5;
    float divergence = ddx.x + ddy.y;
    vec2 densityDiff = vec2(ddx.z, ddy.z);
    
    // Solving for density
    fluidData.z -= dt*dot(vec3(densityDiff, divergence), fluidData.xyz);
    
    // Solving for velocity
    vec2 laplacian = fr.xy + fl.xy + ft.xy + fd.xy - 4.*fluidData.xy;
    vec2 viscosityForce = viscosityThreshold * laplacian;
    
    // Semi-lagrangian advection
    vec2 densityInvariance = s * densityDiff;
    vec2 was = uv - dt*fluidData.xy*stepSize;
    fluidData.xyw = textureLod(velocityField, was, 0.).xyw;
    
    // Calc external force from mouse input
    vec2 extForce = vec2(0.);
    
    if (mouse.z > 1. && prevMouse.z > 1.)
    {
        vec2 dragDir = clamp((mouse.xy - prevMouse.xy) * stepSize * 600., -10., 10.);
        vec2 p = uv - mouse.xy*stepSize;
        //extForce.xy += .75*.0002/(dot(p, p)+.0001) * (.5 - uv);
        extForce.xy += .001/(dot(p, p)) * dragDir;
    }
    
    fluidData.xy += dt*(viscosityForce - densityInvariance + extForce);
    
    // velocity decay
    fluidData.xy = max(vec2(0.), abs(fluidData.xy) - 5e-6)*sign(fluidData.xy);
    
    // Vorticity confinement
	fluidData.w = (fd.x - ft.x + fr.y - fl.y); // curl stored in the w channel
    vec2 vorticity = vec2(abs(ft.w) - abs(fd.w), abs(fl.w) - abs(fr.w));
    vorticity *= vorticityThreshold/(length(vorticity) + 1e-5)*fluidData.w;
    fluidData.xy += vorticity;

    // Boundary conditions
    fluidData.y *= smoothstep(.5,.48,abs(uv.y - .5));
    fluidData.x *= smoothstep(.5,.49,abs(uv.x - .5));
    
    // density stability
    fluidData = clamp(fluidData, vec4(vec2(-velocityThreshold), 0.5 , -velocityThreshold), vec4(vec2(velocityThreshold), 3.0 , velocityThreshold));
    
    return fluidData;
}

// Function 40
v2 SolveQuad(v2 a){v0 e=-a.x/3.;v0 p=a.y+a.x*e,t=p*p*p
,q=-(2.*a.x*a.x-9.*a.y)*e/9.+a.z,d=q*q+4.*t/27.;if(d>.0)
{v1 x=(v1(1,-1)*sqrt(d)-q)*.5;return v2(suv(sign(x)
*pow(abs(x),v1(1./3.)))+e);}v1 m=cs(acos(-sqrt(-27./t)*q*.5)/3.)
*v1(1,sqrt(3.));return v2(m.x+m.x,-suv(m),m.y-m.x)*sqrt(-p/3.)+e;}

// Function 41
vec2 DistanceConstraint(vec2 x, vec2 x2, float restlength, float stiffness)
{
    vec2 delta = x2 -x;
    float deltalength = length(delta);
    float diff = (deltalength-restlength) /deltalength;
    return delta*stiffness*diff;
}

// Function 42
float solve_quadratic0(vec3 fa) {
    float a = fa[2];
    float b = fa[1];
    float c = fa[0];

    // the quadratic solve doesn't work for a=0
    // so we need a branch here.
    if (a == 0.0) {
        return -c / b;
    } else { 
        // (-b +- sqrt(b*b - 4.0*a*c)) / 2.0*a
        float k = -0.5*b/a;
        float q = sqrt(k*k - c/a);
        // pick the closest root right of 0
		return k + ((k <= q)?q:-q);
    }
}

// Function 43
int solveCubic(float a, float b, float c, out float r[3]) {
    float p = b - a*a / 3.;
    float q = a * (2.*a*a - 9.*b) / 27. + c;
    float p3 = p*p*p;
    float d = q*q + 4.*p3 / 27.;
    float offset = -a / 3.;
    if(d >= 0.) { // Single solution
        float z = sqrt(d);
        float u = (-q + z) / 2.;
        float v = (-q - z) / 2.;
        u = cuberoot(u);
        v = cuberoot(v);
        r[0] = offset + u + v;
        return 1;
    }
    float u = sqrt(-p / 3.);
    float v = acos(-sqrt( -27. / p3) * q / 2.) / 3.;
    float m = cos(v), n = sin(v)*1.732050808;
    r[0] = offset + u * (m + m);
    r[1] = offset - u * (n + m);
    r[2] = offset + u * (n - m);
    return 3;
}

// Function 44
v1 solveCubic2b(v0 a,v0 b,v0 c//https://www.shadertoy.com/view/XtdyDn
){v1 p=v1(b-a*a/3.,a)
 ;v0 q=a*(2.*a*a-9.*b)/27.+c
 ,s=p.x*p.x*p.x
 ;c=q*q+4.*s/27.//determinant seperates cases where a root repeats
 ;if(q*q+4.*s/27.>0.)return root23(v1(a),b,c)//both return values are identical
 ;v0 v=acos(-sqrt(-27./s)*q*.5)/3.,m=cos(v),n=sin(v)*sqrt(3.);p/=3.//...does not care for 3rd (middle) root, intended as subroutine for bezier/parabola
 ;return v1(m+m,-n-m)*sqrt(-p.x)-p.y;}

// Function 45
v1 solveCubic2b(v2 a){return solveCubic2b(a.x,a.y,a.z);}

