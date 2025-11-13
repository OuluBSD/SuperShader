// Combined shader from SuperShader modules
// Automatically generated

#version 300 es
precision highp float;

// Begin module: forces_functions
// Reusable Forces Physics Functions
// Automatically extracted from particle/physics simulation-related shaders

// Function 1
vec3 calculate_gravity_accel(in int curBodyNum, in vec4 bodyXYZW, float dTimeAdvance)
{
    vec3 accelVec = vec3(0.0);
    float bodySize = bodyXYZW.w;
        
    for (int i = 0; i < BODIES; i ++) {
        if (i == curBodyNum) continue;
        vec4 otherXYZW;
        vec3 otherVelXYZ;
        getBodyPosVel(i, otherXYZW, otherVelXYZ);
        
        otherXYZW += dTimeAdvance * vec4(otherVelXYZ.xyz, 0);
        
        vec3 diff = otherXYZW.xyz - bodyXYZW.xyz;
        float R = sqrt(dot(diff, diff));
        float otherSize = pow(otherXYZW.w, 3.0);
        float accel = otherSize / R; 
        
        accelVec += normalize(diff) * accel;
    }
    return accelVec;
}

// Function 2
void drawSpring( vec2 coords, vec2 p0, vec2 p1, 
                 float thickness, int loops, vec4 color, inout vec4 outputColor )
{
    vec2 d = p1 - p0;
    if (length(d) < 0.001)
        return;
    
    vec2 dir = normalize(d);
    vec2 per = vec2(dir.y, -dir.x);
    
    vec2 st = d / float(loops * 2);
    vec2 last = p0 + per * thickness / 2.0 + st / 2.0;
    vec2 sw = -thickness * per;
    float th = 1.0;
    
    drawLine(coords, p0, last, th, color, outputColor);
    
    for (int i=0; i<loops*2-1; i++)
    {
        vec2 next = last + st + sw;
        sw = -sw;
        drawLine(coords, last, next, th, color, outputColor);
        last = next;
    }
    
    drawLine(coords, last, p1, th, color, outputColor);
}

// Function 3
void drawSpring(in vec2 fragCoord, in vec2 p0, in vec2 p1, in vec2 thickness,
                in int loops, in vec4 color, inout vec4 outputColor) {

    vec2 n = p1 - p0;
    float l = length(n), hl = l * 0.5;

    fragCoord -= p0;
    fragCoord *= mat2(n, n.y, -n.x) / l;

    float d = max(udTriangleWave(fragCoord, float(loops) / l, thickness.x) - thickness.y, abs(fragCoord.x - hl) - hl);
    outputColor = mix(outputColor, color, smoothstep(1.0, 0.0, d));
}

// Function 4
vec3 p2pForce(vec3 d) 
{
    float dd=dot(d,d);
    float d6=dd*dd*dd;  // lennard jones
    // d<1 repulsive  d>1 attractive
    // varies between -rep and ~+att
    float att=5.;
    float rep=180.;
    return 4.*att*d/sqrt(dd)*(dd-1.)/(dd*dd+4.*att/rep);
    // more lennard-jones-ish
    //return .2*8.*att*d/sqrt(dd)*(d6-1.)/(d6*d6*sqrt(dd)+8.*att/rep);
}

// Function 5
vec3 calc_force(vec3 p)
{
    vec3 p0 = round(p);
    vec3 force = vec3(0.);
    for( int i=-rad; i<=rad; i++ )
        for( int j=-rad; j<=rad; j++ )
            for( int k=-rad; k<=rad; k++ )
    {
        vec3 dx = vec3(i,j,k);
        if(dx != vec3(0))
          force += getForce(p0+dx, p);
    }
    
    return force;
}

// Function 6
float force(float d)
{
    d *= 4.;
    return -2.*(0.5*exp(-0.5*d) - 2.*exp(-3.*d));
}

// Function 7
void calcObjForceAndMom(Particle p, Particle p2, inout vec3 force, inout vec3 torque)
{
    vec3 collpos1;
    vec3 collpos2;
    vec3 dforce=vec3(0);
    vec3 d;
    vec3 dn;

    for(int k=0;k<4;k++)
    {
        //collpos1 = p.pos.xyz  + transformVecByQuat(vec3(0,0.6,0)-vec3(0,1.2,0)*float(k&1),p.quat);
        //collpos2 = p2.pos.xyz + transformVecByQuat(vec3(0,0.6,0)-vec3(0,1.2,0)*float(k&2)*0.5,p2.quat);
        collpos1 = p.pos.xyz  + transformVecByQuat(vec3(0,0.6,0)-vec3(0,1.2,0)*mod(float(k),2.0), p.quat);
        collpos2 = p2.pos.xyz + transformVecByQuat(vec3(0,0.6,0)-vec3(0,1.2,0)*float(k/2)*0.5,    p2.quat);

        d = collpos1 - collpos2;
        dn = p2.pos.xyz-p.pos.xyz;
        if( length(d)<2.0 ) dforce=100.1*d/(length(d)+0.1);
        force+=dforce;
        torque+=1.0*cross(dforce,0.5*(collpos1+collpos2)-p.pos.xyz);
    }
    //TODO: friction force+torque
    //torque+=0.3*cross(cross(dforce,normalize(dn)),0.5*(collpos1+collpos2)-pos);
}

// Function 8
vec2 getRepellForce(float id)
{
    vec2 force = vec2(0,0);
    
    for(float i=0.0; i<N; i++)
    {
        if(i != id)
        {
        	vec2 otherPos = getBoid(i).xy;
       		if(length(getBoid(id).xy - otherPos) < MIN_DIST)
            {
                vec2 d = (getBoid(id).xy - otherPos);
            	force = force + d/length(d) * (MIN_DIST - length(d));
            }
        }
    }
    
    return force;
}

// Function 9
vec2 gravityWave(in float t, in float a, in float k, in float h) {
  float w = sqrt(gravity*k*tanh(k*h));
  return wave(t, a ,k, w*iTime);
}

// Function 10
float hashForCell(in vec3 pos, in float cellSize) {
    float hash = nrand(floor(pos.xz / cellSize) + 68.0);
    return hash;
}

// Function 11
vec2 summ_current_minus_rest_spring_len(vec2 f, vec2 R, particle parts)
{
    vec2 summ_pos = vec2(.0);    
    vec2 fpos = texture(iChannel0, (f+vec2(0., 0.))/R.xy ).xy;
    float xrest = 1.;
    for(float i = 0.; i <= 2.; i++)
    {
        vec2 cd = vec2(.0, i-1.);
        vec2 idf = f + cd;
        if(idf.x < .0 || idf.x > NUM || idf.y < .0 || idf.y > NUM)
            continue;
        if (cd.y == 0.)
            continue;
        float ma = texture(iChannel0, (f+cd+vec2(.0, NUM) )/R).x;
            xrest = DIST;
       	cd = texture(iChannel0, (f+cd)/R.xy ).xy; // need a vec2, reusing cd var
            summ_pos += DistanceConstraint(fpos, cd, xrest*1., 6.*.0+1.0*8./(abs(ma-parts.ma)*.25+1.)+.0 );
    }
	return summ_pos/2.;
}

// Function 12
vec3 forces(ivec2 iv, vec3 pos, sampler2D iChannel) {
    return force_gravity(iv, pos, iChannel) + force_boundary(pos);
}

// Function 13
float ClampTyreForce( inout vec3 vVel, float fLimit )
{
    // Square clamp
    //vVelWheel.x = clamp( vVelWheel.x, -fLimit, fLimit);
    //vVelWheel.z = clamp( vVelWheel.z, -fLimit, fLimit);
	float fSkid = 0.0;
    
    // Circluar clamp
    float fMag = length(vVel);
    if( fMag > 0.0 )
    {	        
        vVel = normalize( vVel );
    }
    else
    {
        vVel = vec3(0.0);
    }
    if ( fMag > fLimit )
    {
        fSkid = fMag - fLimit;
	    fMag = fLimit;        
    }
    vVel = vVel * fMag;
    
    return fSkid;
}

// Function 14
void BodyApplyGravity( inout Body body, float dT )
{
    float fAccel_MpS = -9.81;
    body.vForce.y += body.fMass * fAccel_MpS;
}

// Function 15
vec3 SpringForce (ivec2 iv, vec3 r, vec3 v)
{
  vec3 dr, f;
  ivec2 ivn;
  float spLenD, fSpring, fDamp;
  fSpring = 200.;
  fDamp = 0.5;
  f = vec3 (0.);
  for (int n = 0; n < 4; n ++) {
    ivn = iv + idNeb[n];
    if (ivn.y >= 0 && ivn.y < nBallE && ivn.x >= 0 && ivn.x < nBallE) {
      dr = r - GetR (vec2 (ivn));
      f += fSpring * (spLen - length (dr)) * normalize (dr) -
         fDamp * (v - GetV (vec2 (ivn)));
    }
  }
  spLenD = spLen * sqrt (2.);
  for (int n = 0; n < 4; n ++) {
    ivn = iv + idNebD[n];
    if (ivn.y >= 0 && ivn.y < nBallE && ivn.x >= 0 && ivn.x < nBallE) {
      dr = r - GetR (vec2 (ivn));
      f += 5. * fSpring * (spLenD - length (dr)) * normalize (dr) -
         fDamp * (v - GetV (vec2 (ivn)));
    }
  }
  return f;
}

// Function 16
void calcWheelForceAndTorque(inout Particle p, vec3 wheelPos, float wheelRadius, float steeringAngle, float frictionCoeff, float wheelAngSpeed, float springConst, float clutch, inout vec3 force, inout vec3 torque)
{
    vec3 wheelPosW = p.pos.xyz+transformVecByQuat(wheelPos.xyz,p.quat);
    float dist = getDistanceWorldS(wheelPosW);
    if(dist<wheelRadius)
    {
        // forces by wheels
        vec3 dforce = vec3(0,0,0);
        vec3 distGrad = getDistanceWorldSGradient(wheelPosW,0.1);
        vec3 distDir = normalize(distGrad);
        dforce=distDir*(wheelRadius-dist)*springConst;
        //p.vel.xyz -= distDir*dot(p.vel.xyz,distDir)*0.1;
        float fl = length(dforce);
        //if(fl>5.0) { dforce=dforce/fl*5.0; fl = length(dforce); }
        force  += dforce;
        torque += cross(dforce,wheelPosW-p.pos.xyz);

        // wheel-drive-forces
        vec3 chassisRotVel = transformVecByQuat(cross(-p.angVel.xyz,wheelPos.xyz+vec3(0,0,-WheelRadius)),p.quat);
        vec3 wheelAxe = transformVecByQuat(vec3(cos(steeringAngle),sin(-steeringAngle),0),p.quat);
        vec3 f=normalize(cross(distDir,wheelAxe));
        //vec3 c=wheelAngSpeed*wheelRadius*f-p.vel.xyz-chassisRotVel+dot(p.vel.xyz+chassisRotVel,distDir)*0.9*distDir;
        ///*if(length(c)>1.0)*/ c=normalize(c);
        //dforce = fl*frictionCoeff*c;

        vec3 c=wheelAngSpeed*wheelRadius*f-p.vel.xyz-chassisRotVel;
        dforce = vec3(0.0);
        float ffr = fl*frictionCoeff; if(ffr>5.0) ffr=5.0;
        // in f dir
        dforce += ffr*clutch*sign(dot(c,f))*f;
        // in wheelaxe dir
        dforce += ffr*sign(dot(c,wheelAxe))*wheelAxe;
        // in up dir - pure velocity damping in the dampers
        float du=4.7; /* damping for wheel down (chassis up) */
        float dd=2.7; /* damping for wheel up (chassis down) */
        dforce += ((du+dd)*0.5*sign(dot(c,distDir))-0.5*(du-dd))*distDir;

        force += dforce;
        torque += cross(dforce,wheelPosW+transformVecByQuat(vec3(0,0,-WheelRadius),p.quat)-p.pos.xyz);
    }

}

// Function 17
vec3 force_boundary(vec3 pos) {
    float r = distance(pos, POSB);
    return r >= RADB ? -2.0 * K * (r - RADB) * (pos - vec3(0.5)) : vec3(0);
}

// Function 18
vec3 getGravityWorld(vec3 pos)
{
    return GRAVITY*normalize(getDistanceWorldSGradientSlow(pos, 5.0)+getDistanceWorldSGradientSlow(pos, 10.0));
}

// Function 19
vec3 randomColourForCell(in vec3 pos, in float cellSize) {
	float hash = hashForCell(pos, cellSize); 
    return vec3(
        nrand(vec2(hash * 2.0, hash * 4.0)),
        nrand(vec2(hash * 4.0, hash * 8.0)),
        nrand(vec2(hash * 8.0, hash * 16.0))
	);
	vec3 c = vec3(hash, mod(hash + 0.15, 1.0), mod(hash + 0.3, 1.0)) * 0.75;
}

// Function 20
void BodyApplyForce(inout Body body, vec3 vPos, vec3 vForce) {    
    body.vForce += vForce;
    body.vTorque += cross(vPos - body.vPos, vForce);   
}

// Function 21
void calcWheelForceAndTorque(inout Particle p, vec3 wheelPos, float wheelRadius, float steeringAngle, float frictionCoeff, float wheelAngSpeed, float springConst, float clutch, inout vec3 force, inout vec3 torque)
{
    vec3 wheelPosW = p.pos.xyz+transformVecByQuat(wheelPos.xyz,p.quat);
    float dist = getDistanceWorldS(wheelPosW);
    if(dist<wheelRadius)
    {
        // forces by wheels
        vec3 dforce = vec3(0,0,0);
        vec3 distGrad = getDistanceWorldSGradient(wheelPosW,0.1);
        vec3 distDir = normalize(distGrad);
        dforce=distDir*(wheelRadius-dist)*springConst;
        //p.vel.xyz -= distDir*dot(p.vel.xyz,distDir)*0.1;
        force  += dforce;
        torque += cross(dforce,wheelPosW-p.pos.xyz);

        float fl = length(dforce);

        // wheel-drive-forces
        vec3 chassisRotVel = transformVecByQuat(cross(-p.angVel.xyz,wheelPos.xyz+vec3(0,0,-WheelRadius)),p.quat);
        vec3 wheelAxe = transformVecByQuat(vec3(cos(steeringAngle),sin(-steeringAngle),0),p.quat);
        vec3 f=normalize(cross(distDir,wheelAxe));
        //vec3 c=wheelAngSpeed*wheelRadius*f-p.vel.xyz-chassisRotVel+dot(p.vel.xyz+chassisRotVel,distDir)*0.9*distDir;
        ///*if(length(c)>1.0)*/ c=normalize(c);
        //dforce = fl*frictionCoeff*c;

        vec3 c=wheelAngSpeed*wheelRadius*f-p.vel.xyz-chassisRotVel;
        // in f dir
        dforce += fl*frictionCoeff*clutch*sign(dot(c,f))*f;
        // in wheelaxe dir
        dforce += fl*frictionCoeff*sign(dot(c,wheelAxe))*wheelAxe;
        // in up dir
        dforce += fl*sign(dot(c,distDir))*distDir*0.7;

        force  += dforce;
        torque += cross(dforce,wheelPosW+transformVecByQuat(vec3(0,0,-WheelRadius),p.quat)-p.pos.xyz);
    }

}

// Function 22
float calcGravity(const float mass, const float r){
	return G * (mass / (r * r));
}

// Function 23
void drawSpring(in vec2 fragCoord, in vec2 p0, in vec2 p1, in vec2 thickness,
                in int loops, in vec4 color, inout vec4 outputColor) {
  if (sdSegment(fragCoord, p0, p1) > thickness.x) {
    return;
  }

  vec2 d = p1 - p0;
  if (length(d) < 0.001) {
    return;
  }

  vec2 dir = normalize(d);
  vec2 per = vec2(dir.y, -dir.x);

  vec2 st = d / float(loops * 2);
  vec2 last = p0 + per * thickness.x / 2.0 + st / 2.0;
  vec2 sw = -thickness.x * per;
  float th = thickness.y;

  drawSegment(fragCoord, p0, last, th, color, outputColor);

  for (int i = 0; i < loops * 2 - 1; i++) {
    vec2 next = last + st + sw;
    sw = -sw;
    drawSegment(fragCoord, last, next, th, color, outputColor);
    last = next;
  }

  drawSegment(fragCoord, last, p1, th, color, outputColor);
}

// Function 24
define SPRING(q, i_, j_, f_) { \
		float d = distance(p, q) \
    	, l = llen * length(vec2(i_,j_)); \
        po -= (q - p) * (d - l) * (f_) * dt; \
    }

// Function 25
vec2 getGravity(vec2 res, vec4 particle, vec2 pos) {
    // Anti-gravity
    float MIN_DIST = 0.01;
    float G = 5.0e-1;
    float m = 1.0 / (MIN_DIST * MIN_DIST);
    vec2 dvg = particle.xy - pos.xy; 
    float l2 = length(dvg);
    vec2 dvgn = dvg / l2;
    
    vec2 a = G * dvg / (MIN_DIST + m * l2 * l2);
    
    return a;
}

// Function 26
void BodyApplyForce( inout Body body, vec3 vPos, vec3 vForce )
{    
    body.vForce += vForce;
    body.vTorque += cross(vPos - body.vPos, vForce);     
}

// Function 27
void spring(float force, inout vec4 p1, inout vec4 p2) {
    vec2 f = (p2.xy-p1.xy) * force;
    p1.zw += f;
    p2.zw -= f;
}

// Function 28
void BodyApplyDebugForces( inout Body body )
{
#ifdef ENABLE_DEBUG_FORCES    
    float debugForceMag = 20000.0;
    if ( KeyIsPressed( KEY_LEFT ) )
    {
        vec3 vForcePos = body.vPos;
        vec3 vForce = vec3(-debugForceMag, 0.0, 0.0);
        BodyApplyForce( body, vForcePos, vForce );
    }
    if ( KeyIsPressed( KEY_RIGHT ) )
    {
        vec3 vForcePos = body.vPos;
        vec3 vForce = vec3(debugForceMag, 0.0, 0.0);
        BodyApplyForce( body, vForcePos, vForce );
    }
    if ( KeyIsPressed( KEY_UP ) )
    {
        vec3 vForcePos = body.vPos;
        vec3 vForce = vec3(0.0, 0.0, debugForceMag);
        BodyApplyForce( body, vForcePos, vForce );
    }
    if ( KeyIsPressed( KEY_DOWN ) )
    {
        vec3 vForcePos = body.vPos;
        vec3 vForce = vec3(0.0, 0.0, -debugForceMag);
        BodyApplyForce( body, vForcePos, vForce );
    }
#endif // ENABLE_DEBUG_FORCES                
    
    float debugTorqueMag = 4000.0;
    if ( KeyIsPressed( KEY_COMMA ) )
    {
        vec3 vForcePos = body.vPos;
        vec3 vForce = vec3(0.0, -debugTorqueMag, 0.0);
		vForcePos.x += 2.0;
        BodyApplyForce( body, vForcePos, vForce );
		//vForcePos.x -= 4.0;
        //vForce = -vForce;
        //BodyApplyForce( body, vForcePos, vForce );
    }
    if ( KeyIsPressed( KEY_PER ) )
    {
        vec3 vForcePos = body.vPos;
        vec3 vForce = vec3(0.0, debugTorqueMag, 0.0);
		vForcePos.x += 2.0;
        BodyApplyForce( body, vForcePos, vForce );
		//vForcePos.x -= 4.0;
        //vForce = -vForce;
        //BodyApplyForce( body, vForcePos, vForce );
    }        
}

// Function 29
vec2 getSpring(vec2 res, vec4 particle, vec2 pos) {
    vec2 dv = particle.xy - pos;
    float l = length(dv);
    float k = 0.1;
    float s = sign(k - l);
    vec2 dvn = dv / (E + l);
    l = min(abs(k - l), l);
    
    float SPRING_COEFF = 1.0e2;
    float SPRING_LENGTH = 0.001;
    float X = abs(SPRING_LENGTH - l);
    float F_spring = SPRING_COEFF * X;
    
    if (l >= SPRING_LENGTH) {
    	dv = dvn * SPRING_LENGTH;
    }
    
    
    vec2 a = vec2(0.0);
    
    // Spring force
    a += -dv * F_spring;
    
    return a;
}

// Function 30
vec3 calc_gravity( vec3 r )
{
    float r2 = dot( r, r );
    return r2 < square( g_data.radius ) ?
        - g_data.GM / cube( g_data.radius ) * r :
        - g_data.GM / ( r2 * sqrt( r2 ) ) * r;
}

// Function 31
vec3 getForce(vec3 p0, vec3 p)
{
	particle neighbor = get(fakech0, p0);
    vec3 dx = neighbor.pos.xyz - p;
    //only count if neighbor particle is inside of its cell to exclude repeated forces
    if(maxv(abs(neighbor.pos.xyz - round(p0))) <= 0.5)
        return normalize(dx)*force(length(dx));
    else
        return vec3(0.);
}

// Function 32
vec3 gravity(inout vec3 p, inout vec3 v){//Positio Velocity
 float d=Scene(p+FeetPosition);
    
    
  //v.y=mix(v.y+Gravity*iTimeDelta,0.,(sign(d)*.5+.5));
    
  if(d>.0)v.y=v.y+Gravity*iTimeDelta;
  else v.y =0.;
    

    //p.y-=d*u5((-sign(d)));
    
    p.y-=d*(-sign(d)*.5+.5);
 return v;
}

// Function 33
void calcObjForceAndMom(Particle p, Particle p2, inout vec3 force, inout vec3 torque)
{
    vec3 collpos1;
    vec3 collpos2;
    vec3 dforce=vec3(0);
    vec3 d;
    vec3 dn;

    for(int k=0;k<4;k++)
    {
        //collpos1 = p.pos.xyz  + transformVecByQuat(vec3(0,0.6,0)-vec3(0,1.2,0)*float(k&1),p.quat);
        //collpos2 = p2.pos.xyz + transformVecByQuat(vec3(0,0.6,0)-vec3(0,1.2,0)*float(k&2)*0.5,p2.quat);
        collpos1 = p.pos.xyz  + transformVecByQuat(vec3(0,0.6,0)-vec3(0,1.2,0)*mod(float(k),2.0),    p.quat);
        collpos2 = p2.pos.xyz + transformVecByQuat(vec3(0,0.6,0)-vec3(0,1.2,0)*mod(float(k),4.0)*0.5,p2.quat);

        d = collpos1 - collpos2;
        dn = p2.pos.xyz-p.pos.xyz;
        if( length(d)<2.0 ) dforce=100.1*d/(length(d)+0.1);
        force+=dforce;
        torque+=1.0*cross(dforce,0.5*(collpos1+collpos2)-p.pos.xyz);
    }
    //TODO: friction force+torque
    //torque+=0.3*cross(cross(dforce,normalize(dn)),0.5*(collpos1+collpos2)-pos);
}

// Function 34
void drawSpring( vec2 coords, vec2 p0, vec2 p1, float thickness, int loops )
{
    vec2 d = p1 - p0;
    if (length(d) < 0.001)
        return;
    
    vec2 dir = normalize(d);
    vec2 per = vec2(dir.y, -dir.x);
    
    vec2 st = d / float(loops * 2);
    vec2 last = p0 + per * thickness / 2.0 + st / 2.0;
    vec2 sw = -thickness * per;
    float th = 1.0;
    
    drawLine(coords, p0, last, th);
    
    for (int i=0; i<loops*2-1; i++)
    {
        vec2 next = last + st + sw;
        sw = -sw;
        drawLine(coords, last, next, th);
        last = next;
    }
    
    drawLine(coords, last, p1, th);
}

// Function 35
float spring(vec3 p, int profile) {
    float radius = 0.5;
    float height = 3.0 + sin(iTime);
    float coils = 5.0/(height/3.141);

    vec3 pc = closestPointOnCylinder(p, vec2(radius, height));

    float distToCyl = distance(p, pc);
	float distToCoil = asin(sin(p.z*coils + 0.5*atan(p.x,p.y)))/coils;
    
    vec2 springCoords = vec2(distToCyl, distToCoil);
    
    //the multiplication factor is here to reduce the chance of the ray jumping through the back spring
    return profileForIndex(springCoords, profile) * ( max(radius/2.0-abs(length(p.xy)-radius), 0.0)*0.3 + 0.7);
}

// Function 36
vec2 summ_current_minus_rest_spring_len(vec2 f, vec2 R)
{
    vec2 summ_pos = vec2(.0);    
    vec2 fpos = texture(iChannel0, (f+vec2(0., 0.))/R.xy ).xy;
    float xrest = 1.;
    for(float i = 0.; i < 9.; i++)
    {
        vec2 cd = vec2(float(int((i))%3-1), float(int(i/3.)-1));
        vec2 idf = f + cd;
        if(idf.x < .0 || idf.x > NUM || idf.y < .0 || idf.y > NUM)
            continue;
        if (cd.x == cd.y && cd.x == 0.)
            continue;
        if (abs(cd.x) == abs(cd.y) && abs(cd.x) == 1.) // diagonals
            xrest = Xrd*DIST;
        else
            xrest = Xrs*DIST;
       	cd = texture(iChannel0, (f+cd)/R.xy ).xy; // need a vec2, reusing cd var
            summ_pos += DistanceConstraint(fpos, cd, xrest, .8105106125);
    }
	return summ_pos/8.;
}

// Function 37
float sdSpring(vec3 p, float Radius, float radius, float height, float turns ) {
    float pitch = height/turns;  
    
    // p.xz of Spring cylinder
    vec2 np = normalize(p.xz)*Radius; 
    
    // closest Point On Spring Cylinder
    vec3 pc = vec3(np.x, clamp(p.y, -height*0.5, height*0.5), np.y); 
    
    // closest distance to Spring cylinder, p to pc
    float distanceToCylinder = distance(p, pc); 

    // distance, pc to Spring Coil center
	float pcToSpring = p.y + atan(p.z, p.x)*pitch/TWO_PI;  // atan() range -PI to PI
    
    float distanceToSpring = TriangleFunction(pcToSpring, pitch); 
    // so 'close distance to Spring center for p' is length(vec2(distanceToCylinder, distanceToSpring)
    // we could construct springCoords with origin at Spring Coil center to calculate closest distance. 
    vec2 springCoords = vec2(distanceToCylinder, distanceToSpring); 
    
    return sdCircle(springCoords, radius); // circle shape of spings
}

// Function 38
vec3 calc_gravity_relief( vec3 r, vec3 v )
{
    vec3 omega = cross( r, v ) / dot( r, r );
	return cross( omega, cross( omega, r ) );
}

// Function 39
vec2 gravitationalForce(vec2 planetPos, vec2 sunPos, float sunMass)
{
    vec2 toCenter = sunPos-planetPos;
    float r = length(toCenter);
	return toCenter * ((G * sunMass) / (r*r*r)); // F = rvector * GmM/(r)^3
}

// Function 40
void
    renderSprings( V3 e, V3 r, inout F l, inout V3 h, inout V3 n, inout int i, bool s ) {
        
        V2
            sc = V2( 1.5, .5 );
       
        for( int j = 0; j < int( SCNT ); ++j ) {

            V4
                sp = PX( Ri * sc ),
                pf = PX( Ri * V2( sp.z + .5, 1.5 ) ),
                pt = PX( Ri * V2( sp.a + .5, 1.5 ) );
            
            F
                d = .095 * ( 1. + sin( .1 * T ) ) / max( 1., 1. + ( distance( pf, pt ) - sp.y ) );
            
            hitSpring( e, r, pf.xyz, pt.xyz, d, l, h, n, i );
            
            if( ( i == IDSPRING ) && s )
                
                return;                
            
            ++sc.x;
        }
    }

// Function 41
vec2 gravityWaveD(in float t, in float a, in float k, in float h) {
  float w = sqrt(gravity*k*tanh(k*h));
  return dwave(t, a, k, w*iTime);
}

// Function 42
vec2 calcForce()
{
    float forceDist = distance(uv, ForcePos);
    float forceValue = (1.0-step(ForceRadius,forceDist)) * ForceStrength;
    return vec2(forceValue, 0.0) * iTimeDelta;
}

// Function 43
vec3 BendForce (ivec2 iv, vec3 r)
{
  vec3 dr1, dr2, rt, f;
  ivec2 ivd, ivp, ivm;
  float s, c11, c22, c12, cd, fBend;
  bool doInt;
  fBend = 500.;
  f = vec3 (0.);
  for (int nd = 0; nd < 2; nd ++) {
    ivd = (nd == 0) ? ivec2 (1, 0) : ivec2 (0, 1);
    for (int k = 0; k < 3; k ++) {
      doInt = false;
      if (nd == 0) {
        if (k == 0 && iv.x > 1) {
          ivp = iv - ivd;
          ivm = ivp - ivd;
          doInt = true;
        } else if (k == 2 && iv.x < nBallE - 2) {
          ivm = iv + ivd;
          ivp = ivm + ivd;
          doInt = true;
        } else if (k == 1 && (iv.x > 0 && iv.x < nBallE - 1)) {
          ivp = iv + ivd;
          ivm = iv - ivd;
          doInt = true;
        }
      } else {
        if (k == 0 && iv.y > 1) {
          ivp = iv - ivd;
          ivm = ivp - ivd;
          doInt = true;
        } else if (k == 2 && iv.y < nBallE - 2) {
          ivm = iv + ivd;
          ivp = ivm + ivd;
          doInt = true;
        } else if (k == 1 && (iv.y > 0 && iv.y < nBallE - 1)) {
          ivp = iv + ivd;
          ivm = iv - ivd;
          doInt = true;
        }
      }
      if (doInt) {
        if (k == 0) {
          rt = GetR (vec2 (ivp));
          dr1 = rt - GetR (vec2 (ivm));
          dr2 = r - rt;
          s = -1.;
        } else if (k == 2) {
          rt = GetR (vec2 (ivm));
          dr1 = rt - r;
          dr2 = GetR (vec2 (ivp)) - rt;
          s = -1.;
        } else {
          dr1 = r - GetR (vec2 (ivm));
          dr2 = GetR (vec2 (ivp)) - r;
          s = 1.;
        }
        c11 = 1. / dot (dr1, dr1);
        c12 = dot (dr1, dr2);
        c22 = 1. / dot (dr2, dr2);
        cd = sqrt (c11 * c22);
        s *= fBend * cd * (c12 * cd - 1.);
        if (k <= 1) f += s * (dr1 - c12 * c22 * dr2);
        if (k >= 1) f += s * (c12 * c11 * dr1 - dr2);
      }
    }
  }
  return f;
}

// Function 44
void
    hitSpring( V3 e, V3 r, V3 p0, V3 p1, F rad, inout F l, inout V3 h, inout V3 n, inout int i ) {

        mat3
            q = aprj( p1 - p0 );
        
        V3
            a   = e - p0,
            qa  = q * a;
        
        F
            qaa = dot( a, qa ),
            qsa = dot( r, qa ),
            qss = dot( r, q * r ),
            ds  = qsa * qsa - qss * ( qaa - rad * rad );
        
        if( ds < 0. )
            
            return;
        
        F
            am  = ( -qsa - sqrt( ds ) ) / qss;
        
        if( l < am || am < 0.)
            
            return;
        
        F
            ht = dot( a + am * r, ( p1 - p0 ) / dot( p1 - p0, p1 - p0 ) );
        
        if( ht < 0. || 1. < ht )
            
            return;

        i  = IDSPRING;
        l  = am;
        h  = e + am * r;
        n  = vec3( normalize( q * ( a + am * r ) ) );
    }

// Function 45
vec2 summ_current_minus_rest_spring_len(vec2 f, vec2 R, particle parts)
{
    vec2 summ_pos = vec2(.0);    
    vec2 fpos = texture(iChannel0, (f+vec2(0., 0.))/R.xy ).xy;
    float xrest = 1.;
    for(float i = 0.; i < 9.; i++)
    {
        vec2 cd = vec2(float(int((i))%3-1), float(int(i/3.)-1));
        vec2 idf = f + cd;
        if(idf.x < .0 || idf.x > NUM || idf.y < .0 || idf.y > NUM)
            continue;
        if (cd.x == cd.y && cd.x == 0.)
            continue;
        float ma = texture(iChannel0, (f+cd+vec2(.0, NUM) )/R).x;
        if (abs(cd.x) == abs(cd.y) && abs(cd.x) == 1.) // diagonals
            xrest = Xrd*DIST;
        else
            xrest = Xrs*DIST;
       	cd = texture(iChannel0, (f+cd)/R.xy ).xy; // need a vec2, reusing cd var
            summ_pos += DistanceConstraint(fpos, cd, xrest, 1./(abs(ma-parts.ma)+.125)+.05 );
    }
	return summ_pos/8.;
}

// Function 46
vec3 NormForce (ivec2 iv, vec3 r, vec3 v)
{
  vec3 f, n;
  ivec2 e = ivec2 (1, 0);
  f = vec3 (0.);
  if (iv.y > 0 && iv.y < nBallE - 1 && iv.x > 0 && iv.x < nBallE - 1) {
    n = normalize (cross (GetR (vec2 (iv + e.yx)) - GetR (vec2 (iv - e.yx)),
       GetR (vec2 (iv + e)) - GetR (vec2 (iv - e))));
    f = - 10. * dot (v, n) * n;
  }
  return f;
}

// Function 47
float springODE(in float t, in float t0, in float x0, in float v0,
                in float m, in float u, in float k) {

    t -= t0;
    float alpha = -u / (2.0 * m);
    float discr = alpha * alpha - k / m;
    float beta = sqrt(abs(discr));

    if (discr < 0.0) { // Normal oscillation
        float w = beta * t;
        return (x0 * cos(w) - (alpha * x0 - v0) / beta * sin(w)) * exp(alpha * t);
    }

    if (abs(discr) < 1e-3) { // No oscillation - edge case
        return (x0 - (alpha * x0 - v0) * t) * exp(alpha * t);
    }

    // No oscillation
    return (((beta - alpha) * x0 + v0) * exp((alpha + beta) * t) +
            ((beta + alpha) * x0 - v0) * exp((alpha - beta) * t)) / (2.0 * beta);
}

// Function 48
void BodyApplyForceM( inout Body body, vec3 vPos, vec3 vForce )
{    
    //  body.vForce += vForce;
    body.vTorque += cross(vPos, vForce);

}

// Function 49
SpringState spring(vec3 currentValues, vec3 currentVelocities, vec3 targetValues, float dt) {
    
    float f = 1.0f + 2.0f * dt * DAMPING_RATIO * ANGULAR_FREQUENCY;
    float oo = ANGULAR_FREQUENCY * ANGULAR_FREQUENCY;
    float hoo = dt * oo;
    float hhoo = dt * hoo;
    float detInv = 1.0f / (f + hhoo);
    vec3 detX = f * currentValues + dt * currentVelocities + hhoo * targetValues;
    vec3 detV = currentVelocities + hoo * (targetValues - currentValues);
    
    vec3 newValues = detX * detInv;
    vec3 newVelocities = detV * detInv;
    
    return SpringState(newValues, newVelocities);
}

// Function 50
void integrateSpring(inout Point p1, inout Point p2, in float rest_length, in float stepsize)
{
    float dist = distance(p1.pos, p2.pos);
    
    float factor = clamp((rest_length - dist)/dist, -1., 1.);
    
    vec3 dir = (p1.pos - p2.pos) * factor * stepsize;
    
    p1.vel += dir;
    p2.vel -= dir;
}

// Function 51
void BodyApplyForce( inout Body body, vec3 vPos, vec3 vForce )
{    
    body.vForce += vForce;
    body.vTorque += cross(vPos - body.vPos, vForce);     

}

// Function 52
void forceIA(inout Particle p, int nb[16])
{
    vec3 acc=vec3(0.0);
    
    // gravity
    acc+=vec3(0,0,-5);
    
    // neighbour ia
    vec3 nvel=vec3(0);
    float cnt=0.;
    for(int i=0;i<16;i++)
    {
        if(nb[i]<0) break;
        if(nb[i]>=NumParticles || nb[i]==p.idx) continue;
        Particle pn=readParticle(nb[i]);
        float weight=1.;
        weight/=(1.+length(pn.pos-p.pos));
        nvel+=pn.vel*weight;
        acc+=p2pForce(pn.pos-p.pos);
        cnt+=weight;
    }
    // average velocity of all neighbours
    nvel/=cnt;
    
    // velocity damping
    //acc-=p.vel*.5;
    //acc-=p.vel*dot(p.vel,p.vel)*.1;
    
    float dt=.01;
    
    //neighbour damping
    acc-=(p.vel-nvel)/dt/10.;
    //acc-=(p.vel-nvel)*length(p.vel-nvel)/dt/100.;
    //p.vel=mix(p.vel,nvel,.1*cnt/6.);
    //p.vel=(p.vel*70.+nvel*cnt)/(cnt+70.);
    
    // proceed timestep dt
    p.vel+=acc*dt*.5;
    p.pos+=p.vel*dt;
    p.vel+=acc*dt*.5;
}

// Function 53
vec2 getForce(vec2 x, vec2 v) {
    vec2 force = vec2(0.0);
    
    if(iMouse.z > 0.5) {
        vec2 mouse = iMouse.xy / iResolution.xy * 2.0 - 1.0;
        mouse.x *= iResolution.x / iResolution.y;
        vec2 dir = x.xy - mouse;
        float p = length(dir);        
        force += 5.0 * normalize(dir) / p;
    }
    
    return force;
}

// Function 54
float Force(float d)
{
    return 0.2*exp(-0.05*d)-2.*exp(-0.5*d);
}

// Function 55
void BodyApplyGravity( inout Body body, float f )
{
    float fAccel_MpS = -9.81 * f;
    body.vForce.y += body.fMass * fAccel_MpS;
}

// Function 56
vec3 getSpringForce(vec3 vSpringPos, vec3 vConnexionPos) {
	// force of this string
    vec3 v = vSpringPos - vConnexionPos;
	float dLen = length(v);
    float dSpringF = dSpringK * clamp(dLen - dSpringLen, 0., dSpringLen*4.);
	return v * (dSpringF / dLen);
}

// Function 57
vec3 force_gravity(ivec2 i, vec3 pos, sampler2D iChannel) {
    // https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
    vec3 f = vec3(0);
    for (int j = 0; j < N; j++) {
        vec3 r = texelFetch(iChannel, ivec2(j, IPOS), 0).xyz - pos.xyz;
        f += M * r / pow(pow(length(r), 2.0) + S * S, 1.5);
    }
    return G * M * f;
}

// Function 58
float insideTriforce(vec3 pos, float aspect, out float u, out float v, out float w) {
	// 1st triangles - vertices
	vec3 v0 = vec3(0.5*aspect, 0.8, 1.0);
	vec3 v1 = v0 + vec3(-side*cosA, -side*sinA, 0.0);
	vec3 v2 = v1 + vec3(2.0 * (v0.x - v1.x), 0.0, 0.0);
	
	// test if inside 1st triangle
	barycentric(v0, v1, v2, pos, u, v, w);
	vec3 uvw = vec3(u,v,w);
	vec3 inside = step(zero, uvw) * (1.0 - step(one, uvw));

	if (all_set(inside))
		return 1.0;

	// 2nd triangles - vertices
	float dx = v1.x - v0.x;	// half-side in x
	float dy = v1.y - v0.y;	// half-side in y
	v0 -= vec3(-dx, -dy, 0.0);
	v1 = v0 + vec3(-side*cosA, -side*sinA, 0.0);
	v2 = v1 + vec3(2.0 * (v0.x - v1.x), 0.0, 0.0);	
	
	// test if inside 2nd triangle
	barycentric(v0, v1, v2, pos, u, v, w);
	uvw = vec3(u,v,w);
	inside = step(zero, uvw) * (1.0 - step(one, uvw));
	if (all_set(inside))
		return 1.0;
	
	// 3rd triangles - vertices	
	v0 += vec3(-dx*2.0, 0.0, 0.0);
	v1 = v0 + vec3(-side*cosA, -side*sinA, 0.0);
	v2 = v1 + vec3(2.0 * (v0.x - v1.x), 0.0, 0.0);	

	// test if inside 3rd triangle
	barycentric(v0, v1, v2, pos, u, v, w);
	uvw = vec3(u,v,w);
	inside = step(zero, uvw) * (1.0 - step(one, uvw));
	if (all_set(inside))
		return 1.0;
	
	return 0.0;
}

// Function 59
define SPRING(q, i_, j_, f_) { \
		float d = distance(p, q) \
    	, l = llen * length(vec2(i_,j_)); \
        po -= (q - p) * (d - l) * (f_) * dt; \
        /*p += (q - p) * (d - l) * (f_) * dt;*/ \
    }

// Function 60
bool drawSpring( vec2 coords, vec2 p0, vec2 p1, float thickness, int loops )
{
    vec2 d = p1 - p0;
    if (length(d) < 0.001)
        return false;
    
    vec2 dir = normalize(d);
    vec2 per = vec2(dir.y, -dir.x);
    
    vec2 st = d / float(loops * 2);
    vec2 last = p0 + per * thickness / 2.0 + st / 2.0;
    vec2 sw = -thickness * per;
    float th = 1.0;
    
    bool draw = drawLine(coords, p0, last, th);
    
    for (int i=0; i<loops*2-1; i++)
    {
        vec2 next = last + st + sw;
        sw = -sw;
        draw = draw || drawLine(coords, last, next, th);
        last = next;
    }
    
    draw = draw || drawLine(coords, last, p1, th);
    return draw;
}

// Function 61
void UpdateGravity(inout vec3 position, inout vec3 velocity)
{
    vec3 g = vec3(0.0, Gravity, 0.0) * iTimeDelta;
    vec3 p = (position + FeetPosition);
    
    float d = Scene(p);
    
    if(d <= 0.0)
    {
        position.y += (-d);
        velocity.y = 0.0;
    }
    else
    {
        velocity += g;
    }
}

// Function 62
vec3 PairForce (vec3 r)
{
  vec3 dr, f;
  float rSep;
  int nx, ny;
  f = vec3 (0.);
  nx = 0;
  ny = 0;
  for (int n = 0; n < nBallE * nBallE; n ++) {
    dr = r - GetR (vec2 (nx, ny));
    rSep = length (dr);
    if (rSep > 0.01 && rSep < 1.) f += fOvlap * (1. / rSep - 1.) * dr;
    if (++ nx == nBallE) {
      nx = 0;
      ++ ny;
    }  
  }
  return f;
}


// End module: forces_functions

// Begin module: integration_methods_functions
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


// End module: integration_methods_functions

// Begin module: particle_systems_functions
// Reusable Particle Systems Physics Functions
// Automatically extracted from particle/physics simulation-related shaders

// Function 1
void UpdateMaterial() {
    
}

// Function 2
vec3 getParticle( vec2 id )
{
    return texture( iChannel0, (id+0.5)/iResolution.xy).rgb;
}

// Function 3
vec2 getParticlePosition(int partnr)
{  
   time2 = time_factor*iTime;    
   pst = getParticleStartTime(partnr); // Particle start time
   plt = mix(part_life_time_min, part_life_time_max, random(float(partnr*2-35))); // Particle life time
   time4 = mod(time2 - pst, plt);
   time3 = time4 + pst;
   runnr = floor((time2 - pst)/plt);  // Number of the "life" of a particle    
    
   // Particle "local" time, when a particle is "reborn" its time starts with 0.0
   float part_timefact = mix(part_timefact_min, part_timefact_max, random(float(partnr*2 + 94) + runnr*1.5));
   float ptime = (runnr*plt + pst)*(-1./part_timefact + 1.) + time2/part_timefact;   
   vec2 ppos = vec2(harms(main_x_freq, main_x_amp, main_x_phase, ptime), harms(main_y_freq, main_y_amp, main_y_phase, ptime)) + middlepoint;
   
   // Particles randomly get away the main particle's orbit, in a linear fashion
   vec2 delta_pos = part_max_mov*(vec2(random(float(partnr*3-23) + runnr*4.), random(float(partnr*7+632) - runnr*2.5))-0.5)*(time3 - pst);
   
   // Calculation of the effect of the gravitation on the particles
   vec2 grav_pos = gravitation*pow(time4, 2.)/250.;
   return (ppos + delta_pos + grav_pos)*gen_scale;
}

// Function 4
void writeParticle( int idx, Particle p, inout vec4 fragColor, vec2 fragCoord)
{
    int xc = XC(idx);
    int yc = YC(idx);
    if (isPixel(xc+0,yc,fragCoord)) fragColor.xyzw=p.pos;
    if (isPixel(xc+1,yc,fragCoord)) fragColor.xyzw=p.vel;
    if (isPixel(xc+2,yc,fragCoord)) fragColor.xyzw=p.angVel;
    if (isPixel(xc+3,yc,fragCoord)) fragColor.xyzw=p.quat;
    // not sure if framebuffer has .w, so store quat.w also in next pixel
    if (isPixel(xc+4,yc,fragCoord)) fragColor.xyzw=vec4(p.quat.w,0,0,1);
}

// Function 5
void writeParticle(Particle p, inout vec4 color, vec2 coord)
{
    ivec2 res=textureSize(ParticleTex,0);
    //if(particleIdx(coord)!=p.idx) return;
    color=((int(coord.y)*res.x+int(coord.x))%3==0)?vec4(p.pos,p.idx):color;
    color=((int(coord.y)*res.x+int(coord.x))%3==1)?vec4(p.vel,p.sidx):color;
    color=((int(coord.y)*res.x+int(coord.x))%3==2)?vec4(p.nn[0],p.nn[1],p.nn[2],p.nn[3]):color;
    if(particleIdx(coord)>NumParticles) color=vec4(0,1,0,1);
}

// Function 6
Particle readParticle(int pIdx)
{
    Particle p;
    vec4 p0=getPixel(pIdx*3+0);
    vec4 p1=getPixel(pIdx*3+1);
    vec4 p2=getPixel(pIdx*3+2);
    p.pos=p0.xyz;
    p.idx=int(p0.w);
    p.sidx=int(p1.w);
    p.vel=p1.xyz;
    p.nn[0]=int(p2.x);
    p.nn[1]=int(p2.y);
    p.nn[2]=int(p2.z);
    p.nn[3]=int(p2.w);
    return p;
}

// Function 7
vec2 getParticlePosition(int partnr)
{  
   vec2 pos = vec2(mod(float(partnr+1), iResolution.x)/(iResolution.x+1.), (float(partnr)/(iResolution.x))/(iResolution.y+1.));
   return (texture(iChannel0, pos)).xy;
}

// Function 8
vec3 update(in vec3 vel, vec4 p, in float id) { 
    
    float n1a = fbm(p.xy, p.w);
    float n1b = fbm(p.yx, p.w);
    float nn = fbm(vec2(n1a,n1b),0.)*5.8 + .5;
    
    vec2 dir = vec2(cos(nn), sin(nn));
    vel.xy = mix(vel.xy, dir*1.5, 0.005);
    return vel;
}

// Function 9
void update_console()
{
    const float
        T0 = 0.,
    	T1 = T0 + CONSOLE_XFADE_DURATION,
    	T2 = T1 + CONSOLE_SLIDE_DURATION,
    	T3 = T2 + CONSOLE_TYPE_DURATION,
    	T4 = T3 + CONSOLE_SLIDE_DURATION;
    
    // snap console position to multiples of 2 pixels to avoid shimmering
    // due to the use of noise and dFd* functions
    float ysnap = iResolution.y * .5;
    
    g_console.loaded = linear_step(T0, T1, g_time);
    g_console.expanded = 1.+-.5*(linear_step(T1, T2, g_time) + linear_step(T3, T4, g_time));
    g_console.expanded = floor(g_console.expanded * ysnap + .5) / ysnap;
    g_console.typing = linear_step(0., CONSOLE_TYPE_DURATION, g_time - T2);
}

// Function 10
vec3 drawParticles(vec2 uv)
{  
    // Here the time is "stetched" with the time factor, so that you can make a slow motion effect for example
    time2 = time_factor*iTime;
    vec3 pcol = vec3(0.);
    // Main particles loop
    for (int i=1; i<nb_particles; i++)
    {
        pst = getParticleStartTime(i); // Particle start time
        plt = mix(part_life_time_min, part_life_time_max, random(float(i*2-35))); // Particle life time
        time4 = mod(time2 - pst, plt);
        time3 = time4 + pst;

        runnr = floor((time2 - pst)/plt);  // Number of the "life" of a particle
        vec2 ppos = getParticlePosition(i);
        float dist = distance(uv, ppos);
        if (dist<0.05) // When the current point is further than a certain distance, its impact is neglectable
        {
            // Draws the eight-branched star
            // Horizontal and vertical branches
            vec2 uvppos = uv - ppos;
            float distv = distance(uvppos*part_starhv_dfac + ppos, ppos);
            float disth = distance(uvppos*part_starhv_dfac.yx + ppos, ppos);
            // Diagonal branches
            vec2 uvpposd = 0.707*vec2(dot(uvppos, vec2(1., 1.)), dot(uvppos, vec2(1., -1.)));
            float distd1 = distance(uvpposd*part_stardiag_dfac + ppos, ppos);
            float distd2 = distance(uvpposd*part_stardiag_dfac.yx + ppos, ppos);
            // Initial intensity (random)
            float pint0 = mix(part_int_factor_min, part_int_factor_max, random(runnr*4. + float(i-55)));
            // Middle point intensity star inensity
            float pint1 = 1./(dist*dist_factor + 0.015) + part_starhv_ifac/(disth*dist_factor + 0.01) + part_starhv_ifac/(distv*dist_factor + 0.01) + part_stardiag_ifac/(distd1*dist_factor + 0.01) + part_stardiag_ifac/(distd2*dist_factor + 0.01);


            // Intensity curve and fading over time
            float pint = pint0*(pow(pint1, ppow)/part_int_div)*(-time4/plt + 1.);

            // Initial growing of the paricle's intensity
            pint*= smoothstep(0., grow_time_factor*plt, time4);
            // "Sparkling" of the particles
            float sparkfreq = clamp(part_spark_time_freq_fact*time4, 0., 1.)*part_spark_min_freq + random(float(i*5 + 72) - runnr*1.8)*(part_spark_max_freq - part_spark_min_freq);
            pint*= mix(part_spark_min_int, part_spark_max_int, random(float(i*7 - 621) - runnr*12.))*sin(sparkfreq*twopi*time2)/2. + 1.;

            // Adds the current intensity to the global intensity
            pcol+= getParticleColor(i, pint);
        }
    }
    // Main particle
    vec2 ppos = getParticlePosition_mp();
    float dist = distance(uv, ppos);
    //if (dist<0.25)
    //{
        // Draws the eight-branched star
        // Horizontal and vertical branches
        vec2 uvppos = uv - ppos;
        float distv = distance(uvppos*part_starhv_dfac + ppos, ppos);
        float disth = distance(uvppos*part_starhv_dfac.yx + ppos, ppos);
        // Diagonal branches
        vec2 uvpposd = 0.7071*vec2(dot(uvppos, vec2(1., 1.)), dot(uvppos, vec2(1., -1.)));
        float distd1 = distance(uvpposd*part_stardiag_dfac + ppos, ppos);
        float distd2 = distance(uvpposd*part_stardiag_dfac.yx + ppos, ppos);
        // Middle point intensity star inensity
        float pint1 = 1./(dist*dist_factor + 0.015) + part_starhv_ifac/(disth*dist_factor + 0.01) + part_starhv_ifac/(distv*dist_factor + 0.01) + part_stardiag_ifac/(distd1*dist_factor + 0.01) + part_stardiag_ifac/(distd2*dist_factor + 0.01);
        
        if (part_int_factor_max*pint1>6.)
        {
            float pint = part_int_factor_max*(pow(pint1, ppow)/part_int_div)*mp_int;
            pcol+= getParticleColor_mp(pint);
        }
    //}
    return pcol;
}

// Function 11
void UpdateJumpMovement(in vec3 position, inout vec3 velocity)
{
    float lastSpace = texelFetch(iChannel2, LocSpace, 0).r;
    float currSpace = IsKeyPressed(KEY_SPACE);
    
    if(Scene(position + FeetPosition) > 0.1)
    {
        // Can't jump in mid air
        return;
    }
    
    if(Scene(position + HeadPosition) < 0.0)
    {
        // Can't jump if something is directly above head
        return;
    }
    
    // If space bar was pressed down since last update
    if(currSpace > lastSpace)
    {
    	vec3 jvelocity = vec3(0.0, 1.0, 0.0) * JumpImpulse;
        vec3 jposition = position + jvelocity;
        
        velocity += jvelocity;
    }
}

// Function 12
void tUpdateAppendage(inout vec4 res,vec3 ro,vec3 rd,vec3 p, vec3 q, vec3 w, float A, float B,inout vec3 n_out)
{
    vec3 V=q-p;
    float D=length(V);
    float inverseD = 1.00 / D;
    V*= inverseD;
    
    vec3 W = normalize(w);
    vec3 U = cross(V, W);
    
    float A2 = A * A;
    
    float y = 0.5 * inverseD * (A2 - B * B + D * D);
    float square = A2 - y * y;
    if (square < 0.0) {
        return;
    }
    float x = sqrt(square);
    
    vec3 j = p+U*x+V*y;
    float ooA=  1.0 / 8.0;
    vec3 d= (j- p)*ooA;
    
    vec3 k = p;
    float mind=res.x;
    
    
    
    for(int i = 0; i <= countA; i++)
    {
        float fi=float(i);
        tSPH((k+d*fi),(2.5+2.5*fi*ooA),MAT_LIMBS);
    }
    
    d= (j- q )*ooA;
    k = q;
    for(int i = 0; i < countA; i++)
    {
        tSPH((k+d*float(i)),5.0,MAT_LIMBS);
    }
    
    
}

// Function 13
vec4 UpdateSampleCount( vec4 sampData )
{
    // adaptive frame rate
/*  if there's a run of slow frames, reduce num samples by desired proportion of frame time
    if there's a run of fast frames, creep num samples upwards, and step back when hit first slow frame (so stepping back hopefully hits ideal)
    e.g. -ve => frame count of successive fast frames, +ve => frame count of slow frames
    second value holds current number of samples
*/
    
    if ( iFrame == 0 )
    {
        sampData = vec4(MAX_SAMPLES/4,0,1./60.,1);
    }

#if defined( OFFLINE_RENDER )
    
    // render at a locked, reduced frame rate, so we can afford to have very slow frames
    sampData.x = float(MAX_SAMPLES); // maximum quality
    sampData.z = 1./60.; // final video fps
    
    float renderfps = .5; // actual FPS to render at (can be slower than this, but won't be faster)
// annoyingly shareX auto capture only goes up to 1 fps
    
    sampData.w = 0.; // don't draw
    
    if ( iTime*renderfps >= sampData.y/sampData.z ) // time*fps = frames, time/spf = frames
    {
        sampData.w = 1.; // draw
    	sampData.y += sampData.z; // replacement iTime value
    }
    
#else
        
	// time delta seems to be a bit unsteady, so create a stabilised version
    sampData.z = mix(sampData.z,min(iTimeDelta,1./15.),.25);

#if defined( ADAPTIVE_SAMPLE_COUNT )
    // count how many frames we've been running under/over 60 fps
    if ( sampData.z < 1./58. )
    {
        // faster than 60fps
        sampData.y = min(0.,sampData.y);
        sampData.y--;
        if ( sampData.y < -120. )
        {
            sampData.x++; // creep up slowly
            sampData.y = -90.; // next step up in 30 frames
        }
    }
    else
    {
        // slow frame
        if ( sampData.y < 0. && sampData.x > 3. ) // don't do this so quickly on really low sample counts
        {
            // first slow frame after some fast ones, undo our last creep up
            sampData.x--;
        }
        sampData.y = max(0.,sampData.y);
        sampData.y++;
        if ( sampData.y > 10. )
        {
            sampData.x = floor(sampData.x*(1./60.)/sampData.z); // shoot for an exact target
            sampData.y = 0.;
        }
    }
    
    sampData.x = clamp(sampData.x,1.,float(MAX_SAMPLES));
    
    sampData.w = 1.;
#else
    sampData.x = float(MAX_SAMPLES);
#endif

#endif // OFFLINE_RENDER
    
    return sampData;
}

// Function 14
float particles (vec3 p)
{
	vec3 pos = p;
	pos.y -= iTime*0.02;
	float n = fbm(20.0*pos);
	n = pow(n, 5.0);
	float brightness = noise(10.3*p);
	float threshold = 0.26;
	return smoothstep(threshold, threshold + 0.15, n)*brightness*90.0;
}

// Function 15
vec3 getParticleColor(int partnr, float pint)
{
   vec2 pos = vec2(mod(float(partnr+1), iResolution.x)/(iResolution.x+1.), (50. + float(partnr)/(iResolution.x))/(iResolution.y+1.));
   return (pint*texture(iChannel0, pos)).xyz;  
}

// Function 16
vec4 updateParticle(in vec4 particle) {
    vec2 v = particle.xy - particle.zw;
    
    vec2 pos = particle.xy;
    vec2 dv = pos - 0.5;
    float l = length(dv);
    
    vec2 a = -(dv / (0.001 + l)) * 0.00001 / (l * l + 1.1);
    v += a;
    
    if (particle.x + v.x < 0.0 || particle.x + v.x >= 1.0) {
        v.x = -v.x;
    }
    if (particle.y + v.y < 0.0 || particle.y + v.y >= 1.0) {
        v.y = -v.y;
    }
    
    float maxSpeed = 0.1;
    v = length(v) > maxSpeed ? maxSpeed * v / length(v) : v;
    
    particle.zw = particle.xy;
    particle.xy += v;
        
    return particle;
}

// Function 17
vec4 getParticle(ivec2 id)
{
    return texelFetch(iChannel0, id, 0); //texture(iChannel0, (id+.5)/iResolution.xy);
}

// Function 18
vec4 UpdateForward(vec4 o){return vec4(Rotate(GetLastRotation(),vec3(0,0,1)),.0);}

// Function 19
vec4 getRandomParticle2(in vec3 iResolution, in sampler2D iChannel0, in vec2 fragCoord, in int frame, int i) {
	return texelFetch(iChannel0, getRandomParticlePos(iResolution, iChannel0, fragCoord, frame, i), 0);
}

// Function 20
vec2 getParticlePosition(int partnr)
{  
   // Particle "local" time, when a particle is "reborn" its time starts with 0.0
   float part_timefact = mix(part_timefact_min, part_timefact_max, random(float(partnr*2 + 94) + runnr*1.5));
   float ptime = (runnr*plt + pst)*(-1./part_timefact + 1.) + time2/part_timefact;   
   vec2 ppos = vec2(harms(main_x_freq, main_x_amp, main_x_phase, ptime), harms(main_y_freq, main_y_amp, main_y_phase, ptime)) + middlepoint;
   
   // Particles randomly get away the main particle's orbit, in a linear fashion
   vec2 delta_pos = part_max_mov*(vec2(random(float(partnr*3-23) + runnr*4.), random(float(partnr*7+632) - runnr*2.5))-0.5)*(time3 - pst);
   
   // Calculation of the effect of the gravitation on the particles
   vec2 grav_pos = gravitation*pow(time4, 2.)/250.;
   return (ppos + delta_pos + grav_pos)*gen_scale;
}

// Function 21
void UpdateSoldier( inout vec4 soldier, inout vec4 soldierState, vec4 playerHitBox, float screenWidth, float screenHeight )
{
    float soldierSupport = GetSupport( soldier.xy );    
    if ( soldierState.x == STATE_RUN )
    {
		soldierState.y = mod( soldierState.y + ENEMY_ANIM_SPEED, 2.0 );        
        
        if ( soldier.y != soldierSupport )
        {
            // lost support - either jump or go back
            if ( Rand() > 0.3 )
            {
            	soldierState.x = STATE_JUMP;
            	soldierState.y = 1.0;
            	soldierState.z = 0.0;
			}
            else
            {
            	soldier.z = -soldier.z;
            }
        }
    }
    else if ( soldierState.x == STATE_JUMP )
    {
		soldierState.z += 1.0 / 20.0;
        soldier.y += 3.0 * ( 1.0 - soldierState.z );
        if ( soldierState.z > 1.0 && soldier.y <= soldierSupport )
        {
            soldier.y = soldierSupport;
            soldierState.x = STATE_RUN;
        }
    }
	soldier.x += soldier.z * ENEMY_RUN_SPEED;

    if ( soldier.x > gCamera.x + screenWidth || soldier.x < gCamera.x )
    {
    	soldier.x = -1.0;        
    }

    // soldier death
    if ( soldier.x > 0.0 && soldier.y < WATER_HEIGHT )   
    {
        gExplosion 	= vec4( soldier.xy + vec2( 0.0, SOLDIER_SIZE.y * 0.5 ), 0.0, 0.0 );
		soldier 	= vec4( 0.0, 0.0, 0.0, 0.0 );
    }
    
	if ( soldier.x > 0.0 && Collide( playerHitBox.xy, playerHitBox.zw, soldier.xy, SOLDIER_SIZE ) )
    {
        PlayerHit( playerHitBox );
    }    
}

// Function 22
void UpdateParticle()
{
    vec3 g = -1e-5*O.X; 
    vec3 F = g; 
    
    float scale = 0.21/density; //radius of smoothing
    float Rho = Kernel(0., scale);
    float avgP = 0.;
	vec3  avgC = vec3(O.Color);

    loop(j,6)
    {
        vec4 nb = texel(ch0, i2xy(ivec3(ID, j, 1)));
        loop(i,3)
        {
            if(nb[i] < 0. || nb[i] > float(TN)) continue;
            obj nbO = getObj(int(nb[i]));

            float d = distance(O.X, nbO.X);
            vec3 dv = (nbO.V - O.V); //delta velocity
            vec3 dx = (nbO.X - O.X); //delta position 
            vec3 ndir = dx/(d+1e-3); //neighbor direction
            //SPH smoothing kernel
            float K = Kernel(d, scale);

            vec3 pressure = -0.5*( nbO.Pressure/sqr(nbO.Rho) + 
                                     O.Pressure/sqr(O.Rho) )*ndir*K;//pressure gradient
            vec3 viscosity = 3.*ndir*dot(dv,ndir)*K;
           
            Rho += K;
            avgC += nbO.Color;
            avgP += nbO.Pressure*K;

            F += pressure + viscosity;
        }
    }

    O.Rho = Rho;
    
    O.Scale = scale; //average distance
    
    float r = 7.;
    float D = 1.;
    float waterP = 0.08*density*(pow(abs(O.Rho/density), r) - D);
    O.Pressure = min(waterP,0.03);

    O.V += F*dt;
    O.V -= O.V*(0.5*tanh(8.*(length(O.V)-1.5))+0.5);
    O.X += O.V*dt; //advect

    //color diffusion

    //O.Color = ;
}

// Function 23
Particle updateParticle(Particle p){
    vec2 acc = vec2(0.);
    for(int i = 0; i < Planets; i++){
    	Particle po = particles[i];
    	acc += (po.pos - p.pos) * grav(p,po);
    }
    p.vel += acc / p.mass * delta;
    p.pos += p.vel * delta;
    return p;
}

// Function 24
bool isUpdateNeeded(ivec4 cell)
{
    return cell.w == 0;
}

// Function 25
void gs_update_camera( inout GameState gs, VehicleState vs )
{
    vec2 sc = sincospi( gs.mouselook.z / PI );
    vec3 forward = vec3( gs.mouselook.xy * sc.y, sc.x );
    vec3 right = normalize( cross( UNIT_Z, forward ) );
    vec3 down = cross( forward, right );
    gs.camframe = mat3( forward, right, down );
    if( ( gs.switches & GS_TRMAP ) == 0u )
    {
        if( keypress( KEY_R ) == 1. )
			gs.camzoom = gs.camzoom < 3. ? 3. : gs.camzoom < 8. ? 8. : 1.;
        gs.campos_diff = vs.localB * vec3( .001, 0, -.0015 );
    	gs.campos = vs.localr + gs.campos_diff;
    	gs.camframe = vs.localB * gs.camframe;
    }
}

// Function 26
vec4 UpdateSampleCount( vec4 sampData )
{
    // adaptive frame rate
/*  if there's a run of slow frames, reduce num samples by desired proportion of frame time
    if there's a run of fast frames, creep num samples upwards, and step back when hit first slow frame (so stepping back hopefully hits ideal)
    e.g. -ve => frame count of successive fast frames, +ve => frame count of slow frames
    second value holds current number of samples
*/
    
    if ( iFrame == 0 )
    {
        sampData = vec4(MAX_SAMPLES,0,1./60.,0);
    }
    
    // time delta seems to be a bit unsteady, so create a stabilised version
    sampData.z = mix(sampData.z,min(iTimeDelta,1./15.),.25);

    // count how many frames we've been running under/over 60 fps
    if ( sampData.z < 1./58. )
    {
        // faster than 60fps
        sampData.y = min(0.,sampData.y);
        sampData.y--;
        if ( sampData.y < -120. )
        {
            sampData.x++; // creep up slowly
            sampData.y = -90.; // next step up in 30 frames
        }
    }
    else
    {
        // slow frame
        if ( sampData.y < 0. && sampData.x > 3. ) // don't do this so quickly on really low sample counts
        {
            // first slow frame after some fast ones, undo our last creep up
            sampData.x--;
        }
        sampData.y = max(0.,sampData.y);
        sampData.y++;
        if ( sampData.y > 10. )
        {
            sampData.x = floor(sampData.x*(1./60.)/sampData.z); // shoot for an exact target
            sampData.y = 0.;
        }
    }
    
    sampData.x = clamp(sampData.x,1.,float(MAX_SAMPLES));
    
    return sampData;
}

// Function 27
Particle decodeParticle(vec4 c, ivec2 cell){
    vec2  position = unpackUnorm2x16(floatBitsToUint(c.x))+vec2(cell);
    vec2  velocity = unpackUnorm2x16(floatBitsToUint(c.y))*2.*maxvelocity-maxvelocity;
    float mass = c.z;
    return Particle(position,velocity,mass);
}

// Function 28
vec2 particle(float t, float xv, float yv) {
    t = mod(t, 1.94152);
    vec2 p = vec2(x(t, xv)-0.2*xv, y(t, yv));
    return p;
}

// Function 29
vec3 updatePos(in vec3 prevPos, in vec3 dir)
{
    if (iFrame == 0)
        return vec3(0.0);
    
    vec3 upDir = vec3(0.0, 1.0, 0.0);
    vec3 rightDir = normalize(cross(upDir, dir));
    
    vec3 pos = prevPos;
    
    if (isPressed(KEY_UP) || isPressed(KEY_W))
        pos += dir * iTimeDelta;
    
    if (isPressed(KEY_DOWN) || isPressed(KEY_S))
        pos -= dir * iTimeDelta;
    
    if (isPressed(KEY_RIGHT) || isPressed(KEY_D))
        pos += rightDir * iTimeDelta;
    
    if (isPressed(KEY_LEFT) || isPressed(KEY_A))
        pos -= rightDir * iTimeDelta;
    
    // Move up
    if (isPressed(KEY_SPACE) || isPressed(KEY_SHIFT))
        pos += upDir * iTimeDelta;
    
    // Move down
    if (isPressed(KEY_C) || isPressed(KEY_CTRL))
        pos -= upDir * iTimeDelta;
    
    return pos;
}

// Function 30
void updateRank2x(particle n, inout vec4 O, inout float s0, inout float s1, vec2 I, vec3 R,int seed){
    float sn = score(n,I,R,seed);
    if(sn<s0){
        //Shift down the line
        s1=s0;
        O.zw=O.xy;
        s0=sn;
        O.xy=packParticle(n);
    } else if(sn<s1){
        //Bump off the bottom one
        s1=sn;
        O.zw=packParticle(n);
        
    }
}

// Function 31
Particle readParticle( int idx )
{
    Particle p;
    int xc = XC(idx);
    int yc = YC(idx);
    // first line (y=0) reserved (e.g. for growNum, growIdx)
    p.pos    = getPixel(xc+0,yc);
    p.vel    = getPixel(xc+1,yc);
    p.angVel = getPixel(xc+2,yc);
    p.quat   = getPixel(xc+3,yc);
    // not sure if framebuffer has .w, so store quat.w also in next pixel
    vec4 p2  = getPixel(xc+4,yc);
    p.quat.w = p2.x;
    return p;
}

// Function 32
void gui_decor_update() {
    
    if (fc.x != DECOR_COL) { return; }
    
    for (int i=0; i<4; ++i) {
        if (box_dist(iMouse.xy, decor_ui_box(i)) < 0.) {
            data[i] = 1. - data[i];
        }
    }
    
}

// Function 33
void Entity_UpdateSlideBox( inout Entity entity, float fTimestep )
{
    entity.fYaw += entity.fYawVel * fTimestep;

    const float fStepHeight = 24.1; // https://www.doomworld.com/vb/doom-general/67054-maximum-height-monsters-can-step-on/
    const float fClearanceHeight = 32.;
    
    float fDropOff = 10000.0;
    if ( entity.iType == ENTITY_TYPE_ENEMY )
    {
        // Enemies 
        fDropOff = 24.0;
    }
    entity.vVel.xz *= fTimestep;
    SlideVector( entity.iSectorId, entity.vPos.xz, entity.vVel.xz, entity.vPos.y + fStepHeight, entity.vPos.y + fClearanceHeight, fDropOff );
    entity.vVel.xz /= fTimestep;
}

// Function 34
float particlesLayer(vec2 uv, float seed)
{
   	uv = uv + hash21(seed) * 10.0;
    vec2 rootUV = floor(uv);
    vec2 particleUV = particleCoordFromRootUV(rootUV);
    float particles = particleFromParticleUV(particleUV, uv);
    return particles;
}

// Function 35
void update_ideal_pitch(vec3 pos, vec3 forward, vec3 velocity, inout float ideal_pitch)
{
    if (iMouse.z > 0. || length_squared(velocity.xy) < sqr(WALK_SPEED/4.))
        return;
    
    if (dot(forward, normalize(velocity)) < .7)
    {
        ideal_pitch = 0.;
        return;
    }
    
    // look up/down near stairs
    // totally ad-hoc, but it kind of works...
	const vec3 STAIRS[] = vec3[](vec3(272, 496, 24), vec3(816, 496, 24));

    vec3 to_stairs = closest_point_on_segment(pos, STAIRS[0], STAIRS[1]) - pos;
    float sq_dist = length_squared(to_stairs);
    if (sq_dist < sqr(48.))
        return;
    
    float facing_stairs = dot(to_stairs, forward);
    if (sq_dist > (facing_stairs > 0. ? sqr(144.) : sqr(64.)))
    {
        ideal_pitch = 0.;
        return;
    }
    
    if (facing_stairs * inversesqrt(sq_dist) < .7)
        return;

    ideal_pitch = to_stairs.z < 0. ? -STAIRS_PITCH : STAIRS_PITCH;
}

// Function 36
void UpdateIfIntersected(
    inout float t,
    in float intersectionT, 
    in int intersectionObjectID,
    out int objectID)
{    
    if(intersectionT > EPSILON && intersectionT < t)
    {
        objectID = intersectionObjectID;
        t = intersectionT;
    }
}

// Function 37
vec4 getParticle(int id){
    return texelFetch(iChannel0, locFromID(id), 0);
}

// Function 38
vec3 particle(vec2 st, vec2 p, float r, vec3 col){
 	float d = length(st-p);
    d = smoothstep(r, r-2.0/iResolution.y, d);//d<r?1.0:0.0;
    return d*col;
}

// Function 39
vec4 UpdatePosition( vec3 resetValue )
{
    Camera cam = GetCamera();
    
    cam.pos += cam.i*1.*iTimeDelta*(Key(kD)-Key(kA)+Key(kRight)-Key(kLeft));
    cam.pos += cam.j*1.*iTimeDelta*(Key(kR)-Key(kF));
    cam.pos += cam.k*1.*iTimeDelta*(Key(kW)-Key(kS)+Key(kUp)-Key(kDown));
    
    return reset?vec4(resetValue,0):vec4( cam.pos, 0 );
}

// Function 40
void updateCamera( int strata ) {
    float strataSize = 1.0/float(PIXEL_SAMPLES);
    float r1 = strataSize*(float(strata)+rnd());
    //update camera pos
#ifdef OCULUS_VERSION
	float cameraZ = -1.0;
#else
    float cameraZ = 4.0;
#endif
    vec3 upDir = vec3( 0.0, 1.0, 0.0 );
    vec3 pos1, pos2;
    pos1 = vec3( sin(frameSta*0.154)*2.0, 2.0 + sin(frameSta*0.3)*2.0, cameraZ + sin(frameSta*0.8) );
    pos2 = vec3( sin(frameEnd*0.154)*2.0, 2.0 + sin(frameEnd*0.3)*2.0, cameraZ + sin(frameEnd*0.8) );
    camera.pos = mix( pos1, pos2, r1 );
    
    pos1 = vec3( sin(frameSta*0.4)*0.3, 1.0, -5.0 );
    pos2 = vec3( sin(frameEnd*0.4)*0.3, 1.0, -5.0 );
    camera.target = mix( pos1, pos2, r1 );
    
	vec3 back = normalize( camera.pos-camera.target );
	vec3 right = normalize( cross( upDir, back ) );
	vec3 up = cross( back, right );
    camera.rotate[0] = right;
    camera.rotate[1] = up;
    camera.rotate[2] = back;
}

// Function 41
void writeParticle( int idx, Particle p, inout vec4 fragColor, vec2 fragCoord)
{
    int xc = XC(idx);
    int yc = YC(idx);
    if (isPixel(xc+0,yc,fragCoord)) fragColor.xyzw=p.pos;
    if (isPixel(xc+1,yc,fragCoord)) fragColor.xyzw=p.quat;
    // not sure if framebuffer has .w, so store quat.w also in next pixel
    if (isPixel(xc+2,yc,fragCoord)) fragColor.xyzw=vec4(p.size,p.parent,p.quat.w,1);
}

// Function 42
void Enemy_UpdateState(  inout Entity entity )
{
    int iState = Enemy_GetState( entity );
    
    if( entity.fHealth <= 0.0 )
    {
        Enemy_SetState( entity, ENEMY_STATE_DIE );
        iState = ENEMY_STATE_DIE;
    }

    if ( iState == ENEMY_STATE_DIE )
    {
        if ( entity.fTimer == 0. )
        {            
            entity.iType = ENTITY_TYPE_DECORATION;
            if ( entity.iSubType == ENTITY_SUB_TYPE_ENEMY_TROOPER )
            {
            	entity.iSubType = ENTITY_SUB_TYPE_DECORATION_DEAD_TROOPER;
    		}
            else
            if ( entity.iSubType == ENTITY_SUB_TYPE_ENEMY_SERGEANT )
            {
            	entity.iSubType = ENTITY_SUB_TYPE_DECORATION_DEAD_SERGEANT;
    		}
            else
            if ( entity.iSubType == ENTITY_SUB_TYPE_ENEMY_IMP )
            {
            	entity.iSubType = ENTITY_SUB_TYPE_DECORATION_DEAD_IMP;
    		}
            else
            {
            	entity.iSubType = ENTITY_SUB_TYPE_DECORATION_BLOODY_MESS;            
            }
        }
        
        return;
    }    
     
    // Check if can see player    
    if ( int(entity.fTarget) == ENTITY_NONE )
    {        
		Entity playerEnt = Entity_Read( STATE_CHANNEL, 0 );
        
        bool wakeUp = false;

        if ( Enemy_CanSee( entity, playerEnt ) )
        {
			wakeUp = true;
        }   

        // Wake if player firing weapon
        if ( !wakeUp )
        {
        	if ( FlagSet( playerEnt.iFrameFlags, ENTITY_FRAME_FLAG_FIRE_WEAPON ) )
            {
                if ( Entity_CanHear( entity, playerEnt ) )
                {
	                wakeUp  = true;
				}
            }            
        }

        if ( wakeUp )
        {
            // target player 
            entity.fTarget = 0.;
        	Enemy_SetState( entity, ENEMY_STATE_STAND );
            iState = ENEMY_STATE_STAND;            
        }
    }
    
    
    if ( iState == ENEMY_STATE_IDLE )
    {
    }
	else
    if ( iState == ENEMY_STATE_PAIN )
    {
        if ( entity.fTimer == 0. )
        {
            Enemy_SetState( entity, ENEMY_STATE_STAND );
        }
    }
	else
    if ( 	iState == ENEMY_STATE_STAND ||
        	iState == ENEMY_STATE_FIRE ||
        	iState == ENEMY_STATE_WALK_TO_TARGET ||
        	iState == ENEMY_STATE_WALK_RANDOM
       )
    {
        if ( int(entity.fTarget) != ENTITY_NONE )
        {
            Entity targetEnt = Entity_Read( STATE_CHANNEL, int(entity.fTarget) );

            if ( targetEnt.fHealth <= 0.0 )
            {
                entity.fTarget = float( ENTITY_NONE );
                Enemy_SetState( entity, ENEMY_STATE_IDLE );
            }
        }
        
        if ( entity.fTimer == 0. )
        {
            if ( iState == ENEMY_STATE_FIRE )
            {
	            Enemy_SetRandomHostileState( entity, true );
            }
            else
            {
	            Enemy_SetRandomHostileState( entity, false );
            }                
        }
    }        
}

// Function 43
AppState updateGame( AppState s, float isDemo )
{
    if ( isDemo > 0.0 )
    {
        s.timeAccumulated += 1.0 * iTimeDelta;
    	s.playerPos.y = 5.0 * s.timeAccumulated;
    }
    else
    {
        float playerCellID = floor( s.playerPos.y );
        s.paceScale = saturate( ( playerCellID - 50.0) / 500.0);
        float timeMultiplier = mix( 0.75, 2.0, pow( s.paceScale, 1.0 ) );

        s.timeAccumulated += timeMultiplier * iTimeDelta;
        s.playerPos.y = 5.0 * s.timeAccumulated;
    }    
    
    float playerCellID = floor( s.playerPos.y );

    if ( isDemo > 0.0 )
    {           
        float cellOffset = 1.0;
        float nextPlayerCellID = playerCellID + cellOffset;

        float nextCellCoinRND = hash11( nextPlayerCellID + s.seed ); // skip rnd obstacle every second cell to make room for driving
        nextCellCoinRND *= mix( 1.0, -1.0, step( mod( nextPlayerCellID, 4.0 ), 1.5 ) ); // gaps in coin placing: 2 gaps, 2 coins
        nextCellCoinRND = mix( nextCellCoinRND, -1.0, step( nextPlayerCellID, 5.0 ) ); // head start
        float nextCellCoinCol = floor( 3.0 * nextCellCoinRND );

        // OBSTACLE
        float nextCellObsRND = hash11( 100.0 * nextPlayerCellID + s.seed );
        nextCellObsRND *= mix( 1.0, -1.0, step( mod( nextPlayerCellID, 3.0 ), 1.5 ) );
        nextCellObsRND = mix( nextCellObsRND, -1.0, step( nextPlayerCellID, 7.0 ) ); // head start
        float nextCellObsCol = floor( 3.0 * nextCellObsRND );
        
        float inputObs = 0.0;                
        if ( nextCellObsCol > -0.5 )
        {
            nextCellCoinCol -= 0.5; // pos fix
        	float toObs = nextCellObsCol - s.playerPos.x;
        
            if ( nextCellObsCol == 1.0 )
                inputObs = hash11( nextPlayerCellID + s.seed );
            
            if ( nextCellObsCol < 1.0 )
                inputObs = 1.0;

            if ( nextCellObsCol > 1.0 )
                inputObs = -1.0;
        }
        
        
        float inputCoin = 0.0;
        if ( nextCellCoinCol > -0.5 )
        {               
            nextCellCoinCol -= 0.5; // pos fix
            float toCoin = nextCellCoinCol - s.playerPos.x;
            
			inputCoin = sign(toCoin) * saturate( abs( toCoin ) );
        }

        float inputDir = inputCoin + 5.0 * inputObs;
        inputDir = sign( inputDir ) * 4.0 * saturate( abs( inputDir ) );
        
        s.isPressedLeft  = step( 0.5, -inputDir );
        s.isPressedRight = step( 0.5,  inputDir );
    }

    float speed = mix( 0.1, 0.15, isDemo );
    s.playerPos.x -= speed * s.isPressedLeft; 
    s.playerPos.x += speed * s.isPressedRight; 

    s.playerPos.x = clamp( s.playerPos.x, -0.5, 1.5 );

    if ( playerCellID != s.coin0Pos ) 
    {
        s.coin3Pos 	 = s.coin2Pos;
        s.coin3Taken = s.coin2Taken;

        s.coin2Pos 	 = s.coin1Pos;
        s.coin2Taken = s.coin1Taken;

        s.coin1Pos 	 = s.coin0Pos;
        s.coin1Taken = s.coin0Taken;

        s.coin0Pos = playerCellID;
        s.coin0Taken = 0.0;
    }
 
    // COIN start
    float cellCoinRND = hash11( playerCellID + s.seed ); // skip rnd obstacle every second cell to make room for driving
    cellCoinRND *= mix( 1.0, -1.0, step( mod( playerCellID, 4.0 ), 1.5 ) ); // gaps in coin placing: 2 gaps, 2 coins
    cellCoinRND = mix( cellCoinRND, -1.0, step( playerCellID, 5.0 ) ); // head start
    float cellCoinCol = floor( 3.0 * cellCoinRND );

    vec2 coinPos = -vec2( 0.0, playerCellID )	// cell pos
        +vec2( 0.5, -0.5 )	// move to cell center
        -vec2( cellCoinCol, 0.0 ); // move to column

    if ( cellCoinRND >= 0.0 )
    {        
        float distCoinPlayer = length( coinPos + s.playerPos );

        if ( distCoinPlayer < 0.5 && s.coin0Taken < 0.5 )
        {
            if ( isDemo < 1.0 )
            	s.score++;
            
            s.coin0Taken = 1.0;
            s.timeCollected = iTime;
        }
    }
    // COIN end

    // OBSTACLE start
    float cellObsRND = hash11( 100.0 * playerCellID + s.seed );
    cellObsRND *= mix( 1.0, -1.0, step( mod( playerCellID, 3.0 ), 1.5 ) );
    cellObsRND = mix( cellObsRND, -1.0, step( playerCellID, 7.0 ) ); // head start
    float cellObsCol = floor( 3.0 * cellObsRND );

    if ( cellObsRND >= 0.0 && cellObsCol != cellCoinCol )
    {   
        vec2 obstaclePos = -vec2( 0.0, playerCellID )	// cell pos
            +vec2( 0.5, -0.25 )	// move to cell center
            -vec2(cellObsCol, 0.0 ); // move to column

        float distObstaclePlayer = length( obstaclePos + s.playerPos );

        if ( distObstaclePlayer < 0.5 && isDemo < 1.0 )
        {
            s.timeFailed = iTime;
            s.timeCollected = -1.0;
            s.highscore = max( s.highscore, s.score );
        }
    }
    // OBSTACLE end        
    return s;
}

// Function 44
vec4 configureAndUpdate(vec3 res, sampler2D sampler, vec4 mouse, bool init) {
    configure(res);
    if (!init) {
    	updateControls(sampler);
    }
    
    return processMouse(mouse);
}

// Function 45
Particle getParticle(ivec2 cell){
	return decodeParticle(texelFetch(iChannel0, cell, 0),cell);
}

// Function 46
void gui_dfunc_update() {
    
    if (!(fc.x == DFUNC0_COL || fc.x == DFUNC1_COL)) { return; }
        
    bool is_linked = (load(MISC_COL, TARGET_ROW).x != 0.);

    for (int row=0; row<2; ++row) {  

        int col_for_row = (row == 0 ? DFUNC0_COL : DFUNC1_COL);

        for (int i=0; i<5; ++i) {

            bool update = ( (is_linked && fc.x == DFUNC1_COL) || 
                           (!is_linked && fc.x == col_for_row) );

            if (update) {

                if (box_dist(iMouse.xy, dfunc_ui_box(i, row)) < 0.) {
                    data = vec4(0);
                    if (i > 0) { data[i-1] = 1.; }
                }

            }
        }

    }

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
vec4 encodeParticle(Particle p, ivec2 cell){
	float x = uintBitsToFloat(packUnorm2x16(p.position-vec2(cell)));
    float y = uintBitsToFloat(packUnorm2x16((p.velocity+maxvelocity)/(2.*maxvelocity)));
    float z = p.mass;
    return vec4(x,y,z,0.);
}

// Function 49
Particle GrowNewParticle(int index)
{
    return GrowParticle(index, -1, 0.1);
}

// Function 50
void initParticle2(in int particleID, in vec2 fragCoord, inout vec2 particlePrevPosition, inout vec2 particleCurrPosition)
{
    int start = stars * (particleID / stars);
    vec2 seed = vec2(float(start), 1.0);
	particleCurrPosition = randVec2(seed) * iResolution.xy;
    particleCurrPosition += randVec2(fragCoord) * iResolution.xy / 20.0;
    particlePrevPosition = particleCurrPosition - randNrm2(seed) * particlesSize * 0.0625;
}

// Function 51
void update(in float time) {
    lit1.r = .26 + sin(time * 2.) * .25;
    lit1.l = vec3(2. * sin(time), cos(time), 2. * cos(time));
    lit2.r = .26 + cos(time * 2.) * .25;
    lit2.l = vec3(-2. * sin(time), cos(time), -2. * cos(time));
}

// Function 52
float sdf_particle(vec2 uv){
    vec2 particle_position = parametric_curve(iTime * PARTICLE_SPEED);

    float sdist_particle = sdf_disk(uv, particle_position, PARTICLE_RADIUS);
    return sdist_particle;
}

// Function 53
void densityUpdate (in vec2 offset) {

	// Get the position of the cell
	vec2 cellPosition = floor (particlePosition + offset) + 0.5;

	// Get the particle ID
	vec2 particleId = texture (iChannel2, cellPosition / iResolution.xy).rg;

	// Check whether there is a particle here
	if (offset == vec2 (0.0)) {

		// This is the current particle
		particleIdCheck = particleId;
	} else if (particleId.x > 0.0) {

		// Get the position of this other particle
		vec2 otherParticlePosition = texture (iChannel1, particleId / iResolution.xy).ba;

		// Check whether these 2 particles touch each other
		float dist = length (otherParticlePosition - particlePosition);
		if (dist < 2.0 * PARTICLE_RADIUS) {

			// Compute the density
			float compression = 1.0 - dist / (2.0 * PARTICLE_RADIUS);
			particleDensity += compression * compression * compression;
		}
	}
}

// Function 54
void UpdateRotation(inout vec4 fragColor)
{
    if(GetLastMouseClick().z < 0.5)
    {
        return;
    }
    
    vec4 lastQuat    = GetLastRotation();
    vec2 mouseDelta  = GetLastMouseDelta().xy * CameraSensitivity;
    
    vec3 forward = GetLastForward();
    vec3 right   = normalize(cross(forward, Up));
    vec3 up      = normalize(cross(right, forward));
    
#ifdef INVERT_Y
    lastQuat = QxQ(Quat(up, -mouseDelta.x), lastQuat);
    lastQuat = QxQ(Quat(right, -mouseDelta.y), lastQuat);
#else
    lastQuat = QxQ(Quat(up, -mouseDelta.x), lastQuat);
    lastQuat = QxQ(Quat(right, mouseDelta.y), lastQuat);
#endif
    
    fragColor = lastQuat;
}

// Function 55
Body update_body(Body p, float dt) {
    // Calculate force
    vec2 F = vec2(0);
    F += normalize(A1 - p.pos) * G * (M_A1 * M_BODY) / (d2_3d(vec3(p.pos, 0), vec3(A1, Z_DISTANCE)));
    F += normalize(A2 - p.pos) * G * (M_A2 * M_BODY) / (d2_3d(vec3(p.pos, 0), vec3(A2, Z_DISTANCE)));
    F += normalize(A3 - p.pos) * G * (M_A3 * M_BODY) / (d2_3d(vec3(p.pos, 0), vec3(A3, Z_DISTANCE)));
    
    // Update acceleration, position and velocity
    p.acc = F / M_BODY;
    p.pos = p.pos + p.vel * dt + 0.5 * p.acc * dt * dt;
    p.vel = p.vel + p.acc * dt;

    return p;
}

// Function 56
void UpdatePlayerBullet( inout vec4 playerBullet, float screenWidth, float screenHeight )
{
    if ( !Collide( playerBullet.xy, BULLET_SIZE, vec2( gCamera.x + screenWidth * 0.5, 0.0 ), vec2( screenWidth, screenHeight ) ) )
    {
        playerBullet.x = 0.0;
    }
    if ( playerBullet.x > 0.0 )
    {
    	playerBullet.xy += playerBullet.zw * PLAYER_BULLET_SPEED;
    }
}

// Function 57
void UpdateWASDMovement(in vec3 position, in vec3 forward, inout vec3 velocity)
{
    forward = normalize(vec3(forward.x, 0.0, forward.z));
    vec3 right = cross(forward, vec3(0.0, 1.0, 0.0));
    
    float keyForward = max(IsKeyPressed(KEY_W), IsKeyPressed(KEY_UP));
    float keyDown    = max(IsKeyPressed(KEY_S), IsKeyPressed(KEY_DOWN));
    float keyLeft    = max(IsKeyPressed(KEY_A), IsKeyPressed(KEY_LEFT));
    float keyRight   = max(IsKeyPressed(KEY_D), IsKeyPressed(KEY_RIGHT));
    
    float pressForward = keyForward - keyDown;
    float pressRight   = keyLeft - keyRight;
    
    vec3  direction = normalize((forward * pressForward) + (-right * pressRight));
    float delta     = MoveSpeed * step(0.001, length(direction));
    
    if(delta > 0.0)
    {
        delta *= max(1.0, IsKeyPressed(KEY_SHIFT) * BoostModifier);
        
        vec3 moveVelocity = velocity + (direction * delta);
    	vec3 bposition = position + (moveVelocity * iTimeDelta);
        
        if(Scene(bposition) < (PlayerBounds * 0.25))
        {
            // Continue moving against the normal to prevent sticky walls
            vec3 normal = SceneNormal(bposition);
            moveVelocity.xz *= vec2(1.0) - abs(normal.xz);
            
            bposition = position + (moveVelocity * iTimeDelta);
            
            // Make sure the slide doesnt push us into a wall (prevent slipping into corners)
            if(Scene(bposition) < (PlayerBounds * 0.25))
            {
                moveVelocity = vec3(0.0);
            }
        }
        
        velocity = moveVelocity;
    }
}

// Function 58
void update_time(float bake_time, float uniform_time)
{
    if (bake_time > 0.)
    	g_time = uniform_time - bake_time;
    else
        g_time = -uniform_time;
}

// Function 59
vec4 drawParticles(in vec3 ro, in vec3 rd, in float ints)
{
    vec4 rez = vec4(0);
    vec2 w = 1./iResolution.xy;
    
    for (int i = 0; i < numParticles; i++)
    {
        vec3 pos = texture(iChannel0, vec2(i,100.0)*w).rgb;
        vec3 vel = texture(iChannel0, vec2(i,0.0)*w).rgb;
        
        float st = sin(time*0.6);
        
        for(int j = 0; j < stepsPerFrame; j++)
        {
            float d = mag((ro + rd*dot(pos.xyz - ro, rd)) - pos.xyz);
            d *= 1000.;
            d = 2./(pow(d,1.+ sin(time*0.6)*0.15)+1.5);
            d *= (st+4.)*.8;

            rez.rgb += d*(sin(vec3(.7,2.0,2.5)+float(i)*.015 + time*0.3 + vec3(5,1,6))*0.45+0.55)*0.005;
            
            pos.xyz += vel*0.002*1.5;
        }
    }
    
    return rez;
}

// Function 60
vec4 saveParticle(particle P, vec2 pos)
{
    P.X = clamp(P.X - pos, vec2(-0.5), vec2(0.5));
    return vec4(encode(P.X), encode(P.V), P.M);
}

// Function 61
Particle getParticle(int index) {
    int offset = sizeofHeader + index * sizeofParticle;
    vec4 color = data(offset + colorField);
    vec4 posVel = data(offset + posVelField);
    vec4 wavelengthPhi = data(offset + wavelengthField);
    return Particle(color, posVel.xy, posVel.zw, wavelengthPhi.x, wavelengthPhi.y, wavelengthPhi.z);
}

// Function 62
void updateRank2x(vec2 t, inout vec4 O, inout float s0, inout float s1, vec2 I, vec3 R){
    float sp = score(t,I,R);
    if(sp<s0){
        //Shift down the line
        s1=s0;
        O.zw=O.xy;
        s0=sp;
        O.xy=t;
    } else if(sp<s1){
        //Bump off the bottom one
        s1=sp;
        O.zw=t;
        
    }
}

// Function 63
void Map_UpdateSector( sampler2D mapSampler, MapInfo mapInfo, vec2 vPrev, vec2 vPos, inout int iSectorId )
{    
    if ( vPrev == vPos )
    {
        return;
    }
    
    if ( !Map_PointInSector( mapSampler, mapInfo, vPos, iSectorId ) )
    {
        int iNewSectorId = Map_SeekSector( mapSampler, mapInfo, vPos );
        
        if ( iNewSectorId != SECTOR_NONE )
        {
        	iSectorId = iNewSectorId;
        }                
    }
}

// Function 64
vec2 FindArrivingParticle( vec2 arriveCoord, out vec4 partData )
{
    for( float i = -R; i <= R; i++ )
    {
        for( float j = -R; j <= R; j++ )
        {
            vec2 partCoord = arriveCoord + vec2( i, j );
            
            vec4 part = textureLod( iChannel0, partCoord / iResolution.xy, 0. );
            
            // particle in this bucket?
            if( dot(part,part) < 0.001 )
                continue;
            
            // is the particle going to arrive at the current pixel after one timestep?
            vec2 partPos = GetPos( part );
            vec2 partVel = GetVel( part );
            vec2 nextPos = partPos + partVel;
            // arrival means within half a pixel of this bucket
            vec2 off = nextPos - arriveCoord;
            if( abs(off.x)<=.5 && abs(off.y)<=.5 )
            {
                // yes! greedily take this particle.
                // a better algorithm might be to inspect all particles that arrive here
                // and pick the one with the highest velocity.
                partData = part;
                return partCoord;
            }
        }
    }
    // no particle arriving at this bucket.
    return vec2(-1.);
}

// Function 65
void WheelUpdateSteerAngle( float fSteerAngle, inout Wheel wheel )
{
    if ( !wheel.bSteering )
    {
        wheel.fSteer = 0.0;
    }
    else
    {
        // figure out turning circle if wheel was central
        float turningCircle = wheel.vBodyPos.z / tan( fSteerAngle );
        float wheelTurningCircle = turningCircle - wheel.vBodyPos.x;
        wheel.fSteer = atan( abs(wheel.vBodyPos.z) / wheelTurningCircle);
    }
}

// Function 66
vec2 particleHeadPosition(Particle particle, float time) {
    mat3 M = particleSpaceMatrix(particle.position, particle.velocity);
    float omega = 2.0 * 3.1415 * celerity / particle.wavelength;
    float l = sin(omega * time + particle.phi) * 0.02;
    return M[2].xy - l * M[1].xy;
}

// Function 67
void UpdateParticle()
{
    vec3 g = -5e-9*O.X*length(O.X); 
    vec3 F = g; 
    
    float scale = 0.14*pow(density,-0.333); //radius of smoothing
    float Rho = Kernel(0., scale);
    float avgP = 0.;
	vec3  avgC = vec3(O.Color);

    loop(j,6)
    {
        vec4 nb = texel(ch0, i2xy(ivec3(ID, j, 1)));
        loop(i,3)
        {
            if(nb[i] < 0. || nb[i] > float(TN)) continue;
            obj nbO = getObj(int(nb[i]));

            float d = distance(O.X, nbO.X);
            vec3 dv = (nbO.V - O.V); //delta velocity
            vec3 dx = (nbO.X - O.X); //delta position 
            vec3 ndir = dx/(d+1e-3); //neighbor direction
            //SPH smoothing kernel
            float K = Kernel(d, scale);

            vec3 pressure = -0.5*( nbO.Pressure/sqr(nbO.Rho) + 
                                     O.Pressure/sqr(O.Rho) )*ndir*K;//pressure gradient
            vec3 viscosity = 0.6*ndir*dot(dv,ndir)/(d+1.);
           
            Rho += K;
            avgC += nbO.Color;
            avgP += nbO.Pressure*K;

            F += pressure + viscosity;
        }
    }

    O.Rho = Rho;
    
    //O.Scale = scale; //average distance
    
    float r = 1.;
    float D = 1.;
    float waterP = 0.02*(pow(abs(O.Rho/density), r) - D);
    O.Pressure = min(waterP,1.);

    O.V += F*dt;
    O.V = 0.15*normalize(O.V);
    O.X += O.V*dt; //advect

    //color diffusion

    //O.Color = ;
}

// Function 68
void update(ivec2 p)
{
    randupd(p);
    upd(p, ivec2(0));
    updrad(p, 0);
    updrad(p, 2);
}

// Function 69
void updateCaveGame(
                    inout int gCaveState,
                    inout ivec2 gPlayerCoord,
                    inout ivec4 cellState,
                    inout int gDiamondsHarvested,
                    inout int gMagicWallStarted,
                    inout int gAmoebaState,
                    inout float flashAlpha,
                    inout int gAuxFrame,
                    inout int scoreToAdd,

                    const int cDiamondsRequired,
                    const int cDiamondValue,
                    const int cDiamondBonusValue,
                    const int cAmoebaMagWallTime,
                    const ivec2 cellCoord,
                    const int animFrame,
                    const int gameFrame,
                    const int gStartFrame,

                    float rand
                    )
{

    if (KEY_DOWN(KEY_SPACE))
    {
        bool isPaused = isState(gCaveState, CAVE_STATE_PAUSED);
        if (isPaused)
        {
            delState(gCaveState, CAVE_STATE_PAUSED);
            gAuxFrame = INT_MAX;
        }
        else
        {
            setState(gCaveState, CAVE_STATE_PAUSED);
            gAuxFrame = animFrame;
        }
    }

    if (isState(gCaveState, CAVE_STATE_FADE_IN) ||
        isState(gCaveState, CAVE_STATE_EXITED) ||
        isState(gCaveState, CAVE_STATE_PAUSED) ||
        isState(gCaveState, CAVE_STATE_TIME_OUT) ||
        isState(gCaveState, CAVE_STATE_GAME_OVER) ||
        isState(gCaveState, CAVE_STATE_FADE_OUT))
    {
        return;
    }

    CaveStateArr cave;

    for (int x=0; x<CAV_SIZ.x; x++)
    {
        for (int y=0; y<CAV_SIZ.y; y++)
        {
            ivec2 coord = ivec2(x, y);
            ivec4 cell = ivec4(loadValue(coord));
            cell.w = 0;  // need update
            setCell(cave, coord, cell);
        }
    }

    flashAlpha = max(0.0, flashAlpha - 1.0);

    int mWallStartDelta = gameFrame - gMagicWallStarted;
    int mWallState = (mWallStartDelta < 0) ? MWALL_STATE_DORMANT :
                     (mWallStartDelta < (cAmoebaMagWallTime * GAME_FRAMES_PER_SECOND) ) ? MWALL_STATE_ACTIVE : MWALL_STATE_EXPIRED;

    int amoebaNum = 0;
    bool isAmoebaGrowing = false;
    float amoebaProb = (animFrame - (gStartFrame + ENTRANCE_DURATION_AF)) > int(float(cAmoebaMagWallTime) / ANIM_FRAME_DURATION) ? AMOEBA_FAST_PROB : AMOEBA_SLOW_PROB;

    JoystickState joy = getJoystickState();

    for (int y=CAV_SIZ.y-1; y>=0; y--)
    {
        for (int x=0; x<CAV_SIZ.x; x++)
        {
            ivec2 coord = ivec2(x, y);
            ivec4 cell = getCell(cave, coord);

            if (!isUpdateNeeded(cell))
            {
                continue;
            }

            Fuse fuse = Fuse(CELL_VOID, ivec2(0));

            if (cell.x == CELL_ROCKFORD)
            {
                gPlayerCoord = coord;

                cell.y = (all(equal(joy.dir, DIR_RT))) ? cell.y | ROCKFORD_STATE_RT : cell.y;
                cell.y = (all(equal(joy.dir, DIR_LT))) ? cell.y & ~ROCKFORD_STATE_RT : cell.y;
                bool joyIdle = all(equal(joy.dir, DIR_NONE));
                cell.yz = (!((cell.y & ROCKFORD_STATE_IDLE) > 0) && joyIdle) ? ivec2(cell.y | ROCKFORD_STATE_IDLE, animFrame) : cell.yz;
                cell.y = (!joyIdle) ? cell.y & ~ROCKFORD_STATE_IDLE : cell.y;

                ivec2 coordTarget = coord + joy.dir;
                ivec4 cellTarget = getCell(cave, coordTarget);
                bool isMoved = false;

                if (cellTarget.x == CELL_VOID || cellTarget.x == CELL_DIRT)
                {
                    isMoved = true;
                }
                else if (cellTarget.x == CELL_DIAMOND)
                {
                    gDiamondsHarvested += 1;
                    scoreToAdd += (isState(gCaveState, CAVE_STATE_EXIT_OPENED)) ? cDiamondBonusValue : cDiamondValue;
                    isMoved = true;
                    if (gDiamondsHarvested == cDiamondsRequired)
                    {
                        setState(gCaveState, CAVE_STATE_EXIT_OPENED);
                        flashAlpha = 1.0;
                    }
                }
                else if (cellTarget.x == CELL_EXIT)
                {
                    if (isState(gCaveState, CAVE_STATE_EXIT_OPENED))
                    {
                        setState(gCaveState, CAVE_STATE_EXITED);
                        isMoved = true;
                    }
                }
                else if (cellTarget.x == CELL_BOULDER)
                {
                    if ((joy.dir == DIR_LT || joy.dir == DIR_RT) && !isFalling(cellTarget))
                    {
                        ivec2 coordBoulderTarget = coordTarget + joy.dir;
                        ivec4 cellBoulderTarget = getCell(cave, coordBoulderTarget);
                        if (cellBoulderTarget.x == CELL_VOID && isPushSucceeded(rand))
                        {
                            setCell(cave, coordBoulderTarget, cellTarget);
                            isMoved = true;
                        }
                    }
                }

                setUpdated(cell, true);

                if (isMoved)
                {
                    if (joy.isFirePressed)
                    {
                        setCell(cave, coordTarget, CELL_VOID4);
                    }
                    else
                    {
                        setCell(cave, coordTarget, cell);
                        setCell(cave, coord, CELL_VOID4);
                        gPlayerCoord = coordTarget;
                    }
                }
                else
                {
                    setCell(cave, coord, cell);
                }
            }

            else if (cell.x == CELL_BOULDER || cell.x == CELL_DIAMOND)
            {
                ivec2 coordDn = coord + DIR_DN;
                ivec4 cellDn = getCell(cave, coordDn);
                ivec2 boulderMoveDir = DIR_NONE;

                if (cellDn.x == CELL_VOID)
                {
                    boulderMoveDir = DIR_DN;
                }
                else if (cellDn.x == CELL_MAGIC_WALL && isFalling(cell))
                {
                    boulderMoveDir = DIR_DN2;
                }
                else if (isAbleToRollOff(cellDn))
                {
                    if (getCell(cave, coord + DIR_LT).x == CELL_VOID && getCell(cave, coord + DIR_LT_DN).x == CELL_VOID)
                    {
                        boulderMoveDir = DIR_LT;
                    }
                    else if (getCell(cave, coord + DIR_RT).x == CELL_VOID && getCell(cave, coord + DIR_RT_DN).x == CELL_VOID)
                    {
                        boulderMoveDir = DIR_RT;
                    }
                }

                if (boulderMoveDir == DIR_DN2)
                {
                    if (mWallState == MWALL_STATE_DORMANT)
                    {
                        mWallState = MWALL_STATE_ACTIVE;
                        gMagicWallStarted = gameFrame;
                    }
                    setCell(cave, coord, CELL_VOID4);
                    ivec2 coordTarget = coord + DIR_DN2;
                    ivec4 cellTarget = getCell(cave, coordTarget);
                    if ((mWallState == MWALL_STATE_ACTIVE) && (cellTarget.x == CELL_VOID))
                    {
                        int cellType = (cell.x == CELL_BOULDER) ? CELL_DIAMOND : CELL_BOULDER;
                        cellTarget = ivec4(cellType, 1, 0, 1); // is falling and updated
                        setCell(cave, coordTarget, cellTarget);
                    }
                }
                else if (any(notEqual(boulderMoveDir, DIR_NONE)))
                {
                    setFalling(cell, true);
                    setUpdated(cell, true);
                    setCell(cave, coord + boulderMoveDir, cell);
                    setCell(cave, coord, CELL_VOID4);
                }
                else if (isHitExplosive(cell, cellDn))
                {
                    fuse.type = (cellDn.x == CELL_BUTTERFLY) ? CELL_EXPL_DIAMOND : CELL_EXPL_VOID;
                    fuse.coord = coordDn;
                }
                else
                {
                    setFalling(cell, false);
                    setUpdated(cell, true);
                    setCell(cave, coord, cell);
                }
            }

            else if (cell.x == CELL_MAGIC_WALL)
            {
                cell.y = (mWallState == MWALL_STATE_ACTIVE) ? 1 : 0;
                cell.w = 1;
                setCell(cave, coord, cell);
            }

            else if (cell.x == CELL_FIREFLY || cell.x == CELL_BUTTERFLY)
            {
                setUpdated(cell, true);

                //explosion
                for (int v=0; v<4; v++)
                {
                    ivec4 cellNearby = getCell(cave, coord + DIRS[v]);
                    if ((cellNearby.x == CELL_ROCKFORD) || (cellNearby.x == CELL_AMOEBA))
                    {
                        fuse.type = (cell.x == CELL_BUTTERFLY) ? CELL_EXPL_DIAMOND : CELL_EXPL_VOID;
                        fuse.coord = coord;
                    }
                }

                // movement
                int dirIndex = cell.y;
                ivec2 dirLeft = getDirection(dirIndex, DIR_TURN_LT);
                ivec2 coordLeft = coord + dirLeft;

                if (getCell(cave, coordLeft).x == CELL_VOID)
                {
                    cell.y = dirIndex;
                    setCell(cave, coordLeft, cell);
                    setCell(cave, coord, CELL_VOID4);
                }
                else
                {
                    dirIndex = cell.y;
                    ivec2 dirAhead = DIRS[dirIndex];
                    ivec2 coordAhead = coord + dirAhead;
                    if (getCell(cave, coordAhead).x == CELL_VOID)
                    {
                        cell.y = dirIndex;
                        setCell(cave, coordAhead, cell);
                        setCell(cave, coord, CELL_VOID4);
                    }
                    else
                    {
                        getDirection(cell.y, DIR_TURN_RT);
                        setCell(cave, coord, cell);
                    }
                }
            }

            else if (cell.x == CELL_AMOEBA)
            {
                bool isCooked = (gAmoebaState == AMOEBA_STATE_COOKED);
                bool isOverCooked = (gAmoebaState == AMOEBA_STATE_OVERCOOKED);
                if (isCooked || isOverCooked)
                {
                    ivec4 cellNew = (isCooked) ? ivec4(CELL_DIAMOND, 0, 0, 1) : ivec4(CELL_BOULDER, 0, 0, 1);
                    setCell(cave, coord, cellNew);
                }
                else
                {
                    amoebaNum += 1;

                    rand = fract(rand + dot(vec2(coord), vec2(315.51, 781.64)));
                    bool isWantToSpawn = rand < amoebaProb;
                    ivec2 growCoord = coord + DIRS[int(fract(rand * 12378.1356) * 4.0) % 4];
                    ivec4 growCell = getCell(cave, growCoord);

                    if (isWantToSpawn && ((growCell.x == CELL_VOID) || (growCell.x == CELL_DIRT)))
                    {
                        isAmoebaGrowing = true;
                        amoebaNum += 1;
                        setCell(cave, growCoord, ivec4(CELL_AMOEBA, 0, 0, 1));
                    }
                }

                if (!isAmoebaGrowing)
                {
                    for(int i=0; i<4; i++)
                    {
                        int growCellType = getCell(cave, coord + DIRS[i]).x;
                        isAmoebaGrowing = isAmoebaGrowing || ((growCellType == CELL_VOID) || (growCellType == CELL_DIRT));
                    }
                }
            }

            else if (cell.x == CELL_EXPL_VOID)
            {
                if (cell.y >= EXPLOSION_DURATION_GF)
                {
                    setCell(cave, coord, CELL_VOID4);
                }
                else
                {
                    cell.y += 1;
                    setCell(cave, coord, cell);
                }
            }

            else if (cell.x == CELL_EXPL_DIAMOND)
            {
                if (cell.y >= EXPLOSION_DURATION_GF)
                {
                    setCell(cave, coord, ivec4(CELL_DIAMOND, 0, 0, 1));
                }
                else
                {
                    cell.y += 1;
                    setCell(cave, coord, cell);
                }
            }

            else if (cell.x == CELL_EXPL_ROCKFORD)
            {
                if (cell.y >= (EXPLOSION_DURATION_GF - 1))
                {
                    delState(gCaveState, CAVE_STATE_SPAWNING);
                    setCell(cave, coord, ivec4(CELL_ROCKFORD, ROCKFORD_STATE_IDLE, animFrame, 1));
                }
                else
                {
                    cell.y += 1;
                    setCell(cave, coord, cell);
                }
            }

            else if (cell.x == CELL_ENTRANCE)
            {
                if (cell.y >= ENTRANCE_DURATION_GF)
                {
                    setCell(cave, coord, ivec4(CELL_EXPL_ROCKFORD, 0, 0, 1));
                }
                else
                {
                    cell.y = min(cell.y + 1, ENTRANCE_DURATION_GF);
                    setCell(cave, coord, cell);
                }
            }

            else if (cell.x == CELL_EXIT)
            {
                setCell(cave, coord, ivec4(CELL_EXIT, (isState(gCaveState, CAVE_STATE_EXIT_OPENED)) ? 1 : 0, 0, 1));
            }

            else  // CELL_VOID, CELL_DIRT, CELL_WALL, CELL_TITAN_WALL
            {
                cell.w = 1;
                setCell(cave, coord, cell);
            }

            if (fuse.type != CELL_VOID)
            {
                for (int x=fuse.coord.x-1; x<=fuse.coord.x+1; x++)
                {
                    for (int y=fuse.coord.y-1; y<=fuse.coord.y+1; y++)
                    {
                        ivec2 explCoord = ivec2(x, y);
                        ivec4 explCell = getCell(cave, explCoord);
                        if (explCell.x != CELL_TITAN_WALL)
                        {
                            setCell(cave, explCoord, ivec4(fuse.type, 0, 0, 1));
                        }
                        if (explCell.x == CELL_ROCKFORD)
                        {
                            delState(gCaveState, CAVE_STATE_ALIVE);
                        }
                    }
                }
            }
        }
    }

    gAmoebaState = (!isAmoebaGrowing) ? AMOEBA_STATE_COOKED : gAmoebaState;
    gAmoebaState = (amoebaNum > AMOEBA_OVERCOOKED_NUM) ? AMOEBA_STATE_OVERCOOKED : gAmoebaState;
    cellState = isInCave(cellCoord) ? getCell(cave, cellCoord) : ivec4(0);
}

// Function 70
Particle GrowSplitParticle(int index, int parentIdx)
{
    return GrowParticle(index, parentIdx, 0.3);
}

// Function 71
int getParticle(float id, vec3 R, int iFrame){
    int seed = IHash(int(id))^IHash(int(iFrame));
    return seed;
}

// Function 72
void UpdateBoid( int index, inout Boid thisBoid )
{        
    vec3 vSeparationSteering = vec3(0);
    vec3 vCohesionSteering = vec3(0);
    vec3 vAlignmentSteering = vec3(0);
    vec3 vRandomSteering = vec3(0);
    vec3 vCollisionAvoidSteering = vec3(0);

    
    float fSeparationDist = 0.75;
    float fCohesionDist = 0.75;
    float fAlignmentDist = 0.75;
    float fCollisionAvoidDist = 1.0;
    
    float fCohesionWeight = 0.01 * UI_GetFloat( DATA_COHESION );
    float fSeparationWeight = 0.0002 * UI_GetFloat( DATA_SEPARATION );
    float fAlignmentWeight = 0.1 * UI_GetFloat( DATA_ALIGNMENT );
    float fRandomWalkWeight = 0.002;
    float fCollisionAvoidWeight = 0.001;

	float fMaxSteer = 0.01;    

    float fMinSpeed = 0.03;
    float fMaxSpeed = 0.1;
    
    bool bRestrictTo2d = true;
    
    thisBoid.vCohesionCentre = vec3(0);
    float fCohesionCount = 0.0;
    
    thisBoid.vAlignmentDir = vec3(0);
    float fAlignmentCount = 0.0;
    
    float boidCount = UI_GetFloat( DATA_COUNT );
    for ( int iOtherIndex = 0; iOtherIndex < MAX_BOID_COUNT; iOtherIndex++ )
    {
        if ( iOtherIndex >= int(boidCount) )
        {
            break;
        }
     
        // Don't consider ourself as neighbor
        if ( index == iOtherIndex )
        {
            continue;        
        }
        
        Boid otherBoid = LoadBoid( iOtherIndex );
        
        vec3 vToOther = otherBoid.vPos - thisBoid.vPos;
        
        // wrap world co-ordinates
        vToOther = mod( vToOther + WORLD_SIZE * .5, WORLD_SIZE) - WORLD_SIZE * .5;        
        vec3 vOtherPos = thisBoid.vPos + vToOther;
        
        float fDistToOther = length( vToOther );
        vec3 vDirToOther = normalize(vToOther);
                
        if ( fDistToOther < fSeparationDist )
        {
            float fSeparationStrength = 1.0 / (fDistToOther * fDistToOther);
            vSeparationSteering += -vDirToOther * fSeparationStrength;
        }

        if ( fDistToOther < fCohesionDist )
        {
            thisBoid.vCohesionCentre += vOtherPos;
            fCohesionCount++;
        }
        
        if ( fDistToOther < fAlignmentDist )
        {
            thisBoid.vAlignmentDir += otherBoid.vVel;
            fAlignmentCount++;
        }
    }

    if ( fCohesionCount > 0.0 )
    {
    	thisBoid.vCohesionCentre = thisBoid.vCohesionCentre / fCohesionCount;
    	vCohesionSteering += thisBoid.vCohesionCentre - thisBoid.vPos;    
    }

    if ( fAlignmentCount > 0.0 )
    {
    	thisBoid.vAlignmentDir = thisBoid.vAlignmentDir / fAlignmentCount;
    	vAlignmentSteering += thisBoid.vAlignmentDir - thisBoid.vVel; 
    }

	vRandomSteering = ( hash31( float( index ) + iTime ) * 2.0 - 1.0 );
    
    if ( UI_GetBool(DATA_WALLS) && UI_GetFloat(DATA_PAGE_NO) >= 8.0 )
    {    
        float fSceneDistance = Scene_Distance( thisBoid.vPos );
        if ( fSceneDistance < fCollisionAvoidDist )
        {
            vec3 vNormal = Scene_Normal( thisBoid.vPos ); 
            float fDist = fSceneDistance/ fCollisionAvoidDist;
            fDist = max( fDist, 0.01);
            vCollisionAvoidSteering += vNormal / ( fDist );
        }
    }
    
    vec3 vSteer = vec3( 0 );
    vSteer += vCohesionSteering * fCohesionWeight;
    vSteer += vSeparationSteering * fSeparationWeight;
    vSteer += vAlignmentSteering * fAlignmentWeight;
    vSteer += vRandomSteering * fRandomWalkWeight;
    vSteer += vCollisionAvoidSteering * fCollisionAvoidWeight;

    thisBoid.vSeparationSteer = vSeparationSteering;
    
    if ( bRestrictTo2d )
    {
		vSteer.y = 0.0;
    }
    
    ClampMagnitude( vSteer, 0.0, 0.01 );
    
    thisBoid.vVel += vSteer;
    
    if ( bRestrictTo2d )
    {
		thisBoid.vVel.y = 0.0;
    }

    ClampMagnitude( thisBoid.vVel, fMinSpeed, fMaxSpeed );
        
    // Move
    
    thisBoid.vPos += thisBoid.vVel;
    if ( bRestrictTo2d )
    {
		thisBoid.vPos.y = 0.0;
    }
    
    thisBoid.vPos = mod(  thisBoid.vPos, WORLD_SIZE );
}

// Function 73
vec3 updateDir(in vec3 prevDir)
{
    if (iFrame == 0)
        return normalize(vec3(0.0, -0.1, 1.0));
    
    vec3 dir = prevDir;
    
    vec3 side = normalize(cross(dir, vec3(0.0, 1.0, 0.0)));
    if (isPressed(KEY_R))
    	dir = (rotationMatrix(side, -iTimeDelta) * vec4(dir, 1.0)).xyz;
        
    if (isPressed(KEY_F))
    	dir = (rotationMatrix(side, iTimeDelta) * vec4(dir, 1.0)).xyz;
    
    if (isPressed(KEY_Q))
    	dir = (rotationMatrix(vec3(0.0, 1.0, 0.0), iTimeDelta) * vec4(dir, 1.0)).xyz;
        
    if (isPressed(KEY_E))
    	dir = (rotationMatrix(vec3(0.0, 1.0, 0.0), -iTimeDelta) * vec4(dir, 1.0)).xyz;
    
    return dir;
}

// Function 74
Particle readParticle( int idx )
{
    Particle p;
    int xc = XC(idx);
    int yc = YC(idx);
    // first line (y=0) reserved (e.g. for growNum, growIdx)
    p.pos    = getPixel(xc+0,yc);
    p.vel    = getPixel(xc+1,yc);
    p.angVel = getPixel(xc+2,yc);
    p.quat   = getPixel(xc+3,yc);
    // not sure if framebuffer has .w, so store quat.w also in next pixel
    vec4 p2  = getPixel(xc+4,yc);
    p.quat.w = p2.x;
    p.score  = p2.y;
    return p;
}

// Function 75
vec3 getParticleColor(int partnr, float pint)
{
   float hue;
   float saturation;

   saturation = mix(part_min_saturation, part_max_saturation, random(float(partnr*6 + 44) + runnr*3.3))*0.45/pint;
   hue = mix(part_min_hue, part_max_hue, random(float(partnr + 124) + runnr*1.5)) + hue_time_factor*time2;
    
   return hsv2rgb(vec3(hue, saturation, pint));
}

// Function 76
Particle GrowParticle(int index, int parentIdx, float randness)
{
    Particle p;
    Particle pp = readParticle(parentIdx);

    p.parent = parentIdx;
    p.size = 1.0;
    if(parentIdx>=0)
    {
        p.pos  = pp.pos + pp.size * transformVecByQuat( vec4(0,0,1,0), pp.quat );
        p.quat = calcGrowQuat(pp.quat,randness,index);
    }
    else
    {
        vec4 rand = getRand4(float(index))-vec4(0.5);
        p.pos  = vec4(vec3(0,0,0.1)*1.0+rand.yzw*0.03,0.0);
        p.pos  = vec4(vec3(0,0,0),0.0);
        p.quat = calcGrowQuat(vec4(0,0,0,1),randness,index);
        //p.quat = rand;
    }

    return p;
}

// Function 77
void gui_vertex_update() {    

    if (fc.x != BARY_COL && fc.x != SPSEL_COL) { return; }

    if (length(iMouse.zw - inset_ctr)*inset_scl > 1.) {       

        return; 
        
    } else {

        vec3 q = sphere_from_gui(iMouse.xy);
        
        vec4 spsel;
        int s = tri_snap(q);

        if (abs(iMouse.zw) == iMouse.xy && s >= 0) {
            if (s < 3) {
                if (fc.x == BARY_COL) {
                    data.xyz = bary_from_sphere( tri_verts[s] );
                } else {
                    data = vec4(0);
                }
            } else { 
                if (fc.x == BARY_COL) {
                    data.xyz = bary_from_sphere( tri_spoints[s-3] );
                } else {
                    data = vec4(0);
                    data[s-3] = 1.;
                }
            }
        } else {
            if (fc.x == BARY_COL) {
                data.xyz = bary_from_sphere( tri_closest(q) );
            } else {
                data = vec4(0);
            }
        }

    }
    
}

// Function 78
particle getParticle(float index, sampler2D ch, float iTime, vec4 iMouse, vec3 iResolution){
    
    vec3 p = .03*texture(ch,(.5+vec2(mod(index*3.,R.x),floor(index*3./R.x)))/R.xy).xyz;
    p = p.yzx;
    float r = 5.3;
    vec3 proj = ((p-cam)*view);
    return particle(proj.xy*R.y+R.xy/2.,proj.z,r,vec3(1));
    
}

// Function 79
vec3 surfaceParticles(in vec2 uv, in float pixelSize)
{
    vec3 particles = vec3(0.0);
 	vec2 rootUV = floor(uv);
    
   	vec2 tempRootUV;
    vec2 pointUV;
    float dist;
    vec3 color;
    for (float x = -PARTICLE_ITERATIONS; x <= PARTICLE_ITERATIONS; x += 1.0)
    {
        for (float y = -PARTICLE_ITERATIONS; y <= PARTICLE_ITERATIONS; y += 1.0)
        {
            tempRootUV = rootUV + vec2(x, y);
            pointUV = cellPointFromRootUV(tempRootUV, uv, dist);
          	color = mix(vec3(0), PARTICLE_COLOR, pow(smoothstep(0.3, 0.0, dist), 4.0));
            particles += particleFromUVAndPoint(uv, pointUV, tempRootUV, pixelSize) * color;
        }
    }
    
    return particles;
}

// Function 80
float SDF_particle_wlink(vec3 p0, vec3 p, vec3 p1)
{
    particle point = get(fakech0, p0);
    float pde = length(point.pos.xyz - p) - point.vel.w;
    #ifdef LINKS
        if(length(point.pos.xyz - p1) < 3.)
        {
             pde = min(pde, sdCapsule(p, point.pos.xyz, p1, sphere_rad*0.2));
        }
    #endif
    return pde;
}

// Function 81
void UpdateBossCannon( inout vec4 bossCannon )
{
    float accX 		= -fract( iTime * 1.069 + bossCannon.x * 7.919 ) * 5.0;
    vec4 newBullet	= vec4( bossCannon.xy - vec2( BOSS_CANNON_SIZE.x * 0.5, 0.0), accX, 0.0 );
    
    ++bossCannon.z;
    if ( bossCannon.z > BOSS_CANNON_FIRE_RATE )
    {
        bossCannon.z = 0.0;
        if ( gBossBullet0.x <= 0.0 )
        {
            gBossBullet0 = newBullet;
        }
        else if ( gBossBullet1.x <= 0.0 )
        {
            gBossBullet1 = newBullet;
        }
    }
}

// Function 82
void UpdatePosition(inout vec4 pos)
{
    pos.xyz = texelFetch(iChannel2, ivec2(0), 0).xyz;
}

// Function 83
float drawParticle(vec3 particlePosVel)
{
    vec2 toParticle = (particlePosVel.xy - uv) * iResolution.xy;
    vec2 toParticleAbs = abs(toParticle);
    
    bvec2 particleSquare = lessThan(toParticleAbs, vec2(PIXEL_SCALE_HALF));
    
    float particleness = float(all(particleSquare));
    if(particleSquare.x && toParticle.y < 0.0)
    {
		float trail = 700.* particlePosVel.z / (-toParticle.y);
        particleness += min(trail, 1.0) * 0.8f;
    }
    
    return particleness;
}

// Function 84
void gui_pqr_update() {
    
    if (fc.x != PQR_COL) { return; }
    
    for (int i=0; i<3; ++i) {

        int j = (i+1)%3;
        int k = 3-i-j;

        for (float delta=-1.; delta <= 1.; delta += 2.) {
            
            bool enabled = (delta < 0.) ? data[i] > 2. : data[i] < 5.;
            if (!enabled) { continue; }

            float d = box_dist(iMouse.xy, tri_ui_box(i, delta));       
            if (d > 0.) { continue; }

            data[i] += delta;
            
            int iopp = delta*data[j] > delta*data[k] ? j : k;
            
            for (int cnt=0; cnt<5; ++cnt) {
                if (valid_pqr(data.xyz)) { continue; }
                data[iopp] -= delta; 
            }   
            
        }
    }

}

// Function 85
vec3 getParticleColor(int partnr)
{
   float hue;
   float saturation;

   time2 = time_factor*iTime;
   pst = getParticleStartTime(partnr); // Particle start time
   plt = mix(part_life_time_min, part_life_time_max, random(float(partnr*2-35))); // Particle life time
   runnr = floor((time2 - pst)/plt);  // Number of the "life" of a particle 
    
   saturation = mix(part_min_saturation, part_max_saturation, random(float(partnr*6 + 44) + runnr*3.3));
   hue = mix(part_min_hue, part_max_hue, random(float(partnr + 124) + runnr*1.5)) + hue_time_factor*time2;
    
   return hsv2rgb(vec3(hue, saturation, 1.0));
}

// Function 86
float layeredParticles(vec2 screenUV, vec3 cameraPos)
{
    screenUV *= FOV;
	float particles = 0.0;
    float alpha = 1.0;
    float previousScale = 0.0;
    float targetScale = 1.0;
    float scale = 0.0;
    
    //Painting layers from front to back
    for (float i = 0.0; i < LAYERS_COUNT; i += 1.0)
    {
        //depth offset
        float offset = fract(cameraPos.z);
        
        //blending back and front
        float blend = smoothstep(0.0, FRONT_BLEND_DISTANCE, i - offset + 1.0);
        blend *= smoothstep(0.0, -BACK_BLEND_DISTANCE, i - offset + 1.0 - LAYERS_COUNT);
        
        float fog = mix(alpha * ALPHA_MOD, alpha, offset) * blend;
        
        targetScale = layerScaleFromIndex(i + 1.0);
        
        //dynamic scale - depends on depth offset
        scale = mix(targetScale, previousScale, offset);
        
        //adding layer
     	particles += particlesLayer(screenUV * scale + cameraPos.xy, floor(cameraPos.z) + i) * fog;
        alpha *= ALPHA_MOD;
        previousScale = targetScale;
    }
    
    return particles;
}

// Function 87
void UpdateVoronoi(inout particle U, in vec3 p)
{
    //check neighbours 
    CheckRadius(U, p, 1);
    CheckRadius(U, p, 2);
    CheckRadius(U, p, 3);
    CheckRadius(U, p, 4);
}

// Function 88
void update_perf_stats(inout vec4 fragColor, vec2 fragCoord)
{
    vec4 perf = (iFrame==0) ? vec4(0) : load(ADDR_PERF_STATS);
    perf.x = mix(perf.x, iTimeDelta*1000., 1./16.);
    store(fragColor, fragCoord, ADDR_PERF_STATS, perf);
    
	// shift old perf samples
    const vec4 OLD_SAMPLES = ADDR_RANGE_PERF_HISTORY + vec4(1,0,-1,0);
    if (is_inside(fragCoord, OLD_SAMPLES) > 0.)
        fragColor = texelFetch(iChannel1, ivec2(fragCoord)-ivec2(1,0), 0);

    // add new sample
    if (is_inside(fragCoord, ADDR_RANGE_PERF_HISTORY.xy) > 0.)
    {
        Options options;
        LOAD_PREV(options);
        fragColor = vec4(iTimeDelta*1000., get_downscale(options), 0., 0.);
    }
}

// Function 89
void Update(inout State state, ivec2 R, bool init)
{
    Inputs inp;
    LoadInputs(inp);
    if (state.resolution != R) { // resized?
        init = true; state.resolution = R; 
    }
    if (init) { // if zeroes aren't good enough
        state.eyepos = vec3(0,eyeh+.5*eyeradius,-4);
        state.aimbase = 
        state.eyeaim = vec2(0.,.5);
        state.mbdown = false;
    } else { // update state
        MoveCamera(state, inp);
	    TurnCamera(state, inp);
        if (state.mbdown && !inp.button) // on mouse up
    	    state.aimbase = state.eyeaim; // record aim base
        state.mbdown = inp.button;
    }
}

// Function 90
void updateScore(int scoreToAdd, inout int gScore, inout int gHighScore, inout int gLives, inout float gStripesAlpha)
{
    int newScore = gScore + scoreToAdd;
    if ((newScore / 500) > (gScore / 500))
    {
        addBonusLife(gLives, gStripesAlpha);
    }
    gScore = newScore;
    gHighScore = max(gScore, gHighScore);
}

// Function 91
vec2 particleSpawnPos(int frame, int particleIdx)
{
    float seed = float(frame) + float(particleIdx) / float(NUM_PARTICLES);
    vec2 pos;
    pos.x = rand(seed);
    pos.y = rand(pos.x + seed);
    return floor(pos * iResolution.xy + vec2(0.5));
}

// Function 92
bool FlyCam_Update( inout FlyCamState flyCam, vec3 vStartPos, vec3 vStartAngles )
{    
    //float fMoveSpeed = 0.01;
    float fMoveSpeed = iTimeDelta * 0.5;
    float fRotateSpeed = 3.0;
    
    
    if ( Key_IsPressed( iChannelKeyboard, KEY_SHIFT ) )
    {
        fMoveSpeed *= 4.0;
    }
    
    if ( iFrame == 0 )
    {
        flyCam.vPos = vStartPos;
        flyCam.vAngles = vStartAngles;
        flyCam.vPrevMouse = iMouse;
    }
      
    vec3 vMove = vec3(0.0);
        
    if ( Key_IsPressed( iChannelKeyboard, KEY_W ) )
    {
        vMove.z += fMoveSpeed;
    }
    if ( Key_IsPressed( iChannelKeyboard, KEY_S ) )
    {
        vMove.z -= fMoveSpeed;
    }

    if ( Key_IsPressed( iChannelKeyboard, KEY_A ) )
    {
        vMove.x -= fMoveSpeed;
    }
    if ( Key_IsPressed( iChannelKeyboard, KEY_D ) )
    {
        vMove.x += fMoveSpeed;
    }
    
    vec3 vForwards, vRight, vUp;
    FlyCam_GetAxes( flyCam, vRight, vUp, vForwards );
        
    flyCam.vPos += vRight * vMove.x + vForwards * vMove.z;
    
    vec3 vRotate = vec3(0);
    
    bool bMouseDown = iMouse.z > 0.0;
    bool bMouseWasDown = flyCam.vPrevMouse.z > 0.0;
    
    if ( bMouseDown && bMouseWasDown )
    {
    	vRotate.yx += ((iMouse.xy - flyCam.vPrevMouse.xy) / iResolution.xy) * fRotateSpeed;
    }
    
#if FLY_CAM_INVERT_Y    
    vRotate.x *= -1.0;
#endif    
    
    if ( Key_IsPressed( iChannelKeyboard, KEY_E ) )
    {
        vRotate.z -= fRotateSpeed * 0.01;
    }
    if ( Key_IsPressed( iChannelKeyboard, KEY_Q ) )
    {
        vRotate.z += fRotateSpeed * 0.01;
    }
        
	flyCam.vAngles += vRotate;
    
    flyCam.vAngles.x = clamp( flyCam.vAngles.x, -PI * .5, PI * .5 );
    
    if ( iFrame < 5 || length(vMove) > 0.0 || length( vRotate ) > 0.0 )
    {
        return true;
    }

    return false;
}

// Function 93
void Entity_Update( inout Entity entity, float fTimestep )
{    
    entity.iFrameFlags = 0;
    entity.iEvent = 0;
    entity.fTookDamage = 0.;

    // This is first to ensure consistent state
    if( Entity_SpawnOther( entity ) )
    {
        return;
    }
        
    Entity_Think( entity, fTimestep );

    Entity_Interact( entity, fTimestep );
    
    Entity_Move( entity, fTimestep );    
}

// Function 94
particle getParticle(vec4 data, vec2 pos)
{
    particle P;
    if (data == vec4(0)) return P;
    P.X = decode(data.x) + pos;
    P.M = data.y;
    P.V = data.zw;
    return P;
}

// Function 95
void UpdateParticle( const in vec2 fragCoord, inout vec2 vParticlePos, inout vec2 vParticleVel )
{    
#if 1     
    float fDensity = 0.0;
    float fNearDensity = 0.0;
    
    
    {
        vec2 vOffset;
        vOffset.y = -float(SIZE);
        for( int iY=-SIZE; iY<=SIZE; iY++ )
        {
            vOffset.x = -float(SIZE);
            for( int iX=-SIZE; iX<=SIZE; iX++ )
            {
                vec2 vCoord = fragCoord + vOffset;

                vec4 vSample = SampleCell( vCoord );
                vec2 vOtherPos = vSample.xy;
                vec2 vOtherVel = vSample.zw;
                
                if( vOtherPos.x >= 0.0 )
                {
                    float fDist = length(vOtherPos.xy - vParticlePos);
                    if ( fDist > 0.0 )
                    {
                        if ( fDist < g_fSmoothingRadius )
                        {
                            float fOneMinusQ = 1.0 - (fDist / g_fSmoothingRadius);

                            fDensity += fOneMinusQ * fOneMinusQ;
                        }
                    }
                }                				

                vOffset.x += 1.0;
            }

            vOffset.y += 1.0;
        }      
    }
    
	float fPressure = g_fK * (fDensity - g_fRestDensity);
    
    vec2 vD = vec2( 0.0 );

    vec2 vI = vec2( 0.0 );
    
    {
        vec2 vOffset;
        vOffset.y = -float(SIZE);
        for( int iY=-SIZE; iY<=SIZE; iY++ )
        {
            vOffset.x = -float(SIZE);
            for( int iX=-SIZE; iX<=SIZE; iX++ )
            {
                vec2 vCoord = fragCoord + vOffset;

                vec4 vSample = SampleCell( vCoord );
                vec2 vOtherPos = vSample.xy;
                vec2 vOtherVel = vSample.zw;

                if( vOtherPos.x >= 0.0 )
                {

                    float fDist = length(vOtherPos.xy - vParticlePos);
                    if ( fDist > 0.0 )
                    {
                        if ( fDist < g_fSmoothingRadius )
                        {
                            float fOneMinusQ = 1.0 - (fDist / g_fSmoothingRadius);

                            vec2 vDiff = vOtherPos - vParticlePos;
                            float fDist = length( vDiff );
                            vec2 vDir = normalize( vDiff );

                            if ( fDist < g_fSmoothingRadius )
                            {
                                float fOneMinusQ = 1.0 - (fDist / g_fSmoothingRadius);
                                vD += vDir * (g_fTimeStepSq * (fPressure * fOneMinusQ));


                                vec2 vVelDiff = vOtherVel - vParticleVel;
                                float u = dot( vVelDiff, vDir );

                                if( u > 0.0 )
                                {
                                    vI += vDir * (g_fTimeStep * fOneMinusQ * (g_fViscosity * u * u));
                                }
                            }
                        }                                               
                    }
                }
                else if( vOtherPos.x < -50.0 )
                {
                    vec2 vOtherPos = vCoord;
                    vec2 vDelta = vOtherPos - vParticlePos;
                    float fDist = length( vDelta );
                    float g_CollideRadius = 16.0;
                    if( fDist < g_CollideRadius )
                    {
                    	float fOneMinusQ = 1.0 - (fDist / g_CollideRadius);
                        vParticleVel -= normalize( vDelta ) * fOneMinusQ * 10.0;                        
                    }
                }
                

                vOffset.x += 1.0;
            }

            vOffset.y += 1.0;
        }      
    }

    vParticleVel += vI * 0.5;
	vParticleVel += vD * 0.5 / g_fTimeStep;    
    

#endif    
    
    vParticleVel.y -= 1000.0 * g_fTimeStep;  
    
    float fVelMag = length(vParticleVel);
    if ( fVelMag > 0.0 )
    {
        fVelMag = clamp( fVelMag, 0.0, 8.0 * 60.0 );
	    vParticleVel = normalize( vParticleVel ) * fVelMag;
    }

    vParticlePos += vParticleVel * g_fTimeStep;    
}

// Function 96
vec4 updateParticle(in vec4 particle) {
    vec2 v = particle.xy - particle.zw;
    //v = vec2(0.0);
    
    if (particle.x < 0.0 || particle.x >= 1.0) {
        v.x = -v.x;
    }
    if (particle.y < 0.0 || particle.y >= 1.0) {
        v.y = -v.y;
    }
    
    particle.zw = particle.xy;
    particle.xy += v;
        
    return particle;
}

// Function 97
vec4 saveParticle(particle P, vec2 pos)
{
    P.X = clamp(P.X - pos, vec2(-0.5), vec2(0.5));
    return vec4(encode(P.X), encode(P.V), P.M, P.I);
}

// Function 98
void UpdateTyreTracks( vec3 vCamPosPrev, vec3 vCamPos, inout vec4 fragColor, in vec2 fragCoord )
{
    float fRange = 20.0;
    vec2 vPrevOrigin = floor( vCamPosPrev.xz );
    vec2 vCurrOrigin = floor( vCamPos.xz );

    vec2 vFragOffset = ((fragCoord / iResolution.xy) * 2.0 - 1.0) * fRange;
    vec2 vFragWorldPos = vFragOffset + vCurrOrigin;
	
    vec2 vPrevFragOffset = vFragWorldPos - vPrevOrigin;
	vec2 vPrevUV = ( (vPrevFragOffset / fRange) + 1.0 ) / 2.0;
    vec4 vPrevSample = textureLod( iChannel1, vPrevUV, 0.0 );
    
    vec4 vWheelContactState[4];
    vWheelContactState[0] = LoadVec4( addrVehicle + offsetVehicleWheel0 + offsetWheelContactState );
    vWheelContactState[1] = LoadVec4( addrVehicle + offsetVehicleWheel1 + offsetWheelContactState );
    vWheelContactState[2] = LoadVec4( addrVehicle + offsetVehicleWheel2 + offsetWheelContactState );
    vWheelContactState[3] = LoadVec4( addrVehicle + offsetVehicleWheel3 + offsetWheelContactState );
    
    fragColor = vPrevSample;
    
    if ( vPrevUV.x < 0.0 || vPrevUV.x >= 1.0 || vPrevUV.y < 0.0 || vPrevUV.y >= 1.0 )
    {
        fragColor = vec4(0.0);
    }
    
    for ( int w=0; w<4; w++ )
    {        
        vec2 vContactPos = vWheelContactState[w].xy;
        
        float fDist = length( vFragWorldPos - vContactPos );
        
        if ( vWheelContactState[w].z > 0.01 )
        {
            float fAmount = smoothstep( 0.25, 0.1, fDist );
            fragColor.x = max(fragColor.x, fAmount * vWheelContactState[w].z );
            
            fragColor.y = max(fragColor.y, fAmount * vWheelContactState[w].w * 0.01);
        }		
    }
    
    
    fragColor.x = clamp( fragColor.x, 0.0, 1.0);
    fragColor.y = clamp( fragColor.y, 0.0, 1.0);
    
    if( iFrame < 1 )
    {
    	fragColor.x = 0.0;  
    }
}

// Function 99
void UpdateGravity(inout vec3 position, inout vec3 velocity)
{
    vec3 g = vec3(0.0, Gravity, 0.0) * iTimeDelta;
    vec3 p = (position + FeetPosition);
    
    float d = Scene(p);
    
    if(d <= 0.0)
    {
        position.y += (-d);
        velocity.y = 0.0;
    }
    else
    {
        velocity += g;
    }
}

// Function 100
void Update(vec3 ro, vec3 rd)
{
    for(int obj = 0; obj < NUMBER_OF_OBJECTS; obj++)
    {
        vec2 flat_pos = texelFetch(iChannel0, ivec2(obj, 1), 0).zw;
        float env = texelFetch(iChannel0, ivec2(obj, 0), 0).x;
        vec3 sph_pos = to_polar(flat_pos);
        env = pow(trapezoid(env, TRAPEZOID), 10.5);
        float bounce = env*0.1;
        sph_pos.xz = sph_pos.xz*bounce+sph_pos.xz;
        //Positions: satellite distances, satellite positions
        Positions[obj] = vec4(sphDistances(ro,rd,vec4(sph_pos,SAT_RADIUS)).x, sph_pos);
        Notes[obj] = texelFetch(iChannel0,ivec2(obj,0),0);
    }
}

// Function 101
float bokehParticles(vec2 uv, float radius, vec2 dotProportions, float dotRotation, float animationOffset)
{
 	float voro = voronoi(uv, dotProportions, dotRotation, animationOffset);
    float particles = 1.0 - smoothstep(radius, radius * (2.0), voro);
    return particles;
}

// Function 102
float particleFromUVAndPoint(in vec2 uv, in vec2 point, in vec2 rootUV, in float pixelSize)
{
	float dist = distance(uv, point);
#ifdef RANDOMIZED_SIZE
    dist += (hash1_2(rootUV * 10.0) - 0.5) * PARTICLE_SIZE_VARIATION;
#endif
    float particle = 1.0 - smoothstep(PARTICLE_RADIUS - dist * 0.05, PARTICLE_RADIUS2 - dist * 0.05 + pixelSize, dist);
    return particle * particle;
}

// Function 103
void Update(inout State state, ivec2 R, bool init)
{
    Inputs inp;
    LoadInputs(inp);
    if (state.resolution != R) { // resized?
        init = true; state.resolution = R; 
    }
    if (init) { // if zeroes aren't good enough
        state.eyepos = vec3(0,2.*eyeradius,-4);
        state.aimbase = 
        state.eyeaim = vec2(0.,.5);
        state.mbdown = false;
    } else { // update state
		MoveCamera(state, inp);
	    TurnCamera(state, inp);
        if (state.mbdown && !inp.button) // on mouse up
    	    state.aimbase = state.eyeaim; // record aim base
        state.mbdown = inp.button;
    }
}

// Function 104
int particleIdx(vec2 coord)
{
    ivec2 res=textureSize(ParticleTex,0);
    return (int(coord.y)*res.x+int(coord.x))/3;
}

// Function 105
float particleVis(vec2 uv) {
    float minDist = 1.0;
    
    for(float i = 0.0; i < 400.0; i += 4.0) {
        vec2 bufB_UV = vec2(i / float(iResolution.x), 0.0);
        vec2 particlePos = texture(iChannel0, bufB_UV).xy;
        
        float dist = length(particlePos - uv);
        minDist = min(dist, minDist);
    }
    return step(minDist, 0.05);
}

// Function 106
vec3 particleColor (in float particleVelocity) {
	return mix (vec3 (0.5, 0.6, 0.8), vec3 (0.9, 0.9, 1.0), particleVelocity * PARTICLE_VELOCITY_FACTOR);
}

// Function 107
vec3 drawParticles(vec2 pos, vec3 particolor, float time, vec2 cpos, float gravity, float seed, float timelength){
    vec3 col= vec3(0.0);
    vec2 pp = vec2(1.0,0.0);
    for(float i=1.0;i<=128.0;i++){
        float d=rand(i, seed);
        float fade=(i/128.0)*time;
        vec2 particpos = cpos + time*pp*d;
        pp = rr*pp;
        col = mix(particolor/fade, col, smoothstep(0.0, 0.0001, distance2(particpos, pos)));
    }
    col*=smoothstep(0.0,1.0,(timelength-time)/timelength);
	
    return col;
}

// Function 108
vec4 getParticle(int i)
{
    // read from myself
    return texelFetch(iChannel0, ivec2(i, 0), 0);
}

// Function 109
void UpdateSniper( inout vec4 sniper, vec2 playerTarget )
{
    if ( sniper.x + SNIPER_SIZE.x * 0.5 < gCamera.x )
    {
        sniper.x = 0.0;
    }  
    
    ++sniper.w;
	if ( sniper.x > 0.0 && sniper.w > SNIPER_FIRE_RATE )
    {
        sniper.w = 0.0;
        vec2 pos = sniper.xy + vec2( 0.0, 24.0 );
        SpawnEnemyBullet( pos, normalize( playerTarget - pos ) );
    }
    sniper.z = playerTarget.x > sniper.x ? 1.0 : -1.0;    
}

// Function 110
void UpdateLightDirection()
{
	vec3 normalLight = normalize(vec3(0.2, 0.9, 0.2));
	float radTheta = 2.0 * g_gPi * g_t / 60.0;
	float gSin = sin(radTheta);
	float gCos = cos(radTheta);
	mat2 matRot = mat2(gCos, -gSin, gSin, gCos);
	normalLight.xy = matRot * normalLight.xy;
	g_normalLight = normalLight;
}

// Function 111
float SDF_particle_wlink(vec3 p0, vec3 p, vec3 p1)
{
    particle point = get(fakech0, p0);
    float pde = length(point.pos.xyz - p) - sphere_rad;
    #ifdef LINKS
        if(length(point.pos.xyz - p1) < 4.)
        {
             pde = min(pde, sdCapsule(p, point.pos.xyz, p1, sphere_rad*0.2));
        }
    #endif
    return pde;
}

// Function 112
vec3 powerParticle(vec2 st){
  
    st.y += ((st.x*0.05)*sin(time/10.*PI)+(st.x*0.1)*sin(time/12.*PI))/2.;
    st.x += ((st.y*0.05)*sin(time/10.*PI) + (st.y*0.1)*sin(time/12.*PI))/2.;
    
    vec2 pos = vec2(0.25+0.25*sin(time))-abs(st);

    float r = length(pos);
    float d = distance(st,vec2(0.5))* (sin(time/8.));
    d = distance(vec2(.5),st);
   vec3 colorNew = vec3(0);
   
   float delay = delayAmount;
   float timerChecker = time * speed ;
    for(int i=0;i<10;i++) {
     
      vec3 colorCheck = wooper(st, timerChecker+ float(i)*delay)* (1.-(float(i)/10.0));
      colorNew+= colorCheck ;
    }
    
    return(colorNew);
}

// Function 113
void update_trajectory_heun(in int body, out vec4 XYZW, out vec3 velXYZ)
{
    vec4 bodyXYZW;
    vec3 bodyVelXyz;
    getBodyPosVel(body, bodyXYZW, bodyVelXyz);
    
    // step 1
    vec3 accelVec1 = calculate_gravity_accel(body, bodyXYZW, 0.0);       
    vec3 velXYZ1 = bodyVelXyz.xyz + GRAVITY_COEFF * accelVec1;
    XYZW = bodyXYZW + UPDATE_STEP * vec4(velXYZ1, 0);

    // step 2
    vec3 accelVec2 = calculate_gravity_accel(body, XYZW, UPDATE_STEP);    

    vec3 accelVec_mid = 0.5 * (accelVec1 + accelVec2);
    vec3 velXYZ2 = velXYZ1 + GRAVITY_COEFF * accelVec_mid;
    vec3 velXYZ_mid = 0.5 * (velXYZ1 + velXYZ2);
    
    velXYZ = bodyVelXyz + GRAVITY_COEFF * accelVec_mid; 
    XYZW = bodyXYZW + UPDATE_STEP * vec4(velXYZ_mid, 0.0);
}

// Function 114
vec4 updateParticle(in vec4 particle, vec2 a) {
    vec2 v = particle.xy - particle.zw;
    
    v += a;
    v *= 0.5;
    
    if (particle.x + v.x < 0.0 || particle.x + v.x >= 1.0) {
        v.x = -v.x;
        v *= 0.5;
    }
    if (particle.y + v.y < 0.0 || particle.y + v.y >= 1.0) {
        v.y = -v.y;
        v *= 0.5;
    }
    
    float maxSpeed = 0.01;
    v = length(v) > maxSpeed ? maxSpeed * v / length(v) : v;
    
    particle.zw = particle.xy;
    particle.xy += v;
        
    return particle;
}

// Function 115
vec4 updateTime(in vec4 time)
{
    if (iFrame == 0)
        time = vec4(6.0, 0.0, 0.0, 0.0);
    
    if (isPressed(KEY_G))
        time.x += iTimeDelta;
    
    return time;
}

// Function 116
void update_closest(vec3 p, inout Closest closest, vec3 candidate)
{
    float distance_squared = length_squared(p - candidate);
    if (distance_squared < closest.distance_squared)
    {
        closest.distance_squared = distance_squared;
        closest.point = candidate;
    }
}

// Function 117
float particleFromParticleUV(vec2 particleUV, vec2 uv)
{
 	return 1.0 - smoothstep(0.0, PARTICLE_SIZE, length(particleUV - uv));   
}

// Function 118
float distance2Particle(int id, vec2 fragCoord){
    if(id==-1) return 1e20;
    vec2 delta = getParticle(id).xy-fragCoord;
    return dot(delta, delta);
}

// Function 119
void UpdatePosition(inout vec4 pos)
{
    if (iFrame == 0)
    {
        pos = vec4(-0.25 * UNIVERSE_RADIUS * FWD, INIT_FLY_SPEED);
    }
    else
    {
        vec4 rot = GetCameraRotation();
        if (rot.w == 0.0)
        {
            rot = QUAT_IDENTITY;
        }
        
        //--fly
        float dt = DELTA_TIME * TIME_SCALE;
        float ds = dt * FLY_ACCELERATION;
        int btn = HoldSpeedButton(iMouse, iResolution.xy);
        if (readKey(keyF) || btn == 0) pos.w *= 1.0 + ds;
        if (readKey(keyR) || btn == 1) pos.w *= 1.0 - ds;
        pos.w = min(pos.w, MAX_FLY_SPEED);
        
        vec3 g = CalculateSumAcceleration(pos.xyz);
        vec3 v = CalculateVelocity(RotateFWD(rot) * pos.w, g, dt * TimeSlow(length(g)));
        pos.xyz = CalculatePosition(pos.xyz, v, dt * TimeSlow(length(g)));
        pos.w = length(v);
    }
}

// Function 120
float sdf_particle(vec3 p){
    return length(p.xy)-.02;
    float z=p.z*7.+iTime;
    p.xy*=mat2(cos(z*1.3),sin(z*1.3),-sin(z*1.3),cos(z*1.3));
    p.yz*=mat2(cos(z*1.7),sin(z*1.7),-sin(z*1.7),cos(z*1.7));
    p.zx*=mat2(cos(z*2.1),sin(z*2.1),-sin(z*2.1),cos(z*2.1));
    p=abs(p);
    //return max(max(p.x,p.y),p.z)-.02;
    return length(p)-.02;
}

// Function 121
float arrivingParticle(vec2 coord, out vec4 partData) {
	// scan area from -D to D
    for (int i=-A; i<A; i++) {
        for (int j=-A; j<A; j++) {
            // position to check
            vec2 arrCoord = coord + vec2(i,j);
            vec4 data = texture(iChannel0, arrCoord/iResolution.xy);
            
            // no particles here
            if (dot(data,data)<.1) continue;

            // get next position of particle
            vec2 nextCoord = data.xy + data.zw;

            // distance between next position and current pixel
            vec2 offset = abs(coord - nextCoord);
            // if the distance is within half a pixel pick this particle
            // (other arriving particles are dismissed)
            if (offset.x<.5 && offset.y<.5) {
                partData = data;
                return 1.;
            }
        }
    }
    // no particles arriving here
	return 0.;
}

// Function 122
vec3 particleColor (in float particleVelocity) {
	return mix (vec3 (0.5, 0.5, 1.0), vec3 (1.0), particleVelocity * VELOCITY_COLOR_FACTOR);
}

// Function 123
void UpdatePosition(inout vec4 pos)
{pos.xyz=texelFetch(iChannel1, ivec2(0),0).xyz;}

// Function 124
vec2 getParticlePosition(in int particleID)
{
    int iChannel0_width = int(iChannelResolution[0].x);
	ivec2 particleCoord = ivec2(particleID % iChannel0_width, particleID / iChannel0_width);
    
    return texelFetch(iChannel0, particleCoord, 0).xy;
}

// Function 125
vec4 UpdateRotation( vec2 resetValue )
{
    vec4 camrot = texelFetch(iChannel0,ivec2(1,0),0);
    
    // mouse drag stuff always sucks... hmm...
    // SHOULD be able to just measure difference between xy (current) and zw (last click), if x != 0.
    if ( iMouse.z > 0. )
    {
        if ( camrot.z <= 0. ) camrot.zw = iMouse.zw;
        
        // drag in progress
        // total drag distance is iMouse.xy-iMouse.zw - but we want realtime feedback so remember where we were last frame
        vec2 delta = vec2(-1,1) * (iMouse.xy-camrot.zw)/iResolution.x;
        camrot.zw = iMouse.xy;
        camrot.xy += delta.yx;
    }
    else
    {
        camrot.zw = vec2(-1);
    }
    
    return reset?vec4(resetValue,-1,-1):camrot;
}

// Function 126
float SmokeParticle(vec2 loc, vec2 pos, float size, float rnd)
{
	loc = loc-pos;
	float d = dot(loc, loc)/size;
	// Outside the circle? No influence...
	if (d > 1.0) return 0.0;

	// Rotate the particles...
	float r= time*rnd*1.85;
	float si = sin(r);
	float co = cos(r);
	// Grab the rotated noise decreasing resolution due to Y position.
	// Also used 'rnd' as an additional noise changer.
	d = noise(hash(rnd*828.0)*83.1+mat2(co, si, -si, co)*loc.xy*2./(pos.y*.16)) * pow((1.-d), 3.)*.7;
	return d;
}

// Function 127
float particles(vec2 p)
{
  p *= 200.;
  float f = 0.;
  float amp = 1.0, s = 1.5;
  for (int i=0; i<3; i++)
  { p = m*p*1.2; f += amp*noise( p+iTime*s ); amp = amp*.5; s*=-1.227; }
  return pow(f*.35, 7.)*particle_amount;
}

// Function 128
void dataUpdate(ivec2 ifc, vec2 m){
    components[12].value.x = float(frame)/time;
    if ( ifc.x == MEM_COMPONENTS && ifc.y < componentsLength )
    {
        processActionsComponents(ifc);
        processMouseOnComponents( ifc,  m);
    }
}

// Function 129
void UpdateMouseClick(inout vec4 fragColor, in vec2 mouse)
{
    vec3 lastMouse = GetLastMouseClick();
    float isClicked = step(0.5, iMouse.z);
        
    if((isClicked > 0.5) && lastMouse.z < 0.5)
    {
        fragColor.xy = vec2(mouse.xy / iResolution.xy);
    }
        
    fragColor.z = isClicked;
}

// Function 130
float RockParticle(vec2 loc, vec2 pos, float size, float rnd)
{
	loc = loc-pos;
	float d = dot(loc, loc)/size;
	// Outside the circle? No influence...
	if (d > 1.0) return 0.0;
	float r= time*1.5 * (rnd);
	float si = sin(r);
	float co = cos(r);
	d = noise((rnd*38.0)*83.1+mat2(co, si, -si, co)*loc*143.0) * pow(1.0-d, 15.25);
	return pow(d, 2.)*5.;
	
}

// Function 131
vec3 particle_color(vec3 p)
{
    vec4 a = vec4(1e5);
    vec3 p0 = round(p);
    for( int i=-1; i<=1; i++ )
        for( int j=-1; j<=1; j++ )
            for( int k=-1; k<=1; k++ )
    {
        vec3 dx = vec3(i,j,k);
        particle thisp = get(fakech0, p0+dx);
        a = opunion(a, vec4(jet_range(thisp.pos.w, -0.1, 1.2), SDF_particle(p0+dx, p)));
    }
    return a.xyz;
}

// Function 132
vec3 maybeRenderParticle(
    float dist,
    vec2 particle,
    vec2 pixel,
    int particleIndex)
{
    vec3 color = vec3(0);
    if (dist <= PARTICLE_SIZE) { 
        vec2 delta = particle - pixel;
        float angle = atan(delta.x, delta.y);
        color +=
            getColor(dist, angle, PARTICLE_SIZE)
            * getParticleColor(particleIndex).rgb;            
    }
    return color;
}

// Function 133
vec2 getParticleVelocity(in int particleID)
{
    int iChannel0_width = int(iChannelResolution[0].x);
	ivec2 particleCoord = ivec2(particleID % iChannel0_width, particleID / iChannel0_width);
    
    return texelFetch(iChannel0, particleCoord, 0).xy - texelFetch(iChannel0, particleCoord, 0).zw;
}

// Function 134
vec4 updateCameraPos(inout vec4 camPos, ivec2 playerCoord)
{
    const vec2 threshold = vec2(6.0, 2.0);
    vec2 camPosTarget = getCamTargetPos(playerCoord);
    vec2 mask = vec2(greaterThan(abs(vec2(playerCoord) + vec2(0.5) - camPos.xy), threshold));
    camPos.zw = lerp(camPos.zw, camPosTarget, mask);

    vec2 dif = camPos.zw - camPos.xy;
    vec2 dirIndex = sign(dif);
    camPos.xy += dirIndex * min(vec2(CAMERA_PAN_PER_ANIM_FRAME), abs(dif));

    return camPos;
}

// Function 135
void SpawnParticles(inout vec4 abubble, inout vec4 abubbleState, inout int numToSpawn, inout vec4 spawn, inout float lastArrowSide, in int i)
{
    const vec2 minXPos = vec2(-1.25, -1.);
    const float cellSperation = 3.4/6.;
    
    if(numToSpawn > 0 && abubbleState.x <= 0. && (gTime - abubbleState.z) > BUBBLE_FADE_OUT_TIME)
    {
        numToSpawn--;
        spawn.y++;
        
        // Modify the number of spawn to vary the position
        float extraPos = max((hash(float(iFrame)) - 0.7)/0.3, 0.)*12.;

        float posId = (mod(extraPos + spawn.y, 6.)/6.);
        
        float posX = -1.5 + 1.7*posId*gAR;
        abubble = vec4(posX, -1., -sign(posX)*0.05 , kMoveSpeed);
        float arrowSide = floor(4.0*abs(hash(vec2(10.478*gTime + float(i)*5098.45))));
        if( abs(lastArrowSide-arrowSide) < 1.0e-3 )
        {
            arrowSide = mod(arrowSide +1., 4.);
        }
        lastArrowSide = arrowSide;
        abubbleState = vec4(1., arrowSide , 0., 0.);
    }
}

// Function 136
void updateFlag( inout vec4 fragColor, in vec2 fragCoord )
{
    vec3 flagPos=readFlagPos();
    vec3 posAvg=vec3(0.0);
    for(int i=0;i<MaxParticleNum;i++)
    {
        Particle p = readParticle(i);
        posAvg+=p.pos.xyz;
        vec3 dpos = p.pos.xyz-flagPos;
        if(dot(dpos,dpos)<FlagRadius*FlagRadius)
        {
            writeFlagPos(calcFlagPos(p.pos.xyz), fragColor, fragCoord);
        }
    }
    posAvg/=float(MaxParticleNum);
    if( flagPos.x==0.0 && flagPos.y==0.0 && flagPos.z==0.0 && iFrame>100 )
    {
        writeFlagPos(calcFlagPos(posAvg), fragColor, fragCoord);
    }
}

// Function 137
void updateState(inout state s) {

    // p (object displacement) gets "lerped" towards q
    if (iMouse.z > 0.5) {
        vec2 uvMouse = iMouse.xy / iResolution.xy;
        vec3 camPos;
        vec3 nvCamDir;
        getCamera(s, uvMouse, camPos, nvCamDir);

        float t = -camPos.y/nvCamDir.y;
        if (t > 0.0 && t < 50.0) {
            vec3 center = vec3(0.0);
            s.q = camPos + t*nvCamDir;
            float qToCenter = distance(center, s.q);
            if (qToCenter > 5.0) {
                s.q = mix(center, s.q, 5.0/qToCenter);
            }
        }
    }

    // pr (object rotation unit quaternion) gets "slerped" towards qr
    float tmod = mod(iTime+6.0, 9.0);
    vec4 qr = (
        tmod < 3.0 ? qRot(vec3( SQRT2INV, 0.0, SQRT2INV), 0.75*PI) :
        tmod < 6.0 ? qRot(vec3(-SQRT2INV, 0.0, SQRT2INV), 0.5*PI) :
        QID
    );

    // apply lerp p -> q and slerp pr -> qr
    s.p += 0.25*(s.q - s.p);
    s.pr = normalize(slerp(s.pr, qr, 0.075));

    // object acceleration
    vec3 a = -0.25*(s.q - s.p) + vec3(0.0, -1.0, 0.0);
    mat3 prMatInv = qToMat(qConj(s.pr));
    a = prMatInv*a;

    // hand-wavy torque and angular momentum
    vec3 T = cross(s.v, a);
    s.L = 0.96*s.L + 0.2*T;

    // hand-wavy angular velocity applied from torque
    vec3 w = s.L;
    float ang = 0.25*length(w);
    if (ang > 0.0001) {
        mat3 m = qToMat(qRot(normalize(w), ang));
        s.v = normalize(m*s.v);
    }
}

// Function 138
Particle getParticle(ivec2 cell){
	return decodeParticle(texelFetch(iChannel0, cell, 0), cell);
}

// Function 139
void setUpdated(inout ivec4 cell, bool isUpdated)
{
    cell.w = isUpdated ? 1 : 0;
}

// Function 140
void writeParticle( int idx, Particle p, inout vec4 fragColor, vec2 fragCoord)
{
    int xc = XC(idx);
    int yc = YC(idx);
    if (isPixel(xc+0,yc,fragCoord)) fragColor.xyzw=p.pos;
    if (isPixel(xc+1,yc,fragCoord)) fragColor.xyzw=p.vel;
    if (isPixel(xc+2,yc,fragCoord)) fragColor.xyzw=p.angVel;
    if (isPixel(xc+3,yc,fragCoord)) fragColor.xyzw=p.quat;
    // not sure if framebuffer has .w, so store quat.w also in next pixel
    if (isPixel(xc+4,yc,fragCoord)) fragColor.xyzw=vec4(p.quat.w,p.score,0,1);
}

// Function 141
Particle updateP(Particle p, Particle po, float delta){
    vec2 dir = po.pos - p.pos;
    vec2 acc = dir * grav(p,po);
    p.vel += acc / p.mass   * delta;
    p.pos += p.vel * delta;
    return p;
}

// Function 142
void InitParticles()
{
	// Because pold isn't reference here,
	// compiler should eliminate fetches of
	// unused second pixel of each particle.  I hope!
    flagbblo = vec3(9e9); flagbbhi = vec3(-9e9);
    for (int j = 0; j < nlinky; ++j)
        for (int i = 0; i < nlinkx; ++i) {
		vec3 p = ps[pidx(i, j)] = particle(ClothBuf, ivec2(i, j)).pnew;
        flagbblo = min(flagbblo, p); flagbbhi = max(flagbbhi, p);
    }   
}

// Function 143
float SDF_particle(vec3 p0, vec3 p)
{
    particle point = get(fakech0, p0);
    return length(point.pos.xyz - p) - point.vel.w;
}

// Function 144
void updateRank(vec4 t, inout vec4 O, inout float s, vec2 I, vec3 R){
    float sp = score(t.xy,I,R);
    if(sp<s){
        s=sp;
        O=t;
    }
}

// Function 145
Particle readParticle( int idx )
{
    Particle p;
    int xc = XC(idx);
    int yc = YC(idx);
    // first line (y=0) reserved (e.g. for growNum, growIdx)
    p.pos    = getPixel(xc+0,yc);
    p.quat   = getPixel(xc+1,yc);
    // not sure if framebuffer has .w, so store quat.w also in next pixel
    vec4 p2  = getPixel(xc+2,yc);
    p.quat.w = p2.z;
    p.size   = p2.x;
    p.parent = int(p2.y);
    return p;
}

// Function 146
void ts_update_stable( inout TrnSampler ts, vec3 r, float r0, float res )
{
    bool mustupdate = true;
	if( ts_is_valid( ts ) )
	{
		if( ts_is_uv_safe( ts, r ) )
        {
            vec2 uvnew = ts_uv( ts, r );
            mustupdate = length( uvnew - .5 ) * res * TRN_SCALE >= TRN_UPDATE_THRESHOLD * SCN_SCALE;

            float scalenew = ts_scale( r, r0 );
            float scaleold = ts.e_over_b * ts.invm;

            mustupdate = mustupdate ||
                abs( scalenew - scaleold ) * res >= 4. * TRN_UPDATE_THRESHOLD * scaleold;

            /*
            if( mustupdate )
            {
            	vec3 r_from_uvnew = ts_uv_inverse_lod( ts, round( uvnew * res ) / res ).xyz;
                // if( r_from_uvnew != ZERO )
					r = length(r) / ts.r0 * r_from_uvnew;
                // else
                //	mustupdate = false;
            }
			//*/
        }
    }
	if( mustupdate )
		ts = ts_init( r, r0, UNIT_Z );
}

// Function 147
void update_demo_stage(vec2 fragCoord, vec2 resolution, float downscale, sampler2D noise, int frame, bool thumbnail)
{
    float time = g_time;

    if (!is_demo_mode_enabled(thumbnail))
    {
		g_demo_stage = DEMO_STAGE_NONE;
        return;
    }
    
    resolution *= 1./downscale;
    vec2 uv = clamp(fragCoord/resolution, 0., 1.);
        
    const float TRANSITION_WIDTH = .125;
    const vec2 ADVANCE = vec2(.5, -.125);

    time = max(0., time - INPUT_ACTIVE_TIME);
    time *= 1./DEMO_STAGE_DURATION;
    time += dot(uv, ADVANCE) - ADVANCE.y;

#if !DEMO_MODE_HALFTONE
    time += TRANSITION_WIDTH * sqrt(blue_noise(fragCoord, noise, frame).x);
#else
    const float HALFTONE_GRID = 8.;
    float fraction = clamp(1. - (round(time) - time) * (1./TRANSITION_WIDTH), 0., 1.);
    time += TRANSITION_WIDTH * halftone_classic(fragCoord, HALFTONE_GRID, fraction);
#endif // !DEMO_MODE_HALFTONE

    g_demo_stage = int(mod(time, float(DEMO_NUM_STAGES)));
    g_demo_scene = int(time * (1./float(DEMO_NUM_STAGES)));
}

// Function 148
void updateParticle(inout Particle particle, int particleId) {
    bool init = iTime < 0.1;
    //init = true;
    
    vec2 velocity = particle.velocity;
    vec2 position = particle.position + dt * velocity;
    vec2 headPosition = particleHeadPosition(particle, iTime);
    float wavelength = particle.wavelength;
    float phi = particle.phi;
    float lastHit = particle.lastHit;

    // Out-of-bounds reset
    if ((position.x < 0.0 && velocity.x < 0.0)
        || (position.x > iResolution.x/iResolution.y && velocity.x > 0.0)
        || (position.y < 0.0 && velocity.y < 0.0)
        || (position.y > 1.0 && velocity.y > 1.0)) {
        init = true;
    }
    if (init) {
        velocity = vec2(-1.0, -0.5) * 4.0;
        position = vec2(0.9 + float(particleId) * 0.1, 1.1);
        wavelength = 700.0;  // nm
        phi = float(particleId);
        lastHit = 9999.0;
        
        wavelength = 450.0 + 90.0 * float(particleId % 3);
        float theta = 3.1415 * (0.15 + 0.1 * float(particleId % 3));
        vec2 offset = (0.1 - 0.005 * float(particleId)) * vec2(-sin(theta), cos(theta));
        position = vec2(1.0, 1.0) + 0.5 * vec2(cos(theta), sin(theta)) + offset;
        velocity = -vec2(cos(theta), sin(theta)) * celerity / 300.0;
        
        if (iTime < 0.1) {
            position = vec2(0.5);
        }
    }
    
    if (lastHit >= 0.) {
        lastHit += 1.0;
    }
    
    vec2 mouse = getMouse();
    
    // Collision detection
    if (!init && lastHit > 20.0 && headPosition.x < mouse.x/iResolution.y) {
        Header header = getHeader();
        vec2 deltaMouse = (header.oldMouse.xy - mouse.xy)/iResolution.y;
        
        if (velocity.x <= 0.0) {
            position = particleHeadPosition(particle, iTime) + dt * velocity;
            //velocity *= (1.0 - deltaMouse.x * 1.0);
            //velocity.x -= deltaMouse.x * 10.0;
            velocity.x = -velocity.x;
            wavelength = clamp(wavelength + deltaMouse.x * 2000.0, 380., 750.);
            //wavelength = 9999.0;
            float omega = 2.0 * 3.1415 * celerity / wavelength;
            phi = -omega * iTime;
            //phi += 3.1415;
            //lastHit = 0.0;
        }
        
        position.x = mouse.x/iResolution.y;
    }
    
    // Write
    //particle.color = vec4(1.0, 0.5, float(particleId) * 0.2, 1.0);
    particle.color = vec4(colorFromWavelength(wavelength), 1.0);
    particle.position = position;
    particle.velocity = velocity;
    particle.wavelength = wavelength;
    particle.phi = phi;
    particle.lastHit = lastHit;
}

// Function 149
void updateAABB(in ivec2 p, inout vec4 fragColor){
    const float big = 1000.;
    vec4 box = vec4(big, big, -big, -big);
    for(int i=0; i<numBalls; i++) {
        vec3 pos = balls[i].mtx[3].xyz;
        box.z = mix(box.z, pos.x, float(pos.x > box.z));
        box.w = mix(box.w, pos.z, float(pos.z > box.w));
       	box.x = mix(box.x, pos.x, float(pos.x < box.x));
        box.y = mix(box.y, pos.z, float(pos.z < box.y));
    }
    box.xy -= vec2(1.0, 1.0);
    box.zw += vec2(1.0, 1.0);
    fragColor = (p.y == 30 && p.x == 0)? box : fragColor;
    /*
    //Object partitioning along long side of the AABB
    vec4 lbox = box;	//left child
	vec4 rbox = box;	//right child
    if(box.z - box.x > box.w - box.y) {
        lbox.z = -1000.0;
        rbox.x =  1000.0;
        for(int i=0; i<numBalls; i++) {
            float p = balls[i].mtx[3].x;
            lbox.z = mix(lbox.z, p + 1.0, float(p + 1.0 > lbox.z));
            rbox.x = mix(rbox.x, p - 1.0, float(p - 1.0 < rbox.x));
        }
    } else {
        lbox.w = -1000.0;
        rbox.y =  1000.0;
        for(int i=0; i<numBalls; i++) {
            float p = balls[i].mtx[3].y;
            lbox.w = mix(lbox.z, p + 1.0, float(p + 1.0 > lbox.w));
            rbox.y = mix(rbox.y, p - 1.0, float(p - 1.0 < rbox.y));
        }
    }
    
    fragColor = (p.y == 31 && p.x == 0)? lbox : fragColor;
    fragColor = (p.y == 32 && p.x == 0)? rbox : fragColor;*/
}

// Function 150
Particle particle(sampler2D bufA, ivec2 i)
{
    i.x += i.x;
    vec4 d0 = texelFetch(bufA, i, 0)
    , d1 = texelFetch(bufA, i + ivec2(1, 0), 0);
    return Particle(d0.xyz, d1.xyz);
}

// Function 151
void WheelUpdate( inout Engine engine, inout Body body, inout Wheel wheel, float dT )
{
    vec3 vWheelWorld = ObjToWorld( wheel.vBodyPos, body.mRot) + body.vPos;
    vec3 vWheelDown = ObjToWorld( vec3(0.0, -1.0, 0.0), body.mRot);
    
    float fSuspensionTravel = 0.25;
    C_Intersection intersection = WheelTrace( vWheelWorld, vWheelDown, wheel );
    
    float fTravel = clamp( intersection.fDist - wheel.fRadius, 0.0, fSuspensionTravel);
        
    // Apply suspension force
    // Simple spring-damper
    // (No anti-roll bar)
    float fWheelExt = fTravel / fSuspensionTravel;

    wheel.fOnGround = 1.0 - fWheelExt;
    
    float delta = (wheel.fExtension - fTravel) / fSuspensionTravel;

    float fForce = (1.0 - fWheelExt) * 5000.0 + delta * 15000.0;

    vec3 vForce = Vec3Perp( intersection.vNormal, vWheelDown) * fForce;
    //BodyApplyForce( body, vWheelWorld, vForce );                

    // Apply Tyre force

    // Super simplification of wheel / drivetrain / engine / tyre contact
    // ignoring engine / wheel angular momentum       

    // Figure out how contact patch is moving in world space
    vec3 vIntersectWorld = intersection.vPos;
    wheel.vContactPos = vIntersectWorld.xz;
    vec3 vVelWorld = BodyPointVelocity( body, vIntersectWorld );

    // Transform to body space
    vec3 vVelBody = WorldToObj( vVelWorld, body.mRot );

    // Transform to wheel space
    vec3 vVelWheel = RotY( vVelBody, wheel.fSteer );

    float fWScale = wheel.fRadius;

    float fWheelMOI = 20.0;
    if ( wheel.bIsDriven )
    {
        fWheelMOI = 30.0;

        // consta-torque mega engine
        if( KeyIsPressed( KEY_W ) )
        {
            wheel.fAngularVelocity += 2.0;
        }        

        if( KeyIsPressed( KEY_S ) )
        {
            wheel.fAngularVelocity -= 2.0;
        }        
    }

    if( KeyIsPressed( KEY_SPACE ) )
    {
        wheel.fAngularVelocity = 0.0; // insta-grip super brake
    }        

    vVelWheel.z -= wheel.fAngularVelocity * fWScale;

    vec3 vForceWheel = vVelWheel * body.fMass;

    // Hacked 'slip angle'
    //vForceWheel.x /=  1.0 + abs(wheel.fAngularVelocity * fWScale) * 0.1;

    float fLimit = 9000.0 * (1.0 - fWheelExt);

    wheel.fSkid = ClampTyreForce( vForceWheel, fLimit );    
    
    //vVelWheel.z += wheel.fAngularVelocity * fWScale;
    vec3 vForceBody = RotY( vForceWheel, -wheel.fSteer );

    // Apply force back on wheel

    wheel.fAngularVelocity += ((vForceWheel.z / fWScale) / fWheelMOI) * dT;

    vec3 vForceWorld = ObjToWorld( vForceBody, body.mRot );

    // cancel in normal dir
    vForceWorld = Vec3Parallel( vForceWorld, intersection.vNormal );

    vForce -= vForceWorld;
    //BodyApplyForce( body, vIntersectWorld, -vForceWorld );        
    
    BodyApplyForce( body, vIntersectWorld, vForce );        

    wheel.fExtension = fTravel;
    wheel.fRotation += wheel.fAngularVelocity * dT;    
}

// Function 152
float getParticleStartTime(int partnr)
{
    return start_time*random(float(partnr*2));
}

// Function 153
vec4 updateToolState() {
    vec4 previousToolState = fetchState(STATE_LOCATION_TOOL);
    
    if (isKeyPressed(KEY_1)) {
        return vec4(STATE_TOOL_TYPE_BRUSH, 0.0, 0.0, 1.0);
    } else if (isKeyPressed(KEY_2)) {
        return vec4(STATE_TOOL_TYPE_BRUSH, 1.0, 0.0, 1.0);
    } else if (isKeyPressed(KEY_3)) {
        return vec4(STATE_TOOL_TYPE_BRUSH, 2.0, 0.0, 1.0);
    } else if (isKeyPressed(KEY_E)) {
        return vec4(STATE_TOOL_TYPE_ERASER, 0.0, 0.0, 1.0);
    }
    
    return previousToolState;
}

// Function 154
vec2 getParticlePosition_mp()
{
   vec2 ppos = vec2(harms(main_x_freq, main_x_amp, main_x_phase, time2), harms(main_y_freq, main_y_amp, main_y_phase, time2)) + middlepoint;
   return gen_scale*ppos;
}

// Function 155
void UpdatePosition(vec2 uv, out vec4 result)
{
    if (iFrame == 0)
    {
        result = vec4(vec3(0), GALAXY_RADIUS * 2.0);
    }
    else
    {
        vec4 cs = texture(iChannel1, uv);
        //--track
        vec3 tgt = CalculateCameraTarget();
        float dt = GALAXY_RADIUS / distance(tgt, cs.xyz) * iTimeDelta * trackSpeed;
        cs.xyz = mix(cs.xyz, tgt, dt);
        //--scale
        float ds = iTimeDelta * scaleSpeed;
        if (readKey(keyE))      cs.w *= 1.0 - ds;
        else if (readKey(keyQ)) cs.w *= 1.0 + ds;
        
        result = cs;
    }
}

// Function 156
vec3 particles( vec2 pos )
{
	
	vec3 c = vec3( 0, 0, 0 );
	
	float noiseFactor = fBm( pos, 0.01, 0.1);
	
	for( float i = 1.0; i < ParticleCount+1.0; ++i )
	{
		float cs = cos( Time * HorizontalSpeed * (i/ParticleCount) + noiseFactor ) * HorizontalAmplitude;
		float ss = sin( Time * VerticleSpeed   * (i/ParticleCount) + noiseFactor ) * VerticleAmplitude;
		vec2 origin = vec2( cs , ss );
		
		float t = sin( Time * ParticleBreathingSpeed * i ) * 0.5 + 0.5;
		float particleSize = mix( ParticleMinSize, ParticleMaxSize, t );
		float d = clamp( sin( length( pos - origin )  + particleSize ), 0.0, particleSize);
		
		float t2 = sin( Time * ParticleColorChangeSpeed * i ) * 0.5 + 0.5;
		vec3 color = mix( ParticleColor1, ParticleColor2, t2 );
		c += color * pow( d, 10.0 );
	}
	
	return c;
}

// Function 157
void update_game_rules(inout vec4 fragColor, vec2 fragCoord)
{
    if (is_inside(fragCoord, ADDR_RANGE_TARGETS) > 0.)
    {
        Target target;
        Transitions transitions;
        GameState game_state;
        
        from_vec4(target, fragColor);
        LOAD_PREV(game_state);
        LOAD_PREV(transitions);
        float level = floor(game_state.level);
        float index = floor(fragCoord.x - ADDR_RANGE_TARGETS.x);

        if (target.level != level)
        {
            target.level = level;
            target.shot_no = transitions.shot_no;
            if (level > 0. || index == SKY_TARGET_OFFSET.x)
            	target.hits = 0.;
            to_vec4(fragColor, target);
            return;
        }

        // already processed this shot?
        if (target.shot_no == transitions.shot_no)
            return;
        target.shot_no = transitions.shot_no;
        
        // disable popping during game over animation
        if (game_state.level < 0. && game_state.level != floor(game_state.level))
            return;

        float target_material = index < float(NUM_TARGETS) ? index + float(BASE_TARGET_MATERIAL) : float(MATERIAL_SKY1);
        int hits = 0;

        // The smart thing to do here would be to split the sum over several frames
        // in a binary fashion, but the shader is already pretty complicated,
        // so to make my life easier I'll go with a naive for loop.
        // To save face, let's say I'm doing this to avoid the extra latency
        // of log2(#pellets) frames the smart method would incur...

        for (float f=0.; f<ADDR_RANGE_SHOTGUN_PELLETS.z; ++f)
        {
            vec4 pellet = load(ADDR_RANGE_SHOTGUN_PELLETS.xy + vec2(f, 0.));
            hits += int(pellet.w == target_material);
        }
        
        // sky target is all or nothing
        if (target_material == float(MATERIAL_SKY1))
            hits = int(hits == int(ADDR_RANGE_SHOTGUN_PELLETS.z));
        
        target.hits += float(hits);
        to_vec4(fragColor, target);

        return;
    }
    
    if (is_inside(fragCoord, ADDR_GAME_STATE) > 0.)
    {
        const float
            ADVANCE_LEVEL			= 1. + LEVEL_WARMUP_TIME * .1,
        	FIRST_ROUND_DURATION	= 15.,
        	MIN_ROUND_DURATION		= 6.,
        	ROUND_TIME_DECAY		= -1./8.;
        
        GameState game_state;
        from_vec4(game_state, fragColor);
        
        MenuState menu;
        LOAD(menu);
        float time_delta = menu.open > 0 ? 0. : iTimeDelta;

        if (game_state.level <= 0.)
        {
            float level = ceil(game_state.level);
            if (level < 0. && game_state.level != level)
            {
                game_state.level = min(level, game_state.level + time_delta * .1);
                to_vec4(fragColor, game_state);
                return;
            }
            Target target;
            LOADR(SKY_TARGET_OFFSET, target);
            if (target.hits > 0. && target.level == game_state.level)
            {
                game_state.level = ADVANCE_LEVEL;
                game_state.time_left = FIRST_ROUND_DURATION;
                game_state.targets_left = float(NUM_TARGETS);
            }
        }
        else
        {
            float level = floor(game_state.level);
            if (level != game_state.level)
            {
                game_state.level = max(level, game_state.level - time_delta * .1);
                to_vec4(fragColor, game_state);
                return;
            }
            
            game_state.time_left = max(0., game_state.time_left - time_delta);
            if (game_state.time_left == 0.)
            {
                game_state.level = -(level + BALLOON_SCALEIN_TIME * .1);
                to_vec4(fragColor, game_state);
                return;
            }
            
            float targets_left = 0.;
            Target target;
            for (vec2 addr=vec2(0); addr.x<ADDR_RANGE_TARGETS.z-1.; ++addr.x)
            {
                LOADR(addr, target);
                if (target.hits < ADDR_RANGE_SHOTGUN_PELLETS.z * .5 || target.level != level)
                    ++targets_left;
            }
            
            if (floor(game_state.targets_left) != targets_left)
                game_state.targets_left = targets_left + HUD_TARGET_ANIM_TIME * .1;
            else
                game_state.targets_left = max(floor(game_state.targets_left), game_state.targets_left - time_delta * .1);

            if (targets_left == 0.)
            {
                game_state.level = level + ADVANCE_LEVEL;
                game_state.time_left *= .5;
                game_state.time_left += mix(MIN_ROUND_DURATION, FIRST_ROUND_DURATION, exp2(level*ROUND_TIME_DECAY));
                game_state.targets_left = float(NUM_TARGETS);
            }
        }

        to_vec4(fragColor, game_state);
        return;
    }
}

// Function 158
void UpdatePosition(inout vec4 pos)
{
    vec3 lastPos    = GetLastPosition();
    vec3 dirForward = GetLastForward();
    vec3 dirRight   = cross(dirForward, vec3(0.0, 1.0, 0.0));
    
    float keyForward = max(IsKeyPressed(KEY_W), IsKeyPressed(KEY_UP));
    float keyDown    = max(IsKeyPressed(KEY_S), IsKeyPressed(KEY_DOWN));
    float keyLeft    = max(IsKeyPressed(KEY_A), IsKeyPressed(KEY_LEFT));
    float keyRight   = max(IsKeyPressed(KEY_D), IsKeyPressed(KEY_RIGHT));
    
    float pressForward = keyForward - keyDown;
    float pressRight   = keyLeft - keyRight;
    
    vec3  direction = (dirForward * pressForward) + (-dirRight * pressRight);
    float delta     = MoveSpeed * iTimeDelta * step(0.001, length(direction));
    
    if(delta > 0.0)
    {
        delta    *= max(1.0, IsKeyPressed(KEY_SHIFT) * BoostModifier);
        direction = normalize(direction);
    
        pos.xyz = lastPos + (direction * delta);
        
#ifndef FREE_FLY
        pos.y = StartPos.y;
#endif
    }
}

// Function 159
vec4 updatePressed()
{
    vec4 result = vec4(0.0, 0.0, 0.0, 0.0);
    if (isAnythingPressed())
        result.x = 1.0;
    
    if (isToggled(KEY_T))
        result.y = 1.0;
    
    if (isToggled(KEY_H))
        result.z = 1.0;
    
    return result;
}

// Function 160
vec3 layeredParticles(vec2 uv, float radiusMod, float sizeMod, float alphaMod, int layers, vec2 dotProportions, float dotRotation, float animationOffset) 
{ 
    vec3 particles = vec3(0);
    float size = 1.0;
    float alpha = 1.0;
    float radius = 0.04;
    vec2 offset = vec2(0.0);
    vec3 startColor = myHue(iTime * 0.05 + animationOffset * 0.5) + 0.3;
    vec3 endColor = myHue(iTime * 0.05 + 0.3 + animationOffset * 0.5) + 0.3 + animationOffset;
    vec3 color = startColor;
    for (int i = 0; i < layers; i++)
    {
		particles += bokehParticles(
            (uv * size + vec2(sin(iTime * 0.3), cos(iTime * 0.3)) * 6.0 * intensiveMomentMove() - animationOffset * 0.3) + offset,
            radius, 
            dotProportions, 
            dotRotation + offset.x, 
            animationOffset
        	) * alpha * color;
        color = mix(startColor, endColor, float(i+1) / float(layers));
        offset += hash2_2(vec2(alpha, alpha)) * 10.0;
        alpha *= alphaMod;
        size *= sizeMod;
        radius *= radiusMod;
    }
    return particles;
}

// Function 161
void update_input(inout vec4 fragColor, vec2 fragCoord)
{
    float allow_input	= is_input_enabled();
    vec4 pos			= (iFrame==0) ? DEFAULT_POS : load(ADDR_POSITION);
    vec4 angles			= (iFrame==0) ? DEFAULT_ANGLES : load(ADDR_ANGLES);
    vec4 old_pos		= (iFrame==0) ? DEFAULT_POS : load(ADDR_CAM_POS);
    vec4 old_angles		= (iFrame==0) ? DEFAULT_ANGLES : load(ADDR_CAM_ANGLES);
    vec4 velocity		= (iFrame==0) ? vec4(0) : load(ADDR_VELOCITY);
    vec4 ground_plane	= (iFrame==0) ? vec4(0) : load(ADDR_GROUND_PLANE);
    bool thumbnail		= (iFrame==0) ? true : (int(load(ADDR_RESOLUTION).z) & RESOLUTION_FLAG_THUMBNAIL) != 0;
    
    Transitions transitions;
    LOAD_PREV(transitions);
    
    MenuState menu;
    LOAD_PREV(menu);
    if (iFrame > 0 && menu.open > 0)
        return;
    
    if (iFrame == 0 || is_demo_mode_enabled(thumbnail))
        allow_input = 0.;

    if (allow_input > 0. && fire_weapon(fragColor, fragCoord, old_pos.xyz, old_angles.xyz, transitions.attack, transitions.shot_no))
        return;
    
    Options options;
    LOAD_PREV(options);

    angles.w = max(0., angles.w - iTimeDelta);
    if (angles.w == 0.)
    	angles.y = mix(angles.z, angles.y, exp2(-8.*iTimeDelta));

	vec4 mouse_status	= (iFrame==0) ? vec4(0) : load(ADDR_PREV_MOUSE);
    if (allow_input > 0.)
    {
        float mouse_lerp = MOUSE_FILTER > 0. ?
            min(1., iTimeDelta/.0166 / (MOUSE_FILTER + 1.)) :
        	1.;
        if (iMouse.w > 0.)
        {
            float mouse_y_scale = INVERT_MOUSE != 0 ? -1. : 1.;
            if (test_flag(options.flags, OPTION_FLAG_INVERT_MOUSE))
                mouse_y_scale = -mouse_y_scale;
            float sensitivity = SENSITIVITY * exp2((options.sensitivity - 5.) * .5);
            
            if (iMouse.w > mouse_status.w)
                mouse_status = iMouse;
            vec2 mouse_delta = (iMouse.w > mouse_status.w) ?
                vec2(0) : mouse_status.xy - iMouse.xy;
            mouse_delta.y *= -mouse_y_scale;
            angles.xy += 360. * sensitivity * mouse_lerp / max_component(iResolution.xy) * mouse_delta;
            angles.z = angles.y;
            angles.w = AUTOPITCH_DELAY;
        }
        mouse_status = vec4(mix(mouse_status.xy, iMouse.xy, mouse_lerp), iMouse.zw);
    }
    
    float strafe = cmd_strafe();
    float run = (cmd_run()*.5 + .5) * allow_input;
    float look_side = cmd_look_left() - cmd_look_right();
    angles.x += look_side * (1. - strafe) * run * TURN_SPEED * iTimeDelta;
    float look_up = cmd_look_up() - cmd_look_down();
    angles.yz += look_up * run * TURN_SPEED * iTimeDelta;
    // delay auto-pitch for a bit after looking up/down
    if (abs(look_up) > 0.)
        angles.w = .5;
    if (cmd_center_view() * allow_input > 0.)
        angles.zw = vec2(0);
    angles.x = mod(angles.x, 360.);
    angles.yz = clamp(angles.yz, -80., 80.);

#if NOCLIP
    const bool noclip = true;
#else
    bool noclip = test_flag(options.flags, OPTION_FLAG_NOCLIP);
#endif

    mat3 move_axis = rotation(vec3(angles.x, noclip ? angles.y : 0., 0));

    vec3 input_dir		= vec3(0);
    input_dir			+= (cmd_move_forward() - cmd_move_backward()) * move_axis[1];
    float move_side		= cmd_move_right() - cmd_move_left();
    move_side			= clamp(move_side - look_side * strafe, -1., 1.);
    input_dir	 		+= move_side * move_axis[0];
    input_dir.z 		+= (cmd_move_up() - cmd_move_down());
    float wants_to_move = step(0., dot(input_dir, input_dir));
    float wish_speed	= WALK_SPEED * allow_input * wants_to_move * (1. + -.5 * run);

    float lava_dist		= max_component(abs(pos.xyz - clamp(pos.xyz, LAVA_BOUNDS[0], LAVA_BOUNDS[1])));

	if (noclip)
    {
        float friction = mix(NOCLIP_STOP_FRICTION, NOCLIP_START_FRICTION, wants_to_move);
        float velocity_blend = exp2(-friction * iTimeDelta);
        velocity.xyz = mix(input_dir * wish_speed, velocity.xyz, velocity_blend);
        pos.xyz += velocity.xyz * iTimeDelta;
        ground_plane = vec4(0);
    }
    else
    {
        // if not ascending, allow jumping when we touch the ground
        if (input_dir.z <= 0.)
            velocity.w = 0.;
        
        input_dir.xy = safe_normalize(input_dir.xy);
        
        bool on_ground = is_touching_ground(pos.xyz, ground_plane);
        if (on_ground)
        {
            // apply friction
            float speed = length(velocity.xy);
            if (speed < 1.)
            {
                velocity.xy = vec2(0);
            }
            else
            {
                float drop = max(speed, STOP_SPEED) * GROUND_FRICTION * iTimeDelta;
                velocity.xy *= max(0., speed - drop) / speed;
            }
        }
        else
        {
            input_dir.z = 0.;
        }

        if (lava_dist <= 0.)
            wish_speed *= .25;

        // accelerate
		float current_speed = dot(velocity.xy, input_dir.xy);
		float add_speed = wish_speed - current_speed;
		if (add_speed > 0.)
        {
			float accel = on_ground ? GROUND_ACCELERATION : AIR_ACCELERATION;
			float accel_speed = min(add_speed, accel * iTimeDelta * wish_speed);
            velocity.xyz += input_dir * accel_speed;
		}

        if (on_ground)
        {
            velocity.z -= (GRAVITY * .25) * iTimeDelta;	// slowly slide down slopes
            velocity.xyz -= dot(velocity.xyz, ground_plane.xyz) * ground_plane.xyz;

            if (transitions.stair_step <= 0.)
                transitions.bob_phase = fract(transitions.bob_phase + iTimeDelta * (1./BOB_CYCLE));

            update_ideal_pitch(pos.xyz, move_axis[1], velocity.xyz, angles.z);

            if (input_dir.z > 0. && velocity.w <= 0.)
            {
                velocity.z += JUMP_SPEED;
                // wait for the jump key to be released
                // before jumping again (no auto-hopping)
                velocity.w = 1.;
            }
        }
        else
        {
            velocity.z -= GRAVITY * iTimeDelta;
        }

        if (is_inside(fragCoord, ADDR_RANGE_PHYSICS) > 0.)
            slide_move(pos.xyz, velocity.xyz, ground_plane, transitions.stair_step);
    }

    bool teleport = touch_tele(pos.xyz, 16.);
    if (!noclip)
    	teleport = teleport || ((DEFAULT_POS.z - pos.z) > VIEW_DISTANCE); // falling too far below the map

    if (cmd_respawn() * allow_input > 0. || teleport)
    {
        pos = vec4(DEFAULT_POS.xyz, iTime);
        angles = teleport ? vec4(0) : DEFAULT_ANGLES;
        velocity.xyz = vec3(0, teleport ? WALK_SPEED : 0., 0);
        ground_plane = vec4(0);
        transitions.stair_step = 0.;
        transitions.bob_phase = 0.;
    }
    
    // smooth stair stepping
    transitions.stair_step = max(0., transitions.stair_step - iTimeDelta * STAIR_CLIMB_SPEED);

    vec4 cam_pos = pos;
    cam_pos.z -= transitions.stair_step;
    
    // bobbing
    float speed = length(velocity.xy);
    if (speed < 1e-2)
        transitions.bob_phase = 0.;
    cam_pos.z += clamp(speed * BOB_SCALE * (.3 + .7 * sin(TAU * transitions.bob_phase)), -7., 4.);
    
    vec4 cam_angles = vec4(angles.xy, 0, 0);
    
    // side movement roll
    cam_angles.z += clamp(dot(velocity.xyz, move_axis[0]) * (1./ROLL_SPEED), -1., 1.) * ROLL_ANGLE;

    // lava pain roll
    if (lava_dist <= 32.)
    	cam_angles.z += 5. * clamp(fract(iTime*4.)*-2.+1., 0., 1.);
    
    // shotgun recoil
    cam_angles.y += linear_step(.75, 1., transitions.attack) * RECOIL_ANGLE;

    store(fragColor, fragCoord, ADDR_POSITION, pos);
    store(fragColor, fragCoord, ADDR_ANGLES, angles);
    store(fragColor, fragCoord, ADDR_CAM_POS, cam_pos);
    store(fragColor, fragCoord, ADDR_CAM_ANGLES, cam_angles);
    store(fragColor, fragCoord, transitions);
    store(fragColor, fragCoord, ADDR_PREV_CAM_POS, old_pos);
    store(fragColor, fragCoord, ADDR_PREV_CAM_ANGLES, old_angles);
    store(fragColor, fragCoord, ADDR_PREV_MOUSE, mouse_status);
    store(fragColor, fragCoord, ADDR_VELOCITY, velocity);
    store(fragColor, fragCoord, ADDR_GROUND_PLANE, ground_plane);
}

// Function 162
void UpdateEnemyBullet( inout vec4 enemyBullet, vec4 playerHitBox, float screenWidth, float screenHeight )
{
    if ( !Collide( enemyBullet.xy, BULLET_SIZE, vec2( gCamera.x + screenWidth * 0.5, 0.0 ), vec2( screenWidth, screenHeight ) ) )
    {
        enemyBullet.x = 0.0;
    }
    
	if ( enemyBullet.x > 0.0 )
    {
    	enemyBullet.xy += enemyBullet.zw * ENEMY_BULLET_SPEED;
    }
   
	if ( Collide( playerHitBox.xy, playerHitBox.zw, enemyBullet.xy, BULLET_SIZE ) )
    {
        PlayerHit( playerHitBox );
        enemyBullet.x = 0.0;
    }        
}

// Function 163
vec4 getParticle(ivec2 id)
{
    return texelFetch(iChannel0, ivec2(id), 0);
}

// Function 164
void updateCaveAnim(inout int gAuxFrame, inout vec4 gCamPos, inout int gCaveState, inout float gFade, ivec2 gPlayerCoord,
                    inout float gStripesAlpha, inout float gTimeLeft,
                    inout GameData gd, inout int scoreToAdd
                    )
{
    bool isIntermissionLevel = isIntermission(gd.gCave);
    updateCameraPos(gCamPos, gPlayerCoord);
    gStripesAlpha = max(0.0, gStripesAlpha - STRIPES_DELTA);

    if (!isState(gCaveState, CAVE_STATE_FADE_IN) &&
        !isState(gCaveState, CAVE_STATE_SPAWNING) &&
        !isState(gCaveState, CAVE_STATE_EXITED) &&
        !isState(gCaveState, CAVE_STATE_GAME_OVER) &&
        !isState(gCaveState, CAVE_STATE_TIME_OUT) &&
        !isState(gCaveState, CAVE_STATE_PAUSED) &&
        !isState(gCaveState, CAVE_STATE_FADE_OUT))
    {
        gTimeLeft = max(0.0, gTimeLeft - ANIM_FRAME_DURATION);
        if (gTimeLeft == 0.0)
        {
            setState(gCaveState, CAVE_STATE_TIME_OUT);
            gd.gLives = (isIntermissionLevel || !isState(gCaveState, CAVE_STATE_ALIVE)) ? gd.gLives : gd.gLives - 1;
            gAuxFrame = gd.gFrames.x;
        }
    }

    // cave fade in
    if (isState(gCaveState, CAVE_STATE_FADE_IN))
    {
        gFade = saturate(gFade - FADE_IN_DELTA);
        if (gFade == 0.0)
        {
            delState(gCaveState, CAVE_STATE_FADE_IN);
        }
    }

    // transferring time to score when exited
    if (isState(gCaveState, CAVE_STATE_EXITED) && (gAuxFrame == INT_MAX))
    {
        const float scoreIncrement = 2.0;
        int scoreAddition = int(round(gTimeLeft - max(0.0, gTimeLeft - scoreIncrement)));  // 2 per anim frame, 1 per tv scan
        gTimeLeft = max(0.0, gTimeLeft - scoreIncrement);
        scoreToAdd += scoreAddition;

        if (gTimeLeft <= 0.0)
        {
            gAuxFrame = gd.gFrames.x;
        }
    }
    else if (!isState(gCaveState, CAVE_STATE_ALIVE) && gAuxFrame == INT_MAX)
    {
        if (!isIntermissionLevel)
        {
            gd.gLives -= 1;
        }
        gAuxFrame = gd.gFrames.x;
    }

    bool isAuxTimeSet = gAuxFrame != INT_MAX;
    int auxTimeDelta = gd.gFrames.x - gAuxFrame;
    bool isFadingOut = false;

    if (!isState(gCaveState, CAVE_STATE_GAME_OVER))
    {
        isFadingOut = isFadingOut || (  // Exited
            isState(gCaveState, CAVE_STATE_EXITED) &&
            isAuxTimeSet &&
            (auxTimeDelta > EXIT_COOLDOWN_AF)
        );
        isFadingOut = isFadingOut || (  // Player Death
            !isState(gCaveState, CAVE_STATE_ALIVE) &&
            isAuxTimeSet &&
            (auxTimeDelta > DEATH_COOLDOWN_AF) &&
            KEY_DOWN(KEY_CTRL)
        );
        isFadingOut = isFadingOut || (  // Time Out
            isState(gCaveState, CAVE_STATE_TIME_OUT) &&
            isAuxTimeSet &&
            (auxTimeDelta > DEATH_COOLDOWN_AF) &&
            KEY_DOWN(KEY_CTRL)
        );
        if (isFadingOut && (gd.gLives == 0))  // Game Over Start Timer
        {
            isFadingOut = false;
            setState(gCaveState, CAVE_STATE_GAME_OVER);
            gAuxFrame = gd.gFrames.x;
            auxTimeDelta = 0;
        }
    }

    isFadingOut = isFadingOut || (  // Game Over
        isState(gCaveState, CAVE_STATE_GAME_OVER) &&
        (auxTimeDelta > GAME_OVER_COOLDOWN_AF)
    );

    if (isFadingOut)
    {
        setState(gCaveState, CAVE_STATE_FADE_OUT);
    }

    bool isGameOver = false;
    bool isNextLevel = false;

    if (isState(gCaveState, CAVE_STATE_FADE_OUT))
    {
        gFade = saturate(gFade + FADE_OUT_DELTA);
        if (gFade == 1.0)
        {
            gd.gIsCaveInit = true;

            if (isState(gCaveState, CAVE_STATE_GAME_OVER))
            {
                isGameOver = true;
            }
            else if (isState(gCaveState, CAVE_STATE_EXITED) || isIntermissionLevel)
            {
                isNextLevel = true;
            }
        }
    }

    if (isGameOver)
    {
        gd.gLives = NUMBER_OF_LIVES;
        gd.gCave = 1;
        gd.gLevel = 1;
        gd.gGameState = GAME_STATE_START_SCREEN;
    }

    if (isNextLevel)
    {
        gd.gCave += 1;
        if (gd.gCave > 20)
        {
            gd.gCave = 1;
            gd.gLevel = max(((gd.gLevel + 1) % 6), 1);
        }
    }
}

// Function 165
void initParticle(in vec2 fragCoord, inout vec2 particlePrevPosition, inout vec2 particleCurrPosition)
{
	particleCurrPosition = randVec2(fragCoord) * iResolution.xy;
    particlePrevPosition = particleCurrPosition - randNrm2(fragCoord) * particlesSize * 0.0625;
}

// Function 166
void initParticle(inout Particle p, int idx)
{
    int numXY=int(pow(float(NumParticles),.3333)*.95);
    p.pos=vec3(idx%numXY,(idx/numXY)%numXY,idx/numXY/numXY);
    p.pos-=vec3(numXY/2,numXY/2,0*numXY/2);
    p.vel=vec3(0);
    p.idx=idx;
    p.sidx=(idx*7)%NumParticles;
    // init all neighbours to empty
    p.nn[0]=-1;
    p.nn[1]=-1;
    p.nn[2]=-1;
    p.nn[3]=-1;
}

// Function 167
void update_closest(inout vec4 pd, in vec3 pi, in vec3 q) {
    
    float di = length(pi-q);
    
    if (di < pd.w) { 
        pd.xyz = pi;
        pd.w = di;
    }
    
}

// Function 168
void icon_dist_update(inout vec2 blk_gray, 
                      float d, bool enable) {
    
    if (enable) {
        blk_gray.x = min(blk_gray.x, d);
    } else {
        blk_gray.y = min(blk_gray.y, d);
    }
    
}

// Function 169
void UpdateRotation(inout vec4 rot)
{
    if (rot.w == 0.0)
    {
        rot = QUAT_IDENTITY;
    }
    else
    {
        //--rotate
        float dt = DELTA_TIME * TIME_SCALE;
        float dr = dt * ROTATE_SPEED;
        vec3 eu = vec3(0);
        //key
        int btn = HoldRotateButton(iMouse, iResolution.xy);
        if (readKey(keyW) || readKey(keyUp))    eu.x -= 1.0;
        if (readKey(keyS) || readKey(keyDown))  eu.x += 1.0;
        if (readKey(keyA) || readKey(keyLeft))  eu.y -= 1.0;
        if (readKey(keyD) || readKey(keyRight)) eu.y += 1.0;
        if (readKey(keyE) || btn == 0) eu.z -= 1.0;
        if (readKey(keyQ) || btn == 1) eu.z += 1.0;
        eu *= dr;
        //mouse
        vec3 lm = GetLastMouse();
        vec2 dn = iMouse.xy - iMouse.zw;
        vec2 dp = iMouse.xy - lm.xy;
        vec2 dm = iMouse.z > lm.z ? dn : dp;
        dm = vec2(dm.x, -dm.y) / iResolution.y;
        eu.xy += dm.yx * (dr * 100.0);
        
        rot = QuatMul(rot, QuatEA(eu));
        
        vec4 pos = GetCameraPositionSpeed();
        vec3 g = QuatMul(QuatInv(rot), CalculateSumAcceleration(pos.xyz));
        if (length(g) > 0.0)
        {
            vec3 nv = normalize(CalculateVelocity(FWD * pos.w, g, dt * TimeSlow(length(g))));
            rot = QuatMul(rot, QuatFT(FWD, nv));
        }
        
        UniverseWards(pos.xyz, rot);
        
        rot = normalize(rot);
    }
}

// Function 170
vec3 getParticleColor(in vec2 p) {
    return normalize(vec3(0.1) + texture(iChannel2, p * 0.0001 + iTime * 0.005).rgb);
}

// Function 171
float SDF_particle(vec3 p0, vec3 p)
{
    particle point = get(fakech0, p0);
    return length(point.pos.xyz - p) - sphere_rad;
}

// Function 172
particle unpackParticle(vec2 p){
    uint A = floatBitsToUint(p.x);
    uint B = floatBitsToUint(p.y);
    uint n = (A >> 30) & 0x1u;
    uint r = (A >> 17) & 0x1FFFu;
    uint g = (A >> 4) & 0x1FFFu;
    uint b = ((B >> 22) & 0x1FFu) | ((A & 0xFu) << 9);
    uint y = B & 0x7FFu;
    uint x = (B >> 11) & 0x7FFu;
    return particle(bool(n), vec2(x,y)+.5,(vec3(r,g,b)-1000.)/7000.);
}

// Function 173
vec2 particleCoordFromRootUV(vec2 rootUV){
    return rotate(vec2(0.0, 1.0), globalTime * 3.0 * (hash12(rootUV) - 0.5)) * (0.5 - PARTICLE_SIZE) + rootUV + 0.5;
}

// Function 174
void add_particles
(
    inout vec4 fragColor, vec2 fragCoordNDC,
    vec3 camera_pos, mat3 view_matrix, float depth,
    float attack, float teleport_time
)
{
#if RENDER_PARTICLES
    const float
        WORLD_RADIUS		= 1.5,
    	MIN_PIXEL_RADIUS	= 2.,
    	SPAWN_INTERVAL		= .1,
    	LIFESPAN			= 1.,
    	LIFESPAN_VARIATION	= .5,
    	MAX_GENERATIONS		= ceil(LIFESPAN / SPAWN_INTERVAL),
    	BUNCH				= 4.,
        ATTACK_FADE_START	= .85,
        ATTACK_FADE_END		= .5,
        PELLET_WORLD_RADIUS	= .5;
    const vec3 SPREAD		= vec3(3, 3, 12);
    
    add_teleporter_effect(fragColor, fragCoordNDC, camera_pos, teleport_time);
    
    float depth_scale = MIN_PIXEL_RADIUS * g_downscale/iResolution.x;
    depth *= VIEW_DISTANCE;
    
    // shotgun pellets //
    if (attack > ATTACK_FADE_END)
    {
        // Game stage advances immediately after the last balloon is popped.
        // When we detect a warmup phase (fractional value for game stage)
        // we have to use the previous stage for coloring the particles.

        vec4 game_state = load(ADDR_GAME_STATE);
        float level = floor(abs(game_state.x));
        if (game_state.x != level && game_state.x > 0.)
            --level;

        float fade = sqrt(linear_step(ATTACK_FADE_START, ATTACK_FADE_END, attack));
        vec3 base_pos = camera_pos;
        base_pos.z += (1. - attack) * 8.;

        float num_pellets = ADDR_RANGE_SHOTGUN_PELLETS.z + min(iTime, 0.);
        for (float f=0.; f<num_pellets; ++f)
        {
            vec2 address = ADDR_RANGE_SHOTGUN_PELLETS.xy;
            address.x += f;
            vec2 props = hash2(address);
            if (props.x <= fade)
                continue;
            vec4 pellet = load(address);
            int hit_material = int(pellet.w + .5);
            if (is_material_sky(hit_material))
                continue;
            vec3 pos = pellet.xyz - base_pos;
            float particle_depth = dot(pos, view_matrix[1]) + (-2.*PELLET_WORLD_RADIUS);
            if (particle_depth < 0. || particle_depth > depth)
                continue;
            vec2 ndc_pos = vec2(dot(pos, view_matrix[0]), dot(pos, view_matrix[2]));
            float radius = max(PELLET_WORLD_RADIUS, particle_depth * depth_scale);
            vec2 delta = abs(ndc_pos - fragCoordNDC * particle_depth);
            if (max(delta.x, delta.y) <= radius)
            {
                fragColor = vec4(vec3(.5 * (1.-sqr(props.y))), 0.);
                depth = particle_depth;
			    if (is_material_balloon(hit_material))
                    fragColor.rgb *= get_balloon_color(hit_material, level).rgb * 2.;
            }
        }
    }
    
    Fireball fireball;
    get_fireball_props(g_animTime, fireball);

	#if CULL_PARTICLES
    {
        vec3 mins, maxs;
        get_fireball_bounds(fireball, camera_pos, view_matrix, 40., mins, maxs);
        if (maxs.z <= 0. || mins.z > depth)
            return;

        float slack = 8./mins.z + depth_scale;
        mins.xy -= slack;
        maxs.xy += slack;
        if (mins.z > 0. && is_inside(fragCoordNDC, vec4(mins.xy, maxs.xy - mins.xy)) < 0.)
            return;
    }
	#endif
    
	#if DEBUG_PARTICLE_CULLING
    {
        fragColor.rgb = mix(fragColor.rgb, vec3(1.), .25);
    }
	#endif
    
    float end_time = min(get_landing_time(fireball), g_animTime);
    float end_generation = ceil((end_time - fireball.launch_time) * (1./SPAWN_INTERVAL) - .25);
    
    for (float generation=max(0., end_generation - 1. - MAX_GENERATIONS); generation<end_generation; ++generation)
    {
        float base_time=fireball.launch_time + generation * SPAWN_INTERVAL;
        float base_age = (g_animTime - base_time) * (1./LIFESPAN) + (LIFESPAN_VARIATION * -.5);
        if (base_age > 1.)
            continue;
        
        vec3 base_pos = get_fireball_offset(base_time, fireball) + FIREBALL_ORIGIN;

        for (float f=0.; f<BUNCH; ++f)
        {
            float age = base_age + hash(f + base_time) * LIFESPAN_VARIATION;
            if (age > 1.)
                continue;
            vec3 pos = base_pos - camera_pos;
            pos += hash3(base_time + f*(SPAWN_INTERVAL/BUNCH)) * (SPREAD*2.) - SPREAD;
            pos.z += base_age * 32.;
            float particle_depth = dot(pos, view_matrix[1]);
            if (particle_depth < 0. || particle_depth > depth)
                continue;
            vec2 ndc_pos = vec2(dot(pos, view_matrix[0]), dot(pos, view_matrix[2]));
            float radius = max(WORLD_RADIUS, particle_depth * depth_scale);
            vec2 delta = abs(ndc_pos - fragCoordNDC * particle_depth);
            if (max(delta.x, delta.y) <= radius)
            {
                fragColor = vec4(mix(vec3(.75,.75,.25), vec3(.25), linear_step(.0, .5, age)), 0.);
                depth = particle_depth;
            }
        }
    }
#endif // RENDER_PARTICLES
}

// Function 175
vec3 getParticleColor_mp( float pint)
{
   float hue;
   float saturation;
   
   saturation = 0.75/pow(pint, 2.5) + mp_saturation;
   hue = hue_time_factor*time2 + mp_hue;

   return hsv2rgb(vec3(hue, saturation, pint));
}

// Function 176
vec3 update(in vec3 vel, vec3 pos, in float id)
{   
    vel.xyz = vel.xyz*.999 + (hash3(vel.xyz + time)*2.)*7.;
    
    float d = pow(length(pos)*1.2, 0.75);
    vel.xyz = mix(vel.xyz, -pos*d, sin(-time*.55)*0.5+0.5);
    
    return vel;
}

// Function 177
vec3 readParticlePos( int idx )
{
    return getPixel(XC(idx),YC(idx)).xyz;
}

// Function 178
float particles(vec3 direction)
{
	float accumulate=0.;
    const mat3 p=mat3(13.3,23.5,21.7,21.1,28.7,11.9,21.8,14.7,61.3);
	vec2 uvx=vec2(direction.x,direction.z)+vec2(1.,iResolution.y/iResolution.x)*gl_FragCoord.xy/iResolution.xy;
	float DEPTH = direction.y*direction.y-.3;
	for (float fi=0.;fi<10.;fi++) 
	{
		vec2 q=uvx*(1.+fi*DEPTH)+vec2(DEPTH,0.2*iTime/(1.+fi*DEPTH*.03));
		vec3 n=vec3(floor(q),31.1+fi);
		vec3 m=floor(n)*.0001 + fract(n);
		vec3 r=fract((31415.+m)/fract(p*m));
		vec2 s=abs(mod(q,1.)-.5+.9*r.xy-.45);
		float d=s.x+s.y+0.7*max(s.y,s.x)-.01;
		float edge=.06;
		accumulate+=smoothstep(edge,-edge,d)*r.x;
	}
	return accumulate;
	}

// Function 179
vec4 UpdateMouseDelta(inout vec4 o,vec2 mouse)
{vec3 m=GetLastMouseClick()
;vec2 t=mouse.xy/iResolution.xy-m.xy
,l =GetLastMouseDelta().zw
;if(m.z<.5)return vec4(o.xy,0,0)
;return vec4(t-l,t);}

// Function 180
void gui_misc_update() {
    
    if (fc.x != MISC_COL) { return; }
        
    if (box_dist(iMouse.xy, link_ui_box()) < 0.) {
        data.x = 1. - data.x;
    }
    
    for (int i=0; i<2; ++i) {
        if (box_dist(iMouse.xy, color_ui_box(i)) < 0.) {
            data.y = float(i);
        }
    }
    
}

// Function 181
vec4 drawParticles(in vec2 p)
{
    vec4 rez = vec4(0);
    vec2 w = 1./iResolution.xy;
    
    for (int i = 0; i < numParticles; i++)
    {
        vec2 pos = texture(iChannel0, vec2(i,50.0)*w).rg;
        vec2 vel = texture(iChannel0, vec2(i,0.0)*w).rg;
        float d = mag(p - pos);
        d *= 500.;
        d = .01/(pow(d,1.0)+.001);

        //rez.rgb += d*abs(sin(vec3(2.,3.4,1.2)*(time*.01 + float(i)*.0017 + 2.5) + vec3(0.8,0.,1.2))*0.7+0.3)*0.04;
        rez.rgb += d*abs(sin(vec3(2.,3.4,1.2)*(time*.07 + float(i)*.0017 + 2.5) + vec3(0.8,0.,1.2))*0.7+0.3)*0.04;
        pos.xy += vel*0.002*0.2;
    }
    
    return rez;
}

// Function 182
void update_tiles(inout vec4 fragColor, vec2 fragCoord)
{
    const vec4 SENTINEL_COLOR = vec4(1, 0, 1, 0);
    
    vec4 resolution = vec4(iResolution.xy, 0, 0);
    vec4 old_resolution = (iFrame==0) ? vec4(0) : load(ADDR_RESOLUTION);
    int flags = int(old_resolution.z);
    if (iFrame == 0 && iTime >= THUMBNAIL_MIN_TIME)
        flags |= RESOLUTION_FLAG_THUMBNAIL;
    vec4 atlas_info = (iFrame==0) ? vec4(0) : load(ADDR_ATLAS_INFO);
    int available_mips = int(round(atlas_info.x));
   
    vec2 available_space = (resolution.xy - ATLAS_OFFSET) / ATLAS_CHAIN_SIZE;
    float atlas_lod = max(0., -floor(log2(min(available_space.x, available_space.y))));
    if (atlas_lod != atlas_info.y)
        available_mips = 0;
    if (max(abs(resolution.x-old_resolution.x), abs(resolution.y-old_resolution.y)) > .5)
        flags |= RESOLUTION_FLAG_CHANGED;
    
    // Workaround for Shadertoy double-buffering bug on resize
    // (this.mBuffers[i].mLastRenderDone = 0; in effect.js/Effect.prototype.ResizeBuffer)
    vec2 sentinel_address = ATLAS_OFFSET + ATLAS_CHAIN_SIZE * exp2(-atlas_lod) - 1.;
    vec4 sentinel = (iFrame == 0) ? vec4(0) : load(sentinel_address);
    if (any(notEqual(sentinel, SENTINEL_COLOR)))
    {
        available_mips = 0;
        flags |= RESOLUTION_FLAG_CHANGED;
    }
    
    resolution.z = float(flags);

    if (available_mips > 0)
    	update_mips(fragColor, fragCoord, atlas_lod, available_mips);
    
    if (ALWAYS_REFRESH > 0 || available_mips == 0)
    {
        if (available_mips == 0)
        	store(fragColor, fragCoord, ADDR_RANGE_ATLAS_CHAIN, vec4(0.));
        generate_tiles(fragColor, fragCoord, atlas_lod);
        available_mips = max(available_mips, 1);
    }
    atlas_info.x = float(available_mips);
    atlas_info.y = atlas_lod;

    store(fragColor, fragCoord, ADDR_RESOLUTION, resolution);
    store(fragColor, fragCoord, ADDR_ATLAS_INFO, atlas_info);
    store(fragColor, fragCoord, sentinel_address, SENTINEL_COLOR);
}

// Function 183
vec4 getParticle(in vec3 iResolution, in sampler2D iChannel0, in int frame, int index) {
    ivec2 uv = deserializeUV(iResolution, frame, index);
    //getRandomParticlePos(iResolution, iChannel0, fragCoord, frame, i)
	return texelFetch(iChannel0, uv, 0);
}

// Function 184
vec3 drawParticle(in vec2 p, in float size, in vec3 col) {
  return mix( col, vec3(0.0)  , smoothstep(0., size, dot(p, p) * 90.0 ) );
}

// Function 185
Particle GrowAlongParticle(int index, int parentIdx)
{
    return GrowParticle(index, parentIdx, 0.13);
}

// Function 186
define updatevel(i,v,t){ balldatas[i].xy+=(balldatas[i].zw-(v))*t;balldatas[i].zw=(v);}

// Function 187
void UpdateCamera()
{
    float xRotationValue = (iMouse.z > 0.0) ? (iMouse.y / iResolution.y - 1.75) * (PI * 1.15) : 0.8;
    mat3 xRotationMatrix = Create3x3RotationMatrix(vec3(1.0, 0.0, 0.0), xRotationValue);
    float yRotationValue = (iMouse.z > 0.0) ? (iMouse.x / iResolution.x) * (PI * 2.0) : (iTime * PI) * 0.05;
    mat3 yRotationMatrix = Create3x3RotationMatrix(vec3(0.0, -1.0, 0.0), yRotationValue);

    // Determine our camera info
    const float distanceFromOrigin = 3.5;
    gCamera.mPosition.xyz = vec3(distanceFromOrigin * sin(yRotationValue) * cos(xRotationValue), distanceFromOrigin * sin(xRotationValue), distanceFromOrigin * cos(yRotationValue) * cos(xRotationValue));
    gCamera.mForward.xyz = yRotationMatrix * xRotationMatrix * vec3(0.0, 0.0, -1.0);
    gCamera.mRight.xyz = yRotationMatrix * xRotationMatrix * vec3(1.0, 0.0, 0.0);
}

// Function 188
particle getParticle(vec4 data, vec2 pos)
{
    particle P; 
    P.X = decode(data.x) + pos;
    P.V = decode(data.y);
    P.M = data.zw;
    return P;
}

// Function 189
float particleDistance(int id, vec2 p)
{
    return repD(getParticle(id).xy, p);
}

// Function 190
vec3 getParticle( vec2 id )
{
	id+=vec2(20.0);
	id=clamp(id,0.0,39.0);
    return texture( iChannel0, (id+0.5)/iResolution.xy ).xyz;
}

// Function 191
void update_menu(inout vec4 fragColor, vec2 fragCoord)
{
#if ENABLE_MENU
    if (is_inside(fragCoord, ADDR_MENU) > 0.)
    {
        MenuState menu;
        if (iFrame == 0)
            clear(menu);
        else
            from_vec4(menu, fragColor);

    	if (is_input_enabled() > 0.)
        {
            if (cmd_menu() > 0.)
            {
                menu.open ^= 1;
            }
            else if (menu.open > 0)
            {
                menu.selected += int(is_key_pressed(KEY_DOWN) > 0.) - int(is_key_pressed(KEY_UP) > 0.) + NUM_OPTIONS;
                menu.selected %= NUM_OPTIONS;
            }
        }
       
        to_vec4(fragColor, menu);
        return;
    }
    
    if (is_inside(fragCoord, ADDR_OPTIONS) > 0.)
    {
        if (iFrame == 0)
        {
            Options options;
            clear(options);
            to_vec4(fragColor, options);
            return;
        }
        
        MenuState menu;
        LOAD(menu);

        int screen_size_field = get_option_field(OPTION_DEF_SCREEN_SIZE);
        float screen_size = fragColor[screen_size_field];
        if (is_key_pressed(KEY_1) > 0.) 	screen_size = 10.;
        if (is_key_pressed(KEY_2) > 0.) 	screen_size = 8.;
        if (is_key_pressed(KEY_3) > 0.) 	screen_size = 6.;
        if (is_key_pressed(KEY_4) > 0.) 	screen_size = 4.;
        if (is_key_pressed(KEY_5) > 0.) 	screen_size = 2.;
        if (is_key_pressed(KEY_MINUS) > 0.)	screen_size -= 2.;
        if (is_key_pressed(KEY_PLUS) > 0.)	screen_size += 2.;
        fragColor[screen_size_field] = clamp(screen_size, 0., 10.);
        
        int flags_field = get_option_field(OPTION_DEF_SHOW_FPS);
        int flags = int(fragColor[flags_field]);

        if (is_key_pressed(TOGGLE_TEX_FILTER_KEY) > 0.)
            flags ^= OPTION_FLAG_TEXTURE_FILTER;
        if (is_key_pressed(TOGGLE_LIGHT_SHAFTS_KEY) > 0.)
            flags ^= OPTION_FLAG_LIGHT_SHAFTS;
        if (is_key_pressed(TOGGLE_CRT_EFFECT_KEY) > 0.)
            flags ^= OPTION_FLAG_CRT_EFFECT;
        
        if (is_key_pressed(SHOW_PERF_STATS_KEY) > 0.)
        {
            const int MASK = OPTION_FLAG_SHOW_FPS | OPTION_FLAG_SHOW_FPS_GRAPH;
            // https://fgiesen.wordpress.com/2011/01/17/texture-tiling-and-swizzling/
            // The line below combines Fabian Giesen's trick (offs_x = (offs_x - x_mask) & x_mask)
            // with another one for efficient bitwise integer select (c = a ^ ((a ^ b) & mask)),
            // which I think I also stole from his blog, but I can't find the link
            flags ^= (flags ^ (flags - MASK)) & MASK;
            
            // don't show FPS graph on its own when using keyboard shortcut to cycle through options
            if (test_flag(flags, OPTION_FLAG_SHOW_FPS_GRAPH))
                flags |= OPTION_FLAG_SHOW_FPS;
        }
        
        fragColor[flags_field] = float(flags);

        if (menu.open <= 0)
            return;
        float adjust = is_key_pressed(KEY_RIGHT) - is_key_pressed(KEY_LEFT);

        MenuOption option = get_option(menu.selected);
        int option_type = get_option_type(option);
        int option_field = get_option_field(option);
        if (option_type == OPTION_TYPE_SLIDER)
        {
            fragColor[option_field] += adjust;
            fragColor[option_field] = clamp(fragColor[option_field], 0., 10.);
        }
        else if (option_type == OPTION_TYPE_TOGGLE && (abs(adjust) > .5 || is_key_pressed(KEY_ENTER) > 0.))
        {
            int value = int(fragColor[option_field]);
            value ^= get_option_range(option);
            fragColor[option_field] = float(value);
        }
        
        return;
    }
#endif // ENABLE_MENU
}

// Function 192
void update_entity_state(vec3 camera_pos, vec3 camera_angles, vec3 direction, float depth, bool is_thumbnail)
{
    g_entities.mask = 0u;
    
    g_entities.flame.loop			= fract(floor(g_animTime * 10.) * .1);
    g_entities.flame.sin_cos		= vec2(sin(g_entities.flame.loop * TAU), cos(g_entities.flame.loop * TAU));
    g_entities.fireball.offset		= get_fireball_offset(g_animTime);
    g_entities.fireball.rotation	= axis_angle(normalize(vec3(1, 8, 4)), g_animTime * 360.);

    float base_fov_y = scale_fov(FOV, 9./16.);
    float fov_y = compute_fov(iResolution.xy).y;
    float fov_y_delta = base_fov_y - fov_y;

    vec3 velocity = load(ADDR_VELOCITY).xyz;
    Transitions transitions;
    LOAD(transitions);
    float offset = get_viewmodel_offset(velocity, transitions.bob_phase, transitions.attack);
    g_entities.viewmodel.offset		= camera_pos;
    g_entities.viewmodel.rotation	= mul(euler_to_quat(camera_angles), axis_angle(vec3(1,0,0), fov_y_delta*.5));
    g_entities.viewmodel.offset		+= rotate(g_entities.viewmodel.rotation, vec3(0,1,0)) * offset;
    g_entities.viewmodel.rotation	= conjugate(g_entities.viewmodel.rotation);
    g_entities.viewmodel.attack		= linear_step(.875, 1., transitions.attack);
    
#if USE_ENTITY_AABB
    #define TEST_AABB(pos, rcp_delta, mins, maxs) ray_vs_aabb(pos, rcp_delta, mins, maxs)
#else
    #define TEST_AABB(pos, rcp_delta, mins, maxs) true
#endif
    
    Options options;
    LOAD(options);
    
    const vec3 VIEWMODEL_MINS = vec3(-1.25,       0, -8);
    const vec3 VIEWMODEL_MAXS = vec3( 1.25,      18, -4);
    vec3 viewmodel_ray_origin = vec3(    0, -offset,  0);
    vec3 viewmodel_ray_delta  = rotate(g_entities.viewmodel.rotation, direction);
    bool draw_viewmodel = is_demo_mode_enabled(is_thumbnail) ? (g_demo_scene & 1) == 0 : true;
    draw_viewmodel = draw_viewmodel && test_flag(options.flags, OPTION_FLAG_SHOW_WEAPON);
    if (draw_viewmodel && TEST_AABB(viewmodel_ray_origin, 1./viewmodel_ray_delta, VIEWMODEL_MINS, VIEWMODEL_MAXS))
        g_entities.mask |= 1u << ENTITY_BIT_VIEWMODEL;
    
    vec3 inv_world_ray_delta = 1./(direction*depth);

    const vec3 TORCH_MINS = vec3(-4, -4, -28);
	const vec3 TORCH_MAXS = vec3( 4,  4,  18);
    for (int i=0; i<NUM_TORCHES; ++i)
        if (TEST_AABB(camera_pos - g_ent_pos.torches[i], inv_world_ray_delta, TORCH_MINS, TORCH_MAXS))
            g_entities.mask |= (1u<<ENTITY_BIT_TORCHES) << i;
    
    const vec3 LARGE_FLAME_MINS = vec3(-10, -10, -18);
	const vec3 LARGE_FLAME_MAXS = vec3( 10,  10,  34);
    for (int i=0; i<NUM_LARGE_FLAMES; ++i)
        if (TEST_AABB(camera_pos - g_ent_pos.large_flames[i], inv_world_ray_delta, LARGE_FLAME_MINS, LARGE_FLAME_MAXS))
            g_entities.mask |= (1u<<ENTITY_BIT_LARGE_FLAMES) << i;
        
	const vec3 FIREBALL_MINS = vec3(-10);
	const vec3 FIREBALL_MAXS = vec3( 10);
    if (g_entities.fireball.offset.z > 8. &&
        TEST_AABB(camera_pos - FIREBALL_ORIGIN - g_entities.fireball.offset, inv_world_ray_delta, FIREBALL_MINS, FIREBALL_MAXS))
        g_entities.mask |= 1u << ENTITY_BIT_FIREBALL;

    GameState game_state;
    LOAD(game_state);
    g_entities.target.scale = 0.;
    g_entities.target.indices = 0u;
    if (abs(game_state.level) >= 1.)
    {
        vec2 scale_bias = game_state.level > 0. ? vec2(1, 0) : vec2(-1, 1);
        float fraction = linear_step(BALLOON_SCALEIN_TIME * .1, 0., fract(abs(game_state.level)));
        g_entities.target.scale = fraction * scale_bias.x + scale_bias.y;
        if (g_entities.target.scale > 1e-2)
        {
            float level = floor(abs(game_state.level));
            int set = int(fract(level * PHI + .15) * float(NUM_BALLOON_SETS));
            uint indices = g_ent_pos.balloon_sets[set];
            g_entities.target.scale = overshoot(g_entities.target.scale, .5);
        	g_entities.target.indices = indices;
            
            vec3 BALLOON_MINS = vec3(-28, -28, -20) * g_entities.target.scale;
            vec3 BALLOON_MAXS = vec3( 28,  28,  64) * g_entities.target.scale;
            for (int i=0; i<NUM_TARGETS; ++i, indices>>=4)
            {
                Target target;
                LOADR(vec2(i, 0.), target);
                if (target.hits < ADDR_RANGE_SHOTGUN_PELLETS.z * .5)
                    if (TEST_AABB(camera_pos - g_ent_pos.balloons[indices & 15u], inv_world_ray_delta, BALLOON_MINS, BALLOON_MAXS))
	                    g_entities.mask |= (1u << i);
            }
        }
    }
}

// Function 193
vec4 UpdateRotation(inout vec4 fragColor)
{if(GetLastMouseClick().z<.5)return fragColor
;vec4 lastQuat  =GetLastRotation()
;
 vec2 mouseDelta=GetLastMouseDelta().xy*CameraSensitivity
     ;
#ifdef INVERT_Y
;mouseDelta=-mouseDelta
#endif
;vec3 forw=GetLastForward()
;vec3 righ=normalize(cross(forw,vec3(0,1,0)))
;vec3 up  =normalize(cross(righ,forw))
;lastQuat=QxQ(Quat(up,-mouseDelta.x),lastQuat)
;return QxQ(Quat(righ,mouseDelta.y),lastQuat);}

// Function 194
mat3 particleSpaceMatrix(vec2 origin, vec2 velocity) {
    vec3 O = vec3(origin, 1.);
    vec3 X = normalize(vec3(velocity, 0.));
    vec3 Y = cross(vec3(0., 0., 1.), X);
    return mat3(X, Y, O);
}

// Function 195
void UpdateSpawner( float screenWidth )
{
    SpawnSniper( 15.0, 0.0, screenWidth );
    SpawnPowerUp( 23.0, screenWidth );    
    SpawnSniper( 25.0, 0.0, screenWidth );
    SpawnTurret( 47.0, 2.0, screenWidth );
    SpawnSniper( 48.0, 3.0, screenWidth );
    SpawnSniper( 56.0, 4.0, screenWidth );
    SpawnPowerUp( 55.0, screenWidth ); 
    SpawnTurret( 59.0, 3.0, screenWidth );
    SpawnTurret( 65.0, 3.0, screenWidth );
    SpawnTurret( 72.0, 2.0, screenWidth );
    SpawnTurret( 76.0, 5.0, screenWidth );
    SpawnSniper( 82.0, 2.0, screenWidth );
    SpawnPowerUp( 89.0, screenWidth );    
    SpawnTurret( 94.0, 3.0, screenWidth );
    SpawnTurret( 101.0, 1.0, screenWidth );
    SpawnTurret( 105.0, 1.0, screenWidth );
    SpawnSniper( 109.8, 4.875, screenWidth );

    if ( gCamera.z == 0.0 && Rand() > 0.5 )
    {
        gCamera.z = SOLDIER_SPAWN_RATE - 20.0;
    }

    ++gCamera.z;
    vec4 newSoldier 		= vec4( gCamera.x + screenWidth, 300.0, -1.0, 0.0 );
    vec4 newSoldierState 	= vec4( 0.0, 0.0, 0.0, 0.0 );
	newSoldier.y = GetSupport( newSoldier.xy );    
    if ( gCamera.x < SOLDIER_SPAWN_END && gCamera.z > SOLDIER_SPAWN_RATE && newSoldier.y > WATER_HEIGHT )
    {
        gCamera.z = 0.0;
        
        if ( gSoldier0.x <= 0.0 )
        {
            gSoldier0 		= newSoldier;
            gSoldier0State 	= newSoldierState;
        }
		else if ( gSoldier1.x <= 0.0 )
        {
            gSoldier1 		= newSoldier;
            gSoldier1State 	= newSoldierState;
        }
		else if ( gSoldier2.x <= 0.0 )
        {
            gSoldier2 		= newSoldier;
            gSoldier2State 	= newSoldierState;
        }        
    }
}

// Function 196
vec3 particleColor(vec2 uv, float radius, float offset, float periodOffset)
{
    vec3 color = palette(.4 + offset / 4., p0);
    uv /= pow(periodOffset, .75) * sin(periodOffset * iTime) + sin(periodOffset + iTime);
    vec2 pos = vec2(cos(offset * offsetMult + time + periodOffset),
        		sin(offset * offsetMult + time * 5. + periodOffset * tau));
    
    float dist = radius / distance(uv, pos);
    return color * pow(dist, 2.) * 1.75;
}

// Function 197
float particles(vec2 p)
{
	p *= 100.0; //200.;
	
	float f = 0.;
	float amp = 1.0, s = 1.5;
	
	for (int i = 0; i < 3; i++)
	{
		p = m * p * 1.2; 
		f += amp*seaNoise(p + time * s );
		amp = amp*.5; 
		s*= -1.227; 
	}
	
	return pow(f*.35, 7.)*particle_amount;
}

// Function 198
void UpdateRotation(vec2 uv, out vec4 result)
{
    if (iFrame == 0)
    {
        result = QUAT_IDENTITY;
    }
    else
    {
        vec4 cs = texture(iChannel1, uv);
        //--rotate
        float dr = iTimeDelta * rotateSpeed;
        vec2 eu = vec2(0);
        //mouse
        vec4 lm = GetLastMouse();
        vec2 dn = iMouse.xy - iMouse.zw;
        vec2 dp = iMouse.xy - lm.xy;
        vec2 dm = iMouse.z > lm.z ? dn : dp;
        dm = vec2(dm.x, -dm.y) / iResolution.y;
        eu.xy += dm.yx * (dr * 100.0);
        //key
        if (readKey(keyW)) eu.x -= dr;
        if (readKey(keyS)) eu.x += dr;
        if (readKey(keyA)) eu.y -= dr;
        if (readKey(keyD)) eu.y += dr;
        
        result = QuatMul(cs, QuatEA(eu));
    }
}

// Function 199
ivec2 getRandomParticlePos(in vec3 iResolution, in sampler2D iChannel0, in vec2 fragCoord, in int frame, int i) {
    //uvec2 b = uvec2(i / iter - iter / 2, i % iter - iter / 2);
    uvec2 b = uvec2(0);
    uvec2 p1 = uvec2(fragCoord) + b + uvec2(frame, 13 * frame);
    uvec2 p2 = uvec2(fragCoord.yx) + b + uvec2(29 * frame, frame);
    float f1 = hash(p1);
    float f2 = hash(p2);
    //int xp = min(xParticles, int(iResolution.y));
    //int yp = min(yParticles, int(iResolution.y)); // + 10 * (frame % 10 + 1);
    ivec2 p3 = ivec2(f1 * float(xParticles), f2 * float(yParticles));
    //p3 = ivec2(fragCoord);
    //i = (i + frame) % (iter * iter);
    p3 += ivec2(i / iter - iter / 2, i % iter - iter / 2);
    //p3 += ivec2(i / iter, i % iter);
    p3.x = abs(p3.x % xParticles);
    p3.y = abs(p3.y % yParticles);
    return p3;
}

// Function 200
vec3 drawParticles(vec2 uv, float timedelta)
{  
    // Here the time is "stetched" with the time factor, so that you can make a slow motion effect for example
    time2 = time_factor*(iTime + timedelta);
    vec3 pcol = vec3(0.);
    // Main particles loop
    for (int i=1; i<nb_particles; i++)
    {
        pst = getParticleStartTime(i); // Particle start time
        plt = mix(part_life_time_min, part_life_time_max, random(float(i*2-35))); // Particle life time
        time4 = mod(time2 - pst, plt);
        time3 = time4 + pst;
       // if (time2>pst) // Doesn't draw the paricle at the start
        //{    
           runnr = floor((time2 - pst)/plt);  // Number of the "life" of a particle
           vec2 ppos = getParticlePosition(i);
           float dist = distance(uv, ppos);
           //if (dist<0.05) // When the current point is further than a certain distance, its impact is neglectable
           //{
              // Draws the eight-branched star
              // Horizontal and vertical branches
              vec2 uvppos = uv - ppos;
              float distv = distance(uvppos*part_starhv_dfac + ppos, ppos);
              float disth = distance(uvppos*part_starhv_dfac.yx + ppos, ppos);
              // Diagonal branches
              vec2 uvpposd = 0.707*vec2(dot(uvppos, vec2(1., 1.)), dot(uvppos, vec2(1., -1.)));
              float distd1 = distance(uvpposd*part_stardiag_dfac + ppos, ppos);
              float distd2 = distance(uvpposd*part_stardiag_dfac.yx + ppos, ppos);
              // Initial intensity (random)
              float pint0 = mix(part_int_factor_min, part_int_factor_max, random(runnr*4. + float(i-55)));
              // Middle point intensity star inensity
              float pint1 = 1./(dist*dist_factor + 0.015) + part_starhv_ifac/(disth*dist_factor + 0.01) + part_starhv_ifac/(distv*dist_factor + 0.01) + part_stardiag_ifac/(distd1*dist_factor + 0.01) + part_stardiag_ifac/(distd2*dist_factor + 0.01);
              // One neglects the intentity smaller than a certain threshold
              //if (pint0*pint1>16.)
              //{
                 // Intensity curve and fading over time
                 float pint = pint0*(pow(pint1, ppow)/part_int_div)*(-time4/plt + 1.);
                
                 // Initial growing of the paricle's intensity
                 pint*= smoothstep(0., grow_time_factor*plt, time4);
                 // "Sparkling" of the particles
                 float sparkfreq = clamp(part_spark_time_freq_fact*time4, 0., 1.)*part_spark_min_freq + random(float(i*5 + 72) - runnr*1.8)*(part_spark_max_freq - part_spark_min_freq);
                 pint*= mix(part_spark_min_int, part_spark_max_int, random(float(i*7 - 621) - runnr*12.))*sin(sparkfreq*twopi*time2)/2. + 1.;

                 // Adds the current intensity to the global intensity
                 pcol+= getParticleColor(i, pint);
              //}
           //}
        //}
    }
    // Main particle
    vec2 ppos = getParticlePosition_mp();
    float dist = distance(uv, ppos);
    //if (dist<0.25)
    //{
        // Draws the eight-branched star
        // Horizontal and vertical branches
        vec2 uvppos = uv - ppos;
        float distv = distance(uvppos*part_starhv_dfac + ppos, ppos);
        float disth = distance(uvppos*part_starhv_dfac.yx + ppos, ppos);
        // Diagonal branches
        vec2 uvpposd = 0.7071*vec2(dot(uvppos, vec2(1., 1.)), dot(uvppos, vec2(1., -1.)));
        float distd1 = distance(uvpposd*part_stardiag_dfac + ppos, ppos);
        float distd2 = distance(uvpposd*part_stardiag_dfac.yx + ppos, ppos);
        // Middle point intensity star inensity
        float pint1 = 1./(dist*dist_factor + 0.015) + part_starhv_ifac/(disth*dist_factor + 0.01) + part_starhv_ifac/(distv*dist_factor + 0.01) + part_stardiag_ifac/(distd1*dist_factor + 0.01) + part_stardiag_ifac/(distd2*dist_factor + 0.01);
        
        if (part_int_factor_max*pint1>6.)
        {
            float pint = part_int_factor_max*(pow(pint1, ppow)/part_int_div)*mp_int;
            pcol+= getParticleColor_mp(pint);
        }
    //}
    return pcol;
}

// Function 201
vec4 drawParticles(in vec3 ro, in vec3 rd)
{
    vec4 rez = vec4(0);
    vec2 w = 1./iResolution.xy;
    
    for (int i = 0; i < numParticles; i++)
    {
        vec3 pos = texture(iChannel0, vec2(i,100.0)*w).rgb;
        vec3 vel = texture(iChannel0, vec2(i,0.0)*w).rgb;
        for(int j = 0; j < stepsPerFrame; j++)
        {
            float d = mag((ro + rd*dot(pos.xyz - ro, rd)) - pos.xyz);
            d *= 1000.;
            d = .14/(pow(d,1.1)+.03);
            
            rez.rgb += d*abs(sin(vec3(2.,3.4,1.2)*(time*.06 + float(i)*.003 + 2.) + vec3(0.8,0.,1.2))*0.7+0.3)*0.04;
            //rez.rgb += d*abs(sin(vec3(2.,3.4,1.2)*(time*.06 + float(i)*.003 + 2.75) + vec3(0.8,0.,1.2))*0.7+0.3)*0.04;
            pos.xyz += vel*0.002*0.2;
        }
    }
    rez /= float(stepsPerFrame);
    
    return rez;
}

// Function 202
float particleDistance(int i)
{
    return distance(p, sc(texel(ch0, i2xy(ivec3(i, 0, 0))).xy));
}

// Function 203
vec4 DrawParticle(float id, vec2 p, float size, float t, vec3 color, float powFactor, float duration, float prevDuration, float distort)
{
    p.y *= 1. + sin(id*99.5+ dot(gUV,gUV)*40.+ gT*4.)*0.18 * distort;
    p.x *= 1. + cos(id*189.1+ dot(gUV,gUV)*76.+ gT*3.5)*0.13 * distort;
    
    float dist = dot(p, p)/size;
    float distNorm = smoothstep( 0., 1., pow(dist, powFactor) );
    
    duration = max(duration * (1.- smoothstep(0., 1., pow(dist, 0.1)) ), prevDuration);
    
    vec4 res = mix( vec4(color.rgb, duration), vec4(0., 0., 0., prevDuration), distNorm );
    return mix(mix(vec4(0., 0., 0., max(res.a, prevDuration)), res, smoothstep(0.0, 0.15, t)), vec4(0., 0., 0., max(res.a, prevDuration)), smoothstep(0.5, 1., t));
}

// Function 204
void UpdatePosition(inout vec4 pos)
{vec3 lastPos   =GetLastPosition()
;vec3 dirForward=GetLastForward()
;vec3 dirRight  =ross(dirForward,vec3(0,1,0))
;float pressForward=max(kp(KEY_W),kp(KEY_UP))  -max(kp(KEY_S),kp(KEY_DOWN))
;float pressRight  =max(kp(KEY_A),kp(KEY_LEFT))-max(kp(KEY_D),kp(KEY_RIGHT))
;vec3 direction=dirForward*pressForward-dirRight*pressRight
;float delta=walk*iTimeDelta*step(.001,length(direction))
;if(delta>.0)
 {delta*=max(1.,kp(KEY_SHIFT)*run)
 ;direction=normalize(direction)
 ;pos.xyz=lastPos+(direction*delta);        
 #ifndef FREE_FLY
 pos.y=StartPos.y;
 #endif
 }}

// Function 205
void DrawAParticleSet(inout vec4 color, vec2 uv, float size ){
   float aCellLenght = size;
   vec3 colorTint;
   float randomSeed01 = rand(floor (uv /aCellLenght));
   float randomSeed02 = rand(floor (uv /aCellLenght) + 5.0);
   float randomSeed03 = rand(floor (uv /aCellLenght) + 10.0);
    
  
    colorTint= vec3(randomSeed01, randomSeed02, randomSeed03);
    
   float circleLenght =abs(sin(iTime * randomSeed03 + randomSeed02))  * randomSeed01 * aCellLenght;
   
   float jitterFreedom = 0.5 - circleLenght;
   float jitterAmountX =  jitterFreedom * (randomSeed03 *2.0 -1.0);
   float jitterAmounty =  jitterFreedom * (randomSeed01 *2.0 -1.0); 
   vec2 coord =  fract(uv / aCellLenght);
    
    
   coord -= 0.5;
   float z = 0.0;
   vec3 toReturn; 
   for(int i=0; i < 3; i++) {
       z += 0.015 * celluar2x2(coord + iTime * 0.1).x  /*abs(sin(iTime * randomSeed01 + randomSeed01))*/;
		coord += z;
		toReturn[i] = 1.0 - smoothstep(circleLenght- 30.5/iResolution.y,
                                       circleLenght, distance(coord, vec2(jitterAmountX, jitterAmounty)));
	}
    
   toReturn = mix(color.xyz, colorTint *toReturn, length(toReturn));
   color = vec4(toReturn.xyz, 0.1);
}

// Function 206
void UpdateMouseDelta(inout vec4 fragColor, in vec2 mouse)
{
    vec3 lastMouse  = GetLastMouseClick();
    vec2 totalDelta = (mouse.xy / iResolution.xy) - lastMouse.xy;
    vec2 lastDelta  = GetLastMouseDelta().zw;
       
    if(lastMouse.z < 0.5)
    {
        fragColor.zw = vec2(0.0);
        return;
    }
    
    fragColor.xy = totalDelta - lastDelta;
    fragColor.zw = totalDelta;
}

// Function 207
void updateOrbit(inout CelestialBody sun, inout CelestialBody earth, vec4 earthData, int iFrame)
{
    sun.pos = sunPosition;
    sun.mass = sunMass;
    if (iFrame < 20)
    {
        earth.pos = earthPosition;
    	earth.vel = earthVelocity;
    }
    else
    {
        earth.pos = earthData.xy;
    	earth.vel = earthData.zw;
    }
    
    // Semi-implicit Euler's method
    for (int i = 0; i < 100; ++i)
    {
        earth.vel += calcAccl(earth.vel, earth.pos, sun.pos, sun.mass, h) * h;
    	earth.pos += calcVel(earth.vel, earth.pos, sun.pos, sun.mass, 0.) * h;  
    }
    
    
}

// Function 208
void update_snap(inout float dmin,
                 inout int imin,
                 in int i,
                 in vec3 q,
                 in vec3 p) {
    
    float d = length(p-q);
    
    if (d < dmin) {
        dmin = d;
        imin = i;
    }
    
}

// Function 209
void updateValue2(int index) {
    int type = controls[index].type;
    if (type != CLICKBOX_T) {
    	controls[index].value2 = 1. - controls[index].value;
    }
}

// Function 210
define UPDATE_INTERSECTIONS(p0, s1, s2, r0, rd) { \
    LQI = line_quad_intersect(p0, s1, s2, r0, rd); \
    if (LQI.a == 1.0) { \
        if (intersections == 0) p1 = LQI.xyz; \
        else p2 = LQI.xyz; \
        intersections++; \
    } \
}

// Function 211
void Update()
{
    for(int obj = 0; obj < NUMBER_OF_OBJECTS; obj++)
    {
        Notes[obj] = texelFetch(iChannel0,ivec2(obj,0),0);
        Lines[obj] = texelFetch(iChannel0,ivec2(obj,1),0);
    }
}

// Function 212
vec3 salmpleLightForParticle(in vec3 x, in float time) {
    vec3 Lo = vec3(0.0);	//outgoing radiance

    LightSamplingRecord rec;
    vec3 Li = sampleLightSource( x, vec3(0.0), rnd(), rnd(), rec );

    float dotNWo = rec.w.y;
    if (dotNWo > 0.02 && rec.pdf > EPSILON) {
        float fr = VOLUME_SCAATTERING / (4.0 * PI);
        float phasePdf = 1.0 / (2.0 * PI);
        
        bool v = true;
#ifdef VOLUMETRIC_SHADOWS
        Ray shadowRay = Ray(x, rec.w, time);
        v = isLightVisible( shadowRay );
#endif
        Lo += ((Li * fr * float(v)) / rec.pdf) * misWeight(rec.pdf, phasePdf);
    }

    return Lo;
}

// Function 213
void updateCueAndCamera() {
	vec3 avg_pos = vec3(0.);
    for(int i=0; i<numBalls; i++) {
    	avg_pos += balls[i].mtx[3].xyz;
    }
    avg_pos /= float(numBalls);
    
    
    if(length(balls[0].v) < 0.03) {
        vec3 cue_pos = cue.mtx[3].xyz;
        vec3 cue_dir = cue.mtx[2].xyz;
        vec3 white_ball_pos = balls[0].mtx[3].xyz;
        
        vec3 desired_cue_dir = normalize(normalize(avg_pos-white_ball_pos)/* + vec3(0.0, -0.1, 0.0)*/);

        //mx from -1 to 1
        float mx = (iMouse.x==0.0) ? 0.0 : ((iMouse.x / iResolution.x)*2.0 - 1.0);
       	desired_cue_dir = rotAroundY(mx*2.0) * desired_cue_dir;
        vec3 desired_cue_pos = white_ball_pos - cue_dir * (cue.h + 3.0 + sin(iTime*5.0)*3.0);
        vec3 target = mix(avg_pos, white_ball_pos + cue_dir * (cue.h + 20.0), 0.5);
        
		float rot_angle = acos(dot(cue_dir, desired_cue_dir));
        if(rot_angle > 0.005) {
            vec3 rot_vec = normalize(cross(cue_dir, desired_cue_dir));
            vec3 rot_pos = cue_pos + cue_dir * cue.h;
            mat4 rot = VToRMat(rot_pos, -rot_vec, 0.05*rot_angle);
            cue.mtx = rot * cue.mtx;
        }
        
        cue.mtx[3].xyz = mix(desired_cue_pos, cue.mtx[3].xyz, 0.96);
        //just to make it orthogonal again
        cue.mtx = createCS(	cue.mtx[3].xyz,
                            cue.mtx[2].xyz,
                            cue.mtx[0].xyz);
        
        float my = iMouse.y / iResolution.y;
        vec3 left = normalize(cross(cue_dir, vec3(0.0, -1.0, 0.0)));
        vec3 desired_cam_pos = white_ball_pos - cue_dir * cue.h * (0.2 + my) + vec3(0., 3.0 + 10.0 * my, 0.) + left * 5.0 * my;
        vec3 desired_cam_dir = -normalize(target - desired_cam_pos);
        mat4 desired_mtx = createCS(desired_cam_pos,
                                    desired_cam_dir,
                                    normalize(cross(vec3(0.0, -1.0, 0.0), desired_cam_dir)));

        camera.mtx_prev = camera.mtx;
        camera.mtx = mtxLerp2(camera.mtx, desired_mtx, 0.2);
    }
}

// Function 214
vec3 drawParticles(vec2 uv, float timedelta)
{   
    // Here the time is "stetched" with the time factor, so that you can make a slow motion effect for example
    time2 = time_factor*(iTime + timedelta);
    vec3 pcol = vec3(0.);
    // Main particles loop
    for (int i=1; i<nb_particles; i++)
    {
        pst = getParticleStartTime(i); // Particle start time
        plt = mix(part_life_time_min, part_life_time_max, random(float(i*2-35))); // Particle life time
        time4 = mod(time2 - pst, plt);
        time3 = time4 + pst;
       // if (time2>pst) // Doesn't draw the paricle at the start
        //{    
           runnr = floor((time2 - pst)/plt);  // Number of the "life" of a particle
           vec2 ppos = getParticlePosition(i);
           float dist = distance(uv, ppos);
           //if (dist<0.05) // When the current point is further than a certain distance, its impact is neglectable
           //{
              // Draws the eight-branched star
              // Horizontal and vertical branches
              vec2 uvppos = uv - ppos;
              float distv = distance(uvppos*part_starhv_dfac + ppos, ppos);
              float disth = distance(uvppos*part_starhv_dfac.yx + ppos, ppos);
              // Diagonal branches
              vec2 uvpposd = 0.707*vec2(dot(uvppos, vec2(1., 1.)), dot(uvppos, vec2(1., -1.)));
              float distd1 = distance(uvpposd*part_stardiag_dfac + ppos, ppos);
              float distd2 = distance(uvpposd*part_stardiag_dfac.yx + ppos, ppos);
              // Initial intensity (random)
              float pint0 = mix(part_int_factor_min, part_int_factor_max, random(runnr*4. + float(i-55)));
              // Middle point intensity star inensity
              float pint1 = 1./(dist*dist_factor + 0.015) + part_starhv_ifac/(disth*dist_factor + 0.01) + part_starhv_ifac/(distv*dist_factor + 0.01) + part_stardiag_ifac/(distd1*dist_factor + 0.01) + part_stardiag_ifac/(distd2*dist_factor + 0.01);
              // One neglects the intentity smaller than a certain threshold
              //if (pint0*pint1>16.)
              //{
                 // Intensity curve and fading over time
                 float pint = pint0*(pow(pint1, ppow)/part_int_div)*(-time4/plt + 1.);
                
                 // Initial growing of the paricle's intensity
                 pint*= smoothstep(0., grow_time_factor*plt, time4);
                 // "Sparkling" of the particles
                 float sparkfreq = clamp(part_spark_time_freq_fact*time4, 0., 1.)*part_spark_min_freq + random(float(i*5 + 72) - runnr*1.8)*(part_spark_max_freq - part_spark_min_freq);
                 pint*= mix(part_spark_min_int, part_spark_max_int, random(float(i*7 - 621) - runnr*12.))*sin(sparkfreq*twopi*time2)/2. + 1.;

                 // Adds the current intensity to the global intensity
                 pcol+= getParticleColor(i, pint);
              //}
           //}
        //}
    }
    
    // Main particle
    vec2 ppos = getParticlePosition_mp();
    float dist = distance(uv, ppos);

        // Draws the eight-branched star
        // Horizontal and vertical branches
        vec2 uvppos = uv - ppos;
        float distv = distance(uvppos*part_starhv_dfac + ppos, ppos);
        float disth = distance(uvppos*part_starhv_dfac.yx + ppos, ppos);
        // Diagonal branches
        vec2 uvpposd = 0.7071*vec2(dot(uvppos, vec2(1., 1.)), dot(uvppos, vec2(1., -1.)));
        float distd1 = distance(uvpposd*part_stardiag_dfac + ppos, ppos);
        float distd2 = distance(uvpposd*part_stardiag_dfac.yx + ppos, ppos);
        // Middle point intensity star inensity
        float pint1 = 1./(dist*dist_factor + 0.015) + part_starhv_ifac/(disth*dist_factor + 0.01) + part_starhv_ifac/(distv*dist_factor + 0.01) + part_stardiag_ifac/(distd1*dist_factor + 0.01) + part_stardiag_ifac/(distd2*dist_factor + 0.01);
        
        if (part_int_factor_max*pint1>6.)
        {
            float pint = part_int_factor_max*(pow(pint1, ppow)/part_int_div)*mp_int;
            pcol+= getParticleColor_mp(pint);
        }

    return pcol;
}

// Function 215
vec4 computeParticles(in vec2 fragCoord )
{
    vec4 fragColor = vec4(0.0);
    vec2 res = maxRes;
    mPartitionData pd = getPartitionData(particleBuffer, fragCoord, res);
    
    if (iFrame == 0 || resetPressed) {
        fragColor = vec4(0.0);
        
        vec2 particle = vec2(0.0);
        if (pd.partitionIndex == 0) {
            // position
            vec2 fc = vec2(fromLinear(pd.index, res));
            vec4 data = hash42(fc);
            particle = transformPos(data.xy);
        } else {
            // velocity
            vec2 fc = vec2(fromLinear(pd.futureIndex, res));
            vec4 data = hash42(fc);

            vec2 pos = transformPos(data.xy);
            vec2 vel = 10.0 * (data.zw - 0.5) / res;
            float maxSpeed = 1.0;
            vel = length(vel) > maxSpeed ? maxSpeed * vel / length(vel) : vel;
            vel = vec2(0.0);
            vec2 oldPos = pos - vel;
            particle = oldPos;
        }

        if (pd.overflow) {
            particle = vec2(0.0);            
        }
        
        fragColor.yz = particle;
        
        return fragColor;
    }
    
    vec4 particle1 = vec4(0.0);
    particle1.xy = getPosition(particleBuffer, pd.index, res);
    particle1.zw = getPosition(particleBuffer, pd.pastIndex, res);
    
    const int k = 16;
    const int k2 = 4;
    int w = int(sqrt(float(k)));
    vec2 a1 = vec2(0.0);
    vec2 a2 = vec2(0.0);
    int torusCount = int(pow(2.0, float(int(iTime / 4.0) % 10)));
    int particlesPerTorus = pd.particlesPerPartition / torusCount;
    int wp = int(sqrt(float(particlesPerTorus)));
    int torus = pd.index / particlesPerTorus;
    for (int i = 0; i < k; i++) {
        {
            int index = pd.index % particlesPerTorus;
            vec2 fc = vec2(fromLinear(index, vec2(wp)));
            vec2 offset = vec2(i % w - w / 2, i / w - w / 2);
            if (torus % 3 == 0 && !justSentinels) {
                // Torus
                fc = fc + offset;
            	fc = mod(fc, vec2(wp));
            } else if (torus % 3 == 1 && !justSentinels) {
                // Cloth
                fc = fc + offset;
            	fc = clamp(fc, vec2(0.0), vec2(wp));
            } else {
                // Sentinel
                offset.x = -1.0;
                offset.y = 0.0;
                fc = fc + offset;
                fc = clamp(fc, vec2(0.0), vec2(wp));
                if (index % wp == 0) {
                    fc = vec2(0.0);
                }
            }
            int j = toLinear(fc, vec2(wp)) + pd.index - index;
            vec2 p2 = getPosition(particleBuffer, j, res);
            a1 += getSpring(res, particle1, p2.xy) / float(w);
        }
        for (int i2 = 0; i2 < k2; i2++) {
            int w = int(sqrt(float(k)));
            int index = pd.index % particlesPerTorus;
            int j =
                int(float(particlesPerTorus) * 
                    hash(uvec2(fragCoord + float(i * k + i2) * vec2(13.0, 29.0) * vec2(iFrame))));
            j += pd.index - index;
            vec2 p2 = getPosition(particleBuffer, j, res);
            a1 += getGravity(res, particle1, p2.xy) / float(w * k2);
        }
    }
    
    vec2 updatedParticle = updateParticle(particle1, a1).xy;
	
    fragColor.yz = pd.partitionIndex == 0 ? updatedParticle.xy : extractPosition(pd.futureParticle);
    fragColor.yz = pd.overflow ? vec2(0.0) : fragColor.yz;
    
    return fragColor;
}

// Function 216
particle getParticle(vec4 data, vec2 pos)
{
    particle P; 
    P.X = decode(data.x) + pos;
    P.V = decode(data.y);
    P.M = data.z;
    P.I = data.w;
    return P;
}

// Function 217
void writeParticle(out vec4 fragColor, in int addr) {
    // Frag coord in particle/field space
    int field = addr % sizeofParticle;
    int particleId = addr / sizeofParticle;
    
    // Load particle info
    Particle particle = getParticle(particleId);
        
    // Update all fields (It's required to update all fields even though we
    // only write some of them in a given fragment because for instance
    // wavelength needs to be aware of the changes of velocity/position.)
    updateParticle(particle, particleId);
    
    switch (field) {
    case colorField:
        fragColor = particle.color;
        break;
    case posVelField:
        fragColor = vec4(particle.position, particle.velocity);
        break;
    case wavelengthField:
        fragColor = vec4(particle.wavelength, particle.phi, particle.lastHit, 0.0);
        break;
    }
}

// Function 218
void updateScene() {
    vec3 pos1 	= vec3( 2.0, 2.5 + sin(frameSta*0.15)*1.74, -4.0 + sin(frameSta*0.3)*2.734 );
    vec3 pos2 	= vec3( 2.0, 2.5 + sin(frameEnd*0.15)*1.74, -4.0 + sin(frameEnd*0.3)*2.734 );
    spherelight[0].pos = mix( pos1, pos2, rnd() );
    
    float y1	= 1.0 + sin(frameSta*0.7123);
    float y2 	= 1.0 + sin(frameEnd*0.7123);
    sphereGeometry.pos.y = mix( y1, y2, rnd() );
}

// Function 219
vec4 getParticle(vec2 p)
{
    vec4 mass=vec4(0.,0.,0.,0.);
    float s=.8;//modify this value. .5 is default. 1. is big; above 1 is interesting ex. 2, 3
    float nh=.0;//modify this as well. 0. is default.
    float t=0.;
    for(float i=-R;i<=R;i++){
    	for(float j=-R;j<=R;j++){
            vec4 prt = texture(iChannel0,(p+vec2(j,i))/iResolution.xy);
            if(prt!=vec4(-1.)){
                if( abs(prt.x+prt.z-j)<=s && abs(prt.y+prt.w-i)<=s ){
                    mass+=vec4(prt.x+prt.z-j,prt.y+prt.w-i,prt.z,prt.w);
                    t++;
            	}
            }
            
    	}
    }
    if(t>nh){
    	return mass/t;
    }
    return vec4(-1.);
}

// Function 220
void update_mips(inout vec4 fragColor, vec2 fragCoord, float atlas_lod, inout int available_mips)
{
    int mip_start = ALWAYS_REFRESH > 0 ? 1 : available_mips;
    available_mips = min(available_mips + 1, MAX_MIP_LEVEL + 1 - int(atlas_lod));
    
    float atlas_scale = exp2(-atlas_lod);

    if (is_inside(fragCoord, atlas_chain_bounds(atlas_scale)) < 0.)
        return;
    if (is_inside(fragCoord, atlas_mip0_bounds(atlas_scale)) > 0.)
        return;

    int mip_end = available_mips;
    int mip;
    vec2 atlas_size = ATLAS_SIZE * atlas_scale;
    vec2 ofs;
    for (mip=mip_start; mip<mip_end; ++mip)
    {
        float fraction = exp2(-float(mip));
        ofs = mip_offset(mip) * atlas_size + ATLAS_OFFSET;
        vec2 size = atlas_size * fraction;
        if (is_inside(fragCoord, vec4(ofs, size)) > 0.)
            break;
    }
    
    if (mip == mip_end)
        return;
    
    vec2 src_ofs = mip_offset(mip-1) * atlas_size + ATLAS_OFFSET;
    vec2 uv = fragCoord - ofs - .5;

    // A well-placed bilinear sample would be almost equivalent,
    // except the filtering would be done in sRGB space instead
    // of linear space. Of course, the textures could be created
    // in linear space to begin with, since we're rendering to
    // floating-point buffers anyway... but then we'd be a bit too
    // gamma-correct for 1996 :)

    ivec4 iuv = ivec2(uv * 2. + src_ofs).xyxy + ivec2(0, 1).xxyy;
    vec4 t00 = gamma_to_linear(texelFetch(iChannel1, iuv.xy, 0));
    vec4 t01 = gamma_to_linear(texelFetch(iChannel1, iuv.xw, 0));
    vec4 t10 = gamma_to_linear(texelFetch(iChannel1, iuv.zy, 0));
    vec4 t11 = gamma_to_linear(texelFetch(iChannel1, iuv.zw, 0));

    fragColor = linear_to_gamma((t00 + t01 + t10 + t11) * .25);
}

// Function 221
void gui_theta_update() {
    
    if (fc.x != THETA_COL) { return; }
    
    if (iMouse.z > 2.*inset_ctr.x && iMouse.w > 0.) {
        
        // mouse down somewhere in the pane but not in GUI panel    
        
    	if ( length(iMouse.zw - object_ctr) < 0.45 * iResolution.y) {

            // down somewhere near object
            vec2 disp = (iMouse.xy - object_ctr) * 0.01;
            data.xyz = vec3(-disp.y, disp.x, 1);
            
        } else {
            
            // down far from object
            data.z = 0.;
            
        }
        
    }
    
        
    if (data.z == 0.) {
        float t = iTime;
        data.x = t * 2.*PI/6.; 
        data.y = t * 2.*PI/18.;
    }    
    
}

// Function 222
vec4 UpdateParam0( vec4 resetValue )
{
    // x = zoom, y = focus, z = aperture, w = exposure
    vec4 param0 = texelFetch(iChannel0,ivec2(2,0),0);//GetParameters();
    
    param0.x += 1.*iTimeDelta*(Key(kNumPlus)-Key(kNumMinus) + Key(kPlusEq)-Key(kMinus));
    param0.y += 1.*iTimeDelta*(Key(kPgUp)-Key(kPgDn));
    param0.z += 1.*iTimeDelta*(Key(kSquareRight)-Key(kSquareLeft));
    param0.w += 1.*iTimeDelta*(Key(kPeriod)-Key(kComma));
    
    if ( KeyToggle(kHome) > 0. )
    {
        AutoFocus( param0.y );
    }
    
    return reset?log2(resetValue):param0;
}

// Function 223
vec3 update(in vec3 vel, vec3 pos, in float id)
{
    vec4 sndNFO = texture(iChannel2, vec2(0.75, 0.25));
    float R = 1.5;
    const float r = .5;
    float t= time*2.+id*8.;
    float d= 5.;
    
    float x = ((R-r)*cos(t-time*0.1) + d*cos((R-r)/r*t));
    float y = ((R-r)*sin(t) - d*sin((R-r)/r*t));
    
    vel = mix(vel, vec3(x*1.2,y,sin(time*12.6+id*50. + sndNFO.z*10.)*7.)*5. +hash3(vel*10.+time*0.2)*7., 1.);
    
    //vel.z += sin(time*sndNFO.z)*50.;
    //vel.z += sin(time + sndNFO.z*70.)*10.;
    //vel.z += sin(time)*30.*sndNFO.x;
    
    return vel;
}

// Function 224
vec4 getParticle(vec2 id)
{
    // this is cleverly used elsewhere to interpolate quantities between particles iirc.  or not?
    return texture(iChannel0, (id + .5) / iResolution.xy);
    	//getParticle(ivec2(id)); // won't work quite the same
}

// Function 225
void accelerationUpdate (in vec2 offset) {

	// Get the position of the cell
	vec2 cellPosition = floor (particlePosition + offset) + 0.5;

	// Get the particle ID and the collider
	vec4 data = texture (iChannel2, cellPosition / iResolution.xy);
	vec2 particleId = data.rg;
	float collider = data.a;

	// Check whether there is a particle here
	if (offset == vec2 (0.0)) {

		// This is the current particle
		particleIdCheck = particleId;
	} else if (particleId.x > 0.0) {

		// Get the position of this other particle
		data = texture (iChannel1, particleId / iResolution.xy);
		vec2 otherParticlePosition = data.ba;

		// Compute the distance between these 2 particles
		vec2 direction = otherParticlePosition - particlePosition;
		float dist = length (direction);

		// Check whether these 2 particles touch each other
		if (dist < 2.0 * PARTICLE_RADIUS) {

			// Normalize the direction
			direction /= dist;
			dist /= 2.0 * PARTICLE_RADIUS;

			// Get the velocity and density of this other particle
			vec2 otherParticleVelocity = data.rg;
			data = texture (iChannel0, particleId / iResolution.xy);
			float otherParticleDensity = data.r;
			float otherParticleDensityFactor = data.g;

			// Apply the pressure and viscosity forces (SPH)
			float compression = 1.0 - dist;
			float pressure = PARTICLE_PRESSURE_FACTOR * (particleDensityFactor + otherParticleDensityFactor);
			float viscosity = PARTICLE_VISCOSITY_FACTOR * max (0.0, dot (particleVelocity - otherParticleVelocity, direction)) / ((particleDensity + otherParticleDensity) * dist);
			particleAcceleration -= direction * (pressure + viscosity) * 3.0 * compression * compression;
		}
	}

	// Collision with a collider?
	if (collider > 0.5) {

		// Compute the signed distance between the center of the particle (circle) and the border of the collider (square)
		vec2 direction = cellPosition - particlePosition;
		vec2 distCollider = abs (direction) - COLLIDER_RADIUS;
		float dist = length (max (distCollider, 0.0)) + min (max (distCollider.x, distCollider.y), 0.0);

		// Check whether the particle touches the collider
		if (dist < PARTICLE_RADIUS) {

			// Normalize the direction
			direction = sign (direction) * (dist > 0.0 ? distCollider / dist : step (distCollider.yx, distCollider));

			// Apply the collision force (spring)
			float compression = 1.0 - (dist + COLLIDER_RADIUS) / (PARTICLE_RADIUS + COLLIDER_RADIUS);
			particleAcceleration -= direction * (compression * COLLIDER_SPRING_STIFFNESS + dot (particleVelocity, direction) * COLLIDER_SPRING_DAMPING);
		}
	}
}

// Function 226
float UpdateSceneMode(in float currentSceneMode)
{
    if(WasKeyJustPressed(KEY_LEFT))
    {
        currentSceneMode = (currentSceneMode >= MONTE_CARLO_MODE) ? ATTRACT_MODE : (currentSceneMode + 1.0);
    }
    else if(WasKeyJustPressed(KEY_RIGHT))
    {
        currentSceneMode = (currentSceneMode <= ATTRACT_MODE) ? MONTE_CARLO_MODE : (currentSceneMode - 1.0);
    } 
    return currentSceneMode;
}

// Function 227
vec4 getParticle(int i)
{
    // read from the buffer
    return texelFetch(iChannel0, ivec2(i, 0), 0);
}

// Function 228
void updateSun() {
    float fSpeed = fSunSpeed * iTime;
    v3sunDir = normalize( vec3(cos(fSpeed),sin(fSpeed),0.0) );
}

// Function 229
void UpdateBossBullet( inout vec4 bossBullet, vec4 playerHitBox, float screenWidth, float screenHeight )
{
    if ( !Collide( bossBullet.xy, POWER_BULLET_SIZE, vec2( gCamera.x + screenWidth * 0.5, 0.0 ), vec2( screenWidth, screenHeight ) ) )
    {
        bossBullet.x = 0.0;
    }
    
	if ( bossBullet.x > 0.0 )
    {
        bossBullet.xy += bossBullet.zw;
        bossBullet.w -= 1.0 / 10.0;
    }
   
	if ( Collide( playerHitBox.xy, playerHitBox.zw, bossBullet.xy, POWER_BULLET_SIZE ) )
    {
        PlayerHit( playerHitBox );
        bossBullet.x = 0.0;
    }        
}

// Function 230
vec4 UpdateParam1( vec4 resetValue )
{
    // saturation, white balance
    vec4 param1 = texelFetch(iChannel0,ivec2(3,0),0);
    
// todo adjust param1
    
    return reset?resetValue:param1;
}

// Function 231
vec2 packParticle(particle p){
    uvec2 px = uvec2(p.coord);
    uvec3 c = uvec3(p.color * 7000. + 1000.);
    uint n = uint(p.nil);
    uint x = px.x & 0x7FFu;
    uint y = px.y & 0x7FFu;
    uint r = c.r & 0x1FFFu;
    uint g = c.g & 0x1FFFu;
    uint b = c.b & 0x1FFFu;
    uint A = (b >> 9) | (g << 4) | (r << 17) | (n << 30);
    uint B = (y) | (x << 11) | ((b & 0x1FFu) << 22);
    return vec2(uintBitsToFloat(A),uintBitsToFloat(B));
}

// Function 232
vec4 updateState(vec2 fragCoord) {
    vec4 previousState = fetchSimState(fragCoord, ivec2(0, 0));
    vec4 nextState = previousState;
    
    // Sand falls down if there is empty space below.
    // Sand can only fall into a cell if there is no sand already in it.
    // Each cell (fragment) wants to know "Will I have sand next tick?".
    
    vec4 stateLeft  = fetchSimState(fragCoord, ivec2(-1, 0));
    vec4 stateRight = fetchSimState(fragCoord, ivec2( 1, 0));
    
    if (previousState.x > 0.0) {
        // This cell has sand. Keep it or let it fall below.
        vec4 stateBelow      = fetchSimState(fragCoord, ivec2( 0, -1));
        vec4 stateBelowLeft  = fetchSimState(fragCoord, ivec2(-1, -1));
        vec4 stateBelowRight = fetchSimState(fragCoord, ivec2( 1, -1));
        #ifdef SOLID_GROUND_BELOW
        if (fragCoord.y < 1.0) {
            stateBelow      = vec4(1.0);
            stateBelowLeft  = vec4(1.0);
            stateBelowRight = vec4(1.0);
        }
        #endif  // SOLID_GROUND_BELOW
        #ifdef WALLS_ON_THE_SIDES
        if (fragCoord.x < 1.0)                 { stateBelowLeft  = vec4(1.0); }
        if (fragCoord.x > iResolution.x - 1.0) { stateBelowRight = vec4(1.0); }
        #endif  // WALLS_ON_THE_SIDES
        
        if (stateBelow.x == 0.0) {
            // Fall down.
            nextState.x = 0.0;
            nextState.y = 0.0;
        } else if ( updatingLeft() && stateBelowLeft.x  == 0.0 && stateLeft.x  == 0.0) {
            // Fall down left.
            nextState.x = 0.0;
            nextState.y = 0.0;
        } else if (!updatingLeft() && stateBelowRight.x == 0.0 && stateRight.x == 0.0) {
            // Fall down right.
            nextState.x = 0.0;
            nextState.y = 0.0;
        } else {
            // Keep sand in this cell. Keep previous state.
        }
    } else {
        // TODO: Remove else? Can both steps run in a single pass?
        
        // This cell does not have sand. Try to receive sand from above.
        vec4 stateAbove      = fetchSimState(fragCoord, ivec2( 0, 1));
        vec4 stateAboveLeft  = fetchSimState(fragCoord, ivec2(-1, 1));
        vec4 stateAboveRight = fetchSimState(fragCoord, ivec2( 1, 1));
        
        if (stateAbove.x > 0.0) {
            // Receive from above.
            nextState.x = stateAbove.x;
            nextState.y = stateAbove.y;
        } else if ( updatingLeft() && stateAboveRight.x != 0.0 && stateRight.x != 0.0) {
            // Receive from above right.
            nextState.x = stateAboveRight.x;
            nextState.y = stateAboveRight.y;
        } else if (!updatingLeft() && stateAboveLeft.x  != 0.0 && stateLeft.x  != 0.0) {
            // Receive from above left.
            nextState.x = stateAboveLeft.x;
            nextState.y = stateAboveLeft.y;
        } else {
            // No sand to recieve. Keep previous state.
        }
    }
    
    return nextState;
}

// Function 233
void UpdateForward(inout vec4 fragColor)
{
    fragColor = vec4(Rotate(GetLastRotation(), Forward), 0.0);
}

// Function 234
vec4 updateAutoModeState() {
    return isKeyToggled(KEY_A) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);    
}

// Function 235
vec3 particleColor (in float particleVelocity) {
	return mix (vec3 (0.8, 0.2, 0.2), vec3 (1.0, 1.0, 0.5), particleVelocity * PARTICLE_VELOCITY_FACTOR);
}

// Function 236
void InitParticles()
{
	// Because pold isn't reference here,
	// compiler should eliminate fetches of
	// unused second pixel of each particle.  I hope!
    flagbblo = vec3(9e9); flagbbhi = vec3(-9e9);
    for (int j = 0; j < nlinky; ++j)
    for (int i = 0; i < nlinkx; ++i) {
		vec3 p = ps[pidx(i, j)] = particle(ClothBuf, ivec2(i, j)).pnew;
        flagbblo = min(flagbblo, p); flagbbhi = max(flagbbhi, p);
    }   
}

// Function 237
float particle(vec2 uv, vec2 p, vec2 v, float r, float t) {
    float x = p.x + v.x * t;
    float y = p.y + v.y * t + g / 2.0 * t * t;
    vec2 j = (vec2(x, y) - uv) * 20.0;
    float sparkle = 1.0 / dot(j, j);
    return sparkle;
}

// Function 238
vec4 UpdateMouseClick(vec4 o, vec2 m)
{float l=GetLastMouseClick().z
;float isClicked = step(.5,iMouse.z)//todo: mouse and iMouse, i smell bad style
;if((isClicked>.5)&& l<.5)o.xy=vec2(m.xy/iResolution.xy)
;o.z=isClicked
;return o;}

// Function 239
void UpdateTurret( inout vec4 turret, inout vec4 turretState, vec2 playerTarget )
{
    if ( turret.x + TURRET_SIZE.x * 0.5 < gCamera.x )
    {
        turret.x = 0.0;
    }    
    
	vec2 turretAim = normalize( playerTarget - turret.xy );

    // constrain barrel to one of the 12 possible rotations
    float turretAimAngle = atan( -turretAim.y, turretAim.x );    
    turretAimAngle = turretAimAngle / ( 2.0 * MATH_PI );
    turretAimAngle = floor( turretAimAngle * 12.0 + 0.5 );
    turret.z = mod( turretAimAngle + 6.0, 12.0 );
    turretAimAngle = turretAimAngle * 2.0 * MATH_PI / 12.0;
    turretAim = vec2( cos( turretAimAngle ), -sin( turretAimAngle ) );
    
    ++turretState.y;
	if ( turret.x > 0.0 && turretState.y > TURRET_FIRE_RATE )
    {
        turretState.y = 0.0;
		SpawnEnemyBullet( turret.xy, turretAim );
    }
}

// Function 240
vec4 saveParticle(particle P, vec2 pos)
{
    vec2 x = clamp(P.X - pos, vec2(-0.5), vec2(0.5));
    return vec4(encode(x), P.M, P.V);
}

// Function 241
vec4 getParticle(int id)
{
    return texel(ch1, i2xy(id));
}

// Function 242
void updateControls(sampler2D sampler) {
    for (int i = 0 ; i < controlCount; i++) {
        vec4 readout = getControl(i, sampler);
        controls[i].value = readout[2];
        controls[i].value2 = readout[0];
        controls[i].mouseDown = readout[1] > GRN_MID;
        float alphanorm = 1.;
        if (controls[i].opacity > 0.) {
            alphanorm = (readout[3] / controls[i].opacity);
        }
        controls[i].visible = alphanorm > 0.;
        controls[i].enabled = alphanorm > .5;        
        updateValue2(i);
    }    
}

// Function 243
vec3 calcLightOnParticle( vec3 particlePos, Sphere lightSphere, vec3 Li ) {
    vec3 wi;
    return calcDirectLight( particlePos, wi, lightSphere, Li );
}


// End module: particle_systems_functions

// Default main function
uniform vec2 iResolution;
uniform float iTime;
in vec2 fragCoord;
out vec4 fragColor;

void main() {
    vec2 uv = fragCoord / iResolution.xy;
    // Combined shader output
    fragColor = vec4(uv, 0.5 + 0.5 * sin(iTime), 1.0);
}