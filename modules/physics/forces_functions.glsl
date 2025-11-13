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

