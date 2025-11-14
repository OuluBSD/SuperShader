// Reusable Collision Detection Physics Functions
// Automatically extracted from particle/physics simulation-related shaders

// Function 1
void collide (in vec2 offset) {

	// Get the position of the cell
	vec2 cellPosition = floor (particlePosition + offset) + 0.5;

	// Get the particle ID and the collider
	vec4 data = texture (iChannel1, cellPosition / iResolution.xy);
	vec2 particleId = data.rg;
	float collider = data.a;

	// Check whether there is a particle here
	if (offset == vec2 (0.0)) {

		// This is the current particle
		particleIdCheck = particleId;
	}
	else if (particleId.x > 0.0) {

		// Get the velocity and position of this other particle
		data = texture (iChannel0, particleId / iResolution.xy);
		vec2 otherParticleVelocity = data.rg;
		vec2 otherParticlePosition = data.ba;

		// Compute the distance between these 2 particles
		vec2 direction = otherParticlePosition - particlePosition;
		float distSquared = dot (direction, direction);

		// Check whether these 2 particles touch each other
		if (distSquared < 4.0 * RADIUS_PARTICLE * RADIUS_PARTICLE) {

			// Normalize the direction
			float dist = sqrt (distSquared);
			direction /= dist;

			// Apply the collision force (spring)
			float compression = 2.0 * RADIUS_PARTICLE - dist;
			particleForce -= direction * (compression * COLLISION_SPRING_STIFFNESS_PARTICLE - dot (otherParticleVelocity - particleVelocity, direction) * COLLISION_SPRING_DAMPING);
		}
	}

	// Collision with a collider?
	if (collider > 0.5) {

		// Compute the distance between the center of the particle and the collider
		vec2 direction = cellPosition - particlePosition;
		vec2 distCollider = max (abs (direction) - RADIUS_COLLIDER, 0.0);
		float distSquared = dot (distCollider, distCollider);

		// Check whether the particle touches the collider
		if (distSquared < RADIUS_PARTICLE * RADIUS_PARTICLE) {

			// Normalize the direction
			float dist = sqrt (distSquared);
			direction = sign (direction) * distCollider / dist;

			// Apply the collision force (spring)
			float compression = RADIUS_PARTICLE - dist;
			particleForce -= direction * (compression * COLLISION_SPRING_STIFFNESS_COLLIDER + dot (particleVelocity, direction) * COLLISION_SPRING_DAMPING);
		}
	}
}

// Function 2
vec4 get_collision_plane(int index)
{
    ivec2 addr = ivec2(ADDR_RANGE_COLLISION_PLANES.xy);
    addr.x += index;
    return texelFetch(SETTINGS_CHANNEL, addr, 0);
}

// Function 3
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

// Function 4
bool AABBCollision(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax, out float minDist)
{
    vec3 invRd = 1.0 / rd;
    vec3 t1 = (boxMin - ro) * invRd;
    vec3 t2 = (boxMax - ro) * invRd;

    vec3 mins = min(t1, t2);
    vec3 maxs = max(t1, t2);

    float tmin = max(mins.x, max(mins.y, mins.z));
    float tmax = min(maxs.x, min(maxs.y, maxs.z));

    minDist = tmin;
    return tmax > tmin && tmax > 0.0;
}

// Function 5
vec2 collisionWithPlane(inout Body b0, vec3 plane) {
    vec2 normal = normalize(plane.xy);
    float dist = dot(b0.pos,normal) + plane.z;
    float penetration = BALL_SIZE - dist;
    if(penetration > 0.0) {
        vec2 r0 = -normal * BALL_SIZE;        

        // normal
        vec2 vel0 = b0.vel + cross2(b0.ang_vel,r0);
        vec2 rel_vel = vel0;  
        
        float w1 = cross2(r0,normal);

        float a = (1.0 + ELASTICITY) * dot(normal,rel_vel);
        float b = b0.inv_mass + w1 * w1 * b0.inv_inertia;
        float lambda = max(-a / b, 0.0);

        b0.vel += normal * (lambda * b0.inv_mass);
        b0.ang_vel += cross2(r0, normal) * lambda * b0.inv_inertia;

        // friction
        vel0 = b0.vel + cross2(b0.ang_vel,r0);
        rel_vel = vel0;  

        vec2 tangent = cross2(normal,1.0);
        w1 = cross2(r0,tangent);

        a = (1.0 + ELASTICITY) * dot(tangent,rel_vel);
        b = b0.inv_mass + w1 * w1 * b0.inv_inertia;
        float lambdaF = clamp(-a / b, -lambda, lambda);

        b0.vel += tangent * (lambdaF * b0.inv_mass);
        b0.ang_vel += cross2(r0, tangent) * lambdaF * b0.inv_inertia;
        
        return normal * penetration;
    }
    return vec2(0.0);
}

// Function 6
void BodyCollideShapeSphere( inout Body body, vec3 vSphereOrigin, float fSphereRadius, float dT )
{    
    vec3 vSphereWorld = ObjToWorld( vSphereOrigin, body.mRot) + body.vPos;
    
    ClosestSurface closest = GetSceneClosestSurface( vSphereWorld );
    
    float fDepth = fSphereRadius - closest.fDist;
    
    if ( fDepth < 0.0 )
        return;
    
    vec3 vNormal = GetSceneNormal( vSphereWorld );
    vec3 vHitPos = vSphereWorld - vNormal * closest.fDist;    
    vec3 vPointVel = BodyPointVelocity( body, vHitPos );
    
    float fDot = dot( vPointVel, vNormal );
    
    if( fDot >= 0.0 )
        return;
    
    float fRestitution = 0.5;
    
    vec3 vRelativePos = (vHitPos - body.vPos);
    float fDenom = (1.0/body.fMass );
    float fCr = dot( cross( cross( vRelativePos, vNormal ), vRelativePos), vNormal);
    fDenom += fCr / body.fIT;
    
    float fImpulse = -((1.0 + fRestitution) * fDot) / fDenom;
    
    fImpulse += fDepth / fDenom;
    
    vec3 vImpulse = vNormal * fImpulse;
    
    vec3 vFriction = Vec3Perp( vPointVel, vNormal ) * body.fMass;
    float fLimit = 100000.0;
    float fMag = length(vFriction);
    if( fMag > 0.0 )
    {	        
        vFriction = normalize( vFriction );

        fMag = min( fMag, fLimit );
        vFriction = vFriction * fMag;

        //BodyApplyForce( body, vHitPos, vFriction );
        vImpulse += vFriction * dT;        
    }
    else
    {
        vFriction = vec3(0.0);
    }
    
    BodyApplyImpulse( body, vHitPos, vImpulse );
}

// Function 7
bool Collision(vec3 p, vec3 v, vec3 omega, vec4 n_and_d, out float vrel) {
    if (n_and_d.w < 0.01) { 
    	vrel = dot(n_and_d.xyz, v + cross(omega, p));
        
        return vrel < 0.0;
    }
    
    return false;
}

// Function 8
void CollideEye(inout vec3 p)
{
	// FIXME should actually distribute the movement throughout all these iterations - see fixes in Plumbing Maze
	// multiple collision iterations to prevent tunnelling at low fps
	for (int i = 3; --i >= 0; ) { // repeating helps with getting stuck in crevices
    	CollideSphere(p, eyeradius);
//		pos.y = max(pos.y, radius); // HACK prevent going beneath ground plane just in case
	}
}

// Function 9
void Entity_Collide( inout Entity entity, Entity otherEntity, float fTimestep )
{
    // True if we can be pushed
    if ( CanBePushed( entity, otherEntity ) )
    {
        vec2 vDeltaPos = entity.vPos.xz - otherEntity.vPos.xz;
        vec2 vDeltaVel = entity.vVel.xz - otherEntity.vVel.xz;

        float fLen = length( vDeltaPos );
        float fVelLen = length(vDeltaVel);
        float fCombinedRadius = 20.0;
        if ( fLen > 0.0 && fLen < fCombinedRadius )
        {
            vec2 vNormal = normalize(vDeltaPos);
            
            if ( fVelLen > 0.0 )
            {
                float fProj = dot( vNormal, vDeltaVel );

                if ( fProj < 0.0 )
                {
                    // cancel vel in normal dir
                    vec2 vImpulse = -fProj * vNormal;
                    
                    // Push away
                    float fPenetration = fCombinedRadius - fLen;
                    vImpulse += vNormal * fPenetration * 5.0 * fTimestep;
                    
                    
				    if ( CanBePushed( otherEntity, entity ) )
                    {
                    	entity.vVel.xz += vImpulse * 0.5;
                    }
                    else
                    {
                    	entity.vVel.xz += vImpulse;
                    }
                }
            }            
        }        
    }    
}

// Function 10
void handleInterBallCollisions(int ballIdx, inout vec2 pos, inout vec2 vel)
{
    // Iterate through all the other balls to see if we are colliding with any.
    // If we are, then update our position and velocity accordingly

    int iChanRes0 = int(iChannelResolution[0].x);

    for (int i = 0; i < NUM_BALLS; i++) 
    {
		// If it's the same ball, ignore it (can't collide with self)
        if (i == ballIdx) continue;

        // Get the position and speed of ball i.
        vec4 otherPosAndVel = texelFetch(iChannel0, BufferPixelPosFromBallIndex(i, iChanRes0), 0);
        vec2 otherPos = otherPosAndVel.xy;
        vec2 otherVel = otherPosAndVel.zw;

        // Calculate the distance between the centers of the two balls.
        vec2 delta = pos - otherPos;
        float dist = length(delta);

        // If it's smaller than the diameter, we have a collision.
        if (dist < (R+R)) 
        {
            vec2 collision_normal = normalize(delta);
            vec2 collision_tangent = vec2(collision_normal.y, -collision_normal.x);
            // How fast *the other ball* is going in the "collision_normal" direction.
            float a1 = dot(otherVel, collision_normal);
            // How fast *this* ball is going in the "collision_normal" direction.
            float a2 = dot(vel, collision_normal);
            // How fast *this* ball is going in the "collision tangent" direction.
            float b = dot(vel, collision_tangent);
            // Our new speed.
            vel = collision_normal * (a2 + (a1 - a2) * 0.9) + collision_tangent * b;
            
            // Also, move the ball away to make sure we're not colliding anymore.
            pos = otherPos + delta * R * 2.01 / dist;
        }
    }   
}

// Function 11
void CollideSphere(inout vec3 pos, float radius)
{
    float d;
    vec3 n = SceneNormal(pos, radius, d, IZERO);
    pos -= n * min(0., d - radius);
}

// Function 12
void CollideCapsule(inout vec3 pos, vec3 ofs, float radius)
{
    vec3 p = pos, no; float d = 3.4e38;
	int ncapsph = IZERO + 5; //7; //9; //3; //
	for (float d2, st = 2./float(ncapsph), j = -1.; j <= 1.; j += st) {
		vec3 ns = SceneNormal(p + j * ofs, .1*radius, d2, IZERO);
		if (d2 < d) { d = d2; no = ns; }
	}
	pos -= no * min(0., d  - radius);
}

// Function 13
bool CollideEye(inout vec3 p)
{
	float eh = .5 * eyeh - eyeradius;
	vec3 po = p; // HACK not best way to detect collision; too late anyway
	// FIXME should actually distribute the movement throughout all these iterations - see fixes in Plumbing Maze
	// multiple collision iterations to prevent tunnelling at low fps
	for (int i = 3; --i >= 0; ) { // repeating helps with getting stuck in crevices
		p.y -= eh; // capsule center is below the eyes
		CollideCapsule(p, vec3(0,eh,0), eyeradius);
		p.y += eh;
//		pos.y = max(pos.y, radius + abs(ofs.y)); // HACK prevent going beneath ground plane just in case
	} // FIXME the response when stepping up onto steps can be extremely bouncy
	return dot(po-p,po-p) > 1e-8; // must be very sensitive; probably breaks at high fps
}

// Function 14
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

// Function 15
void debugCollision(int ci,int cj)
{
    vec4 sep = readTex0(ci+CUBECOUNT*12,cj);
    for(int k=0;k<48;k++)
    {
        vec3 cpos = getCollision(ci,cj,k);
        debugdot(cpos);
    	debugline(cpos,cpos + readTex2(ci*12+k%12,cj*4+k/12).xyz*10.0 );
    }
}

// Function 16
void docollisions()
{
    ballwall(0);
    ballwall(1);
    ballvsball(0,1);
    if (ballCount>2)
    {
    	ballwall(2);
    	ballvsball(0,2);
    	ballvsball(1,2);
        if (ballCount>3)
        {
    		ballwall(3);
            ballvsball(0,3);
            ballvsball(1,3);
            ballvsball(2,3);
        }
    }
}

// Function 17
void handle_collision(inout vec3 start, vec3 delta, int slide_plane, out int hit_plane, out int ground_plane)
{
    // We iterate again through all the collision brushes, this time performing two ray intersections:
    // one determines how far we can actually move, while the other does a ground check from the starting
    // point, giving us an approximate ground plane.
    // Note that the ground plane isn't computed from the final position - that would require another pass
    // through all the brushes!
    
    const float LARGE_NUMBER = 1e+6;

    hit_plane = -1;
    ground_plane = -1;
    float travel_dist = 1.;
    float ground_dist = LARGE_NUMBER;
    float eps = 1./(length(delta) + 1e-6);

    int num_brushes = NO_UNROLL(NUM_MAP_COLLISION_BRUSHES);
    for (int i=0; i<num_brushes; ++i)
    {
        int first_plane = cm_brushes.data[i];
        int last_plane = cm_brushes.data[i + 1];
        int plane_enter = -1;
        int plane_enter_ground = -1;
        float t_enter = -LARGE_NUMBER;
        float t_leave = LARGE_NUMBER;
        float t_enter_ground = t_enter;
        float t_leave_ground = t_leave;
        for (int j=first_plane; j<last_plane; ++j)
        {
            vec4 plane = get_collision_plane(j);
            float dist = get_player_distance(start, plane);

            // handle ground ray
            if (plane.z == 0.)
            {
                if (dist > 0.)
                    t_enter_ground = LARGE_NUMBER;
            }
            else
            {
                float height = dist / plane.z;
                if (plane.z > 0.)
                {
                    if (t_enter_ground < height)
                    {
                        plane_enter_ground = j;
                        t_enter_ground = height;
                    }
                }
                else
                {
                    t_leave_ground = min(t_leave_ground, height);
                }
            }

            // handle movement ray
            float align = dot(plane.xyz, delta);
            if (align == 0.)
            {
                if (dist > 0.)
                    t_enter = LARGE_NUMBER;
                continue;
            }
            align = -1./align;
            dist *= align;
            if (align > 0.)
            {
                if (t_enter < dist)
                {
                    plane_enter = j;
                    t_enter = dist;
                }
            }
            else
            {
                t_leave = min(t_leave, dist);
            }
        }

        if (t_leave_ground > t_enter_ground && t_enter_ground > -8.)
        {
            if (t_enter_ground < ground_dist)
            {
                ground_plane = plane_enter_ground;
                ground_dist = t_enter_ground;
            }
        }

        if (t_leave > max(t_enter, 0.) && t_enter > -eps)
        {
            if (t_enter < travel_dist)
            {
                hit_plane = plane_enter;
                travel_dist = t_enter;
            }
        }
    }

    start += delta * clamp(travel_dist, 0., 1.);
    delta *= 1. - clamp(travel_dist, 0., 1.);

    if (hit_plane != -1)
    {
        vec4 plane = get_collision_plane(hit_plane);
        start += 1e-2 * plane.xyz;
        delta -= dot(plane.xyz, delta) * plane.xyz;
    }
}

// Function 18
vec2 collisionWithBody(inout Body b0, in Body b1) {
    vec2 normal = b0.pos - b1.pos;
    float dist = length(normal);
    float penetration = 2.0 * BALL_SIZE - dist;
    if(penetration > 0.0) {
        normal /= dist;

        vec2 r0 = -normal * BALL_SIZE;
        vec2 r1 = normal * BALL_SIZE;
        
        // normal
        vec2 vel0 = b0.vel + cross2(b0.ang_vel,r0);
        vec2 vel1 = b1.vel + cross2(b1.ang_vel,r1);
        vec2 rel_vel = vel0 - vel1;
        
        float w1 = cross2(r0,normal);
        float w2 = cross2(r1,normal);

        float a = (1.0 + ELASTICITY) * dot(normal,rel_vel);
        float b = b0.inv_mass + b1.inv_mass +
            w1 * w1 * b0.inv_inertia +
            w2 * w2 * b1.inv_inertia;
        float lambda = max(-a / b, 0.0);

        b0.vel += normal * (lambda * b0.inv_mass);
        b0.ang_vel += cross2(r0, normal) * lambda * b0.inv_inertia;
        b1.vel -= normal * (lambda * b1.inv_mass);
        b1.ang_vel -= cross2(r1, normal) * lambda * b1.inv_inertia;

        // friction
        vel0 = b0.vel + cross2(b0.ang_vel,r0);
        vel1 = b1.vel + cross2(b1.ang_vel,r1);
        rel_vel = vel0 - vel1;  

        vec2 tangent = cross2(normal,1.0);
        w1 = cross2(r0,tangent);
        w2 = cross2(r1,tangent);

        a = (1.0 + ELASTICITY) * dot(tangent,rel_vel);
        b = b0.inv_mass + b1.inv_mass +
            w1 * w1 * b0.inv_inertia +
            w2 * w2 * b1.inv_inertia;
        float lambdaF = clamp(-a / b, -lambda, lambda);

        b0.vel += tangent * (lambdaF * b0.inv_mass);
        b0.ang_vel += cross2(r0, tangent) * lambdaF * b0.inv_inertia;
        
        return normal * penetration * 0.5;
    }
    return vec2(0.0);
}

// Function 19
void collision(vec2 f, vec2 R, inout particle parts)
{
    vec2 np = parts.pos + parts.vit;
    vec2 cp = vec2(.45, .0);
    vec2 dd = np -cp;
	if (length(dd) < .25+parts.ma/24.)
    {
        //parts.vit *= -.95;
        parts.acc *= -.75;
        parts.vit *= -1./(length(dd)+.505);
    }

}

// Function 20
void BodyCollide( inout Body body, float dT )
{
    BodyCollideShapeSphere( body, vec3( 0.7, 0.7,  1.5), 0.5, dT );
    BodyCollideShapeSphere( body, vec3(-0.7, 0.7,  1.5), 0.5, dT );
    BodyCollideShapeSphere( body, vec3( 0.7, 0.7, -1.5), 0.5, dT );
    BodyCollideShapeSphere( body, vec3(-0.7, 0.7, -1.5), 0.5, dT );
    BodyCollideShapeSphere( body, vec3( 0.5, 1.0,  0.0), 0.7, dT );
    BodyCollideShapeSphere( body, vec3(-0.5, 1.0,  0.0), 0.7, dT );
}

// Function 21
void handleWallCollisions(inout vec2 pos, inout vec2 vel)
{
    // If we are within distance R of the bottom, then we have hit the bottom (and likely gone past it)
    if (pos.y < R) 
    { 
        pos.y = (R+R)-pos.y; // Bounce off the bottom, so that we are ALWAYS at least R away
        vel.y = abs(vel.y * 0.9); // Make sure we are moving up, and lose some energy
    }

    // Similarly for left and right walls
    // If we reach left/right wall, invert the speed in the x direction. 
    if (pos.x < R) { pos.x = (R+R)-pos.x; vel.x = abs(vel.x * 0.9); }
    if (pos.x > 1.0-R) { pos.x = (2.0-(R+R)) - pos.x; vel.x = -abs(vel.x * 0.9); }
}

// Function 22
bool Collide( vec2 p0, vec2 s0, vec2 p1, vec2 s1 )
{
    // pivot x in the middle, and y in the bottom
    p0.x -= s0.x * 0.5;
    p1.x -= s1.x * 0.5;
    
    return 		p0.x <= p1.x + s1.x
        	&& 	p0.y <= p1.y + s1.y
        	&& 	p1.x <= p0.x + s0.x
        	&& 	p1.y <= p0.y + s0.y;
}

// Function 23
float Collide( in vec3 pos, in vec3 destination, in float radius )
{
    radius -= collisionThreshold;
    
    float d = length(destination-pos);
    vec3 r = (destination-pos)/d;
    
	// DON'T do SDF+radius - SDF's gradient is <= 1.0 so ball will hover!
    // instead, sample at the front of the ball
    // this means it won't collide properly in tight nooks
//    pos += r*radius;
//AARGH! that ruins rolling!
// => maybe we need a correct, analytical gradient
// or maybe we can correct it because we have more time
    vec3 n = GetNormal(pos,.001,false);
    pos -= n*radius; // aha! displace it toward closest surface
    
    float h = SDF(pos,false);
    
    // early out
    if ( h >= d
       || ( h < collisionThreshold && dot(r,n) > 0. ) ) // hack, don't collide if we're below the ground & moving outward
    {
        return 1.0;
    }
    
    float t = 0.;
    for ( int i=0; i < 20; i++ )
    {
        t += h;
        h = SDF(pos+r*t,false);
        if ( t > d || h < collisionThreshold )
            break;
    }
    
    t /= d;
    return min(t,1.0);
}

// Function 24
void CollisionImpulse(vec3 x, vec4 q, vec3 v, vec3 omega, mat3 invI, inout vec3 P , inout vec3 L) {
    vec3[8] lamp_box = vec3[](
        vec3(-lamp.x, -lamp.y, -lamp.z),
        vec3(lamp.x, -lamp.y, -lamp.z),
        vec3(-lamp.x, lamp.y, -lamp.z),
        vec3(lamp.x, lamp.y, -lamp.z),
        vec3(-lamp.x, -lamp.y, lamp.z),
        vec3(lamp.x, -lamp.y, lamp.z),
        vec3(-lamp.x, lamp.y, lamp.z),
        vec3(lamp.x, lamp.y, lamp.z));
    
    vec4[8] normal_and_dist;
    for (int i = 0; i < 8; ++i) {
        lamp_box[i] = quatRotate(q, lamp_box[i]);
    	vec3 p = lamp_box[i] + x;
        normal_and_dist[i] = vec4(-sdBoxNormal(p, room), -sdBox(p, room));
        lamp_box[i].y = max(-room.y, lamp_box[i].y);
    }
    
    for (int s = 0; s < 10; ++s){
     	for (int i = 0; i < 8; ++i) {
   			vec3 p = lamp_box[i];
            
            float vrel = 0.0;
        	if (Collision(p, v, omega, normal_and_dist[i], vrel)) {
                const float epsilon = 0.5;
                vec3 n = normal_and_dist[i].xyz;
                vec3 f = n*((-vrel*(1.0 + epsilon)) 
                         / (invMass + dot(n, cross(invI*cross(p, n), p))));
                P += f;
                L += cross(p, f);
                
                v = P*invMass;
                omega = invI*L;
        	}
        }
    }
}

// Function 25
void Collision(vec3 prev, inout vec3 p) {
    if (p.y < 1.0) p = vec3(prev.xz, min(1.0, prev.y)).xzy;
}

// Function 26
vec4 findCollisionPouint()
{
    uint ci = pixelx/12u;
    uint cj = pixely/4u;
    if (cj>=ci) discard;
    
    if (length(getCubePos(ci)-getCubePos(cj))>6.0 && cj!=0u) // bounding check
    {
        return vec4(0.,0.,0.,0.);
    }
    
    uint j = pixelx%12u;
    
    if (pixely%4u<2u) // swap the two cubes to check collision both ways
    {
        uint t = ci;
        ci = cj;
        cj = t;
    }

    vec3 pa = cubeTransform(cj,edge(j,0u)); // a world space edge of cube j
    vec3 pb = cubeTransform(cj,edge(j,1u));
    float ea=0.0;
    float eb=1.0;
    for(uint l=0u;l<((ci==0u)?1u:6u);l++) // clamp it with the 6 planes of cube i
    {
        vec4 pl = getCubePlane(ci,l);
        pl/=length(pl.xyz);
        if (abs(dot(pl.xyz,pb-pa))>0.0001)
        {
            float e = -(dot(pl.xyz,pa)-pl.w)/dot(pl.xyz,pb-pa);
            if (dot(pb-pa,pl.xyz)>0.0)
            {
                eb=min(eb,e);
            }
            else
            {
                ea=max(ea,e);
            }
        }
        else
        {
            ea=999999.0; // edge is parallel to plane
        }
    }
    
    vec3 coll = pa+(pb-pa)*((pixely%2u==0u)?ea:eb);
    if (eb<=ea || cj==0u)
    {
        coll = vec3(0.,0.,0.);
    }
    
    
    return  vec4(coll,0.0);
}

// Function 27
void find_collision(inout vec3 start, inout vec3 delta, out int hit_plane, out float step_height)
{
    const float STEP_SIZE = 18.;

    // We iterate through all the collision brushes, tracking the closest plane the ray hits and the top plane
    // of the colliding brush.
    // If, at the end of the loop, the closest hit plane is vertical and the corresponding top plane
    // is within stepping distance, we move the start position up by the height difference, update the stepping
    // offset for smooth camera interpolation and defer all forward movement to the next step (handle_collision).
    // If we're not stepping up then we move forward as much as possible, discard the remaining forward movement
    // blocked by the colliding plane and pass along what's left (wall sliding) to the next phase.
    
    step_height = 0.;
    hit_plane = -1;
    float travel_dist = 1.;
    int ground_plane = -1;
    float ground_dist = 0.;
    float eps = 1./(length(delta) + 1e-6);
    vec3 dir = normalize(delta);

    int num_brushes = NO_UNROLL(NUM_MAP_COLLISION_BRUSHES);
    for (int i=0; i<num_brushes; ++i)
    {
        int first_plane = cm_brushes.data[i];
        int last_plane = cm_brushes.data[i + 1];
        int plane_enter = -1;
        int brush_ground_plane = -1;
        float brush_ground_dist = 1e+6;
        float t_enter = -1e+6;
        float t_leave = 1e+6;
        for (int j=first_plane; j<last_plane; ++j)
        {
            vec4 plane = get_collision_plane(j);
            float dist = get_player_distance(start, plane);
            
            // Note: top plane detection only takes into account fully horizontal planes.
            // This means that stair stepping won't work with brushes that have an angled top surface, 
            // such as the ramp in the 'Normal' hallway. If you stop on the ramp and let gravity slide
            // you down you'll notice the sliding continues for a bit after the ramp ends - the collision
            // map doesn't fully match the rendered geometry (and now you know why).
            
            if (abs(dir.z) < .7 && plane.z > .99 && brush_ground_dist > dist)
            {
                brush_ground_dist = dist;
                brush_ground_plane = j;
            }
            float align = dot(plane.xyz, delta);
            if (align == 0.)
            {
                if (dist > 0.)
                {
                    t_enter = 2.;
                    break;
                }
                continue;
            }
            align = -1./align;
            dist *= align;
            if (align > 0.)
            {
                if (t_enter < dist)
                {
                    plane_enter = j;
                    t_enter = dist;
                }
            }
            else
            {
                t_leave = min(t_leave, dist);
            }

            if (t_leave <= t_enter)
                break;
        }

        if (t_leave > max(t_enter, 0.) && t_enter > -eps)
        {
            if (t_enter <= travel_dist)
            {
                if (brush_ground_plane != -1 && -brush_ground_dist > ground_dist)
                {
                    ground_plane = brush_ground_plane;
                    ground_dist = -brush_ground_dist;
                }
                hit_plane = plane_enter;
                travel_dist = t_enter;
            }
        }
    }

    vec4 plane;
    bool blocked = hit_plane != -1;
    if (blocked)
    {
        plane = get_collision_plane(hit_plane);
        if (abs(plane.z) < .7 && ground_plane != -1 && ground_dist > 0. && ground_dist <= STEP_SIZE)
        {
            ground_dist += .05;	// fixes occasional stair stepping stutter at low FPS
            step_height = ground_dist;
            start.z += ground_dist;
            return; // defer forward movement to next step
        }
    }

    start += delta * clamp(travel_dist, 0., 1.);
    delta *= 1. - clamp(travel_dist, 0., 1.);

    if (blocked)
    {
        start += 1e-2 * plane.xyz;
        delta -= dot(plane.xyz, delta) * plane.xyz;
    }
}

