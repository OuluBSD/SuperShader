// Reusable Simulation Procedural Functions
// Automatically extracted from procedural-related shaders

// Function 1
vec3 physics(vec3 pos, vec3 vel, vec3 acc, float t)
{
	//this loop processes upto max_bounces collisions... nice :)
	for (int i=0; i<max_floor_bounce; i++)
	{
		float tc = second(acc.y*.5,vel.y,pos.y);
		//now we know that there will be a collision with the plane 
		//in exactly tc seconds
		
		if (t>tc) //if time is greater than time of collision
		{
			t-=tc; //process the collision
			pos = pos + vel*tc + acc*tc*tc*.5;
			vel = vel + acc*tc;
			vel.y*=-.5; //make it bounce
			vel.x=vel.x*.8+sin(pos.x*4.0)*length(vel)*.5;
		}
		else break; //it wont collide, yay!
	}

	pos = pos + vel*t + acc*t*t*.5; // x = v*t + .5*a*t^2

	float ar = iResolution.x/iResolution.y;
	float hwall = 8.0*ar;
	
	for (int i=0; i<max_wall_bounce; i++)
	{
		if (pos.x>+hwall) pos.x = 2.0*hwall-pos.x;
		else if (pos.x<-hwall) pos.x = -2.0*hwall-pos.x;
		else break;
	}
	return pos;
}

// Function 2
vec4 getParticle(int i)
{
    // read from myself
    return texelFetch(iChannel0, ivec2(i, 0), 0);
}

// Function 3
float particlesLayer(vec2 uv, float seed)
{
   	uv = uv + hash21(seed) * 10.0;
    vec2 rootUV = floor(uv);
    vec2 particleUV = particleCoordFromRootUV(rootUV);
    float particles = particleFromParticleUV(particleUV, uv);
    return particles;
}

// Function 4
float particles(vec2 p)
{
  p *= 200.;
  float f = 0.;
  float amp = 1.0, s = 1.5;
  for (int i=0; i<3; i++)
  { p = m*p*1.2; f += amp*noise( p+iTime*s ); amp = amp*.5; s*=-1.227; }
  return pow(f*.35, 7.)*particle_amount;
}

// Function 5
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

// Function 6
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

// Function 7
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

// Function 8
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

// Function 9
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

// Function 10
vec2 getParticlePosition(in int particleID)
{
    int iChannel0_width = int(iChannelResolution[0].x);
	ivec2 particleCoord = ivec2(particleID % iChannel0_width, particleID / iChannel0_width);
    
    return texelFetch(iChannel0, particleCoord, 0).xy;
}

// Function 11
void initParticle(in vec2 fragCoord, inout vec2 particlePrevPosition, inout vec2 particleCurrPosition)
{
	particleCurrPosition = randVec2(fragCoord) * iResolution.xy;
    particlePrevPosition = particleCurrPosition - randNrm2(fragCoord) * particlesSize * 0.0625;
}

// Function 12
vec3 particleColor (in float particleVelocity) {
	return mix (vec3 (0.8, 0.2, 0.2), vec3 (1.0, 1.0, 0.5), particleVelocity * PARTICLE_VELOCITY_FACTOR);
}

// Function 13
vec3 particleColor(vec2 uv, float radius, float offset, float periodOffset)
{
    vec3 color = palette(.4 + offset / 4., p0);
    uv /= pow(periodOffset, .75) * sin(periodOffset * iTime) + sin(periodOffset + iTime);
    vec2 pos = vec2(cos(offset * offsetMult + time + periodOffset),
        		sin(offset * offsetMult + time * 5. + periodOffset * tau));
    
    float dist = radius / distance(uv, pos);
    return color * pow(dist, 2.) * 1.75;
}

// Function 14
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

// Function 15
void writeParticle(Particle p, inout vec4 color, vec2 coord)
{
    ivec2 res=textureSize(ParticleTex,0);
    //if(particleIdx(coord)!=p.idx) return;
    color=((int(coord.y)*res.x+int(coord.x))%3==0)?vec4(p.pos,p.idx):color;
    color=((int(coord.y)*res.x+int(coord.x))%3==1)?vec4(p.vel,p.sidx):color;
    color=((int(coord.y)*res.x+int(coord.x))%3==2)?vec4(p.nn[0],p.nn[1],p.nn[2],p.nn[3]):color;
    if(particleIdx(coord)>NumParticles) color=vec4(0,1,0,1);
}

// Function 16
vec4 getParticle(int i)
{
    // read from the buffer
    return texelFetch(iChannel0, ivec2(i, 0), 0);
}

// Function 17
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

// Function 18
Particle particle(sampler2D bufA, ivec2 i)
{
    i.x += i.x;
    vec4 d0 = texelFetch(bufA, i, 0)
    , d1 = texelFetch(bufA, i + ivec2(1, 0), 0);
    return Particle(d0.xyz, d1.xyz);
}

// Function 19
Particle GrowNewParticle(int index)
{
    return GrowParticle(index, -1, 0.1);
}

// Function 20
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

// Function 21
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

// Function 22
particle getParticle(vec4 data, vec2 pos)
{
    particle P; 
    P.X = decode(data.x) + pos;
    P.V = decode(data.y);
    P.M = data.z;
    P.I = data.w;
    return P;
}

// Function 23
void writeParticle(Particle p, inout vec4 col, vec2 coord, sampler2D s)
{
    if (particleIdx(coord,s)%PNUM==p.idx) col=vec4(p.pos,p.vel);
}

// Function 24
float particleDistance(int id, vec2 p)
{
    return repD(getParticle(id).xy, p);
}

// Function 25
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

// Function 26
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

// Function 27
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

// Function 28
void Simulate(float time, sampler2D txBuf, int creatureId, int nbElt, int eltId, out vec3 rm, out vec3 vm, out vec4 qm, out vec3 wm, out float rad) {

    // Friction between Elements of the creature
    const float fricNB = 5., fricSB = 5., fricSWB = 10., fricTB = 40.;
    
    // Friction to the ground
    const float fricN = 240., fricS = 240., fricSW = 160., fricT =248.;
    
    // Damping factor
    float fDamp = 0.;

    vec3  vmN, wmN, dr, dv, am, wam;
    float rSep, radSum, fc, ft, ms, h, fOvlap = 15000.;
    
    time *= 5.;
    
    vec4 pc1 = Load(txBuf, POSITION, creatureId, eltId);
    qm = Load(txBuf, ORIENTATION,    creatureId, eltId);
    vm = Load(txBuf, VELOCITY,       creatureId, eltId).xyz;
    wm = Load(txBuf, ROT_VELOCITY,   creatureId, eltId).xyz;

    rm = pc1.xyz;
    rad = pc1.w;

    ms = rad*rad*rad*DENSITY; // Mass

    am = wam = vec3(0); // Sum or forces
 
    // Slow it a bit
    vm *= .99;
    wm *= .97;

#ifdef STATIC_EVOLUTION
    // DEBUG position => no moves
     return;
#endif
    
	// Keep it in the air before simulation
    if (time < FIX_DURATION) {
        return;
    }
    
    // Check intersection between Elements of the creature
    for (int n = 0; n < nbElt; n ++) {
        vec4 pc2 = Load(txBuf, POSITION, creatureId, n);

        dr = pc1.xyz - pc2.xyz;
        
        rSep = length(dr);
        radSum = pc1.w + pc2.w;

        if (n != eltId && rSep < radSum) {
            
            // Impulsion de contact
            fc = fOvlap * (radSum / rSep - 1.);
            
            vmN = Load(txBuf, VELOCITY, creatureId, n).xyz;
            wmN = Load(txBuf, ROT_VELOCITY, creatureId, n).xyz;

            dv = vm - vmN;
            h = dot(dr, dv) / (rSep)*(rSep);

            // friction
            fc = max(fc - fricNB * h, 0.);
            am += fc * dr;
            dv -= h * dr + cross ((2.*pc1.w * wm + 2.*pc2.w * wmN) / (2.*pc1.w + 2.*pc2.w), dr);
            ft = min (fricTB, fricSB * abs (fc) * rSep / max (0.001, length (dv)));
            
            am -= ft * dv;
            wam += (ft / rSep) * cross (dr, dv);
        }
    }
    
    
    vec4 connexions = Load(txBuf, LINK, creatureId, eltId);
    mat3 rot = QtToRMat(qm);
    
    float anim1 = .3*cos(time);
    mat3 rot1, rot2;
    
    // For each possible connexions to other Elts
    for (int con_pos=0; con_pos<4; con_pos++) {
        vec4 pc2; 
        if (getConnexionBases(txBuf, creatureId, connexions, con_pos, qm, pc2, rot1, rot2)) {
            
	        int move_con = CON_TYPE(connexions[con_pos]);
          //  if (move_con == 0) continue; // move_con == 9 => No muscles
            anim1 = .3*cos(time + float(move_con/10)*6.28);
            
            float anim = move_con >= 10 && move_con <20 ? 0. : anim1, // move_con == 8 => fixed articulation
			 	  a = PI*2.*float(move_con)/8.,
             	  ca = cos(a), sa = sin(a),
            	  kForce = dSpringK*mix(5., 100.*(pc1.w*pc1.w*pc1.w + pc2.w*pc2.w*pc2.w), .6); // biggest link => strongest
            
            
            // Ligaments
            if (con_pos == 0)
                applyForce(pc1, rot1, pc2, rot2, vec3(0,0,1), vec3(0,0,-1),  0., am, wam, 2.*kForce);
            else 
                applyForce(pc1, rot1, pc2, rot2, vec3(0,0,-1), vec3(0,0,1),  0., am, wam, 2.*kForce);
            
            if (move_con < 10) continue; // move_con == 9 => No muscles
            applyForce(pc1, rot1, pc2, rot2, vec3(ca,sa,0), vec3(ca,sa,0), pc1.w+pc2.w, am, wam, 2.*kForce);
            applyForce(pc1, rot1, pc2, rot2,-vec3(ca,sa,0),-vec3(ca,sa,0), pc1.w+pc2.w, am, wam, 2.*kForce);
			
            // Muscles
            if (move_con == 9) continue; // move_con == 9 => No muscles
            applyForce(pc1, rot1, pc2, rot2, vec3(sa,ca,0),vec3(sa,ca,0), (pc1.w+pc2.w)*(1.+anim), am, wam, kForce);
            applyForce(pc1, rot1, pc2, rot2,-vec3(sa,ca,0),-vec3(sa,ca,0), (pc1.w+pc2.w)*(1.-anim), am, wam, kForce);
        }
    }
    

    // Intersection with the ground 
    float radAv = rad + .5;
    dr = vec3(0,0,rm.z-CUBE_SIZE.z);
    rSep = length (dr);
    
    if (rSep < radAv) {
        // Out of the ground
        rm -= .5*normalize(dr)*(rSep-radAv);

        fc = fOvlap * (radAv / rSep - 1.);
        dv = vm;
        h = dot (dr, dv) / (rSep * rSep);
        fc = max (fc - fricN * h, 0.);

        am += fc * dr;
        dv -= h * dr + cross (wm, dr);

        ft = min (fricT, fricSW * abs (fc) * rSep / max (0.001, length (dv)));
        am -= ft * dv;
        wam += (ft / rSep) * cross (dr, dv);
    }

    // Integrate all
    am -= vec3 (0., 0., -ms*GRAV) + fDamp * vm;

    vm += DT * am/ms;
    rm += DT * vm;
    wm += DT * wam / (ms*rad); //*rad*.5);//*(ms * rad) ;  // (ms*rad*rad*.5) for inertia matrix
  
    qm = normalize(QtMul(RMatToQt(LpStepMat(.5 * DT * wm)), qm));
}

// Function 29
vec4 getParticle(int id)
{
    return texel(ch1, i2xy(id));
}

// Function 30
float calcParticleThicknessH(float depth){
   	
    depth = depth * 2.0 + 0.1;
    depth = max(depth + 0.01, 0.01);
    depth = 1.0 / depth;
    
	return 100000.0 * depth;   
}

// Function 31
void physics(inout vec4 col, vec2 coord)
{
    col   =texelFetch(iChannel0,ivec2(coord),0);
    vec3 pos   =texelFetch(iChannel3,ivec2(0,0),0).xyz;
    vec3 vel   =texelFetch(iChannel3,ivec2(1,0),0).xyz;
    vec4 quat  =texelFetch(iChannel3,ivec2(2,0),0);
    vec3 angVel=texelFetch(iChannel3,ivec2(3,0),0).xyz;
    
    vec3 acc=vec3(0);
    
    float floorConst=100000.; //N/m
    float m=.001; //kg
    float I1 = .25*m*.01*.01;
    float I2 = .5*m*.01*.01;
    mat3 Ii=mat3(1./I1,0,0, 0,1./I2,0, 0,0,1./I1);
    
    int NUM=5;
    for(int i=0;i<3;i++)
    {
    float GRAVITY = 1.0;
    acc+=vec3(0,0,-GRAVITY);
    vec3 n = transformVecByQuat(vec3(0,1,0),quat);
    if(acos(abs(n.z))*360./PI2<.2) break; // dont simulate phys if almost silent
    vec3 pfloor = -(normalize(cross(n,vec3(n.yx*vec2(-1,1),0)))+clamp(.5*n.z,-.1,.1)*n)*.01+pos;
    vec3 f=vec3(0), M=vec3(0);
    f+=max(-pfloor.z,0.)*vec3(0,0,1)*floorConst;
    f=m*GRAVITY*vec3(0,0,1);
    M+=cross(pfloor-pos,f);
    M=transformVecByQuat(M,inverseQuat(quat));
    
    acc+=f/m;
    vec3 angAcc=Ii*M;

    float dt=.01666/float(NUM);
    vel+=acc*dt*.5;
    pos+=vel*dt;
    vel+=acc*dt*.5;
    angVel+=angAcc*dt;
    //vec4 dq=angVec2Quat(transformVecByQuat(-angVel*dt,quat));
    vec4 dq=angVec2Quat(angVel*dt);
    quat=normalize(multQuat(quat,dq));
    //angVel+=angAcc*dt*.5;
    
    vel*=.99;
    angVel*=1.-.001-length((pfloor-pos).xy)*.5;
    pos.z=-(pfloor.z-pos.z);
    }
    
    if(iTime<5. || abs(length(texelFetch(iChannel3,ivec2(2,0),0))-1.)>.001)
    {
        pos=vec3(0,0,.01);
        vel=vec3(0,0,0);
        vec3 r=texelFetch(iChannel1,ivec2(mod(iDate.w,256.),mod(iDate.w/256.,256.)),0).xyz*2.-1.;
        //float r3=texelFetch(iChannel1,ivec2(mod(iDate.w,256.),1),0).x;
        quat = normalize(vec4(-.0*r.x,.0,0,1));
        angVel = vec3(0,r.z*.08,sign(r.x))*6.*6.*(1.-.4*r.y);
    }
    //quat=multQuat(quat,normalize(vec4(0.01,0,0,1)));
    
    if(int(coord.x)==0) col.xyz=pos;
    if(int(coord.x)==1) col.xyz=vel;
    if(int(coord.x)==2) col=quat;
    if(int(coord.x)==3) col.xyz=angVel;
}

// Function 32
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

// Function 33
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

// Function 34
vec2 getParticlePosition(int partnr)
{  
   vec2 pos = vec2(mod(float(partnr+1), iResolution.x)/(iResolution.x+1.), (float(partnr)/(iResolution.x))/(iResolution.y+1.));
   return (texture(iChannel0, pos)).xy;
}

// Function 35
float particleDistance(int i)
{
    return distance(p, sc(texel(ch0, i2xy(ivec3(i, 0, 0))).xy));
}

// Function 36
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

// Function 37
void DoPhysicsInit(inout Particle part, ivec2 i, float dt, float time, int frame, sampler2D ch)
{
    if (frame == 0) {
        // by setting old and new to same position,
        // velocity is initialized to zero.
		part.pold = part.pnew =
            vec3(vec2(i)*llen + vec2(llen*1.2, poleheight * hoist), 0);
    } else {
	    vec3 wind = globalwind;
	 	wind += windvar * vec3(1,.01,.02) * sin(vec3(3,5,2)/6.*time);
		wind *= mix(1., sin(time*.4), windgust); // overall strength waxes and wanes with time, in gusts
    	DoPhysics(part, i, dt, wind, ch);
    }
}

// Function 38
vec3 readParticlePos( int idx )
{
    return getPixel(XC(idx),YC(idx)).xyz;
}

// Function 39
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

// Function 40
particle getParticle(vec4 data, vec2 pos)
{
    particle P; 
    P.X = decode(data.x) + pos;
    P.V = decode(data.y);
    P.M = data.zw;
    return P;
}

// Function 41
vec3 particle(vec2 st, vec2 p, float r, vec3 col){
 	float d = length(st-p);
    d = smoothstep(r, r-2.0/iResolution.y, d);//d<r?1.0:0.0;
    return d*col;
}

// Function 42
vec2 getParticlePosition_mp()
{
   vec2 ppos = vec2(harms(main_x_freq, main_x_amp, main_x_phase, time2), harms(main_y_freq, main_y_amp, main_y_phase, time2)) + middlepoint;
   return gen_scale*ppos;
}

// Function 43
vec3 getParticleColor(int partnr, float pint)
{
   float hue;
   float saturation;

   saturation = mix(part_min_saturation, part_max_saturation, random(float(partnr*6 + 44) + runnr*3.3))*0.45/pint;
   hue = mix(part_min_hue, part_max_hue, random(float(partnr + 124) + runnr*1.5)) + hue_time_factor*time2;
    
   return hsv2rgb(vec3(hue, saturation, pint));
}

// Function 44
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

// Function 45
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

// Function 46
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

// Function 47
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

// Function 48
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

// Function 49
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

// Function 50
float calcParticleThicknessConst(const float depth){
    
	return 100000.0 / max(depth * 2.0 - 0.01, 0.01);   
}

// Function 51
vec4 genParticle(float id){
	vec2 pos = hash(vec2(iFrame, id)).xy * vec2(ASPECT, 1.) * GRID;
    vec4 hp = hexCoord(pos);
    float a = floor((hp.x + HALF_SEG)/SEGMENT);
    vec2 posOnEdge = pos + vec2(-1., 0.) * rotate(PI * a/3.) * hp.y;
    
    //Here we additionally calculate middle point of edge.
    //Can be solved analitically
    vec4 chp = hexCoord(hp.zw);
    vec2 ec = hp.zw + vec2(-1., 0.) * rotate(PI * a/3.) * chp.y;
    float h = sign(hash21(vec2(iTime, id * 2.71)) - .5);
    vec2 mDir = normalize(ec - hp.zw) * rotate(PI/2. * h);
    
    return vec4(posOnEdge, mDir);
}

// Function 52
void simulate(float g, int iteration) {
	for(int i = 0; i < WIDTH * HEIGHT; i++)
		world1[i] = (hash(float(i) * (g + 2.951)) < 0.3 + hash(g + 3.817) * 0.2) ? 1 : 0;
	
	for(int steps = 0; steps < 4; steps++)
	{
		if (steps <= iteration)
		{
			if (mod(float(steps), 2.0) == 0.0)
			{
				generateWorld2();
			}
			else
			{
				generateWorld1();
			}
		}
	}
}

// Function 53
void DoPhysics(inout Particle part, ivec2 i, float dt, vec3 wind, sampler2D ch)
{    
        vec3 pn = part.pnew, po = pn; // original new pos -> new old pos
    	vec3 v = velocity(part, dt);
		v *= exp2(-drag * dt); // drag forces
        //v.y -= gravity * .5 * dt; // 'gravity'
    	po.y += gravity * dt * dt;
		pn += v * dt;
        for (int j = 3; j-- > 0; ) // multiple iterations helps a bit probably? idk anymore w 4 buffers
			CheckLink(pn, po, i.x, i.y, dt, wind, ch); // constraint links
        part.pnew = pn;
        part.pold = po;
}

// Function 54
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

// Function 55
void readParticle(inout Particle p, vec2 coord, sampler2D s)
{
    vec4 pix=getPixel(coord,s);
    p.pos=pix.xy;
    p.vel=pix.zw;
    p.idx=particleIdx(coord,s);
}

// Function 56
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

// Function 57
vec4 moveParticle(inout vec4 pos, float id){
	vec4 hp = hexCoord(pos.xy);
    float a = mod((hp.x - HALF_SEG), SEGMENT);
    float cDst = distance(a, SEGMENT * .5);
    if(cDst > SEGMENT * .495){
    	float h = sign(hash21(vec2(iTime, id * .71)) - .5);
    	pos.zw *= rotate(SEGMENT * h);
        pos.xy += pos.zw * .03;
    }else{
    	pos.xy += pos.zw * (.01 + .02 * hash21(id * vec2(13., .23)));
    }
    
    if(!isInside(pos.xy))
        pos = genParticle(id);
    return pos;
}

// Function 58
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

// Function 59
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

// Function 60
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

// Function 61
vec4 physics(vec4 O) {       // --- simple Newton step
    O.zw += F(O) * dt;       // velocity
    O.xy += O.zw * dt;       // location
    return O;
}

// Function 62
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

// Function 63
vec3 getParticleColor(int partnr, float pint)
{
   vec2 pos = vec2(mod(float(partnr+1), iResolution.x)/(iResolution.x+1.), (50. + float(partnr)/(iResolution.x))/(iResolution.y+1.));
   return (pint*texture(iChannel0, pos)).xyz;  
}

// Function 64
vec4 getParticle(in vec3 iResolution, in sampler2D iChannel0, in int frame, int index) {
    ivec2 uv = deserializeUV(iResolution, frame, index);
    //getRandomParticlePos(iResolution, iChannel0, fragCoord, frame, i)
	return texelFetch(iChannel0, uv, 0);
}

// Function 65
void DoPhysicsInit(inout Particle part, ivec2 i, float dt, float time, int frame, sampler2D ch)
{
    if (frame == 0) {
        // by setting old and new to same position,
        // velocity is initialized to zero.
		part.pold = part.pnew =
            vec3(vec2(i)*llen + vec2(llen*1.2, poleheight * hoist), 0);
    } else {
	    vec3 wind = globalwind;
	 	wind += windvar * vec3(1,.01,.02) * sin(vec3(3,5,2)/6.*time);
		wind *= mix(1., sin(time*.4), windgust); // overall strength waxes and wanes with time, in gusts
		// wind is a velocity
    	DoPhysics(part, i, dt, wind, ch);
    }
}

// Function 66
particle getParticle(float index, sampler2D ch, float iTime, vec4 iMouse, vec3 iResolution){
    
    vec3 p = .03*texture(ch,(.5+vec2(mod(index*3.,R.x),floor(index*3./R.x)))/R.xy).xyz;
    p = p.yzx;
    float r = 5.3;
    vec3 proj = ((p-cam)*view);
    return particle(proj.xy*R.y+R.xy/2.,proj.z,r,vec3(1));
    
}

// Function 67
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

// Function 68
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

// Function 69
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

// Function 70
Particle GrowAlongParticle(int index, int parentIdx)
{
    return GrowParticle(index, parentIdx, 0.13);
}

// Function 71
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

// Function 72
int particleIdx(vec2 coord, sampler2D s)
{
    ivec2 ires=textureSize(s,0);
    return int(coord.x)+int(coord.y)*ires.x;
}

// Function 73
float getParticleStartTime(int partnr)
{
    return start_time*random(float(partnr*2));
}

// Function 74
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

// Function 75
vec3 particleColor (in float particleVelocity) {
	return mix (vec3 (0.5, 0.5, 1.0), vec3 (1.0), particleVelocity * VELOCITY_COLOR_FACTOR);
}

// Function 76
vec3 getParticleColor_mp( float pint)
{
   float hue;
   float saturation;
   
   saturation = 0.75/pow(pint, 2.5) + mp_saturation;
   hue = hue_time_factor*time2 + mp_hue;

   return hsv2rgb(vec3(hue, saturation, pint));
}

// Function 77
vec2 particleSpawnPos(int frame, int particleIdx)
{
    float seed = float(frame) + float(particleIdx) / float(NUM_PARTICLES);
    vec2 pos;
    pos.x = rand(seed);
    pos.y = rand(pos.x + seed);
    return floor(pos * iResolution.xy + vec2(0.5));
}

// Function 78
float particleFromUVAndPoint(in vec2 uv, in vec2 point, in vec2 rootUV, in float pixelSize)
{
	float dist = distance(uv, point);
#ifdef RANDOMIZED_SIZE
    dist += (hash1_2(rootUV * 10.0) - 0.5) * PARTICLE_SIZE_VARIATION;
#endif
    float particle = 1.0 - smoothstep(PARTICLE_RADIUS - dist * 0.05, PARTICLE_RADIUS2 - dist * 0.05 + pixelSize, dist);
    return particle * particle;
}

// Function 79
vec4 saveParticle(particle P, vec2 pos)
{
    vec2 x = clamp(P.X - pos, vec2(-0.5), vec2(0.5));
    return vec4(encode(x), P.M, P.V);
}

// Function 80
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

// Function 81
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

// Function 82
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

// Function 83
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

// Function 84
vec3 calcLightOnParticle( vec3 particlePos, Sphere lightSphere, vec3 Li ) {
    vec3 wi;
    return calcDirectLight( particlePos, wi, lightSphere, Li );
}

// Function 85
vec4 saveParticle(particle P, vec2 pos)
{
    P.X = clamp(P.X - pos, vec2(-0.5), vec2(0.5));
    return vec4(encode(P.X), encode(P.V), P.M);
}

// Function 86
int particleIdx(vec2 coord)
{
    ivec2 res=textureSize(ParticleTex,0);
    return (int(coord.y)*res.x+int(coord.x))/3;
}

// Function 87
vec4 getParticle(int id){
    return texelFetch(iChannel0, locFromID(id), 0);
}

// Function 88
vec2 particleCoord(int idx, sampler2D s)
{
    ivec2 ires=textureSize(s,0);
    return vec2(idx%ires.x,idx/ires.x)+.5;
}

// Function 89
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

// Function 90
void initParticle(inout Particle p, sampler2D s, sampler2D sr, int frame)
{
    vec2 res=vec2(textureSize(s,0));
    //p.pos = vec2((p.idx/2)%NUM_X,(p.idx/2)/NUM_X)*res/vec2(NUM_X,NUM_Y);
    p.pos=getRand(frame+p.idx,sr).xy*res.xy;
    p.vel = (getRand(p.pos,sr).xy-.5)*(float(p.idx%2)-.5)*300.;
}

// Function 91
void initParticle(inout Particle p, int xIdx, int yIdx)
{
    float xnum = float(XNUM);
    float ynum = float(YNUM);
    vec2 delta = vec2(Res.x/xnum,Res.y/ynum);
    p.pos.xy=vec2(xIdx,yIdx)*delta;
    // trigonal grid offs
    p.pos.x+=(float(yIdx%2)-.5)*delta.x*.5;
    p.vel = (getRand(p.pos).xy-.5)*0.;
}

// Function 92
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

// Function 93
vec4 getRandomParticle2(in vec3 iResolution, in sampler2D iChannel0, in vec2 fragCoord, in int frame, int i) {
	return texelFetch(iChannel0, getRandomParticlePos(iResolution, iChannel0, fragCoord, frame, i), 0);
}

// Function 94
float bokehParticles(vec2 uv, float radius, vec2 dotProportions, float dotRotation, float animationOffset)
{
 	float voro = voronoi(uv, dotProportions, dotRotation, animationOffset);
    float particles = 1.0 - smoothstep(radius, radius * (2.0), voro);
    return particles;
}

// Function 95
vec3 particleColor (in float particleVelocity) {
	return mix (vec3 (0.5, 0.6, 0.8), vec3 (0.9, 0.9, 1.0), particleVelocity * PARTICLE_VELOCITY_FACTOR);
}

// Function 96
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

// Function 97
float SDF_particle(vec3 p0, vec3 p)
{
    particle point = get(fakech0, p0);
    return length(point.pos.xyz - p) - sphere_rad;
}

// Function 98
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

// Function 99
vec3 getParticleColor(in vec2 p) {
    return normalize(vec3(0.1) + texture(iChannel2, p * 0.0001 + iTime * 0.005).rgb);
}

// Function 100
void writeParticle( int idx, Particle p, inout vec4 fragColor, vec2 fragCoord)
{
    int xc = XC(idx);
    int yc = YC(idx);
    if (isPixel(xc+0,yc,fragCoord)) fragColor.xyzw=p.pos;
    if (isPixel(xc+1,yc,fragCoord)) fragColor.xyzw=p.quat;
    // not sure if framebuffer has .w, so store quat.w also in next pixel
    if (isPixel(xc+2,yc,fragCoord)) fragColor.xyzw=vec4(p.size,p.parent,p.quat.w,1);
}

// Function 101
vec4 saveParticle(particle P, vec2 pos)
{
    P.X = clamp(P.X - pos, vec2(-0.5), vec2(0.5));
    return vec4(encode(P.X), encode(P.V), P.M, P.I);
}

// Function 102
Particle GrowSplitParticle(int index, int parentIdx)
{
    return GrowParticle(index, parentIdx, 0.3);
}

// Function 103
void DoPhysics(inout Particle part, ivec2 i, float dt, vec3 wind, sampler2D ch)
{    
	vec3 pn = part.pnew, po = pn; // original new pos -> new old pos
	vec3 v = velocity(part, dt);
	v *= exp2(-drag * dt); // drag forces
	//v.y -= gravity * .5 * dt; // 'gravity'
	po.y += gravity * dt * dt;
	pn += v * dt;
    for (int j = 2; j-- > 0; ) // multiple iterations helps a bit probably? idk anymore w 4 buffers
		CheckLink(pn, po, i.x, i.y, dt, wind, ch); // constraint links
	part.pnew = pn;
	part.pold = po;
}

// Function 104
vec2 particleCoordFromRootUV(vec2 rootUV){
    return rotate(vec2(0.0, 1.0), globalTime * 3.0 * (hash12(rootUV) - 0.5)) * (0.5 - PARTICLE_SIZE) + rootUV + 0.5;
}

// Function 105
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
            
            rez.rgb += d*abs(sin(vec3(2.,3.4,1.2)*(iTime*.06 + float(i)*.003 + 2.) + vec3(0.8,0.,1.2))*0.7+0.3)*0.04;
            pos.xyz += vel*0.002*0.2;
        }
    }
    rez /= float(stepsPerFrame);
    
    return rez;
}

// Function 106
float calcParticleThickness(float depth){
   	
    depth = depth * 2.0;
    depth = max(depth + 0.01, 0.01);
    depth = 1.0 / depth;
    
	return 100000.0 * depth;   
}

// Function 107
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

// Function 108
float distance2Particle(int id, vec2 fragCoord){
    if(id==-1) return 1e20;
    vec2 delta = getParticle(id).xy-fragCoord;
    return dot(delta, delta);
}

// Function 109
int getParticle(float id, vec3 R, int iFrame){
    int seed = IHash(int(id))^IHash(int(iFrame));
    return seed;
}

// Function 110
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

// Function 111
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

// Function 112
particle getParticle(vec4 data, vec2 pos)
{
    particle P;
    if (data == vec4(0)) return P;
    P.X = decode(data.x) + pos;
    P.M = data.y;
    P.V = data.zw;
    return P;
}

// Function 113
vec4 physics(vec2 p, int physicsStep, float radius)
{
	vec4 c = vec4(0.0);
	
	int stepNumber = int(mod(float(physicsStep), float(SIMULATION_FRAMES)));
	float frameSeed = float(physicsStep/SIMULATION_FRAMES);
	vec2 startVelocity = hash(vec2(frameSeed, frameSeed));
	
	vec2 ballOrigin  = vec2(0.5, 0.5);
	vec2 ballVelocity = (startVelocity - 0.5) * START_FORCE;
	int count = 0;
	for (int count = 0; count < SIMULATION_FRAMES; count++)
	{
		if (distance(ballOrigin, p) < radius)
		{
			c += TRAIL_RADIUS_MULT * vec4(1.0 - distance(ballOrigin, p) / radius);
		}
		if (count > stepNumber)
		{
			break;
		}
		
		if (ballOrigin.y - radius < 0.0)
		{
			ballOrigin.y = radius;
			ballVelocity.y = -ballVelocity.y * RESTITUTION_Y;
		}
		else if (ballOrigin.y + radius > 1.0)
		{
			ballOrigin.y = 1.0 - radius;
			ballVelocity.y = -ballVelocity.y * RESTITUTION_Y;
		}
		
		if (ballOrigin.x - radius < 0.0)
		{
			ballOrigin.x = radius;
			ballVelocity.x = -ballVelocity.x * RESTITUTION_X;
		}
		else if (ballOrigin.x + radius > 1.0)
		{
			ballOrigin.x = 1.0 - radius;
			ballVelocity.x = -ballVelocity.x * RESTITUTION_X;
		}
		
		ballOrigin += ballVelocity;
		ballVelocity += vec2(GRAVITY_X, GRAVITY_Y);
		
	}
	if (distance(ballOrigin, p) < radius)
	{
		c = vec4(1.0, 0.0, 0.0, 1.0);
	}
	return c;
}

// Function 114
float particleFromParticleUV(vec2 particleUV, vec2 uv)
{
 	return 1.0 - smoothstep(0.0, PARTICLE_SIZE, length(particleUV - uv));   
}

// Function 115
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

// Function 116
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

// Function 117
float SDF_particle(vec3 p0, vec3 p)
{
    particle point = get(fakech0, p0);
    return length(point.pos.xyz - p) - point.vel.w;
}

// Function 118
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

// Function 119
void readParticle(inout Particle p, int idx, sampler2D s)
{
    readParticle(p,particleCoord(idx,s),s);
}

// Function 120
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

// Function 121
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

// Function 122
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

// Function 123
vec3 drawParticle(in vec2 p, in float size, in vec3 col) {
  return mix( col, vec3(0.0)  , smoothstep(0., size, dot(p, p) * 90.0 ) );
}

// Function 124
void particles(vec3 p, inout float curDist, inout vec3 glowColor, inout int id) {
    float t;
    float angle;
    float radius;
    float dist = CAM_FAR;
    const float glowDist = 0.2;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        dist = length(p - ppos[i].xyz) - 0.005;
        if (dist < glowDist && false) {
            float d = dist + rand(dist) * 0.5;
            glowColor += clamp(1.0 - d / glowDist, 0.0, 1.0) * 0.005;
        }
        if (dist < curDist) {
            curDist = dist;
            id = 2;
        }
    }
}

// Function 125
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

