// Reusable Animation Procedural Functions
// Automatically extracted from procedural-related shaders

// Function 1
float BallOscillation() {
	return sin(5. * CyclicTime() + 4.) * exp(-CyclicTime() / 6.) + 0.3;
}

// Function 2
Cam CAM_animate(vec2 uv, float fTime)
{
    Cam cam;
    cam.o = g_animationChannels.camPos;
    cam.D = normalize(g_animationChannels.camLookAt-cam.o);
	cam.R = normalize(cross(cam.D,vec3(0,1,0)));
    cam.U = normalize(cross(cam.R,cam.D));
    cam.lens = 1.2+0.3*sin(fTime*0.1);
    cam.zoom = 3.0+sin(fTime*0.1)/cam.lens;
	return cam;
}

// Function 3
bool resetTime(){
    bool loaded=iTime>.5&&texture(iChannel2,vec2(0.)).xy!=vec2(0.)&&texture(iChannel3,vec2(0.)).xy!=vec2(0.);
	bool noHYet=noHYet();
bool result= ((texelFetch( iChannel1, ivec2(RCODE,0),0).x>0.)||(noHYet&&loaded));
return result;
}

// Function 4
vec3 quat_times_vec(vec4 q, vec3 v)
{
	//http://molecularmusings.wordpress.com/2013/05/24/a-faster-quaternion-vector-multiplication/
	vec3 t = 2. * cross(q.xyz, v);
	return v + q.w * t + cross(q.xyz, t);
}

// Function 5
float animateEntranceSith(float p, inout StickmanData data)
{    
    data.saberLen *= smoothstep(0.05, 0.15, p);
    float pose1 = 1.0 - smoothstep(.52, .6, p);
    poseSaberBackDown(pose1, data);
    
    backLoop(max(smoothstep(.2, .25, p) - smoothstep(0.55, 0.6, p), 0.00001), linearstep(.2, .55, p)*3., data);
    
    float pose2 = smoothstep(.5, .6, p);
    poseSaberBack(pose2, data); 
    return 0.0;
}

// Function 6
float AnimateDensity()
{
    float i = floor(iTime / ANIMATE_DURATION);
    float r = (iTime - ANIMATE_DURATION * i) / ANIMATE_DURATION;
    float sinr = pow(sin(HFPI * r), 2.0);
    float k = ( mod(i, 2.0) == 0.0 ? sinr : 1.0 - sinr );
 	return max(k*DENSITY, 5.0);
}

// Function 7
float escape_time(vec2 point) {
    
    vec2 z = point;
    for (float i = 0.0; i < MAX_ITERS; i++) {
        z = inverse_sierpinski(z);
        
        
        if (length(z) > RADIUS)
            return i;
    }
    return 0.0;
}

// Function 8
float GetTime()
{
	return 0.0;
}

// Function 9
float CyclicTime()
{
	return mod(iTime, 30.);
}

// Function 10
float getTime(float time)
{
    //time2 = camspeed*(iTime + 158.);
    float time2 = camspeed*time;
    
    #ifdef keys
    // When pressing numeric keys, you can go back a different distance
    if (isKeyPressed(KEY_1)) time2-= 2.;
    if (isKeyPressed(KEY_2)) time2-= 5.;
    if (isKeyPressed(KEY_3)) time2-= 10.;
    if (isKeyPressed(KEY_4)) time2-= 20.;
    if (isKeyPressed(KEY_5)) time2-= 50.;
    if (isKeyPressed(KEY_6)) time2-= 100.;
    if (isKeyPressed(KEY_7)) time2-= 200.;
    if (isKeyPressed(KEY_8)) time2-= 500.;
    if (isKeyPressed(KEY_9)) time2-= 1000.;
    #endif
    
    #ifdef varspeed
    time2-= 58.*sin(time/9.) + 25.*cos(time/17.) - 12.*cos(time/5.7);
    #endif
    
    return time2;
}

// Function 11
bool resetTime(){
 return (texelFetch( iChannel1, ivec2(RCODE,0),0).x>0.)||iFrame<100;
}

// Function 12
vec2 Oscillator(float Fo, float Fs, float N)
{
    float phase = (tau*Fo*floor(N))/Fs;
    return vec2(cos(phase),sin(phase));
}

// Function 13
float Mtime(float mval)
{
    return mod(iTime,mval);
}

// Function 14
float TimerInOut(vec4 v)
{
    return smoothstep(v.y,v.y+v.w,v.x) - smoothstep(v.z-v.w,v.z,v.x);
}

// Function 15
vec4 timeEyes(vec2 uv, float time, float dx, vec3 eyeColor, int phase) {
    vec4 color = vec4(0.);
    vec2 euv = vec2(mirror(uv.x), uv.y + time*.1);
    const float eRad = .1;
    const vec2 eCenter = vec2(.25, .99);
    float d = sdCircle(euv - eCenter, eRad);
    vec2 browP = eCenter + eRad*1.*nop.yz;
    vec2 browD = normalize(vec2(1.5, -1.));
    d = sdUnion(d, sdFatLine2(euv, vec3(eCenter + eRad*nop.zz*.3, eRad*.5), vec3(browP+browD*2.*eRad, 0.)));
    d = sdSubtract(d, dot(euv - browP, rot90(browD)));
    vec2 eyeSpace = rot(.2)*(euv - eCenter);
    vec2 pupilSpace = vec2(length(eyeSpace*vec2(2.,1.))-.02*time, d2a(eyeSpace));
    float dPupil = pupilSpace.x - .3*eRad;
    vec4 ec = texture(iChannel0, pupilSpace*vec2(.3,.1)*2.);
    ec.rgb = mix(ec.rgb, vec3(.5), .75);
    ec.rgb *= S(-dx, 0., dPupil);
    
    // highlight
    vec2 worldEyeSpace = eyeSpace;
    if (uv.x > .5) worldEyeSpace.x =  - worldEyeSpace.x;
    vec3 n = normalize(vec3(worldEyeSpace, 1.-sqrt(length(worldEyeSpace/eRad))));
    float hl = .5*pow(abs(dot(n, normalize(vec3(.25, .3*time, 1.)))), 30.);
    hl += .5*pow(abs(dot(n, normalize(vec3(-.25, -.5+.3*time, 1.)))), 30.);
    hl = sqr(hl);
    
    if (d < dx) {
        float edge = S(0., -dx, d);
        color = comp(color, premult(ec.rrr*eyeColor + hl*white, edge));
        if (phase == 2) color = comp(color, premult(red*.7, S(-.015-dx, -.015, d)*edge));
    }
    
    //color.rgb += .2; // show bounds
        
    return color;
}

// Function 16
float tickTime(float t){ return t*2. + tick(t, 4.)*.75; }

// Function 17
vec2 displayTimeWithWave(in vec2 uv, in vec2 id,in float frac)
{    
    id.x -= 10.;
    id.y += 4.;
    vec2 rotation = vec2(0,0);
	vec2 nextRotation = vec2(0,0);
    float time = iDate.w - 1.;
    float nextTime = time + 1.;
    
    float check = 0.;
    
    //digits
    for(int i =0; i < 3; i++){
        for(int j = 0; j < 2; j++){
            check = when_gt(id.x, -1.0) * when_lt(id.x, 4.)* when_gt(id.y, -1.0) * when_lt(id.y,8.);
            
            rotation += getRotation(int(id.x), int(id.y), getNumber(int(mod(time, 60.)),j)) 
                * check;

            nextRotation += getRotation(int(id.x), int(id.y), getNumber(int(mod(nextTime, 60.)), j))
                * check;

            id.x += 4.;
        }
        id.x += 2.;
        time = floor(time / 60.);
        nextTime = floor(nextTime / 60.);
    }
    
    //colons
    id.x -=13.;
    for(int i = 0; i < 2; i++) {
        check = when_gt(id.x, 0.0) * when_lt(id.x, 3.)* when_gt(id.y, 1.0) * when_lt(id.y,6.);
        rotation.x += (270. + 180. * id.x) * check;
        nextRotation.x += (270. + 180. * id.x) * check;
        rotation.y += (0. + 180. * id.y) * check;
        nextRotation.y += (0. + 180. * id.y) * check;
        id.x -=10.;
    }    
    
    //reset id for animation
   	id = floor(uv);   
    
    //lerp between current time and next time(time+1)
    float clockLerp = clamp(mod(iDate.w * 2.,2.),0.,1.);
    float h = mix(rotation.x, nextRotation.x, clockLerp);
    float m = mix(rotation.y, nextRotation.y, clockLerp);
    
    //animate the non clock part
    float animLerp = mod(id.x * .035 + id.y * .035 + frac,2.);
    h += (90. + id.x * 0. - animLerp * 360. ) * (1. - clamp(rotation.x,0.,1.));
    m += (270. + id.y * 0. + animLerp * 360. ) * (1. - clamp(rotation.y,0.,1.));
    
    float radianHour = radians(mod(h,360.));
    float radianMinute = radians(mod(m,360.));
    
    return vec2(radianHour,radianMinute);
}

// Function 18
float getTime()
{
    return texture(iChannel2, vec2(3.5, 0.5) / iResolution.xy).x;
}

// Function 19
mat3 rotationOverTime( ivec2 c )
{
	vec2 co = vec2( c - ivec2( c_iGlassWidth / 2, c_iGlassHeight / 2 ) );
	co += vec2( 0.5, 0.5 );
	vec3 axis = vec3( co.y, -co.x, 0 );
	axis = normalize( axis );
	float fSpeed = max( 4.0 - length( vec2( co ) ), 0.0 );
	fSpeed = pow( fSpeed, 5.0 );
//	fSpeed = 1.0;
	return matAxisAngle( axis, fSpeed * g_fGlassCrashTime );
}

// Function 20
float rand2sTime(vec2 co){
    co *= time;
    return fract(sin(dot(co.xy,vec2(12.9898,78.233))) * 43758.5453);
}

// Function 21
float timeOscillation() {
	return .5*(1. + sin(1.1 * iTime));
}

// Function 22
void SetTime(float t){ProcessLightValue(t);ProcessObjectPos(t);}

// Function 23
void DTimeSet (vec4 d)
{
  idt[0] = DIG2 (floor (d.x / 100.));
  idt[1] = DIG2 (mod (d.x, 100.));
  idt[2] = DIG2 (d.z);
  idt[3] = DIG2 (floor (d.w / 3600.));
  idt[4] = DIG2 (floor (mod (d.w, 3600.) / 60.));
  idt[5] = DIG2 (floor (mod (d.w, 60.)));
  inm[0] = MName (int (d.y));
  inm[1] = DName (DWk (d.xyz));
}

// Function 24
float GetTime()
{
	float fTime = iChannelTime[3] / 8.0;
	#ifdef OVERRIDE_TIME
	fTime = iMouse.x * fSequenceLength / iResolution.x;
	#endif
	
	// hack the preview image
	if(iTime == 10.0)
	{
		fTime = 30.0 / 8.0;
	}
	
	return mod(fTime, fSequenceLength);
}

// Function 25
float glowTime()
{
    return max(0.0, animationTime() - 3.35) / 7.65;
}

// Function 26
vec3 timefly(float t) {
    // main path Called from many places
    t*=.80;
	t += (.125 + sin(t * .125));
	vec3 v =
	vec3(sin(t / 50.) * 20., 0., cos(t / 25.) * 24.) +
		vec3(sin(t / 17.1) * 07., 0., cos(t / 17.1) * 05.) +
		vec3(sin(t / 8.1) * 6., 0., cos(t / 8.1) * 8.) +
		vec3(cos(t / 3.) * 3.,0., sin(t / 3.) * 2.)
        +vec3(cos(t  )*2.,0., sin(t  )*2. );
    v.y=pathterrain(v.x,v.z);
    return v        ;
}

// Function 27
vec4 updateTime(in vec4 time)
{
    if (iFrame == 0)
        time = vec4(6.0, 0.0, 0.0, 0.0);
    
    if (isPressed(KEY_G))
        time.x += iTimeDelta;
    
    return time;
}

// Function 28
vec2 timesc(vec2 a, vec2 b){
    float arg1 = arg(a);
    float arg2 = arg(b);
    return vec2(cos(arg1+arg2),sin(arg1+arg2))*length(a)*length(b);
}

// Function 29
void animate_cam( in float t, in vec2 uv, out vec3 cp, out vec3 cd, out float f )
{
    // Get a new offset every 20 seconds.
    vec3 offset = vec3(7.0, 2.0, 0.0) + vec3(20.0)*floor(t*.05);
    
    // Fade in and out every 10 seconds.
    f = shutterfade(0.0, 10.0, mod(t,10.0), .5);
    
    // Traverse along a path, resetting every 20 seconds.
    cp = offset + vec3(2.0*mod(t,20.0), 0.0, 2.0*mod(t,20.0));
    
    // For the first 10 seconds we look up slightly, for the second 10 we
    // gander downwards a bit.
    if( mod(t,20.0)<10.0 ) cd = CAM_DIR;
    else cd = CAM_DIR*vec3(-1.0, 1.0, 1.0);
    
    camera(uv, cp, cd, 1.0, cp, cd);
}

// Function 30
vec2 timeRotation(float t)
{
    t = 0.2 * t;
    return vec2(-t, 0.5 * PI * (-0.25*sin(t + PI/2.)));
}

// Function 31
float TestInstrument4Times(vec4 freq, vec4 time){
 ;float c=0.
 ;for(int i=0;i<5;i++
 ){
  ;c+=instrumentBanjo(freq[i],time[i])
  ;};return c;}

// Function 32
float outroTime() {
    return clamp(1.0 - (loopTime() - (TOTAL_TIME - INTERMISSION - OUTRO_TIME)) / OUTRO_TIME, 0.0, 1.0);
}

// Function 33
vec2 animateCell1(vec2 noise)
{
 	noise = sin(iTime+MOV_FACTOR*noise);
    return 0.5*noise + 0.5; //NORMALIZE 
}

// Function 34
float loop_time(float u_time, float limit) {
  float mod_time = mod(u_time, limit);
  if (mod_time < limit / 2.0) {
    return mod_time;
  } else {
    return limit - mod_time;
  }
}

// Function 35
float oscillate(float t_low, float t_high, float t_transition, float t_offset) {
    float t_osc = 0.5*(t_high+t_low)+t_transition;
    float h_l = 0.5*t_low/t_osc;
    float h_h = (0.5*t_low+t_transition)/t_osc;
    return smoothstep(0., 1., (clamp(abs(mod(iTime + t_offset, t_osc*2.)/t_osc-1.), h_l, h_h) - h_l) / (h_h - h_l));
}

// Function 36
void animateGlobals()
{

	g_camRotationRates = vec2(0.001125 * g_beatRate, 
                               -0.00125 * g_beatRate);
    
	g_time = iChannelTime[1];
    
    // remap the mouse click ([-1, 1], [-1/ar, 1/ar])
    vec2 click = iMouse.xy / iResolution.xx;    
    click = 2.0 * click - 1.0;   
    
    // camera position
    g_camOrigin = vec3(0., 0.0 , 23.);
    
    float rotateXAngle    = .49 * PI * 
            sin(g_camRotationRates.x * g_time - .88 * PI * click.y);
    float cosRotateXAngle = cos(rotateXAngle);
    float sinRotateXAngle = sin(rotateXAngle);
    
    float rotateYAngle    = g_camRotationRates.y * g_time + 
            TWO_PI * click.x;
    float cosRotateYAngle = cos(rotateYAngle);
    float sinRotateYAngle = sin(rotateYAngle);

    // Rotate the camera around the origin
    g_camOrigin = rotateAroundXAxis(g_camOrigin, 
                                    cosRotateXAngle, 
                                    sinRotateXAngle);
    g_camOrigin = rotateAroundYAxis(g_camOrigin, 
                                    cosRotateYAngle, 
                                    sinRotateYAngle);

    g_camPointAt   = vec3(0., 0., 0.);
        
    // For each audio sample, sample the audio channel along the
    // x-axis (increasing frequency with each sample).
    for (int i = 0; i < NUM_AUDIO_SAMPLES; i += 1) {
        float offsets = float(i) * (0.95/float(NUM_AUDIO_SAMPLES));
        g_audioFreqs[i] = smoothstep(0.4, .7, 
                             texture( iChannel1, 
                             vec2(0.0 + offsets, 0.0)).r);
    }

#if !HEAT_PLUMES_ONLY
    // slightly modify the direction of the wave based on noise to add 
    // effect for the vertical heat wave event.
    g_heatWaveDir = vec3(0., 1., 0.);
    g_heatWaveDir += .3 * vec3(cos(.8 * g_time), 0., sin(.8 * g_time));
                         
#endif

    // calculate the plume origin that changes over time to add
    // interesting detail.
    float cosElev = cos(g_time * .2);
    float sinElev = sin(g_time * .2);

    float cosAzim = cos(g_time * .5);
    float sinAzim = sin(g_time * .5);
    
    g_plumeOrigin = NUM_CELLBOXES/2. * 
                       vec3(cosAzim * cosElev,
                            sinElev,
                            sinAzim * cosElev);
}

// Function 37
float cameraIntroTime()
{
    float t = smoothstep(2.0, 4.0, animationTime() + sin(iTime * 24.0) * .025);
    t *= smoothstep(4.5, 4.0, animationTime());
    return t;
}

// Function 38
vec4 AnimateFish(int id)
{
  vec2 md = vec2(0);
  vec2 vel = vec2(0);
  vec2 acc = vec2(0); 
  vec2 ratio = iResolution.xy / iResolution.y;
  float dt = .03; 
    
  vec4 fish = GetFish(VALUE_BUFFER, id);
        
  // sum forces -----------------------------  
        
  // borders action
  vec2 sumF = vec2(1,1) / abs(fish.xy) - (1.0+0.5*sin(iTime)) / abs(ratio - fish.xy);         

  if (mousePressed)  
  {
    vec2 mpos = iMouse.xy / iResolution.y;         //  0.0 .. 1.0  
    md = fish.xy - mpos;
    sumF += normalize(md) * FLEE_DISTANCE / dot(md,md);
  }
      
  // Calculate repulsion force with other fishes
  float swarmRadius = 3.4 - float(fishCount) / 222.;  
  for (int ni=0; ni < MAX_FISHES; ni++)
  if (ni != id) 
  {
    if (ni >= fishCount) break;      

    vec4 aFish = GetFish(VALUE_BUFFER, ni);   
    
    md = fish.xy - aFish.xy;
    float dist = length(md);
    sumF -= dist > 0.0 
            ? md*(swarmRadius+log(dist*dist)) / exp(dist*dist*2.4) / dist
            : 0.01*hash(float(id));   // if same pos : small ramdom force

  }
  // friction    
  sumF -= fish.zw * RESIST / dt;
        
  // dynamic calculation ---------------------     
        
  // calculate acceleration A = (1/m * sumF) [cool m=1. here!]
  float a1 = length(acc = sumF); 
  acc *= a1 > MAX_ACCELER 
         ? MAX_ACCELER / a1 
         : 1.; // limit acceleration
    
  // calculate speed
  float v1 = length(vel = fish.zw + acc*dt);
  v1 = v1 > MAX_VELOCITY   ? MAX_VELOCITY / v1 : 1.; // limit velocity
  v1 = v1 < MIN_VELOCITY   ? MIN_VELOCITY / v1 : 1.; // limit velocity  
  vel *= v1;  
    
  // return position and velocity of fish (xy = position, zw = velocity) 
  return vec4(fish.xy + vel*dt, vel); 
}

// Function 39
float sineClampedTimescale(float t,float offset,float clampMult) {
    return clamp(t*1.5+offset,0.0,3.1415926*clampMult);
}

// Function 40
void space_time_bending(inout Ray r, inout vec3 p, float k)
{    

    vec3 m_vec = m.pos - p;
    float d = dot(m_vec,m_vec);
    vec3 res = normalize(m_vec) * (GRAV_CONST*m.mass)/(d);
        
    d = min(.92, d);
    r.dir = normalize(r.dir + k*res);
}

// Function 41
vec4 AnimateFish(int id)
{
  vec2 md = vec2(0);
  vec2 vel = vec2(0);
  vec2 acc = vec2(0); 
  vec2 ratio = iResolution.xy / iResolution.y;
  float dt = .03; 
    
  vec4 fish = GetFish(VALUE_BUFFER, id);
        
  // Sum Forces -----------------------------  
        
  // borders action
  vec2 sumF = (vec2(1.0,1.0) / abs(fish.xy) - (1.0+0.5*sin(iTime)) / abs(ratio - fish.xy));         

  if (mousePressed)  
  {
    vec2 mpos = iMouse.xy / iResolution.y;         //  0.0 .. 1.0  
    md = fish.xy - mpos;
    sumF += normalize(md) * FLEE_DISTANCE / dot(md,md);
  }
      
  // Calculate repulsion force with other fishs
  for (int ni=0; ni < MAX_FISHES; ni++)
  if (ni != id) 
  {
    if (ni >= fishCount) break;      

    vec4 aFish = GetFish(VALUE_BUFFER, ni);   
    
    md = fish.xy - aFish.xy;
    float dist = length(md);
    sumF -= dist > 0.0 
            ? md*(6.3+log(dist*dist*.02)) / exp(dist*dist*2.4) / dist
            : .01*hash(float(id)); // if same pos : small ramdom force

  }
  // friction    
  sumF -= fish.zw * RESIST / dt;
        
  // dynamic calculation ---------------------     
        
  // calculate acceleration A = (1/m * sumF) [cool m=1. here!]
  float a1 = length(acc = sumF); 
  acc *= a1 > MAX_ACCELER 
         ? MAX_ACCELER / a1 
         : 1.; // limit acceleration
    
  // calculate speed
  float v1 = length(vel = fish.zw + acc*dt);
  v1 = v1 > MAX_VELOCITY   ? MAX_VELOCITY / v1 : 1.; // limit velocity
  v1 = v1 < MIN_VELOCITY   ? MIN_VELOCITY / v1 : 1.; // limit velocity  
  vel *= v1;  
    
  // return position and velocity of fish (xy = position, zw = velocity) 
  return vec4(fish.xy + vel*dt, vel); 
}

// Function 42
vec4 Time( vec2 uv, float dx )
{
    const vec2 center = vec2(.5, .5);

    // BG 
    vec2 tuv = uv - center;
    float r = length(tuv);
    float a = d2a(tuv);//atan(tuv.x, tuv.y)+pi+iTime;    
    
    int phase = int(time) & 0x3;
    float phaseTime = fract(time);
    
    float phaseTimeEaseOut = sqrt(1.-sqr(1.-phaseTime));
    float phaseTimeEaseOutFast = sqrt(1.-sqr(1.-min(phaseTime/.5, 1.)));
        
    float t = sqrt(1. - sqr(1. - clamp(phaseTime/.7, 0., 1.)))*.5 + .5;

	vec4 color;
    vec4 gearColor;
    vec4 lgc;
    vec4 pendColor;
    switch(phase) {
        case 0: 
        	color = vec4(.3, .1, .1, 1.);
        	gearColor = lgc = vec4(mix(vec3(1., 1., .5), red, phaseTimeEaseOut), 1.); 
        	pendColor = vec4(.4, .1, .1, 1.);
        	lgc = mix(vec4(1., 1., .5, 1.), pendColor, phaseTimeEaseOut);
        	break;
        case 1: 
        	color = vec4(darkblue, 1.);
        	gearColor = lgc = vec4(mix(white, grey,  phaseTimeEaseOut), 1.);
        	pendColor = vec4(blue, 1.);
            lgc = vec4(mix(white, pendColor.rgb,  phaseTimeEaseOut), 1.);
            break;
        case 2: color = vec4(brown, 1.) ; 
        	gearColor = vec4(white, 1.) ; 
        	pendColor = vec4(.5, .5, 1., 1.);
        	lgc = vec4(mix(white, pendColor.rgb,  phaseTimeEaseOut), 1.);
        	break;
        case 3: color = vec4(blue*.2, 1.)    ; 
        	gearColor = vec4(grey, 1.);
        	pendColor = vec4(.3, .3, .3, 1.);
        	lgc = pendColor;
        	break;
    }
    
        
    // little gear color
    vec3 rad = vec3(.09, .12, .15);
    color = comp(color, gear(uv, dx,  phaseTimeEaseOut-.49, vec2(.0,  1.), rad, 4., vec2(.02,.015), lgc, false, phase));
    color = comp(color, gear(uv, dx, -phaseTimeEaseOut-.25, vec2(.17, .8), rad, 4., vec2(.02,.015), lgc, false, phase));
    color = comp(color, gear(uv, dx,  phaseTimeEaseOut-.0,  vec2(.34, 1.), rad, 4., vec2(.02,.015), lgc, false, phase));
    color = comp(color, gear(uv, dx,  phaseTimeEaseOut-.0,  vec2(1., .85), rad, 4., vec2(.02,.015), lgc, false, phase));
    color = comp(color, gear(uv, dx, -phaseTimeEaseOut+.17, vec2(1., .59), rad, 4., vec2(.02,.015), lgc, false, phase));
    
    // pendulum
    float pendulumPos = sin(time*tau/2.);
    float pAngle = pendulumPos*.5;
    vec2 p1 = vec2(0.5, 1.); // top pivot point of pendulum stick
    vec2 stickDir = vec2(-sin(pAngle), -cos(pAngle));
    vec2 p2 = p1 + .95 * stickDir; // center of pendulum bob
    float distToPivot = dot(stickDir, uv - p1);
    // add a little "motion blur"
    vec2 dirToStick = remove(uv - p2, stickDir);
    vec2 bobuv = uv - .25 * (1. - abs(pendulumPos)) * distToPivot * dirToStick;
    // pendulum bob
    float d = length(bobuv-p2) - .05;
    if (d < 0.) color = comp(color, premult(pendColor.rgb, S(0., -dx, d)));
    // pendulum stick
    d = sdCapsule2(bobuv, p1, p2, 0.01);
    if (d < 0.) color = comp(color, premult(pendColor.rgb, S(0., -dx, d)));
    
    // special background
    switch(phase) {
        case 0: color = comp(color, wings( uv, t, mix(lgc, vec4(.2, .15, .3, 1.), phaseTime ), dx )); break;
        case 1: color = comp(color, candles ( uv, t, dx )); break;
        case 2: color = comp(color, webfeet ( uv, t, dx, phase )); break;
        case 3: color = comp(color, graves  ( uv, t, dx )); break;
    }
    
    // base symbol of time
    const float r1 = .23; // inner radius of gear
    const float r2 = .31; // outer radius of gear (base of teeth)
    const float r3 = .40; // outer radius of gear teeth (tip of teeth)
    const float rayWidth = .05;

    // big gear color
    vec4 bgc = gear(uv, dx, phaseTimeEaseOut, vec2(.5, .5), vec3(.23, .31, .40), 10., vec2(.05,.04), gearColor, true, phase);
    color = comp(color, bgc);
    
    // eyes
    if (uv.y > .79 && uv.x > .14 && uv.x < .86) color = comp(color, timeEyes(uv, phaseTimeEaseOut, dx, mix(white,gearColor.rgb,.8), phase));
    
    return color;
}

// Function 43
float explosionTime()
{
    return max(0.0, animationTime() - 3.5) / (10.0 - 3.5);
}

// Function 44
vec2 timeRotation(float t)
{
    t *= 0.15;
    return vec2(1.5*PI-t, 0.5 * PI * (-1. - 0.15*sin(1.5*t)));
}

// Function 45
float controlledTime() {return 5.0;}

// Function 46
vec3 animate(vec3 p){
    p *= 20.0;
    vec3 p1 = p/100.0+vec3(iTime);
    p += vec3(sin(p1.x),sin(p1.y),sin(p1.z))*20.0;
    return p;
}

// Function 47
float grabTime()
{
  	float m = (iMouse.x/iResolution.x)*80.0;
	return (iTime+m+410.)*32.;
}

// Function 48
float getParticleStartTime(int partnr)
{
    return start_time*random(float(partnr*2));
}

// Function 49
float animationTime()
{
	return mod(iTime, 10.0);
}

// Function 50
float introTime() {
    return min(1.0, loopTime() / INTRO_TIME);
}

// Function 51
float lightPhotonStartTime( vec3 finalPos, float finalTime )
{
    float startTime = finalTime;
    
    // my old friend FPI
    for( int i = 0; i < 3; i++ )
    {
        startTime = finalTime - light_time_per_m() * length( light( startTime ) - finalPos );
    }
    
    return startTime;
}

// Function 52
float getTime(float t)
{
	return sin(iTime*t*0.001) * 0.5 + 0.5;
}

// Function 53
void WriteTime()
{
  float c = 0.0;
  c += drawInt(int(mod(iDate.w / 3600.0, 24.0)));    _ddot;
  c += drawInt(int(mod(iDate.w / 60.0 ,  60.0)),2);  _ddot;
  c += drawInt(int(mod(iDate.w,          60.0)),2);  _
  vColor = mix(vColor, drawColor, c);
}

// Function 54
float mechTime()
{
    float t = smoothstep(2.0, 4.0, animationTime() + sin(iTime * 24.0) * .025);
    
    // ugly ugly
    t *= smoothstep(4.5, 4.0, animationTime());    
    return t;
}

// Function 55
void DTimeSet (vec4 d)
{
  float nd;
  int yr, mo, da;
  idt[0] = DIG2 (floor (d.x / 100.));
  idt[1] = DIG2 (mod (d.x, 100.));
  idt[2] = DIG2 (d.z);
  idt[3] = DIG2 (floor (d.w / 3600.));
  idt[4] = DIG2 (floor (mod (d.w, 3600.) / 60.));
  idt[5] = DIG2 (floor (mod (d.w, 60.)));
  inm[0] = MName (int (d.y));
  inm[1] = DName (DWk (ivec3 (d.xyz)));
  nd = mod (float (DElaps (ivec3 (d.x, d.y + 1., d.z)) - DElaps (ivec3 (2020, 1, 30))), 1e4);
  icn[0] = DIG2 (floor (nd / 100.));
  icn[1] = DIG2 (mod (nd, 100.));
}

// Function 56
float oscillateInRange(float min, float max, float T)
{
    float v = (sin(T) + 1.0) * 0.5; // map T to [0.0, 1.0];
    return min + v * (max - min);   // map T to [min, max];
}

// Function 57
void animateCam(in vec2 uv, in float t, out vec3 p, out vec3 d, out vec3 e, out float s )
{
	t = mod(t,35.0);
    
    vec3 u = UP;
    float f = 1.0;
    if(t<PI4)
    {
    	e = vec3(30.0*cos(t*.125),2.0,30.0*sin(t*.125));
   		d = normalize(vec3(0.0)-e);
        s = shutterfade(0.0, PI4, t, .5);
    }
    else if(t<20.0)
    {
        e = mix(PLANET_ROT*-10.0,PLANET_ROT*10.0,smoothstep(PI4,20.0,t));
        e.y += 1.0;
        d = PLANET_ROT;
        s = shutterfade(PI4, 20.0, t, .5);
    }
    else if(t<25.0)
    {
        e = mix(vec3(-10.0,1.0,3.0),vec3(10.0,1.0,3.0),smoothstep(20.0,25.0,t));
        d = vec3(0.948683, 0.316228, 0.0);
        u = vec3(-d.y,d.x,0.0); 
        s = shutterfade(20.0, 25.0, t, .5);
    }
    else if(t<30.0)
    {
        e = mix(vec3(-30.0,10.0,-10.0),vec3(10.0,10.0,10.0),smoothstep(25.0,30.0,t));
        d = vec3(0.666667, -0.333333, 0.666667);
        u = vec3(-d.y,d.x,0.0); 
        s = shutterfade(25.0, 30.0, t, .5);
    }
    else
    {
        e = mix(vec3(1.0,1.0,2.0),vec3(1.0,5.0,1.5),smoothstep(30.0,35.0,t));
        d = UP;
        u = PLANET_ROT;
        f = .5;
        s = shutterfade(30.0, 35.0, t, .5);
    }
    camera(uv, e, d, u, f, p, d);
}

// Function 58
float randomTime()
{
	return fract(sin(dot(gl_FragCoord.xy*iTime, vec2(12.9898,78.233))) * 43758.5453);  
}

// Function 59
float animateEntranceJedi(float p, inout StickmanData data)
{        
    data.saberLen *= smoothstep(0.05, 0.3, p);
    float pose1 = 1.0 - smoothstep(.52, .7, p);
    poseSaberFront(pose1, data);
    
    frontLoop(smoothstep(.3, .4, p) - smoothstep(0.65, 0.7, p), linearstep(.3, .65, p)*2., data);
    
    float pose2 = smoothstep(.5, .7, p);
    poseSaberSide(pose2, data);
    
    return pose2;
}

// Function 60
vec3 animate(vec2 a, vec2 b) {
    vec2 res = a;
    
    if (abs(b.x-a.x) > 0.0) {
        float t = fract(iTime);
        float k1 = (1.0-step(1.0/3.0, t));
        float k2 = step(1.0/3.0, t)*(1.0-step(2.0/3.0, t));
        float k3 = step(2.0/3.0, t);
        
        res.x = k1*a.x + k2*mix(a.x,b.x,3.0*(t-1.0/3.0))+ k3*b.x;
    	res.y = k1*mix(a.y, 5.0, 3.0*t) + k2*5.0 + k3*(mix(5.0, b.y, 3.0*(t-2.0/3.0)));
    }
    return vec3(res, 0.0);
}

// Function 61
vec3 animate( vec3 v)
{
    float time = iTime+13.0;
	time = floor(time*8.0)/8.0; // force 8 fps
    
    // breath
    {
      vec3 p = vec3(0.0,0.05,0.23);
      float f = 1.0-smoothstep(0.0,0.16,length(p-v) );
        
      float b = 1.0 + 0.18*f*(0.5+0.5*sin(time*8.0));
      v = p + (v-p)*b;  
        
    }

    // tail
    {
        float k = v.z - (-0.18);
        if( k<0.0 )
        {

        float bn = sin(time*0.11);
        bn = bn*bn*bn;
        float an = sin(time*2.0 + k*6.0 + bn*10.0 + 2.0);
        an *= 0.5*k*an; an += 0.2;
        float co = cos(an);
        float si = sin(an);
        vec2 p = vec2(0.0,-0.18);
        v.xz = p + mat2(co,-si,si,co)*(v.xz-p);
        }
    }
    
    // head
    {
        float k = v.z - (+0.16);
        if( k>0.0 )
        {
            
        float an = sin(time*0.7*0.5);
        an = an*an*an;
        an = 1.5*k*an;
        float co = cos(an);
        float si = sin(an);
        vec2 p = vec2(0.0,0.16);
        v.xz = p + mat2(co,-si,si,co)*(v.xz-p);
            

        an = sin(time*0.5*0.5);
        an = an*an*an;
        an = -0.95*k*abs(an);
        co = cos(an);
        si = sin(an);
        p = vec2(0.0,0.16);
        v.yz = p + mat2(co,-si,si,co)*(v.yz-p);
        }
    }

    return v;
}

// Function 62
vec3 moveOverTime( ivec2 c )
{
	c -= ivec2( c_iGlassWidth / 2, c_iGlassHeight / 2 );
	vec3 acc = vec3( 0, -5, 0 );
	float velZ = max( 5.0 - length( vec2( c ) ), 0.0 );
	velZ = pow( velZ, 3.0 );
	vec3 vel = vec3( 0, 0, -velZ );
	return acc * g_fGlassCrashTime * g_fGlassCrashTime * 0.5 + vel * g_fGlassCrashTime;
	return vec3( 0, g_fGlassCrashTime * -1.0, 0 );
}

// Function 63
float Stime(float scale)
{
    return fract(iTime*scale)*TAU;
}

// Function 64
void animate(inout vec3 ro, inout vec3 ta)
{
    ro.x = sin(iTime * SPEED) * 0.5;
    ro.y = height;
    ro.z = 0.0;

    ta.x = 0.2;
    ta.y = height + sin((max(fract((iTime) / 40. - 0.4) * 4./3., 1.) - 1.) * 3. * PI);
    ta.z = 0.8;
}

// Function 65
float timeOfMove(float m) {
    return INTRO_TIME + m * TIME_PER_POSITION;
}

// Function 66
float GetBrightnessForTime( float t )
{
    return smoothstep( 0.0, LIGHT_RAMP_UP_TIME, t ) * lightBrightness;
}

// Function 67
float animate(float x) {
    return mod(x + iTime * 0.1, 1.);
}

// Function 68
float time(){
	return abs(0.3*tan(sin((iTime-1.0)/3.))+0.5);
}

// Function 69
float mod_time()
{
    return fract(iTime);
}

// Function 70
void process_text_time_accel( int i, inout int N,
                              inout vec4 params, inout uvec4 phrase, inout vec4 argv, FrameContext fr )
{
    float y = 3. * g_textres.y / 4.;
    if( ( g_game.switches & GS_PAUSE ) == GS_PAUSE )
    {
        if( i == N )
            params = vec4( g_textres.x / 2., y, step( .5, fract( iTime ) ), 12 ),
            phrase = uvec4( 0x50415553, 0x45000000, 0, 5u | TXT_FMT_FLAG_CENTER );
        N++;
    }
    else
    if( fr.timeaccel > 1.0625 )
    {
        if( i == N )
            params = vec4( g_textres.x / 2., y, step( .5, fract( iTime ) ), 12 ),
            phrase = uvec4( 0x54494d45, 0x20d7f500, 0, 7u | TXT_FMT_FLAG_CENTER ),
            argv.x = fr.timeaccel;
        N++;
    }

    if( g_game.camzoom > 1. )
    {
        if( i == N )
            params = vec4( g_textres.x / 2., y - 18., 1, 12 ),
            phrase = uvec4( 0x5a4f4f4d, 0, 0, 4u | TXT_FMT_FLAG_CENTER );
        N++;
    }
}

// Function 71
int timer( inout vec4 fragColor, in ivec2 fragC ){
    int t = read(8,0);
    write(8,0,t==60?0:t+1);
    return t;
}

// Function 72
void endTimer( inout vec4 fragColor, in ivec2 fragC ){
    write(8,0,0);
}

// Function 73
void setupTime(in float time) {
    gTime = time;
}

// Function 74
vec3 repeatTime(vec3 s)
{
    return fract(iTime*s)*3.14159265359*2.0;
}

// Function 75
float TimeSlow(float accel)
{
    return TIME_SLOW_FACTOR / (accel + TIME_SLOW_FACTOR);
}

// Function 76
float light_time_per_m()
{
    return (iMouse.z > 0.) ? (min(1.,max((MOUSEY-0.25)/0.7,0.))*0.06) : (sin(iTime*.25)*.5+.5)*.06 ;
}

// Function 77
float loopTime() {
    return mod(iTime, TOTAL_TIME);
}

// Function 78
void setTime(float tm){ gTime = tm; }

// Function 79
int animatedJulia(float x, float y) {
  float animationOffset = 0.055 * cos(iTime * 2.0);

  complex c = complex(-0.795 + animationOffset, 0.2321);
  complex z = complex(x, y);

  return fractal(c, z);
}

// Function 80
mat4 transformOverTime( ivec2 c )
{
	vec3 pieceOrigin = vec3( pointInCell( c ), 0 );
	mat3 mr = rotationOverTime( c );
	mat4 res = mat4( mr );
	mat4 mt = mat4( 1.0 );
	mt[3].xyz = vec3( -pieceOrigin );
	res = mat4( mr ) * mt;
	mt[3].xyz = pieceOrigin;
	res = mt * res;
	
	mt[3].xyz = moveOverTime( c );
	res = mt * res;
	return res;
}

// Function 81
float GetTime()
{
	return time;
}

// Function 82
void SetTime(v0 t){ProcessLightValue(t);ProcessObjectPos(t);}

// Function 83
float fracturedTime(float offset){
    float _fractured = fract(iTime*timeScale + offset);
    _fractured = distance(_fractured, 0.5) * 2.;
    _fractured = 1. - _fractured;
    return _fractured;
}

// Function 84
vec2 SunAtTime(in float julianDay2000, in float latitude, in float longitude) {
	float zs,rightAscention, declination, sundist,
		t  = julianDay2000,	//= jd - 2451545., // nb julian days since 01/01/2000 (1 January 2000 = 2451545 Julian Days)
		t0 = t/36525.,    		 // nb julian centuries since 2000      
		t1 = t0+1.,   		 	 // nb julian centuries since 1900
		Ls = fract(.779072+.00273790931*t)*PI2, // mean longitude of sun
		Ms = fract(.993126+.0027377785 *t)*PI2, // mean anomaly of sun
		GMST = 280.46061837 + 360.98564736629*t + (0.000387933 - t0/38710000.)*t0*t0, // Greenwich Mean Sidereal Time   
// position of sun
		v = (.39785-.00021*t1)*sin(Ls)-.01*sin(Ls-Ms)+.00333*sin(Ls+Ms),
		u = 1.-.03349*cos(Ms)-.00014*cos(2.*Ls)+.00008*cos(Ls),
		w = -.0001-.04129 * sin(2.*Ls)+(.03211-.00008*t1)*sin(Ms)
			+.00104*sin(2.*Ls-Ms)-.00035*sin(2.*Ls+Ms);
// calcul distance of sun
	sundist = 1.00021*sqrt(u)*AU;
// calcul right ascention
	zs = w / sqrt(u-v*v);
	rightAscention = Ls + atan(zs/sqrt(1.-zs*zs));
// calcul declination
	zs = v / sqrt(u);
	declination = atan(zs/sqrt(1.-zs*zs));
// position relative to geographic location
	float
		sin_dec = sin(declination),   cos_dec = cos(declination),
		sin_lat = sin(TORAD*latitude),cos_lat = cos(TORAD*latitude),
		lmst = mod((GMST + longitude)/15., 24.);
	if (lmst<0.) lmst += 24.;
	lmst = TORAD*lmst*15.;
	float
		ha = lmst - rightAscention,       
		elevation = asin(sin_lat * sin_dec + cos_lat * cos_dec * cos(ha)),
		azimuth   = acos((sin_dec - (sin_lat*sin(elevation))) / (cos_lat*cos(elevation)));
	return vec2(sin(ha)>0.? azimuth:PI2-azimuth, elevation);
}

// Function 85
vec4 getTime()
{
    return texture(iChannel1, vec2(3.5, 0.5) / iResolution.xy);
}

// Function 86
void animateJedi(float t, inout StickmanData data)
{    
    float entranceDur = 4.5;
    float prevTwoHanded = 0.0;
    float twoHanded = 0.0;
    float i = 0.0;
    float hit = 0.0;
    float s, e;
    
#if ANIMATE    
    s = -entranceDur;
    e = 0.0;
    TRANS_POSE(s, e, nullPose, animateEntranceJedi)
    
    t = mod(t, loopTime) * step(0.0, t);
    
	s = e;
    e = s + 0.5;    
    HOLD_POSE(s, e, poseSaberSide)
        
    s = e;
    e = s + 0.7;    
    HOLD_POSE(s, e, poseSaberSide)
    HIT_SEQ(s, e, 1.0, parryDownLeft) 
        
    s = e;
    e = s + 0.2;    
    HOLD_POSE(s, e, poseSaberSide)
        
    s = e;
    e = s + 1.65;    
    HOLD_POSE(s, e, poseSaberSide)
    HIT_SEQ(s, e, 0.60, whirlingHit)  
        
    s = e;
    e = s + 1.0;    
    HOLD_POSE(s, e, poseSaberSide)
                
    s = e;
    e = s + 0.9;    
    HOLD_POSE(s, e, poseSaberSide)
    HIT_SEQ(s, e, 1.0, parryDownLeft)  

    s = e;
    e = s + 0.45;    
    TRANS_POSE(s, e, poseSaberSide, poseSaberFront) 
        
    s = e;
    e = s + 0.9;    
    HOLD_POSE(s, e, poseSaberFront)
    HIT_SEQ(s, e, 1.0, parryDownLeft)                  
        
    s = e;
    e = s + 2.0;    
    HOLD_POSE(s, e, poseSaberFront)
    HIT_SEQ(s, e, 0.4375, upDownHit)   
        
	s = e;
    e = s + 0.4;    
    HOLD_POSE(s, e, poseSaberFront)
        
    s = e;
    e = s + 0.6;    
    TRANS_POSE(s, e, poseSaberFront, poseSaberSide) 
    
    s = e;
    e = loopTime;    
    HOLD_POSE(s, e, poseSaberSide)
#endif        
            
    invKinematics(twoHanded, hit, data);
}

// Function 87
void animateSith(float t, inout StickmanData data)
{        
    float entranceDur = 4.5;
    float twoHanded = 0.0;
    float i = 0.0;
    float hit = 0.0;
    float prevTwoHanded = 0.0;
    
    float s, e;
#if ANIMATE
    //do pose    
    s = -entranceDur;
    e = 0.0;
    TRANS_POSE(s, e, nullPose, animateEntranceSith)
    
    t = mod(t, loopTime) * step(0.0, t);
    
    s = e;
    e = s + 1.7;    
    HOLD_POSE(s, e, poseSaberBack)
    HIT_SEQ(s, e, 0.434, upDownHit)
       
    s = e;
    e = s + 0.6;    
    HOLD_POSE(s, e, poseSaberBack)
        
    s = e;
    e = s + 0.6;    
    HOLD_POSE(s, e, poseSaberBack)
    HIT_SEQ(s, e, 1.0, parryUpRight)
        
    s = e;
    e = s + 0.5;    
    TRANS_POSE(s, e, poseSaberBack, poseSaberBackDown)
        
    s = e;
    e = s + 1.5;    
    HOLD_POSE(s, e, poseSaberBackDown)
    HIT_SEQ(s, e, 0.558, whirlingHit)
        
    s = e;
    e = s + 0.4;    
    TRANS_POSE(s, e, poseSaberBackDown, poseSaberBack)
        
    s = e;
    e = s + 1.0;    
    HOLD_POSE(s, e, poseSaberBack)            
    HIT_SEQ(s, e, 0.6, forwardHit)
    
    s = e;
    e = s + 0.1;    
    HOLD_POSE(s, e, poseSaberBack)  
        
    s = e;
    e = s + 0.42;    
    TRANS_POSE(s, e, poseSaberBack, poseSaberBackDown)
        
    s = e;
    e = s + 1.2;    
    HOLD_POSE(s, e, poseSaberBackDown)
    HIT_SEQ(s, e, 1.0, parryDownLeft)   
        
	s = e;
    e = s + 0.8;    
    HOLD_POSE(s, e, poseSaberBackDown)
        
    s = e;
    e = s + 0.5;    
    TRANS_POSE(s, e, poseSaberBackDown, poseSaberBack)
    
    s = e;
    e = loopTime;    
    HOLD_POSE(s, e, poseSaberBack)
#endif        
    
    invKinematics(twoHanded, hit, data);
}

// Function 88
float loopTime(float iTime) {
	return mod(iTime / 3. + .35, 1.);
}

// Function 89
void SetTime(float t){
 ;ProcessLightValue(t)//also called in final pass
 ;objPos[oCubeMy]=vec3(0) 
 ;objRot[oCubeMy]=aa2q(t*2.,vec3(0,1,0))
 ;objSca[oCubeMy]=vec3(.8)
 ;objPos[oBlackHole]=vec3(5.,sin(t*0.2),-5.)
 ;objRot[oBlackHole]=aa2q(t*2.,vec3(0,1,0))
 ;objSca[oBlackHole]=vec3(1)
 ;objPos[oCubeChil]=vec3(1)
 ;objRot[oCubeChil]=aa2q(t*1.,normalize(objPos[oCubeChil]))
 ;objSca[oCubeChil]=vec3(.4)
 ;float trainV = 2.2
 ;objVel[oTrain]= vec3((floor(mod(trainV*t/16.,2.))*2.-1.)*trainV,0,0)
 ;float trainDir = 1.
 ;if (objVel[oTrain].x < 0.)trainDir = -1.
 ;objPos[oTrain]=vec3(abs(1.-mod(trainV*t/16.,2.))*16.-8.,-.8,9.)
 ;objRot[oTrain]=aa2q(pi*.5,vec3(0,1,0))
 ;objSca[oTrain]= vec3(1.,1.,trainDir/mix(LorentzFactor(trainV*LgthContraction),1.,cLag))
 ;objPos[oTunnel]=vec3(0,-.8,9.)
 ;objRot[oTunnel]=aa2q(pi*.5,vec3(0,1,0))
 ;objSca[oTunnel]=vec3(1.,1.,1)
 ;objPos[oTunnelDoor]=objPos[oTunnel]
 ;objRot[oTunnelDoor]=objRot[oTunnel]
 ;float open = sat((1.-abs(3.*objPos[oTrain].x))*2.)
 ;objSca[oTunnelDoor]= vec3(open,open,1);}

// Function 90
vec3 fmt_time( int arg )
{
    int hours = arg / 3600;
    int minutes = ( arg - 3600 * hours ) / 60;
    int seconds = arg - 60 * minutes - 3600 * hours;
    return vec3( hours, minutes, seconds );
}

// Function 91
float GetSceneTime()
{
	#ifdef LIMIT_FRAMERATE
		return (floor(iTime * kFramesPerSecond) / kFramesPerSecond);
	#else
		return iTime;
	#endif
}

// Function 92
vec2 Oscillator(float Fo, float Fs, float n)
{
    float phase = (tau*Fo*floor(n))/Fs;
    return vec2(cos(phase),sin(phase));
}

// Function 93
float moveTime() {
	return max(0.0, loopTime() - INTRO_TIME);
}

// Function 94
float time()
{vec2 m=iMouse.xy/iResolution.xy;
;return + m.x*64.+iTime*.1;
 //;return + m.x*64.0; //+time
}

