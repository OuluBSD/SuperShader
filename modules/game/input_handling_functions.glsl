// Reusable Input Handling Game Functions
// Automatically extracted from game/interactive-related shaders

// Function 1
bool keyPressed(int keyCode) {
	return bool(texture(iChannel3, vec2((float(keyCode) + 0.5) / 256., .5/3.)).r);   
}

// Function 2
void WriteMousePos(vec2 mPos)
{
  mPos = abs(mPos);
  int digits = 3;
  float radius = 3.0;

  // print dot at mPos
  if (iMouse.z > 0.0) dotColor = mpColor;
  float fDistToPointB = length(mPos - ppos) - radius;
  vColor += mix( vec3(0), dotColor, (1.0 - clamp(fDistToPointB, 0.0, 1.0)));

  // print mouse.x
  tp = mPos + vec2(-4.4 * vFontSize.x, radius + 4.0);
  tp.x = max(tp.x, -vFontSize.x);
  tp.x = min(tp.x, iResolution.x - 8.4*vFontSize.x);
  tp.y = max(tp.y, 1.6 * vFontSize.y);
  tp.y = min(tp.y, iResolution.y - 1.4*vFontSize.y);
  drawColor = mxColor;
  WriteValue(tp, mPos.x, digits, 0);
		
  // print 2nd mouse value
  SPACE
  drawColor = myColor;
  WriteValue(tp, mPos.y, digits, 0);
}

// Function 3
void handleKeys( inout vec4 buffer, in vec2 fragCoord )
{
    // Load keypress data from the buffer.
    vec4 WAS = readTexel(BUFFER, txWAS);
    vec4 DLR = readTexel(BUFFER, txDLR);
    
    // If a key is down, update the position accordingly.
    vec3 was, dlr;
    was.r = texture(KEYBOARD,vec2(KEY_W, KEY_DOWN_POS)).r;
    was.g = texture(KEYBOARD,vec2(KEY_A, KEY_DOWN_POS)).r;
    was.b = texture(KEYBOARD,vec2(KEY_S, KEY_DOWN_POS)).r;
    dlr.r = texture(KEYBOARD,vec2(KEY_D, KEY_DOWN_POS)).r;
    dlr.g = texture(KEYBOARD,vec2(KEY_L, KEY_DOWN_POS)).r;
    dlr.b = texture(KEYBOARD,vec2(KEY_R, KEY_DOWN_POS)).r;
    
    // Store the keys value. That mix eases between full up and
    // full down states, so that the velocity of the camera
    // isn't as jarring.
    write3(buffer.rgb,mix(WAS.rgb,was,.5),txWAS,fragCoord);
    write3(buffer.rgb,mix(DLR.rgb,dlr,.5),txDLR,fragCoord);
}

// Function 4
bool keyToggle(int ascii) 
{
	return (texture(iChannel2,vec2((.5+float(ascii))/256.,0.75)).x > 0.);
}

// Function 5
float key(in int key){return texture(iChannel1,vec2((float(key)+0.5)/256.0, 0.25)).x;}

// Function 6
float KeyToFrequency(int n){
    return pow(Semitone,float(n-49))*440.;
}

// Function 7
bool GetKeyDown(int key)
{
    return ReadKey(key, false);
}

// Function 8
float isKeyPressed(float key)
{
    return texture( iChannel0, vec2(key, 3.5) ).x;
}

// Function 9
float keyState(float key, float default_state) {
    return abs( texture(iChannel0, vec2(key, 0.75)).x - default_state );
}

// Function 10
void processMouseOnComponents(ivec2 ifc, vec2 m){
    //switch(components[ifc.y].type)
    {
        if (components[ifc.y].type == TYPE_CHECKBOX)
        {
            if (isClicking(components[ifc.y].positionSize.xy, vec2(.020),  m) == 1. && onEventPress)
            {
                invertBool(components[ifc.y].value.x);
            }
        }
        //break;

        if (components[ifc.y].type == TYPE_BUTTON)
        {
            if (isClicking(components[ifc.y].positionSize.xy,  vec2(.030)+components[ifc.y].positionSize.zw, m) == 1. && onEventPress)
            {
                invertBool(components[ifc.y].value.x);
            }
        }
        //break;
    }
}

// Function 11
float keyPress(int keyCode) {
	return textureLod(iChannel2, vec2((float(keyCode) + 0.5) / 256., 1.5/3.), 0.0).r;   
}

// Function 12
float mouseSelect(vec2 c) {
	float scale = floor(iResolution.y / 128.);
    c /= scale;
    vec2 r = iResolution.xy / scale;
    float xStart = (r.x - 16. * NUM_ITEMS) / 2.;
    c.x -= xStart;
    if (c.x < NUM_ITEMS * 16. && c.x >= 0. && c.y < 16.) {
        float slot = floor(c.x / 16.);
    	return slot;
    }

    return -1.;
}

// Function 13
bool getKeyPress(int key) {
    return texelFetch(iChannel3, ivec2(key, 0), 0).x > 0.;
}

// Function 14
float IsKeyToggled(float key){return texture(iChannel3, vec2((key+.5)/256.,2.)).r;}

// Function 15
bool keypress(int key) {
    return texelFetch(iChannel0, ivec2(key,2),0).x != 0.0;
}

// Function 16
float keyPress(int keyCode) {
    return textureLod(iChannel2, vec2((float(keyCode) + 0.5) / 256., 1.5/3.), 0.0).r;   
}

// Function 17
vec4 UpdateMouseDelta(inout vec4 o,vec2 mouse)
{vec3 m=GetLastMouseClick()
;vec2 t=mouse.xy/iResolution.xy-m.xy
,l =GetLastMouseDelta().zw
;if(m.z<.5)return vec4(o.xy,0,0)
;return vec4(t-l,t);}

// Function 18
bool keyToggle(int ascii) { return (texture(iChannel2,vec2((.5+float(ascii))/256.,0.75)).x > 0.); }

// Function 19
float ReadKey( int key )
{
	return ReadKeyInternal(key,false);
}

// Function 20
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

// Function 21
vec3 getMouse(vec3 ro) {
    float x = M.xy == vec2(0) ? 0.5 : -(M.y/R.y * 1. - .5) * PI;
    float y = M.xy == vec2(0) ? 0. : (M.x/R.x * 1. - .5) * PI;
    ro.zy *=r2(x);
    ro.xz *=r2(y);
	return ro;   
}

// Function 22
bool isKeyDown(int keycode) {
    return texture(iChannel1,vec2(float(keycode)/255.,0.)).r > 0.5;
}

// Function 23
bool keypress(int key) {
  //return false;
  return texelFetch(iChannel0, ivec2(key,2),0).x != 0.0;
}

// Function 24
float keyShade(float a){a=mod(a,12.);return (mod(a+step(5.,a),2.));}

// Function 25
mat3 fetchMouseRotation(){
    float pi = atan(1.0) * 4.0;
float tau = atan(1.0) * 8.0;
    
        
    //Camera stuff   
    vec3 angles = vec3(0);
    
    if(iMouse.xy == vec2(0,0))
    {
		angles = vec3(vec2(.5) * pi, 0);
        angles.xy *= vec2(1.6, 0.40);//STARTING ANGLE
    }
    else
    {    
    	angles = vec3((iMouse.xy / iResolution.xy) * pi, 0);
        angles.xy *= vec2(2.0, 1.0);
    }
    
    angles.y = clamp(angles.y, 0.0, tau / 4.0);
    return Rotate(vec3(pi/2.,pi,0.))*Rotate(angles.yxz);
}

// Function 26
if keyToggle(65) {               // without strip modulation
       Z = CS(a);  
       O = vec4( .5+.5 * Z, 0, 1);           // complex to colors
     //O = sqrt(O.xxxx);  
    }

// Function 27
void DrawKey( vec2 pos, vec3 keyCol )
{
	vec2 p = pixel-pos;
	vec2 c = p-vec2(5,0);
	
	bool draw = false;
	if ( abs(p.x) < 10.0 && abs(p.y) < 5.0 )
	{
		if ( p.x > 1.0 )
		{
			if ( length(c) < 5.0 && length(c-vec2(1,0)) > 2.0 )
				draw = true;
		}
		else
		{
			if ( p.y < 2.0 &&
				p.y > 10.0-20.0*texture( iChannel1, vec2(0,pixel.x/5.0) ).r )
				draw = true;
		}
	}
	
	if ( draw )
		fragColor.rgb = keyCol;
}

// Function 28
float chromaKey(vec3 color)
{
	vec3 backgroundColor = vec3(0.157, 0.576, 0.129);
	vec3 weights = vec3(4., 1., 2.);

	vec3 hsv = rgb2hsv(color);
	vec3 target = rgb2hsv(backgroundColor);
	float dist = length(weights * (target - hsv));
	return 1. - clamp(3. * dist - 1.5, 0., 1.);
}

// Function 29
bool handleMouse(int index, vec4 mouse, int capturedIndex) {
    float dist = midpointDistNorm(index, typeCheatSheet(index), mouse.xy);
    bool capturable = capturedIndex == index || capturedIndex < 0;
    if (mouse[2] > 0. && dist < 1. && controls[index].visible) {
        if (controls[index].enabled && capturable) {
            if (!controls[index].mouseDown) {
                controls[index].mouseDown = true;

                mouseDownEvt(index, mouse, capturedIndex);
            } else {
            	mouseMoveEvt(index, mouse, capturedIndex, true);
        	}
        }
        return true;
    } else {
        if (controls[index].mouseDown) {
            if (mouse[2] <= 0.) {
                mouseUpEvt(index, mouse, capturedIndex);
            } else {            
                mouseMoveEvt(index, mouse, capturedIndex, false);
            }
        }        
        
        if (capturedIndex != index) {
            controls[index].mouseDown = false;        
        }
        return false;
    }            
}

// Function 30
void writeMousePos(vec2 val, inout vec4 fragColor, in vec2 fragCoord)
{
    if(isPixel(0,0,fragCoord)) fragColor.xy=val;
}

// Function 31
void mouseUpEvt(int index, vec4 mouse, int capturedIndex) {
    // mouseUp event:
    switch (controls[index].type) {
        case PUSH_BTN_T:
        controls[index].value = 0.;
        break;
    }
}

// Function 32
bool keyToggle(float key)
{
	return texture(iChannel1, vec2(key, 2.5/3.) ).x > .0;
}

// Function 33
bool IsInputThread(in vec2 fragCoord)
{
    return ALLOW_KEYBOARD_INPUT != 0 && int(fragCoord.x) == 0 && int(fragCoord.y) == 0;
}

// Function 34
float mouseDif(float c,float m){//return sign(c)*max(-sign(c)*c,m);//alternative to below
 //return min(c*sign(m),m);//shorter variant of below
 float s=sign(m)*c;m-=s;return s+.5*(m-abs(m));}

// Function 35
bool keyToggle(int ascii) 
{
	return (texture(iChannel1,vec2((.5+float(ascii))/256.,0.75)).x > 0.);
}

// Function 36
bool isKeyPressed(float key)
{
	return texture(iChannel1, vec2(key, 0.5/3.) ).x > .0;
}

// Function 37
float keyState(int key, sampler2D chan) {
	return texelFetch( chan, ivec2(key,0), 0 ).x;
}

// Function 38
bool ReadKey(int key, bool toggle)
{
  return 0.5 < texture(iChannel3
    ,vec2((float(key)+0.5) / 256.0, toggle ? 0.75 : 0.25)).x;
}

// Function 39
mat3 getMouseRotMtx()
{
    float f= .05;
    vec2 a = .5*vec2(.3,.2);
    vec2 o = vec2(.2,-.2);
    return rotationXY(-o+a*vec2(sin(f*t), cos(f*t)));
    
	// Use shadertoy mouse uniform
    vec4 m = iMouse;
    vec2 mm = m.xy - abs(m.zw);
    vec2 rv = 0.01*mm;
	mat3 rotmtx = rotationY(rv.x) * rotationX(-rv.y);
    return rotmtx;
}

// Function 40
vec3 CameraDirInput(vec2 vm) {
    vec2 m = vm/iResolution.y;
    m.y = -m.y;
    
    mat3 rotX = mat3(1.0, 0.0, 0.0, 0.0, cos(m.y), sin(m.y), 0.0, -sin(m.y), cos(m.y));
    mat3 rotY = mat3(cos(m.x), 0.0, -sin(m.x), 0.0, 1.0, 0.0, sin(m.x), 0.0, cos(m.x));
    
    return (rotY * rotX) * (vec3(KeyboardInput(), 0.0).xzy+ vec3(KeyboardInput2(), 0.0).xzy);
}

// Function 41
vec2 mouseDelta(){
    vec2 pixelSize = 1. / iResolution.xy;
    float eighth = 1./8.;
    vec4 oldMouse = texture(iChannel3, vec2(7.5 * eighth, 2.5 * eighth));
    vec4 nowMouse = vec4(iMouse.xy / iResolution.xy, iMouse.zw / iResolution.xy);
    if(oldMouse.z > pixelSize.x && oldMouse.w > pixelSize.y && 
       nowMouse.z > pixelSize.x && nowMouse.w > pixelSize.y)
    {
        return nowMouse.xy - oldMouse.xy;
    }
    return vec2(0.);
}

// Function 42
vec4 oldMouse() 
{
    return loadValue(BufferC, slotMouseOld);
}

// Function 43
vec3 getMouse(vec3 p) {
    float x = M.xy == vec2(0) ? 0. : -(M.y/R.y * 1. - .5) * PI;
    float y = M.xy == vec2(0) ? 0. :  (M.x/R.x * 1. - .5) * PI;
    
    p.zy *=r2(x);
    p.xz *=r2(y);
    return p;
}

// Function 44
bool affMouse() 
{
	float R=5.;
	vec2 pix = FragCoord.xy/iResolution.y;
	float pt = max(1e-2,1./iResolution.y); R*=pt;

	vec2 ptr = iMouse.xy/iResolution.y; 
	vec2 val = iMouse.zw/iResolution.y; 
	float s=sign(val.x); val = val*s;
	
	// current mouse pos
    float k = dot(ptr-pix,ptr-pix)/(R*R*.4*.4);
		if (k<1.) 
	    { if (k>.8*.8) FragColor = vec4(0.);
		     else      FragColor = vec4(s,.4,0.,1.); 
		  return true;
		}
	
	// prev mouse pos 
    k = dot(val-pix,val-pix)/(R*R*.4*.4);
		if (k<1.) 
	    { if (k>.8*.8) FragColor = vec4(0.);
		     else      FragColor = vec4(0.,.2,s,1.); 
		  return true;
		}
	
	return false;
}

// Function 45
void mouse(ivec2 var, int clamp) {
    
    vec4 mstate = LOAD(MSTATE);
    
    if (mstate.xy == vec2(0) && iMouse.xy == iMouse.zw) {
        
        vec2 p = LOAD(var).xy;
        vec2 offset = p - mxy;
        
        if (length(offset) < pointSize + 16.0*scl) {
            STORE(MSTATE, vec4(vec2(var), offset));
        }
        
    } else if (mstate.xy == vec2(var)) {
        
        if (mouseIsDown) {
            vec2 pos = mxy + mstate.zw;
            if (clamp == 1) { 
                pos = clampCircle(pos); 
            } else if (clamp != 0) {
                pos = blockCircle(pos);
            }
            demoCaseIsModified = 1.;
            STORE(var, vec4(pos, 0, 0));
        } else {
            STORE(MSTATE, vec4(0));
        } 
        
    }

}

// Function 46
void mixKeyFrame(KeyFrame a, KeyFrame b, float ratio, out KeyFrame c)
{
    ratio = ratio*ratio*(3.0-2.0*ratio); // Thanks iq :D
    
    c.leafAngle		= mix(a.leafAngle , b.leafAngle	  , ratio);
    c.mouthAngle	= mix(a.mouthAngle, b.mouthAngle  , ratio);
    c.spine1		= mix(a.spine1	  , b.spine1	  , ratio);
    c.spine2		= mix(a.spine2	  , b.spine2	  , ratio);
    c.spine3		= mix(a.spine3	  , b.spine3	  , ratio);
    c.neck			= mix(a.neck	  , b.neck		  , ratio);
}

// Function 47
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

// Function 48
vec2 GetMouse()
{
	return iMouse.xy;    
    
    //vec2 vClampRes = min( iResolution.xy, vec2(640.0, 480.0) );    
    //return iMouse.xy * vClampRes / iResolution.xy;
}

// Function 49
bool onKeyPress(float key){
	return texture(iChannel1, vec2(key, 0.0)).x > 0.0;
}

// Function 50
bool keyPress(int ascii){
    return (texture(iChannel1,vec2((.5+float(ascii))/256.,0.25)).x > 0.);
}

// Function 51
vec4 drawKey( vec2 uv, int color ) {
    uv = floor(fract(uv)*32.) - 16.;
        float l = step(abs(uv.y), 1.);
        l = max(l, step(length(uv+vec2(8,0)), 7.5));
        l -= step(length(uv+vec2(8,0)), 4.5);
        l = max(l, step(6.,uv.x)*step(uv.x, 7.)*step(0.,uv.y)*step(abs(uv.y), 5.));
        l = max(l, step(10.,uv.x)*step(uv.x, 11.)*step(0.,uv.y)*step(abs(uv.y), 7.));
        l = max(l, step(14.,uv.x)*step(0.,uv.y)*step(abs(uv.y), 6.));
        
	    vec3 col = vec3(0);
    	col[color] = 1.;
        return vec4( l * (.75 + .25 * texture(iChannel1, uv/64.).x) * col, l );

}

// Function 52
void SpriteKey( inout vec3 color, vec2 p )
{
    p -= vec2( 5., 2. );
    p = p.x < 0. ? vec2( 0. ) : p;    
    
    int v = 0;
	v = p.y == 11. ? ( p.x < 8. ? 139824 : 0 ) : v;
	v = p.y == 10. ? ( p.x < 8. ? 2232611 : 0 ) : v;
	v = p.y == 9. ? ( p.x < 8. ? 1179666 : 0 ) : v;
	v = p.y == 8. ? ( p.x < 8. ? 1245202 : 0 ) : v;
	v = p.y == 7. ? ( p.x < 8. ? 1192482 : 0 ) : v;
	v = p.y == 6. ? ( p.x < 8. ? 74256 : 0 ) : v;
	v = p.y == 5. ? ( p.x < 8. ? 4608 : 0 ) : v;
	v = p.y == 4. ? ( p.x < 8. ? 4608 : 0 ) : v;
	v = p.y == 3. ? ( p.x < 8. ? 4608 : 0 ) : v;
	v = p.y == 2. ? ( p.x < 8. ? 2232832 : 0 ) : v;
	v = p.y == 1. ? ( p.x < 8. ? 135680 : 0 ) : v;
	v = p.y == 0. ? ( p.x < 8. ? 2232832 : 0 ) : v;
    float i = float( ( v >> int( 4. * p.x ) ) & 15 );
    color = i == 1. ? vec3( 0.45 ) : color;
    color = i == 2. ? vec3( 0.83 ) : color;
    color = i == 3. ? vec3( 0.95 ) : color;
}

// Function 53
bool KeyDown(in int key){
	return (texture(iChannel1,vec2((float(key)+0.5)/256.0, 0.25)).x>0.0);
}

// Function 54
bool getInput(inout mat4 state, inout vec4 fragColor, inout vec2 fragCoord) {
    //shadertoy covers up 3 pixels on each top corner, use them for state storage
    if (fragCoord.y > iResolution.y - 1.) {
        //store/read location
        if (fragCoord.x < 1.) {
            if (iFrame == 0) fragColor = INIT_POS;
            else fragColor = vec4(state[0].xyz + state[1].xyz * iTimeDelta, 1.);
            //collision detection?
    	    return true;
        //store/read velocity
        } else if (fragCoord.x < 2.) {
    	  	float rot = state[2].y;
    	    vec3 acc = rotateY(getAcceleration(), rot);
        	fragColor = vec4(state[1].xyz + acc * MAX_ACCELERATION * iTimeDelta, 1.);
            //limit speed
 	        float speed = length(fragColor.xyz);
    	    if (speed > MAX_VELOCITY) fragColor.xyz *= MAX_VELOCITY / speed;
        	else if (speed > FRICTION * iTimeDelta) fragColor.xyz *= (speed - FRICTION * iTimeDelta) / speed;
            else fragColor.xyz = vec3(0.0, 0.0, 0.0);
	        return true;
        //store/read rotation
        } else if (fragCoord.x > iResolution.x - 1.) {
        	if (iMouse.z > 0.) {
		        vec4 mouse = 2.0 * abs(iMouse) / iResolution.y;
		        vec4 rot = state[3];
				fragColor = vec4(clamp(mouse.y - mouse.w + rot.x, -pi_5, pi_5),
                                 	   mouse.x - mouse.z + rot.y, 0.,0.);
            } else fragColor = state[2];
            return true;
        //store/read mouse
        } else if (fragCoord.x > iResolution.x - 2.) {
            if (iFrame == 0) fragColor = INIT_ROT;
            else if (iMouse.z < 0.) fragColor = state[2];
            else fragColor = state[3];
            return true;
        }
    }
    return false;
}

// Function 55
void gs_mouselook( inout GameState gs )
{
	if( keystate( KEY_BACK ) > 0. )
        if( ( gs.switches & GS_TRMAP ) == 0u )
			gs.mouselook = UNIT_X, gs.camzoom = 1.;
		else
            gs_enter_map_mode( gs );
	else
    if( iMouse.z > 0. )
    {
        float zoomres = gs.camzoom * CAM_FOCUS * iResolution.y;
        vec2 dragdelta = 2. * ( iMouse.xy - gs.dragstate ) / zoomres;
        gs.dragstate = iMouse.xy;
        float l = PI / 2. - 0.001 / gs.camzoom;
        float q = cos( gs.mouselook.z ) + 0.25 / gs.camzoom;
        vec2 sc = sincospi( dragdelta.x / ( q * PI ) );
        gs.mouselook.xy = normalize( gs.mouselook.xy * mat2( sc.yx, -sc.x, sc.y ) );
        gs.mouselook.z = clamp( gs.mouselook.z + dragdelta.y, -l, l );
    }
}

// Function 56
bool key(int u){return keyP(vec2( 96+u,.1)+.5);}

// Function 57
vec3 CameraDirInput(vec2 vm) {
    vec2 m = vm/iResolution.x;
    m.y = -m.y;
    
    mat3 rotX = mat3(1.0, 0.0, 0.0, 0.0, cos(m.y), sin(m.y), 0.0, -sin(m.y), cos(m.y));
    mat3 rotY = mat3(cos(m.x), 0.0, -sin(m.x), 0.0, 1.0, 0.0, sin(m.x), 0.0, cos(m.x));
    
    return (rotY * rotX) * vec3(KeyboardInput(), 0.0).xzy;
}

// Function 58
bool keyboard (int keycode, int evt) {
  return texelFetch(iChannel2, ivec2(keycode,evt), 0).r>.5;
}

// Function 59
vec4 iMouseZwFix(vec4 m,bool NewCoke
 ){if(m.z>0.){ //while mouse down
    if(m.w>0.)return m;//mouse was clicked in THIS     iFrame 
    else m.w=-m.w      //mosue was clicked in previous iFrame
    //remember, MouseDrag advances the iFrame Count, even while paused !!
 ;}else{if(!NewCoke||m.w>0.)return m.xyxy; //OPTIONAL onMouseUp (fold or whatever)
    m.zw=-m.zw;}
  return m;}

// Function 60
bool isKeyDown(int key) {
  return texelFetch(iChannel1, ivec2(key, 0), 0).x != 0.0;
}

// Function 61
float ReadKeyInternal( int key, bool toggle )
{
	return texture( iChannel3, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ) ).x;
}

// Function 62
bool KeyIsToggled(float key)
{
	return texture( iChannel1, vec2(key, 1.0) ).x > 0.0;
}

// Function 63
void prevMouse(out vec4 fragColor) {//mouse position
    vec4 v = vec4(iMouse.xy/iChannelResolution[0].xy,1.,0.); 
    fragColor = v; //update value stored (mouse position normalized)
}

// Function 64
bool key_toggle( float ascii ) { return (texture( iChannel0, vec2( ( ascii + .5 ) / 256., 0.75 ) ).x > 0.); }

// Function 65
bool KeyClick(sampler2D keyboard,int keyCode){
	return KeyData(keyboard,keyCode,1) > 0.;
}

// Function 66
float keyPress(int keyCode) {
    return texture(iChannel2, vec2((float(keyCode) + 0.5) / 256., 1.5/3.), 0.0).r;   
}

// Function 67
bool keyToggle(int ascii) {return (texture(iChannel2,vec2((.5+float(ascii))/256.,0.75)).x > 0.); }

// Function 68
vec2 MouseDrag(out vec2 Store,ivec2 iU,ivec2 coord){
    #define MouseTex iChannel1
    #define Scale 32767.
	uvec2 data = floatBitsToUint(GetValue(MouseTex,coord).zw);
	mediump vec4 MousePos = vec4(unpackSnorm2x16(data.x),unpackSnorm2x16(data.y))*Scale,tmpMousePos = vec4(MousePos.xy,0.,0.);
    iMouse.z>0. ? tmpMousePos.xy+=tmpMousePos.zw=iMouse.xy-iMouse.zw : tmpMousePos.xy=MousePos.xy+=MousePos.zw;
    if(IsCoord(iU,coord)) Store=uintBitsToFloat(uvec2(packSnorm2x16(fract(MousePos.xy/Scale)),packSnorm2x16(tmpMousePos.zw/Scale)));
    return tmpMousePos.xy;
}

// Function 69
float keyToggled(int keyCode) {
    return texture(iChannel2, vec2((float(keyCode) + 0.5) / 256., 2.5/3.), 0.0).r;   
}

// Function 70
bool keyIsDown( float key ) {
    return texture( iChannel3, vec2(key,0.75) ).x > .5;
}

// Function 71
float getKey(float key) {
	return texture(iChannel1, vec2(key,0.)/256.).x;   
}

// Function 72
float SampleKey(float key)
{
	return step(0.5, texture(iChannel1, vec2(key, 0.25)).x);
}

// Function 73
bool readKey(int value) {
    float keyVal = texture(iChannel0, vec2((float(value)+0.5)/256.0, 0.25)).x;
	return (keyVal > 0.5) ? true: false;
}

// Function 74
vec4 getInput(vec2 p){
    return textureLod(iChannel0, (p+.5)/iChannelResolution[0].xy,-100.);
}

// Function 75
float keyDown(int keyCode) {
    return texture(iChannel1, vec2((float(keyCode) + 0.5) / 256., .5/3.), 0.0).r;   
}

// Function 76
void handleKeys( inout vec4 buffer, in vec2 fragCoord )
{
    // Load keypress data from the buffer.
    vec4 priorKeys = readTexel(KEYS_BUFFER, txKEYS);
    
    // If a key is down, update the position accordingly.
    vec4 keys;
    keys.r = texture(KEYBOARD,vec2(KEY_W, KEY_DOWN_POS)).r;
    keys.g = texture(KEYBOARD,vec2(KEY_S, KEY_DOWN_POS)).r;
    keys.b = texture(KEYBOARD,vec2(KEY_A, KEY_DOWN_POS)).r;
    keys.a = texture(KEYBOARD,vec2(KEY_D, KEY_DOWN_POS)).r;
    
    // Store the keys value. That mix eases between full up and
    // full down states, so that the velocity of the camera
    // isn't as jarring.
    write4(buffer,mix(priorKeys,keys,.25),txKEYS,fragCoord);
}

// Function 77
bool readKey(int key, bool autorelease) {
    float value = texture(iChannel0,vec2((float(key)+0.5)/256.0,autorelease?0.75:0.25)).r;
    if (value > 0.5)
        return true;
    else
        return false;
}

// Function 78
bool isKeyPressed(float key)
{
	return texture( iChannel2, vec2(key, 0.25) ).x > .0;
}

// Function 79
vec2 KeyboardInput2() {
    INPUT_METHOD2
    
	vec2 i = vec2(key(KEY_BIND_RIGHT)   - key(KEY_BIND_LEFT), 
                  key(KEY_BIND_FORWARD) - key(KEY_BIND_BACKWARD));
    
    float n = abs(abs(i.x) - abs(i.y));
    return i * (n + (1.0 - n)*inversesqrt(2.0));
}

// Function 80
vec3 get_mouse(vec3 ro) {
    float x = iMouse.xy==vec2(0) ? -.2 :
    	(iMouse.y / iResolution.y * .5 - 0.25) * PI;
    float y = iMouse.xy==vec2(0) ? .0 :
    	-(iMouse.x / iResolution.x * 1.0 - .5) * PI;
    float z = 0.0;

    ro.zy *= r2(x);
    ro.zx *= r2(y);
    
    return ro;
}

// Function 81
bool isKeyReleased (in int key) {
    return bool(texelFetch(iChannel2, ivec2(key, 1), 0).x);
}

// Function 82
bool isKeyPressed(int KEY)
{
	return texelFetch( iChannel3, ivec2(KEY,0), 0 ).x > 0.5;
}

// Function 83
float getKey(float key)
{
    return texture(iChannel1, vec2(key, 0.25)).x;
}

// Function 84
bool keypress(int key) {
  return texelFetch(iChannel1,ivec2(key,2),0).x != 0.0;
}

// Function 85
float keyPressed(int keyCode) {
	return texture(iChannel0, vec2((float(keyCode) + 0.5) / 256., .5/3.)).r;   
}

// Function 86
bool keyboard(int t){ // 0-6 <^>vZXS
    float p = 0.;
    if(t<4)p = float(t)+37.5;
    if(t==4)p = 90.5;
    if(t==5)p = 88.5;
    if(t==6)p = 16.5;
    return texture(iChannel1,vec2(p/256.,0)).x > 0.5;
}

// Function 87
bool isKeyPressed(float key)
{
	return texture( iChannel1, vec2(key, 0.0) ).x > 0.0;
}

// Function 88
void getMouse( inout vec3 p ) {
    float x = M.xy == vec2(0) ? 0. : -(M.y/R.y * .25 - .125) * PI;
    float y = M.xy == vec2(0) ? 0. : (M.x/R.x * .25 - .125) * PI;
    p.zy *=r2(x);
    p.xz *=r2(y);   
}

// Function 89
bool isKeyEnabled (in int key) {
    return bool(texelFetch(iChannel1, ivec2(key, 2), 0).x);
}

// Function 90
float keyClick(int ascii) {
	return float(texture(iChannel1,vec2((.5+float(ascii))/256.,0.25)).x > 0.0);
}

// Function 91
if keyDown(64+3) {                // 'C'
                    i++;
                    if ( p > N && p <= N+i-str )
                        O = T(pos(str+p-N-1.));   // clone
                    if (U==vec2(0)) O.x += i-str;
                    return;
                }

// Function 92
bool keyToggle(int ascii) 
{ return (texture(iChannel2,vec2((.5+float(ascii))/256.,0.75)).x > 0.); }

// Function 93
void mouseDownEvt(int index, vec4 mouse, int capturedIndex) {
    //  mouseDown event:
    switch (controls[index].type) {
        case PUSH_BTN_T:
        controls[index].value = 1.;
        break;

        case TOGGLE_BTN_T:
        controls[index].value = 1. - controls[index].value;
        break;

        case CLICKBOX_T:
	    vec2 dr = hitCoordsNormalized(index, mouse.xy) / 2.;
        controls[index].value = normCoordsToClickboxVal(dr);
        controls[index].value2 = controls[index].value;
        break;        
    }
}

// Function 94
void keydown_event()
{
	vec2 uv = (-iResolution.xy + 2. * iMouse.xy) / iResolution.y;
    float asp = iResolution.x/iResolution.y;
    
    //draw map check box
    if(inbox(uv, vec2(-asp*0.9, 0.9), vec2(asp*0.0144)) > 0)
    {
    	SET_DRAW_MAP(GET_DRAW_MAP > 0. ? 0.:1.);
    }
    
    //draw start check box
    else if(inbox(uv, vec2(-asp*0.9, 0.78), vec2(asp*0.0144)) > 0)
    {
    	SET_DRAW_START(GET_DRAW_START > 0. ? 0.:1.);
    }
    
    //show target point check box
    else if(inbox(uv, vec2(-asp*0.9, 0.66), vec2(asp*0.0144)) > 0)
    {
    	SET_SHOW_TAR(GET_SHOW_TAR > 0. ? 0.:1.);
    }
    
    //fix origin point check box
    else if(inbox(uv, vec2(-asp*0.9, 0.54), vec2(asp*0.0144)) > 0)
    {
    	FIX_ORI = FIX_ORI > 0. ? 0.:1.;
    }
    
    //fix target point check box
    else if(inbox(uv, vec2(-asp*0.9, 0.42), vec2(asp*0.0144)) > 0)
    {
    	FIX_TAR = FIX_TAR > 0. ? 0.:1.;
    }
    
    //increase step count
    else if(inbox(uv, vec2(-asp*0.9, 0.30), vec2(asp*0.0144)) > 0)
    {
    	SET_STEP_COUNT(clamp(GET_STEP_COUNT+1., MIN_STEP_COUNT, MAX_STEP_COUNT));
    }
    
    //decrease step count
    else if(inbox(uv, vec2(-asp*0.77, 0.30), vec2(asp*0.0144)) > 0)
    {
    	SET_STEP_COUNT(clamp(GET_STEP_COUNT-1., MIN_STEP_COUNT, MAX_STEP_COUNT));
    }
    
    //increase map count
    else if(inbox(uv, vec2(-asp*0.9, 0.18), vec2(asp*0.0144)) > 0)
    {
    	SET_MAP_NUM(clamp(GET_MAP_NUM+1., MIN_MAP_COUNT, MAX_MAP_COUNT));
    }
    
    //decrease map count
    else if(inbox(uv, vec2(-asp*0.77, 0.18), vec2(asp*0.0144)) > 0)
    {
    	SET_MAP_NUM(clamp(GET_MAP_NUM-1., MIN_MAP_COUNT, MAX_MAP_COUNT));
    }
    
    //move point
    else if(!(uv.x < -asp*0.5 && uv.y > 0.1))
    {
        if(!is(FIX_ORI) && !is(FIX_TAR)){
            uv = clamp(uv, vec2(-asp*0.5, -asp), vec2(asp, asp));
            SET_ORIGIN(uv);
            SET_TARGET(uv);
        }
        KEY_DRAG_FLAG = 1.0;
    }
}

// Function 95
float ReadKey( int key )
{
   	return step(.5,texture( iChannel3, vec2( (float(key)+.5)/256.0, .25)).x);
}

// Function 96
vec4 DrawKeyGlow( vec2 uv, float keyID, vec2 size, float lifeTime)
{
    float lifeSpan = 0.3;
    float h = 0.3;
    lifeTime = clamp(lifeTime,0.,lifeSpan)/lifeSpan;
    lifeTime=lifeTime*lifeTime;
    size.y = (1.-lifeTime)*h;
    vec4 ret = vec4(0.);
    vec2 pos = vec2(keyID*0.2+0.1, 0.);
    vec2 p = (uv - pos);
    float t = 0.;
    if ( abs(p).x < size.x/2. && abs(p).y < size.y )
    {
        ret.a = 1.;
        vec3 col =  vec3(0.4,0.5,0.7);
        float a = (1.0-(p.y/h));
        ret.xyz =col*a*a*a;
        //ret.xyz = vec3(0.4,0.5,0.7);//*pow((1.-lifeTime),20.);
           
    }
    
    return abs(ret);
}

// Function 97
bool ReadKey(int key)
{
	float keyVal = texture( iChannel2, vec2( (float(key)+.5)/256.0, .25 ) ).x;
	return keyVal > 0.0;
}

// Function 98
float ReadKeyInternal( int key, bool toggle )
{
	return textureLod( iChannel3, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ), 0.0 ).x;
}

// Function 99
bool isKeyDown(float key) 
{
  return texture(iChannel1, vec2(key, 0.5)).x > 0.5;
}

// Function 100
vec4 GetLastMouseDelta(){return texelFetch(iChannel0,LastMouseDelta,0);}

// Function 101
void ProcessInput()
{
    const float WaterIorChangeRate = 0.35;
	if(KeyDown(87)) WaterIor += WaterIorChangeRate * iTimeDelta;
    if(KeyDown(83)) WaterIor -= WaterIorChangeRate * iTimeDelta;
    WaterIor = clamp(WaterIor, 1.0, 1.8);
    
    const float WaterTurbulanceChangeRate = 7.0;
	if(KeyDown(69)) WaterTurbulence += WaterTurbulanceChangeRate * iTimeDelta;
    if(KeyDown(68)) WaterTurbulence -= WaterTurbulanceChangeRate * iTimeDelta;
    WaterTurbulence = clamp(WaterTurbulence, 0.0, 50.0);
       
    const float WaterAbsorptionChangeRate = 0.03;
	if(KeyDown(81)) WaterAbsorption += WaterAbsorptionChangeRate * iTimeDelta;
    if(KeyDown(65)) WaterAbsorption -= WaterAbsorptionChangeRate * iTimeDelta;
    WaterAbsorption = clamp(WaterAbsorption, 0.0, 1.0);
    
    const float ColorChangeRate = 0.5;
	if(KeyDown(89)) WaterColor.r += ColorChangeRate * iTimeDelta;
    if(KeyDown(72)) WaterColor.r -= ColorChangeRate * iTimeDelta;
    
    if(KeyDown(85)) WaterColor.g += ColorChangeRate * iTimeDelta;
    if(KeyDown(74)) WaterColor.g -= ColorChangeRate * iTimeDelta;
    
    if(KeyDown(73)) WaterColor.b += ColorChangeRate * iTimeDelta;
    if(KeyDown(75)) WaterColor.b -= ColorChangeRate * iTimeDelta;
    
    WaterColor = clamp(WaterColor, 0.05, 0.99);
}

// Function 102
float isKeyPressed(float key)
{
	return texture( iChannel1, vec2(key, 0.5) ).x;
}

// Function 103
bool
    mouseClicked( ) {
        
        return 0. < iMouse.z;
    }

// Function 104
if keyToggle(32) {                  // === edit mode
        if (U==vec2(0)) O.y *= float( iMouse.z > 0.) ;
        if ( iMouse.z > 0. ) {
            if ( U==vec2(0)) {                    // --- picking a stroke
                if (T(0).y == 0.) {
                    vec4 P = T(pos(1.)), P_;
                    float d = 1e9, s, l, im = -1., str = 1., strm = -1., stri = 1., strmi = -1.;

                    for(float i=2.; i < T(0).x; i++) { // parse recorded mouse pos
                        P_ = T(pos(i));
                        if ( P.z > 0. && P_.z > 0.) {
                            l = line (M, P.xy, P_.xy, s);
                            if ( l < d ) d = l, im = i, strm = str, strmi=stri;
                        } 
                        P = P_;
                        if ( P.z < 0. ) str++, stri = i+1.;// cur stroke
                    }
                    O.y = d <.02 ? strmi+ 1.: 0.; // picked stroke start                  
                }
                O.zw = M;                         // cur mouse pos
            }
          // else // commented because of 'C'
            {                         // --- action on current stroke
                float str = T(0).y-1., N = T(0).x,
                        p = index(U), i; 
                if (str < 0.) return; // || index(U) < str) return;

                vec2 G = vec2(0);
                for(i=str; i < N && T(pos(i)).z > 0.; i++) // parse cur stroke
                    G += T(pos(i)).xy;
                
                if keyDown(64+3) {                // 'C'
                    i++;
                    if ( p > N && p <= N+i-str )
                        O = T(pos(str+p-N-1.));   // clone
                    if (U==vec2(0)) O.x += i-str;
                    return;
                }
                if (U==vec2(0)) return; // why not included below ?
                if (p < str || i < p ) return; 
                G /= i-str;                       // barycenter
                
                vec2 M0 = T(0).zw;                // we are on cur stroke
                if ( keyClick(8) || keyClick(46) )// 'DEL'
                     O.xy = vec2(-1);             // delete
                else if keyClick(64+19)           // 'S'
                    O.xy  = (O.xy-G) * (M-G)/(M0-G) + G; // scale
                else if keyClick(64+18)           // 'R'
                { M-=G; M0-=G; 
                  float a = atan(M.y,M.x) - atan(M0.y,M0.x);
                  O.xy  = (O.xy-G) * rot(a) + G;  // rotate
                }
                else
                    O.xy += M-M0;                 // move
            }
        }}

// Function 105
void keyInput()
{
  if (iFrame > 9)
  {
    animate_pattern   = !ReadKey(KEY_A, true);
    show_reflections  = !ReadKey(KEY_F, true);
    rotation_scene    = !ReadKey(KEY_R, true);
    cross_eye_view    = !ReadKey(KEY_S, true);
  }
}

// Function 106
bool readKey(int keyCode)
{
	bool toggle = false;
    vec2 uv = vec2((float(keyCode)+.5)/256., toggle?.75:.25);
	float keyVal = textureLod(iChannel3,uv,0.).x;
    return keyVal>.5;
}

// Function 107
bool toggleKey(int key)
{
	return texture(iChannel3, vec2((float(key) + 0.5) / 256.0, 0.75)).x > 0.0;
}

// Function 108
bool Key_IsPressed( sampler2D samp, int key )
{
    return texelFetch( samp, ivec2(key, 0), 0 ).x > 0.0;    
}

// Function 109
float keystate( int key )
	{ return 0.; }

// Function 110
float keyDown(int keyCode) {
	return textureLod(iChannel2, vec2((float(keyCode) + 0.5) / 256., .5/3.), 0.0).r;   
}

// Function 111
vec4 setInput(vec2 p){
    if (iFrame == 0){
        return hash44(hash44(iDate)+hash44(vec4(p,75,43)));
    }
    return getInput(p) - lr*getG(0,int(p.y),vec2(0));
}

// Function 112
vec2 readMousePos()
{
    return getPixel(0,0).xy;
}

// Function 113
vec2 KeyboardInput() {
    INPUT_METHOD
    
	return vec2(key(KEY_BIND_RIGHT)   - key(KEY_BIND_LEFT), 
                key(KEY_BIND_FORWARD) - key(KEY_BIND_BACKWARD));
}

// Function 114
void LoadInputs(out Inputs inp)
{
    inp.button = iMouse.z >= 0.;
    inp.mouse = iMouse.xy;
    inp.attract = !inp.button && dot(inp.mouse, inp.mouse) < 4.;
    if (inp.attract) { // icon?
        inp.mouse.x = .02*iTime; // slow spin
        inp.mouse.y = iResolution.y*.5; // don't look at ground
    }
    inp.move =
      vec3(key(KEY_RT) - key(KEY_LF)
         , key(KEY_UW) - key(KEY_DW)
         , key(KEY_FW) - key(KEY_BW))
    + vec3(key(KEY_RIGHT) - key(KEY_LEFT)
         , key(KEY_PGUP ) - key(KEY_PGDN)
         , key(KEY_UP   ) - key(KEY_DOWN)) // arrows alternate controls
      ;
    inp.turn = vec2(key(KEY_TURNR) - key(KEY_TURNL), 0);
    inp.dt = iTimeDelta;
}

// Function 115
bool isKeyToggled(int keyCode) { return texelFetch(iChannel3, ivec2(keyCode, 2), 0).x == 1.0; }

// Function 116
float keySinglePress(int keycode) {
    bool now = bool(keyDown(keycode));
    bool previous = bool(textureLod(iChannel0, vec2(256. + float(keycode) + 0.5, 0.5) / iResolution.xy, 0.0).r);
    return float(now && !previous);
}

// Function 117
float is_key_pressed(float key_code)
{
    return textureLod(iChannel1, vec2((key_code), 0.0),0.0).x;
}

// Function 118
void KeyState( sampler2D sampler, out float[5] keys)    
{
    
    keys[0] = texelFetch(sampler, ivec2(68, 0),0).x;
    keys[1] = texelFetch(sampler, ivec2(70, 0),0).x;
    keys[2]  = texelFetch(sampler, ivec2(32, 0),0).x;    
    keys[3] = texelFetch(sampler, ivec2(74, 0),0).x;
    keys[4]  = texelFetch(sampler, ivec2(75, 0),0).x;

}

// Function 119
float getkey(int x, int y)
{
    return texelFetch(iChannel1,ivec2(x,y),0).x;
}

// Function 120
float ReadKey(int keyCode) {return texelFetch(KeyBoard, ivec2(keyCode, 0), 0).x;}

// Function 121
bool ReadKey(int key)
{
	float keyVal = texture(iChannel1, vec2((float(key)+.5)/256.0, 0.2)).x;
	return (keyVal>.5)?true:false;
}

// Function 122
vec2 RotWithInput(vec2 uv) 
{
    float rot = - MOUSE.x*0.1 - ROTMOUSE*0.3 + ROT - PI - IDLE2; // mouse, idle, keys
    return mat2(cos(rot), -sin(rot), sin(rot), cos(rot)) * uv; 
}

// Function 123
float key_state(int key) {
	return textureLod(iChannel3, vec2((float(key) + .5) / 256.0, .25), 0.0).x;
}

// Function 124
float keyToggled(int keyCode) {
    return texture(iChannel1, vec2((float(keyCode) + 0.5) / 256., 2.5/3.), 0.0).r;   
}

// Function 125
bool key(int k)
{
    return texelFetch(Keyboard, ivec2(k,0), 0).x >= .5;
}

// Function 126
vec3 GetLastMouseClick()
{
    return texelFetch(iChannel0, LastMouseClick, 0).xyz;
}

// Function 127
float isKeyPressed(float key)
{
	return texture( iChannel0, vec2(key, 0.0) ).x;
}

// Function 128
vec3 get_mouse(vec3 ro) {
    ro.zy *= r2(mx());
    ro.zx *= r2(my());
    return ro;
}

// Function 129
vec2 get_mouse()
{
    return texture(iChannel2, (addr_mouse + vec2(0.5, 0.5))/iResolution.xy).rg;
}

// Function 130
float keyToggle(int ascii) {
	return float(texture(iChannel1,vec2((.5+float(ascii))/256.,0.75)).x > 0.0);
}

// Function 131
bool KeyIsPressed(float key)
{
	return texture( iChannel1, vec2(key, 0.0) ).x > 0.0;
}

// Function 132
float mouseSelect(vec2 c,float h) {
	float scale = floor(iResolution.y / 128.);
    c /= scale;
    vec2 r = iResolution.xy / scale;
    float xStart = (r.x - 16. * NUM_ITEMS) / 2.;
    c.x -= xStart;
    if (c.x <NUM_ITEMS * 16. && c.x >= 0. && c.y < 16.*h) {
        float slot = floor(c.x / 16.) + NUM_ITEMS*floor(c.y / 16.);
    	return slot;
    }

    return -1.;
}

// Function 133
bool IsKeyPressed(float key) {
    return bool(texture( iChannel1, vec2(key, 0.5) ).x);
}

// Function 134
if keyClick(32) { O.xy = M-M; return; }

// Function 135
float keySinglePress(int keycode) {
	bool now = bool(keyDown(keycode));
    bool previous = bool(textureLod(iChannel0, vec2(256. + float(keycode) + 0.5, 0.5) / iResolution.xy, 0.0).r);
    return float(now && !previous);
}

// Function 136
vec2 mouseDelta(){
    vec2 pixelSize = 1. / iResolution.xy;
    float eighth = 1./8.;
    vec4 oldMouse = Cell(2);
    vec4 nowMouse = vec4(iMouse.xy / iResolution.xy, iMouse.zw / iResolution.xy);
    if(oldMouse.z > pixelSize.x && oldMouse.w > pixelSize.y && 
       nowMouse.z > pixelSize.x && nowMouse.w > pixelSize.y)
    {
        return nowMouse.xy - oldMouse.xy;
    }
    return vec2(0.);
}

// Function 137
void WriteMousePos(float ytext, vec2 mPos)
{
  int digits = 3;
  float radius = resolution.x / 200.;

  // print dot at mPos.xy
  if (iMouse.z > 0.0) dotColor = mpColor;
  float r = length(mPos.xy - pixelPos) - radius;
  vColor += mix(vec3(0), dotColor, (1.0 - clamp(r, 0.0, 1.0)));

  // print first mouse value
  SetTextPosition(1., ytext);

  // print mouse position
  if (ytext == 7.)
  {
    drawColor = mxColor;
    WriteFloat(mPos.x,6,3);
    BLANK;
    drawColor = myColor;
    WriteFloat(mPos.y,6,3);
  }
  else
  {
    drawColor = mxColor;
    WriteInteger(int(mPos.x));
    BLANK;
    drawColor = myColor;
    WriteInteger(int(mPos.y));
  }
}

// Function 138
float keystatepress( int key )
	{ return 0.; }

// Function 139
int keycount(int key) {
  return 0;
}

// Function 140
float keyToggled(int keyCode) {
	return textureLod(iChannel1, vec2((float(keyCode) + 0.5) / 256., 2.5/3.), 0.0).r;   
}

// Function 141
void LoadInputs(out Inputs inp)
{
    inp.button = iMouse.z >= 0.;
    inp.mouse = iMouse.xy;
    if (iMouse.xyz == vec3(0)) // icon? //dot(iMouse,iMouse) < 1e-3) //
        inp.mouse.y = iResolution.y*.5; // don't look at ground
    inp.move = vec3(key(KEY_RT) - key(KEY_LF)
                  , key(KEY_UW) - key(KEY_DW)
                  , key(KEY_FW) - key(KEY_BW));
    inp.dt = iTimeDelta;
}

// Function 142
bool isKeyPressed(float key, int status) {
    float x = key*256.0;
    vec4 t = texelFetch(iChannel1, ivec2(x,float(status)),0);
    if (t.x > 0.0) {
        return true;
    } else {
        return false;
    }
}

// Function 143
vec4 iMouseZwFix(vec4 m,bool NewCoke
 ){if(m.z>0.){ //while mouse down
    if(m.w>0.)return m;//mouse was clicked in THIS     iFrame 
    else m.w=-m.w      //mosue was clicked in previous iFrame
    //remember, MouseDrag advances the iFrame Count, even while paused !!
 ;}else{
    if(!NewCoke||m.w>0.)return m.xyxy; //OPTIONAL onMouseUp (fold or whatever)
    m.zw=-m.zw;}
  return m;}

// Function 144
bool keyToggle(int ascii) 
{	return (texture(iChannel2,vec2((.5+float(ascii))/256.,0.75)).x > 0.); }

// Function 145
vec2 keys(vec2 uv){
    vec2 result=vec2(0.);
 if(0.<texelFetch( iChannel2, ivec2(KEY_LEFT,0), 0 ).x){
 result+=vec2(-1.,0.);
 }
 if(0.<texelFetch( iChannel2, ivec2(KEY_RIGHT,0), 0 ).x){
 result+=vec2(1.,0.);
 }
 if(0.<texelFetch( iChannel2, ivec2(KEY_UP,0), 0 ).x){
     result+=vec2(0.,1.);
 }
 if(0.<texelFetch( iChannel2, ivec2(KEY_DOWN,0), 0 ).x){
     result+=vec2(0.,-1.);
 }
    return result/40.;
}

// Function 146
bool keyboard (int keycode, int evt) {
  return texelFetch(CH_KEYB, ivec2(keycode,evt), 0).r>.5;
}

// Function 147
bool isKeyPressed(float key) {
    return isKeyPressed(key, 0);
}

// Function 148
float keyDn(float a){return texture(iChannel1,vec2(a,.2)).x;}

// Function 149
vec4 onMouseDrag(vec4 color, vec2 fc, vec2 uv) {
    
    vec2 vppos = vec2(getfp(6),getfp(7));
    vec2 scale = vec2(getfp(3),getfp(4));
    vec2 cmouse = iMouse.xy;
    vec2 pmouse = vec2(float(geti(1)),float(geti(2)));
    
    
    if(isKeyDown(KEY_CTRL)) {
        //zoom with mouse
        float zdelta = scale.y * (pmouse.y-cmouse.y);
        scale += zdelta * iTimeDelta;// / 64.; //double pix size = 64 px
    } else {
        //pan with mouse
        vec2 vpdelta = scale * (pmouse-cmouse);
    	vppos += vpdelta;
    }
    
    switch(int(uv.x*stor_len)) {
        case _vpscl_x: {
            //vp scale x
            color = encfp(scale.x);
            break;
        }
        case _vpscl_y: {
            //vp scale y
            color = encfp(scale.y);
            break;
        }
        case _vppos_x: {
            //vp pos x
            color = encfp(vppos.x);
            break;
        }
        case _vppos_y: {
            //vp pos y
            color = encfp(vppos.y);
            break;
        }
        default: break;
    }
    return color;
}

// Function 150
float keyDif(vec2 a){a/=255.
 ;return texture(iChannel1,vec2(a.x,.2)).x
        -texture(iChannel1,vec2(a.y,.2)).x;}

// Function 151
mat4 look_around_mouse_control( mat4 camera, float pitch, float tan_half_fovy, vec3 aResolution, vec4 aMouse, float dmmx )
{
	float mouse_ctrl = 1.0;
	vec2 mm_offset = vec2( dmmx, pitch );
	vec2 mm = vec2( 0.0, 0.0 );

#ifndef EXTRA_3D_CAMERA
	if ( aMouse.z > 0.0 || STICKY_MOUSE ) mm = ( aMouse.xy - aResolution.xy * 0.5 ) / ( min( aResolution.x, aResolution.y ) * 0.5 );
#endif

	mm.x = -mm.x;
	mm = sign( mm ) * pow( abs( mm ), vec2( 0.9 ) );
	mm *= PI * tan_half_fovy * mouse_ctrl;
	mm += mm_offset;

	return camera * yup_spherical_coords_to_matrix( mm.y, mm.x );
}

// Function 152
bool keypress(int code) {
#if __VERSION__ < 300
    return false;
#else
    return texelFetch(iChannel1, ivec2(code,2),0).x != 0.0;
#endif
}

// Function 153
float keydown(int code)
{
    return texelFetch(iChannel0, ivec2(code, 0),  0).x;
}

// Function 154
bool Key_IsPressed( sampler2D samp, int key)
{
    return texelFetch( samp, ivec2(key, 0), 0 ).x > 0.0;    
}

// Function 155
vec3 getMouseColorAdd(vec2 fragCoord)
{
    if (iMouse.z > 1.0)
    {
        float dist = min(MOUSE_MAX_DIST, distance(fragCoord, iMouse.xy)),
              distLerp = dist / MOUSE_MAX_DIST;
        return (1.0 - distLerp) *
            	vec3(0.5 + (0.5 * sin(distLerp * 5.0 + (iTime * 20.0))));
    }
    return vec3(0.0);
}

// Function 156
vec2 mouseDelta(){
    vec2 pixelSize = 1. / iResolution.xy;
    float eighth = 1./8.;
    vec4 oldMouse = texture(iChannel2, vec2(7.5 * eighth, 2.5 * eighth));
    vec4 nowMouse = vec4(iMouse.xy / iResolution.xy, iMouse.zw / iResolution.xy);
    if(oldMouse.z > pixelSize.x && oldMouse.w > pixelSize.y && 
       nowMouse.z > pixelSize.x && nowMouse.w > pixelSize.y)
    {
        return nowMouse.xy - oldMouse.xy;
    }
    return vec2(0.);
}

// Function 157
vec4 GetLastMouseDelta()
{
    return texelFetch(iChannel0, LastMouseDelta, 0);
}

// Function 158
if keypress(32)
    {
        val *= .5;
        val += tex(p*2.) *.25;
        val += tex(p*4.) *.125;
        val += tex(p*8.) *.0625;
    }

// Function 159
void process_mouse(out vec4 fragColor) {
    fragColor = vec4(iMouse.xy / iResolution.xy, 0, iMouse.w);
}

// Function 160
vec3 LevelsControlInput(vec3 color, float minInput, float gamma, float maxInput)
{
    return GammaCorrection(LevelsControlInputRange(color, minInput, maxInput), gamma);
}

// Function 161
bool key_press(int key) {
    return texelFetch(iChannel3, ivec2(key, 1), 0).x != 0.;
}

// Function 162
vec3 GetLastMouse()
{
    return texture(iChannel1, I2UV(0, iResolution.xy)).xyz;
}

// Function 163
bool key(int code) {
  return texelFetch(iChannel0, ivec2(code,2),0).x != 0.0;
}

// Function 164
bool keypress(int key) {
   return texelFetch(iChannel0, ivec2(key,2),0).x != 0.0;
}

// Function 165
vec2 Mouse(){
    vec2 r = (2.0 * iMouse.xy / iResolution.xy) - 1.0;
    r.x *= iResolution.x / iResolution.y;
    return r;
}

// Function 166
bool KeyIsToggled(int key)
{
	return texelFetch( iChannel1, ivec2(key, 2), 0 ).x > 0.0;
}

// Function 167
void getMouse( inout vec3 p ) {
    float x = M.xy == vec2(0) ? 0. : -(M.y/R.y * .65 - .325) * PI;
    float y = M.xy == vec2(0) ? 0. :  (M.x/R.x * .45 - .225) * PI;
    p.zy *=r2(x);
    p.xz *=r2(y);   
}

// Function 168
bool isKeyPressed(float key)
{
	return texture( iChannel1, vec2(key, 0.3) ).x > 0.5;
}

// Function 169
float getActKey()
{
    float shift=0.;
    if(texture(iChannel3,vec2((SHIFT_+.5)/256.,(0.+.5)/3.)).x>.5) shift=1.;
    if(texture(iChannel3,vec2((BACKSPACE_+.5)/256.,(1.+.5)/3.)).x>.5) return BACKSPACE_+shift*256.;
    if(texture(iChannel3,vec2((DEL_  +.5)/256.,(1.+.5)/3.)).x>.5) return DEL_;
    if(texture(iChannel3,vec2((ENTER_+.5)/256.,(1.+.5)/3.)).x>.5) return ENTER_;
    if(texture(iChannel3,vec2((space_+.5)/256.,(1.+.5)/3.)).x>.5) return space_;
    for(float x=A_;x<A_+26.;x+=1.)
    {
        if(texture(iChannel3,vec2((x+.5)/256.,(1.+.5)/3.)).x>.5) return x+32.*(1.-shift);
    }
    for(float x=D0_;x<D_+10.;x+=1.)
    {
        if(texture(iChannel3,vec2((x+.5)/256.,(1.+.5)/3.)).x>.5) return x;
    }
    return 0.;
}

// Function 170
float keyPress(int ascii) {
	return texture(iChannel2,vec2((.5+float(ascii))/256.,0.25)).x ;
}

// Function 171
bool KeyIsPressed(int ascii) 
{
	return (texture(iChannel1,vec2((.5+float(ascii))/256.,0.25)).x > 0.);
}

// Function 172
vec3 LevelsControlInputRange(vec3 color, float minInput, float maxInput)
{
    return min(max(color - vec3(minInput), vec3(0.0)) / (vec3(maxInput) - vec3(minInput)), vec3(1.0));
}

// Function 173
vec4 get_mouse_consistent(vec4 m) {
    if (m.z <= 0.0) {
        return m;
    } else {
        return vec4(m.xyz, -m.w);
    }
}

// Function 174
void checkKeys(){
	isRedLeft = texture(iChannel0,vec2(RED_LEFT,0.5)).r >0.5;
    isRedRight = texture(iChannel0,vec2(RED_RIGHT,0.5)).r >0.5;
    isYellowLeft = texture(iChannel0,vec2(YELLOW_LEFT,0.5)).r >0.5;
    isYellowRight = texture(iChannel0,vec2(YELLOW_RIGHT,0.5)).r >0.5;
    isGreenUp = texture(iChannel0,vec2(GREEN_UP,0.5)).r >0.5;
    isGreenDown = texture(iChannel0,vec2(GREEN_DOWN,0.5)).r >0.5;
    isBlueUp = texture(iChannel0,vec2(BLUE_UP,0.5)).r >0.5;
    isBlueDown = texture(iChannel0,vec2(BLUE_DOWN,0.5)).r >0.5;    
}

// Function 175
void setInput(inout vec4 fragColor, inout vec2 fragCoord) {
    //shadertoy covers up 3 pixels on each top corner, use them for state storage
    if (fragCoord.y > iResolution.y - 1.) {
        //store/read location
        if (fragCoord.x < 1.) {
            fragColor = INIT_POS;
        //store/read velocity
        } else if (fragCoord.x < 2.) {
    	  	fragColor = INIT_VEL;
        //store/read rotation
        } else if (fragCoord.x > iResolution.x - 1.) {
        	fragColor = INIT_ROT;
        //store/read mouse
        } else if (fragCoord.x > iResolution.x - 2.) {
            fragColor = INIT_ROT;
        }
    }
}

// Function 176
if keyDown(i) {
                i = i-minKey +(maxKey>96. && i>64. && keyClick(16)?32.:0.); 
                // optional: entry validation.       \ special case for shift (16) on letters :
                O.x = i;
                return; 
            }

// Function 177
vec2 get_mouse(void) {
    float ax = (iMouse.x - iResolution.x/2.)*.1;
    float ay = (iMouse.y - iResolution.y/2.)*.1;  
    return vec2(ax,ay);
}

// Function 178
float keystate( int key )
	{ return texelFetch( iChannel3, ivec2( key, 0 ), 0 ).x; }

// Function 179
if keyToggle(32) {                    // --- edit mode
        O.r += .2; O.gb *= .8; 
      //if (index(floor(U*R.y)) < N) { O.g++; return; }
        float i, str = T(0).y-1.; 
        if ( str<0. || ii < str ) return;

        for( i=str; i < N && T(pos(i)).z>0.; i++); // check if on current stroke
        if ( ii <= i ) O.gb *= 0.;        // cur stroke in red

    }

// Function 180
float is_key_pressed(int code)			{ return code != 0 ? texelFetch(iChannel0, ivec2(code, 1), 0).r : 0.; }

// Function 181
float is_key_down(int code)				{ return code != 0 ? texelFetch(iChannel0, ivec2(code, 0), 0).r : 0.; }

// Function 182
void LoadKeyboardState()
{
    vec4 previousKeyboardState = LoadValue(txPreviousKeyboard);
    
    // current keys
    gKeyboardState.mKeyModeForward[0]   = GeyKeyState(KEY_D) || GeyKeyState(KEY_RIGHT);
    gKeyboardState.mKeyModeBackwards[0] = GeyKeyState(KEY_A) || GeyKeyState(KEY_LEFT);
    gKeyboardState.mKeyW[0] = GeyKeyState(KEY_W);
    gKeyboardState.mKeyS[0] = GeyKeyState(KEY_S);
  
    // previous keys
    gKeyboardState.mKeyModeForward[1] 	= (previousKeyboardState.x > 0.0);
    gKeyboardState.mKeyModeBackwards[1] = (previousKeyboardState.y > 0.0);
    gKeyboardState.mKeyW[1] = (previousKeyboardState.z > 0.0);
    gKeyboardState.mKeyS[1] = (previousKeyboardState.w > 0.0);
}

// Function 183
bool keyState(int key) { return keys(key, 2).x < 0.5; }

// Function 184
bool GeyKeyState(in float key)
{
	return (texture(iChannel1, vec2(key, 0.25)).x > 0.5);   
}

// Function 185
float key(float x) {return clamp((sin(x*3.0 - iTime)-0.97)*100.0,0.0,1.0);}

// Function 186
bool keyPressed(int keyCode) {
	float startTime = 0.;
    if (iResolution.x == 288.) startTime += 10.;
    if (iTime <0.1 + startTime && keyCode == 32 && START_PATTERN != 6) return true;
    return bool(texture(iChannel1, vec2((float(keyCode) + 0.5) / 256., .5/3.)).r);   
}

// Function 187
bool keyPress( int ascii ) 
{
	return ( texture( iChannel3, 
	                vec2( ( 0.5 + float( ascii ) ) / 256.0, 0.25 ) ).x > 0.0 );
}

// Function 188
bool keyP(vec2 u){
 return (t0(u).x!=0.);
 //return (t0(u+vec2((key)/iResolution.x,status)).x!=0.);
}

// Function 189
int key_control() {
    if (key_press(KEY_LEFT)) {
        return left_l;
    }
    if (key_press(KEY_RIGHT)) {
        return right_l;
    }
    if (key_press(KEY_UP)) {
        return rotate_l;
    }
    if (key_state(KEY_DOWN)) {
        return down_l;
    }
    return nac;
}

// Function 190
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

// Function 191
vec2 mouse(float xmin, float xmax, float ymin, float ymax)
{
    vec2 xy = iMouse.xy;
    xy.x = map(xy.x, 0.0, iResolution.x, xmin, xmax);
    xy.y = map(xy.y, 0.0, iResolution.y, ymin, ymax);
    return xy;
}

// Function 192
bool Key_IsPressed(float key)
{
    return texture( iChannel1, vec2(key, 0.0) ).x > 0.0;
}

// Function 193
float isKeyPressed(float key)
{
	return texture( iChannel1, vec2(key, 1.0) ).x;
}

// Function 194
bool isKeyPressed(int KEY)
{
	return texelFetch( ch3, ivec2(KEY,0), 0 ).x > 0.5;
}

// Function 195
float keyDown(float key)
{
    return texture(iChannel1, vec2(key / 256.0, 0.2)).x;
}

// Function 196
vec3 chromaKey(vec3 x, vec3 y){
	vec2 c = s(vec2(x.g - x.r * x.y, x.g));
    
    return mix(x, y, c.x * c.y);
}

// Function 197
rayCastResults  getMouseRay(){
       
   vec4 mouseRay=  texture(iChannel3, vec2(0.));
   rayCastResults res;
   res.hit = mouseRay.a!=0.;
   res.mapPos = mouseRay.rgb;
    
   float eN = mouseRay.a -1.;
   res.normal=vec3(mod(eN,3.),floor(mod(eN,9.)/3.),floor(eN/9.))- vec3(1.);  
   return res;
}

// Function 198
float keys(int i) {
    return texelFetch(iChannel1,ivec2(i,0),0).x;
}

// Function 199
vec3 withMouse(vec3 p) {
        vec2 mouse = iMouse.xy/iResolution.xy;
        float a = max(mouse.y, .505) * 3.14 * 2.;
        p.yz *= rot2d(a);
        float aa = mouse.x * 3.14 * 2.;
        p.xy *= rot2d(aa);
        return p;
}

// Function 200
bool isKeyPressed(float key)
{
	return texture(iChannel3, vec2(key, 0.25) ).x > .0;
}

// Function 201
bool keydown(int code)
{
    return texelFetch(iChannel0, ivec2(code, 0),  0).x > 0.;
}

// Function 202
bool checkKey(float key)
{
	return texture(iChannel1, vec2(key, 0.25)).x > 0.5;
}

// Function 203
mat3 fetchMouseRotation(){
    float pi = atan(1.0) * 4.0;
float tau = atan(1.0) * 8.0;
    
        
    //Camera stuff   
    vec3 angles = vec3(0);
    
    if(iMouse.xy == vec2(0,0))
    {
        angles.y = tau * (1.5 / 8.0);
        angles.x = iTime * 0.1;
    }
    else
    {    
    	angles = vec3((1.-(iMouse.xy*vec2(1.,0.)+vec2(0.,iResolution.y)) / iResolution.xy) * pi, 0);
        angles.xy *= vec2(2.0, 1.0);
    }
    
    //angles.y = clamp(angles.y, 0.0, tau / 4.0);
    return Rotate(vec3(0.,0.,-pi/2.))*Rotate(angles.yxz);
}

// Function 204
bool ReadKey( int key, bool toggle )
{
	float keyVal = texture( iChannel1, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ) ).x;
	return (keyVal>.5)?true:false;
}

// Function 205
bool key_state( float ascii ) { return (texture( iChannel0, vec2( ( ascii + .5 ) / 256., 0.25 ) ).x > 0.); }

// Function 206
bool keypress(int key) {
  return texelFetch(iChannel0, ivec2(key,2),0).x != 0.0;
}

// Function 207
bool readKey(in int keyCode)
{
	bool toggle = false;
    vec2 uv = vec2((float(keyCode)+.5)/256., toggle?.75:.25);
	float keyVal = textureLod(iChannel3,uv,0.).x;
    return keyVal>.5;
}

// Function 208
mat3 mouselook(vec4 mouse)
{
    vec3 f = vec3(0, 0, 1);
    vec2 m;
    if (length(mouse.xy) < 1e-2) {
        m = vec2(0);
    	m.x += 1.21;
    } else {
        m = mouse.xy * 2. - 1.;
        m.x += .31;
    }
    m.y += .05;
//    if (mouse.z >= 0.) m -= mouse.zw;
    m *= .7*6.281;
    m.y = clamp(m.y, -1.57, 1.57);
    f = vec3(sin(m.x) * cos(m.y), sin(m.y), cos(m.x) * cos(m.y));
    return cameraMatrix(f);
}

// Function 209
bool ReadKey( int key, bool toggle )
{
	float keyVal = texture( iChannel3, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ) ).x;
	return (keyVal>.5)?true:false;
}

// Function 210
void handlePlayerInput(inout vec4 posVel, inout vec4 data, inout vec4 special, int keyL, int keyR, int keyF, int keyB, int keySpecial1, int keySpecial2)
{
    // unpack data
    vec2 pos = posVel.xy;
    vec2 vel = posVel.zw;
    
    vec2 deltaTemp 	= vec2(0.0);
    
    float moveRight 	= texelFetch( iChannel1, ivec2(keyR			, 0), 0 ).x;
    float moveLeft  	= texelFetch( iChannel1, ivec2(keyL			, 0), 0 ).x;
    float moveForw  	= texelFetch( iChannel1, ivec2(keyF			, 0), 0 ).x;
    float moveBack  	= texelFetch( iChannel1, ivec2(keyB			, 0), 0 ).x;
    float inputSpecial1	= texelFetch( iChannel1, ivec2(keySpecial1	, 0), 0 ).x;
    float inputSpecial2	= texelFetch( iChannel1, ivec2(keySpecial2	, 0), 0 ).x;

    float inputSum = 0.0;
    inputSum += moveRight;
    inputSum += moveLeft;
    inputSum += moveForw;
    inputSum += moveBack;
    float wasInput = abs(inputSum);
    
    
    deltaTemp.x = steeringStrength * (moveRight - moveLeft);
    deltaTemp.y = steeringStrength * (moveForw  - moveBack);
    
    
    if (AUTO_DANCE)
    {
        if (DANCE_ALL_THE_TIME || (special.w > AUTO_DANCE_TIMER * 75.0))
        {
            // dancing
            deltaTemp.xy += 0.03 * vec2(cos(2.3 * iTime), sin(3.9 * iTime));
            deltaTemp = clamp(deltaTemp, vec2(-1.0), vec2(1.0));
        }
    }    
    
    vel = mix(vel, deltaTemp, vec2(0.05));
    
    pos += vel;
    
    data.xy = mix(data.xy, pos, vec2(0.08));

    float lenVel = length(vel);

    // anim timer
    data.z += 1.8 * lenVel;
    
    // anim timer (smooth)
    data.w = mix(data.w, data.z, 0.1);

    // crouch
    //special.x = mix(special.x, 1.0 - inputSpecial1, 0.03);
    special.x = mix(special.x, inputSpecial1, 0.03);
    
    // sing
    special.y = mix(special.y, inputSpecial2, 0.12);
    
    // timer without input
    if (wasInput > 0.01)
    {
    	special.w = 0.0;
    }
    else
    {
    	special.w += 1.0;
    }
    
    // debug
    //special.y = (special.w > AUTO_DANCE_TIMER * 75.0) ? 1.0 : 0.0;
    
    
    // pack data
    posVel.xy = pos;
    posVel.zw = vel;
}

// Function 211
bool ReadKeyBool( int key, bool toggle )
{
	float keyVal = texture( iChannel3, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ) ).x;
	return (keyVal>.5)?true:false;
}

// Function 212
bool isKeyPressed (in int key) {
    return bool(texelFetch(iChannel2, ivec2(key, 0), 0).x);
}

// Function 213
void LoadInputs(out Inputs inp)
{
    inp.button = iMouse.z >= 0.;
    inp.mouse = iMouse.xy;
    if (iMouse.xyz == vec3(0)) // icon?
        inp.mouse.y = iResolution.y*.5; // don't look at ground
    inp.move = vec2(key(KEY_RT) - key(KEY_LF)
                  , key(KEY_FW) - key(KEY_BW));
    inp.dt = iTimeDelta;
}

// Function 214
bool keyIsDown( float key ) {
    return texture( iChannel3, vec2(key,0.25) ).x > .5;
}

// Function 215
void keydrag_event()
{
    vec2 uv = (-iResolution.xy + 2. * iMouse.xy) / iResolution.y;
    if(!is(FIX_ORI) && !is(FIX_TAR))  //drag target
    {
    	SET_TARGET(uv);
    }
	if(is(FIX_ORI) && !is(FIX_TAR))  //drag target
    {
        SET_TARGET(uv);
    }
    else if(!is(FIX_ORI) && is(FIX_TAR))  //drag origin
    {
    	SET_ORIGIN(uv);
    }
}

// Function 216
bool key(int t,bool rep){
    int r = read(t,0);
    return r==1 || (rep && r>10);
}

// Function 217
float inputState(in ivec2 ip)
{
    vec2 p = (vec2(ip) + offset) / iChannelResolution[0].xy;
    return texture(iChannel0, p).x;
}

// Function 218
float keyPressed(int keyCode) {
	return texture(iChannel1, vec2((float(keyCode) + 0.5) / 256., .5/3.)).r;   
}

// Function 219
float showMouse(vec2 u,vec3 r,vec4 m//return distance to 2 mousePositions
){m.xy-=u
 ;m.zw-=u
 ;u=sqrt(vec2(dd(m.xy),dd(m.zw)))     
 ;u=abs(abs(abs(u)-Aa*9.)-Aa*vec2(7,5))-Aa
 ;u=vec2(min(u.x,u.y),max(u.x,u.y))
 ;u.xy/=u.yx
 ;u=ssaa(u)
 ;u.x=mix(u.x,u.y,.2)
 ;return u.x;}

// Function 220
vec4 drawMouse(in vec2 U){vec4 c=vec4(0)
;vec2 u=U/iResolution.xy
;vec3 m = texelFetch(iChannel0, LastMouseClick, 0).xyz
;float clickCircle=sharp(distance(u,m.xy)-.01,1.)
;c=mix(c,vec4(0,0,0,1),clickCircle)
;if(m.z>.5)
 {vec2 totalDelta  =(iMouse.xy / iResolution.xy) - m.xy
 ;vec4 lastDelta   =texelFetch(iChannel0, LastMouseDelta, 0)
 ;vec2 currentDelta=totalDelta - lastDelta.zw
 ;float lastDeltaLine = lineS(u, m.xy, m.xy + lastDelta.zw, 0.005)
 ;c=mix(c,vec4(0,1,0,1),lastDeltaLine)
 ;float dt=lineS(u,m.xy + lastDelta.zw, m.xy + lastDelta.zw + currentDelta, 0.005)
 ;c=mix(c,vec4(1,0,0,1),dt);
 }return c;}

// Function 221
bool isMousePressed()
{
	return iMouse.z > 0.0;
}

// Function 222
float inputState2(in ivec2 ip)
{
    vec2 p = (vec2(ip) + vec2(16.5, 1.5)) / iChannelResolution[0].xy;
    return texture(iChannel0, p).x;
}

// Function 223
bool ReadKey( int key, bool toggle )
{
	float keyVal = textureLod( iChannel2, vec2( (float(key)+.5)/256.0, toggle?.75:.25), 0.0 ).x;
	return (keyVal>.5)?true:false;
}

// Function 224
bool keydown(int vk)
{
    return .5 <= loadValue(Keyboard, vk).x;
}

// Function 225
bool keyIsDown( float key ) {
    return texture( iChannel3, vec2(key, 0.25) ).x > 0.5;
}

// Function 226
float keyPress(int keyCode) {
    return texture(iChannel1, vec2((float(keyCode) + 0.5) / 256., 1.5/3.), 0.0).r;   
}

// Function 227
float tukey(float x, float w, float a)
{
    w *= 2.0;
    x += w*0.5;
    const float pi = 3.14159265358979;
    float xoa = 2.*x/(a*w);
    if (x < 0. || abs(x) > w)
        return 0.;
    if (x < a*w*0.5)
        return 0.5*(1.+cos(pi*(xoa-1.)));
    else if (x < w*(1.-a*0.5))
        return 1.;
    else
        return 0.5*(1.+cos(pi*(xoa - 2./a + 1.)));
}

// Function 228
float keystatepress( int key )
	{ return max( keystate( key ), keypress( key ) ); }

// Function 229
float noteKeyA (float n){float[] notes = float[](1046.5,1174.66,1318.51,1396.91,1567.98,1760.,1975.53, 2093.); 
                         return notes[int(n)%8];}

// Function 230
bool key(int code) {
  return texelFetch(iChannel3, ivec2(code,2),0).x != 0.0;
}

// Function 231
vec2 MouseDragRead(){
    #define MouseTex iChannel0
    #define Scale 32767.
	uvec2 data = floatBitsToUint(GetValue(MouseTex,MOUSE_COORD).zw);
	vec4 MousePos = vec4(unpackSnorm2x16(data.x),unpackSnorm2x16(data.y))*Scale,tmpMousePos = vec4(MousePos.xy,0.,0.);
    iMouse.z>0. ? tmpMousePos.xy+=tmpMousePos.zw=iMouse.xy-iMouse.zw : tmpMousePos.xy=MousePos.xy+=MousePos.zw;
    return tmpMousePos.xy;
}

// Function 232
vec4 drawMouse(in vec2 fragCoord)
{
    vec4 color = vec4(0.0);
    
    vec2 uv = fragCoord / iResolution.xy;
    vec3 lastMouseClick = texelFetch(iChannel0, ivec2(0, 2), 0).xyz;
    
    float clickCircle = Sharpen(distance(uv, lastMouseClick.xy), 0.01, 1.0);
    color = mix(color, vec4(0.0, 0.0, 0.0, 1.0), clickCircle);
    
    if(lastMouseClick.z > 0.5)
    {
        vec2 totalDelta   = (iMouse.xy / iResolution.xy) - lastMouseClick.xy;
        vec4 lastDelta    = texelFetch(iChannel0, ivec2(0, 3), 0);
        vec2 currentDelta = totalDelta - lastDelta.zw;
        
    	float lastDeltaLine = Line(uv, lastMouseClick.xy, lastMouseClick.xy + lastDelta.zw, 0.005);
        color = mix(color, vec4(0.0, 1.0, 0.0, 1.0), lastDeltaLine);
        
        float currentDeltaLine = Line(uv, lastMouseClick.xy + lastDelta.zw, lastMouseClick.xy + lastDelta.zw + currentDelta, 0.005);
    	color = mix(color, vec4(1.0, 0.0, 0.0, 1.0), currentDeltaLine);
    }
    
    return color;
}

// Function 233
bool keyPress(int ascii){
    return (texture(iChannel2,vec2((.5+float(ascii))/256.,0.25)).x > 0.);
}

// Function 234
float get_key(int key_code) {
    return texelFetch(BUFFER_KEYBOARD, ivec2(key_code,0), 0).x;
}

// Function 235
void StoreKeyboardState(inout vec4 fragColor, in vec2 fragCoord)
{
    vec4 previousKeyboardState = vec4(float(gKeyboardState.mKeyModeForward[0]), float(gKeyboardState.mKeyModeBackwards[0]), float(gKeyboardState.mKeyW[0]), float(gKeyboardState.mKeyS[0]));
    StoreValue(txPreviousKeyboard, previousKeyboardState, fragColor, fragCoord);
}

// Function 236
bool KeyDown(in int key){
	return (textureLod(iChannel1,vec2((float(key)+0.5)/256.0, 0.25),0.0).x>0.0);
}

// Function 237
float key(int vk)
{
    return step(.5, texelFetch(Kbd, ivec2(vk, 0), 0).x);
}

// Function 238
float keyDn(vec2 a){return keyDn(a.x)-keyDn(a.y);}

// Function 239
bool keypress(int code) {
  return false;
}

// Function 240
bool keyPress(int ascii) { return (texture(iChannel2,vec2((.5+float(ascii))/256.,0.25)).x > 0.); }

// Function 241
bool ReadKey( int key, bool toggle )
{
	float keyVal = texture( iChannel0, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ) ).x;
	return (keyVal>.5)?true:false;
}

// Function 242
float getKey(float key) {return texture(iChannel0, vec2(key, .25)).x;}

// Function 243
bool isKeyReleased (in int key) {
    return bool(texelFetch(iChannel1, ivec2(key, 1), 0).x);
}

// Function 244
float isKeyPressed(int key, int type)
{
	return texelFetch( iChannel1, ivec2(key, type), 0 ).x;
}

// Function 245
float wkey(float a){return 440.*pow(2.,a/12.);}

// Function 246
vec4 drawKey( vec2 uv, int color ) {
    uv = floor(fract(uv)*64.) - 32.;
    if( abs(uv.x) < 16. && abs(uv.y) < 16. ) {
        float l = step(abs(uv.y), 1.);
        l = max(l, step(length(uv+vec2(8,0)), 7.5));
        l -= step(length(uv+vec2(8,0)), 4.5);
        l = max(l, step(6.,uv.x)*step(uv.x, 7.)*step(0.,uv.y)*step(abs(uv.y), 5.));
        l = max(l, step(10.,uv.x)*step(uv.x, 11.)*step(0.,uv.y)*step(abs(uv.y), 7.));
        l = max(l, step(14.,uv.x)*step(0.,uv.y)*step(abs(uv.y), 6.));
        
	    vec3 col = vec3(0);
    	col[color-7] = 1.;
        return vec4( 2. * l * (.5 + .5 * texture(iChannel1, uv/64.).x) * col, l );
    } else {
        return vec4(0);
    }
}

// Function 247
vec4 processMouse(vec4 mouse) {
    int capturedIndex = -1;
    if (mouse[2] > 0.) {
        for (int i = 0 ; i < controlCount; i++) {
            if (controls[i].mouseDown) {
                capturedIndex = i;
            }
        }
    }
    
    bool handled = false;
    for (int i = controlCount - 1 ; i >= 0; i--) {
        handled = handleMouse(i, mouse, capturedIndex);
        if (handled) {
            break;
        }
    }
    if (handled || capturedIndex >= 0) {
        return vec4(mouse.xy, 0., 0.);
    } else {        
    	return mouse;
    }
}

// Function 248
bool keyToggle(int ascii) {
	return (texture(iChannel3,vec2((.5+float(ascii))/256.,0.75)).x > 0.);
}

// Function 249
bool Key_IsToggled(sampler2D samp, int key )
{
    return texelFetch( samp, ivec2(key, 2), 0 ).x > 0.0;    
}

// Function 250
float keyToggled(int keyCode) {
    return textureLod(iChannel2, vec2((float(keyCode) + 0.5) / 256., 2.5/3.), 0.0).r;   
}

// Function 251
vec4 UpdateMouseClick(vec4 o, vec2 m)
{float l=GetLastMouseClick().z
;float isClicked = step(.5,iMouse.z)//todo: mouse and iMouse, i smell bad style
;if((isClicked>.5)&& l<.5)o.xy=vec2(m.xy/iResolution.xy)
;o.z=isClicked
;return o;}

// Function 252
vec2 keyT(vec2 u){
 u.x*=.7;//scaling it all
 u.x*=64.;
 u.x+=64.+32.+0.;
 //u.x+=127.;//position of "a"
 u.x/=iResolution.x;
 return u;
}

// Function 253
bool ReadKey( int key, bool toggle )
{
	float keyVal = texture( iChannel1, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ) ).x;
	return (keyVal > .5) ? true : false;
}

// Function 254
void deltaMouse(out vec4 fragColor) {//color stored
    vec4 val = texelFetch(iChannel0, ivec2(res)+ivec2(2,0), 0);
    //grab current delta mouse position
    vec4 oldMouse = texelFetch(iChannel0, ivec2(res)+ivec2(1,0), 0);
    //grab previous mouse position (last frame)
    if(iMouse.z > 0.5 && val.w >= 0.5){ //if mouse was not pressed on prev frame, do not update delta
        //this means it only changes when the mouse is down and does not warp colors.
        vec2 deltaM = (iMouse.xy/iChannelResolution[0].xy)-oldMouse.xy;
        //set delta to be the change in mouse positions between frames
    	val.xy+=deltaM;
        //add the delta to val
    }
    if(val.x >= 1.){
    	val.x = 0.;   
    }
    if(val.y >= 1.){
    	val.y = 0.;   
    }
    val.w = iMouse.z; //update click state
    fragColor = val;
}

// Function 255
void LoadInputs(out Inputs inp)
{
    inp.button = iMouse.z >= 0.;
    inp.mouse = iMouse.xy;
    if (iMouse.xyz == vec3(0)) // icon?
        inp.mouse.y = iResolution.y*.5; // don't look at ground
    inp.move = vec3(key(KEY_RT) - key(KEY_LF)
                  , key(KEY_UW) - key(KEY_DW)
                  , key(KEY_FW) - key(KEY_BW));
    inp.dt = iTimeDelta;
}

// Function 256
bool IsKeyDown(float aKeyId)
{
   return texture(iChannel2, vec2(aKeyId, 0.0)).x >= 0.5;
}

// Function 257
float keyDown( in float key ) {
    return texture( iChannel2, vec2(key, 0.25) ).r;
}

// Function 258
vec2 getMouse() {
    vec2 mouse = iMouse.xy;
    if (length(mouse) <= 0.0001) {
        mouse = vec2(0.3 + sin(5.0 * iTime) * 0.1, 0.3) * iResolution.y;
    }
    return mouse;
}

// Function 259
void drawKeyIcon( vec2 lt, vec2 size, inout vec4 color, vec2 coord, int keyColor ) {
    coord = (coord-lt) / size;
    if( coord.x >= 0. && coord.x <= 1. && coord.y >= 0. && coord.y <= 1. ) {    
		vec4 col = drawKey(-coord, keyColor);
        color = mix( color, col, col.a );
    }
}

// Function 260
float keypress( int key )
	{ return texelFetch( iChannel3, ivec2( key, 1 ), 0 ).x; }

// Function 261
mat4 look_around_mouse_control( mat4 camera, float pitch, float tan_half_fovy )
{
	float mouse_ctrl = 1.0;
	vec2 mm_offset = vec2( 0.0, pitch );
	vec2 mm = vec2( 0.0, 0.0 );

#ifndef EXTRA_3D_CAMERA
	if ( iMouse.z > 0.0 )
		mm = ( iMouse.xy - iResolution.xy * 0.5 ) / ( min( iResolution.x, iResolution.y ) * 0.5 );
#endif

	mm.x = -mm.x;
	mm = sign( mm ) * pow( abs( mm ), vec2( 0.9 ) );
	mm *= PI * tan_half_fovy * mouse_ctrl;
	mm += mm_offset;
	return camera * yup_spherical_offset( mm.y, mm.x );
}

// Function 262
float IsKeyToggled(float key)
{
    return texture(iChannel1, vec2(key, 2.0)).r;
}

// Function 263
bool keyDown(int key)  { return keys(key, 0).x > 0.5; }

// Function 264
MouseControlInfo MouseDragWriteRead(out vec2 Store,ivec2 iU){
    #define MouseTex iChannel0
    #define Scale 32767.
	uvec2 data = floatBitsToUint(GetValue(MouseTex,MOUSE_COORD).zw);
	mediump vec4 MousePos = vec4(unpackSnorm2x16(data.x),unpackSnorm2x16(data.y))*Scale,tmpMousePos = vec4(MousePos.xy,0.,0.);
    iMouse.z>0. ? tmpMousePos.xy+=tmpMousePos.zw=iMouse.xy-iMouse.zw : tmpMousePos.xy=MousePos.xy+=MousePos.zw;
    MouseControlInfo info;
    info.position = tmpMousePos.xy;
    info.dataLock = IsCoord(iU,MOUSE_COORD);
    if(info.dataLock)Store=uintBitsToFloat(uvec2(packSnorm2x16(fract(MousePos.xy/Scale)),packSnorm2x16(tmpMousePos.zw/Scale)));
    return info;
}

// Function 265
float keyDown(int keyCode) {
    return textureLod(iChannel2, vec2((float(keyCode) + 0.5) / 256., .5/3.), 0.0).r;   
}

// Function 266
float key(int vk)
{
    return step(.5, fetch(Kbd, ivec2(vk, 0)).x);
}

// Function 267
mat4 look_around_mouse_control( mat4 camera, float pitch, float tan_half_fovy, float look_at_the_abyss )
{
 float mouse_ctrl = 1.0;
 vec2 mm_offset = vec2( 0.0, pitch );
 vec2 mm = vec2( 0.0, 0.0 );
 if ( iMouse.z > 0.0 || false ) mm = ( iMouse.xy - iResolution.xy * 0.5 ) / ( min( iResolution.x, iResolution.y ) * 0.5 );
 float mm_y = mm.y;
 mm.x = -mm.x;
 mm = sign( mm ) * pow( abs( mm ), vec2( 0.9 ) );
 mm *= 3.141592654 * tan_half_fovy * mouse_ctrl;
 mm += mm_offset;
 if ( mm_y < 0. )
 {
  // very special case camera control for getRoofTopCoffeeBreakCamera
  vec3 v = camera[2].xyz;
  v.xy = rotate_with_angle( v.xy, mm.x );
  camera[3].xyz += v * mm.y * look_at_the_abyss;
 }
 return camera * yup_spherical_coords_to_matrix( mm.y, mm.x );
}

// Function 268
bool getKeyDown(float key) {
    return texture(iChannel1, vec2(key / KEY_ALL, 0.5)).x > 0.1;
}

// Function 269
bool isKeyToggled(in int key) {
  return texelFetch(iChannel1, ivec2(key, 2), 0).x > 0.0;
}

// Function 270
bool keypress(int code) {
#if __VERSION__ < 300
    return false;
#else
  return texelFetch(iChannel0, ivec2(code,2),0).x != 0.0;
#endif
}

// Function 271
bool keyPress(int ascii) {
	return (texture(iChannel1,vec2((.5+float(ascii))/256.,0.25)).x > 0.);
}

// Function 272
bool keyToggle( int ascii ) 
{
	return ( texture( iChannel3, 
	                vec2( ( 0.5 + float( ascii ) ) / 256.0, 0.75 ) ).x > 0.0 );	                                                        
}

// Function 273
bool keypress(int code) {
#if __VERSION__ < 300
  return false;
#else
  return texelFetch(iChannel0, ivec2(code,2),0).x != 0.0;
#endif
}

// Function 274
bool key(int key) {
   return texelFetch(iChannel2, ivec2(key,2),0).x != 0.0;
}

// Function 275
bool keypress(int code) {
    return texelFetch(iChannel0, ivec2(code,2),0).x != 0.0;
}

// Function 276
float KeyData(sampler2D keyboard , int keyCode, int state){
	return texelFetch(keyboard,ivec2(keyCode,state),0).x;
}

// Function 277
bool keyPressed(float k) {
    return (texture(iChannel3, vec2(k, .2)).r>0.);
}

// Function 278
bool isKeyDown(int key) {
    return texelFetch(iChannel1, ivec2(key, 0), 0).r>0.0;
}

// Function 279
void mouseMoveEvt(int index, vec4 mouse, int capturedIndex, bool inBounds) {
    // mouseMove event:
    switch (controls[index].type) {
        case PUSH_BTN_T:
        controls[index].value = inBounds ? 1. : 0.;
        break;

        case SPINNER_T:
        controls[index].value = midpointAngle(index, mouse.xy) / 3.14159;
        break;

        case CLICKBOX_T:
	    vec2 dr = hitCoordsNormalized(index, mouse.xy) / 2.;
        controls[index].value2 = normCoordsToClickboxVal(dr);
        break;        
    }
}

// Function 280
bool key_state(int key) {
    return texelFetch(iChannel3, ivec2(key, 0), 0).x != 0.;
}

// Function 281
vec3 CameraDirInput(vec2 vm) {
    vec2 m = vm/iResolution.x;
    
    //m.y = -m.y; //invert up/down key
    
    mat3 rotX = mat3(1.0, 0.0, 0.0, 0.0, cos(m.y), sin(m.y), 0.0, -sin(m.y), cos(m.y));
    mat3 rotY = mat3(cos(m.x), 0.0, -sin(m.x), 0.0, 1.0, 0.0, sin(m.x), 0.0, cos(m.x));
    
    return (rotY * rotX) * vec3(KeyboardInput(), 0.0).xzy;
}

// Function 282
float tukey(vec2 pixCoords,vec2 centerCoords, float theRadius, float r){
	float d = distance(pixCoords,centerCoords);
	float w = d/theRadius;
     if(w > 1.0){
    	return 0.0;
    }
    else if (w <= (1.0 - r/2.0)){
    
    	return 1.0;
    }
    else{
    
    	return (0.5*(1.0+cos(2.0*pi/r*(w-1.0+r/2.0))));
    
    }
    
}

// Function 283
bool keypress(int code) 
{
	return texelFetch(iChannel0, ivec2(code,2), 0).x != 0.0;
}

// Function 284
bool Key_IsToggled(float key)
{
    return texture( iChannel1, vec2(key, 1.0) ).x > 0.0;
}

// Function 285
float isKeyPressed(int key)
{
	return texelFetch( iChannel1, ivec2(key, 0), 0 ).x;
}

// Function 286
bool getMouseDown() {
    return iMouse.z > 0.0;
}

// Function 287
bool keyToggle(int ascii) { return (texture(iChannel1,vec2((.5+float(ascii))/256.,0.75)).x > 0.);}

// Function 288
bool checkKey(float key1, float key2, float key3)
{
    return checkKey(key1) || checkKey(key2) || checkKey(key3);
}

// Function 289
vec2 mouseDelta(vec3 iResolution, vec4 iMouse, sampler2D bufD){
    vec2 pixelSize = 1. / iResolution.xy;
    float eighth = 1./8.;
    vec4 oldMouse = Cell(2, bufD);
    vec4 nowMouse = vec4(iMouse.xy * pixelSize.xy, iMouse.zw * pixelSize.xy);
    if(oldMouse.z > pixelSize.x && oldMouse.w > pixelSize.y && 
       nowMouse.z > pixelSize.x && nowMouse.w > pixelSize.y)
    {
        return nowMouse.xy - oldMouse.xy;
    }
    return vec2(0.);
}

// Function 290
float IsKeyPressed(float key)
{
    return texture(iChannel1, vec2(key, 0.0)).r;
}

// Function 291
bool KeyPressed(int key)
{ 
    return texelFetch( iChannel1, ivec2(key,0.0), 0 ).x > 0.5;
}

// Function 292
vec4 mouseSprite(int lx, int ly, vec4 bg) {
  // line 0
  // 11__ ____ __
  if (ly == 0) {
    if (lx == 0) return c1;
    if (lx == 1) return c1;
  }
  // line 1
  // 101_ ____ __
  if (ly == 1) {
    if (lx == 0) return c1;
    if (lx == 1) return c0;
    if (lx == 2) return c1;
  }
  // line 2
  // 1001 ____ __
  if (ly == 2) {
    if (lx == 0) return c1;
    if (lx == 1) return c0;
    if (lx == 2) return c0;
    if (lx == 3) return c1;
  }
  // line 3
  // 1000 1___ __
  if (ly == 3) {
    if (lx == 0) return c1;
    if (lx == 1) return c0;
    if (lx == 2) return c0;
    if (lx == 3) return c0;
    if (lx == 4) return c1;
  }
  // line 4
  // 1000 01__ __
  if (ly == 4) {
    if (lx == 0) return c1;
    if (lx == 1) return c0;
    if (lx == 2) return c0;
    if (lx == 3) return c0;
    if (lx == 4) return c0;
    if (lx == 5) return c1;
  }
  // line 5
  // 1000 001_ __
  if (ly == 5) {
    if (lx == 0) return c1;
    if (lx == 1) return c0;
    if (lx == 2) return c0;
    if (lx == 3) return c0;
    if (lx == 4) return c0;
    if (lx == 5) return c0;
    if (lx == 6) return c1;
  }
  // line 6
  // 1000 0001 __
  if (ly == 6) {
    if (lx == 0) return c1;
    if (lx == 1) return c0;
    if (lx == 2) return c0;
    if (lx == 3) return c0;
    if (lx == 4) return c0;
    if (lx == 5) return c0;
    if (lx == 6) return c0;
    if (lx == 7) return c1;
  }
  // line 7
  // 1000 0000 1_
  if (ly == 7) {
    if (lx == 0) return c1;
    if (lx == 1) return c0;
    if (lx == 2) return c0;
    if (lx == 3) return c0;
    if (lx == 4) return c0;
    if (lx == 5) return c0;
    if (lx == 6) return c0;
    if (lx == 7) return c0;
    if (lx == 8) return c1;
  }
  // line 8
  // 1000 0111 11
  if (ly == 8) {
    if (lx == 0) return c1;
    if (lx == 1) return c0;
    if (lx == 2) return c0;
    if (lx == 3) return c0;
    if (lx == 4) return c0;
    if (lx == 5) return c1;
    if (lx == 6) return c1;
    if (lx == 7) return c1;
    if (lx == 8) return c1;
    if (lx == 9) return c1;
  }
  // line 9
  // 1001 001_ __
  if (ly == 9) {
    if (lx == 0) return c1;
    if (lx == 1) return c0;
    if (lx == 2) return c0;
    if (lx == 3) return c1;
    if (lx == 4) return c0;
    if (lx == 5) return c0;
    if (lx == 6) return c1;
  }
  // line 10
  // 101_ 1001 __
  if (ly == 10) {
    if (lx == 0) return c1;
    if (lx == 1) return c0;
    if (lx == 2) return c1;
    if (lx == 4) return c1;
    if (lx == 5) return c0;
    if (lx == 6) return c0;
    if (lx == 7) return c1;
  }
  // line 11
  // 11__ 1001 __
  if (ly == 11) {
    if (lx == 0) return c1;
    if (lx == 1) return c1;
    if (lx == 4) return c1;
    if (lx == 5) return c0;
    if (lx == 6) return c0;
    if (lx == 7) return c1;
  }
  // line 12
  // 1___ 1001 __
  if (ly == 12) {
    if (lx == 0) return c1;
    if (lx == 5) return c1;
    if (lx == 6) return c0;
    if (lx == 7) return c0;
    if (lx == 8) return c1;
  }
  // line 13
  // ____ _100 1_
  if (ly == 13) {
    if (lx == 5) return c1;
    if (lx == 6) return c0;
    if (lx == 7) return c0;
    if (lx == 8) return c1;
  }
  // line 14
  // ____ __11 1_
  if (ly == 14) {
    if (lx == 6) return c1;
    if (lx == 7) return c1;
    if (lx == 8) return c1;
  }
  return bg;
}

// Function 293
float keyRe(float a){return texture(iChannel1,vec2(a,.5)).x;}

// Function 294
bool keyClick(int ascii) {
	return (texture(iChannel2,vec2((.5+float(ascii))/256.,0.25)).x > 0.);
}

// Function 295
bool isKeyPressed(in sampler2D keyboard, in int keyCode) {
  return texelFetch(keyboard, ivec2(keyCode, 1), 0).x > 0.0;
}

// Function 296
int keycount(int key) {
  return int(store(0,key).x);
}

// Function 297
bool KeyDown(int char)
{
    return int(texelFetch(iChannel1, ivec2(char, 0), 0).x) > 0;
}

// Function 298
float is_key_pressed(float key_code)
{
    return texture(iChannel0, vec2((key_code), 0.0)).x;
}

// Function 299
float ReadKeyToggle( int key )
{
	return ReadKeyInternal(key,true);
}

// Function 300
bool isKeyPressed(int ascii) {
    return texelFetch(iChannel2, ivec2(ascii, 1), 0).x > 0.0;
}

// Function 301
bool isKeyPressed(int key) {
  return texelFetch(iChannel1, ivec2(key, 1), 0).x == 1.0;
}

// Function 302
vec3 keys(vec2 uv){
    vec3 result=vec3(0.);
 if(0.<texelFetch( iChannel2, ivec2(KEY_SPC,0), 0 ).x){
 result+=vec3(-1.,0.,0.);
 }
 if(0.<texelFetch( iChannel2, ivec2(KEY_SHIFT,0), 0 ).x){
 result+=vec3(1.,0.,0.);
 }
 if(0.<texelFetch( iChannel2, ivec2(KEY_W,0), 0 ).x){
     result+=vec3(0.,-1.,0.);
 }
 if(0.<texelFetch( iChannel2, ivec2(KEY_S,0), 0 ).x){
     result+=vec3(0.,1.,0.);
 }
    if(0.<texelFetch(iChannel2, ivec2(KEY_D,0),0).x){
        result+=vec3(0.,0.,-1.);
    }
    if(0.<texelFetch( iChannel2, ivec2(KEY_A,0), 0 ).x){
        result+=vec3(0.,0.,1.);
    }
    vec3 strt=normalize( vec3(0.-vec2(-0.4,0.4),-2.0));//vec3(1.,1.,-1.)
    return (result/40.)*fetchMouseRotation();
}

// Function 303
float ReadKeyFloat( int key, bool toggle )
{
	float keyVal = texture( iChannel3, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ) ).x;
	return step(.5,keyVal);
}

// Function 304
vec3 Key2Col()
{
	return mix ( vec3(1.0,.4,.0), vec3(1), smoothstep( 0.7, 1.0, sin((pixel.x+pixel.y)*.1-6.0*iTime) ) );
}

// Function 305
float keyDown(int keyCode) {
    return texture(iChannel2, vec2((float(keyCode) + 0.5) / 256., .5/3.), 0.0).r;   
}

// Function 306
bool checkKey(float key1, float key2)
{
    return checkKey(key1) || checkKey(key2);
}

// Function 307
bool readKey( int key )
{
	bool toggle = false;
	float keyVal = texture( iChannel1, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ) ).x;
	return (keyVal>.5)?true:false;
}

// Function 308
ControlInfo MouseDrag(out vec2 Store,ivec2 iU,ivec2 coord){
    #define MouseTex iChannel1
    #define Scale 32767.
	uvec2 data = floatBitsToUint(GetValue(MouseTex,coord).zw);
	mediump vec4 MousePos = vec4(unpackSnorm2x16(data.x),unpackSnorm2x16(data.y))*Scale,tmpMousePos = vec4(MousePos.xy,0.,0.);
    iMouse.z>0. ? tmpMousePos.xy+=tmpMousePos.zw=iMouse.xy-iMouse.zw : tmpMousePos.xy=MousePos.xy+=MousePos.zw;
    ControlInfo info;
    info.position = tmpMousePos.xy;
    info.dataLock = IsCoord(iU,coord);
    if(info.dataLock)Store=uintBitsToFloat(uvec2(packSnorm2x16(fract(MousePos.xy/Scale)),packSnorm2x16(tmpMousePos.zw/Scale)));
    return info;
}

// Function 309
bool KeyDown(in int key){return (texture(iChannel1,vec2((float(key)+0.5)/256.0, 0.25)).x>0.0);}

// Function 310
int keycount(int key) {
  return int(texelFetch(iChannel1, ivec2(0,key),0).x);
}

// Function 311
vec2 MouseDragRead(){
    #define MouseTex iChannel1
    #define Scale 32767.
	uvec2 data = floatBitsToUint(GetValue(MouseTex,MOUSE_COORD).zw);
	vec4 MousePos = vec4(unpackSnorm2x16(data.x),unpackSnorm2x16(data.y))*Scale,tmpMousePos = vec4(MousePos.xy,0.,0.);
    iMouse.z>0. ? tmpMousePos.xy+=tmpMousePos.zw=iMouse.xy-iMouse.zw : tmpMousePos.xy=MousePos.xy+=MousePos.zw;
    return tmpMousePos.xy;
}

// Function 312
bool keyToggle(int ascii) {
	return (texture(iChannel2,vec2((.5+float(ascii))/256.,0.75)).x > 0.);
}

// Function 313
bool ReadKey(int key)
{
	float keyVal = texture( iChannel3, vec2( (float(key)+.5)/256.0, .25 ) ).x;
	return keyVal > 0.0;
}

// Function 314
bool keypress(int key) {
#if __VERSION__ < 300
    return false;
#else
    return texelFetch(iChannel0, ivec2(key,2),0).x != 0.0;
#endif
}

// Function 315
void keyup_event()
{
    KEY_DRAG_FLAG = 0.0;
}

// Function 316
void keyInput()
{
  if (iFrame > 9)
  {
    animate_pattern   = !ReadKey(KEY_A, true);
    show_background   = !ReadKey(KEY_B, true);
    show_reflections  = !ReadKey(KEY_F, true);
    rotation_scene    = !ReadKey(KEY_R, true);
    cross_eye_view    = !ReadKey(KEY_S, true);
  }
}

// Function 317
bool key_toggle(float ascii) { 
	return (texture(iChannel0,vec2((ascii+.5)/256.,0.75)).x > 0.); 
}

// Function 318
bool mouseDoubleClick(){
    
    if(iMouse.z <1. ) {
   
        int changeCount=0;
        for(int i=0;i<20;i++){

            int mouseChange=          
               (load(_old *vec2(i) + _mouse ).z>0.?0:1)
              +(load( _old * vec2(i+1) +_mouse ).z>0.?0:1);


            if(mouseChange==1)changeCount++;
            if(load(_mouseBusy).r>0.) {store1(_mouseBusy,float(1.));return false;}
                               
            if(changeCount>2){
                //if(load(_time).r - load(_old*vec2(i) +_time).r<1.) return false;
                if(length(load(_mouse).xy -load(_old * vec2(i+1) +_mouse).xy)>.05) return false;
                store1(_mouseBusy,float(1.));
                return true;

            }         
        }
    }
    store1(_mouseBusy,float(0.));
    return false; 
}

// Function 319
void WriteMousePos(float ytext, vec2 mPos)
{
  int digits = 3;
  float radius = resolution.x / 200.;

  // print dot at mPos.xy 
  if (iMouse.z > 0.0) dotColor = mpColor;
  float r = length(mPos.xy - pixelPos) - radius;
  vColor += mix(vec3(0), dotColor, (1.0 - clamp(r, 0.0, 1.0)));

  // print first mouse value
  SetTextPosition(1., ytext);

  // print mouse position
  if (ytext == 7.)
  {
    drawColor = mxColor;
    WriteFloat(mPos.x,6,3);
    SPACE;
    drawColor = myColor;
    WriteFloat(mPos.y,6,3);
  }
  else
  {
    drawColor = mxColor;
    WriteInteger(int(mPos.x));
    SPACE;
    drawColor = myColor;
    WriteInteger(int(mPos.y));
  }
}

// Function 320
bool readKey(int key, bool toggle) {
    return texture(iChannel1, vec2((float(key)+0.5)/256.0, toggle ? 0.75 : 0.25)).x > 0.5;
}

// Function 321
vec4 affMouse(vec2 uv)  { // display mouse states ( color )
    vec4 mouse = UI(33);                       // current mouse pos
    float k = length(mouse.xy/R.y-uv)/Mradius,
          s = sign(mouse.z);
	if (k<1.) 
	    if (k>.8) return vec4(1e-10);
		   else   return vec4(s,1.-s,0,1); 
	
    k = length( UI(34).xy/R.y-uv)/Mradius;     // prev mouse pos 
	if (k<1.) 
	    if (k>.8) return vec4(1e-10);
		   else   return vec4(0,0,1,1); 
            
    k = length(abs(mouse.zw)/R.y-uv)/Mradius;  // drag start  mouse pos 
	if (k<1.) 
	    if (k>.8) return vec4(1e-10);
		   else   return vec4(0,.4,s,1); 
	
	return vec4(0);
}

// Function 322
vec2 mouseDelta(){
    vec2 pixelSize = 1. / iResolution.xy;
    float eighth = 1./8.;
    vec4 oldMouse = texture(iChannel2, vec2(7.5 * eighth, 2.5 * eighth));
    vec4 nowMouse = vec4(iMouse.xy * pixelSize.xy, iMouse.zw * pixelSize.xy);
    if(oldMouse.z > pixelSize.x && oldMouse.w > pixelSize.y && 
       nowMouse.z > pixelSize.x && nowMouse.w > pixelSize.y)
    {
        return nowMouse.xy - oldMouse.xy;
    }
    return vec2(0.);
}

// Function 323
bool WasKeyJustPressed(in float key)
{
    if(key == KEY_RIGHT)
    {
        return (gKeyboardState.mKeyModeForward[0] && (gKeyboardState.mKeyModeForward[1] == false));
    }    
	else if(key == KEY_LEFT)
    {
        return (gKeyboardState.mKeyModeBackwards[0] && (gKeyboardState.mKeyModeBackwards[1] == false));
    }
    return false;
}

// Function 324
mat3 mouseRotation(bool enable, vec2 xy) {
    if (enable) {
        vec2 mouse = iMouse.xy / iResolution.xy;

        if (mouse.x != 0. && mouse.y != 0.) {
            xy.x = mouse.x;
            xy.y = mouse.y;
        }
    }
    float rx, ry;
    
    rx = (xy.y + .5) * PI;
    ry = (-xy.x) * 2. * PI;
    
    return sphericalMatrix(rx, ry);
}

// Function 325
bool keyToggle(int ascii) {
	return (texture(iChannel1,vec2((.5+float(ascii))/256.,0.75)).x > 0.);
}

// Function 326
bool keypress(int code) {
  return texelFetch(iChannel0, ivec2(code,2),0).x != 0.0;
}

// Function 327
bool isKeyPressed(float key)
{
	return texture(iChannel3, vec2(key, 0.5/3.0) ).x > 0.5;
}

// Function 328
bool GetKey(int key)
{
    return ReadKey(key, true);
}

// Function 329
void keyManage( inout vec4 fragColor, in ivec2 fragC ){
    if(fragC.x < 7)write(fragC.x,0,keyboard(fragC.x)?read(fragC.x,0)+1:0);
    /*
	for(int i=0;i<7;i++){
        int p = i, q = 0, s = keyboard(i)?read(i,0)+1:0;
    	write(i,0,s);
    }*/
}

// Function 330
bool key(int k){return texelFetch(iChannel3,ivec2(k,0),0).x>0.5;}

// Function 331
bool Key_IsToggled(sampler2D samp, int key)
{
    return texelFetch( samp, ivec2(key, 2), 0 ).x > 0.0;    
}

// Function 332
bool keyIsDown( float key ) {
    return texture( iChannel2, vec2(key,0.25) ).x > .5;
}

// Function 333
void KeyPressed( sampler2D sampler, out float[5] keys)    
{
    
    keys[0] = texelFetch(sampler, ivec2(68, 1),0).x;
    keys[1] = texelFetch(sampler, ivec2(70, 1),0).x;
    keys[2]  = texelFetch(sampler, ivec2(32, 1),0).x;    
    keys[3] = texelFetch(sampler, ivec2(74, 1),0).x;
    keys[4]  = texelFetch(sampler, ivec2(75, 1),0).x;

}

// Function 334
int isKeyPressed(float key)
{
	return texture(iChannel0, vec2(key, 0.25) ).x > .0?1:0;
}

// Function 335
bool mouseEdgeDetect() {
    vec4 oldMouse = read(0.0);
    return oldMouse.z < 0.5 && iMouse.z > 0.5 &&
           oldMouse.w < 0.5 && iMouse.w > 0.5;
}

// Function 336
float key(int a){return texture(iChannel2,vec2((.5+float(a))/256.,0.25)).x;}

// Function 337
bool isKeyDown(int key) {
  return texelFetch(iChannel1, ivec2(key, 0), 0).x == 1.0;
}

// Function 338
float keySinglePress(int keycode) {
    bool now = bool(keyDown(keycode));
    bool previous = bool(texture(iChannel0, vec2(256. + float(keycode) + 0.5, 0.5) / iResolution.xy, 0.0).r);
    return float(now && !previous);
}

// Function 339
bool KeyPressed(sampler2D keyboard,int keyCode){
	return KeyData(keyboard,keyCode,0) > 0.;
}

// Function 340
v0 key(in int key){return texture(iChannel1,v1((v0(key)+0.5)/256.0, 0.25)).x;}

// Function 341
float isKeyPressed(float key)
{
    if (iMouse.z > 0.5) return 0.0;
	return texture( iChannel1, vec2(key, 0.5) ).x;
}

// Function 342
bool keyIsDown( float key ) {
    return texture( iChannel1, vec2(key,0.25) ).x > .5;
}

// Function 343
bool keyPress(int ascii) {
	return (texture(iChannel2,vec2((.5+float(ascii))/256.,0.25)).x > 0.);
}

// Function 344
float isKeyPressed(float key)
{
	return texture( iChannel2, vec2(key, 0.) ).x;
}

// Function 345
Cam CAM_mouseLookAt(vec3 at, float dst)
{
    vec2 res = iResolution.xy; vec2 spdXY = vec2(15.1416,4.0);
    float fMvtX = (iMouse.x/res.x)-0.535;
    if(fMvtX>0.3) dst *= (1.0+(fMvtX-0.3)/0.03);
    else if(fMvtX<-0.3) dst *= (1.0-(fMvtX+0.3)/(-0.2));
	fMvtX += iTime*0.0150;//Auto turn
    return CAM_lookAt(at,spdXY.y*((iMouse.y/res.y)-0.5),dst,spdXY.x*fMvtX);
}

// Function 346
bool keypress(int code) {
#if !defined LOCAL
  return texelFetch(iChannel0, ivec2(code,2),0).x != 0.0;
#else
  return false;
#endif
}

// Function 347
bool isKeyPressed(int key)
{
	return texelFetch( iChannel3, ivec2(key, 0), 0 ).x != 0.0;
}

// Function 348
bool isKeyPressed(float key)
{
	return texture(iChannel0, vec2(key, 0.25) ).x > .0;
}

// Function 349
float keypress( int key )
	{ return 0.; }

// Function 350
float getKeyPressF(int key) {
    return texelFetch(iChannel3, ivec2(key, 0), 0).x;
}

// Function 351
vec3 getMouse(vec3 ro)
{    
    vec4 mPtr = iMouse;
    mPtr.xy = mPtr.xy / iResolution.xy - 0.5;
     float tCur = iTime;
     float az = 0.;
     float el = -0.15 * PI;
    az += 2. * PI * mPtr.x;
    el += PI * mPtr.y;
     mat3 vuMat = StdVuMat (el, az);
	return ro*vuMat;
}

// Function 352
float keyMom(float key)
{
    return IF_NEQ(texture( iChannel3, vec2(key, 0.0) ).x,0.) ;
}

// Function 353
vec2 KeyboardInput() {
    INPUT_METHOD
    
	vec2 i = vec2(key(KEY_BIND_RIGHT)   - key(KEY_BIND_LEFT), 
                  key(KEY_BIND_FORWARD) - key(KEY_BIND_BACKWARD));
    
    float n = abs(abs(i.x) - abs(i.y));
    return i * (n + (1.0 - n)*inversesqrt(2.0));
}

// Function 354
void handlePlayerInput(inout vec4 posVel, inout vec4 data, inout vec4 special, int keyL, int keyR, int keyF, int keyB, int keySpecial1, int keySpecial2)
{
    // unpack data
    vec2 pos = posVel.xy;
    vec2 vel = posVel.zw;
    
    vec2 deltaTemp 	= vec2(0.0);
    
    float moveRight 	= texelFetch( iChannel1, ivec2(keyR			, 0), 0 ).x;
    float moveLeft  	= texelFetch( iChannel1, ivec2(keyL			, 0), 0 ).x;
    float moveForw  	= texelFetch( iChannel1, ivec2(keyF			, 0), 0 ).x;
    float moveBack  	= texelFetch( iChannel1, ivec2(keyB			, 0), 0 ).x;
    float inputSpecial1	= texelFetch( iChannel1, ivec2(keySpecial1	, 0), 0 ).x;
    float inputSpecial2	= texelFetch( iChannel1, ivec2(keySpecial2	, 0), 0 ).x;

    float inputSum = 0.0;
    inputSum += moveRight;
    inputSum += moveLeft;
    inputSum += moveForw;
    inputSum += moveBack;
    float wasInput = abs(inputSum);
    
    
    deltaTemp.x = steeringStrength * (moveRight - moveLeft);
    deltaTemp.y = steeringStrength * (moveForw  - moveBack);
    
    
    if (AUTO_DANCE)
    {
        if (DANCE_ALL_THE_TIME || (special.w > AUTO_DANCE_TIMER * 75.0))
        {
            // dancing
            deltaTemp.xy += 0.03 * vec2(cos(2.3 * iTime), sin(3.9 * iTime));
            deltaTemp = clamp(deltaTemp, vec2(-1.0), vec2(1.0));
        }
    }    
    
    vel = mix(vel, deltaTemp, vec2(0.05));
    
    pos += vel;
    
    data.xy = mix(data.xy, pos, vec2(0.08));

    float lenVel = length(vel);

    // anim timer
    data.z += 1.8 * lenVel;
    
    // anim timer (smooth)
    data.w = mix(data.w, data.z, 0.1);

    // crouch
    special.x = mix(special.x, inputSpecial1, 0.03);
    
    // sing
    special.y = mix(special.y, inputSpecial2, 0.12);
    
    // timer without input
    if (wasInput > 0.01)
    {
    	special.w = 0.0;
    }
    else
    {
    	special.w += 1.0;
    }
    
    // debug
    //special.y = (special.w > AUTO_DANCE_TIMER * 75.0) ? 1.0 : 0.0;
    
    
    // pack data
    posVel.xy = pos;
    posVel.zw = vel;
}

// Function 355
void mixKeyFrame(KeyFrame a, KeyFrame b, float ratio, out KeyFrame c)
{
    c.eyePos		 = mix(a.eyePos			,b.eyePos		  ,ratio);
    c.eyelidsOpen	 = mix(a.eyelidsOpen	,b.eyelidsOpen	  ,ratio);
	c.eyeOpening	 = mix(a.eyeOpening		,b.eyeOpening	  ,ratio);
    c.browBend		 = mix(a.browBend		,b.browBend		  ,ratio);
    c.moustacheBend  = mix(a.moustacheBend  ,b.moustacheBend  ,ratio);
    c.mouthOpenVert  = mix(a.mouthOpenVert  ,b.mouthOpenVert  ,ratio);
    c.mouthOpenHoriz = mix(a.mouthOpenHoriz ,b.mouthOpenHoriz ,ratio);
    c.bendX			 = mix(a.bendX			,b.bendX		  ,ratio);
    c.twistX		 = mix(a.twistX			,b.twistX		  ,ratio);
    c.headRotation	 = mix(a.headRotation	,b.headRotation	  ,ratio);
}

// Function 356
bool isKeyPressed(int keyCode) { return texelFetch(iChannel3, ivec2(keyCode, 1), 0).x == 1.0; }

// Function 357
bool isKeyEnabled (in int key) {
    return bool(texelFetch(iChannel2, ivec2(key, 1), 0).x);
}

// Function 358
float keyPress(float ascii) {
    return texture(iChannel1, vec2((float(ascii))/256., 0.0)).x ;
}

// Function 359
void handle_mouse(in vec2 p, inout vec4 fragColor)
{
    if (iMouse.z > 0.) {
        val = ceil(iMouse.x / iResolution.x * 4096.) *
              ceil(iMouse.y / iResolution.y * 4096.);
        
		float w = 0.001;
        float a = sin(10.*iTime);
		float l =
            sharpen(df_circ(p, iMouse.xy / iResolution.xy * vec2(iResolution.x / iResolution.y, 1.), 
                    .025 + a * .005), w);
   		if (l > 0.) fragColor = vec4(mix(CM, BG, a), 1);
    }
}

// Function 360
int getKeyChar(){
    int key = -1;
    for (int i=0; i<256; i++) {
        if (keyDown(i) && i >= START_CH && i<=122) {
            key = (i-START_CH); 
            break;
        }
    }
    return key;
}

// Function 361
float mouseNeg(float c,float m){return sign(c)*min(m,abs(c));}

// Function 362
bool ReadKey( int key, bool toggle ){
 float keyVal = textureLod( iChannel3,vec2((float(key)+.5)/256.,toggle?.75:.25),.0).x;
 return (keyVal>.5)?true:false;}

// Function 363
bool keyToggle(int ascii) {
	return !(texture(iChannel2,vec2((.5+float(ascii))/256.,0.75)).x > 0.);
}

// Function 364
vec3 Key1Col()
{
	return mix ( vec3(.4,.6,.9), vec3(1), smoothstep( 0.7, 1.0, sin((pixel.x+pixel.y)*.1-6.0*iTime) ) );
}

// Function 365
float is_input_enabled()				{ return step(INPUT_ACTIVE_TIME, g_time); }

// Function 366
vec2 mouseDelta(){
    vec2 pixelSize = 1. / iResolution.xy;
    float eighth = 1./8.;
    vec4 oldMouse = texture(iChannel3, vec2(7.5 * eighth, 3.5 * eighth));
    vec4 nowMouse = vec4(iMouse.xy / iResolution.xy, iMouse.zw / iResolution.xy);
    if(oldMouse.z > pixelSize.x && oldMouse.w > pixelSize.y && 
       nowMouse.z > pixelSize.x && nowMouse.w > pixelSize.y)
    {
        return nowMouse.xy - oldMouse.xy;
    }
    return vec2(0.);
}

// Function 367
float get_key(int key_code) {
    return texelFetch(CHANNEL_KEYS, ivec2(key_code,0), 0).x;
}

// Function 368
bool isMousePressed() { return iMouse.z > 0.0; }

// Function 369
float ReadKeyFloat(int key)
{
	float keyVal = texture( iChannel0, vec2( (float(key)+.5)/256.0, .25 ) ).x;
	return keyVal;
}

// Function 370
void handleMouse( inout vec4 buffer, in vec2 fragCoord )
{
    // Load prior mouse position.
    vec2 pMouse = readTexel(KEYS_BUFFER,txMOUSE).xy;
    // Load last frame's difference between where the mouse is,
    // and where the mouse was most recently clicked.
    vec2 pMouseDX = readTexel(KEYS_BUFFER,txMOUSEDX).xy;
    
    // Get the current mouse position.
    vec4 curMouse = iMouse;
    
    vec2 cMouseDX;
    if(curMouse.z > 0.0 && curMouse.w > 0.0) // LMB down.
    {
        // current difference between click and cur.
        cMouseDX = curMouse.xy-curMouse.zw;
        pMouse += (cMouseDX-pMouseDX)*.5;
        
        // We don't want to look up or down too far, so...
        pMouse.y = clamp(pMouse.y,-iResolution.y,iResolution.y);
    }
    // And store it.
    write2(buffer.xy,pMouse,txMOUSE,fragCoord);
    write2(buffer.xy,cMouseDX,txMOUSEDX,fragCoord);
    
}

// Function 371
bool KeyIsPressed(int key)
{
	return texelFetch( iChannel1, ivec2(key, 0), 0 ).x > 0.0;
}

// Function 372
bool ReadKey( int key, bool toggle )
{
	float keyVal = texture( iChannel2, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ) ).x;
	return (keyVal>.5)?true:false;
}

// Function 373
VehicleInputs vi_read_inputs( VehicleState vs )
{
	VehicleInputs result;

	float shift = keystate( KEY_SHIFT );
    float meta = max( keystate( KEY_CTRL ),
                     max( keystate( KEY_META_FIREFOX ), keystate( KEY_META_CHROME ) ) );
    vec2 arrows = vec2(
		keystatepress( KEY_RIGHT ) - keystatepress( KEY_LEFT ),
        keystatepress( KEY_UP ) - keystatepress( KEY_DOWN ) );
    vec2 WASD = vec2(
        max( keystatepress( KEY_A ), keystatepress( KEY_Q ) ) - keystatepress( KEY_D ),
		max( keystatepress( KEY_W ), keystatepress( KEY_Z ) ) - keystatepress( KEY_S ) );
	float shiftmod = mix( 1., .25, shift );

    result.flapsswitch = keypress( KEY_F ) * ( 1. - 2. * shift );
    result.spoiltoggle = keypress( KEY_V );
    result.gearstoggle = keypress( KEY_G );
    result.gearbrake =
        shiftmod * max( keystate( KEY_B ), keystate( KEY_SPACE ) );
    result.lightstoggle = keypress( KEY_L );
    result.throttlecommand =
        keystate( KEY_SPACE ) > 0. ? -9999. :
    	vs.modes.z == VS_ENG_OFF ? 0. :
        mix( 1., abs( vs.throttle ) < 0.1 ? .0625 : .25, shift ) * WASD.y;

    result.joycommand = ZERO;
    result.trimcommand = 0.;
    result.trimdisplay = false;

    if( vs.modes2.x != VS_AERO_OFF && meta > 0. )
  	{
		result.trimdisplay = true;
        result.trimcommand = mix( 1., .25, shift ) * -arrows.y;
        arrows.y = 0.;
	}

    result.joycommand =
    	shiftmod * vec3( -arrows.y, arrows.x, WASD.x );

    result.tvecswitch = max( keypress( KEY_LESS ),
				        max( keypress( KEY_ACCENT_FIREFOX ),
             				 keypress( KEY_ACCENT_CHROME ) ) ) * ( 2. * shift - 1. );

    result.vjoy_copy = ZERO;
    return result;
}

// Function 374
bool Key(in int key){return (texture(iChannel1,vec2((float(key)+0.5)/256.0, 0.25)).x>0.0);}

// Function 375
bool Key_Typematic( sampler2D samp, int key)
{
    return texelFetch( samp, ivec2(key, 1), 0 ).x > 0.0;    
}

// Function 376
void initkeys() {
  animate = keypress(CHAR_A);
  centre = keypress(CHAR_C);
  dorotate = !keypress(CHAR_D);
  edge = keycount(CHAR_E)%4;
  gradient = !keypress(CHAR_G);
  halfspace = keypress(CHAR_H);
  ilwidth = (1+keycount(CHAR_L))%4;
  //omnitruncated = keypress(CHAR_O);
  region = keypress(CHAR_R);
  slice = keypress(CHAR_S);
  pqr = keycount(CHAR_T)%maxPQR;
  centrevertex = keycount(CHAR_V)%4;
  showparams = keypress(CHAR_X);
}

// Function 377
bool isKeyPressed (in int key) {
    return bool(texelFetch(iChannel1, ivec2(key, 0), 0).x);
}

// Function 378
bool isKeyPressed(float k) {
    return texture(iChannel1, vec2(k, 0)).x > 0.0;
}

// Function 379
vec2 Mouse()
{
    vec2 a = 2.0 *(iMouse.xy / iResolution.xy) - 1.0;
    a.x *= iResolution.x / iResolution.y;
    return a;
    
}

// Function 380
vec4 DrawKeys( vec2 uv, float keyID, vec2 size, float state)
{
    vec4 ret = vec4(0.);
    if (state > 0.5)
        size.y *=1.8;

    vec2 pos = vec2(keyID*0.2+size.x/2., size.y/2.);
    
    if ( abs(uv - pos).x < size.x/2. && abs(uv-pos).y < size.y/2. )
    {
        
        if ( state > 0.5 )
        {
            ret.a = 1.;
            ret.xyz = keyColors[int(keyID)];
            }
       	else
        {
            ret.a = 0.7;
            ret.xyz = keyColors[int(keyID)];
        }
           
    }
    return ret;
}

// Function 381
vec4 GetLastMouse()
{
    return texture(iChannel1, I2UV(0, iResolution.xy));
}

// Function 382
bool ReadKey( int key, bool toggle )
{
	float keyVal = texture( iChannel0, vec2( (float(key) + .5) / 256.0, toggle ? .75 : .25 ) ).x;
	return (keyVal > .5) ? true : false;
}

// Function 383
bool isKeyHeld   (int keyCode) { return texelFetch(iChannel3, ivec2(keyCode, 0), 0).x == 1.0; }

// Function 384
bool isKeyUp(in int key) {
  return texelFetch(iChannel1, ivec2(key, 1), 0).x > 0.0;
}

// Function 385
vec2 Keyboard(vec2 offset) {

    float velocity = 0.1; // This will cause offset to change by 0.01 each time an arrow key is pressed
    
 
    vec2 left = texelFetch(iChannel2, ivec2(KEY_LEFT, 0), 0).x * vec2(-1, 0);
    vec2 up = texelFetch(iChannel2, ivec2(KEY_UP,0), 0).x * vec2(0, 1);
    vec2 right = texelFetch(iChannel2, ivec2(KEY_RIGHT, 0), 0).x * vec2(1, 0);
    vec2 down = texelFetch(iChannel2, ivec2(KEY_DOWN, 0), 0).x * vec2(0, -1);
    
    offset += (left + up + right + down) * velocity;

    return offset;
}

// Function 386
vec3 GetLastMouseClick(){return texelFetch(iChannel0,LastMouseClick,0).xyz;}

// Function 387
bool KeyDown(in int key){return (texture(iChannel1,v1((v0(key)+0.5)/256.0, 0.25)).x>0.0);}

// Function 388
void LoadInputs(out Inputs inp)
{
    inp.button = iMouse.z >= 0.;
    inp.mouse = iMouse.xy;
    inp.move = vec3(key(KEY_RT) - key(KEY_LF)
                  , key(KEY_UW) - key(KEY_DW)
                  , key(KEY_FW) - key(KEY_BW));
    inp.move += vec3(key(KEY_RIGHT) - key(KEY_LEFT)
                  , key(KEY_PGUP) - key(KEY_PGDN)
                  , key(KEY_UP) - key(KEY_DOWN));  // arrows for alternate input
    if (iMouse.z >= 0. && dot(iMouse.xy, iMouse.xy) < 2.) { // preview icon?
        inp.mouse.y = R.y*.5; // don't look at ground
        inp.mouse.x = iTime * -.01 * R.x;
        inp.move.z = .1;
    }
    inp.dt = iTimeDelta; //1./30.; //1./60.; // can lock frame delta for debugging
}

// Function 389
float getKey(float key) {return texture(iChannel1, vec2(key, .25)).x;}

// Function 390
float get_key(int key_code) {
    return texelFetch(KEY_BUFFER, ivec2(key_code,0), 0).x;
}

// Function 391
bool ReadKey( int key, bool toggle )
{
	float keyVal = textureLod( iChannel3, vec2( (float(key)+.5)/256.0, toggle?.75:.25 ), 0.0 ).x;
	return (keyVal>.5)?true:false;
}

// Function 392
float inputState(in ivec2 ip)
{
    vec2 p = (vec2(ip) + vec2(0.5, 1.5)) / iChannelResolution[0].xy;
    return texture(iChannel0, p).x;
}

