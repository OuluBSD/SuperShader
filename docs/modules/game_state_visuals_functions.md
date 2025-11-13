# game_state_visuals_functions

**Category:** game
**Type:** extracted

## Dependencies
texture_sampling, normal_mapping, lighting, raymarching

## Tags
lighting, texturing, color, game

## Code
```glsl
// Reusable Game State Visuals Game Functions
// Automatically extracted from game/interactive-related shaders

// Function 1
void gs_process_menu( inout GameState gs )
{
    gs.menustate.y = 0;
    if( gs.menustate.x > 0 )
    {
        uvec4 currmenu = md_load( iChannel0, gs.menustate.x );
        int n = int( currmenu.w >> 8 ) & 0xff;
        int p = int( currmenu.w >> 16 ) & 0xff;

    #if WORKAROUND_02_FOR_IF
        for( int i = 0, n = 8; i < n; ++i )
            if( i < n )
            	gs_check_menu_item( gs, i, p );
	#else
        n += NOUNROLL;
		for( int i = 0; i < n; ++i )
	       	gs_check_menu_item( gs, i, p );
	#endif
    }
}

// Function 2
void SetState ()
{
  float t;
  tBuild = 14.;
  t = mod (tCur, tPlay);
  onTrk = (t < 8. || t > tBuild);
  onTxt = (t < tBuild + 9.);
  if (t < 0.5 * tPlay) {
    showObj[SH_TrkSt] = (t > 2.);
    showObj[SH_TrkCv] = (t > 3.);
    showObj[SH_Plat] = (t > 4.);
    showObj[SH_Sig] = (t > 5.);
    showObj[SH_Tun] = (t > 6.);
    showObj[SH_Tree] = (t > 7.);
    showObj[SH_Engn] = (t > 9.);
    showObj[SH_Cars] = (t > 10.);
  } else {
    t = tPlay - t + 0.5;
    showObj[SH_TrkSt] = (t > 1.);
    showObj[SH_TrkCv] = (t > 2.);
    showObj[SH_Plat] = (t > 3.);
    showObj[SH_Sig] = (t > 3.);
    showObj[SH_Tun] = (t > 4.);
    showObj[SH_Tree] = (t > 5.);
    showObj[SH_Engn] = (t > 6.);
    showObj[SH_Cars] = (t > 7.);
  }
}

// Function 3
void GameState_Store( GameState gameState, inout vec4 fragColor, in vec2 fragCoord )
{    
    vec4 vData0 = vec4( gameState.iMainState, gameState.fSkill, gameState.fGameTime, gameState.fStateTimer );    

    vec4 vData1 = vec4( gameState.vPrevMouse );    

    vec4 vData2 = vec4( gameState.fMap, gameState.fHudFx, gameState.iMessage, gameState.fMessageTimer );

    ivec2 vAddress = ivec2( 0 );
    StoreVec4( vAddress, vData0, fragColor, fragCoord );
    vAddress.x++;

    StoreVec4( vAddress, vData1, fragColor, fragCoord );
    vAddress.x++;

    StoreVec4( vAddress, vData2, fragColor, fragCoord );
    vAddress.x++;
}

// Function 4
void WheelLoadState( out Wheel wheel, ivec2 addr )
{    
    vec4 vState = LoadVec4( addr + offsetWheelState );
    
    wheel.fSteer = vState.x;
    wheel.fRotation = vState.y;
    wheel.fExtension = vState.z;
    wheel.fAngularVelocity = vState.w;
    
    // output data
    wheel.vContactPos = vec2( 0.0 );
    wheel.fOnGround = 0.0;
    wheel.fSkid = 0.0;
}

// Function 5
void SaveState(inout vec4 c, State state, ivec2 p)
{
    ivec2 R = state.resolution;
    if (p.y == R.y - 1) switch (R.x - 1 - p.x) {
      case slotResolution:
        c.xy = vec2(R);
        break;
      case slotEyePos:
        c.xy = state.eyepos;
        break;
      default:
        break;
    }
}

// Function 6
void load_state(in vec2 fragCoord, bool ctrl) {
    vec2 uv = fragCoord / iResolution.xy;
    vec4 px = texelFetch(iChannel0, ivec2(2, 0), 0);
    float cards_player = floor(px.x);
    float flag0 = floor(px.y);
    float cards_player_atime = (px.z);
    float cards_player_select = floor(px.w);
    px = texelFetch(iChannel0, ivec2(2, 2), 0);
    float card_put_anim = (px.x);
    float card_hID_put_anim = floor(px.y);
    float card_bID_put_anim = floor(px.z);
    float flag1 = floor(px.w);
    vec2 click_pos = vec2(0.);
    float card_select_anim=0.;
    px = texelFetch(iChannel0, ivec2(2, 4), 0);
    vec3 tvg = decodeval(px.x);
    vec2 player_hpmp = tvg.yz;
    bool player_etf = tvg.x == 1.;
    tvg = decodeval(px.y);
    vec2 en_hpmp = tvg.yz;
    bool en_etf = tvg.x == 1.;
    float ett = px.z;
    bool player_turn = ett < 0.;
    ett = abs(ett);
    float flag3 = 0.;
    float egt = px.w;
    if ((flag0 != 1.) || (g_time < extime + 0.1)) {
        egt = 0.;
    } else {
        if (player_hpmp.x < 1.) {
            flag3 = 1.;
            egt = px.w;
        }
        if (en_hpmp.x < 1.) {
            flag3 = 1.;
            egt = px.w;
        }
    }
    if ((iMouse.z > 0.)&&(ctrl)) {
        float anim_t2 = 1. - get_animstate(clamp((g_time - card_put_anim - 0.5)*2., 0., 1.));
        float anim_t = get_animstate(clamp(1. - (g_time - cards_player_atime), 0., 1.));
        if ((flag3 == 0.)&&(anim_t == 0.)&&(anim_t2 == 0.)) { //do not update mouse if anim played
            click_pos = click_control();
            if ((player_hpmp.y > 0.)&&((card_get_hit(click_pos) >= 0) || (hpmp_get_hit_preinit(click_pos,player_etf) > 0))) {
                px = texelFetch(iChannel0, ivec2(2, 1), 0);
                card_select_anim = px.z;
            } else card_select_anim = g_time;
        } else {
            card_select_anim = g_time;
        }
    } else {
        px = texelFetch(iChannel0, ivec2(2, 1), 0);
        card_select_anim = px.z;
        click_pos = px.xy;
    }
    px = texelFetch(iChannel0, ivec2(2, 1), 0);
    float card_draw = floor(px.w);
    if(flag3==1.)card_draw=0.;
    allData = allData_struc(cards_player, card_select_anim, cards_player_atime, click_pos, -1., cards_player_select, card_put_anim, card_hID_put_anim, card_bID_put_anim, flag1, flag0,
            player_hpmp, en_hpmp, flag3, egt, card_draw, player_turn, ett, player_etf, en_etf);
}

// Function 7
bool IsStatePixel(ivec2 i, ivec2 R)
{
   	return i.y == R.y-1 && i.x >= R.x-1-slotCount;
}

// Function 8
void StateOfRobot(out Storage s, Robot r, Guts g)
{
    for (int i = 0; i < r.data.length(); ++i) // SU
        s[i] = r.data[i];
    s[SR + 0].xy = r.pos;
    s[SR + 0].zw = r.vel;
    s[SR + 1].z = r.scan;
    s[SR + 1].w = r.damage;
    s[SR + 2].x = r.aim;
    s[SR + 2].y = r.turn;
    s[SR + 2].z = r.radar;
    s[SR + 2].w = r.shot;
    s[SG + 0].xy = g.pos;
    s[SG + 0].zw = g.vel;
    s[SG + 1].xy = g.bullet.pos;
    s[SG + 1].z  = g.bullet.dir;
    s[SG + 1].w  = g.bullet.timer;
    s[SG + 2].xy = g.explosionpos;
    s[SG + 2].w = g.explosiontimer;
    s[SG + 3].x = g.bearing;
    s[SG + 3].y = g.health;
    s[SG + 3].z = g.reload;
    s[SG + 3].w = g.alive;
}

// Function 9
float outputState2(in int op)
{
    vec2 p = vec2(float(op)+.5, 1.5) / iChannelResolution[1].xy;
    return texture(iChannel1, p).x;
}

// Function 10
float keystate( int key )
	{ return 0.; }

// Function 11
void saveState(vec2 fragCoord, inout vec4 fragValue) 
{
    int pos = 1;
    saveVec3(state.camRight, fragCoord, pos, fragValue);		
    saveVec3(state.camUp, fragCoord, pos, fragValue);			
    saveVec3(state.camForward, fragCoord, pos, fragValue);		
    saveVec3(state.camPosition, fragCoord, pos, fragValue);		
    saveVec2(state.camAngle, fragCoord, pos, fragValue);		
    saveFloat(state.camFovy, fragCoord, pos, fragValue);		
    saveVec2(state.lastMousePos, fragCoord, pos, fragValue);	
    saveFloat(state.isMouseDragging, fragCoord, pos, fragValue);
    saveFloat(state.tilting, fragCoord, pos, fragValue);		
    saveFloat(state.rolling, fragCoord, pos, fragValue);
    saveFloat(state.speed, fragCoord, pos, fragValue);
}

// Function 12
bool key_state( float ascii ) { return (texture( iChannel0, vec2( ( ascii + .5 ) / 256., 0.25 ) ).x > 0.); }

// Function 13
void delState(inout int state, const int value)
{
    state = state & ~value;
}

// Function 14
void ObjState ()
{
  obRnd = Hashv3v3 (cId);
  obDisp = vec3 (0.6, 0.6, 0.2) * bGrid * (obRnd - 0.5);
  obRotCs = cos (0.2 * pi * (obRnd.z - 0.5) * sin (1.5 * pi * (obRnd.y - 0.5) * tCur) +
     vec2 (0., 0.5 * pi));
}

// Function 15
float keyState(float key, float default_state) {
    return abs( texture(iChannel0, vec2(key, 0.75)).x - default_state );
}

// Function 16
void loadState() 
{
    int pos = 1;
    state.camRight = loadVec3(pos);			
    state.camUp = loadVec3(pos);			
    state.camForward = loadVec3(pos);		
    state.camPosition = loadVec3(pos);		
    state.camAngle = loadVec2(pos);		
    state.camFovy = loadFloat(pos);			
    state.lastMousePos = loadVec2(pos);		
    state.isMouseDragging = loadFloat(pos);	
	state.tilting = loadFloat(pos);
	state.rolling = loadFloat(pos);
    state.speed = loadFloat(pos);
}

// Function 17
void Cam_StoreState( ivec2 addr, const in CameraState cam, inout vec4 fragColor, in ivec2 fragCoord )
{
    StoreVec4( addr + ivec2(0,0), vec4( cam.vPos, 0 ), fragColor, fragCoord );
    StoreVec4( addr + ivec2(1,0), vec4( cam.vTarget, cam.fFov ), fragColor, fragCoord );    
    StoreVec4( addr + ivec2(2,0), vec4( cam.vJitter, cam.fPlaneInFocus, 0 ), fragColor, fragCoord );    
}

// Function 18
bool
    paintMenu( ) {
        
        V2
            MSZ = V2( R.x, .3 * R.y );
        
        MSZ.y = ( .9 * PX( Ri * V2( 10.5, 7.5 ) ).w + .1 ) * R.y;
        
        if( I.y < R.y - MSZ.y )
            
            return false;
        
        V2
            sw    = V2( 11., 2. ) * V2( I.x - .5, I.y - .5 - R.y + MSZ.y ) / MSZ,
            swId  = floor( sw ),
            swVal = fract( sw );
        
        V4
            val = PX( Ri * V2( swId.x + .5, 6.5 + swId.y ) );
        
        F
            sel = floor( 4. * ( sw - swId ).y ),
            v   = 0.;
        
        V3
            col = X.yyy;
        
        if( sel < 2. ) {
            
            if( sel < 1. ) {
                
                v   = val.r;
                col = X.zyy;
            }
            else {
                
                v   = val.g;
                col = X.yzy;
            }
        }
        else {
            
            if( sel < 3. ) {
                
                v   = val.b;
                col = X.yyz;
            }
            else {

                v   = val.a;
                col = X.yyy;
            }
        }
        
        O = V4( mix( .3 * V3( ( 1. - 8. * ( swVal.x - .5 ) * ( swVal.x - .5 ) ) * col ), .75 *( .5 * X.zzz + col ), exp( 200. * ( 1.2 * ( swVal.x - .1 ) - v ) * ( v - 1.2 * ( swVal.x - .1 ) ) ) ), 1 );
        
        return true;
    }

// Function 19
void drawMenu(inout vec4 fragColor, in vec2 uv)
{
    if (uv.y < .9) return;
    uv.y -= .9; uv /= .1;
    
    fragColor.xyz *= .2;
    fragColor.xyz += .4;
    
    // current-tool frame
    float frame = max(0., 1.-abs(uv.x-.5 - guistate(G_TOOL).x));
    fragColor.xyz += .6*frame;
    
    // state-brushes
	if (uv.x > 0. && uv.x < 4.)
    	fragColor.xyz = mix(fragColor.xyz, .2+.8*wwcolor(vec4(floor(uv.x),0,0,0))
                         , smoothstep(.5,.3, length(mod(uv,1.)-.5)));
    uv.x -= 4.; 
    // move tool
    if (uv.x > 0. && uv.x < 1.)
    	fragColor.xyz += smoothstep(.1,.0, 
			min(abs(uv.x-.5), abs(uv.y-.5)));
    uv.x -= 1.;
    // zoom tool
    if (uv.x > 0. && uv.x < 1.)
    {
        float l = length(uv-vec2(.6))-.35;
        if (uv.x<.5 && uv.y<.5)
        	l = min(l, abs(uv.x-uv.y)-.03);
        fragColor.xyz += smoothstep(.1,.0, max(l,-l));
    }
    
    uv.x -= 1.;
    // speed bar
    if (uv.x > 0. && uv.x < 4.)
    {
        fragColor.xyz += .3;
        fragColor.xyz += smoothstep(0.1,0., uv.x/4. - guistate(G_SPEED).x)
            			*( vec3(.4,.8,1) - fragColor.xyz);
	}

}

// Function 20
void SetState ()
{
  float tWalk, tCyc, wkSpd, nCyc;
  gGap = 6.;
  tWalk = 6.;
  tCyc = tWalk + 4.;
  wkSpd = gGap / tWalk;
  nCyc = floor (tCur / tCyc);
  tPhs = tCur - tCyc * nCyc;
  wDisp = nCyc * tWalk * wkSpd;
  if (tPhs < tWalk) {
    wkState = 1;
    wDisp += tPhs * wkSpd;
  } else {
    wkState = 0;
    wDisp += tWalk * wkSpd;
  }
}

// Function 21
mat4 getState() {
    //half pixel
    vec2 pxSz = 0.5 / iResolution.xy;
    return mat4(
        tex(iChannel3, vec2(pxSz.x,1.)),	 //loc
        tex(iChannel3, vec2(pxSz.x*3.,1.)),	 //vel
        tex(iChannel3, vec2(1.-pxSz.x,1.)),	 //rot
        tex(iChannel3, vec2(1.-pxSz.x*3.,1.))//mou
    );
}

// Function 22
float get_animstate(float timeval) {
    return SS(0., 1., timeval);
}

// Function 23
bool gs_pace_camera_transition( inout GameState gs, VehicleState vs, float dt )
{
    float u = -expm1( -1.00 * dt * min( 1., .5 * gs.timer ) );
    float v = -expm1( -1.50 * dt * min( 1., .5 * gs.timer ) );
    vec3 a = normalize( mix( gs.camframe[0], normalize( vs.localr - gs.campos ), v ) );
    vec3 b = normalize( mix( normalize( gs.camframe[1] - a * dot( gs.camframe[1], a ) ),
                             normalize( vs.localB[1] - a * dot( vs.localB[1], a ) ), v ) );
    vec3 target = vs.localr - 0.03 * vs.localB[0];
    float alt = mix( length( gs.campos ), length( target ) + .150 * length( target - gs.campos ), u );
    gs.campos = normalize( mix( gs.campos, target, u ) ) * alt;
    gs.camframe = mat3( a, b, cross( a, b ) );
    return dot( normalize( vs.localr - gs.campos ), vs.localB[0] ) >= FRACT_15_16;
    // return distance( gs.campos, vs.localr ) < ( length( target ) - g_data.radius < 25. ? 0.0305 : 0.3 );
}

// Function 24
void LoadState(out State state, sampler2D A, ivec2 R)
{
	vec4[slotCount] data; // at least temporarily
	for (int i = slotCount; i-- > 0; )
        data[i] = fetch(A, R-1-ivec2(i,0));
    state.resolution = ivec2(data[slotResolution].xy);
    state.eyepos = data[slotEyePos].xyz;
    state.eyeaim = data[slotEyeAim].xy;
}

// Function 25
void Enemy_SetState( inout Entity entity, int iNewState )
{
    if ( Enemy_GetState(entity) == ENEMY_STATE_DIE )
    {
        return;
    }
    
    entity.fArmor = float(iNewState);
    
    bool setRandomTimer = false;
    
    if ( iNewState == ENEMY_STATE_PAIN )
    {
        entity.fTimer = 0.2;    
    }
    else
    if ( iNewState == ENEMY_STATE_DIE )
    {
        SetFlag( entity.iFrameFlags, ENTITY_FRAME_FLAG_DROP_ITEM );
        entity.fTimer = 0.4;    
    }
    else
#ifdef CHEAT_NOAI 
    {
    	entity.fArmor = float(ENEMY_STATE_STAND);        
        entity.fTimer = 0.4;    
    }
#else        
    if ( iNewState == ENEMY_STATE_FIRE )
    {
        SetFlag( entity.iFrameFlags, ENTITY_FRAME_FLAG_FIRE_WEAPON );
        entity.fTimer = 0.3;    
    }
    else
    if ( iNewState == ENEMY_STATE_WALK_RANDOM )
    {
    	float fRandom = Hash( float(entity.iId) + iTime + 3.456 );        
        
        entity.fYaw = fRandom * 3.14 * 2.0;

        float fStepScale = 3.14 / 4.0;
        entity.fYaw = floor( entity.fYaw / fStepScale ) * fStepScale;
        
        setRandomTimer = true;
    }
    else
    if ( iNewState == ENEMY_STATE_WALK_TO_TARGET )
    {
        Entity targetEnt = Entity_Read( STATE_CHANNEL, int(entity.fTarget) );
        vec3 vToTarget = targetEnt.vPos - entity.vPos;
        entity.fYaw = atan(vToTarget.x, vToTarget.z);
        
        float fStepScale = 3.14 / 4.0;
        entity.fYaw = floor( entity.fYaw / fStepScale ) * fStepScale;
        
        setRandomTimer = true;
    }
    else
    {
        setRandomTimer = true;
    }    

    if ( setRandomTimer )
    {
    	float fRandom = Hash( float(entity.iId) + iTime + 0.1 );        
        entity.fTimer = 0.5 + fRandom * fRandom * 1.5;
    }
#endif // #ifndef CHEAT_NOAI        
}

// Function 26
vec4 getState(in vec2 p){
    return texture(iChannel0,p);
}

// Function 27
float stateLabel(float state) {
    if (state == 0.0) return halt;
    return label + state;
}

// Function 28
void CameraLoadState( out Camera cam, in vec2 addr )
{
	cam.vPos = LoadVec3( addr + offsetCameraPos );
	cam.vDir = LoadVec3( addr + offsetCameraDir );
    cam.vUp =  LoadVec3( addr + offsetCameraUp );
    cam.vVel = LoadVec3( addr + offsetCameraVel );
}

// Function 29
void setState(inout int state, const int value)
{
    state = state | value;
}

// Function 30
void LoadState(out AppState s)
{
    vec4 data;

    data = LoadValue(0, 0);
    s.menuId    = data.x;
    s.roughness = data.y;
    s.focus     = data.z;
    
    data = LoadValue(1, 0);
    s.focusObjRot   = data.xy;
    s.objRot        = data.zw;
}

// Function 31
void Cam_LoadState( out CameraState cam, sampler2D sampler, ivec2 addr )
{
    vec4 vPos = LoadVec4( sampler, addr + ivec2(0,0) );
    cam.vPos = vPos.xyz;
    vec4 targetFov = LoadVec4( sampler, addr + ivec2(1,0) );
    cam.vTarget = targetFov.xyz;
    cam.fFov = targetFov.w;
    vec4 jitterDof = LoadVec4( sampler, addr + ivec2(2,0) );
    cam.vJitter = jitterDof.xy;
    cam.fPlaneInFocus = jitterDof.z;
}

// Function 32
vec4 SaveState(in AppState s, in vec2 fragCoord)
{
    if (iFrame <= 0)
    {
        s.menuId      = 0.0;
        s.roughness   = 0.5;
        s.focus       = 0.0;
        s.focusObjRot = vec2(0.0);
        s.objRot      = vec2(0.0);
    }
    
    vec4 ret = vec4(0.);
    StoreValue(vec2(0., 0.), vec4(s.menuId, s.roughness, s.focus, 0.0), ret, fragCoord);
    StoreValue(vec2(1., 0.), vec4(s.focusObjRot, s.objRot), ret, fragCoord);
    return ret;
}

// Function 33
void LoadState(out State state, sampler2D A, ivec2 R)
{
	vec4[slotCount] data; // at least temporarily
	for (int i = slotCount; i-- > 0; )
        data[i] = fetch(A, R-1-ivec2(i,0));
    state.resolution = ivec2(data[slotResMBD].xy);
    state.mbdown = data[slotResMBD].z > .5;
    state.eyepos = data[slotEyePos].xyz;
    state.eyeaim = data[slotAzElBase].xy;
    state.aimbase = data[slotAzElBase].zw;
}

// Function 34
float inputState(in ivec2 ip)
{
    vec2 p = (vec2(ip) + offset) / iChannelResolution[0].xy;
    return texture(iChannel0, p).x;
}

// Function 35
void KeyState( sampler2D sampler, out float[5] keys)    
{
    
    keys[0] = texelFetch(sampler, ivec2(68, 0),0).x;
    keys[1] = texelFetch(sampler, ivec2(70, 0),0).x;
    keys[2]  = texelFetch(sampler, ivec2(32, 0),0).x;    
    keys[3] = texelFetch(sampler, ivec2(74, 0),0).x;
    keys[4]  = texelFetch(sampler, ivec2(75, 0),0).x;

}

// Function 36
TeletextState GetState( ivec2 coord )
{
    TeletextState state = TeletextState_Default();
    
    for ( int x = 0; x <= coord.x; x++ )
    {
        // Process commands that are deferred until next character
        switch( state.cmd )
        {
            case CTRL_NUL:
            	TeletextState_SetAlphanumericColor( state, COLOR_BLACK );
            break;
            case CTRL_ALPHANUMERIC_RED:
            	TeletextState_SetAlphanumericColor( state, COLOR_RED );
            break;
            case CTRL_ALPHANUMERIC_GREEN:
            	TeletextState_SetAlphanumericColor( state, COLOR_GREEN );
            break;
            case CTRL_ALPHANUMERIC_MAGENTA:
            	TeletextState_SetAlphanumericColor( state, COLOR_MAGENTA );
            break;
            case CTRL_ALPHANUMERIC_BLUE:
            	TeletextState_SetAlphanumericColor( state, COLOR_BLUE );
            break;
            case CTRL_ALPHANUMERIC_YELLOW:
            	TeletextState_SetAlphanumericColor( state, COLOR_YELLOW );
            break;
            case CTRL_ALPHANUMERIC_CYAN:
            	TeletextState_SetAlphanumericColor( state, COLOR_CYAN );
            break;
            case CTRL_ALPHANUMERIC_WHITE:
            	TeletextState_SetAlphanumericColor( state, COLOR_WHITE );
            break;
            
            case CTRL_GFX_RED:
            	TeletextState_SetGfxColor( state, COLOR_RED );
            break;            
            case CTRL_GFX_GREEN:
            	TeletextState_SetGfxColor( state, COLOR_GREEN );
            break;            
            case CTRL_GFX_YELLOW:
            	TeletextState_SetGfxColor( state, COLOR_YELLOW );
            break;            
            case CTRL_GFX_BLUE:
            	TeletextState_SetGfxColor( state, COLOR_BLUE );
            break;            
            case CTRL_GFX_MAGENTA:
            	TeletextState_SetGfxColor( state, COLOR_MAGENTA );
            break;            
            case CTRL_GFX_CYAN:
            	TeletextState_SetGfxColor( state, COLOR_CYAN );
            break;            
            case CTRL_GFX_WHITE:
            	TeletextState_SetGfxColor( state, COLOR_WHITE );
            break;  
            
            case CTRL_RELEASE_GFX:
            	state.bHoldGfx = false;
            break;            
        }

        state.cmd = -1;
        
        state.char = GetTeletextCode( ivec2(x, coord.y) );
        
        switch( state.char )
        {
            case CTRL_NUL:
            	state.cmd = CTRL_NUL;
            break;            
            case CTRL_ALPHANUMERIC_RED:
            	state.cmd = CTRL_ALPHANUMERIC_RED;
            break;
            case CTRL_ALPHANUMERIC_GREEN:
            	state.cmd = CTRL_ALPHANUMERIC_GREEN;
            break;
            case CTRL_ALPHANUMERIC_MAGENTA:
            	state.cmd = CTRL_ALPHANUMERIC_MAGENTA;
            break;
            case CTRL_ALPHANUMERIC_BLUE:
            	state.cmd = CTRL_ALPHANUMERIC_BLUE;
            break;
            case CTRL_ALPHANUMERIC_YELLOW:
            	state.cmd = CTRL_ALPHANUMERIC_YELLOW;
            break;
            case CTRL_ALPHANUMERIC_CYAN:
            	state.cmd = CTRL_ALPHANUMERIC_CYAN;
            break;
            case CTRL_ALPHANUMERIC_WHITE:
            	state.cmd = CTRL_ALPHANUMERIC_WHITE;
            break;
            case CTRL_GFX_RED:
            	state.cmd = CTRL_GFX_RED;
            break;            
            case CTRL_GFX_GREEN:
            	state.cmd = CTRL_GFX_GREEN;
            break;            
            case CTRL_GFX_YELLOW:
            	state.cmd = CTRL_GFX_YELLOW;
            break;            
            case CTRL_GFX_BLUE:
            	state.cmd = CTRL_GFX_BLUE;
            break;            
            case CTRL_GFX_MAGENTA:
            	state.cmd = CTRL_GFX_MAGENTA;
            break;            
            case CTRL_GFX_CYAN:
            	state.cmd = CTRL_GFX_CYAN;
            break;            
            case CTRL_GFX_WHITE:
            	state.cmd = CTRL_GFX_WHITE;
            break;            
            
            case CTRL_FLASH:
            	state.bFlash = true;
           	break;
            case CTRL_STEADY:
            	state.bFlash = false;
           	break;
            case CTRL_NORMAL_HEIGHT:
            	state.bDoubleHeight = false;
				state.iHeldChar = 0x20;
            break;
            case CTRL_DOUBLE_HEIGHT:
            	state.bDoubleHeight = true;
				state.iHeldChar = 0x20;
            break;
            case CTRL_NEW_BACKGROUND:
            	state.iBgCol = state.iFgCol;
            break;
            case CTRL_BLACK_BACKGROUND:
            	state.iBgCol = COLOR_BLACK;
            break;
            
            case CTRL_HOLD_GFX:
            	state.bHoldGfx = true;
            break;

            case CTRL_RELEASE_GFX:
            	state.cmd = CTRL_RELEASE_GFX;
            break;
            
            case CTRL_CONTIGUOUS_GFX:
            	state.bSeparatedGfx = false;
            break;

            case CTRL_SEPARATED_GFX:
            	state.bSeparatedGfx = true;
            break;
            
            case CTRL_CONCEAL:
            	state.bConceal = true;
            break;
        }
                
        if ( state.bGfx )
        {
            if ( IsHoldCharacter( state.char ) )
            {
            	state.iHeldChar = state.char;
                state.bHeldSeparated = state.bSeparatedGfx;
            }
        }
    }   
    
    return state;
}

// Function 37
void GameStoreState(inout vec4 fragColor, in vec2 fragCoord)
{
    StoreVec3( addrGameState, vec3(gtime, time_scale, input_state), fragColor, fragCoord );
}

// Function 38
void LoadState(out State state, sampler2D A, ivec2 R)
{
	vec4[slotCount] data; // at least temporarily
	for (int i = slotCount; i-- > 0; )
        data[i] = fetch(A, R-1-ivec2(i,0));
    state.resolution = ivec2(data[slotResolution].xy);
    state.eyepos = data[slotEyePos].xy;
}

// Function 39
int TeletextState_GetChar( TeletextState state )
{
    if ( state.bConceal )
    {
        if ( !Reveal() )
        {
        	return _SP;
        }
    }
    
    if ( IsControlCharacter( state.char ) )
    {
        if ( state.bGfx && state.bHoldGfx )
        {
            return state.iHeldChar;
        }
        else
        {
            return _SP;
        }
    }    
    return state.char;
}

// Function 40
void StoreState(inout vec4 fragColor, in vec2 fragCoord)
{
    vec4 state1 = vec4(bikeAPos, bikeAAngle);
    StoreValue(kTexState1, state1, fragColor, fragCoord);
    StoreValue(kTexState1 + ivec2(1, 0), keyStateLURD, fragColor, fragCoord);
    StoreValue(kTexState1 + ivec2(2, 0), vec4(rotQueueA.xy, rotQueueB), fragColor, fragCoord);
    vec4 state2 = vec4(bitsCollected, framerate, gameStage, message);
    StoreValue(kTexState1 + ivec2(3, 0), state2, fragColor, fragCoord);
    StoreValue(kTexState1 + ivec2(4, 0), camFollowPos, fragColor, fragCoord);
    StoreValue(kTexState1 + ivec2(5, 0), indicatorPulse, fragColor, fragCoord);
    vec4 state3 = vec4(bikeBPos, bikeBAngle);
    StoreValue(kTexState1 + ivec2(6, 0), state3, fragColor, fragCoord);
    StoreValue(kTexState1 + ivec2(7, 0), animA, fragColor, fragCoord);
}

// Function 41
float key_state(int key) {
	return textureLod(iChannel3, vec2((float(key) + .5) / 256.0, .25), 0.0).x;
}

// Function 42
vec4 fetchSimState(vec2 fragCoord) {
    return texelFetch(iChannel0, ivec2(fragCoord), 0);
}

// Function 43
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

// Function 44
PrintState UI_PrintState_Init( inout UIContext uiContext, LayoutStyle style, vec2 vPosition )
{
    vec2 vCanvasPos = uiContext.vPixelCanvasPos;
    
    PrintState state = PrintState_InitCanvas( vCanvasPos, vec2(1.0) );
    MoveTo( state, vPosition + UIStyle_FontPadding() );
	PrintBeginNextLine(state, style);

	return state;
}

// Function 45
float inputState2(in ivec2 ip)
{
    vec2 p = (vec2(ip) + vec2(16.5, 1.5)) / iChannelResolution[0].xy;
    return texture(iChannel0, p).x;
}

// Function 46
vec3 colorFromState(vec4 state)
{
    vec3 baseCol = vec3(0.);
    vec3 treeCol = treeColor(state.y);
    vec3 fireCol = fireColor(state.y);
    
    //return vec3(state.y*(1.-step(state.x, 2.*THIRD)),state.y*(1.-step(state.x, THIRD))*step(state.x, 2.*THIRD),0.);
    return fireCol*(1.-step(state.x, 2.*THIRD)) + treeCol*(1.-step(state.x, THIRD))*step(state.x, 2.*THIRD);
}

// Function 47
UIWindowState UI_GetWindowState( UIContext uiContext, int iControlId, int iData, UIWindowDesc desc )
{
    UIWindowState window;    
    
    vec4 vData0 = LoadVec4( iChannelUI, ivec2(iData,0) );
        
    window.rect = Rect( vData0.xy, vData0.zw );
    
    vec4 vData1 = LoadVec4( iChannelUI, ivec2(iData,1) );
    
    window.bMinimized = (vData1.x > 0.0);    
    window.bClosed = (vData1.y > 0.0);    
    
    // Clamp window position so title bar is always on canvas
	vec2 vSafeMin = vec2(24.0);        
	vec2 vSafeMax = vec2(32.0);        
    vec2 vPosMin = vec2( -window.rect.vSize.x + vSafeMin.x, -vSafeMin.y);//vec2( -window.rect.vSize.x, 0.0) + 24.0, -24.0 );
    vec2 vPosMax = uiContext.drawContext.vCanvasSize - vSafeMax;
    window.rect.vPos = clamp( window.rect.vPos, vPosMin, vPosMax );
    
    if ( iFrame == 0 || vData1.z != DIRTY_DATA_MAGIC)
    {
        window.rect = desc.initialRect;
        window.bMinimized = desc.bStartMinimized;
	    window.bClosed = desc.bStartClosed;
    }       
    
    window.uControlFlags = desc.uControlFlags;
    window.vMaxSize = desc.vMaxSize;
    
    window.iControlId = iControlId;
        
    return window;
}

// Function 48
float GetCellState(float cellID, float seed)
{
    float rndState = GetCellRandomValue( cellID, seed );
    rndState = mix(rndState, CS_EMPTY_LANE, step(cellID, CELLS_HEADSTART));

    float cellState = CS_EMPTY_LANE;
    return mix( cellState, rndState, step(0.5, mod(cellID, 2.0)) );
}

// Function 49
void LoadState()
{
    vec4 state1 = LoadValue(kTexState1);
    vec4 state2 = LoadValue(kTexState2);
    vec4 state3 = LoadValue(kTexState3);
    vec4 state4 = LoadValue(kTexState4);
    vec4 state5 = LoadValue(kTexState5);
    
    gGameState            = state1.x;
    gGameStateTime        = state1.y;
    gGameSeed             = state1.z;
    gGameInit             = state1.w;
    gPlayerCoords         = state2.xy;
    gPlayerNextCoords     = state2.zw;
    gPlayerMotionTimer    = state3.x;
    gPlayerRotation       = state3.y;
    gPlayerNextRotation   = state3.z;
    gPlayerScale          = state3.w;
    gPlayerVisualCoords   = state4.xyz;
    gPlayerVisualRotation = state4.w;
    gPlayerDeathCause     = state5.x;
    gPlayerDeathTime      = state5.y;
    gScore                = state5.z;
    gFbScale              = state5.w;
}

// Function 50
float getNeuronState(vec2 ncoord) {
    if (min(ncoord.x, ncoord.y) < 0. ||
        max(ncoord.x-iResolution.x, ncoord.y-iResolution.y) > 0.) return 0.;
    
    return texture(iChannel2, ncoord/iResolution.xy).x;
}

// Function 51
AppState setStateStartGame( in AppState s, float iTime )
{    
    s.stateID 			=  GS_SPLASH;
    s.timeStarted		=  iTime;
    s.playerPos			=  vec2( 0.5, 0.0 );
    s.score				=  0.0;
    s.timeFailed		= -1.0;
    s.timeCollected		= -1.0;
    s.timeAccumulated	=  0.0;
    s.showUI			=  1.0;

    s.coin0Pos		= 0.0;
    s.coin0Taken	= 0.0;
    s.coin1Pos		= 0.0;        
    s.coin1Taken	= 0.0;
    s.coin2Pos		= 0.0;        
    s.coin2Taken	= 0.0;
    s.coin3Pos		= 0.0;        
    s.coin3Taken	= 0.0;    
    
    return s;
}

// Function 52
int readCellState(vec2 id)
{
	return int(1.0 - step(texture(iChannel0, id / iChannelResolution[0].xy).x, 0.99));
}

// Function 53
float expectedOutputState(in int op)
{
    vec2 p = vec2(float(op)+.5, .5) / iChannelResolution[0].xy;
    return texture(iChannel0, p).x;
}

// Function 54
HexSpec transitionHexSpecs(HexSpec a, HexSpec b, HexSpec c) {
    float roundTop = transitionValues(a.roundTop, b.roundTop, c.roundTop);
    float roundCorner = transitionValues(a.roundCorner, b.roundCorner, c.roundCorner);
	float height = transitionValues(a.height, b.height, c.height);
    float thickness = transitionValues(a.thickness, b.thickness, c.thickness);
    float gap = transitionValues(a.gap, b.gap, c.gap);
	return HexSpec(roundTop, roundCorner, height, thickness, gap);
}

// Function 55
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

// Function 56
vec3 readScoreStates()
{
  	return getPixel(2,0).xyz;
}

// Function 57
void SaveState(inout vec4 c, State state, ivec2 p)
{
    ivec2 R = state.resolution;
    if (p.y == R.y - 1) switch (R.x - 1 - p.x) {
      case slotResolution:
        c.xy = vec2(R);
        break;
      case slotEyePosAz:
        c = vec4(state.eyepos, state.eyeaim.x);
        break;
      case slotEyeVelEl:
        c = vec4(state.eyevel, state.eyeaim.y);
        break;
      default:
        break;
    }
}

// Function 58
UIWindowState UI_GetWindowState( UIContext uiContext, int iControlId, int iData, UIWindowDesc desc )
{
    UIWindowState window;    
    
    vec4 vData0 = LoadVec4( iChannelUI, ivec2(iData,0) );
        
    window.rect = Rect( vData0.xy, vData0.zw );
    
    vec4 vData1 = LoadVec4( iChannelUI, ivec2(iData,1) );
    
    window.bMinimized = (vData1.x > 0.0);    
    window.bClosed = (vData1.y > 0.0) && !desc.bOpenWindow;    
    
    // Clamp window position so title bar is always on canvas
	vec2 vSafeMin = vec2(24.0);        
	vec2 vSafeMax = vec2(32.0);        
    vec2 vPosMin = vec2( -window.rect.vSize.x + vSafeMin.x, -vSafeMin.y);//vec2( -window.rect.vSize.x, 0.0) + 24.0, -24.0 );
    vec2 vPosMax = uiContext.drawContext.vCanvasSize - vSafeMax;
    window.rect.vPos = clamp( window.rect.vPos, vPosMin, vPosMax );
    
    if ( iFrame == 0 || vData1.z != DIRTY_DATA_MAGIC)
    {
        window.rect = desc.initialRect;
        window.bMinimized = desc.bStartMinimized;
	    window.bClosed = desc.bStartClosed;
    }       
    
    window.uControlFlags = desc.uControlFlags;
    window.vMaxSize = desc.vMaxSize;
    
    window.iControlId = iControlId;
        
    return window;
}

// Function 59
float state(in float rule[directions], in vec3 s)
{
    uint d = 0u;
    
    d |= uint(bool(round(s.x))) << 2u;
    d |= uint(bool(round(s.y))) << 1u;
    d |= uint(bool(round(s.z))) << 0u;
    
    return rule[7u - d];
}

// Function 60
GameState GameState_Read( sampler2D stateSampler )
{
    GameState gameState;
    
    ivec2 vAddress = ivec2( 0 );
    
    vec4 vData0 = ReadStateData( stateSampler, vAddress );
    vAddress.x++;

    vec4 vData1 = ReadStateData( stateSampler, vAddress );
    vAddress.x++;

    vec4 vData2 = ReadStateData( stateSampler, vAddress );
    vAddress.x++;

    gameState.iMainState = int(vData0.x);
    gameState.fSkill = vData0.y;
    gameState.fGameTime = vData0.z;
    gameState.fStateTimer = vData0.w;    

    gameState.vPrevMouse = vData1;
    
    gameState.fMap = vData2.x;
    gameState.fHudFx = vData2.y;

    gameState.iMessage = int(vData2.z);
    gameState.fMessageTimer = vData2.w;
    
    return gameState;
}

// Function 61
void BodyCalculateDerivedState( inout Body body )
{
    body.mRot = QuatToMat3( body.qRot );    
}

// Function 62
float keystatepress( int key )
	{ return max( keystate( key ), keypress( key ) ); }

// Function 63
bool isState(const int state, const int value)
{
    return (state & value) > 0;
}

// Function 64
vec4 fetchState(ivec2 fragCoord) {
    return texelFetch(iChannel0, fragCoord, 0);
}

// Function 65
vec4 GetNoteState( sampler2D sampler, int track, int noteIdx, vec3 iResolution)
{
    if (noteIdx >= maxNotes)
        return vec4(0.,0.,0.,0.);
    int base = noteStateBaseAddr;
    return GetGameData(sampler, base+track*sheetTrackLen*32+noteIdx, iResolution);
    
}

// Function 66
float getNeuronState(vec2 ncoord) {
    if (min(ncoord.x, ncoord.y) < 0. ||
        max(ncoord.x-iResolution.x, ncoord.y-iResolution.y) > 0.) return 0.;
    
    return texture(iChannel1, ncoord/iResolution.xy).x;
}

// Function 67
vec4 SaveState(in AppState s, in vec2 fragCoord)
{
    vec4 ret = vec4(0.);
    StoreValue(vec2(0., 0.), vec4(s.menuId, s.metal, s.roughness, s.baseColor), ret, fragCoord);
    StoreValue(vec2(1., 0.), vec4(s.focus, s.focusObjRot, s.objRot, 0.), ret, fragCoord);
    ret = iFrame >= 1 ? ret : vec4(0.);
    return ret;
}

// Function 68
vec4 SaveState( in GameState s, in vec2 fragCoord, bool reset )
{
    vec4 ret = vec4( 0. );
    StoreValue( vec2( 0., 0. ), vec4( s.tick, PackXY( s.hp, s.level ), s.xp, s.keyNum ), ret, fragCoord );
    StoreValue( vec2( 1., 0. ), vec4( PackXY( s.playerPos ), PackXY( s.playerFrame, s.playerDir ), PackXY( s.bodyPos ), s.bodyId ), ret, fragCoord );
    StoreValue( vec2( 2., 0. ), vec4( PackXY( s.state, s.keyLock ), s.stateTime, PackXY( s.despawnPos ), s.despawnId ), ret, fragCoord );

    for ( int i = 0; i < ENEMY_NUM; ++i )
    {
        StoreValue( vec2( 3., float( i ) ), 
                   vec4( PackXY( s.enemyPos[ i ] ), 
                         PackXY( s.enemyFrame[ i ], s.enemyDir[ i ] ), 
                         PackXY( s.enemyHP[ i ], s.enemyId[ i ] ),
                         PackXY( s.enemySpawnPos[ i ] ) ), ret, fragCoord );
    }
    
    for ( int i = 0; i < LOG_NUM; ++i )
    {
        StoreValue( vec2( 4., float( i ) ), vec4( s.logPos[ i ], s.logLife[ i ], PackXY( s.logId[ i ], s.logVal[ i ] ) ), ret, fragCoord );
    }

	if ( reset )    
    {
        ret = vec4( 0. );
		StoreValue( vec2( 0., 0. ), vec4( 0., 21., 0., 0. ), ret, fragCoord );        
        StoreValue( vec2( 1., 0. ), vec4( PackXY( 3., 2. ), 0., 0., 0. ), ret, fragCoord );
        StoreValue( vec2( 2., 0. ), vec4( s.state, 0., 0., 0. ), ret, fragCoord );
    }
    
    return ret;
}

// Function 69
void GameState_Reset( out GameState gameState, vec4 vMouse )
{
    gameState.iMainState = MAIN_GAME_STATE_BOOT;
	gameState.fSkill = 0.;
    gameState.fGameTime = 0.;
    gameState.fStateTimer = 0.;
    
    gameState.vPrevMouse = vMouse;
    
    gameState.fMap = 0.0;
    gameState.fHudFx = 0.0;
    
    gameState.iMessage = -1;
    gameState.fMessageTimer = 0.0;
}

// Function 70
void gs_check_menu_item( inout GameState gs, int i, int p )
{
	if( keypress( KEY_1 + i ) == 1. )
	{
        int curr = p + i;
        int next = int( md_load( iChannel0, p + i ).w >> 8 ) & 0xff;
        gs.menustate.x = next == 0 ? 0 : curr;
        gs.menustate.y = next == 0 ? curr : 0;
        gs.menustate.z = gs.menustate.y;
    }
}

// Function 71
void gs_pace_running_state( inout GameState gs, float dt )
{
    if( abs( iMouse.x - iMouse.z ) < 6. &&
        abs( iMouse.y - iMouse.w ) < 6. && iMouse.z > 0. )
    {
        gs.dragstate = iMouse.xy;
    }

    if( keypress( KEY_M ) == 1. )
        if( ( gs.switches & GS_TRMAP ) == 0u )
      		gs_enter_map_mode( gs );
        else
            gs_leave_map_mode( gs );

	if( ( gs.switches & GS_TRMAP ) == 0u )
    	gs_pace_first_person_mode( gs, dt );
	else
      	gs_process_map_mode( gs );

	gs_process_menu( gs );
    if( gs.menustate.y == MENU_QUIT )
        gs = gs_init();
}

// Function 72
bool HandleState(vec2 aFragCoord, out vec4 oNewValue)
{
    if (IsVariable(aFragCoord, VAR_CAMERA_POS_xyz))
    {
        oNewValue = vec4(CalcCameraPos(), 0.0);
     	return true;   
    }
    else if (IsVariable(aFragCoord, VAR_CAMERA_ROT_xy))
    {
        vec2 vRot = CalcCameraRot();
		oNewValue = vec4(vRot.x, vRot.y, 0.0, 0.0);
     	return true;   
    }
    return false;
}

// Function 73
void Cam_StoreState( vec2 addr, const in CameraState cam, inout vec4 fragColor, in vec2 fragCoord )
{
    StoreVec4( addr + vec2(0,0), vec4( cam.vPos, 0 ), fragColor, fragCoord );
    StoreVec4( addr + vec2(1,0), vec4( cam.vTarget, cam.fFov ), fragColor, fragCoord );    
}

// Function 74
void StoreKeyboardState(inout vec4 fragColor, in vec2 fragCoord)
{
    vec4 previousKeyboardState = vec4(float(gKeyboardState.mKeyModeForward[0]), float(gKeyboardState.mKeyModeBackwards[0]), float(gKeyboardState.mKeyW[0]), float(gKeyboardState.mKeyS[0]));
    StoreValue(txPreviousKeyboard, previousKeyboardState, fragColor, fragCoord);
}

// Function 75
TeletextState TeletextState_Default()
{
    TeletextState state;
    
    state.char = 0x20;
    
    state.iFgCol = 7;
    state.iBgCol = 0;
    
    state.iHeldChar = 0x20;
    state.bHeldSeparated = false;
    
    state.bDoubleHeight = false;
    state.bFlash = false;    
    state.bGfx = false;
    state.bConceal = false;
    state.bSeparatedGfx = false;
    state.bHoldGfx = false;
    
    state.cmd = -1;
    
    return state;
}

// Function 76
float keystatepress( int key )
	{ return 0.; }

// Function 77
float keyState(int key, sampler2D chan) {
	return texelFetch( chan, ivec2(key,0), 0 ).x;
}

// Function 78
vec4 saveState( in AppState s, in vec2 fragCoord, int iFrame, float iTime )
{
    if (iFrame <= 0)
    {
        s.seed = fbm3( iDate.yzw );
 		s = setStateStartGame( s, iTime );
	}
    
    vec4 ret = vec4( 0.);
	storeValue( vec2( 0., 0. ), vec4( s.isPressedLeft,		s.isPressedRight,	s.stateID,			s.timeStarted),	ret, fragCoord );    
	storeValue( vec2( 1., 0. ), vec4( s.playerPos,								s.score,			s.timeFailed),	ret, fragCoord );
	storeValue( vec2( 2., 0. ), vec4( s.highscore,			s.timeCollected,	s.timeAccumulated,	s.showUI),		ret, fragCoord );
    storeValue( vec2( 3., 0. ), vec4( s.paceScale,			s.seed,				0.0,				0.0),			ret, fragCoord );
    
    storeValue( vec2( 0., 1. ), vec4( s.coin0Pos, s.coin0Taken, s.coin1Pos, s.coin1Taken ), ret, fragCoord );
    storeValue( vec2( 1., 1. ), vec4( s.coin2Pos, s.coin2Taken, s.coin3Pos, s.coin3Taken ), ret, fragCoord );
    return ret;
}

// Function 79
vec4 guistate(in int s) 
{ return texture(iChannel0, (vec2(float(s),0.)+.5)/iChannelResolution[0].xy); }

// Function 80
void WheelStoreState( ivec2 addr, const in Wheel wheel, inout vec4 fragColor, in vec2 fragCoord )
{
    vec4 vState = vec4( wheel.fSteer, wheel.fRotation, wheel.fExtension, wheel.fAngularVelocity );
    StoreVec4( addr + offsetWheelState, vState, fragColor, fragCoord );

    vec4 vState2 = vec4( wheel.vContactPos.xy, wheel.fOnGround, wheel.fSkid );
    StoreVec4( addr + offsetWheelContactState , vState2, fragColor, fragCoord );
}

// Function 81
void CameraStoreState( Camera cam, in vec2 addr, inout vec4 fragColor, in vec2 fragCoord )
{
    StoreVec3( addr + offsetCameraPos, cam.vPos, fragColor, fragCoord );
    StoreVec3( addr + offsetCameraDir, cam.vDir, fragColor, fragCoord );
    StoreVec3( addr + offsetCameraUp,  cam.vUp,  fragColor, fragCoord );
    StoreVec3( addr + offsetCameraVel, cam.vVel, fragColor, fragCoord );
}

// Function 82
void loadState() 
{
    int pos = 1;
    state.camRight = loadVec3(pos);			
    state.camUp = loadVec3(pos);			
    state.camForward = loadVec3(pos);		
    state.camPosition = loadVec3(pos);		
    state.camAngle = loadVec2(pos);		
    state.camFovy = loadFloat(pos);			
    state.lastMousePos = loadVec2(pos);		
    state.isMouseDragging = loadFloat(pos);			
    state.tilting = loadFloat(pos);
    state.rolling = loadFloat(pos);
    state.speed = loadFloat(pos);
}

// Function 83
void SaveState(inout vec4 c, State state, ivec2 p)
{
    ivec2 R = state.resolution;
    if (p.y == R.y - 1) switch (R.x - 1 - p.x) { // IsStatePixel(p)
      case slotResMBD:
        c = vec4(R, state.mbdown ? 1. : 0., 0);
        break;
      case slotEyePos:
        c = vec4(state.eyepos, 0.);
        break;
      case slotAzElBase:
        c = vec4(state.eyeaim, state.aimbase);
        break;
      default:
        break;
    }
}

// Function 84
void LoadState(out AppState s)
{
    vec4 data;

    data = LoadValue(0, 0);
    s.menuId    = data.x;
    s.roughness = data.y;
    s.focus     = data.z;
    
    data = LoadValue(1, 0);
    s.focusObjRot  	= data.xy;
    s.objRot    	= data.zw;
}

// Function 85
SceneState SetupSceneState()
{
    SceneState sceneState;
    
    sceneState.vehicleState.vPos = LoadVec3( addrVehicle + offsetVehicleBody + offsetBodyPos );
    
    sceneState.vehicleState.qRot = LoadVec4( addrVehicle + offsetVehicleBody + offsetBodyRot );
    sceneState.vehicleState.mRot = QuatToMat3( sceneState.vehicleState.qRot );
    
    vec4 vParam0;
    vParam0 = LoadVec4( addrVehicle + offsetVehicleParam0 );
    sceneState.vehicleState.fSteerAngle = vParam0.x;
    

    return sceneState;
}

// Function 86
void InitSceneState( AnimState animState, vec3 vCamPos )
{   
    vec3 vEyeTarget = animState.vEyeTarget;
    
    g_sceneState.mHeadRot = MatFromAngles( animState.vHeadAngles );    
    
    g_sceneState.vNeckOffset = vec3( 0.0, 1.0, 1.2 );
    
    float ipd = 0.3f;
    g_sceneState.lEyePos = TransformHeadPos( vec3( ipd, 0.0f, 0.0f ) );
    g_sceneState.rEyePos = TransformHeadPos( vec3( -ipd, 0.0f, 0.0f ) );
    
    g_sceneState.lEyeDir = vEyeTarget - g_sceneState.lEyePos;
    g_sceneState.rEyeDir = vEyeTarget - g_sceneState.rEyePos;
    
    ClampEyeDir( g_sceneState.lEyeDir, 1.0 );
    ClampEyeDir( g_sceneState.rEyeDir, -1.0 );
}

// Function 87
void AnimState_StoreState( ivec2 addr, const in AnimState animState, inout vec4 fragColor, in ivec2 fragCoord )
{
    StoreVec4( addr + ivec2(0,0), vec4( animState.vEyeTarget, 0 ), fragColor, fragCoord );
    StoreVec4( addr + ivec2(1,0), vec4( animState.vHeadAngles, 0 ), fragColor, fragCoord );
}

// Function 88
state readState() {
    state s = state(
        vec3(0.0),
        vec3(0.0),
        vec3(0.0, -cos(0.25*PI), sin(0.25*PI)),
        vec3(0.0, 0.5, 0.0),
        QID
    );
    if (iFrame > 0) {
        s.p = texelFetch(LAST_FRAME, ivec2(0, 0), 0).xyz;
        s.q = texelFetch(LAST_FRAME, ivec2(1, 0), 0).xyz;
        s.v = texelFetch(LAST_FRAME, ivec2(2, 0), 0).xyz;
        s.L = texelFetch(LAST_FRAME, ivec2(3, 0), 0).xyz;
        s.pr = texelFetch(LAST_FRAME, ivec2(4, 0), 0);
    }
    return s;
}

// Function 89
vec4 pz_readState(float nr) {
    return pz_stateSample(pz_nr2vec(nr)/pz_realBufferResolution);
}

// Function 90
Voxel getDefaultMenuVoxel (in uint type)
{
	Voxel voxel;
    
    voxel.type = type;
    
    if (type == VOXEL_TYPE_REDSTONE_TORCH) voxel.energy = MAX_REDSTONE_POWER;
    else voxel.energy = 0u;
    
    voxel.facing = checkFlag(type, VOXEL_TYPE_FACING) ? VOXEL_FACING_PLUS_Z
        : VOXEL_FACING_NOWHERE;
    
    return voxel;
}

// Function 91
void gs_respond_to_menu( inout GameState gs, inout VehicleState vs, inout MsgQueue msg )
{
	if( gs.menustate.y >= MENU_INFO_BEGIN && gs.menustate.y < MENU_INFO_BEGIN + MENU_INFO_SIZE )
    	gs.switches = ( gs.switches & ~GS_IPAGE_MASK ) | ( uint( gs.menustate.y - MENU_INFO_BEGIN ) << GS_IPAGE_SHIFT );
    else
	if( gs.menustate.y >= MENU_HMD_BEGIN && gs.menustate.y < MENU_HMD_BEGIN + MENU_HMD_SIZE )
       	vs.modes.x = gs.menustate.y - MENU_HMD_BEGIN;
    else
	if( gs.menustate.y >= MENU_ENG_BEGIN && gs.menustate.y < MENU_ENG_BEGIN + MENU_ENG_SIZE )
    {
        if( gs.menustate.y - MENU_ENG_BEGIN == int( VS_ENG_NOVA ) )
            msg_push_if_empty( msg, uvec4( 0x8c836e6f, 0x7420a900, 0, 7 ), vec4(0) );
        else
        	vs.modes.z = gs.menustate.y - MENU_ENG_BEGIN, vs.throttle = 0.;
    }
    else
	if( gs.menustate.y >= MENU_THR_BEGIN && gs.menustate.y < MENU_THR_BEGIN + MENU_THR_SIZE )
        vs.modes2.z = gs.menustate.y - MENU_THR_BEGIN;
    else
	if( gs.menustate.y >= MENU_AERO_BEGIN && gs.menustate.y < MENU_AERO_BEGIN + MENU_AERO_SIZE )
    {
        int newmode = gs.menustate.y - MENU_AERO_BEGIN;
		if( vs.modes2.x < VS_AERO_ATR && newmode >= VS_AERO_ATR )
            vs.trim = 0.;
        if( vs.modes2.x >= VS_AERO_ATR && newmode < VS_AERO_ATR )
        	vs.trim = vs.EAR.x, vs.aerostuff = vec4(0);
        vs.modes2.x = newmode;
    }
    else
	if( gs.menustate.y >= MENU_RCS_BEGIN && gs.menustate.y < MENU_RCS_BEGIN + MENU_RCS_SIZE )
        vs.modes2.y = gs.menustate.y - MENU_RCS_BEGIN;
    else
    if( gs.menustate.y >= MENU_MMODE_BEGIN && gs.menustate.y < MENU_MMODE_BEGIN + MENU_MMODE_SIZE )
        gs.switches = ( gs.switches & ~GS_MMODE_MASK ) | ( uint( gs.menustate.y - MENU_MMODE_BEGIN ) << GS_MMODE_SHIFT );
    else
    if( gs.menustate.y >= MENU_MPROJ_BEGIN && gs.menustate.y < MENU_MPROJ_BEGIN + MENU_MPROJ_SIZE )
        gs.switches = ( gs.switches & ~GS_MPROJ_MASK ) | ( uint( gs.menustate.y - MENU_MPROJ_BEGIN ) << GS_MPROJ_SHIFT );
	else
    if( gs.menustate.y == MENU_SET_WAYPOINT )
        gs.waypoint = gs.mapmarker, gs.mapmarker = ZERO;
}

// Function 92
bool SideMenu( ivec2 numPanels )
{
	// arrange so that the main view and the side views have the same aspect ratio
	vec2 dims = vec2(
						iResolution.x/float(numPanels.x+numPanels.y), // main view is sv.y times bigger on both axes!
						iResolution.y/float(numPanels.y)
						);


	// which one is selected?
	ivec2 viewIndex = ivec2(floor(iMouse.xy/dims));

	int selectedPanel = 0;
	if ( viewIndex.x < numPanels.x )
	{
		selectedPanel = viewIndex.y+viewIndex.x*numPanels.y;
	}
	

	// figure out which one we're drawing
	viewIndex = ivec2(floor(fragCoord.xy/dims));

	int index;
	vec4 viewport;
	if ( viewIndex.x < numPanels.x )
	{
		viewport.xy = vec2(viewIndex)*dims;
		viewport.zw = dims;
		index = viewIndex.y+viewIndex.x*numPanels.y;
	}
	else
	{
		// main view, determined by where the last click was
		viewport.x = float(numPanels.x)*dims.x;
		viewport.y = 0.0;
		viewport.zw = dims*float(numPanels.y);
		index = selectedPanel;
	}
	
	// highlight currently selected
	if ( index == selectedPanel && viewIndex.x < numPanels.x &&
		( fragCoord.x-viewport.x < 2.0 || viewport.x+viewport.z-fragCoord.x < 2.0 ||
		  fragCoord.y-viewport.y < 2.0 || viewport.y+viewport.w-fragCoord.y < 2.0 ) )
	{
		fragColor = vec4(1,1,0,1);
		return false;
	}
	
	// compute viewport-relative coordinates
	view_FragCoord = fragCoord.xy - viewport.xy;
	view_Resolution = viewport.zw;
	view_Index = index;

	view_selectionRelativeMouse = fract(iMouse/dims.xyxy);
	
	return true;
}

// Function 93
void Cam_LoadState( out CameraState cam, sampler2D sampler, ivec2 addr )
{
    vec4 vPos = LoadVec4( sampler, addr + ivec2(0,0) );
    cam.vPos = vPos.xyz;
    vec4 targetFov = LoadVec4( sampler, addr + ivec2(1,0) );
    cam.vTarget = targetFov.xyz;
    cam.fFov = targetFov.w;
    vec4 vUp = LoadVec4( sampler, addr + ivec2(2,0) );
    cam.vUp = vUp.xyz;
    
    vec4 jitterDof = LoadVec4( sampler, addr + ivec2(3,0) );
    cam.vJitter = jitterDof.xy;
    cam.fPlaneInFocus = jitterDof.z;
}

// Function 94
float expectedOutputState(in int op)
{
    vec2 p = vec2(float(op)+0.5, .5) / iChannelResolution[0].xy;
    return texture(iChannel0, p).x;
}

// Function 95
void GetStateForBallIndex(int ballIdx, out vec2 pos, out vec2 vel, int iChanRes0)
{
    vec4 posAndVel = texelFetch(iChannel0, BufferPixelPosFromBallIndex(ballIdx, iChanRes0), 0);
    pos = posAndVel.xy;
    vel = posAndVel.zw;
}

// Function 96
void Cam_LoadState( out CameraState cam, sampler2D sampler, ivec2 addr )
{
    vec4 vPos = LoadVec4( sampler, addr + ivec2(0,0) );
    cam.vPos = vPos.xyz;
    vec4 targetFov = LoadVec4( sampler, addr + ivec2(1,0) );
    cam.vTarget = targetFov.xyz;
    cam.fFov = targetFov.w;
    vec4 vUp = LoadVec4( sampler, addr + ivec2(2,0) );
    cam.vUp = vUp.xyz;
    
    vec4 jitterDof = LoadVec4( sampler, addr + ivec2(3,0) );
    cam.vJitter = jitterDof.xy;
    cam.fPlaneInFocus = jitterDof.z;
    cam.bStationary = jitterDof.w > 0.0;
}

// Function 97
float outputState(in int op)
{
    vec2 p = vec2(float(op)+.5, .5) / iChannelResolution[1].xy;
    return texture(iChannel1, p).x;
}

// Function 98
int coordToCaveStateArrIndex(ivec2 coord)
{
    return coord.y * CAV_SIZ.x + coord.x;
}

// Function 99
void BodyStoreState(const in Body body,in float af, inout vec4 fragColor, in vec2 fragCoord )
{
    vec2 a=vec2(0.,af)+_bBody;
    storeValue( a+txBodyPos, 			vec4(body.vPos,body.fMass),   	   		fragColor, fragCoord );
    storeValue( a+txBodyRot, 			body.qRot,   	   				fragColor, fragCoord );
    storeValue( a+txBodyMom, 			vec4(body.vMomentum,0.),   	   	fragColor, fragCoord );
    storeValue( a+txBodyAngMom, 		vec4(body.vAngularMomentum,0.), fragColor, fragCoord );
}

// Function 100
void WanderCam_StoreState( ivec2 addr, const WanderCamState wanderCam, inout vec4 fragColor, in ivec2 fragCoord )
{
    StoreVec4( addr + ivec2(0,0), vec4( wanderCam.pos, 0 ), fragColor, fragCoord );
    StoreVec4( addr + ivec2(1,0), vec4( wanderCam.lookAt, 0 ), fragColor, fragCoord );
    StoreVec4( addr + ivec2(2,0), vec4( wanderCam.targetAngle, wanderCam.lookAtAngle, wanderCam.eyeHeight, wanderCam.timer ), fragColor, fragCoord );
    StoreVec4( addr + ivec2(3,0), vec4( 0, wanderCam.iSitting, wanderCam.shoreDistance, wanderCam.lookAtElevation ), fragColor, fragCoord );
}

// Function 101
void menuText( inout vec3 color, vec2 p, in AppState s )
{        
    vec2 scale = vec2( 4., 8. );
    vec2 t = floor( p / scale );   
    
    uint v = 0u;
	v = t.y == 2. ? ( t.x < 4. ? 1768452929u : ( t.x < 8. ? 1768777835u : ( t.x < 12. ? 5653614u : 0u ) ) ) : v;
	v = t.y == 1. ? ( t.x < 4. ? 1918986307u : ( t.x < 8. ? 1147496812u : ( t.x < 12. ? 1752383839u : ( t.x < 16. ? 1835559785u : 5664361u ) ) ) ) : v;
	v = t.y == 0. ? ( t.x < 4. ? 1918986307u : ( t.x < 8. ? 1147496812u : ( t.x < 12. ? 86u : 0u ) ) ) : v;
	v = t.x >= 0. && t.x < 20. ? v : 0u;
    
	float c = float( ( v >> uint( 8. * t.x ) ) & 255u );
    
	vec3 textColor = vec3( 1.0 );

    p = ( p - t * scale ) / scale;
    p.x = ( p.x - .5 ) * .5 + .5;
    float sdf = textSDF( p, c );
    if ( c != 0. )
    {
    	color = mix( textColor, color, smoothstep( -.05, +.05, sdf ) );
    }
}

// Function 102
void SceneObjectStoreState( SceneObject so, int objIndex, inout vec4 fragColor, in vec2 fragCoord )
{
    vec2 addr = addrSceneObjs + float(objIndex)*sizeSceneObj;
    
    StoreVec3( addr + offsetSOPos, so.vPos, fragColor, fragCoord );
    //StoreMat3( addr + offsetSOTransform, so.vTransform, fragColor, fragCoord );
    StoreVec3( addr + offsetSOAttr, vec3(so.vState, so.vMaterial, so.vDist), fragColor, fragCoord );
}

// Function 103
vec4 update_state(vec4 tc, vec4 tcc2) {
    if (int(tc.w) != c_bgr) {
        if (int(tcc2.w) != c_bgr) {
            vec3 crvals = decodeval(tc.x);
            vec3 crvalsx = decodeval(tcc2.x);
            if (int(tc.y) != 1) {
                crvals.y -= crvalsx.z;
                if (crvals.y <= 0.) {
                    tc.y = -g_time;
                    crvals.y = 0.;
                }
            }
            float rexf = encodeval(crvals);
            tc.x = rexf;
        }
        if (int(tc.y) == 1) tc.y = 0.;
    }
    return tc;
}

// Function 104
void SaveState(inout vec4 c, State state, ivec2 p)
{
    ivec2 R = state.resolution;
    if (p.y == R.y - 1) switch (R.x - 1 - p.x) { // IsStatePixel(p)
      case slotResMBD:
        c = vec4(R, state.mbdown ? 1. : 0., 0);
        break;
      case slotEyePos:
        c = vec4(state.eyepos, 0.);
        break;
      case slotEyeVel:
        c = vec4(state.eyevel, 0.);
        break;
      case slotAzElBase:
        c = vec4(state.eyeaim, state.aimbase);
        break;
      default:
        break;
    }
}

// Function 105
void BodyLoadState(inout Body body,in float af )
{
    vec2 a=vec2(0.,af)+_bBody;
    body.id=int(af);
    {
        vec4 bvp = 			loadif(txBodyPos+a	,vec4(body.vPos,0.));
        body.vPos=bvp.xyz;
        body.fMass=bvp.w;
    }
    body.qRot = 			loadif(txBodyRot+a	,body.qRot);
    body.vMomentum =  		loadif(txBodyMom+a	,vec4(body.vMomentum,0.)).xyz;
    body.vAngularMomentum = loadif(txBodyAngMom+a,vec4(body.vAngularMomentum,0.)).xyz;
    body.vCol = 			loadifb(txBodyCol+a,body.vCol);
    body.vColMat = 			loadifb(txBodyColMat+a,body.vCol);

}

// Function 106
void loadGameStateFull(float time, sampler2D storage) {
    loadGameStateMinimal(time, storage);
        
    for (int i=0; i<MAX_BULLETS; i++) {
    	gBulletData[i] = texelFetch(storage, ivec2(i+BULLET_DATA_OFFSET,0), 0).xyz;
    }
    
    for (int i=0; i<MAX_AIRCRAFTS/2; i++) {
        vec4 f = texelFetch(storage, ivec2(i+AIRCRAFT_DATA_OFFSET,0), 0);
        gAircraftData[i*2+0] = f.xy;
        gAircraftData[i*2+1] = f.zw;
    } 
    
    for (int i=0; i<MAX_PARATROOPERS; i++) {
    	gParatrooperData[i] = texelFetch(storage, ivec2(i+PARATROOPER_DATA_OFFSET,0), 0);
    }
}

// Function 107
float transition(float x, float eps)
{
    //return min(floor(x), 1.); // or (1.-step(1., x));
    return smoothstep(1.-eps, 1.+eps, x); // modified rule
}

// Function 108
void SceneObjectLoadState( out SceneObject so, int objIndex )
{
    vec2 addr = addrSceneObjs + float(objIndex)*sizeSceneObj;
    
	so.vPos = LoadVec3( addr + offsetSOPos );
    	
    vec3 attr = LoadVec3 ( addr + offsetSOAttr );
    
    so.vState = attr.x;
    so.vMaterial = attr.y;
    so.vDist = attr.z;
}

// Function 109
void writeScoreStates(vec3 ss, inout vec4 fragColor, vec2 fragCoord)
{
    if(isPixel(2,0,fragCoord)) fragColor.xyz=ss;
}

// Function 110
void AstState ()
{
  float s, r, a;
  hsh = Hashv3v3 (cId);
  s = fract (64. * length (hsh));
  s *= s;
  r = 0.5 * bGrid.x * (0.8 + 0.2 * hsh.x * (1. - s) * abs (sin (3. * pi * hsh.y * (1. - s))));
  a = hsh.z * tCur + hsh.x;
  ast = vec4 ((r - 1.1 * (0.15 - 0.07 * s)) * vec3 (cos (a), sin (a), 0.), 0.15 - 0.07 * s);
}

// Function 111
void draw_menu(inout vec4 fragColor, vec2 fragCoord, Timing timing)
{
    MenuState menu;
    LOAD(menu);

    if (menu.open <= 0)
        return;

    vec4 options = load(ADDR_OPTIONS);

    if (!test_flag(int(options[get_option_field(OPTION_DEF_SHOW_LIGHTMAP)]), OPTION_FLAG_SHOW_LIGHTMAP))
    {
        // vanilla
        fragColor.rgb *= vec3(.57, .47, .23);
        fragColor.rgb = ceil(fragColor.rgb * 24. + .01) / 24.;
    }
    else
    {
        // GLQuake
       	fragColor.rgb *= .2;
    }

    //g_text_scale_shift = 1;
    int text_scale = 1 << g_text_scale_shift;
    float image_scale = float(text_scale);
    vec2 header_size = ADDR2_RANGE_TEX_OPTIONS.zw * image_scale;
    vec2 left_image_size = ADDR2_RANGE_TEX_QUAKE.wz * image_scale;
    float left_image_offset = 120. * image_scale;

    vec2 ref = iResolution.xy * vec2(.5, 1.);
    ref.y -= min(float(CHAR_SIZE.y) * 4. * image_scale, iResolution.y / 16.);

    ref.x += left_image_size.x * .5;
    if (fragCoord.x < ref.x - left_image_offset)
    {
        fragCoord.y -= ref.y - left_image_size.y;
        fragCoord.x -= ref.x - left_image_offset - left_image_size.x;
        ivec2 addr = ivec2(floor(fragCoord)) >> g_text_scale_shift;
        if (uint(addr.x) < uint(ADDR2_RANGE_TEX_QUAKE.w) && uint(addr.y) < uint(ADDR2_RANGE_TEX_QUAKE.z))
	        fragColor.rgb = texelFetch(LIGHTMAP_CHANNEL, addr.yx + ivec2(ADDR2_RANGE_TEX_QUAKE.xy), 0).rgb;
        return;
    }

    ref.y -= header_size.y;
    if (fragCoord.y >= ref.y)
    {
        fragCoord.y -= ref.y;
        fragCoord.x -= ref.x - header_size.x * .5;
        ivec2 addr = ivec2(floor(fragCoord)) >> g_text_scale_shift;
        if (uint(addr.x) < uint(ADDR2_RANGE_TEX_OPTIONS.z) && uint(addr.y) < uint(ADDR2_RANGE_TEX_OPTIONS.w))
	        fragColor.rgb = texelFetch(LIGHTMAP_CHANNEL, addr + ivec2(ADDR2_RANGE_TEX_OPTIONS.xy), 0).rgb;
        return;
    }

    ref.y -= float(CHAR_SIZE.y) * 1. * image_scale;

    const int
        BASE_OFFSET		= CHAR_SIZE.x * 0,
        ARROW_OFFSET	= CHAR_SIZE.x,
        VALUE_OFFSET	= CHAR_SIZE.x * 3,
        MARGIN			= 0,
        LINE_HEIGHT		= MARGIN + CHAR_SIZE.y;

    ivec2 uv = text_uv(fragCoord - ref);
    uv.x -= BASE_OFFSET;
    int line = -uv.y / LINE_HEIGHT;
    if (uint(line) >= uint(NUM_OPTIONS))
        return;
    
    uv.y = uv.y + (line + 1) * LINE_HEIGHT;
    if (uint(uv.y - MARGIN) >= uint(CHAR_SIZE.y))
        return;
    uv.y -= MARGIN;
    
    int glyph = 0;
    if (uv.x < 0)
    {
        int begin = OPTIONS.data[1 + line];
        int end = OPTIONS.data[2 + line];
        int num_chars = end - begin;
        uv.x += num_chars * CHAR_SIZE.x;
    	glyph = glyph_index(uv.x);
        if (uint(glyph) >= uint(num_chars))
            return;
        glyph += begin;
        glyph = get_byte(glyph & 3, OPTIONS.data[OPTIONS.data[0] + 2 + (glyph>>2)]);
    }
    else if (uint(uv.x - ARROW_OFFSET) < uint(CHAR_SIZE.x))
    {
        const float BLINK_SPEED = 2.;
        uv.x -= ARROW_OFFSET;
        if (menu.selected == line && (fract(iTime * BLINK_SPEED) < .5 || test_flag(timing.flags, TIMING_FLAG_PAUSED)))
            glyph = _RIGHT_ARROW_;
    }
    else if (uv.x >= VALUE_OFFSET)
    {
        uv.x -= VALUE_OFFSET;

        int item_height = CHAR_SIZE.y << g_text_scale_shift;

        MenuOption option = get_option(line);
        int option_type = get_option_type(option);
        int option_field = get_option_field(option);
        if (option_type == OPTION_TYPE_SLIDER)
        {
            const float RAIL_HEIGHT = 7.;
            vec2 p = vec2(uv.x, uv.y & 7) + .5;
            vec2 line = lit_line(p, vec2(8, 4), vec2(8 + 11*CHAR_SIZE.x, 4), RAIL_HEIGHT);
            float alpha = linear_step(-.5, .5, -line.y);
            line.y /= RAIL_HEIGHT;
            float intensity = 1. + line.x * step(-.25, line.y);
            intensity = mix(intensity, 1. - line.x * .5, line.y < -.375);
            fragColor.rgb = mix(fragColor.rgb, vec3(.25, .23, .19) * intensity, alpha);

            float value = options[option_field] * .1;
            float thumb_pos = 8. + value * float(CHAR_SIZE.x * 10);
            p.x -= thumb_pos;
            p -= vec2(4);
            float r = length(p);
            alpha = linear_step(.5, -.5, r - 4.);
            intensity = normalize(p).y * .25 + .75;
            p *= vec2(3., 1.5);
            r = length(p);
            intensity += linear_step(.5, -.5, r - 4.) * (safe_normalize(p).y * .125 + .875);

            fragColor.rgb = mix(fragColor.rgb, vec3(.36, .25, .16) * intensity, alpha);
            return;
        }
        else if (option_type == OPTION_TYPE_TOGGLE)
        {
            glyph = glyph_index(uv.x);
            if (uint(glyph) >= 4u)
                return;
    		const int
                OFF = (_O_<<8) | (_F_<<16) | (_F_<<24),
    			ON  = (_O_<<8) | (_N_<<16);
            int value = test_flag(int(options[option_field]), get_option_range(option)) ? ON : OFF;
            glyph = get_byte(glyph & 3, value);
        }
    }
    else
    {
        return;
    }
    
    vec4 color = vec4(.66, .36, .25, 1);
    print_glyph(fragColor, uv, glyph, color);
}

// Function 112
void DrawMenuControls(inout vec3 color, vec2 p, in AppState s)
{
	p -= vec2(-110, 74);

	// radial
	float c2 = Capsule(p - vec2(0., -3.5), 3., 4.);
	float c1 = Circle(p + vec2(0., 7. - 7. * s.metal), 2.5);

	// roughness slider
	p.y += 15.;
	c1 = min(c1, Capsule(p.yx - vec2(0., 20.), 1., 20.));
	c1 = min(c1, Circle(p - vec2(40. * s.roughness, 0.), 2.5));

	p.y += 8.;
	c1 = min(c1, Rectangle(p - vec2(19.5, 0.), vec2(21.4, 4.)));
	color = mix(color, vec3(0.9), Smooth(-c2 * 2.));
	color = mix(color, vec3(0.3), Smooth(-c1 * 2.));

	for (int i = 0; i < 6; ++i)
	{
		vec2 o = vec2(i == int(s.baseColor) ? 2.5 : 3.5);
		color = mix(color, BASE_COLORS[i], Smooth(-2. * Rectangle(p - vec2(2. + float(i) * 7., 0.), o)));
	}
}

// Function 113
vec2 transitionMusic(float introTime){
    
    vec2 mix = vec2(0.0);
    
    const float numEchos = 3.0;
    for(float i=0.0; i<numEchos; i++){

        float echoTime = introTime - i*0.03;
        float echoVol = 1.0-(i/numEchos);

        const float hornOffset = 0.0;

        mix += (
            (introHorn(-4,echoTime)*1.0*pan(-0.6))+
            (introHorn( 3,echoTime)*1.0*pan(-0.2))+
            (introHorn( 5,echoTime)*1.0*pan( 0.2))+
            (introHorn(10,echoTime)*1.0*pan( 0.6))+
            (introHorn(14,echoTime)*0.5*pan( 0.0))
        )*0.5*echoVol;
    }


    float volEnv = cos(pow(introTime*0.5,0.8)*6.2831)*0.5+0.5;
    const float volEnvMin = 0.3;
    volEnv = volEnv*(1.0-volEnvMin)+volEnvMin;


    float fadeIn = max(0.0,min(1.0,introTime/0.3));
    fadeIn = 1.0-pow(1.0-fadeIn,2.0);

    const float fadeOutStart = (transitionEnd-transitionStart) - 0.45;
    const float fadeOutEnd = (transitionEnd-transitionStart) - 0.1;
    float fadeOut = max(0.0,min(1.0,(introTime-fadeOutStart)/(fadeOutEnd-fadeOutStart)));
    fadeOut = 1.0-pow(fadeOut,4.0);

    return mix*volEnv*fadeOut*fadeIn;
    
}

// Function 114
float getNeuronLastState(vec2 ncoord) {
    if (min(ncoord.x, ncoord.y) < 0. ||
        max(ncoord.x-iResolution.x, ncoord.y-iResolution.y) > 0.) return 0.;
    
    return texture(iChannel2, ncoord/iResolution.xy).y;
}

// Function 115
vec2 pz_initializeState(vec2 fragCoord) {
    pz_initializeState();
    
    vec3 position = pz_position();
    fragCoord -= 0.5*iResolution.xy;
    fragCoord *= position.z;
    fragCoord += (0.5 + position.xy) * iResolution.xy ;
    return fragCoord;
}

// Function 116
void handleMMenu( inout vec4 buff, in vec2 fc, in vec2 keys, in vec4 dirs )
{
    // Load current menu selection.
    float curMenu = readTexel(DATA_BUFFER,txMSEL).r;
    float curState = readTexel(DATA_BUFFER,txSTATE).r;
    
    // If the UP or DOWN keys are pressed, we cycle the menu
    // in that direction.
    curMenu -= step(.5,dirs.z);
    curMenu += step(.5,dirs.w);
    curMenu = floor(mod(curMenu,2.0));
    
    // Save the current menu selection in case we have to go another
    // frame here.
    write1(buff.r, curMenu, txMSEL, fc);
    
    // Now we do selection testing.
    float newState = curState;
    
    // curMenu is 0 or 1, so this is works for binary selection.
    newState = mix(ST_INITG,ST_HOWTO,curMenu);
    
    // Write either the existing value or current value.
    // Notice how I assume the buffer holds the previous value? This can be
    // done because the only time writeX() writes to the buffer passed in
    // is when the current fragment coordinate aligns with the position
    // that we want to write to. Also note that the buffer value that is passed
    // in is the texel from the data buffer at the current fragCoord.
    // So in the only actionable situation, buff does hold the prior value.
    write1(buff.r, mix( buff.r,iTime, step(.5,keys.x) ), txPCHANGE, fc);
    
    // Write either our new state or the current state.
    write1(buff.r, mix(curState, newState, step(.5,keys.x) ), txSTATE,fc);
}

// Function 117
void LoadState(out AppState s)
{
	vec4 data;

	data = LoadValue(0, 0);
	s.menuId = data.x;
	s.metal = data.y;
	s.roughness = data.z;
	s.baseColor = data.w;

	data = LoadValue(1, 0);
	s.focus = data.x;
	s.focusObjRot = data.y;
	s.objRot = data.z;
}

// Function 118
void SceneObjectLoadState( out SceneObject so, int objIndex )
{
    vec2 addr = addrSceneObjs + float(objIndex)*sizeSceneObj;
    
	so.vPos = LoadVec3( addr + offsetSOPos );
    
    //LoadMat3(addr + offsetSOTransform, so.vTransform);
	
    vec3 attr = LoadVec3 ( addr + offsetSOAttr );
    
    so.vState = attr.x;
    so.vMaterial = attr.y;
    so.vDist = attr.z;
}

// Function 119
float transitionValues(float a, float b, float c) {
    #ifdef LOOP
        #if LOOP == 1
            return a;
        #endif
        #if LOOP == 2
            return b;
        #endif
        #if LOOP == 3
            return c;
        #endif
    #endif
    float t = time / SCENE_DURATION;
    float scene = floor(mod(t, 3.));
    float blend = fract(t);
    float delay = (SCENE_DURATION - CROSSFADE_DURATION) / SCENE_DURATION;
    blend = max(blend - delay, 0.) / (1. - delay);
    blend = sineInOut(blend);
    float ab = mix(a, b, blend);
    float bc = mix(b, c, blend);
    float cd = mix(c, a, blend);
    float result = mix(ab, bc, min(scene, 1.));
    result = mix(result, cd, max(scene - 1., 0.));
    return result;
}

// Function 120
void VehicleStoreState( ivec2 addr, const in Vechicle vehicle, inout vec4 fragColor, in vec2 fragCoord )
{
    BodyStoreState( addr + offsetVehicleBody, vehicle.body, fragColor, fragCoord );
    WheelStoreState( addr + offsetVehicleWheel0, vehicle.wheel[0], fragColor, fragCoord );
    WheelStoreState( addr + offsetVehicleWheel1, vehicle.wheel[1], fragColor, fragCoord );
    WheelStoreState( addr + offsetVehicleWheel2, vehicle.wheel[2], fragColor, fragCoord );
    WheelStoreState( addr + offsetVehicleWheel3, vehicle.wheel[3], fragColor, fragCoord );

    vec4 vParam0 = vec4( vehicle.fSteerAngle, 0.0, 0.0, 0.0 );
    StoreVec4( addr + offsetVehicleParam0, vParam0, fragColor, fragCoord);
}

// Function 121
void LoadState( out GameState s )
{
    vec4 data;

    data = LoadValue( 0, 0 );
    s.tick 		= data.x;
    s.hp    	= UnpackX( data.y );
    s.level    	= UnpackY( data.y );
    s.xp        = data.z;
    s.keyNum    = data.w;
    
    data = LoadValue( 1, 0 );
    s.playerPos   = UnpackXY( data.x );
    s.playerFrame = UnpackX( data.y );
    s.playerDir   = UnpackY( data.y );
    s.bodyPos	  = UnpackXY( data.z );
    s.bodyId      = data.w;
    
    data = LoadValue( 2, 0 );
    s.state      = UnpackX( data.x );
    s.keyLock    = UnpackY( data.x );
    s.stateTime  = data.y;
    s.despawnPos = UnpackXY( data.z );
    s.despawnId  = data.w;

    for ( int i = 0; i < ENEMY_NUM; ++i )
    {
        data = LoadValue( 3, i );
        s.enemyPos[ i ]      = UnpackXY( data.x );
        s.enemyFrame[ i ]    = UnpackX( data.y );
        s.enemyDir[ i ]      = UnpackY( data.y );
        s.enemyHP[ i ]       = UnpackX( data.z );
        s.enemyId[ i ]       = UnpackY( data.z );
        s.enemySpawnPos[ i ] = UnpackXY( data.w );
    }
    
    for ( int i = 0; i < LOG_NUM; ++i )
    {
		data = LoadValue( 4, i );
    	s.logPos[ i ]  = data.xy;
        s.logLife[ i ] = data.z;
        s.logId[ i ]   = UnpackX( data.w );
        s.logVal[ i ]  = UnpackY( data.w );
    }    
}

// Function 122
float cellState(in int outCell)
{
    float sum = 0.;
	for (int j=0; j<16; ++j)    
    for (int i=0; i<16; ++i)    
    {
        sum += weight(j*16+i, outCell) * inputState(ivec2(i,j));
    }
    return activation(sum);
}

// Function 123
vec4 fetchSimState(vec2 fragCoord, ivec2 offset) {
    return texelFetch(iChannel0, ivec2(fragCoord) + offset, 0);
}

// Function 124
void BodyStoreState( ivec2 addr, const in Body body, inout vec4 fragColor, in vec2 fragCoord )
{
    StoreVec3( addr + offsetBodyPos, body.vPos, fragColor, fragCoord );
    StoreVec4( addr + offsetBodyRot, body.qRot, fragColor, fragCoord );
    StoreVec3( addr + offsetBodyMom, body.vMomentum, fragColor, fragCoord );
    StoreVec3( addr + offsetBodyAngMom, body.vAngularMomentum, fragColor, fragCoord );
}

// Function 125
void GameLoadState()
{
    vec3 state = LoadVec3(addrGameState);
    
 	gtime = state.x;
    time_scale = state.y;
}

// Function 126
void FlyCam_LoadState( out FlyCamState flyCam, sampler2D sampler, ivec2 addr )
{
    vec4 vPos = LoadVec4( sampler, addr + ivec2(0,0) );
    flyCam.vPos = vPos.xyz;
    vec4 vAngles = LoadVec4( sampler, addr + ivec2(1,0) );
    flyCam.vAngles = vAngles.xyz;
    vec4 vPrevMouse = LoadVec4( sampler, addr + ivec2(2,0) );    
    flyCam.vPrevMouse = vPrevMouse;
}

// Function 127
void SaveState(in vec2 currentLoc, out vec4 write)
{
    SaveValue(currentLoc, vec2(0.0), vec4(gameState.playerPos, gameState.movementSpeed), write);
}

// Function 128
vec4 getState(in vec2 p){
    return texture(iChannel0,p/iResolution.xy);
}

// Function 129
void AstState ()
{
  float s, r;
  cHash = Hashv3v3 (cId);
  if (cHash.x > astOcc) {
    s = fract (64. * length (cHash));
    s *= s;
    r = 0.5 * bGrid.x * min (0.8 + 0.3 * cHash.x * (1. - s) * abs (sin (3. * pi * cHash.y * (1. - s))), 1.);
    astV = vec4 ((r - 1.1 * (0.15 - 0.07 * s)) * vec3 (sin ((cHash.z * tCur + cHash.x) +
       vec2 (0.5 * pi, 0.)), 0.), 0.15 - 0.07 * s);
    astCs = vec4 (sin (2. * pi * (cHash.x + 0.09 * cHash.z * tCur) + vec2 (0.5 * pi, 0.)),
       sin (2. * pi * (cHash.y + 0.11 * cHash.z * tCur) + vec2 (0.5 * pi, 0.)));
  }
}

// Function 130
bool save_state(out vec4 fragColor, in vec2 fragCoord) {
    ivec2 ipx = ivec2(fragCoord - 0.5);
    if (max(ipx.x + 5, ipx.y) > 10)return false;
    load_state(fragCoord, true);
    float cards_player = allData.cards_player;
    if (ipx == ivec2(2, 0)) {
        if ((allData.flag0 != 1.) || (g_time < extime + 0.1)) {
            cards_player = 0.;
            fragColor = vec4(cards_player, 1., g_time, allData.this_selected_card);
            return true;
        }
        float stx = allData.egt + 01.;
        if ((iMouse.z > 0.)&&(1. == SS(stx + 02.5, stx + 2. + 02.5, g_time))&&(allData.flag3 > 0.)) {
            cards_player = 0.;
            fragColor = vec4(cards_player, 0., g_time, allData.this_selected_card);
            return true;
        }
        if (allData.flag1 == 1.)cards_player += -1.;
        else {
            float anim_t2 = 1. - get_animstate(clamp((g_time - allData.card_put_anim - 0.5)*2., 0., 1.));
            if ((allData.card_draw > 0.)&&(anim_t2 == 0.)) {
                float anim_t = get_animstate(clamp(1. - (g_time - allData.card_add_anim), 0., 1.));
                if (anim_t == 0.) {
                    cards_player += 1.;
                    if (cards_player > 10.)cards_player = 10.;
                }
            }
        }
        card_get_select(allData.mouse_pos);
        if (cards_player > allData.cards_player)
            fragColor = vec4(cards_player, 1., g_time, allData.this_selected_card);
        else
            if (cards_player < allData.cards_player)
            fragColor = vec4(cards_player, 1., allData.card_add_anim, allData.this_selected_card);
        else
            fragColor = vec4(allData.cards_player, 1., allData.card_add_anim, allData.this_selected_card);
        return true;
    }
    if (ipx == ivec2(2, 1)) {
        float anim_t = get_animstate(clamp(1. - (g_time - allData.card_add_anim), 0., 1.));
        float anim_t2 = 1. - get_animstate(clamp((g_time - allData.card_put_anim - 0.5)*2., 0., 1.));
        float cdr = allData.card_draw;
        if ((allData.flag0 != 1.) || (g_time < extime + 0.1)) {
            cdr = 8.; //draw X cards for players on start
        } else
            if ((anim_t == 0.)&&(anim_t2 == 0.)) {
            cdr += -1.;
            if (cdr < 0.)cdr = 0.;
        }
        if ((cdr == 0.)&&(allData.flag1 == 1.)) {
            vec4 tcx = load_eff_buf();
            if (allData.card_bID_put_anim == 6.)
                if ((int(tcx.w) == c_mn)) {
                    cdr = 2.;
                }
        }
        if ((allData.player_turn)&&(allData.player_etf)) {
            float anim_t2zc = 1. - get_animstate(clamp((g_time - allData.ett - 2.), 0., 1.));
            if (anim_t2zc == 0.) {
                cdr = 1.;
            }
        }
        //reset mouse on new card
        if ((cdr > 0.) || (allData.flag3 == 1.) || (allData.player_etf))
            fragColor = vec4(vec3(0.), cdr);
        else
            fragColor = vec4(allData.mouse_pos, allData.card_select_anim, cdr);
        return true;
    }
    if (ipx == ivec2(2, 2)) {
        if ((allData.flag0 != 1.) || (g_time < extime + 0.1) || (allData.player_etf)) {
            fragColor = vec4(-10.);
            return true;
        }
        if (allData.last_selected_card >= 0.) {
            float anim_t = get_animstate(clamp((g_time - allData.card_select_anim)*2., 0., 1.));
            if (anim_t >= 1.) {
                if (hpmp_get_hit(allData.mouse_pos) > 0) {
                    vec4 tc2 = load_card(int(allData.last_selected_card));
                    if (((hpmp_get_hit(allData.mouse_pos) == 6)&&((int(tc2.w) == c_mn) || (int(tc2.w) == c_he2))) || ((hpmp_get_hit(allData.mouse_pos) == 16)&&(int(tc2.w) == c_at2))) {
                        fragColor = vec4(g_time, allData.last_selected_card, float(hpmp_get_hit(allData.mouse_pos)), 1.);
                    } else {
                        fragColor = vec4(allData.card_put_anim, allData.card_hID_put_anim, allData.card_bID_put_anim, 0.);
                    }
                    return true;
                }
                if (card_get_hit(allData.mouse_pos) >= 0) {
                    int cval = (card_get_hit(allData.mouse_pos) > 9 ? card_get_hit(allData.mouse_pos) - 10 : card_get_hit(allData.mouse_pos));
                    vec4 tc = (card_get_hit(allData.mouse_pos) < 10) ? load_board(cval) : load_board2(cval);
                    vec4 tc2 = load_card(int(allData.last_selected_card));
                    if ((int(tc.w) == c_bgr)&&(is_c_cr(int(tc2.w))&&(card_get_hit(allData.mouse_pos) < 10)))
                        fragColor = vec4(g_time, allData.last_selected_card, float(cval), 1.);
                    else
                        if ((int(tc.w) != c_bgr)&&(is_c_cr(int(tc.w)))&&((int(tc2.w) == c_he1) || (int(tc2.w) == c_at1) || (int(tc2.w) == c_pat) || (int(tc2.w) == c_de)))
                        fragColor = vec4(g_time, allData.last_selected_card, float(card_get_hit(allData.mouse_pos)), 1.);
                    else
                        fragColor = vec4(allData.card_put_anim, allData.card_hID_put_anim, allData.card_bID_put_anim, 0.);
                    return true;
                }
            }
        }
        fragColor = vec4(allData.card_put_anim, allData.card_hID_put_anim, allData.card_bID_put_anim, 0.);
        return true;
    }

    if (ipx == ivec2(2, 3)) {
        effect_buf_logic(fragColor, fragCoord, ipx);
        return true;
    }

    if (ipx == ivec2(2, 4)) {
        hpmp_logic(fragColor, fragCoord, ipx);
        return true;
    }

    for (int i = min(0,iFrame); i < 2; i++)
        for (int j = min(0,iFrame); j < 10; j++) {
            if (ipx == ivec2(i, j)) {
                card_logic(fragColor, fragCoord, ipx);
                return true;
            }
        }
    for (int i = 3+min(0,iFrame); i < 5; i++)
        for (int j = min(0,iFrame); j < 6; j++) {
            if (ipx == ivec2(i, j)) {
                board_logic(fragColor, fragCoord, ipx);
                return true;
            }
        }
    fragColor = vec4(1.);

    return true;
}

// Function 131
void loadState( sampler2D tex, out AppState s )
{
    vec4 data;

	data = loadValue( tex, 0, 0 );
    s.isPressedLeft		= data.x;
    s.isPressedRight	= data.y;
    s.stateID      		= data.z;
	s.timeStarted 		= data.w;    
    
    data = loadValue( tex, 1, 0 );
    s.playerPos			= data.xy;
    s.score				= data.z;
    s.timeFailed 		= data.w;
    
    data = loadValue( tex, 2, 0 );
    s.highscore 		= data.x;
    s.timeCollected		= data.y;
    s.timeAccumulated	= data.z;
    s.showUI			= data.w;
    
    data = loadValue( tex, 3, 0 );
    s.paceScale			= data.x;
    s.seed				= data.y;
   
    data = loadValue( tex, 0, 1 );
    s.coin0Pos = data.x;
    s.coin0Taken = data.y;
    s.coin1Pos = data.z;
    s.coin1Taken = data.w;
    data = loadValue( tex, 1, 1 );
    s.coin2Pos = data.x;
    s.coin2Taken = data.y;
    s.coin3Pos = data.z;
    s.coin3Taken = data.w;
}

// Function 132
void LoadState()
{
    vec4 state1 = LoadValue(kTexState1);
    vec4 state2 = LoadValue(kTexState2);
    vec4 state3 = LoadValue(kTexState3);
    vec4 state4 = LoadValue(kTexState4);
    vec4 state5 = LoadValue(kTexState5);
    
    gGameState            = state1.x;
    gGameStateTime        = state1.y;
    gGameSeed             = state1.z;
    gGameInit             = state1.w;
    gPlayerCoords         = state2.xy; 
    gPlayerNextCoords     = state2.zw;
    gPlayerMotionTimer    = state3.x;
    gPlayerRotation       = state3.y;
    gPlayerNextRotation   = state3.z;
    gPlayerScale          = state3.w;
    gPlayerVisualCoords   = state4.xyz;
    gPlayerVisualRotation = state4.w;
    gPlayerDeathCause     = state5.x;
    gPlayerDeathTime      = state5.y;
    gScore                = state5.z;
    gFbScale              = state5.w;
}

// Function 133
void Enemy_SetRandomHostileState( inout Entity entity, bool notFire )
{
    float fRandom = Hash( float(entity.iId) + iTime );
    
    if ( !notFire && (fRandom < 0.2) )
    {        
        if ( notFire )
        {
        	Enemy_SetState( entity, ENEMY_STATE_STAND );                    
        }
        else
        {
	        Enemy_SetState( entity, ENEMY_STATE_FIRE );        
        }
	}
    else
    if ( fRandom < 0.7 )
    {
        Entity targetEnt = Entity_Read( STATE_CHANNEL, int(entity.fTarget) );
        vec3 vToTarget = targetEnt.vPos - entity.vPos;
        
        if ( length( vToTarget ) < 100.0 )
        {
        	Enemy_SetState( entity, ENEMY_STATE_WALK_RANDOM );
        }
        else
        {
        	Enemy_SetState( entity, ENEMY_STATE_WALK_TO_TARGET );
		}
    }
    else
    {        
        Enemy_SetState( entity, ENEMY_STATE_STAND );        
	}
}

// Function 134
float setStateF(ivec2 index, ivec2 where, float oldState, float newState)
{
    return all(equal(index, where)) ? newState : oldState;
}

// Function 135
void FlyCam_StoreState( ivec2 addr, const in FlyCamState flyCam, inout vec4 fragColor, in ivec2 fragCoord )
{
    StoreVec4( addr + ivec2(0,0), vec4( flyCam.vPos, 0 ), fragColor, fragCoord );
    StoreVec4( addr + ivec2(1,0), vec4( flyCam.vAngles, 0 ), fragColor, fragCoord );
    StoreVec4( addr + ivec2(2,0), vec4( iMouse ), fragColor, fragCoord );
}

// Function 136
void CameraLoadState( out Camera cam, in ivec2 addr )
{
	cam.vPos = LoadVec3( addr + offsetCameraPos );
	cam.vTarget = LoadVec3( addr + offsetCameraTarget );
}

// Function 137
void VehicleLoadState( out Vechicle vehicle, ivec2 addr )
{    
    BodyLoadState( vehicle.body, addr + offsetVehicleBody );
    WheelLoadState( vehicle.wheel[0], addr + offsetVehicleWheel0 );
    WheelLoadState( vehicle.wheel[1], addr + offsetVehicleWheel1 );
    WheelLoadState( vehicle.wheel[2], addr + offsetVehicleWheel2 );
    WheelLoadState( vehicle.wheel[3], addr + offsetVehicleWheel3 );
    
    vec4 vParam0;
    vParam0 = LoadVec4( addr + offsetVehicleParam0 );
    vehicle.fSteerAngle = vParam0.x;
}

// Function 138
void Cam_StoreState( ivec2 addr, const in CameraState cam, inout vec4 fragColor, in ivec2 fragCoord )
{
    StoreVec4( addr + ivec2(0,0), vec4( cam.vPos, 0 ), fragColor, fragCoord );
    StoreVec4( addr + ivec2(1,0), vec4( cam.vTarget, cam.fFov ), fragColor, fragCoord );    
    StoreVec4( addr + ivec2(2,0), vec4( cam.vUp, 0 ), fragColor, fragCoord );    
    StoreVec4( addr + ivec2(3,0), vec4( cam.vJitter, cam.fPlaneInFocus, 0 ), fragColor, fragCoord );    
}

// Function 139
void AnimState_LoadState( out AnimState animState, sampler2D sampler, ivec2 addr )
{
    vec4 vData0 = LoadVec4( sampler, addr + ivec2(0,0) );
    animState.vEyeTarget = vData0.xyz;
    vec4 vData1 = LoadVec4( sampler, addr + ivec2(1,0) );
    animState.vHeadAngles = vData1.xyz;
}

// Function 140
float evalDrawState(inout vec4 drawState)
{	

    float t=ftime-drawState.y;
    bool changeState= t>=60.;    
    drawState.x=changeState?0.:drawState.x;
    return clamp(t,0.,60.);
}

// Function 141
void UI_StoreWindowState( inout UIContext uiContext, UIWindowState window, int iData )
{    
    vec4 vData0;
    vData0.xy = window.rect.vPos;
    vData0.zw = window.rect.vSize;
    
    StoreVec4( ivec2(iData,0), vData0, uiContext.vOutData, ivec2(uiContext.vFragCoord) );        

    vec4 vData1;
    
    vData1.x = window.bMinimized ? 1.0f : 0.0f;
    vData1.y = window.bClosed ? 1.0f : 0.0f;
    vData1.z = DIRTY_DATA_MAGIC;
    vData1.w = 0.0f;

    StoreVec4( ivec2(iData,1), vData1, uiContext.vOutData, ivec2(uiContext.vFragCoord) );        
}

// Function 142
void BodyLoadState( out Body body,in float af )
{
    vec2 a=vec2(0.,af)+_bBody;
    body.id=int(af);
    {
        vec4 bvp = 			loadif(txBodyPos+a	,vec4(body.vPos,0.));
        body.vPos=bvp.xyz;
        body.fMass=bvp.w;
    }
    body.qRot = 			loadif(txBodyRot+a,vec4(0.,0.,0.,1.));
    body.vCol = 			loadifb(txBodyCol+a,body.vCol);
}

// Function 143
Robot RobotOfState(Storage s)
{
    Robot r;
    r.pos    = s[SR + 0].xy;
    r.vel    = s[SR + 0].zw;
    r.scan   = s[SR + 1].z; // x and y are free
    r.damage = s[SR + 1].w;
    r.aim    = s[SR + 2].x;
    r.turn   = s[SR + 2].y;
    r.radar  = s[SR + 2].z;
    r.shot   = s[SR + 2].w;
    return r;
}

// Function 144
void SaveState(inout vec4 c, State state, ivec2 p)
{
    ivec2 R = state.resolution;
    if (p.y == R.y - 1) switch (R.x - 1 - p.x) {
      case slotResolution:
        c.xy = vec2(R);
        break;
      case slotEyePos:
        c.xyz = state.eyepos;
        break;
      case slotEyeAim:
        c.xy = state.eyeaim;
        break;
      default:
        break;
    }
}

// Function 145
void BodyLoadState( out Body body, ivec2 addr )
{
    body.vPos = LoadVec3( addr + offsetBodyPos );
    body.qRot = LoadVec4( addr + offsetBodyRot );
    body.vMomentum = LoadVec3( addr + offsetBodyMom );
    body.vAngularMomentum = LoadVec3( addr + offsetBodyAngMom );
}

// Function 146
void LoadState()
{
    vec4 state1 = LoadValue(kTexState1);
    bikeAPos = state1.xyz;
    bikeAAngle = state1.w;
    keyStateLURD = LoadValue(kTexState1 + ivec2(1, 0));
    vec4 tempRotQueue = LoadValue(kTexState1 + ivec2(2, 0));
    rotQueueA = tempRotQueue.xy;
    rotQueueB = tempRotQueue.zw;
    vec4 state2 = LoadValue(kTexState1 + ivec2(3, 0));
    bitsCollected = state2.x;
    framerate = state2.y;
    gameStage = state2.z;
    message = state2.w;
    camFollowPos = LoadValue(kTexState1 + ivec2(4, 0));
    indicatorPulse = LoadValue(kTexState1 + ivec2(5, 0));
    vec4 state3 = LoadValue(kTexState1 + ivec2(6, 0));
    bikeBPos = state3.xyz;
    bikeBAngle = state3.w;
    animA = LoadValue(kTexState1 + ivec2(7, 0));
}

// Function 147
vec3 cellStateToCellColor(
    vec4 cellState)
{
    return hsbToRgb(vec3(
        fract(0.0471 * cellState.z), 
        0.9, 
        cellState.w));
}

// Function 148
void stepState()
{
	rand = rand ^ (rand << 13u);
	rand = rand ^ (rand >> 17u);
	rand = rand ^ (rand << 5u);
	rand *= 1685821657u;
}

// Function 149
void updateState()
{   
    vec3 axisEnterFront = vec3(0., 0., -1.);
	vec3 axisExitFront = vec3(0., 0., -1.);
    float portalEnterTime = 0.;
    float at = 0.;
    position = vec3(-5., 11.0, 26.);
    
    for(int i = 4; i >= 0; i--) { // Set current Portal
        if(t > portalsA[i].time) {
            portalA = portalsA[i];
            break;
        }
    }
    for(int i = 1; i < 10; ++i) { // Set Camera position
        at += states[i].duration;
        portalEnterTime = mix(portalEnterTime, at-states[i].duration, step(0.5, states[i].isEnterTime));
        
        vec3 axisEnterFrontA = -mix(portalA.front, portalB.front, step(0.5, states[i].enterPortal));
        axisEnterFront = mix(axisEnterFront, axisEnterFrontA, step(0.5, states[i].isEnterTime));
        
        vec3 axisExitFrontA = mix(portalA.front, portalB.front, step(0.5, states[i].exitPortal));
        axisExitFront = mix(axisExitFront, axisExitFrontA, step(0.5, states[i].isEnterTime));
        if(t < at) {
            position = mix(states[i].posStart, states[i].posEnd, clamp((t - (at-states[i].duration)) / (states[i].lerptime), 0., 1.));
            break;
        }
    }
    at = 0.;
    for(int i = 1; i < 15; ++i) { // Set Camera orientation
        at += orientations[i].duration;
        if(t < at) {
            vec3 prevOrientation = mix(modOrientationToPortals(orientations[i-1].orientation, axisEnterFront, axisExitFront), orientations[i-1].orientation, step(portalEnterTime, at-orientations[i].duration-orientations[i-1].duration));
            vec3 currentOrientation = mix(modOrientationToPortals(orientations[i].orientation, axisEnterFront, axisExitFront), orientations[i].orientation, step(portalEnterTime, at-orientations[i].duration));
            orientation = normalize(mix(prevOrientation, currentOrientation, clamp((t - (at-orientations[i].duration)) / (orientations[i].lerptime), 0., 1.)));
            break;
        }
    }
}

// Function 150
void BodyStoreState(const in Body body,in float af, inout vec4 fragColor, in vec2 fragCoord )
{
    vec2 a=vec2(0.,af)+_bBody;
    storeValue( a+txBodyCol, 			body.vCol,   	   	fragColor, fragCoord );
    storeValue( a+txBodyColMat,			body.vColMat,  	   	fragColor, fragCoord );
}

// Function 151
void process_text_command_menu( int i, inout int N,
                                inout vec4 params, inout uvec4 phrase,
                                GameState gs )
{
    uvec4 currmenu = md_load( iChannel0, gs.menustate.x );
    if( i == N )
        params = vec4( 24, g_textres.y - 24., -1, 15 ),
        phrase = currmenu;
    N++;
    int j = i - N;
    int n = int( currmenu.w >> 8 ) & 0xff;
    int p = int( currmenu.w >> 16 ) & 0xff;
    if( n > 0 && j >= 0 )
    {
        float y = g_textres.y - 48. - float( j % n ) * 16.;
        if( j < n )
            params = vec4( 24, y, 1, 15 ), phrase = uvec4( ( ( j + 49 ) << 24 ) | 0x2e2000, 0, 0, 3 );
        else
        if( j < 2 * n )
            params = vec4( 48, y, 1, 15 ), phrase = md_load( iChannel0, p + j - n );
    }
    N += 2 * n;
}

// Function 152
float cmd_menu()						{ return cmd_press(MENU_KEY1,		MENU_KEY2); }

// Function 153
void InitState()
{
    gameState.playerPos = vec2(-0.6, -0.25);
    gameState.movementSpeed = vec2(0.0, 0.0);
}

// Function 154
mat4 getState() {
    //half pixel
    vec2 pxSz = 0.5 / iResolution.xy;
    return mat4(
        tex(iChannel1, vec2(pxSz.x,1.)),	 //loc
        tex(iChannel1, vec2(pxSz.x*3.,1.)),	 //vel
        tex(iChannel1, vec2(1.-pxSz.x,1.)),	 //rot
        tex(iChannel1, vec2(1.-pxSz.x*3.,1.))//mou
    );
}

// Function 155
void GameState(inout Storage s)
{
    // detect a winner if only one
    // left alive, or
    // TODO declare ties, etc.
    // TODO keep score.
    // TODO timeout for Veggy vs. Veggy stalemates
    int nalive = 0;
    for (int i = 1; i <= nrobots; ++i) {
        Guts g = robotguts[i-1];
        if (g.alive >= 0.) ++nalive;
    }
    if (nalive < 2) {
        if ((nalive < 1 && fract(iTime) < .06) // pause
            || mod(iTime, 5.) < .06) // there must be only one!
        s[0] = vec4(0,0,0,1); // raise reset flag
    }// else if (s[0].w > 0.) // clear reset flag next frame
    //    s[0] = vec4(0); // Init will clear it for us
    // else carry on with battle
}

// Function 156
void LoadState(sampler2D tex, out AppState s)
{
    vec4 data;

	data = LoadValue(tex, 0, 0);
	s.isSpacePressed = data.x;
    s.stateID      = data.y;
	s.timeFailed   = data.z;
    s.isLeftLine   = data.w;
    
    data = LoadValue(tex, 1, 0);
    s.isFailed   = data.x;
    s.playerCell = data.y;
    s.score		 = data.z;
    s.highscore	 = data.w;
    
    data = LoadValue(tex, 2, 0);
    s.timeAccumulated = data.x;
    s.paceScale       = data.y;    
	s.seed            = data.z;
    s.timeStarted     = data.w;
}

// Function 157
vec4 ReadStateData( sampler2D stateSampler, ivec2 address )
{
    return LoadVec4( stateSampler, address );
}

// Function 158
vec4 updateAutoModeState() {
    return isKeyToggled(KEY_A) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);    
}

// Function 159
void VehicleStoreState( ivec2 addr, const in Vechicle vehicle, inout vec4 fragColor, in vec2 fragCoord )
{
    BodyStoreState( addr + offsetVehicleBody, vehicle.body, fragColor, fragCoord );
    vec4 vParam0 = vec4( vehicle.fSteerAngle, 0.0, 0.0, 0.0 );
    StoreVec4( addr + offsetVehicleParam0, vParam0, fragColor, fragCoord);
}

// Function 160
mat4 getState() {
    vec2 pxSz = 0.5 / iResolution.xy;
    return mat4(
        tex(iChannel0, vec2(pxSz.x,1.)),	 //loc xyz, material
        tex(iChannel0, vec2(pxSz.x*3.,1.)),	 //vel
        tex(iChannel0, vec2(1.-pxSz.x,1.)),	 //rot
        tex(iChannel0, vec2(1.-pxSz.x*3.,1.))//mou
    );
}

// Function 161
float drawMenuText( in vec2 uv, in float msel )
{
    uv.x *= iResolution.y/iResolution.x; // Unfix aspect ratio
    uv = uv *.5 + .5;
    vec2 scr = uv*vec2(320,180);
    vec2 pos = vec2(30,20);
    
    // "LUNAR LANDER"
    float char = drawChar(CH_L,pos,MAP_SIZE,scr);
    char += drawChar(CH_U,pos,MAP_SIZE,scr); 
    char += drawChar(CH_N,pos,MAP_SIZE,scr); 
    char += drawChar(CH_A,pos,MAP_SIZE,scr); 
    char += drawChar(CH_R,pos,MAP_SIZE,scr); 
    pos.x += KERN; 
    char += drawChar(CH_L,pos,MAP_SIZE,scr); 
    char += drawChar(CH_A,pos,MAP_SIZE,scr); 
    char += drawChar(CH_N,pos,MAP_SIZE,scr); 
    char += drawChar(CH_D,pos,MAP_SIZE,scr); 
    char += drawChar(CH_E,pos,MAP_SIZE,scr); 
    char += drawChar(CH_R,pos,MAP_SIZE,scr);
    
    // "Start Game"
    pos = vec2(210,25);
    float charSA = drawChar(CH_RGHT,pos,MAP_SIZE,scr);
    float charA = drawChar(CH_S,pos,MAP_SIZE,scr);
    charA += drawChar(CH_T,pos,MAP_SIZE,scr); 
    charA += drawChar(CH_A,pos,MAP_SIZE,scr); 
    charA += drawChar(CH_R,pos,MAP_SIZE,scr); 
    charA += drawChar(CH_T,pos,MAP_SIZE,scr); 
    pos.x += KERN;
    charA += drawChar(CH_G,pos,MAP_SIZE,scr); 
    charA += drawChar(CH_A,pos,MAP_SIZE,scr); 
    charA += drawChar(CH_M,pos,MAP_SIZE,scr); 
    charA += drawChar(CH_E,pos,MAP_SIZE,scr);
    charSA += drawChar(CH_LEFT,pos,MAP_SIZE,scr);
    
    // "How to play"
    pos.x = 207.0;
    pos.y -= 10.0;
    float charSB = drawChar(CH_RGHT,pos,MAP_SIZE,scr);
    float charB = drawChar(CH_H,pos,MAP_SIZE,scr); 
    charB += drawChar(CH_O,pos,MAP_SIZE,scr); 
    charB += drawChar(CH_W,pos,MAP_SIZE,scr); 
    pos.x += KERN; 
    charB += drawChar(CH_T,pos,MAP_SIZE,scr); 
    charB += drawChar(CH_O,pos,MAP_SIZE,scr);
    pos.x += KERN; 
    charB += drawChar(CH_P,pos,MAP_SIZE,scr); 
    charB += drawChar(CH_L,pos,MAP_SIZE,scr); 
    charB += drawChar(CH_A,pos,MAP_SIZE,scr); 
    charB += drawChar(CH_Y,pos,MAP_SIZE,scr); 
    charSB += drawChar(CH_LEFT,pos,MAP_SIZE,scr);
    
    float time = iTime - floor(uv.x*320.0)*.003125;
    float sheen = .5 + .5*smoothstep( .995, 1.00, sin(time));
    sheen = floor(sheen*4.0)*.25;
    
    // This relies heavily on MS_START being 0.0 and MS_HOWTO being 1.0
    // to display the correct text.
    return (char*sheen + charA + charSA*sheen + charB*sheen)*(1.0-msel) +
           (char*sheen + charB + charSB*sheen + charA*sheen)*(msel);
}

// Function 162
void GameLoadState()
{
    vec3 state = LoadVec3(addrGameState);
    
 	gtime = state.x;
    time_scale = state.y;
    input_state = state.z;
}

// Function 163
PrintState PrintState_InitCanvas( vec2 vCoords, vec2 vPixelSize )
{
    PrintState state;
    state.vPixelPos = vCoords;
    state.vPixelSize = vPixelSize;
    
    MoveTo( state, vec2(0) );

    ClearPrintResult( state );
    
    return state;
}

// Function 164
void MenuText(inout vec3 color, vec2 p, in AppState s)
{
	p -= vec2(-160, -1);

	vec2 scale = vec2(4., 8.);
	vec2 t = floor(p / scale);

	float tab = 1.;
	if (t.y >= 6. && t.y < 10.)
	{
		p.x -= tab * scale.x;
		t.x -= tab;
	}
	if (t.y >= 0. && t.y < 5.)
	{
		p.x -= tab * scale.x;
		t.x -= tab;
	}
	if (t.y >= 0. && t.y < 3.)
	{
		p.x -= tab * scale.x;
		t.x -= tab;
	}

	uint v = 0u;
	v = t.y == 10. ? (t.x < 4. ? 1718777171u : (t.x < 8. ? 6644577u : 0u)) : v;
	v = t.y == 9. ? (t.x < 4. ? 1635018061u : (t.x < 8. ? 108u : 0u)) : v;
	v = t.y == 8. ? (t.x < 4. ? 1818585412u : (t.x < 8. ? 1920230245u : 25449u)) : v;
	v = t.y == 7. ? (t.x < 4. ? 1735749458u : (t.x < 8. ? 1936027240u : 115u)) : v;
	v = t.y == 6. ? (t.x < 4. ? 1702060354u : (t.x < 8. ? 1819231008u : 29295u)) : v;
	v = t.y == 5. ? (t.x < 4. ? 1751607628u : (t.x < 8. ? 1735289204u : 0u)) : v;
	v = t.y == 4. ? (t.x < 4. ? 1717987652u : (t.x < 8. ? 6648693u : 0u)) : v;
	v = t.y == 3. ? (t.x < 4. ? 1667592275u : (t.x < 8. ? 1918987381u : 0u)) : v;
	v = t.y == 2. ? (t.x < 4. ? 1953720644u : (t.x < 8. ? 1969383794u : 1852795252u)) : v;
	v = t.y == 1. ? (t.x < 4. ? 1936028230u : (t.x < 8. ? 7103854u : 0u)) : v;
	v = t.y == 0. ? (t.x < 4. ? 1836016967u : (t.x < 8. ? 2037544037u : 0u)) : v;
	v = t.x >= 0. && t.x < 12. ? v : 0u;

	float c = float((v >> uint(8. * t.x)) & 255u);

	vec3 textColor = vec3(.3);
	if (t.y == 10. - s.menuId)
	{
		textColor = vec3(0.74, 0.5, 0.12);
	}

	p = (p - t * scale) / scale;
	p.x = (p.x - .5) * .45 + .5;
	float sdf = TextSDF(p, c);
	if (c != 0.)
	{
		color = mix(textColor, color, smoothstep(-.05, +.05, sdf));
	}
}

// Function 165
bool TeletextState_GetSeparatedGfx( TeletextState state )
{
    if ( IsControlCharacter( state.char ) )
    {
        if ( state.bHoldGfx )
        {
            return state.bHeldSeparated;
        }
        else
        {
            return false;
        }
    }    
    
    return state.bSeparatedGfx;
}

// Function 166
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

// Function 167
void pz_initializeState() {
    pz_realBufferResolution     = iChannelResolution[pz_stateBuf].xy;
    pz_originalBufferResolution = pz_stateSample(.5/pz_realBufferResolution).xy;
}

// Function 168
void LoadState(out AppState s)
{
    vec4 data;

    data = LoadValue(0, 0);
    s.menuId = data.x;
    s.metal = data.y;
    s.roughness = data.z;
    s.baseColor = data.w;

    data = LoadValue(1, 0);
    s.focus = data.x;
    s.focusObjRot = data.y;
    s.objRot = data.z;
}

// Function 169
float keystate( int key )
	{ return texelFetch( iChannel3, ivec2( key, 0 ), 0 ).x; }

// Function 170
vec4 mainMenuButton(in vec2 uv, in vec2 min_b, in vec2 max_b, in float _val, in float n)
{
    vec2 center = (min_b + max_b)*0.5;
    vec2 size1 = (max_b - min_b) * 0.5;
    vec2 frame = size1*vec2(0.98, 0.93);
    
    float ratio = iResolution.x / iResolution.y;
    vec2 scl_uv = uv;
    scl_uv.x *= ratio;
    
    vec3 background = vec3(0.18, 0.18, 0.18);
    
    float inside = step(sdBox(uv - center, frame) - 0.01*frame.x, 0.);
    float boundary = step(sdBox(uv - center, size1), 0.) - inside;
    float val = 0.5 + 0.3*sin(PI*0.5 + PI2*saturate(uv.x, min_b.x, max_b.x));
    float start_x = center.x - size1.x;
    float bval = saturate(uv.x, start_x, center.x + size1.x);
    
    float inv_n = 1. / n;
    
    float modeDiscr = floor(_val * n)*inv_n;
    
    vec2 dstart = min_b + vec2(size1.x*.1, size1.y*.75);
    vec4 tcol;
    float tsize = size1.y * .4;
    tcol += drawTextHorizontal(uv, dstart, tsize, vec2[10](_T, _o, _o, _l, _s, _X, _X, _X, _X, _X), 5);
    
    dstart = min_b + vec2(size1.x*.6, size1.y*.75);
    tcol += drawTextHorizontal(uv, dstart, tsize, vec2[10](_C, _o, _l, _o, _r, _X, _X, _X, _X, _X), 5);
    
    dstart = min_b + vec2(size1.x*1.1, size1.y*.75);
    tcol += drawLetter(uv, dstart, tsize, _3);
    dstart.x += tsize*.5;
    tcol += drawLetter(uv, dstart, tsize, _D);
    
    dstart = min_b + vec2(size1.x*1.47, size1.y*.75);
    tcol += drawTextHorizontal(uv, dstart, tsize, vec2[10](_T, _e, _x, _t, _u, _r, _e, _X, _X, _X), 7);
    
    float selZone = (step(modeDiscr, bval) - step(modeDiscr + inv_n, bval));
    tcol.z *= selZone;
    
    vec4 color = clamp(vec4(inside, boundary, 
                            tcol.x, selZone), 0., step(0.5, inside+boundary));

    return color;
}

// Function 171
void loadGameStateMinimal(float time, sampler2D storage) {
    vec4 f;

    f = texelFetch(storage, ivec2(0,0), 0);
#ifdef FIXED_TIME_STEP
    gDT = (1./60.);
#else
    gDT = time - f.x;
#endif
    gCanonMovement = f.y;
    gCanonAngle = f.z;
    gMode = f.w;
    
    f = texelFetch(storage, ivec2(1,0), 0);
    gLastShot = f.x;
    gLastAircraft = f.y;
    gScore = f.z;
    gHighScore = f.w;
    
    f = texelFetch(storage, ivec2(2,0), 0);
    gLastParatrooper = f.x;
    gEndRoundTime = f.y;
    gEndRoundTimeCoolDown = f.z;
    gGameOverTime = f.w;
    
    gDeadParatroopers = texelFetch(storage, ivec2(3,0), 0);
    gParatroopersLeft = texelFetch(storage, ivec2(4,0), 0);
    gParatroopersRight = texelFetch(storage, ivec2(5,0), 0);
    
    gExplosion1 = texelFetch(storage, ivec2(6,0), 0);
    gExplosion2 = texelFetch(storage, ivec2(7,0), 0);
}

// Function 172
void SaveState(inout vec4 c, int u, Storage s)
{
    c = s[u];
}

// Function 173
vec4
paintMenuVoxel (in vec2 pos, in float invScale, in uint type)
{
    return drawVoxelTypeIcon(pos, invScale, type);
}

// Function 174
vec4 getTransitionPanel( in vec2 uv, in float index ) {
    float dither = texture(iChannel2, uv/8.0).r;
    
    // get some light
    const vec2 lightSize = vec2(100.0, 100.0);
    float lightIndex = floor(uv.y / lightSize.y);
    float ss = sign(uv.x);
    ss = ss == 0.0 ? 1.0: ss;
    vec2 uvLight = vec2(ss, lightIndex + 0.5);
    uvLight.y = clamp(uvLight.y, -2.5, 3.5);
    uvLight *= lightSize;
    float lightValue = fract(lightIndex*0.1-gameplayGlobalTime*0.25);
    lightValue *= lightValue; lightValue *= lightValue;
    lightValue *= lightValue; lightValue *= lightValue;
    float distToLight = length(uv-uvLight);
    
    vec4 baseColor = vec4(BACKGROUND_COLOR, 0.0);
    
    // set light pole
    if (distToLight < 6.0) {
        baseColor.a = 1.0;
        if (distToLight > 5.0) baseColor.rgb = vec3(0.0);
        else {
            baseColor.rgb *= 0.2;
            float lens = 1.0-smoothstep(0.0, 6.0, distToLight);
            lens = doDithering(lens, dither, 4.0);
            baseColor.rgb += BACKGROUND_COLOR*lens*0.45*(lightValue*0.5+0.5);
        }
    }
    
    // box center
    const vec2 boxCenter = vec2(0.0, 350.0);
    const vec2 boxDim = vec2(60.0, 40.0);
    vec2 uvBox = uv-boxCenter;
    
    float boxDist = box(uvBox, boxDim)-5.0;
    if (boxDist < 0.0) {
        baseColor.a = 1.0;
        if (boxDist > -1.5) baseColor.rgb = vec3(0.0);
        else {
            float noise = texture(iChannel3, uvBox/256.0, -100.0).r;
            vec3 color = mix(BACKGROUND_COLOR, vec3(0.2), 0.6);
            baseColor.rgb = color+noise*0.1;
            baseColor -= smoothstep(-5.0, 0.0, boxDist)*0.3;
            float inBoxDist = box(uvBox, boxDim-5.0)-5.0;
            if (inBoxDist < 0.0) {
                if (inBoxDist > -2.5) baseColor.rgb = vec3(0.0);
                else {
                    float d = dot(uvBox, vec2(3, 1)*0.1);
                    d = sin(d)*0.5+0.5;
                    d = smoothstep(0.7, 1.0, d);
                    float dd = dot(uvBox, vec2(-5, 3)*0.3);
                    dd = sin(dd)*0.5+0.5;
                    vec3 grey = mix(baseColor.rgb, vec3(0.4), 0.7);
                    baseColor.rgb = mix(grey, vec3(0.3), d);
                    baseColor.rgb -= dd*noise*0.5;
                    baseColor -= smoothstep(-15.0, 0.0, inBoxDist)*0.2;
                    index = min(index, 99.0);
                    float a = mod(index, 10.0);
                    float b = floor(index / 10.0);
                    float digit = 0.0;
                    digit = max(digit, SampleDigit(a, uvBox*0.02 + vec2(-0.15, 0.5)));
                    digit = max(digit, SampleDigit(b, uvBox*0.02 + vec2(0.9, 0.5)));
                    baseColor.rgb += digit*BACKGROUND_COLOR*1.1;
                }
            }
        }
    }
    
    // light glow
    float lightRadius = (1.0-smoothstep(0.0, 45.0, distToLight))*lightValue;
    lightRadius *= 0.5;
    
    baseColor = mix(baseColor, vec4(BACKGROUND_COLOR*2.0, 1.0), lightRadius);
    
    return baseColor;
}

// Function 175
void TeletextState_SetGfxColor( inout TeletextState state, int color )
{
    state.iFgCol = color;
    state.bGfx = true;            
    state.bConceal = false;
}

// Function 176
mat4 getState() {
    return mat4(load(S0), load(S1), load(S2), load(S3));
}

// Function 177
int Enemy_GetState( Entity entity )
{
    return int(entity.fArmor);
}

// Function 178
vec4 stateSave(float pos){
    if(pos == 0.5)
        return vec4(state.init, state.mouse_click, state.mouse_x, state.mouse_y);
    if(pos == 1.5)
        return vec4(state.scale, state.zzzZZZ, state.zzzZZZ, state.zzzZZZ);
}

// Function 179
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

// Function 180
void UI_StoreWindowState( inout UIContext uiContext, UIWindowState window, int iData )
{    
    vec4 vData0;
    vData0.xy = window.rect.vPos;
    vData0.zw = window.rect.vSize;
    
    StoreVec4( ivec2(iData,0), vData0, uiContext.vOutData, ivec2(uiContext.vFragCoord) );        

    vec4 vData1;
    
    vData1.x = window.bMinimized ? 1.0f : 0.0f;
    vData1.y = DIRTY_DATA_MAGIC;
    vData1.zw = vec2(0);

    StoreVec4( ivec2(iData,1), vData1, uiContext.vOutData, ivec2(uiContext.vFragCoord) );        
}

// Function 181
bool keyState(int key) { return keys(key, 2).x < 0.5; }

// Function 182
bool GeyKeyState(in float key)
{
	return (texture(iChannel1, vec2(key, 0.25)).x > 0.5);   
}

// Function 183
vec4 SaveState(in AppState s, in vec2 fragCoord, int iFrame)
{
    if (iFrame <= 0)
    {
		s.stateID = GS_SPLASH;
		s.isSpacePressed  = 0.0;
		s.timeFailed      = 0.0;
        s.isLeftLine      = 0.0;
        
        s.isFailed        = 0.0;
        s.playerCell      = 0.0;
        s.score           = 0.0;
        s.highscore       = 0.0;
        
        s.paceScale       = 0.0;
        s.timeAccumulated = 0.0;
        s.seed            = fbm3(iDate.yzw);
        s.timeStarted     = 0.0;
    }
    
    vec4 ret = vec4(0.);
	StoreValue(vec2(0., 0.), vec4(s.isSpacePressed,  s.stateID,    s.timeFailed,   s.isLeftLine),   ret, fragCoord);
	StoreValue(vec2(1., 0.), vec4(s.isFailed,        s.playerCell, s.score,        s.highscore),    ret, fragCoord);
    StoreValue(vec2(2., 0.), vec4(s.timeAccumulated, s.paceScale,  s.seed,         s.timeStarted),  ret, fragCoord);
    return ret;
}

// Function 184
void writeState(in state s, in vec2 fragCoord, inout vec4 fragColor) {
    if (abs(fragCoord.y - 0.0-0.5) < 0.5) {
        if (abs(fragCoord.x - 0.0-0.5) < 0.5) {
            fragColor = vec4(s.p, 1.0);
        } else if (abs(fragCoord.x - 1.0-0.5) < 0.5) {
            fragColor = vec4(s.q, 1.0);
        } else if (abs(fragCoord.x - 2.0-0.5) < 0.5) {
            fragColor = vec4(s.v, 1.0);
        } else if (abs(fragCoord.x - 3.0-0.5) < 0.5) {
            fragColor = vec4(s.L, 1.0);
        } else if (abs(fragCoord.x - 4.0-0.5) < 0.5) {
            fragColor = s.pr;
        }
    }
}

// Function 185
void saveGameState(ivec2 uv, float time, inout vec4 f) {
    if(uv.x == 0) f = vec4(time, gCanonMovement, gCanonAngle, gMode);
    if(uv.x == 1) f = vec4(gLastShot, gLastAircraft, gScore, gHighScore);
    if(uv.x == 2) f = vec4(gLastParatrooper,gEndRoundTime,gEndRoundTimeCoolDown,gGameOverTime);
    if(uv.x == 3) f = gDeadParatroopers;
    if(uv.x == 4) f = gParatroopersLeft;
    if(uv.x == 5) f = gParatroopersRight;
    if(uv.x == 6) f = gExplosion1;
    if(uv.x == 7) f = gExplosion2;
    
    for (int i=0; i<MAX_BULLETS; i++) {
        if(uv.x == i+BULLET_DATA_OFFSET) f = vec4(gBulletData[i],0);
    }
    for (int i=0; i<MAX_AIRCRAFTS/2; i++) {
        if(uv.x == i+AIRCRAFT_DATA_OFFSET) f = vec4(gAircraftData[i*2+0], gAircraftData[i*2+1]);
    }
    for (int i=0; i<MAX_PARATROOPERS; i++) {
        if(uv.x == i+PARATROOPER_DATA_OFFSET) f = gParatrooperData[i];
    }
}

// Function 186
void LoadState(in sampler2D channel)
{
    vec4 playerFetch = texelFetch(channel, ivec2(0), 0);
    
    gameState.playerPos = playerFetch.xy;
    gameState.movementSpeed = playerFetch.zw;
}

// Function 187
Guts GutsOfState(Storage s)
{
    Guts g;
    g.pos     = s[SG + 0].xy;
    g.vel     = s[SG + 0].zw;
    g.bullet.pos   = s[SG + 1].xy;
    g.bullet.dir   = s[SG + 1].z;
    g.bullet.timer = s[SG + 1].w;
    g.explosionpos = s[SG + 2].xy;
    g.explosiontimer = s[SG + 2].w; // z is free
    g.bearing = s[SG + 3].x;
    g.health  = s[SG + 3].y;
    g.reload  = s[SG + 3].z;
    g.alive   = s[SG + 3].w;
    return g;
}

// Function 188
void LoadState(out State state, sampler2D A, ivec2 R)
{
	vec4[slotCount] data;
	for (int i = slotCount; i-- > 0; )
        data[i] = fetch(A, R-1-ivec2(i,0));
    state.resolution = ivec2(data[slotResolution].xy);
    state.eyepos = data[slotEyePosAz].xyz;
    state.eyevel = data[slotEyeVelEl].xyz;
    state.eyeaim = vec2(data[slotEyePosAz].w
                       ,data[slotEyeVelEl].w);
}

// Function 189
float get_main_menu_val(in sampler2D s)
{
    return texelFetch(s, CTRL_GUI_MENU, 0).w;
}

// Function 190
SceneState SetupSceneState()
{
    SceneState sceneState;
    
    sceneState.vehicleState.vPos = LoadVec3( addrVehicle + offsetVehicleBody + offsetBodyPos );
    
    sceneState.vehicleState.qRot = LoadVec4( addrVehicle + offsetVehicleBody + offsetBodyRot );
    sceneState.vehicleState.mRot = QuatToMat3( sceneState.vehicleState.qRot );

    vec4 vWheelState0 = LoadVec4( addrVehicle + offsetVehicleWheel0 );
    vec4 vWheelState1 = LoadVec4( addrVehicle + offsetVehicleWheel1 );
    vec4 vWheelState2 = LoadVec4( addrVehicle + offsetVehicleWheel2 );
    vec4 vWheelState3 = LoadVec4( addrVehicle + offsetVehicleWheel3 );
    
    sceneState.vehicleState.vWheelState0 = vWheelState0;
    sceneState.vehicleState.vWheelState1 = vWheelState1;
    sceneState.vehicleState.vWheelState2 = vWheelState2;
    sceneState.vehicleState.vWheelState3 = vWheelState3;
    
    sceneState.vehicleState.vWheelSC0 = vec4( sin(vWheelState0.x), cos(vWheelState0.x), sin(vWheelState0.y), cos(vWheelState0.y) );
    sceneState.vehicleState.vWheelSC1 = vec4( sin(vWheelState1.x), cos(vWheelState1.x), sin(vWheelState1.y), cos(vWheelState1.y) );
    sceneState.vehicleState.vWheelSC2 = vec4( sin(vWheelState2.x), cos(vWheelState2.x), sin(vWheelState2.y), cos(vWheelState2.y) );
    sceneState.vehicleState.vWheelSC3 = vec4( sin(vWheelState3.x), cos(vWheelState3.x), sin(vWheelState3.y), cos(vWheelState3.y) );
    
    return sceneState;
}

// Function 191
vec4 wwtransition(in ivec2 p)
{
    ivec3 e = ivec3(1,0,-1);
    // current state
    vec4 s = wwmap(p), n[8];
    // moor neighbourhood
    n[0] = wwmap(p+e.zz); n[1] = wwmap(p+e.yz); n[2] = wwmap(p+e.xz);
	n[3] = wwmap(p+e.zy);                       n[4] = wwmap(p+e.xy);
    n[5] = wwmap(p+e.zx); n[6] = wwmap(p+e.yx); n[7] = wwmap(p+e.xx);
    
	if (STATE(s) == S_EMPTY)
    {
    }
	else if (STATE(s) == S_HEAD)
    {
    	SET_STATE(s, S_TAIL);    
    }
	else if (STATE(s) == S_TAIL)
    {
    	SET_STATE(s, S_WIRE);    
    }
	else //if (STATE(s) == S_WIRE)
    {
        int num = 0;
        for (int i=0; i<8; ++i) { if (STATE(n[i]) == S_HEAD) { ++num; if (num > 2) break; } }
        if (num == 1 || num == 2)
    		SET_STATE(s, S_HEAD);    
    }
    
    return s;
}

// Function 192
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

// Function 193
void GameSetState(float state)
{
    gGameState     = state;
    gGameStateTime = 0.0;
}

// Function 194
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

// Function 195
void storeState(inout vec4 fragColor, in ivec2 fragCoord, mat4 s) {
    store(S0, s[0]);
    store(S1, s[1]);
    store(S2, s[2]);
    store(S3, s[3]);
}

// Function 196
void initState()
{  
    states[0].posStart = vec3(0., 11.0, 26.0); states[0].posEnd = vec3(0., 11.0, 26.0); states[0].duration = 0.; states[0].lerptime = 0.; states[0].isEnterTime = 0.; states[0].enterPortal = 0.;  states[0].exitPortal = 1.;
    states[1].posStart = vec3(0., 11.0, 26.0); states[1].posEnd = vec3(0., 11.0, 23.0); states[1].duration = 6.75; states[1].lerptime = 0.5; states[0].isEnterTime = 0.; states[0].enterPortal = 0.;  states[0].exitPortal = 1.;
    states[2].posStart = vec3(0., 11.0, 23.0); states[2].posEnd = vec3(-12.6, 11.0, 23.0); states[2].duration = 4.0; states[2].lerptime = 1.0; states[0].isEnterTime = 0.; states[0].enterPortal = 0.;  states[0].exitPortal = 1.;
    
    // Go to Portal B
    states[3].posStart = vec3(-12.6, 11.0, 23.0); states[3].posEnd = vec3(-14.5, 11., 23.0); states[3].duration = 0.1; states[3].lerptime = 0.1; states[0].isEnterTime = 0.; states[0].enterPortal = 0.;  states[0].exitPortal = 1.;
    states[4].posStart = vec3(14.5, 11., 7.5); states[4].posEnd = vec3(13.5, 11., 7.5); states[4].duration = 3.0; states[4].lerptime = 0.5; states[0].isEnterTime = 1.; states[0].enterPortal = 0.;  states[0].exitPortal = 1.;
    
    // Go to Portal A
    states[5].posStart = vec3(13.5, 11., 7.5); states[5].posEnd = vec3(14.5, 11., 7.5); states[5].duration = 0.1; states[5].lerptime = 1.25; states[0].isEnterTime = 0.; states[0].enterPortal = 0.;  states[0].exitPortal = 1.;
    states[6].posStart = vec3(-14.5, 11., -2.5); states[6].posEnd = vec3(-11.6, 11., -2.5); states[6].duration = 15.; states[6].lerptime = 1.0; states[0].isEnterTime = 1.; states[0].enterPortal = 0.;  states[0].exitPortal = 1.;
    
    states[7].posStart = vec3(-11.6, 11., -2.5); states[7].posEnd = vec3(-11.6, 11., -23.0); states[7].duration = 7.5; states[7].lerptime = 7.0; states[0].isEnterTime = 0.; states[0].enterPortal = 0.;  states[0].exitPortal = 1.;
    states[8].posStart = vec3(-11.6, 11., -23.0); states[8].posEnd = vec3(0.5, 11., -23.0); states[8].duration = 3.0; states[8].lerptime = 2.5; states[0].isEnterTime = 0.; states[0].enterPortal = 0.;  states[0].exitPortal = 1.;
    states[9].posStart = vec3(0.5, 11., -23.0); states[9].posEnd = vec3(0.5, 11., -28.0); states[9].duration = 10.; states[9].lerptime = 0.5; states[0].isEnterTime = 0.; states[0].enterPortal = 0.;  states[0].exitPortal = 1.;
    
    t = mod(t, 40.);
}

// Function 197
bool key_state(int key) {
    return texelFetch(iChannel3, ivec2(key, 0), 0).x != 0.;
}

// Function 198
void Cam_StoreState( ivec2 addr, const in CameraState cam, inout vec4 fragColor, in ivec2 fragCoord )
{
    StoreVec4( addr + ivec2(0,0), vec4( cam.vPos, 0 ), fragColor, fragCoord );
    StoreVec4( addr + ivec2(1,0), vec4( cam.vTarget, cam.fFov ), fragColor, fragCoord );    
    StoreVec4( addr + ivec2(2,0), vec4( cam.vUp, 0 ), fragColor, fragCoord );    
    StoreVec4( addr + ivec2(3,0), vec4( cam.vJitter, cam.fPlaneInFocus, cam.bStationary ? 1.0f : 0.0f ), fragColor, fragCoord );    
}

// Function 199
UIWindowState UI_GetWindowState( UIContext uiContext, int iControlId, int iData, UIWindowDesc desc )
{
    UIWindowState window;    
    
    vec4 vData0 = LoadVec4( iChannelUI, ivec2(iData,0) );
        
    window.rect = Rect( vData0.xy, vData0.zw );
    
    vec4 vData1 = LoadVec4( iChannelUI, ivec2(iData,1) );
    
    window.bClosed = false;
    window.bMinimized = (vData1.x > 0.0);    
    
    // Clamp window position so title bar is always on canvas
	vec2 vSafeMin = vec2(24.0);        
	vec2 vSafeMax = vec2(32.0);        
    vec2 vPosMin = vec2( -window.rect.vSize.x + vSafeMin.x, -vSafeMin.y);//vec2( -window.rect.vSize.x, 0.0) + 24.0, -24.0 );
    vec2 vPosMax = uiContext.drawContext.vCanvasSize - vSafeMax;
    window.rect.vPos = clamp( window.rect.vPos, vPosMin, vPosMax );
    
    if ( iFrame == 0 || vData1.y != DIRTY_DATA_MAGIC)
    {
        window.rect = desc.initialRect;
        window.bMinimized = desc.bStartMinimized;
    }       
    
    window.uControlFlags = desc.uControlFlags;
    window.vMaxSize = desc.vMaxSize;
    
    window.iControlId = iControlId;
        
    return window;
}

// Function 200
void Cam_LoadState( out CameraState cam, sampler2D sampler, vec2 addr )
{
    vec4 vPos = LoadVec4( sampler, addr + vec2(0,0) );
    cam.vPos = vPos.xyz;
    vec4 targetFov = LoadVec4( sampler, addr + vec2(1,0) );
    cam.vTarget = targetFov.xyz;
    cam.fFov = targetFov.w;
}

// Function 201
JoystickState getJoystickState()
{
    JoystickState res;
    res.isFirePressed = KEY_DOWN(KEY_CTRL);

    res.dir = ivec2(0);
    res.dir += KEY_DOWN(KEY_LT_ARR) ? DIR_LT : DIR_NONE;
    res.dir += KEY_DOWN(KEY_UP_ARR) ? DIR_UP : DIR_NONE;
    res.dir += KEY_DOWN(KEY_RT_ARR) ? DIR_RT : DIR_NONE;
    res.dir += KEY_DOWN(KEY_DN_ARR) ? DIR_DN : DIR_NONE;
    res.dir *= res.dir.x != 0 ? ivec2(1, 0) : ivec2(0, 1);  // horizontal is in priority

    return res;
}

// Function 202
bool HandleState(vec2 aFragCoord, out vec4 oNewValue)
{
    if (IsVariable(aFragCoord, VAR_CAMERA_POS_xyz))
    {
        vec4 vOldValue = ReadVec4(iChannel0, VAR_CAMERA_POS_xyz);
        oNewValue = vOldValue;
     	return true;   
    }
    else if (IsVariable(aFragCoord, VAR_CAMERA_ROT_xy))
    {
        vec4 vOldValue = ReadVec4(iChannel0, VAR_CAMERA_ROT_xy);
        oNewValue = vOldValue;
     	return true;   
    }
    return false;
}

// Function 203
vec4 fetchOtherState(ivec2 fragCoord) {
    return texelFetch(iChannel1, fragCoord, 0);
}

// Function 204
void TeletextState_SetAlphanumericColor( inout TeletextState state, int color )
{
    state.iFgCol = color;
    state.bGfx = false;
    state.bConceal = false;
}

// Function 205
Storage LoadState(sampler2D ch, int v)
{
    Storage s;
    for (int u = SU; u < s.length(); ++u)
        s[u] = texelFetch(ch, ivec2(u,v), 0);
    return s;
}

// Function 206
KnobState CreateKnobState(vec2 p, vec2 r, bool signed, float n)
{
    KnobState state;
    state.p = p;
    state.r = r;
    state.signed = signed;
    state.n = n;

    return state;
}

// Function 207
void LoadState(out State state, sampler2D A, ivec2 R)
{
	vec4[slotCount] data; // at least temporarily
	for (int i = slotCount; i-- > 0; )
        data[i] = fetch(A, R-1-ivec2(i,0));
    state.resolution = ivec2(data[slotResMBD].xy);
    state.mbdown = data[slotResMBD].z > .5;
    state.eyepos = data[slotEyePos].xyz;
    state.eyevel = data[slotEyeVel].xyz;
    state.eyeaim = data[slotAzElBase].xy;
    state.aimbase = data[slotAzElBase].zw;
}

// Function 208
float transition_function(vec2 disk_ring) {
    return sigmoid_mix(sigmoid_ab(disk_ring.x, b1, b2, alpha_n, alpha_n),
                       sigmoid_ab(disk_ring.x, d1, d2, alpha_n, alpha_n), disk_ring.y, alpha_m
                      );
}

// Function 209
void stateLoad(float pos, vec4 v){
    if(pos == 0.5){
		state.init = v.x;    	
        state.mouse_click = v.y;
    	state.mouse_x = v.z; 
        state.mouse_y = v.w;
    }
    if(pos == 1.5){
		state.scale = v.x;    	
        //state.mouse_click = v.y;
    	//state.mouse_x = v.z; 
        //state.mouse_y = v.w;
    }    
    
}

// Function 210
void MenuText(inout vec3 color, vec2 p, in AppState s)
{
    p -= vec2(-160, 62);
    
    vec2 scale = vec2(4., 8.);
    vec2 t = floor(p / scale);   
    
    uint v = 0u;
	v = t.y == 2. ? (t.x < 4. ? 1768452929u : (t.x < 8. ? 1768777835u : (t.x < 12. ? 5653614u : 0u))) : v;
	v = t.y == 1. ? (t.x < 4. ? 1918986307u : (t.x < 8. ? 1147496812u : (t.x < 12. ? 1752383839u : (t.x < 16. ? 1835559785u : 5664361u)))) : v;
	v = t.y == 0. ? (t.x < 4. ? 1918986307u : (t.x < 8. ? 1147496812u : (t.x < 12. ? 86u : 0u))) : v;
	v = t.x >= 0. && t.x < 20. ? v : 0u;
    
	float c = float((v >> uint(8. * t.x)) & 255u);
    
    vec3 textColor = vec3(.3);
    if (t.y == 2. - s.menuId)
    {
        textColor = vec3(0.74, 0.5, 0.12);
	}

    p = (p - t * scale) / scale;
    p.x = (p.x - .5) * .5 + .5;
    float sdf = TextSDF(p, c);
    if (c != 0.)
    {
    	color = mix(textColor, color, smoothstep(-.05, +.05, sdf));
    }
}

// Function 211
void WanderCam_LoadState( out WanderCamState wanderCam, sampler2D sampler, ivec2 addr )
{
    vec4 vPos = LoadVec4( sampler, addr + ivec2(0,0) );
    wanderCam.pos = vPos.xyz;
    vec4 vLookAt = LoadVec4( sampler, addr + ivec2(1,0) );
    wanderCam.lookAt = vLookAt.xyz;
    vec4 vMisc = LoadVec4( sampler, addr + ivec2(2,0) );    
    wanderCam.targetAngle = vMisc.x;
    wanderCam.lookAtAngle = vMisc.y;
    wanderCam.eyeHeight = vMisc.z;
    wanderCam.timer = vMisc.w;
    
    vec4 vMisc2 = LoadVec4( sampler, addr + ivec2(3,0) );    
    wanderCam.iSitting = int( vMisc2.y );
    wanderCam.shoreDistance = vMisc2.z;
    wanderCam.lookAtElevation = vMisc2.w;
}

// Function 212
void LoadState(out State state, sampler2D A, ivec2 R)
{
	vec4[slotCount] data; // at least temporarily
	for (int i = slotCount; i-- > 0; )
        data[i] = fetch(A, R-1-ivec2(i,0));
    state.resolution = ivec2(data[slotResolution].xy);
    state.eyepos = data[slotEyePosAz].xyz;
    state.eyevel = data[slotEyeVelEl].xyz;
    state.eyeaim = vec2(data[slotEyePosAz].w
                       ,data[slotEyeVelEl].w);
}

// Function 213
void StoreState(inout vec4 fragColor, in vec2 fragCoord)
{
    vec4 state1 = vec4(gGameState, gGameStateTime, gGameSeed, gGameInit);
    vec4 state2 = vec4(gPlayerCoords, gPlayerNextCoords);
    vec4 state3 = vec4(gPlayerMotionTimer, gPlayerRotation, gPlayerNextRotation, gPlayerScale);
    vec4 state4 = vec4(gPlayerVisualCoords, gPlayerVisualRotation);
    vec4 state5 = vec4(gPlayerDeathCause, gPlayerDeathTime, gScore, gFbScale);
    
    StoreValue(kTexState1, state1, fragColor, fragCoord);
    StoreValue(kTexState2, state2, fragColor, fragCoord);
    StoreValue(kTexState3, state3, fragColor, fragCoord);
    StoreValue(kTexState4, state4, fragColor, fragCoord);
    StoreValue(kTexState5, state5, fragColor, fragCoord);
}

// Function 214
void VehicleLoadState( out Vechicle vehicle, ivec2 addr )
{    
    BodyLoadState( vehicle.body, addr + offsetVehicleBody );

    vec4 vParam0;
    vParam0 = LoadVec4( addr + offsetVehicleParam0 );
    vehicle.fSteerAngle = vParam0.x;
}

// Function 215
void CameraStoreState( Camera cam, in ivec2 addr, inout vec4 fragColor, in vec2 fragCoord )
{
    StoreVec3( addr + offsetCameraPos, cam.vPos, fragColor, fragCoord );
    StoreVec3( addr + offsetCameraTarget, cam.vTarget, fragColor, fragCoord );    
}

// Function 216
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

// Function 217
float inputState(in ivec2 ip)
{
    vec2 p = (vec2(ip) + vec2(0.5, 1.5)) / iChannelResolution[0].xy;
    return texture(iChannel0, p).x;
}


```