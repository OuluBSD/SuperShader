// Reusable Interaction Feedback Game Functions
// Automatically extracted from game/interactive-related shaders

// Function 1
vec3 applyHighlight(in vec2 uv, in Orb orb, in Light light, in vec3 eye){
    vec3 colour = vec3(0.);
    if (distance(orb.center, uv) >= orb.radius){
        return colour;
     }
    
    vec2 distFromCent = uv - orb.center;    
    float uvHeight = sqrt(orb.radius - (pow(distFromCent.x,2.) + pow(distFromCent.y,2.)));
    vec3 uvw = vec3(uv, uvHeight);
    vec3 normal = normalize(vec3(uv, uvHeight) - vec3(orb.center, 0.));
    vec3 orbToLight = normalize(light.pos - vec3(orb.center, 0.));
    
    return vec3(pow(dot(reflect(normalize(uvw - light.pos),
                                normal),
                        normalize(eye - uvw)),
                    55.));
}

// Function 2
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

// Function 3
float SelectComp(vec4 v4, float i)
{
	i = floor(i);
    return (i == 0.0) ? v4.w : 
    	   (i == 1.0) ? v4.z :
    	   (i == 2.0) ? v4.y :
    	   (i == 3.0) ? v4.x : 0.0;
}

// Function 4
vec4 getSelection() {
	return load(SELECTION);   
}

// Function 5
vec3 selectColor(vec4 q, vec3 eye, vec3 n) {
#if !defined MIRROR
  return defaultColor;
#else
  // The ShaderToy cubemaps are quite high contrast, which
  // tends to emphasise edge artefacts, so mute a little.
  vec3 color = texture(iChannel0,reflect(eye,n)*iMatrix).rgb;
  return min(vec3(0.75),sqrt(color));
#endif
}

// Function 6
bool select(inout vec4 O, vec2 U) {   // --- select block
    int N = int(iResolution.x)/S, t = iFrame;
    if (U==vec2(.5)) O.w = float(t)/float(N)/(iResolution.y/float(S)); // % of image scanned
    return ivec2(U)/S == ivec2(t%N,t/N);
}

// Function 7
vec3 highlight(in Ray ray, in vec3 n) {
    // sun
	vec3 sunDir = normalize(vec3(1,0.3,1));
	float sunDist = distance(sunDir, ray.dir)-0.00;
	return mix(vec3(10,10,8), vec3(0), smoothstep(0.0, 0.2, sunDist));
}

// Function 8
void process_text_select_location( int i, inout int N,
                                   inout vec4 params, inout uvec4 phrase,
                                   GameState gs )
{
	int n = ADDR_START_DATA_COUNT - 1;
    int index = i - N + 1;
    if( index >= 1 && index < n )
    {
        StartData start = st_load( iChannel0, ADDR_START_DATA + ivec2( index, 0 ) );
		vec3 nav = start.iparams.x == 3 && start.iparams.y < ADDR_SCENE_DATA_COUNT ?
	        sd_load( iChannel1, ivec2( 0, ADDR_B_SCENEDATA + ADDR_SCENE_DATA_SIZE * start.iparams.y ) ).navb.xyz :
	    	start.params.xyz * vec3( 1, 1, TRN_SCALE );
        vec3 r = nav2r( vec3( nav.xy, nav.z + g_data.radius ) );
        vec3 v = normalize( r - g_vehicle.localr ) * gs.camframe;
        v = round( 2047.5 * v + 2047.5 );
        params = vec4( v.x + v.y / 4096., v.z, 1, -12 );
      #if WORKAROUND_05_UVEC4
        phrase = uvec4( uint( 64 + index ) << 24u, 0u, 0u, 1u );
      #else
        phrase = uvec4( uint( 64 + index ) << 24u, 0, 0, 1 );
      #endif
    }
    N += n - 1;
}

// Function 9
bool root_selected(int i) {

    return texelFetch(iChannel0, ivec2(i+1,0), 0).w>0.5;
}

// Function 10
float highlight(float circle, vec2 pos, float radius)
{
    float h = smoothstep(0., radius, length(pos));
    h -= 1.-circle;
    return h*(.4+(sin(iTime)+1.)*.1);
}

// Function 11
void paintCursorSelectionInfo(inout vec4 finalColor, in vec3 bgColor, in uvec2 sCoord, in uvec2 sMouseCoord, in uvec2 wMouseCoord)
{
    vec4 mainColor = UI_BACKGROUND_COLOR;
    
    int layerIndex = int(readVar(VARIABLE_LOCATION_LAYER_INDEX).x);
    
    int selectedLayerIndex = layerIndex;
    
    int mode = int(readVar(VARIABLE_LOCATION_MODE).x);
    if (mode == 0)
    {
        while (selectedLayerIndex > 0)
        {
            if (readVoxel(ivec3(wMouseCoord, selectedLayerIndex)).type != VOXEL_TYPE_VOID)
                break;
            --selectedLayerIndex;
        }
    }
    
    
    Voxel voxel = readVoxel(ivec3(wMouseCoord, selectedLayerIndex));
    
    if (voxel.type != VOXEL_TYPE_VOID)
    {
        // Draw box
        vec2 halfInfoScale = 0.5 * UI_INFO_BOX_SCALE;
    	vec2 uuv = vec2(ivec2(sCoord) - ivec2(sMouseCoord)) / iResolution.y;
        float t = sdBox(uuv - vec2(halfInfoScale), vec2(halfInfoScale));
        float a = smoothstep(1.5, -1.5, t * iResolution.y);
        float ao = 1. - .4 * SATURATE(exp(-1.5*t * iResolution.y));
        
        finalColor.rgb *= ao;
        //bgColor = mix(bgColor, mainColor.rgb, mainColor.a);
        
        finalColor.rgb = mix(finalColor.rgb, mix(bgColor, vec3(1), UI_ALPHA), a);
        //finalColor.rgb = mix(finalColor.rgb, vec3(0.0), a);
        
        // Draw text info
        
        float scale = 1.;
        float gly = 1. / UI_INFO_BOX_SCALE.x * 15.;
        float boxPadding = 0.5;
        vec2 uv = uuv * gly - boxPadding;
        float px = gly / iResolution.y;

        float x = 100.;
        float cp = 0.;
        vec4 cur = vec4(0,0,0,.01);
        vec4 us = cur;
        float ital = 0.0;

        int lnr = int(floor(uv.y/2.));
        uv.y = mod(uv.y,2.0)-0.5;
        
        if (lnr >= 0 && lnr <= 3 && voxel.type != VOXEL_TYPE_VOID)
        {
            ITAL;

            DARKGREY;
            
            if (lnr == 3)
            {
                // TODO: Add quotation marks
                _TYPE _dotdot _ BLACK;

                if (voxel.type == VOXEL_TYPE_VOID)
                {
                    _Void;
                }
                else if (voxel.type == VOXEL_TYPE_STONE)
                {
                    _Stone;
                }
                else if (voxel.type == VOXEL_TYPE_REDSTONE_DUST)
                {
                    _RedstoneDust;
                }
                else if (voxel.type == VOXEL_TYPE_REDSTONE_TORCH)
                {
                    _RedstoneTorch;
                }
            }
            else if (lnr == 2)
            {
                _ENERGIZED _dotdot _;

                if (voxel.energy > 0u)
                {
                    ENERGIZED true_;
                }
                else
                {
                    UNENERGIZED false_;
                }
            }
            else if (lnr == 1)
            {
                _ENERGY _dotdot _;

                uint e = voxel.energy;

                cur.rgb = getEnergyColor(e);
                HEX(e)
                    }
            else if (lnr == 0)
            {
                _ADDRESS _dotdot _;

                int addr = getWorldVirtualToAddr(globalWorld, globalImage, ivec3(wMouseCoord, layerIndex));

                BLACK;
                HEX(uint(addr));
                //BLACK ITAL _0 x_;
            }

            vec3 clr = vec3(0.0);

            float weight = 0.04+cur.w*.05;//min(iTime*.02-.05,0.03);//+.03*length(sin(uv*6.+.3*iTime));//+0.02-0.06*cos(iTime*.4+1.);
            finalColor = mix(finalColor, vec4(us.rgb, 1.0), .8*smoothstep(weight+px, weight-px, x));
        }
        
        //bgColor = mix(mainColor.rgb, bgColor, 1.0-mainColor.a);
    }
    
/*
    float boxPadding = 4.;
    vec2 boxScale = vec2(scale * 5. * 1.5, scale * 5.);


    vec2 uv = (uu - boxPadding - vec2(0,4.)) / iResolution.y * iResolution.y / scale;
    float px = 1. / iResolution.y * iResolution.y / scale;

    float x = 100.;
    float cp = 0.;
    vec4 cur = vec4(0,0,0,.01);
    vec4 us = cur;
    float ital = 0.0;

    int lnr = int(floor(uv.y/2.));
    uv.y = mod(uv.y,2.0)-1.0;
	*/
    /*
    uint layerIndex = uint(readVar(layerIndexLocation).x);
    Voxel voxel = readVoxel(ivec3(wMouseCoord, layerIndex));

    if (lnr >= 0 && lnr < 4 && voxel.type != VOXEL_TYPE_VOID)
    {
        //ITAL;

        DARKGREY
            if (lnr == 4)
            {
                _COORDINATES _dotdot;
            }
        else if (lnr == 3)
        {
            // TODO: Add quotation marks
            _TYPE _dotdot _ BLACK;

            if (voxel.type == VOXEL_TYPE_VOID)
            {
                _Void;
            }
            else if (voxel.type == VOXEL_TYPE_STONE)
            {
                _Stone;
            }
            else if (voxel.type == VOXEL_TYPE_REDSTONE_DUST)
            {
                _RedstoneDust;
            }
            else if (voxel.type == VOXEL_TYPE_REDSTONE_TORCH)
            {
                _RedstoneTorch;
            }
        }
        else if (lnr == 2)
        {
            _ENERGIZED _dotdot _;

            if (voxel.energy > 0u)
            {
                ENERGIZED true_;
            }
            else
            {
                UNENERGIZED false_;
            }
        }
        else if (lnr == 1)
        {
            _ENERGY _dotdot _;

            uint e = voxel.energy;

            cur.rgb = getEnergyColor(e);
            HEX(e)
                }
        else if (lnr == 0)
        {
            _ADDRESS _dotdot _;

            int addr = getWorldVirtualToAddr(globalWorld, globalImage, ivec3(wMouseCoord, layerIndex));

            BLACK;
            HEX(uint(addr));
            //BLACK ITAL _0 x_;
        }

        vec3 clr = vec3(0.0);

        float weight = 0.05+cur.w*.02;//min(iTime*.02-.05,0.03);//+.03*length(sin(uv*6.+.3*iTime));//+0.02-0.06*cos(iTime*.4+1.);
        mainColor = mix(mainColor, vec4(us.rgb, 1.0), smoothstep(weight+px, weight-px, x));
    }

    if (voxel.type != VOXEL_TYPE_VOID)
    {
        float t1 = sdBox(coord - boxScale - boxPadding - vec2(0,4.), boxScale + boxPadding);
        float a = smoothstep(1., -1., t1);
        bgColor = mix(mainColor.rgb, bgColor, 1.0-mainColor.a);
        finalColor.rgb *= 1. - .2 * SATURATE(exp(-.6 * t1));
        finalColor.rgb = mix(finalColor.rgb, bgColor, a);
        //fragColor.rgb = mix(fragColor.rgb, mix(blurredColor, color, GUI_TRANSLUCENCY), smoothstep(1., -1., t1));
    }*/
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
void CalculateSelectedCells(in vec4 state, in vec4 mouse, inout vec4 selected0, inout vec4 selected1)
{
    if(mouse.w > 0.5 && mouse.z < 0.5 && selected0.z > 0.5)
    {
        for(float y=0.; y<yCells-0.5;++y) {
            for(float x=0.; x<xCells-0.5;++x){
                vec2 cellPos = GetCellPos(vec2(x, y));
                float d = dsCell(mouse.xy-cellPos.xy);
                if(d < 0.) 
                { 
                    selected0.xy = mix(vec2(x,y), selected0.xy, step(0.5, selected0.w));
                    selected1.xy = mix(vec2(x,y), selected1.xy, step(selected0.w, 0.5));
                    selected0.w += 1.;

                    if(selected0.w > 1.5)
                    {
                        vec2 dir = abs(selected1.xy-selected0.xy);
                        if(dir.x+dir.y > 1.5)
                        {
                            selected0 = selected1; selected0.z = 1.0;
                            selected1 = vec4(-100., -100., 1.0, 0.);
                            selected0.w = 1.0;
                        }
                    }
                    break;
                }
            }
        }
    }
}

// Function 14
vec3 widgetSelected(){    return texture(iChannel0, vec2(.5,2.5)/iResolution.xy).rgb;}

// Function 15
vec3 SelectColor(float i, float j)
{
    float a = mod((i * i + i +j), 6.0);
    if(a==0.0){return red;}
    if(a==1.0){return yellow;}
    if(a==2.0){return green;}
    if(a==3.0){return blue;}
    if(a==4.0){return orange;}
    return purple;
}

// Function 16
vec3 highlights(vec3 pixel, float thres)
{
	float val = (pixel.x + pixel.y + pixel.z) / 3.0;
	return pixel * smoothstep(thres - 0.1, thres + 0.1, val);
}

// Function 17
bool SpriteSelectIterate( vec4 vSpriteInfo, vec2 sheetPixel, inout vec4 vOutSpriteInfo, inout float fOutSpriteIndex, inout float fTestIndex )
{
    bool isInSprite = IsInSprite( sheetPixel, vSpriteInfo );
    
    if ( isInSprite )
    {
        vOutSpriteInfo = vSpriteInfo;
        fOutSpriteIndex = fTestIndex;
    }
    
    fTestIndex++;
    return isInSprite;
}

// Function 18
float getSelectorWave(void) { return .1+(sin(iTime*6.28)*.5+.5)*.3; }

// Function 19
vec4 drawSelectionBox(vec2 c) {
	vec4 o = vec4(0.);
    float d = max(abs(c.x), abs(c.y));
    if (d > 6. && d < 9.) {
        o.a = 1.;
        o.rgb = vec3(0.9);
        if (d < 7.) o.rgb -= 0.3;
        if (d > 8.) o.rgb -= 0.1;
    }
    return o;
}

// Function 20
vec2 selector(vec3 p)
{
    p.xy *= rot(0.5);
 	float t = sdRoundCone(p,.1,.15,.25);
    p.y = abs(p.y);
    return near(vec2(t,PLASTIC),screwHead(p - vec3(0,.9,0.4)));
}

// Function 21
vec3 widgetSelected()
{
    return texture(iChannel0, vec2(.5,2.5)/iResolution.xy).rgb;
}

// Function 22
void functionSelection()
{
  selection = clamp (0.0, maxSelection-1., floor(mpos.y * maxSelection));
//  if (mpos.x > 0.94)
  if (selection < 4.5) 
       functionGroup = int(selection);     // 0..4 
  else functionGroup = 4; // 0..4
}

// Function 23
void Player_SelectTarget( inout Entity playerEnt )
{
    // Select target entity (used to aim shots up / down)
    float fBiggestDot = cos( radians( 4.0 ) );
    
    float fClosest = FAR_CLIP;
    
    playerEnt.fTarget = float(ENTITY_NONE);
    
    vec2 vPlayerForwards = vec2( sin( playerEnt.fYaw ), cos( playerEnt.fYaw ) );
    
    for ( int iOtherEntityIndex=0; iOtherEntityIndex<int(ENTITY_MAX_COUNT); iOtherEntityIndex++ )
    {
        Entity otherEntity = Entity_Read( STATE_CHANNEL, iOtherEntityIndex );

        if ( Entity_IsPlayerTarget( otherEntity ) )
        {
            vec3 vToTarget = otherEntity.vPos - playerEnt.vPos;
            
            float fDist = length( vToTarget.xz );
            
            if ( fDist < fClosest ) 
            {
                vec2 vDirToTarget = normalize( vToTarget.xz );
                float fDot = dot( vDirToTarget, vPlayerForwards );
                
                if ( fDot > fBiggestDot )
                {
                    fClosest = fDist;
                    playerEnt.fTarget = float(iOtherEntityIndex);
                }
            }
        }        
    }    
}

// Function 24
float glassHighlight(vec2 p){
    float d = 0.;
    p.x -= .075;
    p.y -= .01;
    
	for(float i=-1.;i<=1.;i+=2.){
        vec2 pp = p;
        pp.x += i*.075 + sin(iTime*5.)*.05;
        pp *= mat2(cos( sin(.5) + vec4(0,33,11,0)));
        float dd = SS(abs(pp.x), .02);
        pp = p;    
        pp.x += i*.075;
        pp.y -= abs(pp.x*.75);
        pp.y *= 1.2;
        
        float l = length(pp);

        dd *= SS(l, .07);
        
        dd *= cos(iTime*5.);
        
        d+=dd;
    }
    
    return d;
}

// Function 25
vec3 irselect( vec4 a, bool b )
	{ return b ? a.www : a.xyz; }

// Function 26
void card_get_select(vec2 p) {
    float d = 1.;
    const vec2 card_pos = vec2(0., 0.35);
    const vec2 shift_pos = vec2(0.1, 0.);
    const vec2 sp_pos = vec2(-0.75, 0.35);
    const float angle_pos = 0.045;
    allData.this_selected_card = -1.;
    float anim_t = get_animstate(clamp(1. - (g_time - allData.card_add_anim), 0., 1.));
    float anim_t2zb = 1. - get_animstate(clamp((g_time - allData.ett - 6.5), 0., 1.));
    if ((allData.card_draw > 0.) || (anim_t > 0.) || (allData.player_etf) || ((anim_t2zb == 0.)&&(!allData.player_turn)&&(!allData.en_etf)))return;
    float anim_t2z = get_animstate(clamp((g_time - allData.ett)*1.5, 0., 1.));
    if (allData.player_turn)p.y += 0.08 * (1. - anim_t2z);
    else p.y += 0.08 * (anim_t2z);
    for (float i = float(min(0,iFrame)); i < allData.cards_player + 1.; i++) {
        if (i + 1. > allData.cards_player) {
            break;
        }
        if (i + 2. > allData.cards_player) {
            float tv = 0.5 - (allData.cards_player - (allData.cards_player / 2.) - i);
            d = card((p + card_pos) * MD(angle_pos * tv * (1. - anim_t)) - shift_pos * tv * (1. - anim_t) + vec2(sp_pos.x, 0.) * anim_t);
        } else {
            float ad = allData.cards_player - 1. * (anim_t);
            float tv = 0.5 - (ad - (ad / 2.) - i);
            d = card((p + card_pos) * MD(angle_pos * tv) - shift_pos * tv);
        }
        if ((d < 1.)) {
            allData.this_selected_card = i;
        }
    }
    return;
}

// Function 27
bool Feedback(vec4 i64)
{
    bool cin = false;
    
    //Taps
    cin = cin ^^ Bit(i64, 6.0);
    cin = cin ^^ Bit(i64, 13.0);
    cin = cin ^^ Bit(i64, 31.0);
    cin = cin ^^ Bit(i64, 52.0);
    cin = cin ^^ Bit(i64, 63.0);
    
    return cin;
}

