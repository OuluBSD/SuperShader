// Reusable Game Visuals Game Functions
// Automatically extracted from game/interactive-related shaders

// Function 1
float player(vec3 pos, float radius, inout vec3 col)
{
    col =vec3(1.0,0.0,0.0);
    return length(pos+PlayerPosition) - radius;
}

// Function 2
float PrintCharacter( in int char, in vec2 uv )
{
    uint idx = uint(char);
    vec2 charp = vec2( idx&0xFU, 0xFU-(idx>>4U) );

    if ( min(uv.x,uv.y) < .0 || max(uv.x-.5,uv.y-1.) > .0 )
        return 0.;
    uv.x += .25;

//    return step(textureLod(iChannel0, (uv+charp)/16., .0).w,.5);
//    return smoothstep(.53,.47,textureLod(iChannel1, (uv+charp)/16., .0).w );
//    return textureLod(iChannel0, (uv+charp)/16., .0).x;
    return 1.-textureLod(iChannel1, (uv+charp)/16., .0).w;
}

// Function 3
Shape character(vec3 c){
  Shape shape;
  shape.dist = 1000.; // Draw Distance
  shape.color = vec4(1.); // Initial Color
    
  vec3 h = c; // Head
  vec3 b = c; // Body
  vec3 rf = c; // Feet
  vec3 lf = c;
  vec3 a = c; // Arms 
  vec3 e = c; // Eyes
  vec3 bk = c; //Beak  
  vec3 ice = c; //iceberg
  
  
    //Head
    float head = sphere(h+vec3(0., -0.65, 0.),.75); //create head
    vec4 hColor = vec4(.75, .9, 1., 1.);

    
    //Body
    b.y *= .9;
    float body = sphere(b+vec3(0., 0.5, 0.),.99); // create body
    vec4 bColor = vec4(1., 1., 1., 1.);

    
 	//Feet
    lf.yz *= rot(sin(iTime*1.5)*.2);
    float feet1 = sphere(lf+vec3(.3, 1.5, -.2),.12); // create bottom of Feet
    float feet2 = sphere(lf+vec3(.5, 1.5,.3),.12); // create top of Feet
    float feet = fOpUnionRound(feet1, feet2, 1.); // merge to make one Feet
    
    rf.yz *= rot(cos(iTime*1.5)*.2);
    float rfeet1 = sphere(rf+vec3(-.3, 1.5,- .2),.12); // create bottom of Feet
    float rfeet2 = sphere(rf+vec3(-.5, 1.5,.3),.12); // create top of Feet
    float rightfeet = fOpUnionRound(rfeet1, rfeet2, 1.); // merge to make one Feet
    vec4 fColor = vec4(1.5, 1.2, -0.2, 1.);
    
    //Arms
    a.x = abs(a.x) -2.;
    
    a.y *= cos(sin(a.x*1.8))*.5;
    a.z *= cos(sin(a.x*2.5));
    a.zx *= cos(sin(60.))*.75;
    float arm = sphere(a + vec3(.85, 0.2, .25), .2);
    
    //Eyes
    e.x = abs(e.x) -1.;
    float eyes = sphere(e+vec3(.75, -.63, .75),.2);
    float pupil = sphere(e+vec3(.78, -.6, .6),.12);
    //Pupil Color
    vec4 eColor = vec4(-.5, -.5, -.5, 1.);
    vec4 pColor = vec4(2., 2., 2., 1.);
    
     //Beak
    bk -= vec3(0.,0.25,-.7); //moving the center to (0,1,0)
    //if bk was (0,1,0) it will now be (0,1,0)-(0,1,0)=(0,0,0) center
    
    pModPolar(bk.zy, 1.0); 
    bk -= vec3(0.,0.,0.); //moving center after modpolar making it happen 12 times
    bk.yz *= rot(radians(-90.0)); //rotate beak
    
    bk.z *=1.7;//compress beak
    float beak = fCone(bk, .2 ,.3);
    vec4 bkColor = vec4(1.2, 1.2, .1, 1.);

    
    //Iceberg
    float iceberg = fCylinder(ice+vec3(0., 1.7, 0.), 1.75, .075);
    vec4 iceColor = vec4(.1, .5, .7, 1.);
    
    
    
    //draw shapes to screen
    shape.dist = fOpUnionRound(head, body, .25); //merge head and body
    shape.dist = min(shape.dist, feet);
    shape.dist = min(shape.dist, rightfeet);
    
    shape.dist = fOpUnionRound(shape.dist, arm, .06);
    shape.dist = max(shape.dist, -eyes);
    shape.dist = min(shape.dist, pupil);
    shape.dist = min(shape.dist, beak);
    shape.dist = min(shape.dist, iceberg);
    
    shape.color = mix(hColor, bColor, mixColors(head, body, 0.5));
    shape.color = mix(shape.color, fColor, mixColors(feet, body, .3));
    shape.color = mix(shape.color, fColor, mixColors(rightfeet, body, 0.3));
    shape.color = mix(shape.color, bkColor, mixColors(beak, shape.dist, .01));
    shape.color = mix(shape.color, iceColor, mixColors(iceberg, body, 1.));
    shape.color = mix(shape.color, eColor, mixColors(eyes, shape.dist, 0.01));
    shape.color = mix(shape.color, pColor, mixColors(pupil, shape.dist, 0.01));

    

    
  return shape;
}

// Function 4
void spr_player_down(float f, float x, float y)
{
	float c = 0.;
	if (f == 0.) {
		if (y == 0.) c = (x < 8. ? 21504. : 21.); if (y == 1.) c = (x < 8. ? 21760. : 85.);
		if (y == 2.) c = (x < 8. ? 64800. : 2175.); if (y == 3.) c = (x < 8. ? 65312. : 2303.);
		if (y == 4.) c = (x < 8. ? 39840. : 2790.); if (y == 5.) c = (x < 8. ? 48032. : 2798.);
		if (y == 6.) c = (x < 8. ? 43648. : 3754.); if (y == 7.) c = (x < 8. ? 59712. : 3435.);
		if (y == 8.) c = (x < 8. ? 45052. : 16218.); if (y == 9.) c = (x < 8. ? 32751. : 15957.);
		if (y == 10.) c = (x < 8. ? 61355. : 14999.); if (y == 11.) c = (x < 8. ? 28655. : 11007.);
		if (y == 12.) c = (x < 8. ? 61423. : 2391.); if (y == 13.) c = (x < 8. ? 28671. : 85.);
		if (y == 14.) c = (x < 8. ? 15016. : 252.); if (y == 15.) c = (x < 8. ? 16128. : 0.);
		
		float s = SELECT(x,c);
		if (s == 1.) fragColor = RGB(128.,208.,16.);
		if (s == 2.) fragColor = RGB(255.,160.,68.);
		if (s == 3.) fragColor = RGB(228.,92.,16.);
	}
	if (f == 1.) {
		if (y == 0.) c = (x < 8. ? 21504. : 21.); if (y == 1.) c = (x < 8. ? 21760. : 85.);
		if (y == 2.) c = (x < 8. ? 64800. : 2175.); if (y == 3.) c = (x < 8. ? 65312. : 2303.);
		if (y == 4.) c = (x < 8. ? 39840. : 2790.); if (y == 5.) c = (x < 8. ? 48032. : 2798.);
		if (y == 6.) c = (x < 8. ? 43648. : 3754.); if (y == 7.) c = (x < 8. ? 59648. : 3947.);
		if (y == 8.) c = (x < 8. ? 49136. : 2394.); if (y == 9.) c = (x < 8. ? 65468. : 2389.);
		if (y == 10.) c = (x < 8. ? 48812. : 863.); if (y == 11.) c = (x < 8. ? 49084. : 509.);
		if (y == 12.) c = (x < 8. ? 49084. : 351.); if (y == 13.) c = (x < 8. ? 49148. : 213.);
		if (y == 14.) c = (x < 8. ? 10912. : 252.); if (y == 15.) c = (x < 8. ? 0. : 252.);
		
		float s = SELECT(x,c);
		if (s == 1.) fragColor = RGB(128.,208.,16.);
		if (s == 2.) fragColor = RGB(255.,160.,68.);
		if (s == 3.) fragColor = RGB(228.,92.,16.);
	}
	if (f == 2.) {
		if (y == 0.) c = 0.; if (y == 1.) c = (x < 8. ? 41280. : 42.);
		if (y == 2.) c = (x < 8. ? 43472. : 170.); if (y == 3.) c = (x < 8. ? 23252. : 677.);
		if (y == 4.) c = (x < 8. ? 22261. : 12949.); if (y == 5.) c = (x < 8. ? 60917. : 15963.);
		if (y == 6.) c = (x < 8. ? 56791. : 3703.); if (y == 7.) c = (x < 8. ? 32348. : 1021.);
		if (y == 8.) c = (x < 8. ? 31344. : 381.); if (y == 9.) c = (x < 8. ? 60096. : 1375.);
		if (y == 10.) c = (x < 8. ? 43264. : 1370.); if (y == 11.) c = (x < 8. ? 26112. : 2389.);
		if (y == 12.) c = (x < 8. ? 23040. : 2646.); if (y == 13.) c = (x < 8. ? 26944. : 6781.);
		if (y == 14.) c = (x < 8. ? 41296. : 22207.); if (y == 15.) c = (x < 8. ? 0. : 21823.);
		
		float s = SELECT(x,c);
		if (s == 1.) fragColor = RGB(228.,92.,16.);
		if (s == 2.) fragColor = RGB(128.,208.,16.);
		if (s == 3.) fragColor = RGB(255.,160.,68.);
	}
}

// Function 5
void PlayerHit( vec4 playerHitBox )
{
#ifndef GOD_MODE
    if ( gGameState.x == GAME_STATE_LEVEL && gGameState.y > UI_GAME_START_TIME )
    {
        gPlayerState.x 	= STATE_JUMP;
        gPlayerState.y 	= 0.0;        
        gPlayerState.z 	= 1.0;
        gPlayerState.w -= 1.0;    
        gExplosion 		= vec4( gPlayer.xy + vec2( 0.0, playerHitBox.z * 0.5 ), 0.0, 0.0 );
        gPlayer 		= vec4( gCamera.x + 32.0 * 2.0 + 24.0, PLAYER_SPAWN_HEIGHT, PLAYER_SPAWN_HEIGHT, 0.0 );
        gPlayerDir		= vec4( 1.0, 0.0, 0.0, PLAYER_IMMORTALITY_LEN );
        gPlayerWeapon 	= vec4( WEAPON_RIFLE, 0.0, RIFLE_FIRE_RATE, RIFLE_BULLET_NUM );

        if ( gPlayerState.w <= 0.0 )
        {
            gGameState.x 	= GAME_STATE_LEVEL_DIE;
            gGameState.y 	= 0.0;
            gPlayer			= vec4( 0.0, 1000000.0, 0.0, 0.0 );
            gPlayerState.x 	= STATE_FALL;
        }
    }
#endif
}

// Function 6
bool IsBlastThroughCharacter( int char )
{
	if ( char >= _AT && char <= _HASH ) return true;
    
    return false;
}

// Function 7
bool IsHoldCharacter( int char )
{    
    // The Held Graphics Character is only defined during the Graphics Mode.
    // It is then the most recent character with b6=1 in its character code,
    // providing that there has been no intervening change in either 
    // the Alphanumerics/Graphics or the Normal/Double Height modes.
    if ((char & (1 << 5)) != 0) return true;
    
    return false;
}

// Function 8
vec4 GameImage( vec2 vUV, vec2 vResolution )
{
    vec4 vResult = vec4(0.0);

    if ( any( lessThan( vUV, vec2(0.0) ) ) || any( greaterThanEqual( vUV, vec2(1,1) ) ) )
    {
        return vResult;
    }
    
    vec2 vHudPixel = vUV * vResolution.xy;
    vec2 vScenePixel = vUV * vResolution.xy;

    g_playerEnt = Entity_Read( STATE_CHANNEL, 0 );
    Sector playerSector = Map_ReadSector( MAP_CHANNEL, g_playerEnt.iSectorId );
    
	g_gameState = GameState_Read( STATE_CHANNEL );
    
    float fHudFade = 0.0;
    float fGameFade = 10.0;
    
    if ( g_gameState.iMainState == MAIN_GAME_STATE_GAME_RUNNING )
    {
        fHudFade = 0.01 + g_gameState.fStateTimer;
        fGameFade = 0.0;
    }
    if ( g_gameState.iMainState == MAIN_GAME_STATE_WIN )
    {
        fGameFade = 0.01 + g_gameState.fStateTimer;
        fHudFade = 0.0;
    }
    
    VWipe( vHudPixel, fHudFade, vResolution );
    VWipe( vScenePixel, fGameFade, vResolution );       
    
    
    vec3 vRenderImage;
    
#ifdef ALLOW_MAP    
    if ( g_gameState.fMap > 0.0 )
    {
#ifdef HIRES_MAP        
        vRenderImage = DrawMap( vScenePixel, vResolution );
#else // HIRES_MAP       
        vec2 vScenePixelCoord = floor(vScenePixel);

        float fScale = 10.0;
        vec2 vPixelWorldPos = (vScenePixelCoord - vec2(160,100)) * fScale + g_playerEnt.vPos.xz;
        
        vec2 vMapUV = (vPixelWorldPos - vec2(1056, -3616)) / 10.0 + vec2(200, 150);
        vMapUV += vec2(0, 32.0);
        
        vRenderImage = texture( iChannel1, (floor(vMapUV) + 0.5) / iChannelResolution[1].xy ).rgb;
#endif // HIRES_MAP
    }        
    else
#endif // ALLOW_MAP        
    {
 		vRenderImage = SampleScene( vScenePixel, vResolution, playerSector.fLightLevel ).rgb;        
    }
    
    if ( vScenePixel.y <= 32.0 )
    {
        vec4 vHudText = GetHudText( vScenePixel, g_playerEnt.fHealth, g_playerEnt.fArmor );

		vec2 vNoiseScale = vec2(500.0, 300.0);        
        float fNoisePer = 0.8;
        if ( vHudText.a > 0.0 )
        {
            vNoiseScale = vec2(600.0); 
            fNoisePer = 0.5;
        }

        float fNoise = fbm( vUV * vNoiseScale, fNoisePer );
        fNoise = fNoise * 0.5 + 0.5;
        
        if ( vHudText.a > 0.0 )
        {
            vRenderImage = vHudText.rgb * fNoise;
        }
    	else
        {
            vRenderImage = vec3(fNoise * fNoise * 0.65 );
        }    
        
#ifdef FULL_HUD
        // Main relief
        vRenderImage *= Relief( vScenePixel, vec2(0, 0), vec2(46, 31));
        vRenderImage *= Relief( vScenePixel, vec2(48, 0), vec2(104, 31));
        vRenderImage *= Relief( vScenePixel, vec2(106, 0), vec2(142, 31));
        vRenderImage *= Relief( vScenePixel, vec2(178, 0), vec2(235, 31));
        vRenderImage *= Relief( vScenePixel, vec2(249, 0), vec2(319, 31));
        
        // weapon avail
        vRenderImage *= Relief( vScenePixel, vec2(107, 200 - 179), vec2(117, 200 - 171));
        vRenderImage *= Relief( vScenePixel, vec2(119, 200 - 179), vec2(129, 200 - 171));
        vRenderImage *= Relief( vScenePixel, vec2(131, 200 - 179), vec2(141, 200 - 171));
        
        vRenderImage *= Relief( vScenePixel, vec2(107, 200 - 189), vec2(117, 200 - 181));
        vRenderImage *= Relief( vScenePixel, vec2(119, 200 - 189), vec2(129, 200 - 181));
        vRenderImage *= Relief( vScenePixel, vec2(131, 200 - 189), vec2(141, 200 - 181));

        // decoration
        vRenderImage *= Relief( vScenePixel, vec2(237, 200 - 179), vec2(247, 200 - 171));
        vRenderImage *= Relief( vScenePixel, vec2(237, 200 - 189), vec2(247, 200 - 181));
        vRenderImage *= Relief( vScenePixel, vec2(237, 200 - 199), vec2(247, 200 - 191));
        
        vRenderImage *= Relief( vScenePixel, vec2(143, 0), vec2(177, 31));
        
        if ( all( greaterThanEqual( vScenePixel, vec2(144,1) ) ) &&
            all( lessThan( vScenePixel, vec2(177,31) ) ) )
        {
            vRenderImage = vec3(0.0);
        }
#endif // FULL_HUD
            
#ifdef FULL_HUD   
        
        PrintHudMessage( vec2(vScenePixel.x, (vResolution.y - 1.) - (vScenePixel.y + 189.)), MESSAGE_HUD_TEXT, vRenderImage );
        
        PrintState printState;
        Print_Init( printState, vec2(vScenePixel.x, (vResolution.y - 1.) - vScenePixel.y) );        

        
        // HUD text AMMO, HEALTH, ARMS, ARMOR
/*
        Print_Color( printState, vec3(.9 ) );
        vec3 vChar = _SPACE_;
        for ( int i=0; i<24; i++)
        {
            vChar = GetHudTextChar( float(i) );
            if ( Print_Test( printState, vChar, 0.0 ) )
            {
                break;
            }
            if ( vChar.z == 0. )
                break;
        }

        Print_HudChar( printState, vRenderImage, vChar );        
*/
        // Arms numbers
        Print_Color( printState, vec3(.8,.8,0 ) );        
        Print_MoveTo( printState, vec2(109,170) );
        Print_Char( printState, vRenderImage, _2_ );

        if( g_playerEnt.fHaveShotgun <= 0.0 )
        {
			Print_Color( printState, vec3(.25 ) );        
        }
        
        Print_MoveTo( printState, vec2(120,170) );
        Print_Char( printState, vRenderImage, _3_ );
        Print_Color( printState, vec3(.25 ) );        
        Print_MoveTo( printState, vec2(132,170) );
        Print_Char( printState, vRenderImage, _4_ );

        Print_MoveTo( printState, vec2(109,179) );
        Print_Char( printState, vRenderImage, _5_ );
        Print_MoveTo( printState, vec2(120,179) );
        Print_Char( printState, vRenderImage, _6_ );
        Print_MoveTo( printState, vec2(132,179) );
        Print_Char( printState, vRenderImage, _7_ );
#endif // FULL_HUD        
    }    
    
	float fEffectAmount = clamp( abs(g_gameState.fHudFx), 0.0, 1.0 );
            
    if (g_gameState.fHudFx > 0.0) 
    {
        vRenderImage.rgb = mix( vRenderImage.rgb, vec3( 0.5, 1, 0.6), fEffectAmount * 0.75 );
    }

    if (g_gameState.fHudFx < 0.0) 
    {
        vRenderImage.rgb = mix( vRenderImage.rgb, vec3( 1, 0, 0), fEffectAmount * 0.75 );
    }
    
#ifdef FULL_HUD    
#ifdef HUD_MESSAGES
    if ( g_gameState.fMessageTimer > 0.0 )
    {
        if (g_gameState.iMessage >= 0 )
        {
        	PrintHudMessage( vec2(vScenePixel.x, (vResolution.y) - vScenePixel.y), g_gameState.iMessage, vRenderImage );
        }
    }
#endif // HUD_MESSAGES    
#endif // FULL_HUD    
    
    
    vec3 vFrontendImage = vec3(0.0);
    
    if ( vHudPixel.y > 0.0 )
    {
        vFrontendImage = Tex( vHudPixel );
        vec2 vHudTextCoord = vec2(vHudPixel.x, (vResolution.y) - vHudPixel.y);

        if ( g_gameState.iMainState == MAIN_GAME_STATE_WIN )
        {
            float fScale = 0.5;
            vec2 vPos = vec2(58,8);

            PrintHudMessage( (vHudTextCoord * fScale - vPos ), MESSAGE_HANGAR, vFrontendImage );        
            vPos.y += 10.0;
            vPos.x = 56.0;
            PrintHudMessage( (vHudTextCoord * fScale - vPos ), MESSAGE_FINISHED, vFrontendImage );        

        }

        /*
        if ( g_gameState.fMainState == MAIN_GAME_STATE_SKILL_SELECT
           || g_gameState.fMainState == MAIN_GAME_STATE_INIT_LEVEL
           || g_gameState.fMainState == MAIN_GAME_STATE_GAME_RUNNING )
        {            
            float fScale = 0.8;
            vec2 vPos = vec2(32,32);

            PrintHudMessage( (vHudTextCoord * fScale - vPos ), MESSAGE_CHOOSE_SKILL, vFrontendImage );        

            vPos.x += 32.0;
            vPos.y += 24.0;

            PrintHudMessage( (vHudTextCoord * fScale - vPos ), MESSAGE_SKILL_1, vFrontendImage );        
            vPos.y += 16.0;
            PrintHudMessage( (vHudTextCoord * fScale - vPos ), MESSAGE_SKILL_2, vFrontendImage );        
            vPos.y += 16.0;
            PrintHudMessage( (vHudTextCoord * fScale - vPos ), MESSAGE_SKILL_3, vFrontendImage );        
            PrintHudMessage( (vHudTextCoord * fScale - vPos + vec2(16.0, 0) ), MESSAGE_SELECT, vFrontendImage );        
            vPos.y += 16.0;
            PrintHudMessage( (vHudTextCoord * fScale - vPos ), MESSAGE_SKILL_4, vFrontendImage );        
            vPos.y += 16.0;
            PrintHudMessage( (vHudTextCoord * fScale - vPos ), MESSAGE_SKILL_5, vFrontendImage );        
            vPos.y += 16.0;       		        
        }
		*/
    }
    
    
    vec2 vHudUV = vHudPixel / vResolution;
    vec2 vSceneUV = vScenePixel / vResolution;
    if ( fHudFade > fGameFade )
    {
        vResult.rgb = vRenderImage;

        if ( vHudUV.y < 1.0 )
        {
            vResult.rgb = vFrontendImage;    
        }
    }
    else
    {
        vResult.rgb = vFrontendImage;

        if ( vSceneUV.y < 1.0 )
        {
	        vResult.rgb = vRenderImage;
        }
    }
    
    if ( g_gameState.iMainState == MAIN_GAME_STATE_BOOT  ) 
    {
        vResult.rgb = vec3( 0, 0, 0 );
    }

    //vResult *= 0.5 + 0.5 * mod(mod(floor(vScenePixel.x), 2.0) + mod(floor(vScenePixel.y), 2.0), 2.0);
    
	return vResult;    
}

// Function 9
vec4 RenderGame( vec2 _uv )
{
    //ivec2 iuv = ivec2(uv); <- bad, it means art style changes with resolution => do it all in floats
    vec2 uv = _uv/iResolution.xy; // ignore aspect ratio! The gameis stylised to look good around 16:9.
    ivec2 mu = ivec2(uv*vec2(MAP_SIZE));

    vec4 playerData = Get(PLAYER_DATA); // position, magic level, health
    vec2 playerHeading = sign(playerData.xy);
    playerData.xy = abs(playerData.xy);
    playerData.xy = floor(playerData.xy+.5);
    ivec4 playerStats = ivec4(Get(PLAYER_STATS)); // max magic level, max health, magic regen rate, health regen rate
    vec4 attackData = Get(ATTACK_DATA);
    ivec4 inventoryData = ivec4(Get(INVENTORY_DATA)); // bitmask of collectibles, [bitmask of equipment, number of potions, number of lives]

    
    // map
    vec4 o = Get(mu);
    
    vec2 groundTexel = 1./vec2(4.,2.);
    ivec2 groundUV = ivec2(((uv-.5)*vec2(MAP_SIZE)+playerData.xy)/groundTexel);
    float dither = Hash(groundUV);
    
    float gridDither = float( ((groundUV.x&1)*2+(groundUV.y&1)*3)&3 )/3.;

    
    // palette
    const vec4 ground1a[]		= vec4[]( vec4(.8,.7,.5,1), vec4(0,.4,.1,1), vec4(1), vec4(.2) );
    const vec4 ground2a[]		= vec4[]( vec4(.68,.52,.38,1), vec4(.1,.5,.0,1), vec4(.8), vec4(.3) );
    const vec4 shortgrassa[]	= vec4[]( vec4(.5,.6,.3,1), vec4(.5,.6,.3,1), vec4(.4,.3,.2,1), vec4(.6) );
    const vec4 grassa[]		= vec4[]( vec4(.5,.9,.3,1), vec4(.5,.5,.1,1), vec4(.4,.4,.2,1), vec4(.5,.2,.3,1) );

    // blend the colours
    ivec2 muv = mu+ivec2(playerData.xy);
    
/*	// noisy blend - cool but kinda ugly
	ivec2 bary = (muv+ivec2((vec2(Hash(muv),Hash(muv+40))-.5)*100.)) / ivec2(512,512);
    int biome = bary.x+bary.y*2;
    vec4 ground1 = ground1a[biome];
    vec4 ground2 = ground2a[biome];
	vec4 shortgrasscol = shortgrassa[biome];
    vec4 grasscol = grassa[biome];*/
    
    // smoother blend
    vec2 biomeBlend = (vec2(muv)*(60./61.) + (11./61.)*vec2(-muv.y,muv.x)-vec2(512,620))/vec2(200,100)+.5;
    biomeBlend = smoothstep( .0, 1., biomeBlend );
    
    float quantization = 8.;
    biomeBlend = floor(biomeBlend*quantization+vec2(Hash(muv),Hash(muv+40)))/quantization;
    vec4 ground1 = mix( mix( ground1a[0], ground1a[1], biomeBlend.x ), mix( ground1a[2], ground1a[3], biomeBlend.x ), biomeBlend.y );
    vec4 ground2 = mix( mix( ground2a[0], ground2a[1], biomeBlend.x ), mix( ground2a[2], ground2a[3], biomeBlend.x ), biomeBlend.y );
	vec4 shortgrasscol = mix( mix( shortgrassa[0], shortgrassa[1], biomeBlend.x ), mix( shortgrassa[2], shortgrassa[3], biomeBlend.x ), biomeBlend.y );
    vec4 grasscol = mix( mix( grassa[0], grassa[1], biomeBlend.x ), mix( grassa[2], grassa[3], biomeBlend.x ), biomeBlend.y );
    
    if ( o.z < .07 )
    {
        // WATER
        
        // create height
		float dy = 1.;
        vec4 t = Get(ivec2(uv*vec2(MAP_SIZE)+vec2(0,dy))); // vertical side
        o = ground1*.7;
        
        if ( t.z < .07 )
        {
            // water surface
        	t = Get(ivec2(uv*vec2(MAP_SIZE)+vec2(0,dy*2.))); // reflection of vertical side
        	o = mix( vec4(.2,.4,.7,1), mix( vec4(.0), ground1, .3 ), step(.07,t.z)*.4 );
            
            // ripples
            o = mix( o, vec4(.4,.6,.8,1), max(.0,fract((uv.y+playerData.y/float(MAP_SIZE.y))*20.-iTime*.3+2.*Hash(ivec2(mu.x+int(playerData.x),0)))*4.-3.) );
        }
    }
    else
    {
        // GROUND
        // read map with bilinear filtering
        vec3 a = texture(iChannel2,vec2(groundUV)*groundTexel/iChannelResolution[2].xy).xyz;
        
        // earth
        float earth = (dot(a,vec3(1,0,0))-.1)*5.;
        o = mix( ground1, ground2, step(gridDither,earth) );

	    float dither2 = Hash(groundUV+ivec2(0,-1));
        float blade = min(dither,dither2);

        float shortgrass = dot(a,vec3(0,0,1))-.1;
        o = mix( o, shortgrasscol*(1.-.5*(shortgrass-blade)/max(.001,shortgrass)), step(blade,shortgrass) );
        
	    float dither3 = Hash(groundUV+ivec2(0,-2));
        blade = min(blade,dither3);
        
        float grass = dot(a,vec3(0,1,0))*.5-.1;
        
	    float shadow = Hash(groundUV+ivec2(-1,0));
        o = mix( o, o*.7, step(shadow,grass) );
        
        o = mix( o, grasscol*(1.-.5*(grass-blade)/max(.001,grass)), step(blade,grass) );
    }
    
    
    // player character
    const vec2 pdim = vec2(.008,.03);
    vec2 puv = uv-vec2(.5,.5);
    vec2 p;
    
    // player shadow
    p = puv-vec2(pdim.x,0);
    p.y *= 2.;
    o *= .6+.4*smoothstep( .8, 1.5, length(p)/(pdim.x*1.5) );

    // enemy shadows
    for ( int i=0; i < MAX_ENEMIES; i++ )
    {
        vec4 enemyData = Get(ivec2(i,ENEMY_DATA_Y));
        enemyData.xy = floor(enemyData.xy+.5);
        vec2 euv = uv-((enemyData.xy-playerData.xy)/vec2(MAP_SIZE)+.5);
        if ( enemyData.w > 0.
            && length(euv) < .05 )
        {
            //o = vec4(1,1.-enemyData.w,.2,1);
            vec2 edim = vec2(.01,.017) * enemyData.z;
            float eh = .03;

            // shadow
            p = euv-vec2(edim.x,0);
            p.y *= 2.;
            o *= .6+.4*smoothstep( .8, 1.5, length(p)/edim.x );
        }
    }


    // collectibles & their shadows (can draw together because they're never close enough to have shadow sorting errors)
    for( int i=0; i < collectibles.length(); i++ )
    {
        vec2 iuv = uv-((vec2(collectibles[i].xy)-playerData.xy)/vec2(MAP_SIZE)+.5);
        iuv.y /= 2.;//iResolution.x/iResolution.y;
        if ( (inventoryData.x & (1<<i)) == 0 )
        {
            // shadow
            float idim = .01;
            p = iuv-vec2(idim,0);
            p.y *= 4.;
            o *= .6+.4*smoothstep( .7, 1.4, length(p)/idim );

            // item
            iuv.y -= idim;
            float r = dot(abs(iuv),vec2(1));
            if ( r < idim )
            {
                o = vec4(float(iFrame%9+3*int(-sign(iuv.x)))/8.,float(iFrame%11+5*int(sign(iuv.y)))/10.,1,1);
            }
        }
    }
    
    
    // death sequence
    if ( playerData.w < .0 )
    {
        o = vec4(dot(o,vec4(.2126,.7152,.0722,0)))*(vec4(.3)+vec4(.3,-.2,-.2,0)*(.5+.5*cos(playerData.w*6.283/30.)));
    }
    

    // player legs
    vec2 pldim = vec2(.003,.015);
    vec2 gait = vec2(.005,.004);
    vec2 walk = gait*sin(playerData.xy*6.283/vec2(12,8));
    p = puv-vec2(0,pldim.y+gait.y*.5);
    float ls = smoothstep(.02,.005,puv.y);
    if ( abs(p.x-walk.x) < pldim.x && abs(p.y-walk.y) < pldim.y )
        o = vec4(.1,.5,1,1) * (.5+.5*ls*(.8+.2*walk.y/gait.y));
    if ( abs(p.x+walk.x) < pldim.x && abs(p.y+walk.y) < pldim.y )
        o = vec4(.1,.5,1,1) * (.5+.5*ls*(.7+.3*walk.y/gait.y));
    
    // face behind
    vec2 pfdim = vec2(.006,.01);
    if ( playerHeading.y > .0 )
    {
        p = puv - vec2(0,pdim.y*2.-pfdim.y+.002) - playerHeading*vec2(.006,0);
        if ( length(p/pfdim) < 1. )
        {
            o = vec4(.8,.6,.5,1);
            
            if ( length(p/pfdim-vec2(playerHeading.x*.1+.35,-.2)) < 1. )
                o *= .6;
        }
    }
    
    // body
    vec2 pbdim = pdim*vec2(1,.75);
    p = puv-vec2(0,pdim.y*2.-pbdim.y);
//    if ( abs(p.x) < pbdim.x && abs(p.y) < pbdim.y )
    if ( p.y > -pbdim.y && length( vec2(p.x,max(.0,p.y))/pbdim ) < 1. )
        o = vec4(0,.125,1,1) * mix(1.,.6,(p.x/pbdim.x)*.5+.5);
    
    // face in front
    if ( playerHeading.y < .0 )
    {
        p = puv - vec2(0,pdim.y*2.-pfdim.y-.004) - playerHeading*vec2(.006,0);
        if ( length(p/pfdim) < 1. )
        {
            o = vec4(.8,.6,.5,1);
            
            if ( length(p/pfdim-vec2(-playerHeading.x*.1-.35,.2)) > 1. )
                o *= .6;
            
            // eyes
            if ( Hash(ivec2(iFrame/4,0)) > .005 )
            {
                vec2 pe = p;
                pe.x = abs(pe.x-.001*playerHeading.x);
                pe.y /= 2.;
                if ( length(pe-vec2(.0025,0)) < .0015 )
                {
                    o = vec4(0);//.5,1,1,1);
                }
                if ( length(pe-vec2(.0025,.001)) < .0005 )
                {
                    o = vec4(1);
                }
            }
        }
    }

    // hat
    p = puv - vec2(0,pdim.y*2.-pfdim.y-.001) - playerHeading*vec2(.006,.003);
    if ( p.y > .007 && ( ( p.y < .01 && abs(p.x) < .01 )|| -abs(p.x)-p.y*.2 > -.008 ) )
    {
        o = abs(p.y-.014)<.003 ? vec4(.5,.8,1,1) : vec4(0,.1,.8,1);
        o *= 1.-.4*(p.x-p.y*.2+.005)/.01;
    }

    
    // enemies (hovering)
    for ( int i=0; i < MAX_ENEMIES; i++ )
    {
        vec2 edim = vec2(.01,.017);
        float eh = .03;

        vec4 enemyData = Get(ivec2(i,ENEMY_DATA_Y));
        enemyData.xy = floor(enemyData.xy+.5);
        vec2 euv = uv-((enemyData.xy-playerData.xy)/vec2(MAP_SIZE)+.5);
        euv.y -= eh;
        
        edim *= enemyData.z;
        
        if ( enemyData.w > 0.
            && length(euv) < .05*enemyData.z )
        {
            // health: 1->yellow, 2->orange, 3->red, 4->purple, 5->blue
            // is it quicker to just do an array lookup
            vec4 bodyCol = clamp( vec4(2.5,1.2,-1.5,1) + vec4(-.5,-.4,.5,0)*enemyData.w, .0, 1. );
            
            // body
            p = euv - vec2(0);
            if ( length(p/edim) < 1. )
            {
                // eye
	            float t = length(p/edim-vec2(0,-.2))*sqrt(enemyData.z); // smaller eye on bigger enemies
                o = (t > .5 || Hash(ivec2(i,iFrame/8)) < .01) ? bodyCol : t > .2 ? vec4(1) : vec4(0);
                
                // shading
                if ( length(p/edim-vec2(-.2,.2)) > .8 )
                    o *= .8;
            }
            
            // wings
            euv.x = abs(euv.x); // just make one
            p = euv/edim - vec2(.7,.7);
            if ( ((i+iFrame)&4) == 0 ) p = p.yx; // "rotate" wings!
            if ( p.x > .0 && p.y > .0 && p.x < 1. && p.y < .3 )
                o = vec4(.3,.1,.5,1);
        }
    }

    
    // magic effects
    if ( attackData.x < 15. )
    {
        float s = ShootTest((uv-.5-vec2(0,pdim.y))*vec2(MAP_SIZE),attackData.zw);
        if ( s < .0 )
        {
            o += vec4(.5,1.-attackData.x/5.,1,0)*max(.0,-s+sin(length(uv)*30.-2.*float(iFrame))-1.);
        }
    }

    if ( attackData.y < 10. )
    {
        vec2 buv = uv-vec2(.5);
        if ( buv.y > .0 ) buv.y *= 9./16.;
        float l = length(buv*vec2(MAP_SIZE))/(20.-attackData.y);
        if ( l < 1. ) o = mix(vec4(0,-1,-3,1),vec4(10,4,1,1),1.-attackData.y/10.);
    }
    
    
    return o;
}

// Function 10
vec3 characterPosition( float time, float x ) {
    time *= .11; // lateral movement is slower...
    float o = .2;
    x += o * cos( time );
    return vec3( x, .0, o * sin( time ) );
}

// Function 11
float character(int n, vec2 p)
{
	p = floor(p*vec2(4.0, -4.0) + 2.5);
    if (clamp(p.x, 0.0, 4.0) == p.x)
	{
        if (clamp(p.y, 0.0, 4.0) == p.y)	
		{
        	int a = int(round(p.x) + 5.0 * round(p.y));
			if (((n >> a) & 1) == 1) return 1.0;
		}	
    }
	return 0.0;
}

// Function 12
vec4 renderPlayers(vec3 cam, vec2 screenCoord) {
    vec2 worldCoord = camScreen2World(cam, screenCoord);
    
    // Player is a 10px at furthest possible zoom and gets bigger when gets closer.
    float radiusPx = 10.0 * max(cam.z, CAMERA_ZOOM_MIN) / CAMERA_ZOOM_MIN;
    const float radiusAaPx = 1.0;
    
    vec4 color = vec4(0.0);
    float weight = 0.0;
    for(int i = 0; i < N_PLAYERS; i++) {
    	float distPx = camWorld2Px(cam, length(worldCoord - playersPos[i]));
        
        float playerWeight = 1.0 - smoothstep(radiusPx, radiusPx + radiusAaPx, distPx);
        vec4 playerColor = vec4(playersColor[i], playerWeight);
        aggregateColorSiblings(color, weight, playerColor);
  	}    
    return color;
}

// Function 13
float GameSounds( float time )
{
    // play sounds a bit earlier
    time += 0.1;

    float ret = 0.0;
    
    float marioBigJump1 = 27.1;
    float marioBigJump2 = 29.75;
    float marioBigJump3 = 35.05;    
    
    
    // Jump sounds
    float jumpTime = time - 38.7;
    if ( jumpTime <= 0.0 ) { jumpTime = time - marioBigJump3 - 1.2 - 0.75; }    
    if ( jumpTime <= 0.0 ) { jumpTime = time - marioBigJump3 - 1.2; }
    if ( jumpTime <= 0.0 ) { jumpTime = time - marioBigJump3; }
    if ( jumpTime <= 0.0 ) { jumpTime = time - 34.15; }    
    if ( jumpTime <= 0.0 ) { jumpTime = time - 33.7; }
    if ( jumpTime <= 0.0 ) { jumpTime = time - 32.3; }    
    if ( jumpTime <= 0.0 ) { jumpTime = time - marioBigJump2 - 1.0; }    
    if ( jumpTime <= 0.0 ) { jumpTime = time - marioBigJump2; }    
    if ( jumpTime <= 0.0 ) { jumpTime = time - marioBigJump1 - 1.0; }    
    if ( jumpTime <= 0.0 ) { jumpTime = time - marioBigJump1; }    
    if ( jumpTime <= 0.0 ) { jumpTime = time - 25.75; }    
	if ( jumpTime <= 0.0 ) { jumpTime = time - 24.7; }        
    if ( jumpTime <= 0.0 ) { jumpTime = time - 23.0; } 
    if ( jumpTime <= 0.0 ) { jumpTime = time - 21.7; }    
    if ( jumpTime <= 0.0 ) { jumpTime = time - 19.65; }   
    if ( jumpTime <= 0.0 ) { jumpTime = time - 18.7; }    
    if ( jumpTime <= 0.0 ) { jumpTime = time - 15.1; } 
    if ( jumpTime <= 0.0 ) { jumpTime = time - 13.62; }    
    if ( jumpTime <= 0.0 ) { jumpTime = time - 11.05; }
    if ( jumpTime <= 0.0 ) { jumpTime = time - 9.0; }
    if ( jumpTime <= 0.0 ) { jumpTime = time - 7.8; }
    if ( jumpTime <= 0.0 ) { jumpTime = time - 6.05; }
    if ( jumpTime <= 0.0 ) { jumpTime = time - 5.0; }
    if ( jumpTime <= 0.0 ) { jumpTime = time - 4.2; }
	ret += Jump( jumpTime );                           

    // block bump sounds
    float bumpTime = time - 33.9;
    if ( bumpTime <= 0.0 ) { bumpTime = time - 22.4; }
    if ( bumpTime <= 0.0 ) { bumpTime = time - 15.4; }
    if ( bumpTime <= 0.0 ) { bumpTime = time - 5.3; }
    ret += Bump( bumpTime );
    
    // coin sounds
    float coinTime = time - 33.9;
    if ( coinTime <= 0.0 ) { coinTime = time - 22.4; }
    if ( coinTime <= 0.0 ) { coinTime = time - 5.4; }    
    ret += Coin( coinTime );    

    float stompTime = time - 26.3;
    if ( stompTime <= 0.0 ) { stompTime = time - 25.3; }
    if ( stompTime <= 0.0 ) { stompTime = time - 23.5; }    
    if ( stompTime <= 0.0 ) { stompTime = time - 20.29; }    
    if ( stompTime <= 0.0 ) { stompTime = time - 10.3; }    
    ret += Stomp( stompTime );
    
	ret += PowerUp( time - 17.0 );    

    ret += DownTheFlagpole( time - 38.95 );    
    
    return ret;
}

// Function 14
AppState updateGame( AppState s, float isDemo )
{
    if ( isDemo > 0.0 )
    {
        s.timeAccumulated += 5.5 * iTimeDelta;
    	s.playerPos.y = 32.5 * s.timeAccumulated;
    }
    else
    {
        float playerCellID = floor( s.playerPos.y );
        s.paceScale = saturate( ( playerCellID - 60.0) / 400.0);
        float timeMultiplier = mix( 0.85, 3.0, pow( s.paceScale, 2.0 ) );

        s.timeAccumulated += timeMultiplier * iTimeDelta;
        s.playerPos.y = 6.0 * s.timeAccumulated;
    }    
    
    float playerCellID = floor( s.playerPos.y );

    if ( isDemo > 0.0 )
    {           
        float cellOffset = 2.0;
        float nextPlayerCellID = playerCellID + cellOffset;

        float nextCellCoinRND = hash11( nextPlayerCellID + s.seed ); // skip rnd obstacle every second cell to make room for driving
        nextCellCoinRND *= mix( 2.0, -2.0, step( mod( nextPlayerCellID, 5.0 ), 2.5 ) ); // gaps in coin placing: 2 gaps, 2 coins
        nextCellCoinRND = mix( nextCellCoinRND, -2.0, step( nextPlayerCellID, 6.0 ) ); // head start
        float nextCellCoinCol = floor( 4.0 * nextCellCoinRND );

        // OBSTACLE
        float nextCellObsRND = hash11( 200.0 * nextPlayerCellID + s.seed );
        nextCellObsRND *= mix( 2.0, -2.0, step( mod( nextPlayerCellID, 4.0 ), 2.5 ) );
        nextCellObsRND = mix( nextCellObsRND, -2.0, step( nextPlayerCellID, 8.0 ) ); // head start
        float nextCellObsCol = floor( 3.0 * nextCellObsRND );
        
        float inputObs = 0.1;                
        if ( nextCellObsCol > -0.6 )
        {
            nextCellCoinCol -= 0.6; // pos fix
        	float toObs = nextCellObsCol - s.playerPos.x;
        
            if ( nextCellObsCol == 1.0 )
                inputObs = hash11( nextPlayerCellID + s.seed );
            
            if ( nextCellObsCol < 1.0 )
                inputObs = 2.0;

            if ( nextCellObsCol > 1.0 )
                inputObs = -2.0;
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

// Function 15
void gameSetup( int level, inout vec4 fragColor, in ivec2 coord ) {
    StoreVec4( ivec2(0,32 ), ivec4(4,1,3,0), fragColor, coord );
    StoreVec4( ivec2(1,32 ), ivec4(0,0,60,0), fragColor, coord );
    StoreVec4( ivec2(2,32 ), ivec4(0,0,0,0), fragColor, coord );
    StoreVec4( ivec2(3,32 ), ivec4(0), fragColor, coord );
}

// Function 16
void handleGameOver( inout vec4 buff, in vec2 fc, in vec2 keys, inout vec4 fsrt )
{
    float pchange = readTexel(DATA_BUFFER, txPCHANGE).r;
    float dochange = step(.5,keys.x);
    write1(buff.r, mix(buff.r,ST_SPLSH,dochange), txSTATE, fc);
    write1(buff.r, mix(buff.r,iTime,dochange), txPCHANGE, fc);
    // Now that we're done displaying the score, we can clear out the
    // last play's ship, since the initializer can't tell the difference
    // between the last round's ship and the last game's ship.
    fsrt = mix(fsrt,vec4(0),dochange);
}

// Function 17
float sdCharacterTrail( vec3 pos, in float terrain )
{
	vec3 trailOffset = (_CharacterPosition);
	trailOffset.yz  += (_CharacterTrailOffset).yz;
    trailOffset.y = -terrain + _CharacterTrailOffset.y; 

    vec3 trailPos = pos - trailOffset;
    float distanceToPoint = length(trailPos);
    trailPos.x -= _CharacterTrailOffset.x * distanceToPoint;

    // Make it wavy
    trailPos.x += (SmoothTriangleWave( trailPos.z * _CharacterTrailWave.x  ) * _CharacterTrailWave.z * distanceToPoint);

    float trail = sdBox(trailPos - vec3(0.0, 0.0, _CharacterTrailScale.z) , _CharacterTrailScale);
    return trail;
}

// Function 18
GameData getGameData()
{
    GameData gd;

    if (iFrame == 0)
    {
        gd.gGameState = GAME_STATE_LOGO_SCREEN;
        gd.gScore = 0;
        gd.gHighScore = 0;
        gd.gCave = 1;
        gd.gIsCaveInit = false;
        gd.gLevel = 1;
        gd.gFrames = ivec4(0);
        gd.gLives = NUMBER_OF_LIVES;
    }
    else
    {
        loadGameData(gd);
    }

    gd.gFrames.z = int(iTime / ANIM_FRAME_DURATION);
    gd.gFrames.w = gd.gFrames.z / ANIM_FRAMES_IN_GAME_FRAME;

    return gd;
}

// Function 19
void draw_game_info(inout vec4 fragColor, vec2 fragCoord)
{
    GameState game_state;
    LOAD(game_state);
    if (game_state.level == 0.)
        return;

    const int NUM_LINES = GAME_HUD_STATS.data[0];
    const int PREFIX_LENGTH = GAME_HUD_STATS.data[2] - GAME_HUD_STATS.data[1];
    const int NUM_DIGITS = 4;
    const int LINE_LENGTH = PREFIX_LENGTH + NUM_DIGITS;
    
    const float MARGIN = 16.;
    vec2 anchor = vec2(MARGIN, iResolution.y - MARGIN - float((CHAR_SIZE*NUM_LINES) << g_text_scale_shift));
    
    ivec2 uv = text_uv(fragCoord - anchor);
    int line = NUM_LINES - 1 - line_index(uv.y);
    
    // ignore last 2 lines (time/targets left) if game is over
    int actual_num_lines = NUM_LINES - (int(game_state.level < 0.) << 1);
    
    vec4 box = vec4(MARGIN, iResolution.y-MARGIN, ivec2(LINE_LENGTH, (actual_num_lines<<1)-1)<<g_text_scale_shift);
    box.zw *= vec2(CHAR_SIZE);
    box.y -= box.w;
    draw_shadow_box(fragColor, fragCoord, box);
    
    // line spacing
    if ((line & 1) != 0)
        return;
    line >>= 1;
    
    if (uint(line) >= uint(actual_num_lines))
        return;
       
    int start = GAME_HUD_STATS.data[1+line];
    int num_chars = GAME_HUD_STATS.data[2+line] - start;
    int glyph = glyph_index(uv.x);
    if (uint(glyph) < uint(num_chars))
    {
        glyph += start;
        glyph = get_byte(glyph & 3, GAME_HUD_STATS.data[GAME_HUD_STATS.data[0] + 2 + (glyph>>2)]);
    }
    else
    {
        glyph -= num_chars;
        if (uint(glyph) >= uint(NUM_DIGITS))
            return;
        
        int stat;
        switch (line)
        {
            case 0: stat = int(abs(game_state.level)); break;
            case 1: stat = int(game_state.targets_left); break;
            case 2: stat = int(game_state.time_left); break;
            default: stat = 0; break;
        }
		glyph = NUM_DIGITS - 1 - glyph;
        glyph = int_glyph(stat, glyph);
    }

    const vec3 HIGHLIGHT_COLOR = vec3(.60, .30, .23);
    vec4 color = vec4(vec3(.75), 1.);
    if ((line == 0 && fract(game_state.level) > 0.) ||
        (line == 1 && fract(game_state.targets_left) > 0.))
    {
		color.rgb = HIGHLIGHT_COLOR;
    }
    else if (line == 2 && game_state.time_left < 10.)
    {
        float blink_rate = game_state.time_left < 5. ? 2. : 1.;
        if (fract(game_state.time_left * blink_rate) > .75)
            color.rgb = HIGHLIGHT_COLOR;
    }

    print_glyph(fragColor, uv, glyph, color);
}

// Function 20
void GameLoadState()
{
    vec3 state = LoadVec3(addrGameState);
    
 	gtime = state.x;
    time_scale = state.y;
}

// Function 21
float gameoflife( float c, float n ) {
    return ((n==3.0) || (c==1.0 && n==2.0)) ? 1.0 : 0.0;
}

// Function 22
vec4 draw_player( int frame, int dir, ivec2 pos, inout vec4 o, ivec2 iu ) {
    vec4 v = vec4( -1 ) ;
    iu -= pos ;
    if( iINSIDE( iu, ivec2(0), dim_player ) ) {
        frame &= 0x3 ;
        dir   &= 0x3 ;
             if( dir == 1 )  iu = iu.yx ;                    //right
        else if( dir == 2 )  iu = 15 - iu ;                  //down
        else if( dir == 3 )  iu = ivec2( iu.y, 15 - iu.x ) ; //left
        int row_group = frame * 4 + 3 - ( iu.y >> 2 ),
            component = 3 - ( iu.y & 0x3 ),
            sh = 2 * iu.x ;
        uint bits = 0x3U << sh,
             col_ind = ( get_player_br( row_group, component ) & bits ) >> sh ;
        v = get_col( pal_player, col_ind ) ;
    }
    o = v.a > 0. ? v : o ;
    return( v ) ;
}

// Function 23
float get_player_distance(vec3 position, vec4 plane)
{
    return dot(position, plane.xyz) + plane.w - get_player_radius(plane.xyz);
}

// Function 24
vec3 drawPlayer( vec3 col, in vec2 fragCoord, float player, in vec4 playerPosDir, bool dead )
{
    vec2 off = dir2dis(playerPosDir.w);    
    vec2 mPlayerPos = playerPosDir.xy + off*playerPosDir.z;

    vec2 uv = fragCoord.xy /iResolution.xy;
    float xCells = txCells.w*(iResolution.x / iResolution.y);
    vec2 p = uv-vec2(0.5-txCells.z/(xCells*2.), 0.); // center
    p.x *= iResolution.x / iResolution.y;
    
    vec2 q = p - cell2ndc( mPlayerPos );

    float c = sdCircle(q, 0.023);

    vec3 color = mix(p1Color, p2Color, player - 1.);
    
    float phase = 0.5+0.5*sin(2.0*6.2831*iTime);
    if (dead) color = mix(color, vec3(1., 0., 0.), phase);
    col += 0.1*color*exp((-100.0 - (dead ? 50.*phase : 0.))*c);

    return col;
}

// Function 25
vec3 player(Ray ray, vec3 playerPos) {
    vec3 col = vec3(0.45);
    
    // Lines
    float th = fract(ray.pos.y-playerPos.y);
    col = mix(vec3(0.6), col, smoothstep(0.0, 0.006, th)*(1.0-smoothstep(0.994, 1.0, th)));
    float offset = mod(floor(ray.pos.y-playerPos.y),2.0)*0.5;
    float phi = fract(fract(acos(dot(normalize(ray.nor.xz), vec2(1,0)))/(PI/5.0))+offset);
    col = mix(vec3(0.65), col, smoothstep(0.0, 0.025, phi)*(1.0-smoothstep(0.975, 1.0, phi)));

    // Eye
    vec2 m = iMouse.xy/iResolution.xy;
    vec3 w = -normalize(vec3(cos(2.0*PI*m.x), sin(PI*(m.y-0.5)), sin(2.0*PI*m.x)));
    vec3 r = normalize(ray.pos - playerPos);
    
    float eyeStep1 = smoothstep(0.95, 0.959, dot(r, w));
    col = mix(col, vec3(0.1), eyeStep1);
    float eyeStep2 = smoothstep(0.96, 0.999, dot(r, w));
    col = mix(col, vec3(1.0,0.0,0.0), eyeStep2);
    float eyeStep3 = smoothstep(0.998, 1.0, dot(r, w));
    col = mix(col, vec3(1.0,1.0,0.0), eyeStep3);
    
    return col;
}

// Function 26
void handleGameplay( inout vec4 buff, in vec2 fc, in vec2 keys, in vec4 dirs, inout vec4 psvl, inout vec4 fsrt )
{
    // Okay this is a work.
    float escape = texture(KEYBOARD,vec2(KEY_ESC,KD_POS)).r;
    write1(buff.r,mix(buff.r,ST_GMOVR,step(.5,escape)), txSTATE, fc);
    handleCamera(buff,fc,psvl);
    handleCollision(buff,fc,psvl,fsrt);
    handleShip(buff,fc,keys,dirs,psvl,fsrt);
}

// Function 27
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

// Function 28
float PlayerDmg( inout GameState s )
{
    return s.level + 1. + floor( 4. * Rand( vec2( iTime + 11.1, iTime + 11.1 ) ) );
}

// Function 29
void gameLoop( inout vec4 fragColor, in ivec2 coord ) {
    if( coord.y > 33 || coord.y < 32 ) return;
    if( coord.x > 16 ) return;
    
    ivec4 ud1 = LoadVec4( ivec2(0,32 ) );
    ivec4 ud2 = LoadVec4( ivec2(1,32 ) );
    ivec4 ud3 = LoadVec4( ivec2(2,32 ) );
    
    USERCOORD = ud1.xy;
    USERDIR = ud1.z;
    int actionCount = ud1.w;
    
    int action = ud2.x;
    int newAction = ud2.y;
    int live = ud2.z;
    
    USERINV = ud3;
    
    if( actionCount > 0 ) {
        actionCount --;
    }
    
    if( KP(KEY_UP) || KP(KEY_W) ) {
        newAction = FORWARD;
    }
    if( KP(KEY_DOWN) || KP(KEY_S) ) {
        newAction = BACK;
    }
    if( KP(KEY_LEFT) || KP(KEY_A) ) {
        newAction = ROT_LEFT;
    }
    if( KP(KEY_RIGHT) || KP(KEY_D) ) {
        newAction = ROT_RIGHT;
    }
    if( KP(KEY_SPACE) ) {
        newAction = ACTION;
    }
    
    if( actionCount > 8 ) {
        newAction = NONE;
    }
    
    if( actionCount == 0 ) {
        action = newAction;
        newAction = NONE;
        
        if( action == FORWARD ) {
            if( isEmpty( USERCOORD + DIRECTION[USERDIR] ) ) {
                USERCOORD += DIRECTION[USERDIR];
                actionCount = USERMOVESTEPS;
            }
        }
        if( action == BACK ) {
            if( isEmpty( USERCOORD - DIRECTION[USERDIR] ) ) {
                USERCOORD -= DIRECTION[USERDIR];
                actionCount = USERMOVESTEPS;
            }
        }
        if( action == ROT_RIGHT ) {
            USERDIR = (USERDIR + 1) % 4;
            actionCount = USERROTATESTEPS;
        }
        if( action == ROT_LEFT ) {
            USERDIR = (USERDIR + 3) % 4;
            actionCount = USERROTATESTEPS;
        }
        if( action == ACTION ) {
            actionCount = USERACTIONSTEPS;
        }
    }
    
    // store data
    ud1.xy = USERCOORD;
    ud1.z = USERDIR;
    ud1.w = actionCount;
    
    ud2.x = action;
    ud2.y = newAction;
    
    ivec4 map = w(USERCOORD);
    if( map.x > 9 ) {
        live += map.z;
    	StoreVec4( ivec2(3,32 ), ivec4(map.x,map.z,0,0), fragColor, coord );
    } else if( map.x > 5 ) {
        // item
        USERINV[ map.x-6 ] = max( USERINV[ map.x-6], map.z );
    	StoreVec4( ivec2(3,32 ), ivec4(map.x,map.z,0,0), fragColor, coord );
    } else {
    	StoreVec4( ivec2(3,32 ), ivec4(0), fragColor, coord );
    }        
    
    
    if( live > 120 ) {
        live = 120;
    }
    
    for(int i=0; i<4; i++) {
        ivec2 c = USERCOORD + DIRECTION[i];
        ivec4 mo = m(c);
        if( isMonster(mo) && mo.y == 0 ) {
            if( hash12( vec2(c)+iTime ) > .993 - float(mo.w)*.0007 ) {
                live -= 2+int(hash12( vec2(c)-iTime ) * (float(mo.w) + 5.));
            }
        }
    }
    
    ud2.z = live;
    if( live < 0 ) {
        ud2.w = 1;
    	StoreVec4( ivec2(3,32 ), ivec4(-1), fragColor, coord );
    }
        
    StoreVec4( ivec2(0,32 ), ud1, fragColor, coord );
    StoreVec4( ivec2(1,32 ), ud2, fragColor, coord );
    StoreVec4( ivec2(2,32 ), USERINV, fragColor, coord );
}

// Function 30
void attackPlayer(float amount, inout float health, inout float invul, inout float maxHealth) {
 
    if(invul > 0. && amount > 0.) return;
    
    if(amount > 0.) {
     
        health -= amount;
        invul   = 1.;
        
    } else if(amount < 1.) {
     
        health += -amount;
        
    }
    
    health = clamp(health, 0., maxHealth);
    
}

// Function 31
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

// Function 32
GameData readGameData(sampler2D tex, vec2 invRes) {
	GameData data;
    
    data.shipPos = readData3(tex, 0);
    data.shipLastPos = readData3(tex, 1);
    data.shipAccel = readData3(tex, 2);
    data.shipVelocity = readData3(tex, 3);
    data.shipTheta = readData1(tex, 4);
    data.shipDirection = vec3(sin(data.shipTheta), 0.f, cos(data.shipTheta));
    data.touchStart = readData3(tex, 5);
    
    return data;
}

// Function 33
void DrawGame(inout vec3 color, AppState s, vec2 p)
{
    {              
#ifdef DEBUG
        // game
        vec2 p2 = p;
        p2 += vec2(1.5, 0.7);
        p2 *= vec2(7.0, 4.5);
        p2.y += s.playerCell;

        float cellID = floor(p2.y);
        float rndState = step( 0.5, hash11(cellID) );
        if (cellID < CELLS_HEADSTART)
        {
            rndState = CS_EMPTY_LANE;
        }

        float cellState = CS_EMPTY_LANE;
        cellState = mix( cellState, rndState, step(0.5, mod(cellID, 2.0)) );

        // draw obstacles
        if (cellState == CS_RIGHT_LANE)
        {
            vec2 p3 = (p2 -vec2(0.5) -vec2(0.0, cellID));
            color = mix(mix(color, vec3(1.0, 0.0, 0.0), 0.2), color, smoothstep(0.0, 0.01, Circle(p3, 0.5) ));
        }

        if (cellState == CS_LEFT_LANE)
        {
            vec2 p3 = (p2 -vec2(0.5) -vec2(0.0, cellID));
            p3.x += 1.0;
            color = mix(mix(color, vec3(1.0, 0.0, 0.0), 0.2), color, smoothstep(0.0, 0.01, Circle(p3, 0.5) ));
        }

        // draw player
        if (cellID == s.playerCell)
        {
            vec2 p3 = (p2 -vec2(0.5) -vec2(0.0, cellID));
            if (s.isLeftLine == CS_LEFT_LANE)
            {
                p3.x += 1.0;
            }
            color = mix(mix(color, vec3(0.0, 1.0, 0.0), 0.2), color, smoothstep(0.0, 0.01, Circle(p3, 0.45) ));
        }
#endif
    }
}

// Function 34
vec4 GetPlayerUI(vec2 U){
    vec2 uv = U/R*1.4 - vec2(70.,40.)/Cover_Size;
	
    vec2 offset = vec2(-10.,2.)/Cover_Size;
    float ui_play = GetFrontAlpha(uv-offset,vec2(6.,15.));
    float ui_left = GetFrontAlpha(uv-offset-vec2(-0.07,0.),vec2(1.,15.));
    float ui_right = GetFrontAlpha(uv-offset-vec2( 0.07,0.),vec2(7.,15.));
	//float ui_rest = GetFrontAlpha(uv-offset-vec2(-0.14,0.),vec2(4.,15.));
    float ui_panel = GetAlpha(uv.y-55./Cover_Size);
 	float ui_panel_1 = GetAlpha(uv.y+20./Cover_Size);
    // range at 0-500
    float t = floor(500.* fract(A_time/344.));
	vec2 uv_Slider = uv/vec2(1.,R.x/R.y);
    float ui_line = GetAlpha(line(uv_Slider,vec2(160.,10.)/Cover_Size,vec2(660.,10.)/Cover_Size)-2./Cover_Size);
    float ui_line_1 = GetAlpha(line(uv_Slider,vec2(160.,10.)/Cover_Size,vec2(160. + t,10.)/Cover_Size)-1.7/Cover_Size);
    float d_slip = sphere(uv_Slider-vec2(160.+t,10.)/Cover_Size);
    
    float ui_slip = GetAlpha(d_slip - 4./Cover_Size);
    float ui_slip_1 = GetAlpha(d_slip - 1./Cover_Size);
    float ui_slip_2 = GetAlpha(d_slip - 6./Cover_Size);
    
    vec2 uv_Time = uv_Slider*2.;vec2 offset_left = vec2(-8.,0.)/Cover_Size*2.;
    float ui_time_left_0 = GetFrontAlpha(uv_Time-offset_left-vec2(100.,3.)/Cover_Size*2.,vec2(GetTime(A_time,3.),12.));
    float ui_time_left_1 = GetFrontAlpha(uv_Time-offset_left-vec2(110.,3.)/Cover_Size*2.,vec2(GetTime(A_time,2.),12.));
    float ui_time_left_2 = GetFrontAlpha(uv_Time-offset_left-vec2(120.,4.)/Cover_Size*2.,vec2(10.,12.));
    float ui_time_left_3 = GetFrontAlpha(uv_Time-offset_left-vec2(130.,3.)/Cover_Size*2.,vec2(GetTime(A_time,1.),12.));
    float ui_time_left_4 = GetFrontAlpha(uv_Time-offset_left-vec2(140.,3.)/Cover_Size*2.,vec2(GetTime(A_time,0.),12.));
	vec2 offset_right = vec2(570,0.)/Cover_Size*2.;
    float ui_time_right_0 = GetFrontAlpha(uv_Time-offset_right-vec2(100.,3.)/Cover_Size*2.,vec2(0.,12.));
    float ui_time_right_1 = GetFrontAlpha(uv_Time-offset_right-vec2(110.,3.)/Cover_Size*2.,vec2(5.,12.));
    float ui_time_right_2 = GetFrontAlpha(uv_Time-offset_right-vec2(120.,4.)/Cover_Size*2.,vec2(10.,12.));
    float ui_time_right_3 = GetFrontAlpha(uv_Time-offset_right-vec2(130.,3.)/Cover_Size*2.,vec2(4.,12.));
    float ui_time_right_4 = GetFrontAlpha(uv_Time-offset_right-vec2(140.,3.)/Cover_Size*2.,vec2(4.,12.));
    
    float ui_right_5 = GetFrontAlpha((1.-uv_Slider)*1.2+vec2(98,-343)/Cover_Size*2.,vec2(15.,14.));


    vec3 UI_col = vec3(0.);
    UI_col += vec3(0.2,0.22,0.2)*ui_panel;
    UI_col += vec3(0.2,0.22,0.2)*ui_panel_1 * uv.x*(1.-uv.x)*1.5;
    UI_col += vec3(0.55,0.55,0.55) * ui_play;
    UI_col += vec3(0.55,0.55,0.55) * ui_left;
    UI_col += vec3(0.55,0.55,0.55) * ui_right;
    //UI_col += 0.8 * ui_rest;
    UI_col += vec3(0.8) * ui_line;
    UI_col = mix(UI_col,vec3(1.,0.,0.4) , ui_line_1);
    UI_col = mix(UI_col,vec3(0.2,0.2,0.2), ui_slip_2*0.25);
    UI_col = mix(UI_col,vec3(0.9),ui_slip);
    UI_col = mix(UI_col,vec3(1.,0.,0.7), ui_slip_1);
    
    UI_col += ui_time_left_0;
    UI_col += ui_time_left_1;
    UI_col += ui_time_left_2;
    UI_col += ui_time_left_3;
    UI_col += ui_time_left_4;
    UI_col += ui_time_right_0;
    UI_col += ui_time_right_1;
    UI_col += ui_time_right_2;
    UI_col += ui_time_right_3;
    UI_col += ui_time_right_4;
    UI_col += ui_right_5;
    
    float UI_alpha = (ui_play + ui_left + ui_right)*(uv.y*uv.y)*500.
        + ui_panel*0.8 + ui_panel_1*0.1 + ui_slip 
        + ui_time_left_0 + ui_time_left_1 + ui_time_left_2 + ui_time_left_3 + ui_time_left_4
        + ui_time_right_0 + ui_time_right_1 + ui_time_right_2 + ui_time_right_3 + ui_time_right_4
        + ui_right_5;//+ ui_rest;
    return vec4(UI_col,UI_alpha);
}

// Function 35
void handleGame( inout vec4 buff, in vec2 fc, in vec2 keys, in vec4 dirs, inout vec4 psvl, inout vec4 fsrt )
{
    // Okay a whole lot of stuff happens in these branches. So for the sake of
    // the user maybe having MIMD functionality, or proper branching, let's leave this.
    // Doing all of these branches certainly must outweigh conditionality's overhead.
    float curState = readTexel(DATA_BUFFER, txSTATE).r;
    if(curState == ST_SPLSH) handleSplash(buff,fc,keys);
    else if( curState > ST_MMENU-.5 && curState < ST_MMENU+.5 ) handleMMenu(buff,fc,keys,dirs);
    else if( curState > ST_HOWTO-.5 && curState < ST_HOWTO+.5 ) handleHowto(buff,fc,keys);
    else if( curState > ST_INITG-.5 && curState < ST_INITG+.5 ) handleInitialization(buff,fc,psvl,fsrt);
    else if( curState > ST_GAMEP-.5 && curState < ST_GAMEP+.5 ) handleGameplay(buff, fc,keys,dirs,psvl,fsrt);
    else if( curState > ST_CRASH-.5 && curState < ST_CRASH+.5 ) handleLanding(buff,fc,keys,fsrt);
    else if( curState > ST_SCCES-.5 && curState < ST_SCCES+.5 ) handleLanding(buff,fc,keys,fsrt);
    else if( curState > ST_GMOVR-.5 && curState < ST_GMOVR+.5 ) handleGameOver(buff,fc,keys,fsrt);
	else return;
}

// Function 36
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

// Function 37
vec4 getPlayerShipColor( in vec2 uv, in float frac, in float seed ) {
    
    float dither = texture( iChannel2, uv / 8.0 ).r;
    float len = length(uv);
    float theta = atan(uv.y, uv.x)*0.5 + 0.5*PI;
    
    float radius = len - 5.5;
    float x = len / 5.5;
    vec3 norm = normalize(vec3(uv, sqrt(1.0-x*x)*5.5));
    
    vec2 uuv = uv;
    uuv.x = abs(uuv.x);
    float guns = box(uuv-vec2(6, 2), vec2(0, 5)) - 0.0;
    float reguns = box(uuv-vec2(6, -3), vec2(0.5, 1.0)) - 1.0;
    guns = min(guns, reguns);
    
    float cockpit = dot(norm, normalize(vec3(0, 10, 7)));
    cockpit = smoothstep(0.4, 0.5, cockpit);
    
    vec4 colorBase = vec4(mix(FRIEND_COLOR.rgb, vec3(0.8), 0.9), 1.0);
    
    colorBase.rgb = mix(colorBase.rgb, FRIEND_COLOR*0.05, cockpit);
    
    if (radius < 0.0) {
        float d = max(0.0, dot(norm, normalize(vec3(10, 8, 13))));
        colorBase += doDithering(d, dither, 4.0)*0.4;
    } else if (guns < 0.5) {
        colorBase.rgb = FRIEND_COLOR;
        if (reguns < 0.5) colorBase.rgb = mix(colorBase.rgb, vec3(0.3), 0.8);
        if (radius < 1.0) colorBase.rgb = vec3(0);
        if (uv.y > 5.5 && uv.y < 6.5) colorBase.rgb = vec3(0);
    }
        
    float mm = min(radius, guns);
    if (mm > 0.0) {
        if (mm > 1.0) colorBase.a = 0.0;
        else colorBase = vec4(0, 0, 0, 1);
    } else colorBase.a = 1.0;
    
    colorBase.a *= 1.0 - frac;
    
    colorBase.rgb = mix(colorBase.rgb, vec3(1), frac);
    
    // add explosions
    for (int i = 0 ; i < 8 ; i++) {
        vec2 decal = vec2(hash1(float(i)*0.41355+seed),
                          hash1(float(i)*9.00412+seed*9.3153))*2.0-1.0;
        decal *= 10.0;
        float rad = hash1(float(i)*431.412+seed*124.312)*20.0;
        float start = float(i)*0.1;
        float explofrac = smoothstep(start, start+0.1, frac);
        float explo = explosion(float(i)*seed, uv+decal, explofrac, 16.0 + rad);
    	colorBase += vec4(FRIEND_COLOR, 1)*explo;
    }
    
    return colorBase;
}

// Function 38
void Player_GiveArmor( inout Entity entity, float fAmount, bool mega )
{
    if ( mega )
    {
	    entity.fArmor = min( entity.fArmor + fAmount, 200.0 );
    }
    else
    {
        if ( entity.fArmor < 100.0 )
        {
		    entity.fArmor = min( entity.fArmor + fAmount, 100.0 );
        }
	}

}

// Function 39
vec3 gameHud(vec2 uv, float playerVsGpu, vec2 wins, vec2 boosts)
{        
    font_size = 6.;
    print_pos = vec2(-490.*(iResolution.x/iResolution.y)/font_size, 490./font_size-STRHEIGHT(1.0));
    float col = 0.;       
	col += char(ch_P,uv);
    col += char(ch_L,uv);
    col += char(ch_A,uv);
    col += char(ch_Y,uv);
    col += char(ch_E,uv);
    col += char(ch_R,uv);
    col += char(ch_spc,uv);
    col += char(ch_1,uv);
    
    font_size = 4.;
    print_pos = vec2(-490.*(iResolution.x/iResolution.y)/font_size, 400./font_size-STRHEIGHT(1.0));
	col += char(ch_L,uv);
    col += char(ch_i,uv);
    col += char(ch_v,uv);
    col += char(ch_e,uv);
    col += char(ch_s,uv);
    col += char(ch_col,uv);
    col += char(ch_spc,uv);
    col += char(get_digit(5 - int(wins.y)),uv);
    
    print_pos = vec2(-490.*(iResolution.x/iResolution.y)/font_size, 330./font_size-STRHEIGHT(1.0));
	col += char(ch_T,uv);
    col += char(ch_u,uv);
    col += char(ch_r,uv);
    col += char(ch_b,uv);
    col += char(ch_o,uv);
    col += char(ch_s,uv);
    col += char(ch_col,uv);
    col += char(ch_spc,uv);
    col += char(get_digit(int(boosts.x)),uv);
    
    if (wins.x > 4.5)
    {
        font_size = 8.;
        print_pos = vec2(-460.*(iResolution.x/iResolution.y)/font_size + STRWIDTH(1.0)/2., 0.);
        col += char(ch_Y,uv);
        col += char(ch_O,uv);
        col += char(ch_U,uv);
        print_pos = vec2(-460.*(iResolution.x/iResolution.y)/font_size, -STRHEIGHT(1.0));
        col += char(ch_W,uv);
        col += char(ch_I,uv);
        col += char(ch_N,uv);
        col += char(ch_exc,uv);
    }
    else if (wins.y > 4.5)
    {
        font_size = 8.;
        print_pos = vec2(-460.*(iResolution.x/iResolution.y)/font_size + STRWIDTH(1.0)/2., 0.);
        col += char(ch_Y,uv);
        col += char(ch_O,uv);
        col += char(ch_U,uv);
        print_pos = vec2(-460.*(iResolution.x/iResolution.y)/font_size, -STRHEIGHT(1.0));
        col += char(ch_L,uv);
        col += char(ch_O,uv);
        col += char(ch_S,uv);
        col += char(ch_E,uv);
    }
    
    vec3 p1c = p1Color*col;
    
    font_size = 6.;
    
    if (playerVsGpu > 0.5)
    {
        print_pos = vec2(490.*(iResolution.x/iResolution.y)/font_size - STRWIDTH(3.0), 490./font_size-STRHEIGHT(1.0));
        col = char(ch_G,uv);
        col += char(ch_P,uv);
        col += char(ch_U,uv);
    }
    else
    {
        print_pos = vec2(490.*(iResolution.x/iResolution.y)/font_size - STRWIDTH(8.0), 490./font_size-STRHEIGHT(1.0));
        col = char(ch_P,uv);
        col += char(ch_L,uv);
        col += char(ch_A,uv);
        col += char(ch_Y,uv);
        col += char(ch_E,uv);
        col += char(ch_R,uv);
        col += char(ch_spc,uv);
        col += char(ch_2,uv);

        if (wins.y > 4.5)
        {
            font_size = 8.;
            print_pos = vec2(460.*(iResolution.x/iResolution.y)/font_size - STRWIDTH(3.5), 0.);
            col += char(ch_Y,uv);
            col += char(ch_O,uv);
            col += char(ch_U,uv);
            print_pos = vec2(460.*(iResolution.x/iResolution.y)/font_size - STRWIDTH(4.0), -STRHEIGHT(1.0));
            col += char(ch_W,uv);
            col += char(ch_I,uv);
            col += char(ch_N,uv);
            col += char(ch_exc,uv);
        }
        else if (wins.x > 4.5)
        {
            font_size = 8.;
            print_pos = vec2(460.*(iResolution.x/iResolution.y)/font_size - STRWIDTH(3.5), 0.);
            col += char(ch_Y,uv);
            col += char(ch_O,uv);
            col += char(ch_U,uv);
            print_pos = vec2(460.*(iResolution.x/iResolution.y)/font_size - STRWIDTH(4.0), -STRHEIGHT(1.0));
            col += char(ch_L,uv);
            col += char(ch_O,uv);
            col += char(ch_S,uv);
            col += char(ch_E,uv);
    	}
    }
    
    font_size = 4.;
    print_pos = vec2(490.*(iResolution.x/iResolution.y)/font_size - STRWIDTH(9.0), 330./font_size-STRHEIGHT(1.0));
    col += char(ch_T,uv);
    col += char(ch_u,uv);
    col += char(ch_r,uv);
    col += char(ch_b,uv);
    col += char(ch_o,uv);
    col += char(ch_s,uv);
    col += char(ch_col,uv);
    col += char(ch_spc,uv);
    col += char(get_digit(int(boosts.y)),uv);
    
    print_pos = vec2(490.*(iResolution.x/iResolution.y)/font_size - STRWIDTH(8.0), 400./font_size-STRHEIGHT(1.0));
	col += char(ch_L,uv);
    col += char(ch_i,uv);
    col += char(ch_v,uv);
    col += char(ch_e,uv);
    col += char(ch_s,uv);
    col += char(ch_col,uv);
    col += char(ch_spc,uv);
    col += char(get_digit(5 - int(wins.x)),uv);
    
    
                
    vec3 p2c = p2Color*col;
    return p1c + p2c;
}

// Function 40
void PlayerBulletBossCoreTest( inout vec4 playerBullet )
{
	if ( playerBullet.x > 0.0 && Collide( playerBullet.xy, BULLET_SIZE, gBossCore.xy + vec2( 0.0, BOSS_CORE_SIZE.y * 0.25 ), BOSS_CORE_SIZE * 0.5 ) )
    {
		gHit			= vec4( playerBullet.xy, 0.0, 0.0 );
        playerBullet.x 	= 0.0;
		--gBossCore.z;
        if ( gBossCore.z < 0.0 )
        {
            gExplosion 		= vec4( gBossCore.xy + vec2( 0.0, BOSS_CORE_SIZE.y * 0.5 ), 0.0, 0.0 );
            gBossCore.x 	= 0.0;
            gGameState.x 	= GAME_STATE_LEVEL_WIN;
			gGameState.y 	= 0.0;
        }
    }
}

// Function 41
void drawGameFlat( inout vec4 color, vec2 p, AppState s )
{
    // game
	vec2 p0 = p;    
    // float cameraAnim = smoothstep(-0.5, 0.5, sin(iTime) );
    float cameraAnim = 0.0;
	p0 *= mix( 5.0, 10.0, cameraAnim );		// scale field of view
    p0.x += 0.25;							// fix track centering
    p0.y += mix( 2.0, 8.0, cameraAnim );	// move camera pos
    p0.y += s.playerPos.y;
    
    float playerCellID = floor( s.playerPos.y );
    float sPlayer = length( p0 - s.playerPos ) - 0.25;
           
    vec2 p1 = p0;
    p1.y += 2.0 * s.playerPos.y;
    color.rgb = mix( vec3( 1.0 ), color.rgb, smoothstep( 1.5, 1.75, abs( p1.x - 0.5 ) ) );
    color.rgb = mix( texture( iChannel2, fract( p1 ) ).rgb, color.rgb, 0.5 );
       
	// COIN start
    float cellID = floor( p0.y );
    float cellCoinRND = hash11( cellID + g_S.seed );					// skip rnd obstacle every second cell to make room for driving    
    cellCoinRND *= mix( 2.0, -2.0, step( mod( cellID, 5.0 ), 2.5 ) );	// gaps in coin placing: 2 gaps, 2 coins
    cellCoinRND = mix( cellCoinRND, -2.0, step (cellID, 6.0 ) );		// head start
    float cellCoinCol = floor( 4.0 * cellCoinRND );
       
    if ( cellCoinRND >= 0.0 )
    {
        if ( cellID > playerCellID )
           	drawCoin( color.rgb, p0, vec2( cellCoinCol, cellID ) );
        
        if ( cellID == playerCellID && s.coin0Taken < 0.5 )
            drawCoin( color.rgb, p0, vec2( cellCoinCol, cellID ) );
        
        if ( cellID == playerCellID - 1.0 && s.coin1Taken < 0.5 )
            drawCoin( color.rgb, p0, vec2( cellCoinCol, cellID ) );
        
        if ( cellID == playerCellID - 2.0 && s.coin2Taken < 0.5 )
            drawCoin( color.rgb, p0, vec2( cellCoinCol, cellID ) );
       
        if ( cellID == playerCellID - 3.0 && s.coin3Taken < 0.5 )
            drawCoin( color.rgb, p0, vec2( cellCoinCol, cellID ) );
    }    
// COIN end

// OBSTACLE start
    float cellObsRND = hash11( 100.0 * cellID + g_S.seed );		// skip rnd obstacle every second cell to make room for driving
    cellObsRND *= mix( 2.0, -2.0, step( mod( cellID, 4.0 ), 2.5 ) );
    cellObsRND = mix( cellObsRND, -2.0, step( cellID, 8.0) );	// head start
    float cellObsCol = floor( 4.0 * cellObsRND );
    
	if ( cellObsRND >= 0.0 && cellObsCol != cellCoinCol )
    {        
    	float sObstacle = length(
            p0
            -vec2( 0.0, cellID )		// cell pos
            +vec2( 0.5, -0.5 )			// move to cell center
            -vec2( cellObsCol, 0.0 )	// move to column
        ) - 0.25;						// radius of coin
        
    	color.rgb = mix( vec3( 1.0, 0.0, 0.0 ), color.rgb, smoothstep( 0.0, 0.1, sObstacle ) );
        
        vec2 obstaclePos = -vec2( 0.0, cellID )			// cell pos
            				+vec2( 0.5, -0.5 )			// move to cell center
            				-vec2( cellObsCol, 0.0 );	// move to column

        float distObstaclePlayer = length( obstaclePos + s.playerPos );
        
        if ( distObstaclePlayer < 0.5 ) 
        {
            color.rgb += vec3( 0.5 );
        }
    }
    
    color.rgb = mix( vec3( 0.0, 1.0, 0.0 ), color.rgb, smoothstep( 0.0, 0.1, sPlayer ) );

// OBSTACLE end        

}

// Function 42
void FloatToCharacterIndex(ivec2 currentCoord, float floatValue, inout int totalLength, inout vec4 color)
{
    float signValue = sign(floatValue);
    floatValue = signValue * floatValue;
    
    //Handle negetive numbers.
    if(signValue == -1.0)
    {
        SaveValue(currentCoord, ivec2(2, totalLength), 13, color);
        totalLength ++;
    }
    
    if(signValue == 0.0)
    {
        SaveValue(currentCoord, ivec2(2, totalLength), 16, color);
        totalLength ++;
    }
    else
    {
        int valueLength = int(max(floor(logE * log(floatValue)) + 1.0, 1.0));
        
#if defined(GLSL_UNROLL)
        //For some mysterious reasons, the compiler cannot unroll this properly.
        //I tried replace valueLength with a const number, 5 for example, does not work either.
        for (int i = 0; i < valueLength; i++)
        {
            int digit = (int(floatValue) / int(pow(10.0, float(valueLength - i - 1)))) % 10 + 16;
            SaveValue(currentCoord, ivec2(2, totalLength), digit, color);
            totalLength ++; 
        }
#else
        //Unroll 5 times, hard coded
        int i = 0;
        if(i < valueLength)
        {
            int digit = (int(floatValue) / int(pow(10.0, float(valueLength - i - 1)))) % 10 + 16;
            SaveValue(currentCoord, ivec2(2, totalLength), digit, color);
            totalLength ++; 
        }
        i ++;
        if(i < valueLength)
        {
            int digit = (int(floatValue) / int(pow(10.0, float(valueLength - i - 1)))) % 10 + 16;
            SaveValue(currentCoord, ivec2(2, totalLength), digit, color);
            totalLength ++; 
        }
        i ++;
        if(i < valueLength)
        {
            int digit = (int(floatValue) / int(pow(10.0, float(valueLength - i - 1)))) % 10 + 16;
            SaveValue(currentCoord, ivec2(2, totalLength), digit, color);
            totalLength ++; 
        }
        i ++;
        if(i < valueLength)
        {
            int digit = (int(floatValue) / int(pow(10.0, float(valueLength - i - 1)))) % 10 + 16;
            SaveValue(currentCoord, ivec2(2, totalLength), digit, color);
            totalLength ++; 
        }
        i ++;
        if(i < valueLength)
        {
            int digit = (int(floatValue) / int(pow(10.0, float(valueLength - i - 1)))) % 10 + 16;
            SaveValue(currentCoord, ivec2(2, totalLength), digit, color);
            totalLength ++; 
        }
#endif
    }
    
    //Handle two decimals.
    float fracValue = fract(floatValue);
    SaveValue(currentCoord, ivec2(2, totalLength), 14, color);
    totalLength ++;
    SaveValue(currentCoord, ivec2(2, totalLength), int(fracValue * 10.0) % 10 + 16, color);
    totalLength ++;
    SaveValue(currentCoord, ivec2(2, totalLength), int(fracValue * 100.0) % 10 + 16, color);
    totalLength ++;
}

// Function 43
vec4 GetGameData(sampler2D sampler, int pixelIndex, vec3 iResolution)
{
    int w = int(iResolution.x);

    int x = pixelIndex  % w;
    int y = pixelIndex/w;
    
    return texelFetch( sampler, ivec2(x,y),0);
}

// Function 44
float Character(float n, vec2 p)
{
    p = floor(p*vec2(4.0, -4.0) + 2.5);

    if (clamp(p.x, 0.0, 4.0) == p.x)
    {
        if (clamp(p.y, 0.0, 4.0) == p.y)
        {
            if (int(mod(n/exp2(p.x + 5.0*p.y), 2.0)) == 1) return 1.0;
        }
    }
    return 0.0;
}

// Function 45
int Game()
{
    LoadData();
    
    if(state==3. && iTime-time>3.)
    {
        InitData();
        return 0;
    }
    
    if(iMouse.z>0.)
    {
        state=1.;
        vec2 nm=vec2(iMouse.x,iMouse.y)/iResolution.y;
        alpha=atan(obj.y-nm.y,obj.x-nm.x);
        speed=distance(obj,nm)*10.;
        time=iTime;
    }
    
    if(state==1. && alpha>-2.) state=0.;
    if(state==1. && alpha<-3.) state=0.;
    
    
    if(iMouse.z<=0. && mouse.z>0. && state==1.)
    {
        	state=2.;
        	float y=Parabola(target.x-obj.x,alpha,speed)+obj.y;
        	if(abs(y-target.y)<0.022) state=3.;
    }
    
    
	return 0;   
}

// Function 46
vec4 gameOfLife(vec2 uv)
{
    vec3 gen = vec3(0.0,0.0,0.0);
   	
    float textureSize = iResolution.x*iResolution.y;    
    float onePixel = 1.0/textureSize;
    
    float total = 0.0;
    //
    float tl = NearestTextureSample(uv + vec2(-1.0/iResolution.x,-1.0/iResolution.y)).r;
    float tm = NearestTextureSample(uv + vec2(0.0,-1.0/iResolution.y)).r;
    float tr = NearestTextureSample(uv + vec2(1.0/iResolution.x,-1.0/iResolution.y)).r;
    float ml = NearestTextureSample(uv + vec2(-1.0/iResolution.x,0.0)).r;
    float mm = NearestTextureSample(uv + vec2(0.0,0.0)).r;
    float mr = NearestTextureSample(uv + vec2(1.0/iResolution.x,0.0)).r;
    float bl = NearestTextureSample(uv + vec2(-1.0/iResolution.x,1.0/iResolution.y)).r;
    float bm = NearestTextureSample(uv + vec2(0.0,1.0/iResolution.y)).r;
    float br = NearestTextureSample(uv + vec2(1.0/iResolution.x,1.0/iResolution.y)).r;
    float sum = tl + tm + tr + ml + mr + bl + bm + br;
    if(mm < 0.001)
    {
        if((abs(sum-3.0) < 0.001) )
        {
            total += 1.0;   
        }
    }
    else
    {
       if((abs(sum-3.0) < 0.001) )
        {
            total += 1.0;   
        }

        if((abs(sum-1.0) < 0.001) )
        {
            total += 0.0;   
        } 
    }
    
    gen += vec3(1.0,1.0,1.0)*total;   
    
    return vec4(gen,1.0);
}

// Function 47
vec3 gameStartCounter(vec2 uv, float state)
{        
    font_size = 16.;
    print_pos = vec2(-STRWIDTH(1.0)/2.0, -STRHEIGHT(1.0)/2.0);
    float col = char(get_digit(int(-state)+1),uv);       

    return vec3(.1,.9,1.)*col;
}

// Function 48
float sdCharacterShadow( vec3 pos )
{
    pos -= _CharacterPosition;
    vec3 scale = _CharacterScale;
    float scaleMul = min(scale.x, min(scale.y, scale.z));
    
    rY(pos, _CharacterRotation);

    pos /= scale;

    float mainCloak = sdMainCloak( pos );
    float longScarf = sdScarf(pos);

    return min( mainCloak, longScarf) * scaleMul;
}

// Function 49
void TickGame(inout vec4 c, ivec2 p)
{
    Storage s; // = LoadState(iChannel0, v);
    if (p.y > nrobots // unused BufferA rows
    	|| p.x > s.length()) return; // no data pixels after that
    // since every robot needs every other robot's state,
    // pretty much every frame, may as well load them all 
    // and unpack once here into a global array of Guts
    // so other routines won't duplicate loading/unpacking
    for (int i = 1; i <= nrobots; ++i) {
        Storage sn = LoadState(iChannel0, i);
        if (i == p.y) s = sn;
        robotguts[i-1] = GutsOfState(sn);
    }
    c = texelFetch(iChannel0, p, 0); // individual texel will likely get overwritten, but you never know
    if (p.y == 0)
        GameState(s);
    else
        TickRobot(s, p);
    SaveState(c, p.x, s);
}

// Function 50
float get_player_radius(vec3 direction)
{
    const float HORIZONTAL_RADIUS = 16., VERTICAL_RADIUS = 48.;

    direction = abs(direction);
    return direction.z > max(direction.x, direction.y) ? VERTICAL_RADIUS : HORIZONTAL_RADIUS;
}

// Function 51
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

// Function 52
vec2 sdCharacter( vec3 pos )
{
    // Now we are in character space - Booo YA! - I never ever say Boooo YA!. Peter Pimley 
    // says that. Peter: have you been putting comments in my code?
    pos -= _CharacterPosition;
    vec3 scale = _CharacterScale;
    float scaleMul = min(scale.x, min(scale.y, scale.z));
    
    rY(pos, _CharacterRotation);

    pos /= scale;

    float mainCloak = sdMainCloak( pos );
    vec2  mainCloakMat = vec2(mainCloak, MAT_CHARACTER_MAIN_CLOAK );

    float headScarf = sdHeadScarf(pos);
    vec2  headScarfMat = vec2(headScarf, MAT_CHARACTER_NECK_SCARF );

    float longScarf = sdScarf(pos);
    vec2  longScarfMat = vec2( longScarf, MAT_CHARACTER_LONG_SCARF );
    headScarfMat = smin_mat( headScarfMat, longScarfMat, 0.02, 0.1 );

    float head      = sdHead( pos );
    vec2  headMat	= vec2( head, MAT_CHARACTER_BASE );
    headScarfMat    = smin_mat(headScarfMat, headMat, 0.05, 0.75);

    vec2  characterMat = min_mat(mainCloakMat, headScarfMat); 
    characterMat = sdFace( pos, characterMat );

    vec2 legsMat = vec2( sdLegs(pos), MAT_CHARACTER_BASE );
    characterMat = min_mat( characterMat, legsMat );

    // chope the bottom. This is to chop the bottom of right leg. Though
    // I have positioned the character so that the right leg is hidden by terrain. 
    // Commenting it out for now
//    characterMat.x = max( characterMat.x, -sdPlane( pos - vec3(0.0, -0.85, 0.0) ) );
    characterMat.x *= scaleMul;


    return characterMat;
}

// Function 53
bool pix_coll_player_water_track( ivec4 player, ivec2 player_dim, ivec2 track_off, ivec2 track_dim ) {
    //test 8x8 middle pixels of player for water, and if a single pixel is not on water than player lives
    ivec2 d, pos = player.xy, off = track_off ;
    if( ! iRECTS_COLLIDE( ivec4( pos, player_dim ), ivec4( 0, off.y, track_dim ) ) ) {
        return( false ) ;
    }
    int num_pix_coll = 0 ;
    for( d.y = 4 ; d.y < 12 ; ++ d.y ) {
        if( iINSIDE( pos.y + d.y, off.y, off.y + track_dim.y ) ) {
            for( d.x = 4 ; d.x < 12 ; ++ d.x ) {
                vec4 dummy ;
                if( draw_player( int( player.w >= END_JUMP_T ), player.z & 0x3, ivec2( 0 ), dummy, d ).a > 0. ) {
                    vec4 b = texelFetch( iChannel0, ivec2( pos.x+off.x+d.x, pos.y+d.y ) & 0xff, 0 ) ;
                    if( b != col_water ) {
                        return( false ) ;
                    }
                }
            }
        }
    }
    return( true ) ;
}

// Function 54
void GameStoreState(inout vec4 fragColor, in vec2 fragCoord)
{
    StoreVec3( addrGameState, vec3(gtime, time_scale, input_state), fragColor, fragCoord );
}

// Function 55
float PrintCharacter( in int char, in vec2 uv )
{
    uint idx = uint(char);
    vec2 charp = vec2( idx&0xFU, idx>>4U );

//    uv = clamp(uv,vec2(0),vec2(.5,1));
    if ( min(uv.x,uv.y) < .0 || max(uv.x-.5,uv.y-1.) > .0 )
        return 0.;
    uv.x += .25;
//    return step(textureLod(iChannel0, (uv+charp)/16., .0).w,.5);
//    return smoothstep(.53,.47,textureLod(iChannel1, (uv+charp)/16., .0).w );
//    return textureLod(iChannel0, (uv+charp)/16., .0).x;
    return 1.-textureLod(iChannel1, (uv+charp)/16., .0).w;
}

// Function 56
void fakePlayerCameras() {
    if (globalCam.z > CAMERA_ZOOM_MIN) {
        for(int i = 0; i < N_PLAYERS; i++) {
            playersCam[i] = globalCam;
        }
    } else {
        for(int i = 0; i < N_PLAYERS; i++) {
            vec2 playerPosScreen = camWorld2Screen(globalCam, playersPos[i]);
            playersCam[i] = vec3(playersPos[i] - playerPosScreen / CAMERA_ZOOM_MIN, CAMERA_ZOOM_MIN);
        }
    }
}

// Function 57
float player(vec2 p)
{
    vec4 player = get(vPlayer);
    vec2 cam = player.xy * get(vMoveCam).x;
    p += cam;
    float s = 0.5;
    
    p = rot(p - player.xy, player.z) - up * 5.0;
    float r = rect(p, vec2(15.0, 30.0) * s);
    float r2 = rect(p + up * 5.0, vec2(10.0, 14.0) * s);
    float r3 = rect(p + up * 6.0, vec2(8.0, 12.0) * s);
    float t = triangle(p + up * 30.0 * s, PI / 8.0, 25.0 * s);
    return diff(sum(diff(r, r2), r3), -t);
}

// Function 58
vec2 buildCharacter(vec3 p, vec3 posOffset, vec3 posOffset2, vec4 special, float headOffset, float inputWalk, float breathSpeed, float headMix01, float matBody, float matEyes, float matEyesSockets)
{
    inputWalk += 0.5 * PI;
    
    headOffset *= 0.5 + 0.5 * (1.0 - special.x);
    
    // rotations
    mat3 rotEyes 		= rotX(0.05 * PI * sin(2.3 * iTime)) * rotY(0.2 * PI * sin(iTime));
    mat3 rotHandL 		= rotZ( 0.125 * PI) * rotX(0.06 * PI * sin(5.0 * iTime));
    mat3 rotHandR 		= rotZ(-0.125 * PI) * rotX(0.06 * PI * sin(0.5 * PI + 5.0 * iTime));
    mat3 rotWalk  		= rotY(0.3 * PI * sin(inputWalk));
    mat3 rotWalkHead  	= rotY(0.1 * PI * sin(inputWalk)) * rotZ(0.25 * sin((0.4 * PI + inputWalk)));
    mat3 rotHole  		= rotX(0.5 * PI );
    
	float angleHands = 0.25 * PI;
    vec2 cHands = vec2(sin(angleHands),cos(angleHands));    
    
    // transformations
    vec3 posRoot 		= p - posOffset;
    vec3 posRoot2 		= p - posOffset - posOffset2;
    vec3 posBody 		= attachToParent(posRoot	, vec3(0.0, 0.3, 0.0)			, rotWalk);
    vec3 posHead 		= attachToParent(posRoot2	, vec3(0.0, headOffset + 0.08 * sin(PI + iTime * breathSpeed), -0.2)	, rotWalkHead);
    vec3 posHandL		= attachToParent(posHead	, vec3( 1.0, -0.6, -0.2)		, rotHandL);
    vec3 posHandR		= attachToParent(posHead	, vec3(-1.0, -0.6, -0.2)		, rotHandR);
    vec3 posEyeL 		= attachToParent(posHead	, vec3( 0.5, 0.4, -0.7)			, rotEyes);
    vec3 posEyeR 		= attachToParent(posHead	, vec3(-0.5, 0.4, -1.0)			, rotEyes);
    vec3 posEyePupilL 	= attachToParent(posEyeL	, vec3( 0.0, 0.0, -EYE_RADIUS_L), MX_ID);
    vec3 posEyePupilR 	= attachToParent(posEyeR	, vec3( 0.0, 0.0, -EYE_RADIUS_R), MX_ID);
    vec3 posHole 		= attachToParent(posHead	, vec3( 0.3 * sin(0.5 * breathSpeed * iTime), -0.5,-0.6 )		, rotHole);

    // shapes
    //float fLegs 		= sdCappedTorus		(posBody, vec2(sin(LEGS_ANGLE),cos(LEGS_ANGLE)), 1.2, 0.1); 
    float fLegs 		= sdCappedTorus		(posBody, vec2(sin(LEGS_ANGLE),cos(LEGS_ANGLE)), 0.8, 0.1); 
    float fHeadV1 		= sdSphere			(posHead, 1.0 + 0.05 * sin(iTime * breathSpeed));
    float fHeadV2 		= sdRoundBox		(posHead, vec3(0.6 + 0.05 * sin(iTime * breathSpeed), 1.1, 0.7), 0.1);
    float fHead			= mix(fHeadV1, fHeadV2, headMix01);
    float fHandL 		= sdCappedTorus		(posHandL, cHands, 0.6, 0.2 );
    float fHandR 		= sdCappedTorus		(posHandR, cHands, 0.6, 0.2 );
    float fEyeSocketL 	= sdSphere			(posEyeL, 1.2 * EYE_RADIUS_L);
    float fEyeSocketR 	= sdSphere			(posEyeR, 1.2 * EYE_RADIUS_R);
    float fEyeL 		= sdSphere			(posEyeL, EYE_RADIUS_L);
    float fEyeR 		= sdSphere			(posEyeR, EYE_RADIUS_R);
    float fEyePupilL 	= sdSphere			(posEyePupilL, 0.1);
    float fEyePupilR 	= sdSphere			(posEyePupilR, 0.06);
    float fHole 		= sdVerticalCapsule	(posHole, 1.0, -0.02 + special.y * (0.3 + 0.03 * sin(4.0 * breathSpeed * iTime)));
    
    // distances + materials
    vec2 dLegs 			= vec2(fLegs		, matBody);
    vec2 dHead 			= vec2(fHead		, matBody);
    vec2 dHandL			= vec2(fHandL		, matBody);
    vec2 dHandR			= vec2(fHandR		, matBody);
    vec2 dEyeSocketL 	= vec2(fEyeSocketL	, matEyesSockets);
    vec2 dEyeSocketR 	= vec2(fEyeSocketR	, matEyesSockets);
    vec2 dEyeL 			= vec2(fEyeL		, matEyes);
    vec2 dEyeR 			= vec2(fEyeR		, matEyes);
    vec2 dEyePupilL 	= vec2(fEyePupilL	, MAT_EYES_DOTS);
    vec2 dEyePupilR 	= vec2(fEyePupilR	, MAT_EYES_DOTS);
    vec2 dHole 			= vec2(fHole		, matEyesSockets);
    
    // build character
    vec2 dCharacter = opSmoothUnionV2		(dLegs			, dHead			, 1.8);
    dCharacter 		= opSmoothUnionV2		(dHandL			, dCharacter	, 0.15);
    dCharacter 		= opSmoothUnionV2		(dHandR			, dCharacter	, 0.15);
    dCharacter 		= opSmoothSubtractionV2	(dEyeSocketL	, dCharacter	, 0.05);
    dCharacter 		= opSmoothSubtractionV2	(dEyeSocketR	, dCharacter	, 0.05);
    dCharacter 		= opSmoothUnionV2		(dEyeL			, dCharacter	, 0.05);
    dCharacter 		= opSmoothUnionV2		(dEyeR			, dCharacter	, 0.05);
    dCharacter 		= opSmoothUnionV2		(dEyePupilL		, dCharacter	, 0.0);
    dCharacter 		= opSmoothUnionV2		(dEyePupilR		, dCharacter	, 0.0);
    dCharacter 		= opSmoothSubtractionV2	(dHole			, dCharacter	, 0.05);

    vec2 dFloorCutoff = vec2(p.y - 0.01, MAT_FLOOR);
    dCharacter 		= opSmoothSubtractionV2	(dFloorCutoff	, dCharacter	, 0.05);
    
    return dCharacter;
}

// Function 59
void player_AI_logic() {
    int exidx = index_idx();
    int imidx = (ipx.y * int(iResolution.x) + ipx.x) - index_idx()*3;
    if (imidx != 2)return;
    bool is_player=false;
    int el_pos = logicz[0]; //position in array
    int el_ttl = logicz[1]; //timer
    int el_ID = logicz[2]; //element id
    int el_act = logicw[0]; //action
    int el_sc = (logicw[1] << 8) + logicw[2]; //score
    //el_sc++; //debug
    
#ifndef no_AI
    is_player=(exidx == 0);
    if ((is_player) && (lgs2().x != 0.))return; //check for pause if player board not selected
#else
    is_player=(exidx == int(lgs2().x));
    if ((!is_player)&&(el_pos < 17 * 10)&&(el_act!=nac))return;
#endif
    
    //spawn new block
    if (el_act == nac) {
        el_act = draw;
        el_ID = int(float(barr - 1) * rand(vec2(ipx) + vec2(mod(iTime, float(0xffff)), mod(iTime, float(0xffff)) / 2.)));
        //el_ID = exidx%(barr); //debug
        el_ttl = is_player?speed:AIspeed;
#ifdef no_AI
        el_pos = 20 * 10 + 4;
#else
        el_pos = is_player?20 * 10 + 4:AI_pos_gen(el_ID);
#endif
        save_ltmp(el_pos, el_ttl, el_ID, el_act, el_sc);
        return;
    }
    
    //check after block is down
    if (el_act == afc) {
        el_act = after_ac(el_pos, el_ID,el_sc);
        isend(el_pos);
        save_ltmp(el_pos, el_ttl, el_ID, el_act, el_sc);
        return;
    }
    
    if (el_act == afc_e) {
        el_act = afc;
        save_ltmp(el_pos, el_ttl, el_ID, el_act, el_sc);
        return;
    }
    
//move down on timer
    el_ttl = (el_ttl - 1 > 0) ? el_ttl - 1 : 0;
    
    if(is_player){
    //key press move
    int tac = int(lgs2().z);
    if ((tac != nac)&&(el_act == draw)) {
        el_act = ltoe_action(el_pos, tac, el_ID);
        save_ltmp(el_pos, el_ttl, el_ID, el_act, el_sc);
        return;
    }
    if ((el_act == down_e) || (el_act == left_e) || (el_act == right_e) || (el_act == rotate_e)) {
        el_pos = apply_move(el_pos, el_act, el_ID);
        if (el_act == down_e)el_ttl = is_player?speed:AIspeed;
        el_act = draw;
        save_ltmp(el_pos, el_ttl, el_ID, el_act, el_sc);
        return;
    }
    }

    
    if (el_ttl == 0) {
        if (el_act == draw) {
            el_act = ltoe_action(el_pos, down_l, el_ID);
            save_ltmp(el_pos, el_ttl, el_ID, el_act, el_sc);
            return;
        }
        if (el_act == down_e) {
            el_pos = apply_move(el_pos, el_act, el_ID);
            el_act = draw;
            el_ttl = is_player?speed:AIspeed;
            save_ltmp(el_pos, el_ttl, el_ID, el_act, el_sc);
            return;
        }
    }
    save_ltmp(el_pos, el_ttl, el_ID, el_act, el_sc);
}

// Function 60
void GameUpdate()
{
    float behav = GetSceneTileBehaviour(gPlayerNextCoords);
    
    if (behav == kBehavObstacle)
    {
        GameSetState(kStateGameOver);
        gPlayerDeathCause = behav;
        gPlayerDeathTime = iTime;
    }
    else if (gPlayerMotionTimer >= 1.0 && gGameState != kStateGameOver)
    {       
        gPlayerCoords = gPlayerNextCoords;
        gPlayerRotation = gPlayerNextRotation;
        
        if (behav != kBehavGround)
        {
            GameSetState(kStateGameOver);
            gPlayerDeathCause = behav;
            gPlayerDeathTime = iTime;
        }
        else
        {       
            vec2 axes = SampleAxes();

            if (dot(axes, axes) > 0.0)
            {
                vec2 nextCoords = GetNextCoordinates(gPlayerCoords + axes);

                if (GetSceneTileBehaviour(nextCoords) != kBehavObstacle)
                {
                    gPlayerNextCoords = nextCoords;
                    gPlayerMotionTimer = 0.0;
                    gPlayerNextRotation = atan(axes.x, axes.y);
					gScore = max(gScore, floor(nextCoords.y));
                }
            }
        }
    }
    else
    {
        gPlayerMotionTimer += iTimeDelta * kPlayerSpeed;
    }
        
    vec4 coordsVss = GetSceneTileVss(gPlayerCoords);
    vec4 nextCoordsVss = GetSceneTileVss(gPlayerNextCoords);
    gPlayerCoords.x += coordsVss.x * coordsVss.w * iTimeDelta;
    gPlayerNextCoords.x += nextCoordsVss.x * nextCoordsVss.w * iTimeDelta;
    gPlayerVisualCoords.xz = mix(gPlayerCoords, gPlayerNextCoords, clamp(gPlayerMotionTimer, 0.0, 1.0));
    gPlayerVisualCoords.y  = kPlayerJumpHeight * JumpCurve(min(1.0, gPlayerMotionTimer));
    gPlayerVisualRotation  = MixAngle(gPlayerRotation, gPlayerNextRotation, clamp(gPlayerMotionTimer, 0.0, 1.0));
    gPlayerScale = 1.0 + 0.1 * JumpCurve(min(1.0, gPlayerMotionTimer));
    
    if (gGameState == kStateGameOver)
    {
         if (gPlayerDeathCause == kBehavWater)
             gPlayerVisualCoords.y = kPlayerJumpHeight * JumpCurve(gPlayerMotionTimer);
    }
}

// Function 61
int closestPlayer(vec2 screenCoord) {
    vec2 worldCoord = camScreen2World(globalCam, screenCoord);

    int minPlayer = 0;
    float minDist = maxGlobalDist();
    for(int i = 0; i < N_PLAYERS; i++) {
    	float dist = length(worldCoord - playersPos[i]);
        
        if (dist < minDist) {
            minDist = dist;
            minPlayer = i;
        }
  	}
    
    return minPlayer;
}

// Function 62
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

// Function 63
vec4 prepare_character( int n, ivec2 pos, ivec2 iu ) {
    vec3 v = vec3( 0 ) ;
    iu -= pos ;
    if( n > 0 && n <= NUM_FONT_CHARS && iINSIDE( iu, ivec2(0), ch_gfx_dim ) ) {
        int x = iu.x, y = iu.y ;
        n -- ;
        x += 5 * ( n & 3 ) ;
        float fx2 = exp2(-float(x+1)) ;

        n >>= 2 ;
        int part = y < 4 ? n * 2 + 1 : n * 2 ;
        y = 3 - (y&3) ;
        v = vec3( fract( get_bit_row( part, y ) * fx2 ) >= .5 ) ;
    }
    return( vec4( v, 1 ) ) ;
}

// Function 64
vec4 drawGameOverScreen( in vec2 uv, in float state, in float msel, in float pchange, in vec4 fsrt )
{
    // Align the time with the start of the screen.
    float time = iTime-pchange;
    
    // The Y position of the camera.
    float py = 8.5 - smoothstep(0.0,8.0,time)*8.0;
    
    vec3 cam = vec3(8.9,py,3.0);
    
    vec4 bg = drawPlayfield(uv,cam);
   	float t = drawGameOverText(uv,time,fsrt);
    
    return mix(bg, vec4(t,t,t,1), step(.1,t)) * min(1.0, time);
}

// Function 65
void PlayerBulletSoldierTest( inout vec4 playerBullet, inout vec4 soldier )
{
    if ( playerBullet.x > 0.0 && Collide( playerBullet.xy, BULLET_SIZE, soldier.xy, SOLDIER_SIZE ) )
    {
        gExplosion 		= vec4( soldier.xy + vec2( 0.0, SOLDIER_SIZE.y * 0.5 ), 0.0, 0.0 );
        gHit		 	= vec4( playerBullet.xy, 0.0, 0.0 );
		soldier.x 		= 0.0;
        playerBullet.x 	= 0.0;
    }
}

// Function 66
float DrawCharacter(inout vec2 p, in int c)
{
    float fC = float(c);
    float color = 0.0;
	if(p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0)
    {
        color = step(texture(iChannel1, p / 16.0 + fract(floor(vec2(fC, 15.99 - fC / 16.0)) / 16.0)).a, 0.5);
    }
    p.x -= 0.5;
    return color;
}

// Function 67
vec4 SampleCharacterTex( uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vUV = (vec2(iChPos) + vCharUV) / 16.0f;
    return textureLod( iChannelFont, vUV, 0.0 );
}

// Function 68
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

// Function 71
float volumetric_player_shadow(vec3 p, vec3 rel_cam_pos)
{
#if VOLUMETRIC_PLAYER_SHADOW
    vec3 occluder_p0 = rel_cam_pos;
    vec3 occluder_p1 = occluder_p0 - vec3(0, 0, 48);
#if VOLUMETRIC_PLAYER_SHADOW >= 2
    occluder_p0.z -= 20.;
#endif // VOLUMETRIC_PLAYER_SHADOW >= 2

    float window_dist = p.x * (1. / VOL_SUN_DIR.x);
    float occluder_dist = occluder_p0.x * (1. / VOL_SUN_DIR.x);
    p -= VOL_SUN_DIR * max(0., window_dist - occluder_dist);
    vec3 occluder_point = closest_point_on_segment(p, occluder_p0, occluder_p1);
    float vis = linear_step(sqr(16.), sqr(24.), length_squared(p - occluder_point));

#if VOLUMETRIC_PLAYER_SHADOW >= 2
    vis = min(vis, linear_step(sqr(8.), sqr(12.), length_squared(p - rel_cam_pos)));
#endif // VOLUMETRIC_PLAYER_SHADOW >= 2

    return vis;
#else
    return 1.;
#endif // VOLUMETRIC_PLAYER_SHADOW
}

// Function 72
float PrintCharacterInternal( in uint char, in vec2 uv )
{
    vec2 charp = vec2( char&0xFU, 0xFU-(char>>4U) );

/*    if ( min(uv.x,uv.y) < .0 || max(uv.x-.5,uv.y-1.) > .0 )
        return 0.;*/
    uv.x += .25;

    float s = 10./iResolution.x;
    return smoothstep(.5+s,.5-s,textureLod(iChannel1, (uv+charp)/16., .0).w);
}

// Function 73
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

// Function 74
void Player_GiveHealth( inout Entity entity, float fAmount, bool mega )
{    
    if ( mega )
    {
	    entity.fHealth = min( entity.fHealth + fAmount, 200.0 );
    }
    else
    {
        if ( entity.fHealth < 100.0 )
        {
		    entity.fHealth = min( entity.fHealth + fAmount, 100.0 );
        }
	}
}

// Function 75
float drawGameOverText( in vec2 uv, in float time, in vec4 fsrt )
{
    uv.x *= iResolution.y/iResolution.x; // Unfix aspect ratio
    uv = uv *.5 + .5;
    vec2 scr = uv*vec2(320,180);
    vec2 pos = vec2(30,150);
    float charG = 0.0, charS = 0.0, charV = 0.0, charE = 0.0;
    // Current value of the counting-up score.
    float curVal = fsrt.y * clamp( (time-3.0)*.1667, 0.0, 1.0 );
    // Current size of the gets-bigger-when-done counting score.
    vec2 curSize = MAP_SIZE*(1.0+step(fsrt.y*.999,curVal)*.25);
    
    // "Game over"
    charG += drawChar(CH_G,pos,MAP_SIZE,scr);
    charG += drawChar(CH_A,pos,MAP_SIZE,scr);
    charG += drawChar(CH_M,pos,MAP_SIZE,scr);
    charG += drawChar(CH_E,pos,MAP_SIZE,scr);
    pos.x += KERN;
    charG += drawChar(CH_O,pos,MAP_SIZE,scr);
    charG += drawChar(CH_V,pos,MAP_SIZE,scr);
    charG += drawChar(CH_E,pos,MAP_SIZE,scr);
    charG += drawChar(CH_R,pos,MAP_SIZE,scr);
    
    // "Your score: {score}"
    pos.x = 30.0;
    pos.y -= 20.0;
    charS += drawChar(CH_Y,pos,MAP_SIZE,scr);
    charS += drawChar(CH_O,pos,MAP_SIZE,scr);
    charS += drawChar(CH_U,pos,MAP_SIZE,scr);
    charS += drawChar(CH_R,pos,MAP_SIZE,scr);
    pos.x += KERN;
    charS += drawChar(CH_S,pos,MAP_SIZE,scr);
    charS += drawChar(CH_C,pos,MAP_SIZE,scr);
    charS += drawChar(CH_O,pos,MAP_SIZE,scr);
    charS += drawChar(CH_R,pos,MAP_SIZE,scr);
    charS += drawChar(CH_E,pos,MAP_SIZE,scr);
    charS += drawChar(CH_COLN,pos,MAP_SIZE,scr);
    pos.x += 7.0*KERN;
    charV += drawInt(curVal*5000.0,pos,curSize,scr);
    
    // -Press {enter} to exit-
    pos.x = 180.0;
    pos.y = 20.0;
    charE += drawChar(CH_P,pos,MAP_SIZE,scr);  
    charE += drawChar(CH_R,pos,MAP_SIZE,scr);  
    charE += drawChar(CH_E,pos,MAP_SIZE,scr);  
    charE += drawChar(CH_S,pos,MAP_SIZE,scr);  
    charE += drawChar(CH_S,pos,MAP_SIZE,scr);
    pos.x += KERN;    
    charE += drawChar(CH_ENTA,pos,MAP_SIZE,scr); pos.x -= KERN-MAP_SIZE.x;
    charE += drawChar(CH_ENTB,pos,MAP_SIZE,scr);  
    pos.x += KERN;  
    charE += drawChar(CH_T,pos,MAP_SIZE,scr);  
    charE += drawChar(CH_O,pos,MAP_SIZE,scr);  
    pos.x += KERN;  
    charE += drawChar(CH_R,pos,MAP_SIZE,scr);  
    charE += drawChar(CH_E,pos,MAP_SIZE,scr);  
    charE += drawChar(CH_T,pos,MAP_SIZE,scr);  
    charE += drawChar(CH_U,pos,MAP_SIZE,scr);  
    charE += drawChar(CH_R,pos,MAP_SIZE,scr);  
    charE += drawChar(CH_N,pos,MAP_SIZE,scr);  
    
    // All this nonsense with floor() is to make sure that
    // the sheen is pixel perfect relative to the letters,
    // and has a snazzy retro stairstep gradient.
    float sheenTime = iTime - floor(scr.x)*.003125;
    float sheen = .5 + .5*smoothstep( .995, 1.00, sin(sheenTime));
    sheen = floor(sheen*4.0)*.25;
    
    return charG * sheen * step(1.0,time) + 
           charS * sheen * step(2.0,time) + 
           charV *         step(3.0,time) + 
           charE * sheen * step(10.0,time);
}

// Function 76
vec2 initPlayerPos(vec2 basePos, float radius, float basePhase, float timeC) {
    vec2 center = basePos - radius * vec2(cos(basePhase), sin(basePhase));
    float phase = basePhase + iTime * timeC;
    return center + radius * vec2(cos(phase), sin(phase));
}

// Function 77
void drawGame3D( inout vec4 color, vec2 uv, AppState s )
{   
    vec2 mo = iMouse.xy / iResolution.xy;
   
    vec2 bent = getBent();

    float fbm = fbm3( vec3( 1000.0 * iTime ) );
    float crash = step( 0.0, g_S.timeFailed ) * impulse( 2.0, max( 0.0, iTime - g_S.timeFailed ) * 6.0 );
    // camera	    
    float roll = -0.1 * bent.x;
    float arm = 3.5 + 0.2 * s.paceScale;
    float angleH = -0.5 * PI + 0.1 * bent.x;
    float height = 1.2 + bent.y + crash * fbm + 0.05 * g_S.paceScale * fbm;
    float fov = 2.0 - 0.5 * s.paceScale;
    
    vec3 ro = vec3( 0.0 );
    
    if ( s.timeFailed > 0.0 )
    {
        roll = mix( roll, 0.0, saturate( iTime - s.timeFailed ) );
        arm = mix( arm, 3.5, saturate( iTime - s.timeFailed ) );
        angleH += iTime - s.timeFailed;
    }
    
    if ( s.stateID == GS_SPLASH )
    {
        arm += 0.5 * sin( iTime ) * 0.5 + 0.5;
        roll = -0.1 * ( mo.x - 0.5 );
        angleH += 0.5 * ( mo.x - 0.5 );
        height += 0.5 * (mo.y - 0.5 );                
    }
    
    ro = vec3( arm * cos( angleH ), height, arm * sin( angleH ) );
    
#ifdef DEBUG_CAMERA    
    roll = 0.0;
    ro = vec3( 0.5 + 3.5 * cos( 12.0 * mo.x ), 0.5 + 4.0 * mo.y, -0.5 + 3.5 * sin( 12.0 * mo.x ) );
#endif        
    
    vec3 ta = vec3(
        0.0, 
        mix( 1.0, 0.5, step( 0.0, s.timeFailed ) * saturate( iTime - s.timeFailed ) ),
        0.0
    );

#ifdef CAM_STICKED    
    ro.x += s.playerPos.x;
    ta.x += s.playerPos.x;
#endif    
    
    // camera-to-world transformation
    mat3 ca = setCamera( ro, ta, roll );
   
    // ray direction
    vec3 rd = ca * normalize( vec3( uv.xy, fov ) );
    
    // render	
    vec4 col = render( ro, rd );
       
    color = col;
}

// Function 78
float horizontalPlayerCollide(vec2 p1, vec2 p2, float h) {
    vec2 s = (vec2(1) + vec2(.6, h)) / 2.;
    p2.y += h / 2.;
    return rectangleCollide(p1, p2, s);
}

// Function 79
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

// Function 80
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

// Function 81
vec4 SampleCharacter( uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vClampedCharUV = clamp(vCharUV, vec2(0.01), vec2(0.99));
    vec2 vUV = (vec2(iChPos) + vClampedCharUV) / 16.0f;

    vec4 vSample;
    
    float l = length( (vClampedCharUV - vCharUV) );

    // Skip texture sample when not in character boundary
    // Ok unless we have big font weight
    if ( l > 0.01f )
    {
        vSample.rgb = vec3(0);
		vSample.w = 2000000.0; 
    }
    else
    {
		vSample = textureLod( iChannelFont, vUV, 0.0 );    
        vSample.gb = vSample.gb * 2.0f - 1.0f;
        vSample.a -= 0.5f + 1.0/256.0;    
    }
        
    return vSample;
}

// Function 82
bool coll_player_grass( ivec4 player, ivec4 state ) {
    if( player.y > 12 * 16 + 4 ) {
        int pos_x = player.x - 16 ; //hack -16
        int home_ind = pos_x / 48 ;
        int home_center = home_ind * 48 + 8 + 8 ;
        if( abs( pos_x + 8 - home_center ) < 4 ) {
            if( flag( state.z, 1 << home_ind ) ) {
                return( true ) ;
            }
        } else {
            return( true ) ;
        }
    }
    return( false ) ;
}

// Function 83
void SpriteGameOver( inout vec3 color, float x, float y )
{
    float idx = 0.0;

    idx = y == 6.0 ? ( x <= 23.0 ? 6503998.0 : ( x <= 47.0 ? 4063359.0 : 4161399.0 ) ) : idx;
    idx = y == 5.0 ? ( x <= 23.0 ? 7831399.0 : ( x <= 47.0 ? 7536647.0 : 6752119.0 ) ) : idx;
    idx = y == 4.0 ? ( x <= 23.0 ? 8352007.0 : ( x <= 47.0 ? 7536647.0 : 6752054.0 ) ) : idx;
    idx = y == 3.0 ? ( x <= 23.0 ? 6123895.0 : ( x <= 47.0 ? 7536767.0 : 4161334.0 ) ) : idx;
    idx = y == 2.0 ? ( x <= 23.0 ? 4816743.0 : ( x <= 47.0 ? 7536647.0 : 3606300.0 ) ) : idx;
    idx = y == 1.0 ? ( x <= 23.0 ? 4288887.0 : ( x <= 47.0 ? 8323079.0 : 6752028.0 ) ) : idx;
    idx = y == 0.0 ? ( x <= 23.0 ? 4288862.0 : ( x <= 47.0 ? 4063359.0 : 6782728.0 ) ) : idx;

    idx = SPRITE_DEC_2( x, idx );
    idx = x >= 0.0 && x < 71.0 ? idx : 0.0;

    color = idx == 1.0 ? RGB( 255, 255, 255 ) : color;
}

// Function 84
vec4 SampleCharacterTex( sampler2D sFontSampler, uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vUV = (vec2(iChPos) + vCharUV) / 16.0f;
    return textureLod( sFontSampler, vUV, 0.0 );
}

// Function 85
vec4 draw_character( int n, ivec2 pos, ivec2 iu ) {
    vec3 v = vec3( 0 ) ;
    iu -= pos + ivec2(1,0) ;
    if( n > 0 && n <= NUM_FONT_CHARS && iINSIDE( iu, ivec2(0), ch_gfx_dim ) ) {
        int x = iu.x, y = iu.y ;
        n -- ;
        x += 5 * ( n & 3 ) ;
        float fx2 = exp2(-float(x+1)) ;

        n >>= 2 ;
        int part = y < 4 ? n * 2 + 1 : n * 2 ;
        y = 3 - (y&3) ;
        v = vec3( fract( get_bit_row( part, y ) * fx2 ) >= .5 ) ;
    }
    return( vec4( v, 1 ) ) ;
}

// Function 86
vec3 character( in bool v[35], in vec3 _color, in vec2 _st, in float _radius, in float x, in float y, float dist ){
    float sx = x;
    int col = 0;
    for (int i = 0; i < 35; i++){
        _color = mix( _color, redCircle(_color,_st,_radius,x,y), float( true==v[i] ));
        y += dist * float(col==4);
        col++;
        col -= 5 * int(col==5);
        x= sx+dist*float(col);
    }
    return _color;
}

// Function 87
vec2 DropLayer2(vec2 uv, float t) {
    vec2 UV = uv;
    
    uv.y += t*0.75;
    vec2 a = vec2(6., 1.);
    vec2 grid = a*2.;
    vec2 id = floor(uv*grid);
    
    float colShift = N(id.x); 
    uv.y += colShift;
    
    id = floor(uv*grid);
    vec3 n = N13(id.x*35.2+id.y*2376.1);
    vec2 st = fract(uv*grid)-vec2(.5, 0);
    
    float x = n.x-.5;
    
    float y = UV.y*20.;
    float wiggle = sin(y+sin(y));
    x += wiggle*(.5-abs(x))*(n.z-.5);
    x *= .7;
    float ti = fract(t+n.z);
    y = (Saw(.85, ti)-.5)*.9+.5;
    vec2 p = vec2(x, y);
    
    float d = length((st-p)*a.yx);
    
    float mainDrop = S(.4, .0, d);
    
    float r = sqrt(S(1., y, st.y));
    float cd = abs(st.x-x);
    float trail = S(.23*r, .15*r*r, cd);
    float trailFront = S(-.02, .02, st.y-y);
    trail *= trailFront*r*r;
    
    y = UV.y;
    float trail2 = S(.2*r, .0, cd);
    float droplets = max(0., (sin(y*(1.-y)*120.)-st.y))*trail2*trailFront*n.z;
    y = fract(y*10.)+(st.y-.5);
    float dd = length(st-vec2(x, y));
    droplets = S(.3, 0., dd);
    float m = mainDrop+droplets*r*trailFront;
    
    //m += st.x>a.y*.45 || st.y>a.x*.165 ? 1.2 : 0.;
    return vec2(m, trail);
}

// Function 88
void PlayerBulletBossCannonTest( inout vec4 playerBullet, inout vec4 bossCannon )
{
	if ( playerBullet.x > 0.0 && Collide( playerBullet.xy, BULLET_SIZE, bossCannon.xy, BOSS_CANNON_SIZE ) )
    {
		gHit			= vec4( playerBullet.xy, 0.0, 0.0 );
        playerBullet.x 	= 0.0;
		--bossCannon.w;
        if ( bossCannon.w < 0.0 )
        {
            gExplosion 		= vec4( bossCannon.xy + vec2( 0.0, BOSS_CANNON_SIZE.y * 0.5 ), 0.0, 0.0 );
            bossCannon.x 	= 0.0;
        }
    }
}

// Function 89
vec3 drawGame(ivec2 coord)
{
    GameData gd;
    loadGameData(gd);

    vec3 col;

    if (gd.gGameState == GAME_STATE_CAVE)
    {
        col = drawCave(coord, gd);
    }
    else
    {
        col = drawTitleScreen(coord, gd);
    }

    return col;
}

// Function 90
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

// Function 91
vec4 GameLose( ivec2 u )
{
    const int string[] = int[]( 0x50, 0xB7, 0xB1, 0xBD, 0xB5, 0x50, 0xBF, 0xA6, 0xB5, 0xA2,0x50 );
    
    //u.y -= iFrame/2; u.y = (u.y%40)+20;

    vec2 uv = vec2(u) / 40. - vec2(-float(string.length())*.25,-.5);
    float o = 0.;
    for ( int i=0; i < string.length(); i++ )
    {
        o += PrintCharacter( string[i], uv ); uv.x -= .5;
    }
    return mix( vec4(1,0,0,1), vec4(1,1,0,1), smoothstep(.48,.52,o) );
}

// Function 92
void PlayerBulletTurretTest( inout vec4 playerBullet, inout vec4 turret, inout vec4 turretState )
{
	if ( playerBullet.x > 0.0 && Collide( playerBullet.xy, BULLET_SIZE, turret.xy + vec2( 0.0, -TURRET_SIZE.y * 0.5 ), TURRET_SIZE ) )
    {
        gHit			= vec4( playerBullet.xy, 0.0, 0.0 );
        playerBullet.x 	= 0.0;
        
        --turretState.x;        
        if ( turretState.x <= 0.0 )
        {
			gExplosion = vec4( turret.xy, 0.0, 0.0 );
        	turret.x = 0.0;
        }
    }
}

// Function 93
void GameLoadState()
{
    vec3 state = LoadVec3(addrGameState);
    
 	gtime = state.x;
    time_scale = state.y;
    input_state = state.z;
}

// Function 94
void saveGameData(GameData gd, inout vec4 fragColor, ivec2 uvi)
{
    vec4 o;

    o.r = float(gd.gGameState);
    o.g = float(gd.gCave * (gd.gIsCaveInit ? -1 : 1));
    o.b = float(gd.gLevel);
    o.a = float(gd.gLives);
    saveValue(TX_GAME_DATA, o, fragColor, uvi);

    o = vec4(gd.gFrames.xy, gd.gScore, gd.gHighScore);
    saveValue(TX_GAME_FRAMES, o, fragColor, uvi);
}

// Function 95
bool Entity_IsPlayerTarget( Entity entity )
{
    if ( 	entity.iType == ENTITY_TYPE_PLAYER ||
       	 	entity.iType == ENTITY_TYPE_BARREL || 
       		entity.iType == ENTITY_TYPE_ENEMY )
    {
        return true;
    }
    
    return false;
}

// Function 96
vec4 draw_character( int n, ivec2 pos, ivec2 iu ) {
    vec4 v = vec4( 0, 0, 0, 1 ) ;
    iu -= pos + ivec2(1,0) ;
    if( n > 0 && n <= NUM_FONT_CHARS && iINSIDE( iu, ivec2(0), ivec2(5,7) ) ) {
        iu = ivec2( iu.x + ( n - 1 ) * 5, 128 + iu.y ) ;
        v = vec4( texelFetch( iChannel0, iu, 0 ).xxx, 1 ) ;
    }
    return( v ) ;
}

// Function 97
void spr_player_right(float f, float x, float y)
{
	spr_player_left(f, 15. - x, y);
}

// Function 98
void spr_player_up(float f, float x, float y)
{
	float c = 0.;
	if (f == 0. || f == 1.) {
		if (f == 1.) x = 15. - x;
		
		if (y == 0.) c = (x < 8. ? 21504. : 21.); if (y == 1.) c = (x < 8. ? 21760. : 85.);
		if (y == 2.) c = (x < 8. ? 21792. : 2133.); if (y == 3.) c = (x < 8. ? 21856. : 2389.);
		if (y == 4.) c = (x < 8. ? 21984. : 2901.); if (y == 5.) c = (x < 8. ? 24480. : 2805.);
		if (y == 6.) c = (x < 8. ? 32640. : 765.); if (y == 7.) c = (x < 8. ? 64960. : 895.);
		if (y == 8.) c = (x < 8. ? 22000. : 981.); if (y == 9.) c = (x < 8. ? 22000. : 3029.);
		if (y == 10.) c = (x < 8. ? 22464. : 3029.); if (y == 11.) c = (x < 8. ? 64832. : 2687.);
		if (y == 12.) c = (x < 8. ? 21824. : 341.); if (y == 13.) c = (x < 8. ? 24512. : 213.);
		if (y == 14.) c = (x < 8. ? 16320. : 60.); if (y == 15.) c = (x < 8. ? 3840. : 0.);
		
		float s = SELECT(x,c);
		if (s == 1.) fragColor = RGB(128.,208.,16.);
		if (s == 2.) fragColor = RGB(255.,160.,68.);
		if (s == 3.) fragColor = RGB(228.,92.,16.);
	}
	if (f == 2.) {
		if (y == 0.) c = (x < 8. ? 43584. : 2.); if (y == 1.) c = (x < 8. ? 43660. : 10.);
		if (y == 2.) c = (x < 8. ? 43676. : 42.); if (y == 3.) c = (x < 8. ? 43708. : 810.);
		if (y == 4.) c = (x < 8. ? 43636. : 986.); if (y == 5.) c = (x < 8. ? 43380. : 49365.);
		if (y == 6.) c = (x < 8. ? 26004. : 28901.); if (y == 7.) c = (x < 8. ? 22164. : 23897.);
		if (y == 8.) c = (x < 8. ? 43664. : 22362.); if (y == 9.) c = (x < 8. ? 43664. : 21978.);
		if (y == 10.) c = (x < 8. ? 43616. : 5498.); if (y == 11.) c = (x < 8. ? 21924. : 1493.);
		if (y == 12.) c = (x < 8. ? 43685. : 938.); if (y == 13.) c = (x < 8. ? 32789. : 362.);
		if (y == 14.) c = (x < 8. ? 0. : 1360.); if (y == 15.) c = (x < 8. ? 0. : 1360.);
		
		float s = SELECT(x,c);
		if (s == 1.) fragColor = RGB(228.,92.,16.);
		if (s == 2.) fragColor = RGB(128.,208.,16.);
		if (s == 3.) fragColor = RGB(255.,160.,68.);
	}
	
}

// Function 99
void drawGame3D( inout vec4 color, vec2 uv, AppState s )
{   
    vec2 mo = iMouse.xy / iResolution.xy;
   
    vec2 bent = getBent();

    float fbm = fbm3( vec3( 1000.0 * iTime ) );
    float crash = step( 0.0, g_S.timeFailed ) * impulse( 2.0, max( 0.0, iTime - g_S.timeFailed ) * 6.0 );
    // camera	    
    float roll = -0.1 * bent.x;
    float arm = 3.5 + 0.2 * s.paceScale;
    float angleH = -0.5 * PI + 0.1 * bent.x;
    float height = 1.3 + bent.y + crash * fbm + 0.05 * g_S.paceScale * fbm;
    float fov = 1.5 - 0.5 * s.paceScale;
    
    vec3 ro = vec3( 0.0 );
    
    if ( s.timeFailed > 0.0 )
    {
        roll = mix( roll, 0.0, saturate( iTime - s.timeFailed ) );
        arm = mix( arm, 3.5, saturate( iTime - s.timeFailed ) );
        angleH += iTime - s.timeFailed;
    }
    
    if ( s.stateID == GS_SPLASH )
    {
        arm += 0.5 * sin( iTime ) * 0.5 + 0.5;
        roll = -0.1 * ( mo.x - 0.5 );
        angleH += 0.5 * ( mo.x - 0.5 );
        height += 0.5 * (mo.y - 0.5 );                
    }
    
    ro = vec3( arm * cos( angleH ), height, arm * sin( angleH ) );
    
#ifdef DEBUG_CAMERA    
    roll = 0.0;
    ro = vec3( 0.5 + 3.5 * cos( 12.0 * mo.x ), 0.5 + 4.0 * mo.y, -0.5 + 3.5 * sin( 12.0 * mo.x ) );
#endif        
    
    vec3 ta = vec3(
        0.0, 
        mix( 1.0, 0.5, step( 0.0, s.timeFailed ) * saturate( iTime - s.timeFailed ) ),
        0.0
    );

#ifdef CAM_STICKED    
    ro.x += s.playerPos.x;
    ta.x += s.playerPos.x;
#endif    
    
    // camera-to-world transformation
    mat3 ca = setCamera( ro, ta, roll );
   
    // ray direction
    vec3 rd = ca * normalize( vec3( uv.xy, fov ) );
    
    // render	
    vec4 col = render( ro, rd );
       
    color = col;
}

// Function 100
void GameRestart(float state)
{
    GameSetState(state);
    gGameSeed             = iTime;
    gPlayerCoords         = vec2(0.5, 0.5);
    gPlayerNextCoords     = vec2(0.5, 0.5);
    gPlayerMotionTimer    = 0.0;
    gPlayerRotation       = 0.0;
    gPlayerNextRotation   = 0.0;
    gPlayerScale          = 1.0;
    gPlayerVisualCoords   = vec3(gPlayerCoords, 0.0).xzy;
    gPlayerVisualRotation = 0.0;
    gPlayerDeathCause     = 0.0;
    gScore                = 1.0;
}

// Function 101
void drawGameFlat( inout vec4 color, vec2 p, AppState s )
{
    // game
	vec2 p0 = p;    
    // float cameraAnim = smoothstep(-0.5, 0.5, sin(iTime) );
    float cameraAnim = 0.0;
	p0 *= mix( 5.0, 10.0, cameraAnim );		// scale field of view
    p0.x += 0.25;							// fix track centering
    p0.y += mix( 2.0, 8.0, cameraAnim );	// move camera pos
    p0.y += s.playerPos.y;
    
    float playerCellID = floor( s.playerPos.y );
    float sPlayer = length( p0 - s.playerPos ) - 0.25;
           
    vec2 p1 = p0;
    p1.y += 2.0 * s.playerPos.y;
    color.rgb = mix( vec3( 1.0 ), color.rgb, smoothstep( 1.5, 1.75, abs( p1.x - 0.5 ) ) );
    color.rgb = mix( texture( iChannel2, fract( p1 ) ).rgb, color.rgb, 0.5 );
       
	// COIN start
    float cellID = floor( p0.y );
    float cellCoinRND = hash11( cellID + g_S.seed );					// skip rnd obstacle every second cell to make room for driving    
    cellCoinRND *= mix( 1.0, -1.0, step( mod( cellID, 4.0 ), 1.5 ) );	// gaps in coin placing: 2 gaps, 2 coins
    cellCoinRND = mix( cellCoinRND, -1.0, step (cellID, 5.0 ) );		// head start
    float cellCoinCol = floor( 3.0 * cellCoinRND );
       
    if ( cellCoinRND >= 0.0 )
    {
        if ( cellID > playerCellID )
           	drawCoin( color.rgb, p0, vec2( cellCoinCol, cellID ) );
        
        if ( cellID == playerCellID && s.coin0Taken < 0.5 )
            drawCoin( color.rgb, p0, vec2( cellCoinCol, cellID ) );
        
        if ( cellID == playerCellID - 1.0 && s.coin1Taken < 0.5 )
            drawCoin( color.rgb, p0, vec2( cellCoinCol, cellID ) );
        
        if ( cellID == playerCellID - 2.0 && s.coin2Taken < 0.5 )
            drawCoin( color.rgb, p0, vec2( cellCoinCol, cellID ) );
       
        if ( cellID == playerCellID - 3.0 && s.coin3Taken < 0.5 )
            drawCoin( color.rgb, p0, vec2( cellCoinCol, cellID ) );
    }    
// COIN end

// OBSTACLE start
    float cellObsRND = hash11( 100.0 * cellID + g_S.seed );		// skip rnd obstacle every second cell to make room for driving
    cellObsRND *= mix( 1.0, -1.0, step( mod( cellID, 3.0 ), 1.5 ) );
    cellObsRND = mix( cellObsRND, -1.0, step( cellID, 7.0) );	// head start
    float cellObsCol = floor( 3.0 * cellObsRND );
    
	if ( cellObsRND >= 0.0 && cellObsCol != cellCoinCol )
    {        
    	float sObstacle = length(
            p0
            -vec2( 0.0, cellID )		// cell pos
            +vec2( 0.5, -0.5 )			// move to cell center
            -vec2( cellObsCol, 0.0 )	// move to column
        ) - 0.25;						// radius of coin
        
    	color.rgb = mix( vec3( 1.0, 0.0, 0.0 ), color.rgb, smoothstep( 0.0, 0.1, sObstacle ) );
        
        vec2 obstaclePos = -vec2( 0.0, cellID )			// cell pos
            				+vec2( 0.5, -0.5 )			// move to cell center
            				-vec2( cellObsCol, 0.0 );	// move to column

        float distObstaclePlayer = length( obstaclePos + s.playerPos );
        
        if ( distObstaclePlayer < 0.5 ) 
        {
            color.rgb += vec3( 0.5 );
        }
    }
    
    color.rgb = mix( vec3( 0.0, 1.0, 0.0 ), color.rgb, smoothstep( 0.0, 0.1, sPlayer ) );

// OBSTACLE end        

}

// Function 102
void healPlayer(float amount, inout float health, inout float invul, inout float maxHealth) {
 
    attackPlayer(-amount, health, invul, maxHealth);
    
}

// Function 103
vec4 writeGameData(vec4 col, vec2 fragCoord, GameData data) {
    col = writeData(col, fragCoord.xy, 0, data.shipPos);
    col = writeData(col, fragCoord.xy, 1, data.shipLastPos);
    col = writeData(col, fragCoord.xy, 2, data.shipAccel);
    col = writeData(col, fragCoord.xy, 3, data.shipVelocity);
    col = writeData(col, fragCoord.xy, 4, data.shipTheta);
    col = writeData(col, fragCoord.xy, 5, data.touchStart);
    return col;
}

// Function 104
vec4 GameLose( vec2 u, vec4 o )
{
    o = vec4(dot(o,vec4(.2126,.7152,.0722,0)));
    
    const int string[] = int[]( 0x50, 0xB7, 0xB1, 0xBD, 0xB5, 0x50, 0xBF, 0xA6, 0xB5, 0xA2,0x50 );
    
    u -= .5;
    u.y *= 9./16.;
    u *= 256.;
    
    //u.y -= iFrame/2; u.y = (u.y%40)+20;

    vec2 uv = u / 10. - vec2(-float(string.length())*.25,-.5);
    float t = 0.;
    for ( int i=0; i < string.length(); i++ )
    {
        t += PrintCharacter( string[i], uv ); uv.x -= .5;
    }
    return mix( o*.3, vec4(1,0,0,1), smoothstep(.48,.52,t) );
}

// Function 105
vec4 GameWin( ivec2 u )
{
    float o = 0.;

    const int string[] = int[]( 0xB3, 0xBF, 0xBE, 0xB7, 0xA2, 0xB1, 0xA4, 0xA5, 0xBC, 0xB1, 0xA4, 0xB9, 0xBF, 0xBE, 0xA3 );
    vec2 uv = vec2(u) / 25. - vec2(-float(string.length())*.25,-1.1);

    float a = .0;
    for ( int i=0; i < string.length(); i++ )
    {
        a += PrintCharacter( string[i], uv ); uv.x -= .5;
    }
    o += smoothstep(.48,.52,a);
    
    const int string2[] = int[]( 0xA9, 0xBF, 0xA5, 0x50, 0xA7, 0xB9, 0xBE, 0xD1 );
    uv = vec2(u) / 50. - vec2(-float(string2.length())*.25,.0);
    float b = .0;
    for ( int i=0; i < string2.length(); i++ )
    {
        b += PrintCharacter( string2[i], uv ); uv.x -= .5;
    }
    o += smoothstep(.49,.51,b);
    
    vec4 bg = max(sin(float(u.y+iFrame/2)*vec4(.2,.3,.5,1)*6.283/10.),vec4(0));
    return mix( bg, 1.-bg.yzxw, o );
}

// Function 106
void spr_player_left(float f, float x, float y)
{
	float c = 0.;
	if (f == 0.) {
		if (y == 0.) c = (x < 8. ? 16384. : 21.); if (y == 1.) c = (x < 8. ? 43520. : 341.);
		if (y == 2.) c = (x < 8. ? 43648. : 5590.); if (y == 3.) c = (x < 8. ? 43520. : 22010.);
		if (y == 4.) c = (x < 8. ? 63240. : 17918.); if (y == 5.) c = (x < 8. ? 64504. : 1726.);
		if (y == 6.) c = (x < 8. ? 65288. : 687.); if (y == 7.) c = (x < 8. ? 65288. : 85.);
		if (y == 8.) c = (x < 8. ? 23224. : 2389.); if (y == 9.) c = (x < 8. ? 22200. : 10879.);
		if (y == 10.) c = (x < 8. ? 22152. : 10943.); if (y == 11.) c = (x < 8. ? 22536. : 10941.);
		if (y == 12.) c = (x < 8. ? 43016. : 1686.); if (y == 13.) c = (x < 8. ? 21504. : 5461.);
		if (y == 14.) c = (x < 8. ? 0. : 170.); if (y == 15.) c = (x < 8. ? 32768. : 170.);
	}
	if (f == 1.) {
		if (y == 0.) c = 0.; if (y == 1.) c = (x < 8. ? 20480. : 5.);
		if (y == 2.) c = (x < 8. ? 27264. : 85.); if (y == 3.) c = (x < 8. ? 43680. : 1397.);
		if (y == 4.) c = (x < 8. ? 43648. : 5502.); if (y == 5.) c = (x < 8. ? 48576. : 4479.);
		if (y == 6.) c = (x < 8. ? 48892. : 431.); if (y == 7.) c = (x < 8. ? 65480. : 171.);
		if (y == 8.) c = (x < 8. ? 32712. : 21.); if (y == 9.) c = (x < 8. ? 65208. : 421.);
		if (y == 10.) c = (x < 8. ? 64952. : 682.); if (y == 11.) c = (x < 8. ? 62856. : 1706.);
		if (y == 12.) c = (x < 8. ? 22024. : 1450.); if (y == 13.) c = (x < 8. ? 43592. : 10581.);
		if (y == 14.) c = (x < 8. ? 21920. : 10837.); if (y == 15.) c = 2688.;
	}
	if (f == 2.) {
		if (y == 0.) c = 0.; if (y == 1.) c = (x < 8. ? 21504. : 1.);
		if (y == 2.) c = (x < 8. ? 23200. : 21.); if (y == 3.) c = (x < 8. ? 27304. : 93.);
		if (y == 4.) c = (x < 8. ? 43680. : 95.); if (y == 5.) c = (x < 8. ? 61296. : 351.);
		if (y == 6.) c = (x < 8. ? 61375. : 1387.); if (y == 7.) c = (x < 8. ? 65520. : 1066.);
		if (y == 8.) c = (x < 8. ? 24572. : 21.); if (y == 9.) c = (x < 8. ? 43772. : 90.);
		if (y == 10.) c = (x < 8. ? 43760. : 106.); if (y == 11.) c = (x < 8. ? 43584. : 362.);
		if (y == 12.) c = (x < 8. ? 38304. : 1370.); if (y == 13.) c = (x < 8. ? 27296. : 10581.);
		if (y == 14.) c = (x < 8. ? 21864. : 10837.); if (y == 15.) c = (x < 8. ? 170. : 2688.);
	}
	
	float s = SELECT(x,c);
	if (s == 1.) fragColor = RGB(128.,208.,16.);
	if (s == 2.) fragColor = RGB(228.,92.,16.);
	if (s == 3.) fragColor = RGB(255.,160.,68.);
}

// Function 107
bool IsControlCharacter( int char )
{
    if ( char >= 128 )
        return true;
    
    return false;
}

// Function 108
void PlayerBulletSniperTest( inout vec4 playerBullet, inout vec4 sniper )
{
	if ( playerBullet.x > 0.0 && Collide( playerBullet.xy, BULLET_SIZE, sniper.xy, SNIPER_SIZE ) )
    {
        gExplosion		= vec4( sniper.xy + vec2( 0.0, SNIPER_SIZE.y * 0.5 ), 0.0, 0.0 );
        gHit		  	= vec4( playerBullet.xy, 0.0, 0.0 );
		sniper.x		= 0.0;
        playerBullet.x 	= 0.0;
    }
}

// Function 109
void fakePlayers() {
    // Colors taken from http://www.materialui.co/colors (500).
    playersColor[0] = vec3(0.957, 0.263, 0.212);  // red
    playersColor[1] = vec3(0.129, 0.588, 0.953);  // blue
    playersColor[2] = vec3(0.298, 0.686, 0.314);  // green
    playersColor[3] = vec3(0.612, 0.153, 0.69);   // purple
    
    const float timeScale = 0.2;
    playersPos[0] = initPlayerPos(vec2(0.2, 0.1), 2.0, PI, 1.0 * timeScale);
    playersPos[1] = initPlayerPos(vec2(-0.2, 0.1), 1.0, 3.0 * PI * 0.5, 3.0 * timeScale);
    playersPos[2] = initPlayerPos(vec2(-0.2, -0.1), 3.0, 0.0, 0.5 * timeScale);
    playersPos[3] = initPlayerPos(vec2(0.2, -0.1), 2.0, 3.0 * PI / 4.0, 1.0 * timeScale);
}

// Function 110
Shape character(vec3 c){
  Shape shape;
  shape.dist = 1000.; // Draw Distance
  shape.color = vec4(1.); // Initial Color
    
  vec3 w = c; //Torus around sphere of face
  vec3 h = c; //Head is the sphere in the middle
  vec3 l = c; //Links/rays around the sun in polar
  vec3 e = c; //Ellipse eyes for blinking
  vec3 m = c; //Mouth (a box)
  vec3 i = c; //Small spheres inside ellipse for eyes
    
    //Making the torus face the camera
    vec4 wColor = vec4(1., .8, 0., 1.); //Yellow-Orange?
    w.yz *= rot(radians(90.));
    float wheel = sdTorus(w+vec3(0., 0., 0.), vec2(.75, .4));
    
    //Making the head
    vec4 hColor = vec4(1., .6, 0., 1.); //Dark Orange-y Color
    float head = sphere(h, .75);
    
    //Rotating and duplicating the links into polar
    vec4 lColor = vec4(1., sin(iTime)*.2+.7, 0., 1.);
    l.xy *= rot(sin(iTime)*.3);
    pModPolar(l.yx, 6.);
    float links = fBox(l-vec3(1., abs(sin(iTime*.75))*1.5, -.1), vec3(.9, .6, .0));
    
    //Molding the links/rays into the torus to make a connected figure
    float unit = fOpUnionRound(wheel, links, .2);
    
    //Oscillate blinking eyes & mirror them
    vec4 eColor = vec4(1., 0., 0., 1.); //Red
    e.x = abs(e.x)-.3;
    float eye = sdEllipsoid(e+vec3(.0, -.1, 0.7), vec3(.23, abs(sin(iTime))*.15, .6));
    
    //Curve the mouth to look less creepy
    m.xy *= rot(2.*m.x);
    float mouth = fBox(m+vec3(0., 0.35, .6), vec3(.1, .02, .2));
    
    //Doubling the spheres to make eyes
    vec4 iColor = vec4(1., 1., 0., 1.);
    i.x = abs(i.x)-.3;
    float iris = sphere(i+vec3(.03, -.1, .5), .1);
    
    shape.dist = min(unit, head);
    shape.dist = max(shape.dist, -eye); //Cuts out eye part and mouth
    shape.dist = max(shape.dist, -mouth);
    shape.dist = min(shape.dist, iris);
    
    shape.color = mix(hColor, wColor*1.4,
                      mixColors(wheel, head, .6));
    shape.color = mix(shape.color, eColor,
                      mixColors(eye, shape.dist, .9));
    shape.color = mix(shape.color, lColor,
                      mixColors(links, shape.dist, .1));
    shape.color = mix(shape.color, iColor*1.6,
                      mixColors(iris, shape.dist, .1));
    
  return shape;
}

// Function 111
vec2 characterShowcase( float time, vec3 p ) {
    vec2 d = vec2( 1e33, -1. );
        
    float tWizard = MAP_11_01( cos( time * 10. + 0. ) );
    float tTank   = MAP_11_01( cos( time *  5. + 1. ) );
    float tArcher = MAP_11_01( cos( time * 14. + 2. ) );
    
    float xWizard = .6 + .2 * abs( cos( tWizard ) );

    p.y += .5;
    d = minnow( d, wizard( tWizard, p + characterPosition( tWizard, .6 ) ) );
    d = minnow( d, tank(   tTank,   p - characterPosition( tTank, .0 ) ) );
    d = minnow( d, archer( tArcher, p - characterPosition( tArcher, .6 ) ) );
    p.y -= .5;
  
    return d;
}

// Function 112
vec4 characterField(vec3 p, inout float minDist )
{
	vec4 jediField = vec4(maxDist);
    vec4 sithField = vec4(maxDist);
    float jediMinDist = MAX_DIST;
    float sithMinDist = MAX_DIST;
    jediField = fStickman(jediData.invStickmanRot*(p - jediData.stickmanPos), jediData, jediFieldData, jediMinDist);
#ifdef TWO_EGGS    
    sithField = fStickman(sithData.invStickmanRot*(p - sithData.stickmanPos), sithData, sithFieldData, sithMinDist);
#endif    
    minDist = min(jediMinDist, sithMinDist);
    return jediField*step(jediMinDist, minDist) + sithField*step(sithMinDist, minDist);
}

// Function 113
bool Entity_IsAlivePlayer( Entity entity )
{
    if ( entity.iType == ENTITY_TYPE_PLAYER )
    {
        if ( entity.fHealth > 0.0 )
        {
            return true;
        }
    }
    
    return false;
}

// Function 114
void initNewGame(float time) {
    gParatroopersLeft = vec4(0);
    gParatroopersRight = vec4(0);
    gScore = 0.;
    gLastShot = time;
    gGameOverTime = 0.;
    
    for (int i=0; i<MAX_BULLETS; i++) {
    	gBulletData[i].z = -20.;
    }
    for (int i=0; i<MAX_PARATROOPERS; i++) {
    	gParatrooperData[i].x = -20.;
    }
    
    initNewRound(GAME_HELICOPTER, time);
}

// Function 115
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

// Function 116
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

// Function 117
void GameSetState(float state)
{
    gGameState     = state;
    gGameStateTime = 0.0;
}

// Function 118
void PlayerBulletPowerUpTest( inout vec4 playerBullet )
{
	if ( playerBullet.x > 0.0 && gPowerUpState.x == STATE_RUN && Collide( playerBullet.xy, BULLET_SIZE, gPowerUp.xy, POWER_UP_SIZE ) )
    {
		gHit			= vec4( playerBullet.xy, 0.0, 0.0 );
        gExplosion 		= vec4( gPowerUp.xy + vec2( 0.0, POWER_UP_SIZE.y * 0.5 ), 0.0, 0.0 );        
        playerBullet.x 	= 0.0;
        gPowerUpState.x = STATE_JUMP;
        gPowerUp.z		= 1.0;
    }
}

// Function 119
float character(float color, float background, int character, vec2 position, float size, vec2 uv)
{
    if((uv.x > position.x && uv.x < position.x + size) && (uv.y > position.y && uv.y < position.y + size))
    {
        ivec2 pixel = ivec2(ceil((uv.x-position.x)/size*5.0)-1.0, ceil((1.0-(uv.y-position.y)/size)*5.0)-1.0);
        int bit_index = pixel.y*5 + pixel.x;
        int bit = (CHARS[character] >> (24 - bit_index))&1;
        if(bit > 0)
            return color;
    }
    return background;
}

// Function 120
AppState updateGame( AppState s, float isDemo )
{
    if ( isDemo > 0.0 )
    {
        s.timeAccumulated += 4.5 * iTimeDelta;
    	s.playerPos.y = 22.5 * s.timeAccumulated;
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

// Function 121
void loadGameData(inout GameData gd)
{
    vec4 v;

    v = loadValue(TX_GAME_DATA);
    gd.gGameState = int(v.r);
    gd.gCave = abs(int(v.g));
    gd.gIsCaveInit = v.g < 0.0;
    gd.gLevel = int(v.b);
    gd.gLives = int(v.a);

    v = loadValue(TX_GAME_FRAMES);
    gd.gFrames = ivec4(int(v.r), int(v.g), 0, 0);
    gd.gScore = int(v.b);
    gd.gHighScore = int(v.a);
}

// Function 122
bool pix_coll_player_car_track( ivec4 player, ivec2 track_off ) {
    //test of player and cars track, single pixel collision within car track kills player
    const ivec2 track_dim = ivec2( 256, 16 ) ;
    ivec2 d, pos = player.xy, off = track_off ;
    if( ! iRECTS_COLLIDE( ivec4( pos, dim_player ), ivec4( 0, off.y, track_dim ) ) ) {
        return( false ) ;
    }
    int frame = int( player.w >= END_JUMP_T ) ;
    int dir = player.z & 0x3 ;
    for( d.y = 0 ; d.y < dim_player.y ; ++ d.y ) {
        if( iINSIDE( pos.y + d.y, off.y, off.y + track_dim.y ) ) {
            for( d.x = 0 ; d.x < dim_player.x ; ++ d.x ) {
                vec4 dummy ;
                if( draw_player( frame, dir, ivec2( 0 ), dummy, d ).a > 0. ) {
                    vec4 b = texelFetch( iChannel0, ivec2( pos.x+off.x+d.x, pos.y+d.y ) & 0xff, 0 ) ;
                    if( b != col_road ) {
                        return( true ) ;
                    }
                }
            }
        }
    }
    return( false ) ;
}

// Function 123
float player(vec2 p)
{
    vec4 player = get(vPlayer);
    float s = 0.5;
    
    p = rot(p - player.xy, player.z) - up * 5.0;
    float r = rect(p, vec2(15.0, 30.0) * s);
    float r2 = rect(p + up * 5.0, vec2(10.0, 14.0) * s);
    float r3 = rect(p + up * 6.0, vec2(8.0, 12.0) * s);
    float t = triangle(p + up * 30.0 * s, PI / 8.0, 25.0 * s);
    return diff(sum(diff(r, r2), r3), -t);
}

// Function 124
void adjustPlayerCameras(float uniformCamZoom) {
    
    for(int i = 0; i < N_PLAYERS; i++) {
        playersBgColor[i] = playersColor[i];
    }
    float mergeDistMin = 0.6 * CAMERA_ZOOM_MIN;
    float mergeDistMax = 0.8 * CAMERA_ZOOM_MIN;
    for (int k = 0; k < 2; k++) {
        // Moving cameras closer to each other.
        for(int i = 0; i < N_PLAYERS; i++) {
            for(int j = 0; j < N_PLAYERS; j++) {
                if (i < j) {
                    vec2 camPosI = playersCam[i].xy;
                    vec2 camPosJ = playersCam[j].xy;
                    float camDistScreen = length(camPosI - camPosJ) * uniformCamZoom;
                    float mergeWeight = 0.5 * (1.0 - pow(smoothstep(mergeDistMin, mergeDistMax, camDistScreen), 4.0));
                    playersCam[i].xy = mix(camPosI, camPosJ, mergeWeight);
                    playersCam[j].xy = mix(camPosJ, camPosI, mergeWeight);

                    vec3 bgColorI = playersBgColor[i];
                    vec3 bgColorJ = playersBgColor[j];
                    playersBgColor[i] = mix(bgColorI, bgColorJ, mergeWeight);
                    playersBgColor[j] = mix(bgColorJ, bgColorI, mergeWeight);
                }            
            }
        }
        mergeDistMin *= 0.5;
        mergeDistMax *= 0.5;
    }
}

// Function 125
SceneResult Character_GetDistance( vec3 vPos )
{
    SceneResult result = SceneResult( kMaxTraceDist, MAT_BG, vec3(0.0) );


    vec3 vLeftLeg = LegDist( vPos, g_scene.pose.leftLeg );
    vec3 vRightLeg = LegDist( vPos, g_scene.pose.rightLeg );
    vec3 vLeftArm = ArmDist( vPos, g_scene.pose.leftArm );
    vec3 vRightArm = ArmDist( vPos, g_scene.pose.rightArm );
    vec3 vTorsoDist = TorsoDistance( 
        vPos,
        g_scene.pose.leftLeg.vHip, 
        g_scene.pose.rightLeg.vHip,
        g_scene.pose.leftArm.vShoulder,
        g_scene.pose.rightArm.vShoulder,
    	g_scene.charDef.fShoulder, g_scene.charDef.fHip);
        
    vTorsoDist.y += 1.0;
    vLeftArm.y += 2.0;
    vRightArm.y += 2.0;            
    vLeftArm.z += 1.0;
    vLeftLeg.z += 1.0;
        
    vec3 vLimbDist = vec3(10000.0);
    vLimbDist = BodyCombine3( vLimbDist, vLeftLeg );
    vLimbDist = BodyCombine3( vLimbDist, vRightLeg );
    vLimbDist = BodyCombine3( vLimbDist, vLeftArm );
    vLimbDist = BodyCombine3( vLimbDist, vRightArm );        
    vec3 vCharacterDist = BodyCombine3( vLimbDist, vTorsoDist );

    //vCharacterDist.x -= fbm( vLimbDist.xy * 10., 0.9 ) * 2.0;
    
    float fNeckSize = 1.0;
    float fNeckLen = 3.0;

    vec3 vNeckBase = (g_scene.pose.leftArm.vShoulder + g_scene.pose.rightArm.vShoulder) * 0.5;
    vec3 vNeckTop = vNeckBase + g_scene.pose.vHeadUp * fNeckLen;
    vec3 vNeckDist = Segment3( vPos, vNeckBase, vNeckTop, fNeckSize, fNeckSize );
    
    float fHead1 = g_scene.charDef.fHead1;
    float fHead2 = g_scene.charDef.fHead2;
    float fHeadTop = 6.0;
    float fHeadChin = 2.0;
    
    vec3 vHeadBase = vNeckBase + g_scene.pose.vHeadUp * ( fHeadChin + fHead2);
    vec3 vHead2 = vHeadBase  + g_scene.pose.vHeadFd * (fHead2 * .5);
    vec3 vHead1 = vHeadBase + g_scene.pose.vHeadUp * (fHeadTop - fHead1);
    
    vec3 vHeadDist = Segment3( vPos, vHead1, vHead2, fHead1, fHead2 );
    vHeadDist = SmoothMin3( vHeadDist, vNeckDist, 0.5 );

    vec3 vNosePos = vHead1 + g_scene.pose.vHeadFd * fHead1 * 1.2 - g_scene.pose.vHeadUp * 1.5;
    float fNoseDist = length( vPos - vNosePos ) - 1.;
    vHeadDist.x = min( vHeadDist.x, fNoseDist ); // keep material


    vec3 vEyePerp = normalize( cross(g_scene.pose.vHeadFd, g_scene.pose.vHeadUp) );

    vec3 vEyePos1 = vHead1 + g_scene.pose.vHeadFd * fHead1 + vEyePerp * 1.5;
    float fEyeDist1 = length( vPos - vEyePos1 ) - 1.;

    vec3 vEyePos2 = vHead1 + g_scene.pose.vHeadFd * fHead1 - vEyePerp * 1.5;
    float fEyeDist2 = length( vPos - vEyePos2 ) - 1.;
    
    float fEyeDist = min( fEyeDist1, fEyeDist2 );
        
    vHeadDist.x = max( vHeadDist.x, -(fEyeDist - 0.2) );
    
    
    //vCharacterDist = BodyCombine3( vCharacterDist, vHeadDist );
    result = Scene_Union( result, SceneResult( vHeadDist.x, MAT_HEAD, vHeadDist.yzz ) );    
    
    
    
    result = Scene_Union( result, SceneResult( fEyeDist, MAT_EYE, g_scene.charDef.vCol ) );    
        
    result = Scene_Union( result, SceneResult( vCharacterDist.x, MAT_CHARACTER, vCharacterDist.yzz ) );    

         
    return result;
}

// Function 126
vec4 Character(vec2 p)
{
    vec4 legs = vec4(0.0);
    const vec4 skinColor = vec4(0.9 * 1.2, 0.6 * 1.2, 0.3 * 1.2, 1.0);

    vec2 tp = Translate(Rotate(Translate(p, vec2(0.0, -0.4)), sin(iTime * 4.0) * PI / 4.0 * gameState.movementSpeed.x), vec2(0.0, 0.3));
    float rleg = step(Rect(tp, vec2(0.1, 0.3)), 0.001);
    legs = mix(legs, vec4(skinColor.rgb * rleg * 0.97, 1.0), rleg);
    
    vec2 ltp = Translate(Rotate(Translate(p, vec2(0.0, -0.4)), sin(-iTime * 4.0) * PI / 4.0 * gameState.movementSpeed.x), vec2(0.0, 0.3));
    float lleg = step(Rect(ltp, vec2(0.1, 0.3)), 0.001);
    legs = mix(legs, skinColor * lleg, lleg);
    
    float butt = step(Circle(Translate(p, vec2(0.07 * -sign(gameState.movementSpeed.x), -0.40)), 0.13), 0.001);
    float butt2 = step(Circle(Translate(p, vec2(0.1 * -sign(gameState.movementSpeed.x), -0.40)), 0.14), 0.001);
    legs = mix(legs, vec4(skinColor.rgb * butt2 * 0.9, 1.0), butt2);
    legs = mix(legs, skinColor * butt, butt);
    
    float h = step(Circle(Translate(p, vec2(0.0, 0.5)), 0.4), 0.001);
    float eye = step(Circle(Translate(p, vec2(0.2 * sign(gameState.movementSpeed.x), 0.6)), 0.07), 0.001);
    
    vec2 itp = Translate(Rotate(Translate(p, vec2(0.2 * sign(gameState.movementSpeed.x), 0.6)), sin(iTime * 3.0 + 1.21)), vec2(0.0, -0.02));
    float iris = step(Circle(itp, 0.06), 0.001);
    vec4 head = skinColor * h;
    head = mix(head, vec4(eye), eye);
    head = mix(head, vec4(0.0, 0.0, 0.0, 1.0) * iris, iris);
    
    float b = step(Rect(p, vec2(0.1, 0.4)), 0.001);
    vec4 body = skinColor * b;
    body = mix(body, legs, legs.a);
    body = mix(body, head, head.a);
    
    return body;
}

// Function 127
vec4 SampleCharacter( sampler2D sFontSampler, uint iChar, vec2 vCharUV )
{
    uvec2 iChPos = uvec2( iChar % 16u, iChar / 16u );
    vec2 vClampedCharUV = clamp(vCharUV, vec2(0.01), vec2(0.99));
    vec2 vUV = (vec2(iChPos) + vClampedCharUV) / 16.0f;

    vec4 vSample;
    
    float l = length( (vClampedCharUV - vCharUV) );

    // Skip texture sample when not in character boundary
    // Ok unless we have big font weight
    if ( l > 0.01f )
    {
        vSample.rgb = vec3(0);
		vSample.w = 2000000.0; 
    }
    else
    {
		vSample = textureLod( sFontSampler, vUV, 0.0 );    
        vSample.gb = vSample.gb * 2.0f - 1.0f;
        vSample.a -= 0.5f + 1.0/256.0;    
    }
        
    return vSample;
}

// Function 128
void DrawGame( inout vec3 color, float time, float pixelX, float pixelY, float screenWidth, float screenHeight )
{
    float mushroomPauseStart 	= 16.25;    
    float mushroomPauseLength 	= 2.0;    
    float flagPauseStart		= 38.95;
    float flagPauseLength		= 1.5;

    float cameraP1		= clamp( time - mushroomPauseStart, 0.0, mushroomPauseLength );
    float cameraP2		= clamp( time - flagPauseStart,     0.0, flagPauseLength );
    float cameraX 		= floor( min( ( time - cameraP1 - cameraP2 ) * MARIO_SPEED - 240.0, 3152.0 ) );
    float worldX 		= pixelX + cameraX;
    float worldY  		= pixelY - 8.0;
    float tileX			= floor( worldX / 16.0 );
    float tileY			= floor( worldY / 16.0 );
    float tile2X		= floor( worldX / 32.0 );
    float tile2Y		= floor( worldY / 32.0 );    
    float worldXMod16	= mod( worldX, 16.0 );
    float worldYMod16 	= mod( worldY, 16.0 );


    // default background color
    color = RGB( 92, 148, 252 );

    
    // draw hills
    float bigHillX 	 = mod( worldX, 768.0 );
    float smallHillX = mod( worldX - 240.0, 768.0 );
    float hillX 	 = min( bigHillX, smallHillX );
    float hillY      = worldY - ( smallHillX < bigHillX ? 0.0 : 16.0 );
    SpriteHill( color, hillX, hillY );


    // draw clouds and bushes
	float sc1CloudX = mod( worldX - 296.0, 768.0 );
    float sc2CloudX = mod( worldX - 904.0, 768.0 );
    float mcCloudX  = mod( worldX - 584.0, 768.0 );
    float lcCloudX  = mod( worldX - 440.0, 768.0 );    
    float scCloudX  = min( sc1CloudX, sc2CloudX );
    float sbCloudX 	= mod( worldX - 376.0, 768.0 );
    float mbCloudX  = mod( worldX - 664.0, 768.0 );  
	float lbCloudX  = mod( worldX - 184.0, 768.0 );
    float cCloudX	= min( min( scCloudX, mcCloudX ), lcCloudX );
    float bCloudX	= min( min( sbCloudX, mbCloudX ), lbCloudX );
    float sCloudX	= min( scCloudX, sbCloudX );
    float mCloudX	= min( mcCloudX, mbCloudX );
    float lCloudX	= min( lcCloudX, lbCloudX );
    float cloudX	= min( cCloudX, bCloudX );
    float isBush	= bCloudX < cCloudX ? 1.0 : 0.0;
    float cloudSeg	= cloudX == sCloudX ? 0.0 : ( cloudX == mCloudX ? 1.0 : 2.0 );
    float cloudY	= worldY - ( isBush == 1.0 ? 8.0 : ( ( cloudSeg == 0.0 && sc1CloudX < sc2CloudX ) || cloudSeg == 1.0 ? 168.0 : 152.0 ) );
	if ( cloudX >= 0.0 && cloudX < 32.0 + 16.0 * cloudSeg )
    {
        if ( cloudSeg == 1.0 )
        {
        	cloudX = cloudX < 24.0 ? cloudX : cloudX - 16.0;
        }
        if ( cloudSeg == 2.0 )
        {
        	cloudX = cloudX < 24.0 ? cloudX : ( cloudX < 40.0 ? cloudX - 16.0 : cloudX - 32.0 );
        }
        
    	SpriteCloud( color, cloudX, cloudY, isBush );
    }

    
    
    // draw flag pole
    if ( worldX >= 3175.0 && worldX <= 3176.0 && worldY <= 176.0 )        
    {
        color = RGB( 189, 255, 24 );
    }
    
    // draw flag
    float flagX = worldX - 3160.0;
    float flagY = worldY - 159.0 + floor( 122.0 * clamp( ( time - 39.0 ) / 1.0, 0.0, 1.0 ) );
    if ( flagX >= 0.0 && flagX <= 15.0 )
    {
    	SpriteFlag( color, flagX, flagY );
    }     
    
    // draw flagpole end
    float flagpoleEndX = worldX - 3172.0;
    float flagpoleEndY = worldY - 176.0;
    if ( flagpoleEndX >= 0.0 && flagpoleEndX <= 7.0 )
    {
    	SpriteFlagpoleEnd( color, flagpoleEndX, flagpoleEndY );
    }
    
    

    // draw blocks
   	if (    ( tileX >= 134.0 && tileX < 138.0 && tileX - 132.0 > tileY )
         || ( tileX >= 140.0 && tileX < 144.0 && 145.0 - tileX > tileY )
         || ( tileX >= 148.0 && tileX < 153.0 && tileX - 146.0 > tileY && tileY < 5.0 )
         || ( tileX >= 155.0 && tileX < 159.0 && 160.0 - tileX > tileY ) 
         || ( tileX >= 181.0 && tileX < 190.0 && tileX - 179.0 > tileY && tileY < 9.0 )
         || ( tileX == 198.0 && tileY == 1.0 )
       )
    {
        SpriteBlock( color, worldXMod16, worldYMod16 );
    }
    
    
    // draw pipes
    float pipeY = worldY - 16.0;  
    float pipeH	= 0.0;    
    float pipeX = worldX - 179.0 * 16.0;
    if ( pipeX < 0.0 )
    {
        pipeX = worldX - 163.0 * 16.0;
        pipeH = 0.0;
    }
    if ( pipeX < 0.0 )
    {
        pipeX = worldX - 57.0 * 16.0;
        pipeH = 2.0;
    }
    if ( pipeX < 0.0 )
    {
        pipeX = worldX - 46.0 * 16.0;
        pipeH = 2.0;
    } 
    if ( pipeX < 0.0 )
    {
        pipeX = worldX - 38.0 * 16.0;
        pipeH = 1.0;
    }         
    if ( pipeX < 0.0 )
    {
        pipeX = worldX - 28.0 * 16.0;
        pipeH = 0.0;
    }
    if ( pipeX >= 0.0 && pipeX <= 31.0 && pipeY >= 0.0 && pipeY <= 31.0 + pipeH * 16.0 )
	{
		SpritePipe( color, pipeX, pipeY, pipeH );
	}
    
    
    // draw mushroom
    float mushroomStart = 15.7;    
    if ( time >= mushroomStart && time <= 17.0 )
    {
        float jumpTime = 0.5;
        
        float mushroomX = worldX - 1248.0;
        float mushroomY = worldY - 4.0 * 16.0;
        if ( time >= mushroomStart )
        {
            mushroomY = worldY - 4.0 * 16.0 - floor( 16.0 * clamp( ( time - mushroomStart ) / 0.5, 0.0, 1.0 ) );
        }
        if ( time >= mushroomStart + 0.5 )
        {
            mushroomX -= floor( MARIO_SPEED * ( time - mushroomStart - 0.5 ) );
        }
        if ( time >= mushroomStart + 0.5 + 0.4 )
        {
            mushroomY = mushroomY + floor( sin( ( ( time - mushroomStart - 0.5 - 0.4 ) ) * 3.14 ) * 4.0 * 16.0 );
        }
        
        if ( mushroomX >= 0.0 && mushroomX <= 15.0 )
        {
        	SpriteMushroom( color, mushroomX, mushroomY );
        }
    }

    
    // draw coins
    float coinFrame = floor( mod( time * 12.0, 4.0 ) );
    float coinX 	= worldX - 2720.0;
    float coinTime 	= 33.9;    
    float coinY 	= CoinAnimY( worldY, time, coinTime );
    if ( coinX < 0.0 )
    {
    	coinX 		= worldX - 1696.0;
    	coinTime 	= 22.4;    
    	coinY 		= CoinAnimY( worldY, time, coinTime );        
    }
    if ( coinX < 0.0 )
    {
    	coinX 		= worldX - 352.0;
    	coinTime 	= 5.4;    
    	coinY 		= CoinAnimY( worldY, time, coinTime );
    } 
    
    if ( coinX >= 0.0 && coinX <= 15.0 && time >= coinTime + 0.1 )
    {   
        SpriteCoin( color, coinX, coinY, coinFrame );
    }

    
    // draw questions
	float questionT = clamp( sin( time * 6.0 ), 0.0, 1.0 );    
    if (    ( tileY == 4.0 && ( tileX == 16.0 || tileX == 20.0 || tileX == 109.0 || tileX == 112.0 ) )
         || ( tileY == 8.0 && ( tileX == 21.0 || tileX == 94.0 || tileX == 109.0 ) )
         || ( tileY == 8.0 && ( tileX >= 129.0 && tileX <= 130.0 ) )
       )
    {
        SpriteQuestion( color, worldXMod16, worldYMod16, questionT );
    }
    
    
    // draw hitted questions
    float questionHitTime 	= 33.9;
    float questionX 		= worldX - 2720.0;
    if ( questionX < 0.0 )
    {
        questionHitTime = 22.4;
        questionX		= worldX - 1696.0;
    }
    if ( questionX < 0.0 )
    {
        questionHitTime = 15.4;
        questionX		= worldX - 1248.0;
    }
    if ( questionX < 0.0 )
    {
        questionHitTime = 5.3;
        questionX		= worldX - 352.0;
    }    
    questionT		= time >= questionHitTime ? 1.0 : questionT;    
    float questionY = QuestionAnimY( worldY, time, questionHitTime );
    if ( questionX >= 0.0 && questionX <= 15.0 )
    {
    	SpriteQuestion( color, questionX, questionY, questionT );
    }
    if ( time >= questionHitTime && questionX >= 3.0 && questionX <= 12.0 && questionY >= 1.0 && questionY <= 15.0 )
    {
        color = RGB( 231, 90, 16 );
    }    

    
    // draw bricks
   	if (    ( tileY == 4.0 && ( tileX == 19.0 || tileX == 21.0 || tileX == 23.0 || tileX == 77.0 || tileX == 79.0 || tileX == 94.0 || tileX == 118.0 || tileX == 168.0 || tileX == 169.0 || tileX == 171.0 ) )
         || ( tileY == 8.0 && ( tileX == 128.0 || tileX == 131.0 ) )
         || ( tileY == 8.0 && ( tileX >= 80.0 && tileX <= 87.0 ) )
         || ( tileY == 8.0 && ( tileX >= 91.0 && tileX <= 93.0 ) )
         || ( tileY == 4.0 && ( tileX >= 100.0 && tileX <= 101.0 ) )
         || ( tileY == 8.0 && ( tileX >= 121.0 && tileX <= 123.0 ) )
         || ( tileY == 4.0 && ( tileX >= 129.0 && tileX <= 130.0 ) )
       )
    {
        SpriteBrick( color, worldXMod16, worldYMod16 );
    }   
    
    
    // draw castle flag
    float castleFlagX = worldX - 3264.0;
    float castleFlagY = worldY - 64.0 - floor( 32.0 * clamp( ( time - 44.6 ) / 1.0, 0.0, 1.0 ) );
    if ( castleFlagX > 0.0 && castleFlagX < 14.0 )
    {
    	SpriteCastleFlag( color, castleFlagX, castleFlagY );
	}
    
    DrawCastle( color, worldX - 3232.0, worldY - 16.0 );

    // draw ground
    if ( tileY <= 0.0
         && !( tileX >= 69.0  && tileX < 71.0 )
         && !( tileX >= 86.0  && tileX < 89.0 ) 
         && !( tileX >= 153.0 && tileX < 155.0 ) 
       )
    {
        SpriteGround( color, worldXMod16, worldYMod16 );
    }    
    

    // draw Koopa
    float goombaFrame = floor( mod( time * 5.0, 2.0 ) );
    KoopaWalk( color, worldX, worldY, time, goombaFrame, 2370.0 );
    
    
    // draw stomped walking Goombas
    float goombaY 			= worldY - 16.0;        
    float goombaLifeTime 	= 26.3;
    float goombaX 			= GoombaSWalkX( worldX, 2850.0 + 24.0, time, goombaLifeTime );
    if ( goombaX < 0.0 )
    {
        goombaLifeTime 	= 25.3;
        goombaX 		= GoombaSWalkX( worldX, 2760.0, time, goombaLifeTime );
    }
    if ( goombaX < 0.0 ) 
    {
		goombaLifeTime 	= 23.5;
        goombaX 		= GoombaSWalkX( worldX, 2540.0, time, goombaLifeTime );
    }
    if ( goombaX < 0.0 ) 
    {
        goombaLifeTime 	= 20.29;
        goombaX 		= GoombaSWalkX( worldX, 2150.0, time, goombaLifeTime );
    }
    if ( goombaX < 0.0 )
    {
        goombaLifeTime 	= 10.3;
		goombaX 		= worldX - 790.0 - floor( abs( mod( ( min( time, goombaLifeTime ) + 6.3 ) * GOOMBA_SPEED, 2.0 * 108.0 ) - 108.0 ) );
    }
	goombaFrame = time > goombaLifeTime ? 2.0 : goombaFrame;
    if ( goombaX >= 0.0 && goombaX <= 15.0 )
    {
        SpriteGoomba( color, goombaX, goombaY, goombaFrame );
    }    
    
    // draw walking Goombas
    goombaFrame 		= floor( mod( time * 5.0, 2.0 ) );
    float goombaWalkX 	= worldX + floor( time * GOOMBA_SPEED );
    goombaX 			= goombaWalkX - 3850.0 - 24.0;
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 3850.0;
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 2850.0;
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 2760.0 - 24.0;
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 2540.0 - 24.0;
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 2150.0 - 24.0;
    if ( goombaX < 0.0 ) goombaX = worldX - 766.0 - floor( abs( mod( ( time + 6.3 ) * GOOMBA_SPEED, 2.0 * 108.0 ) - 108.0 ) );
    if ( goombaX < 0.0 ) goombaX = worldX - 638.0 - floor( abs( mod( ( time + 6.6 ) * GOOMBA_SPEED, 2.0 * 84.0 ) - 84.0 ) );
    if ( goombaX < 0.0 ) goombaX = goombaWalkX - 435.0;
    if ( goombaX >= 0.0 && goombaX <= 15.0 )
    {
        SpriteGoomba( color, goombaX, goombaY, goombaFrame );
    }
    

    
    // Mario jump
    float marioBigJump1 	= 27.1;
    float marioBigJump2 	= 29.75;
    float marioBigJump3 	= 35.05;    
    float marioJumpTime 	= 0.0;
    float marioJumpScale	= 0.0;
    
    if ( time >= 4.2   ) { marioJumpTime = 4.2;   marioJumpScale = 0.45; }
    if ( time >= 5.0   ) { marioJumpTime = 5.0;   marioJumpScale = 0.5;  }
    if ( time >= 6.05  ) { marioJumpTime = 6.05;  marioJumpScale = 0.7;  }
    if ( time >= 7.8   ) { marioJumpTime = 7.8;   marioJumpScale = 0.8;  }
    if ( time >= 9.0   ) { marioJumpTime = 9.0;   marioJumpScale = 1.0;  }
    if ( time >= 10.3  ) { marioJumpTime = 10.3;  marioJumpScale = 0.3;  }
    if ( time >= 11.05 ) { marioJumpTime = 11.05; marioJumpScale = 1.0;  }
    if ( time >= 13.62 ) { marioJumpTime = 13.62; marioJumpScale = 0.45; }
    if ( time >= 15.1  ) { marioJumpTime = 15.1;  marioJumpScale = 0.5;  }
    if ( time >= 18.7  ) { marioJumpTime = 18.7;  marioJumpScale = 0.6;  }
    if ( time >= 19.65 ) { marioJumpTime = 19.65; marioJumpScale = 0.45; }
    if ( time >= 20.29 ) { marioJumpTime = 20.29; marioJumpScale = 0.3;  }
    if ( time >= 21.8  ) { marioJumpTime = 21.8;  marioJumpScale = 0.35; }
    if ( time >= 22.3  ) { marioJumpTime = 22.3;  marioJumpScale = 0.35; }
    if ( time >= 23.0  ) { marioJumpTime = 23.0;  marioJumpScale = 0.40; }
    if ( time >= 23.5  ) { marioJumpTime = 23.5;  marioJumpScale = 0.3;  }
    if ( time >= 24.7  ) { marioJumpTime = 24.7;  marioJumpScale = 0.45; }
    if ( time >= 25.3  ) { marioJumpTime = 25.3;  marioJumpScale = 0.3;  }
    if ( time >= 25.75 ) { marioJumpTime = 25.75; marioJumpScale = 0.4;  }
    if ( time >= 26.3  ) { marioJumpTime = 26.3;  marioJumpScale = 0.25; }
    if ( time >= marioBigJump1 ) 		{ marioJumpTime = marioBigJump1; 		marioJumpScale = 1.0; }
    if ( time >= marioBigJump1 + 1.0 ) 	{ marioJumpTime = marioBigJump1 + 1.0; 	marioJumpScale = 0.6; }
    if ( time >= marioBigJump2 ) 		{ marioJumpTime = marioBigJump2; 		marioJumpScale = 1.0; }
    if ( time >= marioBigJump2 + 1.0 ) 	{ marioJumpTime = marioBigJump2 + 1.0;	marioJumpScale = 0.6; }    
    if ( time >= 32.3  ) { marioJumpTime = 32.3;  marioJumpScale = 0.7;  }
    if ( time >= 33.7  ) { marioJumpTime = 33.7;  marioJumpScale = 0.3;  }
    if ( time >= 34.15 ) { marioJumpTime = 34.15; marioJumpScale = 0.45; }
    if ( time >= marioBigJump3 ) 				{ marioJumpTime = marioBigJump3; 				marioJumpScale = 1.0; }
    if ( time >= marioBigJump3 + 1.2 ) 			{ marioJumpTime = marioBigJump3 + 1.2; 			marioJumpScale = 0.89; }
    if ( time >= marioBigJump3 + 1.2 + 0.75 ) 	{ marioJumpTime = marioBigJump3 + 1.2 + 0.75; 	marioJumpScale = 0.5; }
    
    float marioJumpOffset 		= 0.0;
    float marioJumpLength 		= 1.5  * marioJumpScale;
    float marioJumpAmplitude	= 76.0 * marioJumpScale;
    if ( time >= marioJumpTime && time <= marioJumpTime + marioJumpLength )
    {
        float t = ( time - marioJumpTime ) / marioJumpLength;
        marioJumpOffset = floor( sin( t * 3.14 ) * marioJumpAmplitude );
    }
    
    
    // Mario land
    float marioLandTime 	= 0.0;
    float marioLandAplitude = 0.0;
    if ( time >= marioBigJump1 + 1.0 + 0.45 ) 			{ marioLandTime = marioBigJump1 + 1.0 + 0.45; 			marioLandAplitude = 109.0; }
    if ( time >= marioBigJump2 + 1.0 + 0.45 ) 			{ marioLandTime = marioBigJump2 + 1.0 + 0.45; 			marioLandAplitude = 109.0; }
	if ( time >= marioBigJump3 + 1.2 + 0.75 + 0.375 ) 	{ marioLandTime = marioBigJump3 + 1.2 + 0.75 + 0.375; 	marioLandAplitude = 150.0; }
    
    float marioLandLength = marioLandAplitude / 120.0;
	if ( time >= marioLandTime && time <= marioLandTime + marioLandLength )
    {
        float t = 0.5 * ( time - marioLandTime ) / marioLandLength + 0.5;
       	marioJumpOffset = floor( sin( t * 3.14 ) * marioLandAplitude );
    }
    
    
    // Mario flag jump
    marioJumpTime 		= flagPauseStart - 0.3;
    marioJumpLength 	= 1.5  * 0.45;
    marioJumpAmplitude	= 76.0 * 0.45;
    if ( time >= marioJumpTime && time <= marioJumpTime + marioJumpLength + flagPauseLength ) 
    {
        float time2 = time;
        if ( time >= flagPauseStart && time <= flagPauseStart + flagPauseLength ) 
        {
            time2 = flagPauseStart;
        }
        else if ( time >= flagPauseStart )
        {
            time2 = time - flagPauseLength;
        }
		float t = ( time2 - marioJumpTime ) / marioJumpLength;
        marioJumpOffset = floor( sin( t * 3.14 ) * marioJumpAmplitude );
    }
    

    // Mario base (ground offset)
    float marioBase = 0.0;
    if ( time >= marioBigJump1 + 1.0 && time < marioBigJump1 + 1.0 + 0.45 )
    {
        marioBase = 16.0 * 4.0;
    }
    if ( time >= marioBigJump2 + 1.0 && time < marioBigJump2 + 1.0 + 0.45 )
    {
        marioBase = 16.0 * 4.0;
    }    
    if ( time >= marioBigJump3 + 1.2 && time < marioBigJump3 + 1.2 + 0.75 )
    {
        marioBase = 16.0 * 3.0;
    }    
    if ( time >= marioBigJump3 + 1.2 + 0.75 && time < marioBigJump3 + 1.2 + 0.75 + 0.375 )
    {
        marioBase = 16.0 * 7.0;
    }

    float marioX		= pixelX - 112.0;
    float marioY		= pixelY - 16.0 - 8.0 - marioBase - marioJumpOffset;    
    float marioFrame 	= marioJumpOffset == 0.0 ? floor( mod( time * 10.0, 3.0 ) ) : 3.0;
    if ( time >= mushroomPauseStart && time <= mushroomPauseStart + mushroomPauseLength )
    {
    	marioFrame = 1.0;
    }    
    if ( time > mushroomPauseStart + 0.7 )
    {
        float t = time - mushroomPauseStart - 0.7;
    	if ( mod( t, 0.2 ) <= mix( 0.0, 0.2, clamp( t / 1.3, 0.0, 1.0 ) ) )
        {
            // super mario offset
            marioFrame += 4.0;
        }
    }    
    if ( marioX >= 0.0 && marioX <= 15.0 && cameraX < 3152.0 )
    {
        SpriteMario( color, marioX, marioY, marioFrame );
    }
}

// Function 129
float SampleFontCharacter( int charIndex, vec2 vCharUV )
{
#if USE_FONT_TEXTURE    
    vec2 vUV;
    
    vCharUV.x = vCharUV.x * 0.6 + 0.25;
    
    vUV.x = (float(charIndex % 16) + vCharUV.x) / 16.0;
    vUV.y = (float(charIndex / 16) + vCharUV.y) / 16.0;
    
	return clamp( ( 0.503 - texture(iChannel1, vUV).w) * 100.0, 0.0, 1.0 );
#else    
	float fCharData = 0.0;
    ivec2 vCharPixel = ivec2(vCharUV * vec2(kCharPixels) );   

    #if !HIRES_FONT
        bool bCharData = CharBitmap12x20( charIndex, vCharPixel );            
        fCharData = bCharData ? 1.0 : 0.0;
    #else
        bool bCharData = CharHiRes( charIndex, vCharUV );
        fCharData = bCharData ? 1.0 : 0.0;
    #endif
    
    return fCharData;
#endif
}

// Function 130
Shape character(vec3 c) {
    Shape shape;
    shape.dist = 1000.;
    shape.color = vec4(1.);
    //instiating vars
    vec3 b = c; //body
    vec3 h = c; //head
    vec3 e = c;	//eyes
    vec3 n = c; //nose
    vec3 m = c; //mouth
   	vec3 ha = c; //hands
    vec3 f = c; //feet
    vec3 bu = c; //butt
    vec3 p = c;
   	vec3 caone = c; //cap
    vec3 catwo = c;
    vec3 cathree = c;
	//changing projected pixels and its distance from the camera
	p.x = abs(p.x) - .075;
    float pupils = sphere(p - vec3(0., .6, -2.75), .025);
    m.xy *= rot(2.5*m.x);
    float mouth = fBox(m - vec3(0., .5, -2.), vec3(.15, .025, .005));
    //float nose = sphere(e - vec3(0., .65, - 2.), .05);
    n.yz *= rot(radians(15.));
	
    //pModPolar(caone.yz, 3.);
    //caone.y /= cos(sin(c.x)); 
	caone.x *= cos(sin(caone.x*3.))*.6;
    float capone = sphere(caone - vec3(0., 1.5, -2.), .4);
    catwo.xy *= rot(radians(10.));
    catwo.y *= cos(sin(catwo.x*3.))*.3;
    float captwo = sphere(catwo - vec3(.8, 0.18, -2.), .1);
    cathree.xy *= rot(-radians(10.));
    cathree.y *= cos(sin(catwo.x*3.))*.3;
    float capthree = sphere(cathree - vec3(-.8, .15, -2.), .1);
    //ca.xy *= rot(1.5 * ca.x);
    //float cap = fBox(ca - vec3(0., 1.5, -2.), vec3(.5, .075, .002));
    float nose = fCone(n - vec3(0., .65, -2.), .05, .5);

    e.x = abs(e.x) - .1;
    float eye = sphere(e - vec3(0., .75, - 2.), .1); 
    //b.y = cos(sin(b.y*1.) * cos(.1));
    float body = sphere(b - vec3(0., -.5, 0.), 1.5);
    
    //float body = fBox(b - vec3(0., -.15, 0.), vec3(.1, .4, .005));
    float butt = sphere(bu - vec3(0., -.5, .01), .11);
    ha.x = abs(ha.x) - .75;
    float hands = sphere(ha - vec3(0., .25, - 1.5), .15);
    float head = sphere(h - vec3(0., 1., -1.), .85);
    
    f.x = abs(f.x) - .4;
    f.y *= cos(sin(f.x*2.))*.6;
    f.z *= cos(sin(f.x*2.5));
    float feet = sphere(f - vec3(0., -.75, -2.5), .1);
    //float feet = sdEllipsoid(f + vec3(.0, -.8, 2.5), vec3(.1, .02, .2));
    //f.x = abs(f.x) - .4;
    //float feet = fBox(f - vec3(0., -.9, -1.9), vec3(.125, .06, .1));
    
	
    //adding the instantiated variables onto the shape.dist. 
    shape.dist = fOpUnionRound(body, head, 1.);
    shape.dist = min(shape.dist, eye);
    shape.dist = min(shape.dist, nose);
    shape.dist = min(shape.dist, mouth);
    shape.dist = min(shape.dist, hands);
    shape.dist = min(shape.dist, feet);
    shape.dist = min(shape.dist, butt);
	shape.dist = min(shape.dist, pupils);
    shape.dist = fOpUnionColumns(shape.dist, capone, .25, 3.);
    shape.dist = fOpUnionColumns(shape.dist, captwo, .25, 1.);
    shape.dist = min(shape.dist, capthree);
    
            
    return shape;
}

// Function 131
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

// Function 132
Shape character(vec3 c){
  Shape shape;
  shape.dist = 1000.; // Draw Distance
  shape.color = vec4(1.); // Initial Color
 //  vec4 color = vec4(0.5, cos(iTime), cos(iTime), 0.); 
    
  vec3 p = c; // Base
  vec3 b = c; // Body
  vec3 a1 = c; // Arm 1 Sect 1
  vec3 a12 = c; // Arm 1 Sect 2 
  vec3 a2 = c; // Arm 2 Sect 1 
  vec3 a22 = c; // Arm 2 Sect 2 
  vec3 a3 = c; // Arm 3 Sect 1
  vec3 a32 = c; // Arm 3 Sect 2 
  vec3 a4 = c; // Arm 4 Sect 1
  vec3 a42 = c; // Arm 4 Sect 2 
  vec3 e1 = c; // Eye 1 
  vec3 e2 = c; // Pupil
  vec3 w1 = c; // Wheel inside 
  vec3 w2 = c; // Wheel outside 
  vec3 h = c; // Hat 
  vec3 t = c; // Tentacles
    
    
  // Body
  vec4 bColor = vec4(1.0, 0.0, 0.3, 0.);  // 
  float body = fBox(b+vec3(0.0, -1.5, 0.0), vec3(0.8, 1.2, 0.75)); // Makes a box (called body) at point p of size = 1.
    
  // Base
  vec4 baColor = vec4(1.0, 0.0, 0.0, 1.0); // 
  float base = fBox(p+vec3(0.0, 0.3, 0.0), vec3(0.55, 0.65, 0.75)); // Makes a box for the base
    
  // Eye Outside 
  vec4 e1Color = vec4(0.,0.,0., 0.); // BLACK
  float eye1 = sdEllipsoid(e1+vec3(0.0,-1.0, 0.4), vec3(.5, abs(sin(iTime))*.5, .6));
   
  // Eye Inside
  vec4 e2Color = vec4(1., 1., 1., 1.); 
  float eye2 = fSphere(e2+vec3((sin(iTime)*0.2), -1.0, 0.5), 0.15); 
   
  // Note: I intiially had wheels hence the wheels here; I might still implement them later.   
    
  // Wheel Outside
  //float wheel1 = fSphere(w1+vec3(0.0,2.,0.0), 1.0); 
 
  // Wheel Inside
  //float wheel2 = fSphere(w2+vec3(0.0,sin(iTime)+2.,sin(iTime*0.1)+0.5), 0.4);  
  
  // Arm 1
  vec4 a1Color = vec4(0.,0.,1.,1.);
  float arm1 = fBox(a1+vec3(1.2,-0.3,0.0), vec3(1.,0.3,0.0)); 
  a12.xy *= rot(radians(45.)); 
  vec4 a12Color = vec4(0.,0.,1.,1.); 
  float arm12 = fBox(a12+vec3(1.8,-1.7,0.0), vec3(.7,0.3,0.0));
    
  // Arm 2
  vec4 a2Color = vec4(0., 0., 1., 1.);
  float arm2 = fBox(a2+vec3(1.2,-1.5,0.0), vec3(1.,0.3,0.0)); 
  a22.xy *= rot(radians(45.)); 
  vec4 a22Color = vec4(0.,0., 1., 1.); 
  float arm22 = fBox(a22+vec3(1.0,-2.5,0.0), vec3(.7,0.3,0.0)); 
    
  // Arm 3
  vec4 a3Color = vec4(0., 0., 1., 1.);
  float arm3 = fBox(a3+vec3(-1.2,-0.3,0.0), vec3(1.,0.3,0.0)); 
  a32.xy *= rot(radians(-45.)); 
  vec4 a32Color = vec4(0., 0., 1., 1.);
  float arm32 = fBox(a32+vec3(-1.8,-1.7,0.0), vec3(.7,0.3,0.0));
    
  // Arm 4
  vec4 a4Color = vec4(0., 0., 1., 1.);
  float arm4 = fBox(a4+vec3(-1.2,-1.5,0.0), vec3(1.,0.3,0.0)); 
  a42.xy *= rot(radians(-45.)); 
  vec4 a42Color = vec4(0., 0., 1., 1.);
  float arm42 = fBox(a42+vec3(-1.0,-2.5,0.0), vec3(.7,0.3,0.0)); 
    
  // Hat
  vec4 hColor = vec4(1.,1.,1.,1.); 
  float hat = fSphere(h-vec3(0.,2.7,0.0), 0.55); 
    
  // Tentacles
  vec4 tColor = vec4(sin(iTime), 0.2, sin(iTime)-0.5, 1.); 
  t.x = abs(t.x)-.15; // Mirror
  t.x = abs(t.x)-0.15; // Mirror again
  t.xy *= rot(radians(180.)); // Rotates it to face the other way
  t.x += sin(t.y * 10. - iTime * 4.) * (1. - t.y) * .03; // Animates the tentacles
  // float tentacles = fCone(t+vec3(0., 0.5, 0.), 0.4, 3.); // Animates the tentacles
  float tentacles = fCone(t+vec3(0., -0.55, 0.), 0.4, 2.); 
      
    
  shape.dist = max(body, -eye1); // Adds the box and eye (difference)
  shape.dist = min(shape.dist, eye2); // Use shape.dist after two shapes 
  shape.dist = min(shape.dist, base); 
  //shape.dist = min(shape.dist, wheel1); 
  //shape.dist = max(shape.dist, -wheel2); 
  shape.dist = min(shape.dist, arm1); 
  shape.dist = min(shape.dist, arm12); 
  shape.dist = min(shape.dist, arm2); 
  shape.dist = min(shape.dist, arm22); 
  shape.dist = min(shape.dist, arm3);
  shape.dist = min(shape.dist, arm32);
  shape.dist = min(shape.dist, arm4);
  shape.dist = min(shape.dist, arm42);
  shape.dist = min(shape.dist, tentacles); 
  shape.dist = fOpUnionColumns(shape.dist,hat, 1., 5.); // Creates the hat ilke effect
  
  shape.color = mix(bColor, baColor, mixColors(base, body, 0.0));
  shape.color = mix(shape.color, baColor, mixColors(body, shape.dist, 0.0));
  shape.color = mix(shape.color, e1Color, mixColors(eye1, shape.dist, 0.0));
  shape.color = mix(shape.color, e2Color, mixColors(eye2, shape.dist, 1.0));
  shape.color = mix(shape.color, a1Color, mixColors(arm1, shape.dist, 0.0));
  shape.color = mix(shape.color, a12Color, mixColors(arm12, shape.dist, 1.0));
  shape.color = mix(shape.color, a2Color, mixColors(arm2, shape.dist, 0.0)); 
  shape.color = mix(shape.color, a22Color, mixColors(arm22, shape.dist, 1.0));
  shape.color = mix(shape.color, a3Color, mixColors(arm3, shape.dist, 0.0));
  shape.color = mix(shape.color, a32Color, mixColors(arm32, shape.dist, 1.0));
  shape.color = mix(shape.color, a4Color, mixColors(arm4, shape.dist, 0.0));
  shape.color = mix(shape.color, a42Color, mixColors(arm42, shape.dist, 1.0));
  shape.color = mix(shape.color, tColor, mixColors(tentacles, shape.dist, 1.0));
  shape.color = mix(shape.color, hColor, mixColors(hat, shape.dist, 0.5)); 
    
    
  return shape;
}

// Function 133
vec4 spreadingGame(vec2 uv,float a, float b)
{
    vec3 gen = vec3(0.0,0.0,0.0);
   	
    float textureSize = iResolution.x*iResolution.y;    
    float onePixel = 1.0/textureSize;
    
    float total = 0.0;
    //
    float tl = NearestTextureSample(uv + vec2(-1.0/iResolution.x,-1.0/iResolution.y)).r;
    float tm = NearestTextureSample(uv + vec2(0.0,-1.0/iResolution.y)).r;
    float tr = NearestTextureSample(uv + vec2(1.0/iResolution.x,-1.0/iResolution.y)).r;
    float ml = NearestTextureSample(uv + vec2(-1.0/iResolution.x,0.0)).r;
    float mm = NearestTextureSample(uv + vec2(0.0,0.0)).r;
    float mr = NearestTextureSample(uv + vec2(1.0/iResolution.x,0.0)).r;
    float bl = NearestTextureSample(uv + vec2(-1.0/iResolution.x,1.0/iResolution.y)).r;
    float bm = NearestTextureSample(uv + vec2(0.0,1.0/iResolution.y)).r;
    float br = NearestTextureSample(uv + vec2(1.0/iResolution.x,1.0/iResolution.y)).r;
    float sum = tl + tm + tr + ml + mm + mr + bl + bm + br;
    if((abs(sum-a) < 0.001) )
    {
    	total += 1.0;   
    }
    
    if((abs(sum-b) < 0.001) )
    {
    	total += (1.0-mm);   
    }
    gen += vec3(1.0,1.0,1.0)*total;   
    
    return vec4(gen,1.0);
}

// Function 134
vec4 GameWin( vec2 u, vec4 o )
{
    o = vec4(dot(o,vec4(.2126,.7152,.0722,0)))*vec4(.3);
    
    u -= .5;
    u.y *= 9./16.;
    
    float idim = .01;
    for ( int i=0; i < collectibles.length(); i++ )
    {
        float a = (float(iFrame)*.003+float(i)/float(collectibles.length()))*6.283;
        vec2 iuv = u-vec2(cos(a),sin(a))*(.13+.12*sin(float(iFrame)*.01));
//        iuv.y /= 2.;//iResolution.x/iResolution.y;
        iuv.y -= idim;
        float r = dot(abs(iuv),vec2(1));
        if ( r < idim )
        {
            o = vec4(float((iFrame+i)%9+3*int(-sign(iuv.x)))/8.,float((iFrame+i)%11+5*int(sign(iuv.y)))/10.,1,1);
        }
    }
    
    u *= 256.;

    float t = 0.;

    const int string[] = int[]( 0xA7, 0xB5, 0xBC, 0xBC, 0x50, 0xB4, 0xBF, 0xBE, 0xB5, 0xD1 );
    vec2 uv = u / 14. - vec2(-float(string.length())*.25,.2);

    float a = .0;
    for ( int i=0; i < string.length(); i++ )
    {
        a += PrintCharacter( string[i], uv ); uv.x -= .5;
    }
    t += smoothstep(.487,.513,a);
    
    // Capitals 0xB1 to 0xAA, Lowercase 0x91 to 0x8A, numerals 0xC0 to 0xC9
    const int string2[] = int[]( 0xA9, 0x9F, 0x85, 0x50, 0x96, 0x9F, 0x85, 0x9e, 0x94, 0x50, 0x91, 0x9C, 0x9C );
    uv = vec2(u) / 10. - vec2(-float(string2.length())*.25,-.9);
    a = .0;
    for ( int i=0; i < string2.length(); i++ )
    {
        a += PrintCharacter( string2[i], uv ); uv.x -= .5;
    }
    t += smoothstep(.48,.52,a);

    const int string3[] = int[]( 0x84, 0x98, 0x95, 0x50, 0x93, 0x82, 0x89, 0x83, 0x84, 0x91, 0x9C, 0x83, 0xDE );
    uv = vec2(u) / 10. - vec2(-float(string2.length())*.25,-2.);
    a = .0;
    for ( int i=0; i < string3.length(); i++ )
    {
        a += PrintCharacter( string3[i], uv ); uv.x -= .5;
    }
    t += smoothstep(.48,.52,a);

    o = mix( o, 1.-o, t );
    
    return o;
}

// Function 135
Shape character(vec3 c){
  Shape shape;
  shape.dist = 1000.; // Draw Distance
  shape.color = vec4(1.); // Initial Color
 //  vec4 color = vec4(0.5, cos(iTime), cos(iTime), 0.); 
    
  vec3 p = c; // Base
  vec3 b = c; // Body
  vec3 a1 = c; // Arm 1 Sect 1
  vec3 a12 = c; // Arm 1 Sect 2 
  vec3 a2 = c; // Arm 2 Sect 1 
  vec3 a22 = c; // Arm 2 Sect 2 
  vec3 a3 = c; // Arm 3 Sect 1
  vec3 a32 = c; // Arm 3 Sect 2 
  vec3 a4 = c; // Arm 4 Sect 1
  vec3 a42 = c; // Arm 4 Sect 2 
  vec3 e1 = c; // Eye 1 
  vec3 e2 = c; // Pupil
  vec3 w1 = c; // Wheel inside 
  vec3 w2 = c; // Wheel outside 
  vec3 l1 = c;
  vec3 h = c; // Hat 
  vec3 t = c; // Tentacles
    
    
  // Body
  float body = fBox(b+vec3(0.0, -1.5, 0.0), vec3(0.8, 1.2, 0.75)); // Makes a box (called body) at point p of size = 1.
    
  // Base
  float base = fBox(p+vec3(0.0, 0.3, 0.0), vec3(0.55, 0.65, 0.75)); // Makes a box for the base
    
  // Eye Outside 
   float eye1 = sdEllipsoid(e1+vec3(0.0,-1.0, 0.4), vec3(.5, abs(sin(iTime))*.5, .6));
   
  // Eye Inside
  float eye2 = fSphere(e2+vec3((sin(iTime)*0.2), -1.0, 0.5), 0.15); 
   
  // Note: I intiially had wheels hence the wheels here; I might still implement them later.   
    
  // Wheel Outside
  //float wheel1 = fSphere(w1+vec3(0.0,2.,0.0), 1.0); 
 
  // Wheel Inside
  //float wheel2 = fSphere(w2+vec3(0.0,sin(iTime)+2.,sin(iTime*0.1)+0.5), 0.4);  
  
  // Arm 1
  float arm1 = fBox(a1+vec3(1.2,-0.3,0.0), vec3(1.,0.3,0.0)); 
  a12.xy *= rot(radians(45.)); 
  float arm12 = fBox(a12+vec3(1.8,-1.7,0.0), vec3(.7,0.3,0.0));
    
  // Arm 2
  float arm2 = fBox(a2+vec3(1.2,-1.5,0.0), vec3(1.,0.3,0.0)); 
  a22.xy *= rot(radians(45.)); 
  float arm22 = fBox(a22+vec3(1.0,-2.5,0.0), vec3(.7,0.3,0.0)); 
    
  // Arm 3
  float arm3 = fBox(a3+vec3(-1.2,-0.3,0.0), vec3(1.,0.3,0.0)); 
  a32.xy *= rot(radians(-45.)); 
  float arm32 = fBox(a32+vec3(-1.8,-1.7,0.0), vec3(.7,0.3,0.0));
    
  // Arm 4
  float arm4 = fBox(a4+vec3(-1.2,-1.5,0.0), vec3(1.,0.3,0.0)); 
  a42.xy *= rot(radians(-45.)); 
  float arm42 = fBox(a42+vec3(-1.0,-2.5,0.0), vec3(.7,0.3,0.0)); 
    
  // Hat
  float hat = fSphere(h-vec3(0.,2.7,0.0), 0.55); 
    
  // Tentacles
  t.x = abs(t.x)-.15; // Mirror
  t.x = abs(t.x)-0.15; // Mirror again
  t.xy *= rot(radians(180.)); // Rotates it to face the other way
  t.x += sin(t.y * 10. - iTime * 4.) * (1. - t.y) * .03; // Animates the tentacles
  float tentacles = fCone(t+vec3(0., 0.5, 0.), 0.4, 3.); // Animates the tentacles
      
    
  shape.dist = max(body, -eye1); // Adds the box and eye (difference)
  shape.dist = min(shape.dist, eye2); // Use shape.dist after two shapes 
  shape.dist = min(shape.dist, base); 
  //shape.dist = min(shape.dist, wheel1); 
  //shape.dist = max(shape.dist, -wheel2); 
  shape.dist = min(shape.dist, arm1); 
  shape.dist = min(shape.dist, arm12); 
  shape.dist = min(shape.dist, arm2); 
  shape.dist = min(shape.dist, arm22); 
  shape.dist = min(shape.dist, arm3);
  shape.dist = min(shape.dist, arm32);
  shape.dist = min(shape.dist, arm4);
  shape.dist = min(shape.dist, arm42);
  shape.dist = min(shape.dist, tentacles); 
  shape.dist = fOpUnionColumns(shape.dist,hat, 1., 5.); // Creates the hat ilke effect
  
 // shape.color = color; 
  return shape;
}

