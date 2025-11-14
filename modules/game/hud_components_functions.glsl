// Reusable Hud Components Game Functions
// Automatically extracted from game/interactive-related shaders

// Function 1
vec3 RenderScore(in vec2 score, in vec2 fragCoord, in vec3 col)
{
    const vec2 displacement = vec2(0.3, -0.65);
    vec2 uv = (2.0*fragCoord-iResolution.xy) / iResolution.y;
    
    col = mix(col, vec3(1.0, 1.0, 1.0), PrintInt((uv + displacement) * vec2(10.0, 7.0), int(score.x)));
    col = mix(col, vec3(1.0, 1.0, 1.0), PrintInt((uv + displacement * vec2(-0.4, 1.0)) * vec2(10.0, 7.0), int(score.y)));
    
    return col;
}

// Function 2
vec3 makeHud(in vec2 p, in float seek)
{
    float sk1 = smoothstep(0.89, 0., seek);
    float sk2 = step(1.-sk1, .4);
    //lens deformation
    float ll = abs(p.x)+abs(p.y)*0.15;
    p *= ll * -.2+1.29;
    p *= 2.;
    vec3 col = vec3(-1);
    float d= 0.;
    //crosshairs
    float rz = crosshair(p*1.1, .8,1.+sk1);
    rz = min(rz,crosshair(p*1.7,1., -time*5.5-0.1-sk1));
    //minimap (top right)
    float d2 = square(p+vec2(-0.45, -0.57))+0.01;
    d = smoothstep(0.2,0.21,d2);
    d = max(d,smoothstep(1.25,.45,min(sin(p.x*70.+0.9),sin(p.y*70.+time*5.))+0.4));
    d = min(d,smoothstep(0.001,0.008,abs(d2-0.2)));
    vec3 enp = enpos()/900.;
    enp.z = 0.-enp.z;
    float en = smoothstep(0.015, 0.023, loz(enp.xz+p-vec2(0.47, 0.4))) ;
    en += mod(floor(time*1.5), 1.);
    d = min(d,en);
    rz = min(d,rz);
    //text (top left)
    rz= min(rz,text2(p));
    //altitude bars
    d = min(rz,sin(p.y*90.+sin(time)*10.)*2.+2.);
    d2 = max(d,(p.x+0.49)*100.);
    d2 = max(d2,-(p.x+0.56)*100.);
    float d3 = max(d,(p.x-0.56)*100.);
    d3 = max(d3,-(p.x-.49)*100.);
    d2 = min(d2,d3);
    d2 += smoothstep(0.49, .5, -p.y);
    d2 += smoothstep(0.49, .5, p.y);
    rz = min(rz,d2);    
    //bottom left "status"
    float num = mod(floor(time*20.),20.);
    vec2 p2 = p+vec2(-0.32,.84);
    d = 1.;
    for(float i=0.;i<7.;i++)
    {
        d = min(d,length(p2)+float(num==i));
    	p2.x -= 0.075;
    }
    d = smoothstep(0.013,.02,d);
    rz = min(d,rz);
    
    vec3 hcol = (sin(vec3(0.25,0.3,0.38)*(2.35)*PALETTE)*0.4+.4);
    hcol.gb -= sk2;
    hcol.r += sk2;
    return hcol*(1.-rz);
}

// Function 3
vec4 drawHud(vec3 ro, vec3 rd, vec3 ww, vec3 uu, vec3 vv, vec2 U)
{
    vec4 result = vec4(0.0);
    
    vec3 P = vec3(0.0, 0.0, 10.0);
    P = (RotateAxisAngle(vec3(1.0, 0.0, 0.0), asin(-ww.y)) * vec4(P, 1.0)).xyz;
    P.y -= state.camPosition.y;
    
	// Draw the overaly hud
    P = 2.0*(P.xyz/P.z);
    vec3 P1 = P + vec3(0.4, 0.0, 0.0);
    vec3 P2 = P + vec3(0.05, 0.0, 0.0);
    vec3 P3 = P + vec3(-0.05, 0.0, 0.0);
    vec3 P4 = P + vec3(-0.4, 0.0, 0.0);
    P1 = (RotateAxisAngle(vec3(0.0, 0.0, 1.0), asin(uu.y)) * vec4(P1, 1.0)).xyz;
    P2 = (RotateAxisAngle(vec3(0.0, 0.0, 1.0), asin(uu.y)) * vec4(P2, 1.0)).xyz;
    P3 = (RotateAxisAngle(vec3(0.0, 0.0, 1.0), asin(uu.y)) * vec4(P3, 1.0)).xyz;
    P4 = (RotateAxisAngle(vec3(0.0, 0.0, 1.0), asin(uu.y)) * vec4(P4, 1.0)).xyz;
    
	result += 
        vec4(0.0, 0.6 * drawLine(P1.xy, P2.xy, 0.002), 0.0, 1.0) +
        vec4(0.0, 0.6 * drawLine(P1.xy*1.2, P2.xy*1.2, 0.002), 0.0, 1.0) +
        vec4(0.0, 0.6 * drawLine(P1.xy*1.3, P2.xy*1.3, 0.002), 0.0, 1.0) +
        vec4(0.0, 0.6 * drawLine(P3.xy, P4.xy, 0.002), 0.0, 1.0) +
        vec4(0.0, 0.6 * drawLine(P3.xy*1.2, P4.xy*1.2, 0.002), 0.0, 1.0) +
        vec4(0.0, 0.6 * drawLine(P3.xy*1.3, P4.xy*1.3, 0.002), 0.0, 1.0) +
    	vec4(0.0, 0.6 * drawLine(vec2(-0.1,-0.6), vec2(0.1,-0.6), 0.002), 0.0, 1.0) +
        vec4(0.0, 0.6 * drawLine(vec2(0.0,-0.7), vec2(0.0,-0.5), 0.002), 0.0, 1.0) - 
        vec4(0.0, 0.6 * drawLine(vec2(-0.01,-0.6), vec2(0.01,-0.6), 0.002), 0.0, 1.0)-
        vec4(0.0, 0.6 * drawLine(vec2(0.0,-0.59), vec2(0.0,-0.61), 0.002), 0.0, 1.0) ;
    
    float h = abs(0.3-0.8)*state.camPosition.y/1.5;
    result += vec4(0.0, 0.6 * drawLine(vec2(1.59,-0.8), vec2(1.59,h-0.8), 0.002), 0.0, 1.0);
    result += vec4(0.0, 0.6 * drawLine(vec2(1.61,-0.8), vec2(1.61, 0.2), 0.002), 0.0, 1.0);
   
    // Print the altitude
    int text[5] = int[5](48+int(h*10.0), 48+int(mod(h*100.0,10.0)), 48+int(mod(h*1000.0, 10.0)), 46, 0);
	vec2 U_t =(U+vec2(-1.38,-h+0.8))*10.0;
    for(int i=0; i<4; i++) 
    { 
        result += vec4(0.0, char(U_t, text[i]).x, 0.0, 1.0); U_t.x-=0.5;
    }
        
    int text2[10] = int[10](83, 80, 69, 69, 68, 58, 32, 48+int(h*10.0), 48+int(mod(state.speed*100.0,10.0)), 48+int(mod(state.speed*1000.0, 10.0)));
    U_t =(U+vec2(1.7,1.5))*10.0;
    for(int i=0; i<10; i++) 
    { 
        result += vec4(0.0, char(U_t, text2[i]).x, 0.0, 1.0);  
        U_t.x-=0.5; 
    }

    // Peinr directions
    float cosA;
    
    if(ww.z >= -0.2)
    {
        cosA = dot(ww.xz, vec2(1.0, 0.0));
        U_t =(U+vec2(cosA*0.2,-0.2))*10.0;
        result += vec4(0.0, char(U_t, 78).x, 0.0, 1.0); 
    }
    
    if(ww.z <= 0.2)
    {
        cosA = dot(ww.xz, vec2(-1.0, -0.0));
        U_t =(U+vec2(cosA*0.2,-0.2))*10.0;
        result += vec4(0.0, char(U_t, 83).x, 0.0, 1.0);  
    }
    
    if(ww.x >= -0.2)
    {
        cosA = dot(ww.xz, vec2(0.0, -1.0));
        U_t =(U+vec2(cosA*0.2,-0.2))*10.0;
        result += vec4(0.0, char(U_t, 87).x, 0.0, 1.0);  
    }
    
    if(ww.x <= 0.2)
    {
        cosA = dot(ww.xz, vec2(0.0, 1.0));
        U_t =(U+vec2(cosA*0.2,-0.2))*10.0;
        result += vec4(0.0, char(U_t, 69).x, 0.0, 1.0);  
    }
    
    return result; 
}

// Function 4
void SpriteBossCore( inout vec3 color, float x, float y )
{
    float idx = 0.0;
    
    idx = y == 30.0 ? ( x <= 7.0 ? 21844.0 : ( x <= 15.0 ? 85.0 : 0.0 ) ) : idx;
    idx = y == 29.0 ? ( x <= 7.0 ? 65533.0 : ( x <= 15.0 ? 21845.0 : 5461.0 ) ) : idx;
    idx = y == 28.0 ? ( x <= 7.0 ? 43689.0 : ( x <= 15.0 ? 65345.0 : 28671.0 ) ) : idx;
    idx = y == 27.0 ? ( x <= 7.0 ? 43689.0 : ( x <= 15.0 ? 43861.0 : 21930.0 ) ) : idx;
    idx = y == 26.0 ? ( x <= 7.0 ? 43685.0 : ( x <= 15.0 ? 43841.0 : 21610.0 ) ) : idx;
    idx = y == 25.0 ? ( x <= 7.0 ? 43665.0 : ( x <= 15.0 ? 43861.0 : 21850.0 ) ) : idx;
    idx = y == 24.0 ? ( x <= 7.0 ? 43605.0 : ( x <= 15.0 ? 43841.0 : 27462.0 ) ) : idx;
    idx = y == 23.0 ? ( x <= 7.0 ? 43293.0 : ( x <= 15.0 ? 43861.0 : 27605.0 ) ) : idx;
    idx = y == 22.0 ? ( x <= 7.0 ? 42361.0 : ( x <= 15.0 ? 27457.0 : 23476.0 ) ) : idx;
    idx = y == 21.0 ? ( x <= 7.0 ? 20969.0 : ( x <= 15.0 ? 23381.0 : 27565.0 ) ) : idx;
    idx = y == 20.0 ? ( x <= 7.0 ? 38825.0 : ( x <= 15.0 ? 17855.0 : 23467.0 ) ) : idx;
    idx = y == 19.0 ? ( x <= 7.0 ? 26281.0 : ( x <= 15.0 ? 55009.0 : 27562.0 ) ) : idx;
    idx = y == 18.0 ? ( x <= 7.0 ? 26276.0 : ( x <= 15.0 ? 38592.0 : 32746.0 ) ) : idx;
    idx = y == 17.0 ? ( x <= 7.0 ? 22928.0 : ( x <= 15.0 ? 39808.0 : 23162.0 ) ) : idx;
    idx = y == 16.0 ? ( x <= 7.0 ? 6544.0 : ( x <= 15.0 ? 39808.0 : 23390.0 ) ) : idx;
    idx = y == 15.0 ? ( x <= 7.0 ? 6544.0 : ( x <= 15.0 ? 39808.0 : 23390.0 ) ) : idx;
    idx = y == 14.0 ? ( x <= 7.0 ? 6544.0 : ( x <= 15.0 ? 39808.0 : 23390.0 ) ) : idx;
    idx = y == 13.0 ? ( x <= 7.0 ? 6564.0 : ( x <= 15.0 ? 39808.0 : 23390.0 ) ) : idx;
    idx = y == 12.0 ? ( x <= 7.0 ? 22953.0 : ( x <= 15.0 ? 39808.0 : 23162.0 ) ) : idx;
    idx = y == 11.0 ? ( x <= 7.0 ? 26281.0 : ( x <= 15.0 ? 38592.0 : 32746.0 ) ) : idx;
    idx = y == 10.0 ? ( x <= 7.0 ? 26281.0 : ( x <= 15.0 ? 38625.0 : 27562.0 ) ) : idx;
    idx = y == 9.0 ? ( x <= 7.0 ? 38569.0 : ( x <= 15.0 ? 17850.0 : 23466.0 ) ) : idx;
    idx = y == 8.0 ? ( x <= 7.0 ? 20905.0 : ( x <= 15.0 ? 24405.0 : 27561.0 ) ) : idx;
    idx = y == 7.0 ? ( x <= 7.0 ? 46441.0 : ( x <= 15.0 ? 27457.0 : 23460.0 ) ) : idx;
    idx = y == 6.0 ? ( x <= 7.0 ? 44313.0 : ( x <= 15.0 ? 43861.0 : 27541.0 ) ) : idx;
    idx = y == 5.0 ? ( x <= 7.0 ? 43861.0 : ( x <= 15.0 ? 43841.0 : 27462.0 ) ) : idx;
    idx = y == 4.0 ? ( x <= 7.0 ? 43729.0 : ( x <= 15.0 ? 43861.0 : 21850.0 ) ) : idx;
    idx = y == 3.0 ? ( x <= 7.0 ? 45045.0 : ( x <= 15.0 ? 43841.0 : 21610.0 ) ) : idx;
    idx = y == 2.0 ? ( x <= 7.0 ? 62804.0 : ( x <= 15.0 ? 65365.0 : 21930.0 ) ) : idx;
    idx = y == 1.0 ? ( x <= 7.0 ? 21504.0 : ( x <= 15.0 ? 21845.0 : 27391.0 ) ) : idx;
    idx = y == 0.0 ? ( x <= 7.0 ? 0.0 : ( x <= 15.0 ? 20480.0 : 5461.0 ) ) : idx;
    
    idx = SPRITE_DEC_4( x, idx );
    idx = x >= 0.0 && x < 24.0 ? idx : 0.0;

    float blink = abs( sin( iTime * 3.0 ) ) + 0.5;
    color = idx == 1.0 ? RGB( 0,   0,   0   ) : color;
    color = idx == 2.0 ? RGB( 192, 192, 192 ) : color;
    color = idx == 3.0 ? RGB( 255, 255, 255 ) : color;
    color = idx == 0.0 && x >= 1.0 && x < 21.0 && y >= 3.0 && y < 30.0 ? blink * RGB( 228, 68, 52 ) : color;
}

// Function 5
float highscoreText( vec2 p )
{        
    vec2 scale = vec2( 4., 8. );
    vec2 t = floor( p / scale );
    
    uint v = 0u;    
	v = t.y == 0. ? ( t.x < 5. ? 2751607624u : ( t.x < 9. ? 2919902579u : 24949u ) ) : v;
	v = t.x >= 0. && t.x < 12. ? v : 0u;
    
	float c = float( ( v >> uint( 8. * t.x ) ) & 255u );
    
    p = ( p - t * scale ) / scale;
    p.x = ( p.x - .5 ) * .5 + .5;
    float sdf = textSDF( p, c );
    return ( c != 0. ) ? smoothstep( -.05, +.05, sdf ) : 1.0;
}

// Function 6
void add_mothership_score( inout ivec4 iscore ) {
    
    #define FIRST_FOR_300   23
    #define PERIOD_FOR_300  15
    
    int bullet_count = iscore.w - FIRST_FOR_300,
        sc = bullet_count >= 0 && bullet_count % PERIOD_FOR_300 == 0 ?
                 300
               : 50 + 50 * rand( 0, 3 ) ;
    iscore.x += sc ;
    iscore.z = sc | (iscore.z & ~0x1ff) ; //keep it in bottom 9 bits for renderer
}

// Function 7
void gs_pace_hud( inout GameState gs, float dt )
{
    if( gs.exposure.x > 0. )
    {
        uint hudbright = ( gs.switches & GS_HMD_BRIGHT_MASK ) >> GS_HMD_BRIGHT_SHIFT;
		gs.hudbright += -expm1( -4. * dt ) * ( exp2pp( 2. * float( hudbright ) - 6. ) * sqrt( gs.exposure.x + 0.00005 ) - gs.hudbright );
    }
}

// Function 8
vec3 Hud(vec2 uv, vec3 col) 
{
    uv *= 5.0; 
    float band = step(uv.y, 0.02);
    uv.x = abs(uv.x) - 0.5; // mirror hud

    float g2, g3, g4, glow;
    g2 = clmp(HudGlow(0.3 - uv.x*uv.y));
    g3 = clmp(HudGlow(0.5 + uv.y));
    g4 = clmp(HudGlow(0.2 + uv.y)); 
    glow = max(min(min(g4, g2), g2), g3);
    glow = 1.0 - exp(-clmp(glow)); // add glow

    uv.x = clamp(uv.x * 5.0, step(0.11, uv.x), 1.0); // chamfer
    uv.y = uv.y * 2.8 + 1.0; // y pos
    float hud = step(uv.y - uv.x, 0.0);

    col = mix(col, black, band);
    col = mix(col, dblue, max(hud, glow));
    
    return col;
}

// Function 9
float getScore( in int index ) {
    if (index < 8) return 1.0;
    if (index < 16) return 5.0;
    if (index < 24) return 10.0;
    if (index < 28) return 5.0;
    if (index < 30) return 50.0;
    if (index < 31) return 150.0;
    return 500.0;
}

// Function 10
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

// Function 11
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

// Function 12
void DrawScore(vec2 p, float score, inout vec3 color)
{
    vec3 pinkColor = vec3(0.650, 0.117, 0.745);
    vec3 blueColor = vec3(0.117, 0.352, 0.745);
    
    float deadt = max(0., gT - kFinishTime);
    
    vec2 startPos = vec2(0.);
    vec2 endPos = vec2(-1.2, -1. + sin(gT)*0.05);
    vec2 pos = mix(startPos, endPos, pow(min(deadt*0.45, 1.), 10.) );
    
    float scale = mix(1., 0.8, pow(min(deadt*0.45, 1.), 10.) );
    p = (p-pos)*scale;
    
    vec2 q = p - vec2(1., 0.85);
    q.x += q.y*0.35;
    float d = uRoundBox(q, vec2(0.5, 0.068), 0.01);
    vec3 bgColor = color*0.8 + blueColor*0.2;
    bgColor = mix(bgColor, blueColor, smoothstep(-0.1, 0.2, q.y));
    color = mix(bgColor, color, smoothstep(-0.0, 0.001, d));
    color = mix(blueColor, color, smoothstep(0.0, 0.01, abs(d)-0.001));
    
    q = p - vec2(0.6, 0.80);
	d = PrintInt(q*10., score);
    vec3 lettersColor = vec3(1.);
    lettersColor = mix(lettersColor, pinkColor, 1.-smoothstep(-0.1, 0.13, q.y));
    color = mix(lettersColor, color, 1.-smoothstep(-0.0, 0.001, d));
}

// Function 13
void HighscoreText(inout vec3 color, vec2 p, in AppState s)
{        
    vec2 scale = vec2(4., 8.);
    vec2 t = floor(p / scale);   
    
    uint v = 0u;    
	v = t.y == 0. ? ( t.x < 4. ? 1751607624u : ( t.x < 8. ? 1919902579u : 14949u ) ) : v;
	v = t.x >= 0. && t.x < 12. ? v : 0u;
    
	float c = float((v >> uint(8. * t.x)) & 255u);
    
    // vec3 textColor = vec3(.3);
	vec3 textColor = vec3(0.75);

    p = (p - t * scale) / scale;
    p.x = (p.x - .5) * .5 + .5;
    float sdf = TextSDF(p, c);
    if (c != 0.)
    {
    	color = mix(textColor, color, smoothstep(-.05, +.05, sdf));
    }
}

// Function 14
void Print_HudChar( inout PrintState printState, inout vec3 vResult, ivec3 vCharacter )
{
    float fBitmap = Font_DecodeBitmap( printState.vPos, vCharacter );
    float fShadow = Font_DecodeBitmap( printState.vPos - vec2( 1, 1), vCharacter );

    if ( fBitmap > 0.0 ) vResult = printState.vColor * 0.5 + 0.5 * (printState.vPos.y / 8.);
    else if ( fShadow > 0.0 ) vResult *= 0.5;
    
    printState.vPos.x -= float(vCharacter.z);
}

// Function 15
vec4 drawHealth( vec2 uv ) {
    uv = floor(fract(uv)*64.) - 32.;
    if( abs(uv.x) < 12. && abs(uv.y) < 12. ) {
        vec4 col = vec4( 1,1,1, smoothstep( 10., 9., length(uv)) );
        col.rgb = mix( col.rgb, vec3(1,0,0), step(abs(uv.y), 1.)*step(abs(uv.x),7.) );
        col.rgb = mix( col.rgb, vec3(1,0,0), step(abs(uv.y), 7.)*step(abs(uv.x),1.) );
        return vec4( 2.*col.rgb * (.5 + .5 * texture(iChannel1, uv/64.).x), col.a );
    } else {
        return vec4(0);
    }
}

// Function 16
float score(vec2 p){
    return -max(.5,length(texture(iChannel0, p/R.xy)));
}

// Function 17
vec3 drawScore( in vec3 col, in vec2 fragCoord, vec2 score, float lives )
{
    // score
    vec2 p = fragCoord/iResolution.y;
    // lives
    float eps = 1.0 / iResolution.y;
    for( int i=0; i<3; i++ )
    {
        float h = float(i);
        vec2 q = p - vec2(0.1 + 0.075*h, 0.7 );
        if( h + 0.5 < lives )
        {
            float c = max(0.0,sdCircle(q, 0.023));

            col += 0.17*vec3(1.0,0.8,0.0)*exp(-1500.0*c*c);
        }
    }

    return col;
}

// Function 18
vec3 drawScore( in vec3 col, in vec2 fragCoord, vec2 score, float lives )
{
    // score
    vec2 p = fragCoord/iResolution.y;
    col += float( PrintInt( (p - vec2(0.05,0.9))*20.0, int(score.x) ));
    col += float( PrintInt( (p - vec2(0.05,0.8))*20.0, int(242.0-score.y) ));
    
    // lives
    float eps = 1.0 / iResolution.y;
    for( int i=0; i<3; i++ )
    {
        float h = float(i);
        vec2 q = p - vec2(0.1 + 0.075*h, 0.7 );
        if( h + 0.5 < lives )
        {
            float c = sdCircle(q, 0.023);
            float f = c;

            {
                vec2 w = normalize( q - vec2(0.005,0.0) );
                w = vec2( w.x, abs( w.y ) );
                float an = 0.5;
                float m = dot( w, vec2(sin(an),cos(an)));
                f = max( f, -m );
            }
            f = 1.0 - smoothstep( -0.5*eps, 0.5*eps, f );
            col = mix( col, vec3(1.0,0.8,0.1), f );

            // glow
            //col += 0.15*vec3(1.0,0.8,0.0)*exp(-1500.0*c*c);
        }
    }

    return col;
}

// Function 19
float score(particle p, vec2 I, vec3 R, int seed){
    if(p.nil) return 1e6;
    
    vec2 Z = forward_mapping(p.coord, R, seed);
    
    vec2 D = Z-I;
    D = mod(D+R.xy/2.,R.xy)-R.xy/2.;
    return max(abs(D.x),abs(D.y));
    
}

// Function 20
vec4 DrawScoreGlow( vec2 uv, float keyID, vec2 size, float lifeTime)
{

    float lifeSpan = 0.5;
    
    lifeTime = clamp(lifeTime,0.,lifeSpan)/lifeSpan;
    size.y +=cos((lifeTime)*3.1415*2.)*0.04;
    
    size.x *=cos(lifeTime*3.1415);
    vec4 ret = vec4(0.);
    vec2 pos = vec2(keyID*0.2+0.1, size.y/2.);
    if ( abs(uv - pos).x < size.x/2. && abs(uv-pos).y < size.y/2. )
    {
        ret.a = 1.;
        ret.xyz = keyColors[int(keyID)]*pow((1.-lifeTime),20.);
           
    }
    return ret;
}

// Function 21
vec4 TexHealthBonus( vec2 vTexCoord, float fRandom, float fHRandom )
{
    float fLen = length( vTexCoord - vec2(8.0, 6.0) ) / 5.5;
    
    vec4 vResult = vec4(0);
    
    vec3 vCol = vec3(0,0,1);
    if ( all( greaterThan( vTexCoord, vec2( 6, 13) ) ) && all( lessThan( vTexCoord, vec2(10, 16 ) ) ) )
	{
        vCol = vec3(1., .5, .2) * 2.;
	}
    
    float fShade = clamp( vTexCoord.y / 10.0, 0.0, 1.0);
    vResult.rgb = vCol * fShade * fRandom;
    if ( fLen < 1.0 )
    {
        vResult.a = 1.;
    }
    
    if ( all( greaterThan( vTexCoord, vec2( 6, 4) ) ) && all( lessThan( vTexCoord, vec2(10, 16 ) ) ) )
	{
        vResult.a = 1.;
	}
    
    
    return vResult;
}

// Function 22
float highscoreText( vec2 p )
{        
    vec2 scale = vec2( 4., 8. );
    vec2 t = floor( p / scale );
    
    uint v = 0u;    
	v = t.y == 0. ? ( t.x < 4. ? 1751607624u : ( t.x < 8. ? 1919902579u : 14949u ) ) : v;
	v = t.x >= 0. && t.x < 12. ? v : 0u;
    
	float c = float( ( v >> uint( 8. * t.x ) ) & 255u );
    
    p = ( p - t * scale ) / scale;
    p.x = ( p.x - .5 ) * .5 + .5;
    float sdf = textSDF( p, c );
    return ( c != 0. ) ? smoothstep( -.05, +.05, sdf ) : 1.0;
}

// Function 23
ivec3 GetHudTextChar( int iChar ) 
{

	#define HUD_TEXT_CHAR(X) if ( iChar == 0 ) return X; iChar--
    
    HUD_TEXT_CHAR( ivec3(6,189, -1) ); // MOVE

    HUD_TEXT_CHAR( _A_ );
    HUD_TEXT_CHAR( _M_ );
    HUD_TEXT_CHAR( _M_ );
    HUD_TEXT_CHAR( _O_ );

    HUD_TEXT_CHAR( ivec3(52,189, -1) ); // MOVE
    
    HUD_TEXT_CHAR( _H_ );
    HUD_TEXT_CHAR( _E_ );
    HUD_TEXT_CHAR( _A_ );
    HUD_TEXT_CHAR( _L_ );
    HUD_TEXT_CHAR( _T_ );
    HUD_TEXT_CHAR( _H_ );
    
    HUD_TEXT_CHAR( ivec3(109,189, -1) ); // MOVE

    HUD_TEXT_CHAR( _A_ );
    HUD_TEXT_CHAR( _R_ );
    HUD_TEXT_CHAR( _M_ );
    HUD_TEXT_CHAR( _S_ );
    
    HUD_TEXT_CHAR( ivec3(187,189, -1) ); // MOVE

    HUD_TEXT_CHAR( _A_ );
    HUD_TEXT_CHAR( _R_ );
    HUD_TEXT_CHAR( _M_ );
    HUD_TEXT_CHAR( _O_ );
    HUD_TEXT_CHAR( _R_ );

    return ivec3(0);
}

// Function 24
void PrintHudMessage( vec2 vTexCoord, int iMessage, inout vec3 vResult )
{
    if ( vTexCoord.y > 8.0 )
        return;

    if ( iMessage >= MESSAGE_COUNT )
        return;
    
    // Message text
    PrintState printState;
    Print_Init( printState, vTexCoord );

    // Fixed size font
    //float fCharIndex = floor( printState.vPos.x / 8. );
    //printState.vPos.x -= fCharIndex * 8.0;
    //vec3 vChar = GetMessageChar( fMessage, fCharIndex );
    
    ivec3 vChar = _SPACE_;
    for ( int i=0; i<NO_UNROLL( 26 ); i++)
    {
        vChar = GetMessageChar( iMessage, i );
        if ( Print_Test( printState, vChar, 0.0 ) )
        {
            break;
        }
        if ( vChar.z == 0 )
            break;
    }
        
    if ( iMessage == MESSAGE_HUD_TEXT || iMessage == MESSAGE_HANGAR )
    {
		Print_Color( printState, vec3(1. ) );        
    	Print_HudChar( printState, vResult, vChar );
    }
    else
    {
    	Print_FancyChar( printState, vResult, vChar );
    }
}

// Function 25
void process_text_hud_numbers( int i, inout int N,
                               inout vec4 params, inout uvec4 phrase, inout vec4 argv,
                               GameState gs, PlanetState ps )
{
    float left = g_textres.x / 4.;
    float right = g_textres.x * 3. / 4.;
    float y = g_textres.y / 2.;
	vec3 localv = ( g_vehicle.localv + 
		cross( vec3( 0, 0, g_vehicle.modes.x == VS_HMD_ORB ? ps.omega : 0. ), g_vehicle.localr ) );
    float spd =  length( localv );
    if( spd >= 0.0005 && i == N++ )
    {
        // speed
        params = vec4( left - CW(4.,15.), y, 1, 15 );       
        if( spd < 9.9995 )
        	phrase = uvec4( 0x202020f5, 0, 0, 4 ), argv.x = 1000. * spd;
        else
		if( spd < 9999.995 )
        	phrase = uvec4( 0xfb6b0000, 0, 0, 2 ), argv.x = spd;
        else
        	phrase = uvec4( 0xfb4d0000, 0, 0, 2 ), argv.x = spd / 1000.;
	}
    if( g_env.H != 0. )
	{
        // mach 
        float mach = length( g_vehicle.localv ) / g_env.atm.w;
        if( mach >= 0.005 && i == N++ )
        {
	        params = vec4( left - CW(2.,12.), y - 16., 1, 12 );
    	    phrase = uvec4( 0x4df90000, 0, 0, 2 );
            argv.x = mach;
        }       
		// dyn pressure
        float Q = .5 * ( 1e6 / 1e5 ) * g_env.atm.z * dot( g_vehicle.localv, g_vehicle.localv );
        if( Q >= 0.005 && i == N++ )
        {            
	        params = vec4( left -CW(2.,12.), y - 32., 1, 12 );            
            phrase = uvec4( 0x51f90000, 0, 0, 2 );
			argv.x = Q;
        }
    }
    if( i == N++ )
    {
	    // altitude
    	float alt = length( g_vehicle.localr ) - g_data.radius;
        params = vec4( right - CW(8.,15.), y, 1, 15 );
        if( alt < 9.9995 )
        	phrase = uvec4( 0x202020f5, 0, 0, 4 ), argv.x = 1000. * alt;
        else
        if( alt < 9999.995 )
        	phrase = uvec4( 0xfb6b0000, 0, 0, 2 ), argv.x = alt;
        else
        	phrase = uvec4( 0xfb4d0000, 0, 0, 2 ), argv.x = alt / 1000.;
	}
    float vs = dot( localv, normalize( g_vehicle.localr ) );
    if( abs( vs ) >= 0.0000005 && i == N++ )
    {    
        // vertical speed
        params = vec4( right - CW(8.,12.), y - 15., 1, 12 );
        if( abs( vs ) < 9.9995 )
        	phrase = uvec4( abs( vs ) < 0.00995 ? 0x202020f6 : 0x202020f5, 0, 0, 4 ), argv.x = 1000. * vs;
        else
		if( abs( vs ) < 9999.5 )
        	phrase = uvec4( 0xfb6b0000, 0, 0, 2 ), argv.x = vs;
        else
        	phrase = uvec4( 0xfb4d0000, 0, 0, 2 ), argv.x = vs / 1000.;
	}
    if( i == N++ )
    {
        // heading
        params = vec4( g_textres.x / 2. - CW(1.5,12.), g_textres.y / 4., 1, 12 );
        phrase = uvec4( 0xf1000000, 0, 0, 1 );
        argv.x = B2bearing( g_vehicle.localr, g_vehicle.localB[0] ) + .5;
    }
    if( i == N++ )
    {
        // g-load
       	params = vec4( g_textres.x / 2. - CW(3.5,12.), g_textres.y / 4. - 18., 1, 12 );
	   	phrase = uvec4( 0x47f90000, 0, 0, 2 );
	    argv.x = -1000. / FDM_STD_G * g_vehicle.acc.z;
    }
}

// Function 26
void EnemyHealth(inout vec3 color, vec2 p)
{
    vec3 EHCOL1 = vec3(.7, .92, .56);
    vec3 EHCOL2 = vec3(.52, .47, .87);
    
    if(p.x > -5. && p.x < 62. && p.y > -17. && p.y < -8.)
        color = EHCOL1;
    if(p.x > -3. && p.x < 60. && p.y > -16. && p.y < -9.)
        color = EHCOL2;
}

// Function 27
float LoadScore(sampler2D txBuf, int idCreature) {
    return texelFetch(txBuf, ivec2(POS_SCORE, idCreature), 0).w;
}

// Function 28
vec3 posCore (vec3 p, float count) {
    
    // polar domain repetition
    vec3 m = moda(p.xz, count);
    p.xz = m.xy;
    
    // linear domain repetition
    float c = .2;
    p.x = mod(p.x,c)-c/2.;
    return p;
}

// Function 29
vec4 drawscore(vec2 p, float score)
{
    vec4 c = vec4(0.0);
    for (int i = 0; i < 6; ++i)
    {
        float dig = score;
        float dd = pow(10.0, float(5 - i));
        dig = mod(floor(dig / dd), 10.0);
		c += vec4(digit(dig, (p - vec2(125.0 + float(i) * 20.0, 110.0)) * 0.05));
    }
    
    return c;
}

// Function 30
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

// Function 31
vec3 DrawScore(vec2 p, vec3 color)
{
    vec2 q = p - vec2(1.8, 1.5) + vec2(10., 0.)*(1.-smoothstep(0., 1., gT-3.45));
    color = DrawUIBox(q*vec2(0.8, 1.), color);
    
    float d = PrintInt(q*3.0-vec2(-4.5, -1.0), gState.z*25.);
    vec3 lettersColor = vec3(0.188, 0.164, 0.133)*0.2;
    lettersColor = mix(lettersColor, vec3(0.396, 0.376, 0.345), smoothstep(-0.3, 0.4, q.y));
    color = mix(lettersColor, color, 1.-smoothstep(-0.0, 0.001, d));
    
    q -= vec2(-1.2, 0.35); q*=0.25;
    caret.x = count = 0.;
    d = S(r(q)); add(); d += C(r(q)); add(); d += O(r(q));  add(); d += R(r(q));  add(); d += E(r(q)); 
    color = mix(color, lettersColor*0.1, smoothstep(0.4, 1.0, d));
    color *= smoothstep(0., 0.005, length(q-vec2(0.27, 0.))-0.008);
    color *= smoothstep(0., 0.005, length(q-vec2(0.27, -0.035))-0.008);
    
    return color;
}

// Function 32
vec4 HUD( in vec2 uv, in vec2 p, in vec2 d )
{
    vec2 up = normalize(d);
    vec2 right = normalize(mat2(0,-1,1,0)*d);
    uv = uv.y * up + uv.x * right;
    uv *= HUD_SCALE;
    uv += p;
    // Exterior.
    float v = max( sBox(uv, vec2(4.0+LINE_WIDTH,4.0+LINE_WIDTH)),
                  -sBox(uv, vec2(4.0-LINE_WIDTH,4.0-LINE_WIDTH)));

    #define HWALL(x,y,w) sBox(uv+vec2(x,y),vec2(w,LINE_WIDTH))
    #define VWALL(x,y,w) sBox(uv+vec2(x,y),vec2(LINE_WIDTH,w))
    // Internal bits. These are the width-1 horizontal walls.
    v = min(v,HWALL(3.5, 3, .5));
    v = min(v,HWALL(2.5, 2, .5));
    v = min(v,HWALL(2.5, 0, .5));
    v = min(v,HWALL(1.5, 1, .5));
    v = min(v,HWALL(1.5, 1, .5));
    v = min(v,HWALL(0.5, 3, .5));
    v = min(v,HWALL(-0.5, 2, .5));
    v = min(v,HWALL(-2.5, 3, .5));
    v = min(v,HWALL(-3.5, 2, .5));
    v = min(v,HWALL(1.5, -1, .5));
    v = min(v,HWALL(0.5, -2, .5));
    v = min(v,HWALL(-2.5, -2, .5));
    
    // Time for the lengthy horizontal walls.
    v = min(v,HWALL(-2,   1, 1 ));
    v = min(v,HWALL(-0,   0, 1 ));
    v = min(v,HWALL(-3,   0, 1 ));
    v = min(v,HWALL( 3,  -2, 1 ));
    v = min(v,HWALL(-1,  -1, 1 ));
    v = min(v,HWALL( 2,  -3, 1 ));
    v = min(v,HWALL(-2.5,-3, 1.5));
    
    // And now for the height-1 vertical walls. (Looking from above)
    v = min(v,VWALL(-0,  1.5, .5));
    v = min(v,VWALL( 2, -1.5, .5));
    v = min(v,VWALL(-0, -1.5, .5));
    v = min(v,VWALL(-3, -1.5, .5));
    v = min(v,VWALL( 1, -2.5, .5));
    v = min(v,VWALL(-1, -2.5, .5));
    v = min(v,VWALL(-0, -3.5, .5));
    
    // Without further adu, the lengthy vertical walls.
    v = min(v,VWALL( 2, 3,1));
    v = min(v,VWALL( 3,.5,1.5));
    v = min(v,VWALL( 1, 1,2));
    v = min(v,VWALL(-1, 3,1));
    v = min(v,VWALL(-2, 2,1));
    v = min(v,VWALL(-2,-1,1));
    
    v = min(v, length(uv-p)-.125);
    
    vec4 r = vec4(0);
    r.a = step(v,LINE_WIDTH);
    r.rgb = vec3(step(v,.0));
    return r;
}

// Function 33
void drawScore( ivec2 uv, ivec2 rt, float score, inout vec3 col ) {
    for (int i=0; i<6; i++) {
        if (score > 0. || i == 0) {
            float s = mod(score, 10.);
            drawSprite(uv, rt, rt+ivec2(8,7), ivec2(72,73) + ivec2(s*8.,0), iChannel1, false, col);
            rt.x -= 8;
            score = floor(score * .1);
        }
    }
}

// Function 34
vec3 readScoreStates()
{
  	return getPixel(2,0).xyz;
}

// Function 35
void Health(inout vec3 color, vec2 p)
{
    uint v = 0u;
	v = p.y == 29. ? (p.x < 16. ? 1431655765u : (p.x < 32. ? 1431655765u : (p.x < 48. ? 1431655765u : 89478485u))) : v;
	v = p.y == 28. ? (p.x < 16. ? 1431655765u : (p.x < 32. ? 1431655765u : (p.x < 48. ? 1431655765u : 89478485u))) : v;
	v = p.y == 27. ? (p.x < 16. ? 2863291045u : (p.x < 32. ? 2863291050u : (p.x < 48. ? 2863291050u : 95050410u))) : v;
	v = p.y == 26. ? (p.x < 16. ? 1521134245u : (p.x < 32. ? 1521134250u : (p.x < 48. ? 1521134250u : 95070890u))) : v;
	v = p.y == 25. ? (p.x < 16. ? 1431655845u : (p.x < 32. ? 1431655765u : (p.x < 48. ? 1431655765u : 94721365u))) : v;
	v = p.y == 24. ? (p.x < 16. ? 1431655845u : (p.x < 32. ? 1431655765u : (p.x < 48. ? 1431655765u : 94721365u))) : v;
	v = p.y == 23. ? (p.x < 16. ? 1445u : (p.x < 32. ? 0u : (p.x < 48. ? 0u : 94699520u))) : v;
	v = p.y == 22. ? (p.x < 16. ? 1445u : (p.x < 32. ? 0u : (p.x < 48. ? 0u : 94699520u))) : v;
	v = p.y == 21. ? (p.x < 16. ? 1431635365u : (p.x < 32. ? 89478485u : (p.x < 48. ? 1431655765u : 94700885u))) : v;
	v = p.y == 20. ? (p.x < 16. ? 1431635365u : (p.x < 32. ? 89478485u : (p.x < 48. ? 1431655765u : 94700885u))) : v;
	v = p.y == 19. ? (p.x < 16. ? 2862941605u : (p.x < 32. ? 95050410u : (p.x < 48. ? 2863291045u : 94700970u))) : v;
	v = p.y == 18. ? (p.x < 16. ? 1520764325u : (p.x < 32. ? 95070890u : (p.x < 48. ? 1521134245u : 94700970u))) : v;
	v = p.y == 17. ? (p.x < 16. ? 1436878245u : (p.x < 32. ? 94721365u : (p.x < 48. ? 1431655845u : 94700965u))) : v;
	v = p.y == 16. ? (p.x < 16. ? 1436878245u : (p.x < 32. ? 94721365u : (p.x < 48. ? 1431655845u : 94700965u))) : v;
	v = p.y == 15. ? (p.x < 16. ? 94700965u : (p.x < 32. ? 94699520u : (p.x < 48. ? 1445u : 94700965u))) : v;
	v = p.y == 14. ? (p.x < 16. ? 94700965u : (p.x < 32. ? 94699520u : (p.x < 48. ? 1445u : 94700965u))) : v;
	v = p.y == 13. ? (p.x < 16. ? 94700965u : (p.x < 32. ? 94700885u : (p.x < 48. ? 89458085u : 94700965u))) : v;
	v = p.y == 12. ? (p.x < 16. ? 1436878245u : (p.x < 32. ? 94700885u : (p.x < 48. ? 1431635365u : 94700965u))) : v;
	v = p.y == 11. ? (p.x < 16. ? 1520764325u : (p.x < 32. ? 94700970u : (p.x < 48. ? 1520764325u : 94700970u))) : v;
	v = p.y == 10. ? (p.x < 16. ? 2862941605u : (p.x < 32. ? 94700970u : (p.x < 48. ? 2862941605u : 94700970u))) : v;
	v = p.y == 9. ? (p.x < 16. ? 1431635365u : (p.x < 32. ? 94700885u : (p.x < 48. ? 1431635365u : 94700885u))) : v;
	v = p.y == 8. ? (p.x < 16. ? 1431635365u : (p.x < 32. ? 94700885u : (p.x < 48. ? 1431635365u : 94700885u))) : v;
	v = p.y == 7. ? (p.x < 16. ? 1445u : (p.x < 32. ? 94699520u : (p.x < 48. ? 1445u : 94699520u))) : v;
	v = p.y == 6. ? (p.x < 16. ? 1445u : (p.x < 32. ? 94699520u : (p.x < 48. ? 1445u : 94699520u))) : v;
	v = p.y == 5. ? (p.x < 16. ? 1431655845u : (p.x < 32. ? 94721365u : (p.x < 48. ? 1431655845u : 94721365u))) : v;
	v = p.y == 4. ? (p.x < 16. ? 1431655845u : (p.x < 32. ? 94721365u : (p.x < 48. ? 1431655845u : 94721365u))) : v;
	v = p.y == 3. ? (p.x < 16. ? 1521134245u : (p.x < 32. ? 95070890u : (p.x < 48. ? 1521134245u : 95070890u))) : v;
	v = p.y == 2. ? (p.x < 16. ? 2863291045u : (p.x < 32. ? 95050410u : (p.x < 48. ? 2863291045u : 95050410u))) : v;
	v = p.y == 1. ? (p.x < 16. ? 1431655765u : (p.x < 32. ? 89478485u : (p.x < 48. ? 1431655765u : 89478485u))) : v;
	v = p.y == 0. ? (p.x < 16. ? 1431655765u : (p.x < 32. ? 89478485u : (p.x < 48. ? 1431655765u : 89478485u))) : v;
    v = p.x >= 0. && p.x < 62. ? v : 0u;

    float i = float((v >> uint(2. * p.x)) & 3u);
    color = i == 1. ? vec3(0.76, 0.51, 0.47) : color;
    color = i == 2. ? vec3(1) : color;
}

// Function 36
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

// Function 37
void HUD(inout vec3 finalColor, vec2 fragCoord) {
	vec2 uv = fragCoord.xy/iResolution.xy;// * 2.0 - 1.0;
    uv.y = 1.0 - uv.y;
    uv *= 8.0;
    uv.x *= 2.0;
    vec2 uvf = fract(uv);
    vec2 uvi = floor(uv);
    float map = texelFetch(iChannel0, ivec2(uvi), 0).w;
    vec2 charPos = vec2(mod(map, 16.0), floor(map / 16.0));
    vec4 tex = texture(iChannel2, (uvf + charPos) / 16.0f, -100.0);
	finalColor = finalColor * pow(saturate(tex.w+0.4), 6.0);
	finalColor = mix(finalColor, vec3(1.0), tex.x);
	//finalColor = vec3(uvi/16.0,0);
    //if (fragCoord.x < iTimeDelta*1000.0) finalColor = vec3(1);
}

// Function 38
Score CreateScoreStruct(vec4 info){
    Score score;
    score.Hit = int(info.x);
	score.Miss = int(info.y);
	return score;
}

// Function 39
float drawHUD( in vec2 uv, in vec4 fsrt, in vec4 psvl, in float state, in float pchange )
{
    vec2 scr = uv*vec2(320,180); // Let's make pixels from normalized screenspace.
    vec2 pos; // To store the current text position.
    // Character presence values for labels and warnings.
    float charL = 0.0, charV = 0.0, charW = 0.0;
    // Several things need to blink.
    float blink = floor(mod((iTime-pchange+.5)*2.0,2.0));
    // What to write when in gameplay.
    if( state > ST_GAMEP-.5 && state < ST_GAMEP+.5 )
    {
        // Score: {score}
        pos = vec2(10,165);
        charL += drawChar(CH_S,pos,MAP_SIZE,scr);
        charL += drawChar(CH_C,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_O,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_R,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_E,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_COLN,pos,MAP_SIZE,scr);
        pos.x += 4.*KERN; 
        charV += drawInt( fsrt.y*5000.0, pos, MAP_SIZE, scr);

        // Fuel: {fuel}
        pos.x =  10.0;
        pos.y -= 10.0;
        charL += drawChar(CH_F,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_U,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_E,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_L,pos,MAP_SIZE,scr);
        charL += drawChar(CH_COLN,pos,MAP_SIZE,scr); 
        pos.x += 5.0*KERN; 
        charV += drawInt( fsrt.x, pos, MAP_SIZE, scr);

        // Alt.: {alt}
        pos.x = 245.0;
        pos.y = 165.0;
        charL += drawChar(CH_A,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_L,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_T,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_FSTP,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_COLN,pos,MAP_SIZE,scr); 
        pos.x += 5.*KERN;
        charV += drawInt(distLunarSurface(psvl.xy)*400.0, pos, MAP_SIZE, scr);

        // H-Vel.: {hvel}
        pos.x = 245.0;
        pos.y -= 10.0;
        charL += drawChar(CH_H,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_HYPH,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_V,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_E,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_L,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_FSTP,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_COLN,pos,MAP_SIZE,scr); 
        pos.x += 3.*KERN; 
        charV += drawInt( psvl.z*20000.0, pos, MAP_SIZE, scr);

        // V-Vel.: {vvel}
        pos.x = 245.0;
        pos.y -= 10.0;
        charL += drawChar(CH_V,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_HYPH,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_V,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_E,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_L,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_FSTP,pos,MAP_SIZE,scr); 
        charL += drawChar(CH_COLN,pos,MAP_SIZE,scr); 
        pos.x += 3.*KERN; 
        charV += drawInt( psvl.w*20000.0, pos, MAP_SIZE, scr);
        
        if( fsrt.x < .1 )
        {
            // "OUT OF FUEL"
            pos.x = 127.0;
            pos.y = 165.0;
            charW += drawChar(CH_O,pos,MAP_SIZE,scr);
            charW += drawChar(CH_U,pos,MAP_SIZE,scr);
            charW += drawChar(CH_T,pos,MAP_SIZE,scr);
            pos.x += KERN; 
            charW += drawChar(CH_O,pos,MAP_SIZE,scr);
            charW += drawChar(CH_F,pos,MAP_SIZE,scr);
            pos.x += KERN; 
            charW += drawChar(CH_F,pos,MAP_SIZE,scr);
            charW += drawChar(CH_U,pos,MAP_SIZE,scr);
            charW += drawChar(CH_E,pos,MAP_SIZE,scr);
            charW += drawChar(CH_L,pos,MAP_SIZE,scr);
        }
        
        else if( fsrt.x < 100.0 )
        {
            // "LOW FUEL"
            pos.x = 137.0;
            pos.y = 165.0;
            charW += drawChar(CH_L,pos,MAP_SIZE,scr);
            charW += drawChar(CH_O,pos,MAP_SIZE,scr);
            charW += drawChar(CH_W,pos,MAP_SIZE,scr);
            pos.x += KERN; 
            charW += drawChar(CH_F,pos,MAP_SIZE,scr);
            charW += drawChar(CH_U,pos,MAP_SIZE,scr);
            charW += drawChar(CH_E,pos,MAP_SIZE,scr);
            charW += drawChar(CH_L,pos,MAP_SIZE,scr);
    	}
    }
    // What to write if the player has crashed.
    else if( state > ST_CRASH-.5 && state < ST_CRASH+.5 )
    {
        float sel = floor(mod(pchange,3.0));
        if(sel < 1.0)
        {
            // "DESTROYED!"
            pos = vec2(132,45);
            charL += drawChar(CH_D,pos,MAP_SIZE,scr);
            charL += drawChar(CH_E,pos,MAP_SIZE,scr);
            charL += drawChar(CH_S,pos,MAP_SIZE,scr);
            charL += drawChar(CH_T,pos,MAP_SIZE,scr);
            charL += drawChar(CH_R,pos,MAP_SIZE,scr);
            charL += drawChar(CH_O,pos,MAP_SIZE,scr);
            charL += drawChar(CH_Y,pos,MAP_SIZE,scr);
            charL += drawChar(CH_E,pos,MAP_SIZE,scr);
            charL += drawChar(CH_D,pos,MAP_SIZE,scr);
            charL += drawChar(CH_EXCL,pos,MAP_SIZE,scr);
        }
        else if(sel < 2.0)
        {
            // "DEAD!"
            pos = vec2(150,45);
            charL += drawChar(CH_D,pos,MAP_SIZE,scr);
            charL += drawChar(CH_E,pos,MAP_SIZE,scr);
            charL += drawChar(CH_A,pos,MAP_SIZE,scr);
            charL += drawChar(CH_D,pos,MAP_SIZE,scr);
            charL += drawChar(CH_EXCL,pos,MAP_SIZE,scr);
        }
        else
        {
            // "WHOOPSIE!"
            pos = vec2(135,45);
            charL += drawChar(CH_W,pos,MAP_SIZE,scr);
            charL += drawChar(CH_H,pos,MAP_SIZE,scr);
            charL += drawChar(CH_O,pos,MAP_SIZE,scr);
            charL += drawChar(CH_O,pos,MAP_SIZE,scr);
            charL += drawChar(CH_P,pos,MAP_SIZE,scr);
            charL += drawChar(CH_S,pos,MAP_SIZE,scr);
            charL += drawChar(CH_I,pos,MAP_SIZE,scr);
            charL += drawChar(CH_E,pos,MAP_SIZE,scr);
            charL += drawChar(CH_EXCL,pos,MAP_SIZE,scr);
        }
        // "100 FUEL LOST" This value is hard coded.
        pos.y -= 10.0;
        pos.x = 120.0;
     	charL += drawChar(CH_1,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_0,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_0,pos,MAP_SIZE,scr);
        pos.x += KERN;
     	charL += drawChar(CH_F,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_U,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_E,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_L,pos,MAP_SIZE,scr); 
        pos.x += KERN;  
     	charL += drawChar(CH_L,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_O,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_S,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_T,pos,MAP_SIZE,scr);   
        charL *= blink;            
    }
    // What to do in the event of a successful landing.
    else if( state > ST_SCCES-.5 && state < ST_SCCES+.5 )
    {        
        // "SUCCESS!"
        pos = vec2(135,45);
    	charL += drawChar(CH_S,pos,MAP_SIZE,scr);
    	charL += drawChar(CH_U,pos,MAP_SIZE,scr);
    	charL += drawChar(CH_C,pos,MAP_SIZE,scr);
    	charL += drawChar(CH_C,pos,MAP_SIZE,scr);
    	charL += drawChar(CH_E,pos,MAP_SIZE,scr);
    	charL += drawChar(CH_S,pos,MAP_SIZE,scr);
    	charL += drawChar(CH_S,pos,MAP_SIZE,scr);
    	charL += drawChar(CH_EXCL,pos,MAP_SIZE,scr);
        charL *= blink;  
        vec2 padLoc = vec2(psvl.x, psvl.y-distLunarSurface(psvl.xy));
        float padVal = padValue(padLoc);
        // "{pad value}x landing{! if greater than 4}"
        pos.y -= 10.0;
        pos.x = 125.0;
     	charL += drawChar(floatToChar(padVal),pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_X,pos,MAP_SIZE,scr);   
        pos.x += KERN;
     	charL += drawChar(CH_L,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_A,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_N,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_D,pos,MAP_SIZE,scr); 
     	charL += drawChar(CH_I,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_N,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_G,pos,MAP_SIZE,scr);   
     	charL += drawChar(CH_EXCL,pos,MAP_SIZE,scr)*step(3.5,padVal);
        charL *= blink;         
    }
        
    charW *= blink;
    return charL*.5 + charV + charW*.75;
    
}

// Function 40
float score(vec2 p, vec2 I, vec3 R){
    if(!inbounds(p,R.xy)) return 1e6; //Bad score for points outside boundry
    //This should get revamped, there is no reasoning to use
    //euclidean distance, this metric probably should reflect the tree strtucture
    //Maybe even output a simple 1 or 0 if the index of this texel leads to the leaf
    //node that this particle p is going towards
    
    //Difference in the noise when using this other metric suggests that 
    //this is indeed screwing performance (likelyhood of missing particles)
    vec2 D = p-I;
    D = mod(D+R.xy/2.,R.xy)-R.xy/2.;
    return max(abs(D.x),abs(D.y));
    //use l infinity in toroidal space
    
    //return dot2(I-p);
}

// Function 41
void writeScoreStates(vec3 ss, inout vec4 fragColor, vec2 fragCoord)
{
    if(isPixel(2,0,fragCoord)) fragColor.xyz=ss;
}

// Function 42
vec4 PrintHUDPercent(const in vec2 vStringUV, const in float fValue )
{
    float fMaxDigits = 3.0;
    if ((vStringUV.y < 0.0) || (vStringUV.y >= 1.0)) return vec4(0.0);
	float fLog10Value = log2(abs(fValue)) / log2(10.0);
	float fBiggestIndex = max(floor(fLog10Value), 0.0);
	float fDigitIndex = fMaxDigits - floor(vStringUV.x);
	float fCharacter = -1.0;
    
	if(fDigitIndex > (-0.0 - 1.01)) {
		if(fDigitIndex <= fBiggestIndex) {
			if(fDigitIndex == -1.0) {
				fCharacter = 10.0; // Percent
			} else {
				float fDigitValue = (abs(fValue / (pow(10.0, fDigitIndex))));
                float kFix = 0.0001;
                fCharacter = floor(mod(kFix+fDigitValue, 10.0));
			}		
		}
	}
    
    return NumFont_Char( fract(vStringUV), int(fCharacter) );
}

// Function 43
void DrawScore (vec2 fragCoord, float score, inout vec3 pixelColor)
{
    // keep score between 0000 and 9999
    score = clamp(score, 0.0, 9999.0);
    
    // digits numbered from right to left
    int digit0 = int(mod(score, 10.0));
    int digit1 = int(mod(score / 10.0, 10.0));
    int digit2 = int(mod(score / 100.0, 10.0));
    int digit3 = int(mod(score / 1000.0, 10.0));
    
    // digit index is from left to right though
    DrawDigit(fragCoord, digit0, 3, pixelColor);
    DrawDigit(fragCoord, digit1, 2, pixelColor);
    DrawDigit(fragCoord, digit2, 1, pixelColor);
    DrawDigit(fragCoord, digit3, 0, pixelColor);
}

// Function 44
vec4 GetHudText( vec2 vPos, float fHealth, float fArmor )
{    
    vPos = floor( vPos );
	vec4 vHealth = PrintHUDPercent( vec2( (vPos - vec2(33,12)) / vec2(14,16)), fHealth );
    if ( vHealth.a > 0.0 )
    	return vHealth;
    
	vec4 vArmor = PrintHUDPercent( vec2( (vPos - vec2(164,12)) / vec2(14,16)), fArmor );
    if ( vArmor.a > 0.0 )
    	return vArmor;
    
    return vec4(0.0);
}

// Function 45
void PrintHudMessage( vec2 vTexCoord, int iMessage, inout vec3 vResult )
{
    if ( vTexCoord.y > 8.0 || vTexCoord.y < 0.0 || vTexCoord.x < 0.0 || vTexCoord.x > 240. )
        return;     
    
    vec2 vUV = vec2( vTexCoord.x, vTexCoord.y );
    vUV.y += float(iMessage * 8);
    vUV.y = (iChannelResolution[0].y - 1.0) - vUV.y;
    vUV = floor( vUV ) + 0.5;
    vUV /= iChannelResolution[0].xy;
    vec4 vSample = texture(iChannel0, vUV);
	if( vSample.a > 0.0)
	{
        vResult = vSample.rgb;
	}
                    
    
                    /*
    // Message text
    PrintState printState;
    Print_Init( printState, vTexCoord );

    // Fixed size font
    //float fCharIndex = floor( printState.vPos.x / 8. );
    //printState.vPos.x -= fCharIndex * 8.0;
    //vec3 vChar = GetMessageChar( fMessage, fCharIndex );
    
    vec3 vChar = _SPACE_;
    for ( int i=0; i<32; i++)
    {
        vChar = GetMessageChar( fMessage, float(i) );
        if ( Print_Test( printState, vChar, 0.0 ) )
        {
            break;
        }
        if ( vChar.z == 0. )
            break;
    }
        	
    Print_FancyChar( printState, vResult, vChar );
	*/
}

// Function 46
float HudGlow(float dist)
{
    return pow(1.5 / dist * 0.05, 0.8);
}

