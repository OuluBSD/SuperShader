# post_processing_functions

**Category:** effects
**Type:** extracted

## Dependencies
texture_sampling, normal_mapping, lighting, raymarching

## Tags
lighting, texturing, effects, color

## Code
```glsl
// Reusable Post Processing Effect Functions
// Automatically extracted from effect-related shaders

// Function 1
void ProcessLightValue(float t
){oliPos[0]=vec3(6.,1.,sin(t))
 ;oliCol[0]=3.*vec4(0.2,1.,.2,1)
 ;oliPos[1]=vec3(-3,-2.2,sin(t*.3)*8.)
 ;oliCol[1]=2.*vec4(1,1,0.5,1)
 ;oliPos[2]=vec3(9.5,1.8,9.5)
 ;oliCol[2]=3.*max(0.,abs(sin(pi*t)))*vec4(1,.2,1,1);}

// Function 2
void process_text_map_markers( int i, inout int N,
                          	   inout vec4 params, inout uvec4 phrase, inout vec4 argv,
                               GameState gs )
{
    if( i == N )
    {
        params = vec4( g_textres.x / 2. - CW(19.,15.) / 2., g_textres.y / 6., 1, 15 );
        phrase = uvec4( 0x102020fb, 0x206b6d20, 0x20202012, 12 );
        float ls = 2. * g_textres.x / g_textres.y * CW(19.,15.) / g_textres.x * g_data.radius / gs.camzoom;
        argv.x = ls;
    }
    N++;
    float x = g_textres.x - 160.;
    float y = g_textres.y - 24.;
    if( gs.waypoint != ZERO )
    {
        vec4 loc = navb( gs.waypoint, ZERO );
        switch( i - N )
        {
        case 0: params = vec4( x, y,       1, 12 ); phrase = uvec4( 0xa7000000, 0, 0, 1 ); break;
        case 1: params = vec4( x, y - 16., 1, 12 ); phrase = uvec4( 0x6c617420, 0xfeb30000, 0, 6 ); argv.x = loc.x; break;
        case 2: params = vec4( x, y - 32., 1, 12 ); phrase = uvec4( 0x6c6f6e67, 0xfeb30000, 0, 6 ); argv.x = loc.y; break;
        case 3: params = vec4( x, y - 48., 1, 12 ); phrase = uvec4( 0x616c7420, 0xfe206b6d, 0, 8 ), argv.x = loc.z - g_data.radius; break;
        }
        N += 4;
        y -= 80.;
    }
    if( gs.mapmarker != ZERO )
    {
        vec4 loc = navb( gs.mapmarker, ZERO );
        switch( i - N )
        {
        case 0: params = vec4( x, y,       1, 12 ); phrase = uvec4( 0x4d61726b, 0x65720000, 0, 6 ); break;
        case 1: params = vec4( x, y - 16., 1, 12 ); phrase = uvec4( 0x6c617420, 0xfeb30000, 0, 6 ); argv.x = loc.x; break;
        case 2: params = vec4( x, y - 32., 1, 12 ); phrase = uvec4( 0x6c6f6e67, 0xfeb30000, 0, 6 ); argv.x = loc.y; break;
        case 3: params = vec4( x, y - 48., 1, 12 ); phrase = uvec4( 0x616c7420, 0xfe206b6d, 0, 8 ), argv.x = loc.z - g_data.radius; break;
        }
        N += 4;
    }
}

// Function 3
bool UI_ShouldProcessWindow( UIWindowState window )
{
    return !window.bMinimized && !window.bClosed;
}

// Function 4
vec3 colorGradingProcess(const in ColorGradingPreset p, in vec3 c){
  float originalBrightness = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = mix(c, c * colorTemperatureToRGB(p.colorTemperature), p.colorTemperatureStrength);
  float newBrightness = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c *= mix(1.0, (newBrightness > 1e-6) ? (originalBrightness / newBrightness) : 1.0, p.colorTemperatureBrightnessNormalization);
  c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, p.presaturation);
  return pow((p.gain * 2.0) * (c + (((p.lift * 2.0) - vec3(1.0)) * (vec3(1.0) - c))), vec3(0.5) / p.gamma);
}

// Function 5
vec2 UIDrawContext_ScreenPosToCanvasPos( UIDrawContext drawContext, vec2 vScreenPos )
{
    vec2 vViewPos = vScreenPos - drawContext.viewport.vPos;
    return vViewPos + drawContext.vOffset;
}

// Function 6
void ProcessCamPos(vec3 pos, vec4 rot)
{
    objects[o_cam].pos = pos;
    //objects[o_cam].pos = vec3(0,0,-5);
    objects[o_cam].rot = rot;
}

// Function 7
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	rgb = pow(rgb, vec3(0.45));

	// Then...
	#define CONTRAST 1.1
	#define SATURATION 1.4
	#define BRIGHTNESS 1.2
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);

	// Vignette...
	rgb *= .5+0.5*pow(180.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.3 );	

	return clamp(rgb, 0.0, 1.0);
}

// Function 8
void UI_ProcessWindowImageControl( inout UIContext uiContext, inout UIData uiData, int iControlId, int iData )
{
    UIWindowDesc desc;
    
    desc.initialRect = Rect( vec2(280, 24), vec2(180, 100) );
    desc.uControlFlags = WINDOW_CONTROL_FLAG_TITLE_BAR;
    desc.bStartClosed = false;
    desc.bStartMinimized = false;
    desc.bOpenWindow = false;
    desc.vMaxSize = vec2(100000.0);

    UIWindowState window = UI_ProcessWindowCommonBegin( uiContext, iControlId, iData, desc );
        
    // Controls...
    if ( UI_ShouldProcessWindow( window ) )
    {
		UILayout uiLayout = UILayout_Reset();
        
		UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );
        UI_ProcessSlider( uiContext, IDC_SLIDER_IMAGE_BRIGHTNESS, uiData.imageBrightness, uiLayout.controlRect );
    }
    
    UI_ProcessWindowCommonEnd( uiContext, window, iData );
}

// Function 9
vec4 post(in vec2 coord, in float time){
	vec2 uv = coord/iResolution.xy;  
    return vec4(0.1, 0.04, 0.16, 1.) + (vec4(image(vec2(uv.x - .0015, uv.y), time).r, image(uv, time).g, image(vec2(uv.x + .0015, uv.y), time).b, 1.)) * (1. + border(coord));                                                                                 
}

// Function 10
vec4 processSliders(in vec2 uv, out vec4 sliderVal) {
    sliderVal = textureLod(iChannel1,vec2(0),0.0);
    if(length(uv.xy)>1.) {
    	return textureLod(iChannel1,uv.xy/iResolution.xy,0.0);
    }
    return vec4(0);
}

// Function 11
bool Processingold(float c, float d
){float a=d-c*varWdth
 ;float b=(c*varWdth+varWdth)-c*varWdth//(c+1.)*varWdth-c*varWdth
 ;return  (0.<a&&a<b);}

// Function 12
void UI_ProcessColorPickerSV( inout UIContext uiContext, int iControlId, inout UIData_Color data, Rect pickerRect )
{
    bool bMouseOver = Inside( uiContext.vMouseCanvasPos, pickerRect ) && uiContext.bMouseInView;
    
    vec3 vHSV = data.vHSV;
    
    if ( uiContext.iActiveControl == IDC_NONE )
    {
        if ( uiContext.bMouseDown && (!uiContext.bMouseWasDown) && bMouseOver && !uiContext.bHandledClick )
        {
            uiContext.iActiveControl = iControlId;
            uiContext.bHandledClick = true;
        }
    }
    else
    if ( uiContext.iActiveControl == iControlId )
    {
        vec2 vPos = (uiContext.vMouseCanvasPos - pickerRect.vPos) / pickerRect.vSize;
        vPos = clamp( vPos, vec2(0), vec2(1) );
        
        vHSV.yz = vPos;
        vHSV.z = 1.0f - vHSV.z;
        
        if ( !uiContext.bMouseDown )
        {
            uiContext.iActiveControl = IDC_NONE;
        }
    }
    
    data.vHSV = vHSV;
    
    bool bActive = (uiContext.iActiveControl == iControlId);
    
    UI_DrawColorPickerSV( uiContext, bActive, vHSV, pickerRect );    
}

// Function 13
void UI_ProcessScrollbarX( inout UIContext uiContext, int iControlId, inout UIData_Value data, Rect sliderRect, float fHandleSize )
{    
    bool bMouseOver = Inside( uiContext.vMouseCanvasPos, sliderRect ) && uiContext.bMouseInView;
        
    if ( uiContext.iActiveControl == IDC_NONE )
    {
        if ( uiContext.bMouseDown && (!uiContext.bMouseWasDown) && bMouseOver && !uiContext.bHandledClick )
        {
            uiContext.iActiveControl = iControlId;
            uiContext.bHandledClick = true;
        }
    }
    else
    if ( uiContext.iActiveControl == iControlId )
    {
        float fSlidePosMin = sliderRect.vPos.x + fHandleSize * 0.5f;
        float fSlidePosMax = sliderRect.vPos.x + sliderRect.vSize.x - fHandleSize * 0.5f;
        float fPosition = (uiContext.vMouseCanvasPos.x - fSlidePosMin) / (fSlidePosMax - fSlidePosMin);
        fPosition = clamp( fPosition, 0.0f, 1.0f );
        data.fValue = data.fRangeMin + fPosition * (data.fRangeMax - data.fRangeMin);
        if ( !uiContext.bMouseDown )
        {
            uiContext.iActiveControl = IDC_NONE;
        }
    }
        
    bool bActive = (uiContext.iActiveControl == iControlId);
    float fPosition = (data.fValue - data.fRangeMin) / (data.fRangeMax - data.fRangeMin);
    
    UI_DrawSliderX( uiContext, bActive, bMouseOver, fPosition, sliderRect, fHandleSize, true );    
}

// Function 14
void post_filmic( inout vec3 col )
{
#if WITH_IMG_FILMIC
    float b = ( g_vrmode ? IMG_FILMIC_VR : IMG_FILMIC ) * min( 1.0, IMG_FILMIC_EXPOSURE_REF * g_exposure.y );
    float c = 0.166666667 * ( sqrt( 200. * ( b * b + b ) + 9. ) + 3. ) / ( b + 1. );
    vec3 d = col * c;
    col = ( 1. + b ) * d * d / ( b + d );
#endif
}

// Function 15
vec3 ApplyPostFX(const in vec2 vUV, const in vec3 vInput) {
	vec3 vFinal = vInput;
	if(apply_vignetting) vFinal = ApplyVignetting(vUV, vFinal);
	if(apply_tonemap) vFinal = ApplyTonemap(vFinal);
	if(apply_gamma) vFinal = ApplyGamma(vFinal);
	if(apply_crushedshadows) vFinal = vFinal* 1.1- 0.1;
	return vFinal;
}

// Function 16
vec3 postProcess(vec3 col, vec2 q)  {
  col = clamp(col,0.0,1.0);
//  col=pow(col,vec3(0.75));
  col=col*0.6+0.4*col*col*(3.0-2.0*col);
  col=mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col*=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

// Function 17
vec3 postEffects( in vec3 col, in vec2 uv )
{    
    // gamma correction
	//col = pow( clamp(col,0.0,1.0), vec3(0.6) );
	//vignetting
	col *= 0.5+0.6*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.8 );
    //noise
    col -= snoise((uv*3.+iTime)*1000.)*.1;
    //col = mix(bw(col), col, sound*3.);
    col*=(1.6,.9,.9);
	return col;
}

// Function 18
void post_overlay( inout vec3 col, vec2 coord, vec3 cc )
{
    bool mapmode = ( g_game.switches & GS_TRMAP ) == GS_TRMAP;
	if( g_game.stage == GS_RUNNING )
    {
    	if( !mapmode )
        {
	    	if( ( g_game.switches & GS_NVISN ) == GS_NVISN )
    	    	hmd_night_vision( col, coord, cc );
    		if( ( g_game.switches & GS_TRDAR ) == GS_TRDAR )
        		hmd_terrain_radar( col, coord, cc );
            if( g_vehicle.modes.x > 0 )
				post_hmd_overlay( col, coord, cc );
        	post_console_overlay( col, coord );
    	}
        else
			post_map_overlay( col, coord );
    }
    post_text_overlay( col, coord, cc );
}

// Function 19
vec3 ApplyPostFx( const in vec3 vIn, const in vec2 fragCoord )
{
	vec2 vUV = fragCoord.xy / iResolution.xy;
	vec2 vCentreOffset = (vUV - 0.5) * 2.0;
	
	vec3 vResult = vIn;
	vResult.xyz *= clamp(1.0 - dot(vCentreOffset, vCentreOffset) * 0.4, 0.0, 1.0);

	vResult.xyz = 1.0 - exp(vResult.xyz * -kExposure);
	
	vResult.xyz = pow(vResult.xyz, vec3(1.0 / 2.2));
	
	return vResult;
}

// Function 20
void ProcessObjectPos(float time)
{
    
    objects[o_myCube].pos = vec3(0,0,0);
    objects[o_myCube].rot = BuildQuat(vec3(0,1,0),time*2.);
    objects[o_myCube].scale = vec3(0.8);
    
    objects[o_blackHole].pos = vec3(5.,sin(time*0.2),-5.);
    objects[o_blackHole].rot = BuildQuat(vec3(0,1,0),time*2.);
    objects[o_blackHole].scale = vec3(1.);
    
    objects[o_myCubeChildren].pos = vec3(1,1,1);
    objects[o_myCubeChildren].rot = BuildQuat(normalize(objects[o_myCubeChildren].pos),time*1.);
    //o_myCubeChildren.rot = vec4(0,0,0,1);
    objects[o_myCubeChildren].scale = vec3(.4,.4,.4);
    
    float trainV = 2.2;
    objects[o_train].vel = vec3((floor(mod(trainV*time/16.,2.))*2.-1.)*trainV,0,0); 
    
    
    float trainDir = 1.;
    if (objects[o_train].vel.x < 0.)
        trainDir = -1.;
    
    objects[o_train].pos = vec3(abs(1.-mod(trainV*time/16.,2.))*16.-8.,-.8,9.);
    objects[o_train].rot = BuildQuat(vec3(0,1,0),PI*.5);
    objects[o_train].scale = vec3(1.,1.,trainDir/mix(LorentzFactor(trainV*LgthContraction),1.,photonLatency)); ///
    
    //objects[o_train].pos.x = 0.;
    objects[o_tunnel].pos = vec3(0.,-.8,9.);
    objects[o_tunnel].rot = BuildQuat(vec3(0,1,0),PI*.5);
    objects[o_tunnel].scale = vec3(1.,1.,1);
    
    objects[o_tunnel_door].pos = objects[o_tunnel].pos;
    objects[o_tunnel_door].rot = objects[o_tunnel].rot;
    float open = clamp((1.-abs(3.*objects[o_train].pos.x))*2.,0.,1.);
    objects[o_tunnel_door].scale = vec3(open,open,1);
}

// Function 21
void gs_process_map_mode( inout GameState gs )
{
    if( keypress( KEY_TAB ) == 1. )
        gs.menustate.x = gs.menustate.x != 0 ? 0 : MENU_MAP;

	float zoomspeed = max( keystatepress( KEY_W ), keystatepress( KEY_Z ) ) -
                           keystatepress( KEY_S );
    zoomspeed *= ( keystate( KEY_SHIFT ) > 0. ? .25 : 1. );
	gs.camzoom = clamp( gs.camzoom * exp2pp( iTimeDelta * zoomspeed ), 0.5, 2048. * TRN_SCALE );

    if( iMouse.z < 0. && gs.dragstate.xy == -iMouse.zw )
    {
        vec4 marker = gs_map_unproject( gs, iMouse.xy + .5, iResolution.xy );
        if( abs( marker.w ) < 1. )
        {            
            marker *= ( g_data.radius + texelFetch( iChannel1, ivec2( iMouse.xy ), 0 ).w );
            if( marker.xyz == gs.mapmarker )
                gs.mapmarker = ZERO;
            else
                gs.mapmarker = marker.xyz;
        }
        gs.dragstate.xy = vec2(0);        
    }
}

// Function 22
bool ProcessingOldest(float a, float pos
){return  (pos>a*varWdth)        //-a*varWdth
        &&(pos<(a+1.)*varWdth);}

// Function 23
void _post(inout vec4 o) { 
    o = 1.-o; 
}

// Function 24
vec3 TonemapProcess( vec3 c )
{
    float YOrig = GetBT709Luminance( c );
    
    // Sort of hue preserving tonemap by scaling the original color by the original and tonempped luminance
    float YNew = GetBT709Luminance( whitePreservingLumaBasedReinhardToneMapping( c ) );
    vec3 result = c * YNew / YOrig;
    
    float desaturated = GetBT709Luminance( result );
        
	// Stylistic desaturate based on luminance - we want pure primary red to desaturate _slightly_ when bright
	float sdrDesaturateSpeed = 0.2f;
	float stylisticDesaturate = TonemapFloat( YOrig * sdrDesaturateSpeed );
    
    
	float stylisticDesaturateScale = 0.8f; // never fully desaturate bright colors
	stylisticDesaturate *= stylisticDesaturateScale;    
    
    result = mix( result, vec3(desaturated), stylisticDesaturate );
    
    return result;
}

// Function 25
UIWindowState UI_ProcessWindowCommonBegin( inout UIContext uiContext, int iControlId, int iData, UIWindowDesc desc )
{   
    UIWindowState window = UI_GetWindowState( uiContext, iControlId, iData, desc );
        
    if ( window.bClosed )
    {
        return window;
    }
    
    UI_PanelBegin( uiContext, window.panelState );
    
    uiContext.vWindowOutColor.rgba = vec4( cWindowBackgroundColor, 1.0 );
    
    window.drawRect = window.rect;
    
    Rect contextRect = window.drawRect;    
    RectShrink( contextRect, UIStyle_WindowBorderSize() );
    
    vec2 vTitleBarSize = UI_WindowGetTitleBarSize( uiContext, window );
    if ( window.bMinimized )
    {
	    window.drawRect.vSize.y = vTitleBarSize.y + UIStyle_WindowBorderSize().y * 2.0;
    }
    
    // Get window main panel view
    Rect panelRect = contextRect;
    
    panelRect.vPos.y += vTitleBarSize.y;
    panelRect.vSize.y -= vTitleBarSize.y;
    
    if ( window.bMinimized )
    {
        panelRect.vSize.y = 0.0;
    }           
    
    
    UIDrawContext panelDesc = UIDrawContext_SetupFromRect( panelRect );
    UIDrawContext panelContext = UIDrawContext_TransformChild( window.panelState.parentDrawContext, panelDesc );
    UI_SetDrawContext( uiContext, panelContext );
    
    if ( FLAG_SET(window.uControlFlags, WINDOW_CONTROL_FLAG_RESIZE_WIDGET) )
    {
        int iWindowResizeControlId = window.iControlId + 2000; // hack        
    	UI_ProcessWindowResizeWidget( uiContext, window, iWindowResizeControlId );
    }
            
    // Get window content panel view
    UIDrawContext contentPanelDesc;
    contentPanelDesc.viewport = Rect( vec2(0.0), uiContext.drawContext.viewport.vSize );
    RectShrink( contentPanelDesc.viewport, UIStyle_WindowContentPadding() );
    contentPanelDesc.vOffset = vec2(0);
    contentPanelDesc.vCanvasSize = contentPanelDesc.viewport.vSize;

    UI_SetDrawContext( uiContext, UIDrawContext_TransformChild( panelContext, contentPanelDesc ) ); 
    
    return window;
}

// Function 26
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

// Function 27
float posterize(float component, float colors)
{
    float temp = floor(pow(component, 0.6) * colors) / colors;
    return pow(temp, 1.666667);
}

// Function 28
void post_text_overlay( inout vec3 col, vec2 coord, vec3 cc )
{
    coord = ( coord - g_overlayframe.xy ) * g_textscale;
	for( int i = 0; i < TXT_FMT_MAX_COUNT; ++i )
		col += g_hudcolor * hmd_txtout( coord, cc, i );
}

// Function 29
void process_text_message_line( int i, inout int N,
                            	inout vec4 params, inout uvec4 phrase, inout vec4 argv )
{
    float x = ( 1. + 2. * ( 1. - fract( 1. - memload( iChannel0, ADDR_MESSAGES, 0 ).x ) ) ) * g_textres.x / 2.;
    float y = g_textres.y / 4. + 16.;
    switch( i - N )
    {
    case 0:
        params = vec4( x - g_textres.x, y, 1, 15 );
        phrase = unpack_uvec4( memload( iChannel0, ADDR_MESSAGES, 1 ) );
        phrase.w |= TXT_FMT_FLAG_CENTER | TXT_FMT_FLAG_HUDCLIP;
        argv = memload( iChannel0, ADDR_MESSAGES, 1 + TXT_MSG_MAX_PHRASES );
        break;
    case 1:
        params = vec4( x, y, 1, 15 );
        phrase = unpack_uvec4( memload( iChannel0, ADDR_MESSAGES, 2 ) );
        phrase.w |= TXT_FMT_FLAG_CENTER | TXT_FMT_FLAG_HUDCLIP;
        argv = memload( iChannel0, ADDR_MESSAGES, 2 + TXT_MSG_MAX_PHRASES );
        break;
    }
    N += 2;
}

// Function 30
bool UI_ProcessWindowMinimizeWidget( inout UIContext uiContext, inout UIWindowState window, int iControlId, Rect minimizeBoxRect )
{    
    bool bPressed = UI_ProcessButton( uiContext, iControlId, minimizeBoxRect );
    
    if ( bPressed )
    {
 		window.bMinimized = !window.bMinimized;        
    }

    bool bActive = (uiContext.iActiveControl == iControlId);
    
    return UI_DrawWindowMinimizeWidget( uiContext, window.bMinimized, minimizeBoxRect );
}

// Function 31
void UI_ProcessWindowCommonEnd( inout UIContext uiContext, inout UIWindowState window, int iData )
{    
    bool bPixelInPanel = uiContext.bPixelInView;
    
    Rect contextRect = window.drawRect;    
    RectShrink( contextRect, UIStyle_WindowBorderSize() );
    
    UIDrawContext windowContextDesc = UIDrawContext_SetupFromRect( contextRect );
    UIDrawContext windowContext = UIDrawContext_TransformChild( window.panelState.parentDrawContext, windowContextDesc );
	UI_SetDrawContext( uiContext, windowContext );
    
    bool inTitleBar = false;
    if (  FLAG_SET(window.uControlFlags, WINDOW_CONTROL_FLAG_TITLE_BAR)  )
    {
    	inTitleBar = UI_ProcessWindowTitleBar( uiContext, window );
    }
    
    UIDrawContext windowBackgroundContextDesc = UIDrawContext_SetupFromRect( window.drawRect );
    UIDrawContext windowBackgroundContext = UIDrawContext_TransformChild( window.panelState.parentDrawContext, windowBackgroundContextDesc );    

    UI_SetDrawContext( uiContext, windowBackgroundContext );
    if ( !bPixelInPanel && !inTitleBar )
    {
        Rect rect = Rect( vec2(0), window.drawRect.vSize );
#ifdef NEW_THEME        
	    DrawBorderRect( uiContext.vPixelCanvasPos, rect, cWindowBorder, uiContext.vWindowOutColor );                            
#else        
	    DrawBorderOutdent( uiContext.vPixelCanvasPos, rect, uiContext.vWindowOutColor );                    
#endif
        
    }    
    
    if ( uiContext.bMouseDown && uiContext.bMouseInView && !uiContext.bHandledClick )
    {
        uiContext.bHandledClick = true;
    }
    
    Rect windowRect = uiContext.drawContext.clip;

    UI_PanelEnd( uiContext, window.panelState );
    UI_ComposeWindowLayer( uiContext, UIStyle_WindowTransparency(), windowRect );
    
    UI_StoreWindowState( uiContext, window, iData );    
}

// Function 32
void UI_ProcessWindowMain( inout UIContext uiContext, inout UIData uiData, int iControlId, int iData )
{
    UIWindowDesc desc;
    
    desc.initialRect = Rect( vec2(32, 128), vec2(380, 180) );
    desc.bStartMinimized = false;
    desc.bStartClosed = false;
    desc.bOpenWindow = false;    
    desc.uControlFlags = WINDOW_CONTROL_FLAG_TITLE_BAR | WINDOW_CONTROL_FLAG_MINIMIZE_BOX | WINDOW_CONTROL_FLAG_RESIZE_WIDGET;    
    desc.vMaxSize = vec2(100000.0);

    UIWindowState window = UI_ProcessWindowCommonBegin( uiContext, iControlId, iData, desc );
    
    if ( UI_ShouldProcessWindow( window ) )
    {
        // Controls...

        UILayout uiLayout = UILayout_Reset();
               
        LayoutStyle style;
        RenderStyle renderStyle;             
        UIStyle_GetFontStyleWindowText( style, renderStyle );       
        
		UILayout_StackControlRect( uiLayout, UIStyle_CheckboxSize() );                
        UI_ProcessCheckbox( uiContext, IDC_CHECKBOX_BACKGROUND_IMAGE, uiData.backgroundImage, uiLayout.controlRect );
        UILayout_StackRight( uiLayout );
        //UILayout_StackDown( uiContext.uiLayout );
        
		UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );                
        UI_ProcessSlider( uiContext, IDC_SLIDER_BACKGROUND_BRIGHTNESS, uiData.backgroundBrightness, uiLayout.controlRect );
        UILayout_StackRight( uiLayout );

        {
        	PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
            uint strA[] = uint[] ( _V, _a, _l, _u, _e, _COLON, _SP );

            ARRAY_PRINT(state, style, strA);

            Print(state, style, uiData.backgroundBrightness.fValue, 2 );

            UI_RenderFont( uiContext, state, style, renderStyle );
            
			UILayout_SetControlRectFromText( uiLayout, state, style );
        }
        
        UILayout_StackDown( uiLayout );    

		UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );                
        UI_ProcessSlider( uiContext, IDC_SLIDER_BACKGROUND_SCALE, uiData.backgroundScale, uiLayout.controlRect );       
        //UILayout_StackDown( uiContext.uiLayout );    
        UILayout_StackRight( uiLayout );

        {
            PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
            uint strA[] = uint[] ( _V, _a, _l, _u, _e, _COLON, _SP );
            ARRAY_PRINT(state, style, strA);
            Print(state, style, uiData.backgroundScale.fValue, 1 );
            UI_RenderFont( uiContext, state, style, renderStyle );
			UILayout_SetControlRectFromText( uiLayout, state, style );
        }
        UILayout_StackDown( uiLayout );
                        
        {
            // Draw color swatch
            vec2 vSwatchSize = vec2( uiLayout.controlRect.vSize.y);
			UILayout_StackControlRect( uiLayout, vSwatchSize );
            if (uiContext.bPixelInView)
            {
                DrawRect( uiContext.vPixelCanvasPos, uiLayout.controlRect, vec4(hsv2rgb(uiData.bgColor.vHSV), 1.0), uiContext.vWindowOutColor );
            }
        }
        
        bool buttonAPressed = UI_ProcessButton( uiContext, IDC_BUTTONA, uiLayout.controlRect ); // Get button position from prev control
        uiData.buttonA.bValue = buttonAPressed; // Only need to do this if we use it in another buffer
        
        if ( buttonAPressed )
        {
            uiData.editWhichColor.fValue = 0.0;
        }        
        
        UILayout_StackRight( uiLayout );        
        {
            PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );
            uint strA[] = uint[] ( _B, _G, _SP, _C, _o, _l );
            ARRAY_PRINT(state, style, strA);
			UILayout_SetControlRectFromText( uiLayout, state, style );            
            UI_RenderFont( uiContext, state, style, renderStyle );
             
        }
                
        UILayout_StackRight( uiLayout );        
        
        {
            // Draw color swatch
            vec2 vSwatchSize = vec2(uiLayout.controlRect.vSize.y);
			UILayout_StackControlRect( uiLayout, vSwatchSize );
            if (uiContext.bPixelInView)
            {
                DrawRect( uiContext.vPixelCanvasPos, uiLayout.controlRect, vec4(hsv2rgb(uiData.imgColor.vHSV), 1.0), uiContext.vWindowOutColor );
            }
        }

        bool buttonBPressed = UI_ProcessButton( uiContext, IDC_BUTTONB, uiLayout.controlRect );        
        
        if ( buttonBPressed )
        {
            uiData.editWhichColor.fValue = 1.0;
        }        

        UILayout_StackRight( uiLayout );        
        
        {
            PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );                    
            uint strA[] = uint[] ( _I, _M, _G, _SP, _B, _SP, _C, _o, _l );
            ARRAY_PRINT(state, style, strA);			            
			UILayout_SetControlRectFromText( uiLayout, state, style );            
            UI_RenderFont( uiContext, state, style, renderStyle );            
        }
        
        UILayout_StackDown( uiLayout );        
        
        #if 1
        // Debug state
        {
            PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );
            uint strA[] = uint[] ( _C, _t, _r, _l, _COLON );
            ARRAY_PRINT(state, style, strA);

            Print(state, style, uiContext.iActiveControl );
            UI_RenderFont( uiContext, state, style, renderStyle );

            UILayout_SetControlRectFromText( uiLayout, state, style );            
        }        
        #endif
    }    
    
    UI_ProcessWindowCommonEnd( uiContext, window, iData );
}

// Function 33
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

// Function 34
void process_text_console( int i, inout int N,
                           inout vec4 params, inout uvec4 phrase, inout vec4 argv )
{
    vec3 FSG_distance = abs( g_vehicle.FSG - ONE );
    FSG_distance.x = min( FSG_distance.x, abs( g_vehicle.FSG.x - 1./9. ) );
    FSG_distance.x = min( FSG_distance.x, abs( g_vehicle.FSG.x - 4./9. ) );

    vec3 FSG_light = max( vec3( .25 ),
                          min( step( FRACT_1_64, g_vehicle.FSG ),
                          	   max( vec3( step( .5, fract( iTime ) ) ),
                                    1. - step( FRACT_1_64, FSG_distance ) ) ) );

    const uvec2 aero_modes[] = uvec2[] (
        uvec2( 0x4f464600, 3u | TXT_FMT_FLAG_CENTER ),
        uvec2( 0x4d414e00, 3u | TXT_FMT_FLAG_CENTER ),
        uvec2( 0x46425700, 3u | TXT_FMT_FLAG_CENTER )
    );

    const uvec2 rcs_modes[] = uvec2[] (
        uvec2( 0x4f464600, 3u | TXT_FMT_FLAG_CENTER ),
        uvec2( 0x4d414e00, 3u | TXT_FMT_FLAG_CENTER ),
        uvec2( 0x52415445, 4u | TXT_FMT_FLAG_CENTER ),
        uvec2( 0x4c564c48, 4u | TXT_FMT_FLAG_CENTER )
    );

    const uvec2 thr_modes[] = uvec2[] (
        uvec2( 0x4f464600, 3u | TXT_FMT_FLAG_CENTER ),
        uvec2( 0x4d414e00, 3u | TXT_FMT_FLAG_CENTER )
    );

    const uvec2 eng_modes[] = uvec2[] (
        uvec2( 0x4f464600, 3u | TXT_FMT_FLAG_CENTER ),
        uvec2( 0x44525600, 3u | TXT_FMT_FLAG_CENTER ),
        uvec2( 0x494d5000, 3u | TXT_FMT_FLAG_CENTER ),
        uvec2( 0x4e4f5641, 4u | TXT_FMT_FLAG_CENTER )
    );
    
    bool trimdisplay =
        max( keystate( KEY_CTRL ), keystate( KEY_META ) ) > 0. && ( g_vehicle.modes2.x != VS_AERO_OFF );

#if WORKAROUND_08_UINT2FLOAT
    float tvec = float( int( ( g_vehicle.switches & VS_TVEC_MASK ) >> VS_TVEC_SHIFT ) );
#else
    float tvec = float( ( g_vehicle.switches & VS_TVEC_MASK ) >> VS_TVEC_SHIFT );
#endif
    float tvec_target = tvec * ( 84. + tvec * ( tvec * 3. - 27. ) );
    float tvec_distance = abs( g_vehicle.tvec - tvec_target );
    float tvec_light = max( step( .5, fract( iTime ) ), 1. - step( 2.5, tvec_distance ) );

    switch( i - N )
    {
    case 0:
        argv.x = 100. * g_vehicle.throttle;
	#if WORKAROUND_04_VEC4
        params = vec4( 32, 8., abs( sign( g_vehicle.throttle ) ), 12 );
	#else
        params = vec4( 32, 8, abs( sign( g_vehicle.throttle ) ), 12 );
	#endif
        phrase = uvec4( abs( argv.x ) < 9.95 ? 0xf6000000 : 0xf4000000, 0, 0, 1 );
        break;
    case 1:
		params = vec4( 96, 8, 1, 12 );
	#if WORKAROUND_05_UVEC4
        phrase = uvec4( 0x13131313u, 0u, 0u, ( g_vehicle.switches & VS_FLAPS_MASK ) >> VS_FLAPS_SHIFT | TXT_FMT_FLAG_RIGHT );
	#else
		phrase = uvec4( 0x13131313, 0, 0, ( g_vehicle.switches & VS_FLAPS_MASK ) >> VS_FLAPS_SHIFT | TXT_FMT_FLAG_RIGHT );
	#endif
        break;
	case 2: params = vec4( 104, 8, FSG_light.x, 15 ); phrase = uvec4( 0x46000000, 0, 0, 1 ); break;
    case 3: params = vec4( 120, 8, FSG_light.y, 15 ); phrase = uvec4( 0x53000000, 0, 0, 1 ); break;
    case 4: params = vec4( 136, 8, FSG_light.z, 15 ); phrase = uvec4( 0x47000000, 0, 0, 1 ); break;
    case 5: params = vec4( 168, 8, g_vehicle.modes2.x == 0 ? .25 : 1., 12 ); phrase = aero_modes[ clamp( g_vehicle.modes2.x, 0, 2 ) ].xxxy; break;
    case 6: params = vec4( 200, 8, g_vehicle.modes2.y == 0 ? .25 : 1., 12 ); phrase = rcs_modes[ clamp( g_vehicle.modes2.y, 0, 3 ) ].xxxy; break;
    // case 7: params = vec4( 232, 8, g_vehicle.modes2.z == 0 ? .25 : 1., 12 ); phrase = thr_modes[ clamp( g_vehicle.modes2.z, 0, 1 ) ].xxxy; break;
    case 8: params = vec4( 264, 8, g_vehicle.modes.z == 0 ? .25 : 1., 12 ); phrase = eng_modes[ clamp( g_vehicle.modes.z, 0, 3 ) ].xxxy; break;
    case 9:
        argv.x = 100. * g_vehicle.trim;
        params = vec4( g_textres.x * .5, 8, float( trimdisplay ), 12 );
        phrase = uvec4( 0x5452494d, abs( argv.x ) < 9.95 ? 0xf6000000 : 0xf4000000, 0, 5 );
        break;
    case 10:
        argv.x = tvec_target;
        params = vec4( g_textres.x * .5 + 80., 8, g_vehicle.tvec >= 2.5 ? 1. : 0., 12 );
        phrase = uvec4( tvec_light > 0. ? 0x564543f4 : 0x202020f4, 0, 0, 4 );
    	break;
    }
    N += 11;
}

// Function 35
vec4 PostFX(sampler2D tex, vec2 uv, float time)
{
  float radius = iResolution.x*1.4;
  float angle = sin(iTime);   //-1.+2.*
  vec2 center = vec2(iResolution.x*.8, iResolution.y)*1.5; 
    
  vec2 texSize = vec2(iResolution.x/.6,iResolution.y/.5);
  vec2 tc = uv * texSize;
  tc -= center;
  float dist = length(tc*sin(iTime/5.)); 
  if (dist < radius) 
  {
    float percent = (radius - dist) / radius;
    float theta = percent * percent * angle * 8.0;
    float s = sin(theta/2.);
    float c = cos(sin(theta/2.));
    tc = vec2(dot(tc, vec2(c, -s)), dot(tc, vec2(s, c)));
  }
  tc += center;
  vec3 color  = texture(iChannel1,(tc / texSize)).rgb;  
  vec3 color2 = texture(iChannel0,(tc / texSize)).rgb; 
  vec3 colmix = mix(color,color2,sin(time*.5)); 
  return vec4(colmix, 1.0);
}

// Function 36
vec3 postProcess(vec3 hdr)
{
    vec3 ldr = hdr * EXPOSURE;
    ldr = ldr / (vec3(1.0) + ldr);
	vec3 gamma = pow(ldr, CONTRAST / vec3(2.2));
    return gamma;
}

// Function 37
void postProcess(in vec2 uv, inout vec3 color)
{
    #if APPLY_LUMINANCE	
    float luminance = getFragLuminance(color);
    luminance = saturate(luminance);
    vec3 resLuminance = vec3(length(color.r * luminance), 
			     length(color.g * luminance), 
			     length(color.b * luminance));
	
    float bloomIntensity = 1.0 / (1.0 - BLOOM_CUTOFF);
    color = resLuminance * bloomIntensity;
    #endif
	
    #if COLOR_GRADING_FILTER

    vec4 filteredFinal = COLOR_GRADING_G * COLOR_GRADING_WEIGHT * vec4(COLOR_GRADING_COLOR, 1.0);
    vec4 realFinal = vec4(color, 1.0) * COLOR_GRADING_REAL_WEIGHT;

    color = color * CONTRAST + 0.5 - CONTRAST * 0.5;
	
    #if   COLOR_GRADING_BLEND_MODE == ADD
    color = vec3(realFinal) + vec3(filteredFinal);
    #elif COLOR_GRADING_BLEND_MODE == SUBTRACT
    color = vec3(realFinal) - vec3(filteredFinal);
    #elif COLOR_GRADING_BLEND_MODE == MULTIPLY
    color = vec3(realFinal) * vec3(filteredFinal);
    #elif COLOR_GRADING_BLEND_MODE == DIVIDE
    color = vec3(realFinal) / vec3(filteredFinal);
    #endif
	
    #endif
	
    color = saturation(color);
    color = desaturation(color);

    #if SEPIA_FILTER
    float greyScale = GetFragLuminance(color);
    color = greyScale * SEPIA_COLOR * SEPIA_INTENSITY;
    #endif
	
    #if VIGNETTE_FILTER
    color *= vec3(VIGNETTE_COLOR) * saturate(1.0 - length(uv / VIGNETTE_ZOOM)) * VIGNETTE_EXPOSURE;
    #endif

    #if HDR && APPLY_TONEMAP
    tonemap(color);
    #endif
    
    #if TONEMAP_TYPE != FILMIC_TONEMAP_ALU
    #if HDR && APPLY_GAMMA_CORRECTION && TONEMAP_TYPE != FILMIC_HEJL2015
    color = pow(color, vec3(1.0 / GAMMA));
    #elif APPLY_GAMMA_CORRECTION
    color = pow(color, vec3(1.0 / GAMMA));
    #endif
    #endif
}

// Function 38
vec3 Posterize(vec3 color)
{
	color = pow(color, vec3(GAMMA, GAMMA, GAMMA));
	color = floor(color * REGIONS)/REGIONS;
	color = pow(color, vec3(1.0/GAMMA));
	return color.rgb;
}

// Function 39
void UI_ProcessWindowImageB( inout UIContext uiContext, inout UIData uiData, int iControlId, int iData )
{
    UIWindowDesc desc;
    
    desc.initialRect = Rect( vec2(32, 8), vec2(192, 192) );
    desc.bStartMinimized = false;
    desc.bStartClosed = false;
    desc.bOpenWindow = false;   
    desc.uControlFlags = WINDOW_CONTROL_FLAG_TITLE_BAR | WINDOW_CONTROL_FLAG_MINIMIZE_BOX | WINDOW_CONTROL_FLAG_RESIZE_WIDGET;
	desc.vMaxSize = vec2(100000.0);

    UIWindowState window = UI_ProcessWindowCommonBegin( uiContext, iControlId, iData, desc );
    
    // Controls...
    if ( UI_ShouldProcessWindow( window ) )
    {    
        UI_WriteCanvasUV( uiContext, iControlId );
    }

    UI_ProcessWindowCommonEnd( uiContext, window, iData );
}

// Function 40
void process_text_info_page( int i, inout int N,
                          	 inout vec4 params, inout uvec4 phrase, inout vec4 argv,
                             int pageno, GameState gs )
{
#define INFO1( a, b, c, d, arg ) if( i == N++ ) { phrase = uvec4( (a), (b), (c), (d) ); argv.x = (arg); }
#define INFO2( a, b, c, d, arg ) if( i == N++ ) { phrase = uvec4( (a), (b), (c), (d) ); argv.xy = (arg); }
#define INFO3( a, b, c, d, arg ) if( i == N++ ) { phrase = uvec4( (a), (b), (c), (d) ); argv.xyz = (arg); } 
    float x = g_textres.x - 128.;
    float y = 64.;
    if( i == N++ )
        params = vec4( x, y, -1, 12 ), phrase = md_load( iChannel0, MENU_INFO_BEGIN + pageno );
	if( i < N )
		return;    
    y -= 20. + 12. * float( ( i - N ) & 3 );
    params = vec4( x, y, 1, 12 );
    if( pageno == GS_INFO_LOCATION )
    {
	    vec4 loc = navb( g_vehicle.localr, g_vehicle.localB[0] ) - vec4( 0, 0, g_data.radius, 0 );    
        INFO1( 0x6c617420, 0xfeb30000, 0, 6, loc.x );
        INFO1( 0x6c6f6e67, 0xfeb30000, 0, 6, loc.y );
        INFO1( 0x616c7420, abs( loc.z ) < 9999.99995 ? 0xfe206b6d : ( loc.z /= 1000., 0xfe204d6d ), 0, 8, loc.z );
        INFO1( 0x68646720, 0xfeb30000, 0, 6, loc.w );
    }
    else
	if( pageno == GS_INFO_WAYPOINT && gs.waypoint != ZERO )
	{
        vec2 arcdist = arcdistance( gs.waypoint, g_vehicle.localr );
        float eta = length( arcdist ) / length( g_vehicle.localv );
        INFO1( 0x62726720, 0xfeb30000, 0, 6, B2bearing( g_vehicle.localr, gs.waypoint - g_vehicle.localr ) );
        INFO1( 0x64737420, 0xfe206b6d, 0, 8, arcdist.x );
        INFO1( 0xb520fe20, 0x6b6d0000, 0, 6, arcdist.y );
        if( dot( g_vehicle.localv, g_vehicle.localv ) >= .25e-6 )
            if( eta < 8640000. )
            	{ INFO3( 0x65746120, 0x2020f33a, 0xf03af020, 12, fmt_time( int( floor( eta ) ) ) ) }
            else
                INFO1( 0x65746120, 0xfe206400, 0, 7, eta / 86400. );
	}
    else
	if( pageno == GS_INFO_ORBIT )
    {
        Kepler K = Kepler( 0., 0., 0., 0., 0. );
        float nu = kepler_init( K, g_vehicle.orbitr, g_vehicle.orbitv, g_data.GM );
        float ap = K.p / ( 1. - K.e ) - g_data.radius;
        float pe = K.p / ( 1. + K.e ) - g_data.radius;
        if( K.e < 0.99995 )
            INFO1( 0x41702020, abs( ap ) < 10000. ? 0xfe206b6d : ( ap /= 1000., 0xfe204d6d ), 0, 8, ap );
        INFO1( 0x50652020, abs( pe ) < 10000. ? 0xfe206b6d : ( pe /= 1000., 0xfe204d6d ), 0, 8, pe );
        INFO1( 0x65202020, 0xfe000000, 0, 5, K.e );
        if( K.e >= .00005 )
            INFO1( 0xb12020fe, 0xb3000000, 0, 5, degrees( nu ) );
    }
    else
	if( pageno == GS_INFO_GLIDE )
    {
        INFO1( 0x434c2020, 0xfe000000, 0, 5, g_vehicle.info.x );
        INFO1( 0x43442020, 0xfe000000, 0, 5, g_vehicle.info.y );
        INFO1( 0x4c2f4420, 0xfe000000, 0, 5, safediv( g_vehicle.info.x, g_vehicle.info.y ) );
		INFO1( 0xb02020fe, 0xb3000000, 0, 5, degrees( g_vehicle.info.z ) );
    }
    else
	if( pageno == GS_INFO_CONTROLS )
    {
        INFO1( 0x656c6576, 0xfe000000, 0, 5, g_vehicle.EAR.x * 100. );
        INFO1( 0x61696c20, 0xfe000000, 0, 5, g_vehicle.EAR.y * 100. );
        INFO1( 0x72756464, 0xfe000000, 0, 5, g_vehicle.EAR.z * 100. );
        INFO1( 0x7472696d, 0xfe000000, 0, 5, g_vehicle.trim * 100. );
    }
    else
	if( pageno == GS_INFO_AIR )
    {
        INFO1( 0x54202020, 0xfeb34300, 0, 7, g_env.atm.x - 273.15 );
        INFO1( 0x50202020, 0xfe206261, 0x72000000, 9, g_env.atm.y );
        INFO1( 0x51202020, 0xfe206261, 0x72000000, 9, 
			.5 * ( 1e6 / 1e5 ) * g_env.atm.z * dot( g_vehicle.localv, g_vehicle.localv ) );
        INFO1( 0xb22020fe, 0x206b672f, 0x6db40000, 10, g_env.atm.z );
    }
    else
    if( pageno == GS_INFO_TIME )
    {
        float tzone = round( navb( g_vehicle.localr, g_vehicle.localB[0] ).y / 15. );
        bool dots = fract( g_game.datetime.x * 1440. * SECONDS_PER_MINUTE ) < .5;
        INFO2( 0x64617465, 0x2020f22d, 0xf0000000, 9, g_game.datetime.zy + 1. );
        INFO2( 0x74696d65, 0x20202020, dots ? 0xf03af000 : 0xf020f000, 11, 
			fmt_time( int( mod( 86400. * g_game.datetime.x, 86400. ) ) ).xy );
        INFO3( 0x6c6f6361, 0x6c202020, dots ? 0xf03af020 : 0xf020f020, 
            ( tzone == 0. ? 11 : tzone < 0. ? 0x202df00f : 0x202bf00f ),
            vec3( fmt_time( int( mod( 86400. * g_game.datetime.x + 3600. * tzone, 86400. ) ) ).xy, abs( tzone ) ) );
    }
#undef INFO1
#undef INFO2
#undef INFO3
}

// Function 41
float posTri(float x)
{
    // thanks Shane for the anti-branching-fix
    return abs(fract(x - .5) - .5)*2.;
    //x=fract(x);
   	//return 2.*(x<.5?x:1.-x);
}

// Function 42
vec3 PostProcessColour(vec3 Colour, vec2 uv)
{
    // Vignette (Darken the pixels nearer the corners of the screen)
    Colour -= vec3(length(uv*0.1));
    
    // Add some random noise to the colour with the Hashing function
	Colour += Hash_From2D(uv*iTime*0.01)*0.02;
    
    // apply the CRT-screen filter
#ifdef CRT_FILTER_ON
    Colour = CRT_Filter(Colour, uv);
#endif 
    
    // Approximate the brightness of the colour by using it as a 3d spacial vector and getting its length in colour space
    float Brightness = length(Colour);
    
    // inrease the colour contrast, by dimming the darker colours and brightening the lighter ones, 
    // via linear interpolation of the colour and its approximated brightness value
	Colour = mix(Colour, vec3(Brightness), Brightness - 0.5);
    
    return Colour;
}

// Function 43
bool Processing(float var, float pos)
{
    return (pos > var * varWdth) && (pos < (var+1.) * varWdth);
}

// Function 44
vec3 PostProcessSnow(vec2 uv, in vec3 rayOrigin, in vec3 rayDirection, inout float depth)
{
    float aspectRatio = iResolution.y / iResolution.x;
    uv.y *= aspectRatio; 
    
    
    // Close Snowflakes
    {
        vec2 closeSnowUV = uv;
        
        // Offsetting by the rotation gives a good enough
        // illusion of 3D snow
        closeSnowUV.x += -GetRotationFactor() * 3.0;
        closeSnowUV.y += iTime / 4.0;
        closeSnowUV = fract(closeSnowUV);

        // This is super lame but I'm tired
        // and it's good enough...
        #define NUM_SNOWFLAKES 10
        vec3 Snowflakes[NUM_SNOWFLAKES];
        Snowflakes[0] = vec3(0.1, 0.7, 100.0);
        Snowflakes[1] = vec3(0.3, 0.3, 200.0);
        Snowflakes[2] = vec3(0.5, 0.5, 150.0);
        Snowflakes[3] = vec3(0.2, 0.73, 50.0);
        Snowflakes[4] = vec3(0.54, 0.94, 88.0);
        Snowflakes[5] = vec3(0.99, 0.34, 295.0);
        Snowflakes[6] = vec3(0.07, 0.28, 196.0);
        Snowflakes[7] = vec3(0.11, 0.32, 161.0);
        Snowflakes[8] = vec3(0.88, 0.9, 254.0);
        Snowflakes[9] = vec3(0.63, 0.01, 17.0);
            
        for(int i = 0; i < NUM_SNOWFLAKES; i++)
        {
            float uvDist = length(Snowflakes[i].xy - closeSnowUV);
            float snowDepth = Snowflakes[i].z;
            if(snowDepth < depth)
            {
                float radius = 0.008 * (1.0 - snowDepth / 300.0);
                if(uvDist < radius)
                {
                    vec3 diffuse = vec3(0.5) * (1.0 - (uvDist / radius));
                    vec3 specular = vec3(0.0);
                    FogPass(rayOrigin, rayDirection, snowDepth, diffuse, specular);
                    return diffuse;
                }
            }
        }
    }
    
    // Distance Snowflakes
    {
        // Offsetting by the rotation gives a good enough
        // illusion of 3D snow
        uv.x += -GetRotationFactor() * 2.0;

        uv.y += iTime / 10.0;
        vec4 noiseValue = texture(iChannel3, uv);
        float snowValue = noiseValue.r;
        float snowDepth = 300.0;//noiseValue.r * SCENE_MAX_T;
        if( (snowDepth < depth && snowValue > 0.95) )
        {
            vec3 diffuse = vec3(0.5);
            vec3 specular = vec3(0.0);
            FogPass(rayOrigin, rayDirection, snowDepth, diffuse, specular);
            return diffuse;
        }

    }
    return vec3(0.0);
}

// Function 45
float processLum( vec2 uv, float lum, vec2 offset )
{
    float angle = 1.;
    vec2 center = vec2(.5,.5);
    float scale = 350.;

    float dots = checker(uv, scale, angle, offset);
    float intens = 5.;
    float intens2 = ( intens - 1.) * .5;
    return lum * intens - intens2 * ( 1. - dots);
}

// Function 46
bool Processing(float c, float d
){//float a=d-c*varWdth
 ;//float b=varWdth//(c+1.)*varWdth-c*varWdth
 ;return  (abs(d-c*varWdth-.5*varWdth)*2.<varWdth)//(0.<a&&a<b)
 ;}

// Function 47
vec3 post(vec3 rgb, vec2 xy)
{
	// Gamma first...
	rgb = pow(rgb, vec3(0.45));

	// Then...
	#define CONTRAST 1.9
	#define SATURATION 1.8
	#define BRIGHTNESS 1.04
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);
	// Noise...
	// rgb = clamp(rgb+Hash(xy*iTime)*.1, 0.0, 1.0);
	// Vignette...
	rgb *= .5+0.5*pow(20.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.2 );	

	return rgb;
}

// Function 48
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	rgb = pow(rgb, vec3(0.45));

	// Then...
	#define CONTRAST 1.4
	#define SATURATION 1.4
	#define BRIGHTNESS 1.3
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);

	// Vignette...
	rgb *= .4+0.6*pow(180.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.35);	

	return clamp(rgb, 0.0, 1.0);
}

// Function 49
void Process( out vec4 fragColor, vec2 vUV, vec2 vScreen, int image )
{
    vec3 vResult = SampleImage2( vUV, vScreen, image );
    
    //vResult = texelFetch( iChannel0, ivec2( fragCoord.xy ), 0 ).rgb;
    
    float fShade = GetVignetting( vUV, 0.7, 2.0, 0.7 );
    
    vResult *= fShade;
    
    //if ( vUV.x > sin(iTime)*0.5+0.5 )
    {
    	vResult = ColorGrade( vResult );
    }
    
	vResult = ApplyGrain( vUV, vResult, 0.1 );             
        
    vResult = vResult * 2.0;
    vResult = Tonemap( vResult );
    fragColor.rgb = vResult;
    fragColor.a = 1.0;    
}

// Function 50
void post_primaries( inout vec3 col )
{
#if WITH_IMG_PRIMARIES
    col = max( ZERO, IMG_PRIMARIES * col );
#endif
}

// Function 51
vec3 posterize(vec3 color, float steps)
{
    return floor(color * steps) / steps;
}

// Function 52
vec4 ProcessScene(in vec2 fragCoord, in vec4 rand)
{
	Ray ray = ComputeRay(vec3(0), camRot, 6.0, 1.0, fragCoord);
    
    RayTraceSceneResult scene = RayTraceScene(ray, false);
    
    vec3 volumetric = vec3(0.);
    int iRand = 0;
    for (int s = 0; s < volSliceCount; ++s)
    {
        float t = float(s)/float(volSliceCount);
        float ct = GetLookupDepth(t) + GetRand(rand, s) * .01;
        float at = volDepthRange * ct;
        if (at > scene.hit.t)
        {
            break;
        }
        vec3 cPos = ray.org + at*ray.dir;
        float lightScattering = GetScatteringFactor(ray.dir, GetLightDirection(cPos));
        float decreaseFactor = GetIntensityFactor(ct);
        volumetric += decreaseFactor * lightScattering * GetLightIntensity(cPos);
    }
    
    vec3 lightIntensity = GetLightIntensity(scene.hit.pos);
    float nl = clamp(dot(scene.hit.nn, GetLightDirection(scene.hit.pos)),.0,1.);
    return vec4(scene.color*lightIntensity*nl + scene.emissive + volumetric,1.0);
}

// Function 53
bool UI_ProcessWindowCloseBox( inout UIContext uiContext, inout UIWindowState window, int iControlId, Rect closeBoxRect )
{
    bool bPressed = UI_ProcessButton( uiContext, iControlId, closeBoxRect );
    
    if ( bPressed )
    {
 		window.bClosed = true;
    }

    bool bActive = (uiContext.iActiveControl == iControlId);
    
    return UI_DrawWindowCloseBox( uiContext, closeBoxRect );
}

// Function 54
vec2 PosToSphere(vec3 pos)
{
  float x = atan(pos.z, pos.x); 
  float y = acos(pos.y / length(pos)); 
  return vec2( x / (2.0 * PI), y / PI);
}

// Function 55
void post_sun_glare( inout vec3 col, vec3 raydir )
{
#if WITH_IMG_SUNGLARE
    vec3 sunshadow = IMG_MIPMAP_HIDE * texelFetch( iChannel1, ivec2( 0, 0 ), 0 ).xyz;
    mat3 frame = g_game.camframe;
    if( g_vrmode )
        frame *= g_vrframe;
    float d = dot( normalize( reject( cross( g_env.L, raydir ), frame[0] ) ),
                   normalize( reject( cross( g_env.L, frame[2] ), frame[0] ) ) );
    float s = 1. / ( 1. + 0.985 * chebychev6(d) );
    float a = square( g_planet_data[0].radius ) / dot( g_vehicle.r, g_vehicle.r );
    float b = 1. - square( .81 / g_game.camzoom );
    float c = 1. - square( .71 / g_game.camzoom );
    float offimage = parabolstep( b, c, dot( frame[0], g_env.L ) );
    vec3 tmp = offimage * g_env.sunlight *
        exp2( -24. * sqrt( max( 0., 1. - square( dot( raydir, g_env.L ) ) ) ) ) *
        a * s * sunshadow / max( vec3( a * s / IMG_EXPOSURE_MAX ), 1. - a * sunshadow - dot( raydir, g_env.L ) );
    col += tmp;
#endif
}

// Function 56
vec3 PostFilmic_IlfordFp4Push(vec3 c, vec2 uv)
{
   // Ilford measured coefficients
   const vec3 cb = vec3( 0.0307479,  0.00030400, -0.04458630);
   const vec3 de = vec3(-0.0095000, -0.00162400, -0.01736670);
   const vec3 df = vec3( 0.1493590,  0.21412400,  1.85780000);
   
   	// Quick approximation of overall response curve in linear space
    // I'm factoring this power shape out of the channel response curves
    // because it improves the curve fit
   c = c * c; 
   
   // evaluate color channels
   vec3 ax = vec3(2.36691,5.14272,0.49020)*c;
   vec3 pn = (c*(ax+cb)+de);
   vec3 pd = (c*(ax+vec3(0.022,0.004,-0.10543))+df);
   
   // collapse color channels
   float  pr = dot(clamp(pn/pd,0.0,1.0),vec3(.5));
 
   // vignette
   float pv = pow(1.0 - dot(uv-.5, uv-.5), -1.758) + -.13;
   return vec3(mix(pr,pr*pr,pv*pr));   // done
}

// Function 57
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

// Function 58
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

// Function 59
vec4 process_text( int index,
                   int offs,
                   FrameContext fr,
                   GameState gs,
                   PlanetState ps )
{
    int N = 0, i = index;
	vec4 params = vec4(0), argv = vec4(0);
    uvec4 phrase = uvec4(0);

    process_text_message_line( i, N, params, phrase, argv );

	if( gs.stage == GS_SELECT_LOCATION && gs.timer >= 3.5 )
    	process_text_select_location( i, N, params, phrase, gs );

    if( gs.stage == GS_RUNNING )
    {
        if( gs.menustate.x > 0 )
            process_text_command_menu( i, N, params, phrase, gs );

        if( ( gs.switches & GS_TRMAP ) == 0u )
        {
            if( g_vehicle.modes.x > VS_HMD_OFF )
            {
            	process_text_hud_numbers( i, N, params, phrase, argv, gs, ps );

                if( g_vehicle.modes.x >= VS_HMD_ORB )
                	process_text_conj_gradients( i, N, params, phrase );
            }

           	process_text_time_accel( i, N, params, phrase, argv, fr );
		    process_text_console( i, N, params, phrase, argv );

			int infopage = int( gs.switches & GS_IPAGE_MASK ) >> GS_IPAGE_SHIFT;
    		if( infopage > 0 && infopage < MENU_INFO_SIZE )
                process_text_info_page( i, N, params, phrase, argv, infopage, gs );
        }
        else
        	process_text_map_markers( i, N, params, phrase, argv, gs );
   	}

    /*
    // debug numbers
	{
        vec4 debug = vec4(0);

        // AtmContext atm = atm_load( iChannel0, ADDR_ATMCONTEXTS + ivec2(1,0) );
        // debug = vec4( atm.r0, atm.htop, atm.r0 + atm.htop, length( gs.campos ) - g_data.radius );
        // debug = vec4( log( g_game.exposure ) / LN10, 0, 0 );
        // debug = g_vehicle.info;
        // debug.xyz = g_vehicle.acc * 1000. / FDM_STD_G;
        // debug.xyz = g_vehicle.accz * 1000. / FDM_STD_G;        
		// debug = g_vehicle.aerostuff;
        // debug.xyz = g_vehicle.rcsstuff;
        // debug = vec4( g_env.phases, g_env.atm2 );
        // debug.xyz = log( g_env.atm.xyz ) / LN10;
        // debug.xyz = g_game.vjoy;
        // debug.xyz = vec3( 1000. * iTimeDelta, 1. / iTimeDelta, iFrameRate ); 

        float x = g_textres.x - 240.;
		float y = 64. - 20. - 12. * float( ( i - N ) & 3 );
        if( i >= N && i < N + 4 )
        {
        	params = vec4( x, y, 1, 12 );
			switch( i - N )
    		{
    		case 0: phrase = uvec4( 0xfe000000, 0, 0, 1 ); argv.x = debug.x; break;
    		case 1: phrase = uvec4( 0xfe000000, 0, 0, 1 ); argv.x = debug.y; break;
    		case 2: phrase = uvec4( 0xfe000000, 0, 0, 1 ); argv.x = debug.z; break;
    		case 3: phrase = uvec4( 0xfe000000, 0, 0, 1 ); argv.x = debug.w; break;
    		}
        }
    	N += 4;
    }
	//*/

    return text_format( offs, params, phrase, argv );
}

// Function 60
void UI_ProcessScrollbarY( inout UIContext uiContext, int iControlId, inout UIData_Value data, Rect sliderRect, float fHandleSize )
{    
    bool bMouseOver = Inside( uiContext.vMouseCanvasPos, sliderRect ) && uiContext.bMouseInView;
    
    if ( uiContext.iActiveControl == IDC_NONE )
    {
        if ( uiContext.bMouseDown && (!uiContext.bMouseWasDown) && bMouseOver && !uiContext.bHandledClick )
        {
            uiContext.iActiveControl = iControlId;
            uiContext.bHandledClick = true;
        }
    }
    else
    if ( uiContext.iActiveControl == iControlId )
    {
        float fSlidePosMin = sliderRect.vPos.y + fHandleSize * 0.5f;
        float fSlidePosMax = sliderRect.vPos.y + sliderRect.vSize.y - fHandleSize * 0.5f;
        float fPosition = (uiContext.vMouseCanvasPos.y - fSlidePosMin) / (fSlidePosMax - fSlidePosMin);
        fPosition = clamp( fPosition, 0.0f, 1.0f );
        data.fValue = data.fRangeMin + fPosition * (data.fRangeMax - data.fRangeMin);
        if ( !uiContext.bMouseDown )
        {
            uiContext.iActiveControl = IDC_NONE;
        }
    }
        
    bool bActive = (uiContext.iActiveControl == iControlId);
    float fPosition = (data.fValue - data.fRangeMin) / (data.fRangeMax - data.fRangeMin);
    
    UI_DrawSliderY( uiContext, bActive, bMouseOver, fPosition, sliderRect, fHandleSize, true );    
}

// Function 61
void post_saturate( inout vec3 col )
{
#if WITH_IMG_SOFT_SATURATE == 1
    col = 1. - exp( -col * 1.1025 );
#elif WITH_IMG_SOFT_SATURATE == 2
    col = 2. / ( 1. + exp( -col * 2.0220 ) ) - 1.;
#elif WITH_IMG_SOFT_SATURATE == 3
    col = sin( min( col * 1.0055, PI / 2. ) );
#else
    col = saturate( col );
#endif
}

// Function 62
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	rgb = pow(rgb, vec3(0.45));
	
	// Then...
	#define CONTRAST 1.1
	#define SATURATION 1.3
	#define BRIGHTNESS 1.3
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);
	// Vignette...
	rgb *= .4+0.5*pow(40.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.2 );	
	return rgb;
}

// Function 63
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	

	// Then...
	#define CONTRAST 1.08
	#define SATURATION 1.5
	#define BRIGHTNESS 1.5
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);
	// Noise...
	//rgb = clamp(rgb+Hash(xy*iTime)*.1, 0.0, 1.0);
	// Vignette...
	rgb *= .5 + 0.5*pow(20.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.2);	

    rgb = pow(rgb, vec3(0.47 ));
	return rgb;
}

// Function 64
void UI_ProcessScrollbarPanelBegin( inout UIContext uiContext, inout UIPanelState scrollbarState, int iControlId, int iData, Rect scrollbarPanelRect, vec2 vScrollbarCanvasSize )
{
    float styleSize = UIStyle_ScrollBarSize();
    
	bool bScrollbarHorizontal = (scrollbarPanelRect.vSize.x < vScrollbarCanvasSize.x);
    if ( bScrollbarHorizontal )
    {        
        scrollbarPanelRect.vSize.y -= styleSize;
    }

    bool bScrollbarVertical = (scrollbarPanelRect.vSize.y < vScrollbarCanvasSize.y);
    if ( bScrollbarVertical )
    {
        scrollbarPanelRect.vSize.x -= styleSize;
    }

    // Adding a vertical scrollbar may mean we now need a horizontal one
    if ( !bScrollbarHorizontal )
    {
        bScrollbarHorizontal = (scrollbarPanelRect.vSize.x < vScrollbarCanvasSize.x);
        if ( bScrollbarHorizontal )
        {        
            scrollbarPanelRect.vSize.y -= styleSize;
        }
    }
    
    // Todo : Force enable or disable ?

	vec4 vData0 = LoadVec4( iChannelUI, ivec2(iData,0) );   
        
    UIData_Value scrollValueX;
    scrollValueX.fRangeMin = 0.0;
    scrollValueX.fRangeMax = max(0.0, vScrollbarCanvasSize.x - scrollbarPanelRect.vSize.x);
        
    UIData_Value scrollValueY;
    scrollValueY.fRangeMin = 0.0;
    scrollValueY.fRangeMax = max(0.0, vScrollbarCanvasSize.y - scrollbarPanelRect.vSize.y);
    
    if ( iFrame == 0 || vData0.z != DIRTY_DATA_MAGIC )
    {
        scrollValueX.fValue = 0.0;
        scrollValueY.fValue = 0.0;
    }
    else
    {
        scrollValueX.fValue = vData0.x;
        scrollValueY.fValue = vData0.y;
    }    
    
    scrollValueX.fValue = clamp( scrollValueX.fValue, scrollValueX.fRangeMin, scrollValueX.fRangeMax );
    scrollValueY.fValue = clamp( scrollValueY.fValue, scrollValueY.fRangeMin, scrollValueY.fRangeMax );
    
    if ( bScrollbarHorizontal )
    {
        Rect scrollbarRect;
        scrollbarRect.vPos = scrollbarPanelRect.vPos;
        scrollbarRect.vPos.y += scrollbarPanelRect.vSize.y;
        scrollbarRect.vSize.x = scrollbarPanelRect.vSize.x;
        scrollbarRect.vSize.y = styleSize;
        
        float fHandleSize = scrollbarRect.vSize.x * (scrollbarPanelRect.vSize.x / vScrollbarCanvasSize.x);

        if ( uiContext.bPixelInView ) 
        {
	        DrawRect( uiContext.vPixelCanvasPos, scrollbarRect, vec4(0.6, 0.6, 0.6, 1.0), uiContext.vWindowOutColor );
        }        
        UI_ProcessScrollbarX( uiContext, iControlId, scrollValueX, scrollbarRect, fHandleSize );
    }
        
    if ( bScrollbarVertical )
    {        
        Rect scrollbarRect;
        scrollbarRect.vPos = scrollbarPanelRect.vPos;
        scrollbarRect.vPos.x += scrollbarPanelRect.vSize.x;
        scrollbarRect.vSize.x = styleSize;
        scrollbarRect.vSize.y = scrollbarPanelRect.vSize.y;
        
        float fHandleSize = scrollbarRect.vSize.y * (scrollbarPanelRect.vSize.y / vScrollbarCanvasSize.y);
        
        if ( uiContext.bPixelInView ) 
        {
	        DrawRect( uiContext.vPixelCanvasPos, scrollbarRect, vec4(0.6, 0.6, 0.6, 1.0), uiContext.vWindowOutColor );
        }
        
        UI_ProcessScrollbarY( uiContext, iControlId + 1000, scrollValueY, scrollbarRect, fHandleSize );
    }
    
    if ( bScrollbarHorizontal && bScrollbarVertical ) 
    {
        Rect cornerRect;
        cornerRect.vPos = scrollbarPanelRect.vPos;
        cornerRect.vPos += scrollbarPanelRect.vSize;
        cornerRect.vSize = vec2(styleSize);
        
        if ( uiContext.bPixelInView ) 
        {
            DrawRect( uiContext.vPixelCanvasPos, cornerRect, vec4(cScrollPanelCorner, 1.0), uiContext.vWindowOutColor );
#ifdef NEW_THEME  
        	DrawBorderRect( uiContext.vPixelCanvasPos, cornerRect, cScrollPanelCornerOutline, uiContext.vWindowOutColor );
#else            
        	DrawBorderIndent( uiContext.vPixelCanvasPos, cornerRect, uiContext.vWindowOutColor );
#endif            
        }
    }

    UI_PanelBegin( uiContext, scrollbarState );    
    
    vData0.x = scrollValueX.fValue;
    vData0.y = scrollValueY.fValue;
    vData0.z = DIRTY_DATA_MAGIC;
    StoreVec4( ivec2(iData,0), vData0, uiContext.vOutData, ivec2(uiContext.vFragCoord) );    
        
            
    UIDrawContext scrollbarPanelContextDesc = UIDrawContext_SetupFromRect( scrollbarPanelRect );
    scrollbarPanelContextDesc.vCanvasSize = vScrollbarCanvasSize;
    scrollbarPanelContextDesc.vOffset = vec2(scrollValueX.fValue, scrollValueY.fValue);

    UIDrawContext scrollbarPanelContext = UIDrawContext_TransformChild( scrollbarState.parentDrawContext, scrollbarPanelContextDesc );
    UI_SetDrawContext( uiContext, scrollbarPanelContext );
}

// Function 65
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	rgb = pow(rgb, vec3(0.45));

	// Then...
	#define CONTRAST 1.3
	#define SATURATION 1.3
	#define BRIGHTNESS 1.2
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);

	// Vignette...
	rgb *= .5+0.5*pow(180.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.3 );	

	return clamp(rgb, 0.0, 1.0);
}

// Function 66
bool UI_ProcessWindowTitleBar( inout UIContext uiContext, inout UIWindowState window )
{
    int iWindowTitleBarControlId = window.iControlId;
    int iWindowMinimizeControlId = window.iControlId + 1000;
    int iWindowCloseControlId = window.iControlId + 3000;
    Rect titleBarRect = Rect( vec2(0.0), UI_WindowGetTitleBarSize( uiContext, window ) );
    
    bool bRenderedWidget = false;
    if ( FLAG_SET(window.uControlFlags, WINDOW_CONTROL_FLAG_MINIMIZE_BOX) )
    {
        Rect minimizeBoxRect = Rect( vec2(0.0), vec2(titleBarRect.vSize.y) );
        RectShrink( minimizeBoxRect, vec2(4.0) );
        
    	bRenderedWidget = UI_ProcessWindowMinimizeWidget( uiContext, window, iWindowMinimizeControlId, minimizeBoxRect );
    }

    if ( FLAG_SET(window.uControlFlags, WINDOW_CONTROL_FLAG_CLOSE_BOX) )
    {
        Rect closeBoxRect = Rect( vec2(0.0), vec2(titleBarRect.vSize.y) ); 
        closeBoxRect.vPos.x = titleBarRect.vSize.x - closeBoxRect.vSize.x;
        RectShrink( closeBoxRect, vec2(4.0) );
        
        if( UI_ProcessWindowCloseBox( uiContext, window, iWindowCloseControlId, closeBoxRect ) )
        {
            bRenderedWidget = true;
        }
    }
            
    bool bMouseOver = Inside( uiContext.vMouseCanvasPos, titleBarRect ) && uiContext.bMouseInView;
        
    if ( uiContext.iActiveControl == IDC_NONE )
    {
        if ( uiContext.bMouseDown && (!uiContext.bMouseWasDown) && bMouseOver && !uiContext.bHandledClick )
        {
            uiContext.iActiveControl = iWindowTitleBarControlId;
            uiContext.vActivePos = window.rect.vPos - uiContext.vMousePos;
            uiContext.bHandledClick = true;
        }
    }
    else
    if ( uiContext.iActiveControl == iWindowTitleBarControlId )
    {
        if ( !uiContext.bMouseDown )
        {
            uiContext.iActiveControl = IDC_NONE;
        }
    }    
    
    bool bActive = (uiContext.iActiveControl == iWindowTitleBarControlId);
    
    if ( bActive )
    {
        window.rect.vPos = uiContext.vMousePos + uiContext.vActivePos;
    }   
    
    if (!bRenderedWidget)
    {
    	UI_DrawWindowTitleBar( uiContext, bActive, titleBarRect, window );
    }
    
    return Inside( uiContext.vPixelCanvasPos, titleBarRect );
}

// Function 67
vec3 postEffects( in vec3 col, in vec2 uv, in float time )
{
	// vigneting
	col *= 0.5+0.5*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.5 );
	return col;
}

// Function 68
float3 worldPosTocubePos(float3 worldPos)
{
    // cube of world space size 4 with bottom face on the ground y=0
    return worldPos*0.15 + float3(0.0,-0.5,0.0);
}

// Function 69
void ProcessLightValue(float time)
{
    time *= 1.;
    
    o_lights[0].pos = vec3(6.,1.,sin(time));
    o_lights[0].colorIntensity = 3.*vec4(0.2,1.,.2,1);
    o_lights[1].pos = vec3(-3,-2.2,sin(time*.3)*8.);
    o_lights[1].colorIntensity = 2.*vec4(1,1,0.5,1);
    o_lights[2].pos = vec3(9.5,1.8,9.5);
    o_lights[2].colorIntensity = 3.*max(0.,abs(sin(PI*time)))*vec4(1,0.2,1,1);
}

// Function 70
void UI_ProcessWindowResizeWidget( inout UIContext uiContext, inout UIWindowState window, int iControlId )
{
    vec2 vCorner = uiContext.drawContext.vCanvasSize;
    float fControlSize = 24.0;
    
    bool bMouseOver = ScreenPosInResizeWidget( uiContext, vCorner, fControlSize, uiContext.vMousePos )
        && uiContext.bMouseInView;
        
    if ( uiContext.iActiveControl == IDC_NONE )
    {
        if ( uiContext.bMouseDown && (!uiContext.bMouseWasDown) && bMouseOver && !uiContext.bHandledClick)
        {
            uiContext.iActiveControl = iControlId;
            
            uiContext.vActivePos = window.rect.vSize - uiContext.vMousePos;
            
            uiContext.bHandledClick = true;
        }
    }
    else
    if ( uiContext.iActiveControl == iControlId )
    {
        if ( !uiContext.bMouseDown )
        {
            uiContext.iActiveControl = IDC_NONE;
        }
    }
        
    bool bActive = (uiContext.iActiveControl == iControlId);        
    
    if ( bActive )
    {
        window.rect.vSize = uiContext.vMousePos + uiContext.vActivePos;
        vec2 vMinWindowSize = vec2( 96.0, 64.0 );
        window.rect.vSize = max( vMinWindowSize, window.rect.vSize );
        window.rect.vSize = min( window.vMaxSize, window.rect.vSize );
    }
    
    
    if ( uiContext.bPixelInView &&
        ScreenPosInResizeWidget( uiContext, vCorner, fControlSize, uiContext.vPixelPos ) )
    {
        vec4 vColor = vec4(cResize, 1.0);
        
        if( bActive )
        {
            vColor = vec4(cResizeActive, 1.0);
        }
        uiContext.vWindowOutColor = vColor;
    }    
}

// Function 71
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

// Function 72
void post_balance( inout vec3 col )
{
#if WITH_IMG_BALANCE
    col *= IMG_BALANCE_ADAPT / hmax( IMG_PRIMARIES * IMG_BALANCE_ADAPT );
#endif
}

// Function 73
vec4 posterize(vec4 color, float numColors)
{
    return floor(color * numColors - 0.5) / numColors;
}

// Function 74
bool UI_ProcessButton( inout UIContext uiContext, int iControlId, Rect buttonRect )
{    
    bool bMouseOver = Inside( uiContext.vMouseCanvasPos, buttonRect ) && uiContext.bMouseInView;
    
    bool bButtonPressed = false;
    
    if ( uiContext.iActiveControl == IDC_NONE )
    {
        if ( uiContext.bMouseDown && (!uiContext.bMouseWasDown) && bMouseOver && !uiContext.bHandledClick )
        {
            uiContext.iActiveControl = iControlId;
            uiContext.bHandledClick = true;
        }
    }
    else
    if ( uiContext.iActiveControl == iControlId )
    {
        if ( !uiContext.bMouseDown )
        {
            uiContext.iActiveControl = IDC_NONE;
            if ( bMouseOver )
            {
                bButtonPressed = true;
            }
        }
    }

    bool bActive = (uiContext.iActiveControl == iControlId);
    
    UI_DrawButton( uiContext, bActive, bMouseOver, buttonRect );    
        
    return bButtonPressed;
}

// Function 75
vec3 postEffects( in vec3 col, in vec2 uv, in float time )
{
	// vigneting
	col += 1.-(0.9+0.1*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.6 ));
	col *= 0.5+0.5*pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y), 0.1 );
	return col;
}

// Function 76
void UI_ProcessWindowImageA( inout UIContext uiContext, inout UIData uiData, int iControlId, int iData )
{
    UIWindowDesc desc;
    
    vec2 vWindowIdealSize = UI_GetWindowSizeForContent( vec2(512, 512) );
    desc.initialRect = Rect( vec2(96, 48 - 32), vec2( vWindowIdealSize.x, 350 ) );
    desc.bStartMinimized = false;
    desc.bStartClosed = false;
    desc.bOpenWindow = false;      
    desc.uControlFlags = WINDOW_CONTROL_FLAG_TITLE_BAR | WINDOW_CONTROL_FLAG_MINIMIZE_BOX | WINDOW_CONTROL_FLAG_RESIZE_WIDGET;

    desc.vMaxSize = vWindowIdealSize;

    UIWindowState window = UI_ProcessWindowCommonBegin( uiContext, iControlId, iData, desc );
        
    // Controls...
    if ( UI_ShouldProcessWindow( window ) )
    {        
        Rect scrollbarPanelRect;

        #if 0
        // ScrollBar panel in fixed location
        scrollbarPanelRect = Rect( vec2(10, 32), vec2(256) );
        #else
        // ScrollBar panel with parent window size        	
        scrollbarPanelRect = Rect( vec2(0), uiContext.drawContext.vCanvasSize );
        #endif

        vec2 vScrollbarCanvasSize = vec2(512);

        UIPanelState scrollbarPanelState;            
        UI_ProcessScrollbarPanelBegin( uiContext, scrollbarPanelState, IDC_SCROLLBAR_PANEL, DATA_SCROLLBAR_PANEL, scrollbarPanelRect, vScrollbarCanvasSize );

        // Controls...
        {
            UI_ProcessWindowImageControl( uiContext, uiData, IDC_WINDOW_IMAGE_CONTROL, DATA_WINDOW_IMAGE_CONTROL );
            UI_ProcessWindowImageB( uiContext, uiData, IDC_WINDOW_IMAGEB, DATA_WINDOW_IMAGEB );

            UI_WriteCanvasPos( uiContext, iControlId );
        }

        UI_ProcessScrollbarPanelEnd(uiContext, scrollbarPanelState);
    }
    
    UI_ProcessWindowCommonEnd( uiContext, window, iData );
}

// Function 77
void process_text_conj_gradients( int i, inout int N,
                                  inout vec4 params, inout uvec4 phrase )
{
    vec3 r_ = g_vehicle.orbitr;
    vec3 v_ = g_vehicle.orbitv;
    float invGM = 1. / g_data.GM;

    float r2 = dot( r_, r_ );
    float v2 = dot( v_, v_ );
    float rv = dot( r_, v_ );
    vec3 h_ = cross( r_, v_ );
	float h2 = dot( h_, h_ );
    float h = sqrt( h2 );
    float r = sqrt( r2 );
    float epsilon = v2 * invGM - 1. / r;
    vec3 e_ = epsilon * r_ - rv * invGM * v_;
    float e2 = dot( e_, e_ );
    float e = sqrt( e2 );
    vec3 f_ = e_ + epsilon * r_;

	vec3[5] dirs = vec3[5](
        cross( v_, h_ ) * sign( 1. - e ),
        f_,
        2. * e * ( 1. + e ) * r_ - invGM * h2 * f_,
        2. * e * ( 1. - e ) * r_ + invGM * h2 * f_,
        h_
    );

    if( lensq( dirs[2] ) * 1e8 < r2 * e )
   	    dirs[2] = abs( rv ) / r2 * r_ + sign( rv ) * e * v_;

    if( lensq( dirs[3] ) * 1e8 < r2 * e )
        dirs[3] = abs( rv ) / r2 * r_ - sign( rv ) * e * v_;

    uvec4[5] phrases = uvec4[5](
        uvec4( 0x00650000, 0, 0, 2 ),		// e
        uvec4( 0x00610000, 0, 0, 2 ),		// a
        uvec4( 0x00417000, 0, 0, 3 ),		// Ap
        uvec4( 0x00506500, 0, 0, 3 ),		// Pe
		uvec4( 0x00680000, 0, 0, 2 ) 		// h
	);

    vec2 tsc = sincospi( g_vehicle.tvec / 180. );
    mat3 tvecrot = g_vehicle.B * 
        mat3( tsc.y, 0, tsc.x, 0, g_vehicle.tvec < 105. ? 1 : -1, 0, -tsc.x, 0, tsc.y ) * 
        transpose( g_vehicle.B );

    for( int j = 0; j < 5; ++j )
    if( lensq( dirs[j] ) * 1e12 >= r2 && i == N++ )
    {
        vec3 dir = tvecrot * normalize( dirs[j] );
        if( dot( dir, g_planet.B * g_game.camframe[0] ) < 0. )
			dir = -dir;
		bool plus = dot( dir, tvecrot * ( j >= 4 ? h_ : cross( h_, dirs[ j ^ 1 ] ) ) ) >= 0.;
        dir = round( 2047.5 * dir * g_planet.B * g_game.camframe + 2047.5 );
        params = vec4( dir.x + dir.y / 4096., dir.z, 1, -12 );
        phrase = phrases[j];
		phrase[0] |= plus ? 0x2b000000u : 0x2d000000u;
    }
}

// Function 78
bool Processingolderstill(float c, float d
){return  (d-c*varWdth>0.)
        &&(d-c*varWdth<(c+1.)*varWdth-c*varWdth);}

// Function 79
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

// Function 80
vec3 ApplyPostFX( const in vec2 vUV, const in vec3 vInput )
{
	vec3 vFinal = ApplyVignetting( vUV, vInput );	
	
	vFinal = Tonemap(vFinal * 1.5);
	    
	return vFinal;
}

// Function 81
void post_exposure( inout vec3 col )
{
#if WITH_IMG_EXPOSURE
  #if WITH_IMG_RODVISION
    float y = dot( col, VIS_SCOTOPIC_Y );
    float suppress = VIS_LIMITS.z / ( y + VIS_LIMITS.z );
    col = g_exposure.z * col + g_exposure.w * suppress * y * .25 * COL_RODVISION;
  #else
	col = g_exposure.z * col;
  #endif
#endif
}

// Function 82
void ProcessObjectPos(float time
){objPos[oCubeMy]=vec3(0) 
 ;objRot[oCubeMy]=aa2q(time*2.,vec3(0,1,0))//es100 error , no array of class allowed
 ;objSca[oCubeMy]=vec3(.8)//es100 error , no array of class allowed
 ;objPos[oBlackHole]=vec3(5.,sin(time*0.2),-5.)//es100 error , no array of class allowed
 ;objRot[oBlackHole]=aa2q(time*2.,vec3(0,1,0))//es100 error , no array of class allowed
 ;objSca[oBlackHole]=vec3(1)//es100 error , no array of class allowed
 ;objPos[oCubeChil]=vec3(1)//es100 error , no array of class allowed
 ;objRot[oCubeChil]=aa2q(time*1.,normalize(objPos[oCubeChil]))//es100 error , no array of class allowed
 ;//o_myCubeChildren.rot = vec4(0,0,0,1)
 ;objSca[oCubeChil]=vec3(.4)//es100 error , no array of class allowed
 ;float trainV = 2.2
 ;objVel[oTrain]= vec3((floor(mod(trainV*time/16.,2.))*2.-1.)*trainV,0,0)//es100 error , no array of class allowed
 ;float trainDir = 1.
 ;if (objVel[oTrain].x < 0.)trainDir = -1.//es100 error , no array of class allowed
 ;objPos[oTrain]=vec3(abs(1.-mod(trainV*time/16.,2.))*16.-8.,-.8,9.)//es100 error , no array of class allowed
 ;objRot[oTrain]=aa2q(pi*.5,vec3(0,1,0))//es100 error , no array of class allowed
 ;objSca[oTrain]= vec3(1.,1.,trainDir/mix(LorentzFactor(trainV*LgthContraction),1.,photonLatency))
 ;//objects[o_train].b.x = 0.//es100 error , no array of class allowed
 ;objPos[oTunnel]=vec3(0,-.8,9.)//es100 error , no array of class allowed
 ;objRot[oTunnel]=aa2q(pi*.5,vec3(0,1,0))//es100 error , no array of class allowed
 ;objSca[oTunnel]=vec3(1.,1.,1)//es100 error , no array of class allowed
 ;objPos[oTunnelDoor]=objPos[oTunnel]//es100 error , no array of class allowed
 ;objRot[oTunnelDoor]=objRot[oTunnel]//es100 error , no array of class allowed
 ;float open = clamp((1.-abs(3.*objPos[oTrain].x))*2.,0.,1.)//es100 error , no array of class allowed
 ;objSca[oTunnelDoor]= vec3(open,open,1);}

// Function 83
vec3 processImage(float time,vec2 fragCoord){
	vec2 uv = fragCoord.xy / iResolution.xy;
	vec2 p = -1. + 2. * uv;
	
	vec3 col = vec3(.32,.45,.25);
	float f =1. - length(p);
	col = mix(col,vec3(.58,.86,.46),f);
	
	p.x*=iResolution.x/iResolution.y; 
	
	f = 1.;
	f = - sin( time +f) * (1. - sin( time+f))*cos(time+f) ;
	
	p.x +=f*3.;
	
	f = sin( time) * (1. - sin( time))*cos(time) ;
	
	float a = f*0.8;
	vec2 p2 = vec2(0.,-1.);
	p -= p2;
	float x =  p.x *cos(a) - (p.y)* sin( a) ;
	float y = (p.y) *cos(a) + p.x *sin(a) ;
	p = vec2(x,y);
	p+=p2;
	
	
	col = mix(col,vec3(0.),rectangle(p,1.3,.5,.0,.0,.0));
	col = mix(col,vec3(0.),rectangle(p,.35,1.5,.0,-.5,.0));
	col = mix(col,vec3(1.),rectangle(p,.2,.2,.45,.0,.0));
	col = mix(col,vec3(1.),rectangle(p,.2,.2,-.45,.0,.0));
	col = mix(col,vec3(1.),rectangle(p,.2,.05,.0,-.15,.0));
	return col;
}

// Function 84
vec2 UIDrawContext_CanvasPosToScreenPos( UIDrawContext drawContext, vec2 vCanvasPos )
{
    return vCanvasPos - drawContext.vOffset + drawContext.viewport.vPos;
}

// Function 85
vec3 postProcess(vec3 col, vec2 q) {
  col = clamp(col, 0.0, 1.0);
  col = pow(col, vec3(1.0/2.2));
  col = col*0.6+0.4*col*col*(3.0-2.0*col);
  col = mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col *=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

// Function 86
void postProcess(in vec2 uv, inout vec3 color)
{
    #if APPLY_LUMINANCE	
    float luminance = GetFragLuminance(color);
    luminance = saturate(luminance);
    vec3 resLuminance = vec3(length(color.r * luminance), 
			     length(color.g * luminance), 
			     length(color.b * luminance));
	
    float bloomIntensity = 1.0 / (1.0 - BLOOM_CUTOFF);
    color = resLuminance * bloomIntensity;
    #endif
	
    #if COLOR_GRADING_FILTER

    vec4 filteredFinal = COLOR_GRADING_G * COLOR_GRADING_WEIGHT * vec4(COLOR_GRADING_COLOR, 1.0);
    vec4 realFinal = vec4(color, 1.0) * COLOR_GRADING_REAL_WEIGHT;

    color = color * CONTRAST + 0.5 - CONTRAST * 0.5;
	
    #if   COLOR_GRADING_BLEND_MODE == ADD
    color = vec3(realFinal) + vec3(filteredFinal);
    #elif COLOR_GRADING_BLEND_MODE == SUBTRACT
    color = vec3(realFinal) - vec3(filteredFinal);
    #elif COLOR_GRADING_BLEND_MODE == MULTIPLY
    color = vec3(realFinal) * vec3(filteredFinal);
    #elif COLOR_GRADING_BLEND_MODE == DIVIDE
    color = vec3(realFinal) / vec3(filteredFinal);
    #endif
	
    #endif
	
    color = saturation(color);
    color = desaturation(color);

    #if SEPIA_FILTER
    float greyScale = GetFragLuminance(color);
    color = greyScale * SEPIA_COLOR * SEPIA_INTENSITY;
    #endif
	
    #if VIGNETTE_FILTER
    color *= vec3(VIGNETTE_COLOR) * saturate(1.0 - length(uv / VIGNETTE_ZOOM)) * VIGNETTE_EXPOSURE;
    #endif

    #if HDR && APPLY_TONEMAP
    color = Tonemap(color);
    #else
    #endif
    
    #if HDR && APPLY_GAMMA_CORRECTION && TONEMAP_TYPE != FILMIC_HEJL2015
    color = pow(color, vec3(1.0 / GAMMA));
    #elif APPLY_GAMMA_CORRECTION
    color = pow(color, vec3(1.0 / GAMMA));
    #endif
}

// Function 87
vec4 processSliders(in vec2 fragCoord)
{
    vec4 sliderVal = texture(iChannel0,vec2(0,0));
	ROUGHNESS_AMOUNT        = sliderVal[1];
    SKY_COLOR               = sliderVal[2];
    ABL_LIGHT_CONTRIBUTION  = sliderVal[3];
    
    if(length(fragCoord.xy-vec2(0,0))>1.)
    {
    	return texture(iChannel0,fragCoord.xy/iResolution.xy);
    }
    return vec4(0);
}

// Function 88
vec3 postEffect(vec3 col, vec2 index)
{
    //float shift = rand(index * 82.345);
    float shift = (index.x + index.y * 3.0) / 9.0;
    vec3 hsv = rgb2hsv(col);
    hsv.x += shift + iTime * 0.1;
    hsv.y = 0.75;
    //hsv.y *= 2.5;
    //hsv.y = hsv.y < 0.25 ? 0.25 : hsv.y;
    //hsv.y = floor(hsv.y * 8.0) / 8.0;
    //hsv.z = 1.0;
    //hsv.z = hsv.z < 0.7 ? hsv.z < 0.5 ? 0.0 : 0.5 : hsv.z;
    hsv.z = floor(hsv.z * 8.0) / 8.0;
    hsv.z = hsv.z < 0.6 ? 0.0 : hsv.z;
    
    hsv.x += hsv.z * 3.0;
	col = hsv2rgb(hsv);
    //col = floor(col * 8.0) / 8.0;

	return col;
}

// Function 89
void post_console_overlay( inout vec3 col, vec2 coord )
{
    vec2 uv = mix( vec2( iResolution.x, 0 ), coord, g_textscale ) / iResolution.xy;
    coord = ( coord - g_overlayframe.xy ) * g_textscale;
    if( !g_vrmode )
    {
        float consolemask = uv.y * iResolution.y - 24.;
        if( ( g_game.switches & GS_IPAGE_MASK ) != 0u )
        {
            vec2 b = ( uv.xy * iResolution.xy - vec2( iResolution.x - 136., 80 ) ) * vec2( -1, 1 );
            float c = max( b.x + b.y, hmax( b ) );
            consolemask = clamp( consolemask, 0., 16. ) + clamp( c, 0., 16. ) - 16.;
        }
        col *= mix( .5, 1., saturate( consolemask / 16. ) );
    }
 	console_throttle_graphics( col, coord - vec2( 16, 12 ) );
}

// Function 90
void ProcessObjectPos(float time
){objPos[oCubeMy]=vec3(0) 
 ;objRot[oCubeMy]=aa2q(time*2.,vec3(0,1,0))//es100 error , no array of class allowed
 ;objSca[oCubeMy]=vec3(.8)//es100 error , no array of class allowed
 ;objPos[oBlackHole]=vec3(5.,sin(time*0.2),-5.)//es100 error , no array of class allowed
 ;objRot[oBlackHole]=aa2q(time*2.,vec3(0,1,0))//es100 error , no array of class allowed
 ;objSca[oBlackHole]=vec3(1)//es100 error , no array of class allowed
 ;objPos[oCubeChil]=vec3(1)//es100 error , no array of class allowed
 ;objRot[oCubeChil]=aa2q(time*1.,normalize(objPos[oCubeChil]))//es100 error , no array of class allowed
 ;//o_myCubeChildren.rot = vec4(0,0,0,1)
 ;objSca[oCubeChil]=vec3(.4)//es100 error , no array of class allowed
 ;float trainV = 2.2
 ;objVel[oTrain]= vec3((floor(mod(trainV*time/16.,2.))*2.-1.)*trainV,0,0)//es100 error , no array of class allowed
 ;float trainDir = 1.
 ;if (objVel[oTrain].x < 0.)trainDir = -1.//es100 error , no array of class allowed
 ;objPos[oTrain]=vec3(abs(1.-mod(trainV*time/16.,2.))*16.-8.,-.8,9.)//es100 error , no array of class allowed
 ;objRot[oTrain]=aa2q(pi*.5,vec3(0,1,0))//es100 error , no array of class allowed
 ;objSca[oTrain]= vec3(1.,1.,trainDir/mix(LorentzFactor(trainV*LgthContraction),1.,cLag))
 ;//objects[o_train].b.x = 0.//es100 error , no array of class allowed
 ;objPos[oTunnel]=vec3(0,-.8,9.)//es100 error , no array of class allowed
 ;objRot[oTunnel]=aa2q(pi*.5,vec3(0,1,0))//es100 error , no array of class allowed
 ;objSca[oTunnel]=vec3(1.,1.,1)//es100 error , no array of class allowed
 ;objPos[oTunnelDoor]=objPos[oTunnel]//es100 error , no array of class allowed
 ;objRot[oTunnelDoor]=objRot[oTunnel]//es100 error , no array of class allowed
 ;float open = clamp((1.-abs(3.*objPos[oTrain].x))*2.,0.,1.)//es100 error , no array of class allowed
 ;objSca[oTunnelDoor]= vec3(open,open,1);}

// Function 91
void UI_ProcessWindowEditColor( inout UIContext uiContext, inout UIData uiData, int iControlId, int iData )
{
    UIWindowDesc desc;
    
    desc.initialRect = Rect( vec2(256, 48), vec2(210, 260) );
    desc.bStartMinimized = false;
    desc.bStartClosed = false;
    desc.bOpenWindow = true;        
    desc.uControlFlags = WINDOW_CONTROL_FLAG_TITLE_BAR | WINDOW_CONTROL_FLAG_CLOSE_BOX;
    desc.vMaxSize = vec2(100000.0);

    UIWindowState window = UI_ProcessWindowCommonBegin( uiContext, iControlId, iData, desc );
    
    bool closeButtonPressed = false;
    
    // Controls...
    if ( UI_ShouldProcessWindow( window ) )
    {    
		UILayout uiLayout = UILayout_Reset();
        
        LayoutStyle style;
        RenderStyle renderStyle;             
        UIStyle_GetFontStyleWindowText( style, renderStyle );
        
        UIData_Color dataColor;
        
        if ( uiData.editWhichColor.fValue == 0.0 )
        {
            dataColor = uiData.bgColor;
        }
        else
        if ( uiData.editWhichColor.fValue == 1.0 )
        {
            dataColor = uiData.imgColor;
        }
        
		UILayout_StackControlRect( uiLayout, UIStyle_ColorPickerSize().xy );                
        UI_ProcessColorPickerSV( uiContext, IDC_COLOR_PICKER, dataColor, uiLayout.controlRect );
        UILayout_StackRight( uiLayout );
		UILayout_StackControlRect( uiLayout, UIStyle_ColorPickerSize().zy );        
        UI_ProcessColorPickerH( uiContext, IDC_COLOR_PICKER+1000, dataColor, uiLayout.controlRect );
        UILayout_StackDown( uiLayout );        
        
        {
            style.vSize *= 0.6;

            PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        

            vec3 vRGB = hsv2rgb(dataColor.vHSV);
            PrintRGB( state, style, vRGB );
                
            UI_RenderFont( uiContext, state, style, renderStyle );
                        
			UILayout_SetControlRectFromText( uiLayout, state, style );
	        UILayout_StackDown( uiLayout );            

            style.vSize /= 0.6;            
        }
        
        if ( uiData.editWhichColor.fValue == 0.0 )
        {
            uiData.bgColor = dataColor;
        }
        else
        if ( uiData.editWhichColor.fValue == 1.0 )
        {
            uiData.imgColor = dataColor;
        }
    
        {
            PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
            uint strA[] = uint[] ( _O, _k );
            ARRAY_PRINT(state, style, strA);
            UI_RenderFont( uiContext, state, style, renderStyle );
			UILayout_SetControlRectFromText( uiLayout, state, style );

            bool buttonPressed = UI_ProcessButton( uiContext, IDC_COLOR_PICKER + 2000, uiLayout.controlRect ); // Use text for button rect
            if ( buttonPressed )
            {
                window.bClosed = true;
            }
	        UILayout_StackDown( uiLayout );                  
        }        
    }
    
    UI_ProcessWindowCommonEnd( uiContext, window, iData );
    
    if ( window.bClosed )
    {
        uiData.editWhichColor.fValue = -1.0;
        //uiData.backgroundImage.bValue = false;
    }    
}

// Function 92
void UI_ProcessSlider( inout UIContext uiContext, int iControlId, inout UIData_Value data, Rect sliderRect )
{    
    float fHandleSize = 8.0;
    
    bool bMouseOver = Inside( uiContext.vMouseCanvasPos, sliderRect ) && uiContext.bMouseInView;
    
    if ( uiContext.iActiveControl == IDC_NONE )
    {
        if ( uiContext.bMouseDown && (!uiContext.bMouseWasDown) && bMouseOver && !uiContext.bHandledClick )
        {
            uiContext.iActiveControl = iControlId;
            uiContext.bHandledClick = true;
        }
    }
    else
    if ( uiContext.iActiveControl == iControlId )
    {
        float fSlidePosMin = sliderRect.vPos.x + fHandleSize * 0.5f;
        float fSlidePosMax = sliderRect.vPos.x + sliderRect.vSize.x - fHandleSize * 0.5f;
        float fPosition = (uiContext.vMouseCanvasPos.x - fSlidePosMin) / (fSlidePosMax - fSlidePosMin);
        fPosition = clamp( fPosition, 0.0f, 1.0f );
        data.fValue = data.fRangeMin + fPosition * (data.fRangeMax - data.fRangeMin);
        if ( !uiContext.bMouseDown )
        {
            uiContext.iActiveControl = IDC_NONE;
        }
    }
        
    bool bActive = (uiContext.iActiveControl == iControlId);
    float fPosition = (data.fValue - data.fRangeMin) / (data.fRangeMax - data.fRangeMin);
    
    UI_DrawSliderX( uiContext, bActive, bMouseOver, fPosition, sliderRect, fHandleSize, false );    
}

// Function 93
vec3 post_get_image( vec2 uv, float subsample )
{
    float k = min( 2.3, 60. * sqrt( sqrt( g_pixelscale ) ) );
    float sharpen = .25;

    uv *= subsample;
    if( g_vrmode && uv.x >= .5 * subsample )
    	uv.x += .5 - .5 * subsample;

	vec3 col = ZERO;
#if WITH_IMG_GLARE
    if( ( g_game.switches & GS_TRMAP ) == 0u )
    {
    #if WITH_IMG_EXPERIMENTAL_ROD_ACUITY
        float y = max( 0., dot( textureLod( iChannel1, uv, 3. ).xyz - textureLod( iChannel1, uv, 4. ).xyz, VIS_SCOTOPIC_Y ) );
    #endif
	    vec3 wsum = ZERO;
    	for( float i = 0.; i < 10.; ++i )
    	{
            float w = 1. / ( 1. + exp2( k * i ) );
            if( i == 1. )
                w -= sharpen;
   		#if WITH_IMG_EXPERIMENTAL_ROD_ACUITY
			if( i < 3. )
            {
    			float suppress = VIS_LIMITS.z / ( 16384. * y + VIS_LIMITS.z );
                w *= g_exposure.y / ( g_exposure.y + suppress * VIS_LIMITS.x );
            }
        #endif
            col += w * textureLod( iChannel1, uv, i ).xyz;
            wsum += w;
    	}
        col = clamp( col / wsum, 0., 16. );
    }
    else
#endif
    	col = textureLod( iChannel1, uv, 0. ).xyz;

#if WITH_IMG_LENS
    if( ( g_game.switches & GS_TRMAP ) == 0u )
    {
	    float bias = .5 * log2( g_pixelscale );
	    col += .25 * FRACT_1_64 * lens_lookup( uv,-.5, -5. - bias, subsample );
	   	col += .25 * FRACT_1_64 * lens_lookup( uv, .5, -4. - bias, subsample );
	   	col += .25 * FRACT_1_64 * lens_lookup( uv, 1., -5.5 - bias, subsample );
	    col += .25 * FRACT_1_64 * lens_lookup( uv, 2., -7. - bias, subsample );
    }
#endif

    // add absolute threshold of vision
    col = col * 1. + vec3( COL_THRESHOLD );
    return col;
}

// Function 94
vec3 PostEffects(vec3 rgb, vec2 xy)
{
	// Gamma first...
	rgb = pow(rgb, vec3(0.45));

	// Then...
	#define CONTRAST 1.2
	#define SATURATION 1.3
	#define BRIGHTNESS 1.4
	rgb = mix(vec3(.5), mix(vec3(dot(vec3(.2125, .7154, .0721), rgb*BRIGHTNESS)), rgb*BRIGHTNESS, SATURATION), CONTRAST);
	// Noise...
	// rgb = clamp(rgb+Hash(xy*iTime)*.1, 0.0, 1.0);
	// Vignette...
	rgb *= .4+0.5*pow(40.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 0.2 );	

	return rgb;
}

// Function 95
void processSliders(in vec2 fragCoord)
{
    vec4 sliderVal = texture(iChannel0,vec2(0,0));
	ROUGHNESS_MAP_UV_SCALE *= 0.1*pow(10.,2.0*sliderVal[0]);
}

// Function 96
vec2 PosToSphere(vec3 pos)
{
  float x = atan(pos.z, pos.x); 
  float y = acos(pos.y / length(pos)); 
  return vec2(2.*x / PI_TWO, 2.*y / PI);
}

// Function 97
void ProcessCamPos(vec3 u, vec4 rot
){objPos[oCam]= u
 ;objRot[oCam]=rot;}

// Function 98
void UI_ProcessScrollbarPanelEnd( inout UIContext uiContext, inout UIPanelState scrollbarState )
{
    UI_PanelEnd( uiContext, scrollbarState );    
}

// Function 99
bool postprocess(inout vec4 fragColor, ivec2 address)
{
	if (iFrame < NUM_LIGHTMAP_FRAMES)
        return false;

    int pass = iFrame - NUM_LIGHTMAP_FRAMES;
    bool blur = pass >= NUM_DILATE_PASSES;

    const ivec2 MAX_COORD = ivec2(LIGHTMAP_SIZE.x - 1u, LIGHTMAP_SIZE.y/4u - 1u);
    vec4
        N  = texelFetch(iChannel1, clamp(address + ivec2( 0, 1), ivec2(0), MAX_COORD), 0),
        S  = texelFetch(iChannel1, clamp(address + ivec2( 0,-1), ivec2(0), MAX_COORD), 0),
        E  = texelFetch(iChannel1, clamp(address + ivec2( 1, 0), ivec2(0), MAX_COORD), 0),
        W  = texelFetch(iChannel1, clamp(address + ivec2(-1, 0), ivec2(0), MAX_COORD), 0),
        NE = texelFetch(iChannel1, clamp(address + ivec2( 1, 1), ivec2(0), MAX_COORD), 0),
        SE = texelFetch(iChannel1, clamp(address + ivec2( 1,-1), ivec2(0), MAX_COORD), 0),
        NW = texelFetch(iChannel1, clamp(address + ivec2(-1, 0), ivec2(0), MAX_COORD), 0),
        SW = texelFetch(iChannel1, clamp(address + ivec2(-1, 0), ivec2(0), MAX_COORD), 0);

    N  = vec4(fragColor.yzw, N.x);
    NE = vec4(E.yzw, NE.x);
    NW = vec4(W.yzw, NW.x);
    S  = vec4(S .w, fragColor.xyz);
    SE = vec4(SE.w, E.xyz);
    SW = vec4(SW.w, W.xyz);

    LightmapSample
        current = decode_lightmap_sample(fragColor),
        total = empty_lightmap_sample();

    accumulate(total, N);
    accumulate(total, S);
    accumulate(total, E);
    accumulate(total, W);
#if USE_DIAGONALS
    accumulate(total, NE);
    accumulate(total, NW);
    accumulate(total, SE);
    accumulate(total, SW);
#endif

    if (blur)
    {
        accumulate(total, current);
	    fragColor = encode(total);
    }
    else
    {
        vec4 neighbors = encode(total);
        fragColor = mix(fragColor, neighbors, lessThanEqual(current.weights, vec4(0)));
    }
    
    return true;
}

// Function 100
void post_map_overlay( inout vec3 col, vec2 coord )
{
    coord = ( coord - g_overlayframe.xy ) * iResolution.xy / g_overlayframe.zw;
	map_position( col, coord );
    if( dot( g_game.mapmarker, g_game.mapmarker ) > 0. )
    	map_marker( col, coord );
    if( dot( g_game.waypoint, g_game.waypoint ) > 0. )
    	map_waypoint( col, coord );
    if( g_env.H == 0. ) 
    	map_orbit_track( col, coord );
}

// Function 101
void UI_ProcessCheckbox( inout UIContext uiContext, int iControlId, inout UIData_Bool data, Rect checkBoxRect )
{    
    bool bMouseOver = Inside( uiContext.vMouseCanvasPos, checkBoxRect ) && uiContext.bMouseInView;
    
    if ( uiContext.iActiveControl == IDC_NONE )
    {
        if ( uiContext.bMouseDown && (!uiContext.bMouseWasDown) && bMouseOver && !uiContext.bHandledClick )
        {
            uiContext.iActiveControl = iControlId;
            uiContext.bHandledClick = true;
        }
    }
    else
    if ( uiContext.iActiveControl == iControlId )
    {
        if ( !uiContext.bMouseDown )
        {
            uiContext.iActiveControl = IDC_NONE;
            if ( bMouseOver )
            {
                data.bValue = !data.bValue;
            }
        }
    }
    
    bool bActive = (uiContext.iActiveControl == iControlId);
    
    UI_DrawCheckbox( uiContext, bActive, bMouseOver, data.bValue, checkBoxRect );    
}

// Function 102
void post_hmd_overlay( inout vec3 col, vec2 coord, vec3 cc )
{
    col += g_hudcolor * hmd_center_dot( coord );
    col += g_hudcolor * hmd_waterline( coord );
    col += g_hudcolor * hmd_flight_path_marker( coord );
   	col += g_hudcolor * hmd_pitch_ladder( coord, cc );
    if( dot( g_game.waypoint, g_game.waypoint ) > 0. )
        col += g_hudcolor * hmd_waypoint( coord );
}

// Function 103
void UI_ProcessColorPickerH( inout UIContext uiContext, int iControlId, inout UIData_Color data, Rect pickerRect )
{
    bool bMouseOver = Inside( uiContext.vMouseCanvasPos, pickerRect ) && uiContext.bMouseInView;
    
    vec3 vHSV = data.vHSV;
    
    if ( uiContext.iActiveControl == IDC_NONE )
    {
        if ( uiContext.bMouseDown && (!uiContext.bMouseWasDown) && bMouseOver && !uiContext.bHandledClick )
        {
            uiContext.iActiveControl = iControlId;
            uiContext.bHandledClick = true;
        }
    }
    else
    if ( uiContext.iActiveControl == iControlId )
    {
        float fPos = (uiContext.vMouseCanvasPos.y - pickerRect.vPos.y) / pickerRect.vSize.y;
        fPos = clamp( fPos, 0.0f, 1.0f );
        
        vHSV.x = fPos;
        
        if ( !uiContext.bMouseDown )
        {
            uiContext.iActiveControl = IDC_NONE;
        }
    }
    
    data.vHSV = vHSV;
    
    bool bActive = (uiContext.iActiveControl == iControlId);
    
    UI_DrawColorPickerH( uiContext, bActive, vHSV, pickerRect );
}


```