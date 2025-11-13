// Reusable Ui Elements Game Functions
// Automatically extracted from game/interactive-related shaders

// Function 1
int HoldSpeedButton(vec4 mouse, vec2 screen)
{
    return HoldButton(mouse, screen, vec2(0.14, 0.04));
}

// Function 2
uvec2 asuint2(vec2 x) { return uvec2(asuint2(x.x ), asuint2(x.y)); }

// Function 3
void UI_DrawSliderX( inout UIContext uiContext, bool bActive, bool bMouseOver, float fPosition, Rect sliderRect, float fHandleSize, bool scrollbarStyle )
{
	if (!uiContext.bPixelInView || Outside( uiContext.vPixelCanvasPos, sliderRect ))
        return;
    
    Rect horizLineRect;
    
    horizLineRect = sliderRect;
    if (!scrollbarStyle)
    {
	    float fMid = sliderRect.vPos.y + sliderRect.vSize.y * 0.5;
    	horizLineRect.vPos.y = fMid - 2.0;
    	horizLineRect.vSize.y = 4.0;
    }

#ifdef NEW_THEME    
    DrawBorderRect( uiContext.vPixelCanvasPos, horizLineRect, cSliderLineCol, uiContext.vWindowOutColor );
#else    
    DrawBorderIndent( uiContext.vPixelCanvasPos, horizLineRect, uiContext.vWindowOutColor );
#endif

    float fSlideMin = sliderRect.vPos.x + fHandleSize * 0.5f;
    float fSlideMax = sliderRect.vPos.x + sliderRect.vSize.x - fHandleSize * 0.5f;

    float fDistSlider = (fSlideMin + (fSlideMax-fSlideMin) * fPosition);

    Rect handleRect;

    handleRect = sliderRect;
    handleRect.vPos.x = fDistSlider - fHandleSize * 0.5f;
    handleRect.vSize.x = fHandleSize;

    vec4 handleColor = vec4(0.75, 0.75, 0.75, 1.0);
    if ( bActive )
    {
        handleColor.rgb += 0.1;
    }       
    
    // highlight
#ifdef NEW_THEME     
    if ( (uiContext.vPixelCanvasPos.y - handleRect.vPos.y) < handleRect.vSize.y * 0.3 )
    {
        handleColor.rgb += 0.05;
    }
#endif    

    DrawRect( uiContext.vPixelCanvasPos, handleRect, handleColor, uiContext.vWindowOutColor );

#ifdef NEW_THEME   
    DrawBorderRect( uiContext.vPixelCanvasPos, handleRect, cSliderHandleOutlineCol, uiContext.vWindowOutColor );
#else    
    DrawBorderOutdent( uiContext.vPixelCanvasPos, handleRect, uiContext.vWindowOutColor );
#endif    
}

// Function 4
vec4 dfunc_ui_box(int idx, int row) {
    
    return vec4(inset_ctr.x + (float(idx - 2))*text_size,
    	        dfunc_y - float(1-row)*text_size,
                vec2(0.45*text_size));
    
}

// Function 5
vec2 UIDrawContext_CanvasPosToScreenPos( UIDrawContext drawContext, vec2 vCanvasPos )
{
    return vCanvasPos - drawContext.vOffset + drawContext.viewport.vPos;
}

// Function 6
vec2 	UIStyle_SliderSize()			{ return vec2(128.0, 32.0f); }

// Function 7
vec4 link_ui_box() {
    
    return vec4(inset_ctr.x + 2.85*text_size,
                dfunc_y - 0.5*text_size,
                0.3*text_size, 0.5*text_size);
    
}

// Function 8
void RebuildFrame(int frame, vec4 rawData, inout Camera camera, inout vec3 normal)
{
    int frameBounce = BounceFrame(frame);
    
    if(frameBounce > 0)
    {        
        if(frameBounce > 1)
        {
            // Jump once
            camera.origin = rawData.yzw;
            Bounce(frame, camera, normal);
        }
        
        camera.origin = camera.origin + camera.direction * rawData.r;
        Bounce(frame, camera, normal);
    }    
}

// Function 9
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
        if ( data.bInteger )
        {
            data.fValue = floor( data.fValue + 0.5 );
        }
        if ( !uiContext.bMouseDown )
        {
            uiContext.iActiveControl = IDC_NONE;
        }
    }
        
    bool bActive = (uiContext.iActiveControl == iControlId);
    float fPosition = (data.fValue - data.fRangeMin) / (data.fRangeMax - data.fRangeMin);
    
    UI_DrawSliderX( uiContext, bActive, bMouseOver, fPosition, sliderRect, fHandleSize, false );    
}

// Function 10
vec4 mainImageUI2AD37(out vec4 o, in vec2 u
){o=vec4(0)
 #ifdef Scene2D
  ;o=pdOver(iDiegeticUIshow(u),o)//ui dots
  ;o=pdOver(o,iCB(o,u))//iCB(o,u)
  ;//o=pdOver(o,ltj3Wc(o,u,iResolution,iMouse))//backsrop is a 2d srawing
 #else
  #ifdef SceneTR
   ;o=pdOver(iDiegeticUIshow(u),o)//ui dots
   ;o=pdOver(o,iCB(o,u))//bezier+appolonean stuff
   ;o=pdOver(o,mTR(o,u)) //backfrop is traced 3d scene (TemporalReprojection+brdf)
  #else
   ;o=pdOver(iDiegeticUIshow(u),o)//ui dots
   ;o=pdOver(o,iCB(o,u))//bezier+appolonean stuff
   ;o=pDoOver(iAD)  //backfrop is marched 3d scene (automatic Differentiation)
  #endif
 #endif
 ;return o      /**/
 ;}

// Function 11
vec4 uiColor(int id){return texture(iChannel0, vec2(float(id)+.5,1.5)/iResolution.xy);}

// Function 12
vec4 squiggles( vec2 uv, float time, float a, float r, float dx )
{
    time += .5;
    float timeOffset = floor(time);
    const float timeSlices = 30.;
    const float width = .008;
        
    const float nrays = 30.;
    float qa = (a/tau+.5) * (nrays);
    float qi = floor(qa);

    const float power = 3.;
    const float scale = 2.;

    float ti1 = floor(fract(time + 0./timeSlices) * timeSlices);
	float ti2 = floor(fract(time + 1./timeSlices) * timeSlices);
	float ti3 = floor(fract(time + 2./timeSlices) * timeSlices);
    
    // background
    vec4 color = vec4(.0);
    if (time < 0.) return color;
    if (ti3 < ti2) return color;
    if (ti2 < ti1) return color;
    
    ivec2 texSize = textureSize( iChannel0 , 0 );
    float ts = float(texSize.x);
    
    float index0 = float(qi*nrays);
    vec4 rnd0 = texture(iChannel0, vec2((index0+timeOffset+.5)/ts, 0.5));
    float lifetime = floor(fract(rnd0.x*rnd0.x*.5) * timeSlices);
    if (ti3 > lifetime) return color;
    
    float index = float(qi*nrays/*+ti2*/);
    vec4 rnd1 = texture(iChannel0, vec2((index+timeOffset+.5)/ts, (ti1+.5)/ts));
    vec4 rnd2 = texture(iChannel0, vec2((index+timeOffset+.5)/ts, (ti2+.5)/ts));
    vec4 rnd3 = texture(iChannel0, vec2((index+timeOffset+.5)/ts, (ti3+.5)/ts));

    float age1 = ti1/lifetime;
    float age2 = ti2/lifetime;
    float age3 = ti3/lifetime;
    float ageDamp1 = 1. - age1;
    float ageDamp2 = 1. - age2;
    float ageDamp3 = 1. - age3;
    float ageMargin1 = ageDamp1 / 2.;
    float ageMargin2 = ageDamp2 / 2.;
    float ageMargin3 = ageDamp3 / 2.;
    
    float qu1 = (( (qi-.0+rnd1.x*ageDamp1+ageMargin1) / nrays) - .5) * tau;
    float qu2 = (( (qi-.0+rnd2.x*ageDamp2+ageMargin2) / nrays) - .5) * tau;
    float qu3 = (( (qi-.0+rnd3.x*ageDamp3+ageMargin3) / nrays) - .5) * tau;

    float rv1 = pow( (ti1+rnd1.y)/10., 1./power ) / scale;
    float rv2 = pow( (ti2+rnd2.y)/10., 1./power ) / scale;
    float rv3 = pow( (ti3+rnd3.y)/10., 1./power ) / scale;

    vec2 qruv1 = rv1 * a2d(qu1);
    vec2 qruv2 = rv2 * a2d(qu2);
    vec2 qruv3 = rv3 * a2d(qu3);

    vec2 pa = qruv1;
    vec2 pb = qruv2;
    vec2 pc = qruv3;
    
    vec4 d = sdBezier( 
        uv, 
        vec3(.5*(pa+pb), width), 
        vec3(pb        , width),
        vec3(.5*(pb+pc), width));
    
    // Try to get rid of POV overlap artifacts. 
    // This might be specific to my mac's display gamma.
    float blend = 10. * dx;
    float bias = blend * -0.1; 
    float alpha = .75*S(1., .75, age2);
    
    float aa = S(0., -dx, d.x);
    float blendIn = S(-blend+bias, blend+bias, d.w);
    float blendOut = S(1.+blend-bias, 1.-blend-bias, d.w);
    if (d.x < dx) color = comp(color, premult(vec4(vec3(1.), alpha*aa)));

    return color;
}

// Function 13
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

// Function 14
vec3 viridis_quintic( float x )
{
	x = saturate( x );
	vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3(
		dot( x1.xyzw, vec4( +0.280268003, -0.143510503, +2.225793877, -14.815088879 ) ) + dot( x2.xy, vec2( +25.212752309, -11.772589584 ) ),
		dot( x1.xyzw, vec4( -0.002117546, +1.617109353, -1.909305070, +2.701152864 ) ) + dot( x2.xy, vec2( -1.685288385, +0.178738871 ) ),
		dot( x1.xyzw, vec4( +0.300805501, +2.614650302, -12.019139090, +28.933559110 ) ) + dot( x2.xy, vec2( -33.491294770, +13.762053843 ) ) );
}

// Function 15
uint to_uint(int n) {
  // Definite conversion of ints to uints.
  return uint(n >= 0? 2*n: -2*n-1);
}

// Function 16
void buildSetup(out SceneSetup res, vec3 target)
{

    res.cup = translate(target);
    
    res.spoon = rotationX(-1.5) *
        		rotationY(-1.0) *
                translate(vec3(2.0, -0.48, 9.0)) * res.cup;
    
    res.ashtray = rotationY(-0.5) * 
        		  translate(vec3(-10.0, -5.0, -20.0)) * res.cup; 
 
    res.cig1 =  rotationX(1.8)*
                translate(vec3(0.0, 1.0, 10.0)) *
                res.ashtray;

    mat4 swizzle = 	mat4( 1, 0, 0, 0,
                          0, 0, 1, 0,
                          0, 1, 0, 0,
                          0, 0, 0, 1);
    
    res.cig2 =  rotationZ(-2.5)*
                translate(vec3(-3, -4.0, 3.5)) *
        		swizzle *
                res.ashtray;
    
    res.jug =  rotationY(3.5) * translate(vec3(18.0, -0.6, -8)) * res.cup;
        
    res.sugar = rotationY(2.0) * translate(vec3(-6.2, -0.4, 4.3)) * res.cup;
}

// Function 17
vec3 	UIStyle_ColorPickerSize()		{ return vec3(128.0, 128.0, 32.0); }

// Function 18
vec2 quickTwoSum(float a, float b, float junk) {
    float s = final(a + b, junk);
    float e = b - (s - a);
    return vec2(s, e);
}

// Function 19
void UIStyle_GetFontStyleTitle( inout LayoutStyle style, inout RenderStyle renderStyle )
{
    style = LayoutStyle_Default();
    style.vSize *= 0.75;
	renderStyle = RenderStyle_Default( vec3(1.0) );
}

// Function 20
void gui_pqr_update() {
    
    if (fc.x != PQR_COL) { return; }
    
    for (int i=0; i<3; ++i) {

        int j = (i+1)%3;
        int k = 3-i-j;

        for (float delta=-1.; delta <= 1.; delta += 2.) {
            
            bool enabled = (delta < 0.) ? data[i] > 2. : data[i] < 5.;
            if (!enabled) { continue; }

            float d = box_dist(iMouse.xy, tri_ui_box(i, delta));       
            if (d > 0.) { continue; }

            data[i] += delta;
            
            int iopp = delta*data[j] > delta*data[k] ? j : k;
            
            for (int cnt=0; cnt<5; ++cnt) {
                if (valid_pqr(data.xyz)) { continue; }
                data[iopp] -= delta; 
            }   
            
        }
    }

}

// Function 21
vec4 gui_arrow_right(vec4 col, vec2 uv, vec2 pos, float scale, bool check)
{
	float unit = asp * 0.01 * scale;
    float h;
    
    h = triangle(uv, pos+vec2(-unit*1.8, -unit*2.), pos+vec2(-unit*1.8, unit*2.), pos+vec2(unit*1.8, 0.));
    if(!check) h = abs(h);
    col = mix(col, vec4(vec3(0.5), 1.), smoothstep(0.01, 0., h));
    
    
    return col;
}

// Function 22
void UI_DrawWindowTitleBar( inout UIContext uiContext, bool bActive, Rect titleBarRect, inout UIWindowState window )
{   
	if (!uiContext.bPixelInView || Outside( uiContext.vPixelCanvasPos, titleBarRect ))
        return;
    
    vec4 colorA = vec4(cTitleBarA, 1.0);
    vec4 colorB = vec4(cTitleBarB, 1.0);
       
    if ( bActive )
    {
        colorA.rgb = cTitleBarAActive;
        colorB.rgb = cTitleBarBActive;
    }

    float t = (uiContext.vPixelCanvasPos.x - titleBarRect.vPos.x) / 512.0;
    t = clamp( t, 0.0f, 1.0f );
    uiContext.vWindowOutColor = mix( colorA, colorB, t );
    
    {
        LayoutStyle style;
        RenderStyle renderStyle;
        UIStyle_GetFontStyleTitle( style, renderStyle );

        vec2 vTextOrigin = vec2(0);
        if ( FLAG_SET(window.uControlFlags, WINDOW_CONTROL_FLAG_MINIMIZE_BOX) )
        {
        	vTextOrigin.x += titleBarRect.vSize.y;
        }
        
        PrintState state = UI_PrintState_Init( uiContext, style, vTextOrigin );    
        PrintWindowTitle( state, style, window.iControlId );    
        RenderFont( state, style, renderStyle, uiContext.vWindowOutColor.rgb );
    }
}

// Function 23
vec4 verticalRadioButton(in vec2 uv, in vec2 min_b, in vec2 max_b, in float _val, in float n)
{
    vec2 center = (min_b + max_b)*0.5;
    vec2 size1 = (max_b - min_b) * 0.5;
    vec2 frame = 0.98 * size1;
    
    float ratio = iResolution.x / iResolution.y;
    vec2 scl_uv = uv;
    scl_uv.x *= ratio;
    
    vec3 background = vec3(0.05, 0.02, 0.01);
    
    float inside = step(sdBox(uv - center, frame) - 0.05*size1.y, 0.);
    float boundary = step(sdBox(uv - center, size1) - 0.06*size1.y, 0.) - inside;
    float val = 0.5 + 0.3*sin(PI*0.5 + PI2*saturate(uv.y, min_b.y, max_b.y));
    float start_y = center.y - size1.y;
    float bval = saturate(uv.y, start_y, center.y + size1.y);
    
    float inv_n = 1. / n;
    
    float modeDiscr = floor(_val * n)*inv_n;
    vec2 center_button = vec2(center.x*ratio, start_y + (modeDiscr + inv_n*0.5)*size1.y*2.);
    float bcircle = length(scl_uv - center_button);
    float button = smoothstep(0.005, 0.007, bcircle) - smoothstep(0.007, 0.009, bcircle);
    
    vec4 color = clamp(vec4(background.x + 0.15*button, background.y + button*0.45 + boundary*val, 
                            button*0.5 + background.z +boundary*val, 1.), 0., step(0.5, inside+boundary));

    return color;
}

// Function 24
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

// Function 25
void gui_vertex_update() {    

    if (fc.x != BARY_COL && fc.x != SPSEL_COL) { return; }

    if (length(iMouse.zw - inset_ctr)*inset_scl > 1.) {       

        return; 
        
    } else {

        vec3 q = sphere_from_gui(iMouse.xy);
        
        vec4 spsel;
        int s = tri_snap(q);

        if (abs(iMouse.zw) == iMouse.xy && s >= 0) {
            if (s < 3) {
                if (fc.x == BARY_COL) {
                    data.xyz = bary_from_sphere( tri_verts[s] );
                } else {
                    data = vec4(0);
                }
            } else { 
                if (fc.x == BARY_COL) {
                    data.xyz = bary_from_sphere( tri_spoints[s-3] );
                } else {
                    data = vec4(0);
                    data[s-3] = 1.;
                }
            }
        } else {
            if (fc.x == BARY_COL) {
                data.xyz = bary_from_sphere( tri_closest(q) );
            } else {
                data = vec4(0);
            }
        }

    }
    
}

// Function 26
vec3 sphere_from_gui(in vec2 p) {
    
    p -= inset_ctr;
    p *= inset_scl;
    
    float dpp = dot(p, p);
    
    if (dpp >= 1.) {
        return vec3(p/sqrt(dpp), 0);
    } else {    
        vec3 p3d = vec3(p, sqrt(1. - dot(p, p)));
        return ortho_proj*p3d;
    }
    
}

// Function 27
vec4 tri_ui_box(int idx, float delta) {
    
    return vec4(char_ui_box(idx).xy + vec2(0, 0.9*delta*text_size), 
                0.4*text_size, 0.3*text_size);
    
}

// Function 28
vec4 solveFluid(sampler2D smp, vec2 uv, vec2 w, float time, vec3 mouse, vec3 lastMouse)
{
	const float K = 0.2;
	const float v = 0.55;
    
    vec4 data = textureLod(smp, uv, 0.0);
    vec4 tr = textureLod(smp, uv + vec2(w.x , 0), 0.0);
    vec4 tl = textureLod(smp, uv - vec2(w.x , 0), 0.0);
    vec4 tu = textureLod(smp, uv + vec2(0 , w.y), 0.0);
    vec4 td = textureLod(smp, uv - vec2(0 , w.y), 0.0);
    
    vec3 dx = (tr.xyz - tl.xyz)*0.5;
    vec3 dy = (tu.xyz - td.xyz)*0.5;
    vec2 densDif = vec2(dx.z ,dy.z);
    
    data.z -= dt*dot(vec3(densDif, dx.x + dy.y) ,data.xyz); //density
    vec2 laplacian = tu.xy + td.xy + tr.xy + tl.xy - 4.0*data.xy;
    vec2 viscForce = vec2(v)*laplacian;
    data.xyw = textureLod(smp, uv - dt*data.xy*w, 0.).xyw; //advection
    
    vec2 newForce = vec2(0);
    #ifndef MOUSE_ONLY
    #if 1
    newForce.xy += 0.75*vec2(.0003, 0.00015)/(mag2(uv-point1(time))+0.0001);
    newForce.xy -= 0.75*vec2(.0003, 0.00015)/(mag2(uv-point2(time))+0.0001);
    #else
    newForce.xy += 0.9*vec2(.0003, 0.00015)/(mag2(uv-point1(time))+0.0002);
    newForce.xy -= 0.9*vec2(.0003, 0.00015)/(mag2(uv-point2(time))+0.0002);
    #endif
    #endif
    
    if (mouse.z > 1. && lastMouse.z > 1.)
    {
        vec2 vv = clamp(vec2(mouse.xy*w - lastMouse.xy*w)*400., -6., 6.);
        newForce.xy += .001/(mag2(uv - mouse.xy*w)+0.001)*vv;
    }
    
    data.xy += dt*(viscForce.xy - K/dt*densDif + newForce); //update velocity
    data.xy = max(vec2(0), abs(data.xy)-1e-4)*sign(data.xy); //linear velocity decay
    
    #ifdef USE_VORTICITY_CONFINEMENT
   	data.w = (tr.y - tl.y - tu.x + td.x);
    vec2 vort = vec2(abs(tu.w) - abs(td.w), abs(tl.w) - abs(tr.w));
    vort *= VORTICITY_AMOUNT/length(vort + 1e-9)*data.w;
    data.xy += vort;
    #endif
    
    data.y *= smoothstep(.5,.48,abs(uv.y-0.5)); //Boundaries
    
    data = clamp(data, vec4(vec2(-10), 0.5 , -10.), vec4(vec2(10), 3.0 , 10.));
    
    return data;
}

// Function 29
float leftBuilding(vec3 p){
    p = opBend(p, -0.1);
    float box = sdBox(p, vec3(1.5, 4., 1));
    float ceiling = sdBox(p + vec3(0,-3.8,0), vec3(1.65, .1, 1.3));
    float window0 = sdBox(p + vec3(-1.4, -1.8, .65), vec3(.14, .2, .1));
    float window1 = sdBox(p + vec3(-1.4, -1.4, -.3), vec3(.14, .2, .1));
    float window2 = sdBox(p + vec3(-1.4, -.7, .1), vec3(.14, .2, .1));
    float windows = min(min(window2, window0), window1);
	float door = sdBox(p + vec3(-1.4, .4, .5), vec3(.15, .6, .1));
    float shape1 = min(windows, door);
    box = min(box, ceiling);
    float building = boolSubtraction(shape1, box);

    return building;
}

// Function 30
float Buildings(vec2 uv, out vec3 color)
{
    Building[5] buildings = Building[5]
    (
        // long thin windows
        Building(
            vec2(0.25, 0.85), vec2(0.406, 0.5),
            BUILDING_COLOR_0,
            vec2(0.00275, 0.08), vec2(-0.358, -0.65),
            vec2(0.0105, 0.18), vec2(11, 9)
        ),
        // many small windows with small gap
        Building(
            vec2(0.25, 0.94), vec2(-0.56, 1.1),
            BUILDING_COLOR_1,
            vec2(0.006, 0.04), vec2(0.37, -0.835),
            vec2(0.014, 0.12), vec2(8, 9)
        ),
        // square windows
        Building(
            vec2(0.27, 0.94), vec2(-0.09, 0.87),
            BUILDING_COLOR_2,
            vec2(0.01, 0.04), vec2(0.025, -0.8351),
            vec2(0.03, 0.12), vec2(6, 9)
        ),
        // lerger wide windows, small gap
        Building(
            vec2(0.27, 0.73), vec2(.78, 0.71),
            BUILDING_COLOR_3,
            vec2(0.02, 0.025), vec2(-0.66, -0.66),
            vec2(0.045, 0.07), vec2(4, 11)
        ),
        // vertically thin horizontally wide windows
        Building(
            vec2(0.38, 0.5), vec2(-1., 0.5),
            BUILDING_COLOR_4,
            vec2(0.02, 0.01), vec2(0.52, -0.45),
            vec2(0.05, 0.05), vec2(5, 11)
        )
    );
    // cache the window color so I don't
    // have to recalculate it for every building
    vec3 windowColor = WindowFBM(uv);
    uv = uv * vec2(1.5, 2) + vec2(1.5, 0.8);
    float scene = Build(uv, buildings[0], windowColor, color);
    
    for (int i = 1; i < 5; i++)
    {
        vec3 bColor;
        float b = max(
            Build(uv, buildings[i], windowColor, bColor), 
            -scene); 
        color = scene < b ? color : bColor;
        scene = min(scene, b);
    }
    
    // repeating some buildings to fill in the gaps!
    vec3 bColor;
    float b = max(Build(uv + vec2(-1.9, 0.5), 
        buildings[0], windowColor, bColor), -scene); 
    LerpWhiteTo(bColor, OUTLINE_COLOR, 0.5);
    color = scene < b ? color : bColor;
    scene = min(scene, b);
    
    b = max(Build(uv + vec2(1.7, 0.1), buildings[1], 
        windowColor, bColor), -scene); 
    LerpWhiteTo(bColor, OUTLINE_COLOR, 0.5);
    color = scene < b ? color : bColor;
    scene = min(scene, b);
    return scene;
}

// Function 31
float ruins(in vec3 p) {
    vec3 rp = p;
    float d = 0.;
    float s = fractal_size;
    
    #define seed fractal_seed
    #define seed2 fractal_seed2
    for (int i = 0; i < fractal_iter; i++) {
        rp -= s/8.;
        d = max(-sdBox(mod(abs(rp), s*2.)-s, vec3(s*.9)), d);
        
        if (mod(float(i),2.) > 0.) {
            rp.xz = abs(rot(rp.xz,float(i)*1.2+seed));
        } else {
            rp.zy = abs(rot(rp.zy,float(i)*1.2+seed2));
        }
        
    	s /= 2.;
    }
                       
    return max(sdTriPrism(p*vec3(1.,-1.,1.), vec2(60., 40.)), d);
}

// Function 32
UIDrawContext UIDrawContext_TransformChild( UIDrawContext parentContext, UIDrawContext childContext )
{
    UIDrawContext result;
    
    // The child canvas size is unmodified
    result.vCanvasSize = childContext.vCanvasSize;

    // Child viewport positions are in the parent's canvas
    // Transform them to screen co-ordinates    
    result.viewport.vPos = UIDrawContext_CanvasPosToScreenPos( parentContext, childContext.viewport.vPos );
    vec2 vMax = childContext.viewport.vPos + childContext.viewport.vSize;
    vec2 vScreenMax = UIDrawContext_CanvasPosToScreenPos( parentContext, vMax );
    result.viewport.vSize = vScreenMax - result.viewport.vPos;
    result.vOffset = childContext.vOffset;
    
    // Now clip the view so that it is within the parent view
    vec2 vViewMin = max( result.viewport.vPos, parentContext.clip.vPos );
    vec2 vViewMax = min( result.viewport.vPos + result.viewport.vSize, parentContext.clip.vPos + parentContext.clip.vSize );

    // Clip view to current canvas
    vec2 vCanvasViewMin = result.viewport.vPos - result.vOffset;
    vec2 vCanvasViewMax = vCanvasViewMin + result.vCanvasSize;
    
    vViewMin = max( vViewMin, vCanvasViewMin );
	vViewMax = min( vViewMax, vCanvasViewMax );
    
    result.clip = Rect( vViewMin, vViewMax - vViewMin );
    
    return result;
}

// Function 33
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

// Function 34
DistBuilding distBuilding(vec3 q1, vec3 id) {
    DistBuilding res;
    float n = n21(id.xy);
    float n1 = fract(n*4553.33);
    float n2 = fract(n*145.33);

    float obj = BLD_RECT;

    if (n1 > .7) {
        obj = BLD_HEX;
    } else if (n2 > .8) {
        obj = BLD_TUBE;
    }

    float baseSize = defaultBaseSize;

    float h = baseSize;

    if (obj == BLD_TUBE && n1 < .2 ) {
        n = (sin(q1.x + (n1*50.)) * .5 + .5);
    }

    float ah = n * .5;

    h += ah;
    q1.z -= ah;
    q1.z -= defaultBaseSize;

    float d;



    vec3 size = vec3(baseSize, baseSize, h);

    if (obj == BLD_HEX) {
        d = sdHexPrism(q1, vec2(size.x, size.z));
    } else if (obj == BLD_TUBE) {
        float tmp = q1.z;
        q1.z = q1.y;
        q1.y = tmp;
        d = sdCappedCylinder(q1, baseSize, size.z);
    } else {
        if (n1 > .3) {
            size.x *= .5;
            size.y *= 1.5;
        } else if (n2 > .5) {
            size.y *= .5;
            size.x *= 1.5;
        }

        if (n < .6) {
            vec3 q2,nsize, nsize3, q3, d3;
            if (n2 < .2 && size.x == size.y) {
                d = sdBox(q1, size);
                q2 = vec3(q1.x, q1.y, q1.z - size.z);
                nsize = vec3(size.xy/1.5, size.z*2.);
                q3 = vec3(q1.x, q1.y, q1.z - size.z - size.z / 1.8);
                nsize3 = vec3(size.xy/(1.5*1.5), size.z*2. + size.z / 1.8);
            } else {
                q1 += vec3(0.1, -0.08, 0.);
                d = sdBox(q1, size);
                float extraH = size.z + size.z * n1;
                q2 = vec3(q1.x - .18, q1.y + .18, q1.z - extraH);
                nsize = vec3(size.xy, size.z + extraH);
                if (n1 > .4) {
                    nsize.xy = nsize.yx;
                }
            }

            float d2 = sdBox(q2, nsize);
            if (d2 < d) {
                q1 = q2;
                size = nsize;
                d = d2;
            }

            if (nsize3.x != 0.) {
                float d3 = sdBox(q3, nsize3);
                if (d3 < d) {
                    q1 = q3;
                    size = nsize3;
                    d = d3;
                }
            }

        } else {
            d = sdBox(q1, size);
        }
    }

    res.d = d;
    res.q1 = q1;
    res.size = size;
    res.objId = obj;
    res.height = size.z;

    return res;
}

// Function 35
vec4 iconUIBox(ivec2 idx) {
    
    vec2 iconCtr = iconCenter;
    
    iconCtr = floor(iconCtr+0.5);
    
    vec2 scl = vec2(2.5*iconSize, 3.*iconSize);
    iconCtr += vec2(float(idx.x), float(-idx.y))*scl + vec2(-1.5, 0.5)*scl; 
    
    return vec4(iconCtr, vec2(iconSize));
    
}

// Function 36
void gui_misc_update() {
    
    if (fc.x != MISC_COL) { return; }
        
    if (box_dist(iMouse.xy, link_ui_box()) < 0.) {
        data.x = 1. - data.x;
    }
    
    for (int i=0; i<2; ++i) {
        if (box_dist(iMouse.xy, color_ui_box(i)) < 0.) {
            data.y = float(i);
        }
    }
    
}

// Function 37
float evalQuintic(in float x, in GeneralQuintic q) {
    return ((((q.a * x + q.b) * x + q.c) * x + q.d) * x + q.e) * x + q.f;
}

// Function 38
void UI_StoreDataBool( inout UIContext uiContext, UIData_Bool dataBool, int iData )
{
    vec4 vData0 = vec4(0);
    vData0.x = dataBool.bValue ? 1.0 : 0.0;
    StoreVec4( ivec2(iData,0), vData0, uiContext.vOutData, ivec2(uiContext.vFragCoord) );            
}

// Function 39
float 	UIStyle_TitleBarHeight() 		{ return 32.0; }

// Function 40
bool UI_DrawWindowCloseBox( inout UIContext uiContext, Rect closeBoxRect )
{
	if (!uiContext.bPixelInView || !Inside( uiContext.vPixelCanvasPos, closeBoxRect ))
        return false;
    
    vec2 vCrossPos = closeBoxRect.vPos + closeBoxRect.vSize * 0.5;        
    vec2 vCrossSize = closeBoxRect.vSize * 0.5 * 0.4;
    vec4 crossColor = vec4(0.0, 0.0, 0.0, 1.0);

    vec2 vCrossSizeFlip = vCrossSize * vec2(1.0, -1.0);
    
    DrawLine( uiContext.vPixelCanvasPos, vCrossPos - vCrossSize, vCrossPos + vCrossSize, 2.0f, crossColor, uiContext.vWindowOutColor );
    DrawLine( uiContext.vPixelCanvasPos, vCrossPos - vCrossSizeFlip, vCrossPos + vCrossSizeFlip, 2.0f, crossColor, uiContext.vWindowOutColor );
    
    return true;
}

// Function 41
void UI_DrawSliderY( inout UIContext uiContext, bool bActive, bool bMouseOver, float fPosition, Rect sliderRect, float fHandleSize, bool scrollbarStyle )
{
	if (!uiContext.bPixelInView || Outside( uiContext.vPixelCanvasPos, sliderRect ))
        return;
    
    Rect horizLineRect;
    
    horizLineRect = sliderRect;
    if (!scrollbarStyle)
    {
	    float fMid = sliderRect.vPos.x + sliderRect.vSize.x * 0.5;
    	horizLineRect.vPos.x = fMid - 2.0;
    	horizLineRect.vSize.x = 4.0;
    }

    DrawBorderIndent( uiContext.vPixelCanvasPos, horizLineRect, uiContext.vWindowOutColor );

    float fSlideMin = sliderRect.vPos.y + fHandleSize * 0.5f;
    float fSlideMax = sliderRect.vPos.y + sliderRect.vSize.y - fHandleSize * 0.5f;

    float fDistSlider = (fSlideMin + (fSlideMax-fSlideMin) * fPosition);

    Rect handleRect;

    handleRect = sliderRect;
    handleRect.vPos.y = fDistSlider - fHandleSize * 0.5f;
    handleRect.vSize.y = fHandleSize;

    vec4 handleColor = vec4(0.75, 0.75, 0.75, 1.0);
    if ( bActive )
    {
        handleColor.rgb += 0.1;
    }

    DrawRect( uiContext.vPixelCanvasPos, handleRect, handleColor, uiContext.vWindowOutColor );
    DrawBorder( uiContext.vPixelCanvasPos, handleRect, uiContext.vWindowOutColor );
}

// Function 42
vec2 	UIStyle_SliderSize()			{ return vec2(192.0, 24.0f); }

// Function 43
bool isButtonPushed(vec2 a){
  return (buttonDownPos()==a);
    //berare that masked buttons can still be pushed, 
    //buttons masked|hidden by buttonMask() still have an effect, if you isist to,
    //filtering them here is overly excessive.
    //instead you can just not give masked buttons any effect.
/*
//sadly the below shortcut is buggy for too many cases:
//no clue, likely strange error, hard to find. its an offset thing, likely with "partition"
  vec2 b=vec2(x,y);
  // b/=fraction-vec2(1);//some shit like that may fix it.
  //b.x-=1./partition;//something like that may fix this shortcut
  b*=iResolution.x;
  b=frame(b);
  vec4 i=c0(b);
  if(i.r<1.)return false;return true;//return if red in that pixel is 1.
/**/
}

// Function 44
float squircle(vec2 pos, float radius, float power)
{
  vec2 p = abs(pos - uv) / radius;
  float d = (pow(p.x,power) + pow(p.y, power) - pow(radius, power)) -1.0;
  return 1.0 - clamp (16.0*d, 0.0, 1.0);
}

// Function 45
vec2 UI_WindowGetTitleBarSize( UIContext uiContext, inout UIWindowState window )
{
    return vec2(window.drawRect.vSize.x - UIStyle_WindowBorderSize().x * 2.0, UIStyle_TitleBarHeight() );
}

// Function 46
void UI_ProcessWindowMain( inout UIContext uiContext, inout UIData uiData, int iControlId, int iData )
{
    UIWindowDesc desc;
    
    desc.initialRect = Rect( vec2(16, 16), vec2(380, 180) );
    desc.bStartMinimized = true;
    desc.uControlFlags = WINDOW_CONTROL_FLAG_TITLE_BAR | WINDOW_CONTROL_FLAG_MINIMIZE_BOX | WINDOW_CONTROL_FLAG_RESIZE_WIDGET;    
    desc.vMaxSize = vec2(380.0, 250.0);

    UIWindowState window = UI_ProcessWindowCommonBegin( uiContext, iControlId, iData, desc );
    
    if ( !window.bMinimized )
    {
        // Controls...
        Rect scrollbarPanelRect = Rect( vec2(0), vec2( 300.0 + UIStyle_ScrollBarSize(), uiContext.drawContext.vCanvasSize.y ) );

        vec2 vScrollbarCanvasSize = vec2(300, 200);

        UIPanelState scrollbarPanelState;            
        UI_ProcessScrollbarPanelBegin( uiContext, scrollbarPanelState, IDC_SCROLLBAR_CONTROLS_WINDOW, DATA_SCROLLBAR_CONTROLS_WINDOW, scrollbarPanelRect, vScrollbarCanvasSize );

        {        
            float tabX = 58.0f;
            //Scroll Panel Controls...
            UILayout uiLayout = UILayout_Reset();

            LayoutStyle style;
            RenderStyle renderStyle;             
            UIStyle_GetFontStyleWindowText( style, renderStyle );       

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                uint strA[] = uint[] ( _A, _n, _i, _m, _a, _t, _e );
                ARRAY_PRINT(state, style, strA);
                UI_RenderFont( uiContext, state, style, renderStyle );
                UILayout_SetControlRectFromText( uiLayout, state, style );
            }            
            UILayout_StackRight( uiLayout );
            
            UILayout_StackControlRect( uiLayout, UIStyle_CheckboxSize() );
            UI_ProcessCheckbox( uiContext, IDC_CHECKBOX_ANIMATE, uiData.animate, uiLayout.controlRect );
            UILayout_StackDown( uiLayout );

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                uint strA[] = uint[] ( _R, _o, _t, _X, _COLON );
                ARRAY_PRINT(state, style, strA);
                UI_RenderFont( uiContext, state, style, renderStyle );
                UILayout_SetControlRectFromText( uiLayout, state, style );
            }
            //UILayout_StackRight( uiLayout );
            UILayout_SetX( uiLayout, tabX );
            
            UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );                
            UI_ProcessSlider( uiContext, IDC_SLIDER_ROT_X, uiData.rotX, uiLayout.controlRect );
            UILayout_StackRight( uiLayout );

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                Print(state, style, uiData.rotX.fValue, 2 );

                UI_RenderFont( uiContext, state, style, renderStyle );

                UILayout_SetControlRectFromText( uiLayout, state, style );
            }

            UILayout_StackDown( uiLayout );    

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                uint strA[] = uint[] ( _R, _o, _t, _Y, _COLON );
                ARRAY_PRINT(state, style, strA);
                UI_RenderFont( uiContext, state, style, renderStyle );
                UILayout_SetControlRectFromText( uiLayout, state, style );
            }
            UILayout_SetX( uiLayout, tabX );
            //UILayout_StackRight( uiLayout );
            
            UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );                
            UI_ProcessSlider( uiContext, IDC_SLIDER_ROT_Y, uiData.rotY, uiLayout.controlRect );
            UILayout_StackRight( uiLayout );

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        

                Print(state, style, uiData.rotY.fValue, 2 );

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
                    DrawRect( uiContext.vPixelCanvasPos, uiLayout.controlRect, vec4(hsv2rgb(uiData.backgroundColor.vHSV), 1.0), uiContext.vWindowOutColor );
                }
            }

            bool buttonPressed = UI_ProcessButton( uiContext, IDC_BACKGROUND_COLOR_BUTTON, uiLayout.controlRect ); // Get button position from prev control

            if ( buttonPressed )
            {
                uiData.editWhichColor.fValue = 0.0;
            }                

            UILayout_StackRight( uiLayout );            
            
            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                uint strA[] = uint[] ( _B, _a, _c, _k, _g, _r, _o, _u, _n, _d, _SP, _C, _o, _l );
                ARRAY_PRINT(state, style, strA);
                UI_RenderFont( uiContext, state, style, renderStyle );
                UILayout_SetControlRectFromText( uiLayout, state, style );
            }            
            
            UILayout_StackDown( uiLayout );  

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                uint strA[] = uint[] ( _I, _n, _t, _COLON );
                ARRAY_PRINT(state, style, strA);
                UI_RenderFont( uiContext, state, style, renderStyle );
                UILayout_SetControlRectFromText( uiLayout, state, style );
            }
            UILayout_SetX( uiLayout, tabX );
            //UILayout_StackRight( uiLayout );
            
            UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );                
            UI_ProcessSlider( uiContext, IDC_SLIDER_INTENSITY, uiData.intensity, uiLayout.controlRect );
            UILayout_StackRight( uiLayout );

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        

                Print(state, style, uiData.intensity.fValue, 2 );

                UI_RenderFont( uiContext, state, style, renderStyle );

                UILayout_SetControlRectFromText( uiLayout, state, style );
            }

            UILayout_StackDown( uiLayout );  
 
            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                uint strA[] = uint[] ( _E, _x, _p, _COLON );
                ARRAY_PRINT(state, style, strA);
                UI_RenderFont( uiContext, state, style, renderStyle );
                UILayout_SetControlRectFromText( uiLayout, state, style );
            }
            UILayout_SetX( uiLayout, tabX );
            //UILayout_StackRight( uiLayout );
            
            UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );                
            UI_ProcessSlider( uiContext, IDC_SLIDER_EXPOSURE, uiData.exposure, uiLayout.controlRect );
            UILayout_StackRight( uiLayout );

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        

                Print(state, style, uiData.exposure.fValue, 2 );

                UI_RenderFont( uiContext, state, style, renderStyle );

                UILayout_SetControlRectFromText( uiLayout, state, style );
            }

            UILayout_StackDown( uiLayout );  
            
            #if 0
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
        
        UI_ProcessScrollbarPanelEnd(uiContext, scrollbarPanelState);
        
    }    
    
    UI_ProcessWindowCommonEnd( uiContext, window, iData );
}

// Function 47
float uiSlider(int id){return texture(iChannel0, vec2(float(id)+.5,0.5)/iResolution.xy).r;}

// Function 48
void sampleEquiAngular(
	Ray ray,
	float maxDistance,
	float Xi,
	vec3 lightPos,
	out float dist_to_sample,
	out float pdf)
{
	// get coord of closest point to light along (infinite) ray
	float delta = dot(lightPos - ray.origin, ray.dir);
	
	// get distance this point is from light
	float D = length(ray.origin + delta*ray.dir - lightPos);

	// get angle of endpoints
	float thetaA = atan(-delta, D);
	float thetaB = atan(maxDistance - delta, D);
	
	// take sample
	float t = D*tan(mix(thetaA, thetaB, Xi));
	dist_to_sample = delta + t;
	pdf = D/((thetaB - thetaA)*(D*D + t*t));
}

// Function 49
float poiseuille(float r, float R) { return (0.25 - (r/R)*(r/R)); }

// Function 50
vec3 material_builder(vec3 p, vec3 rd, float dis, int id) {
    if(id == 0) {        // Heart
		vec3 col = vec3(.5, 0.1, 0.1);
        return col;
    }
    else if(id == 1) {   // Box
        return vec3(1.);
    }
    else if(id == 2) {
    	return vec3(.1, .1, .5);
    }
}

// Function 51
void printUInt8(int num, ivec2 pos) {
	PIX(pos.x, pos.y)
	if(num< 10) {
		printNumber(num);
	} else {
		if(num>= 100) {
			printNumber(num< 200 ? 1 : 2);
			PIX(PIX_xy.x, pos.y)
		}
		printNumber(num/ 10- num/ 100* 10);
		PIX(PIX_xy.x, pos.y)
		printNumber(num- num/ 10* 10);
	}
}

// Function 52
float sdEquilateralTriangle( in vec2 p )
{
    const float k = sqrt(3.0);
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x+k*p.y>0.0 ) p = vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0, 0.0 );
    return -length(p)*sign(p.y);
}

// Function 53
vec3 material_builder(vec3 p, vec3 rd, float dis, int id) {
    if(id == ID_BOTTOM) {
        vec3 col = vec3(.3);
        float t = 5.;
		col = mix(col, vec3(0., 0.3, 0.), p.x > 0. ? smoothstep(0., 35., p.x) : 0.);
        col = mix(col, vec3(0.), smoothstep(.15, .0001, abs(sin(p.x / t ) )) );
        col = mix(col, vec3(0.), smoothstep(.15, .0001, abs(sin(p.z / t ) )) );

        return col;
    }
    else if(id == ID_SURFACE) {
    	vec3 col = vec3(.8, .8, 1.);
        return col;
    }
    else if(id == ID_SKY) {
    	vec3 col = vec3(.02, .1, .2);
        return col;
    }
}

// Function 54
define add_button(x,y,v0)     { nbB++; if (U==vec2(nbB+16,0.)) O = vec4(x,y,0,v0);       }

// Function 55
float eval_quintic_bezier(in float[6] control_points, float t) {
	float t2 = t * t;
	float t3 = t2 * t;
	float t4 = t3 * t;
	float t5 = t4 * t;
	
	float t_inv = 1.0 - t;
	float t_inv2 = t_inv * t_inv;
	float t_inv3 = t_inv2 * t_inv;
	float t_inv4 = t_inv3 * t_inv;
	float t_inv5 = t_inv4 * t_inv;
		
	return (
		control_points[0] *             t_inv5 +
		control_points[1] *  5.0 * t  * t_inv4 +
		control_points[2] * 10.0 * t2 * t_inv3 +
		control_points[3] * 10.0 * t3 * t_inv2 +
		control_points[4] *  5.0 * t4 * t_inv  +
		control_points[5] *        t5
	);
}

// Function 56
float aff_buttons(vec2 U) { // display buttons ( grey level or 0.)
    for (float i=0.; i<16.; i++) {
        if (i>=UI(0).y) break;
        vec4 S = UI(i+17.);
        float l = length(U-S.xy);
        if (l < Bradius) 
            if (S.a>0.) return 1.; 
            else return .3+smoothstep(.7,1.,l/Bradius);
    }
    return 0.;
}

// Function 57
vec3 plasma_quintic( float x )
{
	x = saturate( x );
	vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3(
		dot( x1.xyzw, vec4( +0.063861086, +1.992659096, -1.023901152, -0.490832805 ) ) + dot( x2.xy, vec2( +1.308442123, -0.914547012 ) ),
		dot( x1.xyzw, vec4( +0.049718590, -0.791144343, +2.892305078, +0.811726816 ) ) + dot( x2.xy, vec2( -4.686502417, +2.717794514 ) ),
		dot( x1.xyzw, vec4( +0.513275779, +1.580255060, -5.164414457, +4.559573646 ) ) + dot( x2.xy, vec2( -1.916810682, +0.570638854 ) ) );
}

// Function 58
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

// Function 59
float pUint4(vec2 p, float n)
{
    float v = 0.;
    for (int i = int(n), d = 4; d-- > 0; i /= 10, p.x += .5)
        v += char(p, 48 + i%10);
    return v;
}

// Function 60
bool RebuildBuffer(in vec2 fragCoord)
{
	return (iFrame == 0 ||                                                            // If first frame
            ivec2(texelFetch(iChannel0, ivec2(0), 0).rg) != ivec2(iResolution.xy) ||  // If resolution change
            iTime < TIME_NoiseOctaves ||                                              // If adjusting 2D heightmap
            ((iTime > TIME_2DPause) && (iTime < TIME_3DExtrude)) ||                   // If extruding mountains
            ((iTime > TIME_Water) && (iTime < TIME_Trees)));                          // If extruding trees
}

// Function 61
float distBuilding (in vec3 p, out float id) {

	// Take note of the ground coordinates
	vec2 ground = p.xz;

	// Change coordinates to cell space, and get the id of this building
	p.xz += 0.5;
	id = rand (floor (p.xz));
	if (fract (id * 31.0) > 0.7) {

		// Ground (empty cell)
		id = idGround (ground);
		return p.y;
	}
	p.xz = fract (p.xz) - 0.5;

	// Rotation
	float angle = id * PI * 0.5;
	float c = cos (angle);
	float s = sin (angle);
	p.xz = vec2 (c * p.x + s * p.z, c * p.z - s * p.x);

	// Translation
	angle = id * PI * 5.0;
	p.xz += 0.07 * vec2 (cos (angle), sin (angle));

	// Rounded box
	float boxHalfSize = 0.25 + 0.1 * cos (id * PI * 7.0);
	float boxHeight = 1.5 + id * 2.5;
	float boxRadius = boxHalfSize * (0.5 + 0.5 * cos (id * PI * 11.0));
	vec3 o = abs (p) - vec3 (boxHalfSize, boxHeight, boxHalfSize) + boxRadius;
	float dist = length (max (o, 0.0)) - boxRadius;

	// Carve this rounded box using other (signed) rounded boxes
	#ifdef HOLES
	#ifdef HOLLOW_THICKNESS_MIN
	float thickness = HOLLOW_THICKNESS_MIN + (HOLLOW_THICKNESS_MAX - HOLLOW_THICKNESS_MIN) * fract (id * 13.0);
	boxHalfSize -= thickness;
	boxHeight -= thickness;
	boxRadius = max (0.0, boxRadius - thickness);
	o = abs (p) - vec3 (boxHalfSize, boxHeight, boxHalfSize) + boxRadius;
	dist = max (dist, boxRadius - min (max (o.x, max (o.y, o.z)), 0.0) - length (max (o, 0.0)));
	boxHalfSize += thickness;
	#endif

	float boxPeriod = boxHalfSize * 0.3 * (0.8 + 0.2 * cos (id * PI * 13.0));
	boxHalfSize = boxPeriod * 0.45 * (0.9 + 0.1 * cos (id * PI * 17.0));
	boxRadius = boxHalfSize * (0.5 + 0.5 * cos (id * PI * 19.0));
	o = abs (mod (p, boxPeriod) - 0.5 * boxPeriod) - boxHalfSize + boxRadius;
	dist = max (dist, boxRadius - min (max (o.x, max (o.y, o.z)), 0.0) - length (max (o, 0.0)));
	#endif

	// Ground
	if (dist > p.y) {
		dist = p.y;
		id = idGround (ground);
	}
	return dist;
}

// Function 62
vec3 get_building_palette( vec2 h )
{
 vec3 ivory = vec3( 1, 0.85, 0.7 ); // c11
 vec3 white = vec3( 1., 1., 1. ); // c01
 vec3 c = mix( mix( vec3( 0.32, 0.38, 0.47 ), vec3( 0.35, 0.36, 0.41 ) * 0.5, h.x ), mix( white, ivory, h.x ), h.y );
//	vec3 c = mix( mix( c00, c10, h.x ), mix( c01, c11, h.x ), h.y );
 return c = mix( c, vec3( 0.6, 0.2, 0.2 ), smoothband( h.y - 0.5, 0.045, 0.01 ) * h.x * h.x * h.x ); // add rare reddish colors for occasional red tiles building
}

// Function 63
uvec3 asuint2(vec3 x) { return uvec3(asuint2(x.xy), asuint2(x.z)); }

// Function 64
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

// Function 65
UIData_Value UI_GetDataValue( int iData, float fDefaultValue, float fRangeMin, float fRangeMax, bool bInteger )  
{
    UIData_Value dataValue;
    
    vec4 vData0 = LoadVec4( iChannelUI, ivec2(iData,0) );
    
    if ( iFrame == 0 )
    {
        dataValue.fValue = fDefaultValue;
    }
    else
    {
        dataValue.fValue = vData0.x;
    }
    
    dataValue.fRangeMin = fRangeMin;
    dataValue.fRangeMax = fRangeMax;
    dataValue.bInteger = bInteger;
    
    return dataValue;
}

// Function 66
void gui_dfunc_update() {
    
    if (!(fc.x == DFUNC0_COL || fc.x == DFUNC1_COL)) { return; }
        
    bool is_linked = (load(MISC_COL, TARGET_ROW).x != 0.);

    for (int row=0; row<2; ++row) {  

        int col_for_row = (row == 0 ? DFUNC0_COL : DFUNC1_COL);

        for (int i=0; i<5; ++i) {

            bool update = ( (is_linked && fc.x == DFUNC1_COL) || 
                           (!is_linked && fc.x == col_for_row) );

            if (update) {

                if (box_dist(iMouse.xy, dfunc_ui_box(i, row)) < 0.) {
                    data = vec4(0);
                    if (i > 0) { data[i-1] = 1.; }
                }

            }
        }

    }

}

// Function 67
float text_nui(vec2 U) {
    initMsg;
    U.x+=4.*(0.5-0.2812*(res.x/0.5));
    C(110);C(111);C(32);C(85);C(73);
    endMsg;
}

// Function 68
vec4 BuildQuat(vec3 axis, float angle)
{
    angle *= 0.5;
    float s = sin(angle);
    return NormQuat(vec4(axis*s, cos(angle)));
}

// Function 69
vec4 fluidSolver(sampler2D velocityField, vec2 uv, vec2 stepSize, vec4 mouse, vec4 prevMouse)
{
    float k = .2, s = k/dt;
    
    vec4 fluidData = textureLod(velocityField, uv, 0.);
    vec4 fr = textureLod(velocityField, uv + vec2(stepSize.x, 0.), 0.);
    vec4 fl = textureLod(velocityField, uv - vec2(stepSize.x, 0.), 0.);
    vec4 ft = textureLod(velocityField, uv + vec2(0., stepSize.y), 0.);
    vec4 fd = textureLod(velocityField, uv - vec2(0., stepSize.y), 0.);
    
    vec3 ddx = (fr - fl).xyz * .5;
    vec3 ddy = (ft - fd).xyz * .5;
    float divergence = ddx.x + ddy.y;
    vec2 densityDiff = vec2(ddx.z, ddy.z);
    
    // Solving for density
    fluidData.z -= dt*dot(vec3(densityDiff, divergence), fluidData.xyz);
    
    // Solving for velocity
    vec2 laplacian = fr.xy + fl.xy + ft.xy + fd.xy - 4.*fluidData.xy;
    vec2 viscosityForce = viscosityThreshold * laplacian;
    
    // Semi-lagrangian advection
    vec2 densityInvariance = s * densityDiff;
    vec2 was = uv - dt*fluidData.xy*stepSize;
    fluidData.xyw = textureLod(velocityField, was, 0.).xyw;
    
    // Calc external force from mouse input
    vec2 extForce = vec2(0.);
    
    if (mouse.z > 1. && prevMouse.z > 1.)
    {
        vec2 dragDir = clamp((mouse.xy - prevMouse.xy) * stepSize * 600., -10., 10.);
        vec2 p = uv - mouse.xy*stepSize;
        //extForce.xy += .75*.0002/(dot(p, p)+.0001) * (.5 - uv);
        extForce.xy += .001/(dot(p, p)) * dragDir;
    }
    
    fluidData.xy += dt*(viscosityForce - densityInvariance + extForce);
    
    // velocity decay
    fluidData.xy = max(vec2(0.), abs(fluidData.xy) - 5e-6)*sign(fluidData.xy);
    
    // Vorticity confinement
	fluidData.w = (fd.x - ft.x + fr.y - fl.y); // curl stored in the w channel
    vec2 vorticity = vec2(abs(ft.w) - abs(fd.w), abs(fl.w) - abs(fr.w));
    vorticity *= vorticityThreshold/(length(vorticity) + 1e-5)*fluidData.w;
    fluidData.xy += vorticity;

    // Boundary conditions
    fluidData.y *= smoothstep(.5,.48,abs(uv.y - .5));
    fluidData.x *= smoothstep(.5,.49,abs(uv.x - .5));
    
    // density stability
    fluidData = clamp(fluidData, vec4(vec2(-velocityThreshold), 0.5 , -velocityThreshold), vec4(vec2(velocityThreshold), 3.0 , velocityThreshold));
    
    return fluidData;
}

// Function 70
vec4 DrawSpeedButton(vec2 uv, float thinkness)
{
    vec4 col;
    col.a = Segment(uv, vec2(-0.01,0), vec2(-0.05,0), thinkness);
    col.a += Segment(uv, vec2(0.01,0), vec2(0.05,0), thinkness);
    col.a += Segment(uv, vec2(0.03,0.02), vec2(0.03,-0.02), thinkness);
    col.a = clamp(col.a, 0.0, 1.0);
    col.rgb = vec3(col.a);
    return col;
}

// Function 71
float guitar3(float time, float frequency, float B) {
   float ret = 0.0;
    float energy = 1.0;
    for (int i = 1; i < 15; i++) {
//      float f = frequency * pow(float(i),1.002);
      float f = frequency * float(i) * sqrt(1.0 + float(i) * float(i) * B);
      float v =  pow(0.75, float(i)) *  exp((-1.5 - sqrt(frequency)/20.0) * time);
      if (i == 8 || i == 16 || i == 24) v /= 5.0;
	  float transfer = 1.0 - exp(-f/100.0);
      v *= energy * transfer;
      energy = 2.0 - energy * transfer;
        
      ret += si(f * time) * v;
    }
    return ret * 7.0 / sqrt(frequency);
}

// Function 72
bool Do_Button0(in bool isOpen)
{
    ivec2 p = ivec2(iMouse.xy);
 	
    GUI_Extent extent = Button0_Extent(isOpen);
    
    bool isHit = hitPointAABB(ivec2(extent.x0), ivec2(extent.x1), p);
    return isHit && isKeyReleased(keyPrimaryAction);
}

// Function 73
void UILayout_SetX( inout UILayout uiLayout, float xPos )
{
    uiLayout.vCursor.x = xPos;
    uiLayout.vControlMax.x = uiLayout.vCursor.x;
    uiLayout.vControlMin.x = uiLayout.vCursor.x;
}

// Function 74
vec2 	UIStyle_CheckboxSize() 			{ return vec2(16.0); }

// Function 75
void gui_decor_update() {
    
    if (fc.x != DECOR_COL) { return; }
    
    for (int i=0; i<4; ++i) {
        if (box_dist(iMouse.xy, decor_ui_box(i)) < 0.) {
            data[i] = 1. - data[i];
        }
    }
    
}

// Function 76
float Button( inout vec4 o, uint char,/*uint string[16], int strlen,*/ vec2 p, vec2 uv, bool click )
{
    int strlen = 1;
	float buttonFontScale = iResolution.y / screenHeightInLines;
    vec2 margin = vec2(.3,.0);
    vec2 buttonDim = (vec2(.5*float(strlen),1)+margin)*buttonFontScale*.5;
    
    vec2 buttonMid = abs(p)-sign(p)*buttonDim;
    
    float clickTime = texelFetch(iChannel0,ivec2(3+numButtons/4,0),0)[numButtons%4];
    if ( clickTime > 15. )
    {
        vec2 duv = abs(iMouse.xy-buttonMid)-buttonDim;
        if ( click && max(duv.x,duv.y) < .0 )
            clickTime = 0.;
    }
    clickTimes[numButtons] = clickTime; // for data output
    numButtons++;

    vec2 euv = abs(uv-buttonMid)-buttonDim;
    if ( max(euv.x,euv.y) <= .0 )
    {
        float pushed = clickTime < 5. ? -1. : 1.;
        float l = .8 + .2*pushed*sign(dot(uv-buttonMid,vec2(-1,1)/buttonDim))*step(-2.,max(euv.x,euv.y));
        o.rgb = vec3(l);
        
        vec2 stringUV = (uv-buttonMid+buttonDim-margin*buttonFontScale*.5)/buttonFontScale;
        if ( stringUV.x > .0 && stringUV.x < .5*float(strlen) && stringUV.y > .0 && stringUV.y < 1. )
        {
            o.rgb = mix( o.rgb, vec3(0),
                        PrintCharacterInternal( char/*string[int(stringUV.x/.5)]*/, vec2(fract(stringUV.x/.5)*.5,stringUV.y) )
                        );
        }
        
		o.a = 1.;
    }

    return clickTime;
}

// Function 77
float quintic(float t)			{ return t * t * t * (t * (t * 6. - 15.) + 10.); }

// Function 78
void UI_StoreDataValue( inout UIContext uiContext, UIData_Value dataValue, int iData )
{
    vec4 vData0 = vec4(0);
    vData0.x = dataValue.fValue;
    StoreVec4( ivec2(iData,0), vData0, uiContext.vOutData, ivec2(uiContext.vFragCoord) );            
}

// Function 79
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

// Function 80
vec3 UI_GetColor( int iData )
{
    return texelFetch( iChannelUI, ivec2(iData,0), 0 ).rgb;
}

// Function 81
vec4 color_ui_box(int idx) {
    
    return vec4(inset_ctr.x + (float(idx)-0.5)*text_size,
                dfunc_y - 3.5*text_size,
                vec2(0.45*text_size));
    
}

// Function 82
void paintUI(inout vec4 finalColor, in vec3 bgColor, in vec2 coord, in bool is_open)
{
    vec2 uv = coord / iResolution.xy;
    vec2 p = (2.*coord - iResolution.xy) / iResolution.y;

    float aspect = iResolution.x / iResolution.y;
    // Fake UI reflections kinda based on Windows 7 Aero
    vec2 uu = coord / iResolution.y;
    //uu.y = 1.0 - uu.y;
    float fakeReflX = mix(uu.x, uu.y, 0.3) - 3.2;
    float fakeReflT0 = abs(fract(fakeReflX) * 2.0 - 1.0) - 0.5;
    float fakeReflT1 = abs(fract(fakeReflX * 2.0 - 1.3) * 2.0 - 1.0) - 0.2;
    float fakeReflT2 = abs(fract(fakeReflX * 8.0 - 3.7) * 2.0 - 1.0) - 0.1;
    float fakeRefl = 0.2*smoothstep(64.0, 0.0, fakeReflT0 * iResolution.y) * pow(uv.y, 0.5)
                   + 0.4*smoothstep(32.0, 0.0, fakeReflT1 * iResolution.y/4.0) * pow(uv.y, 0.8)
                   + 0.4*smoothstep(24.0, 0.0, fakeReflT2 * iResolution.y/16.0 ) * pow(uv.y, 16.0);
    
    float t0 = sdBox(uu - vec2(0.1), vec2(0.07));//-(uv.y - 1.) - theme.menuSize;
    //float t1 = -(t0 + theme.menu_band_scale);
    
    float a0 = smoothstep(3., 0., t0*iResolution.y);
    
    float a = false ? 1.0 : a0;

    float w0 = SATURATE(exp(-2.5*abs(uv.x*2.-1.)));
    //float menu_light = a1 * SATURATE(exp(2. * t0/theme.menu_band_scale)) * w0;
    
    vec4 mainColor = UI_BACKGROUND_COLOR;//vec4(mix(UI_BACKGROUND_COLOR.rgb, bg_color, UI_BACKGROUND_COLOR.a) + 0.5 * fakeRefl, 1.0);
    //mainColor.rgb += 0.25 * fakeRefl;
    //Paint_Creeper(main_color, uv);

    
    bool isOpen = false;//bool(fract(iTime)*2.0-1.0);
    
    //paintMainButton(mainColor, uv, isOpen);
	
    if (isOpen)
    {
        paintSliderFloat(mainColor, uv, vec2(0, 1.0), vec2(0.4, 0.05), fract(iTime));
        
        //main_color = mix(main_color, theme.active_color, menu_light);

        finalColor.rgb = mix(bgColor, mainColor.rgb, mainColor.a);
    	//float ao0 = 1. - .2 * SATURATE(exp(-60.*t0)) * w0 * (1.-a0);
        //dest.rgb *= ao0;
        //dest.rgb = mix(dest.rgb, vec3(0), pow(w0, 0.8) * smoothstep(1.5 / iResolution.y, 0.0, abs(t0)));
        float s = 50.*64.;
        float e = s/iResolution.x;
        vec2 uv = coord * e;
        uv -= vec2(0.34, 0.20)*s;
        float alpha = 1.0-pow(glyph(uv, e)*0.8, 0.4545);
        //dest = vec4(vec3(),1.0);
        
        finalColor.rgb = mix(finalColor.rgb, vec3(0), alpha * 0.25);
        
    }
    else
    {
    	paintItemBar(finalColor, bgColor, mainColor, coord);
        
        //paintConfigButton(finalColor, bgColor, uu, UI_CONFIG_BUTTON);
    	//main_color = mix(main_color, theme.active_color, menu_light);
    	//float ao0 = 1. - SATURATE(exp(-0.95*t0 * iResolution.y));
        //finalColor.rgb *= ao0;
        
        //bgColor = mix(bgColor, mainColor.rgb, mainColor.a);
        
        //finalColor.rgb = mix(finalColor.rgb, bgColor, a);
        //dest.rgb = mix(dest.rgb, vec3(0), pow(w0, 0.8) * smoothstep(1.5 / iResolution.y, 0.0, abs(t0)));
    }
}

// Function 83
float sdSquircle(vec2 uv, vec2 origin, float radius, float power, float rot_)
{
    mat2 rot = rot2D(rot_);
    vec2 v = abs((origin*rot) - (uv*rot));
    float d = pow(v.x,power) + pow(v.y, power);
    d -= pow(radius, power);
    return d;
}

// Function 84
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

// Function 85
void setup_gui(vec2 res, float gui) {
    
    //bool show_gui = gui > 0.99 && res.y > 250.;
    if (res.y < 250.) { gui = 0.; }
    

    float inset_sz = 0.20*res.y;

    float margin_px = 6.0;

    text_size = 0.06 * res.y;

    inset_scl = 1.0 / inset_sz;
    inset_sz += margin_px;
    
    inset_ctr = vec2(mix(-inset_sz, inset_sz, gui), inset_sz);

    object_ctr = vec2(0.5*res.x + gui*inset_sz, 0.5*res.y);

    dfunc_y = res.y - text_size;
        


}

// Function 86
vec2 	UIStyle_ControlSpacing() 		{ return  vec2(6.0); }

// Function 87
float evalQuinticPrime(in float x, in GeneralQuintic q) {
    return (((5.0 * q.a * x + 4.0 * q.b) * x + 3.0 * q.c) * x + 2.0 * q.d) * x + q.e;
}

// Function 88
vec3 DrawUIBox(vec2 q, vec3 color)
{
    vec3 scoreCol = vec3(0.890, 0.588, 0.839);
    scoreCol = mix(scoreCol, scoreCol*1.5, smoothstep(-2., 2., q.x));
    float d = dsBox(q, vec2(1.0, 0.35), 0.2);
    color = mix(mix(scoreCol, color, 0.4), color, smoothstep(0., 0.01, d));
    vec3 frameColor = mix(vec3(0.917, 0.062, 0.768), vec3(0.917, 0.062, 0.768)*0.8, step(0.5, fract(Rot(q*2., -0.7).x)));
    frameColor += vec3(1.)*pow(max((1.-abs(q.x)), 0.), 2.5)*0.4;
    frameColor *= exp(-2.8*max(d+0.1, 0.));
    return mix(frameColor, color, smoothstep(0., 0.01, abs(d)-0.06 ));
}

// Function 89
vec4 building(vec3 p, vec2 wh, vec2 roof, float scale)
{
    p.y -= wh.y;
    p /= scale;
    
    vec3 ap = abs(p);
    vec3 q = ap - vec3(wh, wh.x);
    vec2 d = vec2(mv3(q), 0);
    
    p.xz = abs(p.xz);
    
    //roof
    float rf = MAX_DIST; roof.y /= wh.x;
    vec3 rp = p-vec3(0,wh.y+roof.y,0); rp.xz *= roof.y;
    if (roof.x == 0.0){
        rf = max(max(mv2(p.xz)-wh.x, mv2(rp.xz)+rp.y), -rp.y-roof.y)-wh.x/5.0;
    }else if(roof.x == 1.0){
        float rfn = abs(rp.x)+rp.y; rf = max(-rfn, rfn-wh.x/5.0);
        rf = max(rf, mv2(p.xz)-wh.x*1.25);
        rf = min(rf, max(rfn, mv2(p.xz)-wh.x));
    }
    d.x = min(rf, d.x);
    
    //windows
    vec2 wp = p.xy;
    float wl = 0.2, ww = 0.15;
    float v = pModInterval1(wp.x, ww*4.0, -floor(wh.x), floor(wh.x))+floor(wh.x);
    float c = pModInterval1(wp.y, 1.0, -floor(wh.y)+1., floor(wh.y)-1.)+floor(wh.y);
    float wd = min(length(wp.xy)-ww, max(abs(wp.x)-ww, abs(wp.y+wl)-wl));
    //wd = max(wd, abs(p.z-wh.x)-0.1);
    
    d = maxd(d, vec2(-wd, 2));
    
    return vec4(d, v, c);
}

// Function 90
vec3 Fluid(vec2 uv, float t) {
	float t1 = t*0.5;
	float t2 = t1 + 0.5;
	vec2 uv1 = calcNext(uv, t1);
	vec2 uv2 = calcNext(uv, t2);
	float c1 = getPattern(uv1);
	float c2 = getPattern(uv2);
	float c=mix(c2,c1,t);
    float f=1.5-0.5*abs(t-0.5);
	c=pow(c,f)*f;//correcting the contrast/brightness when sliding
	float h=mix(length(uv-uv2),length(uv-uv1),t);
	return 2.0*c*heatmap(clamp(h*0.5,0.0,1.0));//blue means slow, red = fast
}

// Function 91
float pUint(vec2 p, float n)
{
    float v = 0.;
    for (int i = int(n); i > 0; i /= 10, p.x += .5)
        v += char(p, 48 + i%10);
    if (abs(n) < 1e-7)
        v += char(p, 48);
    return v;
}

// Function 92
void UI_PanelBegin( inout UIContext uiContext, inout UIPanelState panelState )
{
    panelState.parentDrawContext = uiContext.drawContext;
    panelState.vParentWindowColor = uiContext.vWindowOutColor;
}

// Function 93
vec4 button(vec4 fragColor, float buttonID) {
    vec4 buttonState = BufA(1.0, 4.0);
    vec3 color = vec3(0.0, 0.15, 0.3);
    if (buttonState.r == buttonID) {
	    if (buttonState.g == buttonID)
            color = vec3(0.0, 0.3, 0.6);
        else
            color = vec3(0.2, 0.25, 0.4);
    }
    return 1.0 - (1.0 - vec4(color, 0.0)) * (1.0 - fragColor);
}

// Function 94
void buildMatSpace(KeyFrame frame, out MatSpace res)
{
    res.moustacheBend = frame.moustacheBend;
    res.twistLower = -0.4 + frame.mouthOpenVert * 0.5;
    res.twistX = frame.twistX;
    res.bendX = frame.bendX;
    res.moustacheBend = frame.moustacheBend;

    res.eyeRad = vec3(0.21, 0.37 * frame.eyeOpening, 0.20) * 0.5;
    res.cheekPos = vec3(0.2 + frame.mouthOpenHoriz * 0.5, -0.014 - frame.mouthOpenVert, -0.19);
    res.cheekRad = vec3(0.51, 0.55 + frame.mouthOpenVert * 0.5, 0.57) * 0.5;
    res.chinPos = vec3(0.0, -0.26 - frame.mouthOpenVert * 0.6, -0.22 - frame.mouthOpenVert * 1.0);
    res.noseRad = vec3(0.42 + frame.mouthOpenHoriz, 0.39 - frame.mouthOpenHoriz * 0.5, 0.41) * 0.5;

    res.mouthPos = vec3(0.0, -0.13 - frame.mouthOpenVert * 1.8, -0.41);
    res.mouthRad = vec3(0.32, 0.16 + frame.mouthOpenVert, 0.31) * 0.5;

    res.lipPos = vec3(0.0, -0.06 - frame.mouthOpenVert * 3.0, -0.36 - frame.mouthOpenVert * 1.5);
    res.lipStretchX = 1.0f - frame.mouthOpenHoriz * 7.0;
    res.lipThickness = 0.1 - frame.mouthOpenVert;

    res.earMat = rotationX3(-65.0 * degToRad + frame.mouthOpenHoriz);
    res.cap1Mat = rotationX3(30.0 * degToRad);
    res.cap2Mat = rotationX3(60.0 * degToRad);

    res.hairTip1 = vec3(0.45 - frame.mouthOpenVert * 0.6,0.06,-0.23);
    res.hairTip2 = vec3(0.42 - frame.mouthOpenVert * 0.25,0.19,-0.28);

    res.browOffset = (frame.eyeOpening - 1.0) * 0.15;
    res.browBend = frame.browBend;

    res.teethPos = vec3(0.0, -0.1 - frame.mouthOpenVert * 0.5, -0.2  - frame.mouthOpenVert);

    res.eyePos = frame.eyePos;
    res.eyelidsOpen = frame.eyelidsOpen;

}

// Function 95
void sampleEquiAngular(
	Ray ray,
	float maxDistance,
	float Xi,
	vec3 lightPos,
	out float dist,
	out float pdf)
{
	// get coord of closest point to light along (infinite) ray
	float delta = dot(lightPos - ray.origin, ray.dir);
	
	// get distance this point is from light
	float D = length(ray.origin + delta*ray.dir - lightPos);

	// get angle of endpoints
	float thetaA = atan(0.0 - delta, D);
	float thetaB = atan(maxDistance - delta, D);
	
	// take sample
	float t = D*tan(mix(thetaA, thetaB, Xi));
	dist = delta + t;
	pdf = D/((thetaB - thetaA)*(D*D + t*t));
}

// Function 96
float suit(vec3 p)
{
    vec3 su = p+vec3(0,80,3);
    su.y += max(abs(p.x*.25)-2.5, 0.0);
    float d = rBox(su, vec3(24,60.,2.2), 4.5);
    d = max(d, -cylinder(p+vec3(0,15,2), vec2(9.5,16)));
    return d;
}

// Function 97
vec3 shadeBuilding(Hit scn, vec3 n, vec3 r) {
 
    vec3 c = vec3(1.);
    return calcLighting(c, scn.p, n, r, 0.);
    
}

// Function 98
vec2 buttonDownPos(){//this u is framed differently!
 if(iMouse.w<0.)return NotPushed;//error code for no mouse button down;
 vec2 b=iMouse.xy;
 return buttonPos(b);}

// Function 99
void UI_WriteCanvasUV( inout UIContext uiContext, int iControlId )        
{
	if (!uiContext.bPixelInView)
        return;
    Rect rect = Rect( vec2(0), uiContext.drawContext.vCanvasSize );
    DrawRect( uiContext.vPixelCanvasPos, rect, vec4(uiContext.vPixelCanvasPos / uiContext.drawContext.vCanvasSize, float(iControlId), -1.0 ), uiContext.vWindowOutColor );
}

// Function 100
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

// Function 101
vec2 	UIStyle_WindowBorderSize() 		{ return vec2(4.0); }

// Function 102
Roots5 solveQuinticBracket(in GeneralQuintic eq, in int bisections, in int newtonSteps) {
    // Solve for the roots of the derivative (a quartic) to get the critical points
    vec4 brackets;
    Roots5 roots = Roots5(0, Float5(0.0, 0.0, 0.0, 0.0, 0.0));
    int nBrackets = solveQuartic(5.0 * eq.a, 4.0 * eq.b, 3.0 * eq.c, 2.0 * eq.d, eq.e, brackets);

    // No brackets, failed to find roots
    // TODO: these cases are pretty simple quintics, look for another root finding solution
    if (nBrackets == 0) return Roots5(0, Float5(0.0, 0.0, 0.0, 0.0, 0.0));

    // Search for roots between critical points
    for (int n=0; n < nBrackets - 1; n++) {
        float brack1 = brackets[n], brack2 = brackets[n + 1];
        float e1 = evalQuintic(brack1, eq), e2 = evalQuintic(brack2, eq);

        // Bracketed section doesn't cross the x axis, no roots
        if (!(min(e1, e2) < 0.0 && max(e1, e2) > 0.0)) return Roots5(0, Float5(0.0, 0.0, 0.0, 0.0, 0.0));

        // Apply bisection, then newton-raphson
        float bmin = min(brack1, brack2), bmax = max(brack1, brack2);
        set(roots.roots, roots.nroots, newton(eq, bisection(eq, bmin, bmax, bisections), bmin, bmax, newtonSteps));
        roots.nroots++;
    }

    // Flip inner bracket on one side to the outside to try capturing roots outside the critical points
    // Not sure if this is foolproof but it appears to work and make sense
    float brack1 = 2.0 * brackets[0] - brackets[1], brack2 = brackets[0];
    float e1 = evalQuintic(brack1, eq), e2 = evalQuintic(brack2, eq);
    if (!(min(e1, e2) < 0.0 && max(e1, e2) > 0.0)) return Roots5(0, Float5(0.0, 0.0, 0.0, 0.0, 0.0));
    float bmin = min(brack1, brack2), bmax = max(brack1, brack2);
    set(roots.roots, roots.nroots, newton(eq, bisection(eq, bmin, bmax, bisections), bmin, bmax, newtonSteps));
    roots.nroots++;

    // Repeat on opposite side
    brack1 = brackets[nBrackets - 1], brack2 = 2.0 * brackets[nBrackets - 1] - brackets[nBrackets - 2];
    e1 = evalQuintic(brack1, eq), e2 = evalQuintic(brack2, eq);
    if (!(min(e1, e2) < 0.0 && max(e1, e2) > 0.0)) return Roots5(0, Float5(0.0, 0.0, 0.0, 0.0, 0.0));
    bmin = min(brack1, brack2), bmax = max(brack1, brack2);
    set(roots.roots, roots.nroots, newton(eq, bisection(eq, bmin, bmax, bisections), bmin, bmax, newtonSteps));
    roots.nroots++;

    return roots;
}

// Function 103
uint HashUInt(vec3  v, uvec3 r) { return Hash(floatBitsToUint(v), r); }

// Function 104
UIData UI_GetControlData()
{
    UIData data;
    
    data.backgroundImage = UI_GetDataBool( DATA_BACKGROUND_IMAGE, false );
    data.showImageWindow = UI_GetDataBool( DATA_CHECKBOX_SHOW_IMAGE, true );
    data.buttonA = UI_GetDataBool( DATA_BUTTONA, false );
    
    data.backgroundBrightness = UI_GetDataValue( DATA_BACKGROUND_BRIGHTNESS, 0.5, 0.0, 1.0 );
    data.backgroundScale = UI_GetDataValue( DATA_BACKGROUND_SCALE, 10.0, 1.0, 10.0 );
    data.imageBrightness = UI_GetDataValue( DATA_IMAGE_BRIGHTNESS, 1.0, 0.0, 1.0 );
    
    data.editWhichColor = UI_GetDataValue( DATA_EDIT_WHICH_COLOR, -1.0, -1.0, 100.0 );
    data.bgColor = UI_GetDataColor( DATA_BG_COLOR, vec3(0, 0.5, 0.5) );
    data.imgColor = UI_GetDataColor( DATA_IMAGE_COLOR, vec3(1.0, 1.0, 1.0) );
    
    return data;
}

// Function 105
void displayUI(inout vec4 fragColor, in vec2 fragCoord)
{
	vec2 coord = fragCoord.xy;
	vec4 ui = load(coord-0.5,iChannel1);
	float brush = load(vec2(9,0),iChannel1).x;
	
	// Hide pixels used for settings
	if (coord.y == 0.5) { coord.y += 1.0; }
	// Hide color picker & band if paint tool is not selected
	if (brush != 0.4 && fragCoord.x > iResolution.x/2.) { ui.a = -1.; }

	if (_key_u < 1.0 && ui.a > 0.0)
	{
		fragColor.rgb = fragColor.rgb*(1.-ui.a) + ui.rgb*ui.a;
	}
}

// Function 106
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

// Function 107
void UI_DrawColorPickerSV( inout UIContext uiContext, bool bActive, vec3 vHSV, Rect pickerRect )
{
	if (!uiContext.bPixelInView || Outside( uiContext.vPixelCanvasPos, pickerRect ))
        return;
    
    vec2 vCurrPixelPos = (uiContext.vPixelCanvasPos - pickerRect.vPos) / pickerRect.vSize;
    vCurrPixelPos.y = 1.0f - vCurrPixelPos.y;
    vec3 vHSVCurr = vHSV;
    vHSVCurr.yz = vCurrPixelPos;

    uiContext.vWindowOutColor = vec4( hsv2rgb( vHSVCurr ), 1.0 );
    
    vec2 vSelectedPos = vHSV.yz;
    vSelectedPos.y = 1.0f - vSelectedPos.y;
    vSelectedPos = vSelectedPos * pickerRect.vSize + pickerRect.vPos;
        
    float l = length( vSelectedPos - uiContext.vPixelCanvasPos );
    float d = l - 3.0;
    d = min(d, 5.0 - l);
    if ( bActive )
    {
        float d2 = l - 5.0;
    	d2 = min(d2, 7.0 - l);
	    d = max(d, d2);
    }
    
    float fBlend = clamp(d, 0.0, 1.0);
    
    uiContext.vWindowOutColor.rgb = mix(uiContext.vWindowOutColor.rgb, vec3(1.0) - uiContext.vWindowOutColor.rgb, fBlend);
}

// Function 108
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

// Function 109
PrintState UI_PrintState_Init( inout UIContext uiContext, LayoutStyle style, vec2 vPosition )
{
    vec2 vCanvasPos = uiContext.vPixelCanvasPos;
    
    PrintState state = PrintState_InitCanvas( vCanvasPos, vec2(1.0) );
    MoveTo( state, vPosition + UIStyle_FontPadding() );
	PrintBeginNextLine(state, style);

	return state;
}

// Function 110
vec3 	UIStyle_ColorPickerSize()		{ return vec3(192.0, 192.0, 32.0); }

// Function 111
void UI_DrawSliderX( inout UIContext uiContext, bool bActive, bool bMouseOver, float fPosition, Rect sliderRect, float fHandleSize, bool scrollbarStyle )
{
	if (!uiContext.bPixelInView || Outside( uiContext.vPixelCanvasPos, sliderRect ))
        return;
    
    Rect horizLineRect;
    
    horizLineRect = sliderRect;
    if (!scrollbarStyle)
    {
	    float fMid = sliderRect.vPos.y + sliderRect.vSize.y * 0.5;
    	horizLineRect.vPos.y = fMid - 2.0;
    	horizLineRect.vSize.y = 4.0;
    }

    DrawBorderIndent( uiContext.vPixelCanvasPos, horizLineRect, uiContext.vWindowOutColor );

    float fSlideMin = sliderRect.vPos.x + fHandleSize * 0.5f;
    float fSlideMax = sliderRect.vPos.x + sliderRect.vSize.x - fHandleSize * 0.5f;

    float fDistSlider = (fSlideMin + (fSlideMax-fSlideMin) * fPosition);

    Rect handleRect;

    handleRect = sliderRect;
    handleRect.vPos.x = fDistSlider - fHandleSize * 0.5f;
    handleRect.vSize.x = fHandleSize;

    vec4 handleColor = vec4(0.75, 0.75, 0.75, 1.0);
    if ( bActive )
    {
        handleColor.rgb += 0.1;
    }

    DrawRect( uiContext.vPixelCanvasPos, handleRect, handleColor, uiContext.vWindowOutColor );
    DrawBorder( uiContext.vPixelCanvasPos, handleRect, uiContext.vWindowOutColor );
}

// Function 112
DistSample SdfPenguin(vec3 p)
{
    DistSample result = SdfBox(p, vec3(0.75), kMaterialNone);
    
    if (result.dist < 0.5)
    {
        if (gGameState == kStateGameOver && gPlayerDeathCause == kBehavHazard)
        {
            DistSample result = SdfBox(Tx(p, vec3(0, 0.047, -0.082)), vec3(0.2799999, 0.04, 0.1204678), kMaterialPenguinBlackFeathers);
            result = OpU(result, SdfBox(Tx(p, vec3(0, 0.107, 0.038)), vec3(0.28, 0.02, 0.24), kMaterialPenguinBlackFeathers));
            result = OpU(result, SdfBox(Tx(p, vec3(0, 0.047, 0.158)), vec3(0.28, 0.04, 0.12), kMaterialPenguinWhiteFeathers));
            result = OpU(result, SdfBox(Tx(p, vec3(0, 0.031, -0.282)), vec3(0.28, 0.024, 0.08000001), kMaterialPenguinBlackFeathers));
            result = OpU(result, SdfBox(Tx(p, vec3(0, 0.095, 0.29)), vec3(0.12, 0.008, 0.08000001), kMaterialPenguinBeak));
            result = OpU(result, SdfBox(Tx(p, vec3(-0.152, -0.001, -0.022)), vec3(0.048, 0.01, 0.06000001), kMaterialPenguinBeak));
            result = OpU(result, SdfBox(Tx(p, vec3(0.152, -0.001, -0.022)), vec3(0.048, 0.01, 0.06000001), kMaterialPenguinBeak));
            result = OpU(result, SdfBox(Tx(p, vec3(0.152, -0.009, 0.01)), vec3(0.112, 0.004, 0.16), kMaterialPenguinBeak));
            result = OpU(result, SdfBox(Tx(p, vec3(-0.152, -0.009, 0.01)), vec3(0.112, 0.004, 0.16), kMaterialPenguinBeak));
            result = OpU(result, SdfBox(Tx(p, vec3(0.136, 0.111, 0.29)), vec3(0.032, 0.006000001, 0.012), kMaterialPenguinEyes));
            result = OpU(result, SdfBox(Tx(p, vec3(-0.136, 0.111, 0.29)), vec3(0.032, 0.006000001, 0.012), kMaterialPenguinEyes));
            result = OpU(result, SdfBox(Tx(p, vec3(0.323, 0.037, -0.002), vec4(0, 0, 0.1736482, 0.9848078)), vec3(0.05, 0.028, 0.12), kMaterialPenguinBlackFeathers));
            result = OpU(result, SdfBox(Tx(p, vec3(-0.342, 0.045, -0.002), vec4(0, 0, -0.1736482, 0.9848078)), vec3(0.04999999, 0.028, 0.12), kMaterialPenguinBlackFeathers));
            return result;

        }
        else
        {
            result = SdfBox(Tx(p, vec3(0, 0.296, -0.0855)), vec3(0.14, 0.2, 0.06023391), kMaterialPenguinBlackFeathers);
            result = OpU(result, SdfBox(Tx(p, vec3(0, 0.596, -0.0255)), vec3(0.14, 0.1, 0.12), kMaterialPenguinBlackFeathers));
            result = OpU(result, SdfBox(Tx(p, vec3(0, 0.296, 0.0345)), vec3(0.14, 0.2, 0.06000001), kMaterialPenguinWhiteFeathers));
            result = OpU(result, SdfBox(Tx(p, vec3(0, 0.216, -0.1855)), vec3(0.14, 0.12, 0.04), kMaterialPenguinBlackFeathers));
            result = OpU(result, SdfBox(Tx(p, vec3(0, 0.536, 0.1005)), vec3(0.06, 0.04, 0.04), kMaterialPenguinBeak));
            result = OpU(result, SdfBox(Tx(p, vec3(-0.076, 0.056, -0.0555)), vec3(0.024, 0.05, 0.03), kMaterialPenguinBeak));
            result = OpU(result, SdfBox(Tx(p, vec3(0.076, 0.056, -0.0555)), vec3(0.024, 0.05, 0.03), kMaterialPenguinBeak));
            result = OpU(result, SdfBox(Tx(p, vec3(0.076, 0.016, -0.0395)), vec3(0.056, 0.02, 0.08000001), kMaterialPenguinBeak));
            result = OpU(result, SdfBox(Tx(p, vec3(-0.076, 0.016, -0.0395)), vec3(0.056, 0.02, 0.08000001), kMaterialPenguinBeak));
            result = OpU(result, SdfBox(Tx(p, vec3(0.068, 0.616, 0.1005)), vec3(0.016, 0.03, 0.006000001), kMaterialPenguinEyes));
            result = OpU(result, SdfBox(Tx(p, vec3(-0.068, 0.616, 0.1005)), vec3(0.016, 0.03, 0.006000001), kMaterialPenguinEyes));
            result = OpU(result, SdfBox(Tx(p, vec3(0.18, 0.316, -0.0455), vec4(0, 0, 0.1736482, 0.9848078)), vec3(0.025, 0.14, 0.06000001), kMaterialPenguinBlackFeathers));
            result = OpU(result, SdfBox(Tx(p, vec3(-0.18, 0.316, -0.0455), vec4(0, 0, -0.1736482, 0.9848078)), vec3(0.025, 0.14, 0.06000001), kMaterialPenguinBlackFeathers));
        }
    }
    return result;
}

// Function 113
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

// Function 114
float ruins(vec3 p) {
    vec3 q=p;
    
    //bumps
    float o= texture(iChannel0, p.xy*0.1 ).x*texture(iChannel0, p.yz*0.1 ).x*0.005;

    //pillars bottom
    p.x=clamp(p.x,-8.0,8.0);                                     //limit x
    if ((p.z<2.0 && p.z>-2.0) && (p.x<0.0 && p.x>-4.0)) p.x=0.0; //chop hole in middle 

    p.x=mod(p.x,2.0)-0.5*2.0;                                    //rep x

    p.z=clamp(p.z,-4.0,4.0);                                     //limit z
    if (q.x>2.0 || q.x<-6.0) p.z=clamp(p.z,-2.0,2.0);
    p.z=mod(p.z,2.0)-0.5*2.0;                                    //rep z

    float r= 0.5-clamp( sin(p.y*1.2+1.58)*0.5, 0.0,0.05);
    float d=sdCylinder(p, vec2(r,1.5)) -o;

    //pillars top
    p.y-=2.8;
    r= 0.4-clamp( sin(p.y*1.8+0.8)*0.5, 0.0,0.05);
    
    float h=1.5;
    if (q.x>2.0) { p.y-=0.5; h=2.0; r=0.4-clamp( sin(p.y*1.15+1.1)*0.5, 0.0,0.05); } //pull first 3x2 pillars up
    float t=sdCylinder(p, vec2(r,h)) -o;    

    //mid platform
    q.y-=1.8;
    float c=sdrBox(q, vec3(7.45,0.25,1.45), 0.05) -o;
    q.x+=2.0;
    c=min(c, sdrBox(q, vec3(3.45,0.25,3.45), 0.05) -o);

    //bottom platform
    q.y+=3.55;
    c=min(c, sdrBox(q, vec3(3.65,0.2,3.65), 0.05) -o);
    q.x-=2.0;
    c=min(c, sdrBox(q, vec3(7.65,0.2,1.65), 0.05) -o);
    
    //ground platform
    q.x+=2.0;
    q.y+=0.8;
    c=min(c, sdrBox(q, vec3(4.65,0.6,4.65), 0.05) -o);
    q.x-=2.0;
    c=min(c, sdrBox(q, vec3(8.65,0.6,2.65), 0.05) -o);
    
    //top part
    q.y-=8.0;
    q.x-=5.0;
    c=min(c, sdrBox(q, vec3(2.45,0.15,1.45), 0.05) -o);
    c=max(c, -sdrBox(q, vec3(1.45,0.25,0.45), 0.05) -o);    //left hole
    
    //top right part
    q.y+=1.0;
    q.x+=6.0;
    c=min(c, sdrBox(q, vec3(4.50,0.15,1.65), 0.05) -o);
    q.x+=1.0;
    q.z-=0.85;
    c=min(c, sdrBox(q, vec3(3.45,0.15,2.5), 0.05) -o);
    q.z+=0.85;
c=max(c, -sdrBox(q, vec3(2.25,4.5,2.25), 0.05) -o);    
    
    d=min(min(c,d),t);
	
    return d;
}

// Function 115
vec4 PaintButtons(vec2 u,vec2 In,vec2 m){
 bool but=(iMouse.w>0.);
 //Out=c0(In);return;//copy PiP-overlay from "BufA" to "Image"
 //above code just copies BuffA.PiP into Image, lazy;
 //below code has the basics to paint over that with custom tiles.
    
 //in here, treat a button as if it is a monitor, aspect datio set by gridscale*partition
 vec2 s=fract(u)-.5;//center of button is vec2(0)
      
 //u.x*=(gridscale*partition).y/(gridscale*partition).x;
 //u.x*=iResolution.x/iResolution.y;
 //above 2 lines would adjust aspect ratios to rations of screen AND button
        
 //d is distance to center of a button, with modified gratient.
 float d=length(s);//distance to center of a button
 //d*=d;
 d=1.-d;
 //d=fract(d*2.);//ugly high contrast
 const float sharpness=.4;//range [0.. .55]
 d=smoothstep(sharpness,1.-sharpness,d);
      
 vec2 pos=buttonPos(In);
 //pos can be used to index the buttons, to paind them differently.
 //insert here hwow a button looks, depending on its position# .
 vec3 c=vec3(0);
 c.yz=d*(fract(pos*.2)*.8+.2);
 if(InButton(m)&&(but)){
  //insert here how a button changes while pushed.
  c.x=1.;//maximize redness;  cheapest highlight ever.
 }return vec4(c,1.);}

// Function 116
Rect UI_GetFontRect( PrintState state, LayoutStyle style )
{
    Rect rect;
    rect = GetFontRect( state, style, true );
    vec2 vExpand = UIStyle_FontPadding();
    vExpand.y += style.vSize.y * style.fLineGap;
    RectExpand( rect, vExpand );
	return rect;
}

// Function 117
float sdEquilateralTriangle(  in vec2 p )
{
    const float k = 1.73205;//sqrt(3.0);
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x + k*p.y > 0.0 ) p = vec2( p.x - k*p.y, -k*p.x - p.y )/2.0;
    p.x += 2.0 - 2.0*clamp( (p.x+2.0)/2.0, 0.0, 1.0 );
    return -length(p)*sign(p.y);
}

// Function 118
vec3 BuildAlbedo(int id)
{
    return Albedos[clamp(id, 0, 3)];
}

// Function 119
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

// Function 120
float buildings( vec3 p, float t )
{
    p.y = abs(p.y)+3.;
    p.y -= max(10.*round(p.y/10.), 10.);
    
    // It's actually the buildings that are moving!
    // (That's all relative, of course.)
    p.x = mod(p.x + 1.0*t, 10.0) - 5.;
    return sdBox(p, vec3(3.,2.5,4.))-0.1;
}

// Function 121
vec3 fluid(vec3 uv1,float iters){
 //fake fluid physics
 vec3 uv = uv1;
 for (float i = 1.; i < iters; i++)
  {
    uv.x += sin((iTime-uv.y)*.5)*1.5/i* sin(i * uv.y + iTime * 0.5);
    uv.y += sin((iTime-uv.z)*.5)*1.5/i* sin(i * uv.z + iTime * 0.5 );
    uv.z += sin((iTime-uv.x)*.5)*1.5/i* sin(i * uv.x + iTime * 0.5 );
  }
  return uv;
}

// Function 122
vec4 guistate(in int s) 
{ return texture(iChannel0, (vec2(float(s),0.)+.5)/iChannelResolution[0].xy); }

// Function 123
void UI_DrawCheckbox( inout UIContext uiContext, bool bActive, bool bMouseOver, bool bChecked, Rect checkBoxRect )
{
	if (!uiContext.bPixelInView || Outside( uiContext.vPixelCanvasPos, checkBoxRect ))
        return;
    
    uiContext.vWindowOutColor = vec4(1.0);
    
    if ( bActive && bMouseOver )
    {
        uiContext.vWindowOutColor = vec4(0.85,0.85,0.85,1.0);
    }

#ifdef NEW_THEME
    DrawBorderRect( uiContext.vPixelCanvasPos, checkBoxRect, cCheckboxOutline, uiContext.vWindowOutColor );
#else    
    DrawBorderIndent( uiContext.vPixelCanvasPos, checkBoxRect, uiContext.vWindowOutColor );
#endif    

    Rect smallerRect = checkBoxRect;
    RectShrink( smallerRect, vec2(6.0));

    if ( bChecked )
    {
        vec4 vCheckColor = vec4(0.0, 0.0, 0.0, 1.0);
        DrawLine( uiContext.vPixelCanvasPos, smallerRect.vPos+ smallerRect.vSize * vec2(0.0, 0.75), smallerRect.vPos+ smallerRect.vSize * vec2(0.25, 1.0), 2.0f, vCheckColor, uiContext.vWindowOutColor );
        DrawLine( uiContext.vPixelCanvasPos, smallerRect.vPos+ smallerRect.vSize * vec2(0.25, 1.0), smallerRect.vPos+ smallerRect.vSize * vec2(1.0, 0.25), 2.0f, vCheckColor, uiContext.vWindowOutColor );
    }
}

// Function 124
vec4 MapBuilding(vec3 p)
{
    vec2 cellId = floor(p.xz);
    vec3 cube = GetCube(cellId);
    
    vec3 buildingPos = vec3(cellId.x + 0.5, cube.y, cellId.y + 0.5);
    float d = dBuilding( p - buildingPos, cube);
    
    return vec4(d, p.xyz);
}

// Function 125
vec4 mainImageUI2AD37(out vec4 o, in vec2 u
){o=vec4(0)
 #ifdef Scene2D
  ;o=pdOver(iDiegeticUIshow(u),o)//ui dots
  ;o=pdOver(o,iCB(o,u))//iCB(o,u)
  ;o=pdOver(o,ltj3Wc(o,u,iResolution,iMouse))//backsrop is a 2d srawing
 #else
  #ifdef SceneTR
   ;o=pdOver(iDiegeticUIshow(u),o)//ui dots
   ;o=pdOver(o,iCB(o,u))//bezier+appolonean stuff
   ;o=pdOver(o,mTR(o,u)) //backfrop is traced 3d scene (TemporalReprojection+brdf)
  #else
   ;o=pdOver(iDiegeticUIshow(u),o)//ui dots
   ;o=pdOver(o,iCB(o,u))//bezier+appolonean stuff
   ;o=pDoOver(iAD)  //backfrop is marched 3d scene (automatic Differentiation)
  #endif
 #endif
 ;return o      /**/
 ;}

// Function 126
float Build(vec2 uv, Building b, vec3 windowColor,
    inout vec3 color)
{
    float scene = sdBox(uv + b.pos, b.scale);
    color = DoOutline(scene, b.color);
    ApplyWindow(scene, uv, b, windowColor, color);
    return scene;
}

// Function 127
void CreateUI(inout vec3 currentColor,in vec2 uv){
    for(int i=0;i<FretCount;i++){
		CreateFret(i,currentColor,uv);
        //DrawNumber(1.,5.,vec2(-0.1,-.45),currentColor,uv);
    }
}

// Function 128
float sdEquilateralTriangle(  vec2 p, float s ){
    const float k = sqrt(3.0);
    p.x = abs(p.x) - s;
    p.y = p.y + s/k;
    if( p.x+k*p.y>0.0 ) p = vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0, 0.0 );
    float d = -length(p)*sign(p.y); 
    d = abs(d) - 0.0025;
    return d;
}

// Function 129
float evalReducedQuintic(in float x, in ReducedQuintic q) {
    return (((x * x + q.p) * x + q.q) * x + q.r) * x + q.s;
}

// Function 130
void drawUI(inout vec3 color, vec2 p, AppState s)
{
    p *= R.y / R.x; // ratio and resolution indepenent scaling
    p *= 1.75;
    
    // splash screen   
    if ( s.stateID == GS_SPLASH )
    {
        color.rgb *= 0.1 + 0.9 * smoothstep( 0.75, 0.0, p.y ); // dark text bg
		vec2 p2 = p;
		p2 *= 50.;
		p2 -= vec2( -45, 27. );
        // color.rgb = mix(color.rgb, vec3(0.0), 1.0-smoothstep(0.0, 0.5, abs(p2.y)) ); // horiz guide
        
        float maskTitle = titleText( p2 ); // Sunset Drive Unlimited
        color.rgb = mix( vec3( 1.0 ), color.rgb, maskTitle );
        
		vec2 p1 = p;
		p1 *= 60. + 5. * abs( sin( 2.0 * iTime ) );
		p1 -= vec2( -47., -42. );
        float maskSpace = spaceText( p1 ); // press [space] to start
        color.rgb = mix( vec3( 1.0 ), color.rgb, maskSpace );

		vec2 p3 = p;
		p3 *= 60.;
		p3 -= vec2( -30, 25. );
        float maskHs = highscoreText( p3 ); // Highscore
        color.rgb = mix( vec3( 1.0 ), color.rgb, maskHs );

		vec2 pScore = p;
        pScore *= 12.0;
        pScore -= vec2( 1.3, 5.3 );
        float sScore = printInt( pScore, s.highscore );
        color.rgb = mix( color.rgb, vec3( 1.0 ), sScore );
    }
    else
    {
        vec2 pScore = p;
        pScore *= 6.0;
        pScore -= vec2( -0.9, 3.4 );
        float maxDigits = ceil( log2( s.score ) / log2( 10.0 ) );
        pScore.x += 0.5 * maxDigits;
        float sScore = printInt( pScore, s.score );
        color.rgb = mix( color.rgb, vec3( 1.0 ), sScore );
    }

	// color.rgb = mix(color.rgb, vec3(0.0), 1.0-smoothstep(0.0, 0.01, abs(p.x)) ); // center guide
    // color.rgb = mix(color.rgb, vec3(0.0), 1.0-smoothstep(0.0, 0.01, abs(p.y)) ); // horiz guide
}

// Function 131
bool buttonMask(vec2 u){return(false
 //leftmost bottom-most button is vec2(0,0)!!
 ||InButton(     vec2(2))  //hide button at position(2,2) (2 right, 2 up)
 ||InButton(     vec2(1,2))//hide button at position(1,2)
 ||InButtonCross(vec2(3))//hide a cross of buttons that is over button at position (3,3)
 ||InButtonY(    vec2(7))//hide a cross of buttons that is over button at position (7,7)
 //evil people hide buttons at random with a seed that changes over time.
 ||InButton (    vec2(6,0)+vec2(floor(fract(iTime)*10.))));
 //just to show that buttons can be masked dnamically.
 ;}

// Function 132
bool isButton(vec2 p){
	vec2 dist=vec2(0.0);
	dist.x = clamp(abs(p.x)-abs(0.35),0.0,1.0);
	dist.y = clamp(abs(p.y)-abs(0.35),0.0,1.0);
	return (length(dist)<0.1);
}

// Function 133
void UILayout_SetControlRect( inout UILayout uiLayout, Rect rect )
{
    uiLayout.controlRect = rect;
    
    uiLayout.vControlMax = max( uiLayout.vControlMax, rect.vPos + rect.vSize );
    uiLayout.vControlMin = max( uiLayout.vControlMin, rect.vPos );    
}

// Function 134
void UI_PanelEnd( inout UIContext uiContext, inout UIPanelState panelState )
{
    if ( !uiContext.bPixelInView )
    {
        // Restore parent window color if outside view
	    uiContext.vWindowOutColor = panelState.vParentWindowColor;    
    }

    UI_SetDrawContext( uiContext, panelState.parentDrawContext );
}

// Function 135
vec3 viridis_quintic( float x )
{
	x = clamp( x, 0.,1. );
	vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3(
		dot( x1.xyzw, vec4( +0.280268003, -0.143510503, +2.225793877, -14.815088879 ) ) + dot( x2.xy, vec2( +25.212752309, -11.772589584 ) ),
		dot( x1.xyzw, vec4( -0.002117546, +1.617109353, -1.909305070, +2.701152864 ) ) + dot( x2.xy, vec2( -1.685288385, +0.178738871 ) ),
		dot( x1.xyzw, vec4( +0.300805501, +2.614650302, -12.019139090, +28.933559110 ) ) + dot( x2.xy, vec2( -33.491294770, +13.762053843 ) ) );
}

// Function 136
bool UIDrawContext_ScreenPosInCanvasRect( UIDrawContext drawContext, vec2 vScreenPos, Rect canvasRect )
{
	vec2 vCanvasPos = UIDrawContext_ScreenPosToCanvasPos( drawContext, vScreenPos );    
    return Inside( vCanvasPos, canvasRect );
}

// Function 137
bool UI_GetBool( int iData )
{
    return UI_GetFloat( iData ) > 0.5;
}

// Function 138
vec3 plasma_quintic( float x )
{
	x = clamp( x, 0.0, 1.0);
	vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3(
		dot( x1.xyzw, vec4( +0.063861086, +1.992659096, -1.023901152, -0.490832805 ) ) + dot( x2.xy, vec2( +1.308442123, -0.914547012 ) ),
		dot( x1.xyzw, vec4( +0.049718590, -0.791144343, +2.892305078, +0.811726816 ) ) + dot( x2.xy, vec2( -4.686502417, +2.717794514 ) ),
		dot( x1.xyzw, vec4( +0.513275779, +1.580255060, -5.164414457, +4.559573646 ) ) + dot( x2.xy, vec2( -1.916810682, +0.570638854 ) ) );
}

// Function 139
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

// Function 140
vec3 magma_quintic( float x )
{
	x = saturate( x );
	vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3(
		dot( x1.xyzw, vec4( -0.023226960, +1.087154378, -0.109964741, +6.333665763 ) ) + dot( x2.xy, vec2( -11.640596589, +5.337625354 ) ),
		dot( x1.xyzw, vec4( +0.010680993, +0.176613780, +1.638227448, -6.743522237 ) ) + dot( x2.xy, vec2( +11.426396979, -5.523236379 ) ),
		dot( x1.xyzw, vec4( -0.008260782, +2.244286052, +3.005587601, -24.279769818 ) ) + dot( x2.xy, vec2( +32.484310068, -12.688259703 ) ) );
}

// Function 141
void UI_StoreControlData( inout UIContext uiContext, UIData data )
{
    UI_StoreDataBool( uiContext, data.checkboxA, DATA_CHECKBOX_A );

    UI_StoreDataValue( uiContext, data.floatA, DATA_FLOAT_A );
    UI_StoreDataValue( uiContext, data.floatB, DATA_FLOAT_B );
    UI_StoreDataValue( uiContext, data.floatC, DATA_FLOAT_C );

    UI_StoreDataValue( uiContext, data.floatSPD, DATA_FLOAT_SPD );
}

// Function 142
vec4 draw_ui(vec2 fc){
    vec2 uv=fc/iResolution.y-res;
    vec3 col=vec3(0.);
    bool is_nuion=loadval(ivec2(1,0)).x<=0.;
    float a=0.;
    if(is_nuion){
    if(is_scrolly(uv)){
        float scp=loadval(ivec2(0,0)).w;
        if((uv.y+res.y<scp+0.05)&&(uv.y+res.y>scp-0.05))
            col=whitel*0.5;
        else
            col=whitel;
        a=0.68;
    }
    else
    if(is_scrollx(uv)){
        float scp=loadval(ivec2(0,0)).z;
        if((uv.x+res.x<scp+0.05)&&(uv.x+res.x>scp-0.05))
            col=whitel*0.5;
        else
            col=whitel;
        a=0.68;
    }}
    
    if(is_nuion)
    if(uv.y>0.3){
        float d=text_z((uv+vec2(0.47817*(res.x/0.5),-0.40))*13.);
        col+=whitel*d;
        a=max(a,d);
        d=text_s((uv+vec2(0.19689*(res.x/0.5),-0.40))*13.);
        col+=whitel*d;
        a=max(a,d);
        d=text_g((uv+vec2(-0.28127*(res.x/0.5),-0.40))*13.);
        col+=whitel*d;
        a=max(a,d);
        d=text_t((uv+vec2(0.11251*(res.x/0.5),-0.30))*13.);
        col+=whitel*d;
        a=max(a,d);
        if((uv.x>0.0562*(res.x/0.5))&&(uv.x<0.28127*(res.x/0.5))&&(uv.y<0.4))
        {
            float max_pos=abs(loadval(ivec2(1,0)).x);
            float max_posy=abs(loadval(ivec2(1,0)).y);
            d=print_int((uv+vec2(-0.19689*(res.x/0.5),-0.30))*13.,int(max_pos/SSIZE)*int(max_posy*H/2.)/2);
            col+=whitel*d;
            a=max(a,d);
        }
    }
    
    if(is_nuion)
    if((uv.x>0.0562*(res.x/0.5))&&(uv.y>-0.1)){
        float d=text_m((uv+vec2(-0.28127*(res.x/0.5),-0.10))*13.);
        col+=whitel*d;
        a=max(a,d);
        d=text_n((uv+vec2(-0.3094*(res.x/0.5),-0.03))*13.);
        col+=whitel*d;
        a=max(a,d);
        d=text_f((uv+vec2(-0.3094*(res.x/0.5),0.03))*13.);
        col+=whitel*d;
        a=max(a,d);
        d=text_ms((uv+vec2(-0.3094*(res.x/0.5),0.09))*13.);
        col+=whitel*d;
        a=max(a,d);
        int sid=int(loadval(ivec2(1,1)).w);
        if(sid==0)
            if(is_n(uv)){
                d=.8;
                col+=green;
                a=max(a,d);
            }
        if(sid==1)
            if(is_f(uv)){
                d=.8;
                col+=green;
                a=max(a,d);
            }
        if(sid==2)
            if(is_s(uv)){
                d=.8;
                col+=green;
                a=max(a,d);
            }
    }
    
    if(is_reset(uv)&&is_nuion)
    {
        float d=1.;
        col+=redd*d;
        a=max(a,d*0.75);
        d=text_r((uv+vec2(-0.3094*(res.x/0.5),0.34))*13.);
        col+=whitel*d;
        a=max(a,d);
    }
    
    if(is_clean(uv)&&is_nuion)
    {
        float d=1.;
        col+=darkb*d;
        a=max(a,d*0.75);
        d=text_c((uv+vec2(-0.3094*(res.x/0.5),0.24))*13.);
        col+=whitel*d;
        a=max(a,d);
    }
    
    if(is_grav(uv))
    {
        vec2 gravity=loadval(ivec2(0,1)).xy;
        float d=1.;
        col=vec3(0.);
        a=max(a,d*(0.6-(is_nuion?0.:0.3)));
        d=draw_grav_w((uv-vec2(0.3937*(res.x/0.5),0.28))*7.5,gravity);
        col=whitel*d;
        a=max(a,d);
    }
    
    if(is_zoom(uv))
    {
        float state=loadval(ivec2(0,0)).x*0.15*2.;
        float d=1.-(is_nuion?0.:0.3);
        col=vec3(0.);
        if(abs(uv.x+0.26994*(res.x/0.5)-state+0.15-0.007)>0.015)
            col=whitel*d;
        else
            col=whitel*0.5;
        a=max(a,d*0.68);
    }
    
    if(is_speed(uv))
    {
        float state=loadval(ivec2(0,0)).y*0.15*2.;
        float d=1.-(is_nuion?0.:0.3);
        col=vec3(0.);
        if(abs(uv.x-0.16994*(res.x/0.5)-state+0.15-0.007)>0.015)
            col=whitel*d;
        else
            col=whitel*0.5;
        a=max(a,d*0.68);
    }
    
    if(is_nui(uv))
    {
        float d=1.-(is_nuion?0.:0.3);
        col+=darkb*d;
        a=max(a,d*0.75);
        d=text_nui((uv+vec2(-0.3094*(res.x/0.5),0.44))*13.);
        col+=whitel*d;
        a=max(a,d);
    }
    
    if(is_floor(uv)&&is_nuion)
    {
        float d=1.;
        if(loadval(ivec2(0,1)).w>=0.)col+=green*d;
        a=max(a,d*0.75);
        d=text_floor((uv+vec2(-0.324*(res.x/0.5),0.13))*18.);
        col+=whitel*d;
        a=max(a,d);
    }
    if(is_floor2(uv)&&is_nuion)
    {
        float d=1.;
        if(loadval(ivec2(0,1)).w<0.)col+=green*d;
        a=max(a,d*0.75);
        d=text_floor2((uv+vec2(-0.324*(res.x/0.5),0.168))*18.);
        col+=whitel*d;
        a=max(a,d);
    }
    
    
    col=clamp(col,0.,1.);
    //a=dot(col,vec3(1.))/3.;
    return vec4(col,a);
}

// Function 143
void UI_ProcessScrollbarPanelEnd( inout UIContext uiContext, inout UIPanelState scrollbarState )
{
    UI_PanelEnd( uiContext, scrollbarState );    
}

// Function 144
bool radioButton(vec3 c, vec2 p, float r, inout vec4 fragColor)
{
    float x = load(c.xy, iChannel0).x;
    float d1 = max(length(_uv-p)-r,-length(_uv-p)+r*0.7);
    float d2 = length(_uv-p)-r*0.8;
    
    if (inside(c.xy) && iMouse.w > 0.0 && length(_uvm-p)-r < 0.0)
    {
        fragColor.x = c.z;
        return true;
    }

    _d = min(_d,min(d1,d2));
    
    if (_d>0.0) { fragColor.w = -1.; return false; }
    if (_d==d1) { fragColor = vec4(_hiCol,1); }
    if (_d==d2) { fragColor = x == c.z ? vec4(_hiCol,1) : vec4(vec3(0),0.8); }
    
    return false;
}

// Function 145
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
            DrawRect( uiContext.vPixelCanvasPos, cornerRect, vec4(0.7, 0.7, 0.7, 1.0), uiContext.vWindowOutColor );
        	DrawBorderIndent( uiContext.vPixelCanvasPos, cornerRect, uiContext.vWindowOutColor );
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

// Function 146
void UI_SetDrawContext( inout UIContext uiContext, UIDrawContext drawContext )
{
    uiContext.drawContext = drawContext;
    
    uiContext.vPixelCanvasPos = UIDrawContext_ScreenPosToCanvasPos( drawContext, uiContext.vPixelPos );
    uiContext.bPixelInView = UIDrawContext_ScreenPosInView( drawContext, uiContext.vPixelPos );

    uiContext.vMouseCanvasPos = UIDrawContext_ScreenPosToCanvasPos( drawContext, uiContext.vMousePos );
    uiContext.bMouseInView = UIDrawContext_ScreenPosInView( drawContext, uiContext.vMousePos );
}

// Function 147
vec4 starGuitar(float t,vec2 u,vec3 r,vec4 m){
 ;ve3 c=ve3(0,0,0,1)
 ;//m.yw=floor(m.yw) //integrated into   ng()
 ;vec3 a=ng(u,m.xy,vec2(2.,1.))//a short jazz-chord  (7short notes)
 ;vec3 b=ng(u,m.zw,vec2(4.,1.))//a long chord        (3 notes long)
 ;a=mig(u,a,b)  //union of 2 notes (repeat to define a song) 
 //you likely want to do floor() instead, and deal with c0Continuity mantually.
     
 ;float key=floor(u.y+floor(ViewZoom))//note keyID
 ;u.y=fract(u.y) //optional lattice.y to show more than 1 waveform

 ;key=noteMIDI(key+69.-24.)//midi key 69 mapos to C3 440hZ , thre is some screenspace offset.
 ;float phase=a.x
 ;c.x=sin(pi*2.*key*u.x)
     //55*8=440 hz for a C3-frequency
     //n2f() uses a *55. scale
     //all this is a compromise between making a waveform visible and audible.
 ;//frequency=key//should be audible
    

 ;u.y=u.y*2.-1.//optional, show negative space of waveform
 ;if(a.x>-noteInf
 ){
  ;vec2 am=vec2(0)
  ;am.x=sat(1.-u.x+a.x)
  ;am.y=sat(1.+u.x+a.y)
  ;am=smoothstep(vec2(-1),vec2(0),am-1.)-1. //lerp  hermite3 (optional)
  ;c.w=max(am.x,am.y)                       //clamp linear 
  ;//c.w=smoothstep(0.,1.,c.w)
  ;c.x=sin(pi*2.*key*u.x)*c.w
  ;c.y=cos((u.x-phase)*key/55.*2.)*c.w//visible waveform (scaled down) 
  ;//c.x=(c.x*.9+u.y) //no, do not mess with c.x, it needs ti be pure for the audio rendering!

  ;c.y=c.y*.9+u.y
  ;c.w=c.w*.9+u.y
  ;c.w=abs(c.w)-.01
  ;//c.x=abs(c.x)-.05 //no, do not mess with c.x, it needs ti be pure for the audio rendering!
  ;c.y=abs(c.y)-.05
 ;}else c.xy=vec2(0,1)     
 ;c.z=a.z-.03
 ;c.yzw=smoothstep(.1,-.1,c.yzw)
 ;c.yzw=sat(c.yzw)
 ;c.y+=sat(c.w)
 ;return c;}

// Function 148
vec3 ui(in vec2 fragCoord, inout vec2 cursor)
{
	// Draw UI
    return vec3(char(ch_w, fragCoord, cursor) + char(ch_o, fragCoord, cursor) +
           char(ch_r, fragCoord, cursor) + char(ch_d, fragCoord, cursor) +
           char(ch_t, fragCoord, cursor) + char(ch_o, fragCoord, cursor)+
           char(ch_y, fragCoord, cursor) + char(ch_sp, fragCoord, cursor)+
           char(ch_v, fragCoord, cursor) + char(ch_1, fragCoord, cursor));
}

// Function 149
vec4 checkButton(in vec2 uv, in vec2 min_b, in vec2 max_b, in bool _val)
{
    vec2 center = (min_b + max_b)*.5;
    vec2 size = (max_b - min_b)*.5;
    vec2 frame1 = size * 0.95;
    vec2 fuv = abs(uv - center);
    
    vec2 fs = max(vec2(0.), fuv - size);
    vec2 fr1 = max(vec2(0.), fuv - frame1);
    
    float f1 = step(fs.x+fs.y, 0.);
    float f2 = step(fr1.x+fr1.y, 0.);
    vec4 color = vec4(f2, f1 - f2, float(_val)*f2, f1);
    
    return color;
}

// Function 150
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

// Function 151
vec4 mainImageUI2AD37(out vec4 o, in vec2 u
){o=vec4(0)
 #ifdef Scene2D
  ;o=pdOver(iDiegeticUIshow(u),o)//ui dots
  ;o=pdOver(o,iCB(o,u))//iCB(o,u)
  ;//o=pdOver(o,ltj3Wc(o,u,iResolution,iMouse))//backsrop is a 2d srawing
 #else
  #ifdef SceneTR
   ;o=pdOver(iDiegeticUIshow(u),o)//ui dots
   ;o=pdOver(o,iCB(o,u))//bezier+appolonean stuff
   ;o=pdOver(o,mTR(o,u)) //backfrop is traced 3d scene (TemporalReprojection+brdf)
  #else
   ;o=pdOver(iDiegeticUIshow(u),o)//ui dots
   ;o=pdOver(o,iCB(o,u))//bezier+appolonean stuff
   ;o=pDoOver(iAD)  //backdrop is marched 3d scene (automatic Differentiation)
  #endif
 #endif
 ;return o;}

// Function 152
void Paint_XButton(inout vec4 dest, in vec2 uv)
{
    bool is_hot = false;
    bool is_active = is_hot && false;
    
    float aspect = iResolution.x / iResolution.y;
    
    float button_scale = theme.menuSize;
    
    uv.x *= aspect;
    uv = rotate(radians(45.)) * (uv - vec2(aspect,1) + button_scale * .5);
    
    float t0 = sdBox(uv, vec2(button_scale*.35));
    float t = max(min(abs(uv.x), abs(uv.y)), t0) - .025*button_scale;
    
    float a = smoothstep(1.5, 0., t * iResolution.y);
    
    vec4 color;
    if (is_active)
    {
        color = theme.active_color;
    }
    else if (is_hot)
    {
     	color = theme.hot_color;   
    }
    else
    {
        color = theme.inactive_color;
    }
    
    dest = mix(dest, color, a);
}

// Function 153
UIData_Bool UI_GetDataBool( int iData, bool bDefault )  
{
    UIData_Bool dataBool;
        
	vec4 vData0 = LoadVec4( iChannelUI, ivec2(iData,0) );
    
    if ( iFrame == 0 )
    {
        dataBool.bValue = bDefault;
    }
    else
    {
        dataBool.bValue =  vData0.x > 0.5;
    }
    
    return dataBool;
}

// Function 154
vec4 drawUI(ivec2 coord, vec3 color0, vec3 color1, int gDiamondsRequired, int cDiamondValue, int cDiamondBonusValue,
            int gDiamondsHarvested, int gCaveState, float gTimeLeft, int gScore, int gLives, int gCave, int gLevel,
            int animFrame, int gAuxFrame)
{
    vec4 res = vec4(0.0);
    vec4 col0 = vec4(color0, 1.0);
    vec4 col1 = vec4(color1, 1.0);
    const int y = TIT_RES.y - 8;

    bool isSpawning = isState(gCaveState, CAVE_STATE_SPAWNING);
    bool isFadingIn = isState(gCaveState, CAVE_STATE_FADE_IN);
    bool isFadingOut = isState(gCaveState, CAVE_STATE_FADE_OUT);
    bool isExitOpened = isState(gCaveState, CAVE_STATE_EXIT_OPENED);
    bool isExited = isState(gCaveState, CAVE_STATE_EXITED);
    bool isTimeOut = isState(gCaveState, CAVE_STATE_TIME_OUT);
    bool isPaused = isState(gCaveState, CAVE_STATE_PAUSED);
    bool isGameOver = isState(gCaveState, CAVE_STATE_GAME_OVER);
    bool isIntermission = ((gCave % 5) == 0);
    bool isTopInfoVisible = ((animFrame - gAuxFrame) % 135) < 35;
    bool isMovingToNextLevel = isFadingOut && (isIntermission || isExited);

    if (isMovingToNextLevel)
    {
        gCave += 1;
        if (gCave > 20)
        {
            gCave = 1;
            gLevel = max(((gLevel + 1) % 6), 1);
        }
    }

    if (isGameOver)
    {
        res = lerp(res, col0, printWord8(coord - ivec2(CEH_RES.x, y), int[8](G, _, A, _, M, _, E, _)));
        res = lerp(res, col0, printWord8(coord - ivec2(CEH_RES.x * 11, y), int[8](O, _, V, _, E, _, R, _)));
    }
    else if (isPaused && isTopInfoVisible && !(isFadingIn || isFadingOut))
    {
        res = lerp(res, col0, printWord8(coord - ivec2(CEH_RES.x, y), int[8](S, P, A, C, E, B, A, R)));
        res = lerp(res, col0, printWord8(coord - ivec2(CEH_RES.x * 10, y), int[8](T, O, _, R, E, S, U, M)));
        res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x * 18, y), E));
    }
    else if (isSpawning && isIntermission)
    {
        res = lerp(res, col0, printWord8(coord - ivec2(CEH_RES.x, y), int[8](B, _, O, _, N, _, U, _)));
        res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x * 9, y), S));
        res = lerp(res, col0, printWord8(coord - ivec2(CEH_RES.x * 12, y), int[8](L, _, I, _, F, _, E, _)));
    }
    else if (isSpawning || isFadingOut)
    {
        res = lerp(res, col0, printWord8(coord - ivec2(0, y), int[8](P, L, A, Y, E, R, _, _)));
        res = lerp(res, col0, printInt(coord - ivec2(CEH_RES.x * 7, y), 1, 1));
        res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x * 8, y), L_COMMA));
        res = lerp(res, col0, printInt(coord - ivec2(CEH_RES.x * 10, y), gLives, 1));
        res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x * 12, y), M));
        res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x * 13, y), (gLives > 1) ? E : A));
        res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x * 14, y), N));
        res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x * 16, y), gCave - gCave / 5));
        res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x * 17, y), L_SLASH));
        res = lerp(res, col0, printInt(coord - ivec2(CEH_RES.x * 18, y), gLevel, 1));
    }
    else if (isTimeOut && isTopInfoVisible)
    {
        res = lerp(res, col0, printWord8(coord - ivec2(CEH_RES.x * 4, y), int[8](O, U, T, _, O, F, _, _)));
        res = lerp(res, col0, printWord8(coord - ivec2(CEH_RES.x * 11, y), int[8](T, I, M, E, _, _, _, _)));
    }
    else
    {
        if (isExitOpened)
        {
            res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x, y), L_DIAMOND));
            res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x * 2, y), L_DIAMOND));
        }
        else
        {
            res = lerp(res, col1, printInt(coord - ivec2(CEH_RES.x, y), gDiamondsRequired, 2));
        }
        res = lerp(res, col0, printLetter(coord - ivec2(CEH_RES.x * 3, y), L_DIAMOND));
        res = lerp(res, col0, printInt(coord - ivec2(CEH_RES.x * 4, y), cDiamondValue, 2));
        res = lerp(res, col1, printInt(coord - ivec2(CEH_RES.x * 7, y), gDiamondsHarvested, 2));
        res = lerp(res, col0, printInt(coord - ivec2(CEH_RES.x * 10, y), int(ceil(gTimeLeft)), 3));
        res = lerp(res, col0, printInt(coord - ivec2(CEH_RES.x * 14, y), gScore, 6));
    }
    return res;
}

// Function 155
void UI_Compose( vec2 fragCoord, inout vec3 vColor, out int windowId, out vec2 vWindowCoord )
{
    vec4 vUISample = texelFetch( iChannelUI, ivec2(fragCoord), 0 );
    
    if ( fragCoord.y < 2.0 )
    {
        // Hide data
        vUISample = vec4(1.0, 1.0, 1.0, 1.0);
    }
    
    vColor.rgb = vColor.rgb * (1.0f - vUISample.w) + vUISample.rgb;
    
    windowId = -1;
    vWindowCoord = vec2(0);
    
    if ( vUISample.a < 0.0 )
    {
        vWindowCoord = vUISample.rg;
        windowId = int(round(vUISample.b));
    }
}

// Function 156
void UI_StoreDataColor( inout UIContext uiContext, UIData_Color dataColor, int iData )
{
    vec4 vData0 = vec4(0);
    vData0.rgb = hsv2rgb( dataColor.vHSV );
        
    StoreVec4( ivec2(iData,0), vData0, uiContext.vOutData, ivec2(uiContext.vFragCoord) );            

    vec4 vData1 = vec4(0);
    vData1.rgb = dataColor.vHSV;
        
    StoreVec4( ivec2(iData,1), vData1, uiContext.vOutData, ivec2(uiContext.vFragCoord) );            
}

// Function 157
vec3 quincunxAA(sampler2D tex, vec2 fragCoord, float blur)
{
	vec3 pixelColor;
	pixelColor =  texture(tex, (fragCoord + vec2( 0.0, 0.0)) / RES).rgb / 2.0;
	pixelColor += texture(tex, (fragCoord + vec2( blur, blur)) / RES).rgb / 8.0;
	pixelColor += texture(tex, (fragCoord + vec2( blur,-blur)) / RES).rgb / 8.0;
	pixelColor += texture(tex, (fragCoord + vec2(-blur,-blur)) / RES).rgb / 8.0;
	pixelColor += texture(tex, (fragCoord + vec2(-blur, blur)) / RES).rgb / 8.0;
	return pixelColor;
}

// Function 158
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

// Function 159
vec2 UI_GetWindowSizeForContent( vec2 vContentSize )
{
    return vContentSize 
        + vec2( 0.0, UIStyle_TitleBarHeight() )
    	+ UIStyle_WindowBorderSize() * 2.0
    	+ UIStyle_WindowContentPadding() * 2.0;
}

// Function 160
void paintConfigButton(inout vec4 finalColor, in vec3 bgColor, in vec2 uu, in Button data)
{
    float invScale = iResolution.y * 2.;
    float t = sdBox(uu - data.base.pos, data.base.scale);
    float a = smoothstep(1.5, -1.5, t * invScale);
    float ao = UI_ButtonAO(t * invScale);
	finalColor.rgb = mix(finalColor.rgb * ao, mix(bgColor, UI_BACKGROUND_COLOR.rgb, UI_BACKGROUND_COLOR.a), a);
    
    // p inside button
    vec2 p = (uu - data.base.pos) / data.base.scale;
    float invScaleIcon = invScale * data.base.scale.y;
    
    if (t <= 0.)
    {
        const float R1 = 0.3, R2 = 0.6, R3 = 0.75;
        const float N = 8.;
        const float W = .15;

        float l = length(p);
        float angle = atan(p.y,p.x)/TAU + .5;
        angle = ((fract(angle * N)*2.-1.) / N)*.5*PI_2;
        vec2 pTeeth = vec2(cos(angle), sin(angle)) * l;

        float tTeeth = max( length(p)-R3,  abs(pTeeth.y)-W );
        float tIcon = max( min(length(p)-R2, tTeeth), -(length(p)-R1));

    	/*
        const float N = 6.;
        float l = length(darkItemBoxP);
        float a = atan(darkItemBoxP.y,darkItemBoxP.x)/TAU + .5;
        a = ((fract(a * N)*2.-1.) / N)*.5*PI_2;
        */
        float aIcon = smoothstep(1.5, -1.5, tIcon * invScaleIcon);

        finalColor.rgb = mix(finalColor.rgb, vec3(0), aIcon);
    }
    
}

// Function 161
void drawUI(inout vec3 color, vec2 p, AppState s)
{
    p *= R.y / R.x; // ratio and resolution indepenent scaling
    p *= 1.75;
    
    // splash screen   
    if ( s.stateID == GS_SPLASH )
    {
        color.rgb *= 0.1 + 0.9 * smoothstep( 0.75, 0.0, p.y ); // dark text bg
		vec2 p2 = p;
		p2 *= 50.;
		p2 -= vec2( -45, 27. );
        // color.rgb = mix(color.rgb, vec3(0.0), 1.0-smoothstep(0.0, 0.5, abs(p2.y)) ); // horiz guide
        
        float maskTitle = titleText( p2 ); // Moonset Drive Unlimited
        color.rgb = mix( vec3( 1.0 ), color.rgb, maskTitle );
        
		vec2 p1 = p;
		p1 *= 60. + 5. * abs( sin( 2.0 * iTime ) );
		p1 -= vec2( -47., -42. );
        float maskSpace = spaceText( p1 ); // press [space] to start
        color.rgb = mix( vec3( 1.0 ), color.rgb, maskSpace );

		vec2 p3 = p;
		p3 *= 60.;
		p3 -= vec2( -30, 25. );
        float maskHs = highscoreText( p3 ); // Highscore
        color.rgb = mix( vec3( 1.0 ), color.rgb, maskHs );

		vec2 pScore = p;
        pScore *= 12.0;
        pScore -= vec2( 1.3, 5.3 );
        float sScore = printInt( pScore, s.highscore );
        color.rgb = mix( color.rgb, vec3( 1.0 ), sScore );
    }
    else
    {
        vec2 pScore = p;
        pScore *= 6.0;
        pScore -= vec2( -0.9, 3.4 );
        float maxDigits = ceil( log2( s.score ) / log2( 10.0 ) );
        pScore.x += 0.5 * maxDigits;
        float sScore = printInt( pScore, s.score );
        color.rgb = mix( color.rgb, vec3( 1.0 ), sScore );
    }

	// color.rgb = mix(color.rgb, vec3(0.0), 1.0-smoothstep(0.0, 0.01, abs(p.x)) ); // center guide
    // color.rgb = mix(color.rgb, vec3(0.0), 1.0-smoothstep(0.0, 0.01, abs(p.y)) ); // horiz guide
}

// Function 162
float evalReducedQuinticPrime(in float x, in ReducedQuintic q) {
    return ((5.0 * x * x + 3.0 * q.p) * x + 2.0 * q.q) * x + q.r;
}

// Function 163
float 	UIStyle_TitleBarHeight() 		{ return 24.0; }

// Function 164
float TraceBuildingSide( const in C_Ray ray )
{
	float fDistance = kMaxDist;
	
	float fStepHeight = 0.14;
	float fStepDepth = 0.2;
	float fStepStart = 7.5;
	fDistance = min(fDistance, TraceBox( ray, vec3(fBuildingMin, -1.5 + fStepHeight * 0.0, fStepStart + fStepDepth * 0.0), vec3(fBuildingMax, -1.5 + fStepHeight * 1.0, fStepStart + 20.0) ));
	fDistance = min(fDistance, TraceBox( ray, vec3(fBuildingMin, -1.5 + fStepHeight * 1.0, fStepStart + fStepDepth * 1.0), vec3(fBuildingMax, -1.5 + fStepHeight * 2.0, fStepStart + 20.0) ));
	fDistance = min(fDistance, TraceBox( ray, vec3(fBuildingMin, -1.5 + fStepHeight * 2.0, fStepStart + fStepDepth * 2.0), vec3(fBuildingMax, -1.5 + fStepHeight * 3.0, fStepStart + 20.0) ));
	fDistance = min(fDistance, TraceBox( ray, vec3(fBuildingMin, -1.5 + fStepHeight * 3.0, fStepStart + fStepDepth * 3.0), vec3(fBuildingMax, -1.5 + fStepHeight * 4.0, fStepStart + 20.0) ));

	float x = -2.0;
	for(int i=0; i<5; i++)
	{
		vec3 vBase = vec3(x * 11.6, 0.0, 0.0);
		x += 1.0;
		
		fDistance = min(fDistance, TraceColumn(ray, vBase + vec3(0.0, 0.0, 8.5)));
		
		
		fDistance = min(fDistance, TracePillar(ray, vBase + vec3(-4.1, 0.0, 8.5)));	
		fDistance = min(fDistance, TracePillar(ray, vBase + vec3(4.0, 0.0, 8.5)));
	}
	

	float fBackWallDist = 9.5;
	float fBuildingHeight = 100.0;
	fDistance = min(fDistance, TraceBox( ray, vec3(fBuildingMin, -3.0, fBackWallDist), vec3(fBuildingMax, fBuildingHeight, fBackWallDist + 10.0) ));

	float fBuildingTopDist = 8.1;
	float fCeilingHeight = 4.7;
	fDistance = min(fDistance, TraceBox( ray, vec3(fBuildingMin, fCeilingHeight, fBuildingTopDist), vec3(fBuildingMax, fBuildingHeight, fBuildingTopDist + 10.0) ));

	float fRoofDistance = 6.0;
	float fRoofHeight = 21.0;
	fDistance = min(fDistance, TraceBox( ray, vec3(fBuildingMin, fRoofHeight, fRoofDistance), vec3(fBuildingMax, fRoofHeight + 0.2, fRoofDistance + 10.0) ));	
	
	return fDistance;
}

// Function 165
float guitar(float time, int key) {
    return guitar2(time,  key); 
}

// Function 166
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

// Function 167
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

// Function 168
void UI_StoreContext( inout UIContext uiContext, int iData )
{
    vec4 vData0 = vec4( uiContext.bMouseDown ? 1.0 : 0.0, float(uiContext.iActiveControl), uiContext.vActivePos.x, uiContext.vActivePos.y );
    StoreVec4( ivec2(iData,0), vData0, uiContext.vOutData, ivec2(uiContext.vFragCoord) );
}

// Function 169
Roots5 solveQuinticPolyDiv(in GeneralQuintic eq, in int newtonSteps) {
    Roots5 roots = Roots5(1, Float5(0.0, 0.0, 0.0, 0.0, 0.0));
    //ReducedQuintic rEq = tschirnhausLinear(eq);

    // TODO: choose better starting point (or is this good enough???)
    float x1 = -0.2 * eq.b / eq.a; // Inflection point
    float y1 = abs(evalQuintic(x1, eq));

    // SOSO (Same Opposite Same Opposite) approximation of roots
    // Roots are approximately solutions to:
    // ax^5 + bx^4 + cx^3 + dx^2 + ex + f = ax^5 - bx^4 + cx^3 - dx^2 + ex - f
    // ---> bx^4 + dx^2 + f = 0 ---> b(x^2)^2 + d(x^2) + f = 0
    // Which is a quadratic in x^2 that has four roots:
    // x1, x2, x3, x4 = (+/-)sqrt((-d +/- sqrt(d^2 - 4bf))/2b)
    float h = eq.d * eq.d - 4.0 * eq.b * eq.f;
    if (h > 0.0) {
        h = sqrt(h);
        float da = 2.0 * eq.b;

        float x2 = (-eq.d + h) / da;
        if (x2 > 0.0) {
            x2 = sqrt(x2);
            float y2 = abs(evalQuintic(x2, eq));
            if (y2 < y1) x1 = x2, y1 = y2;

            x2 = -x2;
            y2 = abs(evalQuintic(x2, eq));
            if (y2 < y1) x1 = x2, y1 = y2;
        }

        x2 = (-eq.d - h) / da;
        if (x2 > 0.0) {
            x2 = sqrt(x2);
            float y2 = abs(evalQuintic(x2, eq));
            if (y2 < y1) x1 = x2, y1 = y2;

            x2 = -x2;
            y2 = abs(evalQuintic(x2, eq));
            if (y2 < y1) x1 = x2, y1 = y2;
        }
    }

    for (int n=0; n < newtonSteps; n++) {
        float newtonStep = evalQuintic(x1, eq) / evalQuinticPrime(x1, eq);
        x1 -= newtonStep;//min(abs(newtonStep), 1.0) * sign(newtonStep);
    }

    set(roots.roots, 0, x1);
    vec4 factorRoots;
    float qa = eq.a, qb = qa * x1 + eq.b, qc = qb * x1 + eq.c, qd = qc * x1 + eq.d, qe = qd * x1 + eq.e;
    int nFactorRoots = solveQuartic(qa, qb, qc, qd, qe, factorRoots);
    for (int n=0; n < nFactorRoots; n++) { set(roots.roots, n + 1, factorRoots[n]); }
    roots.nroots += nFactorRoots;

    return roots;
}

// Function 170
float UI_GetFloat( int iData )
{
    return texelFetch( iChannelUI, ivec2(iData,0), 0 ).x;
}

// Function 171
void UI_ProcessWindowMain( inout UIContext uiContext, inout UIData uiData, int iControlId, int iData )
{
    UIWindowDesc desc;
    
    desc.initialRect = Rect( vec2(32, 128), vec2(380, 180) );
    desc.bStartMinimized = false;
    desc.bStartClosed = true;
    desc.uControlFlags = WINDOW_CONTROL_FLAG_TITLE_BAR | WINDOW_CONTROL_FLAG_MINIMIZE_BOX | WINDOW_CONTROL_FLAG_RESIZE_WIDGET | WINDOW_CONTROL_FLAG_CLOSE_BOX;    
    desc.vMaxSize = vec2(100000.0);
    
    UIWindowState window = UI_ProcessWindowCommonBegin( uiContext, iControlId, iData, desc );
    
    if ( window.bClosed )
    {
        //if ( uiContext.bMouseDown )
        if ( Key_IsPressed( iChannelKeyboard, KEY_SPACE ) )
        {
            window.bClosed = false;
        }
    }
    
    if ( !window.bMinimized )
    {
        // Controls...

        Rect scrollbarPanelRect = Rect( vec2(0), vec2( 300.0 + UIStyle_ScrollBarSize(), uiContext.drawContext.vCanvasSize.y ) );

        vec2 vScrollbarCanvasSize = vec2(300, 200);

        UIPanelState scrollbarPanelState;            
        UI_ProcessScrollbarPanelBegin( uiContext, scrollbarPanelState, IDC_WINDOW_SCROLLBAR, DATA_WINDOW_SCROLLBAR, scrollbarPanelRect, vScrollbarCanvasSize );

        {        
            UILayout uiLayout = UILayout_Reset();

            LayoutStyle style;
            RenderStyle renderStyle;             
            UIStyle_GetFontStyleWindowText( style, renderStyle );       

            
            UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );                
            UI_ProcessSlider( uiContext, IDC_SLIDER_SPD, uiData.floatSPD, uiLayout.controlRect );       
            //UILayout_StackDown( uiContext.uiLayout );    
            UILayout_StackRight( uiLayout );

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                uint strA[] = uint[] ( _S, _P, _D, _COLON, _SP );
                ARRAY_PRINT(state, style, strA);
                Print(state, style, int(uiData.floatSPD.fValue) );
                UI_RenderFont( uiContext, state, style, renderStyle );
                UILayout_SetControlRectFromText( uiLayout, state, style );
            }
            UILayout_StackDown( uiLayout );              

            
            
            UILayout_StackControlRect( uiLayout, UIStyle_CheckboxSize() );                
            UI_ProcessCheckbox( uiContext, IDC_CHECKBOX_A, uiData.checkboxA, uiLayout.controlRect );

            UILayout_StackRight( uiLayout );
            UILayout_StackDown( uiLayout );    


            
            UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );                
            UI_ProcessSlider( uiContext, IDC_SLIDER_FLOAT_A, uiData.floatA, uiLayout.controlRect );

            UILayout_StackRight( uiLayout );

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                uint strA[] = uint[] ( _R, _COLON, _SP );

                ARRAY_PRINT(state, style, strA);

                Print(state, style, uiData.floatA.fValue, 4 );

                UI_RenderFont( uiContext, state, style, renderStyle );

                UILayout_SetControlRectFromText( uiLayout, state, style );
            }

            UILayout_StackDown( uiLayout );    

            UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );                
            UI_ProcessSlider( uiContext, IDC_SLIDER_FLOAT_B, uiData.floatB, uiLayout.controlRect );       
            //UILayout_StackDown( uiContext.uiLayout );    
            UILayout_StackRight( uiLayout );

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                uint strA[] = uint[] ( _G, _COLON, _SP );
                ARRAY_PRINT(state, style, strA);
                Print(state, style, uiData.floatB.fValue, 4 );
                UI_RenderFont( uiContext, state, style, renderStyle );
                UILayout_SetControlRectFromText( uiLayout, state, style );
            }
            UILayout_StackDown( uiLayout );



            UILayout_StackControlRect( uiLayout, UIStyle_SliderSize() );                
            UI_ProcessSlider( uiContext, IDC_SLIDER_FLOAT_C, uiData.floatC, uiLayout.controlRect );       
            //UILayout_StackDown( uiContext.uiLayout );    
            UILayout_StackRight( uiLayout );

            {
                PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        
                uint strA[] = uint[] ( _B, _COLON, _SP );
                ARRAY_PRINT(state, style, strA);
                Print(state, style, uiData.floatC.fValue, 4 );
                UI_RenderFont( uiContext, state, style, renderStyle );
                UILayout_SetControlRectFromText( uiLayout, state, style );
            }
            UILayout_StackDown( uiLayout );     
            
            

            #if 0
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
           
        UI_ProcessScrollbarPanelEnd(uiContext, scrollbarPanelState);
    }    
    
    UI_ProcessWindowCommonEnd( uiContext, window, iData );
}

// Function 172
vec4 getBuildingTexture(TraceResult tr, vec3 normal) {
    vec3 col = vec3(0.);

    vec3 id = tr.id;
    float objId = tr.obj;

    vec3 p = tr.p;

    float baseSize = normal.x == 0. ? tr.dist.building.size.x : tr.dist.building.size.y;

    vec2 size = vec2(baseSize, tr.dist.building.height);

    vec3 cubeUV = getCubeUV(tr.q1, normal, tr.dist.building.size);
    vec2 uv = cubeUV.xy;

    if (objId == BLD_HEX) {
        uv = getHexUV(tr.q1, normal, size);
    }
    if (objId == BLD_TUBE) {
        uv = getTubeUV(tr.q1, normal, size);
    }

    vec4 tc = allWindowsSkyscraperTexture(p, uv, normal, id, cubeUV.z, tr.obj, baseSize, tr.dist.building.size);

    col += tc.rgb;

    return vec4(col, tc.w);
}

// Function 173
vec4 circuit(int i) {
  i /= 2;
#if __VERSION__ < 300
  i += 14;
  i = 1+imod(i,15);
#else
  i += 9;
  i = 1+i%15;
  i = i^(i/2);
#endif
  return vec4(imod(i,2),imod(i/2,2),imod(i/4,2),imod(i/8,2));
}

// Function 174
vec2 	UIStyle_WindowContentPadding() 	{ return vec2(16.0, 8.0); }

// Function 175
float sdEquilateralTriangle(  in vec2 p )
{
    const float k = sqrt(3.0);
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x+k*p.y>0.0 ) p=vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0, 0.0 );
    return -length(p)*sign(p.y);
}

// Function 176
void UI_ComposeWindowLayer( inout UIContext uiContext, float fTransparency, Rect windowRect )
{
#ifdef SHADOW_TEST   
  	if ( !uiContext.bPixelInView )
    {
        return;
    }

#if 1
    // cull window?
    Rect boundsRect = windowRect;
    RectExpand( boundsRect, vec2( 16.0 ) );
    if ( !Inside( uiContext.vPixelPos, boundsRect ) )
    {
        return;
    }
#endif
    
    // We need to compose in the parent drawContext for this to work...
    float fPrevShadow = uiContext.fShadow;
    
    vec2 vShadowOffset = vec2( 5.0, 8.0 );
    float fShadowInner = 3.0;
	float fShadowOuter = 12.0;
    
    Rect shadowRect = windowRect;
    RectShrink( shadowRect, vec2( fShadowInner ) );
    
    vec2 vShadowTestPos = uiContext.vPixelPos - vShadowOffset;
    vec2 vWindowClosest = clamp( vShadowTestPos, shadowRect.vPos, shadowRect.vPos + shadowRect.vSize );

    float fWindowDist = length( vWindowClosest - vShadowTestPos );
    
    float fCurrentShadow = clamp( (fWindowDist) / (fShadowOuter + fShadowInner), 0.0, 1.0 );
    fCurrentShadow = sqrt( fCurrentShadow );
    float fShadowTransparency = 0.5;
	uiContext.fShadow *= fCurrentShadow * (1.0 - fShadowTransparency) + fShadowTransparency; 
#endif    

  	if ( !Inside( uiContext.vPixelPos, windowRect ) )
    {
        return;
    }

    float fBlend = uiContext.fBlendRemaining * (1.0f - fTransparency);

#ifdef SHADOW_TEST
    uiContext.fOutShadow *= fPrevShadow * (fBlend) + (1.0 - fBlend);
#endif
    
    // never blend under "ID" window
    if ( uiContext.vOutColor.a < 0.0 )
    {
        return;
    }
    
    if ( uiContext.vWindowOutColor.a < 0.0 )
    {
        if ( uiContext.fBlendRemaining == 1.0f )
        {
            // Ouput ID without blending
            uiContext.vOutColor = uiContext.vWindowOutColor;
            uiContext.fBlendRemaining = 0.0f;
            return;
        }
        else
        {
            // blending id under existing color - blend in grey instead of ID
            uiContext.vWindowOutColor = vec4(0.75, 0.75, 0.75, 1.0);
        }
    }

    uiContext.vOutColor += uiContext.vWindowOutColor * fBlend;
    
    uiContext.fBlendRemaining *= fTransparency;
}

// Function 177
uint HashUInt(vec4  v, uvec4 r) { return Hash(floatBitsToUint(v), r); }

// Function 178
void UILayout_SetControlRectFromText( inout UILayout uiLayout, PrintState state, LayoutStyle style )
{
    UILayout_SetControlRect( uiLayout, UI_GetFontRect( state, style ) );
}

// Function 179
UIData_Value UI_GetDataValue( int iData, float fDefaultValue, float fRangeMin, float fRangeMax )  
{
    UIData_Value dataValue;
    
    vec4 vData0 = LoadVec4( iChannelUI, ivec2(iData,0) );
    
    if ( iFrame == 0 )
    {
        dataValue.fValue = fDefaultValue;
    }
    else
    {
        dataValue.fValue = vData0.x;
    }
    
    dataValue.fRangeMin = fRangeMin;
    dataValue.fRangeMax = fRangeMax;
    
    return dataValue;
}

// Function 180
bool UI_DrawWindowMinimizeWidget( inout UIContext uiContext, bool bMinimized, Rect minimizeBoxRect )
{
	if (!uiContext.bPixelInView || !Inside( uiContext.vPixelCanvasPos, minimizeBoxRect ))
        return false;
    
    vec2 vArrowPos = minimizeBoxRect.vPos + minimizeBoxRect.vSize * 0.5;        
    vec2 vArrowSize = minimizeBoxRect.vSize * 0.25;
    vec4 arrowColor = vec4(0.0, 0.0, 0.0, 1.0);
    if ( !bMinimized )
    {
        DrawLine( uiContext.vPixelCanvasPos, vArrowPos + vec2(-1.0, -0.5) * vArrowSize, vArrowPos + vec2(0.0, 0.5) * vArrowSize, 2.0f, arrowColor, uiContext.vWindowOutColor );
        DrawLine( uiContext.vPixelCanvasPos, vArrowPos + vec2( 1.0, -0.5) * vArrowSize, vArrowPos + vec2(0.0, 0.5) * vArrowSize, 2.0f, arrowColor, uiContext.vWindowOutColor );
    }
    else
    {
        DrawLine( uiContext.vPixelCanvasPos, vArrowPos + vec2( 0.5, 0.0 )* vArrowSize, vArrowPos + vec2(-0.5, -1.0) * vArrowSize, 2.0f, arrowColor, uiContext.vWindowOutColor );
        DrawLine( uiContext.vPixelCanvasPos, vArrowPos + vec2( 0.5, 0.0 )* vArrowSize, vArrowPos + vec2(-0.5,  1.0) * vArrowSize, 2.0f, arrowColor, uiContext.vWindowOutColor );
    }    
    
    return true;
}

// Function 181
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

// Function 182
bool is_material_liquid(const int material)
{
    return is_material_any_of(material, MATERIAL_MASK_LIQUID);
}

// Function 183
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

// Function 184
vec4 decor_ui_box(int idx) {
    
    return vec4(inset_ctr.x + (float(idx)-1.5)*text_size*1.1,
                dfunc_y - 2.5*text_size,
                vec2(0.45*text_size));
    
}

// Function 185
void
paintGui (inout vec4 fragColor, in vec3 blurredColor, in vec2 coord)
{
    // Panels rendering
    float aspect = iResolution.x / iResolution.y;
    float t1 = -(coord.y - (iResolution.y - float(.1)));
    float s1 = smoothstep(1., -1., t1);
    
    vec4 topPanelColor = vec4(1);
    
    float valOpen = readVar(guiStatePtr).x;
    bool isOpen = bool(valOpen);
    paintSidePanel(fragColor, blurredColor, coord, valOpen, isOpen);
    
    float ao = (1.-.2*SATURATE(exp(-.6 * t1)));
    fragColor.rgb *= ao;
    fragColor.rgb = mix(fragColor.rgb, mix(blurredColor, TRANSLUCENT_COLOR, GUI_TRANSLUCENCY), s1);
	    
	//paintCreeper(fragColor, coord - vec2(0, iResolution.y - float(TOP_PANEL_SIZE)), .65);
    
    
    // Overlay info
    if (false)
    {
       	// Overlay text
        float scale = 10.;
        vec2 uv = coord / iResolution.y * iResolution.y / scale;
        float px = 1. / iResolution.y * iResolution.y / scale;
        
        float x = 100.;
        float cp = 0.;
        vec4 cur = vec4(0,0,0,.01);
        vec4 us = cur;
        float ital = 0.0;

        int lnr = int(floor(uv.y/2.));
        uv.y = mod(uv.y,2.0)-1.0;
        
        if (lnr == 0)
        {
            ITAL DARKGREY W_ o_ r_ l_ d_ _ D_ i_ m_ _dotdot _;
            BLACK _open1 _close1;

            float weight = 0.05+cur.w*.02;//min(iTime*.02-.05,0.03);//+.03*length(sin(uv*6.+.3*iTime));//+0.02-0.06*cos(iTime*.4+1.);
            fragColor.rgb = mix(fragColor.rgb, us.rgb, smoothstep(weight+px, weight-px, x));
        }
    }
}

// Function 186
float get_ui(in sampler2D s)
{
    return texelFetch(s, CTRL_GUI, 0).w;
}

// Function 187
vec3 uiColor(int id){return texture(iChannel0, vec2(float(id)+.5,1.5)/iResolution.xy).rgb;}

// Function 188
Sprite getSpriteUindefined()
{
    // return Sprite(ivec4(1313772435, 3840162105, 1313772435, 3840162105), ivec4(1313772435, 3840162105, 1313772435, 3840162105));  // diagonal lines
    // return Sprite(ivec4(859032780, 859032780, 859032780, 859032780), ivec4(859032780, 859032780, 859032780, 859032780));  // checker
    // return Sprite(ivec4(0, 806105100, 204475440, 62915520), ivec4(62915520, 204475440, 806105100, 0));  // cross
    return Sprite(ivec4(0, 1880911900, 477109360, 130025408), ivec4(130025408, 477109360, 1880911900, 0));  // cross with shadow
}

// Function 189
vec4 horizontalRadioButton(in vec2 uv, in vec2 min_b, in vec2 max_b, in float _val, in float n)
{
    vec2 center = (min_b + max_b)*0.5;
    vec2 size1 = (max_b - min_b) * 0.5;
    vec2 frame = 0.98 * size1;
    
    float ratio = iResolution.x / iResolution.y;
    vec2 scl_uv = uv;
    scl_uv.x *= ratio;
    
    vec3 background = vec3(0.05, 0.02, 0.01);
    
    float inside = step(sdBox(uv - center, frame) - 0.04*size1.x, 0.);
    float boundary = step(sdBox(uv - center, size1) - 0.08*size1.x, 0.) - inside;
    float val = 0.5 + 0.3*sin(PI*0.5 + PI2*saturate(uv.x, min_b.x, max_b.x));
    float start_x = center.x - size1.x;
    float bval = saturate(uv.x, start_x, center.x + size1.x);
    
    float inv_n = 1. / n;
    
    float modeDiscr = floor(_val * n)*inv_n;
    vec2 center_button = vec2(start_x + (modeDiscr + inv_n*0.5)*size1.x*2., center.y);
    center_button.x *= ratio;
    float bcircle = length(scl_uv - center_button);
    float button = smoothstep(0.003, 0.005, bcircle) - smoothstep(0.005, 0.007, bcircle);
    
    vec4 color = clamp(vec4(background.x + 0.15*button, background.y + button*0.45 + boundary*val, 
                            button*0.5 + background.z +boundary*val, 1.), 0., step(0.5, inside+boundary));

    return color;
}

// Function 190
vec3 UVToEquirectCoord(float U, float V, float MinCos)
{
    float Phi = kPi - V * kPi;
    float Theta = U * 2.0 * kPi;
    vec3 Dir = vec3(cos(Theta), 0.0, sin(Theta));
	Dir.y   = clamp(cos(Phi), MinCos, 1.0);
	Dir.xz *= Sqrt(1.0 - Dir.y * Dir.y);
    return Dir;
}

// Function 191
vec3 userInterface() {
	vec2 uv = FragCoord.xy/iResolution.y - vec2(.8,.5);
	vec3 col=vec3(0.); float d;
	vec4 mouse = iMouse/iResolution.y;

	if(!key_toggle(9.)) return col; // 'TAB' key : automatic stars field -> exit
	
	if (mouse.x+mouse.y==0.) mouse.xy=vec2(.3,.1); // 1st mouse position silly
	
	d = length(uv+vec2(.8,.5)-mouse.xy); // color cursor
	if (d<.02) col = vec3(0.,0.,1.);
	
	if(key_toggle(84.))  // 'T' key : tune RGB vs Temperature->Planck Spectrum
	{   // ---  Plank Spectrum mode ---
		float T = 40000.*iMouse.x/iResolution.x;
		star_color = Planck(T);
		// star_luminosity = pow(T,4.);
	} 
	else 
	{   // --- RGB mode ---
		star_color.gb = mouse.xy*star_luminosity; 
		if(key_toggle(32.))  // SPACE key: red or blue dominant, tune the 2 others
		{ star_color=star_color.bgr; col=col.bgr;}
	}
	
	// display the 3-filters analyzor at bottom
	if ((uv.y<-.4)&&(abs(uv.x)<.102)) {
		if (uv.y<-.402) col=  vec3(
			((uv.x>-.10)&&(uv.x<-.031))?1.:0., // red frame
			((uv.x>-.029)&&(uv.x<.029))?1.:0., // green frame
			((uv.x< .10)&&(uv.x> .031))?1.:0.  // blue frame
		)*star_color/star_luminosity;
	if ((abs(uv.x)<.102)&&(col.r+col.g+col.b==0.)) col = vec3(1.);
	}
	
	return col;
}

// Function 192
vec2 circuit(vec3 p)
{
	p = mod(p, 2.0) - 1.0;
	float w = 1e38;
	vec3 cut = vec3(1.0, 0.0, 0.0);
	vec3 e1 = vec3(-1.0);
	vec3 e2 = vec3(1.0);
	float rnd = 0.23;
	float pos, plane, cur;
	float fact = 0.9;
	float j = 0.0;
	for(int i = 0; i < ITS; i ++)
	{
		pos = mix(dot(e1, cut), dot(e2, cut), (rnd - 0.5) * fact + 0.5);
		plane = dot(p, cut) - pos;
		if(plane > 0.0)
		{
			e1 = mix(e1, vec3(pos), cut);
			rnd = fract(rnd * 9827.5719);
			cut = cut.yzx;
		}
		else
		{
			e2 = mix(e2, vec3(pos), cut);
			rnd = fract(rnd * 15827.5719);
			cut = cut.zxy;
		}
		j += step(rnd, 0.2);
		w = min(w, abs(plane));
	}
	return vec2(j / float(ITS - 1), w);
}

// Function 193
UIDrawContext UIDrawContext_SetupFromRect( Rect rect )
{
    UIDrawContext drawContext;
    drawContext.viewport = rect;
    drawContext.vOffset = vec2(0);
    drawContext.vCanvasSize = rect.vSize;
	return drawContext;
}

// Function 194
void gui_theta_update() {
    
    if (fc.x != THETA_COL) { return; }
    
    if (iMouse.z > 2.*inset_ctr.x && iMouse.w > 0.) {
        
        // mouse down somewhere in the pane but not in GUI panel    
        
    	if ( length(iMouse.zw - object_ctr) < 0.45 * iResolution.y) {

            // down somewhere near object
            vec2 disp = (iMouse.xy - object_ctr) * 0.01;
            data.xyz = vec3(-disp.y, disp.x, 1);
            
        } else {
            
            // down far from object
            data.z = 0.;
            
        }
        
    }
    
        
    if (data.z == 0.) {
        float t = iTime;
        data.x = t * 2.*PI/6.; 
        data.y = t * 2.*PI/18.;
    }    
    
}

// Function 195
vec4 uiSlider(int id){return texture(iChannel0, vec2(float(id)+.5,0.5)/iResolution.xy);}

// Function 196
void UIStyle_GetFontStyleTitle( inout LayoutStyle style, inout RenderStyle renderStyle )
{
    style = LayoutStyle_Default();
	renderStyle = RenderStyle_Default( cWindowTitle );
}

// Function 197
vec4 gui_arrow_left(vec4 col, vec2 uv, vec2 pos, float scale, bool check)
{
    float unit = asp * 0.01 * scale;
    float h;
    
    h = triangle(uv, pos+vec2(unit*1.8, -unit*2.), pos+vec2(unit*1.8, unit*2.), pos+vec2(-unit*1.8, 0.));
    if(!check) h = abs(h);
    col = mix(col, vec4(vec3(0.5), 1.), smoothstep(0.01, 0., h));
    
    
    return col;
}

// Function 198
vec2 UIDrawContext_ScreenPosToCanvasPos( UIDrawContext drawContext, vec2 vScreenPos )
{
    vec2 vViewPos = vScreenPos - drawContext.viewport.vPos;
    return vViewPos + drawContext.vOffset;
}

// Function 199
float equirectangular_direction(out vec3 rd)
{
  vec2 uv = gl_FragCoord.xy / iResolution.xy;
  
  // Calculate azimuthal and polar angles from screen coordinates
  float theta =  uv.t * PI,
        phi =  uv.s * 2.0 * PI;
        
  // Calculate ray directions from polar and azimuthal angle
  rd = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
  
  // formulas are on wikipedia:
  // https://en.wikipedia.org/wiki/Spherical_coordinate_system	
  return 1.0;
}

// Function 200
UIData_Color UI_GetDataColor( int iData, vec3 cDefaultRGB )  
{
    UIData_Color dataColor;
    
    vec4 vData1 = LoadVec4( iChannelUI, ivec2(iData,1) );
    
    if ( iFrame == 0 )
    {
        dataColor.vHSV = rgb2hsv( cDefaultRGB );
    }
    else
    {
        dataColor.vHSV = vData1.rgb;
    }
    
    return dataColor;
}

// Function 201
uint HashUInt(float v, uint  r) { return Hash(floatBitsToUint(v), r); }

// Function 202
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

// Function 203
vec4 buildings(vec2 pos){
	vec4 col = vec4(0.0);
    
    float roof = 0.1;
    float bx = pos.x * 20.0;
    float x = 0.05 * floor(bx - 4.0);
    // BX = Position relative to building left
    bx = mod(bx,1.0);
  
    // Build pseudorandom rooftop line
    roof += 0.06 * cos(x * 2.0);
	roof += 0.1 * cos(x * 23.0);
	roof += 0.02 * cos(x * 722.0 );
	roof += 0.03 * cos(x * 1233.0 );
	
    roof += 0.06;
    
    if(pos.y < roof && pos.y > 0.0 && bx > 0.1 * cos(20.0 * pos.x)){
    	col.b += 0.4;
        
        // Draw windows
        float window = abs(sin(200.0 * pos.y));
        window *= abs(sin(20.0 * bx));
        
        // type 1 window
        if(mod(2023.0 * x,2.0) < 0.5){
          	window = floor(1.3 * window);
        	col.rgb += 1.5 * window * vec3(0.9,0.9,0.9);
        }
        // type 2 window
        else if(mod(2983.0 * x,2.0) < 1.3){
        	col.rb += window;
        }
        else {
            if(window > 0.5){
            	col.rg += 0.8;
           	}
        }
      	col.a = 1.0;
    }

    return col;
}

// Function 204
UILayout UILayout_Reset()
{
    UILayout uiLayout;
    
    uiLayout.fTabPosition = 0.0;
    uiLayout.vCursor = vec2(0);
    uiLayout.controlRect = Rect( vec2(0), vec2(0) );
    uiLayout.vControlMax = vec2(0);
    uiLayout.vControlMin = vec2(0);
    
    return uiLayout;
}

// Function 205
vec3 fluidPos(vec3 p)
{
    float lmin=min(FRes.x,min(FRes.y,FRes.z));
    return p*lmin*.5+FRes*.5;
}

// Function 206
float BuildingsDistance(in vec3 position){
  return min(CubeRepetition(position                                ,vec3(80.,0., 90.)),
	         CubeRepetition(position+vec3(350.,sin(time)*30.,0.) ,vec3(90.,0.,100.)));}

// Function 207
void drawUI( inout vec4 c, sampler2D iChannelFont, sampler2D iChannel0, vec2 fragCoord, vec2 iResolution, vec4 iMouse){
    vec2 uv = fragCoord / iResolution.xy;
    vec2 a = vec2( iResolution.x/ iResolution.y, 1.);
    vec2 p = (uv-.5)*a;
    for (int i = 0 ; i < componentsLength; i++)
    {
        Component comp = components[i];
        // switch(comp.type)  // removed to compatibility but losing performance
        {
            //case TYPE_CHECKBOX:
            if (comp.type == TYPE_CHECKBOX)
            {
                drawCheckbox(c, p, comp.positionSize.xy, float(comp.value));
            	drawText(c, p, comp.positionSize.xy+vec2(0.08,0.), .8, comp.textHexStr, vec4(0.)); 
            }
            if (comp.type == TYPE_BUTTON)
            {
                drawButton(c, p, comp.positionSize, float(comp.value));
                drawText( c, p, comp.positionSize.xy+vec2(0.01,0.)-comp.positionSize.zw, 1., comp.textHexStr, vec4(0.)); 
            }
            if (comp.type == TYPE_LABEL)
            {
            	drawText( c, p, comp.positionSize.xy+vec2(0.01,0.), comp.positionSize.z, comp.textHexStr, comp.color); 
            }
            if (comp.type == TYPE_LABEL_NUMBER)
            {
                float len = drawText(c, p, 1.*comp.positionSize.xy+vec2(0.01,0.), comp.positionSize.z, comp.textHexStr, comp.color); 
                drawTextNumber( c, p, 1.*comp.positionSize.xy+vec2(0.01+len*.04*comp.positionSize.z,0.), comp.positionSize.z, comp.value.x, comp.color);
            }
        }
    }
}

// Function 208
float   UIStyle_WindowTransparency() 	{ return 0.025f; }

// Function 209
vec2 Buildings( vec2 uv )
{
//    uv -= .05; // make thick & thin roads
    uv -= .5; // less repetetive buildings! (nice)
    
    uvec2 iuv = uvec2(abs(uv));
    
    // DAMMIT! They're all the same building!
    // can skip the wrap in the last stage, but it's only 1.68 smaller than the next wrap
    // so we get 2x2 blocks of buildings, of 2 (maybe 4) different heights
    uint seed = Hash(iuv.x+(iuv.y<<8U));
    float h = float(seed&0xffffffU)/float(0xffffffU);
    
    // => most buildings have a mirror running through them, so I need a mirror that syncs seeds for straddling buildings
/* e.g: 3D "buildings" pattern, reflection = 90 degree rotation...
=> won't get same angle both sides => will slice through buildings into the ground
Which will break the SDFs (creating flat spots and shallower gradients - fairly safe but annoying)
*/    
/*
It would be easy if we were spawning (not just pickig) tiles
each tile could check its already picked neighbours so would never create invalid transition
though obviously wouldn't be deterministic
*/
/*
I wonder if there's something we can do by colouring in an intermediate buffer...
*/
    
    vec2 d = abs(fract(uv)-.5)-(.5-roadRad);
    float f = max(d.x,d.y);
    if ( f > .0 ) f = length(max(d,0.));
    
    return vec2(f,h);
}

// Function 210
UIContext UI_GetContext( vec2 fragCoord, int iData )
{
    UIContext uiContext;
    
    uiContext.vPixelPos = fragCoord;
    uiContext.vPixelPos.y = iResolution.y - uiContext.vPixelPos.y;
    uiContext.vMousePos = iMouse.xy;
    uiContext.vMousePos.y = iResolution.y - uiContext.vMousePos.y;
    uiContext.bMouseDown = iMouse.z > 0.0;       
    
    vec4 vData0 = LoadVec4( iChannelUI, ivec2(iData,0) );
    
    uiContext.bMouseWasDown = (vData0.x > 0.0);
    
    uiContext.vFragCoord = ivec2(fragCoord);
    uiContext.vOutColor = vec4(0.0);
#ifdef SHADOW_TEST    
    uiContext.fShadow = 1.0;
    uiContext.fOutShadow = 1.0f;
#endif    
    uiContext.fBlendRemaining = 1.0;
    
    uiContext.vOutData = vec4(0.0);
    if ( int(uiContext.vFragCoord.y) < 2 )
    {
        // Initialize data with previous value
	    uiContext.vOutData = texelFetch( iChannelUI, uiContext.vFragCoord, 0 );     
    }
    uiContext.bHandledClick = false;
    
    uiContext.iActiveControl = int(vData0.y);
    uiContext.vActivePos = vec2(vData0.zw);
        
    
    UIDrawContext rootContext;
    
    rootContext.vCanvasSize = iResolution.xy;
    rootContext.vOffset = vec2(0);
    rootContext.viewport = Rect( vec2(0), vec2(iResolution.xy) );
    rootContext.clip = rootContext.viewport;

    UI_SetDrawContext( uiContext, rootContext );
        
    if ( iFrame == 0 )
    {
        uiContext.bMouseWasDown = false;
        uiContext.iActiveControl = IDC_NONE;
    }
    
    return uiContext;
}

// Function 211
float ShoulderButtons(in vec3 p,float controllBase,float oScale
){
 ;vec3 q=p;q.x=abs(q.x)
 ;float d=sdCappedCylinder(q-vec3(1.45,-.1,.10),vec2(1,.12)*oScale)
 ;d=max(d,-sdBox(q-vec3(2.5,0,.42),vec3(.4,1.08,2.9)*oScale))
 ;d=-max(-d,-sdBox(p-vec3(0,-.1,.2),vec3(1.6,.08,.8)*oScale))
 ;d=max(d,-sdBox(p-vec3(0,0,.42),vec3(1.2,1.08,2.9)*oScale))
 ;d=max(d,-sdBox(p-vec3(0,0,-.42),vec3(3.,.68,1.)*oScale))
 ;return d;}

// Function 212
float cruiser(vec3 p) {
	float ship = box(p, vec3(4. + p.y*.3, .8, 1. + p.y*.3));
    p.xy -= vec2(1.5, .3);
	return .6 * min(ship, octahedron(p, 1.4));
}

// Function 213
vec2 bassguitar(float t, float rt, float freq)
{
    return vec2(sine(rt * freq), triangle(rt * freq)) * asd(t, 0.001, 0.0, 0.499) * rot(rt * PI * freq * 0.25);
}

// Function 214
void quinticFromRoots(in float x1, in float x2, in float x3, in float x4, in float x5, inout GeneralQuintic q) {
    q.a = 1.0;
    q.b = -x1 - x2 - x3 - x4 - x5;
    q.c = x1 * x2 + x3 * x4 + (x1 + x2) * (x3 + x4) + (x1 + x2 + x3 + x4) * x5;
    q.d = -(x1 + x2) * x3 * x4 - x1 * x2 * (x3 + x4) - (x1 * x2 + x3 * x4 + (x1 + x2) * (x3 + x4)) * x5;
    q.e = x1 * x2 * x3 * x4 + ((x1 + x2) * x3 * x4 + x1 * x2 * (x3 + x4)) * x5;
    q.f = -x1 * x2 * x3 * x4 * x5;
}

// Function 215
UIData UI_GetControlData()
{
    UIData data;
    
    data.checkboxA = UI_GetDataBool( DATA_CHECKBOX_A, true );
    
    data.floatA = UI_GetDataValue( DATA_FLOAT_A, 0.0583, 0.0, 1.0, false );
    data.floatB = UI_GetDataValue( DATA_FLOAT_B, 0.2416,  0.0, 1.0, false );
    data.floatC = UI_GetDataValue( DATA_FLOAT_C, 0.2000, 0.0, 1.0, false );

    data.floatSPD = UI_GetDataValue( DATA_FLOAT_SPD, 3.0, 0.0, 8.0, true );
        
    return data;
}

// Function 216
void UI_DrawButton( inout UIContext uiContext, bool bActive, bool bMouseOver, Rect buttonRect )
{
	if (!uiContext.bPixelInView)
        return;
    
    if ( bActive && bMouseOver )
    {
    	DrawBorderIndent( uiContext.vPixelCanvasPos, buttonRect, uiContext.vWindowOutColor );
    }
    else
    {
    	DrawBorder( uiContext.vPixelCanvasPos, buttonRect, uiContext.vWindowOutColor );
    }
}

// Function 217
RayHit BuildRayHit(in Ray ray, in vec4 hitInfo)
{
	RayHit hit;
    
    vec3  hitSurfNorm  = UnpackNorm(hitInfo.r);    
    vec3  hitSceneInfo = UnpackR8G8B8(hitInfo.g);  // .r = 2D heightmap value, .g = shadow value, .b = steepness
    float hitDepth     = hitInfo.b;
    
    hit.hit       = (hitInfo.a > Epsilon ? true : false);
    hit.surfPos   = ray.origin + (ray.direction * hitDepth);
    hit.surfNorm  = hitSurfNorm;
    hit.heightmap = hitSceneInfo.r;
    hit.shadow    = hitSceneInfo.g;
    hit.steepness = (hitSceneInfo.b * 2.0) - 1.0;
    
    return hit;
}

// Function 218
Camera Camera_Build() 
{ 
    Camera camera; 
    
    camera.origin  = Camera_GetPosition(); 
    camera.forward = Camera_GetForward(); 
    camera.right   = normalize(cross(camera.forward, vec3(0.0, 1.0, 0.0))); 
    camera.up      = normalize(cross(camera.right, camera.forward)); 
    
    return camera; 
}

// Function 219
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
        vec4 vColor = vec4(0.6, 0.6, 0.6, 1.0);
        
        if( bActive )
        {
            vColor = vec4(0.8, 0.8, 0.8, 1.0);
        }
        uiContext.vWindowOutColor = vColor;
    }    
}

// Function 220
void UI_RenderFont( inout UIContext uiContext, PrintState state, LayoutStyle style, RenderStyle renderStyle )
{
    if( uiContext.bPixelInView )
    {
        RenderFont( state, style, renderStyle, uiContext.vWindowOutColor.rgb );
    }
}

// Function 221
void generate_ui_textures(inout vec4 fragColor, vec2 fragCoord)
{
#if !ALWAYS_REFRESH_TEXTURES
    if (iFrame != 0)
        return;
#endif
    
    const int
		UI_TEXTURE_OPTIONS		= 0,
		UI_TEXTURE_QUAKE_ID		= 1,
        AA_SAMPLES				= clamp(TEXTURE_AA, 1, 128);
    int id = -1;

    vec2 texture_size, bevel_range;
    vec3 base_color;
    
    if (is_inside(fragCoord, ADDR2_RANGE_TEX_OPTIONS) > 0.)
    {
        id = UI_TEXTURE_OPTIONS;
        fragCoord -= ADDR2_RANGE_TEX_OPTIONS.xy;
        texture_size = ADDR2_RANGE_TEX_OPTIONS.zw;
        bevel_range = vec2(1.7, 3.9);
        base_color = vec3(.32, .21, .13);
    }

    if (is_inside(fragCoord, ADDR2_RANGE_TEX_QUAKE) > 0.)
    {
        id = UI_TEXTURE_QUAKE_ID;
        fragCoord -= ADDR2_RANGE_TEX_QUAKE.xy;
        fragCoord = fragCoord.yx;
        texture_size = ADDR2_RANGE_TEX_QUAKE.wz;
        bevel_range = vec2(2.7, 4.9);
        base_color = vec3(.16, .12, .07);
    }
    
    if (id == -1)
        return;

    vec2 base_coord = floor(fragCoord);
    float grain = random(base_coord);

    vec3 accum = vec3(0);
    for (int i=NO_UNROLL(0); i<AA_SAMPLES; ++i)
    {
        fragCoord = base_coord + hammersley(i, AA_SAMPLES);
        vec2 uv = fragCoord / min_component(texture_size);

        float base = weyl_turb(3.5 + uv * 3.1, .7, 1.83);
        if (id == UI_TEXTURE_QUAKE_ID && fragCoord.y < 26. + base * 4. && fragCoord.y > 3. - base * 2.)
        {
            base = mix(base, grain, .0625);
            fragColor.rgb = vec3(.62, .30, .19) * linear_step(.375, .85, base);
            vec2 logo_uv = (uv - .5) * vec2(1.05, 1.5) + .5;
            logo_uv.y += .0625;
            float logo_sdf = sdf_id(logo_uv);
            float logo = sdf_mask(logo_sdf + .25/44., 1.5/44.);
            fragColor.rgb *= 1. - sdf_mask(logo_sdf - 2./44., 1.5/44.);
            fragColor.rgb = mix(fragColor.rgb, vec3(.68, .39, .17) * mix(.5, 1.25, base), logo);
        }
        else
        {
            base = mix(base, grain, .3);
            fragColor.rgb = base_color * mix(.75, 1.25, smoothen(base));
        }

        float bevel_size = mix(bevel_range.x, bevel_range.y, smooth_weyl_noise(uv * 9.));
        vec2 mins = vec2(bevel_size), maxs = texture_size - bevel_size;
        vec2 duv = (fragCoord - clamp(fragCoord, mins, maxs)) * (1./bevel_size);
        float d = mix(length(duv), max_component(abs(duv)), .75);
        fragColor.rgb *= clamp(1.4 - d*mix(1., 1.75, sqr(base)), 0., 1.);
        float highlight = 
            (id == UI_TEXTURE_OPTIONS) ?
            	max(0., duv.y) * step(d, .55) :
        		sqr(sqr(1. + duv.y)) * around(.4, .4, d) * .35;
        fragColor.rgb *= 1. + mix(.75, 2.25, base) * highlight;

        if (DEBUG_TEXT_MASK != 0)
        {
            float sdf = (id == UI_TEXTURE_OPTIONS) ? sdf_Options(fragCoord) : sdf_QUAKE(fragCoord);
            fragColor.rgb = vec3(sdf_mask(sdf, 1.));
            accum += fragColor.rgb;
            continue;
        }

        vec2 engrave = (id == UI_TEXTURE_OPTIONS) ? engraved_Options(fragCoord) : engraved_QUAKE(fragCoord);
        fragColor.rgb *= mix(1., engrave.x, engrave.y);

        if (id == UI_TEXTURE_OPTIONS)
        {
            vec2 side = sign(fragCoord - texture_size * .5); // keep track of side before folding to 'unmirror' light direction
            fragCoord = min(fragCoord, texture_size - fragCoord);
            vec2 nail = add_knob(fragCoord, 1., vec2(6), 1.25, side * vec2(0, -1));
            fragColor.rgb *= mix(clamp(length(fragCoord - vec2(6, 6.+2.*side.y))/2.5, 0., 1.), 1., .25);
            nail.x += pow(abs(nail.x), 16.) * .25;
            fragColor.rgb = mix(fragColor.rgb, vec3(.7, .54, .43) * nail.x, nail.y * .75);
        }

        accum += fragColor.rgb;
    }
    fragColor.rgb = accum * (1./float(AA_SAMPLES));
}

// Function 222
vec4 char_ui_box(int idx) {
    
    const vec2 digit_rad = vec2(0.35, 0.5);
    
    return vec4(inset_ctr.x + (float(idx - 1))*text_size,
                2.*inset_ctr.y + 1.15*text_size,
                digit_rad*text_size);
    
}

// Function 223
bool is_nui(vec2 p){
    return (step(abs(p.x-0.3937*(res.x/0.5)),0.13)*step(abs(p.y+0.4),0.035)>0.5);
}

// Function 224
bool UI_ShouldProcessWindow( UIWindowState window )
{
    return !window.bMinimized && !window.bClosed;
}

// Function 225
void showGuides( inout vec4 fragColour, in vec2 fragCoord ){
    bool isMouse = iMouse.z>0.;
    vec2 ir = isMouse ? iMouse.xy : iResolution.xy;
    if(isMouse) 
    {
        float ar = ir.x/ir.y;
        vec2 ps = ir/DIM,
             pc = fragCoord-ir/2.;

        for(int i =-3; i<4; i++) {
            float f = float(i),
                a = ps.y*DY(f)+ps.x*DX(f),
                b = pc.x,
                c = pc.y*ar;
            if((abs(a+b-c)<1.)||(abs(a-b-c)<1.))
            {
                fragColour = BLK; 
                return;		//early exit
            }
        }
    }
}

// Function 226
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

// Function 227
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

// Function 228
float sd_JPBuildingRoofTopWithObjects( float d, Scope roof, float roof_geom_max_height, bool enable_objects, vec3 rnd )
{
 // roof small border wall
 float dborder = opB_inside( roof.dcc, 0.02 );
 dborder = opI( dborder, opB_inside( -roof.p.z, 0.05 ) );
 // roof ground tiles, perhaps this should be textured but never mind. we get more detail
 float tile_size = 0.05;
 float dtiles = -mincomp( tri_s( roof.p.xy, vec2( tile_size * 0.5 ), vec2( 0.005 ) ) );
 dtiles = opI( roof.dcc + 0.05, dtiles ) + roof.t * 0.004; // add t in spacing so we can see it from far away
 dtiles = opI_round_bevel( dtiles, opB_range( roof.p.z, -0.001, 0.0025 ), 0.00125, 0.75 );
 d = opU( d, opU( dtiles, dborder ) );
 if ( enable_objects )
 {
  // parametric model for small features on roof
  Scope roof_object_scope = roof;
  float droof_object = /* FLT_MAX */1000000.;
  droof_object = sd_RoofTopObject2( d, roof_object_scope, roof_geom_max_height, rnd );
//		droof_object = opI( droof_object, -roof.p.z ); // cut all the bits below roof object level
  d = opU( d, droof_object );
 }
 return d;
}

// Function 229
vec2 	UIStyle_FontPadding() 			{ return vec2(8.0, 2.0); }

// Function 230
Dst sdBuilding(vec3 p, vec3 b) {
    
    vec3 q  = p;
    float c = 3.25;
    q.z    = mod(q.z,c)-.5*c;
    
    return Dst(length(max(abs(q)-b,0.0)), 2);
    
}

// Function 231
void UI_ProcessWindowEditColor( inout UIContext uiContext, inout UIData uiData, int iControlId, int iData )
{
    UIWindowDesc desc;
    
    desc.initialRect = Rect( vec2(256, 48), vec2(265, 310) );
    desc.bStartMinimized = false;
    desc.uControlFlags = WINDOW_CONTROL_FLAG_TITLE_BAR | WINDOW_CONTROL_FLAG_CLOSE_BOX;
    desc.vMaxSize = vec2(100000.0);

    UIWindowState window = UI_ProcessWindowCommonBegin( uiContext, iControlId, iData, desc );
    
    // Controls...
    if ( !window.bMinimized )
    {    
		UILayout uiLayout = UILayout_Reset();
        
        LayoutStyle style;
        RenderStyle renderStyle;             
        UIStyle_GetFontStyleWindowText( style, renderStyle );
                
		UILayout_StackControlRect( uiLayout, UIStyle_ColorPickerSize().xy );                
        UI_ProcessColorPickerSV( uiContext, IDC_COLOR_PICKER, uiData.backgroundColor, uiLayout.controlRect );
        UILayout_StackRight( uiLayout );
		UILayout_StackControlRect( uiLayout, UIStyle_ColorPickerSize().zy );        
        UI_ProcessColorPickerH( uiContext, IDC_COLOR_PICKER+1000, uiData.backgroundColor, uiLayout.controlRect );
        UILayout_StackDown( uiLayout );        
        
        {
            PrintState state = UI_PrintState_Init( uiContext, style, uiLayout.vCursor );        

            vec3 vRGB = hsv2rgb(uiData.backgroundColor.vHSV);
            PrintRGB( state, style, vRGB );
                
            UI_RenderFont( uiContext, state, style, renderStyle );
                        
			UILayout_SetControlRectFromText( uiLayout, state, style );
	        UILayout_StackDown( uiLayout );            
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
    }
    
}

// Function 232
float ShoulderButtons(in vec3 p, float controllBase, vec3 oPos, float oScale)
{
  float d = sdCappedCylinder(p-oPos-vec3(1.45, -0.1, 0.10), vec2(1.0, 0.12)*oScale);
  d = min(d, sdCappedCylinder(p-oPos-vec3(-1.45, -0.1, 0.10), vec2(1.0, 0.12)*oScale));
  d =  min(d, sdBox(p-oPos-vec3(.0, -0.1, 0.2), vec3(1.60, .08, 0.8)*oScale));

  d=  max(d, -sdBox(p-oPos-vec3(.0, 0.0, 0.42), vec3(1.2, 1.08, 2.9)*oScale));
  d=  max(d, -sdBox(p-oPos-vec3(2.50, 0.0, 0.42), vec3(0.4, 1.08, 2.9)*oScale));
  d=  max(d, -sdBox(p-oPos-vec3(-2.50, 0.0, 0.42), vec3(0.4, 1.08, 2.9)*oScale));

  d=  max(d, -sdBox(p-oPos-vec3(0, 0.0, -0.42), vec3(3.0, 0.68, 1.)*oScale));

  return d;
}

// Function 233
vec3 material_builder(vec3 p, vec3 rd, float dis, int id) {
    vec3 n = normal(p);
    if(id == 0) { // Sphere        
		vec3 col = vec3(0.05);
		
        col = mix(col, tex3D(iChannel0, p / 16., n), 1.);
        return col;
    }
    else if(id == 1) { // Plane
        vec3 col = vec3(0.5);
        col = mix(col, tex3D(iChannel1, p/32., n), 1.);
        col = mix(col, vec3(0.0, 1.1, 1.5) * rd.y, fog_exp2(dis, .03) );
        return col;
    }
    else if(id == 2) { // Sky
    	return vec3(0.2, 1.1, 1.4)*rd.y;
    }
}

// Function 234
ParametricBuildingRetval sd_ParametricBuilding( float t, vec3 p, float building_type, bounds2 b2, float height, NearestHighwayRetval nh, vec3 rnd )
{
 if ( building_type == 0. ) return sd_House( p, t, b2, min( height, 2. ), nh );
 return sd_Building( t, p, b2, height, nh, rnd );
}

// Function 235
void paintMainButton(inout vec4 dest, in vec2 uv, in bool isOpen)
{
	bool isHot = false;
    bool isAlive = isHot && false;
    
    float aspect = iResolution.x / iResolution.y;
    
    float button_scale = theme.menuSize;
    
    float t = 1000.0;
    
    const int N = 3;
    for (int i = 0; i < N; ++i)
    {
        vec2 uv0 = uv - vec2(0, 0.02 * button_scale);//(isOpen ? vec2(uv.x, 0.5*button_scale - uv.y) : uv) - vec2(0, 0.2*button_scale);
        uv0.y += 0.7 * button_scale * ((float(i) + 0.5) / float(N));
        uv0.x *= aspect;
    	uv0 = (uv0 - vec2(aspect,1) + button_scale * .5);
        
        uv0 = (isOpen ? vec2(uv0.x, 0.5*button_scale - uv0.y) : uv0) - vec2(0, 0.2*button_scale);
	    uv0 = rotate(radians(60.)) * vec2(abs(uv0.x), uv0.y);

        float t0 = sdBox(uv0, vec2(button_scale*.35));
    	float t1 = max(abs(uv0.x), t0) - .025*button_scale;//max(min(abs(uv.x), abs(uv.y)), t0) - .025*button_scale;
    	t = min(t, t1);
    }    
    t *= iResolution.y;
    
    float a = smoothstep(1.5, 0.0, t);
    //float ao = exp(-2.0 * t) * a;
    
    vec4 color = isAlive ? UI_ALIVE_COLOR : (isHot ? UI_HOT_COLOR : UI_DEAD_COLOR);
    
    //dest.rgb *= (1.0 - 0.9 * ao);
    dest = mix(dest, color, a);
}

// Function 236
vec2 printUInt(vec2 p, uint n) {
    uint rev=0u, digits=0u;
    // reverse digits
    for(uint fwd=n; fwd > 0u; fwd/=10u, ++digits)
        rev = 10u * rev + fwd % 10u;
    digits = max(1u,digits);
    
    // can now print left to right
    for(uint i=0u; i < digits; ++i, rev = rev/10u)
        p = char(p, int(48u + rev % 10u));
    
    return p;
}

// Function 237
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

// Function 238
void UI_DrawCheckbox( inout UIContext uiContext, bool bActive, bool bMouseOver, bool bChecked, Rect checkBoxRect )
{
	if (!uiContext.bPixelInView || Outside( uiContext.vPixelCanvasPos, checkBoxRect ))
        return;
    
    uiContext.vWindowOutColor = vec4(1.0);
    
    if ( bActive && bMouseOver )
    {
        uiContext.vWindowOutColor = vec4(0.85,0.85,0.85,1.0);
    }

    DrawBorderIndent( uiContext.vPixelCanvasPos, checkBoxRect, uiContext.vWindowOutColor );

    Rect smallerRect = checkBoxRect;
    RectShrink( smallerRect, vec2(6.0));

    if ( bChecked )
    {
        vec4 vCheckColor = vec4(0.0, 0.0, 0.0, 1.0);
        DrawLine( uiContext.vPixelCanvasPos, smallerRect.vPos+ smallerRect.vSize * vec2(0.0, 0.75), smallerRect.vPos+ smallerRect.vSize * vec2(0.25, 1.0), 2.0f, vCheckColor, uiContext.vWindowOutColor );
        DrawLine( uiContext.vPixelCanvasPos, smallerRect.vPos+ smallerRect.vSize * vec2(0.25, 1.0), smallerRect.vPos+ smallerRect.vSize * vec2(1.0, 0.25), 2.0f, vCheckColor, uiContext.vWindowOutColor );
    }
}

// Function 239
UIData UI_GetControlData()
{
    UIData data;
    
    data.animate = UI_GetDataBool( DATA_ANIMATE, false );
    
    data.rotX = UI_GetDataValue( DATA_ROT_X, 4.3, 0.0, 5.0 );
    data.rotY = UI_GetDataValue( DATA_ROT_Y, 0.2, 0.0, 5.0 );
    data.intensity = UI_GetDataValue( DATA_INTENSITY, 1.0, 0.0, 5.0 );
    data.exposure = UI_GetDataValue( DATA_EXPOSURE, 0.0, -6.0, 6.0 );    
    
    data.editWhichColor = UI_GetDataValue( DATA_EDIT_WHICH_COLOR, -1.0, 0.0, 100.0 );
    data.backgroundColor = UI_GetDataColor( DATA_BACKGROUND_COLOR, vec3(0.1, 0.5, 1.0) );
    
    return data;
}

// Function 240
uvec4 asuint2(vec4 x) { return uvec4(asuint2(x.xy), asuint2(x.zw)); }

// Function 241
void UI_Compose( vec2 fragCoord, inout vec3 vColor, out int windowId, out vec2 vWindowCoord, out float fShadow )
{
    vec4 vUISample = texelFetch( iChannelUI, ivec2(fragCoord), 0 );
    
    if ( fragCoord.y < 2.0 )
    {
        // Hide data
        vUISample = vec4(1.0, 1.0, 1.0, 1.0);
    }
    
    vColor.rgb = vColor.rgb * (1.0f - vUISample.w) + vUISample.rgb;    
    
    windowId = -1;
    vWindowCoord = vec2(0);
    
    fShadow = 1.0f;
    if ( vUISample.a < 0.0 )
    {
        vWindowCoord = vUISample.rg;
        windowId = int(round(vUISample.b));
        
        fShadow = clamp( -vUISample.a - 1.0, 0.0, 1.0);
    }
}

// Function 242
float UISlider(int id)
{
    return texture(iChannel0, vec2(float(id) + 0.5, 0.5) / iResolution.xy).r;
}

// Function 243
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

// Function 244
ParametricBuildingRetval sd_Building( float t, vec3 p, bounds2 b, float h, NearestHighwayRetval nh, vec3 rnd )
{
 float roof_geom_max_height = 0.3; // a rough but conservative estimate of roof geometry height
 h -= roof_geom_max_height; // fixme: hack: make sure h is bounding height instead
 // some building base scope
 Scope base;
 base.p = p;
 base.b = b;
 base.dcc = sd_bounds( base.p.xy, base.b );
 base.t = t;
 float d = -/* FLT_MAX */1000000.; // opI( base.p.z - h, base.dcc ) is the base block, if you want to visualize it for debug
 vec3 windr = vec3( /* FLT_MAX */1000000. );
 {
  // front facade
  Scope facade = getScopeFacadeX( base, h, 3 );
  vec2 d1 = sd_SurfaceFacade0( facade.p, facade.p.z );
  d = opI( d1.x, d );
  windr = vec3( d1.y, vec2( 1., 0. ) );
 }
 {
  // back facade with flat windows
  Scope facade = getScopeFacadeY( base, h, 3 );
  float d2 = sd_SurfaceFacade7( facade.p, facade.p.z );
  d = opI( d2, d );
 }
 if ( nh.d != /* FLT_MAX */1000000. )
 {
  // highway facing facade using nearest highway clip plane
  float d10 = dot( p.xy - nh.o_clip, nh.n_clip ) - (1.) * 1.3;
  vec3 pr = vec3( dot( p.xy, perp( nh.n_clip ) ), p.z, -d10 ); // pr is the point in 2d facade space
  vec2 d3 = sd_SurfaceFacade0( pr, pr.z );
  d = opI( d3.x, d );
  // pick this window if it's closest
  windr = mix( windr, vec3( d3.y, nh.n_clip ), step( abs( d3.y ), abs( windr.x ) ) );
 }
 float droof = d; // save d before we clamped base on h, that will give us a consistent base dcc to build the roof on
 d = opI( d, base.p.z - h );
 {
  Scope roof = base;
  roof.p.z -= h;
  roof.dcc = droof;
  d = sd_JPBuildingRoofTopWithObjects( d, roof, roof_geom_max_height, true, rnd ); // roof with border + objects on it
 }
 ParametricBuildingRetval ret;
 ret.d = d;
 ret.windr = windr;
 ret.droof = /* FLT_MAX */1000000.;
 return ret;
}

// Function 245
void rayMarchParametricBuilding( Ray ray2, bool permuted, vec2 ri, bounds3 bi, float kk, inout ParametricBuildingHit hit, vec2 cell_index )
{
 if ( ri.y <= ri.x || ri.y < 0. ) return; // no hit
 ri.x = max( ri.x, 0. );
 // warning: if you fiddle with early return here, check what happens to the shadow term!
 bounds2 b2 = mkbounds_unchecked( bi.pmin.xy, bi.pmax.xy ); // b2 is in maybe permuted space and contains the object
 vec2 base_size = size( b2 ); // this is the base size with x and y maybe permuted
//	if ( base_size.x <= 0. || base_size.y <= 0. ) return; // doesn't seem to contribute to the image... and is slower?
 NearestHighwayRetval nh;
 nh.d = /* FLT_MAX */1000000.; // disable the clip plane
 {
  // some basic layout intersection tests outside marching loop... tedious block of code
  vec2 b2c = center( b2 );
  vec2 b2s = base_size; // size( b2 )
  if ( permuted ) // remember: b2 is in permuted space, dont use it in calculations below
  {
   b2c = b2c.yx; // back to world
   b2s = b2s.yx;
  }
  nh = evalHighway( b2c );
  nh.o_clip = nh.p + nh.d2f.xy; // closest highway center line point (b2c==nh.p)
  nh.n_clip = normalize( perp( nh.d2f.zw ) ); // can't use hret2.d2f.xy as it may be null
//		float b2cd = dot( b2c - nh.o_clip, nh.n_clip ); // distance to higway center line, signed
  float b2cd = dot( -nh.d2f.xy, nh.n_clip ); // should be same as above, saves a sub (lol)
  nh.n_clip *= sign( b2cd ); // orient the clip normal so we can build a facade (box center is on positive side)
  b2cd = abs( b2cd ); // now we are on the positive side
  float ml = length( b2s );
  if ( b2cd > ( ml + ((1.)*2.) ) * 0.5 ) nh.d = /* FLT_MAX */1000000.; // building far away enough regardless of rotation, disable the clip plane
  else
  {
   float l = b2s.y; // (fixme: corrected with slope) if we can assume horizonal ish roads
   float clipped_l = ( b2cd + l * 0.5 ) - max( b2cd - l * 0.5, (1.) ); // size left after clipping
   base_size = b2s;
   base_size.y = clipped_l;
   // if the building was ultra thin to begin with, clipping is not going to make things any better
   // that filters out some garbage thin buildings
   if ( maxcomp( base_size ) > 10. * mincomp( base_size ) ) return;
  }
//		nh.d = FLT_MAX; // uncomment to check actual size of building if they weren't clipped
  if ( permuted )
  {
   // back to permuted space
   nh.p = nh.p.yx;
   nh.pr = nh.pr.yx;
   nh.d2f = nh.d2f.yxwz;
   nh.n_clip = nh.n_clip.yx;
   nh.o_clip = nh.o_clip.yx;
  }
 }
 float height = bi.pmax.z; // pmax.z awkward?
 // make height not bigger than n times the smallest dimension on the 2d base, not that base_size may be permuted, we only care about the min dimension
 height = min( height, 8. * mincomp( base_size ) );
 float building_type = height < 3.4 ? 0. : 1.;
 if ( building_type == 1. ) height = ( 0.5 + floor( height / /*FLOOR_HEIGHT*/0.6 - 0.5 ) ) * /*FLOOR_HEIGHT*/0.6; // make building height a multiple of floor height
 float t = ri.x; // start marching from first hit point
 vec3 rnd = vec3( cell_index, kk );
 for ( int j = 0; j < 70 /*FORCE_LOOP*/+min(0,iFrame); ++j )
 {
  // no need to trace further than max cell size == massive win
  // we then have to pick a max trace distance, for that cell 2d diagonal size would be a start
  // but since cell height is higher than max building height, we use max building height instead
  if ( t - ri.x > /* MAX_BUILDING_HEIGHT */14.3 ) break;
  vec3 p = ray2.o + t * ray2.d;
  ParametricBuildingRetval ddd = sd_ParametricBuilding( t, p, building_type, b2, height, nh, rnd );
  float d = ddd.d;
  if ( abs( d ) <= 0.001 * t )
  {
   if ( t < hit.t ) // we need to check vs other objects in the cell
   {
    // record a few things we need to do extra evals deriving from the final hit
    hit.t = t;
    hit.tile_child_index = kk;
    hit.building_type = building_type;
    hit.b2 = b2;
    hit.height = height;
    hit.d = d;
    hit.windr = ddd.windr;
    hit.is_roof = step( abs( d - ddd.droof ), 0.001 );
    hit.ray2 = ray2;
    hit.permuted = permuted;
    hit.nh = nh;
    hit.rnd = rnd;
   }
   break; // "return" is slower on radeon: 29ms -> 31ms (ancient wip timings)
  }
  float dt = d;
//		float dt = d * TFRAC; // shadows a bit better with this
//		float dt = abs( d ); // *TFRAC // only move forward (see inside of buildings...)
  t += dt;
//		p += dt * ray2.d; // do not do this, instead increment t and reevaluate p fully (loss of precision else)
 }
}

// Function 246
void UI_StoreControlData( inout UIContext uiContext, UIData data )
{
    UI_StoreDataBool( uiContext, data.animate, DATA_ANIMATE );

    UI_StoreDataValue( uiContext, data.rotX, DATA_ROT_X );
    UI_StoreDataValue( uiContext, data.rotY, DATA_ROT_Y );
    UI_StoreDataValue( uiContext, data.intensity, DATA_INTENSITY );
    UI_StoreDataValue( uiContext, data.exposure, DATA_EXPOSURE );

    UI_StoreDataValue( uiContext, data.editWhichColor, DATA_EDIT_WHICH_COLOR );
    UI_StoreDataColor( uiContext, data.backgroundColor, DATA_BACKGROUND_COLOR );
}

// Function 247
vec4 drawUI()
{   
  	SETUP_UI;
    
    vec2 centerPt = vec2(floor(RES_X*0.5), floor(RES_Y*0.5));
    
    // Show controls
    if(time_scale == 0.0 && iFrame > 0)
    {
        PLOT(fRect(centerPt, vec2(50, 25)));
        COLOR(BLACK);
        DRAW;    
        PLOT(fRect(centerPt-vec2(0,26), vec2(51, 0)));
        COLOR(GRAY);
        DRAW;
        PLOT(fRect(centerPt+vec2(0,26), vec2(51, 0)));
        COLOR(WHITE);
        DRAW;
        PLOT(fRect(centerPt-vec2(51,0), vec2(0, 25)));
        COLOR(WHITE);
        DRAW;
        PLOT(fRect(centerPt+vec2(51,0), vec2(0, 25)));
        COLOR(GRAY);
        DRAW;
        
        // CONTROLS
        PLOT(fRect(centerPt+vec2(0,16), vec2(16, 0)));
        COLOR(WHITE);
        DRAW;
        STR(44,54) C(_C)C(_O)C(_N)C(_T)C(_R)C(_O)C(_L)C(_S)
        
        // WSAD/ARROWS PITCH+ROLL
        STR(11,43) C(_W)C(_S)C(_A)C(_D)S(_SLASH)C(_A)C(_R)C(_R)C(_O)C(_W)C(_S)
        STR(69,43) C(_P)C(_I)C(_T)C(_C)C(_H)S(_PLUS)C(_R)C(_O)C(_L)C(_L)
        
        // Q+E YAW
        STR(11,36) C(_Q)S(_PLUS)C(_E)
        STR(69,36) C(_Y)C(_A)C(_W)
        
        // J+K THROTTLE
        STR(11,29) C(_J)S(_PLUS)C(_K)
        STR(69,29) C(_T)C(_H)C(_R)C(_O)C(_T)C(_T)C(_L)C(_E)
        
        // H FANTASTIC!
        STR(11,22) C(_H)
        STR(69,22) C(_F)C(_A)C(_N)C(_T)C(_A)C(_S)C(_T)C(_I)C(_C)S(_EXCLAM)
        
        // P UNPAUSE
        STR(11,15) C(_P)
        STR(69,15) C(_U)C(_N)C(_P)C(_A)C(_U)C(_S)C(_E)
        
        // DIVIDERS
        STR(59,43) S(_HYPHEN)
        STR(59,36) S(_HYPHEN)
        STR(59,29) S(_HYPHEN)
        STR(59,22) S(_HYPHEN)
        STR(59,15) S(_HYPHEN)
        COLOR(WHITE);
        DRAW;
    } else
    {
        // Logo
        if(gtime < 16.0)
        {
            int logoX = int(RES_X)-18;
            PLOT(fRect(vec2(logoX,4), vec2(10, 3)));
            COLOR(BLACK);
            FADE_IN(2.0, 3.0);
            FADE_OUT(8.0, 10.0);
            DRAW;

            // LAIKA
            STR(logoX-7,6) C(_L)C(_A)C(_I)C(_K)C(_A)
            COLOR(WHITE);
            FADE_IN(2.0, 3.0);
            FADE_OUT(8.0, 10.0);
            DRAW;
            
            PLOT(fRect(vec2(37,4), vec2(32, 3)));
            COLOR(BLACK);
            FADE_IN(10.0, 11.0);
            FADE_OUT(14.0, 16.0);
            DRAW;
            
			// PRESS P FOR HELP
            STR(8,6) C(_P)C(_R)C(_E)C(_S)C(_S) SPACE C(_P) SPACE C(_F)C(_O)C(_R) SPACE C(_H)C(_E)C(_L)C(_P)
            COLOR(WHITE);
            FADE_IN(10.0, 11.0);
            FADE_OUT(14.0, 16.0);
            DRAW;
        }
        
    	// Crosshairs
    	PLOT(PlotPoint(centerPt.x, centerPt.y+2.) + PlotPoint(centerPt.x, centerPt.y-2.) +
        	 PlotPoint(centerPt.x+2., centerPt.y) + PlotPoint(centerPt.x-2., centerPt.y));
    	COLOR(WHITE);
    	DRAW;
    }
    
    return vec4(drawColor, inPix);
}

// Function 248
void UI_DrawButton( inout UIContext uiContext, bool bActive, bool bMouseOver, Rect buttonRect )
{
	if (!uiContext.bPixelInView)
        return;
    
    if ( bActive && bMouseOver )
    {
#ifdef NEW_THEME
    	DrawBorderRect( uiContext.vPixelCanvasPos, buttonRect, cButtonActive, uiContext.vWindowOutColor );
#else
    	DrawBorderIndent( uiContext.vPixelCanvasPos, buttonRect, uiContext.vWindowOutColor );
#endif        
    }
    else
    {
#ifdef NEW_THEME
    	DrawBorderRect( uiContext.vPixelCanvasPos, buttonRect, cButtonInactive, uiContext.vWindowOutColor );
#else
    	DrawBorderOutdent( uiContext.vPixelCanvasPos, buttonRect, uiContext.vWindowOutColor );
#endif        
    }
}

// Function 249
Element init_circuit(Element e, vec2 p) {

    float y = 0.0;
    e=mova(1.,3.,1., 1.,y,e,p);e=mova(1.,3.,1., 2.,y,e,p);e=mova(1.,2.,0., 3.,y,e,p);e=moa(0.,0., 4.,y,e,p);e=moa(0.,0., 5.,y,e,p);e=moa(0.,0., 6.,y,e,p);e=moa(0.,0., 7.,y,e,p);e=moa(0.,0., 8.,y,e,p);e=moa(0.,0., 9.,y,e,p);
    
    y = 1.;
    e=mova(1.,1.,1., 1.,y,e,p);e=mova(1.,0.,0., 2.,y,e,p);e=mova(1.,0.,1., 3.,y,e,p);e=moa(0.,0., 4.,y,e,p);e=moa(0.,0., 5.,y,e,p);e=moa(0.,0., 6.,y,e,p);e=moa(0.,0., 7.,y,e,p);e=moa(0.,0., 8.,y,e,p);e=moa(0.,0., 9.,y,e,p);
    
    y = 2.;
    e=mova(1.,3.,1., 1.,y,e,p);e=mova(1.,3.,1., 2.,y,e,p);e=mova(1.,2.,0., 3.,y,e,p);e=moa(0.,0., 4.,y,e,p);e=moa(0.,0., 5.,y,e,p);e=moa(0.,0., 6.,y,e,p);e=moa(0.,0., 7.,y,e,p);e=moa(0.,0., 8.,y,e,p);e=moa(0.,0., 9.,y,e,p);
    
    y = 3.;
    e=mova(1.,1.,1., 1.,y,e,p);e=mova(1.,0.,0., 2.,y,e,p);e=mova(1.,0.,1., 3.,y,e,p);e=moa(0.,0., 4.,y,e,p);e=moa(0.,0., 5.,y,e,p);e=moa(0.,0., 6.,y,e,p);e=moa(0.,0., 7.,y,e,p);e=moa(0.,0., 8.,y,e,p);e=moa(0.,0., 9.,y,e,p);
    
    y = 4.;
    e=mova(1.,3.,1., 1.,y,e,p);e=mova(1.,3.,1., 2.,y,e,p);e=mova(1.,2.,0., 3.,y,e,p);e=moa(0.,0., 4.,y,e,p);e=moa(0.,0., 5.,y,e,p);e=moa(0.,0., 6.,y,e,p);e=moa(0.,0., 7.,y,e,p);e=moa(0.,0., 8.,y,e,p);e=moa(0.,0., 9.,y,e,p);
    
    y = 5.;
    e=mova(1.,1.,1., 1.,y,e,p);e=mova(1.,0.,0., 2.,y,e,p);e=mova(1.,0.,1., 3.,y,e,p);e=moa(0.,0., 4.,y,e,p);e=moa(0.,0., 5.,y,e,p);e=moa(0.,0., 6.,y,e,p);e=moa(0.,0., 7.,y,e,p);e=moa(0.,0., 8.,y,e,p);e=moa(0.,0., 9.,y,e,p);
    
    y = 6.;
    e=mova(1.,3.,1., 1.,y,e,p);e=mova(1.,3.,1., 2.,y,e,p);e=mova(1.,2.,0., 3.,y,e,p);e=moa(0.,0., 4.,y,e,p);e=moa(0.,0., 5.,y,e,p);e=moa(0.,0., 6.,y,e,p);e=moa(0.,0., 7.,y,e,p);e=moa(0.,0., 8.,y,e,p);e=moa(0.,0., 9.,y,e,p);
    
    y = 7.;
    e=mova(1.,1.,1., 1.,y,e,p);e=mova(1.,0.,0., 2.,y,e,p);e=mova(1.,0.,1., 3.,y,e,p);e=moa(0.,0., 4.,y,e,p);e=moa(0.,0., 5.,y,e,p);e=moa(0.,0., 6.,y,e,p);e=moa(0.,0., 7.,y,e,p);e=moa(0.,0., 8.,y,e,p);e=moa(0.,0., 9.,y,e,p);
    
    y = 8.;
    e=moa(0.,0., 1.,y,e,p);    e=moa(0.,0., 2.,y,e,p);     e=moa(0.,0., 3.,y,e,p);e=moa(0.,0., 4.,y,e,p);e=moa(0.,0., 5.,y,e,p);e=moa(0.,0., 6.,y,e,p);e=moa(0.,0., 7.,y,e,p);e=moa(0.,0., 8.,y,e,p);e=moa(0.,0., 9.,y,e,p);

    
    return e;
}

// Function 250
vec4 drawGui(vec2 c) {
	float scale = floor(iResolution.y / 128.);
    c /= scale;
    vec2 r = iResolution.xy / scale;
    vec4 o = vec4(0);
    float xStart = (r.x - 16. * numItems) / 2.;
    c.x -= xStart;
    float selected = load(_selectedInventory).r;
    vec2 p = (fract(c / 16.) - .5) * 3.;
    vec2 u = vec2(sqrt(3.)/2.,.5);
    vec2 v = vec2(-sqrt(3.)/2.,.5);
    vec2 w = vec2(0,-1);
    if (c.x < numItems * 16. && c.x >= 0. && c.y < 16.) {
        float slot = floor(c.x / 16.);
    	o = getTexture(48., fract(c / 16.));
        vec3 b = vec3(dot(p,u), dot(p,v), dot(p,w));
        vec2 texCoord;
        //if (all(lessThan(b, vec3(1)))) o = vec4(dot(p,u), dot(p,v), dot(p,w),1.);
        float top = 0.;
        float right = 0.;
        if (b.z < b.x && b.z < b.y) {
        	texCoord = inv2(mat2(u,v)) * p.xy;
            top = 1.;
        }
        else if(b.x < b.y) {
        	texCoord = 1. - inv2(mat2(v,w)) * p.xy;
            right = 1.;
        }
        else {
        	texCoord = inv2(mat2(u,w)) * p.xy;
            texCoord.y = 1. - texCoord.y;
        }
        if (all(lessThanEqual(abs(texCoord - .5), vec2(.5)))) {
            float id = getInventory(slot);
            if (id == 3.) id += top;
            o.rgb = getTexture(id, texCoord).rgb * (0.5 + 0.25 * right + 0.5 * top);
            o.a = 1.;
        }
    }
    vec4 selection = drawSelectionBox(c - 8. - vec2(16. * selected, 0));
    o = mix(o, selection, selection.a);
    return o;
}

// Function 251
vec4 iDiegeticUIshow(vec2 u){
 ;vec4 c=vec4(0),s=bufDrag(statePos)
 ;vec2 e=vec2(0.,UiDotRadius+UiDotBorder)
 //e.x is the inner bound of a dot's black border
 //e.y is the outer bound of a dot's black border 
 ;drawDragDots(s,u,e)
 ;if(e.y<UiDotRadius+UiDotBorder
 ){e.y-=UiDotRadius
  ;     c=vec4(0,0,0,1)  *smoothstep( UiDotBorder,0.         ,(abs(-e.y)))
  ;vec4 d=dotColor(s,e.x)*smoothstep(-UiDotBorder,UiDotBorder,     -e.y)
  ;c=pdOver(c,d)*.4
 ;}//else return vec4(1,0,0,1)//to debug above boundary
 ;if(inRect(u,deleteRect))c.xyz=mix(c.xyz,vec3(1,0,0),.3)
 ;return c;}

// Function 252
UIContext UI_GetContext( vec2 fragCoord, int iData )
{
    UIContext uiContext;
    
    uiContext.vPixelPos = fragCoord;
    uiContext.vPixelPos.y = iResolution.y - uiContext.vPixelPos.y;
    uiContext.vMousePos = iMouse.xy;
    uiContext.vMousePos.y = iResolution.y - uiContext.vMousePos.y;
    uiContext.bMouseDown = iMouse.z > 0.0;       
    
    vec4 vData0 = LoadVec4( iChannelUI, ivec2(iData,0) );
    
    uiContext.bMouseWasDown = (vData0.x > 0.0);
    
    uiContext.vFragCoord = ivec2(fragCoord);
    uiContext.vOutColor = vec4(0.0);
#ifdef SHADOW_TEST    
    uiContext.fShadow = 1.0;
    uiContext.fOutShadow = 1.0f;
#endif    
    uiContext.fBlendRemaining = 1.0;
    
    uiContext.vOutData = vec4(0.0);
    if ( int(uiContext.vFragCoord.y) < 2 )
    {
        // Initialize data with previous value
	    uiContext.vOutData = texelFetch( iChannelUI, uiContext.vFragCoord, 0 );     
    }
    uiContext.bHandledClick = false;
    
    uiContext.iActiveControl = int(vData0.y);
    uiContext.vActivePos = vec2(vData0.zw);
        
    
    UIDrawContext rootContext;
    
    rootContext.vCanvasSize = iResolution.xy;
    rootContext.vOffset = vec2(0);
    rootContext.viewport = Rect( vec2(0), vec2(iResolution.xy) );
    rootContext.clip = rootContext.viewport;

    UI_SetDrawContext( uiContext, rootContext );
    
    uiContext.vWindowOutColor = vec4(0);    
        
    if ( iFrame == 0 )
    {
        uiContext.bMouseWasDown = false;
        uiContext.iActiveControl = IDC_NONE;
    }
    
    return uiContext;
}

// Function 253
vec4 gui_check_box(vec4 col, vec2 uv, vec2 pos, float scale, bool check)
{
    
    float unit = asp * 0.01 * scale;
    float h = box(uv, pos, vec2(1.8*unit));
    col = mix(col, vec4(vec3(0.9, 0.9, 1.), 1.), smoothstep(0.01, 0., h));
    col = mix(col, vec4(vec3(0., 0., 0.5), 1.), smoothstep(0.01, 0., abs(h)));
    
    
    if(check)
    {
        const vec2 dir1 = normalize(vec2(-1., 1.2));
        const vec2 dir2 = normalize(vec2(1., 1.));
    	h = line(uv, pos+vec2(0., unit*-0.5), pos+dir1*unit*2.1);
        col = mix(col, vec4(1., 0., 0., 1.), smoothstep(0.01, 0., abs(h)));
        h = line(uv, pos+vec2(0., unit*-0.5), pos+dir2*unit*4.2);
        col = mix(col, vec4(1., 0., 0., 1.), smoothstep(0.01, 0., abs(h)));
    }
    
    return col;
}

// Function 254
float 	UIStyle_ScrollBarSize() 		{ return 24.0; }

// Function 255
vec4 fruitTexture(vec3 p, vec3 nor, float i)
{
    
    
    float rand = texCube(iChannel2, p*.1 ,nor).x;
    float t = dot(nor, normalize(vec3(.8, .1, .1)));
	vec3 mat = vec3(1.,abs(t)*rand,0);
    mat = mix(vec3(0,1,0), mat, i/10.);

   	return vec4(mat, .5);
}

// Function 256
vec2 buttonPos(vec2 b){
 if(agtmf)return PushedNotOnButton;//error code for "mouse button down, ouside of pip-buttons.
 //when you want to know what button ON a pip-keyboard was pressed.
 return floor(frame(b));}

// Function 257
float uintBitsToFloat01(uint x) {
    return uintBitsToFloat((x >> 9u) | 0x3f800000u) - 1.0;
}

// Function 258
void RenderUI(inout vec4 fragColor, vec2 fragCoord, vec2 uv, float aspect)
{
    uv *= 128.0;
       
    // Render CROSSY PENGUIN logo upon the title screen or during the first
    // seconds of the game (with an animation!).
    
    if (gGameState == kStateTitle || (gGameState == kStateInGame && gGameStateTime < 4.0))
    {
        float logoSdf = kOmega;        
        vec2 logoUv = uv * 0.5;
        logoUv.x -= (gGameState == kStateInGame)? 500.0 * gGameStateTime * gGameStateTime : 0.0;
        logoUv.y += uv.x * 0.13;
                
        vec2 subLogoUv = logoUv * 3.0;
        
        gCharPrintPos = vec2(-34.0, 7.0);
        CharC(logoUv, logoSdf);
        CharR(logoUv, logoSdf);
        CharO(logoUv, logoSdf);
        CharS(logoUv, logoSdf);
        CharS(logoUv, logoSdf);
        CharY(logoUv, logoSdf);
        gCharPrintPos = vec2(-38.0, -7.0);
        CharP(logoUv, logoSdf);
        CharE(logoUv, logoSdf);
        CharN(logoUv, logoSdf);
        CharG(logoUv, logoSdf);
        CharU(logoUv, logoSdf);
        CharI(logoUv, logoSdf);
        CharN(logoUv, logoSdf);
        gCharPrintPos = vec2(-132.0, -40.0);
        CharA(subLogoUv, logoSdf);
        CharSpace();
        CharC(subLogoUv, logoSdf);
        CharR(subLogoUv, logoSdf);
        CharO(subLogoUv, logoSdf);
        CharS(subLogoUv, logoSdf);
        CharS(subLogoUv, logoSdf);
        CharY(subLogoUv, logoSdf);
        CharSpace();
        CharR(subLogoUv, logoSdf);
        CharO(subLogoUv, logoSdf);
        CharA(subLogoUv, logoSdf);
        CharD(subLogoUv, logoSdf);
        CharSpace();
        CharT(subLogoUv, logoSdf);
        CharR(subLogoUv, logoSdf);
        CharI(subLogoUv, logoSdf);
        CharB(subLogoUv, logoSdf);
        CharU(subLogoUv, logoSdf);
        CharT(subLogoUv, logoSdf);
        CharE(subLogoUv, logoSdf);
        
        fragColor.rgb = mix(fragColor.rgb, vec3(0.0), step(-3.0, -logoSdf));
        fragColor.rgb = mix(fragColor.rgb, vec3(1.0), step(0.0, -logoSdf));
    }
    
    // Render the score during the InGame state and hide on GameOver.
    
    if (gGameState == kStateInGame || (gGameState == kStateGameOver && gGameStateTime < 4.0))
    {
        float scoreSdf = kOmega;
        vec2 scoreUv = uv;
        
        if (gGameState == kStateInGame) scoreUv.y -= 64.0 - 64.0 * min(1.0, gGameStateTime);
        else                            scoreUv.y -= 64.0 * min(1.0, gGameStateTime);
        
        gCharPrintPos = vec2(-120.0 * aspect, 104.0);
        Print(scoreUv, gScore, scoreSdf);
        
        fragColor.rgb = mix(fragColor.rgb, vec3(0.0), step(-3.0, -scoreSdf));
        fragColor.rgb = mix(fragColor.rgb, vec3(1.0), step(0.0, -scoreSdf));
    }
    
    // Fade to blue upon GameOver and show score.
   
    if (gGameState == kStateGameOver || gGameState == kStateRestarting)
    {
        fragColor.rgb  = mix(fragColor.rgb, vec3(0.2, 0.7, 1.0), min(0.5, gGameStateTime));
        
        float gameOverSdf = kOmega;        
        vec2 gameOverUv = uv * 0.5;
        float gameOverTime = gGameState == kStateRestarting? 1.0 : min(gGameStateTime * 0.7, 1.0);
        
        gameOverUv.y -= 128.0 * abs(cos(2.0 * kPi * gameOverTime * gameOverTime)) * (1.0 - gameOverTime);
                       
        gCharPrintPos = vec2(-45.0, 0.0);
        CharG(gameOverUv, gameOverSdf);
        CharA(gameOverUv, gameOverSdf);
        CharM(gameOverUv, gameOverSdf);
        CharE(gameOverUv, gameOverSdf);
        CharSpace();
        CharO(gameOverUv, gameOverSdf);
        CharV(gameOverUv, gameOverSdf);
        CharE(gameOverUv, gameOverSdf);
        CharR(gameOverUv, gameOverSdf);
        
        fragColor.rgb = mix(fragColor.rgb, vec3(0.0), step(-3.0, -gameOverSdf));
        fragColor.rgb = mix(fragColor.rgb, vec3(1.0), step(0.0, -gameOverSdf));
        
        float messageSdf = kOmega;
        vec2 messageUv = uv * 1.5;
        float messageTime = gGameState == kStateRestarting? 1.0 : clamp(gGameStateTime * 0.7 - 1.0, 0.0, 1.0);
        
        messageUv.x -= 1024.0 * (1.0 - messageTime) * (1.0 - messageTime);
                 
        if (gPlayerDeathCause == kBehavWater)
        {
            gCharPrintPos = vec2(-170.0, -40.0);
            CharS(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
            CharM(messageUv, messageSdf);
            CharS(messageUv, messageSdf);
            CharSpace();
            CharL(messageUv, messageSdf);
            CharI(messageUv, messageSdf);
            CharK(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
            CharSpace();
            CharT(messageUv, messageSdf);
            CharH(messageUv, messageSdf);
            CharI(messageUv, messageSdf);
            CharS(messageUv, messageSdf);
            CharSpace();
            CharP(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
            CharN(messageUv, messageSdf);
            CharG(messageUv, messageSdf);
            CharU(messageUv, messageSdf);
            CharI(messageUv, messageSdf);
            CharN(messageUv, messageSdf);
            CharSpace();
            CharC(messageUv, messageSdf);
            CharA(messageUv, messageSdf);
            CharN(messageUv, messageSdf);
            CharT(messageUv, messageSdf);
            CharSpace();
            CharS(messageUv, messageSdf);
            CharW(messageUv, messageSdf);
            CharI(messageUv, messageSdf);
            CharM(messageUv, messageSdf);
        }
        else if (gPlayerDeathCause == kBehavOutOfScreen)
        {
            gCharPrintPos = vec2(-220.0, -40.0);
            CharY(messageUv, messageSdf);
            CharO(messageUv, messageSdf);
            CharU(messageUv, messageSdf);
            CharSpace();
            CharW(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
            CharN(messageUv, messageSdf);
            CharT(messageUv, messageSdf);
            CharSpace();
            CharO(messageUv, messageSdf);
            CharN(messageUv, messageSdf);
            CharSpace();
            CharA(messageUv, messageSdf);
            CharSpace();
            CharJ(messageUv, messageSdf);
            CharO(messageUv, messageSdf);
            CharU(messageUv, messageSdf);
            CharR(messageUv, messageSdf);
            CharN(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
            CharY(messageUv, messageSdf);
            CharSpace();
            CharT(messageUv, messageSdf);
            CharO(messageUv, messageSdf);
            CharSpace();
            CharF(messageUv, messageSdf);
            CharA(messageUv, messageSdf);
            CharR(messageUv, messageSdf);
            CharSpace();
            CharA(messageUv, messageSdf);
            CharW(messageUv, messageSdf);
            CharA(messageUv, messageSdf);
            CharY(messageUv, messageSdf);
            CharSpace();
            CharL(messageUv, messageSdf);
            CharA(messageUv, messageSdf);
            CharN(messageUv, messageSdf);
            CharD(messageUv, messageSdf);
            CharS(messageUv, messageSdf);
        }
        else if (gPlayerDeathCause == kBehavHazard)
        {
            gCharPrintPos = vec2(-170.0, -40.0);
            CharD(messageUv, messageSdf);
            CharI(messageUv, messageSdf);
            CharD(messageUv, messageSdf);
            CharSpace();
            CharA(messageUv, messageSdf);
            CharN(messageUv, messageSdf);
            CharY(messageUv, messageSdf);
            CharO(messageUv, messageSdf);
            CharN(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
            CharSpace();
            CharC(messageUv, messageSdf);
            CharA(messageUv, messageSdf);
            CharT(messageUv, messageSdf);
            CharC(messageUv, messageSdf);
            CharH(messageUv, messageSdf);
            CharSpace();
            CharT(messageUv, messageSdf);
            CharH(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
            CharSpace();
            CharL(messageUv, messageSdf);
            CharI(messageUv, messageSdf);
            CharC(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
            CharN(messageUv, messageSdf);
            CharS(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
            CharSpace();
            CharP(messageUv, messageSdf);
            CharL(messageUv, messageSdf);
            CharA(messageUv, messageSdf);
            CharT(messageUv, messageSdf);
            CharE(messageUv, messageSdf);
        }
        
        fragColor.rgb = mix(fragColor.rgb, vec3(0.0), step(-3.0, -gameOverSdf));
        fragColor.rgb = mix(fragColor.rgb, vec3(1.0), step(0.0, -gameOverSdf));
        fragColor.rgb = mix(fragColor.rgb, vec3(0.0), step(-3.0, -messageSdf));
        fragColor.rgb = mix(fragColor.rgb, vec3(1.0), step(0.0, -messageSdf));
    }   
    
    // Fade to white upon Restarting
    
    if (gGameState == kStateRestarting)
        fragColor.rgb = mix(mix(fragColor.rgb, vec3(0.2, 0.7, 1.0), 0.5), vec3(1.0), min(1.0, gGameStateTime));
    
    // Fade from white at the title.
    
    if (gGameState == kStateTitle)
    	fragColor.rgb = mix(vec3(1.0), fragColor.rgb, min(1.0, gGameStateTime));
    
    // Press space to continue.
    
    if (step(0.5, fract(iTime * 0.75)) > 0.5 && (gGameState == kStateTitle || gGameState == kStateGameOver) && gGameStateTime > 1.0)
    {
        float messageSdf = kOmega;
        vec2 messageUv = uv * 1.75;
                 
        gCharPrintPos = vec2(-130.0, -100.0);
        CharP(messageUv, messageSdf);
        CharR(messageUv, messageSdf);
        CharE(messageUv, messageSdf);
        CharS(messageUv, messageSdf);
        CharS(messageUv, messageSdf);
        CharSpace();
        CharS(messageUv, messageSdf);
        CharP(messageUv, messageSdf);
        CharA(messageUv, messageSdf);
        CharC(messageUv, messageSdf);
        CharE(messageUv, messageSdf);
        CharSpace();
        CharT(messageUv, messageSdf);
        CharO(messageUv, messageSdf);
        CharSpace();
        CharC(messageUv, messageSdf);
        CharO(messageUv, messageSdf);
        CharN(messageUv, messageSdf);
        CharT(messageUv, messageSdf);
        CharI(messageUv, messageSdf);
        CharN(messageUv, messageSdf);
        CharU(messageUv, messageSdf);
        CharE(messageUv, messageSdf);
        
        fragColor.rgb = mix(fragColor.rgb, vec3(0.0), step(-3.0, -messageSdf));
        fragColor.rgb = mix(fragColor.rgb, vec3(1.0, 1.0, 0.0), step(0.0, -messageSdf));
    }
}

// Function 259
vec2 	UIStyle_FontPadding() 			{ return vec2(6.0, 2.0); }

// Function 260
void UI_Compose( vec2 fragCoord, inout vec3 vColor, out int windowId, out vec2 vWindowCoord, out float fShadow )
{
    vec4 vUISample = texelFetch( iChannelUI, ivec2(fragCoord), 0 );
    
    if ( fragCoord.y < 2.0 )
    {
        // Hide data
        vUISample = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    
    vColor.rgb = vColor.rgb * (1.0f - vUISample.w) + vUISample.rgb;    
    
    windowId = -1;
    vWindowCoord = vec2(0);
    
    fShadow = 1.0f;
    if ( vUISample.a < 0.0 )
    {
        vWindowCoord = vUISample.rg;
        windowId = int(round(vUISample.b));
        
        fShadow = clamp( -vUISample.a - 1.0, 0.0, 1.0);
    }
}

// Function 261
void build_onb( vec3 z, vec3 x0, out vec3 x, out vec3 y ) { y = normalize( cross( x0, z ) ); x = normalize( cross( z, y ) ); }

// Function 262
v3 iDiegeticUIshow(v1 u){
 ;v3 c=v3(0),s=bufDrag(statePos)
 ;v1 e=v1(0.,UiDotRadius+UiDotBorder)
 //e.x is the inner bound of a dot's black border
 //e.y is the outer bound of a dot's black border 
 ;drawDragDots(s,u,e)
 ;if(e.y<UiDotRadius+UiDotBorder
 ){e.y-=UiDotRadius
  ;     c=v3(0,0,0,1)  *smoothstep( UiDotBorder,0.         ,(abs(-e.y)))
  ;v3 d=dotColor(s,e.x)*smoothstep(-UiDotBorder,UiDotBorder,     -e.y)
  ;c=pdOver(c,d)*.4
 ;}//else return v3(1,0,0,1)//to debug above boundary
 ;if(inRect(u,deleteRect))c.xyz=mix(c.xyz,v2(1,0,0),.3)
 ;return c;}

// Function 263
float UI_ButtonAO(in float t)
{
	return 1.0 - .7 * exp(-.5 * max(t,0.));
}

// Function 264
void sampleEquiAngular(
	float u,
	float maxDistance,
	vec3 rayOrigin,
	vec3 rayDir,
	vec3 lightPos,
	out float dist,
	out float pdf)
{
	// get coord of closest point to light along (infinite) ray
	float delta = dot(lightPos - rayOrigin, rayDir);

	// get distance this point is from light
	float D = length(rayOrigin + delta*rayDir - lightPos);

	// get angle of endpoints
	float thetaA = atan(0.0 - delta, D);
	float thetaB = atan(maxDistance - delta, D);

	// take sample
	float t = D*tan(mix(thetaA, thetaB, u));
	dist = delta + t;
	pdf = D/((thetaB - thetaA)*(D*D + t*t));
}

// Function 265
vec4 triUIBox(int idx, float delta) {
    
    return vec4(digitUIBox(idx).xy + vec2(0, 0.9*delta*textSize), 
                0.4*textSize, 0.3*textSize);
    
}

// Function 266
void UI_StoreControlData( inout UIContext uiContext, UIData data )
{
    UI_StoreDataBool( uiContext, data.backgroundImage, DATA_BACKGROUND_IMAGE );
    UI_StoreDataBool( uiContext, data.showImageWindow, DATA_CHECKBOX_SHOW_IMAGE );
    UI_StoreDataBool( uiContext, data.buttonA, DATA_BUTTONA );

    UI_StoreDataValue( uiContext, data.backgroundBrightness, DATA_BACKGROUND_BRIGHTNESS );
    UI_StoreDataValue( uiContext, data.backgroundScale, DATA_BACKGROUND_SCALE );
    UI_StoreDataValue( uiContext, data.imageBrightness, DATA_IMAGE_BRIGHTNESS );
    
    UI_StoreDataValue( uiContext, data.editWhichColor, DATA_EDIT_WHICH_COLOR );
    UI_StoreDataColor( uiContext, data.bgColor, DATA_BG_COLOR );
    UI_StoreDataColor( uiContext, data.imgColor, DATA_IMAGE_COLOR );
}

// Function 267
vec3 getButton(vec2 uv, vec2 mouse){
	
	vec2 block = floor(uv);
	vec2 pos = fract(uv)-0.5;
	float t = floor(iTime*12.5)/12.5; // Quantize time to 12 FPS
	
	vec3 c;
	
	if ( isButton(pos) ){
		float r = 0.5+0.5*sin(block.x+t+sin(block.y));
		float g = 0.5+0.5*cos(block.y+t+sin(block.x));
		float b = 0.0; // Launchpad has no blue :(
		
		#ifdef CLICKBUTTONS
		if (mouse.x > block.x &&
		   		mouse.x < block.x+1.0 &&
		   		mouse.y > block.y &&
		   		mouse.y < block.y+1.0){
			r=1.0;
			g = 1.0;
		}
		#endif
		
		c = vec3(r,g,b);
		c = floor(c*4.0)/4.0; //Quantize it to 4 steps, like the real thing!
		c *= pow(1.0-length(pos),2.0);// Make the middle of the button glow!
		
		c = clamp(c,0.025,1.0);//add dim button outlines
	}else{ // nothing here!
		c = vec3(0.0);	
	}
	
	
	
	
	return c;	
}

// Function 268
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

// Function 269
void UILayout_StackDown( inout UILayout uiLayout )
{
    uiLayout.vCursor.x = uiLayout.fTabPosition;
    uiLayout.vCursor.y = uiLayout.vControlMax.y + UIStyle_ControlSpacing().y;    
    uiLayout.vControlMax.x = uiLayout.vCursor.x;
    uiLayout.vControlMin.x = uiLayout.vCursor.x;
    uiLayout.vControlMax.y = uiLayout.vCursor.y;
    uiLayout.vControlMin.y = uiLayout.vCursor.y;
}

// Function 270
float sdEquilateralTriangle(  in vec2 p )
{
    const float k = sqrt(3.0);
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x + k*p.y > 0.0 ) p = vec2( p.x - k*p.y, -k*p.x - p.y )/2.0;
    p.x += 2.0 - 2.0*clamp( (p.x+2.0)/2.0, 0.0, 1.0 );
    return -length(p)*sign(p.y);
}

// Function 271
void UIStyle_GetFontStyleWindowText( inout LayoutStyle style, inout RenderStyle renderStyle )
{
    style = LayoutStyle_Default();
	renderStyle = RenderStyle_Default( vec3(0.0) );
}

// Function 272
void UIStyle_GetFontStyleWindowText( inout LayoutStyle style, inout RenderStyle renderStyle )
{
    style = LayoutStyle_Default();
    style.vSize *= 0.75;
	renderStyle = RenderStyle_Default( vec3(0.0) );
}

// Function 273
float dBuilding(vec3 p, vec3 cube)
{
    return dBox(vec3(p.x, p.y, p.z), cube, 0.001);
}

// Function 274
uint reverseuint(uint bits) {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x00ff00ffu) << 8) | ((bits & 0xff00ff00u) >> 8);
    bits = ((bits & 0x0f0f0f0fu) << 4) | ((bits & 0xf0f0f0f0u) >> 4);
    bits = ((bits & 0x33333333u) << 2) | ((bits & 0xccccccccu) >> 2);
    bits = ((bits & 0x55555555u) << 1) | ((bits & 0xaaaaaaaau) >> 1);
    return bits;   
}

// Function 275
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

// Function 276
float gpuIndepentHash(float p) {
    p = fract(p * .1031);
    p *= p + 19.19;
    p *= p + p;
    return fract(p);
}

// Function 277
void drawSystemButton(inout vec3 color, vec2 fragCoord, vec2 position, vec2 size, int type)
{
    drawBox(color, BORDER_FILL, BORDER_STROKE, fragCoord, position, size);
    
    if (type != SYSTEM_BUTTON_MENU) {
        // Draw Bevel
        drawBevel(color, BUTTON_HIGHLIGHT, BUTTON_SHADOW, fragCoord, position + vec2(1.0, 1.0), size - vec2(2.0, 2.0));
        drawBevel(color, BORDER_FILL, BUTTON_SHADOW, fragCoord, position + vec2(2.0, 2.0), size - vec2(4.0, 4.0));
    }
    
    if (type == SYSTEM_BUTTON_MENU) {
        // Draw wacky minus sign thingy
        drawBox(color, vec3(1.0, 1.0, 1.0), BUTTON_SHADOW, fragCoord, position + vec2(4.0, 8.0), vec2(13.0, 3.0));
        drawBox(color, vec3(1.0, 1.0, 1.0), BORDER_STROKE, fragCoord, position + vec2(3.0, 9.0), vec2(13.0, 3.0));
    }
    else if (type == SYSTEM_BUTTON_MINIMIZE) {
        // Draw Downward Facing Triangle... which is somehow not a yoga position
        drawTriangle(color, BORDER_STROKE, fragCoord, position + vec2(6.0, size.y - 15.0), vec2(7.0, 7.0), TRIANGLE_DOWN);
    }
    else if (type == SYSTEM_BUTTON_MAXIMIZE) {
        // Draw Upward Facing Triangle... which is somehow not a yoga position
        drawTriangle(color, BORDER_STROKE, fragCoord, position + vec2(6.0, size.y - 11.0), vec2(7.0, 7.0), TRIANGLE_UP);
    }
}

// Function 278
vec2 EquiRectToCubeMap(vec2 uv)
{
    vec2 gridSize = vec2(4,3); // 4 faces on x, and 3 on y
	vec2 faceSize = 1.0 / gridSize; // 1.0 because normalized coords
    vec2 faceIdXY = floor(uv * gridSize); // face id XY x:0->2 y:0->3
    
    // define the y limit for draw faces
    vec2 limY = vec2(0, uv.y);
    if (faceIdXY.x > 1.5 && faceIdXY.x < 2.5) // top & bottom faces
    	limY = vec2(0,faceSize.y*3.);
    else // all others
        limY = vec2(faceSize.y,faceSize.y*2.);

    // limit display inside the cube faces
    if ( uv.y >= limY.x && uv.y <= limY.y
#ifdef FACE_QUAD_SIZE
        && uv.x <= 1.0 
	#ifdef FACE_QUAD_SIZE_WITH_CENTERING
        && uv.x >= 0.0         
	#endif    
#endif
)
	{
        // get face id
        float faceId = 0.;
        if (faceIdXY.y<0.5) 	faceId = 4.;		 // top
        else if(faceIdXY.y>1.5) faceId = 5.;		 // bottom
        else 				    faceId = faceIdXY.x; // all others

        // face coord uvw
        vec3 p = getFaceUVW(uv,faceId,faceSize);
        
        // spheric to surface
        float theta = atan(p.y,p.x);
        float r = length(p);
        
        // correct spheric distortion for top and bottom faces
        // instead of just atan(p.z,r)
        float phi =  asin(p.z/r);
        
        return 0.5 + vec2(theta / _2pi, -phi / _pi);
    }
    return vec2(0); // outside faces => uv(0,0)
}

// Function 279
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

// Function 280
bool UIDrawContext_ScreenPosInView( UIDrawContext drawContext, vec2 vScreenPos )
{
    return Inside( vScreenPos, drawContext.clip );
}

// Function 281
void UI_DrawWindowTitleBar( inout UIContext uiContext, bool bActive, Rect titleBarRect, inout UIWindowState window )
{   
	if (!uiContext.bPixelInView || Outside( uiContext.vPixelCanvasPos, titleBarRect ))
        return;
    
    vec4 colorA = vec4(0.0, 0.0, 0.5, 1.0);
    vec4 colorB = vec4(0.03, 0.5, 0.8, 1.0);
    if ( bActive )
    {
        colorA.rgb += 0.1;
        colorB.rgb += 0.1;
    }

    float t = (uiContext.vPixelCanvasPos.x - titleBarRect.vPos.x) / 512.0;
    t = clamp( t, 0.0f, 1.0f );
    uiContext.vWindowOutColor = mix( colorA, colorB, t );
    
    {
        LayoutStyle style;
        RenderStyle renderStyle;
        UIStyle_GetFontStyleTitle( style, renderStyle );

        vec2 vTextOrigin = vec2(0);
        if ( FLAG_SET(window.uControlFlags, WINDOW_CONTROL_FLAG_MINIMIZE_BOX) )
        {
        	vTextOrigin.x += titleBarRect.vSize.y;
        }
        
        PrintState state = UI_PrintState_Init( uiContext, style, vTextOrigin );    
        PrintWindowTitle( state, style, window.iControlId );    
        RenderFont( state, style, renderStyle, uiContext.vWindowOutColor.rgb );
    }
}

// Function 282
vec2 fluid(vec2 uv1){
 vec2 uv = uv1;
 float t = iTime;
 for (float i = 1.; i < 15.; i++)
  {
    uv.x -= (t+sin(t+uv.y*i/1.5))/i;
    uv.y -= cos(uv.x*i/1.5)/i;
  }
  return uv;
}

// Function 283
void drawButton( inout vec4 c, vec2 p, vec4 t, float pressed){
    p-=t.xy;
    float area = .03 - sdBox(p,vec2(.020)+t.zw);
    float arean = smoothstep(0.,.01, area);

    c = mix(c, vec4(0.5,.5,0.5,1.), arean);
    c = mix(c, vec4(0.9,.9,0.9,1.), min(arean, (smoothstep(0.,.04, area))));
    c = mix(c, vec4(0.0,1.,0.8,1.), pressed*min(arean, (smoothstep(0.,.04, area))));
    c = mix(c, vec4(0.,0.,0.,1.), smoothstep(0.,.01, .01-abs(.03 - sdBox(p,vec2(.015)+t.zw))));
}

// Function 284
float GuitarElectricBase(float time,int note//https://www.shadertoy.com/view/XscXzn  ???
){float t=max(1.-time,0.)
 ;float freq=110.*pow(2.,float(note)/12.0)*baseFreq*t//or is it time instead ot t?
 ;vec2 a=freq+vec2(0,u5(cos(time*4.)/pi))//duty sqare-wave
 ;a.x=clamp(a.x-a.y,-1.,1.)*pow(max(0.,1.-(time*2.)),3.)
 ;return a.x;}

// Function 285
vec4 digitUIBox(int idx) {
    
    const vec2 digitRad = vec2(0.35, 0.5);
    
    return vec4(textCenter.x + (float(idx - 1))*textSize,
                textCenter.y,
                digitRad*textSize);

}

// Function 286
float rightBuilding(vec3 p){
    p = opBend(p, 0.1);
    float box = sdBox(p, vec3(1.5, 4., 1));
    float ceiling = sdBox(p + vec3(0,-3.8,0), vec3(1.65, .1, 1.3));
    float window0 = sdBox(p + vec3(1.4, -2.1, .1), vec3(.14, .2, .1));
    float window1 = sdBox(p + vec3(1.4, -1.6, -.5), vec3(.14, .2, .1));
    float window2 = sdBox(p + vec3(1.4, -.7, .1), vec3(.14, .2, .1));
    float windows = min(min(window1, window0), window2);
	float door = sdBox(p + vec3(1.4, .4, -.3), vec3(.15, .6, .1));
    float shape1 = min(windows, door);
    box = min(box, ceiling);
    float building = boolSubtraction(shape1, box);

    return building;
}

// Function 287
vec4 IntersectBuildings(vec3 ro, vec3 rd, float tmin, float tmax, out vec3 cube, out float minD)
{
    minD = tmax;
    vec4 res = vec4(-1.); res.x = tmax;

    vec3 rdi = 1./rd;
    vec3 rdia = abs(rdi);
    vec3 rds = sign(rd);
    
    ro += tmin*rd;
    vec2 cell = floor(ro.xz);
    vec2 dis = (cell.xy + 0.5 + rds.xz*0.5 - ro.xz) * rdi.xz;
    
    // Traverse 2D Grid
    for(int i = 0; i < 36; ++i)
    {
        cube = GetCube(cell);
        
        vec3 cellPos = vec3(cell.x + 0.5, cube.y, cell.y + 0.5);
        vec3 cellDir = cellPos - ro;
        
        float distNear = max(mincomp(cellDir*rdi - cube*rdia), 0.);
        float distFar = max(maxcomp(cellDir*rdi + cube*rdia), 0.);
        
        // RayMarch
        if(distNear < distFar && cube.y > 0.0)
        {
            float s = distNear;
            for(int j = 0; j < 14; ++j)
            {
                if(s > distFar)
                {
                    break;
                }

                vec3 currPos = rd*s - cellDir;
                float d = dBuilding(currPos, vec3(cube.x,cube.y, cube.z));
                minD = min(minD, d);
                s += d;
                if(d <= kRayEpsilon)
                {
                    res.x = s;
                    res.yzw = currPos.xyz/cube.xyz;
                    return res;
                }
            }
        }
        
        // step to next cell		
		vec2 mm = step( dis.xy, dis.yx ); 
		dis += mm*rdia.xz;
        cell += mm*rds.xz;
    }
    
    return res;
}

// Function 288
vec3 CalcNormalBuilding( in vec3 pos, in float t )
{
    vec2 e = vec2(1.0, -1.0)*0.01;
    return normalize( e.xyy*MapBuilding( pos + e.xyy ).x + 
					  e.yyx*MapBuilding( pos + e.yyx ).x + 
					  e.yxy*MapBuilding( pos + e.yxy ).x + 
					  e.xxx*MapBuilding( pos + e.xxx ).x );

}

// Function 289
Cam CamBuild()
{Cam c
;c.origin =texelFetch(iChannel0, LastPosition, 0).xyz
;c.forward=texelFetch(iChannel0, LastForward, 0).xyz
;c.right  =normalize(cross(c.forward, vec3(0,1,0)))
;c.up     =normalize(cross(c.right,c.forward))
;return c;}

// Function 290
void buildPlantSpace(KeyFrame frame, out PlantSpace res)
{
    // Leaves
    float leafAngle = frame.leafAngle;//  -(sin(iTime) * 0.2 + 0.1);
    float leafSin = sin(leafAngle);
    float leafCos = cos(leafAngle);
    
    res.matLeaf = mat2(-leafCos, leafSin, leafSin, leafCos);

    // Spine
    res.joint1AngleZ = frame.spine1;
    res.joint2AngleZ = frame.spine2;
    res.joint3AngleZ = frame.spine3;
    
    res.joint1 = joint3DMatrix(3.0, res.joint1AngleZ);
    res.joint2 = rotationY(frame.neck) * joint3DMatrix(3.0, res.joint2AngleZ) * res.joint1;
    
    
    // Head / Mouth
    float MouthAngle = frame.mouthAngle;
    res.mouthAngle = MouthAngle;
    
    float scale = 1.0 - MouthAngle * 0.07;
    res.head = scaleMatrix(vec3(scale, 1, 1)) * joint3DMatrix(3.0, res.joint3AngleZ) * res.joint2;

    float c = cos(MouthAngle);
    float s = sin(MouthAngle);
   	
    res.mouthRot = mat2(c, s, s, -c);
	
    
    float c2 = cos(MouthAngle * 0.5);
    float s2 = sin(MouthAngle * 0.5);
    
    res.teethRot = mat2(s2, -c2,
                     c2, s2);
    
    res.teethRot2 = mat2(s2,  c2,
                        -c2, s2);
    
    res.tPos1 = vec3(s * 1.5, -1.1, 0.0);
    res.tPos2 = vec3(s * 1.2, -0.8, 1.1);
    res.tPos3 = vec3(s * 0.6, -1.0, 1.5);
    res.tPos4 = vec3(-s * 1.5, -1.0, 0.56);
    res.tPos5 = vec3(-s * 1.2, -1.3, 1.3);
}

// Function 291
void sampleEquiAngular(
	float u,
	float maxDistance,
	vec3 rayOrigin,
	vec3 rayDir,
	vec3 lightPos,
	out float dist,
	out float pdf)
{
	// get coord of closest point to light along (infinite) ray
	float delta = dot(lightPos - rayOrigin, rayDir);
	
	// get distance this point is from light
	float D = length(rayOrigin + delta*rayDir - lightPos);

	// get angle of endpoints
	float thetaA = atan(0.0 - delta, D);
	float thetaB = atan(maxDistance - delta, D);
	
	// take sample
	float t = D*tan(mix(thetaA, thetaB, u));
	dist = delta + t;
	pdf = D/((thetaB - thetaA)*(D*D + t*t));
}

// Function 292
void UI_DrawColorPickerH( inout UIContext uiContext, bool bActive, vec3 vHSV, Rect pickerRect )
{
	if (!uiContext.bPixelInView || Outside( uiContext.vPixelCanvasPos, pickerRect ))
        return;
    
    vec2 vCurrPixelPos = (uiContext.vPixelCanvasPos - pickerRect.vPos) / pickerRect.vSize;
    vec3 vHSVCurr = vHSV;
    vHSVCurr.x = vCurrPixelPos.y;
    vHSVCurr.yz = vec2(1.0, 1.0);
    
    float fSelectedPos = vHSV.x * pickerRect.vSize.y + pickerRect.vPos.y;

	uiContext.vWindowOutColor = vec4( hsv2rgb( vHSVCurr ), 1.0 );
        
    float l = length( fSelectedPos - uiContext.vPixelCanvasPos.y );
    float d = l - 1.0;
    d = min(d, 5.0 - l);
    if ( bActive )
    {
        float d2 = l - 4.0;
    	d2 = min(d2, 6.0 - l);
	    d = max(d, d2);
    }
    
    float fBlend = clamp(d, 0.0, 1.0);
    
    uiContext.vWindowOutColor.rgb = mix(uiContext.vWindowOutColor.rgb, vec3(0.5), fBlend);    
}

// Function 293
vec2 	UIStyle_ControlSpacing() 		{ return  vec2(4.0); }

// Function 294
UIWindowState UI_ProcessWindowCommonBegin( inout UIContext uiContext, int iControlId, int iData, UIWindowDesc desc )
{   
    UIWindowState window = UI_GetWindowState( uiContext, iControlId, iData, desc );
        
    UI_PanelBegin( uiContext, window.panelState );
    
    uiContext.vWindowOutColor.rgba = UIStyle_WindowBackgroundColor();
    
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

// Function 295
Rect UILayout_StackControlRect( inout UILayout uiLayout, vec2 vSize )
{
    Rect rect = UILayout_GetStackedControlRect( uiLayout, vSize );
    UILayout_SetControlRect( uiLayout, rect );
    return rect;
}

// Function 296
vec4 DrawRotateButton(vec2 uv, float thinkness)
{
    vec4 col;
    col.a =  Segment(uv, vec2(0.01,-0.02), vec2(0.01,0),     thinkness);
    col.a += Segment(uv, vec2(0.05,0),     vec2(0.01,0),     thinkness);
    col.a += Segment(uv, vec2(0.05,0),     vec2(0.04,0.01),  thinkness);
    col.a += Segment(uv, vec2(-0.01,-0.02),vec2(-0.01,0),    thinkness);
    col.a += Segment(uv, vec2(-0.05,0),    vec2(-0.01,0),    thinkness);
    col.a += Segment(uv, vec2(-0.05,0),    vec2(-0.04,0.01), thinkness);
    col.a = clamp(col.a, 0.0, 1.0);
    col.rgb = vec3(col.a);
    return col;
}

// Function 297
vec4 drawKernelUI(vec2 uv, const Kernel kernel, float radiusStrength, float mipLevel)
{
    vec4 uiCol = vec4(0.0, 0.2, 0.9, 0.6);
    
    // Mip bar
    const float mipBarScale = 0.1;
    vec4 mipBarRect = vec4(0.35, mipLevel / (MIPMAP_MAX_LEVEL + epsilon) * mipBarScale, 0.365, 0.03);
    float mipBar = float(uv.x > mipBarRect.x && uv.x < mipBarRect.z && uv.y > mipBarRect.w && uv.y < (mipBarRect.y + mipBarRect.w));
    mipBar *= float(fract((uv.y - mipBarRect.w) * MIPMAP_MAX_LEVEL / mipBarScale) < 0.9);
    
    // Function graph
    const float vRange = 0.25;
    const float graphBottom = 0.03;
    const float graphMax = 0.25;
    const vec2 graphScale = vec2(0.25, 0.4);
    const float fMaxRadiusRec = 1.0 / (float(KERNEL_MAX_RADIUS) - 0.5);
    float xCenter = (uv.x - 0.5) * 2.0;
    
    float fX = (xCenter / graphScale.x) / ((float(kernel.radius) - 0.5) * fMaxRadiusRec);
    float f = 0.0;
    f += float(kernel.filterType == FILTER_TYPE_GAUSSIAN) * gaussian2d(vec2(fX, 0.0), FILTER_SIGMA);
    f += float(kernel.filterType == FILTER_TYPE_TENT) * tent2d(vec2(fX, 0.0));
    f += float(kernel.filterType == FILTER_TYPE_BOX) * float(abs(fX) <= 1.0);
    f -= float(kernel.filterType >= FILTER_TYPE_LAPLACIAN);
    f *= graphScale.y;
    
    float funcLine = float(abs(xCenter) < graphMax && uv.y > (f * vRange + graphBottom) && uv.y < (f * vRange + graphBottom + 0.005));
    
    // Kernel graph (unnormalized)
    int i = int(abs(xCenter / graphScale.x * (float(KERNEL_MAX_RADIUS) - 0.5)) + 0.5);
    float k = float(i < kernel.radius) * kernel.data[i] * kernel.sum * graphScale.y;
    
    float kernelBars = float(uv.y > graphBottom && uv.y < (k * vRange + graphBottom));
    
    float kernelColFade = 1.0 - (float(i) / float(KERNEL_MAX_RADIUS)) * 0.5;
    vec4 kernelCol = vec4(0.0, kernelColFade, kernelColFade, 0.7) * kernelBars;
    
    // Radius bar
    float radiusStrengthBar = float(uv.y < 0.02 && abs(xCenter) < (floor(radiusStrength) - 0.5) * fMaxRadiusRec * graphScale.x);
    
    // Combine UI elements
    return mix(uiCol * (mipBar + radiusStrengthBar + funcLine), kernelCol, kernelCol.a);
}

// Function 298
uint HashUInt(vec2  v, uvec2 r) { return Hash(floatBitsToUint(v), r); }

// Function 299
vec3 UIColor(int id)
{
    return texture(iChannel1, vec2(float(id) + 0.5, 1.5) / iResolution.xy).rgb;
}

// Function 300
float quintic(float a, float b, float x)
{
    x = clamp((x - a) / (b - a), 0.0, 1.0);
	return x*x*x*(x*(x*6.0 - 15.0) + 10.0);
}

// Function 301
void UILayout_StackRight( inout UILayout uiLayout )
{
    UILayout_SetX( uiLayout, uiLayout.vControlMax.x + UIStyle_ControlSpacing().x );
}

// Function 302
vec3 Buildings( vec2 uv, int layer )
{
    seed = uint( 2. + uv.x/4. );
    uv.x =(fract(uv.x/4.)-.5)*4.;
    
    bool cull = ( pow(float(layer+1)/8.,.3) < rand() );

    seed += 0x1001U*uint(layer);

	// octahedral, but with random distances so some planes won't be seen
    float a = Polygon( uv-vec2(0,0), 0. );
    float b = Polygon( uv-vec2(0,2), .5 );
    float c = Polygon( uv-vec2(0,4), 1. );

    if ( cull ) { a = 1.; b = 1.; c = 1.; }
    
    // ground
	a = min( a, uv.y+.5 );    
    
    vec3 f = vec3(a,min(a,b),min(min(a,b),c)).zyx;//min(min(a,b),c));//a,b,c);//
    vec3 col = vec3(.5+.5*f/(.01+abs(f)));
    
    return vec3(dot(col,vec3(.985,.01,.005)));
}

// Function 303
float quintic_interp(float x) {
    float c = clamp(x, 0., 1.);
	
    return c * c * c * ((6. * c - 15.) * c + 10.);
}

// Function 304
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

// Function 305
float uintBitsToFloat11(uint x) {
    return uintBitsToFloat((x >> 9u) | 0x40000000u) - 3.0;
}

// Function 306
vec3 EquirectToDirection(vec2 uv) {
    uv = NORM2SNORM(uv);
    uv.x *= PI;  // phi
    uv.y *= PIH; // theta
        
    /* Calculate a direction from spherical coords:
	** R = 1
    ** x = R sin(phi) cos(theta)
	** y = R sin(phi) sin(theta)
	** z = R cos(phi)
	*/
    return vec3(cos(uv.x)*cos(uv.y)
              , sin(uv.y)
              , sin(uv.x)*cos(uv.y));
}

// Function 307
vec2 	UIStyle_CheckboxSize() 			{ return vec2(24.0); }

// Function 308
Rect UILayout_GetStackedControlRect( inout UILayout uiLayout, vec2 vSize )
{
    return Rect( uiLayout.vCursor, vSize );
}

// Function 309
float uintToFloat( uint m )
{
    return uintBitsToFloat(0x3F800000u|(m&0x007FFFFFu) ) - 1.0;
}

// Function 310
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
	    DrawBorder( uiContext.vPixelCanvasPos, rect, uiContext.vWindowOutColor );                    
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

// Function 311
void traceBuildings( Ray a_ray, inout TraceCityRetval ct, float split_cells_spacing, bool shadow_trace )
{
 float maxh = /* MAX_BUILDING_HEIGHT */14.3;
 float minh = 2.;
 float rmin = -0.1;
 float rmax = max( 1., maxh );
 // we only dda trace the rmin, rmax z range
 float tbottom = plane_trace_z( a_ray, rmin, 1e-6 );
 float ttop = plane_trace_z( a_ray, rmax, 1e-6 );
 vec2 r0 = vec2( 0., /* MAX_DDA_TRACE */640. );
 vec2 r1 = vec2( min( ttop, tbottom ), max( ttop, tbottom ) );
 vec2 r2 = vec2( max( r0.x, r1.x ), min( r0.y, r1.y ) ); // intersection of r0 and r1
//	if ( r2.y <= r2.x ) return; // non sensical per drop if we return...
 r2 *= step( r2.x, r2.y ); // ...so instead do a zero length iteration
 float start_t = r2.x; // remember initial jump to return something along a_ray
 Ray ray = mkray( a_ray.o + a_ray.d * start_t, a_ray.d ); // warp to tmin
 vec3 ray_end = a_ray.o + a_ray.d * r2.y;
 DDA3 dda = dda_init( ray.o, ray_end, /* CELL_SIZE */vec3( 8., 8., 100. ), false );
 // trace within dda traversed cell
 ParametricBuildingHit hit;
 hit.t = /* FLT_MAX */1000000.;
 float split_case = -1.;
 // dda traverse
 for ( int i = 0; i < ( 37 /*FORCE_LOOP*/+min(0,iFrame) )
    && dot( dda.p - a_ray.o, dda.p - a_ray.o ) < r2.y * r2.y; ++i )
 {
  // raytrace 4 boxes inside each cell
  bounds2 b = mkbounds_unchecked( dda.c.xy * /* CELL_SIZE */vec3( 8., 8., 100. ).xy, ( dda.c.xy + vec2( 1., 1. ) ) * /* CELL_SIZE */vec3( 8., 8., 100. ).xy ); // cell bounds
  ivec2 index = ivec2( dda.c.xy );
  vec4 a; a.xy = /* CELL_SPACING */vec2( 0.47, 1.3 ).xy * 0.5; a.zw = a.xy;
  if ( ( index.x & 1 ) == 0 ) a.xy = a.yx;
  if ( ( index.y & 1 ) == 0 ) a.zw = a.wz;
  b.pmin.xy += a.xz; // shrink cell bounds according to street margins (we alternate wide and narrow streets hence logic above)
  b.pmax.xy -= a.yw;
  vec2 margin = vec2( split_cells_spacing * 0.5 + 0.2 );
  vec2 r55 = hash22_( index ); // split type, permute
  Ray ray2 = ray;
  bool permuted = false;
  if ( r55.y > 0.5 )
  {
   // random permutations, else default patterns look more or less all aligned
   ray2.o.xyz = ray.o.yxz;
   ray2.d.xyz = ray.d.yxz;
   b.pmin.xy = b.pmin.yx;
   b.pmax.xy = b.pmax.yx;
   permuted = true;
  }
  vec4 r4 = icdf( hash42_( index * 0x8da6b343 ) ); // heights hash
  vec4 rheights = mix( vec4( minh ), vec4( maxh ), r4 );
  vec4 r3 = hash42_( index * 0xb68f63e4 ); // split hash
  vec4 r3_0 = r3;
  r3.xyw = mix( vec3( 1. ), vec3( 5. ), r3.xyw ); // ratio of smallest to largest size
  r3.xy = fractions( r3.xyw ); // use r3.xyw as relative unit sizes
  r3.z = mix( 0.2, 0.8, r3.z );
  Ranges_x4 iv;
  Split4bSetup s4su = setup_Split4b( ray2, b.pmin.xy, b.pmax.xy, margin );
  // select a tile split pattern
  if ( r55.x > 0.75 )
  {
   bound_Split4b_xxy( iv, ray2, b.pmin.xy, b.pmax.xy, mix( b.pmin.xxy, b.pmax.xxy, r3.xyz ), rheights, margin );
   trace_Split4b_xxy( iv, ray2, s4su, mix( b.pmin.xxy, b.pmax.xxy, r3.xyz ), rheights );
   split_case = 0.;
  }
  else if ( r55.x > 0.5 )
  {
   r3.xyz = fractions( mix( vec4( 2. ), vec4( 3. ), r3_0 ) );
   bound_Split4b_xxx( iv, ray2, b.pmin.xy, b.pmax.xy, mix( b.pmin.xxx, b.pmax.xxx, r3.xyz ), rheights, margin );
   trace_Split4b_xxx( iv, ray2, s4su, mix( b.pmin.xxx, b.pmax.xxx, r3.xyz ), rheights );
   split_case = 1.;
  }
  else if ( r55.x > 0.25 )
  {
   bound_Split4b_xyy( iv, ray2, b.pmin.xy, b.pmax.xy, mix( b.pmin.xyy, b.pmax.xyy, r3.zxy ), rheights, margin );
   trace_Split4b_xyy( iv, ray2, s4su, mix( b.pmin.xyy, b.pmax.xyy, r3.zxy ), rheights );
   split_case = 2.;
  }
  else
  {
   bound_Split4b_xyx( iv, ray2, b.pmin.xy, b.pmax.xy, mix( b.pmin.xyx, b.pmax.xyx, r3.xzy ), rheights, margin );
   trace_Split4b_xyx( iv, ray2, s4su, mix( b.pmin.xyx, b.pmax.xyx, r3.xzy ), rheights );
   split_case = 3.;
  }
  hit.t = /* FLT_MAX */1000000.;
  hit.tile_child_index = -1.; // no hit
  rayMarchCellObjects( ray2, iv, permuted, hit, dda.c.xy, shadow_trace );
  if ( hit.t != /* FLT_MAX */1000000. ) break; // we have hit, gtfo and fill other extra bits out of the loop
//		if ( hit.t > ct.t ) return; // fixme: no point in continuing, but we should just set dda end point instead
  dda_step_infinite( dda ); // make sure you set a_finite to false in dda_init when calling this version
 }
 if ( hit.t >= ct.t ) return; // ct.t might be FLT_MAX so >= is important here
 // we hit a building
 ct.p = ray.o + hit.t * ray.d;
 ct.t = start_t + hit.t; // remember that we jumped at start
 if ( shadow_trace ) return; // we don't need normal, ao, material... gtfo
 // house type will use type index [0,3], building type will use index [4,7]
 pack_info( ct, dda.c.xy, split_case, hit.tile_child_index + hit.building_type * 4. );
 vec3 p = hit.ray2.o + hit.t * hit.ray2.d;
 vec3 h = vec3( 0.01, 0., 0. ); // h.x *= hit.t; // grainy normals tweak
 ct.n = normalize( vec3( sd_ParametricBuilding( hit.t, p + h.xyz, hit.building_type, hit.b2, hit.height, hit.nh, hit.rnd ).d,
       sd_ParametricBuilding( hit.t, p + h.zxy, hit.building_type, hit.b2, hit.height, hit.nh, hit.rnd ).d,
       sd_ParametricBuilding( hit.t, p + h.yzx, hit.building_type, hit.b2, hit.height, hit.nh, hit.rnd ).d )
       - hit.d ); // hit.d should be equal to sd_ParametricBuilding( hit.t, p, hit.building_type, hit.b2, hit.height, hit.nh, hit.rnd ).d
 // do ao in permuted space
 {
  Ray ao_ray = mkray( p, ct.n );
  { /* ao algo from http://www.iquilezles.org/www/material/nvscene2008/rwwtt.pdf macrofified to avoid repetition */ float _delta = 0.1, _a = 0.0, _b = 1.0; for ( int _i = 0; _i < 5 /*FORCE_LOOP*/+min(0,iFrame); _i++ ) { float _fi = float( _i ); float _ao_t = _delta * _fi;
   float d = sd_ParametricBuilding( _ao_t, ao_ray.o + _ao_t * ao_ray.d, hit.building_type, hit.b2, hit.height, hit.nh, hit.rnd ).d;
  _a += ( _ao_t - d ) * _b; _b *= 0.5; } ct.ao = max( 1.0 - 1.2 * _a, 0.0 ); }
 }
 if ( hit.is_roof > 0. ) ct.type = 8;
 if ( ( abs( hit.d - hit.windr.x ) < 0.001 ) // distance must be close to windows plane, stored in hit.windr.x
   && ( 0.01 < abs( dot( hit.windr.yz, ct.n.xy ) ) ) // normal must match window orientation
   // normal must be vertical
   && ( abs( ct.n.z ) < 0.005 ) ) ct.type = 4;
 if ( hit.permuted ) ct.n.xy = ct.n.yx;
}

// Function 312
vec2 calcEquirectangularFromGnomonicProjection(in vec2 sph, in vec2 centralPoint) {
    vec2 cp = (centralPoint * 2.0 - 1.0) * vec2(PI, PI_2);
	float cos_c = sin(cp.y) * sin(sph.y) + cos(cp.y) * cos(sph.y) * cos(sph.y - cp.y);
    float x = cos(sph.y) * sin(sph.y - cp.y) / cos_c;
    float y = ( cos(cp.y) * sin(sph.y) - sin(cp.y) * cos(sph.y) * cos(sph.y - cp.y) ) / cos_c; 
    return vec2(x, y) + vec2(PI, PI_2); 
}

// Function 313
vec2 circuit(vec2 p)
{
	p = mod(p, 2.0) - 1.0;
	float w = 1e38;
	vec2 cut = vec2(1.0, 0.0);
	vec2 e1 = vec2(-1.0);
	vec2 e2 = vec2(1.0);
	float rnd = 0.23;
	float pos, plane, cur;
	float fact = 0.9;
	float j = 0.0;
	for(int i = 0; i < ITS; i ++)
	{
		pos = mix(dot(e1, cut), dot(e2, cut), (rnd - 0.5) * fact + 0.5);
		plane = dot(p, cut) - pos;
		if(plane > 0.0)
		{
			e1 = mix(e1, vec2(pos), cut);
			rnd = fract(rnd * 19827.5719);
			cut = cut.yx;
		}
		else
		{
			e2 = mix(e2, vec2(pos), cut);
			rnd = fract(rnd * 5827.5719);
			cut = cut.yx;
		}
		j += step(rnd, 0.2);
		w = min(w, abs(plane));
	}
	return vec2(j / float(ITS - 1), w);
}

// Function 314
float guitar2(float time, int key) {
    float frequency = 27.4 * pow(2.001, float(key)/12.0);
     // Maybe express this in terms of frequency.
    // Model bottom curve.
    float B = pow(10.0, (float(key) + 1.0) / 24.0 - 5.15);

    return guitar3(time, frequency, B);
}

// Function 315
bool get_hide_ui(in sampler2D s)
{
    return texelFetch(s, CTRL_HIDE_UI, 0).w > 0.05;
}

// Function 316
uint wang_hash_ui( uint seed )
{
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15);
    return seed;
}

// Function 317
vec2 	UIStyle_WindowBorderSize() 		{ return vec2(6.0); }

// Function 318
vec4 UI_GetFinalColor( UIContext uiContext )
{
    if ( int(uiContext.vFragCoord.y) < 2 )
    {
        return uiContext.vOutData;
    }
    
    if ( uiContext.vOutColor.a >= 0.0 )
    {
        // Apply premultiplied alpha.
        uiContext.vOutColor.rgb *= uiContext.vOutColor.a;
  
#ifdef SHADOW_TEST
        // Shadow composite for premultiplied alpha.
        // Don't even ask how this works - I'm not sure I know
        uiContext.vOutColor.rgb *= uiContext.fOutShadow;
        uiContext.vOutColor.a = 1.0 - ((1.0 - uiContext.vOutColor.a) * uiContext.fOutShadow);
#endif 	
    }
    else
    {
#ifdef SHADOW_TEST
        uiContext.vOutColor.a = -1.0 -uiContext.fOutShadow;
#else
        uiContext.vOutColor.a = -2.0;
#endif 
    }
    
    return uiContext.vOutColor;
}

// Function 319
vec4    UIStyle_WindowBackgroundColor() { return vec4( 0.75, 0.75, 0.75, 1.0 ); }

// Function 320
float UISlider(int id)
{
    return texture(iChannel1, vec2(float(id) + 0.5, 0.5) / iResolution.xy).r;
}

// Function 321
void UI_WriteCanvasPos( inout UIContext uiContext, int iControlId )        
{
	if (!uiContext.bPixelInView)
        return;
    Rect rect = Rect( vec2(0), uiContext.drawContext.vCanvasSize );
    DrawRect( uiContext.vPixelCanvasPos, rect, vec4(uiContext.vPixelCanvasPos, float(iControlId), -1.0 ), uiContext.vWindowOutColor );
}

// Function 322
void UI_DrawSliderY( inout UIContext uiContext, bool bActive, bool bMouseOver, float fPosition, Rect sliderRect, float fHandleSize, bool scrollbarStyle )
{
	if (!uiContext.bPixelInView || Outside( uiContext.vPixelCanvasPos, sliderRect ))
        return;
    
    Rect horizLineRect;
    
    horizLineRect = sliderRect;
    if (!scrollbarStyle)
    {
	    float fMid = sliderRect.vPos.x + sliderRect.vSize.x * 0.5;
    	horizLineRect.vPos.x = fMid - 2.0;
    	horizLineRect.vSize.x = 4.0;
    }

#ifdef NEW_THEME    
    DrawBorderRect( uiContext.vPixelCanvasPos, horizLineRect, cSliderLineCol, uiContext.vWindowOutColor );
#else    
    DrawBorderIndent( uiContext.vPixelCanvasPos, horizLineRect, uiContext.vWindowOutColor );
#endif    

    float fSlideMin = sliderRect.vPos.y + fHandleSize * 0.5f;
    float fSlideMax = sliderRect.vPos.y + sliderRect.vSize.y - fHandleSize * 0.5f;

    float fDistSlider = (fSlideMin + (fSlideMax-fSlideMin) * fPosition);

    Rect handleRect;

    handleRect = sliderRect;
    handleRect.vPos.y = fDistSlider - fHandleSize * 0.5f;
    handleRect.vSize.y = fHandleSize;

    vec4 handleColor = vec4(0.75, 0.75, 0.75, 1.0);
    if ( bActive )
    {
        handleColor.rgb += 0.1;
    }
    
    // highlight
#ifdef NEW_THEME     
    if ( (uiContext.vPixelCanvasPos.y - handleRect.vPos.y) < handleRect.vSize.y * 0.3 )
    {
        handleColor.rgb += 0.05;
    }
#endif    

    DrawRect( uiContext.vPixelCanvasPos, handleRect, handleColor, uiContext.vWindowOutColor );
#ifdef NEW_THEME   
    DrawBorderRect( uiContext.vPixelCanvasPos, handleRect, cSliderHandleOutlineCol, uiContext.vWindowOutColor );
#else     
    DrawBorderOutdent( uiContext.vPixelCanvasPos, handleRect, uiContext.vWindowOutColor );
#endif    
}

// Function 323
float sdEquilateralTriangle(  in vec2 p, in float r )
{
    const float k = sqrt(3.0);
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if( p.x+k*p.y>0.0 ) p=vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0*r, 0.0 );
    return -length(p)*sign(p.y);
}

// Function 324
uint  asuint2(float x) { return x == 0.0 ? 0u : floatBitsToUint(x); }

// Function 325
vec3 inferno_quintic( float x )
{
	x = saturate( x );
	vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3(
		dot( x1.xyzw, vec4( -0.027780558, +1.228188385, +0.278906882, +3.892783760 ) ) + dot( x2.xy, vec2( -8.490712758, +4.069046086 ) ),
		dot( x1.xyzw, vec4( +0.014065206, +0.015360518, +1.605395918, -4.821108251 ) ) + dot( x2.xy, vec2( +8.389314011, -4.193858954 ) ),
		dot( x1.xyzw, vec4( -0.019628385, +3.122510347, -5.893222355, +2.798380308 ) ) + dot( x2.xy, vec2( -3.608884658, +4.324996022 ) ) );
}

// Function 326
Roots5 solveQuinticLinSearch(in GeneralQuintic eq, in float bmin, in float bmax, in float epsilon, in bool wrap, in int searches, in int iters) {
    Roots5 roots = Roots5(0, Float5(0.0, 0.0, 0.0, 0.0, 0.0));
    float interval = 1.0 / float(searches);
    float start = 0.0;

    // Take equally spaced steps over the interval
    for (int search=ZERO; search < searches; search++) {
        // Apply newton-raphson
        float root = mix(bmin, bmax, start);
        for (int nrIter=0; nrIter < iters; nrIter++) {
            float nrStep = root -= evalQuintic(root, eq) / evalQuinticPrime(root, eq);
            if (wrap) root = bmin + mod(root - bmin, bmax - bmin); // Wrap to stay in the interval
            if (abs(nrStep) < epsilon) break; // Potential early out
        }

        // Make sure this is a unique root (get rid of solutions that converge to the same spot)
        bool unique = true;
        for (int n=ZERO; n < roots.nroots; n++) {
            if (abs(root - get(roots.roots, n)) < epsilon) {
                unique = false;
                break;
            }
        }

        // Only use this solution if it is nearly a true root and is unique
        if (abs(evalQuintic(root, eq)) < epsilon && unique) {
            set(roots.roots, roots.nroots, root);
            roots.nroots++;
        }

        // Step forward and take another chance at an early out
        start += interval;
        if (roots.nroots == 5) break;
    }

    return roots;
}

// Function 327
uint fToUint(float f)
{
    return uint(f * UINT_MAX);
}

// Function 328
void drawButtons()
{
  vec2 buttonSize = vec2 (0.15, 1.4 / maxSelection);
  for (float n=0.0; n < maxButtons; n++)
  {
    if (rectangle(pPos(n), buttonSize) > 0.5) color = gray;
  }
  if (rectangle(pPos(selection), buttonSize) > 0.0) color = yellow;
}

// Function 329
vec2 equiRemap(vec2 lonLat, vec2 delta) {
    vec3 v = lonLatToXYZ(lonLat);
	v = yRot(v,delta.x);
    v = xRot(v,delta.y);
    return xyzToLonLat(v);
}

// Function 330
void GetBuildingColor(vec3 cube, vec4 res, vec3 p, vec3 n, vec3 rd, out vec3 color, out vec4 mat)
{   
    vec3 localPos = res.yzw;
    vec2 uv = localPos.xy*abs(dot(n, vec3(0., 0., 1.)))
        	+ localPos.zy*abs(dot(n, vec3(1., 0., 0.)));
 
    // Main CubeColor
    color = mix(vec3(hash(cube.y)*0.5), vec3(hash(150.52 + cube.y), hash(cube.y*9.21), hash(cube.y*59.78)), 0.25);
    color = sqrt(sqrt(color));
    color *= texture(iChannel0, uv.yx*vec2(1., 2.)*1.0 + vec2(cube.y, 0.)).xyz;
    color *= 0.15;
    
    // Windows
    vec2 q = uv;
    
    vec2 winId = 125.692 * floor(q/vec2(0.4, 0.1));
    
    // Set Lighting Properties
    float winLightInt = pow(hash(winId), 4.);
    
    vec2 q2 = q; q2 = mod(q2, vec2(0.4, 0.1)) - vec2(0.20, 0.05);
    vec3 winLightColor = vec3(hash(125.+winId), hash(winId*12.56), hash(winId * 96.58));
    winLightColor += vec3(0.745, 0.666, 0.152);
    winLightColor = min(winLightColor*1.5, 1.);
    
    float dWindow = pow(max(-dBox(q2, vec2(0.15, 0.015), 0.01)/0.01, 0.), 5.5);
    vec3 winColor = winLightInt*mix(color, winLightColor, smoothstep(0., 1., dWindow * step(n.y, 0.5)));
    
    float lFactor = 1.-smoothstep(0., 1., abs(q2.x)/0.15);
    lFactor *= 1.-smoothstep(0., 1., abs(q2.y)/0.03);
    float totalLFactor = pow(lFactor, 0.09);
    totalLFactor  += pow(lFactor, 0.5)*2.;
    totalLFactor  += pow(lFactor, 1.0);
    totalLFactor  += pow(lFactor, 2.0);
    
    winColor += step(0.6, winLightInt)*mix(color, winLightColor, totalLFactor);
    color = mix(winColor, color, smoothstep(0., 0.001, abs(q.x)-0.80) );

    gAmbientFactor = mix(0.00, winLightInt, smoothstep(0., 1., dWindow));
    mat = vec4(0.);
    mat.x = 0.5; mat.y = 0.5; mat.z = 0.0;
}

// Function 331
void GetUintData(){
	uintData.x = packUnorm2x16(data.st);//32bits
	uintData.y = packUnorm2x16(data.pq);//32bits
}

// Function 332
int HoldButton(vec4 mouse, vec2 screen, vec2 pos)
{
    float aspect = screen.x / screen.y;
    vec2 uv = mouse.zw / screen;
    uv -= vec2(0.5, 0);
    uv.x *= aspect;
    uv -= pos;
    if (uv.y >= -0.02 && uv.y <= 0.02)
    {
        if (uv.x >= 0.01 && uv.x <= 0.05) return 0;
        if (uv.x >= -0.05 && uv.x <= -0.01) return 1;
    }
    return -1;
}

// Function 333
int HoldRotateButton(vec4 mouse, vec2 screen)
{
    return HoldButton(mouse, screen, vec2(-0.14, 0.04));
}

// Function 334
vec3 building(vec3 color, vec2 pos, vec2 county, float population, vec3 roofColor, vec2 block) {
    vec2 b = buildingSize * block;
    float terrain = fbm1(terrainScaleInv * b);
    float free = step(townTerrainThresholdLow, terrain) * step(terrain, townTerrainThresholdHigh);
    free *= step(fbm1(treeGrowthScaleInv * b + 100.0), townTreeThreshold);
    free *= step(hash1(block), 1.2 - distance(b, county) / ((1.0 + population) * urbanSprawl));
    
    vec3 buildingHash = hash3(block);
    vec2 size = vec2(0.3, 0.1) + vec2(0.25, 0.25) * buildingHash.xy;
    vec2 center = vec2(1.2 * (buildingHash.z - 0.5), size.y - 0.6);
    vec2 p = pos - block - center;
    
    vec2 roofHash = hash2(block);
    float roofWidth = size.x * (0.4 + 0.6 * roofHash.x);
    float roofside = 1.0 - 2.0 * step(roofHash.y, 0.5);
    float triangle = sdTriangleIsosceles(p - vec2(roofside * (roofWidth - size.x), 0.4 + size.y), vec2(roofWidth, -0.4));
    float sdf = sdBox(p, size);
    sdf = min(sdf, triangle);
    sdf -= bsdf;
    
    float roofSdf = min(triangle, sdTriangleIsosceles(p - vec2(roofside * (size.x - roofWidth), 0.4 + size.y), vec2(roofWidth, -0.4)));
    roofSdf = min(roofSdf, sdBox(p - vec2(0.0, 0.2 + size.y), vec2(size.x - roofWidth, 0.2)));
    roofSdf -= roofSize;
    roofSdf = max(roofSdf, size.y - p.y + bsdf);
    color = mix(color, roofColor, step(roofSdf, 0.0) * free);
    color = mix(color, vec3(0.0), clamp((blw - abs(buildingSize * roofSdf)) * aa, 0.0, 1.0) * free);
    
    vec3 sidingHash = hash3(block + 0.1);
    vec3 siding = vec3(0.2 + 0.3 * sidingHash.x + 0.5 * noise1(vec2(6.0, 10.0) * pos)) + sidingHash.y * vec3(0.2, 0.1 + 0.1 * sidingHash.z, 0.0);
    color = mix(color, siding, step(sdf, 0.0) * free);
    
    vec3 windowHash = hash3(block + 0.2);
    vec2 windowSize = vec2(0.2 + 0.1 * windowHash.x, 0.4 + 0.4 * step(windowHash.y, 0.5));
    vec2 windowCenter = vec2(0.8 * (0.5 - windowHash.z), 0.4 - windowSize.y);
    float windowSdf = sdBox(p - size * windowCenter, size * windowSize);
    color = mix(color, 0.5 * roofColor, 0.3 * step(windowSdf, 0.0) * free);
    color = mix(color, vec3(0.0), clamp((blw - abs(buildingSize * windowSdf)) * aa, 0.0, 1.0) * free);
    
    return mix(color, vec3(0.0), clamp((blw - abs(buildingSize * sdf)) * aa, 0.0, 1.0) * free);
}

