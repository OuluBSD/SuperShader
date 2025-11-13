# color_adjustments_functions

**Category:** effects
**Type:** extracted

## Dependencies
texture_sampling, normal_mapping, lighting, raymarching

## Tags
texturing, animation, particles, lighting, effects, color

## Code
```glsl
// Reusable Color Adjustments Effect Functions
// Automatically extracted from effect-related shaders

// Function 1
vec3 baseColor(vec2 uv)
{
	vec3 col = vec3(max(uv.y,0.0)+max(uv.x,0.0),max(-uv.y,0.0)+max(uv.x,0.0),max(-uv.x,0.0));
    return col;
}

// Function 2
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

// Function 3
float mixColors(float r, float v, float z){ 
  return clamp(0.5+0.5*(v-r)/z, 0., 1.);
}

// Function 4
vec3 adjust_out_of_gamut(vec3 c)
{
    //return adjust_out_of_gamut_sat(c);
	//return adjust_out_of_gamut_lerp(c);
    //return adjust_out_of_gamut_remap(c);
    return adjust_out_of_gamut_maxcomp(c);
    
    //return ACESFitted( c );
}

// Function 5
vec3 sky_color_s(vec3 ray)
{
	vec3 rc = 0.2 + vec3(4.5, 4.9, 4.5)*texture(iChannel2, ray).rgb;
    return rc;
}

// Function 6
vec2 colorshift(vec2 uv, float amount, float rand) {
	
	return vec2(
		uv.x,
		uv.y + amount * rand // * sin(uv.y * iResolution.y * 0.12 + iTime)
	);
}

// Function 7
vec3 GetScreenPixelColor(vec2 ul)
{
    float x = mod(ul.x * TV_SCREEN_RESOLUTION_X / (1600.0/iResolution.x) * 3.0,3.0);

    vec3 tex = texture(iChannel0, vec2(ul.x,ul.y) ).xyz;
    
    return mix(mix(vec3(tex.r,0.0,0.0),vec3(0.0,tex.g,0.0),step(1.0,x)),
              vec3(0.0,0.0,tex.b),step(2.0,x));
}

// Function 8
vec3 map_color(vec3 p) {
    return p;
}

// Function 9
float lambertColor(vec3 p, vec3 nor)
{
    vec3 l = normalize(lightPos - p);
    float lightDistance = length(lightPos - p);
    float atten = ((1. / lightDistance) * .5) + ((1. / (lightDistance * lightDistance)) * .5);                                                  
    return max(dot(nor, l), 0.0) * atten * 500.0;
}

// Function 10
float colorDistAccurate(vec4 c1, vec4 c2){
	vec4 c = c1 - c2;
    float y = c.r * 0.2124681075446384 + c.g * 0.4169973963260294 + c.b * 0.08137907133969426;
  	float i = c.r * 0.3258860837850668 - c.g * 0.14992193838645426 - c.b * 0.17596414539861255;
  	float q = c.r * 0.0935501584120867 - c.g * 0.23119531908149002 + c.b * 0.13764516066940333;
  	return y*y + i*i + q*q;
}

// Function 11
vec4 colorEruption(in vec3 ro, in vec3 rd, in float time,in float kMax, vec3 refVec, in float h, in vec2 scale) {
    refVec = normalize(refVec);
    vec3 pAnim = -(1.+h)*refVec;
    Base b = basis(pAnim, refVec);
    
    vec4 textColor,col = vec4(0);
    // Chang basis
	ro = (ro-b.o)*b.base;
	rd = rd*b.base;

	float k = (abs(rd.z)<.0001) ? -1. : -ro.z/rd.z;   // intersection avec plan y=0; dans base du fire
    float dk = abs(.003/rd.z);
    vec3 p = ro + k*rd;
    if (k>0. && abs(p.x)<.1*scale.x && abs(p.y)<.1*scale.y) {
        k = max(0., k-13.*dk); 
        float tSun = 10.*scale.x -(time)*.04;
        for (int i=0;i<26;i++) {
            if (k>kMax) break;
            p = ro + k*rd;
            textColor = fireTexture(scale*p.yx, tSun, .75*float(i)*p.z/rd.z);
            col += .08*textColor;// + col*(1.-textColor.a) + textColor*textColor.a
            k+=dk;
        }
    }
	return col;
}

// Function 12
vec3 backgroundColorHemi(vec3 dir){
    return textureLod(iChannel0, dir, 9.5).rgb;
}

// Function 13
vec3 getLightColor( in vec3 pos ) {
    vec3 lcol = vec3( 1., .7, .5 );
    
	vec3 pd = pos;
    pd.x = abs( pd.x );
    pd.z *= -sign( pos.x );
    
    float ch = hash( floor( (pd.z+18.*time)/40. ) );
    vec3 pdc = vec3( pd.x, pd.y, mod( pd.z+18.*time, 40.) - 20. );

    if( ch > 0.75 ) { // car
        pdc.x += (ch-0.75)*4.;
        if(  sdSphere( vec3( abs(pdc.x-5.)-1.05, pdc.y-0.55, pdc.z ), 0.25) < 2. ) {
            lcol = vec3( 1., 0.05, 0.01 );
        }
    }
    if( pd.y > 2. && abs(pd.x) > 10. && pd.y < 5. ) {
        float fl = floor( pd.z/13. );
        lcol = 0.4*lcol+0.5*vec3( hash( .1562+fl ), hash( .423134+fl ), 0. );
    }
    if(  abs(pd.x) > 10. && pd.y > 5. ) {
        float fl = floor( pd.z/2. );
        lcol = 0.5*lcol+0.5*vec3( hash( .1562+fl ),  hash( .923134+fl ), hash( .423134+fl ) );
    }
   
    return lcol;
}

// Function 14
vec4 mapPortalColor(vec3 p, vec3 portalPos, float rotY, vec4 cristalcolor, vec4 fxcolor)
{
    vec2 q = rotateY(p-portalPos, rotY).xy; q.y *= 0.55;
    float d = length(q) - 1.4 + sin(q.x*10.+t*2.)*cos(q.y*10.+t*2.) * 0.05;
    return mix(cristalcolor, fxcolor, smoothstep(-0.5, 0.2, d));
}

// Function 15
vec3 getStoneColor(vec3 p, float c, vec3 l, vec3 n, vec3 e) {
    c = min(c + pow(noise_3(vec3(p.x*20.0,0.0,p.z*20.0)),70.0) * 8.0, 1.0);
    float ic = pow(1.0-c,0.5);
    vec3 base = vec3(0.42,0.3,0.2) * 0.35;
    vec3 sand = vec3(0.51,0.41,0.32)*0.9;
    vec3 color = mix(base,sand,c);
        
    float f = pow(1.0 - max(dot(n,-e),0.0), 5.0) * 0.75 * ic;    
    color += vec3(diffuse(n,l,0.5) * WHITE);
    color += vec3(specular(n,l,e,8.0) * WHITE * 1.5 * ic);
    n = normalize(n - normalize(p) * 0.4);    
    color += vec3(specular(n,l,e,80.0) * WHITE * 1.5 * ic);    
    color = mix(color,vec3(1.0),f); 
    
    color *= sqrt(abs(p.y*0.5+0.5)) * 0.4 + 0.6;
    color *= (n.y * 0.5 + 0.5) * 0.4 + 0.6; 
    
    return color;
}

// Function 16
vec3 getColor(float x0, float x1, vec2 uv, vec3 color) {
   
   // First u becomes [0,1] then the range [0.0, 0.5] will be 
   // transformed into [0.0, 1.0] and ]0.5, 1.0] into ]1.0, 0.0].
   float u = (uv.x - x0)/(x1 - x0);
              
   // u <= 0.5
   float ud = (u/0.5) * (1.0 - step(0.5, u));
   // u > 0.5
   ud += (1. - (u/0.5-1.)) * (1.0 - step(u, 0.5));
           
   // Remove aliasing by making the shading points near x0 and x1 darker.
   vec3 col = mix(vec3(0.0), color, smoothstep(.0, .6, ud)); 
            
   // Add lightning by making darker the shading points that are 
   // about to be covered and going "behind" another face. This also
   // removes aliasing since if x1-x0 is small the borders cover the 
   // darker sides and transition to interior area becomes very sharp. 
   float w = (x1 - x0);            
   col *= w / .55;
   return col;            
}

// Function 17
float colormap_blue(float x) {
    if (x < 0.3) {
       return 4.0 * x + 0.5;
    } else {
       return -4.0 * x + 2.5;
    }
}

// Function 18
float colorclose(vec3 yuv, vec3 keyYuv, vec2 tol)
{
    float tmp = sqrt(pow(keyYuv.g - yuv.g, 2.0) + pow(keyYuv.b - yuv.b, 2.0));
    if (tmp < tol.x)
      return 0.0;
   	else if (tmp < tol.y)
      return (tmp - tol.x)/(tol.y - tol.x);
   	else
      return 1.0;
}

// Function 19
vec3 holoGetColor(mat4 head, vec3 p) {
    return holoGetColor(head,p.xz / HOLO_SIZE);
}

// Function 20
vec3 getBaseColor(int i)
{
	if (i == 0) return vec3(1.0, 0.4, 0.0);
	if (i == 1) return vec3(0.4, 1.0, 0.0);
	if (i == 2) return vec3(0.0, 1.0, 0.4);
	if (i == 3) return vec3(0.0, 0.4, 1.0);
	if (i == 4) return vec3(0.4, 0.0, 1.0);
	if (i == 5) return vec3(1.0, 0.0, 0.4);

	return vec3(1.);
}

// Function 21
float fogColor(vec2 uvFog, float time)
{
    //the intensity of the fog
    return clamp(0.5 * (0.8 - uvFog.y +
	   					0.4 * (1.0 - length(uvFog)) * sin(sqrt(length(uvFog)) * 16.0 - time) + 
                       	0.5 * (1.0 - length(vec2(1.0, 0.0) - uvFog)) * sin(sqrt(length(vec2(1.0, 0.0) - uvFog)) * 12.0 - time * 0.7) + 
                       	0.5 * (1.0 - length(vec2(0.5, 0.0) - uvFog)) * sin(sqrt(length(vec2(0.5, 0.0) - uvFog)) * 7.0 - time * 0.9)          
    				), 0.0, 1.0);    
}

// Function 22
colorPair getColorPair(float x)
{
    int i = int(floor(numberOfColors * x));

    // Demonstrate intensity change
    if (i == 0) return colorPair(vec3(1.0, 0.0, 0.0),
                                 vec3(0.0, 1.0, 1.0));
    
    // Demonstrate saturation change
    if (i == 1) return colorPair(vec3(1.0, 0.5, 0.5),
                                 vec3(0.5, 0.5, 1.0));

    // Demonstrate hue change
    if (i == 2) return colorPair(vec3(1.0, 0.0, 0.0),
                                 vec3(1.0, 1.0, 0.0));

    return colorPair(vec3(0.), vec3(1.));
}

// Function 23
vec3 get_color(float color){
    return color == BLUE
    	? vec3(0.149,0.141,0.912)
    :color == GREEN
    	? vec3(0.000,0.833,0.224)
    :color == FOREST_GREEN
    	? rgb(34.0,139.0,34.0)
   	:color == WHITE
    	? vec3(1.0,1.0,1.0)
   	:color == GRAY
    	? vec3(192.0,192.0,192.0)/255.0
    :color == YELLOW
    	? vec3(1.0,1.0,0.0)
   	:color == LIGHTBLUE
    	? rgb(173.0,216.0,230.0)
   	:color == SKYBLUE
        ? rgb(135.0,206.0,235.0)
    :color == SNOW
    	? rgb(255.0,250.0,250.0)
    :color == WHITESMOKE
    	? rgb(245.0,245.0,245.0)
    :color == LIGHTGRAY
    	? rgb(211.0,211.0,211.0)
    :color == LIME
    	? rgb(0.0,255.0,0.0)
    :color == LIGHTYELLOW
    	? rgb(255.0,255.0,153.0)
    :color == BEIGE
    	? rgb(245.0,245.0,220.0)
    :color == TAN
    	? rgb(210.,180.,140.)
    :vec3(0);
}

// Function 24
vec3 parabolicColorMap(float x, vec3 rc, vec3 gc, vec3 bc){
    vec3 col = vec3(rc.x*x*x+rc.y*x+rc.z, 
                          gc.x*x*x+gc.y*x+gc.z, 
                          bc.x*x*x+bc.y*x+bc.z
                         )/vec3(255.0);
    col = pow(col, vec3(2.8)); // need to do this because I copied the colors from a non-gamma corrected jpg
    return col;
}

// Function 25
float color_to_val_3(in vec3 color) {
    return length(color);
}

// Function 26
vec3 sky_color(vec3 ray)
{
    return fogColor;
}

// Function 27
vec4 floatToColor(float value) {
    return vec4(value, value, value, 1.0);
}

// Function 28
vec2 colorize(vec3 p) {
	p.z=abs(2.-mod(p.z,4.));
	float es, l=es=0.;
	float ot=1000.;
	for (int i = 0; i < 15; i++) { 
		p=formula(vec4(p,0.)).xyz;
				float pl = l;
				l = length(p);
				es+= exp(-10. / abs(l - pl));
				ot=min(ot,abs(l-3.));
	}
	return vec2(es,ot);
}

// Function 29
vec3 getThemeColor(vec2 uv, float hue) {
    int shadeIdx = int(uv.x * 13.0);
    int swatchIdx = int((1.0 - uv.y) * 5.0);
    float seedChroma = 1000000.0;

    if (shadeIdx == 0) {
        return vec3(1.0);
    } else if (shadeIdx == 12) {
        return vec3(0.0);
    }

    if (iMouse.z > 0.0) {
        return generateShadeZcam(swatchIdx, shadeIdx, seedChroma, hue, 1.0);
    } else {
        return gamut_clip_preserve_lightness(generateShadeOklab(swatchIdx, shadeIdx, seedChroma, hue, 1.0));
    }
}

// Function 30
vec3 color(vec2 p) 
{
    return pal(hash(p), vec3(1.0), vec3(0.6), vec3(1.0), vec3(0.0, 0.333, 0.666));
}

// Function 31
vec3 GetMacBethColorCOLUMN_COUNT4(const in float yDist)
{
    float compareY = LINE_COUNT;
    
    if(yDist > --compareY)
        return FOLIAGE;
	else if(yDist > --compareY)
		return PURPLE;
	else if(yDist > --compareY)
		return YELLOW;
	else
		return NEUTRAL5;
}

// Function 32
vec3 color_blowout(in vec3 col)
{
    mat3 blowout = mat3(4.72376,  -8.85515,   3.84846,
                        -8.85515,  18.4378,   -8.71892,
                        3.84846,  -8.71892,   4.48226);
    vec3 cent = vec3(0.47968451, 
                     0.450743, 
                     0.45227517);

    
    vec3 dir = col - cent; // blowout * (col - cent);
    
#if DISTORT_INTENSE_COLORS
    dir = blowout * dir;
#endif
    
    vec3 maxes = (step(vec3(0.0), dir) - col)/dir;
    
    float amount = min(maxes.x, min(maxes.y, maxes.z));
    
    col = col + dir * amount;
    
	return col;
}

// Function 33
vec3 getColor_s(vec3 norm, vec3 pos, int objnr)
{
   vec3 col = getBrickColor_s(pos);
   return col;
}

// Function 34
float getColorComponent(float dist, float angle) {
    return
        pow((
            (
            	cos(
                    (angle * RAYS)
            		+ pow(
                        dist * 2.0,
                		(sin(iTime * SPEED) * TWIST_FACTOR)
            		) * 20.0
        		) + sin(
            		dist * RING_PERIOD
        		)
        	) + 2.0
        ) / 2.0, 10.0);
}

// Function 35
vec3 adjust_out_of_gamut_maxcomp(vec3 c)
{
    const float BEGIN_SPILL = 0.8;
    const float END_SPILL = 2.0;
    const float MAX_SPILL = 0.9; //note: <=1
    
    float mc = max(c.r, max(c.g, c.b));
    float t = MAX_SPILL * smootherstep( 0.0, END_SPILL-BEGIN_SPILL, mc-BEGIN_SPILL );
    return mix( c, vec3(mc), t);
}

// Function 36
vec4 color(vec2 uv)
{
  float fsize=1.0;
  #if FRINGE
  	float fringe = fract(floor(gl_FragCoord.y/fsize)*fsize/3.0);
  	vec3 fcol = 1.0-abs(fringe*3.0-vec3(0,1,2));
  	fcol = mix(fcol, vec3(1), 0.5);
  #else
	float fringe = 0.5;
    vec3 fcol=vec3(0.8);
  #endif
  float cx = (curve(time, 0.7)-0.5)*7.0;
  float cy = (curve(time, 0.8)-0.5)*3.0;

  vec3 s=vec3(cx,cy,-10);
  vec3 r=normalize(vec3(-uv,0.6 + curve(time, 0.3)));

  cam(s);
  cam(r);

  vec3 col = vec3(0);

  vec3 p=s;
  float dd=0.0;
  float side=sign(map(p));
  vec3 prod = vec3(1.0);
  int i=0;
  for(i=0; i<STEPS; ++i) {
    float d=map(p)*side;
    if(d<0.001) {
      
      vec3 n=norm(p)*side;
      vec3 l = normalize(vec3(-1));

      if(dot(l,n)<0.0) l=-l;

      vec3 h = normalize(l-r);

      float opa = mat;
      vec3 diff=mix(vec3(1), vec3(1,0.8,0.2), mat);
      vec3 diff2=mix(vec3(1), vec3(1,0.7,0.0), mat);
      float spec=mix(0.2, 1.5, mat);
      float fresnel = pow(1.0-max(0.0,dot(n,-r)),5.0);
      
      col += max(0.0, dot(n,l)) * (spec*(pow(max(0.0,dot(n,h)),50.0) * 0.5 + 0.5*diff2*pow(max(0.0,dot(n,h)),12.0)  )) * diff * prod;
      
      vec3 back = ref(reflect(r,n))*0.5*fresnel;
      col += back;

      side = -side;
      d = 0.01;
      r = refract(r,n,1.0 - 0.05*side*(0.5+0.5*fringe));
      prod *= fcol*0.9;
      if(opa>0.5) {
        /*vec3 back = ref(r)*1.0*fresnel;
        col = mix(col, back, prod);*/
        prod=vec3(0);
        break;
      }
    }
    if(dd>100.0) {
      dd=100.0;
      break;
    }
    p+=r*d;
    dd+=d;
  }
  if(i>99) {
    prod=vec3(0);
  }

  vec3 back = ref(r);
  col = mix(col, back, prod);
  //col *= 3;

  //col *= 3*pow(1-length(uv),0.7);
  vec2 auv = abs(uv)-vec2(0.5,0.0);
  col *= vec3(2.0*pow(1.0-clamp(pow(length(max(vec2(0),auv)),3.0),0.0,1.0),10.0));

  #if 1
    col = smoothstep(0.0,1.0,col);
    col = pow(col, vec3(0.4545));
  #endif
  
  //col = vec3( step(curve(uv.x, 0.04), uv.y*5) );
  //col = fcol;

  return vec4(col,1);
}

// Function 37
vec3 complex_color (vec2 v) {
  vec2 vn = normalize(v);
  vec3 cb = vec3(1.) + vec3(1,-.5,-.5)*vn.x + vec3(0.,.866,-.866)*vn.y;
  return cb/dot(cb,vec3(.213,.715,.072))*dot(v,v);
}

// Function 38
vec3 objcolor(vec3 p){
  return checkerboard(p.xz,0.4);
}

// Function 39
vec3 ColorTemperatureToRGB(float temperatureInKelvins)
{
	vec3 retColor;
	
    temperatureInKelvins = clamp(temperatureInKelvins, 1000.0, 40000.0) / 100.0;
    
    if (temperatureInKelvins <= 66.0)
    {
        retColor.r = 1.0;
        retColor.g = saturate(0.39008157876901960784 * log(temperatureInKelvins) - 0.63184144378862745098);
    }
    else
    {
    	float t = temperatureInKelvins - 60.0;
        retColor.r = saturate(1.29293618606274509804 * pow(t, -0.1332047592));
        retColor.g = saturate(1.12989086089529411765 * pow(t, -0.0755148492));
    }
    
    if (temperatureInKelvins >= 66.0)
        retColor.b = 1.0;
    else if(temperatureInKelvins <= 19.0)
        retColor.b = 0.0;
    else
        retColor.b = saturate(0.54320678911019607843 * log(temperatureInKelvins - 10.0) - 1.19625408914);

    return retColor*vec3(1.04,1.,1.);
}

// Function 40
vec3 sky_color(vec3 ray)
{
    float elev = atan(ray.y);
    float azimuth = atan(ray.x, ray.z);
 
    vec3 sky = ambientColor + vec3(0.4, 0.22, 0.05)*2.5*(0.65 - elev);
    
    // Clouds
    #ifdef procedural_clouds
    float cloudst = smoothstep(-0.2, 0.5, elev)*smoothstep(0.1, 0.97, noise2(11.*cloudSize*ray + vec3(cloudSpeed*currTime)));
    #else
    float cloudst = smoothstep(-0.2, 0.5, elev)*texture(iChannel1, cloudsize*ray.xy).r;
    #endif
    sky = mix(sky, 0.45 + 0.6*vec3(cloudst), smoothstep(0.12, 0.5, cloudst)) + 0.3*vec3(smoothstep(0.2, 0.8, cloudst));
    
    // Ground
    vec3 grass = vec3(0.05, 0.45, 0.3) + vec3(0.19, 0.13, -0.03)*2.7*(0.65 - elev);
    grass = grass*(0.6 + 2.*abs(elev)*texture(iChannel0, 12.*ray.xy).r);
    
    return mix(mix(grass, vec3(0.65)*(0.7 + 0.3*texture(iChannel0, 12.*ray.xy).r), smoothstep(-0.17 - 0.035*abs(azimuth), -0.172 - 0.035*abs(azimuth), elev)), sky, smoothstep(-0.0003, 0.0003, elev)) + getFlares(ray); 
}

// Function 41
vec3 getColor(vec2 p) {

    vec3 camUp = vec3(0,-1,0);
    vec3 camTar = vec3(0.);
    vec3 camPos = vec3(0,0,1.25);

    mat3 camMat = calcLookAtMatrix(camPos, camTar, camUp);
    float focalLength = 2.;
    vec3 rayDirection = normalize(camMat * vec3(p, focalLength));

    CastRay castRay = CastRay(camPos, rayDirection);
    Hit hit = raymarch(castRay);

    #ifdef DEBUG
        return hit.normal * .5 + .5;
    #endif

    if ( hit.isBackground || ! hit.model.material.transparent) {
        return shadeSurface(hit);
    }

    return shadeTransparentSurface(hit);    
}

// Function 42
vec3 getColor(vec2 p) {
    int size = 3;
    float filt[9];
    filt[0] = 0.33;
    filt[1] = 0.33;
    filt[2] = 0.33;
    filt[3] = 0.33;
    filt[4] = 0.33;
    filt[5] = 0.33;
    filt[6] = 0.33;
    filt[7] = 0.33;
    filt[8] = 0.33;
    vec3 c = vec3(0);
    for (int x = 0; x < size; x += 1) {
        for (int y = 0; y < size; y += 1) {
            float xx = float(x)-float(size)/2.;
            float yy = float(y)-float(size)/2.;
            int index = x+(y*size);
            c += getPixel(p+vec2(floor(xx)+1., floor(yy)+1.))*filt[index];
        }
    }
    return c*0.33333;
}

// Function 43
vec3 getColor(vec2 coords){
    coords.x = coords.x-mod(coords.x, size);
    coords.y = coords.y-mod(coords.y, size);
    
    float r = randomize(coords.xy+vec2(sin(iTime*0.5)));
    float g = randomize(coords.xy * 20.0+vec2(sin(iTime*0.5)));
    float b = randomize(coords.xy * 37.0+vec2(sin(iTime*0.5)));
    return vec3(r,g,b);
}

// Function 44
vec4 colorWheel(float theta)
{
    float thetamod = mod(theta, 2.0*PI)/(2.0*PI);
    return vec4(colorSpike(thetamod, 0.0) + colorSpike(thetamod, 1.0),
                colorSpike(thetamod, 1.0/3.0) + colorSpike(thetamod, 4.0/3.0),
                colorSpike(thetamod, 2.0/3.0) + colorSpike(thetamod, -1.0/3.0),
                1.0);
}

// Function 45
vec3 sat_adjust( vec3 rgbIn, float SAT_FACTOR)
{
vec3 RGB2Y = vec3(RGBtoXYZ( REC709_PRI)[0][1], RGBtoXYZ( REC709_PRI)[1][1], RGBtoXYZ( REC709_PRI)[2][1]);
mat3 SAT_MAT = calc_sat_adjust_matrix( SAT_FACTOR, RGB2Y);
return SAT_MAT * rgbIn;
}

// Function 46
PathColor ColorSub( PathColor a, PathColor b )
{
#if SPECTRAL    
    return PathColor( a.fIntensity - b.fIntensity );
#endif    
#if RGB
    return PathColor( a.vRGB - b.vRGB );
#endif    
}

// Function 47
vec3 reflectedColor(in vec3 p, in vec3 rd){
    
    vec3 Ks = vec3(0.7); // specular reflected intensity
    float shininess = 40.0;
    
   	vec3 n = gradient( p );
    vec3 ref = reflect( rd, n );
    vec3 rc = vec3(0);
    
    vec3 light_pos   = vec3( 15.0, 20.0, 5.0 );
	vec3 light_color = vec3( 1.0, 1.0, 1.0 );
	vec3 vl = normalize( light_pos - p );
	vec3 specular = vec3( max( 0.0, dot( vl, ref ) ) );
    vec3 F = fresnel( Ks, normalize( vl - rd ), vl );
	specular = pow( specular, vec3( shininess ) );
	rc += light_color * specular; 
    return rc;
}

// Function 48
vec3 	UIStyle_ColorPickerSize()		{ return vec3(128.0, 128.0, 32.0); }

// Function 49
vec3 obj_color(vec3 norm, vec3 pos)
{
    pos.xy = rotateVec(pos.xy, 0.15*smoothstep(40., 200., iTime)*cos(rot*2.));
    pos.zy = rotateVec(pos.zy, 0.15*smoothstep(40., 200., iTime)*sin(rot*2.));
    
    float vf1 = smoothstep(0.15, 0.16, abs(pos.y));
    vf2 = smoothstep(0.58, 0.59, abs(pos.y))*smoothstep(1.45, 1.42, abs(pos.y));
    float vf3 = smoothstep(0.255, 0.23, length(vec2(angle3, 3.8*abs(pos.y - scrpos))));
    
    vec3 metalcolor = mix(vec3(0.2), vec3(.8, .65, .5), vf1);
    metalcolor = mix(metalcolor, vec3(1.), vf2);
    metalcolor = mix(metalcolor, vec3(0.85, 0.5, 0.4), vf3);
    
    vec3 diffcolor = mix(vec3(0.45, 0.44, 0.43), vec3(1., 0.63, 0.33), vf1);
    diffcolor = mix(diffcolor, vec3(0.95), vf2);
    diffcolor = mix(diffcolor, vec3(0.7, 0.4, 0.35), vf3);
    
    float mdmix = mix(0.85, 0.3, smoothstep(0.15, 0.16, abs(pos.y)));
    mdmix = mix(mdmix, 0.1, vf2);
    mdmix = mix(mdmix, 0.65, vf3);
    
    vec2 posr = rotateVec(pos.xz, rot) + 0.25*pos.yy;
    vec3 ocol = mix(1.1*metalcolor*texture(iChannel0, reflect(campos, norm)).rgb, diffcolor, mdmix) + 0.03;
    
    // Texture
    vec3 textcolor = mix(vec3(0.), vec3(0.6, 0.4, 0.4), smoothstep(0.155, 0.16, abs(pos.y))
                     *smoothstep(0.19, 0.225, length(vec2(angle3, 3.2*abs(pos.y - scrpos)))));
    ocol*= (1.05 - 0.8*textcolor*smoothstep(0.4, 0.7, texture(iChannel1, posr).rrr)
          *(1.2 - 0.8*smoothstep(0.14, 0.4, texture(iChannel1, 2.4*posr).rrr)));
    
    return ocol;
}

// Function 50
vec3 cellColor(float c)
{
    float r = 0.5*sin(c*0.7)+0.5;
    float g = 0.5*cos(c*0.6)+0.5;
    float b = 0.5*sin(c*c*0.1)+0.5;
    return vec3(r,g,b);
}

// Function 51
vec3 backgroundColor(vec3 dir){
    return vec3(0.);
}

// Function 52
vec3 ground_color(vec3 p)
{
    //p /= 10.0;
    float color1 = length(sin(p/100.0))/2.0;
    return vec3(color1,color1/1.8+sin(length(p)/10.0)/20.0,color1/2.0);
}

// Function 53
vec4 getBitmapColor( in vec2 uv )
{
	return getColorFromPalette( getPaletteIndex( uv ) );
}

// Function 54
vec3 getPaletteColor( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d ){
    return a + b*cos(TAU*(c*t+d) );
}

// Function 55
vec3 getBrickColor(vec3 pos)
{
    vec3 brickCol1 = vec3(0.60, 0.27, 0.15);
    vec3 brickCol2 = vec3(0.45, 0.24, 0.13);
    vec3 brickCol3 = vec3(0.29, 0.10, 0.04);
    
    vec3 pos2 = pos;
    pos2.yz+= 0.07*texture(iChannel1, pos.yz*0.005).g;
    pos2.z+= 0.5*brickStep.z*floor(0.5*pos2.y/brickStep.y);
    vec2 nb = floor(pos2.yz/brickStep.yz*vec2(0.5, 1.));
    float nbBrick = nb.x + nb.y*90.;
    float nbBrickf = pow(hash(nbBrick), 6.);
    vec3 brickCol = mix(brickCol1, brickCol2, nbBrickf);
    brickCol = mix(brickCol, brickCol3, pow(smoothstep(0.1, 1.05, 1.2*texture(iChannel2, pos.yz*0.18).x*texture(iChannel2, pos.yz*0.23).x), 1.5));
    brickCol*= 0.4 + 0.6*smoothstep(0.80, 0.60, texture(iChannel1, pos.yz*0.07).b);
    return brickCol;
}

// Function 56
vec3 skyColor(vec3 rd) {
  vec3 sunDir = sunDirection();

  float sunDot = max(dot(rd, sunDir), 0.0);
  
  vec3 final = vec3(0.0);

  final += mix(skyCol1, skyCol2, rd.y);

  final += 0.5*sunCol1*pow(sunDot, 30.0);

  final += 4.0*sunCol2*pow(sunDot, 300.0);
    
  return final;
}

// Function 57
vec3 sinColor(float value)
{
    value *= TAU;
    vec3 color;
    
    color.r = (1.0 + cos(value)) / 2.0;
    color.g = (1.0 + cos(value - TAU / 3.0)) / 2.0;
    color.b = (1.0 + cos(value + TAU / 3.0)) / 2.0;
    
    return color;
}

// Function 58
vec4 interpolateColor(float phaseLim, float w, float a)
{
    vec4 textureColor;
    float phaseLim1 = phaseLim - w;
    float phaseLim2 = phaseLim + w;

    if (a < phaseLim1)
    {
        textureColor = textureColor0;
    }
    else
    if (a > phaseLim2)
    {
        textureColor = textureColor1;
    }
    else
    {
        float p = (a - phaseLim1) / (phaseLim2 - phaseLim1);
        textureColor = mix(textureColor0, textureColor1, p);
    }
    return textureColor;
}

// Function 59
vec3 hsvColorscale (float x) {
    vec3 c1 = rgb2hsv(1.0 - color1);
    vec3 c2 = rgb2hsv(1.0 - color2);
    return 1.0 - hsv2rgb(mix(c1, c2, x));
}

// Function 60
vec3 blackBodyColor(float k){
 vec3 c=vec3(1.,3.375,8.)/(exp((19e3*vec3(2,3,4)/k/64e3))-1.);
 //the -1 barely changes much, could be cut.
 return c/mav(c);}

// Function 61
vec4 DistanceToBoundaryAndColor(float x)
{
    // There is a boundary at x=0 and x=1. return the distance of whichever is closer.
    // Also give the color at that boundary.
    if (x <= 0.5)
        return vec4(x, colorA);
    else
        return vec4(1.0 - x, colorB);
}

// Function 62
vec3 GetColor(ColorArray _tmp,float _m){
	vec2 weight = vec2(0.);
    for(int i = 0;i<4;i++)
    	if(float(i) == _m)
  			weight = Weight[i];
    return floor((_tmp.A*weight.x + _tmp.B*weight.y)/128.);
}

// Function 63
vec3 bgColor(vec3 rayDir)
{
    float b = map(rayDir.x, -1., 1., 0., 1.);
	return vec3( .18, .18, b );
}

// Function 64
vec3 getSkyColor(vec3 e) {
    e.y = max(e.y,0.0);
    return vec3(.3, 1.0-e.y, 0.6+(1.0-e.y)*0.9);
}

// Function 65
vec3 lightColor() {
    return vec3(1.0, 0.95, 0.9);
}

// Function 66
vec3 GetSkyColor(vec3 rayDirection)
{
    
    float p = atan(rayDirection.x, rayDirection.z);
    p = p > 0.0 ? p : p + 2.0 * PI;
    vec2 uv = vec2(p / (2.0 * PI), acos(rayDirection.y) / PI);

    float yCapValue = 0.05 + .01 * sin(uv.x + iTime * 0.1);
    float yLerpValue = min(rayDirection.y, yCapValue) / yCapValue;
    vec3 skyColor = 1.3 * mix(vec3(0.02, 0.26, 0.64), vec3(0.04, 0.04, 0.6), yLerpValue);

    vec3 cloudTexture = texture(iChannel1, uv + vec2(iTime * 0.001, 0.0)).bgr;
    skyColor = skyColor * cloudTexture;
    
    vec2 auroraCenter = vec2(0.525, 0.5);
    float distanceFromAuroraCeter = length(uv - auroraCenter);
    float ring1Start = 0.05;
    float ring1End = 0.1;
    float distanceFromAuroraCeter1 = distanceFromAuroraCeter + 
        0.003 * sin(uv.y * 100.0 + iTime * 1.1) +
        0.001 * noise(vec3(uv * 200.0, iTime * 1.1));
    
    float yDistFromCenter = abs(uv.y - auroraCenter.y); 

    if(distanceFromAuroraCeter1 > ring1Start && distanceFromAuroraCeter1 < ring1End)
    {
        float ring1Center = ring1Start * 0.75 + ring1End * 0.25;
        float ring1Length = (distanceFromAuroraCeter1 < ring1Center) ? 
            (ring1Center - ring1Start) : (ring1End - ring1Center);
        float multiplier = 1.0 - (abs(distanceFromAuroraCeter1 - ring1Center) / ring1Length); 
        multiplier = pow(multiplier, 2.0);
        float borealisLerpValue = noise(vec3(uv.x * 200.0, 0.0, 0.0));
        vec3 borealisColor = mix(vec3(0.0, 0.4, 0.15), vec3(0.0, 0.2, 0.02), borealisLerpValue);
        skyColor += multiplier * borealisColor;
    }
    
    float ring2Start = 0.07;
    float ring2End = 0.1;
    float distanceFromAuroraCeter2 = distanceFromAuroraCeter + 
        0.005 * sin(uv.y * 80.0 + iTime * 1.5) +
        0.003 * noise(vec3(uv * 200.0, iTime * 1.1));
    if(distanceFromAuroraCeter2 > ring2Start && distanceFromAuroraCeter2 < ring2End)
    {
        float ring2Center = ring2Start * 0.75 + ring2End * 0.25;
        float ring2Length = (distanceFromAuroraCeter2 < ring2Center) ? 
            (ring2Center - ring2Start) : (ring2End - ring2Center);
        float multiplier = 1.0 - (abs(distanceFromAuroraCeter2 - ring2Center) / ring2Length); 
        multiplier = pow(multiplier, 2.0);
        float borealisLerpValue = uv.y * noise(vec3(0.0, uv.x * 250.0, 0.0));
        vec3 borealisColor = mix(vec3(0.3, 0, 0.13), vec3(0.12, 0.0, 0.05), borealisLerpValue);
        skyColor += multiplier * borealisColor;
    }
    
    float ring3Start = 0.11;
    float ring3End = 0.16;
    float distanceFromAuroraCeter3 = distanceFromAuroraCeter + 
        0.004 * sin(uv.y * 120.0 + iTime * 0.9) +
        0.003 * noise(vec3(uv * 200.0, iTime));
    if(distanceFromAuroraCeter3 > ring3Start && distanceFromAuroraCeter3 < ring3End)
    {
        float ring3Center = ring3Start * 0.75 + ring3End * 0.25;
        float ring3Length = (distanceFromAuroraCeter3 < ring3Center) ? 
            (ring3Center - ring3Start) : (ring3End - ring3Center);
        float multiplier = 1.0 - (abs(distanceFromAuroraCeter3 - ring3Center) / ring3Length); 
        multiplier = pow(multiplier, 2.0);
        float borealisLerpValue = uv.y * noise(vec3(0.0, uv.x * 250.0, 0.0));
        vec3 borealisColor = mix(vec3(0.05, 0.25, 0.15), vec3(0.0, 0.1, 0.2), borealisLerpValue);
        skyColor += multiplier * borealisColor;
    }
    
    clamp(skyColor, 0.0, 1.0);
    return skyColor;
}

// Function 67
vec3 ReturnColor_2(float r1, float r2){
 return (_Color1+_Color3+  (_Color2 - _Color1) * r1 + (_Color4 - _Color3) *r2)/2.;
  
}

// Function 68
vec3 getSkyColor( vec3 rd )
{
    vec3 color = mix( SKY_COLOR_1 * 2.4, SKY_COLOR_2, rd.y / 10.0 );
	
    float fogFalloff = clamp( 9.0 * rd.y, 0.0, 2.0 );
    color = mix( FOG_COLOR, color, fogFalloff );
    color = mix( color, GRID_COLOR_1, smoothstep( -0.2, -0.3, rd.y ) );

    vec3 sunDir = normalize( SUN_DIRECTION );
    float sunGlow = smoothstep( 1.0, 2.0, dot( rd, sunDir ) );
        
    rd = mix( rd, sunDir, -2.0 ); // easier to bend vectors than fiddle with falloff :P
    float sun = smoothstep( 1.087, 1.09, dot(rd, sunDir ) );
    sun -= smoothstep( 0.2, 1.0, 0.6 );			        
    
    float stripes = mod( 60.0 * ( pow( rd.y + 0.25, 2.5 ) ) + 0.6, 2.0 ) -0.6;
    stripes = smoothstep( 0.3, 0.31, abs( stripes ) );
        
    
    // based on https://www.shadertoy.com/view/tssSz7
    vec2 starTile   = floor( rd.xy * 50.0 );
    vec2 starPos    = fract( rd.xy * 50.0 ) * 3.0 - 2.0;
    vec2 starRand = hash22( starTile );
    starPos += starRand * 3.0 - 2.0;
    float stars = saturate( 2.0 - ( ( sin( iTime * 2.0 + 60.0 * rd.y ) ) * 0.6 + 7.0 ) * length( starPos ) );
    stars *= step( 0.1, -sun );
    stars *= step( 1.0, starRand.x );
    stars *= 6.0;
           
    sun = 3.0 * clamp( sun * stripes, 0.0, 2.0 );
    
    vec3 sunCol = 4.0 * mix( SUN_COLOR_1, SUN_COLOR_2, -( rd.y - 0.2 ) / 0.4 );
    color = mix( color, sunCol, sun );

	color = mix( FOG_COLOR, color, 0.9 + 0.3 * fogFalloff );
    color = mix( color, sunCol, 0.35 * sunGlow );
    
    color += stars;

    // return vec3(stripes);
    // return vec3(sun);
    // return vec3(sunGlow);
    return color;
}

// Function 69
vec3 smoke_color(float x)
{
    return vec3(.5, .5, .5) * x;
}

// Function 70
vec3 color(vec3 p, float time) {
    
    int[5] s; // short for sigma, used in math to represent permutations
    for (int j=0;j<5;j++) {
        s[j]=j;
    }
    
    int t;  // used as temp space by swap
    
    float period = 0.530637531;
    time = mod(time, 6.0 * period);
    while (time > 0.5 * period) {
        time -= period;
        swap(s[0],s[2]);
        swap(s[2],s[4]);
        swap(s[1],s[3]);
    }
    
    
    p = refl(p,vec3(0.0,1.0,0.0));
    p = refl(p,vec3(sinh(time),cosh(time),0));
    if (rdot(p,p) > -0.0) {
        // floating point error, I think, is causing invalid points
        return vec3(1.0,0.0,1.0);
    }
    // mirrors basically pulled from the Python code I wrote ages ago
    vec3 mirror1 = vec3(0.0,1.0,0.0);
    vec3 mirror2 = vec3(0.0,-sqrt(0.5),sqrt(0.5));
    vec3 mirror3 = vec3(0.5558929702514214, 0.0, -1.1441228056353687);
    

    
    int i;
    for (i=0;i<100;i++) {
        if (rdot(p,mirror1) > 0.0001) {
            p = refl(p,mirror1);
            swap(s[2],s[4]);
            continue;
        }
        if (rdot(p,mirror2) > 0.0001) {
            p = refl(p,mirror2);
            swap(s[1],s[4]);
            swap(s[2],s[3]);
            continue;
        }
        if (rdot(p,mirror3) > 0.0001) {
            p = refl(p,mirror3);
            swap(s[0],s[1]);
            swap(s[2],s[4]);
            continue;
        }
        if (s[0]==0) {return vec3(1.0,0.0,0.0);}
        if (s[0]==1) {return vec3(1.0,1.0,0.0);}
        if (s[0]==2) {return vec3(0.0,1.0,0.0);}
        if (s[0]==3) {return vec3(1.0,0.5,0.0);}
        if (s[0]==4) {return vec3(0.0,0.0,1.0);}
    }
    return vec3(1.0,1.0,1.0);
}

// Function 71
vec3 color(vec2 p) {
    //const float varia = 0.02;
    const float varia = 0.3;
    return pal(2.+iTime/3.434+hash2(p).x*varia, 
               vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,0.7,0.4),vec3(0.0,0.15,0.20)+(sound(p).x-0.5)*2.0  );
}

// Function 72
vec3 color(vec3 ro, vec3 rd, vec3 norm, float t)
{
    vec3 p = ro + rd * t;
  
    vec2 f = floor(vec2(p.x - iTime*0.5, p.z - 0.4) * 1.0);
   
    vec3 col = abs(1.0 - smoothstep(-1.4,-0.9, p.y)) * 
        vec3(1) * mod(f.x - f.y, 2.0) * 0.0057;
    
     // Shadowing
    col *= max(0.0055, smoothstep(0.2, 1.9, 0.2 / exp(-t + 2.0)));
    
    col *= 2.0;
    return col;
    
}

// Function 73
void write_color(vec4 rgba, float w) {
    float src_a = w * rgba.a;
    float dst_a = _stack.premultiply?w:src_a;
    _color = _color * (1.0 - src_a) + rgba.rgb * dst_a;    
}

// Function 74
vec3 scene_object_color( vec3 r, mat2x3 Kr, float t, float r0, vec3 N, vec3 V, vec3 albedo,
                         bool submerged, int index )
{
    float h = length( r ) - r0;
    vec3 Z = normalize( r );
    vec3 skyL = texelFetch( iChannel2, ivec2( iResolution.x - 6., int( iResolution.y ) / 2 + 2 * index ), 0 ).xyz;
    vec3 skyR = texelFetch( iChannel2, ivec2( iResolution.x - 4., int( iResolution.y ) / 2 + 2 * index ), 0 ).xyz;
    vec4 skyZ = texelFetch( iChannel2, ivec2( iResolution.x - 2., int( iResolution.y ) / 2 + 2 * index ), 0 );
    vec3 TL = atm_transmittance( Z * max( g_data.radius, length( r ) ), g_env.L, g_atm, true );
    vec3 F = g_env.sunlight;
    vec3 L = g_env.L;
    if( submerged )
    {
        float d = max( 0.001, -.25 * h );
        vec4 M = scene_ocean_normal_and_lensing( r, t, h, d, V, Z );
        F = F * M.w;
        F = F * saturate( 1. - fresnel_schlick( .02, dot( M.xyz, L ) ) );
        L = normalize( -simple_refract( -L, Z ) );
    }
    F *= ts_shadow_sample( g_ts, iChannel1, r ) * TL * skyZ.w;
    float objshadow = scene_raycast_object_shadows( Ray( r, L ) );
    float slope = length( N / dot( N, Z ) - Z );
    vec3 ground = trn_albedo( r, 4. * h, h, slope, Z.z, false ) * ( F * dot( L, Z ) + skyZ.xyz );
    if( submerged )
    {
        skyZ.xyz = .75 * skyZ.xyz + .125 * skyL + .125 * skyR;
        skyL = .125 * skyL + .375 * ( skyZ.xyz + ground );
        skyR = .125 * skyR + .375 * ( skyZ.xyz + ground );
    }
    return scene_object_lighting( albedo, N, L, V, Z, F * objshadow, skyZ.xyz, skyL, skyR, ground );
}

// Function 75
vec3 getObjectColor(vec3 p, vec3 n){
    
    
    //p.xy -= path(p.z);
    float sz0 = 1./2.;
    
    // Texel retrieval.
    vec3 txP = p;
    //txP.xz *= r2(getRndID(svVRnd)*6.2831);
    vec3 col = tex3D(iChannel0, txP*sz0, n );
    col = smoothstep(-.0, .5, col);//*vec3(.5, .8, 1.5);
    col = mix(col, vec3(1)*dot(col, vec3(.299, .587, .114)), .5);
    // Darken the surfaces to bring more attention to the neon lights.
    col /= 16.;
    
   
    // Unique random ID for the hexagon pylon.
    float rnd = getRndID(svV2Rnd);
    
    // Subtly coloring the unlit hexagons... I wasn't feeling it. :)
    //if(svLitID==1. && rnd<=.0) col *= vec3(1, .85, .75)*4.;

    // Applying the glow.
    //
    // It's took a while to hit upon the right combination. You can create a cheap lit object 
    // effect by simply ramping up the object's color intensity. However, your eyes can tell that
    // it's lacking that volumetric haze. Volumetric haze is achievable via a volumetric appoach.
    // However, it's prone to patchiness. The solutionm, of course, is to combine the smoothness
    // of direct object coloring with a portion of the glow. That's what is happining here.

    // Object glow.
    float oGlow = 0.;
    
    // Color every lit object with a gradient based on its vertical positioning.
    if(rnd>0. && svLitID==1.) {
        
        float ht = hexHeight(svV2Rnd);
    	ht = floor(ht*4.99)/4./2. + .02;
        const float s = 1./4./2.*.5; // Four levels, plus the height is divided by two.
     
        oGlow = mix(1., 0., clamp((abs(p.y - (ht - s)))/s*3.*1., 0., 1.));
        oGlow = smoothstep(0., 1., oGlow*1.);
    }
    
    // Mix the object glow in with a small potion of the volumetric glow.
    glow = mix(glow, vec3(oGlow), .75);
    
    // Colorizing the glow, depending on your requirements. I've used a colorful orangey palette,
    // then have modified the single color according to a made up 3D transcental function.
    //glow = pow(vec3(1, 1.05, 1.1)*glow.x, vec3(6, 3, 1));
    glow = pow(vec3(1.5, 1, 1)*glow, vec3(1, 3, 6)); // Mild firey orange.
    glow = mix(glow, glow.xzy, dot(sin(p*4. - cos(p.yzx*4.)), vec3(.166)) + .5); // Mixing in some pink.
    glow = mix(glow, glow.zyx, dot(cos(p*2. - sin(p.yzx*2.)), vec3(.166)) + .5); // Blue tones.
    //glow = mix(glow.zyx, glow, smoothstep(-.1, .1, dot(sin(p + cos(p.yzx)), vec3(.166))));
     
    #ifdef GREEN_GLOW 
    glow = glow.yxz;
    #endif
    
   
    return col;
    
}

// Function 76
vec3 getSpaceColor( in vec3 dir ) {
    float scanline = sin(dir.z * 700.0 - iTime * 5.1)*0.5+0.5;
    scanline *= scanline;
    vec3 color = mix(vec3(0.1, 0.16, 0.26), vec3(0.1), scanline);
    vec2 uv = vec2(atan(dir.y, dir.x) / (2.0 * PI) + 0.5, mod(dir.z, 1.0));
    uv.x = mod(uv.x+2.0*PI, 1.0);
    uv.x *= 100.0;
    uv.y *= 15.00;
    uv *= rot(1.941611+iTime*0.00155);
    vec2 center = floor(uv) + 0.5;
    center.x += noise(center*48.6613) * 0.8 - 0.4;
    center.y += noise(center*-31.1577) * 0.8 - 0.4;
    float radius = smoothstep(0.6, 1.0, noise(center*42.487+
                                              vec2(0.1514, 0.1355)*iTime)); 
    radius *= 0.01;
    vec2 delta = uv-center;
    float dist = dot(delta, delta);
    float frac = 1.0-smoothstep(0.0, radius, dist);
    float frac2 = frac;
    frac2 *= frac2; frac2 *= frac2; frac2 *= frac2;
    vec3 lightColor = mix(vec3(0.988, 0.769, 0.176), 
                          vec3(0.988, 0.434, 0.875), noise(center*74.487));
    return mix(color, lightColor, frac) + vec3(1)*frac2;
}

// Function 77
vec3 backgroundColor(vec3 dir){
    return texture(iChannel0, dir).rgb;
}

// Function 78
vec3 getColor(in vec2 uv, in sampler2D tex) {
    vec3 c0 = texture(tex, uv).rgb;
    vec3 c1 = vec3(0.);
/*
    float bluramount = 1.0 / 800.0;
    const float repeats = 32.0;
    float strata = 1.0/repeats;
    for (float i = 0.; i < repeats; i++) { 
    	float xi = strata*(float(i)+rnd());
        float angle = degrees(xi*360.);
        //vec2 q = vec2(cos(degrees((grid(i,dists)/repeats)*360.)),sin(degrees((grid(i,dists)/repeats)*360.))) * (1./(1.+mod(i,dists)));
        vec2 q = vec2(cos(angle),sin(angle)) *  sqrt(rnd()+bluramount); 
        vec2 uv2 = uv+(q*bluramount);
        c1 += texture(tex,vec2(uv2.x,uv2.y)).rgb/2.;
        
        
        //One more to hide the noise.
        q = vec2(cos(angle),sin(angle)) *  sqrt(rnd()+bluramount); 
        uv2 = uv+(q*bluramount);
        c1 += texture(tex,vec2(uv2.x,uv2.y)).rgb/2.;
    }
    c1 /= repeats;*/
    
    return mix(c0, c1, 0.0);
}

// Function 79
vec4 colorFilter(vec4 c)
{
	float g = (c.x + c.y + c.z) / 3.0;
    c = vec4(g,g,g,1.0);
    
    c.x *= 0.3;
    c.y *= 0.5;
    c.z *= 0.7;
    
    return c;
}

// Function 80
vec3 get_color(vec2 initial_pos) {
    Body p;
    p.pos = initial_pos;
    p.vel = vec2(0);
    p.acc = vec2(0);
    
    vec3 color = vec3(0);
    
    for (int i = 0; i < NOF_ITERATIONS; i++) {
       p = update_body(p, TIMESTEP);
       color += get_closest_attractor_color(p.pos);
    }
    
    return color / float(NOF_ITERATIONS);
}

// Function 81
vec4 colorScreen( in vec2 pos ) 
{	
	//Beat Waves
	float modfactor = 60.0/BPM;
	
	float distToOutput = length(pos); 
	float maxradius = 0.5*REACTABLE_RADIUS;
	float actualWaveRadius = maxradius*mod(iChannelTime[0],modfactor)/modfactor;
	
	float beatWeight = 0.4+0.6*(1.0 - step(0.5,mod(iChannelTime[0],4.0*modfactor)/modfactor));
	
	float waveValue = 1.0 - smoothstep(0.0,0.05,abs(distToOutput-actualWaveRadius));
	float factor = beatWeight * (maxradius-actualWaveRadius) * waveValue; 
	
	//OUTPUT
	factor = mix(1.0,factor,step(SCREEN_OUTPUT_RADIUS,distToOutput));

	float obj1 = factorObj(pos,module1Pos.xz,vec2(0.0), connection.z,vec2(1.0,0.0));
	float obj2 = factorObj(pos,module2Pos.xz,connection.xy,1.0,vec2(1.0,0.0));
	float obj3 = factorObj(pos,module3Pos.xz,vec2(0.0),1.0,vec2(0.5,0.5));
	float obj4 = factorObj(pos,module4Pos.xz,module3Pos.xz,1.0,vec2(0.0,1.0));
	
	factor = factor + obj1 + obj2 + obj3 + obj4;
			
	//FINAL SCREEN MIX
	return mix(vec4(0.00,0.00,0.22,0.05),vec4(0.50,0.50,0.50,0.2), factor);
}

// Function 82
vec3 getBackgroundColor( const vec3 ro, const vec3 rd ) {	
	return 1.4*mix(vec3(.5),vec3(.7,.9,1), .5+.5*rd.y);
}

// Function 83
vec4 sky_color(vec3 ray)
{
    return vec4(0);
}

// Function 84
void DecompressColor(in vec4 color, out vec3 diffuseColor, out vec3 specularColor, out float depth, out int materialID)
{
    depth = fract(color.w);
    materialID = int(color.w - depth);
	depth *= SCENE_MAX_T; 
    
    diffuseColor = fract(color.rgb);
    specularColor = (color.rgb - diffuseColor) / 1000.0;
}

// Function 85
vec3 getColor(vec2 uv, int tex) {
    vec3 c;
    if(tex==0) c = texture( iChannel0, uv ).xyz; else
    if(tex==1) c = texture( iChannel1, uv ).xyz; else
    if(tex==2) c = texture( iChannel2, uv ).xyz; else
    if(tex==3) c = texture( iChannel3, uv ).xyz; else
    if(tex==4) c = getCueTexture(uv); else
    if(tex==5) c = getFabricColor(uv); else
    if(tex > 5) {
        int ballid = tex - 6;
    	c = getBallTexture(uv, ballcolors[ballid], ballid);
    }
    return clamp(c, 0.0, 1.0);
}

// Function 86
vec3 color(in ray r) {
    vec3 col = vec3(0);
    vec3 emitted = vec3(0);
	hit_record rec;
    
    for (int i=0; i<MAX_RECURSION; i++) {
    	if (world_hit(r, EPSILON, MAX_FLOAT, rec)) {
            ray scattered;
            vec3 attenuation;
            vec3 emit = material_emitted(rec);
            emitted += i == 0 ? emit : col * emit;
            
            if (material_scatter(r, rec, attenuation, scattered)) {
                col = i == 0 ? attenuation : col * attenuation;
                r = scattered;
            } else {
                return emitted;
            }
	    } else {
            return emitted;
    	}
        if(dot(col,col) < 0.0001) return emitted; // optimisation
    }
    return emitted;
}

// Function 87
vec3 SingleColorChannelGrayScale(vec3 color)
{
    return vec3(color.r);
    // return vec3(color.g);
    // return vec3(color.b);
}

// Function 88
vec4 getColorFromPaletteData(PaletteData_13 palette, int i) {
	if (i == 0) return palette.c0;
	if (i == 1) return palette.c1;
	if (i == 2) return palette.c2;
	if (i == 3) return palette.c3;
	if (i == 4) return palette.c4;
	if (i == 5) return palette.c5;
	if (i == 6) return palette.c6;
	if (i == 7) return palette.c7;
	if (i == 8) return palette.c8;
	if (i == 9) return palette.c9;
	if (i == 10) return palette.c10;
	if (i == 11) return palette.c11;
	if (i == 12) return palette.c12;
	return vec4(0.0);
}

// Function 89
vec4 getColorAt(vec2 pos, vec2 fragCoord){
    vec2 uv = vec2(pos);
    applyDistortion(uv, fragCoord, .008);
    
    vec4 color = texture(iChannel0, uv);
    
    applyGray(color.rgb, abs(sin(iTime)));
    return color;
}

// Function 90
vec3 GetColorForRay(in vec3 rayPos, in vec3 rayDir, out float hitDistance, int panel, in vec2 pixelPos)
{
    // trace primary ray
	SRayHitInfo hitInfo = RayVsScene(rayPos, rayDir, panel, pixelPos);
    
    // set the hitDistance out parameter
    hitDistance = hitInfo.dist;
    
    if (hitInfo.dist == c_rayMaxDist)
        return texture(iChannel0, rayDir).rgb;
    
    // calculate where the pixel is in world space
	vec3 hitPos = rayPos + rayDir * hitInfo.dist;
    hitPos += hitInfo.normal * c_hitNormalNudge;

    // shoot a shadow ray    
    SRayHitInfo shadowHitInfo = RayVsScene(hitPos, c_lightDir, panel, pixelPos);
    float shadowTerm = (shadowHitInfo.dist == c_rayMaxDist) ? 1.0f : 0.0f;
    
    // do diffuse lighting
    float dp = clamp(dot(hitInfo.normal, c_lightDir), 0.0f, 1.0f);
	return c_lightAmbient * hitInfo.diffuse + dp * hitInfo.diffuse * c_lightColor * shadowTerm;
}

// Function 91
vec3 getStoneColor(vec3 p, float c, vec3 l, vec3 n, vec3 e) {
    c = min(c + pow(noise_3(vec3(p.x*20.0,0.0,p.z*20.0)),70.0) * 8.0, 1.0);
    float ic = pow(1.0-c,0.5);
    vec3 base = vec3(0.42,0.3,0.2) * 0.6;
    vec3 sand = vec3(0.51,0.41,0.32);
    vec3 color = mix(base,sand,c);
        
    float f = pow(1.0 - max(dot(n,-e),0.0), 1.5) * 0.75 * ic;
    color = mix(color,vec3(1.0),f);    
    color += vec3(diffuse(n,l,0.5) * WHITE);
    color += vec3(specular(n,l,e,8.0) * WHITE * 1.5 * ic);
    n = normalize(n - normalize(p) * 0.4);    
    color += vec3(specular(n,l,e,80.0) * WHITE * 1.5 * ic);    
    return color;
}

// Function 92
float diskColorb(in vec2 uv, vec2 offset)
{
    uv = uv - smoothstep(0.01,1.8,texture(iChannel0, (uv*1.0 - vec2((iTime-0.06) /3.0,(iTime-0.06) /8.0)) + offset).r) * 0.3;
    
    float d = length(uv)-RADIUS;
    return smoothstep(0.01,0.015,d);
}

// Function 93
vec4 getSunColor( in vec3 dir, inout float inside ) {
    float dotp = dot(dir, vec3(-0.99, 0.0, 0.1));
    float sunHeight = smoothstep(0.01, 0.29, dir.z);
    inside = smoothstep(0.977, 0.979, dotp);
    float ytemp = abs(dir.y)*dir.y;
    float sunWave = sin(dir.z*300.0+iTime*1.846+
                        sin(ytemp*190.0+iTime*0.45)*1.3)*0.5+0.5;
   	float sunHeight2 = smoothstep(-0.1, 0.2, dir.z);
    sunWave = sunWave * sunHeight2 + 1.0 - sunHeight2;
    sunWave = (1.0-smoothstep(sunHeight2, 1.0, sunWave)) * (1.0 - sunHeight2) + sunHeight2;
    float sun = inside * sunWave;
    return vec4(mix(vec3(0.998, 0.108, 0.47), vec3(0.988, 0.769, 0.176), sunHeight), sun);
}

// Function 94
vec3 sceneColor(vec2 uv, float worldTime) {
    // Thanks to BigWIngs for their mouse input stuff.
    vec2 mouse = iMouse.x > 20.0 && iMouse.y > 20.0
        ? iMouse.xy / iResolution.xy
        : vec2(0.40, 0.50);

    vec3 initialOrigin = vec3(0.0, 3.0, -3.0);
    initialOrigin.yz *= rotate(-mouse.y * 3.14 + 1.0);
    initialOrigin.xz *= rotate(-mouse.x * 6.2831);

    vec3 initialDirection = cameraRay(uv, initialOrigin, vec3(0.0, 0.55, 0.6), 1.45);
    
    vec3 origin = initialOrigin;
    vec3 direction = initialDirection;
    
    float bounce;
    vec3 color = vec3(1.0);
    TraceResult result;
    for (bounce = 0.0; bounce < maximumBounces; bounce += 1.0) {
        TraceResult result = sceneTrace(origin, direction, worldTime);
        
        if (!result.hit) {
            color = color * skyColor(result.incoming) * 2.0;
            break;
        }
        
        vec3 normal = sceneNormal(result.point, worldTime);
        
        // Apply AO to our first hit.
        if (bounce == 0.0) {
            color = color * sceneOcclusion(result.point, normal, worldTime);
        }

        color = color * 0.5;
        
        direction = reflect(direction, normal);
        origin = result.point + direction * 0.001;
    }
    
    if (bounce == maximumBounces) {
        return color * skyColor(direction);
    }

    return color;
}

// Function 95
vec3 getColor(vec2 pos) {
    pos.xy /= iResolution.xy;
    vec4 c = texture(iChannel0, pos);
    if (c.w == 0.0) {
        return texture(iChannel1, pos).xyz;
    } else {
        return c.xyz;
    }
}

// Function 96
vec4 GetCloudColor(vec3 position)
{
    float cloudDensity = GetCloudDenity(position);
    vec3 cloudAlbedo = vec3(1, 1, 1);
    float cloudAbsorption = 0.6;
    float marchSize = 0.25;

    vec3 lightFactor = vec3(1, 1, 1);
    {
        vec3 marchPosition = position;
        int selfShadowSteps = 4;
        for(int i = 0; i < selfShadowSteps; i++)
        {
            marchPosition += GetSunLightDirection() * marchSize;
            float density = cloudAbsorption * GetCloudDenity(marchPosition);
            lightFactor *= BeerLambert(vec3(density, density, density), marchSize);
        }
    }

    return vec4(
        cloudAlbedo * 
        	(mix(GetAmbientShadowColor(), 1.3 * GetSunLightColor(), lightFactor) +
             GetAmbientSkyColor()), 
        min(cloudDensity, 1.0));
}

// Function 97
vec3 colorize(const vec3 col, const float spec, const vec3 n, const vec3 dir, const in vec3 lightPos)
{
	float diffuse = 0.2*max(0.0, dot(n, lightPos));
	vec3 ref = normalize(reflect(dir, n));
	float specular = spec*pow(max(0.0, dot(ref, lightPos)), 3.5);
	return (col + diffuse * vec3(0.9) +	specular * vec3(1.0));
}

// Function 98
vec3 getSunColor(in vec3 p, in float time) {
    return vec3(1,.8,.4);
}

// Function 99
vec3 color(vec3 z, float t) {
    float coh = 0.0;
    bool fl = false;
    
    float r2;
    for(int i=0;i<40;i++) {
        
        octant1(z, coh, fl);
        z -= vec3(s,s,0);
        r2 = dot(z,z);
        if (r2 < s * s) {
            z *= s * s / r2;
            fl = !fl;
            coh = 1.0-coh;
        }
        z += vec3(s,s,0);
        
        
        octant1(z, coh, fl);
        
        r2 = dot(z,z);
        if (r2 < s*s) {
            z *= s * s / r2;
        }
        z.y -= s + 1.0;
        if (dot(z,z) < 1.0) {
            z /= dot(z,z);
        }
        z.y += s + 1.0;
        
        z.x -= s + 1.0;
        if (dot(z,z) < 1.0) {
            z /= dot(z,z);
        }
        z.x += s + 1.0;
        
    }
    octant1(z, coh, fl);
    if (fl) {coh = -coh;}
    coh -= t * 3.0;
    coh = coh / (1.5 + abs(coh));
    return vec3(0.5 + coh * 0.45);
}

// Function 100
vec3 w_color()
{
	return mix( vec3(1,1,1), vec3(0.2,0.2,0.2), selector(1.5,2.0) );
}

// Function 101
vec3 GetBaseSkyColor(vec3 rayDirection)
{
	return mix(
        vec3(0.2, 0.5, 0.8),
        vec3(0.7, 0.75, 0.9),
         max(rayDirection.y, 0.0));
}

// Function 102
vec3 getLampColor(vec2 uv)
{
    return mix(texture(iChannel0, uv).rgb, vec3(0.8), smoothstep(texture(iChannel0, uv).a, 0.07, 1.0));
}

// Function 103
vec3 getBrickColor_s(vec3 pos)
{
    return vec3(0.9, 0.3, 0.05);
}

// Function 104
float get_color_block(in sampler2D s)
{
    return texelFetch(s, CTRL_COLOR_BLOCK, 0).w;
}

// Function 105
vec3 color(vec3 p) {
    if (rdot(p,p) > -0.0) {
        // floating point error, I think, is causing invalid points
        return vec3(1.0,0.0,1.0);
    }
    // mirrors basically pulled from the Python code I wrote ages ago
    vec3 mirror1 = vec3(0.0,1.0,0.0);
    vec3 mirror2 = vec3(0.0,-sqrt(0.5),sqrt(0.5));
    vec3 mirror3 = vec3(0.5558929702514214, 0.0, -1.1441228056353687);
    
    int[5] s; // short for sigma, used in math to represent permutations
    for (int j=0;j<5;j++) {
        s[j]=j;
    }
    int t;  // used as temp space by swap
    
    int i;
    for (i=0;i<100;i++) {
        bool flipped = false;
        if (rdot(p,mirror1) > 0.0001) {
            p = refl(p,mirror1);
            swap(s[2],s[4]);
            continue;
        }
        if (rdot(p,mirror2) > 0.0001) {
            p = refl(p,mirror2);
            swap(s[1],s[4]);
            swap(s[2],s[3]);
            continue;
        }
        if (rdot(p,mirror3) > 0.0001) {
            p = refl(p,mirror3);
            swap(s[0],s[1]);
            swap(s[2],s[4]);
            continue;
        }
        if (s[0]==0) {return vec3(1.0,0.0,0.0);}
        if (s[0]==1) {return vec3(1.0,1.0,0.0);}
        if (s[0]==2) {return vec3(0.0,1.0,0.0);}
        if (s[0]==3) {return vec3(1.0,0.5,0.0);}
        if (s[0]==4) {return vec3(0.0,0.0,1.0);}
    }
    return vec3(1.0,1.0,1.0);
}

// Function 106
vec4 clampColor(vec4 col)
{
    return vec4(min(max(0.0,col.x),1.0), min(max(0.0,col.y),1.0), min(max(0.0,col.z),1.0), min(max(0.0,col.w),1.0));
}

// Function 107
vec4 glyph_color(uint glyph, ivec2 pixel)
{
    uint x = glyph & 7u,
         y = glyph >> 3u;
    pixel = ivec2(ADDR2_RANGE_FONT.xy) + (ivec2(x, y) << 3) + (pixel & 7);
    return texelFetch(LIGHTMAP_CHANNEL, pixel, 0);
}

// Function 108
vec4 mainColor(vec2 p){
   
  //Initial point
  float x = 1.0*sin(0.3*iTime);
  
  float res = 1.0;
  
  //First evaluation x_{1} = f(x_0)
  vec2 current = vec2(x,fun(x));
  res=min(res,drawLine(p,vec2(x,0.0),current));
  vec2 next;
  
  //Iteration prrocess x_{n+1} = f(x_n)
  for(int i=0;i<NUM_ITER;i++){
  	
  	next = vec2(current.y,current.y);  
   	res=min(res,drawLine(p,current,next));
    current = next;
    
    //Main evaluation
    next = vec2(current.x,fun(current.x));
    
    res = min(res,drawLine(p,current,next));
    current = next;
  }


  res = min(res,smoothstep(0.0,2.0*WIDTH,abs(p.y-fun(p.x))));
  res = min(res,smoothstep(0.0,0.5*WIDTH,abs(p.y-p.x)));
  
  //Circle with the fixed point coded in its color
  //In case of non convergence, the color is still showed
  //with the information of the last iteration point
  float cir = 1.0-drawCircle(p-vec2(x,0.0),0.01);
  vec4 colorCir = vec4(getOriginalCoord(current),0.0,1.0);
  return mix(res*WHITECOL,cir*colorCir,cir);
  
}

// Function 109
vec3 jelly_color(vec3 pos)
{
    vec2 hmp = getJellyMPos(pos);
    return getJellyColor(hmp);
}

// Function 110
vec3 GetColorFromSample( vec2 inPixel, inout vec2 blockSize)
{
	vec3 color;
    float distToMouse = length(inPixel - iMouse.xy);
    // This gets the relative position within a block. Note that this may change 
    // blockSize if we are close to the border.
    vec2 pixelInBlock = PixelInGrid(inPixel-0.5, blockSize);
    
    // Test whether this pixel is on a grid boundary.
	// step() returns 0.0 if the second parameter is less than the first, 1.0 otherwise
	// so we get 1.0 if we are on a grid line, 0.0 otherwise. (Malin's comment.)
    //vec2 gridLineTest = step(pixelInBlock, vec2(1.0)); 
    vec2 gridLineTest = vec2(0);
    // Lower Left corner pixel of block
    vec2 blockPixelCoord = (inPixel + 0.5 - pixelInBlock); 

    vec2 normCoord = blockPixelCoord/iResolution.xy; 
    // LL corner of next block up
    vec2 normCoordUp = (blockPixelCoord + vec2(0,blockSize.y))/iResolution.xy;
    // LL corner of next block to right
    vec2 normCoordRt = (blockPixelCoord + vec2(blockSize.x,0))/iResolution.xy;
    // LL corner of block diagonally up and right
    vec2 normCoordUpRt = (blockPixelCoord + blockSize)/iResolution.xy; 

    // pixel sampled from Buffer A
    vec4 fovealColor = vec4(texture(iChannel0, (inPixel+.5)/iResolution.xy));
    color = fovealColor.rgb;
        
    if (distToMouse < fovealRadius)
        color = fovealColor.rgb;
    else
    	{
        if (gridLineTest == vec2(1.0))
        	color = fovealColor.rgb;
        else
           {
        	
            // this renders a blended pixel according to its distance from four sampled points.
        	vec3 color1 = (texture(iChannel0,normCoord)).rgb;
            vec3 color2 = (texture(iChannel0,normCoordUp)).rgb;
            vec3 color3 = (texture(iChannel0,normCoordRt)).rgb;
            vec3 color4 = vec4(texture(iChannel0,normCoordUpRt)).rgb;
            
            // this tree deals with the situation of pixels being outside the ring
            // of resolution and therefore uncalculated. Since uncalculated pixels
            // are black, this is a good way to test for them. (If they were calculated
            // as black, it won't matter in a low-res area anyway.)
            if (color1 != vec3(0))
                {
        		if (color2 == vec3(0)) color2 = color1;
        		if (color3 == vec3(0)) color3 = color1;
            	if (color4 == vec3(0)) color4 = color1;
                }          
            else
            	{
                if (color2 != vec3(0))
                	{
                    color1 = color2;
                    if (color3 == vec3(0)) color3 = color2;
            		if (color4 == vec3(0)) color4 = color2;
                    }
                else
                	{
                    if (color3 != vec3(0))
                    	{
                        color1 = color3;
                    	color2 = color3;
                		if (color4 == vec3(0)) color4 = color3;
                        }
                    else
                    	{
                        if (color4 == vec3(0)) 
                        	color4 = vec3(.5,.5,.5);
                        color1 = color4;
                        color2 = color4;
                    	color3 = color4;
                         }
                    }        
                }
#if BLUR == 1               
            color = mix(
       			mix(color1, color2, (pixelInBlock.y)/blockSize.y),
        		mix(color3, color4, (pixelInBlock.y)/blockSize.y),
        		pixelInBlock.x/blockSize.x);
#else
            color = color1;
           
#endif

            }
        }
		
    return color;
    //if (blockSize != origBlockSize) fragColor = vec4(1.,0,0,1.);
        	
}

// Function 111
vec3 getDColor()
{
    return texture(iChannel2, (addr_color + vec2(0.5, 0.5))/iResolution.xy).rgb;
}

// Function 112
void brightnessAdjust( inout vec3 color, in float b) {
    color += b;
}

// Function 113
vec3 getColor(
    const float dist,
    const float angle,
    const float size
) { 
	return vec3(
        pow(dist / size, WALL_THINNESS)
        * smoothstep(size, size - PARTICLE_EDGE_SMOOTHING, dist)        
    );
}

// Function 114
vec3 getWaterAbsColor(float dist)
{
    return pow(waterColor, vec3(1.5 + pow(dist, 2.5)*10.));
}

// Function 115
vec4 colorAndDepth(vec3 pos, vec3 dir){
    vec3 diffuseColor, emissionColor, n;
    if(!trace(pos, dir, n, diffuseColor, emissionColor))
        return vec4(background(dir), RenderDistance);
    return vec4(emissionColor + diffuseColor * (directLight(pos, n)+ambientLight(pos)), length(CamPos - pos));
}

// Function 116
vec3 metalrings_color(vec3 ray)
{
    return vec3(0.6, 0.4, 0.2);
}

// Function 117
vec3 getColor(
    float dist,
    const float angle,
    float size
) { 
    dist = dist + (sin(angle * 3. + iTime * 1.) + 1.) * .04;
	return vec3(
        pow(dist / size, WALL_THINNESS)
        * smoothstep(size, size - PARTICLE_EDGE_SMOOTHING, dist)        
    );
}

// Function 118
vec3 slime_color(vec3 pos)
{
    return vec3(0.95, 0.9, 0.75);
}

// Function 119
vec3 skyColor(vec3 direction) {
    return texture(iChannel0, direction).xyz;
}

// Function 120
float evaluateColor(in float aRow, in vec2 fragCoord, in float aCycle) {
    float tFinalHue = 0.0;
    float iCurrentTime = iTime - aCycleDelay;
    //float iCurrentTime = iTime - (aCycle * aRow);
    float tPercentTimeUntilAllRed = iCurrentTime/aCycle;
    if (tPercentTimeUntilAllRed > (fragCoord.x/iResolution.x)) {
        tFinalHue = convertHue(RED);
        if (tPercentTimeUntilAllRed > 1.0) {
            float tPercentTimeUntilAllYellow = (iCurrentTime-aCycle*12.0)/aCycle;
            if (tPercentTimeUntilAllYellow > (fragCoord.x/iResolution.x)) {
                tFinalHue = convertHue(YELLOW);
                float tPercentageTimeUntilAllGreen = (iCurrentTime-aCycle*2.0*12.0)/aCycle;
                if (tPercentageTimeUntilAllGreen > (fragCoord.x/iResolution.x)) { 
                    tFinalHue = convertHue(GREEN);
                }
            }
        }
    } else {
        tFinalHue = convertHue(BLUE);
    }
    return tFinalHue;
}

// Function 121
vec3 color_spline(float t, bool wrap)
{
    t = clamp(t, 0.0, 1.0);
    
    const int s = 7;
    
    vec3 p[s];
    p[0] = vec3(238, 64, 53) / 255.0;
    p[1] = vec3(243, 119, 54) / 255.0;
    p[2] = vec3(253, 244, 152) / 255.0;
    p[3] = vec3(123, 192, 67) / 255.0;
    p[4] = vec3(3, 146, 207) / 255.0;
    
    p[s-2] = p[0];
    p[s-1] = p[1];
    
    float m = wrap ? float(s - 2) : float(s - 3);
    float d = m * t;
    
    int b = int(d);
    float dt = d - floor(d);
    
    return catmul_rom(p[((b-1)+s)%s], p[b], p[b+1], p[(b+2)%s], dt);
}

// Function 122
vec3 getBaseColor()
{
	float colorPerSecond = 0.5;
	int i = int(mod(colorPerSecond * iTime, 7.));
	int j = int(mod(float(i) + 1., 7.));

	return mix(getBaseColor(i), getBaseColor(j), fract(colorPerSecond * iTime));
}

// Function 123
void contrastAdjust( inout vec4 color, in float c) {
    float t = 0.5 - c * 0.5; 
    color.rgb = color.rgb * c + t;
}

// Function 124
float evaluateColor(in float aRow, in vec2 fragCoord, in float aCycle) {
    float tFinalHue = 0.0;
    float iCurrentTime = iTime - aCycleDelay;
    //float iCurrentTime = iTime - (aCycle * aRow);
    float tPercentTimeUntilAllRed = iCurrentTime/aCycle;
    if (tPercentTimeUntilAllRed > (fragCoord.x/iResolution.x) + sin(iTime * 5.0)*.1 + sin(iTime * 3.0)*.1 ) {
        tFinalHue = convertHue(RED) + sin(iTime*1.1)*.075;
        if (tPercentTimeUntilAllRed > 1.0) {
            float tPercentTimeUntilAllYellow = (iCurrentTime-aCycle*12.0)/aCycle;
            if (tPercentTimeUntilAllYellow > (fragCoord.x/iResolution.x)  + sin(iTime * 5.0)*.1 + sin(iTime * 3.0)*.1 ) {
                tFinalHue = convertHue(YELLOW) + abs(sin(iTime*0.9)*.05)*-1.0;
                float tPercentageTimeUntilAllGreen = (iCurrentTime-aCycle*2.0*12.0)/aCycle;
                if (tPercentageTimeUntilAllGreen > (fragCoord.x/iResolution.x)  + sin(iTime * 5.0)*.1 + sin(iTime * 3.0)*.1 ) { 
                    tFinalHue = convertHue(GREEN);
                }
            }
        }
    } else {
        tFinalHue = convertHue(BLUE) + abs(sin(iTime * .6)*.075);
    }
    return tFinalHue;
}

// Function 125
vec4 colored(vec4 base){
    vec3 col = vec3(baseColor) * base.r * 1.4;
    return vec4(col, 1.);
}

// Function 126
vec3 Env_GetFogColor(const in vec3 vDir)
{    
	return vec3(0.2, 0.5, 0.6) * 2.0;		
}

// Function 127
vec4 colorMix(vec4 color1, vec4 color2, float cursor) {
    return vec4(
    	mix(color1.x, color2.x, cursor),
    	mix(color1.y, color2.y, cursor),
    	mix(color1.z, color2.z, cursor),
        1.0
    );
}

// Function 128
void mixColorLine(vec2 uv,inout vec3 col,vec2 lineA,vec2 lineB,float scale)
{
    col = mix(
        col , 
        vec3(0.0),//hash3point(lineA+lineB) ,
        1.0 - smoothstep(0.0,1.0,sqrt(sqrt( segment(uv,lineA,lineB).x * scale )))
    );
}

// Function 129
vec3 bulb_color(vec3 ray)
{
    return vec3(0.8, 0.6, 0.3);
}

// Function 130
vec3 floor_color()
{
    return getHexagonColor(hex, 3.);
}

// Function 131
vec3 getSkyColorForPhase(vec3 ro, vec3 rd, float starsQuality, float solarPhase)
{
    float daynight = solarPhase;//sin(TIME)>=0.0?1.0:0.0;//max(0.0, dot(vec3(0.0, 1.0, 0.0), -_light.d));
    vec3 betaR = mix(
        			vec3(0.1e-6, 0.2e-6, 0.5e-6),
        			vec3(5.5e-6, 13.0e-6, 22.4e-6),// DAY
        			daynight);
    vec3 betaM = mix(
        			vec3(1e-7),
        			vec3(21e-6),
        daynight);
    vec3 color = vec3(0.0);
    vec3 miecolor = vec3(0.0);
    
    
    vec3 Lo = mix(_moon.o, _light.o, daynight);
    vec3 L = mix(_moon.d, _light.d, daynight);
    float Li = mix(_moon.power, _light.power, daynight);
    
    color = getSkyLight(ro+vec3(0.0, EARTHRADIUS, 0.0), rd, -L, betaR, betaM, miecolor);//mix(COLORSKYB, COLORSKYT, clamp(rd.y+0.5, 0.0, 1.0));
    
    // stars
    vec3 Psky = getSkyCoord(ro, rd);
    float sN = gaussianNoise3D(rd*120.0);
    color += (1.0-daynight)*vec3(max(0.0, min(1.0,sN-0.95))*10.0);
  
    float sunGrad = 0.0;
    float RdotL = dot(rd, -L);
    if(rd.y>(0.033) && RdotL>0.0/* && tmin>0.0*/)
    {
        sunGrad = daynight*pow(max(0.0, pow(RdotL, 2.0)-0.35), mix(MOONINTENSITY, SUNINTENSITY, daynight));
        sunGrad += pow((pow(RdotL, mix(2056.0, 256.0, daynight))), mix(MOONINTENSITY, SUNINTENSITY, daynight));
    }
    
    float mT = mix(0.001, 1.0, daynight);
    color += miecolor*sunGrad*mT;
    
    vec3 cloudColor = vec3(0.0);
    float cloud_mask = 0.0;
    float Energy = 0.0;
    float density = 0.0;
    
    #ifdef CLOUDS
    
    getCloudColor(Psky, Psky, rd, Lo, L, Li, density, Energy, cloud_mask, cloudColor, daynight);
   	color.r = max(0.0, color.r+cloudColor.r);
    color.g = max(0.0, color.g+cloudColor.g);
    color.b = max(0.0, color.b+cloudColor.b);
    #endif
    
    #ifdef SHAFTS
    int i =0;
    for(i=0;i<5;++i)
    {
        vec3 P = ro+rd*float(i)*5.0;
        vec3 Ld = normalize(Lo-P);
    	getCloudColor(P, L, Lo, -L, density, Energy, cloud_mask, cloudColor);
        if(cloud_mask<1.0)
        {
            color+=(1.0-density)*0.01;
        }
    }
    #endif
    
    return color;
}

// Function 132
vec4 mapcolor(inout vec3 p, in vec4 res, inout vec3 normal, in vec3 rd, out vec3 refpos, out vec3 refdir, out vec4 lparams)
{
    vec4 color = vec4(0.498, 0.584, 0.619, 1.0); lparams = vec4(1.0, 10., 0., 0.);
    refdir = reflect(rd, normal); refpos = p;

    if(res.y < 1.1) { // PortalA
        color = mapPortalColor(p, portalA.pos, portalA.rotateY, vec4(1., 1., 1., 0.1), vec4(0.0, 0.35, 1.0, 1.));
        calculatePosRayDirFromPortals(portalA, portalB, p, rd, refpos, refdir);
    }
    else if(res.y < 2.1) { // PortalB
        color = mapPortalColor(p, portalB.pos, portalB.rotateY, vec4(0.0, 1., 1.0, 0.1), vec4(0.91, 0.46, 0.07, 1.));
        calculatePosRayDirFromPortals(portalB, portalA, p, rd, refpos, refdir);
    }
#if APPLY_COLORS == 1
    else if(res.y < 3.1) { // Water
        color = vec4(0.254, 0.239, 0.007, 1.0); lparams.xy = vec2(2.0, 50.);
        color.rgb = mix(color.rgb, vec3(0.254, 0.023, 0.007), 1.-smoothstep(0.2, 1., fbm((p.xz+vec2(cos(t+p.x*2.)*0.2, cos(t+p.y*2.)*0.2))*0.5)));
        color.rgb = mix(color.rgb, vec3(0.007, 0.254, 0.058), smoothstep(0.5, 1., fbm((p.xz*0.4+vec2(cos(t+p.x*2.)*0.2, cos(t+p.y*2.)*0.2))*0.5)));
    }
    else if(res.y < 4.1) { // Turbina
        color = vec4(0.447, 0.490, 0.513, 1.0);
    }
    else if(res.y < 5.1) { //Window
        color = vec4(0.662, 0.847, 0.898, 0.6); lparams=vec4(3., 5., 0., 0.9);
    }
    else if(res.y < 6.1) { // Metal tube
        color = vec4(0.431, 0.482, 0.650, 0.6); lparams.xy=vec2(2., 5.);
    }
    else if(res.y < 7.1) {// Plastic
        color = vec4(0.8, 0.8, 0.8, 1.); lparams.xy=vec2(0.5, 1.);
    }
    else if(res.y < 8.1) { //Railing
        color = mix(vec4(1.), vec4(1., 1., 1., 0.), smoothstep(0.2, 0.21, fract(p.x)));
        color = mix(vec4(1.), color, smoothstep(0.2, 0.21, fract(p.z)));
        lparams.xy=vec2(1.0, 1.); refdir = rd;
    }
    else if(res.y < 9.1) { // Reflectance -> can be plastic
        color = vec4(1., 1., 1., 0.1); lparams.xy=vec2(1.0, 10.);
    }
    else if(res.y < 10.1) { // Exit
        vec3 q = p - vec3(1.5, 11.0, -31.);
        color = vec4(0.6, 0.6, 0.6, 0.65);
        color.rgb = mix(vec3(0.749, 0.898, 0.909), color.rgb, smoothstep(2., 10., length(q.xy)));        
        color.rgb += mix(vec3(0.1), vec3(0.), smoothstep(2., 5., length(q.xy)));

        vec3 q2 = q;
        vec2 c = vec2(2., 1.5);
        float velsign = mix(-1., 1., step(0.5, fract(q2.y*0.5)));
        q2.x = mod(velsign*t+q2.x+cos(q2.y*3.)*0.5, 1.8);
        q2.y = mod(q2.y, 1.15);
		float d = max(abs(q2.x)-0.9, abs(q2.y)-0.1);
        color.rgb += mix(vec3(0.286, 0.941, 0.992)*1.6, vec3(0.), smoothstep(-0.1, 0.1, d));
        
        vec3 localp = p - vec3(1.5, 11.0, -31.);
        refpos = vec3(1.5, 11.0, 28.0) + localp;
        lparams=vec4(1.0, 10., 0., 0.1); refdir = rd;
    }
    else if(res.y < 11.1) { // Exit border
        vec3 q = p; q.z = abs(q.z); q = q - vec3(0.0, 9.5, 31.);
        color = vec4(0.8, 0.8, 0.8, 1.);
        float d =length(abs(q.x+cos(q.y*0.5)*0.6 -3.0))-0.06;
        d = min(d, length(abs(q.x+cos(PI+q.y*0.5)*0.6 +3.0))-0.06);        
        color.rgb = mix(vec3(0.286, 0.941, 0.992), color.rgb, smoothstep(0., 0.01, d));
        lparams = mix(vec4(0., 0., 0., 1.), lparams, smoothstep(0., 0.2, d));
    }
    else if(res.y < 12.1) { // Fireball base
        vec3 q = p - vec3(10., 9.5, 26.5);
        color = vec4(1.0, 1.0, 1.0, 1.);
        float d = length(q-vec3(0., 0., -2.5)) - 2.0;
        color = mix(vec4(0.976, 0.423, 0.262, 1.), color, smoothstep(-2., 0.01, d));
    }
    else if(res.y < 13.1) { // Fireball
        color = vec4(1., 0.0, 0.0, 1.0);
        color.rgb = mix(color.rgb, vec3(0.75, 0.94, 0.28), smoothstep(26.5, 27.0, t));
    }
    
    else if(res.y > 19. && res.y < 25.) { // Walls
        
        float rand = fbm(point2plane(p, normal));
        vec3 col = vec3(0.498, 0.584, 0.619);
        color = vec4(vec3(col), 1.0);
        color = mix(color, vec4(col*0.75, 1.0), smoothstep(0.2, 1.0, rand));
        color = mix(color, vec4(col*0.80, 1.0), smoothstep(0.4, 1.0, fbm(point2plane(p*1.5, normal))));
        color = mix(color, vec4(col*0.7, 1.0), smoothstep(0.6, 1.0, fbm(point2plane(p*4.5, normal))));
        
        vec3 dirtcolor = mix(vec3(0., 0., 0.), vec3(0.403, 0.380, 0.274)*0.2, rand);
        float dirtheight = 0.1+rand*1.0;
        dirtcolor = mix(dirtcolor, vec3(0.243, 0.223, 0.137), smoothstep(dirtheight, dirtheight + 0.5, p.y));
        dirtheight = rand*2.;
        color.rgb = mix(dirtcolor, color.rgb, smoothstep(dirtheight, dirtheight+2.0, p.y));
        
        vec4 noise = mix(vec4(0.), texture(iChannel0, point2plane(p*0.037, normal)) * 0.2, smoothstep(0.2, 1., rand));
        normal = normalize(normal + vec3(noise.x, 0., noise.z));
        refdir = normalize(reflect(rd, normal));
        
        if(res.y < 20.1) { // BROWN_WALL_BLOCK
            float d = -(p.x-6.1);
            d = max(d, p.y-12.6); d = min(d, p.y-6.5);
            color *= mix(vec4(1.), vec4(0.227, 0.137, 0.011, 1.0), smoothstep(0.0, 0.1, d));
        }
        else if(res.y < 21.1) { // WHITE_PLATFORM_BLOCK
            color *= vec4(0.529, 0.572, 0.709, 1.0);
            vec3 q = p - vec3(11.5, 6.85, 7.0);
            float d = abs(q.y)-0.05;
            color.rgb = mix(vec3(0.945, 0.631, 0.015), color.rgb, smoothstep(0., 0.01, d));
            lparams.w = mix(1., 0., smoothstep(0., 0.2, d));
        }
        else if(res.y < 22.1) { // TRANSPARENT_PLATFORM_BLOCK
            color *= vec4(0.431, 0.482, 0.650, 0.1);
            refdir = rd; lparams.xy=vec2(2., 5.);
        }
        else if(res.y < 23.1) { // CEILING_BLOCK
            color *= mix(vec4(0.227, 0.137, 0.011, 1.0), vec4(1.), smoothstep(0., 0.01, p.z+6.));
        }
    }
#endif    
    return color;
}

// Function 133
vec2 mapColor(in vec3 p0, sampler2D channel) {
    float d = sdGround(p0, channel);
    float dPath = sdPath(p0);
    return min2(vec2(sdShip((p0-gRO)*gbaseShip), ID_SHIP), 
                min2(vec2(dPath, ID_PATH), vec2(d, ID_GROUND))); 
}

// Function 134
PathColor ColorScale( PathColor a, PathColor b )
{
#if SPECTRAL    
    return PathColor( a.fIntensity * b.fIntensity );
#endif    
#if RGB
    return PathColor( a.vRGB * b.vRGB );
#endif    
}

// Function 135
PathColor PathColor_Zero()
{
#if SPECTRAL    
    return PathColor( 0.0 );
#endif    
    
#if RGB
	return PathColor( vec3(0) );
#endif    
    
}

// Function 136
vec3 PoolColor(vec3 pos) {		
	if ((pos.y > HeightPool) || (pos.x > HalfSizePool) || (pos.z > HalfSizePool)) 
		return vec3(0.0);
	float tileSize = 0.2;
	float thickness = 0.015;
	vec3 thick = mod(pos, tileSize);
	if ((thick.x > 0.) && (thick.x < thickness) || (thick.y > 0.) && (thick.y < thickness) || (thick.z > 0.) && (thick.z < thickness))
		return vec3(1);
	return vec3(sin(floor((pos.x + 1.) / tileSize)) * cos(floor((pos.y + 1.) / tileSize)) * sin(floor((pos.z + 1.) / tileSize)) + 3.);
}

// Function 137
vec3 wallColor(vec3 p, vec3 norm) {
    vec2 uv = p.xz;
    uv.y -= 3.0;
    uv *= 0.5;
    
    uv /= 1.0 + tempo*0.4;
    
    float afxOuter = smoothstep(0.045, 0.035, afx(uv));
    float afxInner = smoothstep(0.009, -0.009, afx(uv));
    
    vec3 diffuse = vec3(0);
    vec3 surfaceLight = vec3(0);
    bars(p.x, diffuse, surfaceLight);
    
    const vec3 afxcol = vec3(191, 214, 48)/255.0;
    
    diffuse = mix(diffuse, vec3(0.05), afxOuter);
    diffuse = mix(diffuse, afxcol, afxInner);
    
    surfaceLight *= 1.0 - afxOuter;
    surfaceLight += afxInner * afxcol*40.0;
    
    // modify the normal
    vec2 grad = emboss(uv);
    norm.xz -= grad*2.0;
    norm = normalize(norm);
    
    diffuse = light(p, norm, diffuse);
    
    return diffuse + surfaceLight;
}

// Function 138
vec3 foliage_color(vec3 p)
{
    p /= 1.0;
    float color1 = length(sin(p/100.0))/2.0;
    return vec3(color1,color1,0.0);
}

// Function 139
vec3 getColor(vec3 norm, vec3 pos, int objnr, vec3 ray)
{
   if (objnr==BACKGROUND_OBJ)
   {
      vec3 col = background_color;
      
      col.r+= 0.4*smoothstep(0.92, 1.0, sin(0.008*pos.x*gen_scale + 1.1)*sin(0.008*pos.z*gen_scale - 3.45));
      col.g+= 0.5*smoothstep(0.85, 1.0, sin(0.008*pos.x*gen_scale + 1.1)*sin(0.008*pos.z*gen_scale - 3.45));
      col.b+= 0.6*smoothstep(0.75, 1.0, sin(0.008*pos.x*gen_scale + 1.1)*sin(0.008*pos.z*gen_scale - 3.45));
    
      return col;
   }
   else if (objnr==HEXAGONS_OBJ)
   {
      vec4 hex = hexagon(gen_scale*pos.xz);
      return getHexagonColor(hex, gen_scale*pos);
   }
   else
      return getSkyColor(ray);
}

// Function 140
vec3 colorBurn(in vec3 src, in vec3 dst)
{
    return mix(step(0.0, src) * (1.0 - min(vec3(1.0), (1.0 - dst) / src)),
        vec3(1.0), step(1.0, dst));
}

// Function 141
mat3 calc_sat_adjust_matrix( float sat, vec3 rgb2Y) {
mat3 M;
M[0][0] = (1.0 - sat) * rgb2Y.x + sat;
M[1][0] = (1.0 - sat) * rgb2Y.x;
M[2][0] = (1.0 - sat) * rgb2Y.x;
M[0][1] = (1.0 - sat) * rgb2Y.y;
M[1][1] = (1.0 - sat) * rgb2Y.y + sat;
M[2][1] = (1.0 - sat) * rgb2Y.y;
M[0][2] = (1.0 - sat) * rgb2Y.z;
M[1][2] = (1.0 - sat) * rgb2Y.z;
M[2][2] = (1.0 - sat) * rgb2Y.z + sat;
mat3 R = mat3(vec3(M[0][0], M[0][1], M[0][2]), 
vec3(M[1][0], M[1][1], M[1][2]), vec3(M[2][0], M[2][1], M[2][2]));
R = transpose(R);    
return R;
}

// Function 142
vec3 colorize(vec3 p, vec3 d, float dist)
{
 	vec3 col = vec3(0);
    
    mat3 rotMat = rotationMatrix(vec3(1,1,0), iTime*0.2);
    vec3 boxsize = vec3(clamp(0.3-sdTorus(rotMat*stepround(p, vec3(GS)), vec2(2,0.5))*0.6, 0.0, GS) );
    vec3 relp = mix(mod(p+vec3(GS),vec3(GS*2.0))-vec3(GS), p, step(0.6, sdTorus(rotMat*stepround(p, vec3(GS)), vec2(2,.5))) );
    float a = smoothstep(boxsize.x*0.8, boxsize.x, abs(relp.x)) + smoothstep(boxsize.y*0.8, boxsize.y, abs(relp.y)) + smoothstep(boxsize.z*0.8, boxsize.z, abs(relp.z));
    col = vec3(smoothstep(0.3,0.8,a/3.0)) * vec3(1.0,0.4,0);
    

    vec3 n = calcNormal(p, rotMat);
    col = mix(col, texture(iChannel1, reflect(normalize(d), n)).rgb, (1.0-length(col))*0.2);
    //col = n;
    
    return mix(col, texture(iChannel0, d).rgb, step(10.0, dist));
}

// Function 143
vec4 Scene_GetColorAndDepth( vec3 vRayOrigin, vec3 vRayDir )
{
	vec3 vResultColor = vec3(0.0);
            
	SceneResult firstTraceResult;
    
    float fStartDist = 0.0f;
    float fMaxDist = kMaxTraceDist;
    
    vec3 vRemaining = vec3(1.0);
    
	for( int iPassIndex=0; iPassIndex < 2; iPassIndex++ )
    {
    	SceneResult traceResult = Scene_Trace( vRayOrigin, vRayDir, fStartDist, fMaxDist );

        if ( iPassIndex == 0 )
        {
            firstTraceResult = traceResult;
        }
        
        vec3 vColor = vec3(0);
        vec3 vReflectAmount = vec3(0);
        
		if( traceResult.iObjectId < 0 )
		{
            bool bDrawSun = (iPassIndex == 0);
            vColor = Env_GetSkyColor( vRayOrigin, vRayDir, bDrawSun ).rgb;
            float fDist = abs(length(vRayDir.xz) * 20.0 / vRayDir.y);
			vColor = Env_ApplyAtmosphere( vColor, vRayOrigin, vRayDir, fDist );
        }
        else
        {
            
            SurfaceInfo surfaceInfo = Scene_GetSurfaceInfo( vRayOrigin, vRayDir, traceResult );
            SurfaceLighting surfaceLighting = Scene_GetSurfaceLighting( vRayDir, surfaceInfo );
                
            // calculate reflectance (Fresnel)
			vReflectAmount = Light_GetFresnel( -vRayDir, surfaceInfo.vBumpNormal, surfaceInfo.vR0, surfaceInfo.fGloss );
			
			vColor = (surfaceInfo.vAlbedo * surfaceLighting.vDiffuse + surfaceInfo.vEmissive) * (vec3(1.0) - vReflectAmount); 
            
            vec3 vReflectRayOrigin = surfaceInfo.vPos;
                        
            vec3 vReflectRayDir = normalize( reflect( vRayDir, surfaceInfo.vBumpNormal ) );

            
            {
                float alpha2 = SpecParamFromGloss(surfaceInfo.fGloss);
                
                vec2 vRand = hash23( vRayOrigin + vRayDir + iTime );

                vec3 N = surfaceInfo.vBumpNormal;
                vec3 V = -vRayDir;
                vec3 H = ImportanceSampleGGX( vRand, N, alpha2 );        

				vReflectRayDir = reflect( -V, H );                
            }
			
            
            
            fStartDist = 0.001 / max(0.0000001,abs(dot( vReflectRayDir, surfaceInfo.vNormal ))); 

            vColor += surfaceLighting.vSpecular * vReflectAmount;            

			vColor = Env_ApplyAtmosphere( vColor, vRayOrigin, vRayDir, traceResult.fDist );
			vColor = FX_Apply( vColor, vRayOrigin, vRayDir, traceResult.fDist );
            
            vRayOrigin = vReflectRayOrigin;
            vRayDir = vReflectRayDir;
        }
        
        vResultColor += vColor * vRemaining;
        vRemaining *= vReflectAmount;        
    }
 
    return vec4( vResultColor, EncodeDepthAndObject( firstTraceResult.fDist, firstTraceResult.iObjectId ) );
}

// Function 144
float watercolor (vec2 p) {
       p*=8.;
       vec2 q = vec2(0.);
       q.x = fbm(p);
       q.y = fbm( p + vec2(1.0));
       vec2 r = vec2(0.);
       r.x = fbm( p + 1.0*q + vec2(1.7,9.2));
       r.y = fbm( p + 1.0*q + vec2(8.3,2.8));
       float f = fbm(p+r);
       return clamp(f,0.,1.);
}

// Function 145
vec3 GetSpaceColor(vec3 rayDir)
{ 
  vec3 stars = BoxMap(iChannel0, rayDir, rayDir, 0.5, 0.0).rgb;
  vec3 stars2 = SphereMap(iChannel0, rayDir).rgb; 

  vec3 starPos = rayDir+vec3(iTime*0.0004, iTime*0.0005, iTime*0.0003);
  float starsDetailS = pow(noise(starPos*450.), 1.);
  float starsDetailM = pow(noise(starPos*150.), 2.);
  float starsDetailL = pow(0.45+noise(starPos*52.), 3.);

  vec3 starColor = vec3(1.0);

  starColor.r += abs(starsDetailS-starsDetailM);
  starColor.g += abs(starsDetailL-starsDetailM);
  starColor.b += abs(starsDetailS-starsDetailL);

  starColor=(starColor*0.5)+(starsDetailL*.5);

  float sun = mix(0., pow( clamp( 0.5 + 0.5*dot(sunPos, rayDir), 0.0, 1.0 ), 2.0 ), smoothstep(.33, .0, rayDir.y));
  float sun2 = clamp( 0.75 + 0.25*dot(sunPos, rayDir), 0.0, 1.0 );

  vec3 col = mix(vec3(0, 0, 164)/255., vec3(0, 0, 150)/255., smoothstep(0.8, 0.00, rayDir.y)*sun2);
  col = mix(col, vec3(100, 0, 169)/255., smoothstep(0.015, .0, rayDir.y)*sun2);
  col = mix(col, vec3(160, 0, 136)/255., smoothstep(0.3, 1.0, sun));
  col = mix(col, vec3(255, 0, 103)/255., smoothstep(0.6, 1.0, sun));

  col=col*stars;
  col = mix(col, vec3(starsDetailS*starColor), smoothstep(0.7, 1., starsDetailS));
  col = mix(col, vec3(starsDetailM*starColor), smoothstep(0.7, 1., starsDetailM));

  vec3 nebula = (vec3(stars.r, 0., 0.)*stars2.r);
  nebula = mix(nebula, nebula*2., pow(stars2.r, 2.));
  nebula = mix(nebula, vec3(1.), pow(stars2.r, 4.));        

  vec3 offset = vec3(iTime, iTime*2., 0.)*0.01;
  vec2 addStep = vec2(-0.04, -0.07)*0.05;
  vec2 pp = PosToSphere(rayDir);

  #ifdef COMETS
    vec3 comet = textureLod(iChannel3, (pp)-offset.xy, 1.).rgb;

  for ( int i=0; i<30*int(step(0.2, comet.r)); i++ )
  {
    col = mix(col, vec3(1.), step(0.4, pow(textureLod(iChannel3, (pp*2.)-offset.xy, 1.).r, 6.))/((float(i)+1.)));
    offset.xy+=addStep;
  }   
  #endif

    nebula = mix(nebula, nebula*vec3(1.2, 0.9, .50), max(0., readRGB(ivec2(120, 0)).x));

  return col+nebula;
}

// Function 146
void contrastAdjust( inout vec3 color, in float c) {
    float t = 0.5 - c * 0.5; 
    color = color * c + t;
}

// Function 147
vec3 getDepthWaterColor(float D)
{
    float d = min(1.0, log(1.0+D/WATERSHADING_V4_DEPTH_FACTOR));
    
    return mix(WATERCOLOR1,
        mix(WATERCOLOR2,
        mix(WATERCOLOR3,
        mix(WATERCOLOR4, WATERCOLOR5, d)
            , d)
        	, d)
            , d);
}

// Function 148
void color_correction(inout vec4 fragColor, vec2 fragCoord, bool is_thumbnail)
{
    if (g_demo_stage != DEMO_STAGE_NORMALS)
    {
        Options options;
        LOAD(options);
    	
        float gamma = is_thumbnail ? .8 : 1. - options.brightness * .05;
#if GAMMA_MODE
    	fragColor.rgb = gamma_to_linear(fragColor.rgb);
    	float luma = dot(fragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    	if (luma > 0.)
	    	fragColor.rgb *= pow(luma, gamma) / luma;
	    fragColor.rgb = linear_to_gamma(fragColor.rgb);
#else
	    fragColor.rgb = pow(fragColor.rgb, vec3(gamma));
#endif
    }
    
    // dithering, for smooth depth/lighting visualisation (when not quantized!)
    fragColor.rgb += (BLUE_NOISE(fragCoord).rgb - .5) * (1./127.5);
}

// Function 149
vec3 getSkyColor(vec3 e) {
	float y = e.y;
	vec4 r = snoise(150.0 * e);
	y+=r.w*0.01-r.y*0.005;
	y = atan(2.0*y);
	return mix(vec3(1.0,1.0,0.7), vec3(0.5,0.7,0.4), clamp(y + 1.0, 0.0, 1.0))+
		mix(vec3(0.0), -vec3(0.5,0.7,0.4), clamp(y, 0.0, 1.0));
}

// Function 150
vec4 gen_color2(vec3 iter)
{
    float t1 = 1.0+log(iter.y)/4.0;
    float t2 = 1.0+log(iter.z)/8.0;
    float t3 = t1/t2;
    
    //vec3 comp = vec3(t1,t1,t1);
    vec3 red = vec3(0.9,0.2,0.1);
    vec3 black = vec3(1.0,1.0,1.0);
    vec3 blue = vec3(0.1,0.2,0.9);
    vec3 comp = mix(blue,black,vec3(t2));
    comp = mix(-red, comp,vec3(t1));
    
    return vec4(comp, 1.0);
}

// Function 151
vec3 getFloorColor(vec3 pos)
{
    #ifdef parquet
    pos.x+= mod(floor(pos.z/psize.y), 2.)==1.?psize.x*0.5:0.;
    
    float pnum = floor(pos.x/psize.x) + 1000.*floor(pos.z/psize.y);
    vec2 offset = vec2(hash(pnum*851.12), hash(pnum*467.54));
    vec3 cm = (0.5 + 0.5*hash(pnum*672.75))*floor_color;
    float sf = 0.8 + 0.4*hash(pnum*218.47);
    float ra = 0.15*hash(pnum*951.68);
    
    pos.xz = rotateVec(pos.xz, ra);
    return mix(cm, texture(iChannel0, 0.35*sf*pos.xz + offset).rgb, 0.65);
    #else
    return floor_color;
    #endif
}

// Function 152
vec3 getEnergyColor (in uint energy)
{
	energy = min(energy, MAX_REDSTONE_POWER);
    return getEnergyColor(float(energy) / float(MAX_REDSTONE_POWER));
}

// Function 153
vec3 calc_world_color(ray_t ray) {
	vec3 skycolor=vec3(0.4,0.6,1.0);
	vec3 color=vec3(0.0);
	float frac=1.0; /* fraction of object light that makes it to the camera */
	
	for (int bounce=0;bounce<8;bounce++) 
	{
		ray.D=normalize(ray.D);
	/* Intersect camera ray with world geometry */
		ray_hit_t rh=world_hit(ray,0.0);

		if (rh.t>=invalid_t) {
			color+=frac*skycolor; // sky color
			break; // return color; //<- crashes my ATI
		}

	/* Else do lighting */
		if (rh.s.solid>0.5) { // solid surface 
			if (dot(rh.N,ray.D)>0.01) rh.N=-rh.N; // flip normal to face right way
            
            /*
            // Phong (crude hack, 'sun' sphere works better)
			vec3 H=normalize(L+normalize(-ray.D));
            float specular=rh.s.shiny*pow(clamp(dot(H,rh.N),0.0,1.0),500.0);
            */
			float diffuse=clamp(dot(rh.N,L),0.0,1.0);

			// check shadow ray 
			ray_t shadow_ray=ray_t(rh.P,L, ray_radius(ray,rh.t),0.01);
			ray_hit_t shadow=world_hit(shadow_ray,1.0);
			if (shadow.t<invalid_t) {
				float illum=1.0-shadow.shadowfrac;
				diffuse*=illum; 
				//specular*=illum; 
			}

			float ambient=0.05;

			vec3 curObject=(ambient+diffuse)*rh.s.reflectance; // +specular*vec3(1.0);
			
			color+=frac*rh.frac*curObject;
			//color=rh.N; // debug: show surface normal at hit
        } else { // emissive object
            color+=frac*rh.frac*rh.s.reflectance;
        }
		
	/* Check for ray continuation */
		if (rh.frac<1.0) 
        { // partial hit--continue ray walk to composite background
			if (rh.s.mirror>0.0) { // uh oh, need two recursions
				// fake partial mirror using sky light
				color+=frac*rh.frac*rh.s.mirror*skycolor;
				//color+=vec3(1,0,0); 
			}
			
			frac*=(1.0-rh.frac);
			
			float t=rh.exit_t+0.1;
			ray.r_start=ray_radius(ray,t);
			ray.C=ray_at(ray,t);
		}
		else if (rh.s.mirror>0.0) { // mirror reflection
			frac*=rh.s.mirror;
			float t=rh.t;
			ray.r_start=ray_radius(ray,t);
            float curvature=10.0; // HACK: should depend on radius
            ray.r_per=curvature*ray.r_per; 
			ray.C=ray_at(ray,t);
			//color+=rh.s.mirror*calc_world_color(rh.P,reflect(D,rh.N));
			ray.D=reflect(ray.D,rh.N); // bounce off normal
		}
		else break;
		if (frac<0.005) return color;
	} 
	
	return color;
}

// Function 154
vec3 get_color(in sampler2D s)
{
    return texelFetch(s, CTRL_COLOR_ABC, 0).rgb;
}

// Function 155
vec3 uiColor(int id){return texture(iChannel0, vec2(float(id)+.5,1.5)/iResolution.xy).rgb;}

// Function 156
vec3 normalize_color(vec3 raw) {
    return 2.0 / (exp(-EXPOSURE * raw) + 1.0) - 1.0;
}

// Function 157
vec3 GetDiffuseColor(float l) {

    bvec4 dist = lessThan(
        vec4(l), 
        vec4(
            GradientColorStep1.a,
            GradientColorStep2.a,
            GradientColorStep3.a,
            GradientColorStep4.a
        ));

    if(dist.x) return GradientColorStep1.xyz;
    else if(dist.y) return mixGrad(GradientColorStep1, GradientColorStep2, l);
    else if (dist.z) return mixGrad(GradientColorStep2, GradientColorStep3, l);
    else if(dist.a) return mixGrad(GradientColorStep3, GradientColorStep4, l);
    else return GradientColorStep4.xyz;
}

// Function 158
vec3 getColor(vec2 uv){
	vec3 ray = getRay(uv);
    
    if(ray.y >= -0.01){
        vec3 C = getatm(ray, 0.0) * 1.0 + sun(ray) * 2.0;
     	return C; 
    }
    
	vec3 wfloor = vec3(0.0, -WATER_DEPTH, 0.0);
	vec3 wceil = vec3(0.0, 0.0, 0.0);
	vec3 orig = vec3(0.0, 2.0, 0.0);
	float hihit = intersectPlane(orig, ray, wceil, vec3(0.0, 1.0, 0.0));
	float lohit = intersectPlane(orig, ray, wfloor, vec3(0.0, 1.0, 0.0));
    vec3 hipos = orig + ray * hihit;
    vec3 lopos = orig + ray * lohit;
	float dist = raymarchwater(orig, hipos, lopos, WATER_DEPTH);
    vec3 pos = orig + ray * dist;

	vec3 N = normal(pos.xz, 0.01, WATER_DEPTH);
    vec2 velocity = N.xz * (1.0 - N.y);
    vec3 R = reflect(ray, N);
    float roughness = 1.0 - 1.0 / (dist * 0.01 + 1.0);
    N = normalize(mix(N, vec3(0.0, 1.0, 0.0), roughness));
    R = normalize(mix(R, N, roughness));
    R.y = abs(R.y);
    float fresnel = (0.04 + (1.0-0.04)*(pow(1.0 - max(0.0, dot(-N, ray)), 5.0)));
	
    vec3 C = fresnel * (getatm(R, roughness) + sun(R)) * 2.0;
    
	return C;
}

// Function 159
vec4 colormap_hsv2rgb(float h, float s, float v) {
	float r = v;
	float g = v;
	float b = v;
	if (s > 0.0) {
		h *= 6.0;
		int i = int(h);
		float f = h - float(i);
		if (i == 1) {
			r *= 1.0 - s * f;
			b *= 1.0 - s;
		} else if (i == 2) {
			r *= 1.0 - s;
			b *= 1.0 - s * (1.0 - f);
		} else if (i == 3) {
			r *= 1.0 - s;
			g *= 1.0 - s * f;
		} else if (i == 4) {
			r *= 1.0 - s * (1.0 - f);
			g *= 1.0 - s;
		} else if (i == 5) {
			g *= 1.0 - s;
			b *= 1.0 - s * f;
		} else {
			g *= 1.0 - s * (1.0 - f);
			b *= 1.0 - s;
		}
	}
	return vec4(r, g, b, 1.0);
}

// Function 160
void calculateColor(ray cameraRay, float sa, vec2 fragCoord, out vec3 camHitPosition, out float depth, out vec3 camHitNormal, out vec3 baseColor, out vec3 directLight, out vec3 indirectLight, out pointLight bounceLight, vec3 sunDirection, vec3 sunColor)
{
    const float epsilon = 0.0001;
    float seed = mod(iTime, 1024.0)+0.13*iMouse.x+1.25*iMouse.y;
    
    vec3 bounceColor = vec3(1);
    
    vec3 totalDirect = vec3(0);
    vec3 totalGi = vec3(0);
    
    ray currentRay = cameraRay;
    
    // TODO manually unroll bounces to reduce number of ifs?
    for(int bounce = 0; bounce<2; bounce++)
    {
        currentRay.direction = normalize(currentRay.direction);
        
       
        float traced = -1.0;
        if(bounce == 0)
        {
            traced = intersect(currentRay, 128, 0.005);
        }
        else
        {
            traced = intersect(currentRay, 80, 0.005);
        }
        if(traced < 0.0)
        {
            if( bounce==0 ) 
            {
                // No hit, draw BG
                vec3 bgColor = getSky(currentRay, sunDirection, sunColor);
                totalDirect = bgColor;

                // Out
                directLight = bgColor;
                indirectLight = vec3(0);

                return;
            }
            break;
        }

        vec3 position = currentRay.origin + currentRay.direction*traced;
        vec3 surfaceNormal = calcNormal(position);
        
        vec3 triplanarNormal = surfaceNormal;
		
        float emissiveFactor = saturate((1.0 - orbitTrap.z*50.0)*100000.0);
        
        vec3 emissiveColor = pow(((sin(position.x*5.0+mod(iTime, 1024.0)/2.0)+1.0)/2.0), 8.0)*1.33*pow(vec3(0.35,1.0,0.55),vec3(2.0))*emissiveFactor + 0.02*vec3(0.35,1.0,0.55)*emissiveFactor;

        vec3 surfaceColor1 = vec3(0.7);
        vec3 surfaceColor2 = vec3(0.6, 0.5, 0.8);

        vec3 surfaceColor = mix(surfaceColor1, surfaceColor2, saturate((orbitTrap.y*3.5-0.25)*1.0))*(1.0-emissiveFactor) + emissiveFactor*(vec3(0.5,0.8,1.0));
        
        #ifdef ROUGHNESS_MAP
        	float roughness = saturate(pow(triPlanarMapCatRom(iChannel2, 5.0, triplanarNormal, position*7.0, iChannelResolution[2].xy), vec3(2.0)).r*2.0);
		#else
        	const float roughness = 0.4;
        #endif
        
		// Direct lighting
        vec3 iColor = vec3(0.0);

        // Direct sun light
        vec3 currentSunDir = sunDirection;
        
        float sunDiffuse = 0.0;
        float sunSpec = 0.0;

        if(bounce == 0)
        {
            sunDiffuse = saturate(dot(currentSunDir, surfaceNormal))*0.9;
            sunSpec = GGX(surfaceNormal, -currentRay.direction, currentSunDir, roughness, 0.1);
        }
        else
        {
            sunDiffuse = saturate(dot(currentSunDir, surfaceNormal));
            sunSpec = 0.0;
        }
        float sunShadow = 1.0;
        if(sunDiffuse > 0.0) 
        {
            sunShadow = shadow(ray(position + surfaceNormal*epsilon, currentSunDir), 80);
        }

        iColor += sunColor*sunDiffuse*sunShadow + sunColor*sunSpec*sunShadow;
        
        // Carry surface color through next bounce
        vec3 previousBounceColor = bounceColor;
        bounceColor *= surfaceColor;

		if(bounce == 0)
        {
            totalDirect += bounceColor*iColor + emissiveColor;
            // Out
            camHitPosition = position;
            depth = traced;
            baseColor = surfaceColor;
            camHitNormal = surfaceNormal;
        }
        else if(bounce == 1)
        {
            totalGi += bounceColor*iColor + emissiveColor;

            // Virtual point light from direct lighting of first bounce, accumulated in Buffer B
            bounceLight.worldPosition = position;
            bounceLight.normal = surfaceNormal;
            bounceLight.color = (previousBounceColor*iColor + emissiveColor);

            // TODO texture map
            
            float lightDistance = distance(bounceLight.worldPosition, camHitPosition);
            float NdotL = saturate(dot(normalize(camHitNormal), normalize(bounceLight.worldPosition - camHitPosition)));
            	
            if(NdotL > 0.00001 && length(baseColor) > 0.00001)
            {	
                // Cancel out cosine distribution
                bounceLight.color /= NdotL;
                // Cancel out inverse square attenuation 
                bounceLight.color *= lightDistance*lightDistance;
                // For debugging direct light
                //bounceLight.color *= 0.0;
            }
        }

		// Send bounce ray
        vec3 reflectDirection = reflect(normalize(currentRay.direction), normalize(surfaceNormal));
        currentRay.direction = cosineDirection(surfaceNormal, fragCoord, seed);

        currentRay.origin = position;
    }
    
    // Out
	directLight = totalDirect;
    indirectLight = totalGi;
}

// Function 161
vec3 getColor(int o)
{
	vec4 Z = vec4(0.3, 0.5, 0.6, 0.2);
	vec4 Y = vec4(0.1, 0.5, 1.0, -0.5);
	vec4 X = vec4(0.7, 0.8, 1.0, 0.3);
	vec3 orbitColor = cycle(X.xyz,ot.x)*X.w*ot.x + cycle(Y.xyz,ot.y)*Y.w*ot.y + cycle(Z.xyz,ot.z)*Z.w*ot.z;
	if (orbitColor.x >= 4.) orbitColor.x =0.;
	if (orbitColor.y >= 4.) orbitColor.y =0.;
	if (orbitColor.z >= 4.) orbitColor.z =0.;
	return clamp(3.0*orbitColor,0.0,4.0);
}

// Function 162
vec3 getSphereColor( const vec2 grid ) {
	vec3 col = hash3( grid+vec2(43.12*grid.y,12.23*grid.x) );
    return mix(col,col*col,.8);
}

// Function 163
vec3 GetSceneRayColor (in vec3 rayPos, in vec3 rayDir)
{
    // Returns the lit RGB for this ray intersecting with the scene, ignoring the main object.
    // Used for reflection off the surface of the object, and refraction out the back of the object.
    
    // if we hit the box, return the lit box color
    vec2 uv;
    vec4 rayInfo = RayIntersectBox(rayPos + vec3(0.0, 1.51, 0.0), rayDir, vec3(1.0, 1.0, 1.0), uv);
    if (rayInfo.x >= 0.0)
        return LightPixel(rayPos + rayDir*rayInfo.x, rayDir, Checkerboard(uv), rayInfo.yzw, 100.0, true);
    // else return skybox color
    else
        return texture(iChannel0, rayDir).rgb;
}

// Function 164
vec3 surface_color(vec3 p)
{
    p = animate(p);
    // Normalized pixel coordinates (from 0 to 1)
    //vec2 uv = vec2(p.x,p.y)/(300.0);
    p = (sin(p) + sin(p*4.0))/2.0;
    return sin(vec3(0.0,(p.y+p.x+p.z)*2.0,0.0));
}

// Function 165
Color3 shadowedAtmosphereColor(vec2 fragCoord, vec3 iResolution, float minVal) {
    vec2 rel = 0.65 * (fragCoord.xy - iResolution.xy * 0.5) / iResolution.y;
    const float maxVal = 1.0;
    
    float a = min(1.0,
                  pow(max(0.0, 1.0 - dot(rel, rel) * 6.5), 2.4) + 
                  max(abs(rel.x - rel.y) - 0.35, 0.0) * 12.0 +                   
	              max(0.0, 0.2 + dot(rel, vec2(2.75))) + 
                  0.0
                 );
    
    float planetShadow = mix(minVal, maxVal, a);
    
    return atmosphereColor * planetShadow;

}

// Function 166
vec3 encodeColor(vec3 color){
	return color * 0.001;
}

// Function 167
vec4 colorize(float c) {
	
	float hue = mix(0.6, 1.15, min(c * 1.2 - 0.05, 1.0));
	float sat = 1.0 - pow(c, 4.0);
	float lum = c;
	vec3 hsv = vec3(hue, sat, lum);
	vec3 rgb = hsv2rgb(hsv);
	return vec4(rgb, 1.0);	
}

// Function 168
vec4 gen_color(vec3 iter)
{
    float t1 = 1.0+log(iter.y)/8.0;
    float t2 = 1.0+log(iter.z)/16.0;
    float t3 = t1/t2;
    
    //vec3 comp = vec3(t1,t1,t1);
    vec3 red = vec3(0.9,0.2,0.1);
    vec3 black = vec3(1.0,1.0,1.0);
    vec3 blue = vec3(0.1,0.2,0.9);
    vec3 comp = mix(blue,black,vec3(t2));
    comp = mix(red,comp,vec3(t1));
    
    return vec4(comp, 1.0);
}

// Function 169
void glitchColor(vec2 p, inout vec3 color) {
    vec2 groupSize = vec2(.75,.125) * glitchScale;
    vec2 subGrid = vec2(0,6);
    float speed = 5.;
    GlitchSeed seed = glitchSeed(glitchCoord(p, groupSize), speed);
    seed.prob *= .3;
    if (shouldApply(seed) == 1.) {
        vec2 co = mod(p, groupSize) / groupSize;
        co *= subGrid;
        float a = max(co.x, co.y);
        //color.rgb *= vec3(
        //  min(floor(mod(a - 0., 3.)), 1.),
        //    min(floor(mod(a - 1., 3.)), 1.),
        //    min(floor(mod(a - 2., 3.)), 1.)
        //);
        
        color *= min(floor(mod(a, 2.)), 1.) * 10.;
    }
}

// Function 170
vec3 colorFor(vec3 base, float uv) {
    vec3 first = base;
    
    vec3 second = adjustSV(base, .5, .9);
    vec3 third = adjustSV(base, .82, .56);
    vec3 fourth = adjustSV(base, .2, .96);

    //vec3 first = vec3(0.0 / 255.0, 209.0 / 255.0, 193.0 / 255.0);
    //vec3 second = vec3(110.0 / 255.0, 230.0 / 255.0, 217.0 / 255.0);
    //vec3 third = vec3(26.0 / 255.0, 143.0 / 255.0, 124.0 / 255.0);
    //vec3 fourth = vec3(193.0 / 255.0, 245.0 / 255.0, 240.0 / 255.0);
    // vec3 babu1 = vec3(0.0 / 255.0, 209.0 / 255.0, 193.0 / 255.0);
    
    float x4 = ((uv + 1.0) / 2.0) * 4.0;
    
    if (x4 <= 1.0) {
        return mix(first, second, mod(x4, 1.0));
    }
    
    if (x4 > 1.0 && x4 <= 2.0) {
        return mix(second, third, mod(x4, 1.0));
    }

    if (x4 > 2.0 && x4 <= 3.0) {
        return mix(third, fourth, mod(x4, 1.0));
    }

    //if (x4 > 3.0 && x4 <= 4.0) {
        return mix(fourth, first, mod(x4, 1.0));
    //}
}

// Function 171
vec3 getColor(vec3 norm, vec3 pos, int objnr)
{
   #ifdef lamp_is_glass
   vec3 lampcol = vec3(0);
   #else
   vec3 lampcol = lampg_color(rotateVec2(pos));
   #endif
   return objnr==LAMPG_OBJ?lampcol:(
          objnr==BULB_OBJ?bulb_color(pos):(
          objnr==METALRINGS_OBJ?metalrings_color(pos):(
          objnr==SUPPORTS_OBJ?supports_color(pos):sky_color(pos))));
}

// Function 172
vec3 GetSampleColor(vec2 uv
){RayInfo r
 ;r.dir = vec3(0,0,1)
 ;if (fishEye
 ){vec3 crossv=cross(r.dir,vec3(uv,0))
  ;r.dir=qr(aa2q(length(uv)*FOV,normalize(crossv)),r.dir)
  ;}else r.dir = vec3(uv.xy*FOV,1.)
 ;//apply look dir
 ;r.b = objPos[oCam]//es100 error , no array of class allowed
 ;r.dir = qr(objRot[oCam],r.dir)//es100 error , no array of class allowed
 ;MarchPOV(r,playerTime)
 ;return GetDiffuse(r);}

// Function 173
vec3 sky_color(vec3 ray)
{
	vec3 rc = 1.5*texture(iChannel1, ray).rgb;
    return rc;
}

// Function 174
vec3 colorRamp(float t)
{
    int idx = clamp(int(255.0f * t),0,255);
    return turbo_srgb_floats[idx];
}

// Function 175
vec3 sky_color(vec3 ray)
{
    return ambientColor + vec3(0.4, 0.3, 0.05)*2.8*(0.6-atan(ray.y)); 
}

// Function 176
PathColor ColorScale_sRGB( WaveInfo wave, vec3 sRGB )
{
#if SPECTRAL    
    
#if 1
    vec3 sRGBRanges = vec3( 40, 50, 70.0 );
	vec3 sRGBApproxWavelengths = vec3( 610.0, 549.0, 468.0 );
    float x = FFalloff( wave.wavelength, sRGBApproxWavelengths.x, sRGBRanges.x) * sRGB.x
        + FFalloff( wave.wavelength, sRGBApproxWavelengths.y, sRGBRanges.y) * sRGB.y
        + FFalloff( wave.wavelength, sRGBApproxWavelengths.z, sRGBRanges.z) * sRGB.z;
	return  PathColor( x * 1.5 ); 
#else    
    return  PathColor( dot( sRGB, wave.rgb ));
#endif    
    
#endif
    
    
#if RGB
    return PathColor( sRGB );
#endif
}

// Function 177
vec3 colorTemperatureToRGB(const in float temperature){
  mat3 m = (temperature <= 6500.0) ? mat3(vec3(0.0, -2902.1955373783176, -8257.7997278925690),
                                          vec3(0.0, 1669.5803561666639, 2575.2827530017594),
                                          vec3(1.0, 1.3302673723350029, 1.8993753891711275)) :
                                     mat3(vec3(1745.0425298314172, 1216.6168361476490, -8257.7997278925690),
                                          vec3(-2666.3474220535695, -2173.1012343082230, 2575.2827530017594),
                                          vec3(0.55995389139931482, 0.70381203140554553, 1.8993753891711275));
  return clamp(vec3(m[0] / (vec3(clamp(temperature, 1000.0, 40000.0)) + m[1]) + m[2]), vec3(0.0), vec3(1.0));
}

// Function 178
vec4 colormap(float x) {
    return vec4(colormap_red(x), colormap_green(x), colormap_blue(x), 1.0);
}

// Function 179
vec4 getColorAndRoughness(vec3 p, vec3 N, float ambo)
{
    return vec4(1.0);
}

// Function 180
vec3 computeColor(Ray ray, vec3 playerPos) {
    vec3 col = vec3(0.0);
    
    // Switch on matID
    // a return -> different/no lighting
    // no return -> default lighting
 	if (ray.matID == 0) {
    	return sky(ray.dir);
    } else if (ray.matID == 2) {
        col = player(ray, playerPos);
    } else if (ray.matID == 3) {
        col = islands(ray);
    } else if (ray.matID == 4) {
        col = house(ray);
    } else if (ray.matID == 5) {
        col = roof(ray);
    } else if (ray.matID == 6) {
        col = chimney(ray);
    } else if (ray.matID == 7) {
        col = door();
    } else if (ray.matID == 8) {
        col = doorknob(ray);
    } else if (ray.matID == 9) {
        col = wood(ray);
    } else if (ray.matID == 10) {
        col = apple(ray);
    } else if (ray.matID == 11) {
        col = bird();
    }
    
    // Default lighting
    float sunLight = directionalLightDiffuse(ray.nor, SUN_DIR);
    float sunShadow = softshadow(ray.pos, SUN_DIR, playerPos);

    col = col * (0.8 * sunLight * sunShadow + 0.1);
    
    return col;
}

// Function 181
vec3 color(int id)
{
	// flappy bird colors
	if (id == 0) return vec3(0.0);
	if (id == 1) return vec3(0.320,0.223,0.289);
	if (id == 2) return vec3(0.996,0.449,0.063);
	if (id == 3) return vec3(0.965,0.996,0.965);
	if (id == 4) return vec3(0.996,0.223,0.000);
	if (id == 5) return vec3(0.836,0.902,0.805);
	return vec3(0.965,0.707,0.191);
}

// Function 182
vec3 TEXcolor (vec3 p, vec3 incidentd) {
    vec3 RFXp = RFX(p);
    vec3 headingd = reflect(incidentd, RFXp);
    // lightning that gets obsorbed
    float lightness = 0.;
    // diffuse light
        // point light
        vec3 light1p = vec3(0., 2., 0.);
        vec3 light1d = normalize(light1p-p);
        // directional light
        light1d = normalize(vec3(cos(iTime), 1., sin(iTime)));
    	float diff = 0.;
    	diff = dot(light1d, RFXp);
        // diff = dot(light1d, headingd);
    	// float light1dzx = atan(light1d.z, light1d.x);
    	// float headingdzx = atan(headingd.z, headingd.x);
    	// float headingdzxr = sqrt(headingd.z*headingd.z+headingd.x*headingd.x);
    	// headingd.x = headingdzxr*cos(headingdzx-light1dzx);
    	// headingd.z = headingdzxr*sin(headingdzx-light1dzx);
    	// float light1dyx = atan(light1d.y, light1d.x);
    	// float headingdyx = atan(headingd.y, headingd.x);
    	// float headingdyxr = sqrt(headingd.y*headingd.y+headingd.x*headingd.x);;
    	// headingd.x = headingdyxr*cos(headingdyx+3.141/2.-light1dyx);
    	// headingd.y = headingdyxr*sin(headingdyx+3.141/2.-light1dyx);
    	// diff = acos(headingd.y)/3.14159;
    	// diff = clamp(headingd.y, 0., 1.);
    	// diff = 1.-acos(diff)/3.14159;
    	diff = clamp(diff/2.+1., 0., 1.);
    lightness += diff;
    // ambient light
    // lightness = 0.3+0.7*lightness;
    // lightness = clamp(lightness, 0., 1.);
    
    // texture that absorbs tje light just like in real life
    float czk = mod(floor(p.x)+floor(p.z)+floor(p.y*0.), 2.);
    // czk = mod(floor(p.x*3.)+floor(p.y*3.), 2.)/2.;
    czk = mod(floor(p.y), 2.);
    // czk = 1.;
    vec3 col = vec3(.2)+vec3(0., czk*.5, czk*.5);
    // combined them to get a thing 'sumcol'
    vec3 sumcol = hadamard(lightcol*lightness, col);
    // then add specular wich reflects all the light that hits it
    // from major light sources
    float specular = 0.;
    specular = clamp(dot(light1d, headingd), 0., 1.);
    specular = pow(specular, 3.)/2.;
    sumcol += specular;
    return sumcol;
}

// Function 183
void color(inout vec4 O, vec2 P, float d) {             // --- blend with color scheme
    if (d<2./R.y) {
        float c = texture(iChannel1,(.5+P)/R).x;        // cycle Id
        vec4  C = ( .5+.5*sin(1e3*c+vec4(0,2.1,-2.1,0) ) ) * (.7+.7*rnd(c));
        O = mix(O, C, smoothstep(2./R.y,0.,d));
        //O = vec4(mod(c,N),floor(c/N),0,0)/N, O.b = 1.-O.r-O.g;
    }
}

// Function 184
vec3 AdjustColorForFog(vec3 color, float depth, float height)
{
	vec3 fogColor = AmbientSnowColor;
    float fogHeight = 60.0;

	vec3 lerpFogColor = mix( color, fogColor, 1.0-exp(-0.0045*depth) );
    return mix(lerpFogColor, color, min(max(height, 0.0), fogHeight) / fogHeight);
}

// Function 185
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

// Function 186
float mixColors(float r, float v, float z){
    return clamp(0.5 + 0.5 * (v-r) / z, 0., 1.); 
}

// Function 187
vec4 colorAndDepth(vec3 pos, vec3 dir){
    vec3 normal;
    if(hit(pos, dir, normal)){
        return vec4(max(glassCol(dir, normal), vec3(0)), length(pos-CamPos));
    }
   	return vec4(background(dir), RenderDistance);
}

// Function 188
vec4 BlurColor (in vec2 Coord, in sampler2D Tex, in float MipBias)
{
	vec2 TexelSize = MipBias/iChannelResolution[0].xy;
    
    vec4  Color = texture(Tex, Coord, MipBias);
    Color += texture(Tex, Coord + vec2(TexelSize.x,0.0), MipBias);    	
    Color += texture(Tex, Coord + vec2(-TexelSize.x,0.0), MipBias);    	
    Color += texture(Tex, Coord + vec2(0.0,TexelSize.y), MipBias);    	
    Color += texture(Tex, Coord + vec2(0.0,-TexelSize.y), MipBias);    	
    Color += texture(Tex, Coord + vec2(TexelSize.x,TexelSize.y), MipBias);    	
    Color += texture(Tex, Coord + vec2(-TexelSize.x,TexelSize.y), MipBias);    	
    Color += texture(Tex, Coord + vec2(TexelSize.x,-TexelSize.y), MipBias);    	
    Color += texture(Tex, Coord + vec2(-TexelSize.x,-TexelSize.y), MipBias);    

    return Color/9.0;
}

// Function 189
void getCloudColorOld(
    vec3 Psky,
    vec3 ro, vec3 rd, vec3 Lo, vec3 L,
    out float density,
    out float Energy, out float cloud_mask, inout vec3 cloudColor,
	float daynight)
{
    vec3 pc = Psky;
    vec3 Lpc = normalize(Lo-pc);
    float p = 0.0;
    density = 0.0;
    
    #define CLOUDRMSTEP 0.5
    #define CLOUDSTEPS 5

    Energy = 1.0;
        vec3 t = normalize(vec3(-1.0, 0.0, 1.0))*TIME(iTime);
    vec3 P = pc;
    for(int i=0;i<CLOUDSTEPS;++i)
    {
        //vec3 P = pc+Lpc*(float(i)*CLOUDRMSTEP);
        P += normalize(Lpc)*CLOUDRMSTEP;
        P = RotXYZ(P, normalize(vec3(0.4, 0.4, 0.0)));
            	
        //float height = mix(1.0, 0.1, clamp(P.y/5.0, 0.0, 1.0));
        float mask = clamp(gaussianNoise3D(P*0.006+t), 0.0, 1.0);
        mask *= mask*mask*mask;
        float dt = cloudNoise3D((P/*height*/)*0.0002, t);
        density += dt*mask;
        if(density>1.0)
            break;
    }
    density = clamp(density, 0.0, 1.0);
    // formula from Hzzd
    // Beer-law + Henyey-Greenstein phase function
    //density *= 2.0;
    p = 5.0;//max(3.0, sin(TIME)*9.0); // 3, 6, 9 // paper Hzzd
    float g = mix(0.8, 0.2, daynight); // 0.2 paper Hzzd
    float RdotL = dot(rd, L);
    Energy = getEnergyScattered(p, g, density, RdotL);

    cloud_mask = clamp(density*4.0, 0.0, 1.0);
    cloudColor *= Energy;
}

// Function 190
vec4 gradientColor(float t, float dist){
    vec3 col1 = 1.1*t*vec3(0.0,1.0,0.0) + (1.0-1.1*t)*vec3(0.0,0.0,1.0);
    vec3 col2 = smoothstep(0.1,0.0,t)*vec3(0.6) + smoothstep(0.0,0.1,t)*col1;
    
    float alpha = exp(-4.0*abs(dist)) * t * t; //not finished.
	return vec4(col2,alpha);
}

// Function 191
vec2 colorShift(vec2 uv) {
	return vec2(
		uv.x,
		uv.y + sin(iTime)*0.02
	);
}

// Function 192
vec4 colors(int c) {
    if (c ==  0) return vec4(0x00,0x00,0x00,1);
    if (c ==  1) return vec4(0xFF,0xFF,0xFF,1);
    if (c ==  2) return vec4(0x68,0x37,0x2B,1);
    if (c ==  3) return vec4(0x70,0xA4,0xB2,1);
    if (c ==  4) return vec4(0x6F,0x3D,0x86,1);
    if (c ==  5) return vec4(0x58,0x8D,0x43,1);
    if (c ==  6) return vec4(0x35,0x28,0x79,1);
    if (c ==  7) return vec4(0xB8,0xC7,0x6F,1);
    if (c ==  8) return vec4(0x6F,0x4F,0x25,1);
    if (c ==  9) return vec4(0x43,0x39,0x00,1);
    if (c == 10) return vec4(0x9A,0x67,0x59,1);
    if (c == 11) return vec4(0x44,0x44,0x44,1);
    if (c == 12) return vec4(0x6C,0x6C,0x6C,1);
    if (c == 13) return vec4(0x9A,0xD2,0x84,1);
    if (c == 14) return vec4(0x6C,0x5E,0xB5,1);
    if (c == 15) return vec4(0x95,0x95,0x95,1);
    return vec4(0);
}

// Function 193
vec3 liftedDomainColor(vec2 z)
{
    PolarComplex polar = H_toPolar(z);

    float magnitude = (1.0-1.0/pow(2.0,polar.norm)) * 0.9 + 0.1;
    float logradius = log(polar.norm);

    //black rings
    float fractlog = fract(logradius);
    float ringdist = min(abs(fractlog-0.5), fractlog > 0.5 ? 1.0-fractlog : fractlog);
    float ring = (1.0 - smoothstep(0.00, 0.02, ringdist)) * 0.8;

    //white rays
    float k = 12.0;
    float sectorsize = (tau32) / k;
    float anglemod = mod(polar.argument, sectorsize);
    float sectordist = anglemod > sectorsize/2.0 ? sectorsize-anglemod : anglemod;
    float raywidth = 0.02;
    float ray = (1.0 - smoothstep(0.0, raywidth, sectordist)) * 0.8;

    //infinity will be white
    float infinityFade = pow(magnitude,100000000.0);

    //growth ring shade
    float growth = (fractlog)*0.7 + 0.3;
    float darkening = uclamp(1.5*magnitude * (fractlog*0.5 + 0.5) + ray + infinityFade);

    float hue = polar.argument/tau32;
    float saturation = 1.0 - infinityFade;
    float value = darkening;
    
    vec3 color = hsv2rgb(vec3(hue, saturation, value));
    color = mix(color, vec3(1.0), darkening * ray);
    color = mix(color, vec3(0.0), darkening * (ring-infinityFade));

    return color;
}

// Function 194
vec3 colorBars( float x )
{
    return step(.5, fract(vec3(1. - x) * vec3(2., 1., 4.)));
}

// Function 195
vec3 getSunColor(in vec3 p, in float time) {
    float lava = smoothNoise((p+vec3(time*.01))*SUN_DENSITY );
	vec3 color = mapping(1. - sqrt(lava), 0.,1., LAVA_COLOR, LAVA_COLOR_DIST);
	color += color*color;
// With white area, but 2 time slower 
//    float lava2 = smoothNoise((p+vec3(time*.0025))*SUN_DENSITY*.12 );
//    vec3 color2 = mapping(lava2*lava2, 0.,1., LAVA_COLOR2, LAVA_COLOR_DIST2);
//	color += .5*color2;
    return SunTwinklingFactor*color; // todo: le faire sur une constante
}

// Function 196
vec3 getJellyColor(vec2 uv)
{
    return mix(texture(iChannel0, uv).rgb, vec3(0.8), smoothstep(texture(iChannel0, uv).a, 0.07, 1.0));
}

// Function 197
void mixColorPoint(vec2 uv,inout vec3 col,vec2 colPoint,float scale)
{
    //float dist = length(uv - colPoint) * scale;
    //dist = pow(dist,0.25);
    //dist = 1.0 - smoothstep(0.0,1.0,dist);
    
    vec2 uv_ = (uv - colPoint)*scale*24.0;
    float dist = dot(uv_,uv_);
    dist = 1.0 / ( 1.0 + dist );
    
    col = mix(
        col , 
        hash3point(colPoint) ,
        dist
    );
}

// Function 198
vec4 quadColorVariation (in vec2 center, in float size) {
    // this array will store the grayscale of the samples
    vec3 samplesBuffer[SAMPLES_PER_ITERATION];
    
    // the average of the color components
    vec3 avg = vec3(0);
    
    // we sample the current space by picking pseudo random samples in it 
    for (int i = 0; i < SAMPLES_PER_ITERATION; i++) {
        float fi = float(i);
        // pick a random 2d point using the center of the active quad as input
        // this ensures that for every point belonging to the active quad, we pick the same samples
        vec2 r = hash22(center.xy + vec2(fi, 0.0)) - 0.5;
        vec3 sp = texture(iChannel0, center + r * size).rgb;
        avg+= sp;
        samplesBuffer[i] = sp;
    }
    
    avg/= F_SAMPLES_PER_ITERATION;
    
    // estimate the color variation on the active quad by computing the variance
    vec3 var = vec3(0);
    for (int i = 0; i < SAMPLES_PER_ITERATION; i++) {
    	var+= pow(samplesBuffer[i], vec3(2.0));
    }
    var/= F_SAMPLES_PER_ITERATION;
    var-= pow(avg, vec3(2.0));
        
    return vec4(avg, (var.x+var.y+var.z)/3.0);
}

// Function 199
vec3 getWaveColor( in vec3 p, in vec3 projClosest, in vec3 projSecondClosest,
                  in vec3 dir, float dist, vec2 frag ) {
    float distanceToEdge = abs(projClosest.z-projSecondClosest.z);
    float distanceFrac = smoothstep(-10.0, 100.0, dist);
    distanceFrac *= distanceFrac; distanceFrac *= distanceFrac;
    float frac = smoothstep(0.0, 0.1+distanceFrac*0.9, distanceToEdge);
    // get the reflection
    vec3 norm = normal(p, projClosest);
    vec3 color = getBackgroundColor(reflect(dir, norm));
    // add a screenspace scanline
    frac *= (sin(frag.y/iResolution.y*700.0)*0.5+0.5)*(1.0-distanceFrac);
    return mix(vec3(0.43, 0.77, 0.85), color, frac);
}

// Function 200
vec4 colorize(){
	vec4 c = vec4(0);
	for (int i = 0; i < SPHERES; i++) {
		c +=  clr[i] * pow(frc[i], 1.2);
	}
  return c;
}

// Function 201
vec3 rendererCalculateColor( vec3 ro, vec3 rd )
{
    vec3 normal;
    
    vec3 tcol = vec3(0.0);
    vec3 fcol = vec3(1.0);
    
    for(int i=0; i<BOUNCES; i++)
    {
        vec3 scol;
        vec2 intersec = worldIntersect(ro, rd, normal, scol);
		
        if(intersec.y < 0.0)
        {
            tcol += fcol*worldGetBackground(rd);
            break;
        }
		else
        {
            // When we meet a material, we compute separately the
            // direct and indirect lighting
            float fre = 0.04 + 0.96 * pow(1.-clamp(dot(normal,-rd),0.,1.), 5.);
        	vec3 pos = ro + rd * intersec.x;
        	vec3 dcol = worldDirectLighting(pos, normal, fre);
			// Add direct lighting
            fcol *= scol;
            tcol += fcol * dcol;
            // And then bounce for gathering the next level of indirect lighting
            ro = pos + 1e-4*normal;
            rd = worldGetBRDFRay(pos, normal, rd, fre);
        }
    }
    return tcol;
    //vec3 bounce = reflect(rd, normal);
    //col = worldGetBackground(normalize(bounce)) * scolor;
}

// Function 202
vec3 getVaporWaveColor( float offset )
{   
    /********************
     * Color variations *
	 * R    G     B     * 
     * 80   70   220    *
     * ->	=   <-      *
     * 120  70	190     *
     * => 40 & 30 p sec *
     * intervals        *
     ********************/
    
    float intPartTimeFloat;
    float fracTimeVar = modf(iTime/2.0+offset, intPartTimeFloat)*2.0 - 1.0; // [-1;1] per second
    int intPartTime = int(intPartTimeFloat);
    bool evolIntPart = (intPartTime%2 == 0);
    float rVar, bVar;
    if(evolIntPart)
    {
        rVar = 40.0*fracTimeVar / 255.0; // ( [0;40] -> [40;0] ) / 255 (over 2 sec) 
        bVar = 30.0*fracTimeVar / 255.0; // ( [0;3] -> [30;0] ) / 255 (over 2 sec) 
    }
    else
    {
        rVar = (1.0 - 40.0*fracTimeVar) / 255.0;
        bVar = (1.0 - 30.0*fracTimeVar) / 255.0;
    }
    
    // Time varying pixel color
    return vec3(0.39+rVar, 0.27, 0.8+bVar);
}

// Function 203
vec3 lampg_color(vec3 pos)
{
    //vec3 posr = rotateVec2(pos);
    vec2 hmp = getLampMPos(pos);
    //vec3 lc = mix(vec3(1.), getLampColor(hmp), smoothstep(0.91, 0.97, length(posr.xz)/lampsize.x));
    return getLampColor(hmp);
}

// Function 204
vec3 colorFromWavelength(float wavelength) {
    const float gamma = 0.8;
    float r, g, b;
    if (wavelength >= 380. && wavelength <= 440.) {
        float attenuation = .3 + .7 * (wavelength - 380.) / (440. - 380.);
        r = pow((-(wavelength - 440.) / (440. - 380.)) * attenuation, gamma);
        g = 0.;
        b = pow(1.0 * attenuation, gamma);
    } else if (wavelength >= 440. && wavelength <= 490.) {
        r = 0.;
        g = pow((wavelength - 440.) / (490. - 440.), gamma);
        b = 1.;
    } else if (wavelength >= 490. && wavelength <= 510.) {
        r = 0.;
        g = 1.;
        b = pow(-(wavelength - 510.) / (510. - 490.), gamma);
    } else if (wavelength >= 510. && wavelength <= 580.) {
        r = pow((wavelength - 510.) / (580. - 510.), gamma);
        g = 1.;
        b = 0.;
    } else if (wavelength >= 580. && wavelength <= 645.) {
        r = 1.;
        g = pow(-(wavelength - 645.) / (645. - 580.), gamma);
        b = 0.;
    } else if (wavelength >= 645. && wavelength <= 750.) {
        float attenuation = .3 + .7 * (750. - wavelength) / (750. - 645.);
        r = pow(1. * attenuation, gamma);
        g = 0.;
        b = 0.;
    } else {
        r = 1.;
        g = 1.;
        b = 1.;
    }
    return vec3(r, g, b);
}

// Function 205
vec3 getColor(vec3 normal, vec3 pos) {
	return vec3(1.0,1.0,1.0);
}

// Function 206
vec4 color(vec3 ro, vec3 rd, inout random_state rs, float tm) {   
    vec3 emit_accum = vec3(0.0);
    vec3 attenuation_accum = vec3(1.0);
    int depth = 0;
    bool done = false;
    hit_record rec;
    rec.t = 0.001;
    int max_depth = 1000;
    float tt;
    rd = normalize(rd);
    float initcohdist = 0.0*dot(rd,0.01*vec3(3.0,7.0,11.0));
    rec.cohdist = initcohdist;
    while (!done) {
        tt = rec.t;
        bool hit = rm_hit(ro, rd, rec.t, 1E9, rec, tm);
        
        if (hit && depth < max_depth) {
            if (rec.cohdist>=initcohdist) rec.cohdist += tt;
            vec3 scro, scrd;
            vec3 attenuation;
            vec3 emitcol = emitted(rec);
            emit_accum += emitcol * attenuation_accum;
            if (scatter(rec, ro, rd, attenuation, scro, scrd, rs, rec.cohdist)) {
                attenuation_accum *= attenuation;
                ro = scro;
                rd = normalize(scrd);
                rec.t = 0.001;
            } else {
                done = true;
            }
        } else if (depth >= max_depth) {
            vec3 unit_direction = normalize(rd);
            float t = 0.5 * (unit_direction.y + 1.0);
            vec3 albedo = ((1.0-t)*vec3(1.0) + t*vec3(0.25,0.5,1.0));
            emit_accum += attenuation_accum * albedo * 0.01;
            done = true;
        }    
        depth += 1;
    }
    
    return vec4(emit_accum,abs(rec.cohdist)); 
}

// Function 207
vec4 DistanceToBoundaryAndColor(vec2 p)
{
    vec4 ret = vec4(10000.0, 0.0, 0.0, 0.0);
    
    #if SCENE == 1 || SCENE == 2
    {
        float dist = -(length(p) - 0.4);
        if (dist > -0.01)
        {
        	float angle = atan(p.y, p.x);
            if (angle < 0.0)
                angle += c_pi * 2.0;
            
            #if SCENE == 2
                float percent = angle / (2.0 * c_pi);
            	float shade = sin(angle * 10.0);
            	vec3 color = smoothRainbow(percent) * shade;
            #else
            	float shade = sin(angle * 10.0);
            	vec3 color = vec3(shade, shade, shade);           
            #endif
        	ret.x = dist;
        	ret.yzw = color;
        }
        else
        {
            ret = vec4(0.0, 0.8, 0.8, 0.8);
        }
        
    }
    #elif SCENE == 3
    {
    	LineTest(p, vec2(0.0, -0.2), vec2(0.3, 0.2), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), ret);
    	LineTest(p, vec2(0.3, 0.2), vec2(-0.3, 0.2), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), ret);
    	LineTest(p, vec2(-0.3, 0.2), vec2(0.0, -0.2), vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), ret);
    }
    #elif SCENE == 4
    {
    	LineTest(p, vec2(0.3, -0.2), vec2(0.3, 0.2), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), ret);
    	LineTest(p, vec2(0.3, 0.2), vec2(-0.3, 0.2), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), ret);
    	LineTest(p, vec2(-0.3, 0.2), vec2(-0.3, -0.2), vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), ret);
    }    
    #elif SCENE == 5
    {
        CircleTest(p, vec3(0.0, -0.2, 0.03), vec3(1.0, 0.0, 0.0), ret);
        CircleTest(p, vec3(0.3, 0.2, 0.03), vec3(0.0, 1.0, 0.0), ret);
        CircleTest(p, vec3(-0.3, 0.2, 0.03), vec3(0.0, 0.0, 1.0), ret);

        LineTest(p, vec2(-0.7, 0.4), vec2(-0.7, -0.4), vec3(0.0, 0.0, 0.0), vec3(0.8, 0.8, 0.8), ret); 
        LineTest(p, vec2( 0.7, 0.4), vec2( 0.7, -0.4), vec3(0.8, 0.8, 0.8), vec3(0.0, 0.0, 0.0), ret); 
    }
    #endif
    
    return ret;
}

// Function 208
vec3 getGlassAbsColor(float dist, vec3 color)
{
    return pow(color, vec3(0.1 + pow(dist*2.5, 2.)));
}

// Function 209
PathColor ColorAdd( PathColor a, PathColor b )
{
#if SPECTRAL    
    return PathColor( a.fIntensity + b.fIntensity );
#endif    
#if RGB
    return PathColor( a.vRGB + b.vRGB );
#endif    
}

// Function 210
vec3 lightColor()
{
    return vec3(1.0, 0.9, 0.75) * 3.9;
}

// Function 211
vec3 skyColor(vec3 dir)
{
    // Pure Black Void
    return vec3(0.0);
}

// Function 212
vec3 colorscale (float x) {
    return mix(color1, color2, x);
}

// Function 213
void getCloudColor(
    vec3 Psky,
    vec3 ro, vec3 rd, vec3 Lo, vec3 L, float Li,
    out float density,
    out float Energy, out float cloud_mask, inout vec3 cloudColor,
	float daynight)
{
    vec3 pc = Psky;
    vec3 Lpc = normalize(Lo-pc);
    float p = 0.0;
    density = 1.0;

    Energy = 1.0;
    vec3 t = normalize(vec3(-1.0, 0.0, 1.0))*TIME(iTime);
    vec3 P = pc;
    
    // formula from Hzzd
    // Beer-law + Henyey-Greenstein phase function
    //density *= 2.0;
    p = 9.0;//max(3.0, sin(TIME)*9.0); // 3, 6, 9 // paper Hzzd
    float g = mix(0.8, 0.2, daynight); // 0.2 paper Hzzd
    float RdotL = dot(rd, L);
   
    //https://is.muni.cz/th/396277/fi_m/thesis.pdf
	cloud_mask = 1.0;
    
    float attScat = 1.0;
    
    density = 0.0;
    float energy = 0.0;
    P = ro+rd*(10.0);
  
    for(int i=0;i<4;++i)
    {
        //P = ro+rd*(10.0+100.0*density);
        
        float dt = cloudNoise3D((P/*height*/)*0.0002, t);
        
        if(density>0.5)
            break;
        
        //rd = RotZV3(RotYV3(RotXV3(rd, dt), dt), dt);
        P += rd*100.0*dt;
    	RdotL = dot(rd, normalize(P-Lo));
        
        float e = 0.0;
        e = getEnergyScattered(p, g, density, RdotL)*Li/float(i+1);
        
    	energy += e;//*mix(-1.0, 1.0, Fresnel_Schlick(1.0, 1.1, RdotL));
        density+=dt;
    }

    cloudColor = vec3(energy);
}

// Function 214
vec3 getBorderColor(float x0, float x1, vec2 uv,
              vec3 fragColor) {
        
    vec3 rightCol = vec3(0.);
    vec3 leftCol = vec3(0.);
        
    leftCol = borderColor(x0, x1, uv, 1., 0.);       
    rightCol = borderColor(x0, x1, uv, 0., 1.);
   
    if (leftCol != vec3(0.))
        return leftCol;
    else if (rightCol != vec3(0.))
   		return rightCol;                 
   
    return fragColor;   
}

// Function 215
vec4 rainbowColor(vec2 Coord)
{
    vec2 xy = Coord/iResolution.xy;
    float geometry = -1.*(sin(xy.x*pi/2.)+cos(xy.y*pi/2.))/6.;
    vec4 texColor = vec4(
        fractured(0. + geometry),
        fractured(0.33 + geometry),
        fractured(0.67 + geometry),
        1.0);
    return texColor;
    
}

// Function 216
vec3 colorDodge(in vec3 src, in vec3 dst)
{
    return step(0.0, dst) * mix(min(vec3(1.0), dst/ (1.0 - src)), vec3(1.0), step(1.0, src)); 
}

// Function 217
vec3 getObjectColor(vec3 p, vec3 n, vec3 e) {
    float sum = textureHologram(p.xy,e);
    
    // vec
    vec2 d = p.xy + (sum * 2.0 - 1.0);
    d += dot(e,n) * 5.5;
        
    // get holo color
    float bright = saturate(0.6 + sum * 0.4);
    vec3 color = hologram(d,sum) * bright;
    color *= pow(max(dot(e,-n),0.0),0.6);
            
    // reflection
    vec3 refl = reflect(e,n) + sum * 0.1; 
    vec3 color_refl = texture(iChannel0,refl).xyz;
    color = mix(color,color_refl,(1.0 - sum) * 0.2);    
        
    // lighting
    n.xz += (sum * 2.0 - 1.0) * 0.15;
    color += pow(max(dot(e,-normalize(n)),0.0), 20.0) * 0.6;
        
    return color;    
}

// Function 218
vec3 debugcolors(float value)
{
    //return rainbow(10.0 * value * (1.0 + sqrt(5.0)) / 2.0);
    //return rainbow(rand(vec2(value)));
    return rainbow(rnd(value));
    //return rainbow(fract(sin(value*17.)*1e4));
    //return .5 + .5 * cos(6.3 * (fract(sin(value)*1e4) + vec3(0,1,2)/3.));
}

// Function 219
vec3 scene_surface_color( vec3 r, mat2x3 Kr, float t, vec3 V, vec2 uv, bool submerged,
                          inout vec3 albedo, inout vec3 N )
{
    float Krwidth = sqrt( max( dot( Kr[0], Kr[0] ), dot( Kr[1], Kr[1] ) ) );
    vec4 tsample = ts_sample_fine( g_ts, iChannel1, g_data, r, Krwidth );
    vec2 tshadow = ts_shadow_sample_ao( g_ts, iChannel1, r );
    N = normalize( tsample.xyz );
    vec3 Z = normalize( r );
    float h = tsample.w;
    float pshadow = atm_planet_shadow( dot( g_env.L, Z ), sqrt( max( 0., 1. - g_env.radius * g_env.radius / dot( r, r ) ) ) );
    vec4 sky = atm_skylight_sample( g_ts, iChannel2, r );
    vec3 TL = atm_transmittance( Z * max( g_data.radius, length( r ) ), g_env.L, g_atm, true );
    vec3 F = g_env.sunlight * TL * sky.w * pshadow;
    float slope = length( N / dot( N, Z ) - Z );
    albedo = scene_surface_albedo( r, Kr, h, slope, Z.z, h >= 0. || submerged );
    float d = submerged ?
        max( 0.001, -.25 * h ) :
        max( max( 0.001, 125. * h * h ), 4. * t * sqrt( g_pixelscale ) );
    vec4 M = scene_ocean_normal_and_lensing( r, t, h, d, V, Z );
    vec3 L = g_env.L;
    if( submerged )
    {
        F = F * M.w;
        F = F * saturate( 1. - fresnel_schlick( .02, dot( M.xyz, L ) ) );
        L = normalize( -simple_refract( -L, Z ) );
    }
    float oshadow = scene_raycast_object_shadows( Ray( r, L ) );
    vec3 col = ZERO;
    if( h < 0. && !submerged )
    {
        // water surface
        vec3 To = exp2pp( 1000. * h * g_ocn_beta50 );
        vec4 rsample = atm_reflection_sample( iChannel2, uv );
        vec3 albedo = mix( g_ocn_omega, To * albedo, To );
        F = F * M.w;
        F = F * mix( ONE, vec3( tshadow.x * oshadow ), To );
        const float cld_g = 0.85;
        const float cld_f = cld_g * cld_g;
        float extra_T = pow( sky.w, inversesqrt( 1. - cld_f ) - 1. );
        float a = sqrt( ( .0003 * inversesqrt( g_pixelscale ) + t ) / t ) * 0.8 / g_data.ocn_s2;
        vec3 M = ndist( Z, .25, trn_ripplemap( r + 0.002 * iTime * Z ) );
        col = scene_lighting_ocean( albedo, Z, N, M.xyz, g_env.L, V, F, a, sky.xyz, rsample, extra_T );
    }
    else
        if( t < SCN_ZFAR )
        {
            // land surface
            col = scene_lighting_terrain( albedo, N, L, V, Z, F * oshadow, sky.xyz, tshadow );
        }
    return col;
}

// Function 220
vec3 GetAmbientShadowColor()
{
    return vec3(0, 0, 0.2);
}

// Function 221
vec3 selfColor(vec3 pos) {
    vec3 pol = carToPol(pos);
    return spectrum(1.0*pol.z/PI/2.0+0.5*pol.y/PI);
}

// Function 222
vec3 get_color(float color){
    if(color == BLUE){
    	return vec3(0.149,0.141,0.912);
   	}
    else if(color == GREEN){
    	return vec3(0.000,0.833,0.224);
   	}
    else if(color == RED){
    	return vec3(1.0,0.0,0.0);
   	}
    else if(color == WHITE){
    	return vec3(1.0,1.0,1.0);
   	}
    else if(color == GRAY){
    	return vec3(192.0,192.0,192.0)/255.0;
    }
    else if(color == YELLOW){
    	return vec3(1.0,1.0,0.0);
   	}
    else if(color == ORANGE){
    	return vec3(255,127,80)/255.0;
    }
    else if(color == BLACK){
    	return vec3(0.0,0.0,0.0);
   	}
}

// Function 223
vec3 getColor(vec3 ro, vec3 rd)
{
    vec3 color = vec3(0.0);
    vec3 col = vec3(1.0);
    int id=-1;
    int tm = -1;
    
    for(int i=0; i<6; i++)
    {
    	float t = 10000.0; //seed++;
		
   		vec2 tRoom = intersectCube(ro, rd, box0);          
   		if(tRoom.x < tRoom.y)   t = tRoom.y; 
    
    	intersectscene(ro, rd, t, id, true);
    
    	vec3 hit = ro + rd * t;        
		vec4 mcol = vec4(vec3(0.99),0.0);
    	vec3 normal; 
    	vec2 mref = vec2(0);
      
    	ColorAndNormal(hit, mcol, normal, tRoom, mref, t, id);
    	hit = hit + normal * 0.00001;
         
        vec2 rnd = rand2();
        //rnd.x = 1.0/6.0 * ( float(i) + rnd.x );
        col *= mcol.xyz;
        if(mcol.w>0.0) 
        {
            if(i==0) {color = mcol.xyz; break;}
            float df=max(dot(rd,-normal),0.0)*2.0; //if(tm==1) df *= 19.0;
            color += col*mcol.xyz*mcol.w * df ;
            //if(tm==1) color += col * 1.5;
            break;
        }
		tm = -1;
        if(rnd.x>abs(mref.x))//diffuse
        {
        	rd = CosineWeightedSampleHemisphere ( normal, rnd);      
        	tm = 0;   
        
        	col *= clamp(dot(normal,rd),0.0,1.0);
           // color += col * 0.1;
            
            bool isLight = false;
         	//vec3 rnd3 = vec3(rand2(),rand2().x) *2.0 -1.0;
            rnd = rand2()*2.0-1.0;
            //cw = vec2(-0.4,0.1);
         	vec3 lightf = vec3(cw,2.2) + vec3(rnd.x*0.65,rnd.y * 0.6,0.0);
         	vec3 dl = directLight(hit, normal, lightf, vec3(0.9,0.9,0.9), isLight);
            float nd = max(0.0,dot(lightf,vec3(0.0,0.0,1.0)))+max(0.0,dot(lightf,normal));
         	color += col * dl*5.0 *nd;
         	if(isLight) break;
        }       
        else 
        {
            vec3 nrd = reflect(rd,normal); tm = 1;//reflect
       		/*if(mref.x<0.0)//refract
            {
                //if(id==30)
                    //if(dot(rd,normal)>0.0) normal = -normal;
            	vec3 ior=vec3(1.0,1.52,1.0/1.12); tm = 2;
           	 	vec3 refr=refract(rd,normal,(side>=0.0)?ior.z:ior.y);//calc the probabilty of reflecting instead
           	 	vec2 ca=vec2(dot(normal,rd),dot(normal,refr)),n=(side>=0.0)?ior.xy:ior.yx,nn=vec2(n.x,-n.y);
            	if(rand2().y>0.5*(pow(dot(nn,ca)/dot(n,ca),2.0)+pow(dot(nn,ca.yx)/dot(n,ca.yx),2.0)))
               		nrd=refr;
            }*/
            rd = cosPowDir(nrd, mref.y*1.0);
            col *= 1.2;
        }
        
        ro = hit + rd * 0.0001; 
        
        if(dot(col,col) < 0.1 && i>3) break;
    }
    
 	return color;   
}

// Function 224
float diskColorb(in vec2 uv, vec2 offset)
{
    uv = uv - smoothstep(0.01,1.8,texture(iChannel0, (uv*1.0 - vec2((iTime-0.06) /2.65,(iTime-0.06) /7.0)) + offset).r) * 0.3;
    
    float d = length(uv)-RADIUS;
    return smoothstep(0.01,0.015,d);
}

// Function 225
float textColor(vec2 from, vec2 to, vec2 p)
{
	p *= font_size;
	float inkNess = 0., nearLine, corner;
	nearLine = minimum_distance(from,to,p); // basic distance from segment, thanks http://glsl.heroku.com/e#6140.0
	inkNess += smoothstep(0., 1. , 1.- 14.*(nearLine - STROKEWIDTH)); // ugly still
	inkNess += smoothstep(0., 2.5, 1.- (nearLine  + 5. * STROKEWIDTH)); // glow
	return inkNess;
}

// Function 226
vec3 colorize( vec2 rayHitInfo, vec3 eyePos, vec3 rayDir )
{
    vec3 color;
    
  	if( rayHitInfo.y < 0.0 ) {
  		color = bgColor(rayDir);  
  	} else {
      	vec3 hitPos = eyePos + rayHitInfo.x * rayDir;

      	vec3 surfaceNormal = getSurfaceNormal( hitPos );
      
      	if( rayHitInfo.y >= 1. ){
        	color = objColor( hitPos , surfaceNormal ); 
      	}
  	}
    
    return color;
}

// Function 227
float colorDistFast(vec4 c1, vec4 c2){
	float dr = c1.r - c2.r;
    float dg = c1.g - c2.g;
    float db = c1.b - c2.b;
    return dr * dr + dg * dg + db * db;
}

// Function 228
vec4 materialColorForPixel( vec2 texCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    // 0 is top left
    vec2 uv = (texCoord)*1.4;
    // re-set coordinates to center the flame
    // x and y will be between -0.5 and 0.5
    vec2 position = uv - vec2(0.7, 0.6);
    position.y *= -1.;


    // black as background
    // vec3(1.0, 1.0, 0.0) is yellow
    vec3 color = vec3(0.0, 0.0, 0.0);

    // y0 is top and y1 is bottom, so 10x squared 2 will be flipped downwards
    // x values
    float fx = pow(position.x * 6., 2.);

    // flickering generated by moving the coordinate system by a small amount with time as factor
    position.y += fract(sin(TIME * 24.)) * 0.1;

    // 
    float y = length(position + vec2(position.x, fx));

    color.r += smoothstep(0.0, 0.3, 0.71 - y) * 0.71;
    color.g += smoothstep(0.0, 0.3, 0.71 - y) * 0.54;
    //color.rg += 1. - y;

    // Output to screen
    return vec4(color,1.0);
}

// Function 229
vec3 getStickColor(vec3 pos)
{
    return stickColor;
}

// Function 230
vec4 BlurColor (in vec2 Coord, in sampler2D Tex, in float MipBias)
{
	vec2 TexelSize = MipBias/iChannelResolution[0].xy;
    //    o -= o - length(fwidth(texture(iChannel0,u/iResolution.xy)))*3.;
    vec4  Color = vec4(0.0);//texture(Tex, Coord, MipBias);
    Color += length(fwidth(texture(Tex, Coord + vec2(TexelSize.x,0.0), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(-TexelSize.x,0.0), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(0.0,TexelSize.y), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(0.0,-TexelSize.y), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(TexelSize.x,TexelSize.y), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(-TexelSize.x,TexelSize.y), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(TexelSize.x,-TexelSize.y), MipBias)))*3.;    	
    Color += length(fwidth(texture(Tex, Coord + vec2(-TexelSize.x,-TexelSize.y), MipBias)))*3.;    

    return Color/8.;
}

// Function 231
vec3 getColor( in float t )
{
    vec3 col = vec3(0.6,0.5,0.4);
    col += 0.14*mcos(6.2832*t*  1.0+vec3(0.0,0.5,0.6));
    col += 0.13*mcos(6.2832*t*  3.1+vec3(0.5,0.6,1.0));
    col += 0.12*mcos(6.2832*t*  5.1+vec3(0.1,0.7,1.1));
    col += 0.11*mcos(6.2832*t*  9.1+vec3(0.1,0.5,1.2));
    col += 0.10*mcos(6.2832*t* 17.1+vec3(0.0,0.3,0.9));
    col += 0.09*mcos(6.2832*t* 31.1+vec3(0.1,0.5,1.3));
    col += 0.08*mcos(6.2832*t* 65.1+vec3(0.1,0.5,1.3));
    col += 0.07*mcos(6.2832*t*131.1+vec3(0.3,0.2,0.8));
    return col;
}

// Function 232
vec3 color(vec2 n){
    vec3 p=texture(iChannel0,n).rgb;
    #ifdef BRIGHT_SPOTS
    
    p=BriSp(p);
    #endif
    return p;
}

// Function 233
vec3 flame_color_ramp(float val)
{
    vec3 color_a = 1.2 * vec3(1.2, 0.1, 0.2);  // red
    vec3 color_b = 1.8 * vec3(1.0, 0.8, 0.5); // yellow
    vec3 color_c = 0.95 * vec3(0.25, 0.25, 0.4); // blue
    vec3 color_d = 0.5 * vec3(0.35, 0.2, 0.4);  // violet
    
    float pos_a = 0.35;
    float pos_b = 0.65;
    
    if(val < pos_a)
    {
        return mix(color_a, color_b, val / pos_a);
    }
    else if(val < pos_b)
    {
        return mix(color_b, color_c, (val - pos_a) / (pos_b - pos_a));
    }
    return mix(color_c, color_d, (val - pos_b) / (1.0 - pos_b));   
}

// Function 234
bool saveBrickColorData(in ivec2 fragCoord, out vec4 fragColor) {
  if (fragCoord == brickColorDataLocation) {
    if (iFrame == 0) {
      int index = 2;
      fragColor = vec4(toLinear(colors[index]), float(index));
    } else {
      vec4 color = texelFetch(STORAGE, brickColorDataLocation, 0);
      if (isKeyUp(KEY_C)) {
        color.w += 1.0;
      }
      color.w = float(int(color.w) % colorsLength);
      color.rgb = mix(color.rgb, toLinear(colors[int(color.w)]), 0.15);
        
      fragColor = color;
    }
    return true;
  }
    
  return false;
}

// Function 235
vec3 GetSkyColor(in vec3 rayDirection)
{
    vec3 skyColor = GetBaseSkyColor(rayDirection);
    vec4 cloudColor = GetCloudColor(rayDirection * 4.0);
    skyColor = mix(skyColor, cloudColor.rgb, cloudColor.a);

    return skyColor;
}

// Function 236
vec4 color_at_uv(vec2 uv, vec2 p, float t)
{    
    vec2 rad_x = p - uv * vec2(172., 100.) * vec2(sin(t/10.),cos(t/10.)),
         rad_y = p - uv * vec2(242., 163.);
       
    float ii = dot(sin(rad_x)+sin(rad_y), vec2(1));
    // ii = abs(ii); // this is cool too.

    vec4 a_col = vec4(.9, 1.,  1,1),
         b_col = vec4(0, .75,  1,1),
         c_col = vec4(0,  0,   1,1);
    
    float a_bool = step(1.,ii)    +step(.5, ii),
          b_bool = step(2.*-abs(sin(t/5.)), ii),
          c_bool = step(3.,                 ii);
   
    a_col *= a_bool;
    b_col *= b_bool;
    c_col *= c_bool;
    
    return a_col + b_col + c_col;
}

// Function 237
vec4 get_balloon_color(const int material, const float current_level)
{
    vec4 color = vec4(vec3(.25),.35);
    float hue = float(material-BASE_TARGET_MATERIAL)*(1./float(NUM_TARGETS));
    hue = fract(hue + current_level * 1./6.);
    color.rgb += rainbow(hue) * .5;
    return color;
}

// Function 238
vec3 colorGradient(float gradient, int iter, int iterMax, float choose_palette)
{
    //vec3 color1 = vec3(0.1, 0.0, 0.6); //blue
    //vec3 color2 = vec3(1.0, 0.6, 0.0); //orange
    //vec3 palette = mix(color1, color2, gradient);
    
    #ifdef ANIMATE_COLOR
        gradient += 0.2 * iTime;
    #endif
    
    vec3                     palette = pal(gradient, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.33,0.67) );
    if(choose_palette == 2.) palette = pal(gradient, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.10,0.20) );
    if(choose_palette == 3.) palette = pal(gradient, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.3,0.20,0.20) );
    if(choose_palette == 4.) palette = pal(gradient, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,0.5),vec3(0.8,0.90,0.30) );
    if(choose_palette == 5.) palette = pal(gradient, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,0.7,0.4),vec3(0.0,0.15,0.20) );
    if(choose_palette == 6.) palette = pal(gradient, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(2.0,1.0,0.0),vec3(0.5,0.20,0.25) );
    if(choose_palette == 7.) palette = pal(gradient, vec3(0.8,0.5,0.4),vec3(0.2,0.4,0.2),vec3(2.0,1.0,1.0),vec3(0.0,0.25,0.25) );
    
    vec3 col = iter == iterMax ? vec3(0) : palette;
    return col;
}

// Function 239
void getCloudColor(
    vec3 Psky,
    vec3 ro, vec3 rd, vec3 Lo, vec3 L, float Li,
    out float density,
    out float Energy, out float cloud_mask, inout vec3 cloudColor,
	float daynight)
{
    vec3 pc = Psky;
    vec3 Lpc = normalize(Lo-pc);
    float p = 0.0;
    density = 1.0;

    Energy = 1.0;
    vec3 t = normalize(vec3(-1.0, 0.0, 1.0))*TIME(iTime);
    vec3 P = pc;
    
    // formula from Hzzd
    // Beer-law + Henyey-Greenstein phase function
    //density *= 2.0;
    p = 9.0;//max(3.0, sin(TIME)*9.0); // 3, 6, 9 // paper Hzzd
    float g = mix(0.8, 0.2, daynight); // 0.2 paper Hzzd
    float RdotL = dot(rd, L);
   
    //https://is.muni.cz/th/396277/fi_m/thesis.pdf
	cloud_mask = 1.0;
    
    float attScat = 1.0;
    
    density = 0.0;
    float energy = 0.0;
    P = ro+rd*(10.0);
  
    for(int i=0;i<10;++i)
    {
        //P = ro+rd*(10.0+100.0*density);
        
        float dt = cloudNoise3D((P/*height*/)*0.0002, t);
        
        if(density>0.5)
            break;
        
        //rd = RotZV3(RotYV3(RotXV3(rd, dt), dt), dt);
        P += rd*100.0*dt;
    	RdotL = dot(rd, normalize(P-Lo));
        
        float e = 0.0;
        e = getEnergyScattered(p, g, density, RdotL)*Li/float(i+1);
        
    	energy += e;//*mix(-1.0, 1.0, Fresnel_Schlick(1.0, 1.1, RdotL));
        density+=dt;
    }

    cloudColor = vec3(energy);
}

// Function 240
vec3 jungle_color(vec3 p)
{
    //p /= scale;
    p = sin(p/100.0)*100.0;
    vec2 uv = vec2(p.x,p.y)/(400.0);
    float scale = 5.0;
    vec2 col = (uv.yx*scale*(1.0+sin(uv.x+p.z)/2.0));
    vec2 col2;
    for(float i = 1.0; i < 5.0; i++){
        uv += ceil(col+sin((col.x+col.y)));
        col /= sin(uv.x);
        col2 = (col2+col)/(i*i);
    }
    
    // Output to screen
    return sin(vec3(0.5+uv.y/500.0,col2.x,uv.x/500.0))/2.0;
}

// Function 241
float shiftColor(in float sb, in vec3 c, in float cuv, in float cir) {
    if(sb < lUltraViolet) return cuv * sb / lUltraViolet;
    if(sb < lBlue)        return (c.b - cuv) * (sb - lUltraViolet) / (lBlue - lUltraViolet) + cuv;
    if(sb < lGreen)       return (c.g - c.b) * (sb - lBlue)        / (lGreen - lBlue)       + c.b;
    if(sb < lRed)         return (c.r - c.g) * (sb - lGreen)       / (lRed - lGreen)        + c.g;
    if(sb < lInfraRed)    return (cir - c.r) * (sb - lRed)         / (lInfraRed - lRed)     + c.r;
    return (lInfraRed / sb) * cir;
}

// Function 242
vec4 Scene_GetColorAndDepth( vec3 vRayOrigin, vec3 vRayDir )
{
	vec3 vResultColor = vec3(0.0);
            
	SceneResult firstTraceResult;
    
    float fStartDist = 0.0f;
    float fMaxDist = 10.0f;
    
    vec3 vRemaining = vec3(1.0);
    
	for( int iPassIndex=0; iPassIndex < 3; iPassIndex++ )
    {
    	SceneResult traceResult = Scene_Trace( vRayOrigin, vRayDir, fStartDist, fMaxDist );

        if ( iPassIndex == 0 )
        {
            firstTraceResult = traceResult;
        }
        
        vec3 vColor = vec3(0);
        vec3 vReflectAmount = vec3(0);
        
		if( traceResult.iObjectId < 0 )
		{
            vColor = Env_GetSkyColor( vRayOrigin, vRayDir ).rgb;
        }
        else
        {
            
            SurfaceInfo surfaceInfo = Scene_GetSurfaceInfo( vRayOrigin, vRayDir, traceResult );
            SurfaceLighting surfaceLighting = Scene_GetSurfaceLighting( vRayDir, surfaceInfo );
                
            // calculate reflectance (Fresnel)
			vReflectAmount = Light_GetFresnel( -vRayDir, surfaceInfo.vBumpNormal, surfaceInfo.vR0, surfaceInfo.fSmoothness );
			
			vColor = (surfaceInfo.vAlbedo * surfaceLighting.vDiffuse + surfaceInfo.vEmissive) * (vec3(1.0) - vReflectAmount); 
            
            vec3 vReflectRayOrigin = surfaceInfo.vPos;
            vec3 vReflectRayDir = normalize( reflect( vRayDir, surfaceInfo.vBumpNormal ) );
            fStartDist = 0.001 / max(0.0000001,abs(dot( vReflectRayDir, surfaceInfo.vNormal ))); 

            vColor += surfaceLighting.vSpecular * vReflectAmount;            

			vColor = Env_ApplyAtmosphere( vColor, vRayOrigin, vRayDir, traceResult.fDist );
			vColor = FX_Apply( vColor, vRayOrigin, vRayDir, traceResult.fDist );
            
            vRayOrigin = vReflectRayOrigin;
            vRayDir = vReflectRayDir;
        }
        
        vResultColor += vColor * vRemaining;
        vRemaining *= vReflectAmount;        
    }
 
    return vec4( vResultColor, EncodeDepthAndObject( firstTraceResult.fDist, firstTraceResult.iObjectId ) );
}

// Function 243
vec3 color(vec2 i)
{
    vec3 col = texture(iChannel0,i/256.).rgb;
    float m = min(col.r,min(col.g,col.b));
    float M = max(col.r,max(col.g,col.b));
    col = smoothstep(m*.5,M,col);
    return col*col;
}

// Function 244
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

// Function 245
float coloredNoise(float t, float fc, float df)
{
    // Noise peak centered around frequency fc
    // containing frequencies between fc-df and fc+df.
    // Modulate df-wide noise by an fc-frequency sinusoid
    return sin(TAU*fc*mod(t,1.))*noise(t*df);
}

// Function 246
vec3 colorIQ(float i)
{
  vec3 a = vec3(0.5);
  vec3 b = vec3(0.5);
  vec3 c = vec3(1.0);
  vec3 d = vec3(0.0, 0.1, 0.2);
  return (a + b * cos(((c * i + d) * 6.2831852)));
}

// Function 247
float diskColorr(in vec2 uv, vec2 offset)
{
    uv = uv - smoothstep(0.01,1.8,texture(iChannel0, (uv*1.0 - vec2((iTime+0.06) /3.0,(iTime+0.06) /8.0)) + offset).r) * 0.3;
    
    float d = length(uv)-RADIUS;
    return smoothstep(0.01,0.015,d);
}

// Function 248
vec3 getSkyColor(vec3 ro, vec3 rd, float starsQuality, float solarPhase)
{
    #ifdef DAYANDNIGHT
    return mix(getSkyColorForPhase(ro, rd, starsQuality, 0.0),
               getSkyColorForPhase(ro, rd, starsQuality, 1.0),
               solarPhase);
    #else
    return getSkyColorForPhase(ro, rd, starsQuality, 1.0);
    #endif
}

// Function 249
vec3 holoGetColor(mat4 head, vec2 p) {
    float time = iTime * 0.3;
    vec2 uv = p; uv.x *= HOLO_ASPECT;    
    vec3 pos = vec3(head[0][3],head[1][3],head[2][3]);
    
#ifdef PIXELIZE
    uv = floor(uv*HOLO_RESOLUTION) / HOLO_RESOLUTION;
#endif
    
    float i = 0.0;    
    for(int it = 0; it < HOLO_LINES; it++) {
        // vertical
        vec3 v0 = vec3(-HOLO_SIZE.x + dxdy.x * float(it+1),0.0,-HOLO_SIZE.y);
        vec3 v1 = vec3(v0.x,-HOLO_DEPTH,v0.z);
    	i += projectLine(pos,uv,v0,v1);
        
        v0 = vec3(-HOLO_SIZE.x + dxdy.x * float(it),0.0, HOLO_SIZE.y);
        v1 = vec3(v0.x,-HOLO_DEPTH,v0.z);
    	i += projectLine(pos,uv,v0,v1);      
        
        v0 = vec3(-HOLO_SIZE.x, 0.0, -HOLO_SIZE.y + dxdy.y * float(it));
        v1 = vec3(v0.x,-HOLO_DEPTH,v0.z);
    	i += projectLine(pos,uv,v0,v1);  
        
        v0 = vec3(HOLO_SIZE.x, 0.0, -HOLO_SIZE.y + dxdy.y * float(it+1));
        v1 = vec3(v0.x,-HOLO_DEPTH,v0.z);
    	i += projectLine(pos,uv,v0,v1);
        
        // horizontal
        float h = -float(it)*HOLO_DDEPTH;
        v0 = vec3(-HOLO_SIZE.x,h,-HOLO_SIZE.y);
        v1 = vec3(-v0.x,v0.y,v0.z);
    	i += projectLine(pos,uv,v0,v1);
        
        v0 = vec3(-HOLO_SIZE.x,h,HOLO_SIZE.y);
        v1 = vec3(-v0.x,v0.y,v0.z);
    	i += projectLine(pos,uv,v0,v1);
        
        v0 = vec3(-HOLO_SIZE.x,h,-HOLO_SIZE.y);
        v1 = vec3(v0.x,v0.y,-v0.z);
    	i += projectLine(pos,uv,v0,v1);
        
        v0 = vec3(HOLO_SIZE.x,h,-HOLO_SIZE.y);
        v1 = vec3(v0.x,v0.y,-v0.z);
    	i += projectLine(pos,uv,v0,v1);       
    }
    
    vec3 color = vec3(min(i,1.0));    
    projectCircle(pos,uv,vec3(-0.5,-3.0,0.0),color);
    projectCircle(pos,uv,vec3( 0.5,-1.0,0.0),color);
    projectCircle(pos,uv,vec3( 0.0, 1.0,0.0),color);
    color += texture(iChannel0,uv*0.5).z * 0.3;
    return color;
}

// Function 250
vec3 colormap(float t) {
    return PAL(t, vec3(0.5,0.5,0.5), vec3(0.5,0.5,0.5), vec3(1.0,1.0,1.0), vec3(0.0,0.33,0.67));
}

// Function 251
vec3 sphereColor(vec3 worldPos, float nDotV, float dist, float worldAngle)
{    
    // which planet are we talnikg about already?
    // This is done way to much for final rendering, could be optimized out
    planet p;
	vec3 sector = floor(worldPos);
    GetPlanet(sector, p);

    // Scale AA accourding to disatnce and facing ratio
   	float aaScale = 4.0 - nDotV * 3.8 + min(4.0, dist * dist * 0.025);
    
    // Find local position on the sphere
    vec3 localPos = worldPos - (sector + p.center);
    
    // Random seed that will be used for the two flower layers
    vec4 rnd = N24(vec2(sector.x, sector.y + sector.z * 23.4));
    vec4 rnd2 = N24(rnd.xy * 5.0);
    
    // compensate for the world Z rotation so planets stay upright
    localPos = (rotationZ(-worldAngle) * vec4(localPos, 0.0)).xyz;
    // Planet rotation at random speed
    localPos = (rotationY(iTime * (rnd.w - 0.5)) * vec4(localPos, 0.0)).xyz;
   
    
    // Compute polar coordinates on the sphere
    float lon = (atan(localPos.z, localPos.x)) + pi;  // 0.0 - 2 * pi
    float lat  = (atan(length(localPos.xz), localPos.y)) - halfPi; //-halfPi <-> halfPi
    
    // Compute the number of flowers at the equator according to the size of the planet
    float numAtEquator = floor(3.0 + p.radius * 15.0);
    float angle = pi2 / numAtEquator; // an the angle they cover ath the equator
    
    vec3 col1;
    vec3 col2;
    
    float petalAngle = rnd.w * 45.35 + iTime * 0.1;
    
    // Compute on layer of flower by dividing the sphere in horizontal bands of 'angle' height 
    float eq = (floor(lat / angle + 0.5)) * angle;
    vec2 uvs = ringUv(vec2(lon + eq * rnd.y * 45.0, lat), angle, eq);
    vec4 flPattern1 = flower((vec2(0.5) - uvs) * 0.95, rnd, 2.0, aaScale, petalAngle, col1, 0.8);
    
    
    // Compute a second layer of flowers with bands offset by half angle
    float eq2 = (floor(lat / angle) + 0.5) * angle;
    vec2 uvs2 = ringUv(vec2(lon + eq2 * rnd.x * 33.0, lat), angle, eq2);
    vec4 flPattern2 = flower((vec2(0.5) - uvs2) * 0.95, rnd2, 2.0, aaScale, petalAngle, col2, 0.8);
    

    // Compute flower with planar mapping on xz to cover the poles. 
    vec4 flPattern3 = flower(localPos.xz / p.radius, rnd2, 2.0, aaScale, petalAngle, col2, 0.8);
    
    float bg = (1.0 - nDotV);
    vec3 bgCol = rnd2.y > 0.5 ? col1 : col2; // sphere background is the color of one of the layers
    
    vec3 col = bgCol; 
    
    // mix the 3 layers of flowers together
    col = mix(col, flPattern1.rgb, flPattern1.a);
    col = mix(col, flPattern2.rgb, flPattern2.a);
    col = mix(col, flPattern3.rgb, flPattern3.a);
    
    // add some bogus colored shading
    
    //Front lighting
    //col *= mix(vec3(1.0), bgCol * 0.3, (bg * bg) * 0.8);

    return col;
}

// Function 252
float color_to_val_4(in vec3 color) {
    return color_to_val_1(color) + color_to_val_3(color);
}

// Function 253
vec3 RecolorForeground(vec3 color)
{
	if(color.g > (color.r + color.b)*GREEN_BIAS)
	{
		color.rgb = vec3(0.,0.,0.);
	}

	
	color.rgb = 0.2126*color.rrr + 0.7152*color.ggg + 0.0722*color.bbb;
	
	if(color.r > 0.95)
	{
		
	}
	else if(color .r > 0.75)
	{
		color.r *= 0.9;
	}
	else if(color.r > 0.5)
	{
		color.r *= 0.7;
		color.g *=0.9;
	}
	else if (color.r > 0.25)
	{
		color.r *=0.5;
		color.g *=0.75;
	}
	else
	{
		color.r *= 0.25;
		color.g *= 0.5;
	}
	
	
	return color;
}

// Function 254
vec3 UI_GetColor( int iData )
{
    return texelFetch( iChannelUI, ivec2(iData,0), 0 ).rgb;
}

// Function 255
vec4 oldColor(in vec2 fragCoord, in vec2 dxy) {    
    return texture(iChannel0,  (fragCoord.xy + dxy) / iResolution.xy);
}

// Function 256
vec3 SkyColor( vec3 rd )
{
#if 0
    // Cube Map
	// hide cracks in cube map
	rd -= sign(abs(rd.xyz)-abs(rd.yzx))*.01;

	//return mix( vec3(.2,.6,1), FogColour, abs(rd.y) );
	vec3 ldr = texture( iChannel0, rd ).rgb;
    
	// fake hdr
	vec3 hdr = 1.0/(1.2-ldr) - 1.0/1.2;
	
	return hdr;
#else
    // Black
    //return vec3(0,0,0);
    
    // test 1 UV
    //return rd * 0.5 + 0.5;
    
    return vec3(0,0.5,0);
    
    /*
    // plexus
    vec2 uv = vec2(atan(rd.z/rd.x), atan(length(rd.xz)/rd.y)) / M_PI;
    uv += vec2(10.);
    uv = abs(uv);
    vec2 suv = uv * 50.;
    
    vec2 id = floor(suv) - 0.5;
    vec2 fuv = fract(suv) - 0.5;
	vec2 pp =  getPoint(id, vec2(0));
    vec3 col = vec3(0);
    
    for(int x = -1; x <= 1; x++){
    	for(int y = -1; y <= 1; y++){
            vec2 pos = getPoint(id, vec2(x,y)); //sin(hash22(id + vec2(x,y)) * iTime * 1.5) * 0.4 + vec2(x,y);
            float d = 1.0 / pow(length(fuv - pos), 1.75) * 0.001;
            col += d;

            float len = smoothstep(1.0,0.5,length(pp - pos));
            col += smoothstep(0.025,0.001,sdSegment(fuv, pos, pp)) * len;
        }
    }
	
    return col * sinebow(uv.x);
	*/
    
    // fake unity default sky-box
	//vec3 ground = mix(vec3(0.25,0.4,0.8), vec3(0.2,0.15,0.15), saturate(abs(rd.y) * 25.0));
    //vec3 sky = mix(vec3(0.25,0.4,0.8), vec3(0.001, 0.15, 1.), saturate(abs(rd.y) * 10.0));
    //return rd.y < 0. ? ground :sky;
    
    /*
	// Starfield
    float x = atan(rd.z / rd.x);
    float y = acos(rd.y);
    return vec3(StableStarField(vec2(x,y) * 1000., 0.97 ));
	*/
    
    // wave z
    //vec2 uv = vec2(atan(rd.z/rd.x), atan(length(rd.xz)/rd.y)) / M_PI;
    //return vec3(saturate(uv),0);
    //return vec3(1,0,0) * saturate(sin(rd.z *100.- iTime * 10.));
    //return sinebow(rd.z * 5.- iTime * 3.) * 0.25;
    //return smoothstep(0.01,0.,mod(rd.z - iTime * 0.1, 0.1))*vec3(0.05,0.2,0.5);
#endif
}

// Function 257
vec3 get_color(int color){
    if(color == BLUE){
    	return vec3(0.149,0.141,0.912);
   	}
    else if(color == GREEN){
    	return vec3(0.000,0.833,0.224);
   	}
    else if(color == FOREST_GREEN){
    	return rgb(34.0,139.0,34.0);
   	}
    else if(color == WHITE){
    	return vec3(1.0,1.0,1.0);
   	}
    else if(color == GRAY){
    	return vec3(192.0,192.0,192.0)/255.0;
    }
    else if(color == YELLOW){
    	return vec3(1.0,1.0,0.0);
   	}
    else if(color == LIGHTBLUE){
    	return rgb(173.0,216.0,230.0);
   	}
    else if(color == SKYBLUE){
    	return rgb(135.0,206.0,235.0);
    }
    else if(color == SNOW){
    	return rgb(255.0,250.0,250.0);
    }
    else if(color == WHITESMOKE){
    	return rgb(245.0,245.0,245.0);
    }
    else if(color == LIGHTGRAY){
    	return rgb(211.0,211.0,211.0);
    }
    else if(color == LIME){
    	return rgb(0.0,255.0,0.0);
    }
    else if(color == LIGHTYELLOW){
    	return rgb(255.0,255.0,153.0);
    }
    else if(color == BEIGE){
    	return rgb(245.0,245.0,220.0);
    }
    else if(color == TAN){
    	return rgb(210.,180.,140.);
    }
}

// Function 258
vec3 getMetalColor(vec3 pos)
{
    return metalColor;
}

// Function 259
vec3 SceneColor( C_Ray ray )
{
    float fHitDist = TraceScene(ray);
	vec3 vHitPos = ray.vOrigin + ray.vDir * fHitDist;
	
	vec3 vResult = texture(iChannel0, vHitPos.xyz).rgb;	
	vResult = vResult * vResult;
	
	#ifdef FORCE_SHADOW
	if( abs(vHitPos.z) > 9.48)
	{
		if( abs(vHitPos.x) < 20.0)
		{
			float fIntensity = length(vResult);
			
			fIntensity = min(fIntensity, 0.05);
			
			vResult = normalize(vResult) * fIntensity;
		}
	}
	#endif	
	
	#ifdef ENABLE_REFLECTION
	if(vHitPos.y < -1.4)
	{
		float fDelta = -0.1;
		float vSampleDx = texture(iChannel0, vHitPos.xyz + vec3(fDelta, 0.0, 0.0)).r;	
		vSampleDx = vSampleDx * vSampleDx;

		float vSampleDy = texture(iChannel0, vHitPos.xyz + vec3(0.0, 0.0, fDelta)).r;	
		vSampleDy = vSampleDy * vSampleDy;
		
		vec3 vNormal = vec3(vResult.r - vSampleDx, 2.0, vResult.r - vSampleDy);
		vNormal = normalize(vNormal);
		
		vec3 vReflect = reflect(ray.vDir, vNormal);
		
		float fDot = clamp(dot(-ray.vDir, vNormal), 0.0, 1.0);
		
		float r0 = 0.1;
		float fSchlick =r0 + (1.0 - r0) * (pow(1.0 - fDot, 5.0));
		
		vec3 vResult2 = texture(iChannel1, vReflect).rgb;	
		vResult2 = vResult2 * vResult2;
		float shade = smoothstep(0.3, 0.0, vResult.r);
		vResult += shade * vResult2 * fSchlick * 5.0;
	}
	#endif
	
	if(iMouse.z > 0.0)
	{
		vec3 vGrid =  step(vec3(0.9), fract(vHitPos + 0.01));
		float fGrid = min(dot(vGrid, vec3(1.0)), 1.0);
		vResult = mix(vResult, vec3(0.0, 0.0, 1.0), fGrid);
	}
	
	return sqrt(vResult);    
}

// Function 260
vec3 getRoadColor(vec3 p){
    float lineFill = step(fract(p.x*.25) + .05, .1) * step(.5, fract(p.z*.2));
    float texture = step(fract(p.x*.25) + .05, .1) * step(.5, fract(p.z*.2));
    
    vec3 roadColor = vec3(noise2d(p.xz*4.));
           
    return mix(vec3(0.1, 0.1, 0.11) + roadColor*.05, vec3(1.), lineFill);
}

// Function 261
vec3 getColor(vec3 hit, vec3 rayDir, vec2 dists) {
    
    vec3 col = vec3(0.);
    
    // a surface was hit, do some shading
    if(dists.x < Accuracy) {
		vec3 norm = getNorm(hit);

        vec3 diffuse = vec3(1., .7, .5) * mapSky(norm);
        
        // angle of incidence
        float aoi = pow(1.-dot(norm, -rayDir), 1.);
        
        // ambient occlusion
        float ao = pow(getAO(hit, norm), 4.) * 2.;
        
        // initial color
        col = diffuse;
        
        // reflected sky
        vec3 ref = mapSky(normalize(reflect(rayDir, norm)));
		
        // mix in reflections
        col = mix(col, ref, aoi);
        
        // apply ao
        col *= ao;
        
        // mix sky into color (fog effect)
        col = mix(col, mapSky(rayDir), pow(dists.y/MaxDist, 2.));
        
    } else {
        // return sky only, for there's nothing else
        col = mapSky(rayDir);
    }
    
    return col;
}

// Function 262
vec4 drawColorRing(in vec2 uv){
    float UV = floor(1.0-uv.y );
    float rad = UV * 2.0 * PI+atan(uv.y,uv.x);
    float degree = rad/(2.0*PI)*360.0;
    return vec4(hsv2rgb(vec3(degree,1.0,1.0)),0.0);
}

// Function 263
vec3 sky_color(vec3 ray)
{
	vec3 rc = texture(iChannel2, ray).rrr;
    for (int l=0; l<3; l++)
        rc+= 1.5*normalize(lamps[l].color)*lamps[l].intensity*specint*pow(max(0.0, dot(ray, normalize(lamps[l].position - campos))), 200.);
    return rc;
}

// Function 264
vec4 gen_color(int iter)
{
    vec3 c1 = vec3(1.0,1.0,1.0);
    vec3 c3 = vec3(0.0,0.3,0.6);
    vec3 c2 = vec3(0.0,0.6,0.3);
    vec3 m = vec3(float(iter)/float(MAXITER));
    vec3 base = mix(c1,mix(c2,c3,m),m);
    return vec4(base,1.0);
}

// Function 265
vec3 GetSunLightColor()
{
    return 0.9 * vec3(0.9, 0.75, 0.7);
}

// Function 266
vec3 skyColor(vec3 nvDir) {
    float yy = clamp(nvDir.y+0.1, 0.0, 1.0);
    float horiz0 = pow(1.0 - yy, 30.0);
    float horiz1 = pow(1.0 - yy, 5.0);
    
    vec3 sv = nvDir - vec3(0.0, -1.0, 0.0);
    vec2 uvCloud = 0.25*(sv.xz / sv.y) + vec2(0.5);
    vec2 skyTexVal = skyTex(uvCloud);

    float cloudIntensity = pow(skyTexVal.x, 2.0);
    float starIntensity = pow(skyTexVal.y, 2.0);

    vec3 c = vec3(0.0);
    c = mix(c, vec3(0.2, 0.0, 0.5), horiz1);
    c = mix(c, vec3(1.0), horiz0);
    c = mix(c, vec3(0.45, 0.5, 0.48), (1.0-horiz0)*cloudIntensity);
    c = mix(c, vec3(1.0), (1.0-horiz1)*starIntensity);
    return c;
}

// Function 267
vec3 volumeToColor(vec2 volume)
{
    if( volume.x != 0.0 )
    {
        return mix(color_outer, color_inner, min(1.0,volume.x));
    }
    return mix(color_bg, color_outer, min(1.0,volume.y));
}

// Function 268
vec3 getColor(vec3 norm, vec3 pos, int objnr, vec3 ray)
{
   #ifdef always_cut
      return objnr==C1_OBJ?vec3(0.4, 0.43, 0.6):(
             objnr==C2_OBJ?vec3(0.4, 0.6, 0.4):(
             objnr==CC_OBJ?vec3(0.6, 0.5, 0.4):(
             objnr==WHEEL_OBJ?vec3(0.6, 0.4, 0.5):
                     vec3(0.5))));
   #else
    if (iMouse.z>0.)
      return objnr==C1_OBJ?vec3(0.4, 0.43, 0.6):(
             objnr==C2_OBJ?vec3(0.4, 0.6, 0.4):(
             objnr==CC_OBJ?vec3(0.6, 0.5, 0.4):(
             objnr==WHEEL_OBJ?vec3(0.6, 0.4, 0.5):
                     vec3(0.5))));
   else
      return vec3(0.5);
   #endif
}

// Function 269
vec3 fire_color(float x)
{
	return
        // red
        vec3(1., 0., 0.) * x
        // yellow
        + vec3(1., 1., 0.) * clamp(x - .5, 0., 1.)
        // white
        + vec3(1., 1., 1.) * clamp(x - .7, 0., 1.);
}

// Function 270
void mixColorPoint(vec2 uv,inout vec3 col,vec2 colPoint,float scale)
{
    col = mix(
        col , 
        hash3point(colPoint) ,
        1.0 - smoothstep(0.0,1.0,sqrt(sqrt( length(uv - colPoint)* scale )))
    );
}

// Function 271
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

// Function 272
vec3 color(vec3 position, vec3 normal, vec3 dir, HitInfo hitInfo) {
	vec3 col = vec3(0.0);
	
    vec3 alb = albedo[hitInfo.mat];

    vec3 diff = vec3(0.0);

    for (int i = 0; i < 2; i++)
    {
        Light light = lights[i];

        float shad = shadow(position, lightDir(position, light), normal, distance(position, light.pos));
        float lamb = lambert(normal, light.pos);
        float fres = fresnel(dir, normal);

        diff += ((lightIntensity(position, light) * (lamb + fres)) * shad + light.amb) * (alb * light.col);
    }

    float a1 = 0.3 * ao(position, normal, 0.001);
    float a2 = 0.3 * ao(position, normal, 0.01);
    float a3 = 0.2 * ao(position, normal, 0.04);
    float a4 = 0.2 * ao(position, normal, 0.2);

    float aa = a1 + a2 + a3 + a4;

    col = diff * aa;

	return col;
}

// Function 273
vec3 ColorFetch(vec2 coord)
{
 	return texture(iChannel0, coord).rgb;   
}

// Function 274
vec3 getFaceColor(vec3 uv) {
    
    int regionCode = drawCodes.x;
    
    if (regionCode == 1) {
        return colors[0];
    } else if (regionCode == 2) {
        return colors[1];
    } else if (regionCode == 4) {
        return colors[2];
    } else if (regionCode == 3) {
        return decide2(uv, 0, 1);
    } else if (regionCode == 5) {
        return decide2(uv, 0, 2);
    } else if (regionCode == 6) { 
        return decide2(uv, 1, 2);
    } else {
        return decide3(uv);
    }
    
}

// Function 275
vec3 GetColor(in vec3 ro, in vec3 rd, in float t, float m)
{
       vec3 pos = ro + t*rd;
        vec3 nor = calcNormal( pos );

		//col = vec3(0.6) + 0.4*sin( vec3(0.05,0.08,0.10)*(m-1.0) );
		vec3 col = vec3(0.6) + 0.4*sin( vec3(0.05,0.08,0.10)*(m-1.0) );
		
        float ao = calcAO( pos, nor );

		vec3 lig = normalize( vec3(-0.6, 0.7, -0.5) );
		float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
        float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
        float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);

		float sh = 1.0;
		if( dif>0.02 ) { sh = softshadow( pos, lig, 0.02, 10.0, 7.0 ); dif *= sh; }

		vec3 brdf = vec3(0.0);
		brdf += 0.20*amb*vec3(0.10,0.11,0.13)*ao;
        brdf += 0.20*bac*vec3(0.15,0.15,0.15)*ao;
        brdf += 1.20*dif*vec3(1.00,0.90,0.70);

		float pp = clamp( dot( reflect(rd,nor), lig ), 0.0, 1.0 );
		float spe = sh*pow(pp,16.0);
		float fre = ao*pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );

		return col*brdf + vec3(1.0)*col*spe + 0.2*fre*(0.5+0.5*col);	
}

// Function 276
vec3 GetAlphaColor(in vec3 pos)
{
	vec3 ret = vec3(0.0);
	float angle = atan(pos.z/pos.x);
	
#ifdef DRAW_RED_MAGIC
	ret += vec3(1.0,0.0,0.0)*GetTrail(pos,angle,vec3(0.0,1.1, 0.0),vec3(8.0,0.1,0.5),vec2(4.0,0.0));
#endif
#ifdef DRAW_GREEN_MAGIC
	ret += vec3(0.0,1.0,0.0)*GetTrail(pos,angle,vec3(0.5,2.1, 0.5),vec3(8.0,0.1,0.8),vec2(2.0,1.0));
#endif
#ifdef DRAW_BLUE_MAGIC
	ret += vec3(0.0,0.0,1.0)*GetTrail(pos,angle,vec3(0.9,3.7,-1.0),vec3(8.0,0.1,0.2),vec2(6.0,2.0));
#endif
#ifdef DRAW_YELLOW_MAGIC
	ret += vec3(1.0,1.0,0.0)*GetTrail(pos,angle,vec3(0.1,1.7,0.5),vec3(8.0,0.1,1.0),vec2(2.0,9.0));	
#endif
	return ret;
}

// Function 277
vec3 obj_color(vec3 norm, vec3 pos)
{
  	#ifdef only_shape
  	return vec3(0.35, 0.7, 1.0);
  	#else
    return vec3(0.);
    #endif
}

// Function 278
vec3 getColor( in vec2 fragCoord, in vec3 dir, in float theta ) {
    
    vec3 rnd = hash33(vec3(fragCoord, iFrame));
    
    // super-sample the background
    vec3 backcolor = vec3(0);
    for (int i = 0 ; i < BACKGROUND_SS ; i++) {
        vec3 rndSamp = hash33(vec3(fragCoord, iFrame*BACKGROUND_SS+i));
        vec3 dirTan = normalize(cross(dir, vec3(0, 1, PI)));
        vec3 dirCoTan = cross(dir, dirTan);
        float rot = 2.0*PI*rndSamp.x;
        float the = acos(1.0 - rndSamp.y*(1.0 - cos(theta)));
        float sinThe = sin(the);
        vec3 backDir = dirTan*sinThe*cos(rot) + dirCoTan*sinThe*sin(rot) + dir*cos(the);
        backcolor += background(backDir);
    }
    backcolor /= float(BACKGROUND_SS);
    
    // sine of the pixel angle
    float sinPix = sin(theta);
    // accumulate color front to back
    vec4 acc = vec4(0, 0, 0, 1);
    
    float totdist = 0.0;
    vec3 dummy = vec3(0);
    totdist += rnd.z*de(camPos, 0.0, dummy);
    
	for (int i = 0 ; i < RAY_STEPS ; i++) {
		vec3 p = camPos + totdist * dir;
        vec3 color = backcolor;
        float dist = de(p, totdist, color);
        
        color = applyFog(color, totdist+dist, camPos, dir);
        
        // cone trace the surface
        float prox = dist / (totdist*sinPix);
        float alpha = clamp(prox * -0.5 + 0.5, 0.0, 1.0);
        
        if (alpha > 0.01) {
            // accumulate color
            acc.rgb += acc.a * (alpha*color.rgb);
            acc.a *= (1.0 - alpha);
        }
        
        // hit a surface, stop
        if (acc.a < 0.01) {
            break;
        }
        
        // continue forward
        totdist += max(abs(dist*0.85), 0.001);
	}
    
    vec3 result = vec3(0);
    
    // add background
    result = acc.a*FOG_COLOR + acc.rgb;
    // dithering
    result += rnd * 0.02 - 0.01;
    // gamma correction
    result = pow( result, vec3(1.0/2.2) );
    
    return result;
    
}

// Function 279
vec3 color(vec2 n){
    #ifdef STARS
	vec3 p=vec3(smoothstep(0.7,1.,texture(iChannel1,n).r));
    #else
    vec3 p=texture(iChannel0,n).rgb;
    #endif
    #ifdef BRIGHT_SPOTS
    
    p=BriSp(p);
    #endif
    return p;
}

// Function 280
vec3 skyColor(vec3 d)
{
    return vec3(0.01);
}

// Function 281
vec4 GetSampleColor(ivec2 currentCoord, ivec2 samplePosition, float sampleResolution)
{
    ivec2 sampleOffset = currentCoord - samplePosition;
    ivec2 sampleCoord = ivec2(floor(vec2(sampleOffset) / sampleResolution));
    vec4 sampleColor = texture(iChannel0, vec2(samplePosition + sampleCoord) / iResolution.xy);
    return sampleColor;
}

// Function 282
vec3 colorByDistance(float dst, float falloff, vec3 color, vec3 oldColor)
{
  return mix(color, oldColor, smoothstep(0.0, falloff, dst));
}

// Function 283
vec3 skyColor(vec3 ro, vec3 rd) {
  const vec3 sunDir = normalize(lightPos);
  float sunDot = max(dot(rd, sunDir), 0.0);  
  vec3 final = vec3(0.);

  final += mix(skyCol1, skyCol2, rd.y);
  final += 0.5*sunCol*pow(sunDot, 20.0);
  final += 4.0*sunCol*pow(sunDot, 400.0);    

  float tp  = rayPlane(ro, rd, vec4(vec3(0.0, 1.0, 0.0), 0.505));
  if (tp > 0.0) {
    vec3 pos  = ro + tp*rd;
    vec3 ld   = normalize(lightPos - pos);
    float ts4 = RAYSHAPE(pos, ld);
    vec3 spos = pos + ld*ts4;
    float its4= IRAYSHAPE(spos, ld);
    // Extremely fake soft shadows
    float sha = ts4 == miss ? 1.0 : (1.0-1.0*tanh_approx(its4*1.5/(0.5+.5*ts4)));
    vec3 nor  = vec3(0.0, 1.0, 0.0);
    vec3 icol = 1.5*skyCol1 + 4.0*sunCol*sha*dot(-rd, nor);
    vec2 ppos = pos.xz*0.75;
    ppos = fract(ppos+0.5)-0.5;
    float pd  = min(abs(ppos.x), abs(ppos.y));
    vec3  pcol= mix(vec3(0.4), vec3(0.3), exp(-60.0*pd));

    vec3 col  = icol*pcol;
    col = clamp(col, 0.0, 1.25);
    float f   = exp(-10.0*(max(tp-10.0, 0.0) / 100.0));
    return mix(final, col , f);
  } else{
    return final;
  }
}

// Function 284
vec3 colorsampler(vec3 src, vec3 col)
{
    vec3 delta = src - col;
    if(dot(delta,delta)<=0.1)
    {
        return vec3(0.0,1.0,0.0);
    }
    else
    {
        return src;
    }
}

// Function 285
vec3 getBaseColor()
{
    float colorPerSecond = 0.5;
    int i = int(mod(colorPerSecond * iTime, 7.));
    int j = int(mod(float(i) + 1., 7.));
 
    return mix(getBaseColor(i), getBaseColor(j), fract(colorPerSecond * iTime));
}

// Function 286
vec3 GetSpaceColor(vec3 worldPosition, vec3 rayDirection, vec3 lightPosition)
{
    // main green light
    vec3 directionToMainLightPosition = normalize(lightPosition - worldPosition);
    float mainLightAmount = max( dot( rayDirection, directionToMainLightPosition), 0.0 );

    // small 2 planets - still green light
    vec3 directionToSmallPlanet1 = normalize(vec3(lightPosition.x - 150.0,lightPosition.y,
        lightPosition.z + 80.0) - worldPosition);
    float lightStarAmount1 = max( dot( rayDirection, directionToSmallPlanet1), 0.0 );

    // movement of second planet
    float radius = 250.0f;
    float angle = iTime / 8.0f;
    vec3 directionToSmallPlanet2 = normalize(vec3(lightPosition.x + cos(angle) * radius,lightPosition.y, lightPosition.z + sin(angle) * radius) - worldPosition);
    float lightStarAmount2 = max( dot( rayDirection, directionToSmallPlanet2), 0.0 );

    // horizon color blending
    float v = pow(1.0-max(rayDirection.y,0.0),2.);
    vec3 sky = mix(vec3(.0,0.01,.04), _FarHorizontColor.rgb, v);

    // small 2 planets color
    sky = sky + _MainLightSourceColor.rgb * 0.75f * min(pow(lightStarAmount1, 6000.0), .75); 
    sky = sky + _MainLightSourceColor.rgb * 0.55f * min(pow(lightStarAmount2, 80000.0), .5); 

    // main light source
    sky = sky + _MainLightSourceColor.rgb * mainLightAmount * mainLightAmount * .25;
    sky = sky + _MainLightSourceColor.rgb * min(pow(mainLightAmount, 900.0), .7); 
    
    return clamp(sky, 0.0, 1.0);
}

// Function 287
vec3 MultiplyColorChannel(vec3 color, float redFactor, float greenFactor, float blueFactor)
{
 	return vec3(color.r * redFactor, color.g * greenFactor, color.b * blueFactor);   
}

// Function 288
vec4 sceneColor(vec2 uv)
{
    vec4 outColor = vec4(FXAA(uv, iChannel1, 1.0/iResolution.xy), 1.0);
    
    return outColor;
}

// Function 289
void accumColor(out vec3 acc_c, in vec3 new_c, in float aa_factor)
{
    if(acc_c == vec3(1.0,2.0,3.0))
    {
        acc_c = new_c;
    }
    else
    {
#if SHOW_AA_FACTOR
        acc_c = vec3(aa_factor);
#else
        acc_c = mix(acc_c,new_c,aa_factor);
#endif
    }
}

// Function 290
void print_color( int id, vec3 v){    vec2 puv = uv-vec2(.5);    vec3 select = widgetSelected();    float sl2 = SLIDER_LENGTH/2.;    vec4 color = uiColor(id);    if(color.a == 0.)        color.rgb = v;        bool selected = ( select.r == .2 && select.g*255. == float(id) );    bool mouseAndNoSelect = iMouse.w>.5 && roundBox( iMouse.xy-pos-vec2(sl2,6.), vec2(sl2,3.), 5.) < 0. && select.r == 0.;         if(mouseAndNoSelect || selected)    	color.rgb = hsv2rgb( vec3( (iMouse.x-pos.x)/(SLIDER_LENGTH*.9),1.,1.) );    float d = roundBox( uv-pos-vec2(sl2,6.), vec2(sl2,3.), 5.);    float layer = clamp(sign(-d),0.,1.);    col.rgb += vec3( layer*color*max(.0,sign(uv.x-pos.x-SLIDER_LENGTH*.9)));    col.rgb += WIDGET_COLOR*vec3( clamp( 1.-abs(d)*.75 , 0., 1.) );    col.a += layer + clamp( 1.-abs(d) , 0., 1.);        if((mouseAndNoSelect || selected) && uv.x-pos.x-SLIDER_LENGTH*.9<0.)        col.rgb += layer*hsv2rgb( vec3( (uv.x-pos.x)/(SLIDER_LENGTH*.9),1.,1.) );            if(puv.x == float(id) && puv.y==1.)        col = vec4(color.rgb,1.);        if(puv.x == 0. && puv.y == 2.)    {        if(iMouse.w<.5)            col = vec4(0.);        else if(mouseAndNoSelect)        	col = vec4(.2,float(id)/255.,0.,0.);    }}

// Function 291
vec3 IdxtoColor(float idx)
{
    vec3 shift = exp2(vec3(BITS_RGB.g+BITS_RGB.b, BITS_RGB.b, 0));
    vec3 c = vec3(floor(idx));
    c = floor(c / shift);
    c = mod(c, MAX_RGB);
    return c;
}

// Function 292
vec3 getSkyColor(vec3 ray)
{ 
   return vec3(0.);
}

// Function 293
vec3 matcolor(vec3 pos, float time, float scene){
    
    vec3 ldir = normalize(sin(vec3(1,2,3)+iTime));
    vec3 nor = nf(pos,time, scene);
    vec3 m = mat_id(pos);
    float hue = fract(dot(m,vec3(1.2,17.3,143.8)))*-4.0 + gl_FragCoord.x/iResolution.x + scene*7.0;
    float var = dot(m,vec3(3.2,13.3,171.8));
    float value = sin(var)*.5+1.6+dot(ldir,nor);
    value*=1.2;
    vec3 c = naturalColor(hue, value);
    return c;
}

// Function 294
vec3 getSceneColor(vec2 uv)
{
    return texture(iChannel0, uv).rgb;
}

// Function 295
vec3 getColor(float m, vec3 p, vec3 n) {
    vec3 h = vec3(.5);      
    if(m==1.) {
        h = .6-vec3(hsh*.75);
    }
    if(m==2.) {    
        // strip patterns..
        thp/=1./SCALE;
        float dir = mod(tip.x + tip.y,2.) * 2. - 1.;  

        vec2 cUv = thp.xy-sign(thp.x+thp.y+.001)*.5;
        float angle = atan(cUv.x, cUv.y);
        float a = sin( dir * angle * 6. + iTime * 2.25);
        a = abs(a)-.45;a = abs(a)-.35;
        vec3 nz = hue((p.x+(T*.12))*.25);
        h = mix(nz, vec3(1), smoothstep(.01, .02, a));  
    }
    
    return h;
}

// Function 296
float color_to_val_cmy(in vec3 color) {
    vec3 pair_mins = min(color.rgb, color.gbr);
    float second = max(pair_mins.r, max(pair_mins.g, pair_mins.b));
    float last = min(pair_mins.r, min(pair_mins.g, pair_mins.b));
    return second - last;
}

// Function 297
vec3 getDogColor(vec2 u, float h)
{
  vec3 dogLightBaseColor = mix(vec3(0.91, 0.84, 0.95),
                           vec3(0.75, 0.67, 0.79),
                           lib_gn_lookup[2]);   
  h *= h;       
  h = min(1.0, h * 3.0);
 
  dogLightBaseColor += vec3(u.y * u.y * lib_gn_lookup[3]) * 0.1 * vec3(1.0, u.y, 0.5);
  
  h = mix(h, globalHair, 0.3 * lib_gn_lookup[6]);
  
  return dogLightBaseColor * h * vec3(1.0, 0.85, 0.8);
}

// Function 298
vec4 Env_GetSkyColor( const vec3 vViewPos, const vec3 vViewDir )
{
	vec4 vResult = vec4( 0.0, 0.0, 0.0, kFarDist );

#if 1
    vec3 vEnvMap = textureLod( iChannel1, vViewDir.zyx, 0.0 ).rgb;
    vResult.rgb = vEnvMap;
#endif    
    
#if 0
    vec3 vEnvMap = textureLod( iChannel1, vViewDir.zyx, 0.0 ).rgb;
    vEnvMap = vEnvMap * vEnvMap;
    float kEnvmapExposure = 0.999;
    vResult.rgb = -log2(1.0 - vEnvMap * kEnvmapExposure);

#endif
    
    // Sun
    //float NdotV = dot( g_vSunDir, vViewDir );
    //vResult.rgb += smoothstep( cos(radians(.7)), cos(radians(.5)), NdotV ) * g_vSunColor * 5000.0;

    return vResult;	
}

// Function 299
vec3 doColor( in vec3 sp, in vec3 rd, in vec3 sn, in vec3 lp, vec3 obj) {
	vec3 sceneCol = vec3(0.0);
    
    vec3 ld = lp- sp; 
    float lDist = max(length(ld / 2.), 0.001); 
    ld /= lDist;

    float atten = 1.0 / (1.0 + lDist * 0.025 + lDist * lDist * 0.02);
    float diff = max(dot(sn, ld), .1);
    float spec = pow(max(dot(reflect(-ld, sn), -rd), 1.), 2.0);

    //vec3 objCol = getObjectColor(sp, sn, obj);
    sceneCol += (vec3(1.) * (diff + 1.2) * spec * 2.6) * atten;
   
    return sceneCol;
}

// Function 300
vec3 getParticleColor(in vec2 p) {
    return normalize(vec3(0.1) + texture(iChannel2, p * 0.0001 + iTime * 0.005).rgb);
}

// Function 301
vec4 color_temp(float temp)
{
     if (temp <= orange_temp)
    {
     return interpolate(RED, ORANGE, red_temp, orange_temp, temp); 
    }
    else if (temp <= yellow_temp)
    {
     return interpolate(ORANGE, YELLOW, orange_temp, yellow_temp, temp);  
    }
    else if (temp <= white_temp)
    {
     return interpolate(YELLOW, WHITE, yellow_temp, white_temp, temp);  
    }
    else if (temp <= cyan_temp)
    {
     return interpolate(WHITE, CYAN, white_temp, cyan_temp, temp);  
    }
    else return WHITE;
}

// Function 302
vec3 getEnergyColor (in float iEnergy01)
{
    //return vec3(iEnergy01 > 0.0, 0, 0);
    return vec3(.1 + .9 * SATURATE(iEnergy01), 0, 0);
	//return vec3(/*.15 + .85 * SATURATE(iEnergy01)*/ iEnergy01 == 0.0 ? 0.15 : 1.0, 0, 0);
}

// Function 303
vec4 colormap(float x) {
	if (x < 0.0) {
		return vec4(0.0, 0.0, 0.0, 1.0);
	} else if (1.0 < x) {
		return vec4(0.0, 0.0, 0.0, 1.0);
	} else {
		float h = clamp(-9.42274071356572E-01 * x + 8.74326827903982E-01, 0.0, 1.0);
		float s = 1.0;
		float v = clamp(4.90125513855204E+00 * x + 9.18879034690780E-03, 0.0, 1.0);
		return colormap_hsv2rgb(h, s, v);
	}
}

// Function 304
void zxspectrum_colors( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 pv = floor(fragCoord.xy/LOWREZ);
    vec2 sv = floor(iResolution.xy/LOWREZ);
                
    vec4 cs=smap(texture(iChannel0,pv/sv).rgb);
    
    if( mod(pv.x+pv.y,2.0)==1.0)
    {
		fragColor = vec4(fmap(vec4(floor(cs.rgb+vec3(0.5+(DITHER*0.3))),cs.a)),1.0);
    }
    else
    {  
		fragColor = vec4(fmap(vec4(floor(cs.rgb+vec3(0.5-(DITHER*0.3))),cs.a)),1.0);
    }

}

// Function 305
vec3 color(sampler2D tex, vec2 uv){        
    vec3 color = texture(iChannel0,uv).rgb;
    #ifdef COLOR
    float bw = (color.r + color.g + color.b) / 3.0;
    color = mix(color,vec3(bw,bw,bw),.95);
    float p = 1.5;
    color.r = pow(color.r,p);
    color.g = pow(color.g,p-0.1);
    color.b = pow(color.b,p);
    #endif
    return color;
}

// Function 306
vec3 matColor( const in float mat ) {
	vec3 nor = vec3(0., 0.95, 0.);
	
	if( mat<3.5 ) nor = REDCOLOR;
    if( mat<2.5 ) nor = GREENCOLOR;
	if( mat<1.5 ) nor = WHITECOLOR;
	if( mat<0.5 ) nor = LIGHTCOLOR;
					  
    return nor;					  
}

// Function 307
PathColor PathColor_One()
{
#if SPECTRAL    
    return PathColor( 1.0 );
#endif    
#if RGB
	return PathColor( vec3(1) );
#endif    
}

// Function 308
float normalizeColorChannel(in float value, in float min, in float max) {
    return (value - min)/(max-min);
}

// Function 309
void UI_StoreDataColor( inout UIContext uiContext, UIData_Color dataColor, int iData )
{
    vec4 vData0 = vec4(0);
    vData0.rgb = hsv2rgb( dataColor.vHSV );
        
    StoreVec4( ivec2(iData,0), vData0, uiContext.vOutData, ivec2(uiContext.vFragCoord) );            

    vec4 vData1 = vec4(0);
    vData1.rgb = dataColor.vHSV;
        
    StoreVec4( ivec2(iData,1), vData1, uiContext.vOutData, ivec2(uiContext.vFragCoord) );            
}

// Function 310
vec3 getColor(RayHit r){
    vec3 col = vec3(1.)*r.hit.color;
    vec3 p = r.hit.p;
    /**
    switch(r.hit.id)
    {
        case 0: 
			col = vec3(0.,0.7,0.9); // sky
       		break;
        case 1: 
			col *= vec3(.2,0.12,0.1);
            //col *= (1.+((lad(p.xz*.5)))*.2);
            col *= (1.+((voronoi(p.xz*3.5)))*.5);
       		break;
        case 2: 
			col *= vec3(.9,0.9,0.6);
            col *= 1.-(brick(p*4.)); 
       		break;
      	case 3: 
			col *= vec3(.3,0.9,1.1);
            col *= (1.+((lad(p.xz*.005)))*.3);
           // col *= (1.+((voronoi(p.xz*3.5)))*.5);
       		break;
    }
    /**/
    
	if (r.hit.id == 0)
    {
        col = vec3(0.,0.7,0.9); // sky
    } 
    else if (r.hit.id == 1)
    {
        col *= vec3(.2,0.12,0.1);
        //col *= (1.+((lad(p.xz*.5)))*.2);
        col *= (1.+((voronoi(p.xz*3.5)))*.5);
    }
	else if (r.hit.id == 2)
    {
        col *= vec3(.9,0.9,0.6);
        col *= 1.-(brick(p*4.)); 
    }
	else if (r.hit.id == 3)
    {
        col *= vec3(.3,0.9,1.1);
        col *= (1.+((lad(p.xz*.005)))*.3);
        // col *= (1.+((voronoi(p.xz*3.5)))*.5);
    }
    
    
    return col;
}

// Function 311
vec3 getBackgroundColor( in vec3 dir ) {
    float horizon = 1.0 - smoothstep(0.0, 0.02, dir.z);
    // this is the background with the scanline
    vec3 color = getSpaceColor(dir);
    // get the sun
    float inside = 0.0;
    vec4 sun = getSunColor(dir, inside);
    color = mix(color, vec3(0.1, 0.16, 0.26), inside);
    color = mix(color, sun.rgb, sun.a);
    // the horizon
    color = mix(color, vec3(0.43, 0.77, 0.85), horizon * (1.0 - sun.a * 0.19));
    return color;
}

// Function 312
vec3 linColor(float value)
{
    value = mod(value * 6.0, 6.0);
    vec3 color;
    
    color.r = 1.0 - clamp(value - 1.0, 0.0, 1.0) + clamp(value - 4.0, 0.0, 1.0);
    color.g = clamp(value, 0.0, 1.0) - clamp(value - 3.0, 0.0, 1.0);
    color.b = clamp(value - 2.0, 0.0, 1.0) - clamp(value - 5.0, 0.0, 1.0);
    
    return color;
}

// Function 313
vec3 colorGradingProcess(const in ColorGradingPreset p, in vec3 c){
  float originalBrightness = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = mix(c, c * colorTemperatureToRGB(p.colorTemperature), p.colorTemperatureStrength);
  float newBrightness = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c *= mix(1.0, (newBrightness > 1e-6) ? (originalBrightness / newBrightness) : 1.0, p.colorTemperatureBrightnessNormalization);
  c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, p.presaturation);
  return pow((p.gain * 2.0) * (c + (((p.lift * 2.0) - vec3(1.0)) * (vec3(1.0) - c))), vec3(0.5) / p.gamma);
}

// Function 314
vec3 clipcolor( vec3 c) {
                  float l = lum(c);
                  float n = min(min(c.r, c.g), c.b);
                  float x = max(max(c.r, c.g), c.b);
                
                 if (n < 0.0) {
                     c.r = l + ((c.r - l) * l) / (l - n);
                     c.g = l + ((c.g - l) * l) / (l - n);
                     c.b = l + ((c.b - l) * l) / (l - n);
                 }
                 if (x > 1.25) {
                     c.r = l + ((c.r - l) * (1.0 - l)) / (x - l);
                     c.g = l + ((c.g - l) * (1.0 - l)) / (x - l);
                     c.b = l + ((c.b - l) * (1.0 - l)) / (x - l);
                 }
                 return c;
             }

// Function 315
float encodeColor(vec4 a){
	
    // convert from [0, 1] to [0, color_maxValues]
    // pre-multiply 0.5 to a to avoid overflow
    a = floor(a * (0.5 * color_maxValues) + 0.5);
    
    // bit shift then OR
	return dot( a, color_positions );
}

// Function 316
vec4 GetTextWindowColor(ivec2 currentCoord, ivec2 textColRowIndex, ivec2 characterColRowIndex)
{
    int characterIndex = int(texelFetch(iChannel0, ivec2(1, textColRowIndex.y), 0).a) + textColRowIndex.x;
    uvec4 character = characterSet[int(texelFetch(iChannel0, ivec2(2, characterIndex), 0).a)];
    characterColRowIndex = ivec2(7, 11) - characterColRowIndex;
    
    int bitIndex = characterColRowIndex.x + characterColRowIndex.y * 8;
    int bracketIndex = bitIndex / 24;
    float characterWeight = GetBit(character[3 - bracketIndex], bitIndex - bracketIndex * 24);
    return mix(TEXT_BACKGROUND_COLOR, TEXT_COLOR, characterWeight);
}

// Function 317
vec3 sky_color(vec3 ray)
{ 
    return 0.1 + 1.2*texture(iChannel0, ray).rgb;
}

// Function 318
vec3 colormap(float t) {
    return .5 + .5*cos(TWOPI*( t + vec3(0.0,0.1,0.2) ));
}

// Function 319
void brightnessAdjust( inout vec4 color, in float b) {
    color.rgb += b;
}

// Function 320
vec3 objColor( vec3 hitPos, vec3 surfaceNormal )
{
    vec3 lightPos = vec3( 1. , 1. , 1. );

    // Get direction of light relative to the hit
    vec3 lightDir = lightPos - hitPos;
    lightDir = normalize( lightDir );
    
    // The dot product of the lightDir to the surfaceNormal of the hit
    // Gives us how much that hit is facing the light source
    // -1: faces opposite
    // 1: faces completely the light
    float faceValue = dot( lightDir , surfaceNormal );
    faceValue = max( 0.3 , faceValue );
    
    float b = map( hitPos.x, -1., 1., 0., .8 );
    vec3 color = vec3( .18 , .18 , b );
   	
    // here's how we do some shadowing !
    color *= faceValue;
	return color;
}

// Function 321
vec3 resampleColor(Bounce[WAVELENGTHS] bounces) {
    vec3 col = vec3(0.0);
    
    for (int i = 0; i < WAVELENGTHS; i++) {        
        float reflectance = bounces[i].reflectance;
        float index = float(i) / float(WAVELENGTHS - 1);
        float texCubeIntensity = filmic_gamma_inverse(
            clamp(bounces[i].attenuation * sampleCubeMap(index, bounces[i].ray_direction), 0.0, 0.99)
        );
    	float intensity = texCubeIntensity + reflectance;
        col += sampleWeights(index) * intensity;
    }

    return 1.4 * filmic_gamma(3.0 * col / float(WAVELENGTHS));
}

// Function 322
vec3 rods_color(vec3 pos)
{
    return vec3(0.55);
}

// Function 323
vec3 color(float grad) {
    
    float m2 = iMouse.z < 0.0001 ? 1.15 : iMouse.y * 3.0 / iResolution.y;
    grad =sqrt( grad);
    vec3 color = vec3(1.0 / (pow(vec3(0.5, 0.0, .1) + 2.61, vec3(2.0))));
    vec3 color2 = color;
    color = ramp(grad);
    color /= (m2 + max(vec3(0), color));
    
    return color;

}

// Function 324
vec4 flagColor(float position) {
    // Invert position for GLSL
    position = 1.0 - position;

    float step1 = 0.0;
    float step2 = 0.08333;
    float step3 = 0.1667;
    float step4 = 0.25;
    float step5 = 0.3333;
    float step6 = 0.4167;
    float step7 = 0.5;
    float step8 = 0.5833;
    float step9 = 0.6667;
    float step10 = 0.75;
    float step11 = 0.8333;
    float step12 = 0.9167;
    
    vec4 result = COLOR1;
    result = mix(result, COLOR2, step(step2, position));
    result = mix(result, COLOR3, step(step3, position));
    result = mix(result, COLOR4, step(step4, position));
    result = mix(result, COLOR5, step(step5, position));
    result = mix(result, COLOR6, step(step6, position));
    result = mix(result, COLOR7, step(step7, position));
    result = mix(result, COLOR8, step(step8, position));
    result = mix(result, COLOR9, step(step9, position));
    result = mix(result, COLOR10, step(step10, position));
    result = mix(result, COLOR11, step(step11, position));
    result = mix(result, COLOR12, step(step12, position));

    return result;
}

// Function 325
vec3 framework_color(vec3 pos,vec3 norm)
{
    #ifdef color_changes
    vec3 fc = vec3(0.58 + 0.03*sin(pos.z/687.), 
                   0.41 + 0.09*sin(pos.z/537.), 
                   0.12 + 0.07*sin(pos.z/856.));
    #else
    vec3 fc = vec3(0.58, 0.41, 0.12);
    #endif
    vec2 tpos = vec2(dot(pos.yx, norm.xy) + pos.z, dot(pos.yz, norm.zy) - 1.5*pos.z + 0.2);
    vec3 mc = texture(iChannel0, 0.1*tpos).rgb;
    float mc2 = texture(iChannel1, 0.01*tpos).r;
    vec3 col1 = mix(mix(mc, fc*mc, 0.5), fc, 0.4);
    col1 = mix(col1, vec3(0.05), smoothstep(0.65, 1., mc2));
    col1 = mix(col1, vec3(0.42, 0.19, 0.13), smoothstep(0.55, 0., mc2));
        
    return col1;
}

// Function 326
vec3 GetSunColorReflection(vec3 rayDir, vec3 sunDir)
{
	vec3 localRay = normalize(rayDir);
	float sunIntensity = 1.0 - (dot(localRay, sunDir) * 0.5 + 0.5);
	//sunIntensity = (float)Math.Pow(sunIntensity, 14.0);
	sunIntensity = max(0.0, 0.01 / sunIntensity - 0.025);
	sunIntensity = min(sunIntensity, 40000.0);
	vec3 ground = mix(environmentGroundColor, environmentSphereColor,
					  pow(abs(localRay.y), 0.35)*sign(localRay.y) * 0.5 + 0.5);
	return ground + sunCol * sunIntensity;
}

// Function 327
vec3 getColorImage(vec2 uv)
{
    const float NB_DIV = 8.;
    const float NB_DIV2 = NB_DIV*NB_DIV;
    vec2 uv8 = uv*NB_DIV;
    float posX = floor( uv8.x);
    float posY = floor( uv8.y);
    
    float r = ( posX + posY * NB_DIV ) / NB_DIV2;
    float g = uv8.x - posX;
    float b = uv8.y - posY;
    return vec3( r, g, b);
}

// Function 328
vec3 iq_color_palette(vec3 a, vec3 b, vec3 c, vec3 d, float t)
{
    return a + b * cos(TWO_PI * (c*t + d));
}

// Function 329
vec3 adjust_out_of_gamut_sat(vec3 src)
{
    // Fix out-of-gamut saturation
    // Maximumum channel:
    float m = max(max(src.r, src.g), src.b);

    // Normalized color when the maximum channel exceeds 1.0
    src *= 1.0 / max(1.0, m);

    if (m > 1.0) {
        // When very bright, aggressively roll back intensity
        // to avoid the following desaturation pass for highlights
        // and emissives.
        m = pow(m, 0.2);
    }
    // Fade towards white when the max is bright (like a light saber core)
    src = mix(src, vec3(1.0), min(0.9, pow(2.0 * max(0.0, m - 0.85), 3.0)));
    
    return src;
}

// Function 330
vec4 calcColor( in vec3 pos, in vec3 nor, float material )
{
	vec4 materialColor = vec4(0.0);
	
		 if(material < 0.5) materialColor = colorScreen(pos.xz);
	else if(material < 1.5) materialColor = vec4(0.0,0.0,0.0,0.05);
	else if(material < 2.5) materialColor = vec4(0.15,0.15,0.15,0.2);
	else if(material < 3.5) materialColor = vec4(0.2,0.2,0.2,0.2);
		
	return materialColor;
}

// Function 331
vec3 surface_color(vec3 p)
{
    p /= scale;
    p /= 200.0;
    return sin(vec3(sceneSDF(p/5.0,0.0),sceneSDF(p*3.0,0.0),sceneSDF(p*2.0,0.0))*10.0)/4.0+vec3(.2);
}

// Function 332
vec3 colorBrushStroke(vec2 uv, vec3 inpColor, vec4 brushColor, vec2 p1, vec2 p2, float lineWidth)
{
    // flatten the line to be axis-aligned.
    vec2 rectDimensions = p2 - p1;
    float angle = atan(rectDimensions.x, rectDimensions.y);
    mat2 rotMat = rot2D(-angle);
    p1 *= rotMat;
    p2 *= rotMat;
    float halfLineWidth = lineWidth / 2.;
    p1 -= halfLineWidth;
    p2 += halfLineWidth;
	vec3 ret = colorAxisAlignedBrushStroke(uv * rotMat, uv, inpColor, brushColor, p1, p2);
    // todo: interaction between strokes, smearing like my other shader
    return ret;
}

// Function 333
vec3 getHexagonColor(vec4 h, vec3 pos)
{
    colors[0] = vec3(1., 0., 0.);
    colors[1] = vec3(0., 1., 0.);
    colors[2] = vec3(0., 0., 1.);
    
    int colnr = int(mod(h.x, 3.));
    vec4 h0 = h;
    
    #ifdef specrot
    pos.xz = rotateVec(pos.xz, float(colnr)*pi/3.);
    h = hexagon(pos.xz);
    #endif
    
    float lpx = pos.x - h.x*0.866025;
    float lpz = pos.z - h.x*1.5;
    vec2 pos2 = vec2(lpx, pos.y);
   
    float angle = getAngle(h0);
    pos2 = rotateVec(pos2, angle);
    
    //return colors[colnr];
    return pos2.y>0.?colors[colnr]:1.-colors[colnr];
}

// Function 334
vec3 skyColor(in vec3 rd)
{
    //vec3 par = vec3(0.075, 0.565, .03);
    vec3 par = vec3(.9, .81, .71);
    
    vec3 c = kali_set(sin(rd*6.), par);
    c = pow(min(vec3(1.), c*2.+vec3(1.,.86,.6)), 1.+114.*c);
    
    return clamp(c, 0., 1.);
}

// Function 335
vec3 getColor(vec3 norm, vec3 pos, int objnr, vec3 ray)
{
   if (objnr==FLOOR_OBJ)
      return getFloorColor(pos);   
   else if (objnr==SUPPORT_OBJ)
      return support_color;     
   else if (objnr==ROOM_OBJ)
      return getRoomColor(pos);
}

// Function 336
vec3 getSkyColor( vec3 rd )
{
    vec3 color = mix( SKY_COLOR_1 * 1.4, SKY_COLOR_2, rd.y / 9.0 );
	
    float fogFalloff = clamp( 8.0 * rd.y, 0.0, 1.0 );
    color = mix( FOG_COLOR, color, fogFalloff );
    color = mix( color, GRID_COLOR_1, smoothstep( -0.1, -0.2, rd.y ) );

    vec3 sunDir = normalize( SUN_DIRECTION );
    float sunGlow = smoothstep( 0.9, 1.0, dot( rd, sunDir ) );
        
    rd = mix( rd, sunDir, -1.0 ); // easier to bend vectors than fiddle with falloff :P
    float sun = smoothstep( 0.987, 0.99, dot(rd, sunDir ) );
    sun -= smoothstep( 0.1, 0.9, 0.5 );			        
    
    float stripes = mod( 50.0 * ( pow( rd.y + 0.15, 1.5 ) ) + 0.5, 1.0 ) -0.5;
    stripes = smoothstep( 0.2, 0.21, abs( stripes ) );
        
    
    // based on https://www.shadertoy.com/view/tssSz7
    vec2 starTile   = floor( rd.xy * 40.0 );
    vec2 starPos    = fract( rd.xy * 40.0 ) * 2.0 - 1.0;
    vec2 starRand = hash22( starTile );
    starPos += starRand * 2.0 - 1.0;
    float stars = saturate( 1.0 - ( ( sin( iTime * 1.0 + 50.0 * rd.y ) ) * 0.5 + 6.0 ) * length( starPos ) );
    stars *= step( 0.0, -sun );
    stars *= step( 0.9, starRand.x );
    stars *= 5.0;
           
    sun = 2.0 * clamp( sun * stripes, 0.0, 1.0 );
    
    vec3 sunCol = 4.0 * mix( SUN_COLOR_1, SUN_COLOR_2, -( rd.y - 0.1 ) / 0.3 );
    color = mix( color, sunCol, sun );

	color = mix( FOG_COLOR, color, 0.8 + 0.2 * fogFalloff );
    color = mix( color, sunCol, 0.25 * sunGlow );
    
    color += stars;

    // return vec3(stripes);
    // return vec3(sun);
    // return vec3(sunGlow);
    return color;
}

// Function 337
vec4 padColor( in vec2 uv )
{
    uv.y *= 12.0;
    uv.y = mod(uv.y,4.0);
    vec4 res = vec4(0);
    res += (step(-.5,uv.y)-step(0.5,uv.y))*PRP_COL;
    res += (step(0.5,uv.y)-step(1.5,uv.y))*YEL_COL;
    res += (step(1.5,uv.y)-step(2.5,uv.y))*BLU_COL;
    res += (step(2.5,uv.y)-step(3.5,uv.y))*GRN_COL;
    res += (step(3.5,uv.y)-step(4.5,uv.y))*ORG_COL;
    return res;
}

// Function 338
vec3 getColor(vec3 pos)
{
    vec3 color;
    color = pos;
    return color;   
}

// Function 339
vec4 Env_GetSkyColor( vec3 vViewPos, vec3 vViewDir, bool drawSun )
{
    
	vec4 vResult = vec4( 0.0, 0.0, 0.0, kFarDist );
   
    return vResult;	
}

// Function 340
vec3 compute_color(vec3 ro, vec3 rd, float t)
{
    vec3 p = ro+rd*t;
    //vec3 nor = normal(p);
    //vec3 ref = reflect(rd, nor);
    
    vec2 m = map(p).yz;
    #ifdef COOL_BACKGROUND
    if(m.x == 12.) {
        vec3 n = normal(p);
    	return mat_col(m,p) * pow(dot(n, vec3(0., 0., -1.)),2.);
    }
    #endif
    return mat_col(m,p);

	/*vec3 f = vec3(0.);
    //for(int i = 0; i < 1; ++i) {
    vec3 l = normalize(vec3(0.5, .7, .3));
   
    
    //float shd = calcSoftshadow(p, l, 10.);
    float dif = clamp( dot( nor, l ), 0.0, 1.0 );
   	float fre = pow( clamp(1.0+dot(nor,rd),0.0,1.0), 2.0 );
    
    
    vec3 v = vec3(0.);
    v += .6*vec3(dif);//*shd;
    v += .2*fre*vec3(.8, .7, .6);
 	f += mat_col(m,p)*v;
    //}
    return f;*/
}

// Function 341
vec3 getParticleColor(int partnr, float pint)
{
   float hue;
   float saturation;

   saturation = mix(part_min_saturation, part_max_saturation, random(float(partnr*6 + 44) + runnr*3.3))*0.45/pint;
   hue = mix(part_min_hue, part_max_hue, random(float(partnr + 124) + runnr*1.5)) + hue_time_factor*time2;
    
   return hsv2rgb(vec3(hue, saturation, pint));
}

// Function 342
vec4 colorize(ray _r)
{
	vec4 c = vec4(0.);
	
	// normal
	normal(_r);
	
	// material
	c = material(_r);
	
	// lighting
	c *= diffuse(_r,KL) * softShadow(_r,KL.p) + diffuse(_r,FL) * softShadow(_r,FL.p);
	//c *= diffuse(_r,KL) * hardShadow(_r,KL.p) + diffuse(_r,FL) * hardShadow(_r,FL.p);
	c += specular(_r,KL);
	c += specular(_r,FL);
	//c += ambient(KL);
	//c += ambient(FL);
    
 ///debug
 //test = vec3(softShadow(_r,KL.p));
    	
	return c;
}

// Function 343
void SetGlyphColor(float r,float g,float b){drawColor=vec3(b,g,b);}

// Function 344
vec3 getColorCielab(float rawLightness, float rawChroma, float hue) {
    vec3 lch = vec3(rawLightness * 100.0, rawChroma * 170.0, hue);
    vec3 cielab = lchToLab(lch);
    return xyzToLinearSrgb(cielabToXyz(cielab));
}

// Function 345
vec3 skyColor(vec3 rd){
    
    vec3 outLight = vec3(0.125);
    outLight+= addLight(vec3(0.7,0.5,0.),normalize(-vec3(0.2,0.05,0.2)),rd);
    
    
    outLight+=addLight(vec3(0.1,0.3,0.7),normalize(-vec3(-0.2,0.05,-0.2)),rd);
    return outLight;
}

// Function 346
vec3 GetMaterialsColor(Ray r, float matID
){if(matID>7.)return vec3(0)
 ;float fakeOA = pow((1.-float(r.iter)/float(maxStepRayMarching)),.7)
 ;return rainbow((sqrt(5.)*.5+.5)*matID*2.)*fakeOA
 ;}

// Function 347
vec3 supports_color(vec3 ray)
{
    return vec3(0.1);
}

// Function 348
vec3 colorForIndex(int index)
{
    index = index%12;
    if( index == 0 )
    {
         return vec3(1., 0., 0.);
    }
    else if( index == 1 )
    {
         return vec3(0., 1., 0.);
    }
    else if( index == 2 )
    {
         return vec3(0., 0., 1.);
    }
    else if( index == 3 )
    {
         return vec3(1., 1., 0.);
    }
    else if( index == 4 )
    {
         return vec3(0., 1., 1.);
    }
    else if( index == 5 )
    {
         return vec3(1., 0., 1.);
    }
    else if( index == 6 )
    {
         return vec3(1., 1., 1.);
    }
    else if( index == 7 )
    {
         return vec3(1., 0.5, 0.5);
    }
    else if( index == 8 )
    {
         return vec3(0.5, 1., 0.5);
    }
    else if( index == 9 )
    {
         return vec3(1., 0.5, 0.);
    }
    else if( index == 10 )
    {
         return vec3(0., 0.5, 1.);
    }
    else
    {
         return vec3(0.5, 0.5, 1.);
    }
}

// Function 349
float evaluateColor(in float aRow, in vec2 fragCoord, in float aCycle) {
    float tFinalHue = 0.0;
    float iCurrentTime = iTime - aCycleDelay;
    //float iCurrentTime = iTime - (aCycle * aRow);
    float tPercentTimeUntilAllRed = iCurrentTime/aCycle;
    if (tPercentTimeUntilAllRed > (fragCoord.x/iResolution.x)) {
        tFinalHue = convertHue(RED) + sin(iTime*1.1)*.075;
        if (tPercentTimeUntilAllRed > 1.0) {
            float tPercentTimeUntilAllYellow = (iCurrentTime-aCycle*12.0)/aCycle;
            if (tPercentTimeUntilAllYellow > (fragCoord.x/iResolution.x)) {
                tFinalHue = convertHue(YELLOW) + abs(sin(iTime*0.9)*.05)*-1.0;
                float tPercentageTimeUntilAllGreen = (iCurrentTime-aCycle*2.0*12.0)/aCycle;
                if (tPercentageTimeUntilAllGreen > (fragCoord.x/iResolution.x)) { 
                    tFinalHue = convertHue(GREEN);
                }
            }
        }
    } else {
        tFinalHue = convertHue(BLUE) + abs(sin(iTime * .6)*.075);
    }
    return tFinalHue;
}

// Function 350
vec3 getFragmentColor (in vec3 origin, in vec3 direction) {
	vec3 lightDirection = normalize (LIGHT);
	vec2 delta = vec2 (DELTA, 0.0);

	vec3 fragColor = vec3 (0.0, 0.0, 0.0);
	float intensity = 1.0;

	float distanceFactor = 1.0;
	float refractionRatio = 1.0 / REFRACT_INDEX;
	float rayStepCount = 0.0;
	for (int rayIndex = 0; rayIndex < RAY_COUNT; ++rayIndex) {

		// Ray marching
		float dist = RAY_LENGTH_MAX;
		float rayLength = 0.0;
		for (int rayStep = 0; rayStep < RAY_STEP_MAX; ++rayStep) {
			dist = distanceFactor * getDistance (origin);
			float distMin = max (dist, DELTA);
			rayLength += distMin;
			if (dist < 0.0 || rayLength > RAY_LENGTH_MAX) {
				break;
			}
			origin += direction * distMin;
			++rayStepCount;
		}

		// Check whether we hit something
		vec3 backColor = vec3 (0.0, 0.0, 0.1 + 0.2 * max (0.0, dot (-direction, lightDirection)));
		if (dist >= 0.0) {
			fragColor = fragColor * (1.0 - intensity) + backColor * intensity;
			break;
		}

		// Get the normal
		vec3 normal = normalize (distanceFactor * vec3 (
			getDistance (origin + delta.xyy) - getDistance (origin - delta.xyy),
			getDistance (origin + delta.yxy) - getDistance (origin - delta.yxy),
			getDistance (origin + delta.yyx) - getDistance (origin - delta.yyx)));

		// Basic lighting
		vec3 reflection = reflect (direction, normal);
		if (distanceFactor > 0.0) {
			float relfectionDiffuse = max (0.0, dot (normal, lightDirection));
			float relfectionSpecular = pow (max (0.0, dot (reflection, lightDirection)), SPECULAR_POWER) * SPECULAR_INTENSITY;
			float fade = pow (1.0 - rayLength / RAY_LENGTH_MAX, FADE_POWER);

			vec3 localColor = max (sin (k * k), 0.2);
			localColor = (AMBIENT + relfectionDiffuse) * localColor + relfectionSpecular;
			localColor = mix (backColor, localColor, fade);

			fragColor = fragColor * (1.0 - intensity) + localColor * intensity;
			intensity *= REFRACT_FACTOR;
		}

		// Next ray...
		vec3 refraction = refract (direction, normal, refractionRatio);
		if (dot (refraction, refraction) < DELTA) {
			direction = reflection;
			origin += direction * DELTA * 2.0;
		}
		else {
			direction = refraction;
			distanceFactor = -distanceFactor;
			refractionRatio = 1.0 / refractionRatio;
		}
	}

	// Return the fragment color
	return fragColor * LUMINOSITY_FACTOR + GLOW_FACTOR * rayStepCount / float (RAY_STEP_MAX * RAY_COUNT);
}

// Function 351
vec3 surface_color(vec3 p)
{
    p = animate(p*2.0);
    vec3 col;
    col = p.xyz*sin(p.x+p.y+p.z);
    
    // Output to screen
    return vec3(col.y,col.z,p.x)/500.0;
}

// Function 352
vec3 color(in ray r) {
    vec3 col = vec3(1);  
	hit_record rec;
    
    for (int i=0; i<MAX_RECURSION; i++) {
    	if (world_hit(r, 0.001, MAX_FLOAT, rec)) {
            ray scattered;
            vec3 attenuation;
            if (material_scatter(r, rec, attenuation, scattered)) {
                col *= attenuation;
                r = scattered;
            } else {
                return vec3(0);
            }
	    } else {
            float t = .5*r.direction.y + .5;
            col *= mix(vec3(1),vec3(.5,.7,1), t);
            return col;
    	}
    }
    return vec3(0);
}

// Function 353
vec3 naturalColor( float hue, float value )
{
    vec3  T = .4+.4*cos(hue+vec3(0,2.1,-2.1));
    float z = value;
    return pow(T, vec3(z));  
}

// Function 354
vec3 GetMacBethColorCOLUMN_COUNT5(const in float yDist)
{
    float compareY = LINE_COUNT;
    
    if(yDist > --compareY)
        return BLUEFLOWER;
	else if(yDist > --compareY)
		return YELLOWGREEN;
	else if(yDist > --compareY)
		return MAGENTA;
	else
		return NEUTRAL35;
}

// Function 355
float isRedColor( vec3 color )
{
    return isRedColorRGB(color);
    //return isRedColorHSV(color);
}

// Function 356
float getcolor(vec2 uv) {
    vec4 c = texture(iChannel0, uv);
    return max(fixcolor(c.r),max(fixcolor(c.g),fixcolor(c.b)));
}

// Function 357
float watercolor (vec2 p) {
       p*=15.;
       vec2 q = vec2(0.);
       q.x = fbm(p);
       q.y = fbm( p + vec2(1.0));
       vec2 r = vec2(0.);
       r.x = fbm( p + 1.0*q + vec2(1.7,9.2));
       r.y = fbm( p + 1.0*q + vec2(8.3,2.8));
       float f = fbm(p+r);
       return clamp(f,0.,1.);
}

// Function 358
vec3 GetSampleColor(vec2 uv)
{
    RayInfo r;
    
    //sets up ray direction
    r.dir = vec3(0.,0.,1.);
    
    //applies fov
    if (fishEye)
    {
    	vec3 crossv = cross(r.dir,vec3(uv,0.));
    	r.dir = Rotate(BuildQuat(normalize(crossv),length(uv)*FOV),r.dir);
    }
    else
    {
    	r.dir = vec3(uv.xy*FOV,1.);
    }
    
    //applies look dir
    r.pos = objects[o_cam].pos;
    r.dir = Rotate(objects[o_cam].rot,r.dir);
    
    
    MarchPOV(r,playerTime);
        
    
    return GetDiffuse(r);
}

// Function 359
vec3 GetMacBethColorCOLUMN_COUNT6(const in float yDist)
{
    float compareY = LINE_COUNT;
    
    if(yDist > --compareY)
        return BLUISHGREEN;
	else if(yDist > --compareY)
		return ORANGEYELLOW;
	else if(yDist > --compareY)
		return CYAN;
	else
		return BLACK;
}

// Function 360
vec3 MapColor(vec3 srgb)
{
    #if MODE == 0
    return srgb * sRGBtoAP1;
    #else
    return srgb;
    #endif
}

// Function 361
vec4 colorCorrect(vec3 color)
{
    vec3 x = max(vec3(.0), color*aperture-.004);
    vec3 retColor = (x*(6.2*x+.5))/(x*(6.2*x+1.7)+0.06);
    return vec4(min(retColor, 1.0), 1.0);
}

// Function 362
vec3 baseColor(vec2 uv) {
  return 3. * uv.y * lerp(hue(uv.x), vec3(1), 0.5 + 0.5 * sin(iTime));
}

// Function 363
vec4 calcColor( in vec3 pos, in vec3 nor, in float material, out vec3 normal )
{
	vec4 materialColor = vec4(0.0);
	
	vec3 q = pos - mix(ballPos,vec3(0.0),material);
	float radius = mix(0.5,3.0,material);
	vec2 angles = vec2(atan(abs(q.z)/abs(q.x)),acos(q.y/radius));
	
	float mixer = step(1.0,material);
	
	materialColor = mix(vec4(vec3(1.0,0.2,0.2)*texture(iChannel2,angles).xyz,0.05),vec4(texture(iChannel0,angles).xyz,0.30),mixer);	
	vec2 normalMap = mix(4.0*GetNormalMap(iChannel2,iChannelResolution[2].xy,angles),GetNormalMap(iChannel0,iChannelResolution[0].xy,angles),mixer);
		
	vec3 left = normalize(cross(vec3(0.0,1.0,0.0),nor));
	vec3 up = normalize(cross(left,nor));
	normal = normalize(normalMap.x*left + nor + normalMap.y*up);
	
	return materialColor;
}

// Function 364
vec3 obj_color(vec3 norm, vec3 pos)
{
    vec2 d = vec2(2. + 0.8*sin(iTime*1.2), 0.); 
    float s1 = 1./pow(length(pos + vec3(rotateVec(d, 0.), 0.)), 2.);
    float s2 = -1./pow(length(pos + vec3(rotateVec(d, 2./3.*pi), 0.)), 2.);
    float s =  s1 + s2;
    vec3 col1 = mix(vec3(1., 0.1, 0.1), vec3(0.1, 1., 0.1), clamp(s*6. + 0.6, 0.1, 0.9));
     
    s = 1./pow(length(pos + vec3(rotateVec(d, 4./3.*pi), 0.)), 2.);
    return mix(col1, vec3(0.1, 0.1, 1.0), clamp(s*6. - 0.6, 0.1, 0.9));
}

// Function 365
vec3 getFabricColor(vec2 uv) {
    float aa = texture(iChannel1,uv).r*0.5 + texture(iChannel1,uv.yx).r*0.5;//add some more inperfection
	return vec3(0.3, 0.9, 0.2)*(aa*0.5 + 0.2)*0.6;
}

// Function 366
vec3 GetMacBethColorCOLUMN_COUNT2(const in float yDist)
{
    float compareY = LINE_COUNT;
    
    if(yDist > --compareY)
        return LIGHTSKIN;
	else if(yDist > --compareY)
		return PURPLISHBLUE;
	else if(yDist > --compareY)
		return GREEN;
	else
		return NEUTRAL8;
}

// Function 367
float color_to_val_2(in vec3 color) {
    return  max(abs(color.r - color.g), max(abs(color.g - color.b), abs(color.b - color.r)));
}

// Function 368
vec3 rayColor(in vec3 ro, in vec3 rd)
{
    // magic params for kali-set
    vec3 par1 = vec3(.9, .6+.5*sin(ro.z/50.), 1.),	// scene geometry 
         par2 = vec3(.63, .55, .73),				// normal/bump map
         par3 = vec3(1.02, 0.82, 0.77); 			// normal/texture
    
#if 1
    float t = trace(ro, rd, par1);
#else    
    float t = trace_enhanced(ro, rd, par1);
#endif    
    vec3 p = ro + t * rd;
    float d = DE(p, par1);
    
    vec3 col = vec3(0.);

    // did ray hit?
    if (d < 0.03) 
    {
        float scr_eps = max(0.001, (t-0.1)*0.025);
        // "some" texture values
        vec3 kt = kali_tex(p, par3);
        // surface normal
        vec3 n = DE_norm(p, par1, 0.5*scr_eps), nn = n;
        // normal displacement
        n = normalize(n + 0.3*kali_tex_norm(p, par3+0.1*n, vec3(1), scr_eps));
        n = normalize(n + 0.3*DE_norm(sin(n*3.+kt), par2, 2.*scr_eps)); // micro-bumps
        // reflected ray
        vec3 rrd = reflect(rd,n);
		// normal towards light
        vec3 ln = normalize(path(p.z+.1) - p);
		// 1. - occlusion
        float ao = pow(traceAO(p, n, par1), 1.+3.*t);
        // surface color
        vec3 col1 = .45 * (vec3(.7,1.,.4) + kali_tex(p, par3));
        vec3 col2 = vec3(1.,.8,.6) + .3 * vec3(1.,.7,-.6) * kali_tex(p, par3);
        vec3 k = kali_set_av(sin(p*(1.+3.*ao))*.3, par3);
        vec3 surf = (.1 + .9 * ao) 
            		//* vec3(1.);
            		* mix(col1, col2, min(1., pow(ao*2.2-.8*kt.x,5.)));
		// desaturate
        surf += .24 * (dot(surf,vec3(.3,.6,.1)) - surf);

        // -- lighting --
        
        float fres = pow(max(0., 1.-dot(rrd, n)), 1.) / (1.+2.*t);

        // phong
        surf += .25 * ao * max(0., dot(n, ln));
        // spec
        float d = max(0., dot(rrd, ln));
        surf += .4 * pow(ao*1.2,5.) * (.5 * d + .7 * pow(d, 8.));

        // fresnel highlight
        surf += clamp((t-.06)*8., 0.,1.6) * 
            	(.2+.8*ao) * vec3(.7,.8,1.) * fres;
        
        // environment map
        surf += .2 * (1.-fres) * ao * skyColor(rrd);
    
        // distance fog
    	col = surf * pow(1.-t / max_t, 1.3);
    }
    
    return col;
}

// Function 369
vec3 GetSampleColor(vec2 uv
){Ray r
 ;r.dir = vec3(0,0,1)
 ;if (fishEye
 ){vec3 crossv=cross(r.dir,vec3(uv,0))
  ;r.dir=qr(aa2q(length(uv)*FOV,normalize(crossv)),r.dir)
  ;}else r.dir = vec3(uv.xy*FOV,1.)
 ;//apply look dir
 ;r.b = objPos[oCam]//es100 error , no array of class allowed
 ;r.dir = qr(objRot[oCam],r.dir)//es100 error , no array of class allowed
 ;MarchPOV(r,playerTime)
 ;return GetDiffuse(r);}

// Function 370
vec4 uiColor(int id){return texture(iChannel0, vec2(float(id)+.5,1.5)/iResolution.xy);}

// Function 371
void glitchColor(vec2 p, inout vec3 color) {
    vec2 groupSize = vec2(.85,.225) * glitchScale;
    vec2 subGrid = vec2(0,7);
    float speed = 6.;
    GlitchSeed seed = glitchSeed(glitchCoord(p, groupSize), speed);
    seed.prob *= .4;
    if (shouldApply(seed) == 2.) {
        vec2 co = mod(p, groupSize) / groupSize;
        co *= subGrid;
        float a = max(co.x, co.y);
        //color.rgb *= vec3(
        //  min(floor(mod(a - 0., 3.)), 1.),
        //    min(floor(mod(a - 1., 3.)), 1.),
        //    min(floor(mod(a - 2., 3.)), 1.)
        //);
        
        color *= min(floor(mod(a, 3.)), 2.) * 20.;
    }
}

// Function 372
void setColor(inout vec3 color, in vec3 pos, in vec3 ray, in vec3 light, in Type_Param PARAM){
    // Set color to the front object...
    
    #ifdef ACTIVATE_TEXTURE
    vec3 p = pos+PARAM.t*ray;
    if(PARAM.hit == SKY) return;
    else if(PARAM.hit == SPH) 
        color = .7*spCol+.3*texture(iChannel0,vec2(atan(p.x,p.z),asin(p.y-spO.y))).rgb;
    else if(PARAM.hit == XCYL)
        color = cyCol*(1.- min(vec3(.2),texture(iChannel0, vec2(.5*p.x,atan(p.y,p.z))).rgb));
    else if(PARAM.hit == ZCYL)
        color = cyCol*(1.-min(vec3(.2),texture(iChannel0, vec2(.5*p.z,atan(p.x,p.y))).rgb));
    else if(PARAM.hit == BOX)
        color = boxCol*texture(iChannel2,vec2(p.x+p.z,p.y+p.x)).rgb;
    else if(PARAM.hit == CONE)
        color = .5*(conCol*texture(iChannel3,vec2(.15*atan(p.x-conO.x,p.z-conO.z),p.y)).rgb);
    
    #else
    if(PARAM.hit == SKY) return;
    else if(PARAM.hit == SPH)  color = spCol;
    else if(PARAM.hit == XCYL) color = cyCol;
    else if(PARAM.hit == ZCYL) color = cyCol;
    else if(PARAM.hit == BOX)  color = boxCol;
    else if(PARAM.hit == CONE) color = conCol;
    #endif
        
    color *= max(0.,dot(light,PARAM.n));		// light shading
}

// Function 373
void mixColorLine(vec2 uv,inout vec3 col,vec2 lineA,vec2 lineB,float scale)
{
    col = mix(
        col , 
        hash3point(lineA+lineB) ,
        1.0 - smoothstep(0.0,1.0,sqrt(sqrt( segment(uv,lineA,lineB).x * scale )))
    );
}

// Function 374
vec4 colored(vec4 base, vec2 coord_){
    vec3 col = vec3(rainbowColor(coord_)) * 1.7 * base.r;
    return vec4(col, 1.);
}

// Function 375
vec3 r_color()
{
	return mix( vec3(1,0,0), vec3(16.0 * 15.0 / 255.0, 160.0 / 255.0, 13.0 * 16.0 / 255.0 ), selector(2.25,2.0) );
}

// Function 376
vec3 getColor(float x)
{
  int i = int(floor(13. * x));

  if (i == 0) return vec3(1., 0., 0.);
  if (i == 1) return vec3(1., .5, 0.);
  if (i == 2) return vec3(1., 1., 0.);
  if (i == 3) return vec3(.5, 1., 0.);
  if (i == 4) return vec3(0., 1., 0.);
  if (i == 5) return vec3(0., 1., .5);
  if (i == 6) return vec3(0., 1., 1.);
  if (i == 7) return vec3(0., .5, 1.);
  if (i == 8) return vec3(0., 0., 1.);
  if (i == 9) return vec3(.5, 0., 1.);
  if (i == 10) return vec3(1., 0., 1.);
  if (i == 11) return vec3(1., 0., .5);

  return vec3(1.);
}

// Function 377
vec4 colorAndDepth(vec3 pos, vec3 dir){
    
    vec3 color = vec3(0.0);
    vec3 absorption = vec3(1.0);
    float depth = -1.0;
    
    for(int i = 0; i < LightBounces; i++){
        
        vec3 sphereNormal, hitSpherePos = pos;
        bool hitSphere = sphereIntersect(hitSpherePos, dir, sphereNormal);
        float dSphere = length(pos - hitSpherePos);
            
        vec3 thoriiNormal, hitThoriiPos = pos;
        bool hitThorii = thoriiIntersect(hitThoriiPos, dir, thoriiNormal);
        float dThorii = length(pos - hitThoriiPos);
        
        if(hitSphere && (!hitThorii || dThorii > dSphere)){
            
            pos = hitSpherePos;
            
            if(depth == -1.0) depth = length(CamPos-pos);
            
            float fresnel = fresnel(dir, sphereNormal, Ior);
            vec3 reflectDir = reflect(dir, sphereNormal);
            vec3 reflectColor;
            hitThoriiPos = pos;
            if(thoriiIntersect(hitThoriiPos, reflectDir, thoriiNormal))
                reflectColor = background(reflect(reflectDir, thoriiNormal)) * GoldAlbedo;
            else 
                reflectColor = background(reflectDir);

            //first refraction
            dir = refract(dir, sphereNormal, 1.0/Ior);
           
            vec3 newAbsorption;
            vec3 rayMarchColor = marchRay(pos, dir, newAbsorption);
            
            color += absorption * (fresnel*reflectColor + (1.0-fresnel) * rayMarchColor);
            
            //second refraction
            pos += distanceInSphere(pos, dir) * dir;
            vec3 refractDir = refract(dir, -pos, Ior);
            
            if(refractDir != vec3(0)) dir = refractDir;
                
            absorption *= (1.0-fresnel) * newAbsorption;
           
        } else if(hitThorii){
            
            dir = reflect(dir, thoriiNormal);
            if(depth == -1.0) depth = length(CamPos-hitThoriiPos);
            pos = hitThoriiPos + thoriiNormal * Epsilon;
            
            absorption *= GoldAlbedo;
        } else {
    		if(depth == -1.0) depth = 1000.0;
            break;
        }
        
        if(dir == vec3(0.0)) return vec4(color, depth);
    }
    
    return vec4(color + absorption * background(dir), depth);
}

// Function 378
vec4 colorNote(vec4 n, float y
){float a=mi(abs(n-y))    
 ;y*=.02
 ;vec4 b=vec4(a)
 ;vec4 c=vec4(rainbows(y),1.)
 ;//vec4 c=vec4(1.-sat(vec3(y,.5+.5*cos(y*8.),1.-y)),1)//lazy gradient
 ;//return c
 ;return pdOut(b,c)
 ;}

// Function 379
vec3 randomcolor(float value)
{
    return rainbow(rnd(value));
}

// Function 380
bool colorize(vec2 uv, float time){
	time *= .5;
    return step(.75, abs((uv.x - uv.y * .25 + .125) - sin(time) * 3.)) > 0.;
}

// Function 381
vec3 colormap(float value) {
	float maxv = ClampLevel;
	vec3 c1,c2;
	float t;
	if (value < maxv / 3.) {
		c1 = vec3(1.);   	   c2 = vec3(1., 1., .5);
		t =  1./3.;
	} else if (value < maxv * 2. / 3.) {
		c1 = vec3(1., 1., .5); c2 = vec3(1., 0,  0.);
		t =  2./3. ;
	} else {
		c1 = vec3(1., 0., 0.); c2 = vec3(0.);
		t =  1.;
	}
	t = (t*maxv-value)/(maxv/3.);
	return t*c1 + (1.-t)*c2;
}

// Function 382
vec3 getColor (in Ray ray) {
    vec3 col = vec3 (.0);

    for (int i = 0; i < MAX_SAMPLES; ++i) {
        // accumulate path
        col += trace (ray, MAX_BOUNCES);
    }
    col = col / float (MAX_SAMPLES);

    // apply tonemapping & gamma correction
    col = col / (1. + col);
    col = sqrt (col);

    return col;
}

// Function 383
vec3 heatToColor(float heat)
{
    vec3 col = mix(vec3(0.0), vec3(1., .3, .0),clamp(heat * 15. -2.,   0., 1.));
    col = mix(col, vec3(1., 1., .6), clamp(heat * 15.1-4.,   0., 1.));
    col = mix(col, vec3(1., .9, .8), clamp(heat * 190. - 60.,   0., 1.));
    return col;
}

// Function 384
vec3 sky_color_s(vec3 ray)
{
    float elev = atan(ray.y);
 
    const float cloudsize = 0.25;
    vec3 sky = ambientColor + vec3(0.4, 0.22, 0.05)*2.5*(1. - elev);
    
    vec3 grass = vec3(0.0, 0.35, 0.25) + vec3(0.22, 0.16, -0.03)*2.8*(0.65 - elev);
    
    return mix(mix(grass, vec3(0.65), smoothstep(-0.31, -0.312, elev)), sky, smoothstep(-0.0003, 0.0003, elev)); 
}

// Function 385
vec3  ReturnColor_4(float r1, float r2, float r3){
	vec3 P12 = _Color2- _Color1;
    vec3 P13 = _Color3-_Color1;
    vec3 P14 = _Color4 - _Color1;
    return sampleOnATriangle(r1, r2, _Color1+P12*r3, _Color1+P13*r3, _Color1+P14*r3);
    
}

// Function 386
vec3 circleSampleColor(vec2 dist1, vec2 center1, vec2 dist2, vec2 center2)
{
    vec2 curRadius1 = vec2(0.0);
    vec2 curRadius2 = vec2(0.0);
	vec3 returnValue = vec3(0.0);
    
    for (int c = 0; c < CIRCLE_NUMBER; ++c)
    {
    	float normalizedAngle = 0.0;
        curRadius1 += dist1;
        curRadius2 += dist2;
        for (int s = 0; s < SAMPLE_PER_CIRCLE; ++s)
        {
            float angle = normalizedAngle * 3.1415 * 2.0;
            vec2 uvToSample1 = center1 + vec2(cos(angle), sin(angle)) * curRadius1;
            vec2 uvToSample2 = center2 + vec2(cos(angle), sin(angle)) * curRadius2;
            vec3 sampledColor1 = texture(iChannel0, uvToSample1).rgb;
            vec3 sampledColor2 = texture(iChannel1, uvToSample2).rgb;
            if (passTest(sampledColor1))
				returnValue += sampledColor2 / float(CIRCLE_NUMBER * SAMPLE_PER_CIRCLE);
            else
                returnValue += sampledColor1 / float(CIRCLE_NUMBER * SAMPLE_PER_CIRCLE);
            normalizedAngle += 1.0 / float(SAMPLE_PER_CIRCLE);
        }
    }
    return (returnValue);
}

// Function 387
void applyColor(vec3 paint, inout vec3 col, vec2 p, vec2 a, vec2 b, vec2 c)
{
	if (insideTri(p, a, b, c)) col = mix(col, paint, max(col.r, 1.0));
}

// Function 388
float watercolor (vec2 p) {
       p*=5.;
       vec2 q = vec2(0.);
       q.x = fbm(p);
       q.y = fbm( p + vec2(1.0));
       vec2 r = vec2(0.);
       r.x = fbm( p + 1.0*q + vec2(1.7,9.2));
       r.y = fbm( p + 1.0*q + vec2(8.3,2.8));
       float f = fbm(p+r);
       return clamp(f*1.1,0.,1.);
}

// Function 389
vec3 doColor(in vec3 sp, in vec3 rd, in vec3 sn, in vec3 lp, in float t){
    
    vec3 sceneCol = vec3(0);
    
    if(t<FAR){
        
           // Texture bump the normal.
    	float sz0 = 1./1.;
    	vec3 txP = sp;
        //txP.xy -= path(txP.z);
        //txP.xz *= r2(getRndID(svVRnd)*6.2831);
        sn = texBump(iChannel0, txP*sz0, sn, .005);///(1. + t/FAR)
 

        // Retrieving the normal at the hit point.
        //sn = getNormal(sp);  
        float sh = softShadow(sp, lp, 12.);
        float ao = calcAO(sp, sn);
        sh = min(sh + ao*.3, 1.);

        vec3 ld = lp - sp; // Light direction vector.
        float lDist = max(length(ld), .001); // Light to surface distance.
        ld /= lDist; // Normalizing the light vector.

        // Attenuating the light, based on distance.
        float atten = 1.5/(1. + lDist*.1 + lDist*lDist*.02);

        // Standard diffuse term.
        float diff = max(dot(sn, ld), 0.);
        //if(svLitID == 0.) diff = pow(diff, 4.)*2.;
        // Standard specualr term.
        float spec = pow(max( dot( reflect(-ld, sn), -rd ), 0.0 ), 32.);
        float fres = clamp(1.0 + dot(rd, sn), 0.0, 1.0); // Fresnel reflection term.
        //float Schlick = pow( 1. - max(dot(rd, normalize(rd + ld)), 0.), 5.0);
        //float fre2 = mix(.5, 1., Schlick);  //F0 = .5.
        


        // Coloring the object. You could set it to a single color, to
        // make things simpler, if you wanted.
        vec3 objCol = getObjectColor(sp, sn);


        // Combining the above terms to produce the final scene color.
        sceneCol = objCol*(diff + vec3(1, .6, .3)*spec*4. + .5*ao + vec3(.3, .5, 1)*fres*fres*2.);

        // Fake environment mapping.
        sceneCol += pow(sceneCol, vec3(1.))*envMap(reflect(rd, sn))*4.;
        
       
        // Applying the shadows and ambient occlusion.
        sceneCol *= atten*sh*ao;
        
        // For whatever reason, I didn't want the shadows and such to effect the glow, so I layered
        // it over the top.
        sceneCol += (objCol*6. + 1.)*glow; //*(sh*.35 + .65);
 
        //sceneCol = vec3(sh);
    
    }
    
    

    
    // Return the color. Done once every pass... of which there are
    // only two, in this particular instance.
    return sceneCol;
    
}

// Function 390
vec3 GetMaterialsColor(RayInfo r, int matID)
{
    float fakeOA = pow((1.-float(r.iter)/float(maxStepRayMarching)),.7);
    //fakeOA = 1.;
    
    
    switch(matID){
    case 0:
        return vec3(0.4,0.4,0.4)*fakeOA;
    case 1:
        return vec3(0.8,0.4,0.4)*fakeOA;
    case 2:
        return vec3(0.4,0.8,0.4)*fakeOA;
    case 3:
        return vec3(0.4,0.4,0.8)*fakeOA;
    case 4:
        return vec3(0.8,0.8,0.4)*fakeOA;
    case 5:
        return vec3(0.4,0.8,0.8)*fakeOA;
    case 6:
        return vec3(0.8,0.4,0.8)*fakeOA;
    case 7:
        return vec3(0.9,0.9,0.9)*fakeOA;
    case 8:
        return vec3(0.0,0.0,0.0)*fakeOA;
	}
}

// Function 391
vec4 calcColor( in vec3 pos, in vec3 nor, float material )
{
	vec4 materialColor = vec4(0.0);
	
		 if(material < 0.5) materialColor = vec4(0.7,0.7,0.0,0.05);
	else if(material < 1.5) materialColor = vec4(0.0,0.0,0.0,0.05);
	else if(material < 2.5) materialColor = vec4(0.2,0.0,0.0,0.05);
	else if(material < 3.5) materialColor = calcMarkerColor(pos);
		
	return materialColor;
}

// Function 392
vec3 Color(vec3 ro, vec3 rd, float t, float px, vec3 col, bool bFill, vec2 fragCoord){
	ro+=rd*t;
	bColoring=true;float d=DE(ro);bColoring=false;
	vec2 e=vec2(px*t,0.0);
	vec3 dn=vec3(DE(ro-e.xyy),DE(ro-e.yxy),DE(ro-e.yyx));
	vec3 dp=vec3(DE(ro+e.xyy),DE(ro+e.yxy),DE(ro+e.yyx));
	vec3 N=(dp-dn)/(length(dp-vec3(d))+length(vec3(d)-dn));
	vec3 R=reflect(rd,N);
	vec3 lc=vec3(1.0,0.9,0.8),sc=sqrt(abs(sin(mcol))),rc=Sky(R);
	float sh=clamp(shadao(ro,L,px*t,fragCoord)+0.2,0.0,1.0);
	sh=sh*(0.5+0.5*dot(N,L))*exp(-t*0.125);
	vec3 scol=sh*lc*(sc+rc*pow(max(0.0,dot(R,L)),4.0));
	if(bFill)d*=0.05;
	col=mix(scol,col,clamp(d/(px*t),0.0,1.0));
	return col;
}

// Function 393
vec3 rainbowColor(in vec3 ray_dir) 
{ 
    RAINBOW_DIR = normalize(RAINBOW_DIR);   
		
    float theta = degrees(acos(dot(RAINBOW_DIR, ray_dir)));
    vec3 nd = clamp(1.0 - abs((RAINBOW_COLOR_RANGE - theta) * 0.2), 0.0, 1.0);
    vec3 color = smoothstep(nd) * RAINBOW_INTENSITY;
    
    return color * max((RAINBOW_BRIGHTNESS - 0.75) * 1.5, 0.0);
}

// Function 394
vec3 obj_color(vec3 norm, vec3 pos, float objnr)
{
    vec3 col = sideColors[int(objnr)];
    float zPos2 = pos.z + zPos - (objnr - 1.)*objSpacing;
    
    if (zPos2<-textDepth)
        col = faceColors[int(objnr)];    
    else if (zPos2<-textDepth + textBevel)
        col = bevelColors[int(objnr)];

    return col;
}

// Function 395
vec4 borderColor(float x0, float x1, vec2 uv, 
                 float leftVisible, float rightVisible, 
                 vec4 intCol, vec4 extCol) {

    vec4 white = vec4(1.);
    
   	// the exterior side of the left border
	float outside = (step(uv.x, x0-thick) + step(x0, uv.x));
    vec4 borderLeft = mix(white, extCol, S(x0, x0-thick, uv.x)) 
    	* (1. - outside) * leftVisible;
        
    // the interior side of the left border
    outside = (step(uv.x, x0) + step(x0+thick, uv.x));
    vec4 borderLeft2 = mix(intCol, white, S(x0+thick, x0, uv.x)) 
    	* (1. - outside)* leftVisible;      
    
    // the exterior side of the right border
    outside = (step(uv.x, x1) + step(x1+thick, uv.x));        
    vec4 borderRight = mix(white, extCol, S(x1, x1+thick, uv.x))
        * (1.- outside) * rightVisible ;
    
    // the interior side of the right border
    outside = (step(uv.x, x1-thick) + step(x1, uv.x));            
    vec4 borderRight2 = mix(intCol, white, S(x1-thick, x1, uv.x))
    	* (1.- outside) * rightVisible;
    
    return borderLeft + borderLeft2 + borderRight + borderRight2;     
}

// Function 396
void doShadowColor(in ray primaryRay, inout vec4 col) {
	vec4 returnColor = vec4(0.0);
	vec2 shadowUV;
	vec2 shadowT;
	int shadowId;
	float shadowCheckDelta = light.w;
	ray shadowRay;
	shadowRay.lightColor = primaryRay.lightColor;
	shadowRay.transmittance = primaryRay.transmittance;
	vec3 pos = primaryRay.origin + primaryRay.rayLength*primaryRay.direction;
	shadowRay.origin = pos - 0.001*primaryRay.direction;
	for (int i = 0; i < 7; i++) {
		//soft shadows
		if (i == 0) {
			vec4 tempLight = light;
			tempLight.x += shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else if (i == 1) {
			vec4 tempLight = light;
			tempLight.x -= shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else if (i == 2) {
			vec4 tempLight = light;
			tempLight.y += shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else if (i == 3) {
			vec4 tempLight = light;
			tempLight.y -= shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else if (i == 4) {
			vec4 tempLight = light;
			tempLight.z += shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else if (i == 5) {
			vec4 tempLight = light;
			tempLight.z -= shadowCheckDelta;
			shadowRay.direction = normalize(tempLight.xyz-pos);
		}
		else
			shadowRay.direction = normalize(light.xyz-pos);
		shadowId = intersect(shadowRay, shadowT, shadowUV);
		vec3 shadowHit = shadowRay.origin + shadowT.x * shadowRay.direction;
		
		//if we have a non-negative id, we've hit something
		if (shadowId >= 0 && primaryRay.lastHitObject >= 0) {
			vec4 tempColor;
			if (light.y > 0.0) {
				if (primaryRay.lastHitObject != 1
					&& shadowId == 1
					&& length(light.xyz-shadowRay.origin) > length(shadowHit-shadowRay.origin)) {
					//shade objects that are shadowed by the window
					vec3 nor = sceneWindow.xyz;
					shadowRay.lightColor = doLighting(primaryRay.origin, shadowHit, nor, light.xyz);
					shadowRay.lightColor *= 1.0 - vec4(Voronoi(shadowUV),1.0);
					shadowRay.transmittance = primaryRay.transmittance * glassTransmission;
					tempColor = mix(shadowRay.lightColor, col, shadowRay.transmittance);
				}
				else if (primaryRay.lastHitObject == 1) {
					//shade the back side of the window
					vec3 nor = -sceneWindow.xyz;
					shadowRay.lightColor = doLighting(primaryRay.origin, shadowHit, nor, light.xyz);
					shadowRay.lightColor *= vec4(Voronoi(shadowUV),1.0);
					shadowRay.transmittance = primaryRay.transmittance * glassTransmission;
					tempColor = mix(shadowRay.lightColor, col, shadowRay.transmittance);
				}
				if (primaryRay.lastHitObject != 1 && shadowId != 1) {
					//shadows for everything else in the scene
					shadowRay.lightColor = shadowRay.lightColor;
					shadowRay.transmittance = 0.5*primaryRay.transmittance;
					tempColor = mix(shadowRay.lightColor, col, 1.0-shadowRay.transmittance);
				}
			}
			else if (primaryRay.lastHitObject >= 0) {
				//before "sunrise"
				shadowRay.lightColor = shadowRay.lightColor;
				shadowRay.transmittance = 0.5*primaryRay.transmittance;
				tempColor = mix(shadowRay.lightColor, col, shadowRay.transmittance);
			}
			returnColor += tempColor;
		}
	}
	//if we use a number slightly higher than our iteration count,
	//then we get dark, but not black, shadows.  This also washes
	//out the color of the color of the glass, so it's kind of a
	//trade-off.
	col -= returnColor*(1.0/8.5);
}

// Function 397
vec3 computeColor(Ray ray, vec3 playerPos, vec3 SUN_DIR) {
    vec3 col = vec3(0.0);
    
    // Switch on matID
    // a return -> different/no lighting
    // no return -> default lighting
 	if (ray.matID == 0) {
    	return sky(ray.dir, SUN_DIR);
    } else if (ray.matID == 1){			// reflective balls
        col = vec3(0.8);
    } else if (ray.matID == 2){			// glass balls
        col = vec3(0.8);
    } else if (ray.matID == 3){
		col = centerSphere(ray);
    } else if (ray.matID == 4) {
        col = ground(ray);
    } else if (ray.matID == 5) {
        col = player(ray, playerPos);
    } else if (ray.matID == 6) {
        col = temple(ray);
    }
    
    // Default lighting
    float sunLight = directionalLightDiffuse(ray.nor, SUN_DIR);
    float sunShadow = softshadow(ray.pos, SUN_DIR, playerPos);
    
    float moonLight = directionalLightDiffuse(ray.nor, -SUN_DIR);
    float moonShadow = softshadow(ray.pos, -SUN_DIR, playerPos);
#if (SCENE != 1)
    col = col * (sunLight * sunShadow + 0.4 * moonLight * moonShadow);
#else
    vec3 lightPos = vec3(-50.0, 2.0, 0.0);
    float templeLight = pointLightDiffuse(ray.pos, ray.nor, lightPos, vec3(1.0, 0.0, 0.0));
    //float templeShadow = softshadow(ray.pos, normalize(ray.pos-lightPos));
    col = col * (sunLight * sunShadow + 0.6 * moonLight * moonShadow + templeLight);
#endif
    
    return col;
}

// Function 398
vec3 get_closest_attractor_color(vec2 pos) {
    float d2_A1 = d2(pos, A1);
    float d2_A2 = d2(pos, A2);
    float d2_A3 = d2(pos, A3);
    
    if (d2_A1 < d2_A2 && d2_A1 < d2_A3) return vec3(1.0 / d2_A1,0,0);
    if (d2_A2 < d2_A3) return vec3(0,1.0 / d2_A2,0);
    return vec3(0,0,1.0 / d2_A3);
}

// Function 399
vec3 doColoring(VoxelHit hit, vec3 rd)
{   
    // global position for non-repeating noise
    vec3 hitGlobal = vec3(hit.mapPos) + hit.hitRel + 0.5;
    float f1 = noise(hitGlobal*19.0);
    float f2 = noise(hitGlobal*33.0);
    float f3 = noise(hitGlobal*71.0);
    
    vec3 color = vec3(0.0);
    if (hit.terrainType == VOXEL_WATER) {
        color = vec3(0.4, 0.4, 0.8) * (0.8 + f1*0.1 + f2*0.05 + f3*0.05);
    } else if (hit.terrainType == VOXEL_EARTH) {
        color = vec3(1.0, 0.7, 0.3) * (f1*0.13 + f2*0.13 + f3*0.1 + 0.3);
    } else if (hit.terrainType == VOXEL_SAND) {
        color = vec3(1.0, 1.0, 0.6) * (f1*0.07 + f2*0.07 + f3*0.2 + 0.5);
    } else if (hit.terrainType == VOXEL_STONE) {
        color = vec3(0.5) * (f1*0.3 + f2*0.1 + 0.6);
    } else if (hit.terrainType == VOXEL_GRASS) {
        color = vec3(0.3, 0.7, 0.4) * (f1*0.1 + f3*0.1 + 0.6);
    }  else if (hit.terrainType == VOXEL_NONE) {
        color = vec3(0.0, 1.0, 1.0);
        color += vec3(5.0, 3.0, 0.0)*pow(max(dot(rd, SUN_DIRECTION), 0.0), 128.0);
    }
    
    float shadow = min(marchShadowCheck(hit), 1.0);
    float ambient = 1.0 - calcAmbientOcclusion(hit);
    float diffuse = max(dot(SUN_DIRECTION, hit.hitNormal), 0.0);
    diffuse = diffuse*(1.0-shadow);
    
    color *= diffuse * 0.6 + ambient * 0.4;
    
    vec2 occlusions = smoothstep(vec2(0.0), vec2(10.0, 3.0), hit.volAccum);
    color = mix(color, vec3(0.3, 0.3, 0.5), occlusions.y); // water
    color = mix(color, vec3(0.6), occlusions.x);           // cloud
    
    // blend with other intersection. will be fractional when anti-aliasing or underwater
    color *= hit.weight;
    
    return color;
}

// Function 400
vec3 colorRamp(float t)
{
    t = mix(0.0,0.85,t);
	float r = (9.0  * (1.0-t)*t*t*t);
  	float g = (15.0 * (1.0-t)*(1.0-t)*t*t);
  	float b = (8.5  * (1.0-t)*(1.0-t)*(1.0-t)*t);
    
    return vec3(r,g,b);
}

// Function 401
vec3 getBaseColor(int i)
{
    if (i == 0) return vec3(1.0, 0.4, 0.2);
    if (i == 1) return vec3(0.4, 1.0, 0.2);
    if (i == 2) return vec3(0.2, 1.0, 0.4);
    if (i == 3) return vec3(0.2, 0.4, 1.0);
    if (i == 4) return vec3(0.4, 0.2, 1.0);
    if (i == 5) return vec3(1.0, 0.2, 0.4);
 
    return vec3(1.);
}

// Function 402
float colormap_red(float x) {
    if (x < 0.0) {
        return 54.0 / 255.0;
    } else if (x < 20049.0 / 82979.0) {
        return (829.79 * x + 54.51) / 255.0;
    } else {
        return 1.0;
    }
}

// Function 403
vec4 color(float age) {
	float f = 1.0 - age * 0.05;
	#ifdef BLUE
	return vec4(0.2*f*f, 0.5*f*f+0.05, 0.5*f+0.4, min(f*2.0, 1.0));
	#else
	return vec4(0.5*f+0.4, 0.5*f*f+0.05, 0.2*f*f, min(f*2.0, 1.0));
	#endif
}

// Function 404
vec3 ColorGrade( in vec3 InColor )
{
    // Calculate the three offseted colors up-front
    vec3 OffShadows  = InColor + Shadows;
    vec3 OffMidtones = InColor + Midtones;
    vec3 OffHilights = InColor + Hilights;
    
    // Linearly interpolate between the 3 new colors, piece-wise
    return mix(
        // We pick which of the two control points to interpolate from based on which side of
        // 0.5 the input color channel lands on
        mix(OffShadows,  OffMidtones, InvLerp(vec3(0.0), vec3(0.5), InColor)), // <  0.5
        mix(OffMidtones, OffHilights, InvLerp(vec3(0.5), vec3(1.0), InColor)), // >= 0.5
        greaterThanEqual(InColor, vec3(0.5))
    );
}

// Function 405
vec3 colorchecker(vec2 uv) {
    vec2 xy = vec2(3.0, 2.0) + vec2(3.0, -2.0)*uv;
    vec2 ta = smoothstep(0.45, 0.42, 0.5 - fract(xy));
    vec2 tb = step(abs(0.5 - fract(xy)), vec2(0.45));
    vec3 ca = rsrgb(vec3(64.0/255.0));
    vec3 cb = rsrgb(colors[6*int(xy.y) + int(xy.x)]);
    vec3 cc = mix(vec3(0.02), cb, 0.2 + 0.8*ta.x*ta.y);
    return srgb(mix(ca, cc, tb.x*tb.y));
}

// Function 406
vec4 setColor (vec4 col)
{
#ifdef GAMMA
	col = vec4(pow(col.rgb, vec3(1.0/(GAMMA))), col.a);
#endif
	return col;
}

// Function 407
vec3 getColor(vec3 norm, vec3 pos, int objnr)
{
   vec3 col = objnr==WALL_OBJ?getWallColor(pos):(
              objnr==BRICKS_OBJ?getBrickColor(pos):(
              objnr==CHIMNEY_OBJ?getChimneyColor(pos, norm):sky_color(pos)));

   return col;
}

// Function 408
vec3 colorMap( int index, float v ) {
    vec3[14] arr;
    if (index == 0)
        arr = vec3[] ( 
                // brown
                vec3(69, 40, 60),
                vec3(102, 57, 49),
                vec3(102, 57, 49),
                vec3(102, 57, 49),
                vec3(143, 86, 59),
                vec3(143, 86, 59),
                vec3(143, 86, 59),
                vec3(180, 123, 80),
                vec3(180, 123, 80),
                vec3(180, 123, 80),
                // orange
                vec3(223, 113, 38),
                vec3(255, 182, 45),
                vec3(255, 182, 45),
                vec3(251, 242, 54)
                );
    else
        arr = vec3[] ( 
                // dark blue
                vec3(50,60,57),
                vec3(63,63,116),
                vec3(63,63,116),
                vec3(63,63,116),
                vec3(48,96,130),
                vec3(48,96,130),
                vec3(48,96,130),
                vec3(91,110,225),
                vec3(91,110,225),
                vec3(91,110,225),
                // light blue
                vec3(99,155,255),
                vec3(95,205,228),
                vec3(213,219,252),
                vec3(255)
                );
                
    return arr[ min(14, int(14. * v)) ] / 255.;
}

// Function 409
ColorGradingPreset colorGradingPresetLerp(const in ColorGradingPreset a, const in ColorGradingPreset b, const in float t){
  return ColorGradingPreset(mix(a.gain, b.gain, t),
                            mix(a.gamma, b.gamma, t),
                            mix(a.lift, b.lift, t),
                            mix(a.presaturation, b.presaturation, t),
                            mix(a.colorTemperatureStrength, b.colorTemperatureStrength, t),
                            mix(a.colorTemperature, b.colorTemperature, t),
                            mix(a.colorTemperatureBrightnessNormalization, b.colorTemperatureBrightnessNormalization, t));
}

// Function 410
vec3 wwcolor(in vec4 s)
{
    vec3 c = vec3(0.);
    if (STATE(s) == S_HEAD)
        c = vec3(.3,.5,1.);
    else if (STATE(s) == S_TAIL)
        c = vec3(1.,.2,.1);
    else if (STATE(s) == S_WIRE)
        c = vec3(1.,.6,.2);
    return c;
}

// Function 411
vec3 fadeColor(in vec3 xyz, in float fade, in float radius, in vec2 muv, in vec2 uv)
{
    float ratio = iResolution.x / iResolution.y;
    vec2 diff = abs(muv - uv);
    diff.x *= ratio;
    
    float fx = clamp(1. - length(diff) / radius, 0., 1.);
    int funcEnum = getIValue(fade, INTERP_FUNC_NUM);

    vec2 grad = getGradFunction(fx, funcEnum);

    xyz *= grad.x;
    
    return xyz;
}

// Function 412
vec3 adjust_out_of_gamut_remap(vec3 c)
{
    const float BEGIN_SPILL = 0.5;
    const float END_SPILL = 1.0;
    const float MAX_SPILL = 0.8; //note: <=1
    
    float lum = dot(c, vec3(1.0/3.0));
    //return mix( c, vec3(lum), min(lum,1.0));
    
    float t = (lum-BEGIN_SPILL) / (END_SPILL-BEGIN_SPILL);
    t = clamp( t, 0.0, 1.0 );
    //t = smoothstep( 0.0, 1.0, t );
    t = min(t, MAX_SPILL); //t *= MAX_SPILL;
    
    return mix( c, vec3(lum), t );
}

// Function 413
float colormap_red(float x) {
    if (x < 0.7) {
        return 4.0 * x - 1.5;
    } else {
        return -4.0 * x + 4.5;
    }
}

// Function 414
vec3 baseColorFor(float time) {
    float i = mod(floor(time / 10.0), 2.0);
    float progress = mod(time, 10.0);
    
    vec3 one = vec3(0.0 / 255.0, 209.0 / 255.0, 193.0 / 255.0);
    vec3 two = vec3(123.0 / 255.0, 222.0 / 255.0, 90.0 / 255.0);

    
    if (i == 0.0) {
        if (progress > 8.0) {
          return mix(one, two, (progress - 8.0) / 2.0);
        } else {
          return one;
        }
    }
    
    //if (i == 1.0) {
        if (progress > 8.0) {
          return mix(two, one, (progress - 8.0) / 2.0);
        } else {
          return two;
        }
    //}
}

// Function 415
vec3 ColorOfMetaball(int metaballNumber)
{
	vec3 metaColor = vec3(0.0);
	
	if(metaballNumber == 0)
	{
		metaColor = vec3(0.0, 1.0, 0.0);
	}
	else if(metaballNumber == 1)
	{
		metaColor = vec3(0.0, 0.0, 1.0);	
	}
	else if(metaballNumber == 2)
	{
		metaColor = vec3(1.0, 0.0, 0.0);	
	}
	
	return metaColor;
}

// Function 416
vec3 getChimneyColor(vec3 pos, vec3 norm)
{
    vec3 chcol = vec3(1.25) - 0.7*texture(iChannel2, 0.0007*(5.*pos.xy + cross(norm, pos).yz + cross(pos, norm).zx)).x;
    
    return chcol;
}

// Function 417
vec3 ColorGradeLUT3D_TriLinear(sampler2D _texLUT,vec3 color) {
    vec3 coord = color*vec3(1.,1., _LUT_Size - 1.);
    coord.x = fract(coord.x) / _LUT_Size;
    float deltal = fract(coord.z);
    coord.x += floor(coord.z) / _LUT_Size;
    vec3 frontCol = texture(_texLUT, coord.xy*_LUT_2D_Size/R).rgb;
    coord.x += 1. / _LUT_Size;
    vec3 backCol = texture(_texLUT, coord.xy*_LUT_2D_Size/R).rgb;
    return mix(frontCol,backCol,deltal);
}

// Function 418
vec3 outputColor(float aFinalHue) { return hsv2rgb_smooth(vec3(aFinalHue, 1.0, 1.0)); }

// Function 419
vec4 colorLookup(in float x)
{
	const vec4 yellow = vec4(1., 1., 0., 1.);
    const vec4 red = vec4(1., 0., 0., 1.);
    const vec4 black = vec4(vec3(0.), 1.);
    const vec4 white = vec4(1.);
    const vec3 bound = vec3(0.06, 0.11, 0.15);
    x /= 5.;
    if (x < bound.z) {
        if (x < bound.x) {
            return mix(white, yellow, x * 1. / bound.x);
        }
        if (x < bound.y) {
            return mix(yellow, red, (x-bound.x) * 1./ (bound.y-bound.x));
        }
        return mix(red, black, (x-bound.y) / (bound.z-bound.y));
    } else {
        return black;
    }
}

// Function 420
vec3 RayTracePixelColor (in vec3 rayPos, in vec3 rayDir)
{   
    vec4 bestRayHitInfo = vec4(1000.0, 0.0, 0.0, 0.0);
    vec3 rayHitDiffuse = vec3(1.0);
    vec3 additiveColor = vec3(0.0);
    
    vec3 ret = vec3(0.0);
           
    // see if we've hit the platform and remember if we have
    vec2 uv;    
    vec4 rayInfo = RayIntersectBox(rayPos + vec3(0.0, 1.51, 0.0), rayDir, vec3(1.0, 1.0, 1.0), uv);
    if (rayInfo.x >= 0.0 && rayInfo.x < bestRayHitInfo.x)
    {
        bestRayHitInfo = rayInfo;
        rayHitDiffuse = Checkerboard(uv);
    }
    
    // if we've hit the main object, and it's closer than the platform
    rayInfo = RayIntersectObject(rayPos, rayDir);
    if (rayInfo.x >= 0.0 && rayInfo.x < bestRayHitInfo.x)
    {       
        // light the surface of the ball a bit
        additiveColor += LightPixel(rayPos, rayDir, OBJECT_DIFFUSE, rayInfo.yzw, OBJECT_SPECPOWER, false);
        
        // move the ray to the intersection point
        rayPos += rayDir * rayInfo.x;    
        
        // calculate how much to reflect or transmit (refract or diffuse)
        float reflectMultiplier = FresnelReflectAmount(REFRACTIVE_INDEX_OUTSIDE, REFRACTIVE_INDEX_INSIDE, rayDir, rayInfo.yzw);
        float refractMultiplier = 1.0 - reflectMultiplier;
        
        // get reflection color
        #if DO_REFLECTION
        	vec3 reflectDir = reflect(rayDir, rayInfo.yzw);
        	ret += GetSceneRayColor(rayPos + reflectDir*0.001, reflectDir) * reflectMultiplier;
        #endif
        
        // get refraction color
        #if DO_REFRACTION
        	vec3 refractDir = refract(rayDir, rayInfo.yzw, REFRACTIVE_INDEX_OUTSIDE / REFRACTIVE_INDEX_INSIDE);
        	ret += GetObjectInternalRayColor(rayPos + refractDir*0.001, refractDir) * refractMultiplier;
        #endif
        
        return ret + additiveColor;
    }
    // else we missed the object, so return either the skybox color, or the platform color, as appropriate
    else
    {
        if (bestRayHitInfo.x == 1000.0)    
            return texture(iChannel0, rayDir).rgb;
        else
        {
            // move the ray to the intersection point (so we can shadow) and light the pixel
        	rayPos += rayDir * bestRayHitInfo.x;    
            return LightPixel(rayPos, rayDir, rayHitDiffuse, bestRayHitInfo.yzw, 100.0, true);    
        }
    }
}

// Function 421
float watercolor (vec2 p) {
       p*=5.;
       vec2 q = vec2(0.);
       q.x = fbm(p);
       q.y = fbm( p + vec2(1.0));
       vec2 r = vec2(0.);
       r.x = fbm( p + 1.0*q + vec2(1.7,9.2));
       r.y = fbm( p + 1.0*q + vec2(8.3,2.8));
       float f = fbm(p+r);
       return clamp(f,0.,1.);
}

// Function 422
void DecodeWaterColor(float data)
{
    WaterColor.r = float(int(data) & 63) / 64.0;
    WaterColor.g = float((int(data) >> 6) & 63) / 64.0;
	WaterColor.b = float((int(data) >> 12) & 63) / 64.0;
}

// Function 423
vec4 colorAndDepth(vec3 pos, vec3 dir){
    vec3 n;
    if(!trace(pos, dir, n))
        return vec4(background(dir), RenderDistance);
    return vec4(directLight(pos, n)+ambientLight(pos), length(CamPos - pos));
}

// Function 424
void mixColor(vec4 col, float alpha)
{
    fcol = vec4(mix(fcol.rgb, col.rgb, alpha * col.a), 1.0);
}

// Function 425
vec3 rainbowColor(in vec3 ray_dir) 
{ 
    RAINBOW_DIR = normalize(RAINBOW_DIR);   
		
    float theta = degrees(acos(dot(RAINBOW_DIR, ray_dir)));
    vec3 nd = clamp(1.0 - abs((RAINBOW_COLOR_RANGE - theta) * 0.2), 0.0, 1.0);
    vec3 color = _smoothstep(nd) * RAINBOW_INTENSITY;
    
    return color * max((RAINBOW_BRIGHTNESS - 0.75) * 1.5, 0.0);
}

// Function 426
vec3 borderColor(float x0, float x1, vec2 uv, 
                 float leftVisible, float rightVisible) {

    vec3 edgeCol = vec3(0.05);
    vec3 white = vec3(1.);
    float thick = 0.03;
   	
    // the exterior side of the left border
	float outside = (step(uv.x, x0-thick) + step(x0, uv.x));
    vec3 borderCol = mix(white, edgeCol, smoothstep(x0, x0-thick, uv.x)) 
    	* (1. - outside) * leftVisible;
        
    // the interior side of the left border
    outside = (step(uv.x, x0) + step(x0+thick, uv.x));
    borderCol += mix(edgeCol, white, smoothstep(x0+thick, x0, uv.x)) 
    	* (1. - outside)* leftVisible;      
    
    // the exterior side of the right border
    outside = (step(uv.x, x1) + step(x1+thick, uv.x));        
    borderCol += mix(white, edgeCol, smoothstep(x1, x1+thick, uv.x))
        * (1.- outside) * rightVisible ;
    
    // the interior side of the right border
    outside = (step(uv.x, x1-thick) + step(x1, uv.x));            
    borderCol += mix(edgeCol, white, smoothstep(x1-thick, x1, uv.x))
    	* (1.- outside) * rightVisible;
    
    return borderCol;
}

// Function 427
vec3 colorBurn(in vec3 s, in vec3 d )
{
	return 1.0 - (1.0 - d) / s;
}

// Function 428
vec3 ReturnColor_3(float r1, float r2){
 return (abs(_Color4 -_Color1) * (1.-r2)+  abs(_Color2 - _Color1) * r1 + abs(_Color3 - _Color1) *r2)/2.;
  
}

// Function 429
vec3 GetMacBethColorCOLUMN_COUNT1(const in float yDist)
{
    float compareY = LINE_COUNT;
    
    if(yDist > --compareY)
        return DARK_SKIN;
	else if(yDist > --compareY)
		return ORANGE;
	else if(yDist > --compareY)
		return BLUE;
	else
		return WHITE;
}

// Function 430
vec3 getSeaColor(vec3 p, vec3 n, vec3 l, vec3 eye, vec3 dist) {  
    float fresnel = clamp(1.0 - dot(n,-eye), 0.0, 1.0);
    fresnel = pow(fresnel,3.0) * 0.65;
        
    vec3 reflected = getSkyColor(reflect(eye,n));    
    vec3 refracted = SEA_BASE + diffuse(n,l,80.0) * SEA_WATER_COLOR * 0.12; 
    
    vec3 color = mix(refracted,reflected,fresnel);
    
    float atten = max(1.0 - dot(dist,dist) * 0.001, 0.0);
    color += SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18 * atten;
    
    color += vec3(specular(n,l,eye,60.0));
    
    return color;
}

// Function 431
vec3 testcolor(float x, float l_nm) {
  vec3 color1 = (wavelength_to_srgbl(l_nm)+.85)*.3; // no negative values
  vec3 color2 = wavelength_to_srgbl(l_nm)*.3;       // no white background
  vec3 color = mix(color2,color1,x*x*x*x);

  color = tonemap(color);
#if 0
  // hack - closer to appearance of desired result (still not ideal)
  color1 = tonemap(color1);
  color2 = tonemap(color2);
  color = mix(color2,color1,x*x*x*x);
#endif
  return color;
}

// Function 432
float color_to_val_5(in vec3 color) {
    return color_to_val_cmy(color) + 2.0 * color_to_val_2(color);
}

// Function 433
vec3 surface_color(vec3 p)
{
    float color1 = length(sin(p))/2.0;
    return vec3(color1,color1/1.8+sin(length(p)/10.0)/20.0,color1/2.0);
}

// Function 434
vec3 getColorOklab(float rawLightness, float rawChroma, float hue) {
    vec3 lch = vec3(rawLightness, rawChroma, hue);
    vec3 oklab = lchToLab(lch);
    return oklabToLinearSrgb(oklab);
}

// Function 435
void contributeColor( in vec4 col, inout vec4 sum )
{
#ifndef SHOW_BOUNDS
    // alpha blend in contribution
    sum = sum + col*(1.0 - sum.a);
    sum.a+=0.15*col.a;
#else
   	sum = max(sum, col);
#endif
}

// Function 436
vec3 GetColor(const vec3 inputOffset) {
	return 0.5f + 0.5f * cos(iTime + (inputOffset * 4.0f) + vec3(0, 2, 4));
}

// Function 437
vec3 InvertWithColor(vec3 color)
{
   	color = ColourToYPbPr2(color);

  	color.x = 1.0 - color.x;

  	return YPbPrToColour(color);
}

// Function 438
float diskColorg(in vec2 uv, vec2 offset)
{
    uv = uv - smoothstep(0.01,1.8,texture(iChannel0, (uv*1.0 - vec2(iTime /3.0,(iTime) /8.0)) + offset).r) * 0.3;
    
    float d = length(uv)-RADIUS;
    return smoothstep(0.01,0.015,d);
}

// Function 439
float gridColor(vec2 p, float size, float thicknessx, float thicknessy) {
  float dl1 = sdLines(p.x, size);
  float dl2 = sdLines(p.y, size);

  float f1 = smoothstep(thicknessx * 0.25, thicknessx, dl1);
  float f2 = smoothstep(thicknessy * 0.25, thicknessy, dl2);
  return f1 * f2;
}

// Function 440
vec3 adjust_out_of_gamut_maxcomp(vec3 c)
{
    const float BEGIN_SPILL = 1.0;
    const float END_SPILL = 4.0;
    const float MAX_SPILL = 0.9; //note: <=1
    
    float mc = max(c.r, max(c.g, c.b));
    float t = MAX_SPILL * smootherstep( 0.0, END_SPILL-BEGIN_SPILL, mc-BEGIN_SPILL );
    return mix( c, vec3(mc), t);
}

// Function 441
vec3 container_color(vec3 pos)
{
    return vec3(0.65);
}

// Function 442
vec3 GetAmbientSkyColor()
{
    return SKY_AMBIENT_MULTIPLIER * GetBaseSkyColor(vec3(0, 1, 0));
}

// Function 443
vec3 heatColorMap(float t)
{
    t *= 4.;
    return clamp(vec3(min(t-1.5, 4.5-t), 
                      min(t-0.5, 3.5-t), 
                      min(t+0.5, 2.5-t)), 
                 0., 1.);
}

// Function 444
vec4 getColor(float x0, float x1, vec2 uv, vec4 intCol, vec4 extCol,
              vec4 fragColor) {
      
    vec4 rightCol = vec4(0.);
    vec4 leftCol = vec4(0.);
        
    leftCol = borderColor(x0, x1, uv, 1., 0., intCol, extCol);       
    rightCol = borderColor(x0, x1, uv, 0., 1., intCol, extCol);
   
    if (leftCol != vec4(0.))
        return leftCol;
    else if (rightCol != vec4(0.))
   		return rightCol;                 
   
    return fragColor;   
}

// Function 445
vec3 decodeColor(vec3 color){
	return color * 1000.0;
}

// Function 446
vec3 calcColor(float m) {
    return 0.45 + 0.35*sin( vec3(0.05,0.08,0.10)*(m-1.0) );
}

// Function 447
vec3 convert_to_debug_color(float x)
{        
    const float num_steps = 40.;
    // we want 1 to be super hot (red) and 0. to be super cool (purple)
    float remapx = clamp(1. - x, 0., 1.);

    // avoid looping back on the hue wheel since we want to differentiate red as 1
    // purple as 0.
    float stepx = .8 * floor(num_steps * remapx)/num_steps;    
    float foot = fract(x * 10.);
    
    float fmask = 1.;
    
    vec3 hsl = vec3(stepx, .95, .5 - .1 * mod(stepx * num_steps, 2.) );
    return fmask * hsl_to_rgb(hsl);       
}

// Function 448
vec3 getColor(vec3 dir) {
    
    float dist = sphere( dir, obj_pos, obj_size );    
    
    if(dist > 0.0) {
                        
    	vec3 point = dir * dist;
        
        // Normal
    	vec3 N = normalize(point - obj_pos);  // Normal
        vec3 V = normalize( -dir );           // View vector        

        vec3 specular_color = vec3( 0., 0., 0. );
        vec3 diffuse_refl = vec3( 1., 1., 0.5);
        
        float k_spec = 0.4;
        float k_diff = 0.2;
        float spec_alpha = 30.;
        
        vec3 diffuse_color  = vec3( 0., 0., 0. );
        for( int l = 0; l < NO_LIGHTS; l++ ) {
	        vec3 L = normalize(light_pos[l]);
	        vec3 Lr = reflect(L, N);
	        float NdotL = clamp(dot(L,  N), 0.0, 1.0);
            
	        specular_color += k_spec * light_rgb[l] * vec3( 1., 1., 1. ) * pow( max( dot(Lr, V), 0.), spec_alpha );
	        diffuse_color  += k_diff * light_rgb[l] * diffuse_refl * NdotL;
        }        
        
        float exposure = 0.2;
        vec3 final_color = toneMap( diffuse_color + specular_color, exposure );
        
		return final_color;

    } else {      
        
        return texture(iChannel0, dir).xyz;
    }
}

// Function 449
vec4 getColor(vec2 p){
    float y = 2.-(p.x - (iResolution.x -100.))/50.;
	p = vec2(p.y/iResolution.y,clamp(y,0.,1.));
    vec3 rgb = hsv2rgb(vec3(p.x,1.,p.y));
    return vec4(mix(rgb,vec3(1.),clamp(y-1.,0.,1.)),1.);
}

// Function 450
vec3 TAA_ColorSpace( vec3 color )
{
    return Tonemap(color);
}

// Function 451
float colormap_green(float x) {
    if (x < 0.5) {
        return 4.0 * x - 0.5;
    } else {
        return -4.0 * x + 3.5;
    }
}

// Function 452
vec3 surface_color(vec3 p)
{
    p /= scale*10.0;
    float color1 = length(sin(p/100.0))/2.0;
    return vec3(color1,color1/1.8+sin(length(p)/10.0)/20.0,color1/2.0);
}

// Function 453
vec4 getColorFromPalette( in int palette_index )
{
	int int_color = palette[ palette_index ];
	return vec4( float( int_color & 0xff ) / 255.0,
				float( ( int_color >> 8 ) & 0xff) / 255.0,
				float( ( int_color >> 16 ) & 0xff) / 255.0,
				0 );
}

// Function 454
vec4 getLensColor(float x){
  // color gradient values from http://vserver.rosseaux.net/stuff/lenscolor.png
  // you can try to curve-fitting it, my own tries weren't optically better (and smaller) than the multiple mix+smoothstep solution 
  return vec4(vec3(mix(mix(mix(mix(mix(mix(mix(mix(mix(mix(mix(mix(mix(mix(mix(vec3(1.0, 1.0, 1.0),
                                                                               vec3(0.914, 0.871, 0.914), smoothstep(0.0, 0.063, x)),
                                                                           vec3(0.714, 0.588, 0.773), smoothstep(0.063, 0.125, x)),
                                                                       vec3(0.384, 0.545, 0.631), smoothstep(0.125, 0.188, x)),
                                                                   vec3(0.588, 0.431, 0.616), smoothstep(0.188, 0.227, x)),
                                                               vec3(0.31, 0.204, 0.537), smoothstep(0.227, 0.251, x)),
                                                           vec3(0.192, 0.106, 0.286), smoothstep(0.251, 0.314, x)),
                                                       vec3(0.102, 0.008, 0.341), smoothstep(0.314, 0.392, x)),
                                                   vec3(0.086, 0.0, 0.141), smoothstep(0.392, 0.502, x)),
                                               vec3(1.0, 0.31, 0.0), smoothstep(0.502, 0.604, x)),
                                           vec3(1.0, 0.49, 0.0), smoothstep(0.604, 0.643, x)),
                                       vec3(1.0, 0.929, 0.0), smoothstep(0.643, 0.761, x)),
                                   vec3(1.0, 0.086, 0.424), smoothstep(0.761, 0.847, x)),
                               vec3(1.0, 0.49, 0.0), smoothstep(0.847, 0.89, x)),
                           vec3(0.945, 0.275, 0.475), smoothstep(0.89, 0.941, x)),
                       vec3(0.251, 0.275, 0.796), smoothstep(0.941, 1.0, x))),
                    1.0);
}

// Function 455
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

// Function 456
vec3 colorFloor(vec3 P, vec3 I, vec3 N, float type, float K, float solarPhase)
{
    vec3 L = mix(_moon.d, _light.d, solarPhase);
    vec3 color = vec3(0.0);
    if(type == TYPE_FLOOR)
    {
        float ao = texture(iChannel0, P.xz*FLOOR_TEXTURE_FREQ).r;
        vec3 diff = texture(iChannel1, P.xz*FLOOR_TEXTURE_FREQ).rgb;
        
        float algae_mask = max(0.0, texture(iChannel1, P.xz).r-0.5);
        
        diff = mix(diff, COLORALGAE, algae_mask);
        float roughness = mix(0.2, 1.0, algae_mask);
        
        color = brdf(
            K, // Ks
            K, // Kd
            roughness, // roughness
            1.0, // opacity
            diff,//vec3(1.0), // specular color
            diff, // diffuse color
            I, // I
            N, // N
            -L // L
            )*ao;
    }
    return color;
}

// Function 457
vec4 trueColorEdge(float stepx, float stepy, vec2 center, mat3 kernelX, mat3 kernelY) {
	vec4 edgeVal = edge(stepx, stepy, center, kernelX, kernelY);
	return edgeVal * texture(iChannel0,center);
}

// Function 458
vec4 glyph_color(uint glyph, ivec2 pixel, float variation)
{
    pixel &= 7;
    pixel.y = 7 - pixel.y;
    int bit_index = pixel.x + (pixel.y << 3);
    int bit = glyph_bit(glyph, bit_index);
    int shadow_bit = min(pixel.x, pixel.y) > 0 ? glyph_bit(glyph, bit_index - 9) : 0;
    return vec4(vec3(bit > 0 ? variation : .1875), float(bit|shadow_bit));
}

// Function 459
vec3 getGlowingColor(
    vec2 uv,
    float glowRange,
    float glowStart,
    float brightness
) {
    vec3 color = vec3(.0, glowStart / (20. / glowRange) * 3., .05);
    float farFromCenter = length(uv);
    
    color += brightness/farFromCenter;
    
    return color;
}

// Function 460
vec3 getSceneColor(in vec2 uv )
{
    vec4 fragColor;
//	vec2 uv = fragCoord.xy / iResolution.xy;
    
    if(uv.y > .666)
		fragColor = vec4(gradient(uv.x, color3, color2, colorC, colorD),1);
    else if(uv.y > .333)
		fragColor = vec4(gradient(uv.x, color2, color3, color4, color5, color6),1);
    else
		fragColor = vec4(gradient(uv.x, color0, color1, color2, color3, color4, color5, color6, color7, color8, color9, colorA, colorB, colorC, colorD, colorE, colorF),1);


    // post-processing courtesy of IQ ( https://www.shadertoy.com/view/ll2GD3 )

    // band
    float f = fract(uv.y*3.0);
    // borders
    fragColor.rgb *= smoothstep( 0.49, 0.47, abs(f-0.5) );
    // shadowing
    fragColor.rgb *= 0.5 + 0.5*sqrt(4.0*f*(1.0-f));
    
    return fragColor.rgb;
}

// Function 461
vec3 getParticleColor(int partnr, float pint)
{
   vec2 pos = vec2(mod(float(partnr+1), iResolution.x)/(iResolution.x+1.), (50. + float(partnr)/(iResolution.x))/(iResolution.y+1.));
   return (pint*texture(iChannel0, pos)).xyz;  
}

// Function 462
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

// Function 463
vec3 colormap(float value) {
	float maxv = ClampLevel;
	vec3 c1,c2;
	float t;
	if (value < maxv / 3.) {
		c1 = vec3(0.);   	 c2 = vec3(1.,0.,0.); 	t =  1./3.;
	} else if (value < maxv * 2. / 3.) {
		c1 = vec3(1.,0.,0.); c2 = vec3(1.,1.,.5);	t =  2./3. ;
	} else {
		c1 = vec3(1.,1.,.5); c2 = vec3(1.);      	t =  1.;
	}
	t = (t*maxv-value)/(maxv/3.);
	return t*c1 + (1.-t)*c2;
}

// Function 464
vec3 getColorZcam(float rawLightness, float rawChroma, float hue) {
    ZcamViewingConditions cond = getZcamCond();

    vec3 jch = vec3(rawLightness * 100.0, rawChroma * 170.0, hue);

    if (CLIP_ZCAM) {
        return clipZcamJchToLinearSrgb(jch, cond);
    } else {
        return zcamJchToLinearSrgb(jch, cond);
    }
}

// Function 465
vec3 colorTemperatureToRGB(const in float temperature){
  // Values from: http://blenderartists.org/forum/showthread.php?270332-OSL-Goodness&p=2268693&viewfull=1#post2268693   
  mat3 m = (temperature <= 6500.0) ? mat3(vec3(0.0, -2902.1955373783176, -8257.7997278925690),
	                                      vec3(0.0, 1669.5803561666639, 2575.2827530017594),
	                                      vec3(1.0, 1.3302673723350029, 1.8993753891711275)) : 
	 								 mat3(vec3(1745.0425298314172, 1216.6168361476490, -8257.7997278925690),
   	                                      vec3(-2666.3474220535695, -2173.1012343082230, 2575.2827530017594),
	                                      vec3(0.55995389139931482, 0.70381203140554553, 1.8993753891711275)); 
  return mix(clamp(vec3(m[0] / (vec3(clamp(temperature, 1000.0, 40000.0)) + m[1]) + m[2]), vec3(0.0), vec3(1.0)), vec3(1.0), smoothstep(1000.0, 0.0, temperature));
}

// Function 466
vec3 GetMacBethColorCOLUMN_COUNT3(const in float yDist)
{
    float compareY = LINE_COUNT;
    
    if(yDist > --compareY)
        return BLUESKY;
	else if(yDist > --compareY)
		return MODERATERED;
	else if(yDist > --compareY)
		return RED;
	else
		return NEUTRAL65;
}

// Function 467
vec3 skyColor(in vec3 ray){
    vec3 col = vec3(0.);
    col += vec3( max((ray.x+1.)/2.*(-4.*ray.y+1.)/2.,0.),.1*(1.-ray.y),.2*(1.-ray.y) );
    return col;
}

// Function 468
vec4 colorMap() {
    if (rayPos.y <= -9.8) {//ground
        return texture(iChannel1,rayPos.xz*0.1);
    }
    
    
    //cube
    vec4 samp = texture(iChannel2,normalize(rayPos));
    return samp*0.3+
        samp*max(0.0,dot(lightDir,distMapNormal(rayPos)));
}

// Function 469
vec3 coloredBox(vec2 st, vec3 color, vec4 r){
        
    float avgColor = (color.r+color.g+color.b)/3.;
	float w = (sin(iTime*avgColor*1.)+1.1)*.1;
    color *= 1.-w*5.;
    return vec3(rect(st, vec4(r.x+w, r.y+w, r.z-w, r.w-w)))*color;
}

// Function 470
vec3 getSphereColor( vec2 grid ) {
	float m = hash12( grid.yx ) * 12.;
    return vec3(1.-m*0.08, m*0.03, m*0.06);
}

// Function 471
vec3 DiffuseColor (in vec3 pos){// checkerboard pattern
 return vec3(mod(floor(pos.x*10.)+floor(pos.z * 10.),2.)< 1.?1.:.4);}

// Function 472
vec3 GetColorForRay(in vec3 startRayPos, in vec3 startRayDir, inout uint rngState)
{
    vec3 ret = vec3(0.0f, 0.0f, 0.0f);
    vec3 colorMultiplier = vec3(1.0f, 1.0f, 1.0f);
    
    vec3 rayPos = startRayPos;
    vec3 rayDir = startRayDir;
       
    for (int i = 0; i <= c_numBounces; ++i)
    {
        SRayHitInfo hitInfo;
		hitInfo.hitAnObject = false;
        hitInfo.dist = c_superFar;
        
        // ray trace first, which also gives a maximum distance for ray marching
        RayTraceScene(rayPos, rayDir, hitInfo);
        RayMarchScene(rayPos, rayDir, hitInfo);
        
        if (!hitInfo.hitAnObject)
        {
            // handle ray misses
            ret += texture(iChannel1, rayDir).rgb * c_skyboxMultiplier * colorMultiplier;
            break;
        }
                      
        // update the ray position
        rayPos += rayDir * hitInfo.dist;
               
        // get the material info if it was a ray marched object
        if (hitInfo.rayMarchedObject)
			hitInfo = TestSceneMarch(rayPos);       
                
		// add in emissive lighting
        ret += hitInfo.material.emissive * colorMultiplier;
        
        // figure out whether we are going to shoot out a specular or diffuse ray.
        // If neither, exit
        float diffuseLength = length(hitInfo.material.diffuse);        
        float specularLength = length(hitInfo.material.specular);
        if (diffuseLength + specularLength == 0.0f)
            break;
        float specularWeight = specularLength / (diffuseLength + specularLength);       
        float doSpecular = float(RandomFloat01(rngState) < specularWeight);
       
        // set up the next ray direction
        float roughness = mix(1.0f, hitInfo.material.roughness, doSpecular);
        vec3 reflectDir = reflect(rayDir, hitInfo.normal);
        vec3 randomDir = RandomUnitVector(rngState);
        rayDir = normalize(mix(reflectDir, randomDir, roughness));
        
        if (dot(rayDir, hitInfo.normal) < 0.0f)
            rayDir *= -1.0f;        
        
        // move the ray away from the surface it hit a little bit
        rayPos += hitInfo.normal * c_rayPosNormalNudge;
        
        // Make all future light affected be modulated by either the diffuse or specular reflection color
        // depending on which we are doing.
        // Attenuate diffuse by the dot product of the outgoing ray and the normal (aka multiply diffuse by cosine theta or N dot L)
        float NdotL = dot(hitInfo.normal, rayDir);
        colorMultiplier *= mix(hitInfo.material.diffuse * NdotL, hitInfo.material.specular, doSpecular);        
    }
    
    return ret;
}

// Function 473
vec3 adjust_out_of_gamut_lerp(vec3 c)
{
    float lum = dot(c, vec3(1.0/3.0));
    float t = smoothstep( 0.0, 1.0, lum );
    return mix( c, vec3(lum), t);
}

// Function 474
vec3 colorFunction (vec3 p) 
{
	const vec3 colorA = vec3(0.6, 0.3, 0.1);
	const vec3 colorB = vec3(0.4, 0.6, 0.6);
	const vec3 colorC = vec3(0.1, 0.8, 0.1);
	const vec3 colorD = vec3(0.2, 0.2, 1.0);
	const vec3 colorE = vec3(1.2, 1.2, 1.0);
	return max(0.0,2.0 -ballA(p))*colorA*0.5
		 + max(0.0,2.0 -ballB(p))*colorB*0.5
		 + max(0.0,2.0 -ballC(p))*colorC*0.5
		 + max(0.0,2.0 -ballD(p))*colorD*0.5;
		 + max(0.0,2.0-torusA(p))*colorE*0.5;
}

// Function 475
vec3 color(in vec3 src, in vec3 dst)
{
    vec3 dstHSL = rgb2hsl(dst);
    vec3 srcHSL = rgb2hsl(src);
    return hsl2rgb(vec3(srcHSL.rg, dstHSL.b));
}

// Function 476
vec3 adjust(vec3 YIQ, float H, float S, float B) {
    mat3 M = mat3(  B,      0.0,      0.0,
                  0.0, S*cos(H),  -sin(H), 
                  0.0,   sin(H), S*cos(H) );
    return M * YIQ;
}

// Function 477
vec3 colorize(vec3 s, vec3 d, float dist, vec2 uv)
{
	vec3 p = s+d*dist;  
    float oid = mapid(s+d*dist);
    
    
    //------------------------	Scrolling plane
    PLANEMAT;
    vec3 retcol = texture(iChannel0, vec2( p.xz/10.0 ) + vec2(0,iTime) ).rgb;

    retcol = mix(retcol, vec3(0), step(1000.0, dist));
    //------------------------	Boxes
    BOXMAT;
    vec3 relp = rotMat*(p-vec3(0,0,0)) / (vec3(1) + vec3(texture(iChannel0, vec2(floor(iTime*4.01)/4.01, mod(floor(iTime*4.0)/4.0, 1.0) + floor(iTime*2.183)/2.183)).rgb));
    vec3 boxcol = vec3(((length(relp.xy) - 0.5)/abs(relp.z)) * ((length(relp.xz)-0.5)/abs(relp.y)) * ((length(relp.yz) - 0.5)/abs(relp.x)),0,0);
	//float a = (abs(relp.x)+abs(relp.y)+abs(relp.z))/3.0;
    float a = smoothstep(0.8, 1.0, abs(relp.x)) + smoothstep(0.8, 1.0, abs(relp.y)) + smoothstep(0.8, 1.0, abs(relp.z));
    boxcol = vec3(smoothstep(0.5,0.5,a/3.0)) * vec3(1.0,0.4,0);
    
    retcol = mix(retcol, boxcol, step(1.0, oid));
    
    return retcol;
}

// Function 478
vec3 gamma_adjust_linear( vec3 rgbIn, float GAMMA, float PIVOT)
{
float SCALAR = PIVOT / pow( PIVOT, GAMMA);
vec3 rgbOut = rgbIn;
if (rgbIn.x > 0.0) rgbOut.x = pow( rgbIn.x, GAMMA) * SCALAR;
if (rgbIn.y > 0.0) rgbOut.y = pow( rgbIn.y, GAMMA) * SCALAR;
if (rgbIn.z > 0.0) rgbOut.z = pow( rgbIn.z, GAMMA) * SCALAR;
return rgbOut;
}

// Function 479
vec3 GetMaterialsColor(RayInfo r, int matID
){if(matID>7)return vec3(0)
 ;float fakeOA = pow((1.-float(r.iter)/float(maxStepRayMarching)),.7)
 ;return rainbow((sqrt(5.)*.5+.5)*float(matID*2))*fakeOA
   ;//the *2 is flavor, it shifts most colors into something more like the below
    /*
    switch(matID){  //switch() is VERY incompatible syntax
    case 0:return vec3(4,4,4)*fakeOA*.1;
    case 1:return vec3(8,4,4)*fakeOA*.1;
    case 2:return vec3(4,8,4)*fakeOA*.1;
    case 3:return vec3(4,4,8)*fakeOA*.1;
    case 4:return vec3(8,8,4)*fakeOA*.1;
    case 5:return vec3(4,8,8)*fakeOA*.1;
    case 6:return vec3(8,4,8)*fakeOA*.1;
    case 7:return vec3(9,9,9)*fakeOA*.1;
    case 8:return vec3(0,0,0)*fakeOA*.1;
	}/**/
 ;}

// Function 480
vec3 skyColor( in vec3 rd )
{
    vec3 sundir = normalize( vec3(.0, .1, 1.) );
    rd.y += .02;
    float yd = min(rd.y, 0.), clouds = 0.;
    rd.y = max(rd.y, 0.);
    
    vec3 col = vec3(0.);
    
    col += vec3(.4, .4 - exp( -rd.y*20. )*.3, .0) * exp(-rd.y*9.);
    col += vec3(.4, .6, .7) * (1. - exp(-rd.y*8.) ) * exp(-rd.y*.9);
    
    col = mix(col*1.2, vec3(.3),  1.-exp(yd*100.));
    
    //Clouds
    vec2 pclouds = rd.xz/rd.y;
    clouds += fbm(pclouds*.01);
    col += .1*(clouds*2.-1.);
    
    //Synchronized raindow!
    vec3 raindow = clamp( abs(mod(-rd.x*20.*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );
    raindow = raindow*raindow*(3.0-2.0*raindow);;
    col += raindow * max(-abs(rd.x+.715)+.025,0.)*100.*pow(texture(iChannel3, vec2(mod(rd.x*5.,1.),0.)).r,2.);
    
    return clamp(col,vec3(0.),vec3(1.))*2.;
}

// Function 481
vec3 gradcolors(float t) {
	float over = cos(t* HALF_PI);
	float under = cos(t* HALF_PI+ HALF_PI)+ 1.0;
	return vec3(over, under, over);
}

// Function 482
vec4 ColorConvolution(vec2 UV, vec2 InverseRes)
{
	vec3 InPixel = NTSCCodec(UV, InverseRes).rgb;
	
	// Color Matrix
	float RedValue = dot(InPixel, RedMatrix);
	float GrnValue = dot(InPixel, GrnMatrix);
	float BluValue = dot(InPixel, BluMatrix);
	vec3 OutColor = vec3(RedValue, GrnValue, BluValue);
	
	// DC Offset & Scale
	OutColor = (OutColor * ColorScale) + DCOffset;
	
	// Saturation
	float Luma = dot(OutColor, Gray);
	vec3 Chroma = OutColor - Luma;
	OutColor = (Chroma * Saturation) + Luma;
	
	return vec4(OutColor, 1.0);
}

// Function 483
vec4 getColor(in vec2 coord) {
    
    return texture(iChannel0, vec2(coord.x + sin(iTime + coord.y / 20.0) * 3.0, coord.y + sin(iTime + coord.x / 20.0) * 3.0) / iResolution.xy);
}

// Function 484
void ColorAndNormal(vec3 hit, inout vec4 mcol, inout vec3 normal, vec2 tRoom, inout vec2 mref, inout float t, const int id)
{
	if(t == tRoom.y)
	{            
		mref = vec2(0.0,0.0);
        normal =-normalForCube(hit, box0);   
        if(normal.x>0.0)
        { 
            mcol.xyz = vec3(0.95,0.05,0.05);
        } 
        else if(normal.x<0.0)
        { 
            mcol.xyz = vec3(0.05,0.95,0.05);
        } 
	}     
	else   
	{
        	 if(id==0) {normal = normalForSphere(hit, sfere[0]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==1) {normal = normalForSphere(hit, sfere[1]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==2) {normal = normalForSphere(hit, sfere[2]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==6) {normal = normalForSphere(hit, sfere[3]); mcol = vec4(0.9,0.9,0.9,10.0); mref = vec2(0.0,0.0);}
    }  
}

// Function 485
void write_color(vec4 rgba, float w) {
    float src_a = w * rgba.a;
    float dst_a = _stack.premultiply?w:src_a;
    _color = _color * (1.0 - src_a) + rgba.rgb * dst_a;
}

// Function 486
vec4 getColor(vec2 uv, float spd){
    float tP = iTime/CYCLE*spd+hash12(uv*10.);
    float tN = (iTime/CYCLE*spd+1.)+hash12(uv*10.);
    
    float prev = complicatedNoise(tP, uv);
    float next = complicatedNoise(tN, uv);
    float o = mix(prev, next, ease(fract(tP)));
    return vec4(o,o,o,1.0);
}

// Function 487
vec3 voxelColor(vec3 pos, vec3 norm) {
    vec3 low = vec3(1.0, 0.0, 0.5);
    vec3 mid = vec3(0.8, 0.5, 1.0);
    vec3 hi = vec3(0.0, 0.7, 1.0);
    
    float c = hash1(floor(pos.xy / 3.0) + vec2(0.1));
    c = 0.7 * c + 0.3 * hash1(floor(8.0 * pos.z) + c);
    float a = 0.5 + 0.5 * hash1(floor(8.0 * pos.z) + c);
    c = clamp(2.0 * c, 0.0, 2.0);  
    return a * mix(mix(low, mid, c), mix(mid, hi, c - 1.0), step(1.0, c));
}

// Function 488
vec4 getSkyColor(vec3 ro, vec3 rd)
{
    vec3 blue = smoothstep(.2, 1., rd.y) * vec3(0, 0, .5);
    float nDotL = clamp(dot(rd, normalize(vec3(-1, 1, 0))), 0., 1.);
    vec3 highlight = vec3(pow(nDotL, 100.) * 2.0);
    return vec4(blue + highlight, 1);
}

// Function 489
vec3 ReturnColor(float r1, float r2){
 return abs(_Color2 - _Color1) * r1 + abs(_Color4 - _Color3) *r2;
  
}

// Function 490
vec3 tweakcolor(vec3 col) {
    col *= rotaxis(vec3(1), time*0.2);
    //col = mix(vec3(dot(col, vec3(0.7))), col, 1.0);
    col *= 0.4;
    col *= pow(col, vec3(1.3));

    return col;
}

// Function 491
float isRedColorHSV( vec3 color )
{
    vec3 wantedColor=  vec3( 1.0, .0, .0 );
    vec3 HSVColor = rgb2hsv( color );
    float WantedHue = .0;
    float dist = .3;
    float val =  smoothstep( .0, dist,mod( HSVColor.r - WantedHue,1. ));
    return val;
}

// Function 492
vec3 customColor() {
	return vec3(0.8,0.5,0.8);
}

// Function 493
float fixcolor(float x) {
    return 1.0-round(x*glow_num_steps)/glow_num_steps;
}

// Function 494
vec4 colorEdge(float stepx, float stepy, vec2 center, mat3 kernelX, mat3 kernelY) {
	//get samples around pixel
	vec4 colors[9];
	colors[0] = texture(iChannel0,center + vec2(-stepx,stepy));
	colors[1] = texture(iChannel0,center + vec2(0,stepy));
	colors[2] = texture(iChannel0,center + vec2(stepx,stepy));
	colors[3] = texture(iChannel0,center + vec2(-stepx,0));
	colors[4] = texture(iChannel0,center);
	colors[5] = texture(iChannel0,center + vec2(stepx,0));
	colors[6] = texture(iChannel0,center + vec2(-stepx,-stepy));
	colors[7] = texture(iChannel0,center + vec2(0,-stepy));
	colors[8] = texture(iChannel0,center + vec2(stepx,-stepy));
	
	mat3 imageR, imageG, imageB, imageA;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			imageR[i][j] = colors[i*3+j].r;
			imageG[i][j] = colors[i*3+j].g;
			imageB[i][j] = colors[i*3+j].b;
			imageA[i][j] = colors[i*3+j].a;
		}
	}
	
	vec4 color;
	color.r = convolveComponent(kernelX, kernelY, imageR);
	color.g = convolveComponent(kernelX, kernelY, imageG);
	color.b = convolveComponent(kernelX, kernelY, imageB);
	color.a = convolveComponent(kernelX, kernelY, imageA);
	
	return color;
}

// Function 495
vec4 doColor(in vec3 sp, in vec3 rd, in vec3 sn, in vec3 lp, float t){
    
    vec3 ld = lp-sp; // Light direction vector.
    float lDist = max(length(ld), .001); // Light to surface distance.
    ld /= lDist; // Normalizing the light vector.
    
    // Attenuating the light, based on distance.
    float atten = 1. / (1. + lDist*.2 + lDist*lDist*.1);
    
    // Standard diffuse term.
    float diff = max(dot(sn, ld), 0.);
    // Standard specualr term.
    float spec = pow(max( dot( reflect(-ld, sn), -rd ), 0.), 8.);
    
    // Coloring the object. You could set it to a single color, to
    // make things simpler, if you wanted.
    vec3 objCol = 0.5+0.5*cos(vec3(1.,2.,4.)+hash13(floor(sp.xzx+0.54))*10.);//getObjectColor(sp);
    if(mid == 1.)objCol = vec3(.3,0.,0.7);
    else if(mid == 2.){
        float checkX = floor(sp.x*2.+0.5);
        float checkY = floor(sp.z*2.+0.5);
    
        objCol = vec3(mod(checkX+checkY,2.))*0.5+0.25;
        }
    else if(mid == 3.)objCol = 0.5+0.5*cos(vec3(7.,4.,2.)+hash13(floor(sp.xzx+0.5))*10.);
    // Combining the above terms to produce the final scene color.
    vec3 sceneCol = (objCol*(diff*3. + .15) + vec3(1., 0.95,1.)*spec*3.) * atten;
    
    
    // Fog factor -- based on the distance from the camera.
    float fogF = smoothstep(0., .95, t/FAR);
    //
    // Applying the background fog. Just black, in this case, but you could
    // render sky, etc, as well.
    vec3 sky = mix(vec3(0.8,0.3,0.0),vec3(0.),clamp(sp.y,0.,3.));
    
    
    // Return the color. Performed once every pass... of which there are
    // only two, in this particular instance.
    return vec4(sceneCol,fogF);
    
}

// Function 496
vec4 calcMarkerColor(in vec3 pos)
{
	//Render code 
	vec2 codePos = pos.xz - vec2(-0.45,-0.5);
	
	vec2 segment = floor(codePos*5.555);
	vec2 inSquare = min(sign(abs(min(segment-5.0,0.0))),sign(max(segment+1.0,0.0)));
	float blackFactor = min(inSquare.x,inSquare.y);
	
	vec2 mirrorSeg = vec2(min(segment.x,segment.y),max(segment.x,segment.y));
		
	if (mirrorSeg == vec2(1,3) || mirrorSeg == vec2(2,3) || segment == vec2(1,2))
	{
		blackFactor = 0.0;
	}
	
	//Render Numbers
	
	//ZERO
	vec2 zeroPos = vec2(pos.x + 0.08,max(abs(pos.z - 0.7) - 0.05,0.0));
	float zeroBlack = 1.0 - clamp(sign(abs(0.05 - length(zeroPos))-0.02),0.0,1.0);
	
	//ONE
	vec2 onePos = pos.xz - vec2(0.08,0.7);
	vec2 oneBlack = clamp(sign(abs(onePos)-vec2(0.02,0.12)),0.0,1.0); 
	float numberBlack = max(zeroBlack,1.0 - max(oneBlack.x,oneBlack.y));
	
	return mix(vec4(0.2),vec4(0.0),max(numberBlack,blackFactor));
}

// Function 497
vec3 fps_color(float fps)
{
    return
        fps >= 250. ? vec3(.75, .75,  1.) :
        fps >= 144. ? vec3( 1., .75,  1.) :
        fps >= 120. ? vec3( 1.,  1.,  1.) :
    	fps >= 60.  ? vec3( .5,  1.,  .5) :
    	fps >= 30.  ? vec3( 1.,  1.,  0.) :
    	              vec3( 1.,  0.,  0.);
}

// Function 498
PathColor ColorScale( PathColor a, float s )
{
#if SPECTRAL    
    return PathColor( a.fIntensity * s );
#endif    
#if RGB
    return PathColor( a.vRGB * s );
#endif    
}

// Function 499
vec3 getParticleColor_mp( float pint)
{
   float hue;
   float saturation;
   
   saturation = 0.75/pow(pint, 2.5) + mp_saturation;
   hue = hue_time_factor*time2 + mp_hue;

   return hsv2rgb(vec3(hue, saturation, pint));
}

// Function 500
vec3 getColor(float t)
{
    float r = (sin(t*2.0+3.0)+1.0)/2.0;
    float g = (cos(t*7.0)+1.0)/2.0;
    float b = 1.0-(r+g)/2.0;
    return 0.7*vec3(r,g,b);
}

// Function 501
vec3 getColor( in float t )
{
    vec3 col = vec3(0.4,0.4,0.4);
    col += 0.12*fcos(6.28318*t*  1.0+vec3(0.0,0.8,1.1));
    col += 0.11*fcos(6.28318*t*  3.1+vec3(0.3,0.4,0.1));
    col += 0.10*fcos(6.28318*t*  5.1+vec3(0.1,0.7,1.1));
    col += 0.09*fcos(6.28318*t*  9.1+vec3(0.2,0.8,1.4));
    col += 0.08*fcos(6.28318*t* 17.1+vec3(0.2,0.6,0.7));
    col += 0.07*fcos(6.28318*t* 31.1+vec3(0.1,0.6,0.7));
    col += 0.06*fcos(6.28318*t* 65.1+vec3(0.0,0.5,0.8));
    col += 0.06*fcos(6.28318*t*115.1+vec3(0.1,0.4,0.7));
    col += 0.09*fcos(6.28318*t*265.1+vec3(1.1,1.4,2.7));
    return col;
}

// Function 502
vec4 _assert_color_func(int ifail_count) {
  // Blink in different colors depending on how many times the assert broke
  vec4 blink_color = vec4(1.0, 1.0, 1.0, 1.0);
  if(ifail_count == 1)      blink_color = vec4(0.0, 0.0, 1.0, 1.0);
  else if(ifail_count == 2) blink_color = vec4(0.0, 1.0, 0.0, 1.0);
  else if(ifail_count == 3) blink_color = vec4(0.0, 1.0, 1.0, 1.0);
  else if(ifail_count == 4) blink_color = vec4(1.0, 0.0, 0.0, 1.0);
  else if(ifail_count == 5) blink_color = vec4(1.0, 0.0, 1.0, 1.0);
  else                      blink_color = vec4(1.0, 1.0, 0.0, 1.0);
    
  return (mod(iTime, 1.0) > 0.5) ? blink_color :  vec4(1.0, 1.0, 1.0, 1.0);
}

// Function 503
vec3 getThemeColor(vec2 uv, float hue) {
    int shadeIdx = int(uv.x * 13.0);
    int swatchIdx = int((1.0 - uv.y) * 5.0);
    float seedChroma = 1000000.0;

    if (shadeIdx == 0) {
        return vec3(1.0);
    } else if (shadeIdx == 12) {
        return vec3(0.0);
    }

    if (iMouse.z > 0.0) {
        return gamut_clip_preserve_lightness(generateShadeOklab(swatchIdx, shadeIdx, seedChroma, hue, 1.0));
    } else {
        return generateShadeZcam(swatchIdx, shadeIdx, seedChroma, hue, 1.0);
    }
}

// Function 504
vec3 GetObjectInternalRayColor (in vec3 rayPos, in vec3 rayDir)
{
    // bounce around inside the object as many times as needed (or until max bounces) due total internal reflection
    float multiplier = 1.0;
    vec3 ret = vec3(0.0);
    float absorbDistance = 0.0;
	for (int i = 0; i < MAX_RAY_BOUNCES; ++i)
    {
        // try and intersect the object
    	vec4 rayInfo = RayIntersectObject(rayPos, rayDir);
        
        // should "never" happen but handle it anyways
    	if (rayInfo.x < 0.0)  
            return ret;
        
        // move the ray position to the intersection point.
        rayPos = rayPos + rayDir * rayInfo.x;
        
        // calculate beer's law absorption.
        absorbDistance += rayInfo.x;    
        vec3 absorb = exp(-OBJECT_ABSORB * absorbDistance);
        
        // calculate how much to reflect or transmit (refract or diffuse)
        float reflectMultiplier = FresnelReflectAmount(REFRACTIVE_INDEX_INSIDE, REFRACTIVE_INDEX_OUTSIDE, rayDir, rayInfo.yzw);
        float refractMultiplier = 1.0 - reflectMultiplier;
        
        // add in refraction outside of the object
        vec3 refractDir = refract(rayDir, rayInfo.yzw, REFRACTIVE_INDEX_INSIDE / REFRACTIVE_INDEX_OUTSIDE);
        ret += GetSceneRayColor(rayPos + refractDir*0.001, refractDir) * refractMultiplier * multiplier * absorb;
        
        // add specular highlight based on refracted ray direction
        ret += LightPixel(rayPos, rayDir, OBJECT_DIFFUSE, refractDir, OBJECT_SPECPOWER, false) * refractMultiplier * multiplier * absorb; 
        
        // follow the ray down the internal reflection path.
        rayDir = reflect(rayDir, rayInfo.yzw);
        
        // move the ray slightly down the reflect path
        rayPos += rayDir * 0.001;
        
        // For reflection, we are only going to be reflecting what is refracted on further bounces.
        // So, we just need to make sure the next bounce is added in at the reflectMultiplier amount, recursively.
		multiplier *= reflectMultiplier;        
    }
    
    // return the color we calculated
    return ret;
}

// Function 505
vec3 getColor()
{
	vec3 BaseColor = vec3(0.2,0.2,0.2);
	vec3 OrbitStrength = vec3(0.8, 0.8, 0.8);
	vec4 X = vec4(0.5, 0.6, 0.6, 0.2);
	vec4 Y = vec4(1.0, 0.5, 0.1, 0.7);
	vec4 Z = vec4(0.8, 0.7, 1.0, 0.3);
	vec4 R = vec4(0.7, 0.7, 0.5, 0.1);
    orbitTrap.w = sqrt(orbitTrap.w);
	vec3 orbitColor = X.xyz*X.w*orbitTrap.x + Y.xyz*Y.w*orbitTrap.y + Z.xyz*Z.w*orbitTrap.z + R.xyz*R.w*orbitTrap.w;
	vec3 color = mix(BaseColor,3.0*orbitColor,OrbitStrength);
	return color;
}

// Function 506
vec4 colorRamp(float t)
{
    return mix(vec4(0.800,0.000,0.773,1.), vec4(1.000,0.969,0.000,1.),t)*t;
}

// Function 507
float getColor(
    float dist,
    const float angle,
    float size,
    float phase
) { 
    dist = dist
        + (sin(angle * 3. + iTime * 1. + phase) + 1.) * .02
        + (cos(angle * 5. - iTime * 1.1 + phase) + 1.) * .01;
	return 
        pow(dist / size, WALL_THINNESS)
        * smoothstep(size, size - PARTICLE_EDGE_SMOOTHING, dist)        
    ;
}

// Function 508
float colorSpike(float theta, float spike)
{
    return max(0.0, smoothstep(0.0, 1.0, 1.0 - abs(SPREADFACTOR*(theta - spike))));
}

// Function 509
void initShipColor() {			
	rho_d = vec3(0.0657916, 0.0595705, 0.0581288);
	rho_s = vec3(1.55275, 2.00145, 1.93045);
	alpha = vec3(0.0149977, 0.0201665, 0.0225062);
	ppp = vec3(0.382631, 0.35975, 0.361657);
	F_0 = vec3(4.93242e-13, 1.00098e-14, 0.0103259);
	F_1 = vec3(-0.0401315, -0.0395054, -0.0312454);
	K_ap = vec3(50.1263, 38.8508, 34.9978);
	sh_lambda = vec3(3.41873, 3.77545, 3.78138);
	sh_c = vec3(6.09709e-08, 1.02036e-07, 1.01016e-07);
	sh_k = vec3(46.6236, 40.8229, 39.1812);
	sh_theta0 = vec3(0.183797, 0.139103, 0.117092);
}

// Function 510
float isRedColorRGB( vec3 color )
{
    vec3 wantedColor=  vec3( 1.0, .0, .0 );
    float distToColor = distance( color.rgb, wantedColor ) ;
    return distToColor;
}

// Function 511
vec3 DoColor(vec3 surfacePosition,vec3 worldPosition ,vec3 rayDirection, 
    vec3 normal, vec3 lightPosition, vec4 traceData)
{
    float fogDrift = traceData.y;

    float randomId = GridRandomIdXZ(surfacePosition, vec2(5,5));
    
    // some randomness testing
    vec3 objCol = vec3(1.0, 1.0, 1.0);

    //objCol.rg *= GridRandomId2XZ(surfacePosition, vec2(5,5)); // WORKS WITH SURFACE POSITION
    //objCol.b *= randomId;
    // postprocess variables
    vec3 fogColor = _FarHorizontColor.rgb;
    float fogIntensity = 0.0375f;

    // horizon calculation
    float surfaceDistance = traceData.x;// distance(surfacePosition, worldPosition);
    float lightDistance = distance(lightPosition, worldPosition);

    // space color
    float distanceNormalized = surfaceDistance / MaxDistance;
    if ( distanceNormalized >= 0.9f )
    {
        // main space color
        vec3 spaceColor = GetSpaceColor(worldPosition, rayDirection, lightPosition);
        // adding clouds to spaceColor
        spaceColor = AddClouds(spaceColor, rayDirection);
        // apply fog to space color
        vec3 foggedSpaceColor = ApplyFog(spaceColor, fogColor, lightDistance, fogIntensity);
        // some blending between space color and fogged space color
        vec3 finalSpaceColor =  mix(spaceColor, foggedSpaceColor, 0.35f);
        
        // blowing dust 
        vec3 foggedDustSpaceColor = mix(finalSpaceColor, _DustColor.rgb, fogDrift);

        // rayDirection.y / 6.0f is to prevent weird dust above  columns
        return mix(finalSpaceColor, foggedDustSpaceColor , rayDirection.y / 6.0f);
    }
    else
    {
        vec3 baseNormal = normal;
        float planeColumnDistance = traceData.w;

        // calculating floor texturing					
        float pMax = traceData.z;
        float v = clamp(1.0 - pMax * 4.0,0.0,1.0);						
            
        vec3 c2 = TexTriPlanar(surfacePosition/2.0, normal, iChannel0, iChannel0, iChannel0);
        vec3 c1 = TexTriPlanar(surfacePosition/3.0, normal, iChannel1, iChannel1, iChannel1);

        vec3 normal1 = 0.3 * TexTriPlanar(surfacePosition/2.0, normal, iChannel2, iChannel2, iChannel2);
        normal = normal + mix(normal1, normal1*2.0, v);
        objCol = mix(c1,c2, v);
        
        // calculating column texturing
        if(planeColumnDistance <= 0.0f)
        {
            normal = baseNormal + normal1; 
            //objCol = mix(objCol,c1, clamp(surfacePosition.y,0,1));
        }
    }

    // ============================================================================================================
    // lighting

    // main light
    PhongLightParams lightParams = PhongLightParams(
        objCol.rgb, 1.0,
        _MainLightSourceColor.rgb, 3.5, 0.03
    );

    // applying light
    vec3 lightDirection = normalize(lightPosition);
    vec3 col = PhongDirectionalLight(normal, rayDirection, lightDirection, lightParams);

    // ============================================================================================================
    // postprocess
    // ============================================================================================================

    // ============================================================================================================
    // apply shadows
    float shadow = SoftShadowsSimplified(surfacePosition, lightDirection, 0.1, 25.0);
    vec3 shadowed = col * shadow;
    col = mix(col, shadowed, 0.5);

    // ============================================================================================================
    // blowing dust 
    col = mix(col, _DustColor.rgb, fogDrift);

    // ============================================================================================================
    // we apply fog to dust color too, to have better blending
    col.rgb = ApplyFog(
        col.rgb, fogColor.rgb, _MainLightSourceColor.rgb,
        surfaceDistance, rayDirection, normalize(lightPosition-surfacePosition),
        fogIntensity
    );

    return col;
}

// Function 512
vec3 color(float t) {
    vec3 col;
	vec3 a = vec3(0.15,0.26,0.91);
    vec3 b = vec3(0.34,0.78,0.94);
    vec3 c = vec3(0.89,0.25,0.91);
    
    col = step(t,0.333)*a;
    col +=step(0.333,t)*step(t,0.667)*b;
    col +=step(0.667,t)*c;
    
    return col;
}

// Function 513
vec3 color(vec2 z, float ds, bool fl) {
    vec3[3] colors;
    colors[0]=vec3(1.0,0.5,0.0);
    colors[1]=vec3(0.0,1.0,0.5);
    colors[2]=vec3(0.5,0.0,1.0);
    
    if (fl) {
        colors[0]=1.0-colors[0];
        colors[1]=1.0-colors[1];
        colors[2]=1.0-colors[2];
    }
    
    float r2;
    int n = 60;
    int i;
    for(i=0;i<n;i++) {
        octant1(z, ds, colors);
        z -= vec2(s,s);
        r2 = dot(z,z);
        if (r2 < s * s) {
            z *= s * s / r2; ds *= s * s / r2;
            fl = !fl;
            z += vec2(s,s);
        } else {
            z += vec2(s,s);
        	break;
        }
    }
    octant1(z, ds, colors);
    r2 = dot(z,z);
    float v = (r2 - 2.0 * (z.x + z.y) * s + s * s) / (2.0 * ds * s * s);
    v = min(v,1.0);
    v = 0.75 + 0.25 * float(n-i) / float(n) * v;
    if (fl) v = 1.5 - v;
    float zz = 0.5 * (1.0 - r2);
    if (zz > z.x && zz > z.y) {
        return colors[0] * v * min(1.0, min((zz - z.x) / ds, (zz - z.y) / ds));
    }
    if (z.y > z.x) {
        return colors[1] * v * min(1.0, min((z.y - zz) / ds, (z.y - z.x) / ds));
    }
    return colors[2] * v * min(1.0, min((z.x - zz) / ds, (z.x-z.y) / ds));
}

// Function 514
vec3 backgroundColor(vec3 dir){
    vec3 unit_dir = normalize(dir);
    float t = 0.5*(unit_dir.y+1.0);
    return ((1.0-t)*vec3(1.0)+t*vec3(0.5,0.7,1.0))*skyIntensity;
}

// Function 515
vec3 colorFloor(vec3 P, vec3 I, vec3 N, float K, float solarPhase)
{
    vec3 L = mix(_moon.d, _light.d, solarPhase);
    vec3 color = vec3(0.0);
    float ao = texture(iChannel1, P.xz*FLOOR_TEXTURE_FREQ).r;
    vec3 diff = texture(iChannel2, P.xz*FLOOR_TEXTURE_FREQ).rgb;

    float algae_mask = max(0.0, texture(iChannel2, P.xz).r-0.5);

    diff = mix(diff, COLORALGAE, algae_mask);
    float roughness = mix(0.2, 1.0, algae_mask);

    color = brdf(
        K, // Ks
        K, // Kd
        roughness, // roughness
        1.0, // opacity
        diff,//vec3(1.0), // specular color
        diff, // diffuse color
        I, // I
        N, // N
        -L // L
    )*ao;
    return color;
}

// Function 516
float get_default_draw_color(in sampler2D s)
{
    return texelFetch(s, CTRL_DEF_COLOR, 0).w;
}

// Function 517
vec3 stepColor(vec3 col){
	vec3 hsv = rgb2hsv(col);
    if(hsv.z <= 0.33){
    	hsv.z = lumvals.x;
    }
    else if(hsv.z <= 0.67){
    	hsv.z = lumvals.y;
    }
    else{
    	hsv.z = lumvals.z;
    }
    return hsv2rgb(hsv);
}

// Function 518
vec4 _sampleColor(vec2 pos)
{
    if(pos.x < 0.0) pos.x *= -1.0;
    if(pos.y < 0.0) pos.y *= -1.0;
    if(pos.x > iResolution.x) pos.x -= iResolution.x;
    if(pos.y > iResolution.y) pos.y -= iResolution.y;
    vec2 uv = pos/iResolution.xy;
    return texture(iChannel0, uv);
}

// Function 519
vec4 LUT_Color(vec4 color)
{
    // The palette is an RGB cube. It takes 2 samples for the blue channel.  
    float slice;
    float slice_weight = modf(color.b * (LUT_SIZE - 1.f), slice);
    vec4 slice_color1 = textureLod(iChannel1, LUT_UV(color.r, color.g, slice + 0.0), 0.0);
    vec4 slice_color2 = textureLod(iChannel1, LUT_UV(color.r, color.g, slice + 1.0), 0.0);
    
    return vec4(mix(slice_color1, slice_color2, slice_weight).rgb, color.a);
}

// Function 520
vec3 getDepthWaterColor(float D)
{
    float d = max(0.0, min(1.0, 2.0*log(1.0+D/0.9)));

    return mix(WATERCOLOR1,
        mix(WATERCOLOR2,
        mix(WATERCOLOR3,
        mix(WATERCOLOR4, WATERCOLOR5, d)
            , d)
        	, d)
            , d);
}

// Function 521
float colormap_green(float x) {
    if (x < 20049.0 / 82979.0) {
        return 0.0;
    } else if (x < 327013.0 / 810990.0) {
        return (8546482679670.0 / 10875673217.0 * x - 2064961390770.0 / 10875673217.0) / 255.0;
    } else if (x <= 1.0) {
        return (103806720.0 / 483977.0 * x + 19607415.0 / 483977.0) / 255.0;
    } else {
        return 1.0;
    }
}

// Function 522
void drawColorIcon(vec2 p, float sz, int i, bool enable, inout vec3 color) {
    
    const float k = 0.8660254037844387;
    
    mat2 R = mat2(-0.5, k, -k, -0.5);
    
    vec2 p1 = vec2(k*sz, 0);
    vec2 p2 = vec2(0, 0.5*sz);
    
    mat3 colors;
    
    if (i == 0) {
        colors = mat3(vec3(1, 0, 0),
                      vec3(1, 1, 0),
                      vec3(0, 0, 1));
    } else {
        colors = mat3(vec3(0.6, 0, 0.6),
                      vec3(0.7, 0.4, 0.7),
                      vec3(0.1, 0.5, 0.5));
    }
    
    float ue = enable ? 1. : 0.3;
    float ds = 1e5;
    
    for (int j=0; j<3; ++j) {
        
        vec2 ap = vec2(abs(p.x), abs(p.y-0.5*sz));
        
        vec2 dls = lineSegDist2D(p2, p1, ap);
        
        p = R*p;
        
        color = mix(color, colors[j], smoothstep(1.0, 0.0, -dls.x+0.5) * ue);
        ds = min(ds, dls.y);
    
    }

    color = mix(color, vec3(0), smoothstep(1.0, 0.0, ds-0.05*sz) * ue);
    
}

// Function 523
vec3 get_color(float color){
    return color == BLUE
    	? vec3(0.149,0.141,0.912)
    :color == GREEN
    	? vec3(0.000,0.833,0.224)
   	:color == RED
    	? vec3(1.0,0.0,0.0)
   	:color == WHITE
    	? vec3(1.0,1.0,1.0)
   	:color == GRAY
    	? vec3(192.0,192.0,192.0)/255.0
    :color == YELLOW
    	? vec3(1.0,1.0,0.0)
   	:vec3(0);
}

// Function 524
vec4 getScreenColor(vec3 pos)
{
   pos.xy = getScreenCords(pos);
   vec4 col = texture(iChannel1, pos.xy + vec2(0.5)/iResolution.xy);
   //col.a = 0.97 - min(pow((col.r + col.b + col.g)/3., 0.8), 0.6);
   col.a = 1. - min(pow((col.r + col.b + col.g)/3., 0.8), 0.55);
   if (col.rgb!=vec3(0.))
      col.rgb = normalize(col.rgb);
    
   // Pixel texture
   vec2 ppos = fract(pos.xy*iResolution.xy);
   col.a = mix(col.a, 1. - 1.5*pow(abs(1. - 2.*ppos.x)*abs(1. - 2.*ppos.y), 0.38)*(1. - col.a), smoothstep(4.3, 9.2, get_zoom()));
 
   return col;  
}

// Function 525
vec3 RGBColorWheel(vec2 uv, float s) {
    float a = acos(uv.y / length(uv)) / 6.283185;
    if(uv.x < 0.) a = 1.-a;
    // a = fract(a-iTime*.1); // Rotate the wheel
    a = ceil(a*s-.5) / s;
    return clamp(abs(abs(vec3(a,a-1./3.,a-2./3.))-.5)*6.-1., 0., 1.);
}

// Function 526
vec4 fs_color(vec4 color)
{
    return color;
}

// Function 527
vec3 a_to_color(float a)
{
    return vec3(
        tri_step(0.,0.75, 1.-a),
        tri_step(0.12,0.95, 1.-a),
        tri_step(0.4,1.0, 1.-a)
    );
}

// Function 528
vec3 getColor(float m, float o){
    vec3 h = gethue(o*.25);
    // use orbit number to band coloring
    if(o>7.15  	&& o<7.65) 	h=vec3(1.);
    if(o>8.  	&& o<8.1) 	h=vec3(1.);
    if(o>.0  	&& o<.5) 	h=vec3(1.);
    if(o>-.1  	&& o<-.05) 	h=vec3(1.);
    if(o>-2.4 	&& o<-1.75) h=vec3(1.);
    if(o>-4.8 	&& o<-2.75) h=vec3(1.);
    if(o>-6.  	&& o<-5.75) h=vec3(1.);
    if(o>-9.  	&& o<-8.75) h=vec3(1.);
    if(o>-8.5  	&& o<-6.75) h=vec3(1.);
 	return h;
}

// Function 529
vec3 skyColor(in vec3 rd) {
  vec3 sunDir = sunDirection();

  float sunDot = max(dot(rd, sunDir), 0.0);
  
  vec3 final = vec3(0.0);

  final += mix(skyCol1, skyCol2, rd.y);

  final += 0.5*sunCol1*pow(sunDot, 90.0);

  final += 4.0*sunCol2*pow(sunDot, 900.0);
    
  return final;
}

// Function 530
vec3 largeTrianglesColor(vec3 pos)
{
    float a = (radians(60.0));
    float zoom = 2.0;
	vec2 c = (pos.xy + vec2(0.0, pos.z)) * vec2(sin(a),1.0);//scaled coordinates
    c = ((c+vec2(c.y,0.0)*cos(a))/zoom) + vec2(floor((c.x-c.y*cos(a))/zoom*4.0)/4.0,0.0);//Add rotations
    
    float l = min(min((1.0 - (2.0 * abs(fract((c.x-c.y)*4.0) - 0.5))),
        	      (1.0 - (2.0 * abs(fract(c.y * 4.0) - 0.5)))),
                  (1.0 - (2.0 * abs(fract(c.x * 4.0) - 0.5))));
    l = smoothstep(0.03, 0.02, l);
	
	return mix(0.01, l, 0.5) * vec3(0.2,0.5,1);
}

// Function 531
vec3 findColor(float obj, vec2 uv, vec3 n) {
	if (obj == FLAG) {
// FLAG
		float c = textureInvader(uv);
		return vec3(1.,c, c);
	} else if (obj == PLANET) {
// PLANET
		return mix(vec3(.7,.3,0),vec3(1,0,0), clamp(1.1-5.*(uv.x-1.8),0.1,.9));
	} else if (obj == SHIP_SIDE) {
		float spi = textureSpiral(uv);
		return mix(COLOR_SIDE, .4*COLOR_SIDE, spi);
	} else {
		vec3 c, sp = space(n).xyz;
		if (obj == SHIP_GLOB || obj == SHIP_HUBLOT) {
			c = mix(COLOR_GLOBE1, COLOR_GLOBE2, .5+.5*C2);
			return mix(c, sp, .8);
		} else if (obj == SHIP_ARM) {
			return mix(vec3(1), sp, .2);
		} else {			
			float spi = textureSpiral(uv);
			const vec3 lightblue = .25*vec3(0.5, 0.7, 0.9);
			c = mix(lightblue,lightblue*.4, spi);
			return mix(c, sp, .4);
		}
	}
}

// Function 532
vec3 ColorGrade( vec3 vColor )
{
    vec3 vHue = vec3(1.0, .7, .2);
    
    vec3 vGamma = 1.0 + vHue * 0.6;
    vec3 vGain = vec3(.9) + vHue * vHue * 8.0;
    
    vColor *= 1.5;
    
    float fMaxLum = 100.0;
    vColor /= fMaxLum;
    vColor = pow( vColor, vGamma );
    vColor *= vGain;
    vColor *= fMaxLum;  
    return vColor;
}

// Function 533
vec3 get_color(float color){
    if(color == BLUE){
    	return vec3(0.149,0.141,0.912);
   	}
    else if(color == GREEN){
    	return vec3(0.000,0.833,0.224);
   	}
    else if(color == RED){
    	return vec3(1.0,0.0,0.0);
   	}
    else if(color == WHITE){
    	return vec3(1.0,1.0,1.0);
   	}
    else if(color == GRAY){
    	return vec3(192.0,192.0,192.0)/255.0;
    }
    else if(color == YELLOW){
    	return vec3(1.0,1.0,0.0);
   	}
}

// Function 534
vec3 colorAxisAlignedBrushStroke(vec2 uv, vec2 uvPaper, vec3 inpColor, vec4 brushColor, vec2 p1, vec2 p2)
{
    // how far along is this point in the line. will come in handy.
    vec2 posInLine = smoothstep(p1, p2, uv);//(uv-p1)/(p2-p1);

    // wobble it around, humanize
    float wobbleAmplitude = 0.13;
    uv.x += sin(posInLine.y * pi2 * 0.2) * wobbleAmplitude;

    // distance to geometry
    float d = sdAxisAlignedRect(uv, p1, vec2(p1.x, p2.y));
    d -= abs(p1.x - p2.x) * 0.5;// rounds out the end.
    
    // warp the position-in-line, to control the curve of the brush falloff.
    posInLine = pow(posInLine, vec2((nsin(iTime * 0.5) * 2.) + 0.3));

    // brush stroke fibers effect.
    float strokeStrength = dtoa(d, 100.);
    float strokeAlpha = 0.
        + noise01((p2-uv) * vec2(min(iResolution.y,iResolution.x)*0.25, 1.))// high freq fibers
        + noise01((p2-uv) * vec2(79., 1.))// smooth brush texture. lots of room for variation here, also layering.
        + noise01((p2-uv) * vec2(14., 1.))// low freq noise, gives more variation
        ;
    strokeAlpha *= 0.66;
    strokeAlpha = strokeAlpha * strokeStrength;
    strokeAlpha = strokeAlpha - (1.-posInLine.y);
    strokeAlpha = (1.-posInLine.y) - (strokeAlpha * (1.-posInLine.y));

    // fill texture. todo: better curve, more round?
    const float inkOpacity = 0.85;
    float fillAlpha = (dtoa(abs(d), 90.) * (1.-inkOpacity)) + inkOpacity;

    // todo: splotches ?
    
    // paper bleed effect.
    float amt = 140. + (rand(uvPaper.y) * 30.) + (rand(uvPaper.x) * 30.);
    

    float alpha = fillAlpha * strokeAlpha * brushColor.a * dtoa(d, amt);
    alpha = clamp(alpha, 0.,1.);
    return mix(inpColor, brushColor.rgb, alpha);
}

// Function 535
vec3 surface_color(vec3 p)
{
    //p = floor(mod(p,10.0));
    return sin((vec3(planet_surface(p,3.0),planet_surface(p,5.0),planet_surface(p,7.0))))/10.0+vec3(.6);
}

// Function 536
vec4 RGBColor(float r, float g, float b) {
	return vec4(vec3(r, g, b) / 255.0, 1.0); 
}

// Function 537
vec3 bound_color(int bound) {
    if (bound == bound_volume) {
        return vec3(1.0,0.0,0.5);
    } else if (bound == bound_plane) {
        return vec3(0.0,0.5,1.0);
    } else if (bound == bound_ray) {
        return vec3(1.0,0.5,0.0);
    } else {
        return vec3(0.0);
    }
}

// Function 538
vec3 randomColor (vec2 st) {
    float randNum1 = fract(sin(dot(st.xy, vec2(12.9898,78.233)))*43758.5453123);
    float randNum2 = fract(cos(dot(st.xy, vec2(12.9898,78.233)))*43758.5453123);
    return vec3(randNum1, randNum2, 1.-randNum1);
}

// Function 539
vec3 spaceship_color(vec3 p)
{
    return vec3(0.5,0.5,0.5);
}

// Function 540
vec3 shiftColor(vec3 color, float t)
{
    color = t > 0.0 ?
    vec3(
    mix(color.r, 0.0    , clamp(t, 0.0, 1.0)),
    mix(color.g, color.r, clamp(t, 0.0, 1.0)),
    mix(color.b, color.g, clamp(t, 0.0, 1.0))) :
    vec3(
    mix(color.r, color.g, clamp(-t, 0.0, 1.0)),
    mix(color.g, color.b, clamp(-t, 0.0, 1.0)),
    mix(color.b, 0.0    , clamp(-t, 0.0, 1.0)));

    return color;
}

// Function 541
vec4 getColor(float opCount, vec2 pos) {
    float chr = 0.0; 
    float which=mod(opCount,17.0);    
    if (which==0.0 || which==11.0) {
        chr = drawChar( CH_S, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_s; }
    } else if (which==1.0 || which==6.0 || which==15.0) {
        chr = drawChar( CH_I, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_i; }
    } else if (which==2.0) {
        chr = drawChar( CH_L, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_l; }
    } else if (which==3.0) {
        chr = drawChar( CH_O, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_o; }
    } else if (which==4.0 || which==10.0) {
        chr = drawChar( CH_P, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_p; }
    } else if (which==5.0 ) {
        chr = drawChar( CH_R, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_r; }
    } else if (which==7.0) {
        chr = drawChar( CH_N, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_n; }
    } else if (which==8.0 || which==14.0) {
        chr = drawChar( CH_C, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_c; } 
    } else if (which==9.0 || which==13.0) {
        chr = drawChar( CH_E, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_e; }
    } else if (which==12.0) {
        chr = drawChar( CH_F, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_f; }
    } else if (which==16.0) {
        chr = drawChar( CH_T, charPos, charSize, pos);
        if (chr >0.0) { return black; } else { return color_t; }
    }
}

// Function 542
float EncodeWaterColor()
{
    return float(
        int(WaterColor.r * 64.0) + 
        (int(WaterColor.g * 64.0) << 6) +
        (int(WaterColor.b * 64.0) << 12)); 
}

// Function 543
vec4 getColor(vec2 pos, float theta, float time){
    float nbCroissant = 140.0;
    //nbCroissant = mix(0.0, 200.0, currentTime);
    
    //float theta = atan(pos.y, pos.x);
    float current = (theta + PI); //[0 - 360]
    current = mod(current + PI / 2.0, 2.0 * PI);  //offset to put the beguining at the top
    current = current / (2.0 * PI); //ratio for animation
    
    vec4 firstRedColor = vec4(0.995,0.454,0.192,1.000);
    vec4 firstYellowColor = vec4(0.179,0.179,0.995,1.000);
    vec4 firstColor = mix(firstYellowColor, firstRedColor, easingIn(1.0 - abs((current - 0.5)) * 1.0));
    vec4 secondColor = vec4(0.995,0.924,0.362,1.000);
    vec4 backGroundColor = vec4(0.001,0.000,0.005,1.000);
    
    float circleRadius = mix(0.0, 0.065, time);
    float circleWidth = mix(0.01, 0.065, time);
    
    float dist = generateCroissant(pos, nbCroissant, theta, time);
    dist = max(dist, generateCroissant(pos, nbCroissant / 5.0, theta, time)); // play with thoose
    dist = max(dist, generateCroissant(pos, nbCroissant * 1.8, theta, time)); // play with thoose
    dist = max(dist, generateSphere(pos, theta, time));
    //dist = max(dist, easingIn(circle(pos, circleRadius, circleWidth)));
    vec4 color = dist != 0.0 ? mix(firstColor, secondColor, easingIn(dist)) : backGroundColor;
    
    float width = 0.428;
    float glowRadius = 0.240;
    float rectBorder = max(rect(pos, vec2(width)), 0.0);
    if(rectBorder >= 0.0 && rectBorder < glowRadius) {
        float range = 1.0 - glowRadius;
        float easing = 1.0 - rectBorder;
        float coef = 1.0 / glowRadius;
        rectBorder = (easing - range) * coef;
        color = mix(color, secondColor, easingIn(rectBorder));
    }
    return color;
}

// Function 544
vec4 colorat(vec2 uv) {
	return texture(iChannel0, uv);
}

// Function 545
vec3 getColor(vec3 norm, vec3 pos, int objnr)
{
   #ifdef jelly_not_transparent
   vec3 jellycol = jelly_color(pos);
   #else
   vec3 jellycol = vec3(0);
   #endif
   vec3 col = objnr==JELLY_OBJ?jellycol:(
              objnr==CONTAINER_OBJ?container_color(pos):(
              objnr==RODS_OBJ?rods_color(pos):sky_color(pos)));

   return mix(col, slime_color(pos), slp);
   //return vec3(slp);
   //smoothstep(0.0, 0.5, slp)
}

// Function 546
vec3 sky_color(vec3 ray)
{
	vec3 rc = 1.5*texture(iChannel0, ray).rgb;
    return rc;
}

// Function 547
vec3 getColor(vec3 norm, vec3 pos, int objnr)
{
   return objnr==METAL_OBJ?getMetalColor(pos):
         (objnr==STICK_OBJ?getStickColor(pos):getWaterColor(pos));
}

// Function 548
Material
change_color( Material m, vec3 color )
{
    Material ret = m;
    ret.color = color;
    return ret;
}

// Function 549
vec3 filmGrainColor(vec2 uv, float offset)
{ // by ma (lstGWn)
    vec4 uvs;
    uvs.xy = uv + vec2(offset, offset);
    uvs.zw = uvs.xy + 0.5*vec2(1.0 / iResolution.x, 1.0 / iResolution.y);

    uvs = fract(uvs * vec2(21.5932, 21.77156).xyxy);

    vec2 shift = vec2(21.5351, 14.3137);
    vec2 temp0 = uvs.xy + dot(uvs.yx, uvs.xy + shift);
    vec2 temp1 = uvs.xw + dot(uvs.wx, uvs.xw + shift);
    vec2 temp2 = uvs.zy + dot(uvs.yz, uvs.zy + shift);
    vec2 temp3 = uvs.zw + dot(uvs.wz, uvs.zw + shift);

    vec3 r = vec3(0.0, 0.0, 0.0);
    r += fract(temp0.x * temp0.y * vec3(95.4337, 96.4337, 97.4337));
    r += fract(temp1.x * temp1.y * vec3(95.4337, 96.4337, 97.4337));
    r += fract(temp2.x * temp2.y * vec3(95.4337, 96.4337, 97.4337));
    r += fract(temp3.x * temp3.y * vec3(95.4337, 96.4337, 97.4337));

    return r * 0.25;
}

// Function 550
vec3 skyColor( vec2 uv)
{
    return vec3(.0);
	vec3 colEdge 	= vec3(.1, .5, .3);
	vec3 colCenter  = vec3(.0);
	return mix (colEdge, colCenter, 1.-length(uv ) / .9);
}

// Function 551
vec3 ColorTemperatureToRGB(float temperatureInKelvins)
{
	vec3 retColor;
	
    temperatureInKelvins = clamp(temperatureInKelvins, 1000.0, 40000.0) / 100.0;
    
    if (temperatureInKelvins <= 66.0)
    {
        retColor.r = 1.0;
        retColor.g = saturate(0.39008157876901960784 * log(temperatureInKelvins) - 0.63184144378862745098);
    }
    else
    {
    	float t = temperatureInKelvins - 60.0;
        retColor.r = saturate(1.29293618606274509804 * pow(t, -0.1332047592));
        retColor.g = saturate(1.12989086089529411765 * pow(t, -0.0755148492));
    }
    
    if (temperatureInKelvins >= 66.0)
        retColor.b = 1.0;
    else if(temperatureInKelvins <= 19.0)
        retColor.b = 0.0;
    else
        retColor.b = saturate(0.54320678911019607843 * log(temperatureInKelvins - 10.0) - 1.19625408914);

    return retColor;
}

// Function 552
vec4 getSandColor(vec4 simState) {   
    if (simState.x == 1.0) {
    	return mix(SAND_1_COLOR_1, SAND_1_COLOR_2, simState.y);
    } else if (simState.x == 2.0) {
        return mix(SAND_2_COLOR_1, SAND_2_COLOR_2, simState.y);
    } else if (simState.x == 3.0) {
        return mix(SAND_3_COLOR_1, SAND_3_COLOR_2, simState.y);
    } else {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
}

// Function 553
vec3 color(vec2 z, float ds) {
    float pi = 3.14159265359;
    float theta = pi/8.0;
    // someday I'll explain the cross-ratio magic that got me these numbers
    float r = 2.0 / (1.0 - sqrt(1.0 - 4.0 * sin(theta) * sin(theta)));
    float p = - r * cos(theta);
    bool fl = false;
    vec3[3] colors;
    colors[0] = vec3(1.0,0.5,0.0);
    colors[1] = vec3(0.0,1.0,0.5);
    colors[2] = vec3(0.5,0.0,1.0);
    vec3 t; // for temp space
    for(int i=0;i<100;i++) {
        if (z.x < 0.0) {
            z.x = -z.x;
            colors[2] = 1.0 - colors[2];
            fl = !fl;
            continue;
        }
        if (dot(z,z) < 1.0) {
            z /= dot(z,z);
            ds *= dot(z,z);
            fl = !fl;
            swap(colors[0],colors[1]);
            continue;
        }
        z.x -= p;
        if (dot(z,z) > r*r) {
            ds *= r * r / dot(z,z);
            z *= r * r / dot(z,z);
            fl = !fl;
            z.x += p;
            swap(colors[1],colors[2]);
            continue;
        }
        z.x += p;
        
        break;
        

    }
    vec3 col = colors[0];
    float f = 1.0;
    f = min(f, z.x / ds);
    z.x -= p;
    f = min(f, (r * r - dot(z,z)) / (ds * 2.0 * r));
    z.x += p;
    f = 0.75 + 0.25 * f;
    if (fl) {
        f = 1.5 - f;
    }
    col *= f;
    if (dot(z,z) - 1.0 < ds * 2.0) {
        float t = (dot(z,z) - 1.0) / (ds * 2.0);
        vec3 col2 = colors[1] * (1.5 - f);
        col = (1.0 + t) * col + (1.0 - t) * col2;
        col *= 0.5;
    }
    return col * min(1.0,1.0 / ds);
}

// Function 554
vec3 wangEdgeColoredTriangle(in vec2 uv, vec4 edges)
{
    float x = uv.x;
    float y = uv.y;
    float halfx = x-0.5;
    float halfy = y-0.5;
    float invx = 1. - uv.x;
    float invy = 1. - uv.y;
    
 
    vec3 result = vec3(0.0);
    if (edges.r > 0.8) {
        result.r = max(result.r, float(x <= 0.45-abs(halfy)));
    }
    else if (edges.r > 0.6) {
        result.g = max(result.g, float(x <= 0.45-abs(halfy)));
    }
    else if (edges.r > 0.45) {
        result.b = max(result.b, float(x <= 0.45-abs(halfy)));
    }
    else if (edges.r > 0.2) {
        result.rg = max(result.rg, float(x <= 0.45-abs(halfy)));
    }
    
    if (edges.g > 0.8) {
        result.r = max(result.r, float(invy <= 0.45-abs(halfx)));
    }
    else if (edges.g > 0.6) {
        result.g = max(result.g, float(invy <= 0.45-abs(halfx)));
    }
    else if (edges.g > 0.45) {
        result.b = max(result.b, float(invy <= 0.45-abs(halfx)));
    }
    else if (edges.g > 0.2) {
        result.rg = max(result.rg, float(invy <= 0.45-abs(halfx)));
    }
    
    if (edges.b > 0.8) {
        result.r = max(result.r, float(invx < 0.45-abs(halfy)));
    }   
    else if (edges.b > 0.6) {
        result.g = max(result.g, float(invx < 0.45-abs(halfy)));
    }
    else if (edges.b > 0.45) {
        result.b = max(result.b, float(invx < 0.45-abs(halfy)));
    }
    else if (edges.b > 0.2) {
        result.rg = max(result.rg, float(invx < 0.45-abs(halfy)));
    }
    
    if (edges.a > 0.8) {
        result.r = max(result.r, float(y < 0.45-abs(halfx)));
    }
    else if (edges.a > 0.6) {
        result.g = max(result.g, float(y < 0.45-abs(halfx)));
    }
    else if (edges.a > 0.45) {
        result.b = max(result.b, float(y < 0.45-abs(halfx)));
    }
    else if (edges.a > 0.2) {
        result.rg = max(result.rg, float(y < 0.45-abs(halfx)));
    }
    
    if (x < 0.015 || y < 0.015 || invx < 0.015 || invy < 0.015) { return vec3(0.); }
    
    return result;
}

// Function 555
vec3 normal_color( vec3 x ) {
    return (x-min(x,1.))/(max(x,255.)-min(x,1.));
}

// Function 556
vec3 gridColor(vec3 pos)
{
    float plane5 = abs(sdPlane(pos, vec4(1.0, 0.0, 0.0, 0)));
    float plane6 = abs(sdPlane(pos, vec4(0.0, 1.0, 0.0, 0)));
    float plane7 = abs(sdPlane(pos, vec4(0.0, 0.0, 1.0, 0)));

    float   nearest = abs(mod(plane5, planeDistance) - 0.5 * planeDistance);
    nearest = min(nearest, abs(mod(plane6, planeDistance) - 0.5 * planeDistance));
    nearest = min(nearest, abs(mod(plane7, planeDistance) - 0.5 * planeDistance));

    return mix(vec3(0.3, 0.3, 0.5), vec3(0.2), smoothstep(0.0, lineThickness, nearest));
}

// Function 557
vec3 getWaterColor(vec3 pos)
{
    #ifdef water_transparent
    return vec3(0.);
    #else
    return waterColor2;
    #endif
}

// Function 558
vec3 getSkyColor(vec3 ray)
{ 
    return sky_color;
}

// Function 559
vec3 ClampColor(in vec3 c, in float maxVal) { return min(c, vec3(maxVal)); }

// Function 560
vec4 GetColorPalette(in float x)
{
    float r = smoothstep(0.33, 0., x) + smoothstep(0.66, 1., x);
    float g = smoothstep(0., 0.33, x)* smoothstep(0.66, 0.33, x);
    float b = smoothstep(0.33, 0.66, x)* smoothstep(1., 0.66, x);
    
    vec3 col = sqrt(vec3(r,g,b)); // not sure about this
    
	return vec4(col, 1);
}

// Function 561
vec3 colorFromCoord(vec2 p){
    float t=hash12(p);
    return pal(t, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.10,0.20) );
    //return pal(t, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.33,0.67) );
}

// Function 562
vec3 rainbowColor(in vec3 ray_dir) 
{ 
    RAINBOW_DIR = normalize(RAINBOW_DIR);   
		
    float theta = degrees(acos(dot(RAINBOW_DIR, ray_dir)));
    vec3 nd 	= saturate(1.0 - abs((RAINBOW_COLOR_RANGE - theta) * 0.2));
    vec3 color  = _smoothstep(nd) * RAINBOW_INTENSITY;
    
    return color * max((RAINBOW_BRIGHTNESS - 0.75) * 1.5, 0.0);
}

// Function 563
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

// Function 564
vec3 CalculateColor(float maxPower)
{
	vec3 val = vec3(0.0);
					
	for(int i = 0; i < numberOfMetaballs; i++)
	{
		val += ColorOfMetaball(i) * (PowerOfMetaball(i) / maxPower);
	}
	
	return val;
}

// Function 565
vec4 circle_mask_color(Circle circle, vec2 position)
{
	float d = distance(circle.center, position);
	if(d > circle.radius)
	{
        //Mask color (Black)
		return vec4(0.0, 0.0, 0.0, 1.0);
	}
	
	float distanceFromCircle = circle.radius - d;
	float intencity = smoothstep(
								    0.0, 1.0, 
								    clamp(
                                        //Size of Circles for Goggles (0.0, 1.0, 0.0, 1.0)
									    remap(distanceFromCircle, 0.1, 0.3, 0.0, 1.0),
									    0.0,
									    1.0
								    )
								);
	return vec4(intencity, intencity, intencity, 1.0);
}

// Function 566
vec3 getRoomColor(vec3 pos)
{
    vec3 col = pos.y<roomSize.y - 0.0106?walls_color:ceiling_color;
    
    #ifdef doors
    if (abs(pos.z)<0.5*doorSize.x + 2.*dfSize.x && pos.y<doorSize.y + 2.*dfSize.x && abs(pos.x)<roomSize.x*1.5 || pos.y<2.*dfSize.x)
       col = dframe_color;
    #endif
    return col;
}

// Function 567
vec3 getHexagonColor(vec4 hex, float nbcols)
{
    colors[0] = vec3(.95, .88, .75);
    colors[1] = vec3(.45, .65, .9);
    colors[2] = vec3(.6, .9, .3);
    
    int colnr = int(mod(hex.x, nbcols));
    /*vec3 color = vec3(mod(colnr, 3.)==0., mod(colnr, 3.)==1., mod(colnr, 3.)==2.);
    if (colnr>2.)
        color = vec3(1.) - color;
    
    vec3 color2 = vec3(0.5*color.r + 0.3*color.g + 0.7*color.b,
                       0.3*color.r + 0.7*color.g + 0.4*color.b,
                       0.8*color.r + 0.5*color.g + 0.3*color.b);*/
    
    vec3 color2 = colnr==0?colors[0]:(colnr==1?colors[1]:colors[2]);
    
    return color2;
}

// Function 568
vec3 groundColor(in vec3 pos){
    vec3 col= vec3(0.);
    
    	vec2 ipos = floor(vec2(pos.x,pos.z)*.1);  // integer
    	vec2 fpos = fract(vec2(pos.x,pos.z)*.1);  // fraction
		vec2 tile = truchetPattern(fpos, random( ipos ));		// generate Maze
        vec2 tileXL = truchetPattern(fract(vec2(pos.x,pos.z)*.1), random( floor(vec2(pos.x,pos.z)*.1) ));		// used for impact effect
        
        // Maze
    	col.b += .4*(smoothstep(tile.x-0.05,tile.x,tile.y)-smoothstep(tile.x,tile.x+0.05,tile.y));
        col.b += .5*(1.-smoothstep(.0,.1,length(tile-fract(iTime*.4))));	// Head on top of Truchet pattern
    	
        col.rb += .5*(1.-smoothstep(0.,5.*sphAR,length(pos.xz-sphAO.xz)))*(smoothstep(tile.x-0.05,tile.x,tile.y)-
              		   smoothstep(tile.x,tile.x+0.05,tile.y));		// grid lag below sphere A
        col.gb += .5*(1.-smoothstep(0.,5.*sphBR,length(pos.xz-sphBO.xz)))*(smoothstep(tile.x-0.05,tile.x,tile.y)-
              		   smoothstep(tile.x,tile.x+0.05,tile.y));		// grid lag below sphere B
     	
        
        col += (1.-smoothstep(0.,.02,abs(pos.x)));				// thin white line (main line)
        col.rgb += .3*max(0.,1.-atan(abs(pos.x))*2./PI-.1);		// White line glow
        col.r += (1.-smoothstep(0.,.02,abs(pos.z)));			    // thin red line (crossing signal)
        
        col.r += max(0.,(1.-smoothstep(0., .6, fract(iTime*.1+pos.x*0.00025)))*((1.-smoothstep(0.,.02,abs(pos.z))) + max(0.,1.-atan(abs(pos.z))*2./PI-.1)));	//crossing pulse
        col.b += max(0.,(1.-smoothstep(0., .4, fract(iTime*3.+pos.z*0.01)))*((1.-smoothstep(0.,.02,abs(pos.x))) + max(0.,1.-atan(abs(pos.x))*2./PI-.1)));	//rapid pulse
                
       col.r += 1.*min(.9, smoothstep(0.,1.,(1.-fract(iTime*.1))
                *( smoothstep(tile.x-0.05,tile.x,tile.y) - smoothstep(tile.x,tile.x+0.05,tile.y)+1.*(1.-smoothstep(.0,.1,length(tileXL-fract(iTime*2.)))) )
                *(1.-smoothstep(0.,300000.*fract(iTime*.1), pos.x*pos.x+ pos.z*pos.z))*smoothstep(0.,100000.*(fract(iTime*.1)), pos.x*pos.x+ pos.z*pos.z)  ));  //impact
                                                                      
       col *= min(.8,10./length(.01*pos))+.2; 	// distance fog

    return col;
}

// Function 569
float ColorIntensity( PathColor a )
{
#if SPECTRAL    
    return a.fIntensity;
#endif    
#if RGB
    return dot( a.vRGB, vec3( 1.0 / 3.0 ) );    
#endif        
}

// Function 570
vec3 zColor(vec4 Z,vec4 originalZ)
{
    if( showDerivative )
    {
        vec2 T = Z.zw;
        // mathematically more correct than doing nothing ?
        Z.zw = cdiv(Z.zw,originalZ.xy*.5);
        lineThickness *= 1.;
        //*/
        Z.xy = T;
    }
    
    vec2 d = mod(Z.xy,2.*lineGrid);
    d = min( d , 2.*lineGrid - d);
    
    
    
    float norm2 = log(dot(Z.xy,Z.xy));
    float dnorm2 = log(dot(Z.zw,Z.zw));

    vec3 col = hpluvToRgb_(vec4(
        hueCoeff*atan(Z.y,Z.x)/TWOPI,
        1.,//1.-1.*(.5 * + .5*cos(norm2*64.)),
        .8,
        1.
    ),1.).rgb;
    
    // Gradient isolines
    col = mix( col ,
              derivativeColor.rgb - col,
              //derivativeColor.rgb,
              // vec3(dot(col,vec3(.3,.6,.1))),// ,
       smoothstep( 1. - fwidth(log(length(Z.zw)))/lineThickness/256. , 1. , .5 + .5*cos(dnorm2*8.) )
       * derivativeColor.a
    );
    
    
    // Two grids
    col = mix( col , lineColor1.rgb ,
       smoothstep(lineThickness*length(Z.zw),.0,min( d.x , d.y ))*lineColor1.a*gridOpacity
    );
    
    
    d = mod(Z.xy + lineGrid,2.*lineGrid);
    d = min( d , 2.*lineGrid - d);
    
    col = mix( col , lineColor2.rgb ,
       smoothstep(lineThickness*length(Z.zw),.0,min( d.x , d.y ))*lineColor2.a*gridOpacity
    );
    
    // Pole and zero
    col = mix( col , zeroColor.rgb ,
       smoothstep( 4. , 0. , norm2 - zeroScale ) * zeroColor.a 
    );
    
    col = mix( col , poleColor.rgb ,
       smoothstep( 3. , 1.5 , poleScale - norm2 ) * poleColor.a 
    );
    
    /*col = mix( col , vec3(1.,1.,1.) ,
       smoothstep( 0. , 1. ,
                  //fwidth(1./(1. + dnorm2 - norm2) )
                  fwidth(1./Z.w)
                 ) * 1.
    );*/
    
    // bugs
    #define ISNANORINF(x) (isnan(x) || isinf(x))
    if( ISNANORINF(Z.x) || ISNANORINF(Z.y) || ISNANORINF(Z.z) || ISNANORINF(Z.w) )
    {
        col = vec3(1.);
    }
    
    return col;
}

// Function 571
vec4 getNyanCatColor( vec2 p )
{
	p = clamp(p,0.0,1.0);
	p.x = p.x*40.0/256.0;
	p.y = 0.5 + 1.2*(0.5-p.y);
	p = clamp(p,0.0,1.0);
	float fr = floor( mod( 20.0*iTime, 6.0 ) );
	p.x += fr*40.0/256.0;
	return texture( iChannel0, p );
}

// Function 572
vec3 getHueColor(vec2 pos)
{
#ifdef ANIMATE
	float theta = mod(3.0 + 3.0 * atan(pos.x, pos.y) / M_PI + iTime, 6.0);
#else
	float theta = 3.0 + 3.0 * atan(pos.x, pos.y) / M_PI;
#endif
		
	vec3 color = vec3(0.0);
	
	return clamp(abs(mod(theta + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
}

// Function 573
vec3 outputColor(float aFinalHue, float aFinalSaturation, float aFinalValue) { return hsv2rgb_smooth(vec3(aFinalHue, aFinalSaturation, aFinalValue)); }

// Function 574
vec4 decodeColor(float a){

    // bit shift 
    // (we 1.0 / color_values so that we can use fract() instead of mod()) for bit masking
    vec4 a4 = a * (color_rpositions / color_values);
    
    // bit masking for each component
    // use fract() instead of mod() for performance reason
    a4 = fract(a4);
    
    // since we use fract(), we need to multiply color_values back to a4
    // then convert from [0, color_maxValues] back to [0, 1]
    // multiply by 2.0 to cancel out the 0.5 pre-multiplication in encoding
    return a4 * (2.0 * color_values / color_maxValues);
}

// Function 575
vec3 color(vec2 uv) {
    float a = atan(uv.y,uv.x);
    float r = length(uv);
    
    vec3 c = .5 * ( cos(a*vec3(2.,2.,1.) + vec3(.0,1.4,.4)) + 1. );

    return c * smoothstep(1.,0.,abs(fract(log(r)-iTime*.1)-.5)) // modulus lines
             * smoothstep(1.,0.,abs(fract((a*7.)/3.14+(iTime*.1))-.5)) // phase lines
             * smoothstep(11.,0.,log(r)) // infinity fades to black
             * smoothstep(.5,.4,abs(fract(speed)-.5)); // scene switch
}

// Function 576
vec3 getWallColor(vec3 pos)
{
    return vec3(0.3 + 0.7*texture(iChannel0, 0.9*pos.yz).r)*(0.5 + 0.5*smoothstep(0., 0.12, texture(iChannel3, 0.3*pos.yz).r));
}

// Function 577
vec4 do_color(in float time, in vec2 coords)
{
    float whereami = 
        50.0*distance(vec2(0.5),coords) - 10.0*time;
    //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^
    //          frequency terms           phase terms
    //
    //  ^^^^ how many rings (50/2pi)      ^^^^ how fast they move (2pi/peak)
    //
    //       ^^^^^^^^^^^^^^^^^^^^^^^^^^ radial pattern
    return vec4(0.0,0.0,
                0.5+0.5*sin(whereami),  // render in the blue channel
                1.0);
}

// Function 578
vec3 getRandomColor(float f, float t)
{
    return hsv2rgb(f+t, 0.2+cos(sin(f))*0.3, 0.9);
}

// Function 579
vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = clamp(colormap_blue(x), 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

// Function 580
vec3 getBuildingColor(vec3 p){
    vec3 block = floor(p.xyz + vec3(.25, 0., 0.));   
    float fill = smoothstep(.1, .9, fract(p.y*1.)) * step(.1, fract(p.z*1.)) * step(.1, fract(p.x*2. + 0.5));
    
    float lightForce = Hash3d(block);
    float isLightning = step(.5, Hash3d(block + vec3(52.12, 17.3, 7.5)));
    
    return vec3(fill)*(Hash3d(block) * isLightning + .01);        
}

// Function 581
bool IsColorSignificant(vec3 color)
{
    const float insignificantThreshold = 0.01;
    return color.r > insignificantThreshold || color.b > insignificantThreshold 
        || color.g > insignificantThreshold;
}

// Function 582
vec3 ditherColor(vec3 col, vec2 uv, float xres, float yres) {
    vec3 yuv = RGBtoYUV(col);

    vec3 col1 = floor(yuv * _Colors) / _Colors;
    vec3 col2 = ceil(yuv * _Colors) / _Colors;
    
    // Calculate dither texture UV based on the input texture
    vec2 ditherBlockUV = uv * vec2(xres / 8.0, yres / 8.0);
   
    yuv.x = mix(col1.x, col2.x, ditheredChannel(channelError(yuv.x, col1.x, col2.x), ditherBlockUV));
    yuv.y = mix(col1.y, col2.y, ditheredChannel(channelError(yuv.y, col1.y, col2.y), ditherBlockUV));
    yuv.z = mix(col1.z, col2.z, ditheredChannel(channelError(yuv.z, col1.z, col2.z), ditherBlockUV));
    
    return(YUVtoRGB(yuv));
}

// Function 583
vec3 getColor(vec3 ro, vec3 rd)
{
    vec3 color = vec3(0.0);
    vec3 col = vec3(1.0);
    int id=-1;
    int tm = -1;
    
    for(int i=0; i<6; i++)
    {
    	float t = 10000.0; //seed++;
		
   		vec2 tRoom = intersectCube(ro, rd, box0);          
   		if(tRoom.x < tRoom.y)   t = tRoom.y; 
        vec3 hit = ro + rd * t;  
        if(hit.y > 0.9999 && hit.x<1.3 && hit.x>-1.3 && hit.z<1.99 && hit.z>1.0) t=10000.0;
    
    	intersectscene(ro, rd, t, id, true);
    
    	hit = ro + rd * t;        
		vec4 mcol = vec4(vec3(0.99),0.0);
    	vec3 normal; 
    	vec2 mref = vec2(0);
      
    	ColorAndNormal(hit, mcol, normal, tRoom, mref, t, id);
    	hit = hit + normal * 0.00001;
         
        vec2 rnd = rand2();
        col *= mcol.xyz;
       /* if(mcol.w>0.0) 
        {
            if(i==0) {color = mcol.xyz; break;}
            float df=max(dot(rd,-normal),0.0)*2.0; //if(tm==1) df *= 19.0;
            color += col*mcol.xyz*mcol.w * df ;
            //if(tm==1) color += col * 1.5;
            break;
        }*/
		tm = -1;
        //if(rnd.x>abs(mref.x))//diffuse
        {
        	//rd = CosineWeightedSampleHemisphere ( normal, rnd); 
            rd = cosWeightedRandomHemisphereDirection(normal);
        	tm = 0;   
        
        	col *= clamp(dot(normal,rd),0.0,1.0);
            
            bool isLight = false;
            vec3 rnd = vec3(rand2(),rand2().x)*2.0-1.0;

         	vec3 lightf = light + rnd * sfere[3].center_radius.w;
         	vec3 dl = directLight(hit, normal, lightf, vec3(1.0,1.0,1.0), isLight);
         	color += col * dl*9.0;
         	//if(isLight) break;
        }       
        
        ro = hit + rd * 0.0001; 
        
        if(dot(col,col) < 0.1 && i>3) break;
    }
    
 	return color;   
}

// Function 584
vec4 Unpack_RAR_Color(ivec2 Date){
    return vec4(Date/256,mod(vec2(Date),256.))/256.;
}

// Function 585
float colormap_blue(float x) {
    if (x < 0.0) {
        return 54.0 / 255.0;
    } else if (x < 7249.0 / 82979.0) {
        return (829.79 * x + 54.51) / 255.0;
    } else if (x < 20049.0 / 82979.0) {
        return 127.0 / 255.0;
    } else if (x < 327013.0 / 810990.0) {
        return (792.02249341361393720147485376583 * x - 64.364790735602331034989206222672) / 255.0;
    } else {
        return 1.0;
    }
}

// Function 586
vec3 smallTrianglesColor(vec3 pos)
{
    float a = (radians(60.0));
    float zoom = 0.5;
	vec2 c = (pos.xy + vec2(0.0, pos.z)) * vec2(sin(a),1.0);//scaled coordinates
    c = ((c+vec2(c.y,0.0)*cos(a))/zoom) + vec2(floor((c.x-c.y*cos(a))/zoom*4.0)/4.0,0.0);//Add rotations
    float type = (r(floor(c*4.0))*0.2+r(floor(c*2.0))*0.3+r(floor(c))*0.5);//Randomize type
    type += 0.2 * sin(iTime*5.0*type);
    
    float l = min(min((1.0 - (2.0 * abs(fract((c.x-c.y)*4.0) - 0.5))),
        	      (1.0 - (2.0 * abs(fract(c.y * 4.0) - 0.5)))),
                  (1.0 - (2.0 * abs(fract(c.x * 4.0) - 0.5))));
    l = smoothstep(0.06, 0.04, l);
	
	return mix(type, l, 0.5) * vec3(0.2,0.5,1);
}

// Function 587
vec3 ColorGrade( vec3 vColor )
{
    vec3 vHue = vec3(1.0, .7, .2);
    
    vec3 vGamma = 1.0 + vHue * 0.6;
    vec3 vGain = vec3(.9) + vHue * vHue * 8.0;
    
    vColor *= 1.5;
    
    float fMaxLum = 100.0;
    vColor /= fMaxLum;
    vColor = pow( vColor, vGamma );
    vColor *= vGain;
    vColor *= fMaxLum;
    return vColor;
}

// Function 588
float diskColorr(in vec2 uv, vec2 offset)
{
    uv = uv - smoothstep(0.01,1.8,texture(iChannel0, (uv*1.0 - vec2((iTime+0.06) /3.6,(iTime+0.06) /9.2)) + offset).r) * 0.3;
    
    float d = length(uv)-RADIUS;
    return smoothstep(0.01,0.015,d);
}

// Function 589
vec3 colorFromTemperature( float t )
{
    // Convert a temperature in Kelvin to a color
    
    // Blackbody color data from Mitchell Charity's website
    // http://www.vendian.org/mncharity/dir3/blackbody/
    vec3 col = vec3(0);
    col = mix(col, rgb(0xff3800), clamp(t/1000.,0.,1.));
    col = mix(col, rgb(0xff8912), clamp((t-1000.)/1000.,0.,1.));
    // I'm unlikely to use higher temperatures for realistic flames,
    // but I included them anyway.
    col = mix(col, rgb(0xffb46b), clamp((t-2000.)/1000.,0.,1.));
    col = mix(col, rgb(0xffd1a3), clamp((t-3000.)/1000.,0.,1.));
    col = mix(col, rgb(0xffe4ce), clamp((t-4000.)/1000.,0.,1.));
    col = mix(col, rgb(0xfff3ef), clamp((t-5000.)/1000.,0.,1.));
    col = mix(col, rgb(0xf5f3ff), clamp((t-6000.)/1000.,0.,1.));
    return col*t/3000.;
}

// Function 590
vec3 color(vec2 p) {
	p *= 0.5;

	vec3 pos = vec3(p, 0.0);
	vec3 rd = normalize(vec3(p, 1.0));

	vec3 lig = vec3(cos(iTime) * 0.5, sin(iTime) * 0.2, -1.0) * LightSize;
	vec3 nor = vec3(0.0, 0.0, -1.0);

	vec2 eps = vec2(0.0002, 0.0);
	vec2 grad = vec2(
        	bump(pos.xy - eps.xy) - bump(pos.xy + eps.xy),
			bump(pos.xy - eps.yx) - bump(pos.xy + eps.yx)) / (2.0 * eps.xx);
    
    if (InvertNormal) {
        grad = -grad;
    }

	float r = pow(length(p), 0.1);

	nor = normalize(nor + vec3((grad), 0.0) * BumpFactor * r);
	vec3 ld = normalize(lig - pos);

	float dif = max(dot(nor, ld), 0.0);

	vec3 ref = reflect(-ld, nor);
	float spe = pow(max(dot(ref, -rd), 0.0), 32.0);

	vec3 texCol = tex(pos.xy);

	vec3 brdf = vec3(0.0);
	brdf += dif * vec3(1, 0.97, 0.92) * texCol * 0.7;
	brdf += spe * vec3(1.0, 0.6, 0.2) * 2.0;

	return clamp(brdf, 0.0, 1.0);
}

// Function 591
vec3 selfColor(vec3 pos) {
    vec3 pol = carToPol(pos-vec3(0,0,-0.8));
    return spectrum(0.45*pol.x);
}

// Function 592
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

// Function 593
vec3 lighterColorSmooth( vec3 source, vec3 destination )
{
  float sourceSum = source.r + source.g + source.b;
	float destinationSum = destination.r + destination.g + destination.b;
	float mixValue = sourceSum - destinationSum;
	return mix(destination,source,smoothstep(.0,1.,mixValue));
}

// Function 594
float color_to_val_1(in vec3 color) {
    vec3 radial = color - vec3(1.0) * dot(color, vec3(1.0))/3.0;
    return length(radial);
}

// Function 595
vec3 getcolor(vec3 pos){return de(pos).d>0.01?vec3(1):de(pos).col;}

// Function 596
vec3 adjustSV(vec3 rgb, float s, float v) {
    vec3 hsv = rgb2hsv(rgb);
    hsv.y = s;
    hsv.z = v;
    return hsv2rgb(hsv);
}

// Function 597
vec3 colorTransmission(vec3 ro, vec3 rd, float time, float nsteps)
{
    vec3 trans = vec3(1);
    float dd = length(rd) / nsteps;
    for (float s=.5 + FZERO; s < nsteps-.1; s += 1.) { // start at 0.5 to sample at center of integral part
        vec3 p = ro + s / nsteps * rd;
        float dist;
      #if 1
        vec3 mediaColor = vec3(1);
        float sigma; //vec4 sigmaS;    
        participatingMedia(sigma, mediaColor, dist, p, time); // quite heavy but we ignore most results
        vec3 shadowTint = vec3(1);
        // most of the rest of the colorization
        // happens down in lightFog
        shadowTint = (1.-mediaColor); // tinted shadows
        trans *= exp2(-sigma * dd * shadowTint);
      #else  // FIXME use IntegrateParticipatingMedia or...
        // anyway seems it should actually take inscattering into account
        // (it's light... *seems* to be coming from light source, so... use it!)
        // because it prevents "shadowing" just same as if lit directly by light source
        // anyway here we only take into account the extinction
        // because we don't know the actual light source or its brightness, here.
        // TODO I think there's a better way to compute the shadows more generally, probably.
        vec4 mo, mi;
        participatingMediaB(mo, mi, dist, p, time); // alt
     	trans *= mo.rgb;
      #endif
    }
    return trans;
}

// Function 598
vec3 GetSceneColor(vec3 p, float iTime)
{
    vec3 color;
    
    float voronoi_scale = cos(iTime / 6.) * 5. + 2.;
    VoronoiDist(p * voronoi_scale, 0., color);
    
    return color;
}

// Function 599
vec3 GetMaterialsColor(Ray r, int matID
){if(matID>7)return vec3(0)
 ;float fakeOA = pow((1.-float(r.iter)/float(maxStepRayMarching)),.7)
 ;return rainbow((sqrt(5.)*.5+.5)*float(matID*2))*fakeOA
 ;}

// Function 600
void ColorAndNormal(vec3 hit, inout vec4 mcol, inout vec3 normal, vec2 tRoom, inout vec2 mref, inout float t, const int id)
{
	if(t == tRoom.y)
	{            
		mref = vec2(0.0,0.0);
        normal =-normalForCube(hit, box0);   
        if(abs(normal.x)>0.0)
        { 
            mcol.xyz = vec3(0.95,0.95,0.95);
            mref = vec2(0.0,1.0);
        } 
         else if(normal.y>0.0)
        {
            vec3 tcol = texture(iChannel1,1.0-(hit.xz-vec2(1.5,1.5))/3.5).xyz;
            float s = tcol.y+0.1;//-d
            s = pow(s,3.0)*0.75+0.01;
            mref = vec2((s*0.5+0.1),pow(1.0-s,2.0));
            mcol.xyz = vec3(0.9);//tcol+0.4;
        } 
        else if(abs(normal.z)>0.0)
        {
            mcol.xyz = vec3(0.95,0.15,0.19);
            mref = vec2(0.0,1.0);
            
            if(normal.z<0.0)
			{
            	//cw = vec2(-0.4,0.1);
            	if(	all(lessThanEqual(hit.xy,vec2(-0.05,0.6)+cw)) &&
               		all(greaterThanEqual(hit.xy,vec2(-0.7,-0.6)+cw)) ||
               		all(lessThanEqual(hit.xy,vec2(0.7,0.6)+cw)) &&
               		all(greaterThanEqual(hit.xy,vec2(0.05,-0.6)+cw)))
               		mcol = vec4(vec3(1.1),2.0);
			}
        }
	}     
	else   
	{
        	 if(id==0) {normal = normalForSphere(hit, sfere[0]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==1) {normal = normalForSphere(hit, sfere[1]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==2) {normal = normalForSphere(hit, sfere[2]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==6) {normal = normalForSphere(hit, sfere[3]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
    	else if(id==10) {normal = normalforCylinder(hit, cylinder[0]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,1000.0);}
        else if(id==11) {normal = normalforCylinder(hit, cylinder[1]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,1000.0);}
        else if(id==12) {normal = normalforCylinder(hit, cylinder[2]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,1000.0);}
        else if(id==13) {normal = normalforCylinder(hit, cylinder[3]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,1000.0);}
        else if(id==20) {normal = normalForCube(hit, boxe[0]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==21) {normal = normalForCube(hit, boxe[1]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==22) {normal = normalForCube(hit, boxe[2]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.0,0.0);}
        else if(id==23) {normal = normalForCube(hit, boxe[3]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,9000.0);}
        else if(id==24) {normal = normalForCube(hit, boxe[4]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,9000.0);}
        else if(id==25) {normal = normalForCube(hit, boxe[5]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,10.0);}
        else if(id==26) {normal = normalForCube(hit, boxe[6]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(1.0,10.0);}
        else if(id==27) {normal = normalForCube(hit, boxe[7]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.8,0.8);}
        else if(id==28) {normal = normalForCube(hit, boxe[8]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==29) {normal = normalForCube(hit, boxe[9]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==30) {normal = normalForCube(hit, boxe[10]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==31) {normal = normalForCube(hit, boxe[11]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==32) {normal = normalForCube(hit, boxe[12]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==33) {normal = normalForCube(hit, boxe[13]); mcol = vec4(0.1,0.1,0.1,0.0); mref = vec2(0.6,0.8);}
        else if(id==34) {normal = normalForCube(hit, boxe[14]); mcol = vec4(0.9,0.9,0.9,0.0); mref = vec2(0.05,3.8);}
        
        if(id>19 && id<23)//material for dulap
        {
            vec2 uv = hit.yz;
            uv = abs(normal.y) > 0.0 ? hit.zx : uv;
            uv = abs(normal.z) > 0.0 ? hit.yx : uv; 
            mcol.xyz = texture(iChannel1,1.0-(uv - vec2(1.5,-1.0))/vec2(5.5,0.5)).xyz - vec3(0.35,0.2,0.2);
            mref = vec2(0.0,0.2);// transparent, glossines
            mcol.xyz = vec3(0.1,0.99,0.1);// color
            
            if(id==21)	normal = -normal;
        }
        
        if(id>26 && id<34)//masa scaun
        {
            mcol.xyz = vec3(0.9);
            mref = vec2(0.0,0.7);// transparent, glossines
            //if(id==27) mcol.xyz = vec3(0.9,0.9,0.9);// color
            
            if(id==21)	normal = -normal;
        }
        
        if(id==34)//calorifer
        {
            mcol.xyz = vec3(sin(hit.x*59.0)+2.0-0.2);
            mref = vec2(0.0,0.0);
        }
    }  
}

// Function 601
vec3 surface_color(vec3 p)
{
    return sin(vec3(vines(p,6.0),vines(p,4.0),vines(p,3.0)))/5.0+vec3(.5);
}

// Function 602
vec3 ColorGrade(vec3 col)
{
    col = ACEScct_from_Linear(col);
    {
        vec3 s = vec3(1.1, 1.2, 1.0);
        vec3 o = vec3(0.1, 0.0, 0.1);
        vec3 p = vec3(1.4, 1.3, 1.3);
        
        col = pow(col * s + o, p);
    }
    col = Linear_from_ACEScct(col);
    
    return col;
}

// Function 603
vec4 colorGamma (vec4 col)
{
#ifdef GAMMA
	col = vec4(pow(col.rgb, vec3(GAMMA)), col.a);
#endif
	return col;
}

// Function 604
vec4 color_temp2(float temp)
{
    return WHITE;
}

// Function 605
vec3 surface_color(vec3 p)
{
    p /= scale;
    p /= 200.0;
    return sin(vec3(sceneSDF(p/5.0,0.0),sceneSDF(p*3.0,0.0),sceneSDF(p*2.0,0.0)))/3.0+vec3(.3);
}


```