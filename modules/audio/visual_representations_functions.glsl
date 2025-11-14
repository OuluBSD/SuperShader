// Reusable Visual Representations Audio Functions
// Automatically extracted from audio visualization-related shaders

// Function 1
float 	UIStyle_TitleBarHeight() 		{ return 32.0; }

// Function 2
float SawtoothWave( float q, float x )
{
    float f = fract( x ) - q;
    f /= (f >= 0.0 ? 1.0 : 0.0) - q;
    return f * 2.0 - 1.0;
}

// Function 3
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

// Function 4
vec4 drawWaves(in float x, in float y)
{
    // Bounds checking.
    if(y < WAVES_Y) return TRANS;
    else if(y > WAVES_Y+7.) return L_BLUE;
    else
    {
        // Modulo the time and cast it to an int so the value returned
        // can be used as an index for which frame of animation to use.
        float t = mod(iTime*6.0,4.0);

        // We need to do the usual transform here as well.
        y -= WAVES_Y;
        
		// If we are under the shoreline, we need to use the palette
        // that reflects the shore.
        if(x > SHORE_END)
        {
            // The prior comparison required x to be pristine, so
            // we have to perform this modulo in here.
            x = mod(float(x),64.);
            return ARR4(t,
                        wavesSunnyPalette(wavesC(x,y)),
                        wavesSunnyPalette(wavesA(x,y)),
                        wavesSunnyPalette(wavesB(x,y)),
                        wavesSunnyPalette(wavesD(x,y)));
        }
        // otherwise we use the palette that reflects the clouds.
        else
        {
            x = mod(float(x),64.);
            return ARR4(t,
                        wavesShadowPalette(wavesC(x,y)),
                        wavesShadowPalette(wavesA(x,y)),
                        wavesShadowPalette(wavesB(x,y)),
                        wavesShadowPalette(wavesD(x,y)));
        }
    }
}

// Function 5
float dtrianglewave(float n, float x, float l, float phase){
    float k = n*2.0+1.0;
    return amp*8.0/pi/(k*l)*cos(pi*n)*cos(k*pi*x/l+phase);
}

// Function 6
float Bar(float pos,float bar){pos-=bar;return pos*pos<4.0?0.0:1.0;}

// Function 7
void CPU_FFT_Visual(vec2 u,inout vec4 c){
	//u.x -= 100.;
    u.x = floor(u.x/iResolution.y*360.)-60.;
    u.y = u.y/iResolution.y*350.-50.;
    if(u.x-0.5 < 256. && u.x>=0. && u.y>0.){
        float enegy = texture(iChannel0,vec2(u.x,10.5)/iChannelResolution[0].xy).y*50.;
        if(u.y < enegy){
            c.b = 1.;
        }
    }
}

// Function 8
float solveWave(vec2 dXPow, float uold, float u,vec2 up1, vec2 um1) 
{
	//CFL Stability condition
    float dt = min(DT, r*sqrt(min(dXPow.x,dXPow.y)/alpha));
	// Compute value of u next step (foward Euler, I know we could do better)
    float U = 0.0;
    
    U += alpha*lap(dXPow, u,up1, um1);
    U *= dt*dt;
    
    U += (2.0+dt*beta)*u-uold;
    U /= 1.0+dt*beta;
    return U;
}

// Function 9
float sawtoothwave(float n, float x, float l, float phase){
    n++;
    return amp*2.0/(pi*n)*sin(n*pi*x/l+phase);
}

// Function 10
vec2 dwave(in float t, in float a, in float w, in float p) {
  float dx = 1.0;
  float dy = a*w*cos(t*w + p);
  return vec2(dx, dy);
}

// Function 11
void diffraction_wavevis(inout vec3 color, vec2 uv, vec2 p, float gscale, int nslits, float spacing, float lambda) {
  vec4 rnd = paramdither(vec2(uvec2(gfc)/2u), 28841);
  float xprop = cvis*iTime;
  for (int i=-(nslits/2); i<((nslits+1)/2); i++) {
    float h = spacing*(float(i)+.5*float((nslits+1)%2));
    vec2 s = vec2(0.,h);
    vec2 r = p-s;
    vec2 rn = normalize(r);
    vec2 uvw = mat2(rn.x,-rn.y,rn.y,rn.x)*(uv-p);
    uvw.x+=length(r);
    uvw.x-=uvw.y*r.y/r.x;
    vec2 uvwh = vec2(uvw.x,abs(uvw.y));
    mat2 window = mat2(0., -gscale, length(r)+2.*gscale*10., 2.*gscale);
    mat2 domain = mat2(window[0][0], -2., window[0][0]+window[1][0], 2.);
    float vscale = 1./sqrt(float(nslits));
    float gsm = smoothstep(.05,.2,fwidth(uv.x)/lambda);
    float gin = 1.-gsm;
    vec3 col_single = vec3(0.,.5,1.)*4.;
    vec3 col_sum = vec3(1.,.5,0.)*4.;
    gsm*=.02;
    PLOT_CONTINUOUS(color, uvw, window, domain,  gin*col_single, 0, 1, .02*gscale, cos(2.*pi*fract((x-xprop)/lambda))*vscale );
    PLOT_CONTINUOUS(color, uvw, window, domain,  gsm*col_single, 1, 1, .02*gscale, 1.*vscale );
    PLOT_CONTINUOUS(color, uvw, window, domain, -gsm*col_single, 1, 1, .02*gscale, -1.*vscale );
    #define acompensate (4./x*(r.x*r.x)/dot(r,r))
    float falloff = 5.*exp(-square((uvw.x-length(r))/gscale)/mix(1.,10.,gin));
    window[0][1]*=4.; window[1][1]*=4.; domain = mat2(window[0][0], -2., window[0][0]+window[1][0], 2.);
    if (!KEYBOARD(Z,TOGGLE)) {
      PLOT_CONTINUOUS(color, uvw, window, domain,  gin*falloff*col_sum, 0, 1, .02*gscale, 
        diffraction_pattern( s + rn*x, nslits, spacing, lambda ).x/acompensate*vscale );
      PLOT_CONTINUOUS(color, uvw, window, domain,  gsm*falloff*col_sum, 1, 1, .02*gscale, 
        length(diffraction_pattern( s + rn*x, nslits, spacing, lambda ))/acompensate*vscale );
      PLOT_CONTINUOUS(color, uvw, window, domain, -gsm*falloff*col_sum, 1, 1, .02*gscale, 
        -length(diffraction_pattern( s + rn*x, nslits, spacing, lambda ))/acompensate*vscale );
    }
    #undef acompensate
  }
}

// Function 12
vec3 plotSinWave(vec2 currentUv, float freq, float amp, vec3 color, vec3 bgc)
{
    float dx = lineWidth / iResolution.x;
    //float dy = lineWidth / iResolution.y; //dont use this line, or you may want check what is gonna happen
    float dy = lineWidth / iResolution.y + sin(dx * freq) * amp;
    
	float sy = sin(currentUv.x * freq + iTime) * amp;
    
    float alpha = smoothstep(0.0, dy, abs(currentUv.y - sy));
    
    return mix(color, bgc, alpha);
}

// Function 13
void playBar1() {
    placeKicks(beat2, beatH4, 0.0, 1.0);
    placeKicks(beat4, beatH4, 2.5, 3.0);
    placeSnares(beatH2 - 1.0);
    placeSnares(beat4, beatH4, 1.75, 2.0);
    placeSnares(beat4, beatH4, 2.25, 2.5);
    placeSnares(beat4, beatH4, 3.75, 4.0);
    
    if (SYNTHS) {
        placeSynths(beat4, beatH4, 1.85, 2.25, 0.0);
    	placeSynths(beat4, beatH4, 2.25, 2.5, 1.0);
    	placeSynths(beatH2 - 0.6, 2.0);
    }

    stereo += vec2(hihatClosed(beat2)) * vec2(0.7, 1.0);
}

// Function 14
vec4 mapWave(vec3 p, out vec2 ph )
{
    float t = getWaveTime();
    
    waveFrontDistort(p, 1.0);
    
   	ph = getPhase(p,t) ;
    
   	float sea_h = sea_height ( p ) *  kSeaHeight;
    
    waveMoveForward( p, t, 1.0 );
  
    vec3 df = dfWaveAnim ( p,sea_h, ph);
    return vec4(df.x, df.y, df.z, ph); 
}

// Function 15
void UI_ProcessScrollbarPanelEnd( inout UIContext uiContext, inout UIPanelState scrollbarState )
{
    UI_PanelEnd( uiContext, scrollbarState );    
}

// Function 16
float waves(vec3 p, float y) {
    return planeDF(p, y) - waveDist(p);
}

// Function 17
int wavelength_to_idx(in float wavelength) {
    return int(wavelength - WL_START);
}

// Function 18
float wave() {
	return (1.0 + sin(iTime))*0.5;
}

// Function 19
void DrawFuelBar(vec2 fragCoord, float fuelPercent, inout vec3 pixelColor)
{
    fuelPercent = min(fuelPercent, 1.0);
    float aspectRatio = iResolution.x / iResolution.y;
    vec2 uv = (fragCoord / iResolution.xy) - vec2(0.5);
    uv.x *= aspectRatio;    
    
    const float c_width = 0.2;
    const float c_height = 0.05;
    
    vec2 boxPosLeft = vec2(-0.5 * aspectRatio + 0.01, 0.5 - (c_height + SCORE_SIZE / iResolution.y));
    vec2 boxPosRight = vec2(-0.5 * aspectRatio + 0.01 + c_width, 0.5 - (c_height + SCORE_SIZE / iResolution.y));
    
    // black outer box
    float boxDistance = UDFatLineSegment(uv, boxPosLeft, boxPosRight, c_height);
    boxDistance = 1.0 - smoothstep(0.0, AA_AMOUNT, boxDistance);
    pixelColor = mix(pixelColor, vec3(0.0,0.0,0.0), boxDistance);
    
    // red fuel amount
    if (fuelPercent > 0.0)
    {
        boxPosRight.x = boxPosLeft.x + (boxPosRight.x - boxPosLeft.x) * fuelPercent;
        boxDistance = UDFatLineSegment(uv, boxPosLeft, boxPosRight, c_height);
        boxDistance = 1.0 - smoothstep(0.0, AA_AMOUNT, boxDistance);
        pixelColor = mix(pixelColor, vec3(1.0,0.0,0.0), boxDistance);   
    }
}

// Function 20
vec4 mapWave(vec3 p )
{
    vec2 ph;
    return mapWave(p, ph);
}

// Function 21
void waveFrontDistort( inout vec3 p, float inSign )   
{
    // scale wave height
    p.y *= 0.65;
    
    // distort wave front line 
    p.x -= inSign * (sin(2.0*p.z)*0.2 + cos(10.0*p.z)* 0.05);
    
    // offset 
    p += inSign * WaveOffset;
    
}

// Function 22
float WavelengthToLuminosityLinear( float fWavelength )
{
    float fPos = ( fWavelength - standardObserver1931_w_min ) / (standardObserver1931_w_max - standardObserver1931_w_min);
    float fIndex = fPos * float(standardObserver1931_length);
    float fFloorIndex = floor(fIndex);
    float fBlend = clamp( fIndex - fFloorIndex, 0.0, 1.0 );
    int iIndex0 = int(fFloorIndex);
    int iIndex1 = iIndex0 + 1;
    iIndex0 = min( iIndex0, standardObserver1931_length - 1);
    iIndex1 = min( iIndex1, standardObserver1931_length - 1);    
    return mix( luminousEfficiency[iIndex0], luminousEfficiency[iIndex1], fBlend );
}

// Function 23
float beat_wave(float t, float t0, vec2 uv, vec2 wave_center, float dis){
    float wave_rad=wave_speed*(t-t0);
    
    float dis_abs=abs(dis);
    float dis_sign=sign(dis);
    
    float tmp_dis=1e38;
    
    if(t>t0){
        tmp_dis=distance(uv,wave_center)-wave_rad;
    }
    
    dis_abs=min(dis_abs,abs(tmp_dis));
    dis_sign*=sign(tmp_dis);
    
    return dis_abs*dis_sign;
}

// Function 24
void visualize(vec2 ro, vec2 rd, float t, float d, float i)
{
    vec2 n = getNormal(ro+rd*t);
    
    float x  = clamp(floor(T)-i,0.,1.);
    float f1 = x + (1.-x) * fract(T);
    float f2 = clamp((f1-.75)*16.,0.,1.);
    float f3 = floor(abs(cos(min(f1*8.,1.)*6.283))+.5);
    float a  = mix(atan(-n.y,-n.x),atan(rd.y,rd.x),f2);

    // ray line
    _d = min(_d,sdLine(_p,ro+rd*t,ro+rd*t+vec2(cos(a),sin(a))*d*floor(f3)));

    // step indicator
    _d = min(_d,length(_p-ro-rd*t)-.015);

    if (i == floor(T))
    {
        // circle
        _d = min(_d,abs(length(_p-ro-rd*t)-clamp(d*f3,0.,1e4)));
    }
}

// Function 25
float pausingWave(float x, float a, float b) { //    ___          ___          ___
    x = abs(fract(x) - .5) * 1. - .5 + a;      //   /   \        /   \        /   \ 
    return smoothstep(0., a - b, x);           // --     --------     --------     ------
}

// Function 26
vec4 debug_wave(vec4 wave)
{
    // green -> out of bounds
    vec4 colout = vec4(positivepulse, 0) * wave.x - vec4(negativepulse, 0) * wave.x;
    return clamp(2.*colout, 0., 1.) + vec4(.0, step(1., abs(wave.x)), 0., 0);
}

// Function 27
vec2 getWaveSound( in float time ) {
    // snap to the nearest 1/iSampleRate
    float period = 1.0 / iSampleRate;
    time = floor(time/period)*period;
    float totAmpl = 0.0;
    vec2 audio = vec2(0);
    for (int i = 0 ; i < SAMPLES ; i++) {
        float index = float(i - SAMPLES/2);
        float currStepF = period * index;
        vec2 curr = noise(time + currStepF);
        index /= 2.0; index *= index;
        float ampl = 1.0 - index;
        totAmpl += ampl;
        audio += curr*ampl;
    }
    return audio/totAmpl;
}

// Function 28
vec3 WaveNf (vec3 p, float d)
{
  vec3 vn;
  vec2 e;
  e = vec2 (max (0.01, 0.005 * d * d), 0.);
  p *= 0.5;
  vn.xz = 0.5 * (WaveHt (p.xz) - vec2 (WaveHt (p.xz + e.xy),  WaveHt (p.xz + e.yx)));
  vn.y = e.x;
  return normalize (vn);
}

// Function 29
vec3 wavelength_to_srgbl (float l_nm ) {
    if (l_nm<370.||l_nm>780.) return vec3(0.);
    vec4 l = vec4(1.065, 1.014, 1.839, 0.366);
    vec4 c = vec4(593.,556.3,449.8, 446.);
    vec4 s = vec4(.056,.075,.051, .043);
    if (l_nm<446.) s.a = 0.05; // fix creep from violet back to blue
    if (l_nm>593.) s.r = 0.062; // fix creep from red back to green
    vec4 v = (log(l_nm)-log(c))/s;
    vec4 xyzx = l*exp(-.5*v*v);
    vec3 xyz = xyzx.xyz+vec3(1,0,0)*xyzx.a;
    const mat3 xyz_to_rgb = 
      mat3(3.240,-.969,.056, -1.537,1.876,-.204, -0.499,0.042,1.057);
    vec3 rgb = xyz_to_rgb*xyz;
    return rgb;
}

// Function 30
vec3 barycentricCoordinate(vec2 P,Equerre T)
{
    vec2 PA = P - T.A;
    vec2 PB = P - T.B;
    vec2 PC = P - T.C;
    
    vec3 r = vec3(
        det22(PB,PC),
        det22(PC,PA),
        det22(PA,PB)
    );
    
    return r / (r.x + r.y + r.z);
}

// Function 31
vec2 barycentricToCartesian(vec3 barycentric, Triangle t){
    return vec2(barycentric.x*t.a.x + barycentric.y*t.b.x + barycentric.z*t.c.x,
                barycentric.x*t.a.y + barycentric.y*t.b.y + barycentric.z*t.c.y);
}

// Function 32
vec3  traceHexBaryInside(vec3 uu,vec3 oo,vec3 tt,vec3 b1,vec3 b2,vec3 b3,vec3 b4,vec3 b5,vec3 b6,vec3 X,vec3 Y
){vec2 o=b2c(oo,X,Y)
 ;vec2 t=b2c(tt,X,Y)
 ;vec2 p1=b2c(b1,X,Y)
 ;//vec2 p2=b2c(b2,X,Y)
 ;vec2 p3=b2c(b3,X,Y)
 ;vec2 p4=b2c(b4,X,Y)
 ;vec2 p5=b2c(b5,X,Y)
 ;vec2 p6=b2c(b6,X,Y)
 ;//return sat(floor(uu+1.)) //main triangle is white
 ;//r ia a vec4 with 1 of each color set to >0 foreach linesegment to test
 ;vec4 r=vec4(0)
 ;//uu=tt //for debug mode toggle
 //barycentric culling hell
 ;//if(uu.y<uu.z)uu.yz=uu.zy //mirror
     
 ;if(
     getPosSmall(floor(oo-tt))==2
     //sign(oo.x)==sign(tt.x)
    
    )uu.yz=uu.zy
 ;if(uu.x<0.)r.x=1.
 ;if(uu.z>0.)r.x=1.
 ;if(uu.z<=0.&&uu.x<=1.)r.y=1.//right shade
 ;if(uu.x>0.){ 
  ;if(uu.x<1.)r.y=1.//horizontal
  ;if(uu.y>1.)r.y=1.//right shade
  ;if(uu.y>0.&&uu.y<1.)r.z=1.
  ;if(uu.y<0.)r.w=1.
  ;if(uu.z>uu.y-1.&&uu.z<uu.y+1.)r.w=1.
  ;if(uu.x>1.&&uu.z<uu.y-1.)r.z=1.
 ;} 
 ;return 1.-r.xyz //debug output a
 ;//return r.yzw //debug output a
  ;//r=r.wxyz
  ;float s=suv(r)     

           /*
  ;if(s==4.//if in main triangle, we have this small exclusion method
 
  ){vec3 vs=getPos0V(floor(tt-oo))
   ;//return vs
       //.x points up   OR to the bottom corner
       //.y boints side
       //.z points down
    //;r=vec4(0,0,0,1)
    //;return hxL2(b3,b4,b5,o,t,X,Y).xyy
    /*
    ;if(vs.x==1.){
        ;//return hxL3(b3,b4,b5,b6,o,t,X,Y).xyy
        ;                    r=vec4(1,1,0,0)
        ;}
   else if(vs.y==1.)return hxL3(b2b3,b4,b5,o,t,X,Y).xyy 
   ;else return hxL3(b3,b4,b5,b6,o,t,X,Y).xyy
   ;s=suv(r) 
  ;}
/**/

 ;if(s==3.//test 3
 ){if(r.x==1.)return hxL3(b2,b1,b6,b5,o,t,X,Y).xyy//1110
  ;           return hxL3(b1,b6,b5,b4,o,t,X,Y).xyy//0111
 ;}else if(s==2.//test 2
 ){if(r.x==1.)return hxL2(b2,b1,b6,o,t,X,Y).xyy//1100 b2,b1,b6
  ;if(r.y==1.)return hxL2(b1,b6,b5,o,t,X,Y).xyy//0110 b1,b6,b5
  ;           return hxL2(b6,b5,b4,o,t,X,Y).xyy//0011 b6,b5,b4
 ;}////test 1 
 ;s=suv(r.xy)
 ;if(s==1.
 ){if(r.x==1.)return hxL1(b2,b1,o,t,X,Y).xyy  //1000==b2,b1
  ;           return hxL1(b1,b6,o,t,X,Y).xyy;}//0100==b1,b6
 ; if(r.z==1.)return hxL1(b6,b5,o,t,X,Y).xyy  //0010==b4,b5
 ;            return hxL1(b5,b4,o,t,X,Y).xyy;}

// Function 33
float wave(float T,vec2 pos,vec3 wave){
    
    float w = mod(T,wave.x)-wave.x/2.;
    vec2 xy=vec2(w*wave.z,w*w*wave.z+wave.y); 
    pos*=vec2(1.,1.12);
	return abs(length(xy-pos)-0.1)<0.02*(1.2-fract(T))?0.1:0.;
    
}

// Function 34
float wave(vec2 st, float offset)
{
    return 0.9 * sin(st.x * 30. + sin(st.y * 59.096) + sin(st.y * 18.272 + iTime + offset)) * 0.5 + 0.5;
}

// Function 35
float traceWaveBound ( in vec3 ro, in vec3 rd)
{
    float tmin = 1.0 ;
    float tmax = 50.0;
    
	float precis = 0.01;
    float t = tmin;
    float d = 0.0;

    for( int i=0; i<16; i++ )
    {
	    d = mapWaveCheap( ro+rd*t );
    
        if( d <precis || t>tmax ) break;   
       	t += d ;
    }
    if( t>tmax ) t=-1.0;
    return t;
}

// Function 36
void mainWaves(out vec4 fragColor, in vec2 fragCoord) {
  vec2 uv = fragCoord.xy / iResolution.xy;
 	
  float waterdepth = 2.1;
  vec3 wfloor = vec3(0.0, -waterdepth, 0.0);
  vec3 wceil = vec3(0.0, 0.0, 0.0);
  vec3 orig = vec3(0.0, 0.0, 0.0); // (<>, <viewer's height above water>, <>)
  vec3 ray = getRay(uv);
  float hihit = intersectPlane(orig, ray, wceil, vec3(0.0, 1.0, 0.0));
  if(ray.y >= -0.01){
    // above the horizon
    vec3 C = getatm(ray) * 1.5 + sun(ray); // <> * <horizon lightness> + <>
    C = aces_tonemap(C);
    fragColor = vec4(C,1.0);   
  } else {
    // below the horizon
    float lohit = intersectPlane(orig, ray, wfloor, vec3(0.0, 1.0, 0.0));
    vec3 hipos = orig + ray * hihit;
    vec3 lopos = orig + ray * lohit;
	float dist = raymarchwater(orig, hipos, lopos, waterdepth);
    vec3 pos = orig + ray * dist;

	vec3 N = normal(pos.xz, 0.001, waterdepth); // (<>, <water roughness>, <>)
    vec2 velocity = N.xz * (1.0 - N.y);
    N = mix(vec3(0.0, 1.0, 0.0), N, 1.0 / (dist * dist * 0.01 + 1.0));
    vec3 R = reflect(ray, N);
    float fresnel = (0.04 + (1.0-0.04)*(pow(1.0 - max(0.0, dot(-N, ray)), 5.0)));
	
    vec3 C = fresnel * getatm(R) * 2.0 + fresnel * sun(R);
    //tonemapping
    C = aces_tonemap(C);
    
	fragColor = vec4(C,1.0);
  }
}

// Function 37
void playBar4() {
    placeKicks(beat4, beatH4, 0.5, 1.0);
    placeKicks(beat2, beatH4, 2.5, 3.0);
    placeSnares(beat4, beatH4, 0.25, 0.5);
    placeSnares(beat4, beatH4, 1.75, 2.0);
    placeSnares(beat4, beatH4, 2.25, 2.5);
    placeSnares(beat2, beatH4, 1.0, 1.5);
    placeSnares(beat2, beatH4, 3.5, 4.0);
    
    if (SYNTHS) {
        placeSynths(beat4, beatH4, 1.0, 1.3, 1.0);
    	placeSynths(beat4, beatH4, 1.8, 2.25, 1.0);
    	placeSynths(beat4, beatH4, 2.25, 2.5, 0.0);
    }

    if (beatH4 > 2.5 && beatH4 < 3.0) {
        stereo += vec2(crash(beat2)) * vec2(0.8, 1.0);
    } else {
        stereo += vec2(hihatClosed(beat2)) * vec2(0.65, 1.0);
    }
   
}

// Function 38
void rasterbar(in vec2 xy, in int axis, in vec4 palette_0[4], in vec4 palette_1[4], in float bar_pos, in float size_0)
{
    // Lol, I'm new to GLSL so I just dealt with my lack of knowledge and hardcoded this...
    // This kinda loses it's point if I do it this way though. :( Maybe I'll figure this out later...
    final_col += rasterline(xy, axis, palette_0[0], 0.015 * size_0, bar_pos, 0.2300 * size_0);
    final_col += rasterline(xy, axis, palette_0[1], 0.005 * size_0, bar_pos, 0.2100 * size_0);
    final_col += rasterline(xy, axis, palette_0[0], 0.005 * size_0, bar_pos, 0.2000 * size_0);
    final_col += rasterline(xy, axis, palette_0[1], 0.015 * size_0, bar_pos, 0.1800 * size_0);
    final_col += rasterline(xy, axis, palette_0[0], 0.005 * size_0, bar_pos, 0.1600 * size_0);
    final_col += rasterline(xy, axis, palette_0[1], 0.015 * size_0, bar_pos, 0.1400 * size_0);
    final_col += rasterline(xy, axis, palette_0[2], 0.005 * size_0, bar_pos, 0.1200 * size_0);
    final_col += rasterline(xy, axis, palette_0[1], 0.005 * size_0, bar_pos, 0.1100 * size_0);
    final_col += rasterline(xy, axis, palette_0[2], 0.015 * size_0, bar_pos, 0.0900 * size_0);
    final_col += rasterline(xy, axis, palette_0[3], 0.005 * size_0, bar_pos, 0.0700 * size_0);
    final_col += rasterline(xy, axis, palette_0[2], 0.005 * size_0, bar_pos, 0.0600 * size_0);
    final_col += rasterline(xy, axis, palette_0[3], 0.005 * size_0, bar_pos, 0.0500 * size_0);
    final_col += rasterline(xy, axis, palette_0[2], 0.005 * size_0, bar_pos, 0.0400 * size_0);
    final_col += rasterline(xy, axis, palette_0[3], 0.020 * size_0, bar_pos, 0.0200 * size_0);
    
    final_col += rasterline(xy, axis, palette_1[0], 0.015 * size_0, bar_pos, -0.2300 * size_0);
    final_col += rasterline(xy, axis, palette_1[1], 0.005 * size_0, bar_pos, -0.2100 * size_0);
    final_col += rasterline(xy, axis, palette_1[0], 0.005 * size_0, bar_pos, -0.2000 * size_0);
    final_col += rasterline(xy, axis, palette_1[1], 0.015 * size_0, bar_pos, -0.1800 * size_0);
    final_col += rasterline(xy, axis, palette_1[0], 0.005 * size_0, bar_pos, -0.1600 * size_0);
    final_col += rasterline(xy, axis, palette_1[1], 0.015 * size_0, bar_pos, -0.1400 * size_0);
    final_col += rasterline(xy, axis, palette_1[2], 0.005 * size_0, bar_pos, -0.1200 * size_0);
    final_col += rasterline(xy, axis, palette_1[1], 0.005 * size_0, bar_pos, -0.1100 * size_0);
    final_col += rasterline(xy, axis, palette_1[2], 0.015 * size_0, bar_pos, -0.0900 * size_0);
    final_col += rasterline(xy, axis, palette_1[3], 0.005 * size_0, bar_pos, -0.0700 * size_0);
    final_col += rasterline(xy, axis, palette_1[2], 0.005 * size_0, bar_pos, -0.0600 * size_0);
    final_col += rasterline(xy, axis, palette_1[3], 0.005 * size_0, bar_pos, -0.0500 * size_0);
    final_col += rasterline(xy, axis, palette_1[2], 0.005 * size_0, bar_pos, -0.0400 * size_0);
    final_col += rasterline(xy, axis, palette_1[3], 0.020 * size_0, bar_pos, -0.0200 * size_0);
}

// Function 39
float simple_squarewave(float x, float freq){
	float sgn=-sign(mod(x,1./freq)-.5/freq);
	return sgn;
}

// Function 40
vec3 barycentric(vec2 a, vec2 b, vec2 c, vec2 p)
{
    float d = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    float alpha = ((b.y - c.y) * (p.x - c.x)+(c.x - b.x) * (p.y - c.y)) / d;
    float beta = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / d;
    return vec3(alpha, beta, 1.0 - alpha - beta);
}

// Function 41
float singlewave(float x, float t)
{
    float X = x - t * t;
    return -cos(X) * exp(-X * X);
}

// Function 42
float baryLerp(float a, float b, float c, vec3 x) {
    return a * x.x + b * x.y + c * x.z;
}

// Function 43
float getAirWave(vec2 uv, vec4 PerAmpThickDec, float rep, float time)
{
	float 
        val = abs(PerAmpThickDec.z / (uv.y + PerAmpThickDec.y * sin(uv.x / PerAmpThickDec.x + PerAmpThickDec.w))), 
        mask = 0.;
	uv.x += time;
	uv.x = mod(uv.x,rep) - rep * .5;
	val += -2./dot(uv,uv);
	uv.x += rep/2.;
	val += -2./dot(uv,uv);
	uv.x -= rep/4.;
	mask = rep/2./dot(uv,uv);
	val = step(val, .5) + step(mask, 1.);
	return step(val,.5);
}

// Function 44
float fragWave(vec2 uv, vec2 fragvUV, vec2 fragdUV, vec2 fragsUV, float sindf, float sinf, float fragvs) {
    // frag value value
    float fragv    = texture(iChannel0, fragvUV).x;
    // frag displacement value
	float fragd    = texture(iChannel0, fragdUV).x;
    // frag shift of sin argument, x + value)
	float frags    = texture(iChannel0, fragsUV).x;
    
    // Sine displacement factor value, depending on frag displacement value
    float sdf_ch = sindf * fragd;
    // Comparator sine
    float x = (uv.x * (1.0 - 2.0 * sdf_ch + sinf * frags) + sdf_ch) * PI;
    x = x > PI ? PI : x < 0.0 ? 0.0 : x;
    float cmp = sin(x) * fragv * fragvs;
    cmp *= cmp;
    // ??????? ???, ?? ?????.
    float ywk = uv.y - 0.5;
    ywk *= ywk;
    // (x - a) > -b && (x - a) < b
    // Is equal to:
    // (x - a)^2 < b^2
    if(ywk < cmp)
        return 1.0;
    return 0.0;
}

// Function 45
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

// Function 46
vec2 DemoWaves(float x, float y) {
    float theta = atan(x, y);
 	float d = length(vec2(x,y));
    float t = -iTime;
    
    //------------------
    // This is the place to play with the waves, try altering their direction and phaes etc...
    
    // Radial waves
    vec2 radialwave = 1. * CExp(d, 1., t);
    radialwave += CExp(d, 2., t * 1.01) * rands.x;
    radialwave += CExp(-d, 3., t * 1.005) * rands.y;
    
    // Angular waves
    vec2 angularwave = CExp(theta, 2., t * 1.07) * rands.z;
    angularwave += CExp(-theta, 5., -t) * rands.w;
    
    return 1. * radialwave + 1. * angularwave;
}

// Function 47
vec2 wave(float t) {
	vec2 w = vec2(0);
    for (int l = 0; l < 5; l++) {
		vec2 t2 = t*(1.0 + 0.01*sin(float(l)*vec2(37,42)));
		w += fract(t2)-0.5;
    }
	return w;
}

// Function 48
vec4 baryTriangleOldSemiObsolete(float t,vec2 u,vec3 r,vec4 m,vec2 p3,vec2 p4,vec2 p0,vec2 p1,vec2 p2,vec4 c){
 ;float h=line(u,p0,p1)-.01
 ;h=min(h,line(u,p1,p2)-.01)
 ;h=min(h,line(u,p2,p0)-.01)
 ;c.w=min(h,ray(u,p3,p4)-.01) //draw ray and triangle
 ;c.x=h
 ;c.z=sdTriangle(p0,p1,p2,u)-.04
 ;float carthesianDet=c2bdet(p0,p1,p2)//+1 div (preferred)
 ;vec3 uu=c2b(u,p0,p1,p2,carthesianDet)//+0 div foreach each c2b() with carthesianDet
 ;vec3 ssu=sat(sign(uu))
 ;float uus=suv(ssu)
 ;//return vec4(ssu,1.)
 ;mat3 n=mat3(p0,0,p1,0,p2,0)
 ;float ddd=0.
 ;float gpl=float(getPosLarge(ssu))

 ;vec3 oo=c2b(p3,p0,p1,p2,carthesianDet)
 ;vec3 tt=c2b(p4,p0,p1,p2,carthesianDet) //ray points transformed from carthesian to barycentric
 ;vec3 vs=oo-tt 
 ;vec3 l=vec3(dd(p1-p2),dd(p2-p0),dd(p0-p1))//squared side length
 ;float radius=dd(lengthBary(oo-tt,l))//vector (oo-tt) length*length from barycentric vector
 //;vec3 vv=c2b(p4-p3,p0,p1,p2)
 ;float center=sqrt(lengthBary(oo-uu,l))//vector (oo-uu) length from barycentric vector
 ;c.w=min(c.w,abs(center-radius))//;c.w=min(c.w,abs(length(u-p3)-radius))
 //;c.w=min(h,ray(u,p3,p4)-.01)
 ;vec3 os=sat(floor(oo+1.))
 ;vec3 ts=sat(floor(tt+1.))//sat is needed for scale (large distances to the triangle are cheap)
 ;float ooo=suv(os)
 ;float ttt=suv(ts)//sum of vector components now points at 7 different segments:
     //1* insideTriangle
     //3* largeBorderTile(adjacent) 
     //3* cornerBorder   (only touches triangle corners) 
 ;vec3 linesToCheck=vec3(0)
 ;if(ooo==2.//case 2: origin is in largeBorderTile
 ){if(ttt<2.&&os!=1.-ts)linesToCheck=vec3(0)
  ;else{linesToCheck=getPosSmallV(os)//this one is simple, either misses all, or hits only one.
      //nope nope nope,(vs) doesnt ALWAYS work HERE: 
      //but sure, there are other ways to do this, removed for now
 ;}}else if(ooo==1.){//case 1: origin is in cornerBorder
  ;if(ttt<2.&&os!=ts)linesToCheck=vec3(0)//only the other 2 outer corners miss the triangle
  ;else if(ttt==2.&&os!=1.-ts)linesToCheck=getPosSmallV(ts)//from cornerBorder to ADJACENT largeBorderTile, has 1 border
      ;else{linesToCheck=1.-getPosLargeV(os);
      ;if(1.-getPosLargeV(vs)!=linesToCheck)linesToCheck=vec3(0.)
 ;}}else //if(os==vec3(1) )//case 0: origin is in insideTriangle
  {if(ttt==2.)linesToCheck=getPosSmallV(ts) //target is in any largeBorderTile /single border)
  ;else linesToCheck=1.-getPosSmallV(vs)
 ;}  
 ;if(linesToCheck.x>0.)c.y=min(c.y,segment(u,p1,p2)-.04)//indicating segments that are hit.
 ;if(linesToCheck.y>0.)c.y=min(c.y,segment(u,p2,p0)-.04)//indicating segments that are hit.
 ;if(linesToCheck.z>0.)c.y=min(c.y,segment(u,p1,p0)-.04)//indicating segments that are hit.

 //the faces that are still in may only return the NEAR intersection.
 //there is no case where there is a far intersection, the firt positive intersection can be returned as nearest.
 //i have not implemented a function that takes (linesToCheck), to trace this triangle/prism.   
 //on tracing a triangle/prism efficiently.
 //triangles imply barycentric coordinates, converters exist, but they are not too fast, and should be avoided.
 //just start in barycentric coordinates
     //how about one moore domain, doing a 3d simplex (skewable triangle pyramid)

 //below code is from its canvas source function
 //;c.x=min(c.x,segment(u,p0,p2))                //red triangle shows 3 CVs 
 ;//m.xy=sort(m.xy)                               //m.x<m.y for BezierQuadGeneral()
 ;//c.z=BezierQuadGeneral (u,p0,p1,p2,2.*m.xy-1.)//blue shows bezier segment of parabola
 ;//c.z=BezierQuad        (u,p0,p1,p2)         //blue shows bezier segment of parabola
 ;//o.z+=.2*smoothstep(1.5,0.,(c.x-10./r.y)/fwidth(c.x))//this is just worse; https://www.shadertoy.com/view/XtdyDn
 ;c.z -=9./Aa(t,u,r,m) //line thickness
 ;c.xw-=2./Aa(t,u,r,m) 
 ;c.y -=4./Aa(t,u,r,m)
 ;return c;}

// Function 49
float fetch_wave_power(in vec3 v)
{
    //return singlewave(length(v.xy), mod(tT, REPEAT_MOD));
    float l = length(v.xy);
    return soundwave(l, REPEAT_MOD - mod(tT - WAVE_TIME_OFFSET, REPEAT_MOD), 4.0f, 0.0);
}

// Function 50
vec3 WavelengthToRGBLinear( float fWavelength )
{
     mat3 m = mat3( 2.3706743, -0.9000405, -0.4706338,
	-0.5138850,  1.4253036,  0.0885814,
 	0.0052982, -0.0146949,  1.0093968 );
    return WavelengthToXYZLinear( fWavelength ) * m;
}

// Function 51
float getwaves(vec2 position, int iterations){
  float iter = 10.0;
  float phase = 6.0;
  float speed = 0.5;
  float weight = 1.0;
  float w = 0.0;
  float ws = 0.0;
  for(int i=0;i<iterations;i++){
    vec2 p = vec2(sin(iter), cos(iter));
    vec2 res = wavedx(position, p, speed, phase, Time);
    position += normalize(p) * res.y * weight * DRAG_MULT;
    w += res.x * weight;
    iter += 12.0;
    ws += weight;
    weight = mix(weight, 0.0, 0.2);
    phase *= 1.18;
    speed *= 1.07;
  }
  return w / ws;
}

// Function 52
float getWaveGlow(vec2 pos, float radius, float intensity, float speed,
                  float amplitude, float frequency, float shift){
    
	float dist = abs(pos.y + amplitude * sin(shift + speed * time + pos.x * frequency));
    dist = 1.0/dist;
    dist *= radius;
    dist = pow(dist, intensity);
    return dist;
}

// Function 53
float getWaveformValue(float x, float mode,float e)
{
	return smootherSample(vec2(x,mode),e);
}

// Function 54
float wave(float f, float a, float t){
	return sin(t * f * 6.2831) * a;
}

// Function 55
float wavesA(in float x, in float y)
{
    if(x < 32.) // ARR64 would be a really long line.
    {
        return ARR8(y,
       	ARR32(x,3.,3.,3.,3.,3.,3.,3.,3.,3.,5.,5.,5.,4.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,3.,3.,3.),
		ARR32(x,3.,3.,3.,2.,2.,0.,0.,0.,0.,2.,5.,5.,5.,5.,5.,5.,3.,3.,3.,3.,3.,5.,5.,2.,2.,2.,0.,0.,0.,0.,0.,0.),
		ARR32(x,3.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,3.,3.,3.,3.,2.,0.,0.,0.,0.,5.,5.,5.,5.,5.,5.,2.,2.,2.,2.),
		ARR32(x,0.,0.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,5.,5.,5.,5.,5.,5.),
		ARR32(x,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,5.,5.),
		ARR32(x,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.),
		2.,
        2.);
    }
    else
    {
        x -= 32.;
        return ARR8(y,
		ARR32(x,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,3.,3.,3.,0.,0.,0.,0.,0.,0.,0.,0.,0.,3.,3.,3.),
        ARR32(x,0.,0.,0.,0.,2.,5.,5.,5.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,3.,3.,3.,3.,3.,3.,3.,3.),
        ARR32(x,2.,2.,1.,3.,3.,3.,2.,2.,2.,5.,0.,0.,0.,0.,0.,0.,0.,0.,5.,5.,3.,3.,3.,3.,3.,3.,3.,3.,2.,3.,3.,3.),
        ARR32(x,3.,3.,3.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,5.,5.,5.,3.,3.,3.,3.,3.,3.,2.,2.,2.,3.,3.,3.,3.,2.,0.),
        ARR32(x,5.,5.,5.,5.,5.,5.,2.,0.,0.,0.,0.,5.,5.,3.,3.,3.,3.,3.,3.,3.,2.,2.,2.,3.,3.,3.,3.,2.,0.,0.,0.,2.),
        ARR32(x,0.,0.,0.,0.,0.,2.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,2.,2.,2.,2.,2.,2.,2.,2.,2.),
        2.,
        2.);
    }
}

// Function 56
float 	UIStyle_ScrollBarSize() 		{ return 24.0; }

// Function 57
vec2 dfWave ( vec3 q, vec2 o, float kRot, vec3 kSpike)
{  
    vec2 p = q.xy;
    if ( dot(p,p) > 1.0) return vec2(p.y,1.0);
    float r = length(p); 
    p = rot ( p-o, kRot * 0.5 *PI *r  ) + o;
    
    // (height, distance to center)
    return  vec2(p.y - (hfWave(p.x, kSpike) ),p.x);    
}

// Function 58
vec4 visualizer(vec3 point) {
    vec2 hoz = tex( point );
    vec4 t = textureLod(iChannel0, hoz, 0.0);
    return t;
}

// Function 59
vec2 capillaryWaveD(in float t, in float a, in float k, in float h) {
  float w = sqrt((gravity*k + waterTension*k*k*k)*tanh(k*h));
  return dwave(t, a, k, w*iTime);
}

// Function 60
float wave3(float x) {return abs(sin(x*pi2))*step(fract(x*2.0),0.5);}

// Function 61
float WaveHt (vec2 p)
{
  mat2 qRot = mat2 (0.8, -0.6, 0.6, 0.8);
  vec4 t4, v4;
  vec2 q, t, tw;
  float wFreq, wAmp, h;
  q = 0.5 * p + vec2 (0., tCur);
  h = 0.6 * sin (dot (q, vec2 (-0.05, 1.))) + 0.45 * sin (dot (q, vec2 (0.1, 1.2))) +
     0.3 * sin (dot (q, vec2 (-0.2, 1.4)));
  q = p;
  wFreq = 1.;
  wAmp = 1.;
  tw = tWav * vec2 (1., -1.);
  for (int j = 0; j < 3; j ++) {
    q *= qRot;
    t4 = q.xyxy * wFreq + tw.xxyy;
    t = vec2 (Noisefv2 (t4.xy), Noisefv2 (t4.zw));
    t4 += 2. * t.xxyy - 1.;
    v4 = (1. - abs (sin (t4))) * (abs (sin (t4)) + abs (cos (t4)));
    t = 1. - sqrt (v4.xz * v4.yw);
    t *= t;
    t *= t;
    h += wAmp * dot (t, t);
    wFreq *= 2.;
    wAmp *= 0.5;
  }
  return h;
}

// Function 62
vec3 r56bary(vec3 o){return mirrorBaryY(o).yxz;}

// Function 63
float WaveletNoise(vec2 p, float phase, float scaleFactor) {
    float d = 0.;
    float scale = 1.;
    float mag=0.;
    for(float i=0.; i<4.; i++) {
        d += GetWavelet(p, phase, scale);
        p = p*mat2(.54,-.84, .84, .54)+i;
        mag += 1./scale;
        scale *= scaleFactor; 
    }
    d /= mag;
    return d;
}

// Function 64
vec2 barrelDistortion( vec2 p, vec2 amt )
{
    p = 2.0 * p - 1.0;

    /*
    const float maxBarrelPower = 5.0;
	//note: http://glsl.heroku.com/e#3290.7 , copied from Little Grasshopper
    float theta  = atan(p.y, p.x);
    vec2 radius = vec2( length(p) );
    radius = pow(radius, 1.0 + maxBarrelPower * amt);
    p.x = radius.x * cos(theta);
    p.y = radius.y * sin(theta);

	/*/
    // much faster version
    //const float maxBarrelPower = 5.0;
    //float radius = length(p);
    float maxBarrelPower = sqrt(5.0);
    float radius = dot(p,p); //faster but doesn't match above accurately
    p *= pow(vec2(radius), maxBarrelPower * amt);
	/* */

    return p * 0.5 + 0.5;
}

// Function 65
float distWaves(vec3 pos) {
     vec3 grid = gridFromCoords(pos);
    vec3 nGrid = vec3(grid.x, 0., grid.z);
    vec3 coords = coordsFromGrid(vec3(nGrid.x, 0., nGrid.z));            

    float seaHeight = waveDephthAt(vec3(nGrid.x, 0., nGrid.z));
    float dist = distSphere(pos, coords, seaHeight);

	return dist;
}

// Function 66
vec3 normalWave( in vec3 pos )
{
	vec3 eps = vec3( 0.01, 0.0, 0.0 );
	vec3 nor = vec3(
	    mapWave(pos+eps.xyy).x - mapWave(pos-eps.xyy).x,
	    mapWave(pos+eps.yxy).x - mapWave(pos-eps.yxy).x,
	    mapWave(pos+eps.yyx).x - mapWave(pos-eps.yyx).x );
	return normalize(nor);
}

// Function 67
vec3 waveToXyz(float wave)
{
	// XYZ directly
	/*float x1 = gauss(wave, 29.475, 444.358, 55.1489);
	float x2 = gauss(wave, 40.7142, 446.251, -41.6977);
	float x3 = gauss(wave, 23.5915, 606.057, 24.34);
	float x4 = gauss(wave, 37.6977, 590.361, 73.5741);

	float y1 = gauss(wave, 19.6797, 656.021, -0.778595);
	float y2 = gauss(wave, 42.4066, 565.962, 101.492);
	float y3 = gauss(wave, 27.5225, 462.807, 3.53373);
	float y4 = gauss(wave, 18.2148, 529.439, 8.83261);

	float z1 = gauss(wave, 10.2339, 422.327, 16.9254);
	float z2 = gauss(wave, 11.889, 443.142, 37.6146);
	float z3 = gauss(wave, 8.90569, 467.586, 10.5197);
	float z4 = gauss(wave, 25.8258, 460.901, 48.0458);*/

	// LMS (use with matrix operation to convert to XYZ)
	/*float x1 = gauss(wave, 23.5566, 446.096, 1.89299);
	float x2 = gauss(wave, 11.6691, 500.916, -2.28739);
	float x3 = gauss(wave, 32.3956, 536.797, 53.544);
	float x4 = gauss(wave, 33.11, 592.064, 62.7607);

	float y1 = gauss(wave, 13.3211, 479.559, 3.68812);
	float y2 = gauss(wave, 19.5222, 450.237, 3.33538);
	float y3 = gauss(wave, 17.7555, 521.0, 10.3423);
	float y4 = gauss(wave, 33.3204, 552.385, 77.3657);

	float z1 = gauss(wave, 10.2339, 422.327, 8.74764);
	float z2 = gauss(wave, 11.8916, 443.145, 19.3);
	float z3 = gauss(wave, 8.90568, 467.586, 5.43695);
	float z4 = gauss(wave, 25.8258, 460.901, 24.8318);*/

	// Increased accuracy
	float x1 = gauss(wave, 21.6622, 449.682, 2.36612);
	float x2 = gauss(wave, 11.0682, 477.589, 1.39883);
	float x3 = gauss(wave, 25.7494, 532.488, 34.0478);
	float x4 = gauss(wave, 5.91487, 570.2, 0.243387);
	float x5 = gauss(wave, 34.98, 585.858, 77.8669);

	float y1 = gauss(wave, 19.5222, 450.237, 3.33537);
	float y2 = gauss(wave, 13.3211, 479.559, 3.68813);
	float y3 = gauss(wave, 17.1502, 519.924, 9.68484);
	float y4 = gauss(wave, 3.27696, 542.8, 0.105766);
	float y5 = gauss(wave, 33.3895, 552.158, 77.9298);

	float z1 = gauss(wave, 8.84562, 467.661, 5.32073);
	float z2 = gauss(wave, 1.30608, 444.863, -0.0330768);
	float z3 = gauss(wave, 10.2028, 422.211, 8.58498);
	float z4 = gauss(wave, 11.9848, 443.084, 19.6347);
	float z5 = gauss(wave, 25.7907, 460.886, 24.9128);

	vec3 color = vec3(
		1.0/sqrt(2.0*PI)*(x1 + x2 + x3 + x4 + x5),
		1.0/sqrt(2.0*PI)*(y1 + y2 + y3 + y4 + y5),
		1.0/sqrt(2.0*PI)*(z1 + z2 + z3 + z4 + z5)
	);

	color = mat3(
		1.94735469, 0.68990272, 0,
		-1.41445123, 0.34832189, 0,
		0.36476327, 0, 1.93485343
	)*color;

	return color;
}

// Function 68
vec2 synthWave(float i, Mixer m)
{
    vec2 wave = vec2(0.0);
    SequencerState s = initSequencer(i);

    if (s.seqpos < 32)
    {
        float phi;
        int   note;

        note = lead_pat(lead_seq(imod(s.seqpos, 16)), s.p);
        float nfrq = note2Freq(note);

        // Lead
        if (nfrq > 0.0)
        {
            Resonator r = initResonator();

            for (int n = IIR_COUNT; n >= 0; n--)
            {
                float ii = i - float(n);
                SequencerState s = initSequencer(ii);

                float phi = s.env_mv * nfrq;
                float val = sqrt(s.env) * saw(phi);
                val = limit(val * 5.0);

                float rfrq = m.lead_fq * (32.0 * s.env * phi
                    + 2900.0 + 2400.0 * cos(0.25 * PI * ii / PATLEN));

                if (s.seqpos < 4 || s.seqpos >= 31) rfrq *= 0.25;

                r = updateResonator(r, rfrq, 8.0, val);
            }

            wave = m.lead * vec2(r.pos);
            wave = limit(wave * 3.0);
        }

        // Bass
        note = bass_pat(bass_seq(imod(s.seqpos, 16)), s.p);
        if (s.seqpos >= 4 && note > 0)
        {
            wave = compress(wave, 2.5 - s.env * 2.0);

            phi = s.env * note2Freq(note) * 0.0625;
            float bass_wave = s.env * (square(phi) + square(phi * 1.001));

            wave += m.bass * vec2(bass_wave);
        }

        // Hi-hat
        if (s.seqpos >= 8)
        {
            float env = imod(s.p, 4) == 2 ?
                2.0 * sqrt(s.env) // open
                : 1.3 * s.env2 ;  // closed

            wave += m.hihat * 0.7 * env * (noise(i + 0.1) + 0.3 * sin(80.0 * i));
        }

        // Snare drum
        if (s.seqpos >= 7)
        {
            float env = pow(s.env, 0.7);

            wave += m.snare * 0.9 * env * (1.5 * noise(i) + 0.3 * sin(100.0 * i))
                * (s.seqpos >= 31 ? 1.0 : float(snare_pat(0, s.p)));
        }

        // Bass drum
        if (s.seqpos >= 4)
        {
            if (drum_pat(0, s.p) > 0 || s.seqpos >= 31)
            {
                wave = compress(wave, 1.5 - s.env * 0.8);

                wave += m.drum * (
                            sin( 50.0 * s.env + 0.33)
                    + 2.0 * sin(100.0 * s.env2)
                    + 1.6 * sin(150.0 * pow(s.env2, 32.0)));
            }
        }
    }
    else
    {
        // Crash cymbal
        float env = pow( max(0.0, 1.0 - 0.25 * ((i / PATLEN) - 32.0)), 20.0);
        i = floor(i / 4.0) * 4.0;
        wave += m.crash * env * (noise(i) + 0.3 * sin(10.0 * i));
    }

    wave = limit(wave * 0.2);

    return wave;
}

// Function 69
vec3 bary_from_sphere(vec3 q) {    
    return bary_from_planar(planar_from_sphere(q));
}

// Function 70
float chopped_wave( vec2 p )
{
    p *= 8.0;
    
    vec2 pm = p;
    pm += vec2(-3.3,0.1);
    pm.x += 2.0;
    pm.x = mod(pm.x, 2.*PI);
    pm.x -= 2.0;
    return
        -min(
            -d_sinewave(p),
            d_sphere(pm*vec2(.5,1.), 0.95)
    );
}

// Function 71
float waveFunc(float x){
	return (noiseFunc(x) + noiseFunc(x*1.1))/2.0;
}

// Function 72
float fetch_wave_power(in vec3 v)
{
    return singlewave(length(v.xy), mod(tT, REPEAT_MOD));
}

// Function 73
float wave(float x, float k, float c, float t)
{
    float X = x - c * t;
    return -cos(k * X) * exp(-X * X);
}

// Function 74
void playBar4() {
    placeKicks(beat4, beatH4, 0.5, 1.0);
    placeKicks(beat2, beatH4, 2.5, 3.0);
    placeSnares(beat4, beatH4, 0.25, 0.5);
    placeSnares(beat4, beatH4, 1.75, 2.0);
    placeSnares(beat4, beatH4, 2.25, 2.5);
    placeSnares(beat2, beatH4, 1.0, 1.5);
    placeSnares(beat2, beatH4, 3.5, 4.0);

    if (beatH4 > 2.5 && beatH4 < 3.0) {
        crashWeight += crash(beat2);
    } else {
        hihatClosed(beat2);
    }
}

// Function 75
vec3 plot_testwave (vec2 uvs) {
    vec3 color = vec3(0.);
    vec2 uv = ( uvs - vec2(0.0,0.5) ) * vec2(40.,4.01);
    float offset = iMouse.x/iResolution.x*2.;
    const float testfreq = 0.117;
    float c = iMouse.y/iResolution.y;
    if ( abs( aatestsquare(testfreq, uv.x-offset, c) - uv.y )<0.01 ) { color.rgb += vec3(0.0,0.5,1.0); }
    if ( abs( stestsquare(testfreq, uv.x-offset, c) - uv.y )<0.005 ) { color.rgb += vec3(0.0,0.5,0.0); }
    if ( abs( -1.0 - uv.y )<0.01 ) { color.r = 1.; }
    if ( abs(  1.0 - uv.y )<0.01 ) { color.r = 1.; }
    float xsmp = floor(uv.x+0.5);
    if ( length(vec2((uv.x-xsmp)*0.2,uv.y - aatestsquare(testfreq, xsmp-offset, c) ))<0.03 ) { color.rgb+=vec3(0.0,0.8,1.0); }
    if ( length(vec2((uv.x-xsmp)*0.2,uv.y - stestsquare(testfreq, xsmp-offset, c) ))<0.02 ) { color.g+=0.5; }
    if ( abs(fract(uv.x-0.51)-0.5)<0.02 ) { color.r=0.5; }
    return color;
}

// Function 76
vec2 barrelDistortion( vec2 coord, float amt, float zoom )
{ // based on gtoledo3 (XslGz8)
  // added zoomimg
	vec2 cc = coord - 0.5;
    vec2 p = cc * zoom;
    coord = p + 0.5;
	float dist = dot( cc, cc );
	return coord + cc * dist * amt;
}

// Function 77
float waveform( float time ) {
    float i = isqrt(sin(time*pi*300.0));
    float j = atan(sin(time*pi*700.0));
    float m = iqnoise(vec2(time*100.0,0.0), 1.0, 0.0);
    float g = mix(i, j, m)*ipow(sin(time*2000.0),0.5);
    return g;
}

// Function 78
float3 ComputeWaveLambdaRayleigh(float3 lambda)
{
	const float n = 1.0003;
	const float N = 2.545E25;
	const float pn = 0.035;
	const float n2 = n * n;
	const float pi3 = PI * PI * PI;
	const float rayleighConst = (8.0 * pi3 * pow(n2 - 1.0,2.0)) / (3.0 * N) * ((6.0 + 3.0 * pn) / (6.0 - 7.0 * pn));
	return rayleighConst / (lambda * lambda * lambda * lambda);
}

// Function 79
vec3 wave_color( float d, float s, vec2 uv )
{
    float b = 0.001;
    // border
    float bw = 0.08;
    float innergray =
        mix(.5,0.,smoothstep(-0.01,0.01,d))
      + mix(.0,1.,smoothstep(bw,2.*bw,d));
    vec3 blue = vec3(
        vec2(s)
        + 0.2*length(texture(iChannel0, 0.2*uv))
        , 1.);
    vec3 white = vec3(1.);
    return mix(white, blue, 2.*innergray);
}

// Function 80
float hfWave ( float x, vec3 kSpike )
{
    float cos_h = kSpike.x * ( cos ( PI * x  ) *0.5 +0.5 ) ;
    float spike_h = kSpike.y *pow(abs(1.0-abs(x)), 5.5* kSpike.z ) ;
    return cos_h + spike_h;
}

// Function 81
float wave_model(float phi)
{
    phi = mod(phi, 4. * pi); // was 3.5
    return  min( min(phi,pi/2.) , max(5.*pi/2. - phi, 0.) );
}

// Function 82
float lowAverage()
{
    const int iters = 32;
    float product = 1.0;
    float sum = 0.0;
    
    float smallest = 0.0;
    
    for(int i = 0; i < iters; i++)
    {
        float sound = texture(iChannel1, vec2(float(i)/float(iters), 0.5)).r;
        smallest = 
        
        product *= sound;
        sum += sound;
    }
    return max(sum/float(iters), pow(product, 1.0/float(iters)));
}

// Function 83
void demo_waves(inout vec3 color, vec2 uv, int nslits, float spacing, vec2 mouse, float gscale) {
  float lambda = readctl(CTL_LAMBDA).x;
  diffraction_wavevis(color, uv, mouse, gscale, nslits, spacing, lambda);
}

// Function 84
float GetWaveHeight(vec2 p)
{
    p.x /= 2.;
    p.y /= 2.;
    float h1 = sin(p.y / 2.5 + iTime / 2.45 + cos(p.x / 7.7 + iTime / 5.35));
    h1 = mix(-1. * abs(h1), h1, 0.2);
    float h2 = sin(p.x / 2.7 + cos(iTime / 3.4) + iTime / 4.5) * (-1. * abs(cos(p.y / 4.2 + iTime / 3.46)));
    float h3 = (sin(p.x / 3. + iTime / 3. - sin(p.x / 1.2 + iTime / 3.) + p.y + cos(iTime / 5. + p.y / 2.)) / 2. + 0.5) / 2.5;
    float h4 = sin(p.x / 6. + p.y / 1.76 + cos(iTime/ 2.) - cos(p.x / 7. - p.y / 2. + sin(iTime / 5.))) + cos(p.y - p.x / 6.);
    return mix(SmoothMax(h1, h2, 2.) / 2. + 0.5, h3 + h4 / 4., 0.25);
}

// Function 85
float WaveHt (vec2 p)
{
  mat2 qRot = mat2 (0.8, -0.6, 0.6, 0.8);
  vec4 t4, v4;
  vec2 t;
  float wFreq, wAmp, ht;
  wFreq = 1.;
  wAmp = 1.;
  ht = 0.;
  for (int j = 0; j < 3; j ++) {
    p *= qRot;
    t = tWav * vec2 (1., -1.);
    t4 = (p.xyxy + t.xxyy) * wFreq;
    t = vec2 (Noisefv2 (t4.xy), Noisefv2 (t4.zw));
    t4 += 2. * t.xxyy - 1.;
    v4 = (1. - abs (sin (t4))) * (abs (sin (t4)) + abs (cos (t4)));
    ht += wAmp * dot (pow (1. - sqrt (v4.xz * v4.yw), vec2 (8.)), vec2 (1.));
    wFreq *= 2.;
    wAmp *= 0.5;
  }
  return ht;
}

// Function 86
float dsquarewave(float n, float x, float l, float phase){
    n = n*2.0+1.0;
    return amp*4.0/l*cos(n*pi*x/l+phase);
}

// Function 87
float getWaveHeight(float d) {
    if (d == 1. || d == -1.) 
        return 0.;
    //float tmp = (3.*d+1./SQRT2);
    float tmp = (2.*d+1./SQRT2);
    float zg = 0.5*tmp*exp(-1.*(tmp*tmp));
    float zc = 0.045*(exp(-2.*d)*cos(24.*3.14*d)-1.-d*(exp(-2.)*cos(24.*3.14)-1.));
    if (d < 0.4) {
        if (d < 0.)
    		zc = (-d*0.121);//0.0239; //yay, magical height correction :D
        else
            zc /= (1.-d)*100.;
    }
    #ifndef GRAVITY
    zg = 0.;
    #endif
    #ifndef CAPILLARY
    zc = 0.;
    #endif
    return zg + zc ;
}

// Function 88
vec4 wavesSunnyPalette(in float x)
{
    if(x<4.)
    {
        return ARR4(x, L_BLUE,
					   L_BLUE,
					   L_BLUE,
					   WHITE);
    }
    else return ARR2(x-4., WHITE, WHITE);
}

// Function 89
float singlewave(float x, float c, float t)
{
    float X = x - t * t;
    return -cos(c * X) * exp(-X * X);
}

// Function 90
void wave_3d(vec3 pos, float t, out vec2 im) {
  float amp = 1.0 / (sqrt(6.0) * 81.0 * A0 * A0);
  amp *= R_A_PI_3 * exp(-pos.x / (3.0 * A0));
  amp *= pos.x * pos.x;
  amp *= 3.0 * cos(pos.y) * cos(pos.y) - 1.0;

  float phase = E_3d * t;

  im_exp(amp, phase, im);
}

// Function 91
vec2 directionalWaveNormal(vec2 p, float amp, vec2 dir, float freq, float speed, float time, float k)
{	
	float a = dot(p, dir) * freq + time * speed;
	float b = 0.5 * k * freq * amp * pow((sin(a) + 1.0) * 0.5, k) * cos(a);
	return vec2(dir.x * b, dir.y * b);
}

// Function 92
float wave_pattern2(float t, vec2 uv){
    float dis=1e38;
    
    t=max(0.,t);
    
    float t0=t;
    
    t*=time_fac;
    
    float second=floor(t)-1.;
    
    float speed=min(second+1.,float(max_speed))*float(speed_fac);
    float frac=t-second;
    
    int ind0=int(3.*speed-1.);
    int ind1=0;
    
    vec2 wave_center=vec2(0);
    
    for(int ind=ind1;ind<=ind0;ind++){
        float beat=(second+float(ind)/(3.*speed))/time_fac;
        if(beat < begin+dur*(attack_time+decay_time+sustain_time+increase_time)){
            int type=int(mod(float(ind),3.));

            float t2=disk_animation(beat);

            if(type==0){
                wave_center=original_disk_center1*t2;
            }
            else if(type==1){
                wave_center=original_disk_center2*t2;
            }
            else if(type==2){
                wave_center=original_disk_center3*t2;
            }

            dis=beat_wave(t0,beat,uv,wave_center,dis);
        }
    }
    
    second+=1.;
    speed=min(second+1.,float(max_speed))*float(speed_fac);
    frac=t-second;
    ind0=int(frac*3.*speed);
    ind1=0;
    
    for(int ind=ind1;ind<=ind0;ind++){
        float beat=(second+float(ind)/(3.*speed))/time_fac;
        if(beat < begin+dur*(attack_time+decay_time+sustain_time+increase_time)){
            int type=int(mod(float(ind),3.));

            float t2=disk_animation(beat);

            if(type==0){
                wave_center=original_disk_center1*t2;
            }
            else if(type==1){
                wave_center=original_disk_center2*t2;
            }
            else if(type==2){
                wave_center=original_disk_center3*t2;
            }

            dis=beat_wave(t0,beat,uv,wave_center,dis);
        }
    }

    return dis;
}

// Function 93
void playBar3() {
    placeKicks(beat2, beatH4, 0.0, 1.0);
    placeKicks(beat2, beatH4, 2.5, 3.0);
    placeSnares(beatH2 - 1.0, beatH4, 0.0, 2.0);
    placeSnares(beat2, beatH4, 3.5, 4.0);
    placeSnares(beat4, beatH4, 1.75, 2.0);
    placeSnares(beat4, beatH4, 2.25, 2.5);

    hihatClosed(beat2);
}

// Function 94
float BlockWave(float x, float b, float c) {
	// expects 0<x<1
    // returns a block wave where b is the high part and c is the transition width
    
    return smoothstep(b-c, b, x)*smoothstep(1., 1.-c, x);
}

// Function 95
float wave_bell(float t, float f0)
{
    float op3 = sine(f0 * t * 6.0000             ) * exp(-t * 5.0);
    float op2 = sine(f0 * t * 7.2364 + op3 * 0.20);
    float op1 = sine(f0 * t * 2.0000 + op2 * 0.13) * exp(-t * 2.0);

    return op1;
}

// Function 96
float waves(vec2 coord, vec2 coordMul1, vec2 coordMul2, vec2 phases, vec2 timeMuls, float time) {
    return 0.5 * (sin(dot(coord, coordMul1) + timeMuls.x * time + phases.x) + cos(dot(coord, coordMul2) + timeMuls.y * time + phases.y));
}

// Function 97
v1 projectBary(v1 a,v1 b,v1 c){return b*dot(a,b)/dd(c);}

// Function 98
void waveMoveForward( inout vec3 p, in float t, float inSign)
{
    float shake = max(0.0,sin(t / WaveTimeMax * PI*0.5))*3.0;
    
    float xDisp = getPhaseDispInX(t).y;
    p.x -= inSign * xDisp ;  // moving for1ward
   
}

// Function 99
float waveT(float time){return u5(sin(time+sin(time*.8)+sin(time*.2)*sin(time*2.1)));}

// Function 100
vec2 UI_WindowGetTitleBarSize( UIContext uiContext, inout UIWindowState window )
{
    return vec2(window.drawRect.vSize.x - UIStyle_WindowBorderSize().x * 2.0, UIStyle_TitleBarHeight() );
}

// Function 101
float wave1(float x){return max(sin(x*pi2),0.0);}

// Function 102
vec2 getWaveContribution(in vec3 iResolution, in sampler2D iChannel0, in int frame, in int index, in vec2 pos) {
    vec4 pos22 = getParticle(iResolution, iChannel1, iFrame, index);
    vec2 pos2 = pos22.xy;
    vec2 oldPos2 = pos22.zw;
    vec2 v = pos2 - oldPos2;
    return v;
}

// Function 103
float solve_black_body_fraction_between_wavelengths(float lo, float hi, float temperature){
	return 	solve_black_body_fraction_below_wavelength(hi, temperature) - 
			solve_black_body_fraction_below_wavelength(lo, temperature);
}

// Function 104
void initWave(inout vec4 col, vec2 pos)
{
#if defined   DOUBLE_SLIT
	initWaveDoubleSlit(col,pos);
#elif defined ATOM2D
	initWaveAtom2D(col,pos);
#endif
}

// Function 105
vec3 bar(vec3 color, vec3 background, vec2 position, vec2 diemensions, vec2 uv)
{
    return rectangle(color, background, vec4(position.x, position.y+diemensions.y/2.0, diemensions.x/2.0, diemensions.y/2.0), uv); //Just transform rectangle a little
}

// Function 106
vec3 draw_wave(vec2 uv) {
    float gain = get_sample_gain(uv.x);
    uv = (uv - 0.5)*4.0;
    uv.y -= 1.0;
    uv.y += gain - 0.5;
	float line_width = 2.0/iResolution.y;
    float dist = length(uv.y) - line_width;
    float cut = smoothstep(0.01, 0.0, dist);
    vec3 col = mix3(COL_LOW, COL_MID, COL_HIGH, gain);
    return col * cut;
}

// Function 107
float lowAverage()
{
    const int iters = 32;
    float sum = 0.0;
    
    float last = length(texture(iChannel0, vec2(0.0)));
    float next;
    for(int i = 1; i < iters; i++)
    {
        next = length(texture(iChannel0, vec2(float(i)/float(iters), 0.0)));
        sum += last;//pow(abs(last-next), 1.0);
        last = next;
    }
    return sum/float(iters);
}

// Function 108
float wave_model(float phi)
{
    phi = mod(phi, 4.0 * pi);
    
    return   phi <= pi / 2. ? phi
           : phi <= 2. * pi ? pi / 2.
           : phi <= 5.0 * pi / 2. ? pi / 2. - (phi - 2. * pi)
           : 0.;
}

// Function 109
float squareWave(float time, float freq, float amp) {
    float sine = sin(TWO_PI*time*freq);
    if(sine > 0.) {
        return amp;
    } else {
        return -amp;
    }
}

// Function 110
void bars(float p, out vec3 diffuse, out vec3 surfaceLight) {
    #define BPM 120.2
    p += iTime*0.2;
    p *= 0.8;
    float inUV = fract(p)-0.5;
    float x = floor(p) - floor(iTime * (BPM / 60.0));
    float d = smoothstep( 0.45, 0.40, abs(inUV) );
    diffuse = mix(vec3(0.05), vec3(0.07), d);
    surfaceLight = vec3(50) * d * pow(abs(mod(x, 10.0) / 9.0), 10.0);
}

// Function 111
float lowAverage()
{
    const int iters = 512;
    float product = 1.0;
    float sum = 0.0;
    
    float smallest = 0.0;
    
    for(int i = 0; i < iters; i++)
    {
        float sound = texture(iChannel1, vec2(float(i)/float(iters), 0.75)).r;
        
        product *= sound;
        sum += sound;
    }
    return sum/float(iters);//max(sum/float(iters), pow(product, 1.0/float(iters)));
}

// Function 112
float wave(float t)
{
    return sin(2.0 * pi * t);
}

// Function 113
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

// Function 114
vec3 bars(vec3 color, vec3 background, int bars, sampler2D sound, vec2 uv)
{
    for(int i = 1; i <= bars; i++)
    {
        float len = 0.6 * texture(sound, vec2(float(i)/float(bars), 0.0)).x;
        background = bar(1.0 - color - len*.2, background, vec2(float(i)/float(bars+1), len*.9), vec2(1.0/float(bars+1)*0.8, len/1.5), uv);
    }
    return background;
}

// Function 115
vec4 GetVisual( vec2 U ) {
	vec2 uv = U.xy / R.xy*vec2(2.,2.4)-vec2(550.,400.)/Cover_Size;
    if(any(bvec4(greaterThan(uv,vec2(1.)),lessThan(uv,vec2(0.))))){
    	return vec4(0.);
    }
    float barStart = floor(uv.x * bars) / bars;
	if(uv.x - barStart < barGap || uv.x > barStart + barSize - barGap) 
		return vec4(0.0);
    else
    {
        float intensity = 0.0;
        for(float s = 0.0; s < barSize; s += barSize * sampleSize)
            intensity += texture(iChannel2, vec2(barStart + s, 0.0)).r;
		intensity *= sampleSize; 
		intensity = clamp(intensity, 0.005, 1.0); 
		float i = float(intensity > uv.y);
		return vec4(clamp(intensityToColour(intensity)*0.9,0.,1.) * i, clamp(i * uv.y*5.0,0.,1.));
    }
}

// Function 116
vec4 DrawBar(float playTime, vec2 uv, float size)
{
float viewScale = GetGameData(iChannel1,SPEED,iResolution).x;
    float measurePos = (playTime+uv.y*viewScale);
    float mask = step(uv.x,1.0)-step(uv.x, 0.0);
    float b = abs(sin(playTime*4.*3.1415));
    b = b *b*b*b;
    b = mix(0.8,0.4,b );
    return (vec4(pow(fract(measurePos),400.)))*b*mask;
}

// Function 117
vec2 capillaryWave(in float t, in float a, in float k, in float h) {
  float w = sqrt((gravity*k + waterTension*k*k*k)*tanh(k*h));
  return wave(t, a, k, w*iTime);
}

// Function 118
vec3 summedWaveNormal(vec2 p)
{
    float time = iTime;
	vec2 sum = vec2(0.0);
	sum += directionalWaveNormal(p, 0.5, normalize(vec2(1, 1)), 5.0, 1.5, time, 1.0);
	sum += directionalWaveNormal(p, 0.25,normalize(vec2(1.4, 1.0)), 11.0, 2.4, time, 1.5);
	sum += directionalWaveNormal(p, 0.125, normalize(vec2(-0.8, -1.0)), 10.0, 2.0, time, 2.0);
	sum += directionalWaveNormal(p, 0.0625, normalize(vec2(1.3, 1.0)), 15.0, 4.0, time, 2.2);
	sum += directionalWaveNormal(p, 0.03125, normalize(vec2(-1.7, -1.0)), 5.0, 1.8, time, 3.0);
	return normalize(vec3(-sum.x, -sum.y, 1.0));
}

// Function 119
float getwaves(vec2 position, int iterations){
	float iter = 0.0;
    float phase = 6.0;
    float speed = 2.0;
    float weight = 1.0;
    float w = 0.0;
    float ws = 0.0;
    for(int i=0;i<iterations;i++){
        vec2 p = vec2(sin(iter), cos(iter));
        vec2 res = wavedx(position, p, speed, phase, Time);
        position += normalize(p) * res.y * weight * DRAG_MULT;
        w += res.x * weight;
        iter += 12.0;
        ws += weight;
        weight = mix(weight, 0.0, 0.2);
        phase *= 1.18;
        speed *= 1.07;
    }
    return w / ws;
}

// Function 120
float squareWave(float time, float freq) {
    return fract(0.5 * tau * time * freq) * 2.0 - 1.0;
}

// Function 121
void DrawBare(out vec4 c){
	c = vec4(0.7, 0.7, 0.5, 1.0);
}

// Function 122
vec4 waveform(vec2 p, vec3 baseCol, float eps)
{
    float t = p.x + iTime;
    float envSq = exp(-10.*mod(t,0.5));
    envSq += 0.2*exp(-20.*mod(t,0.25)) * (1.+sin(t*4.));
    envSq += window(0.1,0.2,mod(t,0.25)) * (1.-sin(3.*t)) * 0.02;
    float envenv = 0.9 + 0.1*smoothstep(0.,0.5,mod(t,0.5)) * 0.8*window(0.,4.,mod(t,8.));
    envenv *= step(0.,t);
    envenv *= smoothstep(0.,2.,t);
    envSq *= envenv;
    float env = sqrt(envSq) * 0.7;
    vec3 col = pow(baseCol, 3.*vec3(abs(p.y) + 3.*(1.-envenv)));
    return vec4(col, smoothstep(env+eps,env-eps,abs(p.y)));
}

// Function 123
vec2 gravityWave(in float t, in float a, in float k, in float h) {
  float w = sqrt(gravity*k*tanh(k*h));
  return wave(t, a ,k, w*iTime);
}

// Function 124
vec4 paintHexBary(vec2 u,vec2 o,vec2 t,vec2 p0,vec2 p1,vec2 p2
){float cd=c2bdet(p0,p1,p2)
 ;vec4 c=vec4(1)     
 ;vec2 pro=p1-p0
 ;vec2 prt=p2-p0
 ;vec2 p3=p2-pro
 ;vec2 p4=p0-pro
 ;vec2 p5=p0-prt
 ;vec2 p6=p1-prt

  //corners in barycentric (makes skew and flip simpler
 ;vec3 b0=vec3(1,0,0)
 ,b1=vec3(0,1,0)
 ,b2=vec3(0,0,1)
 ,b3=vec3(1,-1,1)
 ,b4=vec3(2,-1,0)
 ,b5=vec3(2,0,-1)
 ,b6=vec3(1,1,-1)
     
 ;vec3 uu=c2b(u,p0,p1,p2,cd)//only for visualization
 ;vec3 oo=c2b(o,p0,p1,p2,cd)
 ;vec3 tt=c2b(t,p0,p1,p2,cd)
 ;float rotations=0.
 ;if(oo.x>1.){rotations=3.  //half rotation
  ;uu=r36bary(uu-vec3(1,-1,0))
  ;oo=r36bary(oo-vec3(1,-1,0))
  ;tt=r36bary(tt-vec3(1,-1,0))
 ;}if(oo.y<0.){rotations++ //sixt rotation //rotate corner points by +1
  ;uu=r16bary(uu-vec3(1,-1,0))
  ;oo=r16bary(oo-vec3(1,-1,0))
  ;tt=r16bary(tt-vec3(1,-1,0))
 ;}else if(oo.z<0.){rotations-- //negative sixt rotation //rotate corner points by -1
  ;uu=r56bary(uu-vec3(0,1,-1))
  ;oo=r56bary(oo-vec3(0,1,-1))
  ;tt=r56bary(tt-vec3(0,1,-1));}//rotations range[-1..4]
         
 ;if(rotations==-1.) rotate6by1(p6,p5,p4,p3,p2,p1)//reversed
 ;if(rotations== 4.) rotate6by2(p6,p5,p4,p3,p2,p1)//reversed 
 ;if(rotations== 3.) rotate6by3(p1,p2,p3,p4,p5,p6)
 ;if(rotations== 2.) rotate6by2(p1,p2,p3,p4,p5,p6)
 ;if(rotations== 1.) rotate6by1(p1,p2,p3,p4,p5,p6)
 //;if(rotations==1.||rotations==3.||rotations==-1.){
     //half of all cases are skewed the other way
     //instead of flipping our shit, we just do pur pN in barycentric coordinates
 //;}
 ;vec3 vs=c2b(o,p0,p1,p2,cd)
 ;vec3 center=sat((floor(uu+vec3(0,1,1))))

 ;c.y=min(c.y,ray(u,o,t)-.005)//green ray
 ;c.y=min(c.y,abs(segment(u,o,t)-.02))//green ray segment

 //;o=b2c(oo,p0,p1,p2)
 //;t=b2c(tt,p0,p1,p2)

 ;float bariOdet=c2bdet(o,p1,p2)//triangle with o instead of p0
 ;vec3 baryO=c2b(t,o,p1,p2,bariOdet)//triangles bariOcentric coords
 ;vec2 sect=intersectB2c(o,t,p1,p2)
     
 //;vec2 sect=p1-(p1-p2)*tt.z/(tt.z+tt.y)//projection of o on outer border (from center)
     //;vec2 sect=(oo.z*(tt.z+tt.y)-tt.z*(oo.z+oo.y))
 ;//c.y=uu.z*(oo.z+oo.y)-oo.z*(uu.z+uu.y) // iahas a line from corner to point
      
 ;//return mix(c,floor(1.-vec4(frustrumX,frustrumY,frustrumZ,0)),.5)
     
 ;//c.y=min(c.y,abs(sqrt(dd(u-sect))-.02))//intersection fail
  
;if(tt.z*(oo.z+oo.y)>oo.z*(tt.z+tt.y) //oh great its 2 ratios, and dividents flip sdes so its only mults
//;if(tt.z*(oo.z+oo.x)>oo.z*(tt.z+tt.x) //oh great its 2 ratios, and dividents flip sdes so its only mults
//;if(tt.y*(oo.y+oo.x)>oo.y*(tt.y+tt.x) //oh great its 2 ratios, and dividents flip sdes so its only mults

 //tt.y>oo.y//tt.z>tt.y//mirror symmetry on the whole thing
 ){uu.zy=uu.yz
  ;oo.zy=oo.yz
  ;tt.zy=tt.yz
  ;rotate6by3(p1,p3,p5,p2,p6,p4)//strided half rotation == mirror
  ;
  ;}  
/**/
     
  ;//projecting 2 points onto the axis is warely worth it
  ;//, only where we have 2 caes to check and within some other constrrains
  ;//p2=p1+(p2-p1)*.5
  ;//p4=p5+(p4-p5)*.5
     
 ;c.z=min(c.z,abs(length(u-p0)-.03))
 ;c.z=min(c.z,abs(length(u-p1)-.05))//blue circles
 ;c.z=min(c.z,abs(length(u-p2)-.07))//to distinguish inputs (handedness)
 ;c.z=min(c.z,abs(length(u-p3)-.08))//to distinguish inputs (handedness)
 ;c.z=min(c.z,abs(length(u-p4)-.09))//to distinguish inputs (handedness)
 ;c.w=min(c.w,abs(length(u-p5)-.09 ))//to distinguish inputs (handedness)
     
 ;vec3 X=vec3(p0.x,p1.x,p2.x)
 ;vec3 Y=vec3(p0.y,p1.y,p2.y)
 ;p1=b2c(b1,X,Y)    
 ;p2=b2c(b2,X,Y)    
 ;p3=b2c(b3,X,Y)   
 ;p4=b2c(b4,X,Y)  
 ;p5=b2c(b5,X,Y)  
 ;p6=b2c(b6,X,Y)  //testing bN 
 ;c.x=min(c.x,segment(u,p2,p1))
 ;c.x=min(c.x,segment(u,p3,p2))
 ;c.x=min(c.x,segment(u,p4,p3))
 ;c.x=min(c.x,segment(u,p5,p4))
 ;c.x=min(c.x,segment(u,p6,p5))
 ;c.x=min(c.x,segment(u,p1,p6))//red hexagon
 ;c.z=min(c.z,segment(u,p0,p1))
 ;c.z=min(c.z,segment(u,p0,p2))
 ;c.z=min(c.z,segment(u,p1,p2))//blue triangle (indicates pieslice or (o)
     

     
 ;//all mirroring and showing of it is done; 
 ;//all casting pout of the target is done
     
 ;float outU=min(floor(mav(abs(vec3(1,0,0)-uu))),1.)
 ;float outO=min(floor(mav(abs(vec3(1,0,0)-oo))),1.)
     

 ;vec2 intersection=vec2(0)
 ;//if(outO>0.)intersection=traceHexBaryOutside(oo,tt,b1,b2,b3,b4,b5,b6,X,Y)
 ;//else       intersection=traceHexBaryInside(uu,oo,tt,b1,b2,b3,b4,b5,b6,X,Y).xy
    
 
 ;c.w=min(c.w,abs(abs(length(u-intersection)-.03)-.01))//mark intersections
 
 ;c=smoothstep(.01,-.01,c)-.01
 ;c=pdOver(pdOver(v3(1,0,0,1)*c.x,v3(0,1,0,1)*c.y)
          ,pdOver(v3(0,0,1,1)*c.z,v3(1,1,0,1)*c.w))

 ;vec3 cake=(floor(uu)+vec3(0,1,1))
 ;cake=sat(floor(cake))
 ;//c.xyz=cake
 ;//if(cake.x>0.&&cake.y>0.)c.xyz=cake
 ;//if(cake.x>0.&&cake.y==0.)c.xyz=cake
 ;//c.xyz=mix(c.xyz,sat(cake-vec3(outU*.7)),.5)//inside
  
 ;//c.xyz+=sat(floor(uu+1.))//upper triangle is white
 ;//c.xyz+=suv(sat(floor(uu+1.)))*.33//upper triangle is white 4sect
                   /*
 ;//c.xyz+=sat(floor(uu+vec3(-1,1,1)))*.4 //opposing triangle is black
 ;//c.xyz+=suv(sat(floor(uu+vec3(-1,1,1))))*.5//opposing trianggle 4sected
 ;//c.xyz+=sat(floor(uu+1.))*.4 //upper triangle is white
 ;float sect4low=suv(sat(floor(uu+vec3(-1,1,1))))
 ;//float alowerT=suv(sat(floor(uu+vec3(-1,1,1))))
  //;c*=alowerT  
 ;float uIsInside=mav(abs(uu-vec3(1,0,0)))
 ;float aa=suv(sat(floor(uu+vec3(0,1,1))))
 ;float needsPush=float(aa<1.||uIsInside>1.)//bool(inT))
   /**/
 ;//float isSimple=float(suv(sat(floor(uu+vec3(-1,1,1))))==0.||mav(abs(uu-vec3(1,0,0)))>1.)//bool(inT))
 ;//c.xyz=mix(c.xyz,vec3(isSimple),.5)  
     /**/
 ;//branching case wether oo is outside or inside 
  
 ;//c.xyz=mix(c.xyz,traceHexBaryInside(uu,oo,tt,b1,b2,b3,b4,b5,b6,X,Y),5.)//to debug oo is uu
 ;vec3 fuck=innerFrustrum(uu.yz,oo.yz)
 ;c.xyz=mix(c.xyz,fuck,.5)
     
  
 ;return c;}

// Function 125
float wavesD(in float x, in float y)
{
    if(x < 32.) // ARR64 would be a really long line.
    {
        return ARR8(y,
		ARR32(x,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,2.,3.,3.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.),
		ARR32(x,2.,2.,2.,2.,2.,3.,3.,3.,3.,1.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,2.,5.,3.,3.,3.,3.),
		ARR32(x,3.,3.,3.,3.,3.,1.,0.,0.,0.,0.,0.,5.,5.,2.,2.,2.,2.,0.,0.,0.,0.,2.,5.,5.,5.,3.,3.,3.,3.,3.,3.,3.),
		ARR32(x,5.,5.,5.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,5.,5.,5.,5.,5.,5.,5.,5.,3.,3.,3.,3.,3.,3.,3.,3.,5.,0.),
		ARR32(x,0.,4.,4.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,3.,3.,3.,3.,3.,3.,3.,3.,5.,0.,0.,0.,0.,0.),
		ARR32(x,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.,3.,3.,3.,3.,3.,3.,3.,3.,0.,0.,0.,0.,0.,0.,2.,2.,2.,2.,2.,2.),
		ARR32(x,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.),
		2.);
    }
    else
    {
        x -= 32.;
        return ARR8(y,
		ARR32(x,5.,5.,3.,3.,3.,3.,3.,3.,3.,3.,3.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,3.,3.,3.,3.,3.,0.,0.,0.,0.,0.),
		ARR32(x,3.,3.,3.,5.,0.,0.,0.,0.,0.,0.,0.,4.,4.,4.,4.,5.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.),
		ARR32(x,5.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,5.,5.,5.,5.,5.,5.),
		ARR32(x,0.,0.,0.,0.,0.,2.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.,4.,4.,4.,4.,4.,4.,5.),
		ARR32(x,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.),
		ARR32(x,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.),
		2.,
		2.);
    }
}

// Function 126
float WaveRay (vec3 ro, vec3 rd)
{
  vec3 p;
  float dHit, h, s, sLo, sHi, f1, f2;
  dHit = dstFar;
  f1 = 0.4;
  f2 = 0.3;
  s = max (- (ro.y - 1.2 * f1) / rd.y, 0.);
  sLo = s;
  for (int j = 0; j < 80; j ++) {
    p = ro + s * rd;
    h = p.y - f1 * WaveHt (f2 * p.xz);
    if (h < 0.) break;
    sLo = s;
    s += max (0.3, h) + 0.005 * s;
    if (s >= dstFar) break;
  }
  if (h < 0.) {
    sHi = s;
    for (int j = 0; j < 5; j ++) {
      s = 0.5 * (sLo + sHi);
      p = ro + s * rd;
      h = step (0., p.y - f1 * WaveHt (f2 * p.xz));
      sLo += h * (s - sLo);
      sHi += (1. - h) * (s - sHi);
    }
    dHit = sHi;
  }
  return dHit;
}

// Function 127
vec3 WaveNf (vec3 p, float d)
{
  vec2 e = vec2 (max (0.01, 0.001 * d * d), 0.);
  float ht = WaveHt (p);
  return normalize (vec3 (ht - WaveHt (p + e.xyy), e.x, ht - WaveHt (p + e.yyx)));
}

// Function 128
float GetWavelet(vec2 p, float z, float scale) {
	p *= scale;
    
    vec2 id = floor(p);
    p = fract(p)-.5;
    
    float n = Hash21(id);
    p *= Rot(n*100.);
    float d = sin(p.x*10.+z);

    d *= smoothstep(.25, .0, dot(p,p));
    return d/scale;
}

// Function 129
vec4 bar(vec4 color, vec4 background, vec2 position, vec2 diemensions, vec2 uv)
{
    return capsule(color, background, vec4(position.x, position.y+diemensions.y/2.0, diemensions.x/2.0, diemensions.y/2.0), uv); //Just transform rectangle a little
}

// Function 130
float WaveHt (vec2 p)
{
  mat2 qRot = mat2 (0.8, -0.6, 0.6, 0.8);
  vec4 t4, v4;
  vec2 q, t, tw;
  float wFreq, wAmp, h;
  q = 0.5 * p + vec2 (0., tCur);
  h = 0.2 * sin (q.y) + 0.15 * sin (dot (q, vec2 (0.1, 1.2))) +
     0.1 * sin (dot (q, vec2 (-0.2, 1.4)));
  h *= 0.3 * (1. - smoothstep (0.8 * dstFar, dstFar, length (p)));
  q = p;
  wFreq = 0.5;
  wAmp = 0.05;
  tw = 0.5 * tCur * vec2 (1., -1.);
  for (int j = 0; j < 4; j ++) {
    q *= qRot;
    t4 = q.xyxy * wFreq + tw.xxyy;
    t = vec2 (Noisefv2 (t4.xy), Noisefv2 (t4.zw));
    t4 += 2. * t.xxyy - 1.;
    v4 = (1. - abs (sin (t4))) * (abs (sin (t4)) + abs (cos (t4)));
    t = 1. - sqrt (v4.xz * v4.yw);
    t *= t;
    h += wAmp * dot (t, t);
    wFreq *= 2.;
    wAmp *= 0.5;
  }
  return h;
}

// Function 131
vec2 waves(float time) {
	return vec2(0.2*noise(time)*sin(time), 0.2*noise(time+TWOPI*0.5)*cos(time))*(sin(1.*time*TWOPI)*0.25 + 0.75);
}

// Function 132
vec4 Bars(in vec2 uv)
{    
    uv *= 0.566; 
    uv.x = abs(uv.x);
    uv.y += 0.882;
    
    float bars = 1.0 - abs(mod(uv.x * 320.0, 2.0) - 1.0);
    
    Pixellize(uv, 80.0);
     
    float aud = step(uv.y, texture(iChannel0, vec2(uv.x, 0.0)).r);           
    
    vec4 col = mix(vec4(0.0), col1, bars * aud);
    col = mix(col, col3, col.w);
    
    return col;
}

// Function 133
float soundwave(float l, float t, float r1, float s0)
{
    s0 += 10.0f * log( (0.5f * PI * l) + (4.0f/r1) );
    
    return -smoothstep(0.0f, 200.0f, t + 4.0 * s0);
}

// Function 134
vec3 WaveNf (vec3 p, float d)
{
  vec2 e = vec2 (max (0.1, 5e-5 * d * d), 0.);
  float h = WaveHt (p);
  return normalize (vec3 (h - WaveHt (p + e.xyy), e.x, h - WaveHt (p + e.yyx)));
}

// Function 135
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

// Function 136
vec2 synthWave(float t)
{
    bool do_reverb = mod(t, 8.0) > 4.0;
    t = mod(t, 2.0);

    float f0 = 880.0;

    vec2 w = vec2(sine(t * f0) * exp(-t * 2.5));

    if (do_reverb)
    {
        vec2 r = lpnoise(t,  100.0)
               + lpnoise(t,  550.0) * 0.2
               + lpnoise(t, 1050.0) * 0.1 * exp(-t * 5.0);

        w += sine(t * f0 + r * 0.1) * exp(-t * 2.0);
        w -= sine(t * f0          ) * exp(-t * 2.0);
    }

    w *= 1.0 - exp(-t * 800.0);

    return w;
}

// Function 137
vec2 synthWave(float t
){bool do_reverb = mod(t, 8.0) > 4.0
 ;float m = mod(t, 2.0)
 ;vec2 f=cs(t-m)
 ;float f0 =220.*cos(t-m)+880.//FM over time makes it easier to debug timing.
 ;f0+=cos(2.*t)*440.
     // this cos()  illustrates the reverb in the dft(), as cos() breaks some symmetry ofer time.
 ;vec2 w = vec2(co2p(m * f0) * exp(-m * 2.5))
 #ifdef doReverb
 ;vec2 r=lpnoise(m, 100.)
        +lpnoise(m, 550.)*.2
        +lpnoise(m,1050.)*.1*exp(-t*5.)//3 octaves of overtones.
 ;float a=exp(-m*2.)//exponential falloff for reverb
 ;w+=(co2p(m*f0+r*.1)-co2p(m*f0))*a//differential of 2 offset samples. pigmentation-interferrence.
 #endif
 ;w*=1.-exp(-m*800.)//instrument falloff hull
 ;return w;}

// Function 138
vec4 wavesShadowPalette(in float x)
{
    if(x<4.)
    {
        return ARR4(x,  D_BLUE,
			   			WHITE,
			   			L_BLUE,
			   			WHITE);
    }
    else return ARR2(x-4., D_BLUE, L_BLUE);
}

// Function 139
float convert_wave_length_to_black_body_spectrum(float wave_length_nm, float temperature_in_kelvin)
{
    float wave_length_in_meters = wave_length_nm * 1e-9;
    float c1                    = 2.0 * pi * plancks_constant * speed_of_light * speed_of_light;
    float c2                    = plancks_constant * speed_of_light / boltzmanns_constant;
    float m                     = c1 / pow(wave_length_in_meters, 5.0) * 1.0 / (exp(c2 / (wave_length_in_meters * temperature_in_kelvin)) - 1.0);
    return m;
}

// Function 140
void wave(in int i, out float st, out float am, out vec2 di, out float fr, out float sp)
{
    //setup wave params
	st = abs(.35*rand(vec2(float(i))));//qi
	am = .02+.005*rand(vec2(float(i+2)));//ai
    di = (1.e0+vec2(1.7e0*rand(vec2(i,i+1)), 2.e0*rand(vec2(i+1,i))));//di
    fr = 6.+12.*rand(vec2(float(i+5)));//wi
    sp = 55.e-1+52.e-1*rand(vec2(float(i+4)));//phi
}

// Function 141
float getWave(float d, float t, float waveLength, float speed, float phase)
{
    float x = (d - t*speed + phase);
#if ENABLE_WAVE_HIGHLIGHTING
    float intensity = pow(fract(x), 5.);
#else
    float intensity = 1.;
#endif
    return sin(2. * PI * x / waveLength) * intensity;
}

// Function 142
vec3 waveToLms(float wave, vec3 amount)
{
	// LMS Gaussian function parameters for each cone type
	const vec3[5] lParams = vec3[](
		vec3(449.682, 21.6622, 2.36612),
		vec3(477.589, 11.0682, 1.39883),
		vec3(532.488, 25.7494, 34.0478),
		vec3(570.2, 5.91487, 0.243387),
		vec3(585.858, 34.98, 77.8669)
	);

	const vec3[5] mParams = vec3[](
		vec3(450.237, 19.5222, 3.33537),
		vec3(479.559, 13.3211, 3.68813),
		vec3(519.924, 17.1502, 9.68484),
		vec3(542.8, 3.27696, 0.105766),
		vec3(552.158, 33.3895, 77.9298)
	);

	const vec3[5] sParams = vec3[](
		vec3(467.661, 8.84562, 5.32073),
		vec3(422.211, 10.2028, 8.58498),
		vec3(443.084, 11.9848, 19.6347),
		vec3(444.863, 1.30608, -0.0330768),
		vec3(460.886, 25.7907, 24.9128)
	);

	// Color blindness simulation constants
	const vec3 white = inverse(xyzFromLms)*whiteE;
	const vec3 blue = inverse(xyzFromLms)*LmsRgb.primaries[2];
	const vec3 red = inverse(xyzFromLms)*LmsRgb.primaries[0];

	const vec2 prota = inverse(mat2(
		white.g, blue.g,
		white.b, blue.b
	))*vec2(white.r, blue.r);

	const vec2 deuta = inverse(mat2(
		white.r, blue.r,
		white.b, blue.b
	))*vec2(white.g, blue.g);

	const vec2 trita = inverse(mat2(
		white.r, red.r,
		white.g, red.g
	))*vec2(white.b, red.b);

	// Color blindness adjusted parameters for each cone type
	vec3[5] lParamsMod = vec3[](
		mix(lParams[0], mParams[0], amount.x),
		mix(lParams[1], mParams[1], amount.x),
		mix(lParams[2], mParams[2], amount.x),
		mix(lParams[3], mParams[3], amount.x),
		mix(lParams[4], mParams[4], amount.x)
	);

	vec3[5] mParamsMod = vec3[](
		mix(mParams[0], lParams[0], amount.y),
		mix(mParams[1], lParams[1], amount.y),
		mix(mParams[2], lParams[2], amount.y),
		mix(mParams[3], lParams[3], amount.y),
		mix(mParams[4], lParams[4], amount.y)
	);

	vec3[5] sParamsMod = vec3[](
		mix(sParams[0], mParams[0], amount.z),
		mix(sParams[1], mParams[1], amount.z),
		mix(sParams[2], mParams[2], amount.z),
		mix(sParams[3], mParams[3], amount.z),
		mix(sParams[4], mParams[4], amount.z)
	);

	// Color blindness adaptation matrices
	/*mat3 adaptProta = mat3(
		1.0 - amount.x, 0.0, 0.0,
		prota.x*amount.x, 1.0, 0.0,
		prota.y*amount.x, 0.0, 1.0
	);

	mat3 adaptDeuta = mat3(
		1.0, deuta.x*amount.y, 0.0,
		0.0, 1.0 - amount.y, 0.0,
		0.0, deuta.y*amount.y, 1.0
	);

	mat3 adaptTrita = mat3(
		1.0, 0.0, trita.x*amount.z,
		0.0, 1.0, trita.y*amount.z,
		0.0, 0.0, 1.0 - amount.z
	);

	mat3 adapt = adaptTrita*adaptDeuta*adaptProta;*/

	mat3 adapt = mat3(
		1.0 - amount.x, deuta.x*amount.y, trita.x*amount.z,
		prota.x*amount.x, 1.0 - amount.y, trita.y*amount.z,
		prota.y*amount.x, deuta.y*amount.y, 1.0 - amount.z
	);

	// Return the LMS values for the given wavelength
	return (adapt*vec3(1))*vec3(
		// L cone response curve
		gauss(wave, lParamsMod[0].x, lParamsMod[0].y, lParamsMod[0].z) +
		gauss(wave, lParamsMod[1].x, lParamsMod[1].y, lParamsMod[1].z) +
		gauss(wave, lParamsMod[2].x, lParamsMod[2].y, lParamsMod[2].z) +
		gauss(wave, lParamsMod[3].x, lParamsMod[3].y, lParamsMod[3].z) +
		gauss(wave, lParamsMod[4].x, lParamsMod[4].y, lParamsMod[4].z),

		// M cone response curve
		gauss(wave, mParamsMod[0].x, mParamsMod[0].y, mParamsMod[0].z) +
		gauss(wave, mParamsMod[1].x, mParamsMod[1].y, mParamsMod[1].z) +
		gauss(wave, mParamsMod[2].x, mParamsMod[2].y, mParamsMod[2].z) +
		gauss(wave, mParamsMod[3].x, mParamsMod[3].y, mParamsMod[3].z) +
		gauss(wave, mParamsMod[4].x, mParamsMod[4].y, mParamsMod[4].z),

		// S cone response curve
		gauss(wave, sParamsMod[0].x, sParamsMod[0].y, sParamsMod[0].z) +
		gauss(wave, sParamsMod[1].x, sParamsMod[1].y, sParamsMod[1].z) +
		gauss(wave, sParamsMod[2].x, sParamsMod[2].y, sParamsMod[2].z) +
		gauss(wave, sParamsMod[3].x, sParamsMod[3].y, sParamsMod[3].z) +
		gauss(wave, sParamsMod[4].x, sParamsMod[4].y, sParamsMod[4].z)
	)/sqrt(2.0*PI);
}

// Function 143
vec2 synthWave(float t)
{
    t = mod(t, 1.0);
    
    float f0 = 1010.;

    vec2 w = vec2(sine(t * f0) * exp(-t * 2.5));
    vec2 rw = w;
    
    //reverb simulation
    {
        vec2 r = lpnoise(t,  100.0)
               + lpnoise(t,  550.0) * 0.2
               + lpnoise(t, 1050.0) * 0.1 * exp(-t * 5.0);

    	rw += sine(t * f0 + r * 0.1) * exp(-t * 2.0);
    	rw -= sine(t * f0          ) * exp(-t * 2.0);
    }

    w = w*.3 + (rw*.7);
    
    w *= 1.0 - exp(-t * 800.0);

    return w;
}

// Function 144
float spiralWave(vec2 p, float ratio, float rate, float scale) {
    
    float r = length(p);
    
    float theta = atan(p.x,p.y);
   
    float logspiral = log(r)/ratio  + theta;
   
    return sin(rate*iTime + scale*logspiral);
    
}

// Function 145
vec3 Barycentric(vec3 b){
    return b / (b.x + b.y + b.z);
}

// Function 146
vec4 bar(vec2 uv){
    float mul = fftmul( uv.y );
    vec4 fft1 = fft(uv.x,0.0);
    
	return vec4(
        mul*float(fft1.r - fftMin > fftH*uv.y),
        mul*float(fft1.g - fftMin > fftH*uv.y) ,
        mul*float(fft1.b - fftMin > fftH*uv.y) ,
        1.0);
}

// Function 147
vec3 barycentricCoordinate(vec2 P,Pinwheel T)
{
    vec2 PA = P - T.A;
    vec2 PB = P - T.B;
    vec2 PC = P - T.C;
    
    vec3 r = vec3(
        det22(PB,PC),
        det22(PC,PA),
        det22(PA,PB)
    );
    
    return r / (r.x + r.y + r.z);
}

// Function 148
float wave4(float x){return sin(x*pi2*2.0)*step(fract(x),0.5);}

// Function 149
float triwave(float t) {
	return t*2.-max(0.,t*4.-2.);
}

// Function 150
float additiveWave(
    float amplitude,
    float baseFrequncy,
    float time)
{
    float pulse_width = clamp((1.0 + sin(time)) * 0.5, 0.01, 0.99);
    //float pulse_width = PULSE_WIDTH;

    float level = 
             pulseOsc(1.0, baseFrequncy,       time, 0.0, pulse_width)  * 0.75;
    level += pulseOsc(1.0, baseFrequncy * 2.0, time, 0.0, pulse_width)  * 0.25;
    level += pulseOsc(1.0, baseFrequncy * 3.0, time, 0.0, pulse_width)  * 0.05;
    level += pulseOsc(1.0, baseFrequncy * 8.0, time, 0.0, pulse_width)  * 0.02;
    //                                                      sum = 1.07
    
    return (level / 1.07) * amplitude;
}

// Function 151
vec3 r16bary(vec3 o){return mirrorBaryX(o).yxz;}

// Function 152
float wave(float x, float y){return //noisy epcircles
  sin(10.*x+10.*y)/5. 
 +sin(20.*x+15.*y)/3. 
 -sin( 4.*x+10.*y)/4. 
 +sin(y)          /2.
 +sin(x*x*y*20.)
 +sin(x*20.+ 4.)  /5.
 +sin(y*30.    )  /5.
 +sin(x        )  /4.
 ;}

// Function 153
float sinWave(float time, float freq) {
    return sin(tau * time * freq);
}

// Function 154
vec3 WaveNf (vec3 p, float d)
{
  vec3 vn;
  vec2 e;
  e = vec2 (max (0.01, 0.005 * d * d), 0.);
  p *= 0.3;
  vn.xz = 0.4 * (WaveHt (p.xz) - vec2 (WaveHt (p.xz + e.xy),  WaveHt (p.xz + e.yx)));
  vn.y = e.x;
  return normalize (vn);
}

// Function 155
float integral_squarewave(float x, float freq){
    x*=freq;
    
	float sgn=1.;
    
    if(fract(x)>=0.25 && fract(x)<0.75){
    	sgn=-1.;   
    }
    
	x=mod(x+.25,.5)-.25;
	return 2./pi*sgn*Si(x*(pi/2.)*float(n)*8.);
}

// Function 156
float getWaveformDeriv(float x, float mode, float e)
{
	return (smootherSample(vec2(x+e*0.5,mode),e)
		-smootherSample(vec2(x-e*0.5,mode),e))/e;
}

// Function 157
vec3 WavelengthToXYZ( float f )
{    
    return xyzFit_1931( f );    
}

// Function 158
void playBar3() {
    placeKicks(beat2, beatH4, 0.0, 1.0);
    placeKicks(beat2, beatH4, 2.5, 3.0);
    placeSnares(beatH2 - 1.0, beatH4, 0.0, 2.0);
    placeSnares(beat2, beatH4, 3.5, 4.0);
    placeSnares(beat4, beatH4, 1.75, 2.0);
    placeSnares(beat4, beatH4, 2.25, 2.5);
    
    if (SYNTHS) {
    	placeSynths(beat4, beatH4, 1.85, 2.25, 0.0);
    	placeSynths(beat4, beatH4, 2.25, 2.5, 1.0);
    }

    stereo += vec2(hihatClosed(beat2)) * vec2(0.7, 1.0);
}

// Function 159
float round_wave(vec2 r,vec2 p){
	return sin(100.0*length(p-r)-iDate.w*20.0)*0.5+0.5;
}

// Function 160
float tWave(float x, float amplitude, float frequency){
      return abs((fract(x*frequency) *2.)-1.) * amplitude;   
}

// Function 161
float wavesC(in float x, in float y)
{
    if(x < 32.) // ARR64 would be a really long line.
    {
        return ARR8(y,
        ARR32(x,3.,3.,3.,3.,0.,0.,0.,0.,0.,0.,0.,3.,3.,3.,0.,0.,0.,0.,0.,0.,2.,3.,3.,3.,2.,0.,0.,0.,0.,0.,0.,0.),
        ARR32(x,2.,0.,0.,0.,5.,3.,3.,2.,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,3.,3.,3.,2.,2.,2.,3.,2.,2.,0.,0.,3.,3.,3.),
        ARR32(x,0.,0.,0.,0.,0.,0.,0.,4.,4.,4.,2.,2.,2.,2.,3.,3.,3.,3.,3.,5.,0.,0.,0.,0.,0.,0.,5.,5.,3.,3.,3.,3.),
        ARR32(x,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.,4.,4.,5.,5.,5.,5.,5.,5.,0.,0.,0.,0.,0.,5.,5.,5.,3.,3.,3.,1.,3.,4.),
        ARR32(x,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,4.,4.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,3.,3.,1.,1.,3.,3.,4.,0.),
        ARR32(x,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,2.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,4.,0.,0.,0.,0.),
        ARR32(x,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,2.,2.,2.),
        2.);

    }
    else
    {
        x -= 32.;
        return ARR8(y,
		ARR32(x,0.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.,4.,5.,5.,5.,3.,3.,3.,3.,3.),
		ARR32(x,3.,3.,3.,3.,5.,2.,2.,2.,0.,0.,0.,0.,0.,0.,4.,4.,4.,4.,4.,4.,5.,5.,5.,5.,5.,3.,3.,3.,3.,3.,3.,3.),
		ARR32(x,3.,4.,4.,4.,4.,5.,5.,5.,5.,5.,5.,5.,4.,4.,4.,4.,5.,5.,5.,5.,5.,5.,5.,1.,1.,1.,3.,3.,3.,2.,0.,0.),
		ARR32(x,4.,4.,0.,0.,0.,0.,0.,0.,0.,0.,2.,5.,5.,5.,5.,5.,3.,1.,2.,2.,2.,1.,3.,3.,3.,3.,4.,0.,0.,0.,2.,2.),
		ARR32(x,0.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,2.,2.,1.,3.,3.,5.,5.,0.,0.,0.,2.,2.,2.,2.,2.,2.),
		ARR32(x,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.),
		2.,
		2.);
    }
}

// Function 162
vec3 visualizeMotionVectors(vec2 pos)
{
	float lineThickness = 1.0f;
	float blockSize = 32.0f;

	// Divide the frame into blocks of "blockSize x blockSize" pixels
    // and get the screen coordinates of the pixel in the center of closest block
	vec2 currentCenterPos = floor(pos / blockSize) * blockSize + (blockSize * 0.5f);

	// Load motion vector for pixel in the center of the block
    vec2 previousCenterPos01 = getPreviousFrameUVs(currentCenterPos);
    vec2 previousCenterPos = previousCenterPos01 * iResolution.xy;

	// Reject invalid motion vector (e.g. for failed reprojection, indicated by INVALID_UVS value)
    bool rejectReprojection = (previousCenterPos01.x == INVALID_UVS.x && previousCenterPos01.y == INVALID_UVS.y);
    if (rejectReprojection) previousCenterPos = currentCenterPos;

	// Get distance of this pixel from motion vector line on the screen
	float lineDistance = distanceFromLineSegment(pos, currentCenterPos, previousCenterPos);

	// Draw line based on distance
	return (lineDistance < lineThickness) 
        ? vec3(1.0f, 1.0f, 1.0f) 
        : vec3(0.0f, 0.0f, 0.0f);
}

// Function 163
float waveDephthAt(vec3 coords) {
    return 0.10 
        + 0.03 * cos(0.5 * coords.x + 1.2 * sin(0.5*coords.z + iTime) + 5.*iTime) 
        + 0.002 * cos(cos(2.*coords.x) + sin(2.*coords.z) + 4. * iTime)
        ;   
}

// Function 164
float getwaves(vec2 position, int iterations){
    position *= 0.1;
	position += time * 0.1;
	float iter = 0.0;
    float phase = 6.0;
    float speed = 2.0;
    float weight = 1.0;
    float w = 0.0;
    float ws = 0.0;
    for(int i=0;i<iterations;i++){
        vec2 p = vec2(sin(iter), cos(iter));
        vec2 res = wavedx(position, p, speed, phase, iTime);
        position += normalize(p) * res.y * weight * DRAG_MULT;
        w += res.x * weight;
        iter += 12.0;
        ws += weight;
        weight = mix(weight, 0.0, 0.2);
        phase *= 1.18;
        speed *= 1.07;
    }
    return (w / ws);// * supernoise3dX(0.3 *vec3(position.x, position.y, 0.0) + iTime * 0.1);
}

// Function 165
vec3 bary_from_planar(vec2 p) { 

    vec2 bxy = bary_mat * (p - planar_verts[2]);
    return vec3(bxy, 1.-bxy.x-bxy.y);
    
}

// Function 166
vec3 Bars(vec2 f)
{
    vec2 uv = f / iResolution.xy;
    vec3 color = vec3(0.0);
    color += calcSine(uv, 2.0, 0.25, 0.0, 0.5, vec3(0.0, 0.0, 1.0), 0.10, 3.0);
    color += calcSine(uv, 2.6, 0.15, 0.2, 0.5, vec3(0.0, 1.0, 0.0), 0.10, 1.0);
    color += calcSine(uv, 0.9, 0.35, 0.4, 0.5, vec3(1.0, 0.0, 0.0), 0.10, 1.0);
    return color;
}

// Function 167
float WaveHt (vec3 p)
{
  const mat2 qRot = mat2 (1.6, -1.2, 1.2, 1.6);
  vec4 t4, ta4, v4;
  vec2 q2, t2, v2;
  float wFreq, wAmp, pRough, ht;
  wFreq = 0.16;  wAmp = 0.6;  pRough = 5.;
  q2 = p.xz + waterDisp.xz;
  ht = 0.;
  for (int j = 0; j < 5; j ++) {
    t2 = 1.1 * tCur * vec2 (1., -1.);
    t4 = vec4 (q2 + t2.xx, q2 + t2.yy) * wFreq;
    t2 = vec2 (Noisefv2 (t4.xy), Noisefv2 (t4.zw));
    t4 += 2. * vec4 (t2.xx, t2.yy) - 1.;
    ta4 = abs (sin (t4));
    v4 = (1. - ta4) * (ta4 + abs (cos (t4)));
    v2 = pow (1. - pow (v4.xz * v4.yw, vec2 (0.65)), vec2 (pRough));
    ht += (v2.x + v2.y) * wAmp;
    q2 *= qRot;  wFreq *= 1.9;  wAmp *= 0.22;
    pRough = 0.8 * pRough + 0.2;
  }
  return ht;
}

// Function 168
float GetWaveDisplacement(vec3 p)
{
    float time = iTime;
	float waveStrength = 0.1;
	float frequency = 4.0;
	float waveSpeed = 0.15;
	float rotSpeed = 0.1;
	float twist = 0.24;
	float falloffRange = 2.0;	// the other values have been tweaked around this...
	
	float d = length(p);
	p.xz *= rotate(d*twist+(time*rotSpeed)*TAU);
	vec2 dv = p.xz*0.15;
	d = length(dv);
	d = clamp(d,0.0,falloffRange);
	float d2 = d-falloffRange;
	float t = fract(time*waveSpeed)*TAU;
	float s = sin(frequency*d*d+t);
	float k = s * waveStrength * d2*d2;
	k *= p.x*p.z*0.5;
	//k-= 0.4;					// mix it up a little...
	k -= sin(fract(time*0.1)*TAU)*0.4*d2;			// really mix it up... :)
	k = smoothstep(0.0,0.45,k*k);
	return k;
}

// Function 169
vec3 mirrorBaryZ(vec3 o){return vec3(o.xy+o.z,-o.z);}

// Function 170
vec3 phasedfractalwave(vec3 p, float phaseOffset)
{       

    // rotation matrix for noise octaves
    mat3 octaveMatrix = mat3( 0.00,  0.80,  0.60,
                              -0.80,  0.36, -0.48,
                              -0.60, -0.48,  0.64 );

    vec3 signal = .5 * sin(p + phaseOffset);
    p = octaveMatrix*p*1.32;    
    signal += .3 * sin(p + 2.2 * phaseOffset);
    p = octaveMatrix*p*1.83;
    signal += .2 * sin(p + 5.4 * phaseOffset);

    signal /= 1.0;
    return signal;
}

// Function 171
float trianglewave(float n, float x, float l, float phase){
    float k = n*2.0+1.0;
    return amp*8.0/(pi*pi)/(k*k)*cos(pi*n)*sin(k*pi*x/l+phase);
}

// Function 172
vec3 sphere_from_bary(vec3 b) {
    return tri_verts * b;
}

// Function 173
float d_sinewave( vec2 p )
{
    float time = iTime;
    p.x -=
        //(.8+.25*sin(time))*
        p.y; // bent forward
    return p.y - sin(p.x);
}

// Function 174
void GPU_FFT_Visual(vec2 u,inout vec4 c){
	//u.x -= 100.;
    //u.y -= (0.55*iResolution.y); //128.
    u.x = floor(u.x/iResolution.y*180.)-30.;
    u.y = u.y/iResolution.y*350.-170.;
    //show visual effect
    if(u.x-0.5 < 256. && u.x>=0. && u.y>0.){
        vec2 xy = texture(iChannel0,vec2(u.x+0.5,9.5)/iChannelResolution[0].xy).xy;
        //Energy
        vec4 cc = vec4(length(xy));
        //Really Energy 
    	cc = u.x == 0. ? cc/512. : cc/256.;
        cc *= u.x == 0. ? 2.: 20.;
        cc = 20.*log(cc);
        cc = vec4(clamp(cc.r,1.,120.),0.,0.,0.);
        //show visual effect
        if(floor(u.y) < cc.r ){
    		c.g = 1.;
        }
    }   
}

// Function 175
vec3 WavelengthToXYZ( float f )
{    
    //return xyzFit_1931( f ) * mXYZtoSRGB;
    
    return WavelengthToXYZLinear( f );
}

// Function 176
float wave6(float x) {return step(fract(x),0.5)*2.0-1.0;}

// Function 177
vec2 CxyWaves(float x, float y) {
    return DemoWaves(x, y);
}

// Function 178
float getWavelength(float octave)
{
	const float maximumWavelength = 50.0;
    
    float wavelength = TAU * maximumWavelength / pow(2.0, octave);

    // Make it aperiodic with a random factor
    wavelength *= 0.75 + 0.5 * hash11(1.337 * octave);
    
    return wavelength;
}

// Function 179
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

// Function 180
vec2 wavedx(vec2 position, vec2 direction, float speed, float frequency, float timeshift) {
	direction = normalize(direction);
    float x = dot(direction, position) * frequency + timeshift * speed;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return vec2(wave, -dx);
}

// Function 181
float WaveRay (vec3 ro, vec3 rd)
{
  vec3 p;
  float dHit, h, s, sLo, sHi;
  s = 0.;
  sLo = 0.;
  dHit = dstFar;
  for (int j = 0; j < 150; j ++) {
    p = ro + s * rd;
    h = p.y - WaveHt (p);
    if (h < 0.) break;
    sLo = s;
    s += max (0.2, h) + 0.005 * s;
    if (s > dstFar) break;
  }
  if (h < 0.) {
    sHi = s;
    for (int j = 0; j < 7; j ++) {
      s = 0.5 * (sLo + sHi);
      p = ro + s * rd;
      h = step (0., p.y - WaveHt (p));
      sLo += h * (s - sLo);
      sHi += (1. - h) * (s - sHi);
    }
    dHit = sHi;
  }
  return dHit;
}

// Function 182
vec3 visualize_region(vec2 uv, float dist, vec2 mouse)
{
	float mouse_circle = float(circle(uv - mouse, abs(dist)) < 0.0);
	return vec3(0.0, mouse_circle * 0.3, 0.0);
}

// Function 183
vec2 planar_from_bary(vec3 b) {
    return planar_verts * b;
}

// Function 184
void poly_from_bary() {
    
    bool was_select = false;
    
    for (int i=0; i<4; ++i) {
        if (abs(spoint_selector[i] - 1.) < TOL) {
            poly_vertex = tri_spoints[i];
            bary_poly_vertex = bary_from_sphere(poly_vertex);
            was_select = true;
        }
    }
    
    if (!was_select) {
        poly_vertex = normalize(sphere_from_bary(bary_poly_vertex.xyz));
    }
    
}

// Function 185
float wave5(float x) {return abs(sin(x*pi2*2.0))*step(fract(x),0.5);}

// Function 186
void initWaveDoubleSlit(inout vec4 col, vec2 pos)
{
    vec2 p = pos.xy / iResolution.xy - 0.5;
    p.x *= iResolution.x / iResolution.y;
    
   	//vec2 wavedir=normalize(vec2(1,1));
    //vec2  wavepos = vec2(-0.5,-0.225);
    vec2  wavepos = vec2(-0.35,0.0);
    vec2  wavedir=normalize(vec2(1,0));
    float wavelen = WAVELEN;
    float wavesize = .2;
    vec2 K = PI2/wavelen*wavedir;
    col.xy = .5*exp(-sq(length(p-wavepos)/wavesize)) * cis(dot(K,p));
}

// Function 187
vec3 draw_bars(vec2 uv) {
    uv.y *= 2.0;
    float freq = discretize(uv.x, NUM_BARS);
    float gain = get_freq_gain(freq);
    vec3 col = mix3(COL_LOW, COL_MID, COL_HIGH, gain);
    float cut = step(uv.y, max(gain, 0.01));
    return col * cut;
}

// Function 188
float lengthBary(vec3 p,vec3 t){return -suv(t*p.yxx*p.zzy);}

// Function 189
float getWave(in float time){
	float Tau=6.2831853;
 
    float C=Tau*261.6256;
    
    return sin(time*C);
}

// Function 190
float octaveWave(float time, float freq, float amp, float lac, float pers, int octaves) {
    float result = 0.;
    float frequencyInc = 1.;
    float amplitudeInc = 1.;
    for(int i = 0; i < octaves; i++) {
        result += sin(time*freq*frequencyInc)*amp*amplitudeInc;
        frequencyInc *= lac;
        amplitudeInc *= pers;
    }
    return result;
}

// Function 191
float WaveHt (vec2 p)
{
  mat2 qRot;
  vec4 t4, v4;
  vec2 t2;
  float wFreq, wAmp, ht, tWav;
  tWav = 0.2 * tCur;
  qRot = mat2 (0.8, -0.6, 0.6, 0.8);
  wFreq = 1.;
  wAmp = 1.;
  ht = 0.;
  for (int j = 0; j < 3; j ++) {
    p *= qRot;
    t4 = (p.xyxy + tWav * vec2 (1., -1.).xxyy) * wFreq;
    t4 += 2. * vec2 (Noisefv2 (t4.xy), Noisefv2 (t4.zw)).xxyy - 1.;
    t4 = abs (sin (t4));
    v4 = (1. - t4) * (t4 + sqrt (1. - t4 * t4));
    t2 = 1. - sqrt (v4.xz * v4.yw);
    t2 *= t2;
    t2 *= t2;
    ht += wAmp * dot (t2, t2);
    wFreq *= 2.;
    wAmp *= 0.5;
  }
  return ht;
}

// Function 192
float triwave(float x) {
    return 1.-abs(fract(x)-.5)*2.;
}

// Function 193
float WaveHt (vec3 p)
{
  const mat2 qRot = mat2 (1.6, -1.2, 1.2, 1.6);
  vec4 t4, ta4, v4;
  vec2 q2, t2, v2;
  float wFreq, wAmp, pRough, ht;
  wFreq = 0.25;  wAmp = 0.25;  pRough = 5.;
  q2 = p.xz + waterDisp.xz;
  ht = 0.;
  for (int j = 0; j < 5; j ++) {
    t2 = 1.1 * tCur * vec2 (1., -1.);
    t4 = vec4 (q2 + t2.xx, q2 + t2.yy) * wFreq;
    t2 = vec2 (Noisefv2 (t4.xy), Noisefv2 (t4.zw));
    t4 += 2. * vec4 (t2.xx, t2.yy) - 1.;
    ta4 = abs (sin (t4));
    v4 = (1. - ta4) * (ta4 + abs (cos (t4)));
    v2 = pow (1. - sqrt (v4.xz * v4.yw), vec2 (pRough));
    ht += (v2.x + v2.y) * wAmp;
    q2 *= qRot;  wFreq *= 2.;  wAmp *= 0.25;
    pRough = 0.8 * pRough + 0.2;
  }
  return ht;
}

// Function 194
void playBar1() {
    placeKicks(beat2, beatH4, 0.0, 1.0);
    placeKicks(beat4, beatH4, 2.5, 3.0);
    placeSnares(beatH2 - 1.0);
    placeSnares(beat4, beatH4, 1.75, 2.0);
    placeSnares(beat4, beatH4, 2.25, 2.5);
    placeSnares(beat4, beatH4, 3.75, 4.0);

    hihatClosed(beat2);
}

// Function 195
vec2 wavedx(vec2 position, vec2 direction, float speed, float frequency, float timeshift) {
  float x = dot(direction, position) * frequency + timeshift * speed;
  float wave = exp(sin(x) - 1.0);
  float dx = wave * cos(x);
  return vec2(wave, -dx);
}

// Function 196
float finiteDiffWave(sampler2D old, sampler2D curr, vec2 coord, vec2 res, float tau, float att)
{
    // d^u/dt^2 = div(grad(u))
    // (u''(0,0) - 2*u'(0,0) + u(0,0))/tau^2 = (u'(1,0)-2*u'(0,0)+u'(-1,0))/h1^2 + 
    // + (u'(0,1)-2*u'(0,0)+u'(0,-1))/h2^2
    // set h1 = h2 = 1
    // u''(0,0) = -u(0,0) + 2*u'(0,0) + tau^2 * (u'(1,0)+u'(-1,0)+u'(0,1)+u'(0,-1)-4*u'(0,0))
    // u''(0,0) = 2*u'(0,0) - u(0,0)  + tau^2 * (u'(1,0) + u'(0,1) + u'(-1,0) + u'(0,-1) - 4*u'(0,0))
    // 2 - 4*tau*tau >= 0 -> tau<= sqrt(0.5) ~ 0.7
    float p = 1.0 - att;
    // using naive laplacian:
    // float col = 2.0 * u(0,0) - up(0,0) + tau * tau * (u(1,0) + u(0,1) + u(-1,0) + u(0,-1) - 4.0*u(0,0));
    
    // using rotationally ionvariant laplacian:
    float col = 2.0 * u(0,0) - up(0,0) + tau * tau * 1.0/6.0 * 
        (4.0*(u(1,0) + u(0,1) + u(-1,0) + u(0,-1)) +
         u(1,1) + u(-1,1) + u(1,-1) + u(-1,-1) - 20.0*u(0,0));
    
    // 2.0 - 20.0/6.0*tau*tau >= 0 ->  tau<=sqrt(0.6) ~ 0.77
    
    return p*col;
}

// Function 197
float dfWave(vec2 uv, float r)
{
    r *= 0.5;
    
    uv.x -= r;
    uv.x = mod(uv.x, 4.0*r);
    
    vec2 offs = vec2(2.0*r, 0);
    
    float d = dfSemiCircle(mirrorX(uv - offs) - offs, r);
    d = min(d, dfSemiCircle(flipY(mirrorX(uv) - offs), r)); 
    
    return d;
}

// Function 198
float shockwave(float rawX) {
	float x = rawX - STARTTIME/SLOWDOWN;
	float shock = sin(x*50.-PI)/(x*50.-PI);
	float squareFunction = ceil(x/1e10)-ceil((x-SHOCKTIME)/1e10);
	return shock*squareFunction;
}

// Function 199
float sample_wavelength(void) {
    return (float(spectrum_width))*sample_uniform() + float(spectrum_start);
}

// Function 200
vec3 WavelengthToXYZLinear( float fWavelength )
{
    float fPos = ( fWavelength - standardObserver1931_w_min ) / (standardObserver1931_w_max - standardObserver1931_w_min);
    float fIndex = fPos * float(standardObserver1931_length);
    float fFloorIndex = floor(fIndex);
    float fBlend = clamp( fIndex - fFloorIndex, 0.0, 1.0 );
    int iIndex0 = int(fFloorIndex);
    int iIndex1 = iIndex0 + 1;
    iIndex0 = min( iIndex0, standardObserver1931_length - 1);
    iIndex1 = min( iIndex1, standardObserver1931_length - 1);    
    return mix( standardObserver1931[iIndex0], standardObserver1931[iIndex1], fBlend );
}

// Function 201
float SquareWave25( float f, float x )
{
    return floor( 4.0 * floor( f * x ) - floor( 4.0 * f * x ) + 1.0 );
}

// Function 202
float triwave2(float x) {
    return fract(x);
}

// Function 203
vec3 mirrorBaryY(vec3 o){return mirrorBaryZ(o.xzy).xzy;}

// Function 204
float bars(vec2 uv)
{
    vec2 ouv = uv;
    float rep = 0.08;
	float idx = float(int((uv.x+rep*.5)/rep));
    uv.x = mod(uv.x+rep*.5, rep)-rep*.5;
    float h = texelFetch(iChannel0, ivec2(int((idx+8.5)*7.), 0), 0).x;
    float sqr = sdSqr(uv, vec2(.00001,.1+.2*h));
    return max(sqr, -(abs(uv.y)-.05));
}

// Function 205
float wave(float x, float k, float c, float t)
{
    float X = x - c * t;
    return sin(k * X) * exp(-X * X);
}

// Function 206
float wave(float theta, vec2 p) {
	return (cos(dot(p,vec2(cos(theta),sin(theta)))) + 1.) / 2.;
}

// Function 207
float waves(vec2 p)
{
    float t = iTime;
    for (int i = 0; i < 10; i++) p.y += n1D(p.x*float(i)/2.+t*.05)*.05;
    p *= 3.;
    vec2 i_p = floor(p);
    if (mod(i_p.y, 2.) != 0.) p.y += n1D(abs(p.y+p.x+.5))*.5;
    else p.y += n1D(abs(p.y+p.x+.5))*.5;
    p = fract(p)*2.-1.;
    return smoothstep(.0, 1., abs(p.y));
}

// Function 208
vec3 blastBar(float t)
{
    return vec3(1.0,
                smoothstep(0.13,0.66,t),
                smoothstep(0.66,1.00,t));
}

// Function 209
float squarewave(float n, float x, float l, float phase){
    n = n*2.0+1.0;
    return amp*4.0/(n*pi)*sin(n*pi*x/l+phase);
}

// Function 210
float SquareWave50( float f, float x )
{
    return floor( 2.0 * floor( f * x ) - floor( 2.0 * f * x ) + 1.0 );
}

// Function 211
float 	UIStyle_TitleBarHeight() 		{ return 24.0; }

// Function 212
vec3 WavelengthToConeLinear( float fWavelength )
{
    float fPos = ( fWavelength - standardObserver1931_w_min ) / (standardObserver1931_w_max - standardObserver1931_w_min);
    float fIndex = fPos * float(standardObserver1931_length);
    float fFloorIndex = floor(fIndex);
    float fBlend = clamp( fIndex - fFloorIndex, 0.0, 1.0 );
    int iIndex0 = int(fFloorIndex);
    int iIndex1 = iIndex0 + 1;
    iIndex0 = min( iIndex0, standardObserver1931_length - 1);
    iIndex1 = min( iIndex1, standardObserver1931_length - 1);    
    return mix( coneFundamentals[iIndex0], coneFundamentals[iIndex1], fBlend );
}

// Function 213
float wave_pad(float t, float f0, float p0)
{
    float op1 = tsaw(p0 + t * f0 * 0.5000, 0.02);
    float op2 = tsaw(p0 + t * f0 * 0.5086, 0.02);

    return op1 - op2;
}

// Function 214
vec3 barycentric(vec2 a, vec2 b, vec2 c, vec2 p)
{
    float d = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    float alpha = ((b.y - c.y) * (p.x - c.x)+(c.x - b.x) * (p.y - c.y)) / d;
    float beta = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / d;
    float gamma = 1.0 - alpha - beta;
    return vec3(alpha, beta, gamma);
}

// Function 215
vec2 gravityWaveD(in float t, in float a, in float k, in float h) {
  float w = sqrt(gravity*k*tanh(k*h));
  return dwave(t, a, k, w*iTime);
}

// Function 216
vec4 traceWave( in vec3 ro, in vec3 rd )
{
    float tmin = 1.0 ;
    float tmax = 50.0;
    
	float precis = 0.01;
    float t = tmin;
    float d = 0.0;
    float ws = 0.0; // wave coordinates
    vec2 ph = vec2(0,0); // wave phase
    float bd = 0.0; // wave bounding cylinder df
    
    float tb = traceWaveBound(ro,rd);
    
    // the distance field of breaking wave is distorted severely, 
    // so we need more iterations and smaller marching steps 
    for( int i=0; i<128; i++ )
    {
	    vec4 hit = mapWave( ro+rd*t, ph );
        d = hit.x; ws = hit.y; bd = hit.z;
        
        if( d <precis || t>tmax ) break;
        #if 0
        t += (bd > 0.0 )? d *0.3 : d*0.1;
        #else
        if ( tb > -0.5) 
           t += ( bd > 0.0 )? d *0.5 : d *0.1;
	    else
           t += d;
        #endif 
    }

    if( t>tmax ) t=-1.0;
    return vec4( t, ws, ph.x, ph.y );
}

// Function 217
float waveDerivate(float x)
{
    return (cos(x - cos(x)/pi)*(pi + sin(x)))/pi;
}

// Function 218
vec3 mirrorBaryX(vec3 o){return mirrorBaryZ(o.zyx).zyx;}

// Function 219
float wavething(int n, float x){
    float l = ln2 * float(n) + log(x);
    l -= mean;
    return exp(-l * l / stdDev) / 2.0;
}

// Function 220
float sinewave(float time) {
    return sin(TWO_PI * time);
}

// Function 221
vec2 synthWave(float t, Mixer m)
{
    vec2 w = vec2(0.0);
    vec2 n = vec2(0.0);
    float fq = 0.0;

    // Lead
    n = lead_pat(loop(t, 192.0));
    fq = note2Freq(n.x) * 0.25;
    w += m.lead  * ins_lead(n.y, fq);
    w += m.lead2 * ins_lead(n.y, fq * 0.5);

    n = lead_pat(loop(t - 12.0 / STEP, 192.0));
    fq = note2Freq(n.x) * 0.5;
    w += m.lead3 * ins_pad(n.y, fq) * exp(-n.y * 3.0);

    // String pad
    n = pad_pat(loop(t, 192.0));
    fq = note2Freq(n.x) / 8.0;
    w += m.pad * ins_pad(n.y, fq);

    n = rhythm_pat(loop(t, 12.0));

    // Compress dynamic range
    w *= 1.0 - exp(-n.y * 8.0) * 0.7;

    // Bass
    w += m.bass * ins_bass(n.y, fq * 0.5);
    w += m.bass * ins_bass(n.y, fq * 1.0) * 0.3;

    // Drum
    if (n.x >= 1.0) w += m.drum * ins_drum(n.y, n.x == 2.0 ? 116.5 : 130.8);

    // Snare
    if (n.x >= 2.0) w += m.snare * ins_snare(n.y, 116.75) * (n.x == 3.0 ? 0.7 : 1.0);

    // Bell
    w += m.bell * vec2(wave_bell(loop(t, 12.0), 175.0));

    return w;
}

// Function 222
vec3 convert_wave_length_to_XYZ(float wave_length_nm)
{
	// The analytical approximation assumes the wave length is measured in angstroms
	float wave_length_angstroms = wave_length_nm * 10.0;
    
	// Approximate with the sum of gaussian functions
	vec3 XYZ;
	XYZ.x = gaussian(wave_length_angstroms, 1.056, 5998.0, 379.0, 310.0) + gaussian(wave_length_angstroms, 0.362, 4420.0, 160.0, 267.0) + gaussian(wave_length_angstroms, -0.065, 5011.0, 204.0, 262.0);
	XYZ.y = gaussian(wave_length_angstroms, 0.821, 5688.0, 469.0, 405.0) + gaussian(wave_length_angstroms, 0.286, 5309.0, 163.0, 311.0);
	XYZ.z = gaussian(wave_length_angstroms, 1.217, 4370.0, 118.0, 360.0) + gaussian(wave_length_angstroms, 0.681, 4590.0, 260.0, 138.0);
    
    // Done
	return XYZ;
}

// Function 223
float waveDist(vec3 p) {
    vec2 uv = p.xz;
    
    float d = 0.0;
    float freq = WAVE_FREQ;
    float amp = WAVE_AMP;
    for (int i = 0; i < 3; ++i) {
        d += amp * oct((uv + (2.0 + iTime) * WAVE_SPEED) * freq);
        d += amp * oct((uv - (2.0 + iTime) * WAVE_SPEED) * freq);
        amp *= 0.5;
        freq *= 1.5;
    }
    
    return d;
}

// Function 224
float wave(int x)
{
    return texelFetch(iChannel0, ivec2(x, 1), 0).r;
}

// Function 225
float additive_squarewave(float x, float freq){

	float sum=0.;

    for(int k=0;k<max_n;k++){
        if(k<n){
			sum+=sin(mod(2.*pi*(2.*float(k)+1.)*freq*x,2.*pi))/(2.*float(k)+1.);
        }
	}

	return sum*4./pi;
}

// Function 226
bool wave(vec2 r, float box, float width, float wavewidth, float Time){
	if(mod(r.x + width / 2.0, box) <= width || mod(r.y + width / 2.0, box) <= width){
        float dis = abs(r.x) + abs(r.y);
        if(dis <= Time  / 1.0 + wavewidth && dis >= Time / 1.0){
    		return true;
        }
    }
    return false;
}

// Function 227
float getWaveHeight(float d) {
    float tmp = (3.*d+1./SQRT2);
    float zg = 2.*tmp*exp(-1.*(tmp*tmp));
    float zc = 0.045*(exp(-2.*d)*cos(24.*PI*d)-1.-d*(exp(-2.)*cos(24.*PI)-1.));
	if (d < -0.) 
        zc = 0.0239; //yay, magical height correction :D
    return zg + zc ;
}

// Function 228
vec3 mirrorBaryYZ(vec3 o,vec2 a,vec2 b,vec2 c){ //return o, mirrored on ab (triangle abc)
 ;return o.yxz
 ;//vec2 ass=gLLxX(o,o+b-a,a+(b-a)*.5,c)//hard and dumb carthesian within bary
 ;//return o+(ass-o)*2.//hard and dumb carthesian within bary
 ;}

// Function 229
float waterWaves(vec2 q, vec3 e){
	float s = texture(iChannel0, q).z;
	
	vec2 tmp0 = texture(iChannel0, q).rg;
	vec2 tmp1 = texture(iChannel0, q + e.xz).rg;
	vec2 tmp2 = texture(iChannel0, q - e.xz).rg;
	vec2 tmp3 = texture(iChannel0, q + e.zy).rg;
	vec2 tmp4 = texture(iChannel0, q - e.zy).rg;
	
	float p0 = mix(tmp0.g, tmp0.r, s);
	float p1 = mix(tmp1.r, tmp1.g, s);
	float p2 = mix(tmp2.r, tmp2.g, s);
	float p3 = mix(tmp3.r, tmp3.g, s);
	float p4 = mix(tmp4.r, tmp4.g, s);	
	
	return -p0 + (p1 + p2 + p3 + p4) * 0.49999;
}

// Function 230
float wave0(float x){return sin(pi2*x);}

// Function 231
vec3 bary(vec2 a, vec2 b, vec2 c, vec2 p) {
    vec2 v0 = b-a, v1 = c-a, v2 = p-a;
    float d00 = dot(v0,v0);
    float d01 = dot(v0,v1);
    float d11 = dot(v1,v1);
    float d20 = dot(v2,v0);
    float d21 = dot(v2,v1);
    float d = 1./(d00*d11-d01*d01);
    float v = (d11*d20-d01*d21)*d;
    float w = (d00*d21-d01*d20)*d;
    float u = 1.-v-w;
    return vec3(u,v,w);
}

// Function 232
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

// Function 233
float wave7(float x) {return 1.0 - 2.0*fract(x);}

// Function 234
vec2 waveform(float theta) {
    // Controls the minimum amount of the waveform to show 
    const float MIN_ACTIVE = 0.3;
    // Controls the width of the gradient from the active to the passive color
    const float TRANSITION_WIDTH = 0.1;

    theta = abs(theta);

    float level = max(iAudio, MIN_ACTIVE); 
    float color = smoothstep(level, level - TRANSITION_WIDTH, 1.0 - theta);
    level = pow(level, 1.6);
    float highlight = smoothstep(level, level - TRANSITION_WIDTH * 4.0, 1.0 - theta);
    return vec2(color, highlight);
}

// Function 235
float square_wave(float t, float f) {
    return mod( t * f, 1. ) >= .5 ? .0 : 1.;
}

// Function 236
vec2 synthWave(float t){
 t=mod(t,32.);
 return vec2(lead(t,.0));
}

// Function 237
float lowAverage()
{
    const int iters = 32;
    float product = 1.0;
    float sum = 0.0;
    
    float smallest = 0.0;
    
    for(int i = 0; i < iters; i++)
    {
        float sound = texture(iChannel0, vec2(float(i)/float(iters), 0.5)).r;
        smallest = 
        
        product *= sound;
        sum += sound;
    }
    return max(sum/float(iters), pow(product, 1.0/float(iters)));
}

// Function 238
int wave(vec2 uv, float offset, float waveLength, float a, float speed, int layer) {
    if (abs(a * sin((offset + uv.x + iTime * speed) * 1./waveLength) - uv.y) <= WAVE_WIDTH/3. + WAVE_WIDTH * pow(uv.x, 2.)) {
    	return layer;
    }
    return 0;
}

// Function 239
float SinWave( float f, float x )
{
    return sin( f * x * 2.0 * 3.14 );
}

// Function 240
vec4 bar(vec4 color, vec4 background, vec2 position, vec2 diemensions, vec2 uv)
{
    return rectangle(color, background, vec4(position.x, position.y+diemensions.y/2.0, diemensions.x/2.0, diemensions.y/2.0), uv); //Just transform rectangle a little
}

// Function 241
float sdWaveSphere(vec3 p,float radius,int waves,float waveSize)
{
    //bounding Sphere
    float d=length(p)-radius*2.2;
    if(d>0.)return.2;
    // deformation of radius
    d=waveSize*(radius*radius-(p.y*p.y));
    radius+=d*cos(atan(p.x,p.z)*float(waves));
    return.5*(length(p)-radius);
}

// Function 242
float waveFunction(float x)
{
    return sin(x-cos(x)/pi);
}

// Function 243
float smoothWave(int x, int smoothness)
{
    float f = 0.0;
    int accumulated = 0;
    for(int i = 0; i <= smoothness; ++i)
    {
        if(x + i > int(FREQ) || x + i < 0) continue;
        f += wave(x + i);
        ++accumulated;
    }
    return f / float(accumulated);
}

// Function 244
vec3 wavelength_to_srgbl (float l_nm ) {
    if (l_nm<370.||l_nm>780.) return vec3(0.);
    vec4 l = vec4(1.065, 1.014, 1.839, 0.366);
    vec4 c = vec4(593.,556.3,449.8, 446.);
    vec4 s = vec4(.056,.075,.051, .043);
    if (l_nm<446.) s.a = 0.05; // fix creep from violet back to blue
    vec4 v = (log(l_nm)-log(c))/s;
    vec4 xyzx = l*exp(-.5*v*v);
    vec3 xyz = xyzx.xyz+vec3(1,0,0)*xyzx.a;
    const mat3 xyz_to_rgb = 
      mat3(3.240,-.969,.056, -1.537,1.876,-.204, -0.499,0.042,1.057);
    vec3 rgb = xyz_to_rgb*xyz;
    return rgb;
}

// Function 245
float wave( vec2 uv, float time){
	
	float t = (time/16.0-uv.x) * 8.0;

	float x=sin(t);
	
	for (float i = 1.0; i <= HARMONICS; i+=1.0) {
		
		float h = i * 2.0 + 1.0;
		float wave = sin(t*h)/pow(h,2.0);
		
		if (mod(i,2.0) == 0.0) x += wave;
		else x -= wave;
		
	}

	x = x/2.0;
	
	float y = uv.y*2.0-1.0;	
		
	return (x < y) ? 1.0 : 0.0;
}

// Function 246
float WaveRay (vec3 ro, vec3 rd)
{
  vec3 p;
  float dHit, h, s, sLo, sHi;
  dHit = dstFar;
  if (rd.y < 0.) {
    s = 0.;
    sLo = 0.;
    for (int j = 0; j < 100; j ++) {
      p = ro + s * rd;
      h = p.y - WaveHt (p);
      if (h < 0.) break;
      sLo = s;
      s += max (0.3, h) + 0.005 * s;
      if (s > dstFar) break;
    }
    if (h < 0.) {
      sHi = s;
      for (int j = 0; j < 5; j ++) {
	s = 0.5 * (sLo + sHi);
	p = ro + s * rd;
	h = step (0., p.y - WaveHt (p));
	sLo += h * (s - sLo);
	sHi += (1. - h) * (s - sHi);
      }
      dHit = sHi;
    }
  }
  return dHit;
}

// Function 247
vec2 wave(in float t, in float a, in float w, in float p) {
  float x = t;
  float y = a*sin(t*w + p);
  return vec2(x, y);
}

// Function 248
float wave_bass(float t, float f0, float p0)
{
    float op4 = tsaw(p0 + t * f0 * 10.000, 0.01) * (exp(-t * 10.0) + 0.01);
    float op3 = tsaw(p0 + t * f0 * 1.0012 + op4 * 0.17, 0.2);
    float op2 = tsaw(p0 + t * f0 * 0.5008 + op4 * 0.09, 0.2);
    float op1 = sine(p0 + t * f0 * 1.0000 + op4 * 0.16);

    op1 *= env_ad(t, 30.0, 3.0);
    op2 *= env_ad(t, 10.0, 5.0);
    op3 *= env_ad(t, 10.0, 5.0);

    return op1 * 0.92 + op2 * 0.54 + op3 * 0.42;
}

// Function 249
float wave_lead(float t, float f0, float p0)
{
    float op4 = tsaw(p0 + t * f0 * 10.000, 0.3) * (exp(-t * 10.0) + 0.01);
    float op3 = tsaw(p0 + t * f0 * 1.0012 + op4 * 0.22, 0.01);
    float op2 = tsaw(p0 + t * f0 * 0.5008 + op4 * 0.22, 0.02);
    float op1 = tsaw(p0 + t * f0 * 1.0000 + op4 * 0.22, 0.03);

    op1 *= env_ad(t, 4.0, 0.2);
    op2 *= env_ad(t, 6.0, 0.5);
    op3 *= env_ad(t, 6.0, 1.0);

    return op1 * 0.75 + op2 * 0.4 + op3 * 0.26;
}

// Function 250
vec3 plotSinWave(vec2 currentUv, float freq, float amp, vec3 color, vec3 bgc)
{
    float dx = lineWidth / iResolution.x;
    float dy = lineWidth / iResolution.y;// + sin(dx * freq) * amp;
    
    float sy = sin(currentUv.x * freq + iTime) * amp;
    float dsy = cos(currentUv.x * freq + iTime) * amp * freq;

    float alpha = smoothstep(0.0, dy, (abs(currentUv.y - sy))/sqrt(1.0+dsy*dsy));
    
    return mix(color, bgc, alpha);
}

// Function 251
vec3 bars(in float x) {
    vec3 barColor = vec3(pow(max(sin(radians(x*360. * 10.)) + sin(radians(x*360. + sin(iTime*0.7) * 160. + 160.)) ,0.0), 1./5.));
    barColor.rg -= vec2(0.05);
    barColor *= barColor * barColor * barColor;
    barColor /= 5.;
    return barColor;
}

// Function 252
float sineWave(float time, float freq, float amp) {
    return sin(TWO_PI*time*freq)*amp;
}

// Function 253
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

// Function 254
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

// Function 255
float wavesB(in float x, in float y)
{
    if(x < 32.) // ARR64 would be a really long line.
    {
        return ARR8(y,
		ARR32(x,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,3.,3.,3.,3.,3.,3.,3.,3.,3.,5.,5.,2.,0.,0.,0.,0.,0.),
        ARR32(x,0.,0.,0.,0.,0.,0.,0.,2.,2.,2.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,0.,0.,0.,0.,0.,5.,5.,2.,2.,2.,0.),
        ARR32(x,0.,0.,0.,2.,2.,2.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.,4.,4.,2.),
        ARR32(x,2.,2.,3.,3.,3.,3.,3.,3.,3.,3.,3.,2.,2.,3.,3.,3.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,4.),
        ARR32(x,3.,3.,3.,3.,3.,3.,2.,2.,2.,2.,3.,3.,3.,2.,0.,0.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.),
        ARR32(x,3.,3.,3.,2.,2.,2.,3.,3.,3.,3.,0.,0.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.),
		2.,
        2.);
	}
    else
    {
        x -= 32.;
        return ARR8(y,
        ARR32(x,0.,0.,2.,2.,2.,3.,3.,3.,3.,3.,3.,3.,5.,5.,5.,5.,5.,5.,5.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.),
        ARR32(x,0.,0.,0.,0.,0.,0.,0.,2.,2.,3.,3.,5.,5.,5.,5.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,2.,2.,2.,2.),
        ARR32(x,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,2.,3.,3.,3.,3.,2.,3.,2.,0.,0.,0.,0.),
        ARR32(x,4.,4.,4.,4.,4.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,3.,3.,3.,3.,3.,3.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.),
        ARR32(x,0.,0.,0.,4.,4.,4.,4.,4.,5.,5.,5.,5.,5.,5.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.,3.,3.,3.,3.),
        ARR32(x,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,5.,5.,5.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.,3.),
        ARR32(x,2.,2.,2.,2.,2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,3.,3.,3.,3.,3.,3.,3.,3.,3.,2.),
        2.);
    }
}

// Function 256
float triangleWave(float value)
{
	float hval = value*0.5;
	return 2.0*abs(2.0*(hval-floor(hval+0.5)))-1.0;
}

// Function 257
vec3 visualize_dist(vec2 uv, float dist)
{
	const float scale = 0.3;
	vec3 col = vec3(dist, 0.0, -dist) * scale;

	float line = float(abs(dist) < 0.01);

	return col + vec3(0.0, line * 0.5, 0.0);
}

// Function 258
vec4 WaveNfH (vec2 p)
{
  vec3 v;
  vec2 e;
  e = vec2 (0.01, 0.);
  p *= 2.;
  for (int j = VAR_ZERO; j < 3; j ++) v[j] = WaveHt (p + ((j == 0) ? e.yy : ((j == 1) ? e.xy : e.yx)));
  return vec4 (normalize (vec3 (-0.2 * (v.x - v.yz), e.x)), v.x);
}

// Function 259
float solve_black_body_fraction_below_wavelength(float wavelength, float temperature){ 
	const float iterations = 2.;
	const float h = PLANCK_CONSTANT;
	const float k = BOLTZMANN_CONSTANT;
	const float c = SPEED_OF_LIGHT;

	float L = wavelength;
	float T = temperature;

	float C2 = h*c/k;
	float z = C2 / (L*T);
	
	return 15.*(z*z*z + 3.*z*z + 6.*z + 6.) * exp(-z)/(PI*PI*PI*PI);
}

// Function 260
float wave(vec2 p,float x){
	return sin(100.0*p.x-x-iDate.w*20.0)*0.5+0.5;
}

// Function 261
vec3 WavelengthToXYZLinear( float fWavelength )
{
    float fPos = ( fWavelength - standardObserver1931_w_min ) / (standardObserver1931_w_max - standardObserver1931_w_min);
    float fIndex = fPos * float(standardObserver1931_length);
    float fFloorIndex = floor(fIndex);
    float fBlend = clamp( fIndex - fFloorIndex, 0.0, 1.0 );
    int iIndex0 = int(fFloorIndex);
    int iIndex1 = iIndex0 + 1;
    iIndex1 = min( iIndex1, standardObserver1931_length - 1);

    return mix( standardObserver1931[iIndex0], standardObserver1931[iIndex1], fBlend );
}

// Function 262
float waveform(float x)
{
    float prebeat = -sinc((x - 0.37) * 40.0) * 0.6 * triIsolate((x - 0.4) * 1.0);
    float mainbeat = (sinc((x - 0.5) * 60.0)) * 1.2 * triIsolate((x - 0.5) * 0.7) * 1.5;
    float postbeat = sinc((x - 0.91) * 15.0) * 0.85;
    return (prebeat + mainbeat + postbeat) * triIsolate((x - 0.625) * 0.8);
}

// Function 263
float squarewave(float x) {
    return floor(fract(x)*2.);
}

// Function 264
vec3 BaryTri(float pA, float pB, float pC, float pCol1, vec3 col)
{
	vec3 bar = barycentric(unpackCoord(pA), unpackCoord(pB), unpackCoord(pC), uv);
    return mix(col, unpackParams(pCol1), 60.0*vec3(inRangeAll(bar))/256.0);
}

// Function 265
float dsawtoothwave(float n, float x, float l, float phase){
    n++;
    return amp*2.0/l*cos(n*pi*x/l+phase);
}

// Function 266
vec2 qiwave(vec2 x, float h)
{
    float a = -5.0 + 16.0 * h;
    float b =  8.0 - 32.0 * h;
    float c = -4.0 + 16.0 * h;
    
    return x + (x*x) * (a + x * (b + c * x));
}

// Function 267
void bar(float pos, float r, float g, float b)
{
	 if ((position.y <= pos + barsize) && (position.y >= pos - barsize))
		color = mixcol(1.0 - abs(pos - position.y) / barsize, r , g, b);
}

// Function 268
float swave(float x, float a){
    return (sin(x*pi/3.-pi/2.)/sqrt(a*a+sin(x*pi/3.-pi/2.)*sin(x*pi/3.-pi/2.))+1./sqrt(a*a+1.))*0.5;
}

// Function 269
void barWave(vec2 coord, vec4 pos, vec4 color, float angle, inout vec4 result) {
    pos = vec4(min(pos.z,pos.x),min(pos.w,pos.y),max(pos.z,pos.x),max(pos.w,pos.y));
    vec2 rel = coord - pos.xy;
    vec2 scal = vec2(pos.z-pos.x,pos.w-pos.y);
    rel = rotateXYPivot(rel, scal/2.0, angle);
    vec2 erl = rel/scal;
    vec4 waveform = texture(iChannel0, vec2(erl.x, 0.25));
    float wv = (waveform.r+waveform.g+waveform.b+waveform.a)/4.0;
    if (rel.x > 0. && rel.x < scal.x && rel.y > 0. && erl.y < wv) {
        blend(result, vec4(color.r,color.g,color.b,color.a*clamp(wv-erl.y,0.0,1.0)));
    }
}

// Function 270
vec4 triangleWave(vec4 col)
{
	return 
		vec4(
			triangleWave(col.x),
			triangleWave(col.y),
			triangleWave(col.z),
			triangleWave(col.w));
}

// Function 271
float mapWaveCheap(vec3 p)
{
    float t = getWaveTime();
   	waveFrontDistort(p, 1.0);
   	vec2 ph = getPhase(p,t) ;
    // skip sea height field 
   	float sea_h = 0.0;
    waveMoveForward( p, t, 1.0 );
    vec3 df = dfWaveAnim ( p,sea_h, ph);
    return df.z;
}

// Function 272
float WaveOutRay (vec3 ro, vec3 rd)
{
  vec3 p;
  float dHit, h, s, sLo, sHi;
  s = 0.;
  sLo = 0.;
  dHit = dstFar;
  ro.y *= -1.;
  rd.y *= -1.;
  for (int j = 0; j < 150; j ++) {
    p = ro + s * rd;
    h = p.y + WaveHt (p);
    if (h < 0.) break;
    sLo = s;
    s += max (0.2, h) + 0.005 * s;
    if (s > dstFar) break;
  }
  if (h < 0.) {
    sHi = s;
    for (int j = 0; j < 7; j ++) {
      s = 0.5 * (sLo + sHi);
      p = ro + s * rd;
      h = step (0., p.y + WaveHt (p));
      sLo += h * (s - sLo);
      sHi += (1. - h) * (s - sHi);
    }
    dHit = sHi;
  }
  return dHit;
}

// Function 273
float verticalBar(float pos, float uvY, float offset) {
    float edge0 = (pos - range);
    float edge1 = (pos + range);

    float x = smoothstep(edge0, pos, uvY) * offset;
    x -= smoothstep(pos, edge1, uvY) * offset;
    return x;
}

// Function 274
vec3 r36bary(vec3 o){return mirrorBaryZ(o).yxz;}

// Function 275
int wave(int t)
{   
    return(((((t&t&t)>>7)*255)^t)-64);
}

// Function 276
vec3 raymarch_visualize(vec2 ro, vec2 rd, vec2 uv)
{
	//ro - ray origin
	//rd - ray direction

	vec3 ret_col = vec3(0.0);

	const float epsilon = 0.001;
	float t = 0.0;
	for(int i = 0; i < 10; ++i)
	{
		vec2 coords = ro + rd * t;
		float dist = field(coords);
		ret_col += visualize_region(uv, dist, coords);
		t += dist;
		if(abs(dist) < epsilon)
			break;
	}

	//return t;
	return ret_col;
}

// Function 277
vec3 waveToXyz(float wave)
{
	float x1 = (wave-442.0)*((wave<442.0)?0.0624:0.0374);
	float x2 = (wave-599.8)*((wave<599.8)?0.0264:0.0323);
	float x3 = (wave-501.1)*((wave<501.1)?0.0490:0.0382);

	float y1 = (wave-568.8)*((wave<568.8)?0.0213:0.0247);
	float y2 = (wave-530.9)*((wave<530.9)?0.0613:0.0322);

	float z1 = (wave-437.0)*((wave<437.0)?0.0845:0.0278);
	float z2 = (wave-459.0)*((wave<459.0)?0.0385:0.0725);

	return vec3(
		0.362*exp(-0.5*x1*x1) + 1.056*exp(-0.5*x2*x2) - 0.065*exp(-0.5*x3*x3),
		0.821*exp(-0.5*y1*y1) + 0.286*exp(-0.5*y2*y2),
		1.217*exp(-0.5*z1*z1) + 0.681*exp(-0.5*z2*z2)
	);
}

// Function 278
float wave2(float x){return abs(sin(x*pi2));}

// Function 279
float dwave(float n, float x, float l, float phase){
    switch(int(mod(iTime, duration*3.0) / duration))
    {
        case 0: return dsquarewave(n,x,l,phase);
        case 1: return dsawtoothwave(n,x,l,phase);
        case 2: return dtrianglewave(n,x,l,phase);
    }
}

// Function 280
void wave_1s(vec3 pos, float t, out vec2 im) {
  float amp = R_A_PI_3 * exp(-pos.x / A0);
  float phase = E_1s * t;

  im_exp(amp, phase, im);
}

// Function 281
float wave_pattern(float t, vec2 uv){
    t-=pattern_start_time+audio_offset;
    
    if(t<pattern2_start_time){
        return wave_pattern1(t,uv);
    }
    else{
        return wave_pattern2(t-pattern2_start_time,uv);
    }
}

// Function 282
vec4 wave(float x) {
    return vec4(
        texture(iChannel0,vec2(x,0.0)).r,
        texture(iChannel0,vec2(x,0.25)).r,
        texture(iChannel0,vec2(x,0.5)).r,
        texture(iChannel0,vec2(x,0.75)).r
    );
}

// Function 283
float wave(vec2 coord)
{
    float interval = iResolution.x * 0.04;
    vec2 p = coord / interval;

    float py2t = 0.112 * sin(iTime * 0.378);
    float phase1 = dot(p, vec2(0.00, 1.00)) + iTime * 1.338;
    float phase2 = dot(p, vec2(0.09, py2t)) + iTime * 0.566;
    float phase3 = dot(p, vec2(0.08, 0.11)) + iTime * 0.666;

    float pt = phase1 + sin(phase2) * 3.0;
    pt = abs(fract(pt) - 0.5) * interval * 0.5;

    float lw = 2.3 + sin(phase3) * 1.9;
    return saturate(lw - pt);
}

// Function 284
float sinwave(float note, float time)
{
    return 0.5 + 0.5 * sin(tau * time * notefreq(note));
}

// Function 285
void bwaveParams ( float t, float life, out vec2 o, out vec3 kSpike, out float kRot )
{
 	t *= 5.0 * life; 
    o = vec2(0.0); // rotation pivot
    if ( t < 1.0 ) // forming
    {
        kSpike = vec3(0.2*t, 0.0,1.0);
        kRot = 0.0;
    }
    else if ( t < 2.0 ) // breaking 
    {
        t = t - 1.0;
        kSpike = vec3(0.2, 0.8*pow(t,1.5),1.0);
	    kRot = pow(t,0.5);
    }
    else if ( t < 5.0 ) // fallback
    {        
        t = t - 2.0; t /= 3.0;
        o = vec2(-.8,-1.) *t;
        float t2 = 1.0- t;
        kSpike = vec3(0.2*t2, 0.8*t2, t2) ;
	    kRot = 0.9 + 0.1*(1.0-t);
        if ( t > 0.5 ) kRot = 0.95 * (1.0- ( t -0.5) *2.0); 
    }
}

// Function 286
void wave_2p(vec3 pos, float t, out vec2 im) {
  float amp = 1.0 / (sqrt(2.0) * 4.0 * A0);
  amp *= R_A_PI_3 * exp(-pos.x / (2.0 * A0));
  amp *= pos.x * cos(pos.y);

  float phase = E_2p * t;

  im_exp(amp, phase, im);
}

// Function 287
vec4 copperbar(in vec2 uv,
               in float base, in float offset, in float frequency, in float amplitude,
               in vec4 innerColor, in vec4 outerColor) {
    float alpha = (iTime + offset) * frequency; // Offset and scale current time.
    float position = base + (sin(alpha) * amplitude); // Get the copperbar middle position.
    float ratio = abs(uv.y - position) / HEIGHT; // Normalized (to height) distance.
    if (ratio > 1.0) { // Pixel is beyond copperbar limit, set to black.
        return BLACK;
    }
    return mix(innerColor, outerColor, ratio); // Mix to generate a gradient.
}

// Function 288
float getWave(in float x){
    float freq = x;
        
    freq=pow(10.0, freq*2.0-1.0)/10.0; //Logarithmic scale
    
    return texture(iChannel1, vec2(freq, 1.0)).x;   
}

// Function 289
vec2 synthWave(float t)
{
    bool do_reverb = mod(t, 8.0) > 4.0;
    t = mod(t, 2.0);
    
    float f0 = 880.0;

    vec2 w = vec2(sine(t * f0) * exp(-t * 2.5));
    
    if (do_reverb)
    {
        vec2 r = lpnoise(t,  100.0)
               + lpnoise(t,  550.0) * 0.2
               + lpnoise(t, 1050.0) * 0.1 * exp(-t * 5.0);

    	w += sine(t * f0 + r * 0.1) * exp(-t * 2.0);
    	w -= sine(t * f0          ) * exp(-t * 2.0);
    }

    w *= 1.0 - exp(-t * 800.0);

    return w;
}

// Function 290
float WaveletNoise(vec2 p, float z, float k) {
    // https://www.shadertoy.com/view/wsBfzK
    float d=0.,s=1.,m=0., a;
    for(float i=0.; i<4.; i++) {
        vec2 q = p*s, g=fract(floor(q)*vec2(123.34,233.53));
    	g += dot(g, g+23.234);
		a = fract(g.x*g.y)*1e3;// +z*(mod(g.x+g.y, 2.)-1.); // add vorticity
        q = (fract(q)-.5)*mat2(cos(a),-sin(a),sin(a),cos(a));
        d += sin(q.x*10.+z)*smoothstep(.25, .0, dot(q,q))/s;
        p = p*mat2(.54,-.84, .84, .54)+i;
        m += 1./s;
        s *= k; 
    }
    return d/m;
}

// Function 291
float wave_pattern1(float t, vec2 uv){
    float dis=1e38;
    
    dis=beat_wave(t,0.*beat_time_fac,uv,original_disk_center1,dis);
    dis=beat_wave(t,1.*beat_time_fac,uv,original_disk_center2,dis);
    dis=beat_wave(t,3.*beat_time_fac,uv,original_disk_center3,dis);
    dis=beat_wave(t,5.*beat_time_fac,uv,original_disk_center1,dis);
    dis=beat_wave(t,7.*beat_time_fac,uv,original_disk_center2,dis);
    dis=beat_wave(t,8.*beat_time_fac,uv,original_disk_center3,dis);
    
    return dis;
}

// Function 292
vec3 WaveNf (vec2 p, float d)
{
  vec2 e = vec2 (max (0.01, 0.001 * d * d), 0.);
  return normalize (vec3 (WaveHt (p) - vec2 (WaveHt (p + e.xy), WaveHt (p + e.yx)), e.x).xzy);
}

// Function 293
vec3 rii_rasterBars(in float gtime, in vec2 uv, in vec3 background)
{
  vec3 col = background;

  const float unflatTime = 31.5;

  vec2 pol = toSmith(uv);
  rot(pol, smoothstep(unflatTime, unflatTime + 120.0, gtime)*3.14*sin(gtime));
  //pol.x += 0.75*sin(TIME*1.5);
  pol.y += smoothstep(unflatTime, unflatTime + 60.0, gtime)*0.5*cos(gtime);
  //pol.y += sin(TIME + uv.y + uv.x)*0.5;
  //pol.x *= 0.6 + 0.4*cos(TIME + uv.y - 0.5);
  uv = fromSmith(pol);

  for(int i = 0; i < rii_numBars; ++i)
  {
      float shade = (float(i + rii_numBars)/float(2*rii_numBars));
      vec3 barCol = rii_bars[rii_numBars - i - 1]*shade;
      float a = 0.5*float(i) + 2.0*gtime;
      float y = 0.5*sin(a);
      float f = step(-0.1,uv.y + y)*(1.0 - step(0.1,uv.y + y));
      float h =
          rii_flash(gtime, unflatTime) +
          step(unflatTime, gtime)*shade*0.5*cos(30.0*(uv.y + y))*f;
      col = mix(col, barCol + h, 0.9*f);
  }

  return col;
}

// Function 294
vec2 ProjectCoordsWave(vec2 normCoords)
{
	const float MAX_RADIUS = 1.0;
    float rad = sqrt(dot(normCoords, normCoords));
    if(rad > MAX_RADIUS)
        return normCoords;
    
    const float MIN_DEPTH = 0.4;
    const float WAVE_INV_FREQ = 20.0;
    const float WAVE_VEL = -10.0;
    float z = MIN_DEPTH + 
        (MAX_RADIUS - MIN_DEPTH) 
        * 0.5 * (1.0 + sin(WAVE_INV_FREQ * rad + iTime * WAVE_VEL));
//    if(z > 0.2)
//        return normCoords;
    return normCoords / z;
}

// Function 295
float wave(float n, float x, float l, float phase){
    switch(int(mod(iTime, duration*3.0) / duration))
    {
        case 0: return squarewave(n,x,l,phase);
        case 1: return sawtoothwave(n,x,l,phase);
        case 2: return trianglewave(n,x,l,phase);
    }
}

// Function 296
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

// Function 297
vec4 drawWave(vec4 pixel, vec2 uv, float speed, float height, float vOffset) {
    float c = covered(uv, speed, height, vOffset);
    if (c == 1.) {
		pixel += BASE_COLOR * MULTIPLIER;
        
        if (SHOW_TEXTURES) {
            vec2 nUv = uv;
            nUv.x += iTime * sqrt(speed) * 0.1;
            nUv.y -= heightAt(uv, speed, height, vOffset);
            pixel += .015 * noise(nUv * 20.);
    	}
    } else if (c > 0.) {
		pixel *= vec4(vec3(sqrt(1.-c)) ,1.0);
    }
    
	return pixel;
}

// Function 298
vec3 dfWaveAnim ( vec3 q, float sea_h, vec2 phase )
{
	// height-based wave 
    q.y -= sea_h;
    // breaking wave 
    vec2 o; vec3 kSpike; float kRot;
    float spikePhase = 1.0;
    
    bwaveParams ( phase.x, phase.y, o, kSpike, kRot);    
    vec2 d1 = dfWave(q, o, kRot*spikePhase, kSpike*spikePhase);
    vec2 d2 = dfWave(q, o, 0.0, vec3(kSpike.x*spikePhase,.0,.0));
    float d = mix( max ( d1.x,d2.x), min(d1.x,d2.x), smoothstep(-1.0,1.0, q.x ) ) ;    
    float ws = d1.y;
    // cylinder is for accelerating ray tracing
    q.y *= 2.5;
    float bound = sdCylinder(q.xzy,vec2(1.5,4.0));
    
    return vec3(d, ws, bound);
}

// Function 299
int get_wave_index(float t){
    if(t<2.){
    	return 0;
    }
    else if(t<4.){
		return 1;
    }
    else{
		return 2;
    }
}

// Function 300
void initWaveAtom2D(inout vec4 col, vec2 pos)
{
    pos=(pos-vec2(iResolution.xy)*.5);
    
    pos*=.5; // there must be still some mistake somewhere - this should be 1
    
    pos*=1.2; // lets make it a bit smaller than the exact solution
              // so it radiates part of its energy into the sourrounding
              // before it equilibrates
    
    col.xy=vec2(0);
    // lets add some quantum fluctuation
	//col.xy += .5*getSmoothRand2(pos*.2);
    
    float r=length(pos);
    float phi=atan(pos.y,pos.x);
	
    // analytic solutions below to test the simulation were taken from
    // https://www.researchgate.net/publication/13384275_Analytic_solution_of_a_two-dimensional_hydrogen_atom_I_Nonrelativistic_theory    
    
    // psi(n=2,l=1)
    float beta2 = beta(2.);
    col.xy+=50.*beta2*beta2/sqrt(6.)*r*exp(-beta2*r/2.)*vec2(cos(phi),sin(phi));
        
    // psi(n=3,l=1)
    //float beta3 = beta(3.);
    //col.xy+=50.*beta3*beta3/sqrt(30.)*r*(3.-beta3*r)*exp(-beta3*r/2.)*vec2(cos(phi),sin(phi));
        
    // psi(n=3,l=2)
    //float beta3 = beta(3.);
    //col.xy+=50.*beta3*beta3*beta3/sqrt(120.)*r*r*exp(-beta3*r/2.)*vec2(cos(2.*phi),sin(2.*phi));    
}

// Function 301
vec2 traceHexBaryOutside(vec3 o,vec3 t,vec3 b1,vec3 b2,vec3 b3,vec3 b4,vec3 b5,vec3 b6,vec3 X,vec3 Y
){return b2c(b1,X,Y)+vec2(.1)
 ;return vec2(.1,0);}

// Function 302
float getwaves(vec2 position, float itm){
	float iter = 0.0;
    float phase = 6.0;
    float speed = 2.0;
    float weight = 1.0;
    float w = 0.0;
    float ws = 0.0;
    for(int i=0;i<ITERATIONS_RAYMARCH;i++){
        vec2 p = vec2(sin(iter), cos(iter));
        vec2 res = wavedx(position, p, speed, phase, itm);
        position += normalize(p) * res.y * weight * DRAG_MULT;
        w += res.x * weight;
        iter += 12.0;
        ws += weight;
        weight = mix(weight, 0.0, 0.2);
        phase *= 1.18;
        speed *= 1.07;
    }
    return w / ws;
}

// Function 303
float getWaveTime()
{
   // return mod(iTime + 2.5,WaveTimeMax);
    return iTime;
}

// Function 304
vec2 wavedx(vec2 position, vec2 direction, float speed, float frequency, float timeshift) {
    float x = dot(direction, position) * frequency + timeshift * speed;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return vec2(wave, -dx);
}

// Function 305
float timeToWindWaveAmt(float time){
    
    float wave = sin( time + sin(time*0.8) + sin(time*0.2)*sin(time*2.1) );
    return wave*0.5 + 0.5;
    
}

