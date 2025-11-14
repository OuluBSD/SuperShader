// Reusable Fractals Procedural Functions
// Automatically extracted from procedural-related shaders

// Function 1
vec2 mandelbrot(float time, vec2 coords){
    vec4 sequence = vec4(0.0);
    for(int n = 0; n < int(time * 80.); n++){
        sequence.zw = sequence.xy;
        if(n == 0){
            sequence.xy += coords;
        }
        else {
            sequence.xy = complexSquare(sequence.zw) + coords;
        }
    }
    return sequence.xy;
}

// Function 2
float FractalNoise(in vec2 xy)
{
	float w = 1.5;
	float f = 0.0;
    xy *= .08;

	for (int i = 0; i < 5; i++)
	{
		f += texture(iChannel2, .5+xy * w, -99.0).x / w;
		w += w;
	}
	return f*.8;
}

// Function 3
vec3 julia(vec2 xy, int iterations)
{
    vec2 z = xy;
    vec2 c = 2.0 * iMouse.xy / iResolution.xy - 1.0;
    for(int n = 0; n < iterations; n++)
    {
        z = cpow(vec2(-0.00740,0.00666),z);
        z += c * 0.01;
        if(dot(z, z) > 16.0) return vec3(z, float(n));
    }
    return vec3(z, float(iterations));
}

// Function 4
vec3 FractalSpace(vec3 pos, float time)
{
    pos *= 0.1;
    
    float height = length(pos) * 10.0;
    float s=3.;
	for(int i=0;i<fractalSteps;i++){
		pos.xy=abs(pos).xy-s; 
        pos.xy *= rz2(1.4-0.01*time);
        pos.xz *= rz2(time*0.018); 
        pos.yz *= rz2(time*0.005);
		s=s/1.3;
	}
    
    return pos;
}

// Function 5
vec3 GetPixelFractal(vec2 pos, int iterations, float timePercent)
{
    int glyphLast = GetFocusGlyph(iterations-1);
	ivec2 glyphPosLast = GetFocusPos(iterations-2);
	ivec2 glyphPos =     GetFocusPos(iterations-1);
    
	bool isFocus = true;
    ivec2 focusPos = glyphPos;
    
	vec3 color = InitPixelColor();
	for (int r = 0; r <= recursionCount + 1; ++r)
	{
        color = CombinePixelColor(color, timePercent, iterations, r, pos, glyphPos, glyphPosLast);
        
        //if (r == 1 && glyphPos == GetFocusPos(r-1))
	    //    color.z = 1.0; // debug - show focus
        
        if (r > recursionCount)
			return color;
           
        // update pos
        pos -= vec2(glyphMargin*gsfi);
        pos *= glyphSizeF;

        // get glyph and pos within that glyph
        glyphPosLast = glyphPos;
        glyphPos = ivec2(pos);

        // check pixel
        int glyphValue = GetGlyphPixel(glyphPos, glyphLast);
		if (glyphValue == 0 || pos.x < 0.0 || pos.y < 0.0)
			return color;
        
        // next glyph
		pos -= vec2(floor(pos));
        focusPos = isFocus? GetFocusPos(iterations+r) : ivec2(-10);
        glyphLast = GetGlyph(iterations + r, glyphPos, glyphLast, glyphPosLast, focusPos);
        isFocus = isFocus && (glyphPos == focusPos);
	}
	return color;
}

// Function 6
float fractalnoise(vec2 uv, float mag) {
    float d = valuenoise(uv);
    int i;
    float fac = 1.;
    vec2 disp = vec2(0., 1.);
    for (i=0; i<3; i++) {
        uv += mag * iTime * disp * fac;
        disp = mat2(.866, 0.5, -0.5, .866) * disp; //rotate each moving layer
        fac *= 0.5;
        d += valuenoise(uv/fac)*fac;
    }
    return d;
}

// Function 7
vec3 JULIA_ROTATE(vec3 ro, float t)
{
	return QtnRotate(ro,
                  	  vec4(vec3(0.0f, 1.0f, 0.0f) * sin(t), cos(t)));
}

// Function 8
vec2 derivMandelbrot(float time, vec2 coords){
    return (mandelbrot(time + iTimeDelta, coords) - mandelbrot(time, coords)) / iTimeDelta;
}

// Function 9
vec3 fractalMaterial1(vec2 p) {
	float x = 0.;
    float y = 0.;
    float v = 100000.;
    float j = 100000.;
    vec3 col = vec3(0.);
    for(int i=0; i<100;i++) {
    	float xt = x*x-y*y+p.x;
        y = 2.*x*y+p.y;
        x = xt;
        v = min(v, abs(x*x+y*y));
        j = min(j, abs(x*y));
        if(x*x+y*y >= 8.) {
        	float d = (float(i) - (log(log(sqrt(x*x+y*y))) / log(2.))) / 50.;
            v = (1. - v) / 2.;
            j = (1. - j) / 2.;
            col = vec3(d+j,d,d+v);
            return col;
        }
    }
}

// Function 10
float julianDay2000(in int yr, in int mn, in int day, in int hr, in int m, in int s) {
	int im = (mn-14)/12, 
		ijulian = day - 32075 + 1461*(yr+4800+im)/4 + 367*(mn-2-im*12)/12 - 3*((yr+4900+im)/100)/4;
	float f = float(ijulian)-2451545.;
	return f - 0.5 + float(hr)/24. + float(m)/1440. + float(s)/86400.;
}

// Function 11
float fractal(in vec2 uv) {
	float c = cos(1.0/float(SIDES)*TAU);
	float s = sin(1.0/float(SIDES)*TAU);
	
	mat2 m = mat2(c, s, -s, c);
	vec2 p = vec2(1.0, 0.0), r = p;
	
	for (int i = 0; i < 8; ++i) {
		float dmin = length(uv - r);
		for (int j = 0; j < int(SIDES + 1.); ++j) {
			p = m*p;
			float d = length(uv - p); 
			if (d < dmin) {dmin = d; r = p;}
		}
		uv = 2.0*uv - r;
	}
	
	return ((length(uv-r)*(1.+SIDES/5.))-0.15)/pow(2.0, 7.0);
}

// Function 12
vec3 julia2(vec2 z, vec2 c)
{
    int i = 0;
    vec2 zi = z;
    
    float trap1 = 10e5;
    float trap2 = 10e5;
    
    for(int n=0; n < MAXITER; ++n)
    {
        if(dot(zi,zi) > 4.0)
            break;
        i++;
        zi = cmul(zi,zi) + c;
		
        // Orbit trap
        trap1 = min(trap1, sqrt(zi.x*zi.y));
        trap2 = max(trap2, sqrt(zi.y*zi.y));
    }
    
    return vec3(i,trap1,trap2);
}

// Function 13
float fractal(vec3 p){
  p.xz*=rot(t+p.y*.1565);
  p.yz*=rot(cc(t)*.5+p.y*.25+t);
  
  float k,sc = 1.; //test
  for(float i = 0.; i < 8.; i++){
      p.yz *= rot(p.x*.154+i*i+t);
      p=abs(p)-.65;
      k = max(1., .98/dot(p,p));
      p*=k;
      sc*=k;
  }
  float cucube = sbb(p,vec3(.55))/sc;
  
  return cucube*.56;
}

// Function 14
float Fractal(vec2 fragCoord, float t){
    float size = min(iResolution.x, iResolution.y);
    vec2 o = (iResolution.xy - vec2(size)) / 2.0;
    vec2 p = fragCoord.xy;// (fragCoord.xy - o);
    
    float col = 0.0;
    //if(p.x >= 0.0 && p.y >= 0.0 && p.x < size && p.y < size) {
        float s = exp2(fract(t));
    	uint ix = uint(p.x / s);
    	uint iy = uint(p.y / s);
    	uint res = (ix & iy);
    	col = res == 0U ? 0.0 : 1.0;
    //}
    return col;
}

// Function 15
vec3 Julia3D(vec3 x, int seed, float m_N){
    float m_AbsN = abs(m_N);
    float m_Cn = (1. / m_N - 1.) / 2.;
	float z = x.z / m_AbsN;
	float r = 1./pow(dot(x.xy,x.xy) + z*z, -m_Cn);
	float tmp = r * length(x.xy);
	float ang = (atan(x.x,x.y) + pi*2. * float(IHash(seed^0x6acb43d3 )%int(m_AbsN) ) ) / m_N;
	return vec3(tmp * cos(ang),tmp * sin(ang),r * z);
}

// Function 16
float fractalNoise(vec2 vl) {
    float persistance = 2.0;
    float amplitude = 1.2;
    float rez = 0.0;
    vec2 p = vl;
    
    for (float i = 0.0; i < OCTAVES; i++) {
        rez += amplitude * valueNoiseSimple(p);
        amplitude /= persistance;
        p *= persistance; // Actually the size of the grid and noise frequency
        //frequency *= persistance;
    }
    return rez;
}

// Function 17
float fractal(vec2 x)
{
    float f = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 5; i++)
	{
		f += noise(x) * amplitude;
		x *= 2.0;
        amplitude /= 2.0;
	}
    
	return f;
}

// Function 18
vec3 julia(vec2 z, vec2 c)
{
    int i = 0;
    vec2 zi = z;
    
    float trap1 = 10e5;
    float trap2 = 10e5;
    
    for(int n=0; n < MAXITER; ++n)
    {
        if(dot(zi,zi) > 4.0)
            break;
        i++;
        zi = cmul(zi,zi) + c;
		
        // Orbit trap
        trap1 = min(trap1, sqrt(zi.x*zi.y));
        trap2 = min(trap2, sqrt(zi.y*zi.y));
    }
    
    return vec3(i,trap1,trap2);
}

// Function 19
float fractal_noise(vec3 p)
{
    float f = 0.0;
    // add animation
    p = p - vec3(1.0, 1.0, 0.0) * iTime * 0.1;
    p = p * 3.0;
    f += 0.50000 * noise(p); p = 2.0 * p;
	f += 0.25000 * noise(p); p = 2.0 * p;
	f += 0.12500 * noise(p); p = 2.0 * p;
	f += 0.06250 * noise(p); p = 2.0 * p;
    f += 0.03125 * noise(p);
    
    return f;
}

// Function 20
float JuliaDE(vec2 srcCoord,
              out vec2 iterZ,
              out float iterN)
{
	// Convert given coordinates (expected as integers) to the floating range 0...1.0,
    // then scale into the Julia domain (-2.0.xx, 2.0.xx)
    vec2 z = (srcCoord / (ZOOM * zoomAnim())) + OFFS;
    vec2 c = z;
        
    // Iterations
    float maxDist = 200.0;
    vec2 dz = vec2(1, 0);
    float iter = 0.0;
    while (iter < MAX_JULIA_ITER &&
           length(z) < maxDist)
    {
        dz = 2.0 * PolyMul(z, dz);
    	z = PolyMul(z, z) + c;
        iter += 1.0;
    }
    
    // Export iteration count
    iterN = iter;
    
    // Export the iterated coordinate [z]
    iterZ = z;
    
    // Final distance
    float r = length(z);
    float dr = length(dz);
    return 0.5 * r * log(r) / dr;
}

// Function 21
int fractal(complex c, complex z) {
  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {

    // z <- z^2 + c
    float real = z.real * z.real - z.imaginary * z.imaginary + c.real;
    float imaginary = 2.0 * z.real * z.imaginary + c.imaginary;

    z.real = real;
    z.imaginary = imaginary;

    if (z.real * z.real + z.imaginary * z.imaginary > 4.0) {
      return iteration;
    }
  }

  return MAX_ITERATIONS;
}

// Function 22
float calcfractal(vec3 coord) {
    vec3 orbit = coord;
    float dz = 1.0;
    
    for (int i=0; i<iterations; i++) {
        
        float r = length(orbit);
    	float o = acos(orbit.z/r);
    	float p = atan(orbit.y/orbit.x);
        
        dz = 8.0*r*r*r*r*r*r*r*dz + 1.0;
        
        r = r*r*r*r*r*r*r*r;
        o = 8.0*o;
        p = 8.0*p;
        
        orbit = vec3( r*sin(o)*cos(p), r*sin(o)*sin(p), r*cos(o) ) + coord;
        
        if (dot(orbit,orbit) > 4.0) break;
    }
    float z = length(orbit);
    return 0.5*z*log(z)/dz;
}

// Function 23
vec3 fractal(vec2 p){
    float t = iTime/4.*float(1+view);
    vec2 p2 = vec2(0);
    int i;
    for(i=0;i<int(iTime/4.);i++){
    	p2 = timesc(sinc(p),cosc(p2)); // Xn+1 = cos(Xn)*sin(c); X0 = 0
        if(view==1&&dot(p2,p2)>100.)break;
    }
    vec3 col = vec3(0);
    if(view == 0){
    	p = timesc(sinc(p),cosc(p2));
   		p2 = mix(p2,p,clamp(fract(iTime/4.),0.,.5)*2.);
    	float ang = arg(p2);
    	float len = length(p2);
    	col = hue(ang/(2.*pi))*(1./(1.+len));
    }else{
        col = hue(fract(float(i)/20.));
        if(i==int(iTime/4.)) col = vec3(0);
    }
    return col;
}

// Function 24
v0 fractalFungus(v1 u
){return fractalFungus(m2( .28862355854826727,.6997227302779844 , .6535170557707412
                         , .06997493955670424,.6653237235314099 ,-.7432683571499161
                         ,-.9548821651308448 ,.26025457467376617, .14306504491456504)*v2(u,0));}

// Function 25
vec4 mandelbrot( in vec2 fragCoord )
{
    vec2 p = -1.0 + 2.0 * fragCoord.xy / iResolution.xy;
    p.x *= iResolution.x/iResolution.y;

    // animation	
	//float tz = 0.5 - 0.5*cos(0.225*iTime);
    //float zoo = pow( 0.5, 13.0*tz );
    
	//float tz = 0.5 - 0.5*cos(1.225*iTime);
    //float zoo = pow( 0.5, 10. + 2.0*tz );
    
    float tz = 0.5 - 0.5*cos(0.225*iTime);
    float tz2 = 0.5 + 0.5*sin(0.225*iTime*.5);
    float zoo = pow( 0.5, 14.*tz-.5 );
    float a = iTime/6.;
    
	vec2 c = mix( vec2( -1.2577776425405087 , -0.35897390816984004 ),
                  vec2( -0.3083400668881918 , 0.6384704937518136 ),
                  //smoooth((tz-1.)/2.) )
                  //smoooth(smoooth(smoooth(smoooth((tz-1.)/2.)))) )
                  smoooth(smoooth(smoooth(smoooth( tz2 )))) )
        + p*zoo*mat2(cos(a),sin(a),-sin(a),cos(a));

    // iterate
    vec2 z  = vec2(0.0);
    float m2 = 0.0;
    vec2 dz = vec2(0.0);
    int iter = 0;
    for( int i=0; i<ITERATIONS; i++ )
    {
        if( m2>BAILOUT2 ) continue;

		// Z' -> 2Â·ZÂ·Z' + 1
        dz = 2.0*vec2(z.x*dz.x-z.y*dz.y, z.x*dz.y + z.y*dz.x) + vec2(1.0,0.0);
			
        // Z -> ZÂ² + c			
        z = vec2( z.x*z.x - z.y*z.y, 2.0*z.x*z.y ) + c;
			
        m2 = dot(z,z);
        iter = i;
    }

    // distance	
	// d(c) = |Z|Â·log|Z|/|Z'|
	float d = 0.5*sqrt(dot(z,z)/dot(dz,dz))*log(dot(z,z));
	float potential = float(iter) - log( log(length(z)) / log(BAILOUT) )/log(2.);

	
    // do some soft coloring based on distance
	//d = clamp( 2.0*d/zoo, 0.0, 1.0 );
	//d = pow( d, 0.25 );
    
    bool inside = iter == ITERATIONS-1;
    
    return vec4( inside ? 0. : (d/zoo) , potential , iter , inside );
    
}

// Function 26
float julia(in vec2 c, in vec2 z, in vec2 target) {
    float x;
    float d = 1e20;
    for (int j = 0; j < ITERS; j++) {
        if (z.x * z.x + z.y * z.y > 4.0) {
            return d;
        }
        
        x = z.x * z.x - z.y * z.y + c.x;
        z.y = 2.0 * z.x * z.y + c.y;
        z.x = x;
        
        d = min(d, length(z - target));
    }
    return d;
}

// Function 27
int mandelbrot_iters(vec2 p)
{
	vec2 z = p;
	int it = 0;
	for (int i = 1; i < maxiter; ++i) {
		z = cmul(z, z) + p;
		if (length(z) > 2.) {
			it = i;
			break;
		}
	}
	return it;
}

// Function 28
int julia(vec2 z, vec2 c)
{
    int i = 0;
    vec2 zi = z;
    
    for(int n=0; n < MAXITER; ++n)
    {
        if(dot(zi,zi) > 4.0)
            break;
        i++;
        zi = cmul(zi,zi) + c;
        
    }
    
    return i;
}

// Function 29
vec3 fractal(vec2 p) {
    vec2 pos=p;
    float d, ml=100.;
    vec2 mc=vec2(100.);
    p=abs(fract(p*.1)-.5);
    vec2 c=p;
    for(int i=0;i<8;i++) {
        d=dot(p,p);
        p=abs(p+1.)-abs(p-1.)-p;
    	p=p*-1.5/clamp(d,.5,1.)-c;
        mc=min(mc,abs(p));
        if (i>2) ml=min(ml*(1.+float(i)*.1),abs(p.y-.5));
    }
    mc=max(vec2(0.),1.-mc);
    mc=normalize(mc)*.8;
    ml=pow(max(0.,1.-ml),6.);
    return vec3(mc,d*.4)*ml*(step(.7,fract(d*.1+iTime*.5+pos.x*.2)))-ml*.1;
}

// Function 30
vec3 Mandelbrot(vec2 pix)
{
    vec2 uv = (pix / iResolution.xy) * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;
    
    // Position the camera  
    float rad = -2.0;
    uv = mat2(cos(rad), sin(rad), -sin(rad), cos(rad)) * uv;
    uv += vec2(4.91, 22.91);
    uv *= 0.028;
    
    vec2 c = uv;
	vec3 col = vec3(0.05);    
    vec2 z = vec2(0.0);
    float curIt = 0.0;
    
    // Calculate the actual Mandelbrot
    for(int i=0; i<IT; i++)
    {       
    	z = vec2( (z.x*z.x)-(z.y*z.y), 2.0*z.x*z.y) + c;
		if(dot(z,z) > 4.0)
		{
            curIt = float(i);
            break;
		}
    }
    
    col = vec3(1.0);
    col -= min(1.0, curIt / float(20)) * vec3(0.1,0.4,0.5);
    col -= min(1.0, curIt / float(40))* vec3(0.2,0.2,0.2);
    col -= min(1.0, curIt / float(100)) * vec3(0.0,0.5,0.4);
    return col;
}

// Function 31
vec3 fractalMaterial2(vec2 p) {
	vec3 col = vec3(100);
	
	for(int i = 0; i < 15; i++) {
		p = abs(p)/dot(p, p) - vec2(0.5, 0.3);
		col = min(col, vec3(abs(p.x), length(p), abs(3.0*p.y)));
	}
	
	return col;
}

// Function 32
float fractalMarch(vec3 ro, vec3 rd) {
    
    float c = 0., t = EPS;
    
    for (int i = 0; i < 50; i++) {
        
        vec3 p = ro + t * rd;
        
        vec3 q = tile(p);
        float b = sdSphere(q - vec3(1), SD);
        if (b > EPS) break;
        
        float bc = sdSphere(q - vec3(1), .01);
        bc = 1. / (1. + bc * bc * 20.);
        
        float fs = fractal(p); 
        t += 0.02 * exp(-2.0 * fs);
        
        c += 0.04 * bc;
    } 
    
    return c;
}

// Function 33
float fractalNoise(float freq, float lacunarity, float decay, vec2 threshold, vec3 p)
{  
  float res=0.0;
  float currentFreq=freq;
  float weight=1.0;
  float maxValue=0.0;
  // Always 5 octaves because the condition in the loop must be a constant.
  for(int i=0;i<5;i++)
  {

    res+=weight*cnoise(currentFreq*p);
    if(threshold.x==0.0||(res>threshold.x*float(i)&&res<threshold.y*float(i)))
          weight/=decay;
    else weight/=3.0;
    

    currentFreq*=lacunarity;
  }
  return res/5.0;
}

// Function 34
vec3 fractal(vec2 p){
    p.x=-p.x-(cos(iTime)+5.0)/3.0;
    vec3 col=vec3(0.0);
    vec2 z = vec2(0.0);
    int i;
    for (i=0;i<64;i++){

        // different iteration functions generate different snowflakes
        if (sn==0) z=vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+p;
        else if (sn==3) z=vec2(z.x*z.x-z.y*z.y,-2.0*z.x*z.y)+p;
        else if (sn==1) z=vec2(abs(z.x*z.x-z.y*z.y),2.0*z.x*z.y)+p;
        else if (sn==4) z=vec2(abs(z.x*z.x-z.y*z.y),-2.0*z.x*z.y)+p;
        else if (sn==2) z=vec2(z.x*z.x-z.y*z.y,-abs(2.0*z.x*z.y))+p;

        // color function for Mandelbrot (https://www.shadertoy.com/view/wl2SWt)
        float h = dot(z,z);
        if (h>1.8447e+19){
            float n = float(i)-log2(.5*log2(h))+4.;
            float m = exp(-n*n/20000.);
            n = mix(4.*pow((log(n+1.)+1.),2.),n,m);
            m = 5.*sin(.1*(n-6.))+n;
            col += vec3(
                pow(sin((m-8.)/20.),6.),
                pow(sin((m+1.)/20.),4.),
                (.8*pow(sin((m+2.)/20.),2.)+.2)*(1.-pow(abs(sin((m-14.)/20.)),12.))
            );
            break;
        }
    }
    if (i==64) col=vec3(1.0);
    return col;
}

// Function 35
vec3 shadeFractalBump(vec3 pos, vec3 rayDirection) {
    vec3 nor = calcNormal(pos);
    vec3 sn = normalize(nor + 0.04*matbump(pos, nor));
    vec3 sn2 = normalize(nor + 0.4*matbump(pos, nor));
    vec3 ref = normalize(reflect(rayDirection, sn));
    
    vec3 col = vec3(0.);
    col += pow(clamp(dot(-rayDirection, ref), 0.0, 1.0), 10.0);
    col += pow(clamp(1.0 + dot(rayDirection, sn2), 0.0, 1.0), 2.0);
    col *=3.0*matcube(pos, nor);
    return col;
}

// Function 36
float mandelbrot( in vec2 c )
{
    #if 1
    {
        float c2 = dot(c, c);
        // skip computation inside M1 - http://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
        if( 256.0*c2*c2 - 96.0*c2 + 32.0*c.x - 3.0 < 0.0 ) return 0.0;
        // skip computation inside M2 - http://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
        if( 16.0*(c2+2.0*c.x+1.0) - 1.0 < 0.0 ) return 0.0;
    }
    #endif


    const float B = 256.0;
    float l = 0.0;
    vec2 z  = vec2(0.0);
    for( int i=0; i<512; i++ )
    {
        z = vec2( z.x*z.x - z.y*z.y, 2.0*z.x*z.y ) + c;
        if( dot(z,z)>(B*B) ) break;
        l += 1.0;
    }

    if( l>511.0 ) return 0.0;
    
    // ------------------------------------------------------
    // smooth interation count
    //float sl = l - log(log(length(z))/log(B))/log(2.0);

    // equivalent optimized smooth interation count
    float sl = l - log2(log2(dot(z,z))) + 4.0;

    float al = smoothstep( -0.1, 0.0, sin(0.5*6.2831*iTime ) );
    l = mix( l, sl, al );

    return l;
}

// Function 37
vec4 col_fractal(vec4 p) {
	vec3 orbit = vec3(0.0);
	for (int i = 0; i < FRACTAL_ITER; ++i) {
		p.xyz = abs(p.xyz);
		rotZ(p, iFracAng1);
		mengerFold(p);
		rotX(p, iFracAng2);
		p *= iFracScale;
		p.xyz += iFracShift;
		orbit = max(orbit, p.xyz*iFracCol);
	}
	return vec4(orbit, de_box(p, vec3(6.0)));
}

// Function 38
float oldfractal(vec3 xyz){ 
    const float frequency = 0.5; 
    const int octaves=3;
    const float lacunarity = 2.0; 
    const float persistence = 0.6;
    
    float value = 0.0;
    float curpersistence = 1.0;

    vec3 space = xyz * frequency;

    for (int i = 0; i < octaves; i++){
        value += oldnoise(space) * curpersistence;
        curpersistence *= persistence;
        space *= lacunarity;
    }
    return value;
}

// Function 39
Sample fractal(vec3 co, float seed) {
    Sample s;
    
    vec3 center = vec3(1.0);
    vec3 co2 = co;
    int i;
    float th=0.0, thscale = 1.3, dscale = DSCALE;
    float k = 0.0, fi=0.0;

    //s = s_cube(co2, 0.5);
    s = s_sphere(co2, 0.5);
    
    float scale = dscale;
    
    for (i=0; i<IFSSTEPS; i++) {
        float x1, y1;
		
        /*
        co.xy = rot2d(co2.xy, th);
        co2.yz = rot2d(co2.yz, th);
        th += k;
        k += thscale*thscale;
		//*/
        
        co2 = floor(co*scale + 0.5)/scale;
        scale *= dscale;
        co2 = (co - co2);
        
        s_diff(s, s_cube(co2, 1.0/scale));
#ifdef LIGHTS
        float sz = 100.0;
        float f = tent(sz*co[0])*tent(sz*co[1])*tent(sz*co[2]);
        f = float(f > 0.75)*30.0;
        
        s.emission[1] = 0.25*f;
        s.emission[0] = f;
#endif       
        
    }
    
    return s;
}

// Function 40
vec4 sdFractal(vec3 p, float itr, vec2 cell, float sc)
{
    p.xz = p.xz+cell*0.5;
    vec2 cell_i = floor(p.xz/cell);
    p.xz = mod(p.xz, cell) -  cell*0.5;
    float box = sdBox(p.xzy, vec3(cell.xy*0.35,1000.));
    vec3 dp = ((length(cell_i)==0.)?iFracShift:iFracShift*0.9+3.*hash31(dot(cell_i,vec2(1.,sqrt(2.)))));
    float scale = 1.;
    vec3 orbit =  vec3(0.);
    vec3 col = vec3(0.);
    float norm = 0.;
    
    vec2 angl = (cell_i)*0.05 + vec2(iFracAng1,iFracAng2);
    float s1 = sin(angl.x), c1 = cos(angl.x);
    float s2 = sin(angl.y), c2 = cos(angl.y);

	for (float i = 0.; i < itr; i++) 
    {
		p = abs(p);
		rotZ(p, s1, c1);
		mengerFold(p);
		rotX(p, s2, c2);
        scale *= sc;
		p = p*sc + dp;
    	orbit = max(orbit, sin(.9*p*iFracCol));
	}
	return vec4(clamp(1.-0.8*orbit,0.,1.),max(sdBox(p, vec3(6.0))/scale, box));
}

// Function 41
vec4 fractal( in vec2 p ) {
    
    // keep current scale
    float scale = 1.0;
    
    // used to smoothstep
    float aliasBase = 1.0 / iResolution.y;
    
    // accumulated alpha
    float alpha = 0.0;
    // accumulated color
    vec3 color = vec3(0.0);
  
    #define LEVELS 10
    for (int i = 0 ; i < LEVELS ; i++) {
        
        // scale
        float s = 2.0;
		
        // repeat axis according to scale ala TEXTURE_ADDRESS_MIRROR
        p = 1.0 - abs(s*fract(p-0.5)-s*0.5);
        
        // fold
        float theta = float(i) * PI * 0.125;
        //theta = iTime*0.02 * float(i); // try this one
        p *= rot(theta);
        
        // update scale
        scale *= s;
        
        // jump first steps cause they're less interesting
        if (i < 4) continue;
        
        // texture
        
        // borders
        vec2 uv = abs(p);
        float delt1 = abs((hexDist(uv)-0.6)-0.1);
        float delt2 = min(length(uv)-0.2, min(uv.x, uv.y));
        float m = min(delt1, delt2);
        float alias = aliasBase*0.5*scale;
        float f = smoothstep(0.10+alias, 0.10, m)*0.4 + smoothstep(0.22, 0.11, m)*0.6;
        
        // pulse
        float r = length(uv)/0.707106;
        float t = mod(iTime*1.5, float(LEVELS-4)*2.0) - float(i);
        r = (r + 1.0 - t)*step(r*0.5, 1.0);
        r = smoothstep(0.0, 0.8, r) - smoothstep(0.9, 1.0, r);
        
        // mix colors
        vec3 c = vec3(smoothstep(0.06+alias, 0.06, m));
        vec3 hue = hsv2rgb( vec3(iTime*0.03+float(i)*0.08, 0.5, 1.0) );
        c = c*hue;
        c += c*r*1.5;
        
        // front to back compositing
        color = (1.0-alpha)*c+color;
        alpha = (1.0-alpha)*f+alpha;
        
    }
    
    return vec4(color, alpha);
}

// Function 42
float fractal(in vec2 uv) {
	float c = cos(1.0/float(SIDES)*TAU);
	float s = sin(1.0/float(SIDES)*TAU);
	
	mat2 m = mat2(c, s, -s, c);
	vec2 p = vec2(1.0, 0.0), r = p;
	
	for (int i = 0; i < 7; ++i) {
		float dmin = length(uv - r);
		for (int j = 0; j < SIDES; ++j) {
			p = m*p;
			float d = length(uv - p); 
			if (d < dmin) {dmin = d; r = p;}
		}
		uv = 2.0*uv - r;
	}
	
	return (length(uv-r)-0.15)/pow(2.0, 7.0);
}

// Function 43
vec3 mandelbrot(vec2 v, int max_iter)
{
    vec2 z = vec2(0.);
    vec2 c = v / (0.1 + iMouse.y / iResolution.y * 15.) ;
    c += vec2(-1.5+(iMouse.x / iResolution.x),.25);
    int i;
    float min_dist = 10000.;
    for ( i = 0; i < max_iter; i++ )
    {
        if ( length(z) > 2. ) break;
        z = sqrcomp(z) + c;
        
        float nm = length(z);
        if ( nm < min_dist ) min_dist = nm;
    }
    
    
    return mix(vec3(2./255., 0., 36./255.), vec3(0., 212./255., 1.), 1. - min_dist * 2.);
}

// Function 44
float PseudoErosionFractal( vec2 wsPos )
{
	float baseHeight = BaseHeightmap( wsPos );

#if (EROSION_OCTAVE_MODE == 0)
	float erosion = 0.0;
    
#elif (EROSION_OCTAVE_MODE == 1)
	//1 erosion octave
    float erosion = PseudoErosion( wsPos, GRIDSIZE );
    
#elif (EROSION_OCTAVE_MODE == 2)
    //fbm style

    float erosion = 0.0;
    
    const int numOctaves = 4;
    
    float scale = 1.0f;
    float gridSize = GRIDSIZE;
    
    for( int i=0 ; i < numOctaves ; ++i )
    {
	    erosion += PseudoErosion( wsPos, gridSize ) * scale;
        scale *= 0.5;
        gridSize *= 0.5;
    }

#elif (EROSION_OCTAVE_MODE == 3)
    //MacSearlas  style
 
    float r1 = PseudoErosion( wsPos, GRIDSIZE );
	float erosion = sqr(r1);
    
    float r2 = PseudoErosion( wsPos, GRIDSIZE / 2.0 );
    erosion += r1 * r2 / 2.0;
    
    float r3 = PseudoErosion( wsPos, GRIDSIZE / 4.0 );
	erosion += sqrt(r1*r2)*r3 / 3.0;

#endif
    return baseHeight * BASE_HEIGHT + erosion * EROSION_HEIGHT;

}

// Function 45
vec3 fractal(vec2 p)
{   
    
    //--- basic constants ---//
    
    const vec2	O		= vec2(1, 0);
    const float	s3d2	= sqrt(3.0) * 0.5;

    
    //--- moebius transform coefficients ---//
    
    const vec2 a = vec2(-0.5, -s3d2);
	const vec2 b = vec2(+1.5, -s3d2);
	const vec2 c = vec2(+0.5, +s3d2);
	const vec2 d = vec2(+1.5, -s3d2);

    
    //--- rolling cardioid ---//
    
    vec2 z = p;

    z = cSqrt(O - 4.0 * z) - O;
    z = cMobI(z, a, b, c, d);

    
    //--- domain warping as simple movement ---//
    
    float time = iMouse.z > 0.5? 10.0*(iMouse.x/iResolution.x - 0.5) : 2.0 * cos(0.12*iTime) - 2.0;
    
    z.x += time;
    
    
    //--- unrolling cardioid ---//
    
    z = cMob(z, a, b, c, d);
    z = 0.25 * (O - cMul(z + O, z + O));
    

    //--- generating "before unrolling" chessboard ---//
    
    vec2 q = floor(6.0*z);					// checkboard size
    bool ch = mod(q.x + q.y, 2.0) == 0.0;

    
    //--- calculate fractal ---//
    
    p = z;
      
	for (float i = 0.0; i < 160.0; i++)
    {
        z = cMul(z, z) + p;					// Mandelbrot formulae

		if (4.0 < dot(z, z))
        {
            float f = 1.0 - i/160.0;
            f *= f;

            return ch? f*vec3(0.7, 0.8, 0.5) : vec3(0.9*f);	// outside color
		}
	}

    
    //--- fractal body ---//

    time = abs(time);

    float d1 = abs(circlesegment(p + vec2(1.03, 0.0), 0.18, -0.77, +0.77) - 0.016);
    float d2 = abs(       length(p + vec2(0.87, 0.0)) - 0.06);

    if      (time < 0.45) { if (d1 < 0.01)                             return vec3(0); }
    else if (time < 0.65) { if (mix(d1, d2, 5.0*(time - 0.45)) < 0.01) return vec3(0); }
    else                  { if (d2 < 0.01)                             return vec3(0); }
    
    p.y = abs(p.y);
	if (abs(length(p + vec2(1.05, -0.1)) - 0.034) < 0.01) return vec3(0);
    
    return vec3(1);
}

// Function 46
vec3 mandelbrot(vec2 uv, vec2 z, vec2 c, float scale, bool flipX)
{
    vec3 col = vec3(0.0);
    int count = 0;
	for (; count < maxIterations; ++count) 
    {
        if (!flipX)
        {
 			z = complexZ(z) + c;
        }
        else
        {
            z = complexZ(z) - c;
        }
        
 		if (dot(z, z) > 4.0)
        {
            break;
 		}
    
        col = texture(iChannel0, uv).rgb;
        
        float d = abs(dot(z, vec2(0.05)));
    	float f = step(d, 1.0);
        col -= f;
        
        col = mix(col, vec3(0.2, 0.2, 0.6), 0.8);
    }
    return col * float(count) * scale;
}

// Function 47
vec4 mandelbrot(vec2 coord){
    return texture(iChannel0,coord/iResolution.xy);
}

// Function 48
vec3 julian(vec3 x, float power, float dist, int seed){
    //return vec3(0);
    //return Julia3D(x,seed,power);
    float abspower = abs(power);
    float cn = dist/power/2.;
    float a = (atan(x.y,x.x) + pi*2. * float(IHash(seed^0x6acb43d3 )%int(abspower) ) ) / power;
    
    float r = pow(dot(x.xy,x.xy), cn);
    return vec3(r*cos(a),r*sin(a),0);
}

// Function 49
float FractalNoise(in vec2 xy)
{
	float w = .7;
	float f = 0.0;

	for (int i = 0; i < 4; i++)
	{
		f += Noise(xy) * w;
		w *= 0.5;
		xy *= 2.7;
	}
	return f;
}

// Function 50
float fractal_noise(vec3 m) {
    return   0.5333333*simplex3d(m*rot1)
			+0.2666667*simplex3d(2.0*m*rot2)
			+0.1333333*simplex3d(4.0*m*rot3)
			+0.0666667*simplex3d(8.0*m);
}

// Function 51
int mandelbrot(vec2 uv) {
    vec2 z = uv;
    for (int i = 0; i < 300; i++) {
        // dot(z, z) > 4.0 is the same as length(z) > 2.0, but perhaps faster.
        if (dot(z, z) > 4.0) return i;
        // (x+yi)^2 = (x+yi) * (x+yi) = x^2 + (yi)^2 + 2xyi = x^2 - y^2 + 2xyi
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + uv;
    }
    return 0;
}

// Function 52
float JULIA(vec3 p)
{
    // Initialise iteration point (z), escape-time derivative (dz),
    // and Julia constant (c)
    // Also apply uniform scaling
    vec4 c = JULIA_C;
    vec4 z = vec4(p, c.w);
    vec4 dz = vec4(1.0f, 0.0f, 0.0f, 0.0f);

    // Iterate the fractal
    int i = 0;
    float sqrBailout = JULIA_BAILOUT;
    while (i < JULIA_ITER &&
           dot(z, z) < sqrBailout)
    {
        // Update the Julia differential [dz]
        dz = 2.0f * QtnProduct(z, dz);

        // Displace the Julia coordinate [z]
        z = QtnProduct(z, z) + c;

        // Update the iteration count
        i += 1;
    }

    // Compute, return distance
    float r = length(z);
    return (0.5 * log(r) * r / length(dz));
}

// Function 53
float julia(vec3 pos)
{
	vec4 c = vec4(-0.1,0.5,0.5,-0.3);
    vec4 z = vec4( pos, 0.0 );
	vec4 nz;
    
	float md2 = 1.0;
	float mz2 = dot(z,z);

	for(int i=0;i<7;i++)
	{
		md2*=4.0*mz2;
	    nz.x=z.x*z.x-dot(z.yzw,z.yzw);
		nz.yzw=2.0*z.x*z.yzw;
		z=nz+c;

		mz2 = dot(z,z);
		if(mz2>4.0)
        {
			break;
        }
	}

	return 0.25*sqrt(mz2/md2)*log(mz2);
}

// Function 54
vec2 julia(vec2 x, int seed){
    float a = atan(x.y, x.x)/2.0;
    if(Hash(seed)>.5) a += 3.14159;
    float r = pow(dot(x,x), 0.25);
    return vec2(r*cos(a),r*sin(a));
}

// Function 55
int julia(vec2 z, vec2 c){
    for(int i = 0; i < 12; i++){
        z = vec2(z.x*z.x - z.y*z.y, 2.*z.x*z.y) + c;
        if(z.x*z.x+z.y*z.y > 4.) return i;
    }
    return MaxIterations;
}

// Function 56
vec4 fractalNoise(vec2 coord) {
    vec4 value = vec4(0.0);
    float scale = 0.5;
    for (int i = 0; i < 5; i += 1) {
     	value += texture(iChannel0, coord) * scale;
        coord *= 2.0;
        scale *= 0.6;
    }
    return value;
}

// Function 57
vec4 fractal(vec2 cx)
{
	cx.x += 0.1;
	cx.y += 0.1;
	vec2 z = cx;
	float cc = 1.0;
	for(int i=0;i<22;i++)
	{
		vec2 z1;
		z1.x = z.x*z.x - z.y*z.y + cx.x;
		z1.y = 2.0*z.x*z.y + cx.y;
		z = z1;
        if(dot(z,z)>4.0)
        {
            cc = float(i)/21.0;
            break;
        }
		//cc += clamp((dot(z,z)-4.0)*10.0,0.0,1.0)/21.0;
	}
    if(cc>1.0-0.2) return vec4(0,0,0.1,0);
    cc *= 2.5;
    cc += 0.5;
    cc /= 1.0+length(cx);
    return vec4(cc,cc,cc*2.0,0);
}

// Function 58
float fractalblobnoise(vec2 v, float s)
{
    float val = 0.;
    const float n = 4.;
    for(float i = 0.; i < n; i++)
        //val += 1.0 / (i + 1.0) * blobnoise((i + 1.0) * v + vec2(0.0, iTime * 1.0), s);
    	val += pow(0.5, i+1.) * blobnoise(exp2(i) * v + vec2(0, T), s);

    return val;
}

// Function 59
vec3 fractalMaterial3(vec2 uv) {
    const int MAGIC_BOX_ITERS = 15;
	const float MAGIC_BOX_MAGIC = .55;
    const mat3 M = mat3(0.28862355854826727, 0.6997227302779844, 0.6535170557707412,
                    0.06997493955670424, 0.6653237235314099, -0.7432683571499161,
                    -0.9548821651308448, 0.26025457467376617, 0.14306504491456504);
    uv.x += 0.4;
    vec3 p = 0.5*M*vec3(uv, 1.0);
        p = 1.0 - abs(1.0 - mod(p, 2.0));
    
    float lastLength = length(p);
    float tot = 0.0;
    for (int i=0; i < MAGIC_BOX_ITERS; i++) {
      p = abs(p)/(lastLength*lastLength) - MAGIC_BOX_MAGIC;
      float newLength = length(p);
      tot += abs(newLength-lastLength);
      lastLength = newLength;
    }

    tot *= 0.03;
    return vec3(tot*uv.x, tot*tot, 4.*tot*uv.y*uv.y);
}

// Function 60
float fractal( vec2 point )
{
    float sum = 0.0;
    float scale = 0.5;
    for ( int i = 0; i < 5; i++ )
	{
		sum += noise( point ) * scale;
		point *= 2.0;
        scale /= 2.0;
	}
    
	return sum;
}

// Function 61
float fractalblobnoise(vec2 v, float s)
{
    float val = 0.;
    const float n = 4.;
    for(float i = 0.; i < n; i++)
    	val += pow(0.5, i+1.) * blobnoise(exp2(i) * v + vec2(0, T), s);
    return val;
}

// Function 62
float FractalNoise(in vec2 xy)
{
	float w = .8;
	float f = 0.0;

	for (int i = 0; i < 4; i++)
	{
		f += CloudNoise(xy) * w;
		w = w*0.5;
		xy = rotate2D * xy;
	}
	return f;
}

// Function 63
int fractal(vec2 p, vec2 point) {
	vec2 so = (-1.0 + 2.0 * point) * 0.4;
	vec2 seed = vec2(0.098386255 + so.x, 0.6387662 + so.y);
	
	for (int i = 0; i < iters; i++) {
		
		if (length(p) > 2.0) {
			return i;
		}
		vec2 r = p;
		p = vec2(p.x * p.x - p.y * p.y, 2.0* p.x * p.y);
		p = vec2(p.x * r.x - p.y * r.y + seed.x, r.x * p.y + p.x * r.y + seed.y);
	}
	
	return 0;	
}

// Function 64
int mandelbrot(float x, float y) {
  complex c = complex(x, y);
  complex z = complex(0.0, 0.0);

  return fractal(c, z);
}

// Function 65
float fractalNoise(in vec3 loc) {
	float n = 0.0 ;
	for (int octave=1; octave<=NUM_OCTAVES; octave++) {	
		n = n + snoise(loc/float(octave*8)) ; 
	}
	return n ;
}

// Function 66
vec2 julia(vec2 pos, vec2 c) {
	for(int i = 0; i < JULIA_ITERATIONS; i++) {
		vec2 oldpos = pos;
		for(int j = 0; j < JULIA_MULTS; j++) {
			pos = cpx_mult(pos, oldpos);
		}
		pos += c;
	}
	return pos;
}

// Function 67
v0 fractalFungus(v2 p){p=1.-abs(1.-mod(p,2.));v2 f=v2(0,length(p),0)
 ;for(int i=kifsFungusIter;i>0;i--      
 ){p=abs(p)/(f.y*f.y)-kifsFungusSeed;f.z=length(p);f=v2(f.x+abs(f.z-f.y),f.zz);}return f.x;}

// Function 68
vec3 NewtonFractal(vec2 z){
    z = NewtonsMethod(z);

    float dr1 = distance(z,Root1);
    float dr2 = distance(z,Root2);
    float dr3 = distance(z,Root3);
    
    return normalize(color1/dr1+color2/dr2+color3/dr3);
}

// Function 69
vec3 GetPixelFractal(vec2 pos, int iterations, float timePercent)
{
    int glyphLast = GetFocusGlyph(iterations-1);
	ivec2 glyphPosLast = GetFocusPos(-2);
	ivec2 glyphPos =     GetFocusPos(-1);
    
	bool isFocus = true;
    ivec2 focusPos = glyphPos;
    
	vec3 color = InitPixelColor();
	for (int r = 0; r <= recursionCount + 1; ++r)
	{
        color = CombinePixelColor(color, timePercent, iterations, r, pos, glyphPos, glyphPosLast);
        
        //if (r == 1 && glyphPos == GetFocusPos(r-1))
	    //    color.z = 1.0; // debug - show focus
        
        if (r > recursionCount)
			return color;
           
        // update pos
        pos -= vec2(glyphMargin*gsfi);
        pos *= glyphSizeF;

        // get glyph and pos within that glyph
        glyphPosLast = glyphPos;
        glyphPos = ivec2(pos);

        // check pixel
        int glyphValue = GetGlyphPixel(glyphPos, glyphLast);
		if (glyphValue == 0 || pos.x < 0.0 || pos.y < 0.0)
			return color;
        
        // next glyph
		pos -= vec2(floor(pos));
        focusPos = isFocus? GetFocusPos(r) : ivec2(-10);
        glyphLast = GetGlyph(iterations + r, glyphPos, glyphLast, glyphPosLast, focusPos);
        isFocus = isFocus && (glyphPos == focusPos);
	}
}

// Function 70
vec4 getFractalColor(vec2 tc)
{
	float x0 = mod(0.1*tc.x + 0.9, 2.0);
	float y0 = mod(0.025*(1.0 - tc.y) + 0.7, 1.0);

	float z0_r = 0.0;
	float z0_i = 0.0;
	float z1_r = 0.0;
	float z1_i = 0.0;
	float p_r = (x0 + xpos * z_fractal) / z_fractal;
	float p_i = (y0 + ypos * z_fractal) / z_fractal;
	float d = 0.0;

	float nn = 0.0;
	for (float n=0.0; n<iter; n++)
	{
		z1_r = z0_r * z0_r - z0_i * z0_i + p_r;
		z1_i = 2.0 * z0_r * z0_i + p_i;
		d = sqrt(z1_i * z1_i + z1_r * z1_r);
		z0_r = z1_r;
		z0_i = z1_i;
		if (d > iter2) break;
		nn++;
	}

	float c = (1.0*nn) / iter;
	if (c==1.0) c = 0.0;
	c *= 4.0;
	vec4 color = vec4(1.0*c, 1.0*c, 4.0*c, 0.0);
	return color;
}

// Function 71
float FractalNoise(vec3 pos)
{
    return Noise(pos)+Noise(pos*2.0)*0.5+Noise(pos*4.0)*0.25+Noise(pos*8.0)*0.125;
}

// Function 72
float fractal(float t, float repX, float repY, float freq, uvec4 p) {
    vec4 h = vec4(hash(p))*I2F;
    vec2 pos = fract(vec2(t/repX,floor(t/repX)/repY));
    
    //apply folds based off hash
    for (int i = 0; i < 8; i++) {
        float fi = h[i/2], rv = h[(i+1)%4];
        if (i%2 == 0) fi = fract(fi*10.)*10.;
        else fi = floor(fi*10.);
           
        int id = int(fi)%4;
        if (id == 0) {//mirror rotate fold
            pos = (abs(pos)-.5)*r2(rv*6.28);
        } else if (id == 1) {//plane fold
            rv *= 6.28;
            vec2 pnorm = vec2(sin(rv),cos(rv));
            pos -= pnorm*2.*min(0.,dot(pos,pnorm));
        } else {//polar fold
            float sz = .04+rv*1.6,
                ang = mod(atan(pos.y,pos.x),sz)-sz*.5;
            pos = vec2(sin(ang),cos(ang))*length(pos);
        }
        //apply box fold
        float ext = h[i%4];
        pos = clamp(pos,-ext,ext)*2.-pos;
    }
    float l = length(pos)*freq;
    
    
    //try different waves
    return triwave(l)*2.-1.;
    //return squarewave(l)*2.-1.;
    //return triwave2(l)*2.-1.;
    //return sin(l*6.28);
}

// Function 73
float julia(vec3 p, in vec4 c)
{
	vec4 z = vec4(p, 0.);
    float md2 = 1.;
    
    for( int i=0; i<11; i++ )
    {
        // https://www.cs.cmu.edu/~kmcrane/Projects/QuaternionJulia/paper.pdf
        // http://www.fractalforums.com/3d-fractal-generation/true-3d-mandlebrot-type-fractal/435/
        // |dz|^2 -> 4*|dz|^2
        // z'n+1 = 2zn * z'n
        md2 = 4.*dot(z, z)*md2;
        
        // z -> z2 + c
        z = vec4( z.x * z.x - z.y * z.y - z.z * z.z - z.w * z.w, 
                  2.0 * z.x * z.y, 
                  2.0 * z.x * z.z, 
                  2.0 * z.x * z.w) + c;

        // Just like in 2D Julia set: https://www.shadertoy.com/view/XlSyDK
        if(dot(z, z)>4.) break;
    }
    
	// iq explained the following formula here: 
	// http://www.fractalforums.com/3d-fractal-generation/true-3d-mandlebrot-type-fractal/msg8505/#msg8505
    // http://www.fractalforums.com/3d-fractal-generation/true-3d-mandlebrot-type-fractal/450/
    // https://en.wikipedia.org/wiki/Koebe_quarter_theorem
    // DE = (power/4) · |Z|·log |Z| / |dZ|
    return .25*sqrt(dot(z, z)/md2)*log(dot(z, z));
}

// Function 74
float julia(vec3 p,vec4 q) {
    vec4 nz, z = vec4(p,0.0);
    float z2 = dot(p,p), md2 = 1.0;    
    for(int i = 0; i < 11; i++) {
        md2 *= 4.0*z2;
        nz.x = z.x*z.x-dot(z.yzw,z.yzw);
        nz.y = 2.0*(z.x*z.y + z.w*z.z);
        nz.z = 2.0*(z.x*z.z + z.w*z.y);
        nz.w = 2.0*(z.x*z.w - z.y*z.z);
        z = nz + q;
        z2 = dot(z,z);
        if(z2 > 4.0) break;
    }    
	return 0.25*sqrt(z2/md2)*log(z2);    
}

// Function 75
vec3 julia3D(vec3 x, int seed, float m_N){
    float m_AbsN = abs(m_N);
    float m_Cn = (1. / m_N - 1.) / 2.;
	float z = x.z / m_AbsN;
	float r = 1./pow(dot(x.xy,x.xy) + z*z, -m_Cn);
	float tmp = r * length(x.xy);
	float ang = (atan(x.y,x.x) + pi*2. * float(IHash(seed^0x6acb43d3 )%int(m_AbsN) ) ) / m_N;
	return vec3(tmp * cos(ang),tmp * sin(ang),r * z);
}

// Function 76
vec3 julia(in highp vec2 position, highp float power)
{
    highp vec2 z = vec2(0.0, 0.0);
    const vec3 colors[30] = vec3[30]
    (
        vec3(0.1, 0.0, 0.9),
        vec3(0.2, 0.0, 0.8),
        vec3(0.3, 0.0, 0.7),
        vec3(0.4, 0.0, 0.6),
        vec3(0.5, 0.0, 0.5),
        vec3(0.6, 0.0, 0.4),
        vec3(0.7, 0.0, 0.3),
        vec3(0.8, 0.0, 0.2),
        vec3(0.9, 0.0, 0.1),
        vec3(1.0, 0.0, 0.0),
        vec3(0.9, 0.1, 0.0),
        vec3(0.8, 0.2, 0.0),
        vec3(0.7, 0.3, 0.0),
        vec3(0.6, 0.4, 0.0),
        vec3(0.5, 0.5, 0.0),
        vec3(0.4, 0.6, 0.0),
        vec3(0.3, 0.7, 0.0),
        vec3(0.2, 0.8, 0.0),
        vec3(0.1, 0.9, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.9, 0.1),
        vec3(0.0, 0.8, 0.2),
        vec3(0.0, 0.7, 0.3),
        vec3(0.0, 0.6, 0.4),
        vec3(0.0, 0.5, 0.5),
        vec3(0.0, 0.4, 0.6),
        vec3(0.0, 0.3, 0.7),
        vec3(0.0, 0.2, 0.8),
        vec3(0.0, 0.1, 0.9),
        vec3(0.0, 0.0, 1.0)
    );
    for (int i = 0, j; i < 1024; i++)
    {
        z = complexPower(z, vec2(power, 0.0)) + position;
        if (sqrMagnitude(z) > 4.0)
        {
            return colors[i % colors.length()];
        }
    }
    return vec3(0.0, 0.0, 0.0);
}

// Function 77
float simplex3d_fractal(vec3 m) {
    return   0.5333333*simplex3d(m*rot1)
			+0.2666667*simplex3d(2.0*m*rot2)
			+0.1333333*simplex3d(4.0*m*rot3)
			+0.0666667*simplex3d(8.0*m);
}

// Function 78
float mainFractal( vec3 p){
    
    vec3 p1 = p;
    //
    //p.x = abs(p.x)-20.;
    
    //p1.yx = abs(p1.yx);
    p1.yz*=rot(2.34);
    float k, sc = 1.;
    for(float i = 0.;i < 10.; i++){
        p1.xz*=rot(p.y*.06251+i*i+cc(t*.31)+t*.25);
        p1=abs(p1)-.2246-i*.215;
        k = max(1., 1.877/dot(p1,p1));
        p1 *= k;
        sc *= k;
    }
    
    p1.z = (fract(p1.z/20.-.5)-.5)*20.;
    
    float d = sbb(p1, vec3(1.)) / sc;
    fr1 += .15/(.1+d*d);
    d2 = d;
    //d = max(d, -sbb(p, vec3(15.)));
    return d*.26;
}

// Function 79
vec3 juliaPalette(in float t) {
    return palette(t, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), 
                      vec3(1.0, 1.0, 0.5), vec3(0.80, 0.90, 0.30));
}

// Function 80
vec3 drawFractal(vec2 z) {
    for (int iterations = 0; iterations < MAX_ITERATIONS; ++iterations) {
    	z = f(z);
        
        for (int root = 0; root < roots.length(); ++root) {
            vec2 difference = z - roots[root];
            float distance = dot(difference, difference);
            if (distance < THRESHOLD) {
    			return palette[root] * (0.75 + 0.25 * cos(0.25 * (float(iterations) - log2(log(distance) / log(THRESHOLD)))));
            }
        }
    }
}

// Function 81
float fractalNoise(vec2 p) {
  float total = 0.0;
  total += smoothNoise(p);
  total += smoothNoise(p*2.) / 2.;
  total += smoothNoise(p*4.) / 4.;
  total += smoothNoise(p*8.) / 8.;
  total += smoothNoise(p*16.) / 16.;
  total /= 1. + 1./2. + 1./4. + 1./8. + 1./16.;
  return total;
}

// Function 82
float Multifractal( vec3 pos )
{
    float rot = tau*1./3.;
    float c = cos(rot);
    vec2 s = sin(rot)*vec2(-1,1);
    vec3 p = pos;
    p.xy = p.xy*c + p.yx*s;
    float f = 0.;
    float strength = 1.;
    for ( int octave=0; octave < 4; octave++ )
    {
        f += texture(iChannel0,p).r*strength;
        // scale to next octave
        p *= 2.;
        strength *= .5;
        // to hide the grid artefacts, rotate the coordinates
        p.xyz = p.yzx;
        p.x = -p.x;
	    p.xy = p.xy*c + p.yx*s;
    }
    
    // normalize values into [0,1] range
    f /= (2.-strength*2.); // e.g. (1+.5+.25) / (2.-.125*2.) = 1.
    
    return f;
}

// Function 83
float fractal_noise(vec3 p)
{
    float f = 0.0;
    // add animation
    //p = p - vec3(1.0, 1.0, 0.0) * iTime * 0.1;
    p = p * 3.0;
    f += 0.50000 * noise(p); p = 2.0 * p;
	f += 0.25000 * noise(p); p = 2.0 * p;
	f += 0.12500 * noise(p); p = 2.0 * p;
	f += 0.06250 * noise(p); p = 2.0 * p;
    f += 0.03125 * noise(p);
    
    return f;
}

// Function 84
float fractalNoise(vec2 vl, out vec4 der) {
    float persistance = 2.;
    float amplitude = 1.2;
    float rez = 0.0;
    vec2 p = vl;
    vec4 temp;
    float norm = 0.;
    der = vec4(0.);
    for (int i = 0; i < OCTAVES + 2; i++) {
        norm += amplitude;
        rez += amplitude * valueNoiseSimple(p, temp);
        // to use as normals, we need to take into account whole length,
        // we can either normalize vector here or don't apply the amplitude
        der += temp;
        amplitude /= persistance;
        p *= persistance;
    }
    return rez / norm;
}

// Function 85
int animatedJulia(float x, float y) {
  float animationOffset = 0.055 * cos(iTime * 2.0);

  complex c = complex(-0.795 + animationOffset, 0.2321);
  complex z = complex(x, y);

  return fractal(c, z);
}

// Function 86
int julia(in vec2 z0, in vec2 c)
{
    vec2 z = z0;
    if (length(z) > 4.0)
        return 0;
    for (int n=0; n <counts; n++) {
        z = cmult(z,z) +c;
        if (length(z) > 4.0)
            return n;
    }
    return -1;
}

// Function 87
float fractalNoiseLow(vec2 vl) {
    float persistance = 2.;
    float amplitude = 1.2;
    float rez = 0.0;
    vec2 p = vl;
    float norm = 0.0;
    for (int i = 0; i < OCTAVES - 3; i++) {
        norm += amplitude;
        rez += amplitude * valueNoiseSimpleLow(p);
        amplitude /= persistance;
        p *= persistance;
    }
    return rez / norm;
}

// Function 88
float julianDay2000(in float unixTimeMs) {
	return (unixTimeMs/86400.) - 10957.5;// = + 2440587.5-2451545; 
}

// Function 89
int MandelbrotSet(vec2 z0, vec2 c, int iterMax)
{
    float zLength = 0.0;
    int i = 0;
    vec2 z = z0; //z.x = real part; z.y = imaginary part
    
    // if zLength is bigger than 2 the series diverges, then return the iterations needed
    // otherwise go until max-iterations, which suggests the series is converging
    while (zLength <= 2.0 && i < iterMax)
    {
        i++;
        z = product_i(z, z) + c; //zn+1 = zn²+c
        zLength = length(z);     //absolut value of imaginary number z is it's length
    }
    return i;
}

// Function 90
bool fractal(inout vec2 coord, in float r) 
{
    // Complex multiplication: (x + yi)^2 = x^2 + 2*x*(yi) + (yi)^2
	float a = coord.x * coord.x;
    float b = 2.0 * coord.x * coord.y;
    float c = -(coord.y * coord.y); 
    
    coord = vec2(a + c + r, b + CONSTANT);
    
    //Did the coordinate reach our predefined infinity?
    return length(coord) > INFINITY;
}

// Function 91
float fractal_brownian_motion(in vec3 x) {
    vec3 p = rotate(x);
    float f = 0.0;
    f += 0.5000*noise(p); p = p*2.32;
    f += 0.2500*noise(p); p = p*3.03;
    f += 0.1250*noise(p); p = p*2.61;
    f += 0.0625*noise(p);
    return f/0.9375;    
}

// Function 92
float newfractal(ivec3 ixyz, vec3 fxyz){ 
    // instead of floats, we need to use integer period and integer lacunarity
    const int period = 2; 
    const int octaves=3;
    const int lacunarity = 2; 
    const float persistence = 0.6;
    
    float value = 0.0;
    float curpersistence = 1.0;

    ivec3 ispace = ixyz / period;
    vec3 fspace = vec3(ixyz - ispace * period) / vec3(period) + fxyz / vec3(period);

    for (int i = 0; i < octaves; i++){
        value += newnoise(ispace, fspace) * curpersistence;
        curpersistence *= persistence;
        ispace *= lacunarity;
        fspace *= float(lacunarity);
    }
    return value;
}

// Function 93
float fractalNoise(vec2 p) {

    float x = 0.;
    x += smoothNoise(p      );
    x += smoothNoise(p * 2. ) / 2.;
    x += smoothNoise(p * 4. ) / 4.;
    x += smoothNoise(p * 8. ) / 8.;
    x += smoothNoise(p * 16.) / 16.;
    x /= 1. + 1./2. + 1./4. + 1./8. + 1./16.;
    return x;
            
}

// Function 94
float fractal(vec3 p)
{
    const int iterations = 20;
	
    float d = iTime*5. - p.z;
   	p=p.yxz;
    pR(p.yz, 1.570795);
    p.x += 6.5;

    p.yz = mod(abs(p.yz)-.0, 20.) - 10.;
    float scale = 1.25;
    
    p.xy /= (1.+d*d*0.0005);
    
	float l = 0.;
	
    for (int i=0; i<iterations; i++) {
		p.xy = abs(p.xy);
		p = p*scale + vec3(-3. + d*0.0095,-1.5,-.5);
        
		pR(p.xy,0.35-d*0.015);
		pR(p.yz,0.5+d*0.02);
		
        l =length6(p);
	}
	return l*pow(scale, -float(iterations))-.15;
}

// Function 95
float de_fractal(vec4 p) {
	for (int i = 0; i < FRACTAL_ITER; ++i) {
		p.xyz = abs(p.xyz);
		rotZ(p, iFracAng1);
		mengerFold(p);
		rotX(p, iFracAng2);
		p *= iFracScale;
		p.xyz += iFracShift;
	}
	return de_box(p, vec3(6.0));
}

// Function 96
float fractal1(vec3 u){
;u*=scaleVR
;vec3 p=Tile(u)
;vec4 z=vec4(p,1)
;float dG=1e3
;for(float n=.0;n<iterDfFractal;n++ //fractal
){z.xyz=clamp(z.xyz,-1.,1.)*2.-z.xyz
 ;z*=scale/clamp(max(dot(z.xy,z.xy),max(dot(z.xz,z.xz),dot(z.yz,z.yz))),mr,mxr)
 ;z+=p0
 ;if(n==2.)dG=DERect(z,rcL);
}
;dG=min(dG,DERect(z,rc))
#ifdef cutY
;dG=max(dG,u.y);
#endif
;return dG/scaleVR;
}

// Function 97
float fractalNoise(vec2 v) 
{
    // initialize
    float value = 0.0;
    float amplitude = 0.5;
    // loop
    for (int i = 0; i < 4; i++) 
    {
        value += amplitude * gradientNoise(v);
        // double the frequency
        v *= 2.0;
        // half the amplitude
        amplitude *= 0.5;
    }
    return value;
}

// Function 98
vec3 fractal(vec2 p)
{        
    vec2 z = vec2(0);
    
	for (int i = 0; i < ITER; ++i) {  
		z = vec2(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + p;
                
		if (dot(z,z) > 4.) {
			float s = .125662 * float(i);
			return vec3(cos(s + .9), cos(s + .3), cos(s + .2)) * .4 + .6;
		}  
	}

    return COL_IN;
	
}

// Function 99
vec3 mandelbrot(in vec2 uv, vec3 col) {
    uv.x += 1.5;
    uv.x = -uv.x;

    float a=.05*sqrt(abs(Anim)), ca = cos(a), sa = sin(a);
    mat2 rot = mat2(ca,-sa,sa,ca);
    uv *= rot;
	float kk=0., k = abs(.15+.01*Anim);
    uv *= mix(.02, 2., k);
	uv.x-=(1.-k)*1.8;
    vec2 z = vec2(0);
    vec3 c = vec3(0);
    for(int i=0;i<50;i++) {
        if(length(z) >= 4.0) break;
        z = vec2(z.x*z.x-z.y*z.y, 2.*z.y*z.x) + uv;
        if(length(z) >= 4.0) {
            kk = float(i)*.07;
            break; // does not works on some engines !
        }
    }
    return clamp(mix(vec3(.1,.1,.2), clamp(col*kk*kk,0.,1.), .6+.4*Anim),0.,1.);
}

// Function 100
vec2 fractalBox(in vec3 p)
{
   p.xz *= rot(-2. + 2.3*psin((iTime - 17.)*0.06));

   moda(p.xy, 15.);
   pMod1(p.x, 1.);

   float d = sdBox(p,vec3(1.0)) - 0.0;
   vec2 res = vec2( d, T_BOX);

   float bpm = 133.;
   float beat = mod(iTime, 60.f / bpm * 4.f);
   float f = 1.*smoothspike(0.0, 0.45, beat);

   float tim = mod(iTime + 1.7 - 2.0, 4.0);
   float s = 0.7 + 0.3*smoothstep(1.7, 2.0, tim) - 0.3*smoothstep(3.7, 4.0, tim);

   for( int m=0; m<3; m++ )
   {
      vec3 newp = p;
      vec3 a = mod( newp * s, 2.0 ) - 1.0;
      s *= 3.0;
      vec3 r = abs(1.0 - 3.0*abs(a));
      float da = max(r.x,r.y);
      float db = max(r.y,r.z);
      float dc = max(r.z,r.x);
      float c = (min(da,min(db,dc))-1.0)/s;

      if( c>d )
      {
          d = c;
          res = vec2( d, T_BOX);
       }
   }
    

   return res;
}

// Function 101
float juliaRect(vec2 p, vec2 center, vec2 size) {
    vec2 hs = size / 2.;
    p -= center;
    vec2 z = (p / hs) * 1.5;
    vec2 pabs = abs(p);
    if(0. < max(pabs.x - hs.x, pabs.y - hs.y)) return 1000.;
    
    vec2 c = vec2(-0.618, 0.);
    for(int i = 0 ; i < 1024; i++){
    	z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;

        if( dot(z, z) > 4.0){
        	return 1000.;
        }
    }
    return -1.;
}

// Function 102
float fractalNoiseLow(vec2 vl, out float mainWave) {
    #if SHARP_MODE==1
    const float persistance = 2.4;
    float frequency = 2.2;
    const float freq_mul = 2.2;
    float amplitude = .4;
#else
    const float persistance = 3.0;
    float frequency = 2.3;
    const float freq_mul = 2.3;
    float amplitude = .7;
#endif
    
    float rez = 0.0;
    vec2 p = vl;
    
    float mainOfset = (iTime + 40.)/ 2.;
    
    vec2 waveDir = vec2(p.x+ mainOfset, p.y + mainOfset);
    float firstFront = amplitude + 
			        (valueNoiseSimple(p) * 2. - 1.);
    mainWave = firstFront * valueNoiseSimple(p + mainOfset);
    
    rez += mainWave;
    amplitude /= persistance;
    p *= unique_transform;
    p *= frequency;
    

    float timeOffset = iTime / 4.;

    
    for (int i = 1; i < OCTAVES - 5; i++) {
        waveDir = p;
        waveDir.x += timeOffset;
        rez += amplitude * sin(valueNoiseSimple(waveDir * frequency) * .5 );
        amplitude /= persistance;
        p *= unique_transform;
        frequency *= freq_mul;

        timeOffset *= 1.025;

        timeOffset *= -1.;
    }

    return rez;
}

// Function 103
float fractal(vec2 p) {
    float v = 0.5;
    v += noise2D(p*16.); v*=.5;
    v += noise2D(p*8.); v*=.5;
    v += noise2D(p*4.); v*=.5;
    v += noise2D(p*2.); v*=.5;
    v += noise2D(p*1.); v*=.5;
    return v;
}

// Function 104
float distanceToMandelbrot( in vec2 c )
{
    #if 1
    {
        float c2 = dot(c, c);
        // skip computation inside M1 - http://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
        if( 256.0*c2*c2 - 96.0*c2 + 32.0*c.x - 3.0 < 0.0 ) return 0.0;
        // skip computation inside M2 - http://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
        if( 16.0*(c2+2.0*c.x+1.0) - 1.0 < 0.0 ) return 0.0;
    }
    #endif

    // iterate
    float di =  1.0;
    vec2 z  = vec2(0.0);
    float m2 = 0.0;
    vec2 dz = vec2(0.0);
    for( int i=0; i<300; i++ )
    {
        if( m2>1024.0 ) { di=0.0; break; }

		// Z' -> 2·Z·Z' + 1
        dz = 2.0*vec2(z.x*dz.x-z.y*dz.y, z.x*dz.y + z.y*dz.x) + vec2(1.0,0.0);
			
        // Z -> Z² + c			
        z = vec2( z.x*z.x - z.y*z.y, 2.0*z.x*z.y ) + c;
			
        m2 = dot(z,z);
    }

    // distance	
	// d(c) = |Z|·log|Z|/|Z'|
	float d = 0.5*sqrt(dot(z,z)/dot(dz,dz))*log(dot(z,z));
    if( di>0.5 ) d=0.0;
	
    return d;
}

// Function 105
vec3 drawFractal( in float k, in vec2 fragCoord )
{
    vec3 col = vec3(0.0);
    
#if AA>1
    for( int m=0; m<AA; m++ )
    for( int n=0; n<AA; n++ )
    {
        vec2 o = vec2(float(m),float(n)) / float(AA) - 0.5;
        vec2 p = (-iResolution.xy + 2.0*(fragCoord+o))/iResolution.y;
#else    
        vec2 p = (-iResolution.xy + 2.0*fragCoord)/iResolution.y;
#endif

        vec2 c = p * 1.25;

        #if 0
        if( k==2.0 )
        {
        float c2 = dot(c, c);
        // skip computation inside M1 - http://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
    	if( 256.0*c2*c2 - 96.0*c2 + 32.0*c.x - 3.0 < 0.0 ) continue;
    	// skip computation inside M2 - http://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
    	if( 16.0*(c2+2.0*c.x+1.0) - 1.0 < 0.0 ) continue;
        }
        #endif
        
        const float threshold = 64.0;
        vec2 z = vec2( 0.0 );
        float it = 0.0;
        for( int i=0; i<100; i++ )
        {
            z = cpow(z, k) + c;
            if( dot(z,z)>threshold ) break;
            it++;
        }

        vec3 tmp = vec3(0.0);
        if( it<99.5 )
        {
            float sit = it - log2(log2(dot(z,z))/(log2(threshold)))/log2(k); // http://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
            tmp = 0.5 + 0.5*cos( 3.0 + sit*0.075*k + vec3(0.0,0.6,1.0));
        }
        
        col += tmp;
#if AA>1
    }
    col /= float(AA*AA);
#endif

	return col;
}

// Function 106
float getJuliaDE(DualQuaternion dd, vec3 inPosition, out vec3 outNormal, out int i)
{
    Quaternion c = Quaternion(inPosition, 0);

    //gradient
    DualQuaternion dx = DualQuaternion(c, Quaternion(1,0,0,0));
    DualQuaternion dy = DualQuaternion(c, Quaternion(0,1,0,0));
    DualQuaternion dz = DualQuaternion(c, Quaternion(0,0,1,0));

    for(i = 0; i <= 10; i++)
    {
        if(H_sqnorm(dx[0]) > 16.0)
        {
            break;
        }

        dx = D_add(D_multiply(dx, dx), dd);
        dy = D_add(D_multiply(dy, dy), dd);
        dz = D_add(D_multiply(dz, dz), dd);
    }

    //the final position is the same for all partials
    vec3 fp = dx[0].xyz;
    float r = H_norm(dx[0]);
    
    vec3 vdx = vec3(dx[1]);
    vec3 vdy = vec3(dy[1]);
    vec3 vdz = vec3(dz[1]);
    float dr = length(vdx) + length(vdy) + length(vdz);
    outNormal = normalize(vec3(dot(fp, vdx), dot(fp, vdy), dot(fp, vdz)));

  	return 0.5 * log(r) * r / dr;//better for low iteration counts
    //return 0.5 * r / dr;
}

// Function 107
vec2 fractalNoise2d(vec2 p, float t, float scale)
{
    return vec2(map5_value1(vec3(p*scale, t)), 
                map5_value1(vec3(p*scale+ vec2(100.0*scale), t)));   
}

// Function 108
float fractalNoise(vec2 vl, out float mainWave) {
    
#if SHARP_MODE==1
    const float persistance = 2.4;
    float frequency = 2.2;
    const float freq_mul = 2.2;
    float amplitude = .4;
#else
    const float persistance = 3.0;
    float frequency = 2.3;
    const float freq_mul = 2.3;
    float amplitude = .7;
#endif
    
    float rez = 0.0;
    vec2 p = vl;
    
    float mainOfset = (iTime + 40.)/ 2.;
    
    vec2 waveDir = vec2(p.x+ mainOfset, p.y + mainOfset);
    float firstFront = amplitude + 
			        (valueNoiseSimple(p) * 2. - 1.);
    mainWave = firstFront * valueNoiseSimple(p + mainOfset);
    
    rez += mainWave;
    amplitude /= persistance;
    p *= unique_transform;
    p *= frequency;
    

    float timeOffset = iTime / 4.;

    
    for (int i = 1; i < OCTAVES; i++) {
        waveDir = p;
        waveDir.x += timeOffset;
        rez += amplitude * sin(valueNoiseSimple(waveDir * frequency) * .5 );
        amplitude /= persistance;
        p *= unique_transform;
        frequency *= freq_mul;

        timeOffset *= 1.025;

        timeOffset *= -1.;
    }

    return rez;
}

// Function 109
float FractalVoronoi(vec3 p)
{
	float n = 0.0;
	float f = 0.5, a = 0.5;
	mat2 m = mat2(0.8, 0.6, -0.6, 0.8);
	for (int i = 0; i < FBM_ITERATIONS; i++) {
		n += Voronoi(p * f) * a;
		f *= FBM_FREQUENCY_GAIN;
		a *= FBM_AMPLITUDE_GAIN;
		p.xy = m * p.xy;
	}
	return n;
}

// Function 110
vec3 fractal(vec2 p)
{   
    
    //--- basic constants ---//
    
    const vec2 O = vec2(1, 0);
    const vec2 I = vec2(0, -1);
    const float s3d2 = 0.5*sqrt(3.0);

    
    //--- moebius transform coefficients ---//
    
    const vec2 a = vec2(-0.5, -s3d2);
	const vec2 b = vec2(1.5, -s3d2);
	const vec2 c = vec2(0.5, s3d2);
	const vec2 d = vec2(1.5, -s3d2);

    
    //--- horizontal movement ---//
    
    vec2 z = p;
    
    if (iMouse.z > 0.5)
    {
        z.x += 40.0*(iMouse.x / iResolution.x - 0.5);
    }
    else
    {
        z.x += 0.2*iTime;
    }

    
    //--- unrolling cardioid (two stages) ---//
    
    z = cMob(z, a, b, c, d);				// stage 1: unrolling disc
    z = 0.25 * (O - cMul(z + O, z + O));	// stage 2: transform cardioid into disc
    

    //--- generating "before unrolling" chessboard ---//
    
    vec2 q = floor(25.3*z);					// checkboard size
    bool ch = mod(q.x + q.y, 2.0) == 0.0;

    
    //--- calculate fractal ---//
    
    float an = 0.0;
    
    p = z;
      
	for (float i = 0.0; i < 512.0; i++)
    {
        z = cMul(z, z) + p;					// Mandelbrot formulae

		if (dot(z, z) > 4.0)
        {
            float f = 1.0 - i/512.0;
            f *= f;

            return vec3(f, 0.6*f*(ch?0.0:1.0), 0);	// outside color
		}
        
        an += z.x;
	}

    
    //--- inside color ---//
    
    an += iTime;
    return vec3(0.5 + 0.5 * sin(4.0*an));
}

// Function 111
float FractalMeshShape(in vec3 p, in float scale, in float octaves) {
    float value = 0.0;
    float nscale = 1.0;
    float tscale = 0.0;

    for (float octave=0.0; octave < octaves; octave++) {
        value += MeshShape(p * pow(2.0, octave) * scale) * nscale;
        tscale += nscale;
        nscale *= 0.5;
    }

    return value / tscale;
}

// Function 112
float fractal(vec3 p) {
	
	float res = 0.0;
	float x = .7;
    
    p = tile(p);
    p.yz *= rot(T * .6);
    
    vec3 c = p;
	
    for (int i = 0; i < 10; ++i) {
        p = x * abs(p) / dot(p, p) - x;
        p.yz = csqr(p.yz);
        p = p.zxy;
        res += exp(-19. * abs(dot(p, c)));   
	}
    return res / 2.;
}

// Function 113
float fractalNoise(in vec3 vl) {
    const float persistance = 2.;
    const float persistanceA = 2.;
    float amplitude = .5;
    float rez = 0.0;
    float rez2 = 0.0;
    vec3 p = vl;
    
    for (int i = 0; i < OCTAVES / 2; i++) {
        rez += amplitude * valueNoiseSimple3D(p);
        amplitude /= persistanceA;
        p *= persistance;
    }
    
    float h = smoothstep(0., 1., vl.y*.5 + .5 );
    if (h > 0.01) { // small optimization, since Hermit polynom has low front at the start
        // God is in the details
        for (int i = OCTAVES / 2; i < OCTAVES; i++) {
            rez2 += amplitude * valueNoiseSimple3D(p);
            amplitude /= persistanceA;
            p *= persistance;
        }
        rez += mix(0., rez2, h);
    }
    
    return rez;
}

// Function 114
vec3 Julia3D(vec3 x, int seed){
    float m_N = floor(5.*Hash(seed^0xe305c492) ) + 2.;
    float m_AbsN = m_N;
    m_N *= (Hash(seed^0x67793120)>.5?-1.:1.);
    float m_Cn = (1. / m_N - 1.) / 2.;
	float z = x.z / m_AbsN;
	float r = pow(dot(x.xy,x.xy) + z*z, m_Cn);
	float tmp = r * dot(x.xy,x.xy);
	float ang = (atan(x.y,x.x) + pi*2. * floor(m_AbsN*Hash(seed^0x6aca43d3 ) ) ) / m_N;
	return vec3(tmp * cos(ang),tmp * sin(ang),r * z);
}

// Function 115
float FractalNoise(in vec2 xy)
{
	float w = .7;
	float f = 0.0;

	for (int i = 0; i < 3; i++)
	{
		f += Noise(xy) * w;
		w = w*0.6;
		xy = 2.0 * xy;
	}
	return f;
}

// Function 116
vec3 NewtonFractal(vec2 z, vec2 a){
    z = NewtonsMethod(z,a);
    //Roots of sin(z) are always z=Pi*k+0i
    return rainbow(z.x/3.14+z.y);
}

