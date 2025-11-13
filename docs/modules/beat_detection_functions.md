# beat_detection_functions

**Category:** audio
**Type:** extracted

## Dependencies
texture_sampling, lighting

## Tags
lighting, texturing, audio

## Code
```glsl
// Reusable Beat Detection Audio Functions
// Automatically extracted from audio visualization-related shaders

// Function 1
float getPulse(float i, float mult) {
    return (getWeight(squared(i)) * mult);
}

// Function 2
float get_pulse_width(float t){
    return mod(t/4.,.5);
}

// Function 3
float cubicPulse (float w, float x)
{
	x = abs (x);
	if (x > w)
		return 0.0;
	x /= w;
	return 1.0 - x * x * (3.0 - 2.0 * x);
}

// Function 4
float impulse (float k, float x)
{
	float h = k * x;
	return h * exp (1.0 - h);
}

// Function 5
float pulse(float time) {
    float A = -2.;
    float T = 1.5;
    time = mod(time, T);
    
    if (time < T/2.)
        return A*mix(1., 0.8, (T/2.-time)/(T/2.));
    else
        return A*mix(.8,1.1,(T-time)/(T/2.));
}

// Function 6
float cubicPulse( float c, float w, float x )
{
    x = abs(x - c);
    if( x>w ) return 0.0;
    x /= w;
    return 1.0 - x*x*(3.0-2.0*x);
}

// Function 7
float pulse( float morph, float pulse, float phase )
{
	float a, b;
    if( pulse < 0.5 )
        a = morph * pulse * 0.5;
    else
        a = morph * ( 1.0 - pulse ) / 2.0;
    if( phase < pulse )
    {
        if( phase < a )
        {
            b = phase / a - 1.0;
            return 1.0 - b * b;
        }
        if( phase < pulse - a )
            return 1.0;
        b = ( phase - pulse + a ) / a;
        return 1.0 - b * b;
    }
    if( phase < pulse + a )
    {
        b = ( phase - pulse ) / a - 1.0;
        return b * b - 1.0;
    }
    if( phase <= 1.0 - a )
        return -1.0;
    b = ( phase - 1.0 + a ) / a;
    return b * b - 1.0;
}

// Function 8
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

// Function 9
float Pulse125(float f, float x, float v)
{
    return (fract(f * x) < 0.125) ? v : 0.0;
}

// Function 10
float beat(float time, float big)
{
    float v = 0.0;
    float tb = time * tempo;
    tb = mod(tb, 2.0);
    tb = mod(tb, 5.0 / 4.0);
    tb /= tempo;
    v += sin(exp(tb * -1.0) * 800.0 + exp(tb * -100.0) * 200.0) * exp(max(0.1 - tb, 0.0) * -10.0) * exp(tb * -10.0) * 0.5 * big;
    
    tb = time * tempo;
    tb = mod(tb - 0.25, 2.0);
    tb /= tempo;
    v += sin(exp(tb * -1.0) * 800.0 + exp(tb * -100.0) * 200.0) * exp(max(0.1 - tb, 0.0) * -10.0) * exp(tb * -10.0) * 0.5 * big;
    
    tb = time * tempo;
    tb = mod(tb - 1.0, 2.0);
    tb /= tempo;
    v += (sin(exp(tb * -2.0) * 300.0) + hpns(tb, 0.0003) * 0.1) * exp(max(0.1 - tb, 0.0) * -10.0) * exp(tb * -5.0) * big;
    
    tb = time * tempo;
    tb = mod(tb, 2.0);
    tb = mod(tb, 2.5 / 4.0);
    tb = mod(tb - 1.0, 0.25);
    tb /= tempo;
    v += hpns(tb, 0.0002) * exp(tb * -20.0) * 0.3;
    
    tb = time * tempo;
    tb = mod(tb, 0.5);
    tb /= tempo;
    v += hpns(tb, 0.00002) * exp(tb * -5.0) * 0.3;
    
    
    //v = sqr(time, 440.0 * 0.1, 0.5);
    
    return v;
}

// Function 11
float pulse(float e0, float e1, float x)
{
   return step(e0, x) - step(e1, x);
}

// Function 12
float impulse( float k, float x ) {
    float h = k * x;
    return h * exp( 1.0 - h );
}

// Function 13
vec2 repulse(vec4 fish, vec2 target, float dist, float a)
{
    vec2 w = target-fish.xy;
    float x = length(w);
    if (x < EPSILON) return a*0.5*(hash31(float(id))-vec3(0.5)).xy;
    return w*a*(smoothstep(0.0, dist, x)-1.0)/x;
}

// Function 14
float simple_pulse(float x, float freq){
	return sign(mod(x-.5/freq,1./freq)-pulse_width/freq);
}

// Function 15
float bytebeat (float time) {
    // Get the current sample number according to the virtual sample rate
    int t = int(time * VSR);
    // Apply the bytebeat formula
    int byte = mySong(t);
    // Convert bytebeat back to float
    return float(byte % 256) / 128. - 1.;
}

// Function 16
float Pulse(float t, float noise) {
    float p = (sin(t * PulseSpeed)+1.)/2.;
    return (p * (PulseMax - PulseMin) + PulseMin) * noise;
}

// Function 17
float impulse( float k, float x )
{
    float h = k*x;
    return h*exp(1.0-h);
}

// Function 18
float cubic_pulse(float x, float c, float w) {
	x = abs(x-c);
    if (x>w) return 0.0;
    x/=w;
    return 1.0 - x*x*(3.0-2.0*x);
}

// Function 19
vec3 pulsemix(float pulse)
{
    float width = 0.8;
    return
    smoothstep(-width,  width, pulse) * positivepulse +
    smoothstep( width, -width, pulse) * negativepulse;
}

// Function 20
float pulseOsc(
    float amplitude,   // in range [0.0; 1.0]
    float frequency,   // > 0.0
    float time,        
    float phase_shift,
    float pulse_width)
{
    float phase = phase_shift + time * TAU * frequency;
    
    float sign_ = fract(phase / TAU) >= pulse_width ? 1.0 : -1.0;

    return sign_ * amplitude;
}

// Function 21
float beatF(int t) {
    int s = 
        //t*((t>>12|t>>8)&63&t>>4)
        //(t*(t>>5|t>>8))>>(t>>16&7)
        //t*((t>>9|t>>13)&25&t>>6)
        //(t>>6|t|t>>(t>>16&15))*10+((t>>11)&7)
        //(t|t>>9|t>>7)*t&(t>>11|t>>9)
        //t*5&(t>>7)|t*3&(t>>10)
        //(t&t%255)-(t*3&t>>13&t>>6)
        chaosTheory(t)
    ;
    return float(s&255)/255.;
}

// Function 22
float step_pulse(float x, float c, float w) {
	return step(c-w/2.0, x) - step(c+w/2.0, x);   
}

// Function 23
float bytebeat (float time) {
    int t = int(time * (7200. + 3.* time));
    int sig = ((t & t % (950 - t % 16)) << (5 + 3 * int(sin(time*12.))));
    return lim(2. * float(sig)/128. - 1.);
}

// Function 24
float pulse(float center, float width, float x)
{
    return smoothstep(center-width/2., center, x) * smoothstep(center+width/2., center, x);
}

// Function 25
float heartbeat(float x)
{
    float prebeat = -sinc((x - 0.4) * 40.0) * 0.6 * triIsolate((x - 0.4) * 1.0);
    float mainbeat = (sinc((x - 0.5) * 60.0)) * 1.2 * triIsolate((x - 0.5) * 0.7);
    float postbeat = sinc((x - 0.85) * 15.0) * 0.5 * triIsolate((x - 0.85) * 0.6);
    return (prebeat + mainbeat + postbeat) * triIsolate((x - 0.625) * 0.8); // width 1.25
}

// Function 26
float integral_pulse(float x, float freq){
	return (1.-2.*pulse_width)+integral_sawtooth(x,freq)-integral_sawtooth(x-pulse_width/freq,freq);
}

// Function 27
float pulse2(float x) {
	x = x / (1.0 + 1.5 * step(0.0, x));
	return 1.0 - smoothstep(0.0, 1.0, abs(x));	
}

// Function 28
float additive_pulse(float x, float freq){
    
   	x+=(.5*(1.-pulse_width))/freq;

	float sum=0.;

    for(int k=1;k<=max_n;k+=1){
		if(k<=n){
            float factor=sin(mod((pi*float(k)*pulse_width),2.*pi));
			sum+=1./(float(k))*factor*cos(mod(2.*pi*float(k)*freq*x,2.*pi));
		}
	}

    return 1.-2.*(2./pi*sum+pulse_width);
}

// Function 29
float pulse(float t) {
#if   PULSE==1
	return (mod(t,1.)<.1) ? 1.: 0.;      // square signal
#elif PULSE==2
	return pow(.5+.5*cos(6.283*t),20.);  // smoothed signal
#endif
}

// Function 30
float expImpulse(float x, float k) {
    float h = k*x;
    return h*exp(1.0-h);
}

// Function 31
float PeriodicPulse(float x, float p) {
    // pulses from 0 to 1 with a period of 2 pi
    // increasing p makes the pulse sharper
	return pow((cos(x+sin(x))+1.)/2., p);
}

// Function 32
vec2 beat(float q)
{
    return (
        spec(q, 0.0).xy
        +spec(q, 0.05).xy/1.5
        +spec(q, 0.1).xy/4.0
        +spec(q, 0.2).xy/8.0
        //+spec(q, 0.4).xy/16.0
        )/1.8;
}

// Function 33
float pulse(float a)
{
	return sin(saturate(a)*3.14159);
}

// Function 34
vec2 last_beat(float t){
    t=max(0.,t);
    float t0=t;
    
    t*=time_fac;
    
    float second=floor(t);
    
    float speed=min(second+1.,float(max_speed))*float(speed_fac);
    
    float frac=fract(t);
    
    vec2 ret=vec2(0);
    
    //point in time of last beat
    ret.x=(second+floor(frac*3.*speed)/(3.*speed))/time_fac;
    //"position" of last beat
    ret.y=mod(floor(frac*3.*speed),3.);
    
    return ret;
}

// Function 35
float beat()
{
	float pos = mod(iTime * 32., 41.);
    int i = int(pos);
	pos = pos - float(i);
	
	float v1 = st[i];
	float v2 = st[i+1];
	return 1. + (mix(v1, v2, pos) / 100.) * .2;
}

// Function 36
float bytebeat (float time) {
    // Get the current sample number according to the virtual sample rate
    int t = int(time * VSR);
    
    // Apply the bytebeat formula
    int byte = mySong(t);
    
    //byte = t*(1&t>>11);

    // Convert bytebeat back to float
    return mix(-1., 1., float(byte % 256) / 255.);
}

// Function 37
float getBeat()
{
 	float sum = 0.0;
    for (float i = 0.0; i < 16.0; i++)
    {
     	sum += texture(iChannel0, vec2(i * 0.001 + 0.0, 0.0)).r;   
    }
    return smoothstep(0.6, 0.9, pow(sum * 0.06, 2.0));
}

// Function 38
float impulse2 (float k0, float k1, float x)
{
	float k = k0;
	if (x > 1.0/k0)
	{
		x += 1.0/k1 - 1.0/k0;
		k = k1;
	}
	float h = k * x;
	return h * exp (1.0 - h);
}

// Function 39
float pulse1(float x) {
	x = x / (1.0 + 2.5 * step(0.0, x));
	x = clamp(abs(x), 0.0, 1.0);
	return 1.0 - x*x*x*(x*(x*6.0 - 15.0) + 10.0);	
}

// Function 40
float iii_impulseCube(in float gtime, in vec3 p, in float celld, in float celll, out int m)
{
  vec3 bp = p;

  float cs = III_BOXSIZE*0.8;

  float cd = max(length(p.xy) - cs*0.3, sdSphere(bp, 1.2*III_BOXSIZE));
  float hbd = min(max(sdSoftBox(bp, III_BOXSIZE), -sdCross(bp/cs)*cs), cd);
  float bbd = sdBox(bp, vec3(cs));

  vec3 rbp = toSpherical(bp);
  rbp.y += III_TIME_IN_PERIOD*0.5;
  bp = toRectangular(rbp);

  vec3 cell = mod3(bp, vec3(cs*celld));

  float id = sdSoftBox(bp, cs*celll);

  float mbd = max(bbd, id);

  float d = unionRound(hbd, mbd, III_BOXSIZE*0.15);

  if (abs(hbd - d) < 0.01)
  {
    m = 2;
  }
  else
  {
    m = 1;
  }


  return d;
}

// Function 41
vec4 makeBeat(vec2 coord)
{
    float x = -floor(T)-0.6-0.8/N*0.5*(1.+floor(noise(vec3(coord/M,T))+0.5)*2.+2.*coord.x);
    float l = 1.0;
    
    return vec4(
        x, // position
        l, // length
        0,
        1  // visible
    );
}

// Function 42
float beat (float value, float intensity, float frequency) 
{
    float v = atan(sin(value * 3.14 * frequency) * intensity);
    return (v + 3.14 / 2.) / 3.14;
}

// Function 43
bool getLightPulse()
{
    return texture(iChannel2, vec2(2.5, 0.5) / iResolution.xy).y < 0.5;
}

// Function 44
float periodic_pulse(float s, float x)
{
    return exp(s * (-0.5 - 0.5 * cos(x))) - exp(s * (-0.5 + 0.5 * cos(x)));
}

// Function 45
float cubicpulse(float c, float w, float x) {
    x = abs(x - c);
    return (x >= w) ? 0.0 : 1.0 - cubic(x/w);
}

// Function 46
float polyImpulse( float k, float n, float x )
{
    return (n/(n-1.0))*pow((n-1.0)*k,1.0/n)*x/(1.0+k*pow(x,n));
}

// Function 47
float beat2(float time, float big)
{
    float tb, v = 0.0;
    
    tb = time * tempo;
    tb = mod(tb, 2.0);
    tb = mod(tb, 5.0 / 4.0);
    tb /= tempo;
    float kick = sin(exp(tb * -1.0) * 400.0 + exp(tb * -100.0) * 200.0) * exp(max(0.1 - tb, 0.0) * -10.0) * exp(tb * -10.0);
    kick = smoothstep(-0.2, 0.2, kick) * 2.0 - 1.0;
    v = kick * 0.3;
    
    tb = time * tempo;
    tb = mod(tb - 0.5, 1.0);
    tb /= tempo;
    v += (hpns(exp(-tb) * 4.0, 0.0002) * exp(max(tb - 0.1, 0.0) * -10.0) * 0.5 + sin(sin(tb * 100.0) * 5.0 + tb * 2000.0) * exp(max(tb - 0.1, 0.0) * -10.0) * 0.4) * 0.6;
    
    /*tb = time * tempo;
    tb = mod(tb - 0.5, 1.0);
    //tb = mod(tb, 5.0 / 4.0);
    tb /= tempo;
    v += sin(exp(tb * -1.0) * 800.0 + exp(tb * -100.0) * 200.0) * exp(max(0.1 - tb, 0.0) * -10.0) * exp(tb * -10.0);*/
    
    tb = time * tempo + 0.25;
    tb = mod(tb, 2.0);
    tb = mod(tb, 2.5 / 4.0);
    tb = mod(tb - 1.0, 0.25);
    tb /= tempo;
    v += hpns(tb * 4.0, 0.0002) * exp(tb * -25.0) * 0.25;
    
    tb = time * tempo;
    tb = mod(tb, 0.5);
    tb /= tempo;
    v += (hpns(tb * 2.0, 0.00002) + hpns(tb * 100.0, 0.002) * 0.3) * exp(tb * -4.0) * 0.2;
    
    tb = time * tempo;
    tb = mod(tb - 0.25, 0.5);
    tb /= tempo;
    v += (hpns(tb * 9.0, 0.0002)) * exp(tb * -8.0) * 0.3;
    
    //v = sqr(time, 440.0 * 0.1, 0.5);
    
    return v;
}

// Function 48
float impulse (float x, float c, float w)
{
    float d = abs(x-c);
    if ( d > w ) return 0.0;
    return 1.0 - smoothstep(0.0,1.0, d/w);
}


```