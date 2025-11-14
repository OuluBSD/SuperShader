// Reusable Graphics Operations UI/2D Functions
// Automatically extracted from UI/2D graphics-related shaders

// Function 1
float cloudGradient( float norY ) {
    return linearstep( 0., .05, norY ) - linearstep( .8, 1.2, norY);
}

// Function 2
float fill(float d, float s, float i) { return abs(smoothstep(s,.0,d) - i); }

// Function 3
float fill(float d, bool f) { return abs(ssaa(d) - float(f)); }

// Function 4
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

// Function 5
vec3 gradient(float f)
{
    vec3 cols[3];
    
    cols[0] = vec3(0.169,0.086,0.816);
    cols[1] = vec3(0.835,0.216,0.843);
    cols[2] = vec3(1.,1.,1.);
    
    float cnt = 2.;
    float cur = f*cnt;
    float curIdx = floor(cur);
    return mix(cols[int(curIdx)], cols[int(min(curIdx+1., cnt))], sat(fract(cur)));
}

// Function 6
float fill(float d) {
    return smoothstep(0., .003, d);
}

// Function 7
vec2 Gradient(vec2 p) {
    return normalize(vec2(SDF(p+eps.xy).D-SDF(p-eps.xy).D,
                          SDF(p+eps.yx).D-SDF(p-eps.yx).D));
}

// Function 8
vec4 gradient(float i)
{
	i = clamp(i, 0.0, 1.0) * 2.0;
	if (i < 1.0) {
		return (1.0 - i) * gradA + i * gradB;
	} else {
		i -= 1.0;
		return (1.0 - i) * gradB + i * gradC;
	}
}

// Function 9
vec3 humanizeBrushStrokeDonut(vec2 uvLine, float radius_, bool clockwise, float lineLength
){vec2 humanizedUVLine = uvLine
 ;float twistAmt=.24//offset circle along its path for a twisting effect.
 ;float linePosY = humanizedUVLine.y / lineLength// 0 to 1 scale
 ;humanizedUVLine.x += linePosY * twistAmt
 ;float humanizedRadius = radius_ // perturb radius / x
 ;float res = min(iResolution.y,iResolution.x)
 ;humanizedRadius += (noise01(uvLine * 1.)-0.5) * 0.04
 ;humanizedRadius += sin(uvLine.y * 3.) * 0.019// smooth lp wave
 ;humanizedUVLine.x += sin(uvLine.x * 30.) * 0.02// more messin
 ;humanizedUVLine.x += (noise01(uvLine * 5.)-0.5) * 0.005// a sort of random waviness like individual strands are moving around
 ;//humanizedUVLine.x += (noise01(uvLine * res * 0.18)-0.5) * 0.0035;// HP random noise makes it look less scientific
 ; return vec3(humanizedUVLine, humanizedRadius);}

// Function 10
vec3 gradientNoised(vec2 pos, vec2 scale, float seed) 
{
    // gradient noise with derivatives based on Inigo Quilez
    pos *= scale;
    vec4 i = floor(pos).xyxy + vec2(0.0, 1.0).xxyy;
    vec4 f = (pos.xyxy - i.xyxy) - vec2(0.0, 1.0).xxyy;
    i = mod(i, scale.xyxy) + seed;
    
    vec4 hashX, hashY;
    smultiHash2D(i, hashX, hashY);
    vec2 a = vec2(hashX.x, hashY.x);
    vec2 b = vec2(hashX.y, hashY.y);
    vec2 c = vec2(hashX.z, hashY.z);
    vec2 d = vec2(hashX.w, hashY.w);
    
    vec4 gradients = hashX * f.xzxz + hashY * f.yyww;

    vec4 udu = noiseInterpolateDu(f.xy);
    vec2 u = udu.xy;
    vec2 g = mix(gradients.xz, gradients.yw, u.x);
    
    vec2 dxdy = a + u.x * (b - a) + u.y * (c - a) + u.x * u.y * (a - b - c + d);
    dxdy += udu.zw * (u.yx * (gradients.x - gradients.y - gradients.z + gradients.w) + gradients.yz - gradients.x);
    return vec3(mix(g.x, g.y, u.y) * 1.4142135623730950, dxdy);
}

// Function 11
vec3 gradientNoised(vec2 pos, vec2 scale, mat2 transform, float seed) 
{
    // gradient noise with derivatives based on Inigo Quilez
    pos *= scale;
    vec4 i = floor(pos).xyxy + vec2(0.0, 1.0).xxyy;
    vec4 f = (pos.xyxy - i.xyxy) - vec2(0.0, 1.0).xxyy;
    i = mod(i, scale.xyxy) + seed;
    
    vec4 hashX, hashY;
    smultiHash2D(i, hashX, hashY);

    // transform gradients
    vec4 m = vec4(transform);
    vec4 rh = vec4(hashX.x, hashY.x, hashX.y, hashY.y);
    rh = rh.xxzz * m.xyxy + rh.yyww * m.zwzw;
    hashX.xy = rh.xz;
    hashY.xy = rh.yw;

    rh = vec4(hashX.z, hashY.z, hashX.w, hashY.w);
    rh = rh.xxzz * m.xyxy + rh.yyww * m.zwzw;
    hashX.zw = rh.xz;
    hashY.zw = rh.yw;
    
    vec2 a = vec2(hashX.x, hashY.x);
    vec2 b = vec2(hashX.y, hashY.y);
    vec2 c = vec2(hashX.z, hashY.z);
    vec2 d = vec2(hashX.w, hashY.w);
    
    vec4 gradients = hashX * f.xzxz + hashY * f.yyww;

    vec4 udu = noiseInterpolateDu(f.xy);
    vec2 u = udu.xy;
    vec2 g = mix(gradients.xz, gradients.yw, u.x);
    
    vec2 dxdy = a + u.x * (b - a) + u.y * (c - a) + u.x * u.y * (a - b - c + d);
    dxdy += udu.zw * (u.yx * (gradients.x - gradients.y - gradients.z + gradients.w) + gradients.yz - gradients.x);
    return vec3(mix(g.x, g.y, u.y) * 1.4142135623730950, dxdy);
}

// Function 12
float InterleavedGradientNoise(vec2 pixel, int frame) 
{
    pixel += (float(frame) * 5.588238f);
    return fract(52.9829189f * fract(0.06711056f*float(pixel.x) + 0.00583715f*float(pixel.y)));  
}

// Function 13
vec2 gradient_circle(vec2 uv){
	return vec2(2.*uv.x,2.*uv.y);
}

// Function 14
vec2 GradientB(vec2 uv, vec2 d, vec4 selector, int level){
    vec4 dX = 0.5*BlurB(uv + vec2(1.,0.)*d, level) - 0.5*BlurB(uv - vec2(1.,0.)*d, level);
    vec4 dY = 0.5*BlurB(uv + vec2(0.,1.)*d, level) - 0.5*BlurB(uv - vec2(0.,1.)*d, level);
    return vec2( dot(dX, selector), dot(dY, selector) );
}

// Function 15
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

// Function 16
vec3 calcGradient( in vec3 pos )
{
    const vec3 v1 = vec3(1.0,0.0,0.0);
    const vec3 v2 = vec3(0.0,1.0,0.0);
    const vec3 v3 = vec3(0.0,0.0,1.0);
	return (vec3(scene(pos + v1*eps),scene(pos + v2*eps),scene(pos + v3*eps))
           -vec3(scene(pos - v1*eps),scene(pos - v2*eps),scene(pos - v3*eps)))/(2.0*eps);
}

// Function 17
float circleFill(vec2 pos, float radius)
{
    return clamp(((1.0-(length(pos)-radius))-0.99)*100.0, 0.0, 1.0);   
}

// Function 18
vec2 gradient3(vec2 uv){
    return vec2(implicit3_x(uv),implicit3_y(uv));
}

// Function 19
float cloudGradient(float h)
{
    return smoothstep(0., .05, h) * smoothstep(1.25, .5, h);
}

// Function 20
float getgradientnoise( vec2 p ) {
	vec2 fp = floor(p);
	vec2 pfract = p-fp;

	vec2 nw = vec2(fp.x,     fp.y+1.0);
	vec2 ne = vec2(fp.x+1.0, fp.y+1.0);
	vec2 sw = vec2(fp.x,     fp.y);
	vec2 se = vec2(fp.x+1.0, fp.y);

	vec2 dnw = vec2(pfract.x,     pfract.y-1.0);
	vec2 dne = vec2(pfract.x-1.0, pfract.y-1.0);
	vec2 dsw = vec2(pfract.x,     pfract.y);
	vec2 dse = vec2(pfract.x-1.0, pfract.y);

	float vnw = dot(hash22(nw), dnw);
	float vne = dot(hash22(ne), dne);
	float vsw = dot(hash22(sw), dsw);
	float vse = dot(hash22(se), dse);

	//pfract *= pfract*pfract*(pfract*(pfract*6.0-15.0)+10.0);
	float v = mix( mix(vsw, vse, pfract.x), mix(vnw, vne, pfract.x), pfract.y);
	
	return v;
}

// Function 21
float get_gradient_eps() {
    return (1.0 / min_uniform_scale()) * AAINV;
}

// Function 22
vec3 gradient(vec3 position) {

	return vec3(f(position + vec3(delta, 0.0, 0.0)) - f(position - vec3(delta, 0.0, 0.0)),f(position + vec3(0.0,delta, 0.0)) - f(position - vec3(0.0, delta, 0.0)),f(position + vec3(0.0, 0.0, delta)) - f(position - vec3(0.0, 0.0, delta)));

}

// Function 23
float interleavedGradientNoise(vec2 n) {
    float f = 0.06711056 * n.x + 0.00583715 * n.y;
    return fract(52.9829189 * fract(f));
}

// Function 24
vec4 applyBrushFill(vec2 fragCoord, vec4 simulationState, vec4 toolState) {
    if (simulationState.x != 0.0) {
        return simulationState;
    }
    
    float sandType = toolState.y + 1.0;
    
    // Triangle wave /\/\/\/\/\ over time.
    // TODO: mix some spatial noise into this?
    // TODO: pull into Common so Image can use show the current brush in the UI
    float colorVariation = abs(mod(iTime, 2.0) - 1.0);
    
    return vec4(sandType, colorVariation, 0.0, 1.0);
}

// Function 25
vec3 grayscaleGradient(float t) {
	return vec3(t);
}

// Function 26
float gradient(vec2 uv) {
 	return (1.0 - uv.y * uv.y * cloudHeight);   
}

// Function 27
define fillbox(rx,ry,v) {V=abs(U-P.xy-vec2(0,5)*S); if(V.x<rx*S && V.y<ry*S) O=vec4(v);}

// Function 28
float fill(float x, float size) 
{
    return 1. - smoothstep(size,size*1.05, x);
}

// Function 29
void fill_preserve() {
    write_color(_stack.source, calc_aa_blur(_stack.shape.x));
    if (_stack.has_clip) {
	    write_color(_stack.source, calc_aa_blur(_stack.clip.x));        
    }
}

// Function 30
void set_source_linear_gradient(vec3 color0, vec3 color1, vec2 p0, vec2 p1) {
    set_source_linear_gradient(vec4(color0, 1.0), vec4(color1, 1.0), p0, p1);
}

// Function 31
vec4 _gradient4d(uint hash)
{
	vec4 g = vec4(uvec4(hash) & uvec4(0x80000, 0x40000, 0x20000, 0x10000));
	return g * (1.0 / vec4(0x40000, 0x20000, 0x10000, 0x8000)) - 1.0;
}

// Function 32
vec3 Gradient(vec3 p,float d){vec2 e=vec2(.001,0);p*=99.;
 return (vec3(df(p+e.xyy).w,df(p+e.yxy).w,df(p+e.yyx).w)*99.-d)/e.x;}

// Function 33
vec2 sample_biquadratic_gradient(sampler2D channel, vec2 res, vec2 uv) {
    vec2 q = fract(uv * res);
    vec2 c = (q*(q - 1.0) + 0.5) / res;
    vec2 w0 = uv - c;
    vec2 w1 = uv + c;
    vec2 cc = 0.5 / res;
    vec2 ww0 = uv - cc;
    vec2 ww1 = uv + cc;
    float nx0 = texture(channel, vec2(ww1.x, w0.y)).r - texture(channel, vec2(ww0.x, w0.y)).r;
    float nx1 = texture(channel, vec2(ww1.x, w1.y)).r - texture(channel, vec2(ww0.x, w1.y)).r;
    
    float ny0 = texture(channel, vec2(w0.x, ww1.y)).r - texture(channel, vec2(w0.x, ww0.y)).r;
    float ny1 = texture(channel, vec2(w1.x, ww1.y)).r - texture(channel, vec2(w1.x, ww0.y)).r;
    
	return vec2(nx0 + nx1, ny0 + ny1) / 2.0;
}

// Function 34
vec2 GetGradient(vec2 intPos, float t) {
    
    // Uncomment for calculated rand
    //float rand = fract(sin(dot(intPos, vec2(12.9898, 78.233))) * 43758.5453);;
    
    // Texture-based rand (a bit faster on my GPU)
    float rand = hash(intPos / 64.0).r;
    
    // Rotate gradient: random starting rotation, random rotation rate
    float angle = 6.283185 * rand + 4.0 * t * rand;
    return vec2(cos(angle), sin(angle));
}

// Function 35
float get_gradient_func(in sampler2D s)
{
    return texelFetch(s, CTRL_GRADIENT_FUNC, 0).w;
}

// Function 36
vec2 stroke_shape() {
    return abs(_stack.shape) - _stack.line_width/_stack.scale;
}

// Function 37
vec3 colorBrushStrokeLine(vec2 uv, vec3 inpColor, vec4 brushColor, vec2 p1_, vec2 p2_, float lineWidth)
{
    // flatten the line to be axis-aligned.
    float lineAngle = pi-atan(p1_.x - p2_.x, p1_.y - p2_.y);
    mat2 rotMat = rot2D(lineAngle);

    float lineLength = distance(p2_, p1_);
    // make an axis-aligned line from this line.
    vec2 tl = (p1_ * rotMat);// top left
    vec2 br = tl + vec2(0,lineLength);// bottom right
    vec2 uvLine = uv * rotMat;

    // make line slightly narrower at end.
    lineWidth *= mix(1., .9, smoothstep(tl.y,br.y,uvLine.y));
    
    // wobble it around, humanize
    float res = min(iResolution.y,iResolution.x);
    uvLine.x += (noise01(uvLine * 1.)-0.5) * 0.02;
    uvLine.x += cos(uvLine.y * 3.) * 0.009;// smooth lp wave
    uvLine.x += (noise01(uvLine * 5.)-0.5) * 0.005;// a sort of random waviness like individual strands are moving around
//    uvLine.x += (noise01(uvLine * res * 0.18)-0.5) * 0.0035;// HP random noise makes it look less scientific

    // calc distance to geometry. actually just do a straight line, then we will round it out to create the line width.
    float d = sdAxisAlignedRect(uvLine, tl, br) - lineWidth / 2.;
    uvLine = tl - uvLine;
    
    vec2 lineSize = vec2(lineWidth, lineLength);
    
    vec3 ret = colorBrushStroke(vec2(uvLine.x, -uvLine.y), uv, lineSize,
                                d, inpColor, brushColor);
    return ret;
}

// Function 38
vec2 gradient2(vec2 uv){
	return vec2(implicit2_x(uv),implicit2_y(uv));
}

// Function 39
float stroke(float x, float s, float w) { // 04
    float d = step(s, x + w / 2.) - step(s, x - w / 2.);
    return clamp(d, 0., 1.);
}

// Function 40
vec3 Gradient(vec3 intersection, float distance)
{
    vec2 epsilon = vec2(0.01, 0.0);
    return normalize(vec3(fField(intersection + epsilon.xyy),
    fField(intersection + epsilon.yxy),
    fField(intersection + epsilon.yyx))
        - distance);
}

// Function 41
float stroke(float x, float s, float w) {
	float d = step(s,x+w*.5) - step(s, x-w*.5);
    return clamp(d, 0., 1.);
}

// Function 42
void set_source_radial_gradient(vec3 color0, vec3 color1, vec2 p, float r) {
    set_source_radial_gradient(vec4(color0, 1.0), vec4(color1, 1.0), p, r);
}

// Function 43
float fill(float x, float s){
    return 1.-smoothstep(s, s+smoothing, x);
}

// Function 44
void debug_gradient() {
    _color = mix(_color, 
        hsl(_stack.shape.x * 6.0, 
            1.0, (_stack.shape.x>=0.0)?0.5:0.3), 
        0.5);
}

// Function 45
v2 strokeLine(v1 u,v2 r,v3 M,v2 c, v3 b, v3 m, v0 w
){v0 lineAngle=atan(m.x-m.z,m.y-m.w)//axis-align
 ;m1 rotMat =rot2D(lineAngle)
 ;v0 W=length(m.xy-m.zw)    // make an axis-aligned line from this line.
 ;v1 T=m.xy*rotMat// top left
 ;v1 B=T+v1(0,W)// bottom right
 ;v1 l=u*rotMat
 ;l.x+=(noise01(l*1.)-.5)*.02
 ;l.x+=cos(l.y*3.)*.009//lp wave
 ;l.x+=(noise01(l*5.)-.5)*.005;//random waviness like individual strands are moving around
 ;l.x+=(noise01(l*min(r.y,r.x)*.18)-.5)*.0035;// HP random noise makes it look less scientific
 ;v0 d=sdAxisAlignedRect(l,T,B)-w/2.
 ;return colorBrushStroke((T-l)*v1(1,-1),r,M,u,W,d,c,b);}

// Function 46
void fill(inout vec3 bg, in vec2 st, in float a){
    if (a < 1.0){
        bg = hsb2rgb(vec3(getRadius(st)*0.2+sin(0.676+iTime)*.2,0.2,clamp(1.0 - fbm(st*10.)*.5,0.5,0.7)));
    }
}

// Function 47
vec2 GradientA(vec2 uv, vec2 d, vec4 selector, int level){
	vec4 dX = 0.5*BlurA(uv + vec2(1.,0.)*d, level) - 0.5*BlurA(uv - vec2(1.,0.)*d, level);
	vec4 dY = 0.5*BlurA(uv + vec2(0.,1.)*d, level) - 0.5*BlurA(uv - vec2(0.,1.)*d, level);
	return vec2( dot(dX, selector), dot(dY, selector) );
}

// Function 48
bool fillellipse(float2 p, float2 r)
{
    p=abs(p);
    r=abs(r)+0.01;
    float m=max(r.x,r.y);
    r/=m;
    p.x/=r.x;
    p.y/=r.y;

    if(abs(sqrt(p.x*p.x+p.y*p.y))<m)
       return true;
    return false;
}

// Function 49
vec3 Gradient(vec3 p, float Time) {
    return normalize(vec3(
        DF(p+eps.xyy,Time).d-DF(p-eps.xyy,Time).d,
        DF(p+eps.yxy,Time).d-DF(p-eps.yxy,Time).d,
        DF(p+eps.yyx,Time).d-DF(p-eps.yyx,Time).d
    ));
}

// Function 50
vec3 electricGradient(float t) {
	return clamp( vec3(t * 8.0 - 6.3, square(smoothstep(0.6, 0.9, t)), pow(t, 3.0) * 1.7), 0.0, 1.0);	
}

// Function 51
float gradient( vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash2( i + vec2(0,0) ), f - vec2(0,0) ), 
                     dot( hash2( i + vec2(1,0) ), f - vec2(1,0) ), u.x),
                mix( dot( hash2( i + vec2(0,1) ), f - vec2(0,1) ), 
                     dot( hash2( i + vec2(1,1) ), f - vec2(1,1) ), u.x), u.y);
}

// Function 52
float fill(float x, float size) { // 09
    return 1. - sstep(size, x);
}

// Function 53
vec4 filled(float distance, float linewidth, float antialias, vec4 fill)
{
    vec4 frag_color;
    float t = linewidth/2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);

    // Within linestroke
    if( border_distance < 0.0 )
        frag_color = fill;
    // Within shape
    else if( signed_distance < 0.0 )
        frag_color = fill;
    else
        // Outside shape
        if( border_distance > (linewidth/2.0 + antialias) )
            discard;
        else // Line stroke exterior border
            frag_color = vec4(fill.rgb*alpha, 1.0);

    return frag_color;
}

// Function 54
float stroke(float d, float size, float width) {
	return smoothstep(pixel_width,0.0,abs(d-size)-width/2.);
}

// Function 55
vec4 ComputeWaveGradientRGB(float t, vec4 bias, vec4 scale, vec4 freq, vec4 phase)
{
	vec4 rgb = bias + scale * cos(PI * 2.0 * (freq * t + phase));
	return vec4(clamp(rgb.xyz,0.0,1.0), 1.0);
}

// Function 56
float gradientNoise(vec2 v)
{
    return fract(52.9829189 * fract(dot(v, vec2(0.06711056, 0.00583715))));
}

// Function 57
vec3 color_gradient(float t) {
    t *= 0.75;
	vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.0);
    vec3 d = vec3(0.0, 0.33, 0.67);
    return a + b * cos(2.0*PI*(c*t + d));
}

// Function 58
void fill_preserve() {
    write_color(_stack.source, calc_aa_blur(_stack.shape.x));
    if (_stack.has_clip) {
	    write_color(_stack.source, calc_aa_blur(_stack.clip.x));
    }
}

// Function 59
void fill(inout float[9] k){for( int i=0;i<8;i++) { k[i] = 0.;} }

// Function 60
float fill(in float d, in float softness, in float offset){
    return clamp((offset +softness*.5 - d)/softness, 0.0, 1.0);
}

// Function 61
vec2 gradient(in vec2 uv) {
    vec3 offset = vec3(-1.0, 0.0, 1.0);
    vec2 invres = 1.0/iResolution.xy;
    float dx0 = height((uv+offset.xy)*invres);
    float dxf = height((uv+offset.zy)*invres);
    float dy0 = height((uv+offset.yx)*invres);
    float dyf = height((uv+offset.yz)*invres);
    return vec2(dxf - dx0, dyf - dy0);
}

// Function 62
vec3 neonGradient(float t) {
	return clamp(vec3(t * 1.3 + 0.1, square(abs(0.43 - t) * 1.7), (1.0 - t) * 1.7), 0.0, 1.0);
}

// Function 63
float ovalGradient(vec2 st, float radius, float xPos) {
    return smoothstep(radius- .1, radius + .9, 1. -length(st - .5));
}

// Function 64
float gradient( vec2 uv, vec2 interval )
{
	float t = remap( interval.x, interval.y, uv.x );
    return t;
}

// Function 65
void fill_preserve() {
    write_color(_stack.source, calc_aa_blur(_stack.shape.x));
}

// Function 66
float fillMask(float distanceChange, float dist) {
    return smoothstep(distanceChange, -distanceChange, dist);
}

// Function 67
vec2 gradient (in vec2 p)
{
    //return vec2 (dFdx (graph (p)), dFdy (graph (p))) / fwidth (graph (p));
    vec2 e = vec2 (.001, .0);
    return vec2 (graph (p - e.xy) - graph (p + e.xy),
                 graph (p - e.yx) - graph (p + e.yx)) / (.75*e.x);
}

// Function 68
float FillLineDash(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded) {
    float df = LineDistField(uv, pA, pB, vec2(thick), rounded, 1.0);
    return saturate(df / abs(dFdy(uv).y));
}

// Function 69
vec3 gradient(in float t) {
    return FIELD_COL * (0.4 + 0.6 * smoothstep(0., 1., t));
}

// Function 70
vec2 gradient4(vec2 uv){
    return vec2(implicit4_x(uv),implicit4_y(uv));
}

// Function 71
vec3 gradientNormal(vec3 p) {
    return normalize(vec3(
        map(p + vec3(GRADIENT_DELTA, 0, 0)).y - map(p - vec3(GRADIENT_DELTA, 0, 0)).y,
        map(p + vec3(0, GRADIENT_DELTA, 0)).y - map(p - vec3(0, GRADIENT_DELTA, 0)).y,
        map(p + vec3(0, 0, GRADIENT_DELTA)).y - map(p - vec3(0, 0, GRADIENT_DELTA)).y));
}

// Function 72
vec3 rainbowGradient(float t) {
	vec3 c = 1.0 - pow(abs(vec3(t) - vec3(0.65, 0.5, 0.2)) * vec3(3.0, 3.0, 5.0), vec3(1.5, 1.3, 1.7));
	c.r = max((0.15 - square(abs(t - 0.04) * 5.0)), c.r);
	c.g = (t < 0.5) ? smoothstep(0.04, 0.45, t) : c.g;
	return clamp(c, 0.0, 1.0);
}

// Function 73
void fillExample(){
    for (int i = 0; i < 16; i++)
        example[int(float(i)/4.0)][int(mod(float(i),4.0))] = sin(3.14 * float(i) / 16.0);
	}

// Function 74
vec3 gradient(vec3 pos) {
	const vec3 dx = vec3(grad_step, 0.0, 0.0);
	const vec3 dy = vec3(0.0, grad_step, 0.0);
	const vec3 dz = vec3(0.0, 0.0, grad_step);
	return normalize(
		vec3(dist_field(pos + dx) - dist_field(pos - dx),
             dist_field(pos + dy) - dist_field(pos - dy),
             dist_field(pos + dz) - dist_field(pos - dz)));
}

// Function 75
float sharpFill(in float d){return clamp(.5-d, 0.0, 1.0);}

// Function 76
float stroke(float x, float s, float w){
    float d = smoothstep(s, s+smoothing, x+w*.5) - smoothstep(s, s+smoothing, x-w*.5);
    return clamp(d, 0., 1.);
}

// Function 77
vec2 GradientA(vec2 uv, vec2 d, vec4 selector, int level, sampler2D bufA, sampler2D bufD){
    vec4 dX = BlurA(uv + vec2(1.,0.)*d, level, bufA, bufD) - BlurA(uv - vec2(1.,0.)*d, level, bufA, bufD);
    vec4 dY = BlurA(uv + vec2(0.,1.)*d, level, bufA, bufD) - BlurA(uv - vec2(0.,1.)*d, level, bufA, bufD);
    return vec2( dot(dX, selector), dot(dY, selector) );
}

// Function 78
vec3 _gradient3d(uint hash)
{
	vec3 g = vec3(uvec3(hash) & uvec3(0x80000, 0x40000, 0x20000));
	return g * (1.0 / vec3(0x40000, 0x20000, 0x10000)) - 1.0;
}

// Function 79
vec3 ProjDiskImplicit2_Gradient(vec3 rp, vec3 rd, float rr)
{
    vec2 t = (rd.yz * rp.xx) - (rd.xx * rp.yz);
    
    return 2.0 * vec3(-(rr * rd.x) - (rp.y * t.x) - (rp.z * t.y), rp.xx * t);
}

// Function 80
vec3 heatmapGradient(in float t) {
    return clamp((pow(t, 1.5) * .8 + .2) * vec3(smoothstep(0., .35, t) + t * .5, smoothstep(.5, 1., t), max(1. - t * 1.7, t * 7. - 6.)), 0., 1.);
}

// Function 81
float FillLinePix(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded) {
    float scale = abs(dFdy(uv).y);
    thick = (thick * 0.5 - 0.5) * scale;
    float df = LineDistField(uv, pA, pB, vec2(thick), rounded, 0.0);
    return saturate(df / scale);
}

// Function 82
float calcGradientDirAbs( in vec3 pos, in vec3 dir )
{
    return abs(scene(pos + dir*eps) - scene(pos - dir*eps))/(2.0*eps);
}

// Function 83
float gradientNoise(in vec2 uv)
{
    const vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(uv, magic.xy)));
}

// Function 84
float stroke(in float d, in float softness, in float offset, in float width){ 
   d = abs(d-offset);
   return clamp((width/2.0 +softness*.5 - d)/softness, 0.0, 1.0);
}

// Function 85
vec3 drawCircleFill(vec2 resolution, vec2 st, vec2 center, float radius, float blur)
{
    vec2 aspect = resolution / min(resolution.x, resolution.y);
    return vec3(smoothstep(radius - blur, radius, length((st - center) * aspect)));
}

// Function 86
void DoRefill()
{
    vec2 id = floor(gFragCoord);
    if(gFragCoord.y > yCells) 
    {
        id -= vec2(0., yCells);
    }
    
    float numCells = 0.;
    float dy = 0.;
    for(float y=0.; y<yCells-0.5;++y)
    {
        vec2 currId = vec2(id.x, id.y -y);
        if(currId.y < -0.5) { break; }
        
    	vec4 numCellsData = Load(currId, iChannel2, iChannelResolution[2].xy);
        float cellToRefill = step(1.5, numCellsData.x);
        dy += cellToRefill;
        numCells += cellToRefill;
    }
    
    if(dy < 0.5)
    {
        return;
    }
    
    for(float y=1.; y<yCells-0.5;++y)
    {
        vec2 currId = vec2(id.x, id.y +y);
        if(currId.y > yCells-0.5) { break; }
        if(y -0.5 > dy) { break; }
        
        vec4 numCellsData = Load(currId, iChannel2, iChannelResolution[2].xy);
        dy += step(1.5, numCellsData.x);
    }
    
    float baseDuration = 0.35*dy + 0.35*hash(gT);
    float delay = id.y*0.045 + mod(id.x, 3.)*0.045;

    vec2 sourceId = id;
    vec2 targetId = sourceId + vec2(0., dy);

    vec4 cellId = Load(mCellId.xy+targetId, iChannel0, iChannelResolution[0].xy);
    if(targetId.y > yCells-0.5)
    {
        cellId = vec4(GetCellPos(targetId), floor(hash(iTime+2.+(id.x+13.)*(id.y+25.)*18.)*5.), 1.);
    }
    vec4 cellPos = vec4(cellId.xy, gT+delay, baseDuration);

    Save(mCellId.xy +  sourceId.xy, cellId);
    Save(mCellPos.xy + sourceId.xy, cellPos);
}

// Function 87
vec3  sphericalTransmittanceGradient(vec2 L, float r, float h, float z)
{
	float Lx=L.x;
	float Ly=L.y;
	float Y = (r -h);
	float xmax = sqrt(2.0*r*h - h*h);
	float b = xmax;
	float a1 = (-xmax*0.7);
	if (DENS < 2.)
		a1 = 0.0;
	float a12 = a1*a1;float a13 = a12*a1;float a14 = a12*a12; 
	float Lx2 = Lx*Lx;float Lx3 = Lx2*Lx;float Lx4 = Lx3*Lx; float Lx5 = Lx4*Lx;float Lx6 = Lx5*Lx;
	float Ly2 = Ly*Ly;float Ly3 = Ly2*Ly;float Ly4 = Ly2*Ly2;float Ly5 = Ly4*Ly;float Ly6 = Ly5*Ly;
	float xmax3 = xmax*xmax*xmax;
	float Y2 = Y*Y;float Y3 = Y2*Y;float Y4 = Y2*Y2;
	float r2 = r*r;float r4 = r2*r2;
	float R2 = rad*rad;
	float H2 = z*z;
	float S = sqrt(a12*Lx2+Y2*Ly2-a12+2.*a1*Lx*Y*Ly-Y2+r2);
	float c1 = S*xmax3+5.*Lx2*r2*Y2*Ly2-3.*R2*Y2*Ly2+3.*H2*Y2*Ly2-5.*Lx2*Y4*Ly2
		-2.*Lx2*Y2*r2+5.*Ly4*Y2*r2-8.*Y2*Ly2*r2+4.*Lx2*Y4*Ly4-2.*S*a13
		-21.*S*a12*Lx2*Y*Ly+12.*S*Ly3*a12*Lx2*Y+12.*S*Lx4*a12*Y*Ly
		-3.*S*Lx2*Y*Ly*r2-2.*Ly2*a14+22.*Lx4*a14-8.*Lx6*a14-20.*a14*Lx2
		-3.*a12*r2+3.*Y2*a12-4.*Y2*Ly2*a12+Ly4*Y2*a12-8.*Ly2*a14*Lx4
		+4.*Lx4*a12*Y2-7.*Y2*a12*Lx2+10.*Ly2*a14*Lx2+Ly2*a12*r2-4.*Lx4*a12*r2
		+7.*a12*Lx2*r2+6.*a14-20.*Ly3*a13*Lx3*Y-12.*Ly4*a12*Lx2*Y2+11.*Ly3*a13*Lx*Y
		-20.*Lx5*a13*Y*Ly-12.*Lx4*a12*Y2*Ly2+41.*Lx3*a13*Y*Ly+23.*Lx2*Y2*Ly2*a12
		-21.*a13*Lx*Y*Ly+4.*a1*Lx3*Y3*Ly3-7.*a1*Y3*Ly3*Lx+3.*a1*Y3*Lx*Ly
		+4.*a1*Ly5*Y3*Lx-a1*Lx3*Y3*Ly-4.*Ly2*a12*Lx2*r2-6.*S*Y3*Ly+9.*S*Y3*Ly3
		+3.*S*H2*xmax+3.*S*Y2*xmax+3.*R2*Y2-3.*R2*r2-3.*H2*Y2+3.*H2*r2+10.*Y4*Ly2
		+3.*Y2*r2+Lx2*Y4+4.*Ly6*Y4-11.*Ly4*Y4+Ly2*r4+Lx2*r4-3.*Y4-4.*S*Ly5*Y3
		+8.*S*Lx5*a13-3.*S*R2*xmax-18.*S*a13*Lx3+12.*S*a13*Lx+3.*S*R2*Y*Ly
		-6.*S*Ly2*a13*Lx+8.*S*Ly2*a13*Lx3+6.*S*a12*Y*Ly-3.*S*H2*Y*Ly+3.*S*Lx2*Y3*Ly
		+3.*S*Y*Ly*r2-4.*S*Lx2*Y3*Ly3-3.*S*Ly3*Y*a12-3.*S*Ly3*Y*r2-3.*a1*R2*Lx*Y*Ly
		+3.*a1*H2*Lx*Y*Ly+a1*Ly3*Y*Lx*r2+a1*Lx3*Y*Ly*r2;	
	c1 *= (1./3.)*DENS/(S*R2);	
	float c2 = Y2*S-4.*Ly4*Y2*Lx*S+2.*Ly3*Y*S*a1-4.*Ly2*a12*Lx3*S
		+3.*Ly2*a12*Lx*S-8.*Lx4*a1*Y*Ly*S+14.*Lx2*Y*Ly*S*a1-3.*a13
		-4.*Y*Ly*S*a1-Ly2*S*Lx*r2-4.*Lx3*Y2*Ly2*S+7.*Y2*Ly2*Lx*S
		+9.*Lx3*a12*S+R2*Lx*S-2.*Y2*Lx*S-Lx3*S*r2+Lx*S*r2-H2*Lx*S
		-6.*a12*Lx*S-4.*Lx5*a12*S+Lx3*S*Y2-R2*S+a12*S-8.*Ly3*a1*Lx2*Y*S
		+12.*Ly3*a12*Lx3*Y+12.*Ly4*a1*Lx2*Y2-7.*Ly3*a12*Lx*Y+12.*Lx5*a12*Y*Ly
		+12.*Lx4*a1*Y2*Ly2-25.*Lx3*a12*Y*Ly-23.*Lx2*Y2*Ly2*a1+13.*a12*Lx*Y*Ly
		-R2*Lx*Y*Ly+H2*Lx*Y*Ly+5.*Y2*Ly2*a1-2.*Ly4*Y2*a1+4.*Ly2*a13*Lx4
		-3.*Lx4*a1*Y2+4.*Lx3*Y3*Ly3+6.*Y2*a1*Lx2-9.*Y3*Ly3*Lx
		+5.*Y3*Lx*Ly-R2*a1*Lx2+H2*a1*Lx2-5.*Ly2*a13*Lx2+4.*Ly5*Y3*Lx-3.*Lx3*Y3*Ly
		-Ly2*a1*r2+3.*Lx4*a1*r2-5.*a1*Lx2*r2+2.*a1*r2-11.*Lx4*a13+4.*Lx6*a13
		+10.*a13*Lx2+H2*S+R2*a1-3.*Y2*a1-H2*a1+Ly2*a13+3.*Ly2*a1*Lx2*r2
		+3.*Ly3*Y*Lx*r2+3.*Lx3*Y*Ly*r2-4.*Lx*Y*Ly*r2;
	c2 *= DENS/(R2*S);
	if (abs(c2) < 0.1)
		c2 = 0.1; // arbitraire
	float EX1 = exp(c1-c2*xmax);
	float EX2 = exp(c1+c2*xmax);
	float res = -2.*EX1+EX1*c2*c2*R2-EX1*c2*c2*Y2-EX1*c2*c2*H2
		-2.*EX1*c2*xmax-EX1*xmax*xmax*c2*c2+2.*EX2-EX2*c2*c2*R2+EX2*c2*c2*Y2+EX2*c2*c2*H2
		-2.*EX2*c2*xmax+EX2*xmax*xmax*c2*c2;
	res *= -DENS/(rad*rad*c2*c2*c2);
	return vec3(res);
}

// Function 88
vec3 colorBrushStroke(vec2 uvLine, vec2 uvPaper, vec2 lineSize, float sdGeometry, vec3 inpColor, vec4 brushColor)
{
    float posInLineY = (uvLine.y / lineSize.y);// position along the line. in the line is 0-1.

    if(iMouse.z > 0.)
    {
//    return mix(inpColor, vec3(0), dtoa(sdGeometry, 1000.));// reveal geometry.
//    return mix(inpColor, dtocolor(inpColor, uvLine.y), dtoa(sdGeometry, 1000.));// reveal Y
//    return mix(inpColor, dtocolor(inpColor, posInLineY), dtoa(sdGeometry, 1000.));// reveal pos in line.
//    return mix(inpColor, dtocolor(inpColor, uvLine.x), dtoa(sdGeometry, 1000.));// reveal X
    	float okthen = 42.;// NOP
    }
    
    // warp the position-in-line, to control the curve of the brush falloff.
    if(posInLineY > 0.)
    {
        float mouseX = iMouse.x == 0. ? 0.2 : (iMouse.x / iResolution.x);
	    posInLineY = pow(posInLineY, (pow(mouseX,2.) * 15.) + 1.5);
    }

    // brush stroke fibers effect.
    float strokeBoundary = dtoa(sdGeometry, 300.);// keeps stroke texture inside the geometry.
    float strokeTexture = 0.
        + noise01(uvLine * vec2(min(iResolution.y,iResolution.x)*0.2, 1.))// high freq fibers
        + noise01(uvLine * vec2(79., 1.))// smooth brush texture. lots of room for variation here, also layering.
        + noise01(uvLine * vec2(14., 1.))// low freq noise, gives more variation
        ;
    strokeTexture *= 0.333 * strokeBoundary;// 0 to 1 (take average of above)
    strokeTexture = max(0.008, strokeTexture);// avoid 0; it will be ugly to modulate
  	// fade it from very dark to almost nonexistent by manipulating the curve along Y
	float strokeAlpha = pow(strokeTexture, max(0.,posInLineY)+0.09);// add allows bleeding
    // fade out the end of the stroke by shifting the noise curve below 0
    const float strokeAlphaBoost = 1.09;
    if(posInLineY > 0.)
        strokeAlpha = strokeAlphaBoost * max(0., strokeAlpha - pow(posInLineY,0.5));// fade out
    else
        strokeAlpha *= strokeAlphaBoost;

    strokeAlpha = smoothf(strokeAlpha);
    
    // paper bleed effect.
    float paperBleedAmt = 60. + (rand(uvPaper.y) * 30.) + (rand(uvPaper.x) * 30.);
//    amt = 500.;// disable paper bleed    
    
    // blotches (per stroke)
    //float blotchAmt = smoothstep(17.,18.5,magicBox(vec3(uvPaper, uvLine.x)));
    //blotchAmt *= 0.4;
    //strokeAlpha += blotchAmt;

    float alpha = strokeAlpha * brushColor.a * dtoa(sdGeometry, paperBleedAmt);
    alpha = clamp(alpha, 0.,1.);
    return mix(inpColor, brushColor.rgb, alpha);
}

// Function 89
vec2 GradientA(vec2 uv, vec2 d, vec4 selector, int level){
    vec4 dX = 0.5*BlurA(uv + vec2(1.,0.)*d, level) - 0.5*BlurA(uv - vec2(1.,0.)*d, level);
    vec4 dY = 0.5*BlurA(uv + vec2(0.,1.)*d, level) - 0.5*BlurA(uv - vec2(0.,1.)*d, level);
    return vec2( dot(dX, selector), dot(dY, selector) );
}

// Function 90
v2 colorBrushStroke(v1 u,v2 r,v3 m,v1 p,v0 w, v0 sdGeometry, v2 inpColor, v3 bc//brushColor
){w=(u.y/w)// position along the line. in the line is 0-1.
 ;if(false ){ //important for uv debugging
  ;//return mix(inpColor, v2(0), dtoa(sdGeometry, 1000.));// reveal geometry.
  ;//return mix(inpColor, debugDist(u.y), dtoa(sdGeometry, 1000.));// reveal Y
  ;//return mix(inpColor, debugDist(w), dtoa(sdGeometry, 1000.));// reveal pos in line.
  ;return mix(inpColor, debugDist(u.x), dtoa(sdGeometry, 1000.));// reveal X
  ;}
 ;if(w>0.   // warp position-in-line, to control the curve of the brush falloff.
 ){v0 mouseX=m.x==0.?.2:(m.x/r.x)
  ;w = pow(w, (pow(mouseX,2.)*15.)+1.5);}
 ;v0 n=0.//bleed noise
 +noise01(u*v1(min(r.y,r.x)*.2, 1.))//tiny
 +noise01(u*v1(79,1))//fine
 +noise01(u*v1(14,1))//coarse
 ;n*=dtoa(sdGeometry, 300.)/3.// keep stroke texture inside geometry.
 ;n=max(.08,n)//null-evasion
 ;v0 a=pow(n,max(0.,w)+.09)//add allows bleeding
 ;if(w>0.)a=max(0.,a-pow(w,0.5))//optioonal more fading
 ;a=sh4(a)+.4*smoothstep(17.,18.5,fractalFungus(v2(p,u.x)))//hermite+fungalFreckles
 ;bc.a=sat(a*bc.a*dtoa(sdGeometry,paperbleed(p)))
 ;return mix(inpColor,bc.xyz,bc.a);}

// Function 91
vec3 gradient(vec3 p) {
	vec2 e = vec2(0., 0.001);

	return normalize(
		vec3(
			DE(p+e.yxx).x - DE(p-e.yxx).x,
			DE(p+e.xyx).x - DE(p-e.xyx).x,
			DE(p+e.xxy).x - DE(p-e.xxy).x
		)
	);
}

// Function 92
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

// Function 93
vec2 calculateStrokeDerivative() {
    vec3 lastMouse = read(LastMouse).xyz;
    vec2 lastDerivative = lastMouse.z > 0.0 ? read(StrokeDerivative).xy : normalize(iMouse.xy - lastMouse.xy);
    return normalize(iMouse.xy - (lastMouse.xy + lastDerivative * 0.5 * length(iMouse.xy - lastMouse.xy)));
}

// Function 94
vec3 stripeGradient(float t) {
	return vec3(mod(floor(t * 32.0), 2.0) * 0.2 + 0.8);
}

// Function 95
bool in_fill() {
    return (_stack.shape.y <= 0.0);
}

// Function 96
float get_gradient_eps() {
    return _stack.scale/AA;
}

// Function 97
vec3 ansiGradient(float t) {
	return mod(floor(t * vec3(8.0, 4.0, 2.0)), 2.0);
}

// Function 98
vec3 gradient( vec3 pos ) {
	const vec3 dx = vec3( grad_step, 0.0, 0.0 );
	const vec3 dy = vec3( 0.0, grad_step, 0.0 );
	const vec3 dz = vec3( 0.0, 0.0, grad_step );
	return normalize (
		vec3(
			dist_field( pos + dx ) - dist_field( pos - dx ),
			dist_field( pos + dy ) - dist_field( pos - dy ),
			dist_field( pos + dz ) - dist_field( pos - dz )			
		)
	);
}

// Function 99
float fillLine( in vec2 p, float x1, float y1, float x2, float y2, float th1, float th2 )
{ 
	vec2 li = sdSegment( vec2(x1,y1), vec2(x2,y2), p );
    float d = li.x - mix(th1,th2,li.y);
	
    float w = fwidth(d)*4.0;
	return 1.0 - smoothstep(-w, w, d);
}

// Function 100
void strokeBlend(
    vec4 src,
    float srcA,
    vec4 dst,
    out vec4 color
)
{
    color.a = src.a + (1.0-src.a)*dst.a;
    color.rgb = src.rgb*src.a + dst.rgb*dst.a*(1.0-src.a);
    color.rgb /= color.a;

    if (color.a > srcA) {
	    color.a = max(dst.a, srcA);
    }
}

// Function 101
vec4 gradient (vec4 p){
    const vec4 dx = vec4(0.1, 0.0, 0.0, 0.0);
    const vec4 dy = vec4(0.0, 0.1, 0.0, 0.0);
    const vec4 dz = vec4(0.0, 0.0, 0.1, 0.0);
    const vec4 dw = vec4(0.0, 0.0, 0.0, 0.1);
    
    return normalize(vec4(
        		map(p+dx) - map(p-dx),
                map(p+dy) - map(p-dy),
                map(p+dz) - map(p-dz),
                map(p+dw) - map(p-dw)
    			));
}

// Function 102
vec3 humanizeBrushStrokeDonut(vec2 uvLine, float radius_, bool clockwise, float lineLength)
{
    vec2 humanizedUVLine = uvLine;
    
	// offsetting the circle along its path creates a twisting effect.
    float twistAmt = .24;
    float linePosY = humanizedUVLine.y / lineLength;// 0 to 1 scale
    humanizedUVLine.x += linePosY * twistAmt;
    
    // perturb radius / x
    float humanizedRadius = radius_;
    float res = min(iResolution.y,iResolution.x);
    humanizedRadius += (noise01(uvLine * 1.)-0.5) * 0.04;
    humanizedRadius += sin(uvLine.y * 3.) * 0.019;// smooth lp wave
    humanizedUVLine.x += sin(uvLine.x * 30.) * 0.02;// more messin
    humanizedUVLine.x += (noise01(uvLine * 5.)-0.5) * 0.005;// a sort of random waviness like individual strands are moving around
//    humanizedUVLine.x += (noise01(uvLine * res * 0.18)-0.5) * 0.0035;// HP random noise makes it look less scientific
    
    return vec3(humanizedUVLine, humanizedRadius);
}

// Function 103
float fillQuad( in vec2 p, float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4 )
{ 
    float d1 = sdTriangle( vec2(x1,y1), vec2(x2,y2), vec2(x3,y3), p );
    float d2 = sdTriangle( vec2(x1,y1), vec2(x3,y3), vec2(x4,y4), p );
    float d = min( d1, d2 );
    float w = fwidth(d)*4.0;
	return 1.0 - smoothstep(0.0, w, d);
}

// Function 104
vec3 HueGradient(float t)
{
	t += .4;
    vec3 p = abs(fract(t + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
	return (clamp(p - 1.0, 0.0, 1.0));
}

// Function 105
vec4 gradientColor(float t, float dist){
    vec3 col1 = 1.1*t*vec3(0.0,1.0,0.0) + (1.0-1.1*t)*vec3(0.0,0.0,1.0);
    vec3 col2 = smoothstep(0.1,0.0,t)*vec3(0.6) + smoothstep(0.0,0.1,t)*col1;
    
    float alpha = exp(-4.0*abs(dist)) * t * t; //not finished.
	return vec4(col2,alpha);
}

// Function 106
vec3 colorBrushStrokeDonut(vec2 uv, vec3 inpColor, vec4 brushColor, vec2 o, float radius_, float angleStart, float sweepAmt, float lineWidth, bool clockwise)
{
	vec2 uvLine = uv - o;
    float angle = atan(uvLine.x, uvLine.y) + pi;// 0-2pi
    angle = mod(angle-angleStart+pi, pi2);
    if(!clockwise)
        angle = pi2 - angle;
    float lineLength = radius_ * pi2;// this is calculated before any humanizing/perturbance. so it's possible that it's slightly inaccurate, but in ways that will never matter
    uvLine = vec2(
        radius_ - length(uvLine),
        angle / pi2 * lineLength
    );
    
    // make line slightly narrower at end.
    float lineWidth1 = lineWidth * mix(1., .9, smoothstep(0.,lineLength,uvLine.y));
    
    vec3 hu = humanizeBrushStrokeDonut(uvLine, radius_, clockwise, lineLength);
    vec2 humanizedUVLine = hu.xy;
    float humanizedRadius = hu.z;

    float d = opS(sdCircle(uv, o, humanizedRadius),
                  sdCircle(uv, o, humanizedRadius));
    d -= lineWidth1 * 0.5;// round off things just like in the line routine.

    vec3 ret = colorBrushStroke(humanizedUVLine, uv, vec2(lineWidth1, lineLength), d, inpColor, brushColor);
    
    // do the same but for before the beginning of the line. distance field is just a single point
    vec3 ret2 = vec3(1);
    if(angle > pi)
    {
        uvLine.y -= lineLength;
        hu = humanizeBrushStrokeDonut(uvLine, radius_, clockwise, lineLength);
        humanizedUVLine = hu.xy;
        humanizedRadius = hu.z;
        vec2 strokeStartPos = o + vec2(sin(angleStart), cos(angleStart)) * humanizedRadius;
        d = distance(uv, strokeStartPos);
        d -= lineWidth * 0.5 * 1.;// round off things just like in the line routine.
        ret2 = colorBrushStroke(humanizedUVLine, uv, vec2(lineWidth, lineLength), d, inpColor, brushColor);
	}
    return min(ret, ret2);
}

// Function 107
vec2 GetGradient(vec2 intPos, float t) {
    
    // Uncomment for calculated rand
    //float rand = fract(sin(dot(intPos, vec2(12.9898, 78.233))) * 43758.5453);;
    
    // Texture-based rand (a bit faster on my GPU)
    float rand = texture(iChannel0, intPos / 64.0).r;
    
    // Rotate gradient: random starting rotation, random rotation rate
    float angle = 6.283185 * rand + 4.0 * t * rand;
    return vec2(cos(angle), sin(angle));
}

// Function 108
vec3 Gradient(vec3 RP, float Time) {
    return -normalize(vec3(SDF(RP-eps.xyy,Time).D-SDF(RP+eps.xyy,Time).D,
                            SDF(RP-eps.yxy,Time).D-SDF(RP+eps.yxy,Time).D,
                            SDF(RP-eps.yyx,Time).D-SDF(RP+eps.yyx,Time).D));
}

// Function 109
bool in_stroke() {
    float w = stroke_shape().y;
    return (w <= 0.0);
}

// Function 110
vec3 gradientNoised(vec2 pos, vec2 scale, float rotation, float seed) 
{
    vec2 sinCos = vec2(sin(rotation), cos(rotation));
    return gradientNoised(pos, scale, mat2(sinCos.y, sinCos.x, sinCos.x, sinCos.y), seed);
}

// Function 111
vec4 gradient(float pos)
{
    float step1 = 0.333;
    float step2 = 0.667;
    
    vec3 result = vec3(1, 0, 0); //Step0 between 0 and step1
    // Step through each case and choose either the previous result or the new result:
    result = mix(result, vec3(0, 1, 0), step(step1, pos));
    result = mix(result, vec3(0, 0, 1), step(step2, pos));
    return vec4(result, 1);
}

// Function 112
vec3 fill_box(vec2 p, vec2 p1, vec2 p3, vec3 current_colour, vec3 fill_colour) {
    vec3 colour = current_colour;
    vec2 p2 = vec2(p1.x, p3.y);
    vec2 p4 = vec2(p3.x, p1.y);
    vec2 box_center = (p1 + p3)/2.0;
    float bd = rectangle(p-box_center, (p3-p1)/2.0);
    float noise = get_noise2(p);
    noise *= fill_noise;
    if (bd+noise<0.0)
        colour = fill_colour;
    return colour;
}

// Function 113
float stroke(float x, float s, float w) { // 04
    float d = sstep(s, x + w / 2.) - sstep(s, x - w / 2.);
    return clamp(d, 0., 1.);
}

// Function 114
float fill(float d) {
    return smoothstep(.05, .0, d);
}

// Function 115
vec3 colorBrushStroke(vec2 uvLine, vec2 uvPaper, vec2 lineSize, float sdGeometry, vec3 inpColor, vec4 brushColor
){float posInLineY=(uvLine.y/lineSize.y)// position along the line. in the line is 0-1.
 ;if(iMouse.z>0.
 ){
//return mix(inpColor, vec3(0), dtoa(sdGeometry, 1000.));// reveal geometry.
//return mix(inpColor, dtocolor(inpColor, uvLine.y), dtoa(sdGeometry, 1000.));// reveal Y
//return mix(inpColor, dtocolor(inpColor, posInLineY), dtoa(sdGeometry, 1000.));// reveal pos in line.
//return mix(inpColor, dtocolor(inpColor, uvLine.x), dtoa(sdGeometry, 1000.));// reveal X
  ;}
  ;if(posInLineY>0.   // warp position-in-line, to control the curve of the brush falloff.
  ){float mouseX = iMouse.x == 0. ? 0.2 : (iMouse.x / iResolution.x)
   ;posInLineY = pow(posInLineY, (pow(mouseX,2.) * 15.) + 1.5)
  ;}
  // brush stroke fibers effect.
 ;float strokeBoundary=dtoa(sdGeometry, 300.)// keep stroke texture inside geometry.
 ;float strokeTexture=0.
 +noise01(uvLine*vec2(min(iResolution.y,iResolution.x)*.2, 1.))//tiny
 +noise01(uvLine*vec2(79,1))//fine
 +noise01(uvLine*vec2(14,1))//coarse
 ;strokeTexture*=strokeBoundary/3.
 ;strokeTexture=max(.08,strokeTexture)//null-evasion
 ;//fade along y
 ;float strokeAlpha = pow(strokeTexture, max(0.,posInLineY)+.09)// add allows bleeding
 ;const float strokeAlphaBoost=1.09
 ;if(posInLineY>0.)strokeAlpha = strokeAlphaBoost * max(0., strokeAlpha - pow(posInLineY,0.5))// fade out the end
 ;else strokeAlpha*=strokeAlphaBoost
 ;strokeAlpha = smoothf(strokeAlpha)
 ;float paperBleedAmt =60.+(rand(uvPaper.y)*30.) + (rand(uvPaper.x) * 30.)
 ;//paperBleedAmt = 500.// disable paper bleed    
 ;strokeAlpha+=.4*smoothstep(17.,18.5,magicBox(vec3(uvPaper,uvLine.x)))
 ;float alpha=strokeAlpha * brushColor.a*dtoa(sdGeometry, paperBleedAmt)
 ;alpha=sat(alpha)
 ;return mix(inpColor, brushColor.rgb, alpha);}

// Function 116
void stroke(float dist, vec3 color, inout vec3 fragColor, float thickness, float aa)
{
    float alpha = smoothstep(0.5 * (thickness + aa), 0.5 * (thickness - aa), abs(dist));
    fragColor = mix(fragColor, color, alpha);
}

// Function 117
float momentumGradientDescent(float startT, Ray ray, float learningRate, float exponentialFactor)
{
    float t = startT;
    float deltaT = 0.;
    
    for(int i = 0; i < 300; i++)
    {
        deltaT = exponentialFactor * deltaT - learningRate * dSqDistanceToRay_dt(t, ray);
        
        t += deltaT;
    }
    
    return t;
}

// Function 118
vec3 colorBrushStroke(vec2 uvLine, vec2 uvPaper, vec2 lineSize, float sdGeometry, vec3 inpColor, vec4 brushColor
){float posInLineY=(uvLine.y/lineSize.y)// position along the line. in the line is 0-1.
 ;if(iMouse.z>0.
 ){
//    return mix(inpColor, vec3(0), dtoa(sdGeometry, 1000.));// reveal geometry.
//    return mix(inpColor, dtocolor(inpColor, uvLine.y), dtoa(sdGeometry, 1000.));// reveal Y
//    return mix(inpColor, dtocolor(inpColor, posInLineY), dtoa(sdGeometry, 1000.));// reveal pos in line.
//    return mix(inpColor, dtocolor(inpColor, uvLine.x), dtoa(sdGeometry, 1000.));// reveal X
  ;}
  ;if(posInLineY>0.   // warp position-in-line, to control the curve of the brush falloff.
  ){float mouseX = iMouse.x == 0. ? 0.2 : (iMouse.x / iResolution.x)
   ;posInLineY = pow(posInLineY, (pow(mouseX,2.) * 15.) + 1.5)
  ;}
  // brush stroke fibers effect.
 ;float strokeBoundary=dtoa(sdGeometry, 300.)// keep stroke texture inside geometry.
 ;float strokeTexture=0.
 +noise01(uvLine*vec2(min(iResolution.y,iResolution.x)*.2, 1.))//tiny
 +noise01(uvLine*vec2(79,1))//fine
 +noise01(uvLine*vec2(14,1))//coarse
 ;strokeTexture*=strokeBoundary/3.
 ;strokeTexture=max(.08,strokeTexture)//null-evasion
 ;//fade along y
 ;float strokeAlpha = pow(strokeTexture, max(0.,posInLineY)+.09)// add allows bleeding
 ;const float strokeAlphaBoost=1.09
 ;if(posInLineY>0.)strokeAlpha = strokeAlphaBoost * max(0., strokeAlpha - pow(posInLineY,0.5))// fade out the end
 ;else strokeAlpha*=strokeAlphaBoost
 ;strokeAlpha = smoothf(strokeAlpha)
 ;float paperBleedAmt =60.+(rand(uvPaper.y)*30.) + (rand(uvPaper.x) * 30.)
 ;//paperBleedAmt = 500.// disable paper bleed    
 ;strokeAlpha+=.4*smoothstep(17.,18.5,magicBox(vec3(uvPaper,uvLine.x)))
 ;float alpha=strokeAlpha * brushColor.a*dtoa(sdGeometry, paperBleedAmt)
 ;alpha=sat(alpha)
 ;return mix(inpColor, brushColor.rgb, alpha);}

// Function 119
vec3 gradientNormal(vec3 p//6tap 3d derivative. 
){vec2 e=vec2(0,GRADIENT_DELTA)
 ;return normalize(vec3
 (map(p+e.yxx).y-map(p-e.yxx).y
 ,map(p+e.xyx).y-map(p-e.xyx).y
 ,map(p+e.xxy).y-map(p-e.xxy).y));}

// Function 120
void fill() {
    fill_preserve();
    new_path();
}

// Function 121
void strokeBlendUniColor(vec4 src, float srcA, vec4 dst, out vec4 color)
{
    color.rgb = src.rgb;
    color.a = dst.a + (1.0-dst.a)*src.a; // This can be done either way, I like adding to dst.
    if (color.a > srcA) {
	    color.a = max(dst.a, srcA);
    }
}

// Function 122
float fill(float x, float size) {
	return smoothstep(3./iResolution.y, 0., x-size);
}

// Function 123
float fillRectangle( in vec2 p, in float x, in float y, in float dirx, in float diry, in float radx, in float rady )
{
	float d = box(p,x,y,dirx,diry,radx,rady);
    float w = fwidth(d)*4.0;
	return 1.0 - smoothstep(-w, w, d);
}

// Function 124
vec3 techGradient(float t) {
	return pow(vec3(t + 0.01), vec3(120.0, 10.0, 180.0));
}

// Function 125
vec3 hueGradient(float t) {
    vec3 p = abs(fract(t + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
	return (clamp(p - 1.0, 0.0, 1.0));
}

// Function 126
float fillBezier( in vec2 p, float x1, float y1, float x2, float y2, float x3, float y3, float th1, float th2 )
{ 
	vec3 be = sdBezier( vec2(x1,y1), vec2(x2,y2), vec2(x3,y3), p );
    float d = length(be.xy) - mix(th1,th2,be.z);
	
    float w = fwidth(d)*4.0;
	return 1.0 - smoothstep(-w, w, d);
}

// Function 127
vec3 gradient(float s)
{
	return vec3(0.0, max(1.0-s*2.0, 0.0), max(s>0.5?1.0-(s-0.5)*5.0:1.0, 0.0));
}

// Function 128
vec4 stroke(float distance, float linewidth, float antialias, vec4 stroke)
{
    vec4 frag_color;
    float t = linewidth/2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);

    if( border_distance > (linewidth/2.0 + antialias) )
        discard;
    else if( border_distance < 0.0 )
        frag_color = stroke;
    else
        frag_color = vec4(stroke.rgb*alpha, 1.);

    return frag_color;
}

// Function 129
vec3 HeatmapGradient(const in float value) 
	{
		return clamp((pow(value, 1.5) * 0.8 + 0.2) * vec3(smoothstep(0.0, 0.35, value) + value * 0.5, 
													  smoothstep(0.5, 1.0, value), 
													  max(1.0 - value * 1.7, value * 7.0 - 6.0)), 
					 0.0, 1.0);
	}

// Function 130
vec2 GradientB(vec2 uv, vec2 d, vec4 selector, int level, sampler2D bufB, sampler2D bufD){
    vec4 dX = BlurB(uv + vec2(1.,0.)*d, level, bufB, bufD) - BlurB(uv - vec2(1.,0.)*d, level, bufB, bufD);
    vec4 dY = BlurB(uv + vec2(0.,1.)*d, level, bufB, bufD) - BlurB(uv - vec2(0.,1.)*d, level, bufB, bufD);
    return vec2( dot(dX, selector), dot(dY, selector) );
}

// Function 131
vec2 gradient(vec2 uv){
	return vec2(implicit_x(uv),implicit_y(uv));
}

// Function 132
float circleFill(vec2 uv, vec2 center, float radius) {
    float r = length(uv - center);
    return smoothFloat(r, radius);
}

// Function 133
vec3 _gradient(vec3 p, ivec2 coord, sampler2D BUFFB)
{
    vec3 delta = vec3(0);
    int missing = -1;
    vec4 last = portalSurface(coord + circle4(-1));
    for(int i = 0; i < 4 && isActualSurfacePoint(Mode(last)); i++)
    {
        vec4 pointData = portalSurface(coord + circle4(i));


        if(isActualSurfacePoint(Mode(pointData)))
        {
            // Minimize area.
            
            // area = sq(0.5 * cross(b - a, p - a))
            // area = 0.25 * sq(cross(b - a, p - a))
            // d(area)/dp = 0.25 * 2. * cross(b - a, p - a) * d(cross(b - a, p - a))/dp
            // d(cross(b - a, p - a))/dp = cross(b - a, d(p - a)/dp)
            // d(cross(b - a, p - a))/dp = cross(b - a, I)
            // d(cross(b - a, p - a))/dp = crossMatForm(b - a)
            // d(area)/dp = 0.5 * cross(b - a, p - a) * crossMatForm(b - a)
            // d(area)/dp = 0.5 * cross(b - a, cross(b - a, po - a))
            // d(area)/dp = 0.5 * ( dot(b - a, p - a) * (b - a) - sq(b - a) * (p - a) )
                    
            vec3 a = last.xyz;
            vec3 b = pointData.xyz;

            vec3 pa = (p.xyz - a);
            vec3 ba = (b - a);


            delta -= 0.25 * 0.5*(dot(ba, pa) * ba - sq(ba) * pa);
            //delta -= 0.25 * 0.5 * cross(ba, normalize(cross(ba, pa)));
        }
        else missing = i;

        last = pointData;
    }


    // Boundery points
    if(!isActualSurfacePoint(Mode(last)))
    {
        // Minimize length instead
        delta = vec3(0);
        for(int i = missing - 1; i <= missing + 1; i += 2)
        {
            // To have the same "units" as the area gradient, instead of minimizing:
            // l = sq(p - o)
            // I use:
            // l = (sq(p - o))^2
            // Which is the same as minimaizeing the area squared.
            
            // l = (sq(p - o))^2
            // dl/dp = 2.*sq(p - o) * d(sq(p - o))/dp
            // dl/dp = 2.*sq(p - o) * 2.*(p - o)
            // The second part of the multiplication is the gradient i would have without "conserving the units",
            // so it will probably work with or without scaling by 2.*sq(p - o).
            
            vec3 po = p - portalSurface(coord + circle4(i)).xyz;
        
            delta += 0.5 * (2.*po * 2.*sq(po));
        }
        //delta *= .2;
        
        /*float len = length(delta);
        if(len > 0.)
        {
            delta /= len;
            len = max(len, 0.001);
            delta *= len;
        }
        else delta = vec3(0, 0, 0.001);*/
        
        //vec3 badDirection = normalize(portalSurface(coord + circle4(missing + 2)).xyz - p.xyz);
        //delta = delta - abs(dot(badDirection, delta))*badDirection;

        /*if(isActualSurfacePoint(Mode(portalSurface(coord + circle4(missing + 1)))) && isActualSurfacePoint(Mode(portalSurface(coord + circle4(missing - 1)))) && isActualSurfacePoint(Mode(portalSurface(coord + circle4(missing + 2)))))
        {

            vec3 right = normalize(portalSurface(coord + circle4(missing + 1)).xyz - portalSurface(coord + circle4(missing - 1)).xyz);
            vec3 forward = normalize(portalSurface(coord + circle4(missing + 2)).xyz - p.xyz);
            vec3 up = cross(right, forward);

            delta = right * dot(right, delta) + up * dot(up, delta);
        }
        else delta = vec3(0);*/
    }
    
    return delta;
}

// Function 134
vec2 getGradient(vec2 coord)
{
	return 2. * normalize(pseudorandPos(floor(coord))) - 1.0;
}

// Function 135
void set_source_radial_gradient(vec4 color0, vec4 color1, vec2 p, float r) {
    float h = clamp( length(_stack.position.xy - p) / r, 0.0, 1.0 );
    set_source_rgba(mix(color0, color1, h));
}

// Function 136
void stroke_preserve() {
    float w = stroke_shape().x;
    write_color(_stack.source, calc_aa_blur(w));
}

// Function 137
float fill(vec2 fragCoord) {
    float sideLength = min(iResolution.x,iResolution.y) * 0.9;
    vec2 sqMin = (iResolution.xy - vec2(sideLength)) / 2.0;
    vec2 sqMax = (iResolution.xy + vec2(sideLength)) / 2.0;
    
    if(any(lessThan(fragCoord,sqMin)) || any(greaterThan(fragCoord, sqMax))) {
        return 1.0;
    }
    
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = (fragCoord - sqMin) / vec2(sideLength);
    
    float t = 1.0 - mod(iTime * timeScale, 1.0);
	
    vec2 qTuv = quadTreeUV(uv, 6.0, 0.0);
    
    float distS = max(abs(qTuv.x - 0.5), abs(qTuv.y - 0.5)) * 2.0; 		// for squares
    //float distC = length(qTuv - 0.5) * 2.0;							// for circles
    //float distD = (abs(qTuv.x - 0.5) + abs(qTuv.y - 0.5)) * 2.0;		// for diamonds
    //float distO = max(distS, distD / 1.5);							// for octagons
    
    
    float o = pow(distS, 5.0);
    float i = pow(distS, 2.0);
    
    float outer = 1.0 - step(o, 0.8) * (1.0 - step(o, 0.1));
    float inner = step(i, t * 0.6 - 0.1) * (1.0 - step(i, t * 0.6 - 0.2));
    
    return outer - inner;
    
}

// Function 138
vec3 calculateGradientFromDistanceField(vec3 p) {
    
    float d = 0.001;
    float Dx = (distanceToObjects(p+vec3(d,0.0,0.0))-distanceToObjects(p+vec3(-d,0.0,0.0)))/(2.0*d);
    float Dy = (distanceToObjects(p+vec3(0.0,d,0.0))-distanceToObjects(p+vec3(0.0,-d,0.0)))/(2.0*d);
    float Dz = (distanceToObjects(p+vec3(0.0,0.0,d))-distanceToObjects(p+vec3(0.0,0.0,-d)))/(2.0*d);
    return vec3(Dx,Dy,Dz);
}

// Function 139
float stroke(float d, float w, float s, float i) { return abs(smoothstep(0.,s,abs(d)-(w*.5)) - i); }

// Function 140
float Fill(float2 _p0, float2 _p1, float2 uv)
{
    float2 p0;
    float2 p1;
    if(_p0.y<_p1.y)
    {
        p0=_p0;
        p1=_p1;
    }
    else
    {
        p0=_p1;
        p1=_p0;
    }
    if(uv.y<p0.y)
        return 0.0;
    if(uv.y>=p1.y)
        return 0.0;
    float2 dp=p1-p0;
    float2 du=uv-p0;
    if(dot(float2(dp.y,-dp.x),du)>0.0) 
        return 0.0;
    return 0.5;
}

// Function 141
void sdCaveGradientNormal(in vec3 pos, float diff, inout vec3 normal, inout float sd){
    DEBUG_CALLING_NORMAL = true;
	sd = sdCaveGradient(pos);
    vec2 e = vec2(diff, 0.);
    normal = normalize(sd - vec3(
    	sdCaveGradient(pos - e.xyy),
    	sdCaveGradient(pos - e.yxy),
    	sdCaveGradient(pos - e.yyx)
    ));
    DEBUG_CALLING_NORMAL = false;
}

// Function 142
float fill(float d, float size) {
	return smoothstep(pixel_width,0.0,d-size);
}

// Function 143
v2 humanizeBrushStrokeDonut(v1 uvLine, v0 radius_, bool clockwise, v0 lineLength
){v1 humanizedUVLine = uvLine
 ;v0 twistAmt=.24//offset circle along its path for a twisting effect.
 ;v0 linePosY = humanizedUVLine.y / lineLength// 0 to 1 scale
 ;humanizedUVLine.x += linePosY * twistAmt
 ;v0 humanizedRadius = radius_ // perturb radius / x
 ;v0 res = min(iResolution.y,iResolution.x)
 ;humanizedRadius += (noise01(uvLine * 1.)-0.5) * 0.04
 ;humanizedRadius += sin(uvLine.y * 3.) * 0.019// smooth lp wave
 ;humanizedUVLine.x += sin(uvLine.x * 30.) * 0.02// more messin
 ;humanizedUVLine.x += (noise01(uvLine * 5.)-0.5) * 0.005// a sort of random waviness like individual strands are moving around
 ;//humanizedUVLine.x += (noise01(uvLine * res * 0.18)-0.5) * 0.0035;// HP random noise makes it look less scientific
 ; return v2(humanizedUVLine, humanizedRadius);}

// Function 144
float vonronoistroke(vec2 uv) {
    float h = vonronoi(uv);
    float d = 0.0;
    d -= 0.5*vonronoi(uv+vec2(0,dx));
    d -= 0.5*vonronoi(uv+vec2(dx,0));
    h += d;
    h*=10.0;
    return abs(h);
}

// Function 145
float fillMask(float distanceChange, float dist) 
{
    return smoothstep(distanceChange, -distanceChange, dist);
}

// Function 146
float stroke(float d, float w, bool f) {  return abs(ssaa(abs(d)-w*.5) - float(f)); }

// Function 147
vec2 GradientB(vec2 uv, vec2 d, vec4 selector, int level){
    return GradientB(uv, d, selector, level, iChannel1, iChannel3);
}

// Function 148
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

// Function 149
vec3 fireGradient(float t) {
	return max(pow(vec3(min(t * 1.02, 1.0)), vec3(1.7, 25.0, 100.0)), 
			   vec3(0.06 * pow(max(1.0 - abs(t - 0.35), 0.0), 5.0)));
}

// Function 150
vec3 Gradient(vec3 p,float d){vec2 e=vec2(.001,0);p*=99.;
 return (vec3(df2(p+e.xyy),df2(p+e.yxy),df2(p+e.yyx))*99.-d)/e.x;}

// Function 151
vec3 gradient(float factor)
{
	vec3 a = vec3(0.478, 0.4500, 0.500);
	vec3 b = vec3(0.500);
	vec3 c = vec3(0.1688, 0.748, 0.1748);
	vec3 d = vec3(0.1318, 0.388, 0.1908);

	return palette(factor, a, b, c, d);
}

// Function 152
float simpleFill(float d){return clamp(-d, 0.0, 1.0);}

// Function 153
float gradientNoise(vec3 x, float freq)
{
    // grid
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    // quintic interpolant
    vec3 u = w * w * w * (w * (w * 6. - 15.) + 10.);

    
    // gradients
    vec3 ga = hash33(mod(p + vec3(0., 0., 0.), freq));
    vec3 gb = hash33(mod(p + vec3(1., 0., 0.), freq));
    vec3 gc = hash33(mod(p + vec3(0., 1., 0.), freq));
    vec3 gd = hash33(mod(p + vec3(1., 1., 0.), freq));
    vec3 ge = hash33(mod(p + vec3(0., 0., 1.), freq));
    vec3 gf = hash33(mod(p + vec3(1., 0., 1.), freq));
    vec3 gg = hash33(mod(p + vec3(0., 1., 1.), freq));
    vec3 gh = hash33(mod(p + vec3(1., 1., 1.), freq));
    
    // projections
    float va = dot(ga, w - vec3(0., 0., 0.));
    float vb = dot(gb, w - vec3(1., 0., 0.));
    float vc = dot(gc, w - vec3(0., 1., 0.));
    float vd = dot(gd, w - vec3(1., 1., 0.));
    float ve = dot(ge, w - vec3(0., 0., 1.));
    float vf = dot(gf, w - vec3(1., 0., 1.));
    float vg = dot(gg, w - vec3(0., 1., 1.));
    float vh = dot(gh, w - vec3(1., 1., 1.));
	
    // interpolation
    return va + 
           u.x * (vb - va) + 
           u.y * (vc - va) + 
           u.z * (ve - va) + 
           u.x * u.y * (va - vb - vc + vd) + 
           u.y * u.z * (va - vc - ve + vg) + 
           u.z * u.x * (va - vb - ve + vf) + 
           u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);
}

// Function 154
vec2 gradient_limit (mat2 mtx, vec2 x, vec2 y, vec2 d, float eps) {
    vec2 dx;
    dx = _63(_38(surface(mtx, _32(x, eps), y), surface(mtx, _44(x, eps), y)), 1.000000 / (2.000000 * eps));
    vec2 dy;
    dy = _63(_38(surface(mtx, x, _32(y, eps)), surface(mtx, x, _44(y, eps))), 1.000000 / (2.000000 * eps));
    return _29(_63(dx, d . x), _63(dy, d . y));
}

// Function 155
vec3 calcGradientCheap( in vec3 pos ,in float original)
{
    const vec3 v1 = vec3(1.0,0.0,0.0);
    const vec3 v2 = vec3(0.0,1.0,0.0);
    const vec3 v3 = vec3(0.0,0.0,1.0);
	return (vec3(scene(pos + v1*eps),scene(pos + v2*eps),scene(pos + v3*eps))
           -vec3(original))/(eps);
}

// Function 156
float GradientSample(vec2 uv, out float w, vec3 CN, float CD) {
    vec4 GA=texture(iChannel0,uv*IRES+vec2(0.5));
    if (GA.w>99990.) { w=0.; return 0.; }
    w=Weight(CD-GA.w,dot(CN,Read(GA.y).xyz*2.-1.));
    return texture(iChannel0,uv*0.25*IRES+vec2(0.5,0.)).x*w;
}

// Function 157
void getGradients(vec2 fragCoord, out vec2 velocity, out float pressure)
{
    vec4 self = F(0,0);
    vec4 left = self;
    vec4 right = self;
    vec4 bottom = self;
    vec4 top = self;

    if(!wl)
    	left = F(-1, 0);
    if(!wr)
    	right = F(1, 0);
    if(!wb)
    	bottom = F(0, -1);
    if(!wt)
    	top = F(0, 1);
    velocity = vec2(left.z - right.z, bottom.z - top.z);
    pressure = (left.x - right.x) + (bottom.y - top.y);

}

// Function 158
vec3 colorGradient(float gradient, vec3 minColor, vec3 midColor, vec3 maxColor)
{
    if (gradient <= 0.5){
        gradient *= 2.0;
    	return colorGradient(gradient, minColor, midColor);
    }
    else{
        gradient = (gradient - 0.5) * 2.0;
        return colorGradient(gradient, midColor, maxColor);
    }
}

// Function 159
vec2 getGradient(vec2 uv){

    float scale = 0.1;
    float delta = 1e-1;
    
    uv *= scale;
    
    float data = texture(iChannel2, uv).r;
    float gradX = data - texture(iChannel2, uv-vec2(delta, 0.0)).r;
    float gradY = data - texture(iChannel2, uv-vec2(0.0, delta)).r;
    
    return vec2(gradX, gradY);
}

// Function 160
float fillMask(float dist){return clamp(-dist, 0.0, 1.0);}

// Function 161
vec3 heatmapGradient(float t) {
	return clamp((pow(t, 1.5) * 0.8 + 0.2) * vec3(smoothstep(0.0, 0.35, t) + t * 0.5, smoothstep(0.5, 1.0, t), max(1.0 - t * 1.7, t * 7.0 - 6.0)), 0.0, 1.0);
}

// Function 162
vec3 colorGradient(float gradient, vec3 minColor, vec3 maxColor)
{
    return (1.0 - gradient) * minColor + gradient * maxColor;
}

// Function 163
vec3 bgGradient(in vec2 uv) {
	float amt = (uv.y + 0.9) / 1.8;
	return mix(BG_BOT_COLOR, BG_TOP_COLOR, amt);
}

// Function 164
float smoothfill(in float a, in float b, in float epsilon)
{
    // A resolution-aware smooth edge for (a < b)
    return smoothstep(0., epsilon / iResolution.y, b - a);
}

// Function 165
float fillMask(float dist)
{
	return clamp(-dist, 0.0, 1.0);
}

// Function 166
v2 strokeLine(v1 u,v2 r,v3 M,v2 c, v3 b, v3 m, v0 w
){v0 lineAngle=atan(m.x-m.z,m.y-m.w)//axis-align
 ;mat2 rotMat =rot2D(lineAngle)
 ;v0 W=length(m.xy-m.zw)    // make an axis-aligned line from this line.
 ;v1 T=m.xy*rotMat// top left
 ;v1 B=T+v1(0,W)// bottom right
 ;v1 l=u*rotMat
 ;l.x+=(noise01(l*1.)-.5)*.02
 ;l.x+=cos(l.y*3.)*.009//lp wave
 ;l.x+=(noise01(l*5.)-.5)*.005;//random waviness like individual strands are moving around
 ;l.x+=(noise01(l*min(r.y,r.x)*.18)-.5)*.0035;// HP random noise makes it look less scientific
 ;v0 d=sdAxisAlignedRect(l,T,B)-w/2.
 ;return colorBrushStroke((T-l)*vec2(1,-1),r,M,u,W,d,c,b);}

// Function 167
vec3 ProjDiskImplicit_Gradient(vec3 rp, vec3 rd)
{
    float rdx_rcp = 1.0 / rd.x;
    
    vec2 yz = 2.0 * rp.xx * (rp.xx * rd.yz * rdx_rcp - rp.yz) * rdx_rcp;
    
    return vec3((-rd.y * yz.x - rd.z * yz.y) * rdx_rcp, yz);
}

// Function 168
vec3 gradient( vec3 v ) {
		const vec3 delta = vec3( grad_step, 0.0, 0.0 );
		float va = map(v).x;
		return normalize (vec3(map( v + delta.xyy).x - va, map( v + delta.yxy).x - va, map( v + delta.yyx).x - va));
	}

// Function 169
float hardFill(float d){return step(0.0, -d);}

// Function 170
vec3 DF_gradient( in vec3 p )
{
	const float d = 0.001;
	vec3 grad = vec3(DF_composition(p+vec3(d,0,0)).d-DF_composition(p-vec3(d,0,0)).d,
                     DF_composition(p+vec3(0,d,0)).d-DF_composition(p-vec3(0,d,0)).d,
                     DF_composition(p+vec3(0,0,d)).d-DF_composition(p-vec3(0,0,d)).d);
	return grad;
}

// Function 171
vec3 fill_triangle(vec2 p, vec2 p1, vec2 p2, vec2 p3, vec3 current_colour, vec3 fill_colour) {
    vec3 colour = current_colour;
    float d = triangle(p, p1, p2, p3);
    float noise = get_noise2(p);
    noise *= fill_noise;
    if (d+noise<0.0)
        colour = fill_colour;
    return colour;
}

// Function 172
v2 colorBrushStrokeDonut(v1 uv, v2 inpColor, v3 brushColor, v1 o, v0 radius_, v0 angleStart, v0 sweepAmt, v0 lineWidth
){v1 uvLine = uv - o
 ;v0 angle = atan(uvLine.x, uvLine.y) + pi// 0-2pi
 ;angle = mod(angle-angleStart+pi,tau)
 ;v0 lineLength = radius_ * tau// this is calculated before any humanizing/perturbance. so it's possible that it's slightly inaccurate, but in ways that will never matter
 ;uvLine = v1(radius_-length(uvLine),angle / tau * lineLength)
 ;v0 lineWidth1 = lineWidth * mix(1., .9, smoothstep(0.,lineLength,uvLine.y))//narrower end
 ;v2 hu = humanizeBrushStrokeDonut(uvLine, radius_, false, lineLength)
 ;v1 humanizedUVLine = hu.xy
 ;v0 humanizedRadius = hu.z
 ;v0 d = max(-length(uv-o)+humanizedRadius,length(uv-o)-humanizedRadius)
 ;d-=lineWidth1*.5// round off things just like in the line routine.
 ;v2 ret=colorBrushStroke(humanizedUVLine, uv, v1(lineWidth1, lineLength), d, inpColor, brushColor)
 ;v2 ret2 = v2(1)//same,but for line start. distance field is just a single point
 ;if(angle > pi
 ){uvLine.y -= lineLength
  ;hu = humanizeBrushStrokeDonut(uvLine, radius_, false, lineLength)
  ;humanizedUVLine = hu.xy
  ;humanizedRadius = hu.z
  ;v1 strokeStartPos = o + v1(sin(angleStart), cos(angleStart)) * humanizedRadius
  ;d=length(uv-strokeStartPos)
  -lineWidth*.5*1.// round off things just like in the line routine.
  ;ret2= colorBrushStroke(humanizedUVLine,uv,v1(lineWidth,lineLength),d,inpColor,brushColor)
 ;}return min(ret,ret2);}

// Function 173
vec2 gradientVectorAt(float ix, float iy) 
{
    // Deterministic randomness
    
    // arg is always positive
    float arg = (1. + sin(ix * dot(vec2(ix, iy), vec2(12.9898, 78.233)))) * 43758.5453;    
    float random = arg - floor(arg);
    return vec2(random);
}

// Function 174
float quadraticStrokeDistance(vec2 p0, vec2 p1, vec2 velocity, vec2 q) {
    vec2 pV = p0 + length(p1-p0) * 0.5 * velocity;
    return min(min(lineSegDistance(p0, p1, q), lineSegDistance(p0, pV, q)), lineSegDistance(pV, p1, q));
	if (length(p0-p1) < 2.0) {
		return lineSegDistance(p0, p1, q);
    }
    return length(get_distance_vector(p0-q, pV-q, p1-q));
}

// Function 175
float UpsampledGradient(vec2 uv, vec3 CN, float CD) {
    float w0,w1,w2,w3;
    vec2 fluv=(floor(uv)*4.+0.5);
    float G0=GradientSample(fluv,w0,CN,CD);
    float G1=GradientSample(fluv+vec2(4.,0.),w1,CN,CD);
    float G2=GradientSample(fluv+vec2(0.,4.),w2,CN,CD);
    float G3=GradientSample(fluv+vec2(4.),w3,CN,CD);
    vec2 fuv=fract(uv+0.5-0.5);
    return clamp(mix(mix(G0,G1,fuv.x),mix(G2,G3,fuv.x),fuv.y)
    			/(0.001+mix(mix(w0,w1,fuv.x),mix(w2,w3,fuv.x),fuv.y))*3.,0.,1.);
}

// Function 176
void fill(float dist, vec3 color, inout vec3 fragColor, float aa)
{
    float alpha = smoothstep(0.5*aa, -0.5*aa, dist);
    fragColor = mix(fragColor, color, alpha);
}

// Function 177
float fillTriangle( in vec2 p, float x1, float y1, float x2, float y2, float x3, float y3 )
{ 
    float d = sdTriangle( vec2(x1,y1), vec2(x2,y2), vec2(x3,y3), p );
	
    float w = fwidth(d)*4.0;
	return 1.0 - smoothstep(0.0, w, d);
}

// Function 178
vec3 RainbowGradient(const in float value) 
	{
		vec3 c = 1.0 - pow(abs(vec3(value) - vec3(0.65, 0.5, 0.2)) 
						   * vec3(3.0, 3.0, 5.0), vec3(1.5, 1.3, 1.7));
		c.r = max((0.15 - EaseInQuad(abs(value - 0.04) * 5.0)), c.r);
		c.g = (value < 0.5) ? smoothstep(0.04, 0.45, value) : c.g;
		return clamp(c, 0.0, 1.0);
	}

// Function 179
vec3 colorBrushStrokeDonut(vec2 uv, vec3 inpColor, vec4 brushColor, vec2 o, float radius_, float angleStart, float sweepAmt, float lineWidth
){vec2 uvLine = uv - o
 ;float angle = atan(uvLine.x, uvLine.y) + pi// 0-2pi
 ;angle = mod(angle-angleStart+pi,tau)
 ;float lineLength = radius_ * tau// this is calculated before any humanizing/perturbance. so it's possible that it's slightly inaccurate, but in ways that will never matter
 ;uvLine = vec2(radius_-length(uvLine),angle / tau * lineLength)
 ;float lineWidth1 = lineWidth * mix(1., .9, smoothstep(0.,lineLength,uvLine.y))//narrower end
 ;vec3 hu = humanizeBrushStrokeDonut(uvLine, radius_, false, lineLength)
 ;vec2 humanizedUVLine = hu.xy
 ;float humanizedRadius = hu.z
 ;float d = max(-length(uv-o)+humanizedRadius,length(uv-o)-humanizedRadius)
 ;d-=lineWidth1*.5// round off things just like in the line routine.
 ;vec3 ret=colorBrushStroke(humanizedUVLine, uv, vec2(lineWidth1, lineLength), d, inpColor, brushColor)
 ;vec3 ret2 = vec3(1)//same,but for line start. distance field is just a single point
 ;if(angle > pi
 ){uvLine.y -= lineLength
  ;hu = humanizeBrushStrokeDonut(uvLine, radius_, false, lineLength)
  ;humanizedUVLine = hu.xy
  ;humanizedRadius = hu.z
  ;vec2 strokeStartPos = o + vec2(sin(angleStart), cos(angleStart)) * humanizedRadius
  ;d=length(uv-strokeStartPos)
  -lineWidth*.5*1.// round off things just like in the line routine.
  ;ret2= colorBrushStroke(humanizedUVLine,uv,vec2(lineWidth,lineLength),d,inpColor,brushColor)
 ;}return min(ret,ret2);}

// Function 180
v2 colorBrushStrokeLine(v1 uv, v2 inpColor, v3 brushColor, v1 p1_, v1 p2_, v0 lineWidth
){v0 lineAngle = pi-atan(p1_.x - p2_.x, p1_.y - p2_.y)//axis-align
 ;mat2 rotMat = rot2D(lineAngle)
 ;v0 lineLength = distance(p2_, p1_)    // make an axis-aligned line from this line.
 ;v1 tl = (p1_ * rotMat)// top left
 ;v1 br = tl + v1(0,lineLength)// bottom right
 ;v1 uvLine = uv * rotMat
 ;uvLine.x+=(noise01(uvLine*1.)-.5)*.02
 ;uvLine.x+=cos(uvLine.y*3.)*.009// smooth lp wave
 ;uvLine.x+=(noise01(uvLine*5.)-.5)*.005;// a sort of random waviness like individual strands are moving around
 ;uvLine.x+=(noise01(uvLine*min(iResolution.y,iResolution.x)*.18)-.5)*.0035;// HP random noise makes it look less scientific
 ;v0 d = sdAxisAlignedRect(uvLine, tl, br)-lineWidth/2.
 ;uvLine=tl-uvLine
 ;v1 lineSize = v1(lineWidth, lineLength)
 ;v2 ret = colorBrushStroke(v1(uvLine.x, -uvLine.y), uv, lineSize,d, inpColor, brushColor)
 ;return ret;}

// Function 181
void drawGradientRect(vec2 pos, vec2 size, float t, vec4 startCol, vec4 endCol)
{
	vec2 inside = step(pos, xy) * step(xy, pos + size);
	mixColor(mix(startCol, endCol, t), inside.x * inside.y);
}

// Function 182
bool ngonFill( vec2 uv , vec2 PP[ maxCount ] )
{
    
    bool  isInner = false;

    // instead for 
    isInner = innerNgonFill( uv , PP[  4 ] , PP[ 10 ] , isInner ) ;        
    isInner = innerNgonFill( uv , PP[  5 ] , PP[  4 ] , isInner ) ;
    isInner = innerNgonFill( uv , PP[  6 ] , PP[  5 ] , isInner ) ;
    isInner = innerNgonFill( uv , PP[  7 ] , PP[  6 ] , isInner ) ;
    isInner = innerNgonFill( uv , PP[  8 ] , PP[  7 ] , isInner ) ;
    isInner = innerNgonFill( uv , PP[  9 ] , PP[  8 ] , isInner ) ;
	isInner = innerNgonFill( uv , PP[ 10 ] , PP[  9 ] , isInner ) ;

    return isInner;        
}

// Function 183
float FillLine(vec2 uv, vec2 pA, vec2 pB, vec2 thick, float rounded) {
    float df = LineDistField(uv, pA, pB, vec2(thick), rounded, 0.0);
    return saturate(df / abs(dFdy(uv).y));
}

// Function 184
void drawHorzGradientRect(vec2 co, vec2 bottomLeft, vec2 topRight, vec4 leftColor, vec4 rightColor)
{
	if ((co.x < bottomLeft.x) || (co.y < bottomLeft.y) ||
		(co.x > topRight.x) || (co.y > topRight.y))
	{
		return;	
	}
	
	float distanceRatio = (co.x - bottomLeft.x) / (topRight.x - bottomLeft.x); 
	
	fragColor = (1.0 - distanceRatio) * leftColor + distanceRatio * rightColor;
}

// Function 185
float fill(float x, float size) { // 09
    return 1. - step(size, x);
}

// Function 186
vec3 sample_triquadratic_gradient_approx(sampler3D channel, vec3 res, vec3 uv) {
    vec3 q = fract(uv * res);
    vec3 cc = 0.5 / res;
    vec3 ww0 = uv - cc;
    vec3 ww1 = uv + cc;
    float nx = texture(channel, vec3(ww1.x, uv.y, uv.z)).r - texture(channel, vec3(ww0.x, uv.y, uv.z)).r;
    float ny = texture(channel, vec3(uv.x, ww1.y, uv.z)).r - texture(channel, vec3(uv.x, ww0.y, uv.z)).r;
    float nz = texture(channel, vec3(uv.x, uv.y, ww1.z)).r - texture(channel, vec3(uv.x, uv.y, ww0.z)).r;
	return vec3(nx, ny, nz);
}

// Function 187
vec3 colorBrushStrokeDonut(vec2 uv, vec3 inpColor, vec4 brushColor, vec2 o, float radius_, float angleStart, float sweepAmt, float lineWidth
){vec2 uvLine = uv - o
 ;float angle = atan(uvLine.x, uvLine.y) + pi// 0-2pi
 ;angle = mod(angle-angleStart+pi, pi2)
 ;float lineLength = radius_ * pi2// this is calculated before any humanizing/perturbance. so it's possible that it's slightly inaccurate, but in ways that will never matter
 ;uvLine = vec2(radius_-length(uvLine),angle / pi2 * lineLength)
 ;float lineWidth1 = lineWidth * mix(1., .9, smoothstep(0.,lineLength,uvLine.y))//narrower end
 ;vec3 hu = humanizeBrushStrokeDonut(uvLine, radius_, false, lineLength)
 ;vec2 humanizedUVLine = hu.xy
 ;float humanizedRadius = hu.z
 ;float d = max(-length(uv-o)+humanizedRadius,length(uv-o)-humanizedRadius)
 ;d-=lineWidth1*.5// round off things just like in the line routine.
 ;vec3 ret=colorBrushStroke(humanizedUVLine, uv, vec2(lineWidth1, lineLength), d, inpColor, brushColor)
 ;vec3 ret2 = vec3(1)//same,but for line start. distance field is just a single point
 ;if(angle > pi
 ){uvLine.y -= lineLength
  ;hu = humanizeBrushStrokeDonut(uvLine, radius_, false, lineLength)
  ;humanizedUVLine = hu.xy
  ;humanizedRadius = hu.z
  ;vec2 strokeStartPos = o + vec2(sin(angleStart), cos(angleStart)) * humanizedRadius
  ;d=length(uv-strokeStartPos)
  -lineWidth*.5*1.// round off things just like in the line routine.
  ;ret2= colorBrushStroke(humanizedUVLine,uv,vec2(lineWidth,lineLength),d,inpColor,brushColor)
 ;}return min(ret,ret2);}

// Function 188
vec3 colorBrushStroke(vec2 u,vec3 r,vec4 m,vec2 p,float w, float sdGeometry, vec3 inpColor, vec4 bc//brushColor
){w=(u.y/w)// position along the line. in the line is 0-1.
 ;if(false ){ //important for uv debugging
  ;//return mix(inpColor, vec3(0), dtoa(sdGeometry, 1000.));// reveal geometry.
  ;//return mix(inpColor, debugDist(u.y), dtoa(sdGeometry, 1000.));// reveal Y
  ;//return mix(inpColor, debugDist(w), dtoa(sdGeometry, 1000.));// reveal pos in line.
  ;return mix(inpColor, debugDist(u.x), dtoa(sdGeometry, 1000.));// reveal X
  ;}
 ;if(w>0.   // warp position-in-line, to control the curve of the brush falloff.
 ){float mouseX=m.x==0.?.2:(m.x/r.x)
  ;w = pow(w, (pow(mouseX,2.)*15.)+1.5);}
 ;float n=0.//bleed noise
 +noise01(u*vec2(min(r.y,r.x)*.2, 1.))//tiny
 +noise01(u*vec2(79,1))//fine
 +noise01(u*vec2(14,1))//coarse
 ;n*=dtoa(sdGeometry, 300.)/3.// keep stroke texture inside geometry.
 ;n=max(.08,n)//null-evasion
 ;float a=pow(n,max(0.,w)+.09)//add allows bleeding
 ;if(w>0.)a=max(0.,a-pow(w,0.5))//optioonal more fading
 ;a=sh4(a)+.4*smoothstep(17.,18.5,fractalFungus(vec3(p,u.x)))//hermite+fungalFreckles
 ;bc.a=sat(a*bc.a*dtoa(sdGeometry,paperbleed(p)))
 ;return mix(inpColor,bc.xyz,bc.a);}

// Function 189
v2 colorBrushStroke(v1 uvLine, v1 uvPaper, v1 lineSize, v0 sdGeometry, v2 inpColor, v3 brushColor
){v0 posInLineY=(uvLine.y/lineSize.y)// position along the line. in the line is 0-1.
 ;if(iMouse.z>0.
 ){
//return mix(inpColor, v2(0), dtoa(sdGeometry, 1000.));// reveal geometry.
//return mix(inpColor, dtocolor(inpColor, uvLine.y), dtoa(sdGeometry, 1000.));// reveal Y
//return mix(inpColor, dtocolor(inpColor, posInLineY), dtoa(sdGeometry, 1000.));// reveal pos in line.
//return mix(inpColor, dtocolor(inpColor, uvLine.x), dtoa(sdGeometry, 1000.));// reveal X
  ;}
  ;if(posInLineY>0.   // warp position-in-line, to control the curve of the brush falloff.
  ){v0 mouseX = iMouse.x == 0. ? 0.2 : (iMouse.x / iResolution.x)
   ;posInLineY = pow(posInLineY, (pow(mouseX,2.) * 15.) + 1.5)
  ;}
  // brush stroke fibers effect.
 ;v0 strokeBoundary=dtoa(sdGeometry, 300.)// keep stroke texture inside geometry.
 ;v0 strokeTexture=0.
 +noise01(uvLine*v1(min(iResolution.y,iResolution.x)*.2, 1.))//tiny
 +noise01(uvLine*v1(79,1))//fine
 +noise01(uvLine*v1(14,1))//coarse
 ;strokeTexture*=strokeBoundary/3.
 ;strokeTexture=max(.08,strokeTexture)//null-evasion
 ;//fade along y
 ;v0 strokeAlpha = pow(strokeTexture, max(0.,posInLineY)+.09)// add allows bleeding
 ;const v0 strokeAlphaBoost=1.09
 ;if(posInLineY>0.)strokeAlpha = strokeAlphaBoost * max(0., strokeAlpha - pow(posInLineY,0.5))// fade out the end
 ;else strokeAlpha*=strokeAlphaBoost
 ;strokeAlpha = smoothf(strokeAlpha)
 ;v0 paperBleedAmt =60.+(rand(uvPaper.y)*30.) + (rand(uvPaper.x) * 30.)
 ;//paperBleedAmt = 500.// disable paper bleed    
 ;strokeAlpha+=.4*smoothstep(17.,18.5,magicBox(v2(uvPaper,uvLine.x)))
 ;v0 alpha=strokeAlpha * brushColor.a*dtoa(sdGeometry, paperBleedAmt)
 ;alpha=sat(alpha)
 ;return mix(inpColor, brushColor.rgb, alpha);}

// Function 190
vec3 Gradient(in vec3 P)
{
    const vec3 d = vec3(0.05, 0.0, 0.0);
    return vec3(
        Map(P + d.xyy) - Map(P - d.xyy),
        Map(P + d.yxy) - Map(P - d.yxy),
        Map(P + d.yyx) - Map(P - d.yyx)
    );
}

// Function 191
void fillExampleUsingSetIndex(int modNumber){    
    for (int i = 0; i < 16; i++)
        setValue(example, mod(float(i + modNumber), 16.0), sin(3.14 * float(i) / 16.0));
	}

// Function 192
void debug_clip_gradient() {
    vec2 d = _stack.clip;
    _color = mix(_color,
        hsl(d.x * 6.0,
            1.0, (d.x>=0.0)?0.5:0.3),
        0.5);
}

// Function 193
vec3 gradient(vec3 pos)
{
	const float eps=0.0001;
	float mid=distfunc(pos);
	return vec3(
	distfunc(pos+vec3(eps,0.0,0.0))-mid,
	distfunc(pos+vec3(0.0,eps,0.0))-mid,
	distfunc(pos+vec3(0.0,0.0,eps))-mid);
}

// Function 194
vec3 gradient(vec3 p, float t) {
				vec2 e = vec2(0., t);

				return normalize( 
					vec3(
						DE(p+e.yxx).y - DE(p-e.yxx).y,
						DE(p+e.xyx).y - DE(p-e.xyx).y,
						DE(p+e.xxy).y - DE(p-e.xxy).y
					)
				);
			}

// Function 195
float gradient(float p)
{
    vec2 pt0 = vec2(0.00,0.0);
    vec2 pt1 = vec2(0.86,0.1);
    vec2 pt2 = vec2(0.955,0.40);
    vec2 pt3 = vec2(0.99,1.0);
    vec2 pt4 = vec2(1.00,0.0);
    if (p < pt0.x) return pt0.y;
    if (p < pt1.x) return mix(pt0.y, pt1.y, (p-pt0.x) / (pt1.x-pt0.x));
    if (p < pt2.x) return mix(pt1.y, pt2.y, (p-pt1.x) / (pt2.x-pt1.x));
    if (p < pt3.x) return mix(pt2.y, pt3.y, (p-pt2.x) / (pt3.x-pt2.x));
    if (p < pt4.x) return mix(pt3.y, pt4.y, (p-pt3.x) / (pt4.x-pt3.x));
    return pt4.y;
}

// Function 196
vec3 gradient(vec3 pos, mat3 mat)
{
    const vec3 dx = vec3(EPSILON, 0., 0.);
    const vec3 dy = vec3(0., EPSILON, 0.);
    const vec3 dz = vec3(0., 0., EPSILON);
    
    return normalize(vec3(
    	sceneSDF(pos + dx, mat) - sceneSDF(pos - dx, mat),
        sceneSDF(pos + dy, mat) - sceneSDF(pos - dy, mat),
        sceneSDF(pos + dz, mat) - sceneSDF(pos - dz, mat)
    ));
}

// Function 197
float fill(float dist)
{
    return clamp(-dist, 0.0, 1.0);
}

// Function 198
define fillbox0(X,Y,v)  {V=abs(U-vec2(X,Y)*R+vec2(15,-15)); if(max(V.x,V.y)<15.) O=vec4(v);}

// Function 199
float stroke(float d, float w)
{
    return abs(d)-w;
}

// Function 200
float interleavedGradientNoise(vec2 pos)
{
  float f = 0.06711056 * pos.x + 0.00583715 * pos.y;
  return fract(52.9829189 * fract(f));
}

// Function 201
vec3 gradient( vec3 pos ) {
	const vec3 dx = vec3( grad_step, 0.0, 0.0 );
	const vec3 dy = vec3( 0.0, grad_step, 0.0 );
	const vec3 dz = vec3( 0.0, 0.0, grad_step );
	return normalize (
		vec3(
			map( pos + dx ) - map( pos - dx ),
			map( pos + dy ) - map( pos - dy ),
			map( pos + dz ) - map( pos - dz )			
		)
	);
}

// Function 202
vec4 gradient (float v) {
    float steps = 7.;
    float step = 1. / steps;
    vec4 col = black;

    if (v >= .0 && v < step) {
        col = mix (yellow, orange, v * steps);
    } else if (v >= step && v < 2.0 * step) {
        col = mix (orange, red, (v - step) * steps);
    } else if (v >= 2.0 * step && v < 3.0 * step) {
        col = mix (red, magenta, (v - 2.0 * step) * steps);
    } else if (v >= 3.0 * step && v < 4.0 * step) {
        col = mix (magenta, cyan, (v - 3.0 * step) * steps);
    } else if (v >= 4.0 * step && v < 5.0 * step) {
        col = mix (cyan, blue, (v - 4.0 * step) * steps);
    } else if (v >= 5.0 * step && v < 6.0 * step) {
        col = mix (blue, green, (v - 5.0 * step) * steps);
    }

    return col;
}

// Function 203
vec2 GetIntensityGradient(vec2 vCoord)
{
	float fDPixel = 1.0;
	
	float fPX = SampleBackbuffer(vCoord + vec2( fDPixel, 0.0));
	float fNX = SampleBackbuffer(vCoord + vec2(-fDPixel, 0.0));
	float fPY = SampleBackbuffer(vCoord + vec2(0.0,  fDPixel));
	float fNY = SampleBackbuffer(vCoord + vec2(0.0, -fDPixel));
	
	return vec2(fPX - fNX, fPY - fNY);              
}

// Function 204
float gradientDE(vec3 p) {
	last = 0;
	float r = escapeLength(p);
	if (r*r < 2.0) return 0.0;
	gradient = (vec3(escapeLength(p+xDir*EPSILON), escapeLength(p+yDir*EPSILON), escapeLength(p+zDir*EPSILON))-r)/EPSILON;
	return 0.5*r*log(r)/length(gradient);
}

// Function 205
vec4 getGradientValue(in vec2 uv)
{
    vec2 dist =	vec2(1.0, 0.0) - vec2(-1.0, 0.0);
	float val = dot( uv - vec2(-1,0), dist ) / dot( dist, dist );
	clamp( val, 0.0, 1.0 );
    
	vec4 color = mix( color1, color2, val );
	// clamp depending on higher alpha value
	if( color1.a >= color2.a )
		color.a = clamp( color.a, color2.a, color1.a );
	else
		color.a = clamp( color.a, color1.a, color2.a );
	return color;
}

// Function 206
vec2 sample_biquadratic_gradient_approx(sampler2D channel, vec2 res, vec2 uv) {
    vec2 q = fract(uv * res);
    vec2 cc = 0.5 / res;
    vec2 ww0 = uv - cc;
    vec2 ww1 = uv + cc;
    float nx = texture(channel, vec2(ww1.x, uv.y)).r - texture(channel, vec2(ww0.x, uv.y)).r;
    float ny = texture(channel, vec2(uv.x, ww1.y)).r - texture(channel, vec2(uv.x, ww0.y)).r;
	return vec2(nx, ny);
}

// Function 207
float sdCaveGradient(in vec3 pos){
    float density = getDensityBufB(pos);
    float nextDist = density;
    
    return nextDist;
}

// Function 208
float gradientNoise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    float rz =  mix( mix( dot( hashg( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( hashg( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hashg( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( hashg( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
    
    //return rz*0.75+0.5;
    return smoothstep(-.9,.9,rz);
}

// Function 209
vec4 fill(float dist, vec4 col) {
    return col * smoothstep(4.0/iResolution.y, 0.0, dist);
}

// Function 210
vec3 gradientColor (in float v)
{
	vec3 color = vec3 (0.0);
    const vec3 white = vec3 (1.0);
    const vec3 low = vec3 (0.35, 0.3, 0.4);
    const vec3 high = vec3 (0.5, 0.6, 0.55);

    if (v < 0.333) {
        color = v * low / 0.333;
    } else if (v < 0.666) {
        color = (v - 0.333) * (high - low) / 0.333 + low;
    } else {
        color = (v - 0.666) * (white - high) / 0.333 + high;
    }

    return color;
}

// Function 211
void stroke() {
    stroke_preserve();
    new_path();
}

// Function 212
vec3 gradient(float s)
{
	return vec3(22.0, max(1.0-s*2.0, 0.0), max(s>0.5?1.0-(s-0.15)*5.0:21.0, 1.0));
}

// Function 213
vec2 gradient(vec2 fragCoord)
{
    vec2 grad = vec2(0.0);
    grad.x -= texelFetch(iChannel0, ivec2(fragCoord)+ivec2(-1, 0), 0).z;
    grad.x += texelFetch(iChannel0, ivec2(fragCoord)+ivec2(+1, 0), 0).z;
    grad.y -= texelFetch(iChannel0, ivec2(fragCoord)+ivec2(0, -1), 0).z;
    grad.y += texelFetch(iChannel0, ivec2(fragCoord)+ivec2(0, +1), 0).z;
    return grad;
}

// Function 214
float circleFill(float dist, float radius, float thickness)
{
    if (dist <= thickness) { return 1.0; }
    return 0.0;
}

// Function 215
vec3 brightnessGradient(float t) {
	return vec3(t * t);
}

// Function 216
void set_source_linear_gradient(vec4 color0, vec4 color1, vec2 p0, vec2 p1) {
    vec2 pa = _stack.position.xy - p0;
    vec2 ba = p1 - p0;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    set_source_rgba(mix(color0, color1, h));
}

// Function 217
vec3 DF_gradient( in vec3 p )
{
    //The field gradient is the distance derivative along each axis.
    //The surface normal follows the direction where this variation is strongest.
	const float d = 0.001;
	vec3 grad = vec3(DF_composition(p+vec3(d,0,0)).d-DF_composition(p-vec3(d,0,0)).d,
                     DF_composition(p+vec3(0,d,0)).d-DF_composition(p-vec3(0,d,0)).d,
                     DF_composition(p+vec3(0,0,d)).d-DF_composition(p-vec3(0,0,d)).d);
	return grad/(2.0*d);
}

// Function 218
float fillMask(float dist){
	return clamp(-dist, 0.0, 1.0);
}

// Function 219
vec3 Gradient(vec3 p, float t) {
    return normalize(vec3(
        SDF(p+eps.xyy,t).D-SDF(p-eps.xyy,t).D,
        SDF(p+eps.yxy,t).D-SDF(p-eps.yxy,t).D,
        SDF(p+eps.yyx,t).D-SDF(p-eps.yyx,t).D));
}

// Function 220
vec3 gradient(in float t) {	
    vec3 a = vec3(.5, .5, .5);
    vec3 b = vec3(.5, .5, .5);
    vec3 c = vec3(1., 1., .5);
    vec3 d = vec3(.8, .9, .3);
    return a + b * cos(6.28318 * ( c * t + d));
}

// Function 221
void fillNumbers(){
    pV[0] = vec2(0, SIZE);  pV[1] = vec2(SIZE - 1, SIZE);
    pV[2] = vec2(0, 0); 	pV[3] = vec2(SIZE - 1, 0);
    
    for (int i = 0; i < 3; i++)
    	pH[i] = vec2(0, SIZE * i);
    
	}

// Function 222
vec3 CentralDiffGradient(vec3 p, float dt)
{
    vec3 a0 = vec3(p.x+dt,p.y,p.z);
    vec3 b0 = vec3(p.x,p.y+dt,p.z);
    vec3 c0 = vec3(p.x,p.y,p.z+dt);

    vec3 a1 = vec3(p.x-dt,p.y,p.z);
    vec3 b1 = vec3(p.x,p.y-dt,p.z);
    vec3 c1 = vec3(p.x,p.y,p.z-dt);

    vec3 gradient = vec3(
        VolumeIntensity(a0) - VolumeIntensity(a1),
        VolumeIntensity(b0) - VolumeIntensity(b1),
        VolumeIntensity(c0) - VolumeIntensity(c1));

    return normalize(gradient);
}

// Function 223
float stroke_alpha(float distance, float linewidth, float antialias)
{
    float t = linewidth/2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);
    if( border_distance > (linewidth/2.0 + antialias) )
        return 0.0;
    else if( border_distance < 0.0 )
        return 1.0;
    else
        return alpha;
}

// Function 224
void debug_gradient() {
    vec2 d = _stack.shape;
    _color = mix(_color,
        hsl(d.x * 6.0,
            1.0, (d.x>=0.0)?0.5:0.3),
        0.5);
}

// Function 225
vec3 gradient( vec3 v ) {
		const vec3 delta = vec3( grad_step, 0.0, 0.0 );
		float va = map( v ).x;
		return normalize (
			vec3(
				map( v + delta.xyy).x - va,
				map( v + delta.yxy).x - va,
				map( v + delta.yyx).x - va			
			)
		);
	}

// Function 226
vec3 GetSkyGradient( const in vec3 vDir )
{
    const vec3 cColourTop = vec3(0.7, 0.8, 1.0);
    const vec3 cColourHorizon = cColourTop * 0.5;

    float fBlend = clamp(vDir.y, 0.0, 1.0);
    return mix(cColourHorizon, cColourTop, fBlend);
}

// Function 227
vec3 gradient ( float shade ) {
	vec3 colour = vec3( (sin(iTime/2.0)*0.25)+0.25,0.0,(cos(iTime/2.0)*0.25)+0.25);
	
	vec2 mouseScaled = iMouse.xy/iResolution.xy;
	vec3 col1 = vec3(mouseScaled.x, 0.0, 1.0-mouseScaled.x);
	vec3 col2 = vec3(1.0-mouseScaled.x, 0.0, mouseScaled.x);
	vec3 col3 = vec3(mouseScaled.y, 1.0-mouseScaled.y, mouseScaled.y);
	vec3 col4 = vec3((mouseScaled.x+mouseScaled.y)/2.0, (mouseScaled.x+mouseScaled.y)/2.0, 1.0 - (mouseScaled.x+mouseScaled.y)/2.0);
	vec3 col5 = vec3(mouseScaled.y, mouseScaled.y, mouseScaled.y);
	
	colour += band ( shade, 0.0, 0.3, colour, col1 );
	colour += band ( shade, 0.3, 0.6, col1, col2 );
	colour += band ( shade, 0.6, 0.8, col2, col3 );
	colour += band ( shade, 0.8, 0.9, col3, col4 );
	colour += band ( shade, 0.9, 1.0, col4, col5 );
	
	return colour;
}

// Function 228
vec3 colorBrushStrokeLine(vec2 uv, vec3 inpColor, vec4 brushColor, vec2 p1_, vec2 p2_, float lineWidth
){float lineAngle = pi-atan(p1_.x - p2_.x, p1_.y - p2_.y)//axis-align
 ;mat2 rotMat = rot2D(lineAngle)
 ;float lineLength = distance(p2_, p1_)    // make an axis-aligned line from this line.
 ;vec2 tl = (p1_ * rotMat)// top left
 ;vec2 br = tl + vec2(0,lineLength)// bottom right
 ;vec2 uvLine = uv * rotMat
 ;uvLine.x+=(noise01(uvLine*1.)-.5)*.02
 ;uvLine.x+=cos(uvLine.y*3.)*.009// smooth lp wave
 ;uvLine.x+=(noise01(uvLine*5.)-.5)*.005;// a sort of random waviness like individual strands are moving around
 ;uvLine.x+=(noise01(uvLine*min(iResolution.y,iResolution.x)*.18)-.5)*.0035;// HP random noise makes it look less scientific
 ;float d = sdAxisAlignedRect(uvLine, tl, br)-lineWidth/2.
 ;uvLine=tl-uvLine
 ;vec2 lineSize = vec2(lineWidth, lineLength)
 ;vec3 ret = colorBrushStroke(vec2(uvLine.x, -uvLine.y), uv, lineSize,d, inpColor, brushColor)
 ;return ret;}

// Function 229
vec4 gradient(const vec3 s, const vec3 e, const vec2 uv, float levels) {
    // interpolate in linear space and convert back to sRGB
    #if LINEAR_GRADIENT > 0
    vec3 c = OECF(mix(EOCF(s / QUANTIZATION), EOCF(e / QUANTIZATION), uv.x));
    #else
    vec3 c= mix(s / QUANTIZATION, e / QUANTIZATION, uv.x);
    #endif

    // dither in sRGB space
         if (uv.y < 1.0 / 6.0) return Dither_Vlachos(vec4(c, 1.0), levels);
    else if (uv.y < 2.0 / 6.0) return Dither_Interleaved(vec4(c, 1.0), levels);
    else if (uv.y < 3.0 / 6.0) return Dither_TriangleNoise(vec4(c, 1.0), levels);
    else if (uv.y < 4.0 / 6.0) return Dither_Uniform(vec4(c, 1.0), levels);
    else if (uv.y < 5.0 / 6.0) return Dither_Ordered(vec4(c, 1.0), levels);
    else if (uv.y < 6.0 / 6.0) return Dither_None(vec4(c, 1.0), levels);

    return vec4(0.0);
}

// Function 230
vec3 desertGradient(float t) {
	float s = sqrt(clamp(1.0 - (t - 0.4) / 0.6, 0.0, 1.0));
	vec3 sky = sqrt(mix(vec3(1, 1, 1), vec3(0, 0.8, 1.0), smoothstep(0.4, 0.9, t)) * vec3(s, s, 1.0));
	vec3 land = mix(vec3(0.7, 0.3, 0.0), vec3(0.85, 0.75 + max(0.8 - t * 20.0, 0.0), 0.5), square(t / 0.4));
	return clamp((t > 0.4) ? sky : land, 0.0, 1.0) * clamp(1.5 * (1.0 - abs(t - 0.4)), 0.0, 1.0);
}

// Function 231
bool innerNgonFill( vec2 uv, vec2 PPi , vec2 PPj , bool isInner )
    {
        if (( PPi.y < uv.y && PPj.y >= uv.y || PPj.y < uv.y && PPi.y >= uv.y ) && ( PPi.x <= uv.x || PPj.x <= uv.x )) 
        {
            if ( PPi.x + ( uv.y - PPi.y ) / ( PPj.y -PPi.y ) * ( PPj.x - PPi.x ) < uv.x ) 
            {
                isInner=!isInner; 
            }
        }  

        return isInner;
    }

// Function 232
float fillMask(float dist)
{
	return clamp(-(dist+0.01)*100.0, 0.0, 1.0);
}

// Function 233
vec3 gradient( vec3 v ) {
	const vec3 dx = vec3( grad_step, 0.0, 0.0 );
	const vec3 dy = vec3( 0.0, grad_step, 0.0 );
	const vec3 dz = vec3( 0.0, 0.0, grad_step );
	return normalize (
		vec3(
			dist_field( v + dx ) - dist_field( v - dx ),
			dist_field( v + dy ) - dist_field( v - dy ),
			dist_field( v + dz ) - dist_field( v - dz )			
		)
	);
}

// Function 234
vec3 heatmapGradient(float t) {
	return clamp((pow(t, 1.5) * .8 + .2) * vec3(smoothstep(0., .35, t) + t * .5, smoothstep(.5, 1., t), max(1. - t * 1.7, t * 7. - 6.)), 0., 1.);
}

// Function 235
vec2 getGradient(vec2 uv){

    float delta = 1e-1;
    uv *= 0.3;
    
    float data = texture(iChannel1, uv).r;
    float gradX = data - texture(iChannel1, uv-vec2(delta, 0.0)).r;
    float gradY = data - texture(iChannel1, uv-vec2(0.0, delta)).r;
    
    return vec2(gradX, gradY);
}

// Function 236
vec3 gradient( vec3 pos ) {
	 vec3 dx = vec3( grad_step, 0.0, 0.0 );
	 vec3 dy = vec3( 0.0, grad_step, 0.0 );
	 vec3 dz = vec3( 0.0, 0.0, grad_step );
	return normalize (
		vec3(
			dist_field( pos + dx ) - dist_field( pos - dx ),
			dist_field( pos + dy ) - dist_field( pos - dy ),
			dist_field( pos + dz ) * dist_field( pos - dz )			
		)
	);
}

// Function 237
float gradientNoise(vec2 v) 
{
    vec2 i = floor(v);
    vec2 f = fract(v);
	
	vec2 u = smoothstep(0.0, 1.0, f);
	
	// random vectors at square corners
	vec2 randomA = random2(i + vec2(0.0,0.0));
	vec2 randomB = random2(i + vec2(1.0,0.0));
	vec2 randomC = random2(i + vec2(0.0,1.0));
	vec2 randomD = random2(i + vec2(1.0,1.0));
	
	// direction vectors from square corners to v
    //  directionN = v - (i + vec2(x,y)) = f - vec2(x,y)
	vec2 directionA = f - vec2(0.0,0.0);
	vec2 directionB = f - vec2(1.0,0.0);
	vec2 directionC = f - vec2(0.0,1.0);
	vec2 directionD = f - vec2(1.0,1.0);
	
	// "influence values"
	float a = dot(randomA, directionA);
	float b = dot(randomB, directionB);
	float c = dot(randomC, directionC);
	float d = dot(randomD, directionD);
	
    // final result: interpolate from corner values
	return mix( mix(a, b, u.x), mix(c, d, u.x), u.y );
}

// Function 238
vec3 gradient(float factor)
{
	vec3 a = vec3(0.478, 0.500, 0.500);
	vec3 b = vec3(0.500);
	vec3 c = vec3(0.688, 0.748, 0.748);
	vec3 d = vec3(0.318, 0.588, 0.908);

	return palette(factor, a, b, c, d);
}

// Function 239
float fill(float d, float s, float i) { return abs(smoothstep(0.,s,d) - i); }

// Function 240
float fill(float d, float i) { return abs(smoothstep(.0,.02,d) - i); }

// Function 241
float fillFunc(float x, float f, float smth)
{
	return clamp((f-tri01(x))/smth+.33,0.,1.);
}

// Function 242
vec2 gradient(vec2 pt, float dist) {
	float dfdu = glyph_dist(pt+ vec2(0.01, 0.0))- dist/ 1.01;
	float dfdv = glyph_dist(pt+ vec2(0.0, 0.01))- dist/ 1.01;
	vec2 grad = normalize(vec2(dfdu, -dfdv));
	return grad;
}

// Function 243
vec3 sampleMinusGradient(vec2 coord)
{
    vec3	veld	= texture(iChannel1, coord / iResolution.xy).xyz;
    float	left	= texture(iChannel0,(coord + vec2(-1, 0)) / iResolution.xy).x;
    float	right	= texture(iChannel0,(coord + vec2( 1, 0)) / iResolution.xy).x;
    float	bottom	= texture(iChannel0,(coord + vec2( 0,-1)) / iResolution.xy).x;
    float	top 	= texture(iChannel0,(coord + vec2( 0, 1)) / iResolution.xy).x;
    vec2	grad 	= vec2(right - left,top - bottom) * 0.5;
    return	vec3(veld.xy - grad, veld.z);
}

// Function 244
vec3 gradient (vec3 p){
 const float df = 0.1;
 const vec3 dx = vec3(df, 0.0, 0.0);
 const vec3 dy = vec3(0.0, df, 0.0);
 const vec3 dz = vec3(0.0, 0.0, df);
    
    return normalize(
        
        vec3(map( p + dx ) - map ( p - dx),
             map( p + dy ) - map ( p - dy),
             map( p + dz ) - map ( p - dz)
            )
        
        );
    
}

