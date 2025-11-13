// Reusable Patterns Procedural Functions
// Automatically extracted from procedural-related shaders

// Function 1
vec2 tile(vec2 st, float zoom){
    st *= zoom;
        if (mod(floor(st.x), 2.) == 0.){
        st = rotate2D(st, PI*.5);
    }
    return fract(st);
}

// Function 2
float smoothpattern(in vec3 pos)
{
    // squeeze into the smooth region
    vec3 p = pos;
    p.x -= 0.2*p.y; // unstretch a bit
    p *= 0.06;
    p += vec3(0.32, 0.61, 0.48);
    // magic param is a function of input pos
    vec3 param = vec3(p.x); 

    // kali set
	float mag, ac = 0.;
	for (int i=0; i<NUM_ITERS; ++i)
    {
		mag = dot(p, p);
        p = abs(p) / mag - param;
        ac += mag;
    }
    
    return ac / float(NUM_ITERS)
        // keep the intensity roughly the same
        // for all points
        	* 0.9 * (0.75 + 0.25 * pos.x) 
        // and push the surface in the [0,1] range
        - 1.5;
}

// Function 3
bool matches_pattern(mat2 in_pattern,vec2 fragCoord,float mag){
    float current_pixel;
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
        	float current_color = in_pattern[i][j];
            current_pixel = magnify(fragCoord+vec2(i,j),mag);
            if(current_color != current_pixel){
            	return false;
            }
        }
    }
    return true;
}

// Function 4
float tile2(vec2 uv)
{
    return abs(uv.x) - k * .5;
}

// Function 5
float cellTile(in vec3 p){
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    vec4 v, d; 
    d.x = drawSphere(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawSphere(p - vec3(.39, .2, .11));
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawSphere(p - vec3(.62, .24, .06));
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071;
    d.w = drawSphere(p - vec3(.2, .82, .64));

    v.xy = min(d.xz, d.yw), v.z = min(max(d.x, d.y), max(d.z, d.w)), v.w = max(v.x, v.y); 
   
    d.x =  min(v.z, v.w) - min(v.x, v.y); // Maximum minus second order, for that beveled Voronoi look. Range [0, 1].
    //d.x =  min(v.x, v.y);
        
    return d.x*2.66; // Normalize... roughly.
    
}

// Function 6
vec2 hex_tile(in vec2 hex_uv)
{
    vec2 tile = abs(hex_uv.yx);//abs AND flip!!!
    
    
    if(tile.x > tile.y*0.333*INV_BOX.y)//test fliped
        tile *= mat2(-HEX_ROT.y,HEX_ROT.x, HEX_ROT);//rotate fliped
        
    if(hex_uv.y < 0.0)//mirror x to shape a "triangle"
        tile.x = -tile.x;
     
     return tile;
}

// Function 7
vec4 textureNoTile( sampler2D samp, in vec2 uv )
{
    vec2 iuv = floor( uv );
    vec2 fuv = fract( uv );

#ifdef USEHASH    
    // generate per-tile transform (needs GL_NEAREST_MIPMAP_LINEARto work right)
    vec4 ofa = texture( iChannel1, (iuv + vec2(0.5,0.5))/256.0 );
    vec4 ofb = texture( iChannel1, (iuv + vec2(1.5,0.5))/256.0 );
    vec4 ofc = texture( iChannel1, (iuv + vec2(0.5,1.5))/256.0 );
    vec4 ofd = texture( iChannel1, (iuv + vec2(1.5,1.5))/256.0 );
#else
    // generate per-tile transform
    vec4 ofa = hash4( iuv + vec2(0.0,0.0) );
    vec4 ofb = hash4( iuv + vec2(1.0,0.0) );
    vec4 ofc = hash4( iuv + vec2(0.0,1.0) );
    vec4 ofd = hash4( iuv + vec2(1.0,1.0) );
#endif
    
    vec2 ddx = dFdx( uv );
    vec2 ddy = dFdy( uv );

    // transform per-tile uvs
    ofa.zw = sign(ofa.zw-0.5);
    ofb.zw = sign(ofb.zw-0.5);
    ofc.zw = sign(ofc.zw-0.5);
    ofd.zw = sign(ofd.zw-0.5);
    
    // uv's, and derivarives (for correct mipmapping)
    vec2 uva = uv*ofa.zw + ofa.xy; vec2 ddxa = ddx*ofa.zw; vec2 ddya = ddy*ofa.zw;
    vec2 uvb = uv*ofb.zw + ofb.xy; vec2 ddxb = ddx*ofb.zw; vec2 ddyb = ddy*ofb.zw;
    vec2 uvc = uv*ofc.zw + ofc.xy; vec2 ddxc = ddx*ofc.zw; vec2 ddyc = ddy*ofc.zw;
    vec2 uvd = uv*ofd.zw + ofd.xy; vec2 ddxd = ddx*ofd.zw; vec2 ddyd = ddy*ofd.zw;
        
    // fetch and blend
    vec2 b = smoothstep(0.25,0.75,fuv);
    
    if( useOld )
    {
        //original approach from iq
        return mix( mix( textureGrad( samp, uva, ddxa, ddya ), 
                         textureGrad( samp, uvb, ddxb, ddyb ), b.x ), 
                    mix( textureGrad( samp, uvc, ddxc, ddyc ),
                         textureGrad( samp, uvd, ddxd, ddyd ), b.x), b.y );

    }
    
    
    // huwb modification - modify blend based on relative brightness
    // to try to preserve intense features (don't add 50% white)
    vec4 A = textureGrad( samp, uva, ddxa, ddya );
    vec4 B = textureGrad( samp, uvb, ddxb, ddyb );
    vec4 C = textureGrad( samp, uvc, ddxc, ddyc );
    vec4 D = textureGrad( samp, uvd, ddxd, ddyd );
    
    vec4 AB = contrastBlend( A, B, b.x );
    vec4 CD = contrastBlend( C, D, b.x );
    return contrastBlend( AB, CD, b.y );
}

// Function 8
vec3 bloomTile(float lod, vec2 offset, vec2 uv){
    return texture(iChannel1, uv * exp2(-lod) + offset).rgb;
}

// Function 9
float ditherPattern(vec2 coord)
{
 	coord = fract(coord * 0.5);
    return dot(coord, vec2(0.25, 0.5));
}

// Function 10
float XORPattern3d(vec3 pos)
{
  float an = 0.5; //smoothstep(-1.5, 1.5, cos(3.14159*iTime));
  float x = 0.0;
  for( int i=0; i<7; i++ ) 
  {
    vec3 a=floor(pos);      
    vec3 b=fract(pos);      

    x += mod(a.x + a.y + a.z, 2.0) 
       * mix(1.0, 1.5*pow(4.0 *(1.0-b.x)*b.x 
                              *(1.0-b.y)*b.y
                              *(1.0-b.z)*b.z, 0.25), an);
    pos /= 2.0;
    x /= 2.0;
  }
  return x;
}

// Function 11
vec2 rotateTilePattern(vec2 _st){

    //  Scale the coordinate system by 2x2 
    _st *= 2.0;

    //  Give each cell an index number
    //  according to its position
    float index = 0.0;    
    index += step(1., mod(_st.x,2.0));
    index += step(1., mod(_st.y,2.0))*2.0;
    
    //      |
    //  2   |   3
    //      |
    //--------------
    //      |
    //  0   |   1
    //      |

    // Make each cell between 0.0 - 1.0
    _st = fract(_st);

    // Rotate each cell according to the index
    if(index == 1.0){
        //  Rotate cell 1 by 90 degrees
        // _st = rotate2D(_st,PI*0.5);
        _st = rotate2D(_st,PI*.5);
    } else if(index == 2.0){
        //  Rotate cell 2 by -90 degrees
        _st = rotate2D(_st,PI*-.5);
    } else if(index == 3.0){
        //  Rotate cell 3 by 180 degrees
        _st = rotate2D(_st,PI);
    } 

    return _st;
}

// Function 12
vec3 TileableCurlNoise(vec3 p, in float numCells, int octaves)
{
  const float e = .1;
  vec3 dx = vec3( e   , 0.0 , 0.0 );
  vec3 dy = vec3( 0.0 , e   , 0.0 );
  vec3 dz = vec3( 0.0 , 0.0 , e   );

  vec3 p_x0 = snoiseVec3( p - dx, numCells, octaves );
  vec3 p_x1 = snoiseVec3( p + dx, numCells, octaves );
  vec3 p_y0 = snoiseVec3( p - dy, numCells, octaves );
  vec3 p_y1 = snoiseVec3( p + dy, numCells, octaves );
  vec3 p_z0 = snoiseVec3( p - dz, numCells, octaves );
  vec3 p_z1 = snoiseVec3( p + dz, numCells, octaves );

  float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
  float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
  float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

  const float divisor = 1.0 / ( 2.0 * e );
  return normalize( vec3( x , y , z ) * divisor );
  // technically incorrect but I like this better...
  //return normalize(vec3( x , y , z ));
}

// Function 13
vec2 tile (vec2 uv)
{
    float cols = 3.0;
    float rows = 2.0;
    uv.x *= cols;
    uv.y *= rows;
    offset = floor(uv)+floor(iTime / DURATION)*2.0;
    uv = fract(uv);
    uv.x -= ((iResolution.x/cols-iResolution.y/rows)/(iResolution.x/cols))/rows;
    return scale(vec2((iResolution.x/cols)/(iResolution.y/rows), 1.0)) * uv;   
}

// Function 14
float pattern_bg(vec2 p){
    float d=0.;
    p=vec2(mod(p.x+0.01*(floor((mod(p.y,0.04)-0.02)/0.02)),0.02)-0.01,mod(p.y,0.02)-0.01);
    d=SS(-0.001,0.001,sdCircle(p,0.0035));
    return d;
}

// Function 15
float tile4(vec2 uv)
{
    return K2 - length(vec2(abs(uv.x) - .5, abs(uv.y) - .5));
}

// Function 16
vec3 tileWeave(vec2 pos, vec2 scale, float count, float width, float smoothness)
{
    vec2 i = floor(pos * scale);    
    float c = mod(i.x + i.y, 2.0);
    
    vec2 p = fract(pos.st * scale);
    p = mix(p.st, p.ts, c);
    p = fract(p * vec2(count, 1.0));
    
    // Vesica SDF based on Inigo Quilez
    width *= 2.0;
    p = p * 2.0 - 1.0;
    float d = sdfLens(p, width, 1.0);
    vec2 grad = vec2(dFdx(d), dFdy(d));

    float s = 1.0 - smoothstep(0.0, dot(abs(grad), vec2(1.0)) + smoothness, -d);
    return vec3(s , normalize(grad) * smoothstep(1.0, 0.99, s) * smoothstep(0.0, 0.01, s)); 
}

// Function 17
float Grid2Pattern(in vec2 uv)
{
  return 0.5*clamp(10.*sin(PI*uv.x), 0.0, 1.0)
       / 0.5*clamp(10.*sin(PI*uv.y), 0.0, 1.0);
}

// Function 18
float getPattern(vec2 uv) {  //this can be any pattern but moving patterns work best 
	//float w=texture(TEXTURE, uv*0.3).r;   
	float w=clouds(uv*5.0, iTime*0.5);
	return w;
}

// Function 19
float pattern(in vec2 p, out float c) {
    float t = iTime;
    vec2 q = vec2( fbm( p + vec2(0.0, 0.0) ),
                  fbm(  p + vec2(c2*.1, t*.02)) );

    c = fbm( p + 2.0*q + vec2(c1+c2,-t*.01));
    return fbm( p + 2.0*q );
}

// Function 20
float tile0(vec2 uv)
{
    float v = length(uv) - K3;
    float w = K2 - length(vec2(abs(uv.x) - .5, uv.y - .5));
    return v = mix(v, w,
                   //smoothstep(.1, -.1, abs(uv.x) - uv.y)
                   step(abs(uv.x), uv.y)
                  );
}

// Function 21
float pattern( in vec2 p )
{
	return fbm( p + fbm( p + fbm( p ) ) );
}

// Function 22
float pattern( in vec2 p )
  {
    vec2 q = vec2( fbm( p + vec2(0.0,0.0) ),
                   fbm( p + vec2(5.2,1.3) ) );

    return fbm( p + 4.0*q );
  }

// Function 23
float ChessPattern(in vec2 uv)
{
//return clamp(88.*sin(uv.x)* sin(uv.y), 0.0, 1.0);
  return 1. / sin(uv.x) / sin(uv.y);
}

// Function 24
float cellTile(in vec3 p){
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    vec4 d; 
    d.x = drawObject(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawObject(p - vec3(.39, .2, .11));
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawObject(p - vec3(.62, .24, .06));
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071;
    d.w = drawObject(p - vec3(.2, .82, .64));

    // Obtain the minimum, and you're done.
    d.xy = min(d.xz, d.yw);
        
    return min(d.x, d.y)*2.66; // Scale between zero and one... roughly//.
}

// Function 25
void get_tile_colour(in vec2 tile_id, out vec3 tile_hsv)
{
    // hue
    tile_hsv.x = hash21(tile_id);
    // saturation
    tile_hsv.y = 1.0;
    // value
    float modTime = mod(iTime, 100.0);
    float time = floor(modTime * 2.0);
    float level = step(lit_tile_threshold, hash31(vec3(tile_id, time)));
    float lastLevel = step(lit_tile_threshold, hash31(vec3(tile_id, time - 1.0)));
    level = max(level, lastLevel * (1.0 - fract(modTime * 2.0) * 1.5));
    tile_hsv.z = level;
}

// Function 26
float getFloorPattern(vec3 pos)
{
    return texture(iChannel1, pos.xz*textSize).r*
        smoothstep(0.02, 0., pos.y);
}

// Function 27
vec2 pattern6(vec2 uv)
{
    float time = iTime;
    float a = atan( uv.x, uv.y);
    float r = sqrt(dot(uv,uv));
    float u = 0.01*uv.y+0.01*cos(a*5.0)/r; // Looks cool, but a bit smashed, i'm doing it wrong!
    float v = 0.01*uv.x+0.01*sin(a*5.0)/r;
	return vec2(u,v);
}

// Function 28
float pattern2(vec2 uv)
{
    // correct for aspect ratio    
    float aspect = iResolution.y/iResolution.x;
    
    // rotate with time distortion in Y    
    float angle = -iTime * 0.05;
    
    // TODO -- I suspect this could be massively optimized.
    //         We are translating, rotating, scaling, translating
    //         Twice.  Grabbing coordinates within each.

    const float NUM_CELLS = 8.0;
    const float SHIFT_POSITIVE = 32.0; // ensure no negatives are rendered, since we use int(x)
    
    vec2 pStart = uv.xy - vec2(0.5);
    pStart.y *= aspect;
    
    // 1. normal checkerboard
    
    // translate
    vec2 p1 = pStart;
    // rotate
    p1 = rotateXY( p1, angle );
    // translate back
    p1 += vec2(SHIFT_POSITIVE + 0.5);    

    p1.xy = floor(p1.xy * NUM_CELLS);
    
    // 2. 45 degree rotated checkerboard, zoomed to match vertices
    
    // translate
    vec2 p2 = pStart;
    // rotate
    p2 = rotateXY( p2, angle + PI / 4.0);
    // expand
    p2 *= 1.41421356237;
    // translate back
    p2 += vec2(SHIFT_POSITIVE + 0.5);    

    p2.xy = floor(p2.xy * NUM_CELLS);
    
    // combine
    return mod(p1.x+p1.y + p2.x+p2.y, 2.0);
}

// Function 29
float TilePattern(vec2 p){
    
     
    vec2 ip = floor(p); // Cell ID.
    p -= ip + .5; // Cell's local position. Range [vec2(-.5), vec2(.5)].
    
     
    // Using the cell ID to generate a unique random number.
    float rnd = hash21(ip);
    float rnd2 = hash21(ip + 27.93);
    //float rnd3 = hash21(ip + 57.75);
     
    // Random tile rotation.
    float iRnd = floor(rnd*4.);
    p = rot2(iRnd*3.14159/2.)*p;
    // Random tile flipping.
    //p.y *= (rnd>.5)? -1. : 1.;
    
    
    // Rendering the arcs onto the tile.
    //
    float d = 1e5, d1 = 1e5, d2 = 1e5, d3 = 1e5, l;
    
   
    // Three top left arcs.
    l = length(p - vec2(-.5, .5));
    d1 = abs(l - .25);
    d2 = abs(l - .5);
    d3 = abs(l - .75);
    if(rnd2>.33) d3 = abs(length(p - vec2(.125, .5)) - .125);
    
    d = min(min(d1, d2), d3);
    
    // Two small arcs on the bottom right.
    d1 = 1e5;//abs(length(p - vec2(.5, .5)) - .25);
    //if(rnd3>.35) d1 = 1e5;//
    d2 = abs(length(p - vec2(.5, .125)) - .125);
    d3 = abs(length(p - vec2(.5, -.5)) - .25);
    d = min(d, min(d1, min(d2, d3))); 
    
    
    // Three bottom left arcs.
    l = length(p + .5);
    d = max(d, -(l - .75)); // Outer mask.
    
    // Equivalent to the block below:
    //
    //d1 = abs(l - .75);
    //d2 = abs(l - .5);
    //d3 = abs(l - .25);
    //d = min(d, min(min(d1, d2), d3));
	//
    d1 = abs(l - .5);
    d1 = min(d1, abs(d1 - .25));
    d = min(d, d1);
    
    
    // Arc width. 
    d -= .0625;
    
 
    // Return the distance field value for the grid tile.
    return d; 
    
}

// Function 30
float PatternCircles(vec2 p, float m){
  p.x-=m/2.0*step(0.0,sin(PI*p.y/m));
  p = mod(p,m)-m/2.0;
  return 1.0-sm(0.0,(p.x*p.x+p.y*p.y)-1.0);
}

// Function 31
vec3 canvasPattern(vec2 st, float width, float radius, float xPos){
    vec3 color = vec3(0.);
    st *= 100.;
    st.x *= .5;
    
    vec2 st_i = floor(st);

    if (mod(st_i.y,2.) == 1.) {
        st.x -= .5;
    }
    vec2 st_f = fract(st);
    color.r = 214.0/255.0;
    color.g = 206.0/255.0;
    color.b = 192.0/255.0;
    
    float pct = threadedEdges(st_f, width);
    pct += ovalGradient(st_f,radius, xPos);
    color += pct;

    return color;
}

// Function 32
float tileCubeShell(vec3 p, float t) {
	return max(p.y, -min(p.x, p.z));
}

// Function 33
vec2 tile (vec2 v) {return ((1.-fract(v)*2.)*sign(fract(0.5*v)*2.-1.))*0.5+0.5;}

// Function 34
vec3 getPattern(vec3 p)
{
    vec3 nz = snoise_swirl( p, 0.8, 2, 0.75, 1.0,  0.01, 0.01, 0.01);
    
	return nz;
}

// Function 35
vec3 pattern(vec2 uv) {

    vec3 col;
    if (uv.y < 0.02){
        // color trimmings
        col = col1 + (0.5 - step(1.0, length(col1)))*vec3(0.15);
    } else {
        float sel = random();
        return chooseM(uv);
    }
    
    return col;
}

// Function 36
vec4 getTile(int t, int x, int y)
{
	if (t == 0) return RGB(107,140,255);
	if (t == 1) return sprGround(x,y);
	if (t == 2) return sprQuestionBlock(x,y);
	if (t == 3) return sprUsedBlock(x,y);
	
	return RGB(107,140,255);
}

// Function 37
float dhexagonpattern(vec2 p) 
{
    vec2 q = vec2(p.x*1.2, p.y + p.x*.6),
        qi = floor(q),
        pf = fract(q);
    float v = mod(qi.x + qi.y, 3.);
    
    return dot(step(pf.xy,pf.yx), 1.-pf.yx + step(1.,v)*(pf.x+pf.y-1.) + step(2.,v)*(pf.yx-2.*pf.xy));
}

// Function 38
float StarPattern(vec2 p//ttps://www.shadertoy.com/view/4sKXzy 
){p= abs(fract(p*1.5)-.5)//adorable stars, smoothstep() of it is nice, too.
 ;return max(ma(p),mi(p)*2.);}

// Function 39
float tile1(vec2 uv)
{
    return abs(length(uv - .5) - .5) - k * .5;
}

// Function 40
float QCirclePattern(vec2 u){return sin(lengthP(fract(u*4./2.)*2.-1.,4.)*16.);}

// Function 41
float replace_pattern(mat2 in_pattern, mat2 out_pattern, vec2 fragCoord,float mag){
    for (int i = 0; i < 2; i++){
    	for(int j = 0; j < 2; j++){
            if(matches_pattern(in_pattern,fragCoord-vec2(i,j),mag)){
            	return out_pattern[1-i][1-j];
            }
        }
    }
    return magnify(fragCoord,mag);
}

// Function 42
float KaroPattern(in vec2 uv)
{
  return 0.5*clamp(10.*sin(PI*uv.x), 0.0, 1.0)
       + 0.5*clamp(10.*sin(PI*uv.y), 0.0, 1.0);
}

// Function 43
float fingerprintPattern(vec2 uv, float ceed){
    float bounds = smoothstep(15., 16.,length(uv * vec2(.55 + (uv.y + 20.) * .004, .5)));

    float a=0.;
    vec2 h = vec2(ceed, 0.);
    for(int i=0; i<100; i++){
        float s=sign(h.x);
        h = hash2(h) * vec2(15., 50.);
    	a += s*atan(uv.x-h.x, uv.y-h.y);
    }
    uv.y += 10.;
    a+=atan(uv.y, uv.x);

    float w = .5;
    float p=(1.-bounds)*w;
    float s = min(.5, p);
    float l = length(uv)+0.319*a;
    float m = mod(l, 2.);
    return 1. - (1.-smoothstep(2.-s,2.,m))*smoothstep(p,p+s,m);
}

// Function 44
float triRosettePattern(vec2 p//https://www.shadertoy.com/view/4lGyz3
){vec2 d=vec2(sqrt(3.),3)/3.
 ;vec4 O=vec4(0)
 ;for(; O.a++ < 4.; O += DumbEnoughToLoopAWallpaperGroup(p) +DumbEnoughToLoopAWallpaperGroup(p += d*.5))p.x+=d.x
 ;return O.x;}

// Function 45
vec3 noisetile(vec2 uv){
    // clamp probably not (and shouldn't be) needed but anyway
    return vec3(clamp(lerpy(uv), 0.0, 1.0));
}

// Function 46
float TileableNoiseFBM(in vec3 p, float numCells, int octaves)
{
	float f = 0.0;
    
	// Change starting scale to any integer value...
    p = mod(p, vec3(numCells));
	float amp = 0.5;
    float sum = 0.0;
	
	for (int i = 0; i < octaves; i++)
	{
		f += TileableNoise(p, numCells) * amp;
        sum += amp;
		amp *= 0.5;

		// numCells must be multiplied by an integer value...
		numCells *= 2.0;
	}

	return f / sum;
}

// Function 47
float cellTile(in vec3 p){

    // Storage for the closest distance metric, second closest and the current
    // distance for comparisson testing.
    //
    // Set the maximum possible value - dot(vec3(.5), vec3(.5)). I think my reasoning is
    // correct, but I have lousy deductive reasoning, so you may want to double check. :)
    vec3 d = (vec3(.75)); 
   
    
    // Draw some overlapping objects (spheres, in this case) at various positions on the tile.
    // Then do the fist and second order distance checks. Very simple.
    d.z = drawSphere(p - vec3(.81, .62, .53));
    d.x = min(d.x, d.z); //d.y = max(d.x, min(d.y, d.z)); // Not needed on the first iteration.
    
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.z = drawSphere(p - vec3(.39, .2, .11));
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);
    
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawSphere(p - vec3(.62, .24, .06));
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);
    
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071; 
    d.z = drawSphere(p - vec3(.2, .82, .64));
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);

     
	// More spheres means better patterns, but slows things down.
    //p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    //d.z = drawSphere(p - vec3(.48, .29, .2));
    //d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);
    
    //p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    //d.z = drawSphere(p - vec3(.06, .87, .78));
    //d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z); 
	

    
    // Returning what I'm hoping is a normalized result. Not that it
    // matters too much, but I'd like it normalized.
    // 2.66 seems to work, but I'll double check at some stage.
    // d.x: Minimum distance. Regular round Voronoi looking.
    // d.y - d.x - Maximum minus minimum, for that beveled Voronoi look.
    //
    return (d.y - d.x)*2.66; 
    //return 1. - d.x*2.66;
    //return 1. - sqrt(d.x)*1.63299; // etc.

    
}

// Function 48
bool dualTileZoneTest(vec5 p , float value){
    bool down  = all(lessThanEqual(vec4(value),p.a)) && value <= p.v && all(lessThanEqual(vec4(value),vec4(1.0)-p.a)) && value <= (1.0-p.v);
    bool up  = all(greaterThanEqual(vec4(value),p.a)) && value >= p.v && all(greaterThanEqual(vec4(value),vec4(1.0)-p.a)) && value >= (1.0-p.v);
    return down || up;
}

// Function 49
float pattern_bg(vec2 p) {
    float d = 0.;
    p = vec2(mod(p.x + 0.01 * (floor((mod(p.y, 0.04) - 0.02) / 0.02)), 0.02) - 0.01, mod(p.y, 0.02) - 0.01);
    d = SS(-0.001, 0.001, sdCircle(p, 0.0035));
    return d;
}

// Function 50
float cellTile(in vec3 p){
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    vec4 d; 
    d.x = drawObject(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawObject(p - vec3(.39, .2, .11));
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawObject(p - vec3(.62, .24, .06));
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071;
    d.w = drawObject(p - vec3(.2, .82, .64));

    // Obtain the minimum, and you're done.
    d.xy = min(d.xz, d.yw);
        
    return min(d.x, d.y)*2.66; // Scale between zero and one... roughly.
}

// Function 51
float SquareHolePattern(in vec2 uv)
{
  float thickness = 0.8;
  float t = cos(uv.x*2.0) * cos(uv.y*2.0) / thickness;
  return smoothstep(0.1, 0.0, t*t);
}

// Function 52
vec4 textureNoTile( sampler2D samp, in vec2 uv )
{
    vec2 iuv = floor( uv );
    vec2 fuv = fract( uv );

#ifdef USEHASH    
    // generate per-tile transform (needs GL_NEAREST_MIPMAP_LINEARto work right)
    vec4 ofa = texture( iChannel1, (iuv + vec2(0.5,0.5))/256.0 );
    vec4 ofb = texture( iChannel1, (iuv + vec2(1.5,0.5))/256.0 );
    vec4 ofc = texture( iChannel1, (iuv + vec2(0.5,1.5))/256.0 );
    vec4 ofd = texture( iChannel1, (iuv + vec2(1.5,1.5))/256.0 );
#else
    // generate per-tile transform
    vec4 ofa = hash4( iuv + vec2(0.0,0.0) );
    vec4 ofb = hash4( iuv + vec2(1.0,0.0) );
    vec4 ofc = hash4( iuv + vec2(0.0,1.0) );
    vec4 ofd = hash4( iuv + vec2(1.0,1.0) );
#endif
    
    vec2 ddx = dFdx( uv );
    vec2 ddy = dFdy( uv );

    // transform per-tile uvs
    ofa.zw = sign(ofa.zw-0.5);
    ofb.zw = sign(ofb.zw-0.5);
    ofc.zw = sign(ofc.zw-0.5);
    ofd.zw = sign(ofd.zw-0.5);
    
    // uv's, and derivarives (for correct mipmapping)
    vec2 uva = uv*ofa.zw + ofa.xy; vec2 ddxa = ddx*ofa.zw; vec2 ddya = ddy*ofa.zw;
    vec2 uvb = uv*ofb.zw + ofb.xy; vec2 ddxb = ddx*ofb.zw; vec2 ddyb = ddy*ofb.zw;
    vec2 uvc = uv*ofc.zw + ofc.xy; vec2 ddxc = ddx*ofc.zw; vec2 ddyc = ddy*ofc.zw;
    vec2 uvd = uv*ofd.zw + ofd.xy; vec2 ddxd = ddx*ofd.zw; vec2 ddyd = ddy*ofd.zw;
        
    // fetch and blend
    vec2 b = smoothstep(0.25,0.75,fuv);
    
    return mix( mix( textureGrad( samp, uva, ddxa, ddya ), 
                     textureGrad( samp, uvb, ddxb, ddyb ), b.x ), 
                mix( textureGrad( samp, uvc, ddxc, ddyc ),
                     textureGrad( samp, uvd, ddxd, ddyd ), b.x), b.y );
}

// Function 53
vec3 subsubtile(in vec2 st, in vec3 primaryColor, in vec3 secondaryColor) {
    vec3 color = vec3(.2);

    // divide into tiles
    float divisions = 3.;
    vec2 mst = st;
    mst *= divisions;
    // calculate index
    float cellx = floor(mst.x);
    float celly = floor(mst.y);
    float colOffset = mod(celly, 3.)-1.;
    int index = int(cellx + celly*divisions);
    // tile
    mst = mod(mst, 1.);

    vec3 edgeColor = vec3(0.3);

    // Determine fill color
    color = primaryColor;
    if (mod(cellx-colOffset, 3.) == 2.) {
        color = secondaryColor;
    }

    // Draw Borders (Top and Left)
    float edge = 0.;
    float br = 0.18;
    if (mst.x < br) {
        edge += 1.;
        if (mod(cellx-colOffset, 3.) == 1.) {
            edge -= 1.;
        }
    }
    if (mst.y > 1.-br) {
        edge += 1.;
        if (mod(cellx-colOffset, 3.) == 2.) {
            if (primaryColor == secondaryColor) {
                edge -= 1.;
            }
        }
    }
    if (edge >=1.) {
        color = edgeColor;
    }

    return color;
}

// Function 54
float subpattern( vec2 gr )
{
	vec2 x = fract(gr);
	float d = 2.0 * length(x - 0.5);
	if (fract(iTime*0.25) < 0.25) {
		d /=  0.5 + fract(2.0 * iTime);
	}
	float e = tri(d);
	float f = 2.0 * abs(e - 0.5);
	float k = selector( 12.0, 3.0 );
	float ani = selector( 5.0, 2.0 );
	if (k == 0.0) {
		return smoothstep(0.49,0.51,fract(x.y)) * mix( 1.0, up_to_bottom(gr), ani );
	} 
	else if (k == 1.0) {
		return smoothstep(0.49,0.51,fract(x.x)) * mix( 1.0, left_to_right(gr), ani );
	}
	else {
		return smoothstep(0.1,0.5,tri(f)) * mix( 1.0, animator(gr), ani );
	}
}

// Function 55
float stars2d_tile(vec2 s, vec2 x, float scale, float theta) {
	float density = star_density(s*3.5/scale);	
	vec4 star = randomStar(s);
	
    if (star.w*1.2 > density) {
		return 0.0;
	}

	float starMagnitude = 0.7 + star.z*2.0;
	float starBrightness = 4.0 - star.z*4.0;
	vec2 v = starMagnitude*rotate2(x - star.xy, -theta);

	/* bright star with beams */
	if (scale <= 8.0) {
		v*=2.0;
		return 4.0*max(0.0, 0.5-smoothstep(0.0, 1.6, pow(dot(v,v), 0.125))) 			      
			     + max(0.0, 0.5-smoothstep(0.0, 1.0, pow(dot(v,v), 0.25)))
			     + max(0.0, 0.6-dot(abs(v), vec2(16.0, 1.0)))  // beam
			     + max(0.0, 0.6-dot(abs(v), vec2(1.0, 16.0))); // beam
	}
	
	/* cheap trick against aliasing */
	float pixels = min(1.0, 24.0/(scale*starMagnitude));
	v *= max(0.6, pixels);
	starBrightness *= pixels*pixels;
	
	float d = pow(dot(v,v), 0.25);
	return starBrightness*max(0.0, 0.5-smoothstep(0.0, 1.0, d));
}

// Function 56
vec3 smoothpattern_norm(in vec3 pos)
{
    const vec3 e = vec3(NORM_EPS, 0., 0.);
    return normalize(vec3(	smoothpattern(pos+e.xyy) - smoothpattern(pos-e.xyy),
                          	smoothpattern(pos+e.yxy) - smoothpattern(pos-e.yxy),
                          	smoothpattern(pos+e.yyx) - smoothpattern(pos-e.yyx) ));
}

// Function 57
vec3 pattern(vec2 p, float sc, float bv) {
	vec3 e = vec3(-1, 0, 1), r = vec3(1e5);
	vec2 ip = floor(p*sc), tileID = e.yy;
	p -= (ip + .5) / sc;

	float
		h11 = .5 * HT(ip + e.yy, vec2(sc)),
		h10 = .5 * HT(ip + e.xy, vec2(sc)),
		h01 = .5 * HT(ip + e.yz, vec2(sc)),
		h12 = .5 * HT(ip + e.zy, vec2(sc)),
		h21 = .5 * HT(ip + e.yx, vec2(sc)),
		h00 = .5 * HT(ip + e.xz, vec2(sc)),
		h02 = .5 * HT(ip + e.zz, vec2(sc)),
		h22 = .5 * HT(ip + e.zx, vec2(sc)),
		h20 = .5 * HT(ip + e.xx, vec2(sc));

	vec2[4] ctr, l;
	if (mod(ip.x + ip.y, 2.) < .5) {
		l[0] = 1. + vec2(h21 - h10, h11 - h20);
		l[1] = 1. + vec2(h12 - h21, h11 - h22);
		l[2] = 1. + vec2(h01 - h10, h00 - h11);
		l[3] = 1. + vec2(h12 - h01, h02 - h11);
		ctr[0] = vec2(h21, h11);
		ctr[1] = vec2(h21, h11);
		ctr[2] = vec2(h01, h11);
		ctr[3] = vec2(h01, h11);
	} else {
		l[0] = 1. + vec2(h11 - h20, h10 - h21);
		l[1] = 1. + vec2(h22 - h11, h12 - h21);
		l[2] = 1. + vec2(h11 - h00, h01 - h10);
		l[3] = 1. + vec2(h02 - h11, h01 - h12);
		ctr[0] = vec2(h11, h10);
		ctr[1] = vec2(h11, h12);
		ctr[2] = vec2(h11, h10);
		ctr[3] = vec2(h11, h12);
	}

	for (int i=0; i<4; i++) {
		ctr[i] += l[i] * (vec2(i&1, i/2) - .5);
		l[i] /= sc;
		float bx = box1(p - ctr[i]/sc, l[i]/2. - bv/sc);
		if (bx < r.x)
			r = vec3(bx, ip + ctr[i]);
	}

	return r;
}

// Function 58
float Tile( vec3 pos, float invWeight, vec4 rand, bool grass )
{
//    if ( rand.w > .2 ) invWeight = 1.; // isolate some of the blades
    
//    invWeight = invWeight*invWeight; // should be 0 until about .5
//    invWeight = smoothstep(.5,1.,invWeight); // should be 0 until about .5

    // vary the height a little to make it look less even
    pos.y += sqrt(rand.w)*.07;
    
    pos.xz += rand.xy;
    float a = rand.z*6.283;
    pos.xz = pos.xz*cos(a) + sin(a)*vec2(-1,1)*pos.zx;
    
    float f = 1e20;
    
    if ( grass )
    {
        f = min(f,min(min(min(min(
            BladeOfGrass(vec3(.01,0,0),vec3(.03,.15,.04),vec2(.06),pos,false),
            BladeOfGrass(vec3(0),vec3(-.05,.17,.02),vec2(.06),pos,false)),
            BladeOfGrass(vec3(0),vec3(-.01,.10,.02),vec2(.04),pos,false)),
            BladeOfGrass(vec3(0,0,-.01),vec3(-.01,.12,-.03),vec2(.03),pos,false)),
            BladeOfGrass(vec3(.005,0,0),vec3(.03,.16,-.05),vec2(.04),pos,false)
        )) + (1.-invWeight)*.0;
    }
    
    // flowers
    f = min(f, Flower(vec3(.1,0,0),vec3(.1,.2,.05),vec2(.13),pos,grass) + (1.-invWeight)*.0 );
    
    return mix( max(.03,pos.y-.2), f, invWeight );
//    return f;
}

// Function 59
vec3 VorotilesAA( vec2 uv )
{
	#define SAMPLING_STRENGTH 10000000000.0
	#define NB_SAMPLES 3 //0: no anti-aliasing
	
	if (NB_SAMPLES == 0)
	{
		return Vorotiles( uv );
	}
	else
	{
		// calc texture sampling footprint		
		vec2 ddx = dFdx( uv ); 
		vec2 ddy = dFdy( uv ); 
	
		int sx = 1 + int( clamp( SAMPLING_STRENGTH*length(ddx), 0.0, float(NB_SAMPLES-1) ) );
		int sy = 1 + int( clamp( SAMPLING_STRENGTH*length(ddy), 0.0, float(NB_SAMPLES-1) ) );

		vec3 no = vec3(0.0);

		for( int j=0; j<NB_SAMPLES; j++ )
		for( int i=0; i<NB_SAMPLES; i++ )
		{
			if( j<sy && i<sx )
			{
				vec2 st = vec2( float(i), float(j) ) / vec2( float(sx),float(sy) );
				no += Vorotiles( uv + st.x*ddx + st.y*ddy );
			}
		}

		return no / float(sx*sy);
	}
}

// Function 60
float pattern3(vec2 uv)
{
    // distance from center
    // NOTE: 0.499 is to avoid the infinity in mid-Y,
    // which, for our checkerboard calculation, produces NaN,
    // which shows as black (fully visible during transitions!)   
    vec2 dCenter = vec2(0.5, 0.499) - uv.xy;
    
    float X_INV_SCALE = 1.0;
    float Z_INV_SCALE = 0.5;
    
    // 3D perspective: 1/Z = constant
    vec3 cam;
    cam.z = 1.0 / dCenter.y;
    cam.xy = vec2(
        X_INV_SCALE * dCenter.x,
        Z_INV_SCALE)
         * cam.z;

    // rotate
    float angle = (iTime * 0.05) 
        * float(uv.y < 0.5); // only allow the ground to rotate, not the ceiling
    cam.xy = rotateXY( cam.xy, angle );

    // textured
	cam.xy = floor(cam.xy * 2.0); 
    return mod(cam.x+cam.y, 2.0);
}

// Function 61
vec3 textureNoTile( sampler2D samp, in vec2 uv, float v )
{
    vec2 p = floor( uv );
    vec2 f = fract( uv );
	
    // derivatives (for correct mipmapping)
    vec2 ddx = dFdx( uv );
    vec2 ddy = dFdy( uv );
    
	vec3 va = vec3(0.0);
	float w1 = 0.0;
    float w2 = 0.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec2 g = vec2( float(i),float(j) );
		vec4 o = hash4( p + g );
		vec2 r = g - f + o.xy;
		float d = dot(r,r);
        float w = exp(-5.0*d );
        vec3 c = textureGrad( samp, uv + v*o.zw, ddx, ddy ).xyz;
		va += w*c;
		w1 += w;
        w2 += w*w;
    }
    
    // normal averaging --> lowers contrasts
    //return va/w1;

    // contrast preserving average
    float mean = 0.3;// textureGrad( samp, uv, ddx*16.0, ddy*16.0 ).x;
    vec3 res = mean + (va-w1*mean)/sqrt(w2);
    return mix( va/w1, res, v );
}

// Function 62
vec3 tile(vec3 p) {
    return abs(mod(p, 2.) - 1.); // - vec3(1.);
}

// Function 63
vec4 shapeTile(vec2 U) 
{
    vec4 O = vec4(0);
    vec2 A = 0.5 - abs(U);                   // coords from borders 
    vec2 B  = A * 2. / BEVEL;                   // coords in bevel
    float m = min(B.x,B.y);                 // in bevel if [0,1]
    if (A.x<ROUND && A.y<ROUND)                 // round edges
    {
        m = (ROUND-length(ROUND-A)) *2./dot(BEVEL,normalize(ROUND-A));    
    }
    return vec4(clamp(m,0.,1.));
}

// Function 64
float BrickPattern(in vec2 p)
{
  p *= vec2 (1.0, 2.8);  // scale
  vec2 f = floor (p);
  if (2. * floor (f.y * 0.5) != f.y)
    p.x += 0.5;  // brick shift
  p = smoothstep (0.03, 0.08, abs (fract (p + 0.5) - 0.5));
  return 1. - 0.9 * p.x * p.y;
}

// Function 65
vec3 largetile(in vec2 st) {
    vec3 color = vec3(0.5);
    
    // require a 3 row space (in addition to the 9x9=81) for grey border
    st *= 84./81.;
    st.x -= 3./84.;

    // divide into tiles
    float divisions = 9.;
    vec2 mst = st;
    mst *= divisions;
    // calculate index
    float cellx = floor(mst.x);
    float celly = floor(mst.y);
    float colOffset = mod(celly, 3.)-1.;
    int index = int(cellx + celly*divisions);
    // tile
    mst = mod(mst, 1.);

    vec3 red = vec3(0.99, 0.2,0.2);
    vec3 white = vec3(0.97);
    vec3 grey = vec3(0.8);
    vec3 black = vec3(0.05);

    /*
    vec3 primaryColor = white;
    vec3 secondaryColor = white;

    if (mod(celly, 2.) == 1.) {
        primaryColor = red;
    }
    if (mod(cellx, 2.) == 1.) {
        secondaryColor = red;
    }

    if (celly == 0. || celly == 1. || celly == 7. || celly == 8.) {
        primaryColor = black;
    }
    if (cellx == 0. || cellx == 1. || cellx == 7. || cellx == 8.) {
        secondaryColor = black;
    }

    if (celly == 9.) {
        primaryColor = grey;
    }
    if (cellx == -1.) {
        secondaryColor = grey;
    }
	*/
    
    // More Compact Assignment of Colors by @FabriceNeyret2
    vec3 primaryColor = 
        celly == 9. ? grey
       		: celly == 0. || celly == 1. || celly == 7. || celly == 8. ? black
        		: mod(celly, 2.) == 1. ? red
        			: white;
    
    vec3 secondaryColor = 
        cellx == -1. ? grey
            : cellx == 0. || cellx == 1. || cellx == 7. || cellx == 8. ? black
                : mod(cellx, 2.) == 1. ? red
                	: white;

    // Create Subtile
    color = subtile(mst, primaryColor, secondaryColor);

    return color;
}

// Function 66
vec3 DomainRepeatXZGetTile( const in vec3 vPos, const in vec2 vRepeat, out vec2 vTile )
{
    vec3 vResult = vPos;
    vec2 vTilePos = (vPos.xz / vRepeat) + 0.5;
    vTile = (vTilePos + 1000.0);
    vResult.xz = (fract(vTilePos) - 0.5) * vRepeat;
    return vResult;
}

// Function 67
float SquareHolePattern(vec2 u
){u.x=mu(sin(u*2.))
 ;return smoothstep(.1,.0, sq2(u.x)*2.5);}

// Function 68
float pattern(float d, float pOffset) { return sin(d*10.+pOffset); }

// Function 69
vec2 tilerot (vec2 p)
    
{
    //From the book of shaders https://thebookofshaders.com/10/
    vec2 ipos = floor(p*scale);   
     //-0.5 to get -0.5 to 0.5 for rotation
    vec2 fpos = fract(p*scale)-0.5; 
    float index = rnd(ipos);
        
    if(index >= 0.75){ p = fpos*rot(PI/2.);}
    else if (index >= 0.50){ p = fpos*rot(-PI/2.);}
    else if (index >= 0.25){ p = fpos*rot(PI);}
    else p = fpos;//<---need this!
   
 //+0.5 to reverse the earlier process so you for 0.- 1 again
 return p+0.5;   
}

// Function 70
vec3 ComputeLightPattern(vec3 Ln)
{
    Ln =  lightRotate*Ln,
        
    Ln = abs(Ln);
    float a = max(Ln.x, max(Ln.y, Ln.z));
    vec3 c = a == Ln.x ? colorC : a == Ln.y ? colorA : colorB;
    return c * pow(a, lightPatternPower);
}

// Function 71
float df_pattern(vec2 uv)
{
    float l1 = sharpen(CIRCLE(uv, vec2(0), .5), WGHT);
    float l2 = sharpen(CIRCLE(uv, vec2(1), .5), WGHT);
    return max(l1,l2);
}

// Function 72
float sd_TechTilesTestsSub0( vec3 p, int lod, float t, Ray ray, vec2 index, TechTilesArgs0 targs )
{
	float d = FLT_MAX;

//	d = opI( p.z - 1.0, sd_bounds_range( p.xy, vec2( 0, 0 ), vec2( 1, 1 ) ) );
//	return d;

	float e0 = 0.0125 * 2.0;
	float e = e0 + t * 0.001; // else e becomes 0 as far as tracing is concerned... increases cost

	TechTilesArgs args;

	vec4 ha = hash42( index );
	vec4 hb = hash42( index + 100.0 );

	float rnd_type_and_rotation = ha.w;
	vec3 size0_hash = ha.xyz;
	vec4 height0_hash = hb;

	args.sub10 = rnd_type_and_rotation < 0.6;
	args.sub11 = rnd_type_and_rotation < 0.3;

	float rota = fract( rnd_type_and_rotation * 3.0 );
	if ( rota < 0.25 ) p.xy = p.yx;
	else if ( rota < 0.5 ) p.xy = vec2( 1.0 - p.y, p.x );

	float m1 = 0.15;
	args.size0 = m1 + ( 1.0 - m1 * 2.0 ) * size0_hash; // hash32 expensive

	args.size10 = vec3( 0.25, 0.5, 0.75 );
	args.height10 = vec4( 1.0 );
//	args.height10 = hash42( index + 80.0 ) * 0.25; // don't hash all... leave splits is interesting too

	args.size11 = vec3( 0.25, 0.5, 0.5 );
	args.height11 = vec4( 1.0 );
//	args.height11 = hash42( index + 85.0 ) * 0.25;

	args.height0 = mix( vec4( targs.hmin ), vec4( targs.hmax ), height0_hash );

	args.height10 = args.height0 + targs.hdetail * args.height10;
	args.height11 = args.height0 + targs.hdetail * args.height11;

	d = sd_TechTilesTestsSub( p, lod, t, args, e );

	// bevel
	d = opI( d, dot( p - vec3( 0, 0, 0.1 ), vec3( -V45.x, 0, V45.y ) ) );
	d = opI( d, dot( p - vec3( 0, 0, 0.1 ), vec3( 0, -V30.x, V30.y ) ) );

	return d;
}

// Function 73
float cellTile(in vec3 p){
    
     
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    vec4 v, d; 
    d.x = drawObject(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y - p.x, p.y + p.x)*.7071;
    d.y = drawObject(p - vec3(.39, .2, .11));
    p.yz = vec2(p.z - p.y, p.z + p.y)*.7071;
    d.z = drawObject(p - vec3(.62, .24, .06));
    p.xz = vec2(p.z - p.x, p.z + p.x)*.7071;
    d.w = drawObject(p - vec3(.2, .82, .64));

    v.xy = min(d.xz, d.yw), v.z = min(max(d.x, d.y), max(d.z, d.w)), v.w = max(v.x, v.y); 
   
    d.x =  min(v.z, v.w) - min(v.x, v.y); // First minus second order, for that beveled Voronoi look. Range [0, 1].
    //d.x =  min(v.x, v.y); // Minimum, for the cellular look.
        
    const float scale = 2.;
    return min(d.x*2.*scale, 1.); // Normalize.
    
}

// Function 74
float cellTile(in vec3 p){
    
    vec4 d;
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    d.x = drawObject(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawObject(p - vec3(.39, .2, .11));
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawObject(p - vec3(.62, .24, .06));
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071;
    d.w = drawObject(p - vec3(.2, .82, .64));

    d.xy = min(d.xy, d.zw); // Minimum distance determination.
    
    return 1.- min(d.x, d.y)*.166; // Normalize... roughly.
    
}

// Function 75
vec2 tile(vec2 uv, vec2 dimensions)
{
    return mod(uv, dimensions) - dimensions / 2.0;
}

// Function 76
vec3 pattern(vec2 uv) {
	
    // The coordinates for the two rectangles
	vec2 uv1 = uv*uvRotate(radians(30.));
	vec2 uv2 = uv*uvRotate(radians(-30.));
	
    // The signed distance functions
	float sdfr1 = sdfRect(uv1,vec2(.1,.7));
	float sdfr2 = sdfRect(uv2,vec2(.1,.7));
	
    // A fill to keep track of their areas and masks
	float r1 = fill(sdfr1,1.);
	float r2 = fill(sdfr2,1.);

	float r1mask = 1.-r1;
	float r2mask = 1.-r2;

	// Two waves, they are nothing more than the difference between two sine waves
	float wave1 = r1 * max(fill(0.05*sin((uv1.y+.5)*TAU+1.57)-uv1.x,0.),
						   fill(uv1.x-0.05*sin((uv1.y+.5)*TAU),0.));
	
	float wave2 = r1mask * r2 * max(fill(0.05*sin((uv2.y+.5)*TAU+1.57)-uv2.x,0.),
									fill(uv2.x-0.05*sin((uv2.y+.5)*TAU),0.));
	// The background
	vec3 bg = pal(.5-uv.y*.1);
    // Three circles to make the center flower
    float circle = length(uv-vec2(.0,.4));
    bg =  mix(bg, pal(.0), smoothstep(0.4,.0,circle) );
	bg =  mix(bg, pal(.5), smoothstep(0.11,.0,circle) );
	bg =  mix(bg, pal(.9), smoothstep(0.02,.0,circle) );
	
 	// Composing the rectangles and the waves to set up the foreground
	float d =  max(min(max(r1mask*r2,r1),wave1),wave2);
	
    // Colorizing the foreground
	vec3 fg = mix(pal(.9-uv.y*2.),pal(.15+uv.y*.1),d);
	// Adding a black contour to the rectangles 
    fg = mix(fg,vec3(.0),max(r1mask*fill(abs(sdfr2),1.),fill(abs(sdfr1),1.)));
	// Adding a faux 3d to the interlace of the rectangles
    fg = mix(fg,fg*.4,r2*smoothstep(.0,.01,sdfr1)-smoothstep(.0,.1,sdfr1));
	
    // return foreground and background
    return mix(fg,bg,min(r1mask,r2mask));
}

// Function 77
vec4 dirTile(vec2 pos)
{
    float tileSize=5.;
    vec2 qpos=floor(pos/tileSize)*tileSize+tileSize*.5;
    vec2 qrand=texture(iChannel1,qpos/iChannelResolution[1].xy).xy*tileSize*.707;
    vec2 v=getBohm_dQdt(qpos);
    vec2 vn=v.yx*vec2(1,-1);
    vec2 dir=normalize(vn);
    //dir=vec2(1,0);
    float b=sin(1.5*dot(pos-qpos+qrand,1.*dir));
    return vec4(b*b);
}

// Function 78
vec2 doPattern( in vec2 p, in vec4 t )
{
    vec2 z = p;
    vec2 s = vec2(0.0);
    for( int i=0; i<100; i++ ) 
    {
        z = iterate( z, t );

        float d = dot( z-p, z-p ); 
        s.x += abs(p.x-z.x);
        s.y = max( s.y, d );
    }
    s.x /= 100.0;
	return s;
}

// Function 79
float SinePattern(in vec2 p)
{
  return sin(p.x * 20.0 + cos(p.y * 12.0 ));
}

// Function 80
vec3 Pattern( vec2 uv )
{
    // striped patterns

    // trans flag
    vec3 colours[] = vec3[]( vec3(.3,.7,1), vec3(1,.2,.4), vec3(1), vec3(1,.2,.4), vec3(.3,.7,1) );

    // lesbian
    //vec3 colours[] = vec3[]( vec3(.8,.05,.0), vec3(1,.3,.0), vec3(1,.5,.2), vec3(1), vec3(1,.3,.6), vec3(.7,.1,.4), vec3(.5,.0,.2) );
    
    float smoothidx = (1.-uv.y)*float(colours.length()) + .5;
    int idx = int(floor( smoothidx ));
    float fidx = smoothidx - float(idx);
    fidx -= .5;
    
    return mix(colours[max(0,idx-1)],colours[idx],
                //step(.0,fidx)); // aliased
                 smoothstep( -fidx, 1.-fidx, fidx/max(fwidth(smoothidx),.0001) ));// anti-aliased

/*
    // checker
    float pattern = (fract(uv.x/.2)-.5)*(fract(uv.y/.3)-.5);
    return mix( vec3(.03), vec3(1), smoothstep( -fwidth(pattern)*.5, fwidth(pattern)*.5, pattern ) ); // this antialiasing doesn't work
*/
}

// Function 81
vec2 tile (vec2 _st, float _zoom) {
    _st *= _zoom;
    _st.x += step(1., mod(_st.y,2.0)) * 0.5;
    return fract(_st);
}

// Function 82
float PatternCircles(vec2 p,float m//giraffe bubbles of
){p.x-=m*.5*step(0.,sin(pi*p.y/m)) //https://www.shadertoy.com/view/MsSyRz
 ;p=mod(p,m)-m*.5
 ;return 1.-sm(0.,(p.x*p.x+p.y*p.y)-1.);}

// Function 83
vec3 pattern(vec2 p){
    
    
    // Grid ID and local coordinates.
    vec2 ip = floor(p);
    p -= ip + .5;
    
    
    // Distance field container. One for the lines, one for the larger objects,
    // and another for the smaller objects rendered over the top.
    vec3 d = vec3(1e5);
    
    
    // Render some boxes with various sizes in a checkboard fashion.
    float w = (mod(ip.x + ip.y, 2.)>.5)? .28 : .44;
    float s = dist(p, vec2(w));
    d.y = min(d.y, s);
    
    #ifdef FRAMES
    float ody = d.y;
    d.y = max(d.y, -(d.y + .15)); // Picture frame variation.
    #endif
    
    
    // Randomly offset the smaller boxes.
    vec2 rnd = hash22(ip);
    // Random offset factor.
    const float rF = .125;
    
    // Smaller boxes over the top of the larger boxes.
    if(mod(ip.x + ip.y, 2.)<.5){
        
        w = .2;
        s = dist(p - rnd*rF, vec2(w));

        d.z = min(d.z, s);
        #ifdef FRAMES
        d.z = max(d.z, -(d.z + .15)); // Picture frame variation.
        //d.z = max(d.z, -(d.z + .18)); 
        #endif
        
    }
    
    // Line thickness.
    float ew = .011;
       
    
    // Render some lines. It looks more difficult than it is. Basically, read
    // the offset position to each of the four cell neighbors, then render a
    // line between them. There's a maze variation in there too.
    if(mod(ip.x + ip.y, 2.)<.5){
        
        #ifdef MAZELINES
        float rnd3L = hash21(ip + vec2(-1, 0));
        float rnd3T = hash21(ip + vec2(0, 1));
        float rnd3R = hash21(ip + vec2(1, 0));
        float rnd3B = hash21(ip + vec2(0, -1));
        
        if(rnd3L<.5) d.x = min(d.x, lBox(p, vec2(-1, 0), rnd*rF, ew/2.));
        if(rnd3T>=.5) d.x = min(d.x, lBox(p, vec2(0, 1), rnd*rF, ew/2.));
        if(rnd3R<.5) d.x = min(d.x, lBox(p, vec2(1, 0), rnd*rF, ew/2.));
        if(rnd3B>=.5) d.x = min(d.x, lBox(p, vec2(0, -1), rnd*rF, ew/2.)); 
        #else
        #ifndef PARTIAL_LINES
        d.x = min(d.x, lBox(p, vec2(-1, 0), rnd*rF, ew/2.));
        d.x = min(d.x, lBox(p, vec2(0, 1), rnd*rF, ew/2.));
        d.x = min(d.x, lBox(p, vec2(1, 0), rnd*rF, ew/2.));
        d.x = min(d.x, lBox(p, vec2(0, -1), rnd*rF, ew/2.));
        #endif
        #endif
        
        
    }
    else {
        
        vec2 rndL = hash22(ip + vec2(-1, 0));
        vec2 rndT = hash22(ip + vec2(0, 1));
        vec2 rndR = hash22(ip + vec2(1, 0));
        vec2 rndB = hash22(ip + vec2(0, -1));

        
        #ifdef MAZELINES
        float rnd3 = hash21(ip);
        if(rnd3<.5){
            d.x = min(d.x, lBox(p, vec2(-1, 0) + rndL*rF, vec2(0), ew/2.));
            d.x = min(d.x, lBox(p, vec2(1, 0) + rndR*rF, vec2(0), ew/2.));
            
        }    
        else {
            d.x = min(d.x, lBox(p, vec2(0, 1) + rndT*rF, vec2(0), ew/2.));
            d.x = min(d.x, lBox(p, vec2(0, -1) + rndB*rF, vec2(0), ew/2.));
        }
        #else
        d.x = min(d.x, lBox(p, vec2(-1, 0) + rndL*rF, vec2(0), ew/2.));
        d.x = min(d.x, lBox(p, vec2(1, 0) + rndR*rF, vec2(0), ew/2.));
        d.x = min(d.x, lBox(p, vec2(0, 1) + rndT*rF, vec2(0), ew/2.));
        d.x = min(d.x, lBox(p, vec2(0, -1) + rndB*rF, vec2(0), ew/2.));
        #endif
        
        
    }
      
    // Straight lines.
    //d.x = min(d.x, lBox(p, vec2(0, -.5), vec2(0, .5), ew/2.));
    //d.x = min(d.x, lBox(p, vec2(-.5, 0), vec2(.5, 0), ew/2.)); 
    
    // Cut away the lines from the middle of the frame. You don't have to,
    // but I prefer it.
    #ifdef FRAMES
    d.x = max(d.x, -ody);
    #endif
    
    // Set the global ID to the cell object ID. Hacky coding --- There are
    // better ways to do this. :)
    gID = ip;
    
    // Return the distance functions.
    return d;
}

// Function 84
float linePattern(vec2 p, vec2 a, vec2 b){
  
    // Determine the angle between the vertical 12 o'clock vector and the edge
    // we wish to decorate (put lines on), then rotate "p" by that angle prior
    // to decorating. Simple.
    vec2 v1 = vec2(0, 1);
    vec2 v2 = (b - a);

    if(a.x>b.x) v2.y = -v2.y;

    // Angle between vectors.
    //float ang = acos(dot(v1, v2)/(length(v1)*length(v2))); // In general.
    float ang = acos(v2.y/length(v2)); // Trimed down.
    p = rot2(ang - .2)*p; // Putting the angle slightly past 90 degrees is optional.

    float ln = clamp(cos(p.y*64.*2.)*1. - .5, 0., 1.);

    return ln*.25 + clamp(sin(p.y*64.)*3. + 2.95, 0., 1.)*.75 + .15; // Ridges.
 
}

// Function 85
vec3 pattern(vec2 uv, vec4 v, vec3 k)
{
	float a = atan(uv.x, uv.y)/3.14159*floor(k.y);
	float r = length(uv)*4.;
	uv = vec2(a,r);
	uv.x *= floor(uv.y)-k.x;
	uv.x += iTime ;
	vec3 color = mix(startColor, endColor, vec3(floor(uv.y)/6.));
	uv = abs(fract(uv)-0.5);
	float x = uv.x*v.x;
	float y = uv.y*v.y;
	float z = uv.y*v.z;
	return color / (abs(max(x + y,z) - v.w)*k.z);
}

// Function 86
vec3 Pattern( vec2 uv )
{
	if ( view_Index == 0 )
	{
		// nyan cat
		float frame = 0.0;//floor(iTime*8.0)
		vec4 t = texture( iChannel1, uv*vec2(1.0/6.4,-1)+vec2(fract(frame/6.0)*.938,1) );
		float f = uv.y*5.5+1.9; //*tau
		return mix( vec3(cos(f),-sin(f),-cos(f))*.5+.5, ToLinear(t.rgb), t.a );
	}
	
	if ( view_Index == 1 )
	{
		// gay pride (roughly)
		uv.y = floor(uv.y*6.0)/6.0+.3;
		return vec3(cos(uv.y*tau),-sin(uv.y*tau),-cos(uv.y*tau))*.5+.5;
	}

	if ( view_Index == 2 )
	{
		// Sweden (because it's easy)
		return mix( vec3(1,.6,0), vec3(.02,.1,.5), smoothstep(.095,.1,min(abs(uv.x-.4),abs(uv.y-.5))) );
	}
	
//	if ( view_Index == 3 )
	{
		// union jack
		vec3 b = vec3(0,0,.5);
		vec3 w = vec3(1);
		vec3 r = vec3(.8,0,0);
		vec3 col = b;
		
		uv = uv*2.0-1.0;
		col = mix( w, col, smoothstep( .245,.255, min(abs(uv.y-uv.x-.05),abs(uv.y+uv.x-.05)) ) );
		col = mix( r, col, smoothstep( .095,.105, min(abs(uv.y-uv.x),abs(uv.y+uv.x)) ) );
	
		float q = min(abs(uv.x*1.5),abs(uv.y));
		col = mix( w, col, smoothstep( .245,.255, q ) );
		col = mix( r, col, smoothstep( .145,.155, q ) );
		
		return col;
	}
}

// Function 87
float SinePatternCrissCross(vec2 p){return .5+sinePattern(p)*sinePattern(p.yx);}

// Function 88
vec3 Vorotiles(vec2 posSample)
{
    vec2 uv = posSample.xy;
	
	vec2 p = posSample.xy;
	p.x *= iResolution.x / iResolution.y;
    
    vec3 color = vec3(0.0,0.0,0.0);
    float distance2border = 0.0;
	vec2 featurePt = vec2(0.0,0.0);
    float density = AnimateDensity();
	bool noTiles = false;
    color = VoronoiColor(density, uv, distance2border, featurePt, noTiles);
    color += vec3(0.1);

    // Make tiles'borders cleaner
    if (noTiles == false)
		color = mix( vec3(0.0,0.0,0.0), color, smoothstep( 0.0, 0.1, distance2border ) );
        
    // Set the final fragment color.
	return color;
}

// Function 89
vec3 pattern3(vec2 uv)
{
   // uv*=2.;
    float wobble = (sin(iTime*4.-0.79)/3.);//fract(-iTime)/400.;//*0.05*sin(20.*iTime);
    float smoothT = iTime/1. ;//+ sin(iTime)/16.;
    
    vec2 st = uv; ;//for vig later
        uv*=1.25;
     uv.x+=0.75;
    uv.y+=0.75;   

    vec2 sv =uv;//prot(PI*floor(uv.x))*uv;
    
    //adding teh bumpy pebble texture for the red squares
    vec3 col = vec3(pow(texture(iChannel2, sv).x, 3.))*1.;
											 //-sin(iTime/1.)/50. //add this to uv/2. of ichannel0 for movement
    
    //adding the moldy wood? texture for the black squares
    col = mix(col, vec3(1.0, 0.0, .0)*texture(iChannel0,sv/2.+vec2(0., smoothT+wobble)).x, vec3(chess(uv, 2.0))  );
    
    //adding the frame, this one required a lot of hacking
    
    col = mix(col, vec3(texture(iChannel1, uv*0.01+0.75)/1.2+0.1)
              //the sin portion is for the light movement
              +abs(sin(uv.x*3.)),  
              //this calles the stripes function
              vec3(stripes(uv, 2., 1.)) )

        -step(0.98, fract(2.*uv.x))/3.-step(0.98, fract(2.*uv.y))/3.;
    
    
    //This is for vignetting
     st *=  1.0 - st.yx;
    float vig = st.x*st.y*15.;
    vig = pow(vig, 0.09);
    
 return clamp(col, 0., 1.);
    
}

// Function 90
vec3 HexPattern( vec2 vUV, vec3 colInner, vec3 colEdge )
{
    // uncomment to actually scroll :P
    //vUV.y -= iTime * 0.01;
    
    vec4 hex = Hexagon( vUV );
    
    float edgeShade = step( fract(hex.w), 0.5 );

    vec3 col = colInner; 
    col = mix( col, vec3(edgeShade), step(hex.x, 0.3) ); // black / white edge
    col = mix( col, colEdge, step(hex.x, 0.15) ); ; // Yellow Surround
        
    return col;
}

// Function 91
bool get_tile(ivec2 co) {
    bool up = rand(co + ivec2(0, 1)) > 0.5;
    bool down = rand(co + ivec2(0, -1)) > 0.5;
    bool left = rand(co + ivec2(-1, 0)) > 0.5;
    bool right = rand(co + ivec2(1, 0)) > 0.5;
    
    bool here = rand(co) > 0.5;
    
    return !((up == down) && (left == right) && (up != right)) && here;
}

// Function 92
float pattern1( vec2 uv )
{    
    // full circle
    vec2 p = (2.*uv - 1.) / CIRCLE_PERCENTAGE_OF_SCREEN;
   
    // quick semi-distance to circle formula:
    float g = dot( p, p );
    
    float insideCircle = float(
        ((g < 1.0 ) && (g > 0.85 )) ||
        ((g < 0.6 ) && (g > 0.5  )) ||
        ((g < 0.2 ) && (g > 0.1  ))
    );
    
    float insideSpokes = float(mod(atan(p.y, p.x) + iTime / 40., PI/8.) < 0.15);

    return 
    	mod(insideCircle + 
            insideSpokes * (1. - g), 
            1.333);
}

// Function 93
float cellTile(in vec3 p){
   
    vec4 d; 
    
    // Plot four objects.
    d.x = drawObject(p - vec3(.81, .62, .53));
    p.xy *= rM;
    d.y = drawObject(p - vec3(.6, .82, .64));
    p.yz *= rM;
    d.z = drawObject(p - vec3(.51, .06, .70));
    p.zx *= rM;
    d.w = drawObject(p - vec3(.12, .62, .64));

	// Obtaining the minimum distance.
    d.xy = min(d.xz, d.yw);
    
    // Normalize... roughly. Trying to avoid another min call (min(d.x*A, 1.)).
    return  min(d.x, d.y)*2.5;
    
}

// Function 94
float sd_TechTilesTestsSub( vec3 p, int lod, float t, TechTilesArgs args, float e )
{
	vec4 d = vec4( FLT_MAX );
	vec4 heights = args.height0;

//	Split4 b = sd_Split_b_xxx( p.xy, vec2( 0, 0 ), vec2( 1, 1 ), args.size0 );
//	Split4 b = sd_Split_b_xyy( p.xy, vec2( 0, 0 ), vec2( 1, 1 ), args.size0 );
	Split4 b = sd_Split_b_xyx( p.xy, vec2( 0, 0 ), vec2( 1, 1 ), args.size0 );
//	Split4 b = sd_Split_b_H( p.xy,vec2( 0, 0 ), vec2( 1, 1 ), args.size0 );

	d = get_distances( p.xy, b ) + e;

#ifdef SPLIT4_BOUNDS
	if ( args.sub10 )
	{
		// do one more level
		Split4 b2 = sd_Split_b_xyy( p.xy, b.b01.pmin, b.b01.pmax, args.size10 );
		vec4 d2 = get_distances( p.xy, b2 ) + e;
		d.y = getDist4( p.z, d2, args.height10 );
		heights = max( heights, args.height10 );
	}
#endif

#ifdef SPLIT4_BOUNDS
	if ( args.sub11 )
	{
		// do one more level
		Split4 b2 = sd_Split_b_xxx( p.xy, b.b11.pmin, b.b11.pmax, args.size11 );
		vec4 d2 = get_distances( p.xy, b2 ) + e;
		d.w = getDist4( p.z, d2, args.height11 );
		heights = max( heights, args.height11 );
	}
#endif

	return getDist4( p.z, d, heights );
}

// Function 95
vec4 pattern( vec2 p )
{
    float aspect = iResolution.x/iResolution.y;
    float p0 = step(abs(p.x-0.125), 0.01) * step(abs(p.y-0.27), 0.01);
    float p1 = step( length( p-vec2(0.125, 0.45) ), 0.025 );
    
    float p2_0 = step( length( p-vec2(0.08, 0.14) ), 0.0125 );
    float p2_1 = step( length( p-vec2(0.16, 0.125) ), 0.0125 );
    float p2_2 = step( length( p-vec2(0.1, 0.07) ), 0.0125 );
    float p2 = max(p2_0, max(p2_1,p2_2));
    
    return vec4( max( p0, max(p1,p2) ) );
}

// Function 96
float TileableNoise(in vec3 p, in float numCells )
{
	vec3 f, i;
	
	p *= numCells;

	
	f = fract(p);		// Separate integer from fractional
    i = floor(p);
	
    vec3 u = f*f*(3.0-2.0*f); // Cosine interpolation approximation

    return mix( mix( mix( dot( Hash( i + vec3(0.0,0.0,0.0), numCells ), f - vec3(0.0,0.0,0.0) ), 
                          dot( Hash( i + vec3(1.0,0.0,0.0), numCells ), f - vec3(1.0,0.0,0.0) ), u.x),
                     mix( dot( Hash( i + vec3(0.0,1.0,0.0), numCells ), f - vec3(0.0,1.0,0.0) ), 
                          dot( Hash( i + vec3(1.0,1.0,0.0), numCells ), f - vec3(1.0,1.0,0.0) ), u.x), u.y),
                mix( mix( dot( Hash( i + vec3(0.0,0.0,1.0), numCells ), f - vec3(0.0,0.0,1.0) ), 
                          dot( Hash( i + vec3(1.0,0.0,1.0), numCells ), f - vec3(1.0,0.0,1.0) ), u.x),
                     mix( dot( Hash( i + vec3(0.0,1.0,1.0), numCells ), f - vec3(0.0,1.0,1.0) ), 
                          dot( Hash( i + vec3(1.0,1.0,1.0), numCells ), f - vec3(1.0,1.0,1.0) ), u.x), u.y), u.z );
}

// Function 97
float cellTile2(in vec3 p){
    
    float c = .25; // Set the maximum.
    
    c = min(c, drawSphere(p - vec3(.81, .62, .53)));
    c = min(c, drawSphere(p - vec3(.39, .2, .11)));
    
    c = min(c, drawSphere(p - vec3(.62, .24, .06)));
    c = min(c, drawSphere(p - vec3(.2, .82, .64)));
    
    p *= 1.4142;
    
    c = min(c, drawSphere(p - vec3(.48, .29, .2)));
    c = min(c, drawSphere(p - vec3(.06, .87, .78)));

    c = min(c, drawSphere(p - vec3(.6, .86, .0)));
    c = min(c, drawSphere(p - vec3(.18, .44, .58)));
        
    return (c*4.);
    
}

// Function 98
float bloomTile(float lod, vec2 offset, vec2 uv){
    return texture(iChannel1, uv * exp2(-lod) + offset).a;
}

// Function 99
float QCirclePattern(in vec2 p)
{
  vec2 p2 = mod(p*8.0, 4.0)-2.0;
  return sin(lengthN(p2, 4.0)*16.0);
}

// Function 100
float brickPattern(vec2 c) {
	float o = 1.;
    if (mod(c.y, 4.) < 1.) o = 0.;
    if (mod(c.x - 4. * step(4., mod(c.y, 8.)), 8.) > 7.) o = 0.;
    return o;
}

// Function 101
vec2 pattern7(vec2 uv)
{
    float time = iTime;
    float a = atan( uv.x, uv.y);
    float r = sqrt(dot(uv.x,uv.x));
    float u = 0.9/(r+0.5*uv.x); // Looks cool, but a bit smashed, i'm doing it wrong!
    float v = 0.9/(r+0.5*uv.x);
	return vec2(u,v);
}

// Function 102
void tile(in vec3 p, out vec3 sp, out vec3 tp, out vec3 rp, out float mul)
{
	// Apply the forward log-spherical map
	float r = length(p);
	p = vec3(log(r), acos(p.y / length(p)), atan(p.z, p.x));

	// Get a scaling factor to compensate for pinching at the poles
	// (there's probably a better way of doing this)
	float xshrink = 1.0/(abs(p.y-M_PI)) + 1.0/(abs(p.y)) - 1.0/M_PI;
	p.y += height;
	p.z += p.x * 0.3;
	mul = r/lpscale/xshrink;
	p *= lpscale;
	sp = p;

	// Apply rho-translation, which yields zooming
	p.x -= rho_offset + gTime;
	
	// Turn tiled coordinates into single-tile coordinates
	p = fract(p*0.5) * 2.0 - 1.0;
	p.x *= xshrink;
	tp = p;
	pR(p.xy, rot_XY);
	pR(p.yz, rot_YZ);
	rp = p;
}

// Function 103
float TrianglePattern(vec2 u){return step(su(fract(toTri(u))),0.);}

// Function 104
vec3 Tile(vec3 p){vec3 a=vec3(8.0);return abs(mod(p,a)-a*0.5)-a*0.25;}

// Function 105
vec4 getpattern(vec2 uv){

   
   // add a constant displacement to hide the default column
   uv.x+=0.031*sin(uv.y*90.0);
   uv.x+=0.01*sin(uv.y*300.0);
   uv.x+=0.00031*sin(uv.y*1500.0);
   
   //Transform the pattern coordinates too
   float rotate = iTime*0.1;
   uv = iOrigin + mat2(cos(rotate),-sin(rotate), sin(rotate),cos(rotate)) * (uv - iOrigin);
   
   //uv.y+=0.25;
   vec4 c= texture(iChannel0,uv); 
   return c;

    
}

// Function 106
float BrickPattern(vec2 p
){//p*=vec2 (1,2)  // scale
 ;vec2 f=floor(p)
 ;p.x-=step(f.y,2.*floor(f.y*.5))*.5// brick shift
 ;p=abs (fract (p + 0.5) - 0.5)
 ;//p=smoothstep (0.03, 0.08, p)
 ;return min(p.x,p.y)
 ;}

// Function 107
float Pattern(vec2 p, vec4 s, float b){//s=scale and b=offset
	p=abs(p);//typical rug reflection
	p.y+=floor(2.0*fract(p.x))*b; //brick offset
	vec2 c=fract(vec2(p.x+p.y,p.x-p.y)*s.zw)-0.5; //diamond repeat
	p=fract(p*s.xy)-0.5; //square repeat
	return step(p.x*p.y,0.0)+step(c.x*c.y,0.0); //overlaid checkers
}

// Function 108
vec3 subtile(in vec2 st, in vec3 primaryColor, in vec3 secondaryColor) {
    vec3 color = vec3(.2);

    // divide into tiles
    float divisions = 3.;
    vec2 mst = st;
    mst *= divisions;
    // tile
    mst = mod(mst, 1.);

    color = subsubtile(mst, primaryColor, secondaryColor);

    return color;
}

// Function 109
vec2 rotateTilePattern2(vec2 _st){

    //  Scale the coordinate system by 2x2 
    _st *= 2.0;

    //  Give each cell an index number
    //  according to its position
    float index = 0.0;    
    index += step(1., mod(_st.x,2.0));
    index += step(1., mod(_st.y,2.0))*2.0;

    // Make each cell between 0.0 - 1.0
    _st = fract(_st);

    // Rotate each cell according to the index
    if(index == 1.0){
        //  Rotate cell 1 by 90 degrees
    } else if(index == 2.0){
        //  Rotate cell 2 by -90 degrees
        _st = rotate2D(_st,PI);
    } else if(index == 3.0){
        //  Rotate cell 3 by 180 degrees
        _st = rotate2D(_st,PI * 1.5);
    } else {
		_st = rotate2D(_st, PI * 0.5);
    }
    

    return _st;
}

// Function 110
float tileBump( in vec3 p ) {
    const vec3 tileSize = vec3(0.2, 0.2, 0.072);
    p /= tileSize;
    p += PHI*vec3(1, 2, 6);
    p.xy += mod(floor(p.z), 5.0)/3.0;
    vec3 inTile = (fract(p) * 2.0 - 1.0);
    vec3 bump = smoothstep(0.9, 1.0, abs(inTile));
   	return max(max(bump.x, bump.y), bump.z);
}

// Function 111
vec4 PatternRand( uint seed )
{
    return vec4(
        float((seed*0x73494U)&0xfffffU)/float(0x100000),
    	float((seed*0xAF71FU)&0xfffffU)/float(0x100000),
        float((seed*0x67a42U)&0xfffffU)/float(0x100000), // a bit stripey against x and z, but evens out over time
        float((seed*0x95a8cU)&0xfffffU)/float(0x100000) // good vs x or y, not good against z
        );
}

// Function 112
float cellTile(in vec3 p){
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    vec4 v, d; 
    d.x = drawObject(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawObject(p - vec3(.39, .2, .11));
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawObject(p - vec3(.62, .24, .06));
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071;
    d.w = drawObject(p - vec3(.2, .82, .64));

    v.xy = min(d.xz, d.yw), v.z = min(max(d.x, d.y), max(d.z, d.w)), v.w = max(v.x, v.y); 
   
    d.x =  min(v.z, v.w) - min(v.x, v.y); // Maximum minus second order, for that beveled Voronoi look. Range [0, 1].
    //d.x =  min(v.x, v.y);
        
    return d.x*2.66; // Normalize... roughly.
    
}

// Function 113
float dist_tiled_stars(vec2 uv, float r)
{
	return dist_star(
		tile(uv, vec2(sqrt(3.0), 1.0)),
		r
	);
}

// Function 114
vec2 pattern5(vec2 uv)
{
      float time = iTime;
      float r = sqrt(dot(uv,uv));
      float u = 0.1*uv.x/(0.11+r*0.5); // The r's here just make it look four quads
      float v = 0.1*uv.y/(0.11+r*0.5);
      return vec2(u,v);
}

// Function 115
float pattern(float n, vec2 p)
{
	p = p * 4.0;
	p = floor(p + 2.5);
	
	if (clamp(p.x, 0.0, 4.0) == p.x && clamp(p.y, 0.0, 4.0) == p.y)
	{
		float k = p.x + p.y*5.0;
		if (int(mod(n/(pow(2.0,k)),2.0)) == 1) return 0.0;
	}	
	
	return 1.0;
}

// Function 116
vec4 tileColor(vec3 p, float t) {
	p.y = -p.y;
	if (maxcomp(p) > 3.0) return vec4(0.0);
	vec3 uvw = mod(p, 1.0);
	vec3 color = vec3(0.566, 0.533, 0.813);
	float tileBorder = 0.45;
	bool inTile = maxcomp(abs(uvw.xy - 0.5)) < tileBorder;
	inTile = inTile || maxcomp(abs(uvw.xz - 0.5)) < tileBorder;
	inTile = inTile || maxcomp(abs(uvw.yz - 0.5)) < tileBorder;
	
	vec3 tilePos = floor(p) + 0.5;
	vec3 side = vec3(lessThan(p, vec3(0.0000001)));
	vec3 sidePattern = vec3(sideX(tilePos, t - 4.0), sideY(tilePos, t), sideZ(tilePos, t - 9.0)) * float(inTile);

	float tileAlpha = sum(side * sidePattern);
	tileAlpha = min(tileAlpha, 1.0);
	
	vec3 tileColor = mix(translucentColor(p, t + 11.0), color, tileAlpha);
	
	return vec4(tileColor, 1.0);
	
}

// Function 117
float pattern(in vec3 p, inout vec3 q, inout vec3 r)
{
    q.x = fbm4( p + 0.0, 0.0, 1.0, 2.0, 0.33 );
    q.y = fbm4( p + 6.0, 0.0, 1.0, 2.0, 0.33 );

    r.x = fbm8( p + q - 2.4, 0.0, 1.0, 3.0, 0.5 );
    r.y = fbm8( p + q + 8.2, 0.0, 1.0, 3.0, 0.5 );

    q.x = turbulence( q.x );
    q.y = turbulence( q.y );

    float f = fbm4( p + (1.0 * r), 0.0, 1.0, 2.0, 0.5);

    return f;
}

// Function 118
float pattern( vec2 x, float r, float time, float speed1, float speed2 )
{
	x /= r;
	vec2 xy = floor(x);
	float k = smoothstep( 0.45, 0.55, 1.0 - length(fract(x) - 0.5) );

	float i = xy.x + xy.y * 4.0;
	float t = max( time * speed1 - i, 0.0 );
	float phase = max( 0.0, sin( t * speed2 ) * k );

	return phase;
}

// Function 119
vec3 getTiles(vec2 p) {
    vec3 h = vec3(.75),
         j = vec3(.25);  	

    float scale = 1./SCALE;
    p/=scale;

    float dir = mod(tip.x + tip.y ,2.) * 2. - 1.;       
    
    vec3 ca = hue((9.45+p.y*.045)*PI);
    vec3 cb = hue((6.4+p.y*.075)*PI);
    vec3 cc = thsh>.6 ? ca : cb;
    vec3 cf = thsh>.6 ? cb : ca;
    
    vec2 cUv= p.xy-sign(p.x+p.y+.001)*.5;

    float d = length(cUv);
    float mask = smoothstep(.01, .001, abs(abs(abs(d-.5)-.02)-.15)-.03 );
    float mask2 = smoothstep(.01, .001, abs(d-.5)-.08 );
    float angle = atan(cUv.x, cUv.y);
    float a = sin(dir * angle * 32. + iTime * 3.5);
	float b = sin(dir * angle * 12. - iTime * 4.5);
    
    h = mix(cf,cc,smoothstep(.01, .05, a)); 
    j = mix(cf,cc,smoothstep(.01, .05, b)); 
    
    h = mix(cc,h,mask);
    j = mix(cc,j,mask2);
	j = mix(h,j,mask2);
    return j;
}

// Function 120
vec2 truchetPattern(in vec2 _st, in float _index){
    
    if (_index > 0.75) {
        _st = vec2(1.) - _st;
    } else if (_index > 0.5) {
        _st = vec2(1.0-_st.x,_st.y);
    } else if (_index > 0.25) {
        _st = 1.0-vec2(1.0-_st.x,_st.y);
    }
    return _st;
}

// Function 121
float pattern(vec2 uv, float time, inout vec2 q, inout vec2 r) {

      q = vec2( fbm1( uv * .1 + vec2(0.0,0.0) ),
                     fbm1( uv + vec2(5.2,1.3) ) );

      r = vec2( fbm1( uv * .1 + 4.0*q + vec2(1.7 - time / 2.,9.2) ),
                     fbm1( uv + 4.0*q + vec2(8.3 - time / 2.,2.8) ) );

      vec2 s = vec2( fbm1( uv + 5.0*r + vec2(21.7 - time / 2.,90.2) ),
                     fbm1( uv * .05 + 5.0*r + vec2(80.3 - time / 2.,20.8) ) );

      return fbm1( uv * .05 + 4.0*s );
    }

// Function 122
float pattern(vec3 p, mat3 m, float s, float id)
{
	float r = 0.;
	p = abs(fract(p*m*s) - 0.5);
	if (id > 3.) r= max(min(abs(p.x),abs(p.z)),abs(p.y));
    else if (id > 2.) r= max(p.x,abs(p.y)+p.z);
	else if (id > 1.) r= length(p);
    else if (id > 0.) r= max(p.x,-p.y);
	return r;
}

// Function 123
float StarPattern(in vec2 p)
{
  p = abs(fract(p*1.5)-0.5);
  return max(max(p.x, p.y), min(p.x, p.y)*2.);
}

// Function 124
float tileableWorley(in vec2 p, in float numCells)
{
	p *= numCells;
	float d = 1.0e10;
	for (int xo = -1; xo <= 1; xo++)
	{
		for (int yo = -1; yo <= 1; yo++)
		{
			vec2 tp = floor(p) + vec2(xo, yo);
			tp = p - tp - hash22(256. * mod(tp, numCells));
			d = min(d, dot(tp, tp));
		}
	}
	return sqrt(d);
	//return 1.0 - d;// ...Bubbles.
}

// Function 125
vec4 getpattern(vec2 uv){

   
  // uv += iTime*0.01;

   // add a constant displacement to hide the default column
   /*
   uv.x+=0.031*sin(uv.y*90.0);
   uv.x+=0.01*sin(uv.y*300.0);
   uv.x+=0.00031*sin(uv.y*1500.0);
    */
   
   //return texture(iChannel0,uv);
   return vec4( step(fract(uv.x*3.),0.5));

}

// Function 126
vec2 pattern1(vec2 uv)
{
	float s = sin(3.1416*iTime/16.0);
    float c = sin(3.1416*iTime/16.0);
    //float c = 1.0;
    uv = uv*iResolution.xy - vec2(0.5);
    vec2 point = vec2(c*uv.x - s*uv.y, s*uv.x + c*uv.y)*0.01;
   // return vec2 (sin(point.x)+sin(point.y);
    return vec2(sin(point.x),sin(point.y));
}

// Function 127
vec4 sample_tile(sampler2D buffer, vec2 resolution, vec2 uv, float tile_id_f) {
    float tile_height = 1.0 / pow(2.0, tile_id_f);
    
    uv.x += 1.0;
    vec2 area_uv = uv * tile_height;
    
    // Compensate for GL.LINEAR sampling - we need to sample the middle of the pixel
    vec2 tile_resolution = resolution * tile_height;
    vec2 inv_resolution = 1.0 / resolution.xy;
    area_uv -= mod(area_uv, inv_resolution) - inv_resolution * 0.5;
    
    return texture(buffer, area_uv);
}

// Function 128
vec2 arrowTileCenterCoord(vec2 pos) {
	return (floor(pos / ARROW_TILE_SIZE) + 0.5) * ARROW_TILE_SIZE;
}

// Function 129
vec2 diffraction_pattern(vec2 uv, int nslits, float spacing, float lambda) {
  float xprop = cvis*iTime;
  vec2 v = vec2(0.);
  if (uv.x<0.) {
    float d = max(1.,abs(uv.y)/(spacing*float(nslits)/2.))-1.;
    float a = 1./(1.+10.*d*d);
    v = a*compsin(2.*pi*fract((uv.x-xprop)/lambda))*1e6;
  }
  else
  for (int i=-(nslits/2); i<((nslits+1)/2); i++) {
    float h = spacing*(float(i)+.5*float((nslits+1)%2));
    vec2 r = uv-vec2(0.,h);
    float a = 1./length(r)*(r.x*r.x)/dot(r,r);
    v+= a*compsin(2.*pi*fract((length(r)-xprop)/lambda));
  }
  return v;
}

// Function 130
int cellTileID(in vec3 p){
    
    int cellID = 0;
    
    // Storage for the closest distance metric, second closest and the current
    // distance for comparisson testing.
    vec3 d = (vec3(.75)); // Set the maximum.
    
    // Draw some overlapping objects (spheres, in this case) at various positions on the tile.
    // Then do the fist and second order distance checks. Very simple.
    d.z = drawSphere(p - vec3(.81, .62, .53)); if(d.z<d.x) cellID = 1;
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);
    
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.z = drawSphere(p - vec3(.39, .2, .11)); if(d.z<d.x) cellID = 2;
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);
    
    
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawSphere(p - vec3(.62, .24, .06)); if(d.z<d.x) cellID = 3;
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);
   
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071; 
    d.z = drawSphere(p - vec3(.2, .82, .64)); if(d.z<d.x) cellID = 4;
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);

/* 
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.z = drawSphere2(p - vec3(.48, .29, .2)); if(d.z<d.x) cellID = 5;
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);
    
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawSphere2(p - vec3(.06, .87, .78)); if(d.z<d.x) cellID = 6;
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z); 
*/ 
    
    return cellID;
    
}

// Function 131
vec3 scaleTile(vec2 p){
    
    // Contorting the scale a bit to add to the hand-drawn look.
    vec2 scale = vec2(3, -2.);
    
    // One set of scale tiles, which take up half the space.
    float sm = scalesMask(p*scale); // Mask.
    vec3 col = sm*scaleDec(p*scale + vec2(-.5, -.25), 1.); // Decoration.
    float bf2 = bumpValue*sm;
    
    // The other set of scale tiles.
    float sm2 = scalesMask(p*scale + vec2(-.45, -.75)); // Mask.
    vec3 col2 = sm2*scaleDec(p*scale + vec2(-.5, -.75) + vec2(-.45, -.25), -1.); // Decoration.
    
    
    #ifdef SHOW_ALTERNATE_LAYERS
    // A simple way to distinguish between the two layers.
    col2 = col2*.7 + col2.yxz*.3;
    #endif
    
    // Add some highlighting.
    bumpValue = max(bf2, bumpValue*sm2);
    col = max(col, col2);
    
    // Toning the color down a bit. This was a last minute thing.
    return col*.8 + col.zxy*.2;
    
}

// Function 132
int multilevelGridIdx1(inout int idx) {
    for (int i = 0; i < 32; ++i) {
        if (idx / 2 == (idx + 1) / 2)
          idx /= 2;
        else
            break;
    }
    return idx;
}

// Function 133
vec3 pattern(vec2 uv){
    
    // A subtlely spot-lit background. Performed on uv prior to tranformation,
    float bg = max(1. - length(uv), 0.)*.025; 
    
    // Transform the screen coordinates. Comment out the following two lines and 
    // you'll be left with a standard Sierpinski pattern.
    uv = Mobius(uv, vec2(-.75, cos(iTime)*.25), vec2(.5, sin(iTime)*.25));
    uv = spiralZoom(uv, vec2(-.5), 5., 3.14159*.2, .5, vec2(-1, 1)*iTime*.25);
    
     
    vec3 col = vec3(bg); // Set the canvas to the background.
    
    // Sierpinski Carpet - Essentially, space is divided into 3 each iteration, and a 
    // shape of some kind is rendered. In this case, it's a smooth rectangle
    // with a bit of shading around the side.
    //
    // There's some extra steps in there (the "l" and "mod" bits) due to the 
    // shading and coloring, but it's pretty simple.
    //
    // By the way, there are other combinations you could use.
    //
    for(float i=0.; i<4.; i++){
        
        uv = fract(uv)*3.; // Subdividing space.
        
        vec2 w = .5 - abs(uv - 1.5); // Prepare to make a square. Other shapes are also possible.
        
        float l = sqrt(max(16.0*w.x*w.y*(1.0-w.x)*(1.0-w.y), 0.)); // Vignetting (edge shading).
        
        w = smoothstep(0., length(fwidth(w)), w); // Smooth edge stepping.
        
        vec3 lCol = vec3(1)*w.x*w.y*l; // White shaded square with smooth edges.
        
        if(mod(i, 3.)<.5) lCol *= vec3(0.1, 0.8, 1); // Color layers zero and three blue.
        
        col = max(col, lCol); // Taking the max of the four layers.
        
    } 
    
    return col;
    
}

// Function 134
float sinePattern(vec2 p){return sin(p.x*20.+cos(p.y*12.));}

// Function 135
vec2 PipePattern(vec2 p, float lw){
    
    // Distance field variables.
    float d = 1e5, d2 = 1e5;
    

	vec2 ip = floor(p); // Cell ID.
    p -= ip + .5; // Cell's local position. Range [vec2(-.5), vec2(.5)].

    
    // Using the cell ID to generate some unique random numbers.
    //
    float rnd = hash21(ip + 12.53); // Cell type selection.
    float rnd2 = hash21(ip); // Individual random tile flipping.

 
    if(rnd2>.6){
        
        // Standard, double arc Truchet tile.
        
        // Random tile flipping.
        p.y *= (rnd>.5)? -1. : 1.;

        // Diagonal repeat symmetry, in order to draw two arcs with one call.
        p = p.x>-p.y? p : -p;

        // Creating two annuli at diagonal corners.
        float dc = abs(length(p - .5) - .5);
        d = min(d, dc);

        // Without diagonal symmetry, you'd need to draw the other arc.
        //dc = abs(length(p + .5) - .5);
        //d = min(d, dc);
        
    }
    else if(rnd2>.3){
        
        
        // A single line running down the middle of the tile, with dots on
        // the two remaining edges.
        
        // Random tile flipping.
        p = (fract(rnd*151. + .76)>.5)? p.yx : p;

        // Line.
        d = min(d, sBox(p, vec2(0, .5)));

        // Two dots.
        p.x = abs(p.x); // Repeat trick.
        d = min(d, length(p - vec2(.5, 0)));            
        
    }
    else {

        // Cross over lines. This necessitates two distance fields, since 
        // there is rendering order to consider. This tile by itself would
        // create a weave pattern.
        
        // Random tile flipping.
        p = (fract(rnd*57. + .34)>.5)? p.yx : p;
        
        // Verticle line.
        d = min(d, sBox(p, vec2(0, .5)));
        // Horizontal line.
        d2 = min(d2, sBox(p, vec2(.5, 0)));
       
    }

 
    // Field width, or giving the pipe pattern some width.
    d -= lw/2.;
    d2 -= lw/2.;
 
   
    return vec2(d, d2);
    
    
}

// Function 136
vec3 HexPattern( vec2 vUV, vec3 colInner, vec3 colEdge )
{
    vec4 hex = Hexagon( vUV );
    
    float edgeShade = step( fract(hex.w), 0.5 );

    vec3 col = colInner; 
    col = mix( col, vec3(edgeShade), step(hex.x, 0.3) ); // black / white edge
    col = mix( col, colEdge, step(hex.x, 0.15) ); ; // Yellow Surround
        
    return col;
}

// Function 137
float cellTile(in vec3 p){
    
     
    vec3 d = vec3(.75); // Set the maximum.
    
    // Draw four overlapping shapes (circles, in this case) using the darken blend 
    // at various positions on the tile.
    d.z = drawObject(p - vec3(.81, .62, .53));
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.z = drawObject(p - vec3(.39, .2, .11));
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);
    
    
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
     
   
    d.z = drawObject(p - vec3(.62, .24, .06));
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071; 
    d.z = drawObject(p - vec3(.2, .82, .64));
    d.y = max(d.x, min(d.y, d.z)); d.x = min(d.x, d.z);

    //d = sqrt(d);
    
    //return 1. - (d.x*2.66);
    return ((d.y - d.x)*2.66);
    //return (1.-sqrt(d.x)*1.33);
    
}

// Function 138
float cellTile2d(in vec2 p){
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    vec4 d; 
    d.x = drawObject2d(p - vec2(.391, .62));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawObject2d(p - vec2(.24, .587));
    p.yx = vec2(p.x-p.y, p.x + p.y)*.7071;
    d.z = drawObject2d(p - vec2(.778, .14));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.w = drawObject2d(p - vec2(.2623, .783));

    // Obtain the minimum, and you're done.
    d.xy = min(d.xz, d.yw);
        
    return min(d.x, d.y)*2.66; // Scale between zero and one... roughly.
}

// Function 139
vec2 getSubTileCoords( vec2 f_c ){

    //: Pretend the entire screen is one tile for this  -----://
    //: example code. Center is at [0,0] , top left is  -----://
    //: at [-1,-1] and bottom right is at [+1,+1]       -----://
    vec2 uvc   = (f_c -0.5*iResolution.xy)/iResolution.y;
         uvc.y =( 0.0 - uvc.y ); //:Invert Y axis

//+----------------------------------------------------------+//
//| What is the current wang-tile touching mask for          |//
//| the tile you are drawing. In this example,               |//
//| pretend the entire screen is one tile.                   |//
//| TIL == Your One Tile                                     |//
//|          +---+                                           |//
//|          |y_0|                                           |//
//|      +---+---+---+      0|1  0|1  |  0|1  0|1            |//
//|      |x_0|TIL|x_1| -->  ___  ___  |  ___  ___            |//
//|      +---+---+---+      X_0  X_1  |  Y_0  Y_1            |//
//|          |y_1|                                           |//
//|          +---+                                           |//
//|                                                          |//
//| toutang means : Touching Tangent                         |//
//|                                                          |//
//| [-1,-1]                                                  |//
//|     +-------------+                                      |//
//|     |             |                                      |//
//|     |             |                                      |//
//|     |     0.0     | <--[ TIL ] www.twitch.com/kanjicoder |//
//|     |             |                                      |//
//|     |             |                                      |//
//|     +-------------+                                      |//
//|                 [+1,+1]                                  |//
//|                                                          |//
//+----------------------------------------------------------+//

    //:Animate the touching value of your tile.         -----://
    //:Emulating all different combinations of which    -----://
    //:neighbors can exist above,below,left,and right:  -----://

    uint toutang=(uint(int(mod(iTime,16.0))));

//+----------------------------------------------------------+//
//|   +-----------------+ If( touself ==BINARY[ 1000 ] )THEN:|//
//|   |\\     y_0     //|                                    |//
//|   |  \\         //  |   We are in the x_0 pie slice.     |//
//|   |    \\     //    |                                    |//
//|   |      \\ //      | If( touself ==BINARY[ 1010 ] )THEN:|//
//|   |x_0  ( 0.0 )  x_1|                                    |//
//|   |      // \\      |   I fucked up the formula because  |//
//|   |    //     \\    |   only ONE_BIT in touself should   |//
//|   |  //         \\  |   have been set.                   |//
//|   |//     y_1     \\|                                    |//
//|   +-----------------+                                    |//
//|   BITS[  0   0   0   0  ]                                |//
//|   SIDE[ x_0 x_1 y_0 y_1 ]     www.twitch.com/kanjicoder  |//
//|                                                          |//
//|   touself means: "TOUching SELF"                         |//
//+----------------------------------------------------------+//

    //:Figure out which pie slice the pixel of  -------------://
    //:your tile belongs to and set that bit:   -------------://
    #define A abs
    #define U uint
    #define X uvc.x
    #define Y uvc.y
    U touself =( U(0)
    | (( (X <= 0.0 && (A(X)>A(Y))) ? U(1) : U(0) ) << 3)
    | (( (X >= 0.0 && (A(X)>A(Y))) ? U(1) : U(0) ) << 2)
    | (( (Y <= 0.0 && (A(Y)>A(X))) ? U(1) : U(0) ) << 1)
    | (( (Y >= 0.0 && (A(Y)>A(X))) ? U(1) : U(0) ) << 0)
    );;
    #undef A
    #undef U
    #undef X
    #undef Y

    //:If we are on a pie slice that is touching a           ://
    //:neighbor, use the connected gradient(congrad).        ://
    //:If we are on a pie slice that is __NOT__              ://
    //:touching a neighbor, use walled-off gradient(walgrad).://
    float walgrad = float( max( abs(uvc.x),abs(uvc.y) ) );
    float congrad = float( min( abs(uvc.x),abs(uvc.y) ) );
    float tougrad = (( touself & toutang )>=uint(1)) 
                 ? congrad   //:TRUE : Connected  Gradient
                 : walgrad ; //:FALSE: Walled Off Gradient

    float invgrad = (( touself & toutang )>=uint(1)) 
                 ? walgrad   //:INVERTED_GRADIENT
                 : congrad ; //:INVERTED_GRADIENT

    return( vec2( tougrad , invgrad ) );

}

// Function 140
float voronoi_tile(vec2 p) {
    vec2 g = floor(p);
    vec2 f = fract(p);
    vec2 k = f*f*f*(6.0*f*f - 15.0*f + 10.0);

    f -= vec2(0.5);
    g -= vec2(0.5);
    float res = 1.0;
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {
            vec2 b = vec2(i, j);
            float d = length(hash2(g + b) - abs(f) + b);
            res = min(res, d);
        }
    }
    return res;
}

// Function 141
float tile(vec2 uv, int tile)
{
    switch(tile)
    {
        case 0: return 1.414;
        case 1: return max(tile0(uv), .15 - length(uv));
        case 2: return tile0(uv.yx);
        case 3: return tile1(uv);
        case 4: return tile0(vec2(uv.x, -uv.y));
        case 5: return tile2(uv);
        case 6: return tile1(vec2(uv.x, -uv.y));
        case 7: return tile3(uv);
        case 8: return tile0(vec2(uv.y, -uv.x));
        case 9: return tile1(vec2(-uv.x, uv.y));
        case 10: return tile2(uv.yx);
        case 11: return tile3(uv.yx);
        case 12: return tile1(vec2(-uv.x, -uv.y));
        case 13: return tile3(vec2(-uv.x, uv.y));
        case 14: return tile3(vec2(-uv.y, uv.x));
        case 15: return tile4(uv);
    }
}

// Function 142
float cellTile(in vec3 p){
    
    vec3 d;
    
    // Draw three overlapping objects (spherical, in this case) at various positions throughout the tile.
    d.x = drawObject(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawObject(p - vec3(.2, .82, .64));
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawObject(p - vec3(.41, .06, .70));
    
    return 1.- min(min(d.x, d.y), d.z)*.1666; // Normalize... roughly.
    
}

// Function 143
vec3 geartile(vec2 domain, float phase){
	domain = fract(domain);
	return 
		gear(domain, -phase, vec2(-0.25,0.25)) + 
		gear(domain, phase, vec2(-0.25,0.75)) + 
		gear(domain, phase, vec2(1.25,0.25)) + 
		gear(domain,- phase, vec2(1.25,0.75)) + 
		gear(domain, -phase, vec2(0.25,-0.25)) + 
		gear(domain, phase, vec2(0.75,-0.25)) + 
		gear(domain, phase, vec2(0.25,1.25)) + 
		gear(domain, -phase, vec2(0.75,1.25)) + 
		gear(domain, phase, vec2(0.25,0.25)) + 
		gear(domain, -phase, vec2(0.25,0.75)) + 
		gear(domain, -phase, vec2(0.75,0.25)) + 
		gear(domain, phase, vec2(0.75,0.75));		
}

// Function 144
float cellTile(in vec3 p){
    
    float c = .25; // Set the maximum.
    
    // Draw four overlapping objects (spheres, in this case) using the darken blend 
    // at various positions throughout the tile.
    c = min(c, drawSphere(p - vec3(.81, .62, .53)));
    c = min(c, drawSphere(p - vec3(.39, .2, .11)));
    
    c = min(c, drawSphere(p - vec3(.62, .24, .06)));
    c = min(c, drawSphere(p - vec3(.2, .82, .64)));
    
    
    // Add some smaller spheres at various positions throughout the tile.
    
    p *= 1.4142;
    
    c = min(c, drawSphere(p - vec3(.48, .29, .2)));
    c = min(c, drawSphere(p - vec3(.06, .87, .78)));
    
    // More is better, but I'm cutting down to save cycles.
    //c = min(c, drawSphere(p - vec3(.6, .86, .0)));
    //c = min(c, drawSphere(p - vec3(.18, .44, .58)));
        
    return (c*4.); // Normalize.
    
}

// Function 145
float Basketwork2Pattern(in vec2 uv)
{
  vec2 p = uv * 4.0;
  return max (S2( p.x, p.y), S2(p.y, p.x+1.) );
}

// Function 146
float splatterPattern(vec2 st, float radius){
    return splatter(st + vec2(0., -.1), radius + .1) +
            splatter(st + vec2(0., -.2), radius - .1) +
            splatter(st + vec2(-.1, 0.), radius + .2) +
            splatter(st + vec2(-.2, 0.), radius - .2) ;
}

// Function 147
float linePattern(vec2 p, vec2 a, vec2 b){
  
    // Determine the angle between the vertical 12 o'clock vector and the edge
    // we wish to decorate (put lines on), then rotate "p" by that angle prior
    // to decorating. Simple.
    vec2 v1 = vec2(0, 1);
    vec2 v2 = (b - a); 
 
    // Angle between vectors.
    //float ang = acos(dot(v1, v2)/(length(v1)*length(v2))); // In general.
    float ang = acos(v2.y/length(v2)); // Trimed down.
    p = rot2(ang)*p; // Putting the angle slightly past 90 degrees is optional.

    float ln = doHatch(p);//clamp(cos(p.x*96.*6.2831)*.35 + .95, 0., 1.);

    return ln;// *clamp(sin(p.y*96.*6.2831)*.35 + .95, 0., 1.); // Ridges.
 
}

// Function 148
vec4 roundPattern(vec2 uv)
{
    float dist = length(uv);
    
    // Resolution dependant Anti-Aliasing for a prettier thumbnail
    // Thanks Fabrice Neyret & dracusa for pointing this out.
    float aa = 8. / iResolution.x;

    // concentric circles are made by thresholding a triangle wave function
    float triangle = abs(fract(dist * 11.0 + 0.3) - 0.5);
    float circles = S(0.25 - aa * 10.0, 0.25 + aa * 10.0, triangle);

    // a light gradient is applied to the rings
    float grad = dist * 2.0;
    vec3 col = mix(vec3(0.0, 0.5, 0.6),  vec3(0.0, 0.2, 0.5), grad * grad);
    col = mix(col, vec3(1.0), circles);
    
    // border and center are red
    vec3 borderColor = vec3(0.7, 0.2, 0.2);
    col = mix(col, borderColor, S(0.44 - aa, 0.44 + aa, dist));
    col = mix(col, borderColor, S(0.05 + aa, 0.05 - aa, dist));
    
    // computes the mask with a soft shadow
    float mask = S(0.5, 0.49, dist);
    float blur = 0.3;
    float shadow = S(0.5 + blur, 0.5 - blur, dist);
   
    return vec4(col * mask, clamp(mask + shadow * 0.55, 0.0, 1.0)); 
}

// Function 149
float MinimalWeavePattern(vec2 coord)
{
  vec3 bg = vec3(0),  warp = vec3(.5),  weft = vec3(1);
  ivec2 uv = ivec2(floor(coord*8.));
  int mask = (int(iTime / 2.) & 7) << 2;
  vec3 col = (((uv.x ^ uv.y) & mask) == 0
   ? 1 == ((uv.x ^ uv.x >> 1) & 1) ? warp : bg
   : 1 == ((uv.y ^ uv.y >> 1) & 1) ? weft : bg);
  return col.r;
}

// Function 150
float cellTile(in vec3 p){
    
    float c = .25; // Set the maximum.
    
    c = min(c, drawSphere(p - vec3(.81, .62, .53)));
    c = min(c, drawSphere(p - vec3(.39, .2, .11)));
    
    c = min(c, drawSphere(p - vec3(.62, .24, .06)));
    c = min(c, drawSphere(p - vec3(.2, .82, .64)));
    
    p *= 1.4142;
    
    c = min(c, drawSphere(p - vec3(.48, .29, .2)));
    c = min(c, drawSphere(p - vec3(.06, .87, .78)));

    c = min(c, drawSphere(p - vec3(.6, .86, .0)));
    c = min(c, drawSphere(p - vec3(.18, .44, .58)));
        
    return (c*4.);
    
}

// Function 151
vec2 GetStoneTiles(vec3 p) {
    return vec2(p.x + sin(p.z * stones.x), p.z + sin(p.x * stones.y) * stones.z) * stones.w;
}

// Function 152
float cellTile(in vec3 p){
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    vec4 v, d; 
    d.x = drawSphere(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawSphere(p - vec3(.39, .2, .11));
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawSphere(p - vec3(.62, .24, .06));
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071;
    d.w = drawSphere(p - vec3(.2, .82, .64));

    v.xy = min(d.xz, d.yw), v.z = min(max(d.x, d.y), max(d.z, d.w)), v.w = max(v.x, v.y); 
   
    d.x =  min(v.z, v.w) - min(v.x, v.y); // Maximum minus second order, for that beveled Voronoi look. Range [0, 1].
    //d.x =  min(v.x, v.y);
        
    return (d.x*2.66); // Normalize... roughly.
    
}

// Function 153
float woodPattern(vec2 c) {
	float o = 1.;
    if (mod(c.y, 4.) < 1.) o = 0.;
    if (mod(c.x + 2. - 6. * step(4., mod(c.y, 8.)), 16.) > 15.) o = 0.;
    return o;
}

// Function 154
float circlePattern(vec2 st, float radius) {
    return  circle(st+vec2(0.,-.5), radius)+
            circle(st+vec2(0.,.5), radius)+
            circle(st+vec2(-.5,0.), radius)+
            circle(st+vec2(.5,0.), radius);
}

// Function 155
float dotPattern(vec2 p){
    
    // Partition space into multiple squares.
    vec2 fp = abs(fract(p)-0.5)*2.;
    
    // Rounded circle, for the overlay, or vignette, if you prefer.
    fp = pow(fp, vec2(8.));
    float r = max(1. - pow(fp.x + fp.y, 1.), 0.);
    
    // More squarish (Chebyshev) version of the above.
    //fp = pow(fp, vec2(8.));
    //float r = 1. - max(fp.x, fp.y);
    
    // Single value for each square. Used for IDs and a bunch of other things, but in this 
    // case it'll give the square a homogeneous color.
    p = floor(p); 
    
    // The blocky pattern value. Made up, but you could use all kinds of things, like Voronoi, etc. 
    float c = dot(sin(p/4. - cos(p.yx/.2 + iTime/4.)), vec2(.5));

    c = fract(c * 7.0); // Mixing it up, for no particular reason.

    return c*r; // Pixel shade, multiplied by the rounded square vignette. Range: [0, 1].
    
}

// Function 156
float DiamondPattern(vec2 u//https://www.shadertoy.com/view/lsVczV
){u=abs(fract(u)-.5)
 ;return (ma(u));}

// Function 157
float tile3(vec2 uv)
{
    return max(-uv.x - k * .5, K2 - length(vec2(uv.x - .5, abs(uv.y) - .5)));
}

// Function 158
float TestPattern(in vec2 uv)
{
  uv *= 8.0;
  return clamp(88.*sin(uv.x)* sin(uv.y), 0.0, 1.0);
}

// Function 159
float PatternSins(vec3 x){
     //x = (x)+ sin(x)*twoPI*1.0 + cos(x.yzx*0.5)*twoPI*0.25  + cos(x.zxy*0.1)*twoPI*.10;
     x = (x)+ sin(x)*twoPI;
    return iqNoise(x);
}

// Function 160
vec3 geartile(vec2 domain, vec2 aspect, float phase){
	domain = fract(domain);
	return 
		gear(domain, aspect, -phase, vec2(-0.25,0.25)) + 
		gear(domain, aspect, phase, vec2(-0.25,0.75)) + 
		gear(domain, aspect, phase, vec2(1.25,0.25)) + 
		gear(domain, aspect,- phase, vec2(1.25,0.75)) + 
		gear(domain, aspect, -phase, vec2(0.25,-0.25)) + 
		gear(domain, aspect, phase, vec2(0.75,-0.25)) + 
		gear(domain, aspect, phase, vec2(0.25,1.25)) + 
		gear(domain, aspect, -phase, vec2(0.75,1.25)) + 
		gear(domain, aspect, phase, vec2(0.25,0.25)) + 
		gear(domain, aspect, -phase, vec2(0.25,0.75)) + 
		gear(domain, aspect, -phase, vec2(0.75,0.25)) + 
		gear(domain, aspect, phase, vec2(0.75,0.75));		
}

// Function 161
ivec2 multilevelGridIdx(ivec2 idx) {
//  return idx >> findLSB(idx); // findLSB not supported by Shadertoy WebGL version
    return ivec2(multilevelGridIdx1(idx.x), multilevelGridIdx1(idx.y));
}

// Function 162
float Sine2Pattern(in vec2 p)
{
  return 0.5+sin(p.x * 20.0 + cos(p.y * 10.0 ))
            *sin(p.y * 20.0 + cos(p.x * 10.0 ));
}

// Function 163
float TxPattern (vec3 p)
{
  float t, tt, c;
  p = abs (0.5 - fract (4. * p));
  c = 0.;
  t = 0.;
  for (float j = 0.; j < 6.; j ++) {
    p = abs (p + 3.) - abs (p - 3.) - p;
    p /= clamp (dot (p, p), 0., 1.);
    p = 3. - 1.5 * p;
    if (mod (j, 2.) == 0.) {
      tt = t;
      t = length (p);
      c += exp (-1. / abs (t - tt));
    }
  }
  return c;
}

// Function 164
float Grid1Pattern(in vec2 uv)
{
  float col = max(sin(uv.x*10.1), sin(uv.y*10.1));
  return smoothstep(0.5,1.,col);
}

// Function 165
float RosettePattern(in vec2 p)
{
  vec2 d = vec2(0.58,1);
  vec4 O = vec4(0);
  for (; O.a++ < 4.; O += D(p) +D(p += d*.5)) p.x += d.x;
  return O.x;
}

// Function 166
float BrickPattern2(in vec2 p){//brickpattern2 is just a the dumb cousin of BrickPattern()
    const float vSize = 0.30;
  const float hSize = 0.05;
  p.y *= 2.5;    // scale y
  if(mod(p.y, 2.0) < 1.0) p.x += 0.5;
  p = p - floor(p);
  if((p.x+hSize) > 1.0 || (p.y < vSize)) return 1.0;
  return 0.0;
}

// Function 167
vec2 tileMoveCrossed(vec2 _st, float _zoom, float utime, float speed){
    float time = utime * speed;
    _st *= _zoom;
    // horizontal or vertical?
    float ver = step(.5,fract(time));
    float hor = step(ver, 0.);
    // even rows and columns
    float evenY = step(.5, fract(_st.y * .5));
    float oddY = step(evenY,0.);
    float evenX = step(.5, fract(_st.x * .5));
    float oddX = step(evenX,0.);
    // apply movement
    _st.x += ((fract(time) * 2.0) * evenY) * hor;
    _st.x -= ((fract(time) * 2.0) * oddY) * hor;
    _st.y += ((fract(time) * 2.0) * evenX) * ver;
    _st.y -= ((fract(time) * 2.0) * oddX) * ver;
    return fract(_st);    
}

// Function 168
float Wallpaper70sPattern(vec2 p, float time)
{
  p.x *= sign(cos(length(ceil(p))*time));
  return cos(min(length(p = fract(p)), length(--p))*44.);
}

// Function 169
vec3 pattern(vec5 p ){
    
    float hueDelta = 1.0/24.0;
    
    p = mod5(p,1.0);
    
    if(dualTileZoneTest(p , p.a.x)){
        return  hsv2rgb(vec3(0.0, 0.7, 0.75));
    }
    else if(dualTileZoneTest(p,  p.a.y)){
         return hsv2rgb(vec3(hueDelta, 0.3,0.8));
    }
    else if(dualTileZoneTest(p, p.a.z)){
         return hsv2rgb(vec3(2.0*hueDelta, 0.2, 0.95));
    }
    else if(dualTileZoneTest(p, p.a.w)){
         return hsv2rgb(vec3(1.0-hueDelta, 0.5, 0.7));
    }
    else if(dualTileZoneTest(p,  p.v)){
     
         return hsv2rgb(vec3(1.0-2.0*hueDelta,0.3, 0.4));
    }
    else {
          return vec3(0.0);
    }   
}

// Function 170
bool HolePattern(in vec3 p, in vec3 c)
{
    vec2 uv;
    vec3 nn = normalize(p - c);
    uv.x = .5 + atan(nn.x, nn.z)/(2.*pi);
    uv.y = .5 + asin(nn.y)/pi;
    
    const float limit = 3.;
    float cap = sphereSubDiv.y * uv.y;
    
   	if (cap < limit || cap > sphereSubDiv.y - limit + borderSize)
        return false;
    
    vec2 pattern = fract(sphereSubDiv*uv);
    vec2 index = floor(sphereSubDiv*uv);
    pattern -= vec2(.5,.5);
    float border = min(abs(pattern.x+.5-borderSize), 
                       abs(pattern.y+.5-borderSize));
    
    bool r1 = border < borderSize;
    bool r2 = length(pattern.xy - borderSize) < borderSize*1.5;


    if (mod(index.x + index.y, 2.) > epsilon)
    {
        return r1 || r2;
    }
    else
    {
    	return r1;
    }
}

// Function 171
vec2 pattern2(vec2 uv)
{
    float time = iTime;
    float r = sqrt(dot(uv,uv)); // These are equivalent I think
    
    float a = atan( uv.x, uv.y);
    
    float u = r*cos(a+r); // omg this is trppy
    float v = r*sin(a+r);
    return vec2(u,v);
}

// Function 172
float cellTile(in vec3 p){
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    vec4 v, d; 
    d.x = drawSphere(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawSphere(p - vec3(.39, .2, .11));
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawSphere(p - vec3(.62, .24, .06));
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071;
    d.w = drawSphere(p - vec3(.2, .82, .64));

    v.xy = min(d.xz, d.yw), v.z = min(max(d.x, d.y), max(d.z, d.w)), v.w = max(v.x, v.y); 
   
    d.x =  min(v.z, v.w) - min(v.x, v.y); // First minus second order, for that beveled Voronoi look. Range [0, 1].
    //d.x =  min(v.x, v.y); // Minimum, for the cellular look.
        
    return d.x*2.66; // Normalize... roughly.
    
}

// Function 173
vec2 pattern3(vec2 uv)
{
    float time = iTime;
    float r = sqrt(dot(uv,uv)); // These are equivalent I think
    
    float a = atan( uv.y,uv.x );
    
    float u = cos(a+cos(r)/10.0)/r; // This is horribly mind manifesting!
    float v = (sin(a+r)/r); 
    return vec2(u,v);
}

// Function 174
vec4 shapeTile(vec2 U, vec2 Box, float vert, float id) {
    vec4 O = vec4(0);
    vec2 A = Box/2. - abs(U),                   // coords from borders 
         B  = A * 2. / BEVEL;                   // coords in bevel
        float m = min(B.x,B.y);                 // in bevel if [0,1]
    if (A.x<ROUND && A.y<ROUND)                 // round edges
        m = (ROUND-length(ROUND-A)) *2./dot(BEVEL,normalize(ROUND-A));
    
    O += profile( clamp( m ,0.,1.) );           // mask
    
    if (GRAD>0) {
        vec3 R = -1.+2.*hash3f(vec3(id,.3,0));  // rand grad variation
        grad *= rot(grad_randD/2.*R.x) * pow(grad_randA,R.y);
        O.rgb *= .5 + dot(grad,U);              // lum gradient across brick
    }
#if !MM
    if (RAND>0) {                               // color bricks
        vec3 R = hash3f(vec3(id,.2,0));         // brick seed
        O.rgb *= RAND==1 ? R.xxx : R;
    }
    if (TEXT>0) {                               // texture mapped on brick
        vec4 T = texture(iChannel0, U);
        if (TEXT==2 && vert>0.) T = texture(iChannel1, U);
        O *= T;
    }
#endif
    return O;
}

// Function 175
vec2 pattern4(vec2 uv)
{
    float time = iTime;
    float r = sqrt(dot(uv,uv)); // These are equivalent I think
    
    float a = atan( uv.x, uv.y);
    
    float u = (uv.x * time *cos(1.0*r) - uv.y*sin(1.0*time*r)) * 0.01; //Yay it looks like a spiral
    float v = (uv.y*time*cos(1.0*r) + uv.x*sin(1.0*time*r)) * 0.01;
    return vec2(u,v);
}

// Function 176
float DiamondPattern(in vec2 uv)
{
  vec2 dp = abs (fract(uv*2.) - 0.5);
  return 0.3 - cos (19. * max(dp.x, dp.y));
}

// Function 177
float pattern(vec2 p, float a){
	return abs(mod(floor(a*p.x), 2.0)-mod(floor(a*p.y), 2.0));
}

// Function 178
float sd_TechTiles( vec3 p, int lod, float t, Ray ray, TechTilesArgs0 targs, float e )
{
#if ( NUM_TILE_EVALS == 2 )

	// we do 2 evals in alternate checker patterns, "only" x2 cost and relatively clean
	// it still has corner cases (ha..ha..) but help in some situations

	float d = FLT_MAX;

	vec2 index0 = floor( p.xy );
	vec2 indexi = index0;
	float m = mod( indexi.x + indexi.y, 2.0 );

	vec2 dd;

	for ( int k = 0; k < 2; k += 1 )
	{
		vec3 p2 = p;
		vec2 index = index0;
		p2.xy = p.xy - index;

		if ( m == float( k ) )
		{
			vec2 offset = vec2( 0.0 );
			vec2 rp2 = p2.xy - 0.5;
			if ( abs( rp2.y ) > abs( rp2.x ) ) offset.y += rp2.y > 0.0 ? 1.0 : -1.0;
			else offset.x += rp2.x > 0.0 ? 1.0 : -1.0;
			index += offset;
			p2.xy -= offset;
		}

		float ddd = sd_TechTilesTestsSub0( p2, lod, t, ray, index, targs );
#if 0
		dd[k] = ddd; // gpu hangs on desktop (GTX 970)
#else
		if ( k == 0 ) dd.x = ddd;
		else  dd.y = ddd;
#endif

//		d = ddd; // compiler bug? doesn't work on laptop...
	}

	d = opU( dd.x, dd.y );

#else

	vec3 p2 = p;
	vec2 index = floor( p.xy );
	p2.xy = p.xy - index;
	return sd_TechTilesTestsSub0( p2, lod, t, ray, index, targs  ); // only 1 eval

#endif

}

// Function 179
vec3 arrowTilePos(vec3 p) 
{
	return (floor(p / ARROW_TILE_SIZE) + 0.5) * ARROW_TILE_SIZE;
}

// Function 180
vec2 tile (vec2 _st, float _zoom) {
    _st *= _zoom;
    return fract(_st);
}

// Function 181
float HexagonalTruchetPattern(vec2 p)
{
  vec2 h = p + vec2(0.58, 0.15)*p.y;
  vec2 f = fract(h);
  h -= f;
  float v = fract((h.x + h.y) / 3.0);
  (v < 0.6) ? (v < 0.3) ?  h : h++ : h += step(f.yx,f);
  p += vec2(0.5, 0.13)*h.y - h;        // -1/2, sqrt(3)/2
  v = RandomSign;
  return 0.06 / abs(0.5 - min (min
    (length(p - v*vec2(-1., 0.00)  ),  // closest neighbor (even or odd set, dep. s)
     length(p - v*vec2(0.5, 0.87)) ),  // 1/2, sqrt(3)/2
     length(p - v*vec2(0.5,-0.87))));
}

// Function 182
float noisePattern(vec3 pos)
{
    return noise(normalize(pos)*2.5);
}

// Function 183
vec3 debugPattern(vec2 fragCoord, vec2 iResolution)
{
	float x = fragCoord.x / iResolution.x;
    if (x < 0.15)
    {
        // Horizontal lines:
        float size = pow(2., 1.+floor(4. * fragCoord.y / iResolution.y));
        float signal = fragCoord.y;
        return vec3(fract(signal/size) > 0.5);
    }
    else if (x < 0.3)
    {
        // Vertical lines:
        float size = pow(2., 1.+floor(4. * fragCoord.y / iResolution.y));
        float signal = fragCoord.x;
        return vec3(fract(signal/size) > 0.5);
    }
    else if (x < 0.45)
    {
        // Circles:
        float size = pow(2., 1.+floor(4. * fragCoord.y / iResolution.y));
        vec2 uv;
        uv.x = fragCoord.x - (0.3 + 0.5*0.15) * iResolution.x;
        uv.y = mod(fragCoord.y, 0.25 * iResolution.y) - 0.125 * iResolution.y;
        float signal = max(abs(uv.x), abs(uv.y));
        return vec3(fract(signal/size) > 0.5);
    }
    else if (x < 0.6)
    {
        // Squares:
        vec2 uv = vec2((x - 0.3)/0.15, fract(4. * fragCoord.y / iResolution.y));
        float size = pow(2., 1.+floor(4. * fragCoord.y / iResolution.y));
        uv.x = fragCoord.x - (0.45 + 0.5*0.15) * iResolution.x;
        uv.y = mod(fragCoord.y, 0.25 * iResolution.y) - 0.125 * iResolution.y;
        float signal = abs(fract(length(uv)/size) * 2. - 1.)-0.5;
        #if ANTIALISING
        float threeshold = 0.5*fwidth(signal);
        return vec3(smoothstep(-threeshold, threeshold, signal));
        #else
        return vec3(signal > 0.);
        #endif
    }
    else
    {
        // Radial lines:
        vec2 uv = vec2((x - 0.6)/0.4, fragCoord.y / iResolution.x/0.4-0.2);
        vec2 p = uv-0.5;
        float d = length(p);
        float signal = sin(24.*2.*PI*atan(p.y,p.x));
        #if ANTIALISING
        float threeshold = 0.5*fwidth(signal);
        return vec3(smoothstep(-threeshold, threeshold, signal));
        #else
        return vec3(signal > 0.);
        #endif
    }
}

// Function 184
float Basketwork1Pattern(in vec2 uv)
{
  vec2 p = uv * 4.0;
  return max (S1(p.x, p.y), S1(p.y+1.0, p.x));
}

// Function 185
vec2 tile_resolution(vec2 screen_resolution, float tile_id_f) {
    float tile_height = 1.0 / pow(2.0, tile_id_f);
    return screen_resolution * tile_height;
}

// Function 186
float pattern(vec3 p)
{
	p = fract(p) - 0.5;
	return (min(min(abs(p.x), abs(p.y)), abs(p.z)) + 0.56);
}

