# uv_mapping_functions

**Category:** texturing
**Type:** extracted

## Dependencies
texture_sampling, normal_mapping, lighting, raymarching

## Tags
texturing, animation, particles, lighting, color

## Code
```glsl
// Reusable Uv Mapping Texturing Functions
// Automatically extracted from texturing/mapping-related shaders

// Function 1
vec3 tonemapUncharted(vec3 color) {
   color *= 1.0;  // Hardcoded Exposure Adjustment

   float ExposureBias = 2.0;
   vec3 curr = Uncharted2Tonemap(ExposureBias * color);

   vec3 whiteScale = 1.0 / Uncharted2Tonemap(vec3(W));
   color = curr*whiteScale;
      
   return color;
}

// Function 2
vec3 ParallaxMapping(in vec3 position, inout vec3 normal, in float d, in int toggle)
{
    vec3 toBox = -position - BoxCenter;
    vec3 uvBoxSize = vec3(BoxSize.x, 3.5, BoxSize.z);

    vec2 textureCoords = (abs(normal.x) == 1.0) ? (toBox.zy / uvBoxSize.zy) : vec2(0.0);
    textureCoords += (abs(normal.y) == 1.0) ? (toBox.xz / uvBoxSize.xz) : vec2(0.0);
    textureCoords += (abs(normal.z) == 1.0) ? (toBox.xy / uvBoxSize.xy) : vec2(0.0);
    textureCoords = (textureCoords * 0.5 + 0.5) + vec2(0.0, 0.5);   
    
    float height = SampleTexture(textureCoords);
    
    vec3 tangent = normalize(dFdy(textureCoords).y * dFdx(position) - dFdx(textureCoords).y * dFdy(position));
    vec3 temp = cross(normal, tangent);
    tangent = cross(temp, normal);
    tangent = normalize(tangent);
    vec3 binormal = cross(-tangent, normal);
     
    vec3 viewTangentSpace = normalize(gCameraPosition - position) * mat3(tangent, binormal, normal);
   
    const float scale = 0.04;
    const float bias = 0.02; 
	vec2 textureOffset = (viewTangentSpace.xy * (height * scale - bias)) / viewTangentSpace.z;
    
    // Steep Parallax Mapping and POM
    const float numberOfSamples = 10.0;
	const float stepSize = 1.0 / numberOfSamples;
	vec2 deltaOffset = textureOffset / numberOfSamples;
	float currentLayerDepth = 0.0;
    float currentDepth = 0.0;       
    
    if ((toggle == 3) || (toggle == 4))  // Steep Parallax Mapping or POM
    {   
        for(float i = 0.0; i <= numberOfSamples; ++i)
        {
            currentDepth -= stepSize;           
        	textureCoords += deltaOffset;
            height = SampleTexture(textureCoords);
            
            if(currentDepth < height) 
            {	
                break;
            }
        }
        
        if (toggle == 4) // POM
        {
            vec2 previousTextureCoords = textureCoords - deltaOffset;
            float collisionDepth = height - currentDepth;
            float previousDepth = SampleTexture(previousTextureCoords) - currentDepth - stepSize;

            float weight = collisionDepth / (collisionDepth - previousDepth);
            textureCoords = mix(textureCoords, previousTextureCoords, weight);    
            height = SampleTexture(textureCoords);
        }
    }
    else if(toggle == 2) // Parallax Mapping  
    {
        textureCoords = textureCoords + textureOffset; 
        height = SampleTexture(textureCoords);
    }

    // Final Display
    if((toggle >= 1) && (toggle <= 4))
    {
        normal = CalculateNormalMapNormal(textureCoords, height, normal, tangent, binormal);        
    }
    return SampleTexture(textureCoords) * vec3(0.85, 0.85, 1.0);
}

// Function 3
float disMap(vec3 p){
    float dis = 0.0;
    float time = mod(iTime * speed + 30.0,200.0);
    float t = (sin(time) * 2.0);
    vec4 texMap;
    
	if(!texBend)
        texMap = texture(iChannel1,(p.zx + vec2(1,1))) *.1;
    if(doRotate){
        float rot = time * mPi * .20 * rotSpeed;
        p.xy *= mat2(cos(rot),-sin(rot),sin(rot),cos(rot));
        p.yz *= mat2(cos(rot),-sin(rot),sin(rot),cos(rot));
        p=normalize(p);
    }
    p = makeItNoisy(time) * p;

	if(texBend)
        texMap = texture(iChannel1,p.xy);
    
	
    dis = length(texMap) ;
    
    dis += noise(p + time);
    
    return dis;
}

// Function 4
vec2 map1(vec3 pos)
{
    pos = rotateVec2(pos);

    vec2 obj;
    
    float ypos = getCylYPos();
    
    vec2 c1 = vec2(sdCylinder(pos - vec3(-ce/2., ypos, 0.), vec2 (cr, cl)), C1_OBJ);
    vec2 c2 = vec2(sdCylinder(pos - vec3(ce/2., -ypos, 0.), vec2 (cr, cl)), C2_OBJ);
    vec2 box = vec2(sdBox(pos - vec3(0., boxYPos, 0.), boxSize), BOX_OBJ);
    obj = opU(c1, c2);
    obj = opU(obj, box);
 
    //obj.x = max(obj.x, pos.z);

    return obj;
}

// Function 5
vec3 rgbToHsluv(vec3 tuple) {  return lchToHsluv(rgbToLch(tuple)); }

// Function 6
vec3 hsluv_distanceFromPole(vec3 pointx,vec3 pointy) {
    return sqrt(pointx*pointx + pointy*pointy);
}

// Function 7
vec3 voronoiSphereMapping(vec3 n){
	vec2 uv=vec2(atan(n.x,n.z),acos(n.y));
    return getVoronoi(1.5*uv);}

// Function 8
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    // Project ray direction on to the unit cube.
    vec3 absRayDir = abs(rayDir);
    rayDir /= max(absRayDir.x, max(absRayDir.y, absRayDir.z));
    
    // Get the index of the current face being rendered.
    
    int faceIndex = 0;

    if(absRayDir.y > absRayDir.x && absRayDir.y > absRayDir.z)
    {
        faceIndex = 2;
    }
    else if(absRayDir.z > absRayDir.x && absRayDir.z > absRayDir.y)
    {
        faceIndex = 4;
    }

    if(rayDir[faceIndex / 2] > 0.)
        faceIndex |= 1;

    fragColor = vec4(0);
    
    vec2 uv = fragCoord.xy / 1024.;

    const float overlap = .1;

    if(faceIndex == 0)
    {

       uv *= 1. + overlap;
        
        vec2 ouv = uv;

        // Calculate a height map.

        //uv = abs(uv - .5) * 1.5;

        for(int i = 0; i < 120; ++i)
        {
            uv -= sin(uv.yx * 16. + iTime / 3.) * .007 + .01;
            uv.x -= .001;
        }

        uv = fract(uv * 5.5) - .5;

        fragColor.r = (1. - smoothstep(-.1, .7, length(uv)-.1));

        fragColor.r += texture(iChannel0, uv).r * .1;
        fragColor.r += texture(iChannel0, uv * 2.).r * .05;

        fragColor.r += texture(iChannel0, ouv * 2.).r * .005;
        fragColor.r += texture(iChannel0, ouv * 3.).r * .01;

        fragColor.r *= sin(ouv.x * 3.1416 * 2. + 1. * sin(ouv.y * 3.14159 * 2. - iTime / 2.))*.7 + .75;

        fragColor.r += textureLod(iChannel1, ouv * 2. - vec2(1,0)*iTime / 70. + .5, 0.).r * .2;
        fragColor.r -= textureLod(iChannel1, ouv * 2. - vec2(1,0)*iTime / 100., 0.).r * .08;

        float z = textureLod(iChannel0, ouv / 4., 0.).r * 6.;

        fragColor.r += pow(max(0., 1. - fragColor.r), 5.) *
                textureLod(iChannel1, ouv * 2. - vec2(1,0)*iTime / 70. + .5, 0.).r * .1 * (.5 + .5 * cos(z + iTime * 15.));

        fragColor.r = clamp(fragColor.r*.9, 0., 1.);

        // Calculate colour.

        fragColor.gba = textureLod(iChannel2, uv / 7., 1.).rgb * mix(vec3(1), textureLod(iChannel2, uv * 2., 1.).bgr, .5);
    }
    else if(faceIndex == 1)
    {

        fragColor = vec4(0);

        float wsum = 0.;
                
        for(int y = -1; y <= +1; ++y)
            for(int x = -1; x <= +1; ++x)
            {
                vec2 uv2 = uv * 2. - 1. + vec2(x, y) * 2.;
                float mask = 1. - smoothstep(0., overlap, length(max(abs(uv2)-vec2(1.), 0.)));
        		fragColor += textureLod(iChannel3, mapCoord(uv2 * 1. / (1. + overlap * 2.)), 0.) * mask;
                wsum += mask;
            }
        
        fragColor /= wsum;

    }
}

// Function 9
vec3 hsluv_fromLinear(vec3 c) {  return vec3( hsluv_fromLinear(c.r), hsluv_fromLinear(c.g), hsluv_fromLinear(c.b) ); }

// Function 10
vec4 mipmap(vec2 uv) {
    return texture(iChannel0, uv);
}

// Function 11
float materialHeightMap( const vec2 grooves, const vec2 coord ) {
	return min( grooveHeight( grooves.x, 0.01, coord.x ), grooveHeight( grooves.y, 0.01, coord.y ));
}

// Function 12
float map(vec2 p, inout int index) {
    vec2 mouse = (2.0*iMouse.xy-iResolution.xy)/iResolution.y;
    float k = 1.0;
    if (iMouse.z > 0.0) {
        p -= mouse;
        k = dot(p,p);
        p /= k;
        p += mouse;
    }
    const float strong_factor = 4.;
#ifdef DrawVerticalSections
    vec3 q = vec3(p.x, iTime*0.1, p.y + 1.);
#else
    vec3 q = vec3(p, mod(iTime*0.2 - 0.01, SECTION_HEIGHT - 0.02) + 0.01);
#endif
    return k*DE(q, index) * strong_factor;
    
}

// Function 13
vec2 imageWarp_uv(vec2 uv)
{
    vec2 A, B, C, D;
    get_points(A,D,C,B);
    A*=0.5;B*=0.5;C*=0.5;D*=0.5;
    A+=0.5;B+=0.5;C+=0.5;D+=0.5;
    vec2 a, b, c, d;
    a=vec2(0.);
    b=vec2(1.,0.);
    c=vec2(1.,1.);
    d=vec2(0.,1.);
    
    float LU[8*8]=float[](
        a.x,a.y,1.,0.,0.,0.,-a.x*A.x,-a.y*A.x,
        b.x,b.y,1.,0.,0.,0.,-b.x*B.x,-b.y*B.x,
        c.x,c.y,1.,0.,0.,0.,-c.x*C.x,-c.y*C.x,
        d.x,d.y,1.,0.,0.,0.,-d.x*D.x,-d.y*D.x,
        0.,0.,0.,a.x,a.y,1.,-a.x*A.y,-a.y*A.y,
        0.,0.,0.,b.x,b.y,1.,-b.x*B.y,-b.y*B.y,
        0.,0.,0.,c.x,c.y,1.,-c.x*C.y,-c.y*C.y,
        0.,0.,0.,d.x,d.y,1.,-d.x*D.y,-d.y*D.y);
    
    float LB[8]=float[](A.x,B.x,C.x,D.x,A.y,B.y,C.y,D.y);
    int piv[8];
    LUDecomposition(piv,LU);
    
    
    float LC[8];
    vec2 tuv=vec2(0.);
    if(solve(piv,LB,LU,LC))
    {
        tuv.x=(LC[0]*uv.x+LC[1]*uv.y+LC[2])/(LC[6]*uv.x+LC[7]*uv.y+1.0001);
        tuv.y=(LC[3]*uv.x+LC[4]*uv.y+LC[5])/(LC[6]*uv.x+LC[7]*uv.y+1.0001);
    }

    return tuv;
}

// Function 14
vec3 i_spheremap_32( uint data )
{
    vec2 v = unpackSnorm2x16(data);
    float f = dot(v,v);
    return vec3( 2.0*v*sqrt(1.0-f), 1.0-2.0*f );
}

// Function 15
float map(vec3 p) {
    float k = 0.5 * 2.0;
	vec3 q = (fract((p - vec3(0.25, 0.0, 0.25))/ k) - 0.5) * k;
    vec3 s = vec3(q.x, p.y, q.z);
    float d = udRoundBox(s, vec3(0.1, 1.0, 0.1), 0.05);
    
    k = 0.5;
    q = (fract(p / k) - 0.5) * k;
    s = vec3(q.x, abs(p.y) - 1.5, q.z);
    float g = udRoundBox(s, vec3(0.17, 0.5, 0.17), 0.2);
    
    float sq = sqrt(0.5);
    vec3 u = p;
    u.xz *= mat2(sq, sq, -sq, sq);
    d = max(d, -sdBoxXY(u, vec3(0.8, 1.0, 0.8)));
    
    return smin(d, g, 16.0);
}

// Function 16
vec4 o5268_input_color_map(vec2 uv) {
vec2 o5242_0_wat = abs((scale((uv), vec2(0.5+p_o183339_cx, 0.5+p_o183339_cy), vec2((9.0/8.0), (9.0/8.0)))) - 0.5);
float o5242_0_d = o5242_0_wat.x+o5242_0_wat.y;vec4 o5242_0_1_rgba = o5242_gradient_gradient_fct(fract(2.0*(5.0/3.0)*o5242_0_d));
vec4 o183339_0_1_rgba = o5242_0_1_rgba;
vec4 o5264_0_1_rgba = o5264_f(o183339_0_1_rgba);

return o5264_0_1_rgba;
}

// Function 17
float map(vec3 p){
    // rotate
    float r = 3.14159*sin(p.z*0.15)+T*0.25;
    R = mat2(cos(r), sin(r), -sin(r), cos(r));
    p.xy *= R;
    vec3 op = p;
    
    // per-cell random values
    float h = hash(floor(p.x+p.y+p.z));
    float h2 = 3.141*hash(floor(-p.x-p.y-p.z));
    
    // bumpy
    #ifdef BUMPY
    float f = pow(texture(iChannel2, p*0.1).b,4.0);
   	vec3 dd = vec3(sin(p.z*71.), cos(p.x*73.), -cos(p.y*77.))
               -0.6*vec3(cos(p.y*141.), sin(p.z*143.), -sin(p.x*147.));
    p = mix(p, p-dd*0.005, f);
    #endif
    
    // repeat lattice
    const float a = 1.0;
    p = mod(p, a)-a*0.5;
    
    // primitives
    // center sphere
    float v = length(p)-(0.02+(0.18*h*(0.6+0.4*sin(3.0*T+h2)) ));
    // four connecting cylinders
    v = smin(v, length(p.xy+0.01*sin(-3.2*T+13.0*op.z))-0.03, 0.2);
    v = smin(v, length(p.xz+0.01*cos(-4.1*T+11.0*(op.y-op.z)))-0.03, 0.2);
    v = smin(v, length(p.yz+0.01*sin(-5.0*T-8.0*(op.x-op.z)))-0.03, 0.2);
    
    return v;
}

// Function 18
float map(vec3 p)
{
    float d0=dot(sin(p),cos(p.yzx))  +.3 * dot(sin(p*3.),cos(p.yzx*3.));
    float d2=length(p)-9.;
    float d=smax(-d0,d2,3.);   
    return clamp(d,-.5,+.5);
}

// Function 19
vec4   luvToXyz(float x, float y, float z, float a) {return   luvToXyz( vec4(x,y,z,a) );}

// Function 20
vec4 distUV(vec3 pos)
{
    //pos+=.00015*getRand(pos*1.3).xyz*4.;
    //pos+=.00006*getRand(pos*3.).xyz*4.;
    //pos+=.00040*getRand(pos*.5).xyz*4.;
    vec3 p1,p2,p3;
    float d = 10000.;
    
    // sphere in the middle
	//d=min(d,distSphere(pos,.79));
    
    // start with an icosahedron subdivided once
    getIcosaTri(pos, p1, p2, p3);
    //getTriSubDiv(pos, p1, p2, p3);
    // always sort by X, then Y, then Z - to get a unique order of the edges
    sortXYZ(p1,p2,p3);
    vec2 uv;
    d=min(d,distTruchet(pos, p1,p2,p3,.2,uv));
    
    return vec4(d,uv,0);
}

// Function 21
void mainCubemap( out vec4 O, vec2 U,  vec3 C, vec3 D )
{
    ivec2 I = ivec2(U)/128;                                      // tile bi-Id
    vec3 A = abs(D);
    int  f = A.x > A.y ? A.x > A.z ? 0 : 2 : A.y > A.z ? 1 : 2,  // faceId
         s = I.x + 8*I.y,                                        // tile Id
         i = int(1023.* T( mod(U,128.) ) );                      // discretize signal
    if ( D[f] < 0. ) f += 3;                                     // full face Id.
    O = f<4 ? vec4( equal( ivec4(i), s + 64*ivec4(0,1,2,3) + 256*f )) // isolate one value within 256                                
            : vec4(T(U/R*128.),0,0,0);                           //  2 useless : free to show the image ! 

// O = .5*vec4(  ( s + 64*ivec4(0,1,2,3) + 256*f) );    // cubeMap calibration
}

// Function 22
float map4( in vec3 p )
{
	vec3 q = p*cloudScale;
	float f;
    f  = .50000*noise( q ); q = q*2.02;
    f += .25000*noise( q ); q = q*2.03;
    f += .12500*noise( q ); q = q*2.01;
    f += .06250*noise( q );
	return clamp( 1.5 - p.y - 2. + 1.75*f, 0., 1. );
}

// Function 23
vec3 Uncharted2ToneMapping(vec3 color)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	float W = 11.2;
	float exposure = 2.;
	color *= exposure;
	color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
	float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
	color /= white;
	color = pow(color, vec3(1. / gamma));
	return color;
}

// Function 24
Object map(vec3 p) {
    Object o = NewObject;
    
    vec3 q = p;
    
	float off = 20.;
    float sz = 40.;
    
    p.z += sz*0.25;
    
    p.x -= off;
    
    p.x += off*2.;
    
    p.z -= sz*.25;
    p.x -= off;
    
    o = omin(o,mapG(p));
    

    vec3 y = p;    
    p=abs(q);
    q = y;
    
    
    p.xz *= rot(0.125*pi);
    p.xz -= 0.5;
    
    q.xz *= rot(-0.25*pi);
    
    q.x = pmod(q.x, 0.5);
    q.z -= 1.2;
    o = omin(o,sdBox(q,vec3(0.2,15.,0.2)), 4.);
    
    
    o.didHit = true;
    o.d *= 0.6;
    return o;
}

// Function 25
vec3 uv2dir(Camera cam,vec2 uv//get RayDirection
){return normalize(vec3(uv,cam.focalLength))*rotationMatrix(cam.rot);}

// Function 26
float mapSeedNoLight(vec2 f)
{
    DecodeData(texelFetch( iChannel0, ivec2(f),0), seedCoord, seedColor);
    return length((floor(seedCoord)-floor(f)))-seedColor.z*circSizeMult*iResolution.x;
}

// Function 27
vec3 hsluv_fromLinear(vec3 c) {
    return vec3( hsluv_fromLinear(c.r), hsluv_fromLinear(c.g), hsluv_fromLinear(c.b) );
}

// Function 28
vec3 lchToHpluv(float x, float y, float z) {return lchToHpluv( vec3(x,y,z) );}

// Function 29
vec3 bumpMap(sampler2D tex, in vec3 p, in vec3 n, float bumpfactor)
{
    
   
    const vec3 eps = vec3(0.001, 0., 0.);//I use swizzling here, x is eps
    float ref = getGrey(triPlanar(tex, p, n));//reference value 
    
    vec3 grad = vec3(getGrey(triPlanar(tex, p - eps, n)) - ref,
                     //eps.yxz means 0.,0.001, 0. "swizzling
                     getGrey(triPlanar(tex, p - eps.yxz, n)) - ref,
                     getGrey(triPlanar(tex, p - eps.yzx, n)) - ref)/eps.xxx;
    
    //so grad is the normal...then he does:
    grad -= n*dot(grad, n);//takes the dot of the surface normal 

    return normalize(n + grad*bumpfactor);
}

// Function 30
vec2 uv_polar(vec2 domain, vec2 center){
   vec2 c = domain - center;
   float rad = length(c);
   float ang = atan(c.y, c.x);
   return vec2(ang * pi2_inv, rad);
}

// Function 31
float rmap(vec2 uv, RSet2 rs) {
    return RAND(map(uv, rs.q, rs.l), rs.r);
}

// Function 32
int uv2idx(ivec2 uv, int width)
{
	return uv.x + uv.y*width;    
}

// Function 33
float MapThorns( in vec3 pos )
{
	return pos.y * .21 - ThornVoronoi(pos).w  - max(pos.y-5.0, 0.0) * .5 + max(pos.y-5.5, 0.0) * .8;
}

// Function 34
vec3 triPlanarMapCatRom(sampler2D inTexture, float contrast, vec3 normal, vec3 position, vec2 texResolution)
{
    vec3 signs = sign(normal);
    
    vec3 xTex = SampleTextureCatmullRom(inTexture, (position).yz, texResolution, 0.0, 0).rgb;
    vec3 yTex = SampleTextureCatmullRom(inTexture, (position).xz, texResolution, 0.0, 0).rgb;
    vec3 zTex = SampleTextureCatmullRom(inTexture, -(position).xy, texResolution, 0.0, 0).rgb;
    
    vec3 weights = max(abs(normal) - vec3(0.0, 0.4, 0.0), 0.0);
    weights /= max(max(weights.x, weights.y), weights.z);
    float sharpening = 10.0;
    weights = pow(weights, vec3(sharpening, sharpening, sharpening));
    weights /= dot(weights, vec3(1.0, 1.0, 1.0));
  
    return clamp(vec3(xTex*weights.x + yTex*weights.y + zTex*weights.z), vec3(0), vec3(1));
}

// Function 35
vec4 tMapSm(samplerCube iCh, vec3 p){
 
    // Using the 3D coordinate to index into the cubemap and read
    // the isovalue. Basically, we need to convert Z to the particular
    // square slice on the 2D map, the read the X and Y values. 
    //
    // mod(p.xy, 100), will read the X and Y values in a square, and 
    // the offset value will tell you how far down (or is it up) that
    // the square will be.
    vec2 offset = mod(floor(vec2(p.z, p.z/10.)), vec2(10, 10));
    vec2 uv = (mod(p.xy, 100.) + offset*100. + .5)/cubeMapRes;
    
    // Back Z face -- Depending on perspective. Either way, so long as
    // you're consistant. I noticed the Y values need to be flipped...
    // I'd like to arrange so that it's not necessary, but it might be
    // and internal thing, so I'm not sure how, yet.
    //
    // You could also use one of the newer texture functions that 
    // doesn't require the ".5" and "iChannelRes0" division, but I'm
    // keeping it oldschool. :) Actually, if the newer ones are
    // superior, let us know.
    return texture(iCh, vec3(fract(uv) - .5, .5));
}

// Function 36
vec4 cubemap( sampler2D sam, in vec3 d )
{
    // intersect cube
    vec3 n = abs(d);
    vec3 v = (n.x>n.y && n.x>n.z) ? d.xyz: 
             (n.y>n.x && n.y>n.z) ? d.yzx:
                                    d.zxy;
    // project to face    
    vec2 q = v.yz/v.x;
    
    // undistort in the edges
    #if METHOD==-1
	int mode = (int(iTime)%5);
    #else
    const int mode = METHOD;
    #endif
    if( mode==0 ) {}
    if( mode==1 ) q  = atan(q*tan(0.868734829276))/0.868734829276;
    if( mode==2 ) q *= 1.50 - 0.50*abs(q);
    if( mode==3 ) q *= 1.45109572583 - 0.451095725826*abs(q);
    if( mode==4 ) q *= 1.25 - 0.25*q*q;

    // sample texture
    vec2 uv = 0.5+0.5*q;
    vec4 c = texture( iChannel0, uv );
    // add wireframe
	vec2 w = 1.0-smoothstep(0.96,1.0,cos(10.0*uv*6.2831));
    return c*w.x*w.y;
}

// Function 37
vec3 lumaBasedReinhardToneMapping(vec3 color)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= toneMappedLuma / luma;
    color = pow(color, vec3(1. / gamma));
    return color;
}

// Function 38
float hsluv_maxChromaForLH(float L, float H) {

    float hrad = radians(H);

    mat3 m2 = mat3(
         3.2409699419045214  ,-0.96924363628087983 , 0.055630079696993609,
        -1.5373831775700935  , 1.8759675015077207  ,-0.20397695888897657 ,
        -0.49861076029300328 , 0.041555057407175613, 1.0569715142428786  
    );
    float sub1 = pow(L + 16.0, 3.0) / 1560896.0;
    float sub2 = sub1 > 0.0088564516790356308 ? sub1 : L / 903.2962962962963;

    vec3 top1   = (284517.0 * m2[0] - 94839.0  * m2[2]) * sub2;
    vec3 bottom = (632260.0 * m2[2] - 126452.0 * m2[1]) * sub2;
    vec3 top2   = (838422.0 * m2[2] + 769860.0 * m2[1] + 731718.0 * m2[0]) * L * sub2;

    vec3 bound0x = top1 / bottom;
    vec3 bound0y = top2 / bottom;

    vec3 bound1x =              top1 / (bottom+126452.0);
    vec3 bound1y = (top2-769860.0*L) / (bottom+126452.0);

    vec3 lengths0 = hsluv_lengthOfRayUntilIntersect(hrad, bound0x, bound0y );
    vec3 lengths1 = hsluv_lengthOfRayUntilIntersect(hrad, bound1x, bound1y );

    return  min(lengths0.r,
            min(lengths1.r,
            min(lengths0.g,
            min(lengths1.g,
            min(lengths0.b,
                lengths1.b)))));
}

// Function 39
float map(vec3 p) {
    vec3 q = p + 0.2*vec3(3.0, 0.3, 5.0)*mod(iTime,3600.0)*2.0;
    float n = 0.0, f = 0.5;
    n += f*noise(q); q *= 3.001; f *= 0.333;
    n += f*noise(q); q *= 3.002; f *= 0.332;
	n += f*noise(q);
    return n;
}

// Function 40
vec3 uvtomap(vec2 uv, vec3 div) {
    vec2 fuv=floor(uv*div.xy);
    float pz=fuv.x+fuv.y*div.x;
    return vec3(fract(uv*div.xy)*vec2(2.0)-vec2(1.0),(pz/(div.x*div.y))*2.0-1.0)*3.0;
}

// Function 41
vec3 doBumpMap(in vec3 p, in vec3 n, float bumpfactor, inout float edge){
    
    vec2 e = vec2(2.5/iResolution.y, 0);
    
    float f = bumpFunction(p); 
    
    // Samples about the hit point in each of the axial directions.
    float fx = bumpFunction(p - e.xyy); // Same for the nearby sample in the X-direction.
    float fy = bumpFunction(p - e.yxy); // Same for the nearby sample in the Y-direction.
    float fz = bumpFunction(p - e.yyx); // Same for the nearby sample in the Y-direction.

    // Samples from the other side.
    float fx2 = bumpFunction(p + e.xyy); // Same for the nearby sample in the X-direction.
    float fy2 = bumpFunction(p + e.yxy); // Same for the nearby sample in the Y-direction.
    float fz2 = bumpFunction(p + e.yyx); // Same for the nearby sample in the Y-direction.
    
    // We made three extra function calls, so we may as well use them. Taking measurements
    // from either side of the hit point has a slight antialiasing effect.
    vec3 grad = (vec3(fx - fx2, fy - fy2, fz - fz2))/e.x/2.;   

    // Using the samples to provide an edge measurement.
    edge = abs(fx + fy + fz + fx2 + fy2 + fz2 - 6.*f);
    //edge = abs(fx + fx2 - f*2.) + abs(fy + fy2 - f*2.)+ abs(fz + fz2 - f*2.);
    edge = smoothstep(0., 1., edge/e.x);
          
    grad -= n*dot(n, grad);          
                      
    return normalize( n + grad*bumpfactor );
	
}

// Function 42
float map_f_ver(vec3 pos, vec2 delta, float n)
{
    return length(vec2(mod(pos.x + delta.x, fe) - fe*0.6, pos.z + delta.y - fr*sin((pos.y + fe*2. + fe*floor(pos.x/fe))/fe*pi))) - fr*fds*0.86;
}

// Function 43
vec3 bump_mapping(vec3 p, vec3 n, float weight)
{
    vec2 e = vec2(2./iResolution.y, 0); 
    vec3 g=vec3(bump(p-e.xyy, n)-bump(p+e.xyy, n),
                bump(p-e.yxy, n)-bump(p+e.yxy, n),
                bump(p-e.yyx, n)-bump(p+e.yyx, n))/(e.x*2.);  
    g=(g-n*dot(g,n));
    return normalize(n+g*weight);
}

// Function 44
float map(vec3 p, inout int matID, vec3 playerPos, bool drawPlayer) {
    float res = FLT_MAX;
    
    // islands offset
    vec3 ip = p;
    ip.y += 0.8;
    
#if !TEST
    ip.y -= perlinNoise2D(ip.xz, 1.0, 0.3, 1, 697);
    ip.y += (perlinNoise2D(ip.xz+vec2(50.0), 0.5, 5.0, 3, 769)-4.1)*(1.0-smoothstep(6.9, 7.5, ip.y));
#endif
    
    // left island
    propose(res, matID, blend(cone(rX(ip, 180.0), 10.0, 7.5),
                              cylinder(p-vec3(0.0, 6.5, 0.0), 13.0, 0.15), 3.0), 3);
        
#if !TEST  
    propose(res, matID, blend(cone(rX(ip-vec3(1.0, 2.5, -12.0), 180.0), 5.5, 5.0),
                              cylinder(p-vec3(1.0, 6.5, -12.0), 7.0, 0.15), 3.0), 3);
    propose(res, matID, blend(cone(rX(ip-vec3(5.0, 4.5, -18.0), 180.0), 4.0, 3.0),
                              cylinder(p-vec3(5.0, 6.5, -18.0), 5.0, 0.15), 3.0), 3);
    
    // right island
    propose(res, matID, blend(cone(rX(ip-vec3(23.0, 2.5, -40.0), 180.0), 6.0, 5.0),
                              cylinder(p-vec3(23.0, 6.4, -40.0), 6.95, 0.2), 3.0), 3);
    propose(res, matID, blend(cone(rX(ip-vec3(18.0, 4.25, -45.0), 180.0), 4.0, 3.25),
                              cylinder(p-vec3(18.0, 6.4, -45.0), 4.45, 0.2), 3.0), 3);
    
    // house
    vec3 z = rY(vec3(0,0,1), 50.0);
    float cyl1 =  cylinder(rX(rY(p-vec3(-1.0, 10.9, -3.0)+24.0*z, 35.0), 90.0), 22.1, 5.0);
    float cyl2 =  cylinder(rX(rY(p-vec3(-1.0, 10.9, -3.0)-24.0*z, 35.0), 90.0), 22.1, 5.0);
    float house = box(rY(p-vec3(-1.0, 10.5, -3.0), 35.0), vec3(2.0, 3.0, 2.0));
    house = subtractionDist(house, cyl1);
    house = subtractionDist(house, cyl2);
    propose(res, matID, house, 4);
    float roof = blend(triangularPrism(rY(p-vec3(-1.0, 14.0, -3.0), 35.0),
                                        vec2(1.0,2.0), 20.0),
                       cylinder(rX(rY(p-vec3(-1.0, 14.4, -3.0), 35.0), 90.0),
                                0.2, 1.7), 0.6);
    propose(res, matID, roof, 5);
    
    // chimney
    propose(res, matID, roundedbox(rY(p-vec3(-0.8, 13.5, -1.0), 35.0), vec3(0.15, 1.2, 0.15), 0.05), 6);
    
    // door
    propose(res, matID, box(rY(p-vec3(-1.0, 8.5, -5.0), 35.0), vec3(0.4, 1.05, 0.4)), 7);
    propose(res, matID, sphere(p-vec3(-1.0, 8.5, -5.52), 0.05), 8);
    
    // bridge
    vec3 x = rY(vec3(1,0,0), 50.0);
    vec3 cen = vec3(-10.5,7.5,0.0)+36.89*x;
    vec3 bp = p + vec3(0.0, 0.05, 0.0);
    vec2 offset = vec2(iTime/35.0, iTime/30.0);
    bp.y += perlinNoise2D(bp.xz+offset, 0.1+0.05*sin(iTime/6.0)+0.05*cos(iTime/7.0),
                          2.0, 1, 537)*(max(0.0, 1.0-dot(bp.xz-cen.xz,bp.xz-cen.xz)/59.0));
    float bridge = box(rY(bp-cen, 40.0), vec3(0.75,1.5,7.8));
    float planks = box(repeat(rY(bp-vec3(-10.5,7.5,0.0), 40.0), vec3(0.0,0.0,0.3)),
                       vec3(0.5,0.05,0.125));
    propose(res, matID, intersectionDist(planks, bridge), 9);
    
    // bridge ropes
    propose(res, matID, cylinder(rX(rY(bp-cen+0.45*z, 40.0), 90.0), 0.01, 7.7), 9);
    propose(res, matID, cylinder(rX(rY(bp-cen-0.45*z, 40.0), 90.0), 0.01, 7.7), 9);
    
    propose(res, matID, cylinder(rX(rY(bp-cen-0.25*x+0.47*z-UP, 40.0), 90.0), 0.03, 8.25), 9);
    propose(res, matID, cylinder(rX(rY(bp-cen-0.47*z-UP, 40.0), 90.0), 0.03, 8.4), 9);
    
    float ropes1 = cylinder(repeat(rY(bp-vec3(-10.5,8.0,0.0)+0.47*z, 40.0), vec3(0.0,0.0,0.9)), 0.02, 0.5);
    float ropes2 = cylinder(repeat(rY(bp-vec3(-10.5,8.0,0.0)-0.47*z, 40.0), vec3(0.0,0.0,0.9)), 0.02, 0.5);
    propose(res, matID, intersectionDist(ropes1, bridge), 9);
    propose(res, matID, intersectionDist(ropes2, bridge), 9);
    
    float disp = (sin(p.x*p.y)*sin(p.z*p.x))/40.0;
    propose(res, matID, cylinder(rX(rY(p-cen+8.25*x+0.55*z-0.5*UP, 60.0), 30.0), 0.07, 0.6)+disp, 9);
    propose(res, matID, cylinder(rX(rY(p-cen+8.3*x-0.54*z-0.5*UP, 75.0), -15.0), 0.07, 0.6)+disp, 9);
    propose(res, matID, cylinder(rX(rY(p-cen-8.25*x+0.43*z-0.5*UP, 30.0), 30.0), 0.07, 0.65)+disp, 9);
    propose(res, matID, cylinder(p-cen-8.4*x-0.47*z-0.5*UP, 0.07, 0.6)+disp, 9);

    
    // right tree
    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        vec3 tp = p;
        vec2 add = vec2(sin(p.y+fi/4.0*2.0*PI+2.0*PI*hash(36.0*fi)),
                        cos(p.y+fi/4.0*2.0*PI+2.0*PI*hash(83.0*fi)))/4.0;
        float ystep = smoothstep(8.5, 11.0, p.y);
        tp.xz += add*ystep;
        propose(res, matID, cylinder(rX(rY(tp-vec3(27.0, 10.0, -42.0), 180.0*(fi-1.0)/4.0*ystep), 40.0*ystep),
                                 0.07+(max(11.5-p.y, 0.0))/10.0, 4.0), 9);
    }
    
    // right fence
    for(int i = 0; i < 6; i++) {
        float fi = float(i);
    	vec3 off = rY(vec3(1.0,0.0,0.0), 15.0*fi-40.0);
    	vec3 nextOff = rY(vec3(1.0,0.0,0.0), 15.0*float(i+1)-40.0);
    	propose(res, matID, cylinder(p-vec3(23.0, 7.75, -40.0)-6.7*off, 0.07, 0.6), 9);
        vec3 hc = rX(rY(p-vec3(23.0, 7.75, -40.0)-6.7*(off+nextOff)/2.0, 180.0 - (15.0*(fi+0.5)-40.0)), 90.0);
        float a1 = 7.0*sin(fi*2320.0);
        float a2 = 7.0*sin(fi*235.0);
        a1 = mix(a1, -15.0, step(4.5, fi));
        a2 = mix(a2, -25.0, step(4.5, fi));
        propose(res, matID, cylinder(rX(hc, a1), 0.04, 0.87), 9);
        propose(res, matID, cylinder(rX(hc, a2)-vec3(0.0,0.0,0.4*step(fi, 4.5)), 0.04,
                                     0.87+0.1*step(4.5, fi)), 9);
    }

#endif
    
   	// left tree
    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        vec3 tp = p;
        vec2 add = vec2(sin(p.y+fi/5.0*2.0*PI+2.0*PI*hash(50.0*fi)),
                        cos(p.y+fi/5.0*2.0*PI+2.0*PI*hash(50.0*fi)))/4.0;
        float ystep = smoothstep(9.0, 11.5, p.y);
        tp.xz += add*ystep;
        propose(res, matID, cylinder(rX(rY(tp-vec3(6.0, 10.5, 8.0), 360.0*fi/5.0*ystep), 40.0*ystep),
                                 0.1+(max(11.5-p.y, 0.0))/10.0, 5.0), 9);
    }
    
    // left fence
    for(int i = 0; i < 4; i++) {
        float fi = float(i);
    	vec3 off = rY(vec3(1.0,0.0,0.0), 8.0*fi-170.0);
    	vec3 nextOff = rY(vec3(1.0,0.0,0.0), 8.0*float(i+1)-170.0);
    	propose(res, matID, cylinder(p-vec3(0.0, 7.75, 0.0)-12.5*off, 0.07, 0.6), 9);
        vec3 hc = rX(rY(p-vec3(0.0, 7.75, 0.0)-12.5*(off+nextOff)/2.0, 180.0 - (8.0*(fi+0.5)-170.0)), 90.0);
        float a1 = 7.0*sin(fi*2320.0);
        float a2 = 7.0*sin(fi*235.0);
        a1 = mix(a1, -15.0, step(2.5, fi));
        a2 = mix(a2, -25.0, step(2.5, fi));
        propose(res, matID, cylinder(rX(hc, a1), 0.04, 0.87), 9);
        propose(res, matID, cylinder(rX(hc, a2)-vec3(0.0,0.0,0.4*step(fi, 2.5)), 0.04,
                                     0.87+0.1*step(2.5, fi)), 9);
    }
    
    // apple
    float t = max(0.0, iTime -10.0);
    float appleY = min(0.5*9.81*t*t, 5.0);
    propose(res, matID, sphere(p-vec3(8.0, 12.65-appleY, 7.6), 0.15), 10);
    
    // player
    if (drawPlayer) {
        propose(res, matID, sphere(p-playerPos, 0.25), 2);
    }
    
#if BOIDS
    // Sometimes the boids fly backwards so one of these angles is wrong?
    for (int i = 0; i < NUM_BOIDS; i++) {
        vec3 v = texture(iChannel2, vec2(float(i), 1.0)/iChannelResolution[2].xy).xyz;
        float r = length(v);
        float theta = degrees(acos(v.z/r));
        float phi = degrees(atan(v.y, v.x))+180.0;
        vec3 pos = texture(iChannel2, vec2(float(i)/iChannelResolution[2].x, 0.0)).xyz;
        propose(res, matID,
                cone(rX(rY(p-pos, phi), theta), 0.25, 0.5),
                11);
    }
#endif
    
    return res;
}

// Function 45
vec2 uv2LocalUV(vec2 uv, vec2 centerPoint){
  vec2 localUV = TILE_LOCAL_LENGTH * vec2((uv.x - centerPoint.x)/ TILE_LENGTH, 
                                          (uv.y - centerPoint.y)/ TILE_LENGTH);
  return errorCheckLocalUV(localUV, localUV);
}

// Function 46
vec3 eMap(vec3 rd, vec3 sn){

    vec3 tx = tex3D(iChannel0, rd, sn);
    return smoothstep(.15, .5, tx); 
    
}

// Function 47
float remap(float value, float inputMin, float inputMax, float outputMin, float outputMax)
{
    return (value - inputMin) * ((outputMax - outputMin) / (inputMax - inputMin)) + outputMin;
}

// Function 48
float map2(vec3 p, vec3 div) {
 
    #if USE_3D_VOLUME
    	float d = mapdist(p, div);
    #else
    	float d = map(p, time);
    #endif
    
    d=max(d, limit(p));
    d=smin(d, background(p), 0.2);
    return d;
}

// Function 49
vec2 map( vec3 p, float camtime )
{
  vec2 ret = vec2(min(
                        // repeatedSphere(p, vec3(2.0), 0.25),
                        // repeatedBox(p, vec3(0.811), 0.071, 0.031)  ),
                        repeatedBox(p, vec3(1.2), 0.071, 0.031),
                        max( repeatedGrid(p, vec3(1.2), 0.0271),
                             repeatedBox(p+0.05,  vec3(0.1), 0.015, 0.0035)
                           )
                        //repeatedBox(p, vec3(0.524), 0.041, 0.017)
                ), 1.0);

  attributedUnion(ret, repeatedCone(vec3(p.x, p.y, p.z - camtime), 
                                    vec3(1.611, 1.9, 5.0), 
                                    normalize(vec2(1.0, 0.3))), 
                       2.0);

  attributedUnion(ret, max( repeatedSphere(p+0.25, vec3(2.0), 0.25),
                            repeatedSphere(p, vec3(0.1), 0.046)
                          ), 3.0);

  attributedIntersection(ret, -sdSphere(p - camo, 0.03), -1.0);
  return ret;
}

// Function 50
vec3 mapCrapInTheAir( in vec3 pos, in vec3 cur)
{
    vec3 res = cur;
    
    ivec2 id = ivec2(floor((pos.xz+2.0)/4.0));
    pos.xz = mod(pos.xz+2.0,4.0)-2.0;
    float dm = 1e10;
    for( int i=ZERO; i<4; i++ )
    {
        vec3 o = vec3(0.0,3.2,0.0);
        o += vec3(1.7,1.50,1.7)*(-1.0 + 2.0*hash3(float(i)));
        o += vec3(0.3,0.15,0.3)*sin(0.3*iTime + vec3(float(i+id.y),float(i+3+id.x),float(i*2+1+2*id.x)));
        float d = length2(pos - o);
        dm = min(d,dm);
    }
    dm = sqrt(dm)-0.02;
    
    if( dm<res.x )
        res = vec3( dm,MAT_CITA,0);
    
    return res;
}

// Function 51
SDObject SDOMap(in SDObject o,vec3 p)
{
  //  o.ray=ray;
  //  o.ray.ro+=o.pos;
    //o.d=iSphere(o.ray.ro,o.ray.rd);
    o.d=sdSphere(o.mpos=p-o.pos,1.);
    return o;
}

// Function 52
float remap (float x, float a, float b, float c, float d) 
{
	return (x-a)/(b-a)*(d-c) + c; 
}

// Function 53
vec3 aces_tonemap(vec3 color){	
	mat3 m1 = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
	);
	mat3 m2 = mat3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
	);
	vec3 v = m1 * color;    
	vec3 a = v * (v + 0.0245786) - 0.000090537;
	vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
	return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));	
}

// Function 54
float map(vec3 p)
{
    float radius = 0.75;
    
    // Transform coordinate space so spheres repeat
    vec3 q = fract(p) * 2.0 - 1.0;

    
    // Signed distance of sphere
    return sphere(q, radius);
}

// Function 55
vec3 NOISE_volumetricRoughnessMap(vec3 p, float rayLen)
{
    float ROUGHNESS_MAP_UV_SCALE = 6.00;//Valid range : [0.1-100.0]
    vec4 sliderVal = vec4(0.5,0.85,0,0.5);
    ROUGHNESS_MAP_UV_SCALE *= 0.1*pow(10.,2.0*sliderVal[0]);
    
    float f = iTime;
    const mat3 R1  = mat3(0.500, 0.000, -.866,
	                     0.000, 1.000, 0.000,
                          .866, 0.000, 0.500);
    const mat3 R2  = mat3(1.000, 0.000, 0.000,
	                      0.000, 0.500, -.866,
                          0.000,  .866, 0.500);
    const mat3 R = R1*R2;
    p *= ROUGHNESS_MAP_UV_SCALE;
    p = R1*p;
    vec4 v1 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021;
    vec4 v2 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021+1.204*v1.xyz;
    vec4 v3 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021+0.704*v2.xyz;
    vec4 v4 = NOISE_trilinearWithDerivative(p);
    
    return (v1
	      +0.5*(v2+0.25)
	      +0.4*(v3+0.25)
	      +0.6*(v4+0.25)).yzw;
}

// Function 56
float map(vec3 pos)
{
    float angle = mod(iTime*.8, 2.*pi);
    vec3 posr = vec3(pos.x*cos(angle) + pos.z*sin(angle), pos.y, pos.x*sin(angle) - pos.z*cos(angle));
    
    float d1 = length(posr) - 1.35;
    float d2 = pow(1.1 - sqrt(pow(posr.x, 2.)+pow(posr.y, 2.)), 2.) + pow(posr.z, 2.) - 0.1;
    float d3 = max(max(abs(posr.y), abs(posr.z)), abs(posr.x)) - 1.;
    float mx = iMouse.x/iResolution.x*1.2 - 0.3;
    float my = iMouse.y/iResolution.y*1.6 -  0.3;
    return mix(mix(d1, d2, mx), d3, my);
    return d3;
}

// Function 57
void mainCubemap( out vec4 fragColor,  in vec2 fragCoord, 
                  in vec3  fragRayOri, in vec3 fragRayDir )
{
    // cache
    if( iFrame>1 )
    {
        discard;
    }
    
    //---------------------------------
    
    // dome    
    vec3 col = vec3(0.5,0.7,0.8) - max(0.0,fragRayDir.y)*0.4;
    
    // sun
    float s = pow( clamp( dot(fragRayDir,sundir),0.0,1.0),32.0 );
    col += s*vec3(1.0,0.7,0.4)*3.0;

    // ground
    float t = (-5.0-fragRayOri.y)/fragRayDir.y;
    if( t>0.0 )
    {
        vec3 pos = fragRayOri + t*fragRayDir;
        
        vec3 gcol = vec3(0.2,0.1,0.08)*0.9;
        
        float f = 0.50*noise( pos );
              f+= 0.25*noise( pos*1.9 );
        gcol *= 0.5 + 0.5*f;

        col = mix( gcol, col, 1.0-exp(-0.0005*t) );
    }


    // clouds
    vec4 res = raymarch( fragRayOri, fragRayDir, col );
    col = col*(1.0-res.w) + res.xyz;

    
    fragColor = vec4( col, 1.0 );
}

// Function 58
float make_depthmap (vec3 color)
{
	return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// Function 59
vec3 mapRMWaterNormal(vec3 pt, float e) {
    vec3 normal;
    normal.y = mapWaterDetailed(pt);    
    normal.x = mapWaterDetailed(vec3(pt.x+e,pt.y,pt.z)) - normal.y;
    normal.z = mapWaterDetailed(vec3(pt.x,pt.y,pt.z+e)) - normal.y;
    normal.y = e;
    return normalize(normal);
}

// Function 60
vec2 map( in vec3 pos )
{
   // float sound = texture(iChannel0,vec2(0.0,0.0)).x*3.0;
    float eh =  iTime + 55.0;
    float time = eh;//iTime + 35.0;
    float freq = smoothWave(time*5.0, 10.0);
    float amp = smoothWave((time+ 8238.0)*0.14, 0.5);
    float power = smoothWave((time+ 1238.0)*0.33, 5.0)+1.0;
    //apply global warp
     pos = NormalSinPowWarpTest(pos, freq, amp,power);
    
    vec2 res = opU( vec2( sdPlane(     pos), 1.0 ),
	                vec2( sdSphere(    pos-vec3( 0.0,0.25, 0.0), 0.25 ), 46.9 ) );
    res = opU( res, vec2( sdBox(       pos-vec3( 1.0,0.25, 0.0), vec3(0.25) ), 3.0 ) );
    res = opU( res, vec2( udRoundBox(  pos-vec3( 1.0,0.25, 1.0), vec3(0.15), 0.1 ), 41.0 ) );
	res = opU( res, vec2( sdTorus(     pos-vec3( 0.0,0.25, 1.0), vec2(0.20,0.05) ), 25.0 ) );
    res = opU( res, vec2( sdCapsule(   pos,vec3(-1.3,0.10,-0.1), vec3(-0.8,0.50,0.2), 0.1  ), 31.9 ) );
	res = opU( res, vec2( sdTriPrism(  pos-vec3(-1.0,0.25,-1.0), vec2(0.25,0.05) ),43.5 ) );
	res = opU( res, vec2( sdCylinder(  pos-vec3( 1.0,0.30,-1.0), vec2(0.1,0.2) ), 8.0 ) );
	res = opU( res, vec2( sdCone(      pos-vec3( 0.0,0.50,-1.0), vec3(0.8,0.6,0.3) ), 55.0 ) );
	res = opU( res, vec2( sdTorus82(   pos-vec3( 0.0,0.25, 2.0), vec2(0.20,0.05) ),50.0 ) );
	res = opU( res, vec2( sdTorus88(   pos-vec3(-1.0,0.25, 2.0), vec2(0.20,0.05) ),43.0 ) );
	res = opU( res, vec2( sdCylinder6( pos-vec3( 1.0,0.30, 2.0), vec2(0.1,0.2) ), 12.0 ) );
	res = opU( res, vec2( sdHexPrism(  pos-vec3(-1.0,0.20, 1.0), vec2(0.25,0.05) ),17.0 ) );

    res = opU( res, vec2( opS(
		             udRoundBox(  pos-vec3(-2.0,0.2, 1.0), vec3(0.15),0.05),
	                 sdSphere(    pos-vec3(-2.0,0.2, 1.0), 0.25)), 13.0 ) );
    res = opU( res, vec2( opS(
		             sdTorus82(  pos-vec3(-2.0,0.2, 0.0), vec2(0.20,0.1)),
	                 sdCylinder(  opRep( vec3(atan(pos.x+2.0,pos.z)/6.2831,
											  pos.y,
											  0.02+0.5*length(pos-vec3(-2.0,0.2, 0.0))),
									     vec3(0.05,1.0,0.05)), vec2(0.02,0.6))), 51.0 ) );
	res = opU( res, vec2( 0.7*sdSphere(    pos-vec3(-2.0,0.25,-1.0), 0.2 ) + 
					                   0.03*sin(50.0*pos.x)*sin(50.0*pos.y)*sin(50.0*pos.z), 
                                       65.0 ) );
	res = opU( res, vec2( 0.5*sdTorus( opTwist(pos-vec3(-2.0,0.25, 2.0)),vec2(0.20,0.05)), 46.7 ) );

    res = opU( res, vec2(sdConeSection( pos-vec3( 0.0,0.35,-2.0), 0.15, 0.2, 0.1 ), 13.67 ) );

    res = opU( res, vec2(sdEllipsoid( pos-vec3( 1.0,0.35,-2.0), vec3(0.15, 0.2, 0.05) ), 43.17 ) );
        
   	res.x *= .48;
    return res;
}

// Function 61
float remap_11to01(float a)
{
    return a * 0.5 + 0.5;
}

// Function 62
vec2 zoomUv(vec2 uv, float zoom) {
    vec2 uv1 = uv;
    uv1 += .5;
    uv1 += zoom/2.-1.;
    uv1 /= zoom;
    return uv1;
    
}

// Function 63
float map_stone(vec3 pos)
{
   pos = rotateVec2(pos);
   pos.y-= 0.3;
    
   float n1 = bumpfactor*noise2(normalize(pos));
   float df1 = length(pos) - 1.90 + n1;
   float df2 = length(pos) - 1.86 + n1;
   float df = max(df1, -df2);
   if (pos.y>0.)
      df+= smoothstep(0.15, 0.35, pos.y);
   df = max(df, pos.y - 0.3);
   return df;
}

// Function 64
float Map(in vec3 p)
{
	float h = Terrain(p.xz);
    return p.y - h;
}

// Function 65
vec2 getUV(vec3 pos)
{
    vec3 nor = calcNormal(pos);
    float lon = atan(nor.x,nor.z)/3.14;
    float lat = acos(nor.y)/3.14;
    vec2 r = vec2(lat, lon);
    
    return r;
}

// Function 66
vec2 map(vec3 _p)
{
	vec2 d0 = vec2(100000.,0.);
    vec2 d = d0;
	
    vec2 sphere1 = vec2(sphere(_p,vec3(.0,0.,.0),.6),.5);
    vec2 sphere2 = vec2(sphere(_p,vec3(sn,-.5,cn),.8),.8);
    vec2 sphere3 = vec2(sphere(_p,vec3(-sn,-.5,-cn),.8),.8);
    vec2 plane1 = vec2(plane(_p,-1.),MAT_PLANE);
    vec2 light1 = vec2(sphere(_p,KL.p/30.,.1),MAT_KEYLIGHT);
    vec2 light2 = vec2(sphere(_p,FL.p/30.,.1),MAT_KEYLIGHT);
    
	d = opU(sphere2,sphere3);
	d = opI(d,sphere1);
    d = opU(d,plane1);
    
    // lights
    //d = opU(d,light1);
    //â‰¥d = opU(d,light2);
    
    #ifdef debug_sphere
    d = opU(d0,sphere1);
    d = opU(d,plane1);
    #endif
    
	return d;	
}

// Function 67
float mapDamageHigh( vec3 p ) {
    float d = map( p );
    
    float p1 = noise( p*2.3 );
    float p2 = noise( p*5.3 );
    
    float n = max( max( 1.-abs(p.z*.01), 0. )*
                   max( 1.-abs(p.y*.2-1.2), 0. ) *
                   noise( p*.3 )* (p1 +.2 )-.2 - damageMod, 0.);
    
    if( p.y < .1 ) {
        n += max(.1*(1.-abs(d)+7.*noise( p*.7 )+.9*p1+.5*p2)-4.5*damageMod,0.);
    }
    
    if( abs(n) > 0.0 ) {
        n += noise( p*11.) * .05;
        n += noise( p*23.) * .03;
    }
    
	return d + n;
}

// Function 68
vec3 rgbToYuv(vec3 rgb) {
    mat3 transformation = mat3(0.299, 0.587, 0.114, 
                               -0.14713, -0.28886, 0.436,
                              0.615, -0.51499, -0.10001);
    return transformation * rgb;
}

// Function 69
vec2 InitUV(vec2 uv)
{
	// wave
	uv.x += 0.1*sin(2.0*uv.y + 1.0*iTime);
	uv.y += 0.1*sin(2.0*uv.x + 0.8*iTime);
    return uv;
}

// Function 70
vec3 doBumpMap(in vec3 p, in vec3 n, float bumpfactor, inout float edge){
    
    // Resolution independent sample distance... Basically, I want the lines to be about
    // the same pixel with, regardless of resolution... Coding is annoying sometimes. :)
    vec2 e = vec2(2./iResolution.y, 0); 
    
    float f = bumpFunction(p); // Hit point function sample.
    
    float fx = bumpFunction(p - e.xyy); // Nearby sample in the X-direction.
    float fy = bumpFunction(p - e.yxy); // Nearby sample in the Y-direction.
    float fz = bumpFunction(p - e.yyx); // Nearby sample in the Y-direction.
    
    float fx2 = bumpFunction(p + e.xyy); // Sample in the opposite X-direction.
    float fy2 = bumpFunction(p + e.yxy); // Sample in the opposite Y-direction.
    float fz2 = bumpFunction(p+ e.yyx);  // Sample in the opposite Z-direction.
    
     
    // The gradient vector. Making use of the extra samples to obtain a more locally
    // accurate value. It has a bit of a smoothing effect, which is a bonus.
    vec3 grad = vec3(fx - fx2, fy - fy2, fz - fz2)/(e.x*2.);  
    //vec3 grad = (vec3(fx, fy, fz ) - f)/e.x;  // Without the extra samples.


    // Using the above samples to obtain an edge value. In essence, you're taking some
    // surrounding samples and determining how much they differ from the hit point
    // sample. It's really no different in concept to 2D edging.
    edge = abs(fx + fy + fz + fx2 + fy2 + fz2 - 6.*f);
    edge = smoothstep(0., 1., edge/e.x);
    
    // Some kind of gradient correction. I'm getting so old that I've forgotten why you
    // do this. It's a simple reason, and a necessary one. I remember that much. :D
    grad -= n*dot(n, grad);          
                      
    return normalize(n + grad*bumpfactor); // Bump the normal with the gradient vector.
	
}

// Function 71
vec2 errorCheckLocalUV(vec2 checkUV, vec2 returnUV){
  float errorFlg = errorCheckLocalUV(checkUV);
  return step(.1, errorFlg) * (-1. * TILE_LOCAL_LENGTH, -1. * TILE_LOCAL_LENGTH) + step(.1, 1. - errorFlg) * returnUV;
}

// Function 72
vec2 map(vec3 pos)
{

    float bluesphere = length(vec3(sin(iTime*2.)*1.5,cos(iTime/2.)*2.+.75,cos(iTime*2.)*1.5)-pos)-.25;    
    float redsphere = length(vec3(cos(iTime*2.)*1.5,sin(iTime/2.)*2.+.75,sin(iTime*2.)*1.5)-pos)-.25;

    float ico = sdIcosDodecaStar(vec3(.0,1.+sin(iTime/3.),.0)-pos,1.);
    float plane = pos.y+1.5;    

    // columns           
    vec3 colpos = pos;
    float qa = pModPolar(colpos.xz,numrep+4.);
    colpos.x-=2.4;
    float column = sdCylinder(vec3(0)-colpos, vec3(.145));
    float columnbox = sdBox(vec3(0.,0.6,0)-colpos,vec3(1.5,2.5,1.5));
    float columns = max(column,columnbox);

    //gates: block, then hole
    vec3 boxpos = pos;
    float q = pModPolar(boxpos.xz,numrep);
    boxpos.x-=5.;
    float box = sdRoundBox(boxpos,vec3(.7,3.,1.2),.24);    

    vec3 cypos = pos;
    cypos.y /= 10.;
    cypos.yz = r(cypos.yz,PI/2.);
    cypos.xz = r(cypos.xz,PI/2.);
    cypos.yz = r(cypos.yz,.5);

    q = pModPolar(cypos.yz,numrep);
    cypos.x-=.055;
    cypos.z+=.44;
    cypos.z/=6.5;

    float cyhole = sdCylinder(vec3(0.,0.,.0)+cypos,vec3(.0,.1,.21));
    float gates = max(-cyhole,box);
    float doorway = min(cyhole,box);
    doorway = min(doorway, columns);

    //SDF+matID
    vec2 scene = vec2(10.);
    scene = vec2(smoothMin(scene.x,bluesphere,0.4), scene.x<bluesphere ? scene.y:2.);
    scene = vec2(smoothMin(scene.x,ico,0.4), scene.x<ico ? scene.y:3.);
    scene = vec2(smoothMin(scene.x,redsphere,0.4), scene.x<redsphere ? scene.y:4.);
    scene = vec2(smoothMin(scene.x,plane,.4),scene.x<plane ? scene.y:1.);
    scene = vec2(smoothMin(scene.x,columns,.84),scene.x<columns? scene.y:5.);
    scene = vec2(min(scene.x,gates),scene.x<gates ? scene.y:5.);
    return scene;
}

// Function 73
float terrainMapH( const in vec3 pos ) {
    float y = terrainHigh(pos.xz);
    float h = pos.y - y;
    return h;
}

// Function 74
vec2 uvSmooth(vec2 uv,vec2 res)
{
    // no interpolation
    //return uv;
    // sinus interpolation
    //return uv+.8*sin(uv*res*PI2)/(res*PI2);
    // iq's polynomial interpolation
    vec2 f = fract(uv*res);
    return (uv*res+.5-f+3.*f*f-2.0*f*f*f)/res;
}

// Function 75
float cloudMap( vec3 n, mat4 camera, float bias, LameTweaks lame_tweaks )
{
	vec3 n0 = n;
#ifdef EARTH_ROTATION
	n.xy = rotate_with_angle( n.xy, lame_tweaks.earth_rot_time * earth_angular_velocity );
#endif
	float theta = acos( n.z );
	float phi = calc_angle( n.xy ) + PI; // assume range 0,1

	return cloudSphereMap( vec2( phi * 0.5, theta ) * ( 1.0 / PI ), camera, n0, bias, lame_tweaks );
}

// Function 76
vec3 mapShadow( in vec3 pos )
{
    float h = terrain( pos.xz );
    float d = pos.y - h;
    vec3 res = vec3( d, MAT_GROUND, 0.0 );
    
    res = mapGrass(pos,h,res);
    res = mapMoss(pos,h,res);

    vec3 m1 =  pos - mushroomPos1;
    vec3 m2 = (pos - mushroomPos2).zyx;
    if( length2(m2.xz) < length2(m1.xz) ) m1 = m2;
	res = mapMushroom(m1, res);


    vec3 q = worldToLadyBug(pos);
    vec3 d3 = mapLadyBug(q, res.x*4.0); d3.x/=4.0;
    if( d3.x<res.x ) res = d3;

    return res;
}

// Function 77
vec3 mapC(vec3 p)
{
    p += 0.5;
    vec3 b = abs(p);
    
    vec3 c = vec3(1.0);
    
    if(b.y < 7.0 && p.z > -7.0) 
        if(p.x < -7.0) 
            c = vec3(1.0, 0.2, 0.01);//orange wall
        else if(p.x > 7.0) 
            c = vec3(0.01, 0.3, 1.0);// blue wall
        

    return c;
}

// Function 78
float map(float v, float low1, float high1, float low2, float high2) {
	return (v-low1)/(high1-low1)*(high2-low2);
}

// Function 79
vec3 heatmapGradient(float t)
{
	return clamp((pow(t, 1.5) * 0.8 + 0.2) * vec3(smoothstep(0.0, 0.35, t) + t*0.5, smoothstep(0.5, 1.0, t), max(1.0 - t*1.7, t*7.0 - 6.0)), 0.0, 1.0);
}

// Function 80
float waterMap(vec2 p)
{
    p *= 7.7;
    return waterDsp(p, iTime)*3.;
}

// Function 81
vec3 map_sh9( in vec3 p )
{
    vec3 p00 = p - vec3( 0.00, 2.5,0.0);
	vec3 p01 = p - vec3(-1.25, 1.0,0.0);
	vec3 p02 = p - vec3( 0.00, 1.0,0.0);
	vec3 p03 = p - vec3( 1.25, 1.0,0.0);
	vec3 p04 = p - vec3(-2.50,-0.5,0.0);
	vec3 p05 = p - vec3(-1.25,-0.5,0.0);
	vec3 p06 = p - vec3( 0.00,-0.5,0.0);
	vec3 p07 = p - vec3( 1.25,-0.5,0.0);
	vec3 p08 = p - vec3( 2.50,-0.5,0.0);
	float r, d; vec3 n, s, res;
	
	#define SHAPE (vec3(d-abs(r), sign(r),d))
    
	d=length(p00); n=p00/d; r = SH_0_0( n ); s = SHAPE; res = s;
	d=length(p01); n=p01/d; r = SH_1_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p02); n=p02/d; r = SH_1_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p03); n=p03/d; r = SH_1_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p04); n=p04/d; r = SH_2_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p05); n=p05/d; r = SH_2_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p06); n=p06/d; r = SH_2_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p07); n=p07/d; r = SH_2_3( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p08); n=p08/d; r = SH_2_4( n ); s = SHAPE; if( s.x<res.x ) res=s;
	
	return vec3( res.x, 0.5+0.5*res.y, res.z );
}

// Function 82
vec3 envMap(vec3 p){
    
    p *= 4.;
    p.y += iTime;
    
    float n3D2 = n3D(p*2.);
   
    // A bit of fBm.
    float c = n3D(p)*.57 + n3D2*.28 + n3D(p*4.)*.15;
    c = smoothstep(.45, 1., c); // Putting in some dark space.
    
    p = vec3(c, c*c, c*c*c); // Redish tinge.
    
    return mix(p, p.xzy, n3D2*.4); // Mixing in a bit of purple.

}

// Function 83
vec1 suv(vec2 a){return a.x+a.y;}

// Function 84
float getMap(vec2 p){ float res=0.;
                     #ifdef USE_MAP
                     res=max(res,LINE(vec2(0.233333,0.991667), vec2(0.291667,0.991667))); res=max(res,LINE(vec2(0.341667,0.991667), vec2(0.408333,0.991667))); 
                     res=max(res,LINE(vec2(0.208333,0.983333), vec2(0.283333,0.983333))); res=max(res,LINE(vec2(0.3,0.983333), vec2(0.425,0.983333))); 
                     res=max(res,LINE(vec2(0.208333,0.975), vec2(0.416667,0.975))); res=max(res,LINE(vec2(0.5,0.975), vec2(0.5,0.975))); res=max(res,LINE(vec2(0.516667,0.975), vec2(0.533333,0.975))); res=max(res,LINE(vec2(0.6,0.975), vec2(0.6,0.975))); res=max(res,LINE(vec2(0.625,0.975), vec2(0.625,0.975))); res=max(res,LINE(vec2(0.716667,0.975), vec2(0.733333,0.975))); 
                     res=max(res,LINE(vec2(0.158333,0.966667), vec2(0.166667,0.966667))); res=max(res,LINE(vec2(0.183333,0.966667), vec2(0.2,0.966667))); res=max(res,LINE(vec2(0.216667,0.966667), vec2(0.258333,0.966667))); res=max(res,LINE(vec2(0.275,0.966667), vec2(0.408333,0.966667))); res=max(res,LINE(vec2(0.5,0.966667), vec2(0.525,0.966667))); res=max(res,LINE(vec2(0.741667,0.966667), vec2(0.75,0.966667))); 
                     res=max(res,LINE(vec2(0.141667,0.958333), vec2(0.158333,0.958333))); res=max(res,LINE(vec2(0.208333,0.958333), vec2(0.208333,0.958333))); res=max(res,LINE(vec2(0.233333,0.958333), vec2(0.25,0.958333))); res=max(res,LINE(vec2(0.275,0.958333), vec2(0.416667,0.958333))); res=max(res,LINE(vec2(0.508333,0.958333), vec2(0.525,0.958333))); res=max(res,LINE(vec2(0.75,0.958333), vec2(0.75,0.958333))); 
                     res=max(res,LINE(vec2(0.133333,0.95), vec2(0.191667,0.95))); res=max(res,LINE(vec2(0.216667,0.95), vec2(0.216667,0.95))); res=max(res,LINE(vec2(0.241667,0.95), vec2(0.25,0.95))); res=max(res,LINE(vec2(0.283333,0.95), vec2(0.408333,0.95))); res=max(res,LINE(vec2(0.633333,0.95), vec2(0.65,0.95))); res=max(res,LINE(vec2(0.716667,0.95), vec2(0.775,0.95))); res=max(res,LINE(vec2(0.85,0.95), vec2(0.858333,0.95))); 
                     res=max(res,LINE(vec2(0.125,0.941667), vec2(0.141667,0.941667))); res=max(res,LINE(vec2(0.158333,0.941667), vec2(0.158333,0.941667))); res=max(res,LINE(vec2(0.208333,0.941667), vec2(0.208333,0.941667))); res=max(res,LINE(vec2(0.225,0.941667), vec2(0.241667,0.941667))); res=max(res,LINE(vec2(0.316667,0.941667), vec2(0.408333,0.941667))); res=max(res,LINE(vec2(0.625,0.941667), vec2(0.625,0.941667))); res=max(res,LINE(vec2(0.708333,0.941667), vec2(0.775,0.941667))); 
                     res=max(res,LINE(vec2(0.125,0.933333), vec2(0.15,0.933333))); res=max(res,LINE(vec2(0.175,0.933333), vec2(0.175,0.933333))); res=max(res,LINE(vec2(0.191667,0.933333), vec2(0.258333,0.933333))); res=max(res,LINE(vec2(0.316667,0.933333), vec2(0.4,0.933333))); res=max(res,LINE(vec2(0.616667,0.933333), vec2(0.616667,0.933333))); res=max(res,LINE(vec2(0.658333,0.933333), vec2(0.658333,0.933333))); res=max(res,LINE(vec2(0.691667,0.933333), vec2(0.791667,0.933333))); res=max(res,LINE(vec2(0.808333,0.933333), vec2(0.816667,0.933333))); 
                     res=max(res,LINE(vec2(0.125,0.925), vec2(0.175,0.925))); res=max(res,LINE(vec2(0.191667,0.925), vec2(0.208333,0.925))); res=max(res,LINE(vec2(0.225,0.925), vec2(0.266667,0.925))); res=max(res,LINE(vec2(0.316667,0.925), vec2(0.4,0.925))); res=max(res,LINE(vec2(0.608333,0.925), vec2(0.616667,0.925))); res=max(res,LINE(vec2(0.658333,0.925), vec2(0.658333,0.925))); res=max(res,LINE(vec2(0.683333,0.925), vec2(0.816667,0.925))); res=max(res,LINE(vec2(0.841667,0.925), vec2(0.875,0.925))); 
                     res=max(res,LINE(vec2(0.025,0.916667), vec2(0.075,0.916667))); res=max(res,LINE(vec2(0.108333,0.916667), vec2(0.125,0.916667))); res=max(res,LINE(vec2(0.15,0.916667), vec2(0.183333,0.916667))); res=max(res,LINE(vec2(0.208333,0.916667), vec2(0.208333,0.916667))); res=max(res,LINE(vec2(0.225,0.916667), vec2(0.275,0.916667))); res=max(res,LINE(vec2(0.316667,0.916667), vec2(0.4,0.916667))); res=max(res,LINE(vec2(0.533333,0.916667), vec2(0.55,0.916667))); res=max(res,LINE(vec2(0.658333,0.916667), vec2(0.658333,0.916667))); res=max(res,LINE(vec2(0.675,0.916667), vec2(0.9,0.916667))); 
                     res=max(res,LINE(vec2(0.0166667,0.908333), vec2(0.275,0.908333))); res=max(res,LINE(vec2(0.325,0.908333), vec2(0.391667,0.908333))); res=max(res,LINE(vec2(0.508333,0.908333), vec2(0.566667,0.908333))); res=max(res,LINE(vec2(0.6,0.908333), vec2(0.6,0.908333))); res=max(res,LINE(vec2(0.625,0.908333), vec2(0.958333,0.908333))); 
                     res=max(res,LINE(vec2(0.025,0.9), vec2(0.241667,0.9))); res=max(res,LINE(vec2(0.258333,0.9), vec2(0.258333,0.9))); res=max(res,LINE(vec2(0.275,0.9), vec2(0.291667,0.9))); res=max(res,LINE(vec2(0.325,0.9), vec2(0.375,0.9))); res=max(res,LINE(vec2(0.508333,0.9), vec2(0.575,0.9))); res=max(res,LINE(vec2(0.6,0.9), vec2(0.658333,0.9))); res=max(res,LINE(vec2(0.675,0.9), vec2(0.975,0.9))); 
                     res=max(res,LINE(vec2(0.0166667,0.891667), vec2(0.233333,0.891667))); res=max(res,LINE(vec2(0.266667,0.891667), vec2(0.275,0.891667))); res=max(res,LINE(vec2(0.291667,0.891667), vec2(0.291667,0.891667))); res=max(res,LINE(vec2(0.325,0.891667), vec2(0.366667,0.891667))); res=max(res,LINE(vec2(0.408333,0.891667), vec2(0.425,0.891667))); res=max(res,LINE(vec2(0.508333,0.891667), vec2(0.558333,0.891667))); res=max(res,LINE(vec2(0.583333,0.891667), vec2(0.983333,0.891667))); 
                     res=max(res,LINE(vec2(0.0166667,0.883333), vec2(0.241667,0.883333))); res=max(res,LINE(vec2(0.258333,0.883333), vec2(0.283333,0.883333))); res=max(res,LINE(vec2(0.325,0.883333), vec2(0.35,0.883333))); res=max(res,LINE(vec2(0.408333,0.883333), vec2(0.425,0.883333))); res=max(res,LINE(vec2(0.5,0.883333), vec2(0.958333,0.883333))); res=max(res,LINE(vec2(0.975,0.883333), vec2(0.975,0.883333))); 
                     res=max(res,LINE(vec2(0,0.875), vec2(0,0.875))); res=max(res,LINE(vec2(0.0166667,0.875), vec2(0.216667,0.875))); res=max(res,LINE(vec2(0.233333,0.875), vec2(0.241667,0.875))); res=max(res,LINE(vec2(0.275,0.875), vec2(0.283333,0.875))); res=max(res,LINE(vec2(0.333333,0.875), vec2(0.35,0.875))); res=max(res,LINE(vec2(0.491667,0.875), vec2(0.516667,0.875))); res=max(res,LINE(vec2(0.533333,0.875), vec2(0.958333,0.875))); 
                     res=max(res,LINE(vec2(0.0166667,0.866667), vec2(0.208333,0.866667))); res=max(res,LINE(vec2(0.258333,0.866667), vec2(0.266667,0.866667))); res=max(res,LINE(vec2(0.333333,0.866667), vec2(0.35,0.866667))); res=max(res,LINE(vec2(0.483333,0.866667), vec2(0.508333,0.866667))); res=max(res,LINE(vec2(0.533333,0.866667), vec2(0.941667,0.866667))); 
                     res=max(res,LINE(vec2(0.00833333,0.858333), vec2(0.208333,0.858333))); res=max(res,LINE(vec2(0.258333,0.858333), vec2(0.275,0.858333))); res=max(res,LINE(vec2(0.341667,0.858333), vec2(0.341667,0.858333))); res=max(res,LINE(vec2(0.483333,0.858333), vec2(0.516667,0.858333))); res=max(res,LINE(vec2(0.533333,0.858333), vec2(0.891667,0.858333))); res=max(res,LINE(vec2(0.916667,0.858333), vec2(0.933333,0.858333))); 
                     res=max(res,LINE(vec2(0.025,0.85), vec2(0.0416667,0.85))); res=max(res,LINE(vec2(0.0916667,0.85), vec2(0.208333,0.85))); res=max(res,LINE(vec2(0.258333,0.85), vec2(0.275,0.85))); res=max(res,LINE(vec2(0.291667,0.85), vec2(0.291667,0.85))); res=max(res,LINE(vec2(0.458333,0.85), vec2(0.458333,0.85))); res=max(res,LINE(vec2(0.483333,0.85), vec2(0.516667,0.85))); res=max(res,LINE(vec2(0.533333,0.85), vec2(0.85,0.85))); res=max(res,LINE(vec2(0.866667,0.85), vec2(0.891667,0.85))); res=max(res,LINE(vec2(0.908333,0.85), vec2(0.916667,0.85))); 
                     res=max(res,LINE(vec2(0.0333333,0.841667), vec2(0.0333333,0.841667))); res=max(res,LINE(vec2(0.05,0.841667), vec2(0.05,0.841667))); res=max(res,LINE(vec2(0.1,0.841667), vec2(0.208333,0.841667))); res=max(res,LINE(vec2(0.258333,0.841667), vec2(0.291667,0.841667))); res=max(res,LINE(vec2(0.45,0.841667), vec2(0.458333,0.841667))); res=max(res,LINE(vec2(0.5,0.841667), vec2(0.508333,0.841667))); res=max(res,LINE(vec2(0.533333,0.841667), vec2(0.85,0.841667))); res=max(res,LINE(vec2(0.9,0.841667), vec2(0.908333,0.841667))); 
                     res=max(res,LINE(vec2(0.0333333,0.833333), vec2(0.0333333,0.833333))); res=max(res,LINE(vec2(0.1,0.833333), vec2(0.225,0.833333))); res=max(res,LINE(vec2(0.25,0.833333), vec2(0.291667,0.833333))); res=max(res,LINE(vec2(0.45,0.833333), vec2(0.458333,0.833333))); res=max(res,LINE(vec2(0.491667,0.833333), vec2(0.508333,0.833333))); res=max(res,LINE(vec2(0.525,0.833333), vec2(0.841667,0.833333))); res=max(res,LINE(vec2(0.9,0.833333), vec2(0.908333,0.833333))); 
                     res=max(res,LINE(vec2(0.0166667,0.825), vec2(0.0166667,0.825))); res=max(res,LINE(vec2(0.108333,0.825), vec2(0.308333,0.825))); res=max(res,LINE(vec2(0.441667,0.825), vec2(0.458333,0.825))); res=max(res,LINE(vec2(0.491667,0.825), vec2(0.491667,0.825))); res=max(res,LINE(vec2(0.516667,0.825), vec2(0.841667,0.825))); res=max(res,LINE(vec2(0.9,0.825), vec2(0.908333,0.825))); 
                     res=max(res,LINE(vec2(0.00833333,0.816667), vec2(0.00833333,0.816667))); res=max(res,LINE(vec2(0.116667,0.816667), vec2(0.308333,0.816667))); res=max(res,LINE(vec2(0.441667,0.816667), vec2(0.466667,0.816667))); res=max(res,LINE(vec2(0.483333,0.816667), vec2(0.858333,0.816667))); res=max(res,LINE(vec2(0.9,0.816667), vec2(0.9,0.816667))); 
                     res=max(res,LINE(vec2(0.116667,0.808333), vec2(0.308333,0.808333))); res=max(res,LINE(vec2(0.441667,0.808333), vec2(0.441667,0.808333))); res=max(res,LINE(vec2(0.458333,0.808333), vec2(0.466667,0.808333))); res=max(res,LINE(vec2(0.483333,0.808333), vec2(0.858333,0.808333))); res=max(res,LINE(vec2(0.9,0.808333), vec2(0.9,0.808333))); res=max(res,LINE(vec2(0.966667,0.808333), vec2(0.975,0.808333))); 
                     res=max(res,LINE(vec2(0.116667,0.8), vec2(0.3,0.8))); res=max(res,LINE(vec2(0.475,0.8), vec2(0.858333,0.8))); 
                     res=max(res,LINE(vec2(0.125,0.791667), vec2(0.283333,0.791667))); res=max(res,LINE(vec2(0.308333,0.791667), vec2(0.316667,0.791667))); res=max(res,LINE(vec2(0.466667,0.791667), vec2(0.858333,0.791667))); 
                     res=max(res,LINE(vec2(0.125,0.783333), vec2(0.216667,0.783333))); res=max(res,LINE(vec2(0.233333,0.783333), vec2(0.283333,0.783333))); res=max(res,LINE(vec2(0.308333,0.783333), vec2(0.316667,0.783333))); res=max(res,LINE(vec2(0.458333,0.783333), vec2(0.841667,0.783333))); 
                     res=max(res,LINE(vec2(0.133333,0.775), vec2(0.3,0.775))); res=max(res,LINE(vec2(0.466667,0.775), vec2(0.558333,0.775))); res=max(res,LINE(vec2(0.575,0.775), vec2(0.6,0.775))); res=max(res,LINE(vec2(0.616667,0.775), vec2(0.625,0.775))); res=max(res,LINE(vec2(0.641667,0.775), vec2(0.841667,0.775))); 
                     res=max(res,LINE(vec2(0.133333,0.766667), vec2(0.233333,0.766667))); res=max(res,LINE(vec2(0.25,0.766667), vec2(0.275,0.766667))); res=max(res,LINE(vec2(0.291667,0.766667), vec2(0.291667,0.766667))); res=max(res,LINE(vec2(0.466667,0.766667), vec2(0.541667,0.766667))); res=max(res,LINE(vec2(0.558333,0.766667), vec2(0.558333,0.766667))); res=max(res,LINE(vec2(0.575,0.766667), vec2(0.591667,0.766667))); res=max(res,LINE(vec2(0.608333,0.766667), vec2(0.625,0.766667))); res=max(res,LINE(vec2(0.641667,0.766667), vec2(0.841667,0.766667))); 
                     res=max(res,LINE(vec2(0.133333,0.758333), vec2(0.233333,0.758333))); res=max(res,LINE(vec2(0.25,0.758333), vec2(0.275,0.758333))); res=max(res,LINE(vec2(0.45,0.758333), vec2(0.483333,0.758333))); res=max(res,LINE(vec2(0.5,0.758333), vec2(0.5,0.758333))); res=max(res,LINE(vec2(0.516667,0.758333), vec2(0.541667,0.758333))); res=max(res,LINE(vec2(0.583333,0.758333), vec2(0.591667,0.758333))); res=max(res,LINE(vec2(0.608333,0.758333), vec2(0.833333,0.758333))); res=max(res,LINE(vec2(0.858333,0.758333), vec2(0.866667,0.758333))); 
                     res=max(res,LINE(vec2(0.133333,0.75), vec2(0.266667,0.75))); res=max(res,LINE(vec2(0.45,0.75), vec2(0.475,0.75))); res=max(res,LINE(vec2(0.491667,0.75), vec2(0.508333,0.75))); res=max(res,LINE(vec2(0.525,0.75), vec2(0.541667,0.75))); res=max(res,LINE(vec2(0.558333,0.75), vec2(0.558333,0.75))); res=max(res,LINE(vec2(0.583333,0.75), vec2(0.6,0.75))); res=max(res,LINE(vec2(0.616667,0.75), vec2(0.825,0.75))); res=max(res,LINE(vec2(0.85,0.75), vec2(0.858333,0.75))); 
                     res=max(res,LINE(vec2(0.133333,0.741667), vec2(0.266667,0.741667))); res=max(res,LINE(vec2(0.45,0.741667), vec2(0.466667,0.741667))); res=max(res,LINE(vec2(0.491667,0.741667), vec2(0.491667,0.741667))); res=max(res,LINE(vec2(0.508333,0.741667), vec2(0.6,0.741667))); res=max(res,LINE(vec2(0.616667,0.741667), vec2(0.791667,0.741667))); res=max(res,LINE(vec2(0.808333,0.741667), vec2(0.816667,0.741667))); 
                     res=max(res,LINE(vec2(0.133333,0.733333), vec2(0.258333,0.733333))); res=max(res,LINE(vec2(0.441667,0.733333), vec2(0.475,0.733333))); res=max(res,LINE(vec2(0.491667,0.733333), vec2(0.491667,0.733333))); res=max(res,LINE(vec2(0.525,0.733333), vec2(0.6,0.733333))); res=max(res,LINE(vec2(0.616667,0.733333), vec2(0.8,0.733333))); res=max(res,LINE(vec2(0.816667,0.733333), vec2(0.816667,0.733333))); res=max(res,LINE(vec2(0.85,0.733333), vec2(0.85,0.733333))); 
                     res=max(res,LINE(vec2(0.133333,0.725), vec2(0.258333,0.725))); res=max(res,LINE(vec2(0.45,0.725), vec2(0.458333,0.725))); res=max(res,LINE(vec2(0.508333,0.725), vec2(0.508333,0.725))); res=max(res,LINE(vec2(0.525,0.725), vec2(0.525,0.725))); res=max(res,LINE(vec2(0.541667,0.725), vec2(0.6,0.725))); res=max(res,LINE(vec2(0.616667,0.725), vec2(0.791667,0.725))); res=max(res,LINE(vec2(0.816667,0.725), vec2(0.816667,0.725))); res=max(res,LINE(vec2(0.85,0.725), vec2(0.85,0.725))); 
                     res=max(res,LINE(vec2(0.133333,0.716667), vec2(0.258333,0.716667))); res=max(res,LINE(vec2(0.45,0.716667), vec2(0.45,0.716667))); res=max(res,LINE(vec2(0.475,0.716667), vec2(0.491667,0.716667))); res=max(res,LINE(vec2(0.533333,0.716667), vec2(0.533333,0.716667))); res=max(res,LINE(vec2(0.55,0.716667), vec2(0.791667,0.716667))); res=max(res,LINE(vec2(0.816667,0.716667), vec2(0.816667,0.716667))); res=max(res,LINE(vec2(0.841667,0.716667), vec2(0.85,0.716667))); 
                     res=max(res,LINE(vec2(0.141667,0.708333), vec2(0.25,0.708333))); res=max(res,LINE(vec2(0.45,0.708333), vec2(0.491667,0.708333))); res=max(res,LINE(vec2(0.533333,0.708333), vec2(0.533333,0.708333))); res=max(res,LINE(vec2(0.558333,0.708333), vec2(0.791667,0.708333))); res=max(res,LINE(vec2(0.816667,0.708333), vec2(0.816667,0.708333))); res=max(res,LINE(vec2(0.833333,0.708333), vec2(0.85,0.708333))); 
                     res=max(res,LINE(vec2(0.15,0.7), vec2(0.25,0.7))); res=max(res,LINE(vec2(0.45,0.7), vec2(0.491667,0.7))); res=max(res,LINE(vec2(0.566667,0.7), vec2(0.791667,0.7))); res=max(res,LINE(vec2(0.825,0.7), vec2(0.833333,0.7))); 
                     res=max(res,LINE(vec2(0.15,0.691667), vec2(0.241667,0.691667))); res=max(res,LINE(vec2(0.441667,0.691667), vec2(0.508333,0.691667))); res=max(res,LINE(vec2(0.525,0.691667), vec2(0.525,0.691667))); res=max(res,LINE(vec2(0.566667,0.691667), vec2(0.8,0.691667))); res=max(res,LINE(vec2(0.825,0.691667), vec2(0.825,0.691667))); 
                     res=max(res,LINE(vec2(0.15,0.683333), vec2(0.241667,0.683333))); res=max(res,LINE(vec2(0.441667,0.683333), vec2(0.8,0.683333))); 
                     res=max(res,LINE(vec2(0.158333,0.675), vec2(0.216667,0.675))); res=max(res,LINE(vec2(0.241667,0.675), vec2(0.241667,0.675))); res=max(res,LINE(vec2(0.441667,0.675), vec2(0.591667,0.675))); res=max(res,LINE(vec2(0.608333,0.675), vec2(0.8,0.675))); 
                     res=max(res,LINE(vec2(0.166667,0.666667), vec2(0.2,0.666667))); res=max(res,LINE(vec2(0.241667,0.666667), vec2(0.241667,0.666667))); res=max(res,LINE(vec2(0.425,0.666667), vec2(0.8,0.666667))); 
                     res=max(res,LINE(vec2(0.158333,0.658333), vec2(0.158333,0.658333))); res=max(res,LINE(vec2(0.175,0.658333), vec2(0.2,0.658333))); res=max(res,LINE(vec2(0.25,0.658333), vec2(0.25,0.658333))); res=max(res,LINE(vec2(0.433333,0.658333), vec2(0.6,0.658333))); res=max(res,LINE(vec2(0.616667,0.658333), vec2(0.791667,0.658333))); 
                     res=max(res,LINE(vec2(0.175,0.65), vec2(0.2,0.65))); res=max(res,LINE(vec2(0.258333,0.65), vec2(0.258333,0.65))); res=max(res,LINE(vec2(0.433333,0.65), vec2(0.558333,0.65))); res=max(res,LINE(vec2(0.575,0.65), vec2(0.608333,0.65))); res=max(res,LINE(vec2(0.633333,0.65), vec2(0.8,0.65))); 
                     res=max(res,LINE(vec2(0.166667,0.641667), vec2(0.191667,0.641667))); res=max(res,LINE(vec2(0.425,0.641667), vec2(0.558333,0.641667))); res=max(res,LINE(vec2(0.575,0.641667), vec2(0.625,0.641667))); res=max(res,LINE(vec2(0.658333,0.641667), vec2(0.783333,0.641667))); res=max(res,LINE(vec2(0.8,0.641667), vec2(0.8,0.641667))); 
                     res=max(res,LINE(vec2(0.183333,0.633333), vec2(0.191667,0.633333))); res=max(res,LINE(vec2(0.25,0.633333), vec2(0.25,0.633333))); res=max(res,LINE(vec2(0.266667,0.633333), vec2(0.266667,0.633333))); res=max(res,LINE(vec2(0.425,0.633333), vec2(0.625,0.633333))); res=max(res,LINE(vec2(0.658333,0.633333), vec2(0.775,0.633333))); 
                     res=max(res,LINE(vec2(0.183333,0.625), vec2(0.2,0.625))); res=max(res,LINE(vec2(0.225,0.625), vec2(0.225,0.625))); res=max(res,LINE(vec2(0.258333,0.625), vec2(0.266667,0.625))); res=max(res,LINE(vec2(0.425,0.625), vec2(0.625,0.625))); res=max(res,LINE(vec2(0.666667,0.625), vec2(0.7,0.625))); res=max(res,LINE(vec2(0.725,0.625), vec2(0.758333,0.625))); 
                     res=max(res,LINE(vec2(0.0416667,0.616667), vec2(0.0416667,0.616667))); res=max(res,LINE(vec2(0.183333,0.616667), vec2(0.2,0.616667))); res=max(res,LINE(vec2(0.225,0.616667), vec2(0.225,0.616667))); res=max(res,LINE(vec2(0.266667,0.616667), vec2(0.275,0.616667))); res=max(res,LINE(vec2(0.425,0.616667), vec2(0.566667,0.616667))); res=max(res,LINE(vec2(0.583333,0.616667), vec2(0.625,0.616667))); res=max(res,LINE(vec2(0.666667,0.616667), vec2(0.7,0.616667))); res=max(res,LINE(vec2(0.725,0.616667), vec2(0.75,0.616667))); res=max(res,LINE(vec2(0.766667,0.616667), vec2(0.766667,0.616667))); 
                     res=max(res,LINE(vec2(0.191667,0.608333), vec2(0.225,0.608333))); res=max(res,LINE(vec2(0.258333,0.608333), vec2(0.266667,0.608333))); res=max(res,LINE(vec2(0.283333,0.608333), vec2(0.283333,0.608333))); res=max(res,LINE(vec2(0.425,0.608333), vec2(0.566667,0.608333))); res=max(res,LINE(vec2(0.583333,0.608333), vec2(0.616667,0.608333))); res=max(res,LINE(vec2(0.675,0.608333), vec2(0.691667,0.608333))); res=max(res,LINE(vec2(0.733333,0.608333), vec2(0.766667,0.608333))); res=max(res,LINE(vec2(0.8,0.608333), vec2(0.8,0.608333))); 
                     res=max(res,LINE(vec2(0.2,0.6), vec2(0.225,0.6))); res=max(res,LINE(vec2(0.4,0.6), vec2(0.4,0.6))); res=max(res,LINE(vec2(0.425,0.6), vec2(0.566667,0.6))); res=max(res,LINE(vec2(0.591667,0.6), vec2(0.608333,0.6))); res=max(res,LINE(vec2(0.675,0.6), vec2(0.691667,0.6))); res=max(res,LINE(vec2(0.733333,0.6), vec2(0.758333,0.6))); res=max(res,LINE(vec2(0.8,0.6), vec2(0.8,0.6))); 
                     res=max(res,LINE(vec2(0.216667,0.591667), vec2(0.233333,0.591667))); res=max(res,LINE(vec2(0.425,0.591667), vec2(0.575,0.591667))); res=max(res,LINE(vec2(0.591667,0.591667), vec2(0.608333,0.591667))); res=max(res,LINE(vec2(0.675,0.591667), vec2(0.683333,0.591667))); res=max(res,LINE(vec2(0.741667,0.591667), vec2(0.758333,0.591667))); res=max(res,LINE(vec2(0.8,0.591667), vec2(0.8,0.591667))); res=max(res,LINE(vec2(0.866667,0.591667), vec2(0.866667,0.591667))); 
                     res=max(res,LINE(vec2(0.216667,0.583333), vec2(0.233333,0.583333))); res=max(res,LINE(vec2(0.3,0.583333), vec2(0.3,0.583333))); res=max(res,LINE(vec2(0.425,0.583333), vec2(0.575,0.583333))); res=max(res,LINE(vec2(0.591667,0.583333), vec2(0.6,0.583333))); res=max(res,LINE(vec2(0.675,0.583333), vec2(0.683333,0.583333))); res=max(res,LINE(vec2(0.741667,0.583333), vec2(0.766667,0.583333))); res=max(res,LINE(vec2(0.8,0.583333), vec2(0.8,0.583333))); 
                     res=max(res,LINE(vec2(0.233333,0.575), vec2(0.233333,0.575))); res=max(res,LINE(vec2(0.425,0.575), vec2(0.583333,0.575))); res=max(res,LINE(vec2(0.675,0.575), vec2(0.683333,0.575))); res=max(res,LINE(vec2(0.75,0.575), vec2(0.766667,0.575))); res=max(res,LINE(vec2(0.8,0.575), vec2(0.808333,0.575))); 
                     res=max(res,LINE(vec2(0.233333,0.566667), vec2(0.233333,0.566667))); res=max(res,LINE(vec2(0.266667,0.566667), vec2(0.275,0.566667))); res=max(res,LINE(vec2(0.291667,0.566667), vec2(0.3,0.566667))); res=max(res,LINE(vec2(0.425,0.566667), vec2(0.6,0.566667))); res=max(res,LINE(vec2(0.675,0.566667), vec2(0.683333,0.566667))); res=max(res,LINE(vec2(0.75,0.566667), vec2(0.758333,0.566667))); res=max(res,LINE(vec2(0.808333,0.566667), vec2(0.808333,0.566667))); 
                     res=max(res,LINE(vec2(0.233333,0.558333), vec2(0.233333,0.558333))); res=max(res,LINE(vec2(0.266667,0.558333), vec2(0.3,0.558333))); res=max(res,LINE(vec2(0.433333,0.558333), vec2(0.6,0.558333))); res=max(res,LINE(vec2(0.683333,0.558333), vec2(0.683333,0.558333))); res=max(res,LINE(vec2(0.741667,0.558333), vec2(0.741667,0.558333))); res=max(res,LINE(vec2(0.758333,0.558333), vec2(0.758333,0.558333))); res=max(res,LINE(vec2(0.791667,0.558333), vec2(0.791667,0.558333))); 
                     res=max(res,LINE(vec2(0.241667,0.55), vec2(0.3,0.55))); res=max(res,LINE(vec2(0.433333,0.55), vec2(0.6,0.55))); res=max(res,LINE(vec2(0.741667,0.55), vec2(0.741667,0.55))); res=max(res,LINE(vec2(0.808333,0.55), vec2(0.808333,0.55))); 
                     res=max(res,LINE(vec2(0.258333,0.541667), vec2(0.308333,0.541667))); res=max(res,LINE(vec2(0.441667,0.541667), vec2(0.6,0.541667))); res=max(res,LINE(vec2(0.691667,0.541667), vec2(0.691667,0.541667))); res=max(res,LINE(vec2(0.741667,0.541667), vec2(0.741667,0.541667))); res=max(res,LINE(vec2(0.808333,0.541667), vec2(0.808333,0.541667))); 
                     res=max(res,LINE(vec2(0.258333,0.533333), vec2(0.316667,0.533333))); res=max(res,LINE(vec2(0.441667,0.533333), vec2(0.458333,0.533333))); res=max(res,LINE(vec2(0.483333,0.533333), vec2(0.6,0.533333))); res=max(res,LINE(vec2(0.733333,0.533333), vec2(0.75,0.533333))); res=max(res,LINE(vec2(0.783333,0.533333), vec2(0.791667,0.533333))); 
                     res=max(res,LINE(vec2(0.258333,0.525), vec2(0.325,0.525))); res=max(res,LINE(vec2(0.491667,0.525), vec2(0.591667,0.525))); res=max(res,LINE(vec2(0.733333,0.525), vec2(0.733333,0.525))); res=max(res,LINE(vec2(0.75,0.525), vec2(0.75,0.525))); res=max(res,LINE(vec2(0.783333,0.525), vec2(0.791667,0.525))); 
                     res=max(res,LINE(vec2(0.258333,0.516667), vec2(0.325,0.516667))); res=max(res,LINE(vec2(0.5,0.516667), vec2(0.591667,0.516667))); res=max(res,LINE(vec2(0.741667,0.516667), vec2(0.75,0.516667))); res=max(res,LINE(vec2(0.775,0.516667), vec2(0.783333,0.516667))); 
                     res=max(res,LINE(vec2(0.25,0.508333), vec2(0.325,0.508333))); res=max(res,LINE(vec2(0.5,0.508333), vec2(0.583333,0.508333))); res=max(res,LINE(vec2(0.733333,0.508333), vec2(0.75,0.508333))); res=max(res,LINE(vec2(0.766667,0.508333), vec2(0.816667,0.508333))); 
                     res=max(res,LINE(vec2(0.216667,0.5), vec2(0.216667,0.5))); res=max(res,LINE(vec2(0.25,0.5), vec2(0.333333,0.5))); res=max(res,LINE(vec2(0.5,0.5), vec2(0.55,0.5))); res=max(res,LINE(vec2(0.566667,0.5), vec2(0.583333,0.5))); res=max(res,LINE(vec2(0.741667,0.5), vec2(0.75,0.5))); res=max(res,LINE(vec2(0.766667,0.5), vec2(0.783333,0.5))); res=max(res,LINE(vec2(0.816667,0.5), vec2(0.825,0.5))); 
                     res=max(res,LINE(vec2(0.25,0.491667), vec2(0.341667,0.491667))); res=max(res,LINE(vec2(0.5,0.491667), vec2(0.575,0.491667))); res=max(res,LINE(vec2(0.75,0.491667), vec2(0.75,0.491667))); res=max(res,LINE(vec2(0.775,0.491667), vec2(0.783333,0.491667))); res=max(res,LINE(vec2(0.8,0.491667), vec2(0.816667,0.491667))); res=max(res,LINE(vec2(0.833333,0.491667), vec2(0.841667,0.491667))); 
                     res=max(res,LINE(vec2(0.25,0.483333), vec2(0.358333,0.483333))); res=max(res,LINE(vec2(0.5,0.483333), vec2(0.575,0.483333))); res=max(res,LINE(vec2(0.75,0.483333), vec2(0.758333,0.483333))); res=max(res,LINE(vec2(0.775,0.483333), vec2(0.783333,0.483333))); res=max(res,LINE(vec2(0.8,0.483333), vec2(0.8,0.483333))); res=max(res,LINE(vec2(0.833333,0.483333), vec2(0.85,0.483333))); res=max(res,LINE(vec2(0.883333,0.483333), vec2(0.883333,0.483333))); 
                     res=max(res,LINE(vec2(0.25,0.475), vec2(0.358333,0.475))); res=max(res,LINE(vec2(0.5,0.475), vec2(0.575,0.475))); res=max(res,LINE(vec2(0.75,0.475), vec2(0.75,0.475))); res=max(res,LINE(vec2(0.8,0.475), vec2(0.8,0.475))); res=max(res,LINE(vec2(0.833333,0.475), vec2(0.858333,0.475))); res=max(res,LINE(vec2(0.883333,0.475), vec2(0.883333,0.475))); 
                     res=max(res,LINE(vec2(0.25,0.466667), vec2(0.366667,0.466667))); res=max(res,LINE(vec2(0.5,0.466667), vec2(0.575,0.466667))); res=max(res,LINE(vec2(0.758333,0.466667), vec2(0.758333,0.466667))); res=max(res,LINE(vec2(0.85,0.466667), vec2(0.875,0.466667))); res=max(res,LINE(vec2(0.891667,0.466667), vec2(0.891667,0.466667))); 
                     res=max(res,LINE(vec2(0.25,0.458333), vec2(0.366667,0.458333))); res=max(res,LINE(vec2(0.508333,0.458333), vec2(0.541667,0.458333))); res=max(res,LINE(vec2(0.558333,0.458333), vec2(0.575,0.458333))); res=max(res,LINE(vec2(0.758333,0.458333), vec2(0.775,0.458333))); res=max(res,LINE(vec2(0.85,0.458333), vec2(0.866667,0.458333))); 
                     res=max(res,LINE(vec2(0.25,0.45), vec2(0.366667,0.45))); res=max(res,LINE(vec2(0.508333,0.45), vec2(0.566667,0.45))); res=max(res,LINE(vec2(0.783333,0.45), vec2(0.8,0.45))); res=max(res,LINE(vec2(0.858333,0.45), vec2(0.858333,0.45))); res=max(res,LINE(vec2(0.908333,0.45), vec2(0.908333,0.45))); 
                     res=max(res,LINE(vec2(0.258333,0.441667), vec2(0.366667,0.441667))); res=max(res,LINE(vec2(0.508333,0.441667), vec2(0.575,0.441667))); res=max(res,LINE(vec2(0.808333,0.441667), vec2(0.808333,0.441667))); res=max(res,LINE(vec2(0.875,0.441667), vec2(0.875,0.441667))); 
                     res=max(res,LINE(vec2(0.258333,0.433333), vec2(0.358333,0.433333))); res=max(res,LINE(vec2(0.508333,0.433333), vec2(0.575,0.433333))); res=max(res,LINE(vec2(0.825,0.433333), vec2(0.825,0.433333))); res=max(res,LINE(vec2(0.841667,0.433333), vec2(0.841667,0.433333))); res=max(res,LINE(vec2(0.858333,0.433333), vec2(0.858333,0.433333))); 
                     res=max(res,LINE(vec2(0.258333,0.425), vec2(0.358333,0.425))); res=max(res,LINE(vec2(0.508333,0.425), vec2(0.575,0.425))); res=max(res,LINE(vec2(0.825,0.425), vec2(0.841667,0.425))); res=max(res,LINE(vec2(0.858333,0.425), vec2(0.858333,0.425))); 
                     res=max(res,LINE(vec2(0.0666667,0.416667), vec2(0.0666667,0.416667))); res=max(res,LINE(vec2(0.258333,0.416667), vec2(0.358333,0.416667))); res=max(res,LINE(vec2(0.5,0.416667), vec2(0.575,0.416667))); res=max(res,LINE(vec2(0.6,0.416667), vec2(0.6,0.416667))); res=max(res,LINE(vec2(0.816667,0.416667), vec2(0.841667,0.416667))); res=max(res,LINE(vec2(0.858333,0.416667), vec2(0.858333,0.416667))); res=max(res,LINE(vec2(0.983333,0.416667), vec2(0.983333,0.416667))); 
                     res=max(res,LINE(vec2(0.266667,0.408333), vec2(0.358333,0.408333))); res=max(res,LINE(vec2(0.5,0.408333), vec2(0.575,0.408333))); res=max(res,LINE(vec2(0.591667,0.408333), vec2(0.6,0.408333))); res=max(res,LINE(vec2(0.808333,0.408333), vec2(0.841667,0.408333))); res=max(res,LINE(vec2(0.858333,0.408333), vec2(0.858333,0.408333))); res=max(res,LINE(vec2(0.925,0.408333), vec2(0.925,0.408333))); 
                     res=max(res,LINE(vec2(0.275,0.4), vec2(0.358333,0.4))); res=max(res,LINE(vec2(0.5,0.4), vec2(0.566667,0.4))); res=max(res,LINE(vec2(0.591667,0.4), vec2(0.6,0.4))); res=max(res,LINE(vec2(0.808333,0.4), vec2(0.841667,0.4))); res=max(res,LINE(vec2(0.858333,0.4), vec2(0.866667,0.4))); 
                     res=max(res,LINE(vec2(0.275,0.391667), vec2(0.35,0.391667))); res=max(res,LINE(vec2(0.5,0.391667), vec2(0.566667,0.391667))); res=max(res,LINE(vec2(0.591667,0.391667), vec2(0.6,0.391667))); res=max(res,LINE(vec2(0.8,0.391667), vec2(0.866667,0.391667))); 
                     res=max(res,LINE(vec2(0.275,0.383333), vec2(0.35,0.383333))); res=max(res,LINE(vec2(0.508333,0.383333), vec2(0.558333,0.383333))); res=max(res,LINE(vec2(0.591667,0.383333), vec2(0.6,0.383333))); res=max(res,LINE(vec2(0.625,0.383333), vec2(0.625,0.383333))); res=max(res,LINE(vec2(0.8,0.383333), vec2(0.866667,0.383333))); res=max(res,LINE(vec2(0.916667,0.383333), vec2(0.916667,0.383333))); 
                     res=max(res,LINE(vec2(0.275,0.375), vec2(0.35,0.375))); res=max(res,LINE(vec2(0.508333,0.375), vec2(0.558333,0.375))); res=max(res,LINE(vec2(0.591667,0.375), vec2(0.6,0.375))); res=max(res,LINE(vec2(0.783333,0.375), vec2(0.875,0.375))); res=max(res,LINE(vec2(0.925,0.375), vec2(0.925,0.375))); 
                     res=max(res,LINE(vec2(0.275,0.366667), vec2(0.35,0.366667))); res=max(res,LINE(vec2(0.508333,0.366667), vec2(0.558333,0.366667))); res=max(res,LINE(vec2(0.591667,0.366667), vec2(0.591667,0.366667))); res=max(res,LINE(vec2(0.783333,0.366667), vec2(0.875,0.366667))); 
                     res=max(res,LINE(vec2(0.275,0.358333), vec2(0.333333,0.358333))); res=max(res,LINE(vec2(0.508333,0.358333), vec2(0.558333,0.358333))); res=max(res,LINE(vec2(0.591667,0.358333), vec2(0.591667,0.358333))); res=max(res,LINE(vec2(0.783333,0.358333), vec2(0.883333,0.358333))); 
                     res=max(res,LINE(vec2(0.275,0.35), vec2(0.333333,0.35))); res=max(res,LINE(vec2(0.508333,0.35), vec2(0.55,0.35))); res=max(res,LINE(vec2(0.591667,0.35), vec2(0.591667,0.35))); res=max(res,LINE(vec2(0.783333,0.35), vec2(0.883333,0.35))); 
                     res=max(res,LINE(vec2(0.275,0.341667), vec2(0.333333,0.341667))); res=max(res,LINE(vec2(0.508333,0.341667), vec2(0.55,0.341667))); res=max(res,LINE(vec2(0.783333,0.341667), vec2(0.883333,0.341667))); 
                     res=max(res,LINE(vec2(0.275,0.333333), vec2(0.333333,0.333333))); res=max(res,LINE(vec2(0.516667,0.333333), vec2(0.55,0.333333))); res=max(res,LINE(vec2(0.783333,0.333333), vec2(0.883333,0.333333))); 
                     res=max(res,LINE(vec2(0.275,0.325), vec2(0.325,0.325))); res=max(res,LINE(vec2(0.516667,0.325), vec2(0.55,0.325))); res=max(res,LINE(vec2(0.783333,0.325), vec2(0.883333,0.325))); 
                     res=max(res,LINE(vec2(0.275,0.316667), vec2(0.325,0.316667))); res=max(res,LINE(vec2(0.516667,0.316667), vec2(0.55,0.316667))); res=max(res,LINE(vec2(0.783333,0.316667), vec2(0.883333,0.316667))); 
                     res=max(res,LINE(vec2(0.275,0.308333), vec2(0.316667,0.308333))); res=max(res,LINE(vec2(0.516667,0.308333), vec2(0.541667,0.308333))); res=max(res,LINE(vec2(0.783333,0.308333), vec2(0.808333,0.308333))); res=max(res,LINE(vec2(0.841667,0.308333), vec2(0.883333,0.308333))); 
                     res=max(res,LINE(vec2(0.275,0.3), vec2(0.316667,0.3))); res=max(res,LINE(vec2(0.516667,0.3), vec2(0.533333,0.3))); res=max(res,LINE(vec2(0.783333,0.3), vec2(0.791667,0.3))); res=max(res,LINE(vec2(0.85,0.3), vec2(0.875,0.3))); 
                     res=max(res,LINE(vec2(0.275,0.291667), vec2(0.308333,0.291667))); res=max(res,LINE(vec2(0.841667,0.291667), vec2(0.875,0.291667))); res=max(res,LINE(vec2(0.941667,0.291667), vec2(0.941667,0.291667))); 
                     res=max(res,LINE(vec2(0.266667,0.283333), vec2(0.308333,0.283333))); res=max(res,LINE(vec2(0.85,0.283333), vec2(0.875,0.283333))); 
                     res=max(res,LINE(vec2(0.266667,0.275), vec2(0.308333,0.275))); res=max(res,LINE(vec2(0.858333,0.275), vec2(0.866667,0.275))); res=max(res,LINE(vec2(0.95,0.275), vec2(0.95,0.275))); 
                     res=max(res,LINE(vec2(0.266667,0.266667), vec2(0.291667,0.266667))); res=max(res,LINE(vec2(0.95,0.266667), vec2(0.95,0.266667))); 
                     res=max(res,LINE(vec2(0.266667,0.258333), vec2(0.291667,0.258333))); res=max(res,LINE(vec2(0.866667,0.258333), vec2(0.866667,0.258333))); res=max(res,LINE(vec2(0.941667,0.258333), vec2(0.95,0.258333))); 
                     res=max(res,LINE(vec2(0.266667,0.25), vec2(0.291667,0.25))); res=max(res,LINE(vec2(0.866667,0.25), vec2(0.866667,0.25))); res=max(res,LINE(vec2(0.941667,0.25), vec2(0.941667,0.25))); 
                     res=max(res,LINE(vec2(0.266667,0.241667), vec2(0.283333,0.241667))); res=max(res,LINE(vec2(0.933333,0.241667), vec2(0.933333,0.241667))); 
                     res=max(res,LINE(vec2(0.266667,0.233333), vec2(0.283333,0.233333))); res=max(res,LINE(vec2(0.925,0.233333), vec2(0.933333,0.233333))); 
                     res=max(res,LINE(vec2(0.266667,0.225), vec2(0.283333,0.225))); 
                     res=max(res,LINE(vec2(0.266667,0.216667), vec2(0.283333,0.216667))); 
                     res=max(res,LINE(vec2(0.266667,0.208333), vec2(0.275,0.208333))); res=max(res,LINE(vec2(0.658333,0.208333), vec2(0.658333,0.208333))); 
                     res=max(res,LINE(vec2(0.266667,0.2), vec2(0.275,0.2))); 
                     res=max(res,LINE(vec2(0.266667,0.191667), vec2(0.275,0.191667))); res=max(res,LINE(vec2(0.3,0.191667), vec2(0.3,0.191667))); 
                     res=max(res,LINE(vec2(0.266667,0.183333), vec2(0.275,0.183333))); 
                     res=max(res,LINE(vec2(0.316667,0.141667), vec2(0.316667,0.141667))); res=max(res,LINE(vec2(0.341667,0.141667), vec2(0.341667,0.141667))); 
                     res=max(res,LINE(vec2(0.3,0.133333), vec2(0.3,0.133333))); 
                     res=max(res,LINE(vec2(0.308333,0.125), vec2(0.308333,0.125))); 
                     res=max(res,LINE(vec2(0.291667,0.116667), vec2(0.3,0.116667))); res=max(res,LINE(vec2(0.75,0.116667), vec2(0.75,0.116667))); 
                     res=max(res,LINE(vec2(0.283333,0.108333), vec2(0.3,0.108333))); res=max(res,LINE(vec2(0.6,0.108333), vec2(0.616667,0.108333))); res=max(res,LINE(vec2(0.708333,0.108333), vec2(0.783333,0.108333))); res=max(res,LINE(vec2(0.8,0.108333), vec2(0.808333,0.108333))); res=max(res,LINE(vec2(0.825,0.108333), vec2(0.85,0.108333))); 
                     res=max(res,LINE(vec2(0.283333,0.1), vec2(0.283333,0.1))); res=max(res,LINE(vec2(0.591667,0.1), vec2(0.65,0.1))); res=max(res,LINE(vec2(0.691667,0.1), vec2(0.866667,0.1))); 
                     res=max(res,LINE(vec2(0.275,0.0916667), vec2(0.291667,0.0916667))); res=max(res,LINE(vec2(0.558333,0.0916667), vec2(0.566667,0.0916667))); res=max(res,LINE(vec2(0.583333,0.0916667), vec2(0.65,0.0916667))); res=max(res,LINE(vec2(0.675,0.0916667), vec2(0.9,0.0916667))); 
                     res=max(res,LINE(vec2(0.275,0.0833333), vec2(0.3,0.0833333))); res=max(res,LINE(vec2(0.45,0.0833333), vec2(0.458333,0.0833333))); res=max(res,LINE(vec2(0.475,0.0833333), vec2(0.65,0.0833333))); res=max(res,LINE(vec2(0.666667,0.0833333), vec2(0.925,0.0833333))); 
                     res=max(res,LINE(vec2(0.191667,0.075), vec2(0.2,0.075))); res=max(res,LINE(vec2(0.266667,0.075), vec2(0.3,0.075))); res=max(res,LINE(vec2(0.441667,0.075), vec2(0.933333,0.075))); 
                     res=max(res,LINE(vec2(0.125,0.0666667), vec2(0.141667,0.0666667))); res=max(res,LINE(vec2(0.191667,0.0666667), vec2(0.3,0.0666667))); res=max(res,LINE(vec2(0.425,0.0666667), vec2(0.925,0.0666667))); 
                     res=max(res,LINE(vec2(0.0916667,0.0583333), vec2(0.158333,0.0583333))); res=max(res,LINE(vec2(0.183333,0.0583333), vec2(0.291667,0.0583333))); res=max(res,LINE(vec2(0.425,0.0583333), vec2(0.916667,0.0583333))); 
                     res=max(res,LINE(vec2(0.0583333,0.05), vec2(0.283333,0.05))); res=max(res,LINE(vec2(0.391667,0.05), vec2(0.908333,0.05))); 
                     res=max(res,LINE(vec2(0.0333333,0.0416667), vec2(0.266667,0.0416667))); res=max(res,LINE(vec2(0.341667,0.0416667), vec2(0.341667,0.0416667))); res=max(res,LINE(vec2(0.375,0.0416667), vec2(0.925,0.0416667))); 
                     res=max(res,LINE(vec2(0.025,0.0333333), vec2(0.025,0.0333333))); res=max(res,LINE(vec2(0.05,0.0333333), vec2(0.25,0.0333333))); res=max(res,LINE(vec2(0.275,0.0333333), vec2(0.275,0.0333333))); res=max(res,LINE(vec2(0.333333,0.0333333), vec2(0.341667,0.0333333))); res=max(res,LINE(vec2(0.375,0.0333333), vec2(0.908333,0.0333333))); 
                     res=max(res,LINE(vec2(0.0666667,0.025), vec2(0.258333,0.025))); res=max(res,LINE(vec2(0.283333,0.025), vec2(0.3,0.025))); res=max(res,LINE(vec2(0.316667,0.025), vec2(0.341667,0.025))); res=max(res,LINE(vec2(0.383333,0.025), vec2(0.908333,0.025))); 
                     res=max(res,LINE(vec2(0.05,0.0166667), vec2(0.291667,0.0166667))); res=max(res,LINE(vec2(0.325,0.0166667), vec2(0.908333,0.0166667))); 
                     res=max(res,LINE(vec2(0.05,0.00833333), vec2(0.933333,0.00833333))); 
                     res=max(res,LINE(vec2(0.0583333,0), vec2(0.983333,0))); 
                     #endif
                     return res;}

// Function 85
vec3 luvToLch(vec3 tuple) {  float L = tuple.x;  float U = tuple.y;  float V = tuple.z;  float C = length(tuple.yz);  float H = degrees(atan(V,U));  if (H < 0.0) {   H = 360.0 + H;  }   return vec3(L, C, H); }

// Function 86
vec3 mapNormal(vec3 pt, float e) {
    vec3 normal;
    normal.y = mapDetailed(pt).x;    
    normal.x = mapDetailed(vec3(pt.x+e,pt.y,pt.z)).x - normal.y;
    normal.z = mapDetailed(vec3(pt.x,pt.y,pt.z+e)).x - normal.y;
    normal.y = e;
    return normalize(normal);
}

// Function 87
vec2 localUV2uv(vec2 localUV, vec2 startPoint){
  vec2 gainedLocalUV = vec2(localUV.x * TILE_LENGTH, 
                            localUV.y * TILE_LENGTH);
  vec2 uv = vec2((gainedLocalUV.x + startPoint.x) , 
                 (gainedLocalUV.y + startPoint.y) );
  return errorCheckUV(localUV, uv);
}

// Function 88
vec3 linearToneMapping(vec3 color)
{
	float exposure = 1.;
	color = clamp(exposure * color, 0., 1.);
	color = pow(color, vec3(1. / gamma));
	return color;
}

// Function 89
vec2 mapD1(float t)
{
    return -7.0*a*c*cos(t+m)*sin(7.0*t+n) - a*sin(t+m)*(b+c*cos(7.0*t+n));
}

// Function 90
float MapTopWing(vec3 p, float mirrored)
{    
  checkPos = p- vec3(1.15, 1.04, -8.5);
  pR(checkPos.xy, -0.15);  
  float topWing = sdBox( checkPos, vec3(0.014, 0.8, 1.2));
  if (topWing<.15) //Bounding Box test
  {
    float flapDist = MapTailFlap(checkPos, mirrored);

    checkPos = p- vec3(1.15, 1.04, -8.5);
    pR(checkPos.xy, -0.15);  
    // top border    
    topWing = min(topWing, sdBox( checkPos-vec3(0, 0.55, 0), vec3(0.04, 0.1, 1.25)));

    float flapCutout = sdBox(checkPos- vec3(0., -0.04, -1.19), vec3(0.02, .45, 1.0));
    // tailFlap front cutout
    checkPos = p- vec3(1.15, 2., -7.65);
    pR(checkPos.yz, 1.32);
    flapCutout=max(flapCutout, -sdBox( checkPos, vec3(.75, 1.41, 1.6)));

    // make hole for tail flap
    topWing=max(topWing, -flapCutout);

    // front cutouts
    checkPos = p- vec3(1.15, 2., -7.);
    pR(checkPos.yz, 1.02);
    topWing=fOpIntersectionRound(topWing, -sdBox( checkPos, vec3(.75, 1.41, 1.6)), 0.05);

    // rear cutout
    checkPos = p- vec3(1.15, 1., -11.25);  
    pR(checkPos.yz, -0.15);
    topWing=fOpIntersectionRound(topWing, -sdBox( checkPos, vec3(.75, 1.4, 2.0)), 0.05);

    // top roll 
    topWing=min(topWing, sdCapsule(p- vec3(1.26, 1.8, -8.84), vec3(0, 0, -.50), vec3(0, 0, 0.3), 0.06)); 

    topWing = min(topWing, flapDist);
  }
  return topWing;
}

// Function 91
int getMap( vec3 pos ) {	
	vec3 posf = floor( (pos - vec3(32.))  );
    
	float n = posf.x + posf.y*517.0 + 1313.0*posf.z;
    float h = hash(n);
	
	if( h > sqrt( sqrt( dot( posf.yz, posf.yz )*0.16 ) ) - 0.8  ) {
        return 0;
	}	
	
	return int( hash( n * 465.233 ) * 16. );
}

// Function 92
vec3 map(vec3 p){
    float aTime = iTime/2.f;
    
    //BRIDGE
    vec3 q = repeat(p, 12.f , 0.f, 0.f);
    float a0 = arch_sdf(q, vec3(0.f), 7.f, 5.f, 2.f);
    float a1 = arch_sdf(q, vec3(0.f), 8.f, 6.f, 1.5f);
    float d0 = sub(a0, a1, 0.2f);
    
    
    // FLOATING BLOBS
    vec3 qs1 = repeat(p, 40.f, 0.f, 40.f);
    vec3 qs2 = repeat(p, 10.f, 0.f, 10.f);
    float f = 5.f;
    float hf = 15.5f * cos(p.z/40.f);
    vec3 h0 = vec3(f * cos(iTime/4.f), 15.f + (sin(p.x) + cos(p.z)) * sin(iTime) - hf * (sin(p.x/20.f) + cos(p.z/20.f)) * cos(iTime), f * sin(iTime/2.f));
    vec3 h1 = vec3(5.f + -f * cos(iTime/4.f), 20.f + (sin(p.x) + cos(p.z)) * sin(iTime) + hf * (cos(p.x/20.f) + sin(p.z/20.f)) * sin(iTime), 5.f + -f * sin(iTime/2.f));
    float s0 = sphere_sdf(qs1, h0, 3.5f);
    float s1 = sphere_sdf(qs1, h1, 3.5f);
    float d1 = smin(s0 * 0.5f, s1 * 0.5f, 2.f).x;
    
    
    // WATER
    float plane_noise = 2.f;
    #ifdef SINEWAVES
    plane_noise += 1.f *(cos(p.x/10.f - cos(iTime)) + sin(p.z/15.f + sin(iTime))) * ((sin(iTime/5.f) + 1.1f)*0.5);
    plane_noise -= 1.;
    #endif
    
    #if NOISE == 0
    // nop
    #elif NOISE == 1
    // plane_noise += (perlinNoise3D(p) * (sin(iTime/5.f) + 1.1f) * 0.5f);
    // plane_noise += (perlinNoise3D(p) *  0.2f);
    // plane_noise += (perlinNoise3D(p) * (sin(iTime/5.f) + 1.1f) * 0.1f);
    // plane_noise += perlinNoise3D(vec3(p.x, 0., p.z)) * (sin(iTime/5.f) + 1.1f) * 0.1f; // 2D perlin noise is about 3x as fast to march through, but doesn't look as good.
    // plane_noise += perlinNoise3D(vec3(p.x, iTime, p.z)) * (sin(iTime/5.f) + 1.1f) * 0.1f; // Using the time as the third component makes it look better, while still being 2D noise within a single frame so we keep the performance benefits. At least, we should but we don't for some reason.
    // Shifting the noise over time is both fast and looks alright, however.
    vec3 qWater = p + vec3(-iTime*3., 0., sin(iTime/(1.62*3.))*5.);
    plane_noise += perlinNoise3D(vec3(qWater.x, 0., qWater.z)) * (sin(iTime/5.f) + 1.1f) * 0.1f;
    #elif NOISE == 2
    plane_noise += ((worley(p.xz/10.f, abs(sin(iTime) * cos(iTime/2.f)))) * (sin(iTime/5.f) + 1.1f) * 0.3f);
    #elif NOISE == 3
    // plane_noise += iqNoiseLayered(p.xz) * (sin(iTime/5.f) + 1.1f) * 0.5f;
    plane_noise += iqNoiseLayered(p.xz) * 0.2f;
    #endif
    
    float d2 = infinite_plane_sdf(p, plane_noise);
    
    
    // COMPOSITION
    // x: distance, y: material, z: water mask. (Water mask is used for mixing in the reflection term.)
    vec3 res;
    
    #ifdef COMPOSITION_0
    // In this block, water and blobs merge and the bridge is independent.
    res.xz = smin(d1, d2, 3.f);
    res.y = res.z < 1. ? float(BLOB) : float(WATER);

    vec2 bridge_res;
    bridge_res = hardMin(res.x, d0);
    res.x = bridge_res.x;
    res.y = bool(bridge_res.y) ? float(LAMBERT_RED) : res.y;
    res.z = bool(bridge_res.y) ? 0. : res.z;
    #endif
    
    #ifdef COMPOSITION_1
    // In this block, blobs and the bridge merge with the water but not eachother.
    vec3 bridge_water;
    bridge_water.xz = smin(d0, d2, 1.);
    bridge_water.y = float(LAMBERT_RED);
    
    vec3 blob_water;
    blob_water.xz = smin(d1, d2, 3.);
    blob_water.y = float(BLOB);
    
    res = bridge_water.x < blob_water.x ? bridge_water : blob_water;
    res.y = res.z < 1. ? res.y : float(WATER); // For when water reflects into more water.
    #endif
    
    #ifdef COMPOSITION_2
    // Same as composition 1 but the blobs deform around the bridge.
    vec3 bridge_water;
    bridge_water.xz = smin(d0, d2, 1.);
    bridge_water.y = float(LAMBERT_RED);
    
    float expanded_bridge = d0 - 1.;
    float blob = d1;
    //blob = smin(blob, expanded_bridge, 10.).x; // Make the far side of the blob smush away from the bridge; disabled because of artefacts where the bridge is close to the water.
    blob = sub(blob, expanded_bridge, 2.); // Avoid the bridge instead of going through it.
    
    vec3 blob_water;
    blob_water.xz = smin(blob, d2, 3.);
    blob_water.y = float(BLOB);
    
    res = bridge_water.x < blob_water.x ? bridge_water : blob_water;
    res.y = res.z < 1. ? res.y : float(WATER); // For when water reflects into more water.
    #endif
    
    // TODO: add ability to smoothly blend all three materials, then make a scene that does that.
    
    return res;
}

// Function 93
vec2 map(in vec3 p, out vec4 mate){
    
    float tubeTime = floor(time)+smoothstep(0.3,0.7,fract(time));
    p.x += 0.2*sin(2.0*p.z+4.0*time);
    p.z += tubeTime;
    

    // tube
    vec3 pt = p;
    pt.z -= tubeTime;
	const float xOffset = 0.4;
    
    // tube material
    vec3 ptm = pt;
    vec3 pcyl = carToCyl(ptm-vec3(xOffset,0,0));
    vec4 mate0 = vec4(1,1,1,0.2);
    vec4 mate1 = vec4(texture(iChannel0,1.0*vec2(pcyl.y/PI,pcyl.z)).rrr,0.6);
    const float radius = 0.3;
    float modRadius = 0.06*mate1.r+radius+0.1*pow(abs(sin(-3.*time+0.25*pt.z)),25.0);
    mate0 = mix(mate0,mate1,0.7);

    vec2 d0 = vec2(sdCylinder(pt, vec3(xOffset,0.0,modRadius)),1.0);
    
    // leaves
    vec3 pl = pt;
	pl.xy -= vec2(xOffset,0.0);
    pl.z = fract(0.5*pl.z)/0.5;
    float leafIndex = pModPolar(pl.xy,3.);
    pl = rotationXY(vec2(0.0,-2.7))*pl;
    vec2 d1 = vec2(sdVerticalCapsule(pl-vec3(-1.2,0.0,0.4),-1.8,0.03-0.03*pl.z),3.0);
    
    // leaf material
    mate1 = vec4(0.6,0.8,1,0.2);
    d0 = opUC_s(d0, mate0,d1,mate1,0.3,mate0);
    
    // worm body
    mate1 = vec4(1,1,1,0.2);
    vec3 pb = p;

    float wormTimeZ = floor(time)+smoothstep(0.3,0.7,fract(time))-floor(time-0.5)-smoothstep(0.3,0.7,fract(time-0.5));
    vec3 back = vec3(-0.4*wormTimeXB+0.7*wormTimeXF,0.,1.5+wormTimeZ);
    float l = length(back);
    pb.x += 0.7*wormTimeXF+radius-0.19+0.4*(0.5+0.5*sin(1.6*PI*p.z/l-0.8));
    pb = rotationXY(vec2(0.0,-atan(back.x/back.z)))*pb;
    pb.z *= 1.0/length(back);
    
    // worm's freckles
    pcyl = carToCyl(pb-vec3(0.,0.,0.));
    vec2 ps = vec2(10.0*0.5*pcyl.y/PI,20.0*pcyl.z);
   	vec2 fps = fract(ps);
    vec2 ips = floor(ps);
    vec2 ran = hash22(ips);
//    vec2 ran = texelFetch( iChannel0, (ivec2(ips)+10), 0 ).rg;
    float r = 0.2+0.3*ran.x;
    float blob = blob(fps-0.2-0.4*ran.xy, r);
    float freckles = (1.0-blob)*smoothstep(0.0,0.1,pb.z)*smoothstep(1.0,0.9,pb.z);
    mate1 = mix(mate1, vec4(1,0.2,0,0.8), freckles);

    d1 = vec2(sdVerticalCapsule(pb, 1.0, 0.15-0.07*wormTimeZ+0.01*freckles),1.0);

    pb.z *= length(back);
    // head
    vec4 mate2 = vec4(1,0.2,0,0.4);
    vec4 mate3;
    vec2 d2 = vec2(sdSphere(pb-vec3(0,0,-0.2),0.15),2.0);
    d1 = opUC_s(d1, mate1 ,d2, mate2 ,0.1, mate3);
    
    // back end
    d2 = vec2(sdSphere(pb-vec3(0,0,length(back)+0.15),0.15),3.0);
    d1 = opUC_s(d1, mate3 ,d2, mate2 ,0.1, mate3);
    
    vec2 dFinal = opUC_s(d0, mate0, d1, mate3, 0.05, mate);
    return dFinal;
}

// Function 94
vec2 distortUV(vec2 uv) {
    float r = length (uv);
    uv = normalize(uv) * pow(r, mix(1.0,0.025, g_speed));
    
    r = length (uv);
    float rr = r*r;
    float k1 = mix(-0.2, 0.0, g_speed);
    float k2 = mix(-0.1, 0.0, g_speed);
    
    return uv * (1.0 + k1*rr + k2*rr*rr);
}

// Function 95
vec3 distMapNormal(vec3 p) {
    return normalize(vec3(
        distMap(vec3(p.x+NORMAL_EPILSON,p.y,p.z))-distMap(vec3(p.x-NORMAL_EPILSON,p.y,p.z)),
        
		distMap(vec3(p.x,p.y+NORMAL_EPILSON,p.z))-distMap(vec3(p.x,p.y-NORMAL_EPILSON,p.z)),
        
        distMap(vec3(p.x,p.y,p.z+NORMAL_EPILSON))-distMap(vec3(p.x,p.y,p.z-NORMAL_EPILSON))
        ));
}

// Function 96
vec2 equiRemap(vec2 lonLat, vec2 delta) {
    vec3 v = lonLatToXYZ(lonLat);
	v = yRot(v,delta.x);
    v = xRot(v,delta.y);
    return xyzToLonLat(v);
}

// Function 97
vec2 GetUVScroll(const vec2 vInputUV, float t)
{
	vec2 vFontUV = vInputUV;
	vFontUV *= 0.25;
	
    vFontUV.y -= 0.005;
	vFontUV.x += t * 3.0 - 1.5;
	
	return vFontUV;
}

// Function 98
float MapFlare( vec3 p, Missile missile)
{
  TranslateMissilePos(p, missile);
  return sdEllipsoid( p+ vec3(0., 0., 2.4), vec3(.05, 0.05, .15));
}

// Function 99
vec3 yuv_rgb (vec3 yuv) {
    return yuv_rgb(yuv, vec2(0.0722, 0.2126), vec2(0.436, 0.615));
}

// Function 100
float fisheyeMapping(in vec3 uvw, out vec2 texCoords)
{
  float phi = atan(length(uvw.xy),abs(uvw.z));
  float r =  phi / PI * 2.0 / (-0.667 + 1.0);
  if ((r > 1.0) || (r <= 0.0)) return -1.0;
  float theta = atan(uvw.x,uvw.y);
  texCoords.s = fract(0.5 * (1.0 + r* sin(theta)));
  texCoords.t = fract(0.5 * (1.0 + r * cos(theta)));
  return 1.0;
}

// Function 101
float mapRange(float value, float low1, float high1, float low2, float high2) {
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1);
}

// Function 102
vec4 map( in vec3 p, out mat3 s )
{
    
    mat3 dd;
	vec4 d1 = fbmd( p, dd );
    d1.x -= 0.33;
	d1.x *= 0.7;
    d1.yzw = 0.7*d1.yzw;
    dd *= 0.7;
    // clip to box
    vec4 d2 = sdBox( p, vec3(1.5) );
    if(d1.x>d2.x)
    {
        s = dd;
        return d1;
    }
    
    
    s = mat3(0.0);
    return d2;
}

// Function 103
vec3 xyzToLuv(vec3 tuple){
    float X = tuple.x;
    float Y = tuple.y;
    float Z = tuple.z;

    float L = hsluv_yToL(Y);
    
    float div = 1./dot(tuple,vec3(1,15,3)); 

    return vec3(
        1.,
        (52. * (X*div) - 2.57179),
        (117.* (Y*div) - 6.08816)
    ) * L;
}

// Function 104
vec3 luvToXyz(vec3 tuple) {  float L = tuple.x;  float U = tuple.y / (13.0 * L) + 0.19783000664283681;  float V = tuple.z / (13.0 * L) + 0.468319994938791;  float Y = hsluv_lToY(L);  float X = 2.25 * U * Y / V;  float Z = (3./V - 5.)*Y - (X/3.);  return vec3(X, Y, Z); }

// Function 105
float map( in vec3 q )
{
    float h = q.y;

    q *= 0.01*vec3(0.5,1.0,0.5);
    
	float f;
    f  = 0.500000*abs(noise( q )); q = q*2.02;
    f += 0.250000*abs(noise( q )); q = q*2.03;
    f += 0.125000*abs(noise( q )); q = q*2.01;
    f += 0.062500*abs(noise( q )); q = q*2.02;
    f += 0.031250*abs(noise( q )); q = q*2.03;
  //f += 0.015625*abs(noise( q ));
    f = -1.0 + 2.0*f;
    
    f = mix( f, -0.1, 1.0-smoothstep( h, 50.0, 60.0 ));
    f = mix( f, -0.1, 1.0-smoothstep( h, 250.0, 300.0 ));
    f += 0.17;

    return f;
}

// Function 106
vec2 map( in vec3 pos ){
	return map7(pos);
}

// Function 107
vec3 Tonemap_ACESFitted2(vec3 acescg)
{
    vec3 color = acescg * RRT_SAT;
    
   #if 1
    color = ToneTF2(color);
   #else
    color = RRTAndODTFit(color);
   #endif
    
    color = color * ACESOutputMat;

    return color;
}

// Function 108
vec2 noiseStackUV(vec3 pos,int octaves,float falloff,float diff){
	float displaceA = noiseStack(pos,octaves,falloff);
	float displaceB = noiseStack(pos+vec3(3984.293,423.21,5235.19),octaves,falloff);
	return vec2(displaceA,displaceB);
}

// Function 109
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
	// only use one face of cubemap (z+)
    if( rayDir.z<0.0 || abs(rayDir.x)>abs(rayDir.z) || abs(rayDir.y)>abs(rayDir.z)) discard;

    // Output to cubemap
    mainImage(fragColor,(rayDir.xy/rayDir.z*.5+.5)*Res0);
}

// Function 110
vec2 map(vec3 p) {
    float d = length(p) - 4.;
	return vec2(d, 1.);
}

// Function 111
float map(vec2 uv, mat2 q, vec2 l) {
    return sin(2.0 * pow(dot(q * uv + l, uv), 6.0 / 11.0));
}

// Function 112
vec2 map2(vec3 pos)
{
    pos = rotateVec2(pos);

    vec2 obj;
    
    float ypos = getCylYPos();
    float ypos2 = cl - 1.2;
    
    float c1a = sdCylinder(pos - vec3(-ce/2., ypos + ypos2, 0.), vec2(cr, cl));
    float c2a = sdCylinder(pos - vec3(ce/2., -ypos + ypos2, 0.), vec2(cr, cl));
    float c1b = sdCylinder(pos - vec3(-ce/2., ypos, 0.), vec2(cr, 10.));
    float c2b = sdCylinder(pos - vec3(ce/2., -ypos, 0.), vec2(cr, 10.));
    float c1b2 = sdCylinder(pos - vec3(-ce/2., ypos, 0.), vec2(cr - 0.005, 10.));
    float c2b2 = sdCylinder(pos - vec3(ce/2., -ypos, 0.), vec2(cr - 0.01, 10.));
    float c1b3 = sdCylinder(pos - vec3(-ce/2., ypos, 0.), vec2(cr + 0.005, 10.));
    float c2b3 = sdCylinder(pos - vec3(ce/2., -ypos, 0.), vec2(cr + 0.005, 10.));
    float c1c = opS(c1a, c2b);
    float c2c = opS(c2a, c1b);
    
    float cca = max(c1b2, c2b2);
    float mposy = max(ypos, -ypos);
    cca = max(cca, pos.y - mposy - cl - ypos2);
    cca = max(cca, -pos.y + mposy - cl - ypos2);
    
    float cbc = sdCylinder(pos.yxz + vec3(-mposy + 0.85, 0., 0.), vec2(0.12, 0.5));
    float bbc = sdBox(pos - vec3(0., mposy - 1.08, 0.), vec3(0.5, 0.22, 0.12)); 
    cca = opS(cca, cbc);
    cca = opS(cca, bbc);
    
    float cb1 = sdCylinder(pos.yxz + vec3(-ypos + 0.85, 0.17, 0.), vec2(0.115, 0.12));
    c1c = min(c1c, cb1);
    
    float cb2 = sdCylinder(pos.yxz + vec3(ypos + 0.85, -0.17, 0.), vec2(0.115, 0.12));
    c2c = min(c2c, cb2);
    
    vec2 c1 = vec2(c1c, C1_OBJ);
    vec2 c2 = vec2(c2c, C2_OBJ);
    vec2 cc = vec2(cca, CC_OBJ);
    
    float boxa = sdBox(pos - vec3(0., boxYPos, 0.), boxSize);
    boxa = opS(boxa, c1b3);
    boxa = opS(boxa, c2b3);
    float boxi = sdBox(pos - vec3(0., boxYPos - 0.65, 0.), boxSize*0.85);
    boxa = opS(boxa, boxi);
    float cbox = sdCylinder(pos.yxz + vec3(1.8, 0., 0.), vec2(0.12, boxSize.x*1.1));
    boxa = opS(boxa, cbox);
    
    vec2 box = vec2(boxa, BOX_OBJ);
    
    vec3 posr = pos;
    posr.y+= 1.8;
    posr.yz = rotateVec(posr.yz, cms*iTime - pi*0.5);
    float wc0 = sdCylinder(posr.yxz + vec3(0., 0.25, 0.), vec2(0.115, boxSize.x*1.2));
    float wc1 = sdCylinder(posr.yxz + vec3(cma, -ce*0.5 - 0.1, 0.), vec2(0.6, 0.2));
    float wc2 = sdCylinder(posr.yxz + vec3(-cma, ce*0.5 + 0.1, 0.), vec2(0.6, 0.2));
    float wc3 = sdCylinder(posr.yxz + vec3(0., boxSize.x*1.25, 0.), vec2(0.65, 0.1));
    float wc4 = sdCylinder(posr.yxz + vec3(0.55, boxSize.x*1.33, 0.), vec2(0.08, 0.2));
    float wheela = min(min(min(min(wc0, wc1), wc2), wc3), wc4);
    vec2 wheel = vec2(wheela, WHEEL_OBJ);
    
    obj = opU(c1, c2);
    obj = opU(obj, cc);
    obj = opU(obj, box);
 
    #ifndef always_cut
    if (cut_obj)
    #endif
       obj.x = max(obj.x, pos.z);

    obj = opU(obj, wheel);
    
    return obj;
}

// Function 113
float map(in vec3 pos) {
    pos -= vec3(0.19,0.1, 0.10);
    pos = vec3(0.99619469809  * pos.x + 0.08715574274 * pos.z, pos.y, 0.99619469809  * pos.z - 0.08715574274 * pos.x);   
    pos = vec3(0.98480775301 * pos.x + 0.17364817766 * pos.y, 0.98480775301 * pos.y - 0.17364817766 * pos.x, pos.z);
    vec3 saved = pos;
    pos = vec3(pos.x, pos.z, pos.y);
    float m = sdHexPrism(pos, vec2(1.4, 3.0));
    pos = vec3(0.70710678118 * pos.x + 0.70710678118  * pos.y, 0.70710678118 * pos.y - 0.70710678118 * pos.x, pos.z);
    pos = vec3(0.98480775301 * pos.x + 0.17364817766 * pos.z, pos.y, 0.98480775301 * pos.z - 0.17364817766 * pos.x);
    m = max(m-0.02, sdHexPrism(pos, vec2(1.55, 2.5))-0.02);
    m = max(m, sdBox(saved, vec3(1.3, 2.55, 1.4)-0.02));  
    return m-0.02;
}

// Function 114
float height_map( vec2 p )
{
#if USETEXTUREHEIGHT
  float f = 0.15+textureLod(iChannel2, p*0.6, 0.0).r*2.;
#else
  mat2 m = mat2( 0.9563*1.4,  -0.2924*1.4,  0.2924*1.4,  0.9563*1.4 );
  p = p*6.;
  float f = 0.6000*noise1( p ); p = m*p*1.1;
  f += 0.2500*noise1( p ); p = m*p*1.32;
  f += 0.1666*noise1( p ); p = m*p*1.11;
  f += 0.0834*noise( p ); p = m*p*1.12;
  f += 0.0634*noise( p ); p = m*p*1.13;
  f += 0.0444*noise( p ); p = m*p*1.14;
  f += 0.0274*noise( p ); p = m*p*1.15;
  f += 0.0134*noise( p ); p = m*p*1.16;
  f += 0.0104*noise( p ); p = m*p*1.17;
  f += 0.0084*noise( p );
  const float FLAT_LEVEL = 0.525;
  if (f<FLAT_LEVEL)
      f = f;
  else
      f = pow((f-FLAT_LEVEL)/(1.-FLAT_LEVEL), 2.)*(1.-FLAT_LEVEL)*2.0+FLAT_LEVEL; // makes a smooth coast-increase
#endif
  return clamp(f, 0., 10.);
}

// Function 115
float mapQ(vec3 p){
  float s = 0.5;
  for(float i = 1.; i < MAX_LEVEL; i++){
    s *= 2.;
    //if we don't pass the random check, add max to index so we break
    i += step( hash13(floor(p * s)), 0.5 ) * MAX_LEVEL;
  }
  return s;
}

// Function 116
vec3 map( in vec3 p )
{
    vec3 p00 = p - vec3( 0.00, 2.5,0.0);
	vec3 p01 = p - vec3(-1.25, 1.0,0.0);
	vec3 p02 = p - vec3( 0.00, 1.0,0.0);
	vec3 p03 = p - vec3( 1.25, 1.0,0.0);
	vec3 p04 = p - vec3(-2.50,-0.5,0.0);
	vec3 p05 = p - vec3(-1.25,-0.5,0.0);
	vec3 p06 = p - vec3( 0.00,-0.5,0.0);
	vec3 p07 = p - vec3( 1.25,-0.5,0.0);
	vec3 p08 = p - vec3( 2.50,-0.5,0.0);
	vec3 p09 = p - vec3(-3.75,-2.0,0.0);
	vec3 p10 = p - vec3(-2.50,-2.0,0.0);
	vec3 p11 = p - vec3(-1.25,-2.0,0.0);
	vec3 p12 = p - vec3( 0.00,-2.0,0.0);
	vec3 p13 = p - vec3( 1.25,-2.0,0.0);
	vec3 p14 = p - vec3( 2.50,-2.0,0.0);
	vec3 p15 = p - vec3( 3.75,-2.0,0.0);
	
	float r, d; vec3 n, s, res;
	
    #ifdef SHOW_SPHERES
	#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))
	#else
	#define SHAPE (vec3(d-abs(r), sign(r),d))
	#endif
	d=length(p00); n=p00/d; r = SH_0_0( n ); s = SHAPE; res = s;
	d=length(p01); n=p01/d; r = SH_1_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p02); n=p02/d; r = SH_1_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p03); n=p03/d; r = SH_1_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p04); n=p04/d; r = SH_2_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p05); n=p05/d; r = SH_2_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p06); n=p06/d; r = SH_2_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p07); n=p07/d; r = SH_2_3( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p08); n=p08/d; r = SH_2_4( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p09); n=p09/d; r = SH_3_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p10); n=p10/d; r = SH_3_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p11); n=p11/d; r = SH_3_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p12); n=p12/d; r = SH_3_3( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p13); n=p13/d; r = SH_3_4( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p14); n=p14/d; r = SH_3_5( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p15); n=p15/d; r = SH_3_6( n ); s = SHAPE; if( s.x<res.x ) res=s;
	
	return vec3( res.x, 0.5+0.5*res.y, res.z );
}

// Function 117
vec2 map( in vec3 pos )
{
    vec2 res = opU( vec2( sdPlane(     pos), 1.0 ),
	                vec2( sdSphere(    pos-vec3( 0.0,0.25, 0.0), 0.25 ), 646.9 ) );
    res = opU( res, vec2( udRoundBox(  pos-vec3( 1.0,0.3, 1.0), vec3(0.15), 0.1 ), 541.0 ) );
	res = opU( res, vec2( sdTorus(     pos-vec3( 0.0,0.25, 1.0), vec2(0.20,0.05) ), 425.0 ) );
    res = opU( res, vec2( sdCapsule(   pos,vec3(-1.3,0.20,-0.1), vec3(-1.0,0.20,0.2), 0.1  ), 331.9 ) );
	res = opU( res, vec2( sdTorus82(   pos-vec3( 0.0,0.25, 2.0), vec2(0.20,0.05) ),250.0 ) );
	res = opU( res, vec2( sdTorus88(   pos-vec3(-1.0,0.25, 2.0), vec2(0.20,0.05) ),143.0 ) );

    return res;
}

// Function 118
vec2 mapMat(vec3 p){
    vec3 q = p;
    p = vec3(p.x + sin(p.z), p.y + cos(p.z), p.z);
    q = vec3(q.x - sin(q.z), q.y - cos(q.z), q.z);
    
    vec2 helixa = vec2(length(p.xy) - 0.5 - (texture(iChannel0, p.xz) * 0.1).y, 2.0);
    vec2 helixb = vec2(length(q.xy) - 0.5 - (texture(iChannel0, q.xz) * 0.1).y, 3.0);

    return vecMin(helixa, helixb);
}

// Function 119
vec2 GetUV(const in vec2 A, const in vec2 B, const in vec2 C, const in vec2 P)
{
    vec2 vPB = B - P;
    float f1 = Cross(A-B, vPB);
    float f2 = Cross(B-C, vPB);
    float f3 = Cross(C-A, C-P);
    
    return vec2(f1, f2) / (f1 + f2 + f3);
}

// Function 120
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

// Function 121
float map(vec3 p)
{
    vec3 q = p;

    pMod3(q, vec3(.75, .6, .15));
    pMod3(q, vec3(0.9, 1., 0.6));
    
    pMod1(p.x, 1.);
    	q.y = abs(sin(iTime))-0.1;
    
    float s1 = sphere(p, .65); 
    float s2 = sphere(q, .5);
    float s3 = sphere(q, 1.);
    
    float disp = min(0.5 * (sin(p.x/3.) *
                       sin(p.y) *
                       (sin(p.z*10.)) ), 50.);
    	s1 -= disp;
    	s2 *= disp;
    	s3 *= disp;
    	//s1 -= disp;
    	
    
    
  	float df1 = min(s1, s2); // Union
    float df2 = max(s2, s1); // Intersection
    float df3 = max(max(s1, s3), min(s1, s2)); // Difference
    
    return df3;
}

// Function 122
vec3 luvToRgb(vec3 tuple){  return xyzToRgb(luvToXyz(tuple)); }

// Function 123
vec3 bumpMap2(in vec3 p, in vec3 n, float bumpfactor){
    
    const vec2 e = vec2(0.002, 0);
    float ref = bumpFunc2(p, n);                 
    vec3 grad = (vec3(bumpFunc2(p - e.xyy, n),
                      bumpFunc2(p - e.yxy, n),
                      bumpFunc2(p - e.yyx, n) )-ref)/e.x;                     
          
    grad -= n*dot(n, grad);          
                      
    return normalize( n + grad*bumpfactor );
	
}

// Function 124
float map(vec3 pos, int processedMaterial
){float o = 0. //0. if first operation is a SUB(), 1e10 otherwise
 ;maxGISize = 0.
 ;rayObj = 0
 ;vec4 w = vec4(pos,1)//worldSpace
 ;beginMaterial(0)
 ;SUB(box(w,vec3(0,0,0),vec3(10,2.5,10)))
 ;ADD(cylinderY(w,vec3(-6,0,-2.),0.1,3.))
 ;ADD(cylinderY(w,vec3(5,0,-2.),0.1,3.))
 ;endMaterial()
     
     
 ;vec4 c=beginObj(oCubeMy,w)//cubespace, not color
 ;//ADD(box(c,vec3(0,0,0),vec3(1)))
 ;//c.xz = abs(c.xz)
 ;for(int i=0;i<4;i++
 ){beginMaterial(i)
  ;ADD(sphere(c,vec3(0,0,0),1.))
  ;endMaterial()
  ;c=abs(c)
  ;c=beginObj(oCubeChil,c);}
 ;vec4 blackHoleSpace = beginObj(oBlackHole,w)
 ;beginMaterial(8)
 ;ADD(sphere(blackHoleSpace,vec3(0,0,0),.5))
 ;endMaterial()
 ;vec4 tunnelSpace = beginObj(oTunnel,w)
 ;beginMaterial(3)
 ;ADD(box(tunnelSpace,vec3(0,.5,0),vec3(.2,.1,1.5)))
 ;ADD(box(tunnelSpace,vec3(0,-.5,0),vec3(.2,.1,1.5)))
 ;endMaterial()
 ;vec4 tunnelDoorSpace = beginObj(oTunnelDoor,w)
 ;beginMaterial(4)
 ;ADD(box(tunnelDoorSpace,vec3(0,0,1.4),vec3(.2,.4,0.1)))
 ;ADD(box(tunnelDoorSpace,vec3(0,0,-1.4),vec3(.2,.4,0.1)))
 ;endMaterial()
 ;vec4 trainSpace = beginObj(oTrain,w)
 ;beginMaterial(7)
 ;ADD(box(trainSpace,vec3(0,0,-.8),vec3(.1,.1,.18)))
 ;ADD(box(trainSpace,vec3(0,0,-.4),vec3(.1,.1,.18)))
 ;ADD(box(trainSpace,vec3(0,0,0),vec3(.1,.1,.18)))
 ;ADD(box(trainSpace,vec3(0,0,.4),vec3(.1,.1,.18)))
 ;ADD(box(trainSpace,vec3(0,0,.68),vec3(.1,.1,.06)))
 ;ADD(cylinderZ(trainSpace,vec3(0,.04,.8),.07,.18))
 ;endObj()
 ;endMaterial()
 ;float temp = max(0.,(1.-20.*abs(blackHoleSpace.y)))
 ;float tmpGauss = length(blackHoleSpace.xz)-1.5
 ;o=min(o,max(0.1,max(abs(blackHoleSpace.y),.5*abs(tmpGauss))))
 ;temp*=o*pow(2.7,-(tmpGauss*tmpGauss)/.1)
 ;vma+=(1.-vma)*temp*AccretionDisk
 ;for(int L=0; L<3; L++
 ){
  ;ADD(sphere(w,oliPos[L]
              //o_lights[L].b
              ,0.001))
  ;
  ;vec3 relPos = oliPos[L]-pos//o_lights[L].b
  ;oliHal[L]//o_lights[L].haloResult 
      += o*(0.02/(dot(relPos,relPos)+0.01))
  ;}
 ;return o;}

// Function 125
float map(in vec3 rp)
{
    HIT_ID = GROUND;
    // stones
    float x = -(stonepolyfbm(rp.xz * 4.4) - 0.4) * 0.25 + sdBox(rp - vec3(0.0, 0.06, 0.0), vec3(2.0, .03, 2.0));
    
    // all the rest
    rp.y /= clamp((min(2.0 - abs(rp.z), (2.0 - abs(rp.x))) / 0.25), 0.6, 2.0);
    float l = rp.y - polyfbm(rp.xz * 2.4) * 2.2;
    
    float bounds = sdBox(rp - vec3(0.0, 0.5, 0.0), vec3(2.0, .6, 2.0));
    l = max(l, bounds); 
    x = max(x, bounds);
    if (x < l) HIT_ID = STONES; 
    
    return min(l, x);
}

// Function 126
vec3 luvToLch(vec3 tuple) {
    float L = tuple.x;
    float U = tuple.y;
    float V = tuple.z;

    float C = length(tuple.yz);
    float H = degrees(atan(V,U));
    if (H < 0.0) {
        H = 360.0 + H;
    }
    
    return vec3(L, C, H);
}

// Function 127
vec2 map(in vec3 p)
{
    // hit object ID is stored in res.x, distance to object is in res.y

    float rot90 = 6.2831 / 4.0;

    // walls mapping
    vec2 res = vec2(ID_FLOOR, sdBox(p + vec3(0.0, 1.0, 0.0), vec3(15.0, 1.0, 17.0)));
    //vec2 obj = vec2(ID_CEILING, sdBox(p + vec3(0.0, -17.0, 0.0), vec3(15.0, 0.0, 17.0)));
    //if (obj.y < res.y) res = obj;
    //obj = vec2(ID_WALL_BACK, sdBox(p + vec3(-10.5, 0.0, 0.0), vec3(0.5, 17.0, 17.0)));
    //if (obj.y < res.y) res = obj;
    vec2 obj = vec2(ID_WALL_FRONT, sdBox(p + vec3(15.0, 0.0, 0.0), vec3(0.0, 17.0, 17.0)));
    if (obj.y < res.y) res = obj;
    obj = vec2(ID_WALL_LEFT, sdBox(p + vec3(0.0, 0.0, 15.0), vec3(15.0, 17.0, 0.0)));
    if (obj.y < res.y) res = obj;
    //obj = vec2(ID_WALL_RIGHT, sdBox(p + vec3(0.0, 0.0, -17.0), vec3(15.0, 17.0, 0.0)));
    //if (obj.y < res.y) res = obj;

    // baseboards mapping
    vec3 size = vec3(17.0, 0.65, 0.065);
    obj = vec2(ID_BASEBOARD_YZ, sdBox(p + vec3(0.0, 0.0, 15.0), size));
    if (obj.y < res.y) res = obj;
    //obj = vec2(ID_BASEBOARD_YZ, sdBox(p + vec3(0.0, 0.0, -17.0), size));
    //if (obj.y < res.y) res = obj;
    obj = vec2(ID_BASEBOARD_XY, sdBox(rotateY(p + vec3(15.0, 0.0, 0.0), rot90), size));
    if (obj.y < res.y) res = obj;
    //obj = vec2(ID_BASEBOARD_XY, sdBox(rotateY(p + vec3(-10.0, 0.0, 0.0), rot90), size));
    //if (obj.y < res.y) res = obj;

    // carpet mapping
    if (res.x == ID_FLOOR) {
        obj = vec2(ID_CARPET_B, udRoundBox(rotateY(p + vec3(2.0, 0.0, -2.0), -2.0), vec3(8.0, 0.05, 7.0), 0.15));
        if (obj.y < res.y) res = obj;
        obj = vec2(ID_CARPET, sdBox(rotateY(p + vec3(2.0 - 0.075, 0.0, -2.0 + 0.025), -2.0), vec3(7.925, 0.25, 6.925)));
        if (obj.y < res.y) res = obj;
    }

    // tv mapping
    vec3 sizeTV = vec3(3.0, 9.0, 3.0);
    float d1 = sdBox(p + vec3(15.0, 0.0, 6.0), vec3(7.0, 13.0, 2.5));
    float d2 = udRoundBox(p + vec3(15.0, 0.0, 6.0), vec3(5.0, 1.2, 1.5), 0.5);
    float d3 = udRoundBox(p + vec3(12.0, -3.5, 6.0), vec3(1.0, 0.5, 1.5), 0.5);
    size = vec3(4.0, 1.0, 0.08);
    float d4 = sdBox(p + vec3(15.0, -3.5, 6.0), size);
    float d5 = sdBox(p + vec3(15.0, -3.5, 5.6), size);
    float d6 = sdBox(p + vec3(15.0, -3.5, 6.4), size);
    float d7 = udRoundBox(p + vec3(12.0, -7.25, 6.0), vec3(1.0, 0.61, 1.2), 1.0);
    obj = vec2(ID_TV_0, max(-d7, min(d6, min(d5, min(d4, max(-d3, max(-d2, max(d1, udRoundBox(p + vec3(15.0, 0.0, 6.0), sizeTV, 1.0)))))))));
    if (obj.y < res.y) res = obj;
    d1 = sdBox(p + vec3(15.18, 0.25, 3.5), vec3(7.0, 13.0, 0.2));
    size = vec3(0.1, 1.0, 1.0);
    d2 = sdBox(p + vec3(12.8, -3.5, 4.0), size);
    d3 = sdBox(p + vec3(13.2, -3.5, 4.0), size);
    d4 = max(-d3, max(-d2, max(d1, udRoundBox(p + vec3(15.18, 0.25, 3.5), sizeTV, 1.0))));
    d5 = sdBox(p + vec3(15.18, 0.25, 8.5), vec3(7.0, 13.0, 0.2));
    //d6 = sdBox(p + vec3(12.8, -3.5, 8.0), size);
    //d7 = sdBox(p + vec3(13.2, -3.5, 8.0), size);
    //obj = vec2(ID_TV_1, min(d4, max(-d7, max(-d6, max(d5, udRoundBox(p + vec3(15.18, 0.25, 8.5), sizeTV, 1.0))))));
    obj = vec2(ID_TV_1, min(d4, max(d5, udRoundBox(p + vec3(15.18, 0.25, 8.5), sizeTV, 1.0))));
    if (obj.y < res.y) res = obj;
    d1 = sdBox(p + vec3(8.25, -3.5, 6.0), sizeTV);
    obj = vec2(ID_TV_2, max(-d1, udRoundBox(p + vec3(12.0, -3.5, 6.0), vec3(1.0, 0.5, 1.5), 0.5)));
    if (obj.y < res.y) res = obj;
    d1 = sdBox(p + vec3(7.85, -7.25, 6.0), sizeTV);
    d2 = udRoundBox(p + vec3(12.0, -7.25, 6.0), vec3(1.0, 0.6, 1.175), 1.0);
    obj = vec2(ID_TV_3, max(-d2, max(-d1, udRoundBox(p + vec3(12.0, -7.25, 6.0), vec3(1.0, 0.7, 1.25), 1.0))));
    if (obj.y < res.y) res = obj;
    obj = vec2(ID_TV_4, udRoundBox(p + vec3(12.75, -7.25, 6.0), vec3(1.0, 1.05, 1.75), 0.5));
    if (obj.y < res.y) res = obj;
    d1 = sdCappedCylinder(rotateZ(p + vec3(11.25, -5.0, 6.25), rot90), vec2(0.12, 0.425));
    obj = vec2(ID_TV_5, min(d1, sdCappedCylinder(rotateZ(p + vec3(11.25, -5.0, 5.75), rot90), vec2(0.12, 0.325))));
    if (obj.y < res.y) res = obj;
    vec2 sizePot = vec2(0.3, 0.4);
    d1 = sdCappedCylinder(rotateZ(p + vec3(11.25, -5.0, 4.0), rot90), sizePot);
    d2 = sdCappedCylinder(rotateZ(p + vec3(11.25, -5.0, 8.0), rot90), sizePot);
    d3 = sdBox(p + vec3(10.425, -5.0, 4.0), vec3(0.25));
    d4 = sdSphere(p + vec3(10.84, -5.0, 8.0), 0.25);
    d5 = sdBox(p + vec3(10.425, -5.0, 8.0), vec3(0.25));  
    d6 = sdCappedCylinder(rotateZ(p + vec3(10.675, -5.0, 4.0), rot90), vec2(0.175, 0.01));
    d7 = sdCappedCylinder(rotateZ(p + vec3(10.675, -5.0, 8.0), rot90), vec2(0.175, 0.01));
    float rot = 6.2831 / 3.428;
    float d8 = sdBox(rotateZ(p + vec3(10.8, -5.2, 4.0), rot), vec3(0.08, 0.1, 0.04));
    float d9 = sdBox(rotateZ(p + vec3(10.8, -5.2, 8.0), rot), vec3(0.08, 0.1, 0.04));
    obj = vec2(ID_TV_6, min(d9, min(d8, min(d7, min(d6, max(-d5, min(d4, max(-d3, min(d2, min(d1, sdSphere(p + vec3(10.84, -5.0, 4.0), 0.25)))))))))));
    if (obj.y < res.y) res = obj;
    sizePot = vec2(0.29, 0.01);
    d1 = sdCappedCylinder(rotateZ(p + vec3(10.84, -5.0, 4.0), rot90), sizePot);
    d2 = sdCappedCylinder(rotateZ(p + vec3(10.675, -5.0, 4.0), rot90), vec2(0.2, 0.01));
    d3 = sdCappedCylinder(rotateZ(p + vec3(10.675, -5.0, 8.0), rot90), vec2(0.2, 0.01));
    obj = vec2(ID_TV_7, min(d3, min(d2, min(d1, sdCappedCylinder(rotateZ(p + vec3(10.84, -5.0, 8.0), rot90), sizePot)))));
    if (obj.y < res.y) res = obj;

    // chair mapping
    d1 = sdBox(p + vec3(3.0, -2.0, 4.5), vec3(1.8, 3.1, 0.5));
    d2 = max(-d1, sdBox(p + vec3(3.0, -2.5, 4.5), vec3(2.1, 2.9, 0.2)));
    d3 = sdBox(p + vec3(3.0, -2.0, 8.5), vec3(1.8, 3.1, 0.5));
    d4 = max(-d3, sdBox(p + vec3(3.0, -2.5, 8.5), vec3(2.1, 2.9, 0.2)));
    d5 = sdBox(p + vec3(1.05, -2.0, 6.5), vec3(0.5, 4.3, 1.8));
    d6 = max(-d5, sdBox(p + vec3(1.05, -2.5, 6.5), vec3(0.2, 4.1, 2.2)));
    size = vec3(2.0, 0.075, 0.075);
    d7 = sdBox(p + vec3(3.0, -4.35, 4.5), size);
    d8 = sdBox(p + vec3(3.0, -4.35, 8.5), size);
    d9 = sdBox(p + vec3(1.05, -4.35, 6.5), vec3(0.075, 0.075, 2.0));
    float d10 = sdBox(p + vec3(1.05, -5.35, 6.5), vec3(0.075, 0.075, 2.0));
    size = vec3(0.075, 1.0, 0.075);
    d1 = sdBox(p + vec3(2.35, -4.25, 4.5), size);
    d3 = sdBox(p + vec3(3.65, -4.25, 4.5), size);
    d5 = sdBox(p + vec3(2.35, -4.25, 8.5), size);
    float d11 = sdBox(p + vec3(3.65, -4.25, 8.5), size);
    size = vec3(0.075, 1.75, 0.075);
    float d12 = sdBox(p + vec3(1.05, -4.75, 5.85), size);
    //float d13 = sdBox(p + vec3(1.05, -4.75, 7.25), size);
    //obj = vec2(ID_CHAIR_0, min(d13, min(d12, min(d11, min(d5, min(d3, min(d1, min(d10, min(d9, min(d8, min(d7, min(d6, min(d4, min(d2, sdBox(p + vec3(3.0, -3.25, 6.5), vec3(2.1, 0.2, 2.2))))))))))))))));
    obj = vec2(ID_CHAIR_0, min(d9, min(d8, min(d7, min(d6, min(d4, min(d2, sdBox(p + vec3(3.0, -3.25, 6.5), vec3(2.1, 0.2, 2.2)))))))));
    if (obj.y < res.y) res = obj;
    obj = vec2(ID_CHAIR_0, min(d12, min(d11, min(d5, min(d3, min(d1, min(d10, sdBox(p + vec3(1.05, -4.75, 7.25), size))))))));
    if (obj.y < res.y) res = obj;   
    obj = vec2(ID_CHAIR_1, udRoundBox(p + vec3(3.05, -3.26, 6.5), vec3(1.8, 0.2, 1.8), 0.025));
    if (obj.y < res.y) res = obj;
    //obj = mapChair(p);
    //if (obj.y * 0.95 < res.y) res = obj;

    // floor bumps
    if (res.x == ID_FLOOR)
    {
        if (p.x < 9.85 && p.x > -14.85 && p.z < 16.85 && p.z > -14.85)
            res.y += 0.1 * texture(iChannel0, 0.1875 * p.xz).x;
    }

    return res;
}

// Function 128
SDObject mapScene(vec3 p)
{
    SDObject earth=newSDObject(0,vec3(0.));
    SDObject moon=newSDObject(4,vec3(sin(iTime),0.,cos(iTime))*10.5);

        SDObject res=noSDObject();
    //   SDObject res=SDOMap(moon,ray);
    res= SDOMap(earth,p);
//	res= SDUnion(res,SDOMap(moon,p));
	return res;
}

// Function 129
vec3 uv2dir(Camera cam, vec2 uv){
    return normalize(vec3(uv, cam.focalLength)) * rotationMatrix(cam.rot);
}

// Function 130
vec4 DrawtexturedUVQuad(vec2 a, vec2 b, vec2 c, vec2 d,vec2 uva, vec2 uvb, vec2 uvc, vec2 uvd, float t, vec2 co, sampler2D s){
    float i = DrawQuad(a,b,c,d,t,co);
    if (i<=0.) return vec4(0.);
    vec3 baria = toBari(a,b,c,co);
    vec3 barib = toBari(a,d,c,co);
    vec3 baric = toBari(b,c,d,co);
    vec3 barid = toBari(b,a,d,co);
    vec2 coord = vec2(0.);
    coord+= toCartesian(uvb,uvc,uvd,baric);
    coord+= toCartesian(uvb,uva,uvd,barid);
    coord+= toCartesian(uva,uvb,uvc,baria);
    coord+= toCartesian(uva,uvd,uvc,barib);
    
    return texture(s,coord/4.)*i;
}

// Function 131
vec3 getFaceUVW(vec2 uv, float faceIdx, vec2 faceSize)
{
    float a = 2.0*uv.x/faceSize.x;
    float b = 2.0*uv.y/faceSize.y;
    	 if (faceIdx<0.5) return vec3(-1., 	1.-a, 3.-b);// back
    else if (faceIdx<1.5) return vec3(a-3., -1.,  3.-b);// left
    else if (faceIdx<2.5) return vec3(1., 	a-5., 3.-b);// front
    else if (faceIdx<3.5) return vec3(7.-a, 1.,   3.-b);// right
    else if (faceIdx<4.5) return vec3(b-1., a-5., 1.  );// top
    					  return vec3(5.-b, a-5., -1. );// bottom
}

// Function 132
v0 suv(v2 a){return dot(v2(1),a);}

// Function 133
float mapClouds(vec3 p)
{
    float h = FBM(p*.0001);
	return clamp(h-.64, 0.0, 1.0) * smoothstep(40000.0, 2500.0,p.y) * smoothstep(250.0, 2500.0,p.y);
}

// Function 134
float map_room(vec3 pos)
{
   vec3 roomSize2 = roomSize;
   #ifdef doors
   roomSize2.xz-= dfSize.y*smoothstep(2.*dfSize.x + 0.01, 2.*dfSize.x - 0.01, pos.y);
   #endif    
    
   float room = -sdRoundBox(pos + vec3(0, -roomSize2.y*0.5 + 0.01, 0.), roomSize2*0.5, 0.);
   #ifdef doors

   vec3 pos2 = pos;
   pos2.x = abs(pos.x);
   room = max(room, -sdRoundBox(pos2 + vec3(-roomSize.z - doorSize.z, -roomSize.y*0.5 + 0.01, 0.), roomSize*0.5, 0.));
   room = min(room, sdRoundBox(pos2 + vec3(-roomSize.z*0.5 - doorSize.z*0.5, -doorSize.y*0.5 + 0.01, 0.), doorSize.zyx*0.5 + vec3(0.5*dfSize.y, 2.*dfSize.x, 2.*dfSize.x), 0.));
   room = max(room, -sdRoundBox(pos2 + vec3(-roomSize.z*0.5 - doorSize.z*0.5, -doorSize.y*0.5 + 0.01, 0.), doorSize.zyx*0.5 + vec3(0.5*dfSize.y + 0.01, 0., 0.), 0.));
   #endif
    
   return room;
}

// Function 135
float map_spheres(vec3 pos, vec3 center, float d, float r)
{
    pos-= center;
    pos.xz = rotateVec(pos.xz, iTime*0.5);
    pos.x = abs(pos.x);
    pos.x-= d;
    pos.z = abs(pos.z);
    pos.z-= d;
    return length(pos) - r;   
}

// Function 136
vec4 lchToHsluv(vec4 c) {return vec4( lchToHsluv( vec3(c.x,c.y,c.z) ), c.a);}

// Function 137
vec2 map(in vec3 pos)
{
  return vec2(equator_fibers(pos), 1.0);
}

// Function 138
float map(vec3 p){
    
    
     float sf = cellTile(p*.25); // Cellular layer.
    
     p.xy -= path(p.z); // Move the scene around a sinusoidal path.
     p.xy = rot2(p.z/12.)*p.xy; // Twist it about XY with respect to distance.
    
     float n = dot(sin(p*1. + sin(p.yzx*.5 + iTime)), vec3(.25)); // Sinusoidal layer.
     
     return 2. - abs(p.y) + n + sf; // Warped double planes, "abs(p.y)," plus surface layers.
   

     // Standard tunnel. Comment out the above first.
     //vec2 tun = p.xy - path(p.z);
     //return 3. - length(tun) - (0.5-surfFunc(p)) +  dot(sin(p*1. + sin(p.yzx*.5 + iTime)), vec3(.333))*.5+.5;

 
}

// Function 139
float map_detailed(vec3 p) {
    float freq = SEA_FREQ;
    float amp = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    vec2 uv = p.xz; uv.x *= 0.75;
    
    float d, h = 0.0;    
    for(int i = 0; i < ITER_FRAGMENT; i++) {        
    	d = sea_octave((uv+SEA_TIME)*freq,choppy);
    	d += sea_octave((uv-SEA_TIME)*freq,choppy);
        h += d * amp;        
    	uv *= octave_m; freq *= 1.9; amp *= 0.22;
        choppy = mix(choppy,1.0,0.2);
    }
    return p.y - h;
}

// Function 140
void drawMap(inout vec4 c, vec2 fragCoord)
{
    if (iResolution.x < 420.)
        return;
    
    // show map memory.
    ivec2 ifc = ivec2(fragCoord/4.);
    if (ifc.x > 0 && ifc.x < SX && ifc.y > 0 &&ifc.y < SY)  
        c = mix(c, textureBase ( iChannel0, ifc), .5 );
    
}

// Function 141
vec3 Tonemap( vec3 x )
{
#if 0 
    
    vec3 luminanceCoeffsBT709 = vec3( 0.2126f, 0.7152f, 0.0722f );
    float f = dot( x, luminanceCoeffsBT709 );
    x /= f;        
    f = 1.0f - exp(-f);    
    x *= f;    
    x = mix( x, vec3(f), f*f );
    
    return x;
#else       
    float a = 0.010;
    float b = 0.132;
    float c = 0.010;
    float d = 0.163;
    float e = 0.101;

    return ( x * ( a * x + b ) ) / ( x * ( c * x + d ) + e );    
#endif    
}

// Function 142
vec3 uv(vec3 point, vec3 nml, sampler2D samp) {
	vec3 texX = texture(samp, point.zy).xyz;
    vec3 texY = texture(samp, point.xz).xyz;
    vec3 texZ = texture(samp, point.xy).xyz;
    
    return texX * nml.x + texY * nml.y + texZ * nml.z;
}

// Function 143
vec3 hsluv_toLinear(vec3 c) {  return vec3( hsluv_toLinear(c.r), hsluv_toLinear(c.g), hsluv_toLinear(c.b) ); }

// Function 144
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    vec3 skyCol = vec3(80,180,255)/ 255.0;
    vec3 horizonCol = vec3(255,255,255)/ 255.0;
    vec3 groundCol = vec3(0,0,35)/ 255.0;
    
    vec3 sunCol = vec3(1.0, 1.0, 1.0);
    float sunSize = 0.0005;
    float sunBlur = 0.0025;
    float sunScatter = 0.25;
    
    vec3 moonCol = vec3(1.0, 1.0, 1.0);
    float moonSize = 0.00025;
    float moonGlow = 0.05;
    float moonShadow = 0.5;
    float starSize = 0.04;
    
    vec3 sunPos = _sunPos(iTime); 
    vec3 moonPos = _moonPos(iTime);
    vec3 sunDir = normalize(sunPos - rayOri); 
    vec3 moonDir = normalize(moonPos - rayOri); 
    float pitch =  0.5 + rayDir.y * 0.5;
    
    // sky
    // -----------------------------------------------------------
    vec3 col = skyCol;
    horizonCol = mix (vec3(255,55,0)/ 255.0, vec3(255,255,255)/ 255.0, smoothstep(-0.5, 0.75, sunPos.y));
    horizonCol = mix (vec3(55,155,155)/ 255.0, horizonCol, smoothstep(-0.5, 0.0, sunPos.y));        
    col = mix(horizonCol, col, smoothstep(0.4, 0.8, pitch));    
    col = mix(groundCol, col, smoothstep(0.49, 0.5, pitch));
    
    // sun
    // ------------------------------------------------------------
    sunCol = mix(vec3(255,85,0)/ 255.0, vec3(255,255,255)/ 255.0, max(0.0, min(1.0, smoothstep(-0.4, 0.8, sunPos.y))));
    float sun = dot(sunDir, rayDir);    // return [-1,1] based on angle 
    sunSize += sunSize * 100.0 * (1.0-smoothstep(-0.8, 0.11, sunPos.y)); // scale sun based on height
    float sunDisk = smoothstep((1.0 - sunSize) - sunSize*2.0, 1.0 - sunSize, sun); // define sun disk
    sunScatter = smoothstep((1.0 - sunSize) - sunScatter, 1.0 - sunSize, sun) *  smoothstep(-0.8, 0.2, sunPos.y); 
    float sunGlow = smoothstep((1.0 - sunSize) - sunBlur, 1.0 - sunSize, sun);

    float haloSize = 0.02;
    float sunHalo = smoothstep(0.9999 - haloSize, 0.9999, sun);
    sunHalo = mix(0.055, 0.05, fract(sunHalo*2.) );

    //float angle = acos( dot(sunDir, rayDir) / (length(sunDir)*length(rayDir)) );
    //float sunFlare = smoothstep((1.0 - sunSize*2.0) - 0.05*smoothstep(0.0, 3.14*.25,angle*10.) , 1.0 - sunSize*2.0, sun);
    float horizonMask = smoothstep( -0.2, 0.05, rayDir.y); // mask sun along horizon
    sunDisk *= smoothstep( -0.05, 0.025, rayDir.y);
    sunGlow *= smoothstep( -0.05, 0.025, rayDir.y);
    sunHalo *= smoothstep( -0.2, 0.05, rayDir.y); // mask sun along horizon
    
    col = mix(col, sunCol, sunScatter*.25);
    col = mix(col, sunCol, sunGlow*.5);
    col += sunDisk + sunScatter*.05 + sunGlow*.05;
    col += vec3(sunHalo*6.0, 0.0,0.0);

    // night
    // ------------------------------------------------------------
    col *= max(0.35, min(1.0, smoothstep(-1.0, 0.0, sunPos.y)));
    
    // stars
    // ------------------------------------------------------------
    float starNoise = random3D( rayDir, vec3(0.5,0.1,7.0+ iTime*0.001));
    vec2 starCoords = vec2(fract(rayDir.x * 50.0), fract(rayDir.y * 50.0) );  
    float stars = step(0.6, starCoords.y) * (1.0-step(0.6 + starSize, starCoords.y)) * step(0.3, starCoords.x) * (1.0- step(0.3 + starSize, starCoords.x));
    stars *= smoothstep(-0.01, 0.00, rayDir.y);
    col += stars * pow(starNoise, 3.0) * (1.0-max(0.0, min(1.0, smoothstep(-1.0, 0.0, sunPos.y))));
       
    horizonMask = smoothstep( -0.01, 0.01, rayDir.y); // mask sun along horizon
    // moon
    // -------------------------------------------------------------
    float moon = dot(moonDir, rayDir);
    moonShadow = mix(-0.05, -0.0, moonShadow);
    moonShadow = dot(vec4(rotY(moonShadow)*vec4(moonDir,1.0)).xyz, rayDir);
    float moonDisk = smoothstep(1.0-moonSize-0.00025, 1.0-moonSize, moon); 
    moonShadow = smoothstep(1.0-moonSize*2.0-0.00025, 1.0-moonSize*2.0, moonShadow); 
    moonDisk *= (1.0 - moonShadow*.995) * horizonMask  * smoothstep(0.0, 0.3, moonPos.y);    
    col = mix(col, moonCol, moonDisk);
    col += moonDisk;
    
    moonGlow = smoothstep(1.0-moonSize-moonGlow, 1.0-moonSize, moon) * horizonMask  * smoothstep(0.0, 0.3, moonPos.y); 
    col = mix(col, moonCol, moonGlow*.025) ; 
    
    
    // Output to cubemap
    fragColor = vec4(col,1.0);
}

// Function 145
vec2 map0ds(vec2 z)
{

  float s = length(z);
  float m = s/(max(abs(z.x),abs(z.y))+epsilon);
  return m*z;
}

// Function 146
vec3 mapRMNormal(vec3 pt, float e) {
    vec3 normal;
    normal.y = mapRMDetailed(pt).x;    
    normal.x = mapRMDetailed(vec3(pt.x+e,pt.y,pt.z)).x - normal.y;
    normal.z = mapRMDetailed(vec3(pt.x,pt.y,pt.z+e)).x - normal.y;
    normal.y = e;
    return normalize(normal);
}

// Function 147
float map(in vec3 p) {
	
	float res = 0.;
	
    vec3 c = p;
	for (int i = 0; i < 10; ++i) {
        p =.7*abs(p)/dot(p,p) -.7;
        p.yz= csqr(p.yz);
        p=p.zxy;
        res += exp(-19. * abs(dot(p,c)));
        
	}
	return res/2.;
}

// Function 148
float MapThorns( in vec3 pos)
{
    float which;
	return pos.y * .21 - ThornVoronoi(pos, which).w  - max(pos.y-5.0, 0.0) * .5 + max(pos.y-5.5, 0.0) * .8;
}

// Function 149
float map(vec3 p){  
	
    p.y+=0.9;
    
    float sphere_radius = 0.35;
    vec3 sp = p;
    float alle  = 1000.;
    
    sp = p;
    sp.y += 0.5;
    vec3 csp = sp;
    sp.xz = fract(p.xz)-0.5;
    
    ////////////////////CYL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    float ride = spinner(p);
    
    
    /////////////////////SEGS////////////////////////////
    vec2 id = floor(p.xz);
    float ball_gross = fract(sin(dot(vec2(12.55,74.2),id))*452354.);
    float t =  iTime/6.;
    
    
    float gross = 1.3;
    float depth = 0.6;
    
    float height = fbm2(id+t)*gross+depth;
    float heightL = fbm2((id + vec2(0.,1.))+t)*gross+depth;
    float heightR = fbm2((id + vec2(0.,-1.))+t)*gross+depth;
    float heightV = fbm2((id + vec2(-1.,0.))+t)*gross+depth;
    float heightH = fbm2((id + vec2(1.,0.))+t)*gross+depth;
    
    vec3 vor = vec3(-1.,heightV,0.);
    vec3 hin = vec3(1.0,heightH,0.0);
    vec3 links = vec3(0.,heightR,-1.);
    vec3 rechts = vec3(0., heightL,1.0);
    vec3 zentrum = vec3(0.,height,0.);
    
    
    float k = seggy(sp, zentrum, vor);
    float segs = k;
    k = seggy(sp, zentrum, links);
    segs = min(segs, k);
    k = seggy(sp, zentrum, rechts);
    segs = min(segs, k);
    k = seggy(sp, zentrum, hin);
    segs = min(segs, k);
    
    k = seggy(sp, zentrum, vec3(0.,-depth,0.));
    segs = min(segs, k);
    
    alle = min(segs,alle);
    
    //#######################BALL0000000000000000000000000000000000
    
    
    float ball = length(sp - vec3(0.,height,0.))-0.03-ball_gross*0.02;
    
    
    
    alle = min(alle, ball);
    
    ////////////////PLANE_____________________________________
    float plane = sp.y+0.6;
    
    ////////////////////ID@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if(plane < alle && plane < segs && plane < ride)mid = 2.;
    else if(ball < plane && ball < segs && ball < ride){
        mid = 1.;
        //glow += (0.00001/(0.75*pow((ball),2.)));
        }
    else if (segs < plane && segs < ball && segs < ride)mid = 0.;
    else if (ride < plane && ride < ball)mid = 3.;
    
    //float ride = 
    alle = min(alle,ride);
    return min(alle, plane);//-fbm4(p.xz));
}

// Function 150
float mapSeed(vec2 uv)
{
    //uv = (uv + 1.)/2.;
    DecodeData(texelFetch( iChannel0, ivec2(uv*iResolution.xy),0), seedCoord, seedColor);
    return min(LIGHT_DIST, length((floor(seedCoord)-floor(uv*iResolution.xy))/iResolution.x)-seedColor.z/60.);
    //return length(seedCoord/iResolution.xy-uv)-seedSize;
}

// Function 151
vec2 map( in vec3 pos )
{
 
	vec3 center = pos - vec3(0,.7, 0);												
	vec2 o = 			hilbert (center,1.2);										// slightly oversized to show the idea
		 //o = opU( o, 	hilbert(opRep(center,vec3(.25,.25,.25)),.125) );            // repeating does not include rotation :/?
		 o = opU( o, 	hilbertBlock( center               - vec3(.5), center               + vec3(.5),.25) );
		 //o = opU( o, 	hilbertBlock( center + vec3(1,0,0) - vec3(.5), center + vec3(1,0,0) + vec3(.5),.25) ); //next block
		 
	

	vec2 res = opU( vec2( sdPlane(     pos),1. ),
					o);
        
    return res;
}

// Function 152
vec2 remap_11to01(vec2 a)
{
    return vec2(remap_11to01(a.x), remap_11to01(a.y));
}

// Function 153
vec2 reverse_mapping(vec2 Z,vec3 R, int seed){
    
    int p = int(R.x);
    int q = int(R.y);
    
    int x=int(Z.x);
    int y=int(Z.y);
    
    for(int i = 0; i < mapping_iters; i++){
        x = Zmod(x - IHash(y^seed)%p,p);
        y = Zmod(y - IHash(x^seed)%q,q);
    }
        
    return vec2(x,y)+.5;
}

// Function 154
vec4 getBitmapColor( in vec2 uv )
{
        return getColorFromPalette( getPaletteIndex( uv ) );
}

// Function 155
vec3 ToneMap_Uncharted2(vec3 color)
{
    float A = 0.15; // 0.22
	float B = 0.50; // 0.30
	float C = 0.10;
	float D = 0.20;
	float E = 0.02; // 0.01
	float F = 0.30;
	float W = 11.2;
    
    vec4 x = vec4(color, W);
    x = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
	return sRGMGamma(x.xyz / x.w);
}

// Function 156
float map(vec3 p){
    
 
    p.xy -= path(p.z); // Wrap the passage around
    
    vec3 w = p; // Saving the position prior to mutation.
    
    vec3 op = tri(p*.4*3. + tri(p.zxy*.4*3.)); // Triangle perturbation.
   
    
    float ground = p.y + 0.05 + dot(op, vec3(.222))*.13; // Ground plane, slightly perturbed.
 
    p += (op - .15)*.3; // Adding some triangular perturbation.
   
	p = cos(p*.315*1.41 + sin(p.zxy*.875*1.27)); // Applying the sinusoidal field (the rocky bit).
    
    float canyon = (length(p) - 1.05)*.95 - (w.x*w.x)*.51; // Spherize and add the canyon walls.
    
    return min(ground, canyon);

    
}

// Function 157
float map2( in vec3 p, in float id )
{
    float w = 0.05 + 0.35*id;
    return length(max(abs(p)-0.5+w,0.0))-w+0.001;
}

// Function 158
vec4 mapText(vec2 p, vec2 tp, float s, ivec4 text) { 
    vec4 chr = vec4(0.);
    int k = 0;
    p -= tp;
    p+=vec2( .5/s );
    p *= vec2(s);
    int ip = 0;
    float len = 0.;
    for ( int i = 0; i < 20; i++){					 	// ivec4( 0x_01_02_03_04, 0x_05_06_07_08, 0x_09_10_11_12, 0x_13_14_15_16 )
        int chv = map(text,k); // text[k] 
        if (chv == 0) break; 					 	// nothing to process then break
        if (ip == 4) {
            k++; ip=0; // continue; // next index to process
        } else
        {
            //int ch = (chv >> (8*(3-( ip % 4)))) % 256;		// ascii code from ivec4 

            int ch =  modi((shift(chv, 8*(3-(modi((ip),4))))),256); // compatibility mode
            
            if (ch == 0) {
                ip++; //continue; 								// next index to process
            }else
            {
                chr = max(chr,  char(p, ch).x); 				// draw           
                len++;
                p-=vec2(float(.5),0.);						 	// move postion to next char
                ip++;											// next char
            }                
        }
    }  
    return vec4(chr.xyz, len);                   
}

// Function 159
vec3 bumpMap(in vec3 p, in vec3 n, float bumpfactor){
    
    const vec2 e = vec2(0.002, 0);
    float ref = bumpFunc(p, n);                 
    vec3 grad = (vec3(bumpFunc(p - e.xyy, n),
                      bumpFunc(p - e.yxy, n),
                      bumpFunc(p - e.yyx, n) )-ref)/e.x;                     
          
    grad -= n*dot(n, grad);          
                      
    return normalize( n + grad*bumpfactor );
	
}

// Function 160
vec2 map( in vec3 pos )
{
    vec2 res = opU( vec2( sdPlane(     pos), 1.0 ),
	                vec2( sdSphere(    pos-vec3( 0.0,0.25, 0.0), 0.25 ), 46.9 ) );
    res = opU( res, vec2( sdBox(       pos-vec3( 1.0,0.25, 0.0), vec3(0.25) ), 3.0 ) );
    res = opU( res, vec2( udRoundBox(  pos-vec3( 1.0,0.25, 1.0), vec3(0.15), 0.1 ), 41.0 ) );
	res = opU( res, vec2( sdTorus(     pos-vec3( 0.0,0.25, 1.0), vec2(0.20,0.05) ), 25.0 ) );
    res = opU( res, vec2( sdCapsule(   pos,vec3(-1.3,0.10,-0.1), vec3(-0.8,0.50,0.2), 0.1  ), 31.9 ) );
	res = opU( res, vec2( sdTriPrism(  pos-vec3(-1.0,0.25,-1.0), vec2(0.25,0.05) ),43.5 ) );
	res = opU( res, vec2( sdCylinder(  pos-vec3( 1.0,0.30,-1.0), vec2(0.1,0.2) ), 8.0 ) );
	res = opU( res, vec2( sdCone(      pos-vec3( 0.0,0.50,-1.0), vec3(0.8,0.6,0.3) ), 55.0 ) );
	res = opU( res, vec2( sdTorus82(   pos-vec3( 0.0,0.25, 2.0), vec2(0.20,0.05) ),50.0 ) );
	res = opU( res, vec2( sdTorus88(   pos-vec3(-1.0,0.25, 2.0), vec2(0.20,0.05) ),43.0 ) );
	res = opU( res, vec2( sdCylinder6( pos-vec3( 1.0,0.30, 2.0), vec2(0.1,0.2) ), 12.0 ) );
	res = opU( res, vec2( sdHexPrism(  pos-vec3(-1.0,0.20, 1.0), vec2(0.25,0.05) ),17.0 ) );
	res = opU( res, vec2( sdPryamid4(  pos-vec3(-1.0,0.15,-2.0), vec3(0.8,0.6,0.25) ),37.0 ) );
    res = opU( res, vec2( opI( sdBox(    pos-vec3( 2.0,0.2, 1.0), vec3(0.20)),
	                           sdSphere( pos-vec3( 2.0,0.2, 1.0), 0.25)), 113.0 ) );
    res = opU( res, vec2( opS( udRoundBox(  pos-vec3(-2.0,0.2, 1.0), vec3(0.15),0.05),
	                           sdSphere(    pos-vec3(-2.0,0.2, 1.0), 0.25)), 13.0 ) );
    res = opU( res, vec2( opS( sdTorus82(  pos-vec3(-2.0,0.2, 0.0), vec2(0.20,0.1)),
	                           sdCylinder(  opRep( vec3(atan(pos.x+2.0,pos.z)/6.2831, pos.y, 0.02+0.5*length(pos-vec3(-2.0,0.2, 0.0))), vec3(0.05,1.0,0.05)), vec2(0.02,0.6))), 51.0 ) );
    // distance deformation (knobbly sphere):
	res = opU( res, vec2( 0.5*sdSphere(    pos-vec3(-2.0,0.25,-1.0), 0.2 )
                           + 0.03*sin(50.0*pos.x)*sin(50.0*pos.y)*sin(50.0*pos.z)
                         , 65.0 ) );
    
	res = opU( res, vec2( 0.5*sdTorus( opTwist(    pos-vec3(-2.0,0.25, 2.0)),vec2(0.20,0.05)), 46.7 ) );
	res = opU( res, vec2( 0.3*sdTorus( opCheapBend(pos-vec3( 2.0,0.25,-1.0)),vec2(0.20,0.05)), 46.7 ) );

    res = opU( res, vec2( sdConeSection( pos-vec3( 0.0,0.35,-2.0), 0.15, 0.2, 0.1 ), 13.67 ) );
    res = opU( res, vec2( sdEllipsoid( pos-vec3( 1.0,0.35,-2.0), vec3(0.15, 0.2, 0.05) ), 43.17 ) );
    // scaled primitive:
    const float scale = .4;
    res = opU( res, vec2( sdSphere((pos - vec3(-2.0, 0.25, -2.0))/scale, 0.25)*scale, 70. ) );
    
    res = opU( res, vec2( opBlend( sdBox(      pos-vec3( 2.0,0.25, 0.0), vec3(.15,.05,.15) ),
                                   sdCylinder( pos-vec3( 2.0,0.25, 0.0), vec2(0.04,0.2))), 75. ) );
    return res;
}

// Function 161
vec4 hsluvToLch(vec4 c) {return vec4( hsluvToLch( vec3(c.x,c.y,c.z) ), c.a);}

// Function 162
void combined_uv( inout vec4 fragColor, in vec2 uv)
{
    if (ANIM_TURN)
    {
        uv -= win_mouse;
        float angle = sin(iTime * TURN_FREQ) * ANIM_AMPLITUDE;
        vec2 v1 = vec2(cos(angle), sin(angle));
        vec2 v2 = vec2(-v1.y, v1.x);
        uv = v1 * uv.x + v2 * uv.y;
        uv += win_mouse;
    }
    
    if (ANIM_SCALE)
    {
        uv -= win_mouse;
        float scale = exp(cos(iTime * SCALE_FREQ) * ANIM_AMPLITUDE);
        uv *= scale;
        win_scale = 1.0/scale;
        uv += win_mouse;
    }
    
    texture_uv(fragColor, uv);
    dot_uv(fragColor, (uv - win_mouse) * win_size, RING_FREQ);
}

// Function 163
vec2 planeUV(vec3 origin, vec3 n, vec3 p)
{
    vec3 uAxis;
    vec3 vAxis;
    planeUVAxis(origin, n, uAxis, vAxis);
    vec3 diff = p - origin;
    float uVal = dot(diff, uAxis);
    float vVal = dot(diff, vAxis);
    return vec2(uVal, vVal);
}

// Function 164
float map(vec3 p)
{	
// 	fractalscape 
    float f = -0.05-kifs(.4*p);
	if(meep==0) f+=0.002*noise(p*70.);
	return f;
}

// Function 165
vec3 dirFromUv(Camera cam, vec2 uv){
    return normalize(vec3(uv, cam.focalLength)) * rotationMatrix(cam.rot);
}

// Function 166
vec3 reinhardTonemapping(vec3 color) {
    return color = color / (color + vec3(1.0));
}

// Function 167
vec3 Tonemap(vec3 col)
{
    #if 1
    #ifdef USE_ACESCG
	col = Tonemap_ACESFitted2(col);
    #else
	col = Tonemap_ACESFitted(col);
    #endif
    #endif
    
    col = clamp01(col);
    
    return col;
}

// Function 168
vec3 doBumpMap( sampler2D tx, in vec3 p, in vec3 n, float bf)
{
    const vec2 e = vec2(0.001, 0);
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
}

// Function 169
vec3 seaHeightMap(vec3 dir) 
{
    vec3 p = vec3(0.0);
    float x = 1000.0;
	
    if (seaGeometryMap(SEA_ORI + dir * x) > 0.0)
    {
		return p;
    }
    
    float mid = 0.0;
    float m = 0.0;
    float heightMiddle = 0.0;
    for(int i = 0; i < HEIGHTMAP_NUM_STEPS; ++i) 
    {	    
		mid = mix(m, x, 0.5); 
        p = SEA_ORI + dir * mid;
    	heightMiddle = seaGeometryMap(p);
		if (heightMiddle < 0.0) 
		{
            x = mid;
        } 
		else 
		{
            m = mid;
        }
    }
	
    return p;
}

// Function 170
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    const vec2 e = vec2(0.001, 0);
    float ref = bumpSurf3D(p, nor);                 
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy, nor),
                      bumpSurf3D(p - e.yxy, nor),
                      bumpSurf3D(p - e.yyx, nor) )-ref)/e.x;                     
          
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
	
}

// Function 171
float map(vec3 p){
    
    // Floor.
    //float fl = p.y;

    // The extruded blocks.
    vec4 d4 = blocks(p.xzy);
    gID = d4; // Individual block ID.
 
    // Overall object ID.
    objID = p.y<d4.x? 1. : 0.;
    
    // Combining the floor with the extruded image
    return min(p.y, d4.x);
 
}

// Function 172
vec2 UVFromViewSpace(vec2 view)
{
    /* Invert aspect corrected coordinate to uv coordinate */
    view.x *= iResolution.y / iResolution.x;
    return view * 0.5 + 0.5;
}

// Function 173
vec2 uvSmooth(vec2 uv,vec2 res)
{
    return uv+.6*sin(uv*res*PI2)/PI2/res;
}

// Function 174
vec2 getRadialUv(vec2 uv) {
    float angle = atan(uv.x, uv.y);

    vec2 radialUv = vec2(0.0);
    radialUv.x = angle / (M_PI * 2.0) + 0.5;
    radialUv.y = length(uv);

    return radialUv;
}

// Function 175
float hsluv_lToY(float L) {
    return L <= 8.0 ? L / 903.2962962962963 : pow((L + 16.0) / 116.0, 3.0);
}

// Function 176
vec2 map(vec2 uv)
{
    return saw(mobius(uv*2.0-1.0)*1.0*PI);
}

// Function 177
vec3 hsluv_intersectLineLine(vec3 line1x, vec3 line1y, vec3 line2x, vec3 line2y) {  return (line1y - line2y) / (line2x - line1x); }

// Function 178
vec3 Tonemap_ACESFitted(vec3 srgb)
{
    vec3 color = srgb * ACESInputMat;
   
   #if 1
    color = ToneTF2(color);
   #else
    color = RRTAndODTFit(color);
   #endif
    
    color = color * ACESOutputMat;

    return color;
}

// Function 179
float heightMap2(vec2 p){
    
    p /= 2.; // Extra scaling.
    
    float  h = 0., a = 1., sum = 0.; // Height, amplitude, sum.
    
    for(int i=0; i<4; i++){
    
        p = fract(p)*2.666; // Subdividing space.
        // Far more interesting, mutated subdivision, courtesy of Aiekick.
        //p = fract(p+sin(p.yx*9.)*0.025 + cos(p.yx*9.)*0.025)*3.; 
        // Another one with a time component.
        //p = fract(p + sin(p*9. + cos(p.yx*13. + iTime*2.))*0.02)*3.;
        
        vec2 w = .5 - abs(p - 1.5); // Prepare to make a square. Other shapes are also possible.
        float l = sqrt( max(16.0*w.x*w.y*(1.0-w.x)*(1.0-w.y), 0.))*.5+.5; // Edge shaping.
        w = smoothstep(0., .05, w); // Smooth edge stepping.
        h = max(h, w.x*w.y*a*l); // Producing the smooth edged, shaped square.
        //h += w.x*w.y*a*l;
        //h = max(h, abs(abs(w.x)-abs(w.y))*a*l);
        sum += a; // Keep a total... This could be hardcoded to save cycles.
        a *= .4; // Lower the amplitude for the next subdivision, just because it looks tidier.
        //if(i==2)a*=.75;
    }
    
    return h/sum;
    
}

// Function 180
vec2 getUV( Quadrilateral q, in vec2 p)
{
    vec2 e = q.b-q.a;
    vec2 f = q.d-q.a;
    vec2 g = q.a-q.b+q.c-q.d;
    vec2 h = p-q.a;
        
    float k2 = cross2( g, f );
    float k1 = cross2( e, f ) + cross2( h, g );
    float k0 = cross2( h, e );
    
    float w = k1*k1 - 4.0*k0*k2;
    
    if( w<0.0 ) return vec2(-1.0);

    w = sqrt( w );
    
    float v1 = (-k1 - w)/(2.0*k2);
    float v2 = (-k1 + w)/(2.0*k2);
    float u1 = (h.x - f.x*v1)/(e.x + g.x*v1);
    float u2 = (h.x - f.x*v2)/(e.x + g.x*v2);
    bool  b1 = v1>0.0 && v1<1.0 && u1>0.0 && u1<1.0;
    bool  b2 = v2>0.0 && v2<1.0 && u2>0.0 && u2<1.0;
    
    vec2 res = vec2(-1.0);

    if(  b1 && !b2 ) res = vec2( u1, v1 );
    if( !b1 &&  b2 ) res = vec2( u2, v2 );
    
    return res;
}

// Function 181
vec2 mod_uv(vec2 uv)
{
    return vec2(mod(uv.x, g_cw * (1. + g_cwb)), 
                mod(uv.y, g_ch * (1. + g_chb)));
}

// Function 182
vec2 GetNormalMap(in sampler2D s, in vec2 resolution, in vec2 uv)
{
	vec3 eps=vec3(1.0/resolution,0.0);
	vec2 norm = vec2(length(texture(s,uv+eps.xz)) - length(texture(s,uv-eps.xz)),
					 length(texture(s,uv+eps.zy)) - length(texture(s,uv-eps.zy)));
	
	return norm;
}

// Function 183
void sceneMap3D(vec3 pos, out float t, out int obj)
{
    // Initialize to back wall sdf
    t = plane(pos, vec4(0.0, 0.0, -1.0, 5.0));
    obj = IDBackWall;

    float t2;
    // Check left wall
    if((t2 = plane(pos, vec4(1.0, 0.0, 0.0, 5.0))) < t)
    {
        t = t2;
        obj = IDLeftWall;
    }
    // Check right wall
    if((t2 = plane(pos, vec4(-1.0, 0.0, 0.0, 5.0))) < t)
    {
        t = t2;
        obj = IDRightWall;
    }
    // Check top ceiling wall
    if((t2 = plane(pos, vec4(0.0, -1.0, 0.0, 7.5))) < t)
    {
        t = t2;
        obj = IDCeilingWall;
    }
    // Check floor wall
    if((t2 = plane(pos, vec4(0.0, 1.0, 0.0, 2.5))) < t)
    {
        t = t2;
        obj = IDFloorWall;
    }
    // Check for long cube
    if((t2 = box(rotateY(pos + vec3(0, 1, -2), -27.5 * 3.14159 / 180.0), vec3(1.5, 4, 1.5))) < t)
    {
        t = t2;
        obj = IDLongCube;
    }
    // Check for sphere
    if((t2 = sphere(pos, 1.3, vec3(-3.5 * sin(iTime), 0.6 + 2.0 * cos(iTime), 3.5 * cos(iTime)))) < t)
    {
        t = t2;
        obj = IDSphere;
    }
}

// Function 184
vec2 map( in vec3 pos ) {
	float d= smin( sdPlane(pos), sdCube(pos-vec3( 0.0, 0.5, 0.0), 0.5 ) );
	return vec2(d,1.);
}

// Function 185
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

// Function 186
vec3 bumpMap(sampler2D tex, in vec3 p, in vec3 n, float bumpfactor)
{
    //ok so I don't understand this technique yet.
    //I mean I can visualize getting the greyscale values from the texture 
    //at three points around the ref, based on the point and the normal.
    
    //usually if you want a gradient you
    //want to get the difference between the ref and points around it
    
    //we do this when we take the normal although in that cause we are 
    //getting distances. Here we imply distance by getting greyscale values.
    //the resulting gradient then cna be considered a normal because each of
    //it's components is a basis vector that is the slope between the 
    //the components of the ref and the point representing the change from that point
    //to a bit away.
    
    
    
    const vec3 eps = vec3(0.001, 0., 0.);//I use swizzling here, x is eps
    float ref = getGrey(triPlanar(tex, p, n));//reference value 
    
    vec3 grad = vec3(getGrey(triPlanar(tex, p - eps, n)) - ref,
                     //eps.yxz means 0.,0.001, 0. "swizzling
                     getGrey(triPlanar(tex, p - eps.yxz, n)) - ref,
                     getGrey(triPlanar(tex, p - eps.yzx, n)) - ref)/eps.xxx;
    
    //so grad is the normal...then he does:
    grad -= n*dot(grad, n);//takes the dot of the surface normal 
    //and the texture normal (the gradient), so percentage of how similar they are
    //multplies by the surface normal again so scaling it by that percentage
    //and subtracting that from the gradient.
    //so the result is only the portion of the gradient that is not part of n??
    
    // and returning the surface normal + that gradient portion plus a bump factor
    //why???
    return normalize(n + grad*bumpfactor);
}

// Function 187
float map_flame(vec3 pos, bool turb, bool bounding)
{  
    #ifdef show_flame
    if (!traceFlame && bounding)
        return 10.;
    
    ft = iTime*dns;
    
    pos-= flamePos;
    pos.x+= tubeLenght - 0.33;
    pos.y+= pos.x*pos.x*fg - fg;
 
    vec3 q = pos*fts;
    
    if (turb)
    {
        #ifdef flame_a_turb
        float n = 0.07*noise(q*0.6);
        q.xy = rotateVec(-q.xy, pos.z*n);
    	q.yz = rotateVec(-q.yz, pos.x*n);
    	q.zx = rotateVec(-q.zx, pos.y*n);
    	#endif
        
    	q*= vec3(1., 1.5, 1.);
        q+= vec3(ft, 0., 0.);
    	q.x+= 0.5*pos.x*noise(q + vec3(30., 40., 50. + ft));
    	q.y+= 0.5*pos.x*noise(q + vec3(10., 30. + ft, 20.));
    	q.z+= 0.5*pos.x*noise(q + vec3(20., 60. - ft, 40. - ft));
 
    	float dn = (dnf - dnx*pos.x);
    	pos.x+= dn*noise(q + vec3(12., 3.+ ft, 16. - ft)) - dn/2.;
    	pos.y+= dn*noise(q + vec3(14., 7., 20.)) - dn/2.;
    	pos.z+= dn*noise(q + vec3(8. + ft*0.3, 22., 9.)) - dn/2.;
    }        
    
    float df = length(pos.yz) + faf*pos.x + fd0;
    
    if (bounding)
    {
        df-= 0.5*smoothstep(-1.1, -4., pos.x);
        df = mix(df, sdCylinder(pos.yxz + vec3(0., 1., 0.), vec2(tubeDiameter - 0.01, tubeLenght*2.)), smoothstep(-1.3, -1.12, pos.x));   
    }
    else
        df = mix(df, sdCylinder(pos.yxz + vec3(0., 1., 0.), vec2(tubeDiameter + 0.01, tubeLenght*2.)), smoothstep(-1.5, -1.12, pos.x));   

    return df;
    #else
    return 10.;
    #endif
}

// Function 188
float map(in vec3 pos, float time ){
    return GetRayMarchHit(pos, time).distance;
}

// Function 189
float map(vec3 p)
{
    float d= length16(p)-0.5;
	p.xy *= .7;
	return mix(d,length(p)-1.5,0.5);
}

// Function 190
float map(vec3 p){
    float d = 999.;
    vec3 n = normalize(p);
    
    trn(p);
    trn(n);
    
    float h = trimap((p+iTime*.2)*.26, abs(n)).x;
    float r = 1. + h *.7;
    
    float dis = length(p)-r;
    float reg = length(p)-1.;
    
    d = mix(dis, reg, val(p));
    
    return d;
}

// Function 191
vec2 mapSeedt(vec2 uv)
{
    //uv = (uv + 1.)/2.;
    DecodeData(texelFetch( iChannel0, ivec2(uv*iResolution.xy),0), seedCoord, seedColor);
    //return LIGHT_DIST;
    return seedCoord/iResolution.xy-uv;
}

// Function 192
vec1 suv(vec3 a){return a.x+a.y+a.z;}

// Function 193
float mapVoxel(in vec3 p) {
    p=fract(p/minsize);
    return mix( mix(mix(v[0],v[1],p.x),mix(v[2],v[3],p.x),p.y),
                mix(mix(v[4],v[5],p.x),mix(v[6],v[7],p.x),p.y),  
            p.z) ;   
}

// Function 194
vec2 map(in vec3 pos)
{
    float t;
    float tmin = 9999.0;
    float shmin;
    float sh;
    float f;
    vec3  v = vec3(0.0);
    
#ifdef PERFORMANCE
    
    #define updateClosestSH(shfunc, center) \
    	v = pos - center; \
        sh = shfunc(normalize(v).xzy); \
        t = length(v) - abs(sh); \
        f = step(t, tmin); \
        shmin = sh * f + shmin * (1.0 - f); \
    	tmin = min(tmin, t);
    
	updateClosestSH(y00 , vec3( 0.0*HSEP,  1.0*VSEP, 0.0));
    updateClosestSH(y11_, vec3(-1.0*HSEP,  0.0*VSEP, 0.0));
    updateClosestSH(y10 , vec3( 0.0*HSEP,  0.0*VSEP, 0.0));
    updateClosestSH(y11 , vec3( 1.0*HSEP,  0.0*VSEP, 0.0));
    updateClosestSH(y22_, vec3(-2.0*HSEP, -1.0*VSEP, 0.0));
    updateClosestSH(y21_, vec3(-1.0*HSEP, -1.0*VSEP, 0.0));
    updateClosestSH(y20 , vec3( 0.0*HSEP, -1.0*VSEP, 0.0));
    updateClosestSH(y21 , vec3( 1.0*HSEP, -1.0*VSEP, 0.0));
    updateClosestSH(y22 , vec3( 2.0*HSEP, -1.0*VSEP, 0.0));
    
#else
    
    #define updateClosestSH(l, m, center) \
    	v = pos - center; \
        sh = SH(normalize(v).xzy, l, m); \
        t = length(v) - abs(sh); \
        f = step(t, tmin); \
        shmin = sh * f + shmin * (1.0 - f); \
    	tmin = min(tmin, t);
    
	updateClosestSH(0, 0, vec3( 0.0*HSEP,  1.0*VSEP, 0.0));
    updateClosestSH(1,-1, vec3(-1.0*HSEP,  0.0*VSEP, 0.0));
    updateClosestSH(1, 0, vec3( 0.0*HSEP,  0.0*VSEP, 0.0));
    updateClosestSH(1, 1, vec3( 1.0*HSEP,  0.0*VSEP, 0.0));
    updateClosestSH(2,-2, vec3(-2.0*HSEP, -1.0*VSEP, 0.0));
    updateClosestSH(2,-1, vec3(-1.0*HSEP, -1.0*VSEP, 0.0));
    updateClosestSH(2, 0, vec3( 0.0*HSEP, -1.0*VSEP, 0.0));
    updateClosestSH(2, 1, vec3( 1.0*HSEP, -1.0*VSEP, 0.0));
    updateClosestSH(2, 2, vec3( 2.0*HSEP, -1.0*VSEP, 0.0));
    
#endif
    
    return vec2(tmin, shmin);
}

// Function 195
vec2 map(vec3 pos, bool btext)
{
    vec2 res = opU(vec2(map_floor (pos, btext)                                                            , FLOOR_OBJ),
                   vec2(map_tablet(pos, tabPos, vec3(tabWidth, tabHeight, 0.1), 5., tabletRounding, btext), TABLET_OBJ));
    #ifdef spheres
    res = opU(res, vec2(map_spheres(pos, vec3(.0, 0.45, 2.9), 1.7, 0.45)                                        , SPHERES_OBJ));
    #endif
    return res;
}

// Function 196
float map(vec3 p, inout vec4 orbitTrap)
{
    const float s = 1.0;//0.97;
    const float horizontalWrap = sqrt(s*2.0)/2.0;
    
	float scale = 1.0;

	orbitTrap = vec4(1000.0); 
    
    for(int i=0; i<9; i++)
	{
        p.xz /= horizontalWrap;
        vec3 pOffset = (0.5*p+0.5);

        vec3 pOffsetWrap = 2.0*fract(pOffset);
        
        p = -1.0 + pOffsetWrap;
        p.xz *= horizontalWrap;
        
		float r2 = dot(p,p);
		
        if(i < 2)
        {
	        orbitTrap.z = min(orbitTrap.z, vec4(abs(p),r2).z);
        }
        if(i > 2)
        {
            orbitTrap.xyw = min(orbitTrap.xyw, vec4(abs(p),r2).xyw);
        }
        
		float k = s/r2;
		p     *= k;
		scale *= k;
	}
	
	float fractal = 0.33*abs(p.y)/scale;
    return fractal;
}

// Function 197
float map_gled(vec3 pos)
{
    return sdRoundBox(pos - vec3(-supportSize.x + 0.1, 2.*supportSize.y - 0.1, -supportSize.z - 0.01), vec3(0.038, 0.009, 0.009), 0.004);
}

// Function 198
float map(vec3 p){
    
    
    float d =  columns(p); // Repeat columns.
    
    float fl = p.y + 2.5; // Floor.

    p = abs(p);
    
    d = sminP(d, -(p.y - 2.5 - d*.75), 1.5); // Add a smooth ceiling.
    
    d = min(d, -(p.x - 5.85)); // Add the Walls.
    
    d = sminP(d, fl, .25); // Smoothly combine the floor.
     
    return d*.75;
}

// Function 199
vec3 heatColorMap(float t)
{
    t *= 4.;
    return clamp(vec3(min(t-1.5, 4.5-t), 
                      min(t-0.5, 3.5-t), 
                      min(t+0.5, 2.5-t)), 
                 0., 1.);
}

// Function 200
float map(vec3 p)
{
    float radius = 0.25;
    vec3 q = fract(p) * 1.5 - 0.5;
    
    return sphere(q, radius);
}

// Function 201
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 ro, in vec3 rd )
{
    vec3 color = vec3(0); 
    
	#define SSAMPLES 1
	for (int i = 0; i < SSAMPLES; i++) {
		color += rm(ro, rd, iTime+float(i)); 
	}
	color /= float(SSAMPLES);
	    
    // Output to cubemap
    color = mix(color, texture(iChannel0, rd).xyz, 0.99); 
    fragColor = vec4(color,1.0);
}

// Function 202
vec2 map(vec3 pos)
{
    vec2 res;

    float tube = map_tube(pos);
    float ffloor = map_floor(pos);
    float wall1 = map_wall_1(pos);
    float wall2 = map_wall_2(pos);
    float flame = map_flame(pos, false, true);
    
    res = vec2(tube, TUBE_OBJ);
    res = opU(vec2(ffloor, FLOOR_OBJ), res);
    res = opU(vec2(wall1, WALL_1_OBJ), res);
    res = opU(vec2(wall2, WALL_2_OBJ), res);
    //res = opU(vec2(flame, TUBE_OBJ), res);
    res = opU(vec2(flame, FLAME_OBJ), res);

    return res;
}

// Function 203
vec3 NOISE_volumetricRoughnessMap(vec3 p, float rayLen)
{
    vec4 sliderVal = vec4(0.5,0.85,0,0.5);
    ROUGHNESS_MAP_UV_SCALE *= 0.1*pow(10.,2.0*sliderVal[0]);
    
    float f = iTime;
    const mat3 R1  = mat3(0.500, 0.000, -.866,
	                     0.000, 1.000, 0.000,
                          .866, 0.000, 0.500);
    const mat3 R2  = mat3(1.000, 0.000, 0.000,
	                      0.000, 0.500, -.866,
                          0.000,  .866, 0.500);
    const mat3 R = R1*R2;
    p *= ROUGHNESS_MAP_UV_SCALE;
    p = R1*p;
    vec4 v1 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021;
    vec4 v2 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021+1.204*v1.xyz;
    vec4 v3 = NOISE_trilinearWithDerivative(p);
    p = R1*p*2.021+0.704*v2.xyz;
    vec4 v4 = NOISE_trilinearWithDerivative(p);
    
    return (v1
	      +0.5*(v2+0.25)
	      +0.4*(v3+0.25)
	      +0.6*(v4+0.25)).yzw;
}

// Function 204
float map_f_hor(vec3 pos, vec2 delta, float n)
{
    return length(vec2(mod(pos.y + delta.x, fe) - fe*0.6, pos.z + delta.y + fr*sin((pos.x + fe*2. + fe*floor(pos.y/fe))/fe*pi))) - fr*fds*0.86;
}

// Function 205
vec3 GetUVW(vec3 axis,vec3 dir){
    return 1./dot(axis,dir)*dir-axis;
}

// Function 206
vec3 LightSpaceToLightUV (vec3 lightPosition)
{
    // calculate the uv from the light position. X is flipped
    vec3 uv;
    uv.xy = lightPosition.xz;
    uv.x *= -1.0;
        
    // apply x axis modulus to make the light repeat
	uv.y = mod(uv.y + directionalLightModulus * 0.5, directionalLightModulus) - directionalLightModulus * 0.5;   

    // apply scaling of uv over distance to fake projection
	uv.xy /= (1.0 + lightPosition.y * directionalLightUVDistanceScale);   
    
    // calculate the instance index
    uv.z = floor((lightPosition.z + directionalLightModulus * 0.5) / directionalLightModulus);
    
    // return adjusted uv coordinates
    return uv;
}

// Function 207
float Remap(float v, float omin, float omax, float nmin, float nmax)
{
	return nmin+max(0.0, (v-omin))/(omax-omin)*(nmax-nmin);
}

// Function 208
vec3 boxmap( sampler2D s, vec3 p, vec3 n ){
    
    mat3 t = mat3(
        texture( s, p.yz ).rgb,
        texture( s, p.zx ).rgb,
        texture( s, p.xy ).rgb
    );
    n=abs(n);
    return t*n / dot(n,vec3(1));
}

// Function 209
float mapShell( in vec3 p, out vec4 matInfo ) 
{
    
    const float sc = 1.0/1.0;
    p -= vec3(0.05,0.12,-0.09);    

    p *= sc;

    vec3 q = mat3(-0.6333234236, -0.7332753384, 0.2474039592,
                   0.7738444477, -0.6034162289, 0.1924931824,
                   0.0081370606,  0.3133626215, 0.9495986813) * p;

    const float b = 0.1759;
    
    float r = length( q.xy );
    float t = atan( q.y, q.x );
 
    // https://swiftcoder.wordpress.com/2010/06/21/logarithmic-spiral-distance-field/
    float n = (log(r)/b - t)/(2.0*pi);

    float nm = (log(0.11)/b-t)/(2.0*pi);

    n = min(n,nm);
    
    float ni = floor( n );
    
    float r1 = exp( b * (t + 2.0*pi*ni));
    float r2 = r1 * 3.019863;
    
    //-------

    float h1 = q.z + 1.5*r1 - 0.5; float d1 = sqrt((r1-r)*(r1-r)+h1*h1) - r1;
    float h2 = q.z + 1.5*r2 - 0.5; float d2 = sqrt((r2-r)*(r2-r)+h2*h2) - r2;
    
    float d, dx, dy;
    if( d1<d2 ) { d = d1; dx=r1-r; dy=h1; }
    else        { d = d2; dx=r2-r; dy=h2; }


    float di = textureLod( iChannel2, vec2(t+r,0.5), 0. ).x;
    d += 0.002*di;
    
    matInfo = vec4(dx,dy,r/0.4,t/pi);

    vec3 s = q;
    q = q - vec3(0.34,-0.1,0.03);
    q.xy = mat2(0.8,0.6,-0.6,0.8)*q.xy;
    d = smin( d, sdTorus( q, vec2(0.28,0.05) ), 0.06);
    d = smax( d, -sdEllipsoid(q,vec3(0.0,0.0,0.0),vec3(0.24,0.36,0.24) ), 0.03 );

    d = smax( d, -sdEllipsoid(s,vec3(0.52,-0.0,0.0),vec3(0.42,0.23,0.5) ), 0.05 );
    
    return d/sc;
}

// Function 210
vec2 mapRM(vec3 p) {
    vec2 d = vec2(-1.0, -1.0);
    d = vec2(mapTerrain(p-vec3(0.0, FLOOR_LEVEL, 0.0), FLOOR_TEXTURE_AMP), TYPE_FLOOR);
    d = opU(d, vec2(mapSand(p-vec3(0.0, FLOOR_LEVEL, 0.0)), TYPE_SAND));
    d = opU(d, vec2(mapWater(p-vec3(0.0, WATER_LEVEL, 0.0)), TYPE_WATER));
    //d = opU(d, vec2(sdBox(p-BOATPOS, vec3(1.0, 1.0, 1.0)), TYPE_BOAT));
    return d;
}

// Function 211
float map_water(vec3 pos)
{
    float h = (pos.y/tubeRadius + 1.)/2.;
    h+= wavesLev*(noise(pos*wavesFreq + iTime*vec3(0., 0.7, 0.3)) - 0.5);
    return h - waterLevel;   
}

// Function 212
float map(vec3 p){
    for( int i = 0; i<8; ++i){
        float t = iTime*0.2;
        p.xz =rotate(p.xz,t);
        p.xy =rotate(p.xy,t*1.89);
        p.xz = abs(p.xz);
        p.xz-=.5;
	}
	return dot(sign(p),p)/5.;
}

// Function 213
vec3 rgb2yuv(vec3 rgb) 
{
    vec3 yuv;
	yuv.x = rgb.r *  0.299 + rgb.g *  0.587 + rgb.b *  0.114;
	yuv.y = rgb.r * -0.169 + rgb.g * -0.331 + rgb.b *  0.5;
	yuv.z = rgb.r *  0.5   + rgb.g * -0.419 + rgb.b * -0.081;
	return yuv; 
}

// Function 214
vec4 nmapu(vec4 x){ return x*.5+.5; }

// Function 215
float map(vec3 p)
{
    float d1 = sdSphere(p, vec3(-1, sin(iTime)*-1., 0), 1.0);
    float d2 = sdBox(p, vec3(0.5));
    return smin(d1, d2, .4);
}

// Function 216
float map(vec3 p){
    
    float n = (.5-cellTile(p))*1.5;
    return p.y + dot(sin(p/2. + cos(p.yzx/2. + 3.14159/2.)), vec3(.5)) + n;
 
}

// Function 217
float map(vec3 p)
{	
    float t = texture(iChannel1, vec2(0.0, 0.)).x;
    float r = SPHERE.w + displacement(p,1.)*t*15.;
	float d = length(p)-r;
    return d;
}

// Function 218
void mainCubemap( out vec4 O, vec2 U, vec3 C, vec3 D )
{
  //O = vec4(.5+.5*D,0); U = U/1024. - 1./vec2(4,8); O -= .01/dot(U,U); return;
  //int f = faceID(D); O = vec4(f&1,f&2,f&4,0); return;
 
    U *= 2./iResolution.xy;
    int v = ( int(U.x)+2*int(U.y) + 4*faceID(D) + iFrame*24 ) % 264;

    O = vec4( equal( ivec4( 255.* texture(iChannel0, fract(U))),
                     ivec4( v ) ) );
}

// Function 219
Model map( vec3 p ){
    mat3 m = modelRotation();
    p *= m;  
    #ifndef LOOP
    	pR(p.xz, time * PI/16.);
    #endif
    Model model = geodesicModel(p);
    return model;
}

// Function 220
float map_ver_small(vec3 pos, vec2 delta, float n)
{    
    float fx = 145.*random(19.36*floor(pos.x/fe));
    float ad = 1. + ttwd*vsf;            
    
    float angle = ad*twf*pos.y;
    vec2 d1 = rotateVec(vec2(fr*fd, fr*fd), angle);
    vec2 d2 = d1.yx*vec2(1., -1);
    return min(min(min(map_f_ver(pos, d1 + delta, n + 1.), map_f_ver(pos, d2 + delta, n + 2.)), map_f_ver(pos, -d2 + delta, n + 3.)), map_f_ver(pos, -d1 + delta, n + 4.)); 
}

// Function 221
float map(vec3 p)
{
    vec4 tex = texture(iChannel2,p.xy / uvSize);
    float scale = scale;

	float prim =  sphere(p,5.0+scale); 
	float prim3 = curveSphereToPlane(p,prim+1.0);
    
    float dist  = displacement(p / uvSize, prim);
    float dist3 = prim3;
    
    float time = mod(iTime,200.0) * .1;
    dist = smin(dist,dist3, time);
    
    
    return dist * length(tex);
}

// Function 222
vec3 lchToHsluv(vec3 tuple) {  tuple.g /= hsluv_maxChromaForLH(tuple.r, tuple.b) * .01;  return tuple.bgr; }

// Function 223
vec3 mapRMWaterNormal(vec3 pt, float e) {
    vec3 normal;
    normal.y = sdPlane(pt)+waterDetails(pt, iTime);    
    normal.x = (sdPlane(pt)+waterDetails(vec3(pt.x+e,pt.y,pt.z), iTime)) - normal.y;
    normal.z = (sdPlane(pt)+waterDetails(vec3(pt.x,pt.y,pt.z+e), iTime)) - normal.y;
    normal.y = e;
    return normalize(normal);
}

// Function 224
vec3 ToneMappingSRGB( vec3 col )
{
	return pow(col,vec3(1./2.2));
}

// Function 225
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{	    
    int faceID = CubeFaceCoords(rayDir);
    if(faceID!=0 && faceID!=5) discard;
    
    vec2 uv = fract(fragCoord/1024.*vec2(1, -1) );  
    if(faceID==0) mainImage_A(fragColor,uv*cubemapRes); 
    else if(faceID==5)  mainImage_C(fragColor,uv*cubemapRes);
     
}

// Function 226
vec2 map_to_origin(vec2 a, vec2 p){
    if(a == vec2(0.0)){
        return p;
    } else {
        return comp_div(comp_mul(a, conj(a)-conj(p)), comp_mul(conj(a), comp_mul(a, conj(p)) - vec2(1.0, 0.0)));
    }
}

// Function 227
float map(vec3 p)
{  
    float s = smoothstep( -0.7, 0.7, sin(0.5*T));
    float a = mix(0.5, 2.0, s);
    float t = 1.0*T;
    
	float s1 = sphere(p +a*vec3(cos(t*1.1),cos(t*1.3),cos(t*1.7)), 1.0);
    float s2 = sphere(p +a*vec3(cos(t*0.7),cos(t*1.9),cos(t*2.3)), 1.2);
    float s3 = sphere(p +a*vec3(cos(t*0.3),cos(t*2.9),sin(t*1.1)), 1.5);
    float s4 = sphere(p +a*vec3(sin(t*1.3),sin(t*1.7),sin(t*0.7)), 0.4);
    float s5 = sphere(p +a*vec3(sin(t*2.3),sin(t*1.9),sin(t*2.9)), 1.0);
    
    return blob5(s1, s2, s3, s4, s5);
}

// Function 228
vec4 sampleNormalMap(vec3 N) 
{
    float u = 1.-(atan(N.x, N.z)+PI)/(2.*PI);
	float v = (acos(N.y)/PI);	// 1.- becouse the coordinates origin is in the bottom-left, but I backed from top-left
    return texture(iChannel0, vec2(u, v));   
}

// Function 229
vec4 map(vec3 p){
  vec3 color = black;
  float t = 1e20;
  int n = foldH3Count(p);
  float t1 = max(dot(p - pbc5 * 2.2, nca5), dot(p - pbc5 * 2.0, -nca5));
  updateDist(color, t, getRGB(float(n) / 15.0, 1.0, 1.0), t1, 0);
  vec3 guide = getP5(0.0, 0.9, 0.1) * 2.1;
  vec3 v = normalize(cross(pab5 - pbc5, nca5));
  updateDist(color, t, silver, dot(p - guide, v), 1);
  updateDist(color, t, gold, max(dot(p - getP5(0.0, 0.1, 0.9) * 2.1, pbc5 - pca5 + nca5 * 0.1), dot(p - getP5(0.0, 0.05, 0.95) * 2.1, pbc5 - pca5 - nca5 * 0.1)), 0);
  updateDist(color, t, skyblue, sphere(p, 1.0), 0);
  return vec4(color, t);
}

// Function 230
vec2 map1sd(vec2 z)
{
#if 1
  float phi,r;
  float a = z.x;
  float b = z.y;
  
  if (a*a > b*b) {
    r = a;
    phi = (PI/4.)*(b/a);
  } else {
    r = b;
    phi = (PI/2.0) - (PI/4.0)*(a/(b+epsilon));
  }
  return vec2( r*cos(phi), r*sin(phi) );
#else
  float u,v,m,t;
    
  if (abs(z.x) > abs(z.y)) {
    m = z.x;
	t = (PI/4.0)*z.y/z.x;
    v = sin(t);
    u = sqrt(1.0-v*v);     // or cos(t)
  }
  else {
    m = z.y;
    t = (PI/4.0)*z.y*z.x/(z.x*z.x+epsilon);
	//t = (PI/4.0)*z.x/z.y;
    //t = z.y != 0.0 ? t : 0.0;
    u = sin(t);
    v = sqrt(1.0-u*u);    // or cos(t)
  }
    
  return vec2(m*u, m*v);

#endif
}

// Function 231
float map(vec4 uv, mat4 q, vec4 l) {
    return sin(2.0 * pow(dot(q * uv + l, uv), 6.0 / 11.0));
}

// Function 232
vec2 getPolarUV(in vec2 uv)
{
    float angle = atan(uv.y, uv.x);
    angle += PI;
    angle /= (2.*PI);
    
    float dist = distance(vec2(0.), uv);
    
    return vec2(angle, dist);
}

// Function 233
float map( vec3 q ) {                                 // 3D model of Regent Street
    q += vec3(-182,2,2);
    float l = length(q.xz),
    t = min( c1 = l-160.,                             // right facade
             c0 = 184.-l );                           // left facade
    t = min( t, max( l-170., 16.-q.y ));              // top of right facade
 
    t = min(t, max( length(q.xz+vec2(174.1,40))-1.4 , abs(q.y)-1.8 ) ); // bus
    t = min(t, max( length(q.xz+vec2(165,40))-.9 , abs(q.y)-1.45 ) );   // car
    t = min(t, max( length(q.xz+vec2(169.5,25))-.24 , abs(q.y)-1. ) );   // pole
    t = min( t, s = q.y);                             // floor
    t = max( t, q.z -2. );                            // front plane
    return t;
}

// Function 234
vec4 MapPlanet(vec3 p)
{
  vec3 moonPos = p-GetMoonPosition(p);
  vec2 mapPos = PosToSphere(moonPos);
  float heightMap = -fastFBM((moonPos*0.5)*.4);
  float moon = sdSphere(moonPos, 40.+heightMap);
  GetPlanetRotation(p);  
  mapPos = PosToSphere(p);
  heightMap = ((GetTerrainHeight(8.*p)*0.35))*textureLod(iChannel3, 2.*mapPos, log2(mapPos.y*2.)).z*1.5;
  return vec4(min(moon, sdSphere(p, 70.-min(2., (1.-heightMap)))), heightMap, moon, 0.);
}

// Function 235
void tonemap(inout vec3 color)
{
    #if TONEMAP_TYPE == LINEAR_TONEMAP
    color *= vec3(TONEMAP_EXPOSURE);
    #endif
    #if TONEMAP_TYPE == EXPONENTIAL_TONEMAP
    color = 1.0 - exp2(-color * TONEMAP_EXPOSURE);
    #endif
    #if TONEMAP_TYPE == REINHARD_TONEMAP
    color *= TONEMAP_EXPOSURE;
    color = color / (1.0 + color);
    #endif
    #if TONEMAP_TYPE == REINHARD2_TONEMAP
    color *= TONEMAP_EXPOSURE;
    color = (color * (1.0 + color / (LDR_WHITE * LDR_WHITE))) / (1.0 + color);
    #endif
    #if TONEMAP_TYPE == FILMIC_HEJL2015
    color *= TONEMAP_EXPOSURE;
    color = linearTo_sRGB(TonemapFilmic_Hejl2015(TONEMAP_EXPOSURE_BIAS * color));
    #endif
    #if TONEMAP_TYPE == FILMIC_TONEMAP_UNCHARTED2    
    color *= TONEMAP_EXPOSURE;
    vec3 tonemapedColor = Uncharted2Tonemap(TONEMAP_EXPOSURE_BIAS * color);
    vec3 whiteScale = 1.0 / Uncharted2Tonemap(vec3(W));
    color = tonemapedColor * whiteScale;
    #endif
    #if TONEMAP_TYPE == FILMIC_TONEMAP_ACES
    color *= TONEMAP_EXPOSURE;
    vec3 tonemapedColor = TonemapACESFilm(TONEMAP_EXPOSURE_BIAS * color);
    vec3 whiteScale = 1.0 / TonemapACESFilm(vec3(W));
    color = tonemapedColor * whiteScale;
    #endif
    #if TONEMAP_TYPE == FILMIC_TONEMAP_ALU
    color *= TONEMAP_EXPOSURE;
    color = TonemapFilmicALU(color);
    #endif
}

// Function 236
vec2 mapMat(vec3 p){
    vec3 q = p;
    p = vec3(mod(p.x, 5.0) - 2.5, p.y, mod(p.z, 5.0) - 2.5);
    p -= vec3(0.0, 0.0, 0.0);
    float qpi = 3.141592 / 4.0;
    float sub = 10000.0;
    for(float i = 0.0; i < 8.0; i++){
        float x = 0.2 * cos(i * qpi);
        float z = 0.2 * sin(i * qpi);
        vec3 transp = p - vec3(x, 0.0, z);
        vec3 a = vec3(x, 1.2, z);
        vec3 b = vec3(x, -1.2, z);
        sub = min(sub, capsule(transp, a, b, 0.1));
    }
    float ttorus = torus(p - vec3(0.0, -1.5, 0.0), vec2(0.22));
    float btorus = torus(p - vec3(0.0, 1.5, 0.0), vec2(0.22));
    float u = min(btorus, ttorus);
    vec2 column = vec2(min(u, max(-sub, length(p.xz) - 0.35)), 2.0);
    vec2 flo = vec2(q.y + 1.5, 1.0);
    vec2 roof = vec2(-q.y + 1.5, 1.0);
    return vecMin(column, vecMin(flo, roof));
}

// Function 237
traceData finalmap(vec3 point){
    
        //shape finding functions:
        dSphere = sphere(point*iSphereScale, rad);

        dCone.l = ConeD(point,0.4,0.2);

        //dTexMarch.p = texture2D(sTD2DInputs[0],fract(point.xy*0.1)).rgb;
        //dTexMarch.l=length(dTexMarch.p);

        traceData fout;
        //fout.l= pSmoothMin(dCone.l,dSphere.l,0.5);
        fout.l= dSphere.l;
        

        float least = min(dCone.l,dSphere.l);
       // return fout;
       //return pSmoothMin(dCone.l,dSphere.l,0.1);
       return fout;
}

// Function 238
vec3 rgbToHpluv(vec3 tuple) {
    return lchToHpluv(rgbToLch(tuple));
}

// Function 239
vec3 ACESFilmicToneMapping(vec3 col){
	vec3 curr = Uncharted2Tonemap(col);
    const float ExposureBias = 2.0;
	curr *= ExposureBias;
    curr /= Uncharted2Tonemap(vec3(W));
    return LinearToGamma(curr);
}

// Function 240
float mapWDists(vec3 p, out vec3 dists) {
  mat3 r1 = rot3XY(u_time, 3), r2 = rot3XY(u_time * .48597, 1);
  // r1 = rot3XY(u_time * .0, 3), r2 = rot3XY(u_time * .763 * .0, 2);
  vec4 boxEdges = sldBoxEdges((p - vec3(0.4, 1.0 , 0)) * r2 * r2 * r1, vec3(.3)) - .01;
  vec4 box = sldBox(p * r2 * r1, vec3(.3)) - .01;
  vec4 tri = sdTriangle2((p - vec3(-.4, -.9, 0)) * r2 * r1, .3) - .01;
  vec4 tri2 = sdTriangle1((p - vec3(.4, -1.4, 0)) * r2 * r1, .3) - .01;
  vec4 tet;
  tet.x = sdTetrahedron((p - vec3(.4, 1.9, 0)) * r2 * r1, .3, tet.yzw) - .01;

  vec4 tri3;
  tri3.x = sdTriangleGen((p - vec3(.5, 1.9, 0)) * r2 * r1, .3, tri3.yzw) - .01;

  vec4 m = minWith(boxEdges, vec4(999.));
  m = minWith(m, tri2);

  if (HIGH_RESOURCES == 1) {
    m = minWith(m, box);
    m = minWith(m, tri);
  }

  dists = m.yzw;
  return m.x;
}

// Function 241
void mainCubemap( out vec4 fragColour, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    fragColour = textureLod( iChannel0, rayDir, 0. ); // this needs NEAREST filter on the texture
	if ( iFrame == 0 ) fragColour = vec4(0);

    // wait for texture to load (I know the top of the cubemap should not be black)
    if ( textureLod( iChannel1, vec3(0,1,0), 0. ).r == 0. ) return;
    
    // early-out once we've got a good enough result
    if ( fragColour.a > 100.0 ) discard;
    
    const int n = 16;
    for ( int i = 0; i < n; i++ )
    {
        vec3 ray = HemisphereRand(rayDir,uint(i+n*iFrame)+quasi2.y*uint(fragCoord.x)+quasi2.x*uint(fragCoord.y));

        fragColour.rgb += LDRtoHDR(textureLod( iChannel1, ray, 0. ).rgb);
        fragColour.a += 1.;
    }
}

// Function 242
vec2 uv_bipolar(vec2 domain, vec2 northPole, vec2 southPole, float fins, float log_factor, vec2 coord){
   vec2 help_uv = mobius(domain, northPole, southPole);
   return uv_polar_logarithmic(help_uv, vec2(0.5), fins, log_factor, coord);
}

// Function 243
float map(vec3 p) {
  float r = iMouse.z > 0.0 ? iMouse.x / 100.0 : iTime * 0.9;
  p.xz = mirror(p.xz, 10.);
  p.xz = rotate2D(p.xz, r);
  float d = sdBox(p, vec3(1));
  d = min(d, sdBox(p, vec3(0.1, 0.1, 3)));
  d = min(d, sdBox(p, vec3(2.5, 0.3, 0.1)));
  return d;
}

// Function 244
vec2 ScreenUvToWindowPixel(vec2 vUv)
{
	#ifdef LOW_RESOLUTION
		vUv = ((vUv - kWindowMin) / kWindowRange);
	#endif
	return vUv * kWindowResolution;
}

// Function 245
float map(vec4 p)
{
    float box = sdTesseract(p, vec4(0.5));
    return box;
}

// Function 246
vec2 MyCubeMap_faceToUV(MyCubeMap_FaceInfo info)
{
    const float freq = 2.5;
    info.id   += (info.id>=4.99 && info.uv.y>0.5)?1.:0.;
#if SEAMLESS
    const float eps = 0.003;
    bool bHalf = (info.id>5.99);
    if(bHalf)
    {
        info.uv.y -= 0.5;
		info.uv.y = min(info.uv.y,0.5-eps);
    }
    info.uv = min(info.uv,1.-eps);
    info.uv = max(info.uv,eps);
#else
    info.uv.y -= (info.id>5.99)?0.5:0.;
#endif    
    
    vec2 huv = vec2(info.uv.x+info.id,info.uv.y);
    huv.y = huv.y/freq+floor(huv.x/freq)/freq;
    return vec2(fract(huv.x/freq),huv.y);
}

// Function 247
vec3 MapDisk(vec2 p) {
    float angle = 6.2831853072 * p.y;
    return vec3(cos(angle), sin(angle), p.x);
}

// Function 248
float remap(float x, float low1, float high1, float low2, float high2){
	return low2 + (x - low1) * (high2 - low2) / (high1 - low1);
}

// Function 249
vec3 sampleEnvMap(vec3 rd, float lod)
{
    vec2 uv = vec2(atan(rd.z,rd.x),acos(rd.y));
    uv = fract(uv/vec2(2.0*PI,PI));
    
    vec3 col = vec3(0.,0.05*cos(uv.x)+0.05, .1*sin(uv.y)+.1)*1.;
    
    float r = (1.-pow(lod,.5))*1000.+5.;
    col += vec3(1.)* clamp( pow(1.-roundBox(uv-vec2(.5), vec2(.05,.05),.01),r), 0., 1.);
    col += vec3(1.)* clamp( pow(1.-roundBox(uv-vec2(.67,.5), vec2(.05,.05),.01),r), 0., 1.);
    col += vec3(1.)* clamp( pow(1.-roundBox(uv-vec2(.67,.67), vec2(.05,.05),.01),r), 0., 1.);
    col += vec3(1.)* clamp( pow(1.-roundBox(uv-vec2(.5,.67), vec2(.05,.05),.01),r), 0., 1.);
    col += vec3(1.,.5,.1)*2. * clamp( pow(1.-roundBox(uv-vec2(.3,.7), vec2(.01,.01),.2),r), 0., 1.);
    
    return min(col*(1.-lod*.8),vec3(1.));
}

// Function 250
vec2 getuv(in vec2 uv, float l)
{
    vec3 rd = normalize(vec3(uv, 0.4));
    vec2 _uv = vec2(rd.x / abs(rd.y) * l, rd.z / abs(rd.y) * l);
    return _uv;
}

// Function 251
float map_doorbell(vec3 pos)
{
    pos.y = mod(pos.y, floor_height);
    vec3 pos2 = pos;
    if (pos.x>-staircase_length/2. + 1.)
    {
        pos2.z = abs(pos2.z);
        pos2-= vec3(-staircase_length/2. + floor_width/2. + door_width/2. + doorbell_rpos, door_height*doorbell_height, staircase_width/2.);
    }
    else
    {
        pos2.z = abs(pos2.z);
        pos2+= vec3(staircase_length/2., -door_height*doorbell_height, -staircase_width/4. - door_width/2. - doorbell_rpos);
        pos2.xz = pos2.zx;
        pos2.z = -pos2.z;
    }

    float doorbellv = sdRoundBox(pos2, doorbell_plate.xyz, doorbell_plate.w);
    doorbellv = min(doorbellv, length(pos2 - vec3(0., doorbell_button.w, doorbell_srad - doorbell_sdep - doorbell_plate.z - doorbell_plate.w)) - doorbell_srad);
    doorbellv = min(doorbellv, sdCylinder(pos2.xzy - vec3(0., 0., doorbell_button.w), vec2(doorbell_button.x, doorbell_button.y), 0., doorbell_button.z));
    
    doorbellv = max(doorbellv, -sdRoundBox(pos2 - vec3(0., doorbell_nplate_ypos, -doorbell_plate.z - doorbell_plate.w), doorbell_nplate.xyz, doorbell_nplate.w));
    
    
    return doorbellv;
}

// Function 252
void texture_uv( inout vec4 fragColor, in vec2 uv)
{
    fragColor = texture(iChannel0, uv);
}

// Function 253
float Map(vec3 point, out vec3 motion)
{
	/* Distance function */
    #ifdef WARP
    pR(point.xy, snoise(point.z * 0.25) * 4.0 + point.z * 0.125);
    #endif
    motion = vec3(0.0);
    float result = 0.5 - abs(point.y);
    #ifdef ANIMATE
    point.z += iTime;
    #endif
   	point.z = fract(point.z) - 0.5;
    point.x = abs(point.x) - 1.0;
    vec3 q = abs(point) - vec3(0.2, 0.5, 0.2);
    float r = max(q.x, max(q.y, q.z));
    if(r<result)
    {
        result=r;
    	#ifdef ANIMATE
        motion.z = 1.0;
    	#endif
    }
    return result;
}

// Function 254
vec3 hsluvToRgb(vec3 tuple) {
    return lchToRgb(hsluvToLch(tuple));
}

// Function 255
vec3 Uncharted2Tonemap(vec3 x)
{
   return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// Function 256
float Tonemap_Uchimura(float x, float P, float a, float m, float l, float c, float b) {
    // Uchimura 2017, "HDR theory and practice"
    // Math: https://www.desmos.com/calculator/gslcdxvipg
    // Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
    float l0 = ((P - m) * l) / a;
    float L0 = m - m / a;
    float L1 = m + (1.0 - m) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    float w0 = 1.0 - smoothstep(0.0, m, x);
    float w2 = step(m + l0, x);
    float w1 = 1.0 - w0 - w2;

    float T = m * pow(x / m, c) + b;
    float S = P - (P - S1) * exp(CP * (x - S0));
    float L = m + a * (x - m);

    return T * w0 + L * w1 + S * w2;
}

// Function 257
vec3 TriplanarMapping (sampler2D tex, vec3 normal, vec3 position) {
	vec3 normalBlend = BlendNormal(normal);
	vec3 xColor = texture(tex, position.yz).rgb;
	vec3 yColor = texture(tex, position.xz).rgb;
	vec3 zColor = texture(tex, position.xy).rgb;

  	return (xColor * normalBlend.x + yColor * normalBlend.y + zColor * normalBlend.z);
}

// Function 258
vec3 cubemapRayDir(in vec2 fragCoord, vec2 bufferSize) 
{     
    bufferSize.y = min(bufferSize.y, bufferSize.x*0.66667 + 4.0);
    
    float ts = (bufferSize.y - 2.0) * 0.5;
    
    fragCoord = min(fragCoord, 
                    vec2(ts*3.0 - 1.0, 2.0*ts + 1.0));
    
    vec2 tc = vec2(fragCoord.x / ts, 
                   fragCoord.y*2.0 / bufferSize.y); 
    
    vec2 ti = floor(tc) - vec2(1.0, 0.0);
    vec3 n = -vec3((1.0 - abs(ti.x))*(ti.y*2.0 - 1.0), 
                   ti.x*ti.y, ti.x*(1.0 - ti.y));

    float bpy = min(0.9999, fragCoord.y / ts);
    float tpy = max(1.0, (fragCoord.y - 2.0) / ts);

    vec2 p = fract(vec2(tc.x, (bpy * (1.0 - floor(tc.y)) 
                               + tpy * floor(tc.y)))) - 0.5;
    
    vec3 px = vec3(0.5*n.x, p.y, -p.x*n.x) * step(0.5, n.x)
              + vec3(0.5*n.x, -p.x, -p.y*n.x) * step(n.x, -0.5);
    vec3 py = vec3(-p.x*n.y, 0.5*n.y, p.y) * abs(n.y);
    vec3 pz = vec3(p.x*n.z, p.y, 0.5*n.z) * abs(n.z);
    
    return normalize(px + py + pz);
}

// Function 259
vec4 tonemapping(vec4 x)
{
    return ((x*(SHOULDER_STRENGTH*x+LINEAR_ANGLE*LINEAR_STRENGTH)+TOE_STRENGTH*TOE_NUMERATOR)/
            (x*(SHOULDER_STRENGTH*x+LINEAR_STRENGTH)+TOE_STRENGTH*TOE_DENOMINATOR)) - TOE_ANGLE;
}

// Function 260
float remap01to_11(float a)
{
    return a * 2.0 + 1.0;
}

// Function 261
float Tonemapping(float x ){
    
    float calc;
    
    calc = ((x*(shoulderStrenght*x + linearAngle*linearStrenght) + toeStrenght*toeNumerator)/(x*(shoulderStrenght*x+linearStrenght)+toeStrenght* toeDenominator))- toeNumerator/ toeDenominator;                        
    
    return calc;    
}

// Function 262
mat3 sample_map(vec2 x, float w) {
    float q00 = map(x + vec2(-w,-w));
    float q10 = map(x + vec2( 0,-w));
    float q20 = map(x + vec2( w,-w));
    float q01 = map(x + vec2(-w, 0));
    float q11 = map(x + vec2( 0, 0));
    float q21 = map(x + vec2( w, 0));
    float q02 = map(x + vec2(-w, w));
    float q12 = map(x + vec2( 0, w));
    float q22 = map(x + vec2( w, w));
    return mat3(
		(q00 + q10 + q01 + q11) / 4.0, // 00
        (q11 + q10) / 2.0,  // 10
        (q10 + q20 + q11 + q21) / 4.0, // 20
        (q11 + q01) / 2.0, // 01
        q11, // 11
        (q11 + q21) / 2.0, // 21
        (q01 + q11 + q02 + q12) / 4.0, // 02
        (q11 + q12) / 2.0, // 12
        (q11 + q21 + q12 + q22) / 4.0 // 22
    );// * w;
}

// Function 263
vec3 mapImage( vec2 fragCoord )
{

    vec2 uv = fragCoord.xy / iResolution.xy;

    float srat=iResolution.y/iResolution.x;
    vec3 vlight=getLight(uv);
    float light=(vlight.x+2.5)*0.3;

    uv.x/=srat;
    vec3 col=vec3(0.);
    vec2 wmapuv=uv;
    wmapuv.x=mod(wmapuv.x-0.070,1./srat);
    wmapuv.x*=srat*1.004; 
    wmapuv.y+=0.01;
    wmapuv.y*=1.02;

    col=render(vec3(0.,0.,0.5),vec3(.8,.8,.3),getMap(wmapuv)*1.003);

    vec2 pcoord=vec2(remap(LOCATION.x,-180.,180.,0.,1.,true)/srat,remap(LOCATION.y,-90.,90.,0.,1.,true));


    if(light>1.) col*=render(vec3(0.2),vec3(1.,1.,1.),light);
    else col= 0.8 * col* pow(smoothstep(.089,1.,light),0.2)*light + col*0.2;

    float gv=graph(vec2((uv.x*srat),uv.y+0.5-pcoord.y));

    col=render(col,mix(col,vec3(1.,2.0,02.),0.15),gv);
    col+=0.7*render(vec3(0.),vec3(1.,0.,0.),
                    clamp(1.0+(0.5+0.5*sin(iTime*2.))*0.005-distance(vec2(uv.x*0.75,uv.y),vec2(pcoord.x*0.75,pcoord.y)),0.,
                          1.));

    col+=getGrid(vec2(uv.x*srat,uv.y))*0.3;
    return col;
}

// Function 264
vec4 BoxMap( sampler2D sam, in vec3 p, in vec3 n, in float k )
{
  vec3 m = pow( abs(n), vec3(k) );
  vec4 x = texture( sam, p.yz );
  vec4 y = texture( sam, p.zx );
  vec4 z = texture( sam, p.xy );
  return (x*m.x + y*m.y + z*m.z)/(m.x+m.y+m.z);
}

// Function 265
float Map(  vec3 p)
{

  p.y-=21.25;
  float  d=100000.0;
  vec3 checkPos = p;
  winDist=dekoDist=steelDist=lampDist=doorDist = 100000.0;
  d=sdCappedCylinder(p-vec3(0.0, -3.0, 0), vec2(3.20, 12.45));
  if (d>.2) return d;

  float noiseScale=(1.+(0.01*abs(noise(p*22.))));
  float noiseScale2=(1.+(0.03*abs(noise(p*13.))));

  d = sdCappedCylinder(p-vec3(0.0, 3.7, 0), vec2(inRad, .45));

  d=min(d, fCylinderH(p-vec3(0.0, 1.3, 0), radius*noiseScale, 1.80));
  d=min(d, sdConeSection(p-vec3(0.0, -6.0, 0.), 5.3, 2.4*noiseScale, 1.7*noiseScale));
  d=min(d, sdConeSection(p-vec3(0.0, -13.0, 0.), 1.8, 2.8*noiseScale, 2.6*noiseScale));

  // roof /////////////////
  dekoDist=min(dekoDist, sdConeSection(p-vec3(0., 6.7, 0), 0.40, 1.2, 0.8)); 

  checkPos = p;
  checkPos.xz = pModPolar(checkPos.xz, 26.0);   
  checkPos-=vec3(1.2, 6.7, 0);
  pR(checkPos.xy, 0.5);

  dekoDist=fOpUnionChamfer(dekoDist, sdCappedCylinder(checkPos, vec2(0.08, 0.47)), 0.1); // roof

  steelDist=min(steelDist, sdSphere(p-vec3(0., 6.6, 0), 1.05));    
  vec3 pp = p-vec3(0., 8., 0);
  float m = pModInterval1(pp.y, -0.14, 0.0, 2.);         
  steelDist=min(steelDist, sdSphere(pp, 0.20+(0.12*m)));   
  steelDist = fOpUnionChamfer(steelDist, sdCapsule(p-vec3(0., 8., 0), vec3(0, 0., 0), vec3(0, 1.0, 0), 0.013), 0.1);

  checkPos = p;
  // deko and windows steel top
  checkPos.xz = pModPolar(p.xz, 12.0);
  steelDist=min(steelDist, sdCappedCylinder(checkPos-vec3(outRad+0.05, 3.6, 0), vec2(0.03, .42))); // top railing
  steelDist=min(steelDist, sdCappedCylinder(checkPos-vec3(inRad-0.06, 4.4, 0), vec2(0.02, 1.45))); // window grid
  steelDist=min(steelDist, sdBox(checkPos-vec3(inRad-0.19, 6.25, 0), vec3(0.25, .3, 0.25)));
  steelDist=fOpIntersectionChamfer(steelDist, -sdBox(checkPos-vec3(inRad+0.20, 6.25, 0), vec3(0.19, 0.24, 0.19)), 0.12);
  // top window grid
  pp = p-vec3(0.0, 4.4, 0);
  pModInterval1(pp.y, 0.4, 0.0, 2.);          
  steelDist=min(steelDist, sdTorus(pp, vec2(inRad-0.02, .02)));  

  // top railing
  pp = p-vec3(0.0, 3.55, 0);
  m = pModInterval1(pp.y, 0.15, 0.0, 3.);          
  steelDist=min(steelDist, sdTorus(pp, vec2(outRad+0.05, mix(0.02, .035, step(3., m)))));

  #ifdef HIGH_QUALITY  
  d=min(d, sdSphere(p-vec3(0., 4., 0), 0.50));
  // lamp
  lampDist = sdEllipsoid(p-vec3(0., 4.9, 0), vec3(0.5, 0.6, 0.5)*(1.+abs(0.1*cos(p.y*50.))));
  lampDist = min(lampDist, sdCappedCylinder(p-vec3(0.0, 4.5, 0), vec2(0.12, 1.2)));    
  d=min(d, lampDist);
  #endif
    
  // tower "rings"
  pp = p-vec3(0.0, 4., 0);
  m = pModInterval1(pp.y, 1.8, 0.0, 1.);  
  dekoDist=min(dekoDist, sdTorus(pp, vec2(inRad, mix(.11, 0.15, step(1., m)))));                  

  // upper "rings"
  pp = p-vec3(0.0, -0.6, 0);
  m = pModInterval1(pp.y, -1.05, 0.0, 1.);   
  dekoDist=min(dekoDist, sdTorus(pp, vec2(mix(radius+0.15, radius+0.08, step(1., m)), 0.15)));                  


  dekoDist=min(dekoDist, sdTorus(p-vec3(0.0, -.35, 0), vec2(radius-0.05, .15)));
  dekoDist=min(dekoDist, fCylinderH(p-vec3(0.0, -.5, 0), radius+0.02, .15));
  dekoDist=min(dekoDist, fCylinderH(p-vec3(0.0, 3.18, 0), radius+0.35, 0.15));  

  // upper decoration
  pp = p-vec3(0.0, 2.7, 0);     
  dekoDist=min(dekoDist, fCylinderH(pp, radius+0.10, .30)); pp.y-=.15;
  dekoDist=min(dekoDist, fCylinderH(pp, radius+0.28, 0.18)); pp.y-=.15;
  dekoDist=min(dekoDist, fCylinderH(pp, radius+0.46, 0.18));
  checkPos.xz = pModPolar(p.xz, 6.0);
  dekoDist = max(dekoDist, -fCylinderV(checkPos-vec3(0.0, 2.4, 0), 0.6, 2.63));

  // middle and lower "rings"
  pp = p-vec3(0.0, -9., 0);
  m = pModInterval1(pp.y, -2.3, 0.0, 1.);    
  dekoDist=min(dekoDist, sdTorus(pp, vec2( mix( radius+0.6, 2.42, step(1., m)), .25))); 

  #ifdef HIGH_QUALITY
  // windows cutouts   
  checkPos.xz = pModPolar(p.xz, 6.0);   
  d=max(d, -sdBox(checkPos-vec3(2.20, 1.07, 0.), vec3(3.25, 0.6, 0.4))); 
  checkPos.xz = pModPolar(p.xz, 5.0); 
  pp = checkPos-vec3(2.50, -6.83, 0.);
  pModInterval1(pp.y, 3.5, 0.0, 1.);         
  d= max(d, -sdBox(pp, vec3(1.3, 0.35, 0.35)));  
  #endif

  // upper windows   
  checkPos.xz = pModPolar(p.xz, 6.0);   
  winDist = min(winDist, Window(checkPos-vec3(2.20, 0, 0.))); 

  // small windows  (upper deco)
  checkPos.xz = pModPolar(p.xz, 5.0); 

  pp = checkPos-vec3(2.10, -2.44, 0.0);
  m=pModInterval1(pp.y, -3.5, 0., 1.);

  pp-=mix(vec3(0.), vec3(0.28, 0.0, 0.), m);
  dekoDist=min(dekoDist, sdBox(pp, vec3(0.3, 0.4, 0.12)));   
  dekoDist = fOpIntersectionChamfer(dekoDist, -fCylinder(pp+vec3(-.30, -0.4, 0.0), 0.21, 0.63), .03); 
  dekoDist = max(dekoDist, -fCylinder(pp+vec3(-.40, .22, 0.0), 0.51, 0.63));  
  dekoDist=min(dekoDist, sdTorus(p-vec3(0.0, -2.26 - (m*3.55), 0), vec2(radius+0.25, .15)*(1.0+(m*0.14))));

  // small windows  
  pp = checkPos-vec3(2.82, -8.0, 0.);
  m=pModInterval1(pp.y, 3.5, 0., 1.);
  winDist = min(winDist, SmallWindow(pp+mix(vec3(0.), vec3(0.28, 0.0, 0.), m)));   

  #ifdef HIGH_QUALITY
  // make tower hollow
  d=max(d, -sdConeSection(p-vec3(0.0, -6.0, 0.), 5., 2.3, 1.55));
  #endif
    
  dekoDist=min(dekoDist, sdTorus(p-vec3(0., -15.2, 0), vec2(2.5, .75*noiseScale2))); 
  
  dekoDist=min(dekoDist, fCylinder(p-vec3(-0.05, -12.95, 2.25), 0.7, 0.5)); 

  // create door opening    
  float doorOpening = min(sdBox(p-vec3(-0.05, -13.9, 2.5), vec3(1.3, 1.4, 4.6)), fCylinder(p-vec3(-0.05, -12.75, 2.5), 0.6, 4.6));

  dekoDist = min(fOpPipe(dekoDist, doorOpening, 0.13), max(dekoDist, -doorOpening));

  checkPos.xz = pModPolar(p.xz, 8.0);
  d=fOpIntersectionChamfer(d, -fCylinderH(checkPos-vec3(2.95, -15.4, 0), 0.2, 3.6), 0.5);    
  checkPos.xz = pModPolar(p.xz, 16.0);
  d=fOpUnionChamfer(d, fCylinderH(checkPos-vec3(2.2, -10.3, 0), 0.03, 0.8), 0.4);    

  d=max(d, -sdBox(p-vec3(-0., -14., 2.7), vec3(0.6, 1.3, 4.6)));    
  d=max(d, -fCylinder(p-vec3(-0., -12.7, 2.5), 0.6, 4.6));    

  // door   
  doorDist =sdBox(p-vec3(-0., -13.6, 2.0), vec3(0.6, 1.3, 0.4)); 

  // door cutout     
  pp = p-vec3(-0.28, -13., 2.4);
  pModInterval1(pp.x, 0.46, 0., 1.);     
  doorDist=max(doorDist, -sdBox(pp, vec3(0.15, 0.25, 0.08)));   
  pp = p-vec3(-0.28, -13.8, 2.4);   
  doorDist=max(doorDist, -sdBox(pp, vec3(0.15, 0.4, 0.08))); pp.x-=0.46;
  doorDist=max(doorDist, -sdBox(pp, vec3(0.15, 0.4, 0.08))); 

  pp = p-vec3(-0., -15.20, 3.30);
  pp.z+=0.3; pp.y-=0.15;
  dekoDist=min(dekoDist, sdBox(pp, vec3(1.2, .075, 0.4)));  
  pp.z+=0.3; pp.y-=0.15;
  dekoDist=min(dekoDist, sdBox(pp, vec3(1.2, .075, 0.4)));  
      pp.z+=0.3; pp.y-=0.15;
  dekoDist=min(dekoDist, sdBox(pp, vec3(1.2, .075, 0.4)));  
  d=min(d, steelDist);
  d=min(d, dekoDist);
  d=min(d, winDist);
  d=min(d, doorDist);
  return  d;
}

// Function 266
float map(vec3 point){


wP = point + iCamPos;
    
        //shape finding functions:
        dSphere = sphere(point*iSphereScale, 0.3);
        dCone.l = ConeD(point,0.8,0.4);
       return min(dSphere.l,dCone.l);
       //return min(dSphere.l,point.z+20);
}

// Function 267
float map(float value, float min1, float max1, float min2, float max2) {
  return clamp(min2 + (value - min1) * (max2 - min2) / (max1 - min1),min2, max2);
}

// Function 268
vec2 sphereUV(vec3 center, float r, vec3 p)
{
    vec3 pDir = normalize(p - center);
    vec3 worldUp = vec3(0.0, 1.0, 0.0);
    float theta = acos(dot(pDir, worldUp));
    // If p is located at the positive part of z axis, then phi is 0-180.
    // If p is located at the negative part of z axis, then phi is 180-360.
    vec3 xzDir = normalize(vec3(pDir.x, 0.0, pDir.z));
    float phi = acos(dot(xzDir, vec3(1.0, 0.0, 0.0)));

    if(pDir.z < 0.0)
    {
        phi = phi + PI;
    }

    return vec2(theta / PI, phi / (2.0 * PI));
}

// Function 269
vec3 sampleReflectionMap(vec3 p, float b) {
    vec3 col = textureLod(reflectTex, p, b).rgb;
    
    // fake HDR
    //col *= 1.0 + 1.0 * smoothstep(0.5, 1.0, dot(LUMA, col));
    
    return col;
}

// Function 270
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    const vec2 e = vec2(0.001, 0);
    float ref = bumpFunction(p);                 
    vec3 grad = (vec3(bumpFunction(p - e.xyy),
                      bumpFunction(p - e.yxy),
                      bumpFunction(p - e.yyx) )-ref)/e.x;                     
          
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
	
}

// Function 271
float remap(float v, float oMin, float oMax, float rMin, float rMax){
	float result = (v - oMin)/(oMax - oMin);
    result = (rMax - rMin) * result + rMin;
    return result;
}

// Function 272
float remap01(float x, float m_, float _m)
{
 	return clamp((x-m_)/(_m-m_), 0., 1.);
}

// Function 273
float map(vec3 p, inout int matID, vec3 playerPos, bool drawPlayer, bool drawRefractive) {
    float res = FLT_MAX;
    
#if (SCENE == 0) || (SCENE > 3)
    // spheres
    vec3 cen = vec3(0.0, 1.5, 0.0);
    vec3 d = 0.5*normalize(vec3(1.0, 0, -1.0));
    vec3 s = 0.4*vec3(0.0, sin(iTime), 0.0);
    vec3 c = 0.4*vec3(0.0, cos(iTime), 0.0);
    float disp = 0.03*sin(20.0*p.x+iTime)*sin(40.0*p.y+iTime)*sin(60.0*p.z+iTime);
    disp *= (1.0 - smoothstep(-0.25, 0.25, cos(iTime/2.0)));
    propose(res, matID, sphere(p-cen, 0.3)+disp, 3);
    propose(res, matID, sphere(p-(cen+d.xyx+s), 0.15), 1);
    propose(res, matID, sphere(p-(cen+d.xyz+c), 0.15), 1);
    propose(res, matID, sphere(p-(cen+0.5*(-s.yxy)+0.5*vec3(0.0,1.0,0.0)), 0.15), 1);
    
    // refractive spheres
    if (drawRefractive) {
        propose(res, matID, sphere(p-(cen+d.zyz-s), 0.15), 2);
    	propose(res, matID, sphere(p-(cen+d.zyx-c), 0.15), 2);
    	propose(res, matID, sphere(p-(cen+0.5*(s.yxy)+0.5*vec3(0.0,-1.0,0.0)), 0.15), 2);
    }
    
    // reflective cylinder
    propose(res, matID, cylinder(p-vec3(0), 4.0, .15), 1);

#elif (SCENE == 1)
    //temple    
    float pillars = intersectionDist(cylinder(p-vec3(-50.0, 2.5, 0.0), 8.0, 2.5),
                                     cylinder(repeat(p-vec3(0.0,2.5,0.0), vec3(5,0,5)), 0.25, 2.5));    
    float temple = subtractionDist(sphere(p-vec3(-50.0, 0.0, 0.0), 10.0),
                                   cylinder(p-vec3(-50.0, 2.3, 0.0), 11.0, 2.5));
    propose(res, matID, blend(pillars, temple, 1.4), 6); 

#elif (SCENE == 2)
    if (drawRefractive) {
        // blob arm        
        float arm = cylinder(p-vec3(-50.0, 0.201, 0.0), 5.0, 0.1);
        for (int i = 0; i < 11; i++) {
            arm = blend(arm, sphere(p-vec3(-50.0+4.0*float(i)/10.0*cos(float(i+1)+iTime), i+1,
                                           4.0*float(i)/10.0*sin(float(i+1)+iTime)),
                                    float(10-i)/20.0 + 0.1), 1.0-float(i)/15.0);
        }
        propose(res, matID, arm, 2);
    }
    
#elif (SCENE == 3)
    // HORSE
    float blendStep = step(texture(iChannel1, vec2(KEY_H, 0.25)).x, 0.5);
    
    float foot1 = roundedBox(rY(p-vec3(0.0,0.249,0.0), -45.0), vec3(0.11,0.09,0.12), 0.05);
    float ankle1 = cylinder(rX(rY(p-vec3(-0.15,0.75,-0.15), -50.0), 15.0), 0.09, 0.4);
    float leg1 = cylinder(rX(rY(p-vec3(-0.05,1.65,0.03), 145.0), 35.0), 0.12, 0.6);
    float thigh1 = cylinder(rX(rY(p-vec3(-0.05,2.35,0.13), 75.0), -25.0), 0.3, 0.4);
    float fullleg1 = blend(foot1, ankle1, 0.5*blendStep);
    fullleg1 = blend(fullleg1, leg1, 0.5*blendStep);
    fullleg1 = blend(fullleg1, thigh1, 0.7*blendStep);
     
    float foot2 = roundedBox(rY(p-vec3(0.0-1.5,0.249,0.0+2.0), -25.0), vec3(0.11,0.09,0.12), 0.05);
    float ankle2 = cylinder(rX(rY(p-vec3(-0.15-1.5,0.69,-0.35+2.0), -20.0), 35.0), 0.09, 0.4);
    float leg2 = cylinder(rX(rY(p-vec3(-0.05-1.5,1.59,-0.37+2.0), 145.0), 45.0), 0.12, 0.6);
    float thigh2 = cylinder(rX(rY(p-vec3(-0.05-1.5,2.29,-0.27+2.0), -180.0), -25.0), 0.3, 0.4);
    float fullleg2 = blend(foot2, ankle2, 0.5*blendStep);
    fullleg2 = blend(fullleg2, leg2, 0.5*blendStep);
    fullleg2 = blend(fullleg2, thigh2, 0.7*blendStep);
    
    float bod = cylinder(rX(rY(p-vec3(-0.15-0.25,3.5,-0.35+1.75), -40.0), -45.0), 0.7, 1.0);
    float lowerhalf = blend(fullleg1, bod, 1.3*blendStep);
    lowerhalf = blend(lowerhalf, fullleg2, 1.3*blendStep);
    
    float bod2 = cylinder(rX(rY(p-vec3(-0.4+0.7,5.0,1.4+0.8), -40.0), -10.0), 0.5, 0.8);
    float lowerhalf2 = blend(lowerhalf, bod2, 1.4*blendStep);
    
    float uparm1 = cylinder(rX(rY(p-vec3(-0.05+1.5,4.9,0.53+1.3), 115.0), 70.0), 0.12, 0.75);
    float wrist1 = cylinder(rX(rY(p-vec3(-0.05+2.2,4.45,0.53+1.7), 115.0), -10.0), 0.1, 0.6);
    float hand1 = roundedBox(rX(rY(p-vec3(-0.05+2.15,3.95,0.53+1.65), 115.0), 80.0), vec3(0.11,0.09,0.12), 0.05);    
    float lowerhalf3 = blend(uparm1, lowerhalf2, 1.3*blendStep);
    lowerhalf3 = blend(lowerhalf3, wrist1, 0.5*blendStep);
    lowerhalf3 = blend(lowerhalf3, hand1, 0.4*blendStep);
    
    float uparm2 = cylinder(rX(rY(p-vec3(-0.05,4.9,0.53+2.8), -25.0), -70.0), 0.12, 0.75);
    float wrist2 = cylinder(rX(rY(p-vec3(-0.05+0.3,4.45,0.53+3.5), -25.0), 10.0), 0.1, 0.6);
    float hand2 = roundedBox(rX(rY(p-vec3(-0.05+0.25,3.95,0.53+3.45), -25.0), -80.0), vec3(0.11,0.09,0.12), 0.05);
    lowerhalf3 = blend(uparm2, lowerhalf3, 1.3*blendStep);
    lowerhalf3 = blend(lowerhalf3, wrist2, 0.5*blendStep);
    lowerhalf3 = blend(lowerhalf3, hand2, 0.4*blendStep);
    
    float neck = cylinder(rX(rY(p-vec3(-0.4+0.7,6.3,1.4+0.8), -40.0), 5.0), 0.4, 0.6);
    lowerhalf3 = blend(neck, lowerhalf3, 0.7*blendStep);
    
    float head = roundedBox(rX(rY(p-vec3(-0.4+1.1,7.2,1.4+1.2), -40.0), -15.0), vec3(0.2,0.2,0.65), 0.2);
    float horse = blend(head, lowerhalf3, 0.7*blendStep);
    
    float tail = cylinder(rX(rY(p-vec3(-0.15-1.25,3.5,-0.35+0.75), -40.0), 45.0), 0.15, 0.8);
    
    propose(res, matID, blend(horse, tail, 0.5*blendStep), 1);

#endif

    // ground plane
    propose(res, matID, plane(p-vec3(0.0,-1.0,0.0)), 4);
    
    // rounded box grid    
    vec3 v = repeat(p, vec3(GRID_SIZE,0.0,GRID_SIZE));
    vec3 f = p - v;
    float h = 30.0*perlinNoise2D(f.xz, 0.01, 1.0, 2, 37);
    h *= clamp(exp((dot(f.xz,f.xz) - 102.0*102.0)/20000.0)-1.0, 0.0, 1.0);
	propose(res, matID, roundedBox(v, vec3(0.7*GRID_SIZE/2.0,max(0.0, h),0.7*GRID_SIZE/2.0), 0.1), 4);
    
    // player
    if (drawPlayer) {
        propose(res, matID, sphere(p-playerPos, 0.25), 5);
    }
    
    return res;
}

// Function 274
float map(mediump vec2 rad)
{
    float a;
    if (res<0.0015) {
    	//a = noise2(rad.xy*20.6)*0.9+noise2(rad.xy*100.6)*0.1;
        a = noise222(rad.xy,vec2(20.6,100.6),vec2(0.9,0.1));
    } else if (res<0.005) {
        //float a1 = mix(noise2(rad.xy*10.6),1.0,l);
        //a = texture(iChannel0,rad*0.3).x;
        a = noise2(rad.xy*20.6);
        //if (a1<a) a=a1;
    } else a = noise2(rad.xy*10.3);
    return (a-0.5);
}

// Function 275
SDFRes map(in vec3 pos )
{   
    SDFRes plane = SDFRes( sdWaves(  pos - vec3(0.0, 0.0, 0.0) ), 1.0, 0.0 ,0.0);
    
    SDFRes dist = SDFRes( sdSphere( pos - vec3( -0.2,0.5, -0.4 + sin(iTime)*2.0), 0.7 ), 200.0, 0.0, 0.0);
    
    #ifdef BLEND
    dist = blendSDF( dist, SDFRes( sdSphere( pos - vec3( 0.6,0.5, -0.4), 0.5 ), 245.0, 0.0, 0.0), BLEND_AMOUNT );
    #else
    dist = addSDF( dist, SDFRes( sdSphere( pos - vec3( 0.6,0.5, -0.4), 0.5 ), 245.0, 0.0, 0.0) );
    #endif
    
    #ifdef BLEND    
    dist = blendSDF( dist, SDFRes( sdTorus( pos-vec3( 0.0,0.5+ sin(iTime*0.4), 1.0), vec2(0.5, 0.2) ), 40.6, 0.0, 0.0 ), BLEND_AMOUNT);
    #else    
    dist = addSDF( dist, SDFRes( sdTorus( pos-vec3( 0.0,0.5+ sin(iTime*0.4), 1.0), vec2(0.5, 0.2) ), 40.6, 0.0, 0.0 ));
    #endif    
    
    #ifdef BLEND
    dist = blendSDF(dist, plane, BLEND_AMOUNT);
    #else
    dist = addSDF(dist, plane);
    #endif
    
    return dist;
}

// Function 276
vec2 mapTerrain( vec3 p, float t )
{
    float h = -2.0+0.03;

    h += 5.0*textureImproved( iChannel2, iChannelResolution[2].xy, 0.0004*p.xz, 0.0004*t*drddx.xz, 0.0004*t*drddy.xz ).x;
    
    float di = smoothstep(100.0,500.0,length(p.xz) );
    h += 2.0*di;
    h *= 1.0 + 3.0*di;

	const float stonesClip = 100.0;
    if( (p.y-h)<0.5 && t<stonesClip )
    {
        float at = 1.0-smoothstep( stonesClip/2.0, stonesClip, t );
        float gr = textureGrad( iChannel2, 0.004*p.xz, 0.004*t*drddx.xz, 0.004*t*drddy.xz ).x;
        float pi = textureGrad( iChannel0, 0.400*p.xz, 0.400*t*drddx.xz, 0.400*t*drddy.xz ).x;
            
        gr = smoothstep( 0.2, 0.3, gr-pi*0.3+0.15 );
        h += at*(1.0-gr)*0.15*pi;
        h += at*0.1*textureGrad( iChannel2, 0.04*p.xz, 0.04*t*drddx.xz, 0.04*t*drddy.xz ).x;
    }


    float d = 0.8*(p.y-h);
    
    return vec2(d,2.0);
}

// Function 277
float map_bubble(vec3 pos)
{
   #ifdef thick_bottom
   float bubbleThickness2 = bubbleThickness*(1. + 500.*smoothstep(-0.25, -0.4, pos.y/bubbleRadius));
   #else
   float bubbleThickness2 = bubbleThickness;
   #endif
    
   float outside = length(pos) - bubbleRadius;
   outside-= bumpFactor*bubbleBump(pos);
   float inside = length(pos) - bubbleRadius + bubbleThickness2;
   inside-= bumpFactor*bubbleBump(pos);
   float df = max(outside, -inside);
    
   //df = max(df, pos.z);
   return df;
}

// Function 278
vec3 handleMap ( int index, float v) {
    vec3[6] arr;
    if (index == 0)      // blue
        arr = vec3[] ( vec3(63,63,116),vec3(48,96,130),vec3(91,110,225),vec3(95,205,228),vec3(203,219,252),vec3(255));
    else if (index == 1) // gold
        arr = vec3[] ( vec3(69,40,60),vec3(172,50,50),vec3(223,113,38),vec3(255,182,45),vec3(251,242,54),vec3(255));
    else if (index == 2) // green
        arr = vec3[] ( vec3(63,63,116),vec3(48,96,130),vec3(55,148,110),vec3(106,190,48),vec3(153,229,80),vec3(203,219,252));
    else if (index == 3) // brown
        arr = vec3[] ( vec3(69,40,60),vec3(102,57,49),vec3(143,86,59),vec3(180,123,80),vec3(217,160,102),vec3(238,195,154));
    else if (index == 4) // grey
        arr = vec3[] ( vec3(50,60,57),vec3(118,66,138),vec3(105,106,106),vec3(155,173,183),vec3(203,219,252),vec3(255));
    else if (index == 5) // pink
        arr = vec3[] ( vec3(50,60,57),vec3(63,63,116),vec3(118,66,138),vec3(217,87,99),vec3(215,123,186),vec3(238,195,154));
   return arr[ min(5, int(5. * v)) ] / 255.;
}

// Function 279
vec2 uv2lnglat(vec2 uv01) {
    return vec2(pi, halfPi) * (2.0 * uv01 - vec2(1.0));
}

// Function 280
vec2 remap01to_11(vec2 a)
{
    return vec2(remap01to_11(a.x), remap01to_11(a.y));
}

// Function 281
vec2 map(in vec3 p)
{
    vec3 q = p;
    return opU(vec2(sdBox(p, vec3(2., .01, 2.)), 0.), mushroom(q-vec3(0.,.65,0.)));
}

// Function 282
vec3 yuvToRgb(vec3 yuv) {
    mat3 transformation = mat3(1.0, 0.0, 1.13983, 
                               1.0, -0.39465, -0.5806,
                              1.0, 2.03211, 0.0);
    return transformation * yuv;
}

// Function 283
vec3 voronoiSphereMapping(vec3 n){
	vec2 uv=vec2(atan(n.x,n.z),acos(n.y));
   	return getVoronoi(1.5*uv);}

// Function 284
vec2 distortUV(vec2 uv, vec2 nUV)
{
    float intensity = 0.01;
    float scale = 0.01;
    float speed = 0.25;
    
    
    nUV.x += (iTime)*speed;
    nUV.y += (iTime)*speed;
    vec2 noise= texture( iChannel0, nUV*scale).xy;
    
    uv += (-1.0+noise*2.0) * intensity;
    
    return uv;
}

// Function 285
float MapRocks(vec3 p)
{
  GetRockRotation(p);
  vec3 checkPos = p;
  checkPos.xz = pModPolar(checkPos.xz, 230.0);
  checkPos-=vec3(124, 0., 0.);
  pModInterval1(checkPos.x, 6., 0., 4.);
  return sdSphere(checkPos, pow(0.5+noise(p*0.44), 2.)*1.5);
}

// Function 286
vec3 bumpMapNormal( const in vec3 pos, in vec3 nor ) {
    float i = tileId( pos, nor );
    if( i > 0. ) {
        nor+= 0.0125 * vec3( hash(i), hash(i+5.), hash(i+13.) );
        nor = normalize( nor );
    }
    return nor;
}

// Function 287
vec2 pixel2uv(vec2 px, bool bRecenter, bool bUniformSpace)
{
    if(bRecenter)
    {
        px.xy-=iResolution.xy*0.5;
	}
    
    vec2 resolution = bUniformSpace?iResolution.xx:iResolution.xy;
    vec2 uv = px.xy / resolution;
    return uv;
}

// Function 288
vec4 MyCubeMap_cube(vec3 ro, vec3 rd, vec3 pos, vec3 size)
{
    ro = ro-pos;
    float cullingDir = all(lessThan(abs(ro),size))?1.:-1.;
    vec3 viewSign = cullingDir*sign(rd);
    vec3 t = (viewSign*size-ro)/rd;
    vec2 uvx = (ro.zy+t.x*rd.zy)/size.zy; //face uv : [-1,1]
    vec2 uvy = (ro.xz+t.y*rd.xz)/size.xz;
    vec2 uvz = (ro.xy+t.z*rd.xy)/size.xy;
    if(      all(lessThan(abs(uvx),vec2(1))) && t.x > 0.) return vec4(t.x,(uvx+1.)/2.,0.5-viewSign.x/2.0);
    else if( all(lessThan(abs(uvy),vec2(1))) && t.y > 0.) return vec4(t.y,(uvy+1.)/2.,2.5-viewSign.y/2.0);
    else if( all(lessThan(abs(uvz),vec2(1))) && t.z > 0.) return vec4(t.z,(uvz+1.)/2.,4.5-viewSign.z/2.0);
	return vec4(2000.0,0,0,-1);
}

// Function 289
vec3 UVToEquirectCoord(float U, float V, float MinCos)
{
    float Phi = kPi - V * kPi;
    float Theta = U * 2.0 * kPi;
    vec3 Dir = vec3(cos(Theta), 0.0, sin(Theta));
	Dir.y   = clamp(cos(Phi), MinCos, 1.0);
	Dir.xz *= Sqrt(1.0 - Dir.y * Dir.y);
    return Dir;
}

// Function 290
float map( in vec3 p )
{
	float d = length(p-vec3(0.0,1.0,0.0))-1.0;
    d = smin( d, p.y, 1.0 );
    return d;
}

// Function 291
float map_hexagons(vec3 pos)
{
   vec4 h0 = hexagon(pos.xz);
   
   #ifdef specrot
   float colnr = mod(h0.x, 3.);
   pos.xz = rotateVec(pos.xz, colnr*pi/3.);
   vec4 h = hexagon(pos.xz);
   #else
   vec4 h = h0;
   #endif
   
   float lpx = pos.x - h.x*0.866025;
   float lpz = pos.z - h.y*1.5;
   vec3 pos2 = vec3(lpx, pos.y, lpz);
   
   float angle = getAngle(h0);
   //float angle = getAngle(h);
   pos2.xy = rotateVec(pos2.xy, angle);
   vec4 h2 = hexagon(vec2(pos2.x, pos.z - h.x*1.5));
   
   #ifdef border
   float borderhf = (1. - borderheight*smoothstep(borderpos, borderpos + borderwidth, h2.z))*(1. - 1.2*smoothstep(0.14, 0.115, h2.z)); 
   #else
   float borderhf = 1.;
   #endif
   
   float hex = max(-h2.z + bg_width/2. + gap_width, abs(pos2.y) - hex_height/2.*borderhf);
   return h2.x==0.?hex:1000.;
}

// Function 292
float map(vec3 p) {
    #if METHOD == 1
    	float r=sphere(p,1.3);
   	#endif
    #if METHOD == 2
    	float r=box(p,vec3(0.25));
    #endif
    
    return r;
}

// Function 293
vec2 pos2uv(Camera cam,vec3 pos
){vec3 dir=normalize(pos - cam.pos)*inverse(rotationMatrix(cam.rot))
 ;return dir.xy*cam.focalLength/dir.z;}

// Function 294
vec3 uv_to_direction(void)
{
    return(vec3(0.0));
}

// Function 295
vec2 findParallelogramUV(vec3 o, vec3 d, worldSpaceQuad wsQuad)
{
    //Note : This is tricky because axis are not orthogonal.
    vec3 uvX_ref = wsQuad.b-wsQuad.a; //horizontal axis
    vec3 uvY_ref = wsQuad.d-wsQuad.a; //vertical axis
    vec3 quadN = cross(uvY_ref,uvX_ref);
    float t = rayPlaneIntersec(o, d, wsQuad.a, quadN);
        
    vec3 p = o+t*d;
    vec3 X0_N = cross(uvY_ref,quadN);
    vec3 Y0_N = cross(uvX_ref,quadN);
    
    //Vertical component : find the point where plane X0 is crossed
    float t_x0 = rayPlaneIntersec(p, uvX_ref, wsQuad.a, X0_N);
    vec3 pY = p+t_x0*uvX_ref-wsQuad.a;
    //Horizontal component : find the point where plane Y0 is crossed
    float t_y0 = rayPlaneIntersec(p, uvY_ref, wsQuad.a, Y0_N);
    vec3 pX = p+t_y0*uvY_ref-wsQuad.a;
    
    //All is left to find is the relative length ot pX, pY compared to each axis reference
    return vec2(dot(pX,uvX_ref)/dot(uvX_ref,uvX_ref),
	            dot(pY,uvY_ref)/dot(uvY_ref,uvY_ref));
}

// Function 296
float domainRemapping(float minInput, float maxInput, float minOutput, float maxOutput, float domain)
{
    //normalize domain (put it into the 0-1 range)
    float normalizedDomain = (domain - minInput) / (maxInput - minInput);
    
    //use lerp
    return minOutput * (1. - normalizedDomain) + maxOutput * normalizedDomain;
    
    //I found a new method today thay requires less calculations, very interesting!
    //return normalizedDomain * (maxOutput - minOutput) + minOutput;  
}

// Function 297
float map(vec3 p)
{
    vec4 tt = vec4(iTime*0.03,iTime*0.07,iTime*0.5,iTime*0.75) * TAU;
	p.xz *= rotate(tt.x);
    p.zy *= rotate(tt.y);
    return sdBumpedSphere(p);
}

// Function 298
vec3 xyzToLuv(vec3 tuple){  float X = tuple.x;  float Y = tuple.y;  float Z = tuple.z;  float L = hsluv_yToL(Y);   float div = 1./dot(tuple,vec3(1,15,3));   return vec3(   1.,   (52. * (X*div) - 2.57179),   (117.* (Y*div) - 6.08816)  ) * L; }

// Function 299
float MapTailFlap(vec3 p, float mirrored)
{
  p.z+=0.3;
  pR(p.xz, rudderAngle*(-1.*mirrored)); 
  p.z-=0.3;

  float tailFlap =sdBox(p- vec3(0., -0.04, -.42), vec3(0.025, .45, .30));

  // tailFlap front cutout
  checkPos = p- vec3(0., 0., 1.15);
  pR(checkPos.yz, 1.32);
  tailFlap=max(tailFlap, -sdBox( checkPos, vec3(.75, 1.41, 1.6)));

  // tailFlap rear cutout
  checkPos = p- vec3(0., 0, -2.75);  
  pR(checkPos.yz, -0.15);
  tailFlap=fOpIntersectionRound(tailFlap, -sdBox( checkPos, vec3(.75, 1.4, 2.0)), 0.05);

  checkPos = p- vec3(0., 0., -.65);
  tailFlap = min(tailFlap, sdEllipsoid( checkPos-vec3(0.00, 0.25, 0), vec3(0.06, 0.05, 0.15)));
  tailFlap = min(tailFlap, sdEllipsoid( checkPos-vec3(0.00, 0.10, 0), vec3(0.06, 0.05, 0.15)));

  return tailFlap;
}

// Function 300
void CubeMapToSH2(out mat3 shR, out mat3 shG, out mat3 shB) {
    // Initialise sh to 0
    shR = mat3(0.0);
    shG = mat3(0.0);
    shB = mat3(0.0);
    
    vec2 ts = vec2(textureSize(reflectTex, 0));
    float maxMipMap = log2(max(ts.x, ts.y));
    float lodBias = maxMipMap-7.0;
    

    for (int i=0; i < ENV_SMPL_NUM; ++i) {
        vec3 sp = SpherePoints_GoldenAngle(float(i), float(ENV_SMPL_NUM));
        vec3 color = sampleReflectionMap(sp, lodBias);

        mat3 sh = shEvaluate(sp);
        shR = shAdd(shR, shScale(sh, color.r));
        shG = shAdd(shG, shScale(sh, color.g));
        shB = shAdd(shB, shScale(sh, color.b));            
    }

    // integrating over a sphere so each sample has a weight of 4*PI/samplecount (uniform solid angle, for each sample)
    float shFactor = 4.0 * PI / float(ENV_SMPL_NUM);
    shR = shScale(shR, shFactor );
    shG = shScale(shG, shFactor );
    shB = shScale(shB, shFactor );
}

// Function 301
float map_s(vec3 pos)
{  
    vec3 pos0 = pos;
    float fy = 132.*random(1.254*floor(pos.y/fe));
    fr = fr0*(-tdv*0.5 + 1. - 0.5*hsf);

    pos.y+= pyd;
    float fh = length(vec2(mod(pos.y, fe) - fe*0.6, pos.z + fr*sin((pos.x + fe*2. + fe*floor(pos.y/fe))/fe*pi))) - fr*1.1;
 
    pos = pos0;
    
    float fx = 145.*random(1.936*floor(pos.x/fe));
    fr = fr0*(-tdv*0.5 + 1. - 0.5*vsf);
    
    pos.x+= pxd;
    
    float fv = length(vec2(mod(pos.x, fe) - fe*0.6, pos.z - fr*sin((pos.y + fe*2. + fe*floor(pos.x/fe))/fe*pi))) - fr*1.1;
    return min(fh, fv);
}

// Function 302
vec3 trimap(vec3 p, vec3 n){
    vec3 yz = tex(iChannel0, p.yz).xyz;
    vec3 xz = tex(iChannel0, p.xz).xyz;
    vec3 xy = tex(iChannel0, p.xy).xyz;
   
    n /= (n.x + n.y + n.z);
    return yz*n.x + xz*n.y + xy*n.z;
}

// Function 303
vec2 polyconic_uv( vec2 fragCoord_ ) {

    vec2 uv = fragCoord_.xy / iResolution.xy;
    uv += 0.0*vec2( cos(iTime), sin(0.3*iTime) );
    //uv = uv - vec2(0.5);
    //uv *= 2.0;
    //uv -= vec2(0.5);

    vec2 lnglat = uv2lnglat( uv );

    //uv.x = uv.x / cos(uv.y * 3.14 / 2.0);
    //uv = sinusoidal_inv(uv);
    lnglat = polyconic_inv_gores(lnglat);
    lnglat.x += lng_offset;
    if (lnglat.x < -pi) lnglat.x += 2.0 * pi;
    if (lnglat.x > pi) lnglat.x -= 2.0 * pi;
    
    uv = lnglat2uv( lnglat );

    //uv.y = 1.0 - uv.y;
    //uv = clamp(uv, vec2(0.0), vec2(1.0));
    return uv;
}

// Function 304
vec3 envMap(vec3 rd, vec3 sn){
    
    vec3 sRd = rd; // Save rd, just for some mixing at the end.
    
    // Add a time component, scale, then pass into the noise function.
    rd.xy -= iTime*.075;
    rd *= 3.;
    
    float c = n3D(rd)*.57 + n3D(rd*2.)*.28 + n3D(rd*4.)*.15; // Noise value.
    c = smoothstep(0.4, 1., c); // Darken and add contast for more of a spotlight look.
    
    vec3 col = vec3(c, c*c, c*c*c*c); // Simple, warm coloring.
    //vec3 col = vec3(min(c*1.5, 1.), pow(c, 2.5), pow(c, 12.)); // More color.
    
    // Mix in some more red to tone it down and return.
    return mix(col, col.yzx, sRd*.25+.25); 
    
}

// Function 305
vec3 hpluv(float t) {

    const vec3 c0 = vec3(0.8498955002581585, 0.4832532405217748, 0.5421264483909258);
    const vec3 c1 = vec3(-4.409059134214186, 0.389492555133492, -1.168238680182178);
    const vec3 c2 = vec3(48.62277754887025, -0.1355162304606739, -7.080954185520974);
    const vec3 c3 = vec3(-236.4070580335005, 2.645941964751115, 51.62563361795004);
    const vec3 c4 = vec3(478.9041299570086, -8.918428894628779, -97.1930145700894);
    const vec3 c5 = vec3(-423.6767542055926, 8.066163760207662, 75.49542389205953);
    const vec3 c6 = vec3(136.9698973865875, -2.043826721347202, -21.67708580809141);

    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));

}

// Function 306
vec2 map( in vec3 pos )
{
#if 0
    vec2 res = opU( vec2( sdPlane(pos), 1.0 ),
	                vec2( spiral( pos.xzy-vec3( -.5,.5, 0.1), vec2(3.,10.), 0.05 ), 46.9 ) );
#else
	vec2 c = vec2(floor(17.*fract(0.01*iTime))+1., floor(4.*fract(0.05*iTime))+1.);
	vec2 res = opU( vec2( sdPlane(pos), 1.0 ),
	                vec2( dspiral2( pos.xzy-vec3( -.5,.5, 0.1), c, 0.03 ), 46.9 ) );
#endif
	return res;
}

// Function 307
float asteroidMap( const in vec3 p, const in vec3 id) {
    float d = asteroidRock(p, id) + noise(p*4.0) * ASTEROID_DISPLACEMENT;
    return d;
}

// Function 308
float map(vec3 p) {
    float d = sdPlane(p, vec4(0.0, 1.0, 0.0, 0.0));
    
    float rot_x = iTime * 3.1415 * 0.2;
    float cx = cos(rot_x);
    float sx = sin(rot_x);
    
    float rot_z = iTime * 3.1415 * 0.125;
    float cz = cos(rot_z);
    float sz = sin(rot_z);
    
    p = vec3(
        p.x,
        p.y * cx - p.z * sx,
        p.z * cx + p.y * sx
    );
    
    p = vec3(
        p.x * cz - p.y * sz,
        p.y * cz + p.x * sz,
        p.z
    );
    
    d = opU(d, sdBox(p - vec3(0.0, 1.5, -1.5), vec3(1.6, 1.5, 0.1)));
    d = opU(d, sdBox(p - vec3(1.5, 1.5, -0.25), vec3(0.1, 0.75, 2.25)));
     
    d = opU(d, opU_v2(sdSphere(p, 1.0), sdBox(p - vec3(0.75, 0.75, -0.75), vec3(0.75 - 0.025)) - 0.025, 0.1));
    //d = opU(d, opU_v2(sdSphere(p, 1.0), sdBox(p - vec3(0.75 * 3.0, 0.75, -0.75 * 3.0), vec3(0.75)) - 0.025));
    
    return d;
}

// Function 309
vec2 MapIce( in vec3 p , in float mult, in float cryDist)
{ 
	vec3 q = p; 
	q.y += meltHeight;
	
	//basic cube shape
	vec3 iceDimensions = vec3(1.0,1.0,0.5); 
	vec3 d = abs(q) - iceDimensions;
	float dist = (min(max(d.x,max(d.y,d.z)),0.0)+length(max(d,0.0))); 

	//Ice basic deformation 
	dist += meltDeformation*(sin(q.x*2.0+meltRandomPhase.z) + cos(q.y*2.0+meltRandomPhase.w));

	//water pool shape
#ifdef POOL_DEFORMATION
	float angle = atan(p.z,p.x);
	float waterDistort = sin(angle*3.0+meltRandomPhase.x)+0.3*cos(7.0*angle+meltRandomPhase.y);
	float radius = meltRadius *(1.0+0.2*waterDistort);
#else
	float radius = meltRadius+0.1;
#endif
	
	float cyl = max(length(p.xz)-radius,abs(p.y + 1.0)-0.005);
		
	//blending them together
#ifdef USE_SMOOTH_MIN
	float finalDist = SmoothMin(cyl,dist,meltSmooth); 
	
#ifdef USE_SMOOTH_INNER_SHAPE
	finalDist = SmoothMin(finalDist,cryDist+0.5,2.3);
#endif
	
#else 
	float finalDist = min(cyl,dist);
#endif
		
	//make the ice spawn from the logo itself
	finalDist = mix(finalDist,cryDist,iceTransformFactor);
	
	return vec2(mult*finalDist,2.0);
}

// Function 310
float map(vec3 p )
{

    R(p.zy, time * .05); R(p.xy, time * .15);
 
    float t = tau;

    // start animation / scale, swirl
    if (iMouse.z > 0.) 
    {

        t = -mod(time, -tau) + 2.0;
        
        R(p.xy, pi + time * .25);

        // make a spiral    
        if (t > 0. && sin(time * .5) > 0.)
            R(p.xz, p.y - 0.001 * pi + time);
    }

    #define wave(p, d, f, s) float(exp(1.0 - sin(length(cross(d, p)) / length(d) * f + s)))
    
    float f6 = 0.0;
    f6 = wave(p * 1.42, p.yxz, 2.01, t);

    float final = 0.0;
    final = (pi * log(f6) / log(2. * tau));
    
    /* // cross-capped disk
    f6 = wave(p , p-vec3(.1), 2., t);
    final = f6;
	*/
    
    //fake translucent https://www.shadertoy.com/view/XtGyzD
    float sp = length(p) - (t * 0.5);
    float edge = 0.0051; //will help with aa by discard edges
    final = -(0.001 - sqrt(edge+max(sp + final, -sp - final)));

    float thickness = 1.5;
    return pow(final, 2./thickness);
}

// Function 311
float MapEsmPod(vec3 p)
{
  float dist = fCylinder( p, 0.15, 1.0);   
  checkPos =  p- vec3(0, 0, -1.0);
  pModInterval1(checkPos.z, 2.0, .0, 1.0);
  return min(dist, sdEllipsoid(checkPos, vec3(0.15, 0.15, .5)));
}

// Function 312
vec2 DrawSimpleUVQuad(vec2 a, vec2 b, vec2 c, vec2 d,vec2 uva, vec2 uvb, vec2 uvc, vec2 uvd,float t, vec2 co){
    vec3 baria = toBari(a,b,c,co);
    vec3 barib = toBari(a,d,c,co);
    vec3 baric = toBari(b,c,d,co);
    vec3 barid = toBari(b,a,d,co);
    float i = DrawQuad(a,b,c,d,t,co);
    vec2 coord = vec2(0.);
    float j = 0.;
    if (baric.x>0. && baric.x<1. && baric.y>0. && baric.y<1. && baric.z>0. && baric.z<1.){
        coord+= toCartesian(uvb,uvc,uvd,baric);
        j++;
    }
    if (barid.x>0. && barid.x<1. && barid.y>0. && barid.y<1. && barid.z>0. && barid.z<1.){
        coord+= toCartesian(uvb,uva,uvd,barid);
        j++;
    }
    if (baria.x >0. && baria.x<1. && baria.y >0. && baria.y<1. && baria.z >0. && baria.z<1.){
        coord+= toCartesian(uva,uvb,uvc,baria);
        j++;
    }
    if (barib.x>0. && barib.x<1. && barib.y>0. && barib.y<1. && barib.z>0. && barib.z<1.){
        coord+= toCartesian(uva,uvd,uvc,barib);
        j++;
    }
    return coord/j;
}

// Function 313
float map( vec3 p ) {
    float t,a; vec3 q = p;
    q.xy = abs(q.xy), a = max(q.x,q.y);               // --- pyramid
    t = max( (a==q.x?q.y:q.x) -2.,                    // slopes sides
             a/1.3 + clamp(q.z,0.,9.) -9.25 );        // slopes top 
    t = max( t, q.z-7.);                              // top end
    t = min( t, a + clamp(sfloor(q.z),0.,7.) - 9.);   // grades 
    t = max( t,-max(min(q.x,q.y)-.5,abs(q.z-7.5)-.5));// doors
    t = max( t,-max(3.*abs(q.z-7.5),a)+1.5 );         // room
    t = max( t, q.z-9.);                              // top end
    s = q.z;                                          // --- forest. floor, then trees
    q = .03*sin(15.*p); p += q.x+q.y+q.z;             // distortion
    for (int k=0; k<9; k++) {                         // Worley-like dot structure
        vec2 d = vec2(k%3-1,k/3-1);                   // seek for closest dot in 9x9 cells around
        s = min(s, length( hash2x3(floor(p.xy)+d)           // random dot(cell)
                          - vec3(fract(p.xy)-d,p.z) ) -.5); // raypos rel to cur cell
    }
    return min(t,s);
}

// Function 314
vec2 map(vec3 pos)
{
    return map2(pos);
}

// Function 315
float map(vec2 p){
	// Reading distance fields from a texture means taking scaling into
    // consideration. If you zoom coordinates by a scalar (4, in this case), 
    // you need to scale the return distance value accordingly... Why does 
    // everything have to be so difficult? :D
    const float sc = 4.;
    vec4 tex = tx(iChannel0, p/sc);
    gIP = tex.yz; // The object ID is stored in the YZ channels..
    gObjID = tex.w;
    return tex.x*sc;
}

// Function 316
vec2 heightmap(vec2 p) {

    // get polygon distance
    float dpoly = dseg(p, O, A);
    dpoly = min(dpoly, dseg(p, A, B));
    dpoly = min(dpoly, dseg(p, B, C));
    dpoly = min(dpoly, dseg(p, C, D));
    dpoly = min(dpoly, dseg(p, D, O));
    
    // offset from edge
    float k = 0.08;
    
    // base height
    float z = k + 0.01 * noise(5.*p);
    
    if (dpoly < k) {
        // semicircular shoulder
        float w = (dpoly/k - 1.0);
        z *= sqrt(1.0 - w*w);
    } else {
        // depression inwards from edge
        z *= (1.0 - 0.03*smoothstep(k, 2.0*k, dpoly));
    }
    
    // return height and polygon distance
    return vec2(z, dpoly);
    
}

// Function 317
vec4 map(vec3 q3){
    q3.xz*=r2(T*.22); 
    // Scaling and Vars
    const float scale = 2./HEX_SCALE;
	const vec2 l = vec2(scale*1.732/2., scale);
	const vec2 s = l*2.;
    float d = 1e5;
    vec2 p, ip;
    // IDs and Center Points
    vec2 id = vec2(0);
    vec2 cntr = vec2(0);
    const vec2[4] ps4 = vec2[4](vec2(-l.x, l.y), l + vec2(0., l.y), -l, vec2(l.x, -l.y) + vec2(0., l.y));
    // which pass you're on
    float boxID = 0.; 
    for(int i = 0; i<4; i++){
        // Block center.
        cntr = ps4[i]/2.;
        // Local coordinates.
        p = q3.xz - cntr;
        ip = floor(p/s) + .5; // Local tile ID.
        p -= (ip)*s; // New local position.
        // Correct positional individual tile ID.
        vec2 idi = (ip)*s + cntr;
        //float hx=hash2(idi*.5);
        float hx=distance(idi,vec2(.0));
        float th = sampleFreq(.01+hx*.0021)*45.;
        th = abs(th*14.999)/15.*.15; 
        // make shape
        vec3 p3 = vec3(p.x,q3.y,p.y);
        float sp = length(p3-vec3(0.,th,0.))-((th*.12));
        if(sp<d){
            d = sp;
         	id = idi;
            boxID = float(i);
            mid = 2.;
        }   
    }
    // Return the distance, position-base ID and box ID.
    return vec4(d/1.7, id, boxID);
}

// Function 318
vec2 uvcoords(vec2 p) {
	vec2 uv = p / iResolution.xy;
    uv = uv * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;
    return uv;
}

// Function 319
float map(vec3 p)
{
    vec3 p2 = p;
    p2.xz *= mm2(-time*0.4);
    float d = max(cyl(p, vec2(1.,1.)), -sbox(p2 - vec3(1,1.,0), vec3(1.1+mo.x*0.6, 0.8 - mo.y*2.2, 1.2)));
    return max(d, -cyl(p + vec3(0.,-2.2-mo.y*2.2,0), vec2(0.75+sin(time)*0.2,2.)));
}

// Function 320
vec4 boxmap( sampler2D sam, in vec3 p, in vec3 n, in float k )
{
    vec3 m = pow( abs(n), vec3(k) );
	vec4 x = texture( sam, p.yz );
	vec4 y = texture( sam, p.zx );
	vec4 z = texture( sam, p.xy );
	return (x*m.x + y*m.y + z*m.z)/(m.x+m.y+m.z);
}

// Function 321
vec4 tonemappingLUT(vec2 uv)
{
    vec3 ld = vec3(0.002);
    float linReference = 0.18;
    float logReference = 444.;
    float logGamma = 0.45;
    vec4 outColor = texture(iChannel0, uv);
    outColor.rgb = (log(0.4*outColor.rgb/linReference)/ld*logGamma + logReference)/1023.;
    outColor.rgb = clamp(outColor.rgb, 0., 1.);
    float FilmLutWidth = 256.;
    float Padding = .5/FilmLutWidth;
    outColor.r = texture(iChannel0, vec2( mix(Padding,1.-Padding,outColor.r), .5)).r;
    outColor.g = texture(iChannel0, vec2( mix(Padding,1.-Padding,outColor.g), .5)).r;
    outColor.b = texture(iChannel0, vec2( mix(Padding,1.-Padding,outColor.b), .5)).r;
    return outColor;
}

// Function 322
float YUVtoB(float Y, float U, float b){
  float B=U/0.492+Y;
  return B;
}

// Function 323
float fogmap(in vec3 p , float d )
{
    //p.x += time*.0005;
   // p.z += sin(p.x*.00005);
    return triNoise3d(p*2.2/(d * 20.),0.2);
}

// Function 324
vec3   lchToLuv(float x, float y, float z) {return   lchToLuv( vec3(x,y,z) );}

// Function 325
vec3 sampleReflectionMap(vec3 sp, float lodBias){
    #ifdef LOD_BIAS
    	lodBias = LOD_BIAS;
    #endif
    vec3 color = SRGBtoLINEAR(textureLod(iChannel0, sp, lodBias).rgb);
    #if defined (HDR_FOR_POORS)
    	//color *= 1.0 + 2.0*smoothstep(hdrThreshold, 1.0, dot(LUMA, color)); //HDR for poors
    	color = InvTM(color, hdrThreshold);
   	#endif
    return color;
}

// Function 326
float map(vec3 p , out float idx)
{
    float d1 = dsphere(p,vec3(0.0),1.0);
    d1+=0.05*sin(p.x*10.0+iTime*2.0)*sin(p.y*10.0)*sin(p.z*10.0);
    
    float d2 = dsphere(p,vec3(2.15,0.0,0.0),0.5);
    d2 += 0.02*sin(p.x*10.0 + iTime*3.0);
    
    float d3 = dplane(p,vec3(0.0),normalize(vec3(0.0,1.0,0.0)),-1.5);
    
    float d4 = opB(d1,d2,1.3);
    
    return opU(d3,d4,idx);
}

// Function 327
vec3 AcesFilmicToneMap(const vec3 x)
{
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

// Function 328
float map_floor(vec3 pos, bool btext)
{
    float h=0.;
    #ifdef smooth_floor_bump
    #ifdef bumpmaps
   	h = texture(iChannel3, fract(woodSize*pos.xz)).x;
    #endif
    return pos.y - flpos - (btext?0.01*h:0.);
    #else
    #ifdef bumpmaps
    h = texture(iChannel0, woodSize*pos.xz).x;
    return pos.y - flpos - (btext?0.007*h:0.);
    #endif
    #endif
    #ifndef bumpmaps
    return pos.y - flpos;
    #endif
}

// Function 329
float map(vec3 p) {
    float d = plane(p,vec4(0.0,1.0,0.0,1.0));
    d = boolUnion(d,plane(p,vec4(0.0,-1.0,0.0,4.0))); 
    d = boolUnion(d,plane(p,vec4(0.0,0.0,1.0,5.0)));
    d = boolUnion(d,plane(p,vec4(0.0,0.0,-1.0,5.0)));  
    d = boolUnion(d,plane(p,vec4(1.0,0.0,0.0,8.0)));
    d = boolUnion(d,plane(p,vec4(-1.0,0.0,0.0,8.0)));  
    
    d = boolUnion(d,rbox(vec3(0.75,-0.51,0.45)-p,vec3(0.1,0.5,0.1)));    
    d = boolUnion(d,rbox(vec3(0.75,-0.51,-0.45)-p,vec3(0.1,0.5,0.1)));
    d = boolUnion(d,rbox(vec3(-0.75,-0.51,0.45)-p,vec3(0.1,0.5,0.1)));    
    d = boolUnion(d,rbox(vec3(-0.75,-0.51,-0.45)-p,vec3(0.1,0.5,0.1)));
    d = boolUnion(d,rbox(vec3(0.0,-0.06,0.0)-p,vec3(0.85,0.05,0.55)));
    
    d = boolUnion(d,quad(vec3(0.0,0.0,0.0)-p,HOLO_SIZE));
    return d;
}

// Function 330
float mapmask(vec3 p, vec3 div) {
    vec3 cp=clamp((p/3.0),-1.0,1.0)*3.0;
    return clamp(length(p-cp)*100.0,0.0,1.0);
}

// Function 331
float map(vec3 p) {
    float nearest = MAX_T;
    for (float i = 0.; i < N; i++) {
        vec2 sp = sfi(i, N);
        // over time, move points along the spiral
        sp += iTime;
        vec3 sp3 = pointOnSphere(sp.x, sp.y);
        nearest = min(nearest, sphere(p - sp3, .01));
    }
    return nearest;
}

// Function 332
vec3 Tonemap( vec3 x )
{
    float a = 0.010;
    float b = 0.132;
    float c = 0.010;
    float d = 0.163;
    float e = 0.101;

    return ( x * ( a * x + b ) ) / ( x * ( c * x + d ) + e );
}

// Function 333
vec3 filmicTonemapping(in vec3 color)
{
	return ((color*(kShoulderStrength*color+kLinearAngle*kLinearStrength)+kToeStrength*kToeNumerator) /
 			(color*(kShoulderStrength*color+kLinearStrength)+kToeStrength*kToeDenominator))-kToeNumerator/kToeDenominator;
}

// Function 334
float map(vec3 coord) {
    // Make there be a "ground" and "sky" by there being less rocks
    // when you go higher.
    
    vec3 tex_coord = coord * vec3(0.3, 1.0, 0.3) * 0.2;
    
    float rocks = 0.5 + coord.y * 0.0 - sample_tex(tex_coord);
    return rocks;
}

// Function 335
float map(vec3 q){
    
    vec3 p;
	// Scale factor, and distance.
    float s = 3., d = 0.;
    
    for(int i=0; i<3; i++){
 		// Repeat space.
        p = abs(fract(q/s)*s - s/2.); // Equivalent to: p = abs(mod(q, s) - s/2.);
		// Repeat Void Cubes. Cubes with a cross taken out.
 		d = max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - s/3.);
    	s /= 3.; // Divide space (each dimension) by 3.
    }
 
 	return d;    
}

// Function 336
vec2 getBRDFIntegrationMap(vec2 coord, vec2 scaleSize){
    // Avoid reading outside the tile in the atlas
    coord = clamp(coord, 1e-5, 0.99);
    vec2 texCoord = vec2(coord.x/2.0, coord.y / 2.0 + 0.5);
    texCoord *= scaleSize;
    return texture(iChannel2, texCoord).rg;
}

// Function 337
float map_lampg(vec3 pos)
{
    vec2 hmp = getLampMPos(pos);
    float hm = 0.5*getLampBump(hmp)*
               smoothstep(1.12, 0.93, length(pos.y)/lampsize.y)*
               smoothstep(0.91, 0.97, length(pos.xz)/lampsize.x);
    return opS(sdCylinder(pos, lampsize), sdCylinder(pos, lampsize*vec2(0.86, 1.5))) - hm;
}

// Function 338
float map( vec3 p )
{

    //return p.y + 1.0 * fbm( p + iTime * 0.2 );
	//float f = 1.7 - length( p ) * fbm( p );
    
    //p.zy *= rot( iTime * 0.1 );
    
    p.z -= iTime * 0.5;
    
    float f = fbm( p );
    
    return f;
    
}

// Function 339
ComplexMatrix2 M_mapRealsToLine(Complex L, Complex c, Complex R)
{
    return M_mapTripleToTriple(
        Complex(-1, 0), Complex(0, 0), Complex(1, 0),
        L, c, R);
}

// Function 340
float map( vec3 p )
{
    return opU( opU( smin( sdTorus( p + vec3( 0.0, -9.0*texture(iChannel0, vec2( 0.0, 0.0)).x+6.0/*2.0*sin(iTime)*/, 0.0 ), vec2( 2.0, 1.0 ) ), 
                sdRoundBox( p + vec3( 0.0, 4.7, 0.0 ), vec3( 3.0, 3.0, 3.0 ), 0.2 ), 1.8 ),
        			sdFlippedBox( p, vec3( 10.0 ) ) ), sdSphere( p + vec3( 0.0, 0.0, -7.0 ), 1.0 ) ) ;
	//return sdSphere( p + vec3( 0.0, 0.0, -4.0 ), 2.0 );
}

// Function 341
float map(vec3 p){

     vec2 tun = p.xy - path(p.z);
     vec2 tun2 = p.xy - path2(p.z);
     return 1.- smoothMinP(length(tun), length(tun2), 4.) + (0.5-surfFunc(p));
 
}

// Function 342
vec3 map(in vec3 pos) {	

    int ca = int(t) % 26;
    int cb = (ca + 1) % 26;
    int cc = (ca + 2) % 26;

    float da = approx_font_dist(pos.xy, 65+ca);
    float dc = approx_font_dist(pos.xy, 65+cc);
    
    float ft = fract(t);
    
    if (ft > 0.95) {
        da = mix(min(da, dc), dc, smoothstep(0.95, 1.0, ft));
    } else if (ft > 0.9) {
        da = mix(da, min(da, dc), smoothstep(0.9, 0.95, ft));
    }
                   
    float db = approx_font_dist(pos.zy, 65+cb);
                   
    
    return vec3(max(da, db), da, db);
   
}

// Function 343
vec3 tonemap(vec3 color)
{
    // Clamp Values Less Than 0.0
    color = max(color, 0.0);

    // Image Gamma
    const vec3 imageGamma = vec3(1.0/gamma);

    // Reinhard and Gamma-Correction
    color = pow(color/(color+1.0), imageGamma);
    //color = pow(tanh(color), imageGamma);

    // Return Tone-Mapped Color
    return clamp(color, 0.0, 1.0);
}

// Function 344
vec3 TonemapACESFilm(vec3 x)
{
    return saturate((x * (A * x + B)) / (x * (C * x + D) + E));
}

// Function 345
vec2 equirectangularMap(vec3 dir) {
	vec2 longlat = vec2(atan(dir.y,dir.x),acos(dir.z));
 	return longlat/vec2(2.0*PI,PI);
}

// Function 346
vec2 transformUVs( in vec2 iuvCorner, in vec2 uv )
{
    // random in [0,1]^4
	vec4 tx = hash4( iuvCorner );
    // scale component is +/-1 to mirror
    tx.zw = sign( tx.zw - 0.5 );
    // debug vis
    #if JIGGLE
    tx.xy *= .05*sin(5.*iTime+iuvCorner.x+iuvCorner.y);
    #endif
    // random scale and offset
	return tx.zw * uv + tx.xy;
}

// Function 347
float map(vec3 rp)
{
    vec3 pos = rp - vec3(iTime * 0.5, 0.0, 6.0); 
    vec3 pos3 = rp - vec3(0.0, -0.2, 5.7); 
    
    vec3 b = vec3(0.4, 0.0, 0.0);
 
    pos = mod(pos, b)-0.5 * b;
    pos3 = mod(pos3, b)-0.5 * b;
    
    float res = sdBox(pos, vec3(0.1, 1.7, 0.2));
	
    res = opU(res, pos.y + 0.3);
    res = opU(res, sdBox(pos3, vec3(4.0, 0.7, 1.0)));
    res = opU(res, -pos.y + 2.6);
   
    return res;
}

// Function 348
float map( vec3 p )
{
    p.xz *= 0.8;
    p.xyz += 1.000*sin(  2.0*p.yzx );
    p.xyz -= 0.500*sin(  4.0*p.yzx );
    float d = length( p.xyz ) - 1.5;
	return d * 0.25;
}

// Function 349
float remap(float f, float in1, float in2, float out1, float out2) { return mix(out1, out2, clamp((f - in1) / (in2 - in1), 0., 1.)); }

// Function 350
vec3 normalmap(vec2 p) {
    vec2 e = vec2(1e-3, 0);
    return normalize(vec3(
        heightmap(p - e.xy) - heightmap(p + e.xy),
        heightmap(p - e.yx) - heightmap(p + e.yx),
        2. * e.x));
}

// Function 351
void sampleCubemap(vec2 p, out float size, out bool inShape){
    vec4 t = T(p,0);
    inShape = t.x>0.5;
    size = t.g;
}

// Function 352
vec2 map(vec3 q3){
    vec2 res = vec2(1000.,0.);

    q3.x -= T*.35;

    float d = 1e5, t = 1e5;
    vec2 qid=floor((q3.xy+hlf)/size);

    vec3 qm = vec3(
        mod(q3.x+hlf,size)-hlf,
        mod(q3.y+hlf,size)-hlf,
        q3.z
    );

    q3.z+=size;
    vec3 qb = vec3(
        mod(q3.x+hlf,size)-hlf,
        mod(q3.y+hlf,size)-hlf,
        q3.z
    );

    float ht = hash21(qid); 
    vec3 bm = qm;
    // build box parts
    float f = length(bm)-(hlf*1.25);
    float b = fBox(bm,vec3(hlf),.012);
    float c = fBox(bm,vec3(hlf)*.93,.012);
    float di = ht > .28 ? max(c,-b) : max(c,-f);

    float c2 = fBox(qb,vec3(hlf)*.93,.012);

    di=min(c2,di);
    // box
    if(di<d) {
        d = di;
        sip = qid;
        fhp = bm;
        thsh = ht;
    }

    // truchet build parts
    float thx = (.115+.1*sin(q3.y*1.05) ) *size;
    if(ht>.5) qm.x *= -1.;

    float ti = min(
      sdTorus(qm-vec3(hlf,hlf,0),vec2(hlf,thx),0.),
      sdTorus(qm-vec3(-hlf,-hlf,0),vec2(hlf,thx),0.)
    );

    // truchet
    if(ti<t) {
        t = ti;
        sip = qid;
        shp = qm;
        thsh = ht;
    }

    if(d<res.x) res = vec2(d,1.);
    if(t<res.x) res = vec2(t,2.);
    return res;
}

// Function 353
float map( in vec3 pos ) {
    pos.y -= 23.;
    pR(pos.xy,pos.z/20.-time);
    vec3 bp = pos;
    pMod1(bp.z,40.);
    float b = fBoxCheap(bp,vec3(10.,10.,2.));
    	  b = max(b,-fBox2Cheap(pos.xy,vec2(8.+sin(pos.z/10.)))); 
	float d = min(b,-fBox2Cheap(pos.xy, vec2(10.)));
    return d;
}

// Function 354
float hsluv_yToL(float Y){
    return Y <= 0.0088564516790356308 ? Y * 903.2962962962963 : 116.0 * pow(Y, 1.0 / 3.0) - 16.0;
}

// Function 355
float fogmap(in vec3 p, in float d)
{
    p.x += time*1.5;
    p.z += sin(p.x*.5);
    return triNoise3d(p*2.2/(d+20.),0.2)*(1.-smoothstep(0.,.7,p.y));
}

// Function 356
float map (in vec3 p) 
{
	vec3 c = p; 
    float res = 0.0;
	for (int i=0; i < 4; i++) 
	{
		p = abs(p) / dot(p,p) -0.7;
		p.yz = vec2(p.y*p.y-p.z*p.z, 2.*p.y*p.z);
		res += exp(-20.0 * abs(dot(p,c)));
	}
	return res * 0.4;
}

// Function 357
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    rstate = uvec2(fragCoord.xy) + uint(iFrame) / 6U * 4096U;

    // Project ray direction on to the unit cube.
    vec3 absRayDir = abs(rayDir);
    rayDir /= max(absRayDir.x, max(absRayDir.y, absRayDir.z));

    // Get the index of the current face being rendered.
    int faceIndex = 0;

    if(absRayDir.y > absRayDir.x && absRayDir.y > absRayDir.z)
    {
        faceIndex = 2;
    }
    else if(absRayDir.z > absRayDir.x && absRayDir.z > absRayDir.y)
    {
        faceIndex = 4;
    }

    if(rayDir[faceIndex / 2] > 0.)
        faceIndex |= 1;

    // Sample previous result.
    fragColor = textureLod(iChannel0, rayDir,0.);

    // Skip this face if it's not the one chosen for this frame.
    if(faceIndex != (iFrame % 6))
        return;

    // Render for only one of the boxes per frame, as an extra speedup.
    if((iFrame / 12 & 1) == 0)
    {
        vec3 p = vec3(-1), q = vec3(1);
        vec3 samplePoint = (p + q) / 2. + (q - p) * rayDir / 2.;
        vec3 sampleNormal = boxNormal(samplePoint, p, q);
        fragColor.rg += sampleScene(samplePoint, -sampleNormal);
    }
    else
    {
        vec3 p = vec3(-.5, -.9, -.5), q = vec3(.5, -.5, .5);
        vec3 samplePoint = (p + q) / 2. + (q - p) * rayDir / 2.;
        vec3 sampleNormal = boxNormal(samplePoint, p, q);
        fragColor.ba += sampleScene(samplePoint, sampleNormal);
    }

}

// Function 358
vec1 nmaps(vec1 x){ return x*2.-1.; }

// Function 359
float o5229_input_t_map(vec2 uv) {
vec2 o5271_0_wat = abs((uv) - 0.5);
float o5271_0_d = max(o5271_0_wat.x,o5271_0_wat.y);vec4 o5271_0_1_rgba = o5271_gradient_gradient_fct(2.0*abs(fract(-0.2*iTime + p_o5271_repeat*o5271_0_d)-0.5));
float o118692_0_1_f = easeInOutCubic((dot((o5271_0_1_rgba).rgb, vec3(1.0))/3.0));

return o118692_0_1_f;
}

// Function 360
float remapFreq(float freq){
 // linear scale
 //return clamp(freq,fftMinBass,1.0);
 // log scale
 return clamp(to01(- log(1.0-freq/2.0 + 0.01)),fftMinBass,1.0);
}

// Function 361
vec4 cloudsMap( in vec3 pos )
{
    vec4 n = fbmd_8(pos*0.003*vec3(0.6,1.0,0.6)-vec3(0.1,1.9,2.8));
    vec2 h  =  smoothstepd( -60.0, 10.0, pos.y ) -  smoothstepd( 10.0, 500.0, pos.y );
    h.x = 2.0*n.x + h.x - 1.3;
    return vec4( h.x, 2.0*n.yzw*vec3(0.6,1.0,0.6)*0.003 + vec3(0.0,h.y,0.0)  );
}

// Function 362
vec2 uv_polar_logarithmic(vec2 domain, vec2 center, float fins, float log_factor, vec2 coord){
   vec2 polar = uv_polar(domain, center) * vec2(pi2_inv, 1);
   return vec2(polar.x * fins + coord.x, log_factor*log(polar.y) + coord.y);
}

// Function 363
float map_supports(vec3 pos)
{
    vec3 mpos1 = pos.yxz;
    vec3 mpos2 = mpos1;
    mpos2.yz = rotateVec(mpos2.yz, 2.*pi/3.);
    vec3 mpos3 = mpos1;
    mpos3.yz = rotateVec(mpos3.yz, 4.*pi/3.);
    return min(min(min(min(sdCylinder(pos - vec3(0., 1., 0.), vec2(0.45, 0.35)),                    // Socket
                           sdCylinder(pos - vec3(0., 1.4, 0.), vec2(0.08, 1.4))),                   // Cable
                           sdCylinder(mpos1 - vec3(lampsize.y, 0., 0.), vec2(0.044, lampsize.x))),  // Suport 1/3
                           sdCylinder(mpos2 - vec3(lampsize.y, 0., 0.), vec2(0.044, lampsize.x))),  // Suport 1/3
                           sdCylinder(mpos3 - vec3(lampsize.y, 0., 0.), vec2(0.044, lampsize.x)));  // Suport 1/3
}

// Function 364
float MapToScene( const in vec3 vPos )
{   
	float fResult = 1000.0;
	
	float fFloorDist = vPos.y + 3.2;	
	fResult = min(fResult, fFloorDist);
	
	vec3 vBuilding1Pos = vec3(68.8, 0.0, 55.0);
	const float fBuilding1Radius = 58.5;
	vec3 vBuilding1Offset = vBuilding1Pos - vPos;
	float fBuilding1Dist = length(vBuilding1Offset.xz) - fBuilding1Radius;
	
	fResult = min(fResult, fBuilding1Dist);
	
	vec3 vBuilding2Pos = vec3(60.0, 0.0, 55.0);
	const float fBuilding2Radius = 100.0;
	vec3 vBuilding2Offset = vBuilding2Pos - vPos;
	float fBuilding2Dist = length(vBuilding2Offset.xz) - fBuilding2Radius;
	fBuilding2Dist = max(vBuilding2Offset.z - 16.0, -fBuilding2Dist); // back only
	
	fResult = min(fResult, fBuilding2Dist);

	vec3 vBollardDomain = vPos;
	vBollardDomain -= vec3(1.0, -2.0, 14.2);
	//vBollardDomain = RotateY(vBollardDomain, 0.6);
	float fBollardDist = RoundBox(vBollardDomain, vec3(-0.2, .75, -.2));
		
	fResult = min(fResult, fBollardDist);
	
	vec3 vFenceDomain = vPos;
	vFenceDomain -= vec3(-5.5, -2.5, 7.0);
	vFenceDomain = RotateY(vFenceDomain, 1.5);
	float fFenceDist = GetDistanceBox(vFenceDomain, vec3(0.5, 1.2, 0.2));
		
	fResult = min(fResult, fFenceDist);
	
	vec3 vCabDomain = vPos;
	vCabDomain -= vec3(-1.4, -1.55,29.5);
	vCabDomain = RotateY(vCabDomain, 0.1);
	float fCabDist = RoundBox(vCabDomain+vec3(0.0, .85, 0.0), vec3(.8, .54, 2.5));
	fResult = min(fResult, fCabDist);
	fCabDist = RoundBox(vCabDomain, vec3(.6, 1.2, 1.2));
	fResult = sMin(fResult, fCabDist);

	vec3 vBusDomain = vPos;
	vBusDomain -= vec3(-15., 0.0, 29.5);
	vBusDomain = RotateY(vBusDomain, 0.35);
	float fBusDist = RoundBox(vBusDomain, vec3(.55, 1.8, 4.0));
		
	fResult = min(fResult, fBusDist);
		
	vec3 vBusShelter = vPos;
	vBusShelter -= vec3(7.5, -2.0, 30.0);
	vBusShelter = RotateY(vBusShelter, 0.3);
	float fBusShelterDist = RoundBox(vBusShelter, vec3(.725, 5.3, 1.7));
		
	fResult = min(fResult, fBusShelterDist);
	
	vec3 vRailings = vPos;
	vRailings -= vec3(15.0, -.55, 18.0);
	vRailings = RotateY(vRailings, 0.3);
	float fRailings = RoundBox(vRailings, vec3(.0, -.1, 7.5));
		
	fResult = min(fResult, fRailings);
	
	vec3 vCentralPavement = vPos;
	vCentralPavement -= vec3(5.3, -3.0, 8.0);
	vCentralPavement = RotateY(vCentralPavement, 0.6);
	float fCentralPavementDist = GetDistanceBox(vCentralPavement, vec3(0.8, 0.2, 8.0));
		
	fResult = min(fResult, fCentralPavementDist);
	
	return fResult;
}

// Function 365
vec4 terrainMapD( in vec2 p )
{
	const float sca = 0.0010;
    const float amp = 300.0;
    p *= sca;
    vec3 e = fbmd_9( p + vec2(1.0,-2.0) );
    vec2 c = smoothstepd( -0.08, -0.01, e.x );
	e.x = e.x + 0.15*c.x;
	e.yz = e.yz + 0.15*c.y*e.yz;    
    e.x *= amp;
    e.yz *= amp*sca;
    return vec4( e.x, normalize( vec3(-e.y,1.0,-e.z) ) );
}

// Function 366
float map_walls(vec3 pos, bool hasbumps)
{
   float walls = -abs(pos.x) + staircase_length/2.;
   walls = min(walls, -abs(pos.z) + staircase_width/2.);
   walls = max(walls, abs(pos.x) - staircase_length/2. - wall_thickness);
   walls = max(walls, abs(pos.z) - staircase_width/2. - wall_thickness);
   
   float posy2 = mod(pos.y + floor_height/2., floor_height) - floor_height/2.;
   
   if (pos.y>-floor_height/2.)
   {
       walls = min(walls, sdRoundBox(vec3(pos.x - staircase_length/2., posy2 + floor_height/2. - window_posy - window_height/2., pos.z), vec3(windowb_thickness, window_height/2. + windowb_width/2., window_width/2. + windowb_width/2.), 0.005));   
   
       float window = abs(pos.z) - window_width*0.5;
       window = max(window, -posy2 - floor_height*0.5 + window_posy);
       window = max(window, posy2 + floor_height*0.5 - window_posy - window_height);
       window = max(window, -pos.x);
       walls = max(walls, -window);
   }
   
   #ifdef bumps
   if (hasbumps)
   {
       vec3 pos2 = pos;
       pos2.y = mod(pos2.y, floor_height);
       walls = walls - bump_depth*(noise(pos2*bump_size) - noise(pos2*bump_size*3.));
   }
   #endif   
   
   float door1 = pos.x + (staircase_length/2. - floor_width/2. - door_width/2.);
   door1 = max(door1, -pos.x - (staircase_length/2. - floor_width/2. + door_width/2.));
   
   float door2 = -abs(pos.z) + (staircase_width/4. - door_width/2.);
   door2 = max(door2, abs(pos.z) - (staircase_width/4. + door_width/2.));
  
   float door = min(door1, door2);
   float posy3 = mod(pos.y, floor_height);
   door = max(door, -posy3 + 0.01);
   door = max(door,posy3 - door_height);
   door = max(door, pos.x);
   
   walls = max(walls, -door);
   
   return walls;
}

// Function 367
float UVtoIdx(vec2 uv, float stride)
{
    uv = floor(uv);
    stride = floor(stride);
    return stride * uv.y + uv.x;
}

// Function 368
ivec2 idx2uv(int idx, int width)
{ 
    return ivec2(idx % width, idx / width);
}

// Function 369
vec2 DistortUV( vec2 vUV, float f )
{
    vUV -= 0.5;

    float fScale = 0.0005;
    
    float r1 = 1. + f * fScale;
    
    vec3 v = vec3(vUV, sqrt( r1 * r1 - dot(vUV, vUV) ) );
    
    v = normalize(v);
    vUV = v.xy;
    
    
    vUV += 0.5;
    
    return vUV;
}

// Function 370
float map(vec3 p)
{
    float time = iTime;
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    vec3 rd = normalize(vec3(uv, 1.0));
    #ifdef rd
    return 0.0;
    #else
    return length(max(vec3(dot(cos(p), sin(p))), cos(p))*2.0-1.0) - 3.0;
    #endif
}

// Function 371
vec2 map(vec3 pos, bool hasbumps, bool forshadow)
{
    vec2 res;

    if (pos.x<staircase_length/2. + 0.5)
    {    
        floornr = floor((pos.y + 0.1)/floor_height);
        
        float walls = map_walls(pos, hasbumps);
        res = vec2(walls, WALLS_OBJ);

        #ifndef testmode

        float floors = map_floors(pos);
        res = opU(res, vec2(floors, FLOORS_OBJ)); 
        //vec2 res = vec2(floors, FLOORS_OBJ); 

        float stairs = map_stairs(pos);
        res = opU(res, vec2(stairs, STAIRS_OBJ));

        float doors = map_doors(pos);
        res = opU(res, vec2(doors, DOORS_OBJ));

        if (!forshadow)
        {
            float lamps = map_lamps(pos);
            res = opU(res, vec2(lamps, LAMPS_OBJ));
        }

        float handles = map_handles(pos);
        res = opU(res, vec2(handles, HANDLES_OBJ));    

        float handrail = map_handrail(pos);
        res = opU(res, vec2(handrail, HANDRAIL_OBJ));    

        #ifdef floornumber
        if (!forshadow)
        {    
           float floornumbero = map_floornumber(pos);
           res = opU(res, vec2(floornumbero, FLOORNUMB_OBJ));
        }
        #endif

        #ifdef doorbell
        float doorbellv = map_doorbell(pos);
        res = opU(res, vec2(doorbellv, DOORBELL_OBJ)); 
        #endif
    }
    else
    {
        #ifdef cityscape
        float city = map_city(pos);
        res = vec2(city, CITY_OBJ);      
        #endif
    }
    
    #endif
    
    return res;
}

// Function 372
vec2 map(vec3 p)
{  
    float dist = 0.;
    
    if (g.x<m.x ) 
   		dist = length(max(abs(p)-vec3(0.5),0.0)); // cube
    else
    	dist = length(p) - 1.; // sphere
    
    vec2 res = vec2(dist, 1);
    
    return res;
}

// Function 373
float terrainMap( const in vec3 pos ) {
	return pos.y - terrainMed(pos.xz);  
}

// Function 374
float mapTotal( in vec3 pos )
{
    float d1 = mapArlo( pos ).x;
    float d2 = mapTerrain( pos, length(pos) ).x;
    return min(d1,d2);
}

// Function 375
vec2 map (in vec3 p) {
    vec2 v = vec2(1.0 - sdBox(p, vec3(1.0)), 1.0);
    return v;
}

// Function 376
float map(vec3 p)
{
    float d = p.y;
    d -= sin(p.z*0.2 + 1.0 - cos(p.x*0.25))*0.35;
    float att = clamp(p.y*0.3 + 1.3, 0.,1.);
    d += cyclic3D(p*0.3)*att*1. + 1.;  
    return d;
}

// Function 377
vec3 CubemapNormal(in vec2 tile) 
{   
    float s = (2.0*square((tile.x + 1.0)*0.5) - 1.0);
    
    float x = square(tile.x) * square(tile.y + 1.0) * s;
    float y = square(tile.y) * s;
    float z = square(tile.x + 1.0) * square(tile.y + 1.0) * s;
 
    return vec3(x, y, z);
}

// Function 378
float map(vec3 p){
  return finalStellaIcosa(p, 0.5);
}

// Function 379
float map(vec3 q){
        
		vec3 p; float d = 0.;
        
        // One Void Cube.
    	p = abs(mod(q, 3.) - 1.5);
    	d = max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1.);

        // Subdividing into more Void Cubes.    
    	p = abs(mod(q, 1.) - 0.5); // Dividing above by 3.
    	d = max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1./3.);
        
        // And so on.
    	p = abs(mod(q, 1./3.) - 0.5/3.); // Dividing above by 3.
    	d = max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1./3./3.);
        
		// Continue on in this manner. For more levels, you'll want to loop it. There's
		// a commented out example in the code somewhere. Also, you can experiment with 
		// the code to create more interesting variants.

		return d;
	}

// Function 380
vec2 tile_CalcRelPositionUV(Tile_t tile, vec2 coord, float twist)
{
	vec2 relPos = tile_CalcRelPosition(tile, coord, twist);
	
	vec2 uv = (relPos + vec2(tile.radius, tile.radius))/(tile.radius * 2.0);
	
	return uv;
}

// Function 381
vec2 mapArlo( vec3 p )
{

    // body
    vec3 q = p;
    float co = cos(0.2);
    float si = sin(0.2);
    q.xy = mat2(co,si,-si,co)*q.xy;
    float d1 = sdEllipsoid( q, vec3(0.0,0.0,0.0), vec3(1.3,0.75,0.8) );
    float d2 = sdEllipsoid( q, vec3(0.05,0.45,0.0), vec3(0.8,0.6,0.5) );
    float d = smin( d1, d2, 0.4 );
    
    //neck wrinkles
    float r = length(p-vec3(-1.2,0.2,0.0));
    d -= 0.05*abs(sin(35.0*r))*exp(-7.0*abs(r)) * clamp(1.0-(p.y-0.3)*10.0,0.0,1.0);

    // tail
    {
    vec2 b = sdBezier( vec3(1.0,-0.4,0.0), vec3(2.0,-0.96,-0.5), vec3(3.0,-0.5,1.5), p );
    float tr = 0.3 - 0.25*b.y;
    float d3 = b.x - tr;
    d = smin( d, d3, 0.2 );
    }
    
    // neck
    {
    vec2 b = sdBezier( vec3(-0.9,0.3,0.0), vec3(-2.2,0.5,0.0), vec3(-2.6,1.7,0.0), p );
    float tr = 0.35 - 0.23*b.y;
    float d3 = b.x - tr;
    d = smin( d, d3, 0.15 );
    //d = min(d,d3);
	}


    float dn;
    // front-left leg
    {
    vec2 d3 = leg( p, vec3(-0.8,-0.1,0.5), vec3(-1.5,-0.5,0.65), vec3(-1.9,-1.1,0.65), 1.0, 0.0 );
    d = smin(d,d3.x,0.2);
    dn = d3.y;
    }
    // back-left leg
    {
    vec2 d3 = leg( p, vec3(0.5,-0.4,0.6), vec3(0.3,-1.05,0.6), vec3(0.8,-1.6,0.6), 0.5, 1.0 );
    d = smin(d,d3.x,0.2);
    dn = min(dn,d3.y);
    }
    // front-right leg
    {
    vec2 d3 = leg( p, vec3(-0.8,-0.2,-0.5), vec3(-1.0,-0.9,-0.65), vec3(-0.7,-1.6,-0.65), 1.0, 1.0 );
    d = smin(d,d3.x,0.2);
    dn = min(dn,d3.y);
    }
    // back-right leg
    {
    vec2 d3 = leg( p, vec3(0.5,-0.4,-0.6), vec3(0.8,-0.9,-0.6), vec3(1.6,-1.1,-0.7), 0.5, 0.0 );
    d = smin(d,d3.x,0.2);
    dn = min(dn,d3.y);
    }
    
    
    // head
    vec3 s = vec3(p.xy,abs(p.z));
    {
    vec2 l = sdLine( p, vec3(-2.7,2.36,0.0), vec3(-2.6,1.7,0.0) );
    float d3 = l.x - (0.22-0.1*smoothstep(0.1,1.0,l.y));
        
    // mouth
    //l = sdLine( p, vec3(-2.7,2.16,0.0), vec3(-3.35,2.12,0.0) );
    vec3 mp = p-vec3(-2.7,2.16,0.0);
    l = sdLine( mp*vec3(1.0,1.0,1.0-0.2*abs(mp.x)/0.65), vec3(0.0), vec3(-3.35,2.12,0.0)-vec3(-2.7,2.16,0.0) );
        
    float d4 = l.x - (0.12 + 0.04*smoothstep(0.0,1.0,l.y));      
    float d5 = sdEllipsoid( s, vec3(-3.4,2.5,0.0), vec3(0.8,0.5,2.0) );
    d4 = smax( d4, d5, 0.03 );
    
        
    d3 = smin( d3, d4, 0.1 );

        
    // mouth bottom
    {
    vec2 b = sdBezier( vec3(-2.6,1.75,0.0), vec3(-2.7,2.2,0.0), vec3(-3.25,2.12,0.0), p );
    float tr = 0.11 + 0.02*b.y;
    d4 = b.x - tr;
    d3 = smin( d3, d4, 0.001+0.06*(1.0-b.y*b.y) );
    }
        
    // brows    
    vec2 b = sdBezier( vec3(-2.84,2.50,0.04), vec3(-2.81,2.52,0.15), vec3(-2.76,2.4,0.18), s+vec3(0.0,-0.02,0.0) );
    float tr = 0.035 - 0.025*b.y;
    d4 = b.x - tr;
    d3 = smin( d3, d4, 0.025 );


    // eye wholes
    d4 = sdEllipsoid( s, vec3(-2.79,2.36,0.04), vec3(0.12,0.15,0.15) );
    d3 = smax( d3, -d4, 0.025 );    
        
    // nose holes    
    d4 = sdEllipsoid( s, vec3(-3.4,2.17,0.09), vec3(0.1,0.025,0.025) );
    d3 = smax( d3, -d4, 0.04 );    

        
    d = smin( d, d3, 0.01 );
    }
    vec2 res = vec2(d,0.0);
    
    
    // eyes
    float d4 = sdSphere( s, vec3(-2.755,2.36,0.045), 0.16 );
    if( d4<res.x ) res = vec2(d4,1.0);
    
    float te = textureLod( iChannel0, 3.0*p.xy, 0.0 ).x;
    float ve = normalize(p).y;
    res.x -= te*0.01*(1.0-smoothstep(0.6,1.5,length(p)))*(1.0-ve*ve);
    
    if( dn<res.x )  res = vec2(dn,3.0);

    return res;
}

// Function 382
vec3 lightMap( int index, float v ) {
    vec3[5] arr;
    
    if (index == 0)      // blue
        arr = vec3[] ( vec3(255),vec3(203,219,252),vec3(95,205,228), vec3(99,155,255), vec3(91,110,225));
    else if (index == 1) // gold 
        arr = vec3[] ( vec3(255),vec3(251,242,54), vec3(255,182,45), vec3(223,113,38), vec3(172,50,50));
    else if (index == 2) // green
        arr = vec3[] ( vec3(255),vec3(203,219,252),vec3(153,229,80), vec3(106,190,48), vec3(55,148,110));
    else if (index == 3) // brown 
        arr = vec3[] ( vec3(255),vec3(238,195,154),vec3(217,160,102),vec3(180,123,80), vec3(143,86,59));
    else if (index == 4) // grey
        arr = vec3[] ( vec3(255),vec3(203,219,252),vec3(155,173,183),vec3(132,126,135),vec3(105,106,106));
    else if (index == 5) // pink
        arr = vec3[] ( vec3(255),vec3(238,195,154),vec3(215,123,186),vec3(217,87,99),  vec3(118,66,138));
    
    return arr[ min(5, int(v)) ] / 255.;
}

// Function 383
float map(vec3 p)
{    
    // Sphere radius
    float sphereSize = 0.4;
    
    // Transform coordinate space so spheres repeat
    vec3 q = fract(p * 1.5) * 2.5 - 1.2;
    
     int tx = int(q.x);
    float fft  = texelFetch( iChannel0, ivec2(tx,0), 0 ).x; 
	fft *= 2.5;
    // Signed distance of sphere
    float s = sphere(q, sphereSize);
    
    float d = 0.4 * (sin(q.x*6.*fft) * sin(q.y*5.*fft) * sin(q.z*4.*fft) );
    
    float rot = iTime;    
    p *= vec3(cos(rot),-sin(rot),sin(rot));
    //return s +wave;
    return s+d;
}

// Function 384
float heightMapTracing(vec3 ori, vec3 dir, out vec3 p) {  
    float tm = 0.0;
    float tx = 1000.0;    
    float hx = map(ori + dir * tx);
    if(hx > 0.0) return tx;   
    float hm = map(ori + dir * tm);    
    float tmid = 0.0;
    for(int i = 0; i < NUM_STEPS; i++) {
        tmid = mix(tm,tx, hm/(hm-hx));                   
        p = ori + dir * tmid;                   
    	float hmid = map(p);
		if(hmid < 0.0) {
        	tx = tmid;
            hx = hmid;
        } else {
            tm = tmid;
            hm = hmid;
        }
    }
    return tmid;
}

// Function 385
float map_screen(vec3 pos)
{
   return sdRoundBox(pos + vec3(0., -1.96*supportSize.y - screenSize.y, 0.), screenSize, screenRR);
}

// Function 386
vec3 hsluv_intersectLineLine(vec3 line1x, vec3 line1y, vec3 line2x, vec3 line2y) {
    return (line1y - line2y) / (line2x - line1x);
}

// Function 387
vec3 map(vec3 p) {
    //p*=2.;
  //  vec3 d1 = vec3(1000);
  //  for (int u=0; u<4; u++) {
  //  	mv4D = B[u]; //vec4(1,0,0,0);
//	    d1 = dmin(d1,sdTesserac(p));
 //       mv4D = -B[u]; //vec4(1,0,0,0);
//	    d1 = dmin(d1,sdTesserac(p));
//    }
    mv4D = vec4(0);
    return dmin(sdTesserac(p), vec3(
#if SHAPE==0
        999.,
#elif SHAPE==1
        sdCubinder(vec4(p,WCurrent)*B, vec3(.5-RAYON*.5)),
#elif SHAPE==2 
        sdDuoCylinder(vec4(p,WCurrent)*B, vec2(.5-RAYON*.5)),
#elif SHAPE==3
        sdSphere(vec4(p,WCurrent)*B, .5),
#else        
        sdBox(vec4(p,WCurrent)*B, vec4(.5-RAYON*.5)),
#endif
        -.1, WCurrent));
}

// Function 388
float heightmap(vec2 uv) {
    return .2 * moonTexture(uv).r;
}

// Function 389
vec4 colorMap() {
    if (rayPos.y <= -9.8) {//ground
        return texture(iChannel1,rayPos.xz*0.1);
    }
    
    
    //cube
    vec4 samp = texture(iChannel2,normalize(rayPos));
    return samp*0.3+
        samp*max(0.0,dot(lightDir,distMapNormal(rayPos)));
}

// Function 390
float MapGlass(  vec3 p)
{   

  p.y-=21.25;
  vec3 checkPos = p;
  // tower windows
  float d = sdCappedCylinder(p-vec3(0.0, 5.0, 0), vec2(1.00, .8));
  checkPos.xz = pModPolar(p.xz, 6.0);
  // upper windows
  #ifdef HIGH_QUALITY
  d = min(d, sdBox(checkPos-vec3(1.550, 1.1, 0.), vec3(0.01, .60, 0.3)));   
  #else
  d = min(d, sdBox(checkPos-vec3(1.62, 1.1, 0.), vec3(0.01, .60, 0.3)));   
  #endif  
  checkPos.xz = pModPolar(p.xz, 5.0);
  // middle and lower windows 
  #ifdef HIGH_QUALITY
  checkPos-=vec3(2.03, -6.8, 0.);
  #else
  checkPos-=vec3(2.18, -6.8, 0.);
  #endif
    float m=pModInterval1(checkPos.y, 3.5, 0., 1.);
  return min(d, sdBox(checkPos+mix(vec3(0.), vec3(0.28, 0.0, 0.), m), vec3(0.01, 0.4, .3)));
}

// Function 391
vec3 Uncharted2ToneMapping(vec3 color)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    float W = 11.2;
    float exposure = 2.;
    color *= exposure;
    color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
    float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
    color /= white;
    color = pow(color, vec3(1. / gamma));
    return color;
}

// Function 392
vec3 colormap(float t) {
    return .5 + .5*cos(TWOPI*( t + vec3(0.0,0.1,0.2) ));
}

// Function 393
vec2 map2ds(vec2 z)
{
  // animation specific hacks
  if (min(abs(z.x),abs(z.y)) < epsilon) return z;  
  
  vec2  z2  = z*z;
  float d   = z2.x+z2.y;
  float uv  = z.x*z.y;
  float suv = sgn(uv)*(1.0/sqrt(2.0));
  float b   = d-4.*uv*uv;
  float s   = suv*sqrt(d-sgn(b)*sqrt(d*abs(b)));
    
  z += epsilon;
  
  return vec2(s/z.y, s/z.x);
}

// Function 394
float mapSeedEmitter(vec2 uv){
    DecodeData(texelFetch( iChannel0, ivec2(uv*iResolution.xy),0), seedCoord, seedColor);
    return length(seedCoord/iResolution.xy-uv)-seedSize;
}

// Function 395
float Map(in vec3 p)
{
	float h = terrain(p.xz); 
    return p.y - h;
}

// Function 396
vec3 i_spheremap_24( uint data )
{
    vec2 v = unpackSnorm2x12(data);
    float f = dot(v,v);
    return vec3( 2.0*v*sqrt(1.0-f), 1.0-2.0*f );
}

// Function 397
vec4 MapThornsID( in vec3 pos, out float which)
{
    vec4 ret = ThornVoronoi(pos, which);
	return vec4(ret.xyz, pos.y * .21 - ret.w - max(pos.y-5.0, 0.0) * .5 + max(pos.y-5.5, 0.0) * .8);
}

// Function 398
float map_stairs(vec3 pos)
{
   float h = (0.5*floor_height)/(staircase_length - floor_width - interfloor_width)*(nb_stairs - 1.)/nb_stairs - 0.01;
   float s = (staircase_length - floor_width - interfloor_width)/nb_stairs + 0.009;
   float d = mod(staircase_length*0.5 - interfloor_width, s) + 0.001;
   
   vec3 pos2 = pos;
    
   if (pos.z>0.)
   {
       pos2.y = mod(pos.y - floor_height/2., floor_height) + floor_height/2.;
    
       pos2.xz = -pos2.xz;
       pos2.y-= floor_height/2.;
       pos2.x+= floor_width - interfloor_width;
   }
   else
       pos2.y = mod(pos.y, floor_height);   
   
   pos2.y = mod(pos2.y + floor_height*0.4, floor_height) - floor_height*0.4;
   pos2.y-= h*(staircase_length*0.5 - floor_width - 0.23);

   float stairs = pos2.y - 2.*s*h - s*h*floor((pos2.x + d)/s);
   stairs = max(stairs, -mod(pos2.x + d, s) + 0.5*s);
   stairs = min(stairs, pos2.y - s*h - s*h*floor((pos2.x + d)/s));
   
   stairs = max(stairs, h*pos2.x*1.165 - pos2.y - floor_thickness + 0.173);
   
   // Trick to avoid that the further stairs gets "holes"
   stairs = min(stairs, 0.07 + abs(pos.z));    
   
   stairs = max(stairs, pos2.x - staircase_length/2. + interfloor_width);
   stairs = max(stairs, -pos2.x - staircase_length/2. + floor_width);
   stairs = max(stairs, -pos2.z - staircase_width/2. - 0.1);
   stairs = max(stairs, pos2.z + staircase_width/2. - stairs_width);
   
   return stairs;
}

// Function 399
vec3 hsluvToRgb(float x, float y, float z) {return hsluvToRgb( vec3(x,y,z) );}

// Function 400
MarchData map(vec3 p) {
	MarchData r = minResult(room(p), ed209(p));
	float gnd = length(p.y + 3.);
	if (gnd < r.d) {
		r.d = gnd;
		r.mat = vec3(.1);
	}

	return r;
}

// Function 401
float map_tunnel(vec3 pos)
{
    float tc = tunnel_curve(pos.z);
    float dc = dev_tunnel_curve(pos.z);
    pos.x-= tc;
    float zz = pos.z;
    pos.z = 0.;
    float a = atan(dc);
    pos.xz = rotateVec (pos.xz, a*0.5);
    pos.z = zz;
    
    pos.y-= tunnel_curve_y(pos.z);
    
    float tdf = (1. + 0.00007/(1.0011 + cos(tsf*pos.z)));
    float df = -length(pos.xy) + tubeRadius*tdf;
    //df = max(df, pos.y);

    return df;
}

// Function 402
float map(vec3 uv, mat3 q, vec3 l) {
    return sin(2.0 * pow(dot(q * uv + l, uv), 6.0 / 11.0));
}

// Function 403
vec3 envMap(vec3 rd){
    
   
    float c = tetraNoise(rd*3.)*.57 + tetraNoise(rd*6.)*.28 + tetraNoise(rd*12.)*.15; // Noise value.
    c = smoothstep(.4, 1., c); // Darken and add contast for more of a spotlight look.
    
    vec3 col = vec3(c*c*c, c*c, c); // Simple, cool coloring.
    //vec3 col = vec3(min(c*1.5, 1.), pow(c, 2.5), pow(c, 12.)); // Warm color.
    
    // Mix in the reverse color to tone it down and return.
    return mix(col, col.zxy, rd*.25 + .25); 
    
}

// Function 404
vec2 map( vec3 pos ){  
    
   	//vec2 res = vec2( (abs(sin( pos.x * pos.y * pos.z  * 10.)) * 1.9 ) + length( pos ) - 1., 0.0 );
  
    vec2 res = vec2( sin(iTime + cos( pos.x * 2. ) + sin( pos.y )+ cos( pos.z * .59))*.1 + length( pos ) - .8 , 0. );

    
    for( int i = 0; i < lightNum; i++ ){
     
        float d = length( pos - lightPos[i] );
        vec2 res2 =vec2( d - (pow(((sin(float( i) + 4.*sin( iTime ))) / float(lightNum)) , 2.) * 8.5 + .1) , 1. ); 
        res = opU( res ,  res2 );
        
    }
    

    
   	return res;
    
}

// Function 405
float mapTerrain( in vec3 pos )
{
	return pos.y*0.1 + (displacement(pos*vec3(0.8,1.0,0.8)) - 0.4)*(1.0-smoothstep(1.0,3.0,pos.y));
}

// Function 406
float map(vec3 p )
{
    vec3 oldp = p;
    float iz =floor(p.z);
 float ix =floor(p.x);
   
    if(iz > 0. && iz < 20.)
    p.xz = mod(p.xz, 10.) -5.;
    
    p.xz*=rot(p.y*PI/5.);
    p.xz*=rot(iTime/2.);
   // oldp.xz*=rot(iTime);//thought I needed this for surf at end but no

    
    float cyl1 = length(p.xz + vec2(1.0,0.0)) - 0.2;// - ((surf(p*30.)/60.));//+sin(p.y*8.)/50.;
    float cyl2 = length(p.xz - vec2(1.0, 0.0)) - 0.2;// - ((surf(p*30.)/60.));//+sin(p.y*8.)/50.;
    
     p.y = mod(p.y,.4)-.2;
    float bar = max(length(p.yz) - 0.07, abs(p.x) - .9) ;//-surf(p*20.)/50.;
    
	
   // p.xz*=rot(iTime);
	float dna =  min(min(cyl1, bar), cyl2) ;//if I wanted the noise applied to everything
    
    //if I wanted to added another object
    ///float cell = length(p)-.1-surf(p*3.)/4.;
    //return min(dna, cell);
    
    return dna ;// 
    //before I just had 
    //return min(min(cyl1, bar), cyl2);
    
    //other cool things
    //return min(min(cyl1, bar), cyl2) - (1.0-abs(surf(oldp*3.)/50.));
    //return min(min(cyl1, bar), cyl2) - (abs(fbm(oldp.yy*4.)/5.));

}

// Function 407
vec3 doBumpMap(in vec3 p, in vec3 n, float bumpfactor, inout float edge){
    
    // Resolution independent sample distance... Basically, I want the lines to be about
    // the same pixel with, regardless of resolution... Coding is annoying sometimes. :)
    vec2 e = vec2(3./iResolution.y, 0); 
    
    float f = bumpFunction(p); // Hit point function sample.
    
    float fx = bumpFunction(p - e.xyy); // Nearby sample in the X-direction.
    float fy = bumpFunction(p - e.yxy); // Nearby sample in the Y-direction.
    float fz = bumpFunction(p - e.yyx); // Nearby sample in the Y-direction.
    
    float fx2 = bumpFunction(p + e.xyy); // Sample in the opposite X-direction.
    float fy2 = bumpFunction(p + e.yxy); // Sample in the opposite Y-direction.
    float fz2 = bumpFunction(p+ e.yyx);  // Sample in the opposite Z-direction.
    
     
    // The gradient vector. Making use of the extra samples to obtain a more locally
    // accurate value. It has a bit of a smoothing effect, which is a bonus.
    vec3 grad = vec3(fx - fx2, fy - fy2, fz - fz2)/(e.x*2.);  
    //vec3 grad = (vec3(fx, fy, fz ) - f)/e.x;  // Without the extra samples.


    // Using the above samples to obtain an edge value. In essence, you're taking some
    // surrounding samples and determining how much they differ from the hit point
    // sample. It's really no different in concept to 2D edging.
    edge = abs(fx + fy + fz + fx2 + fy2 + fz2 - 6.*f);
    edge = smoothstep(0., 1., edge/e.x);
    
    // Some kind of gradient correction. I'm getting so old that I've forgotten why you
    // do this. It's a simple reason, and a necessary one. I remember that much. :D
    grad -= n*dot(n, grad);          
                      
    return normalize(n + grad*bumpfactor); // Bump the normal with the gradient vector.
	
}

// Function 408
vec3 MapColor(vec3 srgb)
{
    #ifdef USE_ACESCG
    return srgb * sRGBtoAP1;
    #else
    return srgb;
    #endif
}

// Function 409
vec3 SandParallaxOcclusionMapping(vec3 position, vec3 view)
{
    int pomCount = 6;
    float marchSize = 0.3;
    for(int i = 0; i < pomCount; i++)
    {
        if(position.y < GROUND_LEVEL -  SandHeightMap(position)) break;
        position += view * marchSize;
    }
    return position;
}

// Function 410
Impact map(in vec3 pos)
{
    float terrainDistance = dTerrain(pos);
    vec3 terrainColour = texture(iChannel0, vec2( pos.x/10.0,pos.y/10.0)).xyz;

    terrainColour = vec3(abs(sin(pos.z*2.3)/PI),abs(sin(pos.z)/PI),abs(cos(pos.z*1.333))/PI);
    
    terrainColour = pow(terrainColour, vec3(1.8));
  
    Impact terrain = Impact(terrainDistance, 0.0, terrainColour, 0);
    
    //Light globe
    float lightRadius = light1.r;
	float lightDistance = dSphere( pos-light1.p, lightRadius );
    Impact light_p1 = Impact(lightDistance,light1.lum, light1.col, 0);
    Impact closest = getClosest(terrain,light_p1);
    
    return closest;
}

// Function 411
float map(in vec3 position) {
  float result = sdBrick(position, brickSize);
    
  if (position.y < brickSize.y) {  
    float cutout = sdCutout(position, brickSize);
    float studs = sdStudsCutout(position, brickSize); 
      
    bool smallTube = brickSize.x < 1.0 || brickSize.z < 1.0;
      
    result = sdSmoothSubtraction(result, cutout, rounding);
    result = sdSmoothSubtraction(result, studs, rounding);
      
    if (brickSize.x > 0.5 || brickSize.z > 0.5) {
      float tubes = sdTubes(position, brickSize, smallTube ? 0.073 : studRadius);
      result = sdSmoothUnion(result, tubes, rounding);
    }
  }
    
  if (position.y > brickSize.y) {
    position.x -= mod(brickSize.x, 1.0);
    position.z -= mod(brickSize.z, 1.0);
    float studs = sdStuds(position, brickSize);
    if (studs < logoHeight * 2.0) { 
      studs += logo ? sdLogo(position) : 0.0; 
    }
    result = sdSmoothUnion(result, studs, rounded ? 0.015 : 0.0);  
  }    
    
  return result;
}

// Function 412
float map( in vec3 pos ) {
    vec3 q = pos;
    vec3 o = orbit(sin(iTime)*PI2,cos(iTime)*PI2,8.);
    pMod3(q, vec3(50.,40.,50.));
    float d = fCapsule(q,o,-o,5.);
    float dist = distance(-o,q);
    float dist2 = distance(o,q);
    if(dist>dist2){color = vec3(1.,1.1,1.1);}
    if(dist<=dist2){color = vec3(1.,0.,1.);}
    return d;
}

// Function 413
FaceInfo uvToFace(vec2 uv)
{
    //huv is the "horizontally unrolled" wide uv coord, where u.x=[0-6] and u.y=[0-1].
    //tuv is the tile uv coord, back to [0-1]. Note: 6th anf 7th tiles are cut in half and combined.
    const float freq = 2.5;
    uv *= freq;
    vec2 huv = vec2(uv.x+freq*floor(uv.y),fract(uv.y));
    float idx = floor(huv.x);
    vec2 tuv = vec2(fract(huv.x),huv.y+(idx>5.01?0.5:0.));
    return FaceInfo( tuv, min(idx,5.) );
}

// Function 414
vec3 UE3Tonemap(vec3 color)
{
    return color / (color + 0.155) * 1.019;
}

// Function 415
vec3 tonemap(vec3 col){
    
	col *= BRIGHTNESS;
	col = ((col * (A * col + C * B) + D * E) / (col * (A * col + B) + D * F)) - E/F;
	return col;
}

// Function 416
vec2 map(vec3 p){
    float d, mat, aTime = iTime/2.f;
    
    //BRIDGE
    vec3 q = repeat(p, 12.f , 0.f, 0.f);
    float a0 = arch_sdf(q, vec3(0.f), 7.f, 5.f, 2.f);
    float a1 = arch_sdf(q, vec3(0.f), 8.f, 6.f, 1.5f);
    float d0 = sub(a1, a0, 0.2f);
    mat = float(LAMBERT_RED);
    d = d0;
    
    
    // FLOATING BLOBS
    vec3 qs1 = repeat(p, 40.f, 0.f, 40.f);
    vec3 qs2 = repeat(p, 10.f, 0.f, 10.f);
    float f = 5.f;
    float hf = 15.5f * cos(p.z/40.f);
    vec3 h0 = vec3(f * cos(iTime/4.f), 15.f + (sin(p.x) + cos(p.z)) * sin(iTime) - hf * (sin(p.x/20.f) + cos(p.z/20.f)) * cos(iTime), f * sin(iTime/2.f));
    vec3 h1 = vec3(5.f + -f * cos(iTime/4.f), 20.f + (sin(p.x) + cos(p.z)) * sin(iTime) + hf * (cos(p.x/20.f) + sin(p.z/20.f)) * sin(iTime), 5.f + -f * sin(iTime/2.f));
    float s0 = sphere_sdf(qs1, h0, 3.5f);
    float s1 = sphere_sdf(qs1, h1, 3.5f);
    float d1 = smin(s0 * 0.5f, s1 * 0.5f, 2.f);
    d = smin(d, d1, 3.f);
    
    
    
    
    // WATER
    #if NOISE == 0
    float plane_noise = 0.f;
    #elif NOISE == 1
    float plane_noise = ((perlinNoise3D(p)) * (sin(iTime/5.f) + 1.1f) * 0.5f);
    #elif NOISE == 2
    float plane_noise = ((worley(p.xz/10.f, abs(sin(iTime) * cos(iTime/2.f)))) * (sin(iTime/5.f) + 1.1f) * 0.3f);
    #endif
    
    #ifdef SINEWAVES
    plane_noise += 2.f * cos(p.x/10.f - cos(iTime)) + 2.f * sin(p.z/15.f + sin(iTime));
    plane_noise -= 2.5f;
    #endif
    
    float d2 = infinite_plane_sdf(p, 0.25f + plane_noise);
    d = smin(d, d2 * 0.6f, 0.f);
    mat = equals(d, d2) ? float(WATER) : float(mat);

    d = smin(d, d1, 0.f);
    mat = equals(d, d1) ? float(BLOB) : float(mat);

    
    return vec2(d, mat);
}

// Function 417
vec3 simpleReinhardToneMapping(vec3 color)
{
    float exposure = 1.5;
    color *= exposure/(1. + color / exposure);
    color = pow(color, vec3(1. / gamma));
    return color;
}

// Function 418
vec2 compute_uv(vec3 n_direction)
{
    vec2 uv;
    uv.x = (pi + atan(n_direction.x, n_direction.z)) * r_two_pi;
    uv.y = n_direction.y * 0.5 + 0.5;
    return(uv);
}

// Function 419
int mapcolor(vec3 p, int parity) {
  vec4 p4 = unproject(p,parity);
  vec2 uv = vec2(atan(p4.w,p4.x)/TWOPI,
                 atan(p4.z,p4.y)/TWOPI);
  uv += 0.1*iTime;
  uv = getuv(uv);
  uv *= 10.0;
  int i = int(uv.x)+int(uv.y);
  uv = fract(uv);
  vec2 border = min(uv-0.2,0.8-uv);
  if (min(border.x,border.y) > 0.0) return 2;
  else return i%2;
}

// Function 420
vec3 map( vec3 p, vec4 c )
{
    vec4 z = vec4( p, 0.2 );
	
	float m2 = 0.0;
	vec2  t = vec2( 1e10 );

	float dz2 = 1.0;
	for( int i=0; i<10; i++ ) 
	{
        // |dz|² = |3z²|²
		dz2 *= 9.0*lengthSquared(qSquare(z));
        
		// z = z^3 + c		
		z = qCube( z ) + c;
		
        // stop under divergence		
        m2 = dot(z, z);		
        if( m2>10000.0 ) break;				 

        // orbit trapping ( |z|² and z_x  )
		t = min( t, vec2( m2, abs(z.x)) );

	}

	// distance estimator: d(z) = 0.5·log|z|·|z|/|dz|   (see http://iquilezles.org/www/articles/distancefractals/distancefractals.htm)
	float d = 0.25 * log(m2) * sqrt(m2/dz2 );

	return vec3( d, t );
}

// Function 421
vec3 TonemapFilmic_Hejl2015(vec3 hdr) 
{
    vec4 vh = vec4(hdr, TONEMAP_WHITE_POINT);
    vec4 va = 1.425 * vh + 0.05;
    vec4 vf = (vh * va + 0.004) / (vh * (va + 0.55) + 0.0491) - 0.0821;
    return vf.rgb / vf.www;
}

// Function 422
vec4 map(vec3 p)
{
   	float scale = 12.;
    float dist = 0.;
    
    float x = 6.;
    float z = 6.;
    
    vec4 disp = displacement(p);
        
    float y = 1. - smoothstep(0., 1., disp.x) * scale;
    
    #ifdef USE_SPHERE_OR_BOX
        dist = osphere(p, +5.-y);
    #else    
        if ( p.y > 0. ) dist = obox(p, vec3(x,1.-y,z));
        else dist = obox(p, vec3(x,1.,z));
	#endif
    
    return vec4(dist, disp.yzw);
}

// Function 423
float mapEdge(in vec3 rp)
{
    rp.x += getCurve(rp);
    float edgeL = -2.;
    float difx = 2.-abs(rp.x);
    return difx;
}

// Function 424
vec2 complexFromUv(in Window win, in vec2 uv) {
    return vec2(uv.x * win.w + win.x,
                uv.y * win.h + win.y);
}

// Function 425
float map(vec3 p){
    
    // Floor. Not really used here, but if you changed the block dimensions,
    // the you'd want this.
    float fl = -p.z + .1;

    // The extruded blocks.
    vec4 d4 = blocks(p);
    gID = d4.yzw; // Individual block ID.
    
 
    // Overall object ID.
    objID = fl<d4.x? 1. : 0.;
    
    // Combining the floor with the extruded image
    return  min(fl, d4.x);
 
}

// Function 426
float MapRearWing(vec3 p)
{
  float wing2 =sdBox( p- vec3(2.50, 0.1, -8.9), vec3(1.5, 0.017, 1.3)); 
  if (wing2<0.15) //Bounding Box test
  {
    // cutouts
    checkPos = p-vec3(3.0, 0.0, -5.9);
    pR(checkPos.xz, -0.5);
    wing2=fOpIntersectionRound(wing2, -sdBox( checkPos, vec3(6.75, 1.4, 2.0)), 0.2); 

    checkPos = p-vec3(0.0, 0.0, -4.9);
    pR(checkPos.xz, -0.5);
    wing2=fOpIntersectionRound(wing2, -sdBox( checkPos, vec3(3.3, 1.4, 1.70)), 0.2);

    checkPos = p-vec3(3.0, 0.0, -11.70);
    pR(checkPos.xz, -0.05);
    wing2=fOpIntersectionRound(wing2, -sdBox( checkPos, vec3(6.75, 1.4, 2.0)), 0.1); 

    checkPos = p-vec3(4.30, 0.0, -11.80);
    pR(checkPos.xz, 1.15);
    wing2=fOpIntersectionRound(wing2, -sdBox( checkPos, vec3(6.75, 1.4, 2.0)), 0.1);
  }
  return wing2;
}

// Function 427
vec3 decodePalYuv(vec3 yuv)
{
    return vec3(
        dot(yuv, vec3(1., 0., 1.13983)),
        dot(yuv, vec3(1., -0.39465, -0.58060)),
        dot(yuv, vec3(1., 2.03211, 0.))
    );
}

// Function 428
vec3 hsluv_lengthOfRayUntilIntersect(float theta, vec3 x, vec3 y) {  vec3 len = y / (sin(theta) - x * cos(theta));  if (len.r < 0.0) {len.r=1000.0;}  if (len.g < 0.0) {len.g=1000.0;}  if (len.b < 0.0) {len.b=1000.0;}  return len; }

// Function 429
ComplexMatrix2 M_mapTripleTo01I(Complex z0, Complex z1, Complex z2)
{
    return ComplexMatrix2(
        z0 - z2,
        z0 - z1,
        H_multiply(-z1, z0 - z2),
        H_multiply(-z2, z0 - z1));
}

// Function 430
float map(vec3 p) {
    p -= OFFSET;
    p /= SCALE;
   	return mHead(p);
	return length(p) - .3;
}

// Function 431
vec4 map( vec3 p )
{
	p.x += 0.1*sin( 3.0*p.y );
	
	float rr = length(p.xz);
	float ma = 0.0;
	vec2 uv = vec2(0.0);
	
	float d1 = rr - 1.5;
    if( d1<1.8 )
	{
		
		float siz = 6.0;
		vec3 x = p*siz + 0.5;
		vec3 xi = floor( x );
		vec3 xf = fract( x );

		vec2 d3 = vec2( 1000.0, 0.0 );
		for( int k=-1; k<=1; k++ )
        for( int j=-1; j<=1; j++ )
        for( int i=-1; i<=1; i++ )
        {
            vec3 b = vec3( float(i), float(j), float(k) );
			vec3 c = xi + b;
			
			float ic = dot(c.xz,c.xz)/(siz*siz);
			
			float re = 1.5;
			
			if( ic>(1.0*1.0) && ic < (re*re) )
			{
            vec3 r = b - xf + 0.5 + 0.4*(-1.0+2.0*hash3( c ));
			//vec3 r = c + 0.5 - x;

			vec3 ww = normalize( vec3(c.x,0.0,c.z) );
			ww.y += 1.0; ww = normalize(ww);
            ww += 0.25 * (-1.0+2.0*hash3( c+123.123 ));
				
			vec3 uu = normalize( cross( ww, vec3(0.0,1.0,0.0) ) );
			vec3 vv = normalize( cross( uu, ww ) );
			r = mat3(  uu.x, vv.x, ww.x,
					   uu.y, vv.y, ww.y,
					   uu.z, vv.z, ww.z )*r;
            float s = 0.75 + 0.5*hash1( c+167.7 );				
			float d = shape(r,s)/siz;
            if( d < d3.x )
            {
                d3 = vec2( d, 1.0 );
				ma = hash1( c.yzx+712.1 );
				uv = r.xy;
            }
			}
        }
		d1 = mix( rr-1.5, d3.x, d3.y );
	}
	
	d1 = min( d1, rr - 1.0 );

    return vec4(d1, ma, uv );
	
}

// Function 432
vec3 envMap(vec3 p){
    
    p *= 3.;
    p.y += iTime;
    
    float n3D2 = n3D(p*2.);
   
    // A bit of fBm.
    float c = n3D(p)*.57 + n3D2*.28 + n3D(p*4.)*.15;
    c = smoothstep(.4, 1., c); // Putting in some dark space.
    
    p = vec3(c, c*c, c*c*c); // Redish tinge.
    
    return mix(p, p.xzy, n3D2*.4); // Mixing in a bit of purple.

}

// Function 433
float mapDebug(vec3 p) {
    float d = map(p);
    #ifndef DEBUG
        return d;
    #endif
    float plane = min(abs(p.z), abs(p.y));
    hitDebugPlane = plane < abs(d);
    //hitDebugPlane = true;
    return hitDebugPlane ? plane : d;
}

// Function 434
vec3 heightmapNormal(vec2 uv) {
	float xdiff = heightmap(uv) - heightmap(uv+epsi.xy);
	float ydiff = heightmap(uv) - heightmap(uv+epsi.yx);
	return normalize(cross(vec3(epsi.yx, -xdiff), vec3(epsi.xy, -ydiff)));
}

// Function 435
float map(vec3 p) {
    float d = _house <= 0. ? 5. : sdHouse(p) + .1  - _house * .1;
    if (_boat > 0.) d = opU(d, sdBoat(p) + .1  - _boat * .1);
    if (_spaceship > 0.) d = opU(d, sdSpaceship(p) + .1  - _spaceship * .1);
    if (_atmosphere > 0.) d = opU(d, sdAtmosphere(p) + .1 - _atmosphere * .1);
    
    //return d;
    float s = 1.;
    for (int i = 0; i < 4; ++i) {
        tFan(p.xz, 10.);
        p = abs(p);
        p -= _kifsOffset;
        
        p *= _kifsRot;
        s *= 2.;
    }
    
    return opSU(d, sdBox(p * s, vec3(s / 17.)) / s, .1);
}

// Function 436
vec2 map(in vec3 l, in pln p) {
    return vec2(dot(l, p.o[0]), dot(l, p.o[2]));
}

// Function 437
vec3 acesFilmTonemapping(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    
    vec3 color = (x*(a*x+b))/(x*(c*x+d)+e); 
    return clamp(color,0.0,1.0);
}

// Function 438
vec3 mapUnderWaterNormal(vec3 pt, float e) {
    vec3 normal;
    normal.y = mapUnderWater(pt).x;    
    normal.x = mapUnderWater(vec3(pt.x+e,pt.y,pt.z)).x - normal.y;
    normal.z = mapUnderWater(vec3(pt.x,pt.y,pt.z+e)).x - normal.y;
    normal.y = e;
    return normalize(normal);
}

// Function 439
float map( in vec3 pos )
{
    const float GEO_SPHERE_RAD = 0.5;
    return length(pos)-GEO_SPHERE_RAD;
}

// Function 440
vec3 doBumpMap( sampler2D tx, in vec3 p, in vec3 n, float bf){   
    const vec2 e = vec2(0.001, 0);    
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));    
    vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.    
}

// Function 441
vec4 map4(vec3 p) {
  float k;
  return map4(p,k);
}

// Function 442
float map3( in vec3 p )
{
	vec3 q = p*cloudScale;
	float f;
    f  = .50000*noise( q ); q = q*2.02;
    f += .25000*noise( q ); q = q*2.03;
    f += .12500*noise( q );
	return clamp( 1.5 - p.y - 2. + 1.75*f, 0., 1. );
}

// Function 443
vec3 rgbToHpluv(float x, float y, float z) {return rgbToHpluv( vec3(x,y,z) );}

// Function 444
float MapPlane( vec3 p)
{
  float  d=100000.0;
  vec3 pOriginal = p;
  // rotate position 
  p=TranslatePos(p, pitch, roll);
  float mirrored=0.;
  // AABB TEST  
  float test = sdBox( p- vec3(0., -0., -3.), vec3(7.5, 4., 10.6));    
  if (test>1.0) return test;

  // mirror position at x=0.0. Both sides of the plane are equal.
  mirrored = pMirror(p.x, 0.0);

  float body= min(d, sdEllipsoid(p-vec3(0., 0.1, -4.40), vec3(0.50, 0.30, 2.)));
  body=fOpUnionRound(body, sdEllipsoid(p-vec3(0., 0., .50), vec3(0.50, 0.40, 3.25)), 1.);
  body=min(body, sdConeSection(p- vec3(0., 0., 3.8), 0.1, 0.15, 0.06));   

  body=min(body, sdConeSection(p- vec3(0., 0., 3.8), 0.7, 0.07, 0.01));   

  // window
  winDist =sdEllipsoid(p-vec3(0., 0.3, -0.10), vec3(0.45, 0.4, 1.45));
  winDist =fOpUnionRound(winDist, sdEllipsoid(p-vec3(0., 0.3, 0.60), vec3(0.3, 0.6, .75)), 0.4);
  winDist = max(winDist, -body);
  body = min(body, winDist);
  body=min(body, fOpPipe(winDist, sdBox(p-vec3(0., 0., 1.0), vec3(3.0, 1., .01)), 0.03));
  body=min(body, fOpPipe(winDist, sdBox(p-vec3(0., 0., .0), vec3(3.0, 1., .01)), 0.03));

  // front (nose)
  body=max(body, -max(fCylinder(p-vec3(0, 0, 2.5), .46, 0.04), -fCylinder(p-vec3(0, 0, 2.5), .35, 0.1)));
  checkPos = p-vec3(0, 0, 2.5);
  pR(checkPos.yz, 1.57);
  body=fOpIntersectionRound(body, -sdTorus(checkPos+vec3(0, 0.80, 0), vec2(.6, 0.05)), 0.015);
  body=fOpIntersectionRound(body, -sdTorus(checkPos+vec3(0, 2.30, 0), vec2(.62, 0.06)), 0.015);

  // wings       
  frontWingDist = MapFrontWing(p, mirrored);
  d=min(d, frontWingDist);   
  rearWingDist = MapRearWing(p);
  d=min(d, rearWingDist);
  topWingDist = MapTopWing(p, mirrored);
  d=min(d, topWingDist);

  // bottom
  checkPos = p-vec3(0., -0.6, -5.0);
  pR(checkPos.yz, 0.07);  
  d=fOpUnionRound(d, sdBox(checkPos, vec3(0.5, 0.2, 3.1)), 0.40);

  float holder = sdBox( p- vec3(0., -1.1, -4.30), vec3(0.08, 0.4, 0.8));  
  checkPos = p;
  pR(checkPos.yz, 0.85);
  holder=max(holder, -sdBox( checkPos- vec3(0., -5.64, -2.8), vec3(1.75, 1.4, 1.0))); 
  d=fOpUnionRound(d, holder, 0.25);

  // large bomb
  bombDist2 = fCylinder( p- vec3(0., -1.6, -4.0), 0.45, 1.);   
  bombDist2 =min(bombDist2, sdEllipsoid( p- vec3(0., -1.6, -3.20), vec3(0.45, 0.45, 2.)));   
  bombDist2 =min(bombDist2, sdEllipsoid( p- vec3(0., -1.6, -4.80), vec3(0.45, 0.45, 2.)));   

  d=min(d, bombDist2);

  d=min(d, sdEllipsoid(p- vec3(1.05, 0.13, -8.4), vec3(0.11, 0.18, 1.0)));    

  checkPos = p- vec3(0, 0.2, -5.0);
  d=fOpUnionRound(d, fOpIntersectionRound(sdBox( checkPos, vec3(1.2, 0.14, 3.7)), -sdBox( checkPos, vec3(1., 1.14, 4.7)), 0.2), 0.25);

  d=fOpUnionRound(d, sdEllipsoid( p- vec3(0, 0., -4.), vec3(1.21, 0.5, 2.50)), 0.75);

  // engine cutout
  blackDist = max(d, fCylinder(p- vec3(.8, -0.15, 0.), 0.5, 2.4)); 
  d=max(d, -fCylinder(p- vec3(.8, -0.15, 0.), 0.45, 2.4)); 

  // engine
  d =max(d, -sdBox(p-vec3(0., 0, -9.5), vec3(1.5, 0.4, 0.7)));

  engineDist=fCylinder(p- vec3(0.40, -0.1, -8.7), .42, 0.2);
  checkPos = p- vec3(0.4, -0.1, -8.3);
  pR(checkPos.yz, 1.57);
  engineDist=min(engineDist, sdTorus(checkPos, vec2(.25, 0.25)));
  engineDist=min(engineDist, sdConeSection(p- vec3(0.40, -0.1, -9.2), 0.3, .22, .36));

  checkPos = p-vec3(0., 0., -9.24);  
  checkPos.xy-=vec2(0.4, -0.1);
  checkPos.xy = pModPolar(checkPos.xy, 22.0);

  float engineCone = fOpPipe(engineDist, sdBox( checkPos, vec3(.6, 0.001, 0.26)), 0.015);
  engineDist=min(engineDist, engineCone);

  d=min(d, engineDist);
  eFlameDist = sdEllipsoid( p- vec3(0.4, -0.1, -9.45-(speed*0.07)+cos(iTime*40.0)*0.014), vec3(.17, 0.17, .10));
  d=min(d, eFlameDist);

  d=min(d, winDist);
  d=min(d, body);

  d=min(d, sdBox( p- vec3(1.1, 0., -6.90), vec3(.33, .12, .17))); 
  checkPos = p-vec3(0.65, 0.55, -1.4);
  pR(checkPos.yz, -0.35);
  d=min(d, sdBox(checkPos, vec3(0.2, 0.1, 0.45)));

  return min(d, eFlameDist);
}

// Function 445
vec3 sampleReflectionMap(vec3 sp, float lodBias){    
    vec3 color = SRGBtoLINEAR(textureLod(reflectTex, sp, lodBias).rgb);
    #if defined (HDR_FOR_POORS)
    	color *= 1.0 + 2.0*smoothstep(0.7, 1.0, dot(LUMA, color)); //HDR for poors
   	#endif
    return color;
}

// Function 446
float map(vec3 p){

    // Square tunnel.
    // For a square tunnel, use the Chebyshev(?) distance: max(abs(tun.x), abs(tun.y))
    vec2 tun = abs(p.xy - path(p.z))*vec2(0.5, 0.7071);
    float n = 1.- max(tun.x, tun.y) + (0.5-surfFunc(p));
    return min(n, p.y + FH);

/*    
    // Round tunnel.
    // For a round tunnel, use the Euclidean distance: length(tun.y)
    vec2 tun = (p.xy - path(p.z))*vec2(0.5, 0.7071);
    float n = 1.- length(tun) + (0.5-surfFunc(p));
    return min(n, p.y + FH);  
*/
    
/*
    // Rounded square tunnel using Minkowski distance: pow(pow(abs(tun.x), n), pow(abs(tun.y), n), 1/n)
    vec2 tun = abs(p.xy - path(p.z))*vec2(0.5, 0.7071);
    tun = pow(tun, vec2(4.));
    float n =1.-pow(tun.x + tun.y, 1.0/4.) + (0.5-surfFunc(p));
    return min(n, p.y + FH);
*/
 
}

// Function 447
vec2 map0sd(vec2 z)
{

  float s = inversesqrt(dot(z,z)+epsilon);
  float m = s*max(abs(z.x),abs(z.y));
 
  return m*z;
}

// Function 448
vec2 mapChair(in vec3 p)
{
    float d1 = sdBox(p + vec3(3.0, -2.0, 4.5), vec3(1.8, 3.1, 0.5));
    float d2 = max(-d1, sdBox(p + vec3(3.0, -2.5, 4.5), vec3(2.1, 2.9, 0.2)));
    float d3 = sdBox(p + vec3(3.0, -2.0, 8.5), vec3(1.8, 3.1, 0.5));
    float d4 = max(-d3, sdBox(p + vec3(3.0, -2.5, 8.5), vec3(2.1, 2.9, 0.2)));
    float d5 = sdBox(p + vec3(1.05, -2.0, 6.5), vec3(0.5, 4.3, 1.8));
    float d6 = max(-d5, sdBox(p + vec3(1.05, -2.5, 6.5), vec3(0.2, 4.1, 2.2)));
    vec3 size = vec3(2.0, 0.075, 0.075);
    float d7 = sdBox(p + vec3(3.0, -4.35, 4.5), size);
    float d8 = sdBox(p + vec3(3.0, -4.35, 8.5), size);
    float d9 = sdBox(p + vec3(1.05, -4.35, 6.5), vec3(0.075, 0.075, 2.0));
    float d10 = sdBox(p + vec3(1.05, -5.35, 6.5), vec3(0.075, 0.075, 2.0));
    size = vec3(0.075, 1.0, 0.075);
    d1 = sdBox(p + vec3(2.35, -4.25, 4.5), size);
    d3 = sdBox(p + vec3(3.65, -4.25, 4.5), size);
    d5 = sdBox(p + vec3(2.35, -4.25, 8.5), size);
    float d11 = sdBox(p + vec3(3.65, -4.25, 8.5), size);
    size = vec3(0.075, 1.75, 0.075);
    float d12 = sdBox(p + vec3(1.05, -4.75, 5.85), size);
    float d13 = sdBox(p + vec3(1.05, -4.75, 7.25), size);
    vec2 res = vec2(ID_CHAIR_0, min(d13, min(d12, min(d11, min(d5, min(d3, min(d1, min(d10, min(d9, min(d8, min(d7, min(d6, min(d4, min(d2, sdBox(p + vec3(3.0, -3.25, 6.5), vec3(2.1, 0.2, 2.2))))))))))))))));
    vec2 obj = vec2(ID_CHAIR_1, udRoundBox(p + vec3(3.05, -3.26, 6.5), vec3(1.8, 0.2, 1.8), 0.025));
    if (obj.y < res.y) res = obj;
    return res;
}

// Function 449
float Map(vec3 p)
{
	float h = FBM(p);
	return h-cloudy-.42;
}

// Function 450
float map(vec3 p){
    
    objID = 0.;
    
    #ifdef OBJECT_CAMERA_WRAP
    // Wrap the scene around the path. Optional. See the bump mapping function also.    
    p.xy -= camPath(p.z).xy;
    #else   
    p.x += 4.;
    #endif
    
    float d = lattice(p);

     
    return d*.95;//*.7;
}

// Function 451
float map(in float val, in float startIn, in float endIn, in float startOut, in float endOut)
{
    float norm = (val - startIn) / (endIn - startIn);
    return norm * (endOut - startOut) + startOut;
}

// Function 452
vec2 map_bricks(vec3 pos)
{
    vec3 pos2 = pos;
    pos2.yz+= 0.07*texture(iChannel1, pos.yz*0.005).g;
    pos2.z+= 0.5*(brickStep.z + 0.02)*mod(floor(0.5*pos2.y/brickStep.y), 2.);

    vec2 nb = floor(pos2.yz/brickStep.yz*vec2(0.5, 1.));
    float nbBrick = nb.x*2. + nb.y*80.;
    float btd = 1. + 0.3*(hash(nbBrick) - 0.5);
    
    pos2.yz = mod(pos2.yz, brickStep.yz*vec2(2., 1.));
    float bricks = udRoundBox(pos2 - vec3(wallPos.x - wallSize.x + brickSize.x*0.5*btd, brickStep.y, 0.), brickSize, brickBR);
    
    #ifdef brick_bump
    bricks+= 0.01*smoothstep(0.1, 0.95, texture(iChannel3, pos.yz*0.18).r + 0.6*texture(iChannel0, 0.2*pos.yz).r)*smoothstep(-0.2, -0.23, pos.x - wallPos.x + wallSize.x - brickSize.x*0.5*btd);
    #endif
    
    #ifdef show_chimney
    bricks = max(bricks, -sdCylinder(pos.yxz-chimneyOrig, vec2(tubeDiam/2. + 0.1, 2.)));
    #endif
    
    return vec2(bricks, BRICKS_OBJ);
}

// Function 453
vec3 PBR_HDRremap(vec3 c)
{
    float fHDR = smoothstep(2.900,3.0,c.x+c.y+c.z);
    vec3 cRedSky   = mix(c,1.3*vec3(4.5,2.5,2.0),fHDR);
    vec3 cBlueSky  = mix(c,1.8*vec3(2.0,2.5,3.0),fHDR);
    return mix(cRedSky,cBlueSky,SKY_COLOR);
}

// Function 454
float map(vec3 p)
{
    vec3 pA = p + vec3(1.2+sin(iTime)*0.8, 0.0, 0.0);
    vec3 pB = p - vec3(1.2+sin(iTime)*0.8, 0.0, 0.0);
    mat2 r0 = rot(iTime);
    mat2 r1 = rot(-iTime);
    pA.xz *= r0;
    pB.xz *= r1;
	float a = box(pA, vec3(1.0, 6.0, 1.0))-0.3;
    float b = length(pB) - 1.2;
    float dist = fOpUnionRound(a,b,0.5);
    if (a < b) 
    {
    	P = pA;
        tr = r0;
    }
    else
    {
    	P = pB;
    	tr = r1;
    }
	return dist;
}

// Function 455
int map_mode() {
    return (iMouse.w > 0.0) ? MAP_EQUIRECTANGULAR : MAP_FISHEYE;
}

// Function 456
gia1 map(gia1x3 p) {
    //return gia_sub(gia_length(p), 0.5);
   
    float a = iTime * 0.1;
    float c = cos(a);
    float s = sin(a);
    gia1x3 rp = gia1x3(
        gia_add(gia_mul(p.x, c), gia_mul(p.y, s)),
        gia_add(gia_mul(p.x, s), gia_mul(p.y, -c)),
        p.z
    );
    
    gia1 s1 = gia_sub(gia_length(gia_add(p, vec3(0.0,-0.5,0.0))),0.2);
    
    gia1x3 w = gia_abs(rp);
    gia1 s2 = gia_sub(gia_max(gia_max(w.x, gia_add(w.y,0.45)), w.z),0.5);
    
    return gia_min(s2,s1);
}

// Function 457
vec3 env_map(vec3 n, vec3 v)
{
    vec3 r = -reflect(v, n);
    return texture(iChannel0, r).rgb;
}

// Function 458
float map(vec3 p, int id)
{
    if(id == 0) return length(p.xz) - 1.;
    if(id == 1) return abs(p.y + 1.2);
    if(id == 2) return abs(p.y - 2.2);
    if(id == 3) return length(p-lp) - .1;
    if(id == 4) return abs(p.z - 5.);
    if(id == 5) return abs(p.z + 5.);
    if(id == 6) return abs(p.x - 5.);
    if(id == 7) return abs(p.x + 5.);
    return MAX;
}

// Function 459
float map(vec3 p){
    
    vec2 pth = path(p.z);
    
    float sf = surfFunc(p); // Surface perturbation.

    // The terrain base layer.
    float ter = p.y - 3. + dot(sin(p*3.14159/18. - cos(p.yzx*3.14159/18.)), vec3(3)); // 6. smoothing factor.
    //float ter = p.y - 4. + dot(sin(p*3.14159/16.), cos(p.yzx*3.14159/32.))*3.; // 4. smoothing factor.

    float st = stairs(p, pth); // The physical path. Not to be confused with the camera path.

    p.xy -= pth; // Wrap the tunnel around the path.

    float n = 1.5 - length(p.xy*vec2(.5, 1)); // The tunnel to bore through the rock.
    n = smax(n + (.5 - sf)*1.5, ter + (.5 - sf)*3., 6.); // Smoothly boring the tunnel through the terrain.
    n = smax(n, -max(abs(p.x) - 1.75, abs(p.y + 1.5) - 1.5), .5); // Clearing away the rock around the stairs.
 
    // Object ID.
    objID = step(n, st); // Either the physical path or the surrounds.
    
    return min(n, st)*.866; // Return the minimum hit point.
 
}

// Function 460
vec4 hsluvToLch(float x, float y, float z, float a) {return hsluvToLch( vec4(x,y,z,a) );}

// Function 461
vec3 InvTonemap(vec3 color)
{
    color = pow(abs(color), vec3(Gamma));
    color = ReinhardInv(color);
    
    return color;
}

// Function 462
vec2 map_chimney(vec3 pos)
{
    const float cw = 0.16;
    const float ch = 0.02;
    
    pos = pos.yxz;
    pos-= chimneyOrig;
    
    // Horizontal tube
    float chimney = sdCylinder(pos, vec2(tubeDiam/2., tubeLen1));
    // Bands around the tube
    float angle1 = atan(pos.x, pos.z) + 3.*pos.y - 0.07;
    float angle2 = atan(pos.y + tubeLen1 + tubeclen, pos.z) - 3.*pos.x + 1.07;
    chimney-= ch*smoothstep(cw, 0., abs(0.2 - mod(angle1, pi)));
    
    // Curved tube
    float rtube = sdTorusQ(pos.yzx + vec3(tubeLen1, 0., -tubeclen), vec2(tubeclen, tubeDiam/2.));
    vec3 pos2 = pos.yzx + vec3(tubeLen1, 0., -tubeclen);
    // Bands around the tube
    float angle3 = atan(pos2.x - 0.025, pos2.z);
    float angle4 = atan(pos.x*cos(angle3*0.65 - 1.15) + (pos.y + 2.7)*sin(angle3*0.65 - 1.), pos.z) + 4.3*angle3 - 1.72;
    rtube-= ch*smoothstep(cw, 0., abs(0.2 - mod(angle4, pi)));
    
    chimney = min(chimney, rtube);
    
    // Vertical tube
    float tube2 = sdCylinder(pos.yxz + vec3(tubeLen1 + tubeclen, -tubeclen - tubeLen2, 0), vec2(tubeDiam/2., tubeLen2));
    // Bands around the tube
    tube2-= ch*smoothstep(cw, 0., abs(0.2 - mod(angle2, pi)));
    chimney = min(chimney, tube2);
 
    // Broad conic top of the chimney
    float cone1 = sdConeSection(pos.yxz + vec3(tubeLen1 + tubeclen, -tubeclen - tubeLen2 - tubeLen3, 0), tubeLen3/2., tubeDiam/2.6, tubeDiam2/2.23);
    float cone2 = sdConeSection(pos.yxz + vec3(tubeLen1 + tubeclen, -tubeclen - tubeLen2 - tubeLen3*2. - 0.08, 0), tubeLen3/2., tubeDiam2/2.23, tubeDiam/2.56);
    float cone = smin(cone1, cone2, 0.2);
    chimney = smin(chimney, cone, 0.1);
    
    // Small "collar" at the base of the broad part 
    float collar1 = sdCylinder(pos.yxz + vec3(tubeLen1 + tubeclen, -tubeclen - tubeLen2 - 0.6, 0), vec2(tubeDiam/1.85, 0.065));
    chimney = smin(chimney, collar1, 0.15);
    
    // Small rounding at the middle of the broad part  
    float rborder = sdTorus(pos.yxz + vec3(tubeLen1 + tubeclen, -tubeclen - tubeLen2 - tubeLen3 - 0.5, 0.), vec2(tubeDiam2/1.8, 0.01));
    chimney = smin(chimney, rborder, 0.2);
    
    // Inside of the chimney (hole of the chimney)
    float tubeInt = getTubeInt(pos);
    chimney = max(chimney, -tubeInt);
    
    return vec2(chimney, CHIMNEY_OBJ);
}

// Function 463
float map_ver(vec3 pos)
{   
    float fx = 145.*random(1.936*floor(pos.x/fe));
    
    fx = 45.8*random(1.885*floor(pos.x/fe)); 
    pxd = fe*tdd*(1. + 0.45*0.5*sin(pos.y*1.3 + 27.*fx) + 0.3*0.5*sin(pos.y*3.7 + 74.*fx) - 0.25*0.5*sin(pos.y*9.48 + 112.*fx));
    pos.x+= pxd;
    
    vsf = 0.35*tdv*sin(pos.y*4.3 + 31.*fx) - 0.4*tdv*sin(pos.y*5.7 + 58.*fx) - 0.25*tdv*sin(pos.y*8.48 + 38.*fx);
    fr = fr0*(-tdv*0.5 + 1. - 0.5*tdv*vsf);
    
    float angle = twfs*pos.y;
    vec2 d1 = rotateVec(vec2(fr*fds, fr*fds), angle);
    vec2 d2 = d1.yx*vec2(1., -1);
    return min(min(min(map_ver_small(pos, d1, 1.), map_ver_small(pos, d2, 5.)), map_ver_small(pos, -d2, 9.)), map_ver_small(pos, -d1, 13.)); 
}

// Function 464
float map_simple(vec3 pos)
{   
    float angle = 2.*pi*iMouse.x/iResolution.x;
    float angle2 = -2.*pi*iMouse.y/iResolution.y;
    
    vec3 posr = pos;
    posr = vec3(posr.x, posr.y*cos(angle2) + posr.z*sin(angle2), posr.y*sin(angle2) - posr.z*cos(angle2));
    posr = vec3(posr.x*cos(angle) + posr.z*sin(angle), posr.y, posr.x*sin(angle) - posr.z*cos(angle)); 
    
    float d = 1.05;
    float s = atan(posr.y, posr.x);
    
    vec3 flatvec = vec3(cos(s), sin(s), 1.444);
    vec3 flatvec2 = vec3(cos(s), sin(s), -1.072);
     
    float d1 = dot(flatvec, posr) - d;                        // Crown
    d1 = max(dot(flatvec2, posr) - d, d1);                    // Pavillon
    d1 = max(dot(vec3(0., 0., 1.), posr) - 0.35, d1);         // Table
    return d1;
}

// Function 465
vec3 hsluv(float t) {

    const vec3 c0 = vec3(1.112190141256249, 0.2892005552571448, 0.4533874092347643);
    const vec3 c1 = vec3(-2.700459896899569, 1.230016915698432, -8.866692238834126);
    const vec3 c2 = vec3(13.27461904052131, 9.770495221624461, 40.21502987811905);
    const vec3 c3 = vec3(-81.52914608827041, -72.38313527130981, -70.91008175027898);
    const vec3 c4 = vec3(160.2873495871779, 177.6692475972083, 72.98689200890533);
    const vec3 c5 = vec3(-113.7533707882922, -191.0996400048753, -43.90364824070267);
    const vec3 c6 = vec3(25.63678954693401, 74.92090437235763, 10.41364910112663);

    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));

}

// Function 466
vec3 colormap(float t) {
    return PAL(t, vec3(0.5,0.5,0.5), vec3(0.5,0.5,0.5), vec3(1.0,1.0,1.0), vec3(0.0,0.33,0.67));
}

// Function 467
vec2 scale_uv(vec2 uv, vec2 scale, vec2 center) {
    return (uv - center) * scale + center;
}

// Function 468
float map5( in vec3 p )
{
	vec3 q = p*cloudScale;
	float f;
    f  = .50000*noise( q ); q = q*2.02;
    f += .25000*noise( q ); q = q*2.03;
    f += .12500*noise( q ); q = q*2.01;
    f += .06250*noise( q ); q = q*2.02;
    f += .03125*noise( q );
	return clamp( 1.5 - p.y - 2. + 1.75*f, 0., 1. );
}

// Function 469
float mapOLD(vec3 p, float t) {
    p+=vec3(0,2,0);
    float s1 = sph(p, 0.8);
    float s2 = sph(p+vec3(0,0.5,0), 0.8);
    float c1 = cyl2(p.xzy,0.3);
    float base = smax(smin(s1,s2,0.1),-c1,0.2);
    
    for(int i=0; i<200; ++i) {
        float t3 = t+float(i);
        vec3 off=vec3(sin(t3), cos(t3), cos(t3*1.42));
    	base=smax(base, -sph(p+off, 0.5),  0.05);   
    }
    
    for(int i=0; i<200; ++i) {
        float t3 = t*0.2+float(i);
        vec3 off=vec3(sin(t3)*2.0, cos(t3)*1.5, cos(t3*1.42)*2.2);
    	base=smin(base, sph(p+off, 0.2),  0.05);   
    }
    
    return base;
    //return sph(p, 0.5);
}

// Function 470
float getSphereMappedTexture(in vec3 pointOnSphere)
{
    /* Test to determine which face we are drawing on.
     * Opposing faces are taken care of by the absolute
     * value, leaving us only three tests to perform.
     */
    vec2 st = (
        insideBounds(sphereToCube(pointOnSphere)) +
        insideBounds(sphereToCube(pointOnSphere.zyx)) +
        insideBounds(sphereToCube(pointOnSphere.xzy)));
    
    st *= 12.0;
    float k = GetWaveDisplacement(vec3(st.x,0.0,st.y))*0.5;
    k = clamp(k,0.0,1.0);
	return 1.0-k;
    //return textureFunc(st);
}

// Function 471
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir ) {
    if (iFrame > 1) discard; // cache
    vec3 p = 1.5 * rayDir;
    fragColor.x = 0.;
    for (float i = 0.; i < 5.; i++) {
        float c = craters(0.4 * pow(2.2, i) * p);
        float noise = 0.4 * exp(-3. * c) * FBM(10. * p);
        float w = clamp(3. * pow(0.4, i), 0., 1.);
		fragColor.x += w * (c + noise);
	}
    fragColor.x = pow(fragColor.x, 3.);
}

// Function 472
float mapSeed(vec2 f)
{
    return min(LIGHT_DIST2, mapSeedNoLight(f));
}

// Function 473
vec3 yuv2rgb(vec3 yuv)
{
    vec3 rgb;
    rgb.r = yuv.x * 1.0 + yuv.y *  0.0   + yuv.z *  1.4;
	rgb.g = yuv.x * 1.0 + yuv.y * -0.343 + yuv.z * -0.711;
	rgb.b = yuv.x * 1.0 + yuv.y *  1.765 + yuv.z *  0.0;
    return rgb;
}

// Function 474
vec2 uv_aa_smoothstep( vec2 uv, vec2 res, float width )
{
    uv = uv * res;
    vec2 uv_floor = floor(uv + 0.5);
    vec2 uv_fract = fract(uv + 0.5);
    vec2 uv_aa = fwidth(uv) * width * 0.5;
    uv_fract = smoothstep(
        vec2(0.5) - uv_aa,
        vec2(0.5) + uv_aa,
        uv_fract
        );
    
    return (uv_floor + uv_fract - 0.5) / res;
}

// Function 475
void mainCubemap( out vec4 C, in vec2 U, in vec3 pos, in vec3 dir ){
    
    if(iFrame >= 3){
        C.rgb = texture(iChannel0,dir).rgb;
        return;
    }
    if(iFrame == 1){
        vec3 ro = vec3(100, 0.0, 400.0);
        C.rgb = render(vec3(ro.x,  terrainH( ro.xz) + 11.0*SC, ro.z),dir);
    }
    else if(iFrame == 2){
        vec3 axis = getAxis(dir);
        vec3 uv = GetUVW(axis,dir);
        vec3 MipDir = GetMipMapUVW_Dir2(uv,axis);
        float roughness = max((30.-ID)/ID_Range.z,0.);
        
        C.rgb = PrefilterEnvMap(roughness,MipDir);
    }
    
}

// Function 476
vec4 normalMap(vec2 uv) { return heightToNormal(normalChannel, normalSampling, uv, normalStrength); }

// Function 477
vec3 hsluvToLch(vec3 tuple) {
    tuple.g *= hsluv_maxChromaForLH(tuple.b, tuple.r) * .01;
    return tuple.bgr;
}

// Function 478
vec3 Tonemap_Uchimura_RGB(vec3 v) {
  return vec3(Tonemap_Uchimura(v.r), Tonemap_Uchimura(v.g), Tonemap_Uchimura(v.b));
}

// Function 479
vec4 rgbToHsluv(vec4 c) {return vec4( rgbToHsluv( vec3(c.x,c.y,c.z) ), c.a);}

// Function 480
float map(vec3 p){

    // Twist the scene about the Z-axis. It's an old trick. Put simply, you're
    // taking a boring scene object and making it more interesting by twisting it. :)
    p.xy *= rot(p.z*ZTWIST);
    
    // Produce a repeat object. In this case, just a simple lattice variation.
    float d =  lattice(p); 
    
    // Bound the lattice on the outside by a boxed column (max(abs(x), abs(y)) - size) 
    // and smoothly combine it with the lattice. Note that I've perturbed it a little 
    // by the lattice result (d*.5) in order to add some variation. Pretty simple.
    p = abs(p);
    d = sminP(d, -max(p.x, p.y) + 1.5 - d*.5, .25);
     
    return d*.7;
}

// Function 481
Model map(vec3 p) {
    
    // Spin the whole model
    spin(p);
    
	float subdivisions = animSubdivitions(1., 10.);
	vec3 point = geodesicTri(p, subdivisions);

	float sphere = length(p - point) - .195 / subdivisions; 
    
    // Indicate point's position
    vec3 color = point * .5 + .5;

	return Model(sphere, color);
}

// Function 482
float map_wall_1(vec3 pos)
{
    return -pos.x;
}

// Function 483
void mainCubemap( out vec4 fragColour, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    //if ( iFrame > 1 ) discard;
    
	// i was planning to do a complicated volumetric cloud thing here but fake ones look fine.

    // Ray direction as color
    vec3 col;/* = mix( vec3(.5,.7,1), vec3(2), smoothstep(.3, 1., cloudSDF(rayDir*vec3(15,30,15)) ) );
    col = mix( col, vec3(.2,.1,0), smoothstep(.3, 1., cloudSDF(rayDir*vec3(15,33,15)) ) );
    
    col = mix( col, vec3(.3,.35,.4), smoothstep( .0, -.02, rayDir.y ) );*/

	col = GetFogColour(rayDir);
	col = mix( col, vec3(100,8,.8), smoothstep(.99994,.99997,dot(rayDir,sunDir)) );//pow(max(0.,dot(rayDir,sunDir)),20000.) );
    
    // Output to cubemap
    fragColour = vec4(col,1.0);
}

// Function 484
vec3 hsluvToRgb(vec3 tuple) {  return lchToRgb(hsluvToLch(tuple)); }

// Function 485
vec2 pos2uv(Camera cam, vec3 pos){
    vec3 dir = normalize(pos - cam.pos) * inverse(rotationMatrix(cam.rot));
    return dir.xy * cam.focalLength / dir.z;
}

// Function 486
float map(vec3 p)
{
    p.x += sin(p.y*4.+time+sin(p.z))*0.15;
    float d = length(p)-1.;
    float st = sin(time*0.42)*.5+0.5; 
    const float frq = 10.;
    d += sin(p.x*frq + time*.3 + sin(p.z*frq+time*.5+sin(p.y*frq+time*.7)))*0.075*st;
    
    return d;
}

// Function 487
float map(vec3 p)
   {
       //p = fract(p);
     p.y+=sin(p.z*2.+iTime)/200.;
      p.x+=sin(p.y*2.+iTime)/200.;
    float re = 0.0;
       float scale = 1.;
       
       for(int i=0; i<2 ;i++)
       {
        
        scale*=3.;
        re = max(re, -rcScale(p, scale)  );

       }
        
    return re;//rect(p, vec3(2.5));
    }

// Function 488
vec2 map(vec3 pos, bool inside, bool noroom)
{
    vec2 res;
    float screen = map_screen(pos);
    if (inside)
       res = vec2(-screen, SCREEN_OBJ);
    else
    {
       res = vec2(screen, SCREEN_OBJ);
        
       float support = map_support(pos);
       res = opU(res, vec2(support, SUPPORT_OBJ));        
        
       if (!noroom)
       {
          float floor = map_floor(pos);
          res = opU(res, vec2(floor, FLOOR_OBJ));           
           
          float room = map_room(pos);
          res = opU(res, vec2(room, ROOM_OBJ));
       }
       
       #ifdef show_leds
       vec2 rled = vec2(map_rled(pos), RLED_OBJ);
       vec2 gled = vec2(map_gled(pos), GLED_OBJ);

       res = opU(res, rled);
       res = opU(res, gled);
       #endif        
    }
    return res;
}

// Function 489
vec2 map1ds(vec2 z)
{
  float r = length(z);
  float t = atan(z.y,z.x);
   if (t < -.25*PI) t += 2.0*PI;
    
   if (t < .25*PI) {
      return vec2(r,(4.0/PI)*r*t );
    } 
    else if (t < .75*PI) {
      return vec2(-(4.0/PI)*r*(t-.5*PI), r);
    } 
    else if (t < 1.25*PI) {
      return vec2(-r, -(4.0/PI)*r*(t-PI));
    } 
    else {
      return vec2((4.0/PI)*r*(t-1.5*PI), -r);
    }
}

// Function 490
vec3 GetBumpMapNormal (vec2 uv, mat3 tangentSpace)
{    
	float delta = -1.0/512.0;
	float A = texture(iChannel1, uv + vec2(0.0, 0.0)).x;
	float B = texture(iChannel1, uv + vec2(delta, 0.0)).x;
    float C = texture(iChannel1, uv + vec2(0.0, delta)).x;    
    
	vec3 norm = normalize(vec3(A - B, A - C, 0.15));
	
	return normalize(tangentSpace * norm);
}

// Function 491
void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir) {
    vec3 lmn;
    int pageDst;
    lmnFromVCube(rayDir, pageDst, lmn);

    if (pageDst == 1) {
        fragColor = doPage1(lmn);
    } else if (pageDst == 2) {
        fragColor = doPage2(lmn);
    } else {
        discard;
    }
}

// Function 492
float map(vec3 p){
    
    // Back wall
    float wall = -p.z + .01; // Thick wall: abs(p.z - .2) - .21;
     
    // Truchet object and animated metallic balls: This is just a
    // standard 2D animated Truchet with an extruded factor. If you're
    // not sure how it works, myself and others have plenty of 
    // animated Truchet examples on Shadertoy to refer to.
    //
    // Grid construction: Cell ID and local cell coordinates.
    const vec2 GSCALE = vec2(1./3.);
    const vec2 sc = 1./GSCALE, hsc = .5/sc;    
    vec2 iq = floor(p.xy*sc) + .5;    
    vec2 q = p.xy - iq/sc; // Equivalent to: mod(p.xy, 1./sc) - .5/sc;
    
    // Flip random cells. This effectively rotates random cells,
    // but in a cheaper way.
    float rnd = hash21(iq + .37);
    if(rnd<.5) q.y = -q.y;
      
    // Circles on opposite square vertices.
    vec2 d2 = vec2(length(q - hsc), length(q + hsc));
    // Using the above to obtain the closest arc.
    float crv = abs(min(d2.x, d2.y) - hsc.x);
    
    // Flipping the direction on alternate squares so that the animation
    // flows in the right directions -- It's a standard move that I've
    // explained in other examples.  
    float dir = mod(iq.x + iq.y, 2.)<.5? -1. : 1.;
    // Using repeat polar coordinates to create the moving metallic balls.
    vec2 pp = d2.x<d2.y? vec2(q - hsc) : vec2(q + hsc);
    pp *= rot2(iTime*dir); // Animation occurs here.
    float a = atan(pp.y, pp.x); // Polar angle.
    a = (floor(a/6.2831853*8.) + .5)/8.; // Repeat central angular cell position.
    // Polar coordinate.
    vec2 qr = rot2(a*6.2831853)*pp; 
    qr.x -= hsc.x;
     
    // Ridges, for testing purposes.
    //crv += clamp(cos(a*16. + dir*iTime)*2., 0., 1.)*.003;
    
    // A rounded square Truchet tube. Look up the torus formula, if you're
    // not sure about this. However, essentially, you place the rounded curve
    // bit in one vector position and the Z depth in the other, etc. Trust me,
    // it's not hard. :)
    float tr = length(vec2(crv, (p.z) + .05/2. + .02)) - .05;
    //float tr = sBoxS(vec2(crv, (p.z) + .05/2. + .02), vec2(.05, .05), .035);
    
    
 
    // Metallic elements, which includes the joins, metal ball joints
    // and the tracks they're propogating along. This operation needs to be
    // performed prior to hollowing out the tubes. See below.
    q = abs(abs(q) - .5/sc);
    float mtl = min(q.x, q.y) - .01;
    mtl = max(max(mtl, tr - .015), -(tr - .005));
    
    // Adding in the railing.
    float rail = tr + .035 + .01;
    

    // 3D ball position.
    vec3 bq = vec3(qr,  p.z + .05/2. + .02);
    //float ball = max(length(bq.zx) - .02, abs(bq.y) - .03);
    float ball = length(bq) - .02; // Ball.
    //ball = abs(ball + .005) - .005; // Hollow out.
    
    float mtl2 = ball;//max(ball, -(rail - .0025));
    mtl = min(mtl, rail);
    
    // Hollowing out the Truchet tubing. If you don't do this, it can cause
    // refraction issues, but I wanted the tubes to be hollow anyway.
    tr = max(tr, -(tr + .01 + .01));

    // Debug: Take out the glass tubing.
    //tr += 1e5;
    
    // Storing the object ID.
    vObjID = vec4(wall, tr, mtl, mtl2);
    
    // Returning the closest object.
    return min(min(wall, tr), min(mtl, mtl2));
 
}

// Function 493
vec3 envMap(vec3 rd){
    
  
    float c = n3D(rd*4.)*.66 + n3D(rd*8.)*.34; // Noise value.
    c = smoothstep(.3, 1., c); // Darken and add contast for more of a spotlight look.
    
    return vec3(c*c*c, c*c, c); // Simple, cool coloring.
    
}

// Function 494
float map(vec3 pos)
{
    float angle0 = atan(pos.x, pos.z);
    angle = angle0 + rot;
    float angle2 = angle + pos.y*1.1;
    angle3 = fract(5.*angle/pi) - 0.3;
    float angle4 = angle0*3. - iTime*2. - angle3 + 0.3;
    
    pos.xy = rotateVec(pos.xy, 0.15*smoothstep(40., 200., iTime)*cos(rot*2.));
    pos.zy = rotateVec(pos.zy, 0.15*smoothstep(40., 200., iTime)*sin(rot*2.));
    
    //pos.z+= 0.3*pos.y*sin(angle);
    return length(pos.xz)*0.9
                 *(0.8 - (0.43 + 0.05*(smoothstep(-0.9, 0.9, sin(24.*angle))))
                 *smoothstep(0.16, 0.08, abs(pos.y)))
                 + 0.2*smoothstep(0.21, 0.25, abs(pos.y))
                 - 0.5
                 - 0.045*sin(angle4)
                 *smoothstep(0.245, 0.19, length(vec2(angle3, 3.8*abs(pos.y - scrpos))))
                 *(1. + 0.4*smoothstep(0.06, 0.072, abs(angle3 - 0.16 + 2.8*abs(pos.y - scrpos + 0.06))))
                 - 0.8*smoothstep(0.58, 1.35, abs(pos.y))
                 *smoothstep(1.45, 1.27, abs(pos.y))
                 *abs(sin(angle2*6.5))
                 - 0.04*smoothstep(0.58, 0.62, abs(pos.y))*smoothstep(1.46, 1.42, abs(pos.y))
                 + 0.9*pow(abs(pos.y*0.36) + 1.*smoothstep(1.6, 2.3, abs(pos.y*1.01)), 2.6);
}

// Function 495
float map(vec3 p) {
	float d = 1.0E10;
	horizon=0;
	p.z=p.z+camera_pos;

	d =  disp(p,1.0)+sdSphere(p+   vec3(0.0,0.0,sin(T*5.234)),2.0) ;
    for (int x=0; x<BLOBS; x++) {
        float ff=float(x);
        float where=mod(T*1.22+ff*ff,2.0*PI);
        if ((where<PI*0.5 || (where>PI && where<1.5*PI)) ) {
                float d2 = ( 1.0-(0.5+sin(2.0*where+.5*PI)/2.0) )*disp(p,1.0) +sdSphere(p + vec3(0.0,0.0,sin(T*5.234)) +  ( rotateX(ff+T*0.33) * rotateY(ff+T*0.45) * rotateZ(ff+T*0.56) * vec4( 0.0+sin(where)*2.0, 0.0+sin(where)*2.0,  sin(where)*10.0  ,  1.0  )).xyz    ,   1.33*( 1.0-(0.5+sin(2.0*where+1.5*PI)/2.0) ));    d = smin(d,d2,0.3);
        }
    }


    d = min(1.0E10 , d );
    if (d>= 1.0E10) horizon=1;
	return d;

}

// Function 496
vec3 ToneMapFilmic_Hejl2015(vec3 hdr, float whitePt) {
    vec4 vh = vec4(hdr, whitePt);
    vec4 va = 1.425 * vh + 0.05;
    vec4 vf = (vh * va + 0.004) / (vh * (va + 0.55) + 0.0491) - 0.0821;
    return vf.rgb / vf.www;
}

// Function 497
vec3 ToneMappingHQ( vec3 col )
{
    const float a = 1.; // exposure
    const float p = 3.; // top curve - 1. = very smooth, higher = more squashed
    const float r = 1.; // overflow - 1. = no clipping, higher => more clipping
    
    const float d = .05; // desaturation of bright colours
    float brightness = dot(col,componentWeights);
    col = mix( vec3(brightness), col, 1./(1.+d*brightness) );
    col = r*pow(max(vec3(.0000001),1.-1./(1.+pow(a*col/r,vec3(p)))),vec3(1./p));
    
	return col;
}

// Function 498
vec2 map(vec3 q3){
    vec2 res = vec2(100.,0.);

    float k = 5.0/dot(q3,q3); 
    q3 *= k;

    q3.z += speed;

    vec3 qm = q3;
    vec3 qd = q3+hlf;
    qd.xz*=t90;
    vec3 qid=drep(qm);
    vec3 did=drep(qd);
    
    float ht = hash21(qid.xy+qid.z);
    float hy = hash21(did.xz+did.y);
    
    float chk1 = mod(qid.y + qid.x,2.) * 2. - 1.;
    float chk2 = mod(did.y + did.x,2.) * 2. - 1.;

    // truchet build parts
    float thx = .115;
    float thz = .200;

    if(ht>.5) qm.x *= -1.;
    if(hy>.5) qd.x *= -1.;

    float t = truchet(qm,vec3(hlf,hlf,.0),vec2(hlf,thx));
    if(t<res.x) {
        sid = qid;
        hit = qm;
        chx = chk1;
        sdir = ht>.5 ? -1. : 1.;
        res = vec2(t,2.);
    }

    float d = truchet(qd,vec3(hlf,hlf,.0),vec2(hlf,thz));
    if(d<res.x) {

        sid = did;
        hit = qd;
        chx = chk2;
        sdir = hy>.5 ? -1. : 1.;
        res = vec2(d,1.);
    }

    float mul = 1.0/k;
    res.x = res.x * mul / shorten;
    
    return res;
}

// Function 499
vec2 map3sd(vec2 z)
{
  vec2 z2 = 0.5*z*z;
  return vec2(sqrt(1.0-z2.y)*z.x, sqrt(1.0-z2.x)*z.y);
}

// Function 500
vec3 GetUVW(vec3 axis,vec3 dir){
    return dir/dot(axis,dir)-axis;
}

// Function 501
float distMap(vec3 r) {
    float cubeLen = length(max(abs(r)-maxSize,0.0));
    
    if (cubeLen <= prec) {
        vec3 rIn = r;
        rIn += texture(iChannel2,normalize(rIn)).xyz*6.0;
        return min(prec,length(max(abs(rIn)-size,0.0)));
    }
    
    return cubeLen;
}

// Function 502
vec4 map(in vec2 p, in vec2 dir) {
    vec3 v = voronoi(p*2.0, dir)*0.5;
    return vec4(v, disp(v.yz));
}

// Function 503
float map_lamps(vec3 pos)
{
    pos.y = mod(pos.y, floor_height);
    pos-= vec3(-staircase_length/2. + floor_width/2., floor_height - floor_thickness - lamp_thickness/2., 0.);
    return sdCylinder(pos, vec2(lamp_radius, lamp_thickness/2.), 0., lamp_roundness);
}

// Function 504
vec4 map4(vec3 p, out float k) {
  vec4 p4 = inverseStereographic(p,k);
  // Do a one sided quaternion rotation, otherwise known
  // as a Clifford Translation, which seems appropriate.
  if (key(CHAR_R)) p4 = qmul(p4,quat);
  return p4;
}

// Function 505
vec4 hpluvToRgb(vec4 c) {return vec4( hpluvToRgb( vec3(c.x,c.y,c.z) ), c.a);}

// Function 506
vec2 ringUv(vec2 latLon, float angle, float centerLat)
{
    // latlon : latitude / longitude
    // angle: horizontal angle covered by one rep of the pattern over the equator / angular height of the band
    // centerLat : center latitude of the band
    
    
    // Compute y coords by remapping latitude 
    float halfAngle = angle * 0.5;
    float y = remap(latLon.y, centerLat - halfAngle,  centerLat + halfAngle);
    
    float centerRatio = cos(centerLat); // stretch of the horizontal arc of the pattern at the center of the 
   										// band relative to the equator
    
    float centerAngle = angle / centerRatio; // local longitudianl angle to compensate for stretching at the center of the band. 
    
    float nbSpots = floor(pi2 / centerAngle); // with new angle, how many pattern can we fit in the band?
    float spotWidth = pi2 / nbSpots;          // and what angle would they cover (including spacing padding)?
    
    float cellX = fract(latLon.x / spotWidth); // what would be the u in the current cell then?
                  
                  
    float x = (0.5 - cellX) * (spotWidth / centerAngle); // compensate for taper
    x *= (cos(latLon.y) / centerRatio) * 0.5 + 0.5;
    
    vec2 uvs = vec2(x + 0.5, y);
    return uvs;
}

// Function 507
vec2 map2sd(vec2 z)
{

  vec2  z2 = z*z;
  float s  = sqrt(1. - (z2.x * z2.y)/(z2.x+z2.y+epsilon));
  return z*s;  
}

// Function 508
float map(vec3 p){

    
    float sf = cellTile(p/2.5);
    
    // Tunnel bend correction, of sorts. Looks nice, but slays framerate, which is disappointing. I'm
    // assuming that "tan" function is the bottleneck, but I can't be sure.
    //vec2 g = (path(p.z + 0.1) - path(p.z - 0.1))/0.2;
    //g = cos(atan(g));
    p.xy -= path(p.z);
    //p.xy *= g;
  
    // Round tunnel.
    // For a round tunnel, use the Euclidean distance: length(p.xy).
    return 1.- length(p.xy*vec2(0.5, 0.7071)) + (0.5-sf)*.35;

    
/*
    // Rounded square tunnel using Minkowski distance: pow(pow(abs(tun.x), n), pow(abs(tun.y), n), 1/n)
    vec2 tun = abs(p.xy)*vec2(0.5, 0.7071);
    tun = pow(tun, vec2(8.));
    float n =1.-pow(tun.x + tun.y, 1.0/8.) + (0.5-sf)*.35;
    return n;//min(n, p.y + FH);
*/
    
/*
    // Square tunnel.
    // For a square tunnel, use the Chebyshev(?) distance: max(abs(tun.x), abs(tun.y))
    vec2 tun = abs(p.xy - path(p.z))*vec2(0.5, 0.7071);
    float n = 1.- max(tun.x, tun.y) + (0.5-sf)*.5;
    return n;
*/
 
}

// Function 509
vec3 envMap(vec3 rd, vec3 n){
    
    vec3 col = tex3D(iChannel0, rd, n);
    col = smoothstep(.15, .5, col);
    #ifdef WARM
    col *= vec3(1.35, 1, .65);
    #endif
    //col = col*.5 + vec3(1)*pow(min(vec3(1.5, 1, 1)*dot(col, vec3(.299, .587, .114)), 1.), vec3(1, 3, 10))*.5; // Contrast, coloring. 
    
    return col;

}

// Function 510
float map(vec3 p){
    
    // Wrap the tunnel (polar mapped hexagonal pylons) around the path.
    p.xy -= path(p.z);
    
    // Scaling along Z. Changing this value requires tile diameter, tile number and
    // tunnel width adjustments.
    const float zScale = 1.5;
    
    // Number of pylon segments spread around the tunnel. Actually, the final count is
    // twice this amount, due to rendering two pylons side by side. The other two are
    // rendered in front in the X direction.
    const float aNum = 8.;
    

    // Radius of the tunnel. The radial polar coordinate with be edged out by that amount.
    const float tunRad = 1.63;
    
    // Effectively the length of the blocks. Set to something deep enough to block out all
    // the light. Set it to something like ".1" to see thin tiles.
    const float blockRad = 1.;
    
    
    // Just an extra factor that controls random height.
    float rndFactor = .5;
    
    
    // Hexagonal block dimensions. Play around with them to see what they do.
    const vec3 wd = vec3(blockRad, .235*zScale, .235*zScale);
    
    // Holding vector for the cylindrical coordinates. 
    vec3 pC;
    
    // A cheap way to polar map a squarish tunnel. Simply mutate the the regular tunnel
    // wall positions a bit. If you warp things too much, the blocks will get too 
    // distorted, but this here isn't too noticeable.
    //vec2 mut = pow(abs(p.xy), vec2(4))*.125 + .875; // Etc.
    vec2 mut = p.xy*p.xy*.125 + .875;
    p.xy *= mut*vec2(1, 1.2);
    
    //p += sin(p*2. + cos(p.yzx*2.))*.1; // Adding bumps. Too much for this example.
    
    // Scaling Z... This gave me more trouble than I care to admit. Scale here, then
    // scale back when doing the repetition. Simple... now. :)
    p.z /= zScale;
    
    // Relative Z distances of the four tiles. The bit on the end belongs there, but
    // I can't remember why. Some kind of correction in order to get the get correct height
    // values for wall tiling... or something. I should write these things down. :D
    vec4 fPz = floor(vec4(0, .5, .25, .75) + p.z) - vec4(0, .5, .25, .75);
    

    // Angle and angular index.
    float a, ia;
    

    //// Hexagon 1.    
    // Standard polar mapping stuff.
    a = atan(p.y, p.x);
    ia = floor(a/TAU*aNum);
    rnd.x = hash21(vec2(ia, fPz.x));
    #ifdef RIGID_BLOCKS
    p.xy = rot2((ia + .5)*TAU/aNum)*p.xy;
    pC = vec3(p.xy, p.z);
    //pp = vec3(length(p.xy)*vec2(cos((ia + .5)*TAU/aNum), sin((ia + .5)*TAU/aNum)), p.z);
    #else
    p.xy = rot2(a)*p.xy;
    pC = vec3(p.xy, p.z); pC.y = convert(a, aNum);
    #endif
    // First entry when you perform: pp.x = mod(pp.x, rad) - rad/2.;
    pC.x -= tunRad - rnd.x*rndFactor + blockRad; // Base tunnel width.
    pC.z = (mod(pC.z, 1.) - .5)*zScale; // Repetition along the tunnel.
    
    sh.x = objectDetail(pC, wd, rnd.x, spike4.x); // The hexagon pylon object and spike.
    

    //// Hexagon 2.   
    rnd.y = hash21(vec2(ia + .5, fPz.y));
    #ifdef RIGID_BLOCKS    
    pC = vec3(p.xy, p.z + .5);
    #else
    pC = vec3(p.xy, p.z  + .5); pC.y = convert(a, aNum);
    #endif
    pC.x -= tunRad - rnd.y*rndFactor + blockRad; // Tunnel width.
    pC.z = (mod(pC.z, 1.) - .5)*zScale;
    
    sh.y = objectDetail(pC, wd, rnd.y, spike4.y);
    
    //p.xy = q.xy;
	//// Hexagon 3.  
    #ifdef RIGID_BLOCKS
    //p.xy = q.xy;
    p.xy = rot2(-(ia + .5)*TAU/aNum + 3.14159/aNum)*p.xy;
    #else
    p.xy = rot2(-a + 3.14159/aNum)*p.xy;
    #endif
    a = atan(p.y, p.x);
    ia = floor(a/TAU*aNum);
    
    rnd.z = hash21(vec2(ia, fPz.z));
    #ifdef RIGID_BLOCKS
    p.xy = rot2((ia + .5)*TAU/aNum)*p.xy;
    pC = vec3(p.xy, p.z + .25);
    #else
    p.xy = rot2(a)*p.xy;
    pC = vec3(p.xy, p.z + .25); pC.y = convert(a, aNum);
    #endif
    pC.x -= tunRad - rnd.z*rndFactor + blockRad; // Tunnel width.
    pC.z = (mod(pC.z, 1.) - .5)*zScale;
    sh.z = objectDetail(pC, wd, rnd.z, spike4.z);
   
  
	//// Hexagon 4. 
    rnd.w = hash21(vec2(ia + .5, fPz.w));
    #ifdef RIGID_BLOCKS
    pC = vec3(p.xy, p.z + .75);
    #else
    pC = vec3(p.xy, p.z + .75); pC.y = convert(a, aNum);
    #endif
    pC.x -= tunRad - rnd.w*rndFactor + blockRad; // Tunnel width.
    pC.z = (mod(pC.z, 1.) - .5)*zScale;
    sh.w = objectDetail(pC, wd, rnd.w, spike4.w);
    
    // Determining the minimum hexagon pylon distance, then returning it.
    vec2 hx2 = min(sh.xy, sh.zw);
    return min(hx2.x, hx2.y)*.85;
 
}

// Function 511
float map(vec3 p){
 
    
    // Retrieve the 2D surface value from a cube map face.
    float sf2D = surfFunc2D(p);
    
     
    // Path function. Not used here.
    //vec2 pth = path(p.z); 

    
    // Tunnel. Not used here.
    //float tun = 1. - dist((p.xy - pth)*vec2(.7, 1));

    // Mover the mirrored ball object.
    vec3 q = moveBall(p);
    
     
    // Terrain.
    float ter = p.y - sf2D*.5;
    
    // Place a crater beneath the object.
    vec3 q2 = p - vec3(0, 1, 0) - vec3(0, 5. - .55 - 1., 0);
    ter = smax(ter, -(length(q2) - 5.), .5);
    ter += (.0 - sf2D*.5); 
    
    
 
    // Hollowing the tunnel out of the terrain. Not used here.
    //ter = smax(ter, tun, 3.);
    
    // The polyhedral object.
        
    // Face, line and vertex distances. 
    float face = 1e5, line = 1e5, vert = 1e5;

 
    
    #ifdef PENTAKIS_ICOSAHEDRON
    
        // A Pentakis icosahedron: Like an icosahedron, but with 80 sides.
    
        // Construct a regular 20 sided icosahedron, then use the vertices to 
        // subdivide into four extra triangles to produce an 80 sided Pentakis
        // icosahedron. Subdivision is achieved by using the known triangle 
        // face vertex points to precalculate the edge points and triangle center 
        // via basic trigonometry. See e0, e1, e2 above.
        //
        // On a side note, I'd imagine there's a way to fold space directly into a 
        // Pentakis icosahedron, but I got lazy and took the slower subdivided 
        // icosahedron route. If anyone knows how to do it more directly, feel free 
        // to let me know.


        // Local object cell coordinates.
        vec3 objP = opIcosahedron(q);

        // Vertices.
        vert = min(vert, length(objP - v0) - .05); 
        vert = min(vert, length(objP - e0) - .05); 

        // Lines or edges.
        line = min(line, sdCapsule(objP, v0, e0) - .02);
        line = min(line, sdCapsule(objP, e0, e2) - .02);

        float ndg = .97;

        // Vertex triangle facets -- Due to the nature of folding space,
        // all three of these are rendered simultaneously.
        face = min(face, udTriangle(objP, v0*ndg, e0*ndg, e2*ndg) - .03);
        // Middle face.
        face = min(face, udTriangle(objP, e0*ndg, e1*ndg, e2*ndg) - .03);
    
    #else
    
        // The second polyhedral object option:
        //
        // This is an exert from Knighty's awesome Polyhedras with control example, here:
        // https://www.shadertoy.com/view/MsKGzw
        //
        // Here's a very brief explanation: Folding space about various planes will produce various
        // objects -- The simplest object would be a cube, where you're folding once about the YZ, XZ, 
        // and XY planes: p = abs(p); p -= vec3(a, b, c); dist = max(max(p.x, p.y), p.z);
        //
        // Things like icosahedrons require more folds and more advanced plane calculations, but the
        // idea is the same. It's also possible to mix objects, which is what Knighty has cleverly 
        // and elegantly done here. In particular, a Triakis icosahedron (bas = vec3(1, 0, 0), an 
        // icosahedron (bas = vec3(0, 1, 0) and a dodecahedron (bas = vec3(0, 0, 1). Mixtures, like 
        // the default (bas = vec3(1), will give you a compounded mixture of three platonic solids, 
        // which each have their own names, but I'll leave you to investigate that. :)

        // Setup: Folding plane calculation, etc. I've made some minor changes, but it's essesntially
        // Knighty's original code.
        //
        // Animating through various platonic solid compounds.
        //vec3 bas = vec3(sin(iTime/4.)*.5 + .5, cos(iTime*1.25/4.)*.5 + .5,  cos(iTime/1.25/4.)*.5 + .5);
        // A nice blend of all three base solids.
        const vec3 bas = vec3(1);

        vec3 pbc = vec3(scospin, 0, .5); // No normalization so 'barycentric' coordinates work evenly.
        vec3 pca = vec3(0, scospin, cospin);
        //U, V and W are the 'barycentric' coordinates. Not sure if they are really barycentric...
        vec3 pg = normalize(mat3(pab, pbc, pca)*bas); 
        // For slightly better DE. In reality it's not necesary to apply normalization. :) 
        pbc = normalize(pbc); pca = normalize(pca);
        
        p = q; // Initial coordinates set to the ball's position.

        // Fold space.
        for(int i = 0; i<5; i++){
            p.xy = abs(p.xy); // Fold about xz and yz planes.
            p -= 2.*min(dot(p, nc), 0.)*nc; // Fold about nc plane.
        }

        // Analogous to moving local space out to the surface.
        p -= pg;

        // Object face distance.
        float d0 = dot(p, pab), d1 = dot(p, pbc), d2 = dot(p, pca);
        face = max(max(d0, d1), d2);
        //face -= abs((d0 + d1 + d2)/3. - face)*.25; // Subtle surface sparkle.

        // Object line distance.
        float dla = length(p - min(p.x, 0.)*vec3(1, 0, 0));
        float dlb = length(p - min(p.y, 0.)*vec3(0, 1, 0));
        float dlc = length(p - min(dot(p,nc), 0.)*nc);
        line = min(min(dla, dlb), dlc) - .025;

        // Vertices.
        vert = length(p) - .05;
    #endif
 
   
    
    // Storing the terrain, line, face and vertex information.
    vRID = vec4(ter, line, face, vert);

    // Return the minimum distance.
    return min(min(ter, line), min(face, vert));
 
}

// Function 512
vec3 CubemapRayDir(in vec2 fragCoord) 
{
    vec2 t = fragCoord.xy*vec2(4.0, 2.0) / iResolution.xy;
    vec3 n = CubemapNormal(floor(t));
    
    float g = 4.0 / iResolution.x;
    float vo = iResolution.x*0.5 - iResolution.y;
    
    vec2 xzp = fract(min(vec2(4.0, 0.99999), fragCoord.xy * g));
    
    vec2 ypp = vec2(min(0.99999, fragCoord.x * g), max(1.0, (fragCoord.y + vo) * g));
    vec2 ypn = vec2(max(3.0,     fragCoord.x * g), max(1.0, (fragCoord.y + vo) * g));
    vec2 yp = fract(ypp * step(-0.5, n.y) + ypn * (1.0 - step(-0.5, n.y)));
    
    vec2 p = (xzp * (1.0 - abs(n.y)) + yp * abs(n.y)) - 0.5;
    
    vec3 px = vec3(0.5*n.x, p.y, -p.x*n.x) * abs(n.x);
    vec3 py = vec3(p.x*n.y, 0.5*n.y, -p.y) * abs(n.y);
    vec3 pz = vec3(p.x*n.z, p.y, 0.5*n.z) * abs(n.z);
    
   	vec3 rd = px + py + pz; 
    return normalize(rd);
}

// Function 513
map_result Map( vec3 p )
{
    //Grid layout of spheres
    // Metallic on Y axis
    // Roughness on X axis
    vec3 SpherePosition = SphereInitialPosition;
    //	Compute steps for spheres spacing and metallic/roughness value
    float SphereXStep = (SphereRadius * float(SPHERE_GRID_NUM_COLS+10)) / float(SPHERE_GRID_NUM_COLS);
    float SphereYStep = (SphereRadius * float(SPHERE_GRID_NUM_ROWS+10)) / float(SPHERE_GRID_NUM_COLS);
    float MetallicStep = 0.9 / float(SPHERE_GRID_NUM_COLS);
    float RoughnessStep = 0.9 / float(SPHERE_GRID_NUM_ROWS);
    
    map_result Result;
    Result.dp = RAY_MARCHING_MAX_DISTANCE;
    Result.Material.BaseColor = vec3(1.,0.,0.);
    Result.Material.Metallic = 0.1;
    Result.Material.Roughness = 0.1;
    
    float NewDp;
    for ( int i = 0; i < SPHERE_GRID_NUM_COLS; ++i)
    {
        SpherePosition = SphereInitialPosition - 10. * UpVector;
        SpherePosition -= float(i) * SphereXStep * RightVector;
        for ( int j = 0; j < SPHERE_GRID_NUM_ROWS; ++j )
        {
            SpherePosition += SphereYStep * UpVector;
            NewDp = Sphere( p - SpherePosition, SphereRadius );
            if(Result.dp > NewDp)
            {
                Result.dp = NewDp;
                Result.Normal = normalize(p - SpherePosition); 
                Result.Material.Metallic = 0.1 + float(j) * MetallicStep;
                Result.Material.Roughness = 0.1 + float(i) * RoughnessStep;
            }
         }
    }
    
    #if USE_AREA_LIGHT
    NewDp = Sphere( p - LightPosition, LightRadius );
    if ( Result.dp > NewDp )
    {
        Result.dp = NewDp;
        Result.Normal = normalize( p - LightPosition );
        Result.Material.Metallic = 0.001;
        Result.Material.Roughness = 1.;
        Result.Material.BaseColor = vec3(10.,10.,10.);
    }    
    #endif
    return (Result) ;  
}

// Function 514
float MapTerrainSimple( vec3 p)
{
  float terrainHeight = GetTerrainHeight(p);   
  return  p.y - max((terrainHeight+GetTreeHeight(p, terrainHeight)), 0.);
}

// Function 515
float map_flame_s(vec3 pos)
{
   
    vec3 q = pos*0.6;
    q*= vec3(1., 1.5, 1.);
    q+= vec3(ft, 0., 0.);
    float dn = 0.5*(dnf - dnx*pos.x);
    pos.x+= dn*noise(q + vec3(12., 3. + ft, 16.)) - dn/2.;
    pos.y+= dn*noise(q + vec3(14., 7., 20.)) - dn/2.;
    pos.z+= dn*noise(q + vec3(8., 22., 9.)) - dn/2.;

    float df = length(pos.yz) + 0.8*pos.x + 2.;
    
    return df;
}

// Function 516
vec2 map(in vec3 pos) {
    vec2 res = vec2(sdPlane(pos), 1.0);
    
    // bledng cubes by groups of 9
    float gap = 1.2;
	
    vec3 cubeSize = vec3(0.5);
    for(float i=-1.; i<=1.; i+=1.) {
        for(float j=-1.; j<=1.; j+=1.) {
            vec3 p = opRepXZ(pos, vec3(gap, 0., gap))+vec3(i*gap, 0., j*gap);
//            vec3 p = pos+vec3(i*gap, 0., j*gap);
            float rotX = iTime+pos.x*0.3;
	        float rotY = iTime+pos.z*0.3;
            vec3 p2 = rY(rX(p, rotX), rotY);
            
            float dst = sdBox(p2, cubeSize);
            
            // smooth min
            float a = res.x;
            float b = dst;
            // blending power : 0. to 1.
            float k = .5;
            float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
            
		    res = vec2(mix( b, a, h ) - k*h*(1.0-h), 3.0);
        }
    }

    return res;
}

// Function 517
float mapTerrain(vec3 p, float s) {
    float hm = heightMap(p, s);
    return sdPlane(p-vec3(0.0, -0.25-p.z*0.05, 0.0))+hm;
}

// Function 518
vec3 doBumpMap( sampler2D tex, in vec3 p, in vec3 nor, float bumpfactor){
    const float eps = 0.001;
    float ref = (tex3D(tex,  p , nor)).x;                 
    vec3 grad = vec3( (tex3D(tex, vec3(p.x-eps, p.y, p.z), nor).x)-ref,
                      (tex3D(tex, vec3(p.x, p.y-eps, p.z), nor).x)-ref,
                      (tex3D(tex, vec3(p.x, p.y, p.z-eps), nor).x)-ref )/eps;
             
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
}

// Function 519
float map(in vec3 p) {
    float box = sdBox(p, vec3(0.8, 0.8, 0.8));
	float rsphere = sdSphere(p, 1.0);
    float rbox1 = sdBox(p, vec3(2.0, 0.1, 2.0));
    float rbox2 = sdBox(p.yzx, vec3(2.0, 0.1, 2.0));
    float rbox3 = sdBox(p.zxy, vec3(2.0, 0.1, 2.0));
    return sdDifference(box, sdUnion(rsphere, rbox1, rbox2, rbox3));
}

// Function 520
vec2 GetUVRotate(const vec2 vInputUV, float t)
{
	vec2 vFontUV = vInputUV - 0.5;
	
	float s = sin(t);
	float c = cos(t);
	
	vFontUV = vec2(  vFontUV.x * c + vFontUV.y * s,
			        -vFontUV.x * s + vFontUV.y * c );
	
	vFontUV += 0.5;
	
	return vFontUV;
}

// Function 521
vec3 cmap(float W) {
    vec3 C;
    C.x = .5+.5*cos(pi*W);
    C.y = .5+.5*cos(pi*4.*W);
    C.z = .5+.5*cos(pi*2.*W);
    return C;
  }

// Function 522
float MapFrontWing(vec3 p, float mirrored)
{
  missileDist=10000.0;

  checkPos = p;
  pR(checkPos.xy, -0.02);
  float wing =sdBox( checkPos- vec3(4.50, 0.25, -4.6), vec3(3.75, 0.04, 2.6)); 

  if (wing<5.) //Bounding Box test
  {
    // cutouts
    checkPos = p-vec3(3.0, 0.3, -.30);
    pR(checkPos.xz, -0.5);
    wing=fOpIntersectionRound(wing, -sdBox( checkPos, vec3(6.75, 1.4, 2.0)), 0.1);

    checkPos = p - vec3(8.0, 0.3, -8.80);
    pR(checkPos.xz, -0.05);
    wing=fOpIntersectionRound(wing, -sdBox( checkPos, vec3(10.75, 1.4, 2.0)), 0.1);

    checkPos = p- vec3(9.5, 0.3, -8.50);
    wing=fOpIntersectionRound(wing, -sdBox( checkPos, vec3(2.0, 1.4, 6.75)), 0.6);

    // join wing and engine
    wing=min(wing, sdCapsule(p- vec3(2.20, 0.3, -4.2), vec3(0, 0, -1.20), vec3(0, 0, 0.8), 0.04));
    wing=min(wing, sdCapsule(p- vec3(3., 0.23, -4.2), vec3(0, 0, -1.20), vec3(0, 0, 0.5), 0.04));    

    checkPos = p;
    pR(checkPos.xz, -0.03);
    wing=min(wing, sdConeSection(checkPos- vec3(0.70, -0.1, -4.52), 5.0, 0.25, 0.9));   

    checkPos = p;
    pR(checkPos.yz, 0.75);
    wing=fOpIntersectionRound(wing, -sdBox( checkPos- vec3(3.0, -.5, 1.50), vec3(3.75, 3.4, 2.0)), 0.12); 
    pR(checkPos.yz, -1.95);
    wing=fOpIntersectionRound(wing, -sdBox( checkPos- vec3(2.0, .70, 2.20), vec3(3.75, 3.4, 2.0)), 0.12); 

    checkPos = p- vec3(0.47, 0.0, -4.3);
    pR(checkPos.yz, 1.57);
    wing=min(wing, sdTorus(checkPos-vec3(0.0, -3., .0), vec2(.3, 0.05)));   

    // flaps
    wing =max(wing, -sdBox( p- vec3(3.565, 0.1, -6.4), vec3(1.50, 1.4, .5)));
    wing =max(wing, -max(sdBox( p- vec3(5.065, 0.1, -8.4), vec3(0.90, 1.4, 2.5)), -sdBox( p- vec3(5.065, 0., -8.4), vec3(0.89, 1.4, 2.49))));

    checkPos = p- vec3(3.565, 0.18, -6.20+0.30);
    pR(checkPos.yz, -0.15+(0.8*pitch));
    wing =min(wing, sdBox( checkPos+vec3(0.0, 0.0, 0.30), vec3(1.46, 0.007, 0.3)));

    // missile holder
    float holder = sdBox( p- vec3(3.8, -0.26, -4.70), vec3(0.04, 0.4, 0.8));

    checkPos = p;
    pR(checkPos.yz, 0.85);
    holder=max(holder, -sdBox( checkPos- vec3(2.8, -1.8, -3.0), vec3(1.75, 1.4, 1.0))); 
    holder=max(holder, -sdBox( checkPos- vec3(2.8, -5.8, -3.0), vec3(1.75, 1.4, 1.0))); 
    holder =fOpUnionRound(holder, sdBox( p- vec3(3.8, -0.23, -4.70), vec3(1.0, 0.03, 0.5)), 0.1); 

    // bomb
    bombDist = fCylinder( p- vec3(3.8, -0.8, -4.50), 0.35, 1.);   
    bombDist =min(bombDist, sdEllipsoid( p- vec3(3.8, -0.8, -3.50), vec3(0.35, 0.35, 1.0)));   
    bombDist =min(bombDist, sdEllipsoid( p- vec3(3.8, -0.8, -5.50), vec3(0.35, 0.35, 1.0)));   

    // missiles
    checkPos = p-vec3(2.9, -0.45, -4.50);

    // check if any missile has been fired. If so, do NOT mod missile position  
    float maxMissiles =0.; 
    if (mirrored>0.) maxMissiles =  mix(1.0, 0., step(1., missilesLaunched.x));
    else maxMissiles =  mix(1.0, 0., step(1., missilesLaunched.y)); 

    pModInterval1(checkPos.x, 1.8, .0, maxMissiles);
    holder = min(holder, MapMissile(checkPos));

    // ESM Pod
    holder = min(holder, MapEsmPod(p-vec3(7.2, 0.06, -5.68)));

    // wheelholder
    wing=min(wing, sdBox( p- vec3(0.6, -0.25, -3.8), vec3(0.8, 0.4, .50)));

    wing=min(bombDist, min(wing, holder));
  }

  return wing;
}

// Function 523
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

// Function 524
float map_s(vec3 pos)
{
    vec3 posr = rotateVec2(pos);
    posr.y-= 0.45;
    return sdCylinder(posr, lampsize*vec2(1.2, 1.37));
}

// Function 525
vec3 trunkBumpMap(vec3 pos, vec3 nor, float amount) {
	float e = 0.001;

	float ref = luma(trunkColor(pos));

	vec3 gra = -vec3(luma(trunkColor(vec3(pos.x + e, pos.y, pos.z))) - ref,
					 luma(trunkColor(vec3(pos.x, pos.y + e, pos.z))) - ref,
					 luma(trunkColor(vec3(pos.x, pos.y, pos.z + e))) - ref) / e;

	vec3 tgrad = gra - nor * dot(nor, gra);
	return normalize(nor - amount * tgrad);
}

// Function 526
float map2(vec3 pos)
{
    pos.xy = rotateVec(pos.xy, 0.15*smoothstep(40., 200., iTime)*cos(rot*2.));
    pos.zy = rotateVec(pos.zy, 0.15*smoothstep(40., 200., iTime)*sin(rot*2.));
    return length(pos.xz)*0.72 - 0.55*smoothstep(0.2, 0.7, abs(pos.y - campos.y/17.))*smoothstep(1.4 - abs(campos.y)/24., 0.7, abs(pos.y - campos.y/17.));
}

// Function 527
vec3 filmicToneMapping(vec3 color)
{
	color = max(vec3(0.), color - vec3(0.004));
	color = (color * (6.2 * color + .5)) / (color * (6.2 * color + 1.7) + 0.06);


    // May be very wrong? :))))
    color = toGamma(color, 1./3.0);
    color = toGamma(color, gamma);
    
	return color;
}

// Function 528
vec3 TriplanarTextureMapping(const vec3 p, const vec3 n, const int texID)
{
    mat3 samples;
    
    switch(texID)
    {
        // iChannel0 is for the SkyBox
        case 1:
        	samples = mat3 (texture(iChannel1, p.yz).rgb,
                         	texture(iChannel1, p.xz).rgb,
                         	texture(iChannel1, p.xy).rgb );
        	break;
        case 2:
        	samples = mat3 (texture(iChannel2, p.yz).rgb,
                         	texture(iChannel2, p.xz).rgb,
                         	texture(iChannel2, p.xy).rgb );
        	break;
        case 3:
        	samples = mat3 (texture(iChannel3, p.yz).rgb,
                         	texture(iChannel3, p.xz).rgb,
                         	texture(iChannel3, p.xy).rgb );
        	break;
        default:
        	samples = mat3(0);
        	break;
    }
    
    // Weight the samples with the normal to get the one more aligned
    return samples * abs(n);
}

// Function 529
float SandHeightMap(vec3 position)
{
    float sandGrainNoise = 0.1 * fbm(position * 10.0, 2);
    float sandDuneDisplacement = 0.7 * sin(10.0 * fbm_4(10.0 + position / 40.0));
	return sandGrainNoise  + sandDuneDisplacement;
}

// Function 530
vec2 getuv(vec2 uv) {
  uv = uv-floor(uv);
  float u = uv[0];
  float v = uv[1];

  // 0 <= u,v < 1
  float u0 = v+u-1.0, v0 = v-u;
  u = u0; v = v0;
  // -1 < u,v < 1
  if (u < 0.0) { u = -u; v = -v; }
  if (v > 0.5) { v = 1.0-v; u = 1.0-u; }
  if (v < -0.5) { v = -1.0-v; u = 1.0-u; }
  v += 0.5;
  return vec2(u,v);
}

// Function 531
vec2 mapuv(vec2 v){
    v/=R;
    v*=2.0;
    v-=1.0;
    v.x*=R.x/R.y;
    return v*1.2;
}

// Function 532
vec3 depth_map(vec2 coord){
    return texture(iChannel0, vec2((coord.x), (coord.y))).xyz;
}

// Function 533
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    // Project ray direction on to the unit cube.
    vec3 absRayDir = abs(rayDir);
    rayDir /= max(absRayDir.x, max(absRayDir.y, absRayDir.z));

    
    // Get the index of the current face being rendered.
    
    int faceIndex = 0;

    if(absRayDir.y > absRayDir.x && absRayDir.y > absRayDir.z)
    {
        faceIndex = 2;
    }
    else if(absRayDir.z > absRayDir.x && absRayDir.z > absRayDir.y)
    {
        faceIndex = 4;
    }

    if(rayDir[faceIndex / 2] > 0.)
        faceIndex |= 1;

    fragColor = vec4(0);

    if(faceIndex != 1)
        return;

    vec2 uv = fragCoord.xy / 1024.;
    
    // Resample
    fragColor.rg = sampleSnowflakes(uv);
}

// Function 534
float map_alpha(vec3 p) {
#ifndef SOLID
    float c000 = 0.0;
    float c010 = 1.0;
    float c100 = 1.0;
    float c110 = 0.0;
    float c001 = 1.0;
    float c011 = 0.0;
    float c101 = 0.0;
    float c111 = 1.0;
#else
    float c000 = 1.0;
    float c010 = 1.0;
    float c100 = 1.0;
    float c110 = 1.0;
    float c001 = 1.0;
    float c011 = 1.0;
    float c101 = 1.0;
    float c111 = 1.0;
#endif
    float c00 = mix(c000, c001, p.x);
    float c01 = mix(c010, c011, p.x);
    float c10 = mix(c100, c101, p.x);
    float c11 = mix(c110, c111, p.x);
    float c0 = mix(c00, c01, p.y);
    float c1 = mix(c10, c11, p.y);
    float c = mix(c0, c1, p.z);    
    return c*ascale+abias;
}

// Function 535
float colormap_red(float x) {
    if (x < 0.0) {
        return 54.0 / 255.0;
    } else if (x < 20049.0 / 82979.0) {
        return (829.79 * x + 54.51) / 255.0;
    } else {
        return 1.0;
    }
}

// Function 536
float map(vec3 p) {
  float k;
  vec4 p4 = map4(p,k);
  float R = sqrt(TRATIO.x*TRATIO.x/dot(TRATIO,TRATIO));
  float d = fTorus(p4,R);
  return d/k;
}

// Function 537
float map_wall_2(vec3 pos)
{
    return pos.z;
}

// Function 538
float map(in vec3 p){
   
   
    vec4 d;
    
    
    // Back wall.
    
    // Perturbing things a bit.
    vec3 q = p + sin(p*2. - cos(p.zxy*3.5))*.1;
    
    // Grabbing the 2D surface value from the second face of the cubemap.
    float sf2D = surfFunc2D(q);
    
    // Combining the 2D Voronoi value above with an extrusion process to creat some netting.
    d.z = smax(abs(-q.z + 6. - .5) - .05, (sf2D/2. - .025), .02);
    //d.z = -(length(q - vec3(0, 0, -(12. - 6.))) - 12.) + (.5 - sf2D)*.5;
    
    // The back plane itself -- created with a bit of extrusion and addition. 
    d.w = -q.z + 6.;
    float top = (.5 - smoothstep(0., .35, sf2D - .025));
    d.w = smin(d.w, smax(abs(d.w) - .75, -(sf2D/2. - .025 - .04), .02) + top*.1, .02);
    
    
    // The celluar geometric ball object.
    
    // Rotate the object.
    q = rotObj(p);
    // Perturb it a bit.
    q += sin(q*3. - cos(q.yzx*5.))*.05;

    // Retrieve the 3D surface value. Note (in the function) that the 3D value has been 
    // normalized. That way, everything points toward the center.
    float sf3D = surfFunc3D(q);
    
    
    // Adding a small top portion.
    top = (.5 - smoothstep(0., .35, sf3D - .025));
    
    d.x = length(q) - 1.; // The warped spherical base.
    
    // The gold, metallic spikey ball surface -- created via an extrusion process
    d.y = smin(d.x + .1, smax(d.x - .2, -(sf3D/2.-.025 - .06), .02) + top*.05, .1);
    
    // The spherical netting with holes -- created via an extrusion process.
    d.x = smax(abs(d.x) - .025, sf3D/2.-.025, .01);
    
    
    
    // Store the individual object values for sorting later. Sorting multiple objects
    // inside a raymarching loop probably isn't the best idea. :)
    objID = d;
    
    // Return the minimum object in the scene.
    return min(min(d.x, d.y), min(d.z, d.w));
}

// Function 539
vec3 map( in vec3 p )
{
    vec3 p00 = p - vec3( 0.00, 3.0, 0.0);
	vec3 p01 = p - vec3(-1.25, 2.0, 0.0);
	vec3 p02 = p - vec3( 0.00, 2.0, 0.0);
	vec3 p03 = p - vec3( 1.25, 2.0, 0.0);
	vec3 p04 = p - vec3(-2.50, 0.5, 0.0);
	vec3 p05 = p - vec3(-1.25, 0.5, 0.0);
	vec3 p06 = p - vec3( 0.00, 0.5, 0.0);
	vec3 p07 = p - vec3( 1.25, 0.5, 0.0);
	vec3 p08 = p - vec3( 2.50, 0.5, 0.0);
	vec3 p09 = p - vec3(-3.75,-1.0, 0.0);
	vec3 p10 = p - vec3(-2.50,-1.0, 0.0);
	vec3 p11 = p - vec3(-1.25,-1.0, 0.0);
	vec3 p12 = p - vec3( 0.00,-1.0, 0.0);
	vec3 p13 = p - vec3( 1.25,-1.0, 0.0);
	vec3 p14 = p - vec3( 2.50,-1.0, 0.0);
	vec3 p15 = p - vec3( 3.75,-1.0, 0.0);
	
    vec3 p16 = p - vec3(-5.00,-2.7, 0.0);
    vec3 p17 = p - vec3(-3.75,-2.7, 0.0);
    vec3 p18 = p - vec3(-2.50,-2.7, 0.0);
    vec3 p19 = p - vec3(-1.25,-2.7, 0.0);
    vec3 p20 = p - vec3( 0.00,-2.7, 0.0);
    vec3 p21 = p - vec3( 1.25,-2.7, 0.0);
    vec3 p22 = p - vec3( 2.50,-2.7, 0.0);
    vec3 p23 = p - vec3( 3.75,-2.7, 0.0);
    vec3 p24 = p - vec3( 5.00,-2.7, 0.0);
	
	float r, d; vec3 n, s, res;
	
    #ifdef SHOW_SPHERES
	#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))
	#else
	#define SHAPE (vec3(d-abs(r), sign(r),d))
	#endif
	d=length(p00); n=p00/d; r = SH_0_0( n ); s = SHAPE; res = s;
	d=length(p01); n=p01/d; r = SH_1_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p02); n=p02/d; r = SH_1_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p03); n=p03/d; r = SH_1_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p04); n=p04/d; r = SH_2_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p05); n=p05/d; r = SH_2_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p06); n=p06/d; r = SH_2_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p07); n=p07/d; r = SH_2_3( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p08); n=p08/d; r = SH_2_4( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p09); n=p09/d; r = SH_3_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p10); n=p10/d; r = SH_3_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p11); n=p11/d; r = SH_3_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p12); n=p12/d; r = SH_3_3( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p13); n=p13/d; r = SH_3_4( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p14); n=p14/d; r = SH_3_5( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p15); n=p15/d; r = SH_3_6( n ); s = SHAPE; if( s.x<res.x ) res=s;
    
	d=length(p16); n=p16/d; r = SH_4_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p17); n=p17/d; r = SH_4_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p18); n=p18/d; r = SH_4_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p19); n=p19/d; r = SH_4_3( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p20); n=p20/d; r = SH_4_4( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p21); n=p21/d; r = SH_4_5( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p22); n=p22/d; r = SH_4_6( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p23); n=p23/d; r = SH_4_7( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p24); n=p24/d; r = SH_4_8( n ); s = SHAPE; if( s.x<res.x ) res=s;
	
	return vec3( res.x, 0.5+0.5*res.y, res.z );
}

// Function 540
vec2 map(in vec3 l, in sph s) {
    vec3 n = nrm(l, s);
    return vec2(atan(n.z, n.x) + pi, acos(-n.y)) / vec2(pi2, pi);
}

// Function 541
float map(vec3 p, mat3 rotMat)
{
	//float r = sdPlane(p - vec3(0,-1.5,0));	// Return distance
    float r = 1000.0;
	
    #define GS 0.1
    //mat3 rotMat = rotationMatrix(vec3(1,1,0), iTime);
    vec3 boxmod = mix(mod(p+vec3(GS),vec3(GS*2.0))-vec3(GS), p, step(0.55, sdTorus(rotMat*stepround(p, vec3(GS)), vec2(2,.5))) );
    //vec3 boxmod = mod(p+vec3(GS),vec3(GS*2.0))-vec3(GS);
    vec3 sizemod = mix(vec3(clamp(0.3-sdTorus(rotMat*stepround(p, vec3(GS)), vec2(2,0.5))*0.6, 0.0, GS) ), vec3(0), step(0.6, sdTorus(rotMat*stepround(p, vec3(GS)), vec2(2,.5))) );
    
    r = min(r, sdBox(boxmod, sizemod));
    return r;
}

// Function 542
float mapEnvironment(in vec2 p, in int ballID) {
    float distToEnv = mapEnvironmentNoBalls(p);
    for (int id=0; id < NUMBER_OF_BALLS; id++) {
        if (id != ballID) {
            vec2 ballPos = getBallPos(id);
            distToEnv = min(distToEnv, length(p - ballPos) - BALL_RADIUS);
        }
    }

    return distToEnv;
}

// Function 543
vec3 tonemap(vec3 col) {
    return col / (vec3(1.0) + col);
}

// Function 544
vec2 map(in vec3 l, in box b) {
	mat3 o = mat3(v30, nrm(l, b) + eps, v30);
    basis(o[1], o[0], o[2]);
    vec3 r = l * o;
    return r.xz;
}

// Function 545
vec2 cameraRayToUv(ray cameraRay, float projectionDist)
{
    vec2 uv = vec2(normalize(cameraRay.direction).x, normalize(cameraRay.direction).y);
    uv *= projectionDist/dot(normalize(cameraRay.direction), vec3(0, 0, projectionDist));
    return uv;
}

// Function 546
vec2 map(vec2 p){
  return cdiv(vec2(1,0)+p,vec2(1,0)-p);
}

// Function 547
float remap( float t, float a, float b )
{
	return clamp( (t - a) / (b - a), 0.0, 1.0 );
}

// Function 548
vec3 nmap(vec2 t, sampler2D tx, float str)
{
	float d=1.0/1024.0;

	float xy=texture(tx,t).x;
	float x2=texture(tx,t+vec2(d,0)).x;
	float y2=texture(tx,t+vec2(0,d)).x;
	
	float s=(1.0-str)*1.2;
	s*=s;
	s*=s;
	
	return normalize(vec3(x2-xy,y2-xy,s/8.0));
}

// Function 549
float map( vec3 p )
{
    vec3 w = p;
    vec3 q = p;

    q.xz = mod( q.xz+1.0, 2.0 ) -1.0;
    
    float d = (p.y-terrainH(p.xz));
    d=d*min(max(abs(d),0.001)*1000.0,0.125);
    //d=d*d*10.0;

    
   return d;
}

// Function 550
float YUVtoG(float Y, float U, float V){
  float G=Y/0.587-0.299/0.587*YUVtoR(Y,U,V)-0.114/0.587*YUVtoB(Y,U,V);
  return G;
}

// Function 551
vec3 map(in vec3 p){for(int i=0;i<100;i++)p.xzy =abs(vec3(1.3,.99,.75)*(p/dot(p,p)-vec3(1,1,.05)));return p/50.;}

// Function 552
vec3 yuv_rgb (vec3 yuv, vec2 wbwr, vec2 uvmax) {
    vec2 br = yuv.x + yuv.yz * (1.0 - wbwr) / uvmax;
	float g = (yuv.x - dot(wbwr, br)) / (1.0 - wbwr.x - wbwr.y);
	return vec3(br.y, g, br.x);
}

// Function 553
float heightMapWater(vec3 p, float t)
{
    float h = 0.0;
    vec3 op = p;
    #ifdef FLOOD
    float w = (-p.z+sin(TIME*WAVES_SPEED))*FLOOD_AMP;
    #endif
    float a = WATER_AMP;
    float f = WATER_FREQ;
    float T = TIME(t)*WATER_SPEED;
    //h = 0.2*(-1.0+fbm_hash(p.xz+TIME)+fbm_hash(p.xz-TIME)); 
    for(int i = 0;i < 3; ++i)
    {
//     e((−2πikn)/N   )
        float ffta = 1.0;//exp((-2.0*3.14*float(i)*(f*length(p)+T))/6.0);
    	h = a*(-1.0+fbm2Dsimple(f*p.xz+T)+fbm2Dsimple(f*p.xz-T))*ffta; 
        a*= 0.8;
        f *= 1.2;
    }
    //for(int i=0;i<5;++i) {
    //}
    #ifdef WAVES
    h+= wave(op,
             mix(0.05, 0.9, min(1.0, max(0.0,p.z)/3.2)),
             T*5.0)*clamp(h, 0.2, 0.6);
        //*gaussianNoise(op.xz*0.1+T);
    #endif
    #ifdef FLOOD
    return h+w;
    #else
    return h;
    #endif 
}

// Function 554
vec3 depth_map(vec2 coord){
    return texture(iChannel0, vec2((coord.x-image_scale/2.0)/iResolution.x/image_scale, (coord.y-image_scale/2.0)/iResolution.y/image_scale)).xyz;
}

// Function 555
float mapDamage( vec3 p ) {
    float d = map( p );

    float n = max( max( 1.-abs(p.z*.01), 0. )*
                   max( 1.-abs(p.y*.2-1.2), 0. ) *
                   noise( p*.3 )* (noise( p*2.3 ) +.2 )-.2 - damageMod, 0.);
   
	return d + n;
}

// Function 556
vec2 uv_polar(vec2 domain, vec2 center){
   vec2 c = domain - center;
   float rad = length(c);
   float ang = atan2(c.y, c.x);
   return vec2(ang, rad);
}

// Function 557
float map(in vec3 r)
{
    float rho = length(r);
    vec3 u = r / rho;
    
    float a = Ylm(_l,_m,u);
    

    _sgn = sign(a);
    _dist = rho;
    
    //return rho - abs(a);
	return rho - mix(abs(a), c00, clamp(2.0*cos(iTime), 0.0, 1.0));
}

// Function 558
float remapIntensity(float f, float i){
    //return i; // nothing
    float k = to01( trebles(f,bass(f,i))*fftPreamp);
    //return k; // no dynamic
    return k*(k+fftBoost); // more dynamic
}

// Function 559
vec2 GetUVRepeat(const vec2 vInputUV, float t2)
{
	vec2 vFontUV = vInputUV;
	
	vFontUV *= vec2(1.0, 4.0);
	
	vFontUV.x += floor(vFontUV.y) * t2;
	
	vFontUV = fract(vFontUV);
	
	vFontUV /= vec2(1.0, 4.0);
		
	return vFontUV;
}

// Function 560
vec4 map (vec3 p)
{
	return sd2d(p.xz,p.y);
}

// Function 561
float map_rled(vec3 pos)
{
    return sdRoundBox(pos - vec3(-supportSize.x + 0.25, 2.*supportSize.y - 0.1, -supportSize.z - 0.01), vec3(0.038, 0.009, 0.009), 0.004);
}

// Function 562
vec3 normalMap( in vec2 pos )
{
	pos *= 2.0;
	
	float v = texture( iChannel3, 0.015*pos ).x;
	vec3 nor = vec3( texture( iChannel3, 0.015*pos+vec2(1.0/1024.0,0.0)).x - v,
	                 1.0/16.0,
	                 texture( iChannel3, 0.015*pos+vec2(0.0,1.0/1024.0)).x - v );
	nor.xz *= -1.0;
	return normalize( nor );
}

// Function 563
float map(vec2 p) {
    float d0 = length(p) - 0.5;
    float d1 = length(p + vec2(-0.25,0.0)) - 0.3;
    float d2 = length(p + vec2(0.0, 0.5)) - 0.2;
    return max(min(d0,d2), -d1);
}

// Function 564
vec2 map(vec3 p) {
    
    vec2 sphereObj =  vec2(sdSphere(p - lights[0].position, lights[0].radius), LIGHT_ID);    
    vec2 oldObj = sphereObj;
    
   	vec2 resultObj = sphereObj;

    vec2 newObj =  vec2(sdSphere(p - spheres[1].p, spheres[1].r), SPHERE_ID1);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdSphere(p - spheres[2].p, spheres[2].r), SPHERE_ID2);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdSphere(p - spheres[3].p, spheres[3].r), SPHERE_ID3);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdSphere(p - spheres[4].p, spheres[4].r), SPHERE_ID4);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdSphere(p - spheres[5].p, spheres[5].r), SPHERE_ID5);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdSphere(p - spheres[6].p, spheres[6].r), SPHERE_ID6);
    resultObj = opU(resultObj, newObj);
	newObj =  vec2(sdPlane(p - planes[0].p, planes[0].n), FLOOR_ID);
   	resultObj = opU(resultObj, newObj);
    
    return resultObj;
}

// Function 565
vec3 eMap(vec3 rd, vec3 sn){
    
    vec3 sRd = rd; // Save rd, just for some mixing at the end.
    
    // Add a time component, scale, then pass into the noise function.
    rd.xy -= iTime*.25;
    rd *= 3.;
    
    //vec3 tx = tex3D(iChannel0, rd/3., sn);
    //float c = dot(tx*tx, vec3(.299, .587, .114));
    
    float c = n3D(rd)*.57 + n3D(rd*2.)*.28 + n3D(rd*4.)*.15; // Noise value.
    c = smoothstep(0.5, 1., c); // Darken and add contast for more of a spotlight look.
    
    //vec3 col = vec3(c, c*c, c*c*c*c).zyx; // Simple, warm coloring.
    //vec3 col = vec3(min(c*1.5, 1.), pow(c, 2.5), pow(c, 12.)).zyx; // More color.
    vec3 col = pow(vec3(1.5, 1, 1)*c, vec3(1, 2.5, 12)).zyx; // More color.
    
    // Mix in some more red to tone it down and return.
    return mix(col, col.yzx, sRd*.25 + .25); 
    
}

// Function 566
float map(in vec3 p)
{
    return scene(p);
}

// Function 567
vec2 uv_polar_logarithmic(vec2 domain, vec2 center, float fins, float log_factor, vec2 coord){
   vec2 polar = uv_polar(domain, center);
   return vec2(polar.x * fins + coord.x, log_factor*log(polar.y) + coord.y);
}

// Function 568
float rmap(vec3 uv, RSet3 rs) {
    return RAND(map(uv, rs.q, rs.l), rs.r);
}

// Function 569
float calCapsuleUV(vec3 pos)
{
    pos = rot(iTime) * pos;
    
    vec4 sph = sphere_info;
    
    vec3 dir = normalize(pos - sph.xyz);
    vec2 dir2 = normalize(dir.xz); 
    
    float d = acos(dir2.y) / 3.1415926;
    
    if(dir2.x < 0.0)
    {
        d = 2.0 - d;
    }
    
    return d * 0.5;
}

// Function 570
vec3 doBumpMap(in vec3 p, in vec3 n, float bumpfactor, inout float edge, inout float crv){
    
    // Resolution independent sample distance... Basically, I want the lines to be about
    // the same pixel with, regardless of resolution... Coding is annoying sometimes. :)
    vec2 e = vec2(8./iResolution.y, 0); 
    
    float f = bumpFunc(p, n); // Hit point function sample.
    
    float fx = bumpFunc(p - e.xyy, n); // Nearby sample in the X-direction.
    float fy = bumpFunc(p - e.yxy, n); // Nearby sample in the Y-direction.
    float fz = bumpFunc(p - e.yyx, n); // Nearby sample in the Y-direction.
    
    float fx2 = bumpFunc(p + e.xyy, n); // Sample in the opposite X-direction.
    float fy2 = bumpFunc(p + e.yxy, n); // Sample in the opposite Y-direction.
    float fz2 = bumpFunc(p + e.yyx, n);  // Sample in the opposite Z-direction.
    
     
    // The gradient vector. Making use of the extra samples to obtain a more locally
    // accurate value. It has a bit of a smoothing effect, which is a bonus.
    vec3 grad = vec3(fx - fx2, fy - fy2, fz - fz2)/(e.x*2.);  
    //vec3 grad = (vec3(fx, fy, fz ) - f)/e.x;  // Without the extra samples.


    // Using the above samples to obtain an edge value. In essence, you're taking some
    // surrounding samples and determining how much they differ from the hit point
    // sample. It's really no different in concept to 2D edging.
    edge = abs(fx + fy + fz + fx2 + fy2 + fz2 - 6.*f);
    edge = smoothstep(0., 1., edge/e.x*2.);
    
    
    // We may as well use the six measurements to obtain a rough curvature value while we're at it.
    //crv = clamp((fx + fy + fz + fx2 + fy2 + fz2 - 6.*f)*32. + .6, 0., 1.);
    
    // Some kind of gradient correction. I'm getting so old that I've forgotten why you
    // do this. It's a simple reason, and a necessary one. I remember that much. :D
    grad -= n*dot(n, grad);          
                      
    return normalize(n + grad*bumpfactor); // Bump the normal with the gradient vector.
	
}

// Function 571
float mapLeafWaterDrops( in vec3 p )
{
    p -= vec3(-1.8,0.6,-0.75);
    vec3 s = p;
    p = mat3(0.671212, 0.366685, -0.644218,
            -0.479426, 0.877583,  0.000000,
             0.565354, 0.308854,  0.764842)*p;
  
    vec3 q = p;
    p.y += 0.2*exp(-abs(2.0*p.z) );
    
    //---------------
    
    float r = clamp((p.x+2.0)/4.0,0.0,1.0);
    r = r*(1.0-r)*(1.0-r)*6.0;
    float d0 = sdEllipsoid( p, vec3(0.0), vec3(2.0,0.25*r,r) );
    float d1 = sdEllipsoid( q, vec3(0.5,0.0,0.2), 1.0*vec3(0.15,0.13,0.15) );
    float d2 = sdEllipsoid( q, vec3(0.8,-0.07,-0.15), 0.5*vec3(0.15,0.13,0.15) );
    float d3 = sdEllipsoid( s, vec3(0.76,-0.8,0.6), 0.5*vec3(0.15,0.2,0.15) );
    float d4 = sdEllipsoid( q, vec3(-0.5,0.09,-0.2), vec3(0.04,0.03,0.04) );

    d3 = max( d3, p.y-0.01);
    
    return min( min(d1,d4), min(d2,d3) );
}

// Function 572
vec3 map( in vec3 pos )
{
	vec2  p = fract( pos.xz ); 
    vec3  m = mapH( pos.xz );
	float d = dbox( vec3(p.x-0.5,pos.y-0.5*m.x,p.y-0.5), vec3(0.3,m.x*0.5,0.3), 0.1 );
    return vec3( d, m.yz );
}

// Function 573
float hsluv_toLinear(float c) {  return c > 0.04045 ? pow((c + 0.055) / (1.0 + 0.055), 2.4) : c / 12.92; }

// Function 574
float map_container(vec3 pos)
{
    pos.y+= 2.8;
    float outside = udRoundBox(pos, vec3(5.5, 1.5, 2.), 0.6);
    float inside = udRoundBox(pos, vec3(5.35, 1.35, 1.85), 0.6);
    float bottom = max(max(outside, -inside), pos.y - 1.);
    
    pos.y-= 4.3;
    outside = udRoundBox(pos, vec3(5., 0.8, 1.), 0.45);
    inside = udRoundBox(pos, vec3(4.9, 0.35, 0.9), 0.45);
    float top = max(max(outside, -inside), - pos.y + 0.25);
    
    pos.y-= 5.4;
    float top2 = udRoundBox(pos, vec3(2.8, 4., 0.7), 0.3);
    return min(bottom, smin(top, top2, 0.45));
}

// Function 575
vec2 map(vec3 pos)
{
    vec2 fpos = fract(pos.xz); 
	vec2 ipos = floor(pos.xz);
    
    float rid = hash21(ipos) + 0.5;

    return vec2(fCapsule(vec3(fpos.x - 0.5, pos.y, fpos.y - 0.5), vec3(0.0, 0.0, 0.0), vec3(0.0, HEIGHT, 0.0), 0.1), rid);
}

// Function 576
vec3 lchToHsluv(vec3 tuple) {
    tuple.g /= hsluv_maxChromaForLH(tuple.r, tuple.b) * .01;
    return tuple.bgr;
}

// Function 577
float map(vec3 p)
{
    
 float plane = plane(p, vec4(0.0, 1., 0.0, .9  ));//+stripes(p)/20.));
    
    
    //these are options to deform the plane.
   // p.y+=sin(p.x+sin(p.z))/5.+sin(p.z/2.+sin(p.z/9.)*10.)/20.;//+sin(p.z/3.)/4.;
   // p.y+=(floor(abs(p.x))/1.);

    
    //I use a scale factor so I can change the size of the tiling with one variable defined at top.
    float sca = SCALE;
    
    //this line does something interesting
   //p.x += +sin(iTime*floor(p.y)/10.)*8.;
    
    //this line not so much
   //p.y += +sin(iTime*floor(p.x)/100.)*8.;
    
    //height variable not used because changing the heights of blocks based on floor doesn't work out.
    //it creates really bad aliasing and I'm not sure why just yet.
    float height = sin(iTime*floor(p.x))*1.;
    
    
	 vec3 fp;

    fp.xyz = mod(p.xyz, 1./sca)-0.5/sca;
  
    
   /*float circleIn = smoothstep(-0.3, -.2,length(fract(p.xz)*2.0-1.0)-0.9*rnd(floor(p.xz)));
    circleIn -= smoothstep(-0.3, -.1,length(fract(p.xz)*2.0-1.0)-0.8*rnd(floor(p.xz)));*/
    
   /*float circleIn = smoothstep(-0.3, -.23,length(fract(p.xz/20.)*2.0-1.0)-0.5);//*rnd(floor(p.xz)));
    circleIn -= smoothstep(-0.3, -.1,length(fract(p.xz/20.)*2.0-1.0)-0.6);//*rnd(floor(p.xz)));*/
    //-circleIn/20.  //put on p.y in tiles = ...
    
  //another option for height variation, also not used.
  height = ((rnd(floor( p.xz/8.)))  )/10.;
    
    
 //creates the boxes
 float tiles = roundBox(vec3(fp.x, p.y, fp.z), 
                       vec3(0.47/sca, 0.47/sca, 0.47/sca), 0.019/sca);
    
 //creates the dna
 float dna = helix(p);
/*tiles = roundBox(vec3(mod(p, 2.)-1.), 
                       vec3(0.45, 0.001, 0.45), 0.047);*/

                        //vec3(0.43, 0.028+sin(p.z*0.3)/40.-cos(p.x*1.7)/60., 0.43), 0.0157);
 
    
//more not used stuff
 /* float idr = fract(sin(dot(floor(p.xz/20.-10.), vec2(12.23432, 73.24234)))*412343.2);
	p+=idr*4.;
    vec3 sm = mod(p, 30.)-15.;
    float s = length(vec3(sm.x, p.y-10., sm.z))-3.5;*/
    
  return min(tiles,dna); 
      
}

// Function 578
vec3 hpluvToRgb(vec3 tuple) {
    return lchToRgb(hpluvToLch(tuple));
}

// Function 579
vec4 mapping(float dist, float min, float max, mat4 LUT, vec4 LUT_DIST){	
	
	float distLut = (dist - min) / max;
	int i1;
	for(int i=0;i <NB_LUT;++i){
		if(distLut < LUT_DIST[i+1]){i1 = i;break;}
	}
	vec4 col1,col2;
	float mixVal;
	if		(i1 == 0){
		col1 = LUT[0];col2 = LUT[1];
		mixVal = (distLut - LUT_DIST[0]) / (LUT_DIST[1] - LUT_DIST[0]);
	}else if(i1 == 1){
		col1 = LUT[1];col2 = LUT[2];
		mixVal = (distLut - LUT_DIST[1]) / (LUT_DIST[2] - LUT_DIST[1]);
	}else{
		col1 = LUT[2];col2 = LUT[3];
		mixVal = (distLut - LUT_DIST[2]) / (LUT_DIST[3] - LUT_DIST[2]);
	}
	
	
	//return vec4(mixVal);
	return mix(col1,col2,mixVal);
	
}

// Function 580
vec3 iqCubemap(in vec2 q, in vec2 mo) {
    vec2 p = -1.0 + 2.0 * q;
    p.x *= iResolution.x / iResolution.y;
	
    // camera
	float an1 = -6.2831 * (mo.x + 0.25);
	float an2 = clamp( (1.0-mo.y) * 2.0, 0.0, 2.0 );
    vec3 ro = 2.5 * normalize(vec3(sin(an2)*cos(an1), cos(an2)-0.5, sin(an2)*sin(an1)));
    vec3 ww = normalize(vec3(0.0, 0.0, 0.0) - ro);
    vec3 uu = normalize(cross( vec3(0.0, -1.0, 0.0), ww ));
    vec3 vv = normalize(cross(ww, uu));
    return normalize( p.x * uu + p.y * vv + 1.4 * ww );
}

// Function 581
vec3 tonemap(vec3 x){return mapu(x,2.51*x+.06,.14,x,.59+2.43*x);}

// Function 582
vec3 lchToLuv(vec3 tuple) {
    float hrad = radians(tuple.b);
    return vec3(
        tuple.r,
        cos(hrad) * tuple.g,
        sin(hrad) * tuple.g
    );
}

// Function 583
float map (vec3 p) {
  vec3 p1 = p;
  p1.xz *= rotate(iTime * .3);
  p1.yz *= rotate(iTime * .2);
  float s1 = sphereSDF(p1, .6);
  s1 += sin((p1.x + p1.y * p1.z) * (3.14 * 10.) - iTime * 2.) * 
      .015 - sin((p1.x - p1.y * p1.z) * (3.14 * 12.) - iTime) * .01;

  p.z -= iTime * .2;
  vec3 c = vec3(2., 1.3, 2.);
  p = mod(p, c) - .5 * c;

  float s2 = sphereSDF(p, .3);
  s2 += sin((p.x + p.y * p.z) * 20. + iTime * 2.) * 
      .03 + cos(length(p.x - p.y * p.z) * 65. - iTime) * .01;

  return min(s1 * .3, s2 * .28);
}

// Function 584
void mainCubemap( out vec4 O, vec2 U, vec3 o, vec3 d )
{
    if ( max(d.x, max(d.y,d.z)) != d.z ) return; // we want only face 1 
    
    //U -= .5;
    if (iFrame==0) { O = vec4(0); return; }
    vec2 R = iResolution.xy, I;
                                                // --- set multigrid LOD
  //int n = max(0, int( log2(R.y) -4. - log2(float(1+iFrame/60/2)) ));
    int n = max(0, int( log2(R.y) -2. - float(iFrame/60) )); // what is optimum duration per level ?
  //n = 3;
    U =       U / float(1<<n); I = floor(U);
    R = floor(R / float(1<<n));
    
    O = T(U,0,0, n);                            
                                                // --- Laplacian solver. 
 // vec4 D = (   T(U,1,0,n) + T(U,-1,0,n) + T(U,0, 1,n) + T(U, 0,-1,n) - 4.*O ) / 4.;
 // vec4 D = (   T(U,1,0,n) + T(U,-1,0,n) + T(U,0, 1,n) + T(U, 0,-1,n) // higher orders: https://en.wikipedia.org/wiki/Discrete_Laplace_operator
 //            + T(U,1,1,n) + T(U,-1,1,n) + T(U,1,-1,n) + T(U,-1,-1,n)
 //             - 8.*O ) / 8.;
    vec4 D = ( 2.*( T(U,1,0,n) + T(U,-1,0,n) + T(U,0, 1,n) + T(U, 0,-1,n) )
               +    T(U,1,1,n) + T(U,-1,1,n) + T(U,1,-1,n) + T(U,-1,-1,n)
               - 12.*O ) / 12.;
    O += D;  // apparently stable even with coef 1
    
                                                // --- set border constraints
#if 0
    O =  I.y==0. || I.y==R.y-1. ? vec4(I.x/(R.x-1.)) 
       : I.x==0. ? vec4(0) : I.x==R.x-1. ? vec4(1)
       : O;
#else
    O =  I.x==0. ? vec4(1,0,0,1) : I.x==R.x-1. ? vec4(0,1,0,1)
       : I.y==0. ? vec4(0,0,1,1) : I.y==R.y-1. ? vec4(1,1,1,1)
       : O;
#endif
                                                // --- mouse paint
    vec2 M = texelFetch(iChannel1,ivec2(0),0).xy;   // get normalized mouse position
    if ( length(M)>.1 && length(I/R-M)<.1) O = vec4(I.x/R.x > M.x);
}

// Function 585
vec2 mapSolid(vec3 p) {
  p.xz = rotate2D(p.xz, iTime * 1.25);
  p.yx = rotate2D(p.yx, iTime * 1.85);
  p.y += sin(iTime) * 0.25;
  p.x += cos(iTime) * 0.25;

  float d = length(p) - 0.25;
  float id = 1.0;
  float pulse = pow(sin(iTime * 2.) * 0.5 + 0.5, 9.0) * 2.;

  d = mix(d, sdBox_1117569599(p, vec3(0.175)), pulse);

  return vec2(d, id);
}

// Function 586
vec2 spheremapPack(vec3 n)
{
    float p = sqrt(n.z * 8.0 + 8.0);
    vec2 normal = n.xy / p + 0.5;
    return normal;
}

// Function 587
vec2 forward_mapping(vec2 Z,vec3 R, int seed){
    int p = int(R.x);
    int q = int(R.y);
    
    int x=int(Z.x);
    int y=int(Z.y);
    
    for(int i = 0; i < mapping_iters; i++){
        x = Zmod(x + IHash(y^seed)%p,p);
        y = Zmod(y + IHash(x^seed)%q,q);
    }
        
    return vec2(x,y)+.5;
    
}

// Function 588
float map( in vec3 p )
{
	float d = length(deform(p))-1.5;
	
	return d*.1;
}

// Function 589
vec3 rgb_yuv (vec3 rgb, vec2 wbwr, vec2 uvmax) {
	float y = wbwr.y*rgb.r + (1.0 - wbwr.x - wbwr.y)*rgb.g + wbwr.x*rgb.b;
    return vec3(y, uvmax * (rgb.br - y) / (1.0 - wbwr));
}

// Function 590
float map( in vec3 p )
{
	float h = terrain2(p.xz);
    return p.y - h;
}

// Function 591
float hsluv_maxChromaForLH(float L, float H) {  float hrad = radians(H);  mat3 m2 = mat3(   3.2409699419045214 ,-0.96924363628087983 , 0.055630079696993609,   -1.5373831775700935 , 1.8759675015077207 ,-0.20397695888897657 ,   -0.49861076029300328 , 0.041555057407175613, 1.0569715142428786  );  float sub1 = pow(L + 16.0, 3.0) / 1560896.0;  float sub2 = sub1 > 0.0088564516790356308 ? sub1 : L / 903.2962962962963;  vec3 top1 = (284517.0 * m2[0] - 94839.0 * m2[2]) * sub2;  vec3 bottom = (632260.0 * m2[2] - 126452.0 * m2[1]) * sub2;  vec3 top2 = (838422.0 * m2[2] + 769860.0 * m2[1] + 731718.0 * m2[0]) * L * sub2;  vec3 bound0x = top1 / bottom;  vec3 bound0y = top2 / bottom;  vec3 bound1x =    top1 / (bottom+126452.0);  vec3 bound1y = (top2-769860.0*L) / (bottom+126452.0);  vec3 lengths0 = hsluv_lengthOfRayUntilIntersect(hrad, bound0x, bound0y );  vec3 lengths1 = hsluv_lengthOfRayUntilIntersect(hrad, bound1x, bound1y );  return min(lengths0.r,    min(lengths1.r,    min(lengths0.g,    min(lengths1.g,    min(lengths0.b,     lengths1.b))))); }

// Function 592
float map(vec3 p){
    vec3 c = vec3(10.0);
    vec3 q = mod(p, c) - 0.5*c;
    q.y = p.y;
    
    float circle = length(q- vec3(0.0, 3.0, 0.0)) - 2.5;
    float circle2 = length(q- vec3(0.0, -1.0, 0.0)) - 2.3;
    float box = sdBox(q, vec3(2.0, 5.0, 2.0));
    float plane = fplane(p);
    float scene0 = min( max(-(min(circle, circle2)), box), plane);
    return min(scene0, length(p - vec3(0.0, 10.0, 20.0)) - 6.0);
}

// Function 593
vec3 cmap(float W) {
    vec3 C = vec3 (W);
    if (W < .3) C.y=.0; 
    else W < .6 ? C.z=.0 : C.x=.0;
    return C;
  }

// Function 594
vec3 logToneMap(vec3 c){
    
    // in my experience limit=2.2 and contrast=0.35 gives nice result
    // i prefer to leave a lot of highlights
    // also log curve fades to highlights beautifully
    // however i raised this parameters to make it look
    // somewhat identical to other algorithms here
    
    // according to my information the dynamic range of
    // human eye's light perception is around
    // 2.5 of what monitor can provide
    // though it's hightly questionable in terms of
    // both the data and my understanding of it
    
    // P.S.
    // i noticed that applying inverse tonemap transformation to
    // diffuse textures improves the final render a lot
    
    float limit = 2.5;
    float contrast = 0.3;
    
    // do the tone mapping
    
    c = log(c + 1.0) / log(limit + 1.0);
    
    // clamping for hackyContrast because s_curve
    // does not behave properly outside [0; 1]
    // if you want to keep values outside [0; 1]
    // for example for bloom post-effect
    // you would need comething better than
    // hackyContrast()
    
    c = clamp(c,0.0,1.0);
    
    // hacky contrast s_curve
    // btw it's a freatapproximation for 1-cos(x*3.14)*0.5
    
    c = mix(c, c * c * (3.0 - 2.0 * c), contrast);
    
    // this creates pleasant hue shifting
    // i amplified the effect to demonstrate it more clearly
    // usually i use it very subtle
    
    c = pow(c, vec3(1.05,0.9,1));
    
    // gamma correction
    
    return pow(c, vec3(1. / gamma));
}

// Function 595
vec3 ApplyTonemap( vec3 linearCol )
{
	const float kExposure = 0.75;
	
    float a = 0.010;
    float b = 0.132;
    float c = 0.010;
    float d = 0.163;
    float e = 0.101;

    vec3 x = linearCol * kExposure;

    return ( x * ( a * x + b ) ) / ( x * ( c * x + d ) + e );    
}

// Function 596
vec2 deformUv(vec2 uv) 
{
	float yMul = 0.92 - 0.08 * sin(uv.x * PI);
            
    if(uv.y >= 0.5)
    {
    	return vec2(uv.x, yMul*(uv.y-0.5)+0.5 );
    }
    else
    {
    	return vec2(uv.x, 0.5+yMul*(uv.y-0.5));
    }
}

// Function 597
float heightMap(vec3 p, float s) {
    float h = 0.0;
    float a = s;
    float f = FLOOR_TEXTURE_FREQ;
    for(int i=0;i<5;++i) {
        vec3 hm = textureLod(iChannel0, p.xz*f, 0.0).rgb;
        float avg = 1.0-0.33*(hm.r+hm.g+hm.b);
        h += avg*a;
        a *= 0.22;
        //f *= 1.9;
    }
    return h-textureLod(iChannel0, p.xz, 0.0).r*0.02+cos(p.x)*0.02+p.z*0.03*log(length(p));
}

// Function 598
float map(vec3 p){
    vec3 rotPX = rotateX(p, RotX*ROT_SPEED);
    vec3 rotPXY = rotateY(rotPX, RotY*ROT_SPEED);
    if ( iMouse.z > 0. ) {density=iMouse.x/iResolution.x * RUGOSITY_DENSITY_MAX;}
	float rugosity=cos(density*rotPXY.x)*sin(density*rotPXY.y)*sin(density*rotPXY.z)*cos(256.1)*sin(0.8);
	float disp=length(vec4(voronoiSphereMapping(normalize(p)),1.))*0.4-0.8;
    return length(p)-1.+disp+rugosity;}

// Function 599
float Tonemap_Unreal(float x) {
    // Unreal 3, Documentation: "Color Grading"
    // Adapted to be close to Tonemap_ACES, with similar range
    // Gamma 2.2 correction is baked in, don't use with sRGB conversion!
    return x / (x + 0.155) * 1.019;
}

// Function 600
vec2 map( in vec3 pos) {
	vec2 res = vec2(pos.y-pathterrain(pos.x,pos.z), 1.0);
	//vec2 bush =  foliage(pos);
    //res=res.x<bush.x?res:bush;
    
 	return res;
}

// Function 601
vec3 map(vec3 p) {
		
		vec3 u = p;
		float displace[6];
		
		u *= rotY(rotationY);
		u *= rotX(rotationX);
		p = deformation(p);
		vec3 s = vec3(length(p) - 1., 0., 0.);
		
		s.x += displacements(u, displace);		
		s.x *= 0.2;
		return s;
	}

// Function 602
vec2 map( in vec3 pos )
{
    vec2 res = opU( 
        			vec2( 
        				sdPlane(pos), 
                        1.0 ),
	                vec2(
                        opS(   
                        	sdSphere(
                        		pos-position, 
                        		size*selection[0] 
                        	),
                            opUx(
                            	sdSphere(
                        			pos-intersectorPosition,
                            		intersectorSize*intersectorSelection[0]
                            	),
                                opUx(
                                	sdBox(        				    
                            			pos-intersectorPosition,
        			    				vec3(intersectorSize*intersectorSelection[1])
                              		),
                                    opUx(
                                    	sdTorus(
        									pos-intersectorPosition, 
        									vec2(intersectorSize*intersectorSelection[2]) 
                        				),
                                        sdCylinder(   
                                            pos-intersectorPosition, 
                                            vec2(intersectorSize*intersectorSelection[3]) )
                                    )
                                  
                                )
                            )
						),
                    	color 
                    ) 
    			);
    res = opU( 
        		res, 
        		vec2(
                     opS(                        
                    	sdBox(        				    
                            pos-position,
        			    	vec3(size*selection[1])
                              ),
                         	  opUx(
                            	sdSphere(
                        			pos-intersectorPosition,
                            		intersectorSize*intersectorSelection[0]
                            	),
                                opUx(
                                	sdBox(        				    
                            			pos-intersectorPosition,
        			    				vec3(intersectorSize*intersectorSelection[1])
                              		),
                                    opUx(
                                    	sdTorus(
        									pos-intersectorPosition, 
        									vec2(intersectorSize*intersectorSelection[2]) 
                        				),
                                        sdCylinder(   
                                            pos-intersectorPosition, 
                                            vec2(intersectorSize*intersectorSelection[3]) )
                                    )
                                  
                                )
                            )
                    	), 
                    color
                ) 
    		);
    res = opU( res, vec2( 
        				opS(
        					sdTorus(
                            	pos-position,
        						vec2(size*selection[2])),
                             opUx(
                            	sdSphere(
                        			pos-intersectorPosition,
                            		intersectorSize*intersectorSelection[0]
                            	),
                                opUx(
                                	sdBox(        				    
                            			pos-intersectorPosition,
        			    				vec3(intersectorSize*intersectorSelection[1])
                              		),
                                    opUx(
                                    	sdTorus(
        									pos-intersectorPosition, 
        									vec2(intersectorSize*intersectorSelection[2]) 
                        				),
                                        sdCylinder(   
                                            pos-intersectorPosition, 
                                            vec2(intersectorSize*intersectorSelection[3]) )
                                    )
                                  
                                )
                             )
                        ), 
        				color) 
             );
    res = opU( res, vec2( 
        				opS(
        					sdTriPrism(
        							pos-position, 
        							vec2(size*selection[3]) 
                        	),
                            opUx(
                            	sdSphere(
                        			pos-intersectorPosition,
                            		intersectorSize*intersectorSelection[0]
                            	),
                                opUx(
                                	sdBox(        				    
                            			pos-intersectorPosition,
        			    				vec3(intersectorSize*intersectorSelection[1])
                              		),
                                    opUx(
                                    	sdTorus(
        									pos-intersectorPosition, 
        									vec2(intersectorSize*intersectorSelection[2]) 
                        				),
                                        sdCylinder(   
                                            pos-intersectorPosition, 
                                            vec2(intersectorSize*intersectorSelection[3]) )
                                    )
                                  
                                )
                             )
                        ),
        				color) 
             );
    res = opU( res, vec2( 
        				opS(
        					sdCylinder(   
        							pos-position, 
        							vec2(size*selection[4]) 
                            ),
                            opUx(
                            	sdSphere(
                        			pos-intersectorPosition,
                            		intersectorSize*intersectorSelection[0]
                            	),
                                opUx(
                                	sdBox(        				    
                            			pos-intersectorPosition,
        			    				vec3(intersectorSize*intersectorSelection[1])
                              		),
                                    opUx(
                                    	sdTorus(
        									pos-intersectorPosition, 
        									vec2(intersectorSize*intersectorSelection[2]) 
                        				),
                                        sdCylinder(   
                                            pos-intersectorPosition, 
                                            vec2(intersectorSize*intersectorSelection[3]) )
                                    )
                                  
                                )
                             )
                        ),
        				color) 
             );
    res = opU( res, vec2( 
         				opS(
        					sdHexPrism(
        						pos-position, 
        						vec2(size*selection[5]) 
                            ),
                            opUx(
                            	sdSphere(
                        			pos-intersectorPosition,
                            		intersectorSize*intersectorSelection[0]
                            	),
                                opUx(
                                	sdBox(        				    
                            			pos-intersectorPosition,
        			    				vec3(intersectorSize*intersectorSelection[1])
                              		),
                                    opUx(
                                    	sdTorus(
        									pos-intersectorPosition, 
        									vec2(intersectorSize*intersectorSelection[2]) 
                        				),
                                        sdCylinder(   
                                            pos-intersectorPosition, 
                                            vec2(intersectorSize*intersectorSelection[3]) )
                                    )
                                  
                                )
                             )
                          	
                        ),
        				color) 
             );

        
    return res;
}

// Function 603
vec2 normToTextureUv(in vec2 normUv, float texIdx, vec2 resolution)
{
    vec2 texMinRC = vec2(0., floor(texIdx * TEXTURES_INV_ROW));
    texMinRC.x = texIdx - texMinRC.y*TEXTURES_ROW;
    vec2 texMin = TEXTURES_INV_ROW * texMinRC;
    vec2 texMax = texMin + vec2(TEXTURES_INV_ROW); 
    
    vec2 btexMin = ceil(texMin * resolution + 2.) / resolution;
    vec2 btexMax = floor(texMax * resolution - 2.) / resolution;
    
    vec2 resUv = normUv * (btexMax - btexMin);
    
    vec2 tuv = btexMin + resUv;

    return tuv;
}

// Function 604
vec2 random_uv(vec2 tuv, float offset) {
    vec2 noise_of = noise(floor(tuv * vec2(1.0 / 3.0, 1.0 / 2.0)) + offset);
    return (tuv + noise_of) * (sign(mod(noise_of, 0.001) - 0.0005));
}

// Function 605
vec3 UVToFisheyeCoord(float U, float V, float MinCos)
{
    float NdcX = U * 2.0 - 1.0;
    float NdcY = V * 2.0 - 1.0;
    NdcX *= iResolution.x / iResolution.y;
    
    float R = Sqrt(NdcX * NdcX + NdcY * NdcY);
    
    // x - sin(theta), z - cos(theta)
    vec3 Dir = vec3(NdcX / R, 0.0, NdcY / R);
    
    float Phi = clamp(R, 0.0, 1.0) * kPi * 0.75;
    
    Dir.y   = clamp(cos(Phi), MinCos, 1.0);
	Dir.xz *= Sqrt(1.0 - Dir.y * Dir.y);
    return Dir;
}

// Function 606
float map(vec3 p)
{
  p.x += sin(p.z*5.+sin(p.y*5.))*0.3;
  return (length(p)-1.)*0.7;
}

// Function 607
vec3 UVToViewSpaceCoord(float U, float V, float MinCos)
{
    float HalfFovV = kPi / 6.0;
    float AspRatio = iResolution.x / iResolution.y;

    float yScale = cos(HalfFovV) / sin(HalfFovV);
    float xScale = yScale / AspRatio;

    vec3 Dir;
    Dir.z = 10.0 / kKilometersToMeters;
    Dir.x = (U * 2.0 - 1.0) / xScale * Dir.z;
    Dir.y = (V * 2.0 - 1.0) / yScale * Dir.z;
    Dir = normalize(Dir);
    // clamp cosine of zenith angle
    Dir.xz /= Sqrt(1.0 - Dir.y * Dir.y);
    Dir.y   = clamp(Dir.y, MinCos, 1.0);
    Dir.xz *= Sqrt(1.0 - Dir.y * Dir.y);
    return Dir;
}

// Function 608
vec3 SampleNormalMap(in vec2 uv, in float height)
{
    const float strength = 40.0;    
    float d0 = SampleTexture(uv.xy);
    float dX = SampleTexture(uv.xy - vec2(EPSILON, 0.0));
    float dY = SampleTexture(uv.xy - vec2(0.0, EPSILON));
    return normalize(vec3((dX - d0) * strength, (dY - d0) * strength, 1.0));
}

// Function 609
vec3 doBumpMapBrick(in vec3 p, in vec3 nor, float bumpfactor){
	vec3 n = abs(nor);
    const vec2 e = vec2(0.001, 0);
    float ref = bumpSurf3D(p, nor);                 
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy, n), bumpSurf3D(p - e.yxy, n), bumpSurf3D(p - e.yyx, n) )-ref)/e.x;                     
    grad -= nor*dot(nor, grad);                            
    return normalize( nor + grad*bumpfactor );
	
}

// Function 610
float map_rods(vec3 pos)
{
    pos.y+= 0.1;
    for (int i = 0; i < 7; ++i)
    {
    	pos.x=abs(pos.x);
    	pos.x-=0.61;
    }
    return sdCylinder(pos, vec2(0.13, 2.75));   
}

// Function 611
vec3 map(in vec3 p) {
	
	float res = 0.;
	
    vec3 c = p;
	for (int i = 0; i < DETAIL; ++i) {
        p =.7*abs(p)/dot(p,p) -.7;
        p.yz= csqr(p.yz);
        p=p.zxy;
        res += exp(-19. * abs(dot(p,c)))+.02;
        
	}
	return res*COLOR_CONTRAST*0.013*(normalize(p)+(1.0-OPACITY_OF_COLOR)*vec3(1.0));
}

// Function 612
vec3 ToneMapFilmicALU(vec3 x)
{
    x *= 0.665;
    
   #if 0
    x = max(vec3(0.0), x - 0.004f);
    x = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
    
    x = sRGB_InvEOTF(x);
   #else
    x = max(vec3(0.0), x);
    x = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
    
    x = pow(x, vec3(2.2));// using gamma instead of sRGB_InvEOTF + without x - 0.004f looks about the same
   #endif
    
    return x;
}

// Function 613
vec2 map(vec3 p) {
    float d1 = sdSphere(p);
    float d2 = sdPlane(p);
    float id = (d1 < d2) ? 0.: 1.;
    return vec2(min(d1, d2), id);
}

// Function 614
float MapTree( vec3 p)
{  
  float terrainHeight = GetTerrainHeight(p);
  float treeHeight =GetTreeHeight(p, terrainHeight);

  // get terrain height at position and tree height onto that
  return  p.y - terrainHeight-treeHeight;
}

// Function 615
vec2 map(vec3 ro) {
    
    //beveled cubes
    vec2 disp = vec2(1., 0.);
    float d = bevcube(ro);
    d = min(d, bevcube(ro + disp.xyy));
    d = min(d, bevcube(ro - disp.xyy));
    d = min(d, bevcube(ro + disp.yyx));
    d = min(d, bevcube(ro - disp.yyx));
    d = min(d, bevcube(ro - disp.yxy));
    
    float d2 = min(d, length(ro + vec3(0., 10., 0.)) - 9.6);
    if (d2 < d) {
        return vec2(d2, 2.0);
    } else {
    	return vec2(d, 1.0);
    }
}

// Function 616
float map(vec3 p)
{ return sdf(rotate(p, iTime) / SDF_SCALE) * SDF_SCALE; }

// Function 617
vec2 uv (vec3 p) {
	float x = p.x;
    float y = p.y;
    float z = p.z;
    float u = atan(x, z) / (2. * pi) + .5;
    float v = asin(y) / (pi) + .5;
    return vec2(u,v);
}

// Function 618
vec3 triPlanarMap(sampler2D inTexture, float contrast, vec3 normal, vec3 position)
{
    vec3 xTex = textureLod(inTexture, (position).yz, 0.0).rgb;
    vec3 yTex = textureLod(inTexture, (position).xz, 0.0).rgb;
    vec3 zTex = textureLod(inTexture, -(position).xy, 0.0).rgb;
    vec3 weights = normalize(abs(pow(normal.xyz, vec3(contrast))));
    
    return vec3(xTex*weights.x + yTex*weights.y + zTex*weights.z);
}

// Function 619
vec2 lnglat2uv(vec2 lnglatlobe) {
    vec2 result = lnglatlobe.xy / vec2(pi, halfPi);
    result = 0.5 * result + vec2(0.5);
    return result;
}

// Function 620
vec2 map( in vec3 pos )
{
    // We keep a collection of more interesting primitives to show shadows
    
    // central sphere on plane
    vec2 res = opU( vec2( sdPlane(     pos), 1.0 ),
	                vec2( sdSphere(    pos-vec3( 0.0,0.25, 0.0), 0.25 ), 46.9 ) );
    
    // tiled Isohedral Tetrahedron
    float sT = sin(iTime), cT = cos(iTime),
          o = cT*1.5*(1.+cos(2.1+sT)), h=.9,
          s = .15, u = s+s;
    res = opU( res, vec2(ISOTET_BLU(pos-vec3(-o, h, o),s,u), BLU ) );
    res = opU( res, vec2(ISOTET_YEL(pos-vec3( 0, h, o),s,u), YEL ) );
    res = opU( res, vec2(ISOTET_CYN(pos-vec3( o, h, o),s,u), CYN ) );
    res = opU( res, vec2(ISOTET_RED(pos-vec3(-o, h,-o),s,u), RED ) );
    res = opU( res, vec2(ISOTET_GRN(pos-vec3( 0, h,-o),s,u), GRN ) );
    res = opU( res, vec2(ISOTET_PRP(pos-vec3( o, h,-o),s,u), PRP ) );

    // cluster spheres
    res = opU( res, vec2(sdSpheresCursor(pos-vec3(-2, .3, 1),.2,.08), BLU ) );

    // truncated octahedron
    res = opU( res, vec2(sdTruncOct(pos-vec3(-2., .3, 0.),.2,.1), RED ) );

    // regular octahedron
    res = opU( res, vec2(sdOctahedron(pos-vec3(-3., .1, 0.),.25), GRN ) );
        
    // box minus sphere
    res = opU( res, vec2( opS(
		             udRoundBox(  pos-vec3(-1., .3, 0.), vec3(0.15),0.05),
	                 sdSphere(    pos-vec3(-1., .3, 0.), 0.25)), 13.0 ) );

    // tri prism minus box
    res = opU( res, vec2( opS(
					 sdTriPrism(  pos-vec3(2., .3, 0.), vec2(0.45,0.05)), 
		             udRoundBox(  pos-vec3(2., .3, 0.), vec3(0.15),0.05)
             )));
    
  	// twisted torus
	res = opU( res, vec2( 0.5*sdTorus( opTwist(pos-vec3(1., 0.35, 0.)),vec2(.2, .05)), 46.7 ) );

    return res;
}

// Function 621
vec3 GetMipMapUVW_Dir(vec3 _uvw,vec3 _axis){
    _uvw = floor((_uvw+1.)*512.)+0.5;
    vec3 a = exp2(floor(log2(_uvw)));
    return normalize((_uvw*2./a - 3.)*(1.-abs(_axis))+_axis);
}

// Function 622
vec3 jodieReinhardTonemap(vec3 c){
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 tc = c / (c + 1.0);

    return mix(c / (l + 1.0), tc, tc);
}

// Function 623
vec2 get_reflection_uv(vec3 p, vec3 n_reflection)
{
    const float reflection_distane = 1.0;
    
    p += n_reflection * reflection_distane;
    return(compute_uv(normalize(p - cubemap_origin)));
}

// Function 624
vec2 Cam_GetViewCoordFromUV( vec2 vUV, vec2 res )
{
	vec2 vWindow = vUV * 2.0 - 1.0;
	vWindow.x *= res.x / res.y;

	return vWindow;	
}

// Function 625
float Tonemapping(float x ){    
    float calc;
    calc = ((x*(shoulderStrenght*x + linearAngle*linearStrenght) + toeStrenght*toeNumerator)/(x*(shoulderStrenght*x+linearStrenght)+toeStrenght* toeDenominator))- toeNumerator/ toeDenominator;                        
    return calc;    
}

// Function 626
vec4 lchToHpluv(vec4 c) {return vec4( lchToHpluv( vec3(c.x,c.y,c.z) ), c.a);}

// Function 627
float map(vec3 pos){
    float d = min(1.,cube  (pos-vec3(0.,-0.5, 0.),vec3(3.5,0.05,3.5),0.01));
    return d;
}

// Function 628
float seaGeometryMap(vec3 p) 
{
    #if WATER_TYPE == WAVES_WATER
    vec2 uv = p.xz * vec2(0.85, 1.0);
	
    float freq 	 = SEA_FREQ;
    float amp 	 = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    
    float d = 0.0;
    float h = 0.0;    
    for (int i = 0; i < SEA_GEOMETRY_ITERATIONS; ++i) 
    {   
		#if FANTASY_WATER_PATH
        if (uv.x > UV_START_X && uv.x < UV_END_X)
	   	{
			continue;
	   	}
		#endif

    	d =  seaOctave((uv + SEA_CURRENT_TIME) * freq, choppy);
    	d += seaOctave((uv - SEA_CURRENT_TIME) * freq, choppy);
        h += d * amp; 
	    
		freq *= SEA_GEOMETRY_FREQ_MUL; 
		amp  *= SEA_GEOMETRY_AMPLITUDE_MUL;
	    
        choppy = mix(choppy, SEA_CHOPPY_MIX_VALUE, SEA_CHOPPY_MIX_FACTOR);
	    
		uv *= octaveMatrix; 
    }
    return p.y - h;
    #else
    return p.y;
    #endif
}

// Function 629
vec3 filmicToneMapping(vec3 color)
{
	color = max(vec3(0.), color - vec3(0.004));
	color = (color * (6.2 * color + .5)) / (color * (6.2 * color + 1.7) + 0.06);
	return color;
}

// Function 630
vec4   luvToRgb(vec4 c) {return vec4(   luvToRgb( vec3(c.x,c.y,c.z) ), c.a);}

// Function 631
vec3 heightmap (vec3 n)
{
	return vec3(fbm((5.0 * n) + fbm((5.0 * n) * 3.0 - 1000.0) * 0.05),0,0);
}

// Function 632
vec2 map(vec3 pos)
{
    vec4 sphere = sphere_info;
    
    pos = rot(iTime) * pos;
    
    vec2 d1 = vec2(drawSphere(pos,sphere.xyz,sphere.w),0.1);
    
    vec3 posA = vec3(0.0,0.9,0.0);
    vec3 posB = vec3(0.0,-0.9,0.0);
    
    vec2 d2 = vec2(drawCapsule(pos,posA,posB,0.05),1.1);
    
    return opU(d1,d2);
}

// Function 633
float map( in vec3 p )
{
    return p.y - terrain(p.xz);
}

// Function 634
vec3 doBumpMap( sampler2D tx, in vec3 p, in vec3 n, float bf){
   
    const vec2 e = vec2(EPS, 0);
    
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    
    vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
    
}

// Function 635
float map( vec3 p )
{
	vec3 q = vec3( length(p.xz)-2.0, p.y, mod(0.1*iTime + 6.0*atan(p.z,p.x)/3.14,1.0)-0.5 );

    float d1 = length(p) - 1.0;
    float d2 = length(q) - 0.2;
	
	return min(d1,d2);
}

// Function 636
vec3 doBumpMap( sampler2D tx, in vec3 p, in vec3 n, float bf){
   
    const vec2 e = vec2(0.001, 0);
    
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    
    vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
    
}

// Function 637
float clamp_uv(vec2 uv)
{
    return float((uv.x > 0.0 && uv.x < 1.0) && (uv.y > 0.0 && uv.y < 1.0));
}

// Function 638
vec2 get_uv (vec2 coord, vec2 res)
{
    /* remap to [0,1]^2 */
    vec2 uv = coord / res.xy;
    /* remap to [-1,1]^2 */
    uv = 2.0*uv - vec2(1.0);
    /* remap x to [-aspect_ratio, aspect_ratio] */
    float aspect_ratio = res.x/res.y;
    uv.x *= aspect_ratio;
    
    return uv;
}

// Function 639
float mapH(vec3 p) {
    return - p.z + .1 * (t2d(p.xy).a - .015*pow(t2d(p.xy).r, .125));
}

// Function 640
float remap(float x, float a, float b, float c, float d)
{
    return (((x - a) / (b - a)) * (d - c)) + c;
}

// Function 641
vec3 hpluvToLch(vec3 tuple) {  tuple.g *= hsluv_maxSafeChromaForL(tuple.b) * .01;  return tuple.bgr; }

// Function 642
mat2 uvRotate(float a) { return mat2(cos(a),sin(a),-sin(a),cos(a)); }

// Function 643
float map(in vec3 p) {
    
    vec3 q = mod(p+2.0, 4.0)-2.0;
    
 	float d1 = length(q) - 1.0;
    
    d1 += 0.1*sin(10.0*p.x)*sin(10.0*p.y + iTime )*sin(10.0*p.z);
    
 	float d2 = p.y + 1.0;
    
    float k = 1.0;
    float h = clamp(0.5 + 0.5 *(d1-d2)/k, 0.0, 1.0);
        
    return mix(d1, d2, h) - k*h*(1.0-h);
}

// Function 644
vec4 doMap(vec3 voro) {
    vec3 v = voro*0.5;
    float height = 0.1+0.9*disp(v);
    //v.x=(-0.05+v.x);
    return vec4(v, height);
}

// Function 645
float remap( float x, float a, float b ) { return clamp( ( x - a ) / ( b - a ), 0., 1. ); }

// Function 646
vec2 getUV(vec2 fragCoord, float xtiles)
{
    float tileSize = iResolution.x / xtiles;
    float ytiles = floor(iResolution.y / tileSize);
    // step the uvs to get a pixellated effect. later render to a small fbo and resize
    vec2 steps = vec2(xtiles,ytiles);
    vec2 uv = (fragCoord.xy / iResolution.xy)*steps;
    return floor(uv)/steps;
}

// Function 647
vec3 hpluvToRgb(vec3 tuple) {  return lchToRgb(hpluvToLch(tuple)); }

// Function 648
vec2 map( vec3 p )
{
    p.x -= -0.5;
	p.y -= 2.4;
    
    vec2 res = vec2( 2.15+p.y, 0.0 );
    
    // elephant bounding volume
    //float bb = length(p-vec3(1.0,-0.75,0))-2.0;
    //if( bb>res.x )  return res;

    
    vec3 ph = p;
    const float cc = 0.995;
    const float ss = 0.0998745;
    ph.yz = mat2(cc,-ss,ss,cc)*ph.yz;
    ph.xy = mat2(cc,-ss,ss,cc)*ph.xy;
    
    // head
    float d1 = sdEllipsoid( ph, vec3(0.0,0.05,0.0), vec3(0.45,0.5,0.3) );
    d1 = smin( d1, sdEllipsoid( ph, vec3(-0.3,0.15,0.0), vec3(0.2,0.2,0.2) ), 0.1 );

    // nose
    vec2  kk;
    vec2  b1 = sdBezier( vec3(-0.15,-0.05,0.0), vec3(-0.7,0.0,0.0), vec3(-0.7,-0.8,0.0), ph, kk );
    float tr1 = 0.30 - 0.17*smoothstep(0.0,1.0,b1.y);
    vec2  b2 = sdBezier( vec3(-0.7,-0.8,0.0), vec3(-0.7,-1.5,0.0), vec3(-0.4,-1.6,0.2), ph, kk );
    float tr2 = 0.30 - 0.17 - 0.05*smoothstep(0.0,1.0,b2.y);
    float bd1 = b1.x-tr1;
    float bd2 = b2.x-tr2;
    float nl = b1.y*0.5;
    float bd = bd1;
    if( bd2<bd1 )
    {
        nl = 0.5 + 0.5*b2.y;
        bd = bd2;
    }
    float d2 = bd;
    float xx = nl*120.0;
    float ff = sin(xx + sin(xx + sin(xx + sin(xx))));
    d2 += 0.003*ff*(1.0-nl)*(1.0-nl)*smoothstep(0.0,0.1,nl);

    d2 -= 0.005;
    
    float d = smin(d1,d2,0.2);

    // teeth
    vec3 q = vec3( p.xy, abs(p.z) );
    vec3 qh = vec3( ph.xy, abs(ph.z) );
    {
    vec2 s1 = sdSegment( qh, vec3(-0.4,-0.1,0.1), vec3(-0.5,-0.4,0.28) );
    float d3 = s1.x - 0.18*(1.0 - 0.3*smoothstep(0.0,1.0,s1.y));
    d = smin( d, d3, 0.1 );
    }
    
    // eyes
    {
    vec2 s1 = sdSegment( qh, vec3(-0.2,0.2,0.11), vec3(-0.3,-0.0,0.26) );
    float d3 = s1.x - 0.19*(1.0 - 0.3*smoothstep(0.0,1.0,s1.y));
    d = smin( d, d3, 0.03 );

    float st = length(qh.xy-vec2(-0.31,-0.02));
    d += 0.0015*sin(250.0*st)*(1.0-smoothstep(0.0,0.2,st));
        
    const mat3 rot = mat3(0.8,-0.6,0.0,
                          0.6, 0.8,0.0,
                          0.0, 0.0,1.0 );
    float d4 = sdEllipsoid( rot*(qh-vec3(-0.31,-0.02,0.34)), vec3(0.0), vec3(0.1,0.08,0.07)*0.7 );
	d = smax(d, -d4, 0.02 );
    }
   

    // body
    {
    const float co = 0.92106099;
    const float si = 0.38941834;
    vec3 w = p;
    w.xy = mat2(co,si,-si,co)*w.xy;

    float d4 = sdEllipsoid( w, vec3(0.6,0.3,0.0), vec3(0.6,0.6,0.6) );
	d = smin(d, d4, 0.1 );

    d4 = sdEllipsoid( w, vec3(1.8,0.3,0.0), vec3(1.2,0.9,0.7) );
	d = smin(d, d4, 0.2 );

    d4 = sdEllipsoid( w, vec3(2.1,0.55,0.0), vec3(1.0,0.9,0.6) );
	d = smin(d, d4, 0.1 );

    d4 = sdEllipsoid( w, vec3(2.0,0.6,0.0), vec3(0.9,0.7,0.8) );
	d = smin(d, d4, 0.1 );
    }

    // back-left leg
    {
    float d3 = leg( q, vec3(2.6,-0.5,0.3), vec3(2.65,-1.45,0.3), vec3(2.6,-2.1,0.25), 1.0, 0.0 );
    d = smin(d,d3,0.1);
    }
    
    // front-left leg
    {
    float d3 = leg( p, vec3(0.8,-0.4,0.3), vec3(0.7,-1.55,0.3), vec3(0.8,-2.1,0.3), 1.0, 0.0 );
    d = smin(d,d3,0.15);
    d3 = leg( p, vec3(0.8,-0.4,-0.3), vec3(0.4,-1.55,-0.3), vec3(0.4,-2.1,-0.3), 1.0, 0.0 );
    d = smin(d,d3,0.15);
    }
    
    // ear
    const float co = 0.8775825619;
    const float si = 0.4794255386;
    vec3 w = qh;
    w.xz = mat2(co,si,-si,co)*w.xz;
    
    vec2 ep = w.zy - vec2(0.5,0.4);
    float aa = atan(ep.x,ep.y);
    float al = length(ep);
    w.x += 0.003*sin(24.0*aa)*smoothstep(0.0,0.5,dot(ep,ep));
                      
    float r = 0.02*sin( 24.0*atan(ep.x,ep.y))*clamp(-w.y*1000.0,0.0,1.0);
    r += 0.01*sin(15.0*w.z);
    // section        
    float d4 = length(w.zy-vec2( 0.5,-0.2+0.03)) - 0.8 + r;    
    float d5 = length(w.zy-vec2(-0.1, 0.6+0.03)) - 1.5 + r;    
    float d6 = length(w.zy-vec2( 1.8, 0.1+0.03)) - 1.6 + r;    
    d4 = smax( d4, d5, 0.1 );
    d4 = smax( d4, d6, 0.1 );

    float wi = 0.02 + 0.1*pow(clamp(1.0-0.7*w.z+0.3*w.y,0.0,1.0),2.0);
    w.x += 0.05*cos(6.0*w.y);
    
    // cut it!
    d4 = smax( d4, -w.x, 0.03 ); 
    d4 = smax( d4, w.x-wi, 0.03 );     
    
    d = smin( d, d4, 0.3*max(qh.y,0.0) ); // trick -> positional smooth
    
    // conection hear/head
    vec2 s1 = sdBezier( vec3(-0.15,0.3,0.0), vec3(0.1,0.6,0.2), vec3(0.35,0.6,0.5), qh, kk );
    float d3 = s1.x - 0.08*(1.0-0.95*s1.y*s1.y);
    d = smin( d, d3, 0.05 );    
    
    res.x = min( res.x, d );
    
	//------------------
    // teeth
    vec2 b = sdBezier( vec3(-0.5,-0.4,0.28), vec3(-0.5,-0.7,0.32), vec3(-1.0,-0.8,0.45), qh, kk );
    d2 = b.x - 0.10 + 0.08*b.y;
    if( d2<res.x ) 
    {
        res = vec2( d2, 1.0 );
    }
    
	//------------------
    //eyeball
    const mat3 rot = mat3(0.8,-0.6,0.0,
                          0.6, 0.8,0.0,
                          0.0, 0.0,1.0 );
    d4 = sdEllipsoid( rot*(qh-vec3(-0.31,-0.02,0.33)), vec3(0.0), vec3(0.1,0.08,0.07)*0.7 );
    if( d4<res.x ) res = vec2( d4, 2.0 );

    // floor plane
    res.x = smax( res.x, -2.2-p.y, 0.1 );
          
    return res;
}

// Function 649
vec3 mapD1(float t)
{
    return -7.0*a*c*cos(t+m)*sin(7.0*t+n) - a*sin(t+m)*(b+c*cos(7.0*t+n));
}

// Function 650
vec3 Tonemap(vec3 color,float gamma,float luma){
    vec3 c = exp(-1.0 / (2.72 * color + 0.15));
    c = pow(c, vec3(1.0 / (gamma * luma)));
    return c;
}

// Function 651
float errorCheckLocalUV(vec2 checkUV){
  return float((abs(checkUV.x) > .5 * TILE_LOCAL_LENGTH)||(abs(checkUV.y) > .5 * TILE_LOCAL_LENGTH));
}

// Function 652
vec3 doBumpMap(in vec3 p, in vec3 n, float bumpfactor, inout float edge){
    
    // Resolution independent sample distance... Basically, I want the lines to be about
    // the same pixel with, regardless of resolution... Coding is annoying sometimes. :)
    vec2 e = vec2(2./iResolution.y, 0); 
    
    float f = bumpFunction(p); // Hit point function sample.
    
    float fx = bumpFunction(p - e.xyy); // Nearby sample in the X-direction.
    float fy = bumpFunction(p - e.yxy); // Nearby sample in the Y-direction.
    float fz = bumpFunction(p - e.yyx); // Nearby sample in the Y-direction.
    
    float fx2 = bumpFunction(p + e.xyy); // Sample in the opposite X-direction.
    float fy2 = bumpFunction(p + e.yxy); // Sample in the opposite Y-direction.
    float fz2 = bumpFunction(p + e.yyx);  // Sample in the opposite Z-direction.
    
     
    // The gradient vector. Making use of the extra samples to obtain a more locally
    // accurate value. It has a bit of a smoothing effect, which is a bonus.
    vec3 grad = vec3(fx - fx2, fy - fy2, fz - fz2)/(e.x*2.);  
    //vec3 grad = (vec3(fx, fy, fz ) - f)/e.x;  // Without the extra samples.


    // Using the above samples to obtain an edge value. In essence, you're taking some
    // surrounding samples and determining how much they differ from the hit point
    // sample. It's really no different in concept to 2D edging.
    edge = abs(fx + fy + fz + fx2 + fy2 + fz2 - 6.*f);
    edge = smoothstep(0., 1., edge/e.x);
    
    // Some kind of gradient correction. I'm getting so old that I've forgotten why you
    // do this. It's a simple reason, and a necessary one. I remember that much. :D
    grad -= n*dot(n, grad);          
                      
    return normalize(n + grad*bumpfactor); // Bump the normal with the gradient vector.
	
}

// Function 653
vec3 lchToHpluv(vec3 tuple) {
    tuple.g /= hsluv_maxSafeChromaForL(tuple.r) * .01;
    return tuple.bgr;
}

// Function 654
float treesMap( in vec3 p, in float rt, out float oHei, out float oMat, out float oDis )
{
    oHei = 1.0;
    oDis = 0.1;
    oMat = 0.0;
        
    float base = terrainMap(p.xz).x; 
    
    float d = 10.0;
    vec2 n = floor( p.xz );
    vec2 f = fract( p.xz );
    for( int j=0; j<=1; j++ )
    for( int i=0; i<=1; i++ )
    {
        vec2  g = vec2( float(i), float(j) ) - step(f,vec2(0.5));
        vec2  o = hash2( n + g );
        vec2  v = hash2( n + g + vec2(13.1,71.7) );
        vec2  r = g - f + o;

        float height = kMaxTreeHeight * (0.4+0.8*v.x);
        float width = 0.9*(0.5 + 0.2*v.x + 0.3*v.y);
        vec3  q = vec3(r.x,p.y-base-height*0.5,r.y);
        float k = sdEllipsoidY( q, vec2(width,0.5*height) );

        if( k<d )
        { 
            d = k;
            //oMat = hash1(o); //fract(o.x*7.0 + o.y*15.0);
            oMat = o.x*7.0 + o.y*15.0;
            oHei = (p.y - base)/height;
            oHei *= 0.5 + 0.5*length(q) / width;
        }
    }
    oMat = fract(oMat);

    // distort ellipsoids to make them look like trees (works only in the distance really)
    #ifdef LOWQUALITY
    if( rt<350.0 )
    #else
    if( rt<500.0 )
    #endif
    {
        float s = fbm_4( p*3.0 );
        s = s*s;
        oDis = s;
        #ifdef LOWQUALITY
        float att = 1.0-smoothstep(150.0,350.0,rt);
        #else
        float att = 1.0-smoothstep(200.0,500.0,rt);
        #endif
        d += 2.0*s*att*att;
    }
    
    return d;
}

// Function 655
uint spheremap_32( in vec3 nor )
{
    vec2 v = nor.xy * inversesqrt(2.0*nor.z+2.0);
    return packSnorm2x16(v);
}

// Function 656
float map(in vec3 rp, inout AA aa)
{
	float mt = iTime * 0.9;
    float t = sin(mt + rp.x * 1.2);
    t += sin(mt + rp.z * 1.4);

    vec2 off = rot2d(vec3(0.0, 5.0, 0.0), rp.y * 5.* t);
    rp.x -= off.x * 0.04;
    rp.y -= off.y * 0.005;

    float h =
        //texture(iChannel0, uv -100.0).r;
        noise(floor(rp.xz*516.));

#ifdef AA_ENABLED
        rotate(aa, h);
        h = avg(aa);
#endif

    h *=
        //mix(texture(iChannel0, uv * 0.025).r + 0., 1.0, 0.7);
        mix(noise(floor(rp.xz*123.)), 1.0, 0.6)*H;
    warpedRp = rp;
    return rp.y - h;
}

// Function 657
float map(vec3 p){

    // Cheap, lame distortion, if you wanted it.
    //p.xy += sin(p.xy*7. + cos(p.yx*13. + iTime))*.01;
    
    // Back plane, placed at vec3(0, 0, 1), with plane normal vec3(0., 0., -1).
    // Adding some height to the plane from the heightmap. Not much else to it.
    float d = 1. - p.z;
    //if (d<0.2) 
        d-= heightMap(p.xy)*.125;
    return d;
    
}

// Function 658
vec2 map(vec3 pos)
{
    float jelly = map_jelly(pos);
    float container = map_container(pos);
    float rods = map_rods(pos);
    vec2 res = opU(vec2(jelly, JELLY_OBJ), vec2(container, CONTAINER_OBJ));
    res = opU(res, vec2(rods, RODS_OBJ));
    res = opU(res, map_slime(pos));
    return res;
}

// Function 659
float map(vec3 p)
{
    p-=path(p.z);
    float d0=4.0-length(p.xy*vec2(0.2,0.4));  // Tunnel
    float d1=schwarz(p*0.08); // Schwarz
    float d2=dot(sin(p*0.6),cos(p.yzx*0.3)); // Gyroid
    float d=smax(d0,-d1,1.);
    d=smax(d,-d2,1.0);
    return d;
}

// Function 660
float map_jelly(vec3 pos)
{
    vec2 hmp = getJellyMPos(pos);
    float hm = 0.8*getJellyBump(hmp)*smoothstep(0.95, 0.7, abs(pos.x/6.))*smoothstep(1., 0.8, abs(pos.y/2.5));
    float posz2 = pow(abs(pos.z), 1.9) + 0.04;
    //float res = max(sdBox(pos, vec3(7., 2.5, 2.)), 1.1*posz2*smoothstep(0., 0.2, hm) - hm + 0.16) + smoothstep(0., 2.5, abs(pos.z));
    float res = 1.1*posz2*smoothstep(0., 0.2, hm) - hm + 0.1 + smoothstep(0.0, 2.25, abs(pos.z));
    return res;
}

// Function 661
vec2 uv_nearest( vec2 uv, vec2 res )
{
	uv = uv * res;
    return (floor(uv) + 0.5) / res;
}

// Function 662
define UVH2POS(aabb,uvd,pos) { vec3 aabbDim = aabb.max_ - aabb.min_; pos = aabb.min_ + uvd*aabbDim; }

// Function 663
vec3 doBumpMap( sampler2D tex, in vec3 p, in vec3 nor, float bumpfactor){
   
    const float eps = 0.001;
    vec3 grad = vec3( getGrey(tex3D(tex, vec3(p.x-eps, p.y, p.z), nor)),
                      getGrey(tex3D(tex, vec3(p.x, p.y-eps, p.z), nor)),
                      getGrey(tex3D(tex, vec3(p.x, p.y, p.z-eps), nor)));
    
    grad = (grad - getGrey(tex3D(tex,  p , nor)))/eps; 
            
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
	
}

// Function 664
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    if ( iFrame > 120 )
        discard;
    fragColor = textureLod( iChannel0, rayDir, 0.0 );
    fragColor = fragColor * fragColor;
    float kEnvmapExposure = 0.999;
    fragColor = -log2(1.0 - fragColor * kEnvmapExposure);    
    return;
}

// Function 665
float MapMountains(in vec3 p)
{       
  return p.y -  GetMountainHeight(p);
}

// Function 666
vec2 terrainMap( in vec2 p )
{
    const float sca = 0.0010;
    const float amp = 300.0;

    p *= sca;
    float e = fbm_9( p + vec2(1.0,-2.0) );
    float a = 1.0-smoothstep( 0.12, 0.13, abs(e+0.12) ); // flag high-slope areas (-0.25, 0.0)
    e = e + 0.15*smoothstep( -0.08, -0.01, e );
    e *= amp;
    return vec2(e,a);
}

// Function 667
vec2 map( vec3 p )
{
    vec2 d2 = vec2( p.y+1.0, 2.0 );

	float r = 1.0;
	float f = smoothstep( 0.0, 0.5, sin(3.0+iTime) );
	float d = 0.5 + 0.5*sin( 4.0*p.x + 0.13*iTime)*
		                sin( 4.0*p.y + 0.11*iTime)*
		                sin( 4.0*p.z + 0.17*iTime);
    r += f*0.4*pow(d,4.0);//*(0.5-0.5*p.y);
    vec2 d1 = vec2( length(p) - r, 1.0 );

    if( d2.x<d1.x) d1=d2;

	p = vec3( length(p.xz)-2.0, p.y, mod(iTime + 6.0*atan(p.z,p.x)/3.14,1.0)-0.5 );
	//p -= vec3( 1.5, 0.0, 0.0 );
    vec2 d3 = vec2( 0.5*(length(p) - 0.2), 3.0 );
    if( d3.x<d1.x) d1=d3;

	
	return d1;
}

// Function 668
vec4 rgbToHpluv(vec4 c) {return vec4( rgbToHpluv( vec3(c.x,c.y,c.z) ), c.a);}

// Function 669
vec4 earthHeightmap(vec2 p) {

    uint v = 0u;
	v = p.y == 99. ? 0u : v;
	v = p.y == 98. ? 0u : v;
	v = p.y == 97. ? 0u : v;
	v = p.y == 96. ? 0u : v;
	v = p.y == 95. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 555814912u : (p.x < 64. ? 286335794u : (p.x < 72. ? 1114113u : (p.x < 80. ? 572588048u : (p.x < 88. ? 1114402u : 0u))))))))))) : v;
	v = p.y == 94. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 268435456u : (p.x < 56. ? 536870946u : (p.x < 64. ? 74291u : (p.x < 72. ? 858988816u : (p.x < 80. ? 860107571u : (p.x < 88. ? 286401059u : (p.x < 96. ? 17u : (p.x < 104. ? 0u : (p.x < 112. ? 268435456u : (p.x < 120. ? 272u : 0u))))))))))))))) : v;
	v = p.y == 93. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 823197952u : (p.x < 64. ? 572588050u : (p.x < 72. ? 1145320226u : (p.x < 80. ? 1145390148u : (p.x < 88. ? 322122564u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 1052672u : 0u)))))))))))))) : v;
	v = p.y == 92. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 268435456u : (p.x < 56. ? 554766336u : (p.x < 64. ? 553648128u : (p.x < 72. ? 1127359026u : (p.x < 80. ? 1431655748u : (p.x < 88. ? 38028356u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 16u : (p.x < 152. ? 0u : (p.x < 160. ? 16777216u : 0u)))))))))))))))))))) : v;
	v = p.y == 91. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 33624064u : (p.x < 64. ? 0u : (p.x < 72. ? 857735168u : (p.x < 80. ? 1432769860u : (p.x < 88. ? 860111957u : (p.x < 96. ? 1u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 69632u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 16u : 0u)))))))))))))))))))) : v;
	v = p.y == 90. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 268435456u : (p.x < 56. ? 17825792u : (p.x < 64. ? 1u : (p.x < 72. ? 838860800u : (p.x < 80. ? 1717986644u : (p.x < 88. ? 19088726u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 256u : (p.x < 144. ? 0u : (p.x < 152. ? 16u : 0u))))))))))))))))))) : v;
	v = p.y == 89. ? (p.x < 8. ? 1u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 17895424u : (p.x < 48. ? 268435456u : (p.x < 56. ? 0u : (p.x < 64. ? 8464u : (p.x < 72. ? 268435456u : (p.x < 80. ? 1717916739u : (p.x < 88. ? 270611814u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 286261248u : (p.x < 168. ? 17u : (p.x < 176. ? 4369u : 0u)))))))))))))))))))))) : v;
	v = p.y == 88. ? (p.x < 8. ? 0u : (p.x < 16. ? 69888u : (p.x < 24. ? 19018256u : (p.x < 32. ? 1048576u : (p.x < 40. ? 2u : (p.x < 48. ? 0u : (p.x < 56. ? 1048576u : (p.x < 64. ? 2166784u : (p.x < 72. ? 16777216u : (p.x < 80. ? 1717916722u : (p.x < 88. ? 3425365u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 536870912u : (p.x < 120. ? 273u : (p.x < 128. ? 0u : (p.x < 136. ? 16777216u : (p.x < 144. ? 0u : (p.x < 152. ? 553648128u : (p.x < 160. ? 285282850u : (p.x < 168. ? 16777489u : (p.x < 176. ? 286335232u : (p.x < 184. ? 273u : (p.x < 192. ? 285212672u : 17825792u)))))))))))))))))))))))) : v;
	v = p.y == 87. ? (p.x < 8. ? 1u : (p.x < 16. ? 572592384u : (p.x < 24. ? 286331170u : (p.x < 32. ? 286261249u : (p.x < 40. ? 17891601u : (p.x < 48. ? 0u : (p.x < 56. ? 1114385u : (p.x < 64. ? 571473920u : (p.x < 72. ? 285212672u : (p.x < 80. ? 1700090930u : (p.x < 88. ? 4u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 304152576u : (p.x < 120. ? 1118224u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 1u : (p.x < 152. ? 286261248u : (p.x < 160. ? 286265890u : (p.x < 168. ? 286265617u : (p.x < 176. ? 16921104u : (p.x < 184. ? 16781601u : (p.x < 192. ? 285282304u : 571544097u)))))))))))))))))))))))) : v;
	v = p.y == 86. ? (p.x < 8. ? 65536u : (p.x < 16. ? 285217041u : (p.x < 24. ? 571544081u : (p.x < 32. ? 286265873u : (p.x < 40. ? 286331136u : (p.x < 48. ? 268435473u : (p.x < 56. ? 17u : (p.x < 64. ? 1048576u : (p.x < 72. ? 536870913u : (p.x < 80. ? 19158338u : (p.x < 88. ? 0u : (p.x < 96. ? 4384u : (p.x < 104. ? 0u : (p.x < 112. ? 17960960u : (p.x < 120. ? 65536u : (p.x < 128. ? 0u : (p.x < 136. ? 16777216u : (p.x < 144. ? 0u : (p.x < 152. ? 286261248u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1118481u : (p.x < 176. ? 572666368u : (p.x < 184. ? 19014450u : (p.x < 192. ? 572592145u : 274u)))))))))))))))))))))))) : v;
	v = p.y == 85. ? (p.x < 8. ? 0u : (p.x < 16. ? 285212672u : (p.x < 24. ? 555880993u : (p.x < 32. ? 288568098u : (p.x < 40. ? 286331153u : (p.x < 48. ? 4113u : (p.x < 56. ? 0u : (p.x < 64. ? 554762240u : (p.x < 72. ? 0u : (p.x < 80. ? 283970u : (p.x < 88. ? 0u : (p.x < 96. ? 32u : (p.x < 104. ? 0u : (p.x < 112. ? 17899520u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 65536u : (p.x < 144. ? 0u : (p.x < 152. ? 286261248u : (p.x < 160. ? 286331153u : (p.x < 168. ? 4369u : (p.x < 176. ? 572592128u : (p.x < 184. ? 305275442u : (p.x < 192. ? 304222480u : 65537u)))))))))))))))))))))))) : v;
	v = p.y == 84. ? (p.x < 8. ? 0u : (p.x < 16. ? 318767104u : (p.x < 24. ? 842211888u : (p.x < 32. ? 53682738u : (p.x < 40. ? 286261280u : (p.x < 48. ? 69905u : (p.x < 56. ? 0u : (p.x < 64. ? 272u : (p.x < 72. ? 0u : (p.x < 80. ? 17456u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 268435456u : (p.x < 112. ? 74275u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 65536u : (p.x < 144. ? 0u : (p.x < 152. ? 285212672u : (p.x < 160. ? 286331153u : (p.x < 168. ? 285217041u : (p.x < 176. ? 4369u : (p.x < 184. ? 572732466u : (p.x < 192. ? 269558290u : 74256u)))))))))))))))))))))))) : v;
	v = p.y == 83. ? (p.x < 8. ? 0u : (p.x < 16. ? 34676736u : (p.x < 24. ? 590413857u : (p.x < 32. ? 304230947u : (p.x < 40. ? 285282577u : (p.x < 48. ? 1118481u : (p.x < 56. ? 0u : (p.x < 64. ? 4096u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 268435456u : (p.x < 112. ? 4371u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 65536u : (p.x < 144. ? 0u : (p.x < 152. ? 285212672u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 269553937u : (p.x < 184. ? 286331426u : (p.x < 192. ? 18u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 82. ? (p.x < 8. ? 0u : (p.x < 16. ? 1048576u : (p.x < 24. ? 0u : (p.x < 32. ? 589505056u : (p.x < 40. ? 285217041u : (p.x < 48. ? 69905u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 2u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 268435456u : (p.x < 112. ? 1u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 65552u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286401041u : (p.x < 176. ? 286331426u : (p.x < 184. ? 34u : (p.x < 192. ? 2097152u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 81. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 842211584u : (p.x < 40. ? 286401058u : (p.x < 48. ? 1118481u : (p.x < 56. ? 0u : (p.x < 64. ? 286331136u : (p.x < 72. ? 1u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 16u : (p.x < 112. ? 0u : (p.x < 120. ? 16777216u : (p.x < 128. ? 0u : (p.x < 136. ? 1114112u : (p.x < 144. ? 0u : (p.x < 152. ? 268435456u : (p.x < 160. ? 286331153u : (p.x < 168. ? 589509410u : (p.x < 176. ? 287449634u : (p.x < 184. ? 2u : (p.x < 192. ? 1179648u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 80. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 572657664u : (p.x < 40. ? 571548195u : (p.x < 48. ? 69905u : (p.x < 56. ? 0u : (p.x < 64. ? 286331136u : (p.x < 72. ? 17u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 256u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 1114112u : (p.x < 144. ? 0u : (p.x < 152. ? 555810816u : (p.x < 160. ? 554766882u : (p.x < 168. ? 572666418u : (p.x < 176. ? 18944274u : (p.x < 184. ? 0u : (p.x < 192. ? 2170880u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 79. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 589365504u : (p.x < 40. ? 287449907u : (p.x < 48. ? 285282577u : (p.x < 56. ? 17u : (p.x < 64. ? 286331152u : (p.x < 72. ? 4113u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 1u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 1118464u : (p.x < 144. ? 272u : (p.x < 152. ? 287379456u : (p.x < 160. ? 303121202u : (p.x < 168. ? 304226866u : (p.x < 176. ? 303108369u : (p.x < 184. ? 1u : (p.x < 192. ? 4096u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 78. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 1126170624u : (p.x < 40. ? 572736306u : (p.x < 48. ? 286265873u : (p.x < 56. ? 4369u : (p.x < 64. ? 286331152u : (p.x < 72. ? 273u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 268435473u : (p.x < 120. ? 1u : (p.x < 128. ? 1u : (p.x < 136. ? 17895424u : (p.x < 144. ? 1118481u : (p.x < 152. ? 843264528u : (p.x < 160. ? 590627634u : (p.x < 168. ? 555885090u : (p.x < 176. ? 319881234u : (p.x < 184. ? 17u : 0u))))))))))))))))))))))) : v;
	v = p.y == 77. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 285212672u : (p.x < 40. ? 572732194u : (p.x < 48. ? 286331170u : (p.x < 56. ? 268505361u : (p.x < 64. ? 1118481u : (p.x < 72. ? 65536u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 268435456u : (p.x < 112. ? 303108368u : (p.x < 120. ? 4369u : (p.x < 128. ? 0u : (p.x < 136. ? 69632u : (p.x < 144. ? 304156944u : (p.x < 152. ? 861155601u : (p.x < 160. ? 573781043u : (p.x < 168. ? 571548211u : (p.x < 176. ? 18874641u : (p.x < 184. ? 17u : 0u))))))))))))))))))))))) : v;
	v = p.y == 76. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 805306368u : (p.x < 40. ? 573710882u : (p.x < 48. ? 286331426u : (p.x < 56. ? 286261265u : (p.x < 64. ? 17961233u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 285212672u : (p.x < 112. ? 16917026u : (p.x < 120. ? 288u : (p.x < 128. ? 0u : (p.x < 136. ? 65536u : (p.x < 144. ? 555880721u : (p.x < 152. ? 1160843553u : (p.x < 160. ? 841176387u : (p.x < 168. ? 571613747u : (p.x < 176. ? 268505344u : (p.x < 184. ? 2u : 0u))))))))))))))))))))))) : v;
	v = p.y == 75. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 268435456u : (p.x < 40. ? 591667970u : (p.x < 48. ? 286335522u : (p.x < 56. ? 4369u : (p.x < 64. ? 1114385u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 555810816u : (p.x < 112. ? 66083u : (p.x < 120. ? 529u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 286331136u : (p.x < 152. ? 857805585u : (p.x < 160. ? 859059012u : (p.x < 168. ? 304226851u : (p.x < 176. ? 537919488u : 0u)))))))))))))))))))))) : v;
	v = p.y == 74. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 1161048883u : (p.x < 48. ? 286335797u : (p.x < 56. ? 268501009u : (p.x < 64. ? 69888u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 537919488u : (p.x < 112. ? 304087313u : (p.x < 120. ? 0u : (p.x < 128. ? 8704u : (p.x < 136. ? 0u : (p.x < 144. ? 824250640u : (p.x < 152. ? 839984705u : (p.x < 160. ? 858927921u : (p.x < 168. ? 20062754u : (p.x < 176. ? 16847104u : (p.x < 184. ? 4096u : 0u))))))))))))))))))))))) : v;
	v = p.y == 73. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 536870912u : (p.x < 40. ? 1160983347u : (p.x < 48. ? 286401332u : (p.x < 56. ? 1114385u : (p.x < 64. ? 4369u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 268435456u : (p.x < 104. ? 1249826u : (p.x < 112. ? 553660416u : (p.x < 120. ? 18u : (p.x < 128. ? 38993920u : (p.x < 136. ? 268435456u : (p.x < 144. ? 1400267025u : (p.x < 152. ? 304309880u : (p.x < 160. ? 573711154u : (p.x < 168. ? 288568098u : (p.x < 176. ? 8720u : (p.x < 184. ? 4096u : 0u))))))))))))))))))))))) : v;
	v = p.y == 72. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 536870912u : (p.x < 40. ? 1160983347u : (p.x < 48. ? 286401348u : (p.x < 56. ? 286326785u : (p.x < 64. ? 17u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 268435456u : (p.x < 104. ? 12834u : (p.x < 112. ? 805371904u : (p.x < 120. ? 590413824u : (p.x < 128. ? 3425330u : (p.x < 136. ? 0u : (p.x < 144. ? 1198657809u : (p.x < 152. ? 572662306u : (p.x < 160. ? 590488370u : (p.x < 168. ? 17965858u : (p.x < 176. ? 4881u : 0u)))))))))))))))))))))) : v;
	v = p.y == 71. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 268435456u : (p.x < 40. ? 1128547136u : (p.x < 48. ? 286401366u : (p.x < 56. ? 537919488u : (p.x < 64. ? 1u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 8737u : (p.x < 112. ? 0u : (p.x < 120. ? 841158658u : (p.x < 128. ? 36979492u : (p.x < 136. ? 4096u : (p.x < 144. ? 596145249u : (p.x < 152. ? 1735533090u : (p.x < 160. ? 859072630u : (p.x < 168. ? 4899u : (p.x < 176. ? 256u : 0u)))))))))))))))))))))) : v;
	v = p.y == 70. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 858927872u : (p.x < 48. ? 286335844u : (p.x < 56. ? 303104001u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 268435984u : (p.x < 112. ? 0u : (p.x < 120. ? 34799617u : (p.x < 128. ? 590483730u : (p.x < 136. ? 286466560u : (p.x < 144. ? 2022203665u : (p.x < 152. ? 1753781299u : (p.x < 160. ? 878012245u : (p.x < 168. ? 803u : (p.x < 176. ? 4096u : (p.x < 184. ? 49u : 0u))))))))))))))))))))))) : v;
	v = p.y == 69. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 1127354912u : (p.x < 48. ? 17900083u : (p.x < 56. ? 2166784u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 841093136u : (p.x < 112. ? 2u : (p.x < 120. ? 16777216u : (p.x < 128. ? 1142947857u : (p.x < 136. ? 842146338u : (p.x < 144. ? 2558874726u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 862423176u : (p.x < 168. ? 275u : (p.x < 176. ? 285212672u : (p.x < 184. ? 1u : 0u))))))))))))))))))))))) : v;
	v = p.y == 68. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 839974912u : (p.x < 48. ? 1123139u : (p.x < 56. ? 65536u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 19079986u : (p.x < 112. ? 16u : (p.x < 120. ? 0u : (p.x < 128. ? 1107300642u : (p.x < 136. ? 1110647588u : (p.x < 144. ? 2571183460u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 595036297u : (p.x < 168. ? 34u : 0u))))))))))))))))))))) : v;
	v = p.y == 67. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 856694784u : (p.x < 48. ? 1122866u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 268435456u : (p.x < 104. ? 17900069u : (p.x < 112. ? 4368u : (p.x < 120. ? 268435456u : (p.x < 128. ? 268440097u : (p.x < 136. ? 555823925u : (p.x < 144. ? 2164261698u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 292063368u : (p.x < 168. ? 51u : 0u))))))))))))))))))))) : v;
	v = p.y == 66. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 838926336u : (p.x < 48. ? 70196u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 536870912u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1118481u : (p.x < 120. ? 286261248u : (p.x < 128. ? 1118738u : (p.x < 136. ? 825316146u : (p.x < 144. ? 268435986u : (p.x < 152. ? 2560203091u : (p.x < 160. ? 286820488u : (p.x < 168. ? 17825826u : 0u))))))))))))))))))))) : v;
	v = p.y == 65. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 536870912u : (p.x < 48. ? 9012u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 286261248u : (p.x < 104. ? 286331153u : (p.x < 112. ? 17895697u : (p.x < 120. ? 1048576u : (p.x < 128. ? 17900066u : (p.x < 136. ? 321991200u : (p.x < 144. ? 17891347u : (p.x < 152. ? 2560115200u : (p.x < 160. ? 573994274u : (p.x < 168. ? 35651602u : 0u))))))))))))))))))))) : v;
	v = p.y == 64. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 78404u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 286261248u : (p.x < 104. ? 571543825u : (p.x < 112. ? 286331426u : (p.x < 120. ? 17895696u : (p.x < 128. ? 18948640u : (p.x < 136. ? 554762240u : (p.x < 144. ? 17829905u : (p.x < 152. ? 0u : (p.x < 160. ? 591614738u : (p.x < 168. ? 1118498u : 0u))))))))))))))))))))) : v;
	v = p.y == 63. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 279632u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 286326784u : (p.x < 104. ? 571543825u : (p.x < 112. ? 286331442u : (p.x < 120. ? 269554193u : (p.x < 128. ? 19014160u : (p.x < 136. ? 0u : (p.x < 144. ? 286326784u : (p.x < 152. ? 4113u : (p.x < 160. ? 573780752u : (p.x < 168. ? 805376273u : 0u))))))))))))))))))))) : v;
	v = p.y == 62. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 148496u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 286330880u : (p.x < 104. ? 571543553u : (p.x < 112. ? 572662306u : (p.x < 120. ? 553652753u : (p.x < 128. ? 19014144u : (p.x < 136. ? 65536u : (p.x < 144. ? 286261248u : (p.x < 152. ? 268439825u : (p.x < 160. ? 20066833u : (p.x < 168. ? 16u : 0u))))))))))))))))))))) : v;
	v = p.y == 61. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 1327872u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 269484032u : (p.x < 104. ? 286331152u : (p.x < 112. ? 321982737u : (p.x < 120. ? 554766625u : (p.x < 128. ? 287449601u : (p.x < 136. ? 0u : (p.x < 144. ? 286261248u : (p.x < 152. ? 17u : (p.x < 160. ? 19014160u : 0u)))))))))))))))))))) : v;
	v = p.y == 60. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 4464896u : (p.x < 56. ? 0u : (p.x < 64. ? 196608u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 286261248u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331154u : (p.x < 120. ? 286331169u : (p.x < 128. ? 287522818u : (p.x < 136. ? 17u : (p.x < 144. ? 287309824u : (p.x < 152. ? 272u : (p.x < 160. ? 2167297u : (p.x < 168. ? 1u : 0u))))))))))))))))))))) : v;
	v = p.y == 59. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 338755584u : (p.x < 56. ? 3u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 17825792u : (p.x < 104. ? 286327057u : (p.x < 112. ? 268505361u : (p.x < 120. ? 286331170u : (p.x < 128. ? 572784657u : (p.x < 136. ? 290u : (p.x < 144. ? 287309824u : (p.x < 152. ? 0u : (p.x < 160. ? 65792u : (p.x < 168. ? 268435456u : 0u))))))))))))))))))))) : v;
	v = p.y == 58. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 139569u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 285212672u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331426u : (p.x < 128. ? 858062914u : (p.x < 136. ? 1u : (p.x < 144. ? 17825792u : (p.x < 152. ? 0u : (p.x < 160. ? 536936704u : (p.x < 168. ? 268435456u : 0u))))))))))))))))))))) : v;
	v = p.y == 57. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 65792u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 268435456u : (p.x < 104. ? 285282577u : (p.x < 112. ? 286265617u : (p.x < 120. ? 286331442u : (p.x < 128. ? 1245234u : (p.x < 136. ? 0u : (p.x < 144. ? 570425344u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 1u : 0u))))))))))))))))))))) : v;
	v = p.y == 56. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 286261248u : (p.x < 104. ? 269488401u : (p.x < 112. ? 286331153u : (p.x < 120. ? 287445282u : (p.x < 128. ? 4675u : (p.x < 136. ? 0u : (p.x < 144. ? 301989888u : (p.x < 152. ? 0u : (p.x < 160. ? 1048576u : (p.x < 168. ? 1u : 0u))))))))))))))))))))) : v;
	v = p.y == 55. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 3145728u : (p.x < 64. ? 287322112u : (p.x < 72. ? 2u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 286261248u : (p.x < 104. ? 269484305u : (p.x < 112. ? 286331154u : (p.x < 120. ? 823202065u : (p.x < 128. ? 572662340u : 0u)))))))))))))))) : v;
	v = p.y == 54. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 143360u : (p.x < 72. ? 16u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 285212672u : (p.x < 104. ? 285278481u : (p.x < 112. ? 286331392u : (p.x < 120. ? 823202082u : (p.x < 128. ? 304227123u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 33u : 0u)))))))))))))))))))))) : v;
	v = p.y == 53. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 536887568u : (p.x < 72. ? 273u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 268435456u : (p.x < 104. ? 0u : (p.x < 112. ? 286401056u : (p.x < 120. ? 554766609u : (p.x < 128. ? 1118803u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 16u : 0u)))))))))))))))))))))) : v;
	v = p.y == 52. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 838861088u : (p.x < 72. ? 290u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 286335232u : (p.x < 120. ? 303173905u : (p.x < 128. ? 1184306u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 65552u : (p.x < 168. ? 1179648u : 0u))))))))))))))))))))) : v;
	v = p.y == 51. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 4928u : (p.x < 72. ? 17891346u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 286331136u : (p.x < 120. ? 572662033u : (p.x < 128. ? 4370u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 256u : (p.x < 168. ? 65536u : 0u))))))))))))))))))))) : v;
	v = p.y == 50. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 536875026u : (p.x < 72. ? 1118208u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 286331136u : (p.x < 120. ? 1110581521u : (p.x < 128. ? 19u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 4096u : (p.x < 168. ? 268574720u : 0u))))))))))))))))))))) : v;
	v = p.y == 49. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 23u : (p.x < 72. ? 17825792u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 286331136u : (p.x < 120. ? 1110647057u : (p.x < 128. ? 20u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 16777472u : (p.x < 176. ? 1048576u : 0u)))))))))))))))))))))) : v;
	v = p.y == 48. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 3u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 286331136u : (p.x < 120. ? 842211601u : (p.x < 128. ? 19u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 131072u : (p.x < 168. ? 318767104u : (p.x < 176. ? 16777216u : (p.x < 184. ? 17u : 0u))))))))))))))))))))))) : v;
	v = p.y == 47. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 268435456u : (p.x < 64. ? 1u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 286330880u : (p.x < 120. ? 857874705u : (p.x < 128. ? 19u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 2097152u : (p.x < 168. ? 16777216u : (p.x < 176. ? 0u : (p.x < 184. ? 322u : (p.x < 192. ? 1u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 46. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 20u : (p.x < 72. ? 285212672u : (p.x < 80. ? 286331153u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 304222208u : (p.x < 120. ? 590484001u : (p.x < 128. ? 3u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 0u : (p.x < 184. ? 3350528u : (p.x < 192. ? 256u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 45. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 23u : (p.x < 72. ? 286326784u : (p.x < 80. ? 554766592u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 572653568u : (p.x < 120. ? 860037666u : (p.x < 128. ? 1u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 1052944u : (p.x < 176. ? 0u : (p.x < 184. ? 3145728u : 0u))))))))))))))))))))))) : v;
	v = p.y == 44. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 70000u : (p.x < 72. ? 286261248u : (p.x < 80. ? 17895697u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 572653568u : (p.x < 120. ? 590557986u : (p.x < 128. ? 17u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 0u : (p.x < 184. ? 16777216u : 0u))))))))))))))))))))))) : v;
	v = p.y == 43. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 1120384u : (p.x < 72. ? 286331152u : (p.x < 80. ? 17961489u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 590544896u : (p.x < 120. ? 573711154u : (p.x < 128. ? 17u : 0u)))))))))))))))) : v;
	v = p.y == 42. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 1538048u : (p.x < 72. ? 286331136u : (p.x < 80. ? 2232865u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 858988544u : (p.x < 120. ? 571683618u : (p.x < 128. ? 268435474u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 1048576u : (p.x < 184. ? 4096u : 0u))))))))))))))))))))))) : v;
	v = p.y == 41. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 311001600u : (p.x < 72. ? 287379472u : (p.x < 80. ? 1184289u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 590491648u : (p.x < 120. ? 287449634u : (p.x < 128. ? 268435474u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 256u : 0u)))))))))))))))))))))) : v;
	v = p.y == 40. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 948109312u : (p.x < 72. ? 554696977u : (p.x < 80. ? 1122594u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 572661760u : (p.x < 120. ? 19010338u : (p.x < 128. ? 34603008u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 17891600u : (p.x < 184. ? 65536u : 0u))))))))))))))))))))))) : v;
	v = p.y == 39. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 2004877312u : (p.x < 72. ? 286261523u : (p.x < 80. ? 74273u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 572661760u : (p.x < 120. ? 20062754u : (p.x < 128. ? 52428800u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 286331136u : (p.x < 184. ? 69633u : 0u))))))))))))))))))))))) : v;
	v = p.y == 38. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 1731198976u : (p.x < 72. ? 286326802u : (p.x < 80. ? 139809u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 590479360u : (p.x < 120. ? 17969698u : (p.x < 128. ? 35651584u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 268435456u : (p.x < 176. ? 286331152u : (p.x < 184. ? 17895440u : (p.x < 192. ? 268435456u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 37. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 2033188864u : (p.x < 72. ? 286261249u : (p.x < 80. ? 70177u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 590479360u : (p.x < 120. ? 1122850u : (p.x < 128. ? 2228224u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 287309824u : (p.x < 176. ? 286396689u : (p.x < 184. ? 16842769u : 0u))))))))))))))))))))))) : v;
	v = p.y == 36. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 2000683008u : (p.x < 72. ? 286261249u : (p.x < 80. ? 2u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 573636608u : (p.x < 120. ? 1192482u : (p.x < 128. ? 1114112u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 286326784u : (p.x < 176. ? 286335249u : (p.x < 184. ? 269549568u : 0u))))))))))))))))))))))) : v;
	v = p.y == 35. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 1197473792u : (p.x < 72. ? 571473920u : (p.x < 80. ? 2u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 573636608u : (p.x < 120. ? 1258274u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 286326784u : (p.x < 176. ? 17965329u : (p.x < 184. ? 286326784u : (p.x < 192. ? 1u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 34. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 595591168u : (p.x < 72. ? 553648128u : (p.x < 80. ? 3u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 572522496u : (p.x < 120. ? 2306867u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 286326784u : (p.x < 176. ? 17895697u : (p.x < 184. ? 268435456u : (p.x < 192. ? 1u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 33. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 326172672u : (p.x < 72. ? 1u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 570425344u : (p.x < 120. ? 213794u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 286326784u : (p.x < 176. ? 4369u : (p.x < 184. ? 268435472u : (p.x < 192. ? 2u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 32. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 21102592u : (p.x < 72. ? 16777217u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 553648128u : (p.x < 120. ? 13107u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 286261248u : (p.x < 176. ? 1u : (p.x < 184. ? 286261248u : (p.x < 192. ? 1u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 31. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 557973504u : (p.x < 72. ? 1u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 536870912u : (p.x < 120. ? 274u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 17825792u : (p.x < 176. ? 1u : (p.x < 184. ? 554696704u : 0u))))))))))))))))))))))) : v;
	v = p.y == 30. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 288423936u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 0u : (p.x < 184. ? 285212672u : 0u))))))))))))))))))))))) : v;
	v = p.y == 29. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 288555008u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 0u : (p.x < 184. ? 35655680u : 0u))))))))))))))))))))))) : v;
	v = p.y == 28. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 18022400u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 0u : (p.x < 184. ? 0u : (p.x < 192. ? 0u : 34603008u)))))))))))))))))))))))) : v;
	v = p.y == 27. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 19070976u : 0u)))))))) : v;
	v = p.y == 26. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 36831232u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 0u : (p.x < 184. ? 1048576u : (p.x < 192. ? 0u : 69632u)))))))))))))))))))))))) : v;
	v = p.y == 25. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 287444992u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 0u : (p.x < 184. ? 0u : (p.x < 192. ? 0u : 512u)))))))))))))))))))))))) : v;
	v = p.y == 24. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 1187840u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 0u : (p.x < 184. ? 0u : (p.x < 192. ? 0u : 256u)))))))))))))))))))))))) : v;
	v = p.y == 23. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 2236416u : 0u)))))))) : v;
	v = p.y == 22. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 69888u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 256u : 0u)))))))))))))))))) : v;
	v = p.y == 21. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 69632u : 0u)))))))) : v;
	v = p.y == 20. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 4096u : 0u)))))))) : v;
	v = p.y == 19. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 17825792u : 0u)))))))) : v;
	v = p.y == 18. ? 0u : v;
	v = p.y == 17. ? 0u : v;
	v = p.y == 16. ? 0u : v;
	v = p.y == 15. ? 0u : v;
	v = p.y == 14. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 4352u : 0u))))))))) : v;
	v = p.y == 13. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 0u : (p.x < 72. ? 2u : 0u))))))))) : v;
	v = p.y == 12. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 285212672u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 0u : (p.x < 128. ? 268435456u : (p.x < 136. ? 1122866u : (p.x < 144. ? 0u : (p.x < 152. ? 857870336u : (p.x < 160. ? 858993459u : (p.x < 168. ? 554766883u : (p.x < 176. ? 858989106u : (p.x < 184. ? 9011u : 0u))))))))))))))))))))))) : v;
	v = p.y == 11. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 805437440u : (p.x < 72. ? 2u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 0u : (p.x < 112. ? 0u : (p.x < 120. ? 16777216u : (p.x < 128. ? 876884480u : (p.x < 136. ? 1145328964u : (p.x < 144. ? 536870947u : (p.x < 152. ? 1413760067u : (p.x < 160. ? 1146373461u : (p.x < 168. ? 1145324612u : (p.x < 176. ? 1163150404u : (p.x < 184. ? 590558276u : (p.x < 192. ? 546u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 10. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 807403520u : (p.x < 72. ? 36u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 0u : (p.x < 104. ? 554696704u : (p.x < 112. ? 287449634u : (p.x < 120. ? 857870353u : (p.x < 128. ? 1431585588u : (p.x < 136. ? 1146447461u : (p.x < 144. ? 1144127507u : (p.x < 152. ? 1717917013u : (p.x < 160. ? 1431660134u : (p.x < 168. ? 1431655765u : (p.x < 176. ? 1431725669u : (p.x < 184. ? 1145324629u : (p.x < 192. ? 588530739u : 3u)))))))))))))))))))))))) : v;
	v = p.y == 9. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 0u : (p.x < 48. ? 0u : (p.x < 56. ? 0u : (p.x < 64. ? 805306368u : (p.x < 72. ? 35u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 839909376u : (p.x < 104. ? 1716794180u : (p.x < 112. ? 1432774246u : (p.x < 120. ? 1431590501u : (p.x < 128. ? 1717986917u : (p.x < 136. ? 860182118u : (p.x < 144. ? 1412641281u : (p.x < 152. ? 2004313701u : (p.x < 160. ? 1718056823u : (p.x < 168. ? 1717986918u : (p.x < 176. ? 1449551462u : (p.x < 184. ? 1145328981u : (p.x < 192. ? 1128547396u : 35u)))))))))))))))))))))))) : v;
	v = p.y == 8. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 0u : (p.x < 32. ? 0u : (p.x < 40. ? 16847120u : (p.x < 48. ? 572522496u : (p.x < 56. ? 573780770u : (p.x < 64. ? 875770402u : (p.x < 72. ? 2u : (p.x < 80. ? 0u : (p.x < 88. ? 0u : (p.x < 96. ? 1412571392u : (p.x < 104. ? 1716938069u : (p.x < 112. ? 2004318070u : (p.x < 120. ? 2004318071u : (p.x < 128. ? 2004318071u : (p.x < 136. ? 876893798u : (p.x < 144. ? 1430537011u : (p.x < 152. ? 2004318054u : (p.x < 160. ? 1735882615u : (p.x < 168. ? 1717986918u : (p.x < 176. ? 1449551462u : (p.x < 184. ? 1431655765u : (p.x < 192. ? 20132932u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 7. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 858923008u : (p.x < 32. ? 1145324612u : (p.x < 40. ? 573780804u : (p.x < 48. ? 572662306u : (p.x < 56. ? 572662579u : (p.x < 64. ? 69632u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 857800704u : (p.x < 96. ? 1145320243u : (p.x < 104. ? 1717917013u : (p.x < 112. ? 2004318054u : (p.x < 120. ? 2004318071u : (p.x < 128. ? 2004318071u : (p.x < 136. ? 1431725671u : (p.x < 144. ? 1700091221u : (p.x < 152. ? 2004318054u : (p.x < 160. ? 2004318071u : (p.x < 168. ? 1717987191u : (p.x < 176. ? 1432774246u : (p.x < 184. ? 1431655765u : (p.x < 192. ? 2376772u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 6. ? (p.x < 8. ? 0u : (p.x < 16. ? 553648128u : (p.x < 24. ? 841097762u : (p.x < 32. ? 858993459u : (p.x < 40. ? 858993459u : (p.x < 48. ? 858993459u : (p.x < 56. ? 285496115u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 0u : (p.x < 88. ? 571613729u : (p.x < 96. ? 1145254707u : (p.x < 104. ? 1431655492u : (p.x < 112. ? 2003199589u : (p.x < 120. ? 2004318071u : (p.x < 128. ? 2004318071u : (p.x < 136. ? 2004318071u : (p.x < 144. ? 2004318071u : (p.x < 152. ? 2004318071u : (p.x < 160. ? 2004318071u : (p.x < 168. ? 1717991287u : (p.x < 176. ? 1431725670u : (p.x < 184. ? 1146443093u : (p.x < 192. ? 338904132u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 5. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 286326784u : (p.x < 32. ? 858989089u : (p.x < 40. ? 1127428915u : (p.x < 48. ? 1145324612u : (p.x < 56. ? 305345604u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 4369u : (p.x < 88. ? 554762240u : (p.x < 96. ? 1144206114u : (p.x < 104. ? 1431585860u : (p.x < 112. ? 1717986645u : (p.x < 120. ? 2004318054u : (p.x < 128. ? 2004318071u : (p.x < 136. ? 2004318071u : (p.x < 144. ? 2290644855u : (p.x < 152. ? 2004322440u : (p.x < 160. ? 2004318071u : (p.x < 168. ? 1717987191u : (p.x < 176. ? 1431660134u : (p.x < 184. ? 1145324613u : (p.x < 192. ? 78643u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 4. ? (p.x < 8. ? 0u : (p.x < 16. ? 268435456u : (p.x < 24. ? 286331153u : (p.x < 32. ? 572662033u : (p.x < 40. ? 858993186u : (p.x < 48. ? 1145324612u : (p.x < 56. ? 573780788u : (p.x < 64. ? 1118481u : (p.x < 72. ? 0u : (p.x < 80. ? 571473920u : (p.x < 88. ? 858993442u : (p.x < 96. ? 1145320243u : (p.x < 104. ? 1145324612u : (p.x < 112. ? 1431655765u : (p.x < 120. ? 1717986901u : (p.x < 128. ? 2004318071u : (p.x < 136. ? 2272753527u : (p.x < 144. ? 2290649224u : (p.x < 152. ? 2004318088u : (p.x < 160. ? 2004318071u : (p.x < 168. ? 1717986919u : (p.x < 176. ? 1431655766u : (p.x < 184. ? 1145324613u : (p.x < 192. ? 19088452u : 0u)))))))))))))))))))))))) : v;
	v = p.y == 3. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 286327056u : (p.x < 32. ? 572657937u : (p.x < 40. ? 858993442u : (p.x < 48. ? 1145324611u : (p.x < 56. ? 858993460u : (p.x < 64. ? 304226851u : (p.x < 72. ? 858918913u : (p.x < 80. ? 858989090u : (p.x < 88. ? 1127428915u : (p.x < 96. ? 1145324612u : (p.x < 104. ? 1431651396u : (p.x < 112. ? 1431655765u : (p.x < 120. ? 1717982549u : (p.x < 128. ? 2004318054u : (p.x < 136. ? 2004318071u : (p.x < 144. ? 2004318071u : (p.x < 152. ? 2004318071u : (p.x < 160. ? 1719105399u : (p.x < 168. ? 1717986918u : (p.x < 176. ? 1431655782u : (p.x < 184. ? 1431655765u : (p.x < 192. ? 877937732u : 4675u)))))))))))))))))))))))) : v;
	v = p.y == 2. ? (p.x < 8. ? 1179997253u : (p.x < 16. ? 1118499u : (p.x < 24. ? 571543825u : (p.x < 32. ? 1144205858u : (p.x < 40. ? 1145324612u : (p.x < 48. ? 1145324612u : (p.x < 56. ? 858993476u : (p.x < 64. ? 858993459u : (p.x < 72. ? 858993459u : (p.x < 80. ? 1144206131u : (p.x < 88. ? 1145324612u : (p.x < 96. ? 1431655492u : (p.x < 104. ? 1431655765u : (p.x < 112. ? 1431655765u : (p.x < 120. ? 1717982549u : (p.x < 128. ? 1986422374u : (p.x < 136. ? 2004318071u : (p.x < 144. ? 2004318071u : (p.x < 152. ? 2004318071u : (p.x < 160. ? 1717986918u : (p.x < 168. ? 1717986918u : (p.x < 176. ? 1717986918u : (p.x < 184. ? 1431655766u : (p.x < 192. ? 1414878549u : 1431651397u)))))))))))))))))))))))) : v;
	v = p.y == 1. ? (p.x < 8. ? 1432774246u : (p.x < 16. ? 1431655765u : (p.x < 24. ? 1431655765u : (p.x < 32. ? 1431655765u : (p.x < 40. ? 1431655765u : (p.x < 48. ? 1146443093u : (p.x < 56. ? 1145324612u : (p.x < 64. ? 1145324868u : (p.x < 72. ? 1145324612u : (p.x < 80. ? 1145324612u : (p.x < 88. ? 1162101828u : (p.x < 96. ? 1431655765u : (p.x < 104. ? 1431655765u : (p.x < 112. ? 1431655765u : (p.x < 120. ? 1700091221u : 1717986918u))))))))))))))) : v;
	v = p.y == 0. ? (p.x < 8. ? 1717986918u : (p.x < 16. ? 1717986918u : (p.x < 24. ? 1717986918u : (p.x < 32. ? 1431656038u : (p.x < 40. ? 1431721301u : (p.x < 48. ? 1431655765u : (p.x < 56. ? 1431655765u : (p.x < 64. ? 1431655765u : (p.x < 72. ? 1431655765u : (p.x < 80. ? 1431655765u : (p.x < 88. ? 1431655765u : (p.x < 96. ? 1431655765u : (p.x < 104. ? 1431655765u : (p.x < 112. ? 1431655765u : (p.x < 120. ? 1431655765u : (p.x < 128. ? 1717986917u : 1717986918u)))))))))))))))) : v;
    v = p.x >= 0. && p.x < 200. ? v : 0u;

    float i = float((v >> uint(4. * p.x)) & 15u);
    vec3 color = vec3(0.0039);
    color = i == 1. ? vec3(0.067) : color;
    color = i == 2. ? vec3(0.13) : color;
    color = i == 3. ? vec3(0.23) : color;
    color = i == 4. ? vec3(0.33) : color;
    color = i == 5. ? vec3(0.4) : color;
    color = i == 6. ? vec3(0.44) : color;
    color = i == 7. ? vec3(0.51) : color;
    color = i == 8. ? vec3(0.6) : color;
    color = i == 9. ? vec3(0.69) : color;
    
    return vec4(color, 1.0);
}

// Function 670
vec2 uvLerp(vec2 a, vec2 b, vec2 c, vec3 x) {
    return a * x.x + b * x.y + c * x.z;
}

// Function 671
vec3 Tonemap(const TonemapParams tc, vec3 x)
{
	vec3 toe = - tc.mToe.x / (x + tc.mToe.y) + tc.mToe.z;
	vec3 mid = tc.mMid.x * x + tc.mMid.y;
	vec3 shoulder = - tc.mShoulder.x / (x + tc.mShoulder.y) + tc.mShoulder.z;

	vec3 result = mix(toe, mid, step(tc.mBx.x, x));
	result = mix(result, shoulder, step(tc.mBx.y, x));
	return result;
}

// Function 672
vec4 cubemap( sampler2D sam, in vec3 d )
{
    // intersect cube
    vec3 n = abs(d);
    vec3 v = (n.x>n.y && n.x>n.z) ? d.xyz: 
             (n.y>n.x && n.y>n.z) ? d.yzx:
                                    d.zxy;
    // project into face
    vec2 q = v.yz/v.x;
    // undistort in the edges
    q *= 1.25 - 0.25*q*q;
    // sample
    return texture( sam, 0.5+0.5*q );
}

// Function 673
vec4 hpluvToLch(float x, float y, float z, float a) {return hpluvToLch( vec4(x,y,z,a) );}

// Function 674
float MapBridge(vec3 p)
{
  p=TranslateBridge(p);
  // AABB
  if (sdBox(p-vec3(10., -1.0, 0.0), vec3(11.5, 2.50, 2.25))>3.) return 10000.;

  vec3 bPos = p+vec3(0.36, 0.0, 0.0);
  // bottom planks
  pModInterval1(bPos.x, 0.35, 0., 60.);
  float d= sdBox(bPos-vec3(0., 0.0, 0.1), vec3(0.12, 0.08, 1.80));

  // bearing balks
  bPos = p-vec3(-1.75, -0.726, -2.);
  pModInterval1(bPos.x, 3.2, 0., 7.);
  d= min(d, sdBox(bPos-vec3(0., .0, 2.1), vec3(0.15, 0.15, 2.00)));
  float m = pModInterval1(bPos.z, 4.2, 0., 1.);
  d= min(d, sdCappedCylinder(bPos+vec3(0., 0.55, 0.), vec2(0.2, 2.8-m)));

  // side rails      
  bPos = p-vec3(10.8, 0., -1.7);
  m = pModInterval1(bPos.z, 3.60, 0., 1.);
   m = pModInterval1(bPos.y, 1.40, 0., 1.-m);
     
  d= min(d, sdBox(bPos, vec3(10., 0.14, .12)));

  return d;
}

// Function 675
float map(vec3 p){
    float disp=length(vec4(voronoiSphereMapping(normalize(p)),1.))*0.4-0.8;
	return length(p)-1.+disp;}

// Function 676
float MapExplosion( vec3 p, Explosion ex)
{ 
  checkPos = (ex.pos)-vec3(planePos.x, 0., planePos.z); 
  checkPos=p-checkPos;

  float testDist = fSphere(checkPos, 20.0);
  if (testDist>10.)  return testDist;

  float intensity =GetExplosionIntensity(ex);
  float d= fSphere(checkPos, intensity*15.);  

  // terrain clipping
  #ifdef EXACT_EXPLOSIONS
    d=max(d, -MapTerrain(p));
  #else
    d = max(d, -sdBox(checkPos+vec3(0., 50., 0.), vec3(50., 50.0, 50.0)));
  #endif

  // add explosion "noise/flames"
  float displace = fbm(((checkPos) + vec3(1, -2, -1)*iTime)*0.5);
  return d + (displace * 1.5*max(0., 4.*intensity));
}

// Function 677
vec3 map( vec3 p )
{
	float k = 1.0;
	float m = 1e10;
	for( int i=0; i<22; i++ ) 
	{
		m = min( m, dot(p,p)/(k*k) );
		p = (mm*vec4((abs(p)),1.0)).xyz;
		k*= s;
	}
	

	float d = (length(p)-0.25)/k;
	
	float h = p.z - 0.35*p.x;
	
	return vec3( d, m, h );
}

// Function 678
float remap(float value, float low1,float high1, float low2,float high2,bool c){float r=low2 + (value - low1) * (high2 - low2) / (high1 - low1);return c?clamp(r,min(low2,high2),max(low2,high2)):r;}

// Function 679
float remap(float val, float min, float max)
{
    return sat((val - min) / (max - min));
}

// Function 680
float mapSimp(vec3 p)
{
    float d = p.y;
    d -= sin(p.z*0.2 + 1.0 - cos(p.x*0.25))*0.35;
    float att = clamp(p.y*0.3 + 1.3, 0.,1.);  
    d += cyclic3DSimp(p*0.3)*att*1. + 1.;
    return d;
}

// Function 681
float map(vec3 p){
    
    // Floor.
    float fl = -p.z + .03;

    // The extruded blocks.
    vec4 d4 = blocks(p);
    gID = d4.yzw; // Individual block ID.
    
 
    // Overall object ID.
    objID = fl<d4.x? 1. : 0.;
    
    // Combining the floor with the extruded image
    return  min(fl, d4.x);
 
}

// Function 682
vec3 doBumpMap( sampler2D tx, in vec3 p, in vec3 n, float bf){
   
    const vec2 e = vec2(0.001, 0);
    
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    
    vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
	
}

// Function 683
float MapLights( vec3 p)
{
  vec3 pOriginal = p;
  // rotate position 
  p=TranslatePos(p, pitch, roll);   
  // mirror position at x=0.0. Both sides of the plane are equal.
  pMirror(p.x, 0.0);

  return max(sdEllipsoid( p- vec3(0.4, -0.1, -9.5), vec3(0.03, 0.03, 0.03+max(0., (speed*0.07)))), -sdBox(p- vec3(0.4, -0.1, -9.6+2.0), vec3(2.0, 2.0, 2.0)));
}

// Function 684
vec3 nmaps(vec3 x){ return x*2.-1.; }

// Function 685
float map(in vec3 ro, in vec3 rd){ 
    return min(asphere(ro,rd,vec3(0.0,0.0,0.0), 1.5),
               min(asphere(ro,rd,vec3(-2,0.0,0.0),1.0), 
                   min(asphere(ro,rd,vec3(0.0,-2,0.0),1.0),
                       min(asphere(ro,rd,vec3(1.15,1.15,1.15),1.0),
                           min(asphere(ro,rd,vec3(0.0,0.0,-2),1.0),
                              min(asphere(ro,rd,vec3(3.,3.,3.),0.2),
                                 asphere(ro, rd, vec3(-3.,-3.,-3.), 0.2)))))));
}

// Function 686
vec4 map(vec3 p, vec3 ro) {
    vec3 r = vec3(1.0, 2.0, 0.05);
    
    vec4 d = vec4(1e10);
    
    float dist = distance(p, ro);
    
    // Left wall.
    vec3 pp = p;
    pp.xz *= rot(3.141 / 2.0);
    pp.z -= 5.0;
    float id = opRep(pp.xy, r.xy * 2.0, vec2(-15.0, -2.0), vec2(10.0, 1.0));
    d = min4(d, sdConcretePanel(pp, r, id, dist));
    
    // Far wall.
    pp = p;
    id = opRep(pp.xy, r.xy * 2.0, vec2(-5.0, -2.0), vec2(5.0, 1.0));
    d = min4(d, sdConcretePanel(pp - vec3(0.0, 0.0, 33.0), r, id, dist));
    
    // Columns.
    pp = p - vec3(8.0, 0.0, 0.0);
    id = opRep(pp.z, 5.0, -4.0, 3.0);
    d = min4(d, sdSlab(pp, vec3(2.0, 8.0, 1.0), 1.0, dist));
    
    // Floor.
    pp = p;
    pp.yz *= rot(3.141 / 2.0);
    pp.z -= 5.0;
    r = vec3(3.0, 3.0, 0.1);
    id = opRep(pp.xy, r.xy * 2.0, vec2(-5.0, -5.0), vec2(6.0, 6.0));
    d = min4(d, sdSlab(pp, r, id, dist));
    
    // Ceiling.
    d = min4(d, sdSlab(p - vec3(1.0, 10.0, 0.0), vec3(10.0, 3.0, 50.0), 1.0, dist));
    
    return d;
}

// Function 687
float map_hor(vec3 pos)
{  
    float fy = 132.*random(1.254*floor(pos.y/fe));

    fy = 17.5*random(2.452*floor(pos.y/fe));
    pyd = fe*tdd*(1. - 0.45*0.5*sin(pos.x*2.15 + 13.*fy) - 0.3*0.5*sin(pos.x*4.12 + 42.*fy) - 0.25*0.5*sin(pos.y*8.72 + 70.*fy));
    pos.y+= pyd;

    hsf = 0.35*sin(pos.x*4.3 + 20.*fy) + 0.4*sin(pos.x*5.7 + 45.*fy) + 0.25*sin(pos.x*8.48 + 55.*fy);
    fr = fr0*(-tdv*0.5 + 1. - 0.5*tdv*hsf);
    
    float angle = twfs*pos.x;
    vec2 d1 = rotateVec(vec2(fr*fds, fr*fds), angle);
    vec2 d2 = d1.yx*vec2(1., -1);
    return min(min(min(map_hor_small(pos, d1, 1.), map_hor_small(pos, d2, 5.)), map_hor_small(pos, -d2, 9.)), map_hor_small(pos, -d1, 13.)); 
}

// Function 688
vec3 doBumpMap(in vec3 p, in vec3 n, float bumpfactor){
    
    vec2 e = vec2(2.5/iResolution.y, 0);
    float f = bumpFunction2(p); 
    
    float fx = bumpFunction2(p - e.xyy); // Same for the nearby sample in the X-direction.
    float fy = bumpFunction2(p - e.yxy); // Same for the nearby sample in the Y-direction.
    float fz = bumpFunction2(p - e.yyx); // Same for the nearby sample in the Y-direction.
    
    vec3 grad = (vec3(fx, fy, fz )-f)/e.x; 
          
    grad -= n*dot(n, grad);          
                      
    return normalize( n + grad*bumpfactor );
}

// Function 689
float map(vec3 p){
    
	// Surface function to perturb the walls.
    float sf = surfFunc(p);

    // A gyroid object to form the main passage base layer.
    float cav = dot(cos(p*3.14159265/8.), sin(p.yzx*3.14159265/8.)) + 2.;
    
    // Mold everything around the path.
    p.xy -= path(p.z);
    
    // The oval tunnel. Basically, a circle stretched along Y.
    float tun = 1.5 - length(p.xy*vec2(1, .4));
   
    // Smoothly combining the tunnel with the passage base layer,
    // then perturbing the walls.
    tun = smax(tun, 1.-cav, 2.) + .75 + (.5-sf);
    
    float gr = p.y + 7. - cav*.5 + (.5-sf)*.5; // The ground.
    float rf = p.y - 15.; // The roof cutoff point.
    
    // Smoothly combining the passage with the ground, and capping
    // it off at roof height.
    return smax(smin(tun, gr, .1), rf, 1.);
 
 
}

// Function 690
vec4 mapNumber( vec2 p, vec2 tp, float s, float v1){
    float vn = abs(v1), vf = fract(vn), count = .0, vc = vn;
    vec4 chr = vec4(0.);
    for (int i = 0; i < 10; i++) { 																			// count digits 
        count++; vc /= 10.;
        if (vc <= 1.0 && i != 0) break;
    }
    if (v1 < 0.) count++;
    p = (p - tp + vec2(.5/s ) + vec2(-count*.05, 0.)) * vec2(s);
    if (v1 < 0.){
        chr = max(chr, char( (p+vec2(float(count-1.)*.5,0.)), 0x2d).x); 						// minus/(-)
	}
    chr = max(chr, char((p-vec2(float(1)*.5,0.)), 0x2e).x); 									// dot/period (.)
    for (int i = 0; i < 10; i++) {
        if (vn >= 1.0 || i == 0)
            chr = max(chr, char( (p+vec2(float(i)*.5,0.)), 48+int(mod(abs(vn),10.))).x); 		// draw int
        vn /= 10.; vf *= 10.;
        if (i < 2)
                chr = max(chr, char( (p-vec2(float(i+2)*.5,0.)), 48+int(mod(abs(vf),10.))).x); // draw dec
    }
    return chr;
}

// Function 691
float heightmap(vec2 uv) {
	float height = texture(iChannel0, uv*0.15).x*0.04+0.01;
	return height - (wake(vec2(0.28,0.0)-uv)+wake(-uv)+0.1*wake(vec2(-0.3,0.0)-uv))*0.025;
}

// Function 692
bool map(vec3 p)
{
    float o = 2.0; 
    
    return map0(p + vec3( -o, 0.0, 0.0)) != 
           map0(p + vec3(  o, 0.0, 0.0)) != 
           map0(p + vec3(0.0,  -o, 0.0)) != 
           map0(p + vec3(0.0,   o, 0.0)) != 
           map0(p + vec3(0.0, 0.0,  -o)) != 
           map0(p + vec3(0.0, 0.0,   o));
}

// Function 693
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    
    int id = faceIdFromDir(rayDir);
    
    vec2 coord = fragCoord.xy;
    vec2 size = iResolution.xy;
    vec2 uv = coord / size;
    
    vec4 lastFrame = texture(iChannel0, rayDir);
    if (lastFrame.x != 0. && iFrame > 2) {
        fragColor = lastFrame;
    	return;
    }
    
    mat4 space = texToSpace(coord, id, size);
    
    vec3 p0 = space[0].xyz;
    vec3 p1 = space[1].xyz;
    vec3 p2 = space[2].xyz;
    vec3 p3 = space[3].xyz;

    fragColor = vec4(
        map(p0),
        map(p1),
        map(p2),
        map(p3)
    );
}

// Function 694
vec2 getuv_centerX(vec2 fragCoord, vec2 newTL, vec2 newSize, out float vignetteAmt)
{
    vec2 ret = vec2(fragCoord.x / iResolution.x, (iResolution.y - fragCoord.y) / iResolution.y);// ret is now 0-1 in both dimensions
    
    // vignette. return 0-1
    vec2 vignetteCenter = vec2(0.5, 0.5);// only makes sense 0-1 values here
    if(iMouse.z > 0.)
        vignetteCenter = vec2(iMouse.x / iResolution.x, (iResolution.y - iMouse.y) / iResolution.y);// ret is now 0-1 in both dimensions;
	vignetteAmt = 1.0 - distance(ret, vignetteCenter);
//    vignetteAmt = pow(vignetteAmt, 1.);
    
    ret *= newSize;// scale up to new dimensions
    float aspect = iResolution.x / iResolution.y;
    ret.x *= aspect;// orig aspect ratio
    float newWidth = newSize.x * aspect;
    return ret + vec2(newTL.x - (newWidth - newSize.x) / 2.0, newTL.y);
}

// Function 695
vec3 whitePreservingLumaBasedReinhardToneMapping(vec3 color)
{
	float white = 2.;
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma * (1. + luma / (white*white)) / (1. + luma);
	color *= toneMappedLuma / luma;
//	color = pow(color, vec3(1. / 2.2));
	return color;
}

// Function 696
vec4 map(vec3 pos, out float k) {
  vec4 p4 = iproj(pos,k);
  if (dorotate4) {
  	p4 = qmul(ql,p4);
  	p4 = qmul(p4,qr);
  }
  p4 = fold(p4);
  return p4;
}

// Function 697
vec2 screenToUv(vec2 uv, float angle)
{
    vec2 b0 = vec2(sin(angle), -cos(angle));
    b0 = normalize(b0);
    vec2 b1 = vec2(-b0.y, b0.x);
    
    vec2 uvCenter = uv - 0.5;
    uvCenter.x *= (iResolution.x / iResolution.y);
    uvCenter *= 1.5;
    return b0 * uvCenter.x + b1 * uvCenter.y + 0.5;
}

// Function 698
vec4   luvToRgb(float x, float y, float z, float a) {return   luvToRgb( vec4(x,y,z,a) );}

// Function 699
vec3 hsluvToLch(vec3 tuple) {  tuple.g *= hsluv_maxChromaForLH(tuple.b, tuple.r) * .01;  return tuple.bgr; }

// Function 700
vec2 map(vec3 pos, bool inside)
{
   vec2 res;
   float stone = map_stone(pos);
   #ifdef show_water
   float water = map_water(pos);
   if (inside)
       water = -water;
   res = opU(vec2(stone, METAL_OBJ), vec2(water, WATER_OBJ));
   #else
   res = vec2(stone, METAL_OBJ);
   #endif
   #ifdef show_stick
   float stick = map_stick(pos);
   return opU(vec2(stick, STICK_OBJ), res);
   #else
   return res;
   #endif
}

// Function 701
vec3 doBumpMap(in vec3 p, in vec3 n, float bumpfactor, inout float edge, inout float crv){
    
    // Resolution independent sample distance... Basically, I want the lines to be about
    // the same pixel with, regardless of resolution... Coding is annoying sometimes. :)
    vec2 e = vec2(1./iResolution.y, 0); 
    
    float f = bumpFunc(p, n); // Hit point function sample.
    
    float fx = bumpFunc(p - e.xyy, n); // Nearby sample in the X-direction.
    float fy = bumpFunc(p - e.yxy, n); // Nearby sample in the Y-direction.
    float fz = bumpFunc(p - e.yyx, n); // Nearby sample in the Y-direction.
    
    float fx2 = bumpFunc(p + e.xyy, n); // Sample in the opposite X-direction.
    float fy2 = bumpFunc(p + e.yxy, n); // Sample in the opposite Y-direction.
    float fz2 = bumpFunc(p + e.yyx, n);  // Sample in the opposite Z-direction.
    
     
    // The gradient vector. Making use of the extra samples to obtain a more locally
    // accurate value. It has a bit of a smoothing effect, which is a bonus.
    vec3 grad = vec3(fx - fx2, fy - fy2, fz - fz2)/(e.x*2.);  
    //vec3 grad = (vec3(fx, fy, fz ) - f)/e.x;  // Without the extra samples.


    // Using the above samples to obtain an edge value. In essence, you're taking some
    // surrounding samples and determining how much they differ from the hit point
    // sample. It's really no different in concept to 2D edging.
    edge = abs(fx + fy + fz + fx2 + fy2 + fz2 - 6.*f);
    edge = smoothstep(0., 1., edge/e.x*2.);
    
    
    // We may as well use the six measurements to obtain a rough curvature value while we're at it.
    //crv = clamp((fx + fy + fz + fx2 + fy2 + fz2 - 6.*f)*32. + .5, 0., 2.);
    
    // Some kind of gradient correction. I'm getting so old that I've forgotten why you
    // do this. It's a simple reason, and a necessary one. I remember that much. :D
    grad -= n*dot(n, grad);          
                      
    return normalize(n + grad*bumpfactor); // Bump the normal with the gradient vector.
	
}

// Function 702
float hsluv_maxSafeChromaForL(float L){  mat3 m2 = mat3(   3.2409699419045214 ,-0.96924363628087983 , 0.055630079696993609,   -1.5373831775700935 , 1.8759675015077207 ,-0.20397695888897657 ,   -0.49861076029300328 , 0.041555057407175613, 1.0569715142428786  );  float sub0 = L + 16.0;  float sub1 = sub0 * sub0 * sub0 * .000000641;  float sub2 = sub1 > 0.0088564516790356308 ? sub1 : L / 903.2962962962963;  vec3 top1 = (284517.0 * m2[0] - 94839.0 * m2[2]) * sub2;  vec3 bottom = (632260.0 * m2[2] - 126452.0 * m2[1]) * sub2;  vec3 top2 = (838422.0 * m2[2] + 769860.0 * m2[1] + 731718.0 * m2[0]) * L * sub2;  vec3 bounds0x = top1 / bottom;  vec3 bounds0y = top2 / bottom;  vec3 bounds1x =    top1 / (bottom+126452.0);  vec3 bounds1y = (top2-769860.0*L) / (bottom+126452.0);  vec3 xs0 = hsluv_intersectLineLine(bounds0x, bounds0y, -1.0/bounds0x, vec3(0.0) );  vec3 xs1 = hsluv_intersectLineLine(bounds1x, bounds1y, -1.0/bounds1x, vec3(0.0) );  vec3 lengths0 = hsluv_distanceFromPole( xs0, bounds0y + xs0 * bounds0x );  vec3 lengths1 = hsluv_distanceFromPole( xs1, bounds1y + xs1 * bounds1x );  return min(lengths0.r,    min(lengths1.r,    min(lengths0.g,    min(lengths1.g,    min(lengths0.b,     lengths1.b))))); }

// Function 703
vec4 hsluvToRgb(float x, float y, float z, float a) {return hsluvToRgb( vec4(x,y,z,a) );}

// Function 704
vec3 Uncharted2Tonemap(vec3 x){
   	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

// Function 705
vec4 nmaps(vec4 x){ return x*2.-1.; }

// Function 706
vec3 jodieReinhardTonemap(vec3 c)
{
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 tc = c / (c + 1.0);

    return mix(c / (l + 1.0), tc, tc);
}

// Function 707
float map_water(vec3 pos)
{
   pos = rotateVec2(pos);
   pos.y-= 0.3;
    
   float n1 = bumpfactor*noise2(normalize(pos));
   float df = length(pos) - 1.864 + n1;
   float be = 1./(0.07 + 4.*pow(abs(df), 0.35));
   df = max(df, pos.y + 0.3 - 0.06*be);
   return df;
}

// Function 708
vec4 lchToHsluv(float x, float y, float z, float a) {return lchToHsluv( vec4(x,y,z,a) );}

// Function 709
float map(in vec3 p)
{
    return min(s0(p), pl(p));
}

// Function 710
float map(vec3 p){
    
	// Height map to perturb the flat plane. On a side note, I'll usually keep the
    // surface function within a zero to one range, which means I can use it later
    // for a bit of shading, etc. Of course, I could cut things down a bit, but at
    // the expense of confusion elsewhere... if that makes any sense. :)
    float sf = surfFunc(p);

    // Add the height map to the plane.
    return p.y + (.5-sf)*2.; 
 
}

// Function 711
float map(vec3 p){

    // A bit of cheap, lame distortion for the heaving in and out effect.
    p.xy += sin(p.xy*7. + cos(p.yx*13. + iTime))*.01;
    
    // Back plane, placed at vec3(0., 0., 1.), with plane normal vec3(0., 0., -1).
    // Adding some height to the plane from the texture. Not much else to it.
    return 1. - p.z - texture(iChannel0, p.xy).x*.1;

    
    // Flattened tops.
    //float t = texture(iChannel0, p.xy).x;
    //return 1. - p.z - smoothstep(0., .7, t)*.06 - t*t*.03;
    
}

// Function 712
void ReflectUV(Line line, inout vec2 uv) {
    if (CheckDirection(line, uv)) {
    	vec2 pr = ProjectLine(line, uv);
    	uv = pr + (pr - uv);
    }
}

// Function 713
vec3 rgb2yuv(vec3 rgb)
{
    return vec3(0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b,
                -0.147 * rgb.r - 0.289 * rgb.g + 0.436 * rgb.b,
                0.615 * rgb.r - 0.515 * rgb.g - 0.100 * rgb.b);
}

// Function 714
float sceneMap3D(vec3 pos)
{
    // find the smallest t value for marching
    float t = plane(pos, vec4(0.0, 0.0, -1.0, 5.0));

    t = min(t, plane(pos, vec4(1.0, 0.0, 0.0, 5.0)));
    t = min(t, plane(pos, vec4(-1.0, 0.0, 0.0, 5.0)));
    t = min(t, plane(pos, vec4(0.0, -1.0, 0.0, 7.5)));
    t = min(t, plane(pos, vec4(0.0, 1.0, 0.0, 2.5)));
    t = min(t, box(rotateY(pos + vec3(0, 1, -2), -27.5 * 3.14159 / 180.0), vec3(1.5, 4, 1.5)));
    t = min(t, sphere(pos, 1.3, vec3(-3.5 * sin(iTime), 0.6 + 2.0 * cos(iTime), 2.0 + 3.5 * cos(iTime))));

    return t;
}

// Function 715
vec3 GetReadMipMapUVW_Dir(vec3 _uvw,vec3 _axis){
    return normalize(
        ((_uvw+3.)*exp2(vec3(5.-10.,8.-10.,2.-10.))-1.)
        * (1.-abs(_axis)) 
        + _axis
    );
}

// Function 716
vec3 Tonemap( vec3 x )
{
    return TonemapCompressRangeFloat3( x, 0.6 );
}

// Function 717
vec3 rgb2yuv(vec3 RGB)
{
	float y = dot( RGB, vec3( 0.299, 0.587, 0.114 ) );
	float u = dot( RGB, vec3( -0.14713, -0.28889, 0.436 ) );
	float v = dot( RGB, vec3( 0.615, -0.51499, -0.10001 ) );
	return vec3(y,u,v);
}

// Function 718
vec2 map(in vec3 pos) {	

    float d = mix(sdCube(pos,0.5773), length(pos)-1.0, sphere_fraction);    
    vec2 rval = vec2(d, 3.0);

    return rval;

}

// Function 719
float map(in vec3 p, vec2 sctime) {
	
	float res = 0.;
	
    vec3 c = p;
    c.xy = c.xy * sctime.x + vec2(c.y, c.x) * sctime.y;
	for (int i = 0; i < 10; ++i) 
    {
        p =.7*abs(p)/dot(p,p) -.7;
        p.yz= csqr(p.yz);
        p=p.zxy;
        res += exp(-19. * abs(dot(p,c)));        
	}
	return res/2.;
}

// Function 720
vec4 BoxMapFast( sampler2D sam, in vec3 p, in vec3 n, in float k )
{
  vec3 m = pow( abs(n), vec3(k) );
  vec4 x = textureLod( sam, p.yz ,0.4);
  vec4 y = textureLod( sam, p.zx ,0.4);
  vec4 z = textureLod( sam, p.xy ,0.4);
  return (x*m.x + y*m.y + z*m.z)/(m.x+m.y+m.z);
}

// Function 721
vec2 map( in vec3 pos )
{
    float s = 3.0;
    vec2 res = opU( vec2( sdPlane(pos), 1.0 ),
	           opU( vec2( sdSphere(pos-vec3(0.5864523963,0.0812291239,0.8045468825)*s, 0.3 ), 365.0 ), 
               opU( vec2( sdSphere(pos-vec3(0.4332892644,0.8688174619,0.9864079606)*s, 0.6 ), 300.0 ), 
               opU( vec2( sdSphere(pos-vec3(0.4042982001,0.2754675470,0.4787041755)*s, 0.25 ), 20.0 ), 
               opU( vec2( sdSphere(pos-vec3(0.7886073712,0.3325804610,0.0883206339)*s, 0.4 ), 78.0 ), 
               opU( vec2( sdSphere(pos-vec3(0.5443106183,0.8455826224,0.2311476150)*s, 0.4 ), 9421.0 ), 
                    vec2( sdSphere(pos-vec3(0.7178148587,0.9618010959,0.4239931090)*s, 0.4 ), 45.0 )
               ))))));
    return res;
}

// Function 722
float hsluv_lToY(float L) {  return L <= 8.0 ? L / 903.2962962962963 : pow((L + 16.0) / 116.0, 3.0); }

// Function 723
float map(vec3 p) {
    float d = sdPlane(p, vec4(0.0, 1.0, 0.0, 0.0));
    
    float rot_x = iTime * 3.1415 * 0.2;
    float cx = cos(rot_x);
    float sx = sin(rot_x);
    
    float rot_z = iTime * 3.1415 * 0.125;
    float cz = cos(rot_z);
    float sz = sin(rot_z);
    
    p = vec3(
        p.x,
        p.y * cx - p.z * sx,
        p.z * cx + p.y * sx
    );
    
    p = vec3(
        p.x * cz - p.y * sz,
        p.y * cz + p.x * sz,
        p.z
    );
    
    d = opU(d, sdBox(p - vec3(0.0, 1.5, -1.5), vec3(1.6, 1.5, 0.1)));
    d = opU(d, sdBox(p - vec3(1.5, 1.5, -0.25), vec3(0.1, 0.75, 2.25)));
 
    d = opU(d, opU_v2(sdSphere(p, 1.0), sdBox(p - vec3(0.75, 0.75, -0.75), vec3(0.75 - 0.025)) - 0.025, 0.1));
    //d = opU(d, opU_v2(sdSphere(p, 1.0), sdBox(p - vec3(0.75 * 3.0, 0.75, -0.75 * 3.0), vec3(0.75)) - 0.025));
    
    return d;
}

// Function 724
vec2 map(vec3 pos)
{
    vec2 wall = vec2(sdBox(pos - wallPos, wallSize), WALL_OBJ);
    float wallBound = sdBox(pos - wallPos, wallSize*1.04);
    vec2 bricks = max(map_bricks(pos), wallBound);
    #ifdef show_chimney
    vec2 chimney = map_chimney(pos);
    return opU(opU(wall, bricks), chimney);
    #else
    return opU(wall, bricks);
    #endif
}

// Function 725
PrimitiveDist map(vec3 p) {
	PrimitiveDist building = building(p);
    float plane = sdFloor(p);
    PrimitiveDist outSh = building;
    outSh = compPD(outSh, PrimitiveDist(plane, 1));
    return outSh;
}

// Function 726
vec3 hpluvToLch(float x, float y, float z) {return hpluvToLch( vec3(x,y,z) );}

// Function 727
vec3 map( in vec3 p )
{
    vec3 res = vec3(length(p-gVerts[0]),0.0,0.0);

    
    //   2---3
    //  /   /|
    // 6---7 |
    // |   | 1
    // 4---5/
    
    vec3 tmp;
    tmp = sdBilinearPatch(p, gVerts[0], gVerts[2], gVerts[3], gVerts[1]); if( tmp.x<res.x ) res = tmp;
    tmp = sdBilinearPatch(p, gVerts[7], gVerts[6], gVerts[4], gVerts[5]); if( tmp.x<res.x ) res = tmp;
    tmp = sdBilinearPatch(p, gVerts[0], gVerts[1], gVerts[5], gVerts[4]); if( tmp.x<res.x ) res = tmp;
    tmp = sdBilinearPatch(p, gVerts[2], gVerts[6], gVerts[7], gVerts[3]); if( tmp.x<res.x ) res = tmp;
    tmp = sdBilinearPatch(p, gVerts[0], gVerts[4], gVerts[6], gVerts[2]); if( tmp.x<res.x ) res = tmp;
    tmp = sdBilinearPatch(p, gVerts[1], gVerts[3], gVerts[7], gVerts[5]); if( tmp.x<res.x ) res = tmp;
    
    res.x -= kRoundness; // round it a bit
    return res;
}

// Function 728
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    const vec2 e = vec2(.001, 0);
    float ref = bumpSurf3D(p);                 
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy),
                      bumpSurf3D(p - e.yxy),
                      bumpSurf3D(p - e.yyx) )-ref)/e.x;                     
          
    grad -= nor*dot(nor, grad);          
                      
    return normalize(nor + grad*bumpfactor);
	
}

// Function 729
float map(vec3 p)
{
    return min(min(box(p+vec3(+1.0,0.0,0.0)), cylinder(p+vec3(-1.0,0.0,0.0))), halfspace(p));
    //return min(min(box(p+vec3(+1.0,0.0,0.0)), sphere(p+vec3(-1.0,0.0,0.0))), halfspace(p));
    //return box(p);
}

// Function 730
vec4 boxmap(sampler2D sam,vec3 p,vec3 n)
{
    vec3 m = pow(abs(n), vec3(32.) );
	vec4 x = texture( sam, p.yz );
	vec4 y = texture( sam, p.zx );
	vec4 z = texture( sam, p.xy );
	return (x*m.x + y*m.y + z*m.z)/(m.x+m.y+m.z);
}

// Function 731
vec2 mapTransparent( vec3 p, out vec4 matInfo )
{
    matInfo = vec4(0.0);
    
    float d5 = mapDrop( p );
    vec2  res = vec2(d5,4.0);

    float d6 = mapLeafWaterDrops( p );
    res.x = min( res.x, d6 );

    return res;
}

// Function 732
float map_handles(vec3 pos)
{
    pos.y = mod(pos.y, floor_height);
    vec3 pos2 = pos;
    if (pos.x>-staircase_length/2. + 1.)
    {
        pos2.z = abs(pos2.z);
        pos2-= vec3(-staircase_length/2. + floor_width/2. + door_width/2.45, door_height*handle_height, staircase_width/2. + door_depth - handle_length1);
        pos2.xz = rotateVec(pos2.xz + vec2(floor_width/2. - door_width/2., 0.), door_angle) - vec2(floor_width/2. - door_width/2., 0.);
    }
    else
    {
        pos2.z = abs(pos2.z);
        pos2+= vec3(staircase_length/2. + door_depth - handle_length1, -door_height*handle_height, -staircase_width/4. - door_width/2.45);
        pos2.xz = pos2.zx;
        pos2.z = -pos2.z;
        pos2.xz = rotateVec(pos2.xz + vec2(staircase_width/4., 0.), door_angle) - vec2(staircase_width/4., 0.);
    }
    
    float handle = sdCylinder(pos2.yxz - vec3(0., -handle_length2, -handle_length1), vec2(handle_radius, handle_length2), 0., 0.);
    handle = min(handle, sdCylinder(pos2.xzy, vec2(handle_radius, handle_length1), 0., 0.));
    handle = min(handle, length(pos2.yxz - vec3(0., -handle_length2 + handle_length2, -handle_length1)) - handle_radius);

    handle = min(handle, sdRoundBox(pos2 - vec3(0., -0.43*handle_plate.y, handle_length1), handle_plate.xyz, handle_plate.w));
    handle = min(handle, sdCylinder(pos2.xzy - vec3(0., 1.65*handle_plate.z - keycyl_length + handle_length1/2., -0.81*handle_plate.y), vec2(handle_radius, keycyl_length), 0., 0.));
    
    return handle;
}

// Function 733
float map(vec3 p )
{
    vec3 q = p;
 // vec3 N = 2.* noise(q/10.) -1.;                // displacement
    vec3 N = 2.* divfreenoise(q/10.);
    q += .5*N;
    float f = ( 1.2*noise(q/2.+ .1*iTime).x -.2 ) // source noise
              * smoothstep(1.,.8,length(q)/2.);   // source sphere

    f*= smoothstep(.1,.2,abs(p.x));               // empty slice (derivable ) 
    z = length(q)/2.;                             // depth in sphere
    return f;                        
}

// Function 734
vec3 Uncharted2ToneMapping(vec3 color)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	float W = 11.2;
	float exposure = 0.012;
	color *= exposure;
	color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
	float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
	color /= white;
	color = pow(color, vec3(1. / gamma));
	return color;
}

// Function 735
vec3 BakingLabACESTonemap(vec3 color, bool isInverse)
{
	const float A = 0.0245786;
    const float B = 0.000090537;
    const float C = 0.983729;
    const float D = 0.4329510;
    const float E = 0.238081;
    
    if (!isInverse)
    {
        color = (
            (color * (color + A) - B) /
            (color * (C * color + D) + E));
        color = clamp(color, 0.0, 1.0);
        
        return color;
    }
    
    color = abs(
        (
            (A - D * color) -
            sqrt(
                pow(abs(D * color - A), vec3(2.0)) -
                4.0 * (C * color - 1.0) * (B + E * color))
        ) /
        (2.0 * (C * color - 1.0))
    );
    
    return color;
}

// Function 736
vec2 map(vec3 p, float t)
{
 
    float matID, f;
    p.y += getGroundHeight(p.xz);
	float num = (floor(p.z/5.))*5.+(floor(p.x/5.0))*19.;
	p.xz = mod(p.xz, 5.0)-2.5;
    //p.xz *= rotMat2D(p.y*num/300.); // ... No, just too expensive. :)
    
    float d = p.y;
    matID = 0.0;

    float s=1.,ss=1.6;
    
    // Tangent vectors for the branch local coordinate system.
    vec3 w=normalize(vec3(-1.5+abs(hash11(num*4.)*.8),1,-1.));
    vec3 u=normalize(cross(w,vec3(0,1.,0.)));

    float scale=3.5;
    p/=scale;
    vec3 q = p;
    // Make the iterations lessen over distance for speed up...
    int it = 10-int(min(t*.03, 9.0));

	float h  = hash11(num*7.)*.3+.3;
    vec3 uwc = normalize(cross(u,w));
    int dontFold = int(hash11(num*23.0) * 9.0)+3;
    
    float thick = .2/(h-.24);
    for (int i = 0; i < it; i++)
    {
		f = scale*max(p.y-h,max(-p.y,length(p.xz)-.06/(p.y+thick)))/s;
        if (f <= d)
        {
            d = f;
            matID = 1.0;
        }

        // Randomly don't fold the space to give more branch types...
        if (i != dontFold)
        	p.xz = abs(p.xz);

        p.y-=h;
        p*=mat3(u,uwc,w);
        p*=ss;
		s*=ss;
    }

    float fr = .2;
    f = (length(p)-fr)/s;
    if (f <= d)
    {
        d = f;
        matID = 2.0;
    }
    
    q.y -= h*1.84;
    h *= 1.1;
    for (int i = 0; i < it; i++)
    {
      	p = (normalize(hash31(num+float(i+19))-.5))*vec3(h, 0.1, h);
     	p+=q;
        float ds =length(p)-.015;
     	if (ds <= d)
        {
            matID = 3.0+float(i);
         	d = ds;
        }
    }

	return vec2(d, matID);
}

// Function 737
float map(vec3 p){
    
 
    p.xy -= path(p.z); // Wrap the passage around
    
    vec3 w = p; // Saving the position prior to mutation.
    
    vec3 op = tri(p*.4*3. + tri(p.zxy*.4*3.)); // Triangle perturbation.
   
    
    float ground = p.y + 3.5 + dot(op, vec3(.222))*.3; // Ground plane, slightly perturbed.
 
    p += (op - .25)*.3; // Adding some triangular perturbation.
   
	p = cos(p*.315*1.41 + sin(p.zxy*.875*1.27)); // Applying the sinusoidal field (the rocky bit).
    
    float canyon = (length(p) - 1.05)*.95 - (w.x*w.x)*.01; // Spherize and add the canyon walls.
    
    return min(ground, canyon);

    
}

// Function 738
float llamelMap(in vec3 p) {
	const vec3 rt=vec3(0.0,0.0,1.0);	
	p.y += 0.25*llamelScale;
    p.xz -= 0.5*llamelScale;
    p.xz = vec2(-p.z, p.x);
    vec3 pori = p;
        
    p /= llamelScale;
    
	vec2 c=floor(p.xz);
	p.xz=fract(p.xz)-vec2(0.5);
    p.y -= p.x*.04*llamelScale;
	float sa=sin(c.x*2.0+c.y*4.5+llamelTime*0.05)*0.15;

    float b=0.83-abs(p.z);
	float a=c.x+117.0*c.y+sign(p.x)*1.57+sign(p.z)*1.57+llamelTime,ca=cos(a);
	vec3 j0=vec3(sign(p.x)*0.125,ca*0.01,sign(p.z)*0.05),j3=vec3(j0.x+sin(a)*0.1,max(-0.25+ca*0.1,-0.25),j0.z);
	float dL=llamelMapLeg(p,j0,j3,vec3(0.08,0.075,0.12),vec4(0.03,0.02,0.015,0.01),rt*sign(p.x));
	p.y-=0.03;
	float dB=(length(p.xyz*vec3(1.0,1.75,1.75))-0.14)*0.75;
	a=c.x+117.0*c.y+llamelTime;ca=cos(a);sa*=0.4;
	j0=vec3(0.125,0.03+abs(ca)*0.03,ca*0.01),j3=vec3(0.3,0.07+ca*sa,sa);
	float dH=llamelMapLeg(p,j0,j3,vec3(0.075,0.075,0.06),vec4(0.03,0.035,0.03,0.01),rt);
	dB=llamelMapSMin(min(dL,dH),dB,clamp(0.04+p.y,0.0,1.0));
	a=max(abs(p.z),p.y)+0.05;
	return max(min(dB,min(a,b)),length(pori.xz-vec2(0.5)*llamelScale)-.5*llamelScale);
}

// Function 739
bool map0(vec3 p)
{
    p += 0.5;
    vec3 b = abs(p);
    
    bool r;
    
    r =      b.x < 4.0;
    r = r && b.y < 4.0;
    r = r && b.z < 4.0;
    
    r = r && (b.x > 2.0 || b.y > 2.0);
    r = r && (b.z > 2.0 || b.y > 2.0);
    r = r && (b.x > 2.0 || b.z > 2.0);    
    
    return r;
}

// Function 740
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    int idx = dir2face(rayDir);    
    
    vec3 col = vec3(0);
    
    switch (idx) {
        case FACE_X_POS: col = vec3(1.0,0.0,0.0); break;
        case FACE_X_NEG: col = vec3(0.5,0.0,0.0); break;
        case FACE_Y_POS: col = vec3(0.0,1.0,0.0); break;
        case FACE_Y_NEG: col = vec3(0.0,0.5,0.0); break;
        case FACE_Z_POS: col = vec3(0.0,0.0,1.0); break;
        case FACE_Z_NEG: col = vec3(0.0,0.0,0.5); break;
    }
    
    fragColor = vec4(col, 1);
}

// Function 741
vec3 UE3TonemapInv(vec3 color)
{
    return (color * -0.155) / (max(color, 0.01) - 1.019);
}

// Function 742
vec3 rgb2yuv(vec3 rgb) {
  return rgb * rgb2yuv_mat;
}

// Function 743
float mapHeightLQ(in vec3 rp)
{
    return rp.y - mapBottom(rp);
}

// Function 744
vec3 YUVtoRGB(vec3 yuv) {
    yuv.gb -= 0.5;
    return vec3(
        yuv.r * 1.0 + yuv.g * 0.0 + yuv.b * 1.5748,
        yuv.r * 1.0 + yuv.g * -0.187324 + yuv.b * -0.468124,
        yuv.r * 1.0 + yuv.g * 1.8556 + yuv.b * 0.0);
}

// Function 745
float map(vec3 p){
    return length(p) - 1.; //min(s, s2);
}

// Function 746
vec2 map( in vec3 pos ) {
    float d = length(pos);
    float fft = 0.8*texture(iChannel0, vec2((d/32.0),0.3) )[0];	
	return vec2(mandel3D(pos*1.5,fft),fft);
}

// Function 747
float map(vec3 pos)
{     
    float angle = 2.*pi*iMouse.x/iResolution.x;
    float angle2 = -2.*pi*iMouse.y/iResolution.y;
    
    vec3 posr = pos;
    posr = vec3(posr.x, posr.y*cos(angle2) + posr.z*sin(angle2), posr.y*sin(angle2) - posr.z*cos(angle2));
    posr = vec3(posr.x*cos(angle) + posr.z*sin(angle), posr.y, posr.x*sin(angle) - posr.z*cos(angle));
    
    float d = 0.94;
    float b = 0.5;

    float af2 = 4./pi;
    float s = atan(posr.y, posr.x);
    float sf = floor(s*af2 + b)/af2;
    float sf2 = floor(s*af2)/af2;
    
    vec3 flatvec = vec3(cos(sf), sin(sf), 1.444);
    vec3 flatvec2 = vec3(cos(sf), sin(sf), -1.072);
    vec3 flatvec3 = vec3(cos(s), sin(s), 0);
    float csf1 = cos(sf + 0.21);
    float csf2 = cos(sf - 0.21);
    float ssf1 = sin(sf + 0.21);
    float ssf2 = sin(sf - 0.21);
    vec3 flatvec4 = vec3(csf1, ssf1, -1.02);
    vec3 flatvec5 = vec3(csf2, ssf2, -1.02);
    vec3 flatvec6 = vec3(csf2, ssf2, 1.03);
    vec3 flatvec7 = vec3(csf1, ssf1, 1.03);
    vec3 flatvec8 = vec3(cos(sf2 + 0.393), sin(sf2 + 0.393), 2.21);
     
    float d1 = dot(flatvec, posr) - d;                           // Crown, bezel facets
    d1 = max(dot(flatvec2, posr) - d, d1);                       // Pavillon, pavillon facets
    d1 = max(dot(vec3(0., 0., 1.), posr) - 0.3, d1);             // Table
    d1 = max(dot(vec3(0., 0., -1.), posr) - 0.865, d1);          // Cutlet
    d1 = max(dot(flatvec3, posr) - 0.911, d1);                   // Girdle
    d1 = max(dot(flatvec4, posr) - 0.9193, d1);                  // Pavillon, lower-girdle facets
    d1 = max(dot(flatvec5, posr) - 0.9193, d1);                  // Pavillon, lower-girdle facets
    d1 = max(dot(flatvec6, posr) - 0.912, d1);                   // Crown, upper-girdle facets
    d1 = max(dot(flatvec7, posr) - 0.912, d1);                   // Crown, upper-girdle facets
    d1 = max(dot(flatvec8, posr) - 1.131, d1);                   // Crown, star facets
    return d1;
}

// Function 748
void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir){
    
    
    //float a = mod(float(iFrame), 8.)*3.14159/4.;// + rot2(a)*vec2(0, .5)
    // UV coordinates.
    //
    // For whatever reason (which I'd love expained), the Y coordinates flip each
    // frame if I don't negate the coordinates here -- I'm assuming this is internal, 
    // a VFlip thing, or there's something I'm missing. If there are experts out there, 
    // any feedback would be welcome. :)
    vec2 uv = fract(fragCoord/iResolution.y*vec2(1, -1));
    
    gSc = 1.;
 
    // Pixel storage.
    vec4 col;
   
    // Initial conditions -- Performed upon initiation.
    if(abs(tx(uv).w - iResolution.y)>.001){
    
        // Initial conditions: Fill each channel in each cell with some random values.
        col = vec4(hash21(uv), hash21(uv + .17), hash21(uv + .23), 1.);
        col.w = iResolution.y;
    }
    else {
    
        // A very rough Belousov–Zhabotinsky reaction approximation -- Feel free to look
        // up the process in detail, but it's similar to many reaction diffusion like
        // examples: Start off with an initial solution in the form of noise in one or 
        // some of the channels, use filters to blur it over time to similute dispersion,
        // then mix the result with the previous frame. In this case, we can simulate 
        // non-equilibrium by sprinkling in extra noise for volatility... As mentioned,
        // there are others on the net and on Shadertoy who can give you more detail, but
        // that's the general gist of it.
        
        // Thinking a little outside the box, it's possible to use a much cheaper 
        // radial boundary blur with a larger radius to mickick a larger block blur. 
        // It doesn't work in all situations, but it works well enough here.
        vec4 val = bTxCir(uv, 5.); // 12 Taps.
        val = mix(val, tx(uv), 1./25.); // Adding the center pixel.
        //vec4 val = bTx(uv, 7); // Box blur: 49 taps -- Requires rescaling.
        
        //#if 0
        // Alternate, simpler equation.
        //col = clamp(tx(uv) + .08*(val.zxyw - val.yzxw), 0., 1.);
        //#else
        float reactionRate = val.x*val.y*val.z; // Self explanitory.
        //float reactionRate = smoothstep(0., 1., val.x*val.y*val.z); 

        // Producing the new value: For an explanation, you can look up the chemical
        // reaction it pertains to and the mathematical translation which is pretty
        // interesting. From a visual perspective, however, it's just a cute calculus 
        // based equation that produces a cool pattern over time.
        vec4 res = val - reactionRate + val*(val.yzxw - val.zxyw);
        //vec4 params = vec4(1, 1, 1, 0);//
        //vec4 res = val - reactionRate + val*(params*val.yzxw - params.zxyw*val.zxyw);
        
        
 

        // Adding some volatile noise to the system. 
        vec3 t = vec3(1.01, 1.07, 1.03)*fract(iTime);
        vec4 ns = vec4(hash21(uv + .6 + t.x), hash21(uv + .2 + t.y), hash21(uv + .7 + t.z), 0);
        
        // Mixing the new value and noise with the old value. 
        col = mix(tx(uv), res*(.9 + ns*.3), .2);
        //#endif
        
 
        // Using the fourth channel to store resolution.
        col.w = iResolution.y;
    
    }
    
    // Recording the new value and clamping it to a certain range.
    fragColor = vec4(clamp(col.xyz, -1., 1.), iResolution.y);
}

// Function 749
float heightMap(in vec2 uv)
{
    uv /= 6.;
    // mirror repeat
	vec2 m1 = mod(uv, 1.), m2 = mod(uv, 2.);
    uv = mix(m1, 1.-m1, max(vec2(0.), sign(m2-1.)));
    
    vec4 k = texture(iChannel1, uv);
	return 0.2*k.x;    
}

// Function 750
float hsluv_fromLinear(float c) {
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// Function 751
float map(vec3 p){
    
    const float depth = .1; // Depth of the rounded metal plates.
    
    // Mold the scene around the path.
    p.xy -= camPath(p.z).xy;
    
    // The edge of the tunnel. Set at a radius of one, plus the depth.
    float tun = (1. + depth) - length(p.xy); 
    
    //////////////
    
    // The concrete floor. Just a plane, with a circular chunk cut out. It gives it
    // a curb-like appearance.
    float flr = p.y + .695;
    flr = max(flr, tun - depth - .1);
    
    ///////////////
    
    // The tunnel walls. Due to the amount of detailing and the polar conversion, it 
    // looks more complicated than it actually is. To repeat across the XY plane we
    // do something along the lines of "p.xz = mod(p.xz, c) - c/2." To repeat around
    // a circle in the XY plane, we convert to polar coordinates, "p.xy = rot(angle),"
    // (angle is based on "atan(p.y, p.x)," then do the same thing. The rest is basic
    // unit circle trigonometry, etc. By the way, this is a rough description, so if
    // something doesn't quite make sense, it probably doesn't. :)
    
    // Converting the XY plane to polar coordinates. I'm handling the panels (six per
    // circle) and the bolts (18 per circle) at the same time to share some calculations.
    // I'd love to use the geometry of one to constuct the other - in order to save
    // some instructions, but I'm leaving it alone for now.
    vec3 q = p; 
    vec3 q2 = p;    
    
    float a = atan(q.y, q.x)/6.2831853; // Polar angle of "p.xy" coordinate.
    float ia = (floor(a*6.) + .5)/6.*6.2831853; // Angle between "PI/6" intervals.
    float ia2 = (floor(a*18.) + .5)/18.*6.2831853; // Angle between "PI/18" intervals.
    
     // Polar conversion for 6 segments, but offset every second panel... and shifted
    // to the center-cell position (see the Z-repetition).
    q.xy *= rot(ia + sign(mod(q.z + .25, 1.) - .5)*3.14159/18.);
    q2.xy *= rot(ia2); // Polar conversion for 18 segments (for the bolts).
   
    // The X-coordinate is now the radial coordinate, which radiates from the center
    // to infinity. We want to break it into cells that are 2 units wide, but centered
    // in the middle. The result is that the panels will start at radius one.
    q.x = mod(q.x, 2.) - 1.;
    // Plain old linear Z repetion. We want the panels and bolts to be repeated in the
    // Z-direction (down the tunnel) every half unit.
    q.z = mod(q.z, .5) - .25; 
    
    // Moving the bolts out to a distance of 2.1.
    q2.x = mod(q2.x, (2. + depth)) - (2. + depth)/2.;
    
    // Now, it's just a case of drawing an positioning some basic shapes. Boxes and
    // tubes with a hexagonal cross-section.
    q = abs(q);
    q2 = abs(q2);

    // Bolts. Hexagon shapes spaced out eighteen times around the tunnel walls. The 
    // panels are spaced out in sixths, so that means three per panel.
    float blt = max(max(q2.x*.866025 + q2.y*.5, q2.y) - .02, q.z - .08);

    
    // Putting in some extra rails where the mesh and curb meets the tunnel. The extra
    // code is fiddly (not to mention, slows things down), but it makes the structure
    // look a little neater.
    q2 = p;
    q2.xy *= rot(ia - sign(p.x)*3.14159/18.);
    q2 = abs(q2);
    
    // Lines and gaps on the tunnel to give the illusion of metal plating.
    float tunDetail = max(min(min(q.y - .06, q.z - .06), max(q2.y - .06, p.y)), 
                          -min(min(q.y - .01, q.z - .01), max(q2.y - .01, p.y))); 
 
    // Adding the tunnel details (with a circular center taken out) to the tunnel.
    tun = min(tun, max(tunDetail, tun-depth));  
    
    ///////////////
    
    // The metalic mesh elements and light casings. The lights are calculated in this
    // block too.
        // The metalic mesh elements and light casings. The lights are calculated in this
    // block too.
       
    q = abs(p);    
    float mtl = max(q.x - .14, abs(p.y - .88) - .02);  // Top mesh.
    mtl = min(mtl, max(q.x - .396, abs(p.y + .82) - .02)); // Bottom mesh.//.81
    
    q.z = abs(mod(p.z, 2.) - 1.);
    
    float lgt = max(max(q.x - .07, q.z - .07), abs(p.y - 1.) - .255);
    float casings = max(max(q.x - .1, q.z - .1), abs(p.y - 1.) - .23);
    
    q.xz = abs(mod(q.xz, 1./8.) - .5/8.);
    
    mtl = max(mtl, -max(max(q.x - .045, q.z - .045), abs(abs(p.x) - .19) - .14)); // Holes in the mesh.
    mtl = min(mtl, casings ); // Add the light casings to the top mesh.
    
    /*
    // Alternative mesh setup with smaller holes. I like it more, but Moire patterns are a problem
    // with smaller window sizes.
    q = abs(p);    
    float mtl = max(q.x - .13, abs(p.y - .88) - .02);  // Top mesh.
    mtl = min(mtl, max(q.x - .396, abs(p.y + .82) - .02)); // Bottom mesh.//.81
    
    q.z = abs(mod(p.z, 2.) - 1.);
    
    float lgt = max(max(q.x - .07, q.z - .07), abs(p.y - 1.) - .255);
    float casings = max(max(q.x - .1, q.z - .1), abs(p.y - 1.) - .23);
    
    q.xz = abs(mod(q.xz, 1./16.) - .5/16.);
    
    mtl = max(mtl, -max(max(q.x - .025, q.z - .025), abs(abs(p.x) - .19) - .155)); // Holes in the mesh.
    mtl = min(mtl, casings ); // Add the light casings to the top mesh.
    */    
    ///////////////
    
    // Pipes. Electricity... water? Not sure what their function is, but I figured I 
    // should slow the distance function down even more, so put some in. :)
    q = p;
    const float th = 6.283/18.;
    float sx = sign(p.x);
    float pip = length(q.xy - vec2(sx*sin(th*1.4), cos(th*1.4))*1.05) - .015;
    pip = min(pip, length(q.xy - vec2(sx*sin(th*1.6), cos(th*1.6))*1.05) - .015);
    
    ///////////////
    
    // Determine the overall closest object and its corresponding object ID. There's a way
    // to save some cycles and take the object-ID calculations out of the distance function, 
    // but I'm leaving them here for simplicity.
    vec2 d = objMin(vec2(tun, TUN), vec2(blt, BLT));
    d = objMin(d, vec2(mtl, MTL));
    d = objMin(d, vec2(lgt, LGT));
    d = objMin(d, vec2(flr, FLR));
	d = objMin(d, vec2(pip, PIP));
    
    ///////////////
    
    
    objID = d.y; // Set the global object ID.
    
    return d.x; // Return the closest distance.
    
    
}

// Function 752
vec2 map(vec3 pos)
{
    vec2 scene = vec2(0.0, 0.0);
    
    scene.x = pos.y;
    
    vec3 sp = pos + vec3(sin(pos.z+iTime), -.1, 0.0);
    float snake = sdSphere(sp, .5);
    
    for(int i=0;i<10;i++)
    {
        vec3 tsp = sp + vec3(0.0, 0.0, 1.0+1.0*float(i));
        snake = opSmoothUnion(snake, sdSphere(tsp, .5), .5);
    }
    
    // Materials
    scene = opU2(scene, vec2(snake, 1.0));
    
    return scene;
}

// Function 753
float map(vec3 p0) { 
    
	float d= 999.;
    float h = Terrain(p0.xz*.3);
        
    malaxSpace(p0);
 	d = p0.y - h*mix(.002,.04, smoothstep(.1,3.,sign(p0.z)*abs(p0.x))*smoothstep(10.,MAX_DIST,length(p0.xz)));
    
    vec3 p = p0;
    float dc = length(p.xz-vec2(0,2));
    
 	// Rempart
    d = min(d, max(-min(.8-p.y, dc-6.-.3),sdCone(p-vec3(0,6.*8.4,2.), normalize(vec2(1.,.13)))));   
	d = min(d, sdRoundCreneaux(p-vec3(0,.9,2.), 6.55));
    d = min(d, sdStairs(p-vec3(1.2,-.45,8.2))); // Escaliers 
    
    vec3 p1 = p-vec3(0,0,2);
    pMirrorOctant(p1.xz, vec2(3.9)); // pour faire 4 tours d'un coups
    
    p.x = abs(p.x);
    d = min(d, max(-1.1+p1.y, sdCone(p1 - vec3(2.4,5.,-2.4), normalize(vec2(1.,.1))))); // Tour rampart
    d = max(d, -sdTorus82(p-vec3(0,.91,2.), 6.375, vec2(.08, .15))); // Porte tour rampart
    d = min(d, sdRoundCreneaux(p1-vec3(2.4,1.3,-2.4), .5));

    p.z += .05;
  
    vec3 p2 = p, ph = p;

    // Chemin de ronde
    p.z -= .1;
    pReflect(p, normalize(vec3(-1.,0,1.)),1.7);
    pReflect(p, normalize(vec3( 1.,0,1.)),1.2);

    p1 = p;
    p1.x = abs(p1.x); // Pour doubler les tours
    p1 -= vec3(1.2,0.,0.);

    // Tour du chemin de ronde
    d = min(d, max(-1.7+p1.y, sdCone(p1 - vec3(0,7.,0.), normalize(vec2(1.,.05)))));
    d = min(d, sdRoundCreneaux(p1-vec3(0,1.9,0), .35));
    d = min(d, sdWall(p-vec3(.5,0.,-.07), 1.1,1.));  // Mur droit
    d = min(d, sdLineCreneaux(p-vec3(0.,1.3,.0)));

    // Donjon
    d = min(d, sdHouse((vec3(p.x-.2,p.y,p.z+1.2)).zyx*1.6)/1.6);
    d = min(d, sdWall(p-vec3(.0,0.,-1.28),2.,1.));
    
    // Tour du donjon
    float d2 = sdLineCreneaux(p-vec3(0.,2.2,-1.2));
    d2 = min(d2, sdCapsule(p, vec3(.28,1.9,-1.3), vec3(.28,2.7,-1.3), .09, .17));

#ifdef FULL_MAP    
    d = fOpUnionStairs(d, d2, .04, 3.);
#else
    d = min(d, d2);
#endif
    d = min(d, max(-p.y+2.7,
                   min(sdCone((p-vec3(.28,3.3,-1.3)), vec2(1.,.4)),
                       sdCone((p-vec3(.28,3.6,-1.3)), vec2(1.,.22)))));
 	float dWin = sdWindow(p.xy-vec2(.28,2.45));
  
	d = -fOpUnionStairs(dWin,-d, .05,2.);

    ph.z -= .5;
	pR45(ph.zx);
    ph.z -= 4.6;
    ph.x += 1.;
 
    pReflect(ph, normalize(vec3(-1.,0,.7)),1.);
    
    d = min(d, fBlob((ph-vec3(0,1.,0))*4.)/4.); // arbre feuilles
    d = min(d, max(ph.y-1.,length(ph.xz)-.04)); // arbre tronc

    pMirrorOctant(ph.xz, vec2(1.5,1.6));

    // Petites maisons
    d = min(d, sdHouse((vec3(ph.x-.2,ph.y,ph.z+.6))*3.)/3.);
#ifdef FULL_MAP  
    d = min(d, fBoxCheap(ph-vec3(.15,0.,-.95), vec3(.05,.9,.05)));  // cheminee
#endif  
    
    d = min(d, sdStairs(p2-vec3(1.2,-.01,-.285)));    // escaliers

   // r = length(p0.yz-vec2(.4,-2.2));
   // d = min(d, max(abs(p0.x)-.2, r-.04)); 
    
    // Grande porte
    p0.x = abs(p0.x)+.1; // gothique    
 
    float dDoor = min(fCylinder(p0.xzy-vec3(0.,3.5,0.5),.2,6.), fBoxCheap(p0-vec3(0.,.35,3.5), vec3(.2,.18,6.)));

	d = max(-fBoxCheap(p-vec3(1.5,1.35,-.15), vec3(3.5,.15,.07)), d); // Porte chemin de ronde
    d = fOpUnionStairs(d, fBoxCheap(p0-vec3(0,.18,-1.35),vec3(.4,.6,.1)),.1,5.);
    d = -fOpUnionStairs(-d, fBoxCheap(p0-vec3(0,.18,-1.1),vec3(.37,.57,.1)),.02,2.);
    d = fOpDifferenceColumns(d, dDoor, .03,3.);
    d = min(d, fBoxCheap(p0-vec3(0,.185,-1.2),vec3(.38,.05,.1)));
    d = min(d,.025*sdCircleStairs(40.*(vec3(.45-p0.x,p0.y+.09,p0.z+1.1)))); // escalier circulaires de l'entree
	d = min(d, fBoxCheap(vec3(p.x,abs(p.y-1.15)-.95,p.z+1.8), vec3(10.5,.02,0.5)));

    // Puit
    float r = length(p0.xz-vec2(3.,3.1));
    d = min(d, max(p0.y-.3, r-.2)); 
	d = max(d, .14-r);

    return d;
}

// Function 754
float map(vec3 p){
    return smoothMinP(min(water(p), seaPillars(p)), seaGround(p), 1.13);
}

// Function 755
vec4 bufferCubemap(in sampler2D buffer, in vec2 bufferSize, in vec3 d) 
{
    bufferSize.y = min(bufferSize.y, bufferSize.x*0.66667 + 4.0);
    
    vec3 i = 1.0 / min(-d, d);
    vec3 p = d * -0.5 * max(max(i.x, i.y), i.z);
	vec3 n = sign(d) * step(i.yzx, i.xyz) * step(i.zxy, i.xyz);
    
	vec2 px = vec2(-p.z*n.x, p.y) * step(0.5, n.x)
              + vec2(-p.y, p.z*n.x) * step(n.x, -0.5);
    vec2 py = vec2(-p.x*n.y, -p.z) * abs(n.y);
    vec2 pz = vec2(p.x*n.z, p.y) * abs(n.z);
    
	vec2 t = vec2(abs(n.x) + 2.0*(step(n.y, -0.5) + step(n.z, -0.5)),
			      abs(n.y) + step(n.x, -0.5));
    
    vec2 uv = (vec2(t.x, 0.0) + (px + py + pz) + 0.5)
              * (bufferSize.y - 2.0) * 0.5/bufferSize;
    uv.y = abs(t.y - uv.y);
    
    return texture(buffer, uv, -100.0);
}

// Function 756
vec2 hMap(vec2 uv){
    
    // Plain Voronoi value. We're saving it and returning it to use when coloring.
    // It's a little less tidy, but saves the need for recalculation later.
    float h = Voronoi(uv*6.);
    
    // Adding some bordering and returning the result as the height map value.
    float c = smoothstep(0., fwidth(h)*2., h - .09)*h;
    c += (1.-smoothstep(0., fwidth(h)*3., h - .22))*c*.5; 
    
    // Returning the rounded border Voronoi, and the straight Voronoi values.
    return vec2(c, h);
    
}

// Function 757
vec4 o5229_input_color_map(vec2 uv) {
vec2 o5242_0_wat = abs((uv) - 0.5);
float o5242_0_d = o5242_0_wat.x+o5242_0_wat.y;vec4 o5242_0_1_rgba = o5242_gradient_gradient_fct(fract(2.0*(5.0/3.0)*o5242_0_d));
vec4 o5263_0_1_rgba = o5263_f(o5242_0_1_rgba);

return o5263_0_1_rgba;
}

// Function 758
vec2 map(vec3 pos)
{
    vec3 cp = vec3(0.0,0.0,0.0);
    
    vec2 res = opU(vec2(sdPlane(pos - vec3(0.0,0.0,0.0) + cp),1.0),
                   vec2(sdSphere(pos - vec3(0.0,0.5,0.0) + cp,0.5),46.9));
    
    float b = opBlend(udBox(pos - vec3(1.0,0.5,0.0) + cp,vec3(0.5,0.5,0.5)),
                      sdSphere(pos - vec3(1.0,0.5,0.0) + cp,0.5),(sin(iTime)+1.0)/2.0);
    res = opU(res, vec2(b,78.5));
    
    b = opI(udBox(pos - vec3(-1.0,0.5 * (sin(iTime)+1.0)/2.0,0.0) + cp,vec3(0.5,0.5,0.5)),
            sdSphere(pos - vec3(-1.0,0.5,0.0) + cp,0.5));
    res = opU(res, vec2(b,129.8));
    
    b = opS(sdSphere(pos - vec3(-1.0,0.5,-1.0) + cp,0.5),
            udBox(pos - vec3(-1.0,0.5 * (sin(iTime))/1.0,-1.0) + cp,vec3(0.5,0.5,0.5)));
    res = opU(res, vec2(b,22.4));
    
    return res;
}

// Function 759
vec2 GetUVCentre(const vec2 vInputUV)
{
	vec2 vFontUV = vInputUV;
    vFontUV.y -= 0.35;
		
	return vFontUV;
}

// Function 760
vec3 tonemapping(vec3 color)
{
    //Tonemapping and color grading
    color = pow(color, vec3(1.5));
    color = color / (1.0 + color);
    color = pow(color, vec3(1.0 / 1.5));

    
    color = mix(color, color * color * (3.0 - 2.0 * color), vec3(1.0));
    color = pow(color, vec3(1.3, 1.20, 1.0));    

	color = clamp(color * 1.01, 0.0, 1.0);
    
    color = pow(color, vec3(0.7 / 2.2));
    return color;
}

// Function 761
void animUVW(float t){
	U=sin(t)*0.5+0.5;
	V=sin(2.*t)*0.5+0.5;
	W=sin(4.*t)*0.5+0.5;
}

// Function 762
void mainCubemap( out vec4 O, vec2 U,  vec3 C, vec3 D )
{
    ivec2 I = ivec2(U)/128;                                      // tile bi-Id
    vec3 A = abs(D);
    int  f = A.x > A.y ? A.x > A.z ? 0 : 2 : A.y > A.z ? 1 : 2,  // faceId
         s = I.x + 8*I.y,                                        // tile Id
         i = int(1023.* T( mod(U,128.) ) );                      // discretize signal
    if ( D[f] < 0. ) f += 3;                                     // full face Id.
    O = f<4 ? vec4( equal( ivec4(i), s + 64*ivec4(0,1,2,3) + 256*f )) // isolate one value within 256                                
            : vec4(T(U/R*128.),0,0,0);                           //  2 useless : free to show the image ! 
 // O = .5*vec4(  ( s + 64*ivec4(0,1,2,3) + 256*f) );    // cubeMap calibration
}

// Function 763
float map( in vec3 p )
{
    return length(p)-1.5;
}

// Function 764
float heightmap(vec2 uv) {
    float height = 0.0;
 #ifdef COMPLEX
    height += drops(uv, 32.0);
    height += drops(uv, 16.0);
    height += drops(uv, 8.0);
    height += drops(uv, 4.0);
    height += drops(uv, 2.0);
    height /= 8.0;
 #else
    height += drops(uv, 8.0);
    height += drops(uv, 4.0);
    height /= 5.0;
 #endif
    return height * intensity;
}

// Function 765
vec2 remap( in vec2 p ) {
    // flip coordinates so they're easier to work with
    // we'll flip them back before returning it
    float sig = p.x > 0.0 ? +1.0: -1.0;
    p.x = abs(p.x);
    // go to polar coordinates
    float theta = 0.0;
    float radius = 0.0;
    // do the bottom part
    float botPos = atan(p.x, -p.y) / (PI*0.5+CONE_THETA);
    if (botPos < 1.0) {
        theta = (botPos * CONE_LBOT) / CONE_L;
        radius = length(p);
    } else {
        // do the flat part
        float pos = dot(p, CONE_CSLOPE);
        float flatPos = pos / CONE_LFLAT;
        if (flatPos < 1.0) {
            theta = (CONE_LBOT + flatPos*CONE_LFLAT) / CONE_L;
            radius = dot(p, CONE_SLOPE);
        } else {
            // do the top part
            p.y -= CONE_HEIGHT;
            float topPos = (atan(p.y, p.x) - CONE_THETA) / (PI*0.5-CONE_THETA);
            theta = (CONE_LBOT + CONE_LFLAT + topPos*CONE_LTOP) / CONE_L;
            radius = length(p) + (1.0 - CONE_RADIUS);
        }
    }
    // squeeze the angle toward the top of the broccoli
    theta *= theta;
    // go back to cartesian, flip the sign and return
    theta = theta * sig * PI;
    return vec2(sin(theta), -cos(theta))*radius;
}

// Function 766
float map(float value, float min1, float max1, float min2, float max2) {
    return (value - min1) / (max1 - min1) * (max2 - min2) + min2;
}

// Function 767
float map(vec3 p) {
    float freq = SEA_FREQ;
    float amp = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    vec2 uv = p.xz; uv.x *= 0.75;
    
    float d, h = 0.0;    
    for(int i = 0; i < ITER_GEOMETRY; i++) {        
    	d = sea_octave((uv+SEA_TIME)*freq,choppy);
    	d += sea_octave((uv-SEA_TIME)*freq,choppy);
        h += d * amp;        
    	uv *= octave_m; freq *= 1.9; amp *= 0.22;
        choppy = mix(choppy,1.0,0.2);
    }
    return p.y - h;
}

// Function 768
vec4 BoxMap( sampler2D sam, in vec3 p, in vec3 n, in float k, in float LOD)
{
  vec3 m = pow( abs(n), vec3(k) );
  vec4 x = textureLod( sam, p.yz, LOD);
  vec4 y = textureLod( sam, p.zx, LOD);
  vec4 z = textureLod( sam, p.xy, LOD);
  return (x*m.x + y*m.y + z*m.z)/(m.x+m.y+m.z);
}

// Function 769
vec4 MapTerrain( vec3 p)
{       
  float boatDist= 10000.;
  float bridgeDist=10000.;
  treeDist = 10000.;
  float water=0.;
  float height = GetTerrainHeight(p); 
  float tHeight=mix(height, 4., smoothstep(12., 1.98, length(p.xz-vec3(-143, 0., 292).xz))); 
  float boulderHeight = GetBoulderHeight(p.xz, height);
  float stoneHeight = GetStoneHeight(p.xz, tHeight);
  tHeight+= mix(stoneHeight, 0., step(0.1, boulderHeight));

  tHeight= mix(tHeight-.20, tHeight*1.4, smoothstep(0.0, 0.25, tHeight));

  if (tHeight>0.)
  {
    tHeight +=textureLod( iChannel1, p.xz*.2, 0.2 ).x*.03;

    tHeight+=boulderHeight;
      
                   #ifdef TREES   
      vec3 treePos = p-vec3(0.,tHeight+2.,0.);
      vec2 mm = floor( treePos.xz/8.0 );	
	treePos.xz = mod( treePos.xz, 8.0 ) - 4.0;
      
      float treeHeight=GetTreeHeight(mm,p.xz, tHeight);
      
      if(treeHeight>0.05)
      {
          treeDist = sdEllipsoid(treePos,vec3(2.,5.7,2.));
                     treeDist+=(noise(p*1.26)*.6285);
         treeDist+=(noise(p*3.26)*.395);
           treeDist+=(noise(p*6.26)*.09825);
      }
    #endif
      
    #ifdef GRASS
      tHeight+=GetFoliageHeight(p, height, stoneHeight, boulderHeight);
    #endif

  } else
  {
    water = GetWaterWave(p);
  }

    
  #ifdef BRIDGE
    bridgeDist=MapBridge(p);    
  #endif
    #ifdef BOAT
    boatDist=MapBoat(p);
  #endif
    
    return vec4(min(treeDist,min(min(boatDist, bridgeDist), p.y -  max(tHeight, -water*0.05))), boatDist, bridgeDist, height);
}

// Function 770
float mapn(vec3 p, vec3 div) {
    #if USE_3D_VOLUME_NORMAL
    	float d = map2(p, div);
    #else
    	float d = map(p, time);
    #endif
    
    d=max(d, limit(p));
    d=smin(d, background(p), 0.2);
    return d;
}

// Function 771
vec3 PrefilterEnvMap( float Roughness,vec3 Dir ) { 
	vec3 N = Dir; 
	vec3 V = Dir;
	vec3 PrefilteredColor = vec3(0.);
    float TotalWeight = 0.;
	const uint NumSamples = 1024u; 
	for( uint i = 0u; i < NumSamples; i++ ) { 
		vec2 Xi = Hammersley( i, NumSamples ); 
		vec3 H = ImportanceSampleGGX( Xi, N, Roughness); 
		vec3 L = reflect(-V,H);//2. * dot( V, H ) * H - V;
		float NoL = dot( N, L ); 
		if( NoL > 0. ) { 
			PrefilteredColor += texture(Cubemap_Texture,L).rgb * NoL; 
			TotalWeight += NoL; 
		}
	}
	return PrefilteredColor / TotalWeight;
}

// Function 772
vec4 cubemap( sampler2D sam, in vec3 d )
{
    vec3 n = abs(d);
    vec3 s = dFdx(d);
    vec3 t = dFdy(d);
    // intersect cube
         if(n.x>n.y && n.x>n.z) {d=d.xyz;s=s.xyz;t=t.xyz;}
    else if(n.y>n.x && n.y>n.z) {d=d.yzx;s=s.yzx;t=t.yzx;}
    else                        {d=d.zxy;s=s.zxy;t=t.zxy;}
    // project into face
    vec2 q = d.yz/d.x;
    // undistort in the edges
    q *= 1.25 - 0.25*q*q;
    // sample
    // TODO: the derivatives below are wrong, apply chain rule thx
    return textureGrad( sam,  0.5*q + 0.5,
                              0.5*(s.yz-q*s.x)/d.x,
                              0.5*(t.yz-q*t.x)/d.x );
}

// Function 773
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    // Larger sample distances give a less defined bump, but can sometimes lessen the aliasing.
    const vec2 e = vec2(.001, 0); 
    
    // Gradient vector: vec3(df/dx, df/dy, df/dz);
    float ref = bumpSurf3D(p);
   
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy),
                      bumpSurf3D(p - e.yxy),
                      bumpSurf3D(p - e.yyx)) - ref)/e.x; 
    
    /*
    // Six tap version, for comparisson. No discernible visual difference, in a lot of cases.
    vec3 grad = vec3(bumpSurf3D(p - e.xyy) - bumpSurf3D(p + e.xyy),
                     bumpSurf3D(p - e.yxy) - bumpSurf3D(p + e.yxy),
                     bumpSurf3D(p - e.yyx) - bumpSurf3D(p + e.yyx))/e.x*.5;
    */ 
  
    // Adjusting the tangent vector so that it's perpendicular to the normal. It's some kind 
    // of orthogonal space fix using the Gram-Schmidt process, or something to that effect.
    grad -= nor*dot(nor, grad);          
         
    // Applying the gradient vector to the normal. Larger bump factors make things more bumpy.
    return normalize(nor + grad*bumpfactor);
	
}

// Function 774
float map(float p, float ss, float se, float ds, float de)
{
    return ds + (p-ss)*(de-ds)/(se-ss);
}

// Function 775
float map(vec3 p)
{        
    float rad = .1*sin(p.y*1.75) + ((p.y>-3.6)?smoothstep(-3.,-3.6,p.y)*0.35+2.95:2.4);
    float res = min(sdCappedCylinder(p+vec3(0.,6.,0.),vec2(rad,3.)),sdSH(p,1.));
     
    
    if(p.y<-3.4)
    {
  	  	for(int i=0;i<25;i++)
   		{
	    	float cs = cos(float(i)/12.0*PI);
   	    	float sn = sin(float(i)/12.0*PI); 

    		res = max(res,-sdCylinder(p+vec3(cs*(rad),0.,sn*(rad)),vec3(0.,.0,.2)));    
    	}
    }    
    return res;
}

// Function 776
vec4 colormap(float x) {
    return vec4(colormap_red(x), colormap_green(x), colormap_blue(x), 1.0);
}

// Function 777
vec4 map(in vec3 p)
{
    p.z += 0.2*iTime;
	vec4 d1 = fbmd( p );
    d1.x -= 0.37;
	d1.x *= 0.7;
    d1.yzw = normalize(d1.yzw);

    // clip to box
//    vec4 d2 = sdBox( p, vec3(1.5) );
//    return (d1.x>d2.x) ? d1 : d2;
    return d1;
}

// Function 778
vec3 whitePreservingLumaBasedReinhardToneMapping(vec3 color)
{
	float white = 2.;
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma * (1. + luma / (white*white)) / (1. + luma);
	color *= toneMappedLuma / luma;
	color = pow(color, vec3(1. / gamma));
	return color;
}

// Function 779
void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir){
    
    
    // UV coordinates.
    //
    // For whatever reason (which I'd love expained), the Y coordinates flip each
    // frame if I don't negate the coordinates here -- I'm assuming this is internal, 
    // a VFlip thing, or there's something I'm missing. If there are experts out there, 
    // any feedback would be welcome. :)
    vec2 uv = fract(fragCoord/iResolution.y*vec2(1, -1));
 
    // Pixel storage.
    vec3 col;
   
    // Initial conditions -- Performed upon initiation.
    if(abs(tx(iChannel0, uv).w - iResolution.y)>.001){
        
        // INITIAL CONDITIONS.
       
        // Sprinkle some hash noise into the RGB channels on the canvas. 
        // Set the wrapping values.
        
        gSc = 512.;
        col = vec3(hash21(uv), hash21(uv + 7.), hash21(uv + 9.));
        
        #if SCHEME == 0
        // Smooth value noise overlay to achieve that pattern within
        // a pattern look.
        gSc = 24.;
        col *= smoothstep(0., .01, n2D(uv*gSc) - .5)*4. + 1.;
        #endif
        
        // Another square combination.
        //gSc = 512.;
        //col = vec3(1)*hash21(floor(uv*64.)/32.);
        //col = max(col, hash21(uv));
        
        /*
        // Just to show that it works with anything, add the "London" texture to 
        // "iChannel1" and uncomment the this block... For better resolution, you
        // might also wish to switch to the SMALL_BLUR option in the "Common" tab, 
        // and take out the scrolling in the "Image" tab.
        vec3 tx = texture(iChannel1, fract(uv*2. + .5)).xyz; tx *= tx; // Rough SRGB to linear.
        #if SCHEME == 1
        tx = vec3(1)*dot(tx, vec3(.299, .587, .114))
        #endif
        col = tx;
        */
        
    }
    else {
            
       	// Formula: Initial value, minus the larger filter differences,
        // plus the smaller filter difference.
 
        vec3 val = tx(iChannel0, uv).xyz;
        
        // Larger kernel sizes give more controlls blurs, and generally thicker patterns,
        // which I like, but they tend to be slightly slower and more costly. Having said 
        // that, the larger blurs require a total of 74 (49 + 25) texel reads, which the 
        // average GPU can do with ease. The smaller blur dimensions require just 34 
        // (25 + 9) reads.
        
        // Inside a raymarching loop, it'd be just the one read. Plus, you could 
        // stop the blurring process entirely, once you're satisfied with the pattern.
        
        // Only odd sizes differing by two will work.
        #ifdef SMALL_BLUR
        const int blurDimL = 5; // Large filter dimension.
        const int blurDimS = 3; // Small filter dimension.
        #else 
        const int blurDimL = 7; // Large filter dimension.
        const int blurDimS = 5; // Small filter dimension.
        #endif
        
        // You could stop filtering at this point. For raymarched patterns, it'd be 
        // worth saving every cycle you can. I think there's a discard option as well.
        //if(iTime<15.){        
            vec3 val3 = BlurTri(iChannel0, uv, blurDimL).xyz;    
            vec3 val2 = BlurTri(iChannel0, uv, blurDimS).xyz; 
        
            // These figures require a bit of coaxing, but they're not too hard
            // to tweak. With small kernel sizes, the larger blurring speed needs
            // to slow down. Once you get them right, they'll work with virtually
            // any normal range pattern.
            #ifdef SMALL_BLUR
            col = val + (val2 - val3) + (val2 - val);
            #else
            col = val + (val2 - val3)*1.6 + (val2 - val);
            #endif
        //}
        //else col = val;
        
       
          
    }
    
    fragColor = vec4(clamp(col, -1., 1.), iResolution.y);
    
}

// Function 780
vec3 darkMap( int index,  float v ) {
    vec3[5] arr;
    
    if (index == 0)      // blue 
        arr = vec3[] ( vec3(95,205,228), vec3(99,155,255), vec3(91,110,225), vec3(48,96,130),vec3(63,63,116));
    else if (index == 1) // red 
        arr = vec3[] ( vec3(238,195,154),vec3(215,123,186),vec3(217,87,99),  vec3(172,50,50),vec3(69,40,60));
    else if (index == 2) // green 
        arr = vec3[] ( vec3(153,229,80), vec3(106,190,48), vec3(55,148,110), vec3(48,96,130),vec3(63,63,116));
    else if (index == 3) // brown
        arr = vec3[] ( vec3(217,160,102),vec3(180,123,80), vec3(143,86.,59), vec3(102,57,49),vec3(69,40,60));
    else if (index == 4) // grey
        arr = vec3[] ( vec3(155,173,183),vec3(132,126,135),vec3(105,106,106),vec3(89,86,82), vec3(50,60,57));
    else if (index == 5) // pink
        arr = vec3[] ( vec3(215,123,186),vec3(217,87,99),  vec3(118,66,138), vec3(63,63,116),vec3(50,60,57));
   
   return arr[ min(5, int(v)) ] / 255.;
}

// Function 781
float Map(in vec3 p
){
 ;float d =dd1(p)
 ;float d2=dd2(p)
 ;float d3=dd3(p)
 ;float d5=dd5(p)
 ;float d6=ShoulderButtons(p-vec3(0,-0.3,1),d,1.)
 ;float d7=dd7(p)
 ;float d8=dd8(p)
 ;float d9=dd9(p) 
 ;float d4=sdCapsule(p-vec3(0,-.39,2.07),vec3(0),vec3(0))-.11// cable
 ;d4=min(d4,Cable(p-vec3(0,-.4,2.1),vec3(0,0,-1),vec3(0,0,70),.072))
 
 ;objectID =1 
 ;if(d2<d){objectID =2;d=d2;}
 ;if(d3<d){objectID =3;d=d3;}
 ;if(d4<d){objectID =4;d=d4;}
 ;if(d5<d){objectID =5;d=d5;}
 ;if(d6<d){objectID =6;d=d6;}
 ;if(d7<d){objectID =7;d=d7;}
 ;if(d8<d){objectID =8;d=d8;}
 ;if(d9<d){objectID =9;d=d9;}


 ;if(p.y+0.75<d){ d=p.y+0.75;objectID =10;}return d;}

// Function 782
vec3 pwc_tonemap(vec3 c)
{
    c = m1 * c;
    vec3 tmp = vec3(pwc(c.r), pwc(c.g), pwc(c.b));
    
    c = m2 * tmp;
    
    return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
}

// Function 783
vec2 warpUV2(in vec2 p)
{
    
    vec2 q =mix(p, vec2(fbm(p + iTime/20.0), p.y), 0.05);
    
    return q;
}

// Function 784
float mapLeaf( in vec3 p )
{
    p -= vec3(-1.8,0.6,-0.75);
    
    p = mat3(0.671212, 0.366685, -0.644218,
            -0.479426, 0.877583,  0.000000,
             0.565354, 0.308854,  0.764842)*p;
 
    p.y += 0.2*exp(-abs(2.0*p.z) );
    
    
    float ph = 0.25*50.0*p.x - 0.25*75.0*abs(p.z);// + 1.0*sin(5.0*p.x)*sin(5.0*p.z);
    float rr = sin( ph );
    rr = rr*rr;    
    rr = rr*rr;    
    p.y += 0.005*rr;
    
    float r = clamp((p.x+2.0)/4.0,0.0,1.0);
    r = 0.0001 + r*(1.0-r)*(1.0-r)*6.0;
    
    rr = sin( ph*2.0 );
    rr = rr*rr;    
    rr *= 0.5+0.5*sin( p.x*12.0 );

    float ri = 0.035*rr;
    
    float d = sdEllipsoid( p, vec3(0.0), vec3(2.0,0.25*r,r+ri) );

    float d2 = p.y-0.02;
    d = smax( d, -d2, 0.02 );
    
    return d;
}

// Function 785
float2 Map ( in vec3 O ) {
  float3 o=O, ori=O;
  float2 res = float2(999.0);

  // opRotate(O.xz, t*0.05);
  O += vec3(0.0, 0.0, 1.5);
  float blah = -log(exp(-4.0*sdHexagonCircumcircle(O, vec2(0.5, 1.0))) +
                    exp(-1.0*sdSphere(O, 1.0)));;
  blah += dot(texture(iChannel0, O.xz),
              texture(iChannel0, O.zy))*0.05;
  Union(res, blah*0.25, 1.0);
  blah = sdSphere(O - vec3(1.8, 0.2, 2.5), 1.5);
  Union(res, blah*0.5, 1.5);
  opRotate(O.xy, PI/2.0 - PI/4.0);
  blah = sdBox(O - vec3(0.0, -0.8, 2.5),
                   float3(1.25, 0.5, 0.8));
  blah += length(texture(iChannel1, O.xz))*0.1;
  Union(res, blah*0.5, 2.0);

  O=ori;
  blah = sdSphere(O-vec3(0.0, 1.0, 0.0), 1.0);
  //Union(res, sdShell(blah, 0.005)*0.5, 4.0);
  float s = sdShell(sdSphere(O, 10.0), 0.01)*0.5;
  Union(res, s, 7.0);
  O=ori;
  float2 uv = float2(O.x, O.z);
  O.y += length(texture(iChannel0, uv*0.1).xyz)*0.05;
  float d = sdPlane(O, float3(0.0, 1.0, 0.0), 1.0);
  Union(res, d, 7.5);
    
 // O=ori;
  O.y += sin(O.x*2.0)*0.2f + cos((25.32f+O.x)*42.0f)*0.01f;
  Union(res, sdPlane(O, vec3(0.0, 1.0, 0.0), 0.5), 7.7);
    
  //-----------light--------
  for ( int i = 0; i != LIGHTS_LEN; ++ i ) {
    O = inverse(Look_At(lights[i].N))*(ori-lights[i].ori);
    Union(res, sdBox(O, float3(0.01, lights[i].radius)), 100.0+float(i));
  }
  //----
  return res;
}

// Function 786
float map( in vec3 pos )
{
    float d1 = shapeBall(pos);
    float d2 = shapeTable(pos);
    float d3 = shapeSupport(pos);
   
    return min( d1, min(d2, d3) );
}

// Function 787
vec3 envMap(vec3 rd, vec3 n){
    
    return tex3D(iChannel0, rd, n);
}

// Function 788
float llamelMapSMin(const in float a,const in float b,const in float k){
    float h=clamp(0.5+0.5*(b-a)/k,0.0,1.0);return b+h*(a-b-k+k*h);
}

// Function 789
float heightmap(vec2 uv)
{
    return (texture(iChannel0, uv.yx / 256.).r - .1) * 2.;
}

// Function 790
float map(vec3 p){
    	p = abs(mod(p, 3.) - 1.5); // Repeat space.
    	return min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1.; // Void Cube.
	}

// Function 791
vec2 toUV( vec2 p )
{
    vec2 uv = -1.1 + 2.2 * p / iResolution.xy;
    uv.x *= iResolution.x / iResolution.y;
    return uv;
}

// Function 792
vec2 maptouv(vec3 p, vec3 div) {
    p=clamp((p/3.0)*0.5+0.5,0.0,0.999);
    float idz=floor((p.z+0.00001)*div.x*div.y)/div.x;
    vec2 uvz=vec2(fract(idz),floor(idz)/div.y);
    return p.xy/div.xy+uvz;
}

// Function 793
float seaMapHigh(const in vec3 p) {
    float freq = SEA_FREQ;
    float amp = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    vec2 uv = p.xz; uv.x *= 0.75;
    
    float d, h = 0.0;    
    for(int i = 0; i < SEA_ITER_FRAGMENT; i++) {        
    	d = seaOctave((uv+SEA_TIME)*freq,choppy);
    	d += seaOctave((uv-SEA_TIME)*freq,choppy);
        h += d * amp;        
    	uv *= sea_octave_m; freq *= 1.9; amp *= 0.22;
        choppy = mix(choppy,1.0,0.2);
    }
    return p.y - h;
}

// Function 794
vec3 sphericalToCubemap(in vec2 sph) {
    return vec3(sin(sph.y) * sin(sph.x), cos(sph.y), sin(sph.y) * cos(sph.x));
}

// Function 795
float map(vec3 p){
    
    
    p = mod(p, 4.0) - 2.0;
    float bmp = sin(p.x*12.5)*sin(p.y*11.5)*sin(p.z*14.);
    
    bmp *= .045;
    float c = length(p) - 1.0;
    return c+bmp;
}

// Function 796
vec4 getBitmapColor( in vec2 uv )
{
	return getColorFromPalette( getPaletteIndex( uv ) );
}

// Function 797
float map(vec3 p) {

    #ifdef DEBUG
        if (p.x < 0.) {
            hitDebugTorus = true;
            return fTorus(p.xzy, 1., 1.4145);
        }
    #endif

    float k;
    vec4 p4 = inverseStereographic(p,k);

    // The inside-out rotation puts the torus at a different
    // orientation, so rotate to point it at back in the same
    // direction
    pR(p4.zy, time * -PI / 2.);

    // Rotate in 4D, turning the torus inside-out
    pR(p4.xw, time * -PI / 2.);

    vec2 uv;
    float d = fTorus(p4, uv);
    modelUv = uv;

    #ifdef DEBUG
        d = fixDistance(d, k);
        return d;
    #endif

    // Recreate domain to be wrapped around the torus surface
    // xy = surface / face, z = depth / distance
    float uvScale = 2.25; // Magic number that makes xy distances the same scale as z distances
    p = vec3(uv * uvScale, d);

    // Draw some repeated circles

    float n = 10.;
    float repeat = uvScale / n;

    p.xy += repeat / 2.;
    pMod2(p.xy, vec2(repeat));

    d = length(p.xy) - repeat * .4;
    d = smax(d, abs(p.z) - .013, .01);

    d = fixDistance(d, k);

    return d;
}

// Function 798
float map( in vec3 p ) {
    p.xz += PILLAR_SPACING *.5;
    float d = p.y;
    
    vec2 pm = mod( p.xz + vec2(PILLAR_SPACING*.5), 
                  		  vec2(PILLAR_SPACING) ) - vec2(PILLAR_SPACING*.5);
    d = min(d, max(abs(pm.x) - PILLAR_WIDTH_HALF, abs(pm.y) - PILLAR_WIDTH_HALF));
    
    vec2 cm = mod( p.xz,  vec2(PILLAR_SPACING) ) - vec2(PILLAR_SPACING*.5);
    
    d = min( d, CEILING_HEIGHT - p.y );
    d = max( d, -PILLAR_WIDTH_HALF+PILLAR_SPACING*.5-
            length( vec2(p.y-CEILING_HEIGHT, min(abs(cm.x),abs(cm.y)))));
    return d;
}

// Function 799
float TonemapCompressRangeNorm( float x )
{
	return 1.0f - exp( -x );
}

// Function 800
vec3 mapD0(float t)
{
    return 0.25 + a*cos(t+m)*(b+c*cos(t*7.0+n));
}

// Function 801
float map_bulb(vec3 pos)
{
    pos.y*=0.6;
    return length(pos) - 0.65;
}

// Function 802
vec2 map(vec3 p) {
    float bumps = fbm(p * 8.0) * 0.02;
    vec2 d1 = sdDonut(p) - bumps;
    vec2 d2 = sdCream(p);
    vec2 d3 = sdSprinkles(p);
    vec2 d4 = vec2(p.y + 1.7, 3.5);
    
    return min2(d1, min2(d2, min2(d3, d4)));
}

// Function 803
vec3   xyzToLuv(float x, float y, float z) {return   xyzToLuv( vec3(x,y,z) );}

// Function 804
float map (vec3 pos) {
  float chilly = noise(pos * 2.);
  float salty = fbm(pos*20.);
  
  pos.z -= salty*.04;
  salty = smoothstep(.3, 1., salty);
  pos.z += salty*.04;
  pos.xy -= (chilly*2.-1.) * .2;
    
  vec3 p = pos;
  vec2 cell = vec2(1., .5);
  vec2 id = floor(p.xz/cell);
  p.xy *= rot(id.y * .5);
  p.y += sin(p.x + .5);
  p.xz = repeat(p.xz, cell);
    
  vec3 pp = p;
  moda(p.yz, 5.0);
  p.y -= .1;
  float scene = length(p.yz)-.02;
    
  vec3 ppp = pos;
  pp.xz *= rot(pp.y * 5.);
  ppp = repeat(ppp, .1);
  moda(pp.xz, 3.0);
  pp.x -= .04 + .02*sin(pp.y*5.);
  scene = smoothmin(length(pp.xz)-.01, scene, .2);

  p = pos;
  p.xy *= rot(-p.z);
  moda(p.xy, 8.0);
  p.x -= .7;
  p.xy *= rot(p.z*8.);
  p.xy = abs(p.xy)-.02;
  scene = smoothmin(scene, length(p.xy)-.005, .2);

  return scene;
}

// Function 805
vec2 GetUV(vec2 s, vec2 h, float z) 
{
	return z * (h+h-s)/s.y; // central uv
}

// Function 806
float map_lamps(vec3 pos)
{
    float tc = tunnel_curve(pos.z);
    float dc = dev_tunnel_curve(pos.z);
    pos.x-= tc;
    float zz = pos.z;
    pos.z = 0.;
    float a = atan(dc);
    pos.xz = rotateVec (pos.xz, a);
    pos.z = zz;
    
    pos.y-= tunnel_curve_y(pos.z);
    lmppos = pos;
    a = atan(pos.x, pos.y);
    float tsf2 = tsf/(2.*pi);
    pos.z+= tsf2;
    ltr = 0.9;
    ltr+= 2.*(1. - smoothstep(0.6, 0.65, a)*smoothstep(0.95, 0.9, a))*
         (1. - smoothstep(-0.6, -0.65, a)*smoothstep(-0.95, -0.9, a));
    
    ltr+= 0.3*(1. - smoothstep(0.42, 0.58, abs(pos.z - floor(pos.z*tsf2 + 0.5)/tsf2)));
    float df = -length(pos.xy) + tubeRadius*ltr;
    return df;
}

// Function 807
vec4   xyzToLuv(float x, float y, float z, float a) {return   xyzToLuv( vec4(x,y,z,a) );}

// Function 808
vec2 AsciiToUV(int ascii)
{
	return vec2(ascii & 0x0F,int(FONT_COLUMNS-1.0) - (ascii >> 4))/FONT_COLUMNS;   
}

// Function 809
vec4 stateUV(ivec2 id,vec4 data,vec4 info,bool changeOrigins,bool dragActive,bool display3D) {
    vec4 O = vec4(0);
    if (iTime < 5. || data.x == 0.0) {
        O.zw = (vec2(id)-vec2(1,0))/vec2(GRID);
        O.xy = -((vec2(id)-vec2(1,0))/vec2(GRID)*(R-40.)+20.)/R;
        if(O.x == 0.0 && O.y == 0.0) {
            O.x = -0.000001;
        }
        if(id.x>1&&id.x<4&&id.y>1&&id.y<4) O-=vec4(0,0,-0.06,0.036)+.04*(vec2(id)-vec2(2.5,2.5)).xyxy;
        if(id.x>3&&id.x<6&&id.y>1&&id.y<4) O-=vec4(0,0,-0.04,0.035)+.06*(vec2(id)-vec2(4.5,2.5)).xyxy;
        return O;
    }
    if(!changeOrigins && display3D) {
        O = vec4(data);
        return O;
    }
    if(!dragActive) {
        O = vec4(data);
        return O;
    }
    if(id.x != int(info.x) || id.y != int(info.y)) {
        O = vec4(data);
        return O;
    }
    if(changeOrigins) {
        if(display3D) {
            O = vec4(-abs(data.xy), clamp((iMouse.xy-20.-0.125*R)/(R-40.-0.25*R),-0.1875,1.1875));
        } else {
            O = vec4(-abs(data.xy), clamp((iMouse.xy-20.)/(R-40.),0.,1.));
        }
        if(O.x == 0.0 && O.y == 0.0) {
            O.x = -0.000001;
        }
        return O;
    }
    O = vec4(iMouse.xy/R, data.zw);
    if(O.x == 0.0 && O.y == 0.0) {
        O.x = 0.000001;
    }
   
    return O;
}

// Function 810
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

// Function 811
vec3 mapH( in vec2 pos )
{
	vec2 fpos = fract( pos ); 
	vec2 ipos = floor( pos );
	
    float f = 0.0;	
	float id = hash( ipos.x + ipos.y*57.0 );
	f += freqs[0] * clamp(1.0 - abs(id-0.20)/0.30, 0.0, 1.0 );
	f += freqs[1] * clamp(1.0 - abs(id-0.40)/0.30, 0.0, 1.0 );
	f += freqs[2] * clamp(1.0 - abs(id-0.60)/0.30, 0.0, 1.0 );
	f += freqs[3] * clamp(1.0 - abs(id-0.80)/0.30, 0.0, 1.0 );

    f = pow( clamp( f, 0.0, 1.0 ), 2.0 );
    float h = 2.5*f;

    return vec3( h, id, f );
}

// Function 812
float map(vec3 p, float t) {
    float globalScale = 1.0;
    p/=globalScale;
    p+=vec3(0,0,0);
    
#if SIMPLE_HEAD
    
 vec3 TPos_0=mirror(p-vec3(0, 0, 0));

  float res = smin(smin(smin(smin(smin(smin(smin(conecaps(TPos_0, vec3(0, 0, 0), vec3(0, 0.8981191, -0.2021628), 0.65, 0.8194701),
 caps(TPos_0, vec3(0, 0.885, -0.359), vec3(0, 0.6657753, -0.8952476), 0.90), 0.3),
 sph(TPos_0-vec3(0, 1.058, -0.7920001), 0.86), 0.3),
 conecaps(TPos_0, vec3(0, -1.343, -1.035), vec3(0, -0.01590037, -1.035), 0.65, 0.4129604), 0.3),
 conecaps(TPos_0, vec3(0.448, 0.493, 0.413), vec3(0.0964303, 0.4929999, 0.6215981), 0.25, 0.07000001), 0.3),
 conecaps(TPos_0, vec3(0, 0.731, 0.899), vec3(0, 0.9994185, 0.6862392), 0.08, -0.04511364), 0.3),
 conecaps(TPos_0, vec3(0, -0.388, 0.524), vec3(0, 0.2982672, 0.6916121), 0.36, 0.1544515), 0.3),
 conecaps(TPos_0, vec3(0, -1.4863, -0.585), vec3(0, -0.4297584, -0.1084773), 0.28, 0.1755086), 0.3);
    
#else
 vec3 TPos_0=mirror(p-vec3(0, 0, 0));

  float res = min(smax(smin(sph(p-vec3(-0.014, 0.87, -0.641), 1.20),
 sph(p-vec3(0.031, 1.05, -0.291), 0.89), 0.2),
 smin(smin(smin(conecaps(p, vec3(0.06, 1.785, -0.043), vec3(-0.06637037, 1.56574, 0.3384626), 0.15, 0.04923537),
 conecaps(p, vec3(0.197, 1.788, 0.109), vec3(0.4121402, 1.385284, 0.392923), 0.12, 0.03994529), 0.1),
 conecaps(p, vec3(-0.147, 1.735, 0.115), vec3(-0.5633237, 1.244842, 0.6771756), 0.12, -0.01053677), 0.1),
 conecaps(p, vec3(0.373, 1.719, 0.078), vec3(0.7583238, 1.083739, 0.3566932), 0.11, 0.002208568), 0.1), 0.1),
 min(smax(smin(sph(TPos_0-vec3(-0.014, 0.87, -0.641), 1.20),
 sph(TPos_0-vec3(0.031, 0.237, -1.288), 0.68), 0.2),
 smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(conecaps(TPos_0, vec3(0.18, 1.84, 0), vec3(0.9947872, 1.059192, 0), 0.20, 0.080608),
 conecaps(TPos_0, vec3(0.457, 1.778, -0.233), vec3(1.218826, 0.6229788, -0.233), 0.20, 0.06851683), 0.1),
 conecaps(TPos_0, vec3(0.475, 1.751, -0.502), vec3(1.163649, 0.834214, -0.5103351), 0.20, 0.06247119), 0.1),
 conecaps(TPos_0, vec3(0.501, 1.644, -0.687), vec3(1.324742, 0.4498808, -0.815752), 0.20, 0.0), 0.1),
 conecaps(TPos_0, vec3(0.438, 1.681, -0.818), vec3(1.25843, 0.5285492, -1.07989), 0.20, 0.05239521), 0.1),
 conecaps(TPos_0, vec3(0.4150705, 1.692155, -0.892), vec3(0.977915, 0.8010392, -1.295309), 0.20, 0.080608), 0.1),
 conecaps(TPos_0, vec3(0.4665176, 1.612289, -1.137529), vec3(0.9487399, 0.3972942, -1.591065), 0.20, 0.06851687), 0.1),
 conecaps(TPos_0, vec3(0.3073766, 1.580032, -1.345637), vec3(0.6488477, 0.554816, -1.729217), 0.20, 0.0624712), 0.1),
 conecaps(TPos_0, vec3(0.2050478, 1.472336, -1.527339), vec3(0.5545081, 0.120172, -1.940346), 0.20, 0.0), 0.1),
 conecaps(TPos_0, vec3(0.1044063, 1.466208, -1.451), vec3(0.2097557, 0.2754359, -2.251505), 0.20, 0.05239522), 0.1),
 conecaps(TPos_0, vec3(0.87, 0.82, -0.9), vec3(0.4488406, -0.6502159, -0.9097424), 0.20, -0.03828883), 0.1),
 conecaps(TPos_0, vec3(0.828, 0.657, -1.139), vec3(0.3844372, -0.6142089, -1.146863), 0.20, -0.0463496), 0.1),
 conecaps(TPos_0, vec3(0.873, 0.646, -1.581), vec3(0.1269265, -0.3829435, -1.104255), 0.20, 0.09471444), 0.1),
 conecaps(TPos_0, vec3(0.441, 0.689, -1.627), vec3(0.3258284, -0.4353262, -1.433523), 0.20, 0.06247126), 0.1),
 conecaps(TPos_0, vec3(0.258, 0.693, -1.761), vec3(0.02478147, -0.7067026, -1.433148), 0.20, 0.0), 0.1),
 conecaps(TPos_0, vec3(0, 0.874, -1.885), vec3(0, -0.5986498, -1.709854), 0.20, 0.008060798), 0.1),
 conecaps(TPos_0, vec3(0.842, 0.572, -0.745), vec3(0.6800996, -0.4511317, -0.438101), 0.14, -0.0270476), 0.1),
 conecaps(TPos_0, vec3(0.208, 1.842, -0.244), vec3(0.583753, 1.851966, -1.308072), 0.20, 0.080608), 0.1),
 conecaps(TPos_0, vec3(0.098, 1.844, -0.505), vec3(0.3196597, 1.71843, -1.604381), 0.20, 0.080608), 0.1),
 conecaps(TPos_0, vec3(0, 1.844, -0.84), vec3(0, 1.584119, -1.938181), 0.20, 0.08060801), 0.1), 0.1),
 smax(smin(smax(smin(smin(smin(smin(smin(smin(smin(smin(smin(conecaps(TPos_0, vec3(0, 0, 0), vec3(0, 0.8981191, -0.2021628), 0.65, 0.8194701),
 caps(TPos_0, vec3(0, 0.885, -0.359), vec3(0, 0.6657753, -0.8952476), 0.90), 0.3),
 sph(TPos_0-vec3(0, 1.058, -0.7920001), 0.86), 0.3),
 conecaps(TPos_0, vec3(0, -1.343, -1.035), vec3(0, -0.01590037, -1.035), 0.65, 0.4129604), 0.3),
 conecaps(TPos_0, vec3(0.448, 0.493, 0.413), vec3(0.0964303, 0.4929999, 0.6215981), 0.25, 0.07000001), 0.3),
 conecaps(TPos_0, vec3(0, 0.731, 0.899), vec3(0, 0.9994185, 0.6862392), 0.08, -0.04511364), 0.3),
 conecaps(TPos_0, vec3(0, -0.388, 0.524), vec3(0, 0.2982672, 0.6916121), 0.36, 0.1544515), 0.3),
 conecaps(TPos_0, vec3(0.793, 0.575, -0.45), vec3(0.9685718, 0.8660991, -0.5075423), 0.18, 0.07538141), 0.3),
 conecaps(TPos_0, vec3(0.651, -1.4863, -0.585), vec3(0.4176198, -0.006287336, -1.020103), 0.28, -0.2256553), 0.3),
 conecaps(TPos_0, vec3(0, -1.4863, -0.585), vec3(0, -0.4297584, -0.1084773), 0.28, 0.1755086), 0.3),
 -min(min(min(min(conecaps(TPos_0, vec3(0.39, 1.001, 0.687), vec3(0.760278, 1.001, 0.4733404), 0.25, 0.0725),
 conecaps(TPos_0, vec3(0.063, -0.031, 0.736), vec3(0.3386817, -0.03100006, 0.5769249), 0.19, 0.05397813)),
 conecaps(TPos_0, vec3(-0.014, -0.095, 0.604), vec3(0.2963244, -0.09500001, 0.4122402), 0.24, 0.06858408)),
 caps(TPos_0, vec3(0.149, -0.406, 0.994), vec3(0.01727781, -0.406, 0.994), 0.13)),
 conecaps(TPos_0, vec3(0.986, 0.769, -0.405), vec3(0.9680421, 0.520758, -0.405), 0.15, 0.04220953)), 0.1),
 smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smin(smax(caps(TPos_0, vec3(0.143, 1.041, 0.541), vec3(0.4671654, 1.005373, 0.3770713), 0.07),
 -conecaps(TPos_0, vec3(0.274, 0.946, 0.563), vec3(0.4998161, 0.9211818, 0.448806), 0.10, 0.06821328), 0.02),
 smax(smax(caps(TPos_0, vec3(0.175, 0.877, 0.58), vec3(0.5016567, 0.8574979, 0.4183209), 0.07),
 -conecaps(TPos_0, vec3(0.239, 0.967, 0.568), vec3(0.5910456, 1.022967, 0.4102298), 0.07, 0.04818002), 0.04),
 -conecaps(TPos_0, vec3(0.408, 0.945, 0.483), vec3(0.06001997, 0.9721231, 0.6565917), 0.07, 0.04818002), 0.04), 0.05),
 conecaps(TPos_0, vec3(0.41, 0.757, 0.531), vec3(0.07579309, 0.8708386, 0.6575592), 0.08, -0.02930647), 0.05),
 conecaps(TPos_0, vec3(0.855, 0.689, -0.37), vec3(0.837042, 0.4407571, -0.37), 0.15, 0.04220968), 0.05),
 smin(sph(TPos_0-vec3(0, 0.718, 0.877), 0.10),
 conecaps(TPos_0, vec3(0.1, 0.677, 0.783), vec3(-0.05059213, 0.9789532, 0.6192493), 0.08, -0.02930725), 0.05), 0.05),
 conecaps(TPos_0, vec3(0.051, -0.267, 0.751), vec3(0.0411847, -0.1742215, 0.7390867), 0.06, 0.02549449), 0.05),
 conecaps(TPos_0, vec3(0.13, -0.257, 0.691), vec3(0.1219919, -0.1634887, 0.6848335), 0.06, 0.025494), 0.05),
 conecaps(TPos_0, vec3(0.198, -0.22, 0.608), vec3(0.1968478, -0.1176157, 0.6248499), 0.06, 0.015782), 0.05),
 conecaps(TPos_0, vec3(0.04392957, 0.1199243, 0.7371331), vec3(0.04759543, 0.03143565, 0.7688017), 0.06, 0.02549401), 0.05),
 conecaps(TPos_0, vec3(0.094312, 0.12868, 0.65904), vec3(0.1271792, 0.01363337, 0.6877561), 0.06, 0.02549402), 0.05),
 conecaps(TPos_0, vec3(0.1845542, 0.105525, 0.5969098), vec3(0.1959237, 0.002555117, 0.6028855), 0.06, 0.01578201), 0.05),
 smax(sph(TPos_0-vec3(0.28, 0.937, 0.422), 0.12),
 -sph(TPos_0-vec3(0.391, 0.937, 0.672), 0.17), 0.03), 0.05), 0.05),
 -caps(TPos_0, vec3(0.092, 0.6, 0.822), vec3(0.03096008, 0.7223915, 0.7556266), 0.02), 0.05)));
#endif
    //res = min(res, pln(p-vec3(0,-3,0), vec3(0,1,0)));
    
    return res*globalScale;
}

// Function 813
vec3 uvToRayDir( vec2 uv ) {
    vec2 v = PI * ( vec2( 1.5, 1.0 ) - vec2( 2.0, 1.0 ) * uv );
    return vec3(
        sin( v.y ) * cos( v.x ),
        cos( v.y ),
        sin( v.y ) * sin( v.x )
    );
}

// Function 814
float MapFlyingMissile( vec3 p, Missile missile)
{
  TranslateMissilePos(p, missile);  
  // map missile flame
  eFlameDist = min(eFlameDist, sdEllipsoid( p+ vec3(0., 0., 2.2+cos(iTime*90.0)*0.23), vec3(.17, 0.17, 1.0)));
  // map missile 
  return min(MapMissile(p, missile), eFlameDist);
}

// Function 815
float map(vec3 p)
{	
	if (scene_idx>9&&scene_idx<14) 
    {
		float a = sdBox(p,vec3(1.2,1.5,2.))-0.1;
    	pR(p.yz,0.2*time);
    	return max(a,-sdBox(p,vec3(0.9,0.5,.1))+0.04);
    }
    float displace = 0.001*noise(15.*p+time); 
    if (scene_idx == 8 || scene_idx == 9) displace = 0.005*noise(10.*p+time)+0.005*sin(10.*p.x+3.*time)+0.002*sin(14.*p.y+4.*time); //!!!!!!!!!!!!!!!
    if (scene_idx < 4 || (scene_idx > 5 && scene_idx < 10)) return sdBox(p,vec3(1.,1.,1.))-0.4+displace; // box ///////////// vec3 1.1.1
    return length(p)-2.0+0.5*noise(1.5*p-0.02*time);     
}

// Function 816
float map2( in vec3 p )
{
	vec3 q = p*cloudScale;
	float f;
    f  = .50000*noise( q ); q = q*2.02;
    f += .25000*noise( q );
	return clamp( 1.5 - p.y - 2. + 1.75*f, 0., 1. );
}

// Function 817
vec4 hpluvToRgb(float x, float y, float z, float a) {return hpluvToRgb( vec4(x,y,z,a) );}

// Function 818
int map(ivec4 c, int i)
{
	if (i == 0) return c.x;
	if (i == 1) return c.y;
	if (i == 2) return c.z;
	if (i == 3) return c.w;
    return 0;
}

// Function 819
vec4 texMapCh(samplerCube tx, vec3 p){
    
    p *= dims;
    int ch = (int(p.x*4.)&3);
    p = mod(floor(p), dims);
    float offset = dot(p, vec3(1, dims.x, dims.x*dims.y));
    vec2 uv = mod(floor(offset/vec2(1, cubemapRes.x)), cubemapRes);
    // It's important to snap to the pixel centers. The people complaining about
    // seam line problems are probably not doing this.
    uv = fract((uv + .5)/cubemapRes) - .5;
    return vec4(1)*texture(tx, vec3(-.5, uv.yx))[ch];
    
}

// Function 820
vec2 map(vec2 uv) {
  
	float t = floor(mod(speed,10.));
  
	//t = 7.;
	float s = t == 1.? 4.  : t==5.? .6: t==4.? 6. : t==5.? 2.5 : t== 9. ? 13. : 3.;  
   
    uv *= s + s*.2*cos(fract(speed)*6.2830);
    
    vec2 fz, z = toPolar(uv); 
    	
    	 // z + 1 / z - 1
	fz = t == 0. ? zdiv(zadd(z,vec2(1.0)),zsub(z,vec2(1.0,.0)) ) :
         // formula from wikipedia https://en.m.wikipedia.org/wiki/Complex_analysis
		 // fz = (zˆ2 - 1)(z + (2-i))ˆ2 / zˆ2 + (2+2i)
		 t == 1. ? zdiv(zmul(zsub(zpow(z,2.),vec2(1.,0)),zpow(zadd(z,toPolar(vec2(2.,-1.))),2.)),zadd(zpow(z,2.),toPolar(vec2(2.,-2.)))) :
		 // z^(3-i) + 1.
		 t == 2. ? zadd(zpow(z,vec2(3.,acos(-1.))),vec2(1.,.0)) :
		 // tan(z^3) / z^2
		 t == 3. ? zdiv(ztan(zpow(z,3.)),zpow(z,2.)) :
		 // tan ( sin (z) )
		 t == 4. ? ztan(zsin(z)) :
		 // sin ( 1 / z )
		 t == 5. ? zsin(zdiv(vec2(1.,.0),z)) :
		 // the usual coloring methods for the mandelbrot show the outside. 
		 // this technique allows to see the structure of the inside.
		 t == 6. ? mandelbrot(zsub(z,vec2(1.,.0))) : 
         // the julia set 
		 t == 7. ? julia(z) :
		 //https://en.m.wikipedia.org/wiki/Lambert_series
		 t == 8. ? lambert(z) :
		 // this is the Riemman Zeta Function (well, at least part of it... :P)
		 // if you can prove that all the zeros of this function are 
    	 // in the 0.5 + iy line, you will win:
		 // a) a million dollars ! (no, really...)
		 // b) eternal fame and your name will be worshiped in history books
		 // c) you will uncover the deep and misterious connection between PI and the primes
		 // https://en.m.wikipedia.org/wiki/Riemann_hypothesis
    	 // https://www.youtube.com/watch?v=rGo2hsoJSbo
         zeta(zadd(z,vec2(8.,.0)));
   
 
	return toCarte(fz);  
}

// Function 821
float map( vec3 pos ){
    
    
    
    vec3 p2 = vec3(1.0*cos(-0.3*time),1.4*sin(time)*cos(-0.4*time),1.7*sin(time)*sin(-0.3*time));
    float d2 = sdSphere( pos-p2, 0.2);
    vec3 p3 = vec3(1.4*sin(0.3*time),0.2,-1.6*sin(0.3*time+0.3));
    float d3 = sdSphere( pos-p3, 0.2);
    vec3 p4 = vec3(0);
    float d4 = sdSphere( pos-p4, 0.2);
    float d00 = sdUnion_s(d2,d3,0.2);
    
    float d0 = sdUnion_s(d00,d4,0.2);

    vec3 pol = carToPol(pos);
    
    float d1 = sdSphere( pos, 1.0 );
    float wave = 0.07*sin(20.*(pol.y));
    d1 = opOnion(d1+wave, 0.001);
    
    return sdUnion_s(d1,d0,0.3);
    
}

// Function 822
bool rayQuadUV( vec3 ro, vec3 rd, vec3 po, vec3 pn, vec2 psz, out vec2 uv, out float rt )
{
    rt = rayPlane( ro, rd, po, pn );
    if( !(rt > 0.) ) return false; // NaN caught here!
    vec3 pos = ro + rt * rd;
    float x = dot(pos - po, l2w( vec3(1.,0.,0.), pn ) );
    float y = dot(pos - po, l2w( vec3(0.,0.,1.), pn ) );
    uv = vec2(x,y)/psz;
    if( abs(uv.x) >= .5001 || abs(uv.y) >= .5001 ) return false;
    uv += .5; 
    return true;
}

// Function 823
float simpleBitmap(float data, vec2 s, vec2 bitCoord) {
    // 0..1.0
    float x = floor(bitCoord.x * s.x);
    float y = floor(bitCoord.y * s.y);
     
    float i = y * s.x + x;
    
    float datum = float(data) / pow(2.0, i);

    datum = mod(datum, 2.0);
        
    return floor(datum);
}

// Function 824
float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

// Function 825
float heightMapWaterDetailed(vec3 p, float t)
{
    float h = 0.0;
    vec3 op = p;
    #ifdef FLOOD
    float w = (-p.z+sin(TIME*WAVES_SPEED))*FLOOD_AMP;
    #endif
    float a = WATER_AMP;
    float f = WATER_FREQ;
    float T = TIME(t)*WATER_SPEED;
    float R = hash1D(t);
    mat2 M = mat2(cos(R), -sin(R), sin(R), cos(R));
    vec2 W = vec2(T, T);
    //h = a*(-1.0+fbm2Dsimple(p.xz*f+T)+fbm2Dsimple(p.xz*f-T));
    for(int i = 0;i < 4; ++i)
    {
//     e((−2πikn)/N   )
        float ffta = 1.0;//exp((-2.0*3.14*float(i))/7.0);
        h += a*abs(sin(fbm2Dsimple(f*p.xz+W)-0.5)*3.14);
        a*= 0.8;
        f *= 1.2;
        W = mat2(0.2, -0.8, 0.8, 0.2)*W;
    }
    //for(int i=0;i<5;++i) {
    //}
    #ifdef WAVES
    h+= wave(op,
             mix(0.05, 0.9, min(1.0, max(0.0,p.z)/3.2)),
             T*5.0)
        *gaussianNoise(op.xz*0.1+T)*(0.8/p.y);
    #endif
    #ifdef FLOOD
    return h+w;
    #else
    return h;
    #endif   
}

// Function 826
float map_support(vec3 pos)
{
   float support = sdRoundBox(pos + vec3(0., -supportSize.y, 0.), supportSize, supportRR);
    
   float b = 0.05;
   float vh0 = smoothstep(0.2, 0.2 + b, pos.x)*smoothstep(0.78*supportSize.x + b, 0.78*supportSize.x, pos.x);
   vh0*= smoothstep(-0.7*supportSize.y - b, -0.7*supportSize.y, pos.y - supportSize.y)*smoothstep(0.7*supportSize.y + b, 0.7*supportSize.y, pos.y - supportSize.y);
   float vh = pos.z>0.?.0:vh0*smoothstep(0., 0.8, fract(-4.9*pos.y/supportSize.y + 0.1))*smoothstep(1., 0.8, fract(-4.9*pos.y/supportSize.y + 0.1)); 
   //if (support<0.001);
   //   support+= 0.001*(pos.z<0.?1. - vh0:1.)*noise(300.*pos);
      
   if (pos.z<0.)
     support-= 0.015*vh;
    
   support = max(support, -map_screen(pos) + 0.003);
    
   return support;
}

// Function 827
float map_stick(vec3 pos)
{
   pos = rotateVec2(pos);
   pos.xy = rotateVec(pos.xy, 0.33); 

   return sdCylinder(pos, vec2(0.09, 1.5));   
}

// Function 828
ComplexMatrix2 M_mapTripleToTriple(
    Complex a, Complex b, Complex c, 
	Complex p, Complex q, Complex r)
{
	return M_multiply(M_inverse(M_mapTripleTo01I(p, q, r)), M_mapTripleTo01I(a, b, c));
}

// Function 829
vec3 lchToLuv(vec3 tuple) {  float hrad = radians(tuple.b);  return vec3(   tuple.r,   cos(hrad) * tuple.g,   sin(hrad) * tuple.g  ); }

// Function 830
vec3 mapGrass( in vec3 pos, float h, in vec3 cur )
{
    vec3 res = cur;
    
    float db = pos.y-2.6;
    
    if( db<cur.x && pos.z>-1.65 )
    {
        const float gf = 4.0;

        vec3 qos = pos * gf;

        vec2 n = floor( qos.xz );
        vec2 f = fract( qos.xz );
        for( int j=-2; j<=2; j++ )
        for( int i=-2; i<=2; i++ )
        {
            vec2  g = vec2( float(i), float(j) );

            vec2 ra2 = hash2( n + g + vec2(31.0,57.0) );

            if( ra2.x<0.73 ) continue;

            vec2  o = hash2( n + g );
            vec2  r = g - f + o;
            vec2 ra = hash2( n + g + vec2(11.0,37.0) );

            float gh = 2.0*(0.3+0.7*ra.x);

            float rosy = qos.y - h*gf;

            r.xy = reflect( r.xy, normalize(-1.0+2.0*ra) );
            r.x -= 0.03*rosy*rosy;

            r.x *= 4.0;

            float mo = 0.1*sin( 2.0*iTime + 20.0*ra.y )*(0.2+0.8*ra.x);
            vec2 se = sdLineOri( vec3(r.x,rosy,r.y), vec3(4.0 + mo,gh*gf,mo) );

            float gr = 0.3*sqrt(1.0-0.99*se.y);
            float d = se.x - gr;
            d /= 4.0;

            d /= gf;
            if( d<res.x )
            {
                res.x = d;
                res.y = MAT_GRASS;
                res.z = r.y;
            }
        }
    }
    
    return res;
}

// Function 831
float map_floornumber(vec3 pos)
{
    float flnw = (floornr<9.?flnum_height/3.07:(floornr<99.?2.*flnum_height/3.4:3.*flnum_height/3.4)) + (floornr<-1.?flnum_height/3.4:0.);
    vec3 pos2 = pos;
    pos2.y = mod(pos2.y, floor_height);
    return sdRoundBox(pos2 + vec3(staircase_width, -flnum_ypos, 0.), vec3(flnum_depth, flnum_height/2., flnw), 0.001); 
}

// Function 832
vec4 tMap(samplerCube iCh, vec3 p){

    // Multiplying "p" by 100 was style choice.
    p *= 100.;
    
    // Using the 3D coordinate to index into the cubemap and read
    // the isovalue. Basically, we need to convert Z to the particular
    // square slice on the 2D map, the read the X and Y values. 
    //
    // mod(p.xy, 100), will read the X and Y values in a square, and 
    // the offset value will tell you how far down (or is it up) that
    // the square will be.
    
    vec2 offset = mod(floor(vec2(p.z, p.z/10.)), vec2(10, 10));
    vec2 uv = (mod(floor(p.xy), 100.) + offset*100. + .5)/cubeMapRes;
    
    // Back Z face -- Depending on perspective. Either way, so long as
    // you're consistant.
    return texture(iCh, vec3(fract(uv) - .5, .5));
}

// Function 833
vec2 mapSnail( vec3 p, out vec4 matInfo )
{
    vec3 head = vec3(-0.76,0.6,-0.3);
    
    vec3 q = p - head;

    // body
#if 1
    vec4 b1 = sdBezier( vec3(-0.13,-0.65,0.0), vec3(0.24,0.9+0.1,0.0), head+vec3(0.04,0.01,0.0), p );
    float d1 = b1.x;
    d1 -= smoothstep(0.0,0.2,b1.y)*(0.16 - 0.07*smoothstep(0.5,1.0,b1.y));
    b1 = sdBezier( vec3(-0.085,0.0,0.0), vec3(-0.1,0.9-0.05,0.0), head+vec3(0.06,-0.08,0.0), p );
    float d2 = b1.x;
    d2 -= 0.1 - 0.06*b1.y;
    d1 = smin( d1, d2, 0.03 );
    matInfo.xyz = b1.yzw;
#else
    vec4 b1 = sdBezier( vec3(-0.13,-0.65,0.0), vec3(0.24,0.9+0.11,0.0), head+vec3(0.05,0.01-0.02,0.0), p );
    float d1 = b1.x;
    d1 -= smoothstep(0.0,0.2,b1.y)*(0.16 - 0.75*0.07*smoothstep(0.5,1.0,b1.y));
    matInfo.xyz = b1.yzw;
    float d2;
#endif
    d2 = sdSphere( q, vec4(0.0,-0.06,0.0,0.085) );
    d1 = smin( d1, d2, 0.03 );
    
    d1 = smin( d1, sdSphere(p,vec4(0.05,0.52,0.0,0.13)), 0.07 );
    
    q.xz = mat2(0.8,0.6,-0.6,0.8)*q.xz;

    vec3 sq = vec3( q.xy, abs(q.z) );
    
    // top antenas
    vec3 af = 0.05*sin(0.5*iTime+vec3(0.0,1.0,3.0) + vec3(2.0,1.0,0.0)*sign(q.z) );
    vec4 b2 = sdBezier( vec3(0.0), vec3(-0.1,0.2,0.2), vec3(-0.3,0.2,0.3)+af, sq );
    float d3 = b2.x;
    d3 -= 0.03 - 0.025*b2.y;
    d1 = smin( d1, d3, 0.04 );
    d3 = sdSphere( sq, vec4(-0.3,0.2,0.3,0.016) + vec4(af,0.0) );
    d1 = smin( d1, d3, 0.01 );    
    
    // bottom antenas
    vec3 bf = 0.02*sin(0.3*iTime+vec3(4.0,1.0,2.0) + vec3(3.0,0.0,1.0)*sign(q.z) );
    vec2 b3 = udSegment( sq, vec3(0.06,-0.05,0.0), vec3(-0.04,-0.2,0.18)+bf );
    d3 = b3.x;
    d3 -= 0.025 - 0.02*b3.y;
    d1 = smin( d1, d3, 0.06 );
    d3 = sdSphere( sq, vec4(-0.04,-0.2,0.18,0.008)+vec4(bf,0.0) );
    d1 = smin( d1, d3, 0.02 );
    
    // bottom
    vec3 pp = p-vec3(-0.17,0.15,0.0);
    float co = 0.988771078;
    float si = 0.149438132;
    pp.xy = mat2(co,-si,si,co)*pp.xy;
    d1 = smin( d1, sdEllipsoid( pp, vec3(0.0,0.0,0.0), vec3(0.084,0.3,0.15) ), 0.05 );
    d1 = smax( d1, -sdEllipsoid( pp, vec3(-0.08,-0.0,0.0), vec3(0.06,0.55,0.1) ), 0.02 );
    
    // disp
    float dis = textureLod( iChannel1, 5.0*p.xy, 0. ).x;
    float dx = 0.5 + 0.5*(1.0-smoothstep(0.5,1.0,b1.y));
    d1 -= 0.005*dis*dx*0.5;
        
    return vec2(d1,1.0);
}

// Function 834
vec2 Map(in vec3 p)
{
	vec2 a;
	float mat = 0.0;
	float anim = min(sqrt(iTime*.1+0.01) +.2, 1.);
	
	// Tilt depending on height...
	float t = -.9+smoothstep(-50.0, -400.0, p.y*2.2);
	p.zy = Rotate2D(p.zy, t);
	float f = length(p*vec3(1.0, 2.5, 1.0))-50.0;
	
	// Spin faster around centre...
	float l = dot(p.xz, p.xz) * .0162+.5;
	t = sqrt(50.0 / (l+.5));
	p.xz = Rotate2D(p.xz, t*anim*anim);
	
	// arctan needs to wrap in the noise function...
	a.x = (atan(p.x, p.z)+PI)/ (2.0 * PI) * 10.0;
	a.y  = pow(l, .35)*11.3;
	a.y *= smoothstep(15.0/(anim*anim), 0.0, (p.y*.2+2.3)*anim);
    float n = NoiseWrap(a)*40.0-23.0;
	n = n * smoothstep(85.0, 50.0, l);
	f = f + n;
	f = mix(dot(p, p)-2380.0, f, pow(anim, .05));
	
	// Stem...
	n = Cylinder(p-vec3(0.0, -100, 0.0), vec2(4.0, 100.0));
	if (n < f)
	{
		mat = 1.0;
		f = n;
	}
	return vec2(f, mat);
}

// Function 835
vec3 heatmapGradient(float t) {
	return clamp((pow(t, 1.5) * 0.8 + 0.2) * vec3(smoothstep(0.0, 0.35, t) + t * 0.5, smoothstep(0.5, 1.0, t), max(1.0 - t * 1.7, t * 7.0 - 6.0)), 0.0, 1.0);
}

// Function 836
float map(vec3 p){
    return mapMat(p).x;
}

// Function 837
float map(vec3 p){
    return sdEllipsoid(p,vec3(1.6,1,1.2)) + fbm(p*0.2) - 0.65;
}

// Function 838
vec3 triangleRemap(const vec3 n) {
    return vec3(
        triangleRemap(n.x),
        triangleRemap(n.y),
        triangleRemap(n.z)
    );
}

// Function 839
vec3 Tonemap(vec3 color)
{
    color = Reinhard(color);
    color = pow(abs(color), vec3(1.0 / Gamma));
    
    return color;
}

// Function 840
vec2 map(vec3 pos)
{
    float tunnel = map_tunnel(pos);
    vec2 res = vec2(tunnel, TUNNEL_OBJ);
    #ifdef show_water
    float water = map_water(pos);
    if (traceWater)
       res = opU(res, vec2(water, WATER_OBJ));
    #endif
    float lamps = map_lamps(pos);
    #ifdef show_lamps
    res = opU(res, vec2(lamps, LAMPS_OBJ));
    #endif

    return res;
}

// Function 841
uint spheremap_24( in vec3 nor )
{
    vec2 v = nor.xy*inversesqrt(2.0*nor.z+2.0);
    return packSnorm2x12(v);
}

// Function 842
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir ) {
    if (iFrame <= 5) { // initial terrain
        vec3 p = 1.5 * rayDir;
        fragColor.x = 0.;
        for (float i = 0.; i < 5.; i++) {
            float c = craters(0.4 * pow(2.2, i) * p);
            float noise = 0.4 * exp(-3. * c) * FBM(10. * p);
            float w = clamp(3. * pow(0.4, i), 0., 1.);
            fragColor.x += w * (c + noise);
        }
        fragColor.x = pow(fragColor.x, 3.);
        fragColor.x = (5. - fragColor.x) / 15.;
        return;
    }
    
    // which cube face are we on?
    vec3 rayAbs = abs(rayDir);
    int face = rayAbs.x > rayAbs.y ? rayAbs.x > rayAbs.z ? 0 : 2 : rayAbs.y > rayAbs.z ? 1 : 2; // faceID
    if (rayDir[face] < 0.) face += 3;
    
    /* rotation matrix for projecting 2D coords onto the 3D cube
       the face IDs on the unrolled cube look like:
         1
       3 2 0 5
         4
    */
    mat3 faceMat;
    switch (face) {
    case 0: faceMat = rotY(PI/2.); break;
    case 1: faceMat = rotX(-PI/2.); break;
    case 2: faceMat = rotY(0.); break;
    case 3: faceMat = rotY(-PI/2.); break;
    case 4: faceMat = rotX(PI/2.); break;
    case 5: faceMat = rotY(PI); break;
    }
    
    mainImage(fragColor, fragCoord, faceMat);
    
    // seed extra river basins
    vec3 h = hash33(vec3(fragCoord, iFrame));
    if (h.x < 1e-5 && h.y < 1e-5) fragColor.x -= 0.2;
}

// Function 843
float map(vec3 p){
    vec2 tun = p.xy - path(p.z);
    vec2 tun2 = p.xy - path2(p.z);
    float d = 1.- smoothMinP(length(tun), length(tun2), 4.) + (0.5-surfFunc(p));
    float dd = (sin(p.x/2.)+cos(p.z/1.5));

#ifdef JAGGED
    return max(d, noise(p.zx/2.)+p.y+noise(p.xz/3.)+dd+surfFunc(p/2.));
#endif    
    return smoothMaxP(d, (noise(p.zx/2.)+p.y+noise(p.xz/3.)+dd+surfFunc(p/2.)), .5);
}

// Function 844
vec4 mapClouds( in vec3 pos )
{
	vec3 q = pos*0.5 + vec3(0.0,-iTime,0.0);
	
	float d;
    d  = 0.5000*noise( q ); q = q*2.02;
    d += 0.2500*noise( q ); q = q*2.03;
    d += 0.1250*noise( q ); q = q*2.01;
    d += 0.0625*noise( q );
		
	d = d - 0.55;
	d *= smoothstep( 0.5, 0.55, lava(0.1*pos.xz)+0.01 );

	d = clamp( d, 0.0, 1.0 );
	
	vec4 res = vec4( d );

	res.xyz = mix( vec3(1.0,0.8,0.7), 0.2*vec3(0.4,0.4,0.4), res.x );
	res.xyz *= 0.25;
	res.xyz *= 0.5 + 0.5*smoothstep( -2.0, 1.0, pos.y );
	
	return res;
}

// Function 845
float MapRing(vec3 p, float inRadius, float outRadius, float height)
{
  pR(p.xy, 0.15);
  return max(sdCappedCylinder(p, vec2(outRadius, height)), -sdCappedCylinder(p, vec2(inRadius, 15.)));
}

// Function 846
vec2 uv_to_polar(vec2 uv, vec2 p) {
    vec2 translated_uv = uv - p;
    
    // Get polar coordinates
    vec2 polar = vec2(atan(translated_uv.x, translated_uv.y), length(translated_uv));
    
    // Scale to a range of 0 to 1
    polar.s /= TWO_PI;
    polar.s += 0.5;
    
    return polar;
}

// Function 847
float map(float value, float low1, float high1, float low2, float high2){
	return low2 + (value - low1) * (high2 - low2) / (high1 - low1);
}

// Function 848
vec3 Tonemap_ACESFitted2(vec3 acescg)
{
    vec3 color = acescg * RRT_SAT;
    
   #if 1
    color = ToneTF2(color); 
   #elif 1
    color = RRTAndODTFit(color);
   #elif 1
    color = ToneMapFilmicALU(color);
   #endif
    
    color = color * ACESOutputMat;
    //color = ToneMapFilmicALU(color);

    return color;
}

// Function 849
vec4 map( in vec3 p )
{
	vec4 d1 = fbmd( p );
    d1.x -= 0.37;
	d1.x *= 0.7;
    d1.yzw = normalize(d1.yzw);

    // clip to box
    vec4 d2 = sdBox( p, vec3(1.5) );
    return (d1.x>d2.x) ? d1 : d2;
}

// Function 850
vec3 cubemap(vec3 d, vec3 c1, vec3 c2)
{
	return fbm(d) * mix(c1, c2, d * .5 + .5);
}

// Function 851
float map(vec3 p )
{
    vec3 q = p;
 // vec3 N = 2.* noise(q/10.) -1.;                // displacement
    vec3 N = divfreenoise(   q/5.)
      ; // + divfreenoise(2.*q/5.) /2.
        // + divfreenoise(4.*q/5.) /4.
        // + divfreenoise(8.*q/5.) /8.;
    q += N;
    float f = // ( 1.2*noise(q/2.+ .1*iTime).x -.2 ) * // source noise
               smoothstep(1.,.8,dot(q,q)/1.5);   // source sphere

 // f*= smoothstep(.1,.2,abs(p.x));               // empty slice (derivable ) 
    z = length(q)/2.;                             // depth in sphere
    return f;                        
}

// Function 852
float remap_noise_tri_erp( const float v )
{
    float r2 = 0.5 * v;
    float f1 = sqrt( r2 );
    float f2 = 1.0 - sqrt( r2 - 0.25 );    
    return (v < 0.5) ? f1 : f2;
}

// Function 853
vec2
map( in vec3 pos )
{    
    float angle4 = iTime*TAU*0.25;
    
    MPt res;
    res.distance = 1e38;

    float m = mod( floor(pos.x * 2.0) + floor(pos.y * 2.0), 2.0 );
    res = union_op( MPt( plane_sd( pos ),
                         MAT_FLOOR_B * (1.0 - m) + MAT_FLOOR_W * m ),
                    res );
    
    const float axis_r = 0.05;
    res = union_op( MPt( sphere_sd( axis_r*2.0, at_pos( vec3(0.0), pos ) ),
                         MAT_PLASTIC ), res );
    res = union_op( MPt( cline_sd( vec3(0.0), vec3(1.0,0.0,0.0), axis_r, pos ),
                         MAT_RED ),res );
    res = union_op( MPt( cline_sd( vec3(0.0), vec3(0.0,1.0,0.0), axis_r, pos ),
                         MAT_GREEN ),res );
    res = union_op( MPt( cline_sd( vec3(0.0), vec3(0.0,0.0,1.0), axis_r, pos ),
                         MAT_BLUE ),res );
    
    
    
    
    // NOTE(theGiallo): light
    float lamp_r = 0.2, pole_hh=0.5;
    vec3 lp = repeated( rep,at_pos( point_light_pos + vec3(0.0,0.0,0.15), pos ));
    res =
       union_op(
           MPt( capped_cylinder_sd( vec2(0.02,pole_hh) , at_pos( vec3(0.0,0.0,lamp_r+pole_hh),lp ) ), MAT_PLASTIC ),
           res
       );
    res =
       union_op(
           intersect_op(
              subtract_op(
                 MPt( sphere_sd(
                         lamp_r-0.01, lp
                      ), MAT_PLASTIC
                    ),
                 MPt( sphere_sd(
                         lamp_r, lp
                      ), MAT_PLASTIC
                    )
               ),
               MPt( aab_sd( vec3(lamp_r*2.0), at_pos( vec3(0.0,0.0,lamp_r),lp) ), MAT_PLASTIC )
            ), res );
    
    vec3 op = pos;
    pos = repeated( vec3(0.0,4.0,0.0), pos );
    
    // NOTE(theGiallo): tunnel section
    float X = 12.0;
    vec3 P = at_pos(vec3(X,0.0,2.5), pos );
    res =
       union_op(
          subtract_op(
             union_op(
                MPt( capped_cylinder_sd (vec2(2.0,5.0),
                                         at_angle( vec3(HPI,0.0,0.0),
                                                   P ) ), MAT_TUNNEL_WALL_W ),
                intersect_op(
                   MPt( capped_cylinder_sd (vec2(2.05,0.25),
                                            at_angle( vec3(HPI,0.0,0.0),
                                                      repeated( vec3(0.0,1.0,0.0),
                                                                P ) ) ), MAT_ORANGE ),
                   MPt( aab_sd(vec3(5.0,4.0,3.0),at_pos(vec3(0.0,0.0,1.0),P) ), MAT_TUNNEL_WALL_W ) )
                ),
             MPt( aab_sd(vec3(5.0,4.05,5.0),P ), MAT_TUNNEL_WALL_W )
          ),
          res );

    // NOTE(theGiallo): grid
    res =
       union_op(
          subtract_op(
             MPt( aab_sd( vec3(0.02,0.02,0.1),
                          repeated( vec3(0.02,0.02,0.0), at_angle( vec3(0.0,0.0,QPI),
                                    at_pos( vec3(X,0.0,1.0),
                                            pos) ) ) ), MAT_METAL_GRID ),
             MPt( aab_sd(vec3(2.5,4.0,0.02),at_pos(vec3(X,0.0,1.0),pos) ), MAT_METAL_GRID )
          ),
          res );

    
    pos = op;
    
    

	return res;
}

// Function 854
float hsluv_toLinear(float c) {
    return c > 0.04045 ? pow((c + 0.055) / (1.0 + 0.055), 2.4) : c / 12.92;
}

// Function 855
vec2 map3ds(vec2 z)
{
 // animation specific hacks
 vec2 z2 = z*z;
 vec2 a  = 2.+vec2(z2.x-z2.y, -z2.x+z2.y);
 vec2 b  = twosqrt2*z;
 vec2 p  = a+b;
 vec2 m  = a-b;
 vec2 r  = sgn(p)*sqrt(abs(p))-sgn(m)*sqrt(abs(m));
 return 0.5*r;
}

// Function 856
vec2 mapRMDetailed(vec3 p) {
    vec2 d = vec2(-1.0, -1.0);
    d = vec2(mapTerrain(p-vec3(0.0, FLOOR_LEVEL, 0.0), FLOOR_TEXTURE_AMP), TYPE_FLOOR);
    //d = opU(d, vec2(mapWaterDetailed(p-vec3(0.0, WATER_LEVEL, 0.0)), TYPE_WATER));
    //d = opU(d, vec2(sdBox(p-BOATPOS, vec3(1.0, 1.0, 1.0)), TYPE_BOAT));
    return d;
}

// Function 857
Model map( vec3 p , bool glitchMask){
    mat3 m = modelRotation();
    p *= m;
    pR(p.xz, -time*PI);
    if (glitchMask) {
    	return glitchModel(p);
    }
    Model model = mainModel(p);
    return model;
}

// Function 858
float llamelMapLeg(vec3 p, vec3 j0, vec3 j3, vec3 l, vec4 r, vec3 rt){//z joint with tapered legs
	float lx2z=l.x/(l.x+l.z),h=l.y*lx2z;
	vec3 u=(j3-j0)*lx2z,q=u*(0.5+0.5*(l.x*l.x-h*h)/dot(u,u));
	q+=sqrt(max(0.0,l.x*l.x-dot(q,q)))*normalize(cross(u,rt));
	vec3 j1=j0+q,j2=j3-q*(1.0-lx2z)/lx2z;
	u=p-j0;q=j1-j0;
	h=clamp(dot(u,q)/dot(q,q),0.0,1.0);
	float d=length(u-q*h)-r.x-(r.y-r.x)*h;
	u=p-j1;q=j2-j1;
	h=clamp(dot(u,q)/dot(q,q),0.0,1.0);
	d=min(d,length(u-q*h)-r.y-(r.z-r.y)*h);
	u=p-j2;q=j3-j2;
	h=clamp(dot(u,q)/dot(q,q),0.0,1.0);
	return min(d,length(u-q*h)-r.z-(r.w-r.z)*h);
}

// Function 859
vec2 mapOpaque( vec3 p, out vec4 matInfo )
{
    matInfo = vec4(0.0);
    
   	//--------------
    vec2 res = mapSnail( p, matInfo );
    
    //---------------
    vec4 tmpMatInfo;
    float d4 = mapShell( p, tmpMatInfo );    
    if( d4<res.x  ) { res = vec2(d4,2.0); matInfo = tmpMatInfo; }

    //---------------
    
    // plant
    vec4 b3 = sdBezier( vec3(-0.15,-1.5,0.0), vec3(-0.1,0.5,0.0), vec3(-0.6,1.5,0.0), p );
    d4 = b3.x;
    d4 -= 0.04 - 0.02*b3.y;
    if( d4<res.x  ) { res = vec2(d4,3.0); }
	
	//----------------------------
    
    float d5 = mapLeaf( p );
    if( d5<res.x ) res = vec2(d5,4.0);
        
    return res;
}

// Function 860
float map(vec3 p){
    
   
    //float sf = cellTile(p*.25); // Cellular layer.
    //sf = smoothstep(-.1, .5, sf);
    

    // Trancendental gyroid functions and a function to perturb
    // the tunnel. For comparisson, I included a rough triangle
    // function equivalent option.
    #if 1
    vec3 q = p*3.1415926;
    float cav = dot(cos(q/2.), sin(q.yzx/2.5)); // Gyroid one.
    float cav2 = dot(cos(q/6.5), sin(q.yzx/4.5)); // Gyroid two.
    cav = smin(cav, cav2/2., 2.); // Smoothly combine the gyroids.
    
    // Transendental function to perturb the walls.
    float n = dot(sin(q/3. + cos(q.yzx/6.)), vec3(.166));
    //float n = (-cellTile(p*.125) + .5)*.5;
    #else
    vec3 q = p/2.;
    float cav = dot(triC(q/2.), triS(q.yzx/2.5)); // Triangular gyroid one.
	float cav2 = dot(triC(q/6.5), -triS(q.yzx/4.5)); // Triangular gyroid two.
    cav = smin(cav, cav2/2., 2.); // Smoothly combine the gyroids.
    
    // Triangular function to perturb the walls.
    float n = dot(triS(q/3. + triC(q.yzx/6.)), vec3(.166));
    //float n = (-cellTile(p*.125) + .5)*.5;
	#endif

    // Wrap the tunnel around the camera path.
    p.xy -= path(p.z);
    

    // Smoothly combining the wrapped cylinder with the gyroids, then 
    // adding a bit of perturbation on the walls.
    n = smax((2.25 - dist2D(p.xy)), (-cav - .75), 1.) +  n;// - sf*.375;
    
    // Return the distance value for the scene. Sinusoids aren't that great
    // to hone in on, so some ray shortening is a necessary evil.
    return n*.75;
 
}

// Function 861
float map(vec3 p) {
   p.z += 1.;
   R(p.yz, -25.5);// -1.0+iMouse.y*0.003);
   R(p.xz, iMouse.x*0.008*pi+iTime*0.1);
   return SunSurface(p) +  fpn(p*10.+iTime*25.) * 0.45;
}

// Function 862
float map(vec3 p)
{
    
    p.xz *= rot(iTime * .25);
    
    float len = length(p);
    float boost = fract(len * .05 - iTime * .25);
    boost = max(0., boost *9. - 8.);
    boost = boost * boost;
    boost = sin(boost * PI);

   	p *= 1. + boost * .1;
    
    float volNoise = tetraNoise(p * .6 + vec3(0.,sin(iTime * .5 + p.x * .5) * .2,0.));
    volNoise -= boost * .4;
    volNoise = volNoise * volNoise * 1.5;    
    
    float dist = .5 - volNoise;
    
    p = abs(p);
	float cu = (p.x + p.y + p.z) * (.5 + sin(boost * 2.) *.025) - boost * volNoise;
    
    dist = min(dist, 3. - cu);
    return dist;
}

// Function 863
float map(vec3 p){
    p = mod(p, 4.) - 2.0;
    float t = iTime;
    
    float bmp = sin(p.x*BUMP_AMOUNT-t)* sin(p.y*BUMP_AMOUNT+t)* sin(p.z*BUMP_AMOUNT-t*1.5);
    bmp*=.025;
    
    
    return (lengthN(p,8.) - 0.75)+bmp;
}

// Function 864
float o5268_input_t_map(vec2 uv) {
vec2 o5271_0_wat = abs((uv) - 0.5);
float o5271_0_d = max(o5271_0_wat.x,o5271_0_wat.y);vec4 o5271_0_1_rgba = o5271_gradient_gradient_fct(2.0*abs(fract(-0.2*iTime + p_o5271_repeat*o5271_0_d)-0.5));
float o118692_0_1_f = easeInOutCubic((dot((o5271_0_1_rgba).rgb, vec3(1.0))/3.0));

return o118692_0_1_f;
}

// Function 865
float map_background(vec3 pos)
{
   vec4 h = hexagon(pos.xz);
   
   return max(h.z - bg_width/2., abs(pos.y) - bg_height/2.);
}

// Function 866
float Map(in vec3 p)
{
	float h = Terrain(p.xz);
		

	float ff = Noise(p.xz*.3) + Noise(p.xz*3.3)*.5;
	treeLine = smoothstep(ff, .0+ff*2.0, h) * smoothstep(1.0+ff*3.0, .4+ff, h) ;
	treeCol = Trees(p.xz);
	h += treeCol;
	
    return p.y - h;
}

// Function 867
define POS2UVH(aabb,pos,uvd) { vec3 aabbDim = aabb.max_ - aabb.min_; uvd = (pos-aabb.min_)/aabbDim; }

// Function 868
float mapSeed01(vec2 f)
{
    //uv = (uv + 1.)/2.;
    DecodeData(texelFetch( iChannel0, ivec2(f),0), seedCoord, seedColor);
    return min(LIGHT_DIST, length((floor(seedCoord)-floor(f))/iResolution.x)-seedColor.z*circSizeMult);
}

// Function 869
vec3 mapCoord(vec2 uv)
{
	uv = (fract((uv + 1.) / 2.) - .5) * 2.;
    return vec3(-1., -uv.yx * vec2(1, 1));
}

// Function 870
float MapBoat(vec3 p)
{
  p=TranslateBoat(p);
  // AABB
  if (sdBox(p+vec3(0., 0., .50), vec3(1.7, 1.0, 4.))>1.) return 10000.;

  // hull exterior  
  float centerDist =length(p.x-0.);    
  float centerAdd = 0.07*-smoothstep(0.04, 0.1, centerDist);
  float frontDist = max(0.01, 1.3-max(0., (0.25*(0.15*pow(length(p.z-0.), 2.)))));
  float widthAdd =mix(0.06*(floor(((p.y+frontDist) + 0.15)/0.3)), 0., step(2., length(0.-p.y)));

  float d= fCylinder( p, 1.45+widthAdd, 1.3+centerAdd+widthAdd);  
  d =min(d, sdEllipsoid( p- vec3(0, 0, 1.250), vec3(1.45+widthAdd, 1.465+centerAdd, 1.+centerAdd+widthAdd))); 
  d =min(d, sdEllipsoid( p- vec3(0, 0, -1.20), vec3(1.45+widthAdd, 1.465+centerAdd, 3.+centerAdd+widthAdd))); 

  // hull cutouts
  d= max(d, -fCylinder( p- vec3(0, 0.25, -0.10), 1.3+widthAdd, 1.4));  
  d =max(d, -max(sdEllipsoid( p- vec3(0, 0.05, -.60), vec3(1.25+widthAdd, 1.2, 3.40)),-sdBox(p-vec3(0.,0.,4.), vec3(3., 10., 3.1)))); 

  // cut of the to part of the hull to make the boat open

  d=max(d, -sdBox(p-vec3(0., 1.05+centerAdd, 0.), vec3(10., frontDist, 14.)));

  // seats
  return min(d, min(sdBox(p-vec3(0., -0.5, 0.9), vec3(1.3, 0.055, 0.35)), sdBox(p-vec3(0., -0.5, 0.9-2.2), vec3(1.3, 0.055, 0.35))));
}

// Function 871
float map(vec3 p){
 
     return 1.-abs(p.y) - (0.5-surfFunc(p))*1.5;
 
}

// Function 872
vec3 mapMushroom( in vec3 pos, in vec3 cur )
{
    vec3 res = cur;

    vec3 qos = worldToMushrom(pos);
    float db = length(qos-vec3(0.0,1.2,0.0)) - 1.3;
    if( db<cur.x )
    {

        {

            float d1 = sdEllipsoid( qos, vec3(0.0, 1.4,0.0), vec3(0.8,1.0,0.8) );

            d1 -= 0.025*textureLod( iChannel1, 0.05*qos.xz, 0.0 ).x - 0.02;

            float d2 = sdEllipsoid( qos, vec3(0.0, 0.5,0.0), vec3(1.3,1.2,1.3) );
            float d = smax( d1, -d2, 0.1 );
            d *= 0.8;
            if( d<res.x )
            {
                res = vec3( d, MAT_MUSH_HEAD, 0.0 );
            }
        }


        {
            pos.x += 0.3*sin(pos.y) - 0.65;
            float pa = sin( 20.0*atan(pos.z,pos.x) );
            vec2 se = sdLine( pos, vec3(0.0,2.0,0.0), vec3(0.0,0.0,0.0) );

            float tt = 0.25 - 0.1*4.0*se.y*(1.0-se.y);

            float d3 = se.x - tt;

            d3 = smin( d3, sdEllipsoid( pos, vec3(0.0, 1.7 - 2.0*length2(pos.xz),0.0), vec3(0.3,0.05,0.3) ), 0.05);
            d3 += 0.003*pa;
            d3 *= 0.7;
            
            if( d3<res.x )
                res = vec3( d3, MAT_MUSH_NECK, 0.0 );
        }
    
    }
    return res;
}

// Function 873
float map( vec3 p )
{
    p.x *= 0.8;
    p *= 2.6;
    p.xyz += 1.000*sin(  2.0*p.yzx );
    //p.xyz -= 0.500*sin(  4.0*p.yzx );
    float d = length( p.xyz ) - 1.5;
	return d * 0.15;
}

// Function 874
vec2 mapRefract(vec3 p) {
  float d  = icosahedral(p, 1.0);
  float id = 0.0;

  return vec2(d, id);
}

// Function 875
vec3 doBumpMap( in vec3 pos, in vec3 nor )
    {
        float e = 0.0015;
        float b = 0.1;
        
        float ref = fbm( pos, nor );
        vec3 gra = b*vec3( fbm( vec3(pos.x+e, pos.y, pos.z),nor)-ref,
                            fbm( vec3(pos.x, pos.y+e, pos.z),nor)-ref,
                            fbm( vec3(pos.x, pos.y, pos.z+e),nor)-ref )/e;
        
        vec3 tgrad = gra - nor * dot ( nor , gra );
        return normalize ( nor - tgrad );
    }

// Function 876
float digitBitmaps(const in int x) {
	return x==0?961198.0:x==1?279620.0:x==2?953902.0:x==3?953998.0:x==4?700040.0:x==5?929422.0:x==6?929454.0:x==7?952456.0:x==8?962222.0:x==9?962184.0:0.0;
}

// Function 877
vec2 getUv(vec3 normal)
{
    vec2 xzNorm = normalize(normal.xz);
    return vec2((acos(xzNorm.x) / TWO_PI), atan( normal.y/( length(normal.xz) ) ) / TWO_PI);
}

// Function 878
vec3 ToneMap_FilmicALU(vec3 color)
{
    color = max(vec3(0.0), color - 0.004);
    color = (color * (6.2 * color + 0.5)) / (color * (6.2 * color + 1.7) + 0.06);
    return color;
}

// Function 879
vec3 nmapu(vec3 x){ return x*.5+.5; }

// Function 880
void dmap (out vec3 c, vec3 p)
{
	c = p.y > EPS ? 
		(p.y > 9.54 ? (p.y > 10.45 ? vec3(1, 0, 0) : vec3(0)): vec3(1))
		: vec3(.8);
		
	if (length(p - vec3(0, 10, -10)) < 1.5) c = vec3(1);
	else if (length(p - vec3(0, 10, -10)) < 1.8) c = vec3(0);
}

// Function 881
vec3 sphereMap(vec3 d){return vec3(.3,.4,1.2);}

// Function 882
float triangleRemap(float n) {
    float origin = n * 2.0 - 1.0;
    float v = origin / sqrt(abs(origin));
    v = max(-1.0, v);
    v -= sign(origin);
    return v;
}

// Function 883
vec3 TonemapFloat3( vec3 x )
{
    vec3 r;
    r.x = TonemapFloat( x.x );
    r.y = TonemapFloat( x.y );
    r.z = TonemapFloat( x.z );
    
    return r;
}

// Function 884
vec3 rgbToHsluv(float x, float y, float z) {return rgbToHsluv( vec3(x,y,z) );}

// Function 885
vec2 DrawUVQuad(vec2 a, vec2 b, vec2 c, vec2 d,vec2 uva, vec2 uvb, vec2 uvc, vec2 uvd, float t, vec2 co){
    float i = DrawQuad(a,b,c,d,t,co);
    if (i<=0.) return vec2(0);
    vec3 baria = toBari(a,b,c,co);
    vec3 barib = toBari(a,d,c,co);
    vec3 baric = toBari(b,c,d,co);
    vec3 barid = toBari(b,a,d,co);
    vec2 coord = vec2(0);
    coord+= toCartesian(uvb,uvc,uvd,baric);
    coord+= toCartesian(uvb,uva,uvd,barid);
    coord+= toCartesian(uva,uvb,uvc,baria);
    coord+= toCartesian(uva,uvd,uvc,barib);
    
    return (coord/4.)*i;
}

// Function 886
float map_metalrings(vec3 pos)
{
    pos.y = abs(pos.y);
    pos.y-= lampsize.y + 0.08;
    pos.y*= 2.2;
    
    return sdTorus(pos, vec2(lampsize.x*.93, 0.2));
}

// Function 887
vec3 bumpMap(sampler2D tex, vec3 pos, vec3 nor, float amount) {
	float e = 0.001;

	float ref = luma(texcube(tex, pos, nor));

	vec3 gra = -vec3(luma(texcube(tex, vec3(pos.x + e, pos.y, pos.z), nor)) - ref,
					 luma(texcube(tex, vec3(pos.x, pos.y + e, pos.z), nor)) - ref,
					 luma(texcube(tex, vec3(pos.x, pos.y, pos.z + e), nor)) - ref) / e;

	vec3 tgrad = gra - nor * dot(nor, gra);
	return normalize(nor - amount * tgrad);
}

// Function 888
float map(in vec3 p)
{
    
	vec3 q = p;
    //R2d(q.xy, iTime);R2d(q.zy, iTime); 
    q.z-=2.*sin(iTime);
    #define mk(_v,_m)(pow(dot(pow(_v,vec3(_m)),vec3(1)), 1./_m))
	float t =  mk(abs(q), max(1.,mod(floor(4.*iTime), 5.)) ) - 1.;
	t = min(t, 1.5+dot(p, vec3(0, 1, 0)) + (sin(p.x * .55) * sin(3.*iTime-.5*p.z)));
    return t; 
}

// Function 889
vec3 yuv2rgb(vec3 rgb) {
  return rgb * yuv2rgb_mat;
}

// Function 890
vec3 rgbToHsluv(vec3 tuple) {
    return lchToHsluv(rgbToLch(tuple));
}

// Function 891
vec2 dudvmap(vec2 uv) {
    const float eps = 0.01;
    vec2 offset = vec2(eps, 0.0);
    return vec2(
        heightmap(uv+offset.xy) - heightmap(uv-offset.xy),
        heightmap(uv+offset.yx) - heightmap(uv-offset.yx)
    );
}

// Function 892
float map(vec3 p ) 
{
    float f = 0.;
    vec3 q = p;
#if 1
    q += .7*(2.* turb(q/vec3(20,10,10)+.02*iTime).xyz -1.);   // displacement noise
#else
    f -= 1.3*turb(q/8.+ .02*iTime).x;              // density noise
#endif
 // f += smoothstep(1.,.8,length(q)/2.);           // main sphere ( mask )
    f += 1.3 - 3.*( length(q)/2. - .7 );           // main sphere ( mask )
                 

 // f*= smoothstep(.1,.2,abs(p.x));                // empty slice (derivable ) 
 // z = length(q)/2.;                              // depth in sphere
    return f;                        
}

// Function 893
vec2 map(vec3 pos, bool inside)
{
    float bubble = map_bubble(pos);
    if (inside) bubble*=-1.;
    vec2 res = vec2(bubble, BUBBLE_OBJ);
    
    return res;
}

// Function 894
vec3 lumaBasedReinhardToneMapping(vec3 color)
{
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	float toneMappedLuma = luma / (1. + luma);
	color *= toneMappedLuma / luma;
	color = pow(color, vec3(1. / gamma));
	return color;
}

// Function 895
vec2 randUv(vec2 uv) {
    return cos(float(iFrame)*.10924+abs(cos(float(iFrame)*2.345+uv)*128.345)+abs(cos(uv.yx*32.345)*16.234));
}

// Function 896
float map_int(vec3 pos)
{
    pos = pos.yxz;
    pos-= chimneyOrig;
    
    return getTubeInt(pos);
}

// Function 897
v0 suv(v3 a){return dot(v3(1),a);}

// Function 898
float mapAndCap(in float value, in float istart, in float istop, in float ostart, in float ostop) {
    float v = map(value, istart, istop, ostart, ostop);
    v = max( min(ostart,ostop), v);
    v = min( max(ostart,ostop), v);
    return v;
}

// Function 899
vec2 map(vec3 q3){
	q3.y += 1.25;
	//q3.z -= 5.25;
    vec2 res = vec2(1000.,0.);
    
    const float scale = 1./SCALE;		// dimension | length to height ratio.
	const vec2 l = vec2(scale);
	const vec2 s = l*2.;				// helper | size of the repeat cell.

    vec2 p,
         ip,
         id = vec2(0),
         ct = vec2(0);
    const vec2[4] ps4 = vec2[4](vec2(-.5, .5), vec2(.5),   vec2(.5, -.5), vec2(-.5));
    
    float d = 1e5,
          boxID = 0.,
          savedHeight,
          sck;
    
    for(int i = 0; i<4; i++){
        ct = ps4[i]/2. -  ps4[0]/2.;	// Block center.
        p = q3.xy - ct*s;				// Local coordinates. 
        ip = floor(p/s) + .5;			// Local tile ID. 
        p -= (ip)*s; 					// New local position.		   
        vec2 idi = (ip + ct)*s;			// Correct position & tile ID.

        vec3 q = vec3(p.x,p.y,q3.z);
		// if within allowed space sample otherwise
        float ck = (idi.x>-9.5 && idi.x<12.5 && idi.y>-4.5 && idi.y<8.5) ? 1. : .0;
		float pt = ck > 0. ? hm(idi) : .01 ;
		sck=ck;
        pt = floor(pt*75.999)/75.;		// Discreet heights and making 
        float ht=max(pt,.01);			// sure its never less than
        
        vec3 sz = vec3(l.x/2.2,l.y/2.2,ht);
        float di = fBox(q+vec3(0,0,ht),sz,.005);

        // box
        if(di<d) {
            d = di;
            savedHeight = ht;
            savedId = idi;
            vec2 id = ck > 0. ? idi : vec2(4.);
            saveColor= M.z>0. ? getTex(id).rgb : gethue(ht).rgb;
        }
    }
    if(d<res.x&&sck>0.) res = vec2(d,1.);
 	// the floor&&sck>0.
    float g = q3.y+4.;
    if(g<res.x) res = vec2(g,2.);
    // random boxes
    float f = fBox(q3+vec3(-3.75,3.,4.),vec3(1.),.015);
    f=min(fBox(q3+vec3(2.15,3.5,2.75),vec3(.5),.015),f);
    vec3 q4 = vec3(-abs(q3.x),q3.y,q3.z)+vec3(7.75,3.25,3.75);
    q4.xz*=r45;
    f=min(fBox(q4,vec3(.75),.025),f);
    q4.xz*=r24;
    f=min(fBox(q4-vec3(-1.95,-.22,3.5),vec3(.5),.015),f);
   
    if(f<res.x) {
        res = vec2(f,3.);
        saveColor= M.z>0. ? getTex(vec2(5.,5.)).rgb : gethue(2.7).rgb;
    }
    return res;
}

// Function 900
float map(vec3 p)
{	
	pR(p.yx,bounce*.4);
    pR(p.zx,iTime*.3);
	return min(max(sdBox(p,vec3(2.5, 2.5, 0))-.3,-sdBox(p,vec3(1.2, 1.2, 1))+.5)-0.003*noise(55.*p),length(p)-1.6+0.3*noise(3.5*p-.5*iTime));
}

// Function 901
float mapQ(vec3 p){
  float s = 0.5;
  for(float i = 1.; i < MAX_LEVEL; i++){
    s *= 2.;
    
    //if we don't pass the random check, add max to index so we break
    i += step( hash13(floor(p * s)), 0.5 ) * MAX_LEVEL;
  }
  return s;
}

// Function 902
vec2 to_uv(in vec2 in_pixels) {
    return in_pixels / iResolution.xy;
}

// Function 903
float hsluv_fromLinear(float c) {  return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055; }

// Function 904
vec3 RgbTonemap(vec3 rgbLinear)
{
	// Desaturate with luminance

	float gLuminance = GLuminance(rgbLinear);
	rgbLinear = mix(rgbLinear, vec3(gLuminance), GSqr(saturate((gLuminance - 1.0) / 1.0)));

	// Hejl/Burgess-Dawson approx to Hable operator; includes sRGB conversion

	vec3 rgbT = max(vec3(0.0), rgbLinear - 0.004);
	vec3 rgbSrgb = (rgbT * (6.2 * rgbT + 0.5)) / (rgbT * (6.2 * rgbT + 1.7) + 0.06);

	return rgbSrgb;
}

// Function 905
vec3 GetReflectionMap(vec3 rayDir, vec3 normal)
{
  return texture(iChannel3, reflect( rayDir, normal )).rgb;
}

// Function 906
float map( vec3 p) {
    
    float px = mod(p.x+8.,16.)-8.;
    
    float lineId = floor((px-p.x+8.)/16.);
    p.x = px;
    float rnd = hash(lineId+10.);
    p.z += cos(g_time*rnd+2.*rnd)*rnd;
    p.y += cos(g_time+5.*rnd)*rnd;
    
    float d = dToga0(p);
    d = min(d, spaceship0(p, lineId));
        
    vec3 p0 = p;
    p -= headRotCenter;
    p.yz *= g_headRotH;
    p += headRotCenter;
    
	d = min(d, dSkinPart(p0,p));
    p.x = abs(p.x);
    d = min(d, dEye(p- g_eyePos));

    return d;
}

// Function 907
vec2 animUV(vec2 uv, float t) {
    //uv += vec2(sin(t*0.25), cos(t*0.37)) * vec2(0.05,0.03);
    
    float angle = sin(t*0.15) * 0.5;
    float c = cos(angle);
    float s = sin(angle);
    uv = mat2(c,-s,s,c) * uv;
    
    return uv;
}

// Function 908
float map(vec2 p) {
    vec2 pos=p;
    float t=iTime;
    col+=fractal(p);
    vec2 p2=abs(.5-fract(p*8.+4.));
	float h=0.;
    h+=sin(length(p)+t);
    p=floor(p*2.+1.);
    float l=length(p2*p2);
    h+=(cos(p.x+t)+sin(p.y+t))*.5;
    h+=max(0.,5.-length(p-vec2(18.,0.)))*1.5;
    h+=max(0.,5.-length(p+vec2(18.,0.)))*1.5;
    p=p*2.+.2345;
    t*=.5;
    h+=(cos(p.x+t)+sin(p.y+t))*.3;
    return h;
}

// Function 909
vec2 errorCheckUV(vec2 checkUV, vec2 returnUV){
  float errorFlg = errorCheckUV(checkUV);
  return step(0.1,errorFlg) * (-1.,-1.) + (1.0 - step(0.1,errorFlg)) * returnUV;
}

// Function 910
float remap( float range_a_point, float a0, float a1, float b0, float b1 ){
    return (((range_a_point - a0) * (abs(b1-b0)))/abs(a1-a0)) + b0;
}

// Function 911
vec2 to_uv(in vec2 in_pixels) {
    return mod(vec2(1.0) + (in_pixels / iResolution.xy), vec2(1.0));
}

// Function 912
vec2 map( in vec3 p )
{
    vec4 z = vec4( p, 0.0 );
    float dz2 = 1.0;
	float m2  = 0.0;
    float n   = 0.0;
    #ifdef TRAPS
    float o   = 1e10;
    #endif
    
    for( int i=0; i<kNumIte; i++ ) 
	{
        // z' = 3z² -> |z'|² = 9|z²|²
		dz2 *= 9.0*qLength2(qSquare(z));
        
        // z = z³ + c		
		z = qCube( z ) + kC;
        
        // stop under divergence		
        m2 = qLength2(z);		

        // orbit trapping : https://iquilezles.org/www/articles/orbittraps3d/orbittraps3d.htm
        #ifdef TRAPS
        o = min( o, length(z.xz-vec2(0.45,0.55))-0.1 );
        #endif
        
        // exit condition
        if( m2>256.0 ) break;				 
		n += 1.0;
	}
   
	// sdf(z) = log|z|·|z|/|dz| : https://iquilezles.org/www/articles/distancefractals/distancefractals.htm
	float d = 0.25*log(m2)*sqrt(m2/dz2);
    
    #ifdef TRAPS
    d = min(o,d);
    #endif
    #ifdef CUT
    d = max(d, p.y);
    #endif
    
	return vec2(d,n);        
}

// Function 913
float mapX(vec2 p,  float s){

    return max(length(p)-s, min(abs(p.x-p.y), abs(p.x+p.y)));
}

// Function 914
float seaMap(const in vec3 p) {
    float freq = SEA_FREQ;
    float amp = SEA_HEIGHT;
    float choppy = SEA_CHOPPY;
    vec2 uv = p.xz; uv.x *= 0.75;
    
    float d, h = 0.0;    
    for(int i = 0; i < SEA_ITER_GEOMETRY; i++) {        
    	d = seaOctave((uv+SEA_TIME)*freq,choppy);
    	d += seaOctave((uv-SEA_TIME)*freq,choppy);
        h += d * amp;        
    	uv *= sea_octave_m; freq *= 1.9; amp *= 0.22;
        choppy = mix(choppy,1.0,0.2);
    }
    return p.y - h;
}

// Function 915
float map(vec3 p, out int material)
{
    float d2, d, b = 0.0;    
    mat3 rot = mat3(1);
    float c, c2;
    d2 = d = b = 200000.;
	
    p.z += CAM_DEPTH;
    
    //FRAME BOX
    d = min(d, sdBox(p + vec3(0.,0.,0.5), vec3(1,1,0.5), rot));
    //d = max(d, -udRoundBox(p + vec3(0.,0.,-0.15), vec3(0.9,0.9,0.4), 0.04, rot));
    d = max(d, -sdBox(p + vec3(0.,0.,-0.15), vec3(0.9,0.9,0.4), rot));
    if(d < b){ material = 0; b = d; }
    
    
    //CENTER PANEL
    d = min(d, udRoundBox(p + vec3(-0.03,-0.6,0.3), vec3(0.4,0.2,0.18), 0.02, rot));
    d = min(d, udRoundBox(p + vec3(0.17,-0.2,0.3), vec3(0.2,0.2,0.18), 0.02, rot));
    d = min(d, udRoundBox(p + vec3(-0.03,-0.6,0.26), vec3(0.35,0.05,0.18), 0.02, rot));
    d = min(d, udRoundBox(p + vec3(0.07,-0.7,0.26), vec3(0.25,0.05,0.18), 0.02, rot));
    d = min(d, udRoundBox(p + vec3(0.17,-0.25,0.26), vec3(0.15,0.2,0.18), 0.02, rot));
    if(d < b){ material = 2; b = d; }
    
     //LEFT TWIN PIPES
    rot = mat3(1);
    d = min(d, sdCappedCylinder( p + vec3(0.775, 0.0, 0.16), vec2(0.1, 1.0), rot));
    d = min(d, sdCappedCylinder( p + vec3(0.56, 0.0, 0.16), vec2(0.1, 1.0), rot));
    if(d < b){ material = 3; b = d; }
    
    //UPPER RIGHT BOXES
    d = min(d, udRoundBox(p + vec3(-0.25,-0.2,0.26), vec3(0.12,0.05,0.18), 0.02, rot));
    d = min(d, udRoundBox(p + vec3(-0.7,-0.73,0.26), vec3(0.08,0.08,0.18), 0.02, rot));
    if(d < b){ material = 5; b = d; }
    
    
    //BOTTOM RIGHT PANEL
    d2 = udRoundBox(p + vec3(-0.6,0.4,0.24), vec3(0.2,0.4,0.18), 0.04, rot);
    d2 = fOpEngrave(d2, udRoundBox(p + vec3(-0.6,0.4,0.1), vec3(0.18,0.38,0.18), 0.01, rot), 0.015);
    d2 = min(d2, sdSphere(p  + vec3(-0.6,0.1,0.0), 0.05));
   
    d = min(d, d2);
    if(d < b){ material = 4; b = d; }
    
    //RIGHT BIG PIPE
    rot = rotation(X_AXIS, PI * 0.5);
    d2 = min(d2, sdTorus( p + vec3(-0.75, -0.33, 0.16), vec2(0.15, 0.1), rot ));
    d2 = max(d2, sdBox( p + vec3(-0.6, -0.47, 0.16), vec3(0.13, 0.15, 0.1), mat3(1) ));
    d = min(d, d2);
    rot = rotation(Z_AXIS, PI * 0.5);
    d = min(d, sdCappedCylinder( p + vec3(-0.86, -0.478, 0.16), vec2(0.1, 0.14), rot));
    rot = mat3(1);
    d = min(d, sdCappedCylinder( p + vec3(-0.6, -0.123, 0.16), vec2(0.1, 0.2), rot));
    d = min(d, sdCappedCylinder( p + vec3(-0.6, 0.8, 0.16), vec2(0.1, 0.2), rot));
    if(d < b){ material = 1; b = d; }
        
   
    //TORUS CORE
    rot = rotation(X_AXIS, PI * 0.5);
    d = min(d, sdTorus82( p + vec3(0.05, 0.5, 0.12), vec2(0.25, 0.06), rot));
    d = min(d, sdTorus82( p + vec3(0.05, 0.5, 0.12), vec2(0.08, 0.04), rot));
    if(d < b){ material = 5; b = d; }
    rot = mat3(1);
    d = min(d, udRoundBox(p + vec3(0.05,0.5,0.38), vec3(0.33,0.33,0.18), 0.02, rot));
    if(d < b){ material = 2; b = d; }

    
    d = min(d, sdCappedCylinder( p + vec3(0.05, 0.64, 0.12), vec2(0.03, 0.06), rot));
    rot = rotation(Z_AXIS, PI * 0.75);
    d = min(d, sdCappedCylinder( p + vec3(0.18, 0.38, 0.12), vec2(0.03, 0.06), rot));
    
    rot = rotation(Z_AXIS, PI * -0.85);
    d = min(d, sdCappedCylinder( p + vec3(-0.08, 0.2, 0.12), vec2(0.03, 0.09), rot));
    rot = rotation(Z_AXIS, PI * 0.5);
    d = min(d, sdCappedCylinder( p + vec3(-0.255, 0.13, 0.12), vec2(0.03, 0.15), rot));
        
    rot = rotation(X_AXIS, PI * 0.5);
    d = min(d, sdTorus( p + vec3(0.05, 0.5, 0.05), vec2(0.25, 0.025), rot) );
    if(d < b){ material = 1; b = d; }
        
    //UPPER RIGHT PIPES
    rot = mat3(1);
    d = min(d, sdCapsule( p + vec3(-0.2,-0.2,0.06), vec3(0.,0.,0.), vec3(0.,0.25,0.), 0.04, rot ));
    d = min(d, sdCapsule( p + vec3(-0.3,-0.2,0.06), vec3(0.,0.,0.), vec3(0.,0.25,0.), 0.04, rot ));    
    d = min(d, sdCapsule( p + vec3(-0.3,-0.73,0.06), vec3(0.,0.,0.), vec3(0.4,0.0,0.), 0.04, rot ));
        
    if(d < b){ material = 1; b = d; }
    
    return d;
}

// Function 916
vec3 luvToRgb(vec3 tuple){
    return xyzToRgb(luvToXyz(tuple));
}

// Function 917
vec3 bumpMap(vec3 st){
    vec3 sp = st;
    vec2 eps = vec2(4./iResolution.y, 0.);
    float f = bumpFunc(sp.xy); // Sample value multiplied by the amplitude.
    float fx = bumpFunc(sp.xy-eps.xy); // Same for the nearby sample in the X-direction.
    float fy = bumpFunc(sp.xy-eps.yx); // Same for the nearby sample in the Y-direction.

	const float bumpFactor = 0.1;
    fx = (fx-f)/eps.x; // Change in X
    fy = (fy-f)/eps.x; // Change in Y.
    return vec3(fx,fy,0.)*bumpFactor;
}

// Function 918
vec4   lchToLuv(vec4 c) {return vec4(   lchToLuv( vec3(c.x,c.y,c.z) ), c.a);}

// Function 919
float TonemapFloat( float x )
{
	return 1.0f - exp( -x ); // can change the tonemap function here 
}

// Function 920
float mapDrop( in vec3 p )
{
    p -= vec3(-0.26,0.25,-0.02);
    p.x -= 2.5*p.y*p.y;
    return sdCapsule( p, vec3(0.0,-0.06,0.0), vec3(0.014,0.06,0.0), 0.037 );
}

// Function 921
vec3 tonemap(vec3 linear, float scale) {
	vec3 x = linear * scale;
	x = max(vec3(0), x - 0.004);
	vec3 pre_gamma = (x*(6.2*x+.5))/(x*(6.2*x+1.7)+0.06);
    vec3 post_gamma = pow(pre_gamma, vec3(1.0/2.2));
	return post_gamma;
}

// Function 922
vec3 PBR_HDRremap(vec3 c)
{
    float fHDR = smoothstep(2.900,3.0,c.x+c.y+c.z);
    //vec3 cRedSky   = mix(c,1.3*vec3(4.5,2.5,2.0),fHDR);
    vec3 cBlueSky  = mix(c,1.8*vec3(2.0,2.5,3.0),fHDR);
    return cBlueSky;//mix(cRedSky,cBlueSky,SKY_COLOR);
}

// Function 923
vec3 spheremapUnpack(vec2 n) {
    vec4 nn = vec4(2.0 * n - 1.0, 1.0, -1.0);
    float l = dot(nn.xyz,-nn.xyw);
    nn.z = l;
    nn.xy *= sqrt(l);
    return nn.xyz * 2.0 + vec3(0.0, 0.0, -1.0);
}

// Function 924
vec2 split_uv(vec2 uv) {
    return uv - vec2(0.5*sign(uv.x)*iResolution.x/iResolution.y, 0.0);
}

// Function 925
vec3 tonemapReinhard(vec3 color) {
  float l = czm_luminance(color);
  return color  * l / (l + 1.0);
}

// Function 926
vec2 map( vec3 q )
{
    float td = 0.03 * texCube( iChannel0, q, 0.25).x;
	return vec2(length(q - vec3(0.0, 0.22, 0.0)) - 2.25  + td, td);
}

// Function 927
float waterMap(vec3 p) {
    return p.z - (texture(iChannel0, fract(p.xy)).r+texture(iChannel0, fract(p.xy)).g);
}

// Function 928
SphericalHarmonics CubeMapToRadianceSH() {
    // Initialise sh to 0
    SphericalHarmonics shRadiance = shZero();

    vec2 ts = vec2(textureSize(reflectTex, 0));
    float maxMipMap = log2(max(ts.x, ts.y));

    float lodBias = maxMipMap - 5.0;
    

    for (int i=0; i < ENV_SMPL_NUM; ++i) {
        vec3 direction = SpherePoints_GoldenAngle(float(i), float(ENV_SMPL_NUM));
        vec3 radiance = sampleReflectionMap(direction, lodBias);
        shAddWeighted(shRadiance, shEvaluate(direction), radiance);
    }

    // integrating over a sphere so each sample has a weight of 4*PI/samplecount (uniform solid angle, for each sample)
    float shFactor = 4.0 * PI / float(ENV_SMPL_NUM);
    shScale(shRadiance, vec3(shFactor));

    return shRadiance;
}

// Function 929
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    const vec2 e = vec2(0.001, 0);
    float ref = bumpSurf3D(p);                 
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy),
                      bumpSurf3D(p - e.yxy),
                      bumpSurf3D(p - e.yyx) )-ref)/e.x;                     
          
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
	
}

// Function 930
float map(vec3 p)
{
    float d1 = sdSphere(p, vec3(-2, -1, 0), 1.0);
    float d2 = sdBox(p, vec3(0.5));
    return min(d1, d2);
}

// Function 931
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

// Function 932
vec2 ProjectUV(vec3 p, vec3 LP, vec3 LD, vec3 LT, vec3 LB) {
	vec3 CVPos=p-LP;
    vec3 LVPos=vec3(dot(CVPos,LT),dot(CVPos,LB),dot(CVPos,LD));
    return (LVPos.xy/LVPos.z)*InvAF+0.5;
}

// Function 933
float MapCloud( vec3 p)
{
  return GetHorizon(p) - max(-3., (1.3*GetCloudHeight(p)));
}

// Function 934
vec3 YUVToRGB(vec3 YUVColor)
{
	vec3 ret;

	ret.x = dot(YUVColor, vec3(1.0,  0.0,      1.28033));
	ret.y = dot(YUVColor, vec3(1.0, -0.21482, -0.38059));
	ret.z = dot(YUVColor, vec3(1.0,  2.12798,  0.0));

	return ret;
}

// Function 935
float Tonemap_Lottes(float x) {
    // Lottes 2016, "Advanced Techniques and Optimization of HDR Color Pipelines"
    const float a = 1.6;
    const float d = 0.977;
    const float hdrMax = 8.0;
    const float midIn = 0.18;
    const float midOut = 0.267;

    // Can be precomputed
    const float b =
        (-pow(midIn, a) + pow(hdrMax, a) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    const float c =
        (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    return pow(x, a) / (pow(x, a * d) * b + c);
}

// Function 936
void SmoothCubeMapHorizon(inout vec3 c, const vec3 cm, float d)
{
    float m = exp(-d*d*.001);
    c = mix(cm, c, m);
}

// Function 937
vec4 mapN(vec3 p)
{
    return mapL(p) * noi(p);
}

// Function 938
vec3 doBumpMap( sampler2D tex, in vec3 p, in vec3 nor, float bumpfactor){
   
    const float eps = 0.001;
    float ref = tex3D(tex,  p , nor);                 
    vec3 grad = vec3( tex3D(tex, vec3(p.x-eps, p.y, p.z), nor)-ref,
                      tex3D(tex, vec3(p.x, p.y-eps, p.z), nor)-ref,
                      tex3D(tex, vec3(p.x, p.y, p.z-eps), nor)-ref )/eps;
             
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
	
}

// Function 939
void mainCubemap( out vec4 O, vec2 U, vec3 C, vec3 D )
{
  //O = vec4(.5+.5*D,0); U = U/1024. - 1./vec2(4,8); O -= .01/dot(U,U); return;
  //int f = faceID(D); O = vec4(f&1,f&2,f&4,0); return;
 
    O = vec4( equal( ivec4( 255.* texture(iChannel0, U/iResolution.xy)),
                     ivec4( ( faceID(D) + iFrame*6 ) % 258 ) ) );
}

// Function 940
vec3 Uncharted2Tonemap(vec3 x) {
   return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

// Function 941
vec3 doBumpMap(sampler2D tx, in vec3 p, in vec3 n, float bf){
   
    const vec2 e = vec2(0.001, 0);
    
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    
    vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
    
}

// Function 942
Object map(vec3 pos)
{
    Object o;
    o.difVal = 1.0;
    o.dist = 1000.0;
    o.normEps = 0.00001;
    o.color = vec3(0);
    
    float yOff = 0.05*sin(5.0*iTime);
    vec3 offset = vec3(0, yOff, 0);
    
    //ground
    vec3 boardPos = pos;
    boardPos.z = mod(boardPos.z, 10.0);
    boardPos.x = mod(boardPos.x, 12.0);
	float dBoard = sdRoundBox(boardPos - vec3(0,-3,0), vec3(12, 0.5, 10.0), 0.1);
    if(dBoard < o.dist)
    {
        o.dist = dBoard;
        o.difVal = 0.9;
        
        //checker board
        vec3 col;
        float modi = 2.0*(round(step(sin(pos.z*1.*PI), 0.0)) - 0.5);
        float goldMod = step(-2.0, pos.x) * step(pos.x, 2.0);
        
        col = vec3(1.0*(1.0-goldMod)) + vec3(0.0,0.4,0.6)*goldMod;
        col *= (round(step((modi)*sin(pos.x*1.0*PI), 0.0)));
        
        o.color = col;
        o.specVal = 200.0;
        o.specKs = 0.5;
    }
    
    //tree 
    vec3 treePos = pos;
    vec2 id = floor(vec2(treePos.x/3.5, treePos.z/5.5));
    treePos.x = abs(treePos.x);
    treePos.z = mod(treePos.z, 5.5);
    treePos.x = mod(treePos.x, 7.0);
    treePos -= vec3(5.5, -4.5, 2.5);
    
    float h = sin(id.x) * 337.0 * sin(id.y) * 43.3;
    h = -1.0 + mod(h, 3.0);
    float timeMod = 0.5 + mod(id.x*123.0 / id.y*1234.0, 1.0);
    h *= sin(iTime*1.0 + 43.445*id.y + 122.89*id.x);
    treePos.y -= h;
    
    float treeBound = sdVerticalCapsule(treePos, 5.0, 0.75);
    
    if(treeBound < o.dist)
    {
        
        float dTree = sdVerticalCapsule(treePos, 5.0, 0.5);
        dTree = smin(dBoard, dTree, 0.3);
        if(dTree < o.dist)
        {
            o.dist = dTree;   
            o.difVal = 0.9;
            float modi = 2.0*(round(step(sin(pos.z*PI), 0.0)) - 0.5);
            float yStep = smoothstep(0.0, 0.3, treePos.y);

            vec3 colTrunk = vec3(0.4, 0.3, 0) + vec3(sin(10.0*floor(10.0*treePos.y)))*0.05;
            vec3 col = mix(vec3(1)*(round(step((modi)*sin(pos.x*1.0*PI), 0.0))),colTrunk, yStep);

            o.color = col;
            o.specVal = 200.0;
            o.specKs = 0.0;
        }
        //tree leaves
        vec3 leafPos = pos;
        leafPos.x = abs(leafPos.x);
        leafPos.z = mod(leafPos.z, 5.5);
        leafPos.x = mod(leafPos.x, 7.0);
        leafPos -= vec3(5.5, 1.5, 2.5);
        leafPos.y -= h;

        float dLeaf = sdEllipsoid(leafPos, vec3(1.5, 1.0, 1.5));
        dTree = smin(dTree, dLeaf, 0.5);
        if(dLeaf < o.dist)
        {
            o.dist = dTree;   
            o.difVal = 0.9;

            float modi = 2.0*(round(step(sin(pos.z*1.*PI), 0.0)) - 0.5);

            float yStep = smoothstep(-1.0, -0.8, leafPos.y);
            vec3 col = mix(vec3(0.4, 0.3, 0) ,vec3(0, 0.3, 0), yStep);

            o.color = col;
            o.specVal = 200.0;
            o.specKs = 0.0;
        }
    }
    
    //character bounding box
    float dBBChar = sdSphere(pos - vec3(0,-0.9,0), 1.7);     
    if(dBBChar < o.dist)
    {
        //body
		float dSphere = sdSphere(pos - vec3(0,-0.9,0) + offset, 1.0);
        
        //brows
        float ang = 0.0;
        vec3 browPos = pos;
        browPos.x = abs(browPos.x);
        browPos = browPos - vec3(0.35,-0.5,0.85) + offset;  
        browPos.y -= -2.0*browPos.x *(2.0*browPos.x/2.0);
        mat2 browRot = mat2( vec2(cos(ang), -sin(ang)), vec2(sin(ang), cos(ang)) );
        browPos = vec3(browRot * browPos.xy, browPos.z);
        float dBrow = sdEllipsoid(browPos, vec3(0.24, 0.1, 0.16));
        dSphere = smin(dBrow, dSphere, 0.07);
        
        if(dSphere < o.dist)
        {
            o.dist = dSphere;
            float z = pos.y + 1.0;
            vec3 col = vec3(235.0/255.0, 182.0/255.0, 255.0/255.0);
            col = mix(col,vec3(0.2, 0, .3), (z/2.0));
            o.color = col;
            o.specVal = 55.0;
            o.specKs = 0.04;
        }

        //mouth
        vec3 mouthPos = pos - vec3(0, -1.2, 0.9) + offset;
        mouthPos.y -=  2.0*mouthPos.x * (mouthPos.x/2.0);
        float mouthHeight = 0.02 + 0.1*clamp(sin(iTime/2.0), 0.0, 1.0);
        float dMouth = sdEllipsoid(mouthPos, vec3(0.34, mouthHeight, 0.8));
		if(-dMouth > o.dist)
            o.color = vec3(255.0/255.0, 182.0/255.0, 215.0/255.0) * 0.6;
        o.dist = max(o.dist, -dMouth);
        



        //hair sdRoundBox( vec3 p, vec3 b, float r )
        vec3 hairPos = pos - vec3(0, 0.1, 0);
        hairPos.y -= -hairPos.z * (hairPos.z/2.0);
        hairPos.y -= 0.05*sin(hairPos.z*25.0);
        hairPos += offset;
        float dHair = sdRoundBox(hairPos, vec3(0.1, 0.2, 0.7), 0.05);
        if(dHair < o.dist)
        {
            o.dist = dHair;
            //o.color = vec3(0.5, 1.0, 0.5);
            o.color = vec3(1, 0.5, 0.5) + vec3(0, hairPos.y*1.53, 0);
            o.specVal = 2.0;
            o.specKs = 0.0;
        }

        //add bobbing and swinging animation
        //

        //feet
        ang = -PI/4.0;
        vec3 footPos = pos; 
        footPos.x = abs(footPos.x);
        mat2 footRot = mat2( vec2(cos(ang), -sin(ang)), vec2(sin(ang), cos(ang)) );
        vec2 footXZ = footRot * footPos.xz;
        footPos = vec3(footXZ.x, pos.y, footXZ.y);
        float dFoot = sdEllipsoid(footPos - vec3(0.3,-2.3,0.6), vec3(0.3, 0.3, 0.4));
        if(dFoot < o.dist)
        {
            o.dist = dFoot;
            o.color = vec3(0.5, 0., 0.);
            o.specVal = 2.0;
            o.specKs = 0.4;
        }

        //hands
        float hAng = PI/2.0;
        vec3 handPos = pos;
        float modi = handPos.x / abs(handPos.x);
        handPos.x = abs(handPos.x);
        
        handPos = handPos - vec3(1.35+offset.y,-1.5,0.0);
        //handPos += offset;
        //handPos = opCheapBend(handPos);
        mat2 handRot = mat2( vec2(cos(hAng), sin(hAng)), vec2(-sin(hAng), cos(hAng)) );
        vec2 handXZ = handRot * handPos.xz;
        handPos = vec3(handXZ.x, handPos.y, handXZ.y);
        //handRot = mat2( vec2(cos(hAng), -sin(hAng)), vec2(sin(hAng), cos(hAng)) );
        //handPos = vec3(handPos.x, handRot * handPos.yz);
        float dHand = sdEllipsoid(handPos, vec3(0.3, 0.35, 0.23));
        if(dHand < o.dist)
        {
            o.dist = dHand;
            o.color = vec3(1);
            o.specVal = 50.0;
            o.specKs = 0.4;
        }

        //eyes
        vec3 eyePos = pos;
        eyePos.x = abs(eyePos.x);
        eyePos += offset;
        float dEye = sdSphere(eyePos - vec3(0.34,-0.7,0.8), 0.2);
        if(dEye < o.dist)
        {
            o.dist = dEye;
            o.color = vec3(1);
            o.specVal = 100.0;
            o.specKs = 2.0;
        }

        //pupils
        vec3 pupPos = pos;
        pupPos.x = abs(pupPos.x);
        pupPos += offset;
        float dEyePup = sdSphere(pupPos - vec3(0.32,-0.7,0.94), 0.08);
        if(dEyePup < o.dist)
        {
            o.dist = dEyePup;
            o.color = vec3(0);
            o.specVal = 100.0;
            o.specKs = 2.0;
        }
        
        //eye lid
        vec3 lidPos = pos;
        //lidPos.y = clamp(lidPos.y, -0.8,-0.5);
        lidPos.x = abs(lidPos.x);
        lidPos += offset;
        float dLid = sdSphere(lidPos - vec3(0.34,-0.7,0.8), 0.225);
        
        //consulted IQ's happy jumping for a similar blink rate function
        if(dLid < o.dist && lidPos.y > 1.0 - 2.0*pow(sin(iTime),40.0))
        {
            o.dist = dLid;
            o.color = vec3(235.0/255.0, 182.0/255.0, 255.0/255.0);
            o.specVal = 55.0;
            o.specKs = 0.04;
        }
    
    }
    
    
    return o;
}

// Function 943
void mainCubemap( out vec4 O, in vec2 I, in vec3 rayOri, in vec3 rayDir )
{
    if(rayDir.x<abs(rayDir.y)||rayDir.x<abs(rayDir.z)) return;
    vec2 major = floor(I.yx/32.);
    vec2 minor = floor(mod(I.yx,vec2(32.)));
    vec4 A = texelFetch(iChannel0, ivec2(minor),0);
    vec4 B = texelFetch(iChannel0, ivec2(major),0);
    vec2 D = A.xy-B.xy;
    if(length(D)>1.){
        
    	//O.xy = 2e3*normalize(D)/ dot(D,D);
    	O.xy = 1e2*D/ dot(D,D);
    } else {
        O.xy = vec2(0.);
    }
    
}

// Function 944
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    const vec2 e = vec2(2.201, 0);
    float ref = bumpSurf3D(p, nor);                 
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy, nor),
                      bumpSurf3D(p - e.yxy, nor),
                      bumpSurf3D(p - e.yyx, nor) )-ref)/e.x;                     
          
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
	
}

// Function 945
vec4 map(in vec2 p, in vec2 dir) {
    return doMap(voronoi(p*2.0, dir));
}

// Function 946
void mainCubemap( out vec4 fragColor, in vec2 p, in vec3 rayOri, in vec3 rayDir )
{
    //last frame
    vec4 last = T(p,0);
    
    //mouse position on cubemap
    vec2 m = viewportToCube(iMouse.xy);
    
    bool mouseCircle = (1.0-step(15.0/zoom,distance(m,p)))>0.5;
    
    bool space = texelFetch(iChannel2,ivec2(32, 0), 0).x > 0.0;
    bool shift = texelFetch(iChannel2,ivec2(16, 0), 0).x > 0.0;

    bool mask = mouseCircle && space;

    //clear space around ray origin
    if(distance(p,rayOrigin)<50.0/zoom)mask = false; 

    mask = mask || (last.r>0.5);
    
    //erase
    if(shift&&mouseCircle){mask = false;}
    
    //initial noise
    if(iFrame<25){
    
        mask = texture(iChannel3,p/1024.0).g+distance(p,rayOrigin)*0.0005<0.1;
        
        fragColor = vec4(mask,lods,1.0,1.0); return;
    }
    
    
    int largestEmptyLOD = lods;
    
    //find the largest empty LOD (one frame behind)
    for(int i = 0; i <= lods; i++){
    
        if(T(p,i).r>0.0){largestEmptyLOD = max(i-1,0); break;}
    }
    
    //the size of the largest empty LOD
    float size = exp2(float(largestEmptyLOD));
    
    
    // Output to cubemap
    fragColor = vec4(mask,size,1.0,1.0);
}

// Function 947
float map_floor(vec3 pos)
{
    return pos.y;
}

// Function 948
vec4 map( in vec3 pos, in float time, float doDisplace )
{
    time = time*3.0;

    // body
	vec3 bpos = pos;
    bpos.y -= 0.3*sqrt(0.5-0.5*cos(time*2.0+1.0));
    bpos.x -= 0.1;
    bpos.y += 0.35;
    bpos.x -= 0.2*pow(0.5+0.5*cos(time*2.0+0.5),2.0);
    vec3 tpos = bpos - vec3(-0.1,0.45,0.0);
    bpos.xy = -bpos.xy;
    vec4 res2 = vec4(sdEllipsoid(tpos,vec3(0.3,0.7,0.45)),bpos);
    
	// legs
#if 0
    vec4 l1 = leg( bpos-vec3(0.0,0.0, 0.27), 3.1416+time );
    vec4 l2 = leg( bpos-vec3(0.0,0.0,-0.27), time );
    vec4 res = dmin(l1,l2);
    res.w += 0.27*sign(l2.x-l1.x);
#else
    // trick to prevent inlining - compiles faster
    vec4 dl[2];
    for( int i=ZERO; i<2; i++ )
       dl[i] = leg( bpos-vec3(0.0,0.0,((i==0)?1.0:-1.0)*0.27), ((i==0)?3.1416:0.0)+time );
    vec4 res = dmin(dl[0],dl[1]);
    res.w += 0.27*sign(dl[1].x-dl[0].x);
#endif    
        
    res = smin( res, res2, 0.08 );

	// displacement
    float di = disp(res.yzw, sign(pos.z));
	float tempo = 0.5 + 0.5*sin(time);
    tempo = 0.5*tempo + 0.5*tempo*tempo*(3.0-2.0*tempo);
    float an0 = mix(1.0,0.0,tempo);
    di *= 0.8 + 1.7*an0*(smoothstep(-0.6,0.40,res.z)-smoothstep(0.8,1.4,res.z));
	di *= 1.0-smoothstep(1.9,1.91,res.z);
    res.x += (0.015-0.03*di)*doDisplace;
    res.x *= 0.85;

    return res;
}

// Function 949
float Tonemap_ACES(float x) {
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

// Function 950
vec4 MapTerrainReflections( vec3 p)
{
    treeDist = 10000.;
  float boatDist= 10000.;
  float bridgeDist=10000.;
  float height = GetTerrainHeight(p); 
  float tHeight= height + GetStoneHeight(p.xz, height);
  tHeight*=1.4;
  if (tHeight>0.)
  {
    tHeight +=textureLod( iChannel1, p.xz*.2, 0.2 ).x*.03;  
               
      #ifdef TREES   
      vec3 treePos = p-vec3(0.,tHeight+2.,0.);
      vec2 mm = floor( treePos.xz/8.0 );	
	treePos.xz = mod( treePos.xz, 8.0 ) - 4.0;

      float treeHeight=GetTreeHeight(mm,p.xz, tHeight);
      
      if(treeHeight>0.05)
      {             
          treeDist = sdEllipsoid(treePos,vec3(2.,5.7,2.));
                     treeDist+=(noise(p*1.26)*.6285);
         treeDist+=(noise(p*3.26)*.395);
           treeDist+=(noise(p*6.26)*.09825);
      }
    #endif
  }
  #ifdef BRIDGE
    bridgeDist=MapBridge(p);   
  #endif
    #ifdef BOAT
    #ifdef ACCURATE_BOAT_REFLECTION
    boatDist=MapBoat(p); 
    #else
    // fake boat by using ellipsoid
    boatDist=sdEllipsoid( TranslateBoat(p)- vec3(0, -0.20, -1.0), vec3(1.65, 1., 3.40));
    #endif
    
  #endif

    // mask tower position by placing a cone
    return  vec4(min(treeDist,min(min(boatDist, bridgeDist), min(p.y - max(tHeight, 0.), sdConeSection(p-vec3(-143, 0., 292)-vec3(0., 13., 0.), 10.45, 3.70, 1.70)))), boatDist, bridgeDist, tHeight);
}

// Function 951
vec2 warpUV(in vec2 p)
{
    vec2 q =mix(p, vec2(fbm(p + iTime/20.0), fbm(p+ iTime/5.0)), 0.05);
    
    return q;
}

// Function 952
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    // Larger sample distances give a less defined bump, but can sometimes lessen the aliasing.
    const vec2 e = vec2(0.001, 0); 
    
    // Gradient vector: vec3(df/dx, df/dy, df/dz);
    float ref = bumpSurf3D(p);
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy),
                      bumpSurf3D(p - e.yxy),
                      bumpSurf3D(p - e.yyx)) - ref)/e.x; 
    
    /*
    // Six tap version, for comparisson. No discernible visual difference, in a lot of cases.
    vec3 grad = vec3(bumpSurf3D(p - e.xyy) - bumpSurf3D(p + e.xyy),
                     bumpSurf3D(p - e.yxy) - bumpSurf3D(p + e.yxy),
                     bumpSurf3D(p - e.yyx) - bumpSurf3D(p + e.yyx))/e.x*.5;
    */
       
    // Adjusting the tangent vector so that it's perpendicular to the normal. It's some kind 
    // of orthogonal space fix using the Gram-Schmidt process, or something to that effect.
    grad -= nor*dot(nor, grad);          
         
    // Applying the gradient vector to the normal. Larger bump factors make things more bumpy.
    return normalize(nor + grad*bumpfactor);
	
}

// Function 953
float water_map( vec2 p, float height )
{
  vec2 p2 = p*large_wavesize;
  vec2 shift1 = 0.001*vec2( iTime*160.0*2.0, iTime*120.0*2.0 );
  vec2 shift2 = 0.001*vec2( iTime*190.0*2.0, -iTime*130.0*2.0 );

  // coarse crossing 'ocean' waves...
  float f = 0.6000*noise( p );
  f += 0.2500*noise( p*m );
  f += 0.1666*noise( p*m*m );
  float wave = sin(p2.x*0.622+p2.y*0.622+shift2.x*4.269)*large_waveheight*f*height*height ;

  p *= small_wavesize;
  f = 0.;
  float amp = 1.0, s = .5;
  for (int i=0; i<9; i++)
  { p = m*p*.947; f -= amp*abs(sin((noise( p+shift1*s )-.5)*2.)); amp = amp*.59; s*=-1.329; }
 
  return wave+f*small_waveheight;
}

// Function 954
vec2 map(vec3 p) {
    
    vec2 sphereObj =  vec2(sdSphere(p - spheres[1].p, spheres[1].r), SPHERE_ID1);        
    vec2 resultObj = sphereObj;

    vec2 newObj =  vec2(sdSphere(p - spheres[2].p, spheres[2].r), SPHERE_ID2);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdSphere(p - spheres[3].p, spheres[3].r), SPHERE_ID3);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdSphere(p - spheres[4].p, spheres[4].r), SPHERE_ID4);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdSphere(p - spheres[5].p, spheres[5].r), SPHERE_ID5);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdSphere(p - spheres[6].p, spheres[6].r), SPHERE_ID6);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdPlane(p - planes[0].p, planes[0].n), FLOOR_ID);
    resultObj = opU(resultObj, newObj);
    newObj =  vec2(sdSphere(p - spheres[0].p, spheres[0].r), LIGHT_ID);
    resultObj = opU(resultObj, newObj);
    
    return resultObj;
}

// Function 955
vec4 mapL(vec3 p)
{
	p.xy -= path(p.z);													// tunnel path
	
    // mix from displace of last section id with displace of current section id accroding to id range 
    float r = mix(displace(p, lid), displace(p, cid), fract(cid)); 	// id range [0-1]
	
    p *= getRotZMat(p.z*0.05);
	
   	p = mod(p, 10.) - 5.;
    
    return vec4(abs(p.y)+2. - 1. + r * .9,p);
}

// Function 956
float map(vec3 q){
    
    // basic tri planar add
    float dsp = 0.02;
    q.z += dot(texture(iChannel1, q.xy).rgb, vec3(dsp));
	q.x += dot(texture(iChannel1, q.yz).rgb, vec3(dsp));
	q.y += dot(texture(iChannel1, q.xz).rgb, vec3(dsp));
	
    // Layer one. The ".05" on the end varies the hole size.
 	vec3 p = abs(fract(q/3.)*3. - 1.5);
 	float d = min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1. + .05;
    
    // Layer two.
    p =  abs(fract(q) - .5);
 	d = max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1./3. + .05);
   
    // Layer three. 3D space is divided by two, instead of three, to give some variance.
    p =  abs(fract(q*2.)*.5 - .25);
 	d = max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - .5/3. - .015); 

    // Layer four. The little holes, for fine detailing.
    p =  abs(fract(q*3./.5)*.5/3. - .3/6.);
 	return max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1./18. - .015);
    //return max(d, max(max(p.x, p.y), p.z) - 1./18. - .024);
    //return max(d, length(p) - 1./18. - .048);
    
    //p =  abs(fract(q*3.)/3. - .5/3.);
 	//return max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1./9. - .04);
}

// Function 957
float asteroidMapDetailed( const in vec3 p, const in vec3 id) {
    float d = asteroidRock(p, id) + fbm(p*4.0,0.4,2.96) * ASTEROID_DISPLACEMENT;
    return d;
}

// Function 958
vec3 TonemapFilmicALU(vec3 x)
{
    vec3 c = max(vec3(0.0), x - 0.004);
    return (c * (c * 6.2 + 0.5)) / (c * (c * 6.2 + 1.7) + 0.06);
}

// Function 959
vec2 uv_aa_linear( vec2 uv, vec2 res, float width )
{
    uv = uv * res;
    vec2 uv_floor = floor(uv + 0.5);
    uv = uv_floor + clamp( (uv - uv_floor) / fwidth(uv) / width, -0.5, 0.5);
    return uv / res;
}

// Function 960
float endlessFloorMap() {
    return abs(-10.0-rayPos.y);
}

// Function 961
float MapMissile(vec3 p, Missile missile)
{
  float d= fCylinder( p, 0.70, 1.7);
  if (d<1.0)
  {
    d = fCylinder( p, 0.12, 1.2);   
    d =min(d, sdEllipsoid( p- vec3(0, 0, 1.10), vec3(0.12, 0.12, 1.0))); 

    checkPos = p;  
    pR(checkPos.xy, 0.785);
    checkPos.xy = pModPolar(checkPos.xy, 4.0);

    d=min(d, sdHexPrism( checkPos-vec3(0., 0., .60), vec2(0.50, 0.01)));
    d=min(d, sdHexPrism( checkPos+vec3(0., 0., 1.03), vec2(0.50, 0.01)));
    d = max(d, -sdBox(p+vec3(0., 0., 3.15), vec3(3.0, 3.0, 2.0)));
    d = max(d, -fCylinder(p+vec3(0., 0., 2.15), 0.09, 1.2));
  }
  return d;
}

// Function 962
float map2( in vec3 pos )
{
    return min( pos.y+1.0, map(pos).x );
}

// Function 963
vec3 envMap(vec3 p){
   
    // Some functions work, and others don't. The surface is created with the function
    // below, so that makes it somewhat believable.
    float c = cellTile(p*6.);
    c = smoothstep(0.2, 1., c); // Contract gives it more of a lit look... kind of.
    
    return vec3(pow(c, 8.), c*c, c); // Icy glow... for whatever reason. :)
    // Alternate firey glow.
    //return vec3(min(c*1.5, 1.), pow(c, 2.5), pow(c, 12.));

}

// Function 964
vec2 map( in vec3 pos, float time )  
{
    float id = 0.;
    float sphere = sdSphere(pos, 0.5);
    float plane = pos.y + 0.5;
    
    if(plane < sphere) {
    	id = 1.;    
    }
    
    float d = min(sphere, plane);
    
    return vec2(d, id);
}

// Function 965
void mainCubemap( out vec4 O, vec2 U, vec3 C, vec3 D )
{
    // --- part indentical on the 6 faces

    U /= iResolution.xy;
    O  = texture(iChannel0, U );           // source image
    float v = 5.*max(0., O.r - O.g );      // stencil to analyze ( ~= flech )

    
    // --- part specific to each of the 6 faces. https://www.shadertoy.com/view/Xlcczj
    
    vec3 A = abs(D); // seek for max direction: i = invmax(abs(D[i]))
    int i=0; 
    float      M = A.x; 
    if (A.y>M) M = A.y, i=1;
    if (A.z>M) M = A.z, i=2;
    int  faceID = i + 3* int(D[i]<0.);
    
    if (faceID==0) // --- raw image
        O.a = v;   // ( redundant )
    if (faceID==1) // --- compute M1 moments E(x),E(y) and M0=E(I) in ultimate MIPmap LOD. (M1 to be normalized by M0) 
        O = vec4( U, 1, 0 ) * v;
    if (faceID==2) // --- compute M2 matrix E(xx),E(yy),E(xy) in ultimate MIPmap LOD (to be normalized by Imean)  
        O = vec4( U*U, U.x*U.y, 0 ) * v;   
    if (faceID>2)  // --- don't need last faces 
        O -= O;
}

// Function 966
float map(vec3 p) {
  float r = iMouse.z > 0.0 ? iMouse.x / 100.0 : iTime * 0.9;
  p.xz = mirror(p.xz, 4.);
  p.xz = rotate2D(p.xz, r);
  float d = sdBox(p, vec3(1));
  d = min(d, sdBox(p, vec3(0.1, 0.1, 3)));
  d = min(d, sdBox(p, vec3(3, 0.1, 0.1)));
  return d;
}

// Function 967
vec3 mapD2(float t)
{
    return 14.0*a*c*sin(t+m)*sin(7.0*t+n) - a*cos(t+m)*(b+c*cos(7.0*t+n)) - 49.0*a*c*cos(t+m)*cos(7.0*t+n);
}

// Function 968
float errorCheckUV(vec2 checkUV){
  return float((checkUV.x < .0)||(checkUV.y < .0)||(checkUV.x > 1.)||(checkUV.y > 1.));
}

// Function 969
vec4 rgbToHsluv_(vec4 c)
{
    return vec4(
        rgbToHsluv(
            pow(c.rgb,GAMMA)
        )/vec3(360.,100.,100.)
    ,c.a);
}

// Function 970
float map(vec3 p) {
    
    
    // Gaz's path correction. Very handy.
    vec2 pth = path(p.z);
    vec2 dp = (path(p.z + .1) - pth)/.1; // Incremental path diffence.
    vec2 a = cos(atan(dp)); 
    // Wrapping a tunnel around the path.
    float tun = length((p.xy - pth)*a);
    
    
    
    // Obtaining the distanc field values from the 3D data packed into
    // the cube map face. These have been smoothly interpolated.
    vec3 tx3D = texMapSmooth(iChannel0, p/3.).xyz;
    // Using this will show you why interpolation is necessary.
    //vec3 tx3D = tMap(iChannel0, p/3.).xyz;
    
    // The main surface. Just a couple of gradient noise layers. This is used
    // as a magnetic base to wrap the asteroids around.
    float main = (tx3D.x - .55)/2.;
    
    // Calling the function again, but at a higher resolution, for the other
    // surfaces, which consist of very expensive rounded Voronoi.
    tx3D = texMapSmooth(iChannel0, p*2.).xyz;
    
    // Saving the higher resolution gradient noise to add some glow. I patched
    // this in at the last minute.
    glow3 = tx3D;

    
    // Attaching the asteroid field to the gradient surface. Basically, the 
    // rocks group together in the denser regions. With doing this, you'd 
    // end up with a constant density mass of rocks.
    main = smax(main, -(tx3D.z + .05)/6., .17);
    
    // Adding a heavy layer of gradient noise bumps to each rock.
    main += (abs(tx3D.x - .5)*2. - .15)*.04;
   
    // Smoothly running the tunnel through the center, to give the camera
    // something to move through -- Otherwise, it'd bump into rocks. Getting 
    // a tunnel to run through a group of rocks without warping them was 
    // only possilbe because of the way the rocks have been constructed.
    return smax(main, -tun, .25);
    
}

// Function 971
vec3 yuv2rgb(vec3 YUV)
{
	float r = dot( YUV, vec3( 1, 0, 1.13983) );
	float g = dot( YUV, vec3( 1, -0.39465, -0.58060 ) );
	float b = dot( YUV, vec3( 1, 2.03211, 0 ));
	return vec3( r, g, b );
}

// Function 972
float mapArloSimple( vec3 p )
{

    // body
    vec3 q = p;
    float co = cos(0.2);
    float si = sin(0.2);
    q.xy = mat2(co,si,-si,co)*q.xy;
    float d1 = sdEllipsoid( q, vec3(0.0,0.0,0.0), vec3(1.3,0.75,0.8) );
    float d2 = sdEllipsoid( q, vec3(0.05,0.45,0.0), vec3(0.8,0.6,0.5) );
    float d = smin( d1, d2, 0.4 );
    
    // tail
    {
    vec2 b = sdBezier( vec3(1.0,-0.4,0.0), vec3(2.0,-0.96,-0.5), vec3(3.0,-0.5,1.5), p );
    float tr = 0.3 - 0.25*b.y;
    float d3 = b.x - tr;
    d = smin( d, d3, 0.2 );
    }
    
    // neck
    {
    vec2 b = sdBezier( vec3(-0.9,0.3,0.0), vec3(-2.2,0.5,0.0), vec3(-2.6,1.7,0.0), p );
    float tr = 0.35 - 0.23*b.y;
    float d3 = b.x - tr;
    d = smin( d, d3, 0.15 );
    //d = min(d,d3);
	}


    float dn;
    // front-left leg
    {
    vec2 d3 = legSimple( p, vec3(-0.8,-0.1,0.5), vec3(-1.5,-0.5,0.65), vec3(-1.9,-1.1,0.65), 1.0, 0.0 );
    d = smin(d,d3.x,0.2);
    dn = d3.y;
    }
    // back-left leg
    {
    vec2 d3 = legSimple( p, vec3(0.5,-0.4,0.6), vec3(0.3,-1.05,0.6), vec3(0.8,-1.6,0.6), 0.5, 1.0 );
    d = smin(d,d3.x,0.2);
    dn = min(dn,d3.y);
    }
    // front-right leg
    {
    vec2 d3 = legSimple( p, vec3(-0.8,-0.2,-0.5), vec3(-1.0,-0.9,-0.65), vec3(-0.7,-1.6,-0.65), 1.0, 1.0 );
    d = smin(d,d3.x,0.2);
    dn = min(dn,d3.y);
    }
    // back-right leg
    {
    vec2 d3 = legSimple( p, vec3(0.5,-0.4,-0.6), vec3(0.8,-0.9,-0.6), vec3(1.6,-1.1,-0.7), 0.5, 0.0 );
    d = smin(d,d3.x,0.2);
    dn = min(dn,d3.y);
    }
    
    
    // head
    vec3 s = vec3(p.xy,abs(p.z));
    {
    vec2 l = sdLine( p, vec3(-2.7,2.36,0.0), vec3(-2.6,1.7,0.0) );
    float d3 = l.x - (0.22-0.1*smoothstep(0.1,1.0,l.y));
        
    // mouth
    //l = sdLine( p, vec3(-2.7,2.16,0.0), vec3(-3.35,2.12,0.0) );
    vec3 mp = p-vec3(-2.7,2.16,0.0);
    l = sdLine( mp*vec3(1.0,1.0,1.0-0.2*abs(mp.x)/0.65), vec3(0.0), vec3(-3.35,2.12,0.0)-vec3(-2.7,2.16,0.0) );
        
    float d4 = l.x - (0.12 + 0.04*smoothstep(0.0,1.0,l.y));      
    float d5 = sdEllipsoid( s, vec3(-3.4,2.5,0.0), vec3(0.8,0.5,2.0) );
    d4 = smax( d4, d5, 0.03 );
    
        
    d3 = smin( d3, d4, 0.1 );

        
    // mouth bottom
    {
    vec2 b = sdBezier( vec3(-2.6,1.75,0.0), vec3(-2.7,2.2,0.0), vec3(-3.25,2.12,0.0), p );
    float tr = 0.11 + 0.02*b.y;
    d4 = b.x - tr;
    d3 = smin( d3, d4, 0.001+0.06*(1.0-b.y*b.y) );
    }
        
    // brows    
    vec2 b = sdBezier( vec3(-2.84,2.50,0.04), vec3(-2.81,2.52,0.15), vec3(-2.76,2.4,0.18), s+vec3(0.0,-0.02,0.0) );
    float tr = 0.035 - 0.025*b.y;
    d4 = b.x - tr;
    d3 = smin( d3, d4, 0.025 );

    d = smin( d, d3, 0.01 );
    }
    
    return min(d,dn);
}

// Function 973
vec3 lchToHsluv(float x, float y, float z) {return lchToHsluv( vec3(x,y,z) );}

// Function 974
endif
map( in vec3 pos )
{    
    float angle4 = iTime*TAU*0.25;
    
    MPt res;
    #if INDEXED_MATERIALS
    res.x = 1e38;
    #else
    res.distance = 1e38;
    #endif
    
    #if !INDEXED_MATERIALS
    Material plastic_m;
    plastic_m.color = vec3(1.0);
    plastic_m.diffuse_reflection  = 1.0;
    plastic_m.specular_reflection = 1.0;
    plastic_m.ambient_reflection  = 1.0;
    plastic_m.shininess           = 15.0;

    Material floor_m;
    plastic_m.color = vec3(1.0);
    floor_m.diffuse_reflection  = 1.0;
    floor_m.specular_reflection = 0.2;
    floor_m.ambient_reflection  = 0.5;
    floor_m.shininess           = 1.0;

    Material orange_m = plastic_m;
    orange_m.color = ORANGE_RGB;
    
    Material red_m = plastic_m;
    red_m.color = vec3(1.0,0.0,0.0);
    Material green_m = plastic_m;
    green_m.color = vec3(0.0,1.0,0.0);
    Material blue_m = plastic_m;
    blue_m.color = vec3(0.0,0.0,1.0);
    #endif

    float m = mod( floor(pos.x * 2.0) + floor(pos.y * 2.0), 2.0 );
    res = union_op( MPt( plane_sd( pos ),
                        #if INDEXED_MATERIALS
                        MAT_FLOOR_B * (1.0 - m) + MAT_FLOOR_W * m
                        #else
                         change_color( floor_m, vec3( 0.7 + 0.3 * m ) )
                        #endif
                       ),
                    res );
    #if 1
    res = union_op( MPt( aab_sd( vec3(1.0), at_angle( vec3(0.0,pos.y * HPI,0.0), at_pos(vec3(-2.0,pos.y,3.5),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                       ),
                    res );
    #endif
    
        
    res = union_op( MPt( sphere_sd( 0.1, at_pos( vec3(0.0        ), pos ) ),
                        #if INDEXED_MATERIALS
                         MAT_PLASTIC
                        #else
                        plastic_m
                        #endif
                       ), res );
    res = union_op( MPt( sphere_sd( 0.1, at_pos( vec3(1.0,0.0,0.0), pos ) ),
                        #if INDEXED_MATERIALS
                         MAT_RED
                        #else
                        red_m
                        #endif
                       ),res );
    res = union_op( MPt( sphere_sd( 0.1, at_pos( vec3(0.0,1.0,0.0), pos ) ),
                        #if INDEXED_MATERIALS
                         MAT_GREEN
                        #else
                        green_m
                        #endif
                       ),res );
    res = union_op( MPt( sphere_sd( 0.1, at_pos( vec3(0.0,0.0,1.0), pos ) ),
                        #if INDEXED_MATERIALS
                         MAT_BLUE
                        #else
                        blue_m
                        #endif
                       ),res );

    res = union_op( MPt( sphere_sd( 0.5, at_pos( vec3(2.0,-4.0,0.5), pos ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                        orange_m
                        #endif
                       ), res );

    res = union_op( MPt( round_aab_ud( vec3(0.9), 0.05, at_pos( vec3(2.0,-2.0,0.4), pos ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                       ),
                    res );

    res =
       union_op( MPt( aab_sd( vec3(1.0), at_angle( vec3(0.0,0.0,pos.z * HPI * sin(iTime)), at_pos(vec3(2.0,0.0,0.5),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    res = union_op( MPt( aab_sd( vec3(1.0), at_angle( vec3(0.0,0.0,iTime*0.05), at_pos(vec3(2.0,2.0,0.5),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                       ),
                    res );
#if ALL_PRIMITIVES
    
    res =
       union_op( MPt( torus_sd( vec2(0.4,0.1), at_angle( vec3(0.0,0.0,0.0), at_pos(vec3(2.0,4.0,0.1),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( torus_sd( vec2(0.38,0.12), at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,6.0,0.5),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( torus_sd( vec2(0.38,0.12), at_angle( vec3(0.0,HPI, iTime + TAU * pos.z), at_pos(vec3(2.0,8.0,0.5),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( cylinder_sd( 0.5, at_angle( vec3(0.0,/*sin(iTime*TAU/7.0)**/QPI*0.25,angle4), at_pos(vec3(2.0,10.0,2.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( capped_cylinder_sd( vec2( 0.5, 1.0 ), at_angle( vec3(0.0,0.25*QPI*iTime,angle4), at_pos(vec3(2.0,12.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( cone_sd( vec2( 0.5, 1.0 ),
                               at_angle( vec3(0.0,0.125*PI*sin(iTime*TAU/11.0),iTime*TAU/17.0),
                                         at_pos(vec3(2.0,14.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( half_cone_pos_sd( vec2( 0.5, 1.0 ),
                                        at_angle( vec3(0.0,0.125*PI*sin(iTime*TAU/11.0),iTime*TAU/17.0),
                                                  at_pos(vec3(2.0,16.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( half_cone_pos_sd( vec2( 0.5, 1.0 ),
                                        vec3(1.0,1.0,-1.0) *
                                        at_angle( vec3(0.0,0.125*PI*sin(iTime*TAU/11.0),iTime*TAU/17.0),
                                                  at_pos(vec3(2.0,18.0,1.0), pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op(
          intersect_op(
             MPt( half_cone_pos_sd( vec2( 0.5, 1.0 ),
                                    vec3(1.0,1.0,-1.0) * at_pos(vec3(2.0,20.0,2.0), pos) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
             intersect_op(
                MPt( half_space_sd( at_pos(vec3(2.0,20.0,1.5), pos) ),
                     #if INDEXED_MATERIALS
                      MAT_ORANGE
                     #else
                      orange_m
                     #endif
                ),
                MPt( -half_space_sd( at_pos(vec3(2.0,20.0,0.5), pos) ),
                     #if INDEXED_MATERIALS
                      MAT_ORANGE
                     #else
                      orange_m
                     #endif
                )
             )
          ),
          res );
    
    res =
       union_op( MPt( capped_cone_as_intersections_sd(
                         1.0, 0.25, 0.5, at_pos(vec3(2.0,22.0,0.5), pos) ),
                      #if INDEXED_MATERIALS
                       MAT_ORANGE
                      #else
                       orange_m
                      #endif
                    ),
                 res );  
    res =
       union_op( MPt( capped_cone_as_intersections_sd(
                         1.0, 0.5, 0.25, at_pos(vec3(2.0,24,0.5), pos) ),
                      #if INDEXED_MATERIALS
                       MAT_ORANGE
                      #else
                       orange_m
                      #endif
                    ),
                 res );  
    res =
       union_op( MPt( capped_cone_as_intersections_sd(
                         1.0, 0.25, 0.5, at_pos(vec3(2.0,24,1.6 + 0.05 * sin(angle4) ), pos) ),
                      #if INDEXED_MATERIALS
                       MAT_ORANGE
                      #else
                       orange_m
                      #endif
                    ),
                 res );
    res =
       union_op( MPt( hex_prism_sd(vec2(0.5,0.5), at_angle( vec3(angle4,0.0,0.0), at_pos(vec3(2.0,26,0.5), pos) ) ),
                      #if INDEXED_MATERIALS
                       MAT_ORANGE
                      #else
                       orange_m
                      #endif
                    ),
                 res );
    res =
       union_op( MPt( hex_prism_sd(vec2(0.5,1.0), at_angle( vec3(angle4,0.0,0.0), at_pos(vec3(2.0,28,0.866025), pos) ) ),
                      #if INDEXED_MATERIALS
                       MAT_ORANGE
                      #else
                       orange_m
                      #endif
                    ),
                 res );
    res =
       union_op( MPt( tri_prism_sd(vec2(0.5,1.0), at_angle( vec3(angle4,0.0,0.0), at_pos(vec3(2.0,30.0,0.0), pos) ) ),
                      #if INDEXED_MATERIALS
                       MAT_ORANGE
                      #else
                       orange_m
                      #endif
                    ),
                 res );
    res =
       union_op( MPt( tri_prism_bary_sd(vec2(0.5,1.0), at_angle( vec3(angle4,0.0,0.0), at_pos(vec3(2.0,32.0,0.288675135), pos) ) ),
                      #if INDEXED_MATERIALS
                       MAT_ORANGE
                      #else
                       orange_m
                      #endif
                    ),
                 res );
    res = union_op( MPt( sphere_sd( 0.05, at_pos( vec3(2.25,32.0,0.288675135), pos ) ),
                        #if INDEXED_MATERIALS
                         MAT_PLASTIC
                        #else
                        plastic_m
                        #endif
                       ), res );
    res =
       union_op( MPt( tri_prism_bary_r_sd(vec2(0.5,0.5), at_angle( vec3(angle4,0.0,0.0), at_pos(vec3(2.0,34.0,0.5), pos) ) ),
                      #if INDEXED_MATERIALS
                       MAT_ORANGE
                      #else
                       orange_m
                      #endif
                    ),
                 res );
    res = union_op( MPt( sphere_sd( 0.05, at_pos( vec3(2.25,34.0,0.5), pos ) ),
                        #if INDEXED_MATERIALS
                         MAT_PLASTIC
                        #else
                        plastic_m
                        #endif
                       ), res );
    res = union_op( MPt( cline_sd( vec3(2.0,36.0,0.5), vec3(2.0,36,1.5), 0.5, pos ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                        orange_m
                        #endif
                       ), res );
    res = union_op( MPt( cline_sd( vec3(0.0,0.0,-1.0), vec3(0.0,0.0,1.0), 0.25, at_angle( vec3( QPI * 0.5, 0.0, angle4), at_pos( vec3(2.0,38,1.5), pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                        orange_m
                        #endif
                       ), res );
    res = union_op( MPt( ellipsoid_sd( vec3(0.5,0.25,1.0), at_angle( vec3( 0.0, 0.0, angle4), at_pos( vec3(2.0,40,1.5), pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                        orange_m
                        #endif
                       ), res );
    
    res =
       union_op( MPt( torus82_sd( vec2(0.6,0.15), at_angle( vec3(0.0,HPI,iTime*TAU/4.0), at_pos(vec3(2.0,42.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( torus88_sd( vec2(0.6,0.15), at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,44.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( sphere8_sd( 0.5, at_angle( vec3(0.0,HPI,iTime*TAU/4.0), at_pos(vec3(2.0,46.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( torus42_sd( vec2(0.6,0.15), at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,48.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( torus44_sd( vec2(0.6,0.15), at_angle( vec3(0.0,HPI,iTime*TAU/4.0), at_pos(vec3(2.0,50.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( sphere4_sd( 0.5, at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,52.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( torus32_sd( vec2(0.6,0.15), at_angle( vec3(0.0,HPI,iTime*TAU/4.0), at_pos(vec3(2.0,54.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( torus33_sd( vec2(0.6,0.15), at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,56.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( sphere3_sd( 0.5, at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,58.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( torus2mh_sd( vec2(0.6,0.15), at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,60.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( torusmh2_sd( vec2(0.6,0.15), at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,62.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( torusmhmh_sd( vec2(0.6,0.15), at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,64.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( spheremh_sd( 0.5, at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,66.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( capped_cylinder8_sd( vec2(0.5,0.5), at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,68.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
    
    res =
       union_op( MPt( capped_cylindermh_sd( vec2(0.5,0.5), at_angle( vec3(0.0,HPI,angle4), at_pos(vec3(2.0,70.0,1.0),pos) ) ),
                        #if INDEXED_MATERIALS
                         MAT_ORANGE
                        #else
                         orange_m
                        #endif
                    ),
                 res );
#endif
	return res;
}

// Function 975
float map(vec3 p)
{
	p.x += sin(p.z*5.+sin(p.y*5.))*0.3;
    return (length(p)-1.)*0.7;
}

// Function 976
vec2 map_ns(vec3 pos)
{
    float jelly = map_jelly(pos);
    float container = map_container(pos);
    float rods = map_rods(pos);
    vec2 res = opU(vec2(jelly, JELLY_OBJ), vec2(container, CONTAINER_OBJ));
    res = opU(res, vec2(rods, RODS_OBJ));
    return res;
}

// Function 977
vec3 nmap(vec2 t, sampler2D tx, float str)
{
	float d=1.0/1024.0;

	float xy=texture(tx,t).x;
	float x2=texture(tx,t+vec2(d,0)).x;
	float y2=texture(tx,t+vec2(0,d)).x;
	
	float s=(1.0-str)*1.2;
	s*=s;
	s*=s;
	
	return normalize(vec3(x2-xy,y2-xy,s/8.0));///2.0+0.5;
}

// Function 978
vec4 SphereMap(sampler2D sam, in vec3 p)
{
  vec2 spherePos = PosToSphere(p);
  return textureLod(sam, spherePos, 2.*log2(spherePos.y*2.));
}

// Function 979
vec4   lchToLuv(float x, float y, float z, float a) {return   lchToLuv( vec4(x,y,z,a) );}

// Function 980
vec3 PBR_HDRCubemap(vec3 sampleDir, float LOD_01)
{
    vec3 linearGammaColor_sharp = PBR_HDRremap(pow(texture( iChannel2, sampleDir ).rgb,vec3(2.2)));
    vec3 linearGammaColor_blur  = PBR_HDRremap(pow(texture( iChannel3, sampleDir ).rgb,vec3(1)));
    vec3 linearGammaColor = mix(linearGammaColor_sharp,linearGammaColor_blur,saturate(LOD_01));
    return linearGammaColor;
}

// Function 981
float map(vec2 p){
	// Reading distance fields from a texture means taking scaling into
    // consideration. If you zoom coordinates by a scalar (4, in this case), 
    // you need to scale the return distance value accordingly... Why does 
    // everything have to be so difficult? :D
    const float sc = 4.;
    vec4 tex = tx(iChannel0, p/sc);
    gIP = tex.yz; // The object ID is stored in the YZ channels..
    return tex.x*sc;
}

// Function 982
vec4 hsluvToRgb(vec4 c) {return vec4( hsluvToRgb( vec3(c.x,c.y,c.z) ), c.a);}

// Function 983
float simpleBitmap(float data, float w, float h, vec2 bitCoord) {
    // 0..1.0
    float x = floor(bitCoord.x / (1.0 / w));
    float y = floor(bitCoord.y / (1.0 / h));
     
    float i = y * w + x;
    
    float datum = float(data) / pow(2.0, i);

    datum = mod(datum, 2.0);
        
    return floor(datum);
}

// Function 984
vec3 colormap(int index, float t) {
    float c = float(index) + 1.;
    return .5 + .45*cos(2.*PI  * pow(t, 0.4) * c + vec3(c, c+2., c+1.) / .6 + vec3(0, 1, 2));
}

// Function 985
Object map(vec3 p) {
    Object o = NewObject;
    

    //o = omin(o, sdFloorA(p - vec3(0.,-0.14,0)), materials[0]);
    
    //o = omin(o, sdWall(p - vec3(0.9,0.,0)), materials[0]);
    
    ////o = omin(o, -sdWall(p - vec3(-0.4,0.,0)), materials[0]);
    //o = omin(o, sdWall(p - vec3(0.4,0.,0)), materials[1]);
    //o = omin(o, sdSphere(p - vec3(-.6,0.4,0.6), 0.2), materials[0]);
    
    o = omin(o, sdJulia(p - vec3(-0.0,0.0,0.1)).d, materials[2]);
    //p.xz *= rot(0.1);
    p.y += 1.7;
    o = omin(o, sdJulia(p - vec3(4.,6.0,2.)).d, materials[2]);
    
    
    o.didHit = true;
    o.d *= 0.8;
    return o;
}

// Function 986
void UI_WriteCanvasUV( inout UIContext uiContext, int iControlId )        
{
	if (!uiContext.bPixelInView)
        return;
    Rect rect = Rect( vec2(0), uiContext.drawContext.vCanvasSize );
    DrawRect( uiContext.vPixelCanvasPos, rect, vec4(uiContext.vPixelCanvasPos / uiContext.drawContext.vCanvasSize, float(iControlId), -1.0 ), uiContext.vWindowOutColor );
}

// Function 987
vec4   luvToLch(vec4 c) {return vec4(   luvToLch( vec3(c.x,c.y,c.z) ), c.a);}

// Function 988
vec4 cloudMap(vec3 p, float time)
{
    p.xz += vec2(-time*1.0, time*0.25);
    time *= 0.25;
    p.y -= 9.0;
    p *= vec3(0.19,0.3,0.19)*0.45;
    vec3 bp = p;
    float rz = 0.;
    vec3 drv = vec3(0);
    
    float z = 0.5;
    float trk= 0.9;
    float dspAmp = 0.2;
    
    float att = clamp(1.31-abs(p.y - 5.5)*0.095,0.,1.);
    float off = dot(sin(p*.52)*0.7+0.3, cos(p.yzx*0.6)*0.7+0.3)*0.75 - 0.2; //large structures
    float ofst = 12.1 - time*0.1;
    
    for (int i = 0; i<6; i++)
    {
        p += sin(p.yzx*trk - trk*2.0)*dspAmp;
        
        vec3 c = cos(p);
        vec3 s = sin(p);
        vec3 cs = cos(p.yzx + s.xyz + ofst);
        vec3 ss = sin(p.yzx + s.xyz + ofst);
        vec3 s2 = sin(p + s.zxy + ofst);
        vec3 cdrv = (c*(cs - s*ss) - s*ss.yzx - s.zxy*s2)*z;
        
        rz += (dot(s, cs) + off - 0.1)*z; //cloud density
        rz *= att;
        drv += cdrv;
        
        p += cdrv*0.05;
        p.xz += time*0.1;
        
        dspAmp *= 0.7;
        z *= 0.57;
        trk *= 2.1;
        p *= m3x;
    }
    
    return vec4(rz, drv);
}

// Function 989
float map(vec3 p) {
    float nearest = MAX_T;
    for (float i = 0.; i < N; i++) {
        vec2 sp = sfi(i, N);
        vec3 sp3 = pointOnSphere(sp.x, sp.y);
        nearest = min(nearest, sphere(p - sp3, .01));
    }
    return nearest;
}

// Function 990
float map(in vec3 p) {
  float d1,d2;
  d1 = sph(p);
  d2 = plane(p);
  return min(d1,d2);
}

// Function 991
vec4 texMapSmoothCh(samplerCube tx, vec3 p){

    // Voxel corner helper vector.
	const vec3 e = vec3(0, 1, 1./4.);

    // Technically, this will center things, but it's relative, and not necessary here.
    //p -= .5/dimsVox.x;
    
    p *= dimsVox;
    vec3 ip = floor(p);
    p -= ip;

    
    int ch = (int(ip.x)&3), chNxt = ((ch + 1)&3);  //int(mod(ip.x, 4.))
    ip.x /= 4.;

    vec4 c = mix(mix(mix(txChSm(tx, ip + e.xxx, ch), txChSm(tx, ip + e.zxx, chNxt), p.x),
                     mix(txChSm(tx, ip + e.xyx, ch), txChSm(tx, ip + e.zyx, chNxt), p.x), p.y),
                 mix(mix(txChSm(tx, ip + e.xxy, ch), txChSm(tx, ip + e.zxy, chNxt), p.x),
                     mix(txChSm(tx, ip + e.xyy, ch), txChSm(tx, ip + e.zyy, chNxt), p.x), p.y), p.z);

 
 	/*   
    // For fun, I tried a straight up average. It didn't work. :)
    vec4 c = (txChSm(tx, ip + e.xxx*sc, ch) + txChSm(tx, ip + e.yxx*sc, chNxt) +
             txChSm(tx, ip + e.xyx*sc, ch) + txChSm(tx, ip + e.yyx*sc, chNxt) +
             txChSm(tx, ip + e.xxy*sc, ch) + txChSm(tx, ip + e.yxy*sc, chNxt) +
             txChSm(tx, ip + e.xyy*sc, ch) + txChSm(tx, ip + e.yyy*sc, chNxt) + txChSm(tx, ip + e.yyy*.5, ch))/9.;
 	*/
    
    return c;

}

// Function 992
float mapWater(vec3 p) {
    return sdPlane(p)+heightMapWater(p, iTime);
}

// Function 993
vec4  mapScene( vec3 p )
{
   vec4 m =vec4(1000.,0,0,0);
    
    for(int i = 0;i<=4;i++) for(int j0=0;j0<=2;j0++) 
    {
    
       int j= j0 <2?j0:-1, k =i*3+j+1;
     
       vec3 d0=ds(0,k),d1=ds(1,k),d2=ds(2,k),d3=ds(3,k);

       m=mmin(m,sdCapsule(p,j<0? d1: d0,d1)-TK,vec3(1.,0,0));   
       m=mmin(m,sdCapsule(p,j<0  || (j==0 && i==4 ) ? d1: d0,j<0 || (j==0 && i==4 )?d3:d2)-TK,vec3(0,0,1.));      
       m=mmin(m,sdCapsule(p, d1,d2)-TK,vec3(0,1.,0));

       if(j>=0) m=mmin(m ,udTri(d0,d1,d2,p)-TK/2.,vec3(1.));
       if(j<=0) m=mmin(m,udTri(d3,d1,d2,p)-TK/2.,vec3(1.));
    }
    return m;
}

// Function 994
vec2 map( in vec3 p, out vec4 glows)
{	
	vec2 res = distBody(p);

	vec2 filterDist = distModuleObject(p-module1Pos);
	vec2 filterDist2 = distModuleObject(p-module3Pos);
	vec2 cube = distCubeObject(p-module2Pos);
	vec2 cube2 = distCubeObject(p-module4Pos);
	
	if (filterDist.x < res.x) res = filterDist; 
	if (filterDist2.x < res.x) res = filterDist2; 
	if (cube.x < res.x) res = cube; 
	if (cube2.x < res.x) res = cube2; 
	
	glows = vec4(filterDist.x,cube.x,filterDist2.x,cube2.x);
	
	return res;
}

// Function 995
vec3 whitePreservingLumaBasedReinhardToneMapping(vec3 color)
{
    float white = 2.;
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma * (1. + luma / (white*white)) / (1. + luma);
    color *= toneMappedLuma / luma;
    color = pow(color, vec3(1. / gamma));
    return color;
}

// Function 996
float mapSand(vec3 p) {
    return sdPlane(p-vec3(0.0, -0.8-p.z*0.05, 0.0))-0.2;
}

// Function 997
vec4 toneMap(vec4 inputColor, vec3 gamma, vec3 exposure)
{
    vec3 gradedColor = vec3(pow(inputColor.r,gamma.r)*exposure.r,pow(inputColor.g,gamma.g)*exposure.g,pow(inputColor.b,gamma.b)*exposure.b);
    vec4 graded = vec4(1.0-1.0/(gradedColor + vec3(1.0)), inputColor.w);
    
    vec3 x = clamp(graded.xyz,0.0001,0.999);
    
    // ACES tone mapping approximation from https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return vec4(clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0001,0.999), inputColor.z);
}

// Function 998
void mapselect() {
	float d = box((UV - .5) * 256., vec2(98));
	FCol = vec4(1, 0, 0, 1) * (.3 * ls(24., .0, d) * step(.0, d)  + step(abs(d + 1.), 1.));
}

// Function 999
float mapBottom(in vec3 rp)
{
    rp.x += getCurve(rp);
    float bottom = -.7;
    float ax = abs(rp.x);
    bottom += smoothstep(2., 20., ax);
    bottom += (0.6 + 0.6 * noise(rp.xz * .4)) * .6;
    float hill = smoothstep(75., 90., ax);
    bottom += hill * 25.;
    bottom += hill * sin(rp.z * .2) * 3.;
    bottom += hill * sin(rp.z * .25) * 2.;
    return bottom; 
}

// Function 1000
vec4   luvToLch(float x, float y, float z, float a) {return   luvToLch( vec4(x,y,z,a) );}

// Function 1001
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

// Function 1002
vec2 map(vec3 p) {
    p.y += 1.0;
	vec2 res = vec2(sdPlane(p), M_FLOOR);
	p.y -= AppleRadius;
    

	vec2 obj = vec2(sdApple(p, AppleRadius), M_APPLE);
	res = min2(res, obj);

	return res;
}

// Function 1003
vec3 GetMipMapUVW_Dir2(vec3 _uvw,vec3 _axis){
    ivec3 xyz = ID_xyz[int(dot(abs(_axis.yz),vec2(1,2)))];
	vec3 uv = vec3(_uvw[xyz.x],_uvw[xyz.y],_uvw[xyz.z]);
    uv.xy = floor((uv.xy+1.)*512.);
    vec2 a = ceil(log2(uv.xy+1.));
	float max_a = max(a.x,a.y);
    float scale = exp2(max_a-1.);
    vec2 B = step((a.x+a.y)/2.,a);
    ID = max_a*3.-3. + B.y*2. + B.x;
    uv.xy = ((uv.xy+0.5)/scale - B - 0.5)*2.;
    uv = vec3(uv[xyz.x],uv[xyz.y],uv[xyz.z]);
    return normalize(uv+_axis);
}

// Function 1004
vec2 map( vec3 p )
{

    vec2 sp = vec2( sph( p, 1.0 ), 0.0 );
    vec2 pla = vec2( pla( p ), 1.0 );
    /*if( sp.x < pla.x ) pla = sp;
    return pla;*/
    return sp;
    
}

// Function 1005
float seaHeightMapTracing(const in vec3 ori, const in vec3 dir, out vec3 p) {  
    float tm = 0.0;
    float tx = 1000.0;    
    float hx = seaMap(ori + dir * tx);
    if(hx > 0.0) return tx;   
    float hm = seaMap(ori + dir * tm);    
    float tmid = 0.0;
    for(int i = 0; i < SEA_NUM_STEPS; i++) {
        tmid = mix(tm,tx, hm/(hm-hx));                   
        p = ori + dir * tmid;                   
    	float hmid = seaMap(p);
		if(hmid < 0.0) {
        	tx = tmid;
            hx = hmid;
        } else {
            tm = tmid;
            hm = hmid;
        }
    }
    return tmid;
}

// Function 1006
vec3 getCubeMap(vec2 inUV)
{
	vec3 samplePos = vec3(0.0f);
	
	// Crude statement to visualize different cube map faces based on UV coordinates
	int x = int(floor(inUV.x / 0.25f));
	int y = int(floor(inUV.y / (1.0 / 3.0))); 
	if (y == 1) {
		vec2 uv = vec2(inUV.x * 4.0f, (inUV.y - 1.0/3.0) * 3.0);
		uv = 2.0 * vec2(uv.x - float(x) * 1.0, uv.y) - 1.0;
		switch (x) {
			case 0:	// NEGATIVE_X
				samplePos = vec3(-1.0f, uv.y, uv.x);
				break;
			case 1: // POSITIVE_Z				
				samplePos = vec3(uv.x, uv.y, 1.0f);
				break;
			case 2: // POSITIVE_X
				samplePos = vec3(1.0, uv.y, -uv.x);
				break;				
			case 3: // NEGATIVE_Z
				samplePos = vec3(-uv.x, uv.y, -1.0f);
				break;
		}
	} else {
		if (x == 1) { 
			vec2 uv = vec2((inUV.x - 0.25) * 4.0, (inUV.y - float(y) / 3.0) * 3.0);
			uv = 2.0 * uv - 1.0;
			switch (y) {
				case 0: // NEGATIVE_Y
					samplePos = vec3(uv.x, -1.0f, uv.y);
					break;
				case 2: // POSITIVE_Y
					samplePos = vec3(uv.x, 1.0f, -uv.y);
					break;
			}
		}
	}

	return samplePos;  
}

// Function 1007
vec2 computeuv(vec3 P) {
   
    float lat = asin(P.y/length(P));
    float lon = -atan(P.z, P.x);

    return vec2(lon*0.5, lat) / 3.1415 + 0.5;
}

// Function 1008
vec2 map(vec3 p) {
    
    vec2 d = opU(vec2(dancingSphere(p - vec3(1.5, 0., 0.), 1.), 1.), vec2(p.y + 2., 2.));
    
    d = opU(d, vec2(udRoundBox(p - vec3(-1.5, 0., 0.), vec3(.0, 0., 0.), 1.), 3.));
    
    return d;
}

// Function 1009
float map(vec3 p)
{
    float d = mix(length(p)-1.1,length4(p)-1.,sdfsw-0.3);
    d = min(d, -(length4(p)-4.));
    return d*.95;
}

// Function 1010
vec3 map(vec3 p)
{
    vec3 res=vec3(100.);
    res=mapMin(res,vec3(p.y,1.,0.));
    res=mapMin(res,vec3(sdBox(REP(p,vec3(05.,9.,5.)),vec3(1.,1.,1.)),5.,0.));
    res=mapMin(res,vec3(p.z+10.,4.,0.));
    return res;
}

// Function 1011
float map(vec3 p) {
    float b = sdPlane(p);
    float c = dGlass(p);
    return min(b, c);
}

// Function 1012
vec3 aces_tonemap(vec3 color){	
	vec3 v = m1 * color;
    //vec3 v = color;
	vec3 a = v * (v + 0.0245786) - 0.000090537;
	vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
	return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));	
    //return pow(clamp((a / b), 0.0, 1.0), vec3(1.0 / 2.2));	
}

// Function 1013
vec3 sample_map(float x, float w) {
    float q0 = map(x - w);
    float q1 = map(x);
    float q2 = map(x + w);
    return vec3((q0+q1)*0.5,q1,(q1+q2)*0.5);
}

// Function 1014
vec2 convertUv(in vec2 uv, in vec4 from, in vec4 to)
{
    vec2 nd = (uv - from.xy) / (from.zw - from.xy);
    vec2 cuv = to.xy + (to.zw - to.xy) * nd;
    return cuv;
}

// Function 1015
vec4 mapping(float dist, float min, float max){	
	
	float distLut = (dist - min) / (max - min);
	int i1;
	for(int i=0;i <NB_LUT;++i){
		if(distLut < LUT_DIST[i+1]){i1 = i;break;}
	}
	vec4 col1,col2;
	float mixVal;
	if		(i1 == 0){
		col1 = LUT[0];col2 = LUT[1];
		mixVal = (distLut - LUT_DIST[0]) / (LUT_DIST[1] - LUT_DIST[0]);
	}else if(i1 == 1){
		col1 = LUT[1];col2 = LUT[2];
		mixVal = (distLut - LUT_DIST[1]) / (LUT_DIST[2] - LUT_DIST[1]);
	}else{
		col1 = LUT[2];col2 = LUT[3];
		mixVal = (distLut - LUT_DIST[2]) / (LUT_DIST[3] - LUT_DIST[2]);
	}
	
	
	//return vec4(mixVal);
	return mix(col1,col2,mixVal);
	
}

// Function 1016
void mainCubemap( out vec4 fragColour, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    fragColour = textureLod( iChannel0, rayDir, 0. ); // this needs NEAREST filter on the texture
	if ( iFrame == 0 ) fragColour = vec4(0);

    // wait for texture to load (I know the top of the cubemap should not be black)
    if ( textureLod( iChannel1, vec3(0,1,0), 0. ).r == 0. ) return;
    
    // early-out once we've got a good enough result
    if ( fragColour.a > 16.*60.*3. ) return;
    
    const int n = 16;
    for ( int i = 0; i < n; i++ )
    {
        vec3 ray = HemisphereRand(rayDir,uint(i+n*iFrame));

        fragColour.rgb += LDRtoHDR(textureLod( iChannel1, ray, 0. ).rgb);
        fragColour.a += 1.;
    }
}

// Function 1017
float heightMap( in vec2 p ) { 

    // The stone texture is tileable, or repeatable, which means the pattern is slightly
    // repetitive, but not too bad, all things considered. Note that the offscreen buffer 
    // doesn't wrap, so you have to do that yourself. Ie: fract(p) - Range [0, 1].
    return texture(iChannel0, fract(p/2.), -100.).w;

}

// Function 1018
vec2 map(vec3 pos)
{
    bg_width = bg_width0*(1. + 15.*smoothstep(0.5, 1.0, sin(0.008*pos.x*gen_scale + 1.1)*sin(0.008*pos.z*gen_scale - 3.45)));

    float background = map_background(gen_scale*pos);
    vec2 res = vec2(background, BACKGROUND_OBJ);
    
    float hexagons = map_hexagons(gen_scale*pos);
    res = opU(res, vec2(hexagons, HEXAGONS_OBJ)); 
    
    return res;
}

// Function 1019
vec4 map( in vec3 p )
{
	float d = 0.1 + .8 * sin(0.6*p.z)*sin(0.5*p.x) - p.y;

    vec3 q = p;
    float f;
    
    f  = 0.5000*noise( q ); q = q*2.02;
    f += 0.2500*noise( q ); q = q*2.03;
    f += 0.1250*noise( q ); q = q*2.01;
    f += 0.0625*noise( q );
    d += 2.75 * f;

    d = clamp( d, 0.0, 1.0 );
    
    vec4 res = vec4( d );
    
    vec3 col = 1.15 * vec3(1.0,0.95,0.8);
    col += vec3(1.,0.,0.) * exp2(res.x*10.-10.);
    res.xyz = mix( col, vec3(0.7,0.7,0.7), res.x );
    
    return res;
}

// Function 1020
void projectCubemapOnSHCoefficients()
{
    #define SAMPLES 8
    
    float invN = 1.0/float(SAMPLES);
    for (int xx = 0; xx < SAMPLES; ++xx)
        for (int yy = 0; yy < SAMPLES; ++yy)
        {
            vec2 r = (vec2(float(xx), float(yy)) +
                      hash(vec2(float(xx), float(yy)))) * invN;
            //float theta = 2.0 * acos(sqrt(1.0 - r.x)); // This is as in 'Robin Green SH the gritty details' doc
            float theta = acos(1.0 - 2.0 * r.x);
            float phi = 2.0 * PI * r.y;
            vec3 dir = vec3(sin(theta)*cos(phi),
                            cos(theta),
                            sin(theta)*sin(phi));
            
            vec3 texcol = texture(iChannel0, dir).rgb;
            #if PERFORMANCE==1
			SHCoefs[0] += y00 (dir.xzy) * texcol;
			SHCoefs[1] += y11_(dir.xzy) * texcol;
			SHCoefs[2] += y10 (dir.xzy) * texcol;
			SHCoefs[3] += y11 (dir.xzy) * texcol;
			SHCoefs[4] += y22_(dir.xzy) * texcol;
			SHCoefs[5] += y21_(dir.xzy) * texcol;
			SHCoefs[6] += y20 (dir.xzy) * texcol;
			SHCoefs[7] += y21 (dir.xzy) * texcol;
			SHCoefs[8] += y22 (dir.xzy) * texcol;
            #else
            for (int l = 0; l < 3; ++l)
                for (int m = -l; m <= l; ++m)
                {
                    int index = l*(l+1)+m;
                    SHCoefs[index] += SH(dir.xzy, l, m) * texcol;
                }
            #endif
        }
    
    // divide result by weight and total number of samples
    float factor = 4.0 * PI / float(SAMPLES * SAMPLES);
    for (int i = 0; i < 9; ++i)
    {
        SHCoefs[i] = SHCoefs[i] * factor;
    }
}

// Function 1021
vec2 Map( in vec3 p, in float mult)
{
	vec2 ret = MapCry(p);
	vec2 bg = MapBg(p); 
	vec2 ice = MapIce(p,mult,ret.x);
	
	//cut the ice on the ground
	ice.x = max(ice.x,-bg.x);
		
	if (ret.x > bg.x) ret = bg; 
	if (ret.x > ice.x) ret = ice; 
	return ret;
}

// Function 1022
float fogmap(in vec3 p, in float d)
{
    p += iTime * FOGMAP_SPEED * FOGMAP_DIR;
    return saturate(fogNoise(p * 0.05) * fogNoise(p * 0.1) * 0.5);
}

// Function 1023
float map(in vec3 p)
{
    float r =length(p);
    vec2 sph = vec2(acos(p.y/r), atan(p.x, p.z));
    
    matid = 1.;
    float d = r-1.; 
    d += sin(sph.y*7.)*0.02;
    d += sin(sph.y*20.)*0.002;
    float gbh = sin((sph.x+sph.y)*7.+0.5)*0.5+0.5;
    d += sin(sph.y*40.)*0.001*gbh;
    d += sin(sph.x*1.85+2.7)*0.3;
    
    //Leaves
    vec3 p2 = p;
    float rxz2 = dot(p.xz,p.xz);
    float rxz = sqrt(rxz2);
    rxz = exp2(rxz*6.-5.);
    p2.xz = foldPent(p2.xz);
    p2.y -= sqrt(rxz)*0.17 + sin(rxz*2.+p.z*p.x*10.)*0.05;
    float leaves = sbox(p2+vec3(0,-.92-smoothstep(-0.01,.05,rxz2)*0.05,0),vec3(.07- rxz*0.1,0.002+p2.x*0.15,0.8));
    leaves = smin(leaves, cyl(p+vec3(sin(p.y*3.5 + 0.8)*0.3 + 0.3,-1.1,0),vec2(.05,.25))); //Tail
    if (leaves < d)matid = 2.;
    d = min(d, leaves);
    
    float flor = p.y+.65;
    if (flor < d)matid = 0.;
    d = min(d, flor);
    return d;
}

// Function 1024
vec2 inversePerspective_uv(vec2 uv)
{
    vec3 wa, wb, wc, wd;
    vec2 a, b, c, d;
    get_points(a,d,c,b);
    resolvePerspective(a*.5, b*.5, c*.5, d*.5, wa, wb, wc, wd);
    vec3 x_ws = wb-wa;
    vec3 y_ws = wd-wa;
    vec3 p_ws = wa+uv.x*x_ws + uv.y*y_ws;
    vec2 puv = -vec2(p_ws.x/p_ws.z,p_ws.y/p_ws.z);
    return puv+vec2(0.5,0.5);
}

// Function 1025
void rotateUV(inout vec3 vec, vec2 angle) {
    rotateXY(vec.zy, angle.x);
    rotateXY(vec.xz, angle.y);
}

// Function 1026
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

// Function 1027
vec2 MapCry( in vec3 p )
{
	//rotate the logo
	p.xy = vec2(p.x+p.y,p.y-p.x)*sincos45;
	vec3 q = abs(p); 
	
	//distance Rounded Box
	vec4 dim = vec4(0.54,0.54,0.1,0.02);
	float boxDist = length(max(q - dim.xyz,vec3(0.0)))-dim.w;
	
	//distance to Minus Core 
	float sphereDist = length(q.xy + vec2(0.2)) - 0.81;
	return vec2(max(-sphereDist,boxDist),0.0);
}

// Function 1028
float MapMissile(vec3 p)
{
  float d= fCylinder( p, 0.70, 1.7);
  if (d<1.0)
  {
    missileDist = min(missileDist, fCylinder( p, 0.12, 1.2));   
    missileDist =min(missileDist, sdEllipsoid( p- vec3(0, 0, 1.10), vec3(0.12, 0.12, 1.0))); 

    checkPos = p;  
    pR(checkPos.xy, 0.785);
    checkPos.xy = pModPolar(checkPos.xy, 4.0);

    missileDist=min(missileDist, sdHexPrism( checkPos-vec3(0., 0., .60), vec2(0.50, 0.01)));
    missileDist=min(missileDist, sdHexPrism( checkPos+vec3(0., 0., 1.03), vec2(0.50, 0.01)));
    missileDist = max(missileDist, -sdBox(p+vec3(0., 0., 3.15), vec3(3.0, 3.0, 2.0)));
    missileDist = max(missileDist, -fCylinder(p+vec3(0., 0., 2.15), 0.09, 1.2));
  }
  return missileDist;
}

// Function 1029
float map(vec3 p)
{
    p.y += height(p.zx);
    
    vec3 bp = p;
    vec2 hs = nmzHash22(floor(p.zx/4.));
    p.zx = mod(p.zx,4.)-2.;
    
    float d = p.y+0.5;
    p.y -= hs.x*0.4-0.15;
    p.zx += hs*1.3;
    d = smin(d, length(p)-hs.x*0.4);
    
    d = smin(d, vine(bp+vec3(1.8,0.,0),15.,.8) );
    d = smin(d, vine(bp.zyx+vec3(0.,0,17.),20.,0.75) );
    
    return d*1.1;
}

// Function 1030
vec3 simpleReinhardToneMapping(vec3 color)
{
	float exposure = 1.5;
	color *= exposure/(1. + color / exposure);
	color = pow(color, vec3(1. / gamma));
	return color;
}

// Function 1031
float map_doors(vec3 pos)
{
   #ifdef opendoors
   float doornr = floornr*4.;
   if (pos.z>0.)
      doornr+= 1.;
   if (pos.x<-staircase_length/2. + 0.5)
      doornr+= 2.;
      
   float dohash1 = hash(doornr*8543.45);
   float dohash2 = hash(doornr*1462.562);
   float dooropen_prob2 = dooropen_prob + floornr/450.;
   float dooropen_probmove = min(floornr/250., dooropen_prob2*0.7);
   
   door_angle = dohash1<dooropen_prob2 && floornr>0.?(dohash1<dooropen_probmove?-dooropen_minangle - (sin((3.4)*iTime)*0.5 + 0.5)*dohash2*(dooropen_maxangle - dooropen_minangle):-dooropen_minangle - dohash2*(dooropen_maxangle - dooropen_minangle)):0.;
   
   pos.y = mod(pos.y, floor_height);
   pos.z = abs(pos.z);
   vec3 pos1 = pos + vec3(staircase_length/2. - floor_width/2., 0., -staircase_width/2. - door_depth - handle_plate.z - handle_plate.w);
   vec3 pos2 = pos + vec3(staircase_length/2. + door_depth + handle_plate.z + handle_plate.w, 0., -staircase_width/4.);
   pos1.xz = rotateVec(pos1.xz + vec2(door_width/2., 0.), door_angle) - vec2(door_width/2., 0.);
   pos2.xz = rotateVec(pos2.xz + vec2(0., door_width/2.), door_angle) - vec2(0., door_width/2.);
   float doors = sdRoundBox(pos1, vec3(door_width/2., door_height, door_thickness/2.), 0.005);
   doors = min(doors, sdRoundBox(pos2, vec3(door_thickness/2., door_height, door_width/2.), 0.005));
   #else
   float doors = -abs(pos.z) + staircase_width/2. + door_depth;
   doors = min(doors, pos.x + staircase_length/2. + door_depth);
   doors = max(doors, pos.x);
   #endif
   
   return doors;
}

// Function 1032
float map(vec3 pos)
{
	vec2 sa = staircase(pos);
    vec2 wp = walls(pos);
	vec2 pn = pane(pos);
	return min(min(sa.x,wp.x),pn.x);
}

// Function 1033
vec2 IdxtoUV(float id, float stride)
{
    id = floor(id);
    stride = floor(stride);
    return vec2(mod(id, stride), floor(id / stride));
}

// Function 1034
vec3 unToneMap(vec3 crgb, float exposure)
{
    vec3 c = crgb * toneMap(exposure);
    for (int i = 3; i-- > 0; ) c[i] = unToneMap(c[i]);
    return c;
}

// Function 1035
vec3 CalculateNormalMapNormal(in vec2 uv, in float height, in vec3 normal, in vec3 tangent, in vec3 binormal)
{   
    vec3 normalMap = SampleNormalMap(uv, height).rgb;
	return normalize((normal * normalMap.b) + (binormal * normalMap.g) + (tangent * normalMap.r));
}

// Function 1036
vec3 hsluv_toLinear(vec3 c) {
    return vec3( hsluv_toLinear(c.r), hsluv_toLinear(c.g), hsluv_toLinear(c.b) );
}

// Function 1037
vec2 map(vec3 p) {
	fbmc = fbm(p * .6) * 2.;
	vec2 d = sdDonut(p) - fbm(p * 8.) * .02;
	d = min2(d, sdCream(p));
	d = min2(d, sdSprinkles(p));
	d = min2(d, vec2(p.y + 1.7, 3.5));

	vec3 mp = p;
	mp.x = abs(mp.x);

	// Paws.
	vec2 cat = vec2(length(mp - vec3(1.3, 1.4, -3.96)) - .2, 7.5);
	cat = min2(cat, vec2(length(mp - vec3(1.5, 1.4, -4)) - .2, 7.5));
	cat = min2(cat, vec2(length(mp - vec3(1.7, 1.45, -3.86)) - .2, 7.5));
	cat = min2(cat, vec2(length(mp - vec3(1.5, 1.3, -3.5)) - .6, 6.5));

	// Body
	mp.y += (sin(iTime)+0.33*sin(iTime * 3.)) * .5;
	cat = min2(cat, vec2(sdCapsule(mp.xzy, 1.6, 3.), 6.5));

	// Eyes.
	cat = min2(cat, vec2(length(mp - vec3(.8, 2.4, -2.3)) - .7, 5.5));

	// Ears.
	vec3 ep = mp;
	ep.xz *= rot(-.5 + sin(iTime * 2.) * .1);
	float ear = length(ep - vec3(2, 4, 0)) - .8;
	ear = max(ear, -ep.z);
	cat.x = smin(cat.x, ear, .3);

	// Nose.
	vec3 np = mp - vec3(0, 1.9, -2.9);
	float nose = sdCapsule(np, vec3(0), vec3(.16, .16, 0), .15);
	nose = smin(nose, sdCapsule(np * vec3(-1, 1, 1), vec3(0), vec3(.16, .16, 0), .15), .05);
	cat = min2(cat, vec2(nose, 2.5));

	// Mouth.
	np.x = abs(np.x);
	np -= vec3(.2, -.1, -.1);
	float mouth = sdCappedTorus(np, vec2(-1, 0), .2, .05);
	cat = min2(cat, vec2(mouth, 8.5));

	return min2(d, cat);
}

// Function 1038
float unToneMap(float c)
{
    switch (method) {
  	  case 0: // via inverse filmic-like Reinhard-esque combination approximation
        c = c/(6.-6.*c);
		break;
      case 1: // inverse ACES+sRGB
        c = pow(c, 2.2); // from sRGBish gamut
        c = -(sqrt((-10127.*c+13702.)*c+9.)+59.*c-3.) / (486.*c-502.); // inverse ACES according to Maxima
	    break;
      default: // just un-sRGB
        c = pow(c, 2.2);
        break;
    }
    return c;
}

// Function 1039
float Heightmap(vec3 pos)
{
    float octaves     = TimeLerp(1.0, 7.0, TIME_NoiseScale, TIME_NoiseOctaves);  
    float scale       = TimeLerp(1.0, 0.005, TIME_Noise, TIME_NoiseScale);
    float persistence = 0.35;
    float heightMod   = TimeLerp(1.0, 2.05, TIME_NoiseScale, TIME_NoiseOctaves);
    
    float noise = SimplexNoise(pos, octaves, scale, persistence) * heightMod;
    
    return clamp(((noise / heightMod) + 1.0) * 0.5, 0.0, 1.0);
}

// Function 1040
float landMap(vec3 p) {
    return p.z - texture(iChannel0, fract(p.xy)).r;
}

// Function 1041
vec2 stdNormalMap(in vec2 uv) 
{
    float height = texture(heightMap, uv).r;
    return -vec2(dFdx(height), dFdy(height)) * pixelToTexelRatio;
}

// Function 1042
vec2 quadTreeUV(vec2 uv, float depth, float hash) {
    
    vec2 ret = uv;
    
    for(float i = 1.0; i < depth; i++) {
        
        ret = mod(ret, vec2(0.5)) * 2.0;
        
        float x = pow(2.0, i);
        
        if (hash12(floor(uv * x) * 1000.0) > 0.6) {
            
            return ret;
        }
    }
    
    return ret;
}

// Function 1043
vec3 map( in vec3 p ) {
        float sr = 1.0;  // sphere radius
        vec3 box = vec3(3.5,1.7,0.3);
        float rbox = 0.1;
        float id = 1.0;  // 3 = floor, 2 wall, 1 sphere

        // Sphere
        float distS = length(p + vec3( sin(iTime * 0.21) * 3.0,0.3,cos(iTime * 0.21) * 3.0))-sr;
        
        // First wall
        float distQ = length(max(abs(p)-box,0.0))-rbox;
        // Cuboid moved 2.0 to the right (2on wall)
        float distQ2 = length(max(abs(p + vec3(-2.0,.0,2.0)) - vec3(0.3,1.7,3.5), 0.0))-rbox;

        // Window in the wall
        vec3 d = abs(p) - vec3(0.5,0.8,1.0);
        float distQ3 = min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));

        // External wall (parcel limit) without ceiling
        vec3 d2 = vec3(10.0,2.0,10.0) - abs(p);
        float distQ4 = max( min( d2.x,  d2.z ), -d2.y );

        // Column
        float distCol = length(max(abs(p + vec3(2.5,.0,4.5)) - vec3(0.3,1.7,0.3), 0.0))-rbox;


        float distTot = min( min( distS, distQ), distQ2 );
        distTot = max( -distQ3,distTot );
        distTot = min( distCol,distTot );
        distTot = min( distTot, distQ4 );

        if (distS > distTot) {
            id = 2.0;
        }

        float distP = p.y + 1.6;

        if (distP < distTot) {
            id = 3.0;
        }

        distTot = min(distP, distTot);

        return vec3(distTot, id, 0.0 );
    }

// Function 1044
void PrepareTonemapParams(vec2 p1, vec2 p2, vec2 p3, out TonemapParams tc)
{
	float denom = p2.x - p1.x;
	denom = abs(denom) > 1e-5 ? denom : 1e-5;
	float slope = (p2.y - p1.y) / denom;

	{
		tc.mMid.x = slope;
		tc.mMid.y = p1.y - slope * p1.x;
	}

	{
		float denom = p1.y - slope * p1.x;
		denom = abs(denom) > 1e-5 ? denom : 1e-5;
		tc.mToe.x = slope * p1.x * p1.x * p1.y * p1.y / (denom * denom);
		tc.mToe.y = slope * p1.x * p1.x / denom;
		tc.mToe.z = p1.y * p1.y / denom;
	}

	{
		float denom = slope * (p2.x - p3.x) - p2.y + p3.y;
		denom = abs(denom) > 1e-5 ? denom : 1e-5;
		tc.mShoulder.x = slope * pow(p2.x - p3.x, 2.0) * pow(p2.y - p3.y, 2.0) / (denom * denom);
		tc.mShoulder.y = (slope * p2.x * (p3.x - p2.x) + p3.x * (p2.y - p3.y) ) / denom;
		tc.mShoulder.z = (-p2.y * p2.y + p3.y * (slope * (p2.x - p3.x) + p2.y) ) / denom;
	}

    tc.mBx = vec2(p1.x, p2.x);
	tc.mBy = vec2(p1.y, p2.y);
}

// Function 1045
vec3 tonemap (vec3 color) {
  vec3  hue = rgb_to_hue(color);
  float sat = rgb_to_sat(color);
  float lum = rgb_to_lum(color);

  // smooth-clamp
  sat = -log(exp(-sat*10.)+exp(-10.))/10.;

  /* tonemapping options:
       - desaturate when very bright
       - smooth-clamp brightness to a maximum that still
          allows some color variation                              */
  sat = sat*(exp(-lum*lum*2.));
  lum = .8*(1.-exp(-lum));

  color = lum*mix(vec3(1.),hue,sat);
  return color;
}

// Function 1046
vec3 hpluvToLch(vec3 tuple) {
    tuple.g *= hsluv_maxSafeChromaForL(tuple.b) * .01;
    return tuple.bgr;
}

// Function 1047
float remapTo01Clamped(float value, float start, float end)
{
	return clamp((value - start) / (end - start), 0.0, 1.0);
}

// Function 1048
float mapTex(samplerCube tex, vec3 p, vec2 size) {
    // stop x bleeding into the next cell as it's the mirror cut
    #ifdef MIRROR
        p.x = clamp(p.x, -.95, .95);
    #endif
    vec2 sub = texSubdivisions;
    float zRange = sub.x * sub.y * 4. * 6. - 1.;
    float z = p.z * .5 + .5; // range 0:1
    float zFloor = (floor(z * zRange) / zRange) * 2. - 1.;
    float zCeil = (ceil(z * zRange) / zRange) * 2. - 1.;
    vec4 uvcA = spaceToTex(vec3(p.xy, zFloor), size);
    vec4 uvcB = spaceToTex(vec3(p.xy, zCeil), size);
    float a = texture(tex, uvcA.xyz)[int(uvcA.w)];
    float b = texture(tex, uvcB.xyz)[int(uvcB.w)];
    return mix(a, b, range(zFloor, zCeil, p.z));
}

// Function 1049
float map_hor_small(vec3 pos, vec2 delta, float n)
{
    float fy = 132.*random(12.54*floor(pos.y/fe));
    float ad = 1. + ttwd*hsf;
                          
    float angle = ad*twf*pos.x;
    vec2 d1 = rotateVec(vec2(fr*fd, fr*fd), angle);
    vec2 d2 = d1.yx*vec2(1., -1);
    return min(min(min(map_f_hor(pos, d1 + delta, n + 1.), map_f_hor(pos, d2 + delta, n + 2.)), map_f_hor(pos, -d2 + delta, n + 3.)), map_f_hor(pos, -d1 + delta, n + 4.)); 
}

// Function 1050
vec3 doBumpMap( sampler2D tx, in vec3 p, in vec3 n, float bf){   
    const vec2 e = vec2(0.001, 0);
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; 
    g -= n*dot(n, g);
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
}

// Function 1051
float map(vec2 x) {
    float a = iTime * 0.2;
    vec2 p = vec2(cos(a),sin(a)) * vec2(0.8,0.3);
    
    float d1 = length(x - p) - 0.5;
    float d2 = length(x - p + vec2(0.2,0.5)) - 0.5;
    return max(d1, -d2);
}

// Function 1052
float map(vec3 q){
    
    // Layer one. The ".05" on the end varies the hole size.
 	vec3 p = abs(fract(q/3.)*3. - 1.5);
 	float d = min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1. + .05;
    
    // Layer two.
    p =  abs(fract(q) - .5);
 	d = max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1./3. + .05);
   
    // Layer three. 3D space is divided by two, instead of three, to give some variance.
    p =  abs(fract(q*2.)*.5 - .25);
 	d = max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - .5/3. - .015); 

    // Layer four. The little holes, for fine detailing.
    p =  abs(fract(q*3./.5)*.5/3. - .5/6.);
 	return max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1./18. - .015);
    //return max(d, max(max(p.x, p.y), p.z) - 1./18. - .024);
    //return max(d, length(p) - 1./18. - .048);
    
    //p =  abs(fract(q*3.)/3. - .5/3.);
 	//return max(d, min(max(p.x, p.y), min(max(p.y, p.z), max(p.x, p.z))) - 1./9. - .04);
}

// Function 1053
float distMap(vec3 p)
{
	float ret = 0.0;
	vec3 q = rotate(p,  iTime-slowt, vec3(2.0, 1.0, 5.0));
	//ret = distSphere(p, 1.0);
	ret = distBox(q, vec3(0.5, 0.5, 0.5));
    
    //ret = min(distBox(q+1.0, vec3(0.5, 0.5, 0.5)),distBox(q, vec3(0.3, 0.3, 0.3)));
    
	return ret;
}

// Function 1054
vec2 MapBg( in vec3 p)
{
	//background room shape
	vec3 q = p; 
	q.xz = abs(q.xz); 
	return vec2(min(min(p.y + 1.0,5.0 - p.y),min(5.0-q.z,5.0-q.x)),1.0);
}

// Function 1055
void dot_uv( inout vec4 fragColor, in vec2 uv, in float frq)
{
    frq *= PI;
    float dist = length(uv);
    
    vec4 col;
    set_hue(col, uv);
    
    float f = abs(sin(log(dist) * frq));
    if (log_dot)
    	f = smoothstep(0.0, 1.0, f / frq * win_size.y * 0.5);
   	else
    	f = smoothstep(0.0, 1.0, f * dist / frq * win_scale);
    
    fragColor = mix(col, fragColor, f);
}

// Function 1056
vec3   luvToRgb(float x, float y, float z) {return   luvToRgb( vec3(x,y,z) );}

// Function 1057
float map(vec3 p) {
  vec3 temp;
  return mapWDists(p, temp);
}

// Function 1058
vec2 findParallelogramUV(vec3 o, vec3 d, worldSpaceQuad wsQuad)
{
    //Note : This is tricky because axis are not orthogonal.
    vec3 uvX_ref = wsQuad.b-wsQuad.a; //horitonal axis
    vec3 uvY_ref = wsQuad.d-wsQuad.a; //vertical axis
    vec3 quadN = cross(uvY_ref,uvX_ref);
    float t = rayPlaneIntersec(o, d, wsQuad.a, quadN);
        
    vec3 p = o+t*d;
    vec3 X0_N = cross(uvY_ref,quadN);
    vec3 Y0_N = cross(uvX_ref,quadN);
    
    //Vertical component : find the point where plane X0 is crossed
    float t_x0 = rayPlaneIntersec(p, uvX_ref, wsQuad.a, X0_N);
    vec3 pY = p+t_x0*uvX_ref-wsQuad.a;
    //Horizontal component : find the point where plane Y0 is crossed
    float t_y0 = rayPlaneIntersec(p, uvY_ref, wsQuad.a, Y0_N);
    vec3 pX = p+t_y0*uvY_ref-wsQuad.a;
    
    //All is left to find is the relative length ot pX, pY compared to each axis reference
    return vec2(dot(pX,uvX_ref)/dot(uvX_ref,uvX_ref),
	            dot(pY,uvY_ref)/dot(uvY_ref,uvY_ref));
}

// Function 1059
vec2 to_uv(in vec2 in_pixels) {
    return in_pixels / iResolution.xy;
	// return 0.1 + mod(vec2(0.8) + in_pixels / iResolution.xy, vec2(0.9));
}

// Function 1060
float map(vec3 p){
    
    // Perturbing the walls with a sinusoidal function just a bit to give the tunnel a 
    // less man made feel.
    vec3 pert = p*vec3(1, 1, .5);
    pert = sin(pert - cos(pert.yzx*2.))*.25;
    
    //float id = floor(p.z/16.)*16. + 8.;
    //vec3 pos = vec3(path(id), id);
    //vec3 q = p - pos;
     
    // Wrapping the tunnel, vents and floor around the path.
    p.xy -= path(p.z).xy;
    
    // The ground. Nothing fancy. 
    float ground = p.y + 2.375 + pert.y*.125;
    
    
    // A pretty hacky vent shaft object. I'll rewrite this in a better way at some
    // stage, so I wouldn't pay too much attention to it.
    vec3 q = p;
    // Repeting the shafts every 16 units.
    q.z = mod(q.z + 0., 16.) - 8.; // (q.z/16. - floor(q.z/16.))*16. - 8.;
    

    // The shaft cross section. I decided to make them octagonal, for whatever reason, 
    // but it's not mandatory.
    float sCirc = dist(q.xz) - 1.15;
   
    // The ventilation shaft.
    float shafts = max(abs(sCirc) - .1, abs(q.y - (4.6)) - 2.);
    
    // The rim under the vent.
    float rim = dist(vec2(sCirc, q.y - 2.6)) - .15; 
    
    shafts = min(shafts, rim);
    
    
    // Shaft holes. The end bit is needed to stop the shaft holes from continuing
    // through the floor.
    float shaftHoles = max(min(sCirc - .05, rim - .1), -p.y);
    //shaftHoles = smin(shaftHoles, rim, 1.);
    
 
    // Subdividing space into smaller lots to create the vent grids. I probably should
    // had created these separately in order to set unique materials, but I've attached 
    // them to the vent object.
    q.xz = mod(q.xz, .25) - .25/2.;
    q.y -= 3. - .25;
    float ventGrid = length(q.xy) - .035;//min(length(q.yz), length(q.xy)) - .05;
    ventGrid = max(ventGrid, sCirc);
    //shafts = min(shafts, ventGrid);
    
    // Cylindrically mapping the rocky texture onto the walls of the cylinder. The 
    // cylinder has been warped here and then, so it's not an exact fit, but no one
    // will notice.
    float sf = getCylTex(p).a;
    
    // Arrange for the rocky base to effect the sand level ever so slightly.
    ground += (.5 - sf)*.25;

    p.xy *= vec2(.9, 1); // Widen the tunnel just a bit.
    // Add the sinusoidal perturbations to the tunnel via some space warping. You could
    // also do this in height map form, but I chose this way... for some reason that
    // escapes me at present. :)
    p += pert; 
   
    // The tunnel object -- otherwise known as a glorified cylinder. We're approaching
    // the cylinder walls from the inside. Hence, the negative sign.
    float tun =  -(polDist(p.xy) - 2.7);
    
    // There a light sitting about the vents outside the tunnel, so I've given the
    // tunnel a bit of thickness. That's all this is 
    tun = max(tun + (.5-sf)*1., -(tun + 4.));
    // Creating some holes in the tunnel roof for the vents to fit into. A smooth blend
    // is used to smoothen the rocks around the grated end of the vent.
    tun = smax(tun, -shaftHoles, .5);
     
    
    // Save the IDs. For speed, they're sorted outside the loop.
    vObjID = vec4(tun, shafts, ground, ventGrid);
    
    float df = min(min(tun, ventGrid), min(shafts, ground));
    

    // Return the distance -- Actually, a fraction of the distance, since a lot of 
    // this isn't Lipschitz friendly, for want of a better term. :)
    return df*.86;
 
}

// Function 1061
bool ringMap( const in vec3 ro ) {
    return ro.z < RING_HEIGHT/RING_VOXEL_STEP_SIZE && hash(ro)<.5;
}

// Function 1062
bool map( const vec2 vos ) {
	return isObject( vos ) || isWall( vos );
}

// Function 1063
vec3 hsluv_lengthOfRayUntilIntersect(float theta, vec3 x, vec3 y) {
    vec3 len = y / (sin(theta) - x * cos(theta));
    if (len.r < 0.0) {len.r=1000.0;}
    if (len.g < 0.0) {len.g=1000.0;}
    if (len.b < 0.0) {len.b=1000.0;}
    return len;
}

// Function 1064
float map_smooth(vec3 pos)
{
    return smin(map_slime(pos).x, 0.95*map_ns(pos).x, clamp(0., 1., 1. + clamp(0.1*pos.y, -0.3, 1.) - 0.7*smoothstep(2., 6.5, 0.17*pow(abs(pos.x), 2.) + 6.*smoothstep(0.3, 2.3, abs(pos.z)))));
    //return min(map_slime(pos).x, 0.9*map_ns(pos).x);
}

// Function 1065
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    // Larger sample distances give a less defined bump, but can sometimes lessen the aliasing.
    const vec2 e = vec2(.001, 0); 
    
    // Gradient vector: vec3(df/dx, df/dy, df/dz);
    float ref = bumpSurf3D(p);
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy),
                      bumpSurf3D(p - e.yxy),
                      bumpSurf3D(p - e.yyx)) - ref)/e.x; 
    
    /*
    // Six tap version, for comparisson. No discernible visual difference, in a lot of cases.
    vec3 grad = vec3(bumpSurf3D(p - e.xyy) - bumpSurf3D(p + e.xyy),
                     bumpSurf3D(p - e.yxy) - bumpSurf3D(p + e.yxy),
                     bumpSurf3D(p - e.yyx) - bumpSurf3D(p + e.yyx))/e.x*.5;
    */
       
    // Adjusting the tangent vector so that it's perpendicular to the normal. It's some kind 
    // of orthogonal space fix using the Gram-Schmidt process, or something to that effect.
    grad -= nor*dot(nor, grad);          
         
    // Applying the gradient vector to the normal. Larger bump factors make things more bumpy.
    return normalize(nor + grad*bumpfactor);
	
}

// Function 1066
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

// Function 1067
vec2 map(in vec3 pos) {
  // Ring planet.
  vec3 rp = (pos - vec3(0.0, 0.25, 1.6)) * rot3(0.5, 0.1, 0.0);
  vec2 res = vec2(sdSphere(rp, 0.25), 1.5 + rp.y);
  res = opU(res, vec2(sdDiscRing(rp, vec3(0.55, 0.33, 0.003)), 2.0 + length(rp.xz)));

  // Sun.
  //res = opU(res, vec2(sdSphere(pos - SUN_POS, 2.5), 3));

  // Close planet.
  res = opU(res, vec2(sdSphere(pos - CLOSE_PLANET_POS, 1.3), 4));

  // Space station.
  vec3 pos2 = (pos - vec3(0.35, -0.7, -0.45)) * rot3(0.0, 0.3 * iTime, 0.1);
  res = opU(res, vec2(sdCapsule(pos2, vec3(0.0, -0.25, 0.0), vec3(0.0, 0.25, 0.0), 0.03), 5));
  res = opU(res, vec2(sdCapsule(pos2, vec3(0.0, 0.0, -0.22), vec3(0.0, 0.0, 0.22), 0.02), 6));
  float arcpos = atan(pos2.z, pos2.x) * 0.95492965855;
  res = opU(res, vec2(sdDiscRing(pos2, vec3(0.25, 0.2, 0.02)), (fract(arcpos - 0.1) < 0.8) ? 6 : 7));

  return res;
}

// Function 1068
float YUVtoR(float Y, float U, float V){
  float R=(V/0.877)+Y;
  return R;
}

// Function 1069
vec2 Cam_GetUVFromWindowCoord( vec2 vWindow, vec2 res )
{
    vec2 vScaledWindow = vWindow;
    vScaledWindow.x *= res.y / res.x;

    return (vScaledWindow * 0.5 + 0.5);
}

// Function 1070
vec3 terrain_map( vec2 p )
{
  return vec3(0.7, .55, .4)+texture(iChannel1, p*2.).rgb*.5; // test-terrain is simply 'sandstone'
}

// Function 1071
vec4   xyzToLuv(vec4 c) {return vec4(   xyzToLuv( vec3(c.x,c.y,c.z) ), c.a);}

// Function 1072
vec2 map(in vec3 l, in qdr q) {
	//https://en.wikipedia.org/wiki/Quadric#Projective_geometry ???
    return (l.xz - l.y) - (q.l.xz - q.l.y) ;
}

// Function 1073
vec2 Cam_GetUVFromWindowCoord( const in vec2 vWindow, float fAspectRatio )
{
    vec2 vScaledWindow = vWindow;
    vScaledWindow.x /= fAspectRatio;

    return (vScaledWindow * 0.5 + 0.5);
}

// Function 1074
vec3 filmicToneMapping(vec3 color)
{
    color = max(vec3(0.), color - vec3(0.004));
    color = (color * (6.2 * color + .5)) / (color * (6.2 * color + 1.7) + 0.06);
    return color;
}

// Function 1075
float cloudSphereMap( vec2 p, mat4 camera, vec3 n, float bias, LameTweaks lame_tweaks )
{
	vec2 p0 = p;

	float pole = 0.1;
	p.y = ( p.y - pole ) / ( 1.0 - 2.0 * pole );

	// p0 is in x 0,1
	// q0 is in x 0,2

	vec3 q = vec3( p * vec2( 2, 1 ), 0.0 );

	vec3 q0 = q;

//	q += vortex_bombing( q.xy,  1.0, 1.0, 1.0, 0.0 ) * POW0( 0.5 ); // 1
//	q += vortex_bombing( q.xy,  2.0, 1.0, 1.0, 0.0 ) * POW1( 0.5 ); // 2
//	q += vortex_bombing( q.xy,  4.0, 1.0, 1.0, 0.0 ) * POW2( 0.5 ); // 3
	q += vortex_bombing( q.xy,  8.0, 3.0, 1.0, 0.9 ) * POW3( 0.5 ); // 4
//	q += vortex_bombing( q.xy, 16.0, 3.0, 1.0, 1.0 ) * POW4( 0.5 ); // 5
	q += vortex_bombing( q.xy, 32.0, 2.7, 5.5, 0.85 ) * POW5( 0.5 ); // 6
//	q += vortex_bombing( q.xy, 64.0, 1.0, 1.0, 0.0 ) * POW6( 0.5 ); // 7

	vec2 qoff = vec2( 0.0, 0 );
#ifdef CLOUD_FLOW
	qoff.x = lame_tweaks.cloud_flow_time * earth_angular_velocity; //cloud flow (doesn't fix black line)
#endif

	NoiseTiledParams ntp;
	ntp.eye = camera[3].xyz;
	ntp.n = n;
	ntp.p = n * earth_radius;
	ntp.bias = bias;

	float a = fbm5_tiled_clouds( q.xy * 4.0 + qoff, ntp );

	a *= 1.0 - smoothstep( 0.5 - pole * 3.4, 0.5, abs( p0.y - 0.5 ) ); // would like to do better than that...

	float a0 = a;

	{
		//increase density on areas that have vortices
		a += length( q - q0 ) * 0.5;
		a += q.z * q.z * 5.0;
	}

	// add a little bit more oompf detail, helps overall + on cloud close ups
	a += a0 * fbm5_tiled_clouds( q.xy * 8.0 + qoff, ntp ) * 0.5;

	a = contrast( a + 0.05, 2.75 ); // higher contrast = deeper blue if we keep negative cloud
	a = soft_max( a, 0.0, 15.0 );
	return a;
}

// Function 1076
vec2 MapEllipDisk(vec2 p) {
    vec2 v = 2.0 * p - 1.0;
    return v * sqrt(1. - 0.5 * v.yx * v.yx);
}

// Function 1077
vec2 sphereMap(vec3 pos, float rad)
{
    return vec2(atan(pos.z, pos.x), acos(pos.y / rad));
}

// Function 1078
vec2 UVMapping( vec2 target )
{
	// need to march vertically to absorb vertical creases, and horizontally for horizontal ones
	// cheat, by seperating these two
	vec2 uv = vec2(0);
	
	const int n = 16;
	const float fudge = 1.0; // use values > 1 to allow for extra ripples we're not measuring
	vec2 d = target/float(n);
	vec2 l;
	l.x = RippleHeight( vec2(0,target.y) );
	l.y = RippleHeight( vec2(target.x,0) );
	for ( int i=0; i < n; i++ )
	{
		vec2 s;
		s.x = RippleHeight( vec2(d.x*float(i),target.y) );
		s.y = RippleHeight( vec2(target.x,d.y*float(i)) );
		//uv.x += sign(d.x)*sqrt(pow(fudge*,2.0)+d.x*d.x);
		//uv.y += sign(d.y)*sqrt(pow(fudge*,2.0)+d.y*d.y);
		uv += sign(d)*sqrt(pow(fudge*(s-l),vec2(2.0))+d*d);
		l = s;
	}
	
	return (uv+vec2(0,1))/vec2(3.0,2.0);
}

// Function 1079
float remap( float t, float a, float b ) {
	return clamp( (t - a) / (b - a), 0.0, 1.0 );
}

// Function 1080
float mapdist(vec3 p, vec3 div) {
 
    p=clamp((p/3.0)*0.5+0.5,0.0,0.999);
    float idz1 = floor((p.z+0.0001)*div.x*div.y)/div.x;
    vec2 uvz1 = vec2(fract(idz1),floor(idz1)/div.y);
    vec2 uv1 = p.xy/div.xy+uvz1;
    float idz2 = floor((p.z+0.0001)*div.x*div.y+1.0)/div.x;
    vec2 uvz2 = vec2(fract(idz2),floor(idz2)/div.y);
    vec2 uv2 = p.xy/div.xy+uvz2;
    
    float d1 = texture(iChannel0, uv1).r;
    float d2 = texture(iChannel0, uv2).r;
    
    return mix(d1, d2, fract((p.z+0.0001)*div.x*div.y));
}

// Function 1081
vec4 mapEnvironment(in vec2 p, in int ballID) {
    vec2 ballPos = getBallPos(ballID);

    float distToEnv = mapEnvironmentNoBalls(p);
    vec2 colliderVel = vec2(0.0, 0.0);
    float colliderType = 0.0;

    for (int id=0; id < NUMBER_OF_BALLS; id++) {
        if (id != ballID) {
            vec2 colliderPos = getBallPos(id);
            distToEnv = min(distToEnv, length(p - colliderPos) - BALL_RADIUS);

            if (length(ballPos - colliderPos) < 2.0 * BALL_RADIUS) {
                colliderVel = getBallVel(id);
                colliderType = 1.0;
            }
        }
    }

    return vec4(distToEnv, colliderVel, colliderType);
}

// Function 1082
float Map(float value, float old_lo, float old_hi, float new_lo, float new_hi)
{
    float old_range = old_hi - old_lo;
    float new_range = new_hi - new_lo;
    return (((value - old_lo) * new_range) / old_range) + new_lo;
}

// Function 1083
vec2 map( in vec3 p )
{
    // box
    float d = sdBox( p, vec3(1.0) );

    // fbm
    vec2 dt = sdFbm( p+0.5, d );

    dt.y = 1.0+dt.y*2.0; dt.y = dt.y*dt.y;
    
    return dt;
}

// Function 1084
vec2 map(vec2 z)
{
  return z;  // identity
}

// Function 1085
vec2 map(in vec4 pos)
{
    vec4 localPos = repetition(pos, vec4(4, 2, 5, 4));
    
    float distance = FAR;
    float distortedDistance = FAR;
    
    distance = min(distance, tesseract(localPos, 1.0));
    distance = max(distance, -hTorus(localPos, 0.5, 0.5));
    
    distance = min(distance, hCylinder(localPos, 0.05));
    distance = min(distance, hSphere(localPos-vec4(vec3(0.0), 1.0), 0.2));
    

    
    vec4 spos = 16.0*pos;
    distance += 0.01*(sin(spos.x)+sin(spos.y)+sin(spos.z)+sin(spos.w));
    distortedDistance = distance-0.01;
    
    distance = min(distance, pos.y+1.0);
    
    return vec2(distance, min(distortedDistance, distance));
}

// Function 1086
float map(vec3 p){
    //instancing:
    // you transform the space so it's a repeating coordinate system
    vec3 q = fract(p) * 2.0 -1.0;
    
  	//sphere map function is the length of the point minus the radius
    //it's negative on the inside of the sphere and positive on the outside and 0 on the surface.
    float radius = sphere_size;
 	return length(q) - radius;   
}

// Function 1087
float map(vec3 p){

    // The height value.
    float c = heightMap(p.xy);
    
    // Back plane, placed at vec3(0., 0., 1.), with plane normal vec3(0., 0., -1).
    // Adding some height to the plane from the texture. Not much else to it.
    return 1. - p.z - c*.1;//texture(texChannel0, p.xy).x*.1;

    
    // Smoothed out.
    //float t =  heightMap(p.xy);
    //return 1. - p.z - smoothstep(0.1, .8, t)*.06 - t*t*.03;
    
}

// Function 1088
float sampleCubeMap(float i, vec3 rd) {
	vec3 col = textureLod(iChannel0, rd * vec3(1.0,-1.0,1.0), 0.0).xyz; 
    return dot(texCubeSampleWeights(i), col);
}

// Function 1089
vec2 mapD0(float t)
{
    return a*cos(t+m)*(b+c*cos(t*7.0+n));
}

// Function 1090
float map(vec3 p) {
    p.y -= .14;
    #ifndef GIF_EXPORT
        pR(p.xz, sin(4. * fTime * PI * .5) * .05);
        pR(p.zy, sin(4. * fTime * PI * 2.) * .03 + .05);
        if (iMouse.x > 0. && iMouse.y > 0.) {
            pR(p.zx, ((iMouse.x/iResolution.x)*2.-1.)*.5);
            pR(p.zy, ((iMouse.y/iResolution.y)*2.-1.)*.5);
        }
   	#else
    	pR(p.zy, .05);
   	#endif
    float d = mHead(p);
    if (d < .1 && ! isBound) {
        float ds = calcDisplacement(p);
        d += ds * .03;
    }
    return d;
}

// Function 1091
vec2 nmaps(vec2 x){ return x*2.-1.; }

// Function 1092
bool map(vec3 p)
{
    p += 0.5;
    vec3 b = abs(p);
    
    bool r;
    
    r =      b.x < 8.0;
    r = r && b.y < 8.0;
    r = r && b.z < 8.0;
    
    r = r && !(b.x < 7.0 && b.y < 7.0 && p.z > -7.0);
   
    r = r || (p.x > 1.0 && p.x < 5.0 && p.z > 1.0 && p.z < 5.0 && p.y > -8.0 && p.y < -3.0);
    r = r || (p.x >-5.0 && p.x <-1.0 && p.z > -5.0 && p.z <-1.0 && p.y > -8.0 && p.y < 0.0);
    
    float ws = 2.0;
    //if(p.y > 7.0 && b.x < ws && b.z < ws) r = false;
    
    return r;
}

// Function 1093
vec3 tonemap(vec3 color)
{
    // Tonemap (fits colors to 0.0-1.0)
    color = 1.0-exp(-color*exposure);

    // sRGB Color Component Transfer: https://www.color.org/chardata/rgb/sRGB.pdf
    color  = vec3(
    color.r > 0.0031308 ? (pow(color.r, 1.0/2.4)*1.055)-0.055 : color.r*12.92,
    color.g > 0.0031308 ? (pow(color.g, 1.0/2.4)*1.055)-0.055 : color.g*12.92,
    color.b > 0.0031308 ? (pow(color.b, 1.0/2.4)*1.055)-0.055 : color.b*12.92);

    return clamp(color, 0.0, 1.0);
}

// Function 1094
vec3 rgbToHpluv(vec3 tuple) {  return lchToHpluv(rgbToLch(tuple)); }

// Function 1095
vec4 texMapSmooth(samplerCube tx, vec3 p){

    // Used as shorthand to write things like vec3(1, 0, 1) in the short form, e.yxy. 
	vec2 e = vec2(0, 1);
  
    // Multiplying the coordinate value by 100 to put them in the zero to 100 pixel range.
    // It was a style choice... which I'm standing by, for now. :)
    p *= 100.;
    
    
    vec3 ip = floor(p);
    // Set up the cubic grid.
    p -= ip; // Fractional position within the cube.
    
    // Smoothing - for smooth interpolation. Comment it out to see the
    //p = p*p*p*(p*(p*6. - 15.) + 10.); // Quintic smoothing. Slower, but derivaties are smooth too.
    //p = p*p*(3. - 2.*p); // Cubic smoothing. 
    //p = mix(p, smoothstep(0., 1., p), .5);
    //vec3 w = p*p*p; p = ( 7. + (p - 7.)*w)*p;	// Super smooth, but less practical.
    //p = .5 - .5*cos(p*3.14159); // Cosinusoidal smoothing.
    // No smoothing. Gives a blocky appearance.
    
     // Smoothly interpolating between the eight verticies of the cube. Due to the shared verticies between
    // cubes, the result is blending of random values throughout the 3D space.
    vec4 c = mix(mix(mix(tMapSm(tx, ip + e.xxx), tMapSm(tx, ip + e.yxx), p.x),
                     mix(tMapSm(tx, ip + e.xyx), tMapSm(tx, ip + e.yyx), p.x), p.y),
                 mix(mix(tMapSm(tx, ip + e.xxy), tMapSm(tx, ip + e.yxy), p.x),
                     mix(tMapSm(tx, ip + e.xyy), tMapSm(tx, ip + e.yyy), p.x), p.y), p.z);
/*   
    // For fun, I tried a straight up average. It didn't work. :)
    vec4 c = (tMapSm(tx, ip + e.xxx) + tMapSm(tx, ip + e.yxx) +
              tMapSm(tx, ip + e.xyx) + tMapSm(tx, ip + e.yyx) +
              tMapSm(tx, ip + e.xxy) + tMapSm(tx, ip + e.yxy) +
              tMapSm(tx, ip + e.xyy) + tMapSm(tx, ip + e.yyy))/8.;
*/ 
    
    return c;

}

// Function 1096
float map_floor(vec3 pos)
{
   return pos.y;
}

// Function 1097
vec2 MapSmokeTrail( vec3 p, Missile missile)
{
  TranslateMissilePos(p, missile);
  float spreadDistance = 1.5;
  p.z+=3.82;

  // map trail by using mod op and ellipsoids
  float s = pModInterval1(p.z, -spreadDistance, .0, min(12., (missile.pos.z-planePos.z)/spreadDistance));     
  float dist = sdEllipsoid(p+vec3(0.0, 0.0, .4), vec3(0.6, 0.6, 3.));   
  dist-= getTrailDensity(p+vec3(10.*s))*0.25;

  return vec2(dist, s);
}

// Function 1098
vec3 simpleReinhardToneMapping(vec3 color)
{
	float exposure = 1.5; // 1.5
	color *= exposure/(1. + color / exposure);
    //return vec3(exposure/(1. + color / exposure));
	color = pow(color, vec3(1. / 1.2)); // gamma = 2.2
	return clamp(color, 0.0, 1.0);
}

// Function 1099
vec3 uvToWorld( vec2 uv, out vec3 n )
{
    for( int i = 0; i < QUAD_COUNT; i++ )
    {
        vec2 uvoff = (uv - quads[i].uv_c)/quads[i].uv_wh;
        if( abs(uvoff.x) < .5
         && abs(uvoff.y) < .5 )
        {
            n = quads[i].n;
            uvoff *= quads[i].scl;
            vec3 u = l2w( vec3(1.,0.,0.), quads[i].n );
            vec3 v = l2w( vec3(0.,0.,1.), quads[i].n );
            return quads[i].p + uvoff.x * u + uvoff.y * v;
        }
    }
    
    return n = vec3(0.);
}

// Function 1100
vec3 TonemapCompressRangeFloat3( vec3 x, float t )
{
	x.r = TonemapCompressRangeFloat( x.r, t );
	x.g = TonemapCompressRangeFloat( x.g, t );
	x.b = TonemapCompressRangeFloat( x.b, t );
	return x;
}

// Function 1101
float map_city(vec3 pos)
{
    if (pos.x<staircase_length/2.)
        return maxdist;
        
    float city = pos.y + floor_height/2.;
        
    if (pos.x>city_stepsize)
    {        
        float line = floor(pos.x/city_stepsize);
        float col = floor(pos.z/city_stepsize);
        blockindex = line + 200.*col;
        float cityhash1 = hash(842.658*blockindex);
        float cityhash2 = pow(hash(534.685*blockindex), city_blheightpow);

        float blocksize = city_minblsize + cityhash1*(city_maxblsize - city_minblsize);
        float blockheight = city_minblheight + cityhash2*(city_maxblheight - city_minblheight)*min((line - 1.)/4., 1.);

        pos.xz = mod(pos.xz, city_stepsize) - vec2(city_stepsize/2.);
    
        city = min(city, max(abs(pos.x) - blocksize/2., abs(pos.z) - blocksize/2.)); 
        city = max(city, pos.y - blockheight);
    }

    return 0.4*min(city, 10.);
}

// Function 1102
vec3 RomBinDaHouseToneMapping(vec3 color)
{
    color = exp( -1.0 / ( 2.72*color + 0.15 ) );
	color = pow(color, vec3(1. / gamma));
	return color;
}

// Function 1103
float map_tablet(vec3 pos, vec3 orig, vec3 size, float flatn, float r, bool btext)
{
    pos.z*= flatn;
    return length(max(abs(pos-orig)-size,0.0)) - r + (!btext || inPic(pos)?0.:0.008*texture(iChannel1, pos.xy*0.6).x);
}

// Function 1104
vec2 inversePerspective_uv(Cam perspectiveCam, vec2 uv_01, screenSpaceQuad ssQuad, worldSpaceQuad wsQuad )
{
    vec3 x_ws = wsQuad.b-wsQuad.a;
    vec3 y_ws = wsQuad.d-wsQuad.a;
    vec3 p_ws = wsQuad.a+uv_01.x*x_ws + uv_01.y*y_ws;
    vec2 puv = camProj(perspectiveCam,p_ws);
	return puv;
}

// Function 1105
vec2 mapUnderWater(vec3 p) {
    vec2 d = vec2(-1.0, -1.0);
    d = vec2(mapTerrain(p-vec3(0.0, FLOOR_LEVEL, 0.0), FLOOR_TEXTURE_AMP), TYPE_FLOOR);
    d = opU(d, vec2(mapSand(p-vec3(0.0, FLOOR_LEVEL, 0.0)), TYPE_SAND));
    return d;
}

// Function 1106
vec2 uvFromBasis(const in vec3 lobeMean, 
                 const in vec3 lobeTangent, 
                 const in vec3 lobeBitangent,
                 const in vec2 lobeScale,
                 const in vec2 lobeBias,
                 const in vec3 positionWorldSpace) {
	vec3 positionLobeSpace = positionWorldSpace * mat3(lobeTangent, lobeBitangent, lobeMean);

	return positionLobeSpace.xy * lobeScale + lobeBias;
}

// Function 1107
float map( in vec3 p, out vec4 oTrap, in vec4 c )
{
    vec4 z = vec4(p,0.0);
    float md2 = 1.0;
    float mz2 = dot(z,z);

    vec4 trap = vec4(abs(z.xyz),dot(z,z));

    float n = 1.0;
    for( int i=0; i<numIterations; i++ )
    {
        // dz -> 2·z·dz, meaning |dz| -> 2·|z|·|dz|
        // Now we take the 2.0 out of the loop and do it at the end with an exp2
        md2 *= 4.0*mz2;
        // z  -> z^2 + c
        z = qsqr(z) + c;  

        trap = min( trap, vec4(abs(z.xyz),dot(z,z)) );

        mz2 = qlength2(z);
        if(mz2>4.0) break;
        n += 1.0;
    }
    
    oTrap = trap;

    return 0.25*sqrt(mz2/md2)*log(mz2);  // d = 0.5·|z|·log|z|/|z'|
}

// Function 1108
float map(float x, float a1, float a2, float b1, float b2){
  return b1 + (b2-b1) * (x-a1) / (a2-a1);
}

// Function 1109
float map(vec3 p) {
    float d = sdPlane(p, vec4(0.0, 1.0, 0.0, 0.0));
    
    float rot_x = iTime * 3.1415 * 0.2;
    float cx = cos(rot_x);
    float sx = sin(rot_x);
    
    float rot_z = iTime * 3.1415 * 0.125;
    float cz = cos(rot_z);
    float sz = sin(rot_z);
    
    p = vec3(
        p.x,
        p.y * cx - p.z * sx,
        p.z * cx + p.y * sx
    );
    
    p = vec3(
        p.x * cz - p.y * sz,
        p.y * cz + p.x * sz,
        p.z
    );
    
    d = opU(d, sdBox(p - vec3(0.0, 1.5, -1.5), vec3(1.6, 1.5, 0.1)));
    d = opU(d, sdBox(p - vec3(1.5, 1.5, -0.25), vec3(0.1, 0.75, 2.25)));
    
    d = opU(d, opU_v2(sdSphere(p, 1.0), sdBox(p - vec3(0.75, 0.75, -0.75), vec3(0.75 - 0.025)) - 0.025, 0.1));
    //d = opU(d, opU_v2(sdSphere(p, 1.0), sdBox(p - vec3(0.75 * 3.0, 0.75, -0.75 * 3.0), vec3(0.75)) - 0.025));
    
    return d;
}

// Function 1110
vec2 map( vec3 p )
{
    vec3 a = vec3(0.0,-1.0,0.0);
    vec3 b = vec3(0.0, 0.0,0.0);
    vec3 c = vec3(0.0, 0.5,-0.5);
	float th = 0.0;
	float hm = 0.0;
	float id = 0.0;
    
    float dm = length(p-a);

    for( int i=0; i<8; i++ )
	{	
        vec3 bboxMi = min(a,min(b,c))-0.3;
    	vec3 bboxMa = max(a,max(b,c))+0.3;
        
        float bv = sdBox( p-0.5*(bboxMa+bboxMi), 0.5*(bboxMa-bboxMi) );
        //if( bv<dm )
        {            
            vec2 h = sdBezier( p, a, b, c );
            float kh = (th + h.y)/8.0;
            float ra = 0.3 - 0.28*kh + 0.3*exp(-15.0*kh);
            float d = h.x - ra;
            if( d<dm ) { dm=d; hm=kh; }
    	}
		
        vec3 na = c;
		vec3 nb = c + (c-b);
		vec3 dir = normalize(-1.0+2.0*hash3( id+13.0 ));
		vec3 nc = nb + 1.0*dir*sign(-dot(c-b,dir));

		id += 3.71;
		a = na;
		b = nb;
		c = nc;
		th += 1.0;
	}

	return vec2( dm*0.5, hm );
}

// Function 1111
vec4 hsluvToRgb_(vec4 c)
{
    return vec4(
        pow(
        	hsluvToRgb( vec3(c.r*360.,clamp(c.gb*100.,0.,100.)) )
        ,1./GAMMA)
    ,c.a);
}

// Function 1112
float TonemapCompressRangeFloat( float x, float t )
{
	return ( x < t ) ? x : t + TonemapCompressRangeNorm( (x-t) / (1.0f - t) ) * (1.0f - t);
}

// Function 1113
float map(vec3 p){
    
    // spheres
    float d = (-1.*length(p)+3.)+1.5*noise(p);    
    d = min(d, (length(p)-1.5)+1.5*noise(p) );  
    
    // links
    float m = 1.5; float s = .03;    
    d = smin(d, max( abs(p.x)-s, abs(p.y+p.z*.2)-.07 ) , m);          
    d = smin(d, max( abs(p.z)-s, abs(p.x+p.y/2.)-.07 ), m );    
    d = smin(d, max( abs(p.z-p.y*.4)-s, abs(p.x-p.y*.2)-.07 ), m );    
    d = smin(d, max( abs(p.z*.2-p.y)-s, abs(p.x+p.z)-.07 ), m );    
    d = smin(d, max( abs(p.z*-.2+p.y)-s, abs(-p.x+p.z)-.07 ), m );
    
    return d;
}

// Function 1114
v0 suv(v1 a){return a.x+a.y;}

// Function 1115
vec2 calSphereUV(vec3 pos)
{
    pos = rot(iTime) * pos;
    
    vec4 sph = sphere_info;
    
    vec3 dir = normalize(pos - sph.xyz);
    vec2 dir2 = normalize(dir.xz); 
    
    return acos(vec2(dir2.y,dir.y)) / 3.1415926;
}

// Function 1116
vec3 resolveSpriteUV(const in int x, vec2 uv) {
    if (x < 10) {
        return simpleBitmap(digitBitmaps(x), 4.0, 5.0, uv) == 1.0 ? vec3(1.0, 1.0, 1.0) : vec3(-1.0, -1.0, -1.0);
    }
    return vec3(0.0, 0.0, 0.0);
}

// Function 1117
vec3 tonemapping(vec3 color, float exposure)
{
	color *= exposure;
    
    float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.10;
	float E = 0.015;
	float F = 0.40;
	float W = 11.2;
	color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
	float white = ((W * (A * W + C * B) + D * E) / (W * (A * W + B) + D * F)) - E / F;
	color /= white;
    
    return color;
}

// Function 1118
void CubeMapToSH2(out mat3 shR, out mat3 shG, out mat3 shB) {
    // Initialise sh to 0
    shR = mat3(0.0);
    shG = mat3(0.0);
    shB = mat3(0.0);
    
    vec2 ts = vec2(textureSize(iChannel0, 0));
    float maxMipMap = log2(max(ts.x, ts.y));
    float lodBias = maxMipMap-6.0;
    

    for (int i=0; i < ENV_SMPL_NUM; ++i) {
        vec3 rayDir = SpherePoints_GoldenAngle(float(i), float(ENV_SMPL_NUM));
        vec3 color = sampleReflectionMap(rayDir, lodBias);

        mat3 sh = shEvaluate(rayDir);
        shR = shAdd(shR, shScale(sh, color.r));
        shG = shAdd(shG, shScale(sh, color.g));
        shB = shAdd(shB, shScale(sh, color.b));            
    }

    // integrating over a sphere so each sample has a weight of 4*PI/samplecount (uniform solid angle, for each sample)
    float shFactor = 4.0 * PI / float(ENV_SMPL_NUM);
    shR = shScale(shR, shFactor );
    shG = shScale(shG, shFactor );
    shB = shScale(shB, shFactor );
}

// Function 1119
float toneMap(float c)
{
    switch (method) {
  	  case 0: // to sRGB gamma via filmic-like Reinhard-esque combination approximation
	    c = c / (c + .1667); // actually cancelled out the gamma correction!  Probably cheaper than even sRGB approx by itself.
		break; 	// considering I can't tell the difference, using full ACES+sRGB together seems quite wasteful computationally.
  	  case 1: // ACES+sRGB, which I approximated above
	    c = ((c*(2.51*c+.03))/(c*(2.43*c+.59)+.14));
	    c = pow(c, 1./2.2); // to sRGBish gamut
	    break;
  	  default: // sRGB by itself
	    c = pow(c, 1./2.2);
        break;
    }
    return c;
}

// Function 1120
vec3 map( in vec3 p )
{
    vec3 p00 = p;
	
	float r, d; vec3 n, s, res;
	
    #ifdef SHOW_SPHERES
	#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))
	#else
	#define SHAPE (vec3(d-abs(r), sign(r),d))
	#endif

    int l = L;
    int m = M;
    #ifdef AUTOMATIC
    int x = ((iFrame>>1)&0x7FF)/15;
    int Y = int(floor(sqrt(0.25+float(2*x))-0.5));
    int X = int(floor(float(x)-0.5*float(Y+Y*Y)));
    l = Y+1;
    m = X+1;
    #endif
	d=length(p00); n=p00/d; r = SH(l, m, p ); s = SHAPE; res = s;
	
	return vec3( res.x, 0.5+0.5*res.y, res.z );
}

// Function 1121
float mapEnvironmentNoBalls(in vec2 p) {
    float container = min(p.y + 0.4, 0.8 - abs(p.x));

    vec2 q1 = mod(p, vec2(0.1, 0.2)) - vec2(0.05, 0.1);
    vec2 q2 = mod(p + vec2(0.05, 0.1), vec2(0.1, 0.2)) - vec2(0.05, 0.1);
    vec2 q3 = abs(p - vec2(0.0, 0.1)) - vec2(0.51, 0.11);

    float bbox = length(max(q3, 0.0)) + min(max(q3.x, q3.y), 0.0);
    float pegs = max(min(length(q1) - 0.01, length(q2) - 0.01), bbox);

    vec2 q4 = abs(vec2(mod(p.x, 0.1) - 0.05, p.y + 0.25)) - vec2(0.005, 0.15);
    bbox = abs(p.x) - 0.675;
    float bins = max(length(max(q4, 0.0)) + min(max(q4.x, q4.y), 0.0), bbox);

    p.x = abs(p.x);
    p.x -= 0.92;
    p.y -= 0.8;
    p *= mat2(0.5, -0.866, 0.866, 0.5);
    vec2 q5 = abs(p) - vec2(0.01, 1.0);
    float funnel = length(max(q5, 0.0)) + min(max(q5.x, q5.y), 0.0);

    return min(container, min(pegs, min(bins, funnel)));
}

// Function 1122
Model map(vec3 p) {
    
    // Spin the whole model
    spin(p);
    
    // Fold space into an icosahedron,
    // disable this to get a better idea of what
    // geodesicTri is doing
    fold(p);
    
	float subdivisions = animSubdivitions(1., 10.);
	vec3 point = geodesicTri(p, subdivisions);

	float sphere = length(p - point) - .195 / subdivisions; 
    
    // Use red/green to indicate point's position,
    // you can see that space always mirrored at the
    // Icosahedron's schwarz triangle
    vec3 color = vec3(0, point.yx * 3. + .5);
    color = clamp(color, 0., 1.);

	return Model(sphere, color);
}

// Function 1123
vec2 texNormalMap(in vec2 uv)
{
    vec2 s = 1.0/heightMapResolution.xy;
    
    float p = texture(heightMap, uv).x;
    float h1 = texture(heightMap, uv + s * vec2(textureOffset,0)).x;
    float v1 = texture(heightMap, uv + s * vec2(0,textureOffset)).x;
       
   	return (p - vec2(h1, v1));
}

// Function 1124
vec3 map_color(vec3 p) {
    return p;
}

// Function 1125
vec2 map(vec3 pos)
{
    vec3 posr = rotateVec2(pos);
    //return vec2(map_s(posr), LAMPG_OBJ);
    #ifdef bulb
    vec2 res = opU(vec2(map_lampg(posr), LAMPG_OBJ),
                   vec2(map_bulb(posr),  BULB_OBJ));
    #else
    vec2 res = vec2(map_lampg(posr), LAMPG_OBJ);
    #endif
    #ifdef metal_rings
    res = opU(res, vec2(map_metalrings(posr),  METALRINGS_OBJ));
    #endif
    #ifdef supports
    res = opU(res, vec2(map_supports(posr),  SUPPORTS_OBJ));
    #endif
    return res;
}

// Function 1126
vec3 map( in vec3 pos )
{
    vec3 res = mapShadow(pos);
        
    res = mapCrapInTheAir(pos, res);

    return res;
}

// Function 1127
float mapHeightHQ(in vec3 rp)
{
    float bottom = mapBottom(rp);
    float limit = smoothstep(55., 90., abs(rp.x)) * 4.;
    bottom -= (0.4 * smoothstep(0.3, 0.6, noise(rp.xz * .23))) * limit;
    bottom += (0.3 * smoothstep(0.2, 0.6, noise(rp.xz * .43))) * limit;
    return rp.y - bottom;
}

// Function 1128
vec3 hpluvToRgb(float x, float y, float z) {return hpluvToRgb( vec3(x,y,z) );}

// Function 1129
float map(float x) {
    x += sin(iTime * 0.3) * 0.5;
    return max(-pow(x, 2.0)+0.3, abs(x) - 0.5) - 0.2;
}

// Function 1130
float map(in vec3 p) {
    
    
   // Cubes, for a simpler, more orderly scene.
   //p = abs(fract(p) - .5);    
   //return max(max(p.x, p.y), p.z) - .225;
   
   // Unique identifier for the cube, but needs to be converted to a unique ID
   // for the nearest octahedron. The extra ".5" is to save a couple of 
   // of calculations. See below.
   vec3 ip = floor(p) + .5;
    
   p -= ip; // Break space into cubes. Equivalent to: fract(p) - .5.
    
   // Stepping trick used to identify faces in a cube. The center of the cube face also
   // happens to be the center of the nearest octahedron, so that works out rather well. 
   // The result needs to be factored a little (see the hash line), but it basically  
   // provides a unique octahedral ID. Fizzer provided a visual of this, which is easier 
   // to understand, and worth taking a look at.
   vec3 q = abs(p); 
   q = step(q.yzx, q.xyz)*step(q.zxy, q.xyz)*sign(p); // Used for cube mapping also.
   
   // Put the ID into a hash function to produce a unique random number. Reusing "q" to
   // save declaring a float. Don't know if it's faster, but it looks neater, I guess.
   q.x = fract(sin(dot(ip + q*.5, vec3(111.67, 147.31, 27.53)))*43758.5453);
    
   // Use the random number to orient a square tube in one of three random axial
   // directions... See Fizzer's article explanation. It's better. :) By the way, it's
   // possible to rewrite this in "step" form, but I don't know if it's quicker, so I'll
   // leave it as is for now.
   p.xy = abs(q.x>.333 ? q.x>.666 ? p.xz : p.yz : p.xy);
   return max(p.x, p.y) - .2;   

}

// Function 1131
float map( vec3 c ) 
{
	vec3 p = c + 0.5;
	
	float h = textureLod( iChannel0, fract(p.xz/iChannelResolution[0].xy), 0.0 ).x;
    
    float dm = 1e10;
    for( int i=0; i<25; i++ )
    {
        vec2 pa = path( 60.0*float(i)/25.0 ).xz;
        dm = min( dm, length2(pa-p.xz) );
    }
    
    float isc = step( sqrt(dm), 5.0 );
    
    h = -10.0 + 16.0*h - 6.0*isc;//*(0.2 + h*2.0);
    
    h += 10.0*smoothstep( 0.9,0.91,textureLod( iChannel2, 0.25*fract(p.xz/iChannelResolution[2].xy), 0.0 ).x)*(1.0-isc);
    
    
	return step( p.y, h );
}

// Function 1132
vec3   luvToLch(float x, float y, float z) {return   luvToLch( vec3(x,y,z) );}

// Function 1133
vec3 RGBtoYUV(vec3 rgb) {
    vec3 yuv;
    yuv.r = rgb.r * 0.2126 + 0.7152 * rgb.g + 0.0722 * rgb.b;
    yuv.g = (rgb.b - yuv.r) / 1.8556;
    yuv.b = (rgb.r - yuv.r) / 1.5748;

    // Adjust to work on GPU
    yuv.gb += 0.5;

    return yuv;
}

// Function 1134
float mapf(vec2 p) {
    float d0 = length(p) - 0.4;
    float d1 = length(p - vec2(0.3, -0.1)) - 0.4;
    float d2 = length(p - vec2(0.1, mix(0.1,0.3,sin(iTime*0.2)*0.5+0.5))) - 0.1;
    
    //p -= vec2(0.5);
    rotate(p, radians(iTime*10.0));
        
    float d3 = sdBox(p - vec2(0.0, -0.45), vec2(0.25,0.05));
    float d4 = sdBox(p, vec2(2.0/64.0,0.3));
    
    float r = radians(iTime*5.0);
    float d5 = dot(p, anglevec(r));
    float d6 = dot(p, anglevec(-r-radians(90.0)));
    
    float d = min(d2, max(d0, -d1));
    return min(d, d3);
}

// Function 1135
float map(vec3 pos)
{
    float disth = map_hor(pos);
    float distv = map_ver(pos);
    return mix(min(disth, distv), map_s2(pos), gtf);
}

// Function 1136
vec2 map(vec3 p) {
    vec3 v = mod(p,13.)-13./2.-vec3(0,4,0);
	vec2 sv = vec2(length(v)-2., 0.);
    vec2 pv = vec2(p.y+2., 1.);
    return ((sv.x<pv.x)?sv:pv);
}

// Function 1137
vec2 map_s(vec3 pos)
{
    vec2 wall = vec2(sdBox(pos-wallPos, wallSize), BRICKS_OBJ);
    
    return wall;
}

// Function 1138
vec3 Tonemap_ACES(const vec3 x) {
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

// Function 1139
vec4 map(vec3 p)
{
   	float scale = 3.;
    float dist = 0.;
    
    float x = 6.;
    float z = 6.;
    
    vec4 disp = displacement(p);
        
    float y = 1. - smoothstep(0., 1., disp.x) * scale;
    
    #ifdef USE_SPHERE_OR_BOX
    	#ifdef POSITIVE_DISPLACE
        	dist = osphere(p, +5.-y);
    	#else
        	dist = osphere(p, +3.+y);
    	#endif
    #else    
        #ifdef POSITIVE_DISPLACE
        	if ( p.y > 0. ) dist = obox(p, vec3(x,1.-y,z));
        	else dist = obox(p, vec3(x,1.,z));
    	#else
        	if ( p.y > 0. ) dist = obox(p, vec3(x,y,z));
        	else dist = obox(p, vec3(x,1.,z));
    	#endif
	#endif
    
    return vec4(dist, disp.yzw);
}

// Function 1140
vec3 colorMap(vec3 p) {
			
		vec3 u = p;
		vec4 r = 0.8 * vec4(0.09, .077, .055, 0.01);
		float displace[6];
		
		u *= rotY(rotationY);
		u *= rotX(rotationX);
		
		p = deformation(p);
		vec3 s = vec3(length(p) - 1., 0., 0.);
		s.x += displacements(u, displace);
		
		u.x = abs(u.x);
		u -= vec3(0.17, 0.40 - 0.01 * elevation, -0.46);
		float n = length(u.xy);
		s.z = atan(u.y, u.x);
		
		/*
		* Colors for the displacements, the values displace[i]
		* make reference to the return values from the displacement function
		*/
		
		s.y = step(1. + displace[1], 0.98);
		s.y += (3. - s.y) * step(-n, -r.y) * step(n, r.x) * step(u.z, 0.0);
		s.y += (4. - s.y) * step(-n, -r.z) * step(n, r.y) * step(u.z, 0.0);
		s.y += (3. - s.y) * step(n, r.z) * step(u.z, 0.0);
		
		u.xy += vec2(-0.05, -0.03);
		s.y += (0. - s.y) * step(length(u.xy), r.w) * step(u.z, 0.0);
		
		s.y += (0. - s.y) * step(1. - displace[2], 0.9999);
		s.y += (2. - s.y) * step(1. - displace[4], 0.995);
		s.y += (3. - s.y) * step(1. - displace[5], 0.99);
		s.y += (1. - s.y) * step(1. - displace[0], 0.9999);
		
		s.x *= 0.2;
		
		return s;
	}

// Function 1141
vec3 linearToneMapping(vec3 color)
{
    float exposure = 1.;
    color = clamp(exposure * color, 0., 1.);
    color = pow(color, vec3(1. / gamma));
    return color;
}

// Function 1142
float mapWaterDetailed(vec3 p) {
    return sdPlane(p)+heightMapWaterDetailed(p, iTime);
}

// Function 1143
vec2 map_slime(vec3 pos)
{
    pos.y+= 2.8;
    float slime = udRoundBox(pos, vec3(5.35, 1.35, 1.85), 0.6);
    return vec2(max(slime, pos.y -0.45), SLIME_OBJ);
}

// Function 1144
ray uvToCameraRay(vec2 uv, float projectionDist)
{
    ray cameraRay;
    cameraRay.direction = normalize(vec3(uv.x, uv.y, projectionDist));
    return cameraRay;
}

// Function 1145
vec3   luvToXyz(float x, float y, float z) {return   luvToXyz( vec3(x,y,z) );}

// Function 1146
vec4 map(vec3 p)
{
    vec4 res = dBloomObjects(p);
    vec2 res2 = dStructure(p);
    if(res2.x < res.x) res.xy = res2.xy;
    
    res2 = dWater(p);
    if(res2.x < res.x) res.xy = res2.xy;
    
    res2 = dPortalA(p);
    if(res2.x < res.x) res.xy = res2.xy;
    
    res2 = dPortalB(p);
    if(res2.x < res.x) res.xy = res2.xy;
    
    res2 = dPlatforms(p);
    if(res2.x < res.x) res.xy = res2.xy;
    
    return res;
}

// Function 1147
float MapWater(vec3 p)
{
  return p.y - (-GetWaterWave(p)*0.05);
}

// Function 1148
vec3 RomBinDaHouseToneMapping(vec3 color)
{
    color = exp( -1.0 / ( 2.72*color + 0.15 ) );
    color = pow(color, vec3(1. / gamma));
    return color;
}

// Function 1149
float map(vec3 rp){
    vec3 sp = rp/length(rp);
    vec3 cell = floor(sp*60.);
    vec2 index = floor(iResolution.xy*hash23(cell));
    vec3 ABc = vec3(0);
    vec3 d;
    for(d.x = -1.; d.x <= 1.; d.x++){
        for(d.y = -1.; d.y <= 1.; d.y++){
            for(d.z = -1.; d.z <= 1.; d.z++){
                if(dot(d,d)==0.) d.z++;
                vec2 neighborIndex = floor(iResolution.xy*hash23(cell+d));
                vec4 neighbor = texelFetch(iChannel0, ivec2(neighborIndex),0);
                vec3 neighborp = vec3(sin(neighbor.x)*cos(neighbor.y),sin(neighbor.y),cos(neighbor.x)*cos(neighbor.y));
                float s = exp(-100.*length(neighborp-sp));
                ABc += vec3(neighbor.zw,1)*s;
            }
        }
    }
    ABc /= max(.01,ABc.z);
    return min(length(rp)-1.-ABc.x*.10+ABc.y*.01,10.-length(rp));
}

// Function 1150
vec2 map(vec3 p) {
    vec2 d = vec2(-1.0, -1.0);
    d = vec2(mapTerrain(p-vec3(0.0, FLOOR_LEVEL, 0.0), FLOOR_TEXTURE_AMP), TYPE_FLOOR);
    //d = opU(d, vec2(mapSand(p-vec3(0.0, FLOOR_LEVEL, 0.0)), TYPE_SAND));
    return d;
}

// Function 1151
vec2 map( in vec3 pos )
{
    vec2 res = //opU
        ( 
           // vec2( sdPlane(     pos), 1.0 ),
	                vec2( sdSphere2(    pos, 1. ), 86.9 ) 
                  );
    

    
    
//    res = opU( res, vec2( sdBox(       pos-vec3( 1.0,0.25, 0.0), vec3(0.25) ), 3.0 ) );
//    res = opU( res, vec2( udRoundBox(  pos-vec3( 1.0,0.25, 1.0), vec3(0.15), 0.1 ), 41.0 ) );
	///res = opU( res, vec2( sdTorus(     pos-vec3( 0.0,0.50, 2.0), vec2(0.45,0.15) ), 75.0 ) );
//    res = opU( res, vec2( sdCapsule(   pos,vec3(-1.3,0.10,-0.1), vec3(-0.8,0.50,0.2), 0.1  ), 31.9 ) );
//	res = opU( res, vec2( sdTriPrism(  pos-vec3(-1.0,0.25,-1.0), vec2(0.25,0.05) ),43.5 ) );
//	res = opU( res, vec2( sdCylinder(  pos-vec3( 1.0,0.30,-1.0), vec2(0.1,0.2) ), 8.0 ) );
//	res = opU( res, vec2( sdCone(      pos-vec3( 0.0,0.50,-1.0), vec3(0.8,0.6,0.3) ), 55.0 ) );
//	res = opU( res, vec2( sdTorus82(   pos-vec3( 0.0,0.25, 2.0), vec2(0.20,0.05) ),50.0 ) );
//	res = opU( res, vec2( sdTorus88(   pos-vec3(-1.0,0.25, 2.0), vec2(0.20,0.05) ),43.0 ) );
//	res = opU( res, vec2( sdCylinder6( pos-vec3( 1.0,0.30, 2.0), vec2(0.1,0.2) ), 12.0 ) );
//	res = opU( res, vec2( sdHexPrism(  pos-vec3(-1.0,0.20, 1.0), vec2(0.25,0.05) ),17.0 ) );

//    res = opU( res, vec2( opS(
//		             udRoundBox(  pos-vec3(-2.0,0.2, 1.0), vec3(0.15),0.05),
//	                 sdSphere(    pos-vec3(-2.0,0.2, 1.0), 0.25)), 13.0 ) );
 //   res = opU( res, vec2( opS(
//		             sdTorus82(  pos-vec3(-2.0,0.2, 0.0), vec2(0.20,0.1)),
//	                 sdCylinder(  opRep( vec3(atan(pos.x+2.0,pos.z)/6.2831,
//											  pos.y,
//											  0.02+0.5*length(pos-vec3(-2.0,0.2, 0.0))),
//									     vec3(0.05,1.0,0.05)), vec2(0.02,0.6))), 51.0 ) );
//	res = opU( res, vec2( 0.7*sdSphere(    pos-vec3(-2.0,0.25,-1.0), 0.2 ) + 
//					                   0.03*sin(50.0*pos.x)*sin(50.0*pos.y)*sin(50.0*pos.z), 
//                                       65.0 ) );
//	res = opU( res, vec2( 0.5*sdTorus( opTwist(pos-vec3(-2.0,0.25, 2.0)),vec2(0.20,0.05)), 46.7 ) );

//    res = opU( res, vec2(sdConeSection( pos-vec3( 0.0,0.35,-2.0), 0.15, 0.2, 0.1 ), 13.67 ) );

 //   res = opU( res, vec2(sdEllipsoid( pos-vec3( 1.0,0.35,-2.0), vec3(0.15, 0.2, 0.05) ), 43.17 ) );
        
    return res;
}

// Function 1152
vec3 dirFromUv(Camera cam,vec2 uv
){return normalize(vec3(uv,cam.focalLength))*rotationMatrix(cam.rot);}

// Function 1153
vec3 yuv2rgb(vec3 yuv)
{
    return vec3(yuv.r + 1.140 * yuv.b, yuv.r - 0.395*yuv.g - 0.581*yuv.b, yuv.r + 2.032*yuv.g);
}

// Function 1154
vec3 SynthesisTonemap(vec3 color, float m, float a, float s, float L)
{
    float c = SolveC(m, a, s);
    float t = SolveT(m, a, s, c); // slope at y = 0.5 (x = c).
    
    vec3 result;
    result.x = (color.x < 0.5f) ? SynthesisLow(color.x, m, a, s) : SynthesisHigh(color.x, c, t, L);
    result.y = (color.y < 0.5f) ? SynthesisLow(color.y, m, a, s) : SynthesisHigh(color.y, c, t, L);
    result.z = (color.z < 0.5f) ? SynthesisLow(color.z, m, a, s) : SynthesisHigh(color.z, c, t, L);
    
    return result;
}

// Function 1155
vec3 luvToXyz(vec3 tuple) {
    float L = tuple.x;

    float U = tuple.y / (13.0 * L) + 0.19783000664283681;
    float V = tuple.z / (13.0 * L) + 0.468319994938791;

    float Y = hsluv_lToY(L);
    float X = 2.25 * U * Y / V;
    float Z = (3./V - 5.)*Y - (X/3.);

    return vec3(X, Y, Z);
}

// Function 1156
bool rayQuadUV( vec3 ro, vec3 rd, vec3 po, vec3 pn, vec2 psz, out vec2 uv, out float rt )
{
    rt = rayPlane( ro, rd, po, pn );
    if( !(rt > 0.) ) return false; // NaN caught here!
    vec3 pos = ro + rt * rd;
    float x = dot(pos - po, l2w( vec3(1.,0.,0.), pn ) );
    float y = dot(pos - po, l2w( vec3(0.,0.,1.), pn ) );
    uv = vec2(x,y)/psz;
    if( abs(uv.x) >= .5001 || abs(uv.y) >= .5001 ) return false;
    uv += .5;
    return true;
}

// Function 1157
vec3 mapMoss( in vec3 pos, float h, vec3 cur)
{
    vec3 res = cur;

    float db = pos.y-2.2;
    if( db<res.x )
    {
    const float gf = 2.0;
    
    vec3 qos = pos * gf;
    vec2 n = floor( qos.xz );
    vec2 f = fract( qos.xz );

    for( int k=ZERO; k<2; k++ )
    {
        for( int j=-1; j<=1; j++ )
        for( int i=-1; i<=1; i++ )
        {
            vec2  g = vec2( float(i), float(j) );
            vec2  o = hash2( n + g + vec2(float(k),float(k*5)));
            vec2  r = g - f + o;

            vec2 ra  = hash2( n + g + vec2(11.0, 37.0) + float(2*k) );
            vec2 ra2 = hash2( n + g + vec2(41.0,137.0) + float(3*k) );

            float mh = 0.5 + 1.0*ra2.y;
            vec3 ros = qos - vec3(0.0,h*gf,0.0);

            vec3 rr = vec3(r.x,ros.y,r.y);

            rr.xz = reflect( rr.xz, normalize(-1.0+2.0*ra) );

            rr.xz += 0.5*(-1.0+2.0*ra2);
            vec2 se  = sdLineOriY( rr, gf*mh );
            float sey = se.y;
            float d = se.x - 0.05*(2.0-smoothstep(0.0,0.1,abs(se.y-0.9)));

            vec3 pp = vec3(rr.x,mod(rr.y+0.2*0.0,0.4)-0.2*0.0,rr.z);

            float an = mod( 21.0*floor( (rr.y+0.2*0.0)/0.4 ), 1.57 );
            float cc = cos(an);
            float ss = sin(an);
            pp.xz = mat2(cc,ss,-ss,cc)*pp.xz;

            pp.xz = abs(pp.xz);
            vec3 ppp = (pp.z>pp.x) ? pp.zyx : pp; 
            vec2 se2 = sdLineOri( ppp, vec3( 0.4,0.3,0.0) );
            vec2 se3 = sdLineOri( pp,  vec3( 0.2,0.3,0.2) ); if( se3.x<se2.x ) se2 = se3;
            float d2 = se2.x - (0.02 + 0.03*se2.y);

            d2 = max( d2, (rr.y-0.83*gf*mh) );
            d = smin( d, d2, 0.05 );

            d /= gf;
            d *= 0.9;
            if( d<res.x )
            {
                res.x = d;
                res.y = MAT_MOSS;
                res.z = clamp(length(rr.xz)*4.0+rr.y*0.2,0.0,1.0);
                float e = clamp((pos.y - h)/1.0,0.0,1.0);
                res.z *= 0.02 + 0.98*e*e;
                
                if( ra.y>0.85 && abs(se.y-0.95)<0.1 ) res.z = -res.z;
            }
        }
    }

    }
    
    return res;
}

// Function 1158
float map(vec3 p) {
	float s = length(p)-1.0;
	
	return s + snoise((p*10.0 + iTime)-0.8)*0.005;
}

// Function 1159
vec2 map(vec3 p){
	vec2 d= vec2(10e7);
   
    p -= path(p.z);
    
    //p = pmod(p,1.);
   
    //p = abs(p);
    
    
    //d.x = min(d.x, abs(p.y - 1.)-0.);
   
    
    //d.x = min(d.x, abs(p.y + 1.2));
    
    
    p.xz = pmod(p.xz,1.);
   	
    
    vec3 q = p;
    
    q.y = pmod(q.y,1.);
    
    q = abs(q);
    p.y = pmod(q.y,1.);
   	
   	
    //d.x = min(d.x, max(q.x,q.y) - 0.04);
    d.x = min(d.x, max(q.y,q.z) - 0.03);
    d.x = min(d.x, length(p.xz) - 0.03);
    
    
    d.x = min(d.x, length(p) - 0.1);
   
    
    
    d.x = abs(d.x) - 0.00;
    d.x += 0.001;
    d.x *= 0.2;
    
    return d;
}

// Function 1160
float map_s2(vec3 pos)
{
    return mix(map_s(pos), abs(pos.z) - fr*1.1, smoothstep(14., 23., iTime));
}

// Function 1161
vec3 hsluvToLch(float x, float y, float z) {return hsluvToLch( vec3(x,y,z) );}

// Function 1162
vec2 map( vec3 p )
{
	vec3 op = p;
	{
	float an = 0.35;
	float co = cos( an );
	float si = sin( an );
	mat2  ma = mat2( co, -si, si, co );
	p.xy = ma*p.xy;
	}
	
	p.y -= 4.0;

    float d = length( (p-vec3(0.0,-0.1,0.0))*vec3(1.0,3.0,1.0)) - 0.4;
	vec2 res = vec2( d/3.0, 1.0 );
	

	if( p.y>0.0 )
	{

	// palito
	vec3 pp = p;
		
		//vec3 q = p - vec3(0.0,-0.25,0.0);
		vec3 q = (p-vec3(0.0,-0.15,0.0))*vec3(1.0,1.5,1.0);
		pp.y = length(q);
		#if 1
		pp.x = 0.35*0.5*atan( q.x, q.z );
		pp.z = 0.35*acos( q.y/length(q) );
		#else
		//pp.xz *= 1.0 + 2.0*length(p.xz)/(0.5+p.y);
		pp.xz -= normalize(pp.xz) * p.y * 0.5;
		
		#endif
	
	pp.xz = (mod(20.0*(pp.xz+0.5),1.0) - 0.5)/20.0;
	
		
		float hh = 0.0;
	vec2 h = sdSegment( vec3(0,0.0,0.0), vec3(0.0,0.5+hh,0.0), pp );
	float sr = 0.01 + 0.001*smoothstep( 0.9,0.99,h.y );
	d = h.x - sr;
	d *= 0.5;
	res.x = smin( d, res.x );
	}

	p.xz = abs(p.xz);

	
	for( int i=0; i<4; i++ )
	{
		float an = 6.2831*float(i)/14.0;
		float id = an;
		float co = cos( an );
		float si = sin( an );
		mat2  ma = mat2( co, -si, si, co );

		vec2  r = ma*p.xz;//-vec2(1.0,0.0));
		vec3  q = vec3( r.x, p.y, r.y );

		an = 0.02*sin(10.0*an);

		co = cos(0.2+an);
		si = sin(0.2+an);
		q.xy = mat2( co, -si, si, co )*q.xy;

		float ss = 1.0 + 0.1*sin(171.0*an);
		q.x *= ss;
		q.x -= 1.0;
		q.y -= 0.15*q.x*(1.0-q.x);
		
        float ra = 1.0 - 0.3*sin(1.57*q.x);
		d = 0.05*(length( q*vec3(1.0,20.0,4.0*ra) ) - 1.0*0.8);
		
        if( d<res.x ) res = vec2( d, 2.0 );
	}
	
	{
    p = op;	
	float an = 0.35*clamp( p.y/3.8, 0.0, 1.0 );
	float co = cos( an );
	float si = sin( an );
	mat2  ma = mat2( co, -si, si, co );
	p.xy = ma*p.xy;

	vec2 h = sdSegment( vec3(0,0.0,0.0), vec3(0.0,4.0-0.2,0.0), p );
	d = h.x - 0.07;
    if( d<res.x ) res = vec2( d, 3.0 );
	}

	return res;
}

// Function 1163
vec3 hsluv_distanceFromPole(vec3 pointx,vec3 pointy) {  return sqrt(pointx*pointx + pointy*pointy); }

// Function 1164
float remap(float value, float low2,float high2,bool c){return remap(value,0.,1.,low2,high2,c);}

// Function 1165
vec3 map( in vec3 p )
{
    vec3 p00 = p;
	
	float r, d; vec3 n, s, res;
	
    #ifdef SHOW_SPHERES
	#define SHAPE (vec3(d-0.35, -1.0+2.0*clamp(0.5 + 16.0*r,0.0,1.0),d))
	#else
	#define SHAPE (vec3(d-abs(r), sign(r),d))
	#endif
    
	d=length(p00); n=p00/d; r = SH(L, M, n ); s = SHAPE; res = s;
	
	return vec3( res.x, 0.5+0.5*res.y, res.z );
}

// Function 1166
vec3 i_spheremap_16( uint data )
{
    vec2 v = unpackSnorm2x8(data);
    float f = dot(v,v);
    return vec3( 2.0*v*sqrt(1.0-f), 1.0-2.0*f );
}

// Function 1167
vec3 MapColor(vec3 srgb)
{
    #if MODE == 0
    return srgb * sRGBtoAP1;
    #else
    return srgb;
    #endif
}

// Function 1168
void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir) {
    
    if (iFrame > 1) {
        discard;
        return;
    }
    
    vec3 col = vec3(0.);
    for (float i = float(ZERO); i < 1.; i += IBL_smp) {
        for (float j = float(ZERO); j < 1.; j += IBL_smp) {
            // sample cosine hemisphere weighted colors
            vec3 smp = cosWeightedHemisphereDirection(rayDir, vec2(i, j));
            col += srgb_linear(textureLod(iChannel0, smp, 5.5).rgb);
        }
    }
    
    // normalize
    fragColor = vec4(col * IBL_smp2, 1.);
}

// Function 1169
float map( in vec3 p )
{
	return sdMetaBalls( p );
}

// Function 1170
vec3 lchToHpluv(vec3 tuple) {  tuple.g /= hsluv_maxSafeChromaForL(tuple.r) * .01;  return tuple.bgr; }

// Function 1171
uint spheremap_16( in vec3 nor )
{
    vec2 v = nor.xy*inversesqrt(2.0*nor.z+2.0);
    return packSnorm2x8(v);
}

// Function 1172
vec4 hpluvToRgb_(vec4 c,float gamma)
{
    return vec4(
        pow(
        	hpluvToRgb( vec3(c.r*360.,clamp(c.gb*100.,0.,100.)) )
        ,1./vec3(gamma))
    ,c.a);
}

// Function 1173
void mainCubemap( out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir )
{
    if (iFrame < 60)
    {
        vec3 rd = rayDir;
        // the direction and color of the sun
        vec3 sun_col = vec3(1., 0.9, 0.6);
        //vec3 sun_dir = normalize(vec3(sin(iTime), 0.75, cos(iTime)));
        vec3 sun_dir = normalize(vec3(0., 0.75, -1.));

        // using triplanar mapping to map the stars texture onto the sky
        vec3 col = pow(TriplannarStarsTexture(rd * 5., rd), vec3(4.));
        col = mix(col, sun_col * 1.2, pow(max(dot(sun_dir, rd), 0.), 200.));

        fragColor = vec4(col, 1.0);
    }
    else
    {
        // not doing the math to calculate things like this after frame 60 to give better preformace
        fragColor = texture(iChannel0, rayDir);
    }
}

// Function 1174
vec3 map( in vec3 p, in vec4 sh_in )
{
    float scale = 2.0;
    
    vec3 p00 = p - vec3( 0.00, 2.5,0.0);
	vec3 p01 = p - vec3(-1.25, 1.0,0.0);
	vec3 p02 = p - vec3( 0.00, 1.0,0.0);
	vec3 p03 = p - vec3( 1.25, 1.0,0.0);
	float r, d; vec3 n, s, res;
	
	#define SHAPE (vec3(d-abs(r), sign(r),d))
    
	d=length(p00); n=p00/d; r = SH_0_0( n ); s = SHAPE; res = s;
	d=length(p01); n=p01/d; r = SH_1_0( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p02); n=p02/d; r = SH_1_1( n ); s = SHAPE; if( s.x<res.x ) res=s;
	d=length(p03); n=p03/d; r = SH_1_2( n ); s = SHAPE; if( s.x<res.x ) res=s;
    
    // mainSphere
    //vec4 sh = vec4(0.25, 0.12, 0.08, 0.15) * scale;
    vec4 sh = sh_in * scale;
	vec3 pos = p - vec3( 0.0,-1.5,0.0);
    
	d=length(pos); 
    n=pos/d; 
    r = SH_0_0(n) * sh[0] + SH_1_0(n) * sh[1] + SH_1_1(n) * sh[2] + SH_1_2(n) * sh[3]; 
    
    s = vec3(d-abs(r), sign(r),d); 
    //res = s;
    
    if( s.x<res.x ) res=s;
    
	return vec3( res.x, 0.5+0.5*res.y, res.z );
}

// Function 1175
vec4 lchToHpluv(float x, float y, float z, float a) {return lchToHpluv( vec4(x,y,z,a) );}

// Function 1176
float map(vec3 pos)
{
    float t = 0.0;
    
    t = SuitUp(pos);
    t = min(t, head(pos));
    
    return t;
}

// Function 1177
vec3 encodePalYuv(vec3 rgb)
{
    return vec3(
        dot(rgb, vec3(0.299, 0.587, 0.114)),
        dot(rgb, vec3(-0.14713, -0.28886, 0.436)),
        dot(rgb, vec3(0.615, -0.51499, -0.10001))
    );
}

// Function 1178
vec2 uvToScreen(vec2 uv, float angle)
{
    vec2 b0 = vec2(sin(-angle), -cos(-angle));
    b0 = normalize(b0);
    vec2 b1 = vec2(-b0.y, b0.x);
    
    vec2 uvCenter = uv - 0.5;
    uvCenter = b0 * uvCenter.x + b1 * uvCenter.y;
    uvCenter /= 1.5;    
    uvCenter.x *= (iResolution.y / iResolution.x);
    return uvCenter + 0.5;
}

// Function 1179
float map(vec3 p)
{
	float d = 100000.0;

    fUnion(d, pRoundBox(p - vec3(0,-2.0,0), vec3(4,0.1,4), 0.2));
	fUnion(d, pSphere(p - vec3(2,0,2), 1.5));
    fUnion(d, pSphere(p - vec3(3.5,-1.0,0.0), 0.8));
    fUnion(d, pTorus(p - vec3(-2,0,2), vec2(1,0.3)));
	fUnion(d, pTorus2(p - vec3(-3,0,2), vec2(1,0.3)));
    fUnion(d, pRoundBox(p - vec3(2,0.6,-2), vec3(0.1,0.1,1), 0.3));
	fUnion(d, pRoundBox(p - vec3(2,0,-2), vec3(0.1,1.5,0.1), 0.3));
	fUnion(d, pRoundBox(p - vec3(2,-0.4,-2), vec3(1.2,0.1,0.1), 0.3));
    fUnion(d, pCapsule(p, vec3(-2,1.5,-2), vec3(-2,-1,-1.0), 0.3));
	fUnion(d, pCapsule(p, vec3(-2,1.5,-2), vec3(-1.0,-1,-2.5), 0.3));
	fUnion(d, pCapsule(p, vec3(-2,1.5,-2), vec3(-3.0,-1,-2.5), 0.3));
	
	return d;
}

// Function 1180
vec1 nmapu(vec1 x){ return x*.5+.5; }

// Function 1181
float map(vec2 p){
	float color = 0.0;
	for(int i=0;i<10;i++){
	color+=cos(50.0*distance(vec2(rnd(i),rnd(20-i)),p))*0.01;
	}
	//color+=cos(distance(vec2(0.0,0.0),p)*50.0+iTime)*0.01;
    color+=sin(acos(distance(vec2(0.0,0.0)/2.,p)*2.))*10.0;
return color*0.1;}

// Function 1182
float materialHeightMap( vec2 grooves, vec2 coord ) {
	return min( grooveHeight( grooves.x, 0.01, coord.x ), grooveHeight( grooves.y, 0.01, coord.y ));
}

// Function 1183
vec1 suv(vec4 a){return a.x+a.y+a.z+a.w;}

// Function 1184
mat cmap(in ray r, in hit h) {
    vec3 f = tex(iChannel0, r.d).rgb*1.5;
    return mat(
    	vec3(0.),
        f, //gamma correct
    	vec2(0.), //not applicable
        vec2(0.), //not implemented
        -1., 0., _cmap.d);
}

// Function 1185
float map( const in vec3 p ) {
    float d = -sdBox( p, vec3( 28., 14., 63. ) );

    vec3 pm = vec3( abs( p.x ) - 17.8, p.y, mod( p.z, 12.6 ) - 6.);    
    vec3 pm2 = abs(p) - vec3( 14., 25.25, 0. );
    vec3 pm3 = abs(p) - vec3( 6.8, 0., 56.4 );      

    d = opU( d, sdColumn( pm, vec2( 1.8, 1.8 ) ) );        
    d = opS( d, sdBox( p,  vec3( 2.5, 9.5, 74. ) ) );    
    d = opS( d, sdBox( p,  vec3( 5., 18., 73. ) ) );
    d = opS( d, sdBox( p,  vec3( 13.8, 14.88, 63. ) ) );
    d = opS( d, sdBox( p,  vec3( 13.2, 25., 63. ) ) );
    d = opS( d, sdColumn( p,  vec2( 9.5, 63. ) ) ); 
    d = opU( d, sdColumn( pm3, vec2( 1.8, 1.8 ) ) );
    d = opU( d, sdBox( pm2, vec3( 5., .45, 200. ) ) );
    
    return d;
}

// Function 1186
vec2 map(float depth, vec3 p)
{
    vec2 roughObj =       vec2(roundbox_df(rotate_xaxis(p - vec3(0., -.4,  1.2), cos( 0.0), sin( 0.0)), vec3(5., .02, 1.), .01), ROUGH_ID);
    vec2 semiroughObj =   vec2(roundbox_df(rotate_xaxis(p - vec3(0., -.38,  0.), cos(-0.12), sin(-0.12)) , vec3(5., .02, 1.), .01), SEMI_ROUGH_ID);
    vec2 semimirrorObj =  vec2(roundbox_df(rotate_xaxis(p - vec3(0., -.2, -1.2), cos(-0.26), sin(-0.26)) , vec3(5., .02, 1.), .01), SEMI_MIRROR_ID);
    vec2 mirrorObj =      vec2(roundbox_df(rotate_xaxis(p - vec3(0., .2, -2.4), cos(-0.5), sin(-0.5)) , vec3(5., .02, 1.), .01), MIRROR_ID);    
    
    
    vec2 resultObj = union_obj(mirrorObj, semimirrorObj);
    resultObj = union_obj(resultObj, semiroughObj);
    resultObj = union_obj(resultObj, roughObj);    
    
    float backdropDF = roundbox_df(p + vec3(0., 1., 0.), vec3(20.,.2,10.), 0.);
    backdropDF = min(backdropDF, roundbox_df(rotate_xaxis(p - vec3(0., -3.8, -3.2), cos(.5), sin(.5)), vec3(20., 10., .2), 0.));
    vec2 backdropObjs = vec2(backdropDF, BACKDROP_ID);    
    resultObj = union_obj(resultObj, backdropObjs);
    
    if (depth < .5)
    {
        vec4 l = get_light(0);
        float lightDF = sphere_df(p - l.xyz, l.w);
        l = get_light(1);
        lightDF = min(lightDF, sphere_df(p - l.xyz, l.w));
        l = get_light(2);
        lightDF = min(lightDF, sphere_df(p - l.xyz, l.w));
        l = get_light(3);
        lightDF = min(lightDF, sphere_df(p - l.xyz, l.w));
        
        vec2 lightObjs =       vec2(lightDF, LIGHT_ID);
        resultObj = union_obj(resultObj, lightObjs);
    }
    
    resultObj = union_obj(resultObj, vec2( envsphere_df(p, 11.), ENVIRONMENT_ID) );
        
    return resultObj;
}

// Function 1187
vec4 map(vec3 p) {
    vec4 res = vec4(1e10, 0.0, 0.0, 0.0);
    {
    res = opU(res, vec4(sdSphere(p, 1.0),vec3(0.0)));
    }
    return res;
}

// Function 1188
vec3 InvTonemap(const TonemapParams tc, vec3 y)
{
	vec3 inv_toe = - tc.mToe.x / (y - tc.mToe.z) - tc.mToe.y;
	vec3 inv_mid = (y - tc.mMid.y) / tc.mMid.x;
	vec3 inv_shoulder = - tc.mShoulder.x / (y - tc.mShoulder.z) - tc.mShoulder.y;

	vec3 result = mix(inv_toe, inv_mid, step(tc.mBy.x, y));
	result = mix(result, inv_shoulder, step(tc.mBy.y, y));
	return result;
}

// Function 1189
float map(vec3 p)
{
	float d = length(p.xy) - 1.0;
    
    return -d;
}

// Function 1190
vec2 mapD2(float t)
{
    return 14.0*a*c*sin(t+m)*sin(7.0*t+n) - a*cos(t+m)*(b+c*cos(7.0*t+n)) - 49.0*a*c*cos(t+m)*cos(7.0*t+n);
}

// Function 1191
float map( in vec3 p )
{
	float d = -box(p-vec3(0.,10.,0.),vec3(10.));
	d = min(d, box(rotate(vec3(0.,1.,0.), 1.)*(p-vec3(4.,5.,6.)), vec3(3.,5.,3.)) );
	d = min(d, box(rotate(vec3(0.,1.,0.),-1.)*(p-vec3(-4.,2.,0.)), vec3(2.)) );
	d = max(d, -p.z-9.);
	
	return d;
}

// Function 1192
vec2 LUT_UV(float red, float green, float blue_slice)
{
    float row;
    float col = modf(min(blue_slice, LUT_SIZE - 1.0) / LUT_ROWS, row); 
    return vec2
    (
        (col           ) + ((  red * (1.0 - 2.0 * LUT_PADDING) + LUT_PADDING) / LUT_ROWS),
        (row / LUT_ROWS) + ((green * (1.0 - 2.0 * LUT_PADDING) + LUT_PADDING) / LUT_ROWS)
    );
}

// Function 1193
vec3 GetReadMipMapUVW_Dir2(vec3 _uvw,vec3 _axis,float ID){
    ivec3 xyz = ID_xyz[int(dot(abs(_axis.yz),vec2(1,2)))];
	vec3 uv = vec3(_uvw[xyz.x],_uvw[xyz.y],_uvw[xyz.z]);
    float group_ID = floor((ID-=1.)/3.);
    float scale = exp2(group_ID-10.);
    float Bound = 1.-exp2(-group_ID);
    uv.xy = scale*(clamp(uv.xy,-Bound,Bound)+SCACLE_COEF[int(mod(ID,3.))])-1.;
    //f(x) = inverse(inverse(f(x)))
    uv = vec3(uv[xyz.x],uv[xyz.y],uv[xyz.z]);
    return normalize(uv + _axis);
}

// Function 1194
vec2 mapMat(vec3 p){
    vec2 cubeMap = vec2(-length(p) + 4.0, 1.0);
    vec2 mirrorSphere = vec2(length(p - vec3(cos(iTime / 2.0), 0.0, sin(iTime / 2.0))) - 0.25, 2.0);
    return vecMin(cubeMap, mirrorSphere);
}

// Function 1195
vec4 mapPortalColor(vec3 p, vec3 portalPos, float rotY, vec4 cristalcolor, vec4 fxcolor)
{
    vec2 q = rotateY(p-portalPos, rotY).xy; q.y *= 0.55;
    float d = length(q) - 1.4 + sin(q.x*10.+t*2.)*cos(q.y*10.+t*2.) * 0.05;
    return mix(cristalcolor, fxcolor, smoothstep(-0.5, 0.2, d));
}

// Function 1196
vec2 IdxtoUV(float idx, float stride)
{
    idx = floor(idx);
    stride = floor(stride);
    return vec2(mod(idx, stride), floor(idx / stride));
}

// Function 1197
vec2 map(vec2 z)
{
  vec2 m; 
  int  i = methodId;
    
  if (!vizSquare) {  
    if     (i == 0) { m = map0sd(z); }
    else if(i == 1) { m = map1sd(z); }
    else if(i == 2) { m = map2sd(z); }
    else if(i == 3) { m = map3sd(z); }
  } else {
    if     (i == 0) { m = map0ds(z); }
    else if(i == 1) { m = map1ds(z); }
    else if(i == 2) { m = map2ds(z); }
    else if(i == 3) { m = map3ds(z); }
  }
  return m;
}

// Function 1198
vec2 map(in vec3 pos) {
  // vec2 res = vec2(.3 * sdTorus(opTwist(pos - vec3(0., 0., 0.)),
  //                              vec2(0.90 * (sin(iTime) + 2.8) / 2., 0.2)),
  // vec2 mouse = iMouse.xy / iResolution.y;
  vec2 torus = sdTorus(opTwist(pos - vec3(0., 0., 0.)),
                       vec2(0.90 * (1. + 2.8) / 2., 0.2));
  vec2 res = vec2(.3 * torus.x, torus.y);
  // clamp((pos.y+1.)/3., 0., 1.));

  return res;
}

// Function 1199
vec3 map(vec3 p) {
	vec3 q = p;
	q.y -= PalmTrunkLength;
	vec2 res = sdTrunk(q);

	res = min2(res, sdCoconuts(q));

	res = smin(res, sdIsland(p), 0.3);
	res = min2(res, sdWater(p));

    return min3(sdLeaves(q), vec3(res, 0.0));
}

// Function 1200
float map_tube(vec3 pos)
{
    pos-= flamePos;
    pos.x+= tubeLenght;
    pos = pos.yxz;
    
    float df = sdCylinder(pos, vec2(tubeDiameter, tubeLenght));
    df+= -0.006 + 0.003*noise(pos*vec3(70., 20., 70.));
    df = max(df, -sdCylinder(pos, vec2(tubeDiameter - 0.02, tubeLenght + 0.02)));
    return df;
}

// Function 1201
vec2 mapDetailed(vec3 p) {
    vec2 d = vec2(-1.0, -1.0);
    d = vec2(mapTerrain(p-vec3(0.0, FLOOR_LEVEL, 0.0), FLOOR_TEXTURE_AMP), TYPE_FLOOR);
    return d;
}

// Function 1202
float MapTerrain( vec3 p)
{   
  float terrainHeight = GetTerrainHeight(p);   
  terrainHeight= mix(terrainHeight+GetStoneHeight(p, terrainHeight), terrainHeight, smoothstep(0., 1.5, terrainHeight));
  terrainHeight= mix(terrainHeight+(textureLod(iChannel1, (p.xz+planePos.xz)*0.0015, 0.).x*max(0., -0.3+(.5*terrainHeight))), terrainHeight, smoothstep(1.2, 12.5, terrainHeight));

  terrainHeight= mix(terrainHeight-0.30, terrainHeight, smoothstep(-0.5, 0.25, terrainHeight));
  float water=0.;
  if (terrainHeight<=0.)
  {   
    water = (-0.5+(0.5*(noise2D((p.xz+planePos.xz+ vec2(-iTime*0.4, iTime*0.25))*2.60, WATER_LOD))));
    water*=(-0.5+(0.5*(noise2D((p.xz+planePos.xz+ vec2(iTime*.3, -iTime*0.25))*2.90), WATER_LOD)));
  }
  return   p.y -  max((terrainHeight+GetTreeHeight(p, terrainHeight)), -water*0.04);
}

// Function 1203
vec2 fixUV (vec2 uv, int i)
{
#ifdef NYAN
	if (i == 1)
	{
		float run = mod(iTime,16.0) < 9.0 ? 1.0 : 0.0;
		uv.x = clamp(uv.x, 0.0, 0.9) / 6.0;
#ifdef ANIMATE_NYAN
		uv.x += run * floor(mod(iTime*6.0,6.0))*40.0/iChannelResolution[SAMPLERI].x;
#endif
	}
#endif
	return uv;
}

// Function 1204
float remap(float v,float min1,float max1,float min2,float max2
){return min2+(max2-min2)*(v-min1)/(max1-min1);}

// Function 1205
vec3 doBumpMap( sampler2D tx, in vec3 p, in vec3 n, float bf){
   
    const vec2 e = vec2(EPS, 0);
    
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    
    vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
	
}

// Function 1206
float map_handrail(vec3 pos)
{
    float h = (0.5*floor_height)/(staircase_length - floor_width - interfloor_width)*(nb_stairs - 1.)/nb_stairs - 0.01;

    vec3 pos2 = pos;
    pos2.x-= 0.035;
    
    if (pos.z>0.)
    {
        pos2.y = mod(pos.y - floor_height/2., floor_height) + floor_height/2.;
    
        pos2.xz = -pos2.xz;
        pos2.y-= floor_height/2.;
    }
    else
        pos2.y = mod(pos.y, floor_height);
    
    pos2.y-= h*(pos2.x + staircase_width/2. - floor_width) + handrail_height + (pos.z>0.?0.8:0.3);    
    
    pos2.x = mod(pos2.x + handrail_spacing/2., handrail_spacing) - handrail_spacing/2.;
    pos2+= vec3(0., 0., staircase_width/2. - stairs_width + handrail_offset);

    float handrail = sdCylinder(pos2, vec2(handrail_thickness1, handrail_height/2.), 0., 0.);
    
    handrail = min(handrail, sdRoundBox(pos2 + vec3(0., -handrail_height/2., 0.), vec3(staircase_length/2., handrail_thickness2.x, handrail_thickness2.y), handrail_roundness)); 
    
    // Trick to avoid that the further handrail gets "holes"
    handrail = min(handrail, 0.07 + abs(pos.z));    
    
    handrail = max(handrail, pos.x - staircase_length/2. + interfloor_width);
    handrail = max(handrail, -pos.x - staircase_length/2. + floor_width);
    
    return handrail;
}

// Function 1207
vec2 scuv(vec2 uv) {
    float zoom=1.;
    #ifdef SHADEROO
    zoom=1.-iMouseData.z/1000.;
    #endif
    return (uv-.5)*1.2*zoom+.5; 
}

// Function 1208
vec2 map(in vec3 pos)
{
    float t = 9999.0;
    t = min(t, dsphere(pos, vec3(0.0), 1.0));
    return vec2(t, 0.0);
}

// Function 1209
void mainCubemap( out vec4 O, in vec2 I, in vec3 rayOri, in vec3 rayDir )
{
    
    ivec3 XYFace = RayDirToXYFace(rayDir);
    ivec2 XYTall = ivec2(XYFace.x, XYFace.y + 1024*XYFace.z);
    if(XYTall.y < 16){
        int seed = XYFace.x + XYFace.y*16 + iFrame*1024*16;
        seed = IHash(seed);
        O.x = Hash(seed);
    } else  {
        int stage = XYTall.y/16;
        int sortStage = stage-1;
        int ID = XYTall.x*16 + (XYTall.y%16);
        if(stage<106){
            int partner = getPartner(ID,sortStage);
            vec4 A = sampleIDStage(ID, stage-1);
            vec4 B = sampleIDStage(partner, stage-1);
            if(ID > partner){
                if(A.x>B.x){
                    O=A;
                } else {
                    O=B;
                }
            } else {
                if(A.x>B.x){
                    O=B;
                } else {
                    O=A;
                }
            }
        } else if(stage<110){
            //Keep shifting the result down so that JFA pipeline always can find the particles
            O = sampleIDStage(ID, stage-1);
        } else {
            //Detect glitches
            O = vec4(0);
            vec4 A = sampleIDStage(ID-1, stage-1);
            vec4 B = sampleIDStage(ID, stage-1);
            vec4 C = sampleIDStage(ID+1, stage-1);
            if(A.x>B.x || B.x>C.x){
                O += 1.;
            }
            
        }
        
    }
    
    
}

// Function 1210
vec4 map(in vec2 p) {
    return doMap(voronoi(p*2.0));
}

// Function 1211
vec3 map(vec3 p) {	
	
		p.y -= 900.;
        float uAngle = iTime * uSpeed;

		p *= rotY(uAngle);
		
		vec3 s = vec3(100000.0);
		s.x = min(min(min(min(min(min(min(min(s.x, head(p)), helmet(p)), body(p)), arms(p)), hands(p)), hips(p)), legs(p)), shoes(p));
		return s;	
	}

// Function 1212
Object mapG(vec3 p){
    Object o = NewObject;

	o.d = 10e6;

    o.material = 2.;
    float octa = 10e7;
    float sc = 1.;
    float sep = 1.3;
    
    vec3 q = p;
    float rhomb= coolCahedron(q,S*sc);
    
    
    float dBalls = 10e7;
    
    float dXtal = 10e7;
    
    float reps = 4.;
    pmodpol(p.xy,reps);
    pmodpol(p.yz,reps);
    pmodpol(p.xz,reps);
    p *= 0.65;
    
    for(float i = 0.; i < 4. + min(float(iFrame),0.); i++){
        
        sc = pow(0.54, i+1.);
        if(mod(i+2.,2.) > 0.){
            float db = length(p)-0.4*sc;
            if(db< dBalls){
            	dBalls = db;
                materials[3].albedo = 0.5 + sin(vec3(1.,0.5,0.1)+i+2.)/2.;
                materials[3].albedo = max(materials[3].albedo,0.);
                
            }
        }
        
        float sepi = sep*sc;
        p=abs(p);
        p.x -= sepi;
        vec3 v = p;
    	octa= min(octa,coolCahedron(p,S*sc));
        
        if(mod(i+3.,2.) < 1.){
            float db = sdOctahedron(p,S*sc*0.9);
            if(db< dXtal){
            	dXtal = db;
                materials[4].albedo = 0.5 + sin(vec3(1.,0.5,0.1)+i+2.)/2.;
            }
        }
        
        p.x += sepi;
        p.y -= sepi;
        
        vec3 b = p;
    	octa= min(octa,coolCahedron(p,S*sc));
        
        if(mod(i+3.,2.) < 1.){
            float db = sdOctahedron(p,S*sc*0.9);
            if(db< dXtal){
            	dXtal = db;
            }
        }
        
        p.y += sepi;
        p.z -= sepi;
    	octa= min(octa,coolCahedron(p,S*sc));

        if(mod(i+3.,2.) < 1.){
            float db = sdOctahedron(p,S*sc*0.9);
            vec3 q = abs(p);
            if(i<3.){
                q.xz *= rot(0.25*pi);
                q.xy *= rot(0.25*pi);
                vec3 bSz = vec3(S*sc*0.9);
                bSz.x *= 0.05;
                bSz.y *= 100.;
                bSz.z *= 0.05;
                db = min(db,sdBox(q,bSz));
            
            }
            if(db< dXtal){
                dXtal = db;
            }
        }	
    }
    

    
    o.d = min(o.d,octa);
    
    o = omin(o,dBalls, 3.);
    
    return o;
}

// Function 1213
vec3 heatMap(float greyValue) {   
	vec3 heat;      
    heat.r = smoothstep(0.5, 0.8, greyValue);
    if(greyValue >= 0.90) {
    	heat.r *= (1.1 - greyValue) * 5.0;
    }
	if(greyValue > 0.7) {
		heat.g = smoothstep(1.0, 0.7, greyValue);
	} else {
		heat.g = smoothstep(0.0, 0.7, greyValue);
    }    
	heat.b = smoothstep(1.0, 0.0, greyValue);          
    if(greyValue <= 0.3) {
    	heat.b *= greyValue / 0.3;     
    }
	return heat;
}

// Function 1214
float map_floors(vec3 pos)
{
   float posy2 = mod(pos.y + floor_height*0.5, floor_height) - floor_height*0.5;
   float posy3 = mod(pos.y, floor_height);

   float floor = posy2;
   floor = max(floor, -posy2 - floor_thickness);
   floor = max(floor, pos.x + staircase_length/2. - floor_width);
   floor = max(floor, -pos.x - staircase_length/2. - apartment_width);
   floor = max(floor, abs(pos.z) - staircase_width/2. -apartment_width);
   
   #ifdef blood
   float bloodv = getBlood(pos);
   bloodv = 15.*max(bloodv - 0.0075, 0.) + pow(smoothstep(0.008, 0.058, bloodv), 0.4);
   floor-= 0.01*bloodv;
   #endif
   
   float interfloor = posy3 - floor_height*0.5;
   interfloor = max(interfloor, -posy3 + floor_height*0.5 - interfloor_thickness);
   interfloor = max(interfloor, -pos.x + staircase_length/2. - interfloor_width);
   interfloor = max(interfloor, pos.x - staircase_length/2. - 0.1);
   interfloor = max(interfloor, abs(pos.z) - staircase_width/2. -0.1);
   
   return min(floor, interfloor);
}

// Function 1215
vec4 hpluvToLch(vec4 c) {return vec4( hpluvToLch( vec3(c.x,c.y,c.z) ), c.a);}

// Function 1216
float map(vec3 q){

    // Debug usage to compare rigid moving objects with
    // objects that flow with the Truchet tubing.
    #define RIGID_OBJECTS

    // Scaling factor.
    const float sc = 2.;
    
    // Moving object time; A bit redundant here, but helpful when 
    // you want to change the speed without having to refactor everywhere.
    float tm = iTime;
  

    // Back wall
    float wall = -q.z + .1; // Thick wall: (abs(p.z - .2) - .2) + .1;


    // Local hexagonal cell coordinate and cell ID.
    vec4 h = getGrid(q.xy*sc);
    
    // Using the idetifying coordinate - stored in "h.zw," to produce a unique random number
    // for the hexagonal grid cell.
    float rnd = hash21(h.zw + vec2(.11, .31));
    //rnd = fract(rnd + floor(iTime/3.)/10.); // Periodically changing the random number.
    float rnd2 = hash21(h.zw + vec2(.37, 7.83)); // Another random number.
   
    
    // It's possible to control the randomness to form some kind of repeat pattern.
    //rnd = mod(h.z + h.w, 2.)/2.;
    
    
    // Storing the local hexagon cell coordinates in "p". This serves no other
    // purpose than to not have to write "h.xy" everywhere. :)
    vec2 p = h.xy;
    

    // Using the local coordinates to render three arcs, and the cell ID
    // to randomly rotate the local coordinates by factors of PI/3.
    rnd = floor(rnd*144.);
    
    // Random rotation and flow direction..
    float dir = mod(rnd, 2.)*2. - 1.;
    float ang = rnd*3.14159/3.;

    p = rot2(ang)*p; // Random rotate.
    
    
    // Arc radii and thickness variables.
    const float rSm = s.y/6.; // .5/1.732 -> 1.732/2./3.
    const float th = .1; // Arc thickness.

    // The three segment (arc) distances.
    vec3 d;
    
   
    // Metal.
    float mtl = 1e5;
 
    #ifndef RIGID_OBJECTS
    // Angle for non rigid objects.
    float a3;
    #endif
    
    // The Truchet distance.
    float tr = 1e5;
    
    // A scaling constant.
    const float aSc = 1.;
    
    // Is the piece and arc or not. This is an orientation hack that I'll
    // fix later.
    float isArc = 1.;
    
    // Z-based value and a redundant height value that gets used in
    // another example.
    vec3 qZ3, hgt = vec3(0);
    
    // Rotation and minimum coordinate.
    vec2 qR, minP;
    
    if(rnd2<.5){
    
        // Relative local coordinate centers of the two arc and line.
        vec2 p0 = p - vec2(0, -s.y/3.);
        vec2 p1 = p - vec2(0, s.y/3.);
        vec2 p2 = p;
        // Distances.
        d.x = length(p0) - rSm;
        d.y = length(p1) - rSm;
        d.z = abs(p2.y);
        
        d = abs(d)/sc; // Turning the circles into arc segments and scaling.

        // Move the Z-position out to the correct position for all three tubes. 
        // There's a redundant relative height value there for crossover tubes.
        qZ3 = q.z + .045 + hgt;

        // A rounded or square Truchet tube. Look up the torus formula, if you're
        // not sure about this. However, essentially, you place the rounded curve
        // bit in one vector position and the Z depth in the other, etc. Trust me,
        // it's not hard. :)

        // Technically, I could get away with using the minimum 2D arc length and 
        // calculate just one of these, but I'll be extending to include crossover
        // arcs, so I'll leave it in this form.
        d.x = length(vec2(d.x, qZ3.x)) - .05;
        d.y = length(vec2(d.y, qZ3.y)) - .05;
        d.z = length(vec2(d.z, qZ3.z)) - .05;
    /*    
        d.x = sBoxS(vec2(d.x, qZ3.x), vec2(.05, .05), .025);
        d.y = sBoxS(vec2(d.y, qZ3.y), vec2(.05, .05), .025);
        d.z = sBoxS(vec2(d.z, qZ3.z), vec2(.05, .05), .025);
    */    

        
        
        // Arc segment angle calculation.
        if(min(d.x, d.y)<d.z){
            
            // Minimum 
            minP = p1;
            
            // Reverse the direction of the first arc.
            if(d.x<d.y) {
               minP = p0; 
               dir *= -1.;
            }
            
            #ifdef RIGID_OBJECTS
            minP *= rot2(dir*tm); // Animation occurs here.
            float a = atan(minP.y, minP.x); // Polar angle.
            a = (floor(a/6.2831853*6.) + .5)/6.; // Repeat central angular cell position.
            // Polar coordinate.
            qR = rot2(a*6.2831853)*minP; 
            qR.x -= rSm; 
            #else
            a3 = atan(minP.x, minP.y);
            a3 = (a3*(6./6.2831)*aSc - tm*dir);
            #endif
            
        }
        else {
            
            // I guessed a time dialation figure of 3.14159 based on the relative 
            // length of a full circle tube (broken into thirds) and a straight
            // tube (broken into thirds). Pure fluke, but I'll take it. :)
            // Circle tube: length = diameter*PI;
            // Straight tube:  length = diameter;
            // Basically, the objects in the tube will travel just a few percentage
            // points slower than those in the arcs in order to meet up perfectly, 
            // but you'll never notice.
            minP = p2;
            #ifdef RIGID_OBJECTS
            qR = p2;
            qR.x = mod(qR.x - dir*tm/3.14159, 1./3.) - 1./6.;
            isArc = 0.; // Not an arc piece.
            #else
            a3 = minP.x;
            a3 = (a3*(3.)*aSc - tm*dir - aSc*.5);
            #endif
            
        }

    }
    else {
    
        vec2 p0 = p - vec2(-.5, -.5/s.y);
        vec2 p1 = p - vec2(.5, -.5/s.y);
        vec2 p2 = p - vec2(0, s.y/3.);
        d.x = length(p0) - rSm;
        d.y = length(p1) - rSm;
        d.z = length(p2) - rSm;
        
        d = abs(d)/sc; // Turning the circles into arc segments and scaling.

        // Move the Z-position out to the correct position for all three tubes.
        qZ3 = q.z + .045 + hgt;

        // A rounded or square Truchet tube.
        d.x = length(vec2(d.x, qZ3.x)) - .05;
        d.y = length(vec2(d.y, qZ3.y)) - .05;
        d.z = length(vec2(d.z, qZ3.z)) - .05;
    /*    
        d.x = sBoxS(vec2(d.x, qZ3.x), vec2(.05, .05), .025);
        d.y = sBoxS(vec2(d.y, qZ3.y), vec2(.05, .05), .025);
        d.z = sBoxS(vec2(d.z, qZ3.z), vec2(.05, .05), .025);
    */    
        
        // Since the moving objects reside within the tubes, the minimum 3D arc 
        // distance should provide the minimum coordinate upon which to calculate 
        // the angle of the object flowing through it... It will work with this 
        // example, but sometimes, you'll have to calculate all three.
        minP = d.x<d.y && d.x<d.z? p0 : d.y<d.z? p1 : p2;
        
        ///// 
        #ifdef RIGID_OBJECTS
        
        minP *= rot2(dir*tm); // Animation occurs here.
        float a = atan(minP.y, minP.x); // Polar angle.
        a = (floor(a/6.2831853*6.) + .5)/6.; // Repeat central angular cell position.
        // Polar coordinate.
        qR = rot2(a*6.2831853)*minP; 
        qR.x -= rSm; 
        
        #else
      
        // Calculating, scaling and moving the angles.
        a3 = atan(minP.x, minP.y);
        a3 = (a3*(6./6.2831)*aSc - tm*dir);
        
        #endif
        ///// 
    
    }
    
    // The Truchet tube distance is the minimum of all. I could save a couple
    // of "min" calls and set this above, but this will do.
    tr = min(min(d.x, d.y), d.z);
 

    ///// 
    #ifdef RIGID_OBJECTS
    
    // 3D ball position. "qR" is based on "p," which has been scalle
    // by the factor "sc," so needs to be scaled back. "q.z" has not been
    // scaled... Yeah, it can be confusing. :)
    vec3 bq = vec3(qR/2.,  qZ3.x); // All heights are equal, in this example.
    //if(isArc==0.) bq = bq.yxz;
    //float obj =  max(length(bq.zx) - .02, abs(bq.y) - .04); // Cylinder.
    float obj = length(bq) - .02; // Ball.
    // obj = min(tr + .035 + .01, ball); // Adding in the railing.
    
    #else
   
    a3 = abs(fract(a3) - .5) - .25;
    a3 /= (6.*aSc/sc);
    float obj = max(tr + .0325, a3);
    
    #endif
    ///// 
    
    
    // Metallic elements, which includes the joins, metal ball joints
    // and the tracks they're propogating along.
    //
    // Joins.
    vec2 rp = p;
    rp *= rot2(-3.14159/6.); // Animation occurs here.
    float a = atan(rp.y, rp.x); // Polar angle.
    a = (floor(a/6.2831853*6.) + .5)/6.; // Repeat central angular cell position.
    // Polar coordinate.
    rp = rot2(a*6.2831853)*rp; 
    rp.x -= .5; // Moving the element along the radial line to the edge.

    // Construct the joiner rings.
    rp = abs(rp);
    mtl = rp.x - .02;//max(rp.x, rp.y) - .025;
    mtl = max(max(mtl, tr - .015), -(tr - .005));
    
    // Tracks.
    mtl = min(mtl, tr + .045);
    

    
    
    // Hollowing out the Truchet tubing. If you don't do this, it can cause
    // refraction issues, but I wanted the tubes to be hollow anyway. I've 
    // made the walls kind of thick. Obviously, the thickness can effect
    // the way light bounces around, and ultimately the look.
    tr = max(tr, -(tr + .02)); 
    
    
   
    // Debug: Take out the glass tubing, brackets, tracks, etc, to see the inner
    // objects unobstructed.
    //tr += 1e5;
    //mtl += 1e5;
    
   
    // Storing the object ID.
    vObjID = vec4(wall, tr, mtl, obj);
    
    // Returning the closest object.
    return min(min(wall, tr), min(mtl, obj));



}

// Function 1217
vec2 getUVs (vec2 fragCoord)
{
	vec2 coord = vec2(fragCoord.x, iResolution.y - fragCoord.y);
	vec2 hres = vec2(
#ifdef SPLITSCREEN
		floor(iResolution.x / 2.0),
#else
		iResolution.x,
#endif
		iResolution.y);

	coord.x = mod(coord.x, hres.x);
	coord /= hres.xy;
#ifndef UNITBASED
	float aspect = hres.x / hres.y;
	coord.x = coord.x * aspect - (aspect-1.0)/2.0;
#endif
	return coord;
}

// Function 1218
vec3 mapBiome(vec3 x )
{
    x = planetRotation * x;
    
    vec3 p = floor( x );
    vec3 f = fract( x );

	float id = 0.0;
    
    // distance to closest and second closest
    vec2 res = vec2( 100.0 );
    // biome ID for closest and second closest
    vec2 resId = vec2(-1., -1.);
    
    for( int k=-1; k<=1; k++ )
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ )
    {
        vec3 b = vec3(float(i), float(j), float(k));
        vec3 r = vec3( b ) - f + hash( p + b );
        float d = length(r);
        id = mod(abs(dot( p+b, vec3(1.0,57.0,113.0 ))), 3.);

        if( d < res.x )
        {
            res = vec2( d, res.x );
            resId = vec2( id, resId.x );
        }
        else if( d < res.y )
        {
            res.y = d;
            resId.y = id;
        }
    }
    
    float diff = res.y - res.x;
    
    // this is a giant hack. need a better way to blend between the voronoi regions.
    float ratio1 = min(1., pow(smoothstep(1., 3., clamp(res.y / res.x, 1., 3.)), .35) + .5);
    float ratio2 = 1. - ratio1;
        
    return vec3(resId.x == 0. ? ratio1 : resId.y == 0. ? ratio2 : 0.,
                resId.x == 1. ? ratio1 : resId.y == 1. ? ratio2 : 0.,
               	resId.x == 2. ? ratio1 : resId.y == 2. ? ratio2 : 0.);
}

// Function 1219
vec2 toUV( in vec3 pos )
{
	return pos.xz;
}

// Function 1220
vec2 Cam_GetViewCoordFromUV( vec2 vUV, float fAspectRatio )
{
	vec2 vWindow = vUV * 2.0 - 1.0;
	vWindow.x *= fAspectRatio;

	return vWindow;	
}

// Function 1221
float seaFragmentMap(vec3 p) 
{
    vec2 uv = p.xz * vec2(0.85, 1.0); 
    
    float freq 	 = SEA_FREQ;
    float amp    = SEA_HEIGHT;  
    float choppy = SEA_CHOPPY;
	
    float d = 0.0;
    float h = 0.0;    
    for(int i = 0; i < SEA_FRAGMENT_ITERATIONS; ++i) 
    {	    
    	d =  seaOctave((uv + gSeaCurrentTime) * freq, choppy);
		d += seaOctave((uv - gSeaCurrentTime) * freq, choppy); 
		h += d * amp;
	
		freq *= SEA_FREQ_MUL; 
		amp  *= SEA_AMPLITUDE_MUL;
	
		choppy = mix(choppy, SEA_CHOPPY_MIX_VALUE, SEA_CHOPPY_MIX_FACTOR);
	
		uv *= OCTAVE_MATRIX;
    }
    return p.y - h;
}

// Function 1222
vec3 TriMap(vec3 unitCoord,
            vec3 normal)
{
    #ifdef DISTORT
        vec3 t = texture(iChannel1, abs(unitCoord.xy)).rgb;
		vec2 uvX = mix(unitCoord.yz, unitCoord.zy, t.x);
        vec2 uvY = mix(unitCoord.xz, unitCoord.zx, t.y);
        vec2 uvZ = mix(unitCoord.xy, unitCoord.yx, t.z);
    	mat3x3 triKrn = mat3x3(texture(iChannel0, refl(uvX, AUTO_TILE)).rgb,
                           	   texture(iChannel0, refl(uvY, AUTO_TILE)).rgb,
                           	   texture(iChannel0, refl(uvY, AUTO_TILE)).rgb);
    #else
        mat3x3 triKrn = mat3x3(texture(iChannel0, refl(unitCoord.yz, AUTO_TILE)).rgb,
                           	   texture(iChannel0, refl(unitCoord.xz, AUTO_TILE)).rgb,
                           	   texture(iChannel0, refl(unitCoord.xy, AUTO_TILE)).rgb);
    #endif    
    return (triKrn * abs(normal));
}

// Function 1223
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    // Larger sample distances give a less defined bump, but can sometimes lessen the aliasing.
    const vec2 e = vec2(.001, 0); 
    
    // Gradient vector: vec3(df/dx, df/dy, df/dz);
    float ref = bumpSurf3D(p, nor);
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy, nor),
                      bumpSurf3D(p - e.yxy, nor),
                      bumpSurf3D(p - e.yyx, nor)) - ref)/e.x; 
    
    /*
    // Six tap version, for comparisson. No discernible visual difference, in a lot of cases.
    vec3 grad = vec3(bumpSurf3D(p - e.xyy) - bumpSurf3D(p + e.xyy),
                     bumpSurf3D(p - e.yxy) - bumpSurf3D(p + e.yxy),
                     bumpSurf3D(p - e.yyx) - bumpSurf3D(p + e.yyx))/e.x*.5;
    */
       
    // Adjusting the tangent vector so that it's perpendicular to the normal. It's some kind 
    // of orthogonal space fix using the Gram-Schmidt process, or something to that effect.
    grad -= nor*dot(nor, grad);          
         
    // Applying the gradient vector to the normal. Larger bump factors make things more bumpy.
    return normalize(nor + grad*bumpfactor);
	
}

// Function 1224
vec4   luvToXyz(vec4 c) {return vec4(   luvToXyz( vec3(c.x,c.y,c.z) ), c.a);}

// Function 1225
vec4 rgbToHpluv_(vec4 c,float gamma)
{
    return vec4(
        rgbToHpluv(
            pow(c.rgb,vec3(gamma))
        )/vec3(360.,100.,100.)
    ,c.a);
}

// Function 1226
vec3 doBumpMap( sampler2D tex, in vec3 p, in vec3 nor, float bumpfactor){
   
    const float eps = 0.001;
    float ref = getGrey(tex3D(tex,  p , nor));                 
    vec3 grad = vec3( getGrey(tex3D(tex, vec3(p.x-eps, p.y, p.z), nor))-ref,
                      getGrey(tex3D(tex, vec3(p.x, p.y-eps, p.z), nor))-ref,
                      getGrey(tex3D(tex, vec3(p.x, p.y, p.z-eps), nor))-ref )/eps;
             
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
	
}

// Function 1227
float map(in float value, in float istart, in float istop, in float ostart, in float ostop) {
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
}

// Function 1228
float rmap(vec4 uv, RSet4 rs) {
    return RAND(map(uv, rs.q, rs.l), rs.r);
}

// Function 1229
vec3 rgb_yuv (vec3 rgb) {
    return rgb_yuv(rgb, vec2(0.0722, 0.2126), vec2(0.436, 0.615));
}

// Function 1230
vec4 map(vec3 p)
{
   	float scale = 3.;
    float dist = 0.;
    
    float x = 6.;
    float z = 6.;
    
    vec4 disp = displacement(p);
        
    float y = 1. - smoothstep(0., 1., disp.x) * scale;
    
    #ifdef USE_SPHERE_OR_BOX
        dist = osphere(p, +5.-y);
    #else    
        if ( p.y > 0. ) dist = obox(p, vec3(x,1.-y,z));
        else dist = obox(p, vec3(x,1.,z));
	#endif
    
    return vec4(dist, disp.yzw);
}

// Function 1231
vec4 BufferCubemap(in sampler2D buffer, in float bufferAspect, in vec3 d) 
{
    vec3 t = 1.0 / min(-d, d);
    vec3 p = d*(-0.5 * max(max(t.x, t.y), t.z));
    
    vec3 n = -sign(d) * step(t.yzx, t.xyz) * step(t.zxy, t.xyz);
    
    vec2 px = vec2(p.z*n.x, p.y) * abs(n.x);
    vec2 py = vec2(-p.x*n.y, -p.z) * abs(n.y);
    vec2 pz = vec2(-p.x*n.z, p.y) * abs(n.z);

    float tx = (step(0.5, n.z)*2.0 + abs(n.x) 
                 + step(0.5, n.x)*2.0) * (1.0 - abs(n.y))
        			+ step(0.5, n.y)*3.0 * abs(n.y);
    
	float ty = (1.0 - (2.0 - 4.0/bufferAspect)) * abs(n.y);
    
    vec2 uv = (vec2(tx, ty) + (px + py + pz) + 0.5) 
        		* vec2(0.25, bufferAspect*0.25);
    
    return texture(buffer, uv, -100.0);
}

// Function 1232
void AnimateUV (inout vec2 uv)
{
    if (iMouse.z > 0.0)
    {
        uv -= vec2(0.0,0.5) * iResolution.y / iResolution.x;;
        uv *= vec2(iMouse.y / iResolution.y);
        uv += vec2(1.5 * iMouse.x / iResolution.x, 0.0);
        
    }
    else
    {    
    	uv += vec2(sin(iTime * 0.3)*0.5+0.5, sin(iTime * 0.7)*0.5+0.5);
    	uv *= (sin(iTime * 0.3)*0.5+0.5)*3.0 + 0.2;
    }
}

// Function 1233
vec2 px2uv(in vec2 px)
{
	return vec2(px / iResolution.xy);
}

// Function 1234
float Tonemap_Uchimura(float x) {
    const float P = 1.0;  // max display brightness
    const float a = 1.0;  // contrast
    const float m = 0.22; // linear section start
    const float l = 0.4;  // linear section length
    const float c = 1.33; // black
    const float b = 0.0;  // pedestal
    return Tonemap_Uchimura(x, P, a, m, l, c, b);
}

// Function 1235
vec2 polarMap(vec2 uv, float inner) {

    uv = vec2(0.5) - uv;
    
    float px = 1.0 - fract(atan(uv.y, uv.x) / M_TWO_PI + 0.25);
    float py = (sqrt(uv.x * uv.x + uv.y * uv.y) * (1.0 + inner * 2.0) - inner) * 2.0;
    
    return vec2(px, py);
}

// Function 1236
float mapScene(in vec3 p) {
    float r = iTime * 2.0;
    float c = cos(r), s = sin(r);
    mat2 rmat = mat2(c, -s, s, c);

    p.yz *= rmat;
    p.xz *= rmat;

    vec3 q = abs(p) - 0.5;
    float box = max(q.x, max(q.y, q.z));

    box = max(box, -max(q.x, q.y) - 0.03);
    box = max(box, -max(q.x, q.z) - 0.03);
    box = max(box, -max(q.y, q.z) - 0.03);

    return box;
}

// Function 1237
float Map(vec3 point){vec3 motion;return Map(point, motion);}

// Function 1238
Shape map(vec3 c){ // Maps everything together (the Background and Character) 
  Shape face = character(c);
  Shape ground = background(c);
  face.dist = min(face.dist, ground.dist); 
  face.color = mix(face.color*2., ground.color, mixColors(ground.dist, face.dist, 0.5)); 
  return face;
}

// Function 1239
float map(vec3 p){
    
   
    float sf = cellTile(p*.25); // Cellular layer.
    
/*    
     p.xy -= path(p.z); // Move the scene around a sinusoidal path.
     p.xy = rot2(p.z/12.)*p.xy; // Twist it about XY with respect to distance.
    
     float n = dot(sin(p*1. + sin(p.yzx*.5 + iTime*.0)), vec3(.25)); // Sinusoidal layer.
     
     return 2. - abs(p.y) + n + (.5-sf)*.25; // Warped double planes, "abs(p.y)," plus surface layers.
*/

     float n = dot(sin(p*.5 + sin(p.yzx)), vec3(.333));
    
     // Standard tunnel. Comment out the above first.
     return 2.5 - length(p.xy - path(p.z)) - sf*.75 +  n;

 
}

// Function 1240
vec4 BoxMapFast( sampler2D sam, in vec3 p, in vec3 n, in float k )
{
  vec3 m = pow( abs(n), vec3(k) );
  vec4 x = textureLod( sam, p.yz ,0.);
  vec4 y = textureLod( sam, p.zx ,0.);
  vec4 z = textureLod( sam, p.xy ,0.);
  return (x*m.x + y*m.y + z*m.z)/(m.x+m.y+m.z);
}

// Function 1241
float map(vec3 p)
{
    s = sdf(1000.0, p, NONE);
    
    nose(vec3(gl.mp,0));
     
    for (int i = 1; i <= num; i++)
    {
        vec4 fish = load(i);
            
        float fd = length(fish.xy-gl.uv); 
        
        if (fd < 0.5 || gl.option!=0)
    	{
            vec3 fp = vec3(fish.x,fish.y,0);
            vec3 fdir = vec3(fish.zw,0);
            eye(i, fp, normalize(normalize(camPos-fp) + 1.5*fdir));
    	}
    }

    return s.dist;
}

// Function 1242
vec2 map(vec3 o) {
    return _m(vec2(length(o)-1.,0.),vec2(o.y+1.,1.));
}

// Function 1243
float map(vec3 ray) {
    float map = distance(ray, vec3(0,0,0)) - 0.35;
    return map;


}

// Function 1244
float map(vec3 p){
    float d=de(p);
    p.xy+=path(p.z);
    d=max(d,.01-max(abs(p.x),abs(p.y)));
    p.y+=.01;
    d=min(d,max(abs(p.x)-.001,abs(p.y)-.001));
    return d;
}

// Function 1245
vec2 map(vec3 p)
{
    vec3 q = mod(p+0.5*2.,2.)-2.*.5;
    vec2 pl = vec2(.7+p.y,0.);
    
    vec2 sphere = vec2(length(q) - .3,1.);
    
    vec2 box = vec2(sdOctahedron(q,.25),1.);
    
    vec2 morph = vec2(mix(sphere.x,box.x,sin(iTime)),1);

//    sdf = min(sdf,sphere);
	return (morph.x < pl.x) ? morph : pl;

}

// Function 1246
vec4 rgbToHpluv(float x, float y, float z, float a) {return rgbToHpluv( vec4(x,y,z,a) );}

// Function 1247
vec2 nmapu(vec2 x){ return x*.5+.5; }

// Function 1248
vec3 robobo1221sTonemap(vec3 x){
    return x / sqrt(x*x + 1.0);
}

// Function 1249
float hsluv_maxSafeChromaForL(float L){
    mat3 m2 = mat3(
         3.2409699419045214  ,-0.96924363628087983 , 0.055630079696993609,
        -1.5373831775700935  , 1.8759675015077207  ,-0.20397695888897657 ,
        -0.49861076029300328 , 0.041555057407175613, 1.0569715142428786  
    );
    float sub0 = L + 16.0;
    float sub1 = sub0 * sub0 * sub0 * .000000641;
    float sub2 = sub1 > 0.0088564516790356308 ? sub1 : L / 903.2962962962963;

    vec3 top1   = (284517.0 * m2[0] - 94839.0  * m2[2]) * sub2;
    vec3 bottom = (632260.0 * m2[2] - 126452.0 * m2[1]) * sub2;
    vec3 top2   = (838422.0 * m2[2] + 769860.0 * m2[1] + 731718.0 * m2[0]) * L * sub2;

    vec3 bounds0x = top1 / bottom;
    vec3 bounds0y = top2 / bottom;

    vec3 bounds1x =              top1 / (bottom+126452.0);
    vec3 bounds1y = (top2-769860.0*L) / (bottom+126452.0);

    vec3 xs0 = hsluv_intersectLineLine(bounds0x, bounds0y, -1.0/bounds0x, vec3(0.0) );
    vec3 xs1 = hsluv_intersectLineLine(bounds1x, bounds1y, -1.0/bounds1x, vec3(0.0) );

    vec3 lengths0 = hsluv_distanceFromPole( xs0, bounds0y + xs0 * bounds0x );
    vec3 lengths1 = hsluv_distanceFromPole( xs1, bounds1y + xs1 * bounds1x );

    return  min(lengths0.r,
            min(lengths1.r,
            min(lengths0.g,
            min(lengths1.g,
            min(lengths0.b,
                lengths1.b)))));
}

// Function 1250
vec3 mapLadyBug( vec3 p, float curmin )
{
    
    float db = length(p-vec3(0.0,-0.35,0.05))-1.3;
    if( db>curmin ) return vec3(10000.0,0.0,0.0);
    
    float dBody = sdEllipsoid( p, vec3(0.0), vec3(0.8, 0.75, 1.0) );
    dBody = smax( dBody, -sdEllipsoid( p, vec3(0.0,-0.1,0.0), vec3(0.75, 0.7, 0.95) ), 0.05 );
    dBody = smax( dBody, -sdEllipsoid( p, vec3(0.0,0.0,0.8), vec3(0.35, 0.35, 0.5) ), 0.05 );
  	dBody = smax( dBody, sdEllipsoid( p, vec3(0.0,1.7,-0.1), vec3(2.0, 2.0, 2.0) ), 0.05 );
  	dBody = smax( dBody, -abs(p.x)+0.005, 0.02 + 0.1*clamp(p.z*p.z*p.z*p.z,0.0,1.0) );

    vec3 res = vec3( dBody, MAT_LADY_BODY, 0.0 );

    // --------
    vec3 hc = vec3(0.0,0.1,0.8);
    vec3 ph = rotateX(p-hc,0.5);
    float dHead = sdEllipsoid( ph, vec3(0.0,0.0,0.0), vec3(0.35, 0.25, 0.3) );
    dHead = smax( dHead, -sdEllipsoid( ph, vec3(0.0,-0.95,0.0), vec3(1.0) ), 0.03 );
    dHead = min( dHead, sdEllipsoid( ph, vec3(0.0,0.1,0.3), vec3(0.15,0.08,0.15) ) );

    if( dHead < res.x ) res = vec3( dHead, MAT_LADY_HEAD, 0.0 );
    
    res.x += 0.0007*sin(150.0*p.x)*sin(150.0*p.z)*sin(150.0*p.y); // iqiq

    // -------------
    
    vec3 k1 = vec3(0.42,-0.05,0.92);
    vec3 k2 = vec3(0.49,-0.2,1.05);
    float dLegs = 10.0;

    float sx = sign(p.x);
    p.x = abs(p.x);
    for( int k=0; k<3; k++ )
    {   
        vec3 q = p;
        q.y -= min(sx,0.0)*0.1;
        if( k==0) q += vec3( 0.0,0.11,0.0);
        if( k==1) q += vec3(-0.3,0.1,0.2);
        if( k==2) q += vec3(-0.3,0.1,0.6);
        
        vec2 se = sdLine( q, vec3(0.3,0.1,0.8), k1 );
        se.x -= 0.015 + 0.15*se.y*se.y*(1.0-se.y);
        dLegs = min(dLegs,se.x);

        se = sdLine( q, k1, k2 );
        se.x -= 0.01 + 0.01*se.y;
        dLegs = min(dLegs,se.x);

        se = sdLine( q, k2, k2 + vec3(0.1,0.0,0.1) );
        se.x -= 0.02 - 0.01*se.y;
        dLegs = min(dLegs,se.x);
    }
    
    if( dLegs<res.x ) res = vec3(dLegs,MAT_LADY_LEGS, 0.0);


    return res;
}

// Function 1251
float remap(float a, float b, float c, float d, float t) {
	return ((t-a) / (b-a)) * (d-c) + c;
}

// Function 1252
vec3 tonemap (vec3 color) {
  vec3  hue = rgb_to_hue(color);
  float sat = rgb_to_sat(color);
  float lum = rgb_to_lum(color);

  // smooth-clamp
  sat = -log(exp(-sat*10.)+exp(-10.))/10.;

  /* tonemapping options:
       - desaturate when very bright
       - smooth-clamp brightness to a maximum that still
          allows some color variation                              */
  // sat = sat*(exp(-lum*lum*2.));
  // lum = .8*(1.-exp(-lum));

  color = lum*mix(vec3(1.),hue,sat);
  return color;
}

// Function 1253
float hsluv_yToL(float Y){  return Y <= 0.0088564516790356308 ? Y * 903.2962962962963 : 116.0 * pow(Y, 1.0 / 3.0) - 16.0; }

// Function 1254
vec4 mapSmoke(in vec3 pos)
{
    vec3 pos2 = pos;
    pos2-= chimneyOrig + vec3(5.65, -0.8, 0.);
    
    // Calculating the smoke domain (3D space giving the probability to have smoke inside
    float sw = max(tubeDiam*0.84 + 0.25*pos2.y*(1. + max(0.15*pos2.y, 0.)) + 0.2*windIntensity*(pos.y + chimneyOrig.x - tubeclen - tubeLen2 + 0.3), 0.);
    float smokeDomain = smoothstep(1.2 + sw/4.3, 0.7 - sw*0.5, length(pos2.xz)/sw);
    
    float d;
    vec4 res;
    if (smokeDomain>0.1)
    {           
    	// Space modification in function of the time and wind
        vec3 q = pos2*vec3(1., 1. + 0.5*windIntensity, 1.) + vec3(0.0,-currTime*smokeSpeed + 10.,0.0);
    	q/= smokeScale;
        q.y+= 8.*dWindIntensity + 1.5/(0.7 + dWindIntensity);
        
        // Turbulence of the smoke
        #ifdef smoke_turbulence
        if (smokeTurbulence>0.)
        {
        	float n = smoothstep(4., 0., pos2.y + 3.2)*smokeTurbulence*noise(q*smokeTurbulenceScale)/(currTime + 3.);
        	q.xy = rotateVec(-q.xy, pos.z*n);
        	q.yz = rotateVec(-q.yz, pos.x*n);
        	q.zx = rotateVec(-q.zx, pos.y*n);
        }
        #endif
        
        // Calculation of the noise
        d = clamp(0.6000*noise(q), 0.4, 1.); q = q*2.02;  
        d+= 0.2500*noise(q); q = q*2.03;
        d+= 0.1200*noise(q); q = q*2.08;
        d+= 0.0500*noise(q);
        
        #ifdef heat_refraction
        // Calculation of the refraction due to the temperature difference in the air
        float rrf = smokeDomain*(1. - clamp((pos2.y + 2.8)*0.55, 0., 1.))*smoothstep(0., .3, pos2.y + 3.2);
        rayRef.x+= (smokeRefInt*noise(q*3.27 + q*4.12) - 0.5*smokeRefInt)*rrf;
        rayRef.y+= (smokeRefInt*noise(q*3.37 - q*3.96) - 0.5*smokeRefInt)*rrf;
        rayRef.z+= (smokeRefInt*noise(q*3.11 + q*3.82) - 0.5*smokeRefInt)*rrf;
        #endif

        d = d - 0.3 - smokeBias - 0.04*pos.y + 0.05*(1. + windIntensity);
        d = clamp(d, 0.0, 1.0);
        
 		res = vec4(pow(d*smokeDomain, smokePow));

    	// Some modifications of color and alpha
		res.xyz = mix(smokeCol, 0.2*vec3(0.4, 0.4, 0.4), res.x);
		res.xyz*= 0.2 + 0.2*smoothstep(-2.0, 1.0, pos.y);
    	res.w*= max(smokeDens - 1.8*sqrt(pos.y - 4.), 0.);
    }
    else
    {
        d = 0.;
        res = vec4(0.);
    }
	
	return res;
}

// Function 1255
vec3 mapCoord(vec2 uv)
{
    uv = (fract(uv / 4.) - .5) * 2.;
    return vec3(1., -uv.yx * vec2(1, 1) * (1. - 1. / 1024.));
}

// Function 1256
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

// Function 1257
void planeUVAxis(in vec3 p, in vec3 n, out vec3 u, out vec3 v)
{
    vec3 worldUp = vec3(0.0, 1.0, 0.0);
    u = normalize(cross(worldUp, n));
    v = normalize(cross(n, u));
}

// Function 1258
vec2 map7(vec3 pos) {
	pos.xz = rotate(pos.xz, iTime*2.);
    pos.xy = rotate(pos.xy, -iTime*2.);

    float s = roomWidth * 0.3 + sin(iTime*2.)*0.01;
	float dist = fIcosahedron(pos, s);
    float identifier = abs(pos.y);
    vec2 res = vec2(dist, identifier);

    return res;
}

// Function 1259
float heightMap3(in vec2 pos)
{
    pos /= 7.;
    vec2 m1 = mod(pos, 1.), m2 = mod(pos, 2.);
    pos = mix(m1, 1.-m1, max(vec2(0.), sign(m2-1.)));
    //if (m2.x >= 1.) m1.x = 1. - m1.x;
    //pos = m1;
    pos += vec2(4,.5);
    vec3 p = vec3(pos, 1.);
    float d = 0.;
    for (int i=0; i<24; ++i)
    {
        p = abs(p) / dot(p.xy, p.xy);
        //d = min(d, exp(- p.x/p.z));
        d += 1.*( exp(-p.x/p.z*(1.+1.*float(i*i))) )/float(1+i);
        if (float(i)>(18.+6.*sin(iTime)))
            break;
        p.xy -= .99;//+.02*sin(iTime);
    }
    return d;//smoothstep(0.1,.0, d);
}

// Function 1260
float map(vec3 p)
{
    return p.y-(terrain(p.zx*0.07))*2.7-1.;
}


```