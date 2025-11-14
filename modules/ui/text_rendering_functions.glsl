// Reusable Text Rendering UI/2D Functions
// Automatically extracted from UI/2D graphics-related shaders

// Function 1
vec4 blurTexture(sampler2D sampler, vec2 uv, float minMipmap, float maxMipmap)
{
 	vec4 sumCol = vec4(0.0);
    
    //adding different mipmaps
    for (float i = minMipmap; i < maxMipmap; i++)
    {
     	sumCol += texture(sampler, uv, i);   
    }
    
    //average
    return sumCol / (maxMipmap - minMipmap);
}

// Function 2
vec4 textureBicubic(sampler2D sampler, vec2 texCoords) 
{
   vec2 myCoords;
   vec2 invTexSize = 1.0 / vec2(iChannelResolution[1].xy);
   myCoords = texCoords * vec2(iChannelResolution[1].xy) - 0.5;
   vec2 fxy = fract(myCoords);
   myCoords -= fxy;
   vec4 xcubic = cubic(fxy.x);
   vec4 ycubic = cubic(fxy.y);
	vec4 c = myCoords.xxyy + vec2(-0.5,1.5).xyxy;
	vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
	vec4 offset = c + vec4(xcubic.yw, ycubic.yw) / s;
	offset *= invTexSize.xxyy;
	vec4 sample0 = texture(sampler, offset.xz);
	vec4 sample1 = texture(sampler, offset.yz);
	vec4 sample2 = texture(sampler, offset.xw);
	vec4 sample3 = texture(sampler, offset.yw);
	float sx = s.x / (s.x + s.y);
	float sy = s.z / (s.z + s.w);
	return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

// Function 3
float text_z(vec2 U) {
    initMsg;
    U.x+=3.*(0.5-0.2812*(res.x/0.5));
    C(90);C(111);C(111);C(109);
    endMsg;
}

// Function 4
void glyph_Dot()
{
  MoveTo(x);
  Bez3To(x*1.2,x*1.2+y*0.2,x+y*0.2);
  Bez3To(x*0.8+y*0.2,x*0.8,x);
}

// Function 5
vec4 MAT_triplanarTexturing(vec3 p, vec3 n)
{
    p = fract(p+0.5);
    
    float sw = 0.20; //stiching width
    vec3 stitchingFade = vec3(1.)-smoothstep(vec3(0.5-sw),vec3(0.5),abs(p-0.5));
    
    float fTotal = abs(n.x)+abs(n.y)+abs(n.z);
    vec4 cX = abs(n.x)*texture(iChannel1,p.zy);
    vec4 cY = abs(n.y)*texture(iChannel1,p.xz);
    vec4 cZ = abs(n.z)*texture(iChannel1,p.xy);
    
    return  vec4(stitchingFade.y*stitchingFade.z*cX.rgb
                +stitchingFade.x*stitchingFade.z*cY.rgb
                +stitchingFade.x*stitchingFade.y*cZ.rgb,cX.a+cY.a+cZ.a)/fTotal;
}

// Function 6
vec4 put_text_target(vec4 col, vec2 uv, vec2 pos, float scale)
{
	float unit = asp * scale * 0.1;
    float h = 0.;
    vec2 sc = vec2(unit, unit*0.8);
    
    // S
    h = max(h, word_map(uv, pos, 83, sc));
    // h
    h = max(h, word_map(uv, pos+vec2(unit*0.4, 0.), 104, sc));
    // o
    h = max(h, word_map(uv, pos+vec2(unit*0.8, 0.), 111, sc));
    // w
    h = max(h, word_map(uv, pos+vec2(unit*1.2, 0.), 119, sc));
    
    // T
    h = max(h, word_map(uv, pos+vec2(unit*2.0, 0.), 84, sc));
    // a
    h = max(h, word_map(uv, pos+vec2(unit*2.35, 0.), 97, sc));
    // r
    h = max(h, word_map(uv, pos+vec2(unit*2.8, 0.), 114, sc));
    // g
    h = max(h, word_map(uv, pos+vec2(unit*3.15, 0.), 103, sc));
    // e
    h = max(h, word_map(uv, pos+vec2(unit*3.5, 0.), 101, sc));
    // t
    h = max(h, word_map(uv, pos+vec2(unit*3.95, 0.), 116, sc));
    
    col = mix(col, vec4(1.-vec3(h), 1.), h);
    
    return col;
}

// Function 7
vec4 SampleTextureCatmullRom( vec2 uv, vec2 texSize )
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 samplePos = uv * texSize;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
    vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    
    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 texPos0 = texPos1 - vec2(1.0);
    vec2 texPos3 = texPos1 + vec2(2.0);
    vec2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    vec4 result = vec4(0.0);
    result += sampleLevel0( vec2(texPos0.x,  texPos0.y)) * w0.x * w0.y;
    result += sampleLevel0( vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos0.y)) * w3.x * w0.y;

    result += sampleLevel0( vec2(texPos0.x,  texPos12.y)) * w0.x * w12.y;
    result += sampleLevel0( vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos12.y)) * w3.x * w12.y;

    result += sampleLevel0( vec2(texPos0.x,  texPos3.y)) * w0.x * w3.y;
    result += sampleLevel0( vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos3.y)) * w3.x * w3.y;

    return result;
}

// Function 8
vec4 getTexture(float id, vec2 c) {
    vec2 gridPos = vec2(mod(id, 8.), floor(id / 8.));
	return texture(iChannel2, (c + gridPos * 16.) / iChannelResolution[3].xy);
}

// Function 9
vec4 glyph( vec2 p, int iChannel )
{
    vec4 back  = vec4( .1, .2, .5, 1.0 ); // Royal Blue background color

    float scale = (2.25 * 16.0) / iResolution.x;

    if (iChannel >= 2 )
    {
        scale *= 2.0; // 16x16 same area as 8x8
        back *= 0.5; // darken SDF background so it is more obvious
    }
    
    p *= scale;
    p /= iResolution.xy;
    
    float d;
    float a;
    
    if (iChannel == 0)
        a = texture( iChannel0, p ).a;
    if (iChannel == 1)
        a = texture( iChannel1, p ).a;
    if (iChannel == 2)
        a = texture( iChannel2, p ).a;
    if (iChannel == 3)
    {
        d = texture( iChannel2, p ).a;
        
        // w = 0.0 Sharp
        // w = 0.5 Blury
        //float w = 0.5 - 1.0/3.0; // 0.5 - (0.5-1/3) = 0.333, 0.5+(0.5-1/3) = 0.666
        float w = 0.0; // 0.5, 0.5
        if (iMouse.z > 0.0)
            w = (iMouse.x / iResolution.x)*0.5; // [0,1] -> [0.0,0.5]

        a = smoothstep( 0.5 - w, 0.5 + w, d ); // smoothstep() is slightly blurry
    }
    if (iChannel == 4) // sharper
    {
               d = texture( iChannel2, p ).a;
        float  s = d - 0.5;
        float _2 = 0.70710678118; // SQRT2_DIV_2

#ifdef SMOOTH_1
        float  dx = dFdx( s );
        float  dy = dFdy( s );
        float  g  = _2 * length( vec2( dx, dy ) );     
        a  = smoothstep( -g, g, s );
#endif
        
#ifdef SMOOTH_2
        // paulhoux's version: float w = fwidth( d ); a = smoothstep( 0.5 - w, 0.5 + w, d );
        float w = fwidth( d );
        a = smoothstep( 0.5 - w, 0.5 + w, d );
#endif
        
#ifdef SMOOTH_3
        // optimized (1) aa:   float v = s / fwidth( s ); a = clamp( v + 0.5, 0.0, 1.0 );
        // Does anyone know where this comes from?
        // Detheroc mentioned it in http://computergraphics.stackexchange.com/questions/306/sharp-corners-with-signed-distance-fields-fonts
        float v = s / fwidth( s );
        a = clamp( v + 0.5, 0.0, 1.0 );
#endif
    }
    
    vec4 fore  = vec4( 0., .5, 1., 1.0 ); // Sky Blue
    vec4 color = mix( back, fore, a );

    return color;
}

// Function 10
float NumFont_Eight( vec2 vTexCoord )
{
    float fResult = 0.0;
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(2, 1), vec2(10,13) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 2), vec2(11,12) ));
    
    float fHole = NumFont_Rect( vTexCoord, vec2(5, 4), vec2(7,5) );
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(5, 9), vec2(7,10) ));
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(-1, 6), vec2(1,8) ));
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(11, 6), vec2(17,8) ));
    fHole = max( fHole, NumFont_Pixel( vTexCoord, vec2(2, 7) ));
    fHole = max( fHole, NumFont_Pixel( vTexCoord, vec2(10, 7) ));

    fResult = min( fResult, 1.0 - fHole );    
    
    return fResult;
}

// Function 11
vec3 drawText3( in vec4 fragColor, in vec2 fragCoord ) {
    float display_width = 1010.;
    float cc = floor(display_width / (g_cw * (1. + g_cwb))); // character count per line
    
    vec2 uv = (fragCoord.xy) / iResolution.xx;
    uv.y = iResolution.y/iResolution.x - uv.y;  // type from top to bottom, left to right   
    uv *= display_width;

    int cs = int(floor(uv.x / (g_cw * (1. + g_cwb))) + cc * floor(uv.y/(g_ch * (1. + g_chb))));

    uv = mod_uv(uv);
    uv.y = g_ch * (1. + g_chb) - uv.y; // paint the character from the bottom left corner
    vec3 ccol = .35 * vec3(.1, .3, .2) * max(smoothstep(3., 0., uv.x), smoothstep(5., 0., uv.y));   
    uv -= vec2(g_cw * g_cwb * .5, g_ch * g_chb * .5);
    
    float tx = 10000.;
    int idx = 0;
    
    NL 
    NL 
    NL 
    NL 
    NL 
    NL 
    SP SP SP SP SP SP SP SP SP SP SP SP SP SP SP SP SP SP Y O U SP W I N 
    NL
        
    vec3 tcol = vec3(1.0, 0.7, 0.0) * smoothstep(.2, .0, tx);
    
    vec3 terminal_color = tcol;
    
    return terminal_color;
}

// Function 12
mat3 glyph_8_9_numerics(float g) {
    GLYPH(32) 0);
    GLYPH(42)00001000.,00101010.,00011100.,00111110.,00011100.,00101010.,00001000.,0,0);
    GLYPH(46)00000000.,00000000.,00000000.,00000000.,00000000.,00011000.,00011000.,0,0);
    // numerics  ==================================================
    GLYPH(49)00011000.,00001000.,00001000.,00001000.,00001000.,00001000.,00011100.,0,0);
    GLYPH(50)00111100.,01000010.,00000010.,00001100.,00110000.,01000000.,01111110.,0,0);
    GLYPH(51)00111100.,01000010.,00000010.,00011100.,00000010.,01000010.,00111100.,0,0);
    GLYPH(52)01000100.,01000100.,01000100.,00111110.,00000100.,00000100.,00000100.,0,0);
    GLYPH(53)01111110.,01000000.,01111000.,00000100.,00000010.,01000100.,00111000.,0,0);
    GLYPH(54)00111100.,01000010.,01000000.,01011100.,01100010.,01000010.,00111100.,0,0);
    GLYPH(55)00111110.,01000010.,00000010.,00000100.,00000100.,00001000.,00001000.,0,0);
    GLYPH(56)00111100.,01000010.,01000010.,00111100.,01000010.,01000010.,00111100.,0,0);
    GLYPH(57)00111100.,01000010.,01000010.,00111110.,00000010.,00000010.,00111100.,0,0);
    GLYPH(58)00111100.,00100100.,01001010.,01010010.,01010010.,00100100.,00111100.,0,0);
    return mat3(0);
}

// Function 13
bool get_load_texture(in sampler2D s)
{
    return texelFetch(s, CTRL_LOAD_TEXTURE, 0).w > 0.05;
}

// Function 14
vec2 get_text_position(int ascii)
{
	int x = (ascii % 16);
    int y = 15-(ascii / 16);
    
    return vec2(float(x), float(y))*0.0625;
}

// Function 15
vec3 texturize( sampler2D sa, vec3 p, vec3 n )
{
	vec3 x = texture( sa, p.yz ).xyz;
	vec3 y = texture( sa, p.zx ).xyz;
	vec3 z = texture( sa, p.xy ).xyz;
	return x*abs(n.x) + y*abs(n.y) + z*abs(n.z);
}

// Function 16
vec3 TexTLite6_4( vec2 vTexCoord, float fRandom, vec3 vFlatCol, vec3 vLightCol )
{
	vec2 vLocalCoord = vTexCoord;
	vLocalCoord = mod(vLocalCoord, 64.0 );
	
    vec2 vAbsLocal = abs( vLocalCoord - 32. );
    
    float fDist = (vAbsLocal.x + vAbsLocal.y) / 16.0;
    fDist = fDist * fDist;
    
    if ( fDist > 1.0 )
    {
        return vFlatCol * (0.5 + fRandom * 0.5);
    }
    
    float fLight = clamp(1.0 - fDist * fDist, 0.0, 1.0);
	return min( vec3(1.0), vLightCol * (fLight * 0.75 + 0.25) + pow( fLight, 5.0 ) * 0.4);    
}

// Function 17
vec4 put_text_point_num(vec4 col, vec2 uv, vec2 pos, float scale, int num)
{
	float unit = asp * scale * 0.1;
    float h = 0.;
    vec2 sc = vec2(unit, unit);
    int d = num % 10;
    int t = num / 10;
    
    
    h = max(h, word_map(uv, pos, 80, sc));
    
    if(t > 0)  // Ptd
    {
    	h = max(h, word_map(uv, pos+vec2(unit*0.3, -unit*0.1), 48+t, sc*0.6));
        h = max(h, word_map(uv, pos+vec2(unit*0.5, -unit*0.1), 48+d, sc*0.6));
    }
    else  //Pd
    {
    	h = max(h, word_map(uv, pos+vec2(unit*0.3, -unit*0.1), 48+d, sc*0.6));
    }
    
    col = blend(col, vec4(0., 0.435, 1., h));
    
    return col;
}

// Function 18
float NumFont_Circle( vec2 vTexCoord )
{
    float fResult = 0.0;
    
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(2, 2), vec2(10,12) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(4, 1), vec2(8,13) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 4), vec2(11,10) ));
    
    return fResult;
}

// Function 19
float NumFont_Five( vec2 vTexCoord )
{
    float fResult = 0.0;

    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 1), vec2(10,3) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 1), vec2(3,8) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 6), vec2(9,8) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(8, 7), vec2(10,12) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(8, 8), vec2(11,11) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 11), vec2(9,13) ));
    
    return fResult;
}

// Function 20
vec4 texture_blurred2_quantized(in sampler2D tex, vec2 uv, vec3 q)
{
    return (quantize(texture(iChannel0, uv), q)
        + quantize(texture(iChannel0, vec2(uv.x+1.0, uv.y+1.0)), q)
		+ quantize(texture(iChannel0, vec2(uv.x+1.0, uv.y-1.0)), q)
        + quantize(texture(iChannel0, vec2(uv.x-1.0, uv.y+1.0)), q)
		+ quantize(texture(iChannel0, vec2(uv.x-1.0, uv.y-1.0)), q)
		+ quantize(texture(iChannel0, vec2(uv.x+1.0, uv.y)), q)
		+ quantize(texture(iChannel0, vec2(uv.x-1.0, uv.y)), q)
		+ quantize(texture(iChannel0, vec2(uv.x, uv.y+1.0)), q)
		+ quantize(texture(iChannel0, vec2(uv.x, uv.y-1.0)), q))/9.0;
}

// Function 21
vec3 texturef( in vec2 p )
{
    return texture( iChannel0, p ).xyz;
}

// Function 22
vec3 mytexture( vec3 p, vec3 n, float matid )
{
	p += 0.1;
	vec3 ip  = floor(p/20.0);
	vec3 fp  = fract(0.5+p/20.0);

	float id = fract(sin(dot(ip,vec3(127.1,311.7, 74.7)))*58.5453123);
	id = mix( id, 0.3, matid );
	
	float f = mod( ip.x + mod(ip.y + mod(ip.z, 2.0), 2.0), 2.0 );
	
	float g = 0.5 + 1.0*noise( p * mix( vec3(0.2+0.8*f,1.0,1.0-0.8*f), vec3(1.0), matid) );
	
	g *= mix( smoothstep( 0.03, 0.04, abs(fp.x-0.5)/0.5 )*
	          smoothstep( 0.03, 0.04, abs(fp.z-0.5)/0.5 ),
			  1.0,
			  matid );
	
	vec3 col = 0.5 + 0.5*sin( 1.0 + 2.0*id + vec3(0.0,1.0,2.0) );
	
	return col * g;
}

// Function 23
vec3 textureBox1(vec3 p)
{
    vec3 ap=abs(p),f=step(ap.zxy,ap.xyz)*step(ap.yzx,ap.xyz);
    vec2 uv=f.x>.5?p.yz:f.y>.5?p.xz:p.xy;
    float l=clamp(-normalize(p-vec3(0,1,0)).y,0.,1.);
    vec2 b=box2(boxxfrm*p,boxxfrm*(vec3(0,1,0)-p));
    // Some lighting and a shadow (and approximated AO).
    float s=mix(.2,1.,smoothstep(0.,.8,length(p.xz)));
    vec3 d=.6*(1.-smoothstep(-1.,1.,p.y))*vec3(0.3,0.3,.5)*s+smoothstep(0.9,.97,l)*vec3(1,1,.8)*step(b.y,b.x);
    return texture(iChannel1,uv).rgb*d;
}

// Function 24
float glyph2(in vec2 _st, in float seed){
    float r = rand(seed, PHI);
    float a = 0.0;

    _st -= .5;
    if(r <= 1./24.){
        a = drawSpike(_st * rotate2d(TRQT_PI)+vec2(0.470,0.480));
    }else{
        // 7 have same cresent
        if(r <= 7./24.){
            a += drawMoon(_st, .1, .4);
            if(r <= 2./24.){ // Iron
                a += drawSpike(_st * rotate2d(-1.992) * scale(vec2(1.990,1.460)) + translate(vec2(0.430,-0.050)));
                a += drawSpike(_st * rotate2d(-2.648) * scale(vec2(1.990,1.460)) + translate(vec2(0.490,0.020)));
                a += drawCircle(_st-vec2(-0.560,-0.180),0.006);
            }else if(r <= 3./24.){ // Zinc
                a += drawSpike(_st * rotate2d(-1.032) * scale(vec2(1.690,1.560)) + translate(vec2(0.530,-1.080)));
                a += drawSpike(_st * rotate2d(-1.060) * scale(vec2(1.690,1.560)) + translate(vec2(0.300,-0.990)));
                a += drawCircle(_st-vec2(-0.360,-0.390),0.006);
            }else if(r <= 4./24.){ // unkown/H
                a += drawSpike(_st * rotate2d(-0.704) * scale(vec2(1.690,1.560)) + translate(vec2(0.600,-1.070)));
                a += drawSpike(_st * rotate2d(0.348) * scale(vec2(1.690,1.560)) + translate(vec2(0.450,-1.070)));
                a += drawCircle(_st-vec2(-0.360,-0.390),0.006);
            }else if(r <= 5./24.){ // Malatium
                a += drawSpike(_st * rotate2d(2.312) * scale(vec2(1.990,0.800)) + vec2(1.490,0.470));
                a += drawSpike(_st * rotate2d(-1.824) * scale(vec2(1.790,1.560)) + translate(vec2(0.530,-1.120)));
                a += drawSpike(_st * rotate2d(0.548) * scale(vec2(1.690,1.560)) + translate(vec2(0.460,-1.160)));
                a += drawSpike(_st * rotate2d(-0.260) * scale(vec2(1.990,1.660)) + translate(vec2(0.450,-1.150)));
                a += drawSpike(_st * rotate2d(-0.916) * scale(vec2(1.990,1.660)) + translate(vec2(0.500,-1.120)));
                a += drawCircle(_st-vec2(-0.360,-0.390),0.006);
            }else if(r <= 6./24.){ // Lerasium
                _st += 0.012;
                a += drawSpike(_st * rotate2d(1.928) * scale(vec2(1.690,1.560)) + translate(vec2(0.400,-0.270)));
                a += drawSpike(_st * scale(vec2(1.430,1.360)) * rotate2d(2.648)  + translate(vec2(0.590,-0.30))); //slight curve to nail with different order of transforms failed
                a += drawCircle(_st-vec2(-0.360,-0.390),0.006);
                a += partMoon(_st* rotate2d(0.488) * scale(vec2(1.290,0.910))+ vec2(+0.230,+0.110),1.648);
            }else if(r <= 7./24.){ // Electrum
                a += partMoon(_st  * rotate2d(7.088)* scale(vec2(1.3,1.000))+vec2(0.080,-0.020),0.472);
                _st += 0.012;
                a += drawSpike(_st  * rotate2d(3.952) * scale(vec2(1.290,1.260))  + translate(vec2(0.590,-0.30))); //slight curve to nail with different order of transforms failed
                a += drawCircle(_st-vec2(-0.350,-0.570),0.006);
            }
        }else if(r <= 10./24.){
            a += drawMoon(_st* rotate2d(5.512)* scale(vec2(0.960,0.980)),0.050, .28);
            if(r <= 8./24.){ // Cadmium
                a += drawSpike(_st * rotate2d(-PI) * scale(vec2(1.090,1.260)) + translate(vec2(0.580,-0.470)));
                a += drawSpike(_st * rotate2d(-HALF_PI) * scale(vec2(1.990,1.460)) + translate(vec2(0.490,-1.120)));
                a += drawCircle(_st-vec2(-0.480,-0.200),0.006);
            }else if(r <= 9./24.){ // Chromium
                a += drawSpike(_st * rotate2d(-2.160) * scale(vec2(1.390,1.060)) + translate(vec2(0.580,-0.870)));
				a += partMoon(_st  * rotate2d(5.076)* scale(vec2(1.3,1.300))+vec2(-0.110,-0.050),1.968);
                a += drawCircle(_st-vec2(-0.480,-0.200),0.006);
            }else if(r <= 10./24.){ // Malatium
                _st = _st * rotate2d(-1.104); 
                a += drawSpike(_st * rotate2d(-2.524) * scale(vec2(1.790,1.560)) + translate(vec2(0.530,-1.120)));
                a += drawSpike(_st * rotate2d(3.740) * scale(vec2(1.690,1.560)) + translate(vec2(0.560,-0.060)));
				a += drawSpike(_st * rotate2d(-1.500) * scale(vec2(1.990,1.660)) + translate(vec2(0.500,-1.120)));
                a += drawSpike(_st * rotate2d(-0.404) * scale(vec2(1.990,1.660)) + translate(vec2(0.500,-1.120)));
                a += drawCircle(_st-vec2(-0.310,-0.350),0.006);
            }
        }else if(r <= 14./24.){
            if (r <= 12./24.) _st *= rotate2d(PI); // two left two right
            a += drawMoon(_st* rotate2d(-0.600)* scale(vec2(0.780,-0.910))+ vec2(0.010,0.000),0.060, .29);
            if(r <= 11./24.){ // unknown/C
                a += drawSpike(_st * rotate2d(-HALF_PI) * scale(vec2(1.990,1.460)) + translate(vec2(0.490,-1.120)));
                a += drawSpike(_st * rotate2d(+HALF_PI) * scale(vec2(1.990,1.460)) + translate(vec2(0.490,-1.120)));
                a += drawCircle(_st-vec2(-0.490,-0.500),0.006);
            }else if(r <= 12./24.){ // unknown/X
                a += drawSpike(_st * rotate2d(-HALF_PI) * scale(vec2(0.790,1.460)) + translate(vec2(0.540,-0.580)));
                a += drawCircle(_st-vec2(-0.60,-0.500),0.006);
            }else if(r <= 13./24.){ // unknown/J
                a += drawSpike(_st * rotate2d(-HALF_PI) * scale(vec2(1.990,1.460)) + translate(vec2(0.490,-1.120)));
                a += drawCircle(_st-vec2(-0.310,-0.520),0.006);
            }else if(r <= 14./24.){ // Bendalloy
                a += drawSpike(_st * rotate2d(-HALF_PI) * scale(vec2(0.790,1.460)) + translate(vec2(0.540,-0.580)));
                a += drawSpike(_st * rotate2d(-PI) * scale(vec2(0.790,1.460)) + translate(vec2(0.740,-0.580)));
                a += drawCircle(_st-vec2(-0.350,-0.470),0.006);
            }
        }else if(r <= 21./24.){
            if (r > 17./24.) _st *= rotate2d(PI) * scale(vec2(.8)); // three down rest up
            a += drawMoon(_st* rotate2d(TRQT_PI/2.) * scale(vec2(1.3))+ vec2(0.010,0.000),0.036, .29);
            if(r <= 15./24.){ // Tin
                a += drawSpike(_st * rotate2d(-HALF_PI) * scale(vec2(1.00,1.460)) + translate(vec2(0.520,-0.520)));
                a += drawCircle(_st-vec2(-0.490,-0.040),0.006);
                a += drawMoon(_st* rotate2d(TRQT_PI/2.) * scale(vec2(0.8))+ vec2(0.010,0.000),0.036, .29);
            }else if(r <= 16./24.){ // unknown/X
                a += drawSpike(_st * rotate2d(-2.440) * scale(vec2(1.00,1.460)) + translate(vec2(0.520,-0.520)));
                a += drawCircle(_st-vec2(-0.470,-0.640),0.006);
                a += drawMoon(_st* rotate2d(TRQT_PI/2.) * scale(vec2(0.8))+ vec2(0.010,0.000),0.036, .29);
            }else if(r <= 17./24.){ // Pewter
                a += drawSpike(_st * rotate2d(-2.440) * scale(vec2(1.00,1.460)) + translate(vec2(0.520,-0.520)));
                a += drawCircle(_st-vec2(-0.580,-0.590),0.006);
                a += partMoon(_st* rotate2d(TRQT_PI/2.376) * scale(vec2(0.8,1.2))+ vec2(0.010,0.200), 0.610);
            }else if(r <= 18./24.){ // Copper
                a += drawSpike(_st * rotate2d(0.720) * scale(vec2(0.990,1.460)) + translate(vec2(0.470,-0.7040)));
                a += drawSpike(_st * rotate2d(0.736) * scale(vec2(1.990,1.370)) + translate(vec2(0.60,-1.120)));
                a += drawCircle(_st-vec2(-0.350,-0.470),0.006);
            }else if(r <= 19./24.){ // Bronze
                a += drawSpike(_st * rotate2d(4.336) * scale(vec2(1.10,1.460)) + translate(vec2(0.490,-0.5240)));
                a += drawSpike(_st * rotate2d(3.992) * scale(vec2(1.390,1.370)) + translate(vec2(0.40,-0.320)));
                a += drawCircle(_st-vec2(-0.590,-0.640),0.006);
            }else if(r <= 20./24.){ // Atrium
                _st = _st * rotate2d(2.192); 
                a += drawSpike(_st * rotate2d(-2.524) * scale(vec2(2.990,1.560)) + translate(vec2(0.530,-1.380)));
                a += drawSpike(_st * rotate2d(3.740) * scale(vec2(2.890,1.60)) + translate(vec2(0.540,-0.130)));
				a += drawSpike(_st * rotate2d(-1.500) * scale(vec2(2.890,1.660)) + translate(vec2(0.600,-1.220)));
                a += drawSpike(_st * rotate2d(-0.404) * scale(vec2(2.80,1.660)) + translate(vec2(0.540,-1.220)));
                a += drawCircle(_st-vec2(-0.310,-0.350),0.006);
            }else if(r <= 21./24.){ // Aluminum
                _st = _st * rotate2d(2.192); 
                a += drawSpike(_st * rotate2d(-2.588) * scale(vec2(2.990,1.560)) + translate(vec2(0.530,-1.380)));
                a += drawSpike(_st * rotate2d(1.092) * scale(vec2(2.890,1.60)) + translate(vec2(0.460,-1.280)));
                a += drawSpike(_st * rotate2d(-0.740) * scale(vec2(2.80,1.660)) + translate(vec2(0.540,-1.220)));
                a += drawCircle(_st-vec2(-0.310,-0.350),0.006);
            }
        }else if(r <= 22./24.){//Duralumin
        	a += drawMoon(_st* rotate2d(TRQT_PI/1.520) * scale(vec2(-0.900,0.900))+ vec2(-0.010,0.040),0.036, .29);
            a += drawSpike(_st * rotate2d(0.836) * scale(vec2(2.290,1.560)) + translate(vec2(0.480,-1.380)));
            a += drawSpike(_st * rotate2d(2.364) * scale(vec2(2.000,1.560)) + translate(vec2(0.380,-1.380)));
            a += drawSpike(_st * rotate2d(2.364) * scale(vec2(2.000,1.560)) + translate(vec2(0.580,-1.380)));
            a += drawSpike(_st * rotate2d(2.364+PI) * scale(vec2(2.000,1.560)) + translate(vec2(0.650,-1.380)));
            a += drawSpike(_st * rotate2d(2.364+PI) * scale(vec2(2.000,1.560)) + translate(vec2(0.450,-1.380)));
            a += drawCircle(_st-vec2(-0.470,-0.50),0.006);
        }else if(r <= 23./24.){//Duralumin
            _st *= rotate2d(3.304);
        	a += drawMoon(_st* rotate2d(TRQT_PI/1.520) * scale(vec2(-0.900,0.900))+ vec2(-0.010,0.040),0.076, 0.402);
            a += drawSpike(_st * rotate2d(2.024) * scale(vec2(0.990,1.460)) + translate(vec2(0.770,-0.6040)));
            a += drawSpike(_st * rotate2d(2.054) * scale(vec2(1.890,1.570)) + translate(vec2(0.640,-0.210)));
            a += drawCircle(_st-vec2(-0.350,-0.470),0.006);
        }else{ // 24.  Brass
	        a += drawMoon(_st* rotate2d(-1.792) * scale(vec2(-0.900,0.900))+ vec2(-0.010,0.040),0.044, 0.282);
            _st = _st * rotate2d(-2.584) * 0.98+vec2(0.030,0.080); 
            a += drawSpike(_st * rotate2d(-2.524) * scale(vec2(1.790,1.560)) + translate(vec2(0.530,-1.120)));
            a += drawSpike(_st * rotate2d(1.644) * scale(vec2(1.690,1.560)) + translate(vec2(0.490,-1.090)));
			a += drawSpike(_st * rotate2d(-1.500) * scale(vec2(1.990,1.660)) + translate(vec2(0.500,-1.120)));
            a += drawSpike(_st * rotate2d(-0.404) * scale(vec2(1.990,1.660)) + translate(vec2(0.500,-1.120)));
            a += drawCircle(_st-vec2(-0.490,-0.500),0.006);
        }
    }

      
    return 1.-a;
}

// Function 25
float GetLoadingText(vec2 vPixelPos)
{     
        vec2 vCharCoord = floor(vPixelPos / 8.0);
       
        float fChar = GetLoadingStringChar(vCharCoord.x);
       
        float inString = 1.0;
        if(vCharCoord.x < 0.0)
                fChar = 32.0;
       
        if(vCharCoord.y != 0.0)
                fChar = 32.0;
       
        return GetCharPixel(fChar, mod(vPixelPos, 8.0));
}

// Function 26
vec4 texture(sampler2D sampler, vec2 uv)
{
    return texture(sampler, uv);
}

// Function 27
mat3 glyph_8_9_numerics(float g) {
    GLYPH(32) 0);
    GLYPH(42)1000,101010,11100,111110,11100,101010,1000,0,0);
    GLYPH(46)0,0,0,0,0,11000,11000,0,0);
    // numerics  ==================================================
    GLYPH(49)11000,1000,1000,1000,1000,1000,11100,0,0);
    GLYPH(50)111100,1000010,10,1100,110000,1000000,1111110,0,0);
    GLYPH(51)111100,1000010,10,11100,10,1000010,111100,0,0);
    GLYPH(52)1000100,1000100,1000100,111110,100,100,100,0,0);
    GLYPH(53)1111110,1000000,1111000,100,10,1000100,111000,0,0);
    GLYPH(54)111100,1000010,1000000,1011100,1100010,1000010,111100,0,0);
    GLYPH(55)111110,1000010,10,100,100,1000,1000,0,0);
    GLYPH(56)111100,1000010,1000010,111100,1000010,1000010,111100,0,0);
    GLYPH(57)111100,1000010,1000010,111110,10,10,111100,0,0);
    GLYPH(58)111100,100100,1001010,1010010,1010010,100100,111100,0,0);
    return mat3(0);
}

// Function 28
vec4 drawTextureValues(in vec2 uv, in float mode, in int menu, in vec2 tvals)
{
    vec4 tcol;
    float lsize = 0.015;
    vec2 start = vec2(0.016, 0.18);
    
    int mval = getIValue(tvals.x, TEXTURES_NUM);
    int sval = getIValue(tvals.y, TEXTURES_NUM);
    
    if (in_zone(mode, APPL_UI) && menu == MENU_OPT_TEXTURE)
    {
        tcol += drawTextHorizontal(uv, start, lsize, vec2[10](_M, _a, _i, _n, _COLON, _X, _X, _X, _X, _X), 5);
        tcol += drawTextHorizontal(uv, start - vec2(0., 0.03), lsize, _SECONDARY, 10);
        
        vec2 mstart = start + 6.*vec2(lsize, 0.);
        vec2 ststart = start + 6.*vec2(lsize, 0.) - vec2(0., 0.03);
        
        if (mval == DIFFUSE_MAP)
        {
            tcol += drawTextHorizontal(uv, mstart, lsize, _DIFFUSEMAP, 9);
        }
        else if (mval == HEIGHT_MAP)
        {
            tcol += drawTextHorizontal(uv, mstart, lsize, _HEIGHTMAP, 10);
        }
        else if (mval == SPECULAR_MAP)
        {
            tcol += drawTextHorizontal(uv, mstart, lsize, _SPECULAR, 8);
        }
        
        if (sval == DIFFUSE_MAP)
        {
            tcol += drawTextHorizontal(uv, ststart, lsize, _DIFFUSEMAP, 9);
        }
        else if (sval == HEIGHT_MAP)
        {
            tcol += drawTextHorizontal(uv, ststart, lsize, _HEIGHTMAP, 10);
        }
        else if (sval == SPECULAR_MAP)
        {
            tcol += drawTextHorizontal(uv, ststart, lsize, _SPECULAR, 8);
        }
    }
    
    return tcol;
}

// Function 29
void TextureEnvBlured2(in vec3 N, in vec3 Rv, out vec3 iblDiffuse, out vec3 iblSpecular) {
    iblDiffuse = vec3(0.0);
    iblSpecular = vec3(0.0);
	
    mat3 shR, shG, shB;
    
    CubeMapToSH2(shR, shG, shB);
    
    #if 1
    	shR = shDiffuseConvolution(shR);
    	shG = shDiffuseConvolution(shG);
    	shB = shDiffuseConvolution(shB);
    #endif
    
    #if 0
    	shR = shDiffuseConvolutionPI(shR);
    	shG = shDiffuseConvolutionPI(shG);
    	shB = shDiffuseConvolutionPI(shB);
    #endif    
    
    iblDiffuse = SH2toColor(shR, shG, shB, N);
}

// Function 30
vec3 textureAVG(samplerCube tex, vec3 tc) {
    const float diff0 = 0.35;
    const float diff1 = 0.12;
 	vec3 s0 = texture(tex,tc).xyz;
    vec3 s1 = texture(tex,tc+vec3(diff0)).xyz;
    vec3 s2 = texture(tex,tc+vec3(-diff0)).xyz;
    vec3 s3 = texture(tex,tc+vec3(-diff0,diff0,-diff0)).xyz;
    vec3 s4 = texture(tex,tc+vec3(diff0,-diff0,diff0)).xyz;
    
    vec3 s5 = texture(tex,tc+vec3(diff1)).xyz;
    vec3 s6 = texture(tex,tc+vec3(-diff1)).xyz;
    vec3 s7 = texture(tex,tc+vec3(-diff1,diff1,-diff1)).xyz;
    vec3 s8 = texture(tex,tc+vec3(diff1,-diff1,diff1)).xyz;
    
    return (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) * 0.111111111;
}

// Function 31
vec3 TriplannarMoonTexture(vec3 p, vec3 normal)
{
    // the scale of the texture
    float scale = 0.25;
    // the sharpness of the blending between different axises
    float blendSharpness = 1.;
    // finding the different axise's color
    vec3 colX = texture(iChannel3, p.zy * scale).rgb;
    vec3 colY = texture(iChannel3, p.xz * scale).rgb;
    vec3 colZ = texture(iChannel3, p.xy * scale).rgb;
    
    // finding the blending amount for each axis
    vec3 bw = pow(abs(normal), vec3(blendSharpness));
    // making it so the total (x + y + z) is 1
    bw /= dot(bw, vec3(1.));
    
    // finding the final color
    return colX * bw.x + colY * bw.y + colZ * bw.z;
}

// Function 32
vec2 textureCoordinates(in vec3 position, in float ringRadius) {
  vec2 q = vec2(length(position.xz) - ringRadius, position.y);
  float u = (atan(position.x, position.z) + pi) / (2.0 * pi);
  float v = (atan(q.x, q.y) + pi) / (2.0 * pi);
  return vec2(u, v);
}

// Function 33
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

// Function 34
float Font_DecodeBitmap( vec2 vCoord, ivec3 vCharacter )
{
    vCoord = floor( vCoord );

    int iRow = int(vCoord.y) - 1;
    int iCol = int(vCoord.x) - 1;
    
    if ( iRow < 0 || iRow >= 6 ) return 0.0;
    if ( iCol < 0 || iCol >= 7 ) return 0.0;
    
    int iRowBits = 0;
        
   	if ( iRow == 0 ) 			iRowBits = vCharacter.x;
    else  if ( iRow == 1 ) 		iRowBits = vCharacter.x / 128;
    else  if ( iRow == 2 ) 		iRowBits = vCharacter.x / 16384;
    else  if ( iRow == 3 ) 		iRowBits = vCharacter.y;
    else  if ( iRow == 4 ) 		iRowBits = vCharacter.y / 128;
    else 						iRowBits = vCharacter.y / 16384;
      
    return (iRowBits & (1 << iCol )) == 0 ? 0.0 : 1.0;
}

// Function 35
vec4 put_text_step(vec4 col, vec2 uv, vec2 pos, float scale)
{
	float unit = asp * scale * 0.1;
    float h = 0.;
    vec2 sc = vec2(unit, unit*0.8);
    
    // S
    h = max(h, word_map(uv, pos, 83, sc));
    // t
    h = max(h, word_map(uv, pos+vec2(unit*0.35, 0.), 116, sc));
    // e
    h = max(h, word_map(uv, pos+vec2(unit*0.7, 0.), 101, sc));
    // p
    h = max(h, word_map(uv, pos+vec2(unit*1.05, 0.), 112, sc));
    
    col = mix(col, vec4(1.-vec3(h), 1.), h);
    
    return col;
}

// Function 36
float escherTextureContour(vec2 p, float linewidth, float pixel_size)
{
    vec2 pp = mod(p,1.0);
    
    float d = 10000000.0;
    for(int i=0; i<vert.length(); ++i)
    {       
        for(int j=0; j<textureTiles.length(); ++j)
        {
            d = min(d, PointSegDistance2(pp+textureTiles[j], vert[i], vert[(i+1)%vert.length()]));
        }
    }
    
    d = smoothstep(0.0, 1.0, (sqrt(d)-linewidth)/pixel_size);
    
    return d;
}

// Function 37
vec4 SampleFontTex(vec2 uv,float xoff)
{
    uv = uv.yx;
    uv.x = -uv.x;
//    uv += 0.04;
    uv.x += 0.04 + (0.25 * xoff);
    uv *= 0.25;
    vec2 fl = floor(uv + 0.5);
    uv = fl + fract(uv+0.5)-0.5;
    
    // Sample the font texture. Make sure to not use mipmaps.
    // Add a small amount to the distance field to prevent a strange bug on some gpus. Slightly mysterious. :(
    //return texture(iChannel0, (uv+0.5)*((1.0/16.0)*8.0), -100.0) + vec4(0.0, 0.0, 0.0, 0.00001);
    return texture(iChannel0, uv, -100.0) + vec4(0.0, 0.0, 0.0, 0.00001);
}

// Function 38
float font(Data f){
    float c = 0.;
    c = arrayBin(f);
    uv.x -= 3.+ float(adjacency_width);
    return c;
}

// Function 39
bool texture_store(in sampler2D s)
{
    return texelFetch(s, CTRL_TEXTURE, 0).y < 0.5;
}

// Function 40
float checkersTextureGradTri(in vec3 p, in vec3 ddx, in vec3 ddy) {
    vec3 w = max(abs(ddx), abs(ddy)) + 0.01;       // filter kernel
    vec3 i = (pri(p+w)-2.0*pri(p)+pri(p-w))/(w*w); // analytical integral (box filter)
    return 0.1 - 0.5*i.x*i.y*i.z;                  // xor pattern
}

// Function 41
vec4 getTexture(Material mat, vec3 p,  vec3 normal)
{
  float alpha=1.0;
  int texIndex=mat.texture;
  float squareSize=boardSize/2.0;

  // Checkerboard texture for the chess board.
  if(texIndex==1)
  {

    if(normal.y<0.5) mat.color;
    vec2 fmod=vec2(p.x+boardSize,p.z+boardSize)/squareSize;
    float floorx=fmod.x-float(int(fmod.x)); 
    float floory=fmod.y-float(int(fmod.y)); 
    
    if(floorx>0.5^^floory>0.5) 
    	alpha=1.0;
      else
        alpha=0.0;
   
  }
  // Starry texture for the background
  else if(texIndex==2)
    {
     
      float threshold=0.997;
      float starBrightness=rnd(p.xy);	
      if(rnd(p.xy)>threshold) alpha=(starBrightness-threshold)/(1.0-threshold);
      else alpha=0.0;

    }
  // Interpolation between the foreground and background color
 
  return alpha*mat.color+(1.0-alpha)*mat.bgColor;
  
}

// Function 42
function measureText(text) {
	var length = 0;
	for (var i = 0; i < text.length; i++) {
		if (text[i] == " ") {
			length += 3;
		} else {
			if (text[i] in widths) {
				length += widths[text[i]];
				if (i != text.length - 1) {
					length += 2;
				}
			} else {
				console.error("Bad character '" + text[i] + "' at index " + i + " in string \"" + text + "\"");
			}
		}
	}
	return length;
}

// Function 43
float FontTexDf (vec2 p, int ic)
{
  vec3 tx;
  float d;
  tx = texture (txFnt, mod ((vec2 (mod (float (ic), 16.),
     15. - floor (float (ic) / 16.)) + p) * (1. / 16.), 1.)).gba - 0.5;
  qnTex = vec2 (tx.r, - tx.g);
  d = tx.b + 1. / 256.;
  return d;
}

// Function 44
float titleText( vec2 p )
{        
    vec2 scale = vec2( 4., 8. );
    vec2 t = floor( p / scale );   
    
    uint v = 0u;
	v = t.y == 0. ? ( t.x < 4. ? 1397642579u : ( t.x < 8. ? 1142969413u : ( t.x < 12. ? 1163282770u : ( t.x < 16. ? 1280202016u : ( t.x < 20. ? 1414090057u : 17477u ) ) ) ) ) : v;
	v = t.x >= 0. && t.x < 24. ? v : 0u;
    
	float c = float( ( v >> uint( 8. * t.x ) ) & 255u );
    
    p = ( p - t * scale ) / scale;
    p.x = ( p.x - .5 ) * .5 + .5;
    float sdf = textSDF( p, c );
    return ( c != 0. ) ? smoothstep( -.05, +.05, sdf ) : 1.0;
}

// Function 45
vec4 textured(vec2 pos){
    
	ivec2 P = ivec2(pos) / 1 % 1024;
    return texelFetch(iChannel0, P, 0);
}

// Function 46
vec3 flagTexture2(vec2 p)
{
    p.x+=0.05;
    p.y+=0.05;
    float d=1e2;
    d=min(d,length(p-vec2(-0.85,-0.13+p.x*-0.4))-0.5);
    d=min(d,length(p-vec2(-0.8,-1.2))-0.9);
    d=min(d,length(max(vec2(0.0),abs(p-vec2(0.4,-0.63))-vec2(0.5,0.3))));
    d=min(d,length(max(vec2(0.0),abs(p-vec2(0.9,0.0))-vec2(0.3,0.8))));
    d=min(d,length(max(vec2(0.0),abs(p-vec2(-0.9,0.0))-vec2(0.3,0.8))));
    vec2 p2=p+vec2(0.0,-0.07);
    d=min(d,length((p2-vec2(0.1,0.2))*vec2(1.0,0.9))+0.1-p2.y*1.3);
    d=max(d,-length(p-vec2(0.1,0.5))+0.15);
    vec2 p3=p+vec2(0.0,0.1);
    d=min(d,max(p.x, max(-(length((p3-vec2(0.1,0.2))*vec2(1.0,0.9))+0.1-p3.y*1.1), dot(p-vec2(0.0,0.12),normalize(vec2(-0.75,-1.0))))));
    return mix(vec3(0.2,0.1,0.1),vec3(1.0,0.2,0.1)*0.9,(0.3+(1.0-smoothstep(0.0,0.02,d))))*
        mix(0.7,1.0,smoothNoise2(p*40.0)*0.25+smoothNoise2(p*20.0)*0.25);
}

// Function 47
float text( vec2 p ) 
{
    // trick for encoding fonts from CPU
	p.x += 0.2*floor(10.0*(0.5+0.5*sin(iTime)))/10.0;
	
	float x = floor( p.x*100.0 ) - 23.0;
	float y = floor( p.y*100.0 ) - 82.0;

	if( y<0.0 || y> 5.0) return 0.0;
	if( x<0.0 || x>70.0) return 0.0;
	
    float v = 0.0;
	
         if( x>63.5 ) {           v=12288.0;
	                    if(y>2.5) v=30720.0;
	                    if(y>3.5) v=52224.0; }
	else if( x>47.5 ) {           v=12408.0;
	                    if(y>0.5) v=12492.0;
	                    if(y>4.5) v=64632.0; }
	else if( x>31.5 ) {           v=64716.0;
	                    if(y>0.5) v=49360.0;
	                    if(y>1.5) v=49400.0;
	                    if(y>2.5) v=63692.0;
	                    if(y>3.5) v=49356.0;
	                    if(y>4.5) v=64760.0; }
	else if( x>15.5 ) {           v=40184.0;
	                    if(y>0.5) v=40092.0;
	                    if(y>2.5) v=64668.0;
	                    if(y>3.5) v=40092.0;
	                    if(y>4.5) v=28920.0; }
	else	          {           v=30860.0;
    	                if(y>0.5) v=40076.0;
    	                if(y>1.5) v= 7308.0;
    	                if(y>2.5) v=30972.0;
    	                if(y>3.5) v=49292.0;
    	                if(y>4.5) v=30860.0; }
		
	return floor( mod(v/pow(2.0,15.0-mod( x, 16.0 )), 2.0) );
}

// Function 48
vec4 DrawText( sampler2D sampler,float textSize, vec2 uv, vec2 pos, int idx, vec4 color, float flatText)
{
    float halfTextSize = textSize/2.0;
    vec3 mask = BoxMask(uv, pos, vec2(halfTextSize/16.,halfTextSize/16.));
    int x = idx % 16;
    int y = idx / 16;

    vec4 txtValue = texture(sampler,mask.xy/32.+vec2(float(x),float(y))/16.);
    float d = txtValue.w;
    d = smoothstep(0.49,0.5,1.-d);
    
    return vec4(d*color);
}

// Function 49
float sampleCubeTexture(vec2 uv)
{
    uv += 0.3;
    uv /= 0.6;
    return cubeTex[int(uv.x*8.0) + ( int(uv.y*8.0) * 8 ) ];
}

// Function 50
vec3 SampleInterpolationTextureBilinear (vec2 uv)
{
    vec2 pixel = uv * 2.0 - 0.5;
    vec2 pixelFract = fract(pixel);
    
    vec3 pixel00 = SampleInterpolationTexturePixel(floor(pixel) + vec2(0.0, 0.0));
    vec3 pixel10 = SampleInterpolationTexturePixel(floor(pixel) + vec2(1.0, 0.0));
    vec3 pixel01 = SampleInterpolationTexturePixel(floor(pixel) + vec2(0.0, 1.0));
    vec3 pixel11 = SampleInterpolationTexturePixel(floor(pixel) + vec2(1.0, 1.0));
    
    vec3 row0 = mix(pixel00, pixel10, pixelFract.x);
    vec3 row1 = mix(pixel01, pixel11, pixelFract.x);
    
    return mix(row0, row1, pixelFract.y);
}

// Function 51
float text_end(vec2 U) {
    initMsg;C(69);C(110);C(100);endMsg;
}

// Function 52
vec4 textureWall(vec2 uv) {
    const vec2 RES = vec2(32.0, 16.0);
    vec2 iuv = floor(uv * RES);    
    float n = noise1(uv * RES);
    n = n * 0.5 + 0.25;
    float nc = n * (smoothstep(1.0,0.4, iuv.x / RES.x) * 0.5 + 0.5);    
    return vec4(nc * 0.4, nc * 1.0, nc * 0.5, n + uv.x - abs(uv.y-0.5) );
}

// Function 53
float checkersTextureGrad( in vec3 p, in vec3 ddx, in vec3 ddy )
{
  vec3 w = max(abs(ddx), abs(ddy)) + 0.0001; // filter kernel
  vec3 i = (tri(p+0.5*w)-tri(p-0.5*w))/w;    // analytical integral (box filter)
  return 0.5 - 0.5*i.x*i.y*i.z;              // xor pattern
}

// Function 54
float floorTexture(in vec2 p) {
  vec2 fp = mod(p, 2.0);
  bvec2 f = greaterThan(fp, vec2(1.0));
  return float(f.x ^^ f.y);
}

// Function 55
float getTextureForPoint(vec3 p, int type){
	float res;
    if(type == PERLIN_WORLEY){
        
        //Perlin-Worley.
        const float frequency = 8.0;
        float perlinNoise = getPerlinNoise(p, frequency);
        res = perlinNoise;

        //Special weights from example code.
        float worley0 = worley(p, NUM_CELLS*2.0);
        float worley1 = worley(p, NUM_CELLS*8.0);
        float worley2 = worley(p, NUM_CELLS*14.0);

        float worleyFBM = worley0 * 0.625 + worley1 * 0.25 + worley2 * 0.125;
        res = remap(perlinNoise, 0.0, 1.0, worleyFBM, 1.0);
        
	}else{

        //Worley
        float worley0 = worley(p, NUM_CELLS);
        float worley1 = worley(p, NUM_CELLS*2.0);
        float worley2 = worley(p, NUM_CELLS*4.0);
        float worley3 = worley(p, NUM_CELLS*8.0);

        float FBM0 = worley0 * 0.625 + worley1 * 0.25 + worley2 * 0.125;
		float FBM1 = worley1 * 0.625 + worley2 * 0.25 + worley3 * 0.125;
		float FBM2 = worley2 * 0.75 + worley3 * 0.25;

        res = FBM0 * 0.625 + FBM1 * 0.25 + FBM2 * 0.125;
	}
    
	return res;
}

// Function 56
void SliderText(inout vec3 color, vec2 p, in AppState s)
{
    p -= vec2(67, 76);
    
    vec2 scale = vec2(4., 8.);
    vec2 t = floor(p / scale);   
    
    uint v = 0u;
	v = t.y == 0. ? (t.x < 4. ? 1735749458u : (t.x < 8. ? 1936027240u : 14963u)) : v;
	v = t.x >= 0. && t.x < 12. ? v : 0u;
    
	float c = float((v >> uint(8. * t.x)) & 255u);
    
    vec3 textColor = vec3(.3);

    p = (p - t * scale) / scale;
    p.x = (p.x - .5) * .5 + .5;
    float sdf = TextSDF(p, c);
    if (c != 0.)
    {
    	color = mix(textColor, color, smoothstep(-.05, +.05, sdf));
    } 
}

// Function 57
void SetTextPosition(vec2 p,float x,float y){//x=line, y=column
 tp=10.0*p;tp.x=tp.x+17.-x;tp.y=tp.y-9.4+y;}

// Function 58
vec2 UIDrawContext_ScreenPosToCanvasPos( UIDrawContext drawContext, vec2 vScreenPos )
{
    vec2 vViewPos = vScreenPos - drawContext.viewport.vPos;
    return vViewPos + drawContext.vOffset;
}

// Function 59
vec3 textureWall(vec2 pos, vec2 maxPos, vec2 squarer,float s,float height,float dist,vec3 d,vec3 norm
){float randB=rand(squarer*2.0)
 ;vec3 windowColor=(-0.4+randB*0.8)*vec3(0.3,0.3,0.0)
 +(-0.4+fract(randB*10.0)*0.8)*vec3(0.0,0.0,0.3)+(-0.4+fract(randB*10000.)*0.8)*vec3(0.3,0.0,0.0)
 ;float floorFactor=1.
 ;vec2 windowSize=vec2(0.65,0.35)
 ;vec3 wallColor=s*(0.3+1.4*fract(randB*100.))*vec3(0.1,0.1,0.1)
 +(-0.7+1.4*fract(randB*1000.))*vec3(0.02,0.,0.)
 ;wallColor*=1.3
 ;vec3 color=vec3(0)
 ;vec3 conturColor=wallColor/1.5
 ;if (height<0.51
 ){windowColor += vec3(.3,.3,.0)
  ;windowSize=vec2(.4)
  ;floorFactor=0.;}
 ;if (height<.6){floorFactor=0.;}
 ;if (height>.75)windowColor += vec3(0,0,.3)
 ;windowColor*=1.5
 ;float wsize=0.02
 ;wsize+=-0.007+0.014*fract(randB*75389.9365)
 ;windowSize+= vec2(0.34*fract(randB*45696.9365),0.50*fract(randB*853993.5783))
 ;windowSize/=2.
 ;vec2 contur=vec2(0.0)+(fract(maxPos/2.0/wsize))*wsize
  ;vec2 pc=pos-contur
 ;if (contur.x<wsize)contur.x+=wsize
 ;if (contur.y<wsize)contur.y+=wsize
 ;vec2 winPos=(pc)/wsize/2.0-floor((pc)/wsize/2.0)
 ;float numWin=floor((maxPos-contur)/wsize/2.0).x
 ;vec3 n=floor(numWin*vec3(1,2,3)/4.)
 ;vec2 m=numWin*vec2(1,2)/3.
 ;float w=wsize*2.
 ;bvec3 bo=bvec3(isOutOpen(pc.x  ,w*n.y,w+w*n.y)||isOutOpen(maxPos.x,.5,.6)
                ,isOutOpen(pc.xx ,w*m  ,w+w*m  )||isOutOpen(maxPos.x,.6,.7)
                ,isOutOpen(pc.xxx,w*n  ,w+w*n  )||maxPos.x>.7)
 ;bnot(bo)
 ;if(any(bo))return (.9+.2*noise(pos))*conturColor 
 ;if((maxPos.x-pos.x<contur.x)||(maxPos.y-pos.y<contur.y+w)||(pos.x<contur.x)||(pos.y<contur.y))
            return (0.9+0.2*noise(pos))*conturColor
 ;if (maxPos.x<0.14)return (0.9+0.2*noise(pos))*wallColor
 ;vec2 window=floor(pc/w)
 ;float random=rand(squarer*s*maxPos.y+window)
 ;float randomZ=rand(squarer*s*maxPos.y+floor(pc.yy/w))
 ;float windows=floorFactor*sin(randomZ*5342.475379+(fract(975.568*randomZ)*0.15+0.05)*window.x)
 ;float blH=0.06*dist*600./iResolution.x/abs(dot(normalize(d.xy),normalize(norm.xy)))
 ;float blV=0.06*dist*600./iResolution.x/sqrt(abs(1.0-pow(abs(d.z),2.0)))
 ;windowColor +=vec3(1.0,1.0,1.0)
 ;windowColor*=smoothstep(.5-windowSize.x-blH,.5-windowSize.x+blH,winPos.x)
 ;windowColor*=smoothstep(.5+windowSize.x+blH,.5+windowSize.x-blH,winPos.x)
 ;windowColor*=smoothstep(.5-windowSize.y-blV,.5-windowSize.y+blV,winPos.y)
 ;windowColor*=smoothstep(.5+windowSize.y+blV,.5+windowSize.y-blV,winPos.y)
 ;if ((random <0.05*(3.5-2.5*floorFactor))||(windows>0.65)
 ){if (winPos.y<0.5)windowColor*=(1.0-0.4*fract(random*100.))
  ;if ((winPos.y>0.5)&&(winPos.x<0.5))windowColor*=(1.0-0.4*fract(random*10.0))
  ;return (.9+.2*noise(pos))*wallColor+(0.9+0.2*noise(pos))*windowColor
 ;} else windowColor*=0.08*fract(10.0*random)
 ;return (.9+.2*noise(pos))*wallColor+windowColor;}

// Function 60
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

// Function 61
vec3 GetBoatTexture(RayHit marchResult)
{
  vec3 checkPos = TranslateBoat(marchResult.hitPos); 
  vec3 bCol= vec3(62, 52, 47)*1.3/255.;
  float frontDist = max(0., (0.25*(0.16*pow(length(checkPos.z-0.), 2.))));
  float n = 1.+(0.2*noise(vec3(checkPos.zx*0.01, checkPos.x)*34.));
  n *= 0.9+(0.1*noise2D(checkPos.xy*26.));  
  bCol = mix(vec3(0.6), bCol*n, step(-0.625, checkPos.y-frontDist));
  bCol = mix(vec3(0.05), bCol, step(0.08, length(-.7-(checkPos.y-frontDist))));
  bCol = mix(bCol*0.8, bCol*1.2, smoothstep(0., 0.18, length(-0.23-(checkPos.y-frontDist))));   
  bCol = mix(bCol, bCol*0.47, smoothstep(0.0, 0.32, length(0.-mod(checkPos.y-frontDist, 0.3)))); 
  return mix(bCol, bCol*0.8, smoothstep(-.1, 0.8, noise2D(checkPos.xz*3.7)));  
}

// Function 62
void UIStyle_GetFontStyleWindowText( inout LayoutStyle style, inout RenderStyle renderStyle )
{
    style = LayoutStyle_Default();
	renderStyle = RenderStyle_Default( vec3(0.0) );
}

// Function 63
float SampleTexture(in vec2 uv)
{
    return texture(iChannel0, uv).r;
}

// Function 64
bool TeletextState_GetSeparatedGfx( TeletextState state )
{
    if ( IsControlCharacter( state.char ) )
    {
        if ( state.bHoldGfx )
        {
            return state.bHeldSeparated;
        }
        else
        {
            return false;
        }
    }    
    
    return state.bSeparatedGfx;
}

// Function 65
vec3 drawText2( in vec4 fragColor, in vec2 fragCoord ) {
    float display_width = 1010.;
    float cc = floor(display_width / (g_cw * (1. + g_cwb))); // character count per line
    
    vec2 uv = (fragCoord.xy) / iResolution.xx;
    uv.y = iResolution.y/iResolution.x - uv.y;  // type from top to bottom, left to right   
    uv *= display_width;

    int cs = int(floor(uv.x / (g_cw * (1. + g_cwb))) + cc * floor(uv.y/(g_ch * (1. + g_chb))));

    uv = mod_uv(uv);
    uv.y = g_ch * (1. + g_chb) - uv.y; // paint the character from the bottom left corner
    vec3 ccol = .35 * vec3(.1, .3, .2) * max(smoothstep(3., 0., uv.x), smoothstep(5., 0., uv.y));   
    uv -= vec2(g_cw * g_cwb * .5, g_ch * g_chb * .5);
    
    float tx = 10000.;
    int idx = 0;
    
    NL 
    NL 
    NL 
    NL 
    NL 
    NL 
    SP SP SP SP SP SP SP SP SP SP SP P R E S S SP E N T E R SP T O SP S T A R T
    NL
    vec3 tcol = vec3(1.0, 0.7, 0.0) * smoothstep(.2, .0, tx);
    
    vec3 terminal_color = tcol;
    
    return terminal_color;
}

// Function 66
float text(vec2 fragCoord)
{
    vec2 uv = mod(fragCoord.xy, 16.)*.0625;
    vec2 block = fragCoord*.0625 - uv;
    uv = uv*.8+.1; // scale the letters up a bit
    uv += floor(texture(iChannel1, block/iChannelResolution[1].xy + iTime * .002).xy * 16.); // randomize letters
    uv *= .0625; // bring back into 0-1 range
    uv.x = -uv.x; // flip letters horizontally
    return texture(iChannel0, uv).r;
}

// Function 67
vec4 textureAspect(sampler2D tex, vec3 channelResolution, vec3 iResolution, vec2 fragCoord)
{
    vec2 U = fragCoord;
    vec2 margin = vec2(0),
         Sres = iResolution.xy -2.*margin,
         Tres = channelResolution.xy,
         ratio = Sres/Tres;
    
    U -= margin;
    
    // centering the blank part in case of rectangle fit
    U -= .5*Tres*max(vec2(ratio.x-ratio.y,ratio.y-ratio.x),0.);
    
  //U /= Tres*ratio.y;               // fit height, keep ratio
  //U /= Tres*ratio.x;               // fit width, keep ratio
    U /= Tres*min(ratio.x,ratio.y);  // fit rectangle,  keep ratio
    U *= 1.;                         // zoom out factor 
    
	vec4 result = fract(U)==U 
        ? texture(tex, U)
        : vec4(0.0);
        
        return result;
}

// Function 68
float text_f(vec2 U) {
    initMsg;
    U.x+=4.*(0.5-0.2812*(res.x/0.5));
    C(70);C(111);C(114);C(99);C(101);
    endMsg;
}

// Function 69
vec3 processtexture( vec4 t ) {
    return vec3( (t.r+t.g+t.b+t.a)/4.0 );
    //return t.r*vec3(1.0, 0.5, 0.2) + t.g*vec3(0.4, 0.6, 0.2) + t.b*vec3(0.4, 0.3, 0.6) + t.a*vec3(0.3, 0.4, 0.2);
}

// Function 70
vec3 CreateTexture(in RayHit hit)
{
    // Create toned down diffuse textures to account for later gamma correction
    vec3 grass = GrassColor - (texture(iChannel0, hit.surfPos.xz * 0.1).r * 0.3);
    vec3 cliff = CliffColor - (abs(sin(texture(iChannel1, hit.surfPos.xy * 0.01).r)) * 0.5);
    
    vec3 color = mix(cliff, grass, SteepnessRatio(hit.steepness)) * 0.2;
    
    return TimeLerp(vec3(0.2), color, TIME_SteepnessB, TIME_Texture);
}

// Function 71
vec4 waterTexture(in vec3 rp)
{
    rp=planetRotatedVec(rp);
    float T = iTime*.01;
    float S=2.0;
    rp *= rotx(T);
    float c1=textureSpherical(iChannel1, rp, S).x;
    rp *= rotx(1.+T);
    float c2=textureSpherical(iChannel1, rp, S).x;
    rp *= rotx(2.+T);
    float c3=textureSpherical(iChannel1, rp, S).x;
    float B = iTime*2.;
    float col = mix(c3, mix(c1, c2, 0.5+0.5*(sin(PI*0.5+B)*0.5+0.5)), 0.5+0.5*(sin(B)*0.5+0.5));
    return vec4(col);
    
}

// Function 72
vec3 mainObjectTexture(in vec3 rp) {
    return texture(iChannel1,(abs(rp.xy)+abs(rp.zz))*.04).xyz;
}

// Function 73
float gridTexture( in vec2 p )
{
    // coordinates
    vec2 i = step( fract(p), vec2(1.0/N) );
    //pattern
    return (1.0-i.x)*(1.0-i.y);   // grid (N=10)
    
    // other possible patterns are these
    //return 1.0-i.x*i.y;           // squares (N=4)
    //return 1.0-i.x-i.y+2.0*i.x*i.y; // checker (N=2)
}

// Function 74
float NumFont_Zero( vec2 vTexCoord )
{
    float fResult = NumFont_Circle( vTexCoord );

    float fHole = NumFont_Rect( vTexCoord, vec2(6, 4), vec2(6,10) );
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(5, 5), vec2(7,9) ) );

    fResult = min( fResult, 1.0 - fHole );    

    return fResult;
}

// Function 75
void setup_font(){

    CHARS[0] = ivec4(0x00000000, 0x00000000, 0x00000000, 0x00000000); //  0x1e 30
    CHARS[1] = ivec4(0x00000000, 0x00000000, 0x00000000, 0x00000000); //  0x1f 31
    CHARS[2] = ivec4(0x00000000, 0x00000000, 0x00000000, 0x00000000); //   0x20 32
    CHARS[3] = ivec4(0x00000000, 0x00001818, 0x00181818, 0x3c3c3c18); // ! 0x21 33
    CHARS[4] = ivec4(0x00000000, 0x00000000, 0x00000000, 0x00444466); // " 0x22 34
    CHARS[5] = ivec4(0x00000000, 0x00003636, 0x7f36367f, 0x36360000); // # 0x23 35
    CHARS[6] = ivec4(0x00000000, 0x08083e6b, 0x6b381c0e, 0x6b6b3e08); // $ 0x24 36
    CHARS[7] = ivec4(0x00000000, 0x00003049, 0x4b360c18, 0x36694906); // % 0x25 37
    CHARS[8] = ivec4(0x00000000, 0x00006e33, 0x333b6e0c, 0x1c36361c); // & 0x26 38
    CHARS[9] = ivec4(0x00000000, 0x00000000, 0x00000000, 0x00081018); // ' 0x27 39
    CHARS[10] = ivec4(0x00000000, 0x00003018, 0x0c0c0c0c, 0x0c0c1830); // ( 0x28 40
    CHARS[11] = ivec4(0x00000000, 0x00000c18, 0x30303030, 0x3030180c); // ) 0x29 41
    CHARS[12] = ivec4(0x00000000, 0x00000000, 0x663cff3c, 0x66000000); // * 0x2a 42
    CHARS[13] = ivec4(0x00000000, 0x00000000, 0x18187e18, 0x18000000); // + 0x2b 43
    CHARS[14] = ivec4(0x00000000, 0x04080c0c, 0x00000000, 0x00000000); // , 0x2c 44
    CHARS[15] = ivec4(0x00000000, 0x00000000, 0x00007f00, 0x00000000); // - 0x2d 45
    CHARS[16] = ivec4(0x00000000, 0x00000c0c, 0x00000000, 0x00000000); // . 0x2e 46
    CHARS[17] = ivec4(0x00000000, 0x00000001, 0x03060c18, 0x30604000); // / 0x2f 47
    CHARS[18] = ivec4(0x00000000, 0x00003e63, 0x63676f7b, 0x7363633e); // 0 0x30 48
    CHARS[19] = ivec4(0x00000000, 0x00007e18, 0x18181818, 0x181e1c18); // 1 0x31 49
    CHARS[20] = ivec4(0x00000000, 0x00007f63, 0x03060c18, 0x3060633e); // 2 0x32 50
    CHARS[21] = ivec4(0x00000000, 0x00003e63, 0x6060603c, 0x6060633e); // 3 0x33 51
    CHARS[22] = ivec4(0x00000000, 0x00007830, 0x307f3333, 0x363c3830); // 4 0x34 52
    CHARS[23] = ivec4(0x00000000, 0x00003e63, 0x6060603f, 0x0303037f); // 5 0x35 53
    CHARS[24] = ivec4(0x00000000, 0x00003e63, 0x6363633f, 0x0303633e); // 6 0x36 54
    CHARS[25] = ivec4(0x00000000, 0x00000c0c, 0x0c0c1830, 0x6060637f); // 7 0x37 55
    CHARS[26] = ivec4(0x00000000, 0x00003e63, 0x6363633e, 0x6363633e); // 8 0x38 56
    CHARS[27] = ivec4(0x00000000, 0x00003e63, 0x60607e63, 0x6363633e); // 9 0x39 57
    CHARS[28] = ivec4(0x00000000, 0x00000018, 0x18000000, 0x18180000); // : 0x3a 58
    CHARS[29] = ivec4(0x00000000, 0x00081018, 0x18000000, 0x18180000); // ; 0x3b 59
    CHARS[30] = ivec4(0x00000000, 0x00006030, 0x180c060c, 0x18306000); // < 0x3c 60
    CHARS[31] = ivec4(0x00000000, 0x00000000, 0x007e0000, 0x7e000000); // = 0x3d 61
    CHARS[32] = ivec4(0x00000000, 0x0000060c, 0x18306030, 0x180c0600); // > 0x3e 62
    CHARS[33] = ivec4(0x00000000, 0x00001818, 0x00181830, 0x6063633e); // ? 0x3f 63
    CHARS[34] = ivec4(0x00000000, 0x00003c02, 0x6db5a5a5, 0xb9423c00); // @ 0x40 64
    CHARS[35] = ivec4(0x00000000, 0x00006363, 0x63637f63, 0x6363361c); // A 0x41 65
    CHARS[36] = ivec4(0x00000000, 0x00003f66, 0x6666663e, 0x6666663f); // B 0x42 66
    CHARS[37] = ivec4(0x00000000, 0x00003e63, 0x63030303, 0x0363633e); // C 0x43 67
    CHARS[38] = ivec4(0x00000000, 0x00003f66, 0x66666666, 0x6666663f); // D 0x44 68
    CHARS[39] = ivec4(0x00000000, 0x00007f66, 0x46161e1e, 0x1646667f); // E 0x45 69
    CHARS[40] = ivec4(0x00000000, 0x00000f06, 0x06161e1e, 0x1646667f); // F 0x46 70
    CHARS[41] = ivec4(0x00000000, 0x00007e63, 0x63637303, 0x0363633e); // G 0x47 71
    CHARS[42] = ivec4(0x00000000, 0x00006363, 0x6363637f, 0x63636363); // H 0x48 72
    CHARS[43] = ivec4(0x00000000, 0x00003c18, 0x18181818, 0x1818183c); // I 0x49 73
    CHARS[44] = ivec4(0x00000000, 0x00001e33, 0x33303030, 0x30303078); // J 0x4a 74
    CHARS[45] = ivec4(0x00000000, 0x00006766, 0x66361e1e, 0x36666667); // K 0x4b 75
    CHARS[46] = ivec4(0x00000000, 0x00007f66, 0x46060606, 0x0606060f); // L 0x4c 76
    CHARS[47] = ivec4(0x00000000, 0x00006363, 0x63636b7f, 0x7f776341); // M 0x4d 77
    CHARS[48] = ivec4(0x00000000, 0x00006363, 0x63737b7f, 0x6f676361); // N 0x4e 78
    CHARS[49] = ivec4(0x00000000, 0x00003e63, 0x63636363, 0x6363633e); // O 0x4f 79
    CHARS[50] = ivec4(0x00000000, 0x00000f06, 0x06063e66, 0x6666663f); // P 0x50 80
    CHARS[51] = ivec4(0x00000000, 0x00603e7b, 0x6b636363, 0x6363633e); // Q 0x51 81
    CHARS[52] = ivec4(0x00000000, 0x00006766, 0x66363e66, 0x6666663f); // R 0x52 82
    CHARS[53] = ivec4(0x00000000, 0x00003e63, 0x6360301c, 0x0663633e); // S 0x53 83
    CHARS[54] = ivec4(0x00000000, 0x00003c18, 0x18181818, 0x185a7e7e); // T 0x54 84
    CHARS[55] = ivec4(0x00000000, 0x00003e63, 0x63636363, 0x63636363); // U 0x55 85
    CHARS[56] = ivec4(0x00000000, 0x0000081c, 0x36636363, 0x63636363); // V 0x56 86
    CHARS[57] = ivec4(0x00000000, 0x00004163, 0x777f6b63, 0x63636363); // W 0x57 87
    CHARS[58] = ivec4(0x00000000, 0x00006363, 0x363e1c1c, 0x3e366363); // X 0x58 88
    CHARS[59] = ivec4(0x00000000, 0x00003c18, 0x1818183c, 0x66666666); // Y 0x59 89
    CHARS[60] = ivec4(0x00000000, 0x00007f63, 0x43060c18, 0x3061637f); // Z 0x5a 90
    CHARS[61] = ivec4(0x00000000, 0x00003c0c, 0x0c0c0c0c, 0x0c0c0c3c); // [ 0x5b 91
    CHARS[62] = ivec4(0x00000000, 0x00000040, 0x6030180c, 0x06030100); // \ 0x5c 92
    CHARS[63] = ivec4(0x00000000, 0x00003c30, 0x30303030, 0x3030303c); // ] 0x5d 93
    CHARS[64] = ivec4(0x00000000, 0x00000000, 0x00000000, 0x63361c08); // ^ 0x5e 94
    CHARS[65] = ivec4(0x00000000, 0x00ff0000, 0x00000000, 0x00000000); // _ 0x5f 95
    CHARS[66] = ivec4(0x00000000, 0x00000000, 0x00000000, 0x00100818); // ` 0x60 96
    CHARS[67] = ivec4(0x00000000, 0x00006e33, 0x33333e30, 0x1e000000); // a 0x61 97
    CHARS[68] = ivec4(0x00000000, 0x00003e66, 0x66666666, 0x3e060607); // b 0x62 98
    CHARS[69] = ivec4(0x00000000, 0x00003e63, 0x03030363, 0x3e000000); // c 0x63 99
    CHARS[70] = ivec4(0x00000000, 0x00006e33, 0x33333333, 0x3e303038); // d 0x64 100
    CHARS[71] = ivec4(0x00000000, 0x00003e63, 0x037f6363, 0x3e000000); // e 0x65 101
    CHARS[72] = ivec4(0x00000000, 0x00001e0c, 0x0c0c0c0c, 0x3e0c6c38); // f 0x66 102
    CHARS[73] = ivec4(0x0000001e, 0x33303e33, 0x33333333, 0x6e000000); // g 0x67 103
    CHARS[74] = ivec4(0x00000000, 0x00006766, 0x6666666e, 0x36060607); // h 0x68 104
    CHARS[75] = ivec4(0x00000000, 0x00003c18, 0x18181818, 0x1c001818); // i 0x69 105
    CHARS[76] = ivec4(0x0000001e, 0x33333030, 0x30303030, 0x38003030); // j 0x6a 106
    CHARS[77] = ivec4(0x00000000, 0x00006766, 0x361e1e36, 0x66060607); // k 0x6b 107
    CHARS[78] = ivec4(0x00000000, 0x00003c18, 0x18181818, 0x1818181c); // l 0x6c 108
    CHARS[79] = ivec4(0x00000000, 0x0000636b, 0x6b6b6b7f, 0x37000000); // m 0x6d 109
    CHARS[80] = ivec4(0x00000000, 0x00006666, 0x66666666, 0x3b000000); // n 0x6e 110
    CHARS[81] = ivec4(0x00000000, 0x00003e63, 0x63636363, 0x3e000000); // o 0x6f 111
    CHARS[82] = ivec4(0x0000000f, 0x06063e66, 0x66666666, 0x3b000000); // p 0x70 112
    CHARS[83] = ivec4(0x00000078, 0x30303e33, 0x33333333, 0x3e000000); // q 0x71 113
    CHARS[84] = ivec4(0x00000000, 0x00000f06, 0x0606066e, 0x7b000000); // r 0x72 114
    CHARS[85] = ivec4(0x00000000, 0x00003e63, 0x301c0663, 0x3e000000); // s 0x73 115
    CHARS[86] = ivec4(0x00000000, 0x0000182c, 0x0c0c0c0c, 0x3f0c0c08); // t 0x74 116
    CHARS[87] = ivec4(0x00000000, 0x00006e33, 0x33333333, 0x33000000); // u 0x75 117
    CHARS[88] = ivec4(0x00000000, 0x0000081c, 0x36636363, 0x63000000); // v 0x76 118
    CHARS[89] = ivec4(0x00000000, 0x0000367f, 0x6b6b6b6b, 0x63000000); // w 0x77 119
    CHARS[90] = ivec4(0x00000000, 0x00006363, 0x361c3663, 0x63000000); // x 0x78 120
    CHARS[91] = ivec4(0x0000001f, 0x30607e63, 0x63636363, 0x63000000); // y 0x79 121
    CHARS[92] = ivec4(0x00000000, 0x00007f43, 0x060c1831, 0x7f000000); // z 0x7a 122
    CHARS[93] = ivec4(0x00000000, 0x00007018, 0x1818180e, 0x18181870); // { 0x7b 123
    CHARS[94] = ivec4(0x00000000, 0x00001818, 0x18180000, 0x18181818); // | 0x7c 124
    CHARS[95] = ivec4(0x00000000, 0x00000e18, 0x18181870, 0x1818180e); // } 0x7d 125
    CHARS[96] = ivec4(0x00000000, 0x00000000, 0x00000000, 0x00003b6e); // ~ 0x7e 126
}

// Function 76
void WriteText1()
{
  SetTextPosition(1.,1.);
  float c = 0.0;
  _star _ _V _i _e _w _ _S _h _a _d _e _r 
  
  _ _D _a _t _a _ _2 _ _ _v _1 _dot _4 _ _star 
      
  vColor += c * headColor;
}

// Function 77
float textColor(vec2 from, vec2 to, vec2 p)
{
	p *= font_size;
	float inkNess = 0., nearLine, corner;
	nearLine = minimum_distance(from,to,p); // basic distance from segment, thanks http://glsl.heroku.com/e#6140.0
	inkNess += smoothstep(0., 1., 1.- 14.*(nearLine - STROKEWIDTH)); // ugly still
	inkNess += smoothstep(0., 2.5, 1.- (nearLine  + 5. * STROKEWIDTH)); // glow
	return inkNess;
}

// Function 78
void UIStyle_GetFontStyleWindowText( inout LayoutStyle style, inout RenderStyle renderStyle )
{
    style = LayoutStyle_Default();
    style.vSize *= 0.75;
	renderStyle = RenderStyle_Default( vec3(0.0) );
}

// Function 79
vec4 textureFadeHorizontal(sampler2D tex, vec2 uv, float fadeWidth) {
    vec2 offsetuv = uv*vec2(1.0-fadeWidth, 1.0 - fadeWidth);
    
    float scaling = 1.0 - fadeWidth;
    float vBlend = clamp((uv.x-scaling)/fadeWidth, 0.0, 1.0);
    
    float q1Blend = (1.0-vBlend);
    vec2 q1Sample;
    q1Sample.x = fract(offsetuv.x + fadeWidth);
    q1Sample.y = fract(offsetuv.y + fadeWidth);
    vec4 tex1 = texture(tex, q1Sample); 
    vec4 q1Col = q1Blend * tex1;

    float q2Blend = vBlend;
    vec2 q2Sample;
    q2Sample.x = fract(offsetuv.x + (fadeWidth * 2.0));
    q2Sample.y = fract(offsetuv.y + fadeWidth);
    vec4 tex2 = texture(tex, q2Sample);
    vec4 q2Col = q2Blend * tex2;
    
    return q1Col + q2Col;
   
}

// Function 80
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

// Function 81
float getSphereMappedTexture(in vec3 pointOnSphere) {
    /* Test to determine which face we are drawing on.
     * Opposing faces are taken care of by the absolute
     * value, leaving us only three tests to perform.
     */
    vec2 st = abs(
        insideBounds(sphereToCube(pointOnSphere    )) +
        insideBounds(sphereToCube(pointOnSphere.zyx)) +
        insideBounds(sphereToCube(pointOnSphere.xzy)));
    return textureFunc(st);
}

// Function 82
vec2 font_from_screen(vec2 tpos, float font_size, vec2 char_pos) {    
    return (tpos/font_size + char_pos + 0.5)/GLYPHS_PER_UV;
}

// Function 83
vec4 textureAniso(sampler2D T, vec2 p) {
    mat2 J = mat2(dFdx(p),dFdy(p));                 // pixel footprint in texture space
    vec2 A,a; float M,m,l;
    ellips(J, A,a,M,m); 
    A *= M;
    l = log2( m * R.y );                            // MIPmap level corresponding to min radius
    if (M/m>16.) l = log2(M/16.*R.y);               // optional      
    vec4 O = vec4(0);
    for (float i = -7.5; i<8.; i++)                 // sample x16 along main axis at LOD min-radius
        O += textureLod(iChannel0, p+(i/16.)*A, l);
    return O/16.;
}

// Function 84
vec3 sdSegmentExt( in vec2 p, in vec2 a, in vec2 b , in float tk)
{
    vec2 n=normalize(b-a), l= vec2(length(b-a)*.5,tk);
    p=(p-a)*mat2(n.x,n.y,-n.y,n.x) -vec2(l.x,0);
    vec2 d = abs(p)-l;
    return vec3(length(max(d,0.0)) + min(max(d.x,d.y),0.0),p.x/l.x/2.+.5,l.x*2.);
}

// Function 85
vec3 calculate_texture(vec2 uv)
{
    //vec2 uv_tiled = mod(uv * 100.0, vec2(1.0));
    vec2 uv_tiled = mod(uv * vec2(40.0, 4.0), vec2(1.0));
    
     //uv_tiled.y += abs(block_offset.x) * 0.1;
    
    vec3 texture_color;
    texture_color.x = 0.5 + 0.5 * sin(uv_tiled.x * 10.0);
    texture_color.y = 0.5 + 0.5 * sin(uv_tiled.y * 10.0 + cos(uv_tiled.x * 10.0));
    texture_color.z = 0.5;
    
    texture_color = vec3(0.0);
    
    vec2 block_offset = vec2(sin(uv.x * 10.0 + cos(uv.y * 10.0) * 5.0));
    
    float offset = step(mod(uv.x * 40.0, 7.0), 1.0);
    offset += step(mod(uv.x * 40.0, 3.0), 1.0);
    offset += step(mod(uv.x * 40.0, 13.0), 1.0);
    
    float offset_y = step(mod(uv.y * 4.0, 7.0), 1.0);
    offset_y += step(mod(uv.x * 40.0, 3.0), 1.0);
    offset_y *= step(mod(uv.x * 40.0, 3.0), 1.0);
    
    float step_x = step(uv_tiled.x, 0.1 + 0.1 * offset); 	//
    float step_y = step(uv_tiled.y, 0.1 + 0.1 * offset_y);
    
    texture_color *= 0.2;
    
    texture_color = mix(texture_color, vec3(0.8, 0.7, 1.0), step_x);
    texture_color = mix(texture_color, vec3(0.8, 0.7, 1.0), step_y);
    
    
    
    float cube_corner_step_x = step(uv_tiled.x, 0.1);
    float cube_corner_step_y = step(uv_tiled.y, 0.1);
    
    vec3 cube_corner_glow = vec3(0.0);
    cube_corner_glow = mix(cube_corner_glow, vec3(0.5, 0.2, 0.2), cube_corner_step_x);
    cube_corner_glow = mix(cube_corner_glow, vec3(0.5, 0.2, 0.2), cube_corner_step_y);
    
    // TODO: Trace a height field each!
    texture_color = mix(cube_corner_glow, texture_color, 0.5);
    
    return(texture_color);
}

// Function 86
void RastText( inout vec3 color, float t, float l, vec3 textColor )
{
    float alpha = Smooth( 1. - ( 2. * l - 1. ) );
    color = mix( color, vec3( 0. ), saturate( exp( -t * 20. ) ) * alpha );
    color = mix( color, textColor, Smooth( -t * 100. ) * alpha );    
}

// Function 87
vec4 drawTextVertical(in vec2 uv, in vec2 start, in float lsize, in vec2 _text[10], in int _tsize)
{
    vec4 tcol;
    
    for (int i_letter = 0; i_letter < _tsize; i_letter++)
    {
        tcol += drawLetter(uv, start - float(i_letter)*vec2(0., lsize), lsize, _text[i_letter]);
    }
    
    return clamp(tcol, 0., 1.);
}

// Function 88
vec3 drawAllText(vec2 p){
    vec3 buffer = vec3(0);
    
    // Bounding box test.
    if(p.x < 80.0 && iResolution.y - p.y < 130.0){
        
        // Align and scale the text positioning accordingly;
        p.y -= iResolution.y-17.0;
        p /= 15.0;

        // Null-terminated strings.
        // They all have to be the same length due to GLSL limitations.
        const int Len = 13;
        const int Incenter[Len] = int[](73, 110, 99, 101, 110, 116, 101, 114, 0, 0, 0, 0, 0);
        const int Centroid[Len] = int[](67, 101, 110, 116, 114, 111, 105, 100, 0, 0, 0, 0, 0);
        const int Circumcenter[Len] = int[](67, 105, 114, 99, 117, 109, 99, 101, 110, 116, 101, 114, 0);
        const int Orthocenter[Len] = int[](79, 114, 116, 104, 111, 99, 101, 110, 116, 101, 114, 0, 0);
        const int Ninepoint[Len] = int[](78, 105, 110, 101, 112, 111, 105, 110, 116, 0, 0, 0, 0);
        const int Symmedian[Len] = int[](83, 121, 109, 109, 101, 100, 105, 97, 110, 0, 0, 0, 0);
        const int Gergonne[Len] = int[](71, 101, 114, 103, 111, 110, 110, 101, 0, 0, 0, 0, 0);
        const int Nagelpoint[Len] = int[](78, 97, 103, 101, 108, 112, 111, 105, 110, 116, 0, 0, 0);
        const int Mittenpunkt[Len] = int[](77, 105, 116, 116, 101, 110, 112, 117, 110, 107, 116, 0, 0);
        const int Spieker[Len] = int[](83, 112, 105, 101, 107, 101, 114, 0, 0, 0, 0, 0, 0);
        printf(p, buffer, Incenter, Colors[0]);
        printf(p, buffer, Centroid, Colors[1]);
        printf(p, buffer, Circumcenter, Colors[2]);
        printf(p, buffer, Orthocenter, Colors[3]);
        printf(p, buffer, Ninepoint, Colors[4]);
        printf(p, buffer, Symmedian, Colors[5]);
        printf(p, buffer, Gergonne, Colors[6]);
        printf(p, buffer, Nagelpoint, Colors[7]);
        printf(p, buffer, Mittenpunkt, Colors[8]);
        printf(p, buffer, Spieker, Colors[9]);
    }
    return buffer;
}

// Function 89
vec4 text(in vec2 p) {
  vec2 pos = vec2(41, 72);
  vec2 size = vec2(197, 56);
  float bounds = float(all(greaterThanEqual(p, pos)) && all(lessThan(p-pos, size)));
  return bounds*texture(iChannel0, (p - pos) / size);
}

// Function 90
float NumFont_Three( vec2 vTexCoord )
{
    float fResult = NumFont_Circle( vTexCoord );
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 1), vec2(8,13) ));

    float fHole = NumFont_Rect( vTexCoord, vec2(-1, 4), vec2(7,5) );
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(-1, 9), vec2(7,10) ));
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(-1, 6), vec2(3,8) ));
    
    fResult = min( fResult, 1.0 - fHole );    
    
    return fResult;
}

// Function 91
float text(in vec2 uv){
  vec2 scl=vec2(320.,36.);
  vec2 F=floor(uv*scl),P=fract(uv*scl);
  float lo=0.;int io=0;
  if(F.x>75.){io=-75;lo=scl.y;if(F.x>165.){io-=90;lo+=scl.y;if(F.x>240.){io-=75;lo+=scl.y;}}}
  vec2 v=vec2((scl.y-1.-F.y)+1.5+lo,0.5);
  vec4 t=texture(iChannel0,v/R);
  float idx=F.x+t.x+float(io);
  t=texture(iChannel0,(v+vec2(1,0))/R);
  float d=1.;
  if(idx<t.x){
    v=vec2(0.5+mod(idx,R.x),1.5+floor(idx/R.x));
    t=texture(iChannel0,v/R);
    if(t.x==0.)return 1.;
    float r=25.*max(length(fwidth(uv)),0.0008);//
    d=DEC(P,t.xy);
    d=smoothstep(0.53-r,0.53+r,d);
  }
  return d;
}

// Function 92
void post_text_overlay( inout vec3 col, vec2 coord, vec3 cc )
{
    coord = ( coord - g_overlayframe.xy ) * g_textscale;
	for( int i = 0; i < TXT_FMT_MAX_COUNT; ++i )
		col += g_hudcolor * hmd_txtout( coord, cc, i );
}

// Function 93
float textureAtlas(vec2 uv, int hitid)
{
    return alphatex(uv, ivec2(hitid, hitid >> 4) & 15);
    // TODO various symmetry modes to extend the available shapes
    // simple extrusions, lathes on various axes, vary orientation, etc.
    // kind of limited in 2D though.
}

// Function 94
float glyph_dist(in vec2 pt)
{
    float angle = atan(pt.y, pt.x) - iTime * 0.1;
    float len = length(pt);
    float rad = 1.0 - len;
    
    float theta = angle + sin(iTime - len * 10.0) * 0.2;
    
    return rad - abs(sin(theta * 2.5)) * 0.6;
}

// Function 95
float FontTexDf (vec2 p)
{
  vec3 tx;
  float d;
  int ic;
  ic = GetTxChar (p);
  if (ic != 0) {
    tx = texture (txFnt, mod ((vec2 (mod (float (ic), 16.),
       15. - floor (float (ic) / 16.)) + fract (p)) * (1. / 16.), 1.)).gba - 0.5;
    qnFnt = vec2 (tx.r, - tx.g);
    d = tx.b + 1. / 256.;
  } else d = 1.;
  return d;
}

// Function 96
float text_res(vec2 U) {
    initMsg;C(82);C(101);C(115);C(116);C(97);C(114);C(116);endMsg;
}

// Function 97
vec4 text(in vec2 uv) {
  vec2 size = vec2(197, 56);
  vec2 size1 = vec2(116, 16);
  vec2 size2 = vec2(166, 16);
  vec2 size3 = vec2(158, 8);
  vec2 uv1 = uv * size;
  vec2 uv2 = uv1 - vec2(4, 24);
  vec2 uv3 = uv1 - vec2(39, 48);
  float c = text1(uv1 / size1) + text2(uv2 / size2);
  float b = text3(uv3 / size3);
  return vec4(c*0.7803, c*0.7686, c*0.7803 + b*0.7803, 1);
}

// Function 98
void textureSolid(in vec3 block, inout ray ray, inout rayMarchHit hit, inout vec3 colour, bool isReflection, in float time) {
    float concrete = getConcrete(hit.origin, hit.surfaceNormal, true);
    colour = hash33(block.xyx) * vec3(0.25,0.1,0.2) + 0.5;
    colour = clamp(colour,vec3(0.0),vec3(1.0));
    colour *= concrete;
}

// Function 99
float text_d(vec2 U) {
    initMsg;C(68);C(101);C(102);C(101);C(97);C(116);endMsg;
}

// Function 100
vec4 textureGood(sampler2D sam, in vec2 x, in int bits) {
	ivec2 p = ivec2(floor(x));
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    vec4 a = texelFetch(sam, (p+ivec2(0,0)) & bits, 0);
	vec4 b = texelFetch(sam, (p+ivec2(1,0)) & bits, 0);
	vec4 c = texelFetch(sam, (p+ivec2(0,1)) & bits, 0);
	vec4 d = texelFetch(sam, (p+ivec2(1,1)) & bits, 0);
	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Function 101
vec4 texture3D(sampler2D tex, vec3 uvw, vec3 vres)
{
    uvw = mod(floor(uvw * vres), vres);
    
    //XYZ -> Pixel index
    float idx = (uvw.z * (vres.x*vres.y)) + (uvw.y * vres.x) + uvw.x;
    
    //Pixel index -> Buffer uv coords
    vec2 uv = vec2(mod(idx, iResolution.x), floor(idx / iResolution.x));
    
    //WEBGL 2 FIX: texture(...) caused loop unrolling errors, using textureLod(...) fixes this.
    return textureLod(tex, (uv + 0.5) / iResolution.xy, 0.0);
}

// Function 102
float Text(vec2 uv)
{
    float col = 0.0;
    
    print_pos = vec2(res.x/2.0 - STRWIDTH(17.0)/2.0,res.y/2.0 - STRHEIGHT(1.0)/2.0);
    print_pos = floor(print_pos);
       
    col += char(ch_H,uv);
    col += char(ch_e,uv);
    col += char(ch_l,uv);
    col += char(ch_l,uv);
    col += char(ch_o,uv);
    col += char(ch_com,uv);
    
    col += char(ch_spc,uv);
    
    col += char(ch_S,uv);
    col += char(ch_h,uv);
    col += char(ch_a,uv);
    col += char(ch_d,uv);
    col += char(ch_e,uv);
    col += char(ch_r,uv);
    col += char(ch_t,uv);
    col += char(ch_o,uv);
    col += char(ch_y,uv);
    col += char(ch_exc,uv);
    
    print_pos = vec2(2);
    
    col += char(ch_T,uv);
    col += char(ch_i,uv);
    col += char(ch_m,uv);
    col += char(ch_e,uv);
    col += char(ch_col,uv);
    
    col += print_number(iTime,print_pos,uv); 
    
    return col;
}

// Function 103
float gridTexture( in vec2 p )
{
    // coordinates
    vec2 i = step( fract(p), vec2(1.0/N) );
    //pattern
    //return (1.0-i.x)*(1.0-i.y);   // grid (N=10)
    
    // other possible patterns are these
    //return 1.0-i.x*i.y;           // squares (N=4)
    return 1.0-i.x-i.y+2.0*i.x*i.y; // checker (N=2)
}

// Function 104
float text( vec3 pos )
{
    float f = sin(pos.y*.2*sin(iTime*.1)+pos.x*.1-iTime*.25+sin(iTime*.5-pos.y*.033)*.1+cos(iTime*.3+pos.y*.1)*.15)*.9+.9;
    float smoothing = pow(smoothstep(5., 2., abs(pos.y)), 5.)*1.+.6;
    smoothing += f;
    float walls = sdBox(abs(pos)-vec3(0., 3.5+f*5., 0.), vec3(32.8, .0, .0));

    pR(pos.zx, sin(iTime*.2-pos.x)*.01);
    pos.x+=spacing.x*(chars-.75)*.5;
    pos.y-= spacing.w*.5;
    
    float text = 100.;
    float nr = 0.;
    vec2 uv = pos.xy; 
    float width = textWidth;
    line1;
    // bubble the text
    text = length(vec2(text, pos.z));
    width+=(sin(iTime-pos.y+pos.x*.2)*width+width)*.1;
    text-=width;
    
    text = smin(text, walls, smoothing);
    
    return text;
    
}

// Function 105
vec3 rayToTexture( vec3 p ) {
    return (p - vec3(0.0,0.5,0.0)) * 0.2 + 0.5;
}

// Function 106
vec4 textureQuadratic( in sampler3D sam, in vec3 p )
{
    float texSize = float(textureSize(sam,0).x); 

	p = p*texSize;
	vec3 i = floor(p);
	vec3 f = fract(p);
	p = i + f*0.5;
	p = p/texSize;
    
	float w = 0.5/texSize;

	return mix(mix(mix(texture3(sam,p+vec3(0,0,0)),
                       texture3(sam,p+vec3(w,0,0)),f.x),
                   mix(texture3(sam,p+vec3(0,w,0)),
                       texture3(sam,p+vec3(w,w,0)),f.x), f.y),
               mix(mix(texture3(sam,p+vec3(0,0,w)),
                       texture3(sam,p+vec3(w,0,w)),f.x),
                   mix(texture3(sam,p+vec3(0,w,w)),
                       texture3(sam,p+vec3(w,w,w)),f.x), f.y), f.z);
}

// Function 107
float glyphCover(inout vec2 p, vec2 gdata) {
    p = floor(p);
    float c;
    if (p.x >= 0.0 && p.x < 3.0 && p.y >= 0.0 && p.y < 5.0) {
        float bit = dot(p, vec2(1.0, -3.0)) + 12.0;
        c = mod(floor(gdata.x / pow(2.0, bit)), 2.0);
    } else {
        c = 0.0;
    }
    p.x -= gdata.y;
    return c;
}

// Function 108
void textureWall(vec2 block, vec3 position, inout vec3 colour, inout vec3 normal, inout int material) {
    float scale = 2.0;

    float windowHeight =	hash21(block*39.195)*0.4+0.4;
    float windowWidth =		hash21(block*26.389)*0.7+0.2;

    if (windowWidth > 0.8){
        windowWidth=1.0;
    }

    vec3 ramp = fract(position*scale)*2.0-1.0;

    vec2 uv;
    if (windowWidth==1.0) {
        uv.x=0.0; 
    } else if (abs(ramp.x) > abs(ramp.z)) {
        uv.x = ramp.x;
    } else {
        uv.x = ramp.z;
    }
    uv.y = ramp.y;

    if ( (abs(uv.x) < windowWidth) && abs(uv.y) < windowHeight) {
        colour=vec3(0.0);
        material = MAT_WINDOW;
    } else {
        //uv.x=clamp(abs(uv.x)-windowWidth,0.0,1.0)/(1.0-windowWidth);
        //uv.y=clamp(abs(uv.y)-windowHeight,0.0,1.0)/(1.0-windowHeight);

        //Remove vertical lines when windows are close togther
        if (windowWidth>=0.7) {
            uv.x=0.0; 
        }

        //FIXME: apply correct normal calculations, this is NOT how you do it!
        vec3 pNormal;
        uv*=uv*uv*0.5;
        pNormal = vec3(uv.x,uv.y,uv.x);
        normal = normalize(normal-pNormal);

        float concrete = getConcrete(position, normal, false);
        colour = hash33(block.xyx) * vec3(0.25,0.1,0.2) + 0.5;
        colour = clamp(colour,vec3(0.0),vec3(1.0));
        colour *= concrete;
    }
}

// Function 109
vec4 getTexture(float id, vec2 c) {
    vec2 gridPos = vec2(mod(id, 16.), floor(id / 16.));
	return texture(iChannel2, (c + gridPos * 16.) / iChannelResolution[3].xy);
}

// Function 110
vec3 textureRoof(vec2 pos, vec2 maxPos,vec2 squarer
){float wsize=0.025
 ;float randB=rand(squarer*2.0)
 ;vec3 wallColor=(0.3+1.4*fract(randB*100.))*vec3(.1)+(-0.7+1.4*fract(randB*1000.))*vec3(0.02,0.,0.)
 ;vec3 conturColor=wallColor*1.5/2.5
 ;vec2 contur=vec2(0.02)
 ;if ((maxPos.x-pos.x<contur.x)||(maxPos.y-pos.y<contur.y)||(pos.x<contur.x)||(pos.y<contur.y)
 )return (0.9+0.2*noise(pos))*conturColor
 ;float s=.06+.12*fract(randB*562526.2865)
 ;pos-=s;maxPos-=s*2.;if(con(pos,maxPos,contur))return(.9+.2*noise(pos))*conturColor
 ;pos-=s;maxPos-=s*2.;if(con(pos,maxPos,contur))return(.9+.2*noise(pos))*conturColor
 ;pos-=s;maxPos-=s*2.;if(con(pos,maxPos,contur))return(.9+.2*noise(pos))*conturColor
 ;return (.9+.2*noise(pos))*wallColor;}

// Function 111
vec3 textureRoof(vec2 pos, vec2 maxPos,vec2 squarer){
    float wsize = 0.025;
    float randB = rand(squarer*2.0);
    vec3 wallColor = (0.3+1.4*fract(randB*100.0))*vec3(0.1,0.1,0.1)+(-0.7+1.4*fract(randB*1000.0))*vec3(0.02,0.,0.);
	vec3 conturColor = wallColor*1.5/2.5;
    vec2 contur = vec2(0.02);
    if ((maxPos.x-pos.x<contur.x)||(maxPos.y-pos.y<contur.y)||(pos.x<contur.x)||(pos.y<contur.y)){
            return (0.9+0.2*noise(pos))*conturColor;
        
    }
    float step1 = 0.06+0.12*fract(randB*562526.2865);
    pos -=step1;
    maxPos -=step1*2.0;
    if ((pos.x>0.0&&pos.y>0.0&&pos.x<maxPos.x&&pos.y<maxPos.y)&&((abs(maxPos.x-pos.x)<contur.x)||(abs(maxPos.y-pos.y)<contur.y)||(abs(pos.x)<contur.x)||(abs(pos.y)<contur.y))){
            return (0.9+0.2*noise(pos))*conturColor;
        
    }
    pos -=step1;
    maxPos -=step1*2.0;
    if ((pos.x>0.0&&pos.y>0.0&&pos.x<maxPos.x&&pos.y<maxPos.y)&&((abs(maxPos.x-pos.x)<contur.x)||(abs(maxPos.y-pos.y)<contur.y)||(abs(pos.x)<contur.x)||(abs(pos.y)<contur.y))){
            return (0.9+0.2*noise(pos))*conturColor;
        
    }
    pos -=step1;
    maxPos -=step1*2.0;
    if ((pos.x>0.0&&pos.y>0.0&&pos.x<maxPos.x&&pos.y<maxPos.y)&&((abs(maxPos.x-pos.x)<contur.x)||(abs(maxPos.y-pos.y)<contur.y)||(abs(pos.x)<contur.x)||(abs(pos.y)<contur.y))){
            return (0.9+0.2*noise(pos))*conturColor;
        
    }
    
    return (0.9+0.2*noise(pos))*wallColor;
    
}

// Function 112
void paintModeText(inout vec4 col, in uvec2 coord, in vec3 bg)
{   
    vec2 uuv = vec2(coord) / iResolution.y;
    //float H = 10.;
    //float gly = 1. / H * 400.;
    float scale = 3.2;
    vec2 uv = /*uuv * gly - vec2(.5,gly-1.-.5)*/ vec2(coord.x, float(coord.y) - iResolution.y) / iResolution.x * 22.0 * scale + vec2(-.5,1.5);
    float px = 22. / iResolution.x * scale;

    float x = 100.;
    float cp = 0.;
    vec4 cur = vec4(0,0,0,0.5);
    vec4 us = cur;
    float ital = 0.0;

    //int lnr = int(floor(uv.y/2.));
    //uv.y = mod(uv.y,2.0)-0.5;

    if (uv.y >= -1.0 && uv.y <= 1.0) {
        BLACK M_ o_ d_ e_ _dotdot _
        
    	int mode = int(readVar(VARIABLE_LOCATION_MODE));
       	
        NOITAL DARKGREEN ITAL
        switch (mode)
        {
            case MODE_LOWEST_NONVOID_Z:
            	L_ o_ w_ e_ s_ t_ _ N_ o_ n_ _sub V_ o_ i_ d_ _ Z_
                break;
            case MODE_CURRENT_Z:
            	C_ u_ r_ r_ e_ n_ t_ _ Z_
            	break;
            	
        }

        vec3 clr = vec3(0.0);

        float weight = 0.01+cur.w*.05;//min(iTime*.02-.05,0.03);//+.03*length(sin(uv*6.+.3*iTime));//+0.02-0.06*cos(iTime*.4+1.);
        col = mix(col, vec4(us.rgb, 1.0), smoothstep(weight+px, weight-px, x));
    }
}

// Function 113
void RenderFont( PrintState state, LayoutStyle style, RenderStyle renderStyle, inout vec3 color )
{   
    float f = GetFontBlend( state, style, renderStyle.fFontWeight );

    vec3 vCol = renderStyle.vFontColor;
    
    color.rgb = mix( color.rgb, vCol, f);    
}

// Function 114
float NumFont_Six( vec2 vTexCoord )
{
    float fResult = NumFont_Circle( vTexCoord );
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(4, 1), vec2(11,13) ));

    float fHole = NumFont_Rect( vTexCoord, vec2(5, 9), vec2(8,10) );
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(5, 4), vec2(17,5) ));
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(10, 6), vec2(17,6) ));
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(10, 13), vec2(17,13) ));

    fResult = min( fResult, 1.0 - fHole );    
    
    return fResult;
}

// Function 115
vec3 groundTexture(in vec3 rp) {
    return texture(iChannel0,rp.xz*.1).xyz;
}

// Function 116
vec4 texture0( in vec2 x )
{
    //return texture( iChannel0, x );
    vec2 res = iChannelResolution[0].xy;
    vec2 u = x*res - 0.5;
    vec2 p = floor(u);
    vec2 f = fract(u);
    f = f*f*(3.0-2.0*f);    
    vec4 a = texture( iChannel0, (p+vec2(0.5,0.5))/res, -64.0 );
	vec4 b = texture( iChannel0, (p+vec2(1.5,0.5))/res, -64.0 );
	vec4 c = texture( iChannel0, (p+vec2(0.5,1.5))/res, -64.0 );
	vec4 d = texture( iChannel0, (p+vec2(1.5,1.5))/res, -64.0 );
    return mix(mix(a,b,f.x), mix(c,d,f.x),f.y);
}

// Function 117
int text_width(int num_chars)				{ return num_chars << 3; }

// Function 118
vec3 textureGround(vec2 squarer, vec2 pos,vec2 a,vec2 b,float dist
){vec3 color=(0.9+0.2*noise(pos))*vec3(0.1,0.15,0.1)
 ;float randB=rand(squarer*2.)
 ;vec3 wallColor=(.3+1.4*fract(randB*100.))*.1+(-.7+1.4*fract(randB*1000.))*vec3(.02,0,0)
 ;float fund=0.03
 ;float bl=0.01
 ;float f=smoothstep(a.x-fund-bl,a.x-fund,pos.x)
 ;f*=smoothstep(a.y-fund-bl,a.y-fund,pos.y)
 ;f*=smoothstep(b.y+fund+bl,b.y+fund,pos.y)
 ;f*=smoothstep(b.x+fund+bl,b.x+fund,pos.x)
 ;pos -= 0.0
 ;vec2 maxPos=vec2(1)
 ;vec2 contur=vec2(0.06,0.06)
 ;if((pos.x>0.&&pos.y>0.&&pos.x<maxPos.x&&pos.y<maxPos.y)&&((abs(maxPos.x-pos.x)<contur.x)||(abs(maxPos.y-pos.y)<contur.y)||(abs(pos.x)<contur.x)||(abs(pos.y)<contur.y)))
            color= vec3(0.1,0.1,0.1)*(0.9+0.2*noise(pos))
 ;pos -= 0.06
 ;maxPos=vec2(.88)
 ;contur=vec2(.01)
 ;if ((pos.x>0.0&&pos.y>0.0&&pos.x<maxPos.x&&pos.y<maxPos.y)&&((abs(maxPos.x-pos.x)<contur.x)||(abs(maxPos.y-pos.y)<contur.y)||(abs(pos.x)<contur.x)||(abs(pos.y)<contur.y))) color=vec3(0)
 ;color=mix(color,(0.9+0.2*noise(pos))*wallColor*1.5/2.5,f)
 ;pos+=0.06    
#ifdef CARS
 ;if (min(pos.x,pos.y)<0.07||max(pos.x,pos.y)>0.93) color+=cars(squarer,pos,dist,0.);
#endif
 ;return color;}

// Function 119
float text_s(vec2 U) {
    initMsg;
    U.x+=10.*(0.5-0.2812*(res.x/0.5));
    C(83);C(112);C(101);C(101);C(100);C(32);C(115);C(99);C(97);C(108);C(101);
    endMsg;
}

// Function 120
vec4 texture_lod(SamplerState state, vec2 uv, int lod)
{
    float texel_scale = state.atlas_scale * exp2i(-lod);
    bool use_filter = test_flag(state.flags, OPTION_FLAG_TEXTURE_FILTER);
	if (use_filter)
    	uv += -.5 / texel_scale;
    
    uv = fract(uv / state.tile.zw);
    state.tile *= texel_scale;
    uv *= state.tile.zw;
  
    vec2 mip_base = mip_offset(lod) * ATLAS_SIZE * state.atlas_scale + state.tile.xy + ATLAS_OFFSET;

    if (use_filter)
    {
        ivec4 address = ivec2(mip_base + uv).xyxy;
        address.zw++;
        if (uv.x >= state.tile.z - 1.) address.z -= int(state.tile.z);
        if (uv.y >= state.tile.w - 1.) address.w -= int(state.tile.w);

        vec4 s00 = gamma_to_linear(texelFetch(iChannel3, address.xy, 0));
        vec4 s10 = gamma_to_linear(texelFetch(iChannel3, address.zy, 0));
        vec4 s01 = gamma_to_linear(texelFetch(iChannel3, address.xw, 0));
        vec4 s11 = gamma_to_linear(texelFetch(iChannel3, address.zw, 0));

        uv = fract(uv);
        return linear_to_gamma(mix(mix(s00, s10, uv.x), mix(s01, s11, uv.x), uv.y));
    }
    else
    {
        return texelFetch(iChannel3,  ivec2(mip_base + uv), 0);
    }
}

// Function 121
float Text(vec2 uv)
{
    float col = 0.0;
    
    print_pos = vec2(res.x/2.0 - STRWIDTH(17.0)/2.0,res.y/2.0 - STRHEIGHT(1.0)/2.0);
    print_pos = floor(print_pos);
       
    col += char(sg_h,uv);
    col += char(sg_e,uv);
    col += char(sg_l,uv);
    col += char(sg_l,uv);
    col += char(sg_o,uv);
    
    col += char(sg_spc,uv);
    
    col += char(sg_s,uv);
    col += char(sg_h,uv);
    col += char(sg_a,uv);
    col += char(sg_d,uv);
    col += char(sg_e,uv);
    col += char(sg_r,uv);
    col += char(sg_t,uv);
    col += char(sg_o,uv);
    col += char(sg_y,uv);
    
    col += char(sg_end,uv);
    
    return col;
}

// Function 122
void newText(vec2 p, in vec2 o){pixel = p; text= vec4(o, 0,0);}

// Function 123
vec3 GetTextureOffset(vec2 coords, vec2 textureSize, vec2 texelOffset)
{
    vec2 texelSize = 1.0 / textureSize;
    vec2 offsetCoords = coords + texelSize * texelOffset;
    
    vec2 halfTexelSize = texelSize / 2.0;
    vec2 clampedOffsetCoords = clamp(offsetCoords, halfTexelSize, 1.0 - halfTexelSize);
    
    return texture(iChannel0, clampedOffsetCoords).rgb;
}

// Function 124
vec3 textureGround(vec2 squarer, vec2 pos,vec2 vert1,vec2 vert2,float dist){
    vec3 color = (0.9+0.2*noise(pos))*vec3(0.1,0.15,0.1);
    float randB = rand(squarer*2.0);

    vec3 wallColor = (0.3+1.4*fract(randB*100.0))*vec3(0.1,0.1,0.1)+(-0.7+1.4*fract(randB*1000.0))*vec3(0.02,0.,0.);
	float fund = 0.03;
    float bl = 0.01;
    float f = smoothstep(vert1.x-fund-bl,vert1.x-fund,pos.x);
    f *= smoothstep(vert1.y-fund-bl,vert1.y-fund,pos.y);
    f *= smoothstep(vert2.y+fund+bl,vert2.y+fund,pos.y);
    f *= smoothstep(vert2.x+fund+bl,vert2.x+fund,pos.x);

    pos -= 0.0;
    vec2 maxPos = vec2(1.,1.);
    vec2 contur = vec2(0.06,0.06);
    if ((pos.x>0.0&&pos.y>0.0&&pos.x<maxPos.x&&pos.y<maxPos.y)&&((abs(maxPos.x-pos.x)<contur.x)||(abs(maxPos.y-pos.y)<contur.y)||(abs(pos.x)<contur.x)||(abs(pos.y)<contur.y))){
            color =  vec3(0.1,0.1,0.1)*(0.9+0.2*noise(pos));
        
    }
    pos -= 0.06;
    maxPos = vec2(.88,0.88);
    contur = vec2(0.01,0.01);
    if ((pos.x>0.0&&pos.y>0.0&&pos.x<maxPos.x&&pos.y<maxPos.y)&&((abs(maxPos.x-pos.x)<contur.x)||(abs(maxPos.y-pos.y)<contur.y)||(abs(pos.x)<contur.x)||(abs(pos.y)<contur.y))){
            color =  vec3(0.,0.,0.);
        
    }
    color = mix(color,(0.9+0.2*noise(pos))*wallColor*1.5/2.5,f);

    pos+=0.06;
    
#ifdef CARS
    if (pos.x<0.07||pos.x>0.93||pos.y<0.07||pos.y>0.93){
        color+=cars(squarer,pos,dist,0.0);
    }
#endif
    
    return color;
}

// Function 125
vec3 TextureSand(in vec3 p)
{
  vec3 c = vec3(0.68, 0.35, 0.17)*2.5;
  return mix(0.35*c, 0.65*c, Fbm(p));
}

// Function 126
float text_n0(vec2 U) {
    initMsg;C(48);C(48);endMsg;
}

// Function 127
int GetGlyphPixel(ivec2 pos, int g)
{
	if (pos.x >= glyphSize || pos.y >= glyphSize)
		return 0;

    // pull glyph out of hex
	int glyphRow = GetGlyphPixelRow(pos.y, g);
    return 1 & (glyphRow >> (glyphSize - 1 - pos.x) * 4);
}

// Function 128
vec4 texture_Bilinear( sampler2D tex, vec2 t )
{
    vec2 res = iChannelResolution[0].xy;
    vec2 p = res*t - 0.5;
    vec2 f = fract(p);
    vec2 i = floor(p);

    return lerp( f.y, lerp( f.x, SAM(0,0), SAM(1,0)),
                      lerp( f.x, SAM(0,1), SAM(1,1)) );
}

// Function 129
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

// Function 130
float gridTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )
{
	// filter kernel
    vec2 w = max(abs(ddx), abs(ddy)) + 0.01;

	// analytic (box) filtering
    vec2 a = p + 0.5*w;                        
    vec2 b = p - 0.5*w;           
    vec2 i = (floor(a)+min(fract(a)*N,1.0)-
              floor(b)-min(fract(b)*N,1.0))/(N*w);
    //pattern
    return (1.0-i.x)*(1.0-i.y);
}

// Function 131
float sampleFont(vec2 p, float num) {
    float glyph[2];
    if (num < 1.)      { glyph[0] = 0.91333008; glyph[1] = 0.89746094; }
    else if (num < 2.) { glyph[0] = 0.27368164; glyph[1] = 0.06933594; }
    else if (num < 3.) { glyph[0] = 1.87768555; glyph[1] = 1.26513672; }
    else if (num < 4.) { glyph[0] = 1.87719727; glyph[1] = 1.03027344; }
    else if (num < 5.) { glyph[0] = 1.09643555; glyph[1] = 1.51611328; }
    else if (num < 6.) { glyph[0] = 1.97045898; glyph[1] = 1.03027344; }
    else if (num < 7.) { glyph[0] = 0.97045898; glyph[1] = 1.27246094; }
    else if (num < 8.) { glyph[0] = 1.93945312; glyph[1] = 1.03222656; }
    else if (num < 9.) { glyph[0] = 0.90893555; glyph[1] = 1.27246094; }
    else               { glyph[0] = 0.90893555; glyph[1] = 1.52246094; }
    
    float pos = floor(p.x + p.y * 5.);
    if (pos < 13.) {
        return step(1., mod(pow(2., pos) * glyph[0], 2.));
    } else {
        return step(1., mod(pow(2., pos-13.) * glyph[1], 2.));
    }
}

// Function 132
vec3 textureNoTile( in vec2 x)
{
    float v = 1.0;
    
    //float k = texture( iChannel1, 0.005*x ).x; // cheap (cache friendly) lookup
    float k = noise(x * 3.0);
    
    float l = k*8.0;
    float i = floor( l );
    float f = fract( l );
    
    vec2 offa = sin(vec2(3.0,7.0)*(i+0.0)); // can replace with any other hash
    vec2 offb = sin(vec2(3.0,7.0)*(i+1.0)); // can replace with any other hash

    vec3 cola = texture( iChannel0, x + v*offa ).xyz;
    vec3 colb = texture( iChannel0, x + v*offb ).xyz;
    
    return mix( cola, colb, smoothstep(0.2,0.8,f-0.1*sum(cola-colb)) );
}

// Function 133
vec3 betterTextureSample256(sampler2D tex, vec2 uv) {	
	float textureResolution = 256.0;
	uv = uv*textureResolution + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*(3.0-2.0*fuv); // fuv*fuv*fuv*(fuv*(fuv*6.0-15.0)+10.0);;
	uv = (uv - 0.5)/textureResolution;
	return textureLod(tex, uv, 0.0).rgb;
}

// Function 134
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

// Function 135
vec4 textureGround(vec2 uv) {
    const vec2 RES = vec2(8.0, 8.0);    
    float n = noise1(uv * RES);
    n = n * 0.2 + 0.5;
    return vec4(n*0.9,n*0.6,n*0.4,1.0);
}

// Function 136
void TeletextState_SetGfxColor( inout TeletextState state, int color )
{
    state.iFgCol = color;
    state.bGfx = true;            
    state.bConceal = false;
}

// Function 137
float Text(vec2 uv)
{    
    vec2 res = iResolution.xy / DWN_SC,
         pos = vec2(150,350.5-2.*CHR.y);
    
    carret = pos;
    
	float r = 0.;
    r += T(c_A) + T(c_B) + T(c_C) + T(c_D) + T(c_E) + T(c_F)
       + T(c_G) + T(c_H) + T(c_I) + T(c_J) + T(c_K) + T(c_L)
       + T(c_M) + T(c_N) + T(c_O) + T(c_P) + T(c_Q) + T(c_R)
       + T(c_S) + T(c_T) + T(c_U) + T(c_V) + T(c_W) + T(c_X)
       + T(c_Y) + T(c_Z);
    
    carret = pos + vec2(0, -40.);
    
    r += T(c_a) + T(c_b) + T(c_c) + T(c_d) + T(c_e) + T(c_f)
       + T(c_g) + T(c_h) + T(c_i) + T(c_j) + T(c_k) + T(c_l)
       + T(c_m) + T(c_n) + T(c_o) + T(c_p) + T(c_q) + T(c_r)
       + T(c_s) + T(c_t) + T(c_u) + T(c_v) + T(c_w) + T(c_x)
       + T(c_y) + T(c_z);
    
    
    carret = pos + vec2(0, -80.);
    r += T(c_0) + T(c_1) + T(c_2) + T(c_3) + T(c_4) + T(c_5)
       + T(c_6) + T(c_7) + T(c_8) + T(c_9);
    
    carret = pos + vec2(0, -120.);
    r += T(c_exc) + T(c_quo) + T(c_hsh) + T(c_dol) + T(c_pct)
       + T(c_amp) + T(c_apo) + T(c_lbr) + T(c_rbr) + T(c_ast)
       + T(c_crs) + T(c_per) + T(c_dsh) + T(c_com) + T(c_lsl)
       + T(c_col) + T(c_scl) + T(c_les) + T(c_equ) + T(c_grt)
       + T(c_que) + T(c_ats) + T(c_lsb) + T(c_rsl) + T(c_rsb)
       + T(c_pow) + T(c_quo) + T(c_usc) + T(c_lpa) + T(c_bar)
       + T(c_rpa) + T(c_tid) + T(c_lar) + T(c_spc);
        
	r += print_number(iTime, pos + vec2(1,50), uv);
    
    return r;
}

// Function 138
Rect UI_GetFontRect( PrintState state, LayoutStyle style )
{
    Rect rect;
    rect = GetFontRect( state, style, true );
    vec2 vExpand = UIStyle_FontPadding();
    vExpand.y += style.vSize.y * style.fLineGap;
    RectExpand( rect, vExpand );
	return rect;
}

// Function 139
void WriteText1(){
 SetTextPosition(1.,1.);
 float c = 0.0;
 //_star _ _V _i _e _w _ _S _h _a _d _e _r   
 //_ _D _a _t _a _ _2 _ _ _v _1 _dot _1 _ _star 
 vColor+=c*headColor;}

// Function 140
float halfont( vec2 p )
{
    float sc = 0.0033;
    float x = p.x+2.*sc;
    float cn = floor(float(x/(16.*sc)));
    float xx = mod(p.x+2.*sc,16.*sc)-8.*sc;
    p = vec2(xx,p.y);
    float r = 1.0;
    float outer = box(p,vec2(7.,11.)*sc);
    if (cn==-3.) {
        float htop = box(p-vec2(0.,9.25)*sc,vec2(1.7,5.75)*sc);
        float hbot = box(p+vec2(0.,6.25)*sc,vec2(1.7,5.75)*sc);
        outer = max(outer,-htop);
        outer = max(outer,-hbot);
        r = abs(outer+.5*sc)-.01*sc;
    } else if (cn==-2.) {
        float left = line(p-vec2(-2.4,0.)*sc,vec3(-1.,.2,2.3*sc));
        float right = line(p-vec2(2.4,0.)*sc,vec3(1.,.2,2.3*sc));
        float bar = box(p+vec2(0.,5.)*sc,vec2(5.,2.25)*sc);
        left = min(left,right);
        left = min(left,bar);
        outer = max(left,outer);
        r = abs(outer+.5*sc)-.01*sc;
    } else if (cn==-1.) {
        float htop = box(p-vec2(4.5,4.)*sc,vec2(7.,11.)*sc);
        float shrink = box(p-vec2(7.,-9.)*sc,vec2(2.,4.)*sc);
        outer = max(outer,-htop);
        outer = max(outer,-shrink);
        r = abs(outer+.5*sc)-.01*sc;
    } else if (cn==1.) {
        outer = squircle(p*vec2(.9,.5),vec3(5.7*sc,1.4,.5));
        float inner = squircle(p*vec2(1.1,.31),vec3(2.*sc,1.7,.7));
        float outer2 = squircle((p-vec2(-1.2,2.5)*sc),vec3(5.*sc,1.3,.4));
        float inner2 = squircle((p-vec2(0.,4.)*sc)*vec2(1.,.5),vec3(1.5*sc,1.7,.7));
        float cut = box(p-vec2(-4.,-1.)*sc,vec2(4.,3.)*sc);
        outer = max(outer,-inner);
        outer = max(outer,-cut);
        outer = min(outer,outer2);
        outer = max(outer,-inner2);
        r = abs(outer+.5*sc)-.01*sc;
    } else if (cn>1. && cn<5.) {
        outer = squircle(p*vec2(.9,.5),vec3(5.7*sc,1.4,.5));
        float inner = squircle(p*vec2(1.1,.31),vec3(2.*sc,1.7,.7));
        outer = max(outer,-inner);
        r = abs(outer+.5*sc)-.01*sc;
    }
    return r;
}

// Function 141
float spaceText(vec2 p)
{        
    vec2 scale = vec2( 4., 8. );
    vec2 t = floor( p / scale );   
    
    uint v = 0u;    
    v = t.y == 0. ? ( t.x < 4. ? 1936028240u : ( t.x < 8. ? 1935351923u : ( t.x < 12. ? 1701011824u : ( t.x < 16. ? 1869881437u : ( t.x < 20. ? 1635021600u : 29810u ) ) ) ) ) : v;
	v = t.x >= 0. && t.x < 24. ? v : 0u;
    
	float c = float( ( v >> uint( 8. * t.x ) ) & 255u );
    
    p = ( p - t * scale ) / scale;
    p.x = (p.x - .5 ) * .5 + .5;
    float sdf = textSDF( p, c );
    return ( c != 0. ) ? smoothstep( -.05, +.05, sdf ) : 1.0;
}

// Function 142
void MenuText(inout vec3 color, vec2 p, in AppState s)
{
    p -= vec2(-160, 62);
    
    vec2 scale = vec2(4., 8.);
    vec2 t = floor(p / scale);   
    
    uint v = 0u;
	v = t.y == 2. ? (t.x < 4. ? 1768452929u : (t.x < 8. ? 1768777835u : (t.x < 12. ? 5653614u : 0u))) : v;
	v = t.y == 1. ? (t.x < 4. ? 1918986307u : (t.x < 8. ? 1147496812u : (t.x < 12. ? 1752383839u : (t.x < 16. ? 1835559785u : 5664361u)))) : v;
	v = t.y == 0. ? (t.x < 4. ? 1918986307u : (t.x < 8. ? 1147496812u : (t.x < 12. ? 86u : 0u))) : v;
	v = t.x >= 0. && t.x < 20. ? v : 0u;
    
	float c = float((v >> uint(8. * t.x)) & 255u);
    
    vec3 textColor = vec3(.3);
    if (t.y == 2. - s.menuId)
    {
        textColor = vec3(0.74, 0.5, 0.12);
	}

    p = (p - t * scale) / scale;
    p.x = (p.x - .5) * .5 + .5;
    float sdf = TextSDF(p, c);
    if (c != 0.)
    {
    	color = mix(textColor, color, smoothstep(-.05, +.05, sdf));
    }
}

// Function 143
vec3 getCueTexture(vec2 uv) {
    vec3 wood = texture( iChannel1, uv.yx ).xyz;
    
    if(uv.y > 1.0) {
        return wood;
    } else {
    	float k = fract(uv.x / 0.2);
    	float h = 0.3;
    	float a = mix(-1.0, 1.0, float(k < h));
    	return (a*(h-k)*0.3 < uv.y-h)?wood:vec3(0.01);
    }
}

// Function 144
void getGlyphAtIndex(int gi, out vec4 scan0123, out vec4 scan4567)
{
    if(gi==0){scan0123=vec4(0x18,0x3C,0x66,0x7E);scan4567=vec4(0x66,0x66,0x66,0x00);return;}
    if(gi==1){scan0123=vec4(0x7C,0x66,0x66,0x7C);scan4567=vec4(0x66,0x66,0x7C,0x00);return;}
    if(gi==2){scan0123=vec4(0x3C,0x66,0x60,0x60);scan4567=vec4(0x60,0x66,0x3C,0x00);return;}
    if(gi==3){scan0123=vec4(0x78,0x6C,0x66,0x66);scan4567=vec4(0x66,0x6C,0x78,0x00);return;}
    if(gi==4){scan0123=vec4(0x7E,0x60,0x60,0x78);scan4567=vec4(0x60,0x60,0x7E,0x00);return;}
    if(gi==5){scan0123=vec4(0x7E,0x60,0x60,0x78);scan4567=vec4(0x60,0x60,0x60,0x00);return;}
    if(gi==6){scan0123=vec4(0x3C,0x66,0x60,0x6E);scan4567=vec4(0x66,0x66,0x3C,0x00);return;}
    if(gi==7){scan0123=vec4(0x66,0x66,0x66,0x7E);scan4567=vec4(0x66,0x66,0x66,0x00);return;}
    if(gi==8){scan0123=vec4(0x3C,0x18,0x18,0x18);scan4567=vec4(0x18,0x18,0x3C,0x00);return;}
    if(gi==9){scan0123=vec4(0x1E,0x0C,0x0C,0x0C);scan4567=vec4(0x0C,0x6C,0x38,0x00);return;}
    if(gi==10){scan0123=vec4(0x66,0x6C,0x78,0x70);scan4567=vec4(0x78,0x6C,0x66,0x00);return;}
    if(gi==11){scan0123=vec4(0x60,0x60,0x60,0x60);scan4567=vec4(0x60,0x60,0x7E,0x00);return;}
    if(gi==12){scan0123=vec4(0x63,0x77,0x7F,0x6B);scan4567=vec4(0x63,0x63,0x63,0x00);return;}
    if(gi==13){scan0123=vec4(0x66,0x76,0x7E,0x6E);scan4567=vec4(0x66,0x66,0x66,0x00);return;}
    if(gi==14){scan0123=vec4(0x3C,0x66,0x66,0x66);scan4567=vec4(0x66,0x66,0x3C,0x00);return;}
    if(gi==15){scan0123=vec4(0x7C,0x66,0x66,0x66);scan4567=vec4(0x7C,0x60,0x60,0x00);return;}
    if(gi==16){scan0123=vec4(0x3C,0x66,0x66,0x66);scan4567=vec4(0x66,0x3C,0x0E,0x00);return;}
    if(gi==17){scan0123=vec4(0x7C,0x66,0x66,0x7C);scan4567=vec4(0x78,0x6C,0x66,0x00);return;}
    if(gi==18){scan0123=vec4(0x3C,0x66,0x60,0x3C);scan4567=vec4(0x06,0x66,0x3C,0x00);return;}
    if(gi==19){scan0123=vec4(0x7E,0x18,0x18,0x18);scan4567=vec4(0x18,0x18,0x18,0x00);return;}
    if(gi==20){scan0123=vec4(0x66,0x66,0x66,0x66);scan4567=vec4(0x66,0x66,0x3C,0x00);return;}
    if(gi==21){scan0123=vec4(0x66,0x66,0x66,0x66);scan4567=vec4(0x66,0x3C,0x18,0x00);return;}
    if(gi==22){scan0123=vec4(0x63,0x63,0x63,0x6B);scan4567=vec4(0x7F,0x77,0x63,0x00);return;}
    if(gi==23){scan0123=vec4(0x66,0x66,0x3C,0x18);scan4567=vec4(0x3C,0x66,0x66,0x00);return;}
    if(gi==24){scan0123=vec4(0x66,0x66,0x66,0x3C);scan4567=vec4(0x18,0x18,0x18,0x00);return;}
    if(gi==25){scan0123=vec4(0x7E,0x06,0x0C,0x18);scan4567=vec4(0x30,0x60,0x7E,0x00);return;}
    if(gi==26){scan0123=vec4(0x3C,0x66,0x6E,0x76);scan4567=vec4(0x66,0x66,0x3C,0x00);return;}
    if(gi==27){scan0123=vec4(0x18,0x18,0x38,0x18);scan4567=vec4(0x18,0x18,0x7E,0x00);return;}
    if(gi==28){scan0123=vec4(0x3C,0x66,0x06,0x0C);scan4567=vec4(0x30,0x60,0x7E,0x00);return;}
    if(gi==29){scan0123=vec4(0x3C,0x66,0x06,0x1C);scan4567=vec4(0x06,0x66,0x3C,0x00);return;}
    if(gi==30){scan0123=vec4(0x06,0x0E,0x1E,0x66);scan4567=vec4(0x7F,0x06,0x06,0x00);return;}
    if(gi==31){scan0123=vec4(0x7E,0x60,0x7C,0x06);scan4567=vec4(0x06,0x66,0x3C,0x00);return;}
    if(gi==32){scan0123=vec4(0x3C,0x66,0x60,0x7C);scan4567=vec4(0x66,0x66,0x3C,0x00);return;}
    if(gi==33){scan0123=vec4(0x7E,0x66,0x0C,0x18);scan4567=vec4(0x18,0x18,0x18,0x00);return;}
    if(gi==34){scan0123=vec4(0x3C,0x66,0x66,0x3C);scan4567=vec4(0x66,0x66,0x3C,0x00);return;}
    if(gi==35){scan0123=vec4(0x3C,0x66,0x66,0x3E);scan4567=vec4(0x06,0x66,0x3C,0x00);return;}
    if(gi==36){scan0123=vec4(0x00,0x00,0x00,0x00);scan4567=vec4(0x00,0x18,0x18,0x00);return;}
    if(gi==37){scan0123=vec4(0x00,0x66,0x3C,0xFF);scan4567=vec4(0x3C,0x66,0x00,0x00);return;}    
    scan0123 = vec4(0.);scan4567 = vec4(0.);
}

// Function 145
vec3 SphereTexture(sampler2D tex,vec3 normal) {
     float u = atan(normal.z, normal.x) / PI * 2.0;
     float v = asin(normal.y) / PI * 2.0;
     return texture(tex,vec2(u,v)).rgb;
}

// Function 146
float triTexture(in vec2 pos)
{
    vec2 pos2 = pos;
    pos2.x = pos.x  - pos.y * 0.5;
    pos2.y = pos.y / sqrt(3.0) / 0.5;
    
    float ind = 1.0;
    
    if(mod(pos2.x, 1.0) > 0.5)
        ind *= -1.0;
    
    if(mod(pos2.y, 1.0) > 0.5)
        ind *= -1.0;
    
    if(mod(pos2.y, 1.0) > 1.0 - mod(pos2.x, 1.0))
        ind *= -1.0;
    
    ind = max(ind, 0.0);
    
    return ind;
}

// Function 147
void glyph_1()
{
  MoveTo(0.5*x+1.7*y);
  LineTo(x+2.0*y);
  LineTo(x);
  MoveTo(0.5*x);
  LineTo(1.5*x);
}

// Function 148
float textureNoise(vec3 uv)
{
	float c = (linearRand(uv * 1.0) * 32.0 +
			   linearRand(uv * 2.0) * 16.0 + 
			   linearRand(uv * 4.0) * 8.0 + 
			   linearRand(uv * 8.0) * 4.0) / 32.0;
	return c * 0.5 + 0.5;
}

// Function 149
void glyph_5()
{
  MoveTo(x*0.2+y*1.1);
  Bez3To(x*0.7+y*1.5,x*1.8+y*1.4,x*1.8+y*0.65);
  Bez3To(x*1.8-y*0.2,x*0.4-y*0.2,x*0.1+y*0.3);
  MoveTo(x*0.2+y*1.1);
  RLineTo(y*0.9);
  RLineTo(x*1.5);
}

// Function 150
bool drawText(int textNumber, ivec2 completeOffset, ivec2 uv, bool rightJustified)
{
    ivec4 offsets = unpack(textOffsets[textNumber]);

    print_pos = LINE_0_OFFSET(offsets.y);
    bool hit = drawLine(offsets.x, uv);

    if ((!hit) && (offsets.z > -1))
    {
        print_pos = LINE_1_OFFSET(offsets.w);
        hit = drawLine(offsets.z, uv);
    }

    return hit;
}

// Function 151
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

// Function 152
float checkersTexture( in vec3 p )
{
    vec3 q = floor(p);
    return mod( q.x+q.y+q.z, 2.0 );
}

// Function 153
vec4 text(vec2 pos){
	vec4 c= vec4(0.0);
    tp += pos;
    _H _Y _P _O _T _H _E _T _E
    return c;
}

// Function 154
void TextureEnvBlured3(in vec3 N, in vec3 Rv, out vec3 iblDiffuse, out vec3 iblSpecular) {
    vec3 irradiance = vec3(0.0);   
    
    vec2 ts = vec2(textureSize(reflectTex, 0));
    float maxMipMap = log2(max(ts.x, ts.y));
    float lodBias = maxMipMap-7.0;    
    
    // tangent space calculation from origin point
    vec3 up    = vec3(0.0, 1.0, 0.0);
    vec3 right = cross(up, N);
    up            = cross(N, right);
       
    float sampleDelta = PI / 75.0;
    float nrSamples = 0.0f;
    for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // spherical to cartesian (in tangent space)
            vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N; 

            irradiance += sampleReflectionMap(sampleVec, lodBias) * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    iblDiffuse = PI * irradiance * (1.0 / float(nrSamples));    
}

// Function 155
ivec2 raw_text_uv(vec2 fragCoord)			{ return ivec2(floor(fragCoord)); }

// Function 156
float perlinTexture(in vec2 pos)
{
    const float N = 5.0;
    
    vec2 pos2 = pos;
    
    float ind = 0.0;
    float ampl = 1.0;
    
    for(float n = 0.0; n < N; n ++)
    {
    	ind += texture(iChannel0, pos / iChannelResolution[0].x).x * ampl;
    
        ampl *= 0.5;
        pos *= 2.0;
    }
    
    return ind / (1.0 - pow(0.5, N+1.0)) * 0.5;
}

// Function 157
vec4 drawSliderText(in vec2 uv, float mode, in int submenu, int _mainTex)
{
    vec4 tcol = vec4(0.);
    if (in_zone(mode, APPL_UI))
    {
        float lsize = 0.02;
        float y_offset = lsize*2.;
        float x_offset = 0.5*(SELECTION_RADIUS_MAX.x + SELECTION_RADIUS_MIN.x) - lsize*0.5 - SELECTION_RADIUS_MIN.x;
        
        //vec2 sstart = vec2(SELECTION_SHAPE_MIN.x + 4.*x_offset, SELECTION_SHAPE_MIN.y - y_offset);
        //vec2 ovstart = vec2(SELECTION_OVERRIDE_MIN.x + 4.*x_offset, SELECTION_SHAPE_MIN.y - y_offset);
    
        if (submenu == MENU_OPT_TOOLS)
        {
            vec2 rstart = vec2(SELECTION_RADIUS_MIN.x + x_offset, SELECTION_RADIUS_MAX.y - y_offset);
            vec2 brushstart = vec2(SELECTION_BRUSH_MIN.x + x_offset, SELECTION_BRUSH_MAX.y - y_offset);
            vec2 blendstart = vec2(SELECTION_BLEND_FACTOR_MIN.x + x_offset, SELECTION_BLEND_FACTOR_MAX.y - y_offset);
            tcol += drawTextVertical(uv, rstart, lsize, vec2[10](_R, _a, _d, _i, _u, _s, _X, _X, _X, _X), 6);
            tcol += drawTextVertical(uv, blendstart, lsize, vec2[10](_B, _l, _e, _n, _d, _ , _F, _a, _c, _t), 5);
            tcol += drawTextVertical(uv, brushstart, lsize, vec2[10](_B, _r, _u, _s, _h, _X, _X, _X, _X, _X), 5);
        }
        else if (submenu == MENU_OPT_COLOR)
        {
            
            vec2 castart = vec2(SELECTION_COLOR_A_MIN.x + x_offset, SELECTION_COLOR_A_MIN.y + y_offset);
            vec2 cbstart = vec2(SELECTION_COLOR_B_MIN.x + x_offset, SELECTION_COLOR_B_MIN.y + y_offset);
            vec2 ccstart = vec2(SELECTION_COLOR_C_MIN.x + x_offset, SELECTION_COLOR_C_MIN.y + y_offset);
            vec2 heightstart = vec2(SELECTION_COLOR_A_MIN.x + x_offset, SELECTION_COLOR_A_MAX.y - y_offset);
            vec2 specstart = vec2(SELECTION_COLOR_A_MIN.x + 0.002, SELECTION_COLOR_A_MAX.y - .6*y_offset);
            //vec2 shinestart = vec2(SELECTION_COLOR_B_MIN.x + 0.0022, SELECTION_COLOR_B_MAX.y - .5*y_offset);
            
            if (_mainTex == DIFFUSE_MAP)
            {
                tcol += drawLetter(uv, castart, lsize, _R);
            	tcol += drawLetter(uv, cbstart, lsize, _G);
            	tcol += drawLetter(uv, ccstart, lsize, _B);
            }
            else if (_mainTex == HEIGHT_MAP)
            {
                tcol += drawTextVertical(uv, heightstart, lsize, vec2[10](_H, _e, _i, _g, _h, _t, _X, _X, _X, _X), 6);
            }
            else if (_mainTex == SPECULAR_MAP)
            {
                tcol += drawTextVertical(uv, specstart, lsize*.8, vec2[10](_S, _p, _e, _c, _u, _l, _a, _r, _X, _X), 8);
                //tcol += drawTextVertical(uv, shinestart, lsize*.76, vec2[10](_S, _h, _i, _n, _i, _n, _e, _s, _s, _X), 9);
            }
        }
        else if (submenu == MENU_OPT_TEXTURE)
        {
            vec2 mainTexStart = vec2(SELECTION_TEXTURE_MIN.x + x_offset, SELECTION_TEXTURE_MIN.y - y_offset);
			vec2 secTexStart = vec2(SELECTION_SECONDARY_TEXTURE_MIN.x + x_offset, SELECTION_SECONDARY_TEXTURE_MIN.y - y_offset);
            vec2 loadTexStart = vec2(SELECTION_LOAD_TEXTURE_MIN.x + x_offset, SELECTION_LOAD_TEXTURE_MIN.y + .5*y_offset);
            vec2 blendTexStart = vec2(SELECTION_TEXTURE_ALPHA_MIN.x + x_offset, SELECTION_TEXTURE_ALPHA_MAX.y - y_offset);
            
            tcol += drawTextVertical(uv, mainTexStart, lsize, vec2[10](_M, _a, _i, _n, _X, _X, _X, _X, _X, _X), 4);
            tcol += drawTextVertical(uv, secTexStart, lsize, vec2[10](_S, _e, _c, _o, _n, _d, _a, _r, _y, _X), 9);
            tcol += drawTextHorizontal(uv, loadTexStart, lsize, vec2[10](_L, _o, _a, _d, _X, _X, _X, _X, _X, _X), 4);
            tcol += drawTextVertical(uv, blendTexStart, lsize, vec2[10](_B, _l, _e, _n, _d, _ , _F, _a, _c, _t), 5);
        }
		else if (submenu == MENU_OPT_3D)
        {
            vec2 starsStart = vec2(SELECTION_STARS_MAX.x + x_offset, SELECTION_STARS_MIN.y);
            tcol += drawTextHorizontal(uv, starsStart, lsize, vec2[10](_S, _t, _a, _r, _s, _X, _X, _X, _X, _X), 5);
        }
        
    }
    if (in_zone(mode, APPL_TEXTURES))
    {
        float lsize = 0.02;
        float y_offset = lsize*2.;
        float x_offset = .001;
        vec2 hideUiStart = vec2(SELECTION_HIDE_UI_MIN.x + x_offset, SELECTION_HIDE_UI_MIN.y + .5*y_offset);
        
        tcol += drawTextHorizontal(uv, hideUiStart, lsize, vec2[10](_1, _0, _s, _, _h, _i, _d, _e, _X, _X), 8);
    }
    return tcol;
}

// Function 158
float FontTexDf (vec3 p)
{
  vec3 tx;
  float d;
  int ic;
  ic = GetTxChar (p.xy);
  if (ic != 0) {
    tx = texture (txFnt, mod ((vec2 (mod (float (ic), 16.),
       15. - floor (float (ic) / 16.)) + fract (p.xy)) * (1. / 16.), 1.)).gba - 0.5;
    qnTex = vec2 (tx.r, - tx.g);
    d = tx.b + 1. / 256.;
  } else d = 1.;
  return d;
}

// Function 159
vec4 ascii2glyph(int a){
 if(a<16)return gOpr[a];
 if(a<32)return gNum[a-16];
 return mix(gUpp[a-32],gLow[a-64],step(64.,float(a)));
 if(a-64<0)return gUpp[a-32];return gLow[a-64];}

// Function 160
mat3 glyph_8_9_numerics(float g) {
    GLYPH(32) 0);
    GLYPH(42)000001000,000101010,000011100,000111110,000011100,000101010,000001000,0,0);
    GLYPH(46)000000000,000000000,000000000,000000000,000000000,000011000,000011000,0,0);
    // numerics  ==================================================
    GLYPH(49)000011000,000001000,000001000,000001000,000001000,000001000,000011100,0,0);
    GLYPH(50)000111100,001000010,000000010,000001100,000110000,001000000,001111110,0,0);
    GLYPH(51)000111100,001000010,000000010,000011100,000000010,001000010,000111100,0,0);
    GLYPH(52)001000100,001000100,001000100,000111110,000000100,000000100,000000100,0,0);
    GLYPH(53)001111110,001000000,001111000,000000100,000000010,001000100,000111000,0,0);
    GLYPH(54)000111100,001000010,001000000,001011100,001100010,001000010,000111100,0,0);
    GLYPH(55)000111110,001000010,000000010,000000100,000000100,000001000,000001000,0,0);
    GLYPH(56)000111100,001000010,001000010,000111100,001000010,001000010,000111100,0,0);
    GLYPH(57)000111100,001000010,001000010,000111110,000000010,000000010,000111100,0,0);
    GLYPH(58)000111100,000100100,001001010,001010010,001010010,000100100,000111100,0,0);
    return mat3(0);
}

// Function 161
vec4 put_text_drawmap(vec4 col, vec2 uv, vec2 pos, float scale)
{
	float unit = asp * scale * 0.1;
    float h = 0.;
    vec2 sc = vec2(unit, unit*0.8);
    
    // S
    h = max(h, word_map(uv, pos, 83, sc));
    // h
    h = max(h, word_map(uv, pos+vec2(unit*0.4, 0.), 104, sc));
    // o
    h = max(h, word_map(uv, pos+vec2(unit*0.8, 0.), 111, sc));
    // w
    h = max(h, word_map(uv, pos+vec2(unit*1.2, 0.), 119, sc));
    // M
    h = max(h, word_map(uv, pos+vec2(unit*2.0, 0.), 77, sc));
    // a
    h = max(h, word_map(uv, pos+vec2(unit*2.4, 0.), 97, sc));
    // p
    h = max(h, word_map(uv, pos+vec2(unit*2.8, 0.), 112, sc));
    
    
    col = mix(col, vec4(1.-vec3(h), 1.), h);
    
    return col;
}

// Function 162
vec4 NumFont_Char( vec2 vTexCoord, float fDigit )
{
    float fOutline = 0.0;
    float f00 = NumFont_BinChar( vTexCoord + vec2(-1,-1), fDigit );
    float f10 = NumFont_BinChar( vTexCoord + vec2( 0,-1), fDigit );
    float f20 = NumFont_BinChar( vTexCoord + vec2( 1,-1), fDigit );
        
    float f01 = NumFont_BinChar( vTexCoord + vec2(-1, 0), fDigit );
    float f11 = NumFont_BinChar( vTexCoord + vec2( 0, 0), fDigit );
    float f21 = NumFont_BinChar( vTexCoord + vec2( 1, 0), fDigit );
        
    float f02 = NumFont_BinChar( vTexCoord + vec2(-1, 1), fDigit );
    float f12 = NumFont_BinChar( vTexCoord + vec2( 0, 1), fDigit );
    float f22 = NumFont_BinChar( vTexCoord + vec2( 1, 1), fDigit );
        
    float fn1 = NumFont_BinChar( vTexCoord + vec2(-2, 0), fDigit );
    float fn2 = NumFont_BinChar( vTexCoord + vec2(-2, 1), fDigit );
    
    float fn3 = NumFont_BinChar( vTexCoord + vec2(-2, 2), fDigit );
    float f03 = NumFont_BinChar( vTexCoord + vec2(-1, 2), fDigit );
    float f13 = NumFont_BinChar( vTexCoord + vec2( 0, 2), fDigit );
        
    float fOutlineI = min( 1.0, f00 + f10 + f20 + f01 + f11 + f21 + f02 + f12 + f22 );
    float fShadow = min( 1.0, fn1 + f01 + f21 + fn2 + f02 + f12 + fn3 + f03 + f13 );

    float nx = f00 * -1.0 + f20 * 1.0
             + f01 * -2.0 + f21 * 2.0
         	 + f02 * -1.0 + f22 * 1.0;
        
    float ny = f00 * -1.0 + f02 * 1.0
             + f10 * -2.0 + f12 * 2.0
         	 + f20 * -1.0 + f22 * 1.0;
    
    vec3 n = normalize( vec3( nx, ny, 0.1 ) );
    
    vec3 vLight = normalize( vec3( 0.5, -1.0, 0.5 ) );
    
    float NdotL = dot( n, vLight ) * 0.25 + 0.75;
    NdotL = sqrt(NdotL);
    
    if ( (fOutlineI + fShadow) <= 0.0 )
    {
        return vec4(0.0);
    }

    vec4 vResult = vec4(1.0);
    
    if ( fShadow > 0.0 )
    {
        vResult.xyz = vec3(0.2);
    }

    if ( fOutlineI > 0.0 )
    {
	    vec3 vDiff = vec3(0.5,0,0);
        
        if ( f11 > 0.0 )
        {
            vDiff = vec3(1,0,0) * NdotL;
        }
        vResult.rgb = vDiff;
    }
    
    return vResult;
}

// Function 163
float TriplanarTexture(vec3 pos, vec3 n)
{
    pos = TransformPosition(pos);
    n = abs(n);
    float t0 = texture(iChannel1, pos.yz).x * n.x;
    float t1 = texture(iChannel1, pos.zx).y * n.y;
    float t2 = texture(iChannel1, pos.xy).z * n.z;
    
    return t0 + t1 + t2; 
}

// Function 164
vec3 TextureCoordsToPoint(in ivec2 coords){
    int size = 32;
    const int squareSide = 4;
    vec3 p;
    p.xz = mod(vec2(coords.xy),float(size));
    
    ivec2 index = coords.xy/size;
    p.y = float(index.y*squareSide+index.x);
    return p;
}

// Function 165
vec3 checkersTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )
{
    // filter kernel
    vec2 w = max(abs(ddx), abs(ddy)) + 0.01;  
    // analytical integral (box filter)
    vec2 i = 2.0*(abs(fract((p-0.5*w)/2.0)-0.5)-abs(fract((p+0.5*w)/2.0)-0.5))/w;
    // xor pattern
    float v = 0.5 - 0.5*i.x*i.y;
    
    return v * vec3(0.1) + vec3(0.8);
}

// Function 166
void CreditText(inout vec3 color, vec2 p, in AppState s)
{        
    vec2 scale = vec2(4., 8.);
    vec2 t = floor(p / scale);   
    
    uint v = 0u;    
	v = t.y == 0. ? ( t.x < 4. ? 1246186324u : ( t.x < 8. ? 959524914u : ( t.x < 12. ? 2037588026u : ( t.x < 16. ? 1747481710u : ( t.x < 20. ? 1769235753u : ( t.x < 24. ? 539369571u : ( t.x < 28. ? 1702258020u : ( t.x < 32. ? 1836085100u : ( t.x < 36. ? 544501349u : ( t.x < 40. ? 1293973858u : ( t.x < 44. ? 1634231145u : ( t.x < 48. ? 1816862828u : 29551u ) ) ) ) ) ) ) ) ) ) ) ) : v;
	v = t.x >= 0. && t.x < 52. ? v : 0u;    
    
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

// Function 167
bool glyph(const in vec4 p) {
    float ch = p.w;
    if((ch - s_)<.4) return false; //space

    vec2 sl = mod(p.xy,8.);
    float x = sl.x-1.05,   //used by macros
          y = sl.y,
		  b = 0.;
    
    // OMG I think I inadvertently figured out why VIC-20 had wonky pixels(!)
	// likely artifacts of mains noise on comparator(s) in the rasterizer and demodulator
    // tweak this for sub-pixel VDU effect  (US mains hum 60hz, UK 50hz)
    // (could likely take it further with main noise bias in colour channels and interlacing)
#ifdef  VDU_NOISE
    float h = .07*(.3+abs(.3*sin(iTime)))*sin(50.*iTime);
#else
    float h = 0.;
#endif
/*
    if((ch - a_)<1.){
        glyphMaskY(b,h,y,0.5,x,00001000.);
        glyphMaskY(b,h,y,1.5,x,00101010.);
        glyphMaskY(b,h,y,2.5,x,00011100.);
        glyphMaskY(b,h,y,3.5,x,00111110.);
        glyphMaskY(b,h,y,4.5,x,00011100.);
        glyphMaskY(b,h,y,5.5,x,00101010.);
        glyphMaskY(b,h,y,6.5,x,00001000.);
        glyphMaskY(b,h,y,7.5,x,00000000.);
        return modMask(b,y+1.)>.0;
    }
*/
/*
    if((ch - a_)<1.){
             if(y<1.) return glyphMask(x, 00001000.);
        else if(y<2.) return glyphMask(x, 00101010.);
        else if(y<3.) return glyphMask(x, 00011100.);
        else if(y<4.) return glyphMask(x, 00011100.);
        else if(y<5.) return glyphMask(x, 00011100.);
        else if(y<6.) return glyphMask(x, 00101010.);
        else if(y<7.) return glyphMask(x, 00001000.);
        else if(y<8.) return glyphMask(x, 00000000.);
    }
*/    

#ifdef LOW_MEMORY
    #define  a_00101010 a_
    if((ch - a_00101010)<.1){
        MASK_1 00001000.);
        MASK_2 00101010.);
        MASK_3 00011100.);
        MASK_4 00111110.);
        MASK_5 00011100.);
        MASK_6 00101010.);
        MASK_7 00001000.);
        MASK_8 00000000.);
        MASK_END
                        }
#else
	// OK lets see if we can help out the compiler
    // try (partial) red/black tree on value of ch
    if(binDigit(ch,1.)<1.)   // fails on iPad ?! FIX: pow() has issues on iPad see binDigit()
    {
        if(binDigit(ch,2.)<1.) 
        {
            #define n8_00111000 n8_
            if((ch - n8_00111000)<.1){
                MASK_1 00111100.);
                MASK_2 01000010.);
                MASK_3 01000010.);
                MASK_4 00111100.);
                MASK_5 01000010.);
                MASK_6 01000010.);
                MASK_7 00111100.);
                MASK_8 00000000.);
                MASK_END
                    }
            #define  D_01000100 D_
            if((ch - D_01000100)<.1){
                MASK_1 11110000.);
                MASK_2 01001000.);
                MASK_3 01000100.);
                MASK_4 01000100.);
                MASK_5 01000100.);
                MASK_6 01001000.);
                MASK_7 11110000.);
                MASK_8 00000000.);
                MASK_END
                    }
            #define  T_01010100 T_
            if((ch - T_01010100)<.1){
                MASK_1 01111100.);
                MASK_2 00010000.);
                MASK_3 00010000.);
                MASK_4 00010000.);
                MASK_5 00010000.);
                MASK_6 00010000.);
                MASK_7 00010000.);
                MASK_8 00000000.);
                MASK_END
                    }
        } 
        else 
        { //2
            if(binDigit(ch,3.)<1.) 
            {
                #define  a_00101010 a_
                if((ch - a_00101010)<.1){
                    MASK_1 00001000.);
                    MASK_2 00101010.);
                    MASK_3 00011100.);
                    MASK_4 00111110.);
                    MASK_5 00011100.);
                    MASK_6 00101010.);
                    MASK_7 00001000.);
                    MASK_8 00000000.);
                    MASK_END
                        }
                #define n2_00110010 n2_
                if((ch - n2_00110010)<.1){
                    MASK_1 00111100.);
                    MASK_2 01000010.);
                    MASK_3 00000010.);
                    MASK_4 00001100.);
                    MASK_5 00110000.);
                    MASK_6 01000000.);
                    MASK_7 01111110.);
                    MASK_8 00000000.);
                    MASK_END
                        }
                #define  B_01000010 B_
                if((ch - B_01000010)<.1){
                    MASK_1 11111000.);
                    MASK_2 01000100.);
                    MASK_3 01000100.);
                    MASK_4 01111000.);
                    MASK_5 01000100.);
                    MASK_6 01000100.);
                    MASK_7 11111000.);
                    MASK_8 00000000.);
                    MASK_END
                        }
                #define  R_01010010 R_
                if((ch - R_01010010)<.1){
                    MASK_1 01111100.);
                    MASK_2 01000010.);
                    MASK_3 01000010.);
                    MASK_4 01111100.);
                    MASK_5 01001000.);
                    MASK_6 01000100.);
                    MASK_7 01000010.);
                    MASK_8 00000000.);
                    MASK_END
                        }
            } 
            else 
            {//3
                #define  d_00101110 d_
                if((ch - d_00101110)<.1){
                    MASK_1 00000000.);
                    MASK_2 00000000.);
                    MASK_3 00000000.);
                    MASK_4 00000000.);
                    MASK_5 00000000.);
                    MASK_6 00011000.);
                    MASK_7 00011000.);
                    MASK_8 00000000.);
                    MASK_END
                        }
                #define  F_01000110 F_
                if((ch - F_01000110)<.1){
                    MASK_1 01111110.);
                    MASK_2 01000000.);
                    MASK_3 01000000.);
                    MASK_4 01111000.);
                    MASK_5 01000000.);
                    MASK_6 01000000.);
                    MASK_7 01000000.);
                    MASK_8 00000000.);
                    MASK_END
                        }
                #define  V_01010110 V_
                if((ch - V_01010110)<.1){
                    MASK_1 01000010.);
                    MASK_2 01000010.);
                    MASK_3 01000010.);
                    MASK_4 01000010.);
                    MASK_5 00100100.);
                    MASK_6 00100100.);
                    MASK_7 00011000.);
                    MASK_8 00000000.);
                    MASK_END
                        }
            }
        }
    } 
    else // 1
    {
        if(binDigit(ch,2.)<1.) 
        {
            if(binDigit(ch,3.)<1.) 
            {
                #define  A_01000001 A_
                if((ch - A_01000001)<.1){
                    MASK_1 00011000.);
                    MASK_2 00100100.);
                    MASK_3 01000010.);
                    MASK_4 01111110.);
                    MASK_5 01000010.);
                    MASK_6 01000010.);
                    MASK_7 01000010.);
                    MASK_8 00000000.);
                    MASK_END
                        }
                #define  I_01001001 I_
                if((ch - I_01001001)<.1){
                    MASK_1 00111000.);
                    MASK_2 00010000.);
                    MASK_3 00010000.);
                    MASK_4 00010000.);
                    MASK_5 00010000.);
                    MASK_6 00010000.);
                    MASK_7 00111000.);
                    MASK_8 00000000.);
                    MASK_END
                        }
                #define  Y_01011001 Y_
                if((ch - Y_01011001)<.1){
                    MASK_1 01000100.);
                    MASK_2 01000100.);
                    MASK_3 01000100.);
                    MASK_4 00111000.);
                    MASK_5 00010000.);
                    MASK_6 00010000.);
                    MASK_7 00010000.);
                    MASK_8 00000000.);
                    MASK_END
                        }
            }
            else 
            { //3
                #define n5_00110101 n5_
                if((ch - n5_00110101)<.1){
                    MASK_1 01111110.);
                    MASK_2 01000000.);
                    MASK_3 01111000.);
                    MASK_4 00000100.);
                    MASK_5 00000010.);
                    MASK_6 01000100.);
                    MASK_7 00111000.);
                    MASK_8 00000000.);
                    MASK_END
                        }
                #define  E_01000101 E_
                if((ch - E_01000101)<.1){
                    MASK_1 01111110.);
                    MASK_2 01000000.);
                    MASK_3 01000000.);
                    MASK_4 01111000.);
                    MASK_5 01000000.);
                    MASK_6 01000000.);
                    MASK_7 01111110.);
                    MASK_8 00000000.);
                    MASK_END
                        }
                #define  M_01001101 M_
                if((ch - M_01001101)<.1){
                    MASK_1 01000010.);
                    MASK_2 01100110.);
                    MASK_3 01011010.);
                    MASK_4 01000010.);
                    MASK_5 01000010.);
                    MASK_6 01000010.);
                    MASK_7 01000010.);
                    MASK_8 00000000.);
                    MASK_END
                        }
            }
        }
        else 
        { //2
            #define n3_00110011 n3_
            if((ch - n3_00110011)<.1){
                MASK_1 00111100.);
                MASK_2 01000010.);
                MASK_3 00000010.);
                MASK_4 00011100.);
                MASK_5 00000010.);
                MASK_6 01000010.);
                MASK_7 00111100.);
                MASK_8 00000000.);
                MASK_END
                    }
            #define  C_01000011 C_
            if((ch - C_01000011)<.1){
                MASK_1 00111000.);
                MASK_2 01000100.);
                MASK_3 10000000.);
                MASK_4 10000000.);
                MASK_5 10000000.);
                MASK_6 01000100.);
                MASK_7 00111000.);
                MASK_8 00000000.);
                MASK_END
                    }
            #define  S_01010011 S_
            if((ch - S_01010011)<.1){
                MASK_1 00111100.);
                MASK_2 01000010.);
                MASK_3 01000000.);
                MASK_4 00111100.);
                MASK_5 00000010.);
                MASK_6 01000010.);
                MASK_7 00111100.);
                MASK_8 00000000.);
                MASK_END
                    }
        }
    }//1
    
#endif
    
	return false;
}

// Function 168
float text3(in vec2 uv) {
  const vec2 size = vec2(158.0, 8.0);
  int bit = int(ceil(uv.y * size.y));
  int xx = int(ceil(uv.x * size.x));
  float pixels = 0.0;
  if (bit >= 0 && bit <= 25 && xx >= 0 && xx < 158)
  	pixels = bitget(img_copyright[xx], bit);
  return pixels;
}

// Function 169
float GetProgramText(vec2 vPixelPos)
{     
        vec2 vCharCoord = floor(vPixelPos / 8.0);
       
        float fChar = GetProgramStringChar(vCharCoord.x);
       
        if(vCharCoord.y != 0.0)
                fChar = 32.0;
       
        return GetCharPixel(fChar, mod(vPixelPos, 8.0));
}

// Function 170
vec4 text_format( int index, vec4 params, uvec4 phrase, vec4 argv )
{
#define CHROUT(p,chr) if( (p) >= 0 && (p) < 4 ) result[p] = chr;
    vec4 result = vec4(0);
    int argc = 0;
    int nchars = 0;
    int nwords = int( phrase.w & TXT_FMT_LENGTH_MASK );
    for( int i = 0, n = nwords; i < n; ++i )
    {
        int pbase = nchars + 5 - 4 * index;
        int code = int( ( phrase[ i >> 2 ] >> ( ( ~i & 3 ) << 3 ) ) & 0xffu );
        if( code >= 0xf0 && argc < 4 )
        {
            // numeric conversion			integers	decimals
            // -------------------------------------------------
            // 0xf0 .. 0xf2: 	unsigned 	2,3,4		0
            // 0xf3 .. 0xf5: 	signed		2,3,4		0
            // 0xf6 .. 0xf8: 	signed		2,3,4		1
            // 0xf9 .. 0xfb: 	signed		2,3,4		2
            // 0xfc .. 0xfe: 	signed		2,3,4		4
            // 0xff: 			signed		2			6

            const int base = 10;
            float arg = argv[ argc ];
            int p = ( code - 0xf0 ) % 3 + 2;
            int m = ipow( base, p );
			int q = ( code - 0xf0 ) / 3;
            bool overflow = abs(arg) >= float(m);
            bool signed = q > 0;
            if( signed )
            {
                q = q * q / 4;
                int r = ipow( base, q );
                m *= r;
                arg *= float(r);
                arg += .5 * sign( arg );
            }
            else
                arg *= float( base );
            int a = int( arg );
            int k = pbase;
			int jend = min( TXT_FMT_MAX_LEN - nchars - int( signed ), p + q + int( q > 0 ) );
            bool dout = !signed;
            for( int j = -int( signed ); j < jend; ++j )
            {
                int digit = abs(a) / m;
            	float chr = 32.;
            	if( j == p )
                {
                    chr = 46.;
                    dout = true;
                }
                else
				if( !dout && arg < 0. && ( abs(a) * base >= m || j + 2 >= p ) )
                {
                    chr = 45.;
                    a *= base;
                    dout = true;
                }
                else
                if( overflow )
                {
                	chr = j >= 0 ? 42. : 32.;
                }
                else
				{
                    a -= sign(a) * m * digit;
                    a *= base;
                	if( digit > 0 || j + 1 >= p )
                        dout = true;
                    if( dout )
  	            		chr = float( digit ) + 48.;
				}
                CHROUT( k, chr );
                k++;
            }
            nchars += jend + int( signed );
            argc++;
        }
       	else
        if( code >= 0x80 && ( code & 0x7f ) < g_text_data.length() )
        {
            // word substitution for bytes 0x80..0xef
            uvec4 word = g_text_data[ code & 0x7f ];
            int wlen = min( TXT_FMT_MAX_LEN - nchars, int( word.w & 0xffu ) );
            if( pbase >= -wlen && pbase < 4 )
                for( int j = 0; j < wlen; ++j )
                {
                    int chr = int( ( word[ j >> 2 ] >> ( ( ~j & 3 ) << 3 ) ) & 0xffu );
                    if( i == 0 && j == 0 )
                        chr = int( chr ) & ~0x20;
                    CHROUT( pbase + j, float( chr ) );
                }
            nchars += wlen;
            if( i + 1 < nwords )
            { CHROUT( pbase + wlen, 32. ); nchars++; }
        }
        else
        if( code > 0 && nchars < TXT_FMT_MAX_LEN )
        {
            // literal character
            CHROUT( pbase, float( code ) ); nchars++;
        }
    }

    if( index == 0 )
    {
        result = params;
        if( ( phrase.w & TXT_FMT_FLAG_CENTER ) == TXT_FMT_FLAG_CENTER )
            result.x -= abs( result.w ) * float( nchars ) * TXT_FONT_SPACING / 2.;
        else
        if( ( phrase.w & TXT_FMT_FLAG_RIGHT ) == TXT_FMT_FLAG_RIGHT )
            result.x -= abs( result.w ) * float( nchars ) * TXT_FONT_SPACING;
    }
    else
    if( index == 1 )
    {
        result.x = float( nchars );
        if( ( phrase.w & TXT_FMT_FLAG_HUDCLIP ) == TXT_FMT_FLAG_HUDCLIP )
        	result.x = -result.x;
    }
	return result;
#undef CHROUT
}

// Function 171
mat3 glyph_8_9_uppercase(float g) {
    // uppercase ==================================================
    GLYPH(65)11000,100100,1000010,1111110,1000010,1000010,1000010,0,0);
    GLYPH(66)11111000,1000100,1000100,1111000,1000100,1000100,11111000,0,0);
    GLYPH(67)11100,100010,1000000,1000000,1000000,100010,11100,0,0);
    GLYPH(68)11111000,1000100,1000010,1000010,1000010,1000100,11111000,0,0);
    GLYPH(69)1111110,1000000,1000000,1111000,1000000,1000000,1111110,0,0);
    GLYPH(70)1111110,1000000,1000000,1111000,1000000,1000000,1000000,0,0);
    GLYPH(71)111100,1000010,1000000,1001110,1000010,1000010,111100,0,0);
    GLYPH(72)1000010,1000010,1000010,1111110,1000010,1000010,1000010,0,0);
    GLYPH(73)111000,10000,10000,10000,10000,10000,111000,0,0);
    GLYPH(74)1111110,100,100,100,100,1000100,111000,0,0);
    GLYPH(75)1000100,1001000,1010000,1110000,1001000,1000100,1000010,0,0);
    GLYPH(76)100000,100000,100000,100000,100000,100000,111111,0,0);
    GLYPH(77)1000010,1100110,1011010,1000010,1000010,1000010,1000010,0,0);
    GLYPH(78)1000010,1100010,1010010,1001010,1000110,1000010,1000010,0,0);
    GLYPH(79)111100,1000010,1000010,1000010,1000010,1000010,111100,0,0);
    GLYPH(80)11111100,1000010,1000010,1111100,1000000,1000000,1000000,0,0);
    GLYPH(81)111100,1000010,1000010,1000010,1001010,1000110,111101,0,0);
    GLYPH(82)1111100,1000010,1000010,1111100,1001000,1000100,1000010,0,0);
    GLYPH(83)111100,1000010,1000000,111100,10,1000010,111100,0,0);
    GLYPH(84)1111100,10000,10000,10000,10000,10000,10000,0,0);
    GLYPH(85)1000010,1000010,1000010,1000010,1000010,1000010,111100,0,0);
    GLYPH(86)1000010,1000010,1000010,1000010,100100,100100,11000,0,0);
    GLYPH(87)1000011,1000011,1000011,1000011,1011011,1100111,100100,0,0);
    GLYPH(88)1000100,1000100,1000100,111000,101000,1000100,1000100,0,0);
    GLYPH(89)1000100,1000100,1000100,111000,10000,10000,10000,0,0);
    GLYPH(90)1111110,10,100,11000,100000,1000000,1111110,0,0);
    return mat3(0);
}

// Function 172
float vt220Font(vec2 p, float c)
{
    if (c < 1.) return 0.;
    if(p.y > 16.){
        if(c > 2.) return 0.0;
		if(c > 1.) return l(17,1,9);
    }
    if(p.y > 14.){
		if(c > 16.) return l(15,3,8);
		if(c > 15.) return l(15,1,8);
		if(c > 14.) return l(15,1,3)+ l(15,7,9);
		if(c > 13.) return l(15,2,8);
		if(c > 12.) return l(15,1,9);
		if(c > 11.) return l(15,2,8);
		if(c > 10.) return l(15,1,3)+ l(15,6,8);
		if(c > 9.) return l(15,4,6);
        if(c > 8.) return l(15,2,4)+ l(15,5,7);
		if(c > 7.) return l(15,2,8);
		if(c > 6.) return l(15,2,8);
		if(c > 5.) return l(15,2,8);
		if(c > 4.) return l(15,2,9);
		if(c > 3.) return l(15,1,8);
		if(c > 2.) return l(15,2,9);
    }
    if(p.y > 12.){
		if(c > 16.) return l(13,2,4)+ l(13,7,9);
		if(c > 15.) return l(13,2,4)+ l(13,7,9);
		if(c > 14.) return l(13,1,3)+ l(13,7,9);
		if(c > 13.) return l(13,1,3)+ l(13,7,9);
		if(c > 12.) return l(13,1,3);
		if(c > 11.) return l(13,4,6);
		if(c > 10.) return l(13,2,4)+ l(13,5,9);
		if(c > 9.) return l(13,2,8);
		if(c > 8.) return l(13,2,4)+ l(13,5,7);
		if(c > 7.) return l(13,1,3)+ l(13,7,9);
		if(c > 6.) return l(13,1,3)+ l(13,7,9);
		if(c > 5.) return l(13,1,3)+ l(13,7,9);
		if(c > 4.) return l(13,1,3)+ l(15,2,9);
		if(c > 3.) return l(13,1,4)+ l(13,7,9);
		if(c > 2.) return l(13,1,3)+ l(13,6,9);
    }
    if(p.y > 10.){
		if(c > 16.) return l(11,1,3);
		if(c > 15.) return l(11,2,4)+ l(11,7,9);
		if(c > 14.) return l(11,1,9);
		if(c > 13.) return l(11,7,9);
		if(c > 12.) return l(11,2,5);
		if(c > 11.) return l(11,4,6);
		if(c > 10.) return l(11,3,5)+ l(11,6,8);
		if(c > 9.) return l(11,4,6)+ l(11,7,9);
		if(c > 8.) return l(11,1,8);
		if(c > 7.) return l(11,1,3)+ l(11,7,9);
		if(c > 6.) return l(11,1,3)+ l(11,7,9);
		if(c > 5.) return l(11,1,3)+ l(11,7,9);
		if(c > 4.) return l(11,1,3);
		if(c > 3.) return l(11,1,3)+ l(11,7,9);
		if(c > 2.) return l(11,2,9);
    }
    if(p.y > 8.){
		if(c > 16.) return l(9,1,3);
		if(c > 15.) return l(9,2,8);
		if(c > 14.) return l(9,1,3)+ l(9,7,9);
		if(c > 13.) return l(9,4,8);
		if(c > 12.) return l(9,4,8);
		if(c > 11.) return l(9,4,6);
		if(c > 10.) return l(9,4,6);
		if(c > 9.) return l(9,2,8);
		if(c > 8.) return l(9,2,4)+ l(9,5,7);
		if(c > 7.) return l(9,1,3)+ l(9,7,9);
		if(c > 6.) return l(9,1,3)+ l(9,7,9);
		if(c > 5.) return l(9,1,3)+ l(9,7,9);
		if(c > 4.) return l(9,1,3)+ l(9,7,9);
		if(c > 3.) return l(9,1,4)+ l(9,7,9);
		if(c > 2.) return l(9,7,9);
    }
    if(p.y > 6.){
		if(c > 16.) return l(7,1,3);
		if(c > 15.) return l(7,2,4)+ l(7,7,9);
		if(c > 14.) return l(7,2,4)+ l(7,6,8);
		if(c > 13.) return l(7,5,7);
		if(c > 12.) return l(7,7,9);
		if(c > 11.) return l(7,2,6);
		if(c > 10.) return l(7,2,4)+ l(7,5,7);
		if(c > 9.) return l(7,1,3)+ l(7,4,6);
		if(c > 8.) return l(7,1,8);
		if(c > 7.) return l(7,2,8);
		if(c > 6.) return l(7,2,8);
		if(c > 5.) return l(7,2,8);
		if(c > 4.) return l(7,2,8);
		if(c > 3.) return l(7,1,8);
		if(c > 2.) return l(7,2,8);
    }
    if(p.y > 4.){
		if(c > 16.) return l(5,2,4)+ l(5,7,9);
		if(c > 15.) return l(5,2,4)+ l(5,7,9);
		if(c > 14.) return l(5,3,7);
		if(c > 13.) return l(5,6,8);
		if(c > 12.) return l(5,1,3)+ l(5,7,9);
		if(c > 11.) return l(5,3,6);
		if(c > 10.) return l(5,1,5)+ l(5,6,8);
		if(c > 9.) return l(5,2,8);
		if(c > 8.) return l(5,2,4)+ l(5,5,7);
		if(c > 7.) return 0.;
		if(c > 6.) return 0.;
		if(c > 5.) return 0.;
		if(c > 4.) return 0.;
		if(c > 3.) return l(5,1,3);
		if(c > 2.) return 0.;
    }
    if(p.y > 2.){
		if(c > 16.) return l(3,3,8);
		if(c > 15.) return l(3,1,8);
		if(c > 14.) return l(3,4,6);
		if(c > 13.) return l(3,1,9);
		if(c > 12.) return l(3,2,8);
		if(c > 11.) return l(3,4,6);
		if(c > 10.) return l(3,2,4)+ l(3,7,9);
		if(c > 9.) return l(3,4,6);
		if(c > 8.) return l(3,2,4)+ l(3,5,7);
		if(c > 7.) return l(3,2,4)+ l(3,6,8);
		if(c > 6.) return l(3,1,3)+ l(3,4,7);
		if(c > 5.) return l(3,2,4)+ l(3,6,8);
		if(c > 4.) return 0.;
		if(c > 3.) return l(3,1,3);
		if(c > 2.) return 0.;
    }
    else{
		if(c > 7.) return 0.;
		if(c > 6.) return l(1,2,5)+ l(1,6,8);
    }
    return 0.0;      
}

// Function 173
void text(vec2 p, inout vec3 col)
{
    //vec3 t = 0.05*texture(iChannel1, p).xyz;
    //p.xy += vec2(t-0.025);    
    txt1(p, col, 6.0, 11.0);
    txt2(p, col, 10.0, 16.0);
    txt3(p, col, eStart+6.5,  eStart+9.0);
    txt4(p, col, eStart+8.0,  eStart+12.0);
    txt5(p, col, eStart+11.0, eStart+16.0);
}

// Function 174
bool IsTextureLit( uint iTexture )
{
    bool bLit = true;
    switch( iTexture )
    {
        default:
        	break;
        
        case TEX_LAVA1:
        case TEX_TELEPORT:
        case TEX_SKY1:
        case TEX_WATER1:
        case TEX_WATER2:
        	bLit = false;
       	 break;
        
    }
    
    return bLit;
}

// Function 175
void addTexture(inout vec4 col, in vec4 sprite) {
    col = mix(col, vec4(sprite.rgb, 1.), sprite.a);
}

// Function 176
float DrawText( vec2 fragCoord, int pageNo )
{
    vec2 vFontUV = fragCoord / iResolution.xy;
    vFontUV.y = 1.0 - vFontUV.y;
    vFontUV *= 15.0;
    vFontUV.x *= 3.0;
    
    vFontUV -= vec2(2.0, 1.0);
    
    vec2 vCharUV = vec2(0);
    
    int headingPage = pageNo * 2;
    int bodyPage = headingPage + 1;
        
    if ( pageNo < 9 )
    {
        PrintPage( vCharUV, vFontUV, headingPage );
        vFontUV *= 1.5;
        vFontUV.y -= 1.0;
        if ( pageNo == 8 )
        {
        	vFontUV.x -= 8.0;
        	vFontUV.y -= 13.0;
        }
        
        PrintPage( vCharUV, vFontUV, bodyPage );        
    }

    float fFont = textureLod( iChannelFont, vCharUV, 0.0 ).w;
	
	return fFont;
}

// Function 177
vec4 textureImproved( sampler2D tex, in vec2 res, in vec2 uv, in vec2 g1, in vec2 g2 )
{
	uv = uv*res + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*(3.0-2.0*fuv);
	uv = (uv - 0.5)/res;
	return textureGrad( tex, uv, g1, g2 );
}

// Function 178
vec3 BitmapFontTest( vec2 vUV, int testChar )
{
    vec3 col = vec3(0);

#if 1
    vec2 vPixelPos = vUV * vec2(kCharPixels);
    ivec2 iPixelPos = ivec2( vPixelPos );
    iPixelPos.y = 19 - iPixelPos.y;
    bool bResult = CharBitmap12x20( testChar, iPixelPos );
#else
    vec2 vPixelPos = vUV * vec2(kSmallCharPixels);    
    ivec2 iPixelPos = ivec2( vPixelPos );
    iPixelPos.y = 8 - iPixelPos.y;
    bool bResult = CharBitmap5x9( testChar, iPixelPos );
#endif
    
    if ( bResult )
    {
        col = vec3(1);
    }
    else
    {
        col = vec3(0);        
    }
    
    vec2 vSubPixelPos = fract( vPixelPos );
    
    if ( vSubPixelPos.x > 0.9 || vSubPixelPos.x < 0.1)
        col = vec3(0,0,1);
    if ( vSubPixelPos.y > 0.9 || vSubPixelPos.y < 0.1)
        col = vec3(0,0,1);
    
    return col;
}

// Function 179
int GetGlyph(int iterations, ivec2 glyphPos, int glyphLast, ivec2 glyphPosLast, ivec2 focusPos)
{ 
    if (glyphPos == focusPos)
        return GetFocusGlyph(iterations); // inject correct glyph     
            
    int seed = iterations + glyphPos.x * 313 + glyphPos.y * 411 + glyphPosLast.x * 557 + glyphPosLast.y * 121;
    return imod(RandInt(seed), glyphCount); 
}

// Function 180
vec3 SampleInterpolationTextureNearest (vec2 uv)
{
    vec2 pixel = clamp(floor(uv*2.0), 0.0, 1.0);
	return SampleInterpolationTexturePixel(pixel);
}

// Function 181
float text(vec2 uv)
{
    float col = 0.0;
    
    vec2 center = res/2.0;
    
    float hour = floor(iDate.w/60.0/60.0);
    float minute = floor(mod(iDate.w/60.0,60.0));
    float second = floor(mod(iDate.w,60.0));
    
    //Greeting Text
    
    print_pos = floor(center - vec2(STRWIDTH(17.0),STRHEIGHT(1.0))/2.0);
       
    col += char(ch_H,uv);
    col += char(ch_e,uv);
    col += char(ch_l,uv);
    col += char(ch_l,uv);
    col += char(ch_o,uv);
    col += char(ch_com,uv);
    
    col += char(ch_spc,uv);
    
    col += char(ch_S,uv);
    col += char(ch_h,uv);
    col += char(ch_a,uv);
    col += char(ch_d,uv);
    col += char(ch_e,uv);
    col += char(ch_r,uv);
    col += char(ch_t,uv);
    col += char(ch_o,uv);
    col += char(ch_y,uv);
    col += char(ch_exc,uv);
    
    //Date Text
    
    print_pos = vec2(2, 2.0 + STRHEIGHT(2.0));
    
    TEXT_MODE = INVERT;
    col += char(ch_D,uv);
    col += char(ch_a,uv);
    col += char(ch_t,uv);
    col += char(ch_e,uv);
    
    TEXT_MODE = NORMAL;
    col += char(ch_col,uv);
    
    TEXT_MODE = UNDERLINE;
    col += print_integer(iDate.z,2,uv);
    col += char(ch_lsl,uv);
    col += print_integer(iDate.y+1.0,2,uv);
    col += char(ch_lsl,uv);
    col += print_integer(iDate.x,4,uv);
    
    //Time Text
    
    print_pos = vec2(2, 2.0 + STRHEIGHT(1.0));
    
    TEXT_MODE = INVERT;
    col += char(ch_T, uv);
    col += char(ch_i, uv);
    col += char(ch_m, uv);
    col += char(ch_e, uv);
    
    TEXT_MODE = NORMAL;
    col += char(ch_col, uv);
    
    TEXT_MODE = UNDERLINE;

    col += print_integer(hour,2, uv);
    col += char(ch_col, uv);
    col += print_integer(minute,2, uv);
    col += char(ch_col, uv);
    col += print_integer(second,2, uv);
    
    //Resolution Text
    
    print_pos = vec2(2, 2.0 + STRHEIGHT(0.0));
    
    TEXT_MODE = INVERT;  
    col += char(ch_R,uv);
    col += char(ch_e,uv);
    col += char(ch_s,uv);
    col += char(ch_o,uv);
    col += char(ch_l,uv);
    col += char(ch_u,uv);
    col += char(ch_t,uv);
    col += char(ch_i,uv);
    col += char(ch_o,uv);
    col += char(ch_n,uv);
    
    TEXT_MODE = NORMAL;  
    col += char(ch_col,uv);
    
    TEXT_MODE = UNDERLINE;
    col += print_integer(iResolution.x,0,uv); 
    col += char(ch_x,uv);
    col += print_integer(iResolution.y,0,uv); 
    
    return col;
}

// Function 182
vec4 grassTexture(vec3 pos, vec3 nor)
{
    
    float g = texture(iChannel1, pos.xz*.5).x;
    float s = texture(iChannel1, pos.xz*.015).x*.2;
    
    
    vec3 flower = texture(iChannel2, pos.xz*.15).xyz;
    float rand = texture(iChannel1, pos.xz*.003).x;
    rand *= rand*rand;
    
    flower =pow(flower,vec3(8, 15, 5)) *10. * rand;
    vec4 mat = vec4(g*.05+s, g*.65, 0, g*.1);
    mat.xyz += flower;

    // Do the red ground lines...
    pos = fract(pos);
    mat = mix(mat, vec4(.2, 0,0,0), smoothstep(.05, .0,min(pos.x, pos.z))
              					  + smoothstep(.95, 1.,max(pos.x, pos.z)));

    
	return min(mat, 1.0);
}

// Function 183
void mainAnaglyph(out vec4 fragColor, in vec2 fragCoord )
{
    vec2 p = -1.0 + 2.0 * fragCoord.xy / iResolution.xy;
    p.x *= iResolution.x/iResolution.y;

    float ctime = iTime;
    // camera
    vec3 ro = 1.1*vec3(2.5*sin(0.25*ctime),1.0+1.0*cos(ctime*.13),2.5*cos(0.25*ctime));
    vec3 ww = normalize(vec3(0.0) - ro);
    vec3 uu = normalize(cross( vec3(0.0,1.0,0.0), ww ));
    vec3 vv = normalize(cross(ww,uu));
    //vec3 rd = normalize( p.x*uu + p.y*vv + 2.5*ww );
    vec3 ta = vec3(0.0);

	vec3 eyes_line = normalize(cross(vec3(0.0, 1.0, 0.0), ro));
	float d_eyes_2 = 0.08;
	vec3 ro_left = ro + d_eyes_2 * eyes_line;
	vec3 ro_right = ro - d_eyes_2 * eyes_line;
    vec3 ta_left = ta + d_eyes_2 * eyes_line;
    vec3 ta_right = ta - d_eyes_2 * eyes_line;


	// camera-to-world transformation
    mat3 ca_left  = setCamera(ro_left, ta_left, 0.0);
    mat3 ca_right = setCamera(ro_right, ta_right, 0.0);

    // rays' direction
	vec3 rd_left = ca_left * normalize( vec3(p.xy,1.0) );
	vec3 rd_right = ca_right * normalize( vec3(p.xy,1.0) );

    // render both eye
    vec4 col_left;
    mainVR(col_left, fragCoord, ro_left - vec3(0.0,1.0,2.5), rd_left);
    vec4 col_right;
    mainVR(col_right, fragCoord, ro_left - vec3(0.0,1.0,2.5), rd_right);

    //vec3 col = vec3( col_right.r, col_left.g, col_left.b);
    vec3 col = vec3( col_left.r, col_right.g, col_right.b);

    fragColor = vec4(col, 1.0);
}

// Function 184
void textureWall(in vec3 block, inout ray ray, inout rayMarchHit hit, inout vec3 colour, bool isReflection, in float time) {
    vec2 uv;
    vec3 absNormal = abs(hit.surfaceNormal);
    float scale = 2.0;

    if(absNormal.y-0.8 > absNormal.x && absNormal.y-0.8 > absNormal.z) {
        uv.xy=fract(hit.origin.xz*scale)-0.5;
    } else {
        if(absNormal.x > absNormal.z) {
            if(hit.surfaceNormal.x>0.0) {
                uv.x=1.0-fract((hit.origin.z)*scale);
            } else {
                uv.x=fract((hit.origin.z)*scale);
            }
        } else {
            if(hit.surfaceNormal.z>0.0) {
                uv.x=fract((hit.origin.x)*scale);
            } else {
                uv.x=1.0-fract((hit.origin.x)*scale);
            }
        }
        uv.y=fract(hit.origin.y*scale);
		uv-=0.5;
        //vec2 windowSize=vec2(1.0);
        vec2 windowSize=vec2(hash21(block.xy*39.195),hash21(block.xy*26.389))*0.7+0.2;

        if (windowSize.x > 0.8){
            windowSize.x=1.0;
        }

        if (windowSize.y > 0.8){
            windowSize.y=1.0;
        }

        float round=0.0;

        if (windowSize.x < 1.0 && windowSize.y < 1.0) {
            round = min(windowSize.x,windowSize.y) * hash21(block.xy*87.981);
        }

       if ( abs(uv.x*2.0) < windowSize.x+PARALLAX_WINDOW_SURROUND_THICKNESS && abs(uv.y*2.0) < windowSize.y+PARALLAX_WINDOW_SURROUND_THICKNESS) {
            float distance = sdBox(uv*2.0,windowSize-round)-round;
            if(distance < 0.0) {
                vec3 cell = floor(hit.origin*scale);
                bool on = (hash31(cell) + sin(time*0.5)*0.05) > 0.5;
                hit.materialId = MAT_WINDOW;
                if(on) {
                    float brightness = clamp(hash31(cell),0.1,1.0);
                    vec3 lightColour = clamp(hash33(cell)+0.5,0.0,1.0);
                    
                    #if defined(DEBUG_PARALLAX)
                        if(isReflection) {
                            colour=lightColour*3.0*brightness;
                        } else {
                            float distanceRatio=hit.distance/RAY_MAX_DISTANCE;
                            if(distanceRatio<0.5) {
                                textureParallaxWindow(block, ray, hit, uv, cell, lightColour, brightness, colour, time);
                                colour*=3.0;
                                if(distanceRatio>0.25) {
                                    colour=mix(colour,lightColour*3.0*brightness,(distanceRatio-0.25)*4.0);
                                }
                                //shade the edge of the glass a bit.
                                colour = mix(PARALLAX_WINDOW_SURROUND_COLOUR,colour,clamp(abs(distance*20.0),0.0,1.0));
                            } else {
                                colour=lightColour*3.0*brightness;
                            }
                        }
                    #else
                    	colour=lightColour*3.0*brightness;
                    #endif
                } else {
                    colour=vec3(0.0);
                }
            } else if(distance < PARALLAX_WINDOW_SURROUND_THICKNESS) {
                hit.materialId = MAT_WINDOW;
                colour=PARALLAX_WINDOW_SURROUND_COLOUR;
            } 
        }
    }

    if (hit.materialId != MAT_WINDOW){
        float concrete = getConcrete(hit.origin, hit.surfaceNormal, false);
        colour = hash33(block.xyx) * vec3(0.25,0.1,0.2) + 0.5;
        colour = clamp(colour,vec3(0.0),vec3(1.0));
        colour *= concrete;
    }
}

// Function 185
vec3 LitText(vec2 p)
{
    vec3 color = vec3(0.2,0.3,0.2);
    vec4 res = maskSharp(p);
    float shade = 0.0;
    if (res.w>0.0)
    {
        float x = (0.5+sin(iTime*0.75))*2.2;
        float y = (0.5+cos(iTime*1.35))*0.8;
        vec3 lightPos = vec3(x-1.1, y-0.5,0.25);
        vec3 toLight = lightPos - vec3(p, 0.0);
        vec3 normal = normalAt(p);
        color = res.xyz;
        color *= 0.35+clamp(dot(normalize(toLight), normal), 0.0, 1.0) / 0.65 ;;
        
    }
    return color;
}

// Function 186
vec3 texturef( in vec2 p )
{
	vec2 q = p;
	p = p*vec2(6.0,128.0);
	float f = 0.0;
    f += 0.500*noise( p ); p = p*2.02;
    f += 0.250*noise( p ); p = p*2.03;
    f += 0.125*noise( p ); p = p*2.01;
	f /= 0.875;
	
	vec3 col = 0.6 + 0.4*sin( f*2.5 + 1.0+vec3(0.0,0.5,1.0) );
	col *= 0.7 + 0.3*noise( 8.0*q.yx );
	col *= 0.8 + 0.2*clamp(2.0*noise(256.0*q.yx ),0.0,1.0);
    col *= vec3(1.0,0.65,0.5) * 0.85;
    return col;

}

// Function 187
vec4 grassTexture(vec3 pos, vec3 nor)
{
    
    float g = texture(iChannel1, pos.xz*.5).x;
    float s = texture(iChannel1, pos.xz*.015).x*.2;
    
    
    vec3 flower = texture(iChannel2, pos.xz*.15).xyz;
    float rand = texture(iChannel1, pos.xz*.003).x;
    rand *= rand*rand;
    
    flower =pow(flower,vec3(8, 15, 5)) *10. * rand;
    vec4 mat = vec4(g*.05+s, g*.65, 0, g*.1);
    mat.xyz += flower;
    
	return min(mat, 1.0);
}

// Function 188
vec4 draw_font8x8_number_0k( int k, int n, vec4 col, ivec2 pos, inout vec4 o, ivec2 iu ) {
    vec4 v = vec4( 0 ) ;
    int off = 0 ;
    if( n < 0 ) {
        v += draw_font8x8_char( _DASH, col, pos, o, iu ) ;
        n = 0 - n ; //freaking workaround for mac os bug!!!
        off = 8 ;
    }
    ivec2 iu2 = iu - pos ;
    if( iINSIDE( iu2, ivec2(off,0), ivec2(k*8,8) ) ) {
        int p = ( ( k*8 - 1 ) - iu2.x ) / 8, c ;
        //for( int i = 0 ; i < p ; ++ i ) n /= 10 ;
        //c = n % 10 ;
        int d = int( floor( pow( 10., float(p) ) ) ) ;
        c = ( n / d ) % 10 ;
        v += draw_font8x8_char( c + 1, col, pos + ivec2( (k-1-p) * 8, 0 ), o, iu ) ;
    }
    return( v ) ;
}

// Function 189
vec4 texture_Bicubic( sampler2D tex, vec2 t) {
    vec2 res = iChannelResolution[0].xy;
    vec2 p = res*t - .5, f = fract(p), i = floor(p);
    return spline( f.y, spline( f.x, SAM(-1,-1), SAM( 0,-1), SAM( 1,-1), SAM( 2,-1)),
                        spline( f.x, SAM(-1, 0), SAM( 0, 0), SAM( 1, 0), SAM( 2, 0)),
                        spline( f.x, SAM(-1, 1), SAM( 0, 1), SAM( 1, 1), SAM( 2, 1)),
                        spline( f.x, SAM(-1, 2), SAM( 0, 2), SAM( 1, 2), SAM( 2, 2)));
}

// Function 190
float linesTextureGradBox( in float p, in float ddx, in float ddy, int id )
{
    float N = 12.0;//float( 2 + 7*((id>>1)&3) );

    float w = max(abs(ddx), abs(ddy)) + 0.01;
    float a = p + 0.5*w;                        
    float b = p - 0.5*w;           
    return 1.0 - (floor(a)+min(fract(a)*N,1.0)-
                  floor(b)-min(fract(b)*N,1.0))/(N*w);
}

// Function 191
vec4 AntiAliasPointSampleTexture_None(vec2 uv, vec2 texsize) {	
	return texture(iChannel0, (floor(uv+0.5)+0.5) / texsize, -99999.0);
}

// Function 192
vec4 Text( uint[7] text, vec2 uv )
{
    uv /= iResolution.y/25.; // font height
    
    const float charWidth = .5; // proportional to height
    
    int charIndex = int(floor(uv.x/charWidth));
    if ( uv.y < 0. || uv.y >= 1. || charIndex < 0 || charIndex >= text.length()*4 )
        return vec4(0);
    
    uint char = text[charIndex/4];
    char = (char>>(8*(charIndex&3)))&0xffU;
    vec2 charuv = vec2( char&0xFU, 0xFU-(char>>4) );

    uv.x = fract(uv.x/charWidth)*charWidth;

    vec4 t = textureLod(iChannel1,(uv+charuv)/16.+vec2(1./64.,0), 0.);
    
    float s = 10./iResolution.y;

    return vec4(1,1,1,1) * smoothstep(.5+s,.5-s,t.w);
}

// Function 193
vec3 texture_surface(vec3 position, vec3 normal) {
    vec4 noise = texture(iChannel0, position.xz * 1.0 + normal.y * 0.1);
    
    vec3 col = mix(UNDERSIDE_COLOR, SURFACE_COLOR, normal.y + normal.x*normal.z);
    col = mix(col, noise.rgb, 0.5);
    
    return col;
}

// Function 194
float text_des(vec2 U) {
    initMsg;C(68);C(101);C(115);C(116);C(114);C(111);C(121);endMsg;
}

// Function 195
vec3 texture_value(const in texture_ t, const in vec3 p) {
    if (t.type == SOLID) {
	    return t.v;
    } else if (t.type == NOISE) {
        return vec3(.5*(1. + sin(t.v.x*p.x + 5.*fbm((t.v.x)*p, 7))));
    }
}

// Function 196
float texture(vec2 uv )
{
	float t = voronoi( uv * 8.0 + vec2(iTime) );
    t *= 1.0-length(uv * 2.0);
	
	return t;
}

// Function 197
float checkersTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )
{
    // filter kernel
    vec2 w = max(abs(ddx), abs(ddy)) + 0.01;  
    // analytical integral (box filter)
    vec2 i = 2.0*(abs(fract((p-0.5*w)/2.0)-0.5)-abs(fract((p+0.5*w)/2.0)-0.5))/w;
    // xor pattern
    return 0.5 - 0.5*i.x*i.y;                  
}

// Function 198
voxel decodeTextel(vec4 textel) {
	voxel o;
    o.id = textel.r;
    o.light.s = floor(mod(textel.g, 16.));
    o.light.t = floor(mod(textel.g / 16., 16.));
    o.hue = textel.b;
    return o;
}

// Function 199
float Glyph7(const in vec2 uv)
{
    const vec2 vP0 = vec2 ( 0.693 , 0.068 );
    const vec2 vP1 = vec2 ( 0.748 , 0.069 );
    const vec2 vP2 = vec2 ( 0.747 , 0.078 );
    const vec2 vP3 = vec2 ( 0.691 , 0.077 );
    
    return InQuad(vP0, vP1, vP2, vP3, uv);
}

// Function 200
float getLavaStoneTexture(vec2 uv, float f) {
    return getFlowSpots(uv, f);
}

// Function 201
vec3 sampledTexture( in vec2 p )
{
    vec3 color;    
    float u = p.x;
    float v = p.y;
    float f, pattern;

        // background
    color = mix(bg, white, crossing(tranScale(u, 0.5/6.75, 6.75), tranScale(v, 0.5/6.75, 6.75)));
    color = mix(color, grey, crossing(tranScale(u, 0.5/6.45, 6.45), tranScale(v, 0.5/6.45, 6.45)));
    
    // centerlines, just add as patterns do not intersect due to filter
    f = 1.0-rect(tranScale(u, 0.5/6.45, 6.45), tranScale(v, 0.5/6.45, 6.45));
    pattern = f * (line(tranScale(u, 1.0/0.2, 0.3)) +
                   line(tranScale(v, 1.0/0.2, 0.3)) +
                   line(tranScale(u, 1.0/0.2, -0.1)) +
                   line(tranScale(v, 1.0/0.2, -0.1)) );
	// dashed, just add as patterns do not intersect due to filter
    f = 1.0-rect(tranScale(u, 0.5/10.7, 10.7), tranScale(v, 0.5/10.7, 10.7));
    pattern += f * (line(tranScale(u, 1.0/0.15, 3.45))*dash(tranScale(v, 1.0/2.0, 1.0), 0.6) +
                    line(tranScale(v, 1.0/0.15, 3.45))*dash(tranScale(u, 1.0/2.0, 1.0), 0.6) +
                    line(tranScale(u, 1.0/0.15, -3.3))*dash(tranScale(v, 1.0/2.0, 1.0), 0.6) +
                    line(tranScale(v, 1.0/0.15, -3.3))*dash(tranScale(u, 1.0/2.0, 1.0), 0.6));
    // stop lines, add again
    pattern += rect(tranScale(u, 1.0/0.4, -9.55), tranScale(v, 1.0/5.55, -0.6)) +
               rect(tranScale(u, 1.0/0.4,  10.05), tranScale(v, 1.0/5.55, -0.6)) +
               rect(tranScale(u, 1.0/0.4, -9.55), tranScale(v, 1.0/5.55, 6.15)) +
               rect(tranScale(u, 1.0/0.4,  10.05), tranScale(v, 1.0/5.55, 6.15)) +
               rect(tranScale(v, 1.0/0.4, -9.55), tranScale(u, 1.0/5.55, -0.6)) +
               rect(tranScale(v, 1.0/0.4,  10.05), tranScale(u, 1.0/5.55, -0.6)) +
               rect(tranScale(v, 1.0/0.4, -9.55), tranScale(u, 1.0/5.55, 6.15)) +
               rect(tranScale(v, 1.0/0.4,  10.05), tranScale(u, 1.0/5.55, 6.15));
    // pedestrian crossing, add again
    pattern += rect(tranScale(v, 1.0/2.0,  9.05), tranScale(u, 1.0/5.55, 6.15))*dash(tranScale(u, 1.0/0.6, 5.925), 0.5) +
               rect(tranScale(v, 1.0/2.0,  9.05), tranScale(u, 1.0/5.55, -0.6))*dash(tranScale(u, 1.0/0.6, -0.825), 0.5) +
               rect(tranScale(v, 1.0/2.0,  -7.05), tranScale(u, 1.0/5.55, 6.15))*dash(tranScale(u, 1.0/0.6, 5.925), 0.5) +
               rect(tranScale(v, 1.0/2.0,  -7.05), tranScale(u, 1.0/5.55, -0.6))*dash(tranScale(u, 1.0/0.6, -0.825), 0.5) +
               rect(tranScale(u, 1.0/2.0,  9.05), tranScale(v, 1.0/5.55, 6.15))*dash(tranScale(v, 1.0/0.6, 5.925), 0.5) +
               rect(tranScale(u, 1.0/2.0,  9.05), tranScale(v, 1.0/5.55, -0.6))*dash(tranScale(v, 1.0/0.6, -0.825), 0.5) +
               rect(tranScale(u, 1.0/2.0,  -7.05), tranScale(v, 1.0/5.55, 6.15))*dash(tranScale(v, 1.0/0.6, 5.925), 0.5) +
               rect(tranScale(u, 1.0/2.0,  -7.05), tranScale(v, 1.0/5.55, -0.6))*dash(tranScale(v, 1.0/0.6, -0.825), 0.5);
    color = mix(color, white, pattern);

    return color;
}

// Function 202
mat3 glyph_8_9(float g) {
    GLYPH(032.5) 0); 
    GLYPH(042.5) 
        00001000.,
        00101010.,
        00011100.,
        00111110.,
        00011100.,
        00101010.,
        00001000.,
        00000000.,
        00000000.);
    GLYPH(046.5) 
        00000000.,
        00000000.,
        00000000.,
        00000000.,
        00000000.,
        00011000.,
        00011000.,
        00000000.,
        00000000.);
    GLYPH(049.5) 
        00011000.,
        00001000.,
        00001000.,
        00001000.,
        00001000.,
        00001000.,
        00011100.,
        00000000.,
        00000000.);
    GLYPH(050.5) 
        00111100.,
        01000010.,
        00000010.,
        00001100.,
        00110000.,
        01000000.,
        01111110.,
        00000000.,
        00000000.);
    GLYPH(051.5) 
        00111100.,
        01000010.,
        00000010.,
        00011100.,
        00000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(052.5) 
        01000100.,
        01000100.,
        01000100.,
        00111110.,
        00000100.,
        00000100.,
        00000100.,
        00000000.,
        00000000.);
    GLYPH(053.5) 
        01111110.,
        01000000.,
        01111000.,
        00000100.,
        00000010.,
        01000100.,
        00111000.,
        00000000.,
        00000000.);
    GLYPH(054.5) 
        00111100.,
        01000010.,
        01000000.,
        01011100.,
        01100010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(055.5) 
        00111110.,
        01000010.,
        00000010.,
        00000100.,
        00000100.,
        00001000.,
        00001000.,
        00000000.,
        00000000.);
    GLYPH(056.5) 
        00111100.,
        01000010.,
        01000010.,
        00111100.,
        01000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(057.5) 
        00111100.,
        01000010.,
        01000010.,
        00111110.,
        00000010.,
        00000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(058.5) 
        00111100.,
        00100100.,
        01001010.,
        01010010.,
        01010010.,
        00100100.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(059.5) 0); 
    GLYPH(060.5) 0); 
    GLYPH(061.5) 0); 
    GLYPH(062.5) 0); 
    GLYPH(063.5) 0); 
    GLYPH(064.5) 0); 
    GLYPH(065.5) 
        00011000.,
        00100100.,
        01000010.,
        01111110.,
        01000010.,
        01000010.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(066.5) 
        11111000.,
        01000100.,
        01000100.,
        01111000.,
        01000100.,
        01000100.,
        11111000.,
        00000000.,
        00000000.);
    GLYPH(067.5) 
        00011100.,
        00100010.,
        01000000.,
        01000000.,
        01000000.,
        00100010.,
        00011100.,
        00000000.,
        00000000.);
    GLYPH(068.5) 
        11111000.,
        01000100.,
        01000010.,
        01000010.,
        01000010.,
        01000100.,
        11111000.,
        00000000.,
        00000000.);
    GLYPH(069.5) 
        01111110.,
        01000000.,
        01000000.,
        01111000.,
        01000000.,
        01000000.,
        01111110.,
        00000000.,
        00000000.);
    GLYPH(070.5) 
        01111110.,
        01000000.,
        01000000.,
        01111000.,
        01000000.,
        01000000.,
        01000000.,
        00000000.,
        00000000.);
    GLYPH(071.5) 
        00111100.,
        01000010.,
        01000000.,
        01001110.,
        01000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(072.5) 
        01000010.,
        01000010.,
        01000010.,
        01111110.,
        01000010.,
        01000010.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(073.5) 
        00111000.,
        00010000.,
        00010000.,
        00010000.,
        00010000.,
        00010000.,
        00111000.,
        00000000.,
        00000000.);
    GLYPH(074.5) 
        01111110.,
        00000100.,
        00000100.,
        00000100.,
        00000100.,
        01000100.,
        00111000.,
        00000000.,
        00000000.);
    GLYPH(075.5) 
        01000100.,
        01001000.,
        01010000.,
        01110000.,
        01001000.,
        01000100.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(076.5) 
        00100000.,
        00100000.,
        00100000.,
        00100000.,
        00100000.,
        00100000.,
        00111111.,
        00000000.,
        00000000.);
    GLYPH(077.5) 
        01000010.,
        01100110.,
        01011010.,
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(078.5) 
        01100010.,
        01100010.,
        01010010.,
        01001010.,
        01000110.,
        01000010.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(079.5) 
        00111100.,
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(080.5) 
        11111100.,
        01000010.,
        01000010.,
        01111100.,
        01000000.,
        01000000.,
        01000000.,
        00000000.,
        00000000.);
    GLYPH(081.5) 
        00111100.,
        01000010.,
        01000010.,
        01000010.,
        01001010.,
        01000110.,
        00111101.,
        00000000.,
        00000000.);
    GLYPH(082.5) 
        01111100.,
        01000010.,
        01000010.,
        01111100.,
        01001000.,
        01000100.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(083.5) 
        00111100.,
        01000010.,
        01000000.,
        00111100.,
        00000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(084.5) 
        01111100.,
        00010000.,
        00010000.,
        00010000.,
        00010000.,
        00010000.,
        00010000.,
        00000000.,
        00000000.);
    GLYPH(085.5) 
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(086.5) 
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        00100100.,
        00100100.,
        00011000.,
        00000000.,
        00000000.);
    GLYPH(087.5) 
        01000001.,
        01000001.,
        01000001.,
        01001001.,
        01001001.,
        01001001.,
        00110110.,
        00000000.,
        00000000.);
    GLYPH(088.5) 
        01000100.,
        01000100.,
        01000100.,
        00111000.,
        00101000.,
        01000100.,
        01000100.,
        00000000.,
        00000000.);
    GLYPH(089.5) 
        01000100.,
        01000100.,
        01000100.,
        00111000.,
        00010000.,
        00010000.,
        00010000.,
        00000000.,
        00000000.);
    GLYPH(090.5) 
        01111110.,
        00000010.,
        00000100.,
        00010000.,
        00110000.,
        01000000.,
        01111110.,
        00000000.,
        00000000.);
	
    return mat3(0);
/*
    // can likely get at least 14 digits accuracy out dec float.
    GLYPH(256.) 
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.); 
*/
}

// Function 203
vec3 getBallTexture(vec2 uv, vec3 color, int num) {
    vec3 white = vec3(1.0, 1.0, 0.8);
    if(num == 0) {
        return white;
    } else {
        bool solid = (num < 9);
        float edgeBlend = 0.01;
        vec2 dirToCenter = vec2(0.5, 0.5) - vec2(uv.x + (0.5-uv.x)*0.5, uv.y);
        float d = sqrt(dot(dirToCenter, dirToCenter));
        float r = 0.07;
        vec3 res;

        float dirt = texture(iChannel2,uv*1.5).x;

        vec3 non_solid;
        float dd = abs(uv.y - 0.5) * 2.0f;
        if(dd < 0.18 - edgeBlend) {
            non_solid = color;
        } else if(dd > 0.18 + edgeBlend) {
            non_solid = white;
        } else {
            float blende = ((dd - (0.18 - edgeBlend))/(2.0*edgeBlend))*dirt;
            non_solid = mix(color, white, blende);
        }
        vec3 outside_circle = mix(non_solid, color, solid? 1.0 : 0.0);
        vec3 inside_circle = vec3(0.);
        vec2 scale = vec2(5.0, 8.0);
        if(num > 9) {
            vec4 numc1 = char(scale*(uv - 0.5) + vec2(0.3, 0.5), 48.0 + float(num-10));
            vec4 numc2 = char(scale*(uv - 0.5) + vec2(0.7, 0.5), 48.0 + float(num/10));
            numc1.xyz = vec3(1.0) - numc1.xxx*2.0;
            numc2.xyz = vec3(1.0) - numc2.xxx*2.0;
            inside_circle = mix(white, numc1.xyz, numc1.w)*mix(white, numc2.xyz, numc2.w);
        } else {
            vec4 numc = char(scale*(uv - 0.5) + vec2(0.5), 48.0 + float(num));
            numc.xyz = vec3(1.0) - numc.xxx*2.0;
            inside_circle = mix(white, numc.xyz, numc.w);
        }

        bool on_69_mark = (d > 0.047) && (d < 0.057) && (uv.y < 0.5) && (abs(uv.x - 0.5) < 0.03);
        inside_circle *= vec3((num == 6 || num == 9)? (on_69_mark? dirt*dirt : 1.0) : 1.0);
        res = mix(outside_circle, inside_circle, (d < r - edgeBlend)?1.0 : 0.0);

        float blendc = ((d - (r - edgeBlend))/(2.0*edgeBlend))*dirt;
        vec3 on_the_circle = mix(white, color, blendc);
        res = mix(res, on_the_circle, (abs(d - r) < edgeBlend)?1.0 : 0.0);
        return res;
    }
}

// Function 204
float textDemo(in vec2 p){float c=0.;
 SetTextPosition(p,1.,12.);
 c+=drawInt(p,123, 8);   
 c+=drawInt(p,-1234567890);// right now !!!
 _ c+=drawInt(p,0);                
 _ c+=drawInt(p,-1);                
 _ c+=drawFloat(p,-123.456);     // right now !!!
 SetTextPosition(p,1.,13.);
 _ c+=drawInt(p,-123, 8);   
 _ c+=drawInt(p,1234567890,11);
 _ c+=drawFloat(p,0.0,0,0);
 _ c+=drawFloat(p,1.0,0,0);
 _ c+=drawFloat(p,654.321);      // nearly right
 _ c+=drawFloat(p,999.9, 1);
 _ c+=drawFloat(p,pow(10., 3.),1);   
 _ c+=drawFloat(p,pow(10., 6.),1);   
 SetTextPosition(p,1.,14.);c+=drawFloat(p,showSmall,60);
 SetTextPosition(p,1.,15.);c+=drawFloat(p,show60,60);
 SetTextPosition(p,1.,16.);c+=drawFloat(p,show4+sign(show4)*.5*pow(10.,-4.),4);
 return c;}

// Function 205
vec4 fireTexture(in vec2 uv, in float t, in float z) {
	uv*= curvature*SIZE;
	uv.y -= 1.5; //-uv.y;
    
	float gam = mix(GAMMA*2., 0., .5+.5*z);
    float density = ray_density;
	float 	r = sqrt(dot(uv,uv)), // DISTANCE FROM CENTER, A.K.A CIRCLE
	 		x = dot(normalize(uv), vec2(.35,0.))+t,
			y = dot(normalize(uv), vec2(.0,.65))+t;
 
    float val = fbm(vec2(r + y *density, x + density)); // GENERATES THE FLARING
	val = smoothstep(gam*.02-.1,ray_brightness+(gam*0.02-.1)+.001, val);
	val *= 15.; // sqrt(val); // WE DON'T REALLY NEED SQRT HERE, CHANGE TO 15. * val FOR PERFORMANCE
	
	vec3 col = val / vec3(RED,GREEN,BLUE);
	col = 1.-col; // WE DO NOT NEED TO CLAMP THIS LIKE THE NIMITZ SHADER DOES!
    float rad=30. * textureLod(iChannel1, uv, 0.).x; // MODIFY THIS TO CHANGE THE RADIUS OF THE SUNS CENTER
	col = mix(col,vec3(1.), rad - 266.667 * r); // REMOVE THIS TO SEE THE FLARING
    uv.y=uv.y+1.2;
    r = length(uv);
	col = col * (1. - clamp(.3*abs(r*r),0.,1.));
    col = clamp(col,0.,100.);
	return vec4(col, smoothstep(length(col),.0,.2));
}

// Function 206
mat3 glyph_8_9(float g) {
    GLYPH(032.5) 0); 
    GLYPH(042.5) 
        00001000.,
        00101010.,
        00011100.,
        00111110.,
        00011100.,
        00101010.,
        00001000.,
        00000000.,
        00000000.);
    GLYPH(046.5) 
        00000000.,
        00000000.,
        00000000.,
        00000000.,
        00000000.,
        00011000.,
        00011000.,
        00000000.,
        00000000.);
    GLYPH(049.5) 
        00011000.,
        00001000.,
        00001000.,
        00001000.,
        00001000.,
        00001000.,
        00011100.,
        00000000.,
        00000000.);
    GLYPH(050.5) 
        00111100.,
        01000010.,
        00000010.,
        00001100.,
        00110000.,
        01000000.,
        01111110.,
        00000000.,
        00000000.);
    GLYPH(051.5) 
        00111100.,
        01000010.,
        00000010.,
        00011100.,
        00000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(052.5) 
        01000100.,
        01000100.,
        01000100.,
        00111110.,
        00000100.,
        00000100.,
        00000100.,
        00000000.,
        00000000.);
    GLYPH(053.5) 
        01111110.,
        01000000.,
        01111000.,
        00000100.,
        00000010.,
        01000100.,
        00111000.,
        00000000.,
        00000000.);
    GLYPH(054.5) 
        00111100.,
        01000010.,
        01000000.,
        01011100.,
        01100010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(055.5) 
        00111110.,
        01000010.,
        00000010.,
        00000100.,
        00000100.,
        00001000.,
        00001000.,
        00000000.,
        00000000.);
    GLYPH(056.5) 
        00111100.,
        01000010.,
        01000010.,
        00111100.,
        01000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(057.5) 
        00111100.,
        01000010.,
        01000010.,
        00111110.,
        00000010.,
        00000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(058.5) 
        00111100.,
        00100100.,
        01001010.,
        01010010.,
        01010010.,
        00100100.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(065.5) 
        00011000.,
        00100100.,
        01000010.,
        01111110.,
        01000010.,
        01000010.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(066.5) 
        11111000.,
        01000100.,
        01000100.,
        01111000.,
        01000100.,
        01000100.,
        11111000.,
        00000000.,
        00000000.);
    GLYPH(067.5) 
        00011100.,
        00100010.,
        01000000.,
        01000000.,
        01000000.,
        00100010.,
        00011100.,
        00000000.,
        00000000.);
    GLYPH(068.5) 
        11111000.,
        01000100.,
        01000010.,
        01000010.,
        01000010.,
        01000100.,
        11111000.,
        00000000.,
        00000000.);
    GLYPH(069.5) 
        01111110.,
        01000000.,
        01000000.,
        01111000.,
        01000000.,
        01000000.,
        01111110.,
        00000000.,
        00000000.);
    GLYPH(070.5) 
        01111110.,
        01000000.,
        01000000.,
        01111000.,
        01000000.,
        01000000.,
        01000000.,
        00000000.,
        00000000.);
    GLYPH(071.5) 
        00111100.,
        01000010.,
        01000000.,
        01001110.,
        01000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(072.5) 
        01000010.,
        01000010.,
        01000010.,
        01111110.,
        01000010.,
        01000010.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(073.5) 
        00111000.,
        00010000.,
        00010000.,
        00010000.,
        00010000.,
        00010000.,
        00111000.,
        00000000.,
        00000000.);
    GLYPH(074.5) 
        01111110.,
        00000100.,
        00000100.,
        00000100.,
        00000100.,
        01000100.,
        00111000.,
        00000000.,
        00000000.);
    GLYPH(075.5) 
        01000100.,
        01001000.,
        01010000.,
        01110000.,
        01001000.,
        01000100.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(076.5) 
        00100000.,
        00100000.,
        00100000.,
        00100000.,
        00100000.,
        00100000.,
        00111111.,
        00000000.,
        00000000.);
    GLYPH(077.5) 
        01000010.,
        01100110.,
        01011010.,
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(078.5) 
        01100010.,
        01100010.,
        01010010.,
        01001010.,
        01000110.,
        01000010.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(079.5) 
        00111100.,
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(080.5) 
        11111100.,
        01000010.,
        01000010.,
        01111100.,
        01000000.,
        01000000.,
        01000000.,
        00000000.,
        00000000.);
    GLYPH(081.5) 
        00111100.,
        01000010.,
        01000010.,
        01000010.,
        01001010.,
        01000110.,
        00111101.,
        00000000.,
        00000000.);
    GLYPH(082.5) 
        01111100.,
        01000010.,
        01000010.,
        01111100.,
        01001000.,
        01000100.,
        01000010.,
        00000000.,
        00000000.);
    GLYPH(083.5) 
        00111100.,
        01000010.,
        01000000.,
        00111100.,
        00000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(084.5) 
        01111100.,
        00010000.,
        00010000.,
        00010000.,
        00010000.,
        00010000.,
        00010000.,
        00000000.,
        00000000.);
    GLYPH(085.5) 
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        00111100.,
        00000000.,
        00000000.);
    GLYPH(086.5) 
        01000010.,
        01000010.,
        01000010.,
        01000010.,
        00100100.,
        00100100.,
        00011000.,
        00000000.,
        00000000.);
    GLYPH(087.5) 
        01000001.,
        01000001.,
        01000001.,
        01001001.,
        01001001.,
        01001001.,
        00110110.,
        00000000.,
        00000000.);
    GLYPH(088.5) 
        01000100.,
        01000100.,
        01000100.,
        00111000.,
        00101000.,
        01000100.,
        01000100.,
        00000000.,
        00000000.);
    GLYPH(089.5) 
        01000100.,
        01000100.,
        01000100.,
        00111000.,
        00010000.,
        00010000.,
        00010000.,
        00000000.,
        00000000.);
    GLYPH(090.5) 
        01111110.,
        00000010.,
        00000100.,
        00010000.,
        00110000.,
        01000000.,
        01111110.,
        00000000.,
        00000000.);
	
    return mat3(0);
/*
    // can likely get at least 14 digits accuracy out dec float.
    GLYPH(256.) 
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.,
        00000000000000.); 
*/
}

// Function 207
vec3 GetTexture( uint iTextureIndex, vec3 vRayOrigin, vec3 vHitOffset, vec3 vNormal )
{
    vec3 vHitPos = vRayOrigin + vHitOffset;
    
	vec3 vTexCol = vec3(1.0);// * fTextureIndex;
    
    vec2 vProjectedCoord = GetProjectedCoord( vHitPos, vNormal );    
    
	vec2 vTexCoord = vProjectedCoord * 0.01;
    
    //return vec3(fract(vTexCoord.xy), 0.0);
	
	if ( iTextureIndex == TEX_LAVA1 || iTextureIndex == TEX_WATER1 || iTextureIndex == TEX_WATER2 || iTextureIndex == TEX_TELEPORT )
	{
		vec2 vOrig = floor( vTexCoord * 64.0 ) / 64.0;
		vTexCoord.x += sin(vOrig.y * 5.0 + iTime * 1.0) * 0.1;
		vTexCoord.y += sin(vOrig.x * 5.0 + iTime * 1.0) * 0.1;
	}
    
    if ( iTextureIndex == TEX_SKY1 )
    {
        vTexCoord.xy = vHitOffset.xy / vHitOffset.z;
        vTexCoord.y += iTime * 0.1;
    }
    
    vTexCol = SampleTexture( iTextureIndex, vTexCoord * 64.0 );

    if ( iTextureIndex == TEX_SKY1 )
    {
        vTexCoord *= 2.0;
        vTexCoord.y += iTime * 0.1;
	    vec3 sample2 = SampleTexture( TEX_SKY1, vTexCoord * 64.0 );
        
        vTexCol = mix( vTexCol, sample2, step( vTexCol.x, 0.0 ) );
    }
    

    return vTexCol;
}

// Function 208
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

// Function 209
vec3 hexTexture(vec2 uv, vec3 color, vec3 tex, vec2 firstXY)
{
	vec2 u = 6.*uv;;
    vec2 s = vec2(1.,1.732);
    vec2 a = mod(u,s)*2.-s;
    vec2 idA = floor(u/s);
    vec2 b = mod(u+s*.5,s)*2.-s;
    vec2 idB = floor((u+s*.5)/s);
    
    float la = length(a);
    float lb = length(b);
    
    u = la < lb ? a : b;
    vec2 idSeed = la < lb ? idA : idB*1000.;
    float id = rnd(idSeed+firstXY.x*firstXY.y/800.);
    vec2 st = abs(u);
    float q = max(st.x, dot(st,normalize(s)));
    float radius = pow(id*0.2,4.);
    float f = smoothstep(radius + 0., radius + 0.05, 1.0-q);
    //+firstXY.y/16.
    vec3 col = mix(tex,color-fract(8.*id)*0.5,f*step(0.2,id));
    return col;
}

// Function 210
vec3 texToSpace(vec2 coord, int c, int id, vec2 size) {
    vec2 sub = texSubdivisions;
    vec2 subSize = floor(size / sub);
    vec2 subCoord = floor(coord / subSize);
    float z = 0.;
    z += float(id) * 4. * sub.y * sub.x; // face offset
    z += float(c) * sub.y * sub.x; // channel offset
    z += subCoord.y * sub.x; // y offset
    z += subCoord.x; // x offset
    float zRange = sub.x * sub.y * 4. * 6. - 1.;
    z /= zRange;
    vec2 subUv = mod(coord / subSize, 1.);
    vec3 p = vec3(subUv, z);
    p = p * 2. - 1.; // range -1:1
    return p;
}

// Function 211
vec4 texture3DLinear(sampler2D tex, vec3 uvw, vec3 vres)
{
    vec3 blend = fract(uvw*vres);
    vec4 off = vec4(1.0/vres, 0.0);
    
    //2x2x2 sample blending
    vec4 b000 = texture3D(tex, uvw + off.www, vres);
    vec4 b100 = texture3D(tex, uvw + off.xww, vres);
    
    vec4 b010 = texture3D(tex, uvw + off.wyw, vres);
    vec4 b110 = texture3D(tex, uvw + off.xyw, vres);
    
    vec4 b001 = texture3D(tex, uvw + off.wwz, vres);
    vec4 b101 = texture3D(tex, uvw + off.xwz, vres);
    
    vec4 b011 = texture3D(tex, uvw + off.wyz, vres);
    vec4 b111 = texture3D(tex, uvw + off.xyz, vres);
    
    return mix(mix(mix(b000,b100,blend.x), mix(b010,b110,blend.x), blend.y), 
               mix(mix(b001,b101,blend.x), mix(b011,b111,blend.x), blend.y),
               blend.z);
}

// Function 212
vec3 Text(vec2 uv, vec3 col, float dist) 
{
    uv = uv * 42.0 - vec2(-3.5, -2.1);

    float text = FEMSCX(uv - vec2(-15.5, 0.0));
    float metA = meterA(uv - vec2(0.0, 0.6), dist);
    float metB = meterB(uv - vec2(3.5, 1.2), dist);

    // text coloring
    col = mix(blue, col, text);
    col = mix(red, col, metA);
    col = mix(red, col, metB);

    return col;
}

// Function 213
float textColor(vec3 bgColor) {
  float r = bgColor.r * 255.0,
        g = bgColor.g * 255.0,
        b = bgColor.b * 255.0;
  float yiq = (r * 299.0 + g * 587.0 + b * 114.0) / 1000.0;
  return (yiq >= 128.0) ? 0.0 : 1.0;
}

// Function 214
vec4 fetchTexture (vec2 uvCoord, vec2 textureRes)
{
    vec2 fetchCoord = uvCoord * textureRes;
    vec2 fetchFract = fract (fetchCoord);
    vec4 fetch00 = texelFetch (iChannel0, ivec2 (fetchCoord.xy), 0);
    vec4 fetch10 = texelFetch (iChannel0, ivec2 (fetchCoord.xy) + ivec2 (1, 0), 0);
    vec4 fetch01 = texelFetch (iChannel0, ivec2 (fetchCoord.xy) + ivec2 (0, 1), 0);
    vec4 fetch11 = texelFetch (iChannel0, ivec2 (fetchCoord.xy) + ivec2 (1, 1), 0);
    
    return mix (mix (fetch00, fetch10, fetchFract.x), mix (fetch01, fetch11, fetchFract.x), fetchFract.y);
}

// Function 215
vec3 bigtext( vec2 p, float sp, float screenNum, float pixwid ) 
{
    // Choose large label AAA characters - these are drawn at fixed locations
    float c[3];
    vec3 lab = label(mod((sp+screenNum)*11.,10.));
    c[0]=lab.x;//ran(sp,241.,10.,14.);
    c[1]=lab.y;//ran(sp,137.,0.,14.);
    c[2]=lab.z;//ran(sp,113.,13.,14.);

    // Choose small label characters AAA: NN-AA
    float s[9];
    s[0] = ran(sp,277.,313.,14.); 
    s[1] = ran(sp,173.,311.,14.);
    s[2] = ran(sp,113.,433.,14.); 
    s[3] = 24.;// Colon
    s[4] = ran(sp,157.,421.,10.)+14.;
    s[5] = ran(sp,119.,133.,10.)+14.;
    s[6] = 25.;// Dash
    s[7] = ran(sp,139.,313.,14.);
    s[8] = ran(sp,119.,137.,14.);
    
    // Main label location
    float sc1 = 1.8;
    vec2 ma;
    // Small label location
    float sc2 = 7.0;
    float sw = .15;
    float ss = .2/sc2;
    float m = 1.0;//ch((p+sa)*sc2,s1);
    // Find small label offsets 
    float w[9];for (int i=0;i<9;i++) w[i] = chW(s[i])*sw;
    float x[9];x[0]=0.;for (int i=1;i<9;i++) { x[i] = x[i-1]+w[i-1]+w[i]+ss; if(i==4)x[i]+=.05; } 
    
    // Check character
    float aatext = pixwid;
    vec2 b = vec2(0.);
    float d = 1.0;
    float cix = -1.;
    float sc = 1.;
    if (p.x>-.98 && p.x<.98) {
        if ((p.y<-0.06)&&(p.y>-.35)) {
            // Main label - fixed laout
            ma = vec2(-.05,.2);
            int ix = int((p.x+ma.x+1.25)/.5)-1;
            sc = sc1;
            ma.x = mod((p.x+ma.x+1.25),.5)-.25;
            if (ix<=2 && ix>=0) cix = c[ix];
        } else if ((p.y>-0.06)&&(p.y<0.1)) {
            // Small label - proportional layout
            aatext = pixwid*5.;
            ma = vec2(.7,-.06);
            int ix=-1;
            float mx;
            for (int i=9;i>=0;i--) { if ((p.x+ma.x)<(x[i]+w[i])) { ix=i; mx = ma.x-x[i]; } }
            sc = sc2;
            ma.x = p.x+mx;
            if (ix<=8 && ix>=0) cix = s[ix];
        } else if ((p.y>.91 && p.y<1.)) {
            // Screen number
            aatext = pixwid*6.;
            ma = vec2(.87,-.97);
            int ix = int((p.x+ma.x+1.25)/.11)-11;
            sc = 10.;
            ma.x = mod((p.x+ma.x+1.25),.11)-.05;
            if (ix==0) cix=2.;
            if (ix==1) cix=14.+screenNum+1.;
        }
    }
    
    // Draw the character if we are in one
    if (cix>-1.) d = min(d,ch(vec2(ma.x,p.y+ma.y)*sc,cix));
    return vec3(d,aatext,0.);
}

// Function 216
vec3 glyphs(vec2 p,float glyph){
 p*=.08;//optional, making glyphs as large as possible
 p.x*=1.2;
 float x=16.5,y=9.9;//offsets
 SetTextPosition(p,x,y);vec3 c;
 S(65+int(glyph*addresses));
 //S(65+int((a.x+a.y)*addresses));
 //_B//alternatively only draw "B" or an integer
 //c+=drawInt(1,0);//this function may add a leading "+" sign
 return c;}

// Function 217
vec4 glyph3d(vec2 fragCoord, vec2 localCoord, float size, int char) {
    vec4 c = vec4(0.0);
    vec4 glyph3 = glyph(fragCoord,localCoord,vec2(-size,size),char);
    vec3 gLight = vec3(1.0,1.0,0.6);
    vec3 gNormal = normalize(normalize(gLight)-glyph3.xyz);
    float gShadow = dot(normalize(gLight),gNormal);
    vec4 gDiffuse = vec4(vec3(0.76,0.5,0.9)*gShadow,glyph3.a);
    blend(c, gDiffuse);
    return c;
}

// Function 218
vec4 textureBox( in sampler2D tex, in vec3 pos, in vec3 nor )
{
    vec4 cx = texture( tex, pos.yz );
    vec4 cy = texture( tex, pos.xz );
    vec4 cz = texture( tex, pos.xy );
    vec3 m = nor*nor;
    return (cx*m.x + cy*m.y + cz*m.z)/(m.x+m.y+m.z);
}

// Function 219
float get_texture_index(in sampler2D s)
{
    return texelFetch(s, CTRL_TEXTURE, 0).w;
}

// Function 220
vec2 	UIStyle_FontPadding() 			{ return vec2(6.0, 2.0); }

// Function 221
vec3 carTexture(TraceResult tr, vec3 normal) {
    vec3 col = vec3(0.);
    vec3 dir = tr.dist.cars.q1;

    if (normal.y > 0. && dir.x == 0.) {
        if (dir.y > 0.) {
            col.r += 1.;
        } else {
            col.rgb += 1.;
        }
    } else
    if (normal.y < 0. && dir.x == 0.) {
        if (dir.y < 0.) {
            col.r += 1.;
        } else {
            col.rgb += 1.;
        }
    } else

    if (normal.x > 0. && dir.y == 0.) {
        if (dir.x > 0.) {
            col.r += 1.;
        } else {
            col.rgb += 1.;
        }
    } else
    if (normal.x < 0. && dir.y == 0.) {
        if (dir.x < 0.) {
            col.r += 1.;
        } else {
            col.rgb += 1.;
        }
    } else {
        col = COLOR_CAR_ROOF;
    }

    return col;
}

// Function 222
vec3 _texture(vec3 p) {
    p.y += .25;
    if (prevBolt(p) <= nextBolt(p)) {
        tPrevBolt(p);
    } else {
        tNextBolt(p);
    }
    p *= 1.;
    
    // yay, looks like we can sneakily get away with a single sample here
    return pow(texture(iChannel0, p.xy * 2. + p.zz).rgb, vec3(2.2));
    return vec3(1);
}

// Function 223
vec4 font2(int c) {
  vec4 v = vec4(0);
  v=mix(v, vec4(0x3c66, 0x6e6e, 0x6062, 0x3c00), step(-0.500, float(c)));
  v=mix(v, vec4(0x183c, 0x667e, 0x6666, 0x6600), step(0.500, float(c)));
  v=mix(v, vec4(0x7c66, 0x667c, 0x6666, 0x7c00), step(1.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x6060, 0x6066, 0x3c00), step(2.500, float(c)));
  v=mix(v, vec4(0x786c, 0x6666, 0x666c, 0x7800), step(3.500, float(c)));
  v=mix(v, vec4(0x7e60, 0x6078, 0x6060, 0x7e00), step(4.500, float(c)));
  v=mix(v, vec4(0x7e60, 0x6078, 0x6060, 0x6000), step(5.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x606e, 0x6666, 0x3c00), step(6.500, float(c)));
  v=mix(v, vec4(0x6666, 0x667e, 0x6666, 0x6600), step(7.500, float(c)));
  v=mix(v, vec4(0x3c18, 0x1818, 0x1818, 0x3c00), step(8.500, float(c)));
  v=mix(v, vec4(0x1e0c, 0xc0c, 0xc6c, 0x3800), step(9.500, float(c)));
  v=mix(v, vec4(0x666c, 0x7870, 0x786c, 0x6600), step(10.500, float(c)));
  v=mix(v, vec4(0x6060, 0x6060, 0x6060, 0x7e00), step(11.500, float(c)));
  v=mix(v, vec4(0x6377, 0x7f6b, 0x6363, 0x6300), step(12.500, float(c)));
  v=mix(v, vec4(0x6676, 0x7e7e, 0x6e66, 0x6600), step(13.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x6666, 0x6666, 0x3c00), step(14.500, float(c)));
  v=mix(v, vec4(0x7c66, 0x667c, 0x6060, 0x6000), step(15.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x6666, 0x663c, 0xe00), step(16.500, float(c)));
  v=mix(v, vec4(0x7c66, 0x667c, 0x786c, 0x6600), step(17.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x603c, 0x666, 0x3c00), step(18.500, float(c)));
  v=mix(v, vec4(0x7e18, 0x1818, 0x1818, 0x1800), step(19.500, float(c)));
  v=mix(v, vec4(0x6666, 0x6666, 0x6666, 0x3c00), step(20.500, float(c)));
  v=mix(v, vec4(0x6666, 0x6666, 0x663c, 0x1800), step(21.500, float(c)));
  v=mix(v, vec4(0x6363, 0x636b, 0x7f77, 0x6300), step(22.500, float(c)));
  v=mix(v, vec4(0x6666, 0x3c18, 0x3c66, 0x6600), step(23.500, float(c)));
  v=mix(v, vec4(0x6666, 0x663c, 0x1818, 0x1800), step(24.500, float(c)));
  v=mix(v, vec4(0x7e06, 0xc18, 0x3060, 0x7e00), step(25.500, float(c)));
  v=mix(v, vec4(0x3c30, 0x3030, 0x3030, 0x3c00), step(26.500, float(c)));
  v=mix(v, vec4(0xc12, 0x307c, 0x3062, 0xfc00), step(27.500, float(c)));
  v=mix(v, vec4(0x3c0c, 0xc0c, 0xc0c, 0x3c00), step(28.500, float(c)));
  v=mix(v, vec4(0x18, 0x3c7e, 0x1818, 0x1818), step(29.500, float(c)));
  v=mix(v, vec4(0x10, 0x307f, 0x7f30, 0x1000), step(30.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0x0, 0x0), step(31.500, float(c)));
  v=mix(v, vec4(0x1818, 0x1818, 0x0, 0x1800), step(32.500, float(c)));
  v=mix(v, vec4(0x6666, 0x6600, 0x0, 0x0), step(33.500, float(c)));
  v=mix(v, vec4(0x6666, 0xff66, 0xff66, 0x6600), step(34.500, float(c)));
  v=mix(v, vec4(0x183e, 0x603c, 0x67c, 0x1800), step(35.500, float(c)));
  v=mix(v, vec4(0x6266, 0xc18, 0x3066, 0x4600), step(36.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x3c38, 0x6766, 0x3f00), step(37.500, float(c)));
  v=mix(v, vec4(0x60c, 0x1800, 0x0, 0x0), step(38.500, float(c)));
  v=mix(v, vec4(0xc18, 0x3030, 0x3018, 0xc00), step(39.500, float(c)));
  v=mix(v, vec4(0x3018, 0xc0c, 0xc18, 0x3000), step(40.500, float(c)));
  v=mix(v, vec4(0x66, 0x3cff, 0x3c66, 0x0), step(41.500, float(c)));
  v=mix(v, vec4(0x18, 0x187e, 0x1818, 0x0), step(42.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0x18, 0x1830), step(43.500, float(c)));
  v=mix(v, vec4(0x0, 0x7e, 0x0, 0x0), step(44.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0x18, 0x1800), step(45.500, float(c)));
  v=mix(v, vec4(0x3, 0x60c, 0x1830, 0x6000), step(46.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x6e76, 0x6666, 0x3c00), step(47.500, float(c)));
  v=mix(v, vec4(0x1818, 0x3818, 0x1818, 0x7e00), step(48.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x60c, 0x3060, 0x7e00), step(49.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x61c, 0x666, 0x3c00), step(50.500, float(c)));
  v=mix(v, vec4(0x60e, 0x1e66, 0x7f06, 0x600), step(51.500, float(c)));
  v=mix(v, vec4(0x7e60, 0x7c06, 0x666, 0x3c00), step(52.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x607c, 0x6666, 0x3c00), step(53.500, float(c)));
  v=mix(v, vec4(0x7e66, 0xc18, 0x1818, 0x1800), step(54.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x663c, 0x6666, 0x3c00), step(55.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x663e, 0x666, 0x3c00), step(56.500, float(c)));
  v=mix(v, vec4(0x0, 0x1800, 0x18, 0x0), step(57.500, float(c)));
  v=mix(v, vec4(0x0, 0x1800, 0x18, 0x1830), step(58.500, float(c)));
  v=mix(v, vec4(0xe18, 0x3060, 0x3018, 0xe00), step(59.500, float(c)));
  v=mix(v, vec4(0x0, 0x7e00, 0x7e00, 0x0), step(60.500, float(c)));
  v=mix(v, vec4(0x7018, 0xc06, 0xc18, 0x7000), step(61.500, float(c)));
  v=mix(v, vec4(0x3c66, 0x60c, 0x1800, 0x1800), step(62.500, float(c)));
  v=mix(v, vec4(0x0, 0xff, 0xff00, 0x0), step(63.500, float(c)));
  v=mix(v, vec4(0x81c, 0x3e7f, 0x7f1c, 0x3e00), step(64.500, float(c)));
  v=mix(v, vec4(0x1818, 0x1818, 0x1818, 0x1818), step(65.500, float(c)));
  v=mix(v, vec4(0x0, 0xff, 0xff00, 0x0), step(66.500, float(c)));
  v=mix(v, vec4(0x0, 0xffff, 0x0, 0x0), step(67.500, float(c)));
  v=mix(v, vec4(0xff, 0xff00, 0x0, 0x0), step(68.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0xffff, 0x0), step(69.500, float(c)));
  v=mix(v, vec4(0x3030, 0x3030, 0x3030, 0x3030), step(70.500, float(c)));
  v=mix(v, vec4(0xc0c, 0xc0c, 0xc0c, 0xc0c), step(71.500, float(c)));
  v=mix(v, vec4(0x0, 0xe0, 0xf038, 0x1818), step(72.500, float(c)));
  v=mix(v, vec4(0x1818, 0x1c0f, 0x700, 0x0), step(73.500, float(c)));
  v=mix(v, vec4(0x1818, 0x38f0, 0xe000, 0x0), step(74.500, float(c)));
  v=mix(v, vec4(0xc0c0, 0xc0c0, 0xc0c0, 0xffff), step(75.500, float(c)));
  v=mix(v, vec4(0xc0e0, 0x7038, 0x1c0e, 0x703), step(76.500, float(c)));
  v=mix(v, vec4(0x307, 0xe1c, 0x3870, 0xe0c0), step(77.500, float(c)));
  v=mix(v, vec4(0xffff, 0xc0c0, 0xc0c0, 0xc0c0), step(78.500, float(c)));
  v=mix(v, vec4(0xffff, 0x303, 0x303, 0x303), step(79.500, float(c)));
  v=mix(v, vec4(0x3c, 0x7e7e, 0x7e7e, 0x3c00), step(80.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0xff, 0xff00), step(81.500, float(c)));
  v=mix(v, vec4(0x367f, 0x7f7f, 0x3e1c, 0x800), step(82.500, float(c)));
  v=mix(v, vec4(0x6060, 0x6060, 0x6060, 0x6060), step(83.500, float(c)));
  v=mix(v, vec4(0x0, 0x7, 0xf1c, 0x1818), step(84.500, float(c)));
  v=mix(v, vec4(0xc3e7, 0x7e3c, 0x3c7e, 0xe7c3), step(85.500, float(c)));
  v=mix(v, vec4(0x3c, 0x7e66, 0x667e, 0x3c00), step(86.500, float(c)));
  v=mix(v, vec4(0x1818, 0x6666, 0x1818, 0x3c00), step(87.500, float(c)));
  v=mix(v, vec4(0x606, 0x606, 0x606, 0x606), step(88.500, float(c)));
  v=mix(v, vec4(0x81c, 0x3e7f, 0x3e1c, 0x800), step(89.500, float(c)));
  v=mix(v, vec4(0x1818, 0x18ff, 0xff18, 0x1818), step(90.500, float(c)));
  v=mix(v, vec4(0xc0c0, 0x3030, 0xc0c0, 0x3030), step(91.500, float(c)));
  v=mix(v, vec4(0x1818, 0x1818, 0x1818, 0x1818), step(92.500, float(c)));
  v=mix(v, vec4(0x0, 0x33e, 0x7636, 0x3600), step(93.500, float(c)));
  v=mix(v, vec4(0xff7f, 0x3f1f, 0xf07, 0x301), step(94.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0x0, 0x0), step(95.500, float(c)));
  v=mix(v, vec4(0xf0f0, 0xf0f0, 0xf0f0, 0xf0f0), step(96.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0xffff, 0xffff), step(97.500, float(c)));
  v=mix(v, vec4(0xff00, 0x0, 0x0, 0x0), step(98.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0x0, 0xff), step(99.500, float(c)));
  v=mix(v, vec4(0xc0c0, 0xc0c0, 0xc0c0, 0xc0c0), step(100.500, float(c)));
  v=mix(v, vec4(0xcccc, 0x3333, 0xcccc, 0x3333), step(101.500, float(c)));
  v=mix(v, vec4(0x303, 0x303, 0x303, 0x303), step(102.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0xcccc, 0x3333), step(103.500, float(c)));
  v=mix(v, vec4(0xfffe, 0xfcf8, 0xf0e0, 0xc080), step(104.500, float(c)));
  v=mix(v, vec4(0x303, 0x303, 0x303, 0x303), step(105.500, float(c)));
  v=mix(v, vec4(0x1818, 0x181f, 0x1f18, 0x1818), step(106.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0xf0f, 0xf0f), step(107.500, float(c)));
  v=mix(v, vec4(0x1818, 0x181f, 0x1f00, 0x0), step(108.500, float(c)));
  v=mix(v, vec4(0x0, 0xf8, 0xf818, 0x1818), step(109.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0x0, 0xffff), step(110.500, float(c)));
  v=mix(v, vec4(0x0, 0x1f, 0x1f18, 0x1818), step(111.500, float(c)));
  v=mix(v, vec4(0x1818, 0x18ff, 0xff00, 0x0), step(112.500, float(c)));
  v=mix(v, vec4(0x0, 0xff, 0xff18, 0x1818), step(113.500, float(c)));
  v=mix(v, vec4(0x1818, 0x18f8, 0xf818, 0x1818), step(114.500, float(c)));
  v=mix(v, vec4(0xc0c0, 0xc0c0, 0xc0c0, 0xc0c0), step(115.500, float(c)));
  v=mix(v, vec4(0xe0e0, 0xe0e0, 0xe0e0, 0xe0e0), step(116.500, float(c)));
  v=mix(v, vec4(0x707, 0x707, 0x707, 0x707), step(117.500, float(c)));
  v=mix(v, vec4(0xffff, 0x0, 0x0, 0x0), step(118.500, float(c)));
  v=mix(v, vec4(0xffff, 0xff00, 0x0, 0x0), step(119.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0xff, 0xffff), step(120.500, float(c)));
  v=mix(v, vec4(0x303, 0x303, 0x303, 0xffff), step(121.500, float(c)));
  v=mix(v, vec4(0x0, 0x0, 0xf0f0, 0xf0f0), step(122.500, float(c)));
  v=mix(v, vec4(0xf0f, 0xf0f, 0x0, 0x0), step(123.500, float(c)));
  v=mix(v, vec4(0x1818, 0x18f8, 0xf800, 0x0), step(124.500, float(c)));
  v=mix(v, vec4(0xf0f0, 0xf0f0, 0x0, 0x0), step(125.500, float(c)));
  v=mix(v, vec4(0xf0f0, 0xf0f0, 0xf0f, 0xf0f), step(126.500, float(c)));
  return v;
}

// Function 224
float textureAtlas(vec2 uv, int hitid)
{
    return alphatex(uv, ivec2(hitid, hitid >> 4) & 15);
    // TODO various symmetry modes to extend the available shapes
    // simple extrusions, lathes on various axes, vary orientation, etc.
}

// Function 225
void print_console_text(inout vec4 fragColor, vec2 fragCoord)
{
    float MARGIN = 12. * iResolution.x/800.;
    const vec4 COLORS[2] = vec4[2](vec4(vec3(.54), 1), vec4(.62, .30, .19, 1));
    const uint COLORED = (1u<<3) | (1u<<7);
    const int TYPING_LINE = 1;
    
    fragCoord.y -= iResolution.y * (1. - g_console.expanded);
    ivec2 uv = text_uv(fragCoord - MARGIN);
    bool typing = g_console.typing < 1.;
    int cut = int(mix(float(CONSOLE_TEXT.data[0]-1), 2., g_console.loaded));
    if (g_console.typing > 0.)
        --cut;
    
    int line = line_index(uv.y);
    if (uint(line) >= uint(CONSOLE_TEXT.data[0]-cut))
        return;
    line += cut;
    int start = CONSOLE_TEXT.data[1+line];
    int num_chars = CONSOLE_TEXT.data[2+line] - start;
    
    if (num_chars == 1)
    {
        const vec3 LINE_COLOR = vec3(.17, .13, .06);
        float LINE_END = min(iResolution.x - MARGIN*2., 300.);
        vec2 line = lit_line(vec2(uv.x, uv.y & 7) + .5, vec2(4. ,4.), vec2(LINE_END-4., 4.), 4.);
        line.x = mix(1. + .5 * line.x, 1., linear_step(-.5, -1.5, line.y));
        line.x *= 1. + -.25*random(vec2(uv));
		fragColor.rgb = mix(fragColor.rgb, LINE_COLOR * line.x, step(line.y, 0.));
        return;
    }
    
    int glyph = glyph_index(uv.x);
    if (line == TYPING_LINE)
    {
        float type_fraction = clamp(2. - abs(g_console.typing * 4. + -2.), 0., 1.);
        num_chars = clamp(int(float(num_chars-1)*type_fraction) + int(typing), 0, num_chars + int(typing));
    }
    if (uint(glyph) >= uint(num_chars))
        return;

    if (typing && line == TYPING_LINE && glyph == num_chars - 1)
    {
        glyph = fract(iTime*2.) < .5 ? _CARET_ : _SPACE_;
    }
    else
    {
        glyph += start;
        glyph = get_byte(glyph & 3, CONSOLE_TEXT.data[CONSOLE_TEXT.data[0] + 2 + (glyph>>2)]);
    }
    
    uint is_colored = line < 32 ? ((COLORED >> line) & 1u) : 0u;
    vec4 color = COLORS[is_colored];
    print_glyph(fragColor, uv, glyph, color);
}

// Function 226
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

// Function 227
vec4 sphereTexture(in sampler2D _tex, in vec2 _uv) {
  vec2 st = sphereCoords(_uv, 1.0);
  float aspect = iChannelResolution[0].y / iChannelResolution[0].x;
  st.x = fract(st.x * aspect + iTime * speedMoon);
  return textureLod(_tex, st, -16.0);
}

// Function 228
vec4 GetTextureSample(vec2 pos, float freq, float seed)
{
    vec3 hash = hash33(vec3(seed, 0.0, 0.0));
    float ang = hash.x * 2.0 * pi;
    mat2 rotation = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
    
    vec2 uv = rotation * pos * freq + hash.yz;
    return texture(iChannel0, uv);
}

// Function 229
float textLines(vec2 uvG)
{
    float wt = 5. * (iTime + 0.5*sin(iTime*1.4) + 0.2*sin(iTime*2.9)); // wobbly time
    vec2 uvGt = uvG + vec2(0., floor(wt));
    float ll = rand(vec2(uvGt.y, - 1.)) * ROWCOLS.x; // line length
    
    if (uvG.y > ROWCOLS.y - 2.){
        if (ceil(uvG.x) == floor(min(ll, fract(wt)*ROWCOLS.x)))
        	return 2.;
        if (ceil(uvG.x) > floor(min(ll, fract(wt)*ROWCOLS.x)))
        	return 0.;
    }
    if (uvGt.x > 5. && rand(uvGt) < .075)
        return 0.;
    if (max(5., uvGt.x) > ll)
        return 0.;
       
    return rand(uvGt)*15. + 2.;
}

// Function 230
void setup_text(){
    TEXT[0] = ivec4( 0x251C3E20, 0X24515156, 0X4B504902, 0X4854514F);
    TEXT[1] = ivec4( 0x022A4354, 0X4602264B, 0X554D1010, 0X10013556);
    TEXT[2] = ivec4( 0x4354564B, 0X5049022F, 0X350F2631, 0X35101010);
    TEXT[3] = ivec4( 0x012A2B2F, 0X272F024B, 0X55025647, 0X55564B50);
    TEXT[4] = ivec4( 0x4902475A, 0X56475046, 0X4746024F, 0X474F5154);
    TEXT[5] = ivec4( 0x5B101010, 0X02465150, 0X47100100, 0X24434602);
    TEXT[6] = ivec4( 0x45514F4F, 0X43504602, 0X51540248, 0X4B4E4702);
    TEXT[7] = ivec4( 0x50434F47, 0X01002B50, 0X58434E4B, 0X4602464B);
    TEXT[8] = ivec4( 0x54474556, 0X51545B01, 0X00234545, 0X47555502);
    TEXT[9] = ivec4( 0x4647504B, 0X47460100, 0X2E514346, 0X4B504910);
    TEXT[10] = ivec4( 0x10100100, 0X4A565652, 0X551C1111, 0X59595910);
    TEXT[11] = ivec4( 0x554A4346, 0X47545651, 0X5B104551, 0X4F11584B);
    TEXT[12] = ivec4( 0x47591159, 0X564E5B39, 0X4E010002, 0X38514E57);
    TEXT[13] = ivec4( 0x4F47024B, 0X50024654, 0X4B584702, 0X25024B55);
    TEXT[14] = ivec4( 0x022F350F, 0X263135EF, 0X01023851, 0X4E574F47);
    TEXT[15] = ivec4( 0x02354754, 0X4B434E30, 0X574F4447, 0X54024B55);
    TEXT[16] = ivec4( 0x02131415, 0X160F1718, 0X191A0102, 0X264B5447);
    TEXT[17] = ivec4( 0x45565154, 0X5B025148, 0X02251C3E, 0X0101382B);
    TEXT[18] = ivec4( 0x34373510, 0X25312F02, 0X02020202, 0X1412160E);
    TEXT[19] = ivec4( 0x15171402, 0X12190F12, 0X140F1412, 0X0213141C);
    TEXT[20] = ivec4( 0x12124301, 0X352A2326, 0X27343631, 0X3B102531);
    TEXT[21] = ivec4( 0x2F020202, 0X1B0E1913, 0X13021219, 0X0F12160F);
    TEXT[22] = ivec4( 0x14120213, 0X141C1212, 0X43012423, 0X2E2E1025);
    TEXT[23] = ivec4( 0x312F0202, 0X02020202, 0X0213130E, 0X19151402);
    TEXT[24] = ivec4( 0x12190F12, 0X160F1412, 0X0213141C, 0X12124301);
    TEXT[25] = ivec4( 0x292E352E, 0X1025312F, 0X02020202, 0X02020213);
    TEXT[26] = ivec4( 0x120E1312, 0X12021219, 0X0F12160F, 0X14120213);
    TEXT[27] = ivec4( 0x141C1212, 0X43010202, 0X02021602, 0X484B4E47);
    TEXT[28] = ivec4( 0x0A550B02, 0X1415170E, 0X1A1B1702, 0X445B5647);
    TEXT[29] = ivec4( 0x55010202, 0X02020202, 0X02020202, 0X02020202);
    TEXT[30] = ivec4( 0x02020202, 0X1202445B, 0X56475502, 0X48544747);
    TEXT[31] = ivec4( 0x01000000, 0X00000000, 0X00000000, 0X00000000);
}

// Function 231
float fontDist(vec2 tpos, float size, vec2 offset) {

    float scl = 0.63/size;
      
    vec2 uv = tpos*scl;
    vec2 font_uv = (uv+offset+0.5)*(1.0/16.0);
    
    float k = texture(iChannel1, font_uv, -100.0).w + 1e-6;
    
    vec2 box = abs(uv)-0.5;
        
    return max(k-127.0/255.0, max(box.x, box.y))/scl;
    
}

// Function 232
float text_c(vec2 U) {
    initMsg;
    U.x+=4.*(0.5-0.2812*(res.x/0.5));
    C(67);C(108);C(101);C(97);C(110);
    endMsg;
}

// Function 233
bool UIDrawContext_ScreenPosInCanvasRect( UIDrawContext drawContext, vec2 vScreenPos, Rect canvasRect )
{
	vec2 vCanvasPos = UIDrawContext_ScreenPosToCanvasPos( drawContext, vScreenPos );    
    return Inside( vCanvasPos, canvasRect );
}

// Function 234
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

// Function 235
vec3 mainScreenText(vec2 uv, float playerVsGpu)
{
    vec3 c = vec3(0);
    
    float col = 0.0;
    
    font_size = mix(5., 7., playerVsGpu);
    print_pos = vec2(-STRWIDTH(13.0)/2.0, -150./font_size -STRHEIGHT(1.0)/2.0);
    
    col = char(ch_P,uv);
    col += char(ch_L,uv);
    col += char(ch_A,uv);
    col += char(ch_Y,uv);
    col += char(ch_E,uv);
    col += char(ch_R,uv);
    
    col += char(ch_spc,uv);
    
    col += char(ch_v,uv);
    col += char(ch_s,uv);
    
    col += char(ch_spc,uv);
    
    col += char(ch_G,uv);
    col += char(ch_P,uv);
    col += char(ch_U,uv);
    c += mix(vec3(0.5), vec3(1.,.9,0.1), playerVsGpu)*col;
    
    font_size = mix(7., 5., playerVsGpu);
    print_pos = vec2(-STRWIDTH(16.0)/2.0, -250./font_size -STRHEIGHT(1.0)/2.0);
       
    col = char(ch_P,uv);
    col += char(ch_L,uv);
    col += char(ch_A,uv);
    col += char(ch_Y,uv);
    col += char(ch_E,uv);
    col += char(ch_R,uv);
    
    col += char(ch_spc,uv);
    
    col += char(ch_v,uv);
    col += char(ch_s,uv);
    
    col += char(ch_spc,uv);
    
    col += char(ch_P,uv);
    col += char(ch_L,uv);
    col += char(ch_A,uv);
    col += char(ch_Y,uv);
    col += char(ch_E,uv);
    col += char(ch_R,uv);
    c += mix(vec3(1.,.9,.1), vec3(0.5), playerVsGpu)*col;
    
    return c;
}

// Function 236
float checkersTexture( in vec2 p )
{
    vec2 q = floor(p);
    return mod( q.x+q.y, 2.0 );            // xor pattern
}

// Function 237
void glyph_2()
{
  MoveTo(1.8*x+y*0.2);
  LineTo(1.8*x);
  LineTo(0.2*x);    
  Bez3To(float2(0.9,0.625),x*1.8+y*1.1,x*1.8+y*1.5);
  Bez3To(x*1.8+y*2.2,x*0.2+y*2.2,x*0.2+y*1.5);
}

// Function 238
void glyph_8()
{
  MoveTo(x+y*1.1);
  Bez3To(y*1.1-x*0.2,-x*0.2-y*0.05,x-y*0.05);
  Bez3To(2.2*x-y*0.05,y*1.1+2.2*x,x+y*1.1);
  Bez3To(y*1.1,y*2.05,x+y*2.05);
  Bez3To(y*2.05+2.0*x,2.0*x+y*1.1,x+y*1.1);
}

// Function 239
vec3 SampleTexture( uint iTexture, const in vec2 _vUV )
{
    vec3 vCol0 = vec3(0.6);
    vec3 vCol1 = vec3(119.0, 79.0, 43.0) / 255.0;
    vec3 vCol2 = vec3(0);
    
    uint iOrigTexture = iTexture;
    
        
    vec3 col = vec3(1.0, 1.0, 1.0);
#ifdef DEBUG_IDENTIFY_UNDEFINED_TEXTURES    
    col = vec3(1.0, 0.0, 1.0);
#endif  
    
#ifdef DEBUG_IDENTIFY_TEXTURE
    if ( fTexture == DEBUG_IDENTIFY_TEXTURE && (fract(iTime) < 0.5))
    {
        return vec3(1.0, 0.0, 1.0);
    }
#endif     
    
    vec2 vUV = _vUV;
    
    vec2 vSize = vec2(64.0);
    float fPersistence = 0.8;
	float fNoise2Freq = 0.5;
    vec2 vRandomStreak = vec2(0.0);
    
    // Direct Substitutes
    if ( iTexture == TEX_FLOOR1_1 )
    {
        iTexture = TEX_FLAT14;
    }
    else 
    if ( iTexture == TEX_BROWN96 )
    {
        iTexture = TEX_BROWN1;
    }
    else
    if ( iTexture == TEX_COMPTALL)
    {
        iTexture = TEX_STARGR1;
    }
    else
    if ( iTexture == TEX_DOORSTOP )
    {
        iTexture == TEX_DOORTRAK;
    }
    else
    if ( iTexture == TEX_FLAT20 )
    {
        iTexture = TEX_DOORSTOP;
    }
	else
    if (iTexture == TEX_TEKWALL4
       || iTexture == TEX_STEP2)
    {
        // start area pillars
        iTexture = TEX_TEKWALL1;
    }
    else
    if (iTexture == TEX_STEP1)
    {
        // Comp room steps (lights added later)
        iTexture = TEX_STEP6;
    }
    else
    if ( iTexture == TEX_SLADWALL)
    {
        iTexture = TEX_BROWNGRN;
    }
    else
    if ( iTexture == TEX_EXITDOOR )
    {
        iTexture = TEX_DOOR3;
    }
    else
	if ( iTexture == TEX_FLAT23 )
    {
        iTexture = TEX_FLAT18;
    }         
        
        
    
    if ( iTexture == TEX_FLOOR4_8)
    {
        // start area
	    vCol0 = vec3(30.0, 30.0, 30.0);
        vCol1 = vec3(150.0, 150.0, 150.0);
    }
	else        
    if ( iTexture == TEX_FLOOR5_1)
    {
        // Corridor to outside
        iTexture = TEX_FLOOR4_8;
        vCol0 = vec3(51.0, 43.0, 19.0);
        vCol1 = vec3(150.0, 150.0, 150.0);
    }
	else        
    if ( iTexture == TEX_FLOOR5_2 )
    {
        // imp/nukage room
        iTexture = TEX_FLOOR4_8;
        vCol0 = vec3(51.0, 43.0, 19.0);
		vCol1 = vec3(79.0, 59, 35.0);
    }
    
    if ( iTexture == TEX_TLITE6_5)
    {
		vCol0 = vec3(.2);        
		vCol1 = vec3(1,0,0);
    }
    if ( iTexture == TEX_TLITE6_6 )
    {
        iTexture = TEX_TLITE6_5;
        vCol0 = vec3(.25, .2, .1);
        vCol1 = vec3(.8,.6,.4);
    }

    if ( iTexture == TEX_TLITE6_4 )
    {
        vCol0 = vec3(.25, .2, .1);
        vCol1 = vec3(.8,.6,.4);
    }
    else
    if ( iTexture == TEX_TLITE6_1 )
    {
        iTexture = TEX_TLITE6_4;
		vCol0 = vec3(.2);        
		vCol1 = vec3(1);
    }

    if ( iTexture == TEX_BIGDOOR2 )
    {
    	vCol0 = vec3(119) /255.;
    }
    else
    if ( iTexture == TEX_BIGDOOR4 )
    {
        iTexture = TEX_BIGDOOR2;
    	vCol0 = vec3(103,83,63) /255.;
    }
    
    if ( iTexture == TEX_FLOOR7_1 )
    {
	    vCol0 = vec3(51.0, 43.0, 19.0);
        vCol1 = vec3(79.0, 59, 35.0);
	}
    
    if ( iTexture == TEX_CEIL5_2 )
    {
        iTexture = TEX_FLOOR7_1;
	    vCol0 = vec3(51.0, 43.0, 19.0) * .75;
        vCol1 = vec3(79.0, 59, 35.0) * .75;
	}
    
    if ( iTexture == TEX_BROWN1 )
    {
        vCol0 = vec3(119.0, 95.0, 63.0);
        vCol1 = vec3(147.0, 123.0, 99.0);
        vCol2 = vec3(0.3, 0.2, 0.1);
    }
    else
    if ( iTexture == TEX_BROWNGRN )
    {
        iTexture = TEX_BROWN1;
        vCol1 = vec3(43,35,15);
        vCol0 = vec3(47.0, 47.0, 47.0);
        //vCol1 = vec3(147.0, 123.0, 99.0);
        //vCol1 = vec3(123.0, 127.0, 99.0);
        vCol2 = vec3(19,35,11) / 255.;
    }
    
    if ( iTexture == TEX_FLAT14 )
    {
        // Blue noise
 		vCol0 = vec3(0.0, 0.0, 35.0 / 255.0);
        vCol1 = vec3(0.0, 0.0, 200.0 / 255.0);
		fPersistence = 2.0;
    }
    else
    if ( iTexture == TEX_CEIL5_1 )
    {
        // Comp room side ceil
        iTexture = TEX_FLAT14;        
 		vCol0 = vec3(30.0 / 255.0);
        vCol1 = vec3(15.0 / 255.0);
		fPersistence = 2.0;
    }
    else
    if ( iTexture == TEX_FLAT18 )
    {
        // Comp room side floor
        iTexture = TEX_FLAT14;        
 		vCol0 = vec3(70.0 / 255.0);
        vCol1 = vec3(40.0 / 255.0);
		fPersistence = 2.0;
    }   
    else
    if ( iTexture == TEX_COMPSPAN )
    {
        // Comp room wall lower
        iTexture = TEX_FLAT14;        
 		vCol0 = vec3(70.0 / 255.0);
        vCol1 = vec3(30.0 / 255.0);
		fPersistence = 1.0;
    }
    else
    if ( iTexture == TEX_FLOOR6_2 )
    {
        // secret shotgun area ceil
        iTexture = TEX_FLAT14;        
 		vCol0 = vec3(120.0 / 255.0);
        vCol1 = vec3(0.0 / 255.0);
		fPersistence = 1.25;
    }

    if ( iTexture == TEX_FLOOR7_2 )
    {
        // Green armor ceil
        iTexture = TEX_FLAT14;
 		vCol0 = vec3(85,89,60)/255.;
        vCol1 = vec3(0.0, 0.0, 0);
		fPersistence = 0.5;
    }
    
    if ( iTexture == TEX_COMPUTE3 )
    {
        iTexture = TEX_COMPUTE2;
    }       

	if(iTexture == TEX_NUKAGE3)
	{
        float fTest = fract(floor(iTime * 6.0) * (1.0 / 3.0));
        if( fTest < 0.3 )
        {
	        vUV += 0.3 * vSize;
        }
        else if(fTest < 0.6)
        {
            vUV = vUV.yx - 0.3; 
        }
        else
        {
            vUV = vUV + 0.45;
        }
	}
    
    if ( iTexture == TEX_STARTAN1 )
    {
        iTexture = TEX_STARTAN3;
        vCol0 = vec3(131.0, 101.0, 75.0) / 255.0;
        vCol1 = vec3(131.0, 101.0, 75.0) / 255.0;
    }
    else
    if ( iTexture == TEX_STARG3 )
    {
        iTexture = TEX_STARTAN3;
		vCol0 = vec3(0.6);
		vCol1 = vec3(123,127,99) / 255.0;
    }
    else
    if ( iTexture == TEX_STARGR1 )
    {
        iTexture = TEX_STARTAN3;
		vCol0 = vec3(0.6);
		vCol1 = vec3(0.6);  
    }
    else
    if ( iTexture == TEX_SW1STRTN )
    {
        iTexture = TEX_STARTAN3;
        vCol0 = vec3(131.0, 101.0, 75.0) / 255.0;
        vCol1 = vec3(131.0, 101.0, 75.0) / 255.0;
    }   
    
    // Should be a sidedef flag
    if ( iOrigTexture == TEX_TEKWALL1 )
    {
        vUV.x = mod(vUV.x + floor(iTime * 50.0), 64.);
    }
    
	
	if(iTexture == TEX_NUKAGE3) { fPersistence = 1.0; }
	if(iTexture == TEX_F_SKY1) { vSize = vec2(256.0, 128.0); fNoise2Freq = 0.3; }
    if(iTexture == TEX_FLOOR7_1 ||
      iTexture == TEX_CEIL5_2 ) { vSize = vec2(64.0, 32.0); fPersistence = 1.0; }	
    if(iTexture == TEX_FLAT5_5) { fPersistence = 3.0; }
    if(iTexture == TEX_FLOOR4_8) { fPersistence = 0.3; }
    if(iTexture == TEX_CEIL3_5) { fPersistence = 0.9; }	
    if(iTexture == TEX_DOOR3) { vSize = vec2(64.0, 72.0); }	
    if(iTexture == TEX_LITE3) { vSize = vec2(32.0, 128.0); }	
    if(iTexture == TEX_STARTAN3) { vSize = vec2(128.0); fPersistence = 1.0; }	
    if(iTexture == TEX_STARGR1) { vSize = vec2(128.0); fPersistence = 1.0; }	    
    if(iTexture == TEX_BIGDOOR2) { vSize = vec2(128.0, 128.0); fPersistence = 0.5; vRandomStreak = vec2(0,128.); }	    
	if(iTexture == TEX_BROWN1) { vSize = vec2(128.0); fPersistence = 0.7; }	
	if(iTexture == TEX_BROWNGRN) { vSize = vec2(128.0); fPersistence = 0.7; }	    
    if(iTexture == TEX_DOORSTOP) { vSize = vec2(8.0, 128.0); fPersistence = 0.7; }
    if(iTexture == TEX_COMPUTE2) { vSize = vec2(256.0, 56.0); fPersistence = 1.5; }
    if(iTexture == TEX_STEP6) { vSize = vec2(32.0, 16.0); fPersistence = 0.9; }
    if(iTexture == TEX_SUPPORT2) { vSize = vec2(64.0, 128.0); }
    if(iTexture == TEX_DOORTRAK) { vSize = vec2(8.0, 128.0); }
    if(iTexture == TEX_TEKWALL1) {  fPersistence = 1.0;vSize = vec2(64.0, 64.0); }
    if(iTexture == TEX_TLITE6_5) { fPersistence = 1.0; vSize = vec2(64.0, 64.0); }
    if(iTexture == TEX_TLITE6_4) { fPersistence = 1.0; vSize = vec2(64.0, 64.0); }
    if(iTexture == TEX_NUKE24) { vSize = vec2(64.0,24.0); }
    if(iTexture == TEX_COMPTILE) { vSize = vec2(128.); vRandomStreak = vec2(16.0, 0); }
    if(iTexture == TEX_PLANET1) { vSize = vec2(256.0, 128.0); vRandomStreak = vec2(0.0, 255.); }
    if(iTexture == TEX_EXITSIGN) { vSize = vec2(64,16); }
	
#ifdef PREVIEW
	     if(fTexture == TEX_DOOR3) {	vSize = vec2(128.0, 128.0); }	
	else if(fTexture == TEX_COMPUTE2) { vSize = vec2(256.0, 64.0); }
#endif
	
	
#ifdef PREVIEW
    vec2 vTexCoord = floor(fract(vUV) * vSize);
#else
    vec2 vTexCoord = fract(vUV / vSize) * vSize;
    #ifdef PIXELATE_TEXTURES
    vTexCoord = floor(vTexCoord);
    #endif
    vTexCoord.y = vSize.y - vTexCoord.y - 1.0;
#endif
    
	float fHRandom = noise1D(vTexCoord.x * fNoise2Freq);
    float fHOffset =  - ((vTexCoord.y) / vSize.y);

    vec2 vRandomCoord = vTexCoord + float(iTexture);
    vRandomCoord += fHRandom * vRandomStreak;
	float fRandom = fbm( vRandomCoord, fPersistence );

	if(iTexture == TEX_NUKAGE3) 	col = TexNukage3( vTexCoord, fRandom );
	if(iTexture == TEX_F_SKY1) 	col = TexFSky1( vTexCoord, fRandom, fHRandom );
    if(iTexture == TEX_FLOOR7_1) 	col = TexFloor7_1( vTexCoord, fRandom, vCol0, vCol1 );
    if(iTexture == TEX_FLAT5_5) 	col = TexFlat5_5( vTexCoord, fRandom );
    if(iTexture == TEX_FLOOR4_8) 	col = TexFloor4_8( vTexCoord, fRandom, vCol0, vCol1 );
    if(iTexture == TEX_CEIL3_5) 	col = TexCeil3_5( vTexCoord, fRandom );
	if(iTexture == TEX_FLAT14) 	col = TexRandom( vTexCoord, fRandom, vCol0, vCol1 );
	if(iTexture == TEX_DOOR3) 		col = TexDoor3( vTexCoord, fRandom, fHRandom + fHOffset);
	if(iTexture == TEX_LITE3) 		col = TexLite3( vTexCoord );
    if(iTexture == TEX_STARTAN3) 	col = TexStartan3( vTexCoord, fRandom, vCol0, vCol1 );
	if(iTexture == TEX_BIGDOOR2) 	col = TexBigDoor2( vTexCoord, fRandom, fHRandom, vCol0 );
    if(iTexture == TEX_BROWN1) 	col = TexBrown1( vTexCoord, fRandom, fHRandom + fHOffset, vCol0, vCol1, vCol2 );
    if(iTexture == TEX_DOORSTOP) 	col = TexDoorstop( vTexCoord, fRandom );
    if(iTexture == TEX_COMPUTE2) 	col = TexCompute2( vTexCoord, fRandom );
    if(iTexture == TEX_STEP6) 		col = TexStep6( vTexCoord, fRandom, fHRandom + fHOffset );
    if(iTexture == TEX_SUPPORT2) 	col = TexSupport2( vTexCoord, fRandom );
	if(iTexture == TEX_DOORTRAK) 	col = TexDoorTrak( vTexCoord, fRandom );
	if(iTexture == TEX_BROWN144) 	col = TexBrown144( vTexCoord, fRandom, fHRandom  + fHOffset );
    if(iTexture == TEX_TEKWALL1)	col = TexTekWall1( vTexCoord, fRandom );
    if(iTexture == TEX_TLITE6_5)	col = TexTLite6_5( vTexCoord, fRandom, vCol0, vCol1 );
    if(iTexture == TEX_TLITE6_4)	col = TexTLite6_4( vTexCoord, fRandom, vCol0, vCol1 );
    if(iTexture == TEX_NUKE24) 	col = TexNuke24( vTexCoord, fRandom, fHRandom );
	if(iTexture == TEX_PLANET1)	col = TexPlanet1( vTexCoord, fRandom, fHRandom );
#ifndef LINUX_WORKAROUND
	if(iTexture == TEX_EXITSIGN)	col = TexExitSign( vTexCoord, fRandom, fHRandom );
#endif
	
    if (iTexture == TEX_COMPTILE)	col = TexCompTile( vTexCoord, fRandom );
        
    if ( iOrigTexture == TEX_SW1STRTN )
    {
        TexAddSwitch( col, vTexCoord, fRandom );
    }
    else
    if (iOrigTexture == TEX_STEP1)
    {
        /// Add lights
        vec2 d = vTexCoord - vec2(16,8);
        col *= max(0.3, 1.0 - 0.005 * dot(d,d) );
    }    
        
    #ifdef QUANTIZE_TEXTURES
    col = Quantize(col, 32.0);
    #endif
   
#ifdef DEBUG_IDENTIFY_TEXTURES
    vec2 vFontUV = fract(_vUV * vec2(.08) * vec2(1., 2.) ) * 3.0 / vec2(1., 2.);
    if ( PrintValue(vFontUV, fOrigTexture, 2., 0.) > 0.0 )
    {
        col = vec3(1.,0,0);
    }
#endif    

    return col;
}

// Function 240
voxel decodeTextel(vec4 textel) {
	voxel o;
    o.id = textel.r;
    o.sunlight = floor(mod(textel.g, 16.));
    o.torchlight = floor(mod(textel.g / 16., 16.));
    o.hue = textel.b;
    return o;
}

// Function 241
float text_legend(vec2 p) {
    
    float c = 0.0;
    
    if (whichF == 2.0) {
        c += glyphCover(p, GLYPH_M1);
        c += glyphCover(p, GLYPH_M2);
        c += glyphCover(p, GLYPH_A); 
        c += glyphCover(p, GLYPH_X); 
        c += glyphCover(p, GLYPH_P1);
    } 
    if (whichF != 1.0) {
        c += glyphCover(p, GLYPH_d); 
        c += glyphCover(p, GLYPH_A); 
    } 
    if (whichF == 2.0) {
        c += glyphCover(p, GLYPH_K); 
    }
    if (whichF != 0.0) {
        c += glyphCover(p, GLYPH_d); 
        c += glyphCover(p, GLYPH_B); 
    }
    if (whichF == 2.0) {
        c += glyphCover(p, GLYPH_P2); 
    }
    return c;
        
}

// Function 242
float text(vec3 p)
{
    p -= vec3 (-1.5 + iTime, 0.0, 7.0);
    p.x = -p.x;
    p.y += sin(iTime + p.x);
    float letterDistField = (SampleFontTex0(p.xy).w - 0.0390625);
    float cropBox = sdBox(p, vec3(1000000.0, 0.5, 0.25));
    return max(letterDistField, cropBox);
}

// Function 243
float NumFont_Four( vec2 vTexCoord )
{
    float fResult = 0.0;
    
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 1), vec2(4,8) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(8, 1), vec2(11,13) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 6), vec2(11,8) ));

    return fResult;
}

// Function 244
float text_t(vec2 U) {
    initMsg;
    U.x+=5.*(0.5-0.2812*(res.x/0.5));
    C(84);C(111);C(116);C(97);C(108);C(126);
    endMsg;
}

// Function 245
vec3 print_text(vec2 U, ivec3 tpl){
    vec4[3] T=vec4[3](vec4(0.),vec4(0.),vec4(0.));
    vec2 oU=U;
    for(int a=0;a<3;a++){
        if(tpl[a]<0)continue;
        bool ibreak=false;
        U=oU+vec2(0.,0.8*float(a));
        for(int i=0;(i<4)&&(!ibreak);i++){
            uvec4 ta=decodeval32(text_group[tpl[a]][i]);
            for(int j=0;(j<4)&&(!ibreak);j++){
                ibreak=ta[j]==0u;
                if(ibreak)break;
                C(U,T[a],int(ta[j]), false);
            }
        }
        if(length(T[a].yz)==0.)T[a].x=0.;
    }
    return vec3(T[0].x,T[1].x,T[2].x);
}

// Function 246
float NumFont_Percent( vec2 vTexCoord )
{
    float fResult = 0.0;
    
    vec2 vClosestRectMin;
    vClosestRectMin.x = clamp( vTexCoord.x, 1.0, 11.0 );
    vClosestRectMin.y = 12.0 - vClosestRectMin.x;
    
    vec2 vClosestRectMax = vClosestRectMin + vec2(0,3); 
    
    vClosestRectMax.y = min( vClosestRectMax.y, 13.0 );
    
    fResult = max( fResult, NumFont_Rect( vTexCoord, vClosestRectMin, vClosestRectMax ));
    
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 1), vec2(3,3) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(9, 11), vec2(11,13) ));
    
    return fResult;
}

// Function 247
int GetGlyphPixelRow(int y, int g) { return glyphs[g + (glyphSize - 1 - y)*glyphCount]; }

// Function 248
vec4 texture_Bspline( sampler2D tex, vec2 t )
{
    vec2 res = iChannelResolution[0].xy;
    vec2 p = res*t - 0.5;
    
    //half pixel offset for quadratic to make it line up with others
    if(ScreenH == 1.f){
     	p -= .5;   
    }
    vec2 f = fract(p);
    vec2 i = floor(p);

    return spline( f.y, spline( f.x, SAM(-1,-1), SAM( 0,-1), SAM( 1,-1), SAM( 2,-1)),
                        spline( f.x, SAM(-1, 0), SAM( 0, 0), SAM( 1, 0), SAM( 2, 0)),
                        spline( f.x, SAM(-1, 1), SAM( 0, 1), SAM( 1, 1), SAM( 2, 1)),
                        spline( f.x, SAM(-1, 2), SAM( 0, 2), SAM( 1, 2), SAM( 2, 2)));
}

// Function 249
vec4 textureBleed(vec2 uv, vec2 sz)
{
    vec2 rsz = 1.0/sz;
    vec4 t1 = textureLowRes(uv + vec2(-1,-1)*rsz, sz);
    vec4 t2 = textureLowRes(uv + vec2( 0,-1)*rsz, sz);
    vec4 t3 = textureLowRes(uv + vec2( 1,-1)*rsz, sz);
    vec4 t4 = textureLowRes(uv + vec2(-1, 0)*rsz, sz);
    vec4 t5 = textureLowRes(uv + vec2( 0, 0)*rsz, sz);
    vec4 t6 = textureLowRes(uv + vec2( 1, 0)*rsz, sz);
    vec4 t7 = textureLowRes(uv + vec2(-1, 1)*rsz, sz);
    vec4 t8 = textureLowRes(uv + vec2( 0, 1)*rsz, sz);
    vec4 t9 = textureLowRes(uv + vec2( 1, 1)*rsz, sz);
    
    /* TODO: calculate */
    vec4 tr = t5;
    
    return tr;
}

// Function 250
void fontDemo(inout vec4 k, const in vec2 p, float rf) {

	float t = iTime;
    vec2 scT = vec2(sin(t),cos(t)),
               ir = iResolution.xy,
               ir2 = ir/2.;
    vec3 v = vec3(vec2(ir2.x,.65*ir.y),0);
    
    for(int i=0; i<26; i++)
    {
        float f = float(i);
	    vec4 rainbow = palette(mod(f+2., 10.));
        
	    renderChar(k, p, 
                   v + vec3(ellipse(f, t/2., vec2(ir2.x/2.,80./rf)), 90-i), 
                   vec4(scT.xy,mod(f,2.),1));

        renderChar(k, p, 
                   v + vec3(ellipse(f, .1+t/2., vec2(ir2.x/2.,80./rf)), 122-i), 
                   vec4(scT.xy,mod(f,2.),1));
        
	    renderChar(k, p, 
                   v + vec3(ellipse(f,-t, vec2(60)/rf), 97+i), 
                   rainbow);
    }

    msg(k, p, vec3(ir2.x-130., .25*ir.y, 0), true);
    msg(k, p, vec3(ir2.x-130., .25*ir.y, 0), false);
}

// Function 251
float text_nui(vec2 U) {
    initMsg;
    U.x+=4.*(0.5-0.2812*(res.x/0.5));
    C(110);C(111);C(32);C(85);C(73);
    endMsg;
}

// Function 252
vec4 textureSky(vec2 uv) {
    const vec2 RES = vec2(8.0, 32.0);    
    float n = noise1(uv * RES);
    n = n * 0.05 + 0.8;
    return vec4(0.5,n*1.0,n*1.1,1.0);
}

// Function 253
vec2 font_from_screen(vec2 tpos, float font_size, vec2 char_pos) {
    return tpos*TEXELS_PER_UNIT/font_size + (char_pos+0.5)*TEXELS_PER_UNIT;
}

// Function 254
float text_pat(vec2 U) {
    initMsg;C(43);C(65);C(84);endMsg;
}

// Function 255
vec3 getTexture(vec2 p){
	vec4 s = texture(iChannel0, p);
    return s.xyz * s.w;
}

// Function 256
vec3 getTexture(in vec3 p, in float m, in Legoman lego) {
	p = ManRef(p);
	vec3 c;   
  
    if (m == HEAD) {
		c = vec3(1.,1.,0);
#ifdef ANIMATION	
    #ifdef PRECALCULATE
		p.xz*= lego.rhead;
	#else
        p.xz*= Rot(lego.head);
	#endif		
#endif		
		if (p.z<0.) { // draw face			
			vec2 p2 = p.xy;
			p2.y -= 1.46;
			p2 *= 100.; // scale because 
			float face_r = 27.;
			float face_x = face_r*0.453596121, //face_r*cos(a); // precalcul
				  face_y = -face_r*0.89120736; //face_r*sin(a); // precalcul
			float px = abs(p2.x);
			float e = 4.-.08*px;
			float v = (px<face_x && p2.y<-e) ? abs(length(p2)-face_r)-e : 
					  (p2.y<-e) ? length(vec2(px,p2.y)-vec2(face_x,face_y))-e :
					  length(vec2(px,p2.y)-vec2(face_x,-face_y*.1))-1.8*e; 
			v = clamp(v, 0., 1.);
			c = mix(vec3(0), c, v);
		}
    } else {
        c = m == HAND   ? vec3(1.,1.,0.) :
			m == SHIRT  ? lego.c_shirt :
			m == MIDDLE ? lego.c_middle :
			m == LEGS   ? lego.c_legs : lego.c_arms;
    }
	return c;
}

// Function 257
vec4 SampleTextureBilinearlyAndUnpack(sampler2D tex, vec2 uv)
{
    vec4 sample_color = texture(tex, uv, 0.0);
#ifdef PACK_SIGNED_TO_UNSIGNED
    sample_color = 2.0 * sample_color - 1.0;
#endif // PACK_SIGNED_TO_UNSIGNED
    return sample_color;
}

// Function 258
vec3 sampleTextureWithFilter( in vec3 uvw, in vec3 ddx_uvw, in vec3 ddy_uvw, in vec3 nor, in float mid )
{
    int sx = 1 + int( clamp( 4.0*length(ddx_uvw-uvw), 0.0, float(MaxSamples-1) ) );
    int sy = 1 + int( clamp( 4.0*length(ddy_uvw-uvw), 0.0, float(MaxSamples-1) ) );

	vec3 no = vec3(0.0);

	#if 1
    for( int j=0; j<MaxSamples; j++ )
    for( int i=0; i<MaxSamples; i++ )
    {
        if( j<sy && i<sx )
        {
            vec2 st = vec2( float(i), float(j) ) / vec2( float(sx),float(sy) );
            no += mytexture( uvw + st.x*(ddx_uvw-uvw) + st.y*(ddy_uvw-uvw), nor, mid );
        }
    }
    #else
    for( int j=0; j<sy; j++ )
    for( int i=0; i<sx; i++ )
    {
        vec2 st = vec2( float(i), float(j) )/vec2(float(sx),float(sy));
        no += mytexture( uvw + st.x * (ddx_uvw-uvw) + st.y*(ddy_uvw-uvw), nor, mid );
    }
    #endif		

	return no / float(sx*sy);
}

// Function 259
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

// Function 260
int text( in vec2 uv, const float size )
{
    vec2 charSize = vec2( size*vec2(MAP_SIZE)/iResolution.y );
    float spaceSize = float( size*float(MAP_SIZE.x+1)/iResolution.y );
        
    // and a starting position.
    vec2 charPos = vec2(0.05, 0.90);
    // Draw some text!
    int chr = 0;
    // Bitmap text rendering!
    chr += drawChar( CH_B, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_I, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_T, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_M, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_A, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_P, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_BLNK, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_T, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_E, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_X, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_T, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_BLNK, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_R, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_E, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_N, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_D, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_E, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_R, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_I, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_N, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_G, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_EXCL, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_EXCL, charPos, charSize, uv); charPos.x += spaceSize;
    
    // Today's Date: {date}
    charPos = vec2(0.05, .75);
    chr += drawChar( CH_T, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_O, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_D, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_A, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_Y, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_APST, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_S, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_BLNK, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_D, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_A, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_T, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_E, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_BLNK, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_LPAR, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_M, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_M, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_HYPH, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_D, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_D, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_HYPH, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_Y, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_Y, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_Y, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_Y, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_RPAR, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_COLN, charPos, charSize, uv); charPos.x += .1;
    // The date itself.
    charPos.x += .3;
    chr += drawIntCarriage( int(iDate.x), charPos, charSize, uv, 4);
    chr += drawChar( CH_HYPH, charPos, charSize, uv); charPos.x-=spaceSize;
    chr += drawIntCarriage( int(iDate.z)+1, charPos, charSize, uv, 2);
    chr += drawChar( CH_HYPH, charPos, charSize, uv); charPos.x-=spaceSize;
    chr += drawIntCarriage( int(iDate.y)+1, charPos, charSize, uv, 2);
    
    // Shader uptime:
    charPos = vec2(0.05, .6);
    chr += drawChar( CH_I, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_G, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_L, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_O, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_B, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_A, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_L, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_T, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_I, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_M, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_E, charPos, charSize, uv); charPos.x += spaceSize;
    chr += drawChar( CH_COLN, charPos, charSize, uv); charPos.x += spaceSize;
    // The uptime itself.
    charPos.x += .3;
    //chr += drawFixed( iTime, 2, charPos, charSize, uv);
    chr += drawFixed( 0.0, 2, charPos, charSize, uv);
    return chr;
}

// Function 261
vec4 BicubicTexture(in sampler2D tex, in vec2 coord)
{
	vec2 resolution = iResolution.xy;

	coord *= resolution;

	float fx = fract(coord.x);
    float fy = fract(coord.y);
    coord.x -= fx;
    coord.y -= fy;

    fx -= 0.5;
    fy -= 0.5;

    vec4 xcubic = cubic(fx);
    vec4 ycubic = cubic(fy);

    vec4 c = vec4(coord.x - 0.5, coord.x + 1.5, coord.y - 0.5, coord.y + 1.5);
    vec4 s = vec4(xcubic.x + xcubic.y, xcubic.z + xcubic.w, ycubic.x + ycubic.y, ycubic.z + ycubic.w);
    vec4 offset = c + vec4(xcubic.y, xcubic.w, ycubic.y, ycubic.w) / s;

    vec4 sample0 = texture(tex, vec2(offset.x, offset.z) / resolution);
    vec4 sample1 = texture(tex, vec2(offset.y, offset.z) / resolution);
    vec4 sample2 = texture(tex, vec2(offset.x, offset.w) / resolution);
    vec4 sample3 = texture(tex, vec2(offset.y, offset.w) / resolution);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix( mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

// Function 262
float getcloudtexture(vec3 p)
{   
    float c = 0.0;
    float weightsum = 0.0;
    float weight = 0.6;
    vec3 scale = vec3(0.0005, 0.0, 0.002);
    
    p+=vec3(0.1, 0.0, 0.2)*iTime*100.0;
    
    for (int i=0; i<4; i++)
    {
        weightsum+=weight;
        //c += texture(iChannel1, p.xz*scale.z*1.0).r*weight;
    	c += (noise(p*scale)*weight + noise(p*scale + YAXIS*1.0)*weight)*0.5;
        scale *= 1.9;
        weight *= 0.45;
    }
    c/=weightsum;
    return c;
}

// Function 263
vec4 texturei(sampler2D sampler, int texture_index, vec2 uv)
{ vec2 st = vec2
  ( float(texture_index % TEXTURE_RESOLUTION) / float(TEXTURE_RESOLUTION)
  , float(texture_index / TEXTURE_RESOLUTION) / float(TEXTURE_RESOLUTION)
  );
  uv = uv / float(TEXTURE_RESOLUTION);
  return texture(sampler, st + uv);
}

// Function 264
float text_ko(vec2 U) {
    initMsg;C(75);C(79);endMsg;
}

// Function 265
vec3 tileTexture(vec2 uv){
    float f;
    vec3 col = vec3(1.);
    vec2 st = (fract(uv*4.)*2.-1.);
    
    f = pow(1.0-abs(st.x*st.y),80.);
    f = pow(max(abs(st.x),abs(st.y)),15.)-0.2;
    
    col = mix(col, vec3(0.), f);
    return col;
}

// Function 266
float getTextureIndex(float _v)
{
    return floor(_v * TEXTURES_NUM);
}

// Function 267
float titleText( vec2 p )
{        
    vec2 scale = vec2( 4., 8. );
    vec2 t = floor( p / scale );   
    
    uint v = 0u;
	v = t.y == 0. ? ( t.x < 5. ? 2397642579u : ( t.x < 9. ? 2142969413u : ( t.x < 22. ? 2163282770u : ( t.x < 26. ? 2280202016u : ( t.x < 30. ? 2414090057u : 27477u ) ) ) ) ) : v;
	v = t.x >= 0. && t.x < 24. ? v : 0u;
    
	float c = float( ( v >> uint( 8. * t.x ) ) & 255u );
    
    p = ( p - t * scale ) / scale;
    p.x = ( p.x - .5 ) * .5 + .5;
    float sdf = textSDF( p, c );
    return ( c != 0. ) ? smoothstep( -.05, +.05, sdf ) : 1.0;
}

// Function 268
mat3 glyph_8_9_lowercase(float g) {
    // lowercase ==================================================
    GLYPH( 97)00000000.,00000000.,00000000.,00001110.,00010010.,00010010.,00001101.,0,0);
    GLYPH( 98)00000000.,00100000.,00100000.,00111100.,00100010.,00100010.,00111100.,0,0);
    GLYPH( 99)00000000.,00000000.,00000000.,00011100.,00100000.,00100010.,00011100.,0,0);
    GLYPH(100)00000010.,00000010.,00000010.,00011010.,00100110.,00100010.,00011101.,0,0);
    GLYPH(101)00000000.,00000000.,00111000.,01000100.,01111100.,01000000.,00111100.,0,0);
    GLYPH(102)00011000.,00100100.,00100000.,00111000.,00100000.,00100000.,00100000.,0,0);
    GLYPH(103)00000000.,00011000.,00100100.,00100100.,00011100.,00100100.,00011000.,0,0);
    GLYPH(104)00100000.,00100000.,00100000.,00101100.,00110010.,00100010.,00100010.,0,0);
    GLYPH(105)00000000.,00010000.,00000000.,00010000.,00010000.,00010000.,00011000.,0,0);
    GLYPH(106)00000100.,00000000.,00000100.,00000100.,00000100.,00100100.,00011000.,0,0);
    GLYPH(107)01000000.,01000000.,01000000.,01001000.,01110000.,01010000.,01001000.,0,0);
    GLYPH(108)00100000.,00100000.,00100000.,00100000.,00100000.,00100000.,00011100.,0,0);
    GLYPH(109)00000000.,00000000.,00100100.,01011010.,01000010.,01000010.,01000010.,0,0);
    GLYPH(110)00000000.,00000000.,01011000.,01100100.,01000010.,01000010.,01000010.,0,0);
    GLYPH(111)00000000.,00000000.,00011100.,00100010.,00100010.,00100010.,00011100.,0,0);
    GLYPH(112)00000000.,00000000.,01011100.,01100010.,01100010.,01011100.,01000000.,01000000.,00000000);
    GLYPH(113)00000000.,00000000.,00111010.,01000110.,01000110.,00111010.,00000010.,00000011.,00000000);
    GLYPH(114)00000000.,00000000.,00101100.,00110010.,00100000.,00100000.,00100000.,0,0);
    GLYPH(115)00000000.,00000000.,00011100.,00100000.,00011100.,00000010.,00100010.,00011100.,00000000);
    GLYPH(116)00010000.,00010000.,00011100.,00010000.,00010000.,00010010.,00001100.,0,0);
    GLYPH(117)00000000.,00000000.,00100010.,00100010.,00100010.,00100010.,00011100.,0,0);
    GLYPH(118)00000000.,00000000.,00100010.,00100010.,00010100.,00010100.,00001000.,0,0);
    GLYPH(119)00000000.,00000000.,00100010.,00100010.,00101010.,00101010.,00010100.,0,0);
    GLYPH(120)00000000.,00000000.,00100010.,00010100.,00001000.,00010100.,00100010.,0,0);
    GLYPH(121)00000000.,00000000.,00100010.,00100010.,00100110.,00011010.,00000010.,00011100.,00000000);
    GLYPH(122)00000000.,00000000.,00111110.,00000100.,00001000.,00010000.,00111110.,0,0);
    return mat3(0);
}

// Function 269
float text_n(vec2 U, float num) {
    if (num < 1.)return text_ko(U);
    initMsg;
    num = floor(num);
    int maxloop = 2;
    bool x = false;
    if (num < 10.) {
        num = num * 10.;
    } else {
        num = floor(num / 10.)+(num * 10. - floor(num / 10.)*100.);
        if ((num < 10.))x = true;
    }
    while (num >= 1.0) {
        if (maxloop-- < 1)break;
        C((48 + int(num) % 10));
        if (x) {
            C(48);
            x = false;
        }
        num /= 10.0;
    }
    endMsg;
}

// Function 270
float outputText(){return text.w;}

// Function 271
vec2 font_from_screen(vec2 tpos, vec2 char_pos) {    
    return (tpos + char_pos + 0.5)/GLYPHS_PER_UV;
}

// Function 272
float glyphs(vec2 p,vec3 i){
 p*=.2;
 float px=16.1;
 float py=8.4;
 SetTextPosition(p,px,py);
 SetGlyphColor (0., 0., 0.);
 float r=0.;
 if(i.z>0.){
  p.y=-p.y;
  py+=1.;
  SetGlyphColor (0.5, 0.5, 0.5);
 }
    
 SetTextPosition(p,px,py);
 r+=drawInt(int(i.x),3);
 SetTextPosition(p,px,py+1.);
 r+=drawInt(int(i.y),3);
 SetTextPosition(p,px,py+2.);
 r+=drawInt(int(i.z),1);
 return r;
}

// Function 273
vec3 ScrollText1()
{
  tp = uv / FONT_SIZE1;  // set font size
  tp.x = 2.0*(tp.x -4.0 +mod(time*SCROLL_SPEED, SCROLL_LEN));
  float SIN_AMP = 1.5 * iMouse.y  / iResolution.y - 0.5;
  tp.y += 1.0 +SIN_AMP*sin(tp.x*SIN_FREQ +time*SIN_SPEED);

  float c = 0.0;

  _A _n _ _A _m _i _g _a _ _l _i _k _e _ _d _e _m _o _ _c _o _l _l _e _c _t _i _o _n 
      
  _ _w _i _t _h _ _3 _d _ _a _n _t _i _a _l _i _a _s _e _d _  _s _i _n _u _s 
      
  _ _s _c _r _o _l _l _e _r _ _u _s _i _n _g
     
  _ _s _h _a _d _e _r _t _o _y _ _f _o _n _t _ _t _e _x _t _u _r _e 
    
  _ _smily _

  _ _H _o _l _d _ _k _e _y _ _1 _dot _dot _4 _ _t _o 

  _ _w _a _t _c _h _ _o _n _l _y _ _s _e _l _e _c _t _e _d _ _d _e _m _o _ _dot _dot _dot _dot

  vec3 fcol = c * vec3(pos, 0.5+0.5*sin(2.0*time));    
  if (c >= 0.5) return fcol; 
  return mix (aColor, fcol, c);
}

// Function 274
vec2 glyph3(vec2 u,vec2 m){u.y-=m.y*3.-.4;u.x-=m.x*2.-.2;return glyph2(u,m);}

// Function 275
vec4 SampleTextureCatmullRom4Samples(sampler2D tex, vec2 uv, vec2 texSize)
{
    // Based on the standard Catmull-Rom spline: w1*C1+w2*C2+w3*C3+w4*C4, where
    // w1 = ((-0.5*f + 1.0)*f - 0.5)*f, w2 = (1.5*f - 2.5)*f*f + 1.0,
    // w3 = ((-1.5*f + 2.0)*f + 0.5)*f and w4 = (0.5*f - 0.5)*f*f with f as the
    // normalized interpolation position between C2 (at f=0) and C3 (at f=1).
 
    // half_f is a sort of sub-pixelquad fraction, -1 <= half_f < 1.
    vec2 half_f     = 2.0 * fract(0.5 * uv * texSize - 0.25) - 1.0;
 
    // f is the regular sub-pixel fraction, 0 <= f < 1. This is equivalent to
    // fract(uv * texSize - 0.5), but based on half_f to prevent rounding issues.
    vec2 f          = fract(half_f);
 
    vec2 s1         = ( 0.5 * f - 0.5) * f;            // = w1 / (1 - f)
    vec2 s12        = (-2.0 * f + 1.5) * f + 1.0;      // = (w2 - w1) / (1 - f)
    vec2 s34        = ( 2.0 * f - 2.5) * f - 0.5;      // = (w4 - w3) / f
 
    // positions is equivalent to: (floor(uv * texSize - 0.5).xyxy + 0.5 +
    // vec4(-1.0 + w2 / (w2 - w1), 1.0 + w4 / (w4 - w3))) / texSize.xyxy.
    vec4 positions  = vec4((-f * s12 + s1      ) / (texSize * s12) + uv,
                           (-f * s34 + s1 + s34) / (texSize * s34) + uv);
 
    // Determine if the output needs to be sign-flipped. Equivalent to .x*.y of
    // (1.0 - 2.0 * floor(t - 2.0 * floor(0.5 * t))), where t is uv * texSize - 0.5.
    float sign_flip = half_f.x * half_f.y > 0.0 ? 1.0 : -1.0;
 
    vec4 w          = vec4(-f * s12 + s12, s34 * f); // = (w2 - w1, w4 - w3)
    vec4 weights    = vec4(w.xz * (w.y * sign_flip), w.xz * (w.w * sign_flip));
 
    return SampleTextureBilinearlyAndUnpack(tex, positions.xy) * weights.x +
           SampleTextureBilinearlyAndUnpack(tex, positions.zy) * weights.y +
           SampleTextureBilinearlyAndUnpack(tex, positions.xw) * weights.z +
           SampleTextureBilinearlyAndUnpack(tex, positions.zw) * weights.w;
}

// Function 276
float gridTexture( in vec2 p )
{
	// filter kernel
    vec2 w = fwidth(p) + 0.01;

	// analytic (box) filtering
    vec2 a = p + 0.5*w;                        
    vec2 b = p - 0.5*w;           
    vec2 i = (floor(a)+min(fract(a)*N,1.0)-
              floor(b)-min(fract(b)*N,1.0))/(N*w);
    //pattern
    return (1.0-i.x)*(1.0-i.y);
}

// Function 277
void ValueText(inout vec4 col, vec2 uv0, float n)
{
    vec2 p = uv0 * 0.5;
    
    vec2 scale = vec2(4., 8.);
    vec2 t = floor(p / scale);   

    if(t.x < 0.0 || t.x > 5.0 || t.y != 0.0) return;
    if((n == 0.0 || abs(n) == 1.0) && t.x > 1.0) return;
    if(t.x == 0.0 && n >= 0.0) return;
    

    float c = 0.0;
    
    if(t.x == 0.0) c = 45.0;
    else if(t.x == 1.0) c = n == 0.0 ? 48.0 : (abs(n) == 1.0 ? 49.0 : 46.0);
    else
    c = abs(n) == 1.0 ? 48.0 : 48.0 + mod(floor(abs(n)*1000.0 * exp2((4.0-t.x) * -(log2(10.0)/log2(2.0)))), 10.0);

    p = (p - t * scale) / scale;
    p.x = (p.x - 0.5) * 0.5 + 0.5;

    if (c == 0.) return;
    
    float sdf = TextSDF(p, c);
    
    sdf = smoothstep(-0.05, 0.05, sdf);

    col.r = (1.0 - sdf) * 0.75;
}

// Function 278
float text(vec2 uv)
{
    float col = 0.0;
    
    vec2 center = res/2.0;
    
    float hour = floor(iDate.w/60.0/60.0);
    float minute = floor(mod(iDate.w/60.0,60.0));
    float second = floor(mod(iDate.w,60.0));
    
    //Greeting Text
    
    print_pos = floor(vec2(STRWIDTH(5.0),res.y - STRHEIGHT(5.0)));
       
    col += char(ch_Y,uv);
    col += char(ch_o,uv);
    col += char(ch_u,uv);
    col += char(ch_spc,uv);
    col += char(ch_a,uv);
    col += char(ch_r,uv);
    col += char(ch_e,uv);
    col += char(ch_spc,uv);
    col += char(ch_i,uv);
    col += char(ch_n,uv);
    col += char(ch_spc,uv);
    col += char(ch_a,uv);
    col += char(ch_n,uv);
    col += char(ch_spc,uv);
    col += char(ch_i,uv);
    col += char(ch_n,uv);
    col += char(ch_s,uv);
    col += char(ch_t,uv);
    col += char(ch_a,uv);
    col += char(ch_n,uv);
    col += char(ch_c,uv);
    col += char(ch_e,uv);
    col += char(ch_d,uv);
    col += char(ch_dsh,uv);
    col += char(ch_r,uv);
    col += char(ch_e,uv);
    col += char(ch_n,uv);
    col += char(ch_d,uv);
    col += char(ch_e,uv);
    col += char(ch_r,uv);
    col += char(ch_e,uv);
    col += char(ch_d,uv);
    col += char(ch_spc,uv);
    col += char(ch_f,uv);
    col += char(ch_o,uv);
    col += char(ch_r,uv);
    col += char(ch_e,uv);
    col += char(ch_s,uv);
    col += char(ch_t,uv);
    col += char(ch_per,uv);
    
    print_pos = floor(vec2(STRWIDTH(5.0),res.y - STRHEIGHT(7.0)));
    col += char(ch_grt,uv);
    col += char(ch_usc,uv);

    return col;
}

// Function 279
vec4 textureHermite(sampler2D tex, vec2 uv, vec2 res)
{
	uv = uv*res + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*(3.0-2.0*fuv);
	uv = (uv - 0.5)/res;
	return texture( tex, uv );
}

// Function 280
vec4 texturenoiseLod( vec3 r, float lod )
{
    vec3 uvw = r / iChannelResolution[3];
    return textureLod( iChannel3, uvw, lod ) * 2. - 1.;
}

// Function 281
mat3 glyph_8_9_lowercase(float g) {
    // lowercase ==================================================
    GLYPH( 97)0x00000000,0x00000000,0x00000000,0x00001110,0x00010010,0x00010010,0x00001101,0,0);
    GLYPH( 98)0x00000000,0x00100000,0x00100000,0x00111100,0x00100010,0x00100010,0x00111100,0,0);
    GLYPH( 99)0x00000000,0x00000000,0x00000000,0x00011100,0x00100000,0x00100010,0x00011100,0,0);
    GLYPH(100)0x00000010,0x00000010,0x00000010,0x00011010,0x00100110,0x00100010,0x00011101,0,0);
    GLYPH(101)0x00000000,0x00000000,0x00111000,0x01000100,0x01111100,0x01000000,0x00111100,0,0);
    GLYPH(102)0x00011000,0x00100100,0x00100000,0x00111000,0x00100000,0x00100000,0x00100000,0,0);
    GLYPH(103)0x00000000,0x00011000,0x00100100,0x00100100,0x00011100,0x00100100,0x00011000,0,0);
    GLYPH(104)0x00100000,0x00100000,0x00100000,0x00101100,0x00110010,0x00100010,0x00100010,0,0);
    GLYPH(105)0x00000000,0x00010000,0x00000000,0x00010000,0x00010000,0x00010000,0x00011000,0,0);
    GLYPH(106)0x00000100,0x00000000,0x00000100,0x00000100,0x00000100,0x00100100,0x00011000,0,0);
    GLYPH(107)0x01000000,0x01000000,0x01000000,0x01001000,0x01110000,0x01010000,0x01001000,0,0);
    GLYPH(108)0x00100000,0x00100000,0x00100000,0x00100000,0x00100000,0x00100000,0x00011100,0,0);
    GLYPH(109)0x00000000,0x00000000,0x00100100,0x01011010,0x01000010,0x01000010,0x01000010,0,0);
    GLYPH(110)0x00000000,0x00000000,0x01011000,0x01100100,0x01000010,0x01000010,0x01000010,0,0);
    GLYPH(111)0x00000000,0x00000000,0x00011100,0x00100010,0x00100010,0x00100010,0x00011100,0,0);
    GLYPH(112)0x00000000,0x00000000,0x01011100,0x01100010,0x01100010,0x01011100,0x01000000,0x01000000,0x00000000);
    GLYPH(113)0x00000000,0x00000000,0x00111010,0x01000110,0x01000110,0x00111010,0x00000010,0x00000011,0x00000000);
    GLYPH(114)0x00000000,0x00000000,0x00101100,0x00110010,0x00100000,0x00100000,0x00100000,0,0);
    GLYPH(115)0x00000000,0x00000000,0x00011100,0x00100000,0x00011100,0x00000010,0x00100010,0x00011100,0x00000000);
    GLYPH(116)0x00010000,0x00010000,0x00011100,0x00010000,0x00010000,0x00010010,0x00001100,0,0);
    GLYPH(117)0x00000000,0x00000000,0x00100010,0x00100010,0x00100010,0x00100010,0x00011100,0,0);
    GLYPH(118)0x00000000,0x00000000,0x00100010,0x00100010,0x00010100,0x00010100,0x00001000,0,0);
    GLYPH(119)0x00000000,0x00000000,0x00100010,0x00100010,0x00101010,0x00101010,0x00010100,0,0);
    GLYPH(120)0x00000000,0x00000000,0x00100010,0x00010100,0x00001000,0x00010100,0x00100010,0,0);
    GLYPH(121)0x00000000,0x00000000,0x00100010,0x00100010,0x00100110,0x00011010,0x00000010,0x00011100,0x00000000);
    GLYPH(122)0x00000000,0x00000000,0x00111110,0x00000100,0x00001000,0x00010000,0x00111110,0,0);
    return mat3(0);
}

// Function 282
vec3 BicubicHermiteTextureSample (vec2 P)
{
    vec2 pixel = P * c_textureSize + 0.5;
    
    vec2 frac = fract(pixel);
    pixel = floor(pixel) / c_textureSize - vec2(c_onePixel/2.0);
    
    vec3 C00 = texture(iChannel0, pixel + vec2(-c_onePixel ,-c_onePixel)).rgb;
    vec3 C10 = texture(iChannel0, pixel + vec2( 0.0        ,-c_onePixel)).rgb;
    vec3 C20 = texture(iChannel0, pixel + vec2( c_onePixel ,-c_onePixel)).rgb;
    vec3 C30 = texture(iChannel0, pixel + vec2( c_twoPixels,-c_onePixel)).rgb;
    
    vec3 C01 = texture(iChannel0, pixel + vec2(-c_onePixel , 0.0)).rgb;
    vec3 C11 = texture(iChannel0, pixel + vec2( 0.0        , 0.0)).rgb;
    vec3 C21 = texture(iChannel0, pixel + vec2( c_onePixel , 0.0)).rgb;
    vec3 C31 = texture(iChannel0, pixel + vec2( c_twoPixels, 0.0)).rgb;    
    
    vec3 C02 = texture(iChannel0, pixel + vec2(-c_onePixel , c_onePixel)).rgb;
    vec3 C12 = texture(iChannel0, pixel + vec2( 0.0        , c_onePixel)).rgb;
    vec3 C22 = texture(iChannel0, pixel + vec2( c_onePixel , c_onePixel)).rgb;
    vec3 C32 = texture(iChannel0, pixel + vec2( c_twoPixels, c_onePixel)).rgb;    
    
    vec3 C03 = texture(iChannel0, pixel + vec2(-c_onePixel , c_twoPixels)).rgb;
    vec3 C13 = texture(iChannel0, pixel + vec2( 0.0        , c_twoPixels)).rgb;
    vec3 C23 = texture(iChannel0, pixel + vec2( c_onePixel , c_twoPixels)).rgb;
    vec3 C33 = texture(iChannel0, pixel + vec2( c_twoPixels, c_twoPixels)).rgb;    
    
    vec3 CP0X = CubicHermite(C00, C10, C20, C30, frac.x);
    vec3 CP1X = CubicHermite(C01, C11, C21, C31, frac.x);
    vec3 CP2X = CubicHermite(C02, C12, C22, C32, frac.x);
    vec3 CP3X = CubicHermite(C03, C13, C23, C33, frac.x);
    
    return CubicHermite(CP0X, CP1X, CP2X, CP3X, frac.y);
}

// Function 283
float text_a(vec2 U) {
    initMsg;C(65);C(84);C(58);endMsg;
}

// Function 284
float text_w(vec2 U) {
    initMsg;C(86);C(105);C(99);C(116);C(111);C(114);C(121);endMsg;
}

// Function 285
vec4 fontTextureLookup(vec2 xy)
{
    /* low quality font lookup */
    /*return texture(fontChannel,xy);*/
    
    /* high quality font lookup*/
	float dxy = 1024.*1.5;
	vec2 dx = vec2(1.,0.)/dxy;
	vec2 dy = vec2(0.,1.)/dxy;

    return 
        (    texture(fontChannel,xy + dx + dy)
        +    texture(fontChannel,xy + dx - dy)
        +    texture(fontChannel,xy - dx - dy)
        +    texture(fontChannel,xy - dx + dy)
        + 2.*texture(fontChannel,xy)
        )/6.
    ;
}

// Function 286
vec4 texture_Bicubic( sampler2D tex, vec2 t )
{
    vec2 res = iChannelResolution[1].xy;
    vec2 p = res*t - 0.5;
    vec2 f = fract(p);
    vec2 i = floor(p);

    return spline( f.y, spline( f.x, SAM(-1,-1), SAM( 0,-1), SAM( 1,-1), SAM( 2,-1)),
                        spline( f.x, SAM(-1, 0), SAM( 0, 0), SAM( 1, 0), SAM( 2, 0)),
                        spline( f.x, SAM(-1, 1), SAM( 0, 1), SAM( 1, 1), SAM( 2, 1)),
                        spline( f.x, SAM(-1, 2), SAM( 0, 2), SAM( 1, 2), SAM( 2, 2)));
}

// Function 287
float text_n(vec2 U) {
    initMsg;
    U.x+=4.*(0.5-0.2812*(res.x/0.5)); //3.+1.
    C(78);C(111);C(110);C(101);
    endMsg;
}

// Function 288
vec2 glyph2(vec2 u,vec2 m){vec2 c=vec2(0)
 ;c.x=dd(vec2(abs(u.x)-m.x-m.y,stretch(u.y,m.y*3.)))
 ;c.y=dd(vec2(stretch(u.x,m.x),abs(u.y)-m.y))
 ;c.x=miv(c)
 ;u.x=abs(u.x)-m.x//+m.y //thee +m.y makes the mirror symmetry weird
 ;c.y=dd(u)
 ;return c;}

// Function 289
float show_text(vec2 pos,ivec2 text, float text_size)
{
    //inside text
    if(abs(pos.y) < text_size && abs(pos.x) < text_size * 8.)
    {
        //get glyph
        int i = 7 - int(floor((pos.x + text_size * 8.) / text_size / 2.));
        
        //lookup glyph codepoint
        float glyph_codepoint = float(i > 3 ? (text.x >> ((i-4) * 8)) & 255 : (text.y >> (i * 8)) & 255);
        
        //get glyph uv
        vec2 gly_uv = fract((pos + vec2(8.*text_size,text_size)) / text_size / 2.);
        
        //zoom in on glyph to make it larger (causes some clipping)
        gly_uv = (gly_uv -0.5) * vec2(0.6,0.6) + 0.5;
       
        //lookup glyph sdf
        return get_glyph(gly_uv, glyph_codepoint) - 0.5;
    }
    //not inside text
    return 1.0;
}

// Function 290
vec3 BilinearTextureSample (vec2 P)
{
    vec2 pixel = P * c_textureSize + 0.5;
    
    vec2 frac = fract(pixel);
    pixel = (floor(pixel) / c_textureSize) - vec2(c_onePixel/2.0);

    vec3 C11 = texture(iChannel0, pixel + vec2( 0.0        , 0.0)).rgb;
    vec3 C21 = texture(iChannel0, pixel + vec2( c_onePixel , 0.0)).rgb;
    vec3 C12 = texture(iChannel0, pixel + vec2( 0.0        , c_onePixel)).rgb;
    vec3 C22 = texture(iChannel0, pixel + vec2( c_onePixel , c_onePixel)).rgb;

    vec3 x1 = mix(C11, C21, frac.x);
    vec3 x2 = mix(C12, C22, frac.x);
    return mix(x1, x2, frac.y);
}

// Function 291
float font(DataFont f){
    FontUV.x -= 3.+ float(adjacency_width);
    return arrayBin(f);
}

// Function 292
vec4 textureUgly(in sampler2D tex, in vec2 uv, in vec2 res) {
    return textureLod(tex, (floor(uv*res)+.5)/res, 0.0).xxxx;
}

// Function 293
vec4 getTexture(float id, vec2 c) {
    vec2 gridPos = vec2(mod(id, 8.), floor(id / 8.));
	return textureLod(iChannel2, 16. * (c + gridPos) / iChannelResolution[3].xy, 0.0);
}

// Function 294
vec3 texture_cube(vec3 usphp, float scale)
{
    // cheap way of turning a 2D texture into a cube
    // map lookup by ray casting sphere intersection direction 
    // to the appropriate plane of the surrounding cube and then
    // using the uv of that plane intersection as a 2d vector.
    
    vec3 p = usphp;
#if PROJECTION_TYPE == SLIDING_PROJECTION
    p.y -= sin(.1 * iTime);
#endif
    
    float ml = max(abs(p.x), max(abs(p.y), abs(p.z)));
    vec3 ansphp = abs(p/ml); 
    
    // select the plane offset of a unit cube
    vec3 upo = sign(p) * step(vec3(1. - SMALL_FLOAT), ansphp);
    
    // scale the plane we are intersecting by the offset
    vec2 pr = intersect_plane(vec3(0.), p, -upo, scale * upo);
    vec3 pp = pr.y * p;

    // get the uv lookup of the plane intersection.
    vec2 uv = step(1. - SMALL_FLOAT, ansphp.x) * pp.yz;
    uv += step(1. - SMALL_FLOAT, ansphp.y) * pp.xz;
    uv += step(1. - SMALL_FLOAT, ansphp.z) * pp.xy;

#if PROJECTION_TYPE == STATIC_TEXTURE | PROJECTION_TYPE == SLIDING_PROJECTION
    
    // filter texture lookup more when unit cube is closer to
    // unit sphere - cheap hack to compensate for the fact
    // that the texture lookup is more stretched near the
    // corners.
    float f = 1.2 * (1. - length(pp/scale - p));
    
    return texture(iChannel0, .5 * uv + .5, f).rgb;
#elif PROJECTION_TYPE == ANIMATED_NOISE
    
    return hash32(.5 * uv + .5);
    
#endif

}

// Function 295
vec4 textureGood( sampler2D sam, vec2 uv, float lo )
{
    uv = uv*1024.0 - 0.5;
    vec2 iuv = floor(uv);
    vec2 f = fract(uv);
	vec4 rg1 = textureLod( sam, (iuv+ vec2(0.5,0.5))/1024.0, lo );
	vec4 rg2 = textureLod( sam, (iuv+ vec2(1.5,0.5))/1024.0, lo );
	vec4 rg3 = textureLod( sam, (iuv+ vec2(0.5,1.5))/1024.0, lo );
	vec4 rg4 = textureLod( sam, (iuv+ vec2(1.5,1.5))/1024.0, lo );
	return mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
}

// Function 296
float escherTextureX(vec2 p)
{
    vec2 pp = vec2(mod(p.x+0.5, 1.0)-0.5, mod(p.y,2.0));
    
    float d = 1000.0;
    
    for(int i=0; i<19; ++i)
    	if(abs(horizontalDistance(pp, vert[i], vert[i+1])) < abs(d))
        {
            d = horizontalDistance(pp, vert[i], vert[i+1]);
        }
    
    pp = vec2(mod(p.x+0.5, 1.0)-0.5, mod(p.y-1.0,2.0));
    
    for(int i=0; i<19; ++i)
    	if(abs(horizontalDistance(pp, vert[i], vert[i+1])) < abs(d))
        {
            d = horizontalDistance(pp, vert[i], vert[i+1]);
        }
    
    float val = smoothstep(0.0, 1.0, d/0.05);
    val = d;
    
    
    if(mod(p.x-0.5, 2.0)-1.0 > 0.)
        val = -val;
    
    return val;
}

// Function 297
vec4 getTexture(float id, vec2 c) {
    vec2 gridPos = vec2(mod(id, 16.), floor(id / 16.));
	return textureLod(iChannel2, 16. * (c + gridPos) / iChannelResolution[3].xy, 0.0);
}

// Function 298
vec4 textureCrtEffect(vec2 uv)
{
    vec2 sz = VIDEO_RES;
    vec2 rsz = 1.0/sz;
    vec2 vy= uv * sz.xy;  
    vec2 ed = (uv-0.5)*2.0;
    float edg = dot(ed,ed);
       
#if WOBBLE
    float s = fract(uv.y - iTime*0.25);
    float q = WOBBLINESS * 0.004 * (0.05 * sin(s) + 0.2 * sin(vy.y*0.1+iTime*5.0) + 0.1*sin(vy.y*0.2+iTime*301.0));

    uv = fract(uv+vec2(q,0));    
#endif

    float tt;
#if BORDER
    tt = clamp(cos((uv.x-0.5)*PI*0.99)*22.0/BORDERNESS,0.0,1.0) * 
         clamp(cos((uv.y-0.5)*PI*0.99)*22.0/BORDERNESS,0.0,1.0);
#else
    tt = 1.0;
#endif
    
    vec4 tex = textureLowRes(uv, sz) * tt;
    tex = vec4((tex.r * 0.2989 + tex.g * 0.5870 + tex.b * 0.1140)); /* to grayscale */

#if COLOR == 2
    vec2 abr = edg * vec2(-1.0,-1.0) * rsz * 0.55 * ABERRATION;
    vec2 abg = edg * vec2( 1.0, 1.0) * rsz * 0.55 * ABERRATION;
    vec2 abb = edg * vec2( 1.0,-1.0) * rsz * 0.55 * ABERRATION;

    vec4 texr = textureLowRes(uv+abr, sz) * tt;
    vec4 texg = textureLowRes(uv+abg, sz) * tt;
    vec4 texb = textureLowRes(uv+abb, sz) * tt;
    
    tex = mix(tex, vec4(texr.r, texg.g, texb.b, 1), SATURATION);
#elif COLOR == 1
    vec4 texr = textureLowRes(uv, sz) * tt;
    vec4 texg = textureLowRes(uv, sz) * tt;
    vec4 texb = textureLowRes(uv, sz) * tt;
    
    tex = mix(tex, vec4(texr.r, texg.g, texb.b, 1), SATURATION);
#endif

    vec4 zero = vec4(0.0);
    vec4 tx = tex;
    
#if NOISE
    tx = mix(tx,temporalNoise(uv, sz), 0.1*NOISYNESS);
#endif
    
#if SCANLINES==2
    float t = sin(iTime*PI*VIDEO_RATE) > 0.0 ? 0.5 : 0.0;
    float yy = 0.5+0.5*sin(TWO_PI*(vy.y + t));
    tx = lerp(tx, zero, yy*0.5);
#elif SCANLINES==1
    float yy = 0.5+0.5*sin(TWO_PI*vy.y);
    tx = lerp(tx, zero, yy*0.5);
#endif
    
#if VSCAN
    tx *= (0.9 + 0.25 * s);
#endif

#if RGBGRID
    float fr = sin(vy.x * TWO_PI) * sin(vy.y * TWO_PI);
    float fg = sin((vy.x+0.5) * TWO_PI) * sin((vy.y-0.25) * TWO_PI);
    float fb = sin((vy.x+0.25) * TWO_PI) * sin((vy.y+0.25) * TWO_PI);
    
    tx.r = tx.r * (0.9 + 0.1 * fr);
    tx.g = tx.g * (0.9 + 0.1 * fg);
    tx.b = tx.b * (0.9 + 0.1 * fb);
#endif

    return tx;
}

// Function 299
vec3 sTexture(sampler2D smp, vec2 uv) {
 
    vec2 textureResolution = iChannelResolution[1].yy;
	uv = uv*textureResolution + 0.5;
	vec2 iuv = floor( uv );
	uv -= iuv;
	uv = iuv + smoothstep(0., 1., uv); 
    //uv = iuv +  uv*uv*uv*(uv*(uv*6. - 15.) + 10.);
	uv = (uv - .5)/textureResolution;
    return texture(smp, uv).xyz;
    
}

// Function 300
int GetGlyph(int iterations, ivec2 glyphPos, int glyphLast, ivec2 glyphPosLast, ivec2 focusPos)
{ 
    if (glyphPos == focusPos)
        return GetFocusGlyph(iterations); // inject correct glyph     
            
    int seed = iterations + glyphPos.x * 313 + glyphPos.y * 411 + glyphPosLast.x * 557 + glyphPosLast.y * 121;
    return RandInt(seed) % glyphCount; 
}

// Function 301
float text_floor2(vec2 U) {
    initMsg;
    U.x+=4.*(0.5-0.2812*(res.x/0.5));
    C(70);C(108);C(111);C(111);C(114);C(49);
    endMsg;
}

// Function 302
vec4 glyph(vec2 coord, vec2 center, vec2 size, int char) {
    vec2 vc = coord-center;
    vc /= size;
    vc += 0.5;
    vc = 1.0-vc;
    float c = float(char);
    vec2 ch = vec2(mod(c,16.0),floor(c/16.0))/16.0;
    ch = ch;
    vec2 vch = (vc/16.0+ch);
    vec4 g = vec4(0.0);
    if (vc.x > 0.0 && vc.y > 0.0 && vc.x < 1.0 && vc.y < 1.0) {
        g = texture(iChannel2,(vch)).gbar;
    }
    return g;
}

// Function 303
float text_playex2(vec2 U) {
    initMsg;C(70);C(79);C(82);C(32);C(76);C(97);C(117);C(110);C(99);C(104);endMsg;
}

// Function 304
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

// Function 305
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

// Function 306
float filterFlowToTexture(float f, vec2 uv) {
    f += smoothstep(.3, .5, f) * .8 * (1. - f);
    f = smoothstep(0., 1., f);
    f = filterEdgesOfLava(f, uv);
    return f;
}

// Function 307
vec4 baseTexture(in vec2 uv, float depth){    
    float size = 1.;    
    float blur = min(.0005 * (depth * 10.5), .0030);    
    return vec4(.5) * smoothstep(0., blur, smoothmod(uv.x * size, .05, blur * 5.)) * smoothstep(0., blur * RATIO, smoothmod(uv.y * size, .05, blur * RATIO * 3.)) * smoothstep(1., 1. - blur, mod(uv.x * size + .05 * floor(mod(uv.y * size, .1) * 20.), .1) * 20.) + texture(iChannel0, uv * 5.) - vec4(.14, .15, .01, 0.);	
}

// Function 308
float get_secondary_texture_index(in sampler2D s)
{
    return texelFetch(s, CTRL_TEXTURE, 0).z;
}

// Function 309
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

// Function 310
vec4 floorTexture(vec3 p, vec3 q1) {
    vec3 col = vec3(0.);

    float fogMultiplier = 0.;

    vec2 uv = mod((p.xy - HALF_CELL_SIZE), CELL_SIZE) / CELL_SIZE - .5;
    vec2 roadUV = mod((p.xy), CELL_SIZE) / CELL_SIZE;
    vec2 blockID = floor(p.xy / CELL_SIZE);

    if (abs(blockID.x) > bounds.x || abs(blockID.y) > bounds.y) {
        return vec4(COLOR_BUILDING_BASE, fogMultiplier);
    }

    float roadX = step(BULDING_BASE_SIZE, roadUV.x) - step(1. - BULDING_BASE_SIZE, roadUV.x);
    float roadY = step(BULDING_BASE_SIZE, roadUV.y) - step(1. - BULDING_BASE_SIZE, roadUV.y);

    float road = max(roadX, roadY);


    col += road * COLOR_ROAD;

    // col.rg = roadUV;
    // col.rg = uv;

    vec2 baseUV = abs(uv);
    col += step(max(baseUV.x, baseUV.y), BULDING_BASE_SIZE*.9) * COLOR_BUILDING_BASE;

    uv = fract((uv + .26)*8.);

    if (roadX == 0.) {
        uv.x /= 4.;
    }
    if (roadY == 0.) {
        uv.y /= 4.;
    }

    float delimeter = step(max(uv.x, uv.y), .1) * road;

    fogMultiplier = delimeter * 1.5;

    col += delimeter * COLOR_ROAD_DELIMETER;

    if (col.x == 0.) {
        col = COLOR_ROAD_SIDEWALK;
    }

    vec2 zebraUV = roadUV * 12.;
    vec2 zebraID = floor(zebraUV);
    zebraUV = fract(zebraUV);

    float n = n21(blockID);

    if (zebraID.x == 3. && road > 0. && n > .7) {
        col += step(fract((zebraUV - .08)*vec2(1., 3.)).y + .1, .4);
    }
    if (zebraID.y == 3. && road > 0. && fract(n*123.33) > .7) {
        col += step(fract((zebraUV - .08)*vec2(3., 1.)).x + .1, .4);
    }

    return vec4(col, fogMultiplier);
}

// Function 311
vec3 moonTexture(vec2 uv) {
    float d = length(fract(uv) - .5);
    //return exp(-40. * d * d) * vec3(1.);
    return texture(iChannel0, uv / 16.).rgb;
}

// Function 312
vec4 TexTallTechnoPillar( vec2 vTexCoord, float fRandom, float fHRandom )
{
    vec3 vRayOrigin = vec3(0.0, 64.0, -300.0);
    vec3 vRayTarget = vec3( vTexCoord.x - 38. * .5, vTexCoord.y, 0.0);
    vec3 vRayDir = normalize( vRayTarget - vRayOrigin );
    
    vec3 vNormal;
    
    float t = TraceCylinder( vRayOrigin, vRayDir, vec3(0, 3, 0), vec3(0,1,0), 127.0 - 6., 12.0, vNormal );

    vec3 vNormal2;
    float t2; 
    t2 = TraceCylinder( vRayOrigin, vRayDir, vec3(0, 3, 0), vec3(0,1,0), 6.0, 16.0, vNormal2 );
    if ( t2 < t )
    {
        t = t2;
        vNormal = vNormal2;
    }

    t2 = TraceCylinder( vRayOrigin, vRayDir, vec3(0, 127. - 3. - 6., 0), vec3(0,1,0), 6.0, 16.0, vNormal2 );
    if ( t2 < t )
    {
        t = t2;
        vNormal = vNormal2;
    }


    t2 = TraceCylinder( vRayOrigin, vRayDir, vec3(0, 32, 0), vec3(0,1,0), 2.0, 16.0, vNormal2 );
    if ( t2 < t )
    {
        t = t2;
        vNormal = vNormal2;
    }
    
    t2 = TraceCylinder( vRayOrigin, vRayDir, vec3(0, 36, 0), vec3(0,1,0), 2.0, 16.0, vNormal2 );
    if ( t2 < t )
    {
        t = t2;
        vNormal = vNormal2;
    }
    t2 = TraceCylinder( vRayOrigin, vRayDir, vec3(0, 40, 0), vec3(0,1,0), 2.0, 16.0, vNormal2 );
    if ( t2 < t )
    {
        t = t2;
        vNormal = vNormal2;
    }
    
    
    //vec3 vNormal1;
    if ( t > 5000. )
    {
        return vec4(0.);
    }
    
    vec3 vLight = normalize( vec3(-1., -0.5, -2 ) );
    
    float fShade = max(0.0, dot( vNormal, vLight ));
    
    vec3 vCol = vec3(0.2) + fRandom * 0.1;
    
    vec3 vPos = vRayOrigin + vRayDir * t;
    
    if ( vPos.y > 43. && vPos.y < 118. )
    {
        float f = fRandom / .75;
        //f *= 0.75 + fHRandom * 0.25;
        vCol = vec3( pow( f, 5.0) );
    }
    
    vCol *= fShade;
    return vec4(vCol,1);
    
    // float fShade = fRandom - fHRandom * 0.5;
    //return vec4(fShade,fShade, fShade, 1);
}

// Function 313
UIDrawContext UIDrawContext_SetupFromRect( Rect rect )
{
    UIDrawContext drawContext;
    drawContext.viewport = rect;
    drawContext.vOffset = vec2(0);
    drawContext.vCanvasSize = rect.vSize;
	return drawContext;
}

// Function 314
float textureFunc(vec2 U ) {
    const float iter = 2.;
    
    vec2 P = vec2(.5);
    vec2 I = vec2(1,0);
    vec2 J = vec2(0,1);
    
    vec2 l = -I;
    vec2 r;
    
    vec2 qU;
	vec2 tmp;
    
    for (float i = 0.; i < iter; i++) {
        qU      = step(.5,U);         // select quadrant
        bvec2 q = bvec2(qU);          // convert to boolean
        
        U       = 2.*U - qU;          // go to new quadrant
        
        l = q.x ? (q.y ? -J : -I)            // node left segment
                : (q.y ?  l :  J);
                    

        r = (q.x==q.y)?  I : (q.y ?-J:J);    // node right segment
        
        // the heart of Hilbert curve : 
        if (q.x) { // sym
        	symU(U);
            symV(l);
            symV(r);
            swap(l,r);
       	}
        if (q.y) { // rot+sym
            rotU(U); symU(U);
            rotV(l); symV(l);
            rotV(r); symV(r);
       	}
    }
    
    float s=iter* 25.;
    float o=length(l+r) > 0. ? plotC (U-P, l+r) : plot (U-P, l) + plot (U-P, r); 
    return pow(sin(smoothstep(.33+.01*s,.33-.01*s,o)*0.5*PI),2.);
}

// Function 315
void UILayout_SetControlRectFromText( inout UILayout uiLayout, PrintState state, LayoutStyle style )
{
    UILayout_SetControlRect( uiLayout, UI_GetFontRect( state, style ) );
}

// Function 316
float textureSpiral(vec2 uv) {
	float angle = ATAN(uv.y, uv.x),
	shear = length(uv),
	blur = 0.5;
	return smoothstep(-blur, blur, cos_(8.0 * angle + 200.0 * time - 12.0 * shear));
}

// Function 317
float Glyph(int char, vec2 uv)
{
    if(any(lessThan(vec4(uv,1,1), vec4(0,0,uv))))
    {
        return 0.0;
    }
    
    float g = texture(iChannel0, 0.0625*(uv + vec2(char - char/16*16,15 - char/16))).w;
    return smoothstep(0.51, 0.51 - 20.0/(1.0*iResolution.y) , g);
}

// Function 318
void glyph_Minus()
{
  MoveTo(x*0.5+y);
  RLineTo(x);
}

// Function 319
vec4 generate_texture(const int material, vec2 uv)
{
    vec2 tile_size = get_tile(material).zw;

	vec3 clr;
    float shaded = 1.;	// 0 = fullbright; 0.5 = lit; 1.0 = lit+AO
    
    float grain = random(uv*128.);
    
    // gathering FBM parameters first and calling the function once
    // instead of per material reduces the compilation time for this buffer
    // by about 4 seconds (~9.4 seconds vs ~13.4) on my machine...
    
    // array-based version compiles about 0.7 seconds faster
    // than the equivalent switch (~14.1 seconds vs ~14.8)...

    const vec4 MATERIAL_SETTINGS[]=vec4[7](vec4(3,5,1,3),vec4(3,5,1,4),vec4(6,6,.5,3),vec4(10,10,.5,2),vec4(3,5,1,2),vec4(7,3,.5
    ,2),vec4(7,5,.5,2));
    const int MATERIAL_INDICES[]=int[NUM_MATERIALS+1](1,1,1,0,1,0,1,0,0,0,1,6,6,6,1,1,2,3,4,5,0);

    vec4 settings = MATERIAL_SETTINGS[MATERIAL_INDICES[min(uint(material), uint(NUM_MATERIALS))]];
    vec2 base_grid = settings.xy;
    float base_gain = settings.z;
    float base_lacunarity = settings.w;

    if (is_material_sky(material))
        uv += sin(uv.yx * (3.*PI)) * (4./128.);

    vec2 aspect = tile_size / min(tile_size.x, tile_size.y);
    float base = tileable_turb(uv * aspect, base_grid, base_gain, base_lacunarity);
    
    // this switch compiles ~2.2 seconds faster on my machine
    // than an equivalent if/else if chain (~11.5s vs ~13.7s)
    
	#define GENERATE(mat) ((GENERATE_TEXTURES) & (1<<(mat)))

    switch (material)
    {
#if GENERATE(MATERIAL_WIZMET1_2) || GENERATE(MATERIAL_QUAKE)
        case MATERIAL_WIZMET1_2:
        case MATERIAL_QUAKE:
        {
            uv.x *= tile_size.x/tile_size.y;
            uv += vec2(.125, .0625);
            base = mix(base, grain, .2);
            clr = mix(vec3(.16, .13, .06), vec3(.30, .23, .12), sqr(base));
            clr = mix(clr, vec3(.30, .23, .13), sqr(linear_step(.5, .9, base)));
            clr = mix(clr, vec3(.10, .10, .15), smoothen(linear_step(.7, .1, base)));
            if (material == MATERIAL_WIZMET1_2 || (material == MATERIAL_QUAKE && uv.y < .375))
            {
                vec2 knob_pos = floor(uv*4.+.5)*.25;
                vec2 knob = add_knob(uv, 1./64., knob_pos, 3./64., vec2(-.4, .4));
                clr = mix(clr, vec3(.22, .22, .28)*mix(1., knob.x, .8), knob.y);
                knob = add_knob(uv, 1./64., knob_pos, 1.5/64., vec2(.4, -.4));
                clr = mix(clr, .7*vec3(.22, .22, .28)*mix(1., knob.x, .7), knob.y);
            }
            if (material == MATERIAL_QUAKE)
            {
                uv -= vec2(1.375, .15625);
                uv.x = mod(uv.x, 5.);
                uv.y = fract(uv.y);
                vec2 engraved = engraved_QUAKE(uv, 5./64., vec2(0, -1));
                clr *= mix(1., mix(1., engraved.x*1.25, .875), engraved.y);
            }
        }
        break;
#endif

#if GENERATE(MATERIAL_WIZMET1_1)
        case MATERIAL_WIZMET1_1:
        {
            base = mix(base, grain, .4);
            float scratches = linear_step(.15, .9, smooth_noise(vec2(32,8)*rotate(uv, 22.5).x) * base);
            clr = vec3(.17, .17, .16) * mix(.5, 1.5, base);
            clr = mix(clr, vec3(.23, .19, .15), scratches);
            scratches *= linear_step(.6, .25, smooth_noise(vec2(16,4)*rotate(uv, -45.).x) * base);
            clr = mix(clr, vec3(.21, .21, .28) * 1.5, scratches);
            float bevel = .6 *mix(3.5/64., 11./64., base);
            float d = min(1., min(uv.x, 1.-uv.y) / bevel);
            float d2 = min(d, 3. * min(uv.y, 1.-uv.x) / bevel);
            clr *= 1. - (1. - d2) * mix(.3, .8, base);
            clr = mix(clr, vec3(.39, .39, .57) * base, around(.6, .4, d));
        }
        break;
#endif

#if GENERATE(MATERIAL_WIZ1_4)
        case MATERIAL_WIZ1_4:
        {
            base = mix(smoothen(base), grain, .3);
            clr = mix(vec3(.37, .28, .21), vec3(.52, .41, .33), smoothen(base));
            clr = mix(clr, vec3(.46, .33, .15), around(.45, .05, base));
            clr = mix(clr, vec3(.59, .48, .39), around(.75, .09, base)*.75);
            float bevel = mix(4./64., 12./64., FAST32_smooth_noise(uv, vec2(21)));
            vec2 mins = vec2(bevel, bevel * 2.);
            vec2 maxs = 1. - vec2(bevel, bevel * 2.);
            uv = running_bond(uv, 1., 2.) * vec2(1, 2);
            vec2 duv = (fract(uv) - clamp(fract(uv), mins, maxs)) * (1./bevel) * vec2(2, 1);
            float d = mix(length(duv), max_component(abs(duv)), .75);
            clr *= clamp(2.1 - d*mix(.75, 1., sqr(base)), 0., 1.);
            clr *= 1. + mix(.25, .5, base) * max(0., dot(duv, INV_SQRT2*vec2(-1,1)) * step(d, 1.2));
        }
		break;
#endif

#if GENERATE(MATERIAL_WBRICK1_5)
        case MATERIAL_WBRICK1_5:
        {
            vec2 uv2 = uv + sin(uv.yx * (3.*PI)) * (4./64.);
            uv = running_bond(uv + vec2(.5, 0), 1., 2.) * vec2(1, 2);
            base = mix(smoothen(base), grain, .3);
            float detail = tileable_smooth_noise(uv2, vec2(11));
            detail = sqr(around(.625, .25, detail)) * linear_step(.5, .17, base);
            clr = mix(vec3(.21, .17, .06)*.75, vec3(.30, .26, .15), base);
            clr *= mix(.95, 2., sqr(sqr(base)));
            clr = mix(clr, vec3(.41, .32, .14), detail);
            float bevel = mix(4./64., 8./64., base);
            vec2 mins = vec2(bevel, bevel * 1.75);
            vec2 maxs = 1. - vec2(bevel, bevel * 2.);
            vec2 duv = (fract(uv) - clamp(fract(uv), mins, maxs)) * (1./bevel) * vec2(2, 1);
            float d = length(duv);
            if (uv.y > 1. || uv.y < .5)
                d = mix(d, max_component(abs(duv)), .5);
            //clr *= mix(1., mix(.25, .625, base), linear_step(1., 2., d)*step(1.5, d));
            clr *= clamp(2.1 - d*mix(.75, 1., sqr(base)), 0., 1.);
            clr *= 1. + mix(.25, .5, base) * max(0., dot(duv, INV_SQRT2*vec2(-1,1)) * step(d, 1.2));
        }
        break;
#endif

#if GENERATE(MATERIAL_CITY4_7)
        case MATERIAL_CITY4_7:
        {
            base = mix(base, grain, .4);
            vec3 brick = herringbone(uv);
            uv = brick.xy;
            clr = mix(vec3(.23, .14, .07), vec3(.29, .16, .08), brick.z) * mix(.3, 1.7, base);
            clr = mix(clr, vec3(.24, .18, .10), linear_step(.6, .9, base));
            clr = mix(clr, vec3(.47, .23, .12), linear_step(.9, 1., sqr(grain))*.6);
            clr *= (1. + add_bevel(uv, 2., 4., mix(1.5/64., 2.5/64., base), -mix(.05, .15, grain), 0.6));
        }
        break;
#endif

#if GENERATE(MATERIAL_CITY4_6)
		case MATERIAL_CITY4_6:
        {
            base = mix(base, grain, .5);
            vec3 brick = herringbone(uv);
            uv = brick.xy;
            clr = mix(vec3(.09, .08, .01)*1.25, 2.*vec3(.21, .15, .08), sqr(base));
            clr *= mix(.85, 1., brick.z);
            clr = mix(clr, mix(.25, 1.5, sqr(base))*vec3(.11, .11, .22), around(.8, mix(.24, .17, brick.z), (grain)));
            clr = mix(clr, mix(.75, 1.5, base)*vec3(.26, .20, .10), .75*sqr(around(.8, .2, (base))));
            clr *= (1. + add_bevel(uv, 2., 4., 2.1/64., .0, .25));
            clr *= (1. + add_bevel(uv, 2., 4., mix(1.5/64., 2.5/64., base), mix(.25, .05, grain), .35));
        }
        break;
#endif

#if GENERATE(MATERIAL_DEM4_1)
        case MATERIAL_DEM4_1:
        {
            base = mix(base, grain, .2);
            clr = mix(vec3(.18, .19, .21), vec3(.19, .15, .06), linear_step(.4, .7, base));
            shaded = .75; // lit, half-strength AO
        }
        break;
#endif

#if GENERATE(MATERIAL_COP3_4)
        case MATERIAL_COP3_4:
        {
            float sdf = sdf_chickenscratch(uv, vec2(.25, .125), vec2(.75, .375), 1.5/64.);
            base = mix(base, grain, .2);
            base *= mix(1., .625, sdf_mask(sdf, 1./64.));
            clr = mix(vec3(.14, .15, .13), vec3(.41, .21, .12), base);
            clr = mix(clr, vec3(.30, .32, .34), linear_step(.6, 1., base));
            float bevel = mix(2./64., 6./64., sqr(FAST32_smooth_noise(uv, vec2(13))));
            clr *= (1. + add_bevel(uv, 1., 1., bevel, .5, .5));
            clr *= 1.5;
        }
        break;
#endif

#if GENERATE(MATERIAL_BRICKA2_2) || GENERATE(MATERIAL_WINDOW02_1)
        case MATERIAL_BRICKA2_2:
        case MATERIAL_WINDOW02_1:
        {
            vec2 grid = (material == MATERIAL_BRICKA2_2) ? vec2(6., 5.) : vec2(8., 24.);
            uv = (material == MATERIAL_WINDOW02_1) ? fract(uv + vec2(.5, .5/3.)) : uv;
            vec3 c = voronoi(uv, grid);
            if (material == MATERIAL_BRICKA2_2)
            {
                float dark_edge = linear_step(.0, mix(.05, .45, base), c.z);
                float lit_edge = linear_step(.35, .25, c.z);
                float lighting = -normalize(c.xy).y * .5;
                clr = vec3(.25, .18, .10) * mix(.8, 1.2, grain);
                clr *= (1. + lit_edge * lighting) * mix(.35, 1., dark_edge);
                uv = fract(running_bond(uv, 1., 2.) * vec2(1, 2));
                clr *=
                    mix(1., min(1., 4.*min(uv.y, 1.-uv.y)), .5) *
                	mix(1., min(1., 8.*min(uv.x, 1.-uv.x)), .3125);
                clr *= 1.25;
            }
            else
            {
                // Note: using x*48 instead of x/fwidth(x) reduces compilation time
                // for this buffer by ~23% (~10.3 seconds vs ~13.4) on my system
                float intensity = mix(1.25, .75, hash1(uv*grid + c.xy)) * (1. - .5*length(c.xy));
                uv.y *= 3.;
                float flame = sdf_window_flame(uv) * 48.;
                float emblem = sdf_window_emblem(uv) * 48.;
                float edge = linear_step(.0, .15, c.z);
                clr = mix(vec3(1., .94, .22) * 1.125, vec3(.63, .30, .19), clamp(flame, 0., 1.));
                clr = mix(clr, vec3(.55, .0, .0), clamp(1.-emblem, 0., 1.));
                clr = mix(vec3(dot(clr, vec3(1./3.))), clr, intensity);
                edge *= clamp(abs(flame), 0., 1.) * clamp(abs(emblem), 0., 1.);
                edge *= step(max(abs(uv.x - .5) - .5, abs(uv.y - 1.5) - 1.5), -2./64.);
                clr *= intensity * edge;
                shaded = .75; // lit, half-strength AO
            }
    	}
        break;
#endif

#if GENERATE(MATERIAL_LAVA1) || GENERATE(MATERIAL_WATER2) || GENERATE(MATERIAL_WATER1)
        case MATERIAL_LAVA1:
        case MATERIAL_WATER2:
        case MATERIAL_WATER1:
        {
            vec2 grid = (material == MATERIAL_WATER1) ? vec2(5., 7.) : vec2(5., 5.);
            uv += base * (1./31.) * sin(PI * 7. * uv.yx);
            float cellular = Cellular2D(uv, grid);
            float grain_amount = (material == MATERIAL_LAVA1) ? .125 : .25;
            float high_point = (material == MATERIAL_WATER2) ? .8 : .9;
            base = mix(base, grain, grain_amount);
            cellular = sqrt(cellular) + mix(-.3, .3, base);
            base = linear_step(.1, high_point, cellular);
            if (material == MATERIAL_LAVA1)
            {
                clr = mix(vec3(.24,.0,.0), vec3(1.,.40,.14), base);
                clr = mix(clr, vec3(1.,.55,.23), linear_step(.5, 1., base));
            }
            else if (material == MATERIAL_WATER2)
            {
                clr = mix(vec3(.10,.10,.14)*.8, vec3(.17,.17,.24)*.8, base);
                clr = mix(clr, vec3(.16,.13,.06)*mix(.8, 2.5, sqr(sqr(base))), around(.5, .1, grain));
                clr = mix(clr, vec3(.20,.20,.29)*.8, linear_step(.5, 1., base));
            }
            else // if (material == MATERIAL_WATER1)
            {
                clr = mix(vec3(.08,.06,.04), vec3(.30,.23,.13), base);
                clr = mix(clr, vec3(.36,.28,.21), linear_step(.5, 1., base));
            }
            shaded = 0.;
        }
        break;
#endif

#if GENERATE(MATERIAL_WIZWOOD1_5)
		case MATERIAL_WIZWOOD1_5:
        {
            const vec2 GRID = vec2(1, 4);
            uv = running_bond(fract(uv.yx), GRID.x, GRID.y);
            vec2 waviness = vec2(sin(3. * TAU * uv.y), 0) * .0;
            waviness.x += smooth_noise(uv.y * 16.) * (14./64.);
            base = tileable_turb(uv + waviness, vec2(2, 32), .5, 3.);
            clr = mix(vec3(.19, .10, .04)*1.25, vec3(.64, .26, .17), around(.5, .4, smoothen(base)));
            clr = mix(clr, vec3(.32, .17, .08), around(.7, .3, base)*.7);
            
            float across = fract(uv.y * GRID.y);
            clr *= 1. + .35 * linear_step(1.-4./16., 1.-2./16., across) * step(across, 1.-2./16.);
            across = min(across, 1. - across);
            clr *= mix(1., linear_step(0., 2./16., across), mix(.25, .75, base));
			float along = fract(uv.x * GRID.x);
            clr *= 1. + .25 * linear_step(2./64., 0., along);
            clr *= mix(1., linear_step(1., 1.-2.5/64., along), mix(.5, .75, base));
            
            const vec2 LIGHT_DIR = INV_SQRT2 * vec2(-1, 1);
            uv = fract(uv * GRID);
            vec2 side = sign(.5 - uv); // keep track of side before folding to 'unmirror' light direction
            uv = min(uv, 1. - uv) * (1./GRID);
            vec2 nail = add_knob(uv, 1./64., vec2(4./64.), 1./64., side * LIGHT_DIR);
            clr = mix(clr, vec3(.64, .26, .17) * nail.x, nail.y * .75);

            clr *= .9 + grain*.2;
            clr *= .75;
        }
        break;
#endif

#if GENERATE(MATERIAL_TELEPORT)
        case MATERIAL_TELEPORT:
        {
            uv *= 64./3.;
            vec2 cell = floor(uv);
            vec4 n = hash4(cell);
            uv -= cell;
            float radius = mix(.15, .5, sqr(sqr(n.z)));
            n.xy = mix(vec2(radius), vec2(1.-radius), smoothen(n.xy));
            uv = clamp((n.xy - uv) * (1./radius), -1., 1.);
            clr = (1.-length_squared(uv)) * (1.-sqr(sqr(n.w))) * vec3(.44, .36, .26);
            shaded = 0.;
        }
        break;
#endif

#if GENERATE(MATERIAL_FLAME)
        case MATERIAL_FLAME:
        {
            base = mix(base, grain, .1);
            clr = mix(vec3(.34, .0, .0), vec3(1., 1., .66), smoothen(base));
            clr = clamp(clr * 1.75, 0., 1.);
            shaded = 0.;
        }
        break;
#endif

#if GENERATE(MATERIAL_ZOMBIE)
        case MATERIAL_ZOMBIE:
        {
            base = mix(base, grain, .2);
            clr = vec3(.57, .35, .24) * mix(.6, 1., sqr(base));
            clr = mix(clr, vec3(.17, .08, .04), linear_step(.3, .7, base));
        }
        break;
#endif

#if GENERATE(MATERIAL_SKY1)
        case MATERIAL_SKY1:
        {
            clr = vec3(.18, .10, .12) * 1.5 * smoothen(base);
            shaded = 0.;
        }
        break;
#endif

#if GENERATE(MATERIAL_SKY1B)
        case MATERIAL_SKY1B:
        {
            clr = vec3(.36, .19, .23) * 1.125 * smoothen(base);
            shaded = 0.;
        }
        break;
#endif

        default:
        {
            clr = vec3(base * .75);
        }
        break;
    }
    
	#undef GENERATE
    
    clr = clamp(clr, 0., 1.);

    return vec4(clr, shaded);
}

// Function 320
int getTextChAt(int at){
    ivec4 vt = ivec4(0);  
    int s1 = 0;
    int s2 = 0;

    for (int i = 0; i < TEXT_LEN; i++)   {
        if (( at / 16) == i) {
            vt = (TEXT[i]);
            break;
        }
    }

    int b = modi(at/4,4);
    for (int i = 0; i < 4; i++) {
        if ( i == b) {
            s1 = vt[i]; 
            break;
        } 
    }
    return modi(shiftr(s1, 24-modi(at, 4)*8),256);

}

// Function 321
void glyph_6()
{
  MoveTo(x*0.2+y*0.6);
  Bez3To(x*0.2-0.2*y,1.8*x-y*0.2,float2(1.8,0.6));
  Bez3To(1.8*x+y*1.4,x*0.2+1.4*y,x*0.2+y*0.6);
  Bez3To(x*0.0+1.6*y,0.7*x+y*2.3,float2(1.6,1.9));
}

// Function 322
float escherTextureY(vec2 p)
{
    vec2 pp = vec2(mod(p.x, 2.0), mod(p.y-.5, 1.0)+0.5);
    
    float d = 1000.0;
    for(int i=19; i<vert.length()-1; ++i)
    	if(abs(verticalDistance(pp, vert[i], vert[i+1])) < abs(d))
            d=verticalDistance(pp, vert[i], vert[i+1]);
    

    pp = vec2(mod(p.x-1.0, 2.0), mod(p.y-.5, 1.0)+0.5);
    for(int i=19; i<vert.length()-1; ++i)
    	if(abs(verticalDistance(pp, vert[i], vert[i+1])) < abs(d)) 
            d=verticalDistance(pp, vert[i], vert[i+1]);
    
    float val = smoothstep(0.0, 1.0, d/0.05);
    val = d;
    
    if(mod(p.y-0.5, 2.0)-1.0 > 0.)
        val = -val;
    
    return val;
}

// Function 323
float text(vec2 fragCoord)
{
    vec2 uv = mod(fragCoord.xy, 16.)*.0625;
    vec2 block = fragCoord*.0625 - uv;
    uv = uv*.8+.1; // scale the letters up a bit
    uv += floor(texture(iChannel1, block/iChannelResolution[1].xy + iTime*.002).xy * 16.); // randomize letters
    uv *= .0625; // bring back into 0-1 range
    uv.x = -uv.x; // flip letters horizontally
    return texture(iChannel0, uv).r;
}

// Function 324
float text_playupd(vec2 U) {
    initMsg;C(85);C(112);C(100);C(97);C(116);C(101);C(100);endMsg;
}

// Function 325
float text_playex(vec2 U) {
    initMsg;C(82);C(69);C(65);C(68);C(32);C(67);C(111);C(109);C(109);C(111);C(110);endMsg;
}

// Function 326
vec2 pointToTexture(vec3 pt, vec4 tube)
{
    //Get x and y of vector from center of the tube to our point
    float newz = pt.z - tube.y;
    float newx = pt.x - tube.x;
    
    //Calculate the angle in radians between the above vector and an arbitrary
    //starting point; the proportion of this angle to 2PI is the u component
    //and the y-coordinate over the tube's diameter is out v component
	float xfrac =  (atan( tube.w*newz , tube.w*newx)+PI*0.5)/(3.0*PI);
    float yfrac = pt.y/(3.5*PI*tube.z);
    
    return vec2(xfrac,yfrac);
}

// Function 327
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

// Function 328
vec4 textureGrad(sampler2D s,vec2 c,vec2 dx,vec2 dy){return texture2DGradEXT(s,c,dx,dy);}

// Function 329
float textTexture(vec2 uv, int defText)
{
   uv/= textScale;
   int idx = int(uv.x*16./charSpacingFac.x)+1000*int(uv.y*16./charSpacingFac.y); 
    
   int char = 32;
   int chi = 0;
    
   if (defText==OBJ_TAMBAKO) { _Tambako }
   if (defText==OBJ_THE_JAGUAR) { _theJaguar }
   if (defText==OBJ_PRESENTS) { _presents }
   if (defText==OBJ_A_COOL_SHADER) { _acoolshader }
   if (defText==OBJ_POWERED_BY) { _poweredby }
   if (defText==OBJ_SHADERTOY) { _shadertoy }
    
   return char==32?0.9:getChar(uv, char);
}

// Function 330
vec4 texture_Bicubic( sampler2D tex, vec2 t )
{
    vec2 res = iChannelResolution[0].xy;
    vec2 p = res*t - 0.5;
    vec2 f = fract(p);
    vec2 i = floor(p);

    return spline( f.y, spline( f.x, SAM(-1,-1), SAM( 0,-1), SAM( 1,-1), SAM( 2,-1)),
                        spline( f.x, SAM(-1, 0), SAM( 0, 0), SAM( 1, 0), SAM( 2, 0)),
                        spline( f.x, SAM(-1, 1), SAM( 0, 1), SAM( 1, 1), SAM( 2, 1)),
                        spline( f.x, SAM(-1, 2), SAM( 0, 2), SAM( 1, 2), SAM( 2, 2)));
}

// Function 331
vec4 textured(vec2 pos){
    
    int textureScale = 1;
	ivec2 P = ivec2(pos) / textureScale % 1024;
    return texelFetch(iChannel0, P, 0);
}

// Function 332
vec4 getTexture(Material mat, vec3 p)
{

  if(mat==landMaterial)
  {
    // Adds some green areas. 
    if(noiseValue*10.0-float(int(noiseValue2*10.0))>0.55)
    {
      mat=grassMaterial;
      noiseValue=10.0*noiseValue2*noiseValue+3.0*cnoise(noiseValue*(p-vec3(0.1,0.1,0.2))*40.0)+0.6*cnoise((p+vec3(0.2,0.1,-0.3))*200.0);
    } 
    else
      noiseValue=noiseValue2*2.0+noiseValue*4.0+2.0*cnoise((p+vec3(0.6,0.3,0.4))*noiseValue*40.0);
  }

  else if(mat==waterMaterial)
      noiseValue*=10.0;


  noiseValue=clamp(0.0,1.0,noiseValue);
  return noiseValue*mat.color+(1.0-noiseValue)*mat.bgColor;
}

// Function 333
float text_mi(vec2 U) {
    initMsg;C(45);endMsg;
}

// Function 334
float texture_Bicubic(sampler2D tex, vec2 t) {
    vec2 res = iChannelResolution[0].xy;
    vec2 p = res*t - 0.5, f = fract(p), i = floor(p);
    return spline( f.y, spline( f.x, SAM(-1,-1), SAM( 0,-1), SAM( 1,-1), SAM( 2,-1)),
                        spline( f.x, SAM(-1, 0), SAM( 0, 0), SAM( 1, 0), SAM( 2, 0)),
                        spline( f.x, SAM(-1, 1), SAM( 0, 1), SAM( 1, 1), SAM( 2, 1)),
                        spline( f.x, SAM(-1, 2), SAM( 0, 2), SAM( 1, 2), SAM( 2, 2)));
}

// Function 335
vec3 textex(vec3 p)
{
	vec3 ta = texture(iChannel1, vec2(p.y,p.z)).xyz;
    vec3 tb = texture(iChannel1, vec2(p.x,p.z)).xyz;
    vec3 tc = texture(iChannel1, vec2(p.x,p.y)).xyz;
    return (ta + tb + tc) / 3.0;
}

// Function 336
float text_drw(vec2 U) {
    initMsg;C(68);C(114);C(97);C(119);C(32);C(50);endMsg;
}

// Function 337
vec4 drawTextHorizontal(in vec2 uv, in vec2 start, in float lsize, in vec2 _text[10], in int _tsize)
{
    vec4 tcol;
    
    for (int i_letter = 0; i_letter < _tsize; i_letter++)
    {
        tcol += drawLetter(uv, start + float(i_letter)*vec2(lsize*.5, 0.), lsize, _text[i_letter]);
    }
    
    return clamp(tcol, 0., 1.);
}

// Function 338
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

// Function 339
vec3 _texture(vec3 p) {
    vec3 t;
    
    // apply the twist, so we don't use world coordinates
    tTwist(p, _twist);
    bool wall = isWall(p);
    
    // if we're in a column, apply distortion to the fbm space
    t = fbm((p + (wall ? 0. : .1 + .9 * fbm(p * 5.))) * vec3(5., 20., 5.)) * vec3(1., .7, .4) * .75
        + fbm(p * vec3(2., 10., 2.)) * vec3(1., .8, .5) * .25;
    
    // make the walls whiter
    if (wall) t = mix(t, vec3(1), .5);
    return saturate(t);
}

// Function 340
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

// Function 341
vec4 glyph_color(uint glyph, ivec2 pixel)
{
    uint x = glyph & 7u,
         y = glyph >> 3u;
    pixel = ivec2(ADDR2_RANGE_FONT.xy) + (ivec2(x, y) << 3) + (pixel & 7);
    return texelFetch(LIGHTMAP_CHANNEL, pixel, 0);
}

// Function 342
vec4 put_text_drawstart(vec4 col, vec2 uv, vec2 pos, float scale)
{
	float unit = asp * scale * 0.1;
    float h = 0.;
    vec2 sc = vec2(unit, unit*0.8);
    
    // M
    h = max(h, word_map(uv, pos, 77, sc));
    // a
    h = max(h, word_map(uv, pos+vec2(unit*0.4, 0.), 97, sc));
    // r
    h = max(h, word_map(uv, pos+vec2(unit*0.8, 0.), 114, sc));
    // c
    h = max(h, word_map(uv, pos+vec2(unit*1.15, 0.), 99, sc));
    // h
    h = max(h, word_map(uv, pos+vec2(unit*1.5, 0.), 104, sc));
    
    col = mix(col, vec4(1.-vec3(h), 1.), h);
    
    return col;
}

// Function 343
float FontTexDf (vec2 p)
{
  vec3 tx;
  float d;
  int ic;
  ic = GetTxChar (p);
  if (ic != 0) {
    tx = texture (txFnt, mod ((vec2 (mod (float (ic), 16.),
       15. - floor (float (ic) / 16.)) + fract (p)) * (1. / 16.), 1.)).gba - 0.5;
    qnTex = vec2 (tx.r, - tx.g);
    d = tx.b + 1. / 256.;
  } else d = 1.;
  return d;
}

// Function 344
vec4 galaxyTexture(vec2 uv){
    
    // Cut if original uv is not in [0,1]
    bool out_of_tex = abs(uv.x-.5)>.5 || abs(uv.y-.5)>.5;
    
    // Spiral mesh
    vec4 mesh = texture(bufAChannel,uv);
    
    // Galaxy photo distortion to get a nice, not distorted spiral
    vec2 screen_ratio = vec2((R.x/R.y)/(640./360.),1);
    vec2 img_compr = GALAXY_DISTORTION*screen_ratio;
    uv = (uv-.5)*img_compr + GALAXY_CENTER;
    
    vec3 color = mix(texture(galaxyTexChannel,uv).rgb,
                     mesh.rgb,
                     DRAW_MESH ? mesh.a : 0.);
    
    out_of_tex = out_of_tex
          || abs(uv.x-.5)>.5 || abs(uv.y-.5)>.5 // Cut if new uv is not in [0,1]
          || (GALAXY==M51 && uv.x>.700); // Cut the satellite galaxy on M51
    
    return vec4(color, !out_of_tex);
}

// Function 345
float NumFont_Seven( vec2 vTexCoord )
{
    float fResult = 0.0;
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 1), vec2(11,3) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(8, 4), vec2(11,13) ));

    float fHole = NumFont_Rect( vTexCoord, vec2(9, -1), vec2(17,1) );
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(11, -1), vec2(17,3) ));
    fResult = min( fResult, 1.0 - fHole );    
    
    return fResult;
}

// Function 346
ivec2 PointToTextureCoords(in vec3 p){
	int size = 32;
    const float squareSide = 4.;
    p = floor(p.xyz);
    p.xz = mod(p.xz,float(size));
    return ivec2(p.xz)+ivec2(int(mod(p.y,squareSide))*size,int(floor(p.y/squareSide))*size);
}

// Function 347
float Glyph11(const in vec2 uv)
{
    const vec2  vP0 = vec2 ( 0.921 , 0.070 );
    const vec2  vP2 = vec2 ( 0.955 , 0.070 );
    const vec2  vP4 = vec2 ( 0.926 , 0.202 );
    const vec2  vP5 = vec2 ( 0.926 , 0.240 );
    const vec2  vP6 = vec2 ( 0.885 , 0.243 );
    const vec2  vP7 = vec2 ( 0.852 , 0.239 );
    const vec2  vP8 = vec2 ( 0.859 , 0.219 );
    const vec2  vP9 = vec2 ( 0.862 , 0.192 );
    const vec2 vP10 = vec2 ( 0.889 , 0.189 );
    const vec2 vP12 = vec2 ( 0.928 , 0.178 );
    const vec2 vP13 = vec2 ( 0.949 , 0.173 );
    const vec2 vP14 = vec2 ( 0.951 , 0.162 );
    const vec2 vP15 = vec2 ( 0.960 , 0.150 );
    const vec2 vP16 = vec2 ( 0.960 , 0.144 );
    const vec2 vP18 = vec2 ( 0.971 , 0.144 );
    const vec2 vP19 = vec2 ( 0.968 , 0.157 );
    const vec2 vP20 = vec2 ( 0.957 , 0.171 );
    const vec2 vP21 = vec2 ( 0.949 , 0.182 );
    const vec2 vP22 = vec2 ( 0.922 , 0.189 );
    const vec2 vP24 = vec2 ( 0.900 , 0.196 );
    const vec2 vP25 = vec2 ( 0.866 , 0.205 );
    const vec2 vP26 = vec2 ( 0.871 , 0.217 );
    const vec2 vP27 = vec2 ( 0.871 , 0.225 );
    const vec2 vP28 = vec2 ( 0.880 , 0.224 );
    const vec2 vP29 = vec2 ( 0.889 , 0.218 );
    const vec2 vP30 = vec2 ( 0.893 , 0.203 );

    float fDist = 1.0;
    fDist = min( fDist, InCurve2(vP4,vP5,vP6, uv) );
    fDist = min( fDist, InCurve2(vP6,vP7,vP8, uv) );
    fDist = min( fDist, InCurve2(vP8,vP9,vP10, uv) );
    fDist = min( fDist, InCurve(vP12,vP13,vP14, uv) );

    fDist = min( fDist, InCurve(vP14,vP15,vP16, uv) );
    fDist = min( fDist, InCurve2(vP18,vP19,vP20, uv) );
    fDist = min( fDist, InCurve2(vP20,vP21,vP22, uv) );

    fDist = min( fDist, InCurve(vP24,vP25,vP26, uv) );
    fDist = min( fDist, InCurve(vP26,vP27,vP28, uv) );
    fDist = min( fDist, InCurve(vP28,vP29,vP30, uv) );
    
    fDist = min( fDist, InQuad(vP0, vP2, vP4, vP30, uv) );

    fDist = min( fDist, InQuad(vP10, vP12, vP22, vP24, uv) );
        
    fDist = min( fDist, InTri(vP30, vP4, vP6, uv) );
    fDist = min( fDist, InTri(vP30, vP6, vP29, uv) );
    fDist = min( fDist, InTri(vP28, vP29, vP6, uv) );
    fDist = min( fDist, InTri(vP28, vP6, vP27, uv) );
    
    fDist = min( fDist, InTri(vP8, vP27, vP6, uv) );
    
    fDist = min( fDist, InTri(vP8, vP26, vP27, uv) );
    fDist = min( fDist, InTri(vP8, vP25, vP26, uv) );
    fDist = min( fDist, InTri(vP25, vP10, vP24, uv) );
    
    fDist = min( fDist, InTri(vP12, vP13, vP20, uv) );
    fDist = min( fDist, InTri(vP12, vP20, vP22, uv) );
    fDist = min( fDist, InTri(vP13, vP14, vP20, uv) );
    fDist = min( fDist, InTri(vP15, vP20, vP14, uv) );
    fDist = min( fDist, InTri(vP15, vP18, vP20, uv) );
    fDist = min( fDist, InTri(vP15, vP16, vP18, uv) );
    
    return fDist;
}

// Function 348
float text1(in vec2 uv) {
  const vec2 size = vec2(116.0, 16.0);
  int bit = int(uv.y * size.y);
  int xx = int(uv.x * size.x);
  float pixels = 0.0;
  if (bit >= 0 && bit <= 25 && xx >= 0 && xx < 116)
    pixels = bitget(img_text1[xx], bit);
  return pixels;
}

// Function 349
float get_texture_switch_alpha(in sampler2D s)
{
    return texelFetch(s, CTRL_TEXTURE, 0).x;
}

// Function 350
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

// Function 351
float SampleFontCharacter( int charIndex, vec2 vCharUV )
{
#if USE_FONT_TEXTURE    
    vec2 vUV;
    
    vCharUV.x = vCharUV.x * 0.6 + 0.25;
    
    vUV.x = (float(charIndex % 16) + vCharUV.x) / 16.0;
    vUV.y = (float(charIndex / 16) + vCharUV.y) / 16.0;
    
	return clamp( ( 0.503 - texture(iChannel1, vUV).w) * 100.0, 0.0, 1.0 );
#else    
	float fCharData = 0.0;
    ivec2 vCharPixel = ivec2(vCharUV * vec2(kCharPixels) );   

    #if !HIRES_FONT
        bool bCharData = CharBitmap12x20( charIndex, vCharPixel );            
        fCharData = bCharData ? 1.0 : 0.0;
    #else
        bool bCharData = CharHiRes( charIndex, vCharUV );
        fCharData = bCharData ? 1.0 : 0.0;
    #endif
    
    return fCharData;
#endif
}

// Function 352
void UI_StoreContext( inout UIContext uiContext, int iData )
{
    vec4 vData0 = vec4( uiContext.bMouseDown ? 1.0 : 0.0, float(uiContext.iActiveControl), uiContext.vActivePos.x, uiContext.vActivePos.y );
    StoreVec4( ivec2(iData,0), vData0, uiContext.vOutData, ivec2(uiContext.vFragCoord) );
}

// Function 353
DrawContext DrawContext_Init( vec2 vUV, vec3 vClearColor )
{
    vec2 vWidth = fwidth( vUV );
    
    float fEdgeFade = 1.0 / max(abs(vWidth.x), abs(vWidth.y));
    return DrawContext( vUV, vClearColor, fEdgeFade );
}

// Function 354
float get_glyph(vec2 uv, float glyph)  {return    texture(iChannel2, vec2(uv + vec2(floor(mod(glyph, 16.)), 15. - floor(glyph / 16.))) / 16.).a;}

// Function 355
void init_text_scale()
{
	g_text_scale_shift = int(max(floor(log2(iResolution.x)-log2(799.)), 0.));
}

// Function 356
vec4 fruitTexture(vec3 p, vec3 nor, float i)
{
    
    
    float rand = texCube(iChannel2, p*.1 ,nor).x;
    float t = dot(nor, normalize(vec3(.8, .1, .1)));
	vec3 mat = vec3(1.,abs(t)*rand,0);
    mat = mix(vec3(0,1,0), mat, i/10.);

   	return vec4(mat, .5);
}

// Function 357
void SetTextPosition(float x,float y){  //x=line, y=column
 tp=10.0*uv;tp.x=tp.x+17.-x;tp.y=tp.y-9.4+y;}

// Function 358
vec3 texture2(vec2 uv){
    vec3 col = vec3(0.);
    float f = step(0.5,fract(uv.y*8.));
    col = mix(col, vec3(1.,0.,1.),f);
    
    vec2 auv = abs(uv-vec2(0.5,0.));
    col = mix(col, vec3(0.), smoothstep(0.45,0.48,max(auv.x,auv.y)));
    return col;
    
}

// Function 359
vec3 texToVoxCoord(vec2 textelCoord, vec3 offset) {
	vec3 voxelCoord = offset;
    voxelCoord.xy += unswizzleChunkCoord(textelCoord / packedChunkSize);
    voxelCoord.z += mod(textelCoord.x, packedChunkSize.x) + packedChunkSize.x * mod(textelCoord.y, packedChunkSize.y);
    return voxelCoord;
}

// Function 360
float text2(in vec2 uv) {
  const vec2 size = vec2(166.0, 16.0);
  int bit = int(ceil(uv.y * size.y));
  int xx = int(ceil(uv.x * size.x));
  float pixels = 0.0;
  if (bit >= 0 && bit <= 25 && xx >= 0 && xx < 166)
  	pixels = bitget(img_text2[xx], bit);
  return pixels;
}

// Function 361
float checkersTextureGradTri( in vec3 p, in vec3 ddx, in vec3 ddy )
{
    vec3 w = max(abs(ddx), abs(ddy)) + 0.01;       // filter kernel
    vec3 i = (pri(p+w)-2.0*pri(p)+pri(p-w))/(w*w); // analytical integral (box filter)
    return 0.5 - 0.5*i.x*i.y*i.z;                  // xor pattern
}

// Function 362
float glyph(vec2 u, int l) {
    #define glyphPart(i) p = glyphParts[i];p.xy = u-p.xy;d = min(d,length(p.xy-clamp(dot(p.xy,p.zw)/dot(p.zw,p.zw),0.,1.)*p.zw));
    float d = 1e7;
    vec4 p;
    l *= 5;
    glyphPart(l);
    glyphPart(l+1);
    glyphPart(l+2);
    glyphPart(l+3);
    glyphPart(l+4);
    return d-GLYPH_WIDTH;
}

// Function 363
vec3 SampleTexture( uint iTexture, const in vec2 _vUV )
{
#ifdef HIGHLIGHT_TEXTURE    
	if(iTexture == HIGHLIGHT_TEXTURE) return vec3(1,0,0);
#endif    
    
    vec3 vResult = vec3(1.0, 0.0, 1.0);
    vec2 vUV = _vUV;
    
    vec2 vSize = vec2(64.0);
    float fPersistence = 0.8;
    
	if(iTexture == TEX_WIZMET1_2)
	{
		fPersistence = 1.2;
	}
	else if ( iTexture == TEX_LAVA1	)
	{
		fPersistence = 0.4;
	}
    else if ( iTexture == TEX_WATER1 )
    {
		fPersistence = 0.4;
    }
    else if ( iTexture == TEX_WATER2 )
    {
		fPersistence = 0.4;
    }
	else if ( iTexture == TEX_TELEPORT	)
	{
		fPersistence = 10.0;
	}
	else if ( iTexture == TEX_BRICKA2_2 )
	{
		fPersistence = 1.0;
	}
	else if ( iTexture == TEX_CITY4_7 )
	{
		fPersistence = 1.0;
	}
	else if ( iTexture == TEX_CITY4_6 )
	{
		fPersistence = 3.0;
	}
	else if ( iTexture == TEX_WBRICK1_5 )
	{
		fPersistence = 1.3;
	}	
	else if ( iTexture == TEX_SKILL0 || iTexture == TEX_SKILL1 || iTexture == TEX_SKILL2 || iTexture == TEX_QUAKE )
	{
		fPersistence = 1.0;
		//vSize = vec2( 32.0, 96.0 );
	}
    else if ( iTexture == TEX_WIZMET1_1 )
    {
		fPersistence = 1.0;
    }
    else if ( iTexture == TEX_WIZ1_4 )
    {
		fPersistence = 1.0;
    }
	else if ( iTexture == TEX_SKY1 )
	{
		fPersistence = 0.6;
	}
	else if ( iTexture == TEX_WIZWOOD1_5 )
	{
		fPersistence = 0.6;
	}
	else if ( iTexture == TEX_DEM4_1 || iTexture == TEX_WINDOW02_1 )
	{
		fPersistence = 0.6;
	}
    else if ( iTexture == TEX_COP1_1 || iTexture == TEX_COP3_4 )
    {
		fPersistence = 1.2;
    }
         
		
		
#ifdef PREVIEW
    vec2 vTexCoord = floor(fract(vUV) * vSize);
#else
    vec2 vTexCoord = fract(vUV / vSize) * vSize;
    #if PIXELATE_TEXTURES
    vTexCoord = floor(vTexCoord);
    #endif
    vTexCoord.y = vSize.y - vTexCoord.y - 1.0;
#endif
	
	float fRandom = fbm( vTexCoord, fPersistence );
    
    if(iTexture == TEX_WIZMET1_2)
	{
		vResult = WizMet1_2(vTexCoord, fRandom);
	}
	else if ( iTexture == TEX_LAVA1	)
	{
		vResult = Lava1(vTexCoord, fRandom);
	}
	else if ( iTexture == TEX_WATER1	)
	{
		vResult = Water1(vTexCoord, fRandom);
	}
	else if ( iTexture == TEX_WATER2	)
	{
		vResult = Water2(vTexCoord, fRandom);
	}
	else if ( iTexture == TEX_TELEPORT	)
	{
		vResult = Teleport(vTexCoord, fRandom);
	}
	else if ( iTexture == TEX_BRICKA2_2	)
	{
		vResult = BrickA2_2(vTexCoord, fRandom);
	}
	else if ( iTexture == TEX_CITY4_7	)
	{
		vResult = City4_7(vTexCoord, fRandom);
	}
	else if ( iTexture == TEX_WBRICK1_5 )
	{
		vResult = WBrick1_5(vTexCoord, fRandom);
	}
	else if ( iTexture == TEX_SKILL0 || iTexture == TEX_SKILL1 || iTexture == TEX_SKILL2 || iTexture == TEX_QUAKE )
	{
		vResult = Skill2(vTexCoord, fRandom);
	}
    else if ( iTexture == TEX_WIZMET1_1 )
    {
		vResult = WizMet1_1(vTexCoord, fRandom);
    }
    else if ( iTexture == TEX_WIZ1_4 )
    {
		vResult = Wiz1_4(vTexCoord, fRandom);
    }
	else if ( iTexture == TEX_SKY1 )
	{
		vResult = Sky1(vTexCoord, fRandom);
	}
    else if ( iTexture == TEX_WIZWOOD1_5)
    {
		vResult = WizWood(vTexCoord, fRandom);
    }
    else if ( iTexture == TEX_CITY4_6 )
    {
		vResult = City4_6(vTexCoord, fRandom);
    }
	else if ( iTexture == TEX_DEM4_1 || iTexture == TEX_WINDOW02_1 )
    {
		vResult = Window02_1(vTexCoord, fRandom);
    }
    else if ( iTexture == TEX_COP1_1 || iTexture == TEX_COP3_4 )
    {
		vResult = Cop1_1(vTexCoord, fRandom);
    }
        
	
    #if QUANTIZE_TEXTURES
    vResult = Quantize(vResult);
    #endif

    return vResult;
}

// Function 364
float GetFontBlend( PrintState state, LayoutStyle style, float size )
{
    float fFeatherDist = 1.0f * length(state.vPixelSize / style.vSize);    
    float f = clamp( (size-state.fDistance + fFeatherDist * 0.5f) / fFeatherDist, 0.0, 1.0);
    return f;
}

// Function 365
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

// Function 366
vec4 leavesTexture(vec3 p, vec3 nor)
{
    
    vec3 rand = texCube(iChannel2, p*.15,nor);
	vec3 mat = vec3(0.4,1.2,0) *rand;
    if (nor.y < 0.0) mat += vec3(1., 0.5,.5);
    
   	return vec4(mat, .0);
}

// Function 367
mat3 glyph_8_9_numerics(float g) {
    GLYPH(32) 0);
    GLYPH(42)0x00001000,0x00101010,0x00011100,0x00111110,0x00011100,0x00101010,0x00001000,0,0);
    GLYPH(46)0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00011000,0x00011000,0,0);
    // numerics  ==================================================
    GLYPH(49)0x00011000,0x00001000,0x00001000,0x00001000,0x00001000,0x00001000,0x00011100,0,0);
    GLYPH(50)0x00111100,0x01000010,0x00000010,0x00001100,0x00110000,0x01000000,0x01111110,0,0);
    GLYPH(51)0x00111100,0x01000010,0x00000010,0x00011100,0x00000010,0x01000010,0x00111100,0,0);
    GLYPH(52)0x01000100,0x01000100,0x01000100,0x00111110,0x00000100,0x00000100,0x00000100,0,0);
    GLYPH(53)0x01111110,0x01000000,0x01111000,0x00000100,0x00000010,0x01000100,0x00111000,0,0);
    GLYPH(54)0x00111100,0x01000010,0x01000000,0x01011100,0x01100010,0x01000010,0x00111100,0,0);
    GLYPH(55)0x00111110,0x01000010,0x00000010,0x00000100,0x00000100,0x00001000,0x00001000,0,0);
    GLYPH(56)0x00111100,0x01000010,0x01000010,0x00111100,0x01000010,0x01000010,0x00111100,0,0);
    GLYPH(57)0x00111100,0x01000010,0x01000010,0x00111110,0x00000010,0x00000010,0x00111100,0,0);
    GLYPH(58)0x00111100,0x00100100,0x01001010,0x01010010,0x01010010,0x00100100,0x00111100,0,0);
    return mat3(0);
}

// Function 368
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

// Function 369
vec3 textureNoTile( in vec2 x, float v )
{
    float k = texture( iChannel1, 0.005*x ).x; // cheap (cache friendly) lookup
    
    vec2 duvdx = dFdx( x );
    vec2 duvdy = dFdx( x );
    
    float l = k*8.0;
    float f = fract(l);
    
#if 1
    float ia = floor(l); // my method
    float ib = ia + 1.0;
#else
    float ia = floor(l+0.5); // suslik's method (see comments)
    float ib = floor(l);
    f = min(f, 1.0-f)*2.0;
#endif    
    
    vec2 offa = sin(vec2(3.0,7.0)*ia); // can replace with any other hash
    vec2 offb = sin(vec2(3.0,7.0)*ib); // can replace with any other hash

    vec3 cola = textureGrad( iChannel0, x + v*offa, duvdx, duvdy ).xyz;
    vec3 colb = textureGrad( iChannel0, x + v*offb, duvdx, duvdy ).xyz;
    
    return mix( cola, colb, smoothstep(0.2,0.8,f-0.1*sum(cola-colb)) );
}

// Function 370
float checkerTexture(in vec2 pos)
{
    float ind = 1.0;
    
    if(mod(pos.x, 1.0) > 0.5)
        ind *= -1.0;
    
    if(mod(pos.y, 1.0) > 0.5)
        ind *= -1.0;
    
    ind = max(ind, 0.0);
    
    return ind;
}

// Function 371
vec4 textureBox(vec2 uv) {
    const vec2 RES = vec2(8.0, 8.0);
    vec2 iuv = (floor(uv * RES) + 0.5) / RES;  
    float n = noise1(uv * RES);
    n = max(abs(iuv.x - 0.5), abs(iuv.y - 0.5)) * 2.0;
    n = n * n;
    n = 0.5 + n * 0.4 + noise1(uv * RES) * 0.1;
    return vec4(n, n*0.8, n*0.5, 1.0);
}

// Function 372
vec4 textureBlocky(in sampler2D tex, in vec2 uv, in vec2 res) {
    uv *= res; // enter texel coordinate space.
    
    
    vec2 seam = floor(uv+.5); // find the nearest seam between texels.
    
    // here's where the magic happens. scale up the distance to the seam so that all
    // interpolation happens in a one-pixel-wide space.
    uv = (uv-seam)/v2len(dFdx(uv),dFdy(uv))+seam;
    
    uv = clamp(uv, seam-.5, seam+.5); // clamp to the center of a texel.
    
    
    return texture(tex, uv/res, -1000.).xxxx; // convert back to 0..1 coordinate space.
}

// Function 373
float TextSDF( vec2 p, float glyph )
{
    p = abs( p.x - .5 ) > .5 || abs( p.y - .5 ) > .5 ? vec2( 0. ) : p;
    return texture( iChannel3, p / 16. + fract( vec2( glyph, 15. - floor( glyph / 16. ) ) / 16. ) ).w - 127. / 255.;
}

// Function 374
float getBlurredTexture(vec2 uv)
{
    float v = 0.;
    for (int j=0;j<btns ;j++)
    {
       float oy = float(j)*btdist/max(float(aasamples-1), 1.);
       for (int i=0;i<btns ;i++)
       {
          float ox = float(i)*btdist/max(float(aasamples-1), 1.);
          v+= dot(texture(iChannel1, uv + vec2(ox, oy)).rgb, vec3(1./3., 1./3., 1./3.));
       }
    }
    return v/float(btns*btns);
}

// Function 375
vec4 allWindowsSkyscraperTexture(vec3 p, vec2 uv, vec3 normal, vec3 bid, float xr, float obj, float w, vec3 size) {
    vec3 col = vec3(0.15);
    vec2 wuv = uv;

    float frameWidth = .03;

    float frame;

    float fogMultiplier = 0.;

    if (obj == BLD_RECT) {
        if (normal.z == 0.) {
            frame = step(uv.x, frameWidth) +  step((1. - frameWidth) * xr, uv.x) + step((1. - frameWidth), uv.y);
            frame += step(uv.y, frameWidth);
        } else {
            if (size.x > size.y) {
                frame = step(uv.x, frameWidth - .15) +  step((.95 - frameWidth) * xr, uv.x) + step((1. - frameWidth), uv.y);
            } else {
                frame = step(uv.x, frameWidth) +  step((1. - frameWidth) * xr, uv.x) + step((1. - frameWidth), uv.y);
            }
            frame += step(uv.y, frameWidth);
        }
        uv *= 40. * w / xr;
    } else if (obj == BLD_HEX) {
        vec2 huv = uv;
        if (normal.z == 0.) {
            frame = step(fract(huv.x*6. + .53), .1);
            frame += step(1.46, huv.y);
        } else {
            frame = step(1. - frameWidth * 2., hexDist((uv - .5)*rot2d(3.14)));
        }
        float scaleY = 20. * defaultBaseSize / xr;
        uv *= vec2(scaleY*(6.*xr), scaleY);
    } else if (obj == BLD_TUBE) {
        vec2 huv = uv;
        if (normal.z != 0.) {
            float hl = length(huv);
            frame = step(hl, 1.1) - step(hl, 1. - frameWidth * 2.);
        } else {
            frame = step(1. - frameWidth * 2., huv.y);
        }
        float scaleY = 20. * defaultBaseSize / xr;
        uv *= vec2(scaleY*(6.*xr), scaleY);
    }

    col += frame;

    if (normal.z == 0. && frame == 0.) {

        float bn = fract(n21(bid.xy)*567.85);
        float distToBuilding = distance(bid*CELL_SIZE, vec3(camera.x, camera.y, camera.z));

        bool isLight = bn > .6 && distToBuilding > 6. ? true : false;
        col = vec3(0.);
        vec2 id = floor(uv);
        uv = fract(uv);
        float n = n21(id + bid.xy + 22.*floor(normal.xy));
        float borders = (step(uv.x, .3) + step(uv.y, .3));
        if (!isLight && n > .7 && abs(sin(bid.x + bid.y + fract(n*23422.)*110. + iTime/50.)) > fract(n*123.22)) {
            col += COLOR_WINDOW * (1. - borders);
            col += borders * COLOR_WINDOW_TINT;
            fogMultiplier = .3;
        } else {
            if (borders != 0.) {
                col = vec3(0.2);
                if (isLight) {
                    vec2 lights = vec2(sin(wuv.x + iTime + fract(bn * 3322.)*10.), sin(wuv.y + iTime + fract(bn * 3322.)*10.));
                    if (bn > .85) {
                        col.rb += lights;
                    } else {
                        col.rg += lights;
                    }
                }
            }
        }

    }

    return vec4(col, fogMultiplier);
}

// Function 376
float gridTexture( in vec2 p )
{
	const float N = 20.0; // grid ratio

    // filter kernel
    vec2 w = max(abs(dFdx(p)), abs(dFdy(p))) + 0.001;
    //vec2 w = fwidth(p);

	// analytic (box) filtering
    vec2 a = p + 0.5*w;                        
    vec2 b = p - 0.5*w;           
    vec2 i = (floor(a)+min(fract(a)*N,1.0)-
              floor(b)-min(fract(b)*N,1.0))/(N*w);
    //pattern
    return (1.0-i.x)*(1.0-i.y);
}

// Function 377
vec3 getTexture(in vec3 p, in float m) {
    ivec2 id = getId(p);

    vec3 p0 = p;
    p.xz = mod(p.xz, wrapInterval)-0.5*wrapInterval;

    if (m != 333.) {
        if (scene==1) {
            p += deltaMan;
        } else if (scene == 2) {

            float anim = -1.1+cos(float(-id.y)*.7 + 6.*iTime);

            float bodyA = .12*anim;
            float sa=sin(bodyA); 
            float ca=cos(bodyA);
            p.yz *= mat2(ca, -sa, sa, ca);
        }
    }
    vec3 c;   

    if (m==1. || m==2.) {
        // head or body (orange bricks)
        c = vec3(1.,.5,0.);

        // Draw blinds/grate/breadbox texture
        vec2 p2 = p.xy;
        p2.y -= 1.46;
        p2 *= 100.;
        float px = abs(p2.x);
        float e = 4.-.08*px;
        float v =
            p.y < 1.73 && p.y > 0.95 && p.x > -0.5 && p.x < 0.5 ?
            mod(p.y+.1, .15) * mod(e, 5.) * 15.
            : 5000.; // ~infinity
        v = clamp(v, 0., 1.);
        c = mix(vec3(.3, .1, 0.), c, v);

        float g = mod(iTime, TAO*3.);
        //if (id.x==0 && id.y==0 && g > 2.5*TAO) {
        //    R(p.xz, -.8*cos(2.*g+1.57));
        //}
        if (p.z<-1.1) {
            // Draw face
            vec2 p2 = p.xy;
            p2.y -= 0.4;
            p2 *= 100.;
            float mouth_y = -0.;
            float mouth_thickness = 2.;
            float mouth_width = 40.;
            float eye_spacing = face_x;
            float eye_y = -face_y*1.1;
            float eye_gaze_x = -7.;
            if (id.x==0 && id.y==0 && g > 2.5*TAO) {
                eye_gaze_x = -7. * cos(2.*g+1.57);
            }
            float px = abs(p2.x);
            float e = 4.-.08*px;
            float vw = // distance to white
                // whites of the eyes
                abs(p2.y-eye_y) < 5. ? 
                length(vec2(px,p2.y)-vec2(eye_spacing,eye_y))-5.8*e
                : 5000.; // ~infinity 
            vw = clamp(vw, 0., 1.);
            float pupil_x = abs(p2.x + eye_gaze_x);
            float vb = // distance to black
                // mouth
                (px<mouth_width && abs(p2.y-mouth_y)<mouth_thickness) ? 0. :
            // pupils
            abs(p2.y-eye_y) < 5. ? 
                length(vec2(pupil_x,p2.y)-vec2(eye_spacing-5.5,eye_y))-5.8
                : 5000.; // ~infinity 
            vb = clamp(vb, 0., 1.);
            c = mix(vec3(2), c, vw);
            c = mix(vec3(0), c, vb);
        } else {
            // Draw recycling symbol
            float vb = recycling_symbol(p.zy, p.x < 0.);
            vb = clamp(vb, 0., 1.);
            c = mix(vec3(.6, .2, 0.), c, vb);
        }
    } else if (m==10.) {
        // ground
        if (scene!=1) time = 0.;
        float d = .3*sin(2.2+time);
        c = vec3(.75-.25*(mod(floor(p0.x),2.)+mod(floor(p0.z+d-time*.18),2.)));
        //c = vec3(.5+.5*smin(mod(floor(p0.x),2.),mod(floor(p0.z+d-time*.18),2.),1.));
    } else if(m == 3.) {
        // pants crotch/hinge
        c = vec3(.6,.6,.6);
        //} else if(m == 6.) {
        // hands, I guess
        //c = vec3(1.,1.,0);
    } else if(m == 4.) {
        // legs
        c = vec3(.6,.6,.6);

        vec2 p2 = p.zy * 100.;

        //float vg = mod(distance(p2, vec2(0., -80.)), 16.) - 4.;
        vec2 center = vec2(0., -80.); // center of rotation of hinge joint

        // rotate around center
        p2 -= center;
        if (scene == 1) {
            R(p2, -.4*anim);
        }else if (scene == 2) {
            R(p2, -.25*anim);
        }
        p2 += center;

        float dist_to_center = distance(p2, center);
        // using onion technique described here https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
        float vg = abs(abs(dist_to_center - 13.) - 5.) - 2.;

        vg = min(vg,
                 udQuad(p2,
                        center + vec2(-2., -85.),
                        center + vec2(+2., -85.),
                        center + vec2(+2., -20.),
                        center + vec2(-2., -20.)
                       )
                );
        float flare_width = 5.;
        float flare_length = 6.;
        for (float i = 0.; i < 4.; i+=1.) {
            vg = min(vg,
                     udQuad(p2,
                            center + vec2(-flare_length, -85.+flare_width*(i*2.+1.)),
                            center + vec2(+flare_length, -85.+flare_width*(i*2.+1.)),
                            center + vec2(+flare_length, -85.+flare_width*(i*2.+0.)),
                            center + vec2(-flare_length, -85.+flare_width*(i*2.+0.))
                           )
                    );
        }
        vg = clamp(vg, 0., 1.);
        c = mix(vec3(.2, .2, .2), c, vg);
    } else if(m == 0.) {
        c = vec3(1.2,1.2,0); // hat (yellow topper)
    } else if(m == 333.) {
        c = vec3(0.,0.,1.); // recycling bin (blue and white)

        vec3 p2 = p + recycling_bin_offset;
        if (length(p2.xz + vec2(1., -1.)) < .55) {
            // inside + rubbish
            c = vec3(1.);
        } else {
            float vb = recycling_symbol(p2.zy + vec2(-1.3, 1.), true);
            vb = clamp(vb, 0., 1.);
            c = mix(vec3(1.), c, vb);
        }
    } else {
        c = vec3(1);
    }
    if (m==10. || !(id.x==0 && id.y==0)) {
        // black & white
        float a = (c.r+c.g+c.b)*.33;
        c = vec3(1.,.95,.85)*a;
    }

    return c;
}

// Function 378
vec3 texture_wood2(vec3 p)
{
    p /= 2.;
    
    vec3 p0 = p;
    
    
    // Old trick to mix things up. I use it too much. 
    p = sin(p*4.3 + cos(p.yzx*6.7));

    
    float n = dot(p + sin(p*13.)*.03, vec3(3.));
    
    float grain = 1.-abs(dot(sin(p0*120.5 + n*6.283 + sin(p0.zxy*121.3)), vec3(.333)));

    
    // Smooth fract. Old trick. Like fract, but smoother.
    n = fract(n + fract(n*4.)*.1);
    n = min(n, n*(1.-n)*6.); // The final term controls the smoothing.
    
    float w = min(n*.85 + grain*.2, 1.);
    
    // Quick coloring. Needs work. 
    return mix(vec3(.5, .15, .025), vec3(.75, .3, .1)*2., w*.75 + .25)*(w*.6 + .4);
    
    //return vec3(w);
}

// Function 379
int glyph_index(int pixels_x)				{ return pixels_x >> 3; }

// Function 380
vec3 checkerboardTexture(in vec2 uv) 
{
    float lines = 16.0;
    float cols = 16.0;

    uv = vec2(cols, lines) * uv;

    vec3 col;
    if (mod(uv.x + floor(mod(uv.y, 2.0)), 2.0) < 1.0) {
        col = vec3(1.0, 0., 0.0);
    } else {
        col = vec3(1.0);
    }
    return col;
}

// Function 381
vec3 getColorTextura( vec3 p, vec3 nor,  int i)
{	if (i==100 )
    { vec3 col=tex3D(iChannel0, p/32., nor); return col*1.5; }
	if (i==101 ) { return tex3D(iChannel1, p/32., nor); }
	if (i==102 ) { return tex3D(iChannel2, p/32., nor); }
	if (i==103 ) { return tex3D(iChannel3, p/32., nor); }
}

// Function 382
float NumFont_Nine( vec2 vTexCoord )
{
    float fResult = NumFont_Circle( vTexCoord );
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 3), vec2(9,13) ));

    float fHole = NumFont_Rect( vTexCoord, vec2(5, 4), vec2(7,5) );
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(-1, 9), vec2(7,10) ));
    fHole = max( fHole, NumFont_Rect( vTexCoord, vec2(-1, 8), vec2(3,8) ));
    fHole = max( fHole, NumFont_Pixel( vTexCoord, vec2(-1, 7) ));

    fResult = min( fResult, 1.0 - fHole );    
    
    return fResult;
}

// Function 383
void InfoText(inout vec3 color, vec2 p, in AppState s)
{
	p -= vec2(52, 12);
	vec2 q = p;
	if (s.menuId == MENU_METAL || s.menuId == MENU_BASE_COLOR || s.menuId == MENU_DISTR)
	{
		p.y -= 6.;
	}
	if (s.menuId == MENU_DIELECTRIC || s.menuId == MENU_FRESNEL)
	{
		p.y += 6.;
	}
	if (s.menuId == MENU_SPECULAR)
	{
		p.y += 6. * 6.;

		if (p.x < 21. && p.y >= 27. && p.y < 30.)
		{
			p.y = 0.;
		}
		else if (s.menuId == MENU_SPECULAR && p.y > 20. && p.y < 28. && p.x < 21.)
		{
			p.y += 3.;
		}
	}

	vec2 scale = vec2(3., 6.);
	vec2 t = floor(p / scale);

	uint v = 0u;
	if (s.menuId == MENU_SURFACE)
	{
		v = t.y == 2. ? (t.x < 4. ? 1702127169u : (t.x < 8. ? 1768431730u : (t.x < 12. ? 1852404852u : (t.x < 16. ? 1752440935u : (t.x < 20. ? 1970479205u : (t.x < 24. ? 1667327602u : (t.x < 28. ? 1768693861u : 7628903u))))))) : v;
		v = t.y == 1. ? (t.x < 4. ? 1937334642u : (t.x < 8. ? 1717924384u : (t.x < 12. ? 1952671084u : (t.x < 16. ? 1684955424u : (t.x < 20. ? 1717924384u : (t.x < 24. ? 1952670066u : (t.x < 28. ? 32u : 0u))))))) : v;
		v = t.y == 0. ? (t.x < 4. ? 1868784481u : (t.x < 8. ? 1852400754u : (t.x < 12. ? 1869881447u : (t.x < 16. ? 1701729056u : (t.x < 20. ? 1931963500u : (t.x < 24. ? 2002873376u : 0u)))))) : v;
		v = t.x >= 0. && t.x < 32. ? v : 0u;
	}
	if (s.menuId == MENU_METAL)
	{
		v = t.y == 1. ? (t.x < 4. ? 1635018061u : (t.x < 8. ? 1852776556u : (t.x < 12. ? 1914730860u : (t.x < 16. ? 1701602917u : (t.x < 20. ? 544437347u : (t.x < 24. ? 1751607660u : (t.x < 28. ? 1914729332u : (t.x < 32. ? 544438625u : 45u)))))))) : v;
		v = t.y == 0. ? (t.x < 4. ? 544432488u : (t.x < 8. ? 2037149295u : (t.x < 12. ? 1701868320u : (t.x < 16. ? 1634497891u : (t.x < 20. ? 114u : 0u))))) : v;
		v = t.x >= 0. && t.x < 36. ? v : 0u;
	}
	if (s.menuId == MENU_DIELECTRIC)
	{
		v = t.y == 3. ? (t.x < 4. ? 1818585412u : (t.x < 8. ? 1920230245u : (t.x < 12. ? 1914725225u : (t.x < 16. ? 1701602917u : (t.x < 20. ? 544437347u : (t.x < 24. ? 1701868328u : (t.x < 28. ? 1634497891u : (t.x < 32. ? 2107762u : 0u)))))))) : v;
		v = t.y == 2. ? (t.x < 4. ? 543452769u : (t.x < 8. ? 1935832435u : (t.x < 12. ? 1634103925u : (t.x < 16. ? 1931502947u : (t.x < 20. ? 1953784163u : (t.x < 24. ? 544436837u : (t.x < 28. ? 1718182952u : (t.x < 32. ? 1702065510u : 41u)))))))) : v;
		v = t.y == 1. ? (t.x < 4. ? 1751607660u : (t.x < 8. ? 1634869364u : (t.x < 12. ? 539915129u : (t.x < 16. ? 1667592275u : (t.x < 20. ? 1918987381u : (t.x < 24. ? 544434464u : (t.x < 28. ? 1936617315u : (t.x < 32. ? 1953390964u : 0u)))))))) : v;
		v = t.y == 0. ? (t.x < 4. ? 808333438u : (t.x < 8. ? 774909234u : (t.x < 12. ? 13360u : 0u))) : v;
		v = t.x >= 0. && t.x < 36. ? v : 0u;
	}
	if (s.menuId == MENU_ROUGHNESS)
	{
		v = t.y == 2. ? (t.x < 4. ? 1735749458u : (t.x < 8. ? 544367976u : (t.x < 12. ? 1718777203u : (t.x < 16. ? 1936024417u : (t.x < 20. ? 1830825248u : (t.x < 24. ? 543519343u : (t.x < 28. ? 1952539507u : (t.x < 32. ? 1701995892u : 100u)))))))) : v;
		v = t.y == 1. ? (t.x < 4. ? 1818649970u : (t.x < 8. ? 1702126437u : (t.x < 12. ? 1768693860u : (t.x < 16. ? 544499815u : (t.x < 20. ? 1937334642u : (t.x < 24. ? 1851858988u : (t.x < 28. ? 1752440932u : (t.x < 32. ? 2126709u : 0u)))))))) : v;
		v = t.y == 0. ? (t.x < 4. ? 1920298082u : (t.x < 8. ? 1919248754u : (t.x < 12. ? 1717924384u : (t.x < 16. ? 1952671084u : (t.x < 20. ? 1936617321u : 0u))))) : v;
		v = t.x >= 0. && t.x < 36. ? v : 0u;
	}
	if (s.menuId == MENU_BASE_COLOR)
	{
		v = t.y == 1. ? (t.x < 4. ? 544370502u : (t.x < 8. ? 1635018093u : (t.x < 12. ? 1679848300u : (t.x < 16. ? 1852401253u : (t.x < 20. ? 1931506533u : (t.x < 24. ? 1969448304u : (t.x < 28. ? 544366956u : (t.x < 32. ? 1869377379u : 114u)))))))) : v;
		v = t.y == 0. ? (t.x < 4. ? 544370502u : (t.x < 8. ? 1818585444u : (t.x < 12. ? 1920230245u : (t.x < 16. ? 544433001u : (t.x < 20. ? 1768169517u : (t.x < 24. ? 1937073766u : (t.x < 28. ? 1868767333u : (t.x < 32. ? 7499628u : 0u)))))))) : v;
		v = t.x >= 0. && t.x < 36. ? v : 0u;
	}
	if (s.menuId == MENU_LIGHTING)
	{
		v = t.y == 2. ? (t.x < 4. ? 1751607628u : (t.x < 8. ? 1735289204u : (t.x < 12. ? 544434464u : (t.x < 16. ? 1869770849u : (t.x < 20. ? 1634560376u : (t.x < 24. ? 543450484u : 2128226u)))))) : v;
		v = t.y == 1. ? (t.x < 4. ? 1634755955u : (t.x < 8. ? 1769234802u : (t.x < 12. ? 1679845230u : (t.x < 16. ? 1969645161u : (t.x < 20. ? 1629513075u : (t.x < 24. ? 2122862u : 0u)))))) : v;
		v = t.y == 0. ? (t.x < 4. ? 1667592307u : (t.x < 8. ? 1918987381u : (t.x < 12. ? 1836016416u : (t.x < 16. ? 1701736304u : (t.x < 20. ? 544437358u : 0u))))) : v;
		v = t.x >= 0. && t.x < 28. ? v : 0u;
	}
	if (s.menuId == MENU_DIFFUSE)
	{
		v = t.y == 2. ? (t.x < 4. ? 1818324307u : (t.x < 8. ? 1668489324u : (t.x < 12. ? 543517793u : (t.x < 16. ? 1935832435u : (t.x < 20. ? 1634103925u : (t.x < 24. ? 1931502947u : (t.x < 28. ? 1953784163u : (t.x < 32. ? 1852404325u : 8295u)))))))) : v;
		v = t.y == 1. ? (t.x < 4. ? 1635087189u : (t.x < 8. ? 981036140u : (t.x < 12. ? 1835093024u : (t.x < 16. ? 1953654114u : (t.x < 20. ? 1146241568u : (t.x < 24. ? 1713388102u : (t.x < 28. ? 824196384u : (t.x < 32. ? 543780911u : 0u)))))))) : v;
		v = t.y == 0. ? (t.x < 4. ? 1702257960u : (t.x < 8. ? 1914730866u : (t.x < 12. ? 1696627041u : (t.x < 16. ? 1937009016u : (t.x < 20. ? 544106784u : (t.x < 24. ? 1634869345u : (t.x < 28. ? 1679844462u : (t.x < 32. ? 2716265u : 0u)))))))) : v;
		v = t.x >= 0. && t.x < 36. ? v : 0u;
	}
	if (s.menuId == MENU_SPECULAR)
	{
		v = t.y == 8. ? (t.x < 4. ? 1818649938u : (t.x < 8. ? 1702126437u : (t.x < 12. ? 1768693860u : (t.x < 16. ? 779380839u : (t.x < 20. ? 1970492704u : (t.x < 24. ? 2037148769u : 8250u)))))) : v;
		v = t.y == 7. ? (t.x < 4. ? 1802465091u : (t.x < 8. ? 1919898669u : (t.x < 12. ? 1668178290u : (t.x < 16. ? 1998597221u : (t.x < 20. ? 1751345512u : (t.x < 24. ? 1685024032u : 7564389u)))))) : v;
		v = t.y == 6. ? (t.x < 4. ? 1919117677u : (t.x < 8. ? 1667327599u : (t.x < 12. ? 544437349u : (t.x < 16. ? 1919250472u : (t.x < 20. ? 1952671078u : (t.x < 24. ? 1919511840u : 544370546u)))))) : v;
		v = t.y == 5. ? (t.x < 4. ? 1734960488u : (t.x < 8. ? 1634563176u : (t.x < 12. ? 3811696u : 0u))) : v;
		v = t.y == 4. ? (t.x < 4. ? 745285734u : (t.x < 8. ? 1178413430u : (t.x < 12. ? 1747744296u : (t.x < 16. ? 1814578985u : (t.x < 20. ? 1747744300u : (t.x < 24. ? 1747469353u : 41u)))))) : v;
		v = t.y == 3. ? (t.x < 4. ? 538976288u : (t.x < 8. ? 538976288u : (t.x < 12. ? 1848128544u : (t.x < 16. ? 673803447u : (t.x < 20. ? 695646062u : 0u))))) : v;
		v = t.y == 2. ? (t.x < 4. ? 539828294u : (t.x < 8. ? 1936028230u : (t.x < 12. ? 7103854u : 0u))) : v;
		v = t.y == 1. ? (t.x < 4. ? 539828295u : (t.x < 8. ? 1836016967u : (t.x < 12. ? 2037544037u : 0u))) : v;
		v = t.y == 0. ? (t.x < 4. ? 539828292u : (t.x < 8. ? 1953720644u : (t.x < 12. ? 1969383794u : (t.x < 16. ? 1852795252u : 0u)))) : v;
		v = t.x >= 0. && t.x < 28. ? v : 0u;
	}
	if (s.menuId == MENU_DISTR)
	{
		v = t.y == 1. ? (t.x < 4. ? 1702109252u : (t.x < 8. ? 1679846770u : (t.x < 12. ? 1852401253u : (t.x < 16. ? 622883685u : (t.x < 20. ? 543584032u : (t.x < 24. ? 1919117677u : (t.x < 28. ? 1667327599u : 544437349u))))))) : v;
		v = t.y == 0. ? (t.x < 4. ? 1818649970u : (t.x < 8. ? 1769235301u : (t.x < 12. ? 1814062958u : (t.x < 16. ? 1952999273u : (t.x < 20. ? 1919903264u : (t.x < 24. ? 1730175264u : (t.x < 28. ? 1852143209u : 1919509536u))))))) : v;
		v = t.x >= 0. && t.x < 32. ? v : 0u;
	}
	if (s.menuId == MENU_FRESNEL)
	{
		v = t.y == 3. ? (t.x < 4. ? 1702109254u : (t.x < 8. ? 1679846770u : (t.x < 12. ? 1852401253u : (t.x < 16. ? 1629516645u : (t.x < 20. ? 1853189997u : (t.x < 24. ? 1718558836u : 32u)))))) : v;
		v = t.y == 2. ? (t.x < 4. ? 1818649970u : (t.x < 8. ? 1702126437u : (t.x < 12. ? 1768693860u : (t.x < 16. ? 544499815u : (t.x < 20. ? 544370534u : (t.x < 24. ? 1768366177u : 544105846u)))))) : v;
		v = t.y == 1. ? (t.x < 4. ? 1935832435u : (t.x < 8. ? 1851880052u : (t.x < 12. ? 539911523u : (t.x < 16. ? 1629516873u : (t.x < 20. ? 1869770864u : (t.x < 24. ? 1701340001u : 3219571u)))))) : v;
		v = t.y == 0. ? (t.x < 4. ? 544370534u : (t.x < 8. ? 2053206631u : (t.x < 12. ? 543649385u : (t.x < 16. ? 1818717793u : (t.x < 20. ? 29541u : 0u))))) : v;
		v = t.x >= 0. && t.x < 28. ? v : 0u;
	}
	if (s.menuId == MENU_GEOMETRY)
	{
		v = t.y == 2. ? (t.x < 4. ? 1702109255u : (t.x < 8. ? 1679846770u : (t.x < 12. ? 1852401253u : (t.x < 16. ? 1931506533u : (t.x < 20. ? 1868849512u : (t.x < 24. ? 1735289207u : (t.x < 28. ? 543584032u : 0u))))))) : v;
		v = t.y == 1. ? (t.x < 4. ? 1919117677u : (t.x < 8. ? 1667327599u : (t.x < 12. ? 544437349u : (t.x < 16. ? 1701864804u : (t.x < 20. ? 1852400750u : (t.x < 24. ? 1852776551u : (t.x < 28. ? 1701344288u : 2126441u))))))) : v;
		v = t.y == 0. ? (t.x < 4. ? 1634890337u : (t.x < 8. ? 1835362158u : (t.x < 12. ? 7630437u : 0u))) : v;
		v = t.x >= 0. && t.x < 32. ? v : 0u;
	}

	float c = float((v >> uint(8. * t.x)) & 255u);

	vec3 textColor = vec3(.3);

	p = (p - t * scale) / scale;
	p.x = (p.x - .5) * .45 + .5;
	float sdf = TextSDF(p, c);
	if (c != 0.)
	{
		color = mix(textColor, color, smoothstep(-.05, +.05, sdf));
	}

	if (s.menuId == MENU_SPECULAR)
	{
		color = mix(color, textColor, smoothstep(.05, -.05, Capsule(q.yx - vec2(-12.3, 48.), .3, 26.)));
	}
}

// Function 384
vec3 lightsTexture(vec2 uv){
    vec3 col = vec3(0.);
    vec2 auv = abs(uv-vec2(0.5));
    float f = 1.0-smoothstep(0.2,0.35,max(auv.x,auv.y));
    f = pow(f,2.);
    //col = mix(col, vec3(1.2,1.2,0.9), f);
    col = mix(col, vec3(0.95,0.95,1.), f);
    return col;
}

// Function 385
define fontString(f,st) 	{for(int _f_i= 0;_f_i<st.length();_f_i++ )f+= _(st[_f_i]);}

// Function 386
float text_g(vec2 U) {
    initMsg;
    U.x+=6.*(0.5-0.2812*(res.x/0.5));
    C(71);C(114);C(97);C(118);C(105);C(116);C(121);
    endMsg;
}

// Function 387
vec3 BicubicLagrangeTextureSample (vec2 P)
{
    vec2 pixel = P * c_textureSize + 0.5;
    
    vec2 frac = fract(pixel);
    pixel = floor(pixel) / c_textureSize - vec2(c_onePixel/2.0);
    
    vec3 C00 = texture(iChannel0, pixel + vec2(-c_onePixel ,-c_onePixel)).rgb;
    vec3 C10 = texture(iChannel0, pixel + vec2( 0.0        ,-c_onePixel)).rgb;
    vec3 C20 = texture(iChannel0, pixel + vec2( c_onePixel ,-c_onePixel)).rgb;
    vec3 C30 = texture(iChannel0, pixel + vec2( c_twoPixels,-c_onePixel)).rgb;
    
    vec3 C01 = texture(iChannel0, pixel + vec2(-c_onePixel , 0.0)).rgb;
    vec3 C11 = texture(iChannel0, pixel + vec2( 0.0        , 0.0)).rgb;
    vec3 C21 = texture(iChannel0, pixel + vec2( c_onePixel , 0.0)).rgb;
    vec3 C31 = texture(iChannel0, pixel + vec2( c_twoPixels, 0.0)).rgb;    
    
    vec3 C02 = texture(iChannel0, pixel + vec2(-c_onePixel , c_onePixel)).rgb;
    vec3 C12 = texture(iChannel0, pixel + vec2( 0.0        , c_onePixel)).rgb;
    vec3 C22 = texture(iChannel0, pixel + vec2( c_onePixel , c_onePixel)).rgb;
    vec3 C32 = texture(iChannel0, pixel + vec2( c_twoPixels, c_onePixel)).rgb;    
    
    vec3 C03 = texture(iChannel0, pixel + vec2(-c_onePixel , c_twoPixels)).rgb;
    vec3 C13 = texture(iChannel0, pixel + vec2( 0.0        , c_twoPixels)).rgb;
    vec3 C23 = texture(iChannel0, pixel + vec2( c_onePixel , c_twoPixels)).rgb;
    vec3 C33 = texture(iChannel0, pixel + vec2( c_twoPixels, c_twoPixels)).rgb;    
    
    vec3 CP0X = CubicLagrange(C00, C10, C20, C30, frac.x);
    vec3 CP1X = CubicLagrange(C01, C11, C21, C31, frac.x);
    vec3 CP2X = CubicLagrange(C02, C12, C22, C32, frac.x);
    vec3 CP3X = CubicLagrange(C03, C13, C23, C33, frac.x);
    
    return CubicLagrange(CP0X, CP1X, CP2X, CP3X, frac.y);
}

// Function 388
void menuText( inout vec3 color, vec2 p, in AppState s )
{        
    vec2 scale = vec2( 4., 8. );
    vec2 t = floor( p / scale );   
    
    uint v = 0u;
	v = t.y == 2. ? ( t.x < 4. ? 1768452929u : ( t.x < 8. ? 1768777835u : ( t.x < 12. ? 5653614u : 0u ) ) ) : v;
	v = t.y == 1. ? ( t.x < 4. ? 1918986307u : ( t.x < 8. ? 1147496812u : ( t.x < 12. ? 1752383839u : ( t.x < 16. ? 1835559785u : 5664361u ) ) ) ) : v;
	v = t.y == 0. ? ( t.x < 4. ? 1918986307u : ( t.x < 8. ? 1147496812u : ( t.x < 12. ? 86u : 0u ) ) ) : v;
	v = t.x >= 0. && t.x < 20. ? v : 0u;
    
	float c = float( ( v >> uint( 8. * t.x ) ) & 255u );
    
	vec3 textColor = vec3( 1.0 );

    p = ( p - t * scale ) / scale;
    p.x = ( p.x - .5 ) * .5 + .5;
    float sdf = textSDF( p, c );
    if ( c != 0. )
    {
    	color = mix( textColor, color, smoothstep( -.05, +.05, sdf ) );
    }
}

// Function 389
vec4 NumFont_Char( vec2 vCharUV, int iDigit )
{
 	if ( iDigit < 0 )
    	return vec4(0.0);
    
    ivec2 vTexCoord = ivec2(floor(vCharUV * vec2(14.0, 16.0))) + FONT_POS;
    vTexCoord += iDigit * FONT_CHAR;
        
    return texelFetch( iChannel2, vTexCoord, 0 );
}

// Function 390
float text_pmhp(vec2 U, int v) {
    initMsg;C((v * 43 + 45 * (1 - v)));C(72);C(80);endMsg;
}

// Function 391
void DrawText( inout vec3 color, vec2 edge, vec2 center, vec2 world, in GameState s )
{
    // xp
    if ( s.logPos[ 0 ].x > 0. )
    {
        float t = 1e4;
        
        vec2 p = world;
        p -= s.logPos[ 0 ] * 16.;
        p.x += 8.;
        p.y -= s.logLife[ 0 ] * 16.;
        PrintChar( t, p, 43. );
        PrintVal( t, p, s.logVal[ 0 ] );
        PrintChar( t, p, 69. );
        PrintChar( t, p, 88. );
        PrintChar( t, p, 80. );
        
		if ( s.logId[ 0 ] > 0. )
        {
            p = world;
            p -= s.logPos[ 0 ] * 16.;
            p.x += 16.;
            p.y -= s.logLife[ 0 ] * 16. - 8.;
           	PrintChar( t, p, 76. );
            PrintChar( t, p, 69. );
            PrintChar( t, p, 86. );
            PrintChar( t, p, 69. );
            PrintChar( t, p, 76. );
            PrintChar( t, p, 32. );
            PrintChar( t, p, 85. );
            PrintChar( t, p, 80. );
            PrintChar( t, p, 33. );
        }
        
        RastText( color, t, s.logLife[ 0 ], vec3( 1., 1., 0. ) );
    }    
    
    // heal
    if ( s.logPos[ 1 ].x > 0. )
    {
        float t = 1e4; 
        vec2 p = world;
        p -= s.logPos[ 1 ] * 16.;
        p.x += 8.;
        p.y -= s.logLife[ 1 ] * 16.;      
        PrintChar( t, p, 43. );
        PrintVal( t, p, s.logVal[ 1 ] );
        PrintChar( t, p, 72. );
        PrintChar( t, p, 80. );
        RastText( color, t, s.logLife[ 1 ], vec3( 0., 1., 0. ) ); 
    }
    
    // dmg
    for ( int i = 2; i < LOG_NUM; ++i )
    {
		float t = 1e4;        
        
        if ( s.logPos[ i ].x > 0. )
        {
            vec2 p = world;
            p -= s.logPos[ i ] * 16.;
            p.y -= s.logLife[ i ] * 16.;        
            PrintVal( t, p, s.logVal[ i ] );
        }
        
        RastText( color, t, s.logLife[ i ], vec3( 1., 0., 0. ) );     
    }
    
    // game over
    if ( s.state == STATE_GAME_OVER )
    {      
        float alpha = Smooth( ( s.stateTime - 0.33 ) * 4. );
        
        color = mix( color, color.yyy * .5, alpha );
        
        float t = 1e4; 
        
        vec2 p = .25 * center;
        p.x += 24.;
        p.y += 6.;
        PrintChar( t, p, 89. );
        PrintChar( t, p, 79. );
        PrintChar( t, p, 85. );
        p.x -= 4.;
        PrintChar( t, p, 68. );
        PrintChar( t, p, 73. );
        PrintChar( t, p, 69. );
        PrintChar( t, p, 68. );
        
        RastText( color, t, 1. - alpha, vec3( 1., 0., 0. ) );     
    }
    
    // level
    vec2 p = edge + vec2( 2.2, 20.8 );
    float t = 1e4;
    PrintChar( t, p, 48. + s.level + 1. );
    color = mix( color, vec3( 1. ), Smooth( -t * 100. ) ); 
}

// Function 392
vec4 textureSmootherstep(sampler2D tex, vec2 uv, vec2 res)
{
	uv = uv*res + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*fuv*(fuv*(fuv*6.0-15.0)+10.0);
	uv = (uv - 0.5)/res;
	return texture( tex, uv );
}

// Function 393
float textSDF( vec2 p, float glyph )
{
    p = abs( p.x - .5 ) > .5 || abs( p.y - .5 ) > .5 ? vec2( 0. ) : p;
    return 2. * ( texture( iChannel3, p / 16. + fract( vec2( glyph, 15. - floor( glyph / 16. ) ) / 16. ) ).w - 127. / 255. );
}

// Function 394
vec3 TexTekWall1( vec2 vTexCoord, float fRandom )
{
	vec3 col = mix( vec3(39.0, 39.0, 39.0), vec3(51.0, 51.0, 51.0), fRandom * fRandom) / 255.0;
	
	return col;
}

// Function 395
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

// Function 396
vec4 textureAniso(sampler2D T, vec2 p) {
    mat2 J = inverse(mat2(dFdx(p),dFdy(p)));       // dFdxy: pixel footprint in texture space
    J = transpose(J)*J;                            // quadratic form
    float d = determinant(J), t = J[0][0]+J[1][1], // find ellipse: eigenvalues, max eigenvector
          D = sqrt(t*t-4.*d), 
          V = (t-D)/2., v = (t+D)/2.,                     // eigenvalues 
          M = 1./sqrt(V), m = 1./sqrt(v), l =log2(m*R.y); // = 1./radii^2
  //if (M/m>16.) l = log2(M/16.*R.y);                     // optional
    vec2 A = M * normalize(vec2( -J[0][1] , J[0][0]-V )); // max eigenvector = main axis
    vec4 O = vec4(0);
    for (float i = -7.5; i<8.; i++)                       // sample x16 along main axis at LOD min-radius
        O += textureLod(iChannel0, p+(i/16.)*A, l);
    return O/16.;
}

// Function 397
void glyph_3()
{
  MoveTo(x*0.2+y*1.7);
  Bez3To(x*0.4+y*2.15,x*1.7+y*2.15,x*1.7+y*1.55);
  Bez3To(x*1.7+y*1.3,x*1.4+y*1.1,x*0.8+y*1.1);
  Bez3To(x*1.4+y*1.1,x*1.8+y*0.9,x*1.8+y*0.55);
  Bez3To(x*1.8-y*0.2,x*0.4-y*0.2,x*0.2+y*0.3);
}

// Function 398
float bumpTexture(vec3 p) {
    return 1. - _texture(p).g;
}

// Function 399
vec3 textureWall(vec2 pos, vec2 maxPos, vec2 squarer,float s,float height,float dist,vec3 rayDir,vec3 norm){
    float randB = rand(squarer*2.0);
    vec3 windowColor =(-0.4+randB*0.8)*vec3(0.3,0.3,0.0)+(-0.4+fract(randB*10.0)*0.8)*vec3(0.0,0.0,0.3)+(-0.4+fract(randB*10000.0)*0.8)*vec3(0.3,0.0,0.0);
    float floorFactor = 1.0;
    vec2 windowSize = vec2(0.65,0.35);
    vec3 wallColor = s*(0.3+1.4*fract(randB*100.0))*vec3(0.1,0.1,0.1)+(-0.7+1.4*fract(randB*1000.0))*vec3(0.02,0.,0.);
	wallColor*=1.3;
    
    vec3 color = vec3(0.0);
    vec3 conturColor = wallColor/1.5;
    if (height<0.51){
    	windowColor += vec3(0.3,0.3,0.0);
        windowSize = vec2(0.4,0.4);
        floorFactor = 0.0;

    }
    if (height<0.6){floorFactor = 0.0;}
    if (height>0.75){
    	windowColor += vec3(0.0,0.0,0.3);
    }
    windowColor*=1.5;
    float wsize = 0.02;
    wsize+=-0.007+0.014*fract(randB*75389.9365);
    windowSize+= vec2(0.34*fract(randB*45696.9365),0.50*fract(randB*853993.5783));
    
    vec2 contur=vec2(0.0)+(fract(maxPos/2.0/wsize))*wsize;
    if (contur.x<wsize){contur.x+=wsize;}
    if (contur.y<wsize){contur.y+=wsize;}
    
	vec2 winPos = (pos-contur)/wsize/2.0-floor((pos-contur)/wsize/2.0);
    
    float numWin = floor((maxPos-contur)/wsize/2.0).x;
    
    if ( (maxPos.x>0.5&&maxPos.x<0.6) && ( ((pos-contur).x>wsize*2.0*floor(numWin/2.0)) && ((pos-contur).x<wsize*2.0+wsize*2.0*floor(numWin/2.0)) )){
     	   return (0.9+0.2*noise(pos))*conturColor;
    }
    
    if ( (maxPos.x>0.6&&maxPos.x<0.7) &&( ( ((pos-contur).x>wsize*2.0*floor(numWin/3.0)) && ((pos-contur).x<wsize*2.0+wsize*2.0*floor(numWin/3.0)) )||
                                          ( ((pos-contur).x>wsize*2.0*floor(numWin*2.0/3.0)) && ((pos-contur).x<wsize*2.0+wsize*2.0*floor(numWin*2.0/3.0)) )) ){
     	   return (0.9+0.2*noise(pos))*conturColor;
    }
    
    if ( (maxPos.x>0.7) &&( ( ((pos-contur).x>wsize*2.0*floor(numWin/4.0)) && ((pos-contur).x<wsize*2.0+wsize*2.0*floor(numWin/4.0)) )||
                                          ( ((pos-contur).x>wsize*2.0*floor(numWin*2.0/4.0)) && ((pos-contur).x<wsize*2.0+wsize*2.0*floor(numWin*2.0/4.0)) )||
                                          ( ((pos-contur).x>wsize*2.0*floor(numWin*3.0/4.0)) && ((pos-contur).x<wsize*2.0+wsize*2.0*floor(numWin*3.0/4.0)) )) ){
     	   return (0.9+0.2*noise(pos))*conturColor;
    }
    if ((maxPos.x-pos.x<contur.x)||(maxPos.y-pos.y<contur.y+2.0*wsize)||(pos.x<contur.x)||(pos.y<contur.y)){
            return (0.9+0.2*noise(pos))*conturColor;
        
    }
    if (maxPos.x<0.14) {
     	   return (0.9+0.2*noise(pos))*wallColor;
    }
    vec2 window = floor((pos-contur)/wsize/2.0);
    float random = rand(squarer*s*maxPos.y+window);
    float randomZ = rand(squarer*s*maxPos.y+floor(vec2((pos-contur).y,(pos-contur).y)/wsize/2.0));
    float windows = floorFactor*sin(randomZ*5342.475379+(fract(975.568*randomZ)*0.15+0.05)*window.x);
    
	float blH = 0.06*dist*600.0/iResolution.x/abs(dot(normalize(rayDir.xy),normalize(norm.xy)));
    float blV = 0.06*dist*600.0/iResolution.x/sqrt(abs(1.0-pow(abs(rayDir.z),2.0)));
    
	windowColor +=vec3(1.0,1.0,1.0);
    windowColor *= smoothstep(0.5-windowSize.x/2.0-blH,0.5-windowSize.x/2.0+blH,winPos.x);
   	windowColor *= smoothstep(0.5+windowSize.x/2.0+blH,0.5+windowSize.x/2.0-blH,winPos.x);
   	windowColor *= smoothstep(0.5-windowSize.y/2.0-blV,0.5-windowSize.y/2.0+blV,winPos.y);
   	windowColor *= smoothstep(0.5+windowSize.y/2.0+blV,0.5+windowSize.y/2.0-blV,winPos.y);
    
    if ((random <0.05*(3.5-2.5*floorFactor))||(windows>0.65)){
        	if (winPos.y<0.5) {windowColor*=(1.0-0.4*fract(random*100.0));}
        	if ((winPos.y>0.5)&&(winPos.x<0.5)) {windowColor*=(1.0-0.4*fract(random*10.0));}
            return (0.9+0.2*noise(pos))*wallColor+(0.9+0.2*noise(pos))*windowColor;


    } 
    else{
        windowColor*=0.08*fract(10.0*random);
    }
    
    return (0.9+0.2*noise(pos))*wallColor+windowColor;

}

// Function 400
vec3 getTexture(in vec3 p, in float m) {
    ivec2 id = getId(p);

	vec3 p0 = p;
    float k = 5.;
    p.xz = mod(p.xz, k)-0.5*k;
    
	if (scene==1) {
		p += deltaMan;
	} else if (scene == 2) {
		
		float anim = -1.1+cos(float(-id.y)*.7 + 6.*iTime);
		
  		float bodyA = .12*anim;
		float sa=sin(bodyA); 
		float ca=cos(bodyA);
		p.yz *= mat2(ca, -sa, sa, ca);
	}	
    vec3 c;   
  
    if (m==1.) {
		c = vec3(1.,1.,0);
		float g = mod(iTime, TAO*3.);
		if (id.x==0 && id.y==0 && g > 2.5*TAO) {
			R(p.xz, -.8*cos(2.*g+1.57));
		}
		if (p.z<0.) {
			// Draw face
			vec2 p2 = p.xy;
			p2.y -= 1.46;
			p2 *= 100.;
			float px = abs(p2.x);
			float e = 4.-.08*px;
			float v = 
					(px<face_x && p2.y<-e) ? abs(length(p2)-face_r)-e : 
					(p2.y<-e) ? length(vec2(px,p2.y)-vec2(face_x,face_y))-e :
					length(vec2(px,p2.y)-vec2(face_x,-face_y*.1))-1.8*e; 
			v = clamp(v, 0., 1.);
			c = mix(vec3(0), c, v);
		}
    }
    else if (m==2.) {
        c = (id.x==0 && id.y==0) ? mandelbrot(p.xy - vec2(.14,.15)) : vec3(1,0,0);
       
	} else if (m==10.) {
		if (scene!=1) time = 0.;
		float d = .3*sin(2.2+time);
		c = vec3(.75-.25*(mod(floor(p0.x),2.)+mod(floor(p0.z+d-time*.18),2.)));
	 	//c = vec3(.5+.5*smin(mod(floor(p0.x),2.),mod(floor(p0.z+d-time*.18),2.),1.));
	} else {
        c = m == 6. ? vec3(1.,1.,0)  :
			m == 3. ? vec3(.2,.2,.4) :
			m == 4. ? vec3(.1,.1,.2) :
			          vec3(1.,1.,1.);
		
    }
    if (m==10. || !(id.x==0 && id.y==0)) {
		// black & white
        float a = (c.r+c.g+c.b)*.33;
		c = vec3(1.,.95,.85)*a;
    }

	return c;
}

// Function 401
vec4 repeatedTexture(in sampler2D channel, in vec2 uv) {
    return texture(channel, mod(uv, 1.));
}

// Function 402
float escherTextureContour_(vec2 p, float linewidth, float pixel_size)
{
    vec2 pp = mod(p,1.0);
    
    float d = 10000000.0;
    for(int i=0; i<vert.length()-1; ++i)
    {
        // d = min(d, ToroidalDistance(pp, vert[i]));
        d = min(d, PointSegToroidalDistance(pp, vert[i], vert[i+1]));
    }
    
    d = smoothstep(0.0, 1.0, (sqrt(d)-linewidth)/pixel_size);
    
    return d;
}

// Function 403
float text( vec3 pos )
{
    
    vec3 p = pos;
    pR(p.xy, iTime);
    
    pos.x+=15.;
    pos.z-= 2.;
    
   // pR(pos.xy, sin(iTime)*.1);
    //TODO optimize
    spacing.y = spacing.x*.5;
    spacing.z = 1./spacing.x;
    
    float x = 100.;
    float nr = 0.;
    vec2 uv = pos.xz; 
    float width = textWidth;
    line1;
    width+=sin(iTime*3.1415-length(pos.xz))*width*(iMouse.y/iResolution.y);
    x = length(vec2(x, pos.y));
    x-=width;
    
    
    return x;
        
}

// Function 404
vec4 texture3( sampler3D sam, vec3 uv )
{
#ifndef SHOW_DERIVATIVES    
    return texture(sam,uv);
#else    
    float res = float(textureSize(sam,0).x);
    uv = uv*res - 0.5;
    vec3 iuv = floor(uv);
    vec3 f = fract(uv);
	vec4 rg1 = textureLod( sam, (iuv+ vec3(0.5,0.5,0.5))/res, 0.0 );
	vec4 rg2 = textureLod( sam, (iuv+ vec3(1.5,0.5,0.5))/res, 0.0 );
	vec4 rg3 = textureLod( sam, (iuv+ vec3(0.5,1.5,0.5))/res, 0.0 );
	vec4 rg4 = textureLod( sam, (iuv+ vec3(1.5,1.5,0.5))/res, 0.0 );
	vec4 rg5 = textureLod( sam, (iuv+ vec3(0.5,0.5,1.5))/res, 0.0 );
	vec4 rg6 = textureLod( sam, (iuv+ vec3(1.5,0.5,1.5))/res, 0.0 );
	vec4 rg7 = textureLod( sam, (iuv+ vec3(0.5,1.5,1.5))/res, 0.0 );
	vec4 rg8 = textureLod( sam, (iuv+ vec3(1.5,1.5,1.5))/res, 0.0 );
	return mix(mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y ),
               mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y ), f.z );
#endif    
}

// Function 405
vec3 texture_value(const in texture_ t, const in vec3 p) {
    if (t.type == SOLID) {
	    return t.v;
    } else if (t.type == NOISE) {
        return vec3(.5*(1. + sin(t.v.x*p.z + 5.*fbm((t.v.x*.5)*p, 7))));
    }
}

// Function 406
vec4 draw_font8x8_char( int ch, vec4 col, ivec2 pos, inout vec4 o, ivec2 iu ) {
    vec4 v = vec4( -1 ) ;
    iu -= pos ;
    if( ch > 0 && iINSIDE( iu, ivec2(0), ivec2(8) ) ) {
        if( col.a == 0. ) {
            col.a = 1. ;
        } else {
            v = vec4( 0,0,0,1 ) ;
        }
            
        ch -- ;
        int row_group = ( ch >> 2 ) * 2 + 1 - ( iu.y >> 2 ),
            component = 3 - ( iu.y & 0x3 ),
            sh = iu.x ;
        uint bit = 0x1U << sh,
             col_ind = ( get_font8x8_br( row_group, component, ch & 0x3 ) & bit ) >> sh ;
        v = col_ind == 1U ? col : v ;
    }
    o = v.a > 0. ? v : o ;
    return( v ) ;
}

// Function 407
vec3 TextureZepllin(in vec3 p)
{
  return mix(vec3(1.0, 0.99, 0.),vec3(1.0, 0.99, 0.)*1.5,Fbm(p/80.0));   
}

// Function 408
vec4 textureNoTile(sampler2D iChannelT, sampler2D iChannelR, vec2 x, float v) {
    float k = texture(iChannelR, 0.005*x).x; // cheap (cache friendly) lookup
    
    vec2 duvdx = dFdx(x);
    vec2 duvdy = dFdy(x);
    
    float l = k*8.0;
    float f = fract(l);
    
    float ia = floor(l); // my method
    float ib = ia + 1.0;
    
    vec2 offa = sin(vec2(3.0,7.0)*ia); // can replace with any other hash
    vec2 offb = sin(vec2(3.0,7.0)*ib); // can replace with any other hash

    vec4 cola = textureGrad(iChannelT, x + v*offa, duvdx, duvdy);
    vec4 colb = textureGrad(iChannelT, x + v*offb, duvdx, duvdy);
    
    vec3 colc = cola.rgb-colb.rgb;
    
    return mix(cola, colb, smoothstep(0.2,0.8,f-0.1*(colc.x+colc.y+colc.z)));
}

// Function 409
vec4 put_text_fixed(vec4 col, vec2 uv, vec2 pos, float scale, bool p)
{
	float unit = asp * scale * 0.1;
    float h = 0.;
    vec2 sc = vec2(unit, unit*0.8);
    
    // F
    h = max(h, word_map(uv, pos, 70, sc));
    // i
    h = max(h, word_map(uv, pos+vec2(unit*0.35, 0.), 105, sc));
    // x
    h = max(h, word_map(uv, pos+vec2(unit*0.7, 0.), 120, sc));
    // e
    h = max(h, word_map(uv, pos+vec2(unit*1.05, 0.), 101, sc));
    // d
    h = max(h, word_map(uv, pos+vec2(unit*1.4, 0.), 100, sc));
    
    if(p){
        //o
    	h = max(h, word_map(uv, pos+vec2(unit*2.1, 0.), 111, sc));
        //r
        h = max(h, word_map(uv, pos+vec2(unit*2.45, 0.), 114, sc));
        //i
        h = max(h, word_map(uv, pos+vec2(unit*2.8, 0.), 105, sc));
        //g
        h = max(h, word_map(uv, pos+vec2(unit*3.15, 0.), 103, sc));
        //i
        h = max(h, word_map(uv, pos+vec2(unit*3.5, 0.), 105, sc));
        //n
        h = max(h, word_map(uv, pos+vec2(unit*3.85, 0.), 110, sc));
    }
    else{
        //t
    	h = max(h, word_map(uv, pos+vec2(unit*2.1, 0.), 116, sc));
        //a
        h = max(h, word_map(uv, pos+vec2(unit*2.45, 0.), 97, sc));
        //r
        h = max(h, word_map(uv, pos+vec2(unit*2.8, 0.), 114, sc));
        //g
        h = max(h, word_map(uv, pos+vec2(unit*3.15, 0.), 103, sc));
        //e
        h = max(h, word_map(uv, pos+vec2(unit*3.5, 0.), 101, sc));
        //t
        h = max(h, word_map(uv, pos+vec2(unit*3.85, 0.), 116, sc));
    }
    
    col = mix(col, vec4(1.-vec3(h), 1.), h);
    
    return col;
}

// Function 410
void MenuText(inout vec3 color, vec2 p, in AppState s)
{
	p -= vec2(-160, -1);

	vec2 scale = vec2(4., 8.);
	vec2 t = floor(p / scale);

	float tab = 1.;
	if (t.y >= 6. && t.y < 10.)
	{
		p.x -= tab * scale.x;
		t.x -= tab;
	}
	if (t.y >= 0. && t.y < 5.)
	{
		p.x -= tab * scale.x;
		t.x -= tab;
	}
	if (t.y >= 0. && t.y < 3.)
	{
		p.x -= tab * scale.x;
		t.x -= tab;
	}

	uint v = 0u;
	v = t.y == 10. ? (t.x < 4. ? 1718777171u : (t.x < 8. ? 6644577u : 0u)) : v;
	v = t.y == 9. ? (t.x < 4. ? 1635018061u : (t.x < 8. ? 108u : 0u)) : v;
	v = t.y == 8. ? (t.x < 4. ? 1818585412u : (t.x < 8. ? 1920230245u : 25449u)) : v;
	v = t.y == 7. ? (t.x < 4. ? 1735749458u : (t.x < 8. ? 1936027240u : 115u)) : v;
	v = t.y == 6. ? (t.x < 4. ? 1702060354u : (t.x < 8. ? 1819231008u : 29295u)) : v;
	v = t.y == 5. ? (t.x < 4. ? 1751607628u : (t.x < 8. ? 1735289204u : 0u)) : v;
	v = t.y == 4. ? (t.x < 4. ? 1717987652u : (t.x < 8. ? 6648693u : 0u)) : v;
	v = t.y == 3. ? (t.x < 4. ? 1667592275u : (t.x < 8. ? 1918987381u : 0u)) : v;
	v = t.y == 2. ? (t.x < 4. ? 1953720644u : (t.x < 8. ? 1969383794u : 1852795252u)) : v;
	v = t.y == 1. ? (t.x < 4. ? 1936028230u : (t.x < 8. ? 7103854u : 0u)) : v;
	v = t.y == 0. ? (t.x < 4. ? 1836016967u : (t.x < 8. ? 2037544037u : 0u)) : v;
	v = t.x >= 0. && t.x < 12. ? v : 0u;

	float c = float((v >> uint(8. * t.x)) & 255u);

	vec3 textColor = vec3(.3);
	if (t.y == 10. - s.menuId)
	{
		textColor = vec3(0.74, 0.5, 0.12);
	}

	p = (p - t * scale) / scale;
	p.x = (p.x - .5) * .45 + .5;
	float sdf = TextSDF(p, c);
	if (c != 0.)
	{
		color = mix(textColor, color, smoothstep(-.05, +.05, sdf));
	}
}

// Function 411
vec4 textureBorderClamp(sampler2D sampler, vec2 uv, vec2 size, vec4 borderColor, bvec2 wrap)
{
#if 0
    // UNTESTED: to support mipmapping, we should be able to do this:
    float lod = textureQueryLod(sampler, uv).y; // OpImageQueryLod
    vec2 size = textureSize(sampler, lod); // OpImageQuerySizeLod
    vec4 ret = textureLod(sampler, uv, lod); // OpImageSampleExplicitLod
#else
    vec4 ret = texture(sampler, uv); // OpImageSampleImplicitLod
#endif
    vec2 limit = vec2(1.0, 0.5); // 0.5 for GL_CLAMP, 1.0 for GL_CLAMP_TO_BORDER
	vec2 factors = clamp(0.5 + (abs(uv - 0.5) - 0.5) * size, vec2(0.0), limit);
    if (wrap.x)
    	ret = mix(ret, borderColor, factors.x);

    if (wrap.y)
        ret = mix(ret, borderColor, factors.y);
	return ret;
}

// Function 412
float checkersTextureGradBox( in vec3 p, in vec3 ddx, in vec3 ddy )
{
    vec3 w = max(abs(ddx), abs(ddy)) + 0.01;   // filter kernel
    vec3 i = (tri(p+0.5*w)-tri(p-0.5*w))/w;    // analytical integral (box filter)
    return 0.5 - 0.5*i.x*i.y*i.z;              // xor pattern
}

// Function 413
vec3 getTexture1(vec2 p){
	vec4 s = texture(iChannel1, p);
    return s.xyz * s.w;
}

// Function 414
vec3 NearestTextureSample (vec2 P)
{
	float textureSize = iResolution.x*iResolution.y;
    vec2 pixel = P * textureSize;
    
    float onePixel = 1.0/textureSize;
    vec2 frac = fract(pixel);
    pixel = (floor(pixel) / textureSize);
    return texture(iChannel0, pixel + vec2(onePixel/2.0)).rgb;
}

// Function 415
float text_hp(vec2 U) {
    initMsg;C(72);C(80);C(58);endMsg;
}

// Function 416
vec4 textureNoTile_3weights_illegible( sampler2D samp, in vec2 uv )
{
    vec2 fuv = mod( uv, 2. ), iuv = uv - fuv;
    vec3 BL_one = vec3(0.,0.,1.);
    if( fuv.x >= 1. ) fuv.x = 2.-fuv.x, BL_one.x = 2.;
    if( fuv.y >= 1. ) fuv.y = 2.-fuv.y, BL_one.y = 2.;
    
    float w12 = smoothstep( 1.125*BLEND_WIDTH, 1.-1.125*BLEND_WIDTH, dot(fuv,vec2(.5,-.5)) + .5 );
    vec4 res = mix( texture( samp, transformUVs( iuv + BL_one.xz, uv ) ), texture( samp, transformUVs( iuv + BL_one.zy, uv ) ), w12 );

    float w3 = (fuv.x+fuv.y) - 1.; vec2 iuv3;
    if( w3 < 0. ) iuv3 = iuv + BL_one.xy, w3 = -w3;
    else iuv3 = iuv + BL_one.zz;
    w3 = smoothstep( BLEND_WIDTH, 1.-BLEND_WIDTH, w3 );
    return mix( res, texture( samp, transformUVs( iuv3, uv ) ), w3 );
}

// Function 417
vec4 GetTextureSample(vec2 pos, float freq, vec2 nodePoint)
{
    vec3 hash = hash33(vec3(nodePoint.xy, 0));
    float ang = hash.x * 2.0 * pi;
    mat2 rotation = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
    
    vec2 uv = rotation * pos * freq + hash.yz;
    return texture(iChannel0, uv);
}

// Function 418
vec3 getBallTexture(vec2 uv, vec3 color, int num) {
    uv = vec2(1.0 - uv.y, uv.x);
    uv -= vec2(0.5, 0.5);
    uv *= vec2(4.0f, 2.0f);
    uv += vec2(0.5, 0.5);
    uv = min(uv, vec2(0.97));
    uv = max(uv, vec2(0.03));
    
    int px = (num > 7 ? num - 8 : num) * 64 + int(uv.x * 64.0);
    int py = 128 + (num > 7 ? 1 : 0) * 64 + int(uv.y * 64.0);
    uv = vec2(float(px), float(py)) / iResolution.xy;
    return texture(iChannel0,uv).xyz;
}

// Function 419
vec4 texture_bilinear( const in sampler2D t, in vec2 uv )
{
  uv -= 0.5 * texelSize.xx;
  // Calculate pixels to sample and interpolating factor
  vec2 f = fract( uv * TEXTURE_SIZE );
  vec2 uvSnapped = uv - texelSize.xx * f + 0.5 * texelSize.xx;

  // As we are sampling snapped pixels need to override
  // the mip-map selection by selecting a large negative
  // bias. Otherwise at boundaries the gradient of
  // uvSnapped is large and an incorrect mip-level is used
  // leading to artifacts  
  float bias = -10.0;
  vec4 tl = texture(t, uvSnapped, bias);
  vec4 tr = texture(t, uvSnapped + texelSize, bias);
  vec4 bl = texture(t, uvSnapped + texelSize.yx, bias);
  vec4 br = texture(t, uvSnapped + texelSize.xx, bias);
    
  vec4 tA = mix( tl, tr, f.x );
  vec4 tB = mix( bl, br, f.x );
  return mix( tA, tB, f.y );
}

// Function 420
vec4 textureLowRes(vec2 uv, vec2 sz)
{
#if LOWRES
    vec2 rsz = 1.0/sz;

    vec2 pp = (uv * sz);
    vec2 tl = floor(pp);
    vec2 tr = tl + vec2(1.0,0.0);
    vec2 bl = tl + vec2(0.0,1.0);
    vec2 br = tl + vec2(1.0,1.0);
    vec2 sp = fract(pp);
    
    vec4 tlc = texture(iChannel0, tl*rsz);
    vec4 trc = texture(iChannel0, tr*rsz);
    vec4 blc = texture(iChannel0, bl*rsz);
    vec4 brc = texture(iChannel0, br*rsz);
    
    vec4 t = lerp(tlc,trc,sp.x);
    vec4 b = lerp(blc,brc,sp.x);
    
    return lerp(t,b,sp.y);
#else
    return texture(iChannel0, uv);
#endif
}

// Function 421
float FrequencyToTexture(float Frequency){
    return Frequency/440.*ATone;
}

// Function 422
void UI_SetDrawContext( inout UIContext uiContext, UIDrawContext drawContext )
{
    uiContext.drawContext = drawContext;
    
    uiContext.vPixelCanvasPos = UIDrawContext_ScreenPosToCanvasPos( drawContext, uiContext.vPixelPos );
    uiContext.bPixelInView = UIDrawContext_ScreenPosInView( drawContext, uiContext.vPixelPos );

    uiContext.vMouseCanvasPos = UIDrawContext_ScreenPosToCanvasPos( drawContext, uiContext.vMousePos );
    uiContext.bMouseInView = UIDrawContext_ScreenPosInView( drawContext, uiContext.vMousePos );
}

// Function 423
float xorTexture( in vec2 pos )
{
    float xor = 0.0;
    for( int i=0; i<8; i++ )
    {
        xor += mod( floor(pos.x)+floor(pos.y), 2.0 );

        pos *= 0.5;
        xor *= 0.5;
    }
    return xor;
}

// Function 424
int Teletext_FormatDate( int x, vec4 vDate )
{
    int day = int(floor(vDate.z));
    int month = int(floor(vDate.y));

    int daySeconds = int(floor(vDate.w));
    int seconds = daySeconds % 60;
    int minute = (daySeconds / (60)) % 60;
    int hour = (daySeconds / (60*60)) % 24;
        
    int monthText[] = int[] ( 
        _J, _a, _n, 
        _F, _e, _b, 
        _M, _a, _r, 
        _A, _p, _r, 
        _M, _a, _y,
        _J, _u, _n,
        _J, _u, _l,
        _A, _u, _g,
        _S, _e, _p,
        _O, _c, _t,
        _N, _o, _v,
        _D, _e, _c );
    
    int monthChar = x;
    if ( monthChar >=0 && monthChar < 3 )
    {
    	return monthText[ monthChar + month * 3 ];
    }
    
    if ( x == 4 )
    {
    	return  _0 + (day / 10 ); 
    }
    if ( x == 5 )
    {
    	return  _0 + (day % 10 ); 
    }
    if ( x == 6 ) 
    {
        return CTRL_ALPHANUMERIC_YELLOW;
    }
    
    if ( x == 7 )
    {
    	return  _0 + (hour / 10 ); 
    }
    if ( x == 8 )
    {
    	return  _0 + (hour % 10 ); 
    }
    if ( x == 9 )
    {
    	return _COLON; 
    }
    if ( x == 10 )
    {
    	return  _0 + (minute / 10 ); 
    }
    if ( x == 11 )
    {
    	return  _0 + (minute % 10 ); 
    }
    if ( x == 12 )
    {
        return _COLON;
    }
    if ( x == 13 )
    {
    	return  _0 + (seconds / 10 ); 
    }
    if ( x == 14 )
    {
    	return  _0 + (seconds % 10 ); 
    }
    
    return _SP;
}

// Function 425
vec4 GetScrollingTextureSample(vec2 pos, float freq, vec2 nodePoint, vec2 velocity)
{
    vec3 hash = hash33(vec3(nodePoint.xy, 0));
    float ang = hash.x * 2.0 * pi;
    mat2 rotation = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
    
    vec2 dir = normalize(velocity);
    mat2 flowMatrix = mat2(dir.x, dir.y, -dir.y, dir.x);
    mat2 flowStretch = mat2(2.0, 0.0, 0.0, 1.0);
    vec2 flowPos = flowStretch * (inverse(flowMatrix) * pos * freq + vec2(iTime, 0.0));
    vec2 uv = rotation * flowMatrix * flowPos + hash.yz;
    return texture(iChannel0, uv);
}

// Function 426
vec3 texture_wood(vec3 pos) {
    pos = quat_mul(quat(vec3(1,0,0),-0.0), pos);
    //pos.z -= 1.0;
    vec2 core = vec2(cos(pos.z), sin(pos.z))*0.1;
    pos.xy -= core;

    float r = length(pos.xy);
    float a = (TAU/2.0 + atan(pos.x,pos.y)) / TAU;

    float r_noise = noise(vec2(cos(a*TAU*2.0), sin(a*TAU*2.0)));
    r_noise += noise(vec2(10.0) + vec2(cos(a*TAU*4.0), sin(a*TAU*4.0))) * 0.5; // squigglyness
    r_noise += noise(vec2(100.0) + vec2(cos(a*TAU*8.0), sin(a*TAU*8.0))) * 0.4; // squigglyness
    r_noise += noise(vec2(1000.0) + vec2(cos(a*TAU*16.0), sin(a*TAU*16.0))) * 0.2; // squigglyness

    r_noise += noise(pos.z*0.5)*3.0; // knottyness

    r_noise *= noise(r*3.0)*5.0; // whorlyness
    r += r_noise*0.05*clamp(r,0.0,1.0); // scale and reduce at center

    vec3 col = vec3(1.0,0.65,0.35);
    //float c = 0.5 + 0.5*sin(r*100.0); // 100 rings per meter ~ 1cm rings
    float c = fract(r*5.0);
    //c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, (c-0.15)/0.85);
    c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, sqrt(clamp((c-0.15)/0.85,0.0,1.0)));
    //c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, pow(clamp((c-0.15)/0.85,0.0,1.0), 0.25));
    col = mix(col, vec3(0.4,0.1,0.0), c); // ring gradient
    col = mix(col, col*vec3(0.8, 0.5, 0.5), noise(r*20.0)); // ring-to-ring brightness

    return col;
}

// Function 427
vec4 WallOfText( float startLine, uint endLine, vec2 uv )
{
    uv.y -= iResolution.y; // we'll index negative y's from the top of the screen
    
    // convert to line-height scale
    float pixelToLineScale = screenHeightInLines/iResolution.y;
    uv *= pixelToLineScale;
    
    // maybe add pages here, pointing into the lines
    
    // find out which line we're on
    int line = int(-uv.y+startLine);
    if ( line >= int(endLine) )
    {
       	return vec4(0);
    }
    
    int lineIndex = int(lines[line]>>8U);
    int lineIndent = int(lines[line]&0xFU); //Oh ex, eff you!
    int lineColour = int((lines[line]>>4U)&0xFU);
    vec3 textColour = vec3(lineColour&1,(lineColour>>1)&1,(lineColour>>2)&1);
    
    // and which character within the line
    int charIndex = int(floor(uv.x/charWidth-float(lineIndent)*tabWidth));
    charIndex += lineIndex*4;
    
    if ( charIndex >= lineIndex*4 && charIndex < int(lines[line+1]>>8U)*4 )
    {
        uint char = text[charIndex/4];
        char = (char>>(8*(charIndex&3)))&0xffU;
        vec2 charuv = vec2( char&0xFU, 0xFU-(char>>4) );
        
        uv.x = fract(uv.x/charWidth)*charWidth;
        uv.y = fract(uv.y-startLine);
        
        vec4 t = textureLod(iChannel1,(uv+charuv)/16.+vec2(1./64.,0), 0.);
        
	    float s = textSoftness * pixelToLineScale;
        
        return vec4( textColour, smoothstep(.5+s,.5-s,t.w) );
    }
    
    return vec4(0);
}

// Function 428
vec4 draw_font8x8_string20_ivec4( int num_ch, ivec4 s, vec4 col, ivec2 pos, inout vec4 o, ivec2 iu ) {
    vec4 v = vec4( 0 ) ;
    if( num_ch < 0 ) {
        return( v ) ;
    }
    num_ch = num_ch == 0 ? 20 : min( num_ch, 20 ) ;

    ivec2 iu2 = iu - pos ;
    if( iINSIDE( iu2, ivec2(0), ivec2(num_ch*8,8) ) ) {
        int i = iu2.x >> 3,
            j = i / 5,
            sh = 6*(i % 5),
            s5 =  j<2?j<1?  s[0] : s[1]
                 :j<3?      s[2] : s[3],
            m = 0x3f << sh,
            c = (s5 & m) >> sh ;
        pos.x += i * 8 ;
        v += draw_font8x8_char( c, col, pos, o, iu ) ;
    }
    return( v ) ;
}

// Function 429
vec4 SampleFontTex0(vec2 uv)
{
	vec2 fl = floor(uv + 0.5);
	if (fl.y == 0.0) {
		int charIndex = int(fl.x + 3.0);
		int arrIndex = (charIndex / 4) % 73;
		uint wordFetch = textArr0[arrIndex];
		uint charFetch = (wordFetch >> ((charIndex % 4) * 8)) & 0x000000FFu;
		float charX = float (int (charFetch & 0x0000000Fu)     );
		float charY = float (int (charFetch & 0x000000F0u) >> 4);
		fl = vec2(charX, charY);
	}
	uv = fl + fract(uv+0.5)-0.5;
	return texture(iChannel2, (uv+0.5)*(1.0/16.0), -100.0) + vec4(0.0, 0.0, 0.0, 0.000000001) - 0.5;
}

// Function 430
vec4 textuDist(vec2 pixCoords, float A){;
    float Z =5.0; 
    vec2 uv = (pixCoords / iResolution.xy);// - vec2(0.5,0.5);
    vec2 NewCord = uv*(Z-1.0)/(Z -A);// + vec2(0.5,0.5);
    vec4 tempColor = texture(iChannel0, NewCord);
    return tempColor;
}

// Function 431
vec4 myTexture(vec2 uv) {
    
    vec2 res = iChannelResolution[0].xy;
    uv = uv*res + 0.5;
    
    // tweak fractionnal value of the texture coordinate
    vec2 fl = floor(uv);
    vec2 fr = fract(uv);
    vec2 aa = fwidth(uv)*0.75;
    fr = smoothstep( vec2(0.5)-aa, vec2(0.5)+aa, fr);
    
    uv = (fl+fr-0.5) / res;
    return texture(iChannel0, uv);
    
}

// Function 432
void setTexture( out vec4 o, in vec2 fragCoord )
{
    
 	if(fragCoord.x>8.*16. || fragCoord.y >10.*16.) discard;
    vec2 gridPos = floor((fragCoord -vec2(0.,32.))/ 16.) ;
    vec2 c = mod(fragCoord, 16.);
    int id = int(gridPos.x + gridPos.y * 8.);
 
   
    vec2 uv = floor( c );	
    float h = hash12(uv +vec2(float(id)));
    float br = 1. - h * (96./255.);		
	float xm1 = mod((uv.x * uv.x * 3. + uv.x * 81.) / 4., 4.);

    if (iFrame > 10 && iChannelResolution[0].x > 0. && id!=32  ) discard;
    o.a = 1.;
    if (id == 0) { //NO TEXTURE
    	o = vec4(1,0,1,1);
    }
    if (id == 1) { //STONE
       
        o.rgb =  vec3( 127./255., 127./255., 55./255.) *br;        
    }
    if (id == 2) { //GRASS DOWN
        
        //o.rgb =  vec3( 150./255., 108./255.,  74./255.) *br;
        o.rgb = vec3( 127./255., 86./255.,  39./255.)*br;
    }
    if (id == 3) { //GRASS LATERAL
        
         o.rgb = vec3( 127./255., 86./255.,  39./255.)*br;
        if (c.y  + hash( c.x*2.) *3.  > 14. ) 
         o.rgb = //vec3(.5,.3,.05)*br;
        vec3(.2, .22, .08)*1.4*br;
       
    }
    if (id == 4) { //GRASS UP
   		
        //o.rgb = vec3( 106./255., 157./255.,  59./255.)*br;
        o.rgb = vec3(.2, .22, .08)*1.4*br;
    }
    
    if (id == 5) { //ROCK
       
        o.rgb = vec3( 106./255., 170./255.,  64./255.)*br;
        o.rgb = vec3(clamp(pow(1. - tileableWorley(c / 16., 4.), 2.), 0.2, 0.6) + 0.2 * tileableWorley(c / 16., 5.));
 
    }
    

    if (id == 7) { //BRICK
        o.rgb = vec3( 1.1)*br; 
       
		if ( mod(uv.x + (floor(uv.y / 4.) * 5.), 8.) == 0. || mod( uv.y, 4.) == 0.) {
			
            o.rgb = vec3(.9,.9,.7)*br;
		}
        
    	//o.rgb = -0.1 * hash12(c) + mix(vec3(.6,.3,.2) + 0.1 * (1. - brickPattern(c + vec2(-1,1)) * brickPattern(c)), vec3(0.8), 1. - brickPattern(c));
    }
	/*
    if (id == 6 || id == 26) {//LIGHT OR FIREFLY
        float w = 1. - tileableWorley(c / 16., 4.);
        float l = clamp(0.7 * pow(w, 4.) + 0.5 * w, 0., 1.);
        o.rgb = mix(vec3(.3, .1, .05), vec3(1,1,.6), l);
        if (w < 0.2) o.rgb = vec3(0.3, 0.25, 0.05);
    }
	*/
    if (id == 8) {//GOLD
    	o.rgb = mix(vec3(1,1,.2), vec3(1,.8,.1), sin((c.x - c.y) / 3.) * .5 + .5);
        if (any(greaterThan(abs(c - 8.), vec2(7)))) o.rgb = vec3(1,.8,.1);
    }
	
    if (id == 9) { //ROAD
        
         o.rgb= vec3(0.8,0.7,0.2)*(.8 + 0.2 * woodPattern(c))*br;        
    }    
    if (id == 10) {//TREE
		
        if ( h < 0.5 ) {
			br = br * (1.5 - mod(uv.x, 2.));
		}
        o.rgb = vec3( 103./255., 82./255.,  49./255.)*br; 				
	}	
    if (id == 11) {//LEAF
            o.rgb=  vec3(  .12, .35,  .05 )*br;
	        //o.rgb=  vec3(  .6, .3,  .05 )*br;		//AUTUMN
	}
    if (id == 12) {//WATER		
        o.rgb=vec3(  64./255.,  64./255., 255./255.)*br;		
	}	
    if (id == 13) {//SAND
		//getMaterialColor(10,c,o.rgb);
		o.rgb= vec3(0.74,0.78,0.65);
	}
    /*
    if (id == 14) {//RED APPLE	- MIRROR	
		o.rgb= vec3(.95,0.,0.05);
       
	}
    if (id == 15) {//PINK MARBLE	
        o.rgb= vec3(.95,0.5,.5)*br;
    	//o.rgb = mix(vec3(.2,1,1), vec3(1,.8,.1), sin((c.x - c.y) / 3.) * .5 + .5);
       // if (any(greaterThan(abs(c - 8.), vec2(7)))) o.rgb = vec3(.1,.8,1);
       
	}
	
    if (id == 16) { //BEDROcK
        
    
        o.rgb =   .2*vec3( 127./255., 127./255., 127./255.) *br;   
    }
    if (id == 17) {//DIAMOND	
       
    	o.rgb = mix(vec3(.2,1,1), vec3(.1,.8,1), sin((c.x - c.y) / 3.) * .5 + .5);
       if (any(greaterThan(abs(c - 8.), vec2(7)))) o.rgb = vec3(.1,.8,1);
       
	}
    */
    if (id == 20) {//	ROOF
        o.rgb= vec3(0.94, 0.1, 0.1)*br;
       
	}   

    if (id == 21) {//	NEAR WATER
        //o.rgb= vec3(0.68, 0.66, 0.38)*br;
        o.rgb =  vec3( 127./255., 86./255.,  39./255.)*1.2*br;
       
	}

    if (id == 28) {//	
        o.rgb= vec3(0.74, 0.67, 0.64)*br;
              
	}  

	/*
    if (id == 32) { //DESTROYING BLOCK ANIMATION
    	o.rgb = vec3(crackingAnimation(c / 16., load(_pickTimer).r));
    }
    if (id == 48) { 
    	o = vec4(vec3(0.2), 0.7);
        vec2 p = c - 8.;
        float d = max(abs(p.x), abs(p.y));
        if (d > 6.) {
            o.rgb = vec3(0.7);
            o.rgb += 0.05 * hash12(c);
            o.a = 1.;
            if ((d < 7. && p.x < 6.)|| (p.x > 7. && abs(p.y) < 7.)) o.rgb -= 0.3;
        }
        o.rgb += 0.05 * hash12(c);
        
    }
	*/
    
}

// Function 433
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

// Function 434
void TextureEnvBlured(in vec3 N, in vec3 Rv, out vec3 iblDiffuse, out vec3 iblSpecular) {
    iblDiffuse = vec3(0.0);
    iblSpecular = vec3(0.0);

    vec2 sum = vec2(0.0);

    vec2 ts = vec2(textureSize(reflectTex, 0));
    float maxMipMap = log2(max(ts.x, ts.y));

    vec2 lodBias = vec2(maxMipMap - 7.0, 4.0);

    for (int i=0; i < ENV_SMPL_NUM; ++i) {
        vec3 sp = SpherePoints_GoldenAngle(float(i), float(ENV_SMPL_NUM));

        vec2 w = vec2(
            dot(sp, N ) * 0.5 + 0.5,
            dot(sp, Rv) * 0.5 + 0.5);


        w = pow(w, vec2(4.0, 32.0));

        vec3 iblD = sampleReflectionMap(sp, lodBias.x);
        vec3 iblS = sampleReflectionMap(sp, lodBias.y);

        iblDiffuse  += iblD * w.x;
        iblSpecular += iblS * w.y;

        sum += w;
    }

    iblDiffuse  /= sum.x;
    iblSpecular /= sum.y;
}

// Function 435
float textColor(vec2 from, vec2 to, vec2 p)
{
	p *= font_size;
	float inkNess = 0., nearLine, corner;
	nearLine = minimum_distance(from,to,p); // basic distance from segment, thanks http://glsl.heroku.com/e#6140.0
	inkNess += smoothstep(0., 1. , 1.- 14.*(nearLine - STROKEWIDTH)); // ugly still
	inkNess += smoothstep(0., 2.5, 1.- (nearLine  + 5. * STROKEWIDTH)); // glow
	return inkNess;
}

// Function 436
void Text(inout vec3 color, vec2 p)
{
	float glyphRatio = 2.0;
	vec2 glyphScale = 6. * vec2(1., glyphRatio);
    
    // Compute integer cell index of the text
	vec2 t = floor(p / glyphScale + 1e-6);

    // First we will pick v which contains 4 characters in the text based on the cell index
    // We do it using series of if statements which return ASCII codes for a given cell index
    // Alternatively this could be written as v = TextChars[t.x][t.y]
	uint v = 0u;
	v = t.y == 12. ? (t.x < 4. ? 1936287828u : (t.x < 8. ? 544434464u : (t.x < 12. ? 1696624225u : (t.x < 16. ? 1886216568u : (t.x < 20. ? 1746953580u : (t.x < 24. ? 1948284783u : (t.x < 28. ? 1717903471u : (t.x < 32. ? 1768122726u : (t.x < 36. ? 1819569765u : (t.x < 40. ? 1701978233u : (t.x < 44. ? 1919247470u : 0u))))))))))) : v;
	v = t.y == 11. ? (t.x < 4. ? 1735549292u : (t.x < 8. ? 1818370149u : (t.x < 12. ? 1936417647u : (t.x < 16. ? 543584032u : (t.x < 20. ? 1954047348u : (t.x < 24. ? 1700012078u : (t.x < 28. ? 1763734648u : (t.x < 32. ? 1852121203u : (t.x < 36. ? 1701080931u : (t.x < 40. ? 1935745124u : (t.x < 44. ? 1931501856u : 29797u))))))))))) : v;
	v = t.y == 10. ? (t.x < 4. ? 1965057647u : (t.x < 8. ? 1937010281u : (t.x < 12. ? 540092448u : (t.x < 16. ? 1918986339u : (t.x < 20. ? 1954112047u : (t.x < 24. ? 539828325u : (t.x < 28. ? 1751326772u : (t.x < 32. ? 796095073u : (t.x < 36. ? 1953393013u : (t.x < 40. ? 1310731817u : (t.x < 44. ? 544503909u : 0u))))))))))) : v;
	v = t.y == 9. ? (t.x < 4. ? 1701995379u : (t.x < 8. ? 1663069797u : (t.x < 12. ? 1685221231u : (t.x < 16. ? 1918967923u : (t.x < 20. ? 1868767333u : (t.x < 24. ? 1919252078u : (t.x < 28. ? 543450484u : (t.x < 32. ? 1948282740u : (t.x < 36. ? 543517801u : (t.x < 40. ? 1629512809u : (t.x < 44. ? 2122862u : 0u))))))))))) : v;
	v = t.y == 8. ? (t.x < 4. ? 1918986339u : (t.x < 8. ? 544434464u : (t.x < 12. ? 1868784996u : (t.x < 16. ? 778331492u : 0u)))) : v;
	v = t.y == 7. ? 0u : v;
	v = t.y == 6. ? (t.x < 4. ? 1853321028u : (t.x < 8. ? 1701079411u : (t.x < 12. ? 544434464u : (t.x < 16. ? 1952540788u : (t.x < 20. ? 1702257952u : (t.x < 24. ? 1730181490u : (t.x < 28. ? 1752201580u : (t.x < 32. ? 1937075488u : (t.x < 36. ? 1700929652u : (t.x < 40. ? 1701344288u : (t.x < 44. ? 1835103008u : 101u))))))))))) : v;
	v = t.y == 5. ? (t.x < 4. ? 1702521203u : (t.x < 8. ? 1969365036u : (t.x < 12. ? 1852776564u : (t.x < 16. ? 1701344288u : (t.x < 20. ? 1752461088u : (t.x < 24. ? 1746956901u : (t.x < 28. ? 543452769u : (t.x < 32. ? 1730176375u : (t.x < 36. ? 1713402981u : (t.x < 40. ? 544502625u : 0u)))))))))) : v;
	v = t.y == 4. ? (t.x < 4. ? 1886220131u : (t.x < 8. ? 1952541801u : (t.x < 12. ? 544108393u : (t.x < 16. ? 1701669236u : (t.x < 20. ? 1851859059u : (t.x < 24. ? 1919361124u : (t.x < 28. ? 544498021u : (t.x < 32. ? 1953396082u : (t.x < 36. ? 6647145u : 0u))))))))) : v;
	v = t.y == 3. ? (t.x < 4. ? 1718773104u : (t.x < 8. ? 1634562671u : (t.x < 12. ? 543515502u : (t.x < 16. ? 1852404520u : (t.x < 20. ? 543517799u : (t.x < 24. ? 544761204u : (t.x < 28. ? 1668572518u : (t.x < 32. ? 1768959848u : (t.x < 36. ? 694969720u : (t.x < 40. ? 46u : 0u)))))))))) : v;
	v = t.y == 2. ? 0u : v;
	v = t.y == 1. ? (t.x < 4. ? 1868787269u : (t.x < 8. ? 544367972u : (t.x < 12. ? 1629516649u : (t.x < 16. ? 1818845558u : (t.x < 20. ? 1701601889u : (t.x < 24. ? 544108320u : (t.x < 28. ? 1215588679u : (t.x < 32. ? 3826293u : 0u)))))))) : v;
	v = t.y == 0. ? (t.x < 4. ? 1886680168u : (t.x < 8. ? 791624307u : (t.x < 12. ? 1752459623u : (t.x < 16. ? 1663984245u : (t.x < 20. ? 1798270319u : (t.x < 24. ? 1802658158u : (t.x < 28. ? 1667856239u : (t.x < 32. ? 1750282106u : (t.x < 36. ? 1919247457u : (t.x < 40. ? 1417244532u : (t.x < 44. ? 7632997u : 0u))))))))))) : v;
	v = t.x >= 0. && t.x < 48. ? v : 0u;

    // Next we pick one of those 4 characters in the selected uint
	float char = float((v >> uint(8. * t.x)) & 255u);

    // Compute [0;1] position in the current cell
	vec2 posInCell = (p - t * glyphScale) / glyphScale;
	posInCell.x = (posInCell.x - .5) / glyphRatio + .5;

	float sdf = GlyphSDF(posInCell, char);
	if (char != 0.)
	{
		color = mix(vec3(.2), color, smoothstep(-.04, +.04, sdf));
	}
}

// Function 437
float escherTextureContour(vec2 p, float linewidth, float pixel_size)
{
    vec2 pp = mod(p,1.0);
    
    float d = 10000000.0;
    for(int i=0; i<vert.length()-1; ++i)
    {
        //*
        for(int j=0; j<textureTiles.length(); ++j)
        {
            d = min(d, PointSegDistance2(pp+textureTiles[j], vert[i], vert[i+1]));
        }
        /*/
    	d = min(d, PointSegDistance2(pp, vert[i], vert[i+1]));
    	d = min(d, PointSegDistance2(pp-vec2(1.0, 0.0), vert[i], vert[i+1]));
    	d = min(d, PointSegDistance2(pp+vec2(0.0, 1.0), vert[i], vert[i+1]));
        //*/
    }
    
    d = smoothstep(0.0, 1.0, (sqrt(d)-linewidth)/pixel_size);
    
    return d;
}

// Function 438
vec4 SampleFontTex(vec2 uv)
{
    // Do some tricks with the UVs to spell out "TexFont" in the middle.
    vec2 fl = floor(uv + 0.5);
    if (fl.y == 0.0) {
        if (fl.x == -3.0) fl = vec2(4.0, 10.0);
    	else if (fl.x == -2.0) fl = vec2(5.0, 9.0);
    	else if (fl.x == -1.0) fl = vec2(8.0, 8.0);
    	else if (fl.x == 0.0) fl = vec2(6.0, 11.0);
    	else if (fl.x == 1.0) fl = vec2(15.0, 9.0);
    	else if (fl.x == 2.0) fl = vec2(14.0, 9.0);
    	else if (fl.x == 3.0) fl = vec2(4.0, 8.0);
    }
    uv = fl + fract(uv+0.5)-0.5;

    // Sample the font texture. Make sure to not use mipmaps.
    // Add a small amount to the distance field to prevent a strange bug on some gpus. Slightly mysterious. :(
    return texture(iChannel0, (uv+0.5)*(1.0/16.0), -100.0) + vec4(0.0, 0.0, 0.0, 0.000000001);
}

// Function 439
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

// Function 440
vec3 texture_wood(vec3 pos) {
    pos = quat_mul(quat(vec3(1,0,0),-0.0), pos);
   	//pos.z -= 1.0;
    vec2 core = vec2(cos(pos.z), sin(pos.z))*0.1;
    pos.xy -= core;
    
	float r = length(pos.xy);
    float a = (TAU/2.0 + atan(pos.x,pos.y)) / TAU;
    
    float r_noise = noise(vec2(cos(a*TAU*2.0), sin(a*TAU*2.0)));
    r_noise += noise(vec2(10.0) + vec2(cos(a*TAU*4.0), sin(a*TAU*4.0))) * 0.5; // squigglyness
    r_noise += noise(vec2(100.0) + vec2(cos(a*TAU*8.0), sin(a*TAU*8.0))) * 0.4; // squigglyness
    r_noise += noise(vec2(1000.0) + vec2(cos(a*TAU*16.0), sin(a*TAU*16.0))) * 0.2; // squigglyness
    
    r_noise += noise(pos.z*0.5)*3.0; // knottyness
    
    r_noise *= noise(r*3.0)*5.0; // whorlyness
    r += r_noise*0.05*clamp(r,0.0,1.0); // scale and reduce at center
    
    vec3 col = vec3(1.0,0.8,0.35);
    //float c = 0.5 + 0.5*sin(r*100.0); // 100 rings per meter ~ 1cm rings
    float c = fract(r*5.0);
    //c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, (c-0.15)/0.85);
    c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, sqrt(clamp((c-0.15)/0.85,0.0,1.0)));
    //c = smoothstep(0.0,1.0, c/0.15) * smoothstep(1.0,0.0, pow(clamp((c-0.15)/0.85,0.0,1.0), 0.25));
    col = mix(col, vec3(0.5,0.25,0.1)*0.4, c); // ring gradient
    col = mix(col, col*0.8, noise(r*20.0)); // ring-to-ring brightness
    
    return col;
}

// Function 441
float digitTexture(in vec2 pos)
{
    return texture(iChannel1, pos).x;
}

// Function 442
float gridTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )
{
    const float N = 10.0;
    vec2 w = max(abs(ddx), abs(ddy)) + 0.01;
    vec2 a = p + 0.5*w;
    vec2 b = p - 0.5*w;           
    vec2 i = (floor(a)+min(fract(a)*N,1.0)-
              floor(b)-min(fract(b)*N,1.0))/(N*w);
    return (1.0-i.x)*(1.0-i.y);
}

// Function 443
bool glyphMask(float x, float v) {
    float f = 8.-floor(x),
          p1 = pow(10.,f-1.),
          m0 = mod(v,pow(10.,f)),
          m1 = mod(v,p1);
    return (m0-m1)/p1>.1;
}

// Function 444
float TextSDF(vec2 p, float glyph)
{
    p = abs(p.x - .5) > .5 || abs(p.y - .5) > .5 ? vec2(0.) : p;
    return 2. * (texture(iChannel3, p / 16. + fract(vec2(glyph, 15. - floor(glyph / 16.)) / 16.)).w - 127. / 255.);
}

// Function 445
vec4 glyph(vec4 data, float glyph_number, float scale, vec2 fragCoord) {
    fragCoord /= scale;
    fragCoord.x -= glyph_number * glyph_spacing.x;
    fragCoord -= vec2(8);
    
    float transition_fac = smoothstep(new_lat - .1, new_lat, time_remapped);
    float alpha = step(abs(fragCoord.x - 4.), 6.) * step(fragCoord.y, 14.) * step(transition_fac * glyph_spacing.y - 2., fragCoord.y);;
    fragCoord.y -= transition_fac * glyph_spacing.y;
    fragCoord = floor(fragCoord);
    
    float bit = fragCoord.x + fragCoord.y * 8.;
    
    float bright;
    bright =  get_bit(data.x, bit      );
    bright += get_bit(data.y, bit - 24.);
    bright += get_bit(data.z, bit - 48.);
    bright += get_bit(data.w, bit - 72.);
    bright *= 1. - step(8., fragCoord.x);
    bright *= step(0., fragCoord.x);
    
    return vec4(vec3(bright), alpha);
}

// Function 446
void SetTextPosition(vec2 p,float x,float y){  //x=line, y=column
 tp=10.0*p;tp.x=tp.x+17.-x;tp.y=tp.y-9.4+y;}

// Function 447
void SetTextPosition(out vec2 p,v0 x,v0 y){  //x=line, y=column
 //p=10.0*uv;
 p.x=p.x+17.-x;
 p.y=p.y-9.4+y;}

// Function 448
vec4 textureGood( sampler2D sam, in vec2 uv )
{
    uv = uv*1024.0 - 0.5;
    vec2 iuv = floor(uv);
    vec2 f = fract(uv);
    f = f*f*(3.0-2.0*f);
	vec4 rg1 = textureLod( sam, (iuv+ vec2(0.5,0.5))/1024.0, 0.0 );
	vec4 rg2 = textureLod( sam, (iuv+ vec2(1.5,0.5))/1024.0, 0.0 );
	vec4 rg3 = textureLod( sam, (iuv+ vec2(0.5,1.5))/1024.0, 0.0 );
	vec4 rg4 = textureLod( sam, (iuv+ vec2(1.5,1.5))/1024.0, 0.0 );
	return mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
}

// Function 449
vec4 barkTexture(vec3 p, vec3 nor)
{
    vec2 r = floor(p.xz / 5.0) * 0.02;
    float br = texture(iChannel1, r).x;
	vec3 mat = texCube(iChannel3, p*.4, nor) * vec3(.4, .3, .1*br) *br;
    mat += texCube(iChannel3, p*.53, nor)*smoothstep(0.0,.3, mat.x)*br;
   	return vec4(mat, .1);
}

// Function 450
t texture(iChannel0,p*.1,3.
void mainImage( out vec4 f, in vec2 p )
{
    vec4 q = p.xyxy/iResolution.y - .5, c=q-q;
    
    for( float s=0.; s<.1; s+=.01 )
    {
        float x = length( q.xy ), z = 1.; p.y = atan( q.x, q.y );
        
        for( int i=0; i<99; i++ )
        {
            p.x = iTime*3. + s + 1./(x+x*z);
            if( t).x > z ) break;
            z -= .01;
        }

        f = c += t*x)*z*x*.2;
    }
}

// Function 451
vec4 draw_font8x8_number_05( int n, vec4 col, ivec2 pos, inout vec4 o, ivec2 iu ) {
    vec4 v = vec4( 0 ) ;
    ivec2 iu2 = iu - pos ;
    if( iINSIDE( iu2, ivec2(0), ivec2(5*8,8) ) ) {
        int p = ( ( 5*8 - 1 ) - iu2.x ) / 8,
            d = int( round( pow( 10., float(p) ) ) ),
            c = ( n / d ) % 10 ;
        v += draw_font8x8_char( c + 1, col, pos + ivec2( (4-p) * 8, 0 ), o, iu ) ;
    }
    return( v ) ;
}

// Function 452
vec3 texturize(vec2 uv, vec3 inpColor, float dist)
{
    float falloffY = 1.0 - smoothstep4(-0.5, 0.1, 0.4, 1., uv.y);
    float falloffX = (smoothstep(left, right, uv.x)) * 0.6;
    dist -= falloffX * pow(falloffY, 0.6) * 0.09;
    

    float amt = 13. + (max(falloffX, falloffY) * 600.);

    return mix(inpColor, vec3(0.), dtoa(dist, amt));
}

// Function 453
vec3 textureNormal(vec2 uv) {
    vec3 normal = texture( iChannel1, 100.0 * uv ).rgb;
    normal.xy = 2.0 * normal.xy - 1.0;
    
    // Adjust n.z scale with mouse to show how flat normals behave
    normal.z = sqrt(iMouse.x / iResolution.x);
    return normalize( normal );
}

// Function 454
vec3 texturef( in vec2 p )
{
	float f = 0.0;
	
	vec2 q = p;

	p *= 32.0;
    f += 0.500*noise( p ); p = ma*p*2.02;
    f += 0.250*noise( p ); p = ma*p*2.03;
    f += 0.125*noise( p ); p = ma*p*2.01;
	f /= 0.875;
	
	vec3 col = 0.53 + 0.47*sin( f*4.5 + vec3(0.0,0.65,1.1) + 0.6 );
	
	col *= 0.7*clamp( 1.65*noise( 16.0*q.yx ), 0.0, 1.0 );
	
    return col;

}

// Function 455
vec3 TextureTarget(vec2 uv)
{
	return texture(iChannel1, uv).rrr;
}

// Function 456
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

// Function 457
vec3 texToVoxCoord(vec2 textelCoord, vec3 offset,int bufferId) {

    vec2 packedChunkSize= packedChunkSize_C;

	vec3 voxelCoord = offset;
    voxelCoord.xy += unswizzleChunkCoord(textelCoord / packedChunkSize);
    voxelCoord.z += mod(textelCoord.x, packedChunkSize.x) + packedChunkSize.x * mod(textelCoord.y, packedChunkSize.y);
    return voxelCoord;
}

// Function 458
float MAT_scratchTexture(vec2 p)
{
    const float squareWidth = 0.10*2.0;
    const float moveAmp   = squareWidth*0.75;
    const float lineWidth = 0.0005;
    float repeatInterval = squareWidth+moveAmp;
    repeatInfo rInfo = UTIL_repeat(p,repeatInterval);
    float margin = repeatInterval-squareWidth;
    
    vec2 a = moveAmp*noise(rInfo.anchor);
    vec2 b = -moveAmp*noise(rInfo.anchor+10.0);
    float dseg = 1000.0*UTIL_distanceToLineSeg(rInfo.pRepeated, a, b)/squareWidth;
    return saturate(10.0/dseg-0.5)*0.25;
}

// Function 459
mat3 glyph_8_9(float g) {
	if(g<65.) 
        return glyph_8_9_numerics(g);
	else if(g<97.) 
        return glyph_8_9_uppercase(g);
    else
        return glyph_8_9_lowercase(g);
}

// Function 460
vec4 gtexture(sampler2D x, vec2 xy){
    float r = sin(iTime*1.5)*sin(iTime*1.5);
    vec4 colors=texture(x,xy);
    vec4 colorg=texture(x,vec2(1,1));
    if (length(colors-colorg)<0.33 || 
        length(
            (colors/length(colors)) - 
            (colorg/length(colorg))) < .25) { 
        colors=texture(iChannel2,xy);
    } else {
       	colors = (colors*r + (1.-r)*texture(iChannel2,xy));
        colors.a=1.;
    }
    return colors;
}

// Function 461
vec4 font(int c) {
    if (c < 128) return font2(c);
    return vec4(0xffff) - font2(c - 128);
}

// Function 462
float text(vec2 pos) {
  // return texture(iChannel0,pos).r;
  return pow(max(0.,sin(8.*3.1416*pos.x)*sin(8.*3.1416*pos.y)),.8)+.3*(1.-cos(.025*6.28*t));
}

// Function 463
vec3 ScrollText2()
{
  tp = uv / FONT_SIZE2;  // set font size
  tp.x = 1.8*(tp.x -4.0 +mod(time*SCROLL_SPEED, SCROLL_LEN));
  tp.y = (uv.y + 0.88) / 0.2;  // set position & font size

  float c = 0.0;
  _ _star _ _star _ _star _ _star _ _note _note _note _note _
  _p _l _a _y _i _n _g _  _s _o _u _n _d _c _l _o _u _d _
  _m _u _s _i _c _  _note _D _i _v _i _s _i _o _n  _ _R _u _i _n _e _note _
  _f _r _o _m _  _C _a _r _p _e _n _t _e _r _  _B _r _u _t _
  _note _note _note _note _ _star _ _star _ _star _ _star
  // _1 _2 _3 _4 _5 _6 _7 _8 _9 _0
      
  vec3 fcol = c * vec3(pos, 0.5+0.5*sin(time));    
  if (c >= 0.5) return fcol; 
  return mix (aColor, fcol, c);
}

// Function 464
float checkersTexture(in vec2 p)
{
    vec2 q = floor(p);
    return mod( q.x+q.y, 2.0 );            // xor pattern
}

// Function 465
mat3 glyph_8_9_uppercase(float g) {
    // uppercase ==================================================
    GLYPH(65)0x00011000,0x00100100,0x01000010,0x01111110,0x01000010,0x01000010,0x01000010,0,0);
    GLYPH(66)0x11111000,0x01000100,0x01000100,0x01111000,0x01000100,0x01000100,0x11111000,0,0);
    GLYPH(67)0x00011100,0x00100010,0x01000000,0x01000000,0x01000000,0x00100010,0x00011100,0,0);
    GLYPH(68)0x11111000,0x01000100,0x01000010,0x01000010,0x01000010,0x01000100,0x11111000,0,0);
    GLYPH(69)0x01111110,0x01000000,0x01000000,0x01111000,0x01000000,0x01000000,0x01111110,0,0);
    GLYPH(70)0x01111110,0x01000000,0x01000000,0x01111000,0x01000000,0x01000000,0x01000000,0,0);
    GLYPH(71)0x00111100,0x01000010,0x01000000,0x01001110,0x01000010,0x01000010,0x00111100,0,0);
    GLYPH(72)0x01000010,0x01000010,0x01000010,0x01111110,0x01000010,0x01000010,0x01000010,0,0);
    GLYPH(73)0x00111000,0x00010000,0x00010000,0x00010000,0x00010000,0x00010000,0x00111000,0,0);
    GLYPH(74)0x01111110,0x00000100,0x00000100,0x00000100,0x00000100,0x01000100,0x00111000,0,0);
    GLYPH(75)0x01000100,0x01001000,0x01010000,0x01110000,0x01001000,0x01000100,0x01000010,0,0);
    GLYPH(76)0x00100000,0x00100000,0x00100000,0x00100000,0x00100000,0x00100000,0x00111111,0,0);
    GLYPH(77)0x01000010,0x01100110,0x01011010,0x01000010,0x01000010,0x01000010,0x01000010,0,0);
    GLYPH(78)0x01000010,0x01100010,0x01010010,0x01001010,0x01000110,0x01000010,0x01000010,0,0);
    GLYPH(79)0x00111100,0x01000010,0x01000010,0x01000010,0x01000010,0x01000010,0x00111100,0,0);
    GLYPH(80)0x11111100,0x01000010,0x01000010,0x01111100,0x01000000,0x01000000,0x01000000,0,0);
    GLYPH(81)0x00111100,0x01000010,0x01000010,0x01000010,0x01001010,0x01000110,0x00111101,0,0);
    GLYPH(82)0x01111100,0x01000010,0x01000010,0x01111100,0x01001000,0x01000100,0x01000010,0,0);
    GLYPH(83)0x00111100,0x01000010,0x01000000,0x00111100,0x00000010,0x01000010,0x00111100,0,0);
    GLYPH(84)0x01111100,0x00010000,0x00010000,0x00010000,0x00010000,0x00010000,0x00010000,0,0);
    GLYPH(85)0x01000010,0x01000010,0x01000010,0x01000010,0x01000010,0x01000010,0x00111100,0,0);
    GLYPH(86)0x01000010,0x01000010,0x01000010,0x01000010,0x00100100,0x00100100,0x00011000,0,0);
    GLYPH(87)0x01000011,0x01000011,0x01000011,0x01000011,0x01011011,0x01100111,0x00100100,0,0);
    GLYPH(88)0x01000100,0x01000100,0x01000100,0x00111000,0x00101000,0x01000100,0x01000100,0,0);
    GLYPH(89)0x01000100,0x01000100,0x01000100,0x00111000,0x00010000,0x00010000,0x00010000,0,0);
    GLYPH(90)0x01111110,0x00000010,0x00000100,0x00011000,0x00100000,0x01000000,0x01111110,0,0);
    return mat3(0);
}

// Function 466
float sampleFont(ivec2 glyph, ivec2 coord)
{
    ivec2 idx = coord % FNT_RES;
    return (((glyph[idx.y / 4] >> (8 * (idx.y % 4) + idx.x)) & 1) > 0) ? 1.0 : 0.0;
}

// Function 467
float Glyph8(const in vec2 uv)
{ 
    vec2 vP = uv - vec2(0.788, 0.125);
    vP /= 0.065;
    vP.x *= 1.4;
    vP.x += vP.y * 0.25;
    
    vec2 vP2 = vP;
    
    vP.y = abs(vP.y);
    vP.y = pow(vP.y, 1.2);
    float f= length(vP);
    
    vP2.x *= 1.5;
    float f2 = length(vP2 * 1.5 - vec2(0.3, 0.0));
    
    
    return max(f - 1.0, 1.0 - f2) / 20.0;
}

// Function 468
void print_glyph(inout vec4 fragColor, ivec2 pixel, int glyph, vec4 color)
{
    color *= glyph_color(uint(glyph), pixel);
    fragColor.rgb = mix(fragColor.rgb, color.rgb, color.a);
}

// Function 469
mat3 glyph_8_9_uppercase(float g) {
    // uppercase ==================================================
    GLYPH(65)00011000.,00100100.,01000010.,01111110.,01000010.,01000010.,01000010.,0,0);
    GLYPH(66)11111000.,01000100.,01000100.,01111000.,01000100.,01000100.,11111000.,0,0);
    GLYPH(67)00011100.,00100010.,01000000.,01000000.,01000000.,00100010.,00011100.,0,0);
    GLYPH(68)11111000.,01000100.,01000010.,01000010.,01000010.,01000100.,11111000.,0,0);
    GLYPH(69)01111110.,01000000.,01000000.,01111000.,01000000.,01000000.,01111110.,0,0);
    GLYPH(70)01111110.,01000000.,01000000.,01111000.,01000000.,01000000.,01000000.,0,0);
    GLYPH(71)00111100.,01000010.,01000000.,01001110.,01000010.,01000010.,00111100.,0,0);
    GLYPH(72)01000010.,01000010.,01000010.,01111110.,01000010.,01000010.,01000010.,0,0);
    GLYPH(73)00111000.,00010000.,00010000.,00010000.,00010000.,00010000.,00111000.,0,0);
    GLYPH(74)01111110.,00000100.,00000100.,00000100.,00000100.,01000100.,00111000.,0,0);
    GLYPH(75)01000100.,01001000.,01010000.,01110000.,01001000.,01000100.,01000010.,0,0);
    GLYPH(76)00100000.,00100000.,00100000.,00100000.,00100000.,00100000.,00111111.,0,0);
    GLYPH(77)01000010.,01100110.,01011010.,01000010.,01000010.,01000010.,01000010.,0,0);
    GLYPH(78)01000010.,01100010.,01010010.,01001010.,01000110.,01000010.,01000010.,0,0);
    GLYPH(79)00111100.,01000010.,01000010.,01000010.,01000010.,01000010.,00111100.,0,0);
    GLYPH(80)11111100.,01000010.,01000010.,01111100.,01000000.,01000000.,01000000.,0,0);
    GLYPH(81)00111100.,01000010.,01000010.,01000010.,01001010.,01000110.,00111101.,0,0);
    GLYPH(82)01111100.,01000010.,01000010.,01111100.,01001000.,01000100.,01000010.,0,0);
    GLYPH(83)00111100.,01000010.,01000000.,00111100.,00000010.,01000010.,00111100.,0,0);
    GLYPH(84)01111100.,00010000.,00010000.,00010000.,00010000.,00010000.,00010000.,0,0);
    GLYPH(85)01000010.,01000010.,01000010.,01000010.,01000010.,01000010.,00111100.,0,0);
    GLYPH(86)01000010.,01000010.,01000010.,01000010.,00100100.,00100100.,00011000.,0,0);
    GLYPH(87)01000001.,01000001.,01000001.,01000001.,01001001.,01010101.,00100010.,0,0);
    GLYPH(88)01000100.,01000100.,01000100.,00111000.,00101000.,01000100.,01000100.,0,0);
    GLYPH(89)01000100.,01000100.,01000100.,00111000.,00010000.,00010000.,00010000.,0,0);
    GLYPH(90)01111110.,00000010.,00000100.,00011000.,00100000.,01000000.,01111110.,0,0);
    return mat3(0);
}

// Function 470
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

// Function 471
void bake_font(inout vec4 fragColor, vec2 fragCoord)
{
#if !ALWAYS_REFRESH_TEXTURES
    if (iFrame != 0)
        return;
#endif

    ivec2 addr = ivec2(floor(fragCoord - ADDR2_RANGE_FONT.xy));
    if (any(greaterThanEqual(uvec2(addr), uvec2(ADDR2_RANGE_FONT.zw))))
        return;
    
    const int GLYPHS_PER_LINE = int(ADDR2_RANGE_FONT.z) >> 3;
    
    int glyph = (addr.y >> 3) * GLYPHS_PER_LINE + (addr.x >> 3);
    float variation = mix(.625, 1., random(fragCoord));
    fragColor = glyph_color(uint(glyph), addr, variation);
}

// Function 472
vec3 aaBoxFilteredTexture( in vec2 p, in vec2 ddx, in vec2 ddy )
{
    vec3 color;
    vec2 max_dd = max(ddx, ddy);
    vec2 dual_u = vec2(p.x, max_dd.x);
    vec2 dual_v = vec2(p.y, max_dd.y);
    float f, pattern;
    
    // background
    color = mix(bg, white, crossingBox(tranScale(dual_u, 0.5/6.75, 6.75), tranScale(dual_v, 0.5/6.75, 6.75)));
    color = mix(color, grey, crossingBox(tranScale(dual_u, 0.5/6.45, 6.45), tranScale(dual_v, 0.5/6.45, 6.45)));
    
    // centerlines, just add as patterns do not intersect due to filter
    f = 1.0-rectBox(tranScale(dual_u, 0.5/6.45, 6.45), tranScale(dual_v, 0.5/6.45, 6.45));
    pattern = f * (lineBox(tranScale(dual_u, 1.0/0.2, 0.3)) +
                   lineBox(tranScale(dual_v, 1.0/0.2, 0.3)) +
                   lineBox(tranScale(dual_u, 1.0/0.2, -0.1)) +
                   lineBox(tranScale(dual_v, 1.0/0.2, -0.1)) );
	// dashed, just add as patterns do not intersect due to filter
    f = 1.0-rectBox(tranScale(dual_u, 0.5/10.7, 10.7), tranScale(dual_v, 0.5/10.7, 10.7));
    pattern += f * (lineBox(tranScale(dual_u, 1.0/0.15, 3.45))*dashBox(tranScale(dual_v, 1.0/2.0, 1.0), 0.6) +
                    lineBox(tranScale(dual_v, 1.0/0.15, 3.45))*dashBox(tranScale(dual_u, 1.0/2.0, 1.0), 0.6) +
                    lineBox(tranScale(dual_u, 1.0/0.15, -3.3))*dashBox(tranScale(dual_v, 1.0/2.0, 1.0), 0.6) +
                    lineBox(tranScale(dual_v, 1.0/0.15, -3.3))*dashBox(tranScale(dual_u, 1.0/2.0, 1.0), 0.6));
    // stop lines, add again
    pattern += rectBox(tranScale(dual_u, 1.0/0.4, -9.55), tranScale(dual_v, 1.0/5.55, -0.6)) +
               rectBox(tranScale(dual_u, 1.0/0.4,  10.05), tranScale(dual_v, 1.0/5.55, -0.6)) +
               rectBox(tranScale(dual_u, 1.0/0.4, -9.55), tranScale(dual_v, 1.0/5.55, 6.15)) +
               rectBox(tranScale(dual_u, 1.0/0.4,  10.05), tranScale(dual_v, 1.0/5.55, 6.15)) +
               rectBox(tranScale(dual_v, 1.0/0.4, -9.55), tranScale(dual_u, 1.0/5.55, -0.6)) +
               rectBox(tranScale(dual_v, 1.0/0.4,  10.05), tranScale(dual_u, 1.0/5.55, -0.6)) +
               rectBox(tranScale(dual_v, 1.0/0.4, -9.55), tranScale(dual_u, 1.0/5.55, 6.15)) +
               rectBox(tranScale(dual_v, 1.0/0.4,  10.05), tranScale(dual_u, 1.0/5.55, 6.15));
    // pedestrian crossing, add again
    pattern += rectBox(tranScale(dual_v, 1.0/2.0,  9.05), tranScale(dual_u, 1.0/5.55, 6.15))*dashBox(tranScale(dual_u, 1.0/0.6, 5.925), 0.5) +
               rectBox(tranScale(dual_v, 1.0/2.0,  9.05), tranScale(dual_u, 1.0/5.55, -0.6))*dashBox(tranScale(dual_u, 1.0/0.6, -0.825), 0.5) +
               rectBox(tranScale(dual_v, 1.0/2.0,  -7.05), tranScale(dual_u, 1.0/5.55, 6.15))*dashBox(tranScale(dual_u, 1.0/0.6, 5.925), 0.5) +
               rectBox(tranScale(dual_v, 1.0/2.0,  -7.05), tranScale(dual_u, 1.0/5.55, -0.6))*dashBox(tranScale(dual_u, 1.0/0.6, -0.825), 0.5) +
               rectBox(tranScale(dual_u, 1.0/2.0,  9.05), tranScale(dual_v, 1.0/5.55, 6.15))*dashBox(tranScale(dual_v, 1.0/0.6, 5.925), 0.5) +
               rectBox(tranScale(dual_u, 1.0/2.0,  9.05), tranScale(dual_v, 1.0/5.55, -0.6))*dashBox(tranScale(dual_v, 1.0/0.6, -0.825), 0.5) +
               rectBox(tranScale(dual_u, 1.0/2.0,  -7.05), tranScale(dual_v, 1.0/5.55, 6.15))*dashBox(tranScale(dual_v, 1.0/0.6, 5.925), 0.5) +
               rectBox(tranScale(dual_u, 1.0/2.0,  -7.05), tranScale(dual_v, 1.0/5.55, -0.6))*dashBox(tranScale(dual_v, 1.0/0.6, -0.825), 0.5);
    color = mix(color, white, pattern);

    return color;
}

// Function 473
float PxText (vec2 p, int txt)
{
  float s;
  s = 0.;
  if (txt == 0) {
    PxBgn (- vec2 (27., -1.));
    _S _o _f _t _w _a _r _e _spc _F _a _i _l _u _r _e _per _spc _P _r _e _s _s _spc 
    _l _e _f _t _spc _m _o _u _s _e _spc _b _u _t _t _o _n _spc _t _o _spc 
    _c _o _n _t _i _n _u _e _per
    PxBgn (- vec2 (17., 1.));
    _G _u _r _u _spc _M _e _d _i _t _a _t _i _o _n _spc _hsh
    _8 _2 _0 _1 _0 _0 _0 _3 _per _D _E _A _D _B _E _E _F
    PxBgn (- vec2 (7., 5.));
    _B _e _w _a _r _e _exc _spc _C _O _R _O _N _A _exc _exc 
  } else if (txt == 1) {
    PxBgn (- vec2 (2., 0.));
    _A _M _I _G _A
  } else if (txt == 2) {
    PxBgn (- vec2 (4., 0.));
    _C _o _m _m _o _d _o _r _e
   }
  return s;
}

// Function 474
vec4 texture_bicubic(sampler2D tex, vec2 uv, vec4 texelSize)
{
	uv = uv*texelSize.zw + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );

    float g0x = g0(fuv.x);
    float g1x = g1(fuv.x);
    float h0x = h0(fuv.x);
    float h1x = h1(fuv.x);
    float h0y = h0(fuv.y);
    float h1y = h1(fuv.y);

	vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - 0.5) * texelSize.xy;
	vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - 0.5) * texelSize.xy;
	vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - 0.5) * texelSize.xy;
	vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - 0.5) * texelSize.xy;
	
    return g0(fuv.y) * (g0x * texture(tex, p0)  +
                        g1x * texture(tex, p1)) +
           g1(fuv.y) * (g0x * texture(tex, p2)  +
                        g1x * texture(tex, p3));
}

// Function 475
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

// Function 476
float TextSDF(vec2 p, float glyph)
{
	p = abs(p.x - .5) > .5 || abs(p.y - .5) > .5 ? vec2(0.) : p;
	return 2. * (texture(iChannel3, p / 16. + fract(vec2(glyph, 15. - floor(glyph / 16.)) / 16.)).w - 127. / 255.);
}

// Function 477
void TeletextState_SetAlphanumericColor( inout TeletextState state, int color )
{
    state.iFgCol = color;
    state.bGfx = false;
    state.bConceal = false;
}

// Function 478
void WriteText1()
{
  SetTextPosition(1.,1.);
  float c = 0.0;
  //_star _ _V _i _e _w _ _S _h _a _d _e _r   
  //_ _D _a _t _a _ _2 _ _ _v _1 _dot _1 _ _star 
  vColor += c * headColor;
}

// Function 479
vec4 AntiAliasPointSampleTexture_Smoothstep(vec2 uv, vec2 texsize) {	
	vec2 w=fwidth(uv);
	return texture(iChannel0, (floor(uv)+0.5+smoothstep(0.5-w,0.5+w,fract(uv))) / texsize, -99999.0);	
}

// Function 480
float NumFont_BinChar( vec2 vTexCoord, float fDigit )
{
    vTexCoord.y = 15. - vTexCoord.y;
    vTexCoord = floor(vTexCoord);
    if ( fDigit == 0.0 )
    {
		return NumFont_Zero( vTexCoord );
    }
    else
    if ( fDigit == 1.0 )
    {
		return NumFont_One( vTexCoord );
    }
    else
    if ( fDigit == 2.0 )
    {
		return NumFont_Two( vTexCoord );
    }
    else
    if ( fDigit == 3.0 )
    {
		return NumFont_Three( vTexCoord );
    }
    else
    if ( fDigit == 4.0 )
    {
		return NumFont_Four( vTexCoord );
    }
    else
    if ( fDigit == 5.0 )
    {
		return NumFont_Five( vTexCoord );
    }
    else
    if ( fDigit == 6.0 )
    {
		return NumFont_Six( vTexCoord );
    }
    else
    if ( fDigit == 7.0 )
    {
		return NumFont_Seven( vTexCoord );
    }
    else
    if ( fDigit == 8.0 )
    {
		return NumFont_Eight( vTexCoord );
    }
    else
    if ( fDigit == 9.0 )
    {
		return NumFont_Nine( vTexCoord );
    }
    else
    if ( fDigit == 10.0 )
    {
		return NumFont_Percent( vTexCoord );
    }
        
    return 0.0;
}

// Function 481
vec4 traceText(vec3 ray_start, vec3 ray_dir)
{
   float ray_len = 0.0;
   vec3 p = ray_start;
   float c = 0.;
   #ifdef TRACE
   for(int i=0; i<iterationsMarch; ++i) {
   	  float dist = text(p)-.1;
      if (dist < dist_epsMarch) return vec4(p, c);
      if (p.z < ray_max) return vec4(p, -c);
      p += dist*ray_dir;
      ray_len += dist;
      float f = pow(smoothstep(2.2+sin(p.x*3.3-iTime+p.y*.5)*.2, -0., dist), 8.);
      c = max(c, f*9.);
   }
   #else
   for(int i=0; i<iterationsTraceText; ++i) {
   	  float dist = text(p);
      if (dist < dist_epsTrace) return vec4(p, c);
      if (p.z < ray_max) return vec4(p, -c);
      p += traceStep*ray_dir;
      ray_len += traceStep;
      float f = pow(smoothstep(2.2+sin(p.x*3.3-iTime+p.y*.5)*.2, -0., dist), 8.);
      c = max(c, f*9.);
   }
   #endif
   return vec4(p, -c);
}

// Function 482
vec3 sampleTexture( in vec3 uvw, in vec3 nor, in float mid )
{
    return mytexture( uvw, nor, mid );
}

// Function 483
void textureSolid(vec2 block, vec3 position, inout vec3 colour, inout vec3 normal) {
    float concrete = getConcrete(position, normal, true);
    colour = hash33(block.xyx) * vec3(0.25,0.1,0.2) + 0.5;
    colour = clamp(colour,vec3(0.0),vec3(1.0));
    colour *= concrete;
}

// Function 484
float noiseText(in vec3 x) {
    vec3 p = floor(x), f = fract(x);
	f = f*f*(3.-f-f);
	vec2 uv = (p.xy+vec2(37.,17.)*p.z) + f.xy,
	     rg = textureLod( iChannel0, (uv+.5)/256., -100.).yx;
	return mix(rg.x, rg.y, f.z);
}

// Function 485
void glyphMaskY(inout float b, float hmm, float y, float l, float x, float v) {

    float a = abs(y-l);
    
    if(a<.5+hmm) {
        
        float f = 8.-floor(x+hmm),
              p1 = pow(10.,f-1.),
              m0 = mod(v,pow(10.,f)),
              m1 = mod(v,p1);
        b += pow(10.,y)*((m0-m1)/p1>.5 ? 1. : 0.);
    }
}

// Function 486
vec3 textureGround(vec2 uv) {
    const vec2 RES = vec2(16.0, 9.0);    
    float n = hash(floor((uv * RES)));
    n = n * 0.2 + 0.5;
    // return vec3(n*0.4,n*0.2,n*0.8);
    return vec3(n*0.4,n*0.2,n*n); // make blue quadratic
}

// Function 487
vec4 texture_denoise(in sampler2D tex, vec2 uv, float threshold, vec3 q)
{
    vec4 col = texture(tex, uv),
        blurred = texture_blurred_quantized(tex, uv, q);
    
    if (length(col-blurred) <= threshold)
        return blurred;
    else
        return col;
}

// Function 488
float glyph_dist(in vec2 pt) {
	float angle = atan(pt.y, pt.x)- spin0* 2.0* PI;
	return glyph_dist2(pt, angle);
}

// Function 489
float drawText( inout vec4 c, vec2 p, vec2 tp, float s, ivec4 text, vec4 color) { 
    vec4 m = mapText( p, tp, 10./s, text);
    c = mix(c, color,  clamp(m.x, 0.,1.));
    return m.w;
}

// Function 490
vec3 getTexture(vec3 p0, vec3 rd, inout vec3 n, inout vec2 spe, float t,
                sampler2D channel1, sampler2D channel2, sampler2D channel3){ 
    float h = Terrain(p0.xz*.3);
    float elev = p0.y - .002*h;
    
    spe = vec2(0.,1000.);
    
	vec3 p = p0;
   	malaxSpace(p);

    p.x = abs(p.x);
  
    // Texture scale factor.        
    const float tSize1 = 1.; //.5;//1./6.;
	
    // puit
    vec3 pp = p;
    pp.x = abs(pp.x) +.1;// gothique  
  //  pMirror(pp.x, -.1);  
    float rp = length(pp.xz-vec2(3.,3.1));
    
    // arbre
    p.z += .05;
   
    vec3 ph = p;
   
    //vec3 p2 = p, ph = p;

    // Chemin de ronde
    ph.z -= .5;
	pR45(ph.zx);
    ph.z -= 4.6;
    ph.x += 1.;
    pReflect(ph, normalize(vec3(-1.,0,.7)),1.);
    
    vec3 pm = ph;
    pMirrorOctant(pm.xz, vec2(1.5,1.6));

    float ra = length(ph.xz);
    
    int id = rp < .202 ? ID_PUIT :
        //rp<3.1 ? ID_HOUSE_WALL :
        elev<.002 || length(p.xz)>10. ? (abs(p.z+1.9) < 1.9 && abs(p.x) < 2.3 ? ID_GROUND_CASTLE : ID_GROUND) :// sol
        p.y>2.7 ? ID_SMALL_TOUR : // toit tour conique 
        abs(p.z+2.) < 1.55 && abs(p.x) < 2. ? ID_CASTLE :  // chateau
        (length(p.xz-vec2(0,2.)) > 5.83 || (rp>3. && p.z<0.6)) ? ID_STONE :  // rempart 
        //abs(p.x) > 1.8 ? p.y < 2.5 ? vec3(.4,.4,.1) : vec3(.5,.9,.7) : //arbres
        ra < .5 ? (ra < .051 && p.y<.7 ? ID_TREE_1 : ID_TREE_2) :
        p.y < .325 ? ID_HOUSE_WALL : // mur maisonettes   
        ID_HOUSE_ROOF;  // toit maisonettes

    
    vec3 c = vec3(1);
    
    switch(id) {
        case ID_TREE_1 : 
        	n = doBumpMap(channel1, p0.xyz*vec3(1.,.1,1.)*tSize1, n, .07/(1. + t/MAX_DIST));
        	c = vec3(.4,.3,.2); break;
        case ID_TREE_2 :
        	n = doBumpMap(channel1, p0*4.*tSize1, n, .07/(1. + t/MAX_DIST)); 
        	c = vec3(.2,.5,.4); break;
        case ID_PUIT : 
        	n = doBumpMap(channel2, p0*1.95*tSize1, n, .007/(1. + t/MAX_DIST)); 
        	n = doBumpMapBrick(p*30., n, .015); c = .5*vec3(1.,.9,.7); break;
        case ID_GROUND :
        
        
            n = doBumpMap(channel1, p0*tSize1, n, .007/(1. + t/MAX_DIST));//max(1.-length(fwidth(sn)), .001)*hash(sp)/(1.+t/FAR)
			c = NoiseT(1000.*p0.xz)*mix(vec3(.7,.7,.6), vec3(.3,.5,.4), smoothstep(.0,.05, abs(abs(p.x*1.2+.05)-.1)));
       	// test
        	break;
        
        case ID_GROUND_CASTLE :  
        	n = doBumpMapBrick(p0*5., n, .005); 
        	c = vec3(.8,.8,.7); break;
        case ID_SMALL_TOUR : 
        	c = vec3(1.,.7,1); break;
        case ID_CASTLE : 
        	n = doBumpMap(channel3, p0*4.*tSize1, n, .007/(1. + t/MAX_DIST));
        	
        //	c = vec3(.95,.9,.85), smoothstep(0.,.1, sin(10.*p.y))); 
        	c = mix(vec3(1.), vec3(.95,.9,.85), smoothstep(0.,.1, sin(15.*p.y))); 
        	break;
        case ID_STONE : 
        	spe = vec2(.5,99.); 
        	n = doBumpMapBrick(p*8., n, .03);
        	n = doBumpMap(channel1, p0*1.5*tSize1, n, .01/(1. + t/MAX_DIST));
        	c = .5*vec3(1.,.85,.7); break;
        case ID_HOUSE_WALL :
        	//if (length(pm.xz)-.2
        	//n = doBumpMapBrick(p*15., n, .03); 
        	//c = vec3(1.,.9,.7);
            if (abs(pm.x-.0335) <.06 && abs(pm.z+.8) <.2 && pm.y<.285) {
                // porte
                n = doBumpMapBrick(vec3(.3, pm.x+.13, pm.z)*32., n, .03); 
                n = doBumpMap(channel1, 3.*pm.yxz*tSize1, n, .02/(1. + t/MAX_DIST));
                c = .6*vec3(0.,.6,1); 
            } else {	
	        	n = doBumpMap(channel2, p0*1.95*tSize1, n, .007/(1. + t/MAX_DIST)); 
                c = vec3(1.,.95,.9);
            }
                c = c * mix(.4*vec3(.2,.6,.7), vec3(1), 
                      1.-.5*smoothstep(.3,.05, p0.y)*smoothstep(.3, .6,texture(channel2, p0.xy*4.*tSize1).x));
            break;
        case ID_HOUSE_ROOF :
        	spe = vec2(1.,9.); 
        	//n = doBumpMapBrick((p-vec3(0.,.01,0.))*30., n, .03); 
        	n = doBumpMap(channel3, p0*tSize1, n, .025/(1. + t/MAX_DIST));
        	c = vec3(.55,.32,.2) * mix(vec3(1), .7*vec3(.2,.6,.7), 
                  .5*smoothstep(.2,.9,texture(channel2, p0.xy*4.*tSize1).x));
                   //  tex3D(channel2, p0*4.*tSize1, n).x));
        	break;        	
    }
    
    	// prevent normals pointing away from camera (caused by precision errors)
	n = normalize(n - max(.0, dot (n,rd))*rd);
    
    return c;
}

// Function 491
void RenderFont( in PrintState state, in LayoutStyle style, in RenderStyle renderStyle, inout vec3 color )
{
#ifdef FONT_EFFECTS            
    if ( style.bShadow )
    {
        float fSize = renderStyle.fFontWeight + renderStyle.fOutlineWeight;
        float fBlendShadow = clamp( (state.fShadowDistance - fSize - renderStyle.fShadowSpread * 0.5) / -renderStyle.fShadowSpread, 0.0, 1.0);
        color.rgb = mix( color.rgb, vec3(0.0), fBlendShadow * renderStyle.fShadowStrength);    
    }

    if ( renderStyle.fOutlineWeight > 0.0f )
    {        
        float fBlendOutline = GetFontBlend( state, style, renderStyle.fFontWeight + renderStyle.fOutlineWeight );
        color.rgb = mix( color.rgb, renderStyle.vOutlineColor, fBlendOutline);
    }
#endif
    
    float f = GetFontBlend( state, style, renderStyle.fFontWeight );

    vec3 vCol = renderStyle.vFontColor;
	
#ifdef FONT_EFFECTS            
    if ( renderStyle.fBevelWeight > 0.0f )
    {    
        float fBlendBevel = GetFontBlend( state, style, renderStyle.fFontWeight - renderStyle.fBevelWeight );    
        float NdotL = dot( state.vNormal, normalize(renderStyle.vLightDir ) );
        float shadow = 1.0 - clamp(-NdotL, 0.0, 1.0f);
        float highlight = clamp(NdotL, 0.0, 1.0f);
        highlight = pow( highlight, 10.0f);
        vCol = mix( vCol, vCol * shadow + renderStyle.vHighlightColor * highlight, 1.0 - fBlendBevel);
    }
#endif
    
    color.rgb = mix( color.rgb, vCol, f);    
}

// Function 492
vec4 textureFade(sampler2D tex, vec2 uv, vec2 fadeWidth) {
    vec2 offsetuv = uv*vec2(1.0-fadeWidth.x, 1.0 - fadeWidth.y);
    
    vec2 scaling = 1.0 - fadeWidth;
    float hBlend = clamp((uv.y-scaling.y)/fadeWidth.y,0.0,1.0); 
    float vBlend = clamp((uv.x-scaling.x)/fadeWidth.x, 0.0, 1.0);
    
    float q1Blend = hBlend * (1.0-vBlend);
    vec2 q1Sample;
    q1Sample.x = fract(offsetuv.x + fadeWidth.x);
    q1Sample.y = fract(offsetuv.y + (fadeWidth.y * 2.0));
    vec4 tex1 = texture(tex, q1Sample); 
    vec4 q1Col = q1Blend * tex1;

    float q2Blend = hBlend * vBlend;
    vec2 q2Sample;
    q2Sample.x = fract(offsetuv.x + (fadeWidth.x * 2.0));
    q2Sample.y = fract(offsetuv.y + (fadeWidth.y * 2.0));
    vec4 tex2 = texture(tex, q2Sample);
    vec4 q2Col = q2Blend * tex2;
 
    float q3Blend = (1.0-hBlend) * (1.0 - vBlend);
    vec2 q3Sample;
    q3Sample.x = fract(offsetuv.x + fadeWidth.x);
    q3Sample.y = fract(offsetuv.y + fadeWidth.y);
    vec4 tex3 = texture(tex, q3Sample);
	vec4 q3Col = q3Blend * tex3;
    
    float q4Blend = (1.0-hBlend) * vBlend;
    vec2 q4Sample;
    q4Sample.x = fract(offsetuv.x + (fadeWidth.x * 2.0));
    q4Sample.y = fract(offsetuv.y + fadeWidth.y);
    vec4 tex4 = texture(tex, q4Sample);
	vec4 q4Col = q4Blend * tex4;
    
    return q1Col + q2Col + q3Col + q4Col;

}

// Function 493
float Glyph6(const in vec2 uv)
{
    const vec2  vP0 = vec2 ( 0.638 , 0.087 ); 
    const vec2  vP1 = vec2 ( 0.648 , 0.073 ); 
    const vec2  vP2 = vec2 ( 0.673 , 0.068 ); 
    const vec2  vP3 = vec2 ( 0.692 , 0.069 ); 
    const vec2  vP4 = vec2 ( 0.687 , 0.086 ); 
    const vec2  vP5 = vec2 ( 0.688 , 0.104 ); 
    const vec2  vP6 = vec2 ( 0.672 , 0.102 ); 
    const vec2  vP7 = vec2 ( 0.659 , 0.099 ); 
    const vec2  vP8 = vec2 ( 0.663 , 0.092 ); 
    const vec2  vP9 = vec2 ( 0.662 , 0.086 ); 
    const vec2 vP10 = vec2 ( 0.655 , 0.086 ); 
    const vec2 vP11 = vec2 ( 0.644 , 0.087 ); 
    const vec2 vP12 = vec2 ( 0.637 , 0.102 ); 
    const vec2 vP13 = vec2 ( 0.638 , 0.094 ); 

    float fDist = 1.0;
    fDist = min( fDist, InCurve2(vP0,vP1,vP2, uv) ); 
    fDist = min( fDist, InCurve2(vP2,vP3,vP4, uv) ); 
    fDist = min( fDist, InCurve2(vP4,vP5,vP6, uv) ); 
    fDist = min( fDist, InCurve2(vP6,vP7,vP8, uv) ); 
    fDist = min( fDist, InCurve(vP8,vP9,vP10, uv) ); 
    fDist = min( fDist, InCurve(vP10,vP11,vP12, uv) );

    fDist = min( fDist, InQuad(vP2, vP4, vP6, vP8, uv) );
    fDist = min( fDist, InTri(vP9, vP2, vP8, uv) );
    fDist = min( fDist, InTri(vP10, vP2, vP9, uv) );
    fDist = min( fDist, InQuad(vP0, vP2, vP10, vP11, uv) );
    fDist = min( fDist, InTri(vP11, vP12, vP0, uv) );
    
    return fDist;
}

// Function 494
int GetGlyphPixel(ivec2 pos, int g)
{
	if (pos.x >= glyphSize || pos.y >= glyphSize)
		return 0;
    
    // get if bit is on for this pixel in the glyph
    // 0x01110, 0x01110, 
	// 0x11011, 0x11110,
	// 0x11011, 0x01110, 
	// 0x11011, 0x01110,
	// 0x01110, 0x11111
	//  0        1
    
    if (g == 0)
    {
     	if (pos.x > 0 && pos.x < 4 && (pos.y == 0 || pos.y == 4))
            return 1;
     	if (pos.y > 0 && pos.y < 4 && pos.x != 2)
            return 1;
  	    return 0;
    }
    else
    {
        if (pos.x == 0 && (pos.y == 4 || pos.y == 2 || pos.y == 1))
            return 0;
        if (pos.x == 4 && pos.y > 0)
            return 0;
        return 1;
    }
    
    return 0;
}

// Function 495
vec3 TextureRock(in vec3 p)
{
  return mix(vec3(0.92,0.91,0.90),vec3(0.74,0.72,0.72),Fbm(p/50.0));   
}

// Function 496
vec4 text()
{
  float s = 350.;
    vec2 uv = floor(_uv*s);
    vec3 c = vec3(0);

    print_pos = vec2(.1,.9)*s;
    
    float d = 0.0;
    
    TEXT_MODE = INVERT;
    d += char(ch_spc,uv);
    
    if (_t == 0.)
    {
        d += char(ch_s,uv);
        d += char(ch_c,uv);
        d += char(ch_u,uv);
        d += char(ch_l,uv);
        d += char(ch_p,uv);
        d += char(ch_t,uv);
    }
    else if (_t == 1.)
    {
        d += char(ch_s,uv);
        d += char(ch_m,uv);
        d += char(ch_o,uv);
        d += char(ch_o,uv);_key_r = keyInfo(82.,1.);
        d += char(ch_t,uv);
        d += char(ch_h,uv);
        d += char(ch_e,uv);
        d += char(ch_n,uv);
    }
    else if (_t == 2.)
    {
        d += char(ch_s,uv);
        d += char(ch_t,uv);
        d += char(ch_r,uv);
        d += char(ch_e,uv);
        d += char(ch_t,uv);
        d += char(ch_c,uv);
        d += char(ch_h,uv);
    }
    else if (_t == 3.)
    {
        d += char(ch_p,uv);
        d += char(ch_a,uv);
        d += char(ch_i,uv);
        d += char(ch_n,uv);
        d += char(ch_t,uv);
    }
    
    d += char(ch_spc,uv);
    TEXT_MODE = NORMAL;
    
    if (d > 0.) return vec4(vec3(.5),d); d = 0.;
    
  s = 480.;
    uv = floor(_uv*s+.5);

    float x = .22;
    float y = .845;
    
    print_pos = vec2(x,y)*s;

    d += char(ch_s,uv);
    d += char(ch_i,uv);
    d += char(ch_z,uv);
    d += char(ch_e,uv);
    
    print_pos = vec2(x,y-STRHEIGHT(1.)/s*2.+.01)*s;
        
    d += char(ch_b,uv);
    d += char(ch_l,uv);
    d += char(ch_u,uv);
    d += char(ch_r,uv);
    
    print_pos = vec2(x,y-STRHEIGHT(2.)/s*2.+.02)*s;
    
    if (_t == 3.)
    {
        d += char(ch_a,uv);
        d += char(ch_l,uv);
        d += char(ch_p,uv);
        d += char(ch_h,uv);
        d += char(ch_a,uv);
    }
    else
    {
        d += char(ch_s,uv);
        d += char(ch_t,uv);
        d += char(ch_r,uv);
        d += char(ch_e,uv);
        d += char(ch_n,uv); 
        d += char(ch_g,uv); 
        d += char(ch_t,uv); 
        d += char(ch_h,uv); 
    }
    
    if (d > 0.) return vec4(vec3(.5),d); d = 0.;
    
    x = .1;
    y = .57;

  float key_h = keyInfo(72.,1.);
    
    if (key_h < 1.)
    {
        print_pos = vec2(x,y)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_1,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_s,uv);        
        d += char(ch_c,uv);
        d += char(ch_u,uv);
        d += char(ch_l,uv);
        d += char(ch_p,uv);
        d += char(ch_t,uv);

        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_2,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_s,uv);        
        d += char(ch_m,uv);
        d += char(ch_o,uv);
        d += char(ch_o,uv);
        d += char(ch_t,uv);
        d += char(ch_h,uv);
        d += char(ch_e,uv);
        d += char(ch_n,uv);

        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_3,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_s,uv);        
        d += char(ch_t,uv);
        d += char(ch_r,uv);
        d += char(ch_e,uv);
        d += char(ch_t,uv);
        d += char(ch_c,uv);
        d += char(ch_h,uv);

        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_4,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_p,uv);        
        d += char(ch_a,uv);
        d += char(ch_i,uv);
        d += char(ch_n,uv);
        d += char(ch_t,uv);

        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_spb,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_r,uv);        
        d += char(ch_o,uv);
        d += char(ch_t,uv);
        d += char(ch_a,uv);
        d += char(ch_t,uv);
        d += char(ch_e,uv);
        
        y -= .049;
        
        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_7,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_r,uv);        
        d += char(ch_a,uv);
        d += char(ch_n,uv);
        d += char(ch_d,uv);
        d += char(ch_per,uv);
        d += char(ch_spc,uv);
        d += char(ch_s,uv);
        d += char(ch_h,uv);
        d += char(ch_a,uv);
        d += char(ch_p,uv);
        d += char(ch_e,uv);
        d += char(ch_spc,uv);
        d += char(ch_lbr,uv);
        d += char(ch_y,uv);
        d += char(ch_rbr,uv);
        
        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_8,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_r,uv);        
        d += char(ch_a,uv);
        d += char(ch_n,uv);
        d += char(ch_d,uv);
        d += char(ch_per,uv);
        d += char(ch_spc,uv);
        d += char(ch_s,uv);
        d += char(ch_h,uv);
        d += char(ch_a,uv);
        d += char(ch_p,uv);
        d += char(ch_e,uv);
        d += char(ch_spc,uv);
        d += char(ch_lbr,uv);
        d += char(ch_x,uv);
        d += char(ch_com,uv);
        d += char(ch_y,uv);
        d += char(ch_rbr,uv);
        
        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_9,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_r,uv);        
        d += char(ch_a,uv);
        d += char(ch_n,uv);
        d += char(ch_d,uv);
        d += char(ch_per,uv);
        d += char(ch_spc,uv);
        d += char(ch_p,uv);
        d += char(ch_a,uv);
        d += char(ch_i,uv);
        d += char(ch_n,uv);
        d += char(ch_t,uv);
        
        y -= .049;

        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_C,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_c,uv);        
        d += char(ch_l,uv);
        d += char(ch_e,uv);
        d += char(ch_a,uv);
        d += char(ch_r,uv);
        d += char(ch_spc,uv);
        d += char(ch_p,uv);
        d += char(ch_a,uv);
        d += char(ch_i,uv);
        d += char(ch_n,uv);
        d += char(ch_t,uv);

        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_X,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_r,uv);        
        d += char(ch_e,uv);
        d += char(ch_s,uv);
        d += char(ch_e,uv);
        d += char(ch_t,uv);
        d += char(ch_spc,uv);
        d += char(ch_s,uv);
        d += char(ch_h,uv);
        d += char(ch_a,uv);
        d += char(ch_p,uv);
        d += char(ch_e,uv);
        
        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_R,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_t,uv);        
        d += char(ch_o,uv);
        d += char(ch_g,uv);
        d += char(ch_g,uv);
        d += char(ch_l,uv);
        d += char(ch_e,uv);
        d += char(ch_spc,uv);
        d += char(ch_v,uv);
        d += char(ch_i,uv);
        d += char(ch_e,uv);
        d += char(ch_w,uv);

        print_pos = vec2(x,y-=STRHEIGHT(1.)/s+.01)*s;

        TEXT_MODE = INVERT;

        d += char(ch_spc,uv);
        d += char(ch_H,uv);
        d += char(ch_spc,uv);

        TEXT_MODE = NORMAL;

        d += char(ch_spc,uv);
        d += char(ch_t,uv);        
        d += char(ch_o,uv);
        d += char(ch_g,uv);
        d += char(ch_g,uv);
        d += char(ch_l,uv);
        d += char(ch_e,uv);
        d += char(ch_spc,uv);
        d += char(ch_h,uv);
        d += char(ch_e,uv);
        d += char(ch_l,uv);
        d += char(ch_p,uv);
  }
    
    return vec4(vec3(.25),d);
}

// Function 497
vec3 TextureNoise(vec2 uvs)
{
    return textureLod(iChannel3, uvs, 0.0).rgb;
}

// Function 498
vec4 texture3DLinear(sampler2D tex, vec3 uvw, vec3 vres)
{
    vec3 blend = fract(uvw*vres);
    vec4 off = vec4(1.0/vres, 0.0);
    
    vec4 b000 = texture3D(tex, uvw + off.www, vres);
    vec4 b100 = texture3D(tex, uvw + off.xww, vres);
    
    vec4 b010 = texture3D(tex, uvw + off.wyw, vres);
    vec4 b110 = texture3D(tex, uvw + off.xyw, vres);
    
    vec4 b001 = texture3D(tex, uvw + off.wwz, vres);
    vec4 b101 = texture3D(tex, uvw + off.xwz, vres);
    
    vec4 b011 = texture3D(tex, uvw + off.wyz, vres);
    vec4 b111 = texture3D(tex, uvw + off.xyz, vres);
    
    return mix(mix(mix(b000,b100,blend.x), mix(b010,b110,blend.x), blend.y), 
               mix(mix(b001,b101,blend.x), mix(b011,b111,blend.x), blend.y),
               blend.z);
}

// Function 499
float Glyph5(const in vec2 uv)
{
    const vec2  vP0 = vec2 ( 0.507, 0.138 );
    const vec2  vP1 = vec2 ( 0.510, 0.065 );
    const vec2  vP2 = vec2 ( 0.570, 0.066 );
    const vec2  vP3 = vec2 ( 0.598, 0.066 );
    const vec2  vP4 = vec2 ( 0.594, 0.092 );
    const vec2  vP5 = vec2 ( 0.599, 0.131 );
    const vec2  vP6 = vec2 ( 0.537, 0.137 );
    const vec2  vP8 = vec2 ( 0.538, 0.125 );
    const vec2  vP9 = vec2 ( 0.564, 0.129 );
    const vec2 vP10 = vec2 ( 0.574, 0.100 );
    const vec2 vP11 = vec2 ( 0.584, 0.085 );
    const vec2 vP12 = vec2 ( 0.571, 0.079 );
    const vec2 vP13 = vec2 ( 0.557, 0.081 );
    const vec2 vP14 = vec2 ( 0.549, 0.103 );
    const vec2 vP15 = vec2 ( 0.518, 0.166 );
    const vec2 vP16 = vec2 ( 0.557, 0.166 );
    const vec2 vP17 = vec2 ( 0.589, 0.163 );
    const vec2 vP18 = vec2 ( 0.602, 0.137 );
    const vec2 vP20 = vec2 ( 0.602, 0.152 );
    const vec2 vP21 = vec2 ( 0.572, 0.194 );
    const vec2 vP22 = vec2 ( 0.537, 0.185 );
    const vec2 vP23 = vec2 ( 0.503, 0.189 );
    
    float fDist = 1.0;
    fDist = min( fDist, InCurve2(vP0,vP1,vP2, uv) ); 
    fDist = min( fDist, InCurve2(vP2,vP3,vP4, uv) ); 
    fDist = min( fDist, InCurve2(vP4,vP5,vP6, uv) );
    fDist = min( fDist, InCurve(vP8,vP9,vP10, uv) ); 
    fDist = min( fDist, InCurve(vP10,vP11,vP12, uv) ); 
    fDist = min( fDist, InCurve(vP12,vP13,vP14, uv) );
    fDist = min( fDist, InCurve(vP14,vP15,vP16, uv) );
    fDist = min( fDist, InCurve(vP16,vP17,vP18, uv) ); 
    fDist = min( fDist, InCurve2(vP20,vP21,vP22, uv) ); 
    fDist = min( fDist, InCurve2(vP22,vP23,vP0, uv) );

    fDist = min( fDist, InTri(vP0, vP2, vP13, uv) );
    fDist = min( fDist, InTri(vP13, vP2, vP12, uv) );
    fDist = min( fDist, InTri(vP2, vP11, vP12, uv) );
    fDist = min( fDist, InTri(vP2, vP4, vP11, uv) );
    fDist = min( fDist, InTri(vP11, vP4, vP10, uv) );
    fDist = min( fDist, InTri(vP10, vP4, vP9, uv) );
    fDist = min( fDist, InTri(vP6, vP8, vP9, uv) );
    fDist = min( fDist, InTri(vP0, vP13, vP14, uv) );
    fDist = min( fDist, InTri(vP0, vP14, vP15, uv) );
    fDist = min( fDist, InTri(vP15, vP16, vP22, uv) );
    fDist = min( fDist, InTri(vP16, vP17, vP22, uv) );
    fDist = min( fDist, InTri(vP17, vP18, vP20, uv) );
    
    return fDist;
}

// Function 500
vec3 TexTLite6_5( vec2 vTexCoord, float fRandom, vec3 vFlatCol, vec3 vLightCol )
{
	vec2 vLocalCoord = vTexCoord;
	vLocalCoord = mod(vLocalCoord, 32.0 );
	
    float fDist = length( vLocalCoord - 16. ) / 8.0;
    
    if ( fDist > 1.0 )
    {
        return vFlatCol * (0.5 + fRandom * 0.5);
    }
    
    float fLight = clamp(1.0 - fDist * fDist, 0.0, 1.0);
	return min( vec3(1.0), vLightCol * (fLight * 0.75 + 0.25) + pow( fLight, 5.0 ) * 0.4);    
}

// Function 501
int GetFocusGlyph(int i) { return RandInt(i) % glyphCount; }

// Function 502
float Glyph3(const in vec2 uv, vec2 vOffset)
{
    vec2 vP0 = vec2 ( 0.212, 0.112 ) + vOffset;
    vec2 vP2 = vec2 ( 0.243, 0.112 ) + vOffset;
    const vec2  vP4 = vec2 ( 0.234, 0.150 );
    const vec2  vP5 = vec2 ( 0.230, 0.159 );
    const vec2  vP6 = vec2 ( 0.243, 0.164 );
    const vec2  vP7 = vec2 ( 0.257, 0.164 );
    const vec2  vP8 = vec2 ( 0.261, 0.148 );
    const vec2 vP10 = vec2 ( 0.265, 0.164 );
    const vec2 vP11 = vec2 ( 0.256, 0.180 );
    const vec2 vP12 = vec2 ( 0.239, 0.185 );
    const vec2 vP13 = vec2 ( 0.194, 0.194 );
    const vec2 vP14 = vec2 ( 0.203, 0.150 );
    const vec2 vP16 = vec2 ( 0.212, 0.113 );

    float fDist = 1.0;
    fDist = min( fDist, InCurve(vP4,vP5,vP6, uv) );
    fDist = min( fDist, InCurve(vP6,vP7,vP8, uv) );
    fDist = min( fDist, InCurve2(vP10,vP11,vP12, uv) );
    fDist = min( fDist, InCurve2(vP12,vP13,vP14, uv) );

    fDist = min( fDist, InQuad(vP0, vP2, vP4, vP14, uv) );
    fDist = min( fDist, InTri(vP14, vP4, vP5, uv) );
    fDist = min( fDist, InTri(vP14, vP5, vP12, uv) );
    fDist = min( fDist, InTri(vP5, vP6, vP12, uv) );
    fDist = min( fDist, InTri(vP6, vP7, vP12, uv) );
    fDist = min( fDist, InTri(vP6, vP10, vP12, uv) );
    fDist = min( fDist, InTri(vP8, vP10, vP7, uv) );
    
    return fDist;
}

// Function 503
vec4 hqTexture(vec2 xy)
{
  //return texture(iChannel0,xy); /* low quality font lookup */
  vec2 d = vec2(1.0 / 1200.,0.);  
  return (     texture(iChannel0, xy + d + d.yx)
          +    texture(iChannel0, xy + d - d.yx)
          +    texture(iChannel0, xy - d - d.yx)
          +    texture(iChannel0, xy - d + d.yx)
          + 2.*texture(iChannel0,xy)
         ) / 6.0;
}

// Function 504
vec4 TeletextScreen( vec2 vScreenUV )
{      
    ivec2 vCharIndex = ivec2(floor(vScreenUV * vec2(kScreenChars)));

    if ( any( lessThan( vCharIndex, ivec2(0) ) ) )
    {
        return vec4(0);
	}    
    if ( any( greaterThanEqual( vCharIndex, kScreenChars ) ) )
    {
        return vec4(0);
	}
    
    vec4 vCharSample = texelFetch( iChannelCharData, vCharIndex, 0 );
    //vCharSample=vec4(7,GFX_ALPHANUMERIC,NORMAL_HEIGHT,_a);

    int charIndex = int(vCharSample.a);

    vec2 vCharUV = fract( vScreenUV * vec2(kScreenChars) );
    
    // Double height
    switch( int(vCharSample.z) )
    {
        default:
        case NORMAL_HEIGHT:
        break;
        
        case DOUBLE_HEIGHT_TOP:
        	vCharUV.y = vCharUV.y / 2.0;
        break;

        case DOUBLE_HEIGHT_BOTTOM:
        	vCharUV.y = vCharUV.y / 2.0 + 0.5;
        break;
    }
    
    int iFgCol = int(vCharSample.r) & 7;
    int iBgCol = (int(vCharSample.r) >> 8) & 7;
    
    int iGfx = int(vCharSample.g);
    
    vec4 col = vec4( kColors[ iBgCol ], 1.0 );
    if ( iBgCol == 0 )
    {
        col.a = 0.0;
    }

    if ( charIndex >= 0 )
    {
        if ( iGfx == GFX_ALPHANUMERIC )
        {
            float fCharData = SampleFontCharacter( charIndex, vCharUV );
            if ( fCharData > 0.0 )
            {
                vec4 vChCol = vec4( kColors[ iFgCol ], iFgCol == 0 ? 0.0 : 1.0 );
                col = mix( col, vChCol, fCharData );
            }
        }
        else
        {        
            int iGfxCharIndex = GetGfxCharIndex( charIndex );

            ivec2 vGfxCharPixel = ivec2(vCharUV * vec2(kCharPixels) );   

            #if !SHOW_GFX_HEX_VALUES
            ivec2 vGfxPixel = ivec2(vGfxCharPixel.x / 6, vGfxCharPixel.y / 7 );
            
            ivec2 vMosaicPixel = ivec2(vGfxCharPixel.x % 6, vGfxCharPixel.y % 7 );

            bool bGfxPixel = true;
            if ( iGfx == GFX_SEPARATED )
            {
                if ( vMosaicPixel.x > 3 || vMosaicPixel.y > 4 || vMosaicPixel.x < 1 || vMosaicPixel.y < 1 )
                {
                    bGfxPixel = false;
                }
            }
            
            if ( bGfxPixel )
            {
                int iGfxBit = vGfxPixel.x + vGfxPixel.y * 2;

                if ( ((iGfxCharIndex >> iGfxBit) & 1) != 0 )
                {
                    col = vec4( kColors[ iFgCol ], 1.0 );
                    if ( iFgCol == 0 )
                    {
                        col.a = 0.0;
                    }                    
                }
            }

            #else
            // Show hex values
            int ch[] = int[]( _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _A, _B, _C, _D, _E, _F );

            int d = 16;

            if ( vGfxCharPixel.y > 10 )
            {
                vGfxCharPixel.y -= 10;
                d = 1;
            }

            charIndex = ch[ (iGfxCharIndex / d) % 16 ];

            vGfxCharPixel.y *= 2;
            bool bCharData = CharBitmap12x20( charIndex, vGfxCharPixel );
            if ( bCharData )
            {
                col = vec4(kColors[ iFgCol ], 1);
            }		
            #endif
        }
    }
    
	return col;
}

// Function 505
vec4 textureQuadratic( in sampler2D sam, in vec2 p )
{
    float texSize = float(textureSize(sam,0).x); 
    
#if 1
    //Roger/iq style
	p = p*texSize;
	vec2 i = floor(p);
	vec2 f = fract(p);
	p = i + f*0.5;
	p = p/texSize;
    //f = f*f*(3.0-2.0*f); // optional for extra sweet
	float w = 0.5/texSize;
	return mix(mix(texture2(sam,p+vec2(0,0)),
                   texture2(sam,p+vec2(w,0)),f.x),
               mix(texture2(sam,p+vec2(0,w)),
                   texture2(sam,p+vec2(w,w)),f.x), f.y);
    
#else
    // paniq style (https://www.shadertoy.com/view/wtXXDl)
    vec2 f = fract(p*texSize);
    vec2 c = (f*(f-1.0)+0.5) / texSize;
    vec2 w0 = p - c;
    vec2 w1 = p + c;
    return (texture(sam, vec2(w0.x, w0.y))+
    	    texture(sam, vec2(w0.x, w1.y))+
    	    texture(sam, vec2(w1.x, w1.y))+
    	    texture(sam, vec2(w1.x, w0.y)))/4.0;
#endif    

    
}

// Function 506
vec4 NumFont_Char( ivec2 vTexCoord, int iDigit )
{
 	if ( iDigit < 0 || iDigit > 10 )
    	return vec4(0.0);
    
    //vTexCoord = floor(vTexCoord * vec2(14.0, 16.0)) + 0.5 + vec2(480,96);
    vTexCoord = vTexCoord + FONT_POS;
    vTexCoord += FONT_CHAR * iDigit;
    
    float fSample_TL = texelFetch( iChannel1, (vTexCoord - ivec2(-1, 1) ), 0 ).a;
    float fSample_TC = texelFetch( iChannel1, (vTexCoord - ivec2( 0, 1) ), 0 ).a;
    float fSample_TR = texelFetch( iChannel1, (vTexCoord - ivec2( 1, 1) ), 0 ).a;
    
    float fSample_CL = texelFetch( iChannel1, (vTexCoord - ivec2(-1, 0) ), 0 ).a;
    float fSample_CC = texelFetch( iChannel1, (vTexCoord - ivec2( 0, 0) ), 0 ).a;
    float fSample_CR = texelFetch( iChannel1, (vTexCoord - ivec2( 1, 0) ), 0 ).a;
    float fSample_CS = texelFetch( iChannel1, (vTexCoord - ivec2( 2, 0) ), 0 ).a;

    float fSample_BL = texelFetch( iChannel1, (vTexCoord - ivec2(-1,-1) ), 0 ).a;
    float fSample_BC = texelFetch( iChannel1, (vTexCoord - ivec2( 0,-1) ), 0 ).a;
    float fSample_BR = texelFetch( iChannel1, (vTexCoord - ivec2( 1,-1) ), 0 ).a;
    float fSample_BS = texelFetch( iChannel1, (vTexCoord - ivec2( 2,-1) ), 0 ).a;
    
    
    float fSample_SC = texelFetch( iChannel1, (vTexCoord - ivec2( 0,-2) ), 0 ).a;
    float fSample_SR = texelFetch( iChannel1, (vTexCoord - ivec2( 1,-2) ), 0 ).a;
    float fSample_SS = texelFetch( iChannel1, (vTexCoord - ivec2( 2,-2) ), 0 ).a;
   
    float fOutline = min( 1.0, 
		fSample_TL + fSample_TC + fSample_TR +
		fSample_CL + fSample_CC + fSample_CR +
		fSample_BL + fSample_BC + fSample_BR );
    
    float fShadow = min( 1.0, 
		fSample_CC + fSample_CR + fSample_CS +
		fSample_BC + fSample_BR + fSample_BS + 
		fSample_SC + fSample_SR + fSample_SS);
    	
    float fMain = fSample_CC;
    
    vec4 vResult = vec4(0.0);
    
    float fAlpha = min( 1.0, fOutline + fMain + fShadow );
    
    float fShade = fSample_TL * 1.5 + fSample_BR * -1.5 + fSample_TC * 1.0 + fSample_CL * 1.0 
        + fSample_BC * -1.0 + fSample_CR * -1.0;
    
    fShade = clamp( fShade * 0.25, 0.0, 1.0 );
    
    fShade = fShade * .3 + .7;
    
    vec3 vColor = vec3( .2 ); // drop shadow
    
    if ( fOutline > 0.0 )
        vColor = vec3(.4, 0, 0); // outline

    if ( fMain > 0.0 )
        vColor = vec3(fShade, 0, 0); // main text
            
    vResult = vec4(vColor, fAlpha);
    
    return vResult;
}

// Function 507
vec2 UIDrawContext_CanvasPosToScreenPos( UIDrawContext drawContext, vec2 vCanvasPos )
{
    return vCanvasPos - drawContext.vOffset + drawContext.viewport.vPos;
}

// Function 508
float GlyphSDF(vec2 p, float char)
{
    // Convert glyph to appropriate char index in the char texture and compute distance to it
	p = abs(p.x - .5) > .5 || abs(p.y - .5) > .5 ? vec2(0.) : p;
	return 2. * (texture(iChannel0, p / 16. + fract(vec2(char, 15. - floor(char / 16.)) / 16.)).w - 127. / 255.);
}

// Function 509
vec3 _texture(vec3 p) {
    vec3 t = fbm((p*50.)) * vec3(.9, .7, .5) * .75
        + smoothstep(.4, .9, fbm((p*10. + 2.))) * vec3(1., .4, .3)
        - smoothstep(.5, .9, fbm((p*100. + 4.))) * vec3(.4, .3, .2);
    return saturate(t);
}

// Function 510
float text_m(vec2 U) {
    initMsg;
    U.x+=5.5*(0.5-0.2812*(res.x/0.5)); //4.+1.5
    C(77);C(111);C(117);C(115);C(101);;
    endMsg;
}

// Function 511
float escherTexture(vec2 p, float pixel_size)
{
    float x = escherTextureX(p);
    float y = escherTextureY(p);
    
    x = smoothstep(-1.0, 1.0, x/pixel_size);
    y = smoothstep(-1.0, 1.0, y/pixel_size);
    
    float d = x+y - 2.0 * x*y;
    
    return d;
}

// Function 512
float NumFont_Pixel( vec2 vPos, vec2 vPixel )
{
    return NumFont_Rect( vPos, vPixel, vPixel );
}

// Function 513
float SampleDistanceTexture(vec2 texuv,float c)
{
    return Sampx(64u+uint(abs(c)),texuv);
}

// Function 514
float glyph_dist2(in vec2 pt, float angle) {
	float len = length(pt);
	float rad = 1.0- len;
	return rad- abs(sin(angle* spokes/ 2.0))* 0.6;
}

// Function 515
float noiseTexture(vec2 p, float t)
{
    p = .5*p + t;
    float val = tex(p);
    //return val;
    
    if keypress(32)
    {
        val *= .5;
        val += tex(p*2.) *.25;
        val += tex(p*4.) *.125;
        val += tex(p*8.) *.0625;
    }
    
    return val;
}

// Function 516
vec3 texturePixSpace(int x, int y, in vec2 fragCoord, int channelNum, in sampler2D tex)
{
    vec2 uv = fragCoord.xy / iResolution.xy * iChannelResolution[channelNum].xy;
	uv = (uv + vec2(x, y)) / iChannelResolution[channelNum].xy ;
	return texture(tex, uv).xyz;
}

// Function 517
vec3 texture3(vec2 uv){
    vec3 col = vec3(0.);
    vec2 auv = abs(uv-vec2(0.5,0.));
    col = mix(col, vec3(1.), 1.0-smoothstep(0.48,0.55,max(auv.x,auv.y)));
    return col;
    }

// Function 518
vec4 gridTexture(in vec2 uv)
{
    if(uv.y < 0.0)
    {
    	return vec4(0.0,0.0,0.0,0.0);
    }
    float thickness = 0.1;
	float speed = 1.5;
    
    float xPhase = mod(6.0*uv.x, 1.0);
    float yPhase = mod(6.0*uv.y-speed*iTime, 1.0);
            
    float xIntensity = max(0.0, 1.0-abs(0.5-xPhase)/thickness);
    float yIntensity = max(0.0, 1.0-abs(0.5-yPhase)/thickness);
    
    vec4 color = vec4(0.3, 0.7, 1.0, 1.0);
    
    vec4 result = (yIntensity+xIntensity)*color;
	return result;
}

// Function 519
float font2d_dist(vec2 tpos, float size, vec2 offset) {

    float scl = 0.63/size;
      
    vec2 uv = tpos*scl;
    vec2 font_uv = (uv+offset+0.5)*(1.0/16.0);
    
    float k = texture(iChannel2, font_uv, -100.0).w + 1e-6;
    
    vec2 box = abs(uv)-0.5;
        
    return max(k-127.0/255.0, max(box.x, box.y))/scl;
    
}

// Function 520
vec2 screen_from_font(vec2 font_uv, float font_size, vec2 char_pos) {
    return font_uv*UNITS_PER_TEXEL*font_size - (char_pos+0.5)*font_size;
}

// Function 521
vec4 createTexture( in vec2 p )
{
    vec2 cc = vec2( -0.1, 0.68 );

	vec4 dmin = vec4(1000.0);
    float w = 0.0;
    vec2 z = 1.1*(-1.0 + 2.0*p)*vec2(iChannelResolution[0].x/iChannelResolution[0].y,1.0);
    for( int i=0; i<80; i++ )
    {
        z = cc + vec2( z.x*z.x - z.y*z.y, 2.0*z.x*z.y );

		dmin=min(dmin, vec4(length( z-0.5), 
							abs(-0.5+z.x + 0.2*sin(5.0*z.y)), 
							dot(z,z),
						    length( fract(z/8.0)-0.5) ) );
        if( dot(z,z)>4.0 ) w=1.0;
    }
   
    vec3 col = vec3(0.6,0.6,0.6);
    col * 0.4+0.6*w;
    col *= mix( vec3(1.0,0.45,0.1), vec3(1.0), w );
    col *= 0.65 + dmin.w;
    col = mix( col, 1.5*vec3(0.7,0.7,0.7),1.0-clamp(dmin.y*15.0,0.0,1.0) );
    col = mix( col, vec3(1.1,1.1,1.0),1.0-clamp(dmin.x*2.0,0.0,1.0) );
	col *= 0.5 + 0.5*clamp(dmin.z*50.0,0.0,1.0);


    return vec4( col, 1.0 );
}

// Function 522
void findTextureTargets(out vec2 pA, out vec2 pB)
{	
	vec3 Ctot = vec3(0.);
	pA = vec2(0.); float Atot=0.;
	pB = vec2(0.); float Btot=0.;
	for (int j=0; j< SAMPLE; j++)
	  for (int i=0; i< SAMPLE; i++)
	  {
		  vec2 pos = (.5+vec2(i,j))/float(SAMPLE);
		  vec3 c = texture(iChannel0,pos,LEVEL).rgb;
		  Ctot += c;
		  float v;
		  
		  v = match(c,targetA);
		  pA   += pos*v;
		  Atot += v;
		  
		  v = match(c,targetB);
		  pB   += pos*v;
		  Btot += v;	  
	  }
	pA /= Atot;
	pB /= Btot;
	_ambientI = lum(Ctot)/float(SAMPLE*SAMPLE);
	return;		 
}

// Function 523
float getDistortedTexture(vec2 uv, float height){

    float strength = 0.6;
    
    // The texture is distorted in time and we switch between two texture states.
    // The transition is based on Worley noise which will shift the change of differet parts
    // for a more organic result
    float time = 0.25 * iTime + texture(iChannel2, uv).g;
   
    // Make the texture on the upper body of the elemental static and more coarse
    if(height > TRANSITION){
        time = 0.0;
        uv *= 0.2;
    }
    
    float f = fract(time);
    
    // Get the velocity at the current location
    vec2 grad = getGradient(uv);
    vec2 distortion = strength * vec2(grad.x, grad.y);
    
    // Get two shifted states of the texture distorted in time by the local velocity.
    // Loop the distortion from 0 -> 1 using fract(time)
    float distort1 = texture(iChannel2, uv + f * distortion).r;
    float distort2 = texture(iChannel2, 0.1 + uv + fract(time + 0.5) * distortion).r;

    // Mix between the two texture states to hide the sudden jump from 1 -> 0.
    // Modulate the value returned by the velocity to make slower regions darker in the final
    // lava render.
    return (1.0-length(grad)) * (mix(distort1, distort2, abs(1.0 - 2.0 * f)));
}

// Function 524
vec3 texture1(vec2 uv){
    vec3 col = vec3(0.);
    float f = step(0.5,fract(uv.x*8.))*step(0.5,fract(uv.y*8.));
    col = mix(col, vec3(1.),f);
    return col;
    
}

// Function 525
vec2 FontTexNf (vec2 p)
{
  vec2 tx;
  tx = texture (txFnt, mod ((bIdV + p) * (1. / 16.), 1.)).gb - 0.5;
  return vec2 (tx.x, - tx.y);
}

// Function 526
vec3 texture_terrain(vec3 p, vec3 nrm, bool is_edge)
{
    vec3 col = vec3(0.);
    vec3 stone = vec3(.6);
    vec3 grass = vec3(.4, .6, .4);
    vec3 snow = vec3(1.1, 1.1, 1.2)*4.;
    float stone_mask = min(max((dot(nrm, vec3(0,0,1))-.87), 0.)*8., 1.);
    float snow_mask = min(max((p.z-.14), 0.)*64., 1.);
    
    if(is_edge)//draw edge
    {
        float h = get_terrain(p.xy).w-p.z;
        stone_mask += 1.-min(max(h-.01, 0.)*64., 1.);
        
    }
    
    vec3 ground = mix(grass, snow, snow_mask);
    
    col = mix(stone, ground, stone_mask);

    
    return col;
}

// Function 527
float text_floor(vec2 U) {
    initMsg;
    U.x+=4.*(0.5-0.2812*(res.x/0.5));
    C(70);C(108);C(111);C(111);C(114);C(48);
    endMsg;
}

// Function 528
vec3 img_texture(vec2 uv){
    //Bitmap y starts at top
    uv.y = 1.0-uv.y;
    
    int x = int(uv.x*float(IMG_WIDTH)),
        y = int(uv.y*float(IMG_HEIGHT));
    int index = y*IMG_WIDTH + x;
    
    //Wonky bitwise operations to extract
    //correct 4 bits from the hex value.
    uint hex = img[index/8];
    int shift_cnt = 8 - (index%8+1);
    shift_cnt *= 4;
    int pixel = int((hex>>(shift_cnt))&0xfu);
	
    //Given pixel value, return correct
    //rgb color from color table
    int col = pal[pixel];
    
    return vec3(
        float((col>>16)&0xff)/255.0,
        float((col>>8)&0xff)/255.0,
        float((col)&0xff)/255.0
    );
}

// Function 529
void UIStyle_GetFontStyleTitle( inout LayoutStyle style, inout RenderStyle renderStyle )
{
    style = LayoutStyle_Default();
    style.vSize *= 0.75;
	renderStyle = RenderStyle_Default( vec3(1.0) );
}

// Function 530
vec4 SampleTextureCatmullRom(sampler2D sceneTexture, vec2 uv, vec2 texSize, float mipLevel, int getPacked)
{
    vec4 result = vec4(0.0);
    if(getPacked == unpackedNone)
    {
        // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
        // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
        // location [1, 1] in the grid, where [0, 0] is the top left corner.
        vec2 samplePos = uv * texSize;
        vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

        // Compute the fractional offset from our starting texel to our original sample location, which we'll
        // feed into the Catmull-Rom spline function to get our filter weights.
        vec2 f = samplePos - texPos1;

        // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
        // These equations are pre-expanded based on our knowledge of where the texels will be located,
        // which lets us avoid having to evaluate a piece-wise function.
        vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
        vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
        vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
        vec2 w3 = f * f * (-0.5 + 0.5 * f);

        // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
        // simultaneously evaluate the middle 2 samples from the 4x4 grid.
        vec2 w12 = w1 + w2;
        vec2 offset12 = w2 / w12;

        // Compute the final UV coordinates we'll use for sampling the texture
        vec2 texPos0 = texPos1 - vec2(1.0);
        vec2 texPos3 = texPos1 + vec2(2.0);
        vec2 texPos12 = texPos1 + offset12;

        texPos0 /= texSize;
        texPos3 /= texSize;
        texPos12 /= texSize;
        
        result += sampleLevel0(sceneTexture, vec2(texPos0.x,  texPos0.y), mipLevel) * w0.x * w0.y;
        result += sampleLevel0(sceneTexture, vec2(texPos12.x, texPos0.y), mipLevel) * w12.x * w0.y;
        result += sampleLevel0(sceneTexture, vec2(texPos3.x,  texPos0.y), mipLevel) * w3.x * w0.y;

        result += sampleLevel0(sceneTexture, vec2(texPos0.x,  texPos12.y), mipLevel) * w0.x * w12.y;
        result += sampleLevel0(sceneTexture, vec2(texPos12.x, texPos12.y), mipLevel) * w12.x * w12.y;
        result += sampleLevel0(sceneTexture, vec2(texPos3.x,  texPos12.y), mipLevel) * w3.x * w12.y;

        result += sampleLevel0(sceneTexture, vec2(texPos0.x,  texPos3.y), mipLevel) * w0.x * w3.y;
        result += sampleLevel0(sceneTexture, vec2(texPos12.x, texPos3.y), mipLevel) * w12.x * w3.y;
        result += sampleLevel0(sceneTexture, vec2(texPos3.x,  texPos3.y), mipLevel) * w3.x * w3.y;
    }
    
    return result;
}

// Function 531
void glyph_7()
{
  MoveTo(x*0.1+y*1.8);
  RLineTo(y*0.2);
  RLineTo(x*1.8);
  Bez3To(x+y*1.3,x+y*0.5,x);
}

// Function 532
float NumFont_Two( vec2 vTexCoord )
{
    float fResult = 0.0;
    
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(2, 1), vec2(9,3) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(3, 6), vec2(9,8) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 8), vec2(4,13) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(1, 11), vec2(10,13) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(2, 7), vec2(10,7) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(8, 3), vec2(11,6) ));
    fResult = max( fResult, NumFont_Pixel( vTexCoord, vec2(10, 2) ) );

    return fResult;
}

// Function 533
vec4 texture3D(sampler2D tex, vec3 uvw, vec3 vres)
{
    uvw = mod(floor(uvw * vres), vres);
    float idx = (uvw.z * (vres.x*vres.y)) + (uvw.y * vres.x) + uvw.x;
    vec2 uv = vec2(mod(idx, iResolution.x), floor(idx / iResolution.x));
    
    return texture(tex, (uv + 0.5) / iResolution.xy);
}

// Function 534
float textureThreshold(vec2 uv)
{
    vec4 textap = texture(iChannel0, uv);
    float luma = dot(textap.xyz, vec3(0.3, 0.6, 0.1));
    return step(0.5, luma);
}

// Function 535
vec3 SampleInterpolationTextureBicubic (vec2 uv)
{
    vec2 pixel = uv * 2.0 - 0.5;
    vec2 pixelFract = fract(pixel);
    
    vec3 pixelNN = SampleInterpolationTexturePixel(floor(pixel) + vec2(-1.0, -1.0));
    vec3 pixel0N = SampleInterpolationTexturePixel(floor(pixel) + vec2( 0.0, -1.0));
    vec3 pixel1N = SampleInterpolationTexturePixel(floor(pixel) + vec2( 1.0, -1.0));
    vec3 pixel2N = SampleInterpolationTexturePixel(floor(pixel) + vec2( 2.0, -1.0));
    
    vec3 pixelN0 = SampleInterpolationTexturePixel(floor(pixel) + vec2(-1.0,  0.0));
    vec3 pixel00 = SampleInterpolationTexturePixel(floor(pixel) + vec2( 0.0,  0.0));
    vec3 pixel10 = SampleInterpolationTexturePixel(floor(pixel) + vec2( 1.0,  0.0));
    vec3 pixel20 = SampleInterpolationTexturePixel(floor(pixel) + vec2( 2.0,  0.0));   
    
    vec3 pixelN1 = SampleInterpolationTexturePixel(floor(pixel) + vec2(-1.0,  1.0));
    vec3 pixel01 = SampleInterpolationTexturePixel(floor(pixel) + vec2( 0.0,  1.0));
    vec3 pixel11 = SampleInterpolationTexturePixel(floor(pixel) + vec2( 1.0,  1.0));
    vec3 pixel21 = SampleInterpolationTexturePixel(floor(pixel) + vec2( 2.0,  1.0));     
    
    vec3 pixelN2 = SampleInterpolationTexturePixel(floor(pixel) + vec2(-1.0,  2.0));
    vec3 pixel02 = SampleInterpolationTexturePixel(floor(pixel) + vec2( 0.0,  2.0));
    vec3 pixel12 = SampleInterpolationTexturePixel(floor(pixel) + vec2( 1.0,  2.0));
    vec3 pixel22 = SampleInterpolationTexturePixel(floor(pixel) + vec2( 2.0,  2.0));     
    
    vec3 rowN = CubicHermite(pixelNN, pixel0N, pixel1N, pixel2N, pixelFract.x);
    vec3 row0 = CubicHermite(pixelN0, pixel00, pixel10, pixel20, pixelFract.x);
    vec3 row1 = CubicHermite(pixelN1, pixel01, pixel11, pixel21, pixelFract.x);
    vec3 row2 = CubicHermite(pixelN2, pixel02, pixel12, pixel22, pixelFract.x);
    
    return CubicHermite(rowN, row0, row1, row2, pixelFract.y);
}

// Function 536
float Glyph1(const in vec2 uv, const in vec2 vOffset)
{
    vec2 vP0 = vec2 ( 0.171, 0.026 ) + vOffset;
    vec2 vP1 = vec2 ( 0.204, 0.022 ) + vOffset;
    const vec2 vP2 = vec2 ( 0.170, 0.185 );
    const vec2 vP3 = vec2 ( 0.137, 0.185 );
    
    return InQuad(vP0, vP1, vP2, vP3, uv);
}

// Function 537
vec3 glyph(vec2 u,vec2 m){vec3 c=vec3(0)
 ;c.g=length(vec2(abs(u.x)-m.x-m.y,stretch(u.y,m.y)))
 ;c.b=length(vec2(stretch2(u.x,m.x),abs(u.y)-m.y))
 ;u.x=abs(u.x)-m.x//+m.y //thee +m.y makes the mirror symmetry weird
 ;c.r=abs(length(u)-m.y)
 //;c.x=miv(c);c=c.xxx//to greyscale
 ;return c;}

// Function 538
float map_text(vec3 pos, vec3 offset, float depth, float bevel, float bold, int defText)
{   
   if (defText==OBJ_TAMBAKO) nbchars = 7;
   if (defText==OBJ_THE_JAGUAR) nbchars = 10;
   if (defText==OBJ_PRESENTS) nbchars = 8;
   if (defText==OBJ_A_COOL_SHADER) nbchars = 13;
   if (defText==OBJ_POWERED_BY) nbchars = 13;
   if (defText==OBJ_SHADERTOY) nbchars = 9;    
    
   pos+= offset;
   vec2 uv = objscale*pos.xy + vec2(float(nbchars)*charSpacingFac.x*textScale.x/32., 0.025);
   float text = textTexture(uv, defText) - 0.5 - bold;
    
   text+= bevel*smoothstep(-depth + bevel*0.5, -depth - bevel*0.5, pos.z);
    
   float cropBox = sdBox(pos - vec3(0., 0.1, 0.), vec3(float(nbchars)*charSpacingFac.x*textScale.x/(objscale*32.), charSpacingFac.y*textScale.y/(objscale*33.), depth));
    
   return max(text, cropBox);
}

// Function 539
void UIStyle_GetFontStyleTitle( inout LayoutStyle style, inout RenderStyle renderStyle )
{
    style = LayoutStyle_Default();
	renderStyle = RenderStyle_Default( cWindowTitle );
}

// Function 540
vec4 texture_blurred(in sampler2D tex, vec2 uv)
{
    return (texture(iChannel0, uv)
		+ texture(iChannel0, vec2(uv.x+1.0, uv.y))
		+ texture(iChannel0, vec2(uv.x-1.0, uv.y))
		+ texture(iChannel0, vec2(uv.x, uv.y+1.0))
		+ texture(iChannel0, vec2(uv.x, uv.y-1.0)))/5.0;
}

// Function 541
vec4 textureLod(sampler2D s,vec2 c,float b){return texture2DLodEXT(s,c,b);}

// Function 542
vec3 GetBridgeTexture(RayHit marchResult)
{
  vec3 checkPos = TranslateBridge(marchResult.hitPos); 
  vec3 woodTexture = vec3(BoxMap(iChannel1, vec3(checkPos.z*0.01, checkPos.yx*0.31), (marchResult.normal), 0.5).r);
  vec3 bridgeColor =  woodTexture*(0.6+(0.4*noise(checkPos.zx*17.)));
  float n = noise2D(checkPos.xz*1.3);
  return mix(bridgeColor*MOSSCOLOR2, bridgeColor, smoothstep(-.64-(2.*n), 2.269-(2.*n), marchResult.hitPos.y));
}

// Function 543
float fontDist(vec2 p)
{
    p.x *= 0.5;
    p.x += iTime*0.1;
    p.y += iTime*0.07;

    p.x += pMod1(p.y,0.18)*0.2;		// offset X based on yMod
    pMod1(p.x, 1.5);					// xMod
    p += vec2(0.74,0.09);
    p*=5.2;
    
    float d = 1.0;
    float w = 0.45;
    
    d =  char(p,77,d,w);	//M
    d =  char(p,69,d,w);	//E
    d =  char(p,82,d,w);	//R
    d =  char(p,82,d,w);	//R
    d =  char(p,89,d,w);	//Y
    p.x -= w*0.75;
    d =  char(p,67,d,w);	//C
    d =  char(p,72,d,w);	//H
    d =  char(p,82,d,w);	//R
    d =  char(p,73,d,w);	//I
    d =  char(p,83,d,w);	//S
    d =  char(p,84,d,w);	//T
    d =  char(p,77,d,w);	//M
    d =  char(p,65,d,w);	//A
    d =  char(p,83,d,w);	//S
    return d;
}

// Function 544
t texture(iChannel0,p*.1,3.
void mainImage( out vec4 f, in vec2 p )
{
    vec4 q = p.xyxy/iResolution.y - .5, c=q-q;
    
    p.y = atan( q.x, q.y );
    
    for( float s=0.; s<.1; s+=.01 )
    {
        float x = length( q.xy ), z = 1.;
        
        for( ; z>0. && t).x<z ; z-=.01 )
            p.x = iTime*3. + s + 1./(x+x*z);

        f = c += t*x)*z*x*.2;
    }
}

// Function 545
vec4 earthTexture(vec2 p)
{
    uint v = 0u;
	v = p.y == 99. ? 2576980377u : v;
	v = p.y == 98. ? 2576980377u : v;
	v = p.y == 97. ? 2576980377u : v;
	v = p.y == 96. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2576980377u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2005506457u : (p.x < 96. ? 2574743415u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2004457881u : (p.x < 136. ? 2543417207u : 2576980377u))))))))))))))))) : v;
	v = p.y == 95. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2576980377u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2040109465u : (p.x < 96. ? 2004318071u : (p.x < 104. ? 2004318071u : (p.x < 112. ? 2004318071u : (p.x < 120. ? 2004318071u : (p.x < 128. ? 2004318071u : (p.x < 136. ? 2004318020u : (p.x < 144. ? 2004318071u : (p.x < 152. ? 2004318071u : (p.x < 160. ? 2574743415u : 2576980377u)))))))))))))))))))) : v;
	v = p.y == 94. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2575866265u : (p.x < 64. ? 2576910745u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 288638841u : (p.x < 104. ? 858992913u : (p.x < 112. ? 2004317235u : (p.x < 120. ? 322123127u : (p.x < 128. ? 1144197393u : (p.x < 136. ? 1146582323u : (p.x < 144. ? 858993459u : (p.x < 152. ? 1999844147u : (p.x < 160. ? 2004318617u : (p.x < 168. ? 2575857527u : 2576980377u))))))))))))))))))))) : v;
	v = p.y == 93. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2541189529u : (p.x < 64. ? 2576840569u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 286331207u : (p.x < 104. ? 286331153u : (p.x < 112. ? 2005365777u : (p.x < 120. ? 286331767u : (p.x < 128. ? 286331153u : (p.x < 136. ? 823202065u : (p.x < 144. ? 286331187u : (p.x < 152. ? 1125191953u : (p.x < 160. ? 1199011700u : (p.x < 168. ? 2004317251u : (p.x < 176. ? 2541188983u : 2576980377u)))))))))))))))))))))) : v;
	v = p.y == 92. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2004457369u : (p.x < 64. ? 2004103989u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 286331255u : (p.x < 104. ? 286331153u : (p.x < 112. ? 323236113u : (p.x < 120. ? 286331155u : (p.x < 128. ? 286331153u : (p.x < 136. ? 1950626065u : (p.x < 144. ? 286331767u : (p.x < 152. ? 1145245969u : (p.x < 160. ? 2040100727u : (p.x < 168. ? 858994551u : (p.x < 176. ? 2004043636u : (p.x < 184. ? 2004318071u : (p.x < 192. ? 2576980375u : 2576980377u)))))))))))))))))))))))) : v;
	v = p.y == 91. ? (p.x < 8. ? 2576980377u : (p.x < 16. ? 2576980377u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2004318089u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2039978393u : (p.x < 64. ? 823202100u : (p.x < 72. ? 2576839475u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2040109465u : (p.x < 96. ? 286331191u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 930576145u : (p.x < 144. ? 286331155u : (p.x < 152. ? 2574730305u : (p.x < 160. ? 2576976281u : (p.x < 168. ? 288638873u : (p.x < 176. ? 823202627u : (p.x < 184. ? 2000975735u : (p.x < 192. ? 2576979831u : 2576980377u)))))))))))))))))))))))) : v;
	v = p.y == 90. ? (p.x < 8. ? 2006555033u : (p.x < 16. ? 2004318071u : (p.x < 24. ? 2004318071u : (p.x < 32. ? 2536719447u : (p.x < 40. ? 2576979833u : (p.x < 48. ? 2543425689u : (p.x < 56. ? 2005367159u : (p.x < 64. ? 286340215u : (p.x < 72. ? 2574455057u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2004326809u : (p.x < 96. ? 286331155u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286488435u : (p.x < 144. ? 1127511345u : (p.x < 152. ? 2576980343u : (p.x < 160. ? 2004326809u : (p.x < 168. ? 2004384153u : (p.x < 176. ? 286488439u : (p.x < 184. ? 876902193u : (p.x < 192. ? 1145324339u : 2541188164u)))))))))))))))))))))))) : v;
	v = p.y == 89. ? (p.x < 8. ? 1148680055u : (p.x < 16. ? 2004309364u : (p.x < 24. ? 860111959u : (p.x < 32. ? 1395732787u : (p.x < 40. ? 2576840519u : (p.x < 48. ? 2541189017u : (p.x < 56. ? 2039977847u : (p.x < 64. ? 326605207u : (p.x < 72. ? 1967198481u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 1467455897u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286470963u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286345075u : (p.x < 144. ? 2004326227u : (p.x < 152. ? 2576979833u : (p.x < 160. ? 2576840567u : (p.x < 168. ? 1433893273u : (p.x < 176. ? 2004318070u : (p.x < 184. ? 1467446612u : (p.x < 192. ? 286475332u : 2000892211u)))))))))))))))))))))))) : v;
	v = p.y == 88. ? (p.x < 8. ? 825456503u : (p.x < 16. ? 1431655763u : (p.x < 24. ? 876959061u : (p.x < 32. ? 1414808628u : (p.x < 40. ? 2004317303u : (p.x < 48. ? 2004318071u : (p.x < 56. ? 1199011703u : (p.x < 64. ? 932812660u : (p.x < 72. ? 1412501777u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 897030553u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1145245969u : (p.x < 120. ? 860095013u : (p.x < 128. ? 856756531u : (p.x < 136. ? 1145320243u : (p.x < 144. ? 2003068228u : (p.x < 152. ? 2004318071u : (p.x < 160. ? 2004322167u : (p.x < 168. ? 1431656311u : (p.x < 176. ? 1431795285u : (p.x < 184. ? 1431655765u : (p.x < 192. ? 1145328981u : 1984189508u)))))))))))))))))))))))) : v;
	v = p.y == 87. ? (p.x < 8. ? 825509751u : (p.x < 16. ? 572662340u : (p.x < 24. ? 572662306u : (p.x < 32. ? 1428300322u : (p.x < 40. ? 2004313973u : (p.x < 48. ? 2004318071u : (p.x < 56. ? 393705369u : (p.x < 64. ? 2005366611u : (p.x < 72. ? 1947275591u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 286476152u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 627519761u : (p.x < 120. ? 1140850688u : (p.x < 128. ? 1145246530u : (p.x < 136. ? 1428309060u : (p.x < 144. ? 1430541653u : (p.x < 152. ? 1717904725u : (p.x < 160. ? 1428313702u : (p.x < 168. ? 2250069u : (p.x < 176. ? 572880160u : (p.x < 184. ? 572859749u : (p.x < 192. ? 572662306u : 1985303893u)))))))))))))))))))))))) : v;
	v = p.y == 86. ? (p.x < 8. ? 825517636u : (p.x < 16. ? 8772u : (p.x < 24. ? 570425344u : (p.x < 32. ? 1073881634u : (p.x < 40. ? 1968514115u : (p.x < 48. ? 2004318070u : (p.x < 56. ? 343373687u : (p.x < 64. ? 2004317267u : (p.x < 72. ? 1947275591u : (p.x < 80. ? 1200200089u : (p.x < 88. ? 1091637556u : (p.x < 96. ? 286475604u : (p.x < 104. ? 286331153u : (p.x < 112. ? 2446097u : (p.x < 120. ? 872415283u : (p.x < 128. ? 817u : (p.x < 136. ? 622854144u : (p.x < 144. ? 1431446018u : (p.x < 152. ? 572522498u : (p.x < 160. ? 572662309u : (p.x < 168. ? 9554u : (p.x < 176. ? 572945664u : (p.x < 184. ? 626349669u : (p.x < 192. ? 1428300288u : 1431655766u)))))))))))))))))))))))) : v;
	v = p.y == 85. ? (p.x < 8. ? 825438993u : (p.x < 16. ? 536880195u : (p.x < 24. ? 8738u : (p.x < 32. ? 2445824u : (p.x < 40. ? 1145176064u : (p.x < 48. ? 2004252533u : (p.x < 56. ? 897013623u : (p.x < 64. ? 2004317251u : (p.x < 72. ? 1091637523u : (p.x < 80. ? 323459479u : (p.x < 88. ? 823202065u : (p.x < 96. ? 286345044u : (p.x < 104. ? 286331153u : (p.x < 112. ? 805315123u : (p.x < 120. ? 1073741827u : (p.x < 128. ? 3u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 512u : (p.x < 176. ? 622993408u : (p.x < 184. ? 1449481573u : (p.x < 192. ? 1431654946u : 878007637u)))))))))))))))))))))))) : v;
	v = p.y == 84. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 1375732260u : (p.x < 24. ? 89478485u : (p.x < 32. ? 39198720u : (p.x < 40. ? 1145315328u : (p.x < 48. ? 1162168148u : (p.x < 56. ? 322113844u : (p.x < 64. ? 1195656563u : (p.x < 72. ? 823202065u : (p.x < 80. ? 286755207u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 1091637521u : (p.x < 112. ? 288358949u : (p.x < 120. ? 1074003968u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 0u : (p.x < 176. ? 0u : (p.x < 184. ? 1432840034u : (p.x < 192. ? 1414813013u : 860321109u)))))))))))))))))))))))) : v;
	v = p.y == 83. ? (p.x < 8. ? 823202065u : (p.x < 16. ? 1140859443u : (p.x < 24. ? 1735738436u : (p.x < 32. ? 2228261u : (p.x < 40. ? 1074003968u : (p.x < 48. ? 1195656260u : (p.x < 56. ? 286331153u : (p.x < 64. ? 826767219u : (p.x < 72. ? 286331155u : (p.x < 80. ? 286540865u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 1091637521u : (p.x < 112. ? 285212709u : (p.x < 120. ? 279619u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 139264u : (p.x < 176. ? 0u : (p.x < 184. ? 1146443090u : (p.x < 192. ? 1145123925u : 286471236u)))))))))))))))))))))))) : v;
	v = p.y == 82. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 874791697u : (p.x < 24. ? 1127420177u : (p.x < 32. ? 35652980u : (p.x < 40. ? 0u : (p.x < 48. ? 1140868096u : (p.x < 56. ? 286331155u : (p.x < 64. ? 1952937795u : (p.x < 72. ? 286331191u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 1091638065u : (p.x < 112. ? 286261268u : (p.x < 120. ? 16451u : (p.x < 128. ? 0u : (p.x < 136. ? 0u : (p.x < 144. ? 0u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 36000256u : (p.x < 176. ? 139776u : (p.x < 184. ? 286340130u : (p.x < 192. ? 342110515u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 81. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 858993425u : (p.x < 24. ? 286331153u : (p.x < 32. ? 2249777u : (p.x < 40. ? 0u : (p.x < 48. ? 536887296u : (p.x < 56. ? 286331716u : (p.x < 64. ? 1951749185u : (p.x < 72. ? 286331255u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331939u : (p.x < 112. ? 288621379u : (p.x < 120. ? 0u : (p.x < 128. ? 570425344u : (p.x < 136. ? 33562624u : (p.x < 144. ? 2236960u : (p.x < 152. ? 0u : (p.x < 160. ? 0u : (p.x < 168. ? 626349394u : (p.x < 176. ? 539107874u : (p.x < 184. ? 286331188u : (p.x < 192. ? 341132049u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 80. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331697u : (p.x < 24. ? 286331153u : (p.x < 32. ? 148241u : (p.x < 40. ? 2097664u : (p.x < 48. ? 0u : (p.x < 56. ? 288620544u : (p.x < 64. ? 1146373187u : (p.x < 72. ? 286340180u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1091637521u : (p.x < 104. ? 286343748u : (p.x < 112. ? 1127433251u : (p.x < 120. ? 2097698u : (p.x < 128. ? 572662306u : (p.x < 136. ? 572654114u : (p.x < 144. ? 572662306u : (p.x < 152. ? 33563170u : (p.x < 160. ? 0u : (p.x < 168. ? 2105940u : (p.x < 176. ? 1109393408u : (p.x < 184. ? 286331187u : (p.x < 192. ? 323310353u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 79. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 4469009u : (p.x < 40. ? 572662304u : (p.x < 48. ? 4456994u : (p.x < 56. ? 318767104u : (p.x < 64. ? 71573508u : (p.x < 72. ? 286474242u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 588321041u : (p.x < 104. ? 1127490100u : (p.x < 112. ? 572662306u : (p.x < 120. ? 572662306u : (p.x < 128. ? 572662306u : (p.x < 136. ? 1428300322u : (p.x < 144. ? 1431655717u : (p.x < 152. ? 8738u : (p.x < 160. ? 1073750528u : (p.x < 168. ? 36u : (p.x < 176. ? 8704u : (p.x < 184. ? 286331712u : (p.x < 192. ? 288637713u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 78. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 1430458641u : (p.x < 40. ? 1428300288u : (p.x < 48. ? 67248677u : (p.x < 56. ? 805306368u : (p.x < 64. ? 0u : (p.x < 72. ? 286470144u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 856756497u : (p.x < 104. ? 574890561u : (p.x < 112. ? 572662306u : (p.x < 120. ? 572523042u : (p.x < 128. ? 572662306u : (p.x < 136. ? 1431446050u : (p.x < 144. ? 1431655765u : (p.x < 152. ? 622862885u : (p.x < 160. ? 69358114u : (p.x < 168. ? 35790848u : (p.x < 176. ? 8704u : (p.x < 184. ? 286327040u : (p.x < 192. ? 286339857u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 77. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 1144066321u : (p.x < 40. ? 1431437312u : (p.x < 48. ? 2237781u : (p.x < 56. ? 16384u : (p.x < 64. ? 872415232u : (p.x < 72. ? 286273843u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 572801841u : (p.x < 112. ? 572662306u : (p.x < 120. ? 572662306u : (p.x < 128. ? 1431446050u : (p.x < 136. ? 1431655765u : (p.x < 144. ? 1431655765u : (p.x < 152. ? 1431642709u : (p.x < 160. ? 623203669u : (p.x < 168. ? 576016928u : (p.x < 176. ? 8738u : (p.x < 184. ? 286339328u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 76. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 1091637521u : (p.x < 40. ? 1428161056u : (p.x < 48. ? 35792213u : (p.x < 56. ? 209728u : (p.x < 64. ? 0u : (p.x < 72. ? 322122001u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 572662545u : (p.x < 112. ? 572662306u : (p.x < 120. ? 572662306u : (p.x < 128. ? 1431642658u : (p.x < 136. ? 1717986901u : (p.x < 144. ? 1431655765u : (p.x < 152. ? 1700091221u : (p.x < 160. ? 1449481830u : (p.x < 168. ? 576087653u : (p.x < 176. ? 2228770u : (p.x < 184. ? 286339376u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 75. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 17895697u : (p.x < 40. ? 1431438674u : (p.x < 48. ? 572675413u : (p.x < 56. ? 1127428130u : (p.x < 64. ? 1073750016u : (p.x < 72. ? 286331187u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 572662033u : (p.x < 112. ? 572670498u : (p.x < 120. ? 1144266786u : (p.x < 128. ? 894771747u : (p.x < 136. ? 1717986900u : (p.x < 144. ? 1449551462u : (p.x < 152. ? 1701205333u : (p.x < 160. ? 1717986918u : (p.x < 168. ? 1431725670u : (p.x < 176. ? 2228773u : (p.x < 184. ? 286331155u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 74. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 17895697u : (p.x < 40. ? 1700091221u : (p.x < 48. ? 572872022u : (p.x < 56. ? 606356002u : (p.x < 64. ? 858783744u : (p.x < 72. ? 286331155u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 572662579u : (p.x < 112. ? 539247172u : (p.x < 120. ? 823337506u : (p.x < 128. ? 341123633u : (p.x < 136. ? 1719109734u : (p.x < 144. ? 1432774246u : (p.x < 152. ? 1449547093u : (p.x < 160. ? 1717986917u : (p.x < 168. ? 1700161158u : (p.x < 176. ? 840957990u : (p.x < 184. ? 286475025u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 73. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 17895697u : (p.x < 40. ? 1700091221u : (p.x < 48. ? 572872022u : (p.x < 56. ? 572670498u : (p.x < 64. ? 286457858u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1125191953u : (p.x < 104. ? 860181845u : (p.x < 112. ? 607339569u : (p.x < 120. ? 825442850u : (p.x < 128. ? 874660627u : (p.x < 136. ? 1717995620u : (p.x < 144. ? 1431656038u : (p.x < 152. ? 1717986917u : (p.x < 160. ? 1717986918u : (p.x < 168. ? 1716872806u : (p.x < 176. ? 322109477u : (p.x < 184. ? 286470929u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 72. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 17895697u : (p.x < 40. ? 1717986901u : (p.x < 48. ? 572872022u : (p.x < 56. ? 35791394u : (p.x < 64. ? 286340128u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1125191953u : (p.x < 104. ? 288642389u : (p.x < 112. ? 591671603u : (p.x < 120. ? 1431589954u : (p.x < 128. ? 1163220293u : (p.x < 136. ? 1717995601u : (p.x < 144. ? 1716868454u : (p.x < 152. ? 1753778278u : (p.x < 160. ? 1717986918u : (p.x < 168. ? 1143100758u : (p.x < 176. ? 286339074u : (p.x < 184. ? 286331697u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 71. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 823202065u : (p.x < 40. ? 1717986901u : (p.x < 48. ? 576018006u : (p.x < 56. ? 2236962u : (p.x < 64. ? 286331648u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1393627409u : (p.x < 104. ? 286479701u : (p.x < 112. ? 1093865779u : (p.x < 120. ? 1717916468u : (p.x < 128. ? 1164334422u : (p.x < 136. ? 1717986881u : (p.x < 144. ? 1717982566u : (p.x < 152. ? 2290649224u : (p.x < 160. ? 1717986918u : (p.x < 168. ? 861230438u : (p.x < 176. ? 286339649u : (p.x < 184. ? 286331697u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 70. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1717986901u : (p.x < 48. ? 576018006u : (p.x < 56. ? 139810u : (p.x < 64. ? 286331136u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 823202065u : (p.x < 104. ? 1430537556u : (p.x < 112. ? 823407445u : (p.x < 120. ? 1448362291u : (p.x < 128. ? 1449551462u : (p.x < 136. ? 1717986900u : (p.x < 144. ? 2004309606u : (p.x < 152. ? 1717987464u : (p.x < 160. ? 1717921382u : (p.x < 168. ? 1163220310u : (p.x < 176. ? 823198737u : (p.x < 184. ? 286331171u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 69. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1717986899u : (p.x < 48. ? 572872294u : (p.x < 56. ? 536879650u : (p.x < 64. ? 286331154u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 2254853460u : (p.x < 112. ? 286332008u : (p.x < 120. ? 856756497u : (p.x < 128. ? 1718126726u : (p.x < 136. ? 1717986918u : (p.x < 144. ? 2270516838u : (p.x < 152. ? 1718126728u : (p.x < 160. ? 1431656038u : (p.x < 168. ? 874653013u : (p.x < 176. ? 858854161u : (p.x < 184. ? 286331187u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 68. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1701209137u : (p.x < 48. ? 572876390u : (p.x < 56. ? 1073741856u : (p.x < 64. ? 286331155u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1393627409u : (p.x < 104. ? 2290509414u : (p.x < 112. ? 1091782536u : (p.x < 120. ? 1360073524u : (p.x < 128. ? 1720223878u : (p.x < 136. ? 1717986918u : (p.x < 144. ? 1716938342u : (p.x < 152. ? 1717986918u : (p.x < 160. ? 35804502u : (p.x < 168. ? 1109533218u : (p.x < 176. ? 322113809u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 67. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1717981457u : (p.x < 48. ? 36005478u : (p.x < 56. ? 807534592u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1695617297u : (p.x < 104. ? 2290648678u : (p.x < 112. ? 1683392648u : (p.x < 120. ? 1700091752u : (p.x < 128. ? 1720223880u : (p.x < 136. ? 2288412262u : (p.x < 144. ? 1431725672u : (p.x < 152. ? 1717986918u : (p.x < 160. ? 572675414u : (p.x < 168. ? 1377968640u : (p.x < 176. ? 288559377u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 66. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1431580945u : (p.x < 48. ? 1109747302u : (p.x < 56. ? 268645172u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1697714449u : (p.x < 104. ? 2290649190u : (p.x < 112. ? 2290649224u : (p.x < 120. ? 2290649224u : (p.x < 128. ? 1753778312u : (p.x < 136. ? 1751541350u : (p.x < 144. ? 1449551464u : (p.x < 152. ? 1717986917u : (p.x < 160. ? 572675429u : (p.x < 168. ? 1075978754u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 65. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1414795537u : (p.x < 48. ? 321214053u : (p.x < 56. ? 839979281u : (p.x < 64. ? 286331155u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2254901521u : (p.x < 104. ? 2290649224u : (p.x < 112. ? 2290648680u : (p.x < 120. ? 1451788424u : (p.x < 128. ? 1753778310u : (p.x < 136. ? 1717986917u : (p.x < 144. ? 1717995622u : (p.x < 152. ? 1431655766u : (p.x < 160. ? 572663074u : (p.x < 168. ? 805446178u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 64. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 1128468753u : (p.x < 48. ? 324363877u : (p.x < 56. ? 1141969169u : (p.x < 64. ? 286331155u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2290630929u : (p.x < 104. ? 1718126728u : (p.x < 112. ? 2290648678u : (p.x < 120. ? 1183352968u : (p.x < 128. ? 2290648678u : (p.x < 136. ? 1717982533u : (p.x < 144. ? 1717986918u : (p.x < 152. ? 572876390u : (p.x < 160. ? 572653568u : (p.x < 168. ? 318906914u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 63. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 858853649u : (p.x < 48. ? 289760852u : (p.x < 56. ? 856756497u : (p.x < 64. ? 286331188u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2290639633u : (p.x < 104. ? 2290649192u : (p.x < 112. ? 2290649224u : (p.x < 120. ? 1451788424u : (p.x < 128. ? 2290648676u : (p.x < 136. ? 859068023u : (p.x < 144. ? 1449551443u : (p.x < 152. ? 572876390u : (p.x < 160. ? 572522496u : (p.x < 168. ? 858923554u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 62. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 289760577u : (p.x < 56. ? 1127289617u : (p.x < 64. ? 286331188u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2290648337u : (p.x < 104. ? 2290649192u : (p.x < 112. ? 2257094792u : (p.x < 120. ? 1720223880u : (p.x < 128. ? 2290648644u : (p.x < 136. ? 288786568u : (p.x < 144. ? 1431725361u : (p.x < 152. ? 36001109u : (p.x < 160. ? 572522530u : (p.x < 168. ? 856900642u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 61. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 288428305u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 321212977u : (p.x < 56. ? 823211057u : (p.x < 64. ? 286462787u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2290648337u : (p.x < 104. ? 2290649224u : (p.x < 112. ? 2254997640u : (p.x < 120. ? 1753778312u : (p.x < 128. ? 2290648645u : (p.x < 136. ? 288852104u : (p.x < 144. ? 1431651089u : (p.x < 152. ? 1127503189u : (p.x < 160. ? 1107296853u : (p.x < 168. ? 286331715u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 60. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 288428305u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 844244017u : (p.x < 56. ? 286339075u : (p.x < 64. ? 322192145u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2290648081u : (p.x < 104. ? 2290649224u : (p.x < 112. ? 2290649224u : (p.x < 120. ? 2290649224u : (p.x < 128. ? 2290639942u : (p.x < 136. ? 286488712u : (p.x < 144. ? 1431572753u : (p.x < 152. ? 823211093u : (p.x < 160. ? 872423972u : (p.x < 168. ? 286331187u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 59. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 572797201u : (p.x < 56. ? 286339074u : (p.x < 64. ? 286462225u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2288542737u : (p.x < 104. ? 2290649224u : (p.x < 112. ? 2290649224u : (p.x < 120. ? 1753778312u : (p.x < 128. ? 2290504534u : (p.x < 136. ? 286340968u : (p.x < 144. ? 1431572753u : (p.x < 152. ? 286331733u : (p.x < 160. ? 841097764u : (p.x < 168. ? 1125191953u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 58. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 858853649u : (p.x < 56. ? 288637472u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1717986577u : (p.x < 104. ? 2290509414u : (p.x < 112. ? 1753778312u : (p.x < 120. ? 1718126694u : (p.x < 128. ? 1717978214u : (p.x < 136. ? 286331190u : (p.x < 144. ? 1716719889u : (p.x < 152. ? 286331189u : (p.x < 160. ? 575808563u : (p.x < 168. ? 1125191955u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 57. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 289555505u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1700090897u : (p.x < 104. ? 1717986918u : (p.x < 112. ? 1717986950u : (p.x < 120. ? 1449551462u : (p.x < 128. ? 878003557u : (p.x < 136. ? 286331153u : (p.x < 144. ? 1715540241u : (p.x < 152. ? 286331156u : (p.x < 160. ? 572670737u : (p.x < 168. ? 1125191956u : (p.x < 176. ? 286331155u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 56. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 289681681u : (p.x < 64. ? 289686289u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 1377977105u : (p.x < 104. ? 1716872533u : (p.x < 112. ? 1432774246u : (p.x < 120. ? 626353766u : (p.x < 128. ? 1145328997u : (p.x < 136. ? 286331155u : (p.x < 144. ? 1429278993u : (p.x < 152. ? 286331156u : (p.x < 160. ? 574828817u : (p.x < 168. ? 823202068u : (p.x < 176. ? 286331187u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 55. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 860033297u : (p.x < 64. ? 1109533235u : (p.x < 72. ? 286331716u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 572797201u : (p.x < 104. ? 572674594u : (p.x < 112. ? 1381324117u : (p.x < 120. ? 626349397u : (p.x < 128. ? 1717986642u : (p.x < 136. ? 286331156u : (p.x < 144. ? 1410404625u : (p.x < 152. ? 286331156u : (p.x < 160. ? 875639057u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331187u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 54. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 858853649u : (p.x < 64. ? 572662340u : (p.x < 72. ? 286330914u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 572723473u : (p.x < 104. ? 572662306u : (p.x < 112. ? 572662306u : (p.x < 120. ? 626139682u : (p.x < 128. ? 1717986642u : (p.x < 136. ? 286331155u : (p.x < 144. ? 856756497u : (p.x < 152. ? 286331188u : (p.x < 160. ? 319897873u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331203u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 53. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 35791393u : (p.x < 72. ? 288555008u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 607195409u : (p.x < 104. ? 608444962u : (p.x < 112. ? 572662306u : (p.x < 120. ? 572662306u : (p.x < 128. ? 1181115682u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331187u : (p.x < 160. ? 286535953u : (p.x < 168. ? 288559377u : (p.x < 176. ? 286331187u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 52. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 2236961u : (p.x < 72. ? 872415744u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 823202065u : (p.x < 104. ? 1091646259u : (p.x < 112. ? 572662308u : (p.x < 120. ? 1428300288u : (p.x < 128. ? 895903061u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 288633649u : (p.x < 168. ? 288370961u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 51. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 548u : (p.x < 72. ? 1073741824u : (p.x < 80. ? 286331155u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 49u : (p.x < 120. ? 1377959936u : (p.x < 128. ? 324359765u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 289686289u : (p.x < 168. ? 288359473u : 286331153u))))))))))))))))))))) : v;
	v = p.y == 50. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 823202065u : (p.x < 64. ? 2u : (p.x < 72. ? 0u : (p.x < 80. ? 286331188u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 49u : (p.x < 120. ? 1377959936u : (p.x < 128. ? 288642661u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 289407761u : (p.x < 168. ? 858783747u : (p.x < 176. ? 286331155u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 49. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 1091637553u : (p.x < 64. ? 2u : (p.x < 72. ? 0u : (p.x < 80. ? 286340162u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 131137u : (p.x < 120. ? 1413611520u : (p.x < 128. ? 286344533u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 318779665u : (p.x < 168. ? 858783747u : (p.x < 176. ? 321982739u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 48. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 1125191953u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 322191872u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 2236977u : (p.x < 120. ? 1414660096u : (p.x < 128. ? 286331989u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 872681745u : (p.x < 168. ? 1125401649u : (p.x < 176. ? 858853649u : (p.x < 184. ? 286339908u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 47. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 588321041u : (p.x < 64. ? 0u : (p.x < 72. ? 0u : (p.x < 80. ? 841097760u : (p.x < 88. ? 286331155u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 35791889u : (p.x < 120. ? 1431634464u : (p.x < 128. ? 286331685u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 876810513u : (p.x < 168. ? 856895761u : (p.x < 176. ? 856756499u : (p.x < 184. ? 286457892u : (p.x < 192. ? 286331155u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 46. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 1125191953u : (p.x < 64. ? 2u : (p.x < 72. ? 0u : (p.x < 80. ? 1377968674u : (p.x < 88. ? 286331156u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 622994193u : (p.x < 120. ? 1431655762u : (p.x < 128. ? 286331170u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 856756497u : (p.x < 168. ? 286331699u : (p.x < 176. ? 319885585u : (p.x < 184. ? 826277891u : (p.x < 192. ? 286331665u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 45. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 823202065u : (p.x < 64. ? 37u : (p.x < 72. ? 0u : (p.x < 80. ? 626348578u : (p.x < 88. ? 286331156u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1428308241u : (p.x < 120. ? 1431655765u : (p.x < 128. ? 286331170u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 823202065u : (p.x < 168. ? 286340163u : (p.x < 176. ? 286331185u : (p.x < 184. ? 323174403u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 44. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 84u : (p.x < 72. ? 0u : (p.x < 80. ? 626337106u : (p.x < 88. ? 286331155u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1381318929u : (p.x < 120. ? 626336293u : (p.x < 128. ? 286331682u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 858853649u : (p.x < 176. ? 286331185u : (p.x < 184. ? 321991441u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 43. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1363u : (p.x < 72. ? 536870912u : (p.x < 80. ? 844440866u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1381318929u : (p.x < 120. ? 1163206658u : (p.x < 128. ? 286331682u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 876810513u : (p.x < 184. ? 286339345u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 42. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 539317825u : (p.x < 72. ? 572661760u : (p.x < 80. ? 337990946u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 576017169u : (p.x < 120. ? 626336293u : (p.x < 128. ? 1125192738u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 1163146001u : (p.x < 184. ? 286343953u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 41. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 40265009u : (p.x < 72. ? 572661760u : (p.x < 80. ? 321213781u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1431655441u : (p.x < 120. ? 1431655765u : (p.x < 128. ? 1160844322u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 1163220291u : (p.x < 184. ? 286544915u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 40. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449476369u : (p.x < 72. ? 1377959970u : (p.x < 80. ? 321017173u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1431721489u : (p.x < 120. ? 1431655765u : (p.x < 128. ? 1163072324u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 1431655764u : (p.x < 184. ? 288642373u : (p.x < 192. ? 286331153u : 856756497u)))))))))))))))))))))))) : v;
	v = p.y == 39. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449398545u : (p.x < 72. ? 572653602u : (p.x < 80. ? 287462741u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1431725073u : (p.x < 120. ? 1112888661u : (p.x < 128. ? 844370195u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1125191953u : (p.x < 176. ? 1700091221u : (p.x < 184. ? 289756502u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 38. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449333009u : (p.x < 72. ? 572662354u : (p.x < 80. ? 289547554u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1716932881u : (p.x < 120. ? 844453478u : (p.x < 128. ? 877924625u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1430458641u : (p.x < 176. ? 1431655765u : (p.x < 184. ? 341141094u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 37. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449333009u : (p.x < 72. ? 572662354u : (p.x < 80. ? 288637474u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1717977361u : (p.x < 120. ? 1163220326u : (p.x < 128. ? 341119249u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431650577u : (p.x < 176. ? 1431655765u : (p.x < 184. ? 1163224678u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 36. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449398545u : (p.x < 72. ? 572661794u : (p.x < 80. ? 286331712u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1717973265u : (p.x < 120. ? 844453222u : (p.x < 128. ? 324342033u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431655185u : (p.x < 176. ? 1431655765u : (p.x < 184. ? 1431656038u : (p.x < 192. ? 286331156u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 35. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1449398545u : (p.x < 72. ? 35791362u : (p.x < 80. ? 286331184u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1717899537u : (p.x < 120. ? 324359510u : (p.x < 128. ? 288428305u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431655185u : (p.x < 176. ? 1431655765u : (p.x < 184. ? 1431656038u : (p.x < 192. ? 286331154u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 34. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1432621329u : (p.x < 72. ? 35792213u : (p.x < 80. ? 286331152u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1717834001u : (p.x < 120. ? 324359510u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431650577u : (p.x < 176. ? 1700091221u : (p.x < 184. ? 1431656038u : (p.x < 192. ? 286331189u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 33. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1432686865u : (p.x < 72. ? 572675413u : (p.x < 80. ? 286331155u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1430327569u : (p.x < 120. ? 289756501u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431646481u : (p.x < 176. ? 1700091221u : (p.x < 184. ? 1431656038u : (p.x < 192. ? 286331186u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 32. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1432686865u : (p.x < 72. ? 1109534037u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1429278993u : (p.x < 120. ? 286545237u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431638289u : (p.x < 176. ? 1431586133u : (p.x < 184. ? 1431655765u : (p.x < 192. ? 286331156u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 31. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1432686865u : (p.x < 72. ? 340927829u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 1410404625u : (p.x < 120. ? 286340181u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 1431572753u : (p.x < 176. ? 1410404660u : (p.x < 184. ? 1431655765u : (p.x < 192. ? 286331155u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 30. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1432760593u : (p.x < 72. ? 289690965u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 856756497u : (p.x < 120. ? 286331185u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 322113809u : (p.x < 176. ? 823202065u : (p.x < 184. ? 1112888660u : (p.x < 192. ? 286331153u : 286462225u)))))))))))))))))))))))) : v;
	v = p.y == 29. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1431712017u : (p.x < 72. ? 286545237u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 807543361u : (p.x < 192. ? 286331153u : 288559377u)))))))))))))))))))))))) : v;
	v = p.y == 28. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1431646481u : (p.x < 72. ? 286340181u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 288637713u : (p.x < 192. ? 286331153u : 318837009u)))))))))))))))))))))))) : v;
	v = p.y == 27. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1431650577u : (p.x < 72. ? 286331205u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 288428305u : (p.x < 192. ? 286331153u : 288559377u)))))))))))))))))))))))) : v;
	v = p.y == 26. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1431651089u : (p.x < 72. ? 286331155u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 289607953u : (p.x < 192. ? 286331153u : 286475025u)))))))))))))))))))))))) : v;
	v = p.y == 25. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1163215121u : (p.x < 72. ? 286331155u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 286331153u : (p.x < 192. ? 286331153u : 286344513u)))))))))))))))))))))))) : v;
	v = p.y == 24. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 878007057u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331153u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331153u : (p.x < 184. ? 286331153u : (p.x < 192. ? 286331153u : 286340163u)))))))))))))))))))))))) : v;
	v = p.y == 23. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 894792721u : 286331153u)))))))) : v;
	v = p.y == 22. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 341144593u : (p.x < 72. ? 286331153u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331665u : 286331153u)))))))))))))))))) : v;
	v = p.y == 21. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 290808849u : (p.x < 72. ? 286339345u : 286331153u))))))))) : v;
	v = p.y == 20. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 341066513u : 286331153u)))))))) : v;
	v = p.y == 19. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 339947793u : (p.x < 72. ? 286331153u : (p.x < 80. ? 823202065u : 286331153u)))))))))) : v;
	v = p.y == 18. ? 286331153u : v;
	v = p.y == 17. ? 286331153u : v;
	v = p.y == 16. ? 286331153u : v;
	v = p.y == 15. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286339345u : 286331153u))))))))) : v;
	v = p.y == 14. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 286331153u : (p.x < 72. ? 286487875u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 286331153u : (p.x < 136. ? 286331153u : (p.x < 144. ? 286331153u : (p.x < 152. ? 286331153u : (p.x < 160. ? 286331185u : (p.x < 168. ? 286331153u : (p.x < 176. ? 286331697u : 286331153u)))))))))))))))))))))) : v;
	v = p.y == 13. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 1949372689u : (p.x < 72. ? 286340471u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 286331153u : (p.x < 128. ? 823202065u : (p.x < 136. ? 286476151u : (p.x < 144. ? 286331153u : (p.x < 152. ? 1145324593u : (p.x < 160. ? 1199011700u : (p.x < 168. ? 1127499588u : (p.x < 176. ? 2004318068u : (p.x < 184. ? 286471236u : 286331153u))))))))))))))))))))))) : v;
	v = p.y == 12. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 2536575249u : (p.x < 72. ? 286340983u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 286331153u : (p.x < 104. ? 286331153u : (p.x < 112. ? 286331153u : (p.x < 120. ? 856756497u : (p.x < 128. ? 2004304657u : (p.x < 136. ? 2004326809u : (p.x < 144. ? 823211127u : (p.x < 152. ? 2576980343u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2541328793u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 1199012249u : (p.x < 192. ? 286331187u : 286331153u)))))))))))))))))))))))) : v;
	v = p.y == 11. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 286331153u : (p.x < 56. ? 286331153u : (p.x < 64. ? 2541187857u : (p.x < 72. ? 286340985u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 856756497u : (p.x < 104. ? 860123955u : (p.x < 112. ? 1145324339u : (p.x < 120. ? 2004300595u : (p.x < 128. ? 2576979831u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2272556953u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 860321656u : 286331187u)))))))))))))))))))))))) : v;
	v = p.y == 10. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 286331153u : (p.x < 40. ? 286331153u : (p.x < 48. ? 876884241u : (p.x < 56. ? 286331699u : (p.x < 64. ? 2576970803u : (p.x < 72. ? 286332825u : (p.x < 80. ? 286331153u : (p.x < 88. ? 286331153u : (p.x < 96. ? 2000892721u : (p.x < 104. ? 2576906137u : (p.x < 112. ? 2040108953u : (p.x < 120. ? 2576971671u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576840569u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2006555033u : 286331767u)))))))))))))))))))))))) : v;
	v = p.y == 9. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 286331153u : (p.x < 32. ? 1145254161u : (p.x < 40. ? 322188100u : (p.x < 48. ? 2040099601u : (p.x < 56. ? 1950840695u : (p.x < 64. ? 2576971639u : (p.x < 72. ? 286341017u : (p.x < 80. ? 286331153u : (p.x < 88. ? 856756497u : (p.x < 96. ? 2576840548u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576979832u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2576980377u : 286332025u)))))))))))))))))))))))) : v;
	v = p.y == 8. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 286331153u : (p.x < 24. ? 2004304657u : (p.x < 32. ? 2576840567u : (p.x < 40. ? 2004383641u : (p.x < 48. ? 2576971639u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2576980377u : (p.x < 72. ? 286345081u : (p.x < 80. ? 286331153u : (p.x < 88. ? 2000752913u : (p.x < 96. ? 2576980359u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2006555033u : 286331188u)))))))))))))))))))))))) : v;
	v = p.y == 7. ? (p.x < 8. ? 286339891u : (p.x < 16. ? 1145320241u : (p.x < 24. ? 2576980343u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2004318073u : (p.x < 72. ? 860321655u : (p.x < 80. ? 286339891u : (p.x < 88. ? 2575791155u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 932813209u : 858993459u)))))))))))))))))))))))) : v;
	v = p.y == 6. ? (p.x < 8. ? 1145337719u : (p.x < 16. ? 2576840564u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2006555033u : (p.x < 64. ? 1199011703u : (p.x < 72. ? 1145324612u : (p.x < 80. ? 1145337719u : (p.x < 88. ? 2576980343u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2004457881u : 2004326263u)))))))))))))))))))))))) : v;
	v = p.y == 5. ? (p.x < 8. ? 2004318071u : (p.x < 16. ? 2004318071u : (p.x < 24. ? 2576980375u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2308544921u : (p.x < 64. ? 1165457272u : (p.x < 72. ? 1967408196u : (p.x < 80. ? 2004318071u : (p.x < 88. ? 2576975735u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2004457881u : 2004318071u)))))))))))))))))))))))) : v;
	v = p.y == 4. ? (p.x < 8. ? 2004318071u : (p.x < 16. ? 2541188983u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2040109465u : (p.x < 72. ? 2004304999u : (p.x < 80. ? 2574743415u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2004326809u : 2004318071u)))))))))))))))))))))))) : v;
	v = p.y == 3. ? (p.x < 8. ? 2004318071u : (p.x < 16. ? 2576840567u : (p.x < 24. ? 2576980377u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2576980377u : (p.x < 72. ? 2576840569u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2576980377u : 2004318071u)))))))))))))))))))))))) : v;
	v = p.y == 2. ? (p.x < 8. ? 2004453241u : (p.x < 16. ? 2004318071u : (p.x < 24. ? 2576980375u : (p.x < 32. ? 2576980377u : (p.x < 40. ? 2576980377u : (p.x < 48. ? 2576980377u : (p.x < 56. ? 2576980377u : (p.x < 64. ? 2576980377u : (p.x < 72. ? 2576980377u : (p.x < 80. ? 2576980377u : (p.x < 88. ? 2576980377u : (p.x < 96. ? 2576980377u : (p.x < 104. ? 2576980377u : (p.x < 112. ? 2576980377u : (p.x < 120. ? 2576980377u : (p.x < 128. ? 2576980377u : (p.x < 136. ? 2576980377u : (p.x < 144. ? 2576980377u : (p.x < 152. ? 2576980377u : (p.x < 160. ? 2576980377u : (p.x < 168. ? 2576980377u : (p.x < 176. ? 2576980377u : (p.x < 184. ? 2576980377u : (p.x < 192. ? 2576980377u : 2576979833u)))))))))))))))))))))))) : v;
	v = p.y == 1. ? 2576980377u : v;
	v = p.y == 0. ? 2576980377u : v;
    v = p.x >= 0. && p.x < 200. ? v : 0u;

    float i = float((v >> uint(4. * p.x)) & 15u);
    vec3 color = vec3(0.26, 0.31, 0.16);
    color = i == 1. ? vec3(0.1, 0.24, 0.46) : color;
    color = i == 2. ? vec3(0.4, 0.43, 0.26) : color;
    color = i == 3. ? vec3(0.28, 0.35, 0.47) : color;
    color = i == 4. ? vec3(0.35, 0.38, 0.42) : color;
    color = i == 5. ? vec3(0.51, 0.45, 0.36) : color;
    color = i == 6. ? vec3(0.71, 0.63, 0.53) : color;
    color = i == 7. ? vec3(0.64, 0.66, 0.7) : color;
    color = i == 8. ? vec3(0.88, 0.79, 0.69) : color;
    color = i == 9. ? vec3(0.86, 0.87, 0.89) : color;
    
    return vec4(color, 1.0);
}

// Function 546
vec4 myText(vec2 v)
{
    if (v.y < 0.5 || v.y > 22.5 || v.x > 2.5) {
        return vec4(0x202020);
    }
    if (v.y < 1.5) {
        if (v.x < 0.5)  return vec4(0x4c6f72, 0x656d20, 0x697073, 0x756d20);
        if (v.x < 1.5)  return vec4(0x646f6c, 0x6f7220, 0x736974, 0x20616d);
        if (v.x < 2.5)  return vec4(0x65742c, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 2.5) {
        if (v.x < 0.5)  return vec4(0x202063, 0x6f6e73, 0x656374, 0x657475);
        if (v.x < 1.5)  return vec4(0x722061, 0x646970, 0x697363, 0x696e67);
        if (v.x < 2.5)  return vec4(0x202020);
    }
    if (v.y < 3.5) {
        if (v.x < 0.5)  return vec4(0x656c69, 0x742c20, 0x736564, 0x20646f);
        if (v.x < 1.5)  return vec4(0x206569, 0x75736d, 0x6f6420, 0x74656d);
        if (v.x < 2.5)  return vec4(0x706f72, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 4.5) {
        if (v.x < 0.5)  return vec4(0x202069, 0x6e6369, 0x646964, 0x756e74);
        if (v.x < 1.5)  return vec4(0x207574, 0x206c61, 0x626f72, 0x652065);
        if (v.x < 2.5)  return vec4(0x742020, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 5.5) {
        if (v.x < 0.5)  return vec4(0x202020, 0x20646f, 0x6c6f72, 0x65206d);
        if (v.x < 1.5)  return vec4(0x61676e, 0x612061, 0x6c6971, 0x75612e);
        if (v.x < 2.5)  return vec4(0x202020);
    }
    if (v.y < 6.5) {
        if (v.x < 0.5)  return vec4(0x202055, 0x742065, 0x6e696d, 0x206164);
        if (v.x < 1.5)  return vec4(0x206d69, 0x6e696d, 0x207665, 0x6e6961);
        if (v.x < 2.5)  return vec4(0x6d2c20, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 7.5) {
        if (v.x < 0.5)  return vec4(0x207175, 0x697320, 0x6e6f73, 0x747275);
        if (v.x < 1.5)  return vec4(0x642065, 0x786572, 0x636974, 0x617469);
        if (v.x < 2.5)  return vec4(0x6f6e20, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 8.5) {
        if (v.x < 0.5)  return vec4(0x202075, 0x6c6c61, 0x6d636f, 0x206c61);
        if (v.x < 1.5)  return vec4(0x626f72, 0x697320, 0x6e6973, 0x692075);
        if (v.x < 2.5)  return vec4(0x742020, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 9.5) {
        if (v.x < 0.5)  return vec4(0x202020, 0x616c69, 0x717569, 0x702065);
        if (v.x < 1.5)  return vec4(0x782065, 0x612063, 0x6f6d6d, 0x6f646f);
        if (v.x < 2.5)  return vec4(0x202020);
    }
    if (v.y < 10.5) {
        if (v.x < 0.5)  return vec4(0x202020, 0x202020, 0x202020, 0x636f6e);
        if (v.x < 1.5)  return vec4(0x736571, 0x756174, 0x2e2020, 0x202020);
        if (v.x < 2.5)  return vec4(0x202020);
    }
    if (v.y < 13.5) {
        return vec4(0x202020);
    }
    if (v.y < 14.5) {
        if (v.x < 0.5)  return vec4(0x204475, 0x697320, 0x617574, 0x652069);
        if (v.x < 1.5)  return vec4(0x727572, 0x652064, 0x6f6c6f, 0x722069);
        if (v.x < 2.5)  return vec4(0x6e2020, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 15.5) {
        if (v.x < 0.5)  return vec4(0x726570, 0x726568, 0x656e64, 0x657269);
        if (v.x < 1.5)  return vec4(0x742069, 0x6e2076, 0x6f6c75, 0x707461);
        if (v.x < 2.5)  return vec4(0x746520, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 16.5) {
        if (v.x < 0.5)  return vec4(0x207665, 0x6c6974, 0x206573, 0x736520);
        if (v.x < 1.5)  return vec4(0x63696c, 0x6c756d, 0x20646f, 0x6c6f72);
        if (v.x < 2.5)  return vec4(0x652020, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 17.5) {
        if (v.x < 0.5)  return vec4(0x206575, 0x206675, 0x676961, 0x74206e);
        if (v.x < 1.5)  return vec4(0x756c6c, 0x612070, 0x617269, 0x617475);
        if (v.x < 2.5)  return vec4(0x722e20, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 18.5) {
        if (v.x < 0.5)  return vec4(0x202045, 0x786365, 0x707465, 0x757220);
        if (v.x < 1.5)  return vec4(0x73696e, 0x74206f, 0x636361, 0x656361);
        if (v.x < 2.5)  return vec4(0x742020, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 19.5) {
        if (v.x < 0.5)  return vec4(0x202063, 0x757069, 0x646174, 0x617420);
        if (v.x < 1.5)  return vec4(0x6e6f6e, 0x207072, 0x6f6964, 0x656e74);
        if (v.x < 2.5)  return vec4(0x2c2020, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 20.5) {
        if (v.x < 0.5)  return vec4(0x207375, 0x6e7420, 0x696e20, 0x63756c);
        if (v.x < 1.5)  return vec4(0x706120, 0x717569, 0x206f66, 0x666963);
        if (v.x < 2.5)  return vec4(0x696120, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 21.5) {
        if (v.x < 0.5)  return vec4(0x202064, 0x657365, 0x72756e, 0x74206d);
        if (v.x < 1.5)  return vec4(0x6f6c6c, 0x697420, 0x616e69, 0x6d2069);
        if (v.x < 2.5)  return vec4(0x642020, 0x202020, 0x202020, 0x202020);
    }
    if (v.y < 22.5) {
        if (v.x < 0.5)  return vec4(0x202020, 0x202020, 0x202065, 0x737420);
        if (v.x < 1.5)  return vec4(0x6c6162, 0x6f7275, 0x6d2e20, 0x202020);
        if (v.x < 2.5)  return vec4(0x202020);
    }

    return vec4(0x202020);
}

// Function 547
mat3 glyph_8_9_lowercase(float g) {
    // lowercase ==================================================
    GLYPH( 97)0,0,0,1110,10010,10010,1101,0,0);
    GLYPH( 98)0,100000,100000,111100,100010,100010,111100,0,0);
    GLYPH( 99)0,0,0,11100,100000,100010,11100,0,0);
    GLYPH(100)10,10,10,11010,100110,100010,11101,0,0);
    GLYPH(101)0,0,111000,1000100,1111100,1000000,111100,0,0);
    GLYPH(102)11000,100100,100000,111000,100000,100000,100000,0,0);
    GLYPH(103)0,11000,100100,100100,11100,100100,11000,0,0);
    GLYPH(104)100000,100000,100000,101100,110010,100010,100010,0,0);
    GLYPH(105)0,10000,0,10000,10000,10000,11000,0,0);
    GLYPH(106)100,0,100,100,100,100100,11000,0,0);
    GLYPH(107)1000000,1000000,1000000,1001000,1110000,1010000,1001000,0,0);
    GLYPH(108)100000,100000,100000,100000,100000,100000,11100,0,0);
    GLYPH(109)0,0,100100,1011010,1000010,1000010,1000010,0,0);
    GLYPH(110)0,0,1011000,1100100,1000010,1000010,1000010,0,0);
    GLYPH(111)0,0,11100,100010,100010,100010,11100,0,0);
    GLYPH(112)0,0,1011100,1100010,1100010,1011100,1000000,1000000,0);
    GLYPH(113)0,0,111010,1000110,1000110,111010,10,11,0);
    GLYPH(114)0,0,101100,110010,100000,100000,100000,0,0);
    GLYPH(115)0,0,11100,100000,11100,10,100010,11100,0);
    GLYPH(116)10000,10000,11100,10000,10000,10010,1100,0,0);
    GLYPH(117)0,0,100010,100010,100010,100010,11100,0,0);
    GLYPH(118)0,0,100010,100010,10100,10100,1000,0,0);
    GLYPH(119)0,0,100010,100010,101010,101010,10100,0,0);
    GLYPH(120)0,0,100010,10100,1000,10100,100010,0,0);
    GLYPH(121)0,0,100010,100010,100110,11010,10,11100,0);
    GLYPH(122)0,0,111110,100,1000,10000,111110,0,0);
    return mat3(0);
}

// Function 548
float Glyph0(const in vec2 uv)
{
    const vec2  vP0 = vec2 ( 0.112, 0.056 );
    const vec2  vP1 = vec2 ( 0.136, 0.026 );
    const vec2  vP2 = vec2 ( 0.108, 0.022 );
    const vec2  vP3 = vec2 ( 0.083, 0.017 ); 
    const vec2  vP4 = vec2 ( 0.082, 0.036 ); 
    const vec2  vP5 = vec2 ( 0.088, 0.062 ); 
    const vec2  vP6 = vec2 ( 0.115, 0.086 ); 
    const vec2  vP7 = vec2 ( 0.172, 0.147 ); 
    const vec2  vP8 = vec2 ( 0.100, 0.184 ); 
    const vec2  vP9 = vec2 ( 0.034, 0.206 ); 
    const vec2 vP10 = vec2 ( 0.021, 0.160 ); 
    const vec2 vP11 = vec2 ( 0.011, 0.114 ); 
    const vec2 vP12 = vec2 ( 0.052, 0.112 ); 
    const vec2 vP13 = vec2 ( 0.070, 0.108 ); 
    const vec2 vP14 = vec2 ( 0.075, 0.126 );
    const vec2 vP15 = vec2 ( 0.049, 0.124 );
    const vec2 vP16 = vec2 ( 0.047, 0.148 );
    const vec2 vP17 = vec2 ( 0.046, 0.169 );
    const vec2 vP18 = vec2 ( 0.071, 0.171 );
    const vec2 vP19 = vec2 ( 0.098, 0.171 ); 
    const vec2 vP20 = vec2 ( 0.097, 0.143 ); 
    const vec2 vP21 = vec2 ( 0.100, 0.118 ); 
    const vec2 vP22 = vec2 ( 0.080, 0.100 ); 
    const vec2 vP23 = vec2 ( 0.055, 0.083 ); 
    const vec2 vP24 = vec2 ( 0.050, 0.052 ); 
    const vec2 vP25 = vec2 ( 0.052, 0.004 ); 
    const vec2 vP26 = vec2 ( 0.107, 0.010 ); 
    const vec2 vP27 = vec2 ( 0.148, 0.011 ); 
    const vec2 vP28 = vec2 ( 0.140, 0.041 ); 
    const vec2 vP29 = vec2 ( 0.139, 0.069 ); 

    float fDist = 1.0;

	fDist = min( fDist, InCurve2(vP6,vP7,vP8, uv) );
    fDist = min( fDist, InCurve2(vP8,vP9,vP10, uv) );
	fDist = min( fDist, InCurve2(vP10,vP11,vP12, uv) );
    fDist = min( fDist, InCurve2(vP12,vP13,vP14, uv) );
	fDist = min( fDist, InCurve(vP14,vP15,vP16, uv) );
    fDist = min( fDist, InCurve(vP16,vP17,vP18, uv) );
    fDist = min( fDist, InCurve(vP18,vP19,vP20, uv) );
    fDist = min( fDist, InCurve(vP20,vP21,vP22, uv) );
	fDist = min( fDist, InCurve2(vP22,vP23,vP24, uv) );
    fDist = min( fDist, InCurve2(vP24,vP25,vP26, uv) );
    fDist = min( fDist, InCurve2(vP26,vP27,vP28, uv) );
    fDist = min( fDist, InCurve2(vP28,vP29,vP0, uv) );
	fDist = min( fDist, InCurve(vP0,vP1,vP2, uv) );
	fDist = min( fDist, InCurve(vP2,vP3,vP4, uv) );
    fDist = min( fDist, InCurve(vP4,vP5,vP6, uv) );


    fDist = min( fDist, InTri(vP0, vP1, vP28, uv) );
	fDist = min( fDist, InQuad(vP26, vP1, vP2, vP3, uv) );
    fDist = min( fDist, InTri(vP3, vP4, vP24, uv) );
    fDist = min( fDist, InTri(vP4, vP5, vP24, uv) );
    fDist = min( fDist, InTri(vP24, vP5, vP22, uv) );
    fDist = min( fDist, InTri(vP5, vP6, vP22, uv) );
    fDist = min( fDist, InTri(vP22, vP6, vP21, uv) );
    fDist = min( fDist, InTri(vP6, vP8, vP21, uv) );
    fDist = min( fDist, InTri(vP21, vP8, vP20, uv) );
    fDist = min( fDist, InTri(vP20, vP8, vP19, uv) );
    fDist = min( fDist, InTri(vP19, vP8, vP18, uv) );
    fDist = min( fDist, InTri(vP18, vP8, vP10, uv) );
    fDist = min( fDist, InTri(vP10, vP16, vP17, uv) );
    fDist = min( fDist, InTri(vP10, vP15, vP16, uv) );
    fDist = min( fDist, InTri(vP10, vP12, vP16, uv) );
    fDist = min( fDist, InTri(vP12, vP14, vP15, uv) );

    return fDist;
}

// Function 549
float FontTexDf (vec2 p)
{
  return texture (txFnt, mod ((bIdV + p) * (1. / 16.), 1.)).a - 0.5 + 1. / 256.;
}

// Function 550
vec3 SampleTexture( const in float fTexture, const in vec2 _vUV )
{
    vec3 col = vec3(1.0, 0.0, 1.0);
    vec2 vUV = _vUV;
    
    vec2 vSize = vec2(64.0);
    float fPersistence = 0.8;
	float fNoise2Freq = 0.5;

	if(fTexture == TEX_NUKAGE3)
	{
        float fTest = fract(floor(iTime * 6.0) * (1.0 / 3.0));
        if( fTest < 0.3 )
        {
	        vUV += 0.3 * vSize;
        }
        else if(fTest < 0.6)
        {
            vUV = vUV.yx - 0.3; 
        }
        else
        {
            vUV = vUV + 0.45;
        }
	}
	
	     if(fTexture == TEX_NUKAGE3) { fPersistence = 1.0; }
	else if(fTexture == TEX_F_SKY1) { vSize = vec2(256.0, 128.0); fNoise2Freq = 0.3; }
    else if(fTexture == TEX_FLOOR7_1) { vSize = vec2(64.0, 32.0); fPersistence = 1.0; }	
    else if(fTexture == TEX_FLAT5_5) { fPersistence = 3.0; }
    else if(fTexture == TEX_FLOOR4_8) { fPersistence = 0.3; }
    else if(fTexture == TEX_CEIL3_5) { fPersistence = 0.9; }	
    else if(fTexture == TEX_FLAT14) { fPersistence = 2.0; }
    else if(fTexture == TEX_DOOR3) { vSize = vec2(64.0, 72.0); }	
    else if(fTexture == TEX_LITE3) { vSize = vec2(32.0, 128.0); }	
    else if(fTexture == TEX_STARTAN3) { vSize = vec2(128.0); fPersistence = 1.0; }	
	else if(fTexture == TEX_BROWN1) { vSize = vec2(128.0); fPersistence = 0.7; }	
    else if(fTexture == TEX_DOORSTOP) { vSize = vec2(8.0, 128.0); fPersistence = 0.7; }
    else if(fTexture == TEX_COMPUTE2) { vSize = vec2(256.0, 56.0); fPersistence = 1.5; }
    else if(fTexture == TEX_STEP6) { vSize = vec2(32.0, 16.0); fPersistence = 0.9; }
    else if(fTexture == TEX_SUPPORT2) { vSize = vec2(64.0, 128.0); }
    else if(fTexture == TEX_DOORTRAK) { vSize = vec2(8.0, 128.0); }
#ifdef ENABLE_SPRITES	
	else if(fTexture == TEX_BAR1A) { vSize = vec2(23.0, 32.0); }
	else if(fTexture == TEX_PLAYW) { vSize = vec2(57.0, 22.0); fPersistence = 1.0; }
#endif
	
#ifdef PREVIEW
	     if(fTexture == TEX_DOOR3) {	vSize = vec2(128.0, 128.0); }	
	else if(fTexture == TEX_COMPUTE2) { vSize = vec2(256.0, 64.0); }
#ifdef ENABLE_SPRITES	
	else if(fTexture == TEX_BAR1A) { vSize = vec2(32.0, 32.0); }
	else if(fTexture == TEX_PLAYW) { vSize = vec2(64.0, 32.0); }	
#endif
#endif
	
	
#ifdef PREVIEW
    vec2 vTexCoord = floor(fract(vUV) * vSize);
#else
    vec2 vTexCoord = fract(vUV / vSize) * vSize;
    #ifdef PIXELATE_TEXTURES
    vTexCoord = floor(vTexCoord);
    #endif
    vTexCoord.y = vSize.y - vTexCoord.y - 1.0;
#endif
	float fRandom = fbm( vTexCoord, fPersistence );
	float fHRandom = noise1D(vTexCoord.x * fNoise2Freq) - ((vTexCoord.y) / vSize.y);
    
	     if(fTexture == TEX_NUKAGE3) 	col = TexNukage3( vTexCoord, fRandom );
	else if(fTexture == TEX_F_SKY1) 	col = TexFSky1( vTexCoord, fRandom, fHRandom );
    else if(fTexture == TEX_FLOOR7_1) 	col = TexFloor7_1( vTexCoord, fRandom );
    else if(fTexture == TEX_FLAT5_5) 	col = TexFlat5_5( vTexCoord, fRandom );
    else if(fTexture == TEX_FLOOR4_8) 	col = TexFloor4_8( vTexCoord, fRandom );
    else if(fTexture == TEX_CEIL3_5) 	col = TexCeil3_5( vTexCoord, fRandom );
	else if(fTexture == TEX_FLAT14) 	col = TexFlat14( vTexCoord, fRandom );
	else if(fTexture == TEX_DOOR3) 		col = TexDoor3( vTexCoord, fRandom, fHRandom );
	else if(fTexture == TEX_LITE3) 		col = TexLite3( vTexCoord );
    else if(fTexture == TEX_STARTAN3) 	col = TexStartan3( vTexCoord, fRandom );
    else if(fTexture == TEX_BROWN1) 	col = TexBrown1( vTexCoord, fRandom, fHRandom );
    else if(fTexture == TEX_DOORSTOP) 	col = TexDoorstop( vTexCoord, fRandom );
    else if(fTexture == TEX_COMPUTE2) 	col = TexCompute2( vTexCoord, fRandom );
    else if(fTexture == TEX_STEP6) 		col = TexStep6( vTexCoord, fRandom, fHRandom );
    else if(fTexture == TEX_SUPPORT2) 	col = TexSupport2( vTexCoord, fRandom );
	else if(fTexture == TEX_DOORTRAK) 	col = TexDoorTrak( vTexCoord, fRandom );
	else if(fTexture == TEX_BROWN144) 	col = TexBrown144( vTexCoord, fRandom, fHRandom );
#ifdef ENABLE_SPRITES	
	else if(fTexture == TEX_BAR1A) 		col = TexBar1A( vTexCoord, fRandom, fHRandom );
	else if(fTexture == TEX_PLAYW) 		col = TexPlayW( vTexCoord, fRandom, fHRandom );	
#endif
	
    #ifdef QUANTIZE_TEXTURES
    col = Quantize(col);
    #endif

    return col;
}

// Function 551
float NumFont_Rect( vec2 vPos, vec2 bl, vec2 tr )
{
	if ( all( greaterThanEqual( vPos, bl ) ) &&
        all( lessThanEqual( vPos, tr ) ) )
    {
        return 1.0;
    }
        
    return 0.0;
}

// Function 552
mat3 glyph_8_9_uppercase(float g) {
    // uppercase ==================================================
    GLYPH(65)000011000,000100100,001000010,001111110,001000010,001000010,001000010,0,0);
    GLYPH(66)011111000,001000100,001000100,001111000,001000100,001000100,011111000,0,0);
    GLYPH(67)000011100,000100010,001000000,001000000,001000000,000100010,000011100,0,0);
    GLYPH(68)011111000,001000100,001000010,001000010,001000010,001000100,011111000,0,0);
    GLYPH(69)001111110,001000000,001000000,001111000,001000000,001000000,001111110,0,0);
    GLYPH(70)001111110,001000000,001000000,001111000,001000000,001000000,001000000,0,0);
    GLYPH(71)000111100,001000010,001000000,001001110,001000010,001000010,000111100,0,0);
    GLYPH(72)001000010,001000010,001000010,001111110,001000010,001000010,001000010,0,0);
    GLYPH(73)000111000,000010000,000010000,000010000,000010000,000010000,000111000,0,0);
    GLYPH(74)001111110,000000100,000000100,000000100,000000100,001000100,000111000,0,0);
    GLYPH(75)001000100,001001000,001010000,001110000,001001000,001000100,001000010,0,0);
    GLYPH(76)000100000,000100000,000100000,000100000,000100000,000100000,000111111,0,0);
    GLYPH(77)001000010,001100110,001011010,001000010,001000010,001000010,001000010,0,0);
    GLYPH(78)001000010,001100010,001010010,001001010,001000110,001000010,001000010,0,0);
    GLYPH(79)000111100,001000010,001000010,001000010,001000010,001000010,000111100,0,0);
    GLYPH(80)011111100,001000010,001000010,001111100,001000000,001000000,001000000,0,0);
    GLYPH(81)000111100,001000010,001000010,001000010,001001010,001000110,000111101,0,0);
    GLYPH(82)001111100,001000010,001000010,001111100,001001000,001000100,001000010,0,0);
    GLYPH(83)000111100,001000010,001000000,000111100,000000010,001000010,000111100,0,0);
    GLYPH(84)001111100,000010000,000010000,000010000,000010000,000010000,000010000,0,0);
    GLYPH(85)001000010,001000010,001000010,001000010,001000010,001000010,000111100,0,0);
    GLYPH(86)001000010,001000010,001000010,001000010,000100100,000100100,000011000,0,0);
    GLYPH(87)001000011,001000011,001000011,001000011,001011011,001100111,000100100,0,0);
    GLYPH(88)001000100,001000100,001000100,000111000,000101000,001000100,001000100,0,0);
    GLYPH(89)001000100,001000100,001000100,000111000,000010000,000010000,000010000,0,0);
    GLYPH(90)001111110,000000010,000000100,000011000,000100000,001000000,001111110,0,0);
    return mat3(0);
}

// Function 553
vec4 textureNoTile_3weights( sampler2D samp, in vec2 uv )
{
    vec4 res = vec4(0.);
    int sampleCnt = 0; // debug vis
    
    // compute per-tile integral and fractional uvs.
    // flip uvs for 'odd' tiles to make sure tex samples are coherent
    vec2 fuv = mod( uv, 2. ), iuv = uv - fuv;
    vec3 BL_one = vec3(0.,0.,1.); // xy = bot left coords, z = 1
    if( fuv.x >= 1. ) fuv.x = 2.-fuv.x, BL_one.x = 2.;
    if( fuv.y >= 1. ) fuv.y = 2.-fuv.y, BL_one.y = 2.;
    
    
    // weight orthogonal to diagonal edge = 3rd texture sample
    vec2 iuv3;
    float w3 = (fuv.x+fuv.y) - 1.;
    if( w3 < 0. ) iuv3 = iuv + BL_one.xy, w3 = -w3; // bottom left corner, offset negative, weight needs to be negated
    else iuv3 = iuv + BL_one.zz; // use transform from top right corner
    
    #if 0
    
    //
    // Original calculation of w3
    //
    
    w3 = smoothstep(BLEND_WIDTH, 1.-BLEND_WIDTH, w3);
    #else
    
    //
    // Modified calculation of w3
    //
    
    w3 = smoothstep(BLEND_WIDTH, 1.-BLEND_WIDTH, pow(1. - min(length(1. - fuv), length(fuv)) , 1.5));
    
    #endif
    
    // if third sample doesnt dominate, take first two
    if( w3 < 0.999 )
    {
        // use weight along long diagonal edge
        float w12 = dot(fuv,vec2(.5,-.5)) + .5;
        w12 = smoothstep(1.125*BLEND_WIDTH, 1.-1.125*BLEND_WIDTH, w12);

        // take samples from texture for each side of diagonal edge
        if( w12 > 0.001 ) res +=     w12  * texture( samp, transformUVs( iuv + BL_one.zy, uv ) ), sampleCnt++;
        if( w12 < 0.999 ) res += (1.-w12) * texture( samp, transformUVs( iuv + BL_one.xz, uv ) ), sampleCnt++;
    }
    
	// first two samples aren't dominating, take third
    if( w3 > 0.001 ) res = mix( res, texture( samp, transformUVs( iuv3, uv ) ), w3 ), sampleCnt++;

    
    // debug vis: colour based on num samples taken for vis purposes
    if( iMouse.z > 0. )
    {
        if( sampleCnt == 1 ) res.rb *= .25;
        if( sampleCnt == 2 ) res.b *= .25;
        if( sampleCnt == 3 ) res.gb *= .25;
    }
    
    return res;
}

// Function 554
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

// Function 555
vec3 textureBox( sampler2D sam, in vec3 pos, in vec3 nor )
{
    vec3 w = abs(nor);
    return (w.x*texture( sam, pos.yz ).xyz + 
            w.y*texture( sam, pos.zx ).xyz + 
            w.z*texture( sam, pos.xy ).xyz ) / (w.x+w.y+w.z);
}

// Function 556
vec3 texToVoxCoord(vec2 textelCoord, vec3 offset) {

    vec2 packedChunkSize= packedChunkSize;
	vec3 voxelCoord = offset;
    voxelCoord.xy += unswizzleChunkCoord(textelCoord / packedChunkSize);
    voxelCoord.z += mod(textelCoord.x, packedChunkSize.x) + packedChunkSize.x * mod(textelCoord.y, packedChunkSize.y);
    return voxelCoord;
}

// Function 557
vec4 ScrollText(vec2 xuv)
{
    xtp = xuv / FONT_SIZE1;  // set font size
    xtp.x = 2.0*(xtp.x -4. +mod(xtime*SCROLL_SPEED, SCROLL_LEN));
    xtp.y = xtp.y +1.7 +SIN_AMP*sin(xtp.x*SIN_FREQ +xtime*SIN_SPEED);
    float c = 0.0;
    
    S(32.);S(32.);S(32.);S(32.);S(32.);S(32.);S(72.);S(101.);S(108.);S(108.);S(111.);S(32.);
    S(115.);S(104.);S(97.);S(100.);S(101.);S(114.);S(116.);S(111.);S(121.);S(32.);S(33.);S(33.);
    S(32.);S(84.);S(104.);S(105.);S(115.);S(32.);S(105.);S(115.);S(32.);S(97.);S(32.);S(99.);
    S(117.);S(98.);S(101.);S(32.);S(119.);S(105.);S(116.);S(104.);S(32.);S(97.);S(32.);S(115.);
    S(104.);S(97.);S(100.);S(101.);S(114.);S(32.);S(109.);S(97.);S(112.);S(112.);S(101.);S(100.);
    S(32.);S(111.);S(110.);S(32.);S(101.);S(97.);S(99.);S(104.);S(32.);S(102.);S(97.);S(99.);
    S(101.);S(32.);S(99.);S(111.);S(100.);S(101.);S(100.);S(32.);S(105.);S(110.);S(32.);S(49.);
    S(57.);S(57.);S(48.);S(32.);S(97.);S(109.);S(105.);S(103.);S(97.);S(32.);S(114.);S(101.);
    S(116.);S(114.);S(111.);S(32.);S(115.);S(116.);S(121.);S(108.);S(101.);S(32.);S(46.);S(46.);
    S(46.);S(46.);S(32.);S(73.);S(32.);S(106.);S(117.);S(115.);S(116.);S(32.);S(109.);S(105.);
    S(120.);S(101.);S(100.);S(32.);S(115.);S(101.);S(118.);S(101.);S(114.);S(97.);S(108.);S(32.);
    S(115.);S(104.);S(97.);S(100.);S(101.);S(114.);S(115.);S(32.);S(116.);S(111.);S(103.);S(101.);
    S(116.);S(104.);S(101.);S(114.);S(32.);S(46.);S(46.);S(46.);S(46.);S(32.);S(82.);S(101.);
    S(97.);S(100.);S(32.);S(116.);S(104.);S(101.);S(32.);S(99.);S(111.);S(100.);S(101.);S(32.);
    S(102.);S(111.);S(114.);S(32.);S(99.);S(114.);S(101.);S(100.);S(105.);S(116.);S(115.);S(32.);
    S(46.);S(46.);S(46.);S(46.);S(46.);S(32.);S(73.);S(32.);S(99.);S(111.);S(100.);S(101.);
    S(100.);S(32.);S(116.);S(104.);S(105.);S(115.);S(32.);S(98.);S(101.);S(99.);S(97.);S(117.);
    S(115.);S(101.);S(32.);S(73.);S(32.);S(108.);S(111.);S(118.);S(101.);S(32.);S(99.);S(117.);
    S(98.);S(101.);S(115.);S(32.);S(97.);S(110.);S(100.);S(32.);S(65.);S(109.);S(105.);S(103.);
    S(97.);S(32.);S(46.);S(46.);S(46.);S(46.);S(46.);S(32.);S(72.);S(105.);S(109.);S(114.);
    S(101.);S(100.);S(32.);S(46.);S(46.);S(46.);S(46.);S(32.);S(69.);S(79.);S(84.);S(32.);
    S(46.);S(46.);S(46.);S(46.);
    return c * vec4(xpos, 0.5+0.5*sin(2.0*xtime),1.0);
}

// Function 558
vec3 SampleInterpolationTexturePixel (vec2 pixel)
{
    pixel = mod(pixel, 2.0);
    
    // Used only for the interpolation buttons!
    return vec3(
        pixel.x / 3.0,
    	pixel.y / 3.0,
        mod(pixel.x + pixel.y, 2.0)
    );    
}

// Function 559
void glyph_4()
{
  MoveTo(0.1*x+0.6*y);    
  LineTo(x*1.6+2.0*y);
  RLineTo(-y*2.0);
  RLineTo(-0.3*x);
  RLineTo(0.6*x);
  MoveTo(0.1*x+0.6*y);    
  RLineTo(1.8*x);
}

// Function 560
int TeletextState_GetChar( TeletextState state )
{
    if ( state.bConceal )
    {
        if ( !Reveal() )
        {
        	return _SP;
        }
    }
    
    if ( IsControlCharacter( state.char ) )
    {
        if ( state.bGfx && state.bHoldGfx )
        {
            return state.iHeldChar;
        }
        else
        {
            return _SP;
        }
    }    
    return state.char;
}

// Function 561
float Glyph4(const in vec2 uv)
{
    vec2 vP = uv - vec2(0.305, 0.125);
    vP /= 0.065;
    vP.x *= 1.5;
    vP.x += vP.y * 0.25;
    
    vec2 vP2 = vP;

    vP.y = abs(vP.y);
    vP.y = pow(vP.y, 1.2);
    float f= length(vP);
    
    vP2.x *= 1.2;
    float f2 = length(vP2 * 1.5 - vec2(0.6, 0.0));
        
    return max(f - 1.0, 1.0 - f2) / 20.0;
}

// Function 562
vec4 glyph_color(uint glyph, ivec2 pixel, float variation)
{
    pixel &= 7;
    pixel.y = 7 - pixel.y;
    int bit_index = pixel.x + (pixel.y << 3);
    int bit = glyph_bit(glyph, bit_index);
    int shadow_bit = min(pixel.x, pixel.y) > 0 ? glyph_bit(glyph, bit_index - 9) : 0;
    return vec4(vec3(bit > 0 ? variation : .1875), float(bit|shadow_bit));
}

// Function 563
float TextRenderer(inout vec4 fragColor, in vec2 fragCoord) {
    // Font printing and HUD
    float letter = 32.0;
    ivec2 uv = ivec2(fragCoord);
    if ((bitsCollected > 0.0) && (uv == ivec2(0, 0))) letter = 48.0 + bitsCollected;

    // Frame rate printouts
    //if (uv == ivec2(0, 1)) letter = 48.0 + floor(framerate);
    //if (uv == ivec2(1, 1)) letter = 48.0 + mod(floor(framerate*10.0),10.0);
    //if (uv == ivec2(2, 1)) letter = 48.0 + mod(floor(framerate*100.0),10.0);

    ivec2 cursorPos = ivec2(1, 2);
    if ((message >= 10.0) && (message <= 18.0)) {
        cursorPos = ivec2(3, 1);
        _E _S _C _A _P _E _SPACE _T _H _E;
        cursorPos = ivec2(4, 2);
        _G _A _M _E _G _R _I _D;
    }
    if ((message >= 20.0) && (message <= 28.0)) {
        cursorPos = ivec2(6, 1);
        _D _E _A _D;
    }
    if ((message >= 30.0) && (message <= 38.0)) {
        cursorPos = ivec2(5, 1);
        _E _S _C _A _P _E _EXCLAMATION;
    }
    /*if ((message >= 40.0) && (message <= 48.0)) {
        cursorPos = ivec2(1, 1);
        _B _U _F _F _E _R _SPACE _O _V _E _R _R _U _N;
        cursorPos = ivec2(0, 2);
        _P _R _O _G _R _A _M _S _SPACE _E _S _C _A _P _E _D;
    }*/
    if ((message >= 50.0) && (message <= 58.0)) {
        cursorPos = ivec2(5, 1);
        _G _O _SPACE _T _O;
        cursorPos = ivec2(3, 2);
        _I _SLASH _O _SPACE _T _O _W _E _R;
    }
    if ((message >= 60.0) && (message <= 68.0)) {
        cursorPos = ivec2(4, 1);
        _T _R _A _N _S _M _I _T;
    }
    if ((message >= 70.0) && (message <= 78.0)) {
        cursorPos = ivec2(1, 1);
        _C _O _L _L _E _C _T _SPACE _8 _SPACE _B _I _T _S;
    }
    if ((message >= 80.0) && (message <= 88.0)) {
        cursorPos = ivec2(1, 6);
        _C _L _I _C _K _SPACE _O _R _SPACE _S _P _A _C _E;
        cursorPos = ivec2(5, 7);
        _S _T _A _R _T _S;
    }
    if ((message >= 90.0) && (message <= 98.0)) {
        cursorPos = ivec2(2, 1);
        _Y _O _U _SPACE _D _I _D _SPACE _N _O _T;
        cursorPos = ivec2(5, 2);
        _E _S _C _A _P _E;
        cursorPos = ivec2(3, 3);
        _T _R _Y _SPACE _A _G _A _I _N;
    }
    if ((message >= 100.0) && (message <= 108.0)) {
        cursorPos = ivec2(3, 1);
        _E _S _C _A _P _E _SPACE _T _H _E;
        cursorPos = ivec2(4, 2);
        _G _A _M _E _G _R _I _D;
        cursorPos = ivec2(7, 5);
        _B _Y;
        cursorPos = ivec2(3, 6);
        _O _T _A _V _I _O _SPACE _G _O _O _D;
    }

    return letter;
}

// Function 564
vec4 barkTexture(vec3 p, vec3 nor)
{
    vec2 r = floor(p.xz / 5.0) * 0.02;
    
    float br = texture(iChannel1, r).x;
	vec3 mat = texCube(iChannel3, p, nor) * vec3(.35, .25, .25);
    mat += texCube(iChannel3, p*1.73, nor)*smoothstep(0.0,.2, mat.y)*br * vec3(1,.9,.8);
    //mat*=mat*2.5;
   	return vec4(mat, .1);
}

// Function 565
void SpaceText(inout vec3 color, vec2 p, in AppState s)
{        
    vec2 scale = vec2(4., 8.);
    vec2 t = floor(p / scale);   
    
    uint v = 0u;    
    v = t.y == 0. ? ( t.x < 4. ? 1936028240u : ( t.x < 8. ? 1935351923u : ( t.x < 12. ? 1701011824u : ( t.x < 16. ? 1869881437u : ( t.x < 20. ? 1635021600u : 29810u ) ) ) ) ) : v;
	v = t.x >= 0. && t.x < 24. ? v : 0u;
    
	float c = float((v >> uint(8. * t.x)) & 255u);
    
    // vec3 textColor = vec3(.3);
	vec3 textColor = vec3(1.0);

    p = (p - t * scale) / scale;
    p.x = (p.x - .5) * .5 + .5;
    float sdf = TextSDF(p, c);
    if (c != 0.)
    {
    	color = mix(textColor, color, smoothstep(-.05, +.05, sdf));
    }
}

// Function 566
vec4 AntiAliasPointSampleTexture_Linear(vec2 uv, vec2 texsize) {	
	vec2 w=fwidth(uv);
	return texture(iChannel0, (floor(uv)+0.5+clamp((fract(uv)-0.5+w)/w,0.,1.)) / texsize, -99999.0);	
}

// Function 567
vec4 get_texture_params(in sampler2D s)
{
    return texelFetch(s, CTRL_TEXTURE, 0);
}

// Function 568
int GetTeletextCode( ivec2 coord )
{
    return int( texelFetch( iChannelScreenData, ivec2( coord ), 0 ).r );
}

// Function 569
float AsymmetricGlyph(vec2 p){p.y=-p.y;p+=.5;
 float a=min(length(p),length(p-.4)*.5);
 p.y+=.2;return max(a,-length(p)+.5);}

// Function 570
float NumFont_One( vec2 vTexCoord )
{
    float fResult = 0.0;
    
    //fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(4, 1), vec2(8,13), fOutline ));
    //fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(3, 2), vec2(3,4), fOutline ));
    //fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(2, 3), vec2(2,4), fOutline ));
    //fResult = max( fResult, NumFont_Pixel( vTexCoord, vec2(1, 4), fOutline ));
    
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(6, 1), vec2(10,13) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(5, 2), vec2(5,4) ));
    fResult = max( fResult, NumFont_Pixel( vTexCoord, vec2(3, 4) ));
    fResult = max( fResult, NumFont_Rect( vTexCoord, vec2(4, 3), vec2(4,4.1) ));
    

    return fResult;
}

// Function 571
float vt220Font(vec2 p, int c)
{
if(c==0) return 0.0;
//else if(c==0) return ;
//else if(c==1) return l(3,4,6)+ l(5,3,7)+ l(7,2,8)+ l(9,1,9)+ l(11,2,8)+ l(13,3,7)+ l(15,4,6);
//else if(c==2) return l(3,1,3)+ l(3,4,6)+ l(3,7,9)+ l(5,2,4)+ l(5,6,8)+ l(7,1,3)+ l(7,4,6)+ l(7,7,9)+ l(9,2,4)+ l(9,6,8)+ l(11,1,3)+ l(11,4,6)+ l(11,7,9)+ l(13,2,4)+ l(13,6,8)+ l(15,1,3)+ l(15,4,6)+ l(15,7,9);
//else if(c==3) return l(3,1,3)+ l(3,4,6)+ l(5,1,3)+ l(5,4,6)+ l(7,1,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,9)+ l(13,5,7)+ l(15,5,7)+ l(17,5,7)+ l(19,5,7);
//else if(c==4) return l(3,1,6)+ l(5,1,3)+ l(7,1,5)+ l(9,1,3)+ l(11,1,3)+ l(11,4,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,6);
//else if(c==5) return l(3,2,6)+ l(5,1,3)+ l(7,1,3)+ l(9,1,3)+ l(11,2,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,8)+ l(17,4,6)+ l(17,7,9)+ l(19,4,6)+ l(19,7,9);
//else if(c==6) return l(3,1,3)+ l(5,1,3)+ l(7,1,3)+ l(9,1,3)+ l(11,1,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,6);
//else if(c==7) return l(3,3,7)+ l(5,2,4)+ l(5,6,8)+ l(7,3,7);
//else if(c==8) return l(5,4,6)+ l(7,4,6)+ l(9,1,9)+ l(11,4,6)+ l(13,4,6)+ l(15,1,9);
//else if(c==9) return l(3,1,3)+ l(3,5,7)+ l(5,1,4)+ l(5,5,7)+ l(7,1,7)+ l(9,1,3)+ l(9,4,7)+ l(11,1,3)+ l(11,4,7)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,4,9);
//else if(c==10) return l(3,1,3)+ l(3,5,7)+ l(5,1,3)+ l(5,5,7)+ l(7,2,6)+ l(9,2,6)+ l(11,3,9)+ l(13,5,7)+ l(15,5,7)+ l(17,5,7)+ l(19,5,7);
//else if(c==11) return l(1,4,6)+ l(3,4,6)+ l(5,4,6)+ l(7,4,6)+ l(9,1,6);
//else if(c==12) return l(9,1,6)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,4,6);
//else if(c==13) return l(9,4,9)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,4,6);
//else if(c==14) return l(1,4,6)+ l(3,4,6)+ l(5,4,6)+ l(7,4,6)+ l(9,4,9);
//else if(c==15) return l(1,4,6)+ l(3,4,6)+ l(5,4,6)+ l(7,4,6)+ l(9,1,9)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,4,6);
//else if(c==16) return l(1,1,9);
//else if(c==17) return l(5,1,9);
//else if(c==18) return l(9,1,9);
//else if(c==19) return l(13,1,9);
else if(c==20) return l(17,1,9);
//else if(c==21) return l(1,4,6)+ l(3,4,6)+ l(5,4,6)+ l(7,4,6)+ l(9,4,9)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,4,6);
//else if(c==22) return l(1,4,6)+ l(3,4,6)+ l(5,4,6)+ l(7,4,6)+ l(9,1,6)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,4,6);
//else if(c==23) return l(1,4,6)+ l(3,4,6)+ l(5,4,6)+ l(7,4,6)+ l(9,1,9);
//else if(c==24) return l(9,1,9)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,4,6);
//else if(c==25) return l(1,4,6)+ l(3,4,6)+ l(5,4,6)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,4,6);
//else if(c==26) return l(3,7,9)+ l(5,5,7)+ l(7,3,5)+ l(9,1,3)+ l(11,3,5)+ l(13,5,7)+ l(15,7,9)+ l(17,1,9);
//else if(c==27) return l(3,1,3)+ l(5,3,5)+ l(7,5,7)+ l(9,7,9)+ l(11,5,7)+ l(13,3,5)+ l(15,1,3)+ l(17,1,9);
//else if(c==28) return l(7,1,9)+ l(9,3,5)+ l(9,6,8)+ l(11,3,5)+ l(11,6,8)+ l(13,3,5)+ l(13,6,8)+ l(15,2,4)+ l(15,6,8);
//else if(c==29) return l(3,7,9)+ l(5,6,8)+ l(7,1,9)+ l(9,4,6)+ l(11,1,9)+ l(13,2,4)+ l(15,1,3);
//else if(c==30) return l(3,4,8)+ l(5,3,5)+ l(5,7,9)+ l(7,3,5)+ l(9,1,7)+ l(11,3,5)+ l(13,2,7)+ l(15,1,5)+ l(15,6,9)+ l(17,2,4);
//else if(c==31) return l(9,4,6);
//else if(c==32) return l(3,2,8)+ l(5,1,3)+ l(5,7,9)+ l(7,2,5)+ l(9,4,6)+ l(11,4,6)+ l(15,4,6);
//else if(c==33) return l(3,4,6)+ l(5,4,6)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(15,4,6);
//else if(c==34) return l(3,2,4)+ l(3,5,7)+ l(5,2,4)+ l(5,5,7)+ l(7,2,4)+ l(7,5,7);
else if(c==35) return l(3,2,4)+ l(3,5,7)+ l(5,2,4)+ l(5,5,7)+ l(7,1,8)+ l(9,2,4)+ l(9,5,7)+ l(11,1,8)+ l(13,2,4)+ l(13,5,7)+ l(15,2,4)+ l(15,5,7);
else if(c==36) return l(3,4,6)+ l(5,2,8)+ l(7,1,3)+ l(7,4,6)+ l(9,2,8)+ l(11,4,6)+ l(11,7,9)+ l(13,2,8)+ l(15,4,6);
else if(c==37) return l(3,2,4)+ l(3,7,9)+ l(5,1,5)+ l(5,6,8)+ l(7,2,4)+ l(7,5,7)+ l(9,4,6)+ l(11,3,5)+ l(11,6,8)+ l(13,2,4)+ l(13,5,9)+ l(15,1,3)+ l(15,6,8);
//else if(c==38) return l(3,2,6)+ l(5,1,3)+ l(5,5,7)+ l(7,1,3)+ l(7,5,7)+ l(9,2,6)+ l(11,1,3)+ l(11,5,9)+ l(13,1,3)+ l(13,6,8)+ l(15,2,9);
//else if(c==39) return l(3,4,7)+ l(5,4,6)+ l(7,3,5);
//else if(c==40) return l(3,5,7)+ l(5,4,6)+ l(7,3,5)+ l(9,3,5)+ l(11,3,5)+ l(13,4,6)+ l(15,5,7);
//else if(c==41) return l(3,3,5)+ l(5,4,6)+ l(7,5,7)+ l(9,5,7)+ l(11,5,7)+ l(13,4,6)+ l(15,3,5);
//else if(c==42) return l(5,2,4)+ l(5,6,8)+ l(7,3,7)+ l(9,1,9)+ l(11,3,7)+ l(13,2,4)+ l(13,6,8);
//else if(c==43) return l(5,4,6)+ l(7,4,6)+ l(9,1,9)+ l(11,4,6)+ l(13,4,6);
//else if(c==44) return l(13,3,6)+ l(15,3,5)+ l(17,2,4);
//else if(c==45) return l(9,1,9);
//else if(c==46) return l(13,3,6)+ l(15,3,6);
//else if(c==47) return l(3,7,9)+ l(5,6,8)+ l(7,5,7)+ l(9,4,6)+ l(11,3,5)+ l(13,2,4)+ l(15,1,3);
//else if(c==48) return l(3,3,7)+ l(5,2,4)+ l(5,6,8)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,2,4)+ l(13,6,8)+ l(15,3,7);
else if(c==49) return l(3,4,6)+ l(5,3,6)+ l(7,2,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
else if(c==50) return l(3,2,8)+ l(5,1,3)+ l(5,7,9)+ l(7,7,9)+ l(9,4,8)+ l(11,2,5)+ l(13,1,3)+ l(15,1,9);
else if(c==51) return l(3,1,9)+ l(5,6,8)+ l(7,5,7)+ l(9,4,8)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==52) return l(3,5,7)+ l(5,4,7)+ l(7,3,7)+ l(9,2,4)+ l(9,5,7)+ l(11,1,9)+ l(13,5,7)+ l(15,5,7);
//else if(c==53) return l(3,1,9)+ l(5,1,3)+ l(7,1,8)+ l(9,1,4)+ l(9,7,9)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==54) return l(3,3,8)+ l(5,2,4)+ l(7,1,3)+ l(9,1,8)+ l(11,1,4)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==55) return l(3,1,9)+ l(5,7,9)+ l(7,6,8)+ l(9,5,7)+ l(11,4,6)+ l(13,3,5)+ l(15,3,5);
//else if(c==56) return l(3,2,8)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,2,8)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==57) return l(3,2,8)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,6,9)+ l(9,2,9)+ l(11,7,9)+ l(13,6,8)+ l(15,2,7);
//else if(c==58) return l(5,3,6)+ l(7,3,6)+ l(13,3,6)+ l(15,3,6);
//else if(c==59) return l(5,3,6)+ l(7,3,6)+ l(13,3,6)+ l(15,3,5)+ l(17,2,4);
//else if(c==60) return l(3,7,9)+ l(5,5,7)+ l(7,3,5)+ l(9,1,3)+ l(11,3,5)+ l(13,5,7)+ l(15,7,9);
//else if(c==61) return l(7,1,9)+ l(11,1,9);
//else if(c==62) return l(3,1,3)+ l(5,3,5)+ l(7,5,7)+ l(9,7,9)+ l(11,5,7)+ l(13,3,5)+ l(15,1,3);
//else if(c==63) return l(3,2,8)+ l(5,1,3)+ l(5,7,9)+ l(7,5,8)+ l(9,4,6)+ l(11,4,6)+ l(15,4,6);
//else if(c==64) return l(3,2,8)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,4,9)+ l(9,1,5)+ l(9,6,9)+ l(11,1,3)+ l(11,4,9)+ l(13,1,3)+ l(15,2,8);
else if(c==65) return l(3,4,6)+ l(5,3,7)+ l(7,2,4)+ l(7,6,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
else if(c==66) return l(3,1,8)+ l(5,2,4)+ l(5,7,9)+ l(7,2,4)+ l(7,7,9)+ l(9,2,8)+ l(11,2,4)+ l(11,7,9)+ l(13,2,4)+ l(13,7,9)+ l(15,1,8);
else if(c==67) return l(3,3,8)+ l(5,2,4)+ l(5,7,9)+ l(7,1,3)+ l(9,1,3)+ l(11,1,3)+ l(13,2,4)+ l(13,7,9)+ l(15,3,8);
//else if(c==68) return l(3,1,7)+ l(5,2,4)+ l(5,6,8)+ l(7,2,4)+ l(7,7,9)+ l(9,2,4)+ l(9,7,9)+ l(11,2,4)+ l(11,7,9)+ l(13,2,4)+ l(13,6,8)+ l(15,1,7);
//else if(c==69) return l(3,1,9)+ l(5,1,3)+ l(7,1,3)+ l(9,1,7)+ l(11,1,3)+ l(13,1,3)+ l(15,1,9);
//else if(c==70) return l(3,1,9)+ l(5,1,3)+ l(7,1,3)+ l(9,1,7)+ l(11,1,3)+ l(13,1,3)+ l(15,1,3);
//else if(c==71) return l(3,3,8)+ l(5,2,4)+ l(5,7,9)+ l(7,1,3)+ l(9,1,3)+ l(11,1,3)+ l(11,5,9)+ l(13,2,4)+ l(13,7,9)+ l(15,3,8);
//else if(c==72) return l(3,1,3)+ l(3,7,9)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==73) return l(3,2,8)+ l(5,4,6)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
//else if(c==74) return l(3,5,9)+ l(5,6,8)+ l(7,6,8)+ l(9,6,8)+ l(11,6,8)+ l(13,1,3)+ l(13,6,8)+ l(15,2,7);
//else if(c==75) return l(3,1,3)+ l(3,7,9)+ l(5,1,3)+ l(5,5,8)+ l(7,1,6)+ l(9,1,4)+ l(11,1,6)+ l(13,1,3)+ l(13,5,8)+ l(15,1,3)+ l(15,7,9);
//else if(c==76) return l(3,1,3)+ l(5,1,3)+ l(7,1,3)+ l(9,1,3)+ l(11,1,3)+ l(13,1,3)+ l(15,1,9);
//else if(c==77) return l(3,1,3)+ l(3,7,9)+ l(5,1,4)+ l(5,6,9)+ l(7,1,9)+ l(9,1,3)+ l(9,4,6)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==78) return l(3,1,3)+ l(3,7,9)+ l(5,1,4)+ l(5,7,9)+ l(7,1,5)+ l(7,7,9)+ l(9,1,3)+ l(9,4,6)+ l(9,7,9)+ l(11,1,3)+ l(11,5,9)+ l(13,1,3)+ l(13,6,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==79) return l(3,2,8)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==80) return l(3,1,8)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,8)+ l(11,1,3)+ l(13,1,3)+ l(15,1,3);
//else if(c==81) return l(3,2,8)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,5,9)+ l(13,1,3)+ l(13,6,8)+ l(15,2,9);
//else if(c==82) return l(3,1,8)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,8)+ l(11,1,3)+ l(11,5,7)+ l(13,1,3)+ l(13,6,8)+ l(15,1,3)+ l(15,7,9);
//else if(c==83) return l(3,2,8)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(9,2,8)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==84) return l(3,1,9)+ l(5,4,6)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6);
//else if(c==85) return l(3,1,3)+ l(3,7,9)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==86) return l(3,1,3)+ l(3,7,9)+ l(5,1,3)+ l(5,7,9)+ l(7,2,4)+ l(7,6,8)+ l(9,2,4)+ l(9,6,8)+ l(11,3,7)+ l(13,3,7)+ l(15,4,6);
//else if(c==87) return l(3,1,3)+ l(3,7,9)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,4,6)+ l(9,7,9)+ l(11,1,3)+ l(11,4,6)+ l(11,7,9)+ l(13,1,9)+ l(15,2,4)+ l(15,6,8);
//else if(c==88) return l(3,1,3)+ l(3,7,9)+ l(5,2,4)+ l(5,6,8)+ l(7,3,7)+ l(9,4,6)+ l(11,3,7)+ l(13,2,4)+ l(13,6,8)+ l(15,1,3)+ l(15,7,9);
//else if(c==89) return l(3,1,3)+ l(3,7,9)+ l(5,2,4)+ l(5,6,8)+ l(7,3,7)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6);
//else if(c==90) return l(3,1,9)+ l(5,6,8)+ l(7,5,7)+ l(9,4,6)+ l(11,3,5)+ l(13,2,4)+ l(15,1,9);
//else if(c==91) return l(3,3,8)+ l(5,3,5)+ l(7,3,5)+ l(9,3,5)+ l(11,3,5)+ l(13,3,5)+ l(15,3,8);
//else if(c==92) return l(3,1,3)+ l(5,2,4)+ l(7,3,5)+ l(9,4,6)+ l(11,5,7)+ l(13,6,8)+ l(15,7,9);
//else if(c==93) return l(3,2,7)+ l(5,5,7)+ l(7,5,7)+ l(9,5,7)+ l(11,5,7)+ l(13,5,7)+ l(15,2,7);
//else if(c==94) return l(3,4,6)+ l(5,3,7)+ l(7,2,4)+ l(7,6,8)+ l(9,1,3)+ l(9,7,9);
//else if(c==95) return l(15,1,9);
//else if(c==96) return l(3,3,6)+ l(5,4,6)+ l(7,5,7);
else if(c==97) return l(7,2,8)+ l(9,7,9)+ l(11,2,9)+ l(13,1,3)+ l(13,6,9)+ l(15,2,9);
else if(c==98) return l(3,1,3)+ l(5,1,3)+ l(7,1,8)+ l(9,1,4)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,4)+ l(13,7,9)+ l(15,1,8);
else if(c==99) return l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(13,1,3)+ l(15,2,9);
//else if(c==100) return l(3,7,9)+ l(5,7,9)+ l(7,2,9)+ l(9,1,3)+ l(9,6,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,6,9)+ l(15,2,9);
//else if(c==101) return l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,9)+ l(13,1,3)+ l(15,2,8);
//else if(c==102) return l(3,4,8)+ l(5,3,5)+ l(5,7,9)+ l(7,3,5)+ l(9,1,7)+ l(11,3,5)+ l(13,3,5)+ l(15,3,5);
//else if(c==103) return l(7,2,9)+ l(9,1,3)+ l(9,6,8)+ l(11,2,7)+ l(13,1,3)+ l(15,2,8)+ l(17,1,3)+ l(17,7,9)+ l(19,2,8);
//else if(c==104) return l(3,1,3)+ l(5,1,3)+ l(7,1,8)+ l(9,1,4)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==105) return l(3,4,6)+ l(7,3,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
//else if(c==106) return l(3,6,8)+ l(7,6,8)+ l(9,6,8)+ l(11,6,8)+ l(13,6,8)+ l(15,1,3)+ l(15,6,8)+ l(17,1,3)+ l(17,6,8)+ l(19,2,7);
//else if(c==107) return l(3,1,3)+ l(5,1,3)+ l(7,1,3)+ l(7,5,7)+ l(9,1,3)+ l(9,4,6)+ l(11,1,5)+ l(13,1,3)+ l(13,5,7)+ l(15,1,3)+ l(15,7,9);
//else if(c==108) return l(3,3,6)+ l(5,4,6)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,3,7);
//else if(c==109) return l(7,1,4)+ l(7,6,8)+ l(9,1,9)+ l(11,1,3)+ l(11,4,6)+ l(11,7,9)+ l(13,1,3)+ l(13,4,6)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==110) return l(7,1,8)+ l(9,1,4)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==111) return l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==112) return l(7,1,8)+ l(9,1,4)+ l(9,7,9)+ l(11,1,4)+ l(11,7,9)+ l(13,1,8)+ l(15,1,3)+ l(17,1,3)+ l(19,1,3);
//else if(c==113) return l(7,2,9)+ l(9,1,3)+ l(9,6,9)+ l(11,1,3)+ l(11,6,9)+ l(13,2,9)+ l(15,7,9)+ l(17,7,9)+ l(19,7,9);
//else if(c==114) return l(7,1,3)+ l(7,4,8)+ l(9,2,5)+ l(9,7,9)+ l(11,2,4)+ l(13,2,4)+ l(15,2,4);
//else if(c==115) return l(7,2,8)+ l(9,1,3)+ l(11,2,8)+ l(13,7,9)+ l(15,1,8);
//else if(c==116) return l(3,3,5)+ l(5,3,5)+ l(7,1,7)+ l(9,3,5)+ l(11,3,5)+ l(13,3,5)+ l(13,6,8)+ l(15,4,7);
//else if(c==117) return l(7,1,3)+ l(7,6,8)+ l(9,1,3)+ l(9,6,8)+ l(11,1,3)+ l(11,6,8)+ l(13,1,3)+ l(13,6,8)+ l(15,2,9);
//else if(c==118) return l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,2,4)+ l(11,6,8)+ l(13,3,7)+ l(15,4,6);
//else if(c==119) return l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,4,6)+ l(11,7,9)+ l(13,1,9)+ l(15,2,4)+ l(15,6,8);
//else if(c==120) return l(7,1,3)+ l(7,6,8)+ l(9,2,4)+ l(9,5,7)+ l(11,3,6)+ l(13,2,4)+ l(13,5,7)+ l(15,1,3)+ l(15,6,8);
//else if(c==121) return l(7,1,3)+ l(7,6,8)+ l(9,1,3)+ l(9,6,8)+ l(11,1,3)+ l(11,5,8)+ l(13,2,8)+ l(15,6,8)+ l(17,1,3)+ l(17,6,8)+ l(19,2,7);
//else if(c==122) return l(7,1,9)+ l(9,6,8)+ l(11,4,7)+ l(13,3,5)+ l(15,1,9);
//else if(c==123) return l(3,5,9)+ l(5,4,6)+ l(7,5,7)+ l(9,3,6)+ l(11,5,7)+ l(13,4,6)+ l(15,5,9);
//else if(c==124) return l(3,4,6)+ l(5,4,6)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6);
//else if(c==125) return l(3,1,5)+ l(5,4,6)+ l(7,3,5)+ l(9,4,7)+ l(11,3,5)+ l(13,4,6)+ l(15,1,5);
//else if(c==126) return l(3,2,5)+ l(3,7,9)+ l(5,1,3)+ l(5,4,6)+ l(5,7,9)+ l(7,1,3)+ l(7,5,8);
//else if(c==127) return l(3,1,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,3)+ l(7,4,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,7)+ l(13,3,5)+ l(15,3,5)+ l(17,3,5)+ l(19,3,5);
//else if(c==128) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,6)+ l(15,7,9)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==129) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,5)+ l(11,6,8)+ l(13,5,8)+ l(15,6,8)+ l(17,6,8)+ l(19,5,9);
//else if(c==130) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,8)+ l(13,4,6)+ l(13,7,9)+ l(15,6,8)+ l(17,5,7)+ l(19,4,9);
//else if(c==131) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,9)+ l(13,7,9)+ l(15,5,8)+ l(17,7,9)+ l(19,4,8);
//else if(c==132) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,5)+ l(11,6,8)+ l(13,5,8)+ l(15,4,8)+ l(17,3,9)+ l(19,6,8);
//else if(c==133) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,9)+ l(13,4,6)+ l(15,4,8)+ l(17,7,9)+ l(19,4,8);
//else if(c==134) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,5)+ l(11,6,9)+ l(13,5,7)+ l(15,4,8)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==135) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,9)+ l(13,7,9)+ l(15,6,8)+ l(17,5,7)+ l(19,5,7);
//else if(c==136) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,8)+ l(13,4,6)+ l(13,7,9)+ l(15,5,8)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==137) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,8)+ l(13,4,6)+ l(13,7,9)+ l(15,5,9)+ l(17,6,8)+ l(19,4,7);
//else if(c==138) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,9)+ l(17,4,6)+ l(17,7,9)+ l(19,4,6)+ l(19,7,9);
//else if(c==139) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,8)+ l(17,4,6)+ l(17,7,9)+ l(19,4,8);
//else if(c==140) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,9)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,5,9);
//else if(c==141) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,6)+ l(15,7,9)+ l(17,4,6)+ l(17,7,9)+ l(19,4,8);
//else if(c==142) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,9);
//else if(c==143) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,5)+ l(9,1,3)+ l(9,4,6)+ l(11,2,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,6);
//else if(c==144) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,4)+ l(11,5,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,6)+ l(15,7,9)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==145) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,4)+ l(11,6,8)+ l(13,5,8)+ l(15,6,8)+ l(17,6,8)+ l(19,5,9);
//else if(c==146) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,4)+ l(11,5,8)+ l(13,4,6)+ l(13,7,9)+ l(15,6,8)+ l(17,5,7)+ l(19,4,9);
//else if(c==147) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,9)+ l(13,7,9)+ l(15,5,8)+ l(17,7,9)+ l(19,4,8);
//else if(c==148) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,4)+ l(11,6,8)+ l(13,5,8)+ l(15,4,8)+ l(17,3,9)+ l(19,6,8);
//else if(c==149) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,9)+ l(13,4,6)+ l(15,4,8)+ l(17,7,9)+ l(19,4,8);
//else if(c==150) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,4)+ l(11,6,9)+ l(13,5,7)+ l(15,4,8)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==151) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,9)+ l(13,7,9)+ l(15,6,8)+ l(17,5,7)+ l(19,5,7);
//else if(c==152) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,4)+ l(11,5,8)+ l(13,4,6)+ l(13,7,9)+ l(15,5,8)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==153) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,4)+ l(11,5,8)+ l(13,4,6)+ l(13,7,9)+ l(15,5,9)+ l(17,6,8)+ l(19,4,7);
//else if(c==154) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,4)+ l(11,5,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,9)+ l(17,4,6)+ l(17,7,9)+ l(19,4,6)+ l(19,7,9);
//else if(c==155) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,8)+ l(17,4,6)+ l(17,7,9)+ l(19,4,8);
//else if(c==156) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,4)+ l(11,5,9)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,5,9);
//else if(c==157) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,6)+ l(15,7,9)+ l(17,4,6)+ l(17,7,9)+ l(19,4,8);
//else if(c==158) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,9);
//else if(c==159) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,2,6)+ l(9,3,5)+ l(11,1,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,6);
//else if(c==160) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,3)+ l(11,4,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,6)+ l(15,7,9)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==161) return l(3,4,6)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6);
//else if(c==162) return l(5,4,6)+ l(7,2,8)+ l(9,1,3)+ l(9,4,6)+ l(9,7,9)+ l(11,1,3)+ l(11,4,6)+ l(13,1,3)+ l(13,4,6)+ l(13,7,9)+ l(15,2,8)+ l(17,4,6);
//else if(c==163) return l(3,4,8)+ l(5,3,5)+ l(5,7,9)+ l(7,3,5)+ l(9,1,7)+ l(11,3,5)+ l(13,2,7)+ l(15,1,5)+ l(15,6,9)+ l(17,2,4);
//else if(c==164) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,3)+ l(11,4,8)+ l(13,5,8)+ l(15,4,8)+ l(17,3,9)+ l(19,6,8);
//else if(c==165) return l(3,1,3)+ l(3,7,9)+ l(5,2,4)+ l(5,6,8)+ l(7,3,7)+ l(9,4,6)+ l(11,1,9)+ l(13,4,6)+ l(15,4,6);
//else if(c==166) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,3)+ l(11,4,9)+ l(13,5,7)+ l(15,4,8)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==167) return l(3,3,8)+ l(5,2,4)+ l(7,3,7)+ l(9,2,4)+ l(9,6,8)+ l(11,3,7)+ l(13,6,8)+ l(15,2,7);
//else if(c==168) return l(5,1,3)+ l(5,7,9)+ l(7,2,8)+ l(9,2,4)+ l(9,6,8)+ l(11,2,8)+ l(13,1,3)+ l(13,7,9);
//else if(c==169) return l(3,3,7)+ l(5,2,4)+ l(5,6,8)+ l(7,1,3)+ l(7,4,9)+ l(9,1,5)+ l(9,7,9)+ l(11,1,5)+ l(11,7,9)+ l(13,1,3)+ l(13,4,9)+ l(15,2,4)+ l(15,6,8)+ l(17,3,7);
//else if(c==170) return l(3,2,8)+ l(5,7,9)+ l(7,2,9)+ l(9,1,3)+ l(9,6,9)+ l(11,2,9)+ l(15,1,9);
//else if(c==171) return l(3,4,6)+ l(3,7,9)+ l(5,3,5)+ l(5,6,8)+ l(7,2,4)+ l(7,5,7)+ l(9,1,3)+ l(9,4,6)+ l(11,2,4)+ l(11,5,7)+ l(13,3,5)+ l(13,6,8)+ l(15,4,6)+ l(15,7,9);
//else if(c==172) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,3)+ l(11,4,9)+ l(13,4,6)+ l(15,4,6)+ l(17,4,6)+ l(19,5,9);
//else if(c==173) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,3)+ l(11,4,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,6)+ l(15,7,9)+ l(17,4,6)+ l(17,7,9)+ l(19,4,8);
//else if(c==174) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,3)+ l(11,4,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,9);
//else if(c==175) return l(3,2,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,3)+ l(11,4,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,6);
//else if(c==176) return l(3,3,7)+ l(5,2,4)+ l(5,6,8)+ l(7,3,7);
//else if(c==177) return l(5,4,6)+ l(7,4,6)+ l(9,1,9)+ l(11,4,6)+ l(13,4,6)+ l(15,1,9);
//else if(c==178) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(5,4,7)+ l(7,3,5)+ l(9,2,8);
//else if(c==179) return l(1,2,8)+ l(3,6,8)+ l(5,4,7)+ l(7,6,8)+ l(9,2,7);
//else if(c==180) return l(3,1,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,5)+ l(9,1,3)+ l(9,4,6)+ l(11,1,5)+ l(11,6,8)+ l(13,5,8)+ l(15,4,8)+ l(17,3,9)+ l(19,6,8);
//else if(c==181) return l(7,2,4)+ l(7,7,9)+ l(9,2,4)+ l(9,7,9)+ l(11,2,4)+ l(11,7,9)+ l(13,2,5)+ l(13,6,9)+ l(15,2,9)+ l(17,2,4)+ l(19,1,3);
//else if(c==182) return l(3,2,9)+ l(5,1,4)+ l(5,5,9)+ l(7,1,4)+ l(7,5,9)+ l(9,2,9)+ l(11,5,9)+ l(13,5,9)+ l(15,5,9);
//else if(c==183) return l(7,3,6)+ l(9,3,6);
//else if(c==184) return l(3,1,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,5)+ l(9,1,3)+ l(9,4,6)+ l(11,1,8)+ l(13,4,6)+ l(13,7,9)+ l(15,5,8)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==185) return l(1,4,6)+ l(3,3,6)+ l(5,4,6)+ l(7,4,6)+ l(9,3,7);
//else if(c==186) return l(3,2,8)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,2,8)+ l(15,1,9);
//else if(c==187) return l(3,1,3)+ l(3,4,6)+ l(5,2,4)+ l(5,5,7)+ l(7,3,5)+ l(7,6,8)+ l(9,4,6)+ l(9,7,9)+ l(11,3,5)+ l(11,6,8)+ l(13,2,4)+ l(13,5,7)+ l(15,1,3)+ l(15,4,6);
//else if(c==188) return l(3,2,4)+ l(5,1,4)+ l(5,6,8)+ l(7,2,4)+ l(7,5,7)+ l(9,2,6)+ l(11,3,5)+ l(11,6,8)+ l(13,2,4)+ l(13,5,8)+ l(15,1,3)+ l(15,4,8)+ l(17,3,9)+ l(19,6,8);
//else if(c==189) return l(3,2,4)+ l(5,1,4)+ l(5,6,8)+ l(7,2,4)+ l(7,5,7)+ l(9,2,6)+ l(11,3,8)+ l(13,2,6)+ l(13,7,9)+ l(15,1,3)+ l(15,6,8)+ l(17,5,7)+ l(19,4,9);
//else if(c==190) return l(3,1,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,5)+ l(9,1,3)+ l(9,4,6)+ l(11,1,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,9);
//else if(c==191) return l(3,4,6)+ l(7,4,6)+ l(9,4,6)+ l(11,2,5)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==192) return l(1,3,6)+ l(3,5,8)+ l(5,4,6)+ l(7,3,7)+ l(9,2,4)+ l(9,6,8)+ l(11,1,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==193) return l(1,4,7)+ l(3,2,5)+ l(5,4,6)+ l(7,3,7)+ l(9,2,4)+ l(9,6,8)+ l(11,1,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==194) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(5,4,6)+ l(7,3,7)+ l(9,2,4)+ l(9,6,8)+ l(11,1,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==195) return l(1,3,6)+ l(1,7,9)+ l(3,2,4)+ l(3,5,8)+ l(5,4,6)+ l(7,3,7)+ l(9,2,4)+ l(9,6,8)+ l(11,1,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==196) return l(1,2,4)+ l(1,6,8)+ l(5,4,6)+ l(7,3,7)+ l(9,2,4)+ l(9,6,8)+ l(11,1,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==197) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(5,3,7)+ l(7,3,7)+ l(9,2,4)+ l(9,6,8)+ l(11,1,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==198) return l(3,3,9)+ l(5,2,6)+ l(7,1,3)+ l(7,4,6)+ l(9,1,3)+ l(9,4,8)+ l(11,1,6)+ l(13,1,3)+ l(13,4,6)+ l(15,1,3)+ l(15,4,9);
//else if(c==199) return l(3,3,8)+ l(5,2,4)+ l(5,7,9)+ l(7,1,3)+ l(9,1,3)+ l(11,1,3)+ l(13,2,4)+ l(13,7,9)+ l(15,3,8)+ l(17,5,7)+ l(19,3,7);
//else if(c==200) return l(1,2,5)+ l(3,4,7)+ l(5,1,9)+ l(7,1,3)+ l(9,1,7)+ l(11,1,3)+ l(13,1,3)+ l(15,1,9);
//else if(c==201) return l(1,5,8)+ l(3,3,6)+ l(5,1,9)+ l(7,1,3)+ l(9,1,7)+ l(11,1,3)+ l(13,1,3)+ l(15,1,9);
//else if(c==202) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(5,1,9)+ l(7,1,3)+ l(9,1,6)+ l(11,1,3)+ l(13,1,3)+ l(15,1,9);
//else if(c==203) return l(1,2,4)+ l(1,6,8)+ l(5,1,9)+ l(7,1,3)+ l(9,1,7)+ l(11,1,3)+ l(13,1,3)+ l(15,1,9);
//else if(c==204) return l(1,3,6)+ l(3,5,8)+ l(5,2,8)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
//else if(c==205) return l(1,4,7)+ l(3,2,5)+ l(5,2,8)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
//else if(c==206) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(5,2,8)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
//else if(c==207) return l(1,2,4)+ l(1,6,8)+ l(5,2,8)+ l(7,4,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
//else if(c==208) return l(3,1,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,3)+ l(7,4,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,6)+ l(15,7,9)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==209) return l(1,3,6)+ l(1,7,9)+ l(3,2,4)+ l(3,5,8)+ l(5,1,4)+ l(5,7,9)+ l(7,1,5)+ l(7,7,9)+ l(9,1,3)+ l(9,4,6)+ l(9,7,9)+ l(11,1,3)+ l(11,5,9)+ l(13,1,3)+ l(13,6,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==210) return l(1,3,6)+ l(3,5,8)+ l(5,2,8)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==211) return l(1,4,7)+ l(3,2,5)+ l(5,2,8)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==212) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(5,2,8)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==213) return l(1,2,5)+ l(1,6,8)+ l(3,1,3)+ l(3,4,7)+ l(5,2,8)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==214) return l(1,2,4)+ l(1,6,8)+ l(5,2,8)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==215) return l(3,2,9)+ l(5,1,3)+ l(5,4,6)+ l(7,1,3)+ l(7,4,6)+ l(9,1,3)+ l(9,4,8)+ l(11,1,3)+ l(11,4,6)+ l(13,1,3)+ l(13,4,6)+ l(15,2,9);
//else if(c==216) return l(1,7,9)+ l(3,2,8)+ l(5,1,3)+ l(5,6,9)+ l(7,1,3)+ l(7,5,9)+ l(9,1,3)+ l(9,4,6)+ l(9,7,9)+ l(11,1,5)+ l(11,7,9)+ l(13,1,4)+ l(13,7,9)+ l(15,2,8)+ l(17,1,3);
//else if(c==217) return l(1,2,5)+ l(3,4,7)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==218) return l(1,5,8)+ l(3,3,6)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==219) return l(1,4,6)+ l(3,3,7)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==220) return l(1,2,4)+ l(1,6,8)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==221) return l(1,2,4)+ l(1,6,8)+ l(5,1,3)+ l(5,7,9)+ l(7,2,4)+ l(7,6,8)+ l(9,3,7)+ l(11,4,6)+ l(13,4,6)+ l(15,4,6);
//else if(c==222) return l(3,1,5)+ l(5,1,3)+ l(5,4,6)+ l(7,1,3)+ l(7,4,6)+ l(9,1,3)+ l(9,4,6)+ l(11,1,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,9);
//else if(c==223) return l(3,3,7)+ l(5,2,4)+ l(5,6,8)+ l(7,1,3)+ l(7,6,8)+ l(9,1,7)+ l(11,1,3)+ l(11,6,8)+ l(13,1,5)+ l(13,7,9)+ l(15,1,3)+ l(15,4,8)+ l(17,1,3);
//else if(c==224) return l(3,3,6)+ l(5,5,8)+ l(7,2,8)+ l(9,7,9)+ l(11,2,9)+ l(13,1,3)+ l(13,6,9)+ l(15,2,9);
//else if(c==225) return l(3,4,7)+ l(5,2,5)+ l(7,2,8)+ l(9,7,9)+ l(11,2,9)+ l(13,1,3)+ l(13,6,9)+ l(15,2,9);
//else if(c==226) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(7,2,8)+ l(9,7,9)+ l(11,2,9)+ l(13,1,3)+ l(13,6,9)+ l(15,2,9);
//else if(c==227) return l(1,3,6)+ l(1,7,9)+ l(3,2,4)+ l(3,5,8)+ l(7,2,8)+ l(9,7,9)+ l(11,2,9)+ l(13,1,3)+ l(13,6,9)+ l(15,2,9);
//else if(c==228) return l(3,2,4)+ l(3,6,8)+ l(7,2,8)+ l(9,7,9)+ l(11,2,9)+ l(13,1,3)+ l(13,6,9)+ l(15,2,9);
//else if(c==229) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(5,3,7)+ l(7,2,8)+ l(9,7,9)+ l(11,2,9)+ l(13,1,3)+ l(13,6,9)+ l(15,2,9);
//else if(c==230) return l(7,2,8)+ l(9,4,6)+ l(9,7,9)+ l(11,2,9)+ l(13,1,3)+ l(13,4,6)+ l(15,2,9);
//else if(c==231) return l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8)+ l(17,5,7)+ l(19,3,7);
//else if(c==232) return l(3,3,6)+ l(5,5,8)+ l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,9)+ l(13,1,3)+ l(15,2,8);
//else if(c==233) return l(3,4,7)+ l(5,2,5)+ l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,9)+ l(13,1,3)+ l(15,2,8);
//else if(c==234) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,9)+ l(13,1,3)+ l(15,2,8);
//else if(c==235) return l(3,2,4)+ l(3,6,8)+ l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,9)+ l(13,1,3)+ l(15,2,8);
//else if(c==236) return l(3,2,5)+ l(5,4,7)+ l(7,3,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
//else if(c==237) return l(3,4,7)+ l(5,2,5)+ l(7,3,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
//else if(c==238) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(7,3,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
//else if(c==239) return l(3,2,4)+ l(3,6,8)+ l(7,3,6)+ l(9,4,6)+ l(11,4,6)+ l(13,4,6)+ l(15,2,8);
//else if(c==240) return l(3,1,6)+ l(5,1,3)+ l(7,1,5)+ l(9,1,3)+ l(11,1,3)+ l(11,5,8)+ l(13,4,6)+ l(13,7,9)+ l(15,4,6)+ l(15,7,9)+ l(17,4,6)+ l(17,7,9)+ l(19,5,8);
//else if(c==241) return l(1,2,5)+ l(1,6,8)+ l(3,1,3)+ l(3,4,7)+ l(7,1,8)+ l(9,1,4)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,3)+ l(15,7,9);
//else if(c==242) return l(3,3,6)+ l(5,5,8)+ l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==243) return l(3,4,7)+ l(5,2,5)+ l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
else if(c==244) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
else if(c==245) return l(1,2,5)+ l(1,6,8)+ l(3,1,3)+ l(3,4,7)+ l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
else if(c==246) return l(3,2,4)+ l(3,6,8)+ l(7,2,8)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,2,8);
//else if(c==247) return l(7,2,8)+ l(9,1,3)+ l(9,4,6)+ l(9,7,9)+ l(11,1,3)+ l(11,4,9)+ l(13,1,3)+ l(13,4,6)+ l(15,2,9);
//else if(c==248) return l(5,7,9)+ l(7,2,8)+ l(9,1,3)+ l(9,5,9)+ l(11,1,3)+ l(11,4,6)+ l(11,7,9)+ l(13,1,5)+ l(13,7,9)+ l(15,2,8)+ l(17,1,3);
//else if(c==249) return l(3,2,5)+ l(5,4,7)+ l(7,1,3)+ l(7,6,8)+ l(9,1,3)+ l(9,6,8)+ l(11,1,3)+ l(11,6,8)+ l(13,1,3)+ l(13,6,8)+ l(15,2,9);
//else if(c==250) return l(3,4,7)+ l(5,2,5)+ l(7,1,3)+ l(7,6,8)+ l(9,1,3)+ l(9,6,8)+ l(11,1,3)+ l(11,6,8)+ l(13,1,3)+ l(13,6,8)+ l(15,2,9);
//else if(c==251) return l(1,3,7)+ l(3,2,4)+ l(3,6,8)+ l(7,1,3)+ l(7,6,8)+ l(9,1,3)+ l(9,6,8)+ l(11,1,3)+ l(11,6,8)+ l(13,1,3)+ l(13,6,8)+ l(15,2,9);
//else if(c==252) return l(3,2,4)+ l(3,6,8)+ l(7,1,3)+ l(7,6,8)+ l(9,1,3)+ l(9,6,8)+ l(11,1,3)+ l(11,6,8)+ l(13,1,3)+ l(13,6,8)+ l(15,2,9);
//else if(c==253) return l(3,2,4)+ l(3,6,8)+ l(7,1,3)+ l(7,6,8)+ l(9,1,3)+ l(9,6,8)+ l(11,1,3)+ l(11,5,8)+ l(13,2,8)+ l(15,6,8)+ l(17,1,3)+ l(17,6,8)+ l(19,2,7);
//else if(c==254) return l(3,1,6)+ l(5,1,3)+ l(7,1,5)+ l(9,1,3)+ l(11,1,3)+ l(11,4,9)+ l(13,4,6)+ l(15,4,8)+ l(17,4,6)+ l(19,4,9);
//else if(c==255) return l(3,1,9)+ l(5,1,3)+ l(5,7,9)+ l(7,1,3)+ l(7,7,9)+ l(9,1,3)+ l(9,7,9)+ l(11,1,3)+ l(11,7,9)+ l(13,1,3)+ l(13,7,9)+ l(15,1,9);
else return 0.0;      
}

// Function 572
vec4 flagTexture(vec2 p2)
{
    vec2 p=p2.xy*0.7+vec2(0.5);
    vec2 c=flagTC(p);
    vec2 e=vec2(1e-2,0.0);
    float g=max(length((flagTC(p+e.xy)-c)/e.x),length((flagTC(p+e.yx)-c)/e.x));

    float b=step(abs(c.y-0.5),0.3)*step(c.x,0.8);
    float nb=-1.0+abs(c.y-0.5)*3.3;
    return vec4(flagTexture2((c*2.0-vec2(1.0))*vec2(-1.0,1.0))*(1.0-smoothstep(0.0,2.0,g)),
                step(abs(c.y-0.5)*2.0,0.7)*step(abs(c.x-0.5)*2.0,1.0)*smoothstep(nb+0.199,nb+0.2,smoothNoise2(c*vec2(32.0,0.5))));
}

// Function 573
float text_play(vec2 U) {
    initMsg;C(80);C(76);C(65);C(89);endMsg;
}

// Function 574
float textfps(vec2 uv){print_pos=vec2(2,2.+STRHEIGHT(1.));
 float r=intP(1./iTimeDelta,0 _ _amp _ _spc,uv)//fps
     +intP(float(iFrame)/iTime,0 _ _f _ _p _ _s _ _equ,uv)//total average fps
     +intP(float(iFrame),0 _ _f _ _r _ _a _ _m _ _e _ _lsl,uv)+intP(iTime*1000.,0 _ _m _ _s,uv)
     ;
 print_pos=vec2(2,2.+STRHEIGHT(0.));tmi _D _ _a _ _t _ _e,uv);tmn _col,uv);
 r+=intP(iDate.x,4 _ _dsh,uv)+intP(iDate.y+1.,2 _ _dsh,uv)+intP(iDate.z,2 _ _lsl,uv);//date
 r+=intP(floor(iDate.w/3600.),2 _ _col,uv)+intP(floor(mod(iDate.w/60.,60.)),2 _ _col,uv)//hour+min
 +intP(floor(mod(iDate.w,60.)),2,uv);tmn _spc,uv);//sec
 tmi _R _ _e _ _s,uv); tmn _col,uv);return r+intP(iResolution.x,0 _ _x,uv)+intP(iResolution.y,0,uv);}

// Function 575
vec4 leavesTexture(vec3 p, vec3 nor)
{
    
    vec3 rand = texCube(iChannel2, p*.15,nor);
	vec3 mat = vec3(0.4,1.2,0) *rand;
   	return vec4(mat, .0);
}

// Function 576
vec3 draw_glyphs(vec2 fragCoord, float scale, float a, inout vec3 col) {
    vec3 total = vec3(0.);
    float total_alpha = 0.;
    for(int i = 0; i < MAX_GLYPHS; i++) {
        float i_float = float(i);
        vec4 glyphcol = glyph(glyphs[i], i_float, scale, fragCoord);
        float alpha = step(line_appear_time + .05 * i_float, time_remapped);
        alpha *= glyphcol.a;
        alpha *= step(i_float, glyph_count - 1.);
        total = mix(total, glyphcol.rgb, alpha);
        total_alpha = max(total_alpha, alpha);
    }
    // col = mix(col, total, total_alpha * a);
    return total*total_alpha;
    // return (1.-total)*total_alpha;
}

// Function 577
vec3 VortexTexture(in vec2 p)
{
    p *= vec2(0.93, -0.93);
    vec4 col = vec4(1.0);

    for(int i = 0; i < 3; i++) 
        col += texture(iChannel1, p + 0.02 * swirl(3.66 * p + iTime * 0.33)) * col * col;

    // used noise texture has only one channel
    return (col * 0.033).xxx;
}

// Function 578
float hairtexture(vec2 uv, float a)
{
    // the fur thingy basically works like that: stretch some hair noise in one axis, render stretches with some variation in patches
    // and rotate each pattern along the parameter. since the rotation happens around the patch center, you can create fur-like transitions.
    // transitions are enhanced by overlapping patterns. the code below is already simplified for 4k, i might release the fur pattern
    // with a functional breakdown in the future if you're interested! let me know in the comments
	vec2 offsets[9] = vec2[9](vec2(0.0), vec2(0.5), vec2(-0.5),
                              vec2(0.5, 0.0), vec2(-0.5, 0.0),
                              vec2(0.0, 0.5), vec2(0.0, -0.5),
                              vec2(0.5, -0.5), vec2(-0.5, 0.5));

    float f = 0.0;
    for(int i = 0; i < 9; i++)
    {
        lib_random_seed++;
        vec2 u = uv * 10.0 + offsets[i];
        vec2 o = floor(u);
        vec2 hp = fract(u) - vec2(0.5);
        hp = lib_rotate(hp, a + sin(R(o) * 5.0) * TOUSLE);   
        float h =  max(0.0, 1.0 - min(1.0 - smoothstep(0.0, 2.3, lib_hair_noise_pattern((hp + o) * vec2(1.0 / HAIR_LENGTH, 0.5) * 70.0)), 1.0));
        h = pow(h * max(1.0 - length(hp) * BORDER, 0.0), 1.0 / 3.0);
        f = max(f, mix(h, h * lib_gn_lookup[int(lib_random_seed) % 4], h));
    } 
    lib_random_seed = 0.0;
    
    return f;
}

// Function 579
vec3 getColorTextura( vec3 p, vec3 nor,  int i)
{	if (i==100 )
    { vec3 col=tex3D(iChannel0, p/32., nor); return col*1.0; }
	if (i==101 ) { return tex3D(iChannel1, p/32., nor); }
	if (i==102 ) { return tex3D(iChannel2, p/32., nor); }
	if (i==103 ) { return tex3D(iChannel3, p/32., nor); }
}

// Function 580
vec3 NearestTextureSample (vec2 P)
{
    vec2 pixel = P * c_textureSize;
    
    vec2 frac = fract(pixel);
    pixel = (floor(pixel) / c_textureSize);
    return texture(iChannel0, pixel + vec2(c_onePixel/2.0)).rgb;
}

// Function 581
void textureParallaxWindow(in vec3 block, inout ray ray, inout rayMarchHit hit, in vec2 uv, in vec3 cell, in vec3 lightColour, in float brightness, inout vec3 colour, in float time) {
    
    if(abs(uv.x)>PARALLAX_INTERROOM_WALL_THICKNESS||abs(uv.y)>PARALLAX_INTERROOM_WALL_THICKNESS) {
        colour=PARALLAX_INTERROOM_WALL_COLOUR;
        return;
    }
    
    vec3 absNormal = abs(hit.surfaceNormal);
    vec3 flatNormal;

    //flaten the normal so we still project axis aligned with no distortion of y axis curve.
    if(absNormal.y > absNormal.x && absNormal.y > absNormal.z) {
        flatNormal = vec3(0.0,1.0,0.0);
    } else {
        flatNormal = normalize(vec3(hit.surfaceNormal.x,0.0,hit.surfaceNormal.z));
    }    
    
    vec3 roomSize = vec3(1.0) ;
    roomSize.z += hash31(cell*16.8736)*2.0;
        
    vec3 tangent = normalize(cross(vec3(0.0,1.0,0.0),flatNormal));
    vec3 bitangent = normalize(cross(flatNormal,tangent));
    mat3 tMatrix = mat3(tangent,bitangent,flatNormal);
    
    vec3 rayDir = normalize(ray.direction*tMatrix)/roomSize;
    vec3 hitPos = vec3(uv.x,uv.y,0.0)/roomSize;
	
    //Room cube, We assume the room is 1 unit cube from -0.5 to +0.5, with a given wall thickness.
    vec3 roomMin = vec3(-PARALLAX_INTERROOM_WALL_THICKNESS,-PARALLAX_INTERROOM_WALL_THICKNESS,-1.0);
    vec3 roomMax = vec3(PARALLAX_INTERROOM_WALL_THICKNESS,PARALLAX_INTERROOM_WALL_THICKNESS,0.0);
    vec3 roomMid = vec3(0.0);
    
    //we only need to interesct 3 planes per ray, looking at the direction of the ray find which 3 its heading towards.
    vec3 planes = mix(roomMin, roomMax, step(0.0, rayDir));
    
    //now do the parallax calcualtion to find the project position 'into' the window
    vec3 planeIntersect = ((planes-hitPos)/rayDir);
    float distance;

    if(planeIntersect.x < planeIntersect.y && planeIntersect.x < planeIntersect.z) {
        //Left/Right wall
        colour=clamp(hash33(cell*48.2270)+0.7,0.0,1.0);
        distance = planeIntersect.x;
    } else if (planeIntersect.y < planeIntersect.x && planeIntersect.y < planeIntersect.z) {
        if(rayDir.y<0.0) {
            //Floor
            colour = clamp(hash33(cell*81.7619)+0.3,0.0,1.0);
        } else {
            //Ceiling
            colour =mix(clamp(hash33(cell*20.9912)+0.3,0.0,1.0),
                        lightColour*6.0,
                	abs(sin((planeIntersect.y*PI*3.0))));
        }
        distance = planeIntersect.y;
    } else if (planeIntersect.z < planeIntersect.x && planeIntersect.z < planeIntersect.y) {
        //Back wall
        colour=clamp(hash33(cell*54.8454)+0.7,0.0,1.0);
        distance = planeIntersect.z;
    } else {
        //error!
        colour=PARALLAX_INTERROOM_WALL_COLOUR;
        distance = 0.0;   
    }
    vec3 intersectionPos = ((hitPos + rayDir * distance) - roomMin);
	//add some distance and height shadow    
    colour*=clamp(intersectionPos.z*(1.0-intersectionPos.y)+0.3,0.0,1.0)*brightness*lightColour;
}

// Function 582
vec4 textureTriPlanar(vec3 P, vec3 N)
{
    float texScale = 2.0;
    // Absolute world normal
    vec3 sharpness = vec3(1.0);
    vec3 Nb = pow(abs(N), sharpness);     
    // Force weights to sum to 1.0
    float b = (Nb.x + Nb.y + Nb.z);
    Nb /= vec3(b);	
    
    vec4 c0 = texture(iChannel0, P.xy * texScale) * Nb.z;
    vec4 c1 = texture(iChannel0, P.yz * texScale) * Nb.x;
    vec4 c2 = texture(iChannel0, P.xz * texScale) * Nb.y;
    
    //vec4 c0 = vec4(1.0,0.0,0.0,1.0) * Nb.z;
    //vec4 c1 = vec4(0.0,1.0,0.0,1.0) * Nb.x;
    //vec4 c2 = vec4(0.0,0.0,1.0,1.0) * Nb.y;
    
    return c0 + c1 + c2;
}

// Function 583
int GetFocusGlyph(int i) { return imod(RandInt(i), glyphCount); }

// Function 584
void setMoonTexture(vec2 fragCoord, inout vec4 col){
    //Write moon texture to alpha channel.
    vec2 uv_ = fragCoord/iResolution.xy;
    vec2 uv = uv_;
    vec3 p = vec3(uv, 0.0);
    
    //Mix some noises together for a blotchy texture.
    float base = getPerlinNoise(p, 2.6);
	float worley = worley(p, 2.0);
    col.a = saturate(remap(base, 0.45*worley, 1.0, 0.0, 1.0));;
}

// Function 585
vec4 texture_blurred_quantized(in sampler2D tex, vec2 uv, vec3 q)
{
    return (quantize(texture(iChannel0, uv), q)
		+ quantize(texture(iChannel0, vec2(uv.x+1.0, uv.y)), q)
		+ quantize(texture(iChannel0, vec2(uv.x-1.0, uv.y)), q)
		+ quantize(texture(iChannel0, vec2(uv.x, uv.y+1.0)), q)
		+ quantize(texture(iChannel0, vec2(uv.x, uv.y-1.0)), q))/5.0;
}

// Function 586
vec4 textureGamma(samplerCube sampler, vec3 v)
{
    vec4 col = texture(sampler, v);
    #if GAMMA
    	return pow(col, vec4(2.2));
    #else
        return col;
    #endif
}

// Function 587
bool DrawContext_OnCanvas( DrawContext drawContext )
{
    vec2 vUV = drawContext.vUV;
    if ( (vUV.x >= 0.0f) && (vUV.y >= 0.0f) && (vUV.x < 1.0f) && (vUV.y < 1.0f) ) 
    {    
    	return true;
    }
    return false;
}

// Function 588
void UI_RenderFont( inout UIContext uiContext, PrintState state, LayoutStyle style, RenderStyle renderStyle )
{
    if( uiContext.bPixelInView )
    {
        RenderFont( state, style, renderStyle, uiContext.vWindowOutColor.rgb );
    }
}

// Function 589
vec3 checkerTexture(vec2 uv, vec3 dir){
	float sines=sin(dir.x*10.0)*sin(dir.y*10.0)*sin(dir.z*10.0);
    if (sines>0.0)
        return vec3(1.0);
    else
        return vec3(0.0);
}

// Function 590
ivec2 text_uv(vec2 fragCoord)				{ return ivec2(floor(fragCoord)) >> g_text_scale_shift; }

// Function 591
int glyph_bit(uint glyph, int index)
{
    if (glyph >= uint(NUM_GLYPHS))
        return 0;
    uint data = uint(FONT_BITMAP.data[(glyph<<1) + uint(index>=32)]);
    return int(uint(data >> (index & 31)) & 1u);
}

// Function 592
int textArray(int u){
 int o[16]=int[16](33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48);
 //above is a demo of a string of the alphabeticallysorted letters [A till P]
 return o[u%16];
 return o[u];
}

// Function 593
float text(vec2 p) {
    p *= 1024.;
    const vec2 glyph_size = vec2(21., 44.);
    vec2 char_pos = floor(p / glyph_size);
    p = fract(p / glyph_size) - .5;
    vec2 t_glyph_size = iChannelResolution[1].xy / 16.;
    
    char_pos.y = -1. - char_pos.y;
    float char = getchar(char_pos);
    vec2 t_glyph_pos = vec2(mod(char, 16.), 15. - floor(char / 16.));
    p.x *= glyph_size.x / glyph_size.y;
    return smoothstep(.53, .48, texture(iChannel1, (t_glyph_pos + .5 + p) * t_glyph_size / iChannelResolution[1].xy, -8.).a);
}

// Function 594
void paintWorldDimText(inout vec4 col, in uvec2 coord, in vec3 bg)
{   
    vec2 uuv = vec2(coord) / iResolution.y;
    //float H = 10.;
    //float gly = 1. / H * 400.;
    float scale = 3.2;
    vec2 uv = /*uuv * gly - vec2(.5,gly-1.-.5)*/ vec2(coord) / iResolution.x * 22.0 * scale - vec2(.5,1.);
    float px = 22. / iResolution.x * scale;

    float x = 100.;
    float cp = 0.;
    vec4 cur = vec4(0,0,0,0.5);
    vec4 us = cur;
    float ital = 0.0;

    //int lnr = int(floor(uv.y/2.));
    //uv.y = mod(uv.y,2.0)-0.5;

    if (uv.y >= -1.0 && uv.y <= 1.0) {
        ITAL W_ o_ r_ l_ d_ _ D_ i_ m_ _dotdot _
        
    	uvec3 virtualDim = uvec3(globalWorld.virtualDim);
       	
        NOITAL /*DARKBLUE u_ v_ e_ c_ _3*/ BLACK _open3
        
		for (int i = 0; i < 3; ++i)
        {
            DARKGREEN
            DECIMAL(x)
            if (i != 2) { _ _ }
            
        }
        
        BLACK _ _close3
       
        vec3 clr = vec3(0.0);

        float weight = 0.01+cur.w*0.05;//min(iTime*.02-.05,0.03);//+.03*length(sin(uv*6.+.3*iTime));//+0.02-0.06*cos(iTime*.4+1.);
        col = mix(col, vec4(us.rgb, 1.0), smoothstep(weight+px, weight-px, x));
    }
    
}

// Function 595
float text_r(vec2 U) {
    initMsg;
    U.x+=4.*(0.5-0.2812*(res.x/0.5));
    C(82);C(101);C(115);C(101);C(116);
    endMsg;
}

// Function 596
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

// Function 597
Rect GetFontRect( PrintState state, LayoutStyle style, bool initialLineOffset )
{
    Rect rect;
    
    rect.vPos = state.vLayoutStart;
    if ( initialLineOffset )
    {
    	rect.vPos.y += style.vSize.y * (style.fLineGap + g_fFontAscent);
    }
	rect.vPos.y -= style.vSize.y * (g_fFontAscent);
    rect.vSize.x = state.vCursorPos.x - state.vLayoutStart.x;
    rect.vSize.y = style.vSize.y * ( g_fFontAscent + g_fFontDescent );
    
    return rect;
}

// Function 598
int getNumberOfGlyphs(int n){for(int i=0;i<wordl;i++){if(n<=0)return i;n/=10;}return wordl;}

// Function 599
float xorTextureGradBox( in vec2 pos, in vec2 ddx, in vec2 ddy )
{
    float xor = 0.0;
    for( int i=0; i<8; i++ )
    {
        // filter kernel
        vec2 w = max(abs(ddx), abs(ddy)) + 0.01;  
        // analytical integral (box filter)
        vec2 f = 2.0*(abs(fract((pos-0.5*w)/2.0)-0.5)-abs(fract((pos+0.5*w)/2.0)-0.5))/w;
        // xor pattern
        xor += 0.5 - 0.5*f.x*f.y;
        
        // next octave        
        ddx *= 0.5;
        ddy *= 0.5;
        pos *= 0.5;
        xor *= 0.5;
    }
    return xor;
}

// Function 600
vec3 SampleTexture( const in float fTexture, const in vec2 _vUV )
{
    vec2 vTexureSize = vec2(64);    
    float fXCount = 10.0; // TODO : base on resolution?
    
    vec2 vTexturePos = vec2( mod( floor(fTexture), fXCount ), floor( fTexture / fXCount ) ) * vTexureSize;
    
    vec2 vPixel = vTexturePos + fract(_vUV / vTexureSize) * vTexureSize;
    
    vec2 vSampleUV = vPixel / iChannelResolution[1].xy;
    
    vec4 vSample = texture( iChannel1, vSampleUV );
    
    return vSample.xyz;
}

// Function 601
float approx_font_dist(vec2 p, int cidx) {

    float d = max(abs(p.x) - 0.25,
                  max(p.y - 0.3, -0.28 - p.y));
    
    vec2 cpos = vec2(float(cidx%16), float(15-cidx/16));
    vec2 uv = font_from_screen(p, cpos);
    
    float fd = sample_dist_gaussian(uv); 
        
    
    d = max(d, fd);
        
    
    return d;
    
}

// Function 602
vec4 getTexture(float id, vec2 c) {
    vec2 gridPos = vec2(mod(id, 16.), floor(id / 16.));
	return textureLod(iChannel2, 16. * (c + gridPos) / iChannelResolution[2].xy, 0.0);
}

// Function 603
vec2 TransformFromCanvasTextureToFramedTexture(
	vec2 canvasTextureCoord,
	vec2 canvasTextureSize,
	vec2 framedTextureSize)
{	
	vec2 result = (canvasTextureCoord / canvasTextureSize);

	float canvasAspectRatio = (canvasTextureSize.x / canvasTextureSize.y);
	float framedAspectRatio = (framedTextureSize.x / framedTextureSize.y);

	if (framedAspectRatio < canvasAspectRatio)
	{
		float relativeAspectRatio = (canvasAspectRatio / framedAspectRatio);

		result.x *= relativeAspectRatio;
		result.x -= (0.5 * (relativeAspectRatio - 1.0));
	}
	else
	{
		float relativeAspectRatio = (framedAspectRatio / canvasAspectRatio);

		result.y *= relativeAspectRatio;
		result.y -= (0.5 * (relativeAspectRatio - 1.0));
	}

	return result;
}

// Function 604
vec2 	UIStyle_FontPadding() 			{ return vec2(8.0, 2.0); }

// Function 605
vec4 put_text_step_count(vec4 col, vec2 uv, vec2 pos, float scale, int count)
{
	float unit = asp * scale * 0.1;
    float h = 0.;
    vec2 sc = vec2(unit, unit*0.8);
    
    int d = count % 10;
    int t = count / 10;
    
    h = max(h, word_map(uv, pos+vec2(unit*0.35, 0.), 48+d, sc));
    
    if(t > 0)
    {
    	h = max(h, word_map(uv, pos, 48+t, sc));
    }
    
    col = mix(col, vec4(1.-vec3(h), 1.), h);
    
    return col;
}

// Function 606
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

// Function 607
void glyph_9()
{
  MoveTo(float2(1.8,1.4));
  Bez3To(1.8*x+y*0.6,x*0.2+0.6*y,x*0.2+y*1.4);
  Bez3To(x*0.2+2.2*y,1.8*x+y*2.2,float2(1.8,1.4));
  Bez3To(float2(1.9,0.0),float2(1.0,-0.2),float2(0.4,0.2));
}

// Function 608
vec3 betterTextureSample64(sampler2D tex, vec2 uv) {	
	float textureResolution = 64.0;
	uv = uv*textureResolution + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*(3.0-2.0*fuv); // fuv*fuv*fuv*(fuv*(fuv*6.0-15.0)+10.0);;
	uv = (uv - 0.5)/textureResolution;
	return textureLod(tex, uv, 0.0).rgb;
}

// Function 609
vec4 SphereTexture(sampler2D tex, vec3 coord, vec2 scale)
{
    vec2 sphereUV;
    sphereUV.x = 0.5 + (atan(coord.x, coord.z)) / (2.0 * PI);
    sphereUV.y = 0.5 + (asin(coord.y)) / PI;
    return texture(tex, sphereUV * scale);
}

// Function 610
vec4 textureNoTile_3weights( sampler2D samp, in vec2 uv )
{
    vec4 res = vec4(0.);
    int sampleCnt = 0; // debug vis
    
    // compute per-tile integral and fractional uvs.
    // flip uvs for 'odd' tiles to make sure tex samples are coherent
    vec2 fuv = mod( uv, 2. ), iuv = uv - fuv;
    vec3 BL_one = vec3(0.,0.,1.); // xy = bot left coords, z = 1
    if( fuv.x >= 1. ) fuv.x = 2.-fuv.x, BL_one.x = 2.;
    if( fuv.y >= 1. ) fuv.y = 2.-fuv.y, BL_one.y = 2.;
    
    
    // weight orthogonal to diagonal edge = 3rd texture sample
    vec2 iuv3;
    float w3 = (fuv.x+fuv.y) - 1.;
    if( w3 < 0. ) iuv3 = iuv + BL_one.xy, w3 = -w3; // bottom left corner, offset negative, weight needs to be negated
    else iuv3 = iuv + BL_one.zz; // use transform from top right corner
    w3 = smoothstep(BLEND_WIDTH, 1.-BLEND_WIDTH, w3);
    
    // if third sample doesnt dominate, take first two
    if( w3 < 0.999 )
    {
        // use weight along long diagonal edge
        float w12 = dot(fuv,vec2(.5,-.5)) + .5;
        w12 = smoothstep(1.125*BLEND_WIDTH, 1.-1.125*BLEND_WIDTH, w12);

        // take samples from texture for each side of diagonal edge
        if( w12 > 0.001 ) res +=     w12  * texture( samp, transformUVs( iuv + BL_one.zy, uv ) ), sampleCnt++;
        if( w12 < 0.999 ) res += (1.-w12) * texture( samp, transformUVs( iuv + BL_one.xz, uv ) ), sampleCnt++;
    }
    
	// first two samples aren't dominating, take third
    if( w3 > 0.001 ) res = mix( res, texture( samp, transformUVs( iuv3, uv ) ), w3 ), sampleCnt++;

    
    // debug vis: colour based on num samples taken for vis purposes
    if( iMouse.z > 0. )
    {
        if( sampleCnt == 1 ) res.rb *= .25;
        if( sampleCnt == 2 ) res.b *= .25;
        if( sampleCnt == 3 ) res.gb *= .25;
    }
    
    return res;
}

// Function 611
float shadeText(vec3 ray_start, vec3 ray_dir)
{
    vec4 hit = traceText(ray_start, ray_dir);
   	float color;
   	if (hit.w < 0.0) {
    	color = 0.;
   	} else {
      	vec3 dir = hit.xyz - ray_start;
      	vec3 norm = dNormal(hit.xyz);
       	color = max(0., dot(norm, vec3(sin(iTime)*.05, cos(iTime)*.05, 1.))*smoothstep(ray_max,0., hit.z));
   	}
     color+=hit.w*.06*(1.-color);
	return pow(color, 2.);
}

// Function 612
vec3 textureBlured(samplerCube tex, vec3 tc) {
   	vec3 r = textureAVG(tex,vec3(1.0,0.0,0.0));
    vec3 t = textureAVG(tex,vec3(0.0,1.0,0.0));
    vec3 f = textureAVG(tex,vec3(0.0,0.0,1.0));
    vec3 l = textureAVG(tex,vec3(-1.0,0.0,0.0));
    vec3 b = textureAVG(tex,vec3(0.0,-1.0,0.0));
    vec3 a = textureAVG(tex,vec3(0.0,0.0,-1.0));
        
    float kr = dot(tc,vec3(1.0,0.0,0.0)) * 0.5 + 0.5; 
    float kt = dot(tc,vec3(0.0,1.0,0.0)) * 0.5 + 0.5;
    float kf = dot(tc,vec3(0.0,0.0,1.0)) * 0.5 + 0.5;
    float kl = 1.0 - kr;
    float kb = 1.0 - kt;
    float ka = 1.0 - kf;
    
    kr = somestep(kr);
    kt = somestep(kt);
    kf = somestep(kf);
    kl = somestep(kl);
    kb = somestep(kb);
    ka = somestep(ka);    
    
    float d;
    vec3 ret;
    ret  = f * kf; d  = kf;
    ret += a * ka; d += ka;
    ret += l * kl; d += kl;
    ret += r * kr; d += kr;
    ret += t * kt; d += kt;
    ret += b * kb; d += kb;
    
    return ret / d;
}

// Function 613
vec3 drawText( in vec4 fragColor, in vec2 fragCoord ) {
    float display_width = 1010.;
    float cc = floor(display_width / (g_cw * (1. + g_cwb))); // character count per line
    
    vec2 uv = (fragCoord.xy) / iResolution.xx;
    uv.y = iResolution.y/iResolution.x - uv.y;  // type from top to bottom, left to right   
    uv *= display_width;

    int cs = int(floor(uv.x / (g_cw * (1. + g_cwb))) + cc * floor(uv.y/(g_ch * (1. + g_chb))));

    uv = mod_uv(uv);
    uv.y = g_ch * (1. + g_chb) - uv.y; // paint the character from the bottom left corner
    vec3 ccol = .35 * vec3(.1, .3, .2) * max(smoothstep(3., 0., uv.x), smoothstep(5., 0., uv.y));   
    uv -= vec2(g_cw * g_cwb * .5, g_ch * g_chb * .5);
    
    float tx = 10000.;
    int idx = 0;
    
    NL 
    NL 
    NL 
    NL 
    NL 
    NL 
    SP SP SP SP SP SP SP SP SP SP SP SP SP SP SP SP G A M E SP O V E R 
    NL
        
    vec3 tcol = vec3(1.0, 0.7, 0.0) * smoothstep(.2, .0, tx);
    
    vec3 terminal_color = tcol;
    
    return terminal_color;
}

// Function 614
vec3 rayToTexture( vec3 p ) {
    return (p*SCALE + vec3(0.5,0.5,0.5));
}

// Function 615
vec3 Texture(in vec3 pos, in vec3 norm, in float material)
{
    vec3 checker = vec3(clamp(Checkers(pos.xz), 0.4, 0.7));
    vec3 blank   = vec3(1.0);
    
    return mix(checker, blank, material);
}

// Function 616
float escherTextureContour(vec2 p, float linewidth, float pixel_size)
{
    vec2 pp = mod(p,1.0);
    
    float d = 10000000.0;
    for(int i=0; i<vert.length(); ++i)
    {       
        for(int j=0; j<textureTiles.length(); ++j)
        {
            d = min(d, PointSegDistance2(pp+textureTiles[j], vert[i], vert[i+1%vert.length()]));
        }
    }
    
    d = smoothstep(0.0, 1.0, (sqrt(d)-linewidth)/pixel_size);
    
    return d;
}

// Function 617
float splineTexture(float t, int y)
{
    int x = int(t);
    t = fract(t);
    float p0  =  getTexFloat(x-1, y);
    float p1  =  getTexFloat(x, y);
    float p2  =  getTexFloat(x+1, y);
    float p3  =  getTexFloat(x+2, y);
    p0 = spline(p0,p1,p2,p3,t);

    return p0;
}

// Function 618
float text(vec2 uv)
{
    float col = 0.0;
    float t=mod(iTime,38.);// scroll duration 
    vec2 center = vec2(40.,1.);
    
    print_pos = (vec2(30.-t*8.0,1.0+abs(4.*sin(t*2.))) - vec2(STRWIDTH(1.0),STRHEIGHT(1.0))/2.0);
    
    col += char(ch_S,uv);
    col += char(ch_H,uv);
    col += char(ch_A,uv);
    col += char(ch_D,uv);
    col += char(ch_E,uv);
    col += char(ch_R,uv);
    col += char(ch_T,uv);
    col += char(ch_O,uv);
    col += char(ch_Y,uv);
    
    col += char(ch_spc,uv);
  
	col += char(ch_A,uv);
    col += char(ch_T,uv);
    col += char(ch_A,uv);
    col += char(ch_R,uv);
    col += char(ch_I,uv);
    
    col += char(ch_spc,uv);
    
    col += char(ch_T,uv);
    col += char(ch_C,uv);
    col += char(ch_B,uv);
    
    
    col += char(ch_spc,uv);
    
    col += char(ch_M,uv);
    col += char(ch_E,uv);
    col += char(ch_G,uv);
    col += char(ch_A,uv);
    
    col += char(ch_spc,uv);

    col += char(ch_S,uv);
    col += char(ch_C,uv);
    col += char(ch_R,uv);
    col += char(ch_O,uv);
    col += char(ch_L,uv);
    col += char(ch_L,uv);
    col += char(ch_spc,uv);
  
    return col;
}

// Function 619
void updateText(  inout vec4 color, vec2 coord ) {
    uv = (2.*coord/iResolution.y-1.);
    if( abs(uv.y) < .2 ) {
        ivec4 data = LoadVec4(ivec2(3,32));
        
        if( data.x > 0 ) {
		   SetTextPosition(2.5,-0.5);   
		   float c = 0.0;
		   _Y _o _u _ _f _o _u _n _d _ 
                
           if( data.x == 6 ) {    
              _a _ _n _e _w _ _s _w _o _r _d _ _add
			    c += drawInt(data.y);  
           } else if( data.x == 10 ) {
                _f _o _o _d _ _add
			    c += drawInt(data.y);  
           } else {
               _a _
               if( data.x == 7 ) {
                   _R _e _d
               }
               else if( data.x == 8 ) {
                   _G _r _e _e _n
               }
               else if( data.x == 9 ) {
                   _B _l _u _e
               }
               _ _K _e _y
           }
		   color = vec4(1,1,1,min(2.,c * 2.));
        } else if( data.x < 0 ) {   
		   SetTextPosition(2.5,-0.5);
		   float c = 0.0;
           _Y _o _u _ _d _i _e _d
		   color = vec4(1,1,1,min(2.,c * 2.));               
        } else {
           color = texelFetch(iChannel2, ivec2(coord),0); 
           color.a = max(0., color.a - 1./60.);
        }         
    }
}

// Function 620
vec3 TextureSource(vec2 uv)
{
	return texture(iChannel0, uv).rgb;;
}

// Function 621
void glyph_0()
{
  MoveTo(x);
  Bez3To(-0.2*x,2.0*y-x*0.2,float2(1.0,2.0));
  Bez3To(2.0*y+x*2.2,2.2*x,x);
  MoveTo(x*0.85+y*0.7);
  LineTo(x*1.15+y*1.3);
}

// Function 622
float PrintText(vec2 uv, int technique)
{
    float col = 0.0;
    TEXT_MODE = NORMAL;  
    
    // RNM
    if (technique == TECHNIQUE_RNM)
    {
        print_pos = vec2(iResolution.x*0.5 - 0.5*STRHEIGHT(16.0), 2.0 + STRHEIGHT(0.0));

        col += char(ch_R,uv);
        col += char(ch_e,uv);
        col += char(ch_o,uv);
        col += char(ch_r,uv);
        col += char(ch_i,uv);
        col += char(ch_e,uv);
        col += char(ch_n,uv);
        col += char(ch_t,uv);
        col += char(ch_e,uv);
        col += char(ch_d,uv);

        col += char(ch_spc,uv);

        col += char(ch_N,uv);
        col += char(ch_o,uv);
        col += char(ch_r,uv);
        col += char(ch_m,uv);
        col += char(ch_a,uv);
        col += char(ch_l,uv);

        col += char(ch_spc,uv);

        col += char(ch_M,uv);
        col += char(ch_a,uv);
        col += char(ch_p,uv);
        col += char(ch_p,uv);
        col += char(ch_i,uv);
        col += char(ch_n,uv);
        col += char(ch_g,uv);
    }
    else if (technique == TECHNIQUE_PartialDerivatives)
    {
        print_pos = vec2(iResolution.x*0.5 - STRHEIGHT(6.0), 2.0 + STRHEIGHT(0.0));

        col += char(ch_P,uv);
        col += char(ch_a,uv);
        col += char(ch_r,uv);
        col += char(ch_t,uv);
        col += char(ch_i,uv);
        col += char(ch_a,uv);
        col += char(ch_l,uv);

        col += char(ch_spc,uv);

        col += char(ch_D,uv);
        col += char(ch_e,uv);
        col += char(ch_r,uv);
        col += char(ch_i,uv);
        col += char(ch_v,uv);
        col += char(ch_a,uv);
        col += char(ch_t,uv);
        col += char(ch_i,uv);
        col += char(ch_v,uv);
        col += char(ch_e,uv);
        col += char(ch_s,uv);
    }
    else if (technique == TECHNIQUE_Whiteout)
    {
        print_pos = vec2(iResolution.x*0.5 - STRHEIGHT(2.0), 2.0 + STRHEIGHT(0.0));

        col += char(ch_W,uv);
        col += char(ch_h,uv);
        col += char(ch_i,uv);
        col += char(ch_t,uv);
        col += char(ch_e,uv);
        col += char(ch_o,uv);
        col += char(ch_u,uv);
        col += char(ch_t,uv);
    }    
    else if (technique == TECHNIQUE_UDN)
    {
        print_pos = vec2(iResolution.x*0.5 - STRHEIGHT(1.0), 2.0 + STRHEIGHT(0.0));

        col += char(ch_U,uv);
        col += char(ch_D,uv);
        col += char(ch_N,uv);
    }    
    else if (technique == TECHNIQUE_Unity)
    {
        print_pos = vec2(iResolution.x*0.5 - STRHEIGHT(2.0), 2.0 + STRHEIGHT(0.0));

        col += char(ch_U,uv);
        col += char(ch_n,uv);
        col += char(ch_i,uv);
        col += char(ch_t,uv);
        col += char(ch_y,uv);
    } 
    else if (technique == TECHNIQUE_Linear)
    {
        print_pos = vec2(iResolution.x*0.5 - STRHEIGHT(2.0), 2.0 + STRHEIGHT(0.0));

        col += char(ch_L,uv);
        col += char(ch_i,uv);
        col += char(ch_n,uv);
        col += char(ch_e,uv);
        col += char(ch_a,uv);
        col += char(ch_r,uv);
    } 
    else// if (technique == TECHNIQUE_Overlay)
    {
        print_pos = vec2(iResolution.x*0.5 - STRHEIGHT(2.0), 2.0 + STRHEIGHT(0.0));

        col += char(ch_O,uv);
        col += char(ch_v,uv);
        col += char(ch_e,uv);
        col += char(ch_r,uv);
        col += char(ch_l,uv);
        col += char(ch_a,uv);
        col += char(ch_y,uv);
    }    
    
    return col;
}

// Function 623
void SetGlyphColor(float r,float g,float b){drawColor=vec3(b,g,b);}

// Function 624
vec4 textureNoTile_4weights( sampler2D samp, in vec2 uv )
{
    // compute per-tile integral and fractional uvs.
    // flip uvs for 'odd' tiles to make sure tex samples are coherent
    vec2 fuv = mod( uv, 2. ), iuv = uv - fuv;
    vec3 BL_one = vec3(0.,0.,1.); // xy = bot left coords, z = 1
    if( fuv.x >= 1. ) fuv.x = 2.-fuv.x, BL_one.x = 2.;
    if( fuv.y >= 1. ) fuv.y = 2.-fuv.y, BL_one.y = 2.;
    
    // smoothstep for fun and to limit blend overlap
    vec2 b = smoothstep(0.25,0.75,fuv);
    
    // fetch and blend
    vec4 res = mix(
        		mix( texture( samp, transformUVs( iuv + BL_one.xy, uv ) ), 
                     texture( samp, transformUVs( iuv + BL_one.zy, uv ) ), b.x ), 
                mix( texture( samp, transformUVs( iuv + BL_one.xz, uv ) ),
                     texture( samp, transformUVs( iuv + BL_one.zz, uv ) ), b.x),
        		b.y );

    // debug vis: colour based on num samples taken for vis purposes - always takes 4 samples!
    if( iMouse.z > 0. ) res.gb *= .25;
    
    return res;
}

// Function 625
float text_ms(vec2 U) {
    initMsg;
    U.x+=4.*(0.5-0.2812*(res.x/0.5));
    C(83);C(112);C(97);C(119);C(110);
    endMsg;
}

// Function 626
vec4 texture_Bicubic( sampler2D tex, vec2 t )
{
    vec2 res = vec2(textureSize(tex,0));
    vec2 p = res*t - 0.5;
    vec2 f = fract(p);
    vec2 i = floor(p);

    return spline( f.y, spline( f.x, SAM(-1,-1), SAM( 0,-1), SAM( 1,-1), SAM( 2,-1)),
                        spline( f.x, SAM(-1, 0), SAM( 0, 0), SAM( 1, 0), SAM( 2, 0)),
                        spline( f.x, SAM(-1, 1), SAM( 0, 1), SAM( 1, 1), SAM( 2, 1)),
                        spline( f.x, SAM(-1, 2), SAM( 0, 2), SAM( 1, 2), SAM( 2, 2)));
}

// Function 627
vec4 texture2( sampler2D sam, vec2 uv )
{
#ifndef SHOW_DERIVATIVES    
    return texture(sam,uv);
#else    
    float res = float(textureSize(sam,0).x);
    uv = uv*res - 0.5;
    vec2 iuv = floor(uv);
    vec2 f = fract(uv);
	vec4 rg1 = textureLod( sam, (iuv+ vec2(0.5,0.5))/res, 0.0 );
	vec4 rg2 = textureLod( sam, (iuv+ vec2(1.5,0.5))/res, 0.0 );
	vec4 rg3 = textureLod( sam, (iuv+ vec2(0.5,1.5))/res, 0.0 );
	vec4 rg4 = textureLod( sam, (iuv+ vec2(1.5,1.5))/res, 0.0 );
	return mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
#endif    
}

// Function 628
TeletextState TeletextState_Default()
{
    TeletextState state;
    
    state.char = 0x20;
    
    state.iFgCol = 7;
    state.iBgCol = 0;
    
    state.iHeldChar = 0x20;
    state.bHeldSeparated = false;
    
    state.bDoubleHeight = false;
    state.bFlash = false;    
    state.bGfx = false;
    state.bConceal = false;
    state.bSeparatedGfx = false;
    state.bHoldGfx = false;
    
    state.cmd = -1;
    
    return state;
}

// Function 629
int int_glyph(int number, int index)
{
    if (uint(index) >= uint(MAX_POW10_EXPONENT))
        return _SPACE_;
    if (number <= 0)
        return index == 0 ? _0_ : _SPACE_;
    uint power = pow10(uint(index));
    return uint(number) >= power ? _0_ + int((uint(number)/power) % 10u) : _SPACE_;
}

// Function 630
vec3 textureBox2(vec3 p,vec3 p2)
{
    vec3 ap=abs(p),f=step(ap.zxy,ap.xyz)*step(ap.yzx,ap.xyz);
    vec2 uv=f.x>.5?p.yz:f.y>.5?p.xz:p.xy;
    vec3 n=normalize(-transpose(boxxfrm)*(f*sign(p)));
    float l=clamp(-normalize(p2-vec3(0,1,0)).y,0.,1.);
    vec3 d=1.*(1.-smoothstep(-1.,2.5,p2.y))*vec3(0.3,0.3,.7)+smoothstep(0.95,.97,l)*clamp(-n.y,0.,1.)*2.*vec3(1,1,.8)+
        	smoothstep(0.9,1.,l)*clamp(-n.y,0.,1.)*vec3(1,1,.8);
    return texture(iChannel3,uv).rgb*d;
}

// Function 631
vec3 textureNormal(vec2 uv) {
    vec3 normal = texture( iChannel1, 100.0 * uv ).rgb;
    normal.xy = 2.0 * normal.xy - 1.0;
    normal.z = sqrt(iMouse.x / iResolution.x);
    return normalize( normal );
}

// Function 632
G texture(iChannel1
void mainImage(out vec4 O, vec2 U) {    
    vec3  R = iResolution, 
          D = normalize( vec3( U+U, -3.5*R.y ) - R ),
          p = 2./R; 
    p.xy -= 2.*iTime;
    O-=O; vec4 T = O+1.;
    for ( float l,i=0.; i++ < 2e2; p += D )
        R = fract(p) - G,(ceil(p)-.5)/32.).rgb,
        l = .01* i / length( R - dot(R,D)*D ),  
        O += T *  vec4(.4,.6,1,0) * min( 1e3, l*l*l )/i/i,
        T /= exp( max(0., 3.* G,p/128.).a -2.) * vec4(.5,1,2,0) );
    O = sqrt(O); 
}

// Function 633
vec4 put_text_map(vec4 col, vec2 uv, vec2 pos, float scale)
{
	float unit = asp * scale * 0.1;
    float h = 0.;
    vec2 sc = vec2(unit, unit*0.8);
    
    // M
    h = max(h, word_map(uv, pos, 77, sc));
    // a
    h = max(h, word_map(uv, pos+vec2(unit*0.35, 0.), 97, sc));
    // p
    h = max(h, word_map(uv, pos+vec2(unit*0.7, 0.), 112, sc));
    
    col = mix(col, vec4(1.-vec3(h), 1.), h);
    
    return col;
}

// Function 634
vec3 tileTexture(in vec2 uv, in int tileNr, in float tmod)
{
    vec3 col = vec3(0.);
    vec3 light = vec3(0.);
    if (tileNr == SAND)
    {
        float h = noise1(uv*10. + 10.*curTilePos);
        h -= .7 * noise1(uv*22. + 10.*curTilePos);
        h += .3 * noise1(uv*43. + 10.*curTilePos);
        col = 1.6*vec3(.6,.2+.1*h,0.24*h) * (.3+.7*h);
		col *= min(1., 7.*min(abs(abs(uv.x)-1.), abs(abs(uv.y)-1.)));
    }
    else if (tileNr == WALL)
    {
        col = vec3(squareHeight(uv));
        light = lighting(squareNorm(uv));
    }
    else if (tileNr == DIAMOND)
    {
        float dia = diaHeight(uv);
        if (dia > 0.)
        {
        	col = diaCol(tmod);
        	col *= pow(dia, .4);
            light = lighting(diaNorm(uv));
        }
    }
    else if (tileNr == PLAYER)
    {
        int tm = int(tmod+.5);
        mat2 rm = mat2(1,0, 0,1);
        if (tm == S_DOWN) rm = mat2(-1,0, 0,-1);
        if (tm == S_LEFT) rm = mat2(0,-1, -1,0);
        if (tm == S_RIGHT) rm = mat2(0,1, 1,0);
        vec2 dm = playerDist(rm*uv);
        float v = smoothstep(0.1,0.,dm.x);	
		col = v * max(0., 1.-2.*dm.x) * .9 
            	* ( tm == S_SMASHED ? vec3(1,0,0) : playerColor(rm*uv, dm.y) );
        vec3 n = playerNorm(rm*uv);
        n.xy = rm*n.xy;        
        light = v*lightingDia(n);
    }
    else if (tileNr == STONE && length(uv) < 1.)
    {
        col = vec3(sphereHeight(uv));
        light = lighting(sphereNorm(uv));
    }

#if LIGHTING > 0    
    col = .6 * col + .7 * light;
#else
    
#endif
    return clamp(col, 0., 1.);
}

// Function 635
vec3 domeTexture(vec3 p) {
    vec3 q1 = p;
    q1.yz *= rot2d(PI);
    p = q1;
    vec3 col = vec3(.01);
    float x = acos(p.y/length(p));
    float y = atan(p.z, p.x) / 6.28;
    vec2 uv = vec2(x, y) + .5;

    float rize = .1 + sin(iTime/6.)*.1;

    vec2 muv = uv*vec2(1., 5.);
    vec2 id = floor(muv);
    muv = fract(muv) - .5;
    muv += vec2(rize, 0.);

    bool isMoon = false;

    if (id.y == 2.) {
        float muvl = length(muv);
        float ml = muvl * 1.5;
        vec3 mc = step(ml, .1) * vec3(noise(5. + muv*4. + iTime/50., 5));
        if (ml > .1) {
            mc += pow(.05 / muvl, 6.0);
        }
        if (ml < .15) {
            isMoon = true;
        }
        col += mc * vec3(.9, .6, .1);
    }

    vec2 suv = uv * vec2(30., 150.);
    vec2 sid = floor(suv);
    suv = fract(suv) - .5;

    float n = n21(sid);
    if (n > .7 && !isMoon) {
        col += step(length(suv + vec2(fract(n*3432.33) - .5, fract(n*78953.2) - .5)), .04*fract(n*123.123));
    }

    return col;
}

// Function 636
vec3 textureGamma(sampler2D channel, vec2 uv)
{
    vec3 tex = texture(channel, uv).xyz;
    return tex * tex;
}

// Function 637
vec4 textureSpherical(in sampler2D tex, in vec3 rp, float scale)
{
    float lrp = length(rp);
    vec2 uv1 = vec2(atan(rp.y, rp.x), acos(rp.z/lrp));
    vec2 uv2 = uv1; uv2.y = PI-uv1.y;
    float f=uv1.y;

    uv1=polar2Rect(uv1)*scale;
    uv2=polar2Rect(uv2)*scale;
    
    vec4 c1 = texture(tex, uv1);
    vec4 c2 = texture(tex, uv2);
    return mix(c1, c2, smoothstep(PI2-0.01, PI2+0.01, f));
}

// Function 638
float getDistortedTexture(vec2 uv){

    float strength = 0.5;
    
    // The texture is distorted in time and we switch between two texture states.
    // The transition is based on Worley noise which will shift the change of differet parts
    // for a more organic result
    float time = 0.5 * iTime + texture(iChannel1, 0.25*uv).g;
    
    float f = fract(time);
    
    // Get the velocity at the current location
    vec2 grad = getGradient(uv);
    uv *= 1.0;
    vec2 distortion = strength*vec2(grad.x, grad.y) + vec2(0, -0.3);

    // Get two shifted states of the texture distorted in time by the local velocity.
    // Loop the distortion from 0 -> 1 using fract(time)
    float distort1 = texture(iChannel1, uv + f * distortion).r;
    float distort2 = texture(iChannel1, uv + fract(time + 0.5) * distortion).r;

    // Mix between the two texture states to hide the sudden jump from 1 -> 0.
    // Modulate the value returned by the velocity.
    return (1.0-length(grad)) * (mix(distort1, distort2, abs(1.0 - 2.0 * f)));
}

// Function 639
vec3 getTexture(vec2 uv){
    vec4 textureSample = texture(iChannel0, uv);
	return sqrt(textureSample.rgb * textureSample.a);
}

// Function 640
vec4 text(vec2 uv, int value)
{    
    uint[16] font = uint[16](
        0xEAAAEu, // 0
        0x4644Eu, // 1
        0xE8E2Eu, // 2
        0xE8E8Eu, // 3
        0xAAE88u, // 4
        0xE2E8Eu, // 5
        0xE2EAEu, // 6
        0xE8888u, // 7
        0xEAEAEu, // 8
        0xEAE8Eu, // 9
        0xEAEAAu, // A
        0x6A6A6u, // B
        0xE222Eu, // C
        0x6AAA6u, // D
        0xE2E2Eu, // E
        0xE2E22u  // F
    );
    if (uv.x < 0. || uv.y < 0. || uv.x > 3. || uv.y > 5.)
        return vec4(0);
    value = int(mod(float(value), 16.));
    return vec4((font[value]>>int(uv.y*4.+uv.x-1.))&1u);
}

// Function 641
bool UIDrawContext_ScreenPosInView( UIDrawContext drawContext, vec2 vScreenPos )
{
    return Inside( vScreenPos, drawContext.clip );
}

// Function 642
vec3 TriplannarStarsTexture(vec3 p, vec3 normal)
{
    // the scale of the texture
    float scale = 0.25;
    // the sharpness of the blending between different axises
    float blendSharpness = 2.;
    // finding the different axise's color
    vec3 colX = texture(iChannel3, p.zy * scale).rgb;
    vec3 colY = texture(iChannel3, p.xz * scale).rgb;
    vec3 colZ = texture(iChannel3, p.xy * scale).rgb;
    
    // finding the blending amount for each axis
    vec3 bw = pow(abs(normal), vec3(blendSharpness));
    // making it so the total (x + y + z) is 1
    bw /= dot(bw, vec3(1.));
    
    // finding the final color
    return colX * bw.x + colY * bw.y + colZ * bw.z;
}

// Function 643
void texture_uv( inout vec4 fragColor, in vec2 uv)
{
    fragColor = texture(iChannel0, uv);
}

// Function 644
float linesTextureGradBox( in float p, in float ddx, in float ddy, int id )
{
    float N = float( 2 + 7*((id>>1)&3) );

    float w = max(abs(ddx), abs(ddy)) + 0.01;
    float a = p + 0.5*w;                        
    float b = p - 0.5*w;           
    return 1.0 - (floor(a)+min(fract(a)*N,1.0)-
                  floor(b)-min(fract(b)*N,1.0))/(N*w);
}

// Function 645
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

// Function 646
mat3 glyph_8_9_lowercase(float g) {
    // lowercase ==================================================
    GLYPH( 97)000000000,000000000,000000000,000001110,000010010,000010010,000001101,0,0);
    GLYPH( 98)000000000,000100000,000100000,000111100,000100010,000100010,000111100,0,0);
    GLYPH( 99)000000000,000000000,000000000,000011100,000100000,000100010,000011100,0,0);
    GLYPH(100)000000010,000000010,000000010,000011010,000100110,000100010,000011101,0,0);
    GLYPH(101)000000000,000000000,000111000,001000100,001111100,001000000,000111100,0,0);
    GLYPH(102)000011000,000100100,000100000,000111000,000100000,000100000,000100000,0,0);
    GLYPH(103)000000000,000011000,000100100,000100100,000011100,000100100,000011000,0,0);
    GLYPH(104)000100000,000100000,000100000,000101100,000110010,000100010,000100010,0,0);
    GLYPH(105)000000000,000010000,000000000,000010000,000010000,000010000,000011000,0,0);
    GLYPH(106)000000100,000000000,000000100,000000100,000000100,000100100,000011000,0,0);
    GLYPH(107)001000000,001000000,001000000,001001000,001110000,001010000,001001000,0,0);
    GLYPH(108)000100000,000100000,000100000,000100000,000100000,000100000,000011100,0,0);
    GLYPH(109)000000000,000000000,000100100,001011010,001000010,001000010,001000010,0,0);
    GLYPH(110)000000000,000000000,001011000,001100100,001000010,001000010,001000010,0,0);
    GLYPH(111)000000000,000000000,000011100,000100010,000100010,000100010,000011100,0,0);
    GLYPH(112)000000000,000000000,001011100,001100010,001100010,001011100,001000000,001000000,000000000);
    GLYPH(113)000000000,000000000,000111010,001000110,001000110,000111010,000000010,000000011,000000000);
    GLYPH(114)000000000,000000000,000101100,000110010,000100000,000100000,000100000,0,0);
    GLYPH(115)000000000,000000000,000011100,000100000,000011100,000000010,000100010,000011100,000000000);
    GLYPH(116)000010000,000010000,000011100,000010000,000010000,000010010,000001100,0,0);
    GLYPH(117)000000000,000000000,000100010,000100010,000100010,000100010,000011100,0,0);
    GLYPH(118)000000000,000000000,000100010,000100010,000010100,000010100,000001000,0,0);
    GLYPH(119)000000000,000000000,000100010,000100010,000101010,000101010,000010100,0,0);
    GLYPH(120)000000000,000000000,000100010,000010100,000001000,000010100,000100010,0,0);
    GLYPH(121)000000000,000000000,000100010,000100010,000100110,000011010,000000010,000011100,000000000);
    GLYPH(122)000000000,000000000,000111110,000000100,000001000,000010000,000111110,0,0);
    return mat3(0);
}

// Function 647
vec4 SampleTextureCatmullRom( vec2 uv, vec2 texSize )
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 samplePos = uv * texSize;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
    vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    
    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / w12;

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 texPos0 = texPos1 - vec2(1.0);
    vec2 texPos3 = texPos1 + vec2(2.0);
    vec2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    vec4 result = vec4(0.0);
    result += sampleLevel0( vec2(texPos0.x,  texPos0.y)) * w0.x * w0.y;
    result += sampleLevel0( vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos0.y)) * w3.x * w0.y;

    result += sampleLevel0( vec2(texPos0.x,  texPos12.y)) * w0.x * w12.y;
    result += sampleLevel0( vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos12.y)) * w3.x * w12.y;

    result += sampleLevel0( vec2(texPos0.x,  texPos3.y)) * w0.x * w3.y;
    result += sampleLevel0( vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += sampleLevel0( vec2(texPos3.x,  texPos3.y)) * w3.x * w3.y;

    return result;
}

// Function 648
StructBuffer TexturesSampler(sampler2D _buffer,vec2 uv){
	return GetStructBuffer(texture(_buffer,uv));
}

// Function 649
float checkersTexture( in vec3 p )
{
    vec3 q = floor(p);
    return mod( q.x+q.y+q.z, 2.0 );            // xor pattern
}

// Function 650
int textBuffer(int index)
{
	return floatBitsToInt(texelFetch(iChannel2, indexToCoord(index), 0).x);
}

// Function 651
vec4 texturenoise( vec3 r )
{
    vec3 uvw = r / iChannelResolution[3];
    return texture( iChannel3, uvw ) * 2. - 1.;
}

// Function 652
void drawTextNumber( inout vec4 c, vec2 p, vec2 tp, float s, float textNumber, vec4 color) { 
    c = mix(c, color,  clamp(mapNumber(p, tp, 10./s, textNumber).x, 0.,1.));
}

// Function 653
void draw_text(in vec2 p, in vec2 fragCoord, inout vec4 fragColor)
{
    print(val, vec2(1), vec2(iResolution.x - 50.,10.), vec3(0), fragCoord, fragColor);
}

// Function 654
float textureInvader(vec2 uv) {
	float y = 7.-floor((uv.y)*16.+4.);
	if(y < 0. || y > 7.) return 0.;
	float x = floor((abs(uv.x))*16.);
//	if(x < 0. || x > 14.) return 0.;
	float v=(y>6.5)? 6.:(y>5.5)? 40.:(y>4.5)? 47.:(y>3.5)? 63.:
			(y>2.5)? 27.:(y>1.5)? 15.:(y>0.5)? 4.: 8.;
	return floor(mod(v/pow(2.,x), 2.0)) == 0. ? 0.: 1.;
}

// Function 655
float glyph(vec2 uv, float e) 
{
    vec2 p = vec2(0.0);

    float r = 1.0;
    r = min(r, move_to(p, vec2(57, -162), uv, e) );
    r = min(r, conic_to(p, vec2(57, -96), vec2(107, -39), uv, e) );
    r = min(r, conic_to(p, vec2(158, 18), vec2(246, 43), uv, e) );
    r = min(r, conic_to(p, vec2(156, 100), vec2(156, 225), uv, e) );
    r = min(r, conic_to(p, vec2(156, 321), vec2(219, 395), uv, e) );
    r = min(r, conic_to(p, vec2(123, 475), vec2(123, 606), uv, e) );
    r = min(r, conic_to(p, vec2(123, 727), vec2(219, 816), uv, e) );
    r = min(r, conic_to(p, vec2(315, 905), vec2(455, 905), uv, e) );
    r = min(r, conic_to(p, vec2(578, 905), vec2(672, 831), uv, e) );
    r = min(r, conic_to(p, vec2(770, 927), vec2(889, 928), uv, e) );
    r = min(r, conic_to(p, vec2(942, 928), vec2(967, 895), uv, e) );
    r = min(r, conic_to(p, vec2(993, 862), vec2(993, 827), uv, e) );
    r = min(r, conic_to(p, vec2(993, 796), vec2(973, 781), uv, e) );
    r = min(r, conic_to(p, vec2(954, 766), vec2(934, 766), uv, e) );
    r = min(r, conic_to(p, vec2(909, 766), vec2(891, 782), uv, e) );
    r = min(r, conic_to(p, vec2(874, 799), vec2(874, 825), uv, e) );
    r = min(r, conic_to(p, vec2(874, 868), vec2(907, 881), uv, e) );
    r = min(r, conic_to(p, vec2(901, 883), vec2(887, 883), uv, e) );
    r = min(r, conic_to(p, vec2(787, 883), vec2(702, 803), uv, e) );
    r = min(r, conic_to(p, vec2(786, 725), vec2(786, 604), uv, e) );
    r = min(r, conic_to(p, vec2(786, 483), vec2(690, 394), uv, e) );
    r = min(r, conic_to(p, vec2(594, 305), vec2(455, 305), uv, e) );
    r = min(r, conic_to(p, vec2(340, 305), vec2(252, 369), uv, e) );
    r = min(r, conic_to(p, vec2(217, 328), vec2(217, 272), uv, e) );
    r = min(r, conic_to(p, vec2(217, 221), vec2(248, 181), uv, e) );
    r = min(r, conic_to(p, vec2(279, 141), vec2(326, 135), uv, e) );
    r = min(r, line_to(p, /*vec2(340, 133),*/ vec2(479, 133), uv, e) );
    r = min(r, line_to(p, /*vec2(561, 133),*/ vec2(606, 131), uv, e) );
    r = min(r, conic_to(p, vec2(651, 129), vec2(715, 115), uv, e) );
    r = min(r, conic_to(p, vec2(780, 102), vec2(831, 76), uv, e) );
    r = min(r, conic_to(p, vec2(964, 2), vec2(965, -158), uv, e) );
    r = min(r, conic_to(p, vec2(965, -275), vec2(830, -348), uv, e) );
    r = min(r, conic_to(p, vec2(696, -422), vec2(510, -422), uv, e) );
    r = min(r, conic_to(p, vec2(322, -422), vec2(189, -347), uv, e) );
    r = min(r, conic_to(p, vec2(57, -273), vec2(57, -162), uv, e) );
    r = min(r, move_to(p, vec2(164, -162), uv, e) );
    r = min(r, conic_to(p, vec2(164, -246), vec2(263, -310), uv, e) );
    r = min(r, conic_to(p, vec2(362, -375), vec2(512, -375), uv, e) );
    r = min(r, conic_to(p, vec2(659, -375), vec2(758, -311), uv, e) );
    r = min(r, conic_to(p, vec2(858, -248), vec2(858, -162), uv, e) );
    r = min(r, conic_to(p, vec2(858, -101), vec2(823, -62), uv, e) );
    r = min(r, conic_to(p, vec2(788, -23), vec2(716, -7), uv, e) );
    r = min(r, conic_to(p, vec2(645, 8), vec2(595, 11), uv, e) );
    r = min(r, line_to(p, /*vec2(545, 14),*/ vec2(453, 14), uv, e) );
    r = min(r, line_to(p, vec2(332, 14), uv, e) );
    r = min(r, conic_to(p, vec2(262, 10), vec2(213, -41), uv, e) );
    r = min(r, conic_to(p, vec2(164, -92), vec2(164, -162), uv, e) );
    r = min(r, move_to(p, vec2(276, 604), uv, e) );
    r = min(r, conic_to(p, vec2(276, 352), vec2(455, 352), uv, e) );
    r = min(r, conic_to(p, vec2(545, 352), vec2(600, 434), uv, e) );
    r = min(r, conic_to(p, vec2(633, 489), vec2(633, 606), uv, e) );
    r = min(r, conic_to(p, vec2(633, 858), vec2(455, 858), uv, e) );
    r = min(r, conic_to(p, vec2(365, 858), vec2(309, 776), uv, e) );
    r = min(r, conic_to(p, vec2(276, 721), vec2(276, 604), uv, e) );
    
	return r;    
}

// Function 656
mat4 texToSpace(vec2 coord, int id, vec2 size) {
    return mat4(
        vec4(texToSpace(coord, 0, id, size), 0),
        vec4(texToSpace(coord, 1, id, size), 0),
        vec4(texToSpace(coord, 2, id, size), 0),
        vec4(texToSpace(coord, 3, id, size), 0)
    );
}

// Function 657
float get_text(vec2 uv, vec2 pos, int ascii, vec2 unit, sampler2D buf)
{
    vec2 p = clamp(uv-pos, -unit/2., unit/2.);
    p += unit/2.;
    p /= unit*16.;
    
    return smoothstep(0.55, 0.46, textureLod(buf, get_text_position(ascii)+p, 1.).a);
}

