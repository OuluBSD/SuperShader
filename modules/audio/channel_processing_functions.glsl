// Reusable Channel Processing Audio Functions
// Automatically extracted from audio visualization-related shaders

// Function 1
float LeftWall(vec3 p){
	return WallWithOffset(p, vec3(-(ROOM_WIDTH)/2.0,0,0));
}

// Function 2
float colorBrightness(vec3 col)
{
	return col.r + col.b + col.g;
}

// Function 3
vec2 upper_right(vec2 uv)
{
    return fract((uv - 1.) * 0.5);
}

// Function 4
float bracketLeft(vec2 uv){
    uv.x = -uv.x;
    return bracketRight(uv);
}

// Function 5
vec3 calculateUpRight(vec3 normal, out vec3 tangent) {
    if (abs(normal.x) > abs(normal.y)) tangent = normalize(vec3(-normal.z, 0., normal.x));
    else tangent = normalize(vec3(0., normal.z, -normal.y));    
	return cross(normal, tangent);
}

// Function 6
vec3 MazeLeft(vec2 p, vec2 scale)
{
    vec3 res = BACKGROUND;

    vec2 gv = grid(p, scale); // The grid guide
    
    if (gv.x >= 0. && gv.y >= 0. &&
        gv.x <= 15. && gv.y <= 15.) {
        
        #if (DISPLAY_GRID == 1)
        	res = vec3(mod(gv.x + gv.y, 2.) * .05 + BACKGROUND);
        #endif
        
        // Indexing is upside down.
        int y = int(scale.y - gv.y - 5.);

    	float m = 0.;
		Q(0, B,B,B,B,B,B,B,B)
		Q(1, B,_,_,_,_,_,_,_)
		Q(2, B,_,B,B,B,_,B,_)
		Q(3, B,_,_,_,B,_,D,D)
		Q(4, B,_,B,B,B,_,B,_)
		Q(5, B,_,B,_,_,_,B,D)
		Q(6, B,_,_,_,B,_,B,_)
		Q(7, B,_,B,B,B,_,_,_)
		Q(8, B,_,B,_,_,_,_,_)
		Q(9, B,_,B,B,B,_,D,B)
		Q(10,B,_,B,_,D,_,D,_)
        Q(11,B,B,B,_,D,_,D,_)
		Q(12,B,_,_,_,_,_,D,_)
		Q(13,B,_,B,B,B,B,B,_)
		Q(14,B,_,_,_,_,_,_,_) // some of pants and jacket
		Q(15,B,B,B,B,B,B,B,B)
        
    	float ldx = 15. - gv.x; // Calculate the left  bit index
        float rdx = gv.x;       // Calculate the right bit index
        float bit = 0.;
        
        //if (gv.x >= 8.)	bit = mod(m / pow(4., ldx), 4.); // Decode
        //else            bit = mod(m / pow(4., rdx), 4.); // Mirror
        bit = mod(m / pow(4., rdx), 4.); // Decode
    	bit = floor(bit);                                // Sharpen    
    	
        // Colorize
             if (bit > 2.) res = vec3(.6471,.6471,.6471);
        else if (bit > 1.) res = vec3(1.,0.8941176471,0.6745098039);
        else if (bit > 0.) res = vec3(0.8549019608,0.4980392157, 0.1568627451);
    }
    
    return res;
}

// Function 7
vec2 stereoOscDetune(int instrument, float time, int waveform, float ampl, float freq, float p1, float p2, float p3)
{
    vec2 s = vec2(0.);
    int ns = int(instruments[instrument].detuneParams.y);
    if (ns<2)
       return stereoOsc(instrument, time, waveform, ampl, freq, p1, p2, p3);
    else
    {
       for (int n = ZERO; n < ns; n++)
       {
          float fo = freq*0.01*instruments[instrument].detuneParams.x*float(n - ns/2);
          float b = instruments[instrument].detuneParams.z*float(n - ns/2);
           
          s+= balance(b)*stereoOsc(instrument, time, waveform, ampl, freq + fo, p1, p2, p3); 
       }
    }
    return s;
}

// Function 8
vec2 calcSphericalCoordsInStereographicProjection(in vec2 screenCoord, in vec2 centralPoint, in vec2 FoVScale) {
	return calcSphericalCoordsFromProjections(screenCoord, centralPoint, FoVScale, true); 
}

// Function 9
vec4 getStereo(vec2 uv)
{
    uv = uv / 2.;
    
    float repeatWidth = 96.f;
    float XDPI = 500.;
    float EYE_SEP = XDPI*5.0;
    float OBS_DIST = XDPI*12.;
    
    float baseDepth = 120.;
    float depthFact = 60.;
    
    if(mod(floor(uv.y), 2.) == 0.)
    {
        repeatWidth = 96.f;
        XDPI = 500.;
        EYE_SEP = XDPI*5.0;
        OBS_DIST = XDPI*12.;
    
        baseDepth = 120.;
        depthFact = 60.;
    }
    else
    {
        repeatWidth = 96.f;
        XDPI = 1000.;
        EYE_SEP = XDPI*5.0;
        OBS_DIST = XDPI*12.;
    
        baseDepth = 120.;
        depthFact = 120.;
    }
    
    float repeatUv = repeatWidth / iResolution.x;
    float count = floor(iResolution.x / repeatWidth);

    for(int i = 0; i < 1000; i++)
    {
        float depth = 120. + 50. * getDepth(uv / iResolution.xy);
        float sep = EYE_SEP*depth/(depth + OBS_DIST);
        //sep = floor(sep);
        if(  uv.x <= 0.)
            break;
        uv.x -= sep;
        //uv.x = floor(uv.x);
    }

    /*
    uv = uv / iResolution.xy;
    for(int i = 0; i < 100; i++)
    {
        if(uv.x <= repeatUv) break; 
        
        float c = 1. / (count + 1.);
        float b = 1. / count;
        float a = (getDepth(vec2(uv.x, uv.y))) * 0.5;
        uv.x = uv.x - a;
    }
    
    uv.x - mod(uv.x, repeatUv);
    
    uv = uv * iResolution.xy;
    */    
        
    return getColor(uv);
    return uv.xxxx;
}

// Function 10
vec2 lower_right(vec2 uv)
{
    return fract((uv - vec2(1, 0.)) * 0.5);
}

// Function 11
vec2 right(vec2 uv){
    if(uv.x == SIZE+1.0){
    	return vec2(2.0, uv.y);
    }
	return uv+vec2( 1.0, 0.0);
}

// Function 12
vec3 stereographic(vec2 uv)
{
	return vec3(
		(uv.xy*2.)/(1.+dot(uv.xy,uv.xy)),
		(-1.+uv.x*uv.x+uv.y*uv.y)/(1.+uv.x*uv.x+uv.y*uv.y)
	);
}

// Function 13
Ray4Result stereographicProj ( in World _world, in vec3 _pos )
{
    // x;y;z;0 on hyperplane w=0
    // on sphere of radius 1 => (2x/(R+1);2y/(R+1);2x/(R+1);R-1/R+1) where R = x²+y²+z²
    
    float R = dot(_pos,_pos);
    
    vec4 onSphere = vec4(2.0*_pos,R-1.0) / vec4(R+1.0);

	Ray4 raySTC;
    raySTC.m_pos = onSphere;
    raySTC.m_dir = _world.m_cube.m_trans - onSphere;
    
    return rayCastCube ( _world, raySTC );
}

// Function 14
void brightnessAdjust( inout vec4 color, in float b) {
    color.rgb += b;
}

// Function 15
vec3 stereographic_polar_diagram(in vec2 p, in vec2 theta) {
    
    mat3 R = rotX(-theta.x)*rotY(-theta.y);

    float rad = length(planar_verts[0]);
    float scl = 8.0*rad / iResolution.y;
    
    p *= scl;

    float d = 1e5;
    
    vec3 Rctr = R * vec3(0, 0, 1);
    vec3 Rp3d = R * vec3(p, 1);
    vec2 Rp = Rp3d.xy / Rp3d.z;
        
    for (int i=0; i<3; ++i) {
        vec3 tp = (tri_verts[i] * planar_proj * R);
        d = min(d, length(p - tp.xy / tp.z) - 2.*scl);
    }
        
    vec3 pos = sphere_from_planar(Rp) * sign(Rp3d.z);
    mat3 M = tile_sphere(pos);
    
    for (int i=0; i<3; ++i) {
        vec3 e =  M * tri_edges[i] * planar_proj * R;
        e /= length(e.xy);
        d = min(d, abs(dot(vec3(p, 1), e)));
    }    

    vec3 pv = M * poly_vertex * planar_proj * R;
    
    vec3 color = vec3(1);

    if (length(Rp) < rad) {
        color = vec3(1, .5, 1);
    }

    float Mdet = dot(M[0], cross(M[1], M[2]));
    
    color *= mix(0.8, 1.0, step(0.0, Mdet));
    
    color = mix(color, vec3(0, 0, 1), smoothstep(scl, 0.0, abs(length(Rp)-rad)-.5*scl));
    color *= smoothstep(0., scl, d-0.25*scl);
    color = mix(color, vec3(0.7, 0, 0), smoothstep(scl, 0., length(p - pv.xy / pv.z)-3.*scl));
    
    vec3 e = vec3(0, 0, 1) * R;
    e /= length(e.xy);
    d = abs(dot(vec3(p, 1), e));    
    color = mix(color, vec3(0.0, 0, 0.5), smoothstep(scl, 0., d-.5*scl));
    
    return color;
    
}

// Function 16
vec4 blur_horizontal_left_column(vec2 uv, int depth)
{
    float h = pow(2., float(depth)) / iResolution.x;    
    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;

    uv1 = fract(vec2(uv.x - 4.0 * h, uv.y) * 2.);
    uv2 = fract(vec2(uv.x - 3.0 * h, uv.y) * 2.);
    uv3 = fract(vec2(uv.x - 2.0 * h, uv.y) * 2.);
    uv4 = fract(vec2(uv.x - 1.0 * h, uv.y) * 2.);
    uv5 = fract(vec2(uv.x + 0.0 * h, uv.y) * 2.);
    uv6 = fract(vec2(uv.x + 1.0 * h, uv.y) * 2.);
    uv7 = fract(vec2(uv.x + 2.0 * h, uv.y) * 2.);
    uv8 = fract(vec2(uv.x + 3.0 * h, uv.y) * 2.);
    uv9 = fract(vec2(uv.x + 4.0 * h, uv.y) * 2.);

    if(uv.y > 0.5)
    {
        uv1 = upper_left(uv1);
        uv2 = upper_left(uv2);
        uv3 = upper_left(uv3);
        uv4 = upper_left(uv4);
        uv5 = upper_left(uv5);
        uv6 = upper_left(uv6);
        uv7 = upper_left(uv7);
        uv8 = upper_left(uv8);
        uv9 = upper_left(uv9);
    }
    else{
        uv1 = lower_left(uv1);
        uv2 = lower_left(uv2);
        uv3 = lower_left(uv3);
        uv4 = lower_left(uv4);
        uv5 = lower_left(uv5);
        uv6 = lower_left(uv6);
        uv7 = lower_left(uv7);
        uv8 = lower_left(uv8);
        uv9 = lower_left(uv9);
    }

    for(int level = 0; level < 8; level++)
    {
        if(level >= depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);

    sum += texture(iChannel3, uv1) * 0.0162162162;
    sum += texture(iChannel3, uv2) * 0.0540540541;
    sum += texture(iChannel3, uv3) * 0.1216216216;
    sum += texture(iChannel3, uv4) * 0.1945945946;
    sum += texture(iChannel3, uv5) * 0.2270270270;
    sum += texture(iChannel3, uv6) * 0.1945945946;
    sum += texture(iChannel3, uv7) * 0.1216216216;
    sum += texture(iChannel3, uv8) * 0.0540540541;
    sum += texture(iChannel3, uv9) * 0.0162162162;
    
    return sum;
}

// Function 17
void SpriteRight(inout vec3 color, vec2 p)
{
    uint v = 0u;
	v = p.y == 31. ? 0u : v;
	v = p.y == 30. ? 0u : v;
	v = p.y == 29. ? 0u : v;
	v = p.y == 28. ? 0u : v;
	v = p.y == 27. ? (p.x < 8. ? 2415919248u : (p.x < 16. ? 629145u : (p.x < 24. ? 2576351241u : 0u))) : v;
	v = p.y == 26. ? (p.x < 8. ? 2566914192u : (p.x < 16. ? 629145u : (p.x < 24. ? 10027017u : 0u))) : v;
	v = p.y == 25. ? (p.x < 8. ? 160432272u : (p.x < 16. ? 0u : (p.x < 24. ? 626697u : 0u))) : v;
	v = p.y == 24. ? (p.x < 8. ? 160432272u : (p.x < 16. ? 0u : (p.x < 24. ? 39321u : 0u))) : v;
	v = p.y == 23. ? (p.x < 8. ? 160432272u : (p.x < 16. ? 0u : (p.x < 24. ? 629145u : 0u))) : v;
	v = p.y == 22. ? (p.x < 8. ? 2566914192u : (p.x < 16. ? 0u : (p.x < 24. ? 10063881u : 0u))) : v;
	v = p.y == 21. ? (p.x < 8. ? 2566914192u : (p.x < 16. ? 629145u : (p.x < 24. ? 142606345u : 17895424u))) : v;
	v = p.y == 20. ? (p.x < 8. ? 144u : (p.x < 16. ? 620185u : (p.x < 24. ? 355467273u : 286265636u))) : v;
	v = p.y == 19. ? (p.x < 8. ? 272u : (p.x < 16. ? 256u : (p.x < 24. ? 286261248u : 17825809u))) : v;
	v = p.y == 18. ? (p.x < 8. ? 272u : (p.x < 16. ? 16u : (p.x < 24. ? 286261248u : 286326784u))) : v;
	v = p.y == 17. ? (p.x < 8. ? 273u : (p.x < 16. ? 286326801u : (p.x < 24. ? 17891328u : 17895424u))) : v;
	v = p.y == 16. ? (p.x < 8. ? 268435729u : (p.x < 16. ? 286330881u : (p.x < 24. ? 17895425u : 1118464u))) : v;
	v = p.y == 15. ? (p.x < 8. ? 285212945u : (p.x < 16. ? 1118465u : (p.x < 24. ? 17895424u : 69905u))) : v;
	v = p.y == 14. ? (p.x < 8. ? 285212945u : (p.x < 16. ? 1118464u : (p.x < 24. ? 286330880u : 4369u))) : v;
	v = p.y == 13. ? (p.x < 8. ? 17826065u : (p.x < 16. ? 69904u : (p.x < 24. ? 286326784u : 17u))) : v;
	v = p.y == 12. ? (p.x < 8. ? 17895696u : (p.x < 16. ? 69904u : (p.x < 24. ? 286330880u : 0u))) : v;
	v = p.y == 11. ? (p.x < 8. ? 1118464u : (p.x < 16. ? 17895696u : (p.x < 24. ? 286330880u : 0u))) : v;
	v = p.y == 10. ? (p.x < 8. ? 1118464u : (p.x < 16. ? 4368u : (p.x < 24. ? 286330880u : 1u))) : v;
	v = p.y == 9. ? (p.x < 8. ? 1118464u : (p.x < 16. ? 272u : (p.x < 24. ? 268505088u : 17u))) : v;
	v = p.y == 8. ? (p.x < 8. ? 69888u : (p.x < 16. ? 272u : (p.x < 24. ? 69632u : 273u))) : v;
	v = p.y == 7. ? (p.x < 8. ? 69632u : (p.x < 16. ? 268435728u : (p.x < 24. ? 69888u : 4368u))) : v;
	v = p.y == 6. ? (p.x < 8. ? 69632u : (p.x < 16. ? 16777488u : (p.x < 24. ? 4352u : 256u))) : v;
	v = p.y == 5. ? (p.x < 8. ? 0u : (p.x < 16. ? 69888u : (p.x < 24. ? 4352u : 0u))) : v;
	v = p.y == 4. ? (p.x < 8. ? 0u : (p.x < 16. ? 0u : (p.x < 24. ? 256u : 0u))) : v;
	v = p.y == 3. ? 0u : v;
	v = p.y == 2. ? 0u : v;
	v = p.y == 1. ? 0u : v;
	v = p.y == 0. ? 0u : v;
    v = p.x >= 0. && p.x < 32. ? v : 0u;

    float i = float((v >> uint(4. * p.x)) & 15u);
    color = i == 1. ? vec3(1, 0, 0.73) : color;
    color = i == 2. ? vec3(1, 0.0039, 0.73) : color;
    color = i == 3. ? vec3(1, 0.063, 0.75) : color;
    color = i == 4. ? vec3(1, 0.12, 0.76) : color;
    color = i == 5. ? vec3(1, 0.41, 0.84) : color;
    color = i == 6. ? vec3(1, 0.64, 0.9) : color;
    color = i == 7. ? vec3(1, 0.72, 0.93) : color;
    color = i == 8. ? vec3(1, 0.81, 0.95) : color;
    color = i == 9. ? vec3(1) : color;
}

// Function 18
vec4 blur_vertical_left_column(vec2 uv, int depth)
{
    float v = pow(2., float(depth)) / iResolution.y;

    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;
    
    uv1 = fract(vec2(uv.x, uv.y - 4.0*v) * 2.);
    uv2 = fract(vec2(uv.x, uv.y - 3.0*v) * 2.);
    uv3 = fract(vec2(uv.x, uv.y - 2.0*v) * 2.);
    uv4 = fract(vec2(uv.x, uv.y - 1.0*v) * 2.);
    uv5 = fract(vec2(uv.x, uv.y + 0.0*v) * 2.);
    uv6 = fract(vec2(uv.x, uv.y + 1.0*v) * 2.);
    uv7 = fract(vec2(uv.x, uv.y + 2.0*v) * 2.);
    uv8 = fract(vec2(uv.x, uv.y + 3.0*v) * 2.);
    uv9 = fract(vec2(uv.x, uv.y + 4.0*v) * 2.);

    if(uv.x < 0.5)
    {
        if(uv.y > 0.5)
        {
            uv1 = upper_left(uv1);
            uv2 = upper_left(uv2);
            uv3 = upper_left(uv3);
            uv4 = upper_left(uv4);
            uv5 = upper_left(uv5);
            uv6 = upper_left(uv6);
            uv7 = upper_left(uv7);
            uv8 = upper_left(uv8);
            uv9 = upper_left(uv9);
        }
        else
        {
            uv1 = lower_left(uv1);
            uv2 = lower_left(uv2);
            uv3 = lower_left(uv3);
            uv4 = lower_left(uv4);
            uv5 = lower_left(uv5);
            uv6 = lower_left(uv6);
            uv7 = lower_left(uv7);
            uv8 = lower_left(uv8);
            uv9 = lower_left(uv9);
        }
    }
    else
    {
        vec2 uv_s = upper_right(uv*2.)*2.;
        uv1 = clamp(vec2(uv_s.x, uv_s.y - 4.0*v), 0., 1.);
        uv2 = clamp(vec2(uv_s.x, uv_s.y - 3.0*v), 0., 1.);
        uv3 = clamp(vec2(uv_s.x, uv_s.y - 2.0*v), 0., 1.);
        uv4 = clamp(vec2(uv_s.x, uv_s.y - 1.0*v), 0., 1.);
        uv5 = clamp(vec2(uv_s.x, uv_s.y + 0.0*v), 0., 1.);
        uv6 = clamp(vec2(uv_s.x, uv_s.y + 1.0*v), 0., 1.);
        uv7 = clamp(vec2(uv_s.x, uv_s.y + 2.0*v), 0., 1.);
        uv8 = clamp(vec2(uv_s.x, uv_s.y + 3.0*v), 0., 1.);
        uv9 = clamp(vec2(uv_s.x, uv_s.y + 4.0*v), 0., 1.);
        depth--;
        uv1 = upper_right(uv1);
        uv2 = upper_right(uv2);
        uv3 = upper_right(uv3);
        uv4 = upper_right(uv4);
        uv5 = upper_right(uv5);
        uv6 = upper_right(uv6);
        uv7 = upper_right(uv7);
        uv8 = upper_right(uv8);
        uv9 = upper_right(uv9);
    }
    for(int level = 0; level < 8; level++)
    {
        if(level > depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);
    if(uv.x > 0.5 && uv.y > 0.5)
    {
        //return vec4(0);
		sum += texture(iChannel3, uv1) * 0.05;
        sum += texture(iChannel3, uv2) * 0.09;
        sum += texture(iChannel3, uv3) * 0.12;
        sum += texture(iChannel3, uv4) * 0.15;
        sum += texture(iChannel3, uv5) * 0.16;
        sum += texture(iChannel3, uv6) * 0.15;
        sum += texture(iChannel3, uv7) * 0.12;
        sum += texture(iChannel3, uv8) * 0.09;
        sum += texture(iChannel3, uv9) * 0.05;
    }
    else
    {
        sum += texture(iChannel2, uv1) * 0.05;
        sum += texture(iChannel2, uv2) * 0.09;
        sum += texture(iChannel2, uv3) * 0.12;
        sum += texture(iChannel2, uv4) * 0.15;
        sum += texture(iChannel2, uv5) * 0.16;
        sum += texture(iChannel2, uv6) * 0.15;
        sum += texture(iChannel2, uv7) * 0.12;
        sum += texture(iChannel2, uv8) * 0.09;
        sum += texture(iChannel2, uv9) * 0.05;
    }
    return sum/0.98; // normalize
}

// Function 19
vec4 GetStereogramDepth(vec2 vPixel, vec4 vPrev)
{
	// Adjust pixel co-ordinates to be in centre of strip
	return GetDepth(vPixel - vec2(fPixelRepeat * 0.5, 0.0), vPrev);
}

// Function 20
float right(float p){
  return c4(-pi*.5);
}

// Function 21
bool RectShiftLeftContains(Rect rect, vec2 pos, float eyeWidth)
{
    float shift = (eyeWidth - rect.z) / iResolution.x;
    return (rect.x - shift <= pos.x && pos.x < rect.x + rect.w &&
            rect.y <= pos.y && pos.y < rect.y + rect.h);
}

// Function 22
vec2 lower_left(vec2 uv)
{
    uv = fract(uv);
    return fract(uv * 0.5);
}

// Function 23
vec2 lower_right(vec2 uv)
{
    uv = fract(uv);
    return fract((uv - vec2(1, 0.)) * 0.5);
}

// Function 24
vec3 Stereogram(vec2 vPixel)
{
	vec2 vInitialPixel = vPixel;
	#ifdef INTEGER_OFFSET
	vInitialPixel = floor(vInitialPixel + 0.5);
	#endif
	vec2 vIntPixel = vInitialPixel;
	
	// This is an arbitrary number, enough to make sure we will reach the edge of the screen
	for(int i=0; i<64; i++)
	{
		// Step left fPixelRepeat minus depth...
		vec4 vDepth = GetStereogramDepth(vIntPixel, vec4(0.0));
		float fOffset = -fPixelRepeat;

		#ifndef INVERT_DEPTH
		fOffset -= vDepth.w * fDepthScale;
		#else
		fOffset += vDepth.w * fDepthScale;
		#endif		
		
		vIntPixel.x = vIntPixel.x + fOffset;		
		#ifdef INTEGER_OFFSET
		vIntPixel.x = floor(vIntPixel.x + 0.5);
		#endif
		
		// ...until we fall of the screen
		if(vIntPixel.x < 0.0)
		{
			break;
		}
	}

	vIntPixel.x = mod(vIntPixel.x, fPixelRepeat);
	
	vec2 vUV = (vIntPixel + 0.5) / fPixelRepeat;

	vec3 vResult;
	
	#ifdef RANDOM_BACKDROP_OFFSET	
		vUV += Random2(iTime);
	#endif // RANDOM_BACKDROP_OFFSET
	
	#ifdef RANDOM_BACKDROP_OFFSET_PER_LINE
		vUV += Random2(iTime + vUV.y * iResolution.y);
	#endif
	
	const float fMipLod = -32.0;

    if ( gUVScale != 1.0 )
    {    	
        vUV = fract(vUV) * gUVScale;
    }
    
	#ifdef USE_NOISE_IMAGE
		vResult = texture(iChannel1, fract(vec2(vUV)), fMipLod).rrr;
	#else
		vResult = texture(iChannel0, fract(vec2(vUV)), fMipLod).rgb;	
	#endif // USE_NOISE_IMAGE
	
	#ifdef ADD_COLOUR
	vec4 vColour = vec4(0.0, 0.8, 1.0, 1.0);	
	vColour = GetStereogramDepth(vInitialPixel, vColour);
	
	#ifdef DOUBLE_COLOUR
	vColour = GetStereogramDepth(vInitialPixel + vec2(fPixelRepeat, 0.0), vColour);
	#endif // DOUBLE_COLOUR
		
	vResult = mix(vResult, vec3(1.0), fNoiseDesaturation); // desaturate noise
	vColour.rgb = mix(vColour.rgb, vec3(1.0), fColourDesaturation); // desaturate colour
	vResult = vResult * vColour.rgb;
	#endif
	
	return vResult;
}

// Function 25
vec4 fpsTopLeftCorner(vec4 o,vec2 p
){p.y-=.94
 ;p.x+=.01
 ;p *= 16.2
 ;float w=0.
 ;float left=0.
 ;float fpsFrame=iTimeDelta
 ;if(abs(fpsFrame)>0.
 ){fpsFrame=1./fpsFrame
  ;left=getWholeDigits(abs(fpsFrame))
  ;o+=floatShow(p,fpsFrame,left,2.,w).x
 ;}else fpsFrame=0.000001;left=2.
 ;o*=.5*(smoothstep(0.,1.,o.w))
 ;o=max(o,vec4(0))
 ;o=sqrt(o)//cheap cubic bloom    
 ;return o;}

// Function 26
mat4 brightnessMatrix( float b ) {
    return mat4( 
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        b, b, b, 1 );
}

// Function 27
float brightness(vec3 l) {
    return length(l.xy)+l.z;
}

// Function 28
int binarySearchLeftMost(int part, int T, vec2 res, vec2 fragCoord) {
    mPartitionData pd = getPartitionData(sortedBuffer, fragCoord, res);
    int n = pd.particlesPerPartition;
    int maxPartition = getMaxPartition(pd);
    int L = maxPartition * n;
    int R = L + n;

    int i = 0;
    for (i = 0; i < maxBin && L < R; i++) {
        int m = (L + R) / 2;
        int Am = getM(part, m, res).Am;
        L = Am < T ? m + 1 : L;
        R = Am >= T ? m : R;
    }
    int ret = i < maxBin - 1 ? L : -1;
    return ret;
}

// Function 29
vec4 blur_vertical_upper_left(sampler2D channel, vec2 uv)
{
    float v = 1. / iResolution.y;
    vec4 sum = vec4(0.0);
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 4.0*v)) ) * 0.0162162162;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 3.0*v)) ) * 0.0540540541;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 2.0*v)) ) * 0.1216216216;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 1.0*v)) ) * 0.1945945946;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 0.0*v)) ) * 0.2270270270;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 1.0*v)) ) * 0.1945945946;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 2.0*v)) ) * 0.1216216216;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 3.0*v)) ) * 0.0540540541;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 4.0*v)) ) * 0.0162162162;
    return sum;
}

// Function 30
float parryUpRight(float i, float m, inout StickmanData data)
{   
    float p = 1.0 - pow(1.0 - pingPong(i), 4.0);

    mat3 alignRot;
    vec3 restTarget = vec3(0.0, 1.0, 0.0)*data.invSaberRot;    
    vec3 armTarget = vec3(-1.0, 0.15, -0.1);
    rotationAlign(armTarget, restTarget, p, alignRot);
    data.invSaberRot = data.invSaberRot*alignRot;
        
    data.saberPos = mix(data.saberPos, vec3(0.6, 0.42, -1.7), p);
                
    //squint eyes while you parry, just because ....
    data.eyeLDeform.y *= 1.0 - p*0.4;
    data.eyeRDeform.y *= 1.0 - p*0.6;
    
    //stance
    data.footRPos = mix(data.footRPos, ll*vec3(0.8, 0.1, data.footRPos.z + 0.5), p);
    data.footRPos.y += pingPong(p)*ll*0.1;
    return 0.0;
}

// Function 31
vec4 blur_vertical_left_column(vec2 uv, int depth)
{
    float v = pow(2., float(depth)) / iResolution.y;

    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;

    uv1 = fract(vec2(uv.x, uv.y - 4.0*v) * 2.);
    uv2 = fract(vec2(uv.x, uv.y - 3.0*v) * 2.);
    uv3 = fract(vec2(uv.x, uv.y - 2.0*v) * 2.);
    uv4 = fract(vec2(uv.x, uv.y - 1.0*v) * 2.);
    uv5 = fract(vec2(uv.x, uv.y + 0.0*v) * 2.);
    uv6 = fract(vec2(uv.x, uv.y + 1.0*v) * 2.);
    uv7 = fract(vec2(uv.x, uv.y + 2.0*v) * 2.);
    uv8 = fract(vec2(uv.x, uv.y + 3.0*v) * 2.);
    uv9 = fract(vec2(uv.x, uv.y + 4.0*v) * 2.);

    if(uv.x < 0.5)
    {
        if(uv.y > 0.5)
        {
            uv1 = upper_left(uv1);
            uv2 = upper_left(uv2);
            uv3 = upper_left(uv3);
            uv4 = upper_left(uv4);
            uv5 = upper_left(uv5);
            uv6 = upper_left(uv6);
            uv7 = upper_left(uv7);
            uv8 = upper_left(uv8);
            uv9 = upper_left(uv9);
        }
        else
        {
            uv1 = lower_left(uv1);
            uv2 = lower_left(uv2);
            uv3 = lower_left(uv3);
            uv4 = lower_left(uv4);
            uv5 = lower_left(uv5);
            uv6 = lower_left(uv6);
            uv7 = lower_left(uv7);
            uv8 = lower_left(uv8);
            uv9 = lower_left(uv9);
        }
    }
    else
    {
        vec2 uv_s = upper_right(uv*2.)*2.;
        uv1 = clamp(vec2(uv_s.x, uv_s.y - 4.0*v), 0., 1.);
        uv2 = clamp(vec2(uv_s.x, uv_s.y - 3.0*v), 0., 1.);
        uv3 = clamp(vec2(uv_s.x, uv_s.y - 2.0*v), 0., 1.);
        uv4 = clamp(vec2(uv_s.x, uv_s.y - 1.0*v), 0., 1.);
        uv5 = clamp(vec2(uv_s.x, uv_s.y + 0.0*v), 0., 1.);
        uv6 = clamp(vec2(uv_s.x, uv_s.y + 1.0*v), 0., 1.);
        uv7 = clamp(vec2(uv_s.x, uv_s.y + 2.0*v), 0., 1.);
        uv8 = clamp(vec2(uv_s.x, uv_s.y + 3.0*v), 0., 1.);
        uv9 = clamp(vec2(uv_s.x, uv_s.y + 4.0*v), 0., 1.);
        depth--;
        uv1 = upper_right(uv1);
        uv2 = upper_right(uv2);
        uv3 = upper_right(uv3);
        uv4 = upper_right(uv4);
        uv5 = upper_right(uv5);
        uv6 = upper_right(uv6);
        uv7 = upper_right(uv7);
        uv8 = upper_right(uv8);
        uv9 = upper_right(uv9);
    }
    for(int level = 0; level < 8; level++)
    {
        if(level > depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);
    if(uv.x > 0.5 && uv.y > 0.5)
    {
        //return vec4(0);
        sum += texture(iChannel3, uv1) * 0.05;
        sum += texture(iChannel3, uv2) * 0.09;
        sum += texture(iChannel3, uv3) * 0.12;
        sum += texture(iChannel3, uv4) * 0.15;
        sum += texture(iChannel3, uv5) * 0.16;
        sum += texture(iChannel3, uv6) * 0.15;
        sum += texture(iChannel3, uv7) * 0.12;
        sum += texture(iChannel3, uv8) * 0.09;
        sum += texture(iChannel3, uv9) * 0.05;
    }
    else
    {
        sum += texture(iChannel2, uv1) * 0.05;
        sum += texture(iChannel2, uv2) * 0.09;
        sum += texture(iChannel2, uv3) * 0.12;
        sum += texture(iChannel2, uv4) * 0.15;
        sum += texture(iChannel2, uv5) * 0.16;
        sum += texture(iChannel2, uv6) * 0.15;
        sum += texture(iChannel2, uv7) * 0.12;
        sum += texture(iChannel2, uv8) * 0.09;
        sum += texture(iChannel2, uv9) * 0.05;
    }
    return sum/0.98; // normalize
}

// Function 32
mat4 brightnessMatrix( float brightness )
{
    return mat4( 1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0,
                 brightness, brightness, brightness, 1 );
}

// Function 33
vec4 blur_vertical_upper_left(sampler2D channel, vec2 uv)
{
    float v = 1. / iResolution.y;
    vec4 sum = vec4(0.0);
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 4.0*v)) ) * 0.05;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 3.0*v)) ) * 0.09;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 2.0*v)) ) * 0.12;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y - 1.0*v)) ) * 0.15;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 0.0*v)) ) * 0.16;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 1.0*v)) ) * 0.15;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 2.0*v)) ) * 0.12;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 3.0*v)) ) * 0.09;
    sum += texture(channel, upper_left(vec2(uv.x, uv.y + 4.0*v)) ) * 0.05;
    return sum/0.98; // normalize
}

// Function 34
bool RectShiftRightContains(Rect rect, vec2 pos, float eyeWidth)
{
    float shift = (eyeWidth - rect.z) / iResolution.x;
    return (rect.x + shift * 0.5f <= pos.x && pos.x < rect.x + rect.w - shift * 1.5 &&
            rect.y <= pos.y && pos.y < rect.y + rect.h);
}

// Function 35
vec2 intersectionStereogram(vec2 uv, float time, float eyesDist, float faceScreenDist,float rightToLeft)
{
    float xP = cancellation(uv,time,rightToLeft * eyesDist,faceScreenDist);
    float fxP = shapeDistance(vec2(xP,uv.y),time);
    float newX = - rightToLeft * eyesDist + faceScreenDist * ( xP + rightToLeft * eyesDist ) / ( faceScreenDist + fxP );

    return vec2(newX,uv.y);
}

// Function 36
vec2 stereoOsc(int instrument, float time, int waveform, float ampl, float freq, float p1, float p2, float p3)
{
   return vec2(osc(instrument, time, waveform, ampl, freq, 0., p1, p2, p3), osc(instrument, time, waveform, ampl, freq, stereoPhase, p1, p2, p3));
}

// Function 37
float left(float p){
  return c4( pi*.5);
}

// Function 38
void UILayout_StackRight( inout UILayout uiLayout )
{
    UILayout_SetX( uiLayout, uiLayout.vControlMax.x + UIStyle_ControlSpacing().x );
}

// Function 39
vec2 upper_left(vec2 uv)
{
    uv = fract(uv);
    return fract((uv - vec2(0., 1)) * 0.5);
}

// Function 40
vec3 modBrightnessContrast(vec3 val, float brightness, float contrast)
        {
            return (val - 0.5) * contrast + 0.5 + brightness;
        }

// Function 41
vec3 MazeRight(vec2 p, vec2 scale)
{
    vec3 res = BACKGROUND;

    vec2 gv = grid(p, scale); // The grid guide
    
    if (gv.x >= 0. && gv.y >= 0. &&
        gv.x <= 15. && gv.y <= 15.) {
        
        #if (DISPLAY_GRID == 1)
        	res = vec3(mod(gv.x + gv.y, 2.) * .05 + BACKGROUND);
        #endif
        
        // Indexing is upside down.
        int y = int(scale.y - gv.y - 5.);

    	float m = 0.;
		Q(0, B,B,B,B,B,B,B,B)
		Q(1, B,_,_,_,_,_,_,_)
		Q(2, B,_,B,_,B,_,_,_)
		Q(3, B,_,B,D,B,_,D,D)
		Q(4, B,_,_,B,_,_,B,_)
		Q(5, B,B,B,B,_,B,B,_)
		Q(6, B,_,_,_,_,B,_,_)
		Q(7, B,_,B,B,B,B,_,_)
		Q(8, B,_,B,_,_,_,_,_)
		Q(9, B,_,B,B,B,B,D,_)
		Q(10,B,_,_,_,_,_,D,_)
        Q(11,B,B,B,D,D,_,D,_)
		Q(12,B,_,_,_,_,_,D,_)
		Q(13,B,B,B,_,B,B,B,B)
		Q(14,_,_,_,_,B,_,_,_) // some of pants and jacket
		Q(15,B,B,B,B,B,B,B,B)
        
    	float ldx = 15. - gv.x; // Calculate the left  bit index
        float rdx = gv.x;       // Calculate the right bit index
        float bit = 0.;
        
        //if (gv.x >= 8.)	bit = mod(m / pow(4., ldx), 4.); // Decode
        //else            bit = mod(m / pow(4., rdx), 4.); // Mirror
        bit = mod(m / pow(4., ldx), 4.); // Decode
    	bit = floor(bit);                                // Sharpen    
    	
        // Colorize
             if (bit > 2.) res = vec3(.6471,.6471,.6471);
        else if (bit > 1.) res = vec3(1.,0.8941176471,0.6745098039);
        else if (bit > 0.) res = vec3(0.8549019608,0.4980392157, 0.1568627451);
    }
    
    return res;
}

// Function 42
vec4 inverseStereographic(vec3 p, out float k) {
    k = 2.0/(1.0+dot(p,p));
    return vec4(k*p,k-1.0);
}

// Function 43
vec2 upper_right(vec2 uv)
{
//    uv = fract(uv);
    return fract((uv - 1.) * 0.5);
}

// Function 44
vec2 left(vec2 uv){
    if(uv.x == 0.0){
    	return vec2(SIZE-1.0, uv.y);
    }
	return uv+vec2(-1.0, 0.0);
}

// Function 45
vec2 upper_left(vec2 uv)
{
    return fract((uv - vec2(0., 1)) * 0.5);
}

// Function 46
vec3 getStereoDir(vec2 fragCoord)
{
	vec2 p = fragCoord.xy / iResolution.xy;
    float t = 3.+iTime*.08, ct = cos(t), st = sin(t);
	float m = .5;

    p = (p * 2. * m - m)*3.1;
    p.x *= iResolution.x/iResolution.y;
    p *= mat2(ct,st,-st,ct);

	return normalize(vec3(2.*p.x,dot(p,p)-1.,2.*p.y));
}

// Function 47
vec3 calcCubeCoordsInStereographicProjection(in vec2 screenCoord, in vec2 centralPoint, in vec2 FoVScale) {
	return sphericalToCubemap( calcSphericalCoordsInStereographicProjection(screenCoord, centralPoint, FoVScale) );
}

// Function 48
float parryDownLeft(float i, float m, inout StickmanData data)
{   
    float p = 1.0 - pow(1.0 - pingPong(i), 4.0);
    
    mat3 alignRot;
    vec3 restTarget = vec3(0.0, 1.0, 0.0)*data.invSaberRot;    
    vec3 saberTarget = vec3(-0.2, 1.0, -0.5);
    rotationAlign(saberTarget, restTarget, p, alignRot);
    data.invSaberRot = data.invSaberRot*alignRot;

    data.saberPos = mix(data.saberPos, vec3(-0.9, -0.5, -1.4), p);
                
	//squint eyes while you parry, just because ....
    data.eyeLDeform.y *= 1.0 - p*0.4;
    data.eyeRDeform.y *= 1.0 - p*0.6;
    
    //stance
    data.footRPos = mix(data.footRPos, ll*vec3(0.7, 0.1, data.footRPos.z + 0.5), p);
    data.footRPos.y += pingPong(p)*ll*0.1;
    return 0.0;
}

// Function 49
vec2 stereographPack(vec3 n)
{
    float scale = 1.7777;
    vec2 enc = n.xy / ( n.z + 1.0 );
    enc /= scale;
    enc = enc * 0.5 + 0.5;
    return enc;
}

// Function 50
vec2 stereoProject(vec3 p){
	return vec2(p.x / (1. - p.y), p.z / (1. - p.y));
}

// Function 51
vec3 brightnessGradient(float t) {
	return vec3(t * t);
}

// Function 52
float brightness (vec3 color) 
{ 
	return ( 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b ); 
}

// Function 53
vec4 blur_vertical_lower_left(sampler2D channel, vec2 uv)
{
    float v = 1. / iResolution.y;
    vec4 sum = vec4(0.0);
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 4.0*v)) ) * 0.0162162162;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 3.0*v)) ) * 0.0540540541;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 2.0*v)) ) * 0.1216216216;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 1.0*v)) ) * 0.1945945946;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 0.0*v)) ) * 0.2270270270;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 1.0*v)) ) * 0.1945945946;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 2.0*v)) ) * 0.1216216216;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 3.0*v)) ) * 0.0540540541;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 4.0*v)) ) * 0.0162162162;
    return sum;
}

// Function 54
vec2 lower_left(vec2 uv)
{
    return fract(uv * 0.5);
}

// Function 55
vec4 blur_vertical_left_column(vec2 uv, int depth)
{
    float v = pow(2., float(depth)) / iResolution.y;

    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;

    uv1 = fract(vec2(uv.x, uv.y - 4.0*v) * 2.);
    uv2 = fract(vec2(uv.x, uv.y - 3.0*v) * 2.);
    uv3 = fract(vec2(uv.x, uv.y - 2.0*v) * 2.);
    uv4 = fract(vec2(uv.x, uv.y - 1.0*v) * 2.);
    uv5 = fract(vec2(uv.x, uv.y + 0.0*v) * 2.);
    uv6 = fract(vec2(uv.x, uv.y + 1.0*v) * 2.);
    uv7 = fract(vec2(uv.x, uv.y + 2.0*v) * 2.);
    uv8 = fract(vec2(uv.x, uv.y + 3.0*v) * 2.);
    uv9 = fract(vec2(uv.x, uv.y + 4.0*v) * 2.);

    if(uv.x < 0.5)
    {
        if(uv.y > 0.5)
        {
            uv1 = upper_left(uv1);
            uv2 = upper_left(uv2);
            uv3 = upper_left(uv3);
            uv4 = upper_left(uv4);
            uv5 = upper_left(uv5);
            uv6 = upper_left(uv6);
            uv7 = upper_left(uv7);
            uv8 = upper_left(uv8);
            uv9 = upper_left(uv9);
        }
        else
        {
            uv1 = lower_left(uv1);
            uv2 = lower_left(uv2);
            uv3 = lower_left(uv3);
            uv4 = lower_left(uv4);
            uv5 = lower_left(uv5);
            uv6 = lower_left(uv6);
            uv7 = lower_left(uv7);
            uv8 = lower_left(uv8);
            uv9 = lower_left(uv9);
        }
    }
    else
    {
        vec2 uv_s = upper_right(uv*2.)*2.;
        uv1 = clamp(vec2(uv_s.x, uv_s.y - 4.0*v), 0., 1.);
        uv2 = clamp(vec2(uv_s.x, uv_s.y - 3.0*v), 0., 1.);
        uv3 = clamp(vec2(uv_s.x, uv_s.y - 2.0*v), 0., 1.);
        uv4 = clamp(vec2(uv_s.x, uv_s.y - 1.0*v), 0., 1.);
        uv5 = clamp(vec2(uv_s.x, uv_s.y + 0.0*v), 0., 1.);
        uv6 = clamp(vec2(uv_s.x, uv_s.y + 1.0*v), 0., 1.);
        uv7 = clamp(vec2(uv_s.x, uv_s.y + 2.0*v), 0., 1.);
        uv8 = clamp(vec2(uv_s.x, uv_s.y + 3.0*v), 0., 1.);
        uv9 = clamp(vec2(uv_s.x, uv_s.y + 4.0*v), 0., 1.);
        depth--;
        uv1 = upper_right(uv1);
        uv2 = upper_right(uv2);
        uv3 = upper_right(uv3);
        uv4 = upper_right(uv4);
        uv5 = upper_right(uv5);
        uv6 = upper_right(uv6);
        uv7 = upper_right(uv7);
        uv8 = upper_right(uv8);
        uv9 = upper_right(uv9);
    }
    for(int level = 0; level < 8; level++)
    {
        if(level > depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);
    if(uv.x > 0.5 && uv.y > 0.5)
    {
        //return vec4(0);
        sum += texture(iChannel3, uv1) * 0.0162162162;
        sum += texture(iChannel3, uv2) * 0.0540540541;
        sum += texture(iChannel3, uv3) * 0.1216216216;
        sum += texture(iChannel3, uv4) * 0.1945945946;
        sum += texture(iChannel3, uv5) * 0.2270270270;
        sum += texture(iChannel3, uv6) * 0.1945945946;
        sum += texture(iChannel3, uv7) * 0.1216216216;
        sum += texture(iChannel3, uv8) * 0.0540540541;
        sum += texture(iChannel3, uv9) * 0.0162162162;
    }
    else
    {
        sum += texture(iChannel2, uv1) * 0.0162162162;
        sum += texture(iChannel2, uv2) * 0.0540540541;
        sum += texture(iChannel2, uv3) * 0.1216216216;
        sum += texture(iChannel2, uv4) * 0.1945945946;
        sum += texture(iChannel2, uv5) * 0.2270270270;
        sum += texture(iChannel2, uv6) * 0.1945945946;
        sum += texture(iChannel2, uv7) * 0.1216216216;
        sum += texture(iChannel2, uv8) * 0.0540540541;
        sum += texture(iChannel2, uv9) * 0.0162162162;
    }
    return sum; // normalize
}

// Function 56
vec3 stereographUnpack(vec2 n) {
    float scale = 1.7777;
    vec3 nn = vec3(2.0 * scale * n, 0.0) + vec3(-scale,-scale,1.0);
    float g = 2.0 / dot(nn.xyz,nn.xyz);
    vec3 normal;
    normal.xy = g*nn.xy;
    normal.z = g-1.0;
    return normal;
}

// Function 57
void SpriteLeft(inout vec3 color, vec2 p)
{
    uint v = 0u;
	v = p.y == 31. ? 0u : v;
	v = p.y == 30. ? 0u : v;
	v = p.y == 29. ? 0u : v;
	v = p.y == 28. ? 0u : v;
	v = p.y == 27. ? (p.x < 8. ? 0u : (p.x < 16. ? 2004317952u : (p.x < 24. ? 2003828855u : 489335u))) : v;
	v = p.y == 26. ? (p.x < 8. ? 0u : (p.x < 16. ? 2004317952u : (p.x < 24. ? 2003830647u : 7829367u))) : v;
	v = p.y == 25. ? (p.x < 8. ? 0u : (p.x < 16. ? 30464u : (p.x < 24. ? 124782448u : 7798784u))) : v;
	v = p.y == 24. ? (p.x < 8. ? 0u : (p.x < 16. ? 2004317952u : (p.x < 24. ? 2003830647u : 7829367u))) : v;
	v = p.y == 23. ? (p.x < 8. ? 0u : (p.x < 16. ? 489216u : (p.x < 24. ? 2003830647u : 7829367u))) : v;
	v = p.y == 22. ? (p.x < 8. ? 286326784u : (p.x < 16. ? 139537u : (p.x < 24. ? 124782448u : 7798784u))) : v;
	v = p.y == 21. ? (p.x < 8. ? 286330880u : (p.x < 16. ? 1981878545u : (p.x < 24. ? 122685303u : 34672896u))) : v;
	v = p.y == 20. ? (p.x < 8. ? 286330880u : (p.x < 16. ? 1628508433u : (p.x < 24. ? 319815799u : 18088209u))) : v;
	v = p.y == 19. ? (p.x < 8. ? 286326784u : (p.x < 16. ? 286261248u : (p.x < 24. ? 286326784u : 286326801u))) : v;
	v = p.y == 18. ? (p.x < 8. ? 286261248u : (p.x < 16. ? 268435456u : (p.x < 24. ? 286326784u : 286330880u))) : v;
	v = p.y == 17. ? (p.x < 8. ? 286326784u : (p.x < 16. ? 268435456u : (p.x < 24. ? 17895425u : 286331136u))) : v;
	v = p.y == 16. ? (p.x < 8. ? 286326784u : (p.x < 16. ? 268435456u : (p.x < 24. ? 17895681u : 286331152u))) : v;
	v = p.y == 15. ? (p.x < 8. ? 286326784u : (p.x < 16. ? 285212672u : (p.x < 24. ? 286331137u : 285282577u))) : v;
	v = p.y == 14. ? (p.x < 8. ? 17891328u : (p.x < 16. ? 286261248u : (p.x < 24. ? 286331137u : 286265617u))) : v;
	v = p.y == 13. ? (p.x < 8. ? 17891328u : (p.x < 16. ? 286326784u : (p.x < 24. ? 286330881u : 286261265u))) : v;
	v = p.y == 12. ? (p.x < 8. ? 17891328u : (p.x < 16. ? 286330880u : (p.x < 24. ? 286330880u : 286261248u))) : v;
	v = p.y == 11. ? (p.x < 8. ? 17891328u : (p.x < 16. ? 17895696u : (p.x < 24. ? 17895680u : 285212672u))) : v;
	v = p.y == 10. ? (p.x < 8. ? 286326784u : (p.x < 16. ? 1118481u : (p.x < 24. ? 286331136u : 286261249u))) : v;
	v = p.y == 9. ? (p.x < 8. ? 286326784u : (p.x < 16. ? 4369u : (p.x < 24. ? 285282560u : 286261265u))) : v;
	v = p.y == 8. ? (p.x < 8. ? 286331136u : (p.x < 16. ? 17u : (p.x < 24. ? 268505344u : 286261521u))) : v;
	v = p.y == 7. ? (p.x < 8. ? 286331153u : (p.x < 16. ? 0u : (p.x < 24. ? 69888u : 17830161u))) : v;
	v = p.y == 6. ? (p.x < 8. ? 17895697u : (p.x < 16. ? 0u : (p.x < 24. ? 4352u : 17826064u))) : v;
	v = p.y == 5. ? (p.x < 8. ? 17895696u : (p.x < 16. ? 0u : (p.x < 24. ? 4368u : 17825792u))) : v;
	v = p.y == 4. ? (p.x < 8. ? 17891328u : (p.x < 16. ? 0u : (p.x < 24. ? 4352u : 0u))) : v;
	v = p.y == 3. ? 0u : v;
	v = p.y == 2. ? 0u : v;
	v = p.y == 1. ? 0u : v;
	v = p.y == 0. ? 0u : v;
    v = p.x >= 0. && p.x < 32. ? v : 0u;

    float i = float((v >> uint(4. * p.x)) & 15u);
    color = i == 1. ? vec3(1, 0, 0.8) : color;
    color = i == 2. ? vec3(1, 0.2, 0.8) : color;
    color = i == 3. ? vec3(1, 0.4, 0.8) : color;
    color = i == 4. ? vec3(1, 0.4, 1) : color;
    color = i == 5. ? vec3(1, 0.6, 1) : color;
    color = i == 6. ? vec3(1, 0.8, 1) : color;
    color = i == 7. ? vec3(1) : color;
}

// Function 58
vec4 blur_vertical_lower_left(sampler2D channel, vec2 uv)
{
    float v = 1. / iResolution.y;
    vec4 sum = vec4(0.0);
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 4.0*v)) ) * 0.05;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 3.0*v)) ) * 0.09;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 2.0*v)) ) * 0.12;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y - 1.0*v)) ) * 0.15;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 0.0*v)) ) * 0.16;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 1.0*v)) ) * 0.15;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 2.0*v)) ) * 0.12;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 3.0*v)) ) * 0.09;
    sum += texture(channel, lower_left(vec2(uv.x, uv.y + 4.0*v)) ) * 0.05;
    return sum/0.98; // normalize
}

// Function 59
vec2 stereographicSphereToPlane(vec3 spherePos) {
    return vec2(
        spherePos.x / (1. - spherePos.z),
        spherePos.y / (1. - spherePos.z)
    );
}

// Function 60
vec4 brightness(vec4 colors, float brightness)
{
  colors.rgb /= colors.a;
  colors.rgb += brightness;
  colors.rgb *= colors.a;
  return colors;
}

// Function 61
vec3 stereographicPlaneToSphere(vec2 planePos) {
    float x = planePos.x;
    float y = planePos.y;
    float x2 = x*x;
    float y2 = y*y;
    return vec3(
        2.*x / (1. + x2 + y2),
        2.*y / (1. + x2 + y2),
        (-1. + x2 + y2) / (1. + x2 + y2)
    );
}

// Function 62
float bracketRight(vec2 uv){
    uv.x-=size.x*1.5;
    float p = 1.3;
    uv.y = abs(uv.y);
    float a = atan(uv.x, uv.y);
    float x = abs(length(uv)-size.x*2.);
    uv.y = -uv.y;
    x = mix(x, length(uv+vec2(cos(p), sin(p))*size.x*2.), step(-.3, a));
    return x;
}

// Function 63
int binarySearchRightMost(int part, int T, vec2 res, vec2 fragCoord) {
    mPartitionData pd = getPartitionData(sortedBuffer, fragCoord, res);
    int n = pd.particlesPerPartition;
    int maxPartition = getMaxPartition(pd);
    int L = maxPartition * n;
    int R = L + n;

    int i = 0;
    for (i = 0; i < maxBin && L < R; i++) {
        int m = (L + R) / 2;
        int Am = getM(part, m, res).Am;
        L = Am <= T ? m + 1 : L;
        R = Am > T ? m : R;
    }
    int ret = i < maxBin - 1 ? L - 1 : -1;
    return ret;
}

// Function 64
vec4 blur_vertical_left_column(vec2 uv, int depth)
{
    float v = pow(2., float(depth)) / iResolution.y;

    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;

    uv1 = fract(vec2(uv.x, uv.y - 4.0*v) * 2.);
    uv2 = fract(vec2(uv.x, uv.y - 3.0*v) * 2.);
    uv3 = fract(vec2(uv.x, uv.y - 2.0*v) * 2.);
    uv4 = fract(vec2(uv.x, uv.y - 1.0*v) * 2.);
    uv5 = fract(vec2(uv.x, uv.y + 0.0*v) * 2.);
    uv6 = fract(vec2(uv.x, uv.y + 1.0*v) * 2.);
    uv7 = fract(vec2(uv.x, uv.y + 2.0*v) * 2.);
    uv8 = fract(vec2(uv.x, uv.y + 3.0*v) * 2.);
    uv9 = fract(vec2(uv.x, uv.y + 4.0*v) * 2.);

    if(uv.y > 0.5)
    {
        uv1 = upper_left(uv1);
        uv2 = upper_left(uv2);
        uv3 = upper_left(uv3);
        uv4 = upper_left(uv4);
        uv5 = upper_left(uv5);
        uv6 = upper_left(uv6);
        uv7 = upper_left(uv7);
        uv8 = upper_left(uv8);
        uv9 = upper_left(uv9);
    }
    else{
        uv1 = lower_left(uv1);
        uv2 = lower_left(uv2);
        uv3 = lower_left(uv3);
        uv4 = lower_left(uv4);
        uv5 = lower_left(uv5);
        uv6 = lower_left(uv6);
        uv7 = lower_left(uv7);
        uv8 = lower_left(uv8);
        uv9 = lower_left(uv9);
    }

    for(int level = 0; level < 8; level++)
    {
        if(level > depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);

    sum += texture(iChannel2, uv1) * 0.05;
    sum += texture(iChannel2, uv2) * 0.09;
    sum += texture(iChannel2, uv3) * 0.12;
    sum += texture(iChannel2, uv4) * 0.15;
    sum += texture(iChannel2, uv5) * 0.16;
    sum += texture(iChannel2, uv6) * 0.15;
    sum += texture(iChannel2, uv7) * 0.12;
    sum += texture(iChannel2, uv8) * 0.09;
    sum += texture(iChannel2, uv9) * 0.05;

    return sum/0.98; // normalize
}

// Function 65
vec4 blur_horizontal_left_column(vec2 uv, int depth)
{
    float h = pow(2., float(depth)) / iResolution.x;    
    vec2 uv1, uv2, uv3, uv4, uv5, uv6, uv7, uv8, uv9;

    uv1 = fract(vec2(uv.x - 4.0 * h, uv.y) * 2.);
    uv2 = fract(vec2(uv.x - 3.0 * h, uv.y) * 2.);
    uv3 = fract(vec2(uv.x - 2.0 * h, uv.y) * 2.);
    uv4 = fract(vec2(uv.x - 1.0 * h, uv.y) * 2.);
    uv5 = fract(vec2(uv.x + 0.0 * h, uv.y) * 2.);
    uv6 = fract(vec2(uv.x + 1.0 * h, uv.y) * 2.);
    uv7 = fract(vec2(uv.x + 2.0 * h, uv.y) * 2.);
    uv8 = fract(vec2(uv.x + 3.0 * h, uv.y) * 2.);
    uv9 = fract(vec2(uv.x + 4.0 * h, uv.y) * 2.);

    if(uv.y > 0.5)
    {
        uv1 = upper_left(uv1);
        uv2 = upper_left(uv2);
        uv3 = upper_left(uv3);
        uv4 = upper_left(uv4);
        uv5 = upper_left(uv5);
        uv6 = upper_left(uv6);
        uv7 = upper_left(uv7);
        uv8 = upper_left(uv8);
        uv9 = upper_left(uv9);
    }
    else{
        uv1 = lower_left(uv1);
        uv2 = lower_left(uv2);
        uv3 = lower_left(uv3);
        uv4 = lower_left(uv4);
        uv5 = lower_left(uv5);
        uv6 = lower_left(uv6);
        uv7 = lower_left(uv7);
        uv8 = lower_left(uv8);
        uv9 = lower_left(uv9);
    }

    for(int level = 0; level < 8; level++)
    {
        if(level >= depth)
        {
            break;
        }

        uv1 = lower_right(uv1);
        uv2 = lower_right(uv2);
        uv3 = lower_right(uv3);
        uv4 = lower_right(uv4);
        uv5 = lower_right(uv5);
        uv6 = lower_right(uv6);
        uv7 = lower_right(uv7);
        uv8 = lower_right(uv8);
        uv9 = lower_right(uv9);
    }

    vec4 sum = vec4(0.0);

    sum += texture(iChannel3, uv1) * 0.05;
    sum += texture(iChannel3, uv2) * 0.09;
    sum += texture(iChannel3, uv3) * 0.12;
    sum += texture(iChannel3, uv4) * 0.15;
    sum += texture(iChannel3, uv5) * 0.16;
    sum += texture(iChannel3, uv6) * 0.15;
    sum += texture(iChannel3, uv7) * 0.12;
    sum += texture(iChannel3, uv8) * 0.09;
    sum += texture(iChannel3, uv9) * 0.05;

    return sum/0.98; // normalize
}

// Function 66
float RightWall(vec3 p){
    return WallWithOffset(p, vec3(ROOM_WIDTH/2.0,0,0));
}

// Function 67
vec2 intersectionStereogram(vec2 uv, float time, float eyesDist, float faceScreenDist,float rightToLeft)
{
/*
//newX = xP - fxP*( eyesDist + xP )/(faceScreenDist + fxP);

//float fx = shapeDistance(uv,time);
//xP = uv.x + fx * ( uv.x - eyesDist )/faceScreenDist;
//fxP = fx;
//newX = xP - fxP*( eyesDist + xP )/(faceScreenDist + fxP);
//newX = -eyesDist + faceScreenDist / ( faceScreenDist + fxP ) * ( xP + eyesDist );
*/
    float xP = cancellation(uv,time,rightToLeft * eyesDist,faceScreenDist);
    float fxP = shapeDistance(vec2(xP,uv.y),time);
    //float newX = - rightToLeft * eyesDist + faceScreenDist * ( xP + fxP ) / ( faceScreenDist + fxP );
    float newX = - rightToLeft * eyesDist + faceScreenDist * ( xP + rightToLeft * eyesDist ) / ( faceScreenDist + fxP );

    return vec2(newX,uv.y);
}

