// Reusable Edge Detection Effect Functions
// Automatically extracted from effect-related shaders

// Function 1
vec3 draw_edge(vec2 uv, vec2 a, vec2 b, mat4 iT)
{
    
  	vec4 t1 = iT * vec4(vec3(a, map(a)).xzy, 1.0);
  	vec4 t2 = iT * vec4(vec3(b, map(b)).xzy, 1.0);
    
  	t1.xy /= t1.z;
  	t2.xy /= t2.z;
    
    vec3 col = color(vec2(0.4, map(a) * 0.0000002) );
    
    vec3 r = vec3(0.0);
    
   	r += segment(uv, t1.xy, t2.xy) * col;
    r += segment(uv, t2.xy, t1.xy) * col;
    
    return r;
}

// Function 2
float softEdge(float edge, float amt){
    return clamp(1.0 / (clamp(edge, 1.0/amt, 1.0)*amt), 0.,1.);
}

// Function 3
float calc_sobel_res(mat3 I) {
	float gx = dot(sx[0], I[0]) + dot(sx[1], I[1]) + dot(sx[2], I[2]); 
	float gy = dot(sy[0], I[0]) + dot(sy[1], I[1]) + dot(sy[2], I[2]);

	return sqrt(pow(gx, 2.0)+pow(gy, 2.0));
}

// Function 4
float DetermineEdgeBlendFactor (sampler2D  tex2D, vec2 texSize, LuminanceData l, EdgeData e, vec2 uv) {
	vec2 uvEdge = uv;
	vec2 edgeStep;
	if (e.isHorizontal) {
		uvEdge.y += e.pixelStep * 0.5f;
		edgeStep = vec2(texSize.x, 0.0f);
	}
	else {
		uvEdge.x += e.pixelStep * 0.5f;
		edgeStep = vec2(0.0f, texSize.y);
	}

	float edgeLuminance = (l.m + e.oppositeLuminance) * 0.5f;
	float gradientThreshold = e.gradient * 0.25f;

	vec2 puv = uvEdge + edgeStep * edgeSteps[0];
	float pLuminanceDelta = SampleLuminance(tex2D, puv) - edgeLuminance;
	bool pAtEnd = abs(pLuminanceDelta) >= gradientThreshold;

	for (int i = 1; i < EDGE_STEP_COUNT && !pAtEnd; i++) {
		puv += edgeStep * edgeSteps[i];
		pLuminanceDelta = SampleLuminance(tex2D, puv) - edgeLuminance;
		pAtEnd = abs(pLuminanceDelta) >= gradientThreshold;
	}
	if (!pAtEnd) {
		puv += edgeStep * EDGE_GUESS;
	}

	vec2 nuv = uvEdge - edgeStep * edgeSteps[0];
	float nLuminanceDelta = SampleLuminance(tex2D, nuv) - edgeLuminance;
	bool nAtEnd = abs(nLuminanceDelta) >= gradientThreshold;

	for (int i = 1; i < EDGE_STEP_COUNT && !nAtEnd; i++) {
		nuv -= edgeStep * edgeSteps[i];
		nLuminanceDelta = SampleLuminance(tex2D, nuv) - edgeLuminance;
		nAtEnd = abs(nLuminanceDelta) >= gradientThreshold;
	}
	if (!nAtEnd) {
		nuv -= edgeStep * EDGE_GUESS;
	}

	float pDistance, nDistance;
	if (e.isHorizontal) {
		pDistance = puv.x - uv.x;
		nDistance = uv.x - nuv.x;
	}
	else {
		pDistance = puv.y - uv.y;
		nDistance = uv.y - nuv.y;
	}

	float shortestDistance;
	bool deltaSign;
	if (pDistance <= nDistance) {
		shortestDistance = pDistance;
		deltaSign = pLuminanceDelta >= 0.0f;
	}
	else {
		shortestDistance = nDistance;
		deltaSign = nLuminanceDelta >= 0.0f;
	}

	if (deltaSign == (l.m - edgeLuminance >= 0.0f)) {
		return 0.0f;
	}
	return 0.5f - shortestDistance / (pDistance + nDistance);
}

// Function 5
vec3 wangEdgeSimple(in vec2 uv, vec4 edges)
{
    float x = uv.x;
    float y = uv.y;
    float invx = 1. - uv.x;
    float invy = 1. - uv.y;
    
    float result = 0.0;
    if (edges.r > 0.5) {
        result = max(result, float(x < 0.3));
    }
    if (edges.g > 0.5) {
        result = max(result, float(invy < 0.3));
    }
    if (edges.b > 0.5) {
        result = max(result, float(invx < 0.3));
    }
    if (edges.a > 0.5) {
        result = max(result, float(y < 0.3));
    }
    return vec3(result);
}

// Function 6
vec3 edgeStrength(in vec2 uv)
{
   	const float spread = 0.5;
 	vec2 offset = vec2(1.0) / iChannelResolution[0].xy;
    vec2 up    = vec2(0.0, offset.y) * spread;
    vec2 right = vec2(offset.x, 0.0) * spread;
    const float frad =  3.0;
    vec3 v11 = blurSample(uv + up - right, 	right, up);
    vec3 v12 = blurSample(uv + up, 			right, up);
    vec3 v13 = blurSample(uv + up + right, 	right, up);
    
    vec3 v21 = blurSample(uv - right, 		right, up);
    vec3 v22 = blurSample(uv, 				right, up);
    vec3 v23 = blurSample(uv + right, 		right, up);
    
    vec3 v31 = blurSample(uv - up - right, 	right, up);
    vec3 v32 = blurSample(uv - up, 			right, up);
    vec3 v33 = blurSample(uv - up + right, 	right, up);
    
    vec3 laplacian_of_g = v11 * 0.0 + v12 *  1.0 + v13 * 0.0
        				+ v21 * 1.0 + v22 * -4.0 + v23 * 1.0
        				+ v31 * 0.0 + v32 *  1.0 + v33 * 0.0;
   		 laplacian_of_g = laplacian_of_g * 1.0;
    return laplacian_of_g.xyz;
}

// Function 7
vec4 edgeSample(vec2 coord)
{
    float t = iTime*0.002;
    return vec4(sample1(coord + vec2(0., t)),  // left
                sample2(coord + vec2(t, 1.)),  // top
                sample1(coord + vec2(1., t)),  // right
                sample2(coord + vec2(t, 0.))); // bottom
}

// Function 8
float distanceToEdge(vec3 point)
{
    float dx = abs(point.x > 0.5 ? 1.0 - point.x : point.x);
    float dy = abs(point.y > 0.5 ? 1.0 - point.y : point.y);
    if (point.x < 0.0) dx = -point.x;
    if (point.x > 1.0) dx = point.x - 1.0;
    if (point.y < 0.0) dy = -point.y;
    if (point.y > 1.0) dy = point.y - 1.0;
    if ((point.x < 0.0 || point.x > 1.0) && (point.y < 0.0 || point.y > 1.0)) return sqrt(dx * dx + dy * dy);
    return min(dx, dy);
}

// Function 9
void gaussianEdge( out vec4 fragColor, in vec2 fragCoord, in highp float sigma)
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 original = texture(iChannel0, uv);
    vec4 blurred = vec4(0);
    gaussianBlur(blurred, fragCoord, sigma);
    fragColor = original - blurred;
}

// Function 10
void boxEdge( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    vec4 original = texture(iChannel0, uv);
    vec4 blurred = vec4(0);
    boxBlur(blurred, fragCoord);
    fragColor = original - blurred;
}

// Function 11
float threadedEdges(vec2 st, float width){
    return 1.0 - smoothstep(0., width, st.x) + smoothstep(1.-width, 1., st.x);
}

// Function 12
float VoronoiEdgeDist(vec3 p, float threshold, out vec3 color)
{
    vec3 local_p = fract(p); // range [0., +1.]
    
  	vec3 vector_to_closest;
    float min_dist = MAX_DIST;
    
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            for (int z = -1; z <= 1; z++)
            {
                vec3 offset = vec3(x, y, z);
                vec3 id = floor(offset + p);

                vec3 local_center = R33(id);
                vec3 center = id + local_center;

                vec3 point_to_center = center - p;
                float dist = length(point_to_center);

                if (dist < min_dist)
                {
                    min_dist = dist;
                    color = R33(id);
                    vector_to_closest = point_to_center;
                }
            }

    
    min_dist = MAX_DIST;
    
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            for (int z = -1; z <= 1; z++)
            {
                vec3 offset = vec3(x, y, z);
                vec3 id = floor(offset + p);
                
                vec3 local_center = R33(id);
                vec3 center = id + local_center;
                
                vec3 point_to_center = center - p;

                vec3 perpendicularToEdge = point_to_center - vector_to_closest;
                
                if (length(perpendicularToEdge) < 0.01)
                    continue;

                float distanceToEdge = dot(
                    (vector_to_closest + point_to_center) / 2.0,
                    normalize(perpendicularToEdge)
                );
                
                min_dist = min(min_dist, distanceToEdge);
            }

    return min_dist - threshold;
}

// Function 13
float FastEdge(vec2 uv) {
    vec3 e = vec3(1./R, 0.);
    vec4 Center_P = texture(_CameraDepthNormalsTexture,uv);
    vec4 LD = texture(_CameraDepthNormalsTexture, uv + e.xy);
    vec4 RD = texture(_CameraDepthNormalsTexture, uv + vec2(e.x,-e.y));

    float Edge = 0.;
    Edge += CheckDiff(Center_P,LD);
    Edge += CheckDiff(Center_P,RD);
    return float(smoothstep(1., 0., Edge));
}

// Function 14
vec4 sobel(vec2 texCoord) {
	vec2 invTexSize = 2.0 / iChannelResolution[0].xy;

    vec4 nxny = texture(iChannel0, texCoord + invTexSize * vec2(-1, -1));
    vec4 nxby = texture(iChannel0, texCoord + invTexSize * vec2(-1,  0));
    vec4 nxpy = texture(iChannel0, texCoord + invTexSize * vec2(-1,  1));

    vec4 bxny = texture(iChannel0, texCoord + invTexSize * vec2( 0, -1));
    vec4 bxby = texture(iChannel0, texCoord + invTexSize * vec2( 0,  0));
    vec4 bxpy = texture(iChannel0, texCoord + invTexSize * vec2( 0,  1));

    vec4 pxny = texture(iChannel0, texCoord + invTexSize * vec2( 1, -1));
    vec4 pxby = texture(iChannel0, texCoord + invTexSize * vec2( 1,  0));
    vec4 pxpy = texture(iChannel0, texCoord + invTexSize * vec2( 1,  1));

/*
    vec4 sobelx = -nxny - 2*nxby - nxpy + pxny + 2 * pxby + pxpy;
    vec4 sobely = -nxny - 2*bxny - pxny + nxpy + 2 * bxpy + pxpy;
    vec4 sobel = sqrt(sobelx*sobelx + sobely*sobely);
*/

    vec3 vScale = vec3(0.3333); // average color values:
    // vec3 vScale = vec3(0.2126, 0.7152, 0.0722); // 'luma' weights
    float fnxny = dot(nxny.rgb, vScale);
    float fnxby = dot(nxby.rgb, vScale);
    float fnxpy = dot(nxpy.rgb, vScale);
		
    float fbxny = dot(bxny.rgb, vScale);
    float fbxby = dot(bxby.rgb, vScale);
    float fbxpy = dot(bxpy.rgb, vScale);
		
	float fpxny = dot(pxny.rgb, vScale);
	float fpxby = dot(pxby.rgb, vScale);
	float fpxpy = dot(pxpy.rgb, vScale);

    vec2 fsobel;
    fsobel.x = -fnxny - 2.0*fnxby - fnxpy + fpxny + 2.0 * fpxby + fpxpy;
    fsobel.y = -fnxny - 2.0*fbxny - fpxny + fnxpy + 2.0 * fbxpy + fpxpy;
    float sobelf = sqrt(fsobel.x*fsobel.x + fsobel.y*fsobel.y);
    
    return vec4(sobelf);
}

// Function 15
vec2 getEdgeDistAndShading(vec4 uvsf) {
    
    float d = abs(geodesicDist(edges[2], uvsf.xy));
    d = min(d, abs(geodesicDist(edges[1], uvsf.xy)));
    d = min(d, abs(geodesicDist(edges[0], uvsf.xy)));
    
    return vec2(d, mix(uvsf.w < 0. ? 0.8 : 1.0, 0.9, smoothstep(0.5*uvsf.z, 0.0, d)));

}

// Function 16
vec4 sobelOperator(vec4 image)
{
    
    return vec4(1.0,1.0,1.0,1.0);
}

// Function 17
vec3 sobel(vec2 uv) {
    mat3 Y;
    mat3 Co;
    mat3 Cr;
    
    vec3 temp; 
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
        	vec2 pos = uv + vec2(float(i-1) * inv_res.x, float(j-1) * inv_res.y);
            temp = YCoCr(iChannel0, pos);
            Y[i][j] = temp.x;
            Co[i][j] = temp.y;
            Cr[i][j] = temp.z;
	    }
	}
    
	return vec3(calc_sobel_res(Y), calc_sobel_res(Co), calc_sobel_res(Cr));
}

// Function 18
vec3 edge(sampler2D sampler, vec2 uv, float sampleSize)
{
    float dx = sampleSize / iResolution.x;
    float dy = sampleSize / iResolution.y;
    return (
    mix(downsample(sampler, uv - vec2(dx, 0.0), sampleSize), downsample(sampler, uv + vec2(dx, 0.0), sampleSize), mod(uv.x, dx) / dx) +
    mix(downsample(sampler, uv - vec2(0.0, dy), sampleSize), downsample(sampler, uv + vec2(0.0, dy), sampleSize), mod(uv.y, dy) / dy)    
    ).rgb / 2.0 - texture(sampler, uv).rgb;
}

// Function 19
float FXAAVerticalEdge( float lumaO,
                       float lumaN, 
                       float lumaE, 
                       float lumaS, 
                       float lumaW,
                       float lumaNW,
                       float lumaNE,
                       float lumaSW,
                       float lumaSE ) {
    
    // Slices to calculate.
    float top = (0.25 * lumaNW) + (-0.5 * lumaN) + (0.25 * lumaNE);
    float middle = (0.50 * lumaW ) + (-1.0 * lumaO) + (0.50 * lumaE );
    float bottom = (0.25 * lumaSW) + (-0.5 * lumaS) + (0.25 * lumaSE);
    
    // Return value.
    return abs(top) + abs(middle) + abs(bottom);
}

// Function 20
vec4
sobel ( in sampler2D tex, in vec2 fragCoord )
{
    vec4 pnw = texture(tex, (fragCoord.xy - 1.0) / iResolution.xy);
    vec4 pn  = texture(tex, (fragCoord.xy - vec2(0.0, 1.0)) / iResolution.xy);
    vec4 pne = texture(tex, (fragCoord.xy - vec2(-1.0, 1.0)) / iResolution.xy);
    vec4 pw  = texture(tex, (fragCoord.xy - vec2(1.0, 0.0)) / iResolution.xy);
    vec4 pe  = texture(tex, (fragCoord.xy + vec2(1.0, 0.0)) / iResolution.xy);
    vec4 psw = texture(tex, (fragCoord.xy + vec2(-1.0, 1.0)) / iResolution.xy);
    vec4 ps  = texture(tex, (fragCoord.xy + vec2(0.0, 1.0)) / iResolution.xy);
    vec4 pse = texture(tex, (fragCoord.xy + 1.0) / iResolution.xy);


    vec4 gx = -pnw - 2.0*pw - psw + pne + 2.0*pe + pse;
    vec4 gy = -pnw - 2.0*pn - pne + psw + 2.0*ps + pse;

    return sqrt(gx*gx + gy*gy);
}

// Function 21
float DSCellEdge(vec3 pos)
{
	vec2 dPos = abs(PosRound(pos).xy - pos.xy);
	const float sZSlop = 10.0;
	return (pos.z > g_zMax) ? (pos.z - g_zMax - sZSlop) : min(dPos.x, dPos.y);
}

// Function 22
float calcEdge( vec3 pos )
{
    vec3 eps = vec3( 0.05, 0.0, 0.0 );
    float d000 = map( pos ).x;
    float d_100 = map( pos - eps.xyy ).x;
    float d100 = map( pos + eps.xyy ).x;
    float d0_10 = map( pos - eps.yxy ).x;
    float d010 = map( pos + eps.yxy ).x;
    float d00_1 = map( pos - eps.yyx ).x;
    float d001 = map( pos + eps.yyx ).x;
    float edge = abs( d000 - 0.5 * ( d_100 + d100 ) ) +
                 abs( d000 - 0.5 * ( d0_10 + d010 ) ) +
                 abs( d000 - 0.5 * ( d00_1 + d001 ) );

    return clamp( 1.0 - edge * 200.0, 0.0, 1.0 );
}

// Function 23
float getEdgeVal(vec3 p, float edgeDelta)
{ 
	
	// Edge spread of a few pixels, regardless of resolution.
	// Constant values could also be used.
	vec2 e = vec2(edgeDelta/iResolution.y, 0);

	// Nearby sample values.
	float d1 = map(p + e.xyy).x, d2 = map(p - e.xyy).x;
	float d3 = map(p + e.yxy).x, d4 = map(p - e.yxy).x;
	float d5 = map(p + e.yyx).x, d6 = map(p - e.yyx).x;
	float d = map(p).x*2.;

	// Edge value. One of a few ways to do it, depending on
	// the look you're after.
	float edge = abs(d1 + d2 - d) + abs(d3 + d4 - d) + abs(d5 + d6 - d);
	//edge = abs(d1 + d2 + d3 + d4 + d5 + d6 - d*3.);
	edge = smoothstep(0., 1., sqrt(edge/e.x*2.));
	
	return edge;
}

// Function 24
void intersect_plane_edge(int mc, vec4 plane, vec3 ro, vec3 rd, inout vec3 edge_near, inout vec3 edge_far, inout bvec2 has_edge) {
    float d = intersect_plane_line(plane, ro, rd);
    if (abs(d) < 1.0) {
        vec3 p = viewspace(ro + rd * d);
        float d;
        if (mc == 0) {
            d = dFrustumV(p);
        } else {
            d = dFrustumH(p);
        }
        if (d <= 0.0) {
            if (!has_edge[0] || (p.z < edge_near.z)) {
                edge_near = p;
                has_edge[0] = true;
            }
            if (!has_edge[1] || (p.z > edge_far.z)) {
                edge_far = p;
                has_edge[1] = true;
            }
        }
    }
}

// Function 25
bool isEdgeFragment(vec2 fragCoord) {
	float kernel[(int(kernelWidth * kernelHeight))];
	kernel[0] = -1.;
	kernel[1] = -1.;
	kernel[2] = -1.;
	kernel[3] = -1.;
	kernel[4] =  8.;
	kernel[5] = -1.;
	kernel[6] = -1.;
	kernel[7] = -1.;
	kernel[8] = -1.;
	
	vec4 result = vec4(0.);
	vec2 uv = getUV(fragCoord);
	
	for(float y = 0.; y < kernelHeight; ++y) {
		for(float x = 0.; x < kernelWidth; ++x) {
			result += texture(iChannel0, vec2(uv.x + (float(int(x - kernelWidth / 2.)) / iResolution.x), 
												uv.y + (float(int(y - kernelHeight / 2.)) / iResolution.y)))
										   * kernel[int(x + (y * kernelWidth))];
		}
	}
	
	return ((length(result) > 0.2) ? true : false);
}

// Function 26
vec3 decide3Edge(vec3 uv, int i, int j) {
    
    int k = 3 - i - j;
    vec3 g = geodesicFromPoints(generator, verts[k]);
    
    if (geodesicDist(g, uv.xy) * geodesicDist(g, verts[j]) >= 0.) {
        return decide2(uv, j, k);
    } else {
        return decide2(uv, i, k);
    }
    
}

// Function 27
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

// Function 28
vec4 sobelOperator(vec4 image)
{
    return vec4(0.,0.,0.,1.0);
}

// Function 29
vec3 wangEdgeDot(in vec2 uv, vec4 edges)
{
    float x = uv.x;
    float y = uv.y;
    float halfx = x-0.5;
    float halfy = y-0.5;
    float invx = 1. - uv.x;
    float invy = 1. - uv.y;
    
    float result = 0.0;
    if (edges.r > 0.7) {
        result = max(result, float(x*x + halfy*halfy < 0.25));
    }
    if (edges.g > 0.7) {
        result = max(result, float(halfx*halfx + invy*invy < 0.25));
    }
    if (edges.b > 0.7) {
        result = max(result, float(invx*invx + halfy*halfy < 0.25));
    }
    if (edges.a > 0.7) {
        result = max(result, float(halfx*halfx + y*y < 0.25));
    }
    return vec3(result);
}

// Function 30
void paint_frustum_edges(vec2 pt[4]) {
    for (int i = 0; i < 4; ++i) {
		move_to(0.0, 0.0);
		line_to(pt[i]);
    }
    move_to(pt[0]);
    for (int i = 1; i < 4; ++i) {
		line_to(pt[i]);
    }    
    close_path();
}

// Function 31
vec3 sobel(float stepx, float stepy, vec2 center){
	// get samples around pixel
    float tleft = intensity(texture(iChannel0,center + vec2(-stepx,stepy)));
    float left = intensity(texture(iChannel0,center + vec2(-stepx,0)));
    float bleft = intensity(texture(iChannel0,center + vec2(-stepx,-stepy)));
    float top = intensity(texture(iChannel0,center + vec2(0,stepy)));
    float bottom = intensity(texture(iChannel0,center + vec2(0,-stepy)));
    float tright = intensity(texture(iChannel0,center + vec2(stepx,stepy)));
    float right = intensity(texture(iChannel0,center + vec2(stepx,0)));
    float bright = intensity(texture(iChannel0,center + vec2(stepx,-stepy)));
 
	// Sobel masks (see http://en.wikipedia.org/wiki/Sobel_operator)
	//        1 0 -1     -1 -2 -1
	//    X = 2 0 -2  Y = 0  0  0
	//        1 0 -1      1  2  1
	
	// You could also use Scharr operator:
	//        3 0 -3        3 10   3
	//    X = 10 0 -10  Y = 0  0   0
	//        3 0 -3        -3 -10 -3
 
    float x = tleft + 2.0*left + bleft - tright - 2.0*right - bright;
    float y = -tleft - 2.0*top - tright + bleft + 2.0 * bottom + bright;
    float color = sqrt((x*x) + (y*y));
    return vec3(color,color,color);
 }

// Function 32
float find_edge_distance(vec3 p, int brush, int side)
{
    float dist = -1e8;
    
    if (brush < NUM_MAP_AXIAL_BRUSHES)
    {
        vec3[2] deltas;
        deltas[0] = get_axial_point(brush*2) - p;
        deltas[1] = p - get_axial_point(brush*2+1);
        int axis = side >> 1;
        int front = side & 1;
        for (int i=0; i<6; ++i)
            if (i != side)
            	dist = max(dist, deltas[1&~i][i>>1]);
    }
    else
    {
        int begin = get_nonaxial_brush_start(brush - NUM_MAP_AXIAL_BRUSHES);
        int end = get_nonaxial_brush_start(brush - (NUM_MAP_AXIAL_BRUSHES - 1));
        for (int i=begin; i<end; ++i)
        {
            if (i == begin + side)
                continue;
            vec4 plane = get_nonaxial_plane(i);
            dist = max(dist, dot(p, plane.xyz) + plane.w);
        }
    }
    
    return dist;
}

// Function 33
vec4 edge(float stepx, float stepy, vec2 center, mat3 kernelX, mat3 kernelY){
	// get samples around pixel
	mat3 image = mat3(length(texture(iChannel0,center + vec2(-stepx,stepy)).rgb),
					  length(texture(iChannel0,center + vec2(0,stepy)).rgb),
					  length(texture(iChannel0,center + vec2(stepx,stepy)).rgb),
					  length(texture(iChannel0,center + vec2(-stepx,0)).rgb),
					  length(texture(iChannel0,center).rgb),
					  length(texture(iChannel0,center + vec2(stepx,0)).rgb),
					  length(texture(iChannel0,center + vec2(-stepx,-stepy)).rgb),
					  length(texture(iChannel0,center + vec2(0,-stepy)).rgb),
					  length(texture(iChannel0,center + vec2(stepx,-stepy)).rgb));
 	vec2 result;
	result.x = convolve(kernelX, image);
	result.y = convolve(kernelY, image);
	
    float color = clamp(length(result), 0.0, 255.0);
    return vec4(color);
}

// Function 34
float edgeIntensity(vec2 uv)
{
	float edgeIntensityX = 1.0;
    if( uv.x < 0.1)
    {
    	edgeIntensityX = 0.7 + 0.3*(uv.x/0.1);
    }
    else if( uv.x > 0.90)   
    {
    	edgeIntensityX = 0.7 + 0.3*((1.0-uv.x)/0.1);
    }
        
    float edgeIntensityY = 1.0;
    if( uv.y < 0.15)
    {
    	edgeIntensityY = 0.6 + 0.4*(uv.y/0.15);
    }
    else if( uv.y > 0.85)   
    {
    	edgeIntensityY = 0.6 + 0.4*((1.0-uv.y)/0.15);
    }        
    return edgeIntensityX*edgeIntensityY;
}

// Function 35
vec4 detectEdgesSimple(vec2 uv) {
    // Simple central diff detector
    vec4 offset = vec4(1./iResolution.xy, -1./iResolution.xy);
    vec4 hill = texture(iChannel0, uv);
    
    vec4 acc = (hill - texture(iChannel0, uv + offset.x)) / offset.x;
    acc += (hill - texture(iChannel0, uv - offset.x)) / offset.x;
    acc += (hill - texture(iChannel0, uv + offset.y)) / offset.y;
    acc += (hill - texture(iChannel0, uv - offset.y)) / offset.y;
    acc += (hill - texture(iChannel0, uv + offset.xy)) / (.5 * (offset.x + offset.y));
    acc += (hill - texture(iChannel0, uv - offset.xy)) / (.5 * (offset.x + offset.y));
    acc += (hill - texture(iChannel0, uv + offset.zy)) / (.5 * (offset.x + offset.y));
	acc += (hill - texture(iChannel0, uv - offset.xw)) / (.5 * (offset.x + offset.y));

	return abs(acc * .003); // Changing the multiplier we can control the number o edges
}

// Function 36
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

// Function 37
vec4 trueColorEdge(float stepx, float stepy, vec2 center, mat3 kernelX, mat3 kernelY) {
	vec4 edgeVal = edge(stepx, stepy, center, kernelX, kernelY);
	return edgeVal * texture(iChannel0,center);
}

// Function 38
EdgeData DetermineEdge (vec2 texSize, LuminanceData l) {
	EdgeData e;
	float horizontal =
		abs(l.n + l.s - 2.0f * l.m) * 2.0f +
		abs(l.ne + l.se - 2.0f * l.e) +
		abs(l.nw + l.sw - 2.0f * l.w);
	float vertical =
		abs(l.e + l.w - 2.0f * l.m) * 2.0f +
		abs(l.ne + l.nw - 2.0f * l.n) +
		abs(l.se + l.sw - 2.0f * l.s);
	e.isHorizontal = horizontal >= vertical;

	float pLuminance = e.isHorizontal ? l.n : l.e;
	float nLuminance = e.isHorizontal ? l.s : l.w;
	float pGradient = abs(pLuminance - l.m);
	float nGradient = abs(nLuminance - l.m);

	e.pixelStep =
		e.isHorizontal ? texSize.y : texSize.x;
	
	if (pGradient < nGradient) {
		e.pixelStep = -e.pixelStep;
		e.oppositeLuminance = nLuminance;
		e.gradient = nGradient;
	}
	else {
		e.oppositeLuminance = pLuminance;
		e.gradient = pGradient;
	}

	return e;
}

// Function 39
float FXAAHorizontalEdge( float lumaO,
                       float lumaN, 
                       float lumaE, 
                       float lumaS, 
                       float lumaW,
                       float lumaNW,
                       float lumaNE,
                       float lumaSW,
                       float lumaSE ) {
    
    // Slices to calculate.
    float top = (0.25 * lumaNW) + (-0.5 * lumaW) + (0.25 * lumaSW);
    float middle = (0.50 * lumaN ) + (-1.0 * lumaO) + (0.50 * lumaS );
    float bottom = (0.25 * lumaNE) + (-0.5 * lumaE) + (0.25 * lumaSE);
    
    // Return value.
    return abs(top) + abs(middle) + abs(bottom);
}

// Function 40
float detectEdgesSobel(vec2 uv) {
    // Edge detection based on Sobel kernel
    vec4 offset = vec4(1./iResolution.xy, -1./iResolution.xy);
    
    float gx = 0.0;
    float gy = 0.0;
    
    vec4 clr = texture(iChannel0, uv - offset.xy);
    gx += -1. * dot(clr, clr);
    gy += -1. * dot(clr, clr);
    
    clr = texture(iChannel0, uv - offset.x);
    gx += -2. * dot(clr, clr);
    
    clr = texture(iChannel0, uv + offset.zy);
    gx += -1. * dot(clr, clr);
    gy +=  1. * dot(clr, clr);
    
    clr = texture(iChannel0, uv + offset.xw);
    gx +=  1. * dot(clr, clr);
    gy += -1. * dot(clr, clr);
    
    clr = texture(iChannel0, uv + offset.x);
    gx += 2. * dot(clr, clr);
    
    clr = texture(iChannel0, uv + offset.xy);
    gx += 1. * dot(clr, clr);
    gy += 1. * dot(clr, clr);
    
    clr = texture(iChannel0, uv - offset.y);
    gy += -2. * dot(clr, clr);
    
    clr = texture(iChannel0, uv + offset.y);
    gy += 2. * dot(clr, clr);
    
	return gx*gx + gy*gy;
}

